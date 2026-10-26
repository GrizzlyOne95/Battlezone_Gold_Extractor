"""Run with Blender --background --python this_script -- --dump-root PATH.

Build a self-contained material inspection scene without altering native geometry.
Native default colors and emission are decoded; game BRDF and entity overrides
remain separately documented.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
from urllib.parse import unquote

import bpy
from mathutils import Matrix, Quaternion, Vector
from mathutils.kdtree import KDTree


def source_world_matrices(document):
    parents={child:i for i,n in enumerate(document['nodes']) for child in n.get('children',[])}
    worlds={}
    def world(index):
        if index in worlds:return worlds[index]
        n=document['nodes'][index]
        if 'matrix' in n:
            values=n['matrix'];local=Matrix([values[i::4] for i in range(4)])
        else:
            q=n.get('rotation',[0,0,0,1]);local=Matrix.LocRotScale(Vector(n.get('translation',[0,0,0])),Quaternion((q[3],*q[:3])),Vector(n.get('scale',[1,1,1])))
        worlds[index]=(world(parents[index]) @ local) if index in parents else local
        return worlds[index]
    for index in range(len(document['nodes'])):world(index)
    return worlds


def restore_native_socket_transforms(document):
    """Correct scaled bone-tail parenting after Blender's glTF import.

    Blender creates unit-length bones for identity inverse bind matrices. Its
    bone-tail offset on socket empties is wrong under the 32x armature scale.
    Retain the actual parents and use native glTF world matrices to compensate.
    """
    worlds=source_world_matrices(document);basis=Matrix.Rotation(math.pi/2,4,'X')
    objects={o.get('_asura_node_index'):o for o in bpy.data.objects if '_asura_node_index' in o}
    bones={b.bone.get('_asura_node_index'):(o,b) for o in bpy.data.objects if o.type=='ARMATURE' for b in o.pose.bones if '_asura_node_index' in b.bone}
    fixes=[]
    for index,n in enumerate(document['nodes']):
        bpy.context.view_layer.update()
        # Blender also rotates mesh vertex coordinates into its Z-up basis.
        wanted=basis @ worlds[index] @ basis.inverted()
        if index in bones:
            # Importing an action can leave its first pose after it is muted.
            # Restore source node transforms while keeping the imported rest rig.
            armature,bone=bones[index]
            bone.matrix=armature.matrix_world.inverted() @ wanted @ bone.bone.matrix_local
            bpy.context.view_layer.update()
            continue
        if 'mesh' in n:continue
        obj=objects.get(index)
        if obj is None:raise ValueError(f'Missing native object {index}: {n["name"]}')
        error=(obj.matrix_world.translation-wanted.translation).length
        obj.matrix_world=wanted
        bpy.context.view_layer.update()
        fixes.append(dict(node=index,name=n['name'],translation_error_before=error,translation_error_after=(obj.matrix_world.translation-wanted.translation).length))
    return fixes


def validate_assembly(document,source):
    """Compare every assembled, evaluated vertex to independent glTF skin math."""
    import numpy as np
    blob=(source.parent/unquote(document['buffers'][0]['uri'])).read_bytes()
    worlds=source_world_matrices(document);basis=Matrix.Rotation(math.pi/2,4,'X')
    objects={o.get('_asura_node_index'):o for o in bpy.data.objects if '_asura_node_index' in o}
    def accessor(index):
        a=document['accessors'][index];v=document['bufferViews'][a['bufferView']]
        width={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
        dtype={5126:'<f4',5123:'<u2',5125:'<u4',5121:'u1'}[a['componentType']]
        return np.frombuffer(blob,dtype=dtype,count=a['count']*width,offset=v.get('byteOffset',0)+a.get('byteOffset',0)).reshape(a['count'],width)
    results=[];graph=bpy.context.evaluated_depsgraph_get()
    for index,n in enumerate(document['nodes']):
        if 'mesh' not in n:continue
        obj=objects[index];primitives=document['meshes'][n['mesh']]['primitives'];attributes=primitives[0]['attributes']
        positions=accessor(attributes['POSITION']);joints=accessor(attributes['JOINTS_0']);weights=accessor(attributes['WEIGHTS_0'])
        skin=document['skins'][n['skin']];points=np.zeros((len(positions),3));homogeneous=np.c_[positions,np.ones(len(positions))]
        for slot,joint in enumerate(skin['joints']):
            w=np.where(joints==slot,weights,0).sum(1)
            if w.any():points+=(homogeneous @ np.array(basis @ worlds[joint],dtype=float).T)[:,:3]*w[:,None]
        used=np.unique(np.concatenate([accessor(p['indices']).reshape(-1) for p in primitives]));points=points[used]
        evaluated=obj.evaluated_get(graph);mesh=evaluated.to_mesh();actual=[evaluated.matrix_world @ v.co for v in mesh.vertices]
        expected_tree=KDTree(len(points));actual_tree=KDTree(len(actual))
        for i,p in enumerate(points):expected_tree.insert(Vector(p),i)
        for i,p in enumerate(actual):actual_tree.insert(p,i)
        expected_tree.balance();actual_tree.balance()
        error=max(max(expected_tree.find(p)[2] for p in actual),max(actual_tree.find(Vector(p))[2] for p in points))
        results.append(dict(name=obj.name,native_used_vertices=len(points),blender_vertices=len(actual),max_world_position_error=error))
        evaluated.to_mesh_clear()
        tolerance=max(3e-6,float(np.abs(points).max())*1e-6)
        results[-1]['position_tolerance']=tolerance
        if error>tolerance:raise ValueError(f'Assembly geometry mismatch: {obj.name}: {error} > {tolerance}')
    return results


def validate_native_motion(document,source,clip_names=None):
    """Exercise native actions in the assembled, reloaded hierarchy."""
    import numpy as np
    blob=(source.parent/unquote(document['buffers'][0]['uri'])).read_bytes()
    def values(index):
        a=document['accessors'][index];v=document['bufferViews'][a['bufferView']]
        width={'SCALAR':1,'VEC3':3,'VEC4':4}[a['type']]
        return np.frombuffer(blob,dtype='<f4',offset=v.get('byteOffset',0)+a.get('byteOffset',0),count=a['count']*width).reshape(a['count'],width)
    results=[]
    for name in (clip_names or ('Throttle_Left_Fwd','Monitor_Moving_Left_Transition_NoVR','Cables_Mid_Move_Loop')):
        clips=[a for a in document['animations'] if a['name']==name]
        if not clips:raise ValueError('Missing native animation: '+name)
        restore_native_socket_transforms(document)
        assignments=[]
        for obj in bpy.data.objects:
            ad=obj.animation_data
            if not ad:continue
            for track in ad.nla_tracks:
                if track.name==name or (track.name.startswith(name+'.') and track.name[len(name)+1:].isdigit()):
                    strip=track.strips[0];ad.action=strip.action
                    if getattr(strip,'action_slot',None):ad.action_slot=strip.action_slot
                    assignments.append(obj)
        if not assignments:raise ValueError('Missing imported action: '+name)
        print('ASSEMBLED_MOTION_CHECK',name,[o.name for o in assignments],flush=True)
        duration=max(float(values(s['input']).max()) for clip in clips for s in clip['samplers'])
        for time in sorted({0.,duration/2,duration}):
            posed=copy.deepcopy(document)
            for clip in clips:
                for channel in clip['channels']:
                    sampler=clip['samplers'][channel['sampler']];times=values(sampler['input']).reshape(-1);keys=values(sampler['output'])
                    i=min(max(int(np.searchsorted(times,time,side='right'))-1,0),len(times)-1);value=keys[i].tolist();kind=channel['target']['path']
                    if i+1<len(times) and time>times[i]:
                        factor=float((time-times[i])/(times[i+1]-times[i]))
                        if kind=='rotation':
                            a,b=keys[i],keys[i+1];q=Quaternion((a[3],*a[:3])).slerp(Quaternion((b[3],*b[:3])),factor);value=[q.x,q.y,q.z,q.w]
                        else:value=(keys[i]*(1-factor)+keys[i+1]*factor).tolist()
                    posed['nodes'][channel['target']['node']][kind]=value
            frame=time*bpy.context.scene.render.fps;bpy.context.scene.frame_set(math.floor(frame),subframe=frame-math.floor(frame));bpy.context.view_layer.update()
            meshes=validate_assembly(posed,source)
            results.append(dict(clip=name,time=time,meshes_checked=len(meshes),max_world_position_error=max(m['max_world_position_error'] for m in meshes)))
        for obj in assignments:obj.animation_data.action=None
        bpy.context.scene.frame_set(0)
    restore_native_socket_transforms(document)
    return results


def args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump-root',type=Path,required=True)
    parser.add_argument('--render',action='store_true')
    return parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])


def node(nodes,kind,name,location):
    result=nodes.new(kind);result.name=name;result.label=name;result.location=location
    return result


def normal_group():
    group=bpy.data.node_groups.new('Asura packed normal RG → Blender tangent XYZ','ShaderNodeTree')
    group.interface.new_socket(name='Packed RG',in_out='INPUT',socket_type='NodeSocketColor')
    group.interface.new_socket(name='Normal RGB',in_out='OUTPUT',socket_type='NodeSocketColor')
    nodes,links=group.nodes,group.links
    inp=node(nodes,'NodeGroupInput','Source normal texture',(-800,100));out=node(nodes,'NodeGroupOutput','Reconstructed normal',(650,100))
    separate=node(nodes,'ShaderNodeSeparateColor','Native XY',(-600,100));links.new(inp.outputs[0],separate.inputs[0])
    def operation(name,op,x,y,loc):
        n=node(nodes,'ShaderNodeMath',name,loc);n.operation=op
        if isinstance(x,(int,float)):n.inputs[0].default_value=x
        else:links.new(x,n.inputs[0])
        if isinstance(y,(int,float)):n.inputs[1].default_value=y
        else:links.new(y,n.inputs[1])
        return n.outputs[0]
    x=operation('X = 2R − 1','MULTIPLY_ADD',0,0,(-400,250))
    # MULTIPLY_ADD has three inputs; explicitly establish its affine decode.
    xnode=x.node;links.new(separate.outputs[0],xnode.inputs[0]);xnode.inputs[1].default_value=2;xnode.inputs[2].default_value=-1
    ynode=node(nodes,'ShaderNodeMath','Y = 2G − 1',(-400,50));ynode.operation='MULTIPLY_ADD';links.new(separate.outputs[1],ynode.inputs[0]);ynode.inputs[1].default_value=2;ynode.inputs[2].default_value=-1;y=ynode.outputs[0]
    xx=operation('X²','MULTIPLY',x,x,(-200,250));yy=operation('Y²','MULTIPLY',y,y,(-200,50));total=operation('XY length²','ADD',xx,yy,(0,220));remaining=operation('1 − XY²','SUBTRACT',1,total,(150,220));positive=operation('Clamp radicand','MAXIMUM',remaining,0,(300,220))
    sqrt=node(nodes,'ShaderNodeMath','Z = sqrt(max(0,1-X²-Y²))',(450,220));sqrt.operation='SQRT';links.new(positive,sqrt.inputs[0])
    blue=node(nodes,'ShaderNodeMath','Encode Z',(450,-20));blue.operation='MULTIPLY_ADD';links.new(sqrt.outputs[0],blue.inputs[0]);blue.inputs[1].default_value=.5;blue.inputs[2].default_value=.5
    green=operation('DirectX → Blender green','SUBTRACT',1,separate.outputs[1],(100,-150))
    combine=node(nodes,'ShaderNodeCombineColor','Tangent normal RGB',(650,-100));links.new(separate.outputs[0],combine.inputs[0]);links.new(green,combine.inputs[1]);links.new(blue.outputs[0],combine.inputs[2]);out.location=(900,0);links.new(combine.outputs[0],out.inputs[0])
    group['evidence']='DXBC: sample normal RG; XY*2-1; sqrt(max(0,1-dot(XY,XY))); negative binormal contribution.'
    return group


def wire_materials(document,base):
    from asura_materials import surface_parameters
    specs={m['name']:m for m in document['materials']}
    identified={(m['extras'].get('native_namespace'),m['extras']['native_material_hash']):m for m in document['materials']}
    resources=json.loads((base/'resources/materials/index.json').read_text())
    records={(r['namespace'],r['hash']):r for r in resources['records']}
    group=normal_group();report=[]
    def image_path(material,role):
        record=material.get('native_material',{});native=record.get('texture_hashes',{}).get(role)
        if not native:return None
        namespace=material.get('native_namespace') or ('chars__playertanks_pc' if int(material['native_material_hash'],16) in {0x308fb25d,0x308fb25e,0x308fb25f,0x308fb260,0x176d402c,0x6526f3c6,0xe986dad0,0xa01bc89e,0x9e98c691} else 'chars__actors_pc')
        indexed=records.get((namespace,int(material['native_material_hash'],16)))
        matches=indexed.get('exact_texture_hash_matches',[]) if indexed else []
        entry=next((x for x in matches if int(x['hash'],16)==native),None)
        if not entry:return None
        possibilities=entry['sources'];relative=next((s for s in possibilities if s.startswith(namespace+'/')),possibilities[0])
        p=Path(relative);stage=p if p.suffix.lower()=='.dds' else Path(str(p)+'.dds')
        candidate=base/'textures_png'/(str(stage)+'.png')
        return candidate if candidate.exists() else None
    def image_node(nodes,links,uv,path,name,position,data=False):
        n=node(nodes,'ShaderNodeTexImage',name,position);im=bpy.data.images.load(str(path),check_existing=True);n.image=im
        im.alpha_mode='CHANNEL_PACKED'
        im.colorspace_settings.name='Non-Color' if data else 'sRGB'
        n.interpolation='Linear';n.extension='REPEAT';links.new(uv.outputs['UV'],n.inputs['Vector'])
        return n
    for mat in bpy.data.materials:
        spec=identified.get((mat.get('native_namespace'),mat.get('native_material_hash'))) or specs.get(mat.name)
        if not spec:continue
        extras=spec['extras'];native=extras.get('native_material',{});key=int(extras['native_material_hash'],16)
        if not native.get('raw_words'):
            mat['native_material_unresolved']=True
            report.append(dict(material=mat.name,native_hash=extras['native_material_hash'],textures={},unresolved=True,notes=['Native material record absent; original glTF diagnostic material retained.']))
            continue
        parameters=native.get('surface_parameters') or surface_parameters(native['raw_words'])
        mat.use_nodes=True;nodes=mat.node_tree.nodes;links=mat.node_tree.links;nodes.clear()
        out=node(nodes,'ShaderNodeOutputMaterial','Surface output',(1000,100));bsdf=node(nodes,'ShaderNodeBsdfPrincipled','Asura surface approximation',(700,100));links.new(bsdf.outputs[0],out.inputs['Surface'])
        uv=node(nodes,'ShaderNodeUVMap','Native UV0',(-1300,200));uv.uv_map='UVMap'
        paths={role:image_path(extras,role) for role in ('base_color','normal','specular','opacity','emissive','reflection')}
        inputs={};notes=[]
        tint=node(nodes,'ShaderNodeRGB','Native albedo modifier — MARE +0x150',(-1050,800));tint.outputs[0].default_value=(*parameters['albedo_modifier'],1)
        tinted=None
        if paths['base_color']:
            tex=image_node(nodes,links,uv,paths['base_color'],'Native diffuse RGB / roughness A',(-1050,500));inputs['base_color']=tex
            multiply=node(nodes,'ShaderNodeMixRGB','Albedo RGB × native modifier',(-650,500));multiply.blend_type='MULTIPLY';multiply.inputs[0].default_value=1;links.new(tex.outputs['Color'],multiply.inputs[1]);links.new(tint.outputs[0],multiply.inputs[2]);tinted=multiply.outputs[0];links.new(tinted,bsdf.inputs['Base Color'])
            rough=node(nodes,'ShaderNodeMath','PBR roughness = max(albedo alpha, 0.05)',(-300,350));rough.operation='MAXIMUM';rough.inputs[1].default_value=.05;links.new(tex.outputs['Alpha'],rough.inputs[0]);links.new(rough.outputs[0],bsdf.inputs['Roughness'])
        else:bsdf.inputs['Base Color'].default_value=(.2,.2,.2,1);notes.append('Missing base texture')
        is_glass=any('windscreen' in str(p).lower() for p in paths.values() if p)
        is_gui=bool(extras.get('runtime_render_target'))
        bsdf.inputs['Metallic'].default_value=parameters['metallic']
        bsdf.inputs['IOR'].default_value=1.45 if is_glass else 1.5
        mat['preview_metallic']=bsdf.inputs['Metallic'].default_value
        mat['metallic_status']='Native shader selector: model 3 uses metal F0/albedo and suppresses diffuse; other models use dielectric F0=0.04.'
        mat['native_surface_parameters']=json.dumps(parameters)
        if paths['normal']:
            tex=image_node(nodes,links,uv,paths['normal'],'Native packed normal RG',(-1050,100),True)
            decode=node(nodes,'ShaderNodeGroup','Decode native RG normal',(-650,100));decode.node_tree=group;links.new(tex.outputs['Color'],decode.inputs[0])
            normal=node(nodes,'ShaderNodeNormalMap','Tangent normal',(-350,100));normal.uv_map='UVMap';links.new(decode.outputs[0],normal.inputs['Color']);links.new(normal.outputs['Normal'],bsdf.inputs['Normal']);inputs['normal']=tex
        # Resolve the extra native height reference by exact serialized hash match.
        record=records.get((extras.get('native_namespace'),key)) or next((r for (ns,k),r in records.items() if k==key),None)
        height=None
        if record:
            height=next((m for m in record['exact_texture_hash_matches'] if m['byte_offset']==224),None)
        if height:
            relative=Path(next((s for s in height['sources'] if s.startswith(extras.get('native_namespace','')+'/')),height['sources'][0]));hpath=base/'textures_png'/(str(relative)+('.png' if relative.suffix.lower()=='.dds' else '.dds.png'))
            if hpath.exists():
                height_tex=image_node(nodes,links,uv,hpath,'Native parallax height — bump approximation',(-1050,-350),True)
                bump=node(nodes,'ShaderNodeBump','Height detail (inspection only)',(350,-100));bump.inputs['Strength'].default_value=.2;bump.inputs['Distance'].default_value=.00025;links.new(height_tex.outputs['Color'],bump.inputs['Height'])
                if bsdf.inputs['Normal'].is_linked:links.new(bsdf.inputs['Normal'].links[0].from_socket,bump.inputs['Normal'])
                links.new(bump.outputs['Normal'],bsdf.inputs['Normal']);paths['height']=hpath;notes.append('Parallax height shown as small bump; source map retained')
        emission=node(nodes,'ShaderNodeMixRGB','Tinted albedo × native emission color',(-250,-600));emission.blend_type='MULTIPLY';emission.inputs[0].default_value=1;emission.inputs[2].default_value=(*parameters['self_illumination_colour'],1)
        if tinted:links.new(tinted,emission.inputs[1])
        else:links.new(tint.outputs[0],emission.inputs[1])
        emitted=emission.outputs[0]
        if paths['emissive']:
            tex=image_node(nodes,links,uv,paths['emissive'],'Native self-illumination mask — R',(-1050,-750),True);inputs['emissive']=tex
            separate=node(nodes,'ShaderNodeSeparateColor','Emission samples the native red channel',(-650,-750));links.new(tex.outputs['Color'],separate.inputs[0])
            mask=node(nodes,'ShaderNodeMixRGB','Native emission red mask', (0,-600));mask.blend_type='MULTIPLY';mask.inputs[0].default_value=1;links.new(emitted,mask.inputs[1]);links.new(separate.outputs[0],mask.inputs[2]);emitted=mask.outputs[0]
        links.new(emitted,bsdf.inputs['Emission Color']);bsdf.inputs['Emission Strength'].default_value=parameters['self_illumination_scale']*(1-parameters['metallic'])
        gain=node(nodes,'ShaderNodeValue','Native self-illumination scale — MARE +0x48',(350,-600));gain.outputs[0].default_value=bsdf.inputs['Emission Strength'].default_value;links.new(gain.outputs[0],bsdf.inputs['Emission Strength'])
        if paths['specular']:
            tex=image_node(nodes,links,uv,paths['specular'],'Native specular map',(-1050,-1100),True);links.new(tex.outputs['Color'],bsdf.inputs['Specular IOR Level'])
        if is_glass:
            bsdf.inputs['Transmission Weight'].default_value=.92;bsdf.inputs['Metallic'].default_value=0;bsdf.inputs['Roughness'].default_value=.07
            if bsdf.inputs['Roughness'].is_linked:links.remove(bsdf.inputs['Roughness'].links[0])
            # Thin glass reflection and transparent background keep the cabin inspectable.
            transparent=node(nodes,'ShaderNodeBsdfTransparent','Thin glass background',(700,-250));mix=node(nodes,'ShaderNodeMixShader','Native alpha / thin glass visibility',(1030,100));links.new(transparent.outputs[0],mix.inputs[1]);links.new(bsdf.outputs[0],mix.inputs[2]);links.new(mix.outputs[0],out.inputs['Surface']);out.location=(1300,100)
            mix.inputs[0].default_value=.16
            if paths['opacity']:
                alpha=image_node(nodes,links,uv,paths['opacity'],'Native dedicated alpha',(-1050,-1100),True);links.new(alpha.outputs['Color'],mix.inputs[0])
            notes.append('Thin-glass PBR preview; native alpha retained, game refraction/OIT differs')
        if is_gui:
            bsdf.inputs['Metallic'].default_value=0;bsdf.inputs['Roughness'].default_value=.45
            if bsdf.inputs['Roughness'].is_linked:links.remove(bsdf.inputs['Roughness'].links[0])
            if paths['opacity']:
                mask=image_node(nodes,links,uv,paths['opacity'],'Native GUI opacity mask',(-1050,-1100),True)
                transparent=node(nodes,'ShaderNodeBsdfTransparent','GUI transparent background',(700,-250))
                mix=node(nodes,'ShaderNodeMixShader','GUI mask over scene',(1030,100))
                links.new(mask.outputs['Color'],mix.inputs[0]);links.new(transparent.outputs[0],mix.inputs[1]);links.new(bsdf.outputs[0],mix.inputs[2]);links.new(mix.outputs[0],out.inputs['Surface']);out.location=(1300,100)
            notes.append('Native live GUI render target is blank in static assets; no replacement UI fabricated')
        mat['asura_preview_source']=json.dumps({k:str(v) for k,v in paths.items() if v})
        mat['asura_preview_notes']=json.dumps(notes)
        report.append(dict(material=mat.name,native_hash=extras['native_material_hash'],native_surface_parameters=parameters,textures={k:str(v.relative_to(base)) for k,v in paths.items() if v},glass=is_glass,runtime_gui=is_gui,notes=notes))
    return report


def main():
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from asura_materials import surface_parameters
    opts=args();base=opts.dump_root.resolve();source=base/'scenes/standard_player_tank.gltf';destination=base/'scenes/standard_player_tank_materials.blend'
    document=json.loads(source.read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene;scene.render.fps=30
    # Stable identifiers survive Blender's unique-name suffixes and bone conversion.
    for index,n in enumerate(document['nodes']):n.setdefault('extras',{})['_asura_node_index']=index
    temporary=source.with_name('_player_tank_blender_import.gltf')
    temporary.write_text(json.dumps(document))
    try:bpy.ops.import_scene.gltf(filepath=str(temporary))
    finally:temporary.unlink(missing_ok=True)
    for obj in bpy.data.objects:
        if obj.animation_data:
            obj.animation_data.action=None
            for track in obj.animation_data.nla_tracks:track.mute=True
    scene.frame_set(0);bpy.context.view_layer.update()
    socket_fixes=restore_native_socket_transforms(document)
    assembly_validation=validate_assembly(document,source)
    report=wire_materials(document,base)
    # All imported native geometry remains visible and independently selectable.
    for obj in bpy.data.objects:
        if obj.type=='MESH' and obj.name.endswith('_render'):
            obj.hide_render=False;obj.hide_viewport=False
            obj['source_geometry_preserved']=True
    collections={}
    for name in ('01 Exterior','02 Cockpit and controls','03 Screens — native static targets','04 Rig and attachment markers','05 Cameras and lighting'):
        collection=bpy.data.collections.new(name);scene.collection.children.link(collection);collections[name]=collection
    for obj in list(bpy.data.objects):
        if obj.type=='MESH' and obj.name.endswith('_render'):
            gui=obj.data.materials and all(m and m.get('runtime_render_target') for m in obj.data.materials)
            label='03 Screens — native static targets' if gui else ('01 Exterior' if 'merged_VehiclePicker' in obj.name else '02 Cockpit and controls')
        elif obj.type in ('ARMATURE','EMPTY'):label='04 Rig and attachment markers'
        else:continue
        for collection in list(obj.users_collection):collection.objects.unlink(obj)
        collections[label].objects.link(obj)
    world=bpy.data.worlds.new('Native rolling-hills reflection environment');world.use_nodes=True;scene.world=world;nodes=world.node_tree.nodes;nodes.clear();links=world.node_tree.links
    env=node(nodes,'ShaderNodeTexEnvironment','Recovered native cubemap',(-400,0));environment=base/'materials_preview/native_environment/environment_latlong.png'
    if not environment.exists():
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        from asura_textures import prepare_cubemap
        prepare_cubemap(base/'raw/chars__playertanks_pc/graphics/specialfx/cubemaps/bz_cubemap_3_rolling_hills_day.dds',environment.parent)
    env.image=bpy.data.images.load(str(environment),check_existing=True)
    background=node(nodes,'ShaderNodeBackground','Native reflection environment',(-100,0));background.inputs['Strength'].default_value=.35;links.new(env.outputs['Color'],background.inputs['Color'])
    studio=node(nodes,'ShaderNodeBackground','Neutral camera background',(-100,-180));studio.inputs['Color'].default_value=(.025,.032,.045,1);studio.inputs['Strength'].default_value=1
    rays=node(nodes,'ShaderNodeLightPath','Camera background only',(-100,-350));mix=node(nodes,'ShaderNodeMixShader','Reflection / camera background',(150,0));links.new(rays.outputs['Is Camera Ray'],mix.inputs[0]);links.new(background.outputs[0],mix.inputs[1]);links.new(studio.outputs[0],mix.inputs[2]);out=node(nodes,'ShaderNodeOutputWorld','World output',(400,0));links.new(mix.outputs[0],out.inputs['Surface'])
    lights=collections['05 Cameras and lighting']
    def light(name,location,target,power,size,color):
        data=bpy.data.lights.new(name,'AREA');data.energy=power;data.shape='DISK';data.size=size;data.color=color
        obj=bpy.data.objects.new(name,data);lights.objects.link(obj);obj.location=location;obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler();return obj
    cabin=next(o for o in bpy.data.objects if o.get('native_model')=='Tank_Prototype_01_Cockpit').matrix_world.translation.copy()
    light('Soft key',(1.2,-1.0,1.6),(0,0,.15),25,1.1,(1,.94,.84));light('Cool fill',(-1.1,-.4,.65),(0,0,.1),12,.9,(.64,.78,1));light('Rim',(.2,1.1,1.3),(0,0,.1),30,.75,(1,1,1))
    light('Interior broad fill',cabin+Vector((0,-.05,.06)),cabin+Vector((0,.13,-.01)),.8,.14,(.75,.85,1))
    def camera(name,position,target,lens=42):
        data=bpy.data.cameras.new(name);data.lens=lens;data.clip_start=.0001;data.clip_end=100
        obj=bpy.data.objects.new(name,data);lights.objects.link(obj);obj.location=position;obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler();return obj
    exterior=camera('01 Full tank — shaded',(1.25,-1.7,1.0),(0,0,.12),46)
    interior=camera('02 Cockpit — all native surfaces',cabin+Vector((0,-.025,.055)),cabin+Vector((0,.15,-.015)),17)
    console=camera('03 Controls and materials',cabin+Vector((0,-.035,.055)),cabin+Vector((-.07,.08,-.04)),22)
    scene.camera=exterior
    scene.frame_end=200
    for frame,cam in ((1,exterior),(100,interior),(200,console)):
        marker=scene.timeline_markers.new(cam.name,frame=frame);marker.camera=cam
    scene.render.engine='BLENDER_EEVEE';scene.render.resolution_x=1600;scene.render.resolution_y=1000;scene.render.resolution_percentage=100
    scene.view_settings.view_transform='AgX';scene.view_settings.look='AgX - Medium High Contrast';scene.view_settings.exposure=0
    if hasattr(scene.eevee,'use_raytracing'):scene.eevee.use_raytracing=True
    scene['preview_scope']='Native default tint, material-model metal selector and emission constants decoded from DX11 code. Geometry, skeletons and actions preserved. BRDF, cubemap projection, glass, height/parallax and entity overrides remain approximations/unresolved.'
    scene['screen_status']='Live game GUI cannot be rendered from static assets; native screen surfaces and blank target texture are retained.'
    text=bpy.data.texts.new('START HERE — Player tank materials')
    text.write('Camera view (Numpad 0), frames 1: full tank, 100: cockpit, 200: controls. These are inspection cameras, not a recovered runtime eye. All native surfaces are present.\nNative albedo modifier (MARE +0x150), material model (+0x40) and emission gain/color (+0x48/+0x4c) are decoded through loader and constant-buffer code. No invented paint colors.\nMaterials use albedo RGB times native modifier, alpha roughness floor 0.05, packed RG normal reconstruction, native red-channel emission masks and reflection environment.\nThe exact game BRDF, entity callback overrides, glass/OIT, height/parallax and live GUI remain unresolved/approximations. Response LUTs are retained in resources/materials/response_luts.\nNative actions remain in the Action Editor/NLA. All tracks start muted; select the component armature and its native action to inspect motion.\nScaled bone-parent socket offsets are corrected from source glTF matrices. Every assembled render vertex is checked against independent glTF skinning.\n')
    # Save friendly viewport settings and the complete native image set.
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                area.spaces.active.shading.type='MATERIAL';area.spaces.active.shading.use_scene_world=True;area.spaces.active.shading.use_scene_lights=True
                area.spaces.active.region_3d.view_distance=2;area.spaces.active.region_3d.view_location=Vector((0,0,.12))
                area.spaces.active.region_3d.view_perspective='CAMERA';area.spaces.active.overlay.show_overlays=False
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(destination),compress=True)
    camera_names=[o.name for o in (exterior,interior,console)]
    bpy.ops.wm.open_mainfile(filepath=str(destination))
    scene=bpy.context.scene;exterior,interior,console=[bpy.data.objects[name] for name in camera_names]
    reload_validation=validate_assembly(document,source)
    motion_validation=validate_native_motion(document,source)
    images=[dict(name=i.name,packed=bool(i.packed_file),path=i.filepath) for i in bpy.data.images if i.source=='FILE']
    if not all(i['packed'] for i in images):raise ValueError('Scene contains unpacked external images')
    result=dict(file=str(destination),materials=report,images=images,actions=len(bpy.data.actions),mesh_objects=sum(o.type=='MESH' and o.name.endswith('_render') for o in bpy.data.objects),geometry_changed=False,assembly_validation=assembly_validation,reload_validation=reload_validation,motion_validation=motion_validation,import_transform_corrections=socket_fixes,cameras=[dict(name=o.name,matrix=[list(row) for row in o.matrix_world],lens=o.data.lens) for o in (exterior,interior,console)],limitations=[scene['preview_scope'],scene['screen_status'],'Serialized default paint is applied. Entity override callbacks are not reconstructed. Cameras are inspection views, not a verified game POV.'])
    reportpath=base/'reports/player_tank_materials.json';reportpath.write_text(json.dumps(result,indent=2))
    if opts.render:
        for name,cam,frame in [('materials_exterior',exterior,1),('materials_cockpit',interior,100),('materials_controls',console,200)]:
            # Bound camera markers are evaluated again by the render operator.
            scene.frame_set(frame);scene.camera=cam;bpy.context.view_layer.update()
            validate_assembly(document,source)
            scene.render.filepath=str(base/'reports'/(name+'.png'));bpy.ops.render.render(write_still=True)
            if scene.camera!=cam:raise ValueError('Render used an unexpected camera: '+name)
    print('MATERIAL_SCENE_SAVED',destination,len(report),'materials',len(result['images']),'images',flush=True)


if __name__=='__main__':main()
