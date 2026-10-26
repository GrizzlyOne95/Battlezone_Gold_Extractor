"""Prepare an offline BZR port inventory and explicit conversion handoff.

Does not alter source scenes, export BZR binaries, or deploy game files.
python tools/prepare-bzr-port.py --dump-root work/asset_dump_2026-10-24
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import quote, unquote

import numpy as np


def accessor(document, index, folder, buffers):
    a=document['accessors'][index];view=document['bufferViews'][a['bufferView']]
    number=view.get('buffer',0)
    if number not in buffers:buffers[number]=(folder/unquote(document['buffers'][number]['uri'])).read_bytes()
    width={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
    dtype=np.dtype({5121:'u1',5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']])
    array=np.ndarray((a['count'],width),dtype=dtype,buffer=buffers[number],
                     offset=view.get('byteOffset',0)+a.get('byteOffset',0),
                     strides=(view.get('byteStride',width*dtype.itemsize),dtype.itemsize))
    if a.get('normalized') and dtype.kind=='u':array=array.astype(float)/np.iinfo(dtype).max
    return array


def skin_risks(joints, weights, limit=3):
    """Measure distinct positive influences, including duplicate source slots."""
    merged=np.array(weights,dtype=float,copy=True)
    for slot in range(merged.shape[1]):
        for previous in range(slot):
            same=joints[:,slot]==joints[:,previous]
            merged[same,previous]+=merged[same,slot];merged[same,slot]=0
    counts=(merged>0).sum(1);ordered=np.sort(merged,axis=1)[:,::-1]
    lost=ordered[:,limit:].sum(1)
    return dict(max_positive_influences=int(counts.max(initial=0)),
                vertices_above_three=int((counts>limit).sum()),
                max_discarded_weight_if_top3=float(lost.max(initial=0)))


def scene_stats(path):
    document=json.loads(path.read_text());buffers={};seen=set();vertices=triangles=affected=0;maximum=0;discarded=0.
    for mesh in document.get('meshes',[]):
        for primitive in mesh['primitives']:
            attributes=primitive['attributes'];position=attributes['POSITION']
            if position not in seen:
                seen.add(position);vertices+=document['accessors'][position]['count']
                if 'WEIGHTS_0' in attributes:
                    risk=skin_risks(accessor(document,attributes['JOINTS_0'],path.parent,buffers),accessor(document,attributes['WEIGHTS_0'],path.parent,buffers))
                    maximum=max(maximum,risk['max_positive_influences']);affected+=risk['vertices_above_three'];discarded=max(discarded,risk['max_discarded_weight_if_top3'])
            if primitive.get('mode',4)!=4:raise ValueError('Unsupported nontriangle source primitive')
            triangles+=document['accessors'][primitive['indices']]['count']//3
    native=next((n.get('extras',{}) for n in document['nodes'] if n.get('extras',{}).get('native_model')), {})
    unresolved=[m['name'] for m in document.get('materials',[]) if not m.get('extras',{}).get('native_material_hash') or int(m.get('extras',{}).get('native_material_hash','0'),16)==0]
    return dict(name=native.get('native_model',path.stem),namespace=native.get('native_namespace','assembly'),
                nodes=len(document['nodes']),mesh_resources=len(document.get('meshes',[])),vertices=vertices,triangles=triangles,
                largest_skin_joints=max((len(s['joints']) for s in document.get('skins',[])),default=0),
                clip_instances=len(document.get('animations',[])),distinct_clips=len({a['name'] for a in document.get('animations',[])}),
                materials=len(document.get('materials',[])),unresolved_materials=unresolved,
                max_positive_influences=maximum,vertices_above_three=affected,max_discarded_weight_if_top3=discarded)


def alias(index, cockpit=False):
    """Stable scene-local proposal; not a claim of an assigned VDF part role."""
    alphabet='0123456789abcdefghijklmnopqrstuvwxyz';number=index;digits=''
    if index<0 or index>=36**3:raise ValueError('Alias space exhausted')
    for _ in range(3):digits=alphabet[number%36]+digits;number//=36
    return 'gld'+('2' if cockpit else '1')+'1'+digits


def csv_write(path,rows,fields=None):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or list(rows[0]));writer.writeheader();writer.writerows(rows)


def revision(path):
    result={'path':str(path),'exists':path.is_dir()}
    if path.is_dir():
        for field,args in [('revision',['rev-parse','HEAD']),('branch',['branch','--show-current']),('remote',['remote','get-url','origin']),('status',['status','--short'])]:
            run=subprocess.run(['git','-C',str(path),*args],capture_output=True,text=True)
            result[field]=run.stdout.strip() if run.returncode==0 else None
    return result


def stage_assembly(source,root,output):
    """Isolate one complete inspection hierarchy; keep source buffer/texture links."""
    document=json.loads(source.read_text());retained=set()
    def visit(index):
        if index in retained:return
        retained.add(index)
        for child in document['nodes'][index].get('children',[]):visit(child)
    visit(root);indices=sorted(retained);remap={old:new for new,old in enumerate(indices)}
    used_meshes=sorted({document['nodes'][i]['mesh'] for i in indices if 'mesh' in document['nodes'][i]})
    used_skins=sorted({document['nodes'][i]['skin'] for i in indices if 'skin' in document['nodes'][i]})
    mesh_map={old:new for new,old in enumerate(used_meshes)};skin_map={old:new for new,old in enumerate(used_skins)}
    nodes=[document['nodes'][i] for i in indices]
    for old,node in zip(indices,nodes):
        node.setdefault('extras',{})['port_source_node']=old
        if 'children' in node:node['children']=[remap[c] for c in node['children']]
        if 'mesh' in node:node['mesh']=mesh_map[node['mesh']]
        if 'skin' in node:node['skin']=skin_map[node['skin']]
    skins=[document['skins'][i] for i in used_skins]
    for skin in skins:
        skin['joints']=[remap[j] for j in skin['joints']]
        if 'skeleton' in skin:skin['skeleton']=remap[skin['skeleton']]
    animations=[]
    for animation in document.get('animations',[]):
        channels=[c for c in animation['channels'] if c['target']['node'] in retained]
        if channels:
            for channel in channels:channel['target']['node']=remap[channel['target']['node']]
            animation['channels']=channels;animations.append(animation)
    selected=nodes[remap[root]];translation=selected.pop('translation',None)
    selected['extras'].update(inspection_layout_only=False,removed_gallery_translation=translation,
                              port_status='source staging; native socket aliases remain unverified; no BZR retargeting applied')
    selected['name']='Complete player tank source assembly'
    document.update(nodes=nodes,skins=skins,meshes=[document['meshes'][i] for i in used_meshes],animations=animations,
                    scenes=[dict(nodes=[remap[root]])],scene=0)
    used_materials=sorted({p['material'] for mesh in document['meshes'] for p in mesh['primitives'] if 'material' in p})
    material_map={old:new for new,old in enumerate(used_materials)}
    for mesh in document['meshes']:
        for primitive in mesh['primitives']:
            if 'material' in primitive:primitive['material']=material_map[primitive['material']]
    document['materials']=[document['materials'][i] for i in used_materials]
    document['extras']=dict(source_assembly=str(source),source_root=root,native_hierarchy_preserved=True,
                           target_status='unconverted source reference; BZR control rig and units unassigned')
    for item in document.get('buffers',[])+document.get('images',[]):
        if item.get('uri'):item['uri']=quote(Path(os.path.relpath((source.parent/unquote(item['uri'])).resolve(),output.parent)).as_posix(),safe='/')
    output.write_text(json.dumps(document,indent=2),encoding='utf-8')
    return dict(gltf=output.name,source_root=root,removed_gallery_translation=translation,retained_nodes=len(nodes),
                retained_meshes=len(used_meshes),retained_skins=len(used_skins),clip_instances=len(animations),
                assembly_pairings='explicit inspection loadout, not certified stock entity')


def build_blender_source(base,output):
    """Save/reload the staged complete source using the established import fixes."""
    import bpy
    from mathutils import Vector
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    def load(name,filename):
        spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(filename));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    helper=load('port_source_materials','build-player-tank-materials.py');inspector=load('port_source_animations','inspect-asura-animations.py')
    source=output/'player_tank_complete_source.gltf';document=json.loads(source.read_text())
    for index,node in enumerate(document['nodes']):node.setdefault('extras',{})['_asura_node_index']=index
    temporary=output/'_port_source_import.gltf';temporary.write_text(json.dumps(document),encoding='utf-8')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # glTF imports key times at the current FPS; establish it before import.
    bpy.context.scene.render.fps=30
    try:bpy.ops.import_scene.gltf(filepath=str(temporary),import_pack_images=False)
    finally:temporary.unlink(missing_ok=True)
    for obj in bpy.data.objects:
        if obj.animation_data:
            obj.animation_data.action=None
            for track in obj.animation_data.nla_tracks:track.mute=True
    scene=bpy.context.scene;scene.render.fps=30;scene.frame_set(0);helper.restore_native_socket_transforms(document)
    materials=helper.wire_materials(document,base)
    points=[obj.matrix_world@Vector(corner) for obj in bpy.data.objects if obj.type=='MESH' for corner in obj.bound_box]
    low=Vector(tuple(min(p[i] for p in points) for i in range(3)));high=Vector(tuple(max(p[i] for p in points) for i in range(3)));center=(low+high)/2;span=max(high-low)
    camera=bpy.data.cameras.new('Complete tank port source');obj=bpy.data.objects.new(camera.name,camera);scene.collection.objects.link(obj)
    obj.location=center+Vector((1.1,1.5,.9))*span;obj.rotation_euler=(center-obj.location).to_track_quat('-Z','Y').to_euler();camera.lens=42;scene.camera=obj
    for name,position,power in [('Key',(1,1,2),60),('Fill',(-1,0,1),25)]:
        data=bpy.data.lights.new(name,'AREA');data.energy=power*span*span;data.size=span;light=bpy.data.objects.new(name,data);scene.collection.objects.link(light)
        light.location=center+Vector(position)*span;light.rotation_euler=(center-light.location).to_track_quat('-Z','Y').to_euler()
    world=bpy.data.worlds.new('Inspection world');world.use_nodes=True;world.node_tree.nodes['Background'].inputs['Color'].default_value=(.15,.18,.22,1);world.node_tree.nodes['Background'].inputs['Strength'].default_value=.6;scene.world=world
    scene.render.engine='BLENDER_EEVEE';scene.render.resolution_x=1280;scene.render.resolution_y=800;scene.render.resolution_percentage=100
    scene.view_settings.view_transform='AgX'
    bpy.data.texts.new('PORT SOURCE - START HERE').write('Complete native tank, cockpit, mounts and inspection loadout; gallery translation removed.\nNo BZR control rig or units assigned. Keep native rigs, sockets and clips as source references.\nBZR conversion plan: repository BZR_PORTING.md; manifests and proposed aliases beside this file.\nClips are preserved in muted NLA tracks. Use inspect-asura-animations.py with --source pointing to this file\'s .gltf for playback.\n')
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':area.spaces.active.region_3d.view_perspective='CAMERA';area.spaces.active.shading.type='MATERIAL'
    destination=output/'player_tank_complete_source.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(destination),compress=True);bpy.ops.file.make_paths_relative();bpy.ops.wm.save_as_mainfile(filepath=str(destination),compress=True)
    bpy.ops.wm.open_mainfile(filepath=str(destination))
    geometry=helper.validate_assembly(document,source);bindings=inspector.audit_bindings(document,list(bpy.data.objects))
    if not bindings['passed']:raise ValueError('Staged source animation binding failure')
    clips=['Railgun_Fire','Weapon_Hardpoint02_Top_Deploying','Monitor_Moving_Left_Transition_NoVR']
    motion=helper.validate_native_motion(document,source,clips)
    missing=[i.filepath for i in bpy.data.images if i.source=='FILE' and not i.packed_file and not Path(bpy.path.abspath(i.filepath)).exists()]
    if missing:raise ValueError('Missing staged source textures')
    result=dict(geometry=geometry,material_report=materials,binding_audit=bindings,motion_validation=motion,missing_images=missing,
                status='validated native source staging; not a converted BZR model')
    (output/'player_tank_complete_source.validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    scene=bpy.context.scene;scene.render.filepath=str(output/'player_tank_complete_source.png');bpy.ops.render.render(write_still=True)
    print('BZR_PORT_SOURCE_OK',len(geometry),bindings['channels_checked'],len(motion),flush=True)


def prepare(base,output,toolkit):
    manifest=json.loads((base/'blender_library/manifest.json').read_text())
    categories={item['id']:batch['category'] for batch in manifest['batches'] for item in batch['items'] if batch['category'] not in ('assembled_examples','player_weapon_assemblies')}
    output.mkdir(parents=True,exist_ok=True);resources=[]
    for path in sorted((base/'scenes').glob('*/*.gltf')):
        row=scene_stats(path);row.update(gltf=path.relative_to(base).as_posix(),category=categories.get(row['namespace']+'/'+row['name'],'node_only'))
        row['suggested_route']='static_sdf_or_prop' if row['clip_instances']==0 and row['largest_skin_joints']<=1 else 'vdf_controls_plus_ogre_details'
        resources.append(row)
    csv_rows=[{**r,'unresolved_materials':';'.join(r['unresolved_materials'])} for r in resources]
    csv_write(output/'resource_readiness.csv',csv_rows)
    weapons=next(b for b in manifest['batches'] if b['category']=='player_weapon_assemblies')
    complete=next(item for item in weapons['items'] if item['name'].startswith('Player tank'))
    staged=stage_assembly(base/weapons['gltf'],complete['root_node'],output/'player_tank_complete_source.gltf')
    pilots=[]
    for label,path in [('player_tank',base/'scenes/standard_player_tank.gltf'),('exterior_weapons',base/'blender_library/gltf/player_weapon_assemblies.gltf'),('complete_tank',output/'player_tank_complete_source.gltf')]:
        document=json.loads(path.read_text());stats=scene_stats(path);parents={c:i for i,n in enumerate(document['nodes']) for c in n.get('children',[])}
        joint_nodes={i for skin in document.get('skins',[]) for i in skin['joints']};aliases=[]
        for index,node in enumerate(document['nodes']):
            extras=node.get('extras',{});aliases.append(dict(node=index,source_name=node['name'],parent=parents.get(index,''),
                proposed_alias=alias(index),role='source_joint' if index in joint_nodes else 'socket' if extras.get('native_marker') else 'root_motion' if extras.get('native_root_motion_pivot') else 'source_node',
                mapping_status='proposal only; no VDF class assigned; retain exact transform and original name'))
        csv_write(output/(label+'_node_map.csv'),aliases)
        materials=[]
        for material in document.get('materials',[]):
            extra=material.get('extras',{});native=extra.get('native_material',{});surface=native.get('surface_parameters',{})
            materials.append(dict(source_name=material['name'],namespace=extra.get('native_namespace'),native_hash=extra.get('native_material_hash'),
                target_material='GoldPort/'+label+'/'+material['name'],source_surface=surface,native_texture_hashes=native.get('texture_hashes',{}),
                conversion='bake native tint once; unpack normal; separate alpha roughness and opacity; bake tinted-albedo emission',
                unresolved=not native or not native.get('hash'),shader_status='BZBase/BZBaseCockpit conversion pending; exact Gold BRDF not available'))
        clips=[]
        for name in sorted({a['name'] for a in document.get('animations',[])}):
            clips.append(dict(source_clip=name,target_clip=name,driver='explicit controller required; no inferred native trigger',
                target='cockpit' if label=='player_tank' else 'world_or_cockpit_per_export',state_assignment='unassigned',time_policy='absolute simulation time; validate before use'))
        pilot=dict(id=label,source_gltf=Path(os.path.relpath(path,base)).as_posix(),source_gltf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),stats=stats,
            node_map=label+'_node_map.csv',materials=materials,clips=clips,
            calibration='Retain source assembly calibration; choose BZR gameplay scale separately against a measured target chassis.',
            legacy_control_roles=['body','yaw','pitch','POV','weapon hardpoints','collision','deploy controls'],
            legacy_controls_status='unassigned; source visual aliases must not be auto-classified as turret/hardpoint parts')
        (output/(label+'_manifest.json')).write_text(json.dumps(pilot,indent=2),encoding='utf-8');pilots.append(pilot)
    siblings=Path(__file__).resolve().parents[2]
    evidence=[revision(toolkit),revision(siblings/'ExtraUtilities'),revision(siblings/'BZR-OpenShim'),revision(siblings/'Campaign-Reimagined')]
    affected=[r for r in resources if r['vertices_above_three']]
    summary=dict(resources=len(resources),render_resources=sum(r['vertices']>0 for r in resources),
        resources_above_three_influences=len(affected),vertices_above_three_influences=sum(r['vertices_above_three'] for r in resources),
        max_discarded_weight_if_top3=max((r['max_discarded_weight_if_top3'] for r in resources),default=0),
        node_only_resources=sum(r['vertices']==0 for r in resources),unresolved_material_resources=sum(bool(r['unresolved_materials']) for r in resources),
        categories=dict(Counter(r['category'] for r in resources)),
        toolkit_influence_policy='Inspected toolkit collector retains top three weights; four-influence source needs explicit reduction/error checks or upstream exporter support.',
        pilots={p['id']:p['stats'] for p in pilots},staged_complete_tank=staged,dependencies=evidence,
        status='offline preparation; no BZR binaries emitted, runtime controllers assigned, or installation modified')
    (output/'port_readiness.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (output/'README.md').write_text('# BZR port preparation\n\nStart with the repository [BZR_PORTING.md](../../../BZR_PORTING.md).\n\n'
        f"Inventoried {summary['resources']} resources; {summary['resources_above_three_influences']} have vertices above the inspected exporter\u2019s three-influence policy.\n\n"
        '[resource_readiness.csv](resource_readiness.csv) lists per-resource geometry, skinning, candidate route and unresolved materials. '
        '[port_readiness.json](port_readiness.json) records totals and exact local tool revisions/status.\n\n'
        '[player_tank_manifest.json](player_tank_manifest.json) and [exterior_weapons_manifest.json](exterior_weapons_manifest.json) retain source scene references, material conversion recipes and unassigned animation drivers. '
        'The node-map CSVs propose unique eight-character visual aliases; they are not authored VDF gameplay roles. Cockpit shell/control helpers need deliberate 1/2-bone assignment before export.\n\n'
        '[player_tank_complete_source.gltf](player_tank_complete_source.gltf) isolates the full tank, cockpit, mounts and example guns as one hierarchy, with only its gallery translation removed. It links the original buffers/textures; keep this folder with the dump. [complete_tank_manifest.json](complete_tank_manifest.json) and [complete_tank_node_map.csv](complete_tank_node_map.csv) describe that staging assembly. Run this tool through Blender with `--build-blender-source` to save/reload/check the matching .blend and render an inspection image.\n\n'
        'No source transforms, weights, names or game files were changed. These are handoff manifests, not executable mod files.\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('dependencies','pilots','categories')},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dump-root',type=Path,required=True)
    parser.add_argument('--build-blender-source',action='store_true',help='Run through Blender: build/check/render an existing staged complete source')
    parser.add_argument('--output-root',type=Path);parser.add_argument('--toolkit-root',type=Path,default=Path(__file__).resolve().parents[2]/'BZ98RBlenderToolKit')
    import sys
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None);base=args.dump_root.resolve();output=(args.output_root or base/'bzr_port').resolve()
    if args.build_blender_source:build_blender_source(base,output)
    else:prepare(base,output,args.toolkit_root.resolve())


if __name__=='__main__':main()
