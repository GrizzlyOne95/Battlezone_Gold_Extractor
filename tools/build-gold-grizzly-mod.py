"""Build a standalone Grizzly model replacement using shared BZR serializers.

Run through Blender 5.2, with --dump-root and --game-root after --.
The step-1 runtime model is a posed visual assembly with native BZR aim controls.
Original Gold detail rigs/clips remain in the extraction source, not rewritten.
"""
import argparse
import copy
import hashlib
import importlib
import io
import json
import math
import re
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import wave

sys.path.insert(0,str(Path(__file__).resolve().parent))
from bzr_materials import with_stock_import

import bpy
import numpy as np
from mathutils import Matrix, Vector


REPO=Path(__file__).resolve().parents[1]
SIBLINGS=REPO.parent
SOURCE_OWNERS={}


def libraries(toolkit,toolbox):
    sys.path.insert(0,str(toolkit/'bz98tools'));sys.path.insert(0,str(toolbox))
    return {n:importlib.import_module('bzrmodelporter.'+n) for n in ('ogremesh','ogremesh_serializer','ogreskeleton','ogreskeleton_serializer','bzbwd2_serializer','spacial')}


def source_owner(obj):
    if obj is not None and obj.get('_asura_node_index') in SOURCE_OWNERS:return SOURCE_OWNERS[obj['_asura_node_index']]
    while obj is not None and not obj.get('native_model'):obj=obj.parent
    return obj


def freeze_display_pose():
    """Use source ready/deployed poses, leaving each sampled rig frozen in place."""
    scene=bpy.context.scene;clips=set()
    for obj in bpy.data.objects:
        owner=source_owner(obj);ad=obj.animation_data
        if owner is None or ad is None:continue
        name=owner['native_model'].lower()
        if name.startswith('weapon_hardpoint'):suffix='_deploying'
        elif name in ('autocannon','railgun','artillery_weapon','main_gun'):suffix='_idle'
        else:continue
        choices=[t.name for t in ad.nla_tracks if t.name.lower().endswith(suffix)]
        if not choices and name=='main_gun':choices=[t.name for t in ad.nla_tracks if t.name.lower().endswith('_vehiclepicker_idle')]
        if choices:clips.add(choices[0])
    receipts=[]
    for name in sorted(clips):
        assignments=[];end=0
        for obj in bpy.data.objects:
            ad=obj.animation_data
            if not ad:continue
            for track in ad.nla_tracks:
                if track.name==name:
                    strip=track.strips[0];ad.action=strip.action
                    if strip.action_slot:ad.action_slot=strip.action_slot
                    assignments.append(obj);end=max(end,float(strip.action_frame_end))
        frame=end if name.lower().endswith('_deploying') else 0
        scene.frame_set(math.floor(frame),subframe=frame-math.floor(frame));bpy.context.view_layer.update()
        # No further action evaluates these transforms after the snapshot.
        for obj in assignments:obj.animation_data.action=None
        receipts.append(dict(clip=name,frame=frame,owners=[o.name for o in assignments]))
    bpy.context.view_layer.update();return receipts


def linear(rgb):return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
def srgb(rgb):return np.where(rgb<=.0031308,rgb*12.92,1.055*np.maximum(rgb,0)**(1/2.4)-.055)


def materials(document,source,out,game,namespace='GoldGrizzly',texture_prefix='bzg_',hash_length=8,script_name='goldgrizzly.material'):
    paths={};script=[];receipt=[]
    stock=(game/'BZ_ASSETS_CORE/common/BZ_MATERIALS/BZBase.material').read_text()
    for base,label in [('BZBase','MaskedWorld'),('BZBaseCockpit','MaskedCockpit')]:
        start=re.search(r'\bmaterial '+base+r'\s*\{',stock).start();brace=stock.index('{',start);depth=1;end=brace+1
        while depth:
            depth+=(stock[end]=='{')-(stock[end]=='}');end+=1
        block=stock[start:end].replace('material '+base,'material '+namespace+'/'+label,1)
        block=re.sub(r'(\bpass(?:\s*:\s*\w+)?\s*\{)',r'\1\n            alpha_rejection greater 128',block)
        script.append(block)
    def texture(index):
        return (source.parent/document['images'][document['textures'][index]['source']]['uri']).resolve()
    from urllib.parse import unquote
    def pixels(index,size=None):
        path=Path(unquote(str(texture(index))));im=bpy.data.images.load(str(path),check_existing=False);im.colorspace_settings.name='Non-Color'
        if size and tuple(im.size)!=size:im.scale(*size)
        array=np.empty(im.size[0]*im.size[1]*4,dtype=np.float32);im.pixels.foreach_get(array);array=array.reshape(im.size[1],im.size[0],4);bpy.data.images.remove(im);return array
    def save(array,path):
        if array.shape[2]==3:array=np.dstack((array,np.ones(array.shape[:2])))
        im=bpy.data.images.new('Port bake',width=array.shape[1],height=array.shape[0],alpha=True);im.colorspace_settings.name='Non-Color';im.pixels.foreach_set(np.asarray(array,dtype=np.float32).reshape(-1));im.file_format='PNG';im.filepath_raw=str(path);im.save();bpy.data.images.remove(im)
    for material in document['materials']:
        native=material['extras']['native_material'];p=native['surface_parameters'];hash_=material['extras']['native_material_hash'];prefix=texture_prefix+hash_[:hash_length]
        pbr=material['pbrMetallicRoughness'];base= pixels(pbr['baseColorTexture']['index']) if 'baseColorTexture' in pbr else np.ones((4,4,4))
        rgb=linear(base[:,:,:3])*np.array(p['albedo_modifier']);alpha=np.ones(base.shape[:2])
        # Dedicated native opacity is resolved by the exporter to a custom reference.
        opacity=material['extras'].get('native_separate_opacity_texture')
        if opacity:alpha=pixels(opacity['index'],(base.shape[1],base.shape[0]))[:,:,0]
        color=np.dstack((np.clip(srgb(rgb),0,1),alpha));save(color,out/(prefix+'_d.png'))
        normal='flat_n.png'
        if 'normalTexture' in material:
            normal=prefix+'_n.png';shutil.copy2(Path(unquote(str(texture(material['normalTexture']['index'])))),out/normal)
        emission=rgb*np.array(p['self_illumination_colour'])*float(p['self_illumination_scale'])*(1-float(p['metallic']))
        if 'emissiveTexture' in material:emission*=pixels(material['emissiveTexture']['index'],(base.shape[1],base.shape[0]))[:,:,:1]
        save(np.clip(srgb(emission),0,1),out/(prefix+'_e.png'))
        rough=np.maximum(base[:,:,3],.05);spec=np.clip((1-rough)*.45+.04,0,1)
        save(np.repeat(spec[:,:,None],3,axis=2),out/(prefix+'_s.png'))
        for cockpit in (False,True):
            name=namespace+'/'+('Cockpit/' if cockpit else 'World/')+hash_;paths[material['name'],cockpit]=name
            parent=(namespace+'/MaskedCockpit' if cockpit else namespace+'/MaskedWorld') if opacity else ('BZBaseCockpit' if cockpit else 'BZBase')
            script += [f'material {name} : '+parent,'{' ,f'    set_texture_alias DiffuseMap {prefix}_d.png',f'    set_texture_alias NormalMap {normal}',f'    set_texture_alias SpecularMap {prefix}_s.png',f'    set_texture_alias EmissiveMap {prefix}_e.png','}']
        receipt.append(dict(hash=hash_,source_parameters=p,tint_space='sRGB decode, native linear tint, sRGB encode',emission_clipped_pixels=int((emission>1).any(2).sum()),
            limitations='BZBase specular approximation, LDR emission clipping; static GUI target; native BRDF/glass not reproduced'))
    (out/script_name).write_text(with_stock_import('\n'.join(script)+'\n'),encoding='utf-8');return paths,receipt


def stock_worlds(skeleton):
    worlds={}
    def visit(b):
        if b.handle in worlds:return worlds[b.handle]
        q=b.orientation;local=Matrix.LocRotScale(Vector((b.position.x,b.position.y,b.position.z)),__import__('mathutils').Quaternion((q.w,q.x,q.y,q.z)),Vector((b.scale.x,b.scale.y,b.scale.z)))
        worlds[b.handle]=(visit(b.parent)@local) if b.parent else local;return worlds[b.handle]
    for b in skeleton.bones():visit(b)
    return worlds


def mesh_bytes(lib,parts,skeleton_name,material_names,cockpit):
    om=lib['ogremesh'];mesh=om.Mesh();mesh.skeleton_name=skeleton_name;all_positions=[];receipt=[];expected_arrays=[]
    for part in parts:
        # Geometry is exact evaluated source pose; only BZR control pivots move it.
        positions,normals,uv,triangles,slots,material_slots,bone=part['data']
        for slot in sorted(set(slots)):
            selected=triangles[np.array(slots)==slot];used=np.unique(selected);remap=np.full(len(positions),-1,dtype=int);remap[used]=np.arange(len(used));indices=remap[selected].reshape(-1)
            rows=np.c_[positions[used],normals[used],uv[used]].astype('<f4');sub=mesh.create_submesh(part['name']+'_'+str(slot))
            sub.material_name=material_names[material_slots[slot],cockpit];sub.index_count=len(indices);sub.indices_32_bit=len(used)>=65536
            sub.set_index_buffer(indices.astype('<u4' if sub.indices_32_bit else '<u2').tobytes());vd=sub.create_vertex_data(len(used))
            # Match the stock Grizzly's writable position/normal stream and
            # separate colour/UV stream. BZR updates the former during skinning.
            for kind,semantic,offset in [(om.VET.FLOAT3,om.VES.POSITION,0),(om.VET.FLOAT3,om.VES.NORMAL,12)]:vd.vertex_declaration.create_vertex_element(0,kind,semantic,offset,0)
            vd.create_vertex_buffer(np.ascontiguousarray(rows[:,:6]).tobytes(),0,24)
            attributes=np.zeros(len(rows),dtype=np.dtype([('colour','<u4'),('uv','<f4',(2,))]));attributes['colour']=0xffffffff;attributes['uv']=rows[:,6:8]
            vd.vertex_declaration.create_vertex_element(1,om.VET.COLOUR_ARGB,om.VES.COLOUR,0,0)
            vd.vertex_declaration.create_vertex_element(1,om.VET.FLOAT2,om.VES.TEXTURE_COORDINATES,4,0)
            vd.create_vertex_buffer(attributes.tobytes(),1,12)
            for index in range(len(used)):sub.create_bone_assignment(index,bone,1.)
            all_positions.extend(positions[used]);receipt.append(dict(source=part['name'],vertices=len(used),triangles=len(selected),bone=bone,material=sub.material_name))
            expected_arrays.append((rows,indices,bone))
    points=np.array(all_positions);mesh.aabb=om.AxisAlignedBoundingBox(*points.min(0),*points.max(0));mesh.bound_radius=float(np.linalg.norm(points,axis=1).max())
    stream=io.BytesIO();lib['ogremesh_serializer'].MeshSerializer(stream).write(mesh)
    # Independent shared reader verifies positions, primitive counts and bindings.
    reloaded=lib['ogremesh_serializer'].MeshSerializer(io.BytesIO(stream.getvalue())).read()
    if len(list(reloaded.submeshes()))!=len(receipt):raise ValueError('Submesh loss')
    for sm,expected,(rows,indices,bone) in zip(reloaded.submeshes(),receipt,expected_arrays):
        actual=np.frombuffer(sm.vertex_data.get_vertex_buffer(0).buffer,dtype='<f4').reshape(-1,6)
        if not np.isfinite(actual).all() or len(actual)!=expected['vertices']:raise ValueError('Bad exported vertices')
        np.testing.assert_array_equal(actual,rows[:,:6])
        attributes=np.ndarray((len(rows),2),dtype='<f4',buffer=sm.vertex_data.get_vertex_buffer(1).buffer,offset=4,strides=(12,4))
        np.testing.assert_array_equal(attributes,rows[:,6:8]);np.testing.assert_array_equal(sm.index_array,indices)
        assignments=list(sm.bone_assignments())
        if len(assignments)!=len(rows) or any(v.bone_index!=bone or v.weight!=1 for v in assignments):raise ValueError('Exported control binding mismatch')
    return stream.getvalue(),receipt


def optional_weapon(base,out,game):
    """Explicitly BZR-balanced example; does not replace any equipped weapon."""
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:raise ValueError('FFmpeg is required for the optional Gold autocannon sound')
    source=base/'audio_pcm/sounds__gmsnd_asr/sounds/weapons/pc/auto_cannon_02/wpn_pc_auto_cannon_blast_01.wav'
    sound=out/'bzgauto.wav'
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-y','-i',str(source),'-ac','1','-ar','22050','-c:a','pcm_s16le',str(sound)],check=True,capture_output=True)
    text=(game/'StockODFFiles/gminigun.odf').read_text()
    text=text.replace('wpnName = "Minigun"','wpnName = "Gold Autocannon Prototype"').replace('fireSound = "gmini00.wav"','fireSound = "bzgauto.wav"')
    text='// Optional approximation: stock BZR minigun ordnance/0.20 s delay, Gold one-shot audio.\n// Gold weapon-specific numeric balance remains unresolved. Not auto-equipped.\n'+text
    (out/'bzgauto.odf').write_text(text,encoding='ascii')
    with wave.open(str(sound),'rb') as wav:audio=dict(channels=wav.getnchannels(),sample_rate=wav.getframerate(),sample_width=wav.getsampwidth(),frames=wav.getnframes())
    return dict(odf='bzgauto.odf',sound='bzgauto.wav',source_audio=str(source),source_audio_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),output_audio=audio,
                balance='BZR gminigun.odf template with bullet1 ordnance and 0.20 s delay; not Gold values',equipped=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dump-root',type=Path,required=True);parser.add_argument('--game-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path);parser.add_argument('--reuse-textures',action='store_true',help='Reuse existing verified-source material bakes while iterating geometry');parser.add_argument('--toolkit-root',type=Path,default=SIBLINGS/'BZ98RBlenderToolKit');parser.add_argument('--toolbox-root',type=Path,default=SIBLINGS/'Battlezone98_Modding_Toolbox')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);base=args.dump_root.resolve();game=args.game_root.resolve();out=(args.output_root or base/'bzr_mod/goldgrizzly').resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(REPO/'tools'));lib=libraries(args.toolkit_root,args.toolbox_root)
    from battlezone.archives.zfs import ZFSArchive
    archive=ZFSArchive(game/'bzone.zfs');original=archive.read('avtank.vdf');vdf=lib['bzbwd2_serializer'].VDFSerializer(io.BytesIO(original)).read()
    with (game/'BZ_ASSETS/common/models/avtank.skeleton').open('rb') as f:skeleton=lib['ogreskeleton_serializer'].SkeletonSerializer(f).read()
    with (game/'BZ_ASSETS/common/models/avtank.mesh').open('rb') as f:stock_mesh=lib['ogremesh_serializer'].MeshSerializer(f).read()
    source=base/'bzr_port/player_tank_complete_source.gltf';document=json.loads(source.read_text())
    parents={c:i for i,n in enumerate(document['nodes']) for c in n.get('children',[])}
    for index in range(len(document['nodes'])):
        current=index
        while True:
            extras=document['nodes'][current].get('extras',{})
            if extras.get('native_model'):SOURCE_OWNERS[index]=extras;break
            if current not in parents:break
            current=parents[current]
    bpy.ops.wm.open_mainfile(filepath=str(source.with_suffix('.blend')));pose=freeze_display_pose();graph=bpy.context.evaluated_depsgraph_get()
    meshes=[o for o in bpy.data.objects if o.type=='MESH' and '_asura_node_index' in o]
    for obj in meshes:
        if source_owner(obj) is None:raise ValueError(f'No native mesh owner: {obj.name}, index={obj.get("_asura_node_index")}, map size={len(SOURCE_OWNERS)}')
    body=next(o for o in meshes if 'merged_vehiclepicker' in source_owner(o)['native_model'].lower())
    axis=Matrix(((-1,0,0),(0,0,1),(0,1,0)))
    source_bounds=np.array([axis@(body.matrix_world@Vector(c)) for c in body.bound_box]);width=source_bounds[:,0].max()-source_bounds[:,0].min()
    scale=float((stock_mesh.aabb.max_x-stock_mesh.aabb.min_x)/width)
    offset=Vector((-(source_bounds[:,0].min()+source_bounds[:,0].max())/2*scale,stock_mesh.aabb.min_y-source_bounds[:,1].min()*scale,(stock_mesh.aabb.min_z+stock_mesh.aabb.max_z)/2-(source_bounds[:,2].min()+source_bounds[:,2].max())/2*scale))
    def point(p):return axis@p*scale+offset
    cabin=next(o for o in bpy.data.objects if o.get('native_model')=='Tank_Prototype_01_Cockpit')
    eye=point(cabin.matrix_world.translation+Vector((0,-.025,.055)))
    # Authored inspection eye, recorded explicitly; no claim of recovered Gold POV.
    body_bone=skeleton.get_bone_by_name('AGR11BDA');turret=skeleton.get_bone_by_name('AGR11TUR');pov=skeleton.get_bone_by_name('AGR11POV')
    bda_world=stock_worlds(skeleton)[body_bone.handle];local_eye=bda_world.inverted()@eye
    pov.position=lib['spacial'].Vector3(*local_eye);pov_part=vdf.get_object_by_name('AGR11POV');pov_part.transform.px,pov_part.transform.py,pov_part.transform.pz=local_eye
    # Locate each actual fixed gun muzzle after its ready pose.
    muzzles={};muzzle_directions={}
    for obj in bpy.data.objects:
        index=obj.get('_asura_node_index');marker=document['nodes'][index].get('extras',{}).get('native_marker') if index is not None else None;owner=source_owner(obj)
        if marker and marker['name'].lower()=='muzzle0' and owner:
            muzzles[owner['native_model'].lower()]=point(obj.matrix_world.translation)
            muzzle_directions[owner['native_model'].lower()]=list((axis@(obj.matrix_world.to_3x3()@Vector((0,-1,0)))).normalized())
    mappings=[('AGR11GC1','main_gun'),('AGR11Gc2','autocannon'),('AGR11GS1','railgun'),('AGR11GM1','artillery_weapon')]
    worlds=stock_worlds(skeleton)
    for name,weapon in mappings:
        if weapon not in muzzles:raise ValueError('Missing source muzzle '+weapon)
        part=vdf.get_object_by_name(name);bone=skeleton.get_bone_by_name(name)
        if bone.parent!=turret:bone.parent.remove_child(bone);turret.add_child(bone)
        local=worlds[turret.handle].inverted()@muzzles[weapon];bone.position=lib['spacial'].Vector3(*local)
        part.parent=turret.name;part.transform.px,part.transform.py,part.transform.pz=local
    # Separate cosmetic control children keep model-space bind geometry intact.
    handles={};next_handle=max(b.handle for b in skeleton.bones())+1
    for name,parent in [('gld11bod',body_bone),('gld11gun',turret),('gld21cab',pov),('gld21gun',turret)]:
        b=skeleton.create_bone(name,next_handle);parent.add_child(b);handles[name]=next_handle;next_handle+=1
    if args.reuse_textures:
        cached=json.loads((out.parent/'build.json').read_text())
        if cached['source_sha256']!=hashlib.sha256(source.read_bytes()).hexdigest():raise ValueError('Cached materials belong to another source scene')
        for material in document['materials']:
            hash_=material['extras']['native_material_hash']
            for suffix in ('_d.png','_s.png','_e.png'):
                if not (out/('bzg_'+hash_+suffix)).exists():raise ValueError('Missing cached material texture')
        names={(m['name'],ck):'GoldGrizzly/'+('Cockpit/' if ck else 'World/')+m['extras']['native_material_hash'] for m in document['materials'] for ck in (False,True)}
        material_receipt=cached['materials']
    else:names,material_receipt=materials(document,source,out,game)
    material_path=out/'goldgrizzly.material';material_path.write_text(with_stock_import(material_path.read_text()),encoding='utf-8')
    world_parts=[];cockpit_parts=[];geometry_repairs=[]
    for obj in meshes:
        owner=source_owner(obj);is_cockpit=owner.get('native_namespace')=='chars__playertanks_pc';is_body=obj==body
        evaluated=obj.evaluated_get(graph);mesh=evaluated.to_mesh();mesh.calc_loop_triangles()
        positions=[];normals=[];uv=[];triangles=[];slots=[];world=evaluated.matrix_world;normal_matrix=world.to_3x3().inverted().transposed()
        removed=0;repaired=0
        # Per-corner export preserves UV seams and split normals. The source
        # contains collapsed triangles/zero normals; repair only this port copy.
        for triangle in mesh.loop_triangles:
            corners=[point(world@mesh.vertices[mesh.loops[i].vertex_index].co) for i in triangle.loops]
            face_normal=(corners[1]-corners[0]).cross(corners[2]-corners[0])
            if face_normal.length<=1e-10:removed+=1;continue
            face_normal.normalize()
            ids=[]
            for corner,loop_index in zip(corners,triangle.loops):
                loop=mesh.loops[loop_index];ids.append(len(positions));positions.append(corner);normal=axis@(normal_matrix@mesh.corner_normals[loop_index].vector)
                if normal.length<1e-8:normal=face_normal.copy();repaired+=1
                normals.append(normal.normalized());source_uv=mesh.uv_layers.active.data[loop_index].uv;uv.append((source_uv.x,1-source_uv.y))
            triangles.append(ids);slots.append(triangle.material_index)
        if removed or repaired:geometry_repairs.append(dict(source=owner['native_model'],removed_degenerate_triangles=removed,repaired_zero_corner_normals=repaired,scope='target copy only; collapsed faces omitted, missing normals use geometric face normal'))
        materials_= [m.name for m in mesh.materials]
        values=(np.array(positions),np.array(normals),np.array(uv),np.array(triangles),slots,materials_)
        if not is_cockpit:world_parts.append(dict(name=owner['native_model'],data=(*values,handles['gld11bod' if is_body else 'gld11gun'])))
        # Measured at the authored eye: this world housing encloses the camera
        # and blocks the center ray at 0.089 m. Keep it on the world model;
        # the FP model uses the actual cabin surfaces and the mounted gun.
        if not is_body and owner['native_model'].lower()!='weapon_hardpoint02_top':cockpit_parts.append(dict(name=owner['native_model'],data=(*values,handles['gld21cab' if is_cockpit else 'gld21gun'])))
        evaluated.to_mesh_clear()
    reports={}
    for cockpit,parts,name in [(False,world_parts,'avtank'),(True,cockpit_parts,'avtank_cockpit')]:
        blob,report=mesh_bytes(lib,parts,name+'.skeleton',names,cockpit);(out/(name+'.mesh')).write_bytes(blob);reports[name]=report
        stream=io.BytesIO();lib['ogreskeleton_serializer'].SkeletonSerializer(stream).write(skeleton);(out/(name+'.skeleton')).write_bytes(stream.getvalue())
    stream=io.BytesIO();lib['bzbwd2_serializer'].VDFSerializer(stream).write(vdf);(out/'avtank.vdf').write_bytes(stream.getvalue())
    # Verify modified hardpoint/POV records through the separate read-only reader.
    from battlezone.meshes.legacy import read_vdf
    check=read_vdf(stream.getvalue());assert len(check.parts)>0
    shutil.copy2(REPO/'mods/goldgrizzly/goldgrizzly.ini',out/'goldgrizzly.ini')
    weapon=optional_weapon(base,out,game)
    receipt=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),stock_vdf_sha256=hashlib.sha256(original).hexdigest(),scale=scale,offset=list(offset),axis=[list(row) for row in axis],authored_eye=list(eye),pose_samples=pose,
        meshes=reports,geometry_repairs=geometry_repairs,uv_convention='Blender U,V -> Ogre U,1-V, matching shared toolkit pose_compat/kenshi_compat',materials=material_receipt,optional_weapon=weapon,muzzle_mappings=[dict(part=p,source_weapon=w,world=list(muzzles[w]),source_forward=muzzle_directions[w],runtime_forward=[0,0,1]) for p,w in mappings],
        runtime_scope='Stock Grizzly gameplay, posed Gold visuals and native BZR aim controls. Gold detail clips remain in extraction source; not active in this stock-compatible build.',
        first_person_selection=dict(omitted_world_hull=True,omitted_top_housing=True,top_housing_reason='center-ray occlusion at authored eye, measured 0.08864 m; retained in world model'),
        limitations=['Offline prototype; native runtime loading/aim/visibility not yet qualified','Fixed loadout is an inspection mapping, not decoded Gold stock configuration','Stock collision retained; hero visual bounds differ','Gold eye was not recovered; explicit authored inspection eye used','BZBase approximation, static displays and clipped LDR emission'])
    (out.parent/'build.json').write_text(json.dumps(receipt,indent=2,default=lambda value:value.item() if isinstance(value,np.generic) else str(value)),encoding='utf-8')
    (out/'README.txt').write_text('Gold Grizzly standalone model prototype\n\nEnable this Mod in the BZR Mods menu above other Grizzly replacements. Overrides avtank.mesh, avtank.skeleton, avtank.vdf and avtank_cockpit.*. Disable to restore lower-priority assets. No stock files need replacement.\n\nStock Grizzly weapon ODFs/handling remain. Fixed displayed main gun/autocannon/artillery/railgun are not native Gold balance or dynamic weapon swaps. Detailed Gold clips remain in the extraction source for the later controller phase; this baseline bakes a ready display pose and uses BZR aiming. Native runtime behavior is unqualified.\n\nbzgauto.odf is an optional, unequipped weapon prototype: BZR minigun ballistics/0.20 s delay plus a native Gold autocannon blast resampled to mono PCM16 22050 Hz. It is not recovered Gold balance. It does not replace gminigun or change avtank.odf.\n\nKeep separate from Campaign Reimagined unless deliberately testing override priority. Build receipt and offline review files are outside this runtime folder.\n',encoding='utf-8')
    with zipfile.ZipFile(out.parent/'goldgrizzly.zip','w',zipfile.ZIP_DEFLATED) as archive_:
        for path in sorted(out.iterdir()):
            if path.is_file():archive_.write(path,'goldgrizzly/'+path.name)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out.parent/'runtime_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print('GOLD_GRIZZLY_BUILD_OK',json.dumps(dict(output=str(out),world_submeshes=len(reports['avtank']),cockpit_submeshes=len(reports['avtank_cockpit']),scale=scale)),flush=True)


if __name__=='__main__':main()
