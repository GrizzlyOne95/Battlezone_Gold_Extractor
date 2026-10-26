"""Blender worker: Gold projectile glTF -> shared-codec BZR GEO/mesh/skeleton."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0,str(Path(__file__).parent))
from gold_weapons import recipes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dump-root','game-root','output-root','toolkit-root','toolbox-root'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);out=args.output_root.resolve();base=args.dump_root.resolve()
    spec=importlib.util.spec_from_file_location('grizzly_shared',Path(__file__).with_name('build-gold-grizzly-mod.py'));shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
    lib=shared.libraries(args.toolkit_root,args.toolbox_root)
    from bzrmodelporter.bzgeo import Geo
    from bzrmodelporter.bzgeo_serializer import GeoSerializer
    from bzrmodelporter.spacial import Vector3, UV, Color3
    result=[];used=set();short_hashes={}
    for recipe in recipes():
        if not recipe['model']:continue
        key=(recipe['model'],recipe['model_length'])
        if key in used:continue
        used.add(key);stem='bp'+recipe['id'];source=base/'scenes/chars__actors_pc'/(recipe['model']+'.gltf');doc=json.loads(source.read_text())
        for material in doc['materials']:
            digest=material['extras']['native_material_hash'];short=digest[:5]
            if short in short_hashes and short_hashes[short]!=digest:raise ValueError('Short texture hash collision')
            short_hashes[short]=digest
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=str(source))
        for obj in bpy.data.objects:
            if obj.animation_data:
                obj.animation_data.action=None
                for track in obj.animation_data.nla_tracks:track.mute=True
        bpy.context.scene.frame_set(0);bpy.context.view_layer.update();graph=bpy.context.evaluated_depsgraph_get()
        source_mesh_names={m['name'] for m in doc['meshes']}
        objects=[o for o in bpy.data.objects if o.type=='MESH' and o.data.name in source_mesh_names]
        if {o.data.name for o in objects}!=source_mesh_names:raise ValueError('Source mesh missing or renamed during import')
        axis=Matrix(((-1,0,0),(0,0,1),(0,1,0)))
        points=[]
        for obj in objects:
            evaluated=obj.evaluated_get(graph);posed=evaluated.to_mesh()
            points.extend(axis@(evaluated.matrix_world@v.co) for v in posed.vertices)
            evaluated.to_mesh_clear()
        low=Vector(tuple(min(p[i] for p in points) for i in range(3)));high=Vector(tuple(max(p[i] for p in points) for i in range(3)))
        scale=recipe['model_length']/(high.z-low.z);center=(low+high)/2
        names,material_receipt=shared.materials(doc,source,out,args.game_root,namespace='GoldWeapons/'+stem,texture_prefix='g',hash_length=5,script_name=stem+'.material')
        material_path=out/(stem+'.material');material_path.write_text(material_path.read_text().replace('NormalMap flat_n.png','NormalMap bgflatn.png'))
        material_hash={m['name']:m['extras']['native_material_hash'] for m in doc['materials']}
        skeleton=lib['ogreskeleton'].Skeleton();skeleton.create_bone(stem,0)
        parts=[];removed=0;repaired=0
        geo=Geo();geo.name=stem;geo_positions=[];geo_normals=[]
        for obj in objects:
            evaluated=obj.evaluated_get(graph);mesh=evaluated.to_mesh();mesh.calc_loop_triangles();world=evaluated.matrix_world;nm=world.to_3x3().inverted().transposed()
            positions=[];normals=[];uv=[];triangles=[];slots=[];material_slots=[m.name for m in mesh.materials]
            if not material_slots:
                # Blender may omit a glTF material which has no standard PBR
                # properties. Reattach only the unambiguous single material;
                # the native constants above still supply its BZR recipe.
                if len(doc['materials'])!=1:raise ValueError(f'Imported material slots missing: {obj.name}')
                material_slots=[doc['materials'][0]['name']]
            for tri in mesh.loop_triangles:
                corners=[(axis@(world@mesh.vertices[mesh.loops[i].vertex_index].co)-center)*scale for i in tri.loops]
                normal=(corners[1]-corners[0]).cross(corners[2]-corners[0])
                if normal.length<=1e-10:removed+=1;continue
                normal.normalize();indices=[];face=geo.create_face(len(geo.face_list));face.color=Color3(255,255,255);face.plane.surface_normal=Vector3(*normal);face.plane.distance=-normal.dot(corners[0]);face.poly_area=(corners[1]-corners[0]).cross(corners[2]-corners[0]).length/2
                face.texture_name='g'+material_hash[material_slots[tri.material_index]][:5]+'_d.png'
                for p,loop_index in zip(corners,tri.loops):
                    n=axis@(nm@mesh.corner_normals[loop_index].vector)
                    if n.length<1e-8:n=normal.copy();repaired+=1
                    n.normalize();t=mesh.uv_layers.active.data[loop_index].uv;target_uv=(t.x,1-t.y)
                    indices.append(len(positions));positions.append(p);normals.append(n);uv.append(target_uv)
                    index=len(geo_positions);geo_positions.append(p);geo_normals.append(n);face.create_face_node(index,index,UV(*target_uv))
                triangles.append(indices);slots.append(tri.material_index)
            if positions:parts.append(dict(name=obj.name,data=(np.array(positions),np.array(normals),np.array(uv),np.array(triangles),slots,material_slots,0)))
            evaluated.to_mesh_clear()
        blob,mesh_report=shared.mesh_bytes(lib,parts,stem+'.skeleton',names,False);(out/(stem+'.mesh')).write_bytes(blob)
        stream=io.BytesIO();lib['ogreskeleton_serializer'].SkeletonSerializer(stream).write(skeleton);(out/(stem+'.skeleton')).write_bytes(stream.getvalue())
        geo.vert_count=len(geo_positions);geo.vertex_pos_buffer=np.array(geo_positions,dtype='<f4');geo.vertex_normal_buffer=np.array(geo_normals,dtype='<f4')
        stream=io.BytesIO();GeoSerializer(stream).write(geo);(out/(stem+'.geo')).write_bytes(stream.getvalue());read=GeoSerializer(io.BytesIO(stream.getvalue())).read()
        np.testing.assert_array_equal(np.asarray(read.vertex_pos_buffer).reshape(-1,3),geo.vertex_pos_buffer)
        if len(read.face_list)!=len(geo.face_list):raise ValueError('GEO lost faces')
        result.append(dict(id=stem,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_bin_sha256=hashlib.sha256(source.with_suffix('.bin').read_bytes()).hexdigest(),source_model=recipe['model'],target_length=recipe['model_length'],scale=scale,center=list(center),axis=[list(r) for r in axis],uv_convention='Blender U,V -> Ogre U,1-V',meshes=mesh_report,materials=material_receipt,removed_degenerates=removed,repaired_normals=repaired,pose='Source bind pose, rigid projectile body; source detail clips retained in original libraries'))
        print('PROJECTILE_OK '+stem,flush=True)
    (out.parent/'projectiles.json').write_text(json.dumps(result,indent=2));print('GOLD_PROJECTILES_OK',flush=True)


if __name__=='__main__':main()
