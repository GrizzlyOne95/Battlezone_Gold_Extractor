"""Render the emitted projectile meshes in background Blender, without launching BZR."""
import argparse
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mod-root',type=Path,required=True);parser.add_argument('--toolkit-root',type=Path,default=Path(__file__).resolve().parents[2]/'BZ98RBlenderToolKit');args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);root=args.mod_root.resolve();out=root.parent/'review';out.mkdir(exist_ok=True)
    sys.path.insert(0,str(args.toolkit_root/'bz98tools'));from bzrmodelporter.ogremesh_serializer import MeshSerializer
    receipts=json.loads((root.parent/'projectiles.json').read_text());bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene;scene.render.engine='BLENDER_EEVEE';scene.render.resolution_x=1600;scene.render.resolution_y=1000;scene.render.resolution_percentage=100;scene.view_settings.view_transform='AgX'
    world=bpy.data.worlds.new('Projectile studio');world.use_nodes=True;world.node_tree.nodes['Background'].inputs['Color'].default_value=(.025,.035,.05,1);world.node_tree.nodes['Background'].inputs['Strength'].default_value=.5;scene.world=world
    def point(p):return Vector((p[0],-p[2],p[1]))
    mats={};results=[]
    for index,row in enumerate(receipts):
        stem=row['id'];offset=Vector(((index%4)*2.4,(index//4)*2.8,0));collection=bpy.data.collections.new(stem);scene.collection.children.link(collection)
        with (root/(stem+'.mesh')).open('rb') as stream:model=MeshSerializer(stream).read()
        objects=[]
        for j,sub in enumerate(model.submeshes()):
            geometry=np.frombuffer(sub.vertex_data.get_vertex_buffer(0).buffer,dtype='<f4').reshape(-1,6);uv=np.ndarray((len(geometry),2),dtype='<f4',buffer=sub.vertex_data.get_vertex_buffer(1).buffer,offset=4,strides=(12,4));mesh=bpy.data.meshes.new(stem+str(j));mesh.from_pydata([point(p) for p in geometry[:,:3]],[],np.array(sub.index_array).reshape(-1,3).tolist());mesh.update();mesh.normals_split_custom_set_from_vertices([point(n) for n in geometry[:,3:6]]);layer=mesh.uv_layers.new(name='UVMap')
            for polygon in mesh.polygons:
                polygon.use_smooth=True
                for loop in polygon.loop_indices:
                    t=uv[mesh.loops[loop].vertex_index];layer.data[loop].uv=(t[0],1-t[1])
            material=sub.material_name
            if material not in mats:
                mat=bpy.data.materials.new(material);mat.use_nodes=True;nodes=mat.node_tree.nodes;links=mat.node_tree.links;shader=nodes.get('Principled BSDF');shader.inputs['Roughness'].default_value=.45;hash_=material.rsplit('/',1)[-1];prefix='g'+hash_[:5]
                for suffix,socket in [('_d.png','Base Color'),('_e.png','Emission Color')]:
                    image=bpy.data.images.load(str(root/(prefix+suffix)),check_existing=True);texture=nodes.new('ShaderNodeTexImage');texture.image=image;links.new(texture.outputs['Color'],shader.inputs[socket])
                shader.inputs['Emission Strength'].default_value=1
                path=root/(prefix+'_n.png')
                if path.exists():
                    image=bpy.data.images.load(str(path),check_existing=True);image.colorspace_settings.name='Non-Color';tex=nodes.new('ShaderNodeTexImage');tex.image=image;normal=nodes.new('ShaderNodeNormalMap');links.new(tex.outputs['Color'],normal.inputs['Color']);links.new(normal.outputs[0],shader.inputs['Normal'])
                mats[material]=mat
            mesh.materials.append(mats[material]);obj=bpy.data.objects.new(mesh.name,mesh);collection.objects.link(obj);obj.location=offset;objects.append(obj)
        text=bpy.data.curves.new(stem+' label','FONT');text.body=stem+'  '+row['source_model'];text.size=.12;text.align_x='CENTER';label=bpy.data.objects.new(text.name,text);scene.collection.objects.link(label);label.location=offset+Vector((0,-1.15,-.2));label.rotation_euler=(0,0,0)
        results.append(dict(id=stem,submeshes=len(objects),label=text.body))
    for name,position,power,size in [('Key',(2,-5,7),1000,7),('Fill',(9,5,5),800,6)]:
        data=bpy.data.lights.new(name,'AREA');data.energy=power;data.size=size;obj=bpy.data.objects.new(name,data);scene.collection.objects.link(obj);obj.location=position;obj.rotation_euler=(Vector((3.6,1.4,0))-obj.location).to_track_quat('-Z','Y').to_euler()
    data=bpy.data.cameras.new('Reloaded projectile gallery');data.type='ORTHO';data.ortho_scale=10.5;obj=bpy.data.objects.new(data.name,data);scene.collection.objects.link(obj);obj.location=(3.6,-8,10);obj.rotation_euler=(Vector((3.6,1.1,0))-obj.location).to_track_quat('-Z','Y').to_euler();scene.camera=obj
    for label in [o for o in bpy.data.objects if o.type=='FONT']:label.rotation_euler=obj.rotation_euler
    scene.render.filepath=str(out/'projectile_gallery.png');bpy.ops.render.render(write_still=True);bpy.ops.wm.save_as_mainfile(filepath=str(out/'gold_projectiles_review.blend'),compress=True)
    (out/'projectile_review.json').write_text(json.dumps(dict(models=results,source='Reloaded exported Ogre mesh files; static bind poses and approximate materials; no native effects simulation',image=scene.render.filepath),indent=2));print('GOLD_WEAPONS_REVIEW_OK',flush=True)


if __name__=='__main__':main()
