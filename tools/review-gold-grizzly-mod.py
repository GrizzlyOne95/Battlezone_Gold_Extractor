"""Offline review of reloaded BZR binaries; run through Blender, no game launch."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mod-root',type=Path,required=True);parser.add_argument('--toolkit-root',type=Path,default=Path(__file__).resolve().parents[2]/'BZ98RBlenderToolKit')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);root=args.mod_root.resolve();receipt=json.loads((root.parent/'build.json').read_text());out=root.parent/'review';out.mkdir(exist_ok=True)
    sys.path.insert(0,str(args.toolkit_root/'bz98tools'))
    from bzrmodelporter.ogremesh_serializer import MeshSerializer
    bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene;scene.render.engine='BLENDER_EEVEE';scene.render.resolution_x=1280;scene.render.resolution_y=720;scene.render.resolution_percentage=100;scene.view_settings.view_transform='AgX'
    world=bpy.data.worlds.new('Review studio');world.use_nodes=True;world.node_tree.nodes['Background'].inputs['Color'].default_value=(.035,.045,.065,1);world.node_tree.nodes['Background'].inputs['Strength'].default_value=.6;scene.world=world
    def point(p):return Vector((p[0],-p[2],p[1]))
    materials={};collections={};expected_refs=[]
    for base in ('avtank','avtank_cockpit'):
        collection=bpy.data.collections.new(base);scene.collection.children.link(collection);collections[base]=collection
        with (root/(base+'.mesh')).open('rb') as stream:model=MeshSerializer(stream).read()
        for index,sub in enumerate(model.submeshes()):
            geometry=np.frombuffer(sub.vertex_data.get_vertex_buffer(0).buffer,dtype='<f4').reshape(-1,6)
            uv=np.ndarray((len(geometry),2),dtype='<f4',buffer=sub.vertex_data.get_vertex_buffer(1).buffer,offset=4,strides=(12,4))
            data=np.c_[geometry,uv]
            positions=[point(p) for p in data[:,:3]];faces=np.array(sub.index_array).reshape(-1,3)
            mesh=bpy.data.meshes.new(f'{base}_{index}');mesh.from_pydata(positions,[],faces.tolist());mesh.update();mesh.normals_split_custom_set_from_vertices([point(n) for n in data[:,3:6]])
            layer=mesh.uv_layers.new(name='UVMap')
            for polygon in mesh.polygons:
                polygon.use_smooth=True
                for loop in polygon.loop_indices:
                    uv=data[mesh.loops[loop].vertex_index,6:8];layer.data[loop].uv=(uv[0],1-uv[1])
            obj=bpy.data.objects.new(mesh.name,mesh);collection.objects.link(obj);obj['exported_material']=sub.material_name
            if sub.material_name not in materials:
                hash_=sub.material_name.rsplit('/',1)[-1];material=bpy.data.materials.new(sub.material_name);material.use_nodes=True;nodes=material.node_tree.nodes;links=material.node_tree.links;shader=nodes.get('Principled BSDF');shader.inputs['Roughness'].default_value=.55
                for suffix,socket in [('_d.png','Base Color'),('_e.png','Emission Color')]:
                    path=root/('bzg_'+hash_+suffix);expected_refs.append(str(path))
                    image=bpy.data.images.load(str(path),check_existing=True);tex=nodes.new('ShaderNodeTexImage');tex.image=image;links.new(tex.outputs['Color'],shader.inputs[socket])
                    if socket=='Base Color':
                        # Match the runtime alpha-test path, not a fabricated GUI.
                        threshold=nodes.new('ShaderNodeMath');threshold.operation='GREATER_THAN';threshold.inputs[1].default_value=.5;links.new(tex.outputs['Alpha'],threshold.inputs[0]);links.new(threshold.outputs[0],shader.inputs['Alpha']);material.surface_render_method='DITHERED'
                shader.inputs['Emission Strength'].default_value=1.
                normal=root/('bzg_'+hash_+'_n.png')
                if normal.exists():
                    image=bpy.data.images.load(str(normal),check_existing=True);image.colorspace_settings.name='Non-Color';tex=nodes.new('ShaderNodeTexImage');tex.image=image;node=nodes.new('ShaderNodeNormalMap');links.new(tex.outputs['Color'],node.inputs['Color']);links.new(node.outputs[0],shader.inputs['Normal'])
                materials[sub.material_name]=material
            mesh.materials.append(materials[sub.material_name])
    eye=point(receipt['authored_eye'])
    for name,loc,power,size in [('Key',(7,-8,12),1800,8),('Fill',(-8,-3,5),1000,7),('Interior',tuple(eye+Vector((0,-.2,.3))),18,.8)]:
        data=bpy.data.lights.new(name,'AREA');data.energy=power;data.size=size;obj=bpy.data.objects.new(name,data);scene.collection.objects.link(obj);obj.location=loc;obj.rotation_euler=(Vector((0,0,1.4))-obj.location).to_track_quat('-Z','Y').to_euler()
    def camera(name,position,direction,frame,vertical=None):
        data=bpy.data.cameras.new(name);data.clip_start=.01;data.clip_end=200;data.lens=45
        if vertical:data.sensor_fit='VERTICAL';data.sensor_height=24;data.lens=12/math.tan(math.radians(vertical)/2)
        obj=bpy.data.objects.new(name,data);scene.collection.objects.link(obj);obj.location=position;obj.rotation_euler=Vector(direction).to_track_quat('-Z','Y').to_euler();m=scene.timeline_markers.new(name,frame=frame);m.camera=obj;return obj
    points=[v.co for obj in collections['avtank'].objects for v in obj.data.vertices]
    low=Vector(tuple(min(p[i] for p in points) for i in range(3)));high=Vector(tuple(max(p[i] for p in points) for i in range(3)));center=(low+high)/2;span=max(high-low)
    position=center+Vector((1.2,-1.6,.9))*span;exterior=camera('World model',position,center-position,1)
    first=camera('Cockpit actual authored POV',eye,(0,-1,0),100,64)
    down=camera('Cockpit down 25 degrees',eye,(0,-math.cos(math.radians(25)),-math.sin(math.radians(25))),200,64)
    side=camera('Cockpit side 50 degrees',eye,(-math.sin(math.radians(50)),-math.cos(math.radians(50)),0),300,64)
    receipts=[]
    for name,cam,base,frame in [('world',exterior,'avtank',1),('cockpit',first,'avtank_cockpit',100),('cockpit_down',down,'avtank_cockpit',200),('cockpit_side',side,'avtank_cockpit',300),('cockpit_low_light',first,'avtank_cockpit',100)]:
        for key,collection in collections.items():collection.hide_render=key!=base
        low_light=name=='cockpit_low_light'
        for light in bpy.data.objects:
            if light.type=='LIGHT':light.hide_render=low_light and light.name!='Interior'
        world.node_tree.nodes['Background'].inputs['Strength'].default_value=.025 if low_light else .6
        scene.camera=cam;scene.frame_set(frame);scene.render.filepath=str(out/(name+'.png'));bpy.ops.render.render(write_still=True)
        receipts.append(dict(view=name,camera=[list(r) for r in cam.matrix_world],file=scene.render.filepath,source='reloaded Ogre mesh; bind pose; material approximation'))
    for key,collection in collections.items():collection.hide_render=key!='avtank';collection.hide_viewport=key!='avtank'
    for light in bpy.data.objects:
        if light.type=='LIGHT':light.hide_render=False
    world.node_tree.nodes['Background'].inputs['Strength'].default_value=.6
    scene.camera=exterior;scene.frame_set(1);scene['review_status']='Offline reloaded model. Runtime cockpit takeover, materials, collision and aiming still need native qualification.'
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'gold_grizzly_review.blend'),compress=True)
    (out/'review.json').write_text(json.dumps(dict(eye=receipt['authored_eye'],vertical_fov=64,aspect='16:9',views=receipts,missing_images=[p for p in expected_refs if not Path(p).exists()]),indent=2))
    print('GOLD_GRIZZLY_REVIEW_OK',flush=True)


if __name__=='__main__':main()
