"""Render representative collection assets from saved/reopened Blender libraries."""
import argparse
import json
from pathlib import Path
import sys
import re

import bpy
from mathutils import Vector


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dump-root',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []);base=args.dump_root.resolve()
    manifest=json.loads((base/'blender_library/manifest.json').read_text());out=base/'reports/library_previews';out.mkdir(exist_ok=True)
    requests=[('player_tank_02','Player tank 02 with shared cockpit'),('player_tank_03','Player tank 03 with shared cockpit'),('light_tank','Light tank cannon example'),('medium_tank','Medium tank cannon example'),('heavy_tank','Heavy tank cannon example'),('nemesis','Nemesis cannon example'),('turret','Toaster cannon turret example'),('drone','Missile_Drone_6'),('ufo','UFO body and dome example'),('droid','Droid twin gun example'),('prop','Control_Pylon'),('structure','Data_store')]
    weapon_batch=next(b for b in manifest['batches'] if b['category']=='player_weapon_assemblies')
    for i,item in enumerate(weapon_batch['items']):requests.append((f'player_weapon_{i+1:02}',item['name']))
    receipts=[];current=None
    for filename,name in requests:
        matches=[(b,i,item) for b in manifest['batches'] for i,item in enumerate(b['items'],1) if item['name']==name]
        if not matches:raise ValueError('Missing preview asset: '+name)
        batch,frame,item=matches[0];path=base/batch['blend']
        if path!=current:bpy.ops.wm.open_mainfile(filepath=str(path));current=path
        for collection in bpy.data.collections:
            if 'native_resource_id' in collection:collection.hide_render=collection['native_resource_id']!=item['id']
        scene=bpy.context.scene;pose_receipt=[]
        if batch['category']=='player_weapon_assemblies':
            collection=next(c for c in bpy.data.collections if c.get('native_resource_id')==item['id'])
            for obj in collection.objects:
                ad=obj.animation_data
                if not ad:continue
                tracks=list(ad.nla_tracks);owner=obj
                while not owner.get('native_model') and owner.parent:owner=owner.parent
                native=str(owner.get('native_model','')).lower()
                if native.startswith('weapon_hardpoint'):choices=[t for t in tracks if re.search(r'_Deploying(?:\.\d+)?$',t.name,re.I)]
                else:choices=[t for t in tracks if re.search(r'_Idle(?:\.\d+)?$',t.name,re.I)]
                if not choices:continue
                strip=choices[0].strips[0];ad.action=strip.action
                if getattr(strip,'action_slot',None):ad.action_slot=strip.action_slot
                at=float(strip.action_frame_end) if native.startswith('weapon_hardpoint') else float(strip.action_frame_start)
                scene.frame_set(int(at),subframe=at-int(at));bpy.context.view_layer.update();ad.action=None
                pose_receipt.append(dict(object=obj.name,clip=choices[0].name,source_action_frame=at,state='native clip pose frozen for this render only'))
        scene.frame_set(frame)
        if batch['category'] in ('player_weapon_assemblies','assembled_examples'):
            target=Vector(item['gallery_center']);offset=scene.camera.location-target;offset.y=abs(offset.y);scene.camera.location=target+offset;scene.camera.rotation_euler=(target-scene.camera.location).to_track_quat('-Z','Y').to_euler()
        scene.render.resolution_x=1280;scene.render.resolution_y=800;scene.render.resolution_percentage=100;scene.render.filepath=str(out/(filename+'.png'))
        bpy.ops.render.render(write_still=True)
        receipts.append(dict(asset=item['id'],source_blend=str(path),camera=scene.camera.name,camera_matrix=[list(r) for r in scene.camera.matrix_world],frame=frame,file=scene.render.filepath,view='isolated inspection collection; other resource collections hidden for render only',native_pose_samples=pose_receipt))
        print('LIBRARY_PREVIEW',filename,flush=True)
    (out/'index.json').write_text(json.dumps(receipts,indent=2))


if __name__=='__main__':main()
