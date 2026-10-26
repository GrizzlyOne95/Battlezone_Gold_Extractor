"""Prepare categorized native glTF catalogs, then build Blender collection assets.

Python: --dump-root PATH --prepare
Blender: --background --python-exit-code 1 --python this.py -- --dump-root PATH --build
Bulk libraries reference the shared texture tree; native local transforms survive.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict,Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))

PLAYER_GUNS=('main_gun','pistol_weapon','autocannon','gatling_gun','railgun','artillery_weapon','unguided_rocket_launcher','guided_missile_launcher','multi_missile_launcher','laser_missile_launcher','playertank_weapon_cannon_80s')


def player_weapon_role(name):
    n=name.lower()
    if n.startswith('weapon_hardpoint'):return 'external_mount_and_doors'
    if n.endswith('_barrels') and any(w in n for w in ('missile','rocket')):return 'muzzle_controller'
    if n.startswith(('guided_missile_launcher_front_ring','guided_missile_launcher_rear_ring','multi_missile_l_launcher','multi_missile_r_launcher')):return 'weapon_component'
    if any(n==g or n.startswith(g+'_') for g in PLAYER_GUNS):return 'weapon_model_or_variant'
    if n.startswith(('guided_missile','unguided_rocket','multi_missile')):return 'ammunition_or_weapon_component'
    if n.startswith('special_weapon_flare_dispenser'):return 'external_auxiliary_device'
    if n in ('gun80s','weapon_base_shared','tank_prototype_weapon','tank_prototype_special_weapon_dummy'):return 'weapon_interface_or_classic_variant'
    return None


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result);return result


def category(namespace,name):
    n=name.lower()
    if player_weapon_role(n):return 'player_weapons'
    if namespace=='chars__playertanks_pc':return 'cockpit_and_displays'
    if any(w in n for w in ('radar_blip','radar_bl','icon_','intro_screen','menu','popup','controller','callout')):return 'ui_and_helpers'
    if any(w in n for w in ('collision','collsion','col_proxy','occluder','markpoint','markers','invisible','attachmentproxy','tiny_cube')):return 'ui_and_helpers'
    if any(w in n for w in ('destruction','destroyed','chunk','_piece')):return 'debris'
    if 'tank_prototype' in n or 'playertank' in n or namespace.startswith('dlcdata__skin_'):return 'player_tanks'
    if any(w in n for w in ('light_tank','medium_tank','heavy_tank','nemesis_tank','super_tank','lt_striker')):return 'enemy_tanks'
    if any(w in n for w in ('drone','ufo','scout_bike','hopper','soldier_droid','boss_','bomber','jet','minelayer','bomb_truck','bombtruck')):return 'drones_and_enemies'
    if any(w in n for w in ('gun_tower','toaster_gun','cobra_rg','hex_cannon','hexcannon','turret','artillery','sniper')):return 'turrets'
    if any(w in n for w in ('weapon','missile','rocket','autocannon','railgun','pistol','projectile','mine','shotgun','laser','cannon')):return 'weapons'
    return 'props_and_structures'


def bounds(asset):
    worlds=[]
    for node in asset['skeleton'].nodes:
        x,y,z,w=node.rotation
        local=np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w,0],
            [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w,0],
            [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y,0],[0,0,0,1]],dtype=float)
        local[:3,3]=node.translation;worlds.append(local if node.parent<0 else worlds[node.parent]@local)
    mesh=asset['mesh'];points=np.c_[mesh.positions,np.ones(len(mesh.positions))];result=np.zeros((len(points),3))
    for slot in range(4):
        matrices=np.array(worlds)[mesh.joints[:,slot]]
        result+=np.einsum('nij,nj->ni',matrices,points)[:,:3]*mesh.weights[:,slot,None]
    return result.min(0),result.max(0)


def prepare(base,batch_size):
    exporter=module('asura_catalog_export',Path(__file__).with_name('export-asura-scenes.py'))
    root=base/'raw';namespaces=sorted(p.name for p in root.iterdir() if p.is_dir())
    assets,textures,materials,report=exporter.load_assets(root,base/'textures_png',namespaces)
    failures=report['failed_models']+report['failed_animations']+report['failed_material_tables']+report['failed_marker_tables']
    if failures:raise ValueError('Native source validation failed: '+str(failures[:3]))
    groups=defaultdict(list);node_only=[]
    for (ns,name),asset in sorted(assets.items()):
        if len(asset['mesh'].positions):groups[category(ns,name)].append(asset)
        else:node_only.append(dict(namespace=ns,name=name,source=asset['source']))
    output=base/'blender_library';output.mkdir(exist_ok=True);manifest=dict(batches=[],node_only_resources=node_only,limitations=['Native resources include complete meshes, interchangeable components, variants and debris. Category names are heuristics, not entity definitions.','Native geometry/rigs/sockets and candidate clips survive. Entity attachment selection and gameplay behavior are not inferred from filenames.','Gallery root translations are inspection layout only. Clear that root translation to recover native resource placement. Bulk libraries share the dump texture tree.'])
    for label,rows in sorted(groups.items()):
        for start in range(0,len(rows),batch_size):
            batch=rows[start:start+batch_size];number=start//batch_size+1;path=output/'gltf'/f'{label}_{number:02}.gltf';scene=exporter.Scene(path,textures,materials)
            measurements=[bounds(a) for a in batch];cell=max(max((hi-lo)[[0,2]]) for lo,hi in measurements)*1.3+.1;columns=min(5,math.ceil(math.sqrt(len(batch))));items=[]
            for index,(asset,(lo,hi)) in enumerate(zip(batch,measurements)):
                center=(lo+hi)/2;gx=(index%columns)*cell;gy=(index//columns)*cell
                translation=[float(gx-center[0]),float(hi[1]),float(center[2]-gy)]
                identity=f'{asset["namespace"]}/{asset["skeleton"].name}'
                root_index=scene.node('Gallery root: '+asset['skeleton'].name,rotation=[1,0,0,0],translation=translation,extras=dict(library_asset_id=identity,library_category=label,inspection_layout_only=True))
                scene.add_asset(asset,root_index)
                items.append(dict(id=identity,name=asset['skeleton'].name,namespace=asset['namespace'],root_node=root_index,native_bounds=[lo.tolist(),hi.tolist()],gallery_center=[float(gx),float(gy),float((hi[1]-lo[1])/2)],gallery_dimensions=[float(hi[0]-lo[0]),float(hi[2]-lo[2]),float(hi[1]-lo[1])],source=asset['source'],render_source=asset['render_source'],classification='native_resource; may be an entity component',native_sockets=[m['name'] for m in asset['markers']],animations=[a.name for a in asset['animations']]))
            scene.save();manifest['batches'].append(dict(category=label,gltf=path.relative_to(base).as_posix(),blend=f'blender_library/{label}_{number:02}.blend',items=items))
    examples=prepare_assemblies(base,exporter,assets,textures,materials)
    manifest['batches'].append(examples)
    weapon_examples=prepare_player_weapons(base,exporter,assets,textures,materials)
    manifest['batches'].append(weapon_examples)
    manifest['player_weapon_inventory']=[dict(namespace=ns,name=name,role=player_weapon_role(name),source=asset['source'],visible_vertices=len(asset['mesh'].positions),native_sockets=asset['markers'],animations=[a.name for a in asset['animations']]) for (ns,name),asset in assets.items() if player_weapon_role(name)]
    manifest['counts']=dict(render_resources=sum(len(v) for v in groups.values()),node_only_resources=len(node_only),assembly_examples=len(examples['items']),player_weapon_examples=len(weapon_examples['items']),player_weapon_resources=len(manifest['player_weapon_inventory']),batches=len(manifest['batches']),categories={k:len(v) for k,v in groups.items()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));write_library_index(base,manifest);print(json.dumps(manifest['counts'],indent=2),flush=True)


def write_library_index(base,manifest):
    """Human-readable navigation alongside the machine-readable provenance."""
    output=base/'blender_library';counts=manifest['counts']
    lines=['# Battlezone Gold Blender asset library','',
        f"{counts['render_resources']:,} native render resources and {counts['node_only_resources']} node-only resources. Counts include components, repeated resources and variants.",'',
        '## Exterior player weapons','',
        'Open [player_weapon_assemblies.blend](player_weapon_assemblies.blend). Frame 0 is the gallery; use the frames below and Numpad 0 for each camera. Each assembly is a collection asset. Original clips are preserved in muted NLA tracks; enable the intended clip on its owning rig to inspect animation.','',
        '| Frame | Weapon or assembly |','| --- | --- |']
    weapons=next(b for b in manifest['batches'] if b['category']=='player_weapon_assemblies')
    lines += [f"| {i} | {item['name']} |" for i,item in enumerate(weapons['items'],1)]
    lines += ['',f"[player_weapons.csv](player_weapons.csv) lists all {counts['player_weapon_resources']} weapon-related resources, including variants, mounts, ammunition, muzzle controllers and auxiliary devices. These are not {counts['player_weapon_resources']} distinct guns.",'',
        'The final tank example uses explicit inspection aliases for the left autocannon, right railgun, top artillery and central main gun. Native socket matrices and calibrated scale are retained; this combination is not certified as a stock loadout. The guided launcher includes its separate front/rear ring models.','',
        '## Vehicles and other units','',
        '[assembled_examples.blend](assembled_examples.blend) contains 12 inspection assemblies: three player exteriors with the shared cockpit, four enemy cannon tanks, UFO, scout bike, hopper, droid and a cannon turret. Socket/model aliases are recorded in manifest.json and remain unverified against runtime entity state code.','',
        'The packed [standard player tank](../scenes/standard_player_tank_materials.blend) has exterior/cockpit/control cameras at frames 1, 100 and 200.','',
        '## All libraries','', '| Library | Category | Collections |','| --- | --- | --- |']
    for batch in manifest['batches']:
        name=Path(batch['blend']).name
        lines.append(f"| [{name}]({name}) | {batch['category']} | {len(batch['items'])} |")
    lines += ['','Keep the complete dump folder together: bulk libraries use relative links to shared textures. Internal local transforms and rigs remain native; outer Gallery root translations arrange the inspection layout. Clear only that translation to recover native resource placement.','',
        'Default paint, emission and material-model settings are traced from source code. Full game BRDF/LUT response, glass, runtime overrides and dynamic cockpit displays remain incomplete. These files are extraction/inspection assets, not completed BZR gameplay ports.','',
        'Each .validation.json records saved/reopened vertex comparisons and image dependencies. The weapon assembly report also records start/mid/end samples for firing, reload, stow, charge, ring and mount animations. [manifest.json](manifest.json) retains full source paths, aliases and animation inventories.','']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    with (output/'player_weapons.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['namespace','name','role','visible_vertices','source','native_sockets','animations'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for item in sorted(manifest['player_weapon_inventory'],key=lambda r:(r['name'].lower(),r['namespace'])):
            row={key:item[key] for key in fields};row['native_sockets']='; '.join(m['name'] for m in row['native_sockets']);row['animations']='; '.join(row['animations']);writer.writerow(row)
    with (output/'animation_inventory.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['blend','gltf','clip','native_duration_seconds','target_nodes','root_motion_channels','validation_samples_for_clip']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for batch in manifest['batches']:
            document=json.loads((base/batch['gltf']).read_text())
            validation=(base/batch['blend']).with_suffix('.validation.json')
            samples=Counter(r['clip'] for r in json.loads(validation.read_text()).get('motion_validation',[])) if validation.exists() else Counter()
            for animation in document.get('animations',[]):
                targets={c['target']['node'] for c in animation['channels']}
                root_channels=sum(bool(document['nodes'][c['target']['node']].get('extras',{}).get('native_root_motion_pivot') or document['nodes'][c['target']['node']].get('extras',{}).get('native_model')) for c in animation['channels'])
                writer.writerow(dict(blend=batch['blend'],gltf=batch['gltf'],clip=animation['name'],native_duration_seconds=animation.get('extras',{}).get('native_duration'),target_nodes=len(targets),root_motion_channels=root_channels,validation_samples_for_clip=samples[animation['name']]))
    with (output/'README.md').open('a',encoding='utf-8') as stream:
        stream.write('\n## Animation playback\n\n[animation_inventory.csv](animation_inventory.csv) lists exported clip instances, durations, target counts and available sampled checks. Repeated clip names can target different instances. Zero validation samples means preserved but not motion-tested.\n\nUse `tools/inspect-asura-animations.py` through Blender to list clips, activate an exact clip across its owning rigs, optionally validate start/mid/end, and save a separate playable preview. See the repository [animation guide](../../../ANIMATIONS.md) for commands, NLA instructions and limits. The source library stays in its inspection pose.\n')


def prepare_assemblies(base,exporter,assets,textures,materials):
    """Explicit inspection aliases; never claim native entity/loadout decoding."""
    path=base/'blender_library/gltf/assembled_examples.gltf';scene=exporter.Scene(path,textures,materials);items=[]
    def actor(name):return assets['chars__actors_pc',name]
    def start(title,asset):
        index=len(items);lo,hi=bounds(asset);center=(lo+hi)/2;gx=(index%4)*7;gy=(index//4)*7;identity='inspection_assembly/'+title
        root=scene.node('Gallery root: '+title,rotation=[1,0,0,0],translation=[float(gx-center[0]),float(hi[1]),float(center[2]-gy)],extras=dict(library_asset_id=identity,library_category='assembled_examples',inspection_layout_only=True))
        item=dict(id=identity,name=title,namespace='inspection_assembly',root_node=root,native_bounds=[lo.tolist(),hi.tolist()],gallery_center=[gx,gy,float((hi[1]-lo[1])/2)],gallery_dimensions=[float(hi[0]-lo[0])*1.3,float(hi[2]-lo[2])*1.3,float(hi[1]-lo[1])*1.5],source=asset['source'],render_source=asset['render_source'],classification='inspection assembly; component aliases and entity/loadout selection are unverified',attachments=[],native_sockets=[],animations=[])
        items.append(item);return root,item
    # Gameplay bodies provide an independent scale calibration for picker meshes.
    for number,picker,body in [(1,'tank_prototype_01_a_merged_vehiclepicker','tank_prototype_01_a_body'),(2,'tank_prototype_02_vehiclepicker','tank_prototype_02_body'),(3,'tank_prototype_03_vehiclepicker','tank_prototype_03_body')]:
        exterior=actor(picker);original=actor(body);root,item=start(f'Player tank {number:02} with shared cockpit',original)
        sb=exterior['skeleton'].nodes[0].bounds;tb=original['skeleton'].nodes[0].bounds;ratios=(tb[:,1]-tb[:,0])/(sb[:,1]-sb[:,0]);ratio=float(np.median(ratios))
        if not np.allclose(ratios,ratio,rtol=.0001) or not np.allclose(sb*ratio,tb,atol=.00002):raise ValueError('Player picker calibration failed: '+picker)
        transform=np.eye(4);transform[:3,:3]*=ratio;transform[:3,3]=original['skeleton'].nodes[0].translation-exterior['skeleton'].nodes[0].translation*ratio
        _,_,sockets=scene.add_asset(exterior,root,transform=transform);scale=np.eye(4);scale[:3,:3]/=ratio
        wrapper,_,cabin_sockets=scene.add_asset(assets['chars__playertanks_pc','tank_prototype_01_cockpit'],sockets['dome'],transform=scale)
        for socket,model in exporter.COCKPIT_ATTACHMENTS.items():
            asset=assets['chars__playertanks_pc',model];_,_,child_sockets=scene.add_asset(asset,cabin_sockets[socket]);screen=assets.get(('chars__playertanks_pc',model.removesuffix('_gui')+'_guiscreen'))
            if screen and 'screenmarker' in child_sockets:scene.add_asset(screen,child_sockets['screenmarker'])
        for name in ('cockpittop_screen','cockpithl_screen','cockpithr_screen','cockpitshield_screen','cockpitdialogue_screen'):scene.add_asset(assets['chars__playertanks_pc',name],wrapper)
        item['attachments']=[dict(socket='Dome',model='Tank_Prototype_01_Cockpit',scale_cancellation=1/ratio,source_pairing='shared cockpit alias; native sockets and transforms, runtime selection unverified')]
    recipes=[
        ('Light tank cannon example','light_tank_h0',[('body','Tower','light_tank_dome_h0','dome'),('body','Turret','light_tank_turret','turret'),('turret','Weapon','light_tank_cannon','gun')]),
        ('Medium tank cannon example','medium_tank_h0',[('body','Tower','medium_tank_dome_h0','dome'),('body','Turret','medium_tank_turret','turret'),('turret','Weapon','medium_tank_cannon','gun')]),
        ('Heavy tank cannon example','heavy_tank_h0',[('body','Tower','heavy_tank_tower_h0','dome'),('body','Turret','heavy_tank_turret','turret'),('turret','Weapon','heavy_tank_cannon','gun')]),
        ('Nemesis cannon example','nemesis_tank_chassis_h0',[('body','Dome','nemesis_tank_dome_h0','dome'),('dome','Barrel','nemesis_tank_gunbarrel_h0','gun')]),
        ('UFO body and dome example','ufo_h0',[('body','Tower','ufo_dome_h0','dome')]),
        ('Scout bike twin gun example','scout_bike_h0',[('body','Tower','scout_bike_dome_h0','dome'),('body','Turret','scout_bike_turret','turret'),('turret','weapon0','scout_bike_gun','gun0'),('turret','weapon1','scout_bike_gun','gun1')]),
        ('Hopper twin gun example','hopper',[('body','Tower','hopper_dome','dome'),('body','Turret','hopper_turret','turret'),('turret','Weapon0','hopper_gun','gun0'),('turret','Weapon1','hopper_gun','gun1')]),
        ('Droid twin gun example','soldier_droid',[('body','Weapon0','soldier_droid_gun','gun0'),('body','Weapon1','soldier_droid_gun','gun1')]),
        ('Toaster cannon turret example','toaster_gun_base_h0',[('body','Turret','toaster_gun_turret_cannon_h0','turret'),('turret','Barrel0','toaster_gun_barrel_cannon_01','gun')]),
    ]
    for title,name,attachments in recipes:
        asset=actor(name);root,item=start(title,asset);_,_,body_sockets=scene.add_asset(asset,root);sockets={'body':body_sockets}
        for owner,socket,child,label in attachments:
            target=sockets[owner][socket.lower()];_,_,sockets[label]=scene.add_asset(actor(child),target)
            item['attachments'].append(dict(owner=owner,socket=socket,model=child,pairing_status='explicit inspection alias; native socket transform retained, entity selection unverified'))
    scene.save();return dict(category='assembled_examples',gltf=path.relative_to(base).as_posix(),blend='blender_library/assembled_examples.blend',items=items)


def prepare_player_weapons(base,exporter,assets,textures,materials):
    path=base/'blender_library/gltf/player_weapon_assemblies.gltf';scene=exporter.Scene(path,textures,materials);items=[]
    def actor(name):return assets['chars__actors_pc',name]
    def start(name,asset,classification):
        index=len(items);lo,hi=bounds(asset);center=(lo+hi)/2;gx=(index%4)*1.6;gy=(index//4)*1.6;identity='player_weapon_assembly/'+name
        root=scene.node('Gallery root: '+name,rotation=[1,0,0,0],translation=[float(gx-center[0]),float(hi[1]),float(center[2]-gy)],extras=dict(library_asset_id=identity,library_category='player_weapon_assemblies',inspection_layout_only=True))
        item=dict(id=identity,name=name,namespace='player_weapon_assembly',root_node=root,native_bounds=[lo.tolist(),hi.tolist()],gallery_center=[gx,gy,float((hi[1]-lo[1])/2)],gallery_dimensions=[float(hi[0]-lo[0])*1.6,float(hi[2]-lo[2])*1.6,float(hi[1]-lo[1])*1.6],source=asset['source'],render_source=asset['render_source'],classification=classification,attachments=[],animations=[a.name for a in asset['animations']],native_sockets=[m['name'] for m in asset['markers']]);items.append(item);return root,item
    for name in PLAYER_GUNS:
        asset=actor(name);root,item=start(asset['skeleton'].name+' — complete weapon resource',asset,'native weapon hierarchy; external component aliases explicit, runtime loadout unverified')
        _,_,sockets=scene.add_asset(asset,root)
        if name=='guided_missile_launcher':
            for socket,child in [('Front_Ring','guided_missile_launcher_front_ring'),('Rear_Ring','guided_missile_launcher_rear_ring')]:
                scene.add_asset(actor(child),sockets[socket.lower()]);item['attachments'].append(dict(socket=socket,model=child,pairing_status='inspection alias using native named ring socket'))
        controller=name+'_barrels'
        if ('chars__actors_pc',controller) in assets and 'weapon_muzzles' in sockets:
            scene.add_asset(actor(controller),sockets['weapon_muzzles']);item['attachments'].append(dict(socket='Weapon_Muzzles',model=controller,role='native muzzle-controller hierarchy; firing event logic remains raw'))
    for name in ('weapon_hardpoint00_left','weapon_hardpoint01_right','weapon_hardpoint02_top'):
        asset=actor(name);root,item=start(asset['skeleton'].name+' — deployable external mount',asset,'native hardpoint housing, doors, weapon socket and stow/deploy clips');scene.add_asset(asset,root)
    # One complete tank demonstrates source socket mounting, not a stock loadout.
    original=actor('tank_prototype_01_a_body');exterior=actor('tank_prototype_01_a_merged_vehiclepicker');root,item=start('Player tank — external weapon mounting example',original,'inspection loadout: left autocannon, right railgun, top artillery and central main gun; stock compatibility/loadout unverified')
    sb=exterior['skeleton'].nodes[0].bounds;tb=original['skeleton'].nodes[0].bounds;ratio=float(np.median((tb[:,1]-tb[:,0])/(sb[:,1]-sb[:,0])));transform=np.eye(4);transform[:3,:3]*=ratio;transform[:3,3]=original['skeleton'].nodes[0].translation-exterior['skeleton'].nodes[0].translation*ratio
    _,_,sockets=scene.add_asset(exterior,root,transform=transform);scale=np.eye(4);scale[:3,:3]/=ratio
    wrapper,_,cabin_sockets=scene.add_asset(assets['chars__playertanks_pc','tank_prototype_01_cockpit'],sockets['dome'],transform=scale)
    for socket,name in exporter.COCKPIT_ATTACHMENTS.items():
        _,_,child_sockets=scene.add_asset(assets['chars__playertanks_pc',name],cabin_sockets[socket]);screen=assets.get(('chars__playertanks_pc',name.removesuffix('_gui')+'_guiscreen'))
        if screen and 'screenmarker' in child_sockets:scene.add_asset(screen,child_sockets['screenmarker'])
    for name in ('cockpittop_screen','cockpithl_screen','cockpithr_screen','cockpitshield_screen','cockpitdialogue_screen'):scene.add_asset(assets['chars__playertanks_pc',name],wrapper)
    for socket,mount,gun in [('weapon_l_vehiclepicker_grp','weapon_hardpoint00_left','autocannon'),('weapon_r_vehiclepicker_grp','weapon_hardpoint01_right','railgun'),('weapon_hardpoint_top_grp','weapon_hardpoint02_top','artillery_weapon')]:
        _,_,mount_sockets=scene.add_asset(actor(mount),sockets[socket],transform=scale);scene.add_asset(actor(gun),mount_sockets['weapon']);item['attachments'].append(dict(tank_socket=socket,mount=mount,mount_socket='Weapon',weapon=gun,pairing_status='explicit inspection loadout; native transforms and scale cancellation'))
    _,_,central=scene.add_asset(actor('tank_prototype_weapon'),sockets['muzzle1'],transform=scale);scene.add_asset(actor('main_gun'),central['weapon']);item['attachments'].append(dict(tank_socket='Muzzle1',model='tank_prototype_weapon',weapon='main_gun',pairing_status='central weapon inspection alias; native entity mapping unverified'))
    scene.save();return dict(category='player_weapon_assemblies',gltf=path.relative_to(base).as_posix(),blend='blender_library/player_weapon_assemblies.blend',items=items)


def build(base,selected=None):
    import bpy
    from mathutils import Vector
    helper=module('asura_blender_materials',Path(__file__).with_name('build-player-tank-materials.py'))
    inspector=module('asura_animation_inspector',Path(__file__).with_name('inspect-asura-animations.py'))
    manifest=json.loads((base/'blender_library/manifest.json').read_text());results=[];failed=[]
    previous=base/'blender_library/build-summary.json'
    if selected and previous.exists():
        old=json.loads(previous.read_text());results=[r for r in old['completed'] if r['category'] not in selected]
        rebuilding={b['blend'] for b in manifest['batches'] if b['category'] in selected};failed=[r for r in old['failed'] if r['batch'] not in rebuilding]
    for batch in manifest['batches']:
        if selected and batch['category'] not in selected:continue
        destination=base/batch['blend'];report_path=destination.with_suffix('.validation.json')
        print('LIBRARY_BUILD',destination.name,len(batch['items']),flush=True)
        try:
            source=base/batch['gltf'];doc=json.loads(source.read_text())
            for i,n in enumerate(doc['nodes']):n.setdefault('extras',{})['_asura_node_index']=i
            temporary=source.with_name('_blender_'+source.name);temporary.write_text(json.dumps(doc))
            bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene;scene.render.fps=30
            try:bpy.ops.import_scene.gltf(filepath=str(temporary),import_pack_images=False)
            finally:temporary.unlink(missing_ok=True)
            for obj in bpy.data.objects:
                if obj.animation_data:
                    obj.animation_data.action=None
                    for track in obj.animation_data.nla_tracks:track.mute=True
            scene.frame_set(0);bpy.context.view_layer.update();fixes=helper.restore_native_socket_transforms(doc)
            validation=helper.validate_assembly(doc,source);material_report=helper.wire_materials(doc,base)
            parents={c:i for i,n in enumerate(doc['nodes']) for c in n.get('children',[])}
            ownership={};collections={}
            for item in batch['items']:
                col=bpy.data.collections.new(item['name']+' ['+item['namespace']+']');scene.collection.children.link(col);collections[item['id']]=col
                col.asset_mark();col.asset_data.description='Native Asura resource: hierarchy, sockets, UVs and candidate animation clips. '+item['classification'];col.asset_data.tags.new(batch['category']);col['source']=item['source'];col['native_resource_id']=item['id']
            for index,n in enumerate(doc['nodes']):
                current=index
                while True:
                    identity=doc['nodes'][current].get('extras',{}).get('library_asset_id')
                    if identity:ownership[index]=identity;break
                    if current not in parents:break
                    current=parents[current]
            for obj in list(bpy.data.objects):
                identity=ownership.get(obj.get('_asura_node_index'))
                if not identity:continue
                for col in list(obj.users_collection):col.objects.unlink(obj)
                collections[identity].objects.link(obj)
            world=bpy.data.worlds.new('Inspection world');world.use_nodes=True;scene.world=world;bg=world.node_tree.nodes.get('Background');bg.inputs['Color'].default_value=(.15,.18,.22,1);bg.inputs['Strength'].default_value=.6
            all_centers=np.array([i['gallery_center'] for i in batch['items']]);all_dims=np.array([i['gallery_dimensions'] for i in batch['items']]);minimum=(all_centers-all_dims/2).min(0);maximum=(all_centers+all_dims/2).max(0);center=Vector((minimum+maximum)/2);span=max(float(np.max(maximum-minimum)),.1)
            lights=bpy.data.collections.new('Inspection cameras and lights');scene.collection.children.link(lights)
            for name,offset,power in [('Key',(1,-1,2),35),('Fill',(-1,-.3,1),18)]:
                data=bpy.data.lights.new(name,'AREA');data.energy=power*span*span;data.shape='DISK';data.size=span;obj=bpy.data.objects.new(name,data);lights.objects.link(obj);obj.location=center+Vector(offset)*span;obj.rotation_euler=(center-obj.location).to_track_quat('-Z','Y').to_euler()
            def camera(name,target,distance,frame):
                data=bpy.data.cameras.new(name);data.lens=42;data.clip_start=max(distance/10000,1e-6);data.clip_end=max(distance*100,100)
                obj=bpy.data.objects.new(name,data);lights.objects.link(obj);obj.location=target+Vector((1.1,-1.5,1))*distance;obj.rotation_euler=(target-obj.location).to_track_quat('-Z','Y').to_euler();marker=scene.timeline_markers.new(name,frame=frame);marker.camera=obj;return obj
            overview=camera('000 Gallery overview',center,float(span),0)
            for index,item in enumerate(batch['items'],1):
                camera(f'{index:03} '+item['name'],Vector(item['gallery_center']),max(max(item['gallery_dimensions']),.02),index)
                # Collection instances are centered in the Asset Browser.
                root=next(o for o in collections[item['id']].objects if o.get('library_asset_id')==item['id']);collections[item['id']].instance_offset=root.matrix_world.translation
            scene.camera=overview;scene.frame_end=len(batch['items']);scene.render.engine='BLENDER_EEVEE';scene.render.resolution_x=1600;scene.render.resolution_y=1000;scene.view_settings.view_transform='AgX'
            text=bpy.data.texts.new('START HERE');text.write('Frame 0: gallery. Frames 1..N: individual native resources. Numpad 0: camera view. Collections are Asset Browser assets.\nNative default material tint, metal selector, emission and texture channels decoded from code. Exact game BRDF, glass and entity overrides remain unresolved.\nRoot translation is gallery layout only; internal local transforms remain native. Clear the Gallery root translation to restore native placement.\nThis is a native-resource library, not a claim that every collection is a fully equipped entity. Candidate clips are preserved in muted NLA tracks.\nShared texture files must travel with the dump. Full provenance is in blender_library/manifest.json.\n')
            for screen in bpy.data.screens:
                for area in screen.areas:
                    if area.type=='VIEW_3D':area.spaces.active.region_3d.view_perspective='CAMERA';area.spaces.active.shading.type='MATERIAL';area.spaces.active.overlay.show_overlays=False
            bpy.ops.wm.save_as_mainfile(filepath=str(destination),compress=True);bpy.ops.file.make_paths_relative();bpy.ops.wm.save_as_mainfile(filepath=str(destination),compress=True);bpy.ops.wm.open_mainfile(filepath=str(destination));reloaded=helper.validate_assembly(doc,source)
            binding_audit=inspector.audit_bindings(doc,list(bpy.data.objects))
            if not binding_audit['passed']:raise ValueError('Animation curve binding coverage failed: '+str(binding_audit['missing_channels'][:3]))
            motion=[]
            if batch['category']=='player_weapon_assemblies':
                names=sorted({a['name'] for a in doc.get('animations',[])})
                prefixes=['Main_Gun_','Pistol_Weapon_','Autocannon_','Gatling_Gun_','Railgun_','Artillery_','Unguided_Rocket_Launcher_','Guided_Missile_Launcher_','Multi_missile_launcher_','Laser_Missile_Launcher_']
                chosen=[]
                for prefix in prefixes:
                    for state in ('fire','reload','stow'):
                        matches=[name for name in names if name.lower().startswith(prefix.lower()) and re.search(r'(?:^|_)'+state+r'(?:_|$)',name.lower())]
                        if matches:chosen.append(matches[0])
                if 'Railgun_Charge' in names:chosen.append('Railgun_Charge')
                for extra in ('Gatling_Gun_Fire_Loop','Guided_Missile_Front_Ring_Fire','Guided_Missile_Rear_Ring_Fire','Guided_Missile_Launcher_Lockon_Start','Weapon_Hardpoint00_Left_Deploying','Weapon_Hardpoint01_Right_Deploying','Weapon_Hardpoint02_Top_Deploying','Unguided_Rocket_Launcher_ReloadPartial'):
                    if extra in names and extra not in chosen:chosen.append(extra)
                motion=helper.validate_native_motion(doc,source,chosen)
            missing=[i.filepath for i in bpy.data.images if i.source=='FILE' and not i.packed_file and not Path(bpy.path.abspath(i.filepath)).exists()]
            if missing:raise ValueError('Missing image dependencies: '+str(missing[:3]))
            result=dict(file=str(destination),category=batch['category'],assets=len(batch['items']),meshes=len(reloaded),actions=len(bpy.data.actions),materials=len(material_report),unresolved_materials=[r['native_hash'] for r in material_report if r.get('unresolved')],material_report=material_report,reloaded_vertex_checks=reloaded,motion_validation=motion,missing_images=missing,max_position_error=max(r['max_world_position_error'] for r in reloaded),transform_corrections=fixes)
            result['binding_audit']=binding_audit
            report_path.write_text(json.dumps(result,indent=2));results.append({k:v for k,v in result.items() if k not in ('material_report','reloaded_vertex_checks','transform_corrections')});print('LIBRARY_OK',destination.name,result['meshes'],result['max_position_error'],flush=True)
        except Exception as error:
            import traceback;traceback.print_exc();failed.append(dict(batch=batch['blend'],error=str(error)))
    (base/'blender_library/build-summary.json').write_text(json.dumps(dict(completed=results,failed=failed),indent=2))
    if failed:raise ValueError(f'{len(failed)} library batches failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dump-root',type=Path,required=True);parser.add_argument('--prepare',action='store_true');parser.add_argument('--build',action='store_true');parser.add_argument('--index-only',action='store_true',help='Refresh README and CSV inventories from existing catalogs/reports');parser.add_argument('--batch-size',type=int,default=40);parser.add_argument('--categories',help='Comma-separated categories to build')
    options=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None);base=options.dump_root.resolve()
    if not (options.prepare or options.build or options.index_only):parser.error('Choose --prepare, --build or --index-only')
    if options.batch_size<1:parser.error('--batch-size must be positive')
    if options.prepare:prepare(base,options.batch_size)
    if options.build:build(base,options.categories.split(',') if options.categories else None)
    if options.build or options.index_only:write_library_index(base,json.loads((base/'blender_library/manifest.json').read_text()))


if __name__=='__main__':main()
