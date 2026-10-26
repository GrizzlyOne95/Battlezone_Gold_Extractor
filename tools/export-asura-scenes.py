#!/usr/bin/env python3
"""Export actual render meshes, native hierarchy, material links and HCAN keys.

Each asset gets a glTF scene. The cockpit assembly uses named native sockets;
the alias list is explicit and is reported separately from native parent links.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import struct
from urllib.parse import quote
import numpy as np

from asura_scene import (name_hash, sample_rotations, read_skeleton, read_render_mesh,
                         validate_render_binding, read_animation, read_markers,
                         read_materials, coalesce_skin_influences)


class Scene:
    def __init__(self, path, texture_index, materials):
        self.path = Path(path)
        self.texture_index = texture_index
        self.materials = materials
        self.data = bytearray()
        self.doc = dict(asset=dict(version='2.0', generator='Battlezone Gold structured scene exporter'), scene=0, scenes=[dict(nodes=[])], nodes=[], meshes=[], skins=[], animations=[], materials=[], textures=[], images=[], samplers=[dict(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=10497)], accessors=[], bufferViews=[])
        self.material_map = {}
        self.image_map = {}
        self.assets = []
        self.issues = []

    def node(self, name, parent=None, **kwargs):
        index = len(self.doc['nodes'])
        self.doc['nodes'].append(dict(name=name, **kwargs))
        if parent is None:
            self.doc['scenes'][0]['nodes'].append(index)
        else:
            self.doc['nodes'][parent].setdefault('children', []).append(index)
        return index

    def accessor(self, values, dtype='<f4', kind=None, target=None):
        values = np.asarray(values, dtype=dtype)
        components = 1 if values.ndim == 1 else values.shape[1]
        kind = kind or {1: 'SCALAR', 2: 'VEC2', 3: 'VEC3', 4: 'VEC4', 16: 'MAT4'}[components]
        while len(self.data) % 4:
            self.data.append(0)
        view = dict(buffer=0, byteOffset=len(self.data), byteLength=values.nbytes)
        if target:
            view['target'] = target
        vi = len(self.doc['bufferViews']); self.doc['bufferViews'].append(view)
        self.data.extend(values.tobytes())
        item = dict(bufferView=vi, componentType={'<f4': 5126, '<u2': 5123, 'u1': 5121, '<u4': 5125}[dtype], count=len(values), type=kind)
        if values.size and kind != 'MAT4':
            item['min'] = np.min(values, axis=0).reshape(-1).tolist()
            item['max'] = np.max(values, axis=0).reshape(-1).tolist()
        index = len(self.doc['accessors']); self.doc['accessors'].append(item)
        return index

    def texture(self, key, namespace):
        candidates = self.texture_index.get(key, [])
        local = [p for ns, p in candidates if ns == namespace]
        if not local:
            local = [p for ns, p in candidates if ns in ('misc__common_asr', 'chars__actors_pc', 'chars__playertanks_pc')]
        if not local:
            return None
        path = local[0]
        return self.texture_path(path)

    def texture_path(self, path):
        if path not in self.image_map:
            i = len(self.doc['images'])
            self.doc['images'].append(dict(uri=quote(os.path.relpath(path, self.path.parent).replace('\\', '/'), safe='/.:_-')))
            ti = len(self.doc['textures'])
            self.doc['textures'].append(dict(source=i, sampler=0))
            self.image_map[path] = ti
        return dict(index=self.image_map[path])

    def material(self, key, namespace):
        cache_key = (namespace, key)
        if cache_key in self.material_map:
            return self.material_map[cache_key]
        record = self.materials.get(namespace, {}).get(key)
        if record is None:
            record = next((records[key] for records in self.materials.values() if key in records), None)
        result = dict(name=f'mat_{key:08x}', doubleSided=True, pbrMetallicRoughness=dict(metallicFactor=0, roughnessFactor=0.65), extras=dict(native_material_hash=f'{key:08x}',native_namespace=namespace))
        if record:
            parameters=record.get('surface_parameters',{})
            if parameters:
                result['pbrMetallicRoughness']['baseColorFactor']=[*parameters['albedo_modifier'],1]
                result['pbrMetallicRoughness']['metallicFactor']=parameters['metallic']
                result['extras']['material_translation']='Serialized default tint and model selector decoded from native code; Blender/glTF BRDF remains approximate.'
            refs = record['texture_hashes']
            textures = {role: self.texture(value, namespace) for role, value in refs.items() if value}
            result['extras']['native_material'] = record
            result['extras']['resolved_texture_roles'] = [role for role, tex in textures.items() if tex]
            if textures.get('base_color'):
                result['pbrMetallicRoughness']['baseColorTexture'] = textures['base_color']
            if textures.get('normal'):
                # Core glTF requires tangent XYZ, not the native packed RG map.
                from PIL import Image
                from urllib.parse import unquote
                source_image=self.doc['textures'][textures['normal']['index']]['source']
                normal_path=self.path.parent/unquote(self.doc['images'][source_image]['uri'])
                derived=self.path.parent/'_material_textures'/f'{namespace}_{refs["normal"]:08x}_normal_opengl.png'
                derived.parent.mkdir(parents=True,exist_ok=True)
                if not derived.exists() or derived.stat().st_mtime_ns<normal_path.stat().st_mtime_ns:
                    with Image.open(normal_path) as image:packed=np.array(image.convert('RGBA'))
                    xy=packed[:,:,:2].astype(np.float64)/255*2-1
                    normal=np.empty_like(packed);normal[:,:,0]=packed[:,:,0];normal[:,:,1]=255-packed[:,:,1]
                    normal[:,:,2]=np.rint((np.sqrt(np.maximum(0,1-(xy*xy).sum(2)))*.5+.5)*255).astype(np.uint8);normal[:,:,3]=255
                    Image.fromarray(normal).save(derived)
                result['extras']['native_packed_normal_texture']=textures['normal']
                result['extras']['normal_preview_conversion']='RG signed XY, reconstructed positive Z, inverted tangent green; 8-bit derived preview. Native original retained.'
                result['normalTexture'] = self.texture_path(derived.resolve())
            if textures.get('emissive'):
                result.update(emissiveTexture=textures['emissive'], emissiveFactor=[1, 1, 1])
            if textures.get('opacity'):
                result['extras']['native_separate_opacity_texture'] = textures['opacity']
                image_id = self.doc['textures'][textures['opacity']['index']]['source']
                opacity_uri = self.doc['images'][image_id]['uri']
                # Dedicated alpha maps can be carried by core glTF via base RGBA.
                # Render targets and generic effect channels are kept as metadata.
                if '_alpha.' in opacity_uri and textures.get('base_color'):
                    from PIL import Image
                    from urllib.parse import unquote
                    base_id = self.doc['textures'][textures['base_color']['index']]['source']
                    base_path = self.path.parent / unquote(self.doc['images'][base_id]['uri'])
                    opacity_path = self.path.parent / unquote(opacity_uri)
                    derived = self.path.parent / '_material_textures' / f'{namespace}_{key:08x}_rgba.png'
                    derived.parent.mkdir(parents=True, exist_ok=True)
                    if not derived.exists():
                        with Image.open(base_path) as base_img, Image.open(opacity_path) as alpha_img:
                            rgba = base_img.convert('RGBA')
                            alpha = alpha_img.convert('RGB').getchannel('R')
                            if base_img.size != alpha_img.size:
                                alpha = alpha.resize(base_img.size, Image.Resampling.BILINEAR)
                                result['extras']['alpha_resampled_for_gltf'] = True
                            rgba.putalpha(alpha)
                            rgba.save(derived)
                    result['pbrMetallicRoughness']['baseColorTexture'] = self.texture_path(derived.resolve())
                    result['alphaMode'] = 'BLEND'
            if any('guimenu_output' in str(p) for _, p in self.texture_index.get(refs['base_color'], [])):
                result['extras']['runtime_render_target'] = True
                result['extras']['requires_bzr_ui_reimplementation'] = True
        else:
            result['extras']['unresolved'] = True
            self.issues.append(dict(kind='unresolved_material', namespace=namespace, hash=f'{key:08x}'))
        index = len(self.doc['materials']); self.doc['materials'].append(result)
        self.material_map[cache_key] = index
        return index

    def add_asset(self, asset, parent=None, transform=None, label=None, animation_filter=None):
        skeleton, mesh, namespace = asset['skeleton'], asset['mesh'], asset['namespace']
        kwargs = dict(extras=dict(native_source=asset['source'], native_namespace=namespace, native_model=skeleton.name, native_units=True))
        if transform is not None:
            kwargs['matrix'] = np.asarray(transform).T.reshape(-1).tolist()
        wrapper = self.node(label or skeleton.name, parent, **kwargs)
        # Root motion must remain inside a fixed attachment calibration. glTF
        # cannot animate TRS on a node that also has a matrix, and overwriting
        # the inverse picker scale would magnify deployment translation 32x.
        motion_parent=wrapper
        if transform is not None and any(a.root_motion_track and (not animation_filter or a.name in animation_filter) for a in asset['animations']):
            motion_parent=self.node(skeleton.name+' native root motion',wrapper,extras=dict(native_root_motion_pivot=True))
        joints = []
        for i, node in enumerate(skeleton.nodes):
            joints.append(self.node(node.name, motion_parent if node.parent < 0 else joints[node.parent], translation=node.translation.tolist(), rotation=node.rotation.tolist(), extras=dict(native_node_index=i, native_name_hash=f'{name_hash(node.name):08x}',native_node_flags=node.native_flags)))
        if len(mesh.positions):
            export_joints,export_weights,coalesced=coalesce_skin_influences(mesh.joints,mesh.weights)
            attributes = dict(POSITION=self.accessor(mesh.positions, target=34962), NORMAL=self.accessor(mesh.normals, target=34962), TEXCOORD_0=self.accessor(mesh.uv0, target=34962), TEXCOORD_1=self.accessor(mesh.uv1, target=34962), JOINTS_0=self.accessor(export_joints, 'u1', target=34962), WEIGHTS_0=self.accessor(export_weights, target=34962))
            primitives = []
            for part in mesh.parts:
                first, count = part['first_triangle'], part['triangle_count']
                if not count:
                    continue
                primitives.append(dict(attributes=attributes, indices=self.accessor(mesh.triangles[first:first + count].reshape(-1), '<u2', target=34963), material=self.material(part['material_hash'], namespace), mode=4, extras=dict(native_submesh=part)))
            mesh_index = len(self.doc['meshes']); self.doc['meshes'].append(dict(name=skeleton.name, primitives=primitives,extras=dict(native_bone_palettes=mesh.bone_palettes or [],nonfinite_source_uv_components=mesh.nonfinite_uv or [],coalesced_duplicate_influences=coalesced)))
            inverse = self.accessor(np.tile(np.eye(4).reshape(1, 16), (len(joints), 1)), kind='MAT4')
            skin = len(self.doc['skins']); self.doc['skins'].append(dict(joints=joints, inverseBindMatrices=inverse, skeleton=joints[0], extras=dict(native_local_vertex_skinning=True)))
            self.node(skeleton.name + '_render', motion_parent, mesh=mesh_index, skin=skin)
        hashes = {name_hash(node.name): i for i, node in enumerate(skeleton.nodes)}
        clips = []
        for animation in asset['animations']:
            if animation_filter and animation.name not in animation_filter:
                continue
            samplers, channels = [], []
            target_tracks = []
            for track_index, key in enumerate(animation.node_hashes):
                if key not in hashes:
                    continue
                target = joints[hashes[key]]
                target_tracks.append((target, animation.tracks[track_index]))
            if animation.root_motion_track:
                target_tracks.append((motion_parent, animation.root_motion_track))
            for target, track in target_tracks:
                nr, nt, ir, it = map(int, track)
                for kind, times, values in [('rotation', animation.rotation_times[ir:ir + nr], animation.rotations[ir:ir + nr]), ('translation', animation.translation_times[it:it + nt], animation.translations[it:it + nt])]:
                    if not len(times):
                        continue
                    keep = np.r_[np.diff(times) > 0, True]
                    times, values = times[keep], values[keep]
                    native_times = times.tolist()
                    if kind == 'rotation':
                        times, values = sample_rotations(times, values)
                    sampler = len(samplers)
                    samplers.append(dict(input=self.accessor(times), output=self.accessor(values), interpolation='LINEAR', extras=dict(native_key_times=native_times, rotation_sampling_fps=60 if kind == 'rotation' else None)))
                    channels.append(dict(sampler=sampler, target=dict(node=target, path=kind)))
            if channels:
                self.doc['animations'].append(dict(name=animation.name, channels=channels, samplers=samplers, extras=dict(native_duration=animation.duration, native_node_hashes=[f'{h:08x}' for h in animation.node_hashes], source_interpolation_unverified=True)))
                clips.append(animation.name)
        sockets = {}
        for marker in asset.get('markers', []):
            transform = np.eye(4)
            transform[:3, :3] = np.array(marker['basis']).T
            transform[:3, 3] = marker['translation']
            owner = joints[marker['parent']] if marker['parent'] < len(joints) else wrapper
            sockets[marker['name'].lower()] = self.node('socket:' + marker['name'], owner, matrix=transform.T.reshape(-1).tolist(), extras=dict(native_marker=marker))
        self.assets.append(dict(name=skeleton.name, namespace=namespace, nodes=len(joints), triangles=len(mesh.triangles), animations=clips, sockets=list(sockets)))
        return wrapper, joints, sockets

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.data:
            self.doc['buffers'] = [dict(uri=self.path.with_suffix('.bin').name, byteLength=len(self.data))]
        self.doc['extras'] = dict(asset_inventory=self.assets, unresolved=self.issues, source_basis='native X/Y/Z under a 180-degree X rotation for glTF', material_shader_approximation=True)
        self.path.with_suffix('.bin').write_bytes(self.data)
        document = {key: value for key, value in self.doc.items() if not isinstance(value, list) or value}
        self.path.write_text(json.dumps(document, indent=2), encoding='utf-8')


def load_assets(root, texture_root, namespaces):
    assets = {}
    material_tables = {}
    texture_index = defaultdict(list)
    report = dict(models=[], failed_models=[], animations=[], failed_animations=[], failed_material_tables=[], failed_marker_tables=[])
    for ns in namespaces:
        archive = root / ns
        materials = {}
        for p in (archive / 'MARE_chunk').glob('*.dat'):
            try:
                materials.update(read_materials(p.read_bytes()))
            except Exception as error:
                report['failed_material_tables'].append(dict(source=str(p), reason=str(error)))
        material_tables[ns] = materials
        for p in archive.rglob('*'):
            if p.suffix.lower() not in ('.tga', '.dds', '.bmp'):
                continue
            rel = p.relative_to(archive)
            if 'graphics' not in rel.parts:
                continue
            # Native material texture ID: lowercase /path without graphics or suffix.
            key = name_hash('/' + rel.with_suffix('').as_posix().removeprefix('graphics/'))
            staged = rel if p.suffix.lower() == '.dds' else Path(str(rel) + '.dds')
            texture_index[key].append((ns, (texture_root / ns / (str(staged) + '.png')).resolve()))
        animations = []
        for p in sorted((archive / 'HCAN_chunk').glob('*.dat')):
            try:
                anim = read_animation(p.read_bytes())
                animations.append(anim)
                report['animations'].append(dict(namespace=ns, source=str(p), name=anim.name, duration=anim.duration, nodes=len(anim.tracks), rotation_keys=len(anim.rotations), translation_keys=len(anim.translations)))
            except Exception as error:
                report['failed_animations'].append(dict(source=str(p), reason=str(error)))
        markers = {}
        for p in (archive / 'HMPT_chunk').glob('*.dat'):
            try:
                owner, sockets = read_markers(p.read_bytes())
                markers[owner.lower()] = sockets
            except Exception as error:
                report['failed_marker_tables'].append(dict(source=str(p), reason=str(error)))
        for p in sorted((archive / 'HSKN_chunk').glob('*.dat')):
            try:
                skeleton = read_skeleton(p.read_bytes())
                resource = archive / skeleton.name.lower()
                if not resource.exists():
                    raise ValueError('Matching named render resource is absent')
                mesh = read_render_mesh(resource.read_bytes())
                validation = validate_render_binding(mesh, skeleton)
                hashes = {name_hash(n.name) for n in skeleton.nodes}
                bound = [a for a in animations if hashes.issubset(set(a.node_hashes))]
                asset = dict(namespace=ns, skeleton=skeleton, mesh=mesh, source=str(p.resolve()), render_source=str(resource.resolve()), animations=bound, markers=markers.get(skeleton.name.lower(), []), validation=validation)
                assets[ns, skeleton.name.lower()] = asset
                report['models'].append(dict(namespace=ns, source=str(p), name=skeleton.name, animations=[a.name for a in bound], validation=validation))
            except Exception as error:
                report['failed_models'].append(dict(source=str(p), reason=str(error)))
    return assets, texture_index, material_tables, report


COCKPIT_ATTACHMENTS = {
    'pedal_l': 'pedal_left', 'pedal_r': 'pedal_right',
    'throttle_l': 'throttle_left', 'throttle_r': 'throttle_right',
    'radar_dial_l': 'radar_dial_l', 'radar_dial_r': 'radar_dial_r',
    'radar_buttons': 'radar_buttons', 'radar_array': 'radar_array',
    'moving_monitor_l': 'monitor_moving_left_gui', 'moving_monitor_r': 'monitor_moving_right_gui',
    'cables_l': 'cables_l', 'cables_mid': 'cables_mid', 'cables_r': 'cables_r',
    'ammo_flap_l': 'ammo_flap_l', 'ammo_flap_r': 'ammo_flap_r_gui',
    'cockpit_l3': 'monitor_l3_gui', 'cockpit_r3': 'monitor_r3_gui',
    'cockpit_l4': 'monitor_l4_gui', 'cockpit_r4': 'monitor_r4_gui',
    'cockpit_l1': 'front_monitors_gui', 'ceiling_lights': 'ceiling_lights',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--texture-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--namespaces', default='all')
    parser.add_argument('--assembly-only', action='store_true')
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    namespaces = sorted(p.name for p in args.input_root.iterdir() if p.is_dir()) if args.namespaces == 'all' else args.namespaces.split(',')
    assets, textures, materials, report = load_assets(args.input_root.resolve(), args.texture_root.resolve(), namespaces)
    if not args.assembly_only:
        for (ns, name), asset in assets.items():
            scene = Scene(args.output_root / ns / (name + '.gltf'), textures, materials)
            root = scene.node('Native basis', rotation=[1, 0, 0, 0])
            scene.add_asset(asset, root)
            scene.save()
    # The native merged vehicle picker has an exact six-node body/exterior/tracks hierarchy.
    if ('chars__actors_pc', 'tank_prototype_01_a_merged_vehiclepicker') not in assets or ('chars__playertanks_pc', 'tank_prototype_01_cockpit') not in assets:
        report['counts'] = dict(models=len(report['models']), failed_models=len(report['failed_models']), animations=len(report['animations']), failed_animations=len(report['failed_animations']))
        (args.output_root / 'scene-validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report['counts'], indent=2), flush=True)
        return 0
    scene = Scene(args.output_root / 'standard_player_tank.gltf', textures, materials)
    root = scene.node('Standard player tank', rotation=[1, 0, 0, 0])
    exterior = assets['chars__actors_pc', 'tank_prototype_01_a_merged_vehiclepicker']
    original = assets['chars__actors_pc', 'tank_prototype_01_a_body']
    source_bounds = exterior['skeleton'].nodes[0].bounds
    target_bounds = original['skeleton'].nodes[0].bounds
    ratios = (target_bounds[:, 1] - target_bounds[:, 0]) / (source_bounds[:, 1] - source_bounds[:, 0])
    ratio = float(np.median(ratios))
    if not np.allclose(ratios, ratio, rtol=0.0001) or not np.allclose(source_bounds * ratio, target_bounds, atol=0.00002):
        raise ValueError('Merged exterior does not align with native gameplay body bounds')
    calibration = np.eye(4)
    calibration[:3, :3] *= ratio
    calibration[:3, 3] = original['skeleton'].nodes[0].translation - exterior['skeleton'].nodes[0].translation * ratio
    exterior_wrapper, exterior_joints, exterior_sockets = scene.add_asset(exterior, root, transform=calibration, label='Exterior (native merged assembly, calibrated to gameplay body)')
    cockpit = assets['chars__playertanks_pc', 'tank_prototype_01_cockpit']
    cockpit_scale = np.eye(4)
    cockpit_scale[:3, :3] /= ratio
    wrapper, joints, sockets = scene.add_asset(cockpit, exterior_sockets['dome'], transform=cockpit_scale, label='Cockpit interior (native Dome attachment)')
    attachment_report = []
    for socket, model in COCKPIT_ATTACHMENTS.items():
        asset = assets.get(('chars__playertanks_pc', model))
        if asset is None or socket not in sockets:
            attachment_report.append(dict(socket=socket, model=model, status='unresolved'))
            continue
        child_wrapper, child_joints, child_sockets = scene.add_asset(asset, sockets[socket])
        screen_name = model.removesuffix('_gui') + '_guiscreen'
        screen = assets.get(('chars__playertanks_pc', screen_name))
        if screen and 'screenmarker' in child_sockets:
            scene.add_asset(screen, child_sockets['screenmarker'])
        attachment_report.append(dict(socket=socket, model=model, status='native_socket_alias', pairing_verified_in_runtime=False))
    for name in ['cockpittop_screen', 'cockpithl_screen', 'cockpithr_screen', 'cockpitshield_screen', 'cockpitdialogue_screen']:
        screen = assets.get(('chars__playertanks_pc', name))
        if screen:
            scene.add_asset(screen, wrapper, label=screen['skeleton'].name + ' (native local placement)')
    scene.save()
    report['assembly'] = dict(file=str(scene.path.resolve()), attachments=attachment_report, exterior_calibration=dict(scale=ratio, translation=calibration[:3, 3].tolist(), evidence='Three-axis bounds of merged named body match gameplay body at a single uniform scale'), native_exterior_nodes=[dict(name=n.name, parent=n.parent, translation=n.translation.tolist(), rotation=n.rotation.tolist()) for n in exterior['skeleton'].nodes], limitations=['Socket transforms and internal parent tables are native; component aliases and exterior/interior origin alignment require runtime visual comparison.', 'Base/normal/emissive textures resolve by native hashes. Game shaders, separate opacity maps and live GUI render targets require BZR equivalents.', 'HCAN key values and times are retained; LINEAR/spherical interpolation has not been compared to runtime.'])
    report['counts'] = dict(models=len(report['models']), failed_models=len(report['failed_models']), animations=len(report['animations']), failed_animations=len(report['failed_animations']))
    (args.output_root / 'scene-validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['counts'], indent=2), flush=True)
    print('Assembly:', scene.path, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
