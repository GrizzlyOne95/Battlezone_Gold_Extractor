"""List or preview native clips in a saved Asura Blender library.

blender --background library.blend --python-exit-code 1 --python this.py -- --list
blender --background library.blend --python-exit-code 1 --python this.py -- \
    --clip Railgun_Fire --validate --output railgun-preview.blend

The source file is never saved over. Run with the original dump tree present.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys


def matches_clip(track_name, clip_name):
    """Accept only an exact native name or Blender's numeric duplicate suffix."""
    return track_name == clip_name or (
        track_name.startswith(clip_name + '.') and track_name[len(clip_name) + 1:].isdigit())


def source_for_blend(path):
    path = Path(path).resolve()
    if path.parent.name == 'blender_library':
        return path.parent / 'gltf' / (path.stem + '.gltf')
    if path.name == 'standard_player_tank_materials.blend':
        return path.with_name('standard_player_tank.gltf')
    raise ValueError('Cannot infer source glTF; supply --source PATH for a renamed preview.')


def audit_bindings(document, objects):
    """Check each exported channel has all component curves on an owning slot.

    This checks binding coverage, not key values or evaluated motion.
    """
    targets = {}
    for obj in objects:
        if '_asura_node_index' in obj:
            targets.setdefault(obj['_asura_node_index'], []).append((obj, None))
        if obj.type == 'ARMATURE':
            for bone in obj.pose.bones:
                if '_asura_node_index' in bone.bone:
                    targets.setdefault(bone.bone['_asura_node_index'], []).append((obj, bone))
    cache = {}
    def curves(obj, name):
        key = (obj.name, name)
        if key not in cache:
            found = set();ad = obj.animation_data
            if ad:
                for track in ad.nla_tracks:
                    if not matches_clip(track.name, name):continue
                    for strip in track.strips:
                        if not strip.action:continue
                        for layer in strip.action.layers:
                            for action_strip in layer.strips:
                                bag = action_strip.channelbag(strip.action_slot)
                                if bag:found.update((c.data_path, c.array_index) for c in bag.fcurves)
            cache[key] = found
        return cache[key]
    missing = [];checked = 0
    properties = dict(rotation=('rotation_quaternion', 4), translation=('location', 3), scale=('scale', 3))
    for animation in document.get('animations', []):
        name = animation['name']
        for channel in animation['channels']:
            target = channel['target'];prop, width = properties[target['path']];bound = False
            for obj, bone in targets.get(target['node'], []):
                path = bone.path_from_id(prop) if bone else prop
                if all((path, i) in curves(obj, name) for i in range(width)):
                    bound = True;break
            checked += 1
            if not bound:missing.append(dict(clip=name, node=target['node'], path=target['path']))
    return dict(clip_instances=len(document.get('animations', [])), distinct_clips=len({a['name'] for a in document.get('animations', [])}),
                channels_checked=checked, missing_channels=missing, passed=not missing)


def main():
    import bpy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='List native clips and imported owners')
    parser.add_argument('--audit', action='store_true', help='Check all exported channels have imported component curves')
    parser.add_argument('--audit-library', type=Path, help='Audit all saved libraries under this dump root')
    parser.add_argument('--clip', help='Exact native clip name')
    parser.add_argument('--asset', help='Exact native resource ID or collection name; default: all instances')
    parser.add_argument('--source', type=Path, help='Source glTF, normally inferred from the open blend')
    parser.add_argument('--frame', type=float, default=0, help='Preview frame, starting at zero')
    parser.add_argument('--camera-frame', type=int, help='Use the library inspection camera at this frame')
    parser.add_argument('--validate', action='store_true', help='Compare selected clip at start/mid/end on all instances')
    parser.add_argument('--output', type=Path, help='Save a separate playable blend; source overwrite is refused')
    parser.add_argument('--report', type=Path, help='Write inventory and optional motion results as JSON')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    if not args.list and not args.clip and not args.audit and not args.audit_library:parser.error('Choose --list, --audit, --audit-library PATH or --clip NAME')
    if args.audit_library:
        if args.clip or args.asset or args.output or args.validate or args.source or args.camera_frame is not None:
            parser.error('--audit-library accepts only --report')
        base = args.audit_library.resolve();manifest = json.loads((base/'blender_library/manifest.json').read_text())
        results = []
        for batch in manifest['batches']:
            bpy.ops.wm.open_mainfile(filepath=str(base/batch['blend']))
            document = json.loads((base/batch['gltf']).read_text())
            result = dict(blend=batch['blend'], **audit_bindings(document, list(bpy.data.objects)))
            results.append(result);print('ANIMATION_BINDING_AUDIT', result['blend'], result['channels_checked'], result['passed'], flush=True)
        report = dict(libraries=len(results), clip_instances=sum(r['clip_instances'] for r in results),
                      channels_checked=sum(r['channels_checked'] for r in results),
                      passed=all(r['passed'] for r in results), results=results,
                      scope='imported component-curve bindings; key values, sampled motion and game runtime are separate checks')
        output = args.report or base/'reports/blender_animation_bindings.json'
        output.parent.mkdir(parents=True, exist_ok=True);output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('ANIMATION_LIBRARY_AUDIT', json.dumps({k:v for k,v in report.items() if k!='results'}), flush=True)
        if not report['passed']:raise ValueError('Animation binding audit failed; see report')
        return
    if args.validate and not args.clip:parser.error('--validate requires --clip')
    original = Path(bpy.data.filepath).resolve()
    if not bpy.data.filepath:raise ValueError('Open a generated Blender library first')
    if args.output and args.output.resolve() == original:raise ValueError('Choose a separate preview output; source overwrite is refused')
    source = (args.source or source_for_blend(original)).resolve()
    document = json.loads(source.read_text())
    names = sorted({a['name'] for a in document.get('animations', [])})
    collections = [c for c in bpy.data.collections if c.get('native_resource_id')]
    selected = None
    if args.asset:
        candidates = [c for c in collections if args.asset in (c.name, c['native_resource_id'])]
        if len(candidates) != 1:raise ValueError('Asset must match exactly one resource ID or collection name')
        selected = candidates[0]
    objects = list(selected.objects) if selected else list(bpy.data.objects)
    def bindings(name):
        return [(obj, track, strip) for obj in objects if obj.animation_data
                for track in obj.animation_data.nla_tracks if matches_clip(track.name, name)
                for strip in track.strips if strip.action]
    inventory = []
    for name in names:
        found = bindings(name)
        if not found and selected:continue
        inventory.append(dict(clip=name, owners=sorted({o.name for o, _, _ in found}),
                              imported=bool(found), action_slots=len(found)))
    report = dict(source_blend=str(original), source_gltf=str(source),
                  assets=[dict(id=c['native_resource_id'], collection=c.name) for c in collections],
                  clips=inventory, motion_validation=[])
    if args.audit:
        if selected:parser.error('--audit checks the entire file; omit --asset')
        report['binding_audit'] = audit_bindings(document, list(bpy.data.objects))
    if args.clip:
        if args.clip not in names:raise ValueError('Unknown native clip: ' + args.clip)
        assignments = bindings(args.clip)
        if not assignments:raise ValueError('Clip has no imported action on the selected asset')
        spec = importlib.util.spec_from_file_location('asura_motion_helper', Path(__file__).with_name('build-player-tank-materials.py'))
        helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        for obj in bpy.data.objects:
            if obj.animation_data:
                obj.animation_data.action = None
                for track in obj.animation_data.nla_tracks:track.mute = True
        helper.restore_native_socket_transforms(document)
        if args.validate:
            report['motion_validation'] = helper.validate_native_motion(document, source, [args.clip])
        scene = bpy.context.scene
        if args.camera_frame is not None:
            cameras = [m.camera for m in scene.timeline_markers if m.frame == args.camera_frame and m.camera]
            if len(cameras) != 1:raise ValueError('No unique inspection camera at --camera-frame')
            scene.camera = cameras[0]
        # Gallery camera markers must not switch views during animation playback.
        for marker in list(scene.timeline_markers):
            if marker.camera:scene.timeline_markers.remove(marker)
        maximum = 0
        for obj, _, strip in assignments:
            ad = obj.animation_data;ad.action = strip.action
            if getattr(strip, 'action_slot', None):ad.action_slot = strip.action_slot
            maximum = max(maximum, float(strip.action_frame_end))
        scene.frame_start = 0;scene.frame_end = max(1, math.ceil(maximum))
        scene.frame_set(math.floor(args.frame), subframe=args.frame - math.floor(args.frame))
        bpy.context.view_layer.update()
        report.update(active_clip=args.clip, active_owners=sorted({o.name for o, _, _ in assignments}),
                      fps=scene.render.fps / scene.render.fps_base, frame_end=scene.frame_end,
                      frame=args.frame, validation_scope='all instances of selected clip; preview asset filter does not restrict validation')
    if args.output:
        if not args.clip:parser.error('--output requires --clip')
        output = args.output.resolve();output.parent.mkdir(parents=True, exist_ok=True)
        # Resolve shared files before changing the blend's directory.
        bpy.ops.file.make_paths_absolute()
        bpy.ops.wm.save_as_mainfile(filepath=str(output), compress=True)
        bpy.ops.file.make_paths_relative()
        bpy.ops.wm.save_as_mainfile(filepath=str(output), compress=True)
        report['preview_blend'] = str(output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('ASURA_ANIMATION_REPORT', json.dumps(report, indent=2), flush=True)
    if args.audit and not report['binding_audit']['passed']:
        raise ValueError('Some exported animation channels have no complete imported curve binding; see report')


if __name__ == '__main__':main()
