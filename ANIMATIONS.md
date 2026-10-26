# Native animation extraction and Blender playback

Native HCAN clips survive as glTF animation channels and Blender actions. Skeleton parents, joint-local skinning, native node hashes, separate root motion and socket transforms are retained. The original HCAN chunks remain in the raw dump.

The validated local installation has 757 decoded HCAN resources with no decode failures. This count includes repeated resources. All 1,401 HSKN resources export, including 1,288 with render geometry and 113 node-only hierarchies. Clips bind to resources by native node hashes; exported candidates are not a decoded gameplay state machine.

## What has been checked

| Check | Coverage | Meaning |
| --- | --- | --- |
| Native HCAN parsing | All 757 discovered chunks | Supported key arrays, names, times and node hashes decode |
| Imported animation bindings | 23,824 channels across 1,207 clip instances in 39 libraries | Every exported channel has its full component-curve binding on an owning Blender action slot |
| Saved/reopened geometry | All 39 Blender libraries | Evaluated rest vertices match independently computed interchange skinning within recorded coordinate tolerance |
| Exterior weapon motion | 39 selected clips, 117 start/mid/end samples | Fire, reload, stow, charge, Gatling loop, launcher rings/lock-on, partial reload and hardpoint deployment match interchange skinning |
| Assembled cockpit motion | Throttle, moving monitor/screen and blended cables, 9 samples | Animated parents and attached parts remain aligned |

Motion checks compare Blender to independent glTF skinning. They do not establish exact running-game interpolation, loadout selection, coordinated gameplay states, muzzle flash/sound timing or BZR compatibility. Clips not included in sampled validation remain preserved but untested in motion. Each `.validation.json` is the authority for that file's tested clips and errors.

The 1,207 exported clip instances include repeated bindings to component/variant resources and inspection assemblies; they are not 1,207 different source animations. The binding audit checks channel presence, while the sampled motion checks also compare evaluated vertex positions.

## List clips and owners

Run from the extractor repository in PowerShell, with Blender 5.2 installed. Keep the complete dump folder together for shared textures and source glTF buffers.

```powershell
$blenderExe = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
$dumpRoot = 'work/asset_dump_2026-10-24'
& $blenderExe --background "$dumpRoot/blender_library/player_weapon_assemblies.blend" --python-exit-code 1 --python tools/inspect-asura-animations.py -- --list --report "$dumpRoot/reports/weapon_animation_inventory.json"
```

The report lists exact clip names, imported owner objects and resource collection IDs. `imported: false` identifies a source clip without an imported action binding and needs investigation. Add `--asset` with an exact collection name or resource ID to inspect one assembly. Repeated clip instances can drive several rigs, including a root-motion object; select all owners belonging to that clip.

Add `--audit` to check every source channel has all expected component curves on an imported action slot. Audit the entire saved library collection without opening a file first:

```powershell
& $blenderExe --background --python-exit-code 1 --python tools/inspect-asura-animations.py -- --audit-library $dumpRoot
```

This writes `reports/blender_animation_bindings.json`, returns an error on missing bindings and leaves the libraries unchanged. New library builds also enforce the audit after saving and reopening each file.

## Create a playable preview

```powershell
& $blenderExe --background "$dumpRoot/blender_library/player_weapon_assemblies.blend" --python-exit-code 1 --python tools/inspect-asura-animations.py -- --clip Railgun_Fire --camera-frame 5 --validate --output "$dumpRoot/animation_previews/railgun_fire.blend" --report "$dumpRoot/reports/railgun_playback.json"
& $blenderExe --background "$dumpRoot/blender_library/player_weapon_assemblies.blend" --python-exit-code 1 --python tools/inspect-asura-animations.py -- --clip Weapon_Hardpoint02_Top_Deploying --camera-frame 15 --validate --output "$dumpRoot/animation_previews/mount_deploy.blend" --report "$dumpRoot/reports/mount_playback.json"
```

Open the output and press Space to play. The helper restores the native inspection pose, mutes competing NLA tracks, assigns the requested actions/slots to all matching owners, sets the playback range and fixes the selected inspection camera. Default selection activates every instance of the exact clip. Use `--asset` to restrict playback to one collection. `--validate` always checks all instances of the selected clip, independent of that filter. Other moving components stay in their native inspection pose; coordinated deploy/fire states require separate gameplay logic.

The railgun and top-mount example preview files were saved and reopened successfully, checking active owner actions, playback ranges, camera-marker removal and shared image paths. These ready-to-play examples are under `animation_previews/` in the validated dump.

`--frame` selects an initial preview frame, including fractional frames. `--output` must differ from the input path; the source library is never saved over. Shared texture references are recalculated for the new output directory. If opening a renamed preview, use `--source` to identify its original library glTF. Frame numbers use the saved library FPS; do not change FPS to retime a native clip accidentally.

Without the helper, select an owning rig, open the NLA Editor, and unmute only the desired clip. Root motion may be on a separate object, so enable its corresponding track too. The original library's numbered frames select inspection cameras, not animation states. Use a separate copy and remove camera markers before playback if they would switch the view.

## Inventories and rebuilding

```powershell
python tools/build-asura-asset-library.py --dump-root $dumpRoot --index-only
```

This refreshes the generated README, `player_weapons.csv` and `animation_inventory.csv` from existing catalogs and validation reports without rebuilding geometry. The animation CSV records each exported clip instance, duration, target-node count, root-motion channel count and available validation samples. Repeated rows can refer to different resource instances. A zero sample count means no sampled motion check is recorded; it does not mean the clip is missing.

Use `--prepare` to regenerate catalogs after extractor changes. Run the builder through Blender with `--build` to rebuild the files; `--categories player_weapons,player_weapon_assemblies` limits rebuilding to those categories. Rebuilds refresh the indexes after writing validation results. The new playback helper is an external Blender Python script; it is not an extractor UI playback control or a bundled Blender runtime.

## Preservation limits

Native key values/times are decoded. The glTF exporter coalesces duplicate timestamps and adds 60 Hz spherical quaternion samples for Blender interpolation. Original chunks remain available for exact source reconstruction. Static calibration and root-motion TRS use separate nodes so mounted weapon animation retains scale. Repeated joint influences are coalesced algebraically for Blender's vertex groups, with original tuples retained in extras.

Animation event payloads, entity state transitions, looping/blending rules, procedural aiming, recoil triggers, synchronized muzzle effects/audio and dynamic cockpit displays still need runtime interpretation and BZR equivalents. Material/texture wiring survives animation but does not reconstruct the entire native shader. See [SCENE_FORMATS.md](SCENE_FORMATS.md) and [RESOURCE_FORMATS.md](RESOURCE_FORMATS.md).
