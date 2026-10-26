# Porting the Gold assets to Battlezone 98 Redux

## Recommended route

Start with one complete player tank as a new offline prototype: exterior, cockpit, controls, three hardpoint housings and one deliberate weapon loadout. Keep the native Gold rigs as the reference. Add a small BZR gameplay control hierarchy for body, aiming, eye and hardpoints; retain the detailed visual rigs beneath those controls. Export the world and first-person models separately. Qualify this tank before applying the conversion to the rest of the library.

Use two explicit compatibility profiles rather than promising the same behavior everywhere:

| Profile | Best use | Animation and weapon behavior | Tradeoff |
| --- | --- | --- | --- |
| Stock-compatible BZR | Props, structures, enemies and a first drivable tank | VDF/SDF control parts, native class behavior and supported legacy animation slots | Gold clips remain archival unless deliberately retargeted to a supported BZR mechanism; use fixed visible loadout variants first |
| Detailed BZR + EXU | Full cockpit controls, complex weapon assemblies and authored visual clips | BZR owns gameplay pivots; EXU addresses world/cockpit Ogre entities and explicitly seeks detail clips | Requires a capability-checked controller and runtime testing; arbitrary vehicle clips do not advance just because they exist in a skeleton |

OpenShim's authored shot/recoil path is an optional later integration. Its inspected documentation describes a bounded GOG/SP opt-in trial, with class/build restrictions and incomplete live qualification. Do not make it the baseline dependency for every port, especially launchers, beams and charge weapons.

## Prepared source and preflight

Generate the handoff from the extractor repository:

```powershell
python tools/prepare-bzr-port.py --dump-root work/asset_dump_2026-10-24
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' --background --python-exit-code 1 --python tools/prepare-bzr-port.py -- --dump-root work/asset_dump_2026-10-24 --build-blender-source
```

The `bzr_port/` folder contains:

- `player_tank_complete_source.gltf` and the optional checked `.blend`: one complete source hierarchy extracted from the tank inspection assembly. Only its gallery translation is removed; socket matrices, calibration, native rigs and candidate clips survive. It uses the explicit example loadout, not a certified stock Gold entity configuration.
- `complete_tank_manifest.json` and `complete_tank_node_map.csv`: source indices/names, proposed unique eight-character aliases, material recipes and deliberately unassigned animation drivers.
- `player_tank_manifest.json`: the base tank/cockpit reference; `exterior_weapons_manifest.json`: the 15-example weapon/mount gallery. Do not combine the entire gallery into one runtime vehicle.
- `resource_readiness.csv`: all 1,401 native resources, geometry counts, skinning risks and suggested starting routes. Categories and routes are preparation hints, not decoded entity classes.
- `port_readiness.json`: aggregate findings and exact local tool revisions/working-tree status. Dependency snapshots are provenance, not a declaration that every checked-out branch is a release.

The isolated complete example has **38 mesh resources, 102,547 vertices, 84,900 triangles, 27 materials and 97 clip instances (77 distinct names)**. That includes cockpit and installed weapons. Split runtime world/cockpit visibility and author LODs before population-scale testing; do not treat this hero assembly as the budget for every distant unit.

The inspected toolkit collector keeps the top **three** influences per vertex. Across the source library, **seven resources / 2,707 vertices** exceed that policy; the largest dropped weight would be about 0.243. The affected resources are Data_store, Radar_Relay, MRig_FX_PHex, MRig_FX_Smoke_Poly01, City_Component_MiningSpider01, Hangar_Robot_Crane and Repair_Droid_Window_Cleaner. The complete player tank and the weapon examples use at most three and need no weight reduction for that policy. For affected assets, fix/extend the shared exporter upstream or explicitly reduce weights and measure deformation error across relevant clips. Never silently accept its truncation.

The preflight also flags **39 resources using unresolved material IDs**. Keep them visibly diagnostic until resolved; a neutral placeholder is not a finished material.

## Geometry, coordinates and hierarchy

1. Freeze a source reference and record input hashes. Use the new staged assembly or the categorized `.blend` libraries, not OBJ previews. Work on duplicates so native source rigs remain an independent reference.
2. Choose BZR units against a measured chassis and gameplay footprint. The source vehicle-picker scale of 32 and attached inverse 1/32 are internal source calibration, not a BZR metres conversion. Preserve their product when consolidating transforms.
3. Match coordinate conventions explicitly. BZR review tools use game +Y up/+Z forward, mapped into Blender as `(x,-z,y)`. The extractor already has its own basis wrapper. Apply one recorded source-to-target transform to vertices, normals, bones, sockets, eye, collision and all animation translations; do not rotate only the mesh or apply a second axis conversion blindly.
4. Add BZR control pivots with assigned part classes: body, turret yaw, gun pitch, POV, weapon hardpoints, collision and optional deploy controls. Use a verified target class/template. Native Gold sockets supply placement evidence; their names do not assign BZR class behavior.
5. Put each detailed native rig below the appropriate control pivot. Consolidate rigs only with a converter that preserves weights, bind poses and sampled motion. Keep blended cables blended. Separate rigid components where useful for palette/submesh budgets, not by flattening every hierarchy into unrelated pieces.
6. Make every engine-driven Ogre bone match its VDF/GEO control name, pivot and basis. Use explicit unique eight-character legacy aliases. The generated alias CSV is a proposal for visual nodes; it has no assigned VDF classes. Name truncation alone can create collisions and accidental engine semantics.
7. Export through the existing toolkit serializers. In Campaign Reimagined, access them via `Tools/bzmodellibs.py`. Do not write another SDF/VDF/GEO/ANIM/Ogre codec. Verify the emitted version/buffer layout with the chosen toolkit and target renderer rather than requiring an arbitrary external Ogre converter version.

Source rigs with many bones still need target palette/batch and performance checks. Count alone does not establish engine support. Keep driver controls separate from cosmetic detail so BZR's per-frame GEO pose writes do not overwrite the same transform channels as an Ogre clip.

## Complete cockpits

Export a dedicated first-person mesh/skeleton and preserve the actual eye relationship. Check the selected cockpit resource name against the target model/porter suffix rules; stock `_c`, `_fp` and `_cockpit` conventions are not interchangeable guesses.

The name's fourth character has runtime meaning: `1` parts are world geometry and `2` parts are cockpit geometry. Author the necessary cockpit shell/helper bones beneath the correct driven parent, following a verified template. A copied skeleton with unmatched VDF names can stay frozen or inherit the wrong pitch/deploy transform. The actual camera follows the derived POV transform plus view pitch; a plausible bind-pose render is insufficient.

Retain monitors, knobs, muzzle devices, controls and their source UV/material assignments. Blank native render targets do not contain a recoverable functioning radar/health display. Rebuild display behavior using BZR HUD/EXU material or texture controls, using dedicated material instances where needed. Capability-check cockpit entity access because the entity can appear, disappear or be recreated on view/vehicle changes.

Verify first-person forward, down, side and deploy/aim views from the same recorded eye; check occlusion before moving exterior guns. Keep the complete source assembly as the alignment reference even when only a subset of exterior surfaces belongs in the first-person runtime mesh.

## Weapons that swap on the tank

The stock-compatible weapon/effects pilot is now built by `tools/build-gold-weapons-mod.py`; see [GOLD_WEAPONS_MOD.md](GOLD_WEAPONS_MOD.md). It includes 22 authored behavior profiles covering all 19 native English weapon descriptions, 111 tier-name aliases, recovered projectile models/cards/audio, particle graphs, pickups and a Grizzly loadout override. `audit-gold-ordnance.py` checks exported mesh axes/origins against stock and ISDFC references and native front/rear landmarks. This pilot does not yet implement dynamic external gun swaps or Gold detail controllers.

Begin with fixed visible loadout variants using complete gun collections and native mount sockets. This provides a useful stock-compatible port without assuming a generic runtime mesh-attachment API.

For dynamic swaps, first verify a target runtime mechanism for replacement/attachment or subentity visibility and material control. `GetWeaponClass(handle, slot)` can identify a weapon definition; it does not itself attach a model, resolve a source loadout, detect an accepted shot or provide reload/charge state. Maintain an explicit weapon-definition-to-visible-assembly map per hardpoint. Reconcile state after pickup, AI changes, boarding, load, destruction and respawn. Avoid polling a fire input or ammo delta as if it were a verified accepted-shot event.

Preserve the guided launcher's separate front/rear rings and muzzle-controller nodes. Visible muzzle transforms must match the VDF firing hardpoint under yaw, pitch, deploy and recoil. Decorative recoil must not move gameplay aiming/collision. Never move an entire hull or turret as a shortcut to barrel recoil.

Use a controller state table for stowed, deploying, ready, firing, reloading, charging and lock-on only where the target exposes reliable state. Native Gold clip names are source vocabulary, not BZR animation slot IDs. The generated manifests intentionally leave trigger assignments unassigned. Missing event information is a conversion task, not an excuse to discard the source clips.

## Animation conversion

For BZR-native controls, retarget to the target class's supported VDF animation slots and test transitions. Use exact part names and supported pivot behavior; simply exporting arbitrary `Autocannon_Fire` into an Ogre skeleton does not make the vehicle play it.

For retained detail clips, EXU's `exu.animation` API has separate world and cockpit targets. Its inspected documentation says vehicle animation time needs explicit ownership. Use `Has`/`List`/capability checks, enable the intended states, and `Seek` using absolute pause-aware simulation time. Example API shape:

```lua
local world = exu.animation.Target(craft)
local cockpit = exu.animation.TargetCockpit(craft)
-- Controller must supply verified state/time and an exported clip name.
for _, target in ipairs({world, cockpit}) do
    if exu.animation.Has(target, clip) then
        exu.animation.SetEnabled(target, clip, true)
        exu.animation.Seek(target, clip, clipTime)
    end
end
```

This is a control sketch, not a drop-in mission script: load EXU, validate the exported names, choose one clock owner, resolve targets afresh, manage competing states/weights, and stop/reset states on transitions. Keep world/cockpit time synchronized where both show the same moving gun. Preserve independent controllers for unrelated cockpit controls. Do not allow a native VDF pose writer and a detail animation to fight over a shared control bone.

The extraction checks establish 757 decoded HCAN chunks, complete imported bindings and selected sampled motion, not source-game state machines or target-game runtime fidelity. Repeat vertex comparisons after consolidation, aliasing, bind-pose conversion and export. Then test native reload/deploy/charge transitions in BZR; do not certify them from a Blender rest pose.

## Materials, textures, audio and effects

| Source feature | BZR preparation |
| --- | --- |
| Albedo + serialized tint | Bake or apply the traced tint exactly once. Match the existing Blender node result before touching gamma or faction recolors. Preserve original texture separately. |
| Packed normal RG | Reconstruct positive Z; generate a target normal texture and tangent basis together. The derived OpenGL preview's green inversion is for Blender; confirm target handedness with a diagnostic render. |
| Albedo alpha roughness | Extract as a separate source map; make a documented approximation for BZR's specular/gloss controls. Do not reuse it as opacity. |
| Dedicated opacity / glass | Separate opaque, masked and blended materials. Check depth, cockpit compositing and sort behavior. Source double-sided preview flags are not a fix for missing geometry. |
| Self illumination | Preserve `tinted albedo * native emission RGB * mask.red * gain`. Keep high gain separate from LDR texture storage where the target material allows it; do not clip it silently or multiply it twice. |
| Metallic / reflection / response LUT | Preserve source parameters; qualify a BZBase approximation first. Full metallic response, native LUT BRDF, OIT/glass and entity overrides are not reconstructed. |
| Height/parallax | Keep source height; use a documented bump/target approximation unless the selected shader supports the required effect. |
| Dynamic monitors | Port surfaces/UVs now, supply runtime displays separately. Do not bake invented UI into the recovered source textures. |
| DXBC shader resources | Archival/reference inputs. They are not a drop-in shader package for BZR's renderer/material pipeline. |
| Audio | Start from the verified PCM16 copies, then configure BZR weapon/ordnance/vehicle sound references, loop/cadence and events. Do not infer exact sync from filenames. |
| Particles/muzzle effects | Rebuild with supported BZR effects or a qualified presentation adapter; copying recovered chunks does not instantiate native effects. |

Use BZBase for the initial world-material compatibility path and the verified cockpit pass family for first-person materials; validate inherited texture aliases and fallback techniques against the actual target. Keep the Gold material namespace separate from stock resources. Start with reversible PNG/known-good DDS previews, then choose compression/mips per target usage and platform; do not recompress every original texture in place.

If extending Enhanced shaders, their implementation belongs in OpenShim. Campaign Reimagined owns selecting materials and art parameters. Its `Docs/ENHANCED_SHADER_OWNERSHIP.md` lists the cross-repository program-reference/shader checks. A custom Gold PBR shader is a later fidelity task after the BZBase prototype works.

## Conversion order and acceptance

1. **Complete player tank pilot:** assign BZR scale/control pivots/collision/POV, one fixed loadout and separate cockpit. Round-trip binaries through existing readers; compare vertices, bind world transforms, hardpoints, normals, UVs, material references and sampled poses. Render the reloaded files.
2. **Weapon family pilot:** autocannon plus railgun first, then guided rings and hardpoint deployment. Compare actual muzzle/reticle alignment and independent world/cockpit motion. Add dynamic swaps only after a supported runtime mechanism and lifecycle test exist.
3. **Static prop, then native-driven turret:** establish the simple SDF/material path before adding yaw/pitch/deploy behavior. Test collision separately from detailed visual geometry.
4. **Enemy tanks and drones:** assign deliberate BZR gameplay classes and controls. Source silhouette/category does not determine flight or AI behavior. Author LOD/collision/damage variants and measure populated-scene cost.
5. **Runtime qualification:** under the target repository's serialized test harness, check spawn/despawn, boarding, view switches, save/load, pause, weapon replacement, deploy/aim/recoil, restart and missing dependencies. Add multiplayer and renderer/platform matrices only for the scope being shipped.

`prepare-bzr-port.py` changes no installed BZR files and launches no game. The separate Grizzly/weapon builders and installer now produce and copy dedicated, disabled prototype mod folders; see their guides for scope and activation. Native testing and release use the destination repository's normal workflows.

## Evidence and reusable tools

The preparation manifest records local revisions. Primary references inspected on 2026-10-25:

- [Battlezone Blender Toolkit](https://github.com/GrizzlyOne95/BZ98RBlenderToolKit): local `bz98tools/bzrmodelporter/` serializers; `ogrefast/ogre_exporter.py` influence collection; `docs/ORGANIC_REDUX_SKIN_WALKTHROUGH.md` runtime control guidance; `docs/REDUX_COCKPITS_AND_WALKERS.md` cockpit findings. Local HEAD: `65297da3c86d71d17f09193d811a59e85dd5cf0d`. The local checkout includes Blender 5.2 work beyond the browsed older README; use the recorded local implementation, not the web page's older version claim.
- EXU local `docs/ANIMATION_API.md` and `src/Game/AnimationApi.cpp`: explicit world/cockpit targets, inventory, capability checks and time ownership. Local HEAD: `f24b7204ab87f7dc68f4c2b4a811fed54dbffe46`.
- OpenShim local `Docs/ODF_RECOIL_TRIAL.md`, `ODF_MUZZLE_FLASH.md`, and `WEAPON_PRESENTATION_IMPLEMENTATION.md`: opt-in authored presentation, current class/build scope and unresolved qualification. Its exact revision is in the generated readiness manifest.
- Campaign Reimagined local `Tools/bzmodellibs.py`, `Docs/STOCK_ASSET_CONTENT_FIXES.md` and `Docs/ENHANCED_SHADER_OWNERSHIP.md`: shared codecs, verified cockpit hierarchy hazards and shader ownership. Local HEAD: `17cf45d3ab306bc6043acc57111ab4bac6bc16cf`.
- [Ogre exporter skeletal-animation guidance](https://github.com/OGRECave/blender2ogre/blob/master/SkeletalAnimation.md): export requires explicit animation selection/baking; existence in an exported skeleton is separate from runtime animation control.

No sibling repository was edited for this preparation. Resolve missing serializer capabilities upstream rather than embedding new binary codecs in the extractor.
