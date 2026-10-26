# Gold Grizzly standalone prototype

This is the first BZR conversion pilot: a standalone `mapType = "mod"` package that replaces the **NSDF Grizzly (`avtank`)** world model and first-person cockpit. It uses the complete staged Gold tank, a fixed example loadout and BZR's existing Grizzly control skeleton/VDF.

## Package and activation

The generated package is `work/asset_dump_2026-10-24/bzr_mod/goldgrizzly.zip`. Its `goldgrizzly/` directory is the runtime mod folder. The INI source is [mods/goldgrizzly/goldgrizzly.ini](mods/goldgrizzly/goldgrizzly.ini).

Place that folder under the chosen game's `addon/` directory for a local mod. Enable **Gold Grizzly - Tank Model Prototype** in BZR's Mods menu, above other mods replacing `avtank`. Disable it to restore the lower-priority Grizzly assets. Base-game files and stock ODFs do not need overwriting. This prototype is independent of Campaign Reimagined; override priority matters if both are enabled.

The installation helper copies only the generated package into `addon/goldgrizzly`, verifies its hashes, and writes an installation receipt beside the package. It preserves mod selection, does not edit other mods and never launches BZR. An update refuses to overwrite a destination file changed since its previous receipt. `--content-root mods` selects the older layout explicitly; the helper can recognize the recorded hashes after a user moves its installation from `mods/` to `addon/`.

```powershell
python tools/install-gold-grizzly-mod.py --mod-root work/asset_dump_2026-10-24/bzr_mod/goldgrizzly --game-root 'C:/Program Files (x86)/GOG Galaxy/Games/Battlezone 98 Redux'
```

## What is in this build

- `avtank.mesh` / `avtank.skeleton`: Gold hull, external hardpoint housings and fixed main-gun/autocannon/railgun/artillery display assembly.
- `avtank_cockpit.mesh` / `avtank_cockpit.skeleton`: cabin, cables, controls, monitors and selected external weapons, using a separate first-person material family.
- `avtank.vdf`: stock Grizzly gameplay/collision structure, with the eye and four firing hardpoints repositioned for this visual assembly. Muzzles parent to the existing turret control; source muzzle forward directions are checked against target forward.
- `goldgrizzly.material` and `bzg_*.png`: baked native tint, unpacked normals, documented BZBase specular/emission approximations and dedicated alpha-tested overlay parents.
- `bzgauto.odf` / `bzgauto.wav`: an optional, **unequipped** autocannon approximation described below.

This build uses an explicitly authored eye because the actual Gold runtime POV has not been recovered. Its position is recorded in `build.json`; review cameras render from that exact exported eye at a 64-degree vertical FOV and 16:9 aspect. The source vehicle-picker calibration is preserved before fitting the hull's width to the stock Grizzly's width.

The world hull and top weapon housing are omitted from the first-person copy. The housing blocked the center ray just 0.089 m from the authored eye; it remains on the world model. The gun and real cabin surfaces remain. This is a recorded first-person selection, not a claim that the original Gold runtime hides the same surfaces.

Collapsed triangles are omitted and zero corner normals repaired only in the target copy. `build.json` records resource names and counts. Original extracted geometry is retained unchanged.

## Animation and gameplay limits

The runtime model is a **posed step-1 prototype**. It freezes source ready/deployed poses and rigidly binds the visual groups to BZR body, turret and cockpit controls. It does not ship the detailed Gold fire/reload/charge/controller animations as active BZR behavior. Those complete rigs/clips remain in the source Blender/glTF libraries. The next animation phase needs retargeting or explicit EXU controllers as documented in [BZR_PORTING.md](BZR_PORTING.md).

The Grizzly's default weapon definitions and handling remain stock. Displayed guns are a fixed example loadout and do not dynamically swap when BZR weapons change. Stock collision is retained as a gameplay baseline and does not follow every detail of the taller/wider visual attachments. The prototype needs runtime collision-fit, pitch/aim, muzzle/reticle, visibility, damage, lifecycle and renderer checks before claiming a playable release. Offline readers and renders do not prove native loading or cockpit bone takeover.

The separate [Gold Weapons mod](GOLD_WEAPONS_MOD.md) now supplies a four-weapon `avtank.odf` loadout (main cannon, autocannon, railgun, artillery) plus eleven namespaced profiles and pickups. Enable that mod to change weapon behavior; this model package alone still uses stock definitions. The updated model package assigns GS1 to the railgun muzzle and GM1 to artillery, matching the weapon mod's SPEC/MORT categories. Upgrade both packages together.

Materials use BZBase/BZBaseCockpit approximations. High native emission values can clip into LDR textures; counts are recorded. Reflection/LUT response, source glass/OIT and dynamic screens are not reproduced. The recovered blank display surfaces remain present; they are not functioning radar/health instruments.

### Material import failure found in native testing

The October 25 native test exposed a missing `import * from "BZBase.material"` in the generated script. Ogre could not resolve either the copied masked parents' abstract pass bases or BZBase/BZBaseCockpit inheritance. Its log reported blank Gold world/cockpit techniques, followed by repeated D3D11 errors about missing vertex/fragment shaders while entering the tank. OpenShim also recorded a first-chance access violation during that interval; the material failure is established, but the precise native exception path and complete freeze resolution require a fresh runtime test.

The generator now emits the explicit import for both new builds and cached material reuse. Both tank and projectile validators reject missing imports and unresolved stock material/pass parents. The original logs and nine installed material scripts are preserved under `work/asset_dump_2026-10-24/diagnostics/material_import_failure_20261025/`. This regression check is static inheritance visibility, not native shader compilation.

## Gold weapon research

Run the conservative definition recovery independently:

```powershell
python tools/export-gold-weapon-definitions.py --dump-root work/asset_dump_2026-10-24 --executable 'C:/Program Files (x86)/Steam/steamapps/common/Battlezone/bin/battlezone_dx11.exe'
```

The output under `weapon_definitions/` contains **11 family mappings, 61 decoded model/clip controller records, three raw blueprint files with candidate model-parent references, and 28 fields traced from a native weapon-property reader**. Raw controller/blueprint copies and executable SHA/address evidence are retained locally. Optional schema inspection requires `pefile` and `capstone`; they are research dependencies, not extraction/runtime dependencies.

`weapon_property_schema.json` establishes native fields such as `.RateOfFire`, `.Damage`, `.Range`, `.AreaOfEffect`, `.BulletsPerShot`, cone/burst and state references. Reader fallback constants are defaults for missing properties, not weapon-specific tuning. The compiled BLUE inheritance/property layouts are not decoded fully enough to recover named weapon damage/ROF/speed/ammo values. Those fields remain null in the output. Animation duration is never used as a firing interval.

The best first recreation is the autocannon's basic direct-fire behavior. `bzgauto.odf` derives from the installed stock `gminigun.odf`: `machinegun`, `bullet1`, `shotDelay = 0.20`. Its native Gold autocannon blast is converted to **mono PCM16, 22,050 Hz**. Neither its timing nor ballistics claim Gold balance. The original long blast/tail may overlap at the BZR shot cadence and still needs listening/tuning in game. It is available to future authored loadouts; this mod does not equip it or replace `gminigun`.

Railgun precision shots, mortar-like artillery and homing/unguided launchers have plausible BZR starting classes. Charge, heat, reload magazines, ring motion, laser steering and source-specific effects require additional state/control work. See generated `weapon_definitions/recreation_candidates.csv` and `weapon_definitions/README.md` for the per-family assessment.

## Rebuild and checks

Run from the extractor repository. The builder uses the sibling toolkit serializers and toolbox ZFS reader; it does not implement new BZR binary codecs. Set explicit `--toolkit-root` and `--toolbox-root` paths if needed. Blender 5.2 and FFmpeg are required. Full texture baking uses Blender's image API, so Pillow is not required inside Blender.

```powershell
$blenderExe = 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'
$dumpRoot = 'work/asset_dump_2026-10-24'
$gameRoot = 'C:/Program Files (x86)/GOG Galaxy/Games/Battlezone 98 Redux'
& $blenderExe --background --python-exit-code 1 --python tools/build-gold-grizzly-mod.py -- --dump-root $dumpRoot --game-root $gameRoot
python tools/validate-gold-grizzly-mod.py --mod-root "$dumpRoot/bzr_mod/goldgrizzly" --game-root $gameRoot
& $blenderExe --background --python-exit-code 1 --python tools/review-gold-grizzly-mod.py -- --mod-root "$dumpRoot/bzr_mod/goldgrizzly"
python -m unittest discover -s tests -v
```

For geometry-only iteration, `--reuse-textures` reuses existing bakes after checking the source-scene hash. Remove that flag after material-recipe changes so the bakes and `.material` script are regenerated.

The build compares serialized/reloaded position, normal, UV, index and rigid control-binding arrays to the generated target arrays. The standalone validator checks stock part preservation, unchanged collision planes/animation, eye/muzzle placement, skeleton/material/texture links, expected buffer layout, finite/unit normals, optional weapon/audio format and ZIP hashes. The reviewer opens the emitted Ogre mesh files and renders exterior, forward and downward cockpit views. Its material approximation is an offline inspection aid, not a native shader test.

UVs convert from Blender `(U,V)` to Ogre `(U,1-V)`, matching the shared toolkit's pure exporter. The review importer reverses this conversion. A review that reused the same unconverted UVs on both sides could hide a runtime texture inversion; this conversion is recorded explicitly in the build receipt. The gun texture audit under `review/` shows source albedo, tinted diffuse and decoded normal maps for all four installed families. Source gun albedos are mostly plain; vents, grooves and much of the smaller detail reside in normal maps. Their serialized illumination defaults can brighten broad painted surfaces; the native opaque shader permutation permits unmasked illumination, so absence of an emission texture alone does not establish that the gain should be zeroed.

Receipts are outside the runtime folder: `build.json`, `validation.json`, `runtime_sha256.json`, optional `installation.json`, and `review/review.json`. Keep the source dump and these receipts for further fitting and animation conversion. No Workshop publication is performed by these tools.
