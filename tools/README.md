# Tools

Primary toolkit for the universal **Battlezone Gold Extractor**.

## Structured scenes

The default `models` backend is `scene`. Use `--model-texture-root` to point models-only exports at the converted PNG tree; combined runs inherit `--texture-output-root`. `export-asura-scenes.py` and `asura_scene.py` preserve native hierarchy, render geometry, material references, UVs, skinning, markers and animation. See `../SCENE_FORMATS.md` for the layout and validation limits. `convert-audio-to-pcm.py` writes separate PCM16 copies of the native audio.

## Core

- `bzg-extractor.py`: unified backend CLI.
  - `extract` archive extraction (`Native`, `QuickBMS`, or `Auto`)
  - `textures` texture preview conversion
    - supports `--texture-discovery headers` for header-based texture discovery, including extensionless candidates
    - header mode stages files into `--texture-stage-root` with detected extensions before PNG conversion
  - `models` structured glTF export (`scene`, default) or legacy OBJ (`hskn`, `extless`, `both`)
    - supports `--model-discovery headers` to export by geometry/header detection instead of name keywords
    - `--model-max-files 0` means no cap
  - `audio` audio dump/copy with report
    - supports `--audio-discovery headers` for header-based audio discovery, including extensionless candidates
  - `resources` native shader bytecode/disassembly/reflection, material tables, volumes and effects dependencies
  - `run` multi-task pipeline
  - `runtime` resolved runtime-path diagnostics
- `bzg-extractor-ui.py`: desktop UI wrapper for the backend.
- `build-standalone.ps1`: builds standalone executables via PyInstaller.
- `cleanup-workspace.ps1`: archives root scratch `.txt` files to `work/archive`.

## Runtime Helpers

- `extract-asura-native.py`: native Asura extractor.
- `extract-asura-batch.ps1`: batch extraction wrapper.
- `convert-textures-to-png.ps1`: texture conversion wrapper.
- `export-hskn-obj-candidates.py`: preferred HSKN model OBJ export.
- `export-model-obj-candidates.py`: extensionless model OBJ fallback export.
- `export-asura-scenes.py`: native hierarchy, geometry, skinning, sockets and animation.
- `export-asura-resources.py`: native shaders, reflected parameter layouts and resource dependencies.
- `build-player-tank-materials.py`: Blender 5.2 builder for the assembled tank, packed native textures and inspection cameras; reopens and checks all geometry and sampled native motion. Run through Blender with `--python-exit-code 1 --background --python tools/build-player-tank-materials.py -- --dump-root PATH --render` after extracting scenes, textures and resources. See `../SCENE_FORMATS.md`.
- `asura_materials.py`: complete version-2 MARE table/LUT reader and native default tint, material model, emission and selected scalar decoding. See `../RESOURCE_FORMATS.md` for code evidence and remaining shader limits.
- `build-asura-asset-library.py`: prepare categorized glTF catalogs with ordinary Python (`--dump-root PATH --prepare`); build Blender collection assets with Blender (`--background --python-exit-code 1 --python tools/build-asura-asset-library.py -- --dump-root PATH --build`). Bulk libraries use relative references to shared textures, native local hierarchies and camera markers for each asset. Inspection-only assembly aliases are explicit in the manifest.
- `render-asura-library-previews.py`: render representative saved library collections offline, without changing the saved library files.
- `inspect-asura-animations.py`: run through Blender with an existing library open to `--list` clips or `--clip NAME`; supports an exact `--asset` filter, `--camera-frame`, start/mid/end `--validate`, channel-binding `--audit`, JSON `--report`, and separate playable `--output`. `--audit-library DUMP_ROOT` checks all saved libraries without an initially open file. Coordinates action slots across rigs and root-motion objects. See [ANIMATIONS.md](../ANIMATIONS.md) for commands and validation scope. This helper needs an installed Blender; the extractor UI does not provide playback.
- `prepare-bzr-port.py`: ordinary Python `--dump-root PATH` produces resource/weight-risk inventories, source assembly staging and conversion manifests. Through Blender, add `--build-blender-source` to save/reload/check/render the staged complete tank. See [BZR_PORTING.md](../BZR_PORTING.md). This is offline preparation, not a BZR binary exporter or installer.
- `build-gold-grizzly-mod.py`: Blender builder for the standalone posed Grizzly override, using existing shared BZR serializers and native ZFS reading. Produces world/cockpit models, a VDF override, material bakes, optional unequipped autocannon prototype, ZIP and SHA receipt. See [GOLD_GRIZZLY_MOD.md](../GOLD_GRIZZLY_MOD.md).
- `validate-gold-grizzly-mod.py`: ordinary-Python offline binary/dependency/ODF/audio/ZIP checks; requires the shared toolkit/toolbox paths.
- `review-gold-grizzly-mod.py`: Blender renders of the emitted Ogre files from the exact authored eye; not a native BZR renderer test.
- `install-gold-grizzly-mod.py`: hash-checked copy into a dedicated `mods/goldgrizzly` folder; leaves mod enablement to the user and refuses to overwrite changed destination files.
- `export-gold-weapon-definitions.py`: conservative model/clip controller recovery, raw blueprint reference candidates and BZR recreation table. Optional native executable schema tracing requires `pefile` and `capstone`; weapon-specific numeric balance remains unresolved.
- `bms/asura.bms`: QuickBMS script for optional fallback.
- `bin/`: optional bundled third-party binaries (`ffmpeg`, `texconv`, `quickbms`, etc.).

## Quick Commands

```powershell
python .\tools\bzg-extractor.py --help
python .\tools\bzg-extractor.py run --tasks extract,textures,models,resources,audio --game-root work/Battlezone --extract-root work/extracted/full_all --logs-root work/logs/batch_all --extractor Native --texture-discovery headers --audio-discovery headers --target-set AllAsuraHeaders --model-backend scene --model-discovery headers --model-max-files 0 --model-min-triangles 1 --skip-existing
python .\tools\bzg-extractor-ui.py
```

## Standalone Build

The library builder's `--index-only` refreshes README/weapon/animation CSV indexes from existing catalogs. `--prepare` regenerates catalogs and `--build` through Blender regenerates collection assets and validation reports. Blender authoring/playback scripts are external tools, separate from the frozen extraction helpers below.

```powershell
.\tools\build-standalone.ps1 -PythonExe python -DistDir dist -OneFile:$true -Windowed:$true -BundleRuntimeBin:$true -BundleBms:$true
```

Default (`-OneFile:$true`) output:

- `BZGoldExtractor.exe` (public UI application)
- backend/helper executables and project runtime resources embedded into the UI EXE

Folder mode (`-OneFile:$false`) output contains:

- `BZGoldExtractor.exe` (UI)
- `bzg-extractor.exe` (internal CLI helper)
- `extract-asura-native.exe`
- `export-hskn-obj-candidates.exe`
- `export-model-obj-candidates.exe`
- `export-asura-scenes.exe`
- `export-asura-resources.exe`
- runtime scripts/resources under `tools/`

The helper executable names above are internal runtime contracts and intentionally remain unchanged. Optional third-party binaries are only bundled when a `tools/bin` directory exists at build time; otherwise features that need those binaries resolve them from the host system.

## Gold weapons for BZR

`build-gold-weapons-mod.py` builds 22 weapon/ordnance/explosion/pickup profiles, 111 native tier-name aliases, Gold audio/effect cards and an `avtank.odf` loadout. `gold_weapons.py` strictly decodes native HTXT menu text for coverage. Its background-Blender worker `export-gold-projectiles.py` uses shared toolkit serializers for GEO and Ogre models. `audit-gold-ordnance.py` compares axes/origins and native landmarks against installed stock and ISDFC meshes and writes a Matplotlib comparison. `validate-gold-weapons-mod.py` checks dependency/render graphs, charge/salvo/cluster/flare connections, audio, binaries and ZIP; `review-gold-weapons-mod.py` renders reloaded models. Both validators use `bzr_materials.py` to reject missing stock imports or unresolved inheritance. The installer accepts `--mod-name goldweapons`, defaults to the local `addon/` layout, and preserves mod selection. See [GOLD_WEAPONS_MOD.md](../GOLD_WEAPONS_MOD.md) for commands, references and fidelity limits.

## Legacy Scripts

Legacy research and diagnostic scripts were moved to `tools/legacy/`.
