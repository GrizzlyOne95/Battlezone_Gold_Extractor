<img width="1920" height="1032" alt="Battlezone Gold Extractor" src="https://github.com/user-attachments/assets/b4de9d54-479b-49d2-bce4-82d217d5f91d" />

# Battlezone Gold Extractor

Extracts and reverse-engineers previously undocumented proprietary models, textures, audio, and archive/container data from **Battlezone Gold Edition (2017)** on Rebellion's Asura engine.

<img width="728" height="550" alt="Extracted Battlezone Gold model in Blender" src="https://github.com/user-attachments/assets/86c82040-edfc-4c8f-8f5e-20379843b744" />

## Scope

- Document Asura archive/container formats and file signatures.
- Extract supported Asura archive wrappers with the native extractor, with optional QuickBMS fallback.
- Discover and convert texture payloads by file signature rather than extension alone.
- Export native model hierarchies, render meshes, materials, UVs, skinning, sockets and animation to glTF, plus legacy OBJ previews.
- Discover/copy audio payloads by header signature.
- Recover native DXBC shader bytecode, reflection, material tables, volume textures and effects dependencies.
- Provide a desktop GUI and unified CLI for the end-to-end pipeline.

## Format Documentation

- [ASURA_FORMATS.md](./ASURA_FORMATS.md): reverse-engineered technical baseline for Asura wrappers/chunks and discovered asset types, including binary offset maps, structural assumptions, confidence levels, known gaps, and reproducible commands.
- [SCENE_FORMATS.md](./SCENE_FORMATS.md): corrected scene layouts, full player-tank assembly and Blender material inspection.
- [RESOURCE_FORMATS.md](./RESOURCE_FORMATS.md): native shader/resource recovery and remaining runtime limits.
- [GOLD_WEAPONS_MOD.md](./GOLD_WEAPONS_MOD.md): BZR player weapon profiles and native tier names, reference-checked ordnance orientation, particles, audio, pickups and Grizzly loadout integration using ISDFC examples.

## Universal Extractor

Use one entry point for independent or combined dumps of archives, textures, models, and audio:

```powershell
python .\tools\bzg-extractor.py --help
```

Single-task examples:

```powershell
python .\tools\bzg-extractor.py extract --game-root work/Battlezone --extract-root work/extracted/full_all --logs-root work/logs/batch_all --extractor Native --no-skip-existing
python .\tools\bzg-extractor.py textures --extract-root work/extracted/full_all --texture-discovery headers --texture-stage-root work/preview/texture_stage_header --texture-discovery-report notes/texture-header-discovery-report.csv --texture-output-root work/preview/png_all_texconv --texture-report-path notes/texture-conversion-report-full-all-texconv.csv
python .\tools\bzg-extractor.py models --extract-root work/extracted/full_all --model-output-root work/preview/model_obj_universal --model-backend scene --model-discovery headers --model-max-files 0
python .\tools\bzg-extractor.py audio --extract-root work/extracted/full_all --audio-discovery headers --audio-output-root work/preview/audio_dump --audio-report-path notes/audio-dump-report.csv
python .\tools\bzg-extractor.py resources --extract-root work/extracted/full_all --resource-output-root work/preview/resources --game-root work/Battlezone
```

Combined pipeline:

```powershell
python .\tools\bzg-extractor.py run --tasks extract,textures,models,resources,audio --game-root work/Battlezone --extract-root work/extracted/full_all --logs-root work/logs/batch_all --extractor Native --texture-discovery headers --audio-discovery headers --target-set AllAsuraHeaders --model-backend scene --model-discovery headers --model-max-files 0 --model-min-triangles 1 --skip-existing
```

Desktop UI:

```powershell
python .\tools\bzg-extractor-ui.py
```

## Release Builds

The public Windows executable has a stable, versionless name:

- `BZGoldExtractor.exe`

Release archives carry the version and platform, for example:

- `Battlezone_Gold_Extractor-v1.1.0-windows.zip`

Official Windows releases use the shared Battlezone tool-suite metadata:

```text
FileDescription: Battlezone Gold Extractor
ProductName: Battlezone Modding Tools
CompanyName: GrizzlyOne95
OriginalFilename: BZGoldExtractor.exe
```

`FileVersion` and `ProductVersion` are derived from the canonical repository version.

## Standalone Build

Build the toolkit as a single-file executable:

```powershell
.\tools\build-standalone.ps1 -PythonExe python -DistDir dist -OneFile:$true -Windowed:$true -BundleRuntimeBin:$true -BundleBms:$true
```

The single-file application embeds the unified backend, native archive extractor, structured scene and resource exporters, legacy OBJ helpers, runtime PowerShell scripts, and the bundled BMS script. Optional third-party runtime binaries are included only when present under `tools/bin` at build time.

This distinction matters for full feature coverage:

- Native archive extraction and model export are self-contained in the packaged application.
- Texture conversion requires `ffmpeg` unless an FFmpeg binary was bundled under `tools/bin`; `texconv` is optional.
- QuickBMS fallback requires a QuickBMS executable unless one was bundled under `tools/bin`.

For a classic folder bundle instead of one-file mode, set `-OneFile:$false`.

## GitHub Actions Release

Windows release CI is configured in `.github/workflows/windows-release.yml`.

- Pushes to `main` and pull requests build the same packaged application using the canonical `VERSION` value.
- A `chore(release): vX.Y.Z` commit or matching `v*` tag builds and publishes the versioned Windows archive containing `BZGoldExtractor.exe`.
- Manual runs build a downloadable artifact without creating a GitHub Release.

## Validation and full dump behavior

Run the regression checks before a dump:

```powershell
python -m unittest discover -s tests -v
```

`AllAsuraHeaders` is the default target set. It scans file headers as well as named `.pc*`/`.asr*` candidates, covering DLC, GUI, sound and locale containers. `AllPcVariants` remains available for the older restricted selection. Valid `.pc_entheader` references are reported as pointers. Unsupported wrappers, extraction failures and parser warnings cause a nonzero batch exit; inspect the per-archive logs.

Header discovery reads bounded prefixes. Audio sync alone is insufficient to classify MP3: two valid frame headers are required. Detected suffixes are appended when the source suffix differs (for example, `texture.tga.dds` and `.12.dat.wav`), preserving the original name and avoiding collisions. PNG outputs retain the staged suffix, for example `texture.tga.dds.png`. Distinct repeated resource names inside an archive are preserved with `__duplicate_N` suffixes.

The default model backend is now `scene`: glTF with native hierarchy, render geometry, UVs, material references, skinning, attachment markers and HCAN animation. See [SCENE_FORMATS.md](SCENE_FORMATS.md) for validated layouts, results and remaining limits. The legacy OBJ backends are geometry previews. Native texture and audio copies are retained. Pillow provides BC7 texture fallback; install runtime dependencies with `python -m pip install -r requirements.txt`. Copying WAV does not convert its codec to PCM; use `tools/convert-audio-to-pcm.py` for separate PCM16 copies.

`asura_materials.py` now decodes native default tint, material-model and emission parameters traced through DX11 loader/constant-buffer instructions. It also preserves MARE response LUTs; the old `shader_hash` key is a compatibility alias for the response-LUT hash. See [RESOURCE_FORMATS.md](RESOURCE_FORMATS.md).

After scene/texture/resource extraction, `tools/build-asura-asset-library.py --dump-root PATH --prepare` prepares categorized catalogs, a library README and a player-weapon CSV inventory. Run it through Blender with `--build` to create collection-asset libraries for player vehicles/skins, enemy tanks, drones/enemies, turrets, props/structures and exterior player weapons. The validated dump has 39 libraries covering 1,288 render resources, plus 12 unit inspection assemblies and 15 player-weapon examples. The weapon examples contain 11 gun families, three deployable hardpoint housings and one mounted tank inspection loadout. All 119 weapon-related resources are inventoried, including variants, ammunition and node-only interfaces. These counts include components and variants. Native entity state/loadout reconstruction and BZR gameplay integration remain separate work.

Animations open muted in the inspection libraries. `tools/inspect-asura-animations.py` lists native clips and their Blender owners, activates an exact clip across its rigs/root-motion objects, optionally validates sampled motion, and saves a separate playable preview. `build-asura-asset-library.py --dump-root PATH --index-only` refreshes the library README and weapon/animation CSV inventories. See [ANIMATIONS.md](ANIMATIONS.md) for playback commands, tested coverage and remaining runtime limits.

For BZR conversion, [BZR_PORTING.md](BZR_PORTING.md) compares a stock-compatible control-rig route with an EXU detail-animation route. `tools/prepare-bzr-port.py --dump-root PATH` generates per-resource readiness checks, complete tank source staging, material/clip handoff manifests and explicit alias proposals. Its optional Blender mode saves/reloads/checks the staged assembly. It does not emit BZR binaries or install game files.

[GOLD_GRIZZLY_MOD.md](GOLD_GRIZZLY_MOD.md) describes the first standalone model-conversion prototype: a `mapType = "mod"` Grizzly override with separate world/cockpit Ogre files, stock gameplay controls, target material bakes, package validation and installation helper. It is a posed baseline; the detailed Gold clips remain in the extraction source for later controller work. Gold weapon definition research preserves model/clip controller records and native property-reader evidence while leaving unresolved numeric balance explicitly unassigned.

## Known Reverse-Engineering Limits

The native extractor currently supports direct `Asura`, `AsuraZlb`, and `AsuraZbb` wrappers. `AsuraCmp` and xcompress wrappers remain unsupported natively and are intended for optional QuickBMS fallback. All 1,401 HSKN candidates and 757 HCAN chunks in the validated local installation now decode. See `SCENE_FORMATS.md` and `RESOURCE_FORMATS.md` for source UV defects, interpolation assumptions, shader approximations and remaining runtime work. Historical geometry offsets in `ASURA_FORMATS.md` are superseded.

## Legacy Scripts

Older research and diagnostic scripts live under `tools/legacy/` so the top-level toolkit remains focused on production extraction and UI workflows.
