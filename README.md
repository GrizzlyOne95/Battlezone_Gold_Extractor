<<<<<<< HEAD

<img width="2048" height="512" alt="bz_cinematic_07_outro" src="https://github.com/user-attachments/assets/c84e324a-9922-459f-a7f6-15c05472fa6c" />

# Battlezone_Gold_Extractor
Extracts the previously unknown proprietary models, textures, and audio from Battlezone Gold (Asura Engine)

<img width="728" height="550" alt="blender_lqhK5k2TvL" src="https://github.com/user-attachments/assets/86c82040-edfc-4c8f-8f5e-20379843b744" />
=======
# Battlezone Gold 2017 Reverse Engineering Workspace

This repository is a working area for researching file formats and asset extraction for **Battlezone Gold Edition (2017)** on Rebellion's Asura engine.

## Scope

- Document archive/container formats and file signatures.
- Build repeatable extraction and inspection workflows.
- Keep notes, scripts, and reproducible findings under version control.

## Format Documentation

- [ASURA_FORMATS.md](./ASURA_FORMATS.md): reverse-engineered technical baseline for Asura wrappers/chunks and discovered asset types (textures, audio, models, materials), including binary offset maps/structure diagrams and reproducible commands.

## Universal Extractor (Recommended)

Use one entry point for independent or combined dumps of archives, textures, models, and audio:

```powershell
python .\tools\bzg-extractor.py --help
```

Single task examples:

```powershell
python .\tools\bzg-extractor.py extract --game-root work/Battlezone --extract-root work/extracted/full_all --logs-root work/logs/batch_all --extractor Native --no-skip-existing
python .\tools\bzg-extractor.py textures --extract-root work/extracted/full_all --texture-discovery headers --texture-stage-root work/preview/texture_stage_header --texture-discovery-report notes/texture-header-discovery-report.csv --texture-output-root work/preview/png_all_texconv --texture-report-path notes/texture-conversion-report-full-all-texconv.csv
python .\tools\bzg-extractor.py models --extract-root work/extracted/full_all --model-output-root work/preview/model_obj_universal --model-backend hskn --model-discovery headers --model-max-files 0
python .\tools\bzg-extractor.py audio --extract-root work/extracted/full_all --audio-discovery headers --audio-output-root work/preview/audio_dump --audio-report-path notes/audio-dump-report.csv
```

Run combined pipeline:

```powershell
python .\tools\bzg-extractor.py run --tasks extract,textures,models,audio --game-root work/Battlezone --extract-root work/extracted/full_all --logs-root work/logs/batch_all --extractor Native --texture-discovery headers --audio-discovery headers --model-backend hskn --model-discovery headers --model-max-files 0 --skip-existing
```

Desktop UI wrapper:

```powershell
python .\tools\bzg-extractor-ui.py
```

## Standalone Build

Compile the toolkit to a standalone executable (single-file default):

```powershell
.\tools\build-standalone.ps1 -PythonExe python -DistDir dist -OneFile:$true -Windowed:$true -BundleRuntimeBin:$true -BundleBms:$true
```

Runtime binaries can be bundled by placing them in `tools/bin` before build (for example `ffmpeg.exe`, `texconv.exe`, `quickbms_4gb_files.exe`).

For a classic folder bundle instead of onefile, set `-OneFile:$false`.

## GitHub Actions Release

Windows EXE CI is configured in:

- `.github/workflows/windows-release.yml`

Triggers:

- Push a tag matching `v*` (for example `v1.0.0`) to build and publish `dist/BattlezoneGoldExtractor.exe` as a GitHub Release asset.
- Manual run via **Actions -> Windows EXE Release -> Run workflow** (build artifact upload).

## Legacy Scripts

Older research/diagnostic scripts were moved to `tools/legacy/` to keep the top-level toolkit focused on production extraction and UI workflows.
>>>>>>> master
