<img width="1920" height="1032" alt="Battlezone Gold Extractor" src="https://github.com/user-attachments/assets/b4de9d54-479b-49d2-bce4-82d217d5f91d" />

# Battlezone Gold Extractor

Extracts and reverse-engineers previously undocumented proprietary models, textures, audio, and archive/container data from **Battlezone Gold Edition (2017)** on Rebellion's Asura engine.

<img width="728" height="550" alt="Extracted Battlezone Gold model in Blender" src="https://github.com/user-attachments/assets/86c82040-edfc-4c8f-8f5e-20379843b744" />

## Scope

- Document Asura archive/container formats and file signatures.
- Extract supported Asura archive wrappers with the native extractor, with optional QuickBMS fallback.
- Discover and convert texture payloads by file signature rather than extension alone.
- Export HSKN and extensionless model candidates to OBJ for inspection.
- Discover/copy audio payloads by header signature.
- Provide a desktop GUI and unified CLI for the end-to-end pipeline.

## Format Documentation

- [ASURA_FORMATS.md](./ASURA_FORMATS.md): reverse-engineered technical baseline for Asura wrappers/chunks and discovered asset types, including binary offset maps, structural assumptions, confidence levels, known gaps, and reproducible commands.

## Universal Extractor

Use one entry point for independent or combined dumps of archives, textures, models, and audio:

```powershell
python .\tools\bzg-extractor.py --help
```

Single-task examples:

```powershell
python .\tools\bzg-extractor.py extract --game-root work/Battlezone --extract-root work/extracted/full_all --logs-root work/logs/batch_all --extractor Native --no-skip-existing
python .\tools\bzg-extractor.py textures --extract-root work/extracted/full_all --texture-discovery headers --texture-stage-root work/preview/texture_stage_header --texture-discovery-report notes/texture-header-discovery-report.csv --texture-output-root work/preview/png_all_texconv --texture-report-path notes/texture-conversion-report-full-all-texconv.csv
python .\tools\bzg-extractor.py models --extract-root work/extracted/full_all --model-output-root work/preview/model_obj_universal --model-backend hskn --model-discovery headers --model-max-files 0
python .\tools\bzg-extractor.py audio --extract-root work/extracted/full_all --audio-discovery headers --audio-output-root work/preview/audio_dump --audio-report-path notes/audio-dump-report.csv
```

Combined pipeline:

```powershell
python .\tools\bzg-extractor.py run --tasks extract,textures,models,audio --game-root work/Battlezone --extract-root work/extracted/full_all --logs-root work/logs/batch_all --extractor Native --texture-discovery headers --audio-discovery headers --model-backend hskn --model-discovery headers --model-max-files 0 --skip-existing
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

The single-file application embeds the unified backend, native archive extractor, both model-export helpers, runtime PowerShell scripts, and the bundled BMS script. Optional third-party runtime binaries are included only when present under `tools/bin` at build time.

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

## Known Reverse-Engineering Limits

The native extractor currently supports direct `Asura`, `AsuraZlb`, and `AsuraZbb` wrappers. `AsuraCmp` and xcompress wrappers remain unsupported natively and are intended for optional QuickBMS fallback. Model decoding remains partially heuristic; see `ASURA_FORMATS.md` for current confidence levels and validation counts.

## Legacy Scripts

Older research and diagnostic scripts live under `tools/legacy/` so the top-level toolkit remains focused on production extraction and UI workflows.
