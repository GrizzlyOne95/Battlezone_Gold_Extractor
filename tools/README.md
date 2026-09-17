# Tools

Primary toolkit for the universal **Battlezone Gold Extractor**.

## Core

- `bzg-extractor.py`: unified backend CLI.
  - `extract` archive extraction (`Native`, `QuickBMS`, or `Auto`)
  - `textures` texture preview conversion
    - supports `--texture-discovery headers` for header-based texture discovery, including extensionless candidates
    - header mode stages files into `--texture-stage-root` with detected extensions before PNG conversion
  - `models` OBJ export (`hskn`, `extless`, or `both`)
    - supports `--model-discovery headers` to export by geometry/header detection instead of name keywords
    - `--model-max-files 0` means no cap
  - `audio` audio dump/copy with report
    - supports `--audio-discovery headers` for header-based audio discovery, including extensionless candidates
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
- `bms/asura.bms`: QuickBMS script for optional fallback.
- `bin/`: optional bundled third-party binaries (`ffmpeg`, `texconv`, `quickbms`, etc.).

## Quick Commands

```powershell
python .\tools\bzg-extractor.py --help
python .\tools\bzg-extractor.py run --tasks extract,textures,models,audio --game-root work/Battlezone --extract-root work/extracted/full_all --logs-root work/logs/batch_all --extractor Native --texture-discovery headers --audio-discovery headers --model-backend hskn --model-discovery headers --model-max-files 0 --skip-existing
python .\tools\bzg-extractor-ui.py
```

## Standalone Build

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
- runtime scripts/resources under `tools/`

The helper executable names above are internal runtime contracts and intentionally remain unchanged. Optional third-party binaries are only bundled when a `tools/bin` directory exists at build time; otherwise features that need those binaries resolve them from the host system.

## Legacy Scripts

Legacy research and diagnostic scripts were moved to `tools/legacy/`.
