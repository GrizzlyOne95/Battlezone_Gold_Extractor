# Tools

Primary toolkit for the universal **Battlezone Gold Extractor**.

## Core

- `bzg-extractor.py`: unified backend CLI.
  - `extract` archive extraction (native/QuickBMS/auto)
  - `textures` texture preview conversion
    - supports `--texture-discovery headers` for full header-based texture discovery (including extensionless candidates)
    - in header mode, files are staged into `--texture-stage-root` with detected extensions before PNG conversion
  - `models` OBJ export (`hskn`, `extless`, `both`)
    - supports `--model-discovery headers` to export by geometry/header detection instead of name keywords
    - `--model-max-files 0` means no cap (all discovered candidates)
  - `audio` audio dump/copy with report
    - supports `--audio-discovery headers` for full header-based audio discovery (including extensionless candidates)
  - `run` multi-task pipeline
  - `runtime` resolved runtime paths (diagnostics)
- `bzg-extractor-ui.py`: desktop UI wrapper for the backend.
- `build-standalone.ps1`: builds standalone executables (UI + helper tools) via PyInstaller.
- `cleanup-workspace.ps1`: archives root scratch `.txt` files to `work/archive`.

## Runtime Helpers (Used By Core)

- `extract-asura-native.py`: native Asura extractor.
- `extract-asura-batch.ps1`: batch extraction wrapper.
- `convert-textures-to-png.ps1`: texture conversion wrapper.
- `export-hskn-obj-candidates.py`: preferred model OBJ export.
- `export-model-obj-candidates.py`: extensionless model OBJ fallback export.
- `bms/asura.bms`: QuickBMS script (optional fallback).
- `bin/`: optional bundled binaries (`ffmpeg`, `texconv`, `quickbms`, etc.).

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

- `BattlezoneGoldExtractor.exe` (UI)
- helper executables and runtime resources embedded into the UI EXE

Folder mode (`-OneFile:$false`) output contains:

- `BattlezoneGoldExtractor.exe` (UI)
- `bzg-extractor.exe` (CLI helper)
- `extract-asura-native.exe`
- `export-hskn-obj-candidates.exe`
- `export-model-obj-candidates.exe`
- runtime scripts/resources under `tools/`

## Legacy Scripts

Legacy research/diagnostic scripts were moved to:

- `tools/legacy/`
