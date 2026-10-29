# Battlezone Gold Extractor

Extract models, textures, audio, and other resources from **Battlezone Gold Edition (2017)**, which uses Rebellion's Asura engine. The toolkit provides a Windows desktop interface and a Python command-line workflow for asset inspection, conversion, and modding research.

<img width="728" alt="Extracted Battlezone Gold model in Blender" src="https://github.com/user-attachments/assets/86c82040-edfc-4c8f-8f5e-20379843b744" />

## Quick start (Windows)

1. Download the [latest release](https://github.com/GrizzlyOne95/Battlezone_Gold_Extractor/releases/latest) and extract the ZIP.
2. Run **`BZGoldExtractor.exe`**.
3. Set **Game Root** to your installed Battlezone Gold Edition folder. For the standalone EXE, set output and log paths to absolute folders outside the app's temporary runtime directory.
4. Select the tasks you need and click **Run Selected Tasks**. Check the log for unsupported archives or conversion errors.

The release includes the native extractor and model exporters. **Texture-to-PNG conversion requires FFmpeg on your PATH**; `texconv` is an optional fallback. **QuickBMS** is optional for archives the native extractor cannot handle. Use **Runtime Check** in the app to inspect available tools.

You must supply your own game installation; game assets are not included.

## What it does

- **Archives:** Scans Asura containers and extracts supported native wrappers, with optional QuickBMS fallback.
- **Models:** Exports reconstructed scene hierarchies, meshes, UVs, materials, skinning, attachment markers, and animation to **glTF**. Legacy OBJ previews are also available.
- **Textures:** Detects image formats by file headers and converts supported textures to PNG.
- **Audio:** Finds and copies recognized audio data by signature; copying does not convert compressed audio to PCM.
- **Resources:** Recovers shaders, material parameters, texture volumes, and effect/dependency information where supported.

## Run from source

Requires Python 3.10+, PowerShell, and the dependencies in `requirements.txt`. Run these commands from the repository root:

```powershell
python -m pip install -r requirements.txt
python .\tools\bzg-extractor-ui.py
```

The CLI can run individual tasks (`extract`, `textures`, `models`, `resources`, `audio`) or a combined pipeline:

```powershell
python .\tools\bzg-extractor.py run --game-root "C:\Path\To\Battlezone" --tasks extract,textures,models,resources,audio
```

Use `python .\tools\bzg-extractor.py --help` and `python .\tools\bzg-extractor.py run --help` for options. When running from source, unspecified paths use the repository's default `work/` and `notes/` locations.

## Limitations

This is a reverse-engineering toolkit, not a finished asset porting pipeline.

- Native extraction supports `Asura`, `AsuraZlb`, and `AsuraZbb`; `AsuraCmp` and xcompress wrappers are not supported natively. QuickBMS fallback is not guaranteed for every archive.
- Exported geometry, animation, and materials are reconstructed from proprietary formats and may not exactly match the game's runtime rendering or behavior.
- Blender asset libraries and Battlezone 98 Redux mod prototypes require additional tooling and validation; they are not automatically produced by a basic extraction run.

## Detailed documentation

- [Asura archives and binary formats](ASURA_FORMATS.md)
- [Models, scenes, and animations](SCENE_FORMATS.md) · [Animation playback](ANIMATIONS.md)
- [Shaders, materials, and resources](RESOURCE_FORMATS.md)
- [Blender asset tooling and CLI utilities](tools/README.md)
- [Battlezone 98 Redux porting research](BZR_PORTING.md)
- [Gold Grizzly prototype](GOLD_GRIZZLY_MOD.md) · [Gold weapons prototype](GOLD_WEAPONS_MOD.md)

## Development

Run tests:

```powershell
python -m unittest discover -s tests -v
```

Build a Windows executable (requires PyInstaller):

```powershell
python -m pip install pyinstaller
.\tools\build-standalone.ps1
```

The [Windows release workflow](.github/workflows/windows-release.yml) tests and builds on pushes and PRs; version tags publish release ZIPs. Version information comes from [`VERSION`](VERSION).

The extractor is [MIT licensed](LICENSE). Battlezone and its game assets belong to their respective rights holders.
