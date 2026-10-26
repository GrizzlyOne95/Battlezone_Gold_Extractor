# Native shader and resource recovery

The BZR weapon coverage pass now also decodes native HTXT menu text with strict bounds and key/hash pairing; see `read_gold_text` in `tools/gold_weapons.py`. The English menu table has 2,714 entries and 19 `WEAPON_DESC_*` descriptions. Records contain hash/u32 UTF-16-unit-count/terminated UTF-16 text; the header's value byte count excludes record headers. A padded table name and counted null-terminated key byte stream follow. Keys use the native 31-based lowercase byte hash (including signed-byte legacy cases). UI text may include unused content and does not recover numeric blueprint balance. The generated `bzr_weapons/coverage.json` retains descriptions, tier names, source SHA and explicit equipment-controller gaps.

`python tools/bzg-extractor.py resources --extract-root PATH/raw --resource-output-root PATH/resources --game-root GAME`

The `resources` stage is included in the default combined pipeline and the desktop UI. It preserves shader bytecode and indexes material parameters, native effects/UI dependencies, DDS subresources and VTEX volume payloads. No game files are changed.

The October 24 installation yielded 106 banks, 2,780 shader occurrences and 409 unique DXBC programs. All 409 were disassembled and reflected through the Windows `d3dcompiler_47.dll` API, with no resource-pass errors. Other counts: 4,619 material records; 40 cubemap DDS files; 351 VTEX records representing nine unique payloads; 84 native chunk families; 480 named effect groups; 57 texture-animation chunks. Counts include archive repetitions.

Model shader banks contain a u32 count followed by u64 permutation key, u32 size and DXBC bytecode for each entry. Particle banks omit the permutation key and may retain one trailing u32 per program. The reader validates container/chunk bounds, overlapping DXBC chunks and exact bank consumption. It also scans raw resources and optional game EXE/DLL files for additional bounded DXBC containers.

Each unique shader has its original `.dxbc`, disassembly `.asm`, reflection JSON and source-offset metadata under `shaders/programs/`. Reflection includes resources, signatures and recursive constant-buffer type layouts. Original HLSL is not reconstructed. The API interfaces follow Microsoft's [D3DDisassemble](https://learn.microsoft.com/en-us/windows/win32/api/d3dcompiler/nf-d3dcompiler-d3ddisassemble) and [D3DReflect](https://learn.microsoft.com/en-us/windows/win32/api/d3dcompiler/nf-d3dcompiler-d3dreflect) documentation.

MARE records retain raw words, byte/float views and exact serialized texture-hash matches. Reflected material layout schemas are retained separately; the mapping of all serialized parameters to all shader constants remains unresolved. Fields must not be assigned semantics solely from their offset or appearance.

## Code-derived surface defaults

The next pass traced the installed DX11 executable's MARE loader (`0x1400a9650`), primitive constant upload (`0x140390795`–`0x1403908ac`), material constant construction (`0x14038f1d0`) and recovered DXBC. Evidence and the binary SHA-256 are saved in the dump's `reports/shading_code/`. No running-game observation is required for these default-value mappings.

| Serialized record offset | Meaning supported by the code |
| --- | --- |
| `0x08` | Response-LUT hash, **not a shader-program hash**. `shader_hash` remains a compatibility alias in JSON. |
| `0x40` | Material model. The pixel shader's model 3 selects albedo-colored F0 and suppresses diffuse; other models use dielectric F0 0.04. |
| `0x48` | Self-illumination scale. |
| `0x4c` | Self-illumination RGB. Gain times RGB becomes `g_xSelfIlluminationParameters`. |
| `0xdc` / `0x134` | Reflectivity / refractivity constants; not every shader permutation uses them. |
| `0xe4` | Parallax scale. Native bias is scale times -0.5. |
| `0x150` | Albedo RGB modifier. Loaded to runtime material `+0x188`, then copied to `g_xAlbedoModifier` without a gamma operation. |

Shader code multiplies sampled albedo RGB by that modifier, clamps PBR alpha roughness to at least 0.05, reconstructs normal Z from packed RG and uses the red channel of self-illumination maps. Emission contributes through tinted albedo with the native gain/color. Blender's Principled model approximates the full BRDF and lighting; it is not a native shader interpreter. Entity callbacks can replace tint/emission, so serialized defaults do not certify every gameplay state.

The complete MARE version-2 header is `u32 material_count, u32 version, u32 response_lut_count`. Each response entry contains a u32 hash and **256 float4 samples**, followed by the 356-byte material records. All 120 tables consume their source exactly and all response references resolve. The dump now preserves 2,031 LUT occurrences / 156 unique float32 payloads with sample JSON under `materials/response_luts/`. View-dependent channel operations vary by shader permutation; tail samples also contain metadata. These are response tables, not color textures, and their full behavior is not yet translated.

VTEX is a u32 count, then repeated hash/width/height/depth u32 descriptors and width*height*depth*4 raw bytes. Outputs preserve raw channels, NumPy arrays and labeled slice previews. Channel meanings are unverified. DDS originals preserve all native mip/face data; `asura_textures.py` splits legacy cubemap faces without dropping mip chains and makes an inspection environment projection.

`native_chunks/` records hashes, paths and recoverable names for every chunk family. Effects and texture-animation indexes retain named dependencies and exact unknown payloads. This is resource recovery, not a simulation of the game's effects, GUI, entity states, or shader BRDF.
