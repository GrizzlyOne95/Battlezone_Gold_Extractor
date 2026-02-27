# Asura Engine Asset Format Notes (Battlezone Gold Edition)

Last updated: February 27, 2026

This document is a reverse-engineered, reproducible technical baseline for asset formats observed in this repository's Battlezone Gold dataset. It is based on validated extraction/tool outputs in this workspace, not on official vendor documentation.

## 1) Scope And Baseline Dataset

Baseline inputs used for the findings below:

- Game root: `work/Battlezone`
- Extract root: `work/extracted/full_all`
- Texture reports:
  - `notes/texture-header-discovery-report.csv`
  - `notes/texture-conversion-report-full-all-texconv.csv`
- Audio report:
  - `notes/audio-dump-report.csv`
- Model report:
  - `work/preview/model_obj_universal/hskn/hskn-obj-export-report.csv`
- Additional RE summaries:
  - `notes/model_index_priority/model_index_summary.json`
  - `notes/mare_analysis/mare_analysis_summary.json`

All counts in this file are specific to this baseline run and should be treated as "known good for this dataset/date".

## 2) Container And Wrapper Layer

### 2.1 Target archive classes (`AllPcVariants`)

From `work/Battlezone` with `*.pc` and `*.pc_*` matching:

- `228` target archives total
- Extension breakdown:
  - `.pc`: `79`
  - `.pc_textures`: `40`
  - `.pc_entdata`: `38`
  - `.pc_entheader`: `38`
  - locale variants (`.pc_en`, `.pc_fr`, etc.): remaining files

### 2.2 Wrapper signatures observed

Signature of first 8 bytes in targets:

- `AsuraZlb`: `117`
- `Asura   `: `73`
- plain-text `envs\...` signatures: `38` (all `.pc_entheader` pointer files)

`*.pc_entheader` files are plain text pointers to matching `*.pc_entdata`, for example:

- `envs\city01_variation01_mission.pc_entdata`

### 2.3 Placeholder archives

The toolkit treats 12-byte `Asura   ` + `00 00 00 00` files as placeholders:

- Placeholder count in baseline targets: `39`

### 2.4 Native extractor wrapper support

Implemented in `tools/extract-asura-native.py`:

- Supported:
  - `Asura   ` (direct payload)
  - `AsuraZlb` (single zlib wrapper variants)
  - `AsuraZbb` (chunked zlib wrapper)
- Explicitly unsupported (exit code `10`):
  - `AsuraCmp`
  - `xcompress` markers (`0ff512ed`, `0ff512ee`)

## 3) Post-Extraction Asset Inventory

From `work/extracted/full_all`:

- Total files: `27,329`
- Top extensions:
  - `.dat`: `23,107`
  - `.tga`: `2,211`
  - no extension: `1,470`
  - `.wav`: `367`
  - `.dds`: `53`
  - `.bmp`: `50`
- Files in `*_chunk` directories: `23,092`
- Direct (non-`*_chunk`) files: `4,237`

Observed top archive output buckets (by extracted file count):

- `chars__actors_pc`: `8,712`
- `chars__playertanks_pc`: `1,605`
- `envs__3d_frontend_pc`: `763`
- mission/environment archives follow

## 4) Chunk Taxonomy Observed

Unique chunk families observed in extracted data: `50`

Top families by file count:

- `ENTI_chunk`: `7,867`
- `CONA_chunk`: `4,716`
- `FXST_chunk`: `1,384`
- `FXPT_chunk`: `1,384`
- `HSKE_chunk`: `1,023`
- `HSKN_chunk`: `1,023`
- `HSBB_chunk`: `1,023`
- `HCAN_chunk`: `649`
- `SDSM_chunk`: `625`
- `SDEV_chunk`: `386`

Other relevant families include `MARE_chunk`, `TEXT_chunk`, `VTEX_chunk`, `HSND_chunk`, `TXAN_chunk`, etc.

Model index summary cross-check (`notes/model_index_priority/model_index_summary.json`) also reports:

- `HSKN_chunk`: `1023`
- `HSBB_chunk`: `1023`
- `HSKE_chunk`: `1023`
- `HCAN_chunk`: `649`
- `HSKL_chunk`: `290`

## 5) Texture Assets

### 5.1 Observed texture payload behavior

Header-based texture discovery (`--texture-discovery headers`) found:

- `2314` texture candidates
- Source extension mix:
  - `.tga`: `2211`
  - `.dds`: `53`
  - `.bmp`: `50`
- Detected content signature for all staged candidates: `.dds` (`2314/2314`)

Interpretation:

- Many files named `.tga` are DDS-encoded payloads.
- Header-based detection is required for robust conversion and should be preferred over extension-only filtering.

### 5.2 Conversion pipeline behavior

`tools/convert-textures-to-png.ps1`:

- Uses `ffmpeg` for primary conversion
- Can use `texconv` as preferred path or fallback for DDS-heavy cases
- Supports `--skip-existing` behavior

Current baseline conversion report:

- `2314` rows
- Status: all `Skipped` (because outputs already existed)
- Existing PNG outputs in `work/preview/png_all_texconv`: `2314`

## 6) Audio Assets

### 6.1 Header-based audio discovery

`tools/bzg-extractor.py` detects audio by signature:

- `BKHD` -> `.bnk`
- `OggS` -> `.ogg`
- `fLaC` -> `.flac`
- `ID3` or MP3 frame sync -> `.mp3`
- `RIFF` + `WAVE` -> `.wav`
- unknown `RIFF` -> `.wem`

Baseline output (`notes/audio-dump-report.csv`):

- `369` copied files
- Detected extension breakdown:
  - `.wav`: `367`
  - `.mp3`: `2`
- Source extension breakdown:
  - `.wav`: `367`
  - `.dat`: `2`

The two `.dat` files were detected as MP3 payloads:

- `chars__actors_pc/HSKE_chunk/.4786.dat`
- `envs__shield_frozen02_variation01_mission_pc_entdata/ENTI_chunk/.71.dat`

## 7) Model Assets

There are two model extraction backends in toolkit:

1. `HSKN_chunk` parser (`tools/export-hskn-obj-candidates.py`)  
2. Extensionless static-model parser (`tools/export-model-obj-candidates.py`)

### 7.1 `HSKN_chunk` reverse-engineered pattern

Current parser assumptions (validated on many files):

- Model name often read as C-string at offset `0x08`
- Sentinel pattern searched: `FF FF 00 00 00 00`
- `vertex_count` from `sentinel - 0x16` (u16)
- `strip_index_hint` from `sentinel - 0x12` (u16)
- Bounds read near sentinel (`<6f`)
- Vertex block interpreted as packed float triplets (12-byte stride)
- Indices read as u16 strip stream with `0xFFFF` restart

Exporter supports topology decode modes (`strip`, `strip_adaptive`, `fan`, `list`, and record-5 variants), with adaptive edge pruning for cross-mesh bridge suppression.

### 7.2 Extensionless static-model format (non-`HSKN_chunk`)

`tools/export-model-obj-candidates.py` documents/uses this structure:

- Header starts with:
  - `part_count` (u32)
  - `vertex_count` (u32)
  - `index_count` (u32)
  - `tri_count` (u32)
- Expected layout:
  - `header_size = 0x10 + part_count * 0x18 + 0x08`
  - vertex buffer at 40-byte stride per vertex
  - fixed 20-byte gap after vertex block
  - trailing u16 index buffer
- Validation checks:
  - `index_count % 3 == 0`
  - `tri_count == index_count / 3`
  - descriptor index-count sum matches total indices

Position decode remains partially heuristic; exporter tests multiple decode modes (`packed_257`, `packed_258`, `packed_259`, `auto_part_s16`, etc.) and scores geometry quality.

### 7.3 Baseline HSKN export results

From `work/preview/model_obj_universal/hskn/hskn-obj-export-report.csv`:

- Candidates: `1023`
- `ok`: `818` (OBJ written)
- `skipped`: `170` (`below_min_triangles`)
- `parse_failed`: `35` (`no_sentinel`)
- OBJ outputs present: `818`

## 8) Material Linkage (`MARE_chunk`)

`tools/legacy/mare-material-link-analysis.py` and generated outputs indicate:

- `MARE_chunk` fixed-stride records:
  - `RECORD_STRIDE = 0x164`
  - payload-size field expected `0x15C`
- Records include:
  - `material_hash`
  - `link_a`
  - `link_b`
  - `field_10`
  - payload CRC grouping

Baseline summary (`notes/mare_analysis/mare_analysis_summary.json`):

- `mare_record_count`: `4267`
- unique `material_hash`: `507`
- `model_material_usage_rows`: `2426`

This is the strongest current evidence for material indirection from model submesh material IDs into additional effect/material metadata.

## 9) Reproducible End-To-End Commands

### 9.1 Full extraction and asset dump

```powershell
python .\tools\bzg-extractor.py run `
  --tasks extract,textures,models,audio `
  --game-root work/Battlezone `
  --extract-root work/extracted/full_all `
  --logs-root work/logs/batch_all `
  --extractor Native `
  --texture-discovery headers `
  --texture-stage-root work/preview/texture_stage_header `
  --texture-discovery-report notes/texture-header-discovery-report.csv `
  --texture-output-root work/preview/png_all_texconv `
  --texture-report-path notes/texture-conversion-report-full-all-texconv.csv `
  --model-backend hskn `
  --model-discovery headers `
  --model-max-files 0 `
  --audio-discovery headers `
  --audio-output-root work/preview/audio_dump `
  --audio-report-path notes/audio-dump-report.csv `
  --skip-existing
```

### 9.2 Force fresh conversion/dump pass

Use `--no-skip-existing` for a full rewrite:

```powershell
python .\tools\bzg-extractor.py textures --extract-root work/extracted/full_all --texture-discovery headers --no-skip-existing
python .\tools\bzg-extractor.py audio --extract-root work/extracted/full_all --audio-discovery headers --no-skip-existing
```

### 9.3 Quick verification checks

```powershell
# Extracted file count
(Get-ChildItem work/extracted/full_all -Recurse -File | Measure-Object).Count

# PNG/audio/OBJ outputs
(Get-ChildItem work/preview/png_all_texconv -Recurse -File | Measure-Object).Count
(Get-ChildItem work/preview/audio_dump -Recurse -File | Measure-Object).Count
(Get-ChildItem work/preview/model_obj_universal/hskn -Recurse -Filter *.obj -File | Measure-Object).Count
```

## 10) Confidence Levels

- High confidence:
  - Wrapper handling (`Asura`, `AsuraZlb`, `AsuraZbb`)
  - Texture/audio header detection signatures
  - `HSKN_chunk` and extensionless-model structural checks used in exporters
  - `MARE_chunk` fixed record stride and payload-size field
- Medium confidence:
  - Semantic meaning of many non-geometry chunk families (`ENTI`, `CONA`, `FX*`, etc.)
  - Complete material->texture resolution path in engine runtime
- Low confidence / open:
  - Unsupported wrapper decoding (`AsuraCmp`, xcompress)
  - Full semantics for all unknown extensionless binaries

## 11) Known Gaps / Next RE Targets

1. Add native decode for `AsuraCmp` and xcompress wrappers.
2. Build parsers for high-volume non-geometry chunk families (`ENTI_chunk`, `CONA_chunk`) to recover object/entity metadata.
3. Continue formalizing material graph reconstruction from `MARE_chunk` links into deterministic texture assignment.
4. Expand extensionless model decoding to reduce heuristic mode selection and improve UV/material fidelity.

## 12) Binary Layout Diagrams (Offset Maps)

Notation:

- Offsets are hex (`0x...`) unless stated otherwise.
- Integer fields are little-endian unless otherwise noted.
- Some structures are heuristic and marked accordingly.

### 12.1 Wrapper signatures

```text
Asura direct:
  0x00  8 bytes   "Asura   "
  0x08  ...       chunk stream begins

AsuraZlb:
  0x00  8 bytes   "AsuraZlb"
  0x14  ...       zlib payload candidate region
  0x14  u32       candidate compressed size (optional interpretation)
  0x1C  ...       alternative zlib candidate start

AsuraZbb:
  0x00  8 bytes   "AsuraZbb"
  0x0C  u32       full_size (target decompressed size)
  0x10  ...       repeated blocks:
                  u32 zsize, u32 usize, [zsize bytes compressed]
```

`AsuraZlb` and `AsuraZbb` are decoded by trying candidate zlib windows and selecting the candidate where `"Asura   "` appears at best offset (prefer `0x00`, then near start).

### 12.2 Asura payload chunk stream

At resolved payload base (`base_offset`), parser expects:

```text
0x00  8 bytes   "Asura   "
0x08  ...       chunk[0]
```

Each chunk:

```text
chunk +0x00  4 bytes  chunk_name (ASCII-ish, e.g. "RSCF", "ASTS")
chunk +0x04  u32      chunk_size (includes 16-byte chunk header)
chunk +0x08  u32      chunk_ver
chunk +0x0C  u32      dummy/reserved
chunk +0x10  ...      chunk payload
```

Next chunk offset: `chunk_offset += chunk_size`.

### 12.3 `RSCF` payload map

```text
payload +0x00  u32    rtype
payload +0x04  u32    rdummy
payload +0x08  u32    size
payload +0x0C  ...    padded null-terminated name (4-byte stepped reads)
...
chunk_end-size ...    file bytes (data_off = chunk_end - size)
```

Extractor emits `name` file from `[data_off, data_off+size)`.

### 12.4 `ASTS` payload map

```text
payload +0x00  u32    file_count
payload +0x04  u8?    not_archived flag probe if chunk_ver >= 2 (optional)

For each entry:
  name              padded null-terminated string (4-byte block reads)
  +0x00             u8   entry_dummy
  +0x01             u32  size
  if chunk_ver == 0:
      file_off = current cursor (inline data)
      cursor = file_off + size
  else:
      +0x05         u32  file_off (absolute offset in payload buffer)
```

If `not_archived == 1`, entries are listed but not emitted.

### 12.5 `HSKN_chunk` mesh payload (current parser model)

```text
0x08               C-string model name (best effort)
...                search sentinel from >=0x40:
sentinel           FF FF 00 00 00 00

sentinel-0x16      u16 vertex_count
sentinel-0x12      u16 strip_index_hint
sentinel+0x06      6*f32 bounds (min/max axis pairs)
sentinel+0x1E      vertex block (vertex_count * 12 bytes as 3*f32)
vertex_end+0x04    index stream start (u16 sequence)
...                0xFFFF indicates strip restart
```

Topology decode tries strip/list/fan and adaptive variants; triangles are derived from u16 index stream with restart token `0xFFFF`.

### 12.6 Extensionless static model layout (non-`HSKN_chunk`)

```text
0x00  u32  part_count
0x04  u32  vertex_count
0x08  u32  index_count
0x0C  u32  tri_count
0x10  ...  part descriptors (part_count * 0x18 bytes)
            per part: descriptor index count at descriptor+0x0C (u32)
...   +0x08 fixed trailer region in header

header_size            = 0x10 + part_count*0x18 + 0x08
vertex_block           = [header_size, header_size + vertex_count*40)
fixed_gap              = 20 bytes
index_offset_expected  = header_size + vertex_count*40 + 20
index_buffer           = trailing u16 array of length index_count
```

Hard validation constraints used by exporter:

- `1 <= part_count <= 64`
- `index_count % 3 == 0`
- `tri_count == index_count / 3`
- `sum(descriptor_index_counts) == index_count`
- `index_offset == index_offset_expected`
- `max(indices) < vertex_count`

### 12.7 Static model material-hash location (for linkage)

For static models parsed in `mare-material-link-analysis.py`:

```text
0x00  u32 num_submeshes
...
0x14  start of submesh records (24-byte each)
submesh[i] +0x00  u32 material_hash
submesh[i] +0x18  next submesh
```

This `material_hash` is correlated to `MARE_chunk` records.

### 12.8 `MARE_chunk` record table map

From `tools/legacy/mare-material-link-analysis.py`:

- `RECORD_STRIDE = 0x164`
- `RECORD_PAYLOAD_SIZE = 0x15C`

Layout:

```text
0x00                u32 count
...
tail_start          = len(blob) - count*0x164
record[i]           = tail_start + i*0x164

record +0x00        u32 material_hash
record +0x04        u32 rec_size (expected 0x15C)
record +0x08        u32 link_a
record +0x0C        u32 link_b
record +0x10        u32 field_10
record +0x08..+0x163  payload bytes (used for crc grouping)
```

### 12.9 Fast byte-level verification snippets

Use these to inspect candidate files without changing data:

```powershell
# Wrapper signature (first 32 bytes)
Format-Hex -Path .\work\Battlezone\chars\actors.pc -Count 32

# Inspect chunk payload bytes
Format-Hex -Path .\work\extracted\full_all\chars__actors_pc\HSKN_chunk\.3851.dat -Count 128

# Inspect a MARE record source
Format-Hex -Path .\work\extracted\full_all\chars__actors_pc\MARE_chunk\.0.dat -Count 256
```
