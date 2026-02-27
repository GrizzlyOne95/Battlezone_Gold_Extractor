#!/usr/bin/env python3
"""Native Asura archive extractor for Battlezone Gold research.

Supported container wrappers:
- Asura    (direct archive payload)
- AsuraZlb (single zlib-compressed payload)
- AsuraZbb (chunked zlib-compressed payload)

Unsupported wrappers currently return exit code 10:
- AsuraCmp (huffboh)
- xcompress signatures used by some other Asura titles
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


EXIT_OK = 0
EXIT_UNSUPPORTED = 10
EXIT_PARSE_ERROR = 11
EXIT_IO_ERROR = 12

SIG_ASURA = b"Asura   "
SIG_ASURAZLB = b"AsuraZlb"
SIG_ASURAZBB = b"AsuraZbb"
SIG_ASURACMP = b"AsuraCmp"
SIG_XCOMPRESS_1 = bytes.fromhex("0ff512ed")
SIG_XCOMPRESS_2 = bytes.fromhex("0ff512ee")

INVALID_WIN_CHARS = re.compile(r'[<>:"|?*]')


class UnsupportedFormatError(RuntimeError):
    pass


class ParseError(RuntimeError):
    pass


@dataclass
class ExtractSummary:
    input_path: str
    output_dir: str
    signature: str
    wrapper: str
    endian: str = "little"
    base_offset: int = 0
    extracted_files: int = 0
    extracted_chunk_files: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def total_outputs(self) -> int:
        return self.extracted_files + self.extracted_chunk_files


def read_u32(data: bytes, offset: int, endian: str) -> int:
    if offset + 4 > len(data):
        raise ParseError(f"Out of bounds u32 read at 0x{offset:X}")
    return struct.unpack_from(f"{endian}I", data, offset)[0]


def decode_chunk_name(raw: bytes) -> str:
    if len(raw) != 4:
        return "UNKN"
    text = raw.decode("ascii", errors="replace")
    if all(32 <= b <= 126 for b in raw):
        return text
    return "UNKN"


def sanitize_part(part: str) -> str:
    cleaned = INVALID_WIN_CHARS.sub("_", part).strip()
    if cleaned in {"", ".", ".."}:
        return "_"
    return cleaned


def safe_rel_path(raw_name: str) -> Path:
    normalized = raw_name.replace("\\", "/").strip().lstrip("/")
    p = PurePosixPath(normalized)
    parts = [sanitize_part(x) for x in p.parts if x not in {"", ".", ".."}]
    if not parts:
        parts = ["_unnamed"]
    return Path(*parts)


def write_entry(
    data: bytes,
    output_dir: Path,
    rel_name: str,
    offset: int,
    size: int,
) -> bool:
    if size <= 0:
        return False
    if offset < 0 or offset + size > len(data):
        return False
    out_rel = safe_rel_path(rel_name)
    out_path = output_dir / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data[offset : offset + size])
    return True


def read_padded_name(data: bytes, cursor: int, end: int) -> tuple[str, int]:
    chunks: list[bytes] = []
    while cursor + 4 <= end:
        block = data[cursor : cursor + 4]
        chunks.append(block)
        cursor += 4
        if 0 in block:
            break
    raw = b"".join(chunks)
    if not raw:
        return "", cursor
    name = raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
    return name, cursor


def detect_endian(payload: bytes, base_offset: int) -> str:
    # BMS heuristic: inspect first chunk size at base+0x8 and verify boundaries.
    probe = base_offset + 8
    if probe + 8 > len(payload):
        return "<"

    chunk_sz_le = struct.unpack_from("<I", payload, probe + 4)[0]
    chunk_sz_be = struct.unpack_from(">I", payload, probe + 4)[0]
    end_limit = len(payload)

    le_valid = 0 < chunk_sz_le <= (end_limit - probe)
    be_valid = 0 < chunk_sz_be <= (end_limit - probe)

    if le_valid and not be_valid:
        return "<"
    if be_valid and not le_valid:
        return ">"
    if le_valid:
        return "<"
    return ">"


def try_zlib_decompress(blob: bytes) -> bytes | None:
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
        try:
            return zlib.decompress(blob, wbits=wbits)
        except zlib.error:
            pass
    return None


def pick_best_decompressed(candidates: list[tuple[str, bytes]]) -> tuple[bytes, int]:
    if not candidates:
        raise ParseError("No valid zlib decompression candidate.")

    scored: list[tuple[int, int, bytes]] = []
    for _label, data in candidates:
        pos = data.find(SIG_ASURA)
        if pos == 0:
            score = 0
        elif 0 <= pos <= 0x40:
            score = 1
        elif pos >= 0:
            score = 2
        else:
            score = 3
            pos = 0
        scored.append((score, pos, data))

    scored.sort(key=lambda x: (x[0], x[1], -len(x[2])))
    best = scored[0]
    return best[2], best[1]


def unwrap_asurazlb(data: bytes) -> tuple[bytes, int]:
    if len(data) < 0x1C:
        raise ParseError("AsuraZlb payload too small.")

    offset = 0x14
    candidates: list[tuple[str, bytes]] = []

    direct = try_zlib_decompress(data[offset:])
    if direct is not None:
        candidates.append(("offset_0x14", direct))

    if offset + 8 <= len(data):
        zsize = struct.unpack_from("<I", data, offset)[0]
        start = offset + 8
        if 0 < zsize <= len(data) - start:
            dec = try_zlib_decompress(data[start : start + zsize])
            if dec is not None:
                candidates.append(("offset_0x1C_zsize", dec))

        dec2 = try_zlib_decompress(data[start:])
        if dec2 is not None:
            candidates.append(("offset_0x1C_to_end", dec2))

    return pick_best_decompressed(candidates)


def unwrap_asurazbb(data: bytes) -> tuple[bytes, int]:
    if len(data) < 16:
        raise ParseError("AsuraZbb payload too small.")

    full_size = struct.unpack_from("<I", data, 12)[0]
    cursor = 16
    out = bytearray()

    while cursor + 8 <= len(data):
        zsize, usize = struct.unpack_from("<II", data, cursor)
        cursor += 8
        if zsize <= 0:
            break
        if cursor + zsize > len(data):
            break
        comp = data[cursor : cursor + zsize]
        cursor += zsize

        dec = try_zlib_decompress(comp)
        if dec is None:
            raise ParseError("Failed to decompress AsuraZbb chunk.")
        out.extend(dec)

        if full_size > 0 and len(out) >= full_size:
            break

    if full_size > 0 and len(out) > full_size:
        out = out[:full_size]
    if not out:
        raise ParseError("No data produced from AsuraZbb chunks.")

    raw = bytes(out)
    pos = raw.find(SIG_ASURA)
    if pos >= 0:
        return raw, pos

    # Some variants appear to expect a virtual 0x14-byte base offset.
    prefixed = (b"\x00" * 0x14) + raw
    pos2 = prefixed.find(SIG_ASURA)
    if pos2 >= 0:
        return prefixed, pos2

    raise ParseError("Asura signature not found after AsuraZbb decompression.")


def unwrap_input(data: bytes) -> tuple[bytes, int, str]:
    sig8 = data[:8]
    sig4 = data[:4]

    if sig8 == SIG_ASURA:
        return data, 0, "Asura"
    if sig8 == SIG_ASURAZLB:
        payload, base = unwrap_asurazlb(data)
        return payload, base, "AsuraZlb"
    if sig8 == SIG_ASURAZBB:
        payload, base = unwrap_asurazbb(data)
        return payload, base, "AsuraZbb"
    if sig8 == SIG_ASURACMP:
        raise UnsupportedFormatError("AsuraCmp (huffboh) is not supported by native extractor yet.")
    if sig4 in {SIG_XCOMPRESS_1, SIG_XCOMPRESS_2}:
        raise UnsupportedFormatError("xcompress wrapper is not supported by native extractor yet.")

    raise UnsupportedFormatError("Unknown Asura wrapper/signature.")


def extract_archive_payload(
    payload: bytes,
    base_offset: int,
    output_dir: Path,
    extract_chunk_blobs: bool,
    summary: ExtractSummary,
) -> None:
    if base_offset + 8 > len(payload):
        raise ParseError("Base offset is outside payload.")
    if payload[base_offset : base_offset + 8] != SIG_ASURA:
        raise ParseError("Asura archive signature not found at resolved base offset.")

    endian = detect_endian(payload, base_offset)
    summary.endian = "little" if endian == "<" else "big"
    summary.base_offset = base_offset

    chunk_offset = base_offset + 8
    chunk_index = 0

    while chunk_offset + 16 <= len(payload):
        chunk_raw = payload[chunk_offset : chunk_offset + 4]
        chunk_name = decode_chunk_name(chunk_raw)
        chunk_size = read_u32(payload, chunk_offset + 4, endian)
        chunk_ver = read_u32(payload, chunk_offset + 8, endian)
        _dummy = read_u32(payload, chunk_offset + 12, endian)

        if chunk_size <= 16:
            summary.warnings.append(f"Chunk {chunk_index}: invalid size {chunk_size}, stopping parse.")
            break
        chunk_end = chunk_offset + chunk_size
        if chunk_end > len(payload):
            summary.warnings.append(
                f"Chunk {chunk_index}: size overruns payload (off=0x{chunk_offset:X}, size={chunk_size}), stopping parse."
            )
            break

        cursor = chunk_offset + 16

        if chunk_name == "RSCF":
            if cursor + 12 > chunk_end:
                summary.warnings.append(f"RSCF chunk {chunk_index}: truncated header.")
            else:
                _rtype = read_u32(payload, cursor, endian)
                _rdummy = read_u32(payload, cursor + 4, endian)
                size = read_u32(payload, cursor + 8, endian)
                cursor += 12
                name, _ = read_padded_name(payload, cursor, chunk_end)
                if not name:
                    name = f"RSCF_{chunk_index:04d}.bin"
                data_off = chunk_end - size
                if write_entry(payload, output_dir, name, data_off, size):
                    summary.extracted_files += 1
                else:
                    summary.warnings.append(
                        f"RSCF chunk {chunk_index}: invalid file bounds for '{name}' (off=0x{data_off:X}, size={size})."
                    )

        elif chunk_name == "ASTS":
            if cursor + 4 > chunk_end:
                summary.warnings.append(f"ASTS chunk {chunk_index}: missing file count.")
            else:
                file_count = read_u32(payload, cursor, endian)
                cursor += 4
                not_archived = False

                if chunk_ver >= 2 and cursor < chunk_end:
                    probe = payload[cursor]
                    if probe in (0, 1):
                        not_archived = probe == 1
                        cursor += 1

                for i in range(file_count):
                    if cursor >= chunk_end:
                        summary.warnings.append(f"ASTS chunk {chunk_index}: file table truncated at entry {i}.")
                        break

                    name, cursor = read_padded_name(payload, cursor, chunk_end)
                    if not name:
                        name = f"ASTS_{chunk_index:04d}_{i:04d}.bin"

                    if cursor + 5 > chunk_end:
                        summary.warnings.append(
                            f"ASTS chunk {chunk_index}: truncated entry header for '{name}'."
                        )
                        break

                    _entry_dummy = payload[cursor]
                    cursor += 1
                    size = read_u32(payload, cursor, endian)
                    cursor += 4

                    if chunk_ver == 0:
                        file_off = cursor
                    else:
                        if cursor + 4 > chunk_end:
                            summary.warnings.append(
                                f"ASTS chunk {chunk_index}: missing offset for '{name}'."
                            )
                            break
                        file_off = read_u32(payload, cursor, endian)
                        cursor += 4

                    if not not_archived:
                        if write_entry(payload, output_dir, name, file_off, size):
                            summary.extracted_files += 1
                        else:
                            summary.warnings.append(
                                f"ASTS chunk {chunk_index}: invalid bounds for '{name}' (off=0x{file_off:X}, size={size})."
                            )

                    if chunk_ver == 0:
                        cursor = file_off + size

        else:
            if extract_chunk_blobs:
                rel_name = f"{chunk_name}_chunk/.{chunk_index}.dat"
                size = chunk_end - cursor
                if write_entry(payload, output_dir, rel_name, cursor, size):
                    summary.extracted_chunk_files += 1
                else:
                    summary.warnings.append(
                        f"Chunk {chunk_index} ({chunk_name}): failed to emit raw chunk payload."
                    )

        chunk_index += 1
        chunk_offset = chunk_end


def main() -> int:
    parser = argparse.ArgumentParser(description="Native extractor for Asura archives.")
    parser.add_argument("--input", required=True, type=Path, help="Path to archive file (.pc, .asr, etc).")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write extracted files.")
    parser.add_argument(
        "--no-chunk-blobs",
        action="store_true",
        help="Skip fallback extraction of unknown chunk payloads into <chunk>_chunk/.N.dat files.",
    )
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional path for JSON summary output.")
    args = parser.parse_args()

    try:
        raw = args.input.read_bytes()
    except OSError as exc:
        print(f"I/O error reading input: {exc}", file=sys.stderr)
        return EXIT_IO_ERROR

    sig = raw[:8]
    sig_txt = sig.decode("ascii", errors="replace")
    summary = ExtractSummary(
        input_path=str(args.input.resolve()),
        output_dir=str(args.output_dir.resolve()),
        signature=sig_txt,
        wrapper=sig_txt,
    )

    if len(raw) == 12 and raw[:8] == SIG_ASURA and raw[8:] == b"\x00\x00\x00\x00":
        summary.warnings.append("Placeholder stub archive (12-byte Asura marker).")
        out_json = json.dumps(summary.__dict__, indent=2)
        print(out_json)
        if args.summary_json:
            args.summary_json.write_text(out_json, encoding="utf-8")
        return EXIT_OK

    try:
        payload, base_offset, wrapper = unwrap_input(raw)
        summary.wrapper = wrapper
    except UnsupportedFormatError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_UNSUPPORTED
    except ParseError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_PARSE_ERROR

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        extract_archive_payload(
            payload=payload,
            base_offset=base_offset,
            output_dir=args.output_dir,
            extract_chunk_blobs=not args.no_chunk_blobs,
            summary=summary,
        )
    except OSError as exc:
        print(f"I/O error writing outputs: {exc}", file=sys.stderr)
        return EXIT_IO_ERROR
    except ParseError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_PARSE_ERROR

    out_json = json.dumps(summary.__dict__, indent=2)
    print(out_json)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(out_json, encoding="utf-8")

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

