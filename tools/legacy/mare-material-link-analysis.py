#!/usr/bin/env python3
"""Analyze Asura MARE material-link records and model material-hash usage.

This script correlates extensionless model submesh material hashes with
MARE_chunk tail records (fixed stride 0x164, payload size field 0x15C).
It is useful for tracing texture/material indirection when direct file-name
references are not present in mesh exports.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


RECORD_STRIDE = 0x164
RECORD_PAYLOAD_SIZE = 0x15C


@dataclass
class MareRecord:
    archive: str
    mare_file: str
    record_index: int
    record_offset: int
    material_hash: int
    link_a: int
    link_b: int
    field_10: int
    payload_crc32: int


@dataclass
class ModelMaterialUse:
    archive: str
    model_name: str
    rel_path: str
    submesh_index: int
    material_hash: int


def rel_to(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def parse_mare_records(path: Path, input_root: Path) -> list[MareRecord]:
    blob = path.read_bytes()
    if len(blob) < 4:
        return []
    count = struct.unpack_from("<I", blob, 0)[0]
    if count <= 0:
        return []
    table_size = count * RECORD_STRIDE
    if table_size > len(blob):
        return []
    start = len(blob) - table_size
    records: list[MareRecord] = []
    archive = path.parts[len(input_root.parts)]
    for i in range(count):
        off = start + i * RECORD_STRIDE
        if off + RECORD_STRIDE > len(blob):
            break
        material_hash, rec_size, link_a, link_b, field_10 = struct.unpack_from("<IIIII", blob, off)
        if rec_size != RECORD_PAYLOAD_SIZE:
            continue
        payload = blob[off + 8 : off + RECORD_STRIDE]
        payload_crc32 = 0
        # zlib crc32 is stable and fast for grouping near-identical records.
        import zlib

        payload_crc32 = zlib.crc32(payload) & 0xFFFFFFFF
        records.append(
            MareRecord(
                archive=archive,
                mare_file=path.name,
                record_index=i,
                record_offset=off,
                material_hash=material_hash,
                link_a=link_a,
                link_b=link_b,
                field_10=field_10,
                payload_crc32=payload_crc32,
            )
        )
    return records


def looks_like_static_model(blob: bytes) -> bool:
    if len(blob) < 64:
        return False
    try:
        num_submeshes, total_verts, total_indices, total_tris = struct.unpack_from("<IIII", blob, 0)
    except struct.error:
        return False
    if num_submeshes <= 0 or num_submeshes > 2048:
        return False
    if total_verts <= 0 or total_verts > 500_000:
        return False
    if total_indices <= 0 or total_indices > 4_000_000:
        return False
    if total_tris > total_indices:
        return False
    header_size = 20 + num_submeshes * 24 + 24
    expected_min = header_size + total_verts * 40 + total_indices * 2
    return expected_min <= len(blob)


def parse_model_material_usage(path: Path, input_root: Path) -> list[ModelMaterialUse]:
    blob = path.read_bytes()
    if not looks_like_static_model(blob):
        return []
    num_submeshes = struct.unpack_from("<I", blob, 0)[0]
    rel = rel_to(path, input_root)
    archive = path.parts[len(input_root.parts)]
    model_name = path.name
    out: list[ModelMaterialUse] = []
    off = 20
    for i in range(num_submeshes):
        material_hash = struct.unpack_from("<I", blob, off)[0]
        out.append(
            ModelMaterialUse(
                archive=archive,
                model_name=model_name,
                rel_path=rel,
                submesh_index=i,
                material_hash=material_hash,
            )
        )
        off += 24
    return out


def parse_keywords(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.strip().lower() for x in text.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze MARE material-link records and model material hash usage.")
    parser.add_argument("--input-root", type=Path, default=Path("work/extracted/full_all"))
    parser.add_argument("--output-dir", type=Path, default=Path("notes/mare_analysis"))
    parser.add_argument("--keywords", default="mine,hopper")
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    keywords = parse_keywords(args.keywords)

    mare_records: list[MareRecord] = []
    for p in sorted(input_root.rglob("MARE_chunk/.*.dat")):
        mare_records.extend(parse_mare_records(p, input_root))

    model_usage: list[ModelMaterialUse] = []
    for archive_dir in sorted(input_root.iterdir()):
        if not archive_dir.is_dir():
            continue
        for child in archive_dir.iterdir():
            if not child.is_file():
                continue
            if child.suffix:
                continue
            model_usage.extend(parse_model_material_usage(child, input_root))

    by_material: dict[int, list[MareRecord]] = defaultdict(list)
    for rec in mare_records:
        by_material[rec.material_hash].append(rec)

    link_a_counts = Counter(rec.link_a for rec in mare_records)
    link_b_counts = Counter(rec.link_b for rec in mare_records)

    mare_csv = output_dir / "mare_records.csv"
    with mare_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "archive",
                "mare_file",
                "record_index",
                "record_offset",
                "material_hash",
                "material_hex",
                "link_a",
                "link_a_hex",
                "link_b",
                "link_b_hex",
                "field_10",
                "field_10_hex",
                "payload_crc32",
                "payload_crc32_hex",
                "link_a_usage_count",
                "link_b_usage_count",
            ],
        )
        writer.writeheader()
        for rec in mare_records:
            writer.writerow(
                {
                    "archive": rec.archive,
                    "mare_file": rec.mare_file,
                    "record_index": rec.record_index,
                    "record_offset": rec.record_offset,
                    "material_hash": rec.material_hash,
                    "material_hex": f"0x{rec.material_hash:08X}",
                    "link_a": rec.link_a,
                    "link_a_hex": f"0x{rec.link_a:08X}",
                    "link_b": rec.link_b,
                    "link_b_hex": f"0x{rec.link_b:08X}",
                    "field_10": rec.field_10,
                    "field_10_hex": f"0x{rec.field_10:08X}",
                    "payload_crc32": rec.payload_crc32,
                    "payload_crc32_hex": f"0x{rec.payload_crc32:08X}",
                    "link_a_usage_count": link_a_counts[rec.link_a],
                    "link_b_usage_count": link_b_counts[rec.link_b],
                }
            )

    usage_csv = output_dir / "model_material_usage.csv"
    with usage_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "archive",
                "model_name",
                "path",
                "submesh_index",
                "material_hash",
                "material_hex",
                "mare_match_count",
                "mare_archives",
            ],
        )
        writer.writeheader()
        for use in model_usage:
            matches = by_material.get(use.material_hash, [])
            writer.writerow(
                {
                    "archive": use.archive,
                    "model_name": use.model_name,
                    "path": use.rel_path,
                    "submesh_index": use.submesh_index,
                    "material_hash": use.material_hash,
                    "material_hex": f"0x{use.material_hash:08X}",
                    "mare_match_count": len(matches),
                    "mare_archives": "|".join(sorted({m.archive for m in matches})) if matches else "",
                }
            )

    focus_rows: list[dict[str, object]] = []
    for use in model_usage:
        model_lc = use.model_name.lower()
        if keywords and not any(k in model_lc for k in keywords):
            continue
        matches = by_material.get(use.material_hash, [])
        if matches:
            for rec in matches:
                focus_rows.append(
                    {
                        "archive": use.archive,
                        "model_name": use.model_name,
                        "path": use.rel_path,
                        "submesh_index": use.submesh_index,
                        "material_hash": use.material_hash,
                        "material_hex": f"0x{use.material_hash:08X}",
                        "mare_archive": rec.archive,
                        "mare_file": rec.mare_file,
                        "mare_record_index": rec.record_index,
                        "link_a": rec.link_a,
                        "link_a_hex": f"0x{rec.link_a:08X}",
                        "link_b": rec.link_b,
                        "link_b_hex": f"0x{rec.link_b:08X}",
                        "link_a_usage_count": link_a_counts[rec.link_a],
                        "link_b_usage_count": link_b_counts[rec.link_b],
                    }
                )
        else:
            focus_rows.append(
                {
                    "archive": use.archive,
                    "model_name": use.model_name,
                    "path": use.rel_path,
                    "submesh_index": use.submesh_index,
                    "material_hash": use.material_hash,
                    "material_hex": f"0x{use.material_hash:08X}",
                    "mare_archive": "",
                    "mare_file": "",
                    "mare_record_index": "",
                    "link_a": "",
                    "link_a_hex": "",
                    "link_b": "",
                    "link_b_hex": "",
                    "link_a_usage_count": "",
                    "link_b_usage_count": "",
                }
            )

    focus_csv = output_dir / "focus_material_links.csv"
    with focus_csv.open("w", newline="", encoding="utf-8") as f:
        keys = [
            "archive",
            "model_name",
            "path",
            "submesh_index",
            "material_hash",
            "material_hex",
            "mare_archive",
            "mare_file",
            "mare_record_index",
            "link_a",
            "link_a_hex",
            "link_b",
            "link_b_hex",
            "link_a_usage_count",
            "link_b_usage_count",
        ]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(focus_rows)

    summary = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "keywords": keywords,
        "mare_record_count": len(mare_records),
        "unique_mare_material_hashes": len(by_material),
        "model_material_usage_rows": len(model_usage),
        "focus_rows": len(focus_rows),
        "top_link_a": [
            {"value": v, "hex": f"0x{v:08X}", "count": c}
            for v, c in link_a_counts.most_common(20)
        ],
        "top_link_b": [
            {"value": v, "hex": f"0x{v:08X}", "count": c}
            for v, c in link_b_counts.most_common(20)
        ],
        "mare_csv": str(mare_csv),
        "usage_csv": str(usage_csv),
        "focus_csv": str(focus_csv),
    }
    summary_json = output_dir / "mare_analysis_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

