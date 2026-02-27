#!/usr/bin/env python
"""Index Asura model-related chunk data with a tank/enemy/cockpit focus."""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


CHUNK_FAMILIES = ("HSKN_chunk", "HSKE_chunk", "HSBB_chunk", "HSKL_chunk", "HCAN_chunk")
DEFAULT_KEYWORDS = (
    "tank",
    "enemy",
    "drone",
    "cockpit",
    "weapon",
    "turret",
    "cannon",
    "missile",
    "laser",
    "bike",
    "ufo",
    "boss",
    "spider",
    "nemesis",
)
ASCII_RE = re.compile(rb"[A-Za-z0-9_#/\-:.]{3,}")
CHUNK_ID_RE = re.compile(r"^\.(\d+)\.dat$", re.IGNORECASE)


@dataclass
class ChunkRecord:
    archive: str
    chunk: str
    chunk_id: int
    file_size: int
    primary_name: str
    alt_names: list[str] = field(default_factory=list)
    hske_link: int | None = None
    u32_0: int | None = None
    u32_1: int | None = None
    rel_path: str = ""


@dataclass
class LinkEntry:
    archive: str
    chunk_id: int
    chunks_present: set[str] = field(default_factory=set)
    names: set[str] = field(default_factory=set)
    hske_link: int | None = None
    max_chunk_size: int = 0
    paths: list[str] = field(default_factory=list)


def rel_to(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def read_u32_le(data: bytes, offset: int) -> int | None:
    if offset + 4 > len(data):
        return None
    return struct.unpack_from("<I", data, offset)[0]


def read_cstring(data: bytes, start: int, max_len: int = 96) -> str:
    if start >= len(data):
        return ""
    end = min(len(data), start + max_len)
    raw = data[start:end].split(b"\x00", 1)[0]
    if not raw:
        return ""
    if any(b < 32 or b > 126 for b in raw):
        return ""
    return raw.decode("ascii", errors="ignore")


def extract_ascii_strings(data: bytes, limit: int = 12) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for match in ASCII_RE.finditer(data):
        value = match.group(0).decode("ascii", errors="ignore")
        if len(value) < 3:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= limit:
            break
    return out


def is_name_like(value: str) -> bool:
    if len(value) < 3 or len(value) > 80:
        return False
    if "\\" in value or "/" in value or ":" in value:
        return False
    if not any(ch.isalpha() for ch in value):
        return False
    return True


def pick_primary_name(strings: list[str]) -> str:
    if not strings:
        return ""
    for value in strings:
        low = value.lower()
        if "_" in value and any(c.isalpha() for c in value):
            return value
        if any(k in low for k in DEFAULT_KEYWORDS):
            return value
    return strings[0]


def parse_chunk_file(path: Path, input_root: Path) -> ChunkRecord | None:
    chunk_name = path.parent.name
    if chunk_name not in CHUNK_FAMILIES:
        return None

    chunk_match = CHUNK_ID_RE.match(path.name)
    if not chunk_match:
        return None

    data = path.read_bytes()
    rel = path.relative_to(input_root)
    archive = rel.parts[0]
    chunk_id = int(chunk_match.group(1))

    strings = [s for s in extract_ascii_strings(data) if is_name_like(s)]
    offset_name = read_cstring(data, 8)
    if offset_name and is_name_like(offset_name) and offset_name not in strings:
        strings.insert(0, offset_name)

    primary_name = pick_primary_name(strings)
    alt_names = [s for s in strings if s != primary_name][:6]

    record = ChunkRecord(
        archive=archive,
        chunk=chunk_name,
        chunk_id=chunk_id,
        file_size=path.stat().st_size,
        primary_name=primary_name,
        alt_names=alt_names,
        rel_path=rel_to(path, input_root),
    )

    if chunk_name == "HSKE_chunk":
        record.hske_link = read_u32_le(data, 0)
    if chunk_name in ("HSKN_chunk", "HCAN_chunk"):
        record.u32_0 = read_u32_le(data, 0)
        record.u32_1 = read_u32_le(data, 4)

    return record


def find_direct_extensionless(input_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for archive_dir in input_root.iterdir():
        if not archive_dir.is_dir():
            continue
        for child in archive_dir.iterdir():
            if not child.is_file():
                continue
            if child.suffix:
                continue
            rows.append(
                {
                    "archive": archive_dir.name,
                    "name": child.name,
                    "size": child.stat().st_size,
                    "rel_path": rel_to(child, input_root),
                }
            )
    return rows


def build_link_map(records: Iterable[ChunkRecord]) -> list[LinkEntry]:
    by_key: dict[tuple[str, int], LinkEntry] = {}
    for rec in records:
        key = (rec.archive, rec.chunk_id)
        if key not in by_key:
            by_key[key] = LinkEntry(archive=rec.archive, chunk_id=rec.chunk_id)
        entry = by_key[key]
        entry.chunks_present.add(rec.chunk)
        entry.max_chunk_size = max(entry.max_chunk_size, rec.file_size)
        entry.paths.append(rec.rel_path)
        if rec.primary_name:
            entry.names.add(rec.primary_name)
        entry.names.update(n for n in rec.alt_names if n)
        if rec.chunk == "HSKE_chunk":
            entry.hske_link = rec.hske_link
    return sorted(by_key.values(), key=lambda e: (e.archive, e.chunk_id))


def build_name_groups(records: Iterable[ChunkRecord]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], dict[str, object]] = {}
    for rec in records:
        if not rec.primary_name or not is_name_like(rec.primary_name):
            continue
        key = (rec.archive, rec.primary_name)
        if key not in groups:
            groups[key] = {
                "Archive": rec.archive,
                "Name": rec.primary_name,
                "ChunkTypes": set(),
                "ChunkIds": set(),
                "RecordCount": 0,
                "MaxChunkSize": 0,
                "SamplePaths": [],
            }
        g = groups[key]
        g["ChunkTypes"].add(rec.chunk)
        g["ChunkIds"].add(rec.chunk_id)
        g["RecordCount"] += 1
        g["MaxChunkSize"] = max(g["MaxChunkSize"], rec.file_size)
        if len(g["SamplePaths"]) < 4:
            g["SamplePaths"].append(rec.rel_path)

    out: list[dict[str, object]] = []
    for (_, _), g in groups.items():
        out.append(
            {
                "Archive": g["Archive"],
                "Name": g["Name"],
                "RecordCount": g["RecordCount"],
                "ChunkTypeCount": len(g["ChunkTypes"]),
                "ChunkTypes": "|".join(sorted(g["ChunkTypes"])),
                "ChunkIdCount": len(g["ChunkIds"]),
                "ChunkIds": "|".join(str(v) for v in sorted(g["ChunkIds"])[:20]),
                "MaxChunkSize": g["MaxChunkSize"],
                "SamplePaths": "|".join(g["SamplePaths"]),
            }
        )
    return sorted(out, key=lambda r: (r["Archive"], r["Name"]))


def score_name(name: str, archive: str, direct_count: int, max_direct_size: int, chunk_types: set[str], keywords: tuple[str, ...]) -> int:
    low = name.lower()
    score = 0
    if any(k in low for k in keywords):
        score += 10
    if archive.startswith("chars__"):
        score += 4
    elif archive.startswith("envs__"):
        score += 1
    score += min(len(chunk_types), 6)
    if direct_count > 0:
        score += 2
    if max_direct_size >= 200_000:
        score += 2
    return score


def build_focus_candidates(
    name_groups: list[dict[str, object]],
    direct_files: list[dict[str, object]],
    keywords: tuple[str, ...],
) -> list[dict[str, object]]:
    by_name_archive: dict[tuple[str, str], dict[str, object]] = {}

    for row in direct_files:
        key = (str(row["archive"]), str(row["name"]))
        if key not in by_name_archive:
            by_name_archive[key] = {
                "archive": row["archive"],
                "name": row["name"],
                "chunk_types": set(),
                "chunk_ids": set(),
                "direct_count": 0,
                "max_direct_size": 0,
                "sample_paths": [],
            }
        entry = by_name_archive[key]
        entry["direct_count"] += 1
        entry["max_direct_size"] = max(entry["max_direct_size"], int(row["size"]))
        if len(entry["sample_paths"]) < 4:
            entry["sample_paths"].append(str(row["rel_path"]))

    for group in name_groups:
        archive = str(group["Archive"])
        name = str(group["Name"])
        key = (archive, name)
        if key not in by_name_archive:
            by_name_archive[key] = {
                "archive": archive,
                "name": name,
                "chunk_types": set(),
                "chunk_ids": set(),
                "direct_count": 0,
                "max_direct_size": 0,
                "sample_paths": [],
            }
        entry = by_name_archive[key]
        entry["chunk_types"].update(
            t for t in str(group["ChunkTypes"]).split("|") if t
        )
        entry["chunk_ids"].update(
            int(v) for v in str(group["ChunkIds"]).split("|") if v.strip().isdigit()
        )
        for path in str(group["SamplePaths"]).split("|"):
            if path and len(entry["sample_paths"]) < 4:
                entry["sample_paths"].append(path)

    out: list[dict[str, object]] = []
    for (_, _), entry in by_name_archive.items():
        name = str(entry["name"])
        archive = str(entry["archive"])
        if not is_name_like(name):
            continue
        chunk_types = set(entry["chunk_types"])
        direct_count = int(entry["direct_count"])
        max_direct_size = int(entry["max_direct_size"])
        score = score_name(name, archive, direct_count, max_direct_size, chunk_types, keywords)
        low = name.lower()
        keyword_hit = any(k in low for k in keywords)
        if not keyword_hit and score < 12:
            continue
        out.append(
            {
                "Archive": archive,
                "Name": name,
                "Score": score,
                "KeywordHit": keyword_hit,
                "DirectFileCount": direct_count,
                "MaxDirectFileSize": max_direct_size,
                "ChunkTypeCount": len(chunk_types),
                "ChunkTypes": "|".join(sorted(chunk_types)),
                "ChunkIdCount": len(entry["chunk_ids"]),
                "ChunkIds": "|".join(str(v) for v in sorted(entry["chunk_ids"])[:20]),
                "SamplePaths": "|".join(entry["sample_paths"][:4]),
            }
        )

    return sorted(out, key=lambda r: (-int(r["Score"]), r["Archive"], r["Name"]))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build model-focused chunk index from extracted Asura data.")
    parser.add_argument("--input-root", default="work/extracted/full_all", help="Root folder containing extracted archive folders.")
    parser.add_argument("--output-dir", default="notes/model_index", help="Directory for CSV/JSON output.")
    parser.add_argument(
        "--keywords",
        default=",".join(DEFAULT_KEYWORDS),
        help="Comma-separated keywords for high-priority model assets.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    keywords = tuple(k.strip().lower() for k in args.keywords.split(",") if k.strip())

    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")

    chunk_records: list[ChunkRecord] = []
    for chunk_name in CHUNK_FAMILIES:
        for path in input_root.rglob(f"{chunk_name}/.*.dat"):
            record = parse_chunk_file(path, input_root)
            if record is not None:
                chunk_records.append(record)

    if not chunk_records:
        raise SystemExit(f"No model chunk records found under: {input_root}")

    links = build_link_map(chunk_records)
    name_groups = build_name_groups(chunk_records)
    direct_files = find_direct_extensionless(input_root)
    focus = build_focus_candidates(name_groups, direct_files, keywords)
    keyword_focus = [row for row in focus if bool(row["KeywordHit"])]

    chunk_rows = []
    for rec in sorted(chunk_records, key=lambda r: (r.archive, r.chunk, r.chunk_id)):
        chunk_rows.append(
            {
                "Archive": rec.archive,
                "Chunk": rec.chunk,
                "ChunkId": rec.chunk_id,
                "FileSize": rec.file_size,
                "PrimaryName": rec.primary_name,
                "AltNames": "|".join(rec.alt_names),
                "HSKE_Link": "" if rec.hske_link is None else rec.hske_link,
                "U32_0": "" if rec.u32_0 is None else rec.u32_0,
                "U32_1": "" if rec.u32_1 is None else rec.u32_1,
                "Path": rec.rel_path,
            }
        )

    link_rows = []
    for link in links:
        link_rows.append(
            {
                "Archive": link.archive,
                "ChunkId": link.chunk_id,
                "ChunksPresent": "|".join(sorted(link.chunks_present)),
                "NameCount": len(link.names),
                "Names": "|".join(sorted(link.names)[:10]),
                "HSKE_Link": "" if link.hske_link is None else link.hske_link,
                "MaxChunkSize": link.max_chunk_size,
                "SamplePaths": "|".join(link.paths[:4]),
            }
        )

    direct_rows = []
    for row in sorted(direct_files, key=lambda r: (str(r["archive"]), str(r["name"]))):
        direct_rows.append(
            {
                "Archive": row["archive"],
                "Name": row["name"],
                "FileSize": row["size"],
                "Path": row["rel_path"],
            }
        )

    chunk_csv = output_dir / "model_chunk_records.csv"
    link_csv = output_dir / "model_id_link_map.csv"
    direct_csv = output_dir / "model_direct_files.csv"
    name_group_csv = output_dir / "model_name_groups.csv"
    focus_csv = output_dir / "model_focus_candidates.csv"
    keyword_focus_csv = output_dir / "model_focus_keyword_hits.csv"
    summary_json = output_dir / "model_index_summary.json"

    write_csv(
        chunk_csv,
        chunk_rows,
        ["Archive", "Chunk", "ChunkId", "FileSize", "PrimaryName", "AltNames", "HSKE_Link", "U32_0", "U32_1", "Path"],
    )
    write_csv(
        link_csv,
        link_rows,
        ["Archive", "ChunkId", "ChunksPresent", "NameCount", "Names", "HSKE_Link", "MaxChunkSize", "SamplePaths"],
    )
    write_csv(
        direct_csv,
        direct_rows,
        ["Archive", "Name", "FileSize", "Path"],
    )
    write_csv(
        name_group_csv,
        name_groups,
        [
            "Archive",
            "Name",
            "RecordCount",
            "ChunkTypeCount",
            "ChunkTypes",
            "ChunkIdCount",
            "ChunkIds",
            "MaxChunkSize",
            "SamplePaths",
        ],
    )
    write_csv(
        focus_csv,
        focus,
        [
            "Archive",
            "Name",
            "Score",
            "KeywordHit",
            "DirectFileCount",
            "MaxDirectFileSize",
            "ChunkTypeCount",
            "ChunkTypes",
            "ChunkIdCount",
            "ChunkIds",
            "SamplePaths",
        ],
    )
    write_csv(
        keyword_focus_csv,
        keyword_focus,
        [
            "Archive",
            "Name",
            "Score",
            "KeywordHit",
            "DirectFileCount",
            "MaxDirectFileSize",
            "ChunkTypeCount",
            "ChunkTypes",
            "ChunkIdCount",
            "ChunkIds",
            "SamplePaths",
        ],
    )

    chunk_counts = Counter(rec.chunk for rec in chunk_records)
    summary = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "keywords": list(keywords),
        "chunk_record_count": len(chunk_records),
        "link_record_count": len(links),
        "name_group_count": len(name_groups),
        "direct_file_count": len(direct_rows),
        "focus_candidate_count": len(focus),
        "keyword_focus_count": len(keyword_focus),
        "chunk_counts": dict(sorted(chunk_counts.items())),
        "top_focus_candidates": focus[:25],
        "top_keyword_focus": keyword_focus[:25],
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {chunk_csv}")
    print(f"Wrote: {link_csv}")
    print(f"Wrote: {direct_csv}")
    print(f"Wrote: {name_group_csv}")
    print(f"Wrote: {focus_csv}")
    print(f"Wrote: {keyword_focus_csv}")
    print(f"Wrote: {summary_json}")
    print(
        "Summary: "
        f"chunk_records={len(chunk_records)}, links={len(links)}, name_groups={len(name_groups)}, "
        f"direct_files={len(direct_rows)}, focus_candidates={len(focus)}, keyword_focus={len(keyword_focus)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
