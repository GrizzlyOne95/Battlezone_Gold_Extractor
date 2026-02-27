#!/usr/bin/env python3
"""Find likely texture files for extensionless static model assets.

This tool does not assume plain-text texture references exist in model blobs.
Instead it combines:
1) model submesh material hashes
2) MARE material-link records
3) peer model names sharing the same material hashes
4) archive-local texture inventories
to build a ranked shortlist of likely texture files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


RECORD_STRIDE = 0x164
RECORD_PAYLOAD_SIZE = 0x15C
TEXTURE_EXTS = {".tga", ".dds", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
STOP_TOKENS = {
    "l0",
    "l1",
    "l2",
    "l3",
    "l4",
    "l5",
    "h0",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "chunk",
    "chunks",
    "destroyed",
    "piece",
    "markpoints",
    "markers",
    "collision",
    "col",
    "proxy",
    "gun",
    "tank",
    "vehicle",
    "vehicles",
    "object",
    "objects",
    "icon",
    "icons",
    "panel",
    "shield",
    "component",
    "interior",
    "exterior",
    "body",
    "front",
    "rear",
    "left",
    "right",
}


@dataclass(frozen=True)
class ModelKey:
    archive: str
    model_name: str
    rel_path: str


@dataclass(frozen=True)
class MareRecord:
    archive: str
    material_hash: int
    link_a: int
    link_b: int
    record_index: int


def rel_to(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def tokenize(text: str) -> set[str]:
    parts = re.split(r"[^a-z0-9]+", text.lower())
    out: set[str] = set()
    for token in parts:
        if not token or token in STOP_TOKENS:
            continue
        if token.isdigit():
            continue
        if len(token) < 3:
            continue
        out.add(token)
    return out


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


def parse_model_materials(path: Path, input_root: Path) -> tuple[ModelKey, list[int]] | None:
    blob = path.read_bytes()
    if not looks_like_static_model(blob):
        return None
    num_submeshes = struct.unpack_from("<I", blob, 0)[0]
    rel = rel_to(path, input_root)
    archive = path.parts[len(input_root.parts)]
    key = ModelKey(archive=archive, model_name=path.name, rel_path=rel)
    mats: list[int] = []
    off = 20
    for _ in range(num_submeshes):
        mats.append(struct.unpack_from("<I", blob, off)[0])
        off += 24
    return key, mats


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
    archive = path.parts[len(input_root.parts)]
    out: list[MareRecord] = []
    for i in range(count):
        off = start + i * RECORD_STRIDE
        if off + RECORD_STRIDE > len(blob):
            break
        material_hash, rec_size, link_a, link_b, _field_10 = struct.unpack_from("<IIIII", blob, off)
        if rec_size != RECORD_PAYLOAD_SIZE:
            continue
        out.append(
            MareRecord(
                archive=archive,
                material_hash=material_hash,
                link_a=link_a,
                link_b=link_b,
                record_index=i,
            )
        )
    return out


def score_texture(
    texture_rel: str,
    model_tokens: set[str],
    peer_tokens: set[str],
) -> tuple[int, list[str]]:
    rel_lc = texture_rel.lower()
    score = 0
    reasons: list[str] = []
    is_vehicle_path = "/graphics/objects/vehicles/" in rel_lc

    if is_vehicle_path:
        score += 25
        reasons.append("vehicles_path")
    if "/graphics/menus/icons/" in rel_lc:
        score -= 140
        reasons.append("icon_path_penalty")
    if "/graphics/objects/effects/" in rel_lc:
        score -= 20
        reasons.append("effects_penalty")

    basename = Path(rel_lc).stem
    if "enemies" in basename:
        score += 35
        reasons.append("enemy_atlas_name")
    if "vehicle" in basename or "vehicles" in basename:
        score += 20
        reasons.append("vehicle_atlas_name")

    token_hits = 0
    for token in model_tokens:
        if token in rel_lc:
            token_hits += 1
    if token_hits:
        score += 90 + token_hits * 15
        reasons.append(f"model_token_hits={token_hits}")

    peer_hits = 0
    for token in peer_tokens:
        if token in rel_lc:
            peer_hits += 1
    if peer_hits:
        if is_vehicle_path:
            score += min(80, peer_hits * 12)
            reasons.append(f"peer_token_hits={peer_hits}")
        else:
            score += min(10, peer_hits * 2)
            reasons.append(f"peer_token_hits_low_weight={peer_hits}")

    return score, reasons


def parse_models_arg(text: str) -> list[str]:
    return [x.strip().lower() for x in text.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank likely texture files for selected extensionless models.")
    parser.add_argument("--input-root", type=Path, default=Path("work/extracted/full_all"))
    parser.add_argument("--output-dir", type=Path, default=Path("notes/texture_candidates"))
    parser.add_argument("--models", default="mine,hopper", help="Comma-separated model-name substrings.")
    parser.add_argument("--max-candidates", type=int, default=25)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_queries = parse_models_arg(args.models)

    model_materials: dict[ModelKey, list[int]] = {}
    uses_by_material: dict[int, list[ModelKey]] = defaultdict(list)

    for archive_dir in sorted(input_root.iterdir()):
        if not archive_dir.is_dir():
            continue
        for child in archive_dir.iterdir():
            if not child.is_file() or child.suffix:
                continue
            parsed = parse_model_materials(child, input_root)
            if not parsed:
                continue
            key, mats = parsed
            model_materials[key] = mats
            for mat in mats:
                uses_by_material[mat].append(key)

    mare_records: list[MareRecord] = []
    for p in sorted(input_root.rglob("MARE_chunk/.*.dat")):
        mare_records.extend(parse_mare_records(p, input_root))

    mare_by_archive_and_material: dict[tuple[str, int], list[MareRecord]] = defaultdict(list)
    for rec in mare_records:
        mare_by_archive_and_material[(rec.archive, rec.material_hash)].append(rec)

    textures_by_archive: dict[str, list[str]] = defaultdict(list)
    for p in sorted(input_root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TEXTURE_EXTS:
            continue
        rel = rel_to(p, input_root)
        archive = p.parts[len(input_root.parts)]
        textures_by_archive[archive].append(rel)

    selected_models: list[ModelKey] = []
    for key in sorted(model_materials.keys(), key=lambda k: (k.archive, k.model_name, k.rel_path)):
        name_lc = key.model_name.lower()
        if model_queries and not any(q in name_lc for q in model_queries):
            continue
        selected_models.append(key)

    if not selected_models:
        raise SystemExit("No matching models found for --models query.")

    summary_rows: list[dict[str, object]] = []
    for key in selected_models:
        mats = model_materials[key]
        unique_mats = []
        for m in mats:
            if m not in unique_mats:
                unique_mats.append(m)

        peer_counter: Counter[str] = Counter()
        peer_tokens: set[str] = set()
        for mat in unique_mats:
            for peer in uses_by_material.get(mat, []):
                if peer == key:
                    continue
                peer_counter[peer.model_name] += 1
                peer_tokens.update(tokenize(peer.model_name))

        model_tokens = tokenize(key.model_name)
        archive_textures = textures_by_archive.get(key.archive, [])

        scored: list[tuple[int, str, list[str]]] = []
        for tex in archive_textures:
            score, reasons = score_texture(tex, model_tokens, peer_tokens)
            if score <= 0:
                continue
            scored.append((score, tex, reasons))
        scored.sort(key=lambda x: (-x[0], x[1]))

        mat_entries: list[dict[str, object]] = []
        for mat in unique_mats:
            recs = mare_by_archive_and_material.get((key.archive, mat), [])
            mat_entries.append(
                {
                    "material_hash": mat,
                    "material_hex": f"0x{mat:08X}",
                    "mare_records_in_archive": [
                        {
                            "record_index": r.record_index,
                            "link_a": r.link_a,
                            "link_a_hex": f"0x{r.link_a:08X}",
                            "link_b": r.link_b,
                            "link_b_hex": f"0x{r.link_b:08X}",
                        }
                        for r in recs
                    ],
                }
            )

        model_report = {
            "archive": key.archive,
            "model_name": key.model_name,
            "model_path": key.rel_path,
            "material_count": len(unique_mats),
            "materials": mat_entries,
            "peer_models": [{"name": n, "shared_material_count": c} for n, c in peer_counter.most_common(30)],
            "texture_inventory_count": len(archive_textures),
            "top_texture_candidates": [
                {"score": s, "path": p, "reasons": r} for s, p, r in scored[: args.max_candidates]
            ],
        }

        report_path = output_dir / f"{key.archive}__{key.model_name}_texture_candidates.json"
        report_path.write_text(json.dumps(model_report, indent=2), encoding="utf-8")

        for rank, (score, tex, reasons) in enumerate(scored[: args.max_candidates], start=1):
            summary_rows.append(
                {
                    "archive": key.archive,
                    "model_name": key.model_name,
                    "model_path": key.rel_path,
                    "rank": rank,
                    "score": score,
                    "texture_path": tex,
                    "reasons": "|".join(reasons),
                }
            )

    summary_csv = output_dir / "texture_candidate_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["archive", "model_name", "model_path", "rank", "score", "texture_path", "reasons"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "models_query": model_queries,
        "selected_models": len(selected_models),
        "summary_csv": str(summary_csv),
    }
    summary_json = output_dir / "texture_candidate_run_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
