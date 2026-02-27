#!/usr/bin/env python3
"""Export candidate OBJ meshes from Battlezone Gold Asura extensionless model files.

This exporter targets the repeated model container layout observed in extracted assets:
- Header: dynamic size based on part count
- Vertex records: fixed 40-byte stride
- Gap: 20 bytes
- Index buffer: trailing u16 triangle list

Because vertex semantics are still partially reverse engineered, the script emits one or more
position decode variants and scores each variant. This makes it practical to open candidates in
Blender and quickly identify the most usable geometry.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


PRIORITY_KEYWORDS = [
    "tank",
    "enemy",
    "drone",
    "cockpit",
    "weapon",
    "turret",
    "bomber",
    "hopper",
    "nemesis",
    "scout",
    "droid",
    "ufo",
    "artillery",
    "gun",
]

ENV_KEYWORDS = [
    "hangar",
    "bunker",
    "building",
    "pylon",
    "relay",
    "tower",
    "factory",
]


@dataclass
class ParsedModel:
    rel_path: str
    abs_path: Path
    file_size: int
    part_count: int
    vertex_count: int
    index_count: int
    triangle_count: int
    header_size: int
    index_offset: int
    descriptor_index_counts: list[int]
    vertices_u32: np.ndarray
    vertices_u16: np.ndarray
    indices_u16: np.ndarray


@dataclass
class ModeResult:
    mode: str
    score: float
    nondeg_ratio: float
    unique_ratio: float
    span_x: float
    span_y: float
    span_z: float
    edge_median: float
    edge_p95: float
    edge_p99: float
    vertices: np.ndarray


def parse_keywords(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.strip().lower() for x in text.split(",") if x.strip()]


def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def keyword_score(name_lc: str, keywords: list[str]) -> int:
    score = 0
    for kw in keywords:
        if kw in name_lc:
            score += 1
    return score


def discover_files(
    input_root: Path,
    include_environment: bool,
    keywords: list[str],
    max_files: int,
    discovery_mode: str,
) -> list[Path]:
    namespaces: list[Path] = []
    if discovery_mode == "headers":
        namespaces = [d for d in sorted(input_root.iterdir()) if d.is_dir()]
    else:
        for d in sorted(input_root.iterdir()):
            if not d.is_dir():
                continue
            if d.name in {"chars__actors_pc", "chars__playertanks_pc"}:
                namespaces.append(d)
            elif include_environment and d.name.startswith("envs__"):
                namespaces.append(d)

    candidates: list[tuple[int, int, str, Path]] = []
    all_keywords = list(keywords)
    if include_environment:
        all_keywords.extend(ENV_KEYWORDS)

    for ns in namespaces:
        for p in ns.iterdir():
            if not p.is_file() or p.suffix:
                continue
            name_lc = p.name.lower()
            if discovery_mode == "headers":
                kscore = 0
            else:
                kscore = keyword_score(name_lc, all_keywords)
                if keywords and kscore == 0:
                    continue
            size = p.stat().st_size
            rel = str(p.relative_to(input_root))
            candidates.append((kscore, size, rel, p))

    candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
    if max_files <= 0:
        return [x[3] for x in candidates]
    selected = [x[3] for x in candidates[:max_files]]
    return selected


def parse_model_file(path: Path, input_root: Path) -> tuple[ParsedModel | None, str | None]:
    blob = path.read_bytes()
    if len(blob) < 0x40:
        return None, "too_small"

    part_count = int.from_bytes(blob[0:4], "little", signed=False)
    vertex_count = int.from_bytes(blob[4:8], "little", signed=False)
    index_count = int.from_bytes(blob[8:12], "little", signed=False)
    tri_count = int.from_bytes(blob[12:16], "little", signed=False)

    if not (1 <= part_count <= 64):
        return None, "bad_part_count"
    if vertex_count <= 0 or index_count <= 0:
        return None, "bad_counts"
    if index_count % 3 != 0:
        return None, "non_triangle_index_count"
    if tri_count != index_count // 3:
        return None, "tri_count_mismatch"

    header_size = 0x10 + part_count * 0x18 + 0x08
    index_offset = len(blob) - (index_count * 2)
    expected_index_offset = header_size + vertex_count * 40 + 20

    if index_offset != expected_index_offset:
        return None, "layout_mismatch"
    if index_offset <= header_size:
        return None, "bad_offsets"

    descriptor_counts: list[int] = []
    for i in range(part_count):
        rec = 0x10 + i * 0x18
        cnt = int.from_bytes(blob[rec + 0x0C : rec + 0x10], "little", signed=False)
        descriptor_counts.append(cnt)
    if sum(descriptor_counts) != index_count:
        return None, "descriptor_sum_mismatch"

    vb = blob[header_size : header_size + vertex_count * 40]
    ib = blob[index_offset : index_offset + index_count * 2]

    vertices_u32 = np.frombuffer(vb, dtype="<u4").reshape(vertex_count, 10).copy()
    vertices_u16 = np.frombuffer(vb, dtype="<u2").reshape(vertex_count, 20).copy()
    indices_u16 = np.frombuffer(ib, dtype="<u2").copy()

    if int(indices_u16.max(initial=0)) >= vertex_count:
        return None, "index_out_of_range"

    model = ParsedModel(
        rel_path=str(path.relative_to(input_root)),
        abs_path=path,
        file_size=len(blob),
        part_count=part_count,
        vertex_count=vertex_count,
        index_count=index_count,
        triangle_count=tri_count,
        header_size=header_size,
        index_offset=index_offset,
        descriptor_index_counts=descriptor_counts,
        vertices_u32=vertices_u32,
        vertices_u16=vertices_u16,
        indices_u16=indices_u16,
    )
    return model, None


def decode_packed_257(model: ParsedModel) -> np.ndarray:
    # Best-performing candidate so far: dword[2], dword[5], dword[7] as 16.16 fixed.
    return model.vertices_u32[:, [2, 5, 7]].astype(np.int32).astype(np.float64) / 65536.0


def decode_packed_258(model: ParsedModel) -> np.ndarray:
    return model.vertices_u32[:, [2, 5, 8]].astype(np.int32).astype(np.float64) / 65536.0


def decode_packed_259(model: ParsedModel) -> np.ndarray:
    return model.vertices_u32[:, [2, 5, 9]].astype(np.int32).astype(np.float64) / 65536.0


def decode_s16_10_11_12(model: ParsedModel) -> np.ndarray:
    return model.vertices_u16[:, [10, 11, 12]].astype(np.int16).astype(np.float64)


DECODE_MODES: dict[str, Callable[[ParsedModel], np.ndarray]] = {
    "packed_257": decode_packed_257,
    "packed_258": decode_packed_258,
    "packed_259": decode_packed_259,
    "s16_10_11_12": decode_s16_10_11_12,
}


S16_MODE_RE = re.compile(r"^s16_(\d+)_(\d+)_(\d+)$")
AUTO_PART_S16_CANDIDATES = [
    (4, 10, 12),
    (5, 10, 12),
    (4, 10, 11),
    (4, 11, 12),
    (10, 11, 12),
    (10, 11, 15),
    (10, 11, 17),
    (10, 12, 19),
    (11, 12, 19),
    (4, 12, 16),
    (5, 12, 16),
    (4, 12, 14),
    (5, 12, 14),
]


def decode_auto_part_s16(model: ParsedModel) -> np.ndarray:
    tri_all = model.indices_u16.reshape(-1, 3)
    source = model.vertices_u16.astype(np.int16).astype(np.float64)
    out = np.zeros((model.vertex_count, 3), dtype=np.float64)
    assigned = np.zeros(model.vertex_count, dtype=bool)

    def part_combo_score(tri: np.ndarray, verts: np.ndarray) -> float:
        a = verts[tri[:, 0]]
        b = verts[tri[:, 1]]
        c = verts[tri[:, 2]]
        area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
        nondeg_ratio = float(np.mean(area2 > 1e-9))
        edges = np.concatenate(
            [
                np.linalg.norm(b - a, axis=1),
                np.linalg.norm(c - b, axis=1),
                np.linalg.norm(a - c, axis=1),
            ]
        )
        edge_median = float(np.median(edges))
        edge_p95 = float(np.percentile(edges, 95))
        edge_p99 = float(np.percentile(edges, 99))
        if edge_median > 1e-12:
            edge_ratio95 = edge_p95 / edge_median
            edge_ratio99 = edge_p99 / edge_median
        else:
            edge_ratio95 = 1e9
            edge_ratio99 = 1e9
        used_idx = np.unique(tri.reshape(-1))
        used_verts = verts[used_idx]
        rounded = np.round(used_verts, 5)
        unique_count = len({(float(v[0]), float(v[1]), float(v[2])) for v in rounded})
        unique_ratio = unique_count / max(1, len(used_idx))
        mins = used_verts.min(axis=0)
        maxs = used_verts.max(axis=0)
        span = maxs - mins
        max_span = float(np.max(span))
        min_span = float(np.min(span))
        span_balance = 0.0 if max_span <= 0 else min_span / max_span
        edge_score = (1.0 / (1.0 + edge_ratio95 / 10.0) + 1.0 / (1.0 + edge_ratio99 / 20.0)) * 0.5
        return (
            nondeg_ratio * 0.65
            + span_balance * 0.10
            + min(1.0, unique_ratio * 1.5) * 0.10
            + edge_score * 0.15
        )

    cursor = 0
    for count in model.descriptor_index_counts:
        tri_count = count // 3
        if tri_count <= 0:
            continue
        tri = tri_all[cursor : cursor + tri_count]
        cursor += tri_count
        used_idx = np.unique(tri.reshape(-1))
        if len(used_idx) == 0:
            continue
        best_score = -1.0
        best_combo = AUTO_PART_S16_CANDIDATES[0]
        for combo in AUTO_PART_S16_CANDIDATES:
            verts = source[:, list(combo)]
            score = part_combo_score(tri, verts)
            if score > best_score:
                best_score = score
                best_combo = combo
        part_verts = source[:, list(best_combo)]
        out[used_idx] = part_verts[used_idx]
        assigned[used_idx] = True

    if not np.all(assigned):
        fallback = source[:, [10, 11, 12]]
        out[~assigned] = fallback[~assigned]

    return out


def resolve_mode_decoder(mode_name: str) -> Callable[[ParsedModel], np.ndarray] | None:
    if mode_name == "auto_part_s16":
        return decode_auto_part_s16

    decoder = DECODE_MODES.get(mode_name)
    if decoder is not None:
        return decoder

    match = S16_MODE_RE.match(mode_name)
    if not match:
        return None

    lanes = tuple(int(match.group(i)) for i in (1, 2, 3))
    if any(lane < 0 or lane > 19 for lane in lanes):
        return None

    def decode_dynamic_s16(model: ParsedModel) -> np.ndarray:
        return model.vertices_u16[:, list(lanes)].astype(np.int16).astype(np.float64)

    return decode_dynamic_s16


def evaluate_mode(model: ParsedModel, mode_name: str, verts: np.ndarray) -> ModeResult:
    tri = model.indices_u16.reshape(-1, 3)
    a = verts[tri[:, 0]]
    b = verts[tri[:, 1]]
    c = verts[tri[:, 2]]
    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    nondeg_ratio = float(np.mean(area2 > 1e-9))

    edges = np.concatenate(
        [
            np.linalg.norm(b - a, axis=1),
            np.linalg.norm(c - b, axis=1),
            np.linalg.norm(a - c, axis=1),
        ]
    )
    edge_median = float(np.median(edges))
    edge_p95 = float(np.percentile(edges, 95))
    edge_p99 = float(np.percentile(edges, 99))
    if edge_median > 1e-12:
        edge_ratio95 = edge_p95 / edge_median
        edge_ratio99 = edge_p99 / edge_median
    else:
        edge_ratio95 = 1e9
        edge_ratio99 = 1e9

    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    span = maxs - mins
    max_span = float(np.max(span))
    min_span = float(np.min(span))
    span_balance = 0.0 if max_span <= 0 else min_span / max_span

    # Stable approximate uniqueness ratio (rounded to reduce float noise).
    rounded = np.round(verts, 5)
    unique_count = len({(float(v[0]), float(v[1]), float(v[2])) for v in rounded})
    unique_ratio = unique_count / max(1, model.vertex_count)

    edge_score = (1.0 / (1.0 + edge_ratio95 / 10.0) + 1.0 / (1.0 + edge_ratio99 / 20.0)) * 0.5
    score = (
        nondeg_ratio * 0.65
        + span_balance * 0.10
        + min(1.0, unique_ratio * 1.5) * 0.10
        + edge_score * 0.15
    )
    if mode_name == "packed_257":
        score += 0.005
    if mode_name == "auto_part_s16":
        score += 0.01

    return ModeResult(
        mode=mode_name,
        score=score,
        nondeg_ratio=nondeg_ratio,
        unique_ratio=unique_ratio,
        span_x=float(span[0]),
        span_y=float(span[1]),
        span_z=float(span[2]),
        edge_median=edge_median,
        edge_p95=edge_p95,
        edge_p99=edge_p99,
        vertices=verts,
    )


def transform_vertices(verts: np.ndarray, scale: float, center: bool) -> np.ndarray:
    out = verts.copy()
    if center:
        mins = out.min(axis=0)
        maxs = out.max(axis=0)
        out -= (mins + maxs) / 2.0
    if scale != 1.0:
        out *= scale
    return out


def write_obj(
    out_path: Path,
    model: ParsedModel,
    result: ModeResult,
    scale: float,
    center: bool,
    edge_prune_median_multiplier: float | None,
) -> None:
    verts = transform_vertices(result.vertices, scale=scale, center=center)
    tri = model.indices_u16.reshape(-1, 3)
    tri_keep = np.ones(len(tri), dtype=bool)

    if edge_prune_median_multiplier is not None and edge_prune_median_multiplier > 0:
        a = verts[tri[:, 0]]
        b = verts[tri[:, 1]]
        c = verts[tri[:, 2]]
        e0 = np.linalg.norm(b - a, axis=1)
        e1 = np.linalg.norm(c - b, axis=1)
        e2 = np.linalg.norm(a - c, axis=1)
        edge_all = np.concatenate([e0, e1, e2])
        med = float(np.median(edge_all))
        if med > 1e-12:
            edge_limit = med * edge_prune_median_multiplier
            tri_keep &= (e0 <= edge_limit) & (e1 <= edge_limit) & (e2 <= edge_limit)
        area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
        tri_keep &= area2 > 1e-12

    kept_tris = int(np.count_nonzero(tri_keep))
    kept_ratio = kept_tris / max(1, len(tri))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="\n", encoding="utf-8") as f:
        f.write(f"# source: {model.rel_path}\n")
        f.write(f"# mode: {result.mode}\n")
        f.write(f"# score: {result.score:.6f}\n")
        f.write(f"# nondeg_ratio: {result.nondeg_ratio:.6f}\n")
        f.write(f"# vertices: {model.vertex_count} indices: {model.index_count} triangles: {model.triangle_count}\n")
        f.write(f"# part_count: {model.part_count} descriptor_counts: {model.descriptor_index_counts}\n")
        f.write(f"# scale: {scale} center: {center}\n")
        f.write(f"# edge_median: {result.edge_median:.6f} edge_p95: {result.edge_p95:.6f} edge_p99: {result.edge_p99:.6f}\n")
        f.write(f"# triangle_prune_multiplier: {edge_prune_median_multiplier}\n")
        f.write(f"# triangles_kept: {kept_tris}/{len(tri)} ({kept_ratio:.6f})\n")

        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        cursor = 0
        for part_idx, count in enumerate(model.descriptor_index_counts):
            tri_count = count // 3
            if tri_count <= 0:
                continue
            f.write(f"g part_{part_idx:02d}\n")
            tri_start = cursor // 3
            for local_idx in range(tri_count):
                tri_idx = tri_start + local_idx
                if tri_idx >= len(tri_keep) or not tri_keep[tri_idx]:
                    cursor += 3
                    continue
                i0 = int(model.indices_u16[cursor]) + 1
                i1 = int(model.indices_u16[cursor + 1]) + 1
                i2 = int(model.indices_u16[cursor + 2]) + 1
                cursor += 3
                f.write(f"f {i0} {i1} {i2}\n")

        # Safety fallback if descriptor table doesn't fully consume all faces.
        if cursor < model.index_count:
            f.write("g part_remainder\n")
            while cursor + 2 < model.index_count:
                tri_idx = cursor // 3
                if tri_idx >= len(tri_keep) or not tri_keep[tri_idx]:
                    cursor += 3
                    continue
                i0 = int(model.indices_u16[cursor]) + 1
                i1 = int(model.indices_u16[cursor + 1]) + 1
                i2 = int(model.indices_u16[cursor + 2]) + 1
                cursor += 3
                f.write(f"f {i0} {i1} {i2}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Blender-loadable OBJ candidates from extracted model binaries")
    parser.add_argument("--input-root", type=Path, default=Path("work/extracted/full_all"))
    parser.add_argument("--output-dir", type=Path, default=Path("work/preview/model_obj_candidates"))
    parser.add_argument("--keywords", default=",".join(PRIORITY_KEYWORDS))
    parser.add_argument("--max-files", type=int, default=240)
    parser.add_argument("--discovery-mode", choices=["keywords", "headers"], default="keywords")
    parser.add_argument(
        "--modes",
        default=(
            "auto_part_s16,packed_257,packed_258,packed_259,"
            "s16_10_11_12,s16_4_10_12,s16_5_10_12,s16_4_10_11,s16_4_11_12,"
            "s16_10_11_15,s16_10_11_17,s16_10_12_19,s16_11_12_19,s16_5_12_16,s16_4_12_16"
        ),
    )
    parser.add_argument("--top-modes", type=int, default=2)
    parser.add_argument("--scale", type=float, default=0.001)
    parser.add_argument("--no-center", action="store_true")
    parser.add_argument("--edge-prune-median-multiplier", type=float, default=None)
    parser.add_argument("--include-environment", action="store_true")
    parser.add_argument("--report-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    keywords = parse_keywords(args.keywords)
    mode_names = [m.strip() for m in args.modes.split(",") if m.strip()]
    mode_names = [m for m in mode_names if resolve_mode_decoder(m) is not None]
    if not mode_names:
        raise SystemExit("No valid decode modes selected.")

    files = discover_files(
        input_root=input_root,
        include_environment=args.include_environment,
        keywords=keywords,
        max_files=args.max_files,
        discovery_mode=args.discovery_mode,
    )

    report_rows: list[dict[str, object]] = []
    success_files = 0
    failed_files = 0
    written_objs = 0

    for src in files:
        model, parse_error = parse_model_file(src, input_root)
        if model is None:
            failed_files += 1
            report_rows.append(
                {
                    "source": str(src.relative_to(input_root)),
                    "status": "parse_failed",
                    "reason": parse_error,
                }
            )
            continue

        mode_results: list[ModeResult] = []
        for mode_name in mode_names:
            decoder = resolve_mode_decoder(mode_name)
            if decoder is None:
                continue
            verts = decoder(model)
            if verts.shape != (model.vertex_count, 3):
                continue
            result = evaluate_mode(model, mode_name, verts)
            mode_results.append(result)

        if not mode_results:
            failed_files += 1
            report_rows.append(
                {
                    "source": model.rel_path,
                    "status": "decode_failed",
                    "reason": "no_mode_results",
                }
            )
            continue

        mode_results.sort(key=lambda r: r.score, reverse=True)
        best = mode_results[: max(1, args.top_modes)]

        safe_rel = sanitize_name(model.rel_path.replace("\\", "__").replace("/", "__"))
        for rank, result in enumerate(best, start=1):
            obj_name = f"{safe_rel}__{rank:02d}_{result.mode}.obj"
            obj_path = output_dir / obj_name
            write_obj(
                out_path=obj_path,
                model=model,
                result=result,
                scale=args.scale,
                center=not args.no_center,
                edge_prune_median_multiplier=args.edge_prune_median_multiplier,
            )
            written_objs += 1

        success_files += 1

        for result in mode_results:
            report_rows.append(
                {
                    "source": model.rel_path,
                    "status": "ok",
                    "file_size": model.file_size,
                    "part_count": model.part_count,
                    "vertex_count": model.vertex_count,
                    "index_count": model.index_count,
                    "triangle_count": model.triangle_count,
                    "mode": result.mode,
                    "score": f"{result.score:.6f}",
                    "nondeg_ratio": f"{result.nondeg_ratio:.6f}",
                    "unique_ratio": f"{result.unique_ratio:.6f}",
                    "span_x": f"{result.span_x:.6f}",
                    "span_y": f"{result.span_y:.6f}",
                    "span_z": f"{result.span_z:.6f}",
                    "edge_median": f"{result.edge_median:.6f}",
                    "edge_p95": f"{result.edge_p95:.6f}",
                    "edge_p99": f"{result.edge_p99:.6f}",
                    "exported": "yes" if result in best else "no",
                }
            )

    report_csv = args.report_csv.resolve() if args.report_csv else (output_dir / "obj-export-report.csv")
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        # Union all keys to keep report robust.
        keys: list[str] = []
        seen: set[str] = set()
        for row in report_rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(report_rows)

    summary = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "total_candidates": len(files),
        "success_files": success_files,
        "failed_files": failed_files,
        "written_objs": written_objs,
        "discovery_mode": args.discovery_mode,
        "modes": mode_names,
        "top_modes": args.top_modes,
        "scale": args.scale,
        "centered": not args.no_center,
        "edge_prune_median_multiplier": args.edge_prune_median_multiplier,
        "report_csv": str(report_csv),
    }

    summary_json = args.summary_json.resolve() if args.summary_json else (output_dir / "obj-export-summary.json")
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
