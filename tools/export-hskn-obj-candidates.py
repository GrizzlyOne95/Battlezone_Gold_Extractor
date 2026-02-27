#!/usr/bin/env python3
"""Export OBJ candidates directly from HSKN_chunk mesh data."""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

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
    "base",
]


@dataclass
class ParsedHSKN:
    rel_path: str
    archive: str
    chunk_file: str
    model_name: str
    file_size: int
    vertex_count: int
    strip_index_hint: int
    index_count: int
    triangle_count: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    vertices: np.ndarray
    raw_indices: np.ndarray
    record_stride: int
    triangles: np.ndarray


def parse_keywords(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.strip().lower() for x in text.split(",") if x.strip()]


def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def read_cstring(blob: bytes, start: int, max_len: int = 96) -> str:
    if start >= len(blob):
        return ""
    end = min(len(blob), start + max_len)
    raw = blob[start:end].split(b"\x00", 1)[0]
    if not raw:
        return ""
    if any(b < 32 or b > 126 for b in raw):
        return ""
    return raw.decode("ascii", errors="ignore")


def keyword_score(name_lc: str, keywords: list[str]) -> int:
    score = 0
    for kw in keywords:
        if kw in name_lc:
            score += 1
    return score


def discover_hskn_files(
    input_root: Path,
    include_environment: bool,
    keywords: list[str],
    max_files: int,
    discovery_mode: str,
) -> list[Path]:
    candidates: list[tuple[int, int, str, Path]] = []
    all_keywords = list(keywords)
    if include_environment:
        all_keywords.extend(ENV_KEYWORDS)

    for archive_dir in sorted(input_root.iterdir()):
        if not archive_dir.is_dir():
            continue
        if discovery_mode == "keywords":
            is_chars = archive_dir.name.startswith("chars__")
            is_env = archive_dir.name.startswith("envs__")
            if not is_chars and not (include_environment and is_env):
                continue
        chunk_dir = archive_dir / "HSKN_chunk"
        if not chunk_dir.is_dir():
            continue
        for p in sorted(chunk_dir.glob(".*.dat")):
            size = p.stat().st_size
            blob = p.read_bytes()
            model_name = read_cstring(blob, 8) or p.name
            name_lc = model_name.lower()
            if discovery_mode == "headers":
                score = 0
            else:
                score = keyword_score(name_lc, all_keywords)
                if keywords and score == 0:
                    continue
            rel = str(p.relative_to(input_root))
            candidates.append((score, size, rel, p))

    candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
    if max_files <= 0:
        return [x[3] for x in candidates]
    return [x[3] for x in candidates[:max_files]]


def strip_to_triangles(indices: np.ndarray) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    strip: list[int] = []
    for v in indices.tolist():
        if v == 0xFFFF:
            strip.clear()
            continue
        strip.append(v)
        if len(strip) < 3:
            continue
        a, b, c = strip[-3], strip[-2], strip[-1]
        if a == b or b == c or a == c:
            continue
        # Triangle strip winding alternates with parity.
        if (len(strip) % 2) == 0:
            triangles.append((a, c, b))
        else:
            triangles.append((a, b, c))
    if not triangles:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(triangles, dtype=np.int32)


def strip_to_triangles_adaptive(
    indices: np.ndarray,
    verts: np.ndarray,
    edge_multiplier: float | None,
    local_multiplier: float | None,
    aspect_max: float | None,
) -> np.ndarray:
    # Fallback to plain strip when adaptive controls are disabled.
    if (edge_multiplier is None or edge_multiplier <= 0) and (local_multiplier is None or local_multiplier <= 0) and (
        aspect_max is None or aspect_max <= 0
    ):
        return strip_to_triangles(indices)

    base = strip_to_triangles(indices)
    if len(base) == 0:
        return base

    # Baseline scale estimates from plain strip decode.
    a0 = verts[base[:, 0]]
    b0 = verts[base[:, 1]]
    c0 = verts[base[:, 2]]
    be0 = np.linalg.norm(b0 - a0, axis=1)
    be1 = np.linalg.norm(c0 - b0, axis=1)
    be2 = np.linalg.norm(a0 - c0, axis=1)
    global_med = float(np.median(np.concatenate([be0, be1, be2])))
    if global_med <= 1e-12:
        return base

    # Per-vertex local edge medians from plain strip decode.
    vcount = len(verts)
    local_lists: list[list[float]] = [[] for _ in range(vcount)]
    for ti, (i0, i1, i2) in enumerate(base):
        e01 = float(be0[ti])
        e12 = float(be1[ti])
        e20 = float(be2[ti])
        i0i = int(i0)
        i1i = int(i1)
        i2i = int(i2)
        local_lists[i0i].append(e01)
        local_lists[i1i].append(e01)
        local_lists[i1i].append(e12)
        local_lists[i2i].append(e12)
        local_lists[i2i].append(e20)
        local_lists[i0i].append(e20)
    local_med = np.full(vcount, global_med, dtype=np.float64)
    for vi, arr in enumerate(local_lists):
        if arr:
            local_med[vi] = float(np.median(np.asarray(arr, dtype=np.float64)))

    triangles: list[tuple[int, int, int]] = []
    strip: list[int] = []
    for v in indices.tolist():
        if v == 0xFFFF:
            strip.clear()
            continue
        strip.append(v)
        if len(strip) < 3:
            continue
        i0 = int(strip[-3])
        i1 = int(strip[-2])
        i2 = int(strip[-1])
        if i0 == i1 or i1 == i2 or i0 == i2:
            continue

        p0 = verts[i0]
        p1 = verts[i1]
        p2 = verts[i2]
        e01 = float(np.linalg.norm(p1 - p0))
        e12 = float(np.linalg.norm(p2 - p1))
        e20 = float(np.linalg.norm(p0 - p2))
        edge_max = max(e01, e12, e20)
        edge_min = max(min(e01, e12, e20), 1e-12)
        aspect = edge_max / edge_min

        reject = False
        if edge_multiplier is not None and edge_multiplier > 0 and edge_max > global_med * edge_multiplier:
            reject = True
        if not reject and aspect_max is not None and aspect_max > 0 and aspect > aspect_max:
            reject = True
        if not reject and local_multiplier is not None and local_multiplier > 0:
            lim01 = max((local_med[i0] + local_med[i1]) * 0.5, 1e-12) * local_multiplier
            lim12 = max((local_med[i1] + local_med[i2]) * 0.5, 1e-12) * local_multiplier
            lim20 = max((local_med[i2] + local_med[i0]) * 0.5, 1e-12) * local_multiplier
            if e01 > lim01 or e12 > lim12 or e20 > lim20:
                reject = True

        if reject:
            # Virtual strip restart when triangle looks like a cross-mesh bridge.
            strip = [i2]
            continue

        if (len(strip) % 2) == 0:
            triangles.append((i0, i2, i1))
        else:
            triangles.append((i0, i1, i2))

    if not triangles:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(triangles, dtype=np.int32)


def list_to_triangles(indices: np.ndarray) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    acc: list[int] = []
    for v in indices.tolist():
        if v == 0xFFFF:
            acc.clear()
            continue
        acc.append(v)
        if len(acc) < 3:
            continue
        a, b, c = acc
        acc.clear()
        if a == b or b == c or a == c:
            continue
        triangles.append((a, b, c))
    if not triangles:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(triangles, dtype=np.int32)


def fan_to_triangles(indices: np.ndarray) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    seg: list[int] = []

    def flush_segment() -> None:
        if len(seg) < 3:
            return
        root = seg[0]
        for i in range(1, len(seg) - 1):
            a, b, c = root, seg[i], seg[i + 1]
            if a == b or b == c or a == c:
                continue
            triangles.append((a, b, c))

    for v in indices.tolist():
        if v == 0xFFFF:
            flush_segment()
            seg.clear()
            continue
        seg.append(v)
    flush_segment()

    if not triangles:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(triangles, dtype=np.int32)


def grouped_topology(indices: np.ndarray, group_size: int, mode: str) -> np.ndarray:
    if group_size <= 0 or len(indices) % group_size != 0:
        return np.zeros((0, 3), dtype=np.int32)
    tri: list[tuple[int, int, int]] = []
    groups = indices.reshape(-1, group_size)
    for g in groups:
        arr = g.astype(np.int32)
        if mode == "strip":
            tri.extend(strip_to_triangles(arr).tolist())
        elif mode == "fan":
            tri.extend(fan_to_triangles(arr).tolist())
        elif mode == "list":
            tri.extend(list_to_triangles(arr).tolist())
        else:
            raise ValueError(f"unsupported grouped mode: {mode}")
    if not tri:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(tri, dtype=np.int32)


def grouped_templates(indices: np.ndarray, group_size: int, templates: list[tuple[int, int, int]]) -> np.ndarray:
    if group_size <= 0 or len(indices) % group_size != 0:
        return np.zeros((0, 3), dtype=np.int32)
    tri: list[tuple[int, int, int]] = []
    groups = indices.reshape(-1, group_size)
    for g in groups:
        rec = g.astype(np.int32)
        for a, b, c in templates:
            if a >= group_size or b >= group_size or c >= group_size:
                continue
            i0 = int(rec[a])
            i1 = int(rec[b])
            i2 = int(rec[c])
            if i0 == 0xFFFF or i1 == 0xFFFF or i2 == 0xFFFF:
                continue
            if i0 == i1 or i1 == i2 or i0 == i2:
                continue
            tri.append((i0, i1, i2))
    if not tri:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(tri, dtype=np.int32)


def rec_column_topology(indices: np.ndarray, group_size: int, column: int, mode: str) -> np.ndarray:
    if group_size <= 0 or len(indices) % group_size != 0:
        return np.zeros((0, 3), dtype=np.int32)
    if column < 0 or column >= group_size:
        return np.zeros((0, 3), dtype=np.int32)
    groups = indices.reshape(-1, group_size)
    seq = groups[:, column].astype(np.int32)
    if mode == "strip":
        return strip_to_triangles(seq)
    if mode == "fan":
        return fan_to_triangles(seq)
    if mode == "list":
        return list_to_triangles(seq)
    raise ValueError(f"unsupported rec-column mode: {mode}")


def build_triangles_for_topology(
    parsed: ParsedHSKN,
    topology: str,
    strip_adaptive_edge_multiplier: float | None = None,
    strip_adaptive_local_multiplier: float | None = None,
    strip_adaptive_aspect_max: float | None = None,
) -> np.ndarray:
    if topology == "strip":
        return parsed.triangles
    if topology == "strip_adaptive":
        return strip_to_triangles_adaptive(
            parsed.raw_indices,
            parsed.vertices,
            edge_multiplier=strip_adaptive_edge_multiplier,
            local_multiplier=strip_adaptive_local_multiplier,
            aspect_max=strip_adaptive_aspect_max,
        )
    if topology == "list":
        return list_to_triangles(parsed.raw_indices)
    if topology == "fan":
        return fan_to_triangles(parsed.raw_indices)
    if topology == "rec5_strip":
        return grouped_topology(parsed.raw_indices, 5, "strip")
    if topology == "rec5_list":
        return grouped_topology(parsed.raw_indices, 5, "list")
    if topology == "rec5_fan":
        return grouped_topology(parsed.raw_indices, 5, "fan")
    if topology == "rec5_tpl_014":
        return grouped_templates(parsed.raw_indices, 5, [(0, 1, 4)])
    if topology == "rec5_tpl_024":
        return grouped_templates(parsed.raw_indices, 5, [(0, 2, 4)])
    if topology == "rec5_tpl_014_024":
        return grouped_templates(parsed.raw_indices, 5, [(0, 1, 4), (0, 2, 4)])
    m = re.match(r"^rec5col([0-4])_(strip|fan|list)$", topology)
    if m:
        col = int(m.group(1))
        mode = m.group(2)
        return rec_column_topology(parsed.raw_indices, 5, col, mode)
    raise ValueError(f"unsupported topology: {topology}")


def topology_quality_score(verts: np.ndarray, tri: np.ndarray) -> float:
    if len(tri) == 0:
        return -1e9
    a = verts[tri[:, 0]]
    b = verts[tri[:, 1]]
    c = verts[tri[:, 2]]
    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    nondeg = float(np.mean(area2 > 1e-12))
    e0 = np.linalg.norm(b - a, axis=1)
    e1 = np.linalg.norm(c - b, axis=1)
    e2 = np.linalg.norm(a - c, axis=1)
    edges = np.concatenate([e0, e1, e2])
    med = float(np.median(edges))
    if med <= 1e-12:
        return -1e9
    p95 = float(np.percentile(edges, 95))
    p99 = float(np.percentile(edges, 99))
    ratio95 = p95 / med
    ratio99 = p99 / med
    edge_score = (1.0 / (1.0 + ratio95 / 10.0) + 1.0 / (1.0 + ratio99 / 20.0)) * 0.5
    density = min(1.0, len(tri) / max(128.0, len(verts) * 1.2))
    return nondeg * 0.55 + edge_score * 0.35 + density * 0.10


def parse_hskn(path: Path, input_root: Path) -> tuple[ParsedHSKN | None, str | None]:
    blob = path.read_bytes()
    if len(blob) < 128:
        return None, "too_small"

    model_name = read_cstring(blob, 8) or path.stem
    sentinel = blob.find(b"\xff\xff\x00\x00\x00\x00", 0x40)
    if sentinel < 0:
        return None, "no_sentinel"
    if sentinel < 0x20:
        return None, "bad_sentinel_offset"

    vertex_count = struct.unpack_from("<H", blob, sentinel - 0x16)[0]
    strip_index_hint = struct.unpack_from("<H", blob, sentinel - 0x12)[0]
    if vertex_count <= 0 or vertex_count > 200_000:
        return None, "bad_vertex_count"

    bounds_start = sentinel + 0x06
    vertex_start = sentinel + 0x1E
    vertex_end = vertex_start + vertex_count * 12
    index_start = vertex_end + 4
    if index_start + 2 > len(blob):
        return None, "bad_offsets"
    if bounds_start + 24 > len(blob):
        return None, "bad_bounds"

    try:
        bounds = struct.unpack_from("<6f", blob, bounds_start)
    except struct.error:
        return None, "bounds_unpack_failed"
    bounds_min = (float(bounds[0]), float(bounds[2]), float(bounds[4]))
    bounds_max = (float(bounds[1]), float(bounds[3]), float(bounds[5]))

    try:
        verts = np.frombuffer(blob, dtype="<f4", count=vertex_count * 3, offset=vertex_start).reshape(vertex_count, 3).copy()
    except ValueError:
        return None, "vertex_unpack_failed"
    if not np.isfinite(verts).all():
        return None, "non_finite_vertices"

    # Indices are a strip stream with 0xFFFF restart; stream ends when values leave range.
    idx_vals: list[int] = []
    for off in range(index_start, len(blob) - 1, 2):
        v = struct.unpack_from("<H", blob, off)[0]
        if v == 0xFFFF or v < vertex_count:
            idx_vals.append(v)
        else:
            break
    if len(idx_vals) < 3:
        return None, "too_few_indices"

    indices = np.array(idx_vals, dtype=np.int32)
    triangles = strip_to_triangles(indices)
    if len(triangles) == 0:
        return None, "no_triangles"
    record_stride = 0
    if strip_index_hint > 0 and len(indices) % strip_index_hint == 0:
        record_stride = len(indices) // strip_index_hint

    rel_path = str(path.relative_to(input_root))
    archive = path.parent.parent.name
    parsed = ParsedHSKN(
        rel_path=rel_path,
        archive=archive,
        chunk_file=path.name,
        model_name=model_name,
        file_size=len(blob),
        vertex_count=vertex_count,
        strip_index_hint=int(strip_index_hint),
        index_count=len(indices),
        triangle_count=len(triangles),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        vertices=verts.astype(np.float64),
        raw_indices=indices,
        record_stride=record_stride,
        triangles=triangles,
    )
    return parsed, None


def transform_vertices(verts: np.ndarray, scale: float, center: bool) -> np.ndarray:
    out = verts.copy()
    if center:
        mins = out.min(axis=0)
        maxs = out.max(axis=0)
        out -= (mins + maxs) / 2.0
    if scale != 1.0:
        out *= scale
    return out


def triangle_keep_mask(
    verts: np.ndarray,
    tri: np.ndarray,
    edge_prune_median_multiplier: float | None,
    local_edge_prune_multiplier: float | None,
    aspect_ratio_max: float | None,
) -> np.ndarray:
    keep = np.ones(len(tri), dtype=bool)
    if len(tri) == 0:
        return keep

    a = verts[tri[:, 0]]
    b = verts[tri[:, 1]]
    c = verts[tri[:, 2]]
    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    keep &= area2 > 1e-12

    e0 = np.linalg.norm(b - a, axis=1)
    e1 = np.linalg.norm(c - b, axis=1)
    e2 = np.linalg.norm(a - c, axis=1)

    med = float(np.median(np.concatenate([e0, e1, e2])))
    if edge_prune_median_multiplier is not None and edge_prune_median_multiplier > 0 and med > 1e-12:
        lim = med * edge_prune_median_multiplier
        keep &= (e0 <= lim) & (e1 <= lim) & (e2 <= lim)

    if aspect_ratio_max is not None and aspect_ratio_max > 0:
        edge_max = np.maximum(np.maximum(e0, e1), e2)
        edge_min = np.minimum(np.minimum(e0, e1), e2)
        aspect = edge_max / np.maximum(edge_min, 1e-12)
        keep &= aspect <= aspect_ratio_max

    if local_edge_prune_multiplier is not None and local_edge_prune_multiplier > 0:
        # Local per-vertex median edge length catches bridge triangles that global median misses.
        vcount = len(verts)
        local_lists: list[list[float]] = [[] for _ in range(vcount)]
        for ti, (i0, i1, i2) in enumerate(tri):
            e01 = float(e0[ti])
            e12 = float(e1[ti])
            e20 = float(e2[ti])
            i0i = int(i0)
            i1i = int(i1)
            i2i = int(i2)
            local_lists[i0i].append(e01)
            local_lists[i1i].append(e01)
            local_lists[i1i].append(e12)
            local_lists[i2i].append(e12)
            local_lists[i2i].append(e20)
            local_lists[i0i].append(e20)
        local_med = np.full(vcount, med if med > 1e-12 else 1.0, dtype=np.float64)
        for vi, arr in enumerate(local_lists):
            if arr:
                local_med[vi] = float(np.median(np.asarray(arr, dtype=np.float64)))
        e01_local = np.maximum((local_med[tri[:, 0]] + local_med[tri[:, 1]]) * 0.5, 1e-12)
        e12_local = np.maximum((local_med[tri[:, 1]] + local_med[tri[:, 2]]) * 0.5, 1e-12)
        e20_local = np.maximum((local_med[tri[:, 2]] + local_med[tri[:, 0]]) * 0.5, 1e-12)
        keep &= (e0 <= e01_local * local_edge_prune_multiplier)
        keep &= (e1 <= e12_local * local_edge_prune_multiplier)
        keep &= (e2 <= e20_local * local_edge_prune_multiplier)

    return keep


def write_obj(
    out_path: Path,
    parsed: ParsedHSKN,
    topology: str,
    triangles: np.ndarray,
    scale: float,
    center: bool,
    edge_prune_median_multiplier: float | None,
    local_edge_prune_multiplier: float | None,
    aspect_ratio_max: float | None,
) -> tuple[int, int]:
    verts = transform_vertices(parsed.vertices, scale=scale, center=center)
    tri = triangles
    keep = triangle_keep_mask(
        verts,
        tri,
        edge_prune_median_multiplier=edge_prune_median_multiplier,
        local_edge_prune_multiplier=local_edge_prune_multiplier,
        aspect_ratio_max=aspect_ratio_max,
    )
    tri_kept = tri[keep]

    used = np.unique(tri_kept.reshape(-1)) if len(tri_kept) else np.zeros((0,), dtype=np.int32)
    remap = {int(old): i + 1 for i, old in enumerate(used)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# source: {parsed.rel_path}\n")
        f.write(f"# model_name: {parsed.model_name}\n")
        f.write(f"# vertices: {parsed.vertex_count} strip_indices: {parsed.index_count} triangles: {parsed.triangle_count}\n")
        f.write(f"# topology: {topology}\n")
        f.write(f"# bounds_min: {parsed.bounds_min}\n")
        f.write(f"# bounds_max: {parsed.bounds_max}\n")
        f.write(f"# strip_index_hint: {parsed.strip_index_hint}\n")
        f.write(f"# scale: {scale} center: {center}\n")
        f.write(f"# triangle_prune_multiplier: {edge_prune_median_multiplier}\n")
        f.write(f"# local_edge_prune_multiplier: {local_edge_prune_multiplier}\n")
        f.write(f"# aspect_ratio_max: {aspect_ratio_max}\n")
        f.write(f"# triangles_kept: {len(tri_kept)}/{len(tri)}\n")
        for old in used:
            v = verts[int(old)]
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("g mesh\n")
        for i0, i1, i2 in tri_kept:
            f.write(f"f {remap[int(i0)]} {remap[int(i1)]} {remap[int(i2)]}\n")
    return len(used), len(tri_kept)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Blender-loadable OBJ candidates from HSKN chunks")
    parser.add_argument("--input-root", type=Path, default=Path("work/extracted/full_all"))
    parser.add_argument("--output-dir", type=Path, default=Path("work/preview/model_obj_candidates_hskn"))
    parser.add_argument("--keywords", default=",".join(PRIORITY_KEYWORDS))
    parser.add_argument("--max-files", type=int, default=320)
    parser.add_argument("--discovery-mode", choices=["keywords", "headers"], default="keywords")
    parser.add_argument("--min-triangles", type=int, default=24)
    parser.add_argument(
        "--topology-mode",
        choices=[
            "auto",
            "strip",
            "strip_adaptive",
            "list",
            "fan",
            "rec5_strip",
            "rec5_list",
            "rec5_fan",
            "rec5_tpl_014",
            "rec5_tpl_024",
            "rec5_tpl_014_024",
            "rec5col0_strip",
            "rec5col1_strip",
            "rec5col2_strip",
            "rec5col3_strip",
            "rec5col4_strip",
            "rec5col0_fan",
            "rec5col1_fan",
            "rec5col2_fan",
            "rec5col3_fan",
            "rec5col4_fan",
            "all",
        ],
        default="auto",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--no-center", action="store_true")
    parser.add_argument("--edge-prune-median-multiplier", type=float, default=None)
    parser.add_argument("--local-edge-prune-multiplier", type=float, default=None)
    parser.add_argument("--aspect-ratio-max", type=float, default=None)
    parser.add_argument("--strip-adaptive-edge-multiplier", type=float, default=6.0)
    parser.add_argument("--strip-adaptive-local-multiplier", type=float, default=2.5)
    parser.add_argument("--strip-adaptive-aspect-max", type=float, default=10.0)
    parser.add_argument("--include-environment", action="store_true")
    parser.add_argument("--report-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    keywords = parse_keywords(args.keywords)
    files = discover_hskn_files(
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
        parsed, err = parse_hskn(src, input_root)
        if parsed is None:
            failed_files += 1
            report_rows.append(
                {
                    "source": str(src.relative_to(input_root)),
                    "status": "parse_failed",
                    "reason": err,
                }
            )
            continue

        topo_candidates: list[tuple[str, np.ndarray]] = []
        base_tops = ["strip", "strip_adaptive", "list", "fan"]
        rec5_tops = [
            "rec5_strip",
            "rec5_list",
            "rec5_fan",
            "rec5_tpl_014",
            "rec5_tpl_024",
            "rec5_tpl_014_024",
            "rec5col0_strip",
            "rec5col1_strip",
            "rec5col2_strip",
            "rec5col3_strip",
            "rec5col4_strip",
            "rec5col0_fan",
            "rec5col1_fan",
            "rec5col2_fan",
            "rec5col3_fan",
            "rec5col4_fan",
        ]
        supported_rec5 = parsed.record_stride == 5
        if args.topology_mode == "all":
            for top in base_tops:
                topo_candidates.append(
                    (
                        top,
                        build_triangles_for_topology(
                            parsed,
                            top,
                            strip_adaptive_edge_multiplier=args.strip_adaptive_edge_multiplier,
                            strip_adaptive_local_multiplier=args.strip_adaptive_local_multiplier,
                            strip_adaptive_aspect_max=args.strip_adaptive_aspect_max,
                        ),
                    )
                )
            if supported_rec5:
                for top in rec5_tops:
                    topo_candidates.append(
                        (
                            top,
                            build_triangles_for_topology(
                                parsed,
                                top,
                                strip_adaptive_edge_multiplier=args.strip_adaptive_edge_multiplier,
                                strip_adaptive_local_multiplier=args.strip_adaptive_local_multiplier,
                                strip_adaptive_aspect_max=args.strip_adaptive_aspect_max,
                            ),
                        )
                    )
        elif args.topology_mode in set(base_tops + rec5_tops):
            if args.topology_mode.startswith("rec5_") and not supported_rec5:
                topo_candidates = []
            else:
                topo_candidates.append(
                    (
                        args.topology_mode,
                        build_triangles_for_topology(
                            parsed,
                            args.topology_mode,
                            strip_adaptive_edge_multiplier=args.strip_adaptive_edge_multiplier,
                            strip_adaptive_local_multiplier=args.strip_adaptive_local_multiplier,
                            strip_adaptive_aspect_max=args.strip_adaptive_aspect_max,
                        ),
                    )
                )
        else:
            scored: list[tuple[float, str, np.ndarray]] = []
            for top in base_tops:
                tri_top = build_triangles_for_topology(
                    parsed,
                    top,
                    strip_adaptive_edge_multiplier=args.strip_adaptive_edge_multiplier,
                    strip_adaptive_local_multiplier=args.strip_adaptive_local_multiplier,
                    strip_adaptive_aspect_max=args.strip_adaptive_aspect_max,
                )
                scored.append((topology_quality_score(parsed.vertices, tri_top), top, tri_top))
            if supported_rec5:
                for top in rec5_tops:
                    tri_top = build_triangles_for_topology(
                        parsed,
                        top,
                        strip_adaptive_edge_multiplier=args.strip_adaptive_edge_multiplier,
                        strip_adaptive_local_multiplier=args.strip_adaptive_local_multiplier,
                        strip_adaptive_aspect_max=args.strip_adaptive_aspect_max,
                    )
                    scored.append((topology_quality_score(parsed.vertices, tri_top), top, tri_top))
            if not scored:
                scored.append((-1e9, "strip", np.zeros((0, 3), dtype=np.int32)))
            scored.sort(key=lambda x: x[0], reverse=True)
            topo_candidates.append((scored[0][1], scored[0][2]))

        exported_for_file = 0
        for topology, tri in topo_candidates:
            if len(tri) < args.min_triangles:
                continue

            safe_rel = sanitize_name(parsed.rel_path.replace("\\", "__").replace("/", "__"))
            safe_name = sanitize_name(parsed.model_name)
            obj_name = f"{safe_rel}__{safe_name}__{topology}.obj"
            obj_path = output_dir / obj_name
            v_kept, t_kept = write_obj(
                out_path=obj_path,
                parsed=parsed,
                topology=topology,
                triangles=tri,
                scale=args.scale,
                center=not args.no_center,
                edge_prune_median_multiplier=args.edge_prune_median_multiplier,
                local_edge_prune_multiplier=args.local_edge_prune_multiplier,
                aspect_ratio_max=args.aspect_ratio_max,
            )
            written_objs += 1
            exported_for_file += 1
            report_rows.append(
                {
                    "source": parsed.rel_path,
                    "status": "ok",
                    "model_name": parsed.model_name,
                    "file_size": parsed.file_size,
                    "vertex_count": parsed.vertex_count,
                    "strip_index_hint": parsed.strip_index_hint,
                    "index_count": parsed.index_count,
                    "record_stride": parsed.record_stride,
                    "triangle_count": len(tri),
                    "topology": topology,
                    "obj": obj_name,
                    "obj_vertices": v_kept,
                    "obj_triangles": t_kept,
                }
            )

        if exported_for_file == 0:
            report_rows.append(
                {
                    "source": parsed.rel_path,
                    "status": "skipped",
                    "reason": "below_min_triangles",
                    "triangle_count": parsed.triangle_count,
                    "model_name": parsed.model_name,
                }
            )
            continue

        success_files += 1

    report_csv = args.report_csv.resolve() if args.report_csv else (output_dir / "hskn-obj-export-report.csv")
    with report_csv.open("w", newline="", encoding="utf-8") as f:
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
        "keywords": keywords,
        "discovery_mode": args.discovery_mode,
        "min_triangles": args.min_triangles,
        "topology_mode": args.topology_mode,
        "scale": args.scale,
        "centered": not args.no_center,
        "edge_prune_median_multiplier": args.edge_prune_median_multiplier,
        "local_edge_prune_multiplier": args.local_edge_prune_multiplier,
        "aspect_ratio_max": args.aspect_ratio_max,
        "strip_adaptive_edge_multiplier": args.strip_adaptive_edge_multiplier,
        "strip_adaptive_local_multiplier": args.strip_adaptive_local_multiplier,
        "strip_adaptive_aspect_max": args.strip_adaptive_aspect_max,
        "report_csv": str(report_csv),
    }
    summary_json = args.summary_json.resolve() if args.summary_json else (output_dir / "hskn-obj-export-summary.json")
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
