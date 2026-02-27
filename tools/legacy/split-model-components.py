#!/usr/bin/env python3
"""Split decoded model triangles into connected OBJ components.

Useful when a direct decode yields a large tangled mesh. This script prunes long
spike triangles, then exports connected triangle components as individual OBJs.
"""

from __future__ import annotations

import argparse
import csv
import re
import struct
from collections import deque
from pathlib import Path

import numpy as np


S16_MODE_RE = re.compile(r"^s16_(\d+)_(\d+)_(\d+)$")


def parse_model(path: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    blob = path.read_bytes()
    if len(blob) < 0x40:
        raise ValueError("file too small")
    part_count, vertex_count, index_count, tri_count = struct.unpack_from("<IIII", blob, 0)
    if index_count % 3 != 0 or tri_count != index_count // 3:
        raise ValueError("invalid index/triangle counts")
    header_size = 0x10 + part_count * 0x18 + 0x08
    index_offset = len(blob) - index_count * 2
    expected = header_size + vertex_count * 40 + 20
    if index_offset != expected:
        raise ValueError("layout mismatch")
    counts = [struct.unpack_from("<I", blob, 0x10 + i * 0x18 + 0x0C)[0] for i in range(part_count)]
    if sum(counts) != index_count:
        raise ValueError("descriptor count mismatch")
    vb = np.frombuffer(blob[header_size : header_size + vertex_count * 40], dtype="<u2").reshape(vertex_count, 20).copy()
    ib = np.frombuffer(blob[index_offset : index_offset + index_count * 2], dtype="<u2").copy()
    return vb, ib.reshape(-1, 3), counts


def decode_vertices(vb_u16: np.ndarray, mode: str) -> np.ndarray:
    if mode == "packed_257":
        vb_u32 = vb_u16.view("<u4").reshape(vb_u16.shape[0], 10)
        return vb_u32[:, [2, 5, 7]].astype(np.int32).astype(np.float64) / 65536.0
    match = S16_MODE_RE.match(mode)
    if not match:
        raise ValueError(f"unsupported mode: {mode}")
    lanes = [int(match.group(i)) for i in (1, 2, 3)]
    if any(l < 0 or l > 19 for l in lanes):
        raise ValueError("invalid s16 lanes")
    return vb_u16[:, lanes].astype(np.int16).astype(np.float64)


def prune_triangles(verts: np.ndarray, tri: np.ndarray, edge_mult: float) -> np.ndarray:
    a = verts[tri[:, 0]]
    b = verts[tri[:, 1]]
    c = verts[tri[:, 2]]
    e0 = np.linalg.norm(b - a, axis=1)
    e1 = np.linalg.norm(c - b, axis=1)
    e2 = np.linalg.norm(a - c, axis=1)
    med = float(np.median(np.concatenate([e0, e1, e2])))
    keep = np.ones(len(tri), dtype=bool)
    if med > 1e-12 and edge_mult > 0:
        lim = med * edge_mult
        keep &= (e0 <= lim) & (e1 <= lim) & (e2 <= lim)
    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    keep &= area2 > 1e-12
    return keep


def split_components(tri: np.ndarray, verts: np.ndarray, weld_epsilon: float) -> list[np.ndarray]:
    if len(tri) == 0:
        return []
    if weld_epsilon <= 0:
        weld_epsilon = 1e-6

    quant = np.round(verts / weld_epsilon).astype(np.int64)
    weld_key_to_id: dict[tuple[int, int, int], int] = {}
    vert_to_weld = np.zeros(len(verts), dtype=np.int64)
    next_id = 0
    for vi, q in enumerate(quant):
        key = (int(q[0]), int(q[1]), int(q[2]))
        wid = weld_key_to_id.get(key)
        if wid is None:
            wid = next_id
            weld_key_to_id[key] = wid
            next_id += 1
        vert_to_weld[vi] = wid

    tri_by_weld: dict[int, list[int]] = {}
    for ti, (i0, i1, i2) in enumerate(tri):
        tri_by_weld.setdefault(int(vert_to_weld[int(i0)]), []).append(ti)
        tri_by_weld.setdefault(int(vert_to_weld[int(i1)]), []).append(ti)
        tri_by_weld.setdefault(int(vert_to_weld[int(i2)]), []).append(ti)

    seen = np.zeros(len(tri), dtype=bool)
    components: list[np.ndarray] = []
    for seed in range(len(tri)):
        if seen[seed]:
            continue
        q: deque[int] = deque([seed])
        seen[seed] = True
        tris: list[int] = []
        while q:
            t = q.popleft()
            tris.append(t)
            i0, i1, i2 = tri[t]
            for vi in (int(i0), int(i1), int(i2)):
                wid = int(vert_to_weld[vi])
                for nb in tri_by_weld.get(wid, []):
                    if not seen[nb]:
                        seen[nb] = True
                        q.append(nb)
        components.append(np.array(tris, dtype=np.int32))
    return components


def write_component_obj(out_path: Path, verts: np.ndarray, tri: np.ndarray, comp_idx: int) -> tuple[int, int]:
    comp_tri = tri[comp_idx]
    used = np.unique(comp_tri.reshape(-1))
    remap = {int(old): i + 1 for i, old in enumerate(used)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for old in used:
            v = verts[int(old)]
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for i0, i1, i2 in comp_tri:
            f.write(f"f {remap[int(i0)]} {remap[int(i1)]} {remap[int(i2)]}\n")
    return len(used), len(comp_tri)


def main() -> int:
    parser = argparse.ArgumentParser(description="Split decoded model into connected OBJ components")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", default="s16_4_10_12")
    parser.add_argument("--edge-prune-multiplier", type=float, default=8.0)
    parser.add_argument("--min-triangles", type=int, default=80)
    parser.add_argument("--top-components", type=int, default=20)
    parser.add_argument("--scale", type=float, default=0.001)
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--weld-epsilon", type=float, default=0.01)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vb_u16, tri_all, _counts = parse_model(args.input_file)
    verts = decode_vertices(vb_u16, args.mode)
    keep = prune_triangles(verts, tri_all, args.edge_prune_multiplier)
    tri = tri_all[keep]
    if args.center:
        mins = verts.min(axis=0)
        maxs = verts.max(axis=0)
        verts = verts - (mins + maxs) / 2.0
    if args.scale != 1.0:
        verts = verts * args.scale

    comps = split_components(tri, verts, args.weld_epsilon)
    comps.sort(key=lambda c: len(c), reverse=True)
    comps = [c for c in comps if len(c) >= args.min_triangles][: args.top_components]

    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.input_file.stem)
    report_rows: list[dict[str, object]] = []
    for i, comp in enumerate(comps, start=1):
        out = args.output_dir / f"{stem}__{args.mode}__comp_{i:03d}.obj"
        vcount, tcount = write_component_obj(out, verts, tri, comp)
        report_rows.append(
            {
                "component": i,
                "obj": out.name,
                "vertices": vcount,
                "triangles": tcount,
            }
        )

    report = args.output_dir / f"{stem}__{args.mode}__components.csv"
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["component", "obj", "vertices", "triangles"])
        writer.writeheader()
        writer.writerows(report_rows)

    print(
        {
            "input_file": str(args.input_file),
            "mode": args.mode,
            "kept_triangles": int(len(tri)),
            "components_written": len(report_rows),
            "report": str(report),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
