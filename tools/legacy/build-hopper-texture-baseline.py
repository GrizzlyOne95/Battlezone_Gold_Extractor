#!/usr/bin/env python3
"""Build a UV/texture baseline for Battlezone Gold hopper-style static models.

This script:
1) parses extensionless static model binaries (40-byte vertex records),
2) evaluates a small UV decode sweep against candidate texture PNGs,
3) ranks texture fits per material hash,
4) exports preview OBJ/MTL files for each UV mode, and
5) writes a baseline JSON with the best-ranked mapping table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class UvMode:
    name: str
    value_type: str  # "f16" | "u16norm"
    lane_u: int
    lane_v: int


@dataclass
class ModelData:
    name: str
    path: Path
    num_submeshes: int
    vertex_count: int
    index_count: int
    triangle_count: int
    material_hashes: list[int]
    submesh_index_counts: list[int]
    vertices: np.ndarray  # float32 [N, 3]
    indices: np.ndarray  # uint16 [M]
    u16_lanes: np.ndarray  # uint16 [N, 20]


DEFAULT_UV_MODES = [
    UvMode("f16_15_7", "f16", 15, 7),
    UvMode("f16_7_15", "f16", 7, 15),
    UvMode("f16_12_13", "f16", 12, 13),
    UvMode("u16norm_0_6", "u16norm", 0, 6),
]

DEFAULT_TEXTURES = [
    "chars__actors_pc/graphics/objects/vehicles/enemies.png",
    "chars__actors_pc/graphics/objects/vehicles/enemies_desert.png",
    "chars__actors_pc/graphics/objects/vehicles/enemies_icefields.png",
    "chars__actors_pc/graphics/objects/vehicles/enemies_wastelands.png",
    "chars__actors_pc/graphics/objects/vehicles/vehicles_metal_dif.png",
    "chars__actors_pc/graphics/objects/vehicles/heavy_tank/heavy_tank_dif.png",
]


def parse_uv_modes(text: str | None) -> list[UvMode]:
    if not text:
        return list(DEFAULT_UV_MODES)
    out: list[UvMode] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split("_")
        if len(parts) != 3:
            raise ValueError(f"Bad UV mode '{item}'. Expected form: f16_15_7 or u16norm_0_6")
        value_type, lane_u_s, lane_v_s = parts
        if value_type not in {"f16", "u16norm"}:
            raise ValueError(f"Unsupported UV mode type '{value_type}' in '{item}'")
        lane_u = int(lane_u_s)
        lane_v = int(lane_v_s)
        if lane_u < 0 or lane_u > 19 or lane_v < 0 or lane_v > 19:
            raise ValueError(f"UV lane out of range in '{item}'")
        out.append(UvMode(item, value_type, lane_u, lane_v))
    if not out:
        raise ValueError("No UV modes parsed from --uv-modes")
    return out


def parse_texture_list(text: str | None) -> list[str]:
    if not text:
        return list(DEFAULT_TEXTURES)
    out = [x.strip().replace("\\", "/") for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("No textures parsed from --textures")
    return out


def parse_hash_list(text: str | None) -> set[int]:
    if not text:
        return set()
    out: set[int] = set()
    for token in text.split(","):
        raw = token.strip()
        if not raw:
            continue
        out.add(int(raw, 0))
    return out


def parse_override_material_map(text: str | None) -> dict[int, str]:
    if not text:
        return {}
    out: dict[int, str] = {}
    for token in text.split(","):
        item = token.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad override '{item}'. Expected form: 0xAABBCCDD=path/to/texture.png")
        lhs, rhs = item.split("=", 1)
        mat_hash = int(lhs.strip(), 0)
        tex_rel = rhs.strip().replace("\\", "/")
        if not tex_rel:
            raise ValueError(f"Bad override '{item}': missing texture path")
        out[mat_hash] = tex_rel
    return out


def parse_static_model(path: Path) -> ModelData:
    blob = path.read_bytes()
    if len(blob) < 64:
        raise ValueError(f"{path}: too small to parse static model")

    num_submeshes, vertex_count, index_count, triangle_count = struct.unpack_from("<IIII", blob, 0)
    if num_submeshes <= 0 or num_submeshes > 2048:
        raise ValueError(f"{path}: invalid submesh count {num_submeshes}")
    if vertex_count <= 0 or index_count <= 0:
        raise ValueError(f"{path}: invalid vertex/index counts")

    desc_off = 20
    material_hashes: list[int] = []
    submesh_index_counts: list[int] = []
    for _ in range(num_submeshes):
        mat_hash, _idx_start, idx_count, _v_start, _v_count, _unk = struct.unpack_from("<IIIIII", blob, desc_off)
        material_hashes.append(mat_hash)
        submesh_index_counts.append(idx_count)
        desc_off += 24

    # bb max/min
    bb_max = struct.unpack_from("<fff", blob, desc_off)
    desc_off += 12
    bb_min = struct.unpack_from("<fff", blob, desc_off)
    desc_off += 12

    vb_size = vertex_count * 40
    ib_size = index_count * 2
    if desc_off + vb_size + ib_size > len(blob):
        raise ValueError(f"{path}: layout exceeds file size")

    vb = blob[desc_off : desc_off + vb_size]
    ib = blob[desc_off + vb_size : desc_off + vb_size + ib_size]
    u16_lanes = np.frombuffer(vb, dtype="<u2").reshape(vertex_count, 20).copy()
    indices = np.frombuffer(ib, dtype="<u2").copy()

    # Decode packed positions (same heuristic used in existing hopper tooling).
    rng = [bb_max[i] - bb_min[i] for i in range(3)]
    rng = [r if r != 0 else 1.0 for r in rng]
    verts = np.empty((vertex_count, 3), dtype=np.float64)
    verts[:, 0] = bb_min[0] + (u16_lanes[:, 0].astype(np.float64) / 65535.0) * rng[0]
    verts[:, 1] = bb_min[1] + (u16_lanes[:, 1].astype(np.float64) / 65535.0) * rng[1]
    verts[:, 2] = bb_min[2] + (u16_lanes[:, 2].astype(np.float64) / 65535.0) * rng[2]

    return ModelData(
        name=path.name,
        path=path,
        num_submeshes=num_submeshes,
        vertex_count=vertex_count,
        index_count=index_count,
        triangle_count=triangle_count,
        material_hashes=material_hashes,
        submesh_index_counts=submesh_index_counts,
        vertices=verts,
        indices=indices,
        u16_lanes=u16_lanes,
    )


def decode_uvs(model: ModelData, mode: UvMode) -> np.ndarray:
    if mode.value_type == "f16":
        f16 = model.u16_lanes.view("<f2").astype(np.float64)
        uv = np.stack([f16[:, mode.lane_u], f16[:, mode.lane_v]], axis=1)
    else:
        uv = np.stack(
            [
                model.u16_lanes[:, mode.lane_u].astype(np.float64) / 65535.0,
                model.u16_lanes[:, mode.lane_v].astype(np.float64) / 65535.0,
            ],
            axis=1,
        )
    return uv


def submesh_vertex_sets(model: ModelData) -> list[set[int]]:
    out: list[set[int]] = []
    cursor = 0
    for idx_count in model.submesh_index_counts:
        verts: set[int] = set()
        for i in range(0, idx_count, 3):
            if cursor + i + 2 >= len(model.indices):
                break
            i0 = int(model.indices[cursor + i + 0])
            i1 = int(model.indices[cursor + i + 1])
            i2 = int(model.indices[cursor + i + 2])
            if i0 < model.vertex_count and i1 < model.vertex_count and i2 < model.vertex_count:
                verts.add(i0)
                verts.add(i1)
                verts.add(i2)
        out.append(verts)
        cursor += idx_count
    return out


def sample_texture_metrics(image_rgb: np.ndarray, uv: np.ndarray) -> dict[str, float]:
    h, w, _ = image_rgb.shape
    u = np.mod(uv[:, 0], 1.0)
    v = np.mod(uv[:, 1], 1.0)
    x = np.clip((u * (w - 1)).astype(np.int32), 0, w - 1)
    y = np.clip(((1.0 - v) * (h - 1)).astype(np.int32), 0, h - 1)

    px = image_rgb[y, x]
    lum = 0.2126 * px[:, 0] + 0.7152 * px[:, 1] + 0.0722 * px[:, 2]

    mean = float(lum.mean())
    std = float(lum.std())
    dark = float((lum < 20.0).mean())
    bright = float((lum > 235.0).mean())

    # Heuristic score tuned to avoid near-solid black/white assignments.
    score = std - abs(mean - 110.0) * 0.25 - dark * 40.0 - max(0.0, bright - 0.60) * 30.0
    return {
        "mean_luma": mean,
        "std_luma": std,
        "dark_ratio_lt20": dark,
        "bright_ratio_gt235": bright,
        "score": float(score),
    }


def export_preview_obj_mtl(
    out_dir: Path,
    model: ModelData,
    mode: UvMode,
    uv: np.ndarray,
    material_to_texture: dict[int, str],
    texture_paths: dict[str, Path],
) -> tuple[Path, Path]:
    stem = f"{model.name}__{mode.name}"
    obj_path = out_dir / f"{stem}.obj"
    mtl_path = out_dir / f"{stem}.mtl"

    with mtl_path.open("w", encoding="utf-8", newline="\n") as mtl:
        mtl.write(f"# Auto baseline material map for {model.name} / {mode.name}\n")
        for mat_hash in sorted(set(model.material_hashes)):
            mat_name = f"mat_{mat_hash:08X}"
            mtl.write(f"\nnewmtl {mat_name}\n")
            mtl.write("Kd 0.800000 0.800000 0.800000\n")
            mtl.write("Ka 0.000000 0.000000 0.000000\n")
            mtl.write("Ks 0.000000 0.000000 0.000000\n")
            tex_rel = material_to_texture.get(mat_hash)
            if tex_rel:
                tex_abs = texture_paths.get(tex_rel)
                if tex_abs is None:
                    mtl.write(f"map_Kd {tex_rel}\n")
                else:
                    mtl_rel = Path(os.path.relpath(tex_abs, out_dir)).as_posix()
                    mtl.write(f"map_Kd {mtl_rel}\n")

    with obj_path.open("w", encoding="utf-8", newline="\n") as obj:
        obj.write(f"# source: {model.path}\n")
        obj.write(f"# uv_mode: {mode.name}\n")
        obj.write(f"mtllib {mtl_path.name}\n")

        for v in model.vertices:
            obj.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for t in uv:
            u = 0.0 if not math.isfinite(float(t[0])) else float(t[0])
            v = 0.0 if not math.isfinite(float(t[1])) else float(t[1])
            obj.write(f"vt {u:.6f} {1.0 - v:.6f}\n")

        cursor = 0
        for sub_idx, idx_count in enumerate(model.submesh_index_counts):
            mat_hash = model.material_hashes[sub_idx]
            obj.write(f"g submesh_{sub_idx:02d}\n")
            obj.write(f"usemtl mat_{mat_hash:08X}\n")
            for i in range(0, idx_count, 3):
                if cursor + i + 2 >= len(model.indices):
                    break
                i0 = int(model.indices[cursor + i + 0])
                i1 = int(model.indices[cursor + i + 1])
                i2 = int(model.indices[cursor + i + 2])
                if i0 >= model.vertex_count or i1 >= model.vertex_count or i2 >= model.vertex_count:
                    continue
                a = i0 + 1
                b = i1 + 1
                c = i2 + 1
                obj.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
            cursor += idx_count

    return obj_path, mtl_path


def resolve_texture_pngs(texture_root: Path, texture_rel_paths: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for rel in texture_rel_paths:
        p = texture_root / rel
        if p.exists():
            out[rel] = p
    return out


def compare_lod_materials(input_root: Path, archive: str, names: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name in names:
        path = input_root / archive / name
        if not path.exists():
            continue
        model = parse_static_model(path)
        mats = [f"0x{x:08X}" for x in sorted(set(model.material_hashes))]
        out[name] = mats
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build hopper UV/texture baseline from extracted assets.")
    parser.add_argument("--input-root", type=Path, default=Path("work/extracted/full_all"))
    parser.add_argument("--archive", default="chars__actors_pc")
    parser.add_argument("--model", default="hopper")
    parser.add_argument("--texture-root", type=Path, default=Path("work/preview/png_all_texconv"))
    parser.add_argument("--textures", default=",".join(DEFAULT_TEXTURES))
    parser.add_argument("--uv-modes", default=",".join(m.name for m in DEFAULT_UV_MODES))
    parser.add_argument("--selected-uv-mode", default=None, help="Force selected UV mode in baseline JSON (for example: f16_12_13).")
    parser.add_argument(
        "--untextured-material-hashes",
        default="",
        help="Comma-separated material hashes to keep untextured in MTL output (for example: 0x00000000).",
    )
    parser.add_argument(
        "--override-material-map",
        default="",
        help=(
            "Comma-separated material overrides as material_hash=texture_rel_path; "
            "for example: 0x6E84DBDC=chars__actors_pc/graphics/objects/vehicles/vehicles_metal_dif.png"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("notes/hopper_texture_baseline"))
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    texture_root = args.texture_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = input_root / args.archive / args.model
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    uv_modes = parse_uv_modes(args.uv_modes)
    untextured_material_hashes = parse_hash_list(args.untextured_material_hashes)
    override_material_map = parse_override_material_map(args.override_material_map)
    texture_rel_paths = parse_texture_list(args.textures)
    texture_paths = resolve_texture_pngs(texture_root, texture_rel_paths)
    for _mat_hash, tex_rel in override_material_map.items():
        if tex_rel not in texture_paths:
            p = texture_root / tex_rel
            if p.exists():
                texture_paths[tex_rel] = p
    if not texture_paths:
        raise SystemExit("No candidate textures found under --texture-root")

    model = parse_static_model(model_path)
    sm_vert_sets = submesh_vertex_sets(model)

    tex_rgb_cache: dict[str, np.ndarray] = {}
    for rel, p in texture_paths.items():
        tex_rgb_cache[rel] = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)

    mode_reports: list[dict[str, object]] = []

    for mode in uv_modes:
        uv = decode_uvs(model, mode)
        finite_mask = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
        finite_ratio = float(finite_mask.mean())
        u_min = float(np.nanmin(uv[:, 0]))
        u_max = float(np.nanmax(uv[:, 0]))
        v_min = float(np.nanmin(uv[:, 1]))
        v_max = float(np.nanmax(uv[:, 1]))

        material_rankings: list[dict[str, object]] = []
        material_to_texture: dict[int, str] = {}
        mode_score = 0.0

        for sub_idx, mat_hash in enumerate(model.material_hashes):
            verts = sorted(sm_vert_sets[sub_idx])
            if not verts:
                continue

            if mat_hash in untextured_material_hashes:
                material_rankings.append(
                    {
                        "submesh_index": sub_idx,
                        "material_hash": mat_hash,
                        "material_hex": f"0x{mat_hash:08X}",
                        "vertex_sample_count": len(verts),
                        "skipped": True,
                        "skip_reason": "material marked untextured",
                    }
                )
                continue

            uv_sub = uv[np.array(verts, dtype=np.int32)]

            tex_scores: list[dict[str, object]] = []
            for tex_rel in texture_paths:
                metrics = sample_texture_metrics(tex_rgb_cache[tex_rel], uv_sub)
                tex_scores.append(
                    {
                        "texture": tex_rel,
                        **metrics,
                    }
                )

            tex_scores.sort(key=lambda x: float(x["score"]), reverse=True)
            top = tex_scores[0]
            forced = False
            if mat_hash in override_material_map:
                forced_tex = override_material_map[mat_hash]
                forced_top = next((x for x in tex_scores if str(x["texture"]) == forced_tex), None)
                if forced_top is None:
                    raise SystemExit(
                        f"Override texture '{forced_tex}' for material 0x{mat_hash:08X} "
                        "was not found under --texture-root"
                    )
                top = forced_top
                forced = True
            material_to_texture[mat_hash] = str(top["texture"])
            mode_score += float(top["score"])
            material_rankings.append(
                {
                    "submesh_index": sub_idx,
                    "material_hash": mat_hash,
                    "material_hex": f"0x{mat_hash:08X}",
                    "vertex_sample_count": len(verts),
                    "top_texture": top["texture"],
                    "top_score": top["score"],
                    "top_metrics": {
                        "mean_luma": top["mean_luma"],
                        "std_luma": top["std_luma"],
                        "dark_ratio_lt20": top["dark_ratio_lt20"],
                        "bright_ratio_gt235": top["bright_ratio_gt235"],
                    },
                    "forced_override": forced,
                    "ranking": tex_scores,
                }
            )

        obj_path, mtl_path = export_preview_obj_mtl(
            output_dir,
            model,
            mode,
            uv,
            material_to_texture,
            texture_paths,
        )

        mode_reports.append(
            {
                "mode": mode.name,
                "type": mode.value_type,
                "lanes": {"u": mode.lane_u, "v": mode.lane_v},
                "model_uv_range": {
                    "u_min": u_min,
                    "u_max": u_max,
                    "v_min": v_min,
                    "v_max": v_max,
                    "finite_ratio": finite_ratio,
                },
                "mode_score_sum": mode_score,
                "preview_obj": str(obj_path),
                "preview_mtl": str(mtl_path),
                "material_rankings": material_rankings,
                "material_to_texture": {f"0x{k:08X}": v for k, v in material_to_texture.items()},
            }
        )

    mode_reports.sort(key=lambda x: float(x["mode_score_sum"]), reverse=True)
    best_auto = mode_reports[0]
    if args.selected_uv_mode:
        best = next((x for x in mode_reports if str(x["mode"]) == args.selected_uv_mode), None)
        if best is None:
            raise SystemExit(
                f"--selected-uv-mode '{args.selected_uv_mode}' not found in evaluated modes: "
                f"{', '.join(m.name for m in uv_modes)}"
            )
    else:
        best = best_auto

    lod_check = compare_lod_materials(
        input_root=input_root,
        archive=args.archive,
        names=["hopper", "l1#hopper", "l2#hopper", "l3#hopper"],
    )

    baseline = {
        "input_root": str(input_root),
        "archive": args.archive,
        "model": args.model,
        "texture_root": str(texture_root),
        "candidate_textures_found": sorted(texture_paths.keys()),
        "override_material_map": {f"0x{k:08X}": v for k, v in sorted(override_material_map.items(), key=lambda x: x[0])},
        "untextured_material_hashes": [f"0x{x:08X}" for x in sorted(untextured_material_hashes)],
        "selected_uv_mode_source": "forced" if args.selected_uv_mode else "auto_score",
        "auto_selected_uv_mode": best_auto["mode"],
        "auto_selected_uv_mode_score_sum": best_auto["mode_score_sum"],
        "selected_uv_mode": best["mode"],
        "selected_uv_mode_score_sum": best["mode_score_sum"],
        "selected_material_texture_map": best["material_to_texture"],
        "lod_material_sets": lod_check,
        "mode_reports": mode_reports,
    }

    out_json = output_dir / f"{args.archive}__{args.model}__baseline.json"
    out_json.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(json.dumps({"baseline_json": str(out_json), "selected_uv_mode": best["mode"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
