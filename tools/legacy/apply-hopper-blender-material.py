#!/usr/bin/env python3
"""Apply a Battlezone-style hopper material node setup in Blender.

Usage (inside Blender):
1) Import hopper OBJ/MTL (for example hopper__f16_12_13.obj).
2) Select hopper mesh objects.
3) Run this script from Blender Text Editor, or from CLI:
   blender --python tools/apply-hopper-blender-material.py -- \
     --baseline-json notes/hopper_texture_baseline/chars__actors_pc__hopper__baseline.json \
     --apply-to selected
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_BASELINE_JSON = REPO_ROOT / "notes/hopper_texture_baseline/chars__actors_pc__hopper__baseline.json"
DEFAULT_MASK_REL = "chars__actors_pc/graphics/objects/vehicles/vehicle_colour_albedo_roughness.png"


def _script_argv() -> list[str]:
    if "--" not in sys.argv:
        return []
    idx = sys.argv.index("--")
    return sys.argv[idx + 1 :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply hopper material node graph in Blender.")
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON)
    parser.add_argument("--apply-to", choices=("selected", "all"), default="selected")
    parser.add_argument("--team-color", default="#ff2a2a", help="Hex RGB, for example #ff2a2a")
    parser.add_argument("--tint-strength", type=float, default=0.9)
    parser.add_argument("--emissive-strength", type=float, default=4.0)
    parser.add_argument("--emissive-threshold", type=float, default=0.82)
    parser.add_argument("--mask-texture-rel", default=DEFAULT_MASK_REL)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(_script_argv())


def parse_hex_rgb(text: str) -> tuple[float, float, float, float]:
    s = text.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        raise ValueError(f"Bad team color '{text}', expected 6 hex chars")
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return (r, g, b, 1.0)


def find_hash_from_material_name(name: str) -> str | None:
    m = re.search(r"mat_([0-9A-Fa-f]{8})", name)
    if not m:
        return None
    return f"0x{m.group(1).upper()}"


def load_image(path: Path) -> bpy.types.Image | None:
    if not path.exists():
        return None
    return bpy.data.images.load(str(path), check_existing=True)


def clear_tree(tree: bpy.types.NodeTree) -> None:
    for node in list(tree.nodes):
        tree.nodes.remove(node)


def apply_material_graph(
    mat: bpy.types.Material,
    base_image: bpy.types.Image | None,
    mask_image: bpy.types.Image | None,
    team_color: tuple[float, float, float, float],
    tint_strength: float,
    emissive_strength: float,
    emissive_threshold: float,
) -> None:
    mat.use_nodes = True
    nt = mat.node_tree
    if nt is None:
        return
    clear_tree(nt)

    nodes = nt.nodes
    links = nt.links

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (980, 160)

    add_shader = nodes.new("ShaderNodeAddShader")
    add_shader.location = (780, 160)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (560, 260)
    bsdf.inputs["Roughness"].default_value = 0.55
    bsdf.inputs["Metallic"].default_value = 0.15

    emission = nodes.new("ShaderNodeEmission")
    emission.location = (560, 40)

    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-1040, 220)

    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-840, 220)

    base_tex = nodes.new("ShaderNodeTexImage")
    base_tex.location = (-620, 340)
    base_tex.label = "Base Atlas"
    if base_image:
        base_tex.image = base_image

    mask_tex = nodes.new("ShaderNodeTexImage")
    mask_tex.location = (-620, 120)
    mask_tex.label = "Team Mask"
    if mask_image:
        mask_tex.image = mask_image
        if mask_tex.image.colorspace_settings is not None:
            mask_tex.image.colorspace_settings.name = "Non-Color"

    team_rgb = nodes.new("ShaderNodeRGB")
    team_rgb.location = (-620, -80)
    team_rgb.outputs[0].default_value = team_color

    sep_rgb = nodes.new("ShaderNodeSeparateRGB")
    sep_rgb.location = (-400, 110)

    tint_factor = nodes.new("ShaderNodeMath")
    tint_factor.location = (-180, 90)
    tint_factor.operation = "MULTIPLY"
    tint_factor.inputs[1].default_value = float(tint_strength)

    tint_mul = nodes.new("ShaderNodeMixRGB")
    tint_mul.location = (-180, 320)
    tint_mul.blend_type = "MULTIPLY"
    tint_mul.inputs["Fac"].default_value = 1.0

    tint_mix = nodes.new("ShaderNodeMixRGB")
    tint_mix.location = (40, 300)
    tint_mix.blend_type = "MIX"

    to_bw = nodes.new("ShaderNodeRGBToBW")
    to_bw.location = (-180, -80)

    bright_gate = nodes.new("ShaderNodeMath")
    bright_gate.location = (30, -70)
    bright_gate.operation = "GREATER_THAN"
    bright_gate.inputs[1].default_value = float(emissive_threshold)

    emissive_mul = nodes.new("ShaderNodeMath")
    emissive_mul.location = (240, -70)
    emissive_mul.operation = "MULTIPLY"
    emissive_mul.inputs[1].default_value = float(emissive_strength)

    links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], base_tex.inputs["Vector"])
    links.new(mapping.outputs["Vector"], mask_tex.inputs["Vector"])

    if mask_image:
        links.new(mask_tex.outputs["Color"], sep_rgb.inputs["Image"])
        links.new(sep_rgb.outputs["R"], tint_factor.inputs[0])
    else:
        tint_factor.inputs[0].default_value = 0.0

    links.new(base_tex.outputs["Color"], tint_mul.inputs["Color1"])
    links.new(team_rgb.outputs["Color"], tint_mul.inputs["Color2"])
    links.new(base_tex.outputs["Color"], tint_mix.inputs["Color1"])
    links.new(tint_mul.outputs["Color"], tint_mix.inputs["Color2"])
    links.new(tint_factor.outputs["Value"], tint_mix.inputs["Fac"])

    links.new(tint_mix.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(tint_mix.outputs["Color"], emission.inputs["Color"])

    links.new(base_tex.outputs["Color"], to_bw.inputs["Color"])
    links.new(to_bw.outputs["Val"], bright_gate.inputs[0])
    links.new(bright_gate.outputs["Value"], emissive_mul.inputs[0])
    links.new(emissive_mul.outputs["Value"], emission.inputs["Strength"])

    links.new(bsdf.outputs["BSDF"], add_shader.inputs[0])
    links.new(emission.outputs["Emission"], add_shader.inputs[1])
    links.new(add_shader.outputs["Shader"], out.inputs["Surface"])


def target_materials(mode: str) -> list[bpy.types.Material]:
    mats: list[bpy.types.Material] = []
    if mode == "all":
        return [m for m in bpy.data.materials]

    for obj in bpy.context.selected_objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            if slot.material and slot.material not in mats:
                mats.append(slot.material)
    return mats


def main() -> int:
    args = parse_args()
    team_color = parse_hex_rgb(args.team_color)

    baseline_path = args.baseline_json.resolve()
    if not baseline_path.exists():
        raise SystemExit(f"Baseline JSON not found: {baseline_path}")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    texture_root = Path(baseline["texture_root"]).resolve()
    selected_map: dict[str, str] = baseline.get("selected_material_texture_map", {})
    untextured = set(baseline.get("untextured_material_hashes", []))

    mask_rel = str(args.mask_texture_rel).replace("\\", "/")
    mask_path = texture_root / mask_rel
    mask_image = load_image(mask_path)
    if mask_image is None:
        print(f"[warn] Mask texture not found: {mask_path}")

    mats = target_materials(args.apply_to)
    if not mats:
        print("[warn] No target materials found (select hopper meshes or use --apply-to all).")
        return 0

    updated = 0
    skipped = 0

    for mat in mats:
        mat_hash = find_hash_from_material_name(mat.name)
        if mat_hash is None:
            skipped += 1
            continue
        if mat_hash in untextured:
            skipped += 1
            continue

        tex_rel = selected_map.get(mat_hash)
        if not tex_rel:
            skipped += 1
            continue

        tex_path = texture_root / tex_rel
        base_image = load_image(tex_path)
        if base_image is None:
            print(f"[warn] Base texture missing for {mat.name}: {tex_path}")
            skipped += 1
            continue

        if args.dry_run:
            print(f"[dry-run] would update {mat.name} -> {tex_rel}")
            updated += 1
            continue

        apply_material_graph(
            mat=mat,
            base_image=base_image,
            mask_image=mask_image,
            team_color=team_color,
            tint_strength=args.tint_strength,
            emissive_strength=args.emissive_strength,
            emissive_threshold=args.emissive_threshold,
        )
        updated += 1
        print(f"[ok] updated {mat.name} using {tex_rel}")

    print(f"[done] updated={updated} skipped={skipped} apply_to={args.apply_to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
