#!/usr/bin/env python3
"""Universal Battlezone Gold extraction launcher.

This consolidates validated workflows so audio, textures, and models can be
dumped independently (or in one run).
"""

from __future__ import annotations

import argparse
import csv
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


IS_FROZEN = getattr(sys, "frozen", False)
if IS_FROZEN:
    APP_ROOT = Path(sys.executable).resolve().parent
    TOOLS_DIR = APP_ROOT / "tools"
    REPO_ROOT = APP_ROOT
else:
    TOOLS_DIR = Path(__file__).resolve().parent
    REPO_ROOT = TOOLS_DIR.parent

DEFAULT_KEYWORDS = "tank,enemy,drone,cockpit,weapon,turret,bomber,hopper,nemesis,scout,droid,ufo"


def to_abs_path(path_value: str | Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def resolve_runtime_binary(name: str) -> str | None:
    candidates = [
        TOOLS_DIR / "bin" / name,
        TOOLS_DIR / name,
        REPO_ROOT / name,
    ]
    for local in candidates:
        if local.exists():
            return str(local)
    found = shutil.which(name)
    return found


def has_powershell() -> bool:
    return bool(shutil.which("powershell") or shutil.which("pwsh"))


def powershell_exe() -> str:
    return shutil.which("powershell") or shutil.which("pwsh") or "powershell"


def ps_bool(v: bool) -> str:
    return "$true" if v else "$false"


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    workdir = cwd or REPO_ROOT
    print("$ " + shlex.join(cmd))
    proc = subprocess.run(cmd, cwd=workdir)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def ps_single_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def run_powershell_file_with_switches(
    script_path: Path,
    named_args: dict[str, str],
    switch_args: dict[str, bool],
    cwd: Path | None = None,
) -> None:
    parts = ["&", ps_single_quote(str(script_path))]
    for key, value in named_args.items():
        parts.append(f"-{key}")
        parts.append(ps_single_quote(value))
    for key, value in switch_args.items():
        parts.append(f"-{key}:${'true' if value else 'false'}")
    command_text = " ".join(parts)
    cmd = [
        powershell_exe(),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        command_text,
    ]
    run_command(cmd, cwd=cwd)


def sniff_texture_extension(path: Path) -> str | None:
    try:
        head = path.read_bytes()[:64]
    except Exception:
        return None
    if len(head) < 4:
        return None
    if head.startswith(b"DDS "):
        return ".dds"
    if head.startswith(b"BM"):
        return ".bmp"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if head.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return ".gif"
    if len(head) >= 18:
        cmap = head[1]
        image_type = head[2]
        width = int.from_bytes(head[12:14], "little", signed=False)
        height = int.from_bytes(head[14:16], "little", signed=False)
        bpp = head[16]
        if cmap in {0, 1} and image_type in {1, 2, 3, 9, 10, 11} and 0 < width <= 16384 and 0 < height <= 16384 and bpp in {8, 16, 24, 32}:
            return ".tga"
    return None


def sniff_audio_extension(path: Path) -> str | None:
    try:
        head = path.read_bytes()[:512]
    except Exception:
        return None
    if len(head) < 4:
        return None
    if head.startswith(b"BKHD"):
        return ".bnk"
    if head.startswith(b"OggS"):
        return ".ogg"
    if head.startswith(b"fLaC"):
        return ".flac"
    if head.startswith(b"ID3"):
        return ".mp3"
    if len(head) >= 3 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return ".mp3"
    if len(head) >= 12 and head[0:4] == b"RIFF":
        if b"WAVE" in head:
            return ".wav"
        # Unknown RIFF audio-ish payload; preserve as .wem for wwise-style files.
        return ".wem"
    return None


def rel_with_extension(rel: Path, extension: str) -> Path:
    if rel.suffix:
        return rel
    return rel.with_suffix(extension)


def run_extract(args: argparse.Namespace) -> None:
    script = to_abs_path("tools/extract-asura-batch.ps1")
    if not has_powershell():
        raise SystemExit("PowerShell is required for extraction task.")
    native_extractor = args.native_extractor
    if IS_FROZEN and native_extractor == "tools/extract-asura-native.py":
        native_exe = resolve_runtime_binary("extract-asura-native.exe")
        if native_exe:
            native_extractor = native_exe
    quickbms_exe = args.quickbms_exe or resolve_runtime_binary("quickbms_4gb_files.exe") or ""
    cmd = [
        powershell_exe(),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-GameRoot",
        str(to_abs_path(args.game_root)),
        "-ExtractRoot",
        str(to_abs_path(args.extract_root)),
        "-LogsRoot",
        str(to_abs_path(args.logs_root)),
        "-Extractor",
        args.extractor,
        "-PythonExe",
        args.python_exe,
        "-NativeExtractor",
        str(to_abs_path(native_extractor)),
        "-TargetSet",
        args.target_set,
        f"-SkipExisting:{ps_bool(args.skip_existing)}",
    ]
    if quickbms_exe:
        cmd.extend(["-QuickBmsExe", str(to_abs_path(quickbms_exe))])
    if args.bms_script:
        cmd.extend(["-BmsScript", str(to_abs_path(args.bms_script))])
    run_command(cmd)


def run_textures(args: argparse.Namespace) -> None:
    if args.texture_discovery == "headers":
        run_textures_header_discovery(args)
        return
    script = to_abs_path("tools/convert-textures-to-png.ps1")
    if not has_powershell():
        raise SystemExit("PowerShell is required for texture conversion task.")
    texconv_exe = args.texconv_exe or resolve_runtime_binary("texconv.exe") or ""
    named_args = {
        "InputRoot": str(to_abs_path(args.extract_root)),
        "OutputRoot": str(to_abs_path(args.texture_output_root)),
        "ReportPath": str(to_abs_path(args.texture_report_path)),
    }
    if texconv_exe:
        named_args["TexconvExe"] = str(to_abs_path(texconv_exe))
    switch_args = {
        "SkipExisting": bool(args.skip_existing),
        "UseTexconvFallback": bool(args.use_texconv_fallback),
        "PreferTexconv": bool(args.prefer_texconv),
    }
    run_powershell_file_with_switches(script_path=script, named_args=named_args, switch_args=switch_args)


def run_textures_header_discovery(args: argparse.Namespace) -> None:
    in_root = to_abs_path(args.extract_root)
    stage_root = to_abs_path(args.texture_stage_root)
    discovery_report = to_abs_path(args.texture_discovery_report)
    stage_root.mkdir(parents=True, exist_ok=True)
    discovery_report.parent.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    rows: list[dict[str, str]] = []

    for src in in_root.rglob("*"):
        if not src.is_file():
            continue
        tex_ext = sniff_texture_extension(src)
        if not tex_ext:
            continue
        rel = src.relative_to(in_root)
        out_rel = rel_with_extension(rel, tex_ext)
        dst = stage_root / out_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and dst.exists():
            skipped += 1
            status = "skipped"
        else:
            shutil.copy2(src, dst)
            copied += 1
            status = "copied"
        rows.append(
            {
                "source": str(src),
                "relative_path": str(rel).replace("\\", "/"),
                "detected_extension": tex_ext,
                "staged_path": str(out_rel).replace("\\", "/"),
                "status": status,
            }
        )

    with discovery_report.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "relative_path", "detected_extension", "staged_path", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        "Texture header discovery complete: "
        f"copied={copied} skipped={skipped} stage_root={stage_root} report={discovery_report}"
    )

    # Convert staged texture files to PNG through existing conversion pipeline.
    script = to_abs_path("tools/convert-textures-to-png.ps1")
    if not has_powershell():
        raise SystemExit("PowerShell is required for texture conversion task.")
    texconv_exe = args.texconv_exe or resolve_runtime_binary("texconv.exe") or ""
    named_args = {
        "InputRoot": str(stage_root),
        "OutputRoot": str(to_abs_path(args.texture_output_root)),
        "ReportPath": str(to_abs_path(args.texture_report_path)),
    }
    if texconv_exe:
        named_args["TexconvExe"] = str(to_abs_path(texconv_exe))
    switch_args = {
        "SkipExisting": bool(args.skip_existing),
        "UseTexconvFallback": bool(args.use_texconv_fallback),
        "PreferTexconv": bool(args.prefer_texconv),
    }
    run_powershell_file_with_switches(script_path=script, named_args=named_args, switch_args=switch_args)


def run_models(args: argparse.Namespace) -> None:
    py = args.python_exe
    in_root = str(to_abs_path(args.extract_root))
    out_root = to_abs_path(args.model_output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    discovery_mode = args.model_discovery
    model_keywords = args.model_keywords if discovery_mode == "keywords" else ""

    backends: list[str]
    if args.model_backend == "both":
        backends = ["hskn", "extless"]
    else:
        backends = [args.model_backend]

    if "hskn" in backends:
        out_dir = out_root / "hskn"
        out_dir.mkdir(parents=True, exist_ok=True)
        hskn_exe = resolve_runtime_binary("export-hskn-obj-candidates.exe") if IS_FROZEN else None
        if IS_FROZEN and hskn_exe:
            cmd = [
                hskn_exe,
                "--input-root",
                in_root,
                "--output-dir",
                str(out_dir),
                "--keywords",
                model_keywords,
                "--max-files",
                str(args.model_max_files),
                "--discovery-mode",
                discovery_mode,
                "--min-triangles",
                str(args.model_min_triangles),
                "--topology-mode",
                args.hskn_topology_mode,
            ]
        elif IS_FROZEN and not hskn_exe:
            raise SystemExit("Frozen runtime missing export-hskn-obj-candidates.exe")
        else:
            cmd = [
                py,
                str(to_abs_path("tools/export-hskn-obj-candidates.py")),
                "--input-root",
                in_root,
                "--output-dir",
                str(out_dir),
                "--keywords",
                model_keywords,
                "--max-files",
                str(args.model_max_files),
                "--discovery-mode",
                discovery_mode,
                "--min-triangles",
                str(args.model_min_triangles),
                "--topology-mode",
                args.hskn_topology_mode,
            ]
        if args.include_environment:
            cmd.append("--include-environment")
        if args.model_edge_prune_median_multiplier is not None:
            cmd.extend(
                [
                    "--edge-prune-median-multiplier",
                    str(args.model_edge_prune_median_multiplier),
                ]
            )
        run_command(cmd)

    if "extless" in backends:
        out_dir = out_root / "extless"
        out_dir.mkdir(parents=True, exist_ok=True)
        ext_exe = resolve_runtime_binary("export-model-obj-candidates.exe") if IS_FROZEN else None
        if IS_FROZEN and ext_exe:
            cmd = [
                ext_exe,
                "--input-root",
                in_root,
                "--output-dir",
                str(out_dir),
                "--keywords",
                model_keywords,
                "--max-files",
                str(args.model_max_files),
                "--discovery-mode",
                discovery_mode,
                "--top-modes",
                str(args.extless_top_modes),
            ]
        elif IS_FROZEN and not ext_exe:
            raise SystemExit("Frozen runtime missing export-model-obj-candidates.exe")
        else:
            cmd = [
                py,
                str(to_abs_path("tools/export-model-obj-candidates.py")),
                "--input-root",
                in_root,
                "--output-dir",
                str(out_dir),
                "--keywords",
                model_keywords,
                "--max-files",
                str(args.model_max_files),
                "--discovery-mode",
                discovery_mode,
                "--top-modes",
                str(args.extless_top_modes),
            ]
        if args.include_environment:
            cmd.append("--include-environment")
        if args.model_edge_prune_median_multiplier is not None:
            cmd.extend(
                [
                    "--edge-prune-median-multiplier",
                    str(args.model_edge_prune_median_multiplier),
                ]
            )
        run_command(cmd)


@dataclass
class AudioDumpResult:
    copied: int
    skipped: int
    report_path: Path


def run_audio(args: argparse.Namespace) -> AudioDumpResult:
    if args.audio_discovery == "headers":
        return run_audio_header_discovery(args)

    in_root = to_abs_path(args.extract_root)
    out_root = to_abs_path(args.audio_output_root)
    report_path = to_abs_path(args.audio_report_path)
    out_root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    exts = {x.strip().lower() for x in args.audio_extensions.split(",") if x.strip()}
    if not exts:
        raise SystemExit("No audio extensions provided.")
    exts = {x if x.startswith(".") else f".{x}" for x in exts}

    copied = 0
    skipped = 0
    rows: list[dict[str, str]] = []

    for src in in_root.rglob("*"):
        if not src.is_file():
            continue
        if src.suffix.lower() not in exts:
            continue
        rel = src.relative_to(in_root)
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and dst.exists():
            skipped += 1
            status = "skipped"
        else:
            shutil.copy2(src, dst)
            copied += 1
            status = "copied"
        rows.append(
            {
                "source": str(src),
                "relative_path": str(rel).replace("\\", "/"),
                "destination": str(dst),
                "status": status,
            }
        )

    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "relative_path", "destination", "status"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Audio dump complete: copied={copied} skipped={skipped} report={report_path}")
    return AudioDumpResult(copied=copied, skipped=skipped, report_path=report_path)


def run_audio_header_discovery(args: argparse.Namespace) -> AudioDumpResult:
    in_root = to_abs_path(args.extract_root)
    out_root = to_abs_path(args.audio_output_root)
    report_path = to_abs_path(args.audio_report_path)
    out_root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    rows: list[dict[str, str]] = []

    for src in in_root.rglob("*"):
        if not src.is_file():
            continue
        aud_ext = sniff_audio_extension(src)
        if not aud_ext:
            continue

        rel = src.relative_to(in_root)
        out_rel = rel_with_extension(rel, aud_ext)
        dst = out_root / out_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and dst.exists():
            skipped += 1
            status = "skipped"
        else:
            shutil.copy2(src, dst)
            copied += 1
            status = "copied"

        rows.append(
            {
                "source": str(src),
                "relative_path": str(rel).replace("\\", "/"),
                "detected_extension": aud_ext,
                "destination": str(dst),
                "status": status,
            }
        )

    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "relative_path", "detected_extension", "destination", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Audio header dump complete: copied={copied} skipped={skipped} report={report_path}")
    return AudioDumpResult(copied=copied, skipped=skipped, report_path=report_path)


def add_shared_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--game-root", default="work/Battlezone")
    parser.add_argument("--extract-root", default="work/extracted/full_all")
    parser.add_argument("--logs-root", default="work/logs/batch_all")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")


def add_extract_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--extractor", choices=("Native", "QuickBMS", "Auto"), default="Native")
    parser.add_argument("--native-extractor", default="tools/extract-asura-native.py")
    parser.add_argument("--quickbms-exe", default="")
    parser.add_argument("--bms-script", default="tools/bms/asura.bms")
    parser.add_argument("--target-set", choices=("Legacy", "AllPcVariants"), default="AllPcVariants")


def add_texture_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--texture-discovery", choices=("headers", "extensions"), default="headers")
    parser.add_argument(
        "--texture-stage-root",
        default="work/preview/texture_stage_header",
        help="Used when --texture-discovery=headers. Extension-normalized staging area for conversion.",
    )
    parser.add_argument(
        "--texture-discovery-report",
        default="notes/texture-header-discovery-report.csv",
        help="Used when --texture-discovery=headers.",
    )
    parser.add_argument("--texture-output-root", default="work/preview/png_all_texconv")
    parser.add_argument(
        "--texture-report-path",
        default="notes/texture-conversion-report-full-all-texconv.csv",
    )
    parser.add_argument("--texconv-exe", default="")
    parser.add_argument("--use-texconv-fallback", dest="use_texconv_fallback", action="store_true", default=True)
    parser.add_argument("--no-use-texconv-fallback", dest="use_texconv_fallback", action="store_false")
    parser.add_argument("--prefer-texconv", action="store_true", default=False)


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-output-root", default="work/preview/model_obj_universal")
    parser.add_argument("--model-backend", choices=("hskn", "extless", "both"), default="hskn")
    parser.add_argument("--model-discovery", choices=("headers", "keywords"), default="headers")
    parser.add_argument("--model-keywords", default=DEFAULT_KEYWORDS)
    parser.add_argument("--model-max-files", type=int, default=0, help="0 means no cap (export all discovered files).")
    parser.add_argument("--model-min-triangles", type=int, default=20)
    parser.add_argument("--hskn-topology-mode", default="auto")
    parser.add_argument("--extless-top-modes", type=int, default=1)
    parser.add_argument("--model-edge-prune-median-multiplier", type=float, default=20.0)
    parser.add_argument("--include-environment", action="store_true", default=False)


def add_audio_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--audio-discovery", choices=("headers", "extensions"), default="headers")
    parser.add_argument("--audio-output-root", default="work/preview/audio_dump")
    parser.add_argument("--audio-report-path", default="notes/audio-dump-report.csv")
    parser.add_argument("--audio-extensions", default=".wav,.ogg,.mp3,.wem,.bnk,.flac,.aif,.aiff")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Universal Battlezone Gold extractor launcher.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Extract archives")
    add_shared_io_args(p_extract)
    add_extract_args(p_extract)

    p_textures = sub.add_parser("textures", help="Convert texture previews to PNG")
    add_shared_io_args(p_textures)
    add_texture_args(p_textures)

    p_models = sub.add_parser("models", help="Export model OBJ previews")
    add_shared_io_args(p_models)
    add_model_args(p_models)

    p_audio = sub.add_parser("audio", help="Dump audio files from extracted data")
    add_shared_io_args(p_audio)
    add_audio_args(p_audio)

    p_run = sub.add_parser("run", help="Run multiple tasks in sequence")
    add_shared_io_args(p_run)
    add_extract_args(p_run)
    add_texture_args(p_run)
    add_model_args(p_run)
    add_audio_args(p_run)
    p_run.add_argument(
        "--tasks",
        default="extract,textures,models,audio",
        help="Comma-separated tasks from: extract,textures,models,audio",
    )
    p_runtime = sub.add_parser("runtime", help="Show resolved runtime tool paths")
    add_shared_io_args(p_runtime)
    return parser


def parse_task_list(raw: str) -> list[str]:
    valid = {"extract", "textures", "models", "audio"}
    out = [x.strip().lower() for x in raw.split(",") if x.strip()]
    for x in out:
        if x not in valid:
            raise SystemExit(f"Unsupported task '{x}'. Valid tasks: {', '.join(sorted(valid))}")
    return out


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "extract":
        run_extract(args)
        return 0
    if args.command == "textures":
        run_textures(args)
        return 0
    if args.command == "models":
        run_models(args)
        return 0
    if args.command == "audio":
        run_audio(args)
        return 0
    if args.command == "run":
        tasks = parse_task_list(args.tasks)
        for task in tasks:
            print(f"\n=== Running task: {task} ===")
            if task == "extract":
                run_extract(args)
            elif task == "textures":
                run_textures(args)
            elif task == "models":
                run_models(args)
            elif task == "audio":
                run_audio(args)
        return 0
    if args.command == "runtime":
        rows = {
            "repo_root": str(REPO_ROOT),
            "tools_dir": str(TOOLS_DIR),
            "frozen": str(IS_FROZEN),
            "powershell": powershell_exe(),
            "ffmpeg": resolve_runtime_binary("ffmpeg.exe") or resolve_runtime_binary("ffmpeg") or "",
            "texconv": resolve_runtime_binary("texconv.exe") or "",
            "quickbms": resolve_runtime_binary("quickbms_4gb_files.exe")
            or resolve_runtime_binary("quickbms.exe")
            or "",
        }
        for k, v in rows.items():
            print(f"{k}={v}")
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
