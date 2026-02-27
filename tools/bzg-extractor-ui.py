#!/usr/bin/env python3
"""Desktop UI wrapper for tools/bzg-extractor.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BACKEND = SCRIPT_DIR / "bzg-extractor.py"
BUILD_PS1 = SCRIPT_DIR / "build-standalone.ps1"
UI_STATE_PATH = SCRIPT_DIR / "bzg-extractor-ui-state.json"
IS_FROZEN = getattr(sys, "frozen", False)


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Battlezone Gold Extractor")
        self.root.geometry("1080x780")
        self.root.minsize(980, 700)
        self.proc: subprocess.Popen[str] | None = None
        self.ui_state_path = self._resolve_ui_state_path()
        self._theme = self._theme_palette()
        self._title_font = ("Segoe UI", 20, "bold")
        self._subtitle_font = ("Segoe UI", 10, "bold")
        self._heading_font = ("Segoe UI", 16, "bold")
        self._ui_font = ("Segoe UI", 10)

        self.task_extract = tk.BooleanVar(value=True)
        self.task_textures = tk.BooleanVar(value=True)
        self.task_models = tk.BooleanVar(value=True)
        self.task_audio = tk.BooleanVar(value=False)

        self.game_root = tk.StringVar(value="work/Battlezone")
        self.extract_root = tk.StringVar(value="work/extracted/full_all")
        self.logs_root = tk.StringVar(value="work/logs/batch_all")
        self.extractor = tk.StringVar(value="Native")
        self.skip_existing = tk.BooleanVar(value=True)

        self.texture_output_root = tk.StringVar(value="work/preview/png_all_texconv")
        self.texture_report_path = tk.StringVar(value="notes/texture-conversion-report-full-all-texconv.csv")
        self.texture_discovery = tk.StringVar(value="headers")
        self.texture_stage_root = tk.StringVar(value="work/preview/texture_stage_header")
        self.texture_discovery_report = tk.StringVar(value="notes/texture-header-discovery-report.csv")
        self.texconv_exe = tk.StringVar(value="")
        self.use_texconv_fallback = tk.BooleanVar(value=True)
        self.prefer_texconv = tk.BooleanVar(value=False)

        self.model_output_root = tk.StringVar(value="work/preview/model_obj_universal")
        self.model_backend = tk.StringVar(value="hskn")
        self.model_discovery = tk.StringVar(value="headers")
        self.model_keywords = tk.StringVar(value="tank,enemy,drone,cockpit,weapon,turret,bomber,hopper,nemesis,scout,droid,ufo")
        self.model_max_files = tk.StringVar(value="0")
        self.model_min_triangles = tk.StringVar(value="20")
        self.hskn_topology_mode = tk.StringVar(value="auto")
        self.extless_top_modes = tk.StringVar(value="1")
        self.model_edge_prune = tk.StringVar(value="20")
        self.include_environment = tk.BooleanVar(value=False)

        self.quickbms_exe = tk.StringVar(value="")
        self.bms_script = tk.StringVar(value="tools/bms/asura.bms")
        self.target_set = tk.StringVar(value="AllPcVariants")

        self.audio_output_root = tk.StringVar(value="work/preview/audio_dump")
        self.audio_report_path = tk.StringVar(value="notes/audio-dump-report.csv")
        self.audio_discovery = tk.StringVar(value="headers")
        self.audio_extensions = tk.StringVar(value=".wav,.ogg,.mp3,.wem,.bnk,.flac,.aif,.aiff")

        self.dist_dir = tk.StringVar(value="dist")
        self.onefile = tk.BooleanVar(value=True)
        self.windowed = tk.BooleanVar(value=True)
        self.bundle_runtime_bin = tk.BooleanVar(value=True)
        self.bundle_bms = tk.BooleanVar(value=True)

        self.run_button: ttk.Button | None = None
        self.stop_button: ttk.Button | None = None
        self.package_button: ttk.Button | None = None
        self.progress: ttk.Progressbar | None = None
        self.log_box: ScrolledText | None = None

        self._init_theme()
        self._build()
        self._load_state()

    def _theme_palette(self) -> dict[str, str]:
        return {
            "bg": "#050A16",
            "panel": "#0C1526",
            "panel_alt": "#101E37",
            "entry_bg": "#0A1222",
            "text": "#DCEBFF",
            "muted": "#7FA1CC",
            "cyan": "#61D6FF",
            "cyan_dim": "#2A8BB8",
            "gold": "#FFB13B",
            "gold_dim": "#CC7A18",
            "danger": "#FF6969",
            "log_bg": "#040811",
        }

    def _best_font(self, candidates: list[str], size: int, weight: str = "normal") -> tuple[str, int, str]:
        families = set(tkfont.families(self.root))
        for name in candidates:
            if name in families:
                return (name, size, weight)
        return ("Segoe UI", size, weight)

    def _init_theme(self) -> None:
        self.root.configure(bg=self._theme["bg"])
        self._title_font = self._best_font(["Orbitron", "Eurostile", "BankGothic Md BT", "Segoe UI"], 22, "bold")
        self._subtitle_font = self._best_font(["Orbitron", "Eurostile", "Segoe UI"], 10, "bold")
        self._heading_font = self._best_font(["Orbitron", "Eurostile", "Segoe UI"], 16, "bold")
        self._ui_font = self._best_font(["Bahnschrift", "Segoe UI", "Tahoma"], 10)

        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure(".", font=self._ui_font)
        style.configure("TFrame", background=self._theme["panel"])
        style.configure("TLabel", background=self._theme["panel"], foreground=self._theme["text"])
        style.configure(
            "TLabelframe",
            background=self._theme["panel"],
            borderwidth=1,
            relief="solid",
            bordercolor=self._theme["cyan_dim"],
            lightcolor=self._theme["cyan_dim"],
            darkcolor=self._theme["cyan_dim"],
        )
        style.configure(
            "TLabelframe.Label",
            background=self._theme["panel"],
            foreground=self._theme["gold"],
            font=self._subtitle_font,
        )
        style.configure(
            "TNotebook",
            background=self._theme["bg"],
            borderwidth=0,
            tabmargins=(4, 0, 4, 0),
        )
        style.configure(
            "TNotebook.Tab",
            background=self._theme["panel_alt"],
            foreground=self._theme["muted"],
            padding=(14, 7),
            borderwidth=1,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self._theme["panel"]), ("active", self._theme["panel"])],
            foreground=[("selected", self._theme["gold"]), ("active", self._theme["cyan"])],
        )

        style.configure(
            "TButton",
            background=self._theme["panel_alt"],
            foreground=self._theme["text"],
            borderwidth=1,
            focusthickness=1,
            focuscolor=self._theme["cyan"],
            padding=(9, 5),
        )
        style.map(
            "TButton",
            background=[("active", self._theme["cyan_dim"]), ("pressed", self._theme["cyan_dim"])],
            foreground=[("disabled", "#75839A"), ("active", self._theme["text"])],
        )
        style.configure("Run.TButton", background="#154066", foreground=self._theme["cyan"])
        style.map("Run.TButton", background=[("active", "#1F5D95"), ("pressed", "#1F5D95")], foreground=[("active", "#EAF6FF")])
        style.configure("Stop.TButton", background="#56202D", foreground=self._theme["danger"])
        style.map("Stop.TButton", background=[("active", "#7A2A3E"), ("pressed", "#7A2A3E")], foreground=[("active", "#FFDADB")])

        style.configure(
            "TCheckbutton",
            background=self._theme["panel"],
            foreground=self._theme["text"],
        )
        style.map(
            "TCheckbutton",
            foreground=[("active", self._theme["cyan"]), ("selected", self._theme["gold"])],
        )
        style.configure("TEntry", fieldbackground=self._theme["entry_bg"], foreground=self._theme["text"])
        style.configure(
            "TCombobox",
            fieldbackground=self._theme["entry_bg"],
            background=self._theme["panel_alt"],
            foreground=self._theme["text"],
            arrowcolor=self._theme["gold"],
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", self._theme["entry_bg"])],
            foreground=[("readonly", self._theme["text"])],
            selectbackground=[("readonly", self._theme["panel_alt"])],
            selectforeground=[("readonly", self._theme["text"])],
        )
        style.configure(
            "TProgressbar",
            troughcolor=self._theme["entry_bg"],
            background=self._theme["gold"],
            bordercolor=self._theme["cyan_dim"],
            lightcolor=self._theme["gold"],
            darkcolor=self._theme["gold_dim"],
        )

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        header = tk.Frame(frame, bg=self._theme["bg"], bd=1, highlightthickness=1, highlightbackground=self._theme["cyan_dim"])
        header.pack(fill=tk.X, pady=(0, 10))
        tk.Label(
            header,
            text="BATTLEZONE",
            bg=self._theme["bg"],
            fg=self._theme["cyan"],
            font=self._title_font,
        ).pack(anchor=tk.W, padx=12, pady=(6, 0))
        tk.Label(
            header,
            text="GOLD EDITION EXTRACTOR",
            bg=self._theme["bg"],
            fg=self._theme["gold"],
            font=self._subtitle_font,
        ).pack(anchor=tk.W, padx=12, pady=(0, 8))

        tabs = ttk.Notebook(frame)
        tabs.pack(fill=tk.BOTH, expand=True, pady=(10, 8))

        tab_pipeline = ttk.Frame(tabs, padding=8)
        tab_advanced = ttk.Frame(tabs, padding=8)
        tabs.add(tab_pipeline, text="Pipeline")
        tabs.add(tab_advanced, text="Advanced")
        tab_package: ttk.Frame | None = None
        if not IS_FROZEN:
            tab_package = ttk.Frame(tabs, padding=8)
            tabs.add(tab_package, text="Standalone")

        self._build_pipeline_tab(tab_pipeline)
        self._build_advanced_tab(tab_advanced)
        if tab_package is not None:
            self._build_package_tab(tab_package)

        actions = ttk.Frame(frame)
        actions.pack(fill=tk.X, pady=(2, 6))
        self.run_button = ttk.Button(actions, text="Run Selected Tasks", command=self.start_run, style="Run.TButton")
        self.run_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(actions, text="Stop", command=self.stop_run, state=tk.DISABLED, style="Stop.TButton")
        self.stop_button.pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="Runtime Check", command=self.runtime_check).pack(side=tk.LEFT)
        ttk.Button(actions, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="Save UI State", command=self.save_state).pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=(0, 6))

        self.log_box = ScrolledText(frame, height=16, wrap=tk.WORD)
        self.log_box.configure(
            bg=self._theme["log_bg"],
            fg=self._theme["text"],
            insertbackground=self._theme["cyan"],
            selectbackground=self._theme["cyan_dim"],
            selectforeground=self._theme["text"],
            relief=tk.FLAT,
            borderwidth=8,
            font=self._best_font(["Consolas", "Cascadia Mono", "Courier New"], 10),
        )
        self.log_box.pack(fill=tk.BOTH, expand=True)
        self.log("[ready] Configure tasks and run.")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_pipeline_tab(self, parent: ttk.Frame) -> None:
        tasks = ttk.LabelFrame(parent, text="Tasks", padding=8)
        tasks.pack(fill=tk.X)
        ttk.Checkbutton(tasks, text="Extract Archives", variable=self.task_extract).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(tasks, text="Convert Textures", variable=self.task_textures).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(tasks, text="Export Models", variable=self.task_models).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(tasks, text="Dump Audio", variable=self.task_audio).pack(side=tk.LEFT, padx=8)

        paths = ttk.LabelFrame(parent, text="Core Paths", padding=8)
        paths.pack(fill=tk.X, pady=8)
        self._path_row(paths, "Game Root", self.game_root, is_dir=True)
        self._path_row(paths, "Extract Root", self.extract_root, is_dir=True)
        self._path_row(paths, "Logs Root", self.logs_root, is_dir=True)
        self._path_row(paths, "Texture Output", self.texture_output_root, is_dir=True)
        self._path_row(paths, "Model Output", self.model_output_root, is_dir=True)
        self._path_row(paths, "Audio Output", self.audio_output_root, is_dir=True)

        reports = ttk.LabelFrame(parent, text="Reports", padding=8)
        reports.pack(fill=tk.X)
        self._path_row(reports, "Texture Report", self.texture_report_path, is_dir=False, save_file=True)
        self._path_row(reports, "Texture Detect", self.texture_discovery_report, is_dir=False, save_file=True)
        self._path_row(reports, "Audio Report", self.audio_report_path, is_dir=False, save_file=True)

        basics = ttk.LabelFrame(parent, text="Basic Options", padding=8)
        basics.pack(fill=tk.X, pady=8)
        self._combo_row(basics, "Extractor", self.extractor, ["Native", "QuickBMS", "Auto"])
        self._combo_row(basics, "Texture Discovery", self.texture_discovery, ["headers", "extensions"])
        self._combo_row(basics, "Model Backend", self.model_backend, ["hskn", "extless", "both"])
        self._combo_row(basics, "Model Discovery", self.model_discovery, ["headers", "keywords"])
        self._combo_row(basics, "Audio Discovery", self.audio_discovery, ["headers", "extensions"])
        self._entry_row(basics, "Model Keywords (opt)", self.model_keywords)
        ttk.Checkbutton(basics, text="Skip Existing Outputs", variable=self.skip_existing).pack(anchor=tk.W, pady=3)

    def _build_advanced_tab(self, parent: ttk.Frame) -> None:
        extract = ttk.LabelFrame(parent, text="Extraction", padding=8)
        extract.pack(fill=tk.X)
        self._combo_row(extract, "Target Set", self.target_set, ["AllPcVariants", "Legacy"])
        self._path_row(extract, "QuickBMS EXE", self.quickbms_exe, is_dir=False)
        self._path_row(extract, "BMS Script", self.bms_script, is_dir=False)

        tex = ttk.LabelFrame(parent, text="Textures", padding=8)
        tex.pack(fill=tk.X, pady=8)
        self._path_row(tex, "Stage Root", self.texture_stage_root, is_dir=True)
        self._path_row(tex, "Texconv EXE", self.texconv_exe, is_dir=False)
        ttk.Checkbutton(tex, text="Use Texconv Fallback", variable=self.use_texconv_fallback).pack(anchor=tk.W, pady=3)
        ttk.Checkbutton(tex, text="Prefer Texconv", variable=self.prefer_texconv).pack(anchor=tk.W, pady=3)

        models = ttk.LabelFrame(parent, text="Models", padding=8)
        models.pack(fill=tk.X)
        self._entry_row(models, "Max Files", self.model_max_files)
        self._entry_row(models, "Min Triangles", self.model_min_triangles)
        self._entry_row(models, "HSKN Topology", self.hskn_topology_mode)
        self._entry_row(models, "Extless Top Modes", self.extless_top_modes)
        self._entry_row(models, "Edge Prune", self.model_edge_prune)
        ttk.Checkbutton(models, text="Include Environment Assets", variable=self.include_environment).pack(anchor=tk.W, pady=3)

        audio = ttk.LabelFrame(parent, text="Audio", padding=8)
        audio.pack(fill=tk.X, pady=8)
        self._entry_row(audio, "Extensions", self.audio_extensions)

    def _build_package_tab(self, parent: ttk.Frame) -> None:
        info = ttk.LabelFrame(parent, text="Standalone Build", padding=8)
        info.pack(fill=tk.X)
        ttk.Label(
            info,
            text="Builds a standalone UI executable via PyInstaller. In Onefile mode, helper tools and runtime assets are embedded into the EXE.",
            wraplength=980,
        ).pack(anchor=tk.W, pady=(0, 6))

        self._path_row(info, "Dist Dir", self.dist_dir, is_dir=True)
        ttk.Checkbutton(info, text="Onefile", variable=self.onefile).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(info, text="Windowed", variable=self.windowed).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(info, text="Bundle tools/bin", variable=self.bundle_runtime_bin).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(info, text="Bundle tools/bms", variable=self.bundle_bms).pack(anchor=tk.W, pady=2)

        buttons = ttk.Frame(info)
        buttons.pack(fill=tk.X, pady=(8, 2))
        self.package_button = ttk.Button(buttons, text="Build Standalone EXE", command=self.build_standalone)
        self.package_button.pack(side=tk.LEFT)
        ttk.Button(buttons, text="Open Dist Dir", command=self.open_dist_dir).pack(side=tk.LEFT, padx=8)

    def _entry_row(self, parent: ttk.Widget, label: str, var: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=16).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _combo_row(self, parent: ttk.Widget, label: str, var: tk.StringVar, values: list[str]) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=16).pack(side=tk.LEFT)
        box = ttk.Combobox(row, textvariable=var, values=values, state="readonly")
        box.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _path_row(self, parent: ttk.Widget, label: str, var: tk.StringVar, is_dir: bool, save_file: bool = False) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=16).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse", width=8, command=lambda: self._browse(var, is_dir=is_dir, save_file=save_file)).pack(side=tk.LEFT, padx=6)

    def _browse(self, var: tk.StringVar, is_dir: bool, save_file: bool) -> None:
        current = var.get().strip()
        base = REPO_ROOT / current if current and not Path(current).is_absolute() else Path(current or REPO_ROOT)
        if is_dir:
            value = filedialog.askdirectory(initialdir=str(base if base.exists() else REPO_ROOT))
        elif save_file:
            value = filedialog.asksaveasfilename(initialdir=str(base.parent if base.parent.exists() else REPO_ROOT))
        else:
            value = filedialog.askopenfilename(initialdir=str(base.parent if base.parent.exists() else REPO_ROOT))
        if value:
            var.set(value)

    def clear_log(self) -> None:
        if self.log_box:
            self.log_box.delete("1.0", tk.END)

    def log(self, text: str) -> None:
        if not self.log_box:
            return
        self.log_box.insert(tk.END, text + "\n")
        self.log_box.see(tk.END)
        self.root.update_idletasks()

    def _int_arg(self, value: str, name: str) -> str:
        try:
            int(value)
            return value
        except Exception:
            raise ValueError(f"{name} must be an integer")

    def build_run_command(self) -> list[str]:
        tasks: list[str] = []
        if self.task_extract.get():
            tasks.append("extract")
        if self.task_textures.get():
            tasks.append("textures")
        if self.task_models.get():
            tasks.append("models")
        if self.task_audio.get():
            tasks.append("audio")
        if not tasks:
            raise ValueError("Select at least one task.")

        cmd = [
            *self._backend_prefix(),
            "run",
            "--tasks",
            ",".join(tasks),
            "--game-root",
            self.game_root.get().strip(),
            "--extract-root",
            self.extract_root.get().strip(),
            "--logs-root",
            self.logs_root.get().strip(),
            "--extractor",
            self.extractor.get().strip(),
            "--target-set",
            self.target_set.get().strip(),
            "--texture-output-root",
            self.texture_output_root.get().strip(),
            "--texture-discovery",
            self.texture_discovery.get().strip(),
            "--texture-stage-root",
            self.texture_stage_root.get().strip(),
            "--texture-discovery-report",
            self.texture_discovery_report.get().strip(),
            "--texture-report-path",
            self.texture_report_path.get().strip(),
            "--model-output-root",
            self.model_output_root.get().strip(),
            "--model-backend",
            self.model_backend.get().strip(),
            "--model-discovery",
            self.model_discovery.get().strip(),
            "--model-keywords",
            self.model_keywords.get().strip(),
            "--model-max-files",
            self._int_arg(self.model_max_files.get().strip(), "Max Files"),
            "--model-min-triangles",
            self._int_arg(self.model_min_triangles.get().strip(), "Min Triangles"),
            "--hskn-topology-mode",
            self.hskn_topology_mode.get().strip(),
            "--extless-top-modes",
            self._int_arg(self.extless_top_modes.get().strip(), "Extless Top Modes"),
            "--model-edge-prune-median-multiplier",
            self.model_edge_prune.get().strip(),
            "--audio-output-root",
            self.audio_output_root.get().strip(),
            "--audio-discovery",
            self.audio_discovery.get().strip(),
            "--audio-report-path",
            self.audio_report_path.get().strip(),
            "--audio-extensions",
            self.audio_extensions.get().strip(),
        ]

        if self.quickbms_exe.get().strip():
            cmd.extend(["--quickbms-exe", self.quickbms_exe.get().strip()])
        if self.bms_script.get().strip():
            cmd.extend(["--bms-script", self.bms_script.get().strip()])
        if self.texconv_exe.get().strip():
            cmd.extend(["--texconv-exe", self.texconv_exe.get().strip()])
        if self.use_texconv_fallback.get():
            cmd.append("--use-texconv-fallback")
        else:
            cmd.append("--no-use-texconv-fallback")
        if self.prefer_texconv.get():
            cmd.append("--prefer-texconv")
        if self.include_environment.get():
            cmd.append("--include-environment")
        if self.skip_existing.get():
            cmd.append("--skip-existing")
        else:
            cmd.append("--no-skip-existing")
        return cmd

    def start_run(self) -> None:
        try:
            cmd = self.build_run_command()
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        self._start_process(cmd)

    def runtime_check(self) -> None:
        try:
            cmd = [*self._backend_prefix(), "runtime"]
        except RuntimeError as exc:
            messagebox.showerror("Runtime error", str(exc))
            return
        self._start_process(cmd)

    def _backend_prefix(self) -> list[str]:
        if IS_FROZEN:
            cli = self._resolve_frozen_helper("bzg-extractor.exe")
            if cli.exists():
                return [str(cli)]
            raise RuntimeError("Missing bundled helper: bzg-extractor.exe")
        return [sys.executable, str(BACKEND)]

    def _resolve_frozen_helper(self, name: str) -> Path:
        local = Path(sys.executable).resolve().with_name(name)
        if local.exists():
            return local
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            embedded = Path(meipass) / name
            if embedded.exists():
                return embedded
            embedded_tools = Path(meipass) / "tools" / name
            if embedded_tools.exists():
                return embedded_tools
        return local

    def _resolve_ui_state_path(self) -> Path:
        if not IS_FROZEN:
            return UI_STATE_PATH
        appdata = os.environ.get("LOCALAPPDATA")
        if appdata:
            base = Path(appdata)
        else:
            base = Path.home() / "AppData" / "Local"
        return base / "BattlezoneGoldExtractor" / "bzg-extractor-ui-state.json"

    def build_standalone(self) -> None:
        if not BUILD_PS1.exists():
            messagebox.showerror("Missing script", f"Missing build script: {BUILD_PS1}")
            return
        cmd = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(BUILD_PS1),
            "-DistDir",
            self.dist_dir.get().strip(),
            f"-OneFile:{'$true' if self.onefile.get() else '$false'}",
            f"-Windowed:{'$true' if self.windowed.get() else '$false'}",
            f"-BundleRuntimeBin:{'$true' if self.bundle_runtime_bin.get() else '$false'}",
            f"-BundleBms:{'$true' if self.bundle_bms.get() else '$false'}",
        ]
        self._start_process(cmd)

    def open_dist_dir(self) -> None:
        dist = Path(self.dist_dir.get().strip())
        if not dist.is_absolute():
            dist = (REPO_ROOT / dist).resolve()
        dist.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(dist)])

    def _set_running(self, running: bool) -> None:
        if self.run_button:
            self.run_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        if self.package_button:
            self.package_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        if self.stop_button:
            self.stop_button.configure(state=tk.NORMAL if running else tk.DISABLED)
        if self.progress:
            if running:
                self.progress.start(12)
            else:
                self.progress.stop()

    def _start_process(self, cmd: list[str]) -> None:
        if self.proc is not None:
            messagebox.showwarning("Busy", "A process is already running.")
            return
        self.log("$ " + " ".join(cmd))
        self._set_running(True)
        t = threading.Thread(target=self._run_thread, args=(cmd,), daemon=True)
        t.start()

    def _run_thread(self, cmd: list[str]) -> None:
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True,
            )
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.root.after(0, self.log, line.rstrip())
            rc = self.proc.wait()
            self.root.after(0, self.log, f"[done] exit_code={rc}")
        except Exception as exc:  # pragma: no cover
            self.root.after(0, self.log, f"[error] {exc}")
        finally:
            self.proc = None
            self.root.after(0, self._set_running, False)

    def stop_run(self) -> None:
        if self.proc is None:
            return
        self.log("[action] stopping process...")
        try:
            self.proc.terminate()
        except Exception as exc:
            self.log(f"[warn] terminate failed: {exc}")

    def _state_dict(self) -> dict[str, object]:
        return {
            "task_extract": self.task_extract.get(),
            "task_textures": self.task_textures.get(),
            "task_models": self.task_models.get(),
            "task_audio": self.task_audio.get(),
            "game_root": self.game_root.get(),
            "extract_root": self.extract_root.get(),
            "logs_root": self.logs_root.get(),
            "extractor": self.extractor.get(),
            "skip_existing": self.skip_existing.get(),
            "texture_output_root": self.texture_output_root.get(),
            "texture_report_path": self.texture_report_path.get(),
            "texture_discovery": self.texture_discovery.get(),
            "texture_stage_root": self.texture_stage_root.get(),
            "texture_discovery_report": self.texture_discovery_report.get(),
            "texconv_exe": self.texconv_exe.get(),
            "use_texconv_fallback": self.use_texconv_fallback.get(),
            "prefer_texconv": self.prefer_texconv.get(),
            "model_output_root": self.model_output_root.get(),
            "model_backend": self.model_backend.get(),
            "model_discovery": self.model_discovery.get(),
            "model_keywords": self.model_keywords.get(),
            "model_max_files": self.model_max_files.get(),
            "model_min_triangles": self.model_min_triangles.get(),
            "hskn_topology_mode": self.hskn_topology_mode.get(),
            "extless_top_modes": self.extless_top_modes.get(),
            "model_edge_prune": self.model_edge_prune.get(),
            "include_environment": self.include_environment.get(),
            "quickbms_exe": self.quickbms_exe.get(),
            "bms_script": self.bms_script.get(),
            "target_set": self.target_set.get(),
            "audio_output_root": self.audio_output_root.get(),
            "audio_report_path": self.audio_report_path.get(),
            "audio_discovery": self.audio_discovery.get(),
            "audio_extensions": self.audio_extensions.get(),
            "dist_dir": self.dist_dir.get(),
            "onefile": self.onefile.get(),
            "windowed": self.windowed.get(),
            "bundle_runtime_bin": self.bundle_runtime_bin.get(),
            "bundle_bms": self.bundle_bms.get(),
        }

    def save_state(self) -> None:
        self.ui_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.ui_state_path.write_text(json.dumps(self._state_dict(), indent=2), encoding="utf-8")
        self.log(f"[state] saved to {self.ui_state_path}")

    def _load_state(self) -> None:
        if not self.ui_state_path.exists():
            return
        try:
            data = json.loads(self.ui_state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        for key, value in data.items():
            if not hasattr(self, key):
                continue
            var = getattr(self, key)
            try:
                var.set(value)
            except Exception:
                pass
        self.log(f"[state] loaded from {self.ui_state_path}")

    def on_close(self) -> None:
        self.save_state()
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
