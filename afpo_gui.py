"""Browser GUI for afpo.py: dataset setup, live training monitor, and model explorer.

Start with `python afpo.py --gui` (or choose mode 3), or run `python afpo_gui.py`.
The server listens on 127.0.0.1:8778 only.  Each training run is a child process
(`afpo_gui.py --run SPEC`) so the page stays responsive, model scoring can still
fork its worker pool safely, and Stop behaves exactly like the terminal's first
Ctrl-C (finish the generation, write the checkpoint).  Outputs land where the
CLI puts them: afpo_runs/<run>/ and best_model.py in the working directory.
"""
from __future__ import annotations

import argparse
import collections
import multiprocessing
import contextlib
import io
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

# afpo must load before numpy: it pins the BLAS pool to one thread, which
# only works before numpy starts.  Imported the other way round, every training
# run and its workers got a 16-thread pool and ran ~40x slower.
import afpo
from afpo_lib import editor_format
from afpo_lib.csv_editor import CsvEditor

import numpy as np
import pandas as pd

HTML = Path(__file__).with_name("afpo_gui.html")
GUI_RUNS = Path("afpo_gui_runs")
GUI_CONFIGS = Path("afpo_gui_configs")   # named Setup-tab configurations, one JSON file each
DEFAULT_PORT = 8778
MAX_IMAGE_BYTES = 64 << 20   # one uploaded image
SNAPSHOT_INTERVAL = 1.0      # seconds between live frontier snapshots
MAX_POPULATION_POINTS = 800
MAX_ARCHIVE_POINTS = 400
MAX_FIT_POINTS = 2500
MAX_GRID_POINTS = 150        # per axis of a 2D/3D grid
MAX_GENERATE_ROWS = 5_000_000
GENERATE_CHUNK = 50_000
# Options with their own place in the form (or not meaningful from the GUI).
FORM_HANDLED = {"resume", "migrate_checkpoint", "allow_unsafe_pickle", "gui", "port", "test_csv",
                "constraint_metadata", "sequence_group", "input_relations", "output_relations", "custom_op", "max_generations", "population", "seed", "workers", "adf_mode", "help"}


# ───────────────────────── helpers ─────────────────────────
def clean(obj):
    """NaN/inf -> None, numpy -> Python, tuples -> lists, recursively (strict JSON)."""
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [clean(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def list_dir(path=None):
    path = Path(path or os.getcwd()).expanduser().resolve()
    if path.is_file():
        path = path.parent
    if not path.is_dir():
        raise FileNotFoundError(path)
    dirs, files = [], []
    for entry in sorted(path.iterdir(), key=lambda p: p.name.lower()):
        if entry.name.startswith("."):
            continue
        try:
            if entry.is_dir():
                dirs.append(entry.name)
                continue
            suffix = entry.suffix.lower()
            kind = "csv" if suffix in (".csv", ".tsv", ".txt", ".dat") else "json" if suffix == ".json" else "file"
            files.append({"name": entry.name, "kind": kind, "size": entry.stat().st_size, "mtime": entry.stat().st_mtime})
        except OSError:
            continue
    return {"path": str(path), "parent": str(path.parent), "dirs": dirs, "files": files}


def save_upload(name, stream, length, folder="uploads", chunk=1 << 20):
    """Stream a browser-picked file into <cwd>/uploads without holding it in memory."""
    name = os.path.basename(str(name or "upload.csv")).strip() or "upload.csv"
    dest_dir = Path(os.getcwd()) / folder
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / name
    stem, suffix, n = dest.stem, dest.suffix, 1
    while dest.exists():
        if dest.stat().st_size == length:      # the same upload again: reuse it
            stream.read(length)
            return {"path": str(dest), "size": length, "reused": True}
        dest = dest_dir / f"{stem}-{n}{suffix}"
        n += 1
    tmp = dest.with_suffix(dest.suffix + ".part")
    left = length
    with tmp.open("wb") as f:
        while left > 0:
            data = stream.read(min(chunk, left))
            if not data:
                break
            f.write(data)
            left -= len(data)
    if left:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Upload interrupted")
    tmp.replace(dest)
    return {"path": str(dest), "size": length, "reused": False}


def resolve_path(path, must_exist=True):
    if not path or not str(path).strip():
        raise ValueError("A path is required")
    resolved = Path(str(path).strip().strip('"').strip("'")).expanduser()
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"No such file: {resolved}")
    return resolved.resolve()


def read_frame(path, delimiter, max_rows=0):
    return afpo.read_dataset(resolve_path(path), delimiter or ",", max_rows)


# Column statistics and type suggestions come from a uniform sample this
# size: inspecting a multi-gigabyte CSV must not load all of it.
INSPECT_ROWS = 100_000


def _histogram(values, bins=24):
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return {"lo": lo, "hi": hi, "counts": [int(len(values))]}
    counts, _ = np.histogram(values, bins=bins, range=(lo, hi))
    return {"lo": lo, "hi": hi, "counts": counts.tolist()}


def inspect_dataset(path, delimiter=","):
    """Columns with the CLI's default type suggestions, a histogram each, and a preview."""
    df = read_frame(path, delimiter, INSPECT_ROWS)
    if not len(df.columns):
        raise ValueError("The file has no columns")
    sample = df.attrs.get("afpo_row_sample")
    # CSV Editor files: reserved columns are locked to "ignore", image codes cannot be outputs.
    schema = editor_format.editor_schema_from_frame(df)
    input_only = editor_format.editor_input_only_columns(schema)
    booleans = editor_format.editor_kind_columns(schema, editor_format.EDITOR_BOOL)
    usable = [i for i, c in enumerate(df.columns) if df[c].dropna().nunique() > 1 and not editor_format.editor_reserved(c)]
    outputs = [i for i in usable if str(df.columns[i]) not in input_only]
    suggested_output = outputs[-1] if outputs else -1
    columns = []
    for i, col in enumerate(df.columns):
        values = df[col].dropna()
        unique = int(values.nunique())
        numeric = bool(pd.api.types.is_numeric_dtype(df[col]))
        class_like = bool(numeric and unique <= 10 and len(values) and
                          np.all(np.isclose(values.to_numpy(float), np.round(values.to_numpy(float)))))
        reserved = editor_format.editor_reserved(col)
        if unique <= 1 or reserved:
            suggested = 0
        elif i == suggested_output:
            suggested = 5 if numeric and str(col) not in booleans else 6    # an editor Bool output is a true/false class
        else:
            suggested = 1 if numeric else 2
        info = {"index": i, "name": str(col), "numeric": numeric, "unique": unique, "missing": int(df[col].isna().sum()),
                "boolean": str(col) in booleans,
                "constant": unique <= 1 or reserved, "reserved": reserved, "input_only": str(col) in input_only,
                "class_like": class_like, "suggested": suggested,
                "examples": [str(v) for v in values.iloc[:3].tolist()]}
        if numeric:
            z = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            info["hist"] = _histogram(z)
            finite = z[np.isfinite(z)]
            if len(finite):
                info["min"], info["max"], info["mean"] = float(finite.min()), float(finite.max()), float(finite.mean())
        else:
            top = values.astype(str).value_counts().head(8)
            info["top"] = [[str(k), int(v)] for k, v in top.items()]
        columns.append(info)
    preview = df.head(12).astype(object).where(df.head(12).notna(), None).values.tolist()
    return {"path": str(resolve_path(path)), "rows": int(sample["source_rows"] if sample else len(df)),
            "sampled_rows": int(len(df)) if sample else None, "columns": columns,
            "editor": bool(schema),
            "preview": [[None if v is None else str(v) for v in row] for row in preview]}


def options():
    """Everything the setup form needs, taken from afpo itself so the two cannot drift."""
    parser = afpo.build_arg_parser()
    advanced = []
    for action in parser._actions:
        if action.dest in FORM_HANDLED or not action.option_strings:
            continue
        kind = ("bool" if isinstance(action, argparse._StoreTrueAction) else
                "choice" if action.choices else
                "int" if action.type is int else "float" if action.type is float else "text")
        advanced.append({"dest": action.dest, "flag": action.option_strings[0], "kind": kind, "default": action.default,
                         "choices": list(action.choices) if action.choices else None,
                         "help": (action.help or "").replace("%%", "%")})
    defaults = {a.dest: a.default for a in parser._actions if a.option_strings}
    groups = [{"id": gid, "name": name, "ops": list(ops), "default": gid in afpo.DEFAULT_GROUP_IDS}
              for gid, (name, ops) in afpo.OPERATOR_GROUPS.items()]
    roles = [{"id": role, "name": label.split(":", 1)[0], "description": label.split(":", 1)[1].strip()}
             for role, label in afpo.ISLAND_ROLES.items()]
    return {"groups": groups, "roles": roles, "advanced": advanced, "defaults": defaults, "profiles": list(afpo.PROFILES),
            "adf_modes": ["off", "flat", "nested"], "cwd": os.getcwd(),
            "op_cost": {op: afpo.OP_COMPLEXITY_BONUS.get(op, 2) for op in afpo.OPS}}


# ───────────────────────── run specification ─────────────────────────
def _flag(value):
    return value is not None and str(value).strip() != ""


def build_argv(form):
    """Translate the form's run options into afpo command-line arguments."""
    run = form.get("run") or {}
    argv = []
    for dest, flag in (("max_generations", "--max-generations"), ("population", "--population"),
                       ("seed", "--seed"), ("workers", "--workers")):
        if _flag(run.get(dest)):
            argv += [flag, str(run[dest]).strip()]
    argv += ["--adf-mode", str(run.get("adf_mode") or "nested")]
    parser = afpo.build_arg_parser()
    by_dest = {a.dest: a for a in parser._actions if a.option_strings}
    for dest, value in (form.get("advanced") or {}).items():
        action = by_dest.get(dest)
        if action is None or dest in FORM_HANDLED:
            continue
        if isinstance(action, argparse._StoreTrueAction):
            if value in (True, "true", "1", 1):
                argv.append(action.option_strings[0])
        elif _flag(value) and str(value) != str(action.default):
            argv += [action.option_strings[0], str(value)]
    data = form.get("data") or {}
    if _flag(data.get("test_path")):
        argv += ["--test-csv", str(resolve_path(data["test_path"]))]
    if _flag(data.get("metadata_path")):
        argv += ["--constraint-metadata", str(resolve_path(data["metadata_path"]))]
    for line in str(data.get("sequence_groups") or "").splitlines():
        if line.strip():
            argv += ["--sequence-group", line.strip()]
    for field, flag in (("input_relations", "--input-relations"), ("output_relations", "--output-relations"), ("custom_ops", "--custom-op")):
        for line in str(data.get(field) or "").splitlines():
            if line.strip():
                argv += [flag, line.strip()]
    return argv


def parse_argv(argv):
    """afpo.parse_cli, turning argparse's exit-on-error into a readable ValueError."""
    captured = io.StringIO()
    try:
        with contextlib.redirect_stderr(captured):
            return afpo.parse_cli(argv)[1]
    except SystemExit:
        message = captured.getvalue().strip().splitlines()
        raise ValueError(message[-1].split("error:", 1)[-1].strip() if message else "invalid options")


def selected_operators(form):
    ops = form.get("operators") or {}
    groups = [str(g) for g in ops.get("groups") or []]
    if not groups:
        raise ValueError("Select at least one operator group")
    excluded = set(ops.get("excluded") or [])
    chosen = [op for op in afpo.resolve_operator_groups(groups) if op not in excluded]
    if not chosen:
        raise ValueError("Every operator in the selected groups is switched off")
    return chosen


def setup_answers(form):
    """The JSON-safe answers to the CLI's setup prompts (the runner re-reads the CSV)."""
    data, search = form.get("data") or {}, form.get("search") or {}
    path = resolve_path(data.get("path"))
    types = [int(t) for t in data.get("types") or []]
    if not any(t in (5, 6) for t in types) or not any(t in (1, 2) for t in types):
        raise ValueError("Select at least one input and one output column")
    mode = data.get("validation_mode") or "percent"
    val_path = "0" if mode == "none" else str(resolve_path(data.get("validation_path"))) if mode == "file" else ""
    metadata = None
    if _flag(data.get("metadata_text")) and not _flag(data.get("metadata_path")):
        try:
            metadata = json.loads(data["metadata_text"])
        except ValueError as exc:
            raise ValueError(f"Constraint metadata is not valid JSON: {exc}")
    islands = max(1, int(search.get("islands") or 1))
    ops = selected_operators(form)
    roles_on = bool(search.get("roles")) and islands > 1
    # One dropdown per island after the first; island 1 stays the generalist.
    assignments = afpo.validate_island_roles(search.get("island_roles") or [], islands, ops) if roles_on else []
    stage_mode = search.get("stage_mode") or "off"
    stages = afpo.stage_config(stage_mode, count=int(search.get("stages") or 3), interval=int(search.get("stage_interval") or 5),
                               age_gap=int(search.get("stage_age_gap") or 10), schedule=search.get("stage_schedule") or "polynomial",
                               threshold_quantile=float(search.get("stage_quantile") or .5))
    return {"path": str(path), "delimiter": data.get("delimiter") or ",", "types": types, "ops": ops,
            "affine_on": bool(search.get("affine", True)), "coev": bool(search.get("coev", False)),
            "dynamic_pressure_on": bool(search.get("dynamic_pressure", True)), "adf_enabled": bool(search.get("adf", False)),
            "nodes": max(3, int(search.get("nodes") or 31)), "depth": max(1, int(search.get("depth") or 6)),
            "island_count": islands,
            "migration_interval": max(1, int(search.get("migration_interval") or 25)) if islands > 1 else 0,
            "migrants_per_island": max(1, int(search.get("migrants") or 2)) if islands > 1 else 0,
            "stages": stages,
            "roles": afpo.role_config(roles_on, interval=int(search.get("role_interval") or 10), assignments=assignments),
            "val_path": val_path, "validation_percent": float(data.get("validation_percent") or 0) if mode == "percent" else None,
            "metadata": metadata}


def check_form(form):
    """Validate a start request without launching anything."""
    mode = form.get("mode") or "train"
    argv = build_argv(form)
    if mode == "resume":
        argv += ["--resume", str(resolve_path(form.get("checkpoint")))]
        parse_argv(argv)
        return {"ok": True, "argv": argv}
    args = parse_argv(argv)
    setup = setup_answers(form)
    if setup["island_count"] * setup["stages"]["count"] > args.population // 8:
        raise ValueError("Islands x stages need at least eight models each; raise the population or choose fewer islands/stages")
    rows, columns = afpo.csv_shape(resolve_path(setup["path"]), setup["delimiter"])
    if len(setup["types"]) != len(columns):
        raise ValueError(f"The column types list has {len(setup['types'])} entries but the file has {len(columns)} columns; inspect the dataset again")
    # Image codes of a CSV Editor file are inputs only (afpo.py enforces it too; this says so before a run starts).
    images = {name for header in columns for column in editor_format.editor_parse_header(header) or [] if column["kind"] == editor_format.EDITOR_IMAGE
              for name in editor_format.editor_image_columns(column["name"], column.get("grid"))}
    outputs = [str(column) for column, kind in zip(columns, setup["types"]) if kind in (5, 6) and str(column) in images]
    if outputs:
        raise ValueError(f"Image columns can only be inputs: {', '.join(outputs)}")
    return {"ok": True, "argv": argv, "operators": len(setup["ops"]), "rows": int(rows),
            "training_rows": int(min(rows, args.max_rows)) if args.max_rows else int(rows)}


# ───────────────────────── training child process ─────────────────────────
class Telemetry:
    """Turns afpo.PROGRESS_HOOK calls into JSON lines the GUI server tails."""

    def __init__(self, stream, island_count, stage_count=1, roles=None):
        self.stream, self.island_count = stream, max(1, int(island_count))
        self.stage_count = max(1, int(stage_count))
        # Role per island (island 1 the generalist) when island roles are on.
        self.roles = ["generalist", *roles] if roles else None
        self.islands = {}          # cell label (or id(archive)) -> index
        self.latest = {}           # index -> latest hook kwargs
        self.last_snapshot = 0.
        self.validation_cache = {}
        self.announced = False
        self.announced_outputs = None

    def emit(self, kind, **data):
        self.stream.write(json.dumps(clean({"kind": kind, "time": time.time(), **data}), allow_nan=False) + "\n")
        self.stream.flush()

    def _validation(self, model, Xv, Yv, cats, constraints, out_names):
        if model is None or Xv is None:
            return None
        key = (repr(model.trees), repr(model.scales))
        if key not in self.validation_cache:
            if len(self.validation_cache) > 2000:
                self.validation_cache.clear()
            try:
                self.validation_cache[key] = afpo.frozen_metrics(model, Xv, Yv, cats, constraints, out_names)["loss"]
            except (ArithmeticError, IndexError, RecursionError, ValueError):
                self.validation_cache[key] = None
        return self.validation_cache[key]

    def cell_name(self, index):
        """Cells arrive island-major (island * stages + stage), as afpo loops over them."""
        island, stage = divmod(index, self.stage_count)
        parts = [f"island {island + 1}"] if self.island_count > 1 else []
        if self.stage_count > 1:
            parts.append(f"stage {stage + 1}")
        if self.roles and island < len(self.roles):
            parts.append(f"({self.roles[island]})")
        return " ".join(parts) or f"island {index + 1}"

    def hook(self, **kw):
        # Cells evolved in parallel processes arrive as fresh objects each
        # generation, so key on the stable (island, stage) label when given.
        names, out_names, cats = kw["names"], kw["out_names"], kw["cats"]
        # Separate-output runs search one output after another: each search
        # announces itself afresh, and the live view follows the current one.
        if self.announced and list(out_names) != self.announced_outputs:
            self.announced = False; self.islands = {}; self.latest = {}; self.validation_cache = {}
        island = self.islands.setdefault(kw.get("cell") or id(kw["archive"]), len(self.islands))
        self.latest[island] = kw
        if not self.announced:
            self.announced_outputs = list(out_names)
            self.announced = True
            self.emit("config", names=names, outputs=out_names, cats=cats, islands=self.island_count, stages=self.stage_count,
                      train_rows=len(kw["Xt"]), validation_rows=0 if kw["Xv"] is None else len(kw["Xv"]))
        best = kw["best_so_far"]
        population = [m for m in kw["population"] if m.feasible]
        losses = [afpo.aggregate_loss(m) for m in population]
        finite = [v for v in losses if math.isfinite(v)]
        self.emit("gen", generation=kw["generation"], island=island, elapsed=time.time() - kw["started"],
                  best_loss=None if best is None else afpo.aggregate_loss(best),
                  best_bits=None if best is None else afpo.model_complexity(best),
                  best_validation=self._validation(best, kw["Xv"], kw["Yv"], cats, kw["constraints"], out_names),
                  population_median=float(np.median(finite)) if finite else None,
                  feasible=len(population) / max(1, len(kw["population"])), archive=len(kw["archive"].items))
        now = time.time()
        if kw["generation"] == 0 or now - self.last_snapshot >= SNAPSHOT_INTERVAL:
            self.last_snapshot = now
            self.snapshot(kw["generation"])

    def _point(self, model, names, out_names, cats, equation=False):
        loss, bits = afpo.aggregate_loss(model), afpo.model_complexity(model)
        if not (model.feasible and math.isfinite(loss) and math.isfinite(bits)):
            return None
        point = {"loss": loss, "bits": bits, "age": int(model.age), "origin": model.origin or "seed",
                 "nodes": int(sum(afpo.node_size(t) for t in model.trees))}
        if equation:
            point["equation"] = afpo.equations(model, names, out_names, cats)
            point["path"] = afpo.history_path(model.history)
        return point

    def snapshot(self, generation):
        any_kw = next(iter(self.latest.values()))
        names, out_names, cats = any_kw["names"], any_kw["out_names"], any_kw["cats"]
        archive, population, origins, stats = [], [], collections.Counter(), []
        best = None
        for index in sorted(self.latest):
            kw = self.latest[index]
            for model in kw["archive"].items[:MAX_ARCHIVE_POINTS]:
                point = self._point(model, names, out_names, cats, equation=True)
                if point:
                    point["island"] = index
                    if self.stage_count > 1 or self.roles:
                        point["cell"] = self.cell_name(index)
                    archive.append(point)
            for model in kw["population"]:
                origins[model.origin or "seed"] += 1
                point = self._point(model, names, out_names, cats)
                if point and len(population) < MAX_POPULATION_POINTS:
                    point["island"] = index
                    if self.stage_count > 1 or self.roles:
                        point["cell"] = self.cell_name(index)
                    population.append(point)
            candidate = kw["best_so_far"]
            if candidate is not None and (best is None or afpo.secondary_key(candidate) < afpo.secondary_key(best[0])):
                best = (candidate, kw)
            prefix = f"{self.cell_name(index)}: " if len(self.latest) > 1 else ""
            for item in (kw["archive"].stats(), kw["pressure"].stats(), kw["semantic_qd"].stats(), kw["structural_qd"].stats(),
                         kw["qd_controller"].stats(), kw["library"].stats() if kw["library"] is not None else None):
                if item:
                    stats.append(prefix + item)
            if kw["evaluator"] is not None:
                stats.append(prefix + "evaluator: " + ", ".join(f"{k}={v}" for k, v in kw["evaluator"].diagnostics().items()))
        best_info = None
        if best is not None:
            model, kw = best
            best_info = self._point(model, names, out_names, cats, equation=True)
            if best_info:
                best_info["validation"] = self._validation(model, kw["Xv"], kw["Yv"], cats, kw["constraints"], out_names)
                best_info["losses"] = list(afpo.model_losses(model))
        self.emit("snapshot", generation=generation, archive=archive, population=population,
                  origins=dict(origins.most_common()), best=best_info, stats=stats)

    def choose(self, labels, choices, evaluation):
        """Offer the CLI's save choices to the browser and wait for its pick on stdin."""
        source, entries = evaluation
        any_kw = next(iter(self.latest.values()), None)
        names = any_kw["names"] if any_kw else None
        out_names = any_kw["out_names"] if any_kw else None
        cats = any_kw["cats"] if any_kw else None
        items = []
        for label, model in zip(labels, choices):
            metrics = next((e[2] for e in entries if e[0] is model), {})
            items.append({"label": label, "metrics": metrics, "train_loss": afpo.aggregate_loss(model),
                          "equation": afpo.equations(model, names, out_names, cats) if names else repr(model.trees)})
        self.emit("choose", source=source, options=items, outputs=list(out_names or []))
        line = sys.stdin.readline()
        try:
            index = int(line.strip() or 0)
        except ValueError:
            index = 0
        index = max(0, min(len(choices) - 1, index))
        self.emit("chosen", index=index, label=labels[index])
        return index


def run_spec(spec_path):
    """Entry point of the training child process."""
    spec = json.loads(Path(spec_path).read_text())
    with open(spec["events"], "a", encoding="utf-8") as stream:
        setup_spec = spec.get("setup") or {}
        roles = setup_spec.get("roles") or {}
        telemetry = Telemetry(stream, setup_spec.get("island_count", 1), (setup_spec.get("stages") or {}).get("count", 1),
                              (roles.get("assignments") or ["auto"] * (setup_spec.get("island_count", 1) - 1)) if roles.get("enabled") else None)
        telemetry.emit("started", pid=os.getpid(), mode=spec["mode"], argv=spec["argv"])
        try:
            args = afpo.parse_cli(spec["argv"])[1]
            afpo.PROGRESS_HOOK = telemetry.hook
            if spec["mode"] == "resume":
                afpo.resume_main(args)
                telemetry.emit("done", checkpoint=str(Path(args.resume).resolve()), best_model=str(Path("best_model.py").resolve()))
                return 0
            answers = spec["setup"]
            df = afpo.read_dataset(answers["path"], answers["delimiter"], args.max_rows, afpo.row_sample_seed(args))
            afpo.report_loaded(df, args.max_rows)
            if len(answers["types"]) != len(df.columns):
                raise ValueError("Column types do not match the dataset's columns")
            # Only the setup holds the frame, so training can free it once encoded.
            setup = {**answers, "path": Path(answers["path"]), "df": df}
            del df
            print(f"Operators ({len(setup['ops'])}): {', '.join(setup['ops'])}")
            print("Structural objective = MDL model-description bits (uniform enabled grammar; exact constants and affine coefficients included).")
            result = afpo.train_from_setup(args, setup, choose_model=telemetry.choose)
            telemetry.emit("done", **result, best_model=str(Path("best_model.py").resolve()))
            return 0
        except KeyboardInterrupt:
            telemetry.emit("error", message="Interrupted before the run could finish")
            return 130
        except BaseException as exc:          # the GUI must learn about every failure
            traceback.print_exc()
            telemetry.emit("error", message=f"{type(exc).__name__}: {exc}")
            return 1


class TrainingSession:
    """One training child process at a time, plus the tail of its console and events."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.run_dir = None
        self.reset()

    def reset(self):
        self.events_offset = self.console_offset = 0
        self.history = []
        self.snapshot = None
        self.snapshot_seq = 0
        self.config = None
        self.choose = None
        self.done = None
        self.error = None
        self.console = collections.deque(maxlen=4000)
        self.console_count = 0
        self.stop_requests = 0
        self.started_at = None
        self.mode = None
        self.chosen = None

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, form):
        with self.lock:
            if self.running():
                raise RuntimeError("A run is already in progress; stop it first")
            info = check_form(form)
            if self.proc is not None and self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
            mode = form.get("mode") or "train"
            argv = info["argv"]
            stamp = time.strftime("%Y%m%d-%H%M%S")
            run_dir = GUI_RUNS / stamp
            suffix = 1
            while run_dir.exists():
                suffix += 1
                run_dir = GUI_RUNS / f"{stamp}-{suffix}"
            run_dir.mkdir(parents=True)
            spec = {"mode": mode, "argv": argv, "events": str((run_dir / "events.jsonl").resolve()),
                    "setup": setup_answers(form) if mode == "train" else None, "form": form}
            (run_dir / "spec.json").write_text(json.dumps(clean(spec), indent=2))
            (run_dir / "events.jsonl").touch()
            self.reset()
            self.run_dir, self.mode, self.started_at = run_dir, mode, time.time()
            log = open(run_dir / "console.log", "wb")
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            # A new session keeps a Ctrl-C in the GUI's terminal from also hitting
            # the run; Stop sends the run its own SIGINT instead.
            self.proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run", str(run_dir / "spec.json")],
                                         stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, cwd=os.getcwd(),
                                         env=env, start_new_session=True)
            log.close()
            return {"started": True, "run_dir": str(run_dir.resolve()), "argv": argv}

    def stop(self):
        with self.lock:
            if not self.running():
                return {"stopping": False}
            if self.choose is not None and self.chosen is None:
                raise RuntimeError("The search has finished; pick a model to save instead")
            self.stop_requests += 1
            os.kill(self.proc.pid, signal.SIGINT)
            return {"stopping": True, "requests": self.stop_requests}

    def pick(self, index):
        with self.lock:
            if not self.running() or self.choose is None:
                raise RuntimeError("No run is waiting for a model choice")
            if self.chosen is not None:
                raise RuntimeError("A model was already chosen")
            self.chosen = int(index)
            self.proc.stdin.write(f"{int(index)}\n".encode())
            self.proc.stdin.flush()
            return {"chosen": int(index)}

    def _read_new(self):
        if self.run_dir is None:
            return
        events = self.run_dir / "events.jsonl"
        with events.open("rb") as handle:
            handle.seek(self.events_offset)
            data = handle.read()
        complete = data[:data.rfind(b"\n") + 1]
        self.events_offset += len(complete)
        for line in complete.decode("utf-8", "replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get("kind")
            if kind == "gen":
                self.history.append({k: event.get(k) for k in ("generation", "island", "elapsed", "best_loss", "best_bits",
                                                              "best_validation", "population_median", "feasible", "archive")})
            elif kind == "snapshot":
                self.snapshot = event
                self.snapshot_seq += 1
            elif kind == "config":
                if self.config is not None:          # the next output's search of a separate-output run
                    self.history = []; self.snapshot = None
                    # The previous output's choice is settled; this search runs (and can be stopped) afresh.
                    self.choose = None; self.chosen = None; self.stop_requests = 0
                self.config = event
            elif kind == "choose":
                # A separate-output run asks once per output: every prompt waits for its own pick.
                self.choose = event; self.chosen = None
            elif kind == "done":
                self.done = event
            elif kind == "error":
                self.error = event.get("message")
        log = self.run_dir / "console.log"
        if log.exists():
            with log.open("rb") as handle:
                handle.seek(self.console_offset)
                data = handle.read()
            complete = data[:data.rfind(b"\n") + 1]
            self.console_offset += len(complete)
            for line in complete.decode("utf-8", "replace").splitlines():
                if line.strip():
                    self.console.append(line)
                    self.console_count += 1

    def status(self, since=0, console_since=0, snapshot_seq=0):
        with self.lock:
            self._read_new()
            code = None if self.proc is None else self.proc.poll()
            if code is not None and self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
            if self.proc is None:
                state = "idle"
            elif code is None:
                state = ("choosing" if self.choose is not None and self.chosen is None else
                         "saving" if self.chosen is not None else
                         "stopping" if self.stop_requests else "running")
            else:
                state = "finished" if code == 0 and self.done is not None else "failed"
            since = max(0, int(since or 0))
            console_since = max(0, int(console_since or 0))
            first = self.console_count - len(self.console)
            return {"state": state, "returncode": code, "mode": self.mode,
                    "run_dir": None if self.run_dir is None else str(self.run_dir.resolve()),
                    "started_at": self.started_at, "history_total": len(self.history), "history": self.history[since:],
                    "snapshot_seq": self.snapshot_seq,
                    "snapshot": self.snapshot if int(snapshot_seq or 0) != self.snapshot_seq else None,
                    "config": self.config, "choose": self.choose if self.chosen is None else None, "chosen": self.chosen,
                    "done": self.done, "error": self.error, "stop_requests": self.stop_requests,
                    "console_total": self.console_count, "console": list(self.console)[max(0, console_since - first):]}

    def shutdown(self):
        if self.running():
            print("A training run is still active: asking it to stop after this generation and save its checkpoint.", flush=True)
            try:
                if self.choose is None:
                    os.kill(self.proc.pid, signal.SIGINT)
                self.proc.stdin.close()     # an unanswered model choice falls back to Best Score
            except OSError:
                pass


# ───────────────────────── model explorer ─────────────────────────
SYMBOLIC_TIMEOUT = 120.      # seconds a rendered-equation conversion may take


def _latex_payload(model, names, out_names, cats, positive, Xt):
    """The rendered-equation view of a model, as plain JSON-ready data (runs in a child process)."""
    result = afpo.symbolic_model(model, names, out_names, cats, positive, Xt)
    if result is None:
        return {"available": False, "reason": "Install sympy to see the rendered equation."}
    outputs = []
    for name, entry in result.items():
        forms = {mode: {"latex": None, "error": entry.get("errors", {}).get(mode)} if entry[mode][0] is None else
                 {"latex": entry[mode][1], "text": str(entry[mode][0]),
                  "mathml": afpo.mathml_expression(entry[mode][0], entry[mode][0].free_symbols)} for mode in ("exact", "raw")}
        agreement = entry["agreement"]
        outputs.append({"name": name, "name_latex": entry.get("name_latex") or afpo.latex_symbol_name(name.replace(" ", "_")), **forms,
                        "agreement": None if agreement is None else
                        {"defined": agreement[0], "gap": None if not math.isfinite(agreement[1]) else agreement[1]},
                        "output": entry.get("output"), "decision": entry.get("decision")})
    return {"available": True, "outputs": outputs}


class IsolatedJob:
    """Run one CPU-heavy pure-Python call in a forked child, so it neither holds
    the GIL against the server's other threads nor outlives a newer request:
    starting a job (or cancel()) kills the one before it."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None

    def cancel(self):
        with self.lock:
            proc, self.proc = self.proc, None
        if proc is not None and proc.is_alive():
            proc.kill()

    def run(self, function, args, timeout):
        if "fork" not in multiprocessing.get_all_start_methods():
            return function(*args)              # no fork: run in this (request) thread
        context = multiprocessing.get_context("fork")
        reader, writer = context.Pipe(duplex=False)

        def target():
            try:
                writer.send(("ok", function(*args)))
            except BaseException as error:  # report, never hang the waiting request
                writer.send(("error", f"{type(error).__name__}: {error}"))

        self.cancel()
        proc = context.Process(target=target, daemon=True)
        with self.lock:
            proc.start(); self.proc = proc
        writer.close()
        try:
            if not reader.poll(timeout):
                proc.kill()
                raise TimeoutError(f"gave up after {timeout:g} s")
            status, value = reader.recv()
        except EOFError:
            raise InterruptedError("cancelled by a newer request") from None
        finally:
            reader.close(); proc.join(1)
            with self.lock:
                if self.proc is proc: self.proc = None
        if status == "error":
            raise RuntimeError(value)
        return value


class ModelExplorer:
    """Loads a checkpoint in the GUI process to browse, plot, predict with and export its models."""

    def __init__(self):
        self.lock = threading.Lock()
        self.state = None
        self.models = []
        self.path = None
        self.files = {}
        self.job = None
        self.symbolic = IsolatedJob()

    def _require(self):
        if self.state is None:
            raise RuntimeError("Load a checkpoint first")

    def load(self, path):
        path = resolve_path(path)
        with self.lock:
            generation, pop, bayes, archive, state = afpo.load_checkpoint(path, allow_unsafe_pickle=False)
            maps = state["maps"]
            afpo.SEQUENCE_LAYOUT = maps.get(afpo.SEQUENCE_LAYOUT_KEY)
            self.symbolic.cancel()
            # Settings the models were searched with: numeric limits and user operators (names, bodies).
            afpo.set_numeric_limits(state.get("clip", afpo.DEFAULT_CLIP), state.get("eps", afpo.DEFAULT_EPS))
            base = state.get("custom_feature_base")
            afpo.configure_custom_ops(state.get("custom_ops", ()), list(state["names"]) if base is None else list(state["names"])[:base],
                                      state.get("source_columns", ()), state.get("types", ()))
            tolerance = state.get("parsimony_quality_tolerance", 0.)
            simplifier_keys = set()
            if state.get("island_states"):
                islands = [afpo.island_from_snapshot(item, len(state["Xt"]), tolerance) for item in state["island_states"]]
                simplifier_keys = afpo.simplifier_identities(islands)
                pool = [m for island in islands for m in [*island.archive.items, island.best_models.model] if m is not None]
                populations = [m for island in islands for m in island.population]
            else:
                pool, populations = list(archive.items), list(pop)
                best = state.get("best_model", {}).get("model")
                if best:
                    pool.append(afpo.Model(**best))
            # The run's shortened and snapped final candidates (no island holds them), so its chosen model is offered here too.
            final, final_simplifier = afpo.final_candidates_from_state(state)
            pool += final
            simplifier_keys |= final_simplifier
            constraints = afpo.compile_constraints(state.get("profile", "general"), state.get("constraint_metadata", {}))
            cats, names, out_names = state["cats"], state["names"], state["out_names"]
            Xt, Yt, Xv, Yv = state["Xt"], state["Yt"], state.get("Xv"), state.get("Yv")
            loss_tolerance = state.get("selection_loss_tolerance", .01)
            candidates = afpo.unique_models([*pool, *populations])
            evaluation = afpo.selection_evaluation(candidates, Xv, Yv, cats, constraints, out_names)
            labels, choices, selection = afpo.model_options([e[0] for e in evaluation[1]], cats=cats, loss_tolerance=loss_tolerance, evaluation=evaluation,
                                                           simplifier_keys=simplifier_keys)
            recommended = {id(model): label for label, model in zip(labels, choices)}
            pool_ids = {afpo.selection_identity(m) for m in pool}
            frontier_ids = {id(e[0]) for e in afpo.selection_frontier(evaluation[1])}
            entries = [e for e in evaluation[1] if afpo.selection_identity(e[0]) in pool_ids or id(e[0]) in recommended or id(e[0]) in frontier_ids]
            entries.sort(key=lambda e: (e[2]["mdl_bits"], e[2]["loss"]))
            self.models = []
            for model, scored, metrics in entries:
                train = afpo.frozen_metrics(model, Xt, Yt, cats, constraints, out_names)
                self.models.append({"model": model, "metrics": metrics, "train": train,
                                    "label": recommended.get(id(model)), "frontier": id(model) in frontier_ids})
            self.state, self.path, self.constraints = state, path, constraints
            self.generation = generation
            saved = state.get("selection") or {}
            return self.summary(saved)

    def summary(self, saved=None):
        state = self.state
        out = []
        for index, item in enumerate(self.models):
            model = item["model"]
            out.append({"index": index, "label": item["label"], "frontier": item["frontier"],
                        "equation": afpo.equations(model, state["names"], state["out_names"], state["cats"]),
                        "bits": item["metrics"]["mdl_bits"], "loss": item["metrics"]["loss"], "shape": item["metrics"]["shape"],
                        "train_loss": item["train"]["loss"], "nodes": int(sum(afpo.node_size(t) for t in model.trees)),
                        "age": int(model.age), "origin": model.origin})
        inputs = []
        typical = self._typical_row()
        booleans = editor_format.editor_kind_columns(self._schema(), editor_format.EDITOR_BOOL)
        for column, kind in zip(state["source_columns"], state["types"]):
            if kind == 1:
                values = state["Xt"][:, state["names"].index(column)] if column in state["names"] else np.empty(0)
                inputs.append({"name": column, "kind": "numeric", "range": (state.get("input_ranges") or {}).get(column),
                               "typical": state["maps"].get("__afpo_numeric_fills__", {}).get(column),
                               "boolean": column in booleans,    # a CSV Editor Bool input: a True/False choice in the predict form
                               "integer": bool(len(values) and np.all(np.isclose(values, np.round(values))))})
            elif kind == 2:
                inputs.append({"name": column, "kind": "categorical", "classes": state["maps"].get(column, []),
                               "typical": typical.get(column)})
        return {"path": str(self.path), "generation": self.generation, "source": "validation" if state.get("Xv") is not None else "training",
                "outputs": state["out_names"], "cats": state["cats"], "inputs": inputs, "models": out, "editor": self._editor_inputs(inputs),
                "text_columns": self._text_columns(),
                "dataset": state.get("dataset_path"), "seed": state.get("run_seed"),
                "train_rows": len(state["Xt"]), "validation_rows": 0 if state.get("Xv") is None else len(state["Xv"]),
                "test_rows": 0 if state.get("Xtest") is None else len(state["Xtest"]),
                "saved_selection": (saved or {}).get("selected_choice"), "manifest": state.get("manifest")}

    def _schema(self):
        return (self.state or {}).get("maps", {}).get(editor_format.EDITOR_SCHEMA_KEY)

    def _text_columns(self):
        """How the page turns an encoded number back into text, per encoded column (inputs and outputs):
        {"codes": [[code, string], …] in code order, "lone": spacing of a lone string} for the two
        string-code kinds (see editor_decode_text), {"chars": True} for each character position."""
        columns = {}
        for column in (self._schema() or {}).get("columns", ()):
            if column["kind"] in editor_format.EDITOR_STRING_CODES and column.get("codes"):
                columns[column["name"]] = {"codes": sorted([float(code), str(text)] for text, code in column["codes"].items()),
                                           "lone": .5 if column["kind"] == editor_format.EDITOR_LABEL else editor_format.EDITOR_TEXT_STEP / 2}
            elif column["kind"] == editor_format.EDITOR_CHARS:
                columns.update({name: {"chars": True} for name in column.get("columns") or []})
        return columns

    def _editor_inputs(self, inputs):
        """Text and image columns among the inputs: one field each instead of their encoded numbers."""
        names = {item["name"] for item in inputs}
        groups = []
        for column in (self._schema() or {}).get("columns", ()):
            used = [name for name in column.get("columns") or [] if name in names]
            if used and column["kind"] in (editor_format.EDITOR_TEXT, editor_format.EDITOR_LABEL, editor_format.EDITOR_CHARS, editor_format.EDITOR_IMAGE):
                groups.append({"name": column["name"], "kind": column["kind"], "columns": used, "length": column.get("length"),
                               "grid": column.get("grid"), "strings": sorted(column.get("codes") or {})[:500]})
        return groups

    def _editor_row(self, row):
        """A prediction row with its text and image fields turned into the encoded numbers."""
        schema = self._schema()
        if not schema or not row:
            return row
        frame = editor_format.editor_prepare_frame(pd.DataFrame([row]).astype(object), schema)
        return {key: value for key, value in frame.iloc[0].to_dict().items() if not (isinstance(value, float) and math.isnan(value))}

    def _model(self, index):
        self._require()
        index = int(index)
        if not 0 <= index < len(self.models):
            raise ValueError("Unknown model index")
        return self.models[index]["model"]

    def detail(self, index):
        with self.lock:
            model, state = self._model(index), self.state
            names, out_names, cats = state["names"], state["out_names"], state["cats"]
            item = self.models[int(index)]
            validation = (afpo.frozen_metrics(model, state["Xv"], state["Yv"], cats, self.constraints, out_names)
                          if state.get("Xv") is not None else None)
            test = (afpo.frozen_metrics(model, state["Xtest"], state["Ytest"], cats, self.constraints, out_names)
                    if state.get("Xtest") is not None else None)
            per_output = [{"name": name, "train_loss": item["train"]["losses"][j],
                           "validation_loss": None if validation is None else validation["losses"][j],
                           "test_loss": None if test is None else test["losses"][j]}
                          for j, name in enumerate(out_names)]
            return {"index": int(index), "label": item["label"], "equations": afpo.equations(model, names, out_names, cats).split("; "),
                    "adfs": afpo.adf_display_definitions(model, names),
                    "constants": [list(afpo.constant_vector(tree)) for tree in model.trees],
                    "scales": [list(scale) for scale in model.scales], "metrics": item["metrics"], "train": item["train"],
                    "per_output": per_output, "svg": afpo.tree_map_svg(model, names, out_names, cats),
                    "nodes": int(sum(afpo.node_size(t) for t in model.trees)), "age": int(model.age), "origin": model.origin,
                    "features_used": [names[i] for i in afpo.used_feature_indices(model) if i < len(names)],
                    "inputs_used": self._inputs_used(model),
                    "history": [dict(record) for record in model.history], "history_text": afpo.describe_history(model.history)}

    def latex(self, index):
        """Exact and raw equation forms of a model for the rendered-equation view (needs sympy).

        SymPy simplification of a large equation can take minutes of pure
        Python: it runs in a child process (IsolatedJob) outside the explorer
        lock, so browsing data, plotting and sampling stay responsive, and a
        newer request (another model) cancels it."""
        with self.lock:
            model, state = self._model(index), self.state
            item = self.models[int(index)]
            if item.get("latex") is not None:
                return item["latex"]
            names, out_names, cats, Xt = state["names"], state["out_names"], state["cats"], state["Xt"]
            positive = tuple(i for i in range(Xt.shape[1]) if len(Xt) and np.all(Xt[:, i] > 0))
        try:
            payload = self.symbolic.run(_latex_payload, (model, names, out_names, cats, positive, Xt), SYMBOLIC_TIMEOUT)
        except InterruptedError:
            return {"available": False, "cancelled": True, "reason": "Cancelled: another model was selected."}
        except TimeoutError as error:
            return {"available": False, "reason": f"The equation is too large to render ({error}); the text form above is exact."}
        except Exception as error:  # an unsupported operator must not break the model view
            return {"available": False, "reason": f"Symbolic conversion failed: {error}"}
        with self.lock:
            if self.state is state and int(index) < len(self.models) and self.models[int(index)] is item:
                item["latex"] = payload
        return payload

    def _head_labels(self):
        state = self.state
        labels = []
        for name, classes in zip(state["out_names"], state["cats"]):
            labels.extend([f"{name}[{c!r}] score" for c in classes] if classes is not None and len(classes) > 2 else [name])
        return labels

    def edit_text(self, index):
        """Each equation head with its readout, in the editable syntax (afpo.parse_equation)."""
        with self.lock:
            model, state = self._model(index), self.state
            return {"heads": [{"label": label, "text": afpo.readout_text(tree, scale, state["names"])}
                              for label, tree, scale in zip(self._head_labels(), model.trees, model.scales)]}

    def edit(self, index, texts, refit=False):
        """A new candidate from edited equations (optionally with its constants re-fitted); returns its index."""
        with self.lock:
            original, state = self._model(index), self.state
            names, out_names, cats = state["names"], state["out_names"], state["cats"]
            if len(texts) != len(original.trees):
                raise ValueError(f"Expected {len(original.trees)} equation(s)")
            trees = []
            for label, text in zip(self._head_labels(), texts):
                try:
                    trees.append(afpo.parse_equation(text, names, original.adfs))
                except ValueError as error:
                    raise ValueError(f"{label}: {error}") from None
            used = {node[0] for tree in trees for node in afpo.walk_tree(tree) if node[0] not in ("x", "c", "arg")}
            model = afpo.Model(trees, [(1., 0.)] * len(trees), origin="edited", mdl_operators=tuple(dict.fromkeys([*original.mdl_operators, *sorted(used)])),
                               mdl_feature_count=original.mdl_feature_count or len(names), adfs=dict(original.adfs), history=original.history)
            Xt, Yt = state["Xt"], state["Yt"]
            if refit:
                afpo.tune_model_constants(model, Xt, Yt, state.get("affine_on", True), cats)
                afpo.assess(model, Xt, Yt, state.get("affine_on", True), cats, fit_affine=True, constraints=self.constraints, output_names=out_names)
            else:
                afpo.assess(model, Xt, Yt, False, cats, fit_affine=False, constraints=self.constraints, output_names=out_names)
            if not model.feasible:
                raise ValueError(f"The edited model cannot be scored ({model.invalid_reason})")
            entries = afpo.selection_evaluation([model], state.get("Xv"), state.get("Yv"), cats, self.constraints, out_names)[1]
            if not entries:
                raise ValueError("The edited model gives non-finite results")
            _, _, metrics = entries[0]
            train = afpo.frozen_metrics(model, Xt, Yt, cats, self.constraints, out_names)
            self.models.append({"model": model, "metrics": metrics, "train": train,
                                "label": f"Edited (from #{int(index) + 1})" + (", refitted" if refit else ""), "frontier": False})
            return {"index": len(self.models) - 1, "summary": self.summary(state.get("selection") or {})}

    def _inputs_used(self, model):
        """Source input columns the model reads (a text column counts if any of its categories is used)."""
        state = self.state
        used = [state["names"][i] for i in afpo.used_feature_indices(model) if i < len(state["names"])]
        return [column for column, kind in zip(state["source_columns"], state["types"]) if kind in (1, 2) and
                any(name == column or name.startswith(column + "=") or name.startswith(column + "[") for name in used)]

    def fit(self, index, split="train"):
        """Predicted vs. actual (regression) or a confusion matrix (classification)."""
        with self.lock:
            model, state = self._model(index), self.state
            X, Y, split = self._split(split)
            prediction = afpo.predict_targets(model, X, state["cats"])
            rows = np.arange(len(X))
            if len(rows) > MAX_FIT_POINTS:
                rows = np.random.default_rng(0).choice(rows, MAX_FIT_POINTS, replace=False)
            outputs = []
            integers = editor_format.editor_kind_columns(self._schema(), editor_format.EDITOR_INT)
            for j, (name, labels) in enumerate(zip(state["out_names"], state["cats"])):
                if labels is None:
                    residual = prediction[:, j] - Y[:, j]
                    outputs.append({"name": name, "kind": "regression", "actual": Y[rows, j], "predicted": prediction[rows, j],
                                    "residual_hist": _histogram(residual, 30),
                                    "rmse": float(np.sqrt(np.mean(residual ** 2))), "mae": float(np.mean(np.abs(residual))),
                                    "r2": float(1 - np.sum(residual ** 2) / max(np.sum((Y[:, j] - Y[:, j].mean()) ** 2), afpo.EPS))})
                    if name in integers:    # an editor Integer output is predicted rounded: how often that lands exactly
                        outputs[-1]["exact"] = float(np.mean(editor_format.editor_round(prediction[:, j]) == editor_format.editor_round(Y[:, j])))
                else:
                    k = len(labels)
                    matrix = np.zeros((k, k), dtype=int)
                    truth, guess = Y[:, j].astype(int), prediction[:, j].astype(int)
                    for t, g in zip(truth, guess):
                        if 0 <= t < k and 0 <= g < k:
                            matrix[t, g] += 1
                    outputs.append({"name": name, "kind": "classification", "labels": labels, "matrix": matrix,
                                    "accuracy": float(np.mean(truth == guess))})
            return {"split": split, "rows": len(X), "outputs": outputs}

    def _split(self, split):
        """(X, Y, name) of the training, validation or test rows; an unavailable split falls back to training."""
        state = self.state
        if split == "validation" and state.get("Xv") is not None:
            return state["Xv"], state["Yv"], "validation"
        if split == "test" and state.get("Xtest") is not None:
            return state["Xtest"], state["Ytest"], "test"
        return state["Xt"], state["Yt"], "train"

    def _encode(self, rows):
        state = self.state
        frame = pd.DataFrame([{column: row.get(column) for column in state["source_columns"]} for row in rows],
                             columns=state["source_columns"])
        for column, kind in zip(state["source_columns"], state["types"]):
            if kind == 1:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return afpo.append_custom_features(afpo.encode(frame, state["types"], state["maps"])[0])

    def _typical_row(self):
        state = self.state
        row = {}
        fills = state["maps"].get("__afpo_numeric_fills__", {})
        for column, kind in zip(state["source_columns"], state["types"]):
            if kind == 1:
                row[column] = fills.get(column, 0.)
            elif kind == 2:
                classes = state["maps"].get(column, [])
                columns = [state["names"].index(f"{column}={cl}") for cl in classes if f"{column}={cl}" in state["names"]]
                counts = state["Xt"][:, columns].sum(axis=0) if columns else []
                row[column] = classes[int(np.argmax(counts))] if len(counts) else (classes[0] if classes else None)
        return row

    def _decode(self, model, X):
        state = self.state
        values, distributions = afpo.predict_targets(model, X, state["cats"], probabilities=True)
        results = []
        for r in range(len(X)):
            row = {}
            for j, (name, labels) in enumerate(zip(state["out_names"], state["cats"])):
                if labels is None:
                    row[name] = float(values[r, j])
                else:
                    label_index = int(values[r, j])
                    row[name] = labels[label_index] if 0 <= label_index < len(labels) else None
                    if distributions[j] is not None and distributions[j].shape[1] == len(labels):
                        row[name + " probabilities"] = {str(labels[c]): float(distributions[j][r, c]) for c in range(len(labels))}
            results.append(row)
        return results

    def predict(self, index, row):
        with self.lock:
            model = self._model(index)
            given = {k: v for k, v in (row or {}).items() if v not in (None, "")}
            full = {**self._typical_row(), **self._editor_row(given)}
            outputs = self._decode(model, self._encode([full]))[0]
            outputs.update(editor_format.editor_decode_row(outputs, self._schema()))
            schema = self._schema()     # Bool and Integer inputs are shown as the model read them (1/0, rounded)
            coerced = editor_format.editor_kind_columns(schema, editor_format.EDITOR_BOOL) | editor_format.editor_kind_columns(schema, editor_format.EDITOR_INT)
            return {"inputs": {**full, **{key: value for key, value in given.items() if key not in coerced or key not in full}}, "outputs": outputs}

    def _axis(self, column, lo=None, hi=None, points=120):
        """Values one explored input takes: an even numeric range, or every category."""
        state = self.state
        if column not in state["source_columns"]:
            raise ValueError(f"Unknown input column {column!r}")
        kind = state["types"][state["source_columns"].index(column)]
        if kind == 1:
            low, high = (state.get("input_ranges") or {}).get(column, [0., 1.])
            pad = .05 * (high - low) if high > low else .5
            lo = low - pad if lo in (None, "") else float(lo)
            hi = high + pad if hi in (None, "") else float(hi)
            if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
                raise ValueError(f"The range for {column!r} needs a finite upper end above its lower end")
            return "numeric", np.linspace(lo, hi, max(2, int(points))).tolist()
        if kind == 2:
            return "categorical", list(state["maps"].get(column, []))
        raise ValueError(f"{column!r} is not a model input")

    def _base(self, base):
        """Typical training values, overridden by whatever the user froze the inputs at."""
        row = self._typical_row()
        for key, value in (base or {}).items():
            if key in row and value not in (None, ""):
                row[key] = value
        return row

    def _surface(self, model, rows):
        """Numeric outputs as values; class outputs as each class's probability."""
        state = self.state
        values, distributions = afpo.predict_targets(model, self._encode(rows), state["cats"], probabilities=True)
        outputs = {}
        for j, (name, labels) in enumerate(zip(state["out_names"], state["cats"])):
            if labels is None:
                outputs[name] = {"kind": "value", "values": values[:, j]}
                continue
            dist = distributions[j]
            outputs[name] = {"kind": "class", "labels": [str(label) for label in labels], "predicted": values[:, j].astype(int),
                             "prob": ({str(labels[c]): dist[:, c] for c in range(len(labels))}
                                      if dist is not None and dist.shape[1] == len(labels) else {})}
        return outputs

    def _point_sets(self, columns):
        """Validation and test rows (when the run had them) in _training_points' form, keyed by split."""
        return {split: self._training_points(columns, split) for split in ("validation", "test")
                if self.state.get("Xv" if split == "validation" else "Xtest") is not None}

    def _training_points(self, columns, split="train"):
        """A sample of training rows: the explored inputs (category index for text inputs) and the targets."""
        state = self.state
        X, Y, _ = self._split(split)
        rows = np.arange(len(X))
        if len(rows) > MAX_FIT_POINTS:
            rows = np.random.default_rng(1).choice(rows, MAX_FIT_POINTS, replace=False)
        coords = {}
        for column in columns:
            kind = state["types"][state["source_columns"].index(column)]
            if kind == 1 and column in state["names"]:
                coords[column] = X[rows, state["names"].index(column)]
            elif kind == 2:
                classes = state["maps"].get(column, [])
                onehot = [state["names"].index(f"{column}={cl}") for cl in classes if f"{column}={cl}" in state["names"]]
                if len(onehot) != len(classes):
                    return None
                coords[column] = np.argmax(X[np.ix_(rows, onehot)], axis=1)
            else:
                return None        # sequence-derived inputs have no single raw column
        return {"coords": coords, "targets": {name: Y[rows, j] for j, name in enumerate(state["out_names"])}}

    def sweep(self, index, column, base=None, lo=None, hi=None, points=160):
        """1D: one input varied, every other input frozen (typical values unless overridden)."""
        with self.lock:
            model = self._model(index)
            kind, xs = self._axis(column, lo, hi, min(int(points), 1000))
            base = self._base(base)
            return {"column": column, "kind": kind, "x": xs, "base": base,
                    "outputs": self._surface(model, [{**base, column: x} for x in xs]),
                    "data": self._training_points([column]), "data_sets": self._point_sets([column])}

    def grid(self, index, x, y, base=None, x_lo=None, x_hi=None, y_lo=None, y_hi=None, points=60):
        """2D/3D: two inputs over a grid, the rest frozen.  Values are row-major: z[yi][xi]."""
        with self.lock:
            if x == y:
                raise ValueError("Choose two different inputs for the axes")
            model = self._model(index)
            n = max(5, min(int(points), MAX_GRID_POINTS))
            x_kind, xs = self._axis(x, x_lo, x_hi, n)
            y_kind, ys = self._axis(y, y_lo, y_hi, n)
            base = self._base(base)
            outputs = self._surface(model, [{**base, x: xv, y: yv} for yv in ys for xv in xs])
            shape = (len(ys), len(xs))
            for item in outputs.values():
                if item["kind"] == "value":
                    item["values"] = np.asarray(item["values"], float).reshape(shape)
                else:
                    item["predicted"] = np.asarray(item["predicted"]).reshape(shape)
                    item["prob"] = {label: np.asarray(v, float).reshape(shape) for label, v in item["prob"].items()}
            return {"x": {"column": x, "kind": x_kind, "values": xs}, "y": {"column": y, "kind": y_kind, "values": ys},
                    "base": base, "outputs": outputs, "data": self._training_points([x, y]), "data_sets": self._point_sets([x, y])}

    def predict_csv(self, index, path, delimiter=","):
        with self.lock:
            model, state = self._model(index), self.state
            frame = editor_format.editor_prepare_frame(read_frame(path, delimiter), self._schema())
            missing = [c for c, k in zip(state["source_columns"], state["types"]) if k in (1, 2) and c not in frame.columns]
            if missing:
                raise ValueError(f"The CSV lacks input column(s): {', '.join(missing)}")
            rows = frame.to_dict("records")
            X = self._encode(rows)
            decoded = self._decode(model, X)
            result = frame.copy()
            for name in state["out_names"]:
                result[f"predicted_{name}"] = [row.get(name) for row in decoded]
            texts = [editor_format.editor_decode_row(row, self._schema()) for row in decoded]
            for name in dict.fromkeys(key for row in texts for key in row):
                result[f"predicted_{name}"] = [row.get(name) for row in texts]
            destination = Path(os.getcwd()) / f"afpo_predictions_{Path(path).stem}.csv"
            result.to_csv(destination, index=False)
            self.files[str(destination.resolve())] = True
            metrics = None
            if all(name in frame.columns for name in state["out_names"]):
                Xe, Ye, *_ = afpo.encode(frame[state["source_columns"]] if all(c in frame.columns for c in state["source_columns"]) else
                                         frame.reindex(columns=state["source_columns"]), state["types"], state["maps"])
                metrics = afpo.frozen_metrics(model, afpo.append_custom_features(Xe), Ye, state["cats"], self.constraints, state["out_names"])
            preview = result.head(25).astype(object).where(result.head(25).notna(), None)
            return {"path": str(destination.resolve()), "rows": int(len(result)), "columns": [str(c) for c in result.columns],
                    "preview": [[None if v is None else str(v) for v in row] for row in preview.values.tolist()], "metrics": metrics}

    def generate_defaults(self, index):
        """Per-output error of the selected model on its training rows (a guide for added noise)."""
        with self.lock:
            model, state = self._model(index), self.state
            prediction = afpo.predict_targets(model, state["Xt"], state["cats"])
            outputs = []
            for j, (name, labels) in enumerate(zip(state["out_names"], state["cats"])):
                if labels is None:
                    residual = prediction[:, j] - state["Yt"][:, j]
                    outputs.append({"name": name, "kind": "numeric", "rmse": float(np.sqrt(np.mean(residual ** 2))),
                                    "std": float(np.std(state["Yt"][:, j]))})
                else:
                    outputs.append({"name": name, "kind": "class", "labels": [str(label) for label in labels],
                                    "accuracy": float(np.mean(prediction[:, j].astype(int) == state["Yt"][:, j].astype(int)))})
            stem = Path(str(state.get("dataset_path") or "model")).stem
            return {"outputs": outputs, "default_name": f"{stem}_synthetic.csv"}

    def generate(self, index, spec):
        """Start writing a synthetic dataset from the selected model in the background."""
        with self.lock:
            if self.job is not None and self.job.running():
                raise RuntimeError("A dataset is already being generated")
            model, state = self._model(index), self.state
            name = str(spec.get("path") or "").strip() or "synthetic.csv"
            destination = Path(name).expanduser()
            if not destination.is_absolute():
                destination = Path(os.getcwd()) / destination
            if destination.suffix.lower() != ".csv":
                destination = destination.with_name(destination.name + ".csv")
            if not destination.parent.is_dir():
                raise FileNotFoundError(f"No such folder: {destination.parent}")
            if destination.exists() and not spec.get("overwrite"):
                return {"exists": True, "path": str(destination.resolve())}
            plan = generation_plan(state, spec)
            self.job = GenerateJob(state, model, plan, destination.resolve(), self.files)
            self.job.start()
            return {"started": True, "path": str(destination.resolve()), "rows": plan["rows"]}

    def generate_status(self):
        job = self.job
        return {"state": "idle"} if job is None else job.status()

    def export(self, index):
        with self.lock:
            model, state = self._model(index), self.state
            names = state["names"] if afpo.CUSTOM_FEATURE_BASE is None else state["names"][:afpo.CUSTOM_FEATURE_BASE]
            afpo.export_model(afpo.inline_custom_features(model, state["names"]), names, state["out_names"], state["cats"], state["maps"],
                              state["source_columns"], state["types"], state.get("export_fixture"), state.get("input_ranges"))
            written = [p for p in ("best_model.py", "model_tree.svg", "best_model_fixture.csv", "best_model_fixture_predictions.csv") if Path(p).exists()]
            return {"written": [str(Path(p).resolve()) for p in written],
                    "equation": afpo.equations(model, state["names"], state["out_names"], state["cats"])}


def _encode_frame(state, frame):
    """Encode raw input columns exactly as training did (outputs and ignored columns left empty)."""
    frame = frame.reindex(columns=state["source_columns"])
    for column, kind in zip(state["source_columns"], state["types"]):
        if kind == 1:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return afpo.append_custom_features(afpo.encode(frame, state["types"], state["maps"])[0])


def generation_plan(state, spec):
    """Validate a generation request into a concrete per-input plan."""
    sampling = spec.get("sampling") or "uniform"
    if sampling not in ("uniform", "lhs", "grid", "training"):
        raise ValueError(f"Unknown sampling method {sampling!r}")
    rows = int(spec.get("rows") or 0)
    if not 1 <= rows <= MAX_GENERATE_ROWS:
        raise ValueError(f"Rows must be between 1 and {MAX_GENERATE_ROWS:,}")
    ranges = state.get("input_ranges") or {}
    requested = spec.get("inputs") or {}
    inputs = []
    for column, kind in zip(state["source_columns"], state["types"]):
        if kind not in (1, 2):
            continue
        item = requested.get(column) or {}
        vary = bool(item.get("vary", True))
        if kind == 1:
            low, high = ranges.get(column, [0., 1.])
            lo = low if item.get("lo") in (None, "") else float(item["lo"])
            hi = high if item.get("hi") in (None, "") else float(item["hi"])
            if vary and (not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo):
                raise ValueError(f"{column}: the range needs a finite upper end at or above its lower end")
            value = item.get("value")
            if not vary:
                if value in (None, ""):
                    value = state["maps"].get("__afpo_numeric_fills__", {}).get(column, 0.)
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"{column}: the fixed value must be a finite number")
            inputs.append({"name": column, "kind": "numeric", "vary": vary, "lo": lo, "hi": hi,
                           "integer": bool(item.get("integer", False)), "value": value})
        else:
            classes = list(state["maps"].get(column, []))
            value = item.get("value")
            if not vary and value not in classes:
                raise ValueError(f"{column}: choose one of its categories as the fixed value")
            inputs.append({"name": column, "kind": "categorical", "vary": vary, "classes": classes, "value": value})
    outputs = []
    requested_outputs = spec.get("outputs") or {}
    for name, labels in zip(state["out_names"], state["cats"]):
        item = requested_outputs.get(name) or {}
        if labels is None:
            noise = float(item.get("noise") or 0.)
            if not math.isfinite(noise) or noise < 0:
                raise ValueError(f"{name}: noise must be a non-negative number")
            outputs.append({"name": name, "kind": "numeric", "noise": noise})
        else:
            outputs.append({"name": name, "kind": "class", "labels": list(labels), "sample": item.get("mode") == "sample",
                            "probabilities": bool(item.get("probabilities", False))})
    jitter = float(spec.get("jitter") or 0.) / 100.
    seed = spec.get("seed")
    seed = None if seed in (None, "") else int(seed)
    plan = {"sampling": sampling, "rows": rows, "inputs": inputs, "outputs": outputs, "jitter": jitter, "seed": seed}
    if sampling == "grid":
        varied_numeric = [i for i in inputs if i["vary"] and i["kind"] == "numeric"]
        categories = int(np.prod([len(i["classes"]) for i in inputs if i["vary"] and i["kind"] == "categorical"] or [1]))
        per_axis = max(2, int(math.floor((rows / categories) ** (1 / len(varied_numeric)) + 1e-9))) if varied_numeric else 1
        for item in varied_numeric:
            item["axis"] = np.linspace(item["lo"], item["hi"], per_axis)
            if item["integer"]:
                item["axis"] = np.unique(np.rint(item["axis"]))
        for item in inputs:
            if item["vary"] and item["kind"] == "categorical":
                item["axis"] = np.arange(len(item["classes"]))
        plan["axes"] = [i for i in inputs if "axis" in i]
        plan["rows"] = int(np.prod([len(i["axis"]) for i in plan["axes"]] or [1]))
        if plan["rows"] > MAX_GENERATE_ROWS:
            raise ValueError(f"That grid would have {plan['rows']:,} rows; reduce rows or vary fewer inputs")
    return plan


class GenerateJob:
    """Writes model predictions for sampled inputs to CSV in chunks, in a background thread."""

    def __init__(self, state, model, plan, destination, registry):
        self.state, self.model, self.plan, self.destination, self.registry = state, model, plan, destination, registry
        self.written = 0
        self.error = None
        self.done = False
        self.preview = None
        self.started = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def running(self):
        return self.thread.is_alive()

    def status(self):
        return {"state": "failed" if self.error else "finished" if self.done else "running", "written": self.written,
                "rows": self.plan["rows"], "path": str(self.destination), "error": self.error, "preview": self.preview,
                "elapsed": time.time() - self.started}

    def _inputs(self, rng, start, stop, lhs):
        """Raw input columns for rows [start, stop)."""
        plan, state, count = self.plan, self.state, stop - start
        columns = {}
        if plan["sampling"] == "training":
            picks = rng.integers(0, len(state["Xt"]), count)
        if plan["sampling"] == "grid":
            shape = [len(item["axis"]) for item in plan["axes"]]
            coords = np.unravel_index(np.arange(start, stop), shape) if shape else []
            by_name = {item["name"]: coord for item, coord in zip(plan["axes"], coords)}
        for item in plan["inputs"]:
            name = item["name"]
            if not item["vary"]:
                if item["kind"] == "numeric" and item["integer"] and float(item["value"]).is_integer():
                    columns[name] = np.full(count, int(item["value"]), dtype=np.int64)
                else:
                    columns[name] = np.full(count, item["value"], dtype=object if item["kind"] == "categorical" else float)
                continue
            if item["kind"] == "categorical":
                if plan["sampling"] == "grid":
                    index = by_name[name]
                elif plan["sampling"] == "training":
                    onehot = [state["names"].index(f"{name}={cl}") for cl in item["classes"]]
                    index = np.argmax(state["Xt"][np.ix_(picks, onehot)], axis=1)
                else:
                    index = rng.integers(0, len(item["classes"]), count)
                columns[name] = np.asarray(item["classes"], dtype=object)[index]
                continue
            lo, hi = item["lo"], item["hi"]
            if plan["sampling"] == "grid":
                values = item["axis"][by_name[name]]
            elif plan["sampling"] == "lhs":
                values = lo + lhs[name][start:stop] * (hi - lo)
            elif plan["sampling"] == "training":
                values = state["Xt"][picks, state["names"].index(name)].astype(float)
                if plan["jitter"]:
                    values = values + rng.normal(0, plan["jitter"] * (hi - lo), count)
            else:
                values = lo + rng.random(count) * (hi - lo)
            if item["integer"]:
                values = np.rint(values).astype(np.int64)      # written as 8, not 8.0
            columns[name] = values
        return columns

    def _outputs(self, rng, columns, count):
        state = self.state
        X = _encode_frame(state, pd.DataFrame(columns))
        values, distributions = afpo.predict_targets(self.model, X, state["cats"], probabilities=True)
        result = {}
        for j, item in enumerate(self.plan["outputs"]):
            if item["kind"] == "numeric":
                column = values[:, j].astype(float)
                if item["noise"]:
                    column = column + rng.normal(0, item["noise"], count)
                result[item["name"]] = column
                continue
            labels = np.asarray(item["labels"], dtype=object)
            dist = distributions[j]
            if item["sample"] and dist is not None and dist.shape[1] == len(labels):
                cumulative = np.cumsum(dist, axis=1)
                index = np.minimum((cumulative < rng.random(count)[:, None] * cumulative[:, -1:]).sum(axis=1), len(labels) - 1)
            else:
                index = np.clip(values[:, j].astype(int), 0, len(labels) - 1)
            result[item["name"]] = labels[index]
            if item["probabilities"] and dist is not None and dist.shape[1] == len(labels):
                for c, label in enumerate(labels):
                    result[f"P({item['name']}={label})"] = dist[:, c]
        return result

    def _run(self):
        temporary = self.destination.with_name(self.destination.name + ".part")
        try:
            plan, state = self.plan, self.state
            rng = np.random.default_rng(plan["seed"])
            lhs = {}
            if plan["sampling"] == "lhs":
                # One stratum per row on every numeric axis, strata shuffled independently.
                for item in plan["inputs"]:
                    if item["vary"] and item["kind"] == "numeric":
                        lhs[item["name"]] = (rng.permutation(plan["rows"]) + rng.random(plan["rows"])) / plan["rows"]
            order = [c for c, k in zip(state["source_columns"], state["types"]) if k in (1, 2, 5, 6)]
            with temporary.open("w", newline="") as handle:
                for start in range(0, plan["rows"], GENERATE_CHUNK):
                    stop = min(plan["rows"], start + GENERATE_CHUNK)
                    columns = self._inputs(rng, start, stop, lhs)
                    outputs = self._outputs(rng, columns, stop - start)
                    frame = pd.DataFrame({**columns, **outputs})
                    extra = [c for c in frame.columns if c not in order]
                    frame = frame[[c for c in order if c in frame.columns] + extra]
                    frame.to_csv(handle, index=False, header=start == 0)
                    if start == 0:
                        head = frame.head(20).astype(object).where(frame.head(20).notna(), None)
                        self.preview = {"columns": [str(c) for c in frame.columns],
                                        "rows": [[None if v is None else str(v) for v in row] for row in head.values.tolist()]}
                    self.written = stop
            temporary.replace(self.destination)
            self.registry[str(self.destination)] = True
            self.done = True
        except Exception as exc:
            traceback.print_exc()
            temporary.unlink(missing_ok=True)
            self.error = f"{type(exc).__name__}: {exc}"


def recent_runs(limit=30):
    """afpo_runs/* newest first, with enough detail to pick one."""
    root = Path("afpo_runs")
    runs = []
    if root.is_dir():
        for run in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            checkpoint = run / "checkpoint_latest.json"
            info = {"name": run.name, "path": str(run.resolve()), "checkpoint": str(checkpoint.resolve()) if checkpoint.exists() else None,
                    "modified": (checkpoint if checkpoint.exists() else run).stat().st_mtime, "model_card": (run / "model_card.json").exists()}
            try:
                manifest = json.loads((run / "manifest.json").read_text())
                info["dataset"] = manifest.get("dataset", {}).get("path")
                info["rows"] = manifest.get("dataset", {}).get("rows")
                info["seed"] = manifest.get("seed")
            except (OSError, ValueError):
                pass
            runs.append(info)
    return {"runs": runs}


def _config_file(name):
    """afpo_gui_configs/<name>.json for a plain name (letters, digits, spaces, . _ -)."""
    name = str(name or "").strip()
    if not name or len(name) > 80 or name.startswith(".") or any(not (ch.isalnum() or ch in " ._-") for ch in name):
        raise ValueError("A configuration name is 1-80 letters, digits, spaces, dots, dashes or underscores, and does not start with a dot")
    return GUI_CONFIGS / f"{name}.json"


def list_configs():
    """Saved Setup configurations, newest first."""
    files = sorted(GUI_CONFIGS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if GUI_CONFIGS.is_dir() else []
    return {"configs": [{"name": p.stem, "modified": p.stat().st_mtime} for p in files], "folder": str(GUI_CONFIGS.resolve())}


def save_config(name, config):
    """Write one configuration (the browser's Setup state, stored as given) and return the new list."""
    if not isinstance(config, dict):
        raise ValueError("A configuration is a JSON object")
    target = _config_file(name)
    GUI_CONFIGS.mkdir(exist_ok=True)
    partial = target.with_name(target.name + ".tmp")
    partial.write_text(json.dumps({"name": target.stem, "saved": time.time(), "config": clean(config)}, indent=1), encoding="utf-8")
    os.replace(partial, target)
    return list_configs()


def load_config(name):
    target = _config_file(name)
    if not target.is_file():
        raise ValueError(f"No saved configuration named {target.stem!r}")
    config = json.loads(target.read_text(encoding="utf-8")).get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{target} is not a saved configuration")
    return {"name": target.stem, "config": config}


def delete_config(name):
    target = _config_file(name)
    if not target.is_file():
        raise ValueError(f"No saved configuration named {target.stem!r}")
    target.unlink()
    return list_configs()


# ───────────────────────── HTTP server ─────────────────────────
def run_gui(host="127.0.0.1", port=DEFAULT_PORT, open_browser=True):
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    session, explorer = TrainingSession(), ModelExplorer()
    # New CSVs and uploaded images are kept next to the scripts.
    editor = CsvEditor(Path(__file__).resolve().parent)
    routes = {
        "/api/options": lambda b: options(),
        "/api/fs": lambda b: list_dir(b.get("path")),
        "/api/dataset/inspect": lambda b: inspect_dataset(b["path"], b.get("delimiter") or ","),
        "/api/editor": lambda b: editor.handle(b),
        "/api/train/check": lambda b: check_form(b),
        "/api/train/start": lambda b: session.start(b),
        "/api/train/stop": lambda b: session.stop(),
        "/api/train/choose": lambda b: session.pick(b["index"]),
        "/api/train/status": lambda b: session.status(b.get("since", 0), b.get("console_since", 0), b.get("snapshot_seq", 0)),
        "/api/runs": lambda b: recent_runs(),
        "/api/configs": lambda b: list_configs(),
        "/api/configs/save": lambda b: save_config(b.get("name"), b.get("config")),
        "/api/configs/load": lambda b: load_config(b.get("name")),
        "/api/configs/delete": lambda b: delete_config(b.get("name")),
        "/api/models/load": lambda b: explorer.load(b["path"]),
        "/api/models/detail": lambda b: explorer.detail(b["index"]),
        "/api/models/latex": lambda b: explorer.latex(b["index"]),
        "/api/models/edit_text": lambda b: explorer.edit_text(b["index"]),
        "/api/models/edit": lambda b: explorer.edit(b["index"], b.get("texts") or [], bool(b.get("refit"))),
        "/api/models/fit": lambda b: explorer.fit(b["index"], b.get("split", "train")),
        "/api/models/predict": lambda b: explorer.predict(b["index"], b.get("row") or {}),
        "/api/models/sweep": lambda b: explorer.sweep(b["index"], b["column"], b.get("base"), b.get("lo"), b.get("hi"), b.get("points", 160)),
        "/api/models/grid": lambda b: explorer.grid(b["index"], b["x"], b["y"], b.get("base"), b.get("x_lo"), b.get("x_hi"),
                                                    b.get("y_lo"), b.get("y_hi"), b.get("points", 60)),
        "/api/models/predict_csv": lambda b: explorer.predict_csv(b["index"], b["path"], b.get("delimiter") or ","),
        "/api/models/export": lambda b: explorer.export(b["index"]),
        "/api/models/generate/defaults": lambda b: explorer.generate_defaults(b["index"]),
        "/api/models/generate": lambda b: explorer.generate(b["index"], b.get("spec") or {}),
        "/api/models/generate/status": lambda b: explorer.generate_status(),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="application/json", extra=None):
            data = body if isinstance(body, bytes) else json.dumps(clean(body), allow_nan=False, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def _handle(self, body):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
            if path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            if path == "/download":
                # Only files this GUI wrote (prediction CSVs) can be downloaded.
                target = str(Path(body.get("path", "")).resolve())
                if target not in explorer.files or not Path(target).is_file():
                    return self._send(404, {"error": "unknown file"})
                return self._send(200, Path(target).read_bytes(), "text/csv",
                                  {"Content-Disposition": f'attachment; filename="{Path(target).name}"'})
            try:
                if path == "/api/editor/thumb":
                    return self._send(200, editor.thumbnail(body.get("path", "")), "image/png")
                if path not in routes:
                    return self._send(404, {"error": f"unknown route {path}"})
                self._send(200, routes[path](body))
            except Exception as exc:
                if not isinstance(exc, (ValueError, KeyError, FileNotFoundError, FileExistsError, RuntimeError)):
                    traceback.print_exc()
                self._send(400, {"error": f"{type(exc).__name__}: {exc}"})

        def do_GET(self):
            self._handle({k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            url = urlparse(self.path)
            if url.path == "/api/upload":
                try:
                    name = parse_qs(url.query).get("name", ["upload.csv"])[0]
                    return self._send(200, save_upload(name, self.rfile, n))
                except Exception as exc:
                    return self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
            if url.path == "/api/editor/image":
                # An image for a table cell (row, col) or, without them, for a prediction field.
                try:
                    query = {k: v[0] for k, v in parse_qs(url.query).items()}
                    if n > MAX_IMAGE_BYTES:
                        raise ValueError(f"The image is larger than {MAX_IMAGE_BYTES >> 20} MB")
                    return self._send(200, editor.save_image(query.get("name"), self.rfile.read(n), query.get("row"), query.get("col")))
                except Exception as exc:
                    return self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._send(400, {"error": "invalid JSON body"})
            self._handle(body)

    server = None
    for p in range(int(port), int(port) + 20):
        try:
            server = ThreadingHTTPServer((host, p), Handler)
            break
        except OSError:
            continue
    if server is None:
        raise OSError(f"No free port in {port}-{int(port) + 19}")
    url = f"http://{host}:{server.server_address[1]}/"
    if server.server_address[1] != int(port):
        print(f"Port {port} is busy; using {server.server_address[1]} instead.", flush=True)
    print(f"AFPO GUI running at {url}  (Ctrl+C to stop; working directory {os.getcwd()})", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping GUI.")
    finally:
        session.shutdown()
        server.server_close()


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Browser GUI for afpo.py")
    cli.add_argument("--port", type=int, default=DEFAULT_PORT)
    cli.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    cli.add_argument("--run", metavar="SPEC", help=argparse.SUPPRESS)
    options_ = cli.parse_args()
    if options_.run:
        sys.exit(run_spec(options_.run))
    run_gui(port=options_.port, open_browser=not options_.no_browser)
