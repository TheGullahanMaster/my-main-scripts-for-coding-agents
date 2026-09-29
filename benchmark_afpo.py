#!/usr/bin/env python3
"""Deterministic comparison benchmark for afpo.py search configurations.

Runs the real trainer (``afpo.train_from_setup``) headlessly on fixed
synthetic cases (the same families as ``benchmark_evo68.py``) plus the real
``MyTempos_v2.csv`` ratio-wrap data, under several search configurations and
fixed seeds, and records for every run:

- time and generation of the first "strong" model (training R^2 >= threshold),
- the best-so-far loss / R^2 trajectory per generation,
- the chosen model's held-out test R^2 (interpolation rows never trained on),
  validation loss, MDL bits, node count and equation,
- evaluation work (row x model evaluations) and wall-clock time.

Every configuration gets the same population and generation budget; the
evaluation counts are reported so any extra work a mechanism spends is visible
rather than hidden.  Output: a JSON file plus a summary table, with each
configuration compared against ``baseline``.

Examples:
  python benchmark_afpo.py                          # full matrix, 3 seeds
  python benchmark_afpo.py --quick                  # small smoke-sized matrix
  python benchmark_afpo.py --configs baseline,stages_fitness --cases piecewise,ratio_wrap
"""

import argparse
import concurrent.futures
import contextlib
import io
import json
import math
import multiprocessing
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import afpo

HERE = Path(__file__).resolve().parent

# name -> setup overrides (and CLI flags) on top of a single-population run.
CONFIGS = {
    "baseline": {},
    "no_equivalence": {"flags": ["--equivalence-collapse", "off"]},
    "legacy_qd_parents": {"flags": ["--qd-parent-choice", "legacy"]},
    "no_residual_archive": {"flags": ["--residual-archive", "off"]},
    "scale_balanced": {"flags": ["--scale-balanced-selection", "on"]},
    "stages_fitness": {"stages": {"mode": "fitness", "count": 3, "interval": 5}},
    "stages_age": {"stages": {"mode": "age", "count": 3, "interval": 5, "age_gap": 10}},
    "stages_both": {"stages": {"mode": "both", "count": 3, "interval": 5, "age_gap": 10}},
    "islands": {"island_count": 3},
    "islands_roles": {"island_count": 3, "roles": {"enabled": True, "interval": 10}},
    "islands_stages_roles": {"island_count": 3, "stages": {"mode": "both", "count": 2, "interval": 5, "age_gap": 10},
                             "roles": {"enabled": True, "interval": 10}},
}


def synthetic_cases(rows):
    """(train X, train y, test X, test y) per case; test rows interleave the training grid."""
    def grid(n, lo, hi, offset=0.):
        step = (hi - lo) / (n - 1)
        return np.linspace(lo, hi, n) + offset * step

    def build(n, offset):
        x = grid(n, .08, 1.25, offset)[:, None]
        z = grid(n, -.8, .8, offset)
        percent = grid(n, 55., 190., offset)
        ratio = 100. * 1.28555555555556 / percent
        small = max(8, n // 3)
        xs = grid(small, .08, 1.25, offset)[:, None]
        return {
            "polynomial": (x, 1.5 * x[:, 0] ** 2 - .7 * x[:, 0] + 2.),
            "rational": (x, 2. / (x[:, 0] + 1.2) - .4),
            "log_reciprocal": (x, -12. / np.log(x[:, 0] + 1.5) + 4.),
            "exp_interaction": (np.column_stack((x[:, 0], z)), np.exp(.7 * x[:, 0]) * (1. + z ** 2)),
            "piecewise": (x, np.where(x[:, 0] < .65, 2. * x[:, 0], 1.3 - x[:, 0])),
            "multi_scale": (x, 1e-3 + 400. * x[:, 0] ** 4),
            "ratio_wrap": (np.column_stack((np.full(n, 1.28555555555556), percent)),
                           np.where(ratio > 1.65, ratio / 2., np.where(ratio < .8, ratio * 2., ratio))),
            "small_data": (xs, 1.5 * xs[:, 0] ** 2 - .7 * xs[:, 0] + 2.),
        }
    train, test = build(rows, 0.), build(rows - 1, .5)
    # Midpoint rows lie strictly inside the training range (interpolation only).
    return {name: (*train[name], *test[name]) for name in train}


def mytempos_case():
    path = HERE / "MyTempos_v2.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    X, y = df.iloc[:, :-1].to_numpy(float), df.iloc[:, -1].to_numpy(float)
    # 25 rows cannot spare a test set; its "test" is the full data (reported as such).
    return X, y, X, y


def r2(pred, y):
    pred, y = np.asarray(pred, float).ravel(), np.asarray(y, float).ravel()
    total = float(np.sum((y - y.mean()) ** 2))
    if total <= 0:
        return 1. if np.allclose(pred, y) else 0.
    return 1. - float(np.sum((y - pred) ** 2)) / total


class Recorder:
    """PROGRESS_HOOK sink: best-so-far trajectory across all cells of a run."""

    def __init__(self, strong_r2):
        self.strong_r2 = strong_r2
        self.started = time.perf_counter()
        self.by_generation = {}
        self.first_strong = None
        self.evaluator = None
        self._r2_cache = {}

    def hook(self, **kw):
        best = kw["best_so_far"]
        self.evaluator = kw.get("evaluator") or self.evaluator
        if best is None:
            return
        key = (repr(best.trees), repr(best.scales))
        if key not in self._r2_cache:
            pred = afpo.predict_targets(best, kw["Xt"], kw["cats"])[:, 0]
            self._r2_cache[key] = r2(pred, kw["Yt"][:, 0])
        score, loss, generation = self._r2_cache[key], afpo.aggregate_loss(best), int(kw["generation"])
        entry = self.by_generation.setdefault(generation, {"generation": generation, "best_loss": math.inf, "best_r2": -math.inf})
        entry["best_loss"] = min(entry["best_loss"], loss)
        entry["best_r2"] = max(entry["best_r2"], score)
        entry["elapsed"] = round(time.perf_counter() - self.started, 4)
        if self.first_strong is None and score >= self.strong_r2:
            self.first_strong = {"generation": generation, "seconds": entry["elapsed"]}


def run_one(case, data, config_name, seed, args):
    X, y, X_test, y_test = data
    config = CONFIGS[config_name]
    columns = [f"x{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns); df["y"] = y
    ops = afpo.resolve_operator_groups(args.operator_groups.split(","))
    island_count = config.get("island_count", 1)
    setup = {"df": df, "types": [1] * len(columns) + [5], "delimiter": ",", "ops": ops, "affine_on": True, "coev": False,
             "dynamic_pressure_on": True, "adf_enabled": False, "nodes": args.nodes, "depth": args.depth,
             "island_count": island_count, "migration_interval": 10 if island_count > 1 else 0,
             "migrants_per_island": 2 if island_count > 1 else 0, "val_path": "", "validation_percent": 20, "metadata": {},
             "stages": config.get("stages"), "roles": config.get("roles")}
    argv = ["--population", str(args.population), "--max-generations", str(args.generations), "--seed", str(seed),
            "--workers", str(args.workers), *config.get("flags", [])]
    cli = afpo.parse_cli(argv)[1]
    recorder = Recorder(args.strong_r2)
    chosen = {}

    def choose(labels, choices, evaluation):
        chosen["model"], chosen["label"] = choices[0], labels[0]
        return 0

    log = io.StringIO()
    previous_hook = afpo.PROGRESS_HOOK
    with tempfile.TemporaryDirectory(prefix="afpo-bench-") as directory, contextlib.chdir(directory):
        df.to_csv("data.csv", index=False)
        setup["path"] = Path("data.csv")
        afpo.PROGRESS_HOOK = recorder.hook
        started = time.perf_counter()
        try:
            with contextlib.redirect_stdout(log):
                result = afpo.train_from_setup(cli, setup, choose_model=choose)
        finally:
            afpo.PROGRESS_HOOK = previous_hook
        elapsed = time.perf_counter() - started
        if args.keep_logs:
            Path(args.keep_logs).mkdir(parents=True, exist_ok=True)
            (Path(args.keep_logs) / f"{case}-{config_name}-{seed}.log").write_text(log.getvalue())
        _, _, _, _, state = afpo.load_checkpoint(result["checkpoint"], False)
    model = chosen["model"]
    test_pred = afpo.predict_targets(model, np.asarray(X_test, float), [None])[:, 0]
    train_pred = afpo.predict_targets(model, np.asarray(X, float), [None])[:, 0]
    selection = state.get("selection", {})
    island_config = state.get("island_config", {})
    trajectory = [recorder.by_generation[g] for g in sorted(recorder.by_generation)]
    diagnostics = recorder.evaluator.diagnostics() if recorder.evaluator is not None else {}
    return {
        "case": case, "config": config_name, "seed": seed,
        "rows": int(len(X)), "test_rows": int(len(X_test)), "features": int(X.shape[1]),
        "population": args.population, "generations": args.generations,
        "elapsed_seconds": round(elapsed, 3),
        "row_model_evaluations": int(diagnostics.get("row_model_evaluations", 0)),
        "first_strong_generation": None if recorder.first_strong is None else recorder.first_strong["generation"],
        "first_strong_seconds": None if recorder.first_strong is None else recorder.first_strong["seconds"],
        "strong_r2_threshold": args.strong_r2,
        "train_r2": r2(train_pred, y), "test_r2": r2(test_pred, y_test),
        "validation_loss": (selection.get("selected_metrics") or {}).get("loss"),
        "mdl_bits": afpo.model_complexity(model), "nodes": int(sum(afpo.node_size(t) for t in model.trees)),
        "equation": result["equation"],
        "equivalence_redraws": int(afpo.EQUIVALENCE_STATS["children_redrawn"]),
        "stages": {k: island_config.get("stages", {}).get(k) for k in ("mode", "count", "promotion_events", "promoted", "evicted", "reseeds")},
        "roles": {k: island_config.get("roles", {}).get(k) for k in ("enabled", "updates", "retirements", "collapses", "fragment_migrants")},
        "trajectory": trajectory,
    }


def summarize(runs, strong_r2):
    """Per-configuration means over cases and seeds, plus deltas against baseline."""
    table = {}
    for config in dict.fromkeys(run["config"] for run in runs):
        mine = [run for run in runs if run["config"] == config]
        strong = [run["first_strong_generation"] for run in mine if run["first_strong_generation"] is not None]
        table[config] = {
            "runs": len(mine),
            "mean_test_r2": float(np.mean([max(run["test_r2"], -1.) for run in mine])),
            "solved": sum(run["test_r2"] >= strong_r2 for run in mine),
            "reached_strong": len(strong),
            "median_first_strong_generation": None if not strong else float(np.median(strong)),
            "mean_mdl_bits": float(np.mean([run["mdl_bits"] for run in mine])),
            "mean_evaluations": float(np.mean([run["row_model_evaluations"] for run in mine])),
            "mean_seconds": float(np.mean([run["elapsed_seconds"] for run in mine])),
        }
    base = table.get("baseline")
    if base:
        for row in table.values():
            row["delta_test_r2_vs_baseline"] = row["mean_test_r2"] - base["mean_test_r2"]
            row["evaluations_vs_baseline"] = row["mean_evaluations"] / base["mean_evaluations"] if base["mean_evaluations"] else None
    per_case = {}
    for run in runs:
        per_case.setdefault(run["case"], {}).setdefault(run["config"], []).append(run["test_r2"])
    per_case = {case: {config: float(np.mean(values)) for config, values in configs.items()} for case, configs in per_case.items()}
    return table, per_case


def print_summary(table, per_case, strong_r2):
    print(f"\n{'config':<22}{'runs':>5}{'test R2':>9}{'solved':>8}{'strong':>8}{'1st gen':>9}{'bits':>8}{'evals x':>9}{'sec':>8}")
    for config, row in table.items():
        first = "-" if row["median_first_strong_generation"] is None else f"{row['median_first_strong_generation']:.0f}"
        ratio = "-" if row.get("evaluations_vs_baseline") is None else f"{row['evaluations_vs_baseline']:.2f}"
        print(f"{config:<22}{row['runs']:>5}{row['mean_test_r2']:>9.4f}{row['solved']:>8}{row['reached_strong']:>8}{first:>9}"
              f"{row['mean_mdl_bits']:>8.0f}{ratio:>9}{row['mean_seconds']:>8.1f}")
    print(f"(solved = held-out test R2 >= {strong_r2}; strong = training R2 reached it during search; "
          "1st gen = median generation it did; evals x = evaluation work relative to baseline)")
    configs = list(table)
    print("\nMean held-out test R2 per case:")
    print(f"{'case':<16}" + "".join(f"{c[:14]:>15}" for c in configs))
    for case, values in per_case.items():
        print(f"{case:<16}" + "".join(f"{values.get(c, float('nan')):>15.4f}" for c in configs))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--seeds", default="20260716,20260717,20260718", help="Comma-separated fixed seeds (at least three recommended)")
    parser.add_argument("--cases", default="all", help="Comma-separated case names, or 'all' (includes mytempos when the CSV exists)")
    parser.add_argument("--configs", default="all", help=f"Comma-separated from: {', '.join(CONFIGS)}")
    parser.add_argument("--operator-groups", default="1,2,3,7", help="afpo operator group IDs (default: arithmetic, powers, exp/log, conditionals)")
    parser.add_argument("--nodes", type=int, default=21)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1, help="afpo scoring processes per run (1 keeps runs deterministic)")
    parser.add_argument("--jobs", type=int, default=1, help="Runs executed in parallel processes (results are identical; wall-clock timings get noisier)")
    parser.add_argument("--strong-r2", type=float, default=.99)
    parser.add_argument("--quick", action="store_true", help="Smoke-sized matrix: 12 generations, population 48, one seed")
    parser.add_argument("--keep-logs", metavar="DIR", help="Save each run's trainer output to DIR")
    parser.add_argument("--output", type=Path, default=Path("benchmark_afpo.json"))
    args = parser.parse_args()
    if args.quick:
        args.generations, args.population, args.seeds = 12, 48, args.seeds.split(",")[0]
    if args.generations < 1 or args.rows < 12:
        parser.error("generations >= 1 and rows >= 12 are required")

    cases = synthetic_cases(args.rows)
    tempos = mytempos_case()
    if tempos is not None:
        cases["mytempos"] = tempos
    if args.cases != "all":
        wanted = args.cases.split(",")
        unknown = [c for c in wanted if c not in cases]
        if unknown:
            parser.error(f"Unknown case(s): {', '.join(unknown)}; available: {', '.join(cases)}")
        cases = {name: cases[name] for name in wanted}
    configs = list(CONFIGS) if args.configs == "all" else args.configs.split(",")
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        parser.error(f"Unknown config(s): {', '.join(unknown)}; available: {', '.join(CONFIGS)}")
    cells = max(CONFIGS[c].get("island_count", 1) * (CONFIGS[c].get("stages") or {}).get("count", 1) for c in configs)
    if args.population < 8 * cells:
        parser.error(f"--population must be at least {8 * cells} for the selected configurations (8 models per island x stage)")
    seeds = [int(s) for s in args.seeds.split(",")]

    jobs = [(case, data, config, seed) for case, data in cases.items() for config in configs for seed in seeds]
    runs, total, started = [], len(jobs), time.perf_counter()

    def report(run):
        runs.append(run)
        print(f"[{len(runs)}/{total}] {run['case']:<16} {run['config']:<22} seed {run['seed']}: test R2={run['test_r2']:.4f} "
              f"first strong gen={run['first_strong_generation']} {run['elapsed_seconds']:.1f}s  {run['equation']}", flush=True)

    if args.jobs <= 1:
        for job in jobs:
            report(run_one(*job, args))
    else:
        # One BLAS thread per process so parallel runs do not oversubscribe the CPU.
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(name, "1")
        with concurrent.futures.ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            for future in concurrent.futures.as_completed([pool.submit(run_one, *job, args) for job in jobs]):
                report(future.result())
    order = {(case, config, seed): index for index, (case, _, config, seed) in enumerate(jobs)}
    runs.sort(key=lambda run: order[(run["case"], run["config"], run["seed"])])
    table, per_case = summarize(runs, args.strong_r2)
    print_summary(table, per_case, args.strong_r2)
    payload = {"settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "cases": list(cases), "configs": {c: CONFIGS[c] for c in configs}, "seeds": seeds,
               "wall_seconds": round(time.perf_counter() - started, 2), "summary": table, "per_case_test_r2": per_case, "runs": runs}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=True))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
