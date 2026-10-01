#!/usr/bin/env python3
"""Convergence benchmark for afpo.py on harsh, discontinuous targets.

The smooth families in ``benchmark_afpo.py`` say little about landscapes made
of jumps: modulo sawtooths, stairs, if/else branches and integer rules, where
a model is either structurally right (and then exact) or plateaus far away.
This harness runs the real trainer (through ``benchmark_afpo.run_one``) on
such targets and reports, per configuration:

- solve rate: held-out test R^2 >= ``--solve-r2`` (default 0.999, i.e. exact
  up to round-off; a near miss on a jump target is still the wrong structure),
- median generation at which training R^2 first reached that level,
- mean held-out test R^2 (floored at -1) and wall-clock time,

with paired (case, seed) bootstrap intervals against ``baseline``.

Examples:
  python bench_jump.py --jobs 4                         # baseline, 10 seeds
  python bench_jump.py --configs baseline,<name> --jobs 4
"""

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import time
from pathlib import Path

for _blas_threads in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_blas_threads, "1")

import numpy as np

import benchmark_afpo as base

# name -> setup overrides / CLI flags, as in benchmark_afpo.CONFIGS.
CONFIGS = {
    # The search as it was before the jump mechanisms existed.
    "baseline": {"flags": ["--jump-constant-scan", "off", "--jump-mutation-weight", "0"]},
    "jump_scan": {"flags": ["--jump-constant-scan", "on", "--jump-mutation-weight", "0"]},
    "jump_mutation": {"flags": ["--jump-constant-scan", "off", "--jump-mutation-weight", "1"]},
    "jump_scan_mutation": {"flags": ["--jump-constant-scan", "on", "--jump-mutation-weight", "1"]},
}
base.CONFIGS.update(CONFIGS)


def harsh_cases(rows):
    """(train X, train y, test X, test y); test rows never appear in training."""
    def grid(n, lo, hi, offset=0.):
        step = (hi - lo) / (n - 1)
        return np.linspace(lo, hi, n) + offset * step

    def one_d(fn, lo, hi):
        x, xt = grid(rows, lo, hi)[:, None], grid(rows - 1, lo, hi, .5)[:, None]
        return x, fn(x[:, 0]), xt, fn(xt[:, 0])

    def two_d(fn, lo, hi, seed):
        draw = np.random.default_rng(seed)
        x, xt = draw.uniform(lo, hi, (4 * rows, 2)), draw.uniform(lo, hi, (4 * rows, 2))
        return x, fn(x[:, 0], x[:, 1]), xt, fn(xt[:, 0], xt[:, 1])

    def integers(fn, hi, seed):
        n = np.random.default_rng(seed).permutation(np.arange(1, hi + 1)).astype(float)
        cut = (2 * len(n)) // 3
        x, xt = np.sort(n[:cut])[:, None], np.sort(n[cut:])[:, None]
        return x, fn(x[:, 0]), xt, fn(xt[:, 0])

    return {
        # modulo with a non-integer period
        "mod_saw": one_d(lambda x: np.mod(x, 2.5), 0., 10.),
        # trend plus sawtooth: needs both parts at once
        "mod_trend": one_d(lambda x: x + 2. * np.mod(x, 3.), 0., 12.),
        # triangle wave: abs of a shifted sawtooth
        "triangle": one_d(lambda x: np.abs(np.mod(x, 4.) - 2.), 0., 12.),
        # staircase
        "stair": one_d(lambda x: 2. * np.floor(x / 1.5) + .5, 0., 9.),
        # if/else on a threshold of one input, different law per branch
        "branch_threshold": two_d(lambda a, b: np.where(a > .4, 3. * b, b * b), -1., 1., 11),
        # if/else on a comparison between inputs
        "branch_compare": two_d(lambda a, b: np.where(a > b, a - b, 2. * b), -1., 1., 12),
        # integer rule: odd n keeps n, even n gives 0
        "parity": integers(lambda n: np.mod(n, 2.) * n, 120, 13),
        # Collatz step: n/2 for even n, 3n+1 for odd n
        "collatz": integers(lambda n: np.where(np.mod(n, 2.) == 0, n / 2., 3. * n + 1.), 120, 14),
    }


def run_job(case, data, config, seed, args):
    base.CONFIGS.update(CONFIGS)  # spawned workers import this module afresh
    return base.run_one(case, data, config, seed, args)


def summarize(runs, solve_r2):
    table = {}
    configs = list(dict.fromkeys(run["config"] for run in runs))
    for config in configs:
        mine = [run for run in runs if run["config"] == config]
        firsts = [run["first_strong_generation"] for run in mine if run["first_strong_generation"] is not None]
        table[config] = {
            "runs": len(mine),
            "solved": int(sum(run["test_r2"] >= solve_r2 for run in mine)),
            "solve_rate": float(np.mean([run["test_r2"] >= solve_r2 for run in mine])),
            "reached_train": len(firsts),
            "median_first_generation": None if not firsts else float(np.median(firsts)),
            "mean_test_r2": float(np.mean([max(run["test_r2"], -1.) for run in mine])),
            "mean_seconds": float(np.mean([run["elapsed_seconds"] for run in mine])),
            "mean_evaluations": float(np.mean([run["row_model_evaluations"] for run in mine])),
        }
    if "baseline" in table:
        ref = {(r["case"], r["seed"]): r for r in runs if r["config"] == "baseline"}
        draw = np.random.default_rng(0)
        for config in configs:
            if config == "baseline":
                continue
            pairs = [(r, ref[(r["case"], r["seed"])]) for r in runs if r["config"] == config and (r["case"], r["seed"]) in ref]
            if not pairs:
                continue
            solved = np.array([(a["test_r2"] >= solve_r2) - (b["test_r2"] >= solve_r2) for a, b in pairs], float)
            r2 = np.array([max(a["test_r2"], -1.) - max(b["test_r2"], -1.) for a, b in pairs])
            idx = draw.integers(0, len(pairs), size=(4000, len(pairs)))
            table[config]["paired"] = {
                "n": len(pairs),
                "solve_rate_difference": float(solved.mean()),
                "solve_ci95": [float(np.quantile(solved[idx].mean(1), .025)), float(np.quantile(solved[idx].mean(1), .975))],
                "test_r2_difference": float(r2.mean()),
                "test_r2_ci95": [float(np.quantile(r2[idx].mean(1), .025)), float(np.quantile(r2[idx].mean(1), .975))],
                "newly_solved": int(np.sum(solved > 0)), "newly_failed": int(np.sum(solved < 0)),
            }
    per_case = {}
    for run in runs:
        cell = per_case.setdefault(run["case"], {}).setdefault(run["config"], {"solved": 0, "runs": 0, "firsts": []})
        cell["runs"] += 1
        cell["solved"] += int(run["test_r2"] >= solve_r2)
        if run["first_strong_generation"] is not None:
            cell["firsts"].append(run["first_strong_generation"])
    return table, per_case


def print_summary(table, per_case, solve_r2):
    print(f"\n{'config':<26}{'runs':>5}{'solved':>8}{'rate':>7}{'1st gen':>9}{'test R2':>9}{'sec':>7}  paired vs baseline")
    for config, row in table.items():
        first = "-" if row["median_first_generation"] is None else f"{row['median_first_generation']:.0f}"
        line = (f"{config:<26}{row['runs']:>5}{row['solved']:>8}{row['solve_rate']:>7.2f}{first:>9}"
                f"{row['mean_test_r2']:>9.3f}{row['mean_seconds']:>7.1f}")
        paired = row.get("paired")
        if paired:
            lo, hi = paired["solve_ci95"]
            line += (f"  solve {paired['solve_rate_difference']:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                     f" (+{paired['newly_solved']}/-{paired['newly_failed']})")
        print(line)
    print(f"(solved = held-out test R2 >= {solve_r2}; 1st gen = median generation training R2 first reached it)")
    configs = list(table)
    print("\nSolved runs per case (median first generation):")
    print(f"{'case':<20}" + "".join(f"{c[:22]:>24}" for c in configs))
    for case, cells in per_case.items():
        out = []
        for c in configs:
            cell = cells.get(c)
            if cell is None:
                out.append("-")
                continue
            first = f" ({np.median(cell['firsts']):.0f})" if cell["firsts"] else ""
            out.append(f"{cell['solved']}/{cell['runs']}{first}")
        print(f"{case:<20}" + "".join(f"{o:>24}" for o in out))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--first-seed", type=int, default=20261001)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--configs", default="baseline", help=f"Comma-separated from: {', '.join(CONFIGS)}")
    parser.add_argument("--operator-groups", default="1,2,7,8", help="arithmetic (incl. mod), powers, conditionals, rounding")
    parser.add_argument("--nodes", type=int, default=21)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--solve-r2", type=float, default=.999)
    parser.add_argument("--keep-logs", metavar="DIR")
    parser.add_argument("--output", type=Path, default=Path("bench_jump.json"))
    args = parser.parse_args()
    args.strong_r2 = args.solve_r2  # read by benchmark_afpo.run_one

    cases = harsh_cases(args.rows)
    if args.cases != "all":
        cases = {name: cases[name] for name in args.cases.split(",")}
    configs = args.configs.split(",")
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        parser.error(f"Unknown config(s): {', '.join(unknown)}")
    seeds = [args.first_seed + i for i in range(args.seed_count)]
    jobs = [(case, data, config, seed) for case, data in cases.items() for config in configs for seed in seeds]
    runs, started = [], time.perf_counter()

    def report(run):
        runs.append(run)
        print(f"[{len(runs)}/{len(jobs)}] {run['case']:<17} {run['config']:<24} seed {run['seed']}: test R2={run['test_r2']:.4f} "
              f"first gen={run['first_strong_generation']} {run['elapsed_seconds']:.1f}s  {run['equation']}", flush=True)

    if args.jobs <= 1:
        for job in jobs:
            report(run_job(*job, args))
    else:
        with concurrent.futures.ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            for future in concurrent.futures.as_completed([pool.submit(run_job, *job, args) for job in jobs]):
                report(future.result())
    order = {(case, config, seed): i for i, (case, _, config, seed) in enumerate(jobs)}
    runs.sort(key=lambda run: order[(run["case"], run["config"], run["seed"])])
    table, per_case = summarize(runs, args.solve_r2)
    print_summary(table, per_case, args.solve_r2)
    for run in runs:
        run.pop("trajectory", None)
    payload = {"settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "configs": {c: CONFIGS[c] for c in configs}, "seeds": seeds,
               "wall_seconds": round(time.perf_counter() - started, 1), "summary": table, "runs": runs}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=True))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
