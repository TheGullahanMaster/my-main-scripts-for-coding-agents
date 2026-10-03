#!/usr/bin/env python3
"""Island-role benchmark for afpo.py: which fixed roles help beside "auto"?

Every configuration runs the same number of islands (default 5: island 1 is
the generalist, islands 2-5 get the listed roles), the same population and
the same generation budget, so only the roles differ:

  all_auto       auto, auto, auto, auto          (the self-organising default)
  plus_simplifier auto, auto, auto, simplifier
  plus_explorer  auto, auto, auto, explorer
  plus_refiner   auto, auto, auto, refiner
  all_roles      auto, simplifier, explorer, refiner
  families       auto, auto, family:A, family:B   (A, B chosen per case: the
                 operator groups its target uses, so this one gets a hint)

Cases: benchmark_afpo.py's synthetic one-input families, bench_complex.py's
multi-input targets, and a synthetic Newton's-law cooling case with
thermometer noise (sd 0.1).  "Solved" means held-out R^2 >= --solve-r2.
Results are compared with all_auto, paired on (case, seed), with bootstrap
95% intervals.  Runs use --cell-workers 1 so --jobs parallel runs do not
oversubscribe the CPU.

Example:
  python bench_roles.py --jobs 4 --seed-count 3
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

import bench_complex
import benchmark_afpo as base

ROLE_SETS = {
    "all_auto": ["auto", "auto", "auto", "auto"],
    "plus_simplifier": ["auto", "auto", "auto", "simplifier"],
    "plus_explorer": ["auto", "auto", "auto", "explorer"],
    "plus_refiner": ["auto", "auto", "auto", "refiner"],
    "all_roles": ["auto", "simplifier", "explorer", "refiner"],
    "families": None,     # per case, see FAMILIES
}
# Operator groups (afpo.OPERATOR_GROUPS) each target is built from: the
# families config gives one island to each.  Group 1 (arithmetic) is always
# included in a family island, so "1" alone means arithmetic only.
FAMILIES = {
    "polynomial": ("2", "1"), "rational": ("1", "2"), "log_reciprocal": ("3", "1"),
    "exp_interaction": ("3", "2"), "piecewise": ("7", "1"), "multi_scale": ("2", "3"),
    "ratio_wrap": ("7", "1"), "small_data": ("2", "1"),
    "reuse_poly3": ("2", "1"), "ratio_chain5": ("1", "2"), "products6": ("1", "2"), "exp_reuse4": ("3", "2"),
    "log_exp4": ("3", "2"), "trig_mix4": ("4", "2"), "branch_mix5": ("7", "1"), "sum8": ("1", "2"),
    "newton_cooling": ("3", "1"),
}
DEFAULT_CASES = ("polynomial", "rational", "log_reciprocal", "exp_interaction", "piecewise", "multi_scale", "ratio_wrap",
                 "reuse_poly3", "ratio_chain5", "products6", "exp_reuse4", "log_exp4", "trig_mix4", "branch_mix5", "sum8",
                 "newton_cooling")


def newton_cooling_case(rows, seed=2027):
    """T = Ta + (T0 - Ta) exp(-k t) with Ta = 22, k = 0.05 and sd 0.1 noise; inputs T0, t."""
    draw = np.random.default_rng(seed)
    def sample(n):
        X = np.column_stack((draw.uniform(60, 95, n), draw.uniform(0, 60, n)))
        return X, 22 + (X[:, 0] - 22) * np.exp(-.05 * X[:, 1]) + draw.normal(0, .1, n)
    return (*sample(rows), *sample(rows))


def all_cases(rows, complex_rows):
    cases = base.synthetic_cases(rows)
    cases.update(bench_complex.complex_cases(complex_rows))
    cases["newton_cooling"] = newton_cooling_case(complex_rows)
    return cases


def config_for(config, case, islands, interval):
    roles = ROLE_SETS[config]
    if roles is None:
        first, second = FAMILIES[case]
        roles = ["auto"] * (islands - 3) + [f"family:{first}", f"family:{second}"]
    else:
        roles = list(roles)
    if len(roles) != islands - 1:
        # Other island counts: keep the listed fixed roles, pad with auto.
        fixed = [role for role in roles if role != "auto"]
        roles = ["auto"] * (islands - 1 - len(fixed)) + fixed
    return {"island_count": islands, "roles": {"enabled": True, "interval": interval, "assignments": roles},
            "flags": ["--cell-workers", "1"]}


def run_job(case, data, config, seed, args):
    name = f"{config}@{case}"
    base.CONFIGS[name] = config_for(config, case, args.islands, args.role_interval)
    run = base.run_one(case, data, name, seed, args)
    run["config"] = config
    run["roles"] = base.CONFIGS[name]["roles"]["assignments"]
    run.pop("trajectory", None)
    return run


def paired(runs, config, reference, solve_r2, draw):
    ref = {(r["case"], r["seed"]): r for r in runs if r["config"] == reference}
    pairs = [(r, ref[(r["case"], r["seed"])]) for r in runs if r["config"] == config and (r["case"], r["seed"]) in ref]
    if not pairs:
        return None
    solved = np.array([(a["test_r2"] >= solve_r2) - (b["test_r2"] >= solve_r2) for a, b in pairs], float)
    r2 = np.array([max(a["test_r2"], -1.) - max(b["test_r2"], -1.) for a, b in pairs])
    bits = np.array([a["mdl_bits"] - b["mdl_bits"] for a, b in pairs], float)
    index = draw.integers(0, len(pairs), size=(4000, len(pairs)))
    interval = lambda values: [float(np.quantile(values[index].mean(1), .025)), float(np.quantile(values[index].mean(1), .975))]
    return {"n": len(pairs), "solve_difference": float(solved.mean()), "solve_ci95": interval(solved),
            "newly_solved": int(np.sum(solved > 0)), "newly_failed": int(np.sum(solved < 0)),
            "test_r2_difference": float(r2.mean()), "test_r2_ci95": interval(r2),
            "bits_difference": float(bits.mean()), "bits_ci95": interval(bits)}


def summarize(runs, solve_r2, reference="all_auto"):
    table = {}
    draw = np.random.default_rng(0)
    for config in dict.fromkeys(run["config"] for run in runs):
        mine = [run for run in runs if run["config"] == config]
        table[config] = {
            "runs": len(mine), "solved": int(sum(run["test_r2"] >= solve_r2 for run in mine)),
            "mean_test_r2": float(np.mean([max(run["test_r2"], -1.) for run in mine])),
            "mean_mdl_bits": float(np.mean([run["mdl_bits"] for run in mine])),
            "mean_evaluations": float(np.mean([run["row_model_evaluations"] for run in mine])),
            "mean_seconds": float(np.mean([run["elapsed_seconds"] for run in mine])),
        }
        if config != reference:
            table[config]["paired"] = paired(runs, config, reference, solve_r2, draw)
    per_case = {}
    for run in runs:
        per_case.setdefault(run["case"], {}).setdefault(run["config"], []).append(run["test_r2"] >= solve_r2)
    per_case = {case: {config: int(sum(values)) for config, values in configs.items()} for case, configs in per_case.items()}
    return table, per_case


def print_summary(table, per_case, solve_r2, seeds):
    print(f"\n{'config':<16}{'runs':>5}{'solved':>8}{'test R2':>9}{'bits':>7}{'evals':>10}{'sec':>7}")
    for config, row in table.items():
        print(f"{config:<16}{row['runs']:>5}{row['solved']:>8}{row['mean_test_r2']:>9.4f}{row['mean_mdl_bits']:>7.0f}"
              f"{row['mean_evaluations']:>10.3g}{row['mean_seconds']:>7.0f}")
    print(f"(solved = held-out R2 >= {solve_r2}; bits = MDL of the chosen model)")
    print("\nPaired against all_auto (same case and seed); bootstrap 95% intervals:")
    print(f"{'config':<16}{'solve diff':>11}{'interval':>18}{'+/-':>7}{'R2 diff':>10}{'interval':>20}{'bits diff':>10}{'interval':>16}")
    for config, row in table.items():
        info = row.get("paired")
        if not info:
            continue
        s, r, b = info["solve_ci95"], info["test_r2_ci95"], info["bits_ci95"]
        record = f"{info['newly_solved']}/{info['newly_failed']}"
        solve, fit, bits = f"[{s[0]:+.3f}, {s[1]:+.3f}]", f"[{r[0]:+.4f}, {r[1]:+.4f}]", f"[{b[0]:+.0f}, {b[1]:+.0f}]"
        print(f"{config:<16}{info['solve_difference']:>+11.3f}{solve:>18}{record:>7}{info['test_r2_difference']:>+10.4f}"
              f"{fit:>20}{info['bits_difference']:>+10.1f}{bits:>16}")
    configs = list(table)
    print(f"\nSolved runs per case (out of {seeds}):")
    print(f"{'case':<16}" + "".join(f"{c[:13]:>14}" for c in configs))
    for case, values in per_case.items():
        print(f"{case:<16}" + "".join(f"{values.get(c, 0):>14}" for c in configs))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--population", type=int, default=600)
    parser.add_argument("--islands", type=int, default=5)
    parser.add_argument("--role-interval", type=int, default=10)
    parser.add_argument("--rows", type=int, default=64, help="Rows of the one-input synthetic cases")
    parser.add_argument("--complex-rows", type=int, default=200, help="Rows of the multi-input and cooling cases")
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--first-seed", type=int, default=20261003)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--configs", default=",".join(ROLE_SETS))
    parser.add_argument("--operator-groups", default="1,2,3,4,5,7", help="arithmetic, powers, exp/log, trig, smooth, conditionals")
    parser.add_argument("--nodes", type=int, default=25)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--solve-r2", type=float, default=.999)
    parser.add_argument("--keep-logs", metavar="DIR")
    parser.add_argument("--output", type=Path, default=Path("bench_roles.json"))
    args = parser.parse_args()
    args.strong_r2 = args.solve_r2
    if args.population < 8 * args.islands:
        parser.error(f"--population must be at least {8 * args.islands} (8 models per island)")

    cases = all_cases(args.rows, args.complex_rows)
    wanted = args.cases.split(",")
    unknown = [c for c in wanted if c not in cases]
    if unknown:
        parser.error(f"Unknown case(s): {', '.join(unknown)}; available: {', '.join(cases)}")
    configs = args.configs.split(",")
    unknown = [c for c in configs if c not in ROLE_SETS]
    if unknown:
        parser.error(f"Unknown config(s): {', '.join(unknown)}; available: {', '.join(ROLE_SETS)}")
    seeds = [args.first_seed + i for i in range(args.seed_count)]
    jobs = [(case, cases[case], config, seed) for seed in seeds for case in wanted for config in configs]
    runs, started = [], time.perf_counter()
    log = args.output.with_suffix(".jsonl")
    log.write_text("")

    def report(run):
        runs.append(run)
        with log.open("a") as handle:
            handle.write(json.dumps(run, allow_nan=True) + "\n")
        print(f"[{len(runs)}/{len(jobs)}] {run['case']:<15} {run['config']:<16} seed {run['seed']}: test R2={run['test_r2']:.5f} "
              f"bits={run['mdl_bits']:.0f} {run['elapsed_seconds']:.0f}s  {run['equation']}", flush=True)

    with concurrent.futures.ProcessPoolExecutor(max(1, args.jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run_job, *job, args) for job in jobs]):
            report(future.result())
    order = {(case, config, seed): i for i, (case, _, config, seed) in enumerate(jobs)}
    runs.sort(key=lambda run: order[(run["case"], run["config"], run["seed"])])
    table, per_case = summarize(runs, args.solve_r2)
    print_summary(table, per_case, args.solve_r2, len(seeds))
    payload = {"settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "role_sets": ROLE_SETS, "families": FAMILIES, "seeds": seeds,
               "wall_seconds": round(time.perf_counter() - started, 1), "summary": table, "per_case_solved": per_case, "runs": runs}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=True))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
