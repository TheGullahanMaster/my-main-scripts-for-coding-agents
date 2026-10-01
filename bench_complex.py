#!/usr/bin/env python3
"""Recovery benchmark for afpo.py on complex multi-variable equations.

Targets use 3-8 inputs, reuse the same input several times and chain many
operators (products, ratios, powers, exp/log/sin, plus a few jumps).  Rows
are random draws from [0.5, 2]^d (positive, so every operator is defined);
the held-out rows are a second independent draw.  "Recovered" means held-out
R^2 >= --solve-r2 (default 0.999).  Configurations and reporting are those of
bench_jump.py.

Example:
  python bench_complex.py --configs baseline,jump_scan_mutation --jobs 4
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

import bench_jump as jump

TARGETS = {
    # name: (input count, target)
    "reuse_poly3": (3, lambda x: x[0] * x[1] + x[0] * x[2] - x[1] * x[2] + x[0] ** 2),
    "ratio_chain5": (5, lambda x: (x[0] + x[1]) * (x[2] - x[3]) / (1 + x[4] ** 2)),
    "products6": (6, lambda x: x[0] * x[1] + x[2] / x[3] - x[4] * x[5]),
    "exp_reuse4": (4, lambda x: np.exp(-x[0] * x[1]) * (x[0] + x[2]) + np.sqrt(x[1] * x[3])),
    "log_exp4": (4, lambda x: np.log(1 + x[0] ** 2) * x[1] + x[2] * np.exp(.5 * x[3]) - x[0] * x[3]),
    "trig_mix4": (4, lambda x: np.sin(x[0] * x[1]) * x[2] + np.cos(x[3]) * x[0] ** 2),
    "branch_mix5": (5, lambda x: np.where(x[0] > x[1], x[2] * x[3], x[2] + x[3]) + np.mod(3 * x[4], 2.)),
    "sum8": (8, lambda x: 2 * x[0] * x[1] + 3 * x[2] * x[3] - x[4] * x[5] + x[6] / x[7]),
}


def complex_cases(rows, seed=2026):
    cases = {}
    for offset, (name, (d, f)) in enumerate(TARGETS.items()):
        draw = np.random.default_rng(seed + offset)
        X, Xt = draw.uniform(.5, 2., (rows, d)), draw.uniform(.5, 2., (rows, d))
        cases[name] = (X, f(X.T), Xt, f(Xt.T))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--first-seed", type=int, default=20261001)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--configs", default="baseline", help=f"Comma-separated from: {', '.join(jump.CONFIGS)}")
    parser.add_argument("--operator-groups", default="1,2,3,4,7", help="arithmetic, powers, exp/log, trig, conditionals")
    parser.add_argument("--nodes", type=int, default=31)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--solve-r2", type=float, default=.999)
    parser.add_argument("--keep-logs", metavar="DIR")
    parser.add_argument("--output", type=Path, default=Path("bench_complex.json"))
    args = parser.parse_args()
    args.strong_r2 = args.solve_r2

    cases = complex_cases(args.rows)
    if args.cases != "all":
        cases = {name: cases[name] for name in args.cases.split(",")}
    configs = args.configs.split(",")
    seeds = [args.first_seed + i for i in range(args.seed_count)]
    jobs = [(case, data, config, seed) for case, data in cases.items() for config in configs for seed in seeds]
    runs, started = [], time.perf_counter()

    def report(run):
        runs.append(run)
        print(f"[{len(runs)}/{len(jobs)}] {run['case']:<14} {run['config']:<20} seed {run['seed']}: test R2={run['test_r2']:.4f} "
              f"first gen={run['first_strong_generation']} {run['elapsed_seconds']:.1f}s  {run['equation']}", flush=True)

    with concurrent.futures.ProcessPoolExecutor(max(1, args.jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in concurrent.futures.as_completed([pool.submit(jump.run_job, *job, args) for job in jobs]):
            report(future.result())
    order = {(case, config, seed): i for i, (case, _, config, seed) in enumerate(jobs)}
    runs.sort(key=lambda run: order[(run["case"], run["config"], run["seed"])])
    table, per_case = jump.summarize(runs, args.solve_r2)
    jump.print_summary(table, per_case, args.solve_r2)
    for run in runs:
        run.pop("trajectory", None)
    payload = {"settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "configs": {c: jump.CONFIGS[c] for c in configs}, "seeds": seeds,
               "wall_seconds": round(time.perf_counter() - started, 1), "summary": table, "runs": runs}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=True))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
