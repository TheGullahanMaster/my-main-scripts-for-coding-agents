#!/usr/bin/env python3
"""Smooth-activation benchmark for afpo.py: SiLU, GELU, Mish and a soft switch.

These targets sit next to cheap approximations: relu(x) alone reaches R^2
0.997 against SiLU and x*sigmoid(1.7x) reaches 0.99996 against GELU, so the
usual "solved at R^2 >= .99" says nothing here.  A run counts as solved only
when the chosen model is near exact on held-out midpoints (--solved-r2,
default 0.99999) AND keeps the shape outside the training range
(--extrap-r2, default 0.9999 on x in [-8,-4.2] and [4.2,8]); a pile of relu
kinks or a polynomial patch fails the second test.

Reuses ``benchmark_afpo.run_one`` (the real ``afpo.train_from_setup`` run,
headless and seeded).  Training rows are a grid on [-4, 4]; test rows are the
midpoints.

Examples:
  python bench_activation.py --jobs 4
  python bench_activation.py --cases gelu_erf --seed-count 3 --config no_gate
"""

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
import time
from pathlib import Path

for _blas_threads in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_blas_threads, "1")

import numpy as np

import afpo
import benchmark_afpo

_erf = np.vectorize(math.erf)


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


CASES = {
    "silu": (lambda x: x * _sigmoid(x), "x*sigmoid(x)"),
    "gelu_erf": (lambda x: .5 * x * (1. + _erf(x / math.sqrt(2.))), "0.5x(1+erf(x/sqrt 2))"),
    "gelu_tanh": (lambda x: .5 * x * (1. + np.tanh(math.sqrt(2. / math.pi) * (x + .044715 * x ** 3))),
                  "0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))"),
    "mish": (lambda x: x * np.tanh(np.logaddexp(0., x)), "x*tanh(softplus(x))"),
    "soft_switch": (lambda x: (1. - _sigmoid(4. * (x - 1.))) * .5 * x + _sigmoid(4. * (x - 1.)) * (2. - x),
                    "0.5x blended into 2-x by sigmoid(4(x-1))"),
}


def build(case, rows):
    f = CASES[case][0]
    x = np.linspace(-4., 4., rows)
    xt = (x[:-1] + x[1:]) / 2
    xe = np.concatenate([np.linspace(-8., -4.2, 20), np.linspace(4.2, 8., 20)])
    return x[:, None], f(x), xt[:, None], f(xt), xe[:, None], f(xe)


def r2(pred, y):
    pred = np.asarray(pred, float)
    if not np.all(np.isfinite(pred)):
        return -1.
    return float(1. - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2))


def run_case(case, config, seed, args):
    """benchmark_afpo.run_one plus R^2 of the chosen model outside the training range."""
    X, y, Xt, yt, Xe, ye = build(case, args.rows)
    seen, predict = [], afpo.predict_targets

    def spy(model, *rest, **kwargs):
        seen.append(model)
        return predict(model, *rest, **kwargs)

    afpo.predict_targets = spy
    try:
        run = benchmark_afpo.run_one(case, (X, y, Xt, yt), config, seed, args)
    finally:
        afpo.predict_targets = predict
    run["extrap_r2"] = r2(predict(seen[-1], np.asarray(Xe, float), [None])[:, 0], ye)
    return run


def solved(run, args):
    return run["test_r2"] >= args.solved_r2 and run["extrap_r2"] >= args.extrap_r2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261001, help="First seed")
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--cases", default="all", help=f"Comma-separated from: {', '.join(CASES)}")
    parser.add_argument("--config", default="baseline", help="Comma-separated benchmark_afpo.CONFIGS names")
    parser.add_argument("--operator-groups", default=",".join(afpo.DEFAULT_GROUP_IDS),
                        help="afpo operator groups (default: afpo's own defaults, 1-9)")
    parser.add_argument("--nodes", type=int, default=21)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1, help="Runs in parallel processes (results identical; timings noisier)")
    parser.add_argument("--strong-r2", type=float, default=.99999)
    parser.add_argument("--solved-r2", type=float, default=.99999)
    parser.add_argument("--extrap-r2", type=float, default=.9999)
    parser.add_argument("--keep-logs", metavar="DIR")
    parser.add_argument("--output", type=Path, default=Path("bench_activation.json"))
    args = parser.parse_args()
    names = list(CASES) if args.cases == "all" else args.cases.split(",")
    configs = args.config.split(",")
    unknown = [c for c in names if c not in CASES] + [c for c in configs if c not in benchmark_afpo.CONFIGS]
    if unknown:
        parser.error(f"Unknown case(s) or config(s): {', '.join(unknown)}")

    seeds = [args.seed + i for i in range(args.seed_count)]
    jobs = [(name, config, seed) for config in configs for name in names for seed in seeds]
    runs, started = [], time.perf_counter()

    def report(run):
        runs.append(run)
        print(f"[{len(runs)}/{len(jobs)}] {run['config']:<10} {run['case']:<11} seed {run['seed']}: "
              f"test R2={run['test_r2']:.7f} extrap R2={run['extrap_r2']:.5f} {'solved' if solved(run, args) else '      '} "
              f"{run['elapsed_seconds']:.0f}s  {run['equation']}", flush=True)

    if args.jobs <= 1:
        for job in jobs:
            report(run_case(*job, args))
    else:
        with concurrent.futures.ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            for future in concurrent.futures.as_completed([pool.submit(run_case, *job, args) for job in jobs]):
                report(future.result())
    order = {job: i for i, job in enumerate(jobs)}
    runs.sort(key=lambda r: order[(r["case"], r["config"], r["seed"])])
    summary = {}
    print(f"\n{'config':<11}{'case':<13}{'runs':>5}{'solved':>8}{'med test R2':>13}{'med extrap':>12}{'nodes':>7}{'sec':>6}")
    for config in configs:
        for name in names:
            mine = [r for r in runs if r["case"] == name and r["config"] == config]
            row = {"runs": len(mine), "solved": sum(solved(r, args) for r in mine),
                   "median_test_r2": float(np.median([r["test_r2"] for r in mine])),
                   "median_extrap_r2": float(np.median([max(r["extrap_r2"], -1.) for r in mine])),
                   "median_nodes": float(np.median([r["nodes"] for r in mine])),
                   "mean_seconds": float(np.mean([r["elapsed_seconds"] for r in mine]))}
            summary[f"{config}/{name}"] = row
            print(f"{config:<11}{name:<13}{row['runs']:>5}{row['solved']:>8}{row['median_test_r2']:>13.7f}"
                  f"{row['median_extrap_r2']:>12.5f}{row['median_nodes']:>7.0f}{row['mean_seconds']:>6.0f}")
    print(f"(solved = test R2 >= {args.solved_r2} and extrapolation R2 >= {args.extrap_r2})")
    payload = {"settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "cases": {n: CASES[n][1] for n in names}, "seeds": seeds,
               "wall_seconds": round(time.perf_counter() - started, 2), "summary": summary, "runs": runs}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=True, default=str))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
