#!/usr/bin/env python3
"""Harsh-gradient benchmark for afpo.py: discontinuous and non-smooth targets.

Reuses ``benchmark_afpo.run_one`` (the real ``afpo.train_from_setup`` run,
headless and seeded) on targets whose loss landscape is flat or jumps:
steps, staircases, modulo sawtooths, square waves, if/else branches with a
jump, an integer modulo, a 2-D conditional and the existing ratio_wrap.

Training rows lie on a grid; test rows are the midpoints between them
(interpolation only).  A test row whose two neighbouring training rows straddle
a discontinuity is dropped: the training data cannot say which side of the gap
the jump sits on, so scoring it would measure luck, not convergence.

Per run it records success (held-out test R^2 >= --solved-r2 on the kept
rows), the first generation and seconds at which training R^2 reached
--strong-r2, and wall time.  Output: JSON plus a per-case table.

Examples:
  python bench_harsh.py                             # all cases, 10 seeds
  python bench_harsh.py --cases step,sawtooth --seed-count 3 --jobs 4
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

import benchmark_afpo


def _one_d(lo, hi, f, breaks):
    """1-D case: grid training rows, midpoint test rows, straddled midpoints dropped."""
    def build(rows):
        x = np.linspace(lo, hi, rows)
        left, right = x[:-1], x[1:]
        xt = (left + right) / 2
        keep = np.array([not any(a < b <= c for b in breaks) for a, c in zip(left, right)])
        return x[:, None], f(x), xt[keep][:, None], f(xt[keep]), int((~keep).sum())
    return build


def _multiples(step, lo, hi, phase=0.):
    k = np.arange(np.ceil((lo - phase) / step), np.floor((hi - phase) / step) + 1)
    return list(phase + k * step)


def _int_mod(rows):
    k = np.arange(rows + rows // 4, dtype=float)
    test = np.random.default_rng(7).permutation(len(k))[: rows // 4]
    mask = np.ones(len(k), bool); mask[test] = False
    y = np.mod(k, 7.)
    return k[mask][:, None], y[mask], k[~mask][:, None], y[~mask], 0


def _cond2d(rows):
    def f(X):
        return np.where(X[:, 0] > X[:, 1], X[:, 0] * X[:, 1], X[:, 0] + X[:, 1])
    side = int(np.ceil(np.sqrt(rows)))
    g = np.linspace(.1, 1.1, side)
    X = np.array([(a, b) for a in g for b in g])
    h = (g[1] - g[0]) / 2
    gt = g[:-1] + h
    Xt = np.array([(a, b) for a in gt for b in gt])
    # Drop test points whose cell is cut by the x0 = x1 boundary.
    keep = np.abs(Xt[:, 0] - Xt[:, 1]) > h * 1.01
    return X, f(X), Xt[keep], f(Xt[keep]), int((~keep).sum())


def _ratio_wrap(rows):
    def f(p):
        r = 100. * 1.28555555555556 / p
        return np.where(r > 1.65, r / 2., np.where(r < .8, r * 2., r))
    breaks = [100. * 1.28555555555556 / 1.65, 100. * 1.28555555555556 / .8]
    x, y, xt, yt, dropped = _one_d(55., 190., f, breaks)(rows)
    const = np.full((len(x), 1), 1.28555555555556)
    const_t = np.full((len(xt), 1), 1.28555555555556)
    return np.hstack((const, x)), y, np.hstack((const_t, xt)), yt, dropped


CASES = {
    # name: (builder(rows) -> X, y, X_test, y_test, dropped, formula)
    "step": (_one_d(0., 1., lambda x: np.where(x < .53, 1., 3.), [.53]), "1 if x<0.53 else 3"),
    "staircase": (_one_d(0., 2., lambda x: np.floor(2.5 * x), _multiples(.4, 0., 2.)), "floor(2.5x)"),
    "sawtooth": (_one_d(0., 3., lambda x: np.mod(x, .7), _multiples(.7, 0., 3.)), "x mod 0.7"),
    "scaled_mod": (_one_d(0., 2., lambda x: 2. * np.mod(3. * x + .4, 1.) - 1., _multiples(1 / 3, 0., 2., .2)),
                   "2*((3x+0.4) mod 1)-1"),
    "square_wave": (_one_d(0., 4., lambda x: np.where(np.mod(x, .8) < .4, 1., -1.), _multiples(.4, 0., 4.)),
                    "1 if x mod 0.8 < 0.4 else -1"),
    "jump_ifelse": (_one_d(0., 1.5, lambda x: np.where(x < .6, x ** 2 + 1., 3. - 2. * x), [.6]),
                    "x^2+1 if x<0.6 else 3-2x"),
    "kink_ifelse": (_one_d(.08, 1.25, lambda x: np.where(x < .65, 2. * x, 1.3 - x), []), "2x if x<0.65 else 1.3-x"),
    "int_mod": (_int_mod, "k mod 7, k integer"),
    "cond2d": (_cond2d, "x0*x1 if x0>x1 else x0+x1"),
    "ratio_wrap": (_ratio_wrap, "wrap 128.56/p into [0.8, 1.65]"),
}


def summarize(runs, args):
    table = {}
    for case in dict.fromkeys(r["case"] for r in runs):
        mine = [r for r in runs if r["case"] == case]
        strong = [r for r in mine if r["first_strong_generation"] is not None]
        test = np.array([max(r["test_r2"], -1.) for r in mine])
        table[case] = {
            "runs": len(mine),
            "solved": int(sum(r["test_r2"] >= args.solved_r2 for r in mine)),
            "reached_strong": len(strong),
            "median_test_r2": float(np.median(test)), "mean_test_r2": float(test.mean()),
            "worst_test_r2": float(test.min()),
            "median_first_strong_generation": float(np.median([r["first_strong_generation"] for r in strong])) if strong else None,
            "median_first_strong_seconds": float(np.median([r["first_strong_seconds"] for r in strong])) if strong else None,
            "mean_seconds": float(np.mean([r["elapsed_seconds"] for r in mine])),
            "dropped_test_rows": mine[0]["dropped_test_rows"], "test_rows": mine[0]["test_rows"],
        }
    return table


def print_summary(table, args):
    print(f"\n{'case':<13}{'runs':>5}{'solved':>8}{'strong':>8}{'med R2':>9}{'worst':>8}{'1st gen':>9}{'1st s':>7}{'sec':>7}")
    for case, row in table.items():
        gen = "-" if row["median_first_strong_generation"] is None else f"{row['median_first_strong_generation']:.0f}"
        sec = "-" if row["median_first_strong_seconds"] is None else f"{row['median_first_strong_seconds']:.1f}"
        print(f"{case:<13}{row['runs']:>5}{row['solved']:>8}{row['reached_strong']:>8}{row['median_test_r2']:>9.4f}"
              f"{row['worst_test_r2']:>8.3f}{gen:>9}{sec:>7}{row['mean_seconds']:>7.1f}")
    print(f"(solved = held-out test R2 >= {args.solved_r2}; strong = training R2 >= {args.strong_r2} during search; "
          "1st gen / 1st s = median over runs that got there; R2 floored at -1)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261001, help="First seed")
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--cases", default="all", help=f"Comma-separated from: {', '.join(CASES)}")
    parser.add_argument("--config", default="baseline", help="A benchmark_afpo.CONFIGS name")
    parser.add_argument("--operator-groups", default="1,2,3,7,8",
                        help="afpo operator groups (default: arithmetic incl. mod/floordiv, powers, exp/log, conditionals, rounding)")
    parser.add_argument("--nodes", type=int, default=21)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1, help="Runs in parallel processes (results identical; timings noisier)")
    parser.add_argument("--strong-r2", type=float, default=.99)
    parser.add_argument("--solved-r2", type=float, default=.99)
    parser.add_argument("--keep-logs", metavar="DIR")
    parser.add_argument("--output", type=Path, default=Path("bench_harsh.json"))
    args = parser.parse_args()
    names = list(CASES) if args.cases == "all" else args.cases.split(",")
    unknown = [c for c in names if c not in CASES]
    if unknown:
        parser.error(f"Unknown case(s): {', '.join(unknown)}")
    if args.config not in benchmark_afpo.CONFIGS:
        parser.error(f"Unknown config {args.config}")

    data, dropped = {}, {}
    for name in names:
        *xy, dropped[name] = CASES[name][0](args.rows)
        data[name] = tuple(xy)
    seeds = [args.seed + i for i in range(args.seed_count)]
    jobs = [(name, data[name], args.config, seed) for name in names for seed in seeds]
    runs, started = [], time.perf_counter()

    def report(run):
        run["dropped_test_rows"] = dropped[run["case"]]
        run["formula"] = CASES[run["case"]][1]
        runs.append(run)
        print(f"[{len(runs)}/{len(jobs)}] {run['case']:<12} seed {run['seed']}: test R2={run['test_r2']:.4f} "
              f"strong gen={run['first_strong_generation']} {run['elapsed_seconds']:.1f}s  {run['equation']}", flush=True)

    if args.jobs <= 1:
        for job in jobs:
            report(benchmark_afpo.run_one(*job, args))
    else:
        with concurrent.futures.ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            for future in concurrent.futures.as_completed([pool.submit(benchmark_afpo.run_one, *job, args) for job in jobs]):
                report(future.result())
    order = {(job[0], job[3]): i for i, job in enumerate(jobs)}
    runs.sort(key=lambda r: order[(r["case"], r["seed"])])
    table = summarize(runs, args)
    print_summary(table, args)
    payload = {"settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "cases": {n: CASES[n][1] for n in names}, "seeds": seeds,
               "wall_seconds": round(time.perf_counter() - started, 2), "summary": table, "runs": runs}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=True))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
