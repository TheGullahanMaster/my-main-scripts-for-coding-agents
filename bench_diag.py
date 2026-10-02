#!/usr/bin/env python3
"""Diagnostic benchmark for afpo.py: one case per known search bottleneck.

bench_harsh saturated (89/100 on main, 2026-10-02), so it could no longer rank
convergence changes.  This suite was chosen by probing 44 candidates on main
(4 seeds each) and keeping the ones that are neither always nor never solved:

- Multivariate structure: conditionals and max over 2-3 interacting inputs.
- Deep composition: f(g(h(x))), so shallow approximations fall short.
- Structure inside a composition: tanh(T1+T2), T1/(1+T2), exp(T1*T2), so an
  additive multi-term readout cannot solve the case on its own.
- Repeated motif: the same sub-expression several times (ADF territory).
- High precision: simple structure with awkward constants (constant fitter).
- Discontinuous / piecewise: jumps and folds, no ambiguous held-out rows.
- Deceptive: a mediocre simple approximation exists next to the exact form.
- Irrelevant features: few used inputs among noise columns.
- Scale / pathology: tiny outputs, correlated inputs, a nearby pole.
- Network distillation: a small fixed random MLP.

R^2 >= 0.99 does not separate these cases (most smooth ones pass it with an
approximation), so a run counts as *exact* when held-out 1 - R^2 <= 1e-9, and
every run also gets a continuous score: digits = -log10(1 - R^2), capped at 12.

Per run it records the generation, cumulative row x model evaluations and
seconds at which training R^2 first reached .9, .99, .999, .9999 and exact;
the generation-0 best R^2 and loss (a better start versus better evolution);
an anytime score (mean capped training digits over all generations, so a change
that reaches the same end twice as fast still shows); and the final held-out
R^2, digits, nodes, MDL bits and wall time.

FROZEN.  The cases, data generators, split rules, budget, thresholds and seeds
are version BENCH_VERSION.  bench_diag_baseline.json records main's results and
a hash of every case's data; test_bench_diag.py fails when the data changes.
Do not edit or replace a case without bumping BENCH_VERSION and re-recording
the baseline, and never because a feature happened to solve it.

The --reserve cases are held back: feature work never runs or tunes on them.
Their main baseline is recorded once; run them again only once, at the end.

Examples:
  python bench_diag.py --jobs 4                                  # main suite, 10 seeds
  python bench_diag.py --jobs 4 --flags "--foo on" --compare bench_diag_baseline.json
  python bench_diag.py --cases cond_sum,motif_tanh3 --seed-count 3
"""

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import shlex
import time
from pathlib import Path

for _blas_threads in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_blas_threads, "1")

import numpy as np

import benchmark_afpo

BENCH_VERSION = 1
FIRST_SEED = 20263001
SEED_COUNT = 10
EXACT_GAP = 1e-9            # exact: 1 - R^2 <= this
DIGITS_CAP = 12.
THRESHOLDS = {"0.9": .9, "0.99": .99, "0.999": .999, "0.9999": .9999, "exact": 1. - EXACT_GAP}
MAIN_GROUPS = "1,2,3,7,8"   # arithmetic incl. mod/floordiv, powers, exp/log, conditionals, rounding
SMOOTH_GROUPS = "1,2,3,5,7,8"  # plus sigmoid/tanh/relu for the tanh and sigmoid cases


# --- data builders: each returns (X, y, X_test, y_test) -----------------------------------------

def _grid_1d(f, lo, hi, label=None, rows=64):
    """Grid training rows, midpoint test rows; a midpoint whose two neighbours lie in different
    branches of ``label`` is dropped (the data cannot say where the jump is)."""
    x = np.linspace(lo, hi, rows)
    xt = (x[:-1] + x[1:]) / 2
    keep = np.ones(len(xt), bool) if label is None else label(x[:-1]) == label(x[1:])
    return x[:, None], f(x), xt[keep][:, None], f(xt[keep])


def _random(f, d, lo, hi, seed, label=None, rows=200, prep=None):
    """Independent uniform training and test draws; a test row whose nearest training row lies in
    another branch of ``label`` is dropped."""
    draw = np.random.default_rng(seed)
    lo, hi = np.broadcast_to(lo, (d,)), np.broadcast_to(hi, (d,))
    X, Xt = draw.uniform(lo, hi, (rows, d)), draw.uniform(lo, hi, (rows, d))
    if prep is not None:
        X, Xt = prep(X), prep(Xt)
    if label is not None:
        nearest = np.argmin(((Xt[:, None, :] - X[None, :, :]) ** 2).sum(-1), axis=1)
        Xt = Xt[label(Xt) == label(X[nearest])]
    return X, f(X), Xt, f(Xt)


def _integers(f, hi, seed):
    n = np.random.default_rng(seed).permutation(np.arange(1, hi + 1)).astype(float)
    cut = (2 * len(n)) // 3
    x, xt = np.sort(n[:cut]), np.sort(n[cut:])
    return x[:, None], f(x), xt[:, None], f(xt)


def _mlp(seed, d, hidden, act):
    draw = np.random.default_rng(seed)
    W, b, v = draw.normal(0, 1.2, (hidden, d)), draw.normal(0, .5, hidden), draw.normal(0, 1, hidden)
    return lambda X: v @ act(W @ X.T + b[:, None])


def _correlated(X):
    X = X.copy()
    X[:, 1] = X[:, 0] + .05 * X[:, 1]
    return X


def _sigmoid(z):
    return 1. / (1. + np.exp(-z))


_fl, _th = np.floor, np.tanh


def _c(X, i):
    return X[:, i]


# name: (family, formula, builder, operator groups, nodes, depth, reserve)
CASES = {
    # Multivariate structure
    "cond_sum": ("multivariate", "x0*x1 if x0+x1>1.2 else x0-x1",
                 lambda: _random(lambda X: np.where(X[:, 0] + X[:, 1] > 1.2, X[:, 0] * X[:, 1], X[:, 0] - X[:, 1]),
                                 2, 0., 1., 31, lambda X: X[:, 0] + X[:, 1] > 1.2), MAIN_GROUPS, 21, 5, False),
    "cond3d": ("multivariate", "x2*x0 if x0>x1 else x2+x1",
               lambda: _random(lambda X: np.where(X[:, 0] > X[:, 1], X[:, 2] * X[:, 0], X[:, 2] + X[:, 1]),
                               3, .1, 1.1, 32, lambda X: X[:, 0] > X[:, 1]), MAIN_GROUPS, 21, 5, False),
    "nested_cond": ("multivariate", "(x0+x1 if x1>.3 else 2x1) if x0>.5 else x0*x1",
                    lambda: _random(lambda X: np.where(X[:, 0] > .5, np.where(X[:, 1] > .3, X[:, 0] + X[:, 1], 2 * X[:, 1]),
                                                       X[:, 0] * X[:, 1]),
                                    2, 0., 1., 33, lambda X: (X[:, 0] > .5) * 2 + (X[:, 1] > .3)), MAIN_GROUPS, 21, 5, False),
    "max_planes": ("multivariate", "max(x0-x1, .5x2, x1*x2)",
                   lambda: _random(lambda X: np.maximum(np.maximum(X[:, 0] - X[:, 1], .5 * X[:, 2]), X[:, 1] * X[:, 2]),
                                   3, 0., 1., 36), MAIN_GROUPS, 21, 5, False),
    # Deep composition
    "deep_rational2": ("deep", "1/(1+(x0*x1-1)^2)",
                       lambda: _random(lambda X: 1 / (1 + (X[:, 0] * X[:, 1] - 1) ** 2), 2, .2, 2., 51),
                       MAIN_GROUPS, 21, 5, False),
    "deep_log_chain3": ("deep", "log(1+sqrt(x0)*x1)^2 * x2",
                        lambda: _random(lambda X: np.log(1 + np.sqrt(X[:, 0]) * X[:, 1]) ** 2 * X[:, 2], 3, .2, 3., 52),
                        MAIN_GROUPS, 25, 6, False),
    "deep_sqrt_exp": ("deep", "sqrt(exp(x)-x)", lambda: _grid_1d(lambda x: np.sqrt(np.exp(x) - x), -2., 3.),
                      MAIN_GROUPS, 21, 5, False),
    # Structure inside a composition
    "comp_tanh_sum": ("composition", "tanh(x0*x1 - x2/x3)",
                      lambda: _random(lambda X: _th(X[:, 0] * X[:, 1] - X[:, 2] / X[:, 3]), 4, .5, 2., 71),
                      SMOOTH_GROUPS, 21, 5, False),
    "comp_ratio": ("composition", "x0*x1/(1+x2^2*x0)",
                   lambda: _random(lambda X: X[:, 0] * X[:, 1] / (1 + X[:, 2] ** 2 * X[:, 0]), 3, .2, 2., 72),
                   MAIN_GROUPS, 21, 5, False),
    "comp_exp_prod": ("composition", "exp(-(x0+x1)*(x2-.5x0))",
                      lambda: _random(lambda X: np.exp(-(X[:, 0] + X[:, 1]) * (X[:, 2] - .5 * X[:, 0])), 3, .2, 1.5, 73),
                      MAIN_GROUPS, 21, 5, False),
    "comp_log_mix": ("composition", "log(1+x0*x1+x2/x3)",
                     lambda: _random(lambda X: np.log(1 + X[:, 0] * X[:, 1] + X[:, 2] / X[:, 3]), 4, .5, 2., 74),
                     MAIN_GROUPS, 21, 5, False),
    # Repeated motif
    "motif_square3": ("motif", "(x0^2+x1)^2+(x1^2+x2)^2+(x2^2+x0)^2",
                      lambda: _random(lambda X: (X[:, 0] ** 2 + X[:, 1]) ** 2 + (X[:, 1] ** 2 + X[:, 2]) ** 2
                                      + (X[:, 2] ** 2 + X[:, 0]) ** 2, 3, -1., 1., 53), MAIN_GROUPS, 31, 6, False),
    "motif_tanh3": ("motif", "tanh(1.3x0-.4x1+.2)+.7tanh(-.8x0+1.1x1-.5)-1.2tanh(.5x0+.9x1)",
                    lambda: _random(lambda X: _th(1.3 * X[:, 0] - .4 * X[:, 1] + .2) + .7 * _th(-.8 * X[:, 0] + 1.1 * X[:, 1] - .5)
                                    - 1.2 * _th(.5 * X[:, 0] + .9 * X[:, 1]), 2, -2., 2., 54), SMOOTH_GROUPS, 35, 6, False),
    # High precision
    "prec_exp2": ("precision", "1.2345e^(-.6789x) + .1111e^(.4321x)",
                  lambda: _grid_1d(lambda x: 1.2345 * np.exp(-.6789 * x) + .1111 * np.exp(.4321 * x), 0., 5.),
                  MAIN_GROUPS, 21, 5, False),
    "prec_rational": ("precision", "(2.718x0+.3333)/(1.4142+x1^2)",
                      lambda: _random(lambda X: (2.718 * X[:, 0] + .3333) / (1.4142 + X[:, 1] ** 2), 2, -1., 1., 55),
                      MAIN_GROUPS, 21, 5, False),
    # Discontinuous / piecewise
    "three_piece": ("piecewise", "1-2x if x<.3; .4+x^2 if x<.7; 2.5-x",
                    lambda: _grid_1d(lambda x: np.where(x < .3, 1 - 2 * x, np.where(x < .7, .4 + x * x, 2.5 - x)), 0., 1.,
                                     lambda x: (x >= .3).astype(int) + (x >= .7)), MAIN_GROUPS, 21, 5, False),
    "triangle_amp": ("piecewise", "x*abs((x mod 1)-.5)",
                     lambda: _grid_1d(lambda x: x * np.abs(np.mod(x, 1.) - .5), 0., 4.), MAIN_GROUPS, 21, 5, False),
    "saw_in_exp": ("piecewise", "x*exp(-(x mod 1.3))",
                   lambda: _grid_1d(lambda x: x * np.exp(-np.mod(x, 1.3)), 0., 5., lambda x: _fl(x / 1.3)),
                   MAIN_GROUPS, 21, 5, False),
    "mod_affine3": ("piecewise", "(x0*x1+x2) mod 1",
                    lambda: _random(lambda X: np.mod(X[:, 0] * X[:, 1] + X[:, 2], 1.), 3, 0., 1.5, 82,
                                    lambda X: _fl(X[:, 0] * X[:, 1] + X[:, 2])), MAIN_GROUPS, 21, 5, False),
    # Deceptive
    "decept_small_term": ("deceptive", "x*e^-x + .02x^3",
                          lambda: _grid_1d(lambda x: x * np.exp(-x) + .02 * x ** 3, 0., 4.), MAIN_GROUPS, 21, 5, False),
    "narrow_bump": ("deceptive", "e^(-200(x-.4)^2) + .3x",
                    lambda: _grid_1d(lambda x: np.exp(-200 * (x - .4) ** 2) + .3 * x, 0., 1.), MAIN_GROUPS, 21, 5, False),
    # Irrelevant features
    "irrelevant8": ("irrelevant", "x0*x3 - sqrt(x5) + .5x0^2 (8 inputs)",
                    lambda: _random(lambda X: X[:, 0] * X[:, 3] - np.sqrt(X[:, 5]) + .5 * X[:, 0] ** 2, 8, .2, 2., 56),
                    MAIN_GROUPS, 21, 5, False),
    "step_distractors": ("irrelevant", "(2 if x2>.6 else -1) + .5x0 (5 inputs)",
                         lambda: _random(lambda X: np.where(X[:, 2] > .6, 2., -1.) + X[:, 0] * .5, 5, 0., 1., 37,
                                         lambda X: X[:, 2] > .6), MAIN_GROUPS, 21, 5, False),
    # Scale / pathology
    "scale_tiny_out": ("scale", "1e-7*(x0*e^-x1 + x1)",
                       lambda: _random(lambda X: 1e-7 * (X[:, 0] * np.exp(-X[:, 1]) + X[:, 1]), 2, 0., 3., 58),
                       MAIN_GROUPS, 21, 5, False),
    "correlated_diff": ("scale", "exp(20(x1-x0))+x0, x1 = x0 + .05u",
                        lambda: _random(lambda X: np.exp(20 * (X[:, 1] - X[:, 0])) + X[:, 0], 2, 0., 1., 59, prep=_correlated),
                        MAIN_GROUPS, 21, 5, False),
    "near_pole": ("scale", "1/(x-1.03)+x on [0,1]", lambda: _grid_1d(lambda x: 1 / (x - 1.03) + x, 0., 1.),
                  MAIN_GROUPS, 21, 5, False),
    # Network distillation
    "mlp_tanh_2x3": ("distillation", "random 2-3-1 tanh MLP (seed 60)",
                     lambda: _random(_mlp(60, 2, 3, _th), 2, -1.5, 1.5, 61), SMOOTH_GROUPS, 35, 6, False),
    "mlp_relu_2x3": ("distillation", "random 2-3-1 relu MLP (seed 62)",
                     lambda: _random(_mlp(62, 2, 3, lambda z: np.maximum(z, 0.)), 2, -1.5, 1.5, 63), MAIN_GROUPS, 35, 6, False),
    # Reserve: never run or tuned on during feature work; run once at the end.
    "comp_cond_inside": ("composition", "sqrt(1 + (x0*x2 if x0>x1 else x1+x2))",
                         lambda: _random(lambda X: np.sqrt(1 + np.where(X[:, 0] > X[:, 1], X[:, 0] * X[:, 2], X[:, 1] + X[:, 2])),
                                         3, .1, 1.1, 75, lambda X: X[:, 0] > X[:, 1]), MAIN_GROUPS, 21, 5, True),
    "abs_kinks2d": ("multivariate", "|x0-.4| + 2|x1-.7| - |x0-x1|",
                    lambda: _random(lambda X: np.abs(X[:, 0] - .4) + 2 * np.abs(X[:, 1] - .7) - np.abs(X[:, 0] - X[:, 1]),
                                    2, 0., 1., 38), MAIN_GROUPS, 21, 5, True),
    "decept_soft_abs": ("deceptive", "sqrt(x^2+.04)", lambda: _grid_1d(lambda x: np.sqrt(x * x + .04), -1., 1.),
                        MAIN_GROUPS, 21, 5, True),
    "decept_softplus": ("deceptive", "log(1+e^(6x))/6", lambda: _grid_1d(lambda x: np.log1p(np.exp(6 * x)) / 6, -1., 1.),
                        MAIN_GROUPS, 21, 5, True),
    "mod3_branch": ("piecewise", "n/3 if n mod 3 = 0 else 2n-1 (integers)",
                    lambda: _integers(lambda n: np.where(np.mod(n, 3.) == 0, n / 3., 2 * n - 1), 150, 21), MAIN_GROUPS, 21, 5, True),
    "scale_mixed": ("scale", "2.5e-6*x0^2 + 3e3/x1, x0 in [1e3,5e4], x1 in [.01,.1]",
                    lambda: _random(lambda X: 2.5e-6 * X[:, 0] ** 2 + 3e3 / X[:, 1], 2, [1e3, 1e-2], [5e4, 1e-1], 57),
                    MAIN_GROUPS, 21, 5, True),
    "mlp_sigmoid_3x2": ("distillation", "random 3-2-1 sigmoid MLP (seed 64)",
                        lambda: _random(_mlp(64, 3, 2, _sigmoid), 3, -1.5, 1.5, 65), SMOOTH_GROUPS, 31, 6, True),
}
MAIN_CASES = [name for name, spec in CASES.items() if not spec[6]]
RESERVE_CASES = [name for name, spec in CASES.items() if spec[6]]


def build(name):
    return tuple(np.asarray(part, float) for part in CASES[name][2]())


def data_hash(data):
    """Platform-tolerant fingerprint: shapes plus every value at 10 significant digits."""
    digest = hashlib.sha256()
    for part in data:
        digest.update(repr(part.shape).encode())
        digest.update(",".join(f"{v:.10g}" for v in part.ravel()).encode())
    return digest.hexdigest()


def digits(r2):
    return float(np.clip(-np.log10(max(1. - r2, 10. ** -DIGITS_CAP)), 0., DIGITS_CAP))


def run_job(name, data, seed, args):
    family, formula, _, groups, nodes, depth, reserve = CASES[name]
    job_args = argparse.Namespace(**vars(args))
    job_args.operator_groups, job_args.nodes, job_args.depth = groups, nodes, depth
    job_args.strong_r2 = THRESHOLDS["0.999"]
    benchmark_afpo.CONFIGS["bench_diag"] = {"flags": shlex.split(args.flags)}
    run = benchmark_afpo.run_one(name, data, "bench_diag", seed, job_args)
    trajectory = run.pop("trajectory")
    reached = {}
    for label, threshold in THRESHOLDS.items():
        hit = next((entry for entry in trajectory if entry["best_r2"] >= threshold), None)
        reached[label] = None if hit is None else {"generation": hit["generation"], "evaluations": hit.get("evaluations"),
                                                   "seconds": hit["elapsed"]}
    first = trajectory[0] if trajectory else {}
    run.update({
        "family": family, "formula": formula, "reserve": reserve, "operator_groups": groups, "max_nodes": nodes, "max_depth": depth,
        "flags": args.flags, "reached": reached,
        "generation0_best_r2": first.get("best_r2"), "generation0_best_loss": first.get("best_loss"),
        "anytime_digits": float(np.mean([digits(entry["best_r2"]) for entry in trajectory])) if trajectory else 0.,
        "final_train_loss": trajectory[-1]["best_loss"] if trajectory else None,
        "test_digits": digits(run["test_r2"]), "exact": bool(1. - run["test_r2"] <= EXACT_GAP),
    })
    for key in ("config", "stages", "roles", "strong_r2_threshold", "first_strong_generation", "first_strong_seconds"):
        run.pop(key, None)
    return run


def summarize(runs):
    table = {}
    for name in dict.fromkeys(run["case"] for run in runs):
        mine = [run for run in runs if run["case"] == name]
        row = {"family": mine[0]["family"], "runs": len(mine), "exact": sum(run["exact"] for run in mine),
               "test_r2_0.999": sum(run["test_r2"] >= .999 for run in mine), "test_r2_0.99": sum(run["test_r2"] >= .99 for run in mine),
               "mean_test_digits": float(np.mean([run["test_digits"] for run in mine])),
               "mean_anytime_digits": float(np.mean([run["anytime_digits"] for run in mine])),
               "median_generation0_r2": float(np.median([run["generation0_best_r2"] for run in mine])),
               "mean_seconds": float(np.mean([run["elapsed_seconds"] for run in mine])),
               "mean_nodes": float(np.mean([run["nodes"] for run in mine]))}
        for label in THRESHOLDS:
            hits = [run["reached"][label]["generation"] for run in mine if run["reached"][label]]
            row[f"reached_{label}"] = len(hits)
            row[f"median_generation_{label}"] = float(np.median(hits)) if hits else None
        table[name] = row
    return table


def compare(runs, reference):
    """Paired (case, seed) differences against a reference JSON, with bootstrap 95% intervals."""
    ref = {(run["case"], run["seed"]): run for run in reference["runs"]}
    pairs = [(run, ref[(run["case"], run["seed"])]) for run in runs if (run["case"], run["seed"]) in ref]
    if not pairs:
        return {}
    resample = np.random.default_rng(0)

    def paired(values):
        values = np.asarray(values, float)
        boot = values[resample.integers(0, len(values), size=(4000, len(values)))].mean(axis=1)
        return {"n": int(len(values)), "mean": float(values.mean()),
                "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                "wins": int(np.sum(values > 1e-9)), "losses": int(np.sum(values < -1e-9))}

    def block(group):
        return {"test_digits": paired([a["test_digits"] - b["test_digits"] for a, b in group]),
                "anytime_digits": paired([a["anytime_digits"] - b["anytime_digits"] for a, b in group]),
                "exact": paired([float(a["exact"]) - float(b["exact"]) for a, b in group]),
                "exact_counts": [sum(a["exact"] for a, _ in group), sum(b["exact"] for _, b in group)]}

    result = {"overall": block(pairs)}
    for family in dict.fromkeys(a["family"] for a, _ in pairs):
        result[family] = block([(a, b) for a, b in pairs if a["family"] == family])
    return result


def print_summary(table):
    print(f"\n{'case':<18}{'family':<14}{'runs':>5}{'exact':>6}{'.999':>6}{'.99':>5}{'digits':>8}{'anytime':>8}"
          f"{'gen0 R2':>9}{'gen .999':>9}{'gen ex':>8}{'sec':>6}")
    for name, row in table.items():
        g999 = "-" if row["median_generation_0.999"] is None else f"{row['median_generation_0.999']:.0f}"
        gex = "-" if row["median_generation_exact"] is None else f"{row['median_generation_exact']:.0f}"
        print(f"{name:<18}{row['family']:<14}{row['runs']:>5}{row['exact']:>6}{row['test_r2_0.999']:>6}{row['test_r2_0.99']:>5}"
              f"{row['mean_test_digits']:>8.2f}{row['mean_anytime_digits']:>8.2f}{row['median_generation0_r2']:>9.3f}"
              f"{g999:>9}{gex:>8}{row['mean_seconds']:>6.0f}")
    total = sum(row["runs"] for row in table.values())
    print(f"total exact {sum(row['exact'] for row in table.values())}/{total}; mean digits "
          f"{np.mean([row['mean_test_digits'] for row in table.values()]):.2f}; mean anytime "
          f"{np.mean([row['mean_anytime_digits'] for row in table.values()]):.2f}")
    print(f"(exact = held-out 1-R2 <= {EXACT_GAP:g}; digits = -log10(1-R2) capped at {DIGITS_CAP:g}; anytime = mean training digits "
          "over generations; gen .999 / gen ex = median generation training R2 got there, over runs that did)")


def print_comparison(result):
    print(f"\nPaired against the reference on the same case and seed (this run minus reference, bootstrap 95% interval):")
    print(f"{'group':<14}{'pairs':>6}{'exact':>10}{'digits diff':>13}{'interval':>20}{'anytime diff':>14}{'interval':>20}")
    for group, info in result.items():
        d, a = info["test_digits"], info["anytime_digits"]
        print(f"{group:<14}{d['n']:>6}{info['exact_counts'][0]:>5}/{info['exact_counts'][1]:<4}{d['mean']:>+13.3f}"
              f"{'[%+.3f, %+.3f]' % tuple(d['ci95']):>20}{a['mean']:>+14.3f}{'[%+.3f, %+.3f]' % tuple(a['ci95']):>20}")
    print("(exact = this run / reference; an interval that excludes 0 is a difference beyond seed noise)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default="main", help="'main', 'reserve', 'all', or comma-separated names")
    parser.add_argument("--reserve", action="store_true", help="Same as --cases reserve (run once, at the end of feature work)")
    parser.add_argument("--seed", type=int, default=FIRST_SEED, help="First seed (frozen default)")
    parser.add_argument("--seed-count", type=int, default=SEED_COUNT)
    parser.add_argument("--quick", action="store_true", help="3 seeds instead of 10")
    parser.add_argument("--flags", default="", help="Extra afpo CLI flags for the configuration under test, as one string")
    parser.add_argument("--compare", type=Path, help="Reference JSON (e.g. bench_diag_baseline.json) to pair against")
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1, help="Runs in parallel processes (results identical; timings noisier)")
    parser.add_argument("--keep-logs", metavar="DIR")
    parser.add_argument("--output", type=Path, default=Path("bench_diag.json"))
    args = parser.parse_args()
    if args.reserve:
        args.cases = "reserve"
    if args.quick:
        args.seed_count = 3
    names = {"main": MAIN_CASES, "reserve": RESERVE_CASES, "all": list(CASES)}.get(args.cases) or args.cases.split(",")
    unknown = [name for name in names if name not in CASES]
    if unknown:
        parser.error(f"Unknown case(s): {', '.join(unknown)}")
    if (args.generations, args.population) != (60, 96):
        print("Note: generations/population differ from the frozen budget (60, 96); results are not comparable to the baseline.")

    data = {name: build(name) for name in names}
    seeds = [args.seed + i for i in range(args.seed_count)]
    jobs = [(name, data[name], seed) for name in names for seed in seeds]
    runs, started = [], time.perf_counter()

    def report(run):
        runs.append(run)
        print(f"[{len(runs)}/{len(jobs)}] {run['case']:<18} seed {run['seed']}: test R2={run['test_r2']:.10f} "
              f"gen0 R2={run['generation0_best_r2']:.3f} anytime={run['anytime_digits']:.2f} {run['elapsed_seconds']:.0f}s  "
              f"{run['equation']}", flush=True)

    if args.jobs <= 1:
        for job in jobs:
            report(run_job(*job, args))
    else:
        with concurrent.futures.ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            for future in concurrent.futures.as_completed([pool.submit(run_job, *job, args) for job in jobs]):
                report(future.result())
    order = {(job[0], job[2]): i for i, job in enumerate(jobs)}
    runs.sort(key=lambda run: order[(run["case"], run["seed"])])
    table = summarize(runs)
    print_summary(table)
    payload = {"bench_version": BENCH_VERSION,
               "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "thresholds": THRESHOLDS, "exact_gap": EXACT_GAP, "digits_cap": DIGITS_CAP,
               "cases": {name: {"family": CASES[name][0], "formula": CASES[name][1], "operator_groups": CASES[name][3],
                                "nodes": CASES[name][4], "depth": CASES[name][5], "reserve": CASES[name][6],
                                "train_rows": int(len(data[name][0])), "test_rows": int(len(data[name][2])),
                                "data_sha256": data_hash(data[name])} for name in names},
               "seeds": seeds, "wall_seconds": round(time.perf_counter() - started, 1), "summary": table, "runs": runs}
    if args.compare:
        reference = json.loads(args.compare.read_text())
        if reference.get("bench_version") != BENCH_VERSION:
            print(f"Warning: reference is bench version {reference.get('bench_version')}, this is {BENCH_VERSION}.")
        payload["comparison"] = compare(runs, reference)
        if payload["comparison"]:
            print_comparison(payload["comparison"])
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=True))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
