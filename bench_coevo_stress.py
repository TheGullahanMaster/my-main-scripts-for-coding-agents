"""Stress bench for evo13's predator–prey co-evolution batch resolution.

Drives the REAL engine (evolve_afpo, single process) through the same per-chunk
loop the main script runs — co-evolved batch selection → 50-generation chunk on
that fixed batch → full-data rescore → global-HoF admission — on a bigger
regression dataset and a bigger classification dataset, with the co-evolution
subset forced to the extremes the user reported:

    COEVO_CASE_SUBSET = 1     (a single adversarial row per chunk)
    COEVO_CASE_SUBSET = 256   (6% of a 4096-row complex dataset)

At these settings the chunk's in-sample fitness can no longer resolve which
host is genuinely better (with >=2 rows the two affine parameters alone tie
everyone at ~0 loss), so evolution degenerates to a random walk and the final
full-data R² / accuracy collapses vs a full-data control at the same budget.

Usage:
    python bench_coevo_stress.py [reg|cls|all] [seeds] [generations] [legacy|new]

`legacy` forces the pre-resolution-governor behaviour (no eval floor, no
adaptive batch growth, no stratified anchor) when run against code that has
those knobs; on older code both labels run identically.
"""
import os, sys, time, random
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import numpy as np
import evo13
from evo13 import (
    CGPEquation, Individual, HallOfFame,
    generate_seeds_v5, random_cgp, evolve_afpo, set_ops_mode,
    BINARY_OPS_EVAL, UNARY_OPS_EVAL,
)

CHUNK = 50          # generations per co-evolution batch (mirrors MIGRATION_FREQ)
POP   = 70


def _set_ops(ops_list, if_else=False):
    evo13.ALLOWED_OPS = list(ops_list)
    evo13.IF_ELSE_ENABLED = if_else
    CGPEquation.OPS_BINARY = [op for op in BINARY_OPS_EVAL if op in evo13.ALLOWED_OPS]
    CGPEquation.OPS_UNARY = [op for op in UNARY_OPS_EVAL if op in evo13.ALLOWED_OPS]
    CGPEquation.OPS_TERNARY = ['if_else'] if (if_else and 'if_else' in evo13.ALLOWED_OPS) else []
    CGPEquation.OPS_ALL = (
        CGPEquation.OPS_BINARY + CGPEquation.OPS_UNARY + CGPEquation.OPS_TERNARY
        + (['const'] if 'const' in evo13.ALLOWED_OPS else [])
    )
    CGPEquation.OPS_BINARY_SET = set(CGPEquation.OPS_BINARY)
    CGPEquation.OPS_UNARY_SET = set(CGPEquation.OPS_UNARY)
    CGPEquation.OPS_TERNARY_SET = set(CGPEquation.OPS_TERNARY)


def _configure(seed, legacy):
    np.random.seed(seed)
    random.seed(seed)
    set_ops_mode(True)
    evo13.INTERVAL_MODE = False
    evo13.USE_SEEDS = True
    _set_ops(['+', '-', '*', '/', 'sin', 'cos', 'sqrt', 'abs', 'square',
              'tanh', 'log', 'const'])
    evo13.CGP_NODES = 32
    evo13.CGP_MUT_RATE = 3
    evo13.PARSIMONY_STRENGTH = 0.005
    evo13.AFFINE_SCALING_ENABLED = True
    evo13._INIT_PHASE = False
    evo13.SOBOLEV_ENABLED = False
    evo13.FEATURE_PRIORS = None
    evo13.INSTANCE_REWEIGHT_ENABLED = False
    evo13._DIFFICULTY_ACTIVE = None
    if hasattr(evo13, "ADAPTIVE_COSTS_ENABLED"):
        evo13.ADAPTIVE_COSTS_ENABLED = False
    # ---- co-evolution stress configuration ----
    evo13.COEVOLUTION_ENABLED = True
    evo13._COEVO_RUNTIME = None
    if legacy:
        # Reproduce the pre-resolution behaviour on new code (no-ops on old code).
        if hasattr(evo13, "COEVO_MIN_EVAL_ROWS"):  evo13.COEVO_MIN_EVAL_ROWS = 0
        if hasattr(evo13, "COEVO_RES_MAX_FACTOR"): evo13.COEVO_RES_MAX_FACTOR = 1.0
        if hasattr(evo13, "COEVO_STRATIFY"):       evo13.COEVO_STRATIFY = False


def _data_reg(seed, n=8192):
    """Bigger + complex: 8 variables, multiplicative/trig/log interplay."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-3.0, 3.0, size=(n, 8))
    y = (X[:, 0] * X[:, 1]
         + X[:, 2] * np.sin(2.0 * X[:, 3])
         + np.log1p(np.abs(X[:, 4] * X[:, 5])) * X[:, 6]
         - 0.8 * X[:, 7] ** 2
         + 0.5 * X[:, 0] * X[:, 4])
    return X, y


def _data_cls(seed, n=8192):
    """Bigger + complex boundary, ~20/80 imbalance (minority repr matters)."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-3.0, 3.0, size=(n, 6))
    score = (np.sin(2.0 * X[:, 0]) + 0.6 * X[:, 1] * X[:, 2]
             - 0.4 * X[:, 3] + 0.3 * X[:, 4] * X[:, 0])
    y = (score > 1.1).astype(np.float64)
    return X, y


def run_one(task, subset, seed, generations, legacy):
    _configure(seed, legacy)
    if task == "reg":
        X, y = _data_reg(seed)
        tc = 5
    else:
        X, y = _data_cls(seed)
        tc = 6
    n, nf = X.shape
    feat_names = [f"x{i}" for i in range(nf)]
    evo13.COEVO_CASE_SUBSET = subset if subset > 0 else n + 1   # 0 → full-data control
    evo13._COEVO_RUNTIME = None
    evo13._set_data_scale_hint(X, y)
    Y2 = y.reshape(-1, 1)

    pop = list(generate_seeds_v5(nf, feat_names))
    while len(pop) < POP:
        pop.append(Individual(random_cgp(nf, evo13.CGP_NODES, feat_names)))
    for ind in pop:
        ind.calculate_fitness(X, y, tc)
    hof = HallOfFame(out_type=tc)
    for ind in pop:
        hof.update(ind)

    rescore = getattr(evo13, "_coevo_admission_rescore", None)
    stag, done = 0, 0
    while done < generations:
        batch_idx = evo13._select_batch_indices(X, Y2, [hof], [tc], 0)
        X_b = X[batch_idx] if batch_idx is not None else X
        y_b = y[batch_idx] if batch_idx is not None else y
        # elitist HoF injection (mirrors the main loop)
        hof_best = hof.get_best_overall()
        if hof_best is not None and pop:
            pop_best_loss = min(i2.loss for i2 in pop)
            if hof_best.loss < pop_best_loss - 1e-9:
                worst = max(pop, key=lambda i2: i2.fitness)
                pop.remove(worst)
                import copy as _copy
                elite = _copy.deepcopy(hof_best)
                elite.age = 0
                pop.append(elite)
        local_hof = HallOfFame(out_type=tc)
        ngen = min(CHUNK, generations - done)
        pop, stag = evolve_afpo(pop, X_b, y_b, tc, nf, feat_names,
                                target_size=POP, n_generations=ngen, hof=local_hof,
                                stag_counter=stag, ext_patience=300)
        for c, ind in local_hof.best_by_complexity.items():
            if rescore is not None:
                rescore(ind, X, y, tc, None)
            else:
                evo13._rescore_individual_full_data(ind, X, y, tc, None)
            hof.update(ind)
        done += ngen

    best = hof.get_best_overall()
    if best is not None:
        best.affine_fitted = False
        best.calculate_fitness(X, y, tc, use_cache=False)
    r2 = best.r2 if best is not None else -1.0
    acc = best.accuracy if best is not None else 0.0
    loss = best.loss if best is not None else float('inf')
    # co-evo runtime diagnostics (if it ran)
    rt = evo13._COEVO_RUNTIME
    diag = ""
    if rt is not None:
        eff = getattr(rt, "_effective_rows", None)
        gap = getattr(rt, "gap_ema", None)
        if eff is not None and gap is not None:
            diag = f"  [rows={eff()} gap={gap:.2f}]"
    evo13.COEVOLUTION_ENABLED = False
    evo13._COEVO_RUNTIME = None
    return loss, r2, acc, diag


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1]
    gens = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    mode = sys.argv[4] if len(sys.argv) > 4 else "new"
    legacy = (mode == "legacy")
    tasks = ["reg", "cls"] if which == "all" else [which]
    subsets = [int(s) for s in os.environ.get("SUBSETS", "0,1,256").split(",")]
    # 0 = full-data control
    for task in tasks:
        metric_name = "R²" if task == "reg" else "acc"
        for subset in subsets:
            vals = []
            t0 = time.time()
            for s in seeds:
                loss, r2, acc, diag = run_one(task, subset, s, gens, legacy)
                m = r2 if task == "reg" else acc
                vals.append(m)
                lab = "full" if subset == 0 else f"sub={subset}"
                print(f"  [{task} {lab} {mode}] seed={s}  loss={loss:.4f}  "
                      f"{metric_name}={m:.4f}{diag}", flush=True)
            lab = "full" if subset == 0 else f"sub={subset}"
            print(f"== {task} {lab} ({mode}): mean {metric_name}="
                  f"{np.mean(vals):.4f}  best={np.max(vals):.4f}  "
                  f"({time.time()-t0:.1f}s)\n", flush=True)


if __name__ == "__main__":
    main()
