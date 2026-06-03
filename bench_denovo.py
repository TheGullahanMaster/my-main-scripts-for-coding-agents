"""
Benchmark harness for *de-novo* discovery — the scenarios where seeds do NOT
help AFPO:

  1. bitwise-only op set on a bit-twiddling target
  2. seeds turned OFF on a regression target
  3. a regression target that matches no built-in seed template

Drives evolve_afpo directly (no interactive prompts), mirroring
test_conditional.run_evolution.  Prints best R²/loss so we can compare a
baseline against a candidate change with a fixed RNG seed budget.
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


def _configure(seed, use_seeds, ops_list, if_else=False, cgp_nodes=32,
               bitwise_mode='int'):
    np.random.seed(seed)
    random.seed(seed)
    set_ops_mode(True)
    evo13.INTERVAL_MODE = False
    evo13.USE_SEEDS = use_seeds
    evo13.BITWISE_MODE = bitwise_mode
    _set_ops(ops_list, if_else=if_else)
    evo13.CGP_NODES = cgp_nodes
    evo13.CGP_MUT_RATE = 3
    evo13.PARSIMONY_STRENGTH = 0.005
    evo13.AFFINE_SCALING_ENABLED = True
    evo13._INIT_PHASE = False
    evo13.SOBOLEV_ENABLED = False
    evo13.FEATURE_PRIORS = None


# Fraction of the *initial* random population shaped by the data prior.  Default
# 0.0 matches production AFPO (which does not bias the initial pool — validated
# as the most robust setting; init biasing over-commits the start and regresses
# additive/periodic targets).  Tunable via env for experimentation only.
INIT_PRIOR_FRAC = float(os.environ.get("INIT_PRIOR_FRAC", "0.0"))


def run(X, y, *, generations, pop_size, feat_names, type_code, seed):
    n_features = X.shape[1]
    evo13._set_data_scale_hint(X, y)
    # Data-driven operator prior shapes a *fraction* of the initial random
    # population (the rest stays fully random to preserve exploration
    # diversity).  evolve_afpo recomputes the same prior internally for
    # mutation/immigrants.  Guarded so the bench also runs against baseline
    # code that predates these helpers.
    _have_prior = hasattr(evo13, "compute_data_operator_prior")
    _dp = evo13.compute_data_operator_prior(X, y, type_code) if _have_prior else None
    pop = list(generate_seeds_v5(n_features, feat_names))
    while len(pop) < pop_size:
        if _dp is not None and random.random() < INIT_PRIOR_FRAC:
            pop.append(Individual(random_cgp(n_features, evo13.CGP_NODES,
                                             feat_names, op_prior=_dp)))
        else:
            pop.append(Individual(random_cgp(n_features, evo13.CGP_NODES,
                                             feat_names)))
    for ind in pop:
        ind.calculate_fitness(X, y, type_code)
    hof = HallOfFame()
    for ind in pop:
        hof.update(ind)
    stag = 0
    chunk = 50
    done = 0
    n_seeds = len([p for p in pop if p is not None]) - 0
    while done < generations:
        n = min(chunk, generations - done)
        pop, stag = evolve_afpo(pop, X, y, type_code, n_features, feat_names,
                                target_size=pop_size, n_generations=n, hof=hof,
                                stag_counter=stag, ext_patience=300)
        for ind in pop:
            hof.update(ind)
        done += n
    best = hof.get_best_overall()
    if best is not None:
        best.affine_fitted = False
        best.calculate_fitness(X, y, type_code)
    return best


def case_bitwise(seed, generations=1500, pop_size=80):
    """Target: y = (x & 12) ^ (x >> 1)  — pure bit twiddling, integer inputs."""
    _configure(seed, use_seeds=True,
               ops_list=['bitwise_and', 'bitwise_or', 'bitwise_xor',
                         'lshift', 'rshift', 'bitwise_not', 'const'],
               bitwise_mode='int')
    rng = np.random.RandomState(seed)
    x = rng.randint(0, 256, size=(400, 1)).astype(np.float64)
    y = (np.bitwise_and(x[:, 0].astype(np.int64), 12)
         ^ (x[:, 0].astype(np.int64) >> 1)).astype(np.float64)
    best = run(x, y, generations=generations, pop_size=pop_size,
               feat_names=["x"], type_code=5, seed=seed)
    return best


def case_seedsoff_poly(seed, generations=1500, pop_size=80):
    """Target: y = 0.5*x0^2 - 1.3*x1 + sin(x2), seeds OFF, full-ish op set."""
    _configure(seed, use_seeds=False,
               ops_list=['+', '-', '*', '/', 'pow', 'exp', 'log', 'sqrt',
                         'sin', 'cos', 'square', 'const'])
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2.0, 2.0, size=(500, 3))
    y = 0.5 * X[:, 0] ** 2 - 1.3 * X[:, 1] + np.sin(X[:, 2])
    best = run(X, y, generations=generations, pop_size=pop_size,
               feat_names=["x0", "x1", "x2"], type_code=5, seed=seed)
    return best


def case_seedmismatch(seed, generations=1500, pop_size=80):
    """Target: y = |x0| * sign(x1) + 0.3*x0*x1 — odd shape, seeds ON but unlikely
    to match.  Uses a trig/abs op set."""
    _configure(seed, use_seeds=True,
               ops_list=['+', '-', '*', '/', 'abs', 'sign', 'tanh', 'sin',
                         'square', 'const'])
    rng = np.random.RandomState(seed)
    X = rng.uniform(-3.0, 3.0, size=(500, 2))
    y = np.abs(X[:, 0]) * np.sign(X[:, 1]) + 0.3 * X[:, 0] * X[:, 1]
    best = run(X, y, generations=generations, pop_size=pop_size,
               feat_names=["x0", "x1"], type_code=5, seed=seed)
    return best


def case_rational(seed, generations=1500, pop_size=80):
    """Target: y = x0 / (1 + x1^2) — rational, seeds OFF."""
    _configure(seed, use_seeds=False,
               ops_list=['+', '-', '*', '/', 'square', 'sqrt', 'abs', 'const'])
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2.5, 2.5, size=(500, 2))
    y = X[:, 0] / (1.0 + X[:, 1] ** 2)
    best = run(X, y, generations=generations, pop_size=pop_size,
               feat_names=["x0", "x1"], type_code=5, seed=seed)
    return best


def case_trig(seed, generations=1500, pop_size=80):
    """Target: y = sin(2*x0) + 0.5*cos(3*x1) — periodic, seeds ON but trig set."""
    _configure(seed, use_seeds=True,
               ops_list=['+', '-', '*', '/', 'sin', 'cos', 'tanh', 'square',
                         'const'])
    rng = np.random.RandomState(seed)
    X = rng.uniform(-3.0, 3.0, size=(500, 2))
    y = np.sin(2.0 * X[:, 0]) + 0.5 * np.cos(3.0 * X[:, 1])
    best = run(X, y, generations=generations, pop_size=pop_size,
               feat_names=["x0", "x1"], type_code=5, seed=seed)
    return best


def case_highmag_affineoff(seed, generations=1500, pop_size=80):
    """Target: y = 5000*x0^2 + 1500*x1 - 800, AFFINE SCALING OFF, seeds OFF.

    The hardest scale regime: with the affine wrapper disabled the CGP must
    grow the large multiplicative/offset constants itself (a=1, b=0 are locked),
    and with seeds off it cannot transcribe them from an OLS fit.  Exercises
    whether evolve_afpo dials in big constants via its constant-optimiser before
    structurally-correct-but-wrong-scale individuals are Pareto-evicted."""
    _configure(seed, use_seeds=False,
               ops_list=['+', '-', '*', '/', 'square', 'sqrt', 'const'])
    evo13.AFFINE_SCALING_ENABLED = False
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2.0, 2.0, size=(500, 2))
    y = 5000.0 * X[:, 0] ** 2 + 1500.0 * X[:, 1] - 800.0
    best = run(X, y, generations=generations, pop_size=pop_size,
               feat_names=["x0", "x1"], type_code=5, seed=seed)
    return best


def case_seedsoff_periodic(seed, generations=1500, pop_size=80):
    """Target: y = sin(2*x0) + 0.5*cos(3*x1), seeds OFF.

    The periodic regime where seeds cannot help: with seeding disabled the
    data-driven operator prior is the *only* thing that can steer the search
    toward the trig family, so this case directly exercises the frequency-swept
    periodic detection in compute_data_operator_prior.  A single-frequency
    sin(x) probe is near-orthogonal to a sin(2x)/cos(3x) target and used to
    leave the prior with no trig signal at all."""
    _configure(seed, use_seeds=False,
               ops_list=['+', '-', '*', '/', 'sin', 'cos', 'tanh', 'square',
                         'const'])
    rng = np.random.RandomState(seed)
    X = rng.uniform(-3.0, 3.0, size=(500, 2))
    y = np.sin(2.0 * X[:, 0]) + 0.5 * np.cos(3.0 * X[:, 1])
    best = run(X, y, generations=generations, pop_size=pop_size,
               feat_names=["x0", "x1"], type_code=5, seed=seed)
    return best


def case_discontinuous(seed, generations=1500, pop_size=80):
    """Target: y = sign(sin(3*x0)) + 0.5*floor(x1) + 0.3*x0 — heavily
    discontinuous (square wave + staircase + ramp), seeds ON.

    A rugged/cliffy loss landscape: most parameter polishing cannot cross the
    flat steps, so progress depends on the discontinuity-escape / heavy-tail
    structural machinery rather than smooth gradient-like refinement."""
    _configure(seed, use_seeds=True,
               ops_list=['+', '-', '*', '/', 'sin', 'cos', 'sign', 'floor',
                         'round', 'abs', 'const'])
    rng = np.random.RandomState(seed)
    X = rng.uniform(-3.0, 3.0, size=(500, 2))
    y = (np.sign(np.sin(3.0 * X[:, 0]))
         + 0.5 * np.floor(X[:, 1])
         + 0.3 * X[:, 0])
    best = run(X, y, generations=generations, pop_size=pop_size,
               feat_names=["x0", "x1"], type_code=5, seed=seed)
    return best


CASES = {
    "bitwise": case_bitwise,
    "seedsoff_poly": case_seedsoff_poly,
    "seedmismatch": case_seedmismatch,
    "rational": case_rational,
    "trig": case_trig,
    "highmag_affineoff": case_highmag_affineoff,
    "discontinuous": case_discontinuous,
    "seedsoff_periodic": case_seedsoff_periodic,
}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
    gens = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
    names = list(CASES) if which == "all" else [which]
    for name in names:
        r2s, losses = [], []
        t0 = time.time()
        for s in seeds:
            best = CASES[name](s, generations=gens)
            r2 = best.r2 if best is not None else -1.0
            loss = best.loss if best is not None else float('inf')
            r2s.append(r2); losses.append(loss)
            print(f"  [{name}] seed={s}  r2={r2:.4f}  loss={loss:.5f}")
        print(f"== {name}: mean r2={np.mean(r2s):.4f}  median r2={np.median(r2s):.4f}  "
              f"best r2={np.max(r2s):.4f}  ({time.time()-t0:.1f}s)\n")


if __name__ == "__main__":
    main()
