"""
Programmatic test harness for evo13.py — focuses on conditional / threshold
discovery problems.

Bypasses the interactive train_mode() prompts by setting evo13's globals
directly, then driving a small AFPO-style evolution loop via the public
helpers (generate_seeds_v5, evolve_afpo, ...).

Each test case:
  1. Generates synthetic data (X, y) for a piecewise / threshold function.
  2. Configures evo13 with the Full op set + IF/ELSE branching.
  3. Runs N generations of AFPO evolution.
  4. Reports best R² and best discovered expression.

Designed to be FAST (small pop, short gen budget) so that we can iterate on
fixes within seconds-to-minutes per test rather than hours.
"""
import numpy as np
import random
import time
import sys
import os

# Quiet pandas/sklearn output
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import evo13
from evo13 import (
    CGPEquation, Individual, HallOfFame,
    generate_seeds_v5, random_cgp,
    evolve_afpo,
    set_ops_mode,
    BINARY_OPS_EVAL, UNARY_OPS_EVAL,
)


def configure_evo13(
    *,
    ops_mode: str = "safe",
    ops_set: str = "full",
    if_else: bool = True,
    diff_branching: bool = True,
    cgp_nodes: int = 32,
    pop_size: int = 100,
    seed: int = 0,
):
    """Set up evo13 globals for a programmatic test run."""
    np.random.seed(seed)
    random.seed(seed)

    # Operator safety mode
    set_ops_mode(ops_mode == "safe")
    evo13.INTERVAL_MODE = False

    # Choose operator set
    if ops_set == "full":
        # Same as Full preset (all ops)
        evo13.ALLOWED_OPS = list(evo13.ALL_OP_DESCRIPTIONS.keys())
    elif ops_set == "ext_sci":
        evo13.ALLOWED_OPS = [
            '+', '-', '*', '/', 'pow', 'exp', 'log', 'sqrt',
            'abs', 'square', 'cube', 'const',
            'gt', 'lt', 'gte', 'lte', 'if_else',
            'relu', 'tanh', 'sigmoid', 'neg', 'min', 'max',
        ]
    else:
        evo13.ALLOWED_OPS = ops_set  # custom list

    # IF/ELSE flag
    evo13.IF_ELSE_ENABLED = if_else
    evo13.DIFFERENTIABLE_BRANCHING = diff_branching
    evo13.DIFF_BRANCH_STEEPNESS = 20.0

    # Rebuild CGPEquation op tables
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

    evo13.CGP_NODES = cgp_nodes
    evo13.CGP_MUT_RATE = 3
    evo13.PARSIMONY_STRENGTH = 0.005
    evo13.AFFINE_SCALING_ENABLED = True
    evo13._INIT_PHASE = False
    evo13.SOBOLEV_ENABLED = False

    return pop_size


def run_evolution(X, y, *, generations=2000, pop_size=60, n_features=None,
                  feat_names=None, type_code=5, verbose_every=200):
    """Run a programmatic AFPO evolution loop on the given (X, y)."""
    if n_features is None:
        n_features = X.shape[1]
    if feat_names is None:
        feat_names = [f"x{i}" for i in range(n_features)]

    # Build initial population.  Mirroring evo13's real flow: keep ALL hand-
    # crafted seeds (don't truncate — that silently drops late-added piecewise/
    # ELU/clip patterns) and pad with random individuals only if seed count is
    # below the configured floor.  evo13's train_mode does the same.
    pop = list(generate_seeds_v5(n_features, feat_names))
    while len(pop) < pop_size:
        pop.append(Individual(random_cgp(n_features, evo13.CGP_NODES, feat_names)))

    # Initial fitness
    for ind in pop:
        ind.calculate_fitness(X, y, type_code)

    hof = HallOfFame()
    for ind in pop:
        hof.update(ind)

    stag_counter = 0
    chunk_size = 50  # generations per chunk
    total_gens = 0
    start = time.time()

    while total_gens < generations:
        n_gens = min(chunk_size, generations - total_gens)
        pop, stag_counter = evolve_afpo(
            pop, X, y, type_code,
            n_features, feat_names,
            target_size=pop_size,
            n_generations=n_gens,
            hof=hof,
            stag_counter=stag_counter,
            ext_patience=300,
        )
        for ind in pop:
            hof.update(ind)
        total_gens += n_gens
        if total_gens % verbose_every == 0 or total_gens >= generations:
            best = hof.get_best_overall()
            if best is not None:
                # Re-evaluate on full data for clean R² report
                best.affine_fitted = False
                best.calculate_fitness(X, y, type_code)
                elapsed = time.time() - start
                print(
                    f"  gen {total_gens:5d}  loss={best.loss:.4f}  "
                    f"r2={best.r2:.4f}  comp={best.complexity:.0f}  "
                    f"({elapsed:.1f}s)"
                )

    best = hof.get_best_overall()
    return best, hof


def test_diode(generations=1500, seed=0):
    """V_out = V_in if V_in > 0.7 else 0  (exact threshold gate)."""
    print("\n" + "=" * 70)
    print("TEST: DIODE  →  V_out = V_in if V_in > 0.7 else 0")
    print("=" * 70)
    pop_size = configure_evo13(seed=seed)

    rng = np.random.RandomState(seed)
    n = 600
    V_in = rng.uniform(-1.5, 2.5, size=(n, 1)).astype(np.float64)
    V_out = np.where(V_in[:, 0] > 0.7, V_in[:, 0], 0.0)

    best, hof = run_evolution(V_in, V_out, generations=generations,
                              pop_size=pop_size,
                              feat_names=["V_in"])
    if best is not None:
        eq = str(best.tree)
        print(f"\nBest: r2={best.r2:.4f}  loss={best.loss:.4f}  "
              f"complexity={best.complexity:.0f}")
        print(f"Equation: {eq}")

        # Re-evaluate on a held-out grid
        test_V = np.linspace(-1.5, 2.5, 200).reshape(-1, 1)
        target = np.where(test_V[:, 0] > 0.7, test_V[:, 0], 0.0)
        preds = best._predict_with_boosts(test_V)
        ss_res = float(np.sum((preds - target) ** 2))
        ss_tot = float(np.sum((target - target.mean()) ** 2))
        test_r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        print(f"Test R² (held-out grid): {test_r2:.4f}")
        return test_r2
    return -1.0


def test_elu(generations=2000, seed=0):
    """ELU(x) = x if x > 0 else (exp(x) - 1)  (alpha = 1)."""
    print("\n" + "=" * 70)
    print("TEST: ELU      →  x if x > 0 else (exp(x) - 1)")
    print("=" * 70)
    pop_size = configure_evo13(seed=seed)

    rng = np.random.RandomState(seed)
    n = 600
    x = rng.uniform(-3.0, 3.0, size=(n, 1)).astype(np.float64)
    y = np.where(x[:, 0] > 0, x[:, 0], np.exp(x[:, 0]) - 1.0)

    best, hof = run_evolution(x, y, generations=generations,
                              pop_size=pop_size,
                              feat_names=["x"])
    if best is not None:
        eq = str(best.tree)
        print(f"\nBest: r2={best.r2:.4f}  loss={best.loss:.4f}  "
              f"complexity={best.complexity:.0f}")
        print(f"Equation: {eq}")

        test_x = np.linspace(-3.0, 3.0, 200).reshape(-1, 1)
        target = np.where(test_x[:, 0] > 0, test_x[:, 0],
                          np.exp(test_x[:, 0]) - 1.0)
        preds = best._predict_with_boosts(test_x)
        ss_res = float(np.sum((preds - target) ** 2))
        ss_tot = float(np.sum((target - target.mean()) ** 2))
        test_r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        print(f"Test R² (held-out grid): {test_r2:.4f}")
        return test_r2
    return -1.0


def test_quadratic_disc(generations=2500, seed=0):
    """Discriminant + root1: D = b² - 4ac;  root1 = (-b + sqrt(D))/(2a) if D >= 0 else 0.

    Inputs: a, b, c.
    Output: root1 (or 0 when D < 0).
    """
    print("\n" + "=" * 70)
    print("TEST: QUADRATIC ROOT1 →  (-b+sqrt(b²-4ac))/(2a) if D≥0 else 0")
    print("=" * 70)
    pop_size = configure_evo13(seed=seed)

    rng = np.random.RandomState(seed)
    n = 800
    a = rng.uniform(0.2, 2.0, size=n)
    b = rng.uniform(-3.0, 3.0, size=n)
    c = rng.uniform(-2.0, 2.0, size=n)
    D = b * b - 4 * a * c
    root1 = np.where(D >= 0, (-b + np.sqrt(np.maximum(D, 0))) / (2 * a), 0.0)
    X = np.stack([a, b, c], axis=1)

    best, hof = run_evolution(X, root1, generations=generations,
                              pop_size=pop_size,
                              feat_names=["a", "b", "c"])
    if best is not None:
        eq = str(best.tree)
        print(f"\nBest: r2={best.r2:.4f}  loss={best.loss:.4f}  "
              f"complexity={best.complexity:.0f}")
        print(f"Equation: {eq}")
        return best.r2
    return -1.0


def test_golu(generations=2500, seed=0):
    """GoLU: x * exp(-exp(-x))   (Gompertz-gated linear unit)."""
    print("\n" + "=" * 70)
    print("TEST: GoLU    →  x * exp(-exp(-x))")
    print("=" * 70)
    pop_size = configure_evo13(seed=seed)
    rng = np.random.RandomState(seed)
    n = 600
    x = rng.uniform(-3.0, 3.0, size=(n, 1)).astype(np.float64)
    y = x[:, 0] * np.exp(-np.exp(-x[:, 0]))
    best, hof = run_evolution(x, y, generations=generations,
                              pop_size=pop_size, feat_names=["x"])
    if best is not None:
        print(f"\nBest: r2={best.r2:.4f}  loss={best.loss:.4f}  "
              f"complexity={best.complexity:.0f}")
        print(f"Equation: {best.tree}")
        return best.r2
    return -1.0


def test_gelu(generations=2500, seed=0):
    """GELU (approx): 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))."""
    print("\n" + "=" * 70)
    print("TEST: GELU    →  0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³)))")
    print("=" * 70)
    pop_size = configure_evo13(seed=seed)
    rng = np.random.RandomState(seed)
    n = 600
    x = rng.uniform(-3.0, 3.0, size=(n, 1)).astype(np.float64)
    y = 0.5 * x[:, 0] * (1 + np.tanh(np.sqrt(2 / np.pi) *
                                     (x[:, 0] + 0.044715 * x[:, 0]**3)))
    best, hof = run_evolution(x, y, generations=generations,
                              pop_size=pop_size, feat_names=["x"])
    if best is not None:
        print(f"\nBest: r2={best.r2:.4f}  loss={best.loss:.4f}  "
              f"complexity={best.complexity:.0f}")
        print(f"Equation: {best.tree}")
        return best.r2
    return -1.0


def test_damage_armor(generations=2500, seed=0):
    """Damage split: dmg attacks armor first, overflow hits hp.
    new_hp    = hp - max(0, dmg - armor)        # hp loses overflow only
    new_armor = max(0, armor - dmg)              # armor caps at zero
    Test the new_armor function (the simpler of the pair).
    """
    print("\n" + "=" * 70)
    print("TEST: ARMOR-AFTER-DAMAGE  →  max(0, armor - dmg)")
    print("=" * 70)
    pop_size = configure_evo13(seed=seed)
    rng = np.random.RandomState(seed)
    n = 800
    hp = rng.uniform(1, 100, size=n)
    armor = rng.uniform(0, 50, size=n)
    dmg = rng.uniform(0, 60, size=n)
    new_armor = np.maximum(0.0, armor - dmg)
    X = np.stack([hp, armor, dmg], axis=1)
    best, hof = run_evolution(X, new_armor, generations=generations,
                              pop_size=pop_size,
                              feat_names=["hp", "armor", "dmg"])
    if best is not None:
        print(f"\nBest: r2={best.r2:.4f}  loss={best.loss:.4f}  "
              f"complexity={best.complexity:.0f}")
        print(f"Equation: {best.tree}")
        return best.r2
    return -1.0


def test_damage_overflow(generations=2500, seed=0):
    """Damage overflow to hp: new_hp = hp - max(0, dmg - armor).
    Hp loses only the part of damage exceeding armor.
    """
    print("\n" + "=" * 70)
    print("TEST: HP-AFTER-OVERFLOW  →  hp - max(0, dmg - armor)")
    print("=" * 70)
    pop_size = configure_evo13(seed=seed)
    rng = np.random.RandomState(seed)
    n = 800
    hp = rng.uniform(20, 200, size=n)
    armor = rng.uniform(0, 50, size=n)
    dmg = rng.uniform(0, 60, size=n)
    new_hp = hp - np.maximum(0.0, dmg - armor)
    X = np.stack([hp, armor, dmg], axis=1)
    best, hof = run_evolution(X, new_hp, generations=generations,
                              pop_size=pop_size,
                              feat_names=["hp", "armor", "dmg"])
    if best is not None:
        print(f"\nBest: r2={best.r2:.4f}  loss={best.loss:.4f}  "
              f"complexity={best.complexity:.0f}")
        print(f"Equation: {best.tree}")
        return best.r2
    return -1.0


def test_reload(generations=2500, seed=0):
    """Reload: newAmmo = ammo + min(reserve, MAX_CLIP - ammo)  with MAX_CLIP=30."""
    print("\n" + "=" * 70)
    print("TEST: RELOAD  →  newAmmo = ammo + min(reserve, 30 - ammo)")
    print("=" * 70)
    pop_size = configure_evo13(seed=seed)
    rng = np.random.RandomState(seed)
    n = 800
    ammo = rng.randint(0, 31, size=n).astype(np.float64)
    reserve = rng.randint(0, 200, size=n).astype(np.float64)
    new_ammo = ammo + np.minimum(reserve, 30 - ammo)
    X = np.stack([ammo, reserve], axis=1)
    best, hof = run_evolution(X, new_ammo, generations=generations,
                              pop_size=pop_size, feat_names=["ammo", "reserve"])
    if best is not None:
        print(f"\nBest: r2={best.r2:.4f}  loss={best.loss:.4f}  "
              f"complexity={best.complexity:.0f}")
        print(f"Equation: {best.tree}")
        return best.r2
    return -1.0


def test_relu_step(generations=1200, seed=0):
    """Simple step: y = 1 if x > 0 else 0  (sanity check — should be VERY easy)."""
    print("\n" + "=" * 70)
    print("TEST: STEP    →  1 if x > 0 else 0")
    print("=" * 70)
    pop_size = configure_evo13(seed=seed)

    rng = np.random.RandomState(seed)
    n = 400
    x = rng.uniform(-2.0, 2.0, size=(n, 1)).astype(np.float64)
    y = np.where(x[:, 0] > 0, 1.0, 0.0)

    best, hof = run_evolution(x, y, generations=generations,
                              pop_size=pop_size,
                              feat_names=["x"])
    if best is not None:
        print(f"\nBest: r2={best.r2:.4f}  loss={best.loss:.4f}  "
              f"complexity={best.complexity:.0f}")
        print(f"Equation: {best.tree}")
        return best.r2
    return -1.0


def test_clip_relu(generations=1500, seed=0):
    """y = max(0, x - 0.7) — same as a Diode but RELU-able from continuous ops."""
    print("\n" + "=" * 70)
    print("TEST: CLIPPED-RELU  →  max(0, x - 0.7)")
    print("=" * 70)
    pop_size = configure_evo13(seed=seed)

    rng = np.random.RandomState(seed)
    n = 500
    x = rng.uniform(-1.5, 2.5, size=(n, 1)).astype(np.float64)
    y = np.maximum(0.0, x[:, 0] - 0.7)

    best, hof = run_evolution(x, y, generations=generations,
                              pop_size=pop_size,
                              feat_names=["x"])
    if best is not None:
        print(f"\nBest: r2={best.r2:.4f}  loss={best.loss:.4f}  "
              f"complexity={best.complexity:.0f}")
        print(f"Equation: {best.tree}")
        return best.r2
    return -1.0


if __name__ == "__main__":
    args = set(sys.argv[1:])
    print("evo13 conditional-discovery test harness")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  evo13 module: {evo13.__file__}")

    results = {}
    if not args or "diode" in args:
        results["diode"] = test_diode(generations=int(os.environ.get("DIODE_GENS", 1500)))
    if not args or "step" in args:
        results["step"] = test_relu_step(generations=int(os.environ.get("STEP_GENS", 800)))
    if not args or "clip_relu" in args:
        results["clip_relu"] = test_clip_relu(generations=int(os.environ.get("CLIP_GENS", 1200)))
    if not args or "elu" in args:
        results["elu"] = test_elu(generations=int(os.environ.get("ELU_GENS", 2500)))
    if "quadratic" in args:
        results["quadratic"] = test_quadratic_disc(generations=int(os.environ.get("QUAD_GENS", 3000)))
    if "golu" in args:
        results["golu"] = test_golu(generations=int(os.environ.get("GOLU_GENS", 3000)))
    if "gelu" in args:
        results["gelu"] = test_gelu(generations=int(os.environ.get("GELU_GENS", 3000)))
    if "reload" in args:
        results["reload"] = test_reload(generations=int(os.environ.get("RELOAD_GENS", 3000)))
    if "damage_armor" in args:
        results["damage_armor"] = test_damage_armor(
            generations=int(os.environ.get("DMG_ARMOR_GENS", 2500)))
    if "damage_overflow" in args:
        results["damage_overflow"] = test_damage_overflow(
            generations=int(os.environ.get("DMG_OVER_GENS", 3000)))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, r2 in results.items():
        ok = "PASS" if r2 > 0.95 else ("PARTIAL" if r2 > 0.7 else "FAIL")
        print(f"  {name:20s}  r²={r2:.4f}   [{ok}]")
