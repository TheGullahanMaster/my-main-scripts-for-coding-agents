"""Regression tests for three AFPO improvements:

  1. Population diversity-collapse prevention — the periodic diversity check now
     measures a scale-invariant clone signal and actively replaces redundant
     behavioural clones with fresh blood while stalling.  Smoke-tested for
     validity (no crash, healthy pool) on a collapse-prone target.

  2. Inter-chunk slowdown control — ``apply_simplification_pass`` honours a
     wall-clock budget and stops early, leaving the remainder for next time.

  3. Variable-step generation schedule — ``_gen_crossed`` fires a periodic task
     exactly once per period even when the chunk size straddles the boundary
     (e.g. 475 -> 525 with period 500), which the legacy ``gen % period == 0``
     test silently skipped.
"""
import io
import os
import random
import contextlib

os.environ.setdefault("PYTHONWARNINGS", "ignore")
import numpy as np
import evo13 as e


# ──────────────────────────────────────────────────────────────────────────
# 3. _gen_crossed — variable-step boundary scheduling
# ──────────────────────────────────────────────────────────────────────────
def test_gen_crossed_basic():
    # Fires AT the round multiple (matching legacy gen % period == 0 intent).
    assert e._gen_crossed(450, 500, 500) is True
    assert e._gen_crossed(500, 550, 500) is False      # already fired at 500
    assert e._gen_crossed(0, 50, 500) is False         # nowhere near 500
    # The reported skip scenario: a step straddles 500 (475 -> 525).
    assert e._gen_crossed(475, 525, 500) is True
    assert e._gen_crossed(525, 575, 500) is False      # no boundary in (525,575]
    # Degenerate / guard inputs.
    assert e._gen_crossed(10, 20, 0) is False
    print("PASS test_gen_crossed_basic")


def test_legacy_modulo_skips_but_crossed_does_not():
    """The exact failure the user reported: gen jumps 400→425→475→525, never
    equalling 500, so ``gen % 500 == 0`` never fires — but ``_gen_crossed`` does
    fire once, on the 475→525 step."""
    seq = [400, 425, 475, 525, 575]
    legacy_fires = [g for g in seq if g % 500 == 0]
    assert legacy_fires == [], "legacy modulo should skip 500 here"

    crossed = []
    prev = seq[0]
    for g in seq[1:]:
        if e._gen_crossed(prev, g, 500):
            crossed.append((prev, g))
        prev = g
    assert crossed == [(475, 525)], crossed
    print("PASS test_legacy_modulo_skips_but_crossed_does_not")


def test_gen_crossed_fires_once_per_interval_variable_step():
    """Over a long run with a randomly halving chunk size, every 500-gen
    boundary fires exactly once — no skips, no double fires."""
    rng = random.Random(0)
    gen = 0
    fires = []
    while gen < 5000:
        prev = gen
        gen += rng.choice([25, 50])      # MIGRATION_FREQ or its halved value
        if e._gen_crossed(prev, gen, 500):
            fires.append(gen // 500)
    expected = list(range(1, gen // 500 + 1))
    assert sorted(fires) == expected, (fires, expected)
    assert len(fires) == len(set(fires)), f"double fire: {fires}"
    print("PASS test_gen_crossed_fires_once_per_interval_variable_step")


def test_gen_crossed_offset_stagger():
    """A half-period offset (used to stagger the simplification pass off the
    500/1000 heavy-pass cluster) fires on its own phase and never coincides
    with the unoffset schedule."""
    fires_0, fires_250 = [], []
    prev = 0
    for gen in range(50, 1501, 50):
        if e._gen_crossed(prev, gen, 500):
            fires_0.append(gen)
        if e._gen_crossed(prev, gen, 500, offset=250):
            fires_250.append(gen)
        prev = gen
    assert fires_0 == [500, 1000, 1500], fires_0
    assert fires_250 == [250, 750, 1250], fires_250
    assert not (set(fires_0) & set(fires_250)), "stagger should never coincide"
    print("PASS test_gen_crossed_offset_stagger")


# ──────────────────────────────────────────────────────────────────────────
# 2. apply_simplification_pass wall-clock budget
# ──────────────────────────────────────────────────────────────────────────
def _make_pop(nf, feat, X, y, n, otype=5):
    pop = [e.Individual(e.random_cgp(nf, e.CGP_NODES, feat)) for _ in range(n)]
    for ind in pop:
        ind.calculate_fitness(X, y, otype)
    return pop


def test_simplification_budget_defers_remainder():
    np.random.seed(0); random.seed(0)
    e.set_ops_mode(True); e._INIT_PHASE = False
    nf, feat = 2, ["x0", "x1"]
    rng = np.random.RandomState(0)
    X = rng.uniform(-2.0, 2.0, size=(80, 2))
    y = X[:, 0] * X[:, 1] + 0.3 * X[:, 0]
    Y = y.reshape(-1, 1)

    # A near-zero budget must stop early and print the deferral note; an
    # unbounded pass must NOT print it.
    pop = _make_pop(nf, feat, X, y, 24)
    hof = e.HallOfFame(out_type=5)
    for ind in pop:
        hof.update(ind)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        e.apply_simplification_pass([[pop]], [hof], X, Y, [5],
                                    time_budget_sec=1e-6)
    out_budget = buf.getvalue()
    assert "budget" in out_budget, out_budget

    pop2 = _make_pop(nf, feat, X, y, 24)
    hof2 = e.HallOfFame(out_type=5)
    for ind in pop2:
        hof2.update(ind)
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        e.apply_simplification_pass([[pop2]], [hof2], X, Y, [5],
                                    time_budget_sec=None)
    out_unbounded = buf2.getvalue()
    assert "budget" not in out_unbounded, out_unbounded
    # Unbounded processes strictly more individuals than the near-zero budget.
    def _count(s):
        # line: "  [Simplify] N individuals | ..."
        for tok in s.replace("[Simplify]", " ").split():
            if tok.isdigit():
                return int(tok)
        return -1
    assert _count(out_unbounded) > _count(out_budget), (out_budget, out_unbounded)
    print("PASS test_simplification_budget_defers_remainder")


# ──────────────────────────────────────────────────────────────────────────
# 1. Diversity-collapse prevention — functional smoke test
# ──────────────────────────────────────────────────────────────────────────
def test_diversity_injection_runs_and_keeps_pool_valid():
    """On a constant target every individual collapses onto the same prediction
    (clone collapse) and, since it is already optimal, the pool stalls — exactly
    the regime that arms the active de-duplication.  Verify the worker runs
    cleanly and returns a structurally valid pool, with the feature both on and
    off (so neither code path regresses)."""
    nf, feat = 2, ["x0", "x1"]
    rng = np.random.RandomState(3)
    X = rng.uniform(-2.0, 2.0, size=(100, 2))
    y = np.full(X.shape[0], 1.5)        # constant → affine fits all → clones

    for flag in (True, False):
        np.random.seed(7); random.seed(7)
        e.set_ops_mode(True)
        e.AFFINE_SCALING_ENABLED = True
        e._INIT_PHASE = False
        e.DIVERSITY_INJECTION_ENABLED = flag
        pop = [e.Individual(e.random_cgp(nf, e.CGP_NODES, feat)) for _ in range(40)]
        for ind in pop:
            ind.calculate_fitness(X, y, 5)
        hof = e.HallOfFame(out_type=5)
        # ext_patience low enough that local_stag crosses the inject threshold
        # within the run, n_generations spanning several diversity checks.
        pop, stag = e.evolve_afpo(pop, X, y, 5, nf, feat, target_size=40,
                                  n_generations=160, hof=hof,
                                  stag_counter=0, ext_patience=80)
        assert len(pop) > 0
        assert len(set(id(i) for i in pop)) == len(pop), "no aliased individuals"
        assert all(getattr(i, "age", 0) >= 0 for i in pop)
        assert all(np.isfinite(getattr(i, "fitness", np.inf)) or i.fitness >= 1e9
                   for i in pop)
        print(f"  diversity-injection={flag}: pool={len(pop)} stag={stag}")
    e.DIVERSITY_INJECTION_ENABLED = True   # restore default
    print("PASS test_diversity_injection_runs_and_keeps_pool_valid")


if __name__ == "__main__":
    test_gen_crossed_basic()
    test_legacy_modulo_skips_but_crossed_does_not()
    test_gen_crossed_fires_once_per_interval_variable_step()
    test_gen_crossed_offset_stagger()
    test_simplification_budget_defers_remainder()
    test_diversity_injection_runs_and_keeps_pool_valid()
    print("\nALL AFPO IMPROVEMENT TESTS PASSED")
