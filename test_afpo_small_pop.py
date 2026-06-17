"""Regression tests for AFPO small-population diversity tuning.

``evolve_afpo`` is the shared base core for every AFPO variant.  The default
user population (50) doubles to a 100-member pool, and harder tasks are often
run with even smaller pools — which collapse onto a single behaviour faster and
recover worse (fewer surviving niches) than large ones.  Three knobs make the
diversity-collapse machinery population-aware, each reducing to EXACTLY the
legacy value at the default-and-larger pool so existing runs are unchanged:

  1. ``_diversity_check_freq`` — the periodic diversity check (which drives both
     the structural-operator boost and the active de-duplication) fires more
     often as the pool shrinks, capped at the legacy 50 for pools >= 100.

  2. ``_diversity_dedup_stag_gate`` — the active de-dup's stagnation gate is
     relaxed by collapse SEVERITY: a near-monomorphic pool fires at a lower
     (never zero) stagnation bar, while a mild collapse still clears the full
     bar so genuine convergence onto an optimum is left alone.  Applies at every
     pool size.

  3. ``_diversity_dedup_cap`` — the replacement cap scales up for small pools (a
     fixed 15% of a tiny pool is too few distinct newcomers to re-diversify),
     bounded by ``DIVERSITY_INJECT_MAX_FRAC_CAP``; pools >= the pivot keep the
     legacy 15%.
"""
import os
import random

os.environ.setdefault("PYTHONWARNINGS", "ignore")
import numpy as np
import evo13 as e


# ──────────────────────────────────────────────────────────────────────────
# 1. Cadence — population-aware, legacy at the default pool
# ──────────────────────────────────────────────────────────────────────────
def test_check_freq_legacy_at_default_pool_and_larger():
    # The default user pop (50) doubles to a 100-member AFPO pool: cadence must
    # be IDENTICAL to the historical fixed 50 there and for every larger pool.
    for ts in (100, 120, 200, 500):
        assert e._diversity_check_freq(ts) == e.DIVERSITY_CHECK_FREQ_MAX == 50, ts
    print("PASS test_check_freq_legacy_at_default_pool_and_larger")


def test_check_freq_shrinks_for_small_pools():
    # Strictly more frequent as the pool shrinks, monotone non-increasing in ts,
    # floored at DIVERSITY_CHECK_FREQ_MIN.
    prev = -1
    for ts in (10, 20, 30, 40, 50, 60, 80, 100):
        f = e._diversity_check_freq(ts)
        assert e.DIVERSITY_CHECK_FREQ_MIN <= f <= e.DIVERSITY_CHECK_FREQ_MAX, (ts, f)
        if prev >= 0:
            assert f >= prev, f"cadence must be non-decreasing in pop size: {ts}"
        prev = f
    # A genuinely small pool checks at least twice as often as the default.
    assert e._diversity_check_freq(40) <= 25 < e._diversity_check_freq(100)
    assert e._diversity_check_freq(10) == e.DIVERSITY_CHECK_FREQ_MIN
    print("PASS test_check_freq_shrinks_for_small_pools")


# ──────────────────────────────────────────────────────────────────────────
# 2. Severity-scaled stagnation gate
# ──────────────────────────────────────────────────────────────────────────
def test_collapse_severity_endpoints():
    nfp = 20
    # At the trip threshold severity is 0; at total collapse (1 fingerprint) it
    # saturates to 1.
    assert e._diversity_collapse_severity(e.DIVERSITY_UNIQUE_FRAC, nfp) == 0.0
    assert e._diversity_collapse_severity(1.0 / nfp, nfp) == 1.0
    # Monotone decreasing unique_frac -> increasing severity.
    sevs = [e._diversity_collapse_severity(uf, nfp)
            for uf in (0.55, 0.45, 0.35, 0.25, 0.15, 0.05)]
    assert all(b >= a for a, b in zip(sevs, sevs[1:])), sevs
    print("PASS test_collapse_severity_endpoints")


def test_stag_gate_relaxes_with_severity_but_never_zero():
    nfp = 20
    base = e.DIVERSITY_INJECT_STAG_FRAC
    # Mild collapse keeps (essentially) the full bar; total collapse relaxes to
    # exactly (1 - relax)x base, and never below.
    assert abs(e._diversity_dedup_stag_gate(e.DIVERSITY_UNIQUE_FRAC, nfp) - base) < 1e-12
    floor = base * (1.0 - e.DIVERSITY_DEEP_COLLAPSE_RELAX)
    assert abs(e._diversity_dedup_stag_gate(1.0 / nfp, nfp) - floor) < 1e-12
    for uf in np.linspace(0.0, e.DIVERSITY_UNIQUE_FRAC, 40):
        g = e._diversity_dedup_stag_gate(float(uf), nfp)
        assert floor - 1e-12 <= g <= base + 1e-12, (uf, g)
    print("PASS test_stag_gate_relaxes_with_severity_but_never_zero")


def test_severe_collapse_fires_dedup_where_mild_does_not():
    """The concrete small-pool benefit: at a stagnation level BETWEEN the
    relaxed and full bars, a near-monomorphic pool now triggers the active
    de-dup while a mild collapse (possible benign convergence) still does not."""
    nfp = 20
    ext_patience = 100
    # Sit between the total-collapse gate (0.125*ext) and the full gate
    # (0.25*ext): legacy behaviour (flat 0.25) would NOT fire here at all.
    local_stag = 0.18 * ext_patience
    mild_gate = e._diversity_dedup_stag_gate(0.50, nfp)    # just under threshold
    severe_gate = e._diversity_dedup_stag_gate(1.0 / nfp, nfp)
    assert local_stag <= mild_gate * ext_patience, "mild collapse must NOT fire"
    assert local_stag > severe_gate * ext_patience, "severe collapse MUST fire"
    print("PASS test_severe_collapse_fires_dedup_where_mild_does_not")


# ──────────────────────────────────────────────────────────────────────────
# 3. Replacement cap
# ──────────────────────────────────────────────────────────────────────────
def test_dedup_cap_legacy_at_pivot_and_larger():
    for ts in (80, 100, 200, 400):
        assert e._diversity_dedup_cap(ts) == max(1, int(ts * e.DIVERSITY_INJECT_MAX_FRAC)), ts
    print("PASS test_dedup_cap_legacy_at_pivot_and_larger")


def test_dedup_cap_larger_for_small_pools_but_bounded():
    # Small pools refresh a larger absolute share than the legacy 15%...
    for ts in (20, 30, 40):
        assert e._diversity_dedup_cap(ts) > max(1, int(ts * e.DIVERSITY_INJECT_MAX_FRAC)), ts
    # ...but never beyond the hard fraction ceiling.
    for ts in (5, 10, 20, 30, 50, 100, 200):
        assert e._diversity_dedup_cap(ts) <= int(ts * e.DIVERSITY_INJECT_MAX_FRAC_CAP) + 1, ts
    print("PASS test_dedup_cap_larger_for_small_pools_but_bounded")


# ──────────────────────────────────────────────────────────────────────────
# 4. End-to-end smoke — small pool stays valid with the feature on AND off
# ──────────────────────────────────────────────────────────────────────────
def _distinct_fingerprints(pop, X_probe):
    fps = set()
    for ind in pop:
        try:
            p = np.nan_to_num(ind.tree.evaluate(X_probe), nan=0.0,
                              posinf=1e9, neginf=-1e9)
            fps.add(tuple(float(f"{v:.3g}") for v in p))
        except Exception:
            pass
    return len(fps)


def test_small_pool_evolve_afpo_runs_clean():
    """A small pool on a collapse-prone target must run cleanly through the new
    population-aware paths with the injection feature both ON and OFF (so neither
    branch regresses), returning a structurally valid, non-aliased pool."""
    nf, feat = 2, ["x0", "x1"]
    rng = np.random.RandomState(5)
    X = rng.uniform(-2.0, 2.0, size=(120, 2))
    y = np.full(X.shape[0], 0.75)        # constant -> affine fits all -> clones

    results = {}
    for flag in (True, False):
        np.random.seed(11); random.seed(11)
        e.set_ops_mode(True)
        e.AFFINE_SCALING_ENABLED = True
        e._INIT_PHASE = False
        e.DIVERSITY_INJECTION_ENABLED = flag
        TS = 24                          # genuinely small pool (well under pivot)
        pop = [e.Individual(e.random_cgp(nf, e.CGP_NODES, feat)) for _ in range(TS)]
        for ind in pop:
            ind.calculate_fitness(X, y, 5)
        hof = e.HallOfFame(out_type=5)
        # ext_patience low enough that local_stag crosses the (relaxed) gate.
        pop, stag = e.evolve_afpo(pop, X, y, 5, nf, feat, target_size=TS,
                                  n_generations=140, hof=hof,
                                  stag_counter=0, ext_patience=60)
        assert len(pop) > 0
        assert len(set(id(i) for i in pop)) == len(pop), "no aliased individuals"
        assert all(getattr(i, "age", 0) >= 0 for i in pop)
        assert all(np.isfinite(getattr(i, "fitness", np.inf)) or i.fitness >= 1e9
                   for i in pop)
        results[flag] = (len(pop), stag, _distinct_fingerprints(pop, X[:16]))
        print(f"  injection={flag}: pool={results[flag][0]} "
              f"stag={results[flag][1]} distinct_fp={results[flag][2]}")
    e.DIVERSITY_INJECTION_ENABLED = True   # restore default
    print("PASS test_small_pool_evolve_afpo_runs_clean")


if __name__ == "__main__":
    test_check_freq_legacy_at_default_pool_and_larger()
    test_check_freq_shrinks_for_small_pools()
    test_collapse_severity_endpoints()
    test_stag_gate_relaxes_with_severity_but_never_zero()
    test_severe_collapse_fires_dedup_where_mild_does_not()
    test_dedup_cap_legacy_at_pivot_and_larger()
    test_dedup_cap_larger_for_small_pools_but_bounded()
    test_small_pool_evolve_afpo_runs_clean()
    print("\nALL AFPO SMALL-POP TESTS PASSED")
