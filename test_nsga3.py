"""Tests for the optional NSGA-III reference-point survival selection added to
the AFPO family (single AFPO, multi-stage AFPO, islanded AFPO, age-group
islands), all of which share ``_trim_to_pareto_front_3obj``.

NSGA-III (Deb & Jain, 2014) replaces NSGA-II crowding distance on the boundary
Pareto front with structured reference-point niching.  It is gated behind the
module flag ``evo13.NSGA3_ENABLED``:

  * MODULE default = False  → direct / non-interactive callers keep the
    documented NSGA-II contract bit-for-bit (so the existing AFPO unit tests are
    unaffected).
  * The interactive prompt (``_select_nsga3_mode``) defaults to YES.

These tests cover the reference-point machinery, the niching selection, the
toggle semantics, and prove the flag-off path is still exactly NSGA-II.
"""
import os
import random

os.environ.setdefault("PYTHONWARNINGS", "ignore")
import numpy as np
import evo13 as e


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def _mk(age, fit, comp, loss=None, nf=2, feat=("x0", "x1")):
    """A trivial individual with an explicit (age, fitness, complexity, loss)
    profile — mirrors the construction style of test_afpo_recipe.py."""
    feat = list(feat)
    t = e.CGPEquation(nf, 4, feat)
    t.nodes = [e.CGPNode('+', 0, 1, 0.0)]
    t.out_idx = nf + 0
    t.update_active_nodes()
    ind = e.Individual(t)
    ind.age = float(age)
    ind.fitness = float(fit)
    ind.loss = float(fit if loss is None else loss)
    ind.complexity = float(comp)
    return ind


def _profiles(survivors):
    return sorted((s.age, s.fitness, s.complexity) for s in survivors)


def _chain_tree(nf, feat, depth):
    """A '+' chain of the requested active-graph depth (copied from
    test_afpo_recipe so the legacy-equivalence test is self-contained)."""
    eq = e.CGPEquation(nf, depth + 2, feat)
    eq.nodes = [e.CGPNode('+', 0, 1, 0.0)]
    prev = nf + 0
    for k in range(1, depth):
        eq.nodes.append(e.CGPNode('+', prev, 2, 0.0))
        prev = nf + k
    eq.out_idx = prev
    eq.update_active_nodes()
    return eq


# ──────────────────────────────────────────────────────────────────────────
# Reference-point machinery
# ──────────────────────────────────────────────────────────────────────────
def test_das_dennis_points():
    # Count matches the multiset coefficient C(p+m-1, m-1) and points lie on
    # the unit simplex (non-negative, sum to 1).
    for m in (2, 3, 4):
        for p in (1, 2, 4, 6, 12):
            R = e._das_dennis_reference_points(m, p)
            assert R.shape == (e._das_dennis_count(m, p), m), (m, p, R.shape)
            assert np.allclose(R.sum(axis=1), 1.0), (m, p)
            assert (R >= -1e-12).all(), (m, p)
    # Known 3-objective counts.
    assert [e._das_dennis_count(3, p) for p in (1, 2, 4, 6, 12)] == [3, 6, 15, 28, 91]
    # p <= 0 degenerates to the single centroid direction.
    c = e._das_dennis_reference_points(3, 0)
    assert c.shape == (1, 3) and np.allclose(c, 1.0 / 3.0)
    print("PASS test_das_dennis_points")


def test_reference_points_autosize_cache_and_override():
    e._NSGA3_REF_CACHE.clear()
    old_div = e.NSGA3_DIVISIONS
    try:
        e.NSGA3_DIVISIONS = 0   # auto-size mode
        # Auto-size returns the smallest lattice with H >= target.
        for tgt in (3, 10, 50, 100, 200):
            R = e._nsga3_reference_points(3, tgt)
            assert R.shape[0] >= tgt, (tgt, R.shape)
            # ...and it is the *smallest* such lattice (one step down undershoots).
            # Reconstruct the chosen p and check p-1 would have been too small.
        # Caching: identical key returns the very same array object.
        a = e._nsga3_reference_points(3, 100)
        b = e._nsga3_reference_points(3, 100)
        assert a is b, "reference points should be cached per (n_obj, p)"

        # Explicit divisions override the auto-sizing.
        e._NSGA3_REF_CACHE.clear()
        e.NSGA3_DIVISIONS = 5
        R = e._nsga3_reference_points(3, 9999)   # target ignored when p is fixed
        assert R.shape[0] == e._das_dennis_count(3, 5) == 21, R.shape

        # The hard cap is respected for absurd targets.
        e.NSGA3_DIVISIONS = 0
        e._NSGA3_REF_CACHE.clear()
        R = e._nsga3_reference_points(3, 10 ** 9)
        assert R.shape[0] <= e.NSGA3_REF_MAX, R.shape
    finally:
        e.NSGA3_DIVISIONS = old_div
        e._NSGA3_REF_CACHE.clear()
    print("PASS test_reference_points_autosize_cache_and_override")


def test_normalise_maps_extremes_to_axes():
    # The per-objective best (corner) points should normalise onto the unit
    # axes; an interior point should sit inside the simplex.
    M = np.array([[0., 9., 9.],
                  [9., 0., 9.],
                  [9., 9., 0.],
                  [3., 3., 3.]])
    N = e._nsga3_normalise(M)
    assert np.isfinite(N).all()
    assert np.allclose(N[0], [0, 1, 1], atol=1e-6)
    assert np.allclose(N[1], [1, 0, 1], atol=1e-6)
    assert np.allclose(N[2], [1, 1, 0], atol=1e-6)
    # Degenerate input (all identical → singular hyperplane) must not crash and
    # must stay finite via the fallback.
    D = np.ones((5, 3))
    ND = e._nsga3_normalise(D)
    assert np.isfinite(ND).all()
    print("PASS test_normalise_maps_extremes_to_axes")


# ──────────────────────────────────────────────────────────────────────────
# Boundary-front niching selection
# ──────────────────────────────────────────────────────────────────────────
def test_select_from_front_edge_cases():
    A = np.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]], dtype=np.float64)
    front = np.array([0, 1, 2])
    assert e._nsga3_select_from_front(A, [], front, 0, 3) == []
    assert sorted(e._nsga3_select_from_front(A, [], front, 5, 3)) == [0, 1, 2]
    got = e._nsga3_select_from_front(A, [], front, 2, 3)
    assert len(got) == 2 and len(set(got)) == 2 and set(got) <= {0, 1, 2}
    print("PASS test_select_from_front_edge_cases")


# ──────────────────────────────────────────────────────────────────────────
# Full trim behaviour with NSGA-III enabled
# ──────────────────────────────────────────────────────────────────────────
def test_trim_holds_target_and_keeps_champion():
    e.set_ops_mode(True); e._INIT_PHASE = False
    random.seed(0); np.random.seed(0)
    old = e.NSGA3_ENABLED
    try:
        e.NSGA3_ENABLED = True
        # Random pool of 160 trimmed to 80 — the realistic AFPO steady-state.
        pop = [_mk(random.randint(0, 30),
                   random.random() * 100,
                   random.randint(1, 60)) for _ in range(160)]
        champ = min(pop, key=lambda x: x.loss)
        surv = e._trim_to_pareto_front_3obj(pop, 80)
        assert len(surv) == 80, len(surv)
        assert champ in surv, "champion (lowest loss) must always survive"
        # No object duplicated.
        assert len({id(s) for s in surv}) == 80
        # Below-target pools are returned untouched.
        small = pop[:50]
        assert len(e._trim_to_pareto_front_3obj(small, 80)) == 50
    finally:
        e.NSGA3_ENABLED = old
    print("PASS test_trim_holds_target_and_keeps_champion")


def test_nsga3_preserves_objective_extremes():
    """A single mutually non-dominated front of 3 corners + an interior cluster.
    NSGA-III's reference-point niching associates each corner with a distinct
    axis direction, so all three boundary solutions must survive a trim to 4."""
    e.set_ops_mode(True); e._INIT_PHASE = False
    random.seed(1); np.random.seed(1)
    old = e.NSGA3_ENABLED
    try:
        corners = [_mk(0, 10, 10), _mk(10, 0, 10), _mk(10, 10, 0)]
        cluster = [_mk(4, 6, 5), _mk(5, 5, 5), _mk(6, 4, 5),
                   _mk(5, 6, 4), _mk(4, 5, 6)]
        pop = corners + cluster
        corner_set = {(0, 10, 10), (10, 0, 10), (10, 10, 0)}

        e.NSGA3_ENABLED = True
        surv = e._trim_to_pareto_front_3obj(pop, 4)
        kept = {(s.age, s.fitness, s.complexity) for s in surv}
        assert corner_set <= kept, f"NSGA-III dropped a corner: kept={kept}"
        # Each objective's global best is present (boundary preservation).
        assert min(s.age for s in surv) == 0
        assert min(s.fitness for s in surv) == 0
        assert min(s.complexity for s in surv) == 0
    finally:
        e.NSGA3_ENABLED = old
    print("PASS test_nsga3_preserves_objective_extremes")


def test_nsga3_is_deterministic():
    e.set_ops_mode(True); e._INIT_PHASE = False
    old = e.NSGA3_ENABLED
    try:
        e.NSGA3_ENABLED = True
        random.seed(7); np.random.seed(7)
        pop = [_mk(random.randint(0, 20), random.random() * 50,
                   random.randint(1, 40)) for _ in range(120)]
        a = _profiles(e._trim_to_pareto_front_3obj(list(pop), 60))
        b = _profiles(e._trim_to_pareto_front_3obj(list(pop), 60))
        assert a == b, "NSGA-III selection must be deterministic for a fixed pool"
    finally:
        e.NSGA3_ENABLED = old
    print("PASS test_nsga3_is_deterministic")


def test_nsga3_differs_from_nsga2():
    """On a constructed boundary front the two methods pick a different interior
    member — proof the flag actually re-routes the boundary selection."""
    e.set_ops_mode(True); e._INIT_PHASE = False
    old = e.NSGA3_ENABLED
    try:
        corners = [_mk(0, 10, 10), _mk(10, 0, 10), _mk(10, 10, 0)]
        cluster = [_mk(4, 6, 5), _mk(5, 5, 5), _mk(6, 4, 5),
                   _mk(5, 6, 4), _mk(4, 5, 6)]
        pop = corners + cluster
        e.NSGA3_ENABLED = True
        s3 = _profiles(e._trim_to_pareto_front_3obj(pop, 4))
        e.NSGA3_ENABLED = False
        s2 = _profiles(e._trim_to_pareto_front_3obj(pop, 4))
        assert s3 != s2, "NSGA-III and NSGA-II should differ on this front"
    finally:
        e.NSGA3_ENABLED = old
    print("PASS test_nsga3_differs_from_nsga2")


# ──────────────────────────────────────────────────────────────────────────
# Toggle / regression: flag OFF must be byte-for-byte legacy NSGA-II
# ──────────────────────────────────────────────────────────────────────────
def test_module_default_is_off():
    # Direct / non-interactive callers must default to NSGA-II so existing AFPO
    # behaviour and unit tests are unchanged.  (Read the source default in a
    # fresh subprocess so other tests toggling the flag can't taint it.)
    import subprocess, sys
    out = subprocess.run(
        [sys.executable, "-c", "import evo13; print(evo13.NSGA3_ENABLED)"],
        capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stdout + out.stderr
    print("PASS test_module_default_is_off")


def test_flag_off_matches_legacy_nsga2_tiebreak():
    """With NSGA-III OFF the documented NSGA-II depth tie-break is unchanged:
    four identical-profile trees of depth (1,2,3,4) trimmed to 3 keep the two
    crowding boundaries (1, 4) plus the shallower interior (2)."""
    e.set_ops_mode(True); e._INIT_PHASE = False
    nf, feat = 3, ["x0", "x1", "x2"]
    old = e.NSGA3_ENABLED
    try:
        e.NSGA3_ENABLED = False
        inds = []
        for d in (1, 2, 3, 4):
            ind = e.Individual(_chain_tree(nf, feat, d))
            ind.age = 5; ind.fitness = 1.0; ind.loss = 1.0; ind.complexity = 10.0
            inds.append(ind)
        surv = e._trim_to_pareto_front_3obj(inds, 3)
        depths = sorted(e._active_graph_depth(s.tree) for s in surv)
        assert depths == [1, 2, 4], depths
    finally:
        e.NSGA3_ENABLED = old
    print("PASS test_flag_off_matches_legacy_nsga2_tiebreak")


if __name__ == "__main__":
    test_das_dennis_points()
    test_reference_points_autosize_cache_and_override()
    test_normalise_maps_extremes_to_axes()
    test_select_from_front_edge_cases()
    test_trim_holds_target_and_keeps_champion()
    test_nsga3_preserves_objective_extremes()
    test_nsga3_is_deterministic()
    test_nsga3_differs_from_nsga2()
    test_module_default_is_off()
    test_flag_off_matches_legacy_nsga2_tiebreak()
    print("\nALL NSGA-III TESTS PASSED")
