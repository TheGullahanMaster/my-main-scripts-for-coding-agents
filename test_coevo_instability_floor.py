"""Regression tests for the instability-signal loss-trend floor in evo13's
predator–prey co-evolution.

The stabiliser's instability ∈ [0,1] blends two terms (see _update_progress):

  • a *relative* loss-trend rise  (fast EMA − slow EMA) / slow, and
  • an *absolute* forgetting fraction (mean per-row regression).

The relative trend term has a denominator (the slow loss EMA) that goes to ~0 as
the hosts converge.  With only a tiny epsilon guard, a near-solved model whose
loss merely drifts from, say, 1e-4 to 4e-3 — utterly negligible on the [0,1)
hardness scale — produces a >100% *relative* rise and saturates instability.  The
stabiliser then widens the anchor and damps the predators exactly when they
should be honing the last few hard rows, slowing the final approach to a perfect
fit.

The fix floors that denominator on the absolute [0,1) scale
(`instab_loss_floor`, module knob COEVO_INSTAB_LOSS_FLOOR) so the trend term only
counts once the loss is meaningfully nonzero.  The absolute forgetting term is
untouched, so the many-variable forgetting signal still fires below the floor.

Run:  python test_coevo_instability_floor.py
"""
import numpy as np

import evo13


# --------------------------------------------------------------------------
# fakes (a champion whose per-row residual pattern we control + can mutate)
# --------------------------------------------------------------------------
class _MutTree:
    def __init__(self, y):
        self.y = y.astype(np.float64)
        self.resid = np.zeros_like(self.y)
    def evaluate(self, X):
        ids = X[:, 0].astype(int)
        return (self.y + self.resid)[ids]


class _MutChampion:
    def __init__(self, y):
        self.tree = _MutTree(y)
        self.affine_a = 1.0
        self.affine_b = 0.0


class _MutHoF:
    def __init__(self, champ):
        self._c = champ
    def get_best_overall(self):
        return self._c


def _X_ids(n):
    X = np.zeros((n, 1), dtype=np.float64)
    X[:, 0] = np.arange(n)
    return X


def ok(name):
    print(f"  ✓ {name}")


def _make(n=300, subset=30, **kw):
    return evo13._PredatorPreyCoevolution(
        n_rows=n, subset=subset, pop_size=16, mut_rate=0.3, virulence=0.75,
        seed=1, **kw)


# A deterministic near-solved loss curve: a tiny absolute level that ramps UP
# (1e-4 → 4e-3) then holds.  Negligible on the [0,1) scale, but a large RELATIVE
# rise — the exact regime that made the un-floored trend term explode.
def _tiny_rising_levels(steps=60, ramp=30, lo=1e-4, hi=4e-3):
    return [lo + (hi - lo) * min(t, ramp) / ramp for t in range(steps)]


# --------------------------------------------------------------------------
# 1.  the bug: an un-floored denominator saturates on a near-solved model
# --------------------------------------------------------------------------
def test_unfloored_denominator_reproduces_false_positive():
    print("1: a ~0 floor reproduces the spurious instability near convergence")
    n = 300
    # instab_loss_floor ~ 0 reproduces the old (epsilon-only) behaviour.
    co = _make(n=n, instab_loss_floor=1e-9)
    peak = 0.0
    for lvl in _tiny_rising_levels():
        co._update_progress(np.full(n, lvl))
        peak = max(peak, co.instability)
    assert co.forget_frac < 1e-2, \
        f"forgetting must stay ~0 here (got {co.forget_frac:.3f}) — the signal is the trend term"
    assert peak > 0.5, \
        f"un-floored: a near-solved model should (wrongly) read high instability (peak={peak:.3f})"
    ok(f"un-floored denominator saturates on an essentially-solved model (peak={peak:.3f})")
    print()


# --------------------------------------------------------------------------
# 2.  the fix: flooring the denominator keeps instability ~0 on a solved model
# --------------------------------------------------------------------------
def test_floor_suppresses_false_positive():
    print("2: flooring the loss-trend denominator removes the false positive")
    n = 300
    co = _make(n=n, instab_loss_floor=0.02)      # the shipped default
    peak = 0.0
    for lvl in _tiny_rising_levels():
        co._update_progress(np.full(n, lvl))
        peak = max(peak, co.instability)
    assert peak < 0.15, \
        f"floored: a near-solved model must stay ~calm (peak instability={peak:.3f})"
    # And the stabiliser therefore leaves the batch shape essentially at config.
    assert abs(co._effective_anchor_frac() - co.anchor_frac) < 0.03, \
        f"anchor should stay ~configured on a solved model ({co._effective_anchor_frac():.3f})"
    assert co._effective_virulence() > 0.9 * co.virulence, \
        f"virulence should stay ~configured on a solved model ({co._effective_virulence():.3f})"
    ok(f"floored denominator keeps instability calm near convergence (peak={peak:.3f})")
    ok("→ effective anchor & virulence stay at their configured values (predators keep honing)")
    print()


# --------------------------------------------------------------------------
# 3.  the floor does NOT change behaviour where the loss is meaningful
# --------------------------------------------------------------------------
def test_floor_is_inert_at_meaningful_loss():
    print("3: at a meaningful loss the floor is inert (no behaviour change)")
    n = 300
    co_floor = _make(n=n, instab_loss_floor=0.02)
    co_eps   = _make(n=n, instab_loss_floor=1e-9)
    # A loss that lives well ABOVE the floor the whole time (0.30 → 0.45): the
    # denominator is loss_slow in both cases, so the two must track identically.
    for t in range(40):
        lvl = 0.30 + 0.15 * min(t, 20) / 20.0
        h = np.full(n, lvl)
        co_floor._update_progress(h)
        co_eps._update_progress(h)
    assert abs(co_floor.instability - co_eps.instability) < 1e-6, \
        (f"above the floor the two must match "
         f"({co_floor.instability:.6f} vs {co_eps.instability:.6f})")
    ok(f"identical instability when loss ≫ floor ({co_floor.instability:.4f})")
    print()


# --------------------------------------------------------------------------
# 4.  the absolute forgetting term still fires (many-variable signal preserved)
# --------------------------------------------------------------------------
def test_forgetting_signal_preserved_below_floor():
    print("4: genuine forgetting still raises instability (floor only tames the trend)")
    n = 300
    co = _make(n=n, instab_loss_floor=0.02, noise_guard=False)
    # Learn everything (best-ever → ~0), staying below the loss floor throughout.
    for _ in range(20):
        co._update_progress(np.zeros(n))
    assert co.instability < 1e-6, "a flat solved run must read ~0 instability"
    # Now DROP a third of the rows: forgetting (mean regression) is large.
    h = np.zeros(n); h[:100] = 0.9
    for _ in range(8):
        co._update_progress(h)
    assert co.forget_frac > 0.1, f"forgetting fraction should be large ({co.forget_frac:.3f})"
    assert co.instability > 0.3, \
        f"forgetting must still drive instability up ({co.instability:.3f})"
    ok(f"dropped-variable forgetting still fires (forget={co.forget_frac:.3f} → "
       f"instability={co.instability:.3f})")
    print()


# --------------------------------------------------------------------------
# 5.  closed loop: on a near-converged plateau with a tiny upward loss creep
#     (the realistic regime — the model is essentially solved but adversarial
#     pressure nudges the full-data residual up a hair) the un-floored stabiliser
#     over-inflates the anchor; the floor keeps it near the configured value.
# --------------------------------------------------------------------------
def _plateau_run(instab_loss_floor, chunks=90):
    """A champion that converges to a low plateau (~2% residual) and then drifts
    up very slightly — a negligible ABSOLUTE move but a real *relative* rise.
    Returns the mean effective anchor fraction over the settled tail.

    The residual is uniform across rows and follows a fixed deterministic
    schedule, so the per-row hardness — and hence the whole instability signal —
    depends only on the schedule, not on the data (kept simple on purpose)."""
    n, subset = 400, 40
    y = np.zeros(n)
    X, Y = _X_ids(n), y.reshape(-1, 1) + 1.0   # any non-degenerate target spread
    sd = 1.0
    champ = _MutChampion(np.zeros(n))
    champ.tree.y = (Y[:, 0]).astype(np.float64)
    hof = [_MutHoF(champ)]
    co = evo13._PredatorPreyCoevolution(
        n_rows=n, subset=subset, pop_size=24, mut_rate=0.3, virulence=0.75,
        seed=0, anchor_frac=0.25, fitness_sharing=True, noise_guard=False,
        stabilize=True, instab_loss_floor=instab_loss_floor)
    anchors = []
    for t in range(chunks):
        if t < 30:
            level = 0.5 * (0.6 ** (t / 5.0))              # converge to a plateau
        else:
            level = 0.02 + 0.0015 * (t - 30)              # tiny upward creep
        champ.tree.resid[:] = level * sd
        co.next_batch(X, Y, hof, [0])
        if t >= chunks - 30:
            anchors.append(co._effective_anchor_frac())
    return float(np.mean(anchors)), co.anchor_frac


def test_plateau_creep_does_not_inflate_anchor():
    print("5: closed loop — the floor stops a near-solved plateau over-inflating the anchor")
    a_floor, cfg = _plateau_run(instab_loss_floor=0.02)
    a_eps,   _   = _plateau_run(instab_loss_floor=1e-9)
    print(f"    configured anchor            = {cfg:.2f}")
    print(f"    ~0 floor  (old) tail anchor  = {a_eps:.3f}   (over-inflated)")
    print(f"    0.02 floor (fix) tail anchor = {a_floor:.3f}")
    assert a_eps > cfg + 0.05, \
        f"un-floored: a tiny low-level creep should (wrongly) inflate the anchor ({a_eps:.3f})"
    assert a_floor < a_eps - 0.02, \
        f"the floor should clearly reduce the over-inflation ({a_floor:.3f} vs {a_eps:.3f})"
    assert (a_floor - cfg) < (a_eps - cfg), "floored anchor must sit closer to configured"
    ok(f"floor keeps the anchor near configured on a solved plateau "
       f"({a_eps:.3f} → {a_floor:.3f}, cfg {cfg:.2f})")
    print()


# --------------------------------------------------------------------------
# 6.  the knob is wired through the module + runtime builder
# --------------------------------------------------------------------------
def test_knob_wired_through_runtime():
    print("6: COEVO_INSTAB_LOSS_FLOOR is threaded into the lazily-built runtime")
    assert hasattr(evo13, "COEVO_INSTAB_LOSS_FLOOR"), "module knob missing"
    n = 200
    X = np.random.RandomState(0).randn(n, 1)
    Y = np.zeros((n, 1))
    saved = (evo13.COEVOLUTION_ENABLED, evo13.COEVO_CASE_SUBSET,
             evo13.COEVO_INSTAB_LOSS_FLOOR, evo13._COEVO_RUNTIME)
    try:
        evo13.COEVOLUTION_ENABLED = True
        evo13.COEVO_CASE_SUBSET = 20
        evo13.COEVO_INSTAB_LOSS_FLOOR = 0.037
        evo13._COEVO_RUNTIME = None
        evo13._coevo_next_batch(X, Y, [_MutHoF(None)], [0])   # builds the runtime
        assert evo13._COEVO_RUNTIME is not None, "runtime should be built"
        assert abs(evo13._COEVO_RUNTIME.instab_loss_floor - 0.037) < 1e-9, \
            "the module knob must propagate into the runtime instance"
        ok("COEVO_INSTAB_LOSS_FLOOR propagates into _PredatorPreyCoevolution")
    finally:
        (evo13.COEVOLUTION_ENABLED, evo13.COEVO_CASE_SUBSET,
         evo13.COEVO_INSTAB_LOSS_FLOOR, evo13._COEVO_RUNTIME) = saved
    print()


if __name__ == "__main__":
    test_unfloored_denominator_reproduces_false_positive()
    test_floor_suppresses_false_positive()
    test_floor_is_inert_at_meaningful_loss()
    test_forgetting_signal_preserved_below_floor()
    test_plateau_creep_does_not_inflate_anchor()
    test_knob_wired_through_runtime()
    print("ALL INSTABILITY-FLOOR TESTS PASSED ✓")
