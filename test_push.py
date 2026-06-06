"""
Regression tests for the Discovery Push — the selection-only "extra push"
metric in evo13.py (shape + smooth intrinsic components folded into
``parsimony_fitness`` via its ``push`` argument, plus the unified novelty knob).

Guards the contract that the push:
  • is a BONUS — it only ever lowers (improves) the selection fitness, is
    clamped non-negative, and is bounded by PUSH_INTRINSIC_MAX so it can never
    invert a genuine loss ranking;
  • leaves every legacy three-argument ``parsimony_fitness`` call bit-for-bit
    unchanged (so the parsimony/MDL contracts in test_parsimony.py still hold);
  • SHAPE rewards rank (Spearman) agreement — including a monotone NON-linear
    warp that the affine-fitted MSE is blind to — and is sign-invariant;
  • SMOOTH rewards bounded extrapolation and drives a blow-up / non-finite
    extrapolation to ~0;
  • the combined intrinsic bonus is capped and fully disabled by PUSH_ENABLED.

Runnable two ways:
    python test_push.py        # prints a short report, exits non-zero on failure
    pytest test_push.py        # standard test discovery
"""
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import evo13

rng = np.random.default_rng(20260604)


def _with_push(fn, **knobs):
    """Run fn() with the given PUSH_* / PARSIMONY_* knobs, then restore."""
    keys = ("PUSH_ENABLED", "PUSH_SHAPE_WEIGHT", "PUSH_SMOOTH_WEIGHT",
            "PUSH_NOVELTY_WEIGHT", "PUSH_INTRINSIC_MAX", "PUSH_SHAPE_MAX_ROWS",
            "PUSH_SMOOTH_PROBE_K", "PUSH_SMOOTH_EXPAND",
            "PUSH_TAILS_WEIGHT", "PUSH_TAILS_FRAC",
            "PUSH_ANNEAL_ENABLED", "PUSH_ANNEAL_FLOOR", "PUSH_NOVELTY_GRADED",
            "PUSH_SHAPE_MAG_GATE",
            "PARSIMONY_MODE", "PARSIMONY_STRENGTH",
            "MDL_DATA_WEIGHT", "MDL_COMPLEXITY_BITS")
    saved = {k: getattr(evo13, k) for k in keys}
    try:
        for k, v in knobs.items():
            setattr(evo13, k, v)
        return fn()
    finally:
        for k, v in saved.items():
            setattr(evo13, k, v)


class _StubTree:
    """Minimal duck-typed tree: only ``evaluate`` is used by the smooth path."""
    def __init__(self, f):
        self._f = f

    def evaluate(self, X):
        return np.asarray(self._f(X), dtype=np.float64)


# ─────────────────────────── parsimony_fitness(push=) ────────────────────────

def test_push_is_a_bonus_and_legacy_unchanged():
    """push lowers fitness monotonically in BOTH modes; the 3-arg call equals
    push=0.0 exactly (so the parsimony contracts are untouched)."""
    def check():
        for mode, knobs in [("linear", {"PARSIMONY_STRENGTH": 0.01}),
                            ("mdl", {"MDL_DATA_WEIGHT": 3.0, "MDL_COMPLEXITY_BITS": 0.02})]:
            evo13.PARSIMONY_MODE = mode
            for k, v in knobs.items():
                setattr(evo13, k, v)
            base = evo13.parsimony_fitness(0.3, 40.0, 0.0)
            assert evo13.parsimony_fitness(0.3, 40.0, 0.0) == \
                   evo13.parsimony_fitness(0.3, 40.0, 0.0, 0.0), mode
            prev = base
            for p in [0.001, 0.01, 0.02]:
                f = evo13.parsimony_fitness(0.3, 40.0, 0.0, p)
                assert f < prev, f"{mode}: push did not improve fitness ({f} !< {prev})"
                assert abs((base - f) - p) < 1e-12, f"{mode}: push must subtract exactly"
                prev = f
    _with_push(check)


def test_negative_push_is_clamped_to_zero():
    """A negative push must NOT penalise (clamped to 0) — it is a reward only."""
    def check():
        evo13.PARSIMONY_MODE = "mdl"
        f0 = evo13.parsimony_fitness(0.3, 40.0, 0.0, 0.0)
        fneg = evo13.parsimony_fitness(0.3, 40.0, 0.0, -5.0)
        assert abs(f0 - fneg) < 1e-12, (f0, fneg)
    _with_push(check)


# ─────────────────────────────── SHAPE component ─────────────────────────────

def test_shape_rewards_monotone_nonlinear_warp():
    """A monotone but NON-linear warp of the target (right ordering, wrong
    shape — invisible to affine-fitted MSE) must score ≈1."""
    def check():
        y = np.sort(rng.uniform(0.1, 4.0, 1500))
        preds = np.exp(y)                       # strictly monotone, very non-linear
        s = evo13._push_shape_score(preds, y, evo13.PUSH_SHAPE_MAX_ROWS)
        assert s > 0.999, s
    _with_push(check)


def test_shape_is_sign_invariant():
    """An anti-correlated prediction is just as recoverable (affine sign flip),
    so |ρ| ⇒ a perfectly inverted model also scores ≈1."""
    def check():
        y = rng.normal(0, 1, 1200)
        s = evo13._push_shape_score(-3.0 * y + 7.0, y, evo13.PUSH_SHAPE_MAX_ROWS)
        assert s > 0.999, s
    _with_push(check)


def test_shape_zero_for_random_and_degenerate():
    """Random predictions ⇒ ≈0; a constant prediction or target ⇒ exactly 0."""
    def check():
        y = rng.normal(0, 1, 2000)
        assert evo13._push_shape_score(rng.normal(0, 1, 2000), y) < 0.2
        assert evo13._push_shape_score(np.full(2000, 3.0), y) == 0.0
        assert evo13._push_shape_score(y, np.full(2000, 3.0)) == 0.0
        assert evo13._push_shape_score(y[:4], y[:4]) == 0.0   # too few points
    _with_push(check)


def test_shape_subsample_is_bounded_and_consistent():
    """The capped subsample keeps a perfect-rank model at ≈1 even for N >> cap,
    and the score is stable across calls (deterministic subsample)."""
    def check():
        evo13.PUSH_SHAPE_MAX_ROWS = 512
        y = np.sort(rng.uniform(0, 10, 20000))
        preds = y ** 3 + 1.0
        s1 = evo13._push_shape_score(preds, y, evo13.PUSH_SHAPE_MAX_ROWS)
        s2 = evo13._push_shape_score(preds, y, evo13.PUSH_SHAPE_MAX_ROWS)
        assert s1 == s2 and s1 > 0.999, (s1, s2)
    _with_push(check)


# ─────────────────────────────── SMOOTH component ────────────────────────────

def test_smooth_rewards_bounded_penalises_blowup():
    """A bounded extrapolation scores high; an exp() blow-up scores ~0; a
    non-finite extrapolation scores exactly 0."""
    def check():
        X = np.linspace(-2.0, 2.0, 400).reshape(-1, 1)
        Xp = evo13._push_probe_inputs(X)
        assert Xp is not None

        # Bounded (sin) — stays within / near the in-sample band.
        bounded = _StubTree(lambda Z: np.sin(Z[:, 0]))
        in_lo, in_hi = -1.0, 1.0
        s_b = evo13._push_smooth_score(bounded, Xp, 1.0, 0.0, in_lo, in_hi)
        assert s_b > 0.7, s_b

        # Blow-up (steep exp) — leaves the in-sample band far behind.
        blow = _StubTree(lambda Z: np.exp(4.0 * Z[:, 0]))
        in_lo, in_hi = float(np.exp(-8.0)), float(np.exp(8.0))
        s_x = evo13._push_smooth_score(blow, Xp, 1.0, 0.0, in_lo, in_hi)
        assert s_x < 0.2, s_x
        assert s_x < s_b

        # Non-finite extrapolation ⇒ the strongest "won't generalise" signal.
        nan_tree = _StubTree(lambda Z: np.full(Z.shape[0], np.inf))
        assert evo13._push_smooth_score(nan_tree, Xp, 1.0, 0.0, -1.0, 1.0) == 0.0
    _with_push(check)


def test_smooth_linear_is_only_mildly_penalised():
    """A gentle linear trend leaves the band slowly ⇒ score stays high (it is
    blow-ups, not benign growth, that we push away from)."""
    def check():
        X = np.linspace(0.0, 10.0, 500).reshape(-1, 1)
        Xp = evo13._push_probe_inputs(X)
        line = _StubTree(lambda Z: Z[:, 0])
        s = evo13._push_smooth_score(line, Xp, 1.0, 0.0, 0.0, 10.0)
        assert s > 0.85, s
    _with_push(check)


# ──────────────────────────── compute_push_intrinsic ─────────────────────────

def test_intrinsic_bonus_capped_positive_and_gated():
    """The combined shape+smooth+tails bonus is positive for a good model, never
    exceeds PUSH_INTRINSIC_MAX, and collapses to 0 when disabled / zero-weight."""
    def check():
        X = np.linspace(-3.0, 3.0, 800).reshape(-1, 1)
        y = X[:, 0].copy()
        identity = _StubTree(lambda Z: Z[:, 0])
        preds = y.copy()                                  # perfect shape, bounded

        b = evo13.compute_push_intrinsic(preds, y, identity, X, 1.0, 0.0)
        assert 0.0 < b <= evo13.PUSH_INTRINSIC_MAX + 1e-12, b
        # shape≈1, tails≈1 and smooth high ⇒ raw sum exceeds the cap ⇒ capped.
        assert abs(b - evo13.PUSH_INTRINSIC_MAX) < 1e-9, b

        # Disabled master switch ⇒ no bonus.
        evo13.PUSH_ENABLED = False
        assert evo13.compute_push_intrinsic(preds, y, identity, X, 1.0, 0.0) == 0.0
        evo13.PUSH_ENABLED = True

        # Zero ALL intrinsic weights ⇒ no bonus (the gating contract now spans
        # the third, TAILS, weight too).
        evo13.PUSH_SHAPE_WEIGHT = 0.0
        evo13.PUSH_SMOOTH_WEIGHT = 0.0
        evo13.PUSH_TAILS_WEIGHT = 0.0
        assert evo13.compute_push_intrinsic(preds, y, identity, X, 1.0, 0.0) == 0.0
    _with_push(check, PUSH_SHAPE_WEIGHT=0.02, PUSH_SMOOTH_WEIGHT=0.01,
              PUSH_TAILS_WEIGHT=0.008, PUSH_INTRINSIC_MAX=0.025)


def test_intrinsic_orders_good_shape_above_bad_at_equal_loss():
    """End-to-end intent: at identical loss+complexity, the model that captures
    the target's shape gets a strictly better selection fitness."""
    def check():
        X = np.linspace(0.1, 5.0, 600).reshape(-1, 1)
        y = np.sort(rng.uniform(0.1, 5.0, 600))
        good = _StubTree(lambda Z: Z[:, 0])               # tracks ordering
        good_preds = np.sort(y) + rng.normal(0, 0.01, y.size)
        bad = _StubTree(lambda Z: np.zeros(Z.shape[0]))   # no structure
        bad_preds = rng.permutation(y)                    # shuffled ⇒ no rank info

        pg = evo13.compute_push_intrinsic(good_preds, y, good, X, 1.0, 0.0)
        pb = evo13.compute_push_intrinsic(bad_preds, y, bad, X, 0.0, float(y.mean()))
        fg = evo13.parsimony_fitness(0.5, 30.0, 0.0, push=pg)
        fb = evo13.parsimony_fitness(0.5, 30.0, 0.0, push=pb)
        assert pg > pb and fg < fb, (pg, pb, fg, fb)
    _with_push(check, PARSIMONY_MODE="mdl", MDL_DATA_WEIGHT=3.0,
               MDL_COMPLEXITY_BITS=0.02)


def test_novelty_weight_defaults_preserve_legacy_magnitude():
    """The unified novelty knob must default to the historical 0.005 / 0.015."""
    assert evo13.PUSH_NOVELTY_WEIGHT == 0.005
    assert 3.0 * evo13.PUSH_NOVELTY_WEIGHT == 0.015


# ─────────────────────────────── TAILS signal ───────────────────────────────

def test_tails_rewards_extreme_ordering_and_is_robust():
    """TAILS scores perfect ordering 1, a degenerate input 0, and — the point of
    the signal — collapses for a model that tracks the dense bulk but loses the
    ordering of the informative extremes (a saturating / clipped tail), a loss
    full-sample SHAPE largely dilutes away."""
    n = 1000
    y = np.sort(rng.uniform(-4.0, 4.0, n))

    # Perfect ⇒ 1; constant / too-few ⇒ 0 (reuses the shape-scorer guards).
    assert evo13._push_tails_score(y.copy(), y, 0.2) > 0.999
    assert evo13._push_tails_score(np.full(n, 2.0), y, 0.2) == 0.0
    assert evo13._push_tails_score(y[:8], y[:8], 0.2) == 0.0

    # A model perfect in the central bulk but SATURATED (collapsed to the median)
    # on the top + bottom 5%: full-sample SHAPE stays high (the dense middle is
    # still perfectly ordered), but TAILS — looking only at the extremes —
    # collapses to 0, because the saturated extremes carry no ordering at all.
    k = int(0.05 * n)
    sat = y.copy()
    sat[:k] = np.median(y)
    sat[-k:] = np.median(y)
    s_shape = evo13._push_shape_score(sat, y)
    s_tails = evo13._push_tails_score(sat, y, 0.05)
    assert s_shape > 0.7, s_shape                       # bulk still ordered
    assert s_tails == 0.0, s_tails                      # extremes carry no order
    assert s_tails < s_shape - 0.3, (s_tails, s_shape)  # TAILS sees what SHAPE hides


# ─────────────────────── magnitude-gap gating (shape→magnitude) ──────────────

def test_magnitude_gap_is_one_minus_clipped_r2():
    """The gap is 0 for a perfect fit, 1 for a mean-predictor (R²=0), monotone in
    fit quality, and 1.0 on degenerate inputs (no down-weighting)."""
    y = np.linspace(-2.0, 3.0, 600)
    assert evo13._magnitude_gap(y.copy(), y) < 1e-9                       # perfect ⇒ 0
    assert abs(evo13._magnitude_gap(np.full_like(y, y.mean()), y) - 1.0) < 1e-9  # mean ⇒ 1
    prev = None
    for frac in [0.0, 0.3, 0.6, 0.9, 1.0]:        # increasing variance explained
        g = evo13._magnitude_gap(y.mean() + (y - y.mean()) * frac, y)
        assert 0.0 - 1e-12 <= g <= 1.0 + 1e-12, g
        if prev is not None:
            assert g <= prev + 1e-9, (frac, g, prev)   # better fit ⇒ smaller gap
        prev = g
    assert evo13._magnitude_gap(y, np.full_like(y, 4.0)) == 1.0           # constant target
    assert evo13._magnitude_gap(y[:1], y[:1]) == 1.0                      # too few points


def test_shape_mag_gate_focuses_push_on_unrealised_magnitude():
    """With the gate ON, two models with the SAME ordering (so identical SHAPE/
    TAILS) are no longer rewarded equally: the one whose magnitude is still wrong
    keeps (most of) its push while the one the affine has already realised has its
    shape/tails reward gated toward 0 — the literal 'shape first, then magnitude'
    behaviour.  With the gate OFF they are identical (legacy)."""
    def check():
        X = np.linspace(0.0, 1.0, 800).reshape(-1, 1)
        y = X[:, 0].copy()
        tree = _StubTree(lambda Z: Z[:, 0])
        realised   = y.copy()       # ρ=1, R²=1  → magnitude already there
        unrealised = y ** 3         # ρ=1, R²<1  → right shape, wrong magnitude

        # SMOOTH off so the (ungated) smooth term doesn't mask the contrast.
        evo13.PUSH_SMOOTH_WEIGHT = 0.0

        evo13.PUSH_SHAPE_MAG_GATE = False
        off_real   = evo13.compute_push_intrinsic(realised,   y, tree, X, 1.0, 0.0)
        off_unreal = evo13.compute_push_intrinsic(unrealised, y, tree, X, 1.0, 0.0)
        assert abs(off_real - off_unreal) < 1e-9, (off_real, off_unreal)  # same ordering

        evo13.PUSH_SHAPE_MAG_GATE = True
        on_real   = evo13.compute_push_intrinsic(realised,   y, tree, X, 1.0, 0.0)
        on_unreal = evo13.compute_push_intrinsic(unrealised, y, tree, X, 1.0, 0.0)
        assert on_real < 1e-9, on_real                       # R²=1 ⇒ gated to ~0
        assert on_unreal > on_real + 1e-6, (on_unreal, on_real)
        assert on_real < off_real - 1e-6, (on_real, off_real)  # gate strictly reduced it
    _with_push(check, PUSH_SHAPE_WEIGHT=0.02, PUSH_TAILS_WEIGHT=0.008,
               PUSH_SMOOTH_WEIGHT=0.0, PUSH_INTRINSIC_MAX=0.05)


# ───────────────────────────── push annealing ───────────────────────────────

def test_push_anneal_scale_shape():
    """push_anneal_scale rides 1.0 → floor monotonically, clamps its argument,
    and is a flat 1.0 when disabled."""
    def check():
        s0 = evo13.push_anneal_scale(0.0)
        s1 = evo13.push_anneal_scale(1.0)
        assert abs(s0 - 1.0) < 1e-12, s0
        assert abs(s1 - evo13.PUSH_ANNEAL_FLOOR) < 1e-12, s1
        prev = None
        for f in np.linspace(0.0, 1.0, 11):
            v = evo13.push_anneal_scale(f)
            if prev is not None:
                assert v <= prev + 1e-12, (f, v, prev)   # non-increasing
            prev = v
        assert evo13.push_anneal_scale(-5.0) == s0       # clamped
        assert evo13.push_anneal_scale(5.0) == s1
    _with_push(check, PUSH_ANNEAL_ENABLED=True, PUSH_ANNEAL_FLOOR=0.15)

    def off():
        assert evo13.push_anneal_scale(0.0) == 1.0
        assert evo13.push_anneal_scale(1.0) == 1.0
    _with_push(off, PUSH_ANNEAL_ENABLED=False)


def test_push_anneal_scales_the_fitness_bonus():
    """The live _PUSH_ANNEAL_SCALE multiplies the push inside parsimony_fitness,
    so a late-epoch (small-scale) push perturbs the ranking less than an early
    one — and the default scale of 1.0 leaves legacy behaviour unchanged."""
    saved = evo13._PUSH_ANNEAL_SCALE
    try:
        base = evo13.parsimony_fitness(0.3, 40.0, 0.0, push=0.0)
        evo13._PUSH_ANNEAL_SCALE = 1.0
        full = base - evo13.parsimony_fitness(0.3, 40.0, 0.0, push=0.02)
        evo13._PUSH_ANNEAL_SCALE = 0.25
        faded = base - evo13.parsimony_fitness(0.3, 40.0, 0.0, push=0.02)
        assert abs(full - 0.02) < 1e-12, full           # scale 1.0 ⇒ legacy
        assert abs(faded - 0.25 * 0.02) < 1e-12, faded   # scale 0.25 ⇒ ¼ effect
        assert 0.0 < faded < full
    finally:
        evo13._PUSH_ANNEAL_SCALE = saved


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    import sys
    failures = 0
    for t in _TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} passed")
    sys.exit(1 if failures else 0)
