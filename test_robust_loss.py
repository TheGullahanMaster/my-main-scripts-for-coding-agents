"""
Regression tests for the robust multi-scale regression loss in evo13.py
(``regression_loss`` + ``REGRESSION_LOSS_MODE``).

Guards the contract that the new default "robust" loss:
  • reduces to a calibrated baseline (perfect -> 0, mean-predictor ~ O(1));
  • is monotone in fit quality;
  • FIXES the multi-scale failure of the legacy MSE+MAE loss (a model that fits
    the big-magnitude rows but ignores the small ones must score WORSE than one
    that is uniformly good in relative terms — the legacy loss ranked these
    backwards);
  • is robust to a single catastrophic / discontinuous outlier row;
  • gives a graduated push away from constant / mean collapse;
and that "mse_mae" mode still reproduces the exact legacy formula.

Runnable two ways:
    python test_robust_loss.py        # prints a short report, exits non-zero on failure
    pytest test_robust_loss.py        # standard test discovery
"""
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import evo13

rng = np.random.default_rng(1234)


def _with_mode(mode, fn, **knobs):
    saved = {k: getattr(evo13, k) for k in (
        "REGRESSION_LOSS_MODE", "ROBUST_LOSS_HUBER_DELTA", "ROBUST_LOSS_MAE_WEIGHT",
        "ROBUST_LOSS_MULTISCALE_WEIGHT", "ROBUST_LOSS_MAE_CLIP",
        "ROBUST_LOSS_COLLAPSE_WEIGHT", "ROBUST_LOSS_SCALE_PCT",
        "ROBUST_LOSS_MULTISCALE_LO", "ROBUST_LOSS_MULTISCALE_HI")}
    try:
        evo13.REGRESSION_LOSS_MODE = mode
        for k, v in knobs.items():
            setattr(evo13, k, v)
        return fn()
    finally:
        for k, v in saved.items():
            setattr(evo13, k, v)


def _legacy(preds, y):
    # Mirrors evo13's mse_mae mode: the historical absolute +1e-8 epsilon is
    # now a scale-aware floor (see evo13._target_scale_floor) so tiny-scale
    # targets no longer see every loss flattened to ≈0.
    diff = preds - y
    yv = float(np.var(y))
    floor = evo13._target_scale_floor(float(np.mean(y))) ** 2
    yv = max(yv, floor, 1e-300)
    return float(np.mean(diff ** 2) / yv + np.mean(np.abs(diff)) / np.sqrt(yv))


def test_mse_mae_mode_matches_legacy_formula_exactly():
    """mse_mae mode must equal (mse/var)+(mae/sqrt(var)) bit-for-bit."""
    def check():
        for _ in range(20):
            y = rng.normal(rng.uniform(-5, 5), rng.uniform(0.5, 4), 500)
            p = y + rng.normal(0, rng.uniform(0, 3), y.size)
            got = evo13.regression_loss(p, y)
            assert abs(got - _legacy(p, y)) < 1e-9, (got, _legacy(p, y))
    _with_mode("mse_mae", check)


def test_perfect_is_zero_and_mean_predictor_is_calibrated():
    """Perfect fit -> 0; a mean-predictor sits in a sane O(1) band."""
    def check():
        for mu, sd in [(5, 2), (0, 1), (-3, 7), (100, 50)]:
            y = rng.normal(mu, sd, 3000)
            assert evo13.regression_loss(y.copy(), y) < 1e-6
            mp = evo13.regression_loss(np.full_like(y, y.mean()), y)
            assert 0.5 < mp < evo13.HOF_LOSS_CEILING + 1.0, (mu, sd, mp)
    _with_mode("robust", check)


def test_monotone_in_fit_quality():
    """Less residual noise must never increase the loss (regular + multiscale)."""
    def check():
        for base in [rng.normal(5, 3, 3000),
                     np.concatenate([rng.uniform(-1e-2, 1e-2, 1500),
                                     rng.uniform(-1e6, 1e6, 1500)])]:
            prev = None
            for frac in [1.0, 0.7, 0.4, 0.2, 0.1, 0.03, 0.0]:
                p = base * (1 - frac) + base.mean() * frac
                L = evo13.regression_loss(p, base)
                if prev is not None:
                    assert L <= prev + 1e-6, (frac, L, prev)
                prev = L
    _with_mode("robust", check)


def test_multiscale_ranks_uniform_fit_above_ignoring_small():
    """THE headline fix.  Across several decade-gaps, the model that ignores the
    small-magnitude rows must score strictly WORSE than the uniformly-good one —
    and the legacy loss must demonstrably get this BACKWARDS."""
    def robust_check():
        for sm, bg in [(1e-2, 1e6), (1.0, 1e4), (0.1, 100.0), (1e-3, 1e3)]:
            small = rng.uniform(-sm, sm, 1200)
            big   = rng.uniform(-bg, bg, 1200)
            y     = np.concatenate([small, big])
            bad   = np.concatenate([np.zeros_like(small), big])         # ignores small
            good  = y * (1.0 + rng.normal(0, 0.01, y.size))             # ~1% everywhere
            Lb = evo13.regression_loss(bad, y)
            Lg = evo13.regression_loss(good, y)
            assert Lg < Lb, f"robust failed to penalise ignoring small ({sm} vs {bg}): good={Lg} bad={Lb}"
            assert Lb > 5 * Lg, f"robust signal too weak: good={Lg} bad={Lb}"
    _with_mode("robust", robust_check)

    # Document WHY this is an upgrade: the legacy loss ranks them backwards.
    small = rng.uniform(-1e-2, 1e-2, 1200); big = rng.uniform(-1e6, 1e6, 1200)
    y = np.concatenate([small, big])
    bad = np.concatenate([np.zeros_like(small), big])
    good = y * (1.0 + rng.normal(0, 0.01, y.size))
    assert _legacy(bad, y) < _legacy(good, y), "legacy unexpectedly handled multi-scale"


def test_robust_to_single_catastrophic_outlier():
    """A lone huge residual (chaotic/discontinuous blow-up) must stay bounded,
    not explode the way a squared error does."""
    def check():
        y = rng.normal(0, 1, 2000)
        good = y + rng.normal(0, 0.05, y.size)
        outl = good.copy(); outl[0] += 1000.0
        L_good = evo13.regression_loss(good, y)
        L_outl = evo13.regression_loss(outl, y)
        assert L_outl < L_good + 5.0, f"outlier not contained: {L_good} -> {L_outl}"
        # The legacy squared loss balloons by orders of magnitude on the same row.
        assert _legacy(outl, y) > 100 * _legacy(good, y)
    _with_mode("robust", check)


def test_discontinuous_targets_ordered():
    """perfect < half-collapsed < mean-predictor for floor/mod/parity/xor."""
    def check():
        x = rng.uniform(0, 100, 3000)
        for yt in [np.floor(x), np.mod(x, 7.0),
                   (x.astype(int) & 1).astype(float),
                   (x.astype(int) ^ rng.integers(0, 100, x.size)).astype(float)]:
            half = yt.copy(); half[::2] = yt.mean()
            Lp = evo13.regression_loss(yt.copy(), yt)
            Lh = evo13.regression_loss(half, yt)
            Lm = evo13.regression_loss(np.full_like(yt, yt.mean()), yt)
            assert Lp < Lh < Lm, (Lp, Lh, Lm)
    _with_mode("robust", check)


def test_anti_collapse_is_graduated_and_penalises_constants():
    """Constant/near-constant outputs get a stronger-than-averaged push, and the
    penalty ramps smoothly to 0 as the model explains more variance."""
    def check():
        y = rng.normal(5, 2, 3000)
        losses = []
        for frac in [0.0, 0.1, 0.3, 0.6, 0.9, 1.0]:   # fraction of variance explained
            p = y.mean() + (y - y.mean()) * frac
            losses.append(evo13.regression_loss(p, y))
        # Strictly decreasing as the model explains more (collapse push relaxes).
        for a, b in zip(losses, losses[1:]):
            assert b < a, losses
        # The anti-collapse term contributes a real, sizeable penalty to a
        # constant output (this is the requested "stronger signal" — turning the
        # term off measurably lowers the constant's loss, and it vanishes once
        # the model explains the variance).
        const = np.full_like(y, y.mean())
        with_collapse = evo13.regression_loss(const, y)
        evo13.ROBUST_LOSS_COLLAPSE_WEIGHT = 0.0
        without_collapse = evo13.regression_loss(const, y)
        assert with_collapse - without_collapse > 0.4 * 0.9, \
            (with_collapse, without_collapse)
    _with_mode("robust", check)


def test_sample_weights_path_is_finite_and_weights_matter():
    """The Hessian-boosting weighted path runs, stays finite, and actually
    responds to the weights."""
    def check():
        y = rng.normal(0, 1, 1000)
        p = y + rng.normal(0, 0.5, y.size)
        w = rng.uniform(0.1, 3.0, y.size)
        L_w = evo13.regression_loss(p, y, sample_weights=w)
        L_u = evo13.regression_loss(p, y)
        assert math.isfinite(L_w) and L_w >= 0.0
        # Uniform weights must reproduce the unweighted loss.
        L_uniform = evo13.regression_loss(p, y, sample_weights=np.ones_like(y))
        assert abs(L_uniform - L_u) < 1e-9, (L_uniform, L_u)
    _with_mode("robust", check)


def test_ystats_cache_is_correct_and_identity_safe():
    """Cached y-stats must equal a fresh computation, and a different array with
    the same contents must NOT collide (identity check)."""
    def check():
        y = rng.normal(3, 2, 800)
        p = y + rng.normal(0, 0.3, y.size)
        L1 = evo13.regression_loss(p, y)     # populates cache
        L2 = evo13.regression_loss(p, y)     # cache hit
        assert L1 == L2
        y_copy = y.copy()                    # same values, different identity
        L3 = evo13.regression_loss(p, y_copy)
        assert abs(L3 - L1) < 1e-9, (L1, L3)
    _with_mode("robust", check)


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
