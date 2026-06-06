"""
Regression tests for the affine magnitude-finetune fit in evo13.py
(``_fit_affine`` + AFFINE_ROBUST_FIT): the closed-form rescaling y ≈ a·f(x) + b
that lets the search focus on SHAPE while a post-fit handles the MAGNITUDE.

Guards the contract that the fit:
  • reproduces the exact textbook OLS slope/intercept when the robust refit is
    off, and under the pure-squared "mse_mae" loss (where OLS is already optimal);
  • returns the identity UNLOCKED (fitted=False) for a (near-)constant prediction,
    so the caller refits after a structural mutation instead of freezing the
    individual;
  • robustly refines OLS — the IRLS fit is NEVER worse than OLS by the robust loss
    it is judged on, and on a heavy-tailed target it is strictly better (it
    down-weights the outlier rows OLS is dragged off by);
  • is deterministic.

Runnable two ways:
    python test_affine.py        # prints a short report, exits non-zero on failure
    pytest test_affine.py        # standard test discovery
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import evo13

rng = np.random.default_rng(20260606)


def _with_affine(fn, **knobs):
    """Run fn() with the given AFFINE_* / loss knobs, then restore."""
    keys = ("AFFINE_ROBUST_FIT", "AFFINE_ROBUST_FIT_ITERS",
            "REGRESSION_LOSS_MODE", "ROBUST_LOSS_HUBER_DELTA")
    saved = {k: getattr(evo13, k) for k in keys}
    try:
        for k, v in knobs.items():
            setattr(evo13, k, v)
        return fn()
    finally:
        for k, v in saved.items():
            setattr(evo13, k, v)


def _ols(p, y):
    pm, ym = float(p.mean()), float(y.mean())
    a = float(np.mean((p - pm) * (y - ym)) / np.var(p))
    return a, ym - a * pm


def test_ols_seed_matches_closed_form():
    """With the robust refit off, _fit_affine == textbook OLS and locks (True)."""
    def check():
        for _ in range(25):
            p = rng.normal(0, rng.uniform(0.5, 3.0), 400)
            y = rng.uniform(-2, 2) * p + rng.uniform(-5, 5) + rng.normal(0, 0.3, p.size)
            a, b, fitted = evo13._fit_affine(p, y)
            ea, eb = _ols(p, y)
            assert fitted is True, fitted
            assert abs(a - ea) < 1e-9 and abs(b - eb) < 1e-9, (a, ea, b, eb)
    _with_affine(check, AFFINE_ROBUST_FIT=False)


def test_mse_mae_mode_is_pure_ols():
    """Under the pure-squared legacy loss, OLS is already optimal ⇒ no IRLS even
    with AFFINE_ROBUST_FIT on."""
    def check():
        p = rng.normal(0, 2, 500)
        y = 3.0 * p - 1.0 + rng.normal(0, 0.5, p.size)
        a, b, fitted = evo13._fit_affine(p, y)
        ea, eb = _ols(p, y)
        assert fitted and abs(a - ea) < 1e-9 and abs(b - eb) < 1e-9, (a, ea, b, eb)
    _with_affine(check, REGRESSION_LOSS_MODE="mse_mae",
                 AFFINE_ROBUST_FIT=True, AFFINE_ROBUST_FIT_ITERS=3)


def test_iters_zero_is_pure_ols():
    """Zero IRLS passes ⇒ the OLS seed is returned unchanged."""
    def check():
        p = rng.normal(0, 2, 500)
        y = 2.0 * p + 4.0 + rng.normal(0, 1.0, p.size)
        a, b, _ = evo13._fit_affine(p, y)
        ea, eb = _ols(p, y)
        assert abs(a - ea) < 1e-9 and abs(b - eb) < 1e-9
    _with_affine(check, AFFINE_ROBUST_FIT=True, AFFINE_ROBUST_FIT_ITERS=0,
                 REGRESSION_LOSS_MODE="robust")


def test_near_constant_returns_identity_unlocked():
    """A (near-)constant prediction has no identifiable slope ⇒ (1, 0, False)."""
    def check():
        y = rng.normal(0, 1, 300)
        assert evo13._fit_affine(np.full(300, 2.5), y) == (1.0, 0.0, False)
        a, b, fitted = evo13._fit_affine(2.5 + rng.normal(0, 1e-7, 300), y)
        assert fitted is False and a == 1.0 and b == 0.0, (a, b, fitted)
    _with_affine(check)


def test_robust_fit_never_worse_and_better_on_outliers():
    """IRLS refit ≤ OLS by the robust loss everywhere (it returns min{OLS,IRLS}),
    and STRICTLY better when a few heavy-tail rows drag the OLS line off."""
    def check():
        p = rng.normal(0, 2, 800)
        y = 2.0 * p + 1.0 + rng.normal(0, 0.4, p.size)

        # Clean data: the robust fit must not be worse than OLS by the loss.
        a_r, b_r, _ = evo13._fit_affine(p, y)
        a_o, b_o = _ols(p, y)
        assert evo13.regression_loss(a_r * p + b_r, y) \
            <= evo13.regression_loss(a_o * p + b_o, y) + 1e-9

        # Heavy-tailed: a few percent of rows get a huge perturbation.  The
        # guaranteed inequality holds every draw; across several draws the robust
        # fit must STRICTLY beat OLS at least once (it usually wins every time).
        strict_wins = 0
        for _ in range(8):
            y2 = y.copy()
            k = max(1, y2.size // 25)
            y2[rng.choice(y2.size, k, replace=False)] += rng.normal(0, 100.0, k)
            ar, br, _ = evo13._fit_affine(p, y2)
            ao, bo = _ols(p, y2)
            Lr = evo13.regression_loss(ar * p + br, y2)
            Lo = evo13.regression_loss(ao * p + bo, y2)
            assert Lr <= Lo + 1e-9, (Lr, Lo)
            if Lr < Lo - 1e-9:
                strict_wins += 1
        assert strict_wins >= 1, "robust IRLS never beat OLS on heavy-tailed targets"
    _with_affine(check, AFFINE_ROBUST_FIT=True, AFFINE_ROBUST_FIT_ITERS=3,
                 REGRESSION_LOSS_MODE="robust")


def test_deterministic():
    """The fit is a pure function of (preds, y)."""
    def check():
        p = rng.normal(0, 2, 600)
        y = 1.5 * p + rng.normal(0, 1.0, p.size)
        assert evo13._fit_affine(p, y) == evo13._fit_affine(p, y)
    _with_affine(check, AFFINE_ROBUST_FIT=True, AFFINE_ROBUST_FIT_ITERS=3,
                 REGRESSION_LOSS_MODE="robust")


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
