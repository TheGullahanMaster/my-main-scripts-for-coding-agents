"""
Regression tests for the degenerate-output guard in evo13.py.

A model whose EXPORTED prediction is (near-)constant explains none of a varying
target, so calculate_fitness must score it like a constant predictor — loss ≈
the median-constant baseline, R² = 0 — never the spurious loss≈0 / R²≈1 it used
to report.  Covers:

  • a pure-constant tree (2·(x/x)-style):  loss is the median baseline, R²==0;
  • the DIFFERENTIABLE_BRANCHING "dead branch" — IF(always-false, expr, C) whose
    soft if_else leaks a varying sliver of `expr` so the smooth output the loss
    sees fits the target (R²≈1) even though the exported HARD model is the
    constant C — now scored as degenerate (loss==baseline, R²==0);
  • evaluate(force_hard=True) recovers the hard export decision;
  • _fit_affine refuses to manufacture a slope from machine-noise variance on a
    large-magnitude near-constant prediction.

Runnable two ways:
    python test_degenerate_guard.py     # prints a short report, non-zero on fail
    pytest test_degenerate_guard.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import evo13

evo13.set_ops_mode(True)   # populate the SAFE op dispatch tables
rng = np.random.default_rng(20260608)

REG = 0   # any type_code != 6 is regression


def _const_tree(n_features, feat_names, value):
    """A tree whose output is the constant `value` for every row."""
    return evo13._build_seed(n_features, feat_names, [('const', 0, 0, value)])


def _dead_branch_tree(feat_names):
    """if_else(cond=0.0  [always false], true=x0, else=C=7.0).

    HARD export → C (constant).  Under DIFFERENTIABLE_BRANCHING the soft if_else
    blends in a constant fraction of x0, so the smooth output VARIES with x0 and
    the affine can rescale it onto a y=x0 target (the leak this guard kills).
    """
    return evo13._build_seed(1, feat_names, [
        ('const', 0, 0, 0.0),    # node@1: condition value 0.0  (< 0.5 ⇒ false)
        ('const', 0, 0, 7.0),    # node@2: else-branch constant C
        ('if_else', 1, 0, 0.0, 2),  # node@3: if_else(cond@1, x0@0, C@2)
    ], out_offset=2)


def _with(**knobs):
    saved = {k: getattr(evo13, k) for k in knobs}
    for k, v in knobs.items():
        setattr(evo13, k, v)
    return saved


def _restore(saved):
    for k, v in saved.items():
        setattr(evo13, k, v)


def _enable_if_else():
    """Force `if_else` into the active ternary op set (off by default) and return
    the previous (IF_ELSE_ENABLED, OPS_TERNARY, OPS_TERNARY_SET) for restoration."""
    cls = evo13.CGPEquation
    prev = (evo13.IF_ELSE_ENABLED, list(cls.OPS_TERNARY), set(cls.OPS_TERNARY_SET))
    evo13.IF_ELSE_ENABLED = True
    tern = list(cls.OPS_TERNARY)
    if 'if_else' not in tern:
        tern = ['if_else'] + tern
    cls.OPS_TERNARY = tern
    cls.OPS_TERNARY_SET = set(tern)
    return prev


def _restore_if_else(prev):
    en, tern, tset = prev
    evo13.IF_ELSE_ENABLED = en
    evo13.CGPEquation.OPS_TERNARY = tern
    evo13.CGPEquation.OPS_TERNARY_SET = tset


def test_pure_constant_scores_as_median_baseline():
    """A constant tree gets the median-constant loss and R²=0, never loss≈0/R²≈1."""
    x = np.linspace(-3, 3, 400).reshape(-1, 1)
    y = 2.5 * x[:, 0] + 1.0              # genuinely varying target
    ind = _const_tree(1, ["x0"], 5.0)    # outputs 5.0 everywhere
    ind.affine_fitted = False
    ind.calculate_fitness(x, y, REG)
    baseline = evo13.regression_loss(np.full_like(y, np.median(y)), y)
    assert ind.r2 == 0.0, f"constant R² should be 0, got {ind.r2}"
    assert abs(ind.loss - baseline) < 1e-6, (ind.loss, baseline)
    # It must NOT look near-perfect and must stay well under the old +10 wall.
    assert ind.loss > 0.1, ind.loss
    assert ind.loss < evo13.HOF_LOSS_CEILING, ind.loss


def test_force_hard_recovers_export_decision():
    """evaluate(force_hard=True) on the dead branch is the constant else-branch."""
    saved = _with(DIFFERENTIABLE_BRANCHING=True)
    prev = _enable_if_else()
    try:
        ind = _dead_branch_tree(["x0"])
        x = np.linspace(-3, 3, 200).reshape(-1, 1)
        soft = ind.tree.evaluate(x)                    # smooth: varies with x0
        hard = ind.tree.evaluate(x, force_hard=True)   # hard: constant 7.0
        assert ind.tree.has_conditional() is True
        assert np.std(hard) < 1e-9, np.std(hard)
        assert np.allclose(hard, 7.0), hard[:3]
        assert np.std(soft) > 1e-9, "soft blend should leak variance"
    finally:
        _restore_if_else(prev)
        _restore(saved)


def test_differentiable_dead_branch_is_degenerate():
    """The soft-branch leak no longer earns loss≈0 / R²≈1 — it scores as baseline."""
    saved = _with(DIFFERENTIABLE_BRANCHING=True)
    prev = _enable_if_else()
    try:
        x = np.linspace(-3, 3, 400).reshape(-1, 1)
        y = x[:, 0].copy()                 # smooth leak (∝ x0) could fit this
        ind = _dead_branch_tree(["x0"])
        ind.affine_fitted = False
        ind.calculate_fitness(x, y, REG)
        baseline = evo13.regression_loss(np.full_like(y, np.median(y)), y)
        assert ind.r2 == 0.0, f"dead-branch R² should be 0, got {ind.r2}"
        assert abs(ind.loss - baseline) < 1e-6, (ind.loss, baseline)
    finally:
        _restore_if_else(prev)
        _restore(saved)


def test_affine_rejects_machine_noise_slope():
    """_fit_affine treats a large-magnitude near-constant prediction as constant."""
    p = 1e6 + 1e-4 * rng.standard_normal(500)   # var ≈ 1e-8 (> 1e-10) but CV ≈ 1e-10
    y = rng.standard_normal(500)
    a, b, fitted = evo13._fit_affine(p, y)
    assert fitted is False, "near-constant (relative) prediction must stay unlocked"
    assert (a, b) == (1.0, 0.0)


def test_genuine_model_unaffected():
    """A real, varying model keeps its true (good) loss / R² — guard never fires."""
    x = np.linspace(-3, 3, 400).reshape(-1, 1)
    y = 2.5 * x[:, 0] + 1.0
    # tree = x0  (the affine finishes the 2.5·x0 + 1 fit)
    ind = evo13._build_seed(1, ["x0"], [('+', 0, 0)])   # x0 + x0 ; affine rescales
    ind.affine_fitted = False
    ind.calculate_fitness(x, y, REG)
    assert ind.r2 > 0.99, ind.r2
    assert ind.loss < 0.05, ind.loss


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(1 if _run() else 0)
