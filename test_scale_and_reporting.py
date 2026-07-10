"""
Regression tests for the scale-free metrics + honest-reporting fixes in evo13.py.

Covers the bug cluster where flat-constant (or junk) equations were reported
with R²=1 / loss=0 and where targets far from O(1) scale broke the metrics:

  • ABSOLUTE epsilons (+1e-8) in regression_loss / R² flattened every loss to
    ≈0 and every R² to ≈1 on tiny-scale targets (std(y) ≲ 1e-4) — scores must
    now be invariant to rescaling the target;
  • the degenerate-output guard skipped targets with std(y) ≤ 1e-9 even when
    they genuinely varied — a constant model must score R²=0 there too;
  • the fitness-cache fingerprint rounded constants to 4 significant figures,
    so trees differing beyond that inherited each other's loss/R² wholesale;
  • simplify_cgp_tree folded |c| < 1e-9 multipliers to literal zero,
    collapsing legitimately fitted models into flat constants;
  • calculate_fitness's exception path kept a stale (possibly perfect) loss
    on a broken tree — and the HoF is keyed on loss;
  • str(tree) printed constants as :.4f, so every |c| < 5e-5 displayed as
    "0.0000" and fitted equations *read* as flat constants;
  • SymPy-simplified expressions were adopted for export after only a syntax
    check — _sympy_expr_diverges must catch a semantic mismatch.

Runnable two ways:
    python test_scale_and_reporting.py     # prints a short report, non-zero on fail
    pytest test_scale_and_reporting.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import evo13

evo13.set_ops_mode(True)   # populate the SAFE op dispatch tables
rng = np.random.default_rng(20260710)

REG = 0   # any type_code != 6 is regression


def _const_tree(value):
    return evo13._build_seed(1, ["x0"], [('const', 0, 0, value)])


def _fresh(specs, n_features=1, names=None):
    ind = evo13._build_seed(n_features, names or ["x0"], specs)
    ind.affine_fitted = False
    return ind


def test_scores_invariant_to_target_scale():
    """Rescaling y must not change loss or R² — for constants OR junk models."""
    x = np.linspace(-3, 3, 500).reshape(-1, 1)
    base = np.sin(2 * x[:, 0])
    ref = {}
    for scale in [1.0, 1e-3, 1e-6, 1e-9, 1e-12, 1e6]:
        y = scale * base
        c = _fresh([('const', 0, 0, 5.0)])
        c.calculate_fitness(x, y, REG, use_cache=False)
        j = _fresh([('tanh', 0, 0), ('+', 1, 0)])   # tanh(x)+x — wrong shape
        j.calculate_fitness(x, y, REG, use_cache=False)
        if not ref:
            ref = {'cl': c.loss, 'cr': c.r2, 'jl': j.loss, 'jr': j.r2}
            continue
        assert np.isclose(c.loss, ref['cl'], rtol=1e-6), (scale, c.loss, ref['cl'])
        assert c.r2 == ref['cr'] == 0.0, (scale, c.r2)
        assert np.isclose(j.loss, ref['jl'], rtol=1e-6), (scale, j.loss, ref['jl'])
        assert np.isclose(j.r2, ref['jr'], rtol=1e-4), (scale, j.r2, ref['jr'])
        # The old absolute epsilons reported junk as perfect on tiny scales.
        assert j.r2 < 0.9, (scale, j.r2)
        assert j.loss > 0.1, (scale, j.loss)


def test_constant_on_tiny_target_scores_as_baseline():
    """std(y)=7e-10 target: a constant must get R²=0 and the median-baseline
    loss (the old guard skipped any target with std ≤ 1e-9)."""
    x = np.linspace(-3, 3, 500).reshape(-1, 1)
    y = 1e-9 * np.sin(2 * x[:, 0])
    ind = _fresh([('const', 0, 0, 0.0)])   # constant AT the target mean
    ind.calculate_fitness(x, y, REG, use_cache=False)
    baseline = evo13.regression_loss(np.full_like(y, np.median(y)), y)
    assert ind.r2 == 0.0, ind.r2
    assert abs(ind.loss - baseline) < 1e-6, (ind.loss, baseline)
    assert ind.loss > 0.1, ind.loss     # never the old spurious ≈0
    # A perfect fit still scores ≈0 at this scale.
    assert evo13.regression_loss(y.copy(), y) < 1e-6


def test_safe_r2_constant_target_and_normal_target():
    """_safe_r2: constant target → 0.0; ordinary targets match 1 − ss_res/ss_tot."""
    assert evo13._safe_r2(0.0, 0.0) == 0.0
    assert evo13._safe_r2(5.0, 0.0) == 0.0
    assert np.isclose(evo13._safe_r2(1.0, 4.0), 0.75)
    assert evo13._safe_r2(1.0, float('nan')) == 0.0


def test_fitness_cache_distinguishes_fine_constants():
    """Trees differing only in the 5th+ significant digit of a constant must
    NOT share a cache entry (x+1.0e8 vs x+1.00001e8 differ by 1000 in output)."""
    evo13._FITNESS_CACHE.clear()
    x = np.linspace(0.0, 1.0, 400).reshape(-1, 1)
    y = 1e8 + x[:, 0]
    A = _fresh([('const', 0, 0, 1.0e8), ('+', 0, 1)])
    A.calculate_fitness(x, y, REG)
    B = _fresh([('const', 0, 0, 1.00001e8), ('+', 0, 1)])
    B.calculate_fitness(x, y, REG)
    B2 = _fresh([('const', 0, 0, 1.00001e8), ('+', 0, 1)])
    B2.calculate_fitness(x, y, REG, use_cache=False)
    fpA = evo13._FITNESS_CACHE._fingerprint(A.tree)
    fpB = evo13._FITNESS_CACHE._fingerprint(B.tree)
    assert fpA != fpB, "constants beyond 4 sig figs must yield distinct keys"
    assert np.isclose(B.loss, B2.loss, rtol=1e-9), (B.loss, B2.loss)
    assert A.r2 > 0.9999 and B.r2 < 0.5, (A.r2, B.r2)
    # Identical trees still hit the cache.
    A2 = _fresh([('const', 0, 0, 1.0e8), ('+', 0, 1)])
    h0 = evo13._FITNESS_CACHE.hits
    A2.calculate_fitness(x, y, REG)
    assert evo13._FITNESS_CACHE.hits == h0 + 1


def test_simplify_preserves_tiny_multipliers():
    """x * 1e-10 must survive simplification (the old |c|<1e-9 fold collapsed
    the whole varying term to the literal constant 0)."""
    x = np.linspace(1.0, 2.0, 100).reshape(-1, 1)
    ind = _fresh([('const', 0, 0, 1e-10), ('*', 0, 1)])
    simp = evo13.simplify_cgp_tree(ind.tree)
    assert np.std(simp.evaluate(x)) > 0.0, str(simp)
    assert np.allclose(simp.evaluate(x), ind.tree.evaluate(x))
    # Exact identities still fold.
    z = evo13.simplify_cgp_tree(
        _fresh([('const', 0, 0, 0.0), ('*', 0, 1)]).tree)
    assert np.std(z.evaluate(x)) == 0.0 and float(z.evaluate(x)[0]) == 0.0
    one = evo13.simplify_cgp_tree(
        _fresh([('const', 0, 0, 1.0), ('*', 0, 1)]).tree)
    assert np.allclose(one.evaluate(x), x[:, 0])


def test_exception_path_resets_loss():
    """A tree that starts raising must not keep its previous (perfect) loss."""
    x = np.linspace(0.0, 1.0, 200).reshape(-1, 1)
    y = 2.0 * x[:, 0] + 1.0
    ind = _fresh([('+', 0, 0)])
    ind.calculate_fitness(x, y, REG, use_cache=False)
    assert ind.loss < 0.01, ind.loss
    def _boom(*a, **k):
        raise RuntimeError("boom")
    ind.tree.evaluate = _boom
    ind.calculate_fitness(x, y, REG, use_cache=False)
    assert ind.loss >= 1e9, ind.loss
    assert ind.fitness >= 1e9, ind.fitness


def test_str_keeps_small_constants_visible():
    """str(tree) must not render |c| < 5e-5 as '0.0000' — pasted equations
    were resolving into flat constants purely from display rounding."""
    ind = _fresh([('const', 0, 0, 1e-10), ('*', 0, 1)])
    s = str(ind.tree)
    assert "0.0000" not in s, s
    assert "1e-10" in s, s
    # full_expr_string embeds the affine with round-trip precision.
    ind.affine_a, ind.affine_b = 1.0, 1e8 + 0.5
    fs = ind.full_expr_string()
    assert repr(1e8 + 0.5) in fs, fs


def test_sympy_expr_divergence_detector():
    """_sympy_expr_diverges must flag a constant rewrite of a varying tree and
    accept a faithful rewrite."""
    x = np.linspace(1.0, 2.0, 128).reshape(-1, 1)
    ind = _fresh([('const', 0, 0, 1e-10), ('*', 0, 1)])
    assert evo13._sympy_expr_diverges("0.0", ind.tree, ["x0"], x) is True
    assert evo13._sympy_expr_diverges("x0 * 1e-10", ind.tree, ["x0"], x) is False
    # Unevaluable expressions are treated as diverging (don't trust them).
    assert evo13._sympy_expr_diverges("nope(", ind.tree, ["x0"], x) is True


def test_huge_offset_target_keeps_anti_collapse_pressure():
    """y = 1e10 + sin(x): the anti-collapse push must stay active (the old
    guard compared mad(y) against 1e-9·|mean| + 1e-9 and went silent)."""
    x = np.linspace(0, 6, 500)
    y_small = np.sin(x)
    y_big = 1e10 + np.sin(x)
    l_small = evo13.regression_loss(np.full_like(y_small, np.mean(y_small)), y_small)
    l_big = evo13.regression_loss(np.full_like(y_big, np.mean(y_big)), y_big)
    assert np.isclose(l_small, l_big, rtol=1e-2), (l_small, l_big)
    assert l_big > 0.5, l_big


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
