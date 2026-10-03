"""
Regression tests for the domain-guard-exploit fixes in evo13.py.

Covers the failure reported on a "multiplications until underflow" dataset,
where the winning equation was ``1075·log_number((number < number)) − 0.49``:
``number < number`` is identically zero, so the protected log's epsilon guard
supplied a hidden constant ``ln(1e-30) ≈ −69`` — the model fit (R²=1) but the
equation is degenerate ("log of zero"), SymPy renders it as ``zoo``, and a
faithful re-implementation NaNs.  Guards added:

  • a default-on DOMAIN-GUARD PENALTY folds the (already computed)
    ``last_eval_domain_margin`` into ``.loss``, so guard-epsilon exploits lose
    to honest reformulations of the same fit;
  • simplify_cgp_tree folds SELF-COMPARISONS (x<x → 0, x≥x → 1, x−x → 0, …)
    so the degenerate construct is at least exposed as a literal;
  • generate_script mirrors training's pre-affine output sanitisation
    (clip ±1e9, NaN→0) and falls back to a HARDENED per-node function
    (mirroring the evaluator's per-node sanitisation) whenever the compact
    inline expression diverges numerically from the evaluator;
  • the final report flags SymPy simplifications containing zoo/oo/nan.

Runnable two ways:
    python test_domain_guard_and_export.py   # short report, non-zero on fail
    pytest test_domain_guard_and_export.py
"""
import os
import re
import tempfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import evo13

evo13.set_ops_mode(True)
for _op in ('log_base', 'lt', 'gt', 'lte', 'gte', 'eq', '-', 'delta',
            'min', 'max', '/'):
    if _op not in evo13.CGPEquation.OPS_BINARY:
        evo13.CGPEquation.OPS_BINARY = evo13.CGPEquation.OPS_BINARY + [_op]
evo13.CGPEquation.OPS_BINARY_SET = set(evo13.CGPEquation.OPS_BINARY)

REG = 5
rng = np.random.default_rng(20260802)


def _underflow_data():
    """number ∈ (0,1) → y = ln(5e-324)/ln(number), the user's failing dataset."""
    number = rng.uniform(0.05, 0.95, 600)
    y = np.log(5e-324) / np.log(number)
    df = pd.DataFrame({"number": number, "y": y})
    dp = evo13.DataProcessor(scale=False, normalize="no")
    dp.set_configs([("number", 1), ("y", 5)])
    X, Y = dp.fit_transform(df)
    return dp, X, Y


def _fit(dp, X, Y, specs):
    ind = evo13._build_seed(1, dp.input_map, specs)
    ind.affine_fitted = False
    ind.calculate_fitness(X, Y[:, 0], REG, use_cache=False)
    return ind


def test_domain_guard_penalty_prefers_honest_form():
    """log_base((x<x), x) exploits the log guard (margin=1) and must carry the
    penalty; the honest log_base(C, x) fit of the SAME target must not."""
    dp, X, Y = _underflow_data()
    exploit = _fit(dp, X, Y, [('lt', 0, 0), ('log_base', 1, 0)])
    honest = _fit(dp, X, Y, [('const', 0, 0, 361.0), ('log_base', 1, 0)])
    assert exploit.r2 > 0.999 and honest.r2 > 0.999
    assert exploit.tree.last_eval_domain_margin > 0.99
    assert honest.tree.last_eval_domain_margin < 0.05
    assert exploit.loss > honest.loss + 0.5, (exploit.loss, honest.loss)


def test_domain_guard_penalty_can_be_disabled():
    """DOMAIN_GUARD_PENALTY_WEIGHT = 0 restores the legacy scoring."""
    dp, X, Y = _underflow_data()
    saved = evo13.DOMAIN_GUARD_PENALTY_WEIGHT
    try:
        evo13.DOMAIN_GUARD_PENALTY_WEIGHT = 0.0
        exploit = _fit(dp, X, Y, [('lt', 0, 0), ('log_base', 1, 0)])
        assert exploit.loss < 0.01, exploit.loss
    finally:
        evo13.DOMAIN_GUARD_PENALTY_WEIGHT = saved


def test_self_comparison_folds():
    """x<x→0, x≥x→1, x−x→0, min(x,x)→x — semantics-preserving, and
    comparisons of DIFFERENT inputs are left alone."""
    x = np.linspace(-2.0, 2.0, 100).reshape(-1, 1)
    cases = [
        (('lt', 0, 0), 0.0), (('gt', 0, 0), 0.0),
        (('lte', 0, 0), 1.0), (('gte', 0, 0), 1.0), (('eq', 0, 0), 1.0),
        (('-', 0, 0), 0.0), (('delta', 0, 0), 0.0),
    ]
    for spec, want in cases:
        tree = evo13._build_seed(1, ["x0"], [spec]).tree
        simp = evo13.simplify_cgp_tree(tree)
        got = simp.evaluate(x)
        assert np.allclose(got, want), (spec, got[:3])
        assert np.allclose(tree.evaluate(x), got), spec
    # min(x, x) → x (redirect)
    mm = evo13.simplify_cgp_tree(
        evo13._build_seed(1, ["x0"], [('min', 0, 0)]).tree)
    assert np.allclose(mm.evaluate(x), x[:, 0])
    # lt over two DIFFERENT features must not fold
    two = evo13._build_seed(2, ["x0", "x1"], [('lt', 0, 1)]).tree
    x2 = np.column_stack([x[:, 0], -x[:, 0]])
    keep = evo13.simplify_cgp_tree(two)
    assert np.std(keep.evaluate(x2)) > 0.1


def test_hardened_fn_matches_evaluator():
    """The per-node hardened emitter reproduces evaluate() + output clip."""
    dp, X, Y = _underflow_data()
    ind = _fit(dp, X, Y, [('lt', 0, 0), ('log_base', 1, 0)])
    src = evo13._tree_to_hardened_fn_src(ind.tree, dp.input_map,
                                         "_hard_fn", safe=True)
    env = {'np': np}
    exec(src, env)
    got = env['_hard_fn'](X[:, 0])
    want = np.clip(np.nan_to_num(ind.tree.evaluate(X, force_hard=True),
                                 nan=0.0, posinf=1e9, neginf=-1e9), -1e9, 1e9)
    assert got.shape == want.shape
    assert np.allclose(got, want, rtol=1e-12), float(np.max(np.abs(got - want)))


def test_export_never_emits_nan_predictions():
    """Unsafe-mode intermediate overflow: training sanitises per node
    (Inf → 1e9) but the inline composition doesn't — generate_script must
    detect the divergence and emit a faithful hardened function."""
    xv = np.linspace(600.0, 800.0, 400)          # exp overflows above ~709
    yv = 5.0 / np.minimum(np.exp(xv), 1e9)
    df = pd.DataFrame({"x0": xv, "y": yv})
    dp = evo13.DataProcessor(scale=False, normalize="no")
    dp.set_configs([("x0", 1), ("y", 5)])
    X, Y = dp.fit_transform(df)
    evo13.set_ops_mode(False)                    # UNSAFE ops
    try:
        # tree: 5.0 / exp(x0) — training: exp→Inf→1e9 per node → 5/1e9;
        # inline: 5/np.exp(x) → 0.0 on overflow rows (diverges).
        ind = evo13._build_seed(1, dp.input_map,
                                [('const', 0, 0, 5.0), ('exp', 0, 0),
                                 ('/', 1, 2)])
        ind.affine_fitted = False
        ind.calculate_fitness(X, Y[:, 0], REG, use_cache=False)
        out = os.path.join(tempfile.gettempdir(), "evo13_test_export.py")
        evo13.generate_script([ind], dp, filename=out, X_data=X)
        src = open(out).read()
        assert "_model_output_0" in src, "hardened function expected"
        m = re.search(r"y_pred\[:, 0\] = (.+)", src)
        env = {'np': np, 'x0': X[:, 0]}
        hm = re.search(r"(def _model_output_0.*?\n)(?=\n)", src, re.S)
        exec(hm.group(1), env)
        pred = np.broadcast_to(np.asarray(eval(m.group(1), env), float),
                               (X.shape[0],))
        assert np.all(np.isfinite(pred)), "exported predictions must be finite"
        train_pred = ind.affine_a * np.clip(np.nan_to_num(
            ind.tree.evaluate(X), nan=0.0, posinf=1e9, neginf=-1e9),
            -1e9, 1e9) + ind.affine_b
        assert np.allclose(pred, train_pred, rtol=1e-9,
                           atol=1e-12 + 1e-9 * float(np.max(np.abs(train_pred))))
    finally:
        evo13.set_ops_mode(True)
        if os.path.exists(os.path.join(tempfile.gettempdir(),
                                       "evo13_test_export.py")):
            os.remove(os.path.join(tempfile.gettempdir(),
                                   "evo13_test_export.py"))


def test_degenerate_sympy_atoms_are_detectable():
    """The zoo/oo/nan atom check used by the final report flags undefined
    forms and passes ordinary expressions."""
    import sympy
    n = sympy.Symbol("number")
    bad = sympy.Float(1075) * sympy.zoo / sympy.log(n)
    good = sympy.Float(-69.08) / sympy.log(sympy.Abs(n) + sympy.Float(1e-30))
    def flagged(e):
        return (e.has(sympy.zoo) or e.has(sympy.oo)
                or e.has(-sympy.oo) or e.has(sympy.nan))
    assert flagged(bad) is True
    assert flagged(good) is False


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
