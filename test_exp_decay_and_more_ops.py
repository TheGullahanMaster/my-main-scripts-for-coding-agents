"""Tests for the exponential-decay / hyperbolic operator additions:

    * exp_decay (binary)  — exp(-x·y)        exponential decay (rate·input)
    * rbf       (binary)  — exp(-(x-y)²)     Gaussian / RBF similarity kernel
    * expm1     (unary)   — exp(x) - 1       accurate exponential near 0
    * log1p     (unary)   — log(1 + x)       accurate logarithm near 0
    * sinh      (unary)   — sinh(x)          hyperbolic sine
    * cosh      (unary)   — cosh(x)          hyperbolic cosine

Every new op is exercised end-to-end: dispatch-table evaluation, the inline
NumPy export (`tree_to_python_expr`) and the SymPy export (`tree_to_sympy`)
must all agree, in SAFE and UNSAFE mode.  SAFE forms must stay finite even at
the nasty inputs (huge magnitudes, the log1p pole at x=-1).

Mirrors the Part-1 structure of test_new_ops_and_objectives.py.
"""
import os
import math

os.environ.setdefault("PYTHONWARNINGS", "ignore")
import numpy as np
import sympy
import evo13 as e


NEW_BINARY = ['exp_decay', 'rbf']
NEW_UNARY  = ['expm1', 'log1p', 'sinh', 'cosh']
NEW_ALL    = NEW_BINARY + NEW_UNARY


# ──────────────────────────────────────────────────────────────────────────
# Helpers (mirror test_new_ops_and_objectives.py)
# ──────────────────────────────────────────────────────────────────────────
def _all_ops_live():
    """Make every op in ALL_OP_DESCRIPTIONS selectable and rebuild the
    CGPEquation arity sets so the new primitives land in the right buckets."""
    e.ALLOWED_OPS = list(e.ALL_OP_DESCRIPTIONS.keys())
    e.CGPEquation.OPS_BINARY  = [o for o in e.BINARY_OPS_EVAL if o in e.ALLOWED_OPS]
    e.CGPEquation.OPS_UNARY   = [o for o in e.UNARY_OPS_EVAL  if o in e.ALLOWED_OPS]
    e.CGPEquation.OPS_TERNARY = e._build_ternary_ops_list()
    e.CGPEquation.OPS_BINARY_SET  = set(e.CGPEquation.OPS_BINARY)
    e.CGPEquation.OPS_UNARY_SET   = set(e.CGPEquation.OPS_UNARY)
    e.CGPEquation.OPS_TERNARY_SET = set(e.CGPEquation.OPS_TERNARY)


def _build(op, arity, feat=('x0', 'x1', 'x2')):
    t = e.CGPEquation(3, 4, list(feat))
    in2 = 1 if arity >= 2 else 0
    in3 = 2 if arity == 3 else 0
    t.nodes = [e.CGPNode(op, 0, in2, 0.0, in3=in3)]
    t.out_idx = 3
    t.update_active_nodes()
    return t


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────
def test_new_ops_registered_everywhere():
    """Each new op must live in the right arity dispatch table (both safety
    modes) and carry a cost + a human-readable description."""
    for safe in (True, False):
        e.set_ops_mode(safe)
        for op in NEW_BINARY:
            assert op in e.BINARY_OPS_EVAL, (op, safe)
        for op in NEW_UNARY:
            assert op in e.UNARY_OPS_EVAL, (op, safe)
    for op in NEW_ALL:
        assert op in e.ALL_OP_DESCRIPTIONS, op
        assert op in e.OP_COSTS, op
        assert isinstance(e.op_cost(op), (int, float))
    print("PASS test_new_ops_registered_everywhere")


def test_new_ops_numeric_values():
    e.set_ops_mode(True)
    # exp_decay(x, y) = exp(-x·y)
    ed = e.BINARY_OPS_EVAL['exp_decay'](
        np.array([0.0, 1.0, 2.0]), np.array([5.0, 1.0, 3.0]))
    assert np.allclose(ed, [1.0, np.exp(-1.0), np.exp(-6.0)])
    # rbf(x, y) = exp(-(x-y)²): equal args → 1, distance 2 → exp(-4)
    rb = e.BINARY_OPS_EVAL['rbf'](
        np.array([3.0, 0.0]), np.array([3.0, 2.0]))
    assert np.allclose(rb, [1.0, np.exp(-4.0)])
    # expm1 / log1p are exact inverses on a sane domain
    assert np.allclose(e.UNARY_OPS_EVAL['expm1'](np.array([0.0, 1.0])),
                       [0.0, np.e - 1.0])
    assert np.allclose(e.UNARY_OPS_EVAL['log1p'](np.array([0.0, np.e - 1.0])),
                       [0.0, 1.0])
    # sinh is odd, cosh is even; cosh(0)=1, sinh(0)=0
    assert np.allclose(e.UNARY_OPS_EVAL['sinh'](np.array([0.0, 1.0, -1.0])),
                       [0.0, np.sinh(1.0), -np.sinh(1.0)])
    assert np.allclose(e.UNARY_OPS_EVAL['cosh'](np.array([0.0, 2.0, -2.0])),
                       [1.0, np.cosh(2.0), np.cosh(2.0)])
    print("PASS test_new_ops_numeric_values")


def test_new_ops_are_finite_in_safe_mode():
    """SAFE forms must never emit NaN/Inf — even at huge magnitudes and the
    log1p pole at x=-1."""
    e.set_ops_mode(True)
    bad = np.array([-1e9, -100.0, -1.0, 0.0, 1.0, 1e9])
    for op in NEW_UNARY:
        out = e.UNARY_OPS_EVAL[op](bad)
        assert np.all(np.isfinite(out)), (op, out)
    # exp_decay with a large negative product would overflow without the clamp
    out = e.BINARY_OPS_EVAL['exp_decay'](bad, bad[::-1])
    assert np.all(np.isfinite(out)), out
    out = e.BINARY_OPS_EVAL['rbf'](bad, bad[::-1])
    assert np.all(np.isfinite(out)), out
    print("PASS test_new_ops_are_finite_in_safe_mode")


def test_new_ops_export_roundtrip():
    """Dispatch-table eval == inline-NumPy export == SymPy export, in BOTH
    safety modes."""
    feat = ['x0', 'x1', 'x2']
    fvars = list(sympy.symbols('x0 x1 x2'))
    # Moderate, mixed-sign inputs (in-clamp so SAFE clamps don't diverge);
    # log1p gets strictly-positive args (well clear of its x=-1 pole).
    Xmix = np.array([[0.7, -1.3, 2.0], [-2.5, 1.1, -0.4],
                     [3.2, 0.6, -2.7], [-0.9, -2.2, 1.8]])
    Xpos = np.array([[0.7, 1.3, 2.0], [2.5, 1.1, 0.4],
                     [3.2, 0.6, 2.7], [0.2, 2.2, 1.8]])
    cases = [('exp_decay', 2, Xmix), ('rbf', 2, Xmix),
             ('sinh', 1, Xmix), ('cosh', 1, Xmix),
             ('expm1', 1, Xmix), ('log1p', 1, Xpos)]
    for safe in (True, False):
        e.set_ops_mode(safe)
        _all_ops_live()
        for op, ar, X in cases:
            t = _build(op, ar, feat)
            direct = np.nan_to_num(t.evaluate(X))

            expr = e.tree_to_python_expr(t, feat, safe=safe)
            ns = {'np': np, 'math': math}
            for i, fn in enumerate(feat):
                ns[fn] = X[:, i]
            pye = np.nan_to_num(np.broadcast_to(
                np.asarray(eval(expr, ns), dtype=float), direct.shape))
            assert np.allclose(direct, pye, atol=1e-5, rtol=1e-4), \
                (op, safe, 'pyexpr', expr, direct, pye)

            s = e.tree_to_sympy(t, fvars, safe=safe)
            fn = sympy.lambdify(fvars, s, 'numpy')
            sye = np.nan_to_num(np.broadcast_to(
                np.asarray(fn(*[X[:, i] for i in range(3)]), dtype=float),
                direct.shape))
            assert np.allclose(direct, sye, atol=1e-5, rtol=1e-4), \
                (op, safe, 'sympy', s, direct, sye)
    print("PASS test_new_ops_export_roundtrip")


def test_new_ops_string_repr():
    assert e._binary_str('exp_decay', 'a', 'b') == "exp(-(a)·(b))"
    assert e._binary_str('rbf', 'a', 'b') == "exp(-((a) - (b))²)"
    # unary ops fall through to the generic name(arg) form
    assert e._unary_str('expm1', 'a') == "expm1(a)"
    assert e._unary_str('log1p', 'a') == "log1p(a)"
    assert e._unary_str('sinh', 'a') == "sinh(a)"
    assert e._unary_str('cosh', 'a') == "cosh(a)"
    print("PASS test_new_ops_string_repr")


if __name__ == "__main__":
    test_new_ops_registered_everywhere()
    test_new_ops_numeric_values()
    test_new_ops_are_finite_in_safe_mode()
    test_new_ops_export_roundtrip()
    test_new_ops_string_repr()
    print("\nALL EXP_DECAY / EXTRA-OPS TESTS PASSED")
