"""
Regression tests for the residual-composite seeding in evo13.py — the feature
that lets the search assemble a COMPLEX MULTIVARIABLE correction term onto the
current best when the one-shot startup solvers (OLS / log-OLS / rational) miss
it.

The targeted failure mode: an additive mix of heterogeneous terms, e.g.

    y = x0·x1/(x2·x3) + 0.5·x4² − 2·x5

The polynomial OLS captures the additive part, the log-OLS bails on the mixed
signs, and the multiplicative interaction term is never seeded — so plain CGP
almost never assembles it.  These tests guard that:

  • the tree-composition helper splices  scale·A + B  exactly;
  • the interaction-basis ranker surfaces the true missing term at the top;
  • the correction transcriber renders every basis kind faithfully;
  • a composite that recovers the full heterogeneous structure is produced and
    scores far better than the best analytical seed alone;
  • ``generate_importance_biased_seeds`` emits such a composite at startup; and
  • the machinery is inert (no crash, empty result) when seeds are off or the
    ``+`` operator is unavailable.

Runnable two ways:
    python test_residual_composites.py     # prints a short report, non-zero on fail
    pytest test_residual_composites.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import evo13
from evo13 import (
    CGPEquation, CGPNode, Individual, set_ops_mode,
    BINARY_OPS_EVAL, UNARY_OPS_EVAL,
    _compose_additive_tree, _residual_basis_candidates,
    _build_residual_correction_tree, _make_residual_composites,
    _fit_affine, generate_importance_biased_seeds,
    generate_ols_basis_seeds, generate_log_ols_seeds,
)


def _set_ops(ops_list):
    """Install an op whitelist the same way the benches do."""
    evo13.ALLOWED_OPS = list(ops_list)
    evo13.IF_ELSE_ENABLED = False
    CGPEquation.OPS_BINARY = [op for op in BINARY_OPS_EVAL if op in evo13.ALLOWED_OPS]
    CGPEquation.OPS_UNARY = [op for op in UNARY_OPS_EVAL if op in evo13.ALLOWED_OPS]
    CGPEquation.OPS_TERNARY = []
    CGPEquation.OPS_ALL = (
        CGPEquation.OPS_BINARY + CGPEquation.OPS_UNARY
        + (['const'] if 'const' in evo13.ALLOWED_OPS else []))
    CGPEquation.OPS_BINARY_SET = set(CGPEquation.OPS_BINARY)
    CGPEquation.OPS_UNARY_SET = set(CGPEquation.OPS_UNARY)
    CGPEquation.OPS_TERNARY_SET = set(CGPEquation.OPS_TERNARY)


def _configure(ops=('+', '-', '*', '/', 'square', 'sqrt', 'pow', 'log', 'exp',
                    'sin', 'cos', 'abs', 'const')):
    set_ops_mode(True)
    evo13.USE_SEEDS = True
    evo13.INTERVAL_MODE = False
    evo13.BITWISE_MODE = 'off'
    evo13.CGP_NODES = 48
    evo13.AFFINE_SCALING_ENABLED = True
    evo13.SOBOLEV_ENABLED = False
    evo13.FEATURE_PRIORS = None
    evo13._INIT_PHASE = False
    _set_ops(ops)


def _r2(ind, X, y):
    ind.affine_fitted = False
    ind.calculate_fitness(X, y, 5)
    return ind.r2


def _mix_data(seed=0, n=600):
    """y = x0*x1/(x2*x3) + 0.5*x4^2 - 2*x5  (additive mix, positive features)."""
    X = np.random.RandomState(seed).uniform(0.5, 3.0, size=(n, 6))
    y = (X[:, 0] * X[:, 1] / (X[:, 2] * X[:, 3])
         + 0.5 * X[:, 4] ** 2 - 2.0 * X[:, 5])
    return X, y, [f"x{i}" for i in range(6)]


# ───────────────────────────────────────────────────────────────────────────
def test_compose_additive_tree_is_exact():
    _configure()
    nf = 4
    feat = [f"x{i}" for i in range(nf)]
    X = np.random.RandomState(1).uniform(0.5, 3.0, size=(200, nf))

    # A = 0.5*x0^2 - 2*x1 ;  B = x2*x3
    A = CGPEquation(nf, 6, feat)
    A.nodes = [CGPNode('square', 0, 0), CGPNode('const', 0, 0, 0.5),
               CGPNode('*', nf + 0, nf + 1), CGPNode('const', 0, 0, -2.0),
               CGPNode('*', nf + 3, 1), CGPNode('+', nf + 2, nf + 4)]
    A.out_idx = nf + 5
    A.update_active_nodes()
    B = CGPEquation(nf, 1, feat)
    B.nodes = [CGPNode('*', 2, 3)]
    B.out_idx = nf
    B.update_active_nodes()

    a_val = 0.5 * X[:, 0] ** 2 - 2.0 * X[:, 1]
    b_val = X[:, 2] * X[:, 3]

    c1 = _compose_additive_tree(A, B, scale_a=1.0)
    assert np.allclose(c1.evaluate(X), a_val + b_val), "A + B mismatch"

    c3 = _compose_additive_tree(A, B, scale_a=3.0)
    assert np.allclose(c3.evaluate(X), 3.0 * a_val + b_val), "3A + B mismatch"

    # Compaction: composite copies only ACTIVE nodes, so it stays small.
    assert len(c1.nodes) <= len(A.nodes) + len(B.nodes) + 1
    print("  PASS  test_compose_additive_tree_is_exact")


def test_basis_ranker_surfaces_true_term():
    _configure()
    X, y, feat = _mix_data(0)
    # Residual = the exact monomial term -> its ratprod must rank first.
    resid = X[:, 0] * X[:, 1] / (X[:, 2] * X[:, 3])
    cands = _residual_basis_candidates(6, X, resid)
    assert cands, "no candidates produced"
    corr, kind, params, _ = cands[0]
    assert kind == 'ratprod' and set(params) == {0, 1, 2, 3}, \
        f"top candidate was {kind}{params}, expected ratprod(0,1,2,3)"
    assert corr > 0.99, f"top corr {corr:.3f} should be ~1 for an exact term"
    print("  PASS  test_basis_ranker_surfaces_true_term")


def test_correction_tree_renders_every_kind():
    _configure()
    nf = 4
    feat = [f"x{i}" for i in range(nf)]
    X = np.random.RandomState(2).uniform(0.6, 2.5, size=(150, nf))
    checks = [
        (('feat', (1,)), 2.0, 2.0 * X[:, 1]),
        (('square', (0,)), 1.5, 1.5 * X[:, 0] ** 2),
        (('product', (0, 1)), -0.7, -0.7 * X[:, 0] * X[:, 1]),
        (('ratio', (0, 1)), 1.0, X[:, 0] / X[:, 1]),
        (('ratprod', (0, 1, 2, 3)), 2.0, 2.0 * X[:, 0] * X[:, 1] / (X[:, 2] * X[:, 3])),
        (('unary', ('sin', 2)), 1.0, np.sin(X[:, 2])),
    ]
    for (kind, params), w, expected in checks:
        tree = _build_residual_correction_tree(nf, feat, [(kind, params)], [w], 0.0)
        assert tree is not None, f"{kind} tree was None"
        got = tree.evaluate(X)
        assert np.allclose(got, expected, atol=1e-6), f"{kind} render mismatch"
    # multi-term + bias chain
    tree = _build_residual_correction_tree(
        nf, feat, [('product', (0, 1)), ('feat', (2,))], [2.0, -1.0], 3.0)
    exp = 2.0 * X[:, 0] * X[:, 1] - 1.0 * X[:, 2] + 3.0
    assert np.allclose(tree.evaluate(X), exp, atol=1e-6), "multi-term chain mismatch"
    print("  PASS  test_correction_tree_renders_every_kind")


def test_composite_recovers_additive_mix():
    _configure()
    X, y, feat = _mix_data(0)
    y_arr = y.astype(float)
    y_var = float(np.var(y_arr))

    # Best closed-form analytical seed (what the search would lock onto).
    seeds = (generate_ols_basis_seeds(6, feat, X, y)
             + generate_log_ols_seeds(6, feat, X, y))
    assert seeds, "no analytical seeds"
    best_t = best_a = best_b = best_r = None
    best_sc = np.inf
    for s in seeds:
        bp = np.clip(np.nan_to_num(s.tree.evaluate(X), nan=0.0,
                                   posinf=1e9, neginf=-1e9), -1e9, 1e9)
        a, b, _ = _fit_affine(bp, y_arr)
        r = y_arr - (a * bp + b)
        sc = float(np.mean(r ** 2)) / (y_var + 1e-12)
        if sc < best_sc:
            best_sc, best_t, best_a, best_b, best_r = sc, s.tree, a, b, r
    base_r2 = 1.0 - best_sc

    base = Individual(best_t.clone())
    base.affine_a, base.affine_b = float(best_a), float(best_b)
    base.boost_stages = []
    comps = _make_residual_composites(6, feat, X, best_r, base, max_composites=4)
    assert comps, "no composites produced for an additive-mix target"
    best_comp = max(_r2(c, X, y) for c in comps)

    assert best_comp > base_r2 + 0.05, \
        f"composite r2 {best_comp:.3f} did not beat base {base_r2:.3f}"
    assert best_comp > 0.9, f"composite r2 {best_comp:.3f} should clear 0.9"
    print(f"  PASS  test_composite_recovers_additive_mix "
          f"(base r2={base_r2:.3f} -> composite r2={best_comp:.3f})")


def test_importance_seeds_include_strong_composite():
    _configure()
    X, y, feat = _mix_data(3)
    seeds, n_priority = generate_importance_biased_seeds(
        6, feat, X, y, return_priority_count=True)
    assert seeds, "no importance-biased seeds"
    # The composite must live in the pinned priority block, and lift the best
    # priority seed comfortably above the ~0.83 the lone OLS analytical seed
    # reaches on this additive-mix target.
    assert n_priority >= 1
    best_r2 = max(_r2(s, X, y) for s in seeds[:n_priority])
    assert best_r2 > 0.87, \
        f"best priority seed r2 {best_r2:.3f} should clear 0.87 with composites on"
    print(f"  PASS  test_importance_seeds_include_strong_composite "
          f"(best priority r2={best_r2:.3f})")


def test_inert_when_disabled_or_no_plus():
    _configure()
    X, y, feat = _mix_data(0)
    resid = y - float(np.mean(y))
    base = Individual(generate_ols_basis_seeds(6, feat, X, y)[0].tree.clone())
    base.affine_a, base.affine_b, base.boost_stages = 1.0, 0.0, []

    # Seeds off → empty.
    evo13.USE_SEEDS = False
    assert _make_residual_composites(6, feat, X, resid, base) == []
    evo13.USE_SEEDS = True

    # No '+' operator → cannot compose → empty, no crash.
    _set_ops(['-', '*', '/', 'square', 'const'])
    assert _make_residual_composites(6, feat, X, resid, base) == []
    assert _compose_additive_tree(base.tree, base.tree) is None
    _configure()  # restore

    # Constant residual → nothing to correct.
    assert _make_residual_composites(6, feat, X, np.zeros(len(y)), base) == []
    print("  PASS  test_inert_when_disabled_or_no_plus")


def main():
    tests = [
        test_compose_additive_tree_is_exact,
        test_basis_ranker_surfaces_true_term,
        test_correction_tree_renders_every_kind,
        test_composite_recovers_additive_mix,
        test_importance_seeds_include_strong_composite,
        test_inert_when_disabled_or_no_plus,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
