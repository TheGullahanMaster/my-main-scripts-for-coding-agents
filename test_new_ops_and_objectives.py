"""Tests for the operator and NSGA-III-objective additions:

  Part 1 — new operators
    * distance_2  (binary)  — sqrt(x²+y²)        2-variable Euclidean distance
    * distance_3  (ternary) — sqrt(x²+y²+z²)     3-variable Euclidean distance
    * log_base    (binary)  — log_y(x)           logarithm to an arbitrary base
    * root3..root10 (unary) — x**(1/k)           native fractional-power roots
    * oom         (unary)   — floor(log10|x|)    integer order of magnitude

  Part 2 — optional NSGA-III extra objectives (instability / shape / diversity)
    Widen the AFPO survival trim past the classic 3 objectives, gated on BOTH
    NSGA3_ENABLED and NSGA3_EXTRA_OBJ_ENABLED.

  Part 3 — A-NSGA-III is the interactive default for the reference-vector module.

Every new op is exercised end-to-end: dispatch-table evaluation, the inline
NumPy export (`tree_to_python_expr`) and the SymPy export (`tree_to_sympy`)
must all agree, in SAFE and UNSAFE mode, including the tricky negative-base
odd/even root sign rule.
"""
import os
import math
import random
import builtins
import contextlib

os.environ.setdefault("PYTHONWARNINGS", "ignore")
import numpy as np
import sympy
import evo13 as e


NEW_BINARY  = ['distance_2', 'log_base']
NEW_TERNARY = ['distance_3']
NEW_UNARY   = ['root3', 'root4', 'root5', 'root6', 'root7', 'root8',
               'root9', 'root10', 'oom']
NEW_ALL     = NEW_BINARY + NEW_TERNARY + NEW_UNARY


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def feed_input(answers):
    """Temporarily replace builtins.input with a scripted iterator."""
    it = iter(answers)
    orig = builtins.input
    builtins.input = lambda *a, **k: next(it)
    try:
        yield
    finally:
        builtins.input = orig


@contextlib.contextmanager
def _saved_nsga3_globals():
    keys = ('NSGA3_ENABLED', 'NSGA3_DIVISIONS', 'NSGA3_VARIANT',
            'NSGA3_EXTRA_OBJ_ENABLED', 'NSGA3_EXTRA_OBJECTIVES')
    saved = {k: getattr(e, k) for k in keys}
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(e, k, v)
        e._NSGA3_REF_CACHE.clear()


def _all_ops_live():
    """Make every op in ALL_OP_DESCRIPTIONS selectable and rebuild the
    CGPEquation arity sets (so distance_3 lands in the ternary set)."""
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


def _mk(age, fit, comp, *, shape=1.0, instab=1.0, sig=None, loss=None):
    """Trivial individual with an explicit objective profile (mirrors
    test_nsga3._mk) plus the optional extra-objective metrics."""
    t = e.CGPEquation(2, 4, ['x0', 'x1'])
    t.nodes = [e.CGPNode('+', 0, 1, 0.0)]
    t.out_idx = 2
    t.update_active_nodes()
    ind = e.Individual(t)
    ind.age = float(age)
    ind.fitness = float(fit)
    ind.loss = float(fit if loss is None else loss)
    ind.complexity = float(comp)
    ind.shape_obj = float(shape)
    ind.instability_obj = float(instab)
    ind.behavior_sig = sig
    return ind


# ──────────────────────────────────────────────────────────────────────────
# Part 1 — new operators
# ──────────────────────────────────────────────────────────────────────────
def test_new_ops_registered_everywhere():
    """Each new op must live in the right arity dispatch table (both safety
    modes) and carry a cost + a human-readable description."""
    for safe in (True, False):
        e.set_ops_mode(safe)
        for op in NEW_BINARY:
            assert op in e.BINARY_OPS_EVAL, (op, safe)
        for op in NEW_TERNARY:
            assert op in e.TERNARY_OPS_EVAL, (op, safe)
        for op in NEW_UNARY:
            assert op in e.UNARY_OPS_EVAL, (op, safe)
    for op in NEW_ALL:
        assert op in e.ALL_OP_DESCRIPTIONS, op
        assert op in e.OP_COSTS, op
        assert isinstance(e.op_cost(op), (int, float))
    print("PASS test_new_ops_registered_everywhere")


def test_new_ops_numeric_values():
    e.set_ops_mode(True)
    d2 = e.BINARY_OPS_EVAL['distance_2'](np.array([3.0, 5.0]), np.array([4.0, 12.0]))
    assert np.allclose(d2, [5.0, 13.0])
    d3 = e.TERNARY_OPS_EVAL['distance_3'](
        np.array([3.0]), np.array([4.0]), np.array([12.0]))
    assert np.allclose(d3, [13.0])
    # log_base: log_2(8)=3, log_10(1000)=3, log_3(81)=4
    lb = e.BINARY_OPS_EVAL['log_base'](
        np.array([8.0, 1000.0, 81.0]), np.array([2.0, 10.0, 3.0]))
    assert np.allclose(lb, [3.0, 3.0, 4.0], atol=1e-9)
    # odd roots keep sign; even roots use |x|
    assert np.allclose(e.UNARY_OPS_EVAL['root3'](np.array([-8.0, 27.0])), [-2.0, 3.0])
    assert np.allclose(e.UNARY_OPS_EVAL['root5'](np.array([-32.0])), [-2.0])
    assert np.allclose(e.UNARY_OPS_EVAL['root4'](np.array([16.0, 81.0])), [2.0, 3.0])
    assert np.allclose(e.UNARY_OPS_EVAL['root6'](np.array([-64.0])), [2.0])  # |x|
    # order of magnitude
    assert np.allclose(e.UNARY_OPS_EVAL['oom'](np.array([1234.0, 0.05, 7.0])),
                       [3.0, -2.0, 0.0])
    print("PASS test_new_ops_numeric_values")


def test_new_ops_are_finite_in_safe_mode():
    """SAFE forms must never emit NaN/Inf — even at the nasty inputs (negative
    bases, zero, base-1 logarithm)."""
    e.set_ops_mode(True)
    bad = np.array([-100.0, -1.0, 0.0, 1.0, 1e9])
    for op in NEW_UNARY:
        out = e.UNARY_OPS_EVAL[op](bad)
        assert np.all(np.isfinite(out)), (op, out)
    # log_base with a base ≈ 1 (log(base)→0) must stay finite via the guard
    out = e.BINARY_OPS_EVAL['log_base'](np.array([10.0, -3.0, 0.0]),
                                        np.array([1.0, 1.0, 1.0]))
    assert np.all(np.isfinite(out)), out
    out = e.BINARY_OPS_EVAL['distance_2'](bad, bad[::-1])
    assert np.all(np.isfinite(out))
    print("PASS test_new_ops_are_finite_in_safe_mode")


def test_new_ops_export_roundtrip():
    """Dispatch-table eval == inline-NumPy export == SymPy export, in BOTH
    safety modes, including negative inputs (root sign rule) and the
    base-y logarithm."""
    feat = ['x0', 'x1', 'x2']
    fvars = list(sympy.symbols('x0 x1 x2'))
    # Mixed-sign, in-domain-ish inputs (positive for log_base's args).
    Xpos = np.array([[8.0, 2.0, 27.0], [1000.0, 10.0, 16.0], [5.0, 3.0, 81.0]])
    Xneg = np.array([[-8.0, 2.0, -32.0], [27.0, 3.0, -64.0], [-1.0, 5.0, 16.0]])
    cases = [('distance_2', 2, Xneg), ('log_base', 2, Xpos),
             ('distance_3', 3, Xneg), ('root3', 1, Xneg), ('root4', 1, Xneg),
             ('root7', 1, Xneg), ('root8', 1, Xneg), ('oom', 1, Xneg)]
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
    assert e._binary_str('distance_2', 'a', 'b') == "sqrt((a)² + (b)²)"
    assert e._binary_str('log_base', 'a', 'b') == "log_b(a)"
    assert e._ternary_str('distance_3', 'a', 'b', 'c') == "sqrt((a)² + (b)² + (c)²)"
    assert e._unary_str('root3', 'a') == "(a)^(1/3)"
    assert e._unary_str('root10', 'a') == "(a)^(1/10)"
    assert e._unary_str('oom', 'a') == "oom(a)"
    print("PASS test_new_ops_string_repr")


# ──────────────────────────────────────────────────────────────────────────
# Part 2 — optional NSGA-III extra objectives
# ──────────────────────────────────────────────────────────────────────────
def test_behavior_signature_is_zscored_and_degenerate_safe():
    sig = e._behavior_signature(np.linspace(0.0, 100.0, 200), k=16)
    assert sig.shape == (16,)
    assert abs(float(sig.mean())) < 1e-9 and abs(float(sig.std()) - 1.0) < 1e-6
    # Constant predictions → zero vector (all such models are behaviourally one).
    z = e._behavior_signature(np.full(50, 7.0), k=16)
    assert np.allclose(z, 0.0)
    assert e._behavior_signature(np.array([]), k=16) is None
    print("PASS test_behavior_signature_is_zscored_and_degenerate_safe")


def test_extra_objective_columns_gated_and_shaped():
    with _saved_nsga3_globals():
        pop = [_mk(1, 2, 3, shape=0.4, instab=0.1, sig=np.random.randn(16))
               for _ in range(6)]
        # OFF by default.
        e.NSGA3_ENABLED = True
        e.NSGA3_EXTRA_OBJ_ENABLED = False
        assert e._afpo_extra_objective_columns(pop) == []
        # Enabled but NSGA-III off → still gated out (extras ride on NSGA-III).
        e.NSGA3_ENABLED = False
        e.NSGA3_EXTRA_OBJ_ENABLED = True
        e.NSGA3_EXTRA_OBJECTIVES = ('instability', 'shape', 'diversity')
        assert e._afpo_extra_objective_columns(pop) == []
        # Both on → three columns, each length n.
        e.NSGA3_ENABLED = True
        cols = e._afpo_extra_objective_columns(pop)
        assert [c[0] for c in cols] == ['instability', 'shape', 'diversity']
        assert all(c[1].shape == (6,) for c in cols)
        # A subset selects only those columns, in canonical order.
        e.NSGA3_EXTRA_OBJECTIVES = ('shape',)
        cols = e._afpo_extra_objective_columns(pop)
        assert [c[0] for c in cols] == ['shape']
    print("PASS test_extra_objective_columns_gated_and_shaped")


def test_diversity_objective_penalises_clones():
    """Behavioural clones (identical signatures) must score HIGHER (worse, since
    minimised) on the diversity objective than behaviourally unique models."""
    np.random.seed(0)
    clone = np.ones(16)
    clones = [_mk(5, 5, 5, sig=clone.copy()) for _ in range(5)]
    uniques = [_mk(5, 5, 5, sig=np.random.randn(16) * 5.0) for _ in range(5)]
    div = e._afpo_diversity_column(clones + uniques)
    assert float(np.mean(div[:5])) > float(np.mean(div[5:])), div
    # Too few signatures → inert (all-zero) column, never a crash.
    assert np.allclose(e._afpo_diversity_column([_mk(1, 1, 1)]), 0.0)
    print("PASS test_diversity_objective_penalises_clones")


def test_trim_with_extras_holds_target_and_keeps_champion():
    e.set_ops_mode(True); e._INIT_PHASE = False
    with _saved_nsga3_globals():
        e.NSGA3_ENABLED = True
        e.NSGA3_VARIANT = 'a'
        e.NSGA3_EXTRA_OBJ_ENABLED = True
        e.NSGA3_EXTRA_OBJECTIVES = ('instability', 'shape', 'diversity')
        random.seed(0); np.random.seed(0)
        pop = [_mk(random.randint(0, 30), random.random() * 100,
                   random.randint(1, 60),
                   shape=random.random(), instab=random.random(),
                   sig=np.random.randn(16)) for _ in range(160)]
        champ = min(pop, key=lambda x: x.loss)
        surv = e._trim_to_pareto_front_3obj(pop, 80)
        assert len(surv) == 80
        assert champ in surv
        assert len({id(s) for s in surv}) == 80
        # Deterministic for a fixed pool.
        a = sorted((s.age, s.fitness, s.complexity)
                   for s in e._trim_to_pareto_front_3obj(list(pop), 60))
        b = sorted((s.age, s.fitness, s.complexity)
                   for s in e._trim_to_pareto_front_3obj(list(pop), 60))
        assert a == b
    print("PASS test_trim_with_extras_holds_target_and_keeps_champion")


def test_extras_reshape_selection_vs_3obj():
    """Turning the extra objectives on must be able to change which boundary
    members survive — proof the columns actually enter the trim."""
    e.set_ops_mode(True); e._INIT_PHASE = False
    with _saved_nsga3_globals():
        e.NSGA3_ENABLED = True
        e.NSGA3_VARIANT = 'a'
        random.seed(3); np.random.seed(3)

        def fresh_pool():
            random.seed(3); np.random.seed(3)
            return [_mk(random.randint(0, 10), random.random() * 20,
                        random.randint(1, 15),
                        shape=random.random(), instab=random.random(),
                        sig=np.random.randn(16)) for _ in range(40)]

        e.NSGA3_EXTRA_OBJ_ENABLED = False
        base = sorted((s.age, s.fitness, s.complexity)
                      for s in e._trim_to_pareto_front_3obj(fresh_pool(), 12))
        e.NSGA3_EXTRA_OBJ_ENABLED = True
        e.NSGA3_EXTRA_OBJECTIVES = ('instability', 'shape', 'diversity')
        extra = sorted((s.age, s.fitness, s.complexity)
                       for s in e._trim_to_pareto_front_3obj(fresh_pool(), 12))
        assert base != extra, "extra objectives did not change the survivor set"
    print("PASS test_extras_reshape_selection_vs_3obj")


def test_calculate_fitness_caches_extra_objectives():
    """A full-data regression evaluation must populate shape_obj, instability_obj
    and behavior_sig when the feature is armed."""
    e.set_ops_mode(True); e._INIT_PHASE = False
    with _saved_nsga3_globals():
        e.NSGA3_ENABLED = True
        e.NSGA3_EXTRA_OBJ_ENABLED = True
        e.NSGA3_EXTRA_OBJECTIVES = ('instability', 'shape', 'diversity')
        np.random.seed(0)
        X = np.random.uniform(0.5, 4.0, size=(120, 1))
        y = (2.0 * X[:, 0] + 1.0)            # perfectly ordered → shape_obj ≈ 0
        feat = ['x0']
        t = e.CGPEquation(1, 4, feat)
        t.nodes = [e.CGPNode('+', 0, 0, 0.0)]     # f = x0 + x0 (monotone in x0)
        t.out_idx = 1
        t.update_active_nodes()
        ind = e.Individual(t)
        ind.calculate_fitness(X, y, 5)
        assert isinstance(ind.behavior_sig, np.ndarray) and ind.behavior_sig.size
        assert 0.0 <= ind.instability_obj <= 1.0
        # A monotone increasing model of a monotone target nails the ordering.
        assert ind.shape_obj < 1e-6, ind.shape_obj
    print("PASS test_calculate_fitness_caches_extra_objectives")


def test_extra_objectives_module_default_off():
    """Direct / non-interactive callers default to OFF (read in a fresh
    subprocess so other tests toggling the flags can't taint it)."""
    import subprocess, sys
    out = subprocess.run(
        [sys.executable, "-c",
         "import evo13; print(evo13.NSGA3_EXTRA_OBJ_ENABLED, "
         "len(evo13.NSGA3_EXTRA_OBJECTIVES))"],
        capture_output=True, text=True)
    assert out.stdout.strip() == "False 0", out.stdout + out.stderr
    print("PASS test_extra_objectives_module_default_off")


# ──────────────────────────────────────────────────────────────────────────
# Part 3 — A-NSGA-III is the interactive default; extra-objective prompt wiring
# ──────────────────────────────────────────────────────────────────────────
def test_interactive_variant_default_is_a():
    """Pressing Enter at the module prompt now selects A-NSGA-III (the default),
    and leaving the extra-objectives prompt blank keeps them off."""
    with _saved_nsga3_globals():
        # answers: use NSGA-III? / divisions / module / extra-objectives
        with feed_input(["", "", "", ""]):
            e._select_nsga3_mode()
        assert e.NSGA3_ENABLED is True
        assert e.NSGA3_VARIANT == "a"
        assert e.NSGA3_EXTRA_OBJ_ENABLED is False
        assert e.NSGA3_EXTRA_OBJECTIVES == ()
    print("PASS test_interactive_variant_default_is_a")


def test_interactive_basic_still_selectable():
    with _saved_nsga3_globals():
        with feed_input(["y", "", "basic", ""]):
            e._select_nsga3_mode()
        assert e.NSGA3_VARIANT == "basic"
    print("PASS test_interactive_basic_still_selectable")


def test_interactive_extra_objectives_all_and_subset():
    with _saved_nsga3_globals():
        with feed_input(["", "", "a", "all"]):
            e._select_nsga3_mode()
        assert e.NSGA3_EXTRA_OBJ_ENABLED is True
        assert set(e.NSGA3_EXTRA_OBJECTIVES) == {
            'instability', 'shape', 'diversity'}
    with _saved_nsga3_globals():
        with feed_input(["", "", "a", "i,d"]):
            e._select_nsga3_mode()
        assert e.NSGA3_EXTRA_OBJ_ENABLED is True
        assert e.NSGA3_EXTRA_OBJECTIVES == ('instability', 'diversity')
    print("PASS test_interactive_extra_objectives_all_and_subset")


def test_interactive_disable_clears_extras():
    with _saved_nsga3_globals():
        # First arm everything…
        e.NSGA3_EXTRA_OBJ_ENABLED = True
        e.NSGA3_EXTRA_OBJECTIVES = ('shape',)
        # …then decline NSGA-III: extras must be cleared with it.
        with feed_input(["n"]):
            e._select_nsga3_mode()
        assert e.NSGA3_ENABLED is False
        assert e.NSGA3_EXTRA_OBJ_ENABLED is False
        assert e.NSGA3_EXTRA_OBJECTIVES == ()
    print("PASS test_interactive_disable_clears_extras")


if __name__ == "__main__":
    # Part 1
    test_new_ops_registered_everywhere()
    test_new_ops_numeric_values()
    test_new_ops_are_finite_in_safe_mode()
    test_new_ops_export_roundtrip()
    test_new_ops_string_repr()
    # Part 2
    test_behavior_signature_is_zscored_and_degenerate_safe()
    test_extra_objective_columns_gated_and_shaped()
    test_diversity_objective_penalises_clones()
    test_trim_with_extras_holds_target_and_keeps_champion()
    test_extras_reshape_selection_vs_3obj()
    test_calculate_fitness_caches_extra_objectives()
    test_extra_objectives_module_default_off()
    # Part 3
    test_interactive_variant_default_is_a()
    test_interactive_basic_still_selectable()
    test_interactive_extra_objectives_all_and_subset()
    test_interactive_disable_clears_extras()
    print("\nALL NEW-OPS / EXTRA-OBJECTIVE TESTS PASSED")
