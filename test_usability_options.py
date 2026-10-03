"""Nesting limits, stop rules and the SymPy/LaTeX export.

Run with: python -B -m unittest -v test_usability_options
"""
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import afpo as a


class NestingTest(unittest.TestCase):
    def test_parse_and_detect(self):
        rules = a.parse_nesting_rules("exp>exp, sin>cos")
        self.assertEqual(rules, {("exp", "exp"), ("sin", "cos")})
        with patch.object(a, "NESTING_RULES", rules):
            self.assertEqual(a.nesting_violation(("exp", ("+", ("x", 0), ("exp", ("x", 1))))), "exp>exp")
            self.assertEqual(a.nesting_violation(("sin", ("*", ("c", 2.), ("cos", ("x", 0))))), "sin>cos")
            self.assertEqual(a.nesting_violation(("cos", ("sin", ("x", 0)))), "")
        with self.assertRaises(ValueError):
            a.parse_nesting_rules("exp-exp")

    def test_assess_marks_violations_infeasible(self):
        X = np.linspace(0, 1, 10)[:, None]; Y = X.copy()
        model = a.Model(trees=[("exp", ("exp", ("x", 0)))], scales=[(1., 0.)])
        with patch.object(a, "NESTING_RULES", frozenset({("exp", "exp")})):
            a.assess(model, X, Y, True, [None])
        self.assertFalse(model.feasible)
        self.assertEqual(model.invalid_reason, "nesting:exp>exp")

    def test_cli_rejects_unknown_operator(self):
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            a.parse_cli(["--forbid-nesting", "foo>exp"])


class StopRuleTest(unittest.TestCase):
    def island(self, loss):
        model = a.Model(trees=[("x", 0)], scales=[(1., 0.)], objectives=(loss, 0., 3., 0))
        return SimpleNamespace(best_models=SimpleNamespace(model=model))

    def test_time_and_loss(self):
        args = SimpleNamespace(max_time=0., stop_at_loss=None)
        with patch("builtins.print"):
            self.assertFalse(a.stop_rule_reached(args, time.time(), [self.island(1.)]))
            args.max_time = 5
            self.assertTrue(a.stop_rule_reached(args, time.time() - 6, [self.island(1.)]))
            args.max_time = 0; args.stop_at_loss = 1e-3
            self.assertFalse(a.stop_rule_reached(args, time.time(), [self.island(1.)]))
            self.assertTrue(a.stop_rule_reached(args, time.time(), [self.island(1.), self.island(1e-4)]))


@unittest.skipUnless(__import__("importlib").util.find_spec("sympy"), "sympy not installed")
class SymbolicTest(unittest.TestCase):
    def test_positive_inputs_simplify(self):
        import sympy as sp
        tree = ("+", ("log", ("*", ("x", 0), ("x", 0))), ("sqrt", ("x", 1)))
        model = a.Model(trees=[tree], scales=[(2., 1.)])
        result = a.symbolic_model(model, ["u", "v"], ["y"], [None], positive=(0, 1))["y"]
        self.assertNotIn("Abs", str(result["exact"][0]))
        self.assertNotIn("sign", str(result["exact"][0]))
        u, v = sp.symbols("u v", positive=True)
        self.assertEqual(result["raw"][0], 4 * sp.log(u) + 2 * sp.sqrt(v) + 1)
        self.assertIn(r"\ln", result["raw"][1])

    def test_raw_form_drops_guards_and_reads_constants(self):
        tree = ("*", ("c", 3.14159265358979), ("/", ("log", ("x", 0)), ("x", 1)))
        model = a.Model(trees=[tree], scales=[(0.333333333, 0.)])
        result = a.symbolic_model(model, ["x1", "speed_ms"], ["y"], [None])["y"]
        self.assertEqual(result["raw"][1], r"\frac{\pi\,\ln{\left(x_{1} \right)}}{3\,\mathrm{speed\_ms}}")
        self.assertIn("Piecewise", str(result["exact"][0]))  # guarded division
        self.assertIn("1.0e-12", str(result["exact"][0]))  # log(|x| + eps)

    def test_multi_letter_names_stay_one_word(self):
        import sympy as sp
        self.assertEqual(a.latex_symbol_name("x3"), "x_{3}")
        self.assertEqual(a.latex_symbol_name("theta2"), r"\theta_{2}")
        self.assertEqual(a.latex_symbol_name("foobar"), r"\mathrm{foobar}")
        speed = sp.Symbol("speed_ms")
        self.assertIn('<mi mathvariant="normal">speed_ms</mi>', a.mathml_expression(speed ** 2, [speed]))

    def test_readable_constants(self):
        import sympy as sp
        self.assertEqual(a.readable_constant(0.75), sp.Rational(3, 4))
        self.assertEqual(a.readable_constant(-0.66666666667), sp.Rational(-2, 3))
        self.assertEqual(a.readable_constant(1.570796327), sp.pi / 2)
        self.assertEqual(str(a.readable_constant(1.2345432)), "1.23454")
        self.assertEqual(str(a.readable_constant(0.6666667, exact=True)), "0.6666667")
        self.assertEqual(a.readable_constant(0.1, exact=True), sp.Rational(1, 10))

    def test_exact_form_reproduces_the_model(self):
        import sympy as sp
        rng = np.random.default_rng(0)
        X = rng.uniform(-3, 3, size=(200, 2)); X[:4, 1] = 0
        tree = ("+", ("exp", ("pow", ("x", 0), ("c", 1.3))), ("/", ("log", ("x", 0)), ("sqrt", ("x", 1))))
        model = a.Model(trees=[tree], scales=[(.7, -1.2)])
        result = a.symbolic_model(model, ["u", "v"], ["y"], [None], X=X)["y"]
        u, v = sp.symbols("u v", real=True)
        values = sp.lambdify((u, v), result["exact"][0], "numpy")(X[:, 0], X[:, 1])
        # Rows with v = 0 hit the guarded division, where the per-node +/-1e12 clamp (left out) decides.
        np.testing.assert_allclose(values[4:], a.predict_model(model, X)[4:, 0], rtol=1e-9)
        share, gap = result["agreement"]  # the raw form is undefined on most of these rows
        self.assertLess(share, .5)

    def test_failed_conversion_does_not_raise(self):
        model = a.Model(trees=[("*", ("x", 0), ("x", 1))], scales=[(3., 0.)])
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"), \
                patch.object(a, "_symbolic_form", side_effect=ValueError("boom")):
            path = os.path.join(directory, "out.txt")
            result = a.write_symbolic_export(model, ["u", "v"], ["y"], [None], np.ones((4, 2)), path)
            with open(path) as handle:
                content = handle.read()
        self.assertEqual(result["y"]["errors"]["raw"], "ValueError: boom")
        self.assertIn("y (raw): not converted (ValueError: boom)", content)

    def test_slow_simplification_is_cut_off(self):
        import time
        with self.assertRaises(a._SymbolicTimeout):
            a._within_time_limit(lambda: time.sleep(2), .2)

    def test_writes_file(self):
        model = a.Model(trees=[("*", ("x", 0), ("x", 1))], scales=[(3., 0.)])
        X = np.ones((4, 2))
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            path = os.path.join(directory, "out.txt")
            a.write_symbolic_export(model, ["u", "v"], ["y"], [None], X, path)
            with open(path) as handle:
                content = handle.read()
        self.assertIn("y (exact) = 3*u*v", content)
        self.assertIn("y (raw) = 3*u*v", content)
        self.assertIn(r"LaTeX (raw): y = 3\,u\,v", content)
        self.assertIn("defined on 100.0% of rows", content)


class LossModeTest(unittest.TestCase):
    def test_squared_and_relative_losses(self):
        y = np.array([1., 10., 100., 1000.]); pred = y * 1.01
        huber = a.robust_loss(pred, y)
        with patch.object(a, "LOSS_MODE", "relative"):
            relative = a.robust_loss(pred, y)
        self.assertAlmostEqual(relative, .5 * .01 ** 2, places=12)
        self.assertNotAlmostEqual(huber, relative)
        outlier = np.array([0., 0., 0., 100.]); flat = np.zeros(4)
        with patch.object(a, "LOSS_MODE", "squared"):
            squared = a.robust_loss(flat, outlier)
        self.assertGreater(squared, a.robust_loss(flat, outlier))

    def test_relative_fit_matches_small_targets(self):
        # Targets over six decades: a relative fit must not ignore the small rows.
        X = np.logspace(0, 3, 40)[:, None]; y = 2.5 * X[:, 0] ** 2
        tree = ("pow", ("x", 0), ("c", 1.7))
        with patch.object(a, "LOSS_MODE", "relative"), patch.object(a, "JUMP_CONSTANT_SCAN", False):
            fitted = a.fit_tree_constants(tree, X, y, iterations=60)
            a_, b_ = a.affine(a.evaluate_cached(fitted, X, {}), y)
        self.assertAlmostEqual(a.constant_vector(fitted)[0], 2., places=4)
        prediction = a_ * a.evaluate_cached(fitted, X, {}) + b_
        self.assertLess(np.max(np.abs(prediction / y - 1)), 1e-3)

    def test_cli(self):
        args = a.parse_cli(["--loss", "relative", "--huber-delta", "3"])[1]
        self.assertEqual((args.loss, args.huber_delta), ("relative", 3.))


class IntervalTest(unittest.TestCase):
    def test_interval_covers_true_constant(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 2, size=(200, 1)); Y = (3 * np.exp(-1.3 * X[:, 0]) + rng.normal(0, .01, 200))[:, None]
        model = a.Model(trees=[("exp", ("*", ("c", -1.3), ("x", 0)))], scales=[(3., 0.)])
        report = a.constant_intervals(model, X, Y, [None])
        constant = report[0]
        self.assertEqual(constant["kind"], "constant")
        self.assertLess(constant["low"], -1.3); self.assertGreater(constant["high"], -1.3)
        self.assertLess(constant["high"] - constant["low"], .05)

    def test_unidentified_constant(self):
        X = np.linspace(0, 1, 20)[:, None]; Y = X.copy()
        model = a.Model(trees=[("+", ("x", 0), ("*", ("c", 0.), ("gt", ("x", 0), ("c", 5.))))], scales=[(1., 0.)])
        report = a.constant_intervals(model, X, Y, [None])
        self.assertTrue(any(item["se"] is None for item in report if item["kind"] == "constant"))


class UnitTest(unittest.TestCase):
    def setUp(self):
        a.configure_units("x=m,t=s,v=m/s^1,k=1/s", ["x", "t", "v", "k", "free"])

    def tearDown(self):
        a.configure_units("", [])

    def test_parse(self):
        self.assertEqual(a.parse_unit("kg*m/s^2"), {"kg": 1., "m": 1., "s": -2.})
        self.assertEqual(a.parse_unit("m^(1/2)"), {"m": .5})
        self.assertEqual(a.parse_unit("1/s"), {"s": -1.})
        self.assertEqual(a.parse_unit("1"), {})

    def test_consistent_trees_pass(self):
        for tree in (("+", ("x", 0), ("*", ("x", 2), ("x", 1))),          # x + v*t
                     ("exp", ("neg", ("*", ("x", 3), ("x", 1)))),          # exp(-k t)
                     ("sin", ("*", ("c", 2.), ("x", 0))),                  # constant carries 1/m
                     ("gt", ("x", 0), ("c", 1.)),
                     ("sqrt", ("*", ("x", 0), ("x", 0))),
                     ("+", ("x", 4), ("x", 0)),                            # unlisted column is free
                     ("exp_decay", ("x", 1), ("x", 3)),                    # exp(-t k): s * 1/s
                     ("exp_decay", ("x", 3), ("x", 1)),
                     ("exp_decay", ("x", 1), ("c", .5)),                   # constant carries 1/s
                     ("+", ("x", 0), ("round2", ("x", 0), ("c", 2.))),     # round2 keeps m
                     ("+", ("x", 0), ("perceptronReLU2", ("x", 0), ("c", 1.)))):
            self.assertEqual(a.unit_violation(tree), "", tree)

    def test_inconsistent_trees_fail(self):
        for tree in (("+", ("x", 0), ("x", 1)),                            # m + s
                     ("exp", ("x", 0)),                                    # exp(m)
                     ("gt", ("x", 0), ("x", 2)),                           # m > m/s
                     ("pow", ("x", 0), ("x", 1)),                          # m ** s
                     ("exp_decay", ("x", 0), ("x", 1)),                    # exp(-m s)
                     ("exp_decay", ("x", 1), ("x", 1)),                    # exp(-s^2)
                     ("round2", ("x", 0), ("x", 1)),                       # s decimal places
                     ("perceptronReLU2", ("x", 0), ("x", 1))):             # m + s
            self.assertNotEqual(a.unit_violation(tree), "", tree)

    def test_assess_and_cli(self):
        X = np.ones((4, 5)); Y = np.ones((4, 1))
        model = a.Model(trees=[("+", ("x", 0), ("x", 1))], scales=[(1., 0.)])
        a.assess(model, X, Y, True, [None])
        self.assertTrue(model.invalid_reason.startswith("units:"))
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            a.parse_cli(["--units", "x=m$"])


class CacheMemoryTest(unittest.TestCase):
    def setUp(self):
        self.budget = a.EVALUATION_CACHE_ELEMENTS
        self.addCleanup(lambda: setattr(a, "EVALUATION_CACHE_ELEMENTS", self.budget))
        self.addCleanup(a._EVALUATION_CACHE.clear)

    def test_tree_keys_are_compact_digests_of_the_full_identity(self):
        big = ("x", 0)
        for i in range(400): big = ("+", big, ("c", float(i)))
        self.assertEqual(len(a.tree_digest(big)), 16)
        self.assertIs(a.tree_fingerprint(big), a.tree_fingerprint(big))          # memoized per tree object
        self.assertNotEqual(a.tree_digest(("c", 0.)), a.tree_digest(("c", -0.)))  # repr decides identity
        self.assertNotEqual(a.tree_digest(("c", 1.)), a.tree_digest(("c", 1)))
        self.assertNotEqual(a.tree_digest([big, ("x", 1)]), a.tree_digest([("x", 1), big]))
        self.assertEqual(len(a.equivalence_key(big)), 16)
        self.assertIs(a.interned_grammar(["+", "-"]), a.interned_grammar(("+", "-")))
        X = np.ones((5, 1)); a.evaluate_cached(big, X)
        self.assertTrue(all(len(key[0]) == 16 for key in a._EVALUATION_CACHE))   # no repr text in the keys

    def test_budget_counts_entries_and_workers_get_a_share(self):
        a.configure_cache_memory(1)                                              # 125,000 elements
        X = np.ones((10, 1))
        for i in range(5000): a.evaluate_cached(("+", ("x", 0), ("c", float(i))), X)
        self.assertLessEqual(a._EVALUATION_CACHE_SIZE[0], a.EVALUATION_CACHE_ELEMENTS)
        self.assertLessEqual(len(a._EVALUATION_CACHE), a.EVALUATION_CACHE_ELEMENTS // (10 + a.EVALUATION_CACHE_ENTRY_COST) + 1)
        a.configure_cache_memory(128)
        with patch("signal.signal"): a._worker_init()
        self.assertEqual(a.EVALUATION_CACHE_ELEMENTS, int(16_000_000 * a.WORKER_CACHE_SHARE))
        self.assertFalse(a._EVALUATION_CACHE)
        with self.assertRaises(ValueError): a.configure_cache_memory(0)
        self.assertEqual(a.parse_cli(["--cache-memory", "32"])[1].cache_memory, 32.)


if __name__ == "__main__":
    unittest.main()
