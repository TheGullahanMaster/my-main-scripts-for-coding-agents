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
        tree = ("+", ("log", ("*", ("x", 0), ("x", 0))), ("sqrt", ("x", 1)))
        model = a.Model(trees=[tree], scales=[(2., 1.)])
        result = a.symbolic_model(model, ["u", "v"], ["y"], [None], positive=(0, 1))
        text = str(result["y"][0])
        self.assertNotIn("Abs", text)
        self.assertNotIn("sign", text)
        self.assertIn("log", result["y"][1])

    def test_writes_file(self):
        model = a.Model(trees=[("*", ("x", 0), ("x", 1))], scales=[(3., 0.)])
        X = np.ones((4, 2))
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            path = os.path.join(directory, "out.txt")
            a.write_symbolic_export(model, ["u", "v"], ["y"], [None], X, path)
            with open(path) as handle:
                content = handle.read()
        self.assertIn("3.0*u*v", content)
        self.assertIn("LaTeX", content)


if __name__ == "__main__":
    unittest.main()
