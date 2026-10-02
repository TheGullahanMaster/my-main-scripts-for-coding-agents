"""Semantic backpropagation mutation: inverses, library matching, and wiring.

Run with: python -B -m unittest -v test_backprop
"""
import unittest
from unittest.mock import patch

import numpy as np

import afpo as a

rng = np.random.default_rng(5)
X = rng.uniform(.5, 2, size=(60, 3))
OPS = ["+", "-", "*", "/", "sin", "exp", "log", "square", "sqrt", "tanh"]


def values(tree):
    return a.evaluate_cached(tree, X, {})


class InverseTest(unittest.TestCase):
    def check(self, op, k, args, inverse):
        target = a.op_eval(op, [np.asarray(v, float) for v in args])
        child = inverse(op, k, target, args)
        restored = [np.asarray(v, float) for v in args]
        restored[k] = child
        ok = np.isfinite(child)
        self.assertGreater(ok.mean(), .9, op)
        np.testing.assert_allclose(a.op_eval(op, restored)[ok], target[ok], rtol=1e-6, atol=1e-8, err_msg=op)

    def test_exact_inverses_round_trip(self):
        u, v = X[:, 0], X[:, 1]
        for op, k in (("+", 0), ("+", 1), ("-", 0), ("-", 1), ("*", 0), ("*", 1), ("/", 0), ("/", 1)):
            self.check(op, k, [u, v], a.exact_inverse)
        for op in ("neg", "exp", "log", "tanh", "sigmoid"):
            self.check(op, 0, [u - 1.2], a.exact_inverse)

    def test_exact_mode_refuses_branchy_ops(self):
        with patch.object(a, "BACKPROP_INVERSE", "exact"):
            self.assertIsNone(a.child_desired("sin", 0, np.zeros(3), [np.zeros(3)]))

    def test_numeric_inverse_stays_on_current_branch(self):
        current = X[:, 0] + 2 * np.pi  # sin is periodic: the nearest root is the one near current
        target = np.sin(current + .05)
        child = a.numeric_inverse("sin", 0, target, [current])
        np.testing.assert_allclose(np.sin(child), target, atol=1e-6)
        self.assertLess(np.max(np.abs(child - current)), .2)
        self.check("square", 0, [-X[:, 1]], a.numeric_inverse)
        self.check("pow", 1, [X[:, 0] + 1, X[:, 1]], a.numeric_inverse)


class MutationTest(unittest.TestCase):
    def setUp(self):
        a.rng.seed(2)

    def test_finds_missing_factor(self):
        y = np.sin(X[:, 0]) * X[:, 1] / X[:, 2]
        parent = ("*", ("sin", ("x", 0)), ("x", 1))  # needs x1/x2 where x1 is
        a.set_backprop_context(X, y)
        try:
            best = min((a.backprop_mutate(parent, OPS, 21, 6) for _ in range(30)),
                       key=lambda t: float(np.mean((values(t) - y) ** 2)))
        finally:
            a.set_backprop_context()
        np.testing.assert_allclose(values(best), y, rtol=1e-9)

    def test_without_context_returns_parent(self):
        parent = ("x", 0)
        a.set_backprop_context()
        self.assertIs(a.backprop_mutate(parent, OPS, 21, 6), parent)

    def test_desired_inverts_readout(self):
        model = a.Model(trees=[("x", 0)], scales=[(2., 1.)])
        Y = np.array([[3.], [5.]])
        np.testing.assert_allclose(a.backprop_desired(model, 0, Y, {0: 0}), [1., 2.])
        model.scales = [(0., 1.)]
        self.assertIsNone(a.backprop_desired(model, 0, Y, {0: 0}))

    def test_portfolio_and_cli(self):
        with patch.object(a, "BACKPROP_MUTATION_WEIGHT", 0.):
            self.assertEqual(a.MutationPortfolio().weights["backprop"], 0.)
        args = a.parse_cli(["--backprop-mutation-weight", "1", "--backprop-inverse", "exact"])[1]
        self.assertEqual((args.backprop_mutation_weight, args.backprop_inverse), (1., "exact"))


if __name__ == "__main__":
    unittest.main()
