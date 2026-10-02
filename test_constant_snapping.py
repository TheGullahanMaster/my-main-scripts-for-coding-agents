"""Final constant snapping and the --fit-iterations / --semantic-max-delta knobs.

Run with: python -B -m unittest -v test_constant_snapping
"""
import math
import unittest
from unittest.mock import patch

import numpy as np

import afpo as a

CATS = [None]


def model(tree):
    return a.Model(trees=[tree], scales=[(1., 0.)])


def snap(tree, Xt, Yt, Xv=None, Yv=None):
    return a.snap_model_constants(model(tree), Xt, Yt, Xv, Yv, True, CATS)


class SnapValuesTest(unittest.TestCase):
    def test_simplest_first(self):
        self.assertEqual(a.snap_values(2.0003)[0], 2.0)
        self.assertEqual(a.snap_values(0.004)[0], 0.0)
        self.assertIn(math.pi, a.snap_values(3.14159))
        self.assertIn(0.5, a.snap_values(0.512))
        self.assertNotIn(0.512, a.snap_values(0.512))

    def test_window_bounds_candidates(self):
        for value in (-7.3, 0.003, 12.6, 1e5 + 0.4):
            for option in a.snap_values(value):
                self.assertLessEqual(abs(option - value), a.SNAP_WINDOW * max(abs(value), 1.) + 1e-12)


class SnapModelTest(unittest.TestCase):
    def setUp(self):
        self.X = np.linspace(-2, 2, 61)[:, None]

    def test_noise_level_constant_snaps(self):
        y = np.sin(2 * self.X[:, 0])[:, None]
        snapped, count = snap(("sin", ("*", ("c", 2.0000003), ("x", 0))), self.X, y)
        self.assertEqual(count, 1)
        self.assertEqual(a.constant_vector(snapped.trees[0]), (2.0,))

    def test_real_constant_is_kept(self):
        y = np.sin(2.37 * self.X[:, 0])[:, None]
        tree = ("sin", ("*", ("c", 2.37), ("x", 0)))
        snapped, count = snap(tree, self.X, y)
        self.assertEqual(count, 0)
        self.assertEqual(snapped.trees[0], tree)

    def test_validation_blocks_threshold_snap(self):
        # Training rows leave a gap around the jump at 0.512; a validation row
        # at 0.505 sits inside it, so 0.5 matches training but not validation.
        Xt = np.concatenate([np.linspace(0, .45, 20), np.linspace(.55, 1, 20)])[:, None]
        Xv = np.array([[.2], [.505], [.8]])
        target = lambda X: (X[:, 0] > .512).astype(float)[:, None]
        tree = ("gt", ("x", 0), ("c", .512))
        free, _ = snap(tree, Xt, target(Xt))
        self.assertEqual(a.constant_vector(free.trees[0]), (.5,))
        guarded, _ = snap(tree, Xt, target(Xt), Xv, target(Xv))
        self.assertGreater(a.constant_vector(guarded.trees[0])[0], .505)

    def test_off_mode_leaves_candidates(self):
        y = np.sin(2 * self.X[:, 0])[:, None]
        candidate = model(("sin", ("*", ("c", 2.0000003), ("x", 0))))
        a.assess(candidate, self.X, y, True, CATS)
        with patch.object(a, "CONSTANT_SNAPPING", "off"):
            result, summary = a.snap_final_candidates([candidate], self.X, y, None, None, True, CATS)
        self.assertIs(result[0], candidate)
        self.assertEqual(summary["models_tried"], 0)
        with patch.object(a, "CONSTANT_SNAPPING", "final"):
            result, summary = a.snap_final_candidates([candidate, candidate], self.X, y, None, None, True, CATS)
        self.assertEqual(summary["models_snapped"], 1)
        self.assertEqual([a.constant_vector(m.trees[0]) for m in result], [(2.0,), (2.0,)])


class KnobTest(unittest.TestCase):
    def test_cli_defaults_and_validation(self):
        args = a.parse_cli([])[1]
        self.assertEqual((args.fit_iterations, args.semantic_max_delta, args.constant_snapping, args.snap_tolerance), (12, 5., "final", 1e-6))
        self.assertEqual(a.parse_cli(["--semantic-max-delta", "inf"])[1].semantic_max_delta, float("inf"))
        for bad in (["--fit-iterations", "0"], ["--semantic-max-delta", "0"], ["--snap-tolerance", "-1"]):
            with self.assertRaises(SystemExit), patch("sys.stderr"):
                a.parse_cli(bad)

    def test_fit_iterations_read_at_call_time(self):
        X = np.linspace(.1, 3, 40)[:, None]; y = np.exp(-1.7 * X[:, 0])
        tree = ("exp", ("*", ("c", -.2), ("x", 0)))
        with patch.object(a, "JUMP_CONSTANT_SCAN", False), patch.object(a, "FIT_BACKEND", "python"):
            with patch.object(a, "CONSTANT_FIT_ITERATIONS", 1):
                short = a.constant_vector(a.fit_tree_constants(tree, X, y))[0]
            with patch.object(a, "CONSTANT_FIT_ITERATIONS", 40):
                long = a.constant_vector(a.fit_tree_constants(tree, X, y))[0]
        self.assertLess(abs(long + 1.7), abs(short + 1.7))

    def test_semantic_max_delta_read_at_call_time(self):
        X = np.linspace(-2, 2, 30)[:, None]
        portfolio = a.MutationPortfolio()
        big = ("*", ("c", 1000.), ("x", 0))
        with patch.object(portfolio, "apply", return_value=(big, "point")):
            with patch.object(a, "SEMANTIC_MAX_DELTA", 5.):
                self.assertIsNone(a.semantic_mutate(("x", 0), X, portfolio, 1, ["*"], 9, 4)[1])
            with patch.object(a, "SEMANTIC_MAX_DELTA", float("inf")):
                self.assertEqual(a.semantic_mutate(("x", 0), X, portfolio, 1, ["*"], 9, 4)[0], big)


if __name__ == "__main__":
    unittest.main()
