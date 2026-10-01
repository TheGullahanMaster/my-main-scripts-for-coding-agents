"""Jump-constant scan and jump mutation (harsh, discontinuous targets).

Run with: python -B -m unittest -v test_jump_constants
"""
import unittest
from unittest.mock import patch

import numpy as np

import afpo as a


def fit(tree, X, y):
    return a.fit_tree_constants(tree, X, y)


class JumpConstantScanTest(unittest.TestCase):
    def setUp(self):
        a.rng.seed(3)

    def test_slots_cover_only_constants_under_a_jump(self):
        tree = ("+", ("mod", ("x", 0), ("c", 2.)), ("*", ("c", 3.), ("x", 0)))
        self.assertEqual(a.jump_constant_slots(a._FlatTree(tree)), [0])
        # Only the condition of if_else is a jump; its branches are smooth.
        branch = ("if_else", ("gt", ("x", 0), ("c", .1)), ("c", 2.), ("x", 0))
        self.assertEqual(a.jump_constant_slots(a._FlatTree(branch)), [0])

    def test_scan_places_a_mod_period_the_gradient_fit_cannot_move(self):
        X = np.linspace(0, 10, 64)[:, None]; y = np.mod(X[:, 0], 2.5)
        tree = ("mod", ("x", 0), ("c", 2.2))
        with patch.object(a, "JUMP_CONSTANT_SCAN", False):
            self.assertGreater(abs(fit(tree, X, y)[2][1] - 2.5), .1)   # LM stalls in a local step
        self.assertAlmostEqual(fit(tree, X, y)[2][1], 2.5, places=6)

    def test_scan_places_a_comparison_threshold_between_rows(self):
        X = np.random.default_rng(0).uniform(-1, 1, (200, 2))
        y = np.where(X[:, 0] > .4, 3 * X[:, 1], -X[:, 1])
        tree = ("if_else", ("gt", ("x", 0), ("c", -.5)), ("x", 1), ("neg", ("x", 1)))
        threshold = fit(tree, X, y)[1][2][1]
        below, above = X[X[:, 0] <= .4, 0].max(), X[X[:, 0] > .4, 0].min()
        self.assertTrue(below <= threshold < above, threshold)

    def test_smooth_trees_are_untouched(self):
        X = np.linspace(0, 2, 40)[:, None]; y = np.exp(.7 * X[:, 0])
        tree = ("exp", ("*", ("c", .5), ("x", 0)))
        with patch.object(a, "JUMP_CONSTANT_SCAN", False):
            want = fit(tree, X, y)
        before = dict(a.JUMP_SCAN_STATS)
        self.assertEqual(fit(tree, X, y), want)
        self.assertEqual(a.JUMP_SCAN_STATS, before)

    def test_scan_never_returns_a_worse_tree(self):
        X = np.linspace(0, 10, 64)[:, None]; y = np.mod(X[:, 0], 2.5)
        tree = ("mod", ("x", 0), ("c", 2.5))                        # already right
        self.assertEqual(fit(tree, X, y), tree)


class JumpMutationTest(unittest.TestCase):
    def test_off_by_default_and_respects_limits(self):
        self.assertEqual(a.MutationPortfolio().weights["jump"], a.JUMP_MUTATION_WEIGHT)
        a.rng.seed(5)
        ops = ["+", "*", "mod", "floordiv", "if_else", "gt"]
        seen = set()
        for _ in range(200):
            child = a.jump_mutate(("+", ("x", 0), ("x", 1)), 2, ops, 15, 5)
            self.assertLessEqual(a.node_size(child), 15)
            self.assertLessEqual(a.node_depth(child), 5)
            seen.update(node[0] for node in a.walk_tree(child))
        self.assertTrue({"mod", "floordiv", "if_else", "gt"} <= seen)
        # Without any jump operator in the grammar the move is a no-op.
        self.assertEqual(a.jump_mutate(("x", 0), 1, ["+", "*"], 15, 5), ("x", 0))


if __name__ == "__main__":
    unittest.main()
