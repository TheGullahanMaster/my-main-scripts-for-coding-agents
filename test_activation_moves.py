"""Squash swap, smooth swap and gate mutations, and the data-derived loss noise floor.

Run with: python -B -m unittest -v test_activation_moves
"""
import math
import unittest
from unittest.mock import patch

import numpy as np

import afpo as a

OPS = ["+", "-", "*", "/", "sigmoid", "tanh", "erf", "relu", "softplus", "abs", "sign"]
X = np.linspace(-4, 4, 41)[:, None]


def values(tree):
    return a.evaluate_cached(tree, X, {})


class SquashSwapTest(unittest.TestCase):
    def setUp(self):
        a.rng.seed(1)

    def test_swap_keeps_level_range_and_slope(self):
        parent = ("*", ("x", 0), ("sigmoid", ("*", ("c", 1.7), ("x", 0))))
        seen = set()
        for _ in range(40):
            child = a.squash_swap_mutate(parent, OPS, 21, 6)
            seen.update(node[0] for node in a.walk_tree(child))
            # erf differs from tanh by under 2% of its range at matched slope.
            self.assertLess(np.max(np.abs(values(child) - values(parent))), .05 * 4)
        self.assertTrue({"tanh", "erf"} <= seen)

    def test_tanh_to_sigmoid_is_exact(self):
        parent = ("tanh", ("x", 0))
        for _ in range(20):
            child = a.squash_swap_mutate(parent, ["+", "*", "tanh", "sigmoid"], 21, 6)
            np.testing.assert_allclose(values(child), values(parent), atol=1e-12)

    def test_no_squash_or_no_room_is_a_no_op(self):
        self.assertEqual(a.squash_swap_mutate(("relu", ("x", 0)), OPS, 21, 6), ("relu", ("x", 0)))
        tree = ("sigmoid", ("x", 0))
        self.assertEqual(a.squash_swap_mutate(tree, ["+", "*", "sigmoid", "erf"], 2, 6), tree)


class SmoothSwapAndGateTest(unittest.TestCase):
    def setUp(self):
        a.rng.seed(2)

    def test_smooth_swap_stays_close_to_the_hard_piece(self):
        for parent in (("relu", ("x", 0)), ("abs", ("x", 0)), ("sign", ("x", 0))):
            child = a.smooth_swap_mutate(parent, OPS, 21, 6)
            self.assertNotEqual(child, parent)
            self.assertLess(np.mean(np.abs(values(child) - values(parent))), .2)

    def test_gate_builds_silu_and_gelu_in_one_move(self):
        shapes = set()
        for _ in range(300):
            child = a.gate_mutate(("x", 0), 1, OPS, 21, 6)
            self.assertLessEqual(a.node_size(child), 21)
            shapes.add(child[0] if child[0] != "*" else next(n[0] for n in child[1:] if n[0] != "x"))
        self.assertTrue({"sigmoid", "+"} <= shapes)
        self.assertEqual(a.gate_mutate(("x", 0), 1, ["+", "*"], 21, 6), ("x", 0))

    def test_portfolio_weights(self):
        weights = a.MutationPortfolio().weights
        self.assertEqual(weights["squash"], a.SQUASH_SWAP_WEIGHT)
        self.assertEqual(weights["smooth"], a.SMOOTH_SWAP_WEIGHT)
        self.assertEqual(weights["gate"], a.GATE_MUTATION_WEIGHT)


class LossNoiseFloorTest(unittest.TestCase):
    y = np.linspace(-4, 4, 64) / (1 + np.exp(-np.linspace(-4, 4, 64)))

    def test_full_precision_targets_get_the_minimum(self):
        self.assertEqual(a.estimate_loss_noise_floor(self.y, [None]), a.LOSS_NOISE_FLOOR_MIN)

    def test_seven_digit_targets_sit_above_an_exact_model(self):
        rounded = np.array([float(f"{v:.7g}") for v in self.y])
        floor = a.estimate_loss_noise_floor(rounded, [None])
        self.assertGreater(floor, 10 * a.robust_loss(self.y, rounded))
        self.assertLess(floor, a.LOSS_NOISE_FLOOR_MAX)

    def test_coarse_and_categorical_targets_keep_the_old_floor(self):
        self.assertEqual(a.estimate_loss_noise_floor(np.arange(50.), [None]), a.LOSS_NOISE_FLOOR_MAX)
        self.assertEqual(a.estimate_loss_noise_floor(self.y, [{"a": 0}]), a.LOSS_NOISE_FLOOR_MAX)

    def test_final_choice_prefers_a_far_better_loss(self):
        exact = a.Model(trees=[("x", 0)], scales=[(1., 0.)])
        short = a.Model(trees=[("c", 0.)], scales=[(1., 0.)])
        entries = [(exact, exact, {"loss": 1e-25, "shape": 0., "mdl_bits": 40.}),
                   (short, short, {"loss": 1e-11, "shape": 0., "mdl_bits": 20.})]
        with patch.object(a, "LOSS_NOISE_FLOOR", 1e-9):
            self.assertIs(a._select_best(("validation", entries), .01)[0], short)
        with patch.object(a, "LOSS_NOISE_FLOOR", a.LOSS_NOISE_FLOOR_MIN):
            self.assertIs(a._select_best(("validation", entries), .01)[0], exact)


if __name__ == "__main__":
    unittest.main()
