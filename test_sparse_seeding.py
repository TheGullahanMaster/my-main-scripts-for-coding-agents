"""Sparse-regression seeding: basis, matching pursuit, and seed models.

Run with: python -B -m unittest -v test_sparse_seeding
"""
import unittest
from unittest.mock import patch

import numpy as np

import afpo as a

rng = np.random.default_rng(7)
X = rng.uniform(.5, 2, size=(200, 4))
OPS = ["+", "-", "*", "/", "sin", "exp", "max"]


class BasisTest(unittest.TestCase):
    def test_basis_is_modest_and_unique(self):
        trees, B = a.sparse_basis(X, OPS)
        self.assertEqual(B.shape, (200, len(trees)))
        self.assertIn(("*", ("x", 0), ("x", 1)), trees)
        self.assertIn(("/", ("x", 2), ("x", 3)), trees)
        # Nothing is composed on top of a basis term.
        self.assertTrue(all(a.node_depth(t) <= 1 for t in trees))
        self.assertLessEqual(len(a.sparse_basis(X, OPS, limit=10)[0]), 10)


class FitTest(unittest.TestCase):
    def test_recovers_sum_of_products(self):
        y = X[:, 0] * X[:, 1] + X[:, 2] / X[:, 3] - 0.5 * X[:, 3] * X[:, 3]
        trees, B = a.sparse_basis(X, OPS)
        bic, support, coefficients, intercept, r2 = a.sparse_fits(B, y)[0]
        self.assertGreater(r2, 1 - 1e-10)
        self.assertEqual({trees[i] for i in support},
                         {("*", ("x", 0), ("x", 1)), ("/", ("x", 2), ("x", 3)), ("*", ("x", 3), ("x", 3))})

    def test_seed_models(self):
        Y = (X[:, 0] * X[:, 1] + np.sin(X[:, 2]))[:, None]
        a.SPARSE_SEED_STATS.update(seeds=0, best_r2=None, basis=0)
        models = a.sparse_seed_models(X, Y, [None], OPS, 21, 6, 5, 1)
        self.assertTrue(1 <= len(models) <= 5)
        self.assertTrue(all(m.origin == "sparse_seed" for m in models))
        a.assess(models[0], X, Y, True, [None])
        self.assertLess(a.aggregate_loss(models[0]), 1e-12)
        self.assertGreater(a.SPARSE_SEED_STATS["best_r2"], 1 - 1e-10)

    def test_classifier_heads_fit_their_class_indicator(self):
        Y = np.column_stack([(X[:, 0] > 1).astype(float)])
        seeds = a.sparse_seed_models(X, Y, [["no", "yes"]], OPS, 21, 6, 5, 1)
        self.assertTrue(seeds)
        self.assertIn(("x", 0), list(a.walk_tree(seeds[0].trees[0])))

    def test_single_class_outputs_are_skipped(self):
        Y = np.zeros((len(X), 1))
        self.assertEqual(a.sparse_seed_models(X, Y, [["only"]], OPS, 21, 6, 5, 1), [])

    def test_cli(self):
        args = a.parse_cli(["--sparse-seeding", "on", "--sparse-basis-size", "50"])[1]
        self.assertEqual((args.sparse_seeding, args.sparse_basis_size), ("on", 50))
        self.assertEqual(a.parse_cli([])[1].sparse_seeding, "off")


if __name__ == "__main__":
    unittest.main()
