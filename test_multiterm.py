"""Multiterm (multigene) readout: term splitting, coefficient solve, pruning, gene crossover.

Run with: python -B -m unittest -v test_multiterm
"""
import unittest
from unittest.mock import patch

import numpy as np

import afpo as a

rng = np.random.default_rng(3)
X = rng.uniform(.5, 2, size=(80, 4))


def values(tree):
    return a.evaluate_cached(tree, X, {})


class SplitJoinTest(unittest.TestCase):
    def test_round_trip_keeps_semantics(self):
        tree = ("-", ("+", ("*", ("c", 2.5), ("x", 0)), ("*", ("x", 1), ("c", -3.))), ("neg", ("sin", ("x", 2))))
        terms = a.split_terms(tree)
        self.assertEqual([c for c, _ in terms], [2.5, -3., 1.])
        np.testing.assert_allclose(values(a.join_terms(terms)), values(tree), rtol=1e-12)

    def test_constants_are_split_off(self):
        self.assertEqual(a.split_terms(("+", ("x", 0), ("c", 4.))), [(1., ("x", 0)), (4., None)])


    def test_join_respects_the_grammar(self):
        terms = [(-2., ("x", 0)), (1., ("x", 1)), (-1., ("x", 2))]
        joined = a.join_terms(terms, ("+", "*"))
        self.assertTrue(all(op not in repr(joined) for op in ("'-'", "'neg'")))
        np.testing.assert_allclose(values(joined), values(a.join_terms(terms)), rtol=1e-12)
        self.assertIsNone(a.join_terms([(2., ("x", 0)), (1., ("x", 1))], ("+",)))
        self.assertIsNone(a.join_terms([(1., ("x", 0)), (1., ("x", 1))], ("*",)))


class RefitTest(unittest.TestCase):
    def test_recovers_term_coefficients(self):
        y = 3.7 * X[:, 0] * X[:, 1] - 0.25 * X[:, 2] / X[:, 3] + 1.2
        tree = ("+", ("*", ("x", 0), ("x", 1)), ("/", ("x", 2), ("x", 3)))
        refit = a.multiterm_refit(tree, X, y)
        coefficients = sorted(c for c, body in a.split_terms(refit) if body is not None)
        np.testing.assert_allclose(coefficients, [-0.25, 3.7], rtol=1e-8)

    def test_prunes_collinear_duplicate(self):
        y = 2 * X[:, 0] + np.sin(X[:, 1])
        tree = ("+", ("+", ("x", 0), ("*", ("c", 1.0000000001), ("x", 0))), ("sin", ("x", 1)))
        refit = a.multiterm_refit(tree, X, y)
        bodies = [body for _, body in a.split_terms(refit) if body is not None]
        self.assertEqual(len(bodies), 2)
        self.assertLess(max(abs(c) for c, b in a.split_terms(refit) if b is not None), 10)

    def test_max_terms_is_enforced(self):
        y = X[:, 0] + X[:, 1] + X[:, 2] + 0.001 * X[:, 3]
        tree = a.join_terms([(1., ("x", i)) for i in range(4)])
        with patch.object(a, "MAX_TERMS", 3):
            refit = a.multiterm_refit(tree, X, y)
        self.assertEqual(len([b for _, b in a.split_terms(refit) if b is not None]), 3)

    def test_refit_stays_inside_the_model_grammar(self):
        y = (2 * X[:, 0] - 3 * X[:, 1])[:, None]
        model = a.Model(trees=[("+", ("x", 0), ("x", 1))], scales=[(1., 0.)], mdl_operators=("+", "*"))
        with patch.object(a, "READOUT_MODE", "multiterm"):
            a.tune_model_constants(model, X, y, True, [None])
        self.assertNotIn("'-'", repr(model.trees)); self.assertNotIn("'neg'", repr(model.trees))
        a.model_description(model, 4)  # raises if an operator left the grammar

    def test_single_term_unchanged(self):
        tree = ("sin", ("x", 0))
        self.assertIs(a.multiterm_refit(tree, X, np.sin(X[:, 0])), tree)

    def test_tuning_uses_multiterm_only_when_enabled(self):
        y = (3.7 * X[:, 0] * X[:, 1] - 0.25 * X[:, 2] / X[:, 3])[:, None]
        tree = ("+", ("*", ("x", 0), ("x", 1)), ("/", ("x", 2), ("x", 3)))
        for mode, expect_exact in (("affine", False), ("multiterm", True)):
            model = a.Model(trees=[tree], scales=[(1., 0.)])
            with patch.object(a, "READOUT_MODE", mode):
                a.tune_model_constants(model, X, y, True, [None])
            a.assess(model, X, y, True, [None])
            self.assertEqual(a.aggregate_loss(model) < 1e-12, expect_exact, mode)


class GeneCrossoverTest(unittest.TestCase):
    def test_child_takes_a_whole_term(self):
        a.rng.seed(4)
        left = ("+", ("x", 0), ("x", 1))
        right = ("+", ("sin", ("x", 2)), ("*", ("x", 3), ("x", 3)))
        donors = [b for _, b in a.split_terms(right)]
        for _ in range(20):
            child = a.gene_crossover(left, right, 21, 6)
            self.assertTrue(any(b in donors for _, b in a.split_terms(child)))

    def test_cli(self):
        args = a.parse_cli(["--readout", "multiterm", "--max-terms", "6"])[1]
        self.assertEqual((args.readout, args.max_terms, args.gene_crossover_rate), ("multiterm", 6, 0.))
        self.assertEqual(a.parse_cli([])[1].readout, "multiterm")
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            a.parse_cli(["--max-terms", "1"])


if __name__ == "__main__":
    unittest.main()
