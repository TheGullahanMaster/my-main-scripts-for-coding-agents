"""Class balancing for categorical outputs (--class-balance).

Run with: python -B -m unittest -v test_class_balance
"""
import unittest
from unittest.mock import patch

import numpy as np

import afpo as a
from test_afpo import model


class ClassBalanceWeightTests(unittest.TestCase):
    def test_every_present_class_gets_equal_total_weight(self):
        truth = np.r_[np.zeros(90), np.ones(10), [-1.]]
        w = a.class_balance_weights(truth, 3)
        self.assertAlmostEqual(float(w[:90].sum()), float(w[90:100].sum()))
        self.assertAlmostEqual(float(w[:100].mean()), 1.)
        self.assertEqual(w[-1], 1.)  # an unseen label keeps its unit weight

    def test_off_gives_uniform_weights(self):
        with patch.object(a, "CLASS_BALANCE", False):
            self.assertTrue(np.all(a.class_balance_weights(np.r_[np.zeros(9), 1.], 2) == 1.))

    def test_balanced_rows_repeat_the_rare_class(self):
        X = np.arange(100.)[:, None]; truth = np.r_[np.zeros(95), np.ones(5)]
        rows = a.class_balanced_rows(X, truth, 2, 40)
        self.assertEqual(int(np.sum(truth[rows] == 0)), int(np.sum(truth[rows] == 1)))
        self.assertEqual(set(rows[truth[rows] == 1]), set(range(95, 100)))


class BalancedScoringTests(unittest.TestCase):
    X = np.arange(100.)[:, None]
    Y = np.r_[np.zeros(90), np.ones(10)][:, None]
    cats = [["common", "rare"]]

    def test_majority_only_model_scores_a_balanced_error_rate(self):
        m = model(("c", 0.))
        a.assess(m, self.X, self.Y, False, self.cats)
        self.assertAlmostEqual(m.objectives[1], .5)
        with patch.object(a, "CLASS_BALANCE", False):
            a.assess(m, self.X, self.Y, False, self.cats)
        self.assertAlmostEqual(m.objectives[1], .1)

    def test_readout_offset_drops_the_class_prior(self):
        raw = np.zeros((100, 1))
        _, offset = a.fit_classifier_affine(raw, self.Y[:, 0], 2)[0]
        self.assertAlmostEqual(offset - .5, 0., places=2)
        with patch.object(a, "CLASS_BALANCE", False):
            _, offset = a.fit_classifier_affine(raw, self.Y[:, 0], 2)[0]
        self.assertAlmostEqual(offset - .5, np.log(10 / 90), places=1)

    def test_lexicase_examines_rare_class_rows_first_as_often(self):
        X = np.arange(20.)[:, None]; Y = np.r_[np.zeros(18), np.ones(2)][:, None]
        majority = model(("c", 0.))                      # misses both rare rows
        catches_rare = model(("-", ("x", 0), ("c", 13.)))  # catches them, misses 4 common rows
        a.rng.seed(3); np.random.seed(3)
        balanced = a.lexicase_parents([majority, catches_rare], 600, X, Y, self.cats)
        with patch.object(a, "CLASS_BALANCE", False):
            plain = a.lexicase_parents([majority, catches_rare], 600, X, Y, self.cats)
        self.assertGreater(sum(p is catches_rare for p in balanced) / 600, .7)
        self.assertLess(sum(p is catches_rare for p in plain) / 600, .45)


class ClassificationSearchTests(unittest.TestCase):
    def test_residual_signature_tells_which_class_is_missed(self):
        X = np.arange(40.)[:, None]; Y = np.r_[np.zeros(30), np.ones(10)][:, None]
        archive = a.ResidualQualityDiversityArchive(X, Y, [["a", "b"]], 1)
        misses_b = model(("c", 0.)); misses_a = model(("c", 1.))
        self.assertGreater(np.abs(archive.descriptor(misses_b) - archive.descriptor(misses_a)).max(), .5)
        legacy = a.ResidualQualityDiversityArchive.from_snapshot({**archive.snapshot(), "class_groups": None})
        self.assertEqual(len(legacy.groups), len(archive.groups) - 1)

    def test_sparse_seeding_fits_classifier_heads(self):
        r = np.random.default_rng(0); X = r.uniform(-1, 1, (200, 2))
        Y = (X[:, 0] + .5 * X[:, 1] > .8).astype(float)[:, None]
        seeds = a.sparse_seed_models(X, Y, [["no", "yes"]], ["+", "-", "*"], 15, 5, 4, 1)
        self.assertTrue(seeds)
        used = {node[1] for node in a.walk_tree(seeds[0].trees[0]) if node[0] == "x"}
        self.assertEqual(used, {0, 1})


class StratifiedSplitTests(unittest.TestCase):
    def test_each_class_keeps_its_share_and_a_training_row(self):
        strata = np.array(["a"] * 94 + ["b"] * 5 + ["c"])
        train, validation = a.holdout_split_indices(100, 20, 7, strata)
        self.assertEqual(sorted(np.r_[train, validation].tolist()), list(range(100)))
        self.assertEqual(len(validation), 20)
        self.assertEqual(int(np.sum(strata[validation] == "b")), 1)
        self.assertIn(99, train)  # a single-row class is never held out
        again = a.holdout_split_indices(100, 20, 7, strata)
        self.assertTrue(np.array_equal(train, again[0]) and np.array_equal(validation, again[1]))

    def test_without_strata_the_split_is_unchanged(self):
        indices = np.random.default_rng(5).permutation(50)
        train, validation = a.holdout_split_indices(50, 10, 5)
        self.assertTrue(np.array_equal(train, indices[:-10]) and np.array_equal(validation, indices[-10:]))


if __name__ == "__main__":
    unittest.main()
