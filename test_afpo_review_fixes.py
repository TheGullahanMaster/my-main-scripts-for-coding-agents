"""Regression checks for afpo.py review fixes (state leaks, caches, grammar, export).

Run with: python -B -m unittest -v test_afpo_review_fixes
"""
import contextlib
import importlib.util
import os
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import afpo as a


class RunStateTests(unittest.TestCase):
    def tearDown(self):
        a.SEQUENCE_GROUP_REQUEST = (); a.SEQUENCE_LAYOUT = None
        a.LOSS_MODE = "huber"; a.ROBUST_LOSS_DELTA = 1.5

    def test_sequence_layout_does_not_leak_into_the_next_encoding(self):
        a.SEQUENCE_GROUP_REQUEST = (("K", ["k0", "k1"]),)
        a.encode(pd.DataFrame({"k0": [1., 2, 3, 4], "k1": [2., 3, 4, 5], "y": [1., 2, 3, 4]}), [1, 1, 5])
        self.assertIsNotNone(a.SEQUENCE_LAYOUT)
        a.SEQUENCE_GROUP_REQUEST = ()
        a.encode(pd.DataFrame({"u": [1., 2, 3, 4], "y": [1., 2, 3, 4]}), [1, 5])
        self.assertIsNone(a.SEQUENCE_LAYOUT)

    def test_constant_readout_cache_respects_the_loss(self):
        y = np.array([0., 0, 0, 0, 0, 0, 0, 100.])
        a.reset_run_caches()
        a.LOSS_MODE = "squared"; squared = a.affine(np.ones(8), y)
        a.LOSS_MODE = "huber"; huber = a.affine(np.ones(8), y)
        a._CONSTANT_AFFINE_CACHE.clear()
        self.assertEqual(huber, a.affine(np.ones(8), y))
        self.assertNotEqual(squared, huber)

    def test_reset_run_caches_forgets_setting_dependent_results(self):
        a._PARTICLE_SCORE_CACHE["k"] = 1; a._FRAGMENT_FIT_CACHE["k"] = 1
        a._CONSTANT_AFFINE_CACHE["k"] = 1; a.INVALID_DIAGNOSTICS["k"] = 1
        a.reset_run_caches()
        for cache in (a._PARTICLE_SCORE_CACHE, a._FRAGMENT_FIT_CACHE, a._CONSTANT_AFFINE_CACHE, a.INVALID_DIAGNOSTICS):
            self.assertNotIn("k", cache)


class GrammarTests(unittest.TestCase):
    def test_backprop_mutation_stays_inside_the_selected_grammar(self):
        a.rng.seed(0)
        X = np.random.default_rng(0).uniform(1, 2, (40, 2)); y = 3*X[:, 0]*X[:, 1]+2
        ops = ["-", "/", "sqrt", "neg"]
        a.set_backprop_context(X, y, ())
        try:
            for _ in range(100):
                child = a.backprop_mutate(("-", ("x", 0), ("x", 1)), ops, 15, 5)
                used = {node[0] for node in a.walk_tree(child)}-{"x", "c"}
                self.assertLessEqual(used, set(ops), child)
        finally:
            a.set_backprop_context()


class ParticleTests(unittest.TestCase):
    def test_resampling_never_indexes_past_the_last_particle(self):
        population = a.PosteriorParticlePopulation(capacity=4)
        population.particles = [a.Model([("x", 0)], [(1., 0.)]) for _ in range(4)]
        # Weights whose cumulative sum stops just short of 1.
        population.weights = np.array([.25, .25, .25, .25-1e-12])
        a.rng.random = lambda: 1-1e-16
        try: population._resample()
        finally: del a.rng.random
        self.assertEqual(len(population.particles), 4)

    def test_bank_snapshot_restores_the_generator_settings(self):
        banks = a.PerOutputBayesianBanks(["+", "*"], 2, 1, particles=4)
        bank = banks.banks[0]
        bank.base_exploration, bank.decay, bank.floor, bank.temperature = .4, .9, .2, 2.
        restored = a.bayesian_banks_from_snapshot(a.bayesian_banks_snapshot(banks)).banks[0]
        self.assertEqual((restored.base_exploration, restored.decay, restored.floor, restored.temperature), (.4, .9, .2, 2.))


class CheckpointTests(unittest.TestCase):
    def test_pickle_checkpoints_are_refused_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"legacy.pkl"
            path.write_bytes(pickle.dumps({"format_version": 3}))
            with self.assertRaisesRegex(ValueError, "Refusing legacy pickle"):
                a.load_checkpoint(path)


class ExportTests(unittest.TestCase):
    def test_exported_predictions_are_clamped_like_predict_model(self):
        frame = pd.DataFrame({"x": [1., 2., 3., 5e3], "y": [1., 2., 3., 4.]})
        types = [1, 5]
        X, _, names, outputs, cats, maps = a.encode(frame, types)
        model = a.Model([("x", 0)], [(1e9, 0.)], mdl_operators=("+",), mdl_feature_count=1)
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            a.export_model(model, names, outputs, cats, maps, list(frame.columns), types, frame)
            spec = importlib.util.spec_from_file_location("exported_best_model", Path(directory)/"best_model.py")
            exported = importlib.util.module_from_spec(spec); spec.loader.exec_module(exported)
            predicted = exported.predict_frame(frame)["y"].to_numpy(float)
            self.assertTrue(exported.verify_fixture(directory))
        np.testing.assert_array_equal(predicted, a.predict_model(model, X)[:, 0])
        self.assertEqual(predicted.max(), a.CLIP)


if __name__ == "__main__":
    unittest.main()
