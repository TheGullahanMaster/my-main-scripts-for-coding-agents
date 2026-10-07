"""Two-stage search: --stop-at-loss-action compress, --stop-at-loss-fraction.

Run with: python -B -m unittest -v test_afpo_compression
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import afpo as a
from test_afpo import advance, island, model

X0 = ("x", 0)


def scored(tree, loss, bits, ops=("+", "-", "*", "square", "sin")):
    m = model(tree, ops=tuple(ops)); m.objectives = (loss, 0., float(bits), 0); return m


def holder(loss, bits=40., tree=("square", X0), index=0):
    """A stand-in island: best model of the given training loss, a population holding it."""
    best = scored(tree, loss, bits)
    return SimpleNamespace(best_models=SimpleNamespace(model=best), population=[best], compress=None, island_index=index, stage=0)


def args(**overrides):
    values = dict(max_time=0., stop_at_loss=1e-3, stop_at_loss_fraction=0., stop_at_loss_action="stop", compress_band=.05, compress_patience=0)
    values.update(overrides); return SimpleNamespace(**values)


class StopFractionTests(unittest.TestCase):
    def test_required_island_count(self):
        self.assertEqual(a.loss_target_count(args(), 4), 1)                       # 0 = any one island
        self.assertEqual(a.loss_target_count(args(stop_at_loss_fraction=1.), 4), 4)
        self.assertEqual(a.loss_target_count(args(stop_at_loss_fraction=.5), 4), 2)
        self.assertEqual(a.loss_target_count(args(stop_at_loss_fraction=.3), 4), 2)   # rounded up
        self.assertEqual(a.loss_target_count(args(stop_at_loss_fraction=1.), 1), 1)
        self.assertEqual(a.loss_target_count(SimpleNamespace(), 3), 1)              # older callers: any island

    def test_fraction_decides_when_the_search_stops(self):
        islands = [holder(1e-4), holder(1.), holder(1e-4), holder(1.)]
        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertTrue(a.stop_rule_reached(args(), 0., islands))
            self.assertTrue(a.stop_rule_reached(args(stop_at_loss_fraction=.5), 0., islands))
            self.assertFalse(a.stop_rule_reached(args(stop_at_loss_fraction=.75), 0., islands))
            self.assertFalse(a.stop_rule_reached(args(stop_at_loss_fraction=1.), 0., islands))
        self.assertIn("2 of 4 islands", log.getvalue())

    def test_cli_validation(self):
        self.assertEqual(a.parse_cli(["--stop-at-loss", "0.1", "--stop-at-loss-action", "compress"])[1].stop_at_loss_action, "compress")
        for bad in (["--stop-at-loss-fraction", "1.5"], ["--compress-band", "-1"], ["--compress-patience", "-2"],
                    ["--stop-at-loss-action", "compress"]):
            with self.subTest(bad=bad), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                a.parse_cli(bad)


class CompressionStageTests(unittest.TestCase):
    def test_reaching_the_target_anchors_instead_of_stopping(self):
        islands = [holder(1e-4, 80., ("+", ("square", X0), ("sin", X0))), holder(1.)]
        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertFalse(a.stop_rule_reached(args(stop_at_loss_action="compress"), 0., islands, generation=7))
        first, second = islands
        self.assertIsNotNone(first.compress); self.assertIsNone(second.compress)       # the other keeps searching for loss
        self.assertEqual(first.compress["anchor_loss"], 1e-4)
        self.assertEqual(first.compress["ceiling"], 1e-3)                              # --compress-ceiling target: up to the target
        self.assertEqual((first.compress["cap"], first.compress["since"], first.compress["shortest"]), (5, 7, 80.))
        self.assertIn("Compression stage started", log.getvalue())
        # The second island joins once its own best model meets the target.
        second.best_models.model = second.population[0] = scored(X0, 5e-4, 10)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(a.stop_rule_reached(args(stop_at_loss_action="compress"), 0., islands, generation=9))
        self.assertEqual(second.compress["anchor_loss"], 5e-4)
        self.assertEqual(first.compress["anchor_loss"], 1e-4)                          # anchors never move

    def test_anchor_ceiling_and_per_output_targets(self):
        cell = holder(1e-4)
        with contextlib.redirect_stdout(io.StringIO()):
            a.stop_rule_reached(args(stop_at_loss_action="compress", compress_ceiling="anchor"), 0., [cell])
        self.assertAlmostEqual(cell.compress["ceiling"], 1e-4 + max(.05e-4, a.LOSS_NOISE_FLOOR))
        self.assertIsNone(cell.compress["output_ceilings"])
        self.assertIn("training loss <= 0.000105", a.describe_compression_ceiling(cell.compress))
        two = a.Model([("square", X0), X0], [(1., 0.), (1., 0.)], mdl_operators=("square",), mdl_feature_count=1)
        two.objectives = (1e-4, 0., .5, 0., 30., 0)                                    # y=1e-4, z=0.5
        cell = SimpleNamespace(best_models=SimpleNamespace(model=two), population=[two], compress=None, island_index=0, stage=0)
        context = {"X": None, "Y": None, "cats": [None, None], "constraints": None, "out_names": ["y", "z"]}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(a.stop_rule_reached(args(stop_at_loss="y=1e-3", stop_at_loss_action="compress"), 0., [cell], context))
        self.assertIsNone(cell.compress["ceiling"])
        self.assertEqual(cell.compress["output_ceilings"], [1e-3, .5 * 1.05])           # z has no target: the anchor's band
        worse_z = a.Model([("square", X0), X0], [(1., 0.), (1., 0.)], mdl_operators=("square",), mdl_feature_count=1)
        worse_z.objectives = (5e-4, 0., .6, 0., 20., 0)
        self.assertEqual(a.compression_lane([two, worse_z], cell.compress), [two])
        worse_z.objectives = (5e-4, 0., .52, 0., 20., 0)
        self.assertEqual(a.compression_lane([two, worse_z], cell.compress), [worse_z, two])

    def test_fraction_delays_the_compression_stage(self):
        islands = [holder(1e-4), holder(1.)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(a.stop_rule_reached(args(stop_at_loss_action="compress", stop_at_loss_fraction=1.), 0., islands))
        self.assertTrue(all(cell.compress is None for cell in islands))

    def test_patience_ends_a_stalled_compression(self):
        cell = holder(1e-4, 80.)
        rule = args(stop_at_loss_action="compress", compress_patience=3)
        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertFalse(a.stop_rule_reached(rule, 0., [cell], generation=0))
            shorter = scored(X0, 1.02e-4, 30); cell.population.append(shorter)          # in band, shorter: progress
            self.assertFalse(a.stop_rule_reached(rule, 0., [cell], generation=2))
            self.assertEqual((cell.compress["shortest"], cell.compress["improved"]), (30., 2))
            cell.population.append(scored(("c", 1.), 1., 5))                            # shorter but out of band: no progress
            self.assertFalse(a.stop_rule_reached(rule, 0., [cell], generation=4))
            self.assertTrue(a.stop_rule_reached(rule, 0., [cell], generation=5))
        self.assertIn("shortest in-band model 30.0 MDL bits", log.getvalue())
        self.assertIn("--compress-patience", log.getvalue())

    def test_lane_keeps_shortest_in_band_models_no_larger_than_the_anchor(self):
        anchor = scored(("+", ("square", X0), ("sin", ("*", X0, ("c", 2.)))), 1., 120)    # 7 nodes
        close = [scored(("+", ("square", X0), ("c", float(i))), 1.04, 60 + i) for i in range(6)]
        bigger = scored(("+", ("+", ("square", X0), ("sin", X0)), ("*", X0, ("c", 3.))), .9, 200)
        junk = [scored(("c", float(i)), 5. + i, 5) for i in range(10)]
        cell = SimpleNamespace(best_models=SimpleNamespace(model=anchor))
        self.assertTrue(a.anchor_compression(cell, .05, 0))
        pool = [anchor, *close, bigger, *junk]
        lane = a.compression_lane(pool, cell.compress)
        self.assertEqual(lane, [*close, anchor])                                        # bigger is too large, junk out of band
        survivors = a.anchored_survivors(pool, 8, "intercept", .01, a.COMPRESS_SURVIVOR_SHARE, lane=lane)
        self.assertEqual(survivors[:6], close)
        self.assertEqual(len(survivors), 8)

    def test_compressing_cell_settings(self):
        cell = holder(1e-4)
        cell.role = {"kind": "explorer", "params": {"novelty": .25, "crossover_rate": .5, "bayesian_proposal_rate": .4}}
        self.assertEqual(a.cell_search_settings(cell, .35, .25, 15)[:2], (.5, .4))
        with contextlib.redirect_stdout(io.StringIO()):
            a.stop_rule_reached(args(stop_at_loss_action="compress"), 0., [cell])
        settings = a.cell_role_settings(cell)
        self.assertNotIn("novelty", settings)
        self.assertTrue(settings["neutral_shrink"]); self.assertEqual(settings["compress"], cell.compress)
        self.assertEqual(settings["mutation_bias"], a.ROLE_MUTATION_BIAS["simplifier"])
        self.assertEqual(a.cell_search_settings(cell, .35, .25, 15)[:2], (.25, .2))

    def test_menu_offers_the_shortest_model_in_the_compression_band(self):
        def entry(tree, loss, bits):
            m = scored(tree, loss, bits); return (m, m, {"loss": loss, "shape": 0., "mdl_bits": float(bits)})
        entries = [entry(("+", ("square", X0), ("sin", X0)), 1., 120), entry(("square", X0), 1.03, 60), entry(X0, 2., 10)]
        evaluation = ("training", entries)
        label, chosen = a.compression_choice(evaluation, None, .05)                    # anchor: within 5% of the best
        self.assertIs(chosen, entries[1][0]); self.assertIn("within 5%", label)
        label, chosen = a.compression_choice(evaluation, 2.5, .05, ["y"])              # target: anything meeting it
        self.assertIs(chosen, entries[2][0]); self.assertIn("meeting --stop-at-loss", label)
        self.assertIsNone(a.compression_choice(evaluation, .5, .05, ["y"]))
        self.assertIs(a.compression_choice(evaluation, {"y": 1.05}, .05, ["y"])[1], entries[1][0])
        labels, choices, _ = a.model_options([e[0] for e in entries], cats=[None], evaluation=evaluation,
                                             compression=a.compression_choice(evaluation, None, .05))
        self.assertIn("compression stage", labels[1])
        self.assertIs(choices[1], entries[1][0])
        labels, _, _ = a.model_options([e[0] for e in entries], cats=[None], evaluation=evaluation)
        self.assertFalse(any("compression stage" in label for label in labels))


class CompressionGenerationTests(unittest.TestCase):
    OPS = ["+", "-", "*", "square", "sin"]

    def setUp(self):
        saved = {name: value for name, value in vars(a).items()
                 if name.isupper() and isinstance(value, (bool, int, float, str, tuple, frozenset, type(None)))}
        self.addCleanup(lambda: vars(a).update(saved))
        r = np.random.default_rng(3)
        self.X = r.uniform(-2, 2, (60, 1)); self.Y = self.X ** 2 + .5 * self.X
        self.cats = [None]
        self.ev = a.ModelEvaluator(1, {"train": (self.X, self.Y)}, True, self.cats, a.compile_constraints(), ["y0"])
        self.addCleanup(self.ev.close)

    def test_compressing_generation_fills_its_lane(self):
        cell = island(self.X, self.cats, self.OPS)
        for generation in range(3): advance(cell, generation, self.X, self.Y, self.cats, self.OPS, self.ev)
        self.assertTrue(a.anchor_compression(cell, .05, 3, target=.2))       # x^2 alone (loss ~0.13) is good enough
        for generation in range(3, 8):
            advance(cell, generation, self.X, self.Y, self.cats, self.OPS, self.ev, role_settings=a.cell_role_settings(cell))
        self.assertEqual(len(cell.population), cell.population_size)
        lane = a.compression_lane(cell.population, cell.compress)
        self.assertGreaterEqual(len(lane), 3)
        self.assertLessEqual(a.model_complexity(lane[0]), cell.compress["anchor_bits"])
        self.assertTrue(all(a.aggregate_loss(m) <= cell.compress["ceiling"] for m in lane))


class CompressionRunTests(unittest.TestCase):
    def setUp(self):
        saved = {name: value for name, value in vars(a).items()
                 if name.isupper() and isinstance(value, (bool, int, float, str, tuple, frozenset, type(None)))}
        self.addCleanup(lambda: vars(a).update(saved))

    def test_two_stage_run_compresses_checkpoints_and_resumes(self):
        x = np.random.default_rng(0).uniform(1, 3, (60, 2))
        df = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": x[:, 0] ** 2 + 2 * x[:, 1]})
        flags = ["--workers", "1", "--seed", "4", "--stop-at-loss", "10", "--stop-at-loss-action", "compress"]
        with tempfile.TemporaryDirectory(prefix="afpo-compress-") as directory, contextlib.chdir(directory):
            df.to_csv("d.csv", index=False)
            run_args = a.parse_cli(["--population", "48", "--max-generations", "6", *flags])[1]
            setup = {"path": Path("d.csv"), "df": df, "types": [1, 1, 5], "delimiter": ",", "ops": ["+", "-", "*", "square"],
                     "affine_on": True, "coev": False, "dynamic_pressure_on": False, "adf_enabled": False, "nodes": 15, "depth": 4,
                     "island_count": 2, "migration_interval": 2, "migrants_per_island": 2, "val_path": "", "validation_percent": 20,
                     "metadata": {}}
            with contextlib.redirect_stdout(io.StringIO()) as log:
                result = a.train_from_setup(run_args, setup, choose_model=lambda *_: 0)
            output = log.getvalue()
            self.assertEqual(result["generation"], 6)                       # the loss target did not end the search
            self.assertIn("Compression stage started", output)
            self.assertIn("Compression stage: 2 of 2 cell(s) compressed", output)
            _, _, _, _, state = a.load_checkpoint(result["checkpoint"], False)
            compress = [cell.get("compress") for cell in state["island_states"]]
            self.assertTrue(all(compress))
            self.assertTrue(all(c["shortest"] <= c["anchor_bits"] for c in compress))
            with contextlib.redirect_stdout(io.StringIO()):
                a.resume_main(a.parse_cli(["--resume", result["checkpoint"], "--max-generations", "8", *flags])[1])
            _, _, _, _, resumed = a.load_checkpoint(result["checkpoint"], False)
            self.assertEqual([cell["compress"]["anchor_loss"] for cell in resumed["island_states"]], [c["anchor_loss"] for c in compress])


if __name__ == "__main__":
    unittest.main()
