"""AFPO stages (fitness / age / both ladders) and equivalence-collapse checks.

Run with: python -B -m unittest -v test_afpo_stages
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import afpo as a
from test_afpo import advance, island, model

X0, X1, X2 = ("x", 0), ("x", 1), ("x", 2)


class EquivalenceKeyTests(unittest.TestCase):
    def assert_same(self, p, q):
        self.assertEqual(a.equivalence_key(p), a.equivalence_key(q), (p, q))

    def assert_different(self, p, q):
        self.assertNotEqual(a.equivalence_key(p), a.equivalence_key(q), (p, q))

    def test_algebraically_equal_trees_share_a_key(self):
        self.assert_same(("+", X0, X1), ("+", X1, X0))
        self.assert_same(("+", ("+", X0, X1), X2), ("+", X0, ("+", X1, X2)))
        self.assert_same(("+", X0, X0), ("*", ("c", 2.), X0))
        self.assert_same(("*", X0, X0), ("square", X0))
        self.assert_same(("*", ("*", X0, X1), X0), ("*", ("square", X0), X1))
        self.assert_same(("-", X0, X0), ("c", 0.))
        self.assert_same(("neg", ("neg", X0)), X0)
        self.assert_same(("*", ("neg", X0), X1), ("neg", ("*", X0, X1)))
        self.assert_same(("max", X0, X1), ("max", X1, X0))
        self.assert_same(("sin", ("+", X0, X1)), ("sin", ("+", X1, X0)))
        self.assert_same(("+", ("c", .1), ("+", ("c", .2), X0)), ("+", X0, ("c", .3)))

    def test_different_trees_keep_different_keys(self):
        self.assert_different(("-", X0, X1), ("-", X1, X0))
        self.assert_different(("sin", X0), ("cos", X0))
        self.assert_different(("/", X0, X1), ("/", X1, X0))
        self.assert_different(("*", ("c", 2.), X0), ("*", ("c", 2.5), X0))
        # Distribution is deliberately not applied: x*(y+z) is one factor product.
        self.assert_different(("*", X0, ("+", X1, X2)), ("+", ("*", X0, X1), ("*", X0, X2)))

    def test_equal_keys_imply_equal_predictions_on_random_trees(self):
        a.rng.seed(5)
        X = np.random.default_rng(5).uniform(-3, 3, size=(64, 3))
        ops = ["+", "-", "*", "neg", "square", "cube", "max", "sin"]
        groups = {}
        for _ in range(3000):
            tree = a.random_tree(3, ops, 9, 3)
            groups.setdefault(a.equivalence_key(tree), []).append(tree)
        collapsed = [trees for trees in groups.values() if len({repr(t) for t in trees}) > 1]
        self.assertGreater(len(collapsed), 10)          # the generator does produce equivalents
        for trees in collapsed:
            reference = a.evaluate(trees[0], X)
            for tree in trees[1:]:
                np.testing.assert_allclose(a.evaluate(tree, X), reference, rtol=1e-9, atol=1e-9, err_msg=repr((trees[0], tree)))

    def test_equivalent_models_collapse_in_pools_and_archive(self):
        X = np.random.default_rng(1).uniform(1, 2, size=(20, 2)); Y = (X[:, 0] + X[:, 1])[:, None]
        left, right = model(("+", X0, X1), features=2), model(("+", X1, X0), features=2)
        for m in (left, right): a.assess(m, X, Y, True, [None])
        self.assertEqual(len(a.unique_models([left, right])), 1)
        self.assertEqual(len(a.novelty_pool([left, right], X)), 1)
        archive = a.ParetoArchive()
        archive.update([left, right])
        self.assertEqual(len(archive.items), 1)

    def test_switch_off_restores_syntactic_identity(self):
        self.addCleanup(setattr, a, "EQUIVALENCE_COLLAPSE", a.EQUIVALENCE_COLLAPSE)
        a.EQUIVALENCE_COLLAPSE = False
        self.assertNotEqual(a.equivalence_key(("+", X0, X1)), a.equivalence_key(("+", X1, X0)))

    def test_offspring_equivalent_to_population_are_redrawn(self):
        a.rng.seed(3); np.random.seed(3)
        X = np.random.default_rng(0).uniform(1, 3, (30, 2)); Y = (X[:, 0] ** 2 + X[:, 1])[:, None]; cats = [None]
        ops = ["+", "*", "square"]
        state = island(X, cats, ops)
        ev = a.ModelEvaluator(1, {"train": (X, Y)}, True, cats, a.compile_constraints(), ["y0"]); self.addCleanup(ev.close)
        a.EQUIVALENCE_STATS["children_redrawn"] = 0
        for generation in range(4):
            advance(state, generation, X, Y, cats, ops, ev)
        self.assertGreater(a.EQUIVALENCE_STATS["children_redrawn"], 0)
        self.assertEqual(len(state.population), state.population_size)


class StageConfigTests(unittest.TestCase):
    def test_validation_and_defaults(self):
        self.assertEqual(a.stage_config()["count"], 1)
        self.assertEqual(a.stage_config("off", count=5)["count"], 1)
        for bad in ({"mode": "height"}, {"mode": "age", "count": 1}, {"mode": "age", "interval": 0},
                    {"mode": "fitness", "threshold_quantile": 0}, {"mode": "age", "schedule": "cubic"}):
            with self.assertRaises(ValueError):
                a.stage_config(**bad)
        self.assertEqual(a.island_config_stages({"count": 2})["mode"], "off")   # pre-stage checkpoints

    def test_age_limits_follow_schedule_and_top_is_unlimited(self):
        for schedule, expected in (("linear", [10, 20, 30]), ("polynomial", [10, 40, 90]), ("exponential", [10, 20, 40])):
            config = a.stage_config("age", count=4, age_gap=10, schedule=schedule)
            self.assertEqual([a.stage_age_limit(k, config) for k in range(3)], expected)
            self.assertIsNone(a.stage_age_limit(3, config))
        self.assertIsNone(a.stage_age_limit(0, a.stage_config("fitness", count=3)))


class StagePromotionTests(unittest.TestCase):
    def setUp(self):
        a.rng.seed(12); np.random.seed(12)
        self.ops = ["+", "-", "*", "square"]
        self.X = np.linspace(-2, 2, 30)[:, None]; self.Y = self.X ** 2 + self.X; self.cats = [None]
        self.ev = a.ModelEvaluator(1, {"train": (self.X, self.Y)}, True, self.cats, a.compile_constraints(), ["y0"])
        self.addCleanup(self.ev.close)

    def ladder(self, stages, islands=1):
        cells = []
        for index in range(islands * stages):
            cell = island(self.X, self.cats, self.ops, index=index)
            cell.island_index, cell.stage = divmod(index, stages)
            self.ev.assess(cell.population, "train")
            cells.append(cell)
        return cells

    def promote(self, cells, config, generation, islands=1):
        return a.promote_stages(cells, islands, config, generation, X=self.X, n_features=1, ops=self.ops, nodes=15, depth=4,
                                nsga_normalization="intercept", parsimony_quality_tolerance=.01, evaluator=self.ev)

    def test_fitness_promotion_moves_good_models_up_and_keeps_sizes(self):
        cells = self.ladder(2); bottom, top = cells
        champion = model(("+", ("square", X0), X0)); champion.birth_generation = 0
        bottom.population[0] = champion
        config = a.stage_config("fitness", count=2)
        promoted, _, _ = self.promote(cells, config, 5)
        self.assertGreater(promoted, 0)
        self.assertIn(champion.lineage_id, {m.lineage_id for m in top.population})
        self.assertNotIn(champion.lineage_id, {m.lineage_id for m in bottom.population})
        self.assertEqual(len(top.population), top.population_size)
        self.assertEqual(len(bottom.population), bottom.population_size)    # vacated slots refilled with fresh models
        self.assertTrue(any(m.origin == "stage_seed" for m in bottom.population))

    def test_fitness_threshold_is_never_loosened(self):
        cells = self.ladder(2)
        config = a.stage_config("fitness", count=2)
        self.promote(cells, config, 5); first = config["thresholds"]["0:1"]
        for m in cells[1].population: m.objectives = (1e9, *m.objectives[1:])    # stage above gets worse
        self.promote(cells, config, 10)
        self.assertLessEqual(config["thresholds"]["0:1"], first)

    def test_age_mode_evicts_old_models_and_reseeds_stage_zero(self):
        cells = self.ladder(3); bottom, middle, _ = cells
        config = a.stage_config("age", count=3, age_gap=4)
        for m in middle.population: m.age = 100     # far past the stage-1 limit (4 * 2**2 = 16)
        before = {m.lineage_id for m in middle.population}
        _, _, reseeded = self.promote(cells, config, 8)
        self.assertEqual(reseeded, 1)
        self.assertTrue(all(m.origin == "stage_seed" and m.birth_generation == 8 for m in bottom.population))
        self.assertEqual(len(bottom.population), bottom.population_size)
        self.assertGreaterEqual(len(middle.population), 2)                 # never emptied
        self.assertLessEqual(len({m.lineage_id for m in middle.population} & before), 2)

    def test_off_mode_changes_nothing(self):
        cells = self.ladder(1)
        before = [m.lineage_id for m in cells[0].population]
        self.assertEqual(self.promote(cells, a.stage_config(), 5), (0, 0, 0))
        self.assertEqual([m.lineage_id for m in cells[0].population], before)

    def test_migration_stays_on_its_stage_level(self):
        cells = self.ladder(2, islands=2)
        config = {"count": 2, "migration_interval": 1, "migrants_per_island": 2, "migration_events": 0,
                  "stages": a.stage_config("fitness", count=2, interval=1000)}
        top_ids = {m.lineage_id for m in cells[1].population}
        with contextlib.redirect_stdout(io.StringIO()):
            a.advance_topology(cells, config, 1, X=self.X, n_features=1, ops=self.ops, nodes=15, depth=4,
                               nsga_normalization="intercept", parsimony_quality_tolerance=.01, evaluator=self.ev)
        stage0 = {m.lineage_id for cell in (cells[0], cells[2]) for m in cell.population}
        self.assertFalse(top_ids & stage0)
        self.assertEqual(config["migration_events"], 1)

    def test_cell_snapshot_round_trip_keeps_grid_position(self):
        cells = self.ladder(2, islands=2)
        restored = [a.island_from_snapshot(a.island_snapshot(cell), len(self.X), .01) for cell in cells]
        self.assertEqual([(c.island_index, c.stage) for c in restored], [(0, 0), (0, 1), (1, 0), (1, 1)])
        legacy = a.island_snapshot(cells[3]); legacy.pop("island"); legacy.pop("stage")
        self.assertEqual((a.island_from_snapshot(legacy, len(self.X), .01).stage), 0)


class RoleTests(unittest.TestCase):
    def setUp(self):
        a.rng.seed(7); np.random.seed(7)
        self.ops = ["+", "-", "*", "square"]
        self.X = np.linspace(-2, 2, 40)[:, None]
        self.Y = np.where(self.X > 0, self.X ** 2, -self.X)          # two regimes
        self.cats = [None]
        self.ev = a.ModelEvaluator(1, {"train": (self.X, self.Y)}, True, self.cats, a.compile_constraints(), ["y0"])
        self.addCleanup(self.ev.close)

    def cells(self, islands=3, stages=1):
        cells = []
        for index in range(islands * stages):
            cell = island(self.X, self.cats, self.ops, index=index)
            cell.island_index, cell.stage = divmod(index, stages)
            cells.append(cell)
        return cells

    def config(self, islands=3, stages=1, **roles):
        return {"count": islands, "stages": a.stage_config("fitness", count=stages) if stages > 1 else a.stage_config(),
                "roles": a.role_config(True, **roles)}

    def set_best(self, cell, tree):
        m = model(tree); a.assess(m, self.X, self.Y, True, self.cats); cell.best_models.model = m

    def update(self, cells, config):
        return a.update_roles(cells, config, 10, X=self.X, Y=self.Y, cats=self.cats, crossover_rate=.35,
                              bayesian_proposal_rate=.25, nodes=15)

    def test_parameters_spread_and_generalist_keeps_defaults(self):
        cells = self.cells(4)
        a.assign_role_parameters(cells, 4, .35, .25, 20)
        self.assertEqual(cells[0].role, {})
        nodes = [cell.role["params"]["nodes"] for cell in cells[1:]]
        self.assertEqual(nodes, sorted(nodes)); self.assertEqual((nodes[0], nodes[-1]), (10, 20))
        self.assertEqual(a.cell_search_settings(cells[0], .35, .25, 20), (.35, .25, 20, None))

    def test_specialists_weight_the_rows_they_win(self):
        cells = self.cells(3)
        self.set_best(cells[0], ("square", X0))                 # right regime only
        self.set_best(cells[1], ("-", ("c", 0.), X0))            # left regime only
        self.set_best(cells[2], ("square", X0))
        config = self.config()
        self.update(cells, config)
        self.assertNotIn("case_weights", cells[0].role)
        weights = cells[1].role["case_weights"]
        self.assertAlmostEqual(float(np.mean(weights)), 1., places=9)
        left = self.X[:, 0] < 0
        self.assertGreater(weights[left].mean(), weights[~left].mean())
        self.assertGreater(cells[1].role["contribution"], 0)
        self.assertEqual(config["roles"]["updates"], 1)

    def test_non_contributing_specialist_is_retired(self):
        cells = self.cells(2)
        for cell in cells: self.set_best(cell, ("square", X0))  # identical: nothing unique to offer
        config = self.config(2, retire_after=2)
        for _ in range(2): retired, _ = self.update(cells, config)
        self.assertEqual(retired, 1)
        self.assertEqual(cells[1].role["stale"], 0)
        self.assertEqual(config["roles"]["retirements"], 1)

    def test_weighted_lexicase_prefers_the_heavily_weighted_row(self):
        X = np.array([[0.], [1.]]); Y = np.array([[0.], [0.]])
        on_first = model(("c", 0.)); on_second = model(("-", X0, ("c", 1.)))  # exact on row 0 / row 1 only
        weights = np.array([1e6, 1e-6])
        picks = a.lexicase_parents([on_first, on_second], 200, X, Y, [None], case_weights=weights)
        self.assertGreater(sum(p is on_first for p in picks), 190)

    def test_weighted_generation_keeps_population_size(self):
        cells = self.cells(2); state = cells[1]
        advance(state, 0, self.X, self.Y, self.cats, self.ops, self.ev, case_weights=np.linspace(.1, 2, len(self.X)))
        self.assertEqual(len(state.population), state.population_size)

    def test_fragments_migrate_on_probation(self):
        cells = self.cells(2)
        cells[0].library.items = {repr(("square", X0)): {"tree": ("square", X0), "support": 9, "contribution": 2., "uses": 3,
                                                         "rejections": 0, "source": "observed", "family": "power"},
                                  repr(("adf_1", X0)): {"tree": ("adf_1", X0), "support": 20, "contribution": 5., "uses": 0,
                                                        "rejections": 0, "source": "observed", "family": "adf"}}
        cells[1].library.items = {}
        moved = a.migrate_fragments(cells, 4, generation=3)
        item = cells[1].library.items.get(repr(("square", X0)))
        self.assertIsNotNone(item)
        self.assertEqual((item["support"], item["contribution"], item["source"]), (1, 0., "island_migrant"))
        self.assertEqual(item["provenance"]["island"], 0)
        self.assertNotIn(repr(("adf_1", X0)), cells[1].library.items)     # receiver lacks that ADF
        self.assertEqual(moved, 1)

    def test_roles_require_two_islands(self):
        df = pd.DataFrame({"a": np.arange(10.), "y": np.arange(10.)})
        args = a.parse_cli(["--population", "32", "--max-generations", "1", "--seed", "1"])[1]
        setup = {"path": Path("unused.csv"), "df": df, "types": [1, 5], "delimiter": ",", "ops": ["+"], "affine_on": True,
                 "coev": False, "dynamic_pressure_on": False, "adf_enabled": False, "nodes": 7, "depth": 3, "island_count": 1,
                 "migration_interval": 0, "migrants_per_island": 0, "val_path": "0", "validation_percent": None, "metadata": {},
                 "roles": {"enabled": True}}
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "at least two islands"):
            a.train_from_setup(args, setup)


class StagedRunTests(unittest.TestCase):
    def test_staged_islanded_run_checkpoints_and_resumes(self):
        rng = np.random.default_rng(0); x = rng.uniform(1, 3, (40, 2))
        df = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": x[:, 0] ** 2 + 2 * x[:, 1]})
        with tempfile.TemporaryDirectory(prefix="afpo-stages-") as directory, contextlib.chdir(directory):
            df.to_csv("d.csv", index=False)
            args = a.parse_cli(["--population", "64", "--max-generations", "4", "--seed", "4", "--workers", "1"])[1]
            setup = {"path": Path("d.csv"), "df": df, "types": [1, 1, 5], "delimiter": ",", "ops": ["+", "-", "*", "square"],
                     "affine_on": True, "coev": False, "dynamic_pressure_on": True, "adf_enabled": False, "nodes": 15, "depth": 4,
                     "island_count": 2, "migration_interval": 2, "migrants_per_island": 2, "val_path": "", "validation_percent": 20,
                     "metadata": {}, "stages": {"mode": "both", "count": 2, "interval": 2, "age_gap": 2},
                     "roles": {"enabled": True, "interval": 2}}
            with contextlib.redirect_stdout(io.StringIO()) as log:
                result = a.train_from_setup(args, setup, choose_model=lambda *_: 0)
            self.assertIn("Stage promotion", log.getvalue())
            _, _, _, _, state = a.load_checkpoint(result["checkpoint"], False)
            self.assertEqual(len(state["island_states"]), 4)
            self.assertEqual(state["island_config"]["stages"]["mode"], "both")
            self.assertGreater(state["island_config"]["stages"]["promotion_events"], 0)
            self.assertTrue(state["equivalence_collapse"])
            self.assertGreater(state["island_config"]["roles"]["updates"], 0)
            self.assertIn("case_weights", next(c for c in state["island_states"] if c["island"] == 1)["role"])
            resume = a.parse_cli(["--resume", result["checkpoint"], "--max-generations", "6", "--workers", "1"])[1]
            with contextlib.redirect_stdout(io.StringIO()) as log:
                a.resume_main(resume)
            self.assertIn("Resume complete at generation 6", log.getvalue())
            self.assertIn("stages per island", log.getvalue())
            self.assertIn("Island roles", log.getvalue())

    def test_too_many_cells_for_population_is_rejected(self):
        df = pd.DataFrame({"a": np.arange(10.), "y": np.arange(10.)})
        args = a.parse_cli(["--population", "32", "--max-generations", "1", "--seed", "1"])[1]
        setup = {"path": Path("unused.csv"), "df": df, "types": [1, 5], "delimiter": ",", "ops": ["+"], "affine_on": True,
                 "coev": False, "dynamic_pressure_on": False, "adf_enabled": False, "nodes": 7, "depth": 3, "island_count": 2,
                 "migration_interval": 5, "migrants_per_island": 1, "val_path": "0", "validation_percent": None, "metadata": {},
                 "stages": {"mode": "age", "count": 3}}
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "eight models each"):
            a.train_from_setup(args, setup)


if __name__ == "__main__":
    unittest.main()
