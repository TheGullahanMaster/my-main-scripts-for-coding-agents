"""AFPO stages (fitness / age / both ladders) and equivalence-collapse checks.

Run with: python -B -m unittest -v test_afpo_stages
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
        self.assertEqual(cells[0].role, {"kind": "generalist"})
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
        item = cells[1].library.items.get(a.fragment_key(("square", X0)))
        self.assertIsNotNone(item)
        self.assertEqual((item["support"], item["contribution"], item["source"]), (1, 0., "island_migrant"))
        self.assertEqual(item["provenance"]["island"], 0)
        self.assertNotIn(a.fragment_key(("adf_1", X0)), cells[1].library.items)   # receiver lacks that ADF
        self.assertEqual(item["probation"], a.FRAGMENT_MIGRANT_PROBATION)
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


class PresetRoleTests(unittest.TestCase):
    """Fixed, user-chosen island roles beside the self-organising "auto" ones."""
    OPS = ["+", "-", "*", "square", "sin", "exp"]

    def setUp(self):
        # train_from_setup sets module settings (the loss noise floor, ...);
        # restore them so later tests see the import-time values.
        saved = {name: value for name, value in vars(a).items()
                 if name.isupper() and isinstance(value, (bool, int, float, str, tuple, frozenset, type(None)))}
        self.addCleanup(lambda: vars(a).update(saved))
        r = np.random.default_rng(3)
        self.X = r.uniform(-2, 2, (60, 1)); self.Y = np.where(self.X > 0, self.X ** 2, -self.X)
        self.cats = [None]
        self.ev = a.ModelEvaluator(1, {"train": (self.X, self.Y)}, True, self.cats, a.compile_constraints(), ["y0"])
        self.addCleanup(self.ev.close)

    def cells(self, islands):
        cells = []
        for index in range(islands):
            cell = island(self.X, self.cats, self.OPS, index=index); cell.island_index, cell.stage = index, 0
            cells.append(cell)
        return cells

    def test_role_choices_follow_the_selected_operator_groups(self):
        choices = a.island_role_choices(self.OPS)
        for role in ("auto", "generalist", "simplifier", "explorer", "refiner", "family:1", "family:2", "family:3", "family:4"):
            self.assertIn(role, choices)
        self.assertNotIn("family:7", choices)                         # no comparison operators selected
        self.assertEqual(a.family_role_operators("family:4", self.OPS), ["+", "-", "*", "sin"])

    def test_validation(self):
        self.assertEqual(a.validate_island_roles([], 4, self.OPS), ["auto"] * 3)
        self.assertEqual(a.validate_island_roles(["simplifier", " auto", "family:3"], 4, self.OPS), ["simplifier", "auto", "family:3"])
        with self.assertRaisesRegex(ValueError, "one entry per island"):
            a.validate_island_roles(["auto"], 4, self.OPS)
        with self.assertRaisesRegex(ValueError, "Unknown island role"):
            a.validate_island_roles(["auto", "wizard", "auto"], 4, self.OPS)
        with self.assertRaisesRegex(ValueError, "Unknown island role"):
            a.validate_island_roles(["family:7", "auto", "auto"], 4, self.OPS)

    def test_assignment_spreads_auto_islands_and_applies_presets(self):
        cells = self.cells(5)
        a.assign_role_parameters(cells, 5, .35, .25, 20, ["auto", "simplifier", "auto", "family:3"], self.OPS)
        self.assertEqual(cells[0].role, {"kind": "generalist"})
        self.assertEqual([a.role_kind(c) for c in cells], ["generalist", "auto", "simplifier", "auto", "family:3"])
        self.assertEqual((cells[1].role["params"]["t"], cells[3].role["params"]["t"]), (0., 1.))   # spread over auto islands only
        simplifier = cells[2].role["params"]
        self.assertNotIn("nodes", simplifier); self.assertTrue(simplifier["gather_migrants"] and simplifier["anchored"])
        self.assertTrue(a.cell_role_settings(cells[2])["anchored"])
        self.assertEqual(a.cell_role_settings(cells[4]), {"ops": ["+", "-", "*", "exp"]})
        self.assertIsNone(a.cell_role_settings(cells[1])); self.assertIsNone(a.cell_role_settings(cells[0]))

    def test_role_updates_leave_fixed_roles_alone(self):
        cells = self.cells(3)
        a.assign_role_parameters(cells, 3, .35, .25, 15, ["auto", "refiner"], self.OPS)
        for cell in cells:
            m = model(("square", X0)); a.assess(m, self.X, self.Y, True, self.cats); cell.best_models.model = m
        config = {"count": 3, "stages": a.stage_config(), "roles": a.role_config(True, retire_after=1, assignments=["auto", "refiner"])}
        before = dict(cells[2].role["params"])
        for _ in range(3):
            a.update_roles(cells, config, 10, X=self.X, Y=self.Y, cats=self.cats, crossover_rate=.35, bayesian_proposal_rate=.25, nodes=15)
        self.assertIn("case_weights", cells[1].role)
        self.assertNotIn("case_weights", cells[2].role); self.assertNotIn("stale", cells[2].role)
        self.assertEqual(cells[2].role["params"], before)
        self.assertIn("contribution", cells[2].role)
        self.assertEqual(config["roles"]["retirements"], 3)          # only the auto island is ever retired

    def test_simplifier_gathers_every_islands_elites(self):
        cells = self.cells(4)
        a.assign_role_parameters(cells, 4, .35, .25, 15, ["auto", "simplifier", "auto"], self.OPS)
        for cell in cells: advance(cell, 0, self.X, self.Y, self.cats, self.OPS, self.ev)
        a.migrate_islands(cells, 1, X=self.X, nsga_normalization="intercept", parsimony_quality_tolerance=.01, evaluator=self.ev)
        arrived = [m for m in cells[2].population if m.origin == "island_migrant"]
        ring = [m for m in cells[3].population if m.origin == "island_migrant"]
        self.assertLessEqual(len(ring), 1)
        self.assertGreaterEqual(len(arrived), 1)
        self.assertEqual(len(cells[2].population), cells[2].population_size)
        self.assertEqual(len({id(m) for cell in cells for m in cell.population}), sum(len(cell.population) for cell in cells))

    def test_prune_always_shrinks_and_neutral_shrink_is_accepted(self):
        a.rng.seed(1)
        tree = ("+", ("sin", ("*", X0, ("c", 2.))), ("square", ("+", X0, ("c", 1.))))
        for _ in range(50):
            self.assertLess(a.node_size(a.prune_mutate(tree)), a.node_size(tree))
        dead = ("abs", ("square", X0))                                # abs of a square changes nothing
        portfolio = a.MutationPortfolio(); portfolio.bias = {kind: 0. for kind in portfolio.weights}; portfolio.bias["shrink"] = 1.
        child, kind, _ = a.semantic_mutate(dead, self.X, portfolio, 1, self.OPS, 15, 4)
        self.assertEqual((child, kind), (dead, None))                # the default guard rejects an unchanged output
        child, kind, _ = a.semantic_mutate(dead, self.X, portfolio, 1, self.OPS, 15, 4, neutral_shrink=True)
        self.assertEqual((child, kind), (("square", X0), "shrink"))

    def test_portfolio_bias_steers_and_records_role_only_kinds(self):
        portfolio = a.MutationPortfolio(); portfolio.bias = {kind: 0. for kind in portfolio.weights}; portfolio.bias["prune"] = 1.
        self.assertEqual({portfolio.choose() for _ in range(20)}, {"prune"})
        portfolio.record("prune", True)
        self.assertEqual((portfolio.tries["prune"], portfolio.wins["prune"]), (1, 1))

    def test_family_island_builds_from_its_operators_but_is_priced_on_the_full_grammar(self):
        state = self.cells(2)[1]
        calls = []
        original = a.random_tree
        def recording(n_features, ops, *args, **kwargs):
            calls.append(tuple(ops)); return original(n_features, ops, *args, **kwargs)
        a.random_tree = recording
        try:
            # Generation 1: every 5th generation the Bayesian bank also
            # rejuvenates its particle catalogue, which keeps the run's grammar.
            advance(state, 1, self.X, self.Y, self.cats, self.OPS, self.ev, role_settings={"ops": ["+", "-", "*", "exp"]})
        finally:
            a.random_tree = original
        self.assertTrue(calls)
        self.assertTrue(all(set(ops) <= {"+", "-", "*", "exp"} for ops in calls))
        for m in state.population:
            self.assertTrue({"+", "-", "*", "exp"} <= set(m.mdl_operators), m.mdl_operators)
            used = {node[0] for tree in m.trees for node in a.walk_tree(tree) if node[0] not in ("x", "c")}
            self.assertTrue(used <= set(m.mdl_operators))

    def test_role_settings_change_selection_and_novelty(self):
        state = self.cells(2)[1]
        advance(state, 0, self.X, self.Y, self.cats, self.OPS, self.ev,
                role_settings={"novelty": 1., "parsimony": .05, "semantic_max_delta": float("inf"), "mutation_bias": {"prune": 2.}})
        self.assertEqual(len(state.population), state.population_size)
        self.assertEqual(state.portfolio.bias, {"prune": 2.})
        advance(state, 1, self.X, self.Y, self.cats, self.OPS, self.ev)
        self.assertIsNone(state.portfolio.bias)                       # a generation without a role clears it

    def test_preset_roles_run_parallel_and_resume_exactly(self):
        x = np.random.default_rng(0).uniform(1, 3, (40, 2))
        df = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": x[:, 0] ** 2 + 2 * x[:, 1]})
        roles = {"enabled": True, "interval": 2, "assignments": ["simplifier", "family:2", "explorer"]}
        def run(plan, cell_workers):
            with tempfile.TemporaryDirectory(prefix="afpo-roles-") as directory, contextlib.chdir(directory):
                df.to_csv("d.csv", index=False)
                flags = ["--workers", "1", "--cell-workers", str(cell_workers)]
                args = a.parse_cli(["--population", "64", "--max-generations", str(plan[0]), "--seed", "4", *flags])[1]
                setup = {"path": Path("d.csv"), "df": df, "types": [1, 1, 5], "delimiter": ",", "ops": ["+", "-", "*", "square"],
                         "affine_on": True, "coev": False, "dynamic_pressure_on": True, "adf_enabled": False, "nodes": 15, "depth": 4,
                         "island_count": 4, "migration_interval": 2, "migrants_per_island": 2, "val_path": "", "validation_percent": 20,
                         "metadata": {}, "roles": dict(roles)}
                with contextlib.redirect_stdout(io.StringIO()) as log:
                    checkpoint = a.train_from_setup(args, setup, choose_model=lambda *_: 0)["checkpoint"]
                    for target in plan[1:]:
                        a.resume_main(a.parse_cli(["--resume", checkpoint, "--max-generations", str(target), *flags])[1])
                _, _, _, _, state = a.load_checkpoint(checkpoint, False)
            # Lineage ids are bookkeeping that already differs between serial and
            # parallel cells, so crossover partner ids are left out of the comparison.
            histories = lambda m: [{k: v for k, v in r.items() if k != "partner"} for r in m["history"]]
            return [[(repr(m["trees"]), histories(m)) for m in cell["population"]] for cell in state["island_states"]], state, log.getvalue()
        serial, state, log = run([6], 1)
        parallel, _, _ = run([4, 6], 4)
        self.assertEqual(serial, parallel)                                   # trees and histories alike
        records = [r for cell in state["island_states"] for m in cell["population"] for r in m["history"]]
        self.assertTrue(any(r["event"] == "migrated" and r.get("how") == "gathered" for r in records))
        self.assertTrue(all(m["history"] and m["history"][0]["event"] == "born" for cell in state["island_states"] for m in cell["population"]))
        self.assertEqual(state["island_config"]["roles"]["assignments"], roles["assignments"])
        self.assertEqual([cell["role"].get("kind") for cell in state["island_states"]], ["generalist", "simplifier", "family:2", "explorer"])
        self.assertIn("island 2 simplifier, island 3 family:2, island 4 explorer", log)
        self.assertIn("(simplifier)=", log)

    def test_terminal_setup_asks_for_each_islands_role(self):
        answers = {"Island count": "3", "Island roles": "1", "Roles for islands": "simplifier, family:3"}
        def ask(prompt, default=""):
            return next((value for key, value in answers.items() if prompt.startswith(key)), str(default))
        df = pd.DataFrame({"a": np.arange(10.), "y": np.arange(10.)})
        args = a.parse_cli(["--population", "48"])[1]
        with tempfile.NamedTemporaryFile(suffix=".csv") as handle, \
                patch.object(a, "ask", side_effect=lambda prompt, default="": handle.name if prompt == "Dataset path" else ask(prompt, default)), \
                patch.object(a, "configure", return_value=(df, [1, 5], ",")), \
                patch.object(a, "choose_operator_groups", return_value=self.OPS), \
                contextlib.redirect_stdout(io.StringIO()) as log:
            setup = a.collect_training_setup(args)
        self.assertEqual(setup["roles"]["assignments"], ["simplifier", "family:3"])
        self.assertTrue(setup["roles"]["enabled"])
        self.assertIn("family:3 = Operators: arithmetic + exponential and logarithmic", log.getvalue())

    def scored(self, tree, loss, bits):
        m = model(tree, ops=tuple(self.OPS)); m.objectives = (loss, 0., float(bits), 0); return m

    def test_anchored_band_is_noise_floor_aware(self):
        exact = self.scored(("square", X0), 0., 40)
        dust = self.scored(("square", ("+", X0, ("c", 1e-9))), a.LOSS_NOISE_FLOOR / 2, 30)
        worse = self.scored(X0, 1e-3, 10)
        anchor, limit, inside = a.anchored_band([worse, dust, exact], .05)
        self.assertIs(anchor, exact)
        self.assertEqual(limit, a.LOSS_NOISE_FLOOR)                    # 5% of zero would leave no room at all
        self.assertEqual({id(m) for m in inside}, {id(exact), id(dust)})
        anchor, limit, inside = a.anchored_band([self.scored(X0, 2., 10), self.scored(X0, 2.09, 9), self.scored(X0, 2.11, 8)], .05)
        self.assertAlmostEqual(limit, 2.1); self.assertEqual(len(inside), 2)

    def test_anchored_survival_keeps_the_shortest_in_band_models(self):
        best = self.scored(("+", ("square", X0), ("sin", ("*", X0, ("c", 2.)))), 1., 120)    # 7 nodes
        close = [self.scored(("+", ("square", X0), ("c", float(i))), 1.02, 60 + i) for i in range(6)]     # in band, shorter
        bigger = self.scored(("+", ("+", ("square", X0), ("sin", X0)), ("*", X0, ("c", 3.))), 1.01, 200)  # in band, 9 nodes > anchor
        junk = [self.scored(("c", float(i)), 5. + i, 5) for i in range(10)]                    # short but far out of band
        pool = [best, *close, bigger, *junk]
        lane, cap = a.anchored_lane(pool)
        self.assertEqual(cap, 7); self.assertNotIn(bigger, lane)
        self.assertEqual(lane[0], close[0]); self.assertEqual(lane[-1], best)
        survivors = a.anchored_survivors(pool, 8, "intercept", .01)
        self.assertEqual(len(survivors), 8)
        self.assertEqual(survivors[:4], close[:4])                     # half the slots: shortest in-band first
        plain = a.select_nsga(pool, 8, "intercept", .01)
        self.assertGreater(sum(m in junk for m in plain), sum(m in junk for m in survivors[:4]))

    def test_anchored_emigrants_prefer_the_final_choice_band(self):
        best = self.scored(("square", X0), 1., 50)
        near = self.scored(("square", ("+", X0, ("c", 1.))), 1.005, 45)          # within 1%
        wider = self.scored(X0, 1.04, 10)                                        # within 5% only
        self.assertEqual(a.anchored_emigrants([best, near, wider], 1, "intercept", .01), [near])
        self.assertEqual(a.anchored_emigrants([best, wider], 2, "intercept", .01)[0], best)    # 1% band: just the anchor
        self.assertEqual(a.anchored_emigrants([self.scored(X0, 1., 50), wider], 1, "intercept", .01)[0].objectives[2], 50.)

    def test_anchored_generation_keeps_its_lane_and_size(self):
        state = self.cells(2)[1]
        for generation in range(3):
            advance(state, generation, self.X, self.Y, self.cats, self.OPS, self.ev, role_settings={"anchored": True, "neutral_shrink": True})
        self.assertEqual(len(state.population), state.population_size)
        lane, cap = a.anchored_lane(state.population)
        self.assertTrue(lane)
        self.assertTrue(all(a.tree_size_cap(m) <= cap for m in lane))

    def test_menu_offers_the_simplifiers_shortest_near_best_model(self):
        def entry(tree, loss, bits):
            m = model(tree, ops=tuple(self.OPS)); m.objectives = (loss, 0., float(bits), 0)
            return (m, m, {"loss": loss, "shape": 0., "mdl_bits": float(bits)})
        best = entry(("+", ("square", X0), ("sin", X0)), 1., 120)
        near = entry(("square", X0), 1.03, 60)                                   # within 5%, outside 1%
        nearer_elsewhere = entry(("+", X0, ("c", 1.)), 1.04, 40)                   # within 5%, not the simplifier's
        far = entry(X0, 2., 10)
        entries = [best, near, nearer_elsewhere, far]
        keys = {a.selection_identity(near[0]), a.selection_identity(far[0])}
        self.assertIs(a.simplifier_choice(entries, keys), near[0])
        self.assertIsNone(a.simplifier_choice(entries, {a.selection_identity(far[0])}))   # nothing in band
        self.assertIsNone(a.simplifier_choice(entries, set()))
        labels, choices, _ = a.model_options([e[0] for e in entries], cats=[None], evaluation=("training", entries), simplifier_keys=keys)
        index = next(i for i, label in enumerate(labels) if "simplifier island" in label)
        self.assertIs(choices[index], near[0]); self.assertIn("5%", labels[index])
        plain, _, _ = a.model_options([e[0] for e in entries], cats=[None], evaluation=("training", entries))
        self.assertFalse(any("simplifier" in label for label in plain))

    def test_unknown_role_in_setup_is_rejected(self):
        df = pd.DataFrame({"a": np.arange(40.), "y": np.arange(40.)})
        args = a.parse_cli(["--population", "32", "--max-generations", "1", "--seed", "1"])[1]
        setup = {"path": Path("unused.csv"), "df": df, "types": [1, 5], "delimiter": ",", "ops": ["+"], "affine_on": True,
                 "coev": False, "dynamic_pressure_on": False, "adf_enabled": False, "nodes": 7, "depth": 3, "island_count": 2,
                 "migration_interval": 5, "migrants_per_island": 1, "val_path": "0", "validation_percent": None, "metadata": {},
                 "roles": {"enabled": True, "assignments": ["family:3"]}}
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Unknown island role"):
            a.train_from_setup(args, setup)


class HistoryTests(unittest.TestCase):
    """Main-line model history: observational records of birth, variation, crossover and movement."""
    PLACE = {"island": 1, "stage": 0, "role": "simplifier"}

    def scored(self, tree, loss, bits, history=()):
        m = model(tree); m.objectives = (loss, 0., float(bits), 0); m.history = tuple(history); return m

    def test_variations_on_one_island_collapse_into_one_run(self):
        parent = self.scored(X0, 1., 50); a.history_born(parent, 3, self.PLACE)
        child = self.scored(("square", X0), .5, 60); a.history_varied(child, parent, 4, self.PLACE, ["subtree", None])
        grandchild = self.scored(("square", X0), .4, 40); a.history_varied(grandchild, child, 6, self.PLACE, ["prune"])
        self.assertEqual([r["event"] for r in grandchild.history], ["born", "variation"])
        run = grandchild.history[-1]
        self.assertEqual((run["start"], run["end"], run["count"], run["kinds"]), (4, 6, 2, {"subtree": 1, "prune": 1}))
        self.assertEqual((run["bits"], run["loss"]), ([50, 40], [1., .4]))
        self.assertEqual(child.history[-1]["count"], 1)                       # the parent's record was not mutated
        moved = self.scored(X0, .4, 40); a.history_varied(moved, grandchild, 7, {**self.PLACE, "island": 2}, ["point"])
        self.assertEqual(len(moved.history), 3)                               # a new place starts a new run

    def test_crossover_inherits_the_first_parent_and_names_the_partner(self):
        first = self.scored(X0, 1., 50); a.history_born(first, 0, self.PLACE)
        second = self.scored(X1, 2., 30); a.history_born(second, 0, {"island": 3, "stage": 0, "role": "explorer"})
        child = self.scored(("+", X0, X1), .5, 70); a.history_crossed(child, first, second, 5, self.PLACE)
        record = child.history[-1]
        self.assertEqual(child.history[0], first.history[0])
        self.assertEqual((record["event"], record["partner"], record["partner_from"]), ("crossover", second.lineage_id, {"island": 3, "role": "explorer"}))
        later = self.scored(("+", X0, X1), .4, 60); a.history_varied(later, child, 6, self.PLACE, ["constant"])
        self.assertEqual([r["event"] for r in later.history], ["born", "crossover", "variation"])   # never merged across a crossover

    def test_long_histories_keep_their_landmarks(self):
        m = self.scored(X0, 1., 50); a.history_born(m, 0, self.PLACE)
        for generation in range(300):
            place = {"island": generation % 3, "stage": 0, "role": "auto"}
            child = self.scored(X0, 1., 50)
            if generation % 7 == 0: a.history_crossed(child, m, m, generation, place)
            else: a.history_varied(child, m, generation, place, ["subtree"])
            if generation % 50 == 0: a.history_moved(child, "migrated", generation, place, {**place, "island": 9}, "ring")
            m = child
        self.assertLessEqual(len(m.history), a.HISTORY_LIMIT)
        self.assertEqual(m.history[0]["event"], "born")
        self.assertEqual(sum(r["event"] == "migrated" for r in m.history), 6)
        self.assertTrue(any(r.get("merged") for r in m.history))
        self.assertEqual(m.history[-1]["end"], 299)
        self.assertEqual(len(a.describe_history(m.history)), len(m.history))

    def test_history_is_observational(self):
        m = self.scored(X0, 1., 50); twin = m.clone(); a.history_born(twin, 0, self.PLACE)
        self.assertEqual(m, twin)                                             # not part of equality
        self.assertEqual(a.model_equivalence_key(m), a.model_equivalence_key(twin))
        self.assertIs(twin.clone().history, twin.history)

    def test_checkpoint_interning_round_trip_leaves_live_state_alone(self):
        shared = ({"event": "born", "generation": 0, "island": 0, "stage": 0, "role": None, "how": "seed", "bits": 3, "loss": 1.},)
        models = [self.scored(X0, 1., 3, shared + ({"event": "variation", "start": i, "end": i, "island": 0, "stage": 0, "role": None,
                                                   "count": 1, "kinds": {}, "bits": [3, 3], "loss": [1., 1.]},)) for i in range(3)]
        data = [a.PosteriorParticlePopulation._model_data(m) for m in models]
        live = {"island_states": [{"population": data}], "adf": {"history": {"keep": "me"}}, "mirror": data}   # shared, as island 0's runtime is
        payload = {"population": [dict(d) for d in data], "state": live}
        undo = a._intern_histories(payload)
        self.assertEqual(len(payload["history_records"]), 4)                  # the shared birth is stored once
        self.assertTrue(all(isinstance(i, int) for i in payload["population"][0]["history"]))
        restored = {"population": [dict(d) for d in payload["population"]], "history_records": payload["history_records"]}
        a._restore_histories(restored)
        self.assertEqual(restored["population"][2]["history"], list(models[2].history))
        for record, history in undo: record["history"] = history
        self.assertEqual(live["island_states"][0]["population"][1]["history"], list(models[1].history))
        self.assertEqual(live["adf"]["history"], {"keep": "me"})              # other "history" keys are untouched

    def test_snapping_is_the_final_record(self):
        X = np.linspace(.5, 2, 30)[:, None]; Y = 2. * X
        m = model(("*", ("c", 2.0000001), X0)); a.assess(m, X, Y, True, [None]); a.history_born(m, 0, self.PLACE)
        snapped, count = a.snap_model_constants(m, X, Y, None, None, True, [None])
        self.assertEqual(count, 1)
        record = snapped.history[-1]
        self.assertEqual((record["event"], record["constants"], record["values"]), ("snapped", 1, [[2.0000001, 2.]]))
        self.assertIn("final: snapped 1 constant(s)", a.describe_history(snapped.history)[-1])

    def test_gathered_copies_do_not_inherit_the_ring_record(self):
        X = np.random.default_rng(1).uniform(-2, 2, (40, 1)); Y = X ** 2; cats = [None]; ops = ["+", "-", "*", "square"]
        ev = a.ModelEvaluator(1, {"train": (X, Y)}, True, cats, a.compile_constraints(), ["y0"]); self.addCleanup(ev.close)
        cells = []
        for index in range(4):
            cell = island(X, cats, ops, index=index); cell.island_index, cell.stage = index, 0; cells.append(cell)
        a.assign_role_parameters(cells, 4, .35, .25, 15, ["auto", "simplifier", "auto"], ops)
        for cell in cells: advance(cell, 0, X, Y, cats, ops, ev, place=a.history_place(cell))
        a.migrate_islands(cells, 1, X=X, nsga_normalization="intercept", parsimony_quality_tolerance=.01, evaluator=ev, generation=7)
        for index, cell in enumerate(cells):
            for m in cell.population:
                moves = [r for r in m.history if r["event"] == "migrated" and r["generation"] == 7]
                self.assertLessEqual(len(moves), 1)
                if moves:
                    self.assertEqual(moves[0]["to"]["island"], index)
                    self.assertEqual(moves[0]["how"], "gathered" if index == 2 else "anchored emigrant" if index == 3 else "ring")


class FragmentAdmissionTests(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(0); n = 200
        self.X = r.normal(size=(n, 3))
        self.noise = r.normal(scale=.3, size=n) + (r.random(n) < .1) * 3      # skewed noise: a nonzero-mean residual

    def fitted(self, tree, Y, founders=None, ops=("+", "*", "sin", "gt")):
        m = a.Model([tree], [(1., 0.)], mdl_operators=ops, mdl_feature_count=3, **({"founder_ids": founders} if founders else {}))
        a.assess(m, self.X, Y, True, [None]); return m

    def library(self, models, Y):
        lib = a.FragmentLibrary(); lib.observe(models, self.X, Y, [None]); return lib

    def test_chance_and_constant_fragments_are_rejected(self):
        Y = (2 * self.X[:, 0] + self.noise)[:, None]
        tree = ("+", ("*", ("x", 0), ("c", 2.)), ("+", ("*", ("sin", ("x", 1)), ("c", 1e-9)), ("gt", ("x", 1), ("c", 1e9))))
        lib = self.library([self.fitted(tree, Y)], Y)
        for fragment in (("sin", ("x", 1)), ("gt", ("x", 1), ("c", 1e9))):
            self.assertNotIn(a.fragment_key(fragment), lib.items)

    def test_real_residual_structure_is_admitted_once(self):
        Y = (2 * self.X[:, 0] + np.sin(3 * self.X[:, 2]) + .1 * self.noise)[:, None]
        tree = ("+", ("*", ("x", 0), ("c", 2.)), ("*", ("sin", ("*", ("x", 2), ("c", 3.))), ("c", 1e-9)))
        lib = self.library([self.fitted(tree, Y, ops=("+", "*", "sin"))], Y)
        item = lib.items.get(a.fragment_key(("sin", ("*", ("x", 2), ("c", 3.)))))
        self.assertIsNotNone(item)
        self.assertGreater(item["contribution"], a.FRAGMENT_MIN_CONTRIBUTION)
        # c*sin(3*x2) and sin(3*x2) are one fragment after the affine fit.
        self.assertEqual(a.fragment_key(("*", ("sin", ("*", ("x", 2), ("c", 3.))), ("c", 1e-9))), a.fragment_key(item["tree"]))

    def test_support_counts_independent_lineages_only(self):
        Y = (2 * self.X[:, 0] + self.noise)[:, None]
        founder = self.fitted(("x", 0), Y)
        siblings = [self.fitted(("+", ("*", ("x", 0), ("c", c)), ("sin", ("x", 1))), Y, founder.founder_ids) for c in (1., 2., 3.)]
        lib = self.library(siblings, Y)
        self.assertEqual(lib.items[a.fragment_key(("sin", ("x", 1)))]["support"], 1)   # three siblings, one lineage
        strangers = [self.fitted(("+", ("*", ("x", 0), ("c", c)), ("sin", ("x", 1))), Y) for c in (1., 2.)]
        self.assertEqual(self.library(strangers, Y).items[a.fragment_key(("sin", ("x", 1)))]["support"], 2)


class ResidualArchiveTests(unittest.TestCase):
    def setUp(self):
        self.X = np.linspace(.1, 3, 60)[:, None]; self.Y = np.exp(self.X)      # targets span ~1 to ~20
        self.archive = a.ResidualQualityDiversityArchive(self.X, self.Y, [None], seed=1, capacity=8)

    def specialist(self, tree):
        m = model(tree, ops=("+", "-", "*", "square", "exp")); a.assess(m, self.X, self.Y, True, [None]); return m

    def test_descriptor_is_an_error_share_per_group(self):
        d = self.archive.descriptor(self.specialist(("x", 0)))
        self.assertEqual(len(d), a.RESIDUAL_TARGET_BINS + a.RESIDUAL_REGION_BINS)
        self.assertAlmostEqual(float(d[:a.RESIDUAL_TARGET_BINS].sum()), 1.)
        self.assertAlmostEqual(float(d[a.RESIDUAL_TARGET_BINS:].sum()), 1.)

    def test_complementary_partial_models_occupy_different_cells(self):
        low = self.specialist(("+", ("c", 1.), ("x", 0)))                  # right for small targets only
        high = self.specialist(("square", ("square", ("x", 0))))           # steep: fits the top end better
        self.archive.update([low, high])
        self.assertNotEqual(self.archive.cell(low), self.archive.cell(high))
        self.assertEqual(len(self.archive.cells), 2)
        restored = a.ResidualQualityDiversityArchive.from_snapshot(self.archive.snapshot())
        self.assertEqual(sorted(restored.cells), sorted(self.archive.cells))

    def test_runtime_gets_one_only_when_targets_are_given(self):
        cats = [None]
        self.assertIsNone(island(self.X, cats, ["+", "*"]).residual_qd)
        kwargs = dict(X=self.X, Xt=self.X, cats=cats, ops=["+", "*"], nodes=15, depth=4, head_count=1, bayesian_particles=12,
                      run_seed=1, island_index=0, qd_parent_rate=.2, nsga_normalization="intercept", parsimony_quality_tolerance=.01,
                      dynamic_pressure_on=True, stagnation_window=100, adf_enabled=False, adf_mode="nested",
                      evaluation_budget="baseline", evaluation_refresh=2, interaction_discovery={})
        state = a.new_island_runtime(12, Yt=self.Y, **kwargs)
        self.assertIsNotNone(state.residual_qd)
        ev = a.ModelEvaluator(1, {"train": (self.X, self.Y)}, True, cats, a.compile_constraints(), ["y0"]); self.addCleanup(ev.close)
        for generation in range(3): advance(state, generation, self.X, self.Y, cats, ["+", "*"], ev, residual_qd=state.residual_qd)
        self.assertTrue(state.residual_qd.cells)
        restored = a.island_from_snapshot(a.island_snapshot(state), len(self.X), .01)
        self.assertEqual(len(restored.residual_qd.cells), len(state.residual_qd.cells))


class ParentChoiceTests(unittest.TestCase):
    def archive(self, losses):
        archive = a.QualityDiversityArchive(np.zeros((1, 1)), [None], seed=1)
        for key, loss in enumerate(losses):
            m = model(("x", 0)); m.objectives = (loss, 0., 3., 0); archive.cells[key] = m
        return archive

    def picks(self, archive, choice):
        self.addCleanup(setattr, a, "QD_PARENT_CHOICE", a.QD_PARENT_CHOICE)
        a.QD_PARENT_CHOICE = choice; a.rng.seed(4)
        return [parent.cell for parent in archive.sample_tagged(4000, "semantic", uniform_rate=0.)]

    def test_quality_rank_is_preferred_but_bounded(self):
        picks = self.picks(self.archive([.1, .5, 1., 5.]), "quality_coverage")
        counts = [picks.count(k) for k in range(4)]
        self.assertGreater(counts[0], counts[3] * 3)          # the best cell is clearly preferred...
        self.assertGreater(counts[3], 0)                      # ...but the worst is still drawn
        self.assertLess(counts[0] / len(picks), .6)           # and nothing takes over

    def test_coverage_bonus_favours_untried_cells(self):
        archive = self.archive([1., 1.])
        archive.cell_trials = {0: 50.}; archive.cell_successes = {0: 25.}
        picks = self.picks(archive, "quality_coverage")
        self.assertGreater(picks.count(1), picks.count(0))

    def test_legacy_ignores_quality(self):
        picks = self.picks(self.archive([.1, 5.]), "legacy")
        self.assertAlmostEqual(picks.count(0) / len(picks), .5, delta=.05)


class ScaleBalancedSelectionTests(unittest.TestCase):
    def test_every_magnitude_band_gets_equal_weight(self):
        Y = np.r_[np.full(90, .01), np.linspace(1, 1000, 10)][:, None]
        w = a.scale_balanced_row_weights(Y, [None], bins=2)
        self.assertAlmostEqual(float(w.mean()), 1.)
        self.assertAlmostEqual(float(w[:90].sum()), float(w[90:].sum()), places=6)

    def test_small_target_errors_count_in_balanced_mode(self):
        X = np.array([[0.], [1.]]); Y = np.array([[.01], [1000.]])
        small_ok = model(("c", .01)); big_ok = model(("c", 1000.))
        plain = a.lexicase_parents([small_ok, big_ok], 400, X, Y, [None])
        balanced = a.lexicase_parents([small_ok, big_ok], 400, X, Y, [None], scale_balanced=True)
        self.assertGreater(sum(p is small_ok for p in balanced), 150)
        self.assertEqual(len(plain), 400)


class NumericGuardTests(unittest.TestCase):
    """Models may not use afpo's numeric safety guards as hidden nonlinearities."""
    def setUp(self):
        r = np.random.default_rng(1); n = 80
        clip = r.integers(0, 18, n).astype(float); reserve = r.integers(0, 40, n).astype(float)
        self.X = np.column_stack([clip, reserve]); self.Y = np.minimum(clip + reserve, 17.)[:, None]

    def scored(self, tree):
        m = a.Model([tree], [(1., 0.)], mdl_operators=tuple(a.OPS), mdl_feature_count=2)
        a.assess(m, self.X, self.Y, True, [None]); return m

    def test_guard_exploits_are_infeasible(self):
        exploits = {
            # min(x, 17) through the +/-1e12 value clamp (the reported equation).
            "value_clamp": ("*", ("+", ("+", ("x", 0), ("c", -3.58557)), ("x", 1)), ("c", -7.45466e10)),
            # A plateau from pow's exponent clip, and one from exp_decay's input clip amplified by pow.
            "input_clip": ("pow", ("c", 1.0000003), ("delta", ("*", ("c", -1.), ("x", 0)), ("+", ("c", -4.99999), ("x", 1)))),
        }
        for reason, tree in exploits.items():
            m = self.scored(tree)
            self.assertFalse(m.feasible)
            self.assertEqual(m.invalid_reason, f"numeric_guard:{reason}")
        composed = self.scored(("pow", ("exp_decay", ("+", ("x", 0), ("x", 1)), ("c", 2.94118)), ("c", -1.35472e-07)))
        self.assertEqual(composed.invalid_reason, "numeric_guard:input_clip")
        self.assertFalse(self.scored(("sinh", ("*", ("c", 10.), ("x", 0)))).feasible)

    def test_honest_and_naturally_saturating_models_stay_valid(self):
        for tree in (("min", ("+", ("x", 0), ("x", 1)), ("c", 17.)),
                     ("sigmoid", ("*", ("c", 1000.), ("-", ("x", 0), ("c", 8.5)))),     # steep step
                     ("exp", ("*", ("c", -20.), ("x", 1))),                             # decays to ~0
                     ("pow", ("x", 0), ("c", 2.)),
                     ("+", ("sin", ("x", 0)), ("log", ("x", 1)))):
            m = self.scored(tree)
            self.assertTrue(m.feasible, (tree, m.invalid_reason))

    def test_constant_fitter_does_not_tune_into_a_guard(self):
        x = self.X[:, :1] + self.X[:, 1:]
        tree = ("*", ("+", ("x", 0), ("c", -3.)), ("c", 2.))
        tuned = a.fit_tree_constants(tree, x, self.Y[:, 0])
        self.assertEqual(a.guard_engagement([tuned], x), "")

    def test_switch_off_restores_the_old_behaviour(self):
        self.addCleanup(setattr, a, "GUARD_EXPLOIT_CHECK", a.GUARD_EXPLOIT_CHECK)
        a.GUARD_EXPLOIT_CHECK = False
        self.assertTrue(self.scored(("*", ("+", ("+", ("x", 0), ("c", -3.58557)), ("x", 1)), ("c", -7.45466e10))).feasible)


class ConstantSelectionTests(unittest.TestCase):
    def setUp(self):
        self.X = np.linspace(0, 1, 20)[:, None]; self.Y = (1 + .3 * self.X[:, 0])[:, None]
        self.linear = model(("x", 0)); self.constant = model(("c", 1.))
        for m in (self.linear, self.constant): a.assess(m, self.X, self.Y, True, [None])

    def options(self, Xv, Yv):
        evaluation = a.selection_evaluation([self.linear, self.constant], Xv, Yv, [None])
        return a.model_options([self.linear, self.constant], cats=[None], evaluation=evaluation)

    def test_constant_win_on_validation_is_explained_and_the_training_fit_offered(self):
        Xv = np.array([[0.], [.5], [1.]]); Yv = np.array([[1.4], [.9], [1.1]])     # the trend fails on held-out rows
        labels, choices, selection = self.options(Xv, Yv)
        self.assertIs(choices[0], self.constant)
        self.assertIn("did not find structure", selection["warning"])
        offered = dict(zip(labels, choices))["Lowest Training Loss (not supported by validation)"]
        self.assertIs(offered, self.linear)

    def test_no_warning_when_a_real_model_wins(self):
        Xv = np.array([[0.], [.5], [1.]]); Yv = 1 + .3 * Xv
        labels, choices, selection = self.options(Xv, Yv)
        self.assertIs(choices[0], self.linear)
        self.assertIsNone(selection["warning"])


class InterpolationCheckTests(unittest.TestCase):
    """Equations that only memorise the training grid must lose to honest ones."""
    def setUp(self):
        import benchmark_afpo
        self.X, y, self.Xtest, self.ytest = benchmark_afpo.synthetic_cases(64)["ratio_wrap"]
        self.Y = y[:, None]
        c = lambda v: ("c", float(v))
        ratio = ("/", ("*", c(100.), ("x", 0)), ("x", 1))
        self.true = ("if_else", ("gt", ratio, c(1.65)), ("/", ratio, c(2.)), ("if_else", ("lt", ratio, c(.8)), ("*", ratio, c(2.)), ratio))
        # From the 20-seed benchmark: fits the training grid (R2 0.92), test R2 0.003.
        self.sawtooth = ("mod", ("+", c(-103.542), ("x", 1)), c(-1.08552))
        # Partial but honest: handles the upper wrap, misses the lower one.
        self.smooth = ("if_else", ("gt", ratio, c(1.65)), ("/", ratio, c(2.)), ratio)

    def loss(self, tree, check=True):
        self.addCleanup(setattr, a, "INTERPOLATION_CHECK", a.INTERPOLATION_CHECK)
        a.INTERPOLATION_CHECK = check
        m = a.Model([tree], [(1., 0.)], mdl_operators=tuple(a.OPS), mdl_feature_count=2)
        a.assess(m, self.X, self.Y, True, [None]); return a.aggregate_loss(m)

    def test_memoriser_wins_without_the_check_and_loses_with_it(self):
        self.assertLess(self.loss(self.sawtooth, False), self.loss(self.smooth, False))
        self.assertGreater(self.loss(self.sawtooth), self.loss(self.smooth))
        self.assertEqual(self.loss(self.true), 0.)            # an exactly correct model pays nothing

    def test_probes_sit_between_neighbours_at_irregular_fractions(self):
        probes, rows, partner = a.neighbour_probes(self.X)
        self.assertEqual(len(probes), a.INTERPOLATION_PROBES)
        spacing = float(np.median(np.diff(np.sort(self.X[:, 1]))))
        offsets = np.array([np.min(np.abs(self.X[:, 1] - p)) for p in probes[:, 1]]) / spacing
        self.assertTrue(np.all(offsets > .1))                      # never on a training row
        self.assertGreater(float(np.std(offsets)), .05)            # not one fixed fraction (aliasing)
        self.assertTrue(np.all(probes[:, 0] == self.X[0, 0]))       # constant column copied, not blended
        self.assertIs(a.neighbour_probes(self.X)[0], probes)


class NoiseFloorTests(unittest.TestCase):
    """At loss ~ 0, rounding noise must not decide between an exact model and bloated copies."""
    def setUp(self):
        r = np.random.default_rng(3)
        self.X = r.normal(size=(400, 2)) * 1e-2
        self.Y = np.round(self.X[:, :1] * self.X[:, 1:], 9)            # a CSV-style ~7-digit product
        self.exact = model(("*", ("x", 0), ("x", 1)), ops=tuple(a.OPS), features=2)
        # The same product times a factor that is ~1 on the data, fitted into the noise.
        self.bloated = model(("*", ("*", ("x", 0), ("x", 1)), ("log10", ("+", ("c", 9.99999), ("*", ("x", 0), ("c", 1e-7))))),
                             ops=tuple(a.OPS), features=2)
        for m in (self.exact, self.bloated): a.assess(m, self.X, self.Y, True, [None])

    def scored(self, tree, loss, shape, bits):
        m = model(tree, ops=tuple(a.OPS), features=2); m.objectives = (loss, shape, float(bits), 0); return m

    def run_pair(self):
        # The reported run: the exact product lost by 1.5% of a 5e-11 loss (rounding noise).
        exact = self.scored(("*", ("x", 0), ("x", 1)), 4.764e-11, 2.62e-19, 51)
        bloated = self.scored(("*", ("*", ("x", 0), ("x", 1)), ("log10", ("c", 10.))), 4.695e-11, 2.61e-19, 160)
        return exact, bloated

    def test_selection_prefers_the_shorter_model_within_the_noise_floor(self):
        exact, bloated = self.run_pair()
        entries = [(m, m, {"loss": m.objectives[0], "shape": m.objectives[1], "mdl_bits": m.objectives[2]}) for m in (bloated, exact)]
        chosen, _ = a._select_best(("validation", entries), .01)
        self.assertIs(chosen, exact)

    def test_archive_drops_noise_better_bloat(self):
        exact, bloated = self.run_pair()
        archive = a.ParetoArchive(parsimony_quality_tolerance=.01)
        archive.update([bloated, exact])
        self.assertEqual([repr(m.trees) for m in archive.items], [repr(exact.trees)])

    def test_best_so_far_is_offered_to_the_archive(self):
        cats = [None]; ops = ["+", "-", "*", "square"]
        state = island(self.X, cats, ops)
        ev = a.ModelEvaluator(1, {"train": (self.X, self.Y)}, True, cats, a.compile_constraints(), ["y0"]); self.addCleanup(ev.close)
        exact = model(("*", ("x", 0), ("x", 1)), ops=tuple(ops), features=2); a.assess(exact, self.X, self.Y, True, cats)
        state.best_models.model = exact                                  # found as a child, never in the population
        advance(state, 0, self.X, self.Y, cats, ops, ev)
        self.assertIn(a.model_equivalence_key(exact), {a.model_equivalence_key(m) for m in state.archive.items})


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
            self.assertIn("residual", state["island_states"][0]["runtime"]["quality_diversity"])
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


class ParallelCellTests(unittest.TestCase):
    """Cells own their random streams, so serial, parallel and resumed runs agree exactly."""
    def run_cells(self, plan, cell_workers):
        x = np.random.default_rng(0).uniform(1, 3, (40, 2))
        df = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": x[:, 0] ** 2 + 2 * x[:, 1]})
        with tempfile.TemporaryDirectory(prefix="afpo-cells-") as directory, contextlib.chdir(directory):
            df.to_csv("d.csv", index=False)
            flags = ["--workers", "1", "--cell-workers", str(cell_workers)]
            args = a.parse_cli(["--population", "64", "--max-generations", str(plan[0]), "--seed", "4", *flags])[1]
            setup = {"path": Path("d.csv"), "df": df, "types": [1, 1, 5], "delimiter": ",", "ops": ["+", "-", "*", "square"],
                     "affine_on": True, "coev": False, "dynamic_pressure_on": True, "adf_enabled": False, "nodes": 15, "depth": 4,
                     "island_count": 2, "migration_interval": 2, "migrants_per_island": 2, "val_path": "", "validation_percent": 20,
                     "metadata": {}, "stages": {"mode": "both", "count": 2, "interval": 2, "age_gap": 2}, "roles": {"enabled": True, "interval": 2}}
            with contextlib.redirect_stdout(io.StringIO()):
                checkpoint = a.train_from_setup(args, setup, choose_model=lambda *_: 0)["checkpoint"]
                for target in plan[1:]:
                    a.resume_main(a.parse_cli(["--resume", checkpoint, "--max-generations", str(target), *flags])[1])
            _, _, _, _, state = a.load_checkpoint(checkpoint, False)
        return [[repr(m["trees"]) for m in cell["population"]] for cell in state["island_states"]], state

    def test_parallel_and_resumed_cells_match_a_straight_serial_run(self):
        serial, state = self.run_cells([6], 1)
        parallel, _ = self.run_cells([4, 6], 4)
        self.assertEqual(serial, parallel)
        self.assertTrue(all("streams" in cell for cell in state["island_states"]))
        lineages = {m["lineage_id"] for cell in state["island_states"] for m in cell["population"]}
        self.assertTrue(any(lineage >= 1 << a.LINEAGE_RANGE_BITS for lineage in lineages))

    def test_scoring_workers_do_not_change_results(self):
        # Scoring (tuning, the jump-constant scan) runs in --workers processes;
        # nothing there may draw from the run's random streams.
        x = np.random.default_rng(1).uniform(-2, 2, (60, 2))
        df = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": np.where(x[:, 0] > .4, 3., 1.) + np.mod(x[:, 1], 1.3)})
        def run(workers):
            with tempfile.TemporaryDirectory(prefix="afpo-workers-") as directory, contextlib.chdir(directory):
                df.to_csv("d.csv", index=False)
                args = a.parse_cli(["--population", "48", "--max-generations", "4", "--seed", "9", "--workers", str(workers)])[1]
                setup = {"path": Path("d.csv"), "df": df, "types": [1, 1, 5], "delimiter": ",",
                         "ops": ["+", "-", "*", "gt", "mod", "floor", "if_else"], "affine_on": True, "coev": False,
                         "dynamic_pressure_on": True, "adf_enabled": False, "nodes": 21, "depth": 5, "island_count": 1,
                         "migration_interval": 0, "migrants_per_island": 0, "val_path": "", "validation_percent": 20, "metadata": {}}
                with contextlib.redirect_stdout(io.StringIO()):
                    checkpoint = a.train_from_setup(args, setup, choose_model=lambda *_: 0)["checkpoint"]
                population = a.load_checkpoint(checkpoint, False)[1]
            return [(repr(m.trees), m.objectives[:-1]) for m in population]
        self.assertEqual(run(1), run(3))

    def test_jump_scan_leaves_the_run_streams_alone(self):
        X = np.random.default_rng(2).uniform(-2, 2, (40, 1)); y = np.floor(X[:, 0])
        tree = ("+", ("+", ("gt", ("x", 0), ("c", .1)), ("gt", ("x", 0), ("c", -.9))),
                ("+", ("gt", ("x", 0), ("c", 1.2)), ("mod", ("x", 0), ("c", 2.))))   # 4 jump constants, 3 are scanned
        flat = a._FlatTree(tree); start = np.array(a.constant_vector(tree))
        state = a.rng.getstate(); first = a.scan_jump_constants(flat, start, X, y, a.loss_scale(y))
        self.assertEqual(a.rng.getstate(), state)
        a.rng.seed(123); np.testing.assert_array_equal(a.scan_jump_constants(flat, start, X, y, a.loss_scale(y)), first)

    def test_cell_streams_restore_the_shared_streams(self):
        cell = type("Cell", (), {"streams": None})()
        a.seed_cell_streams([cell], 7)
        before = (a.rng.getstate(), a.np.random.get_state()[1].tobytes(), a._NEXT_LINEAGE_ID)
        with a.cell_streams(cell): a.rng.random(); a.np.random.random(); a.next_lineage_id()
        self.assertEqual(before, (a.rng.getstate(), a.np.random.get_state()[1].tobytes(), a._NEXT_LINEAGE_ID))
        self.assertEqual(cell.streams["lineage_next"], (1 << a.LINEAGE_RANGE_BITS) + 1)


if __name__ == "__main__":
    unittest.main()
