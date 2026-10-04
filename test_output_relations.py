"""Input relations, separate per-output searches and staged output relations.

Run with: python -B -m unittest -v test_output_relations
"""
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import afpo as a

X = lambda i: ("x", i)
C = lambda v: ("c", float(v))


class InputRelationTests(unittest.TestCase):
    names = ["HH1", "MM1", "HH2", "MM2", "z"]

    def setUp(self):
        a.configure_input_relations(["HH1,MM1;HH2,MM2"], self.names, self.names, [1] * 5)

    def tearDown(self):
        a.configure_input_relations((), self.names)

    def test_parts_of_a_relation_do_not_meet_outside_inputs(self):
        self.assertTrue(a.relation_violation(("-", X(2), X(0))))
        self.assertTrue(a.relation_violation(("+", X(4), X(0))))
        self.assertTrue(a.relation_violation(("*", ("+", X(0), X(1)), X(2))))

    def test_complete_relations_combine(self):
        minutes = lambda h, m: ("+", ("*", C(60), X(h)), X(m))
        self.assertEqual(a.relation_violation(("-", minutes(2, 3), minutes(0, 1))), "")
        self.assertEqual(a.relation_violation(("*", X(0), ("sin", X(1)))), "")
        self.assertEqual(a.relation_violation(("+", X(4), ("+", X(0), X(1)))), "")

    def test_scoring_marks_violations_infeasible(self):
        Xd = np.random.default_rng(0).uniform(1, 2, (20, 5)); Y = Xd[:, [0]]
        m = a.Model([("-", X(2), X(0))], [(1., 0.)], mdl_operators=("-",), mdl_feature_count=5)
        a.assess(m, Xd, Y, True, [None])
        self.assertFalse(m.feasible)
        self.assertTrue(m.invalid_reason.startswith("relations:"))

    def test_random_trees_respect_relations(self):
        a.rng.seed(4)
        trees = [a.admissible_random_tree(5, ["+", "-", "*"], 9, 4) for _ in range(200)]
        self.assertGreater(sum(not a.relation_violation(t) for t in trees) / len(trees), .95)

    def test_bad_specs_are_rejected(self):
        with self.assertRaises(ValueError): a.parse_relations("HH1")
        with self.assertRaises(ValueError): a.parse_relations("a,b;b,c")
        with self.assertRaises(ValueError): a.configure_input_relations("HH1,nope", self.names, self.names, [1] * 5)


def result(model, names, cats, types=None):
    return {"model": model, "names": names, "cats": cats, "types": types or [], "maps": {}}


class MergeTests(unittest.TestCase):
    """Staged features are replaced by the source equations, exactly."""
    names = ["u", "v"]
    Xd = np.random.default_rng(1).uniform(-2, 2, (200, 2))

    def merged_prediction(self, source, source_cats, stage_tree):
        staged_name = "s_pred"
        sub_names = self.names + ([staged_name] if source_cats is None else [f"{staged_name}={label}" for label in source_cats])
        results = {"s": result(source, self.names, [source_cats]),
                   "t": result(a.Model([stage_tree], [(2., 1.)], mdl_operators=("+", "*")), sub_names, [None])}
        cats = [source_cats, None]
        merged = a.merge_separate_models([("s", ()), ("t", ("s",))], results, {"s": staged_name}, self.names, ["s", "t"], cats)
        predicted = a.predict_targets(source, self.Xd, [source_cats])[:, 0]
        if source_cats is None: staged = predicted[:, None]
        else: staged = (predicted[:, None] == np.arange(len(source_cats))[None, :]).astype(float)
        expected = a.predict_targets(results["t"]["model"], np.hstack([self.Xd, staged]), [None])[:, 0]
        return a.predict_targets(merged, self.Xd, cats)[:, 1], expected

    def test_numeric_source(self):
        source = a.Model([("*", X(0), X(1))], [(3., -1.)])
        got, want = self.merged_prediction(source, None, ("+", X(2), X(0)))
        np.testing.assert_allclose(got, want)

    def test_binary_source(self):
        source = a.Model([("-", X(0), X(1))], [(1.5, .5)])
        got, want = self.merged_prediction(source, ["no", "yes"], ("+", X(3), X(0)))
        np.testing.assert_allclose(got, want)

    def test_multiclass_source(self):
        source = a.Model([X(0), X(1), ("*", X(0), X(1))], [(1., 0.), (1., .2), (2., 0.)])
        got, want = self.merged_prediction(source, ["a", "b", "c"], ("+", ("*", X(3), C(5)), X(4)))
        np.testing.assert_allclose(got, want)

    def test_adfs_of_two_searches_keep_apart(self):
        adf = {"adf_0": {"tree": ("+", ("arg", 0), C(1)), "arity": 1}}
        first = a.Model([("adf_0", X(0))], [(1., 0.)], adfs=dict(adf))
        second = a.Model([("adf_0", X(1))], [(1., 0.)], adfs={"adf_0": {"tree": ("*", ("arg", 0), C(3)), "arity": 1}})
        results = {"p": result(first, self.names, [None]), "q": result(second, self.names, [None])}
        merged = a.merge_separate_models([("p", ()), ("q", ())], results, {}, self.names, ["p", "q"], [None, None])
        got = a.predict_targets(merged, self.Xd, [None, None])
        np.testing.assert_allclose(got[:, 0], self.Xd[:, 0] + 1)
        np.testing.assert_allclose(got[:, 1], self.Xd[:, 1] * 3)


class PlanTests(unittest.TestCase):
    columns = ["x", "year", "month", "day", "lat", "lon", "z"]
    types = [1, 5, 5, 6, 5, 5, 5]

    def plan(self, spec):
        return a.separate_output_plan(self.columns, self.types, a.parse_output_relations(spec))

    def test_chains_read_every_ancestor(self):
        self.assertEqual(self.plan("year -> month -> day; lat -> lon"),
                         [("year", ()), ("month", ("year",)), ("day", ("year", "month")), ("lat", ()), ("lon", ("lat",)), ("z", ())])

    def test_stage_groups_and_comma_chains(self):
        self.assertEqual(self.plan("lat, year -> day")[:3], [("lat", ()), ("year", ()), ("day", ("lat", "year"))])
        self.assertEqual(self.plan("lon,lat")[:2], [("lon", ()), ("lat", ("lon",))])

    def test_ancestors_are_searched_first(self):
        self.assertEqual([column for column, _ in self.plan("month -> day; year -> month")][:3], ["year", "month", "day"])

    def test_bad_graphs_are_rejected(self):
        with self.assertRaises(ValueError): self.plan("year -> month; month -> year")
        with self.assertRaises(ValueError): self.plan("x -> year")
        with self.assertRaises(ValueError): self.plan("year ->")


class OpaqueTests(unittest.TestCase):
    """An inlined earlier equation is a leaf to the structural rules, not a reason to switch them off."""
    names = ["HH1", "MM1", "z"]

    def setUp(self):
        a.configure_input_relations("HH1,MM1", self.names, self.names, [1, 1, 1])

    def tearDown(self):
        a.configure_input_relations((), self.names)

    def test_opaque_subtree_stands_for_a_staged_feature(self):
        inlined = ("*", C(2), X(0))                  # reads only part of (HH1, MM1)
        tree = ("+", inlined, X(2))
        self.assertTrue(a.relation_violation(tree))
        self.assertEqual(a.relation_violation(tree, frozenset([inlined])), "")
        self.assertTrue(a.relation_violation(("+", ("+", inlined, X(0)), X(2)), frozenset([inlined])))
        Xd = np.random.default_rng(0).uniform(1, 2, (20, 3))
        m = a.Model([tree], [(1., 0.)], mdl_operators=("+", "*"), mdl_feature_count=3, opaque=(inlined,))
        a.assess(m, Xd, Xd[:, [2]], True, [None], fit_affine=False)
        self.assertTrue(m.feasible)


class SeparateRunTests(unittest.TestCase):
    """A short real run: one search per output, merged and exported."""

    def test_end_to_end(self):
        r = np.random.default_rng(0); n = 120
        u = r.uniform(-2, 2, n); v = r.uniform(-2, 2, n)
        frame = pd.DataFrame({"u": u, "v": v, "p": 2 * u + 1, "q": u * v, "k": np.where(u > 0, "hi", "lo")})
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            frame.to_csv("data.csv", index=False)
            setup = {"df": frame, "path": Path("data.csv"), "types": [1, 1, 5, 5, 6], "delimiter": ",", "ops": ["+", "-", "*"],
                     "affine_on": True, "coev": False, "dynamic_pressure_on": True, "adf_enabled": False, "nodes": 9, "depth": 4,
                     "island_count": 1, "migration_interval": 0, "migrants_per_island": 0, "val_path": "", "validation_percent": 20,
                     "metadata": {}, "stages": None, "roles": None}
            args = a.parse_cli(["--population", "24", "--max-generations", "2", "--seed", "3", "--workers", "1",
                                "--output-relations", "p -> q", "--symbolic-export", "off", "--constant-intervals", "off"])[1]
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                outcome = a.train_from_setup(args, setup, choose_model=lambda labels, choices, evaluation: 0)
            text = log.getvalue()
            self.assertEqual(len(outcome["checkpoints"]), 3)
            self.assertIn("Output 2/3: q (numeric; reads predicted p)", text)
            self.assertNotIn("differs from the per-output choices", text)
            self.assertTrue(Path("best_model.py").exists())
            self.assertTrue(Path(outcome["manifest"]).exists())


if __name__ == "__main__":
    unittest.main()
