"""--custom-op: partly known operators with v0/v1 parameters, optional v# slots and fixed columns.

Run with: python -B -m unittest -v test_custom_ops
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import afpo as a

NAMES = ["amount", "seconds", "m", "c", "bar", "kpop"]
X = lambda i: ("x", i)
ARG = lambda i: ("arg", i)


def configure(*specs):
    return a.configure_custom_ops(list(specs), NAMES, NAMES, [1] * len(NAMES))


class ParseTests(unittest.TestCase):
    def tearDown(self):
        a.configure_custom_ops((), NAMES)

    def test_columns_and_parameters(self):
        configure("fq = amount / v0", "speed = v0/seconds")
        self.assertEqual(a.CUSTOM_OPS["adf_fq"]["tree"], ("/", X(0), ARG(0)))
        self.assertEqual(a.CUSTOM_OPS["adf_speed"]["tree"], ("/", ARG(0), X(1)))

    def test_shared_parameters(self):
        configure("foobar = (v0-v1)/(v0+v1)")
        item = a.CUSTOM_OPS["adf_foobar"]
        self.assertEqual(item["arity"], 2)
        self.assertEqual(item["tree"], ("/", ("-", ARG(0), ARG(1)), ("+", ARG(0), ARG(1))))

    def test_no_parameters_is_a_derived_feature(self):
        derived = configure("E = m*c**2")
        self.assertEqual(derived, [("E", ("*", X(2), ("square", X(3))))])
        self.assertEqual(a.CUSTOM_OPS, {})

    def test_optional_slot_drops_its_operation(self):
        derived = configure("foo = bar + v#*kpop")
        self.assertEqual(a.CUSTOM_OPS["adf_foo"]["tree"], ("+", X(4), ("*", ARG(0), X(5))))
        self.assertEqual(derived, [("foo()", ("+", X(4), X(5)))])

    def test_earlier_operators_are_reused(self):
        configure("foobar = (v0-v1)/(v0+v1)", "maybefoobar = v#*foobar", "twice = foobar(v0, 2) * 2")
        self.assertEqual(a.CUSTOM_OPS["adf_maybefoobar"]["arity"], 3)          # foobar's two parameters come along
        self.assertEqual(a.CUSTOM_OPS["adf_maybefoobar__0"]["tree"], a.CUSTOM_OPS["adf_foobar"]["tree"])
        self.assertEqual(a.CUSTOM_OPS["adf_twice"]["tree"],
                         ("*", ("/", ("-", ARG(0), ("c", 2.)), ("+", ARG(0), ("c", 2.))), ("c", 2.)))

    def test_bad_specs(self):
        for spec in ("nonsense", "f = nope + v0", "sin = v0", "f = sin(v0, v1)", "f = v0 if v1 else 2"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                configure(spec)


class EvaluationTests(unittest.TestCase):
    Xd = np.random.default_rng(0).uniform(1, 3, (30, 6))

    def tearDown(self):
        a.configure_custom_ops((), NAMES)

    def test_operator_evaluates_and_displays_by_name(self):
        configure("fq = amount / v0")
        tree = ("adf_fq", ("c", 2.))
        np.testing.assert_allclose(a.evaluate(tree, self.Xd), self.Xd[:, 0] / 2)
        self.assertEqual(a.expr(tree, NAMES), "fq(2)")
        self.assertEqual(a.operator_arity("adf_fq"), 1)

    def test_user_operator_bodies_are_not_charged_as_definitions(self):
        configure("fq = amount / v0")
        m = a.Model([("adf_fq", ("c", 2.))], [(1., 0.)], mdl_operators=("+", "adf_fq"), mdl_feature_count=6)
        self.assertEqual(a.model_description(m)["adf_definition_bits"], 0)
        self.assertEqual(a.used_feature_indices(m), (0,))       # the body's column counts for the export contract

    def test_export_inlines_derived_features(self):
        derived = configure("E = m*c**2")
        names = NAMES + [label for label, _ in derived]
        m = a.Model([("+", X(6), X(0))], [(1., 0.)])
        inlined = a.inline_custom_features(m, names)
        np.testing.assert_allclose(a.predict_model(inlined, self.Xd)[:, 0], self.Xd[:, 2] * self.Xd[:, 3] ** 2 + self.Xd[:, 0])


class RunTests(unittest.TestCase):
    # A run sets module-wide options (symbolic export, intervals, ...); later test files expect the defaults.
    SAVED = ("SYMBOLIC_EXPORT", "CONSTANT_INTERVALS", "CLIP", "EPS")

    def setUp(self):
        self.saved = {name: getattr(a, name) for name in self.SAVED}

    def tearDown(self):
        for name, value in self.saved.items(): setattr(a, name, value)
        a.configure_custom_ops((), [])
        a.configure_input_relations((), [])

    def test_a_run_uses_and_exports_custom_operators(self):
        r = np.random.default_rng(1); n = 120
        frame = pd.DataFrame({"m": r.uniform(1, 3, n), "c": r.uniform(1, 3, n), "z": r.uniform(1, 3, n)})
        frame["E"] = frame.m * frame.c ** 2 + frame.z
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            frame.to_csv("data.csv", index=False)
            setup = {"df": frame, "path": Path("data.csv"), "types": [1, 1, 1, 5], "delimiter": ",", "ops": ["+", "-", "*"],
                     "affine_on": True, "coev": False, "dynamic_pressure_on": True, "adf_enabled": False, "nodes": 9, "depth": 4,
                     "island_count": 1, "migration_interval": 0, "migrants_per_island": 0, "val_path": "", "validation_percent": 20,
                     "metadata": {}, "stages": None, "roles": None}
            args = a.parse_cli(["--population", "24", "--max-generations", "3", "--seed", "1", "--workers", "1",
                                "--custom-op", "mc2 = m*c**2", "--custom-op", "plus = z + v0",
                                "--symbolic-export", "off", "--constant-intervals", "off"])[1]
            with contextlib.redirect_stdout(io.StringIO()):
                a.train_from_setup(args, setup, choose_model=lambda labels, choices, evaluation: 0)
            exported = Path("best_model.py").read_text()
            self.assertNotIn("'mc2'", exported.split("'contract'")[1].split("'feature_count'")[0])
            spec = __import__("importlib.util").util.spec_from_file_location("exported", "best_model.py")
            module = __import__("importlib.util").util.module_from_spec(spec); spec.loader.exec_module(module)
            self.assertEqual(len(module.predict_frame(frame)), n)


if __name__ == "__main__":
    unittest.main()


class NumericLimitTests(unittest.TestCase):
    def tearDown(self):
        a.set_numeric_limits(a.DEFAULT_CLIP, a.DEFAULT_EPS)

    def test_ordinary_data_keeps_the_defaults(self):
        self.assertEqual(a.automatic_limits(np.array([[0.01, 500.], [-3., 2.]])), (1e12, 1e-12))

    def test_large_and_small_data_move_the_limits(self):
        clip, eps = a.automatic_limits(np.array([3e8, 1.]))
        self.assertGreater(clip, (3e8) ** 2)
        clip, eps = a.automatic_limits(np.array([1e-9, 1.]))
        self.assertLess(eps, 1e-18)
        self.assertEqual(a.resolve_numeric_limits("default", np.array([3e8])), (1e12, 1e-12))
        self.assertEqual(a.resolve_numeric_limits("1e20", np.array([1.]))[0], 1e20)

    def test_clamp_and_guard_follow_the_limits(self):
        x = np.array([3e8])
        self.assertEqual(float(a.op_eval("square", [x])[0]), 1e12)       # default clamp
        a.set_numeric_limits(1e40, 1e-30)
        self.assertEqual(float(a.op_eval("square", [x])[0]), 9e16)
        self.assertEqual(float(a.op_eval("/", [np.array([1e-20]), np.array([1e-20])])[0]), 1.)


class EquationTextTests(unittest.TestCase):
    """The GUI's editable equation text round-trips exactly."""
    names = ["a", "b c", "x=1"]

    def tearDown(self):
        a.configure_custom_ops((), NAMES)

    def test_round_trip(self):
        for tree in [("+", ("*", ("c", 2.5), X(0)), ("sin", X(1))), ("pow", ("neg", X(2)), ("c", .1 + .2)),
                     ("gt", X(0), ("c", -1e-7)), ("max", X(0), ("mod", X(1), ("c", 3.))), ("floordiv", X(0), ("c", 7.))]:
            with self.subTest(tree=tree):
                self.assertEqual(a.parse_equation(a.tree_text(tree, self.names), self.names), tree)

    def test_readout_and_custom_operators(self):
        configure("fq = amount / v0")
        text = a.readout_text(("adf_fq", ("c", 2.)), (3., -1.), NAMES)
        self.assertEqual(text, "3.0 * fq(2.0) + -1.0")
        self.assertEqual(a.parse_equation(text, NAMES), ("+", ("*", ("c", 3.), ("adf_fq", ("c", 2.))), ("c", -1.)))
        with self.assertRaises(ValueError): a.parse_equation("fq(1, 2)", NAMES)
