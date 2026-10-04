"""End-to-end checks for the afpo browser GUI's server side.

Run with: python -B -m unittest -v test_afpo_gui
Everything runs in a temporary working directory (runs, exports and uploads land there).
"""
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import afpo_gui as gui


def form(csv, **search):
    return {
        "mode": "train",
        "data": {"path": str(csv), "delimiter": ",", "types": [1, 1, 1, 5], "validation_mode": "percent", "validation_percent": 20},
        "operators": {"groups": ["1", "4"], "excluded": ["mod", "floordiv"]},
        "search": {"nodes": 15, "depth": 4, "affine": True, "islands": 1, **search},
        "run": {"max_generations": 6, "population": 40, "seed": 3, "workers": 2},
        "advanced": {"checkpoint_every": 3},
    }


def wait_for(session, states, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = session.status()
        if status["state"] in states:
            return status
        time.sleep(.2)
    raise AssertionError(f"timed out waiting for {states}; last state {status['state']}\n" + "\n".join(status["console"][-20:]))


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous = os.getcwd()
        cls.tmp = tempfile.TemporaryDirectory()
        os.chdir(cls.tmp.name)
        r = np.random.default_rng(0)
        X = r.uniform(-2, 2, (120, 3))
        cls.csv = Path(cls.tmp.name) / "toy.csv"
        pd.DataFrame({"a": X[:, 0], "b": X[:, 1], "c": X[:, 2], "y": X[:, 0] * X[:, 1] + np.sin(X[:, 2])}).to_csv(cls.csv, index=False)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.previous)
        cls.tmp.cleanup()

    def test_inspect_suggests_cli_defaults(self):
        info = gui.inspect_dataset(self.csv, ",")
        self.assertEqual(info["rows"], 120)
        self.assertEqual([c["suggested"] for c in info["columns"]], [1, 1, 1, 5])
        self.assertTrue(all(c["hist"] for c in info["columns"]))

    def test_check_rejects_bad_forms(self):
        bad = form(self.csv)
        bad["data"]["types"] = [1, 1, 1, 1]
        with self.assertRaises(ValueError):
            gui.check_form(bad)
        bad = form(self.csv)
        bad["advanced"] = {"crossover_rate": 2}
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            gui.check_form(bad)
        self.assertTrue(gui.check_form(form(self.csv))["ok"])

    def test_relation_fields_become_flags(self):
        related = form(self.csv)
        related["data"]["input_relations"] = "a,b\n"
        related["data"]["output_relations"] = "y -> y2\n\n"
        related["data"]["custom_ops"] = "fq = a / v0\n"
        argv = gui.build_argv(related)
        self.assertIn(["--input-relations", "a,b"], [argv[i:i + 2] for i in range(len(argv))])
        self.assertIn(["--output-relations", "y -> y2"], [argv[i:i + 2] for i in range(len(argv))])
        self.assertIn(["--custom-op", "fq = a / v0"], [argv[i:i + 2] for i in range(len(argv))])
        self.assertIn("output_mode", {item["dest"] for item in gui.options()["advanced"]})
        self.assertNotIn("input_relations", {item["dest"] for item in gui.options()["advanced"]})

    def test_stage_and_role_fields_reach_the_setup(self):
        staged = form(self.csv, islands=2, migration_interval=5, migrants=1, stage_mode="both", stages=2,
                      stage_interval=4, stage_quantile=.4, stage_age_gap=6, stage_schedule="linear", roles=True, role_interval=7)
        setup = gui.setup_answers(staged)
        self.assertEqual({k: setup["stages"][k] for k in ("mode", "count", "interval", "threshold_quantile", "age_gap", "schedule")},
                         {"mode": "both", "count": 2, "interval": 4, "threshold_quantile": .4, "age_gap": 6, "schedule": "linear"})
        self.assertEqual((setup["roles"]["enabled"], setup["roles"]["interval"]), (True, 7))
        self.assertFalse(gui.setup_answers(form(self.csv, roles=True))["roles"]["enabled"])     # one island: no roles
        self.assertTrue(gui.check_form(staged)["ok"])
        crowded = form(self.csv, islands=2, stage_mode="age", stages=3)                          # 40 models over 6 cells
        with self.assertRaisesRegex(ValueError, "eight models each"):
            gui.check_form(crowded)

    def test_island_role_dropdowns_reach_the_setup(self):
        options = gui.options()
        self.assertEqual([r["id"] for r in options["roles"]], ["auto", "generalist", "simplifier", "explorer", "refiner"])
        self.assertTrue(all(r["name"] and r["description"] for r in options["roles"]))
        chosen = form(self.csv, islands=4, roles=True, island_roles=["simplifier", "auto", "family:4"])
        setup = gui.setup_answers(chosen)
        self.assertEqual(setup["roles"]["assignments"], ["simplifier", "auto", "family:4"])
        self.assertEqual(gui.setup_answers(form(self.csv, islands=3, roles=True))["roles"]["assignments"], ["auto", "auto"])
        self.assertEqual(gui.setup_answers(form(self.csv, islands=3, roles=False, island_roles=["simplifier", "auto"]))["roles"]["assignments"], [])
        with self.assertRaisesRegex(ValueError, "Unknown island role"):                     # group 3 is not ticked
            gui.setup_answers(form(self.csv, islands=2, roles=True, island_roles=["family:3"]))
        with self.assertRaisesRegex(ValueError, "one entry per island"):
            gui.setup_answers(form(self.csv, islands=3, roles=True, island_roles=["auto"]))

    def test_live_labels_carry_island_roles(self):
        telemetry = gui.Telemetry(None, 3, 2, ["simplifier", "auto"])
        self.assertEqual([telemetry.cell_name(i) for i in (0, 3, 5)], ["island 1 stage 1 (generalist)", "island 2 stage 2 (simplifier)", "island 3 stage 2 (auto)"])
        self.assertEqual(gui.Telemetry(None, 2).cell_name(1), "island 2")
        history = ({"event": "born", "generation": 0, "island": 1, "stage": 0, "role": "explorer", "how": "seed", "bits": 5, "loss": 1.},
                   {"event": "migrated", "generation": 5, "from": {"island": 1, "stage": 0, "role": "explorer"},
                    "to": {"island": 3, "stage": 0, "role": "simplifier"}, "how": "gathered"})
        self.assertEqual(gui.afpo.history_path(history), ["island 2 (explorer)", "island 4 (simplifier)"])

    def test_train_choose_then_explore(self):
        session = gui.TrainingSession()
        session.start(form(self.csv))
        status = wait_for(session, {"choosing", "failed", "finished"})
        self.assertEqual(status["state"], "choosing", "\n".join(status["console"][-30:]))
        self.assertEqual({row["generation"] for row in status["history"]}, set(range(6)))
        self.assertIsNotNone(status["snapshot"])
        self.assertTrue(status["snapshot"]["archive"] and status["snapshot"]["population"])
        self.assertGreaterEqual(len(status["choose"]["options"]), 1)
        session.pick(len(status["choose"]["options"]) - 1)
        status = wait_for(session, {"finished", "failed"})
        self.assertEqual(status["state"], "finished", status["error"])
        checkpoint = status["done"]["checkpoint"]
        self.assertTrue(Path(checkpoint).is_file())
        self.assertTrue(Path("best_model.py").is_file())

        explorer = gui.ModelExplorer()
        summary = explorer.load(checkpoint)
        self.assertEqual(summary["outputs"], ["y"])
        self.assertTrue(summary["models"])
        self.assertTrue(any(m["label"] for m in summary["models"]))
        index = next(m["index"] for m in summary["models"] if m["label"])
        detail = explorer.detail(index)
        self.assertIn("<svg", detail["svg"])
        self.assertTrue(detail["history"])
        self.assertEqual(detail["history"][0]["event"], "born")
        self.assertEqual(len(detail["history_text"]), len(detail["history"]))
        math = explorer.latex(index)
        if math["available"]:  # sympy is optional
            first = math["outputs"][0]
            self.assertEqual(first["name"], "y")
            self.assertTrue(first["raw"]["latex"] and first["exact"]["latex"])
            self.assertIn("<m", first["raw"]["mathml"])
        self.assertIs(explorer.latex(index), math)  # cached per model
        fit = explorer.fit(index, "validation")
        self.assertEqual(fit["split"], "validation")
        self.assertEqual(fit["outputs"][0]["kind"], "regression")
        sweep = explorer.sweep(index, "a")
        self.assertEqual(len(sweep["x"]), len(sweep["outputs"]["y"]["values"]))
        # Frozen inputs are honoured: every point equals a single-row prediction at those values.
        frozen = explorer.sweep(index, "a", base={"b": 1.5, "c": -.5}, lo=-1, hi=1, points=5)
        self.assertEqual(frozen["x"], [-1., -.5, 0., .5, 1.])
        for x, value in zip(frozen["x"], frozen["outputs"]["y"]["values"]):
            self.assertAlmostEqual(value, explorer.predict(index, {"a": x, "b": 1.5, "c": -.5})["outputs"]["y"], places=9)
        grid = explorer.grid(index, "a", "b", base={"c": .3}, x_lo=-1, x_hi=1, y_lo=0, y_hi=2, points=7)
        values = np.asarray(grid["outputs"]["y"]["values"])
        self.assertEqual(values.shape, (7, 7))
        self.assertAlmostEqual(values[6, 0], explorer.predict(index, {"a": -1, "b": 2, "c": .3})["outputs"]["y"], places=9)
        self.assertEqual(set(grid["data"]["coords"]), {"a", "b"})
        self.assertEqual(set(grid["data_sets"]), {"validation"})           # no test CSV in this run
        # Editing: the text round-trips, and an edit becomes a new, scored candidate.
        heads = explorer.edit_text(index)["heads"]
        self.assertEqual(len(heads), 1)
        same = explorer.edit(index, [heads[0]["text"]])
        np.testing.assert_allclose(explorer.sweep(same["index"], "a")["outputs"]["y"]["values"], sweep["outputs"]["y"]["values"])
        shifted = explorer.edit(index, [f"({heads[0]['text']}) + 1"])
        np.testing.assert_allclose(np.asarray(explorer.sweep(shifted["index"], "a")["outputs"]["y"]["values"]),
                                   np.asarray(sweep["outputs"]["y"]["values"]) + 1)
        self.assertTrue(shifted["summary"]["models"][shifted["index"]]["label"].startswith("Edited"))
        with self.assertRaisesRegex(ValueError, "Unknown input"):
            explorer.edit(index, ["a + nope"])
        with self.assertRaises(ValueError):
            explorer.grid(index, "a", "a")
        with self.assertRaises(ValueError):
            explorer.sweep(index, "a", lo=2, hi=1)
        self.assertEqual(detail_inputs := explorer.detail(index)["inputs_used"], [c for c in ("a", "b", "c") if c in detail_inputs])
        single = explorer.predict(index, {"a": 1, "b": 2, "c": 0})
        batch = explorer.predict_csv(index, self.csv)
        self.assertEqual(batch["rows"], 120)
        first = pd.read_csv(self.csv).iloc[0]
        row = explorer.predict(index, {"a": first.a, "b": first.b, "c": first.c})["outputs"]["y"]
        self.assertAlmostEqual(row, float(pd.read_csv(batch["path"])["predicted_y"].iloc[0]), places=9)
        self.assertIsInstance(single["outputs"]["y"], float)
        self.assertIsNotNone(batch["metrics"])
        self.check_generation(explorer, index)
        exported = explorer.export(index)
        self.assertTrue(any(p.endswith("best_model.py") for p in exported["written"]))

    def generate(self, explorer, index, spec):
        started = explorer.generate(index, spec)
        self.assertTrue(started.get("started"), started)
        deadline = time.time() + 120
        while explorer.generate_status()["state"] == "running" and time.time() < deadline:
            time.sleep(.05)
        status = explorer.generate_status()
        self.assertEqual(status["state"], "finished", status["error"])
        return pd.read_csv(status["path"])

    def check_generation(self, explorer, index):
        self.assertGreater(explorer.generate_defaults(index)["outputs"][0]["rmse"], -1)
        frame = self.generate(explorer, index, {"path": "gen_uniform.csv", "rows": 1000, "seed": 1, "sampling": "uniform",
                                                "inputs": {"a": {"lo": -1, "hi": 1}, "c": {"vary": False, "value": .5}}})
        self.assertEqual(list(frame.columns), ["a", "b", "c", "y"])
        self.assertEqual(len(frame), 1000)
        self.assertTrue((frame.c == .5).all() and frame.a.between(-1, 1).all())
        row = frame.iloc[0]
        self.assertAlmostEqual(row.y, explorer.predict(index, {"a": row.a, "b": row.b, "c": row.c})["outputs"]["y"], places=6)
        # Latin hypercube: exactly one row in each of the n equal strata of every varied input.
        frame = self.generate(explorer, index, {"path": "gen_lhs.csv", "rows": 100, "seed": 2, "sampling": "lhs",
                                                "inputs": {"a": {"lo": 0, "hi": 1}}})
        self.assertEqual(sorted(np.floor(frame.a * 100).astype(int)), list(range(100)))
        frame = self.generate(explorer, index, {"path": "gen_grid.csv", "rows": 100, "sampling": "grid",
                                                "inputs": {"c": {"vary": False, "value": 0}}})
        self.assertEqual(len(frame), 100)
        self.assertEqual(len(frame[["a", "b"]].drop_duplicates()), 100)
        frame = self.generate(explorer, index, {"path": "gen_train.csv", "rows": 300, "seed": 3, "sampling": "training"})
        self.assertTrue(frame.a.isin(pd.read_csv(self.csv).a).all())
        frame = self.generate(explorer, index, {"path": "gen_int.csv", "rows": 50, "seed": 6, "inputs": {"b": {"integer": True, "lo": 0, "hi": 5}}})
        self.assertTrue(pd.api.types.is_integer_dtype(frame.b) and frame.b.between(0, 5).all())
        noisy = self.generate(explorer, index, {"path": "gen_noise.csv", "rows": 20000, "seed": 4, "outputs": {"y": {"noise": .5}}})
        clean = explorer.predict_csv(index, "gen_noise.csv")
        residual = pd.read_csv(clean["path"]).eval("y - predicted_y")
        self.assertAlmostEqual(residual.std(), .5, delta=.02)
        self.assertTrue(explorer.generate(index, {"path": "gen_noise.csv", "rows": 5})["exists"])
        with self.assertRaises(ValueError):
            explorer.generate(index, {"path": "bad.csv", "rows": 0})
        self.assertFalse(Path("bad.csv").exists() or Path("gen_uniform.csv.part").exists())

    def test_generate_classification_samples_valid_labels(self):
        r = np.random.default_rng(4)
        X = r.uniform(-2, 2, (150, 2))
        labels = np.where(X[:, 0] ** 2 + X[:, 1] ** 2 < 1.5, "inside", np.where(X[:, 0] > 0, "right", "left"))
        path = Path(self.tmp.name) / "cls.csv"
        pd.DataFrame({"a": X[:, 0], "g": np.where(X[:, 1] > 0, "up", "down"), "label": labels}).to_csv(path, index=False)
        request = form(path)
        request["data"]["types"] = [1, 2, 6]
        session = gui.TrainingSession()
        session.start(request)
        wait_for(session, {"choosing", "failed"})
        session.pick(0)
        explorer = gui.ModelExplorer()
        explorer.load(wait_for(session, {"finished", "failed"})["done"]["checkpoint"])
        frame = self.generate(explorer, 0, {"path": "gen_cls.csv", "rows": 2000, "seed": 5,
                                            "inputs": {"g": {"vary": False, "value": "up"}},
                                            "outputs": {"label": {"mode": "sample", "probabilities": True}}})
        self.assertEqual(list(frame.columns[:3]), ["a", "g", "label"])
        self.assertTrue((frame.g == "up").all() and frame.label.isin(["inside", "left", "right"]).all())
        probabilities = frame[[c for c in frame.columns if c.startswith("P(label=")]]
        self.assertEqual(probabilities.shape[1], 3)
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1))

    def test_stop_saves_checkpoint_and_offers_choice(self):
        session = gui.TrainingSession()
        stoppable = form(self.csv)
        stoppable["run"]["max_generations"] = 0      # run until stopped
        session.start(stoppable)
        deadline = time.time() + 120
        while time.time() < deadline and not session.status()["history"]:
            time.sleep(.2)
        session.stop()
        status = wait_for(session, {"choosing", "failed"})
        self.assertEqual(status["state"], "choosing", "\n".join(status["console"][-30:]))
        self.assertTrue(any("Search stopped; checkpoint saved" in line for line in status["console"]))
        session.pick(0)
        self.assertEqual(wait_for(session, {"finished", "failed"})["state"], "finished")

    def test_resume_continues_a_checkpoint(self):
        session = gui.TrainingSession()
        session.start(form(self.csv))
        status = wait_for(session, {"choosing", "failed"})
        session.pick(0)
        checkpoint = wait_for(session, {"finished", "failed"})["done"]["checkpoint"]
        resume = {"mode": "resume", "checkpoint": checkpoint, "run": {"max_generations": 9, "workers": 1}}
        session.start(resume)
        status = wait_for(session, {"finished", "failed"})
        self.assertEqual(status["state"], "finished", status["error"])
        self.assertEqual(sorted({row["generation"] for row in status["history"]}), [6, 7, 8])


if __name__ == "__main__":
    unittest.main()


def _slow(seconds):
    time.sleep(seconds)
    return seconds


def _fail():
    raise ValueError("boom")


class IsolatedJobTests(unittest.TestCase):
    """Heavy symbolic work runs in a killable child, never under the explorer lock."""

    def test_result_timeout_error_and_cancel(self):
        job = gui.IsolatedJob()
        self.assertEqual(job.run(_slow, (0.01,), 10), 0.01)
        with self.assertRaises(TimeoutError):
            job.run(_slow, (30,), 0.3)
        with self.assertRaisesRegex(RuntimeError, "boom"):
            job.run(_fail, (), 10)
        import threading
        outcome = {}
        waiter = threading.Thread(target=lambda: outcome.setdefault("first", self._capture(job, 30)))
        waiter.start(); time.sleep(0.3)
        started = time.time()
        self.assertEqual(job.run(_slow, (0.01,), 10), 0.01)           # the newer request kills the older one
        waiter.join(5)
        self.assertIsInstance(outcome.get("first"), InterruptedError)
        self.assertLess(time.time() - started, 5)

    @staticmethod
    def _capture(job, seconds):
        try:
            return job.run(_slow, (seconds,), 60)
        except Exception as error:
            return error
