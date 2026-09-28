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
        fit = explorer.fit(index, "validation")
        self.assertEqual(fit["split"], "validation")
        self.assertEqual(fit["outputs"][0]["kind"], "regression")
        sweep = explorer.sweep(index, "a")
        self.assertEqual(len(sweep["x"]), len(sweep["outputs"]["y"]))
        single = explorer.predict(index, {"a": 1, "b": 2, "c": 0})
        batch = explorer.predict_csv(index, self.csv)
        self.assertEqual(batch["rows"], 120)
        first = pd.read_csv(self.csv).iloc[0]
        row = explorer.predict(index, {"a": first.a, "b": first.b, "c": first.c})["outputs"]["y"]
        self.assertAlmostEqual(row, float(pd.read_csv(batch["path"])["predicted_y"].iloc[0]), places=9)
        self.assertIsInstance(single["outputs"]["y"], float)
        self.assertIsNotNone(batch["metrics"])
        exported = explorer.export(index)
        self.assertTrue(any(p.endswith("best_model.py") for p in exported["written"]))

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
