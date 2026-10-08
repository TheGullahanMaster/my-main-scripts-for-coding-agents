"""Checks for the .tvq reader (EasyNN-plus / JustNN) and the importers built on it.

Run with: python -B -m unittest -v test_tvq_reader
The AFPO and mlpRes6 parts run in a temporary working directory (imports land there) and are skipped
when their dependencies are missing; the whole file is skipped for sample files that are not present.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

try:                    # torch has to load before pandas in some environments
    import torch
except ImportError:
    torch = None

import tvq_reader

HERE = Path(__file__).resolve().parent


def sample(name):
    path = HERE / name
    if not path.is_file():
        raise unittest.SkipTest(f"{name} is not in {HERE}")
    return str(path)


class InTempDir(unittest.TestCase):
    def setUp(self):
        self._cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()


class ReaderTests(unittest.TestCase):
    def test_numeric_file(self):
        t = tvq_reader.read_tvq(sample("Fibonacci revisited.tvq"))
        self.assertEqual(t["version"], 0)
        self.assertEqual([c["name"] for c in t["columns"]], [f"#{i}" for i in range(8)])
        self.assertEqual([c["role"] for c in t["columns"]], [0, 0, 1, 1, 1, 1, 1, 1])
        self.assertEqual(len(t["examples"]), 358)
        self.assertEqual(t["examples"][0]["name"], "Query")
        self.assertEqual(t["examples"][1]["raw"][:4], [24.0, 22.0, 46.0, 68.0])      # each value is the sum of the two before
        self.assertEqual(t["tail"], b"")

    def test_scaled_values_follow_lowest_highest(self):
        t = tvq_reader.read_tvq(sample("Fibonacci revisited.tvq"))
        for e in t["examples"][1:20]:
            for c, raw, norm in zip(t["columns"], e["raw"], e["norm"]):
                self.assertAlmostEqual(norm, (raw - c["min"]) / (c["max"] - c["min"]), places=9)

    def test_text_codes_follow_the_easynn_rule(self):
        t = tvq_reader.read_tvq(sample("WTF BMI.tvq"))
        codes = tvq_reader.codebook(t, 3)
        self.assertEqual(codes["Normal"], 2108)           # 78*6 + 111*5 + 114*4 + 109*3 + 97*2 + 108
        self.assertEqual(sum(ord(ch) * (3 - i) for i, ch in enumerate("Dog")), 529)    # the manual's example

    def test_bmi_data_is_plausible(self):
        """The file's BMI column is weight / height^2 (height in cm): the decoded grid must agree with itself."""
        ds = tvq_reader.dataset(tvq_reader.read_tvq(sample("WTF BMI.tvq")))
        self.assertEqual([c["name"] for c in ds["columns"]], ["Height", "Weight", "BMI", "BMI Index"])
        for height, weight, bmi, label in ds["rows"]:
            self.assertAlmostEqual(bmi, weight / (height / 100) ** 2, delta=0.01)
        labels = {row[3] for row in ds["rows"]}
        self.assertEqual(labels, {"Normal", "OverWight", "UnderWight", "Obese"})

    def test_bad_inputs(self):
        with tempfile.NamedTemporaryFile(suffix=".tvq") as empty:
            with self.assertRaises(ValueError):
                tvq_reader.read_tvq(empty.name)

    def test_wide_strings_are_decoded(self):
        t = tvq_reader.read_tvq(sample("Petra.tvq"))
        self.assertTrue(all("\x00" not in c["name"] for c in t["columns"]))

    def test_na_like_labels_are_renamed(self):
        self.assertEqual(tvq_reader.safe_text("N/A"), "N/A_")
        self.assertEqual(tvq_reader.safe_text("Normal"), "Normal")


class NetworkTests(unittest.TestCase):
    def test_easynn_network_reproduces_its_targets(self):
        t = tvq_reader.read_tvq(sample("Fibonacci revisited.tvq"))
        net = tvq_reader.network(t)
        self.assertEqual(net["sizes"], [2, 8, 4, 5, 6])
        self.assertEqual(net["scale"], 1.0)
        self.assertLess(tvq_reader.network_quality(t, net)["ratio"], 0.05)

    def test_justnn_weights_are_double(self):
        t = tvq_reader.read_tvq(sample("Demons Detector.tvq"))
        self.assertEqual(tvq_reader.weight_scale(t), 2.0)
        net = tvq_reader.network(t)
        self.assertLess(tvq_reader.network_quality(t, net)["ratio"], 0.1)       # about 0.02 with the factor, 0.7 without

    def test_stored_node_state_matches_the_formulas(self):
        """net input = bias + sum(weight * source activation), activation = logistic(net)."""
        import math
        t = tvq_reader.read_tvq(sample("Fibonacci revisited.tvq"))
        for n in t["nodes"]:
            if n["kind"] == 0:
                continue
            net = n["d"][4] + sum(t["weights"][i]["w"] * t["nodes"][t["weights"][i]["src"]]["d"][1]
                                  for i in range(n["w_first"], n["w_end"]))
            self.assertAlmostEqual(net, n["d"][0], places=6)
            self.assertAlmostEqual(1 / (1 + math.exp(-n["d"][0])), n["d"][1], places=6)

    def test_unusable_networks_say_why(self):
        t = tvq_reader.read_tvq(sample("Merboy.tvq"))
        report = tvq_reader.network_report(t)
        self.assertFalse(report["compatible"])
        self.assertTrue(report["reasons"])


class AfpoImportTests(InTempDir):
    def setUp(self):
        super().setUp()
        try:
            import afpo_gui
        except ImportError as exc:
            self.skipTest(f"afpo_gui is not importable here: {exc}")
        self.gui = afpo_gui

    def test_inspect_converts_and_suggests_roles(self):
        result = self.gui.inspect_dataset(sample("WTF BMI.tvq"))
        self.assertTrue(result["converted_from"].endswith("WTF BMI.tvq"))
        self.assertEqual(result["delimiter"], ",")
        self.assertEqual({c["name"]: c["suggested"] for c in result["columns"] if not c["reserved"]},
                         {"Height": 1, "Weight": 1, "BMI": 0, "BMI Index": 6})
        self.assertTrue(Path(result["path"]).is_file())

    def test_text_becomes_afpo_encoded_text_with_the_files_codes(self):
        import pandas as pd
        from afpo_lib import editor_format as editor
        result = self.gui.inspect_dataset(sample("WTF BMI.tvq"))
        schema = editor.editor_schema_from_frame(pd.read_csv(result["path"]))
        column = next(c for c in schema["columns"] if c["name"] == "BMI Index")
        self.assertEqual(column["kind"], editor.EDITOR_TEXT)
        self.assertEqual(column["codes"]["Normal"], 2108)
        self.assertEqual(editor.editor_decode_text(4300, column["codes"]), "OverWight")

    def test_unusable_files_are_refused_with_a_reason(self):
        with self.assertRaisesRegex(ValueError, "no usable columns"):
            self.gui.inspect_dataset(sample("Copy of Merboy.tvq"))
        with tempfile.NamedTemporaryFile(suffix=".tvq") as empty:
            with self.assertRaisesRegex(ValueError, "could not be read"):
                self.gui.inspect_dataset(empty.name)

    def test_starting_a_run_on_a_raw_tvq_is_refused(self):
        with self.assertRaisesRegex(ValueError, "Inspect columns"):
            self.gui.check_form({"mode": "train", "operators": {"groups": ["1"], "excluded": []},
                                 "data": {"path": sample("WTF BMI.tvq"), "delimiter": ",", "types": [1, 1, 0, 6]},
                                 "run": {"population": 40}, "search": {"islands": 1}})


@unittest.skipIf(torch is None, "torch is not installed")
class MlpRes6Tests(InTempDir):
    def setUp(self):
        super().setUp()
        import mlpRes6
        self.m = mlpRes6

    def test_hashed_text_type(self):
        m = self.m
        self.assertEqual(m.hash_codebook(["Dog", "Cat", "Dog"]), {"Dog": 529, "Cat": 713})
        self.assertEqual(m.hash_label(520, {"Dog": 529, "Cat": 713}), "Dog")
        Path("c.csv").write_text("size,colour,price\n" + "\n".join(f"{i},{['Red', 'Green', 'Blue'][i % 3]},{i * 10}" for i in range(1, 11)))
        vocab = {}
        m._setup_vocab_for_col("colour", "inhash", "c.csv", ",", vocab, {})
        types = m.lower_hash_types({"size": "in", "colour": "inhash", "price": "out"})
        self.assertEqual(types["colour"], "in")
        ds = m.CustomDataset("c.csv", ",", ["size", "colour"], ["price"], types, vocab, {}, {})
        self.assertEqual(len(ds), 10)
        self.assertEqual(sorted(ds.vocabularies["colour"]), ["Blue", "Green", "Red"])
        codes = ds.vocabularies["colour"]
        encoded = m.encode_sample_input([3, "Blue"], types, ds.vocabularies, ds.scalings, {})
        lo, hi = min(codes.values()), max(codes.values())
        self.assertAlmostEqual(encoded[1], -1 + 2 * (codes["Blue"] - lo) / (hi - lo))

    def test_network_import_matches_the_original(self):
        for name, sizes in (("Fibonacci revisited.tvq", [2, 8, 4, 5, 6]),       # numeric
                            ("Demons Detector.tvq", [6, 6, 1]),                # JustNN weights, hashed text
                            ("WTF BMI.tvq", [2, 4, 6, 5, 1]),                  # text output
                            ("Louie.tvq", [2, 6, 4, 4, 4])):                   # last layer as wide as the output: identity skip
            with self.subTest(name):
                result = self.m.gui_tvq_network(sample(name))
                if sizes:
                    self.assertEqual(result["sizes"], sizes)
                self.assertLess(result["input_error"], 1e-5)
                self.assertLess(result["output_error"], 1e-5)

    def test_diverged_network_is_refused(self):
        report = tvq_reader.network_report(tvq_reader.read_tvq(sample("Add.tvq")))      # one bias in this file is NaN
        self.assertFalse(report["compatible"])
        self.assertIn("NaN", " ".join(report["reasons"]))

    def test_current_model_is_only_replaced_when_asked(self):
        m = self.m
        first = m.gui_tvq_network(sample("WTF BMI.tvq"), activate=True)
        self.assertTrue(first["activated"])
        before = Path("config.json").read_text()
        second = m.gui_tvq_network(sample("Fibonacci revisited.tvq"), activate=True)
        self.assertFalse(second["activated"])
        self.assertTrue(second["needs_overwrite"])
        self.assertEqual(Path("config.json").read_text(), before)
        third = m.gui_tvq_network(sample("Fibonacci revisited.tvq"), activate=True, overwrite=True)
        self.assertTrue(third["activated"])
        self.assertTrue(all(Path(p).is_file() for p in third["backed_up"]))
        self.assertEqual(Path(third["backed_up"][1]).read_text(), before)

    def test_imported_model_runs_in_the_explorer(self):
        self.m.gui_tvq_network(sample("WTF BMI.tvq"), activate=True)
        sampler = self.m.InteractiveSampler("model.pt", "config.json")
        prediction = sampler.predict({"Height": 180, "Weight": 120})["outputs"][0]
        self.assertEqual(prediction["value"], "OverWight")
        self.assertAlmostEqual(prediction["raw"], 4326.97, delta=1.0)           # the query row's output stored in the file

    def test_config_records_the_csv_format(self):
        """Without it Resume guesses the delimiter from the column names, which fails for names with spaces or commas."""
        self.m.gui_tvq_network(sample("Analyser of DT.tvq"), activate=True)
        config = json.loads(Path("config.json").read_text())
        self.assertEqual(config["csv_format"]["delimiter"], ",")
        self.assertTrue(config["csv_format"]["header"])

    def test_repeated_validating_rows_are_not_a_held_out_set(self):
        """EasyNN often repeats training rows as validating ones; as a held-out set they would remove every training row."""
        result = self.m.gui_tvq_network(sample("Analyser of DT.tvq"), activate=True)
        self.assertEqual(result["validation_rows"], 0)
        self.assertIn("repeat training rows", result["validation_note"])
        self.assertFalse(Path("model_validation.csv").exists())

    def test_distinct_validating_rows_become_the_held_out_set(self):
        result = self.m.gui_tvq_network(sample("formrates.tvq"), activate=True)
        self.assertEqual(result["validation_rows"], 100)
        self.assertEqual(len(self.m.read_table("model_validation.csv", delimiter=",")), 100)
        kwargs, val_dataset, report = self.m.prepare_resume("config.json", "model.pt")
        self.assertEqual(len(val_dataset), 100)
        self.assertEqual(report["val_rows_also_in_training"] if "val_rows_also_in_training" in report else 0, 0)

    def test_resume_trains_the_imported_network(self):
        """Resume current model on an imported network must start training, not fail or do nothing."""
        import time
        for name in ("Analyser of DT.tvq", "WTF BMI.tvq", "formrates.tvq"):        # repeated validating rows / text / held-out set
            with self.subTest(name):
                for old in Path(".").glob("model*"):
                    old.unlink()
                self.m.gui_tvq_network(sample(name), activate=True, overwrite=True)
                session = self.m.TrainingSession()
                session.start({"resume": True, "optimizer": "Adam", "lr": 0.01, "batch_size": 8})
                deadline = time.time() + 120
                while time.time() < deadline:
                    status = session.status(0)
                    if status["state"] == "error" or status["total"] >= 20:
                        break
                    time.sleep(0.2)
                session.stop()
                for _ in range(100):
                    if session.status(0)["state"] in ("done", "error", "idle"):
                        break
                    time.sleep(0.2)
                self.assertIsNone(status["error"], status["error"])
                self.assertGreaterEqual(status["total"], 20)
                self.assertTrue(status["summary"]["resumed"])

    def test_preview_of_a_tvq_presets_roles(self):
        preview = self.m.gui_preview_dataset(sample("WTF BMI.tvq"))
        self.assertEqual({c["name"]: c["suggested"] for c in preview["columns"]},
                         {"Height": "in", "Weight": "in", "BMI": "i", "BMI Index": "outhash"})
        self.assertTrue(preview["network"]["compatible"])
        self.assertTrue(preview["converted_from"].endswith("WTF BMI.tvq"))


if __name__ == "__main__":
    unittest.main()
