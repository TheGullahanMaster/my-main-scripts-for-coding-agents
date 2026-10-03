"""Dataset handling: streamed --max-rows sampling, encoding, and compact checkpoints.

Run with: python -B -m unittest -v test_afpo_data
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import afpo as a
import afpo_gui as gui
from afpo_lib.checkpoint_format import from_json_value, to_json_value


def write_csv(directory, rows=1234, late_text=False):
    """A CSV with quoting, embedded newlines, missing values and duplicate headers."""
    rng = np.random.default_rng(1)
    lines = ['a,b,a,txt,flag,"q,uote",mixed' + (",late" if late_text else "")]
    for i in range(rows):
        txt = rng.choice(['x', '"y, z"', '"multi\nline"', 'NA', '', ' sp '])
        line = f'{i},{rng.normal():.17g},{i % 7},{txt},{rng.choice(["True", "False"])},{i % 3},{"007" if i % 5 else "abc"}'
        lines.append(line + ("," + ("7" if i < rows - 100 else "zz") if late_text else ""))
    path = Path(directory) / "data.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


class ReadDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(a, "CSV_CHUNK_ROWS", 100); patcher.start(); self.addCleanup(patcher.stop)

    def test_full_read_matches_python_engine(self):
        path = write_csv(self.tmp.name)
        pd.testing.assert_frame_equal(a.read_dataset(path, ","), pd.read_csv(path, sep=",", engine="python"))
        self.assertNotIn("afpo_row_sample", a.read_dataset(path, ",", 5000).attrs)

    def test_sample_is_uniform_seeded_and_in_file_order(self):
        path = write_csv(self.tmp.name)
        full = pd.read_csv(path)
        sample = a.read_dataset(path, ",", 300, seed=3)
        info = sample.attrs["afpo_row_sample"]
        self.assertEqual((info["source_rows"], info["max_rows"], len(sample)), (1234, 300, 300))
        self.assertEqual(info["rows"], sorted(set(info["rows"])))
        pd.testing.assert_frame_equal(sample, full.iloc[info["rows"]].reset_index(drop=True))
        self.assertEqual(a.read_dataset(path, ",", 300, seed=3).attrs["afpo_row_sample"]["rows"], info["rows"])
        self.assertNotEqual(a.read_dataset(path, ",", 300, seed=4).attrs["afpo_row_sample"]["rows"], info["rows"])
        self.assertLess(abs(np.mean(info["rows"]) - 617) / 617, .15)

    def test_column_turning_text_late_is_parsed_from_text(self):
        path = write_csv(self.tmp.name, late_text=True)
        sample = a.read_dataset(path, ",", 300, seed=3)
        full = pd.read_csv(path)
        pd.testing.assert_frame_equal(sample, full.iloc[sample.attrs["afpo_row_sample"]["rows"]].reset_index(drop=True),
                                      check_dtype=False)
        self.assertFalse(pd.api.types.is_numeric_dtype(sample["late"]))

    def test_csv_shape_and_other_delimiters(self):
        path = write_csv(self.tmp.name)
        rows, columns = a.csv_shape(path, ",")
        self.assertEqual((rows, columns), (1234, list(pd.read_csv(path).columns)))
        for sep in (";", " ", "\t", "::"):
            other = Path(self.tmp.name) / "other.csv"
            pd.read_csv(path).drop(columns=["txt"]).to_csv(other, sep=sep[0], index=False)
            if len(sep) > 1: other.write_text(other.read_text().replace(sep[0], sep))
            pd.testing.assert_frame_equal(a.read_dataset(other, sep), pd.read_csv(other, sep=sep, engine="python"))

    def test_cli_validates_max_rows(self):
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            a.parse_cli(["--max-rows", "-1"])
        self.assertEqual(a.parse_cli(["--max-rows", "500"])[1].max_rows, 500)


class EncodeTests(unittest.TestCase):
    def test_one_hot_and_class_codes(self):
        df = pd.DataFrame({"x": [1., np.nan, 3., 4.], "c": ["p", None, "q", "p"], "y": [1., 2., 3., 4.], "k": ["u", "v", "u", "w"]})
        X, Y, names, outputs, cats, maps = a.encode(df, [1, 2, 5, 6])
        self.assertEqual(names, ["x", "c=__MISSING__", "c=p", "c=q"])
        np.testing.assert_array_equal(X, [[1, 0, 1, 0], [3, 1, 0, 0], [3, 0, 0, 1], [4, 0, 1, 0]])
        self.assertTrue(X.flags.c_contiguous)
        np.testing.assert_array_equal(Y[:, 1], [0, 1, 0, 2])
        Xv, Yv, *_ = a.encode(pd.DataFrame({"x": [2.], "c": ["new"], "y": [0.], "k": ["unseen"]}), [1, 2, 5, 6], maps)
        np.testing.assert_array_equal(Xv, [[2, 0, 0, 0]]); self.assertEqual(Yv[0, 1], -1)

    def test_blocked_nearest_rows_match_all_pairs(self):
        rng = np.random.default_rng(0)
        X = np.round(rng.normal(size=(600, 3)), 1); X[5] = X[6]
        rows = rng.choice(600, 64, replace=False)
        distance = np.sum((X[rows, None, :] - X[None, :, :]) ** 2, axis=2); distance[distance <= 1e-18] = np.inf
        expected = np.argmin(distance, axis=1)
        for block in (1 << 25, 4096, 8):
            with patch.object(a, "NEAREST_ROW_BLOCK_BYTES", block):
                partner, usable = a.nearest_other_rows(X, rows)
            np.testing.assert_array_equal(partner, expected)
            np.testing.assert_array_equal(usable, np.isfinite(distance[np.arange(64), expected]))

    def test_row_cache_keys_are_compact(self):
        rows = np.arange(100_000)
        self.assertEqual(len(a.rows_digest(rows)), 16)
        self.assertEqual(a.rows_digest(rows), a.rows_digest(rows.astype(np.int32)))
        self.assertNotEqual(a.rows_digest(rows), a.rows_digest(rows[::-1]))


class CheckpointArrayTests(unittest.TestCase):
    def test_numeric_arrays_round_trip_bit_exact(self):
        for array in (np.array([1.5, np.nan, np.inf, -np.inf, -0.]), np.zeros((0, 3)), np.arange(12, dtype=np.int32).reshape(3, 4)[:, ::2],
                      np.array([True, False]), np.arange(4, dtype=">f8")):
            encoded = json.loads(json.dumps(to_json_value(array)))
            self.assertEqual(encoded["__afpo_type__"], "ndarray_zlib")
            restored = from_json_value(encoded)
            self.assertEqual((restored.dtype, restored.shape), (array.dtype, array.shape))
            self.assertEqual(restored.tobytes(), np.ascontiguousarray(array).tobytes())
            self.assertTrue(restored.flags.writeable)

    def test_corrupt_binary_arrays_are_rejected(self):
        bad = to_json_value(np.arange(4.)); bad["shape"] = [5]
        with self.assertRaises(ValueError): from_json_value(bad)
        bad = to_json_value(np.arange(4.)); bad["dtype"] = "|O"
        with self.assertRaises(ValueError): from_json_value(bad)
        self.assertEqual(to_json_value(np.array(["a"]))["__afpo_type__"], "ndarray")

    def test_training_arrays_are_saved_once(self):
        X = np.arange(6.).reshape(3, 2); Y = X[:, :1].copy()
        state = {"X": X, "Y": Y, "Xt": X, "Yt": Y, "Xv": None, "Yv": None}
        saved = a._checkpoint_state_for_save(state)
        self.assertNotIn("X", saved); self.assertIn("X", state)
        restored = a.checkpoint_arrays(saved)
        self.assertIs(restored[0], X); self.assertIs(restored[1], Y)
        state["X"] = X.copy()
        self.assertIs(a._checkpoint_state_for_save(state), state)


class SampledRunTests(unittest.TestCase):
    def test_max_rows_run_records_sample_and_resumes(self):
        rng = np.random.default_rng(0); x = rng.uniform(1, 3, (400, 2))
        frame = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "y": x[:, 0] ** 2 + 2 * x[:, 1]})
        with tempfile.TemporaryDirectory(prefix="afpo-data-") as directory, contextlib.chdir(directory):
            frame.to_csv("d.csv", index=False)
            args = a.parse_cli(["--population", "32", "--max-generations", "2", "--seed", "4", "--workers", "1", "--max-rows", "120"])[1]
            df = a.read_dataset("d.csv", ",", args.max_rows, a.row_sample_seed(args))
            setup = {"path": Path("d.csv"), "df": df, "types": [1, 1, 5], "delimiter": ",", "ops": ["+", "*"], "affine_on": True,
                     "coev": False, "dynamic_pressure_on": False, "adf_enabled": False, "nodes": 7, "depth": 3, "island_count": 1,
                     "migration_interval": 0, "migrants_per_island": 0, "val_path": "", "validation_percent": 20, "metadata": {}}
            with contextlib.redirect_stdout(io.StringIO()):
                result = a.train_from_setup(args, setup, choose_model=lambda *_: 0)
            self.assertNotIn("df", setup)
            manifest = json.loads(Path(result["manifest"]).read_text())
            sample = manifest["configuration"]["row_sample"]
            self.assertEqual((manifest["dataset"]["rows"], sample["source_rows"], len(sample["rows"])), (120, 400, 120))
            self.assertEqual(len(manifest["split"]["train_indices"]) + len(manifest["split"]["validation_indices"]), 120)
            wrapper = json.loads(Path(result["checkpoint"]).read_text())
            self.assertEqual(wrapper["format_version"], a.SAFE_CHECKPOINT_FORMAT)
            _, _, _, _, state = a.load_checkpoint(result["checkpoint"], False)
            self.assertNotIn("X", state)
            self.assertEqual(len(state["Xt"]) + len(state["Xv"]), 120)
            resume = a.parse_cli(["--resume", result["checkpoint"], "--max-generations", "3", "--workers", "1"])[1]
            with contextlib.redirect_stdout(io.StringIO()) as log:
                a.resume_main(resume)
            self.assertIn("Resume complete at generation 3", log.getvalue())

    def test_gui_inspects_a_sample_and_reports_training_rows(self):
        with tempfile.TemporaryDirectory(prefix="afpo-data-") as directory:
            path = Path(directory) / "d.csv"
            x = np.random.default_rng(0).uniform(-2, 2, (300, 3))
            pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "c": x[:, 2], "y": x.sum(axis=1)}).to_csv(path, index=False)
            with patch.object(gui, "INSPECT_ROWS", 50):
                info = gui.inspect_dataset(str(path), ",")
            self.assertEqual((info["rows"], info["sampled_rows"]), (300, 50))
            self.assertIsNone(gui.inspect_dataset(str(path), ",")["sampled_rows"])
            request = {"mode": "train", "data": {"path": str(path), "delimiter": ",", "types": [1, 1, 1, 5], "validation_mode": "percent", "validation_percent": 20},
                       "operators": {"groups": ["1"]}, "search": {"nodes": 15, "depth": 4, "islands": 1},
                       "run": {"max_generations": 1, "population": 40, "seed": 3, "workers": 1}, "advanced": {"max_rows": 100}}
            checked = gui.check_form(request)
            self.assertEqual((checked["rows"], checked["training_rows"]), (300, 100))
            self.assertIn("--max-rows", checked["argv"])


if __name__ == "__main__":
    unittest.main()
