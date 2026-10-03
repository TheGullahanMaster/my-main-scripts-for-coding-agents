"""bench_diag is frozen: its data must match the recorded baselines.

Run with: python -B -m unittest -v test_bench_diag
"""
import json
import unittest
from pathlib import Path

import numpy as np

import bench_diag as b

HERE = Path(__file__).resolve().parent
BASELINES = ("bench_diag_baseline.json", "bench_diag_reserve_baseline.json")


class FrozenBenchDiagTest(unittest.TestCase):
    def test_every_case_matches_its_recorded_data_hash(self):
        recorded = {}
        for name in BASELINES:
            payload = json.loads((HERE / name).read_text())
            self.assertEqual(payload["bench_version"], b.BENCH_VERSION,
                             f"{name} was recorded for another version; re-record it after a version bump")
            recorded.update({case: info["data_sha256"] for case, info in payload["cases"].items()})
        self.assertEqual(set(recorded), set(b.CASES), "every case needs a recorded baseline")
        for case, digest in recorded.items():
            with self.subTest(case=case):
                self.assertEqual(b.data_hash(b.build(case)), digest,
                                 "case data changed: bump BENCH_VERSION and re-record the baselines")

    def test_main_and_reserve_split(self):
        main = json.loads((HERE / BASELINES[0]).read_text())
        reserve = json.loads((HERE / BASELINES[1]).read_text())
        self.assertEqual(list(main["cases"]), b.MAIN_CASES)
        self.assertEqual(list(reserve["cases"]), b.RESERVE_CASES)
        self.assertEqual(main["seeds"], [b.FIRST_SEED + i for i in range(b.SEED_COUNT)])

    def test_data_is_finite_and_branch_straddling_test_rows_are_dropped(self):
        for case in b.CASES:
            X, y, Xt, yt = b.build(case)
            with self.subTest(case=case):
                self.assertTrue(np.isfinite(np.concatenate([X.ravel(), y, Xt.ravel(), yt])).all())
                self.assertEqual(X.shape[1], Xt.shape[1])
        # three_piece jumps at .3 and .7: both straddled midpoints go.
        self.assertEqual(len(b.build("three_piece")[2]), 61)

    def test_digits_score(self):
        self.assertEqual(b.digits(1.), b.DIGITS_CAP)
        self.assertAlmostEqual(b.digits(.999), 3.)
        self.assertEqual(b.digits(-5.), 0.)


if __name__ == "__main__":
    unittest.main()
