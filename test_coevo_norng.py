"""Functional tests for the two evo13 additions:

  1. The ALL-NORNG operator preset (preset "11").
  2. Hillis-style predator–prey competitive co-evolution.

Run:  python test_coevo_norng.py
"""
import builtins
import contextlib
import numpy as np

import evo13


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
@contextlib.contextmanager
def feed_input(answers):
    """Temporarily replace builtins.input with a scripted iterator."""
    it = iter(answers)
    orig = builtins.input
    builtins.input = lambda *a, **k: next(it)
    try:
        yield
    finally:
        builtins.input = orig


class _FakeTree:
    """Minimal stand-in for a CGPEquation: evaluate(X) returns preset preds."""
    def __init__(self, fn):
        self._fn = fn
    def evaluate(self, X):
        return self._fn(X)


class _FakeChampion:
    def __init__(self, fn, a=1.0, b=0.0):
        self.tree = _FakeTree(fn)
        self.affine_a = a
        self.affine_b = b


class _FakeHoF:
    def __init__(self, champion):
        self._c = champion
    def get_best_overall(self):
        return self._c


def ok(name):
    print(f"  ✓ {name}")


# --------------------------------------------------------------------------
# TASK 1 — ALL-NORNG preset
# --------------------------------------------------------------------------
def test_allnorng_preset():
    print("TASK 1: ALL-NORNG preset")

    assert "11" in evo13.OP_PRESETS, "preset 11 missing"
    p = evo13.OP_PRESETS["11"]
    assert "ALL-NORNG" in p["name"], p["name"]
    assert p["ops"] is None, "ALL-NORNG must resolve dynamically (ops=None)"
    ok("OP_PRESETS['11'] present, named ALL-NORNG, resolved dynamically")

    # End-to-end through select_allowed_ops with scripted input.
    # Full preset ("9")
    evo13.PERCEPTRON_ENABLED = False
    with feed_input(["9", "n"]):            # preset 9, ADF -> no
        full = list(evo13.select_allowed_ops())
    # ALL-NORNG preset ("11")
    with feed_input(["11", "n"]):           # preset 11, ADF -> no
        norng = list(evo13.select_allowed_ops())

    assert "python_rng" in full and "perlin_noise" in full, \
        "Full preset should contain the RNG/noise ops to begin with"
    ok("Full preset contains python_rng and perlin_noise")

    assert "python_rng" not in norng, "python_rng leaked into ALL-NORNG"
    assert "perlin_noise" not in norng, "perlin_noise leaked into ALL-NORNG"
    ok("ALL-NORNG omits python_rng and perlin_noise")

    assert norng == [op for op in full if op not in ("python_rng", "perlin_noise")], \
        "ALL-NORNG must equal Full minus exactly the two RNG/noise ops"
    assert set(full) - set(norng) == {"python_rng", "perlin_noise"}, \
        "ALL-NORNG should differ from Full by exactly those two ops"
    ok("ALL-NORNG == Full minus exactly {python_rng, perlin_noise}")

    # CGPEquation op-lists were rebuilt and exclude the RNG/noise ops too.
    assert "python_rng" not in evo13.CGPEquation.OPS_ALL
    assert "perlin_noise" not in evo13.CGPEquation.OPS_ALL
    ok("CGPEquation.OPS_ALL rebuilt without the RNG/noise ops")
    print()


# --------------------------------------------------------------------------
# TASK 2 — competitive co-evolution
# --------------------------------------------------------------------------
def test_reduced_virulence():
    print("TASK 2a: reduced-virulence scoring")
    co = evo13._PredatorPreyCoevolution(n_rows=100, subset=10, pop_size=8,
                                        mut_rate=0.3, virulence=0.75, seed=0)
    # Build a hardness vector so a parasite's mean-hardness == chosen x.
    h = np.zeros(100)
    h[:50] = 1.0   # half hard, half easy → a parasite over k hard rows has x=k/10

    def score_at(x):
        # parasite with x fraction hard rows
        khard = int(round(x * co.subset))
        par = np.concatenate([np.arange(khard),
                              np.arange(50, 50 + (co.subset - khard))]).astype(np.int64)
        return co._score(par, h)

    # λ=0.75 → reward peaks at x=0.75, lower at x=1.0
    s075 = score_at(0.8)      # nearest grid point to λ on a size-10 subset
    s100 = score_at(1.0)
    assert s100 < s075, f"reduced virulence: expected peak below x=1 (got {s075=}, {s100=})"
    ok(f"λ=0.75: score(x≈0.8)={s075:.3f} > score(x=1.0)={s100:.3f}  (disengagement-safe)")

    # λ=1.0 → monotone increasing, max at x=1.0
    co.virulence = 1.0
    assert score_at(1.0) > score_at(0.5) > score_at(0.0), "λ=1 should be monotone in x"
    ok("λ=1.0: score is monotone increasing in host-failure")
    print()


def test_cold_start_and_genome_validity():
    print("TASK 2b: cold start + parasite-genome validity")
    n, subset = 300, 32
    co = evo13._PredatorPreyCoevolution(n_rows=n, subset=subset, pop_size=16,
                                        mut_rate=0.3, virulence=1.0, seed=1)
    X = np.random.RandomState(0).randn(n, 2)
    Y = np.zeros((n, 1))
    hofs = [_FakeHoF(None)]          # no champion yet
    out_types = [0]

    b = co.next_batch(X, Y, hofs, out_types)
    assert b.shape == (subset,), f"cold-start batch wrong size: {b.shape}"
    assert len(set(b.tolist())) == subset, "cold-start batch has duplicate rows"
    assert b.min() >= 0 and b.max() < n, "cold-start batch index out of range"
    ok("cold start (no champion) → valid random subset of the right size")

    # Every parasite in the population is a valid genome.
    for p in co.parasites:
        assert p.shape == (subset,)
        assert len(set(p.tolist())) == subset
        assert p.min() >= 0 and p.max() < n
    ok("all parasite genomes are distinct-index subsets within range")
    print()


def test_arms_race():
    print("TASK 2c: arms race concentrates the batch on the hard rows")
    n, subset = 200, 20
    hard = set(range(20))                       # the only rows the champion fails
    X = np.arange(n, dtype=np.float64).reshape(-1, 1)
    Y = np.zeros((n, 1))

    def champ_fn(Xin):
        rows = Xin[:, 0].astype(int)
        preds = np.zeros(len(rows))
        preds[np.isin(rows, list(hard))] = 10.0  # large residual on hard rows only
        return preds

    hofs = [_FakeHoF(_FakeChampion(champ_fn))]
    out_types = [0]

    # Isolate the *core* arms race: turn the robustness guards off so this test
    # exercises pure predator concentration (the anchor/sharing/noise-guard each
    # have their own dedicated tests in test_coevo_robustness.py).  In particular
    # the noise guard would (correctly) suppress these rows — the champion here is
    # a fixed stub that never improves, so from the guard's view the hard rows are
    # "irreducible" and it would stop chasing them; that behaviour is verified
    # separately.
    co = evo13._PredatorPreyCoevolution(n_rows=n, subset=subset, pop_size=32,
                                        mut_rate=0.3, virulence=1.0, seed=7,
                                        anchor_frac=0.0, fitness_sharing=False,
                                        noise_guard=False)

    first = None
    best = 0
    for chunk in range(60):
        b = co.next_batch(X, Y, hofs, out_types)
        hard_count = len(set(b.tolist()) & hard)
        if first is None:
            first = hard_count
        best = max(best, hard_count)
    final = hard_count

    print(f"    hard-rows-in-active-batch: chunk0={first}/20, final={final}/20, best={best}/20")
    assert best >= 16, f"arms race failed to concentrate on hard rows (best={best}/20)"
    assert final > first, "no improvement over the run"
    ok("predators learned to feed the hosts their hardest rows (arms race works)")
    print()


def test_select_batch_indices():
    print("TASK 2d: _select_batch_indices dispatch")
    n = 150
    X = np.random.RandomState(3).randn(n, 2)
    Y = np.zeros((n, 1))
    hofs = [_FakeHoF(None)]
    out_types = [0]

    # --- co-evolution OFF ---
    evo13.COEVOLUTION_ENABLED = False
    assert evo13._select_batch_indices(X, Y, hofs, out_types, 0) is None, \
        "off + batch_size 0 should be full dataset (None)"
    rb = evo13._select_batch_indices(X, Y, hofs, out_types, 16)
    assert rb is not None and len(rb) == 16, "off + batch_size 16 should give a 16-row batch"
    ok("co-evolution OFF reproduces legacy behaviour (None / random mini-batch)")

    # --- co-evolution ON ---
    evo13.COEVOLUTION_ENABLED = True
    evo13.COEVO_CASE_SUBSET = 24
    evo13.COEVO_POP_SIZE = 16
    evo13.COEVO_MUT_RATE = 0.3
    evo13.COEVO_VIRULENCE = 0.8
    evo13._COEVO_RUNTIME = None
    cb = evo13._select_batch_indices(X, Y, hofs, out_types, 0)
    # The batch is the parasite subset padded up to the evaluation floor
    # (in-sample selection on very few rows cannot resolve host quality).
    expect = max(24, min(evo13.COEVO_MIN_EVAL_ROWS, n))
    assert cb is not None and len(cb) == expect, \
        f"co-evo batch wrong size: {None if cb is None else len(cb)} (expected {expect})"
    assert evo13._COEVO_RUNTIME is not None, "runtime manager should be lazily created"
    ok("co-evolution ON returns the parasite subset padded to the eval floor")

    # --- subset >= n degenerates to full data path ---
    evo13.COEVO_CASE_SUBSET = n + 10
    evo13._COEVO_RUNTIME = None
    deg = evo13._select_batch_indices(X, Y, hofs, out_types, 0)
    assert deg is None, "subset >= n with batch_size 0 should fall through to full dataset"
    ok("subset ≥ dataset rows degenerates safely to full-dataset evaluation")

    # restore defaults
    evo13.COEVOLUTION_ENABLED = False
    evo13._COEVO_RUNTIME = None
    print()


def test_select_coevolution_mode_prompt():
    print("TASK 2e: _select_coevolution_mode interactive wiring")

    # Disable path
    with feed_input(["n"]):
        evo13._select_coevolution_mode(1000)
    assert evo13.COEVOLUTION_ENABLED is False
    assert evo13._COEVO_RUNTIME is None
    ok("answering 'n' leaves co-evolution disabled")

    # Enable path with explicit params
    with feed_input(["y", "40", "20", "0.25", "0.6"]):
        evo13._select_coevolution_mode(1000)
    assert evo13.COEVOLUTION_ENABLED is True
    assert evo13.COEVO_CASE_SUBSET == 40
    assert evo13.COEVO_POP_SIZE == 20
    assert abs(evo13.COEVO_MUT_RATE - 0.25) < 1e-9
    assert abs(evo13.COEVO_VIRULENCE - 0.6) < 1e-9
    ok("answering 'y' + params sets COEVO_* globals correctly")

    # Enable path taking all defaults (blank answers)
    with feed_input(["y", "", "", "", ""]):
        evo13._select_coevolution_mode(50)   # small n → subset default clamps to n//2
    assert evo13.COEVOLUTION_ENABLED is True
    assert evo13.COEVO_CASE_SUBSET <= 50
    ok("blank answers fall back to (clamped) defaults")

    # restore
    evo13.COEVOLUTION_ENABLED = False
    evo13._COEVO_RUNTIME = None
    print()


if __name__ == "__main__":
    test_allnorng_preset()
    test_reduced_virulence()
    test_cold_start_and_genome_validity()
    test_arms_race()
    test_select_batch_indices()
    test_select_coevolution_mode_prompt()
    print("ALL TESTS PASSED ✓")
