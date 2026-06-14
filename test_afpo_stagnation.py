"""Regression test for the AFPO cross-chunk stagnation counter.

Bug (fixed): ``evolve_afpo`` keyed ``local_stag`` off ``hof.update()``, but every
AFPO worker (``evolve_afpo_stage_worker`` / ``evolve_afpo_island_chunk``)
recreates the Hall of Fame EMPTY for each chunk.  An empty/sparse HoF is
trivially "improved" by the first child (and by every child that opens a new
complexity bucket), so ``local_stag`` was reset to 0 at the start of essentially
every chunk.  That pinned the returned stagnation counter near 0 for the whole
run and silently disabled ALL stagnation-driven machinery shared by every AFPO
mode: directed extinction (needs stag > ext_patience), the stagnation immigrant
pulse (> ext_patience//3), complexity-pressure relaxation, escape-tolerance's
stagnation path (stag_frac > 0.3), the orchestrator's chunk-size halving
(> ext_patience//2) and staged-AFPO's stagnation-scaled graduation volume.

The fix tracks the best PERFORMANCE seen (primed from the incoming population),
so the counter accumulates across chunks as the thresholds above assume.

This test reproduces the real worker pattern (a fresh HoF per chunk) on a target
that cannot be improved — a constant, which every individual fits exactly via the
affine wrapper — so a correct counter MUST climb well past a single chunk.
"""
import os, random
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import numpy as np
import evo13 as e


def test_stag_accumulates_across_chunks():
    np.random.seed(0); random.seed(0)
    e.set_ops_mode(True)
    e.AFFINE_SCALING_ENABLED = True   # lets any tree fit a constant target via a/b
    e._INIT_PHASE = False
    nf, feat = 2, ["x0", "x1"]
    rng = np.random.RandomState(0)
    X = rng.uniform(-2.0, 2.0, size=(120, 2))
    y = np.full(X.shape[0], 3.0)      # constant target → optimal loss ~0 for all

    pop = [e.Individual(e.random_cgp(nf, e.CGP_NODES, feat)) for _ in range(40)]
    for ind in pop:
        ind.calculate_fitness(X, y, 5)
    assert min(ind.loss for ind in pop) < 1e-6, \
        "constant target should be fit ~perfectly by the affine wrapper"

    CHUNK = 30
    ext = 10_000                      # high so directed extinction never resets stag
    stag = 0
    returned = []
    for _ in range(4):
        lh = e.HallOfFame(out_type=5)  # FRESH empty HoF per chunk == real worker
        pop, stag = e.evolve_afpo(pop, X, y, 5, nf, feat, target_size=40,
                                  n_generations=CHUNK, hof=lh,
                                  stag_counter=stag, ext_patience=ext)
        returned.append(stag)

    # Under the bug the empty-per-chunk HoF reset local_stag to ~0 every chunk,
    # so the returned counter could never exceed a fraction of one chunk.  With
    # the fix it accumulates (no improvement is possible on a constant target),
    # climbing well past a single chunk's length.
    assert max(returned) > CHUNK, \
        f"stagnation counter did not accumulate across chunks: returned={returned}"
    assert returned[-1] >= returned[0], \
        ("stagnation counter should be non-decreasing on an unimprovable target: "
         f"returned={returned}")
    print(f"  returned stag per chunk = {returned} (chunk={CHUNK})")
    print("PASS test_stag_accumulates_across_chunks")


def test_classification_stag_path_runs():
    """The fix added a type_code==6 (accuracy) branch; make sure it runs."""
    np.random.seed(1); random.seed(1)
    e.set_ops_mode(True); e._INIT_PHASE = False
    nf, feat = 3, ["a", "b", "c"]
    rng = np.random.RandomState(1)
    X = rng.uniform(-2.0, 2.0, size=(200, 3))
    y = (X[:, 0] * X[:, 1] - 0.5 * X[:, 2] > 0).astype(np.float64)
    pop = [e.Individual(e.random_cgp(nf, e.CGP_NODES, feat)) for _ in range(40)]
    for ind in pop:
        ind.calculate_fitness(X, y, 6)
    stag = 0
    for _ in range(3):
        lh = e.HallOfFame(out_type=6)
        pop, stag = e.evolve_afpo(pop, X, y, 6, nf, feat, target_size=40,
                                  n_generations=40, hof=lh,
                                  stag_counter=stag, ext_patience=120)
    assert len(set(id(i) for i in pop)) == len(pop), "no aliased individuals"
    assert all(getattr(i, "age", 0) >= 0 for i in pop)
    print(f"  classification run ok: returned stag={stag}")
    print("PASS test_classification_stag_path_runs")


if __name__ == "__main__":
    test_stag_accumulates_across_chunks()
    test_classification_stag_path_runs()
    print("\nALL AFPO STAGNATION TESTS PASSED")
