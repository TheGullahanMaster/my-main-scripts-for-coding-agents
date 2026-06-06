"""Functional tests for the optional per-example difficulty-weighting feature.

Run: python test_difficulty_weighting.py
Exercises the helpers, the loss/fitness split invariant, the per-generation
weight update, and a short end-to-end evolve_afpo run with the feature on.
"""
import numpy as np
import evo13 as e


def _reset_difficulty():
    e._DIFFICULTY_ACTIVE = None
    e._DIFFICULTY_BY_SIG = {}


def test_active_weights_and_sel_extra():
    _reset_difficulty()
    e.INSTANCE_REWEIGHT_ENABLED = True
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    e._difficulty_begin(y)
    assert e._DIFFICULTY_ACTIVE is not None
    assert np.allclose(e._DIFFICULTY_ACTIVE, 1.0), "weights must start at 1.0"

    # full-data subset
    w = e._difficulty_active_weights(len(y))
    assert w is not None and len(w) == len(y)
    # batch subset
    wb = e._difficulty_active_weights(2, batch_indices=np.array([1, 3]))
    assert wb is not None and len(wb) == 2
    # shape mismatch -> None
    assert e._difficulty_active_weights(99) is None

    # sel_extra: upweighting the high-error rows must raise the weighted mean
    per_row = np.array([0.0, 0.0, 0.0, 0.0, 10.0])  # last row is the hard one
    e._DIFFICULTY_ACTIVE = np.array([1., 1., 1., 1., 5.])
    dw = e._difficulty_active_weights(len(y))
    extra = e._difficulty_sel_extra(per_row, None, dw)
    assert extra > 0.0, f"hard-row upweighting should increase loss, got {extra}"

    # disabled -> None everywhere
    e.INSTANCE_REWEIGHT_ENABLED = False
    assert e._difficulty_active_weights(len(y)) is None
    print("PASS test_active_weights_and_sel_extra")


def test_loss_unchanged_fitness_steered():
    """The HoF-keyed loss must be identical with/without weighting; only the
    selection fitness (via sel_extra) may change."""
    _reset_difficulty()
    e.INSTANCE_REWEIGHT_ENABLED = False
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = X[:, 0] * 2.0 - X[:, 1]
    feat = ["a", "b"]

    ind = e.Individual(e.random_cgp(2, e.CGP_NODES, feat))
    # First eval with feature OFF locks affine and gives the true loss.
    ind.calculate_fitness(X, y, 5)
    loss_true = ind.loss
    assert np.isfinite(loss_true)

    # Now enable, make the weights very non-uniform, re-evaluate (no cache).
    e.INSTANCE_REWEIGHT_ENABLED = True
    e._difficulty_begin(y)
    e._DIFFICULTY_ACTIVE[:] = 1.0
    e._DIFFICULTY_ACTIVE[:5] = e.INSTANCE_REWEIGHT_MAX  # upweight a few rows hard
    ind.calculate_fitness(X, y, 5, use_cache=False)

    assert abs(ind.loss - loss_true) < 1e-9, (
        f"raw loss (HoF key) must be unchanged: {ind.loss} vs {loss_true}")
    assert ind.sel_extra != 0.0, "sel_extra should be non-zero under non-uniform weights"
    # fitness should reflect loss + sel_extra (+ parsimony terms)
    print(f"  loss_true={loss_true:.5f} sel_extra={ind.sel_extra:+.5f} fitness={ind.fitness:.5f}")
    print("PASS test_loss_unchanged_fitness_steered")


def test_update_grows_weights_on_failures():
    _reset_difficulty()
    e.INSTANCE_REWEIGHT_ENABLED = True
    e.INSTANCE_REWEIGHT_INCREMENT = 0.5
    e.INSTANCE_REWEIGHT_MAX = 4.0
    e.INSTANCE_REWEIGHT_UPDATE_EVERY = 1
    e.INSTANCE_REWEIGHT_REG_TOL = 0.25

    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 2))
    y = X[:, 0].copy()
    # Inject two clear outliers the model can't fit.
    y[7] += 50.0
    y[19] -= 50.0
    feat = ["a", "b"]

    pop = [e.Individual(e.random_cgp(2, e.CGP_NODES, feat)) for _ in range(8)]
    for ind in pop:
        ind.calculate_fitness(X, y, 5)
    e._difficulty_begin(y)
    start = e._DIFFICULTY_ACTIVE.copy()
    assert np.allclose(start, 1.0)

    for g in range(20):
        e._difficulty_update(pop, X, y, 5, g)

    w = e._DIFFICULTY_ACTIVE
    assert w.max() <= e.INSTANCE_REWEIGHT_MAX + 1e-9, "weights must respect the cap"
    assert w.min() >= 1.0 - 1e-9, "weights never drop below the initial 1.0"
    assert w.max() > 1.0, "some failed row must have grown above 1.0"
    # The injected outliers are unfittable, so they MUST be persistent failures
    # and reach the cap (an untrained champion fails many rows, so we only
    # assert the outliers are maximally weighted, not uniquely so).
    assert w[7] >= e.INSTANCE_REWEIGHT_MAX - 1e-9, f"outlier row 7 should hit the cap, got {w[7]}"
    assert w[19] >= e.INSTANCE_REWEIGHT_MAX - 1e-9, f"outlier row 19 should hit the cap, got {w[19]}"
    print(f"  max weight={w.max():.2f}  w[7]={w[7]:.2f}  w[19]={w[19]:.2f}")
    print("PASS test_update_grows_weights_on_failures")


def test_end_to_end_evolve_afpo():
    _reset_difficulty()
    e.INSTANCE_REWEIGHT_ENABLED = True
    e.INSTANCE_REWEIGHT_INCREMENT = 0.3
    e.INSTANCE_REWEIGHT_MAX = 6.0
    e.INSTANCE_REWEIGHT_UPDATE_EVERY = 1
    e.INSTANCE_REWEIGHT_REG_TOL = 0.25

    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 2))
    y = X[:, 0] * 1.5 + 0.5
    y[3] += 30.0   # outlier
    feat = ["a", "b"]

    pop = [e.Individual(e.random_cgp(2, e.CGP_NODES, feat)) for _ in range(12)]
    for ind in pop:
        ind.calculate_fitness(X, y, 5)
    hof = e.HallOfFame(out_type=5)
    ret_pop, stag = e.evolve_afpo(
        pop, X, y, 5, 2, feat, target_size=12,
        n_generations=25, hof=hof, stag_counter=0, ext_patience=40)

    assert len(ret_pop) > 0
    best = hof.get_best_overall()
    assert best is not None and np.isfinite(best.loss), "HoF must hold a finite-loss model"
    # Weights should have grown during the run, and the unfittable outlier
    # (row 3) should be at least as hard as the median row — i.e. the search
    # has differentiated it from the easy majority (focus-on-outliers).
    w = e._DIFFICULTY_ACTIVE
    assert w is not None and w.max() > 1.0, "difficulty weights should have grown"
    assert w[3] >= np.median(w) - 1e-9, (
        f"outlier row 3 should be at least as hard as the median; "
        f"w[3]={w[3]:.2f} median={np.median(w):.2f}")
    print(f"  evolve_afpo ok: best loss={best.loss:.5f}  max weight={w.max():.2f}  "
          f"w[3]={w[3]:.2f}  median={np.median(w):.2f}")
    print("PASS test_end_to_end_evolve_afpo")


def test_feature_off_is_noop():
    """With the feature off, get_case_errors / calculate_fitness behave exactly
    as before (sel_extra stays 0, errors unscaled)."""
    _reset_difficulty()
    e.INSTANCE_REWEIGHT_ENABLED = False
    rng = np.random.default_rng(3)
    X = rng.normal(size=(25, 2))
    y = X[:, 1] - 0.3
    feat = ["a", "b"]
    ind = e.Individual(e.random_cgp(2, e.CGP_NODES, feat))
    ind.calculate_fitness(X, y, 5)
    assert ind.sel_extra == 0.0
    errs = ind.get_case_errors(X, y, 5, np.arange(len(y)))
    assert len(errs) == len(y)
    print("PASS test_feature_off_is_noop")


if __name__ == "__main__":
    test_active_weights_and_sel_extra()
    test_loss_unchanged_fitness_steered()
    test_update_grows_weights_on_failures()
    test_feature_off_is_noop()
    test_end_to_end_evolve_afpo()
    print("\nALL DIFFICULTY-WEIGHTING TESTS PASSED")
