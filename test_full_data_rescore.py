"""Regression test for the mini-batch / co-evolution fitness-reporting bug.

Bug: island/AFPO workers receive the mini-batch / co-evolution subset as their
ENTIRE dataset (X_b passed with batch_indices=None), so the affine is fit and the
loss / R² are measured IN-SAMPLE on the subset.  On a small subset the 2-parameter
affine alone fits perfectly, so a model that cannot fit the full target (e.g. one
variable of a three-variable sum) reports loss 0 / R² 1 — and that subset-overfit
estimate was admitted straight into the global HoF, printed as "New Best", and
reported as the final result.

Fix: `_rescore_individual_full_data` re-scores each candidate on the FULL data
(refitting the affine) before it is offered to the global HoF.

Run:  python test_full_data_rescore.py
"""
import numpy as np
import evo13


def ok(name):
    print(f"  ✓ {name}")


# Minimal "output = x0" model (the underparameterised culprit).
def x0_model():
    eq = evo13.CGPEquation(n_features=3, max_nodes=4)
    eq.nodes = [evo13.CGPNode('add', 0, 0) for _ in range(4)]
    eq.out_idx = 0                       # output == feature 0 == x0
    eq.update_active_nodes()
    return evo13.Individual(eq)


def _setup_globals():
    evo13.AFFINE_SCALING_ENABLED   = True
    evo13.PUSH_ENABLED             = False
    evo13.SOBOLEV_ENABLED          = False
    evo13.INSTANCE_REWEIGHT_ENABLED = False
    evo13._DIFFICULTY_ACTIVE       = None
    evo13.DIFFERENTIABLE_BRANCHING = False
    evo13.REGRESSION_LOSS_MODE     = "mse_mae"


def _data(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = X[:, 0] + X[:, 1] + X[:, 2]      # genuinely needs all three variables
    return X, y


def test_bug_is_reproducible():
    print("1: the worker path (subset-as-full) fabricates a perfect score")
    _setup_globals()
    X, y = _data()
    # The exact worker path: a tiny subset handed in AS the whole dataset.
    idx = np.array([10, 20])             # two rows → 2-param affine fits exactly
    ind = x0_model()
    ind.calculate_fitness(X[idx], y[idx], type_code=5,
                          batch_indices=None, use_cache=False)
    assert ind.loss < 1e-6 and ind.r2 > 0.999, \
        f"expected the bug (perfect on a 2-row subset), got loss={ind.loss}, R2={ind.r2}"
    ok(f"x0-only model 'fits' a 2-row subset perfectly (loss={ind.loss:.2e}, R²={ind.r2:.4f})")
    print()


def test_rescore_makes_it_honest():
    print("2: _rescore_individual_full_data restores the true full-data fit")
    _setup_globals()
    X, y = _data()
    idx = np.array([10, 20])
    ind = x0_model()
    ind.calculate_fitness(X[idx], y[idx], type_code=5,
                          batch_indices=None, use_cache=False)
    assert ind.affine_fitted and ind.r2 > 0.999           # subset-overfit state

    evo13._rescore_individual_full_data(ind, X, y, 5, None)
    # x0 explains ~1/3 of var(x0+x1+x2): R² ≈ 0.33, nowhere near perfect.
    assert 0.25 < ind.r2 < 0.45, f"full-data R² should be ~0.33, got {ind.r2}"
    assert ind.loss > 0.5, f"full-data loss should be clearly > 0, got {ind.loss}"
    assert abs(ind.affine_a - 1.0) < 0.1, f"affine refit on full data (a≈1), got {ind.affine_a}"
    ok(f"after full-data rescore: loss={ind.loss:.4f}, R²={ind.r2:.4f} (honest)")
    print()


def test_admission_uses_full_data():
    print("3: the global HoF admission pattern reports honest metrics")
    _setup_globals()
    X, y = _data()

    # Build a 'local HoF' the way a worker would: scored on a small subset.
    local_hof = evo13.HallOfFame()
    idx = np.random.default_rng(1).choice(len(y), size=3, replace=False)
    for _ in range(4):
        ind = x0_model()
        ind.calculate_fitness(X[idx], y[idx], 5, batch_indices=None, use_cache=False)
        local_hof.update(ind)
    subset_best = local_hof.get_best_overall()
    assert subset_best.r2 > 0.9, "local (subset) HoF should look near-perfect (the bug)"

    # ---- merge WITHOUT the fix (legacy): subset metrics leak into the global HoF
    global_legacy = evo13.HallOfFame()
    for c, ind in local_hof.best_by_complexity.items():
        global_legacy.update(ind)
    legacy_best = global_legacy.get_best_overall()

    # ---- merge WITH the fix: re-score on full data before admission
    local_hof2 = evo13.HallOfFame()
    for _ in range(4):
        ind = x0_model()
        ind.calculate_fitness(X[idx], y[idx], 5, batch_indices=None, use_cache=False)
        local_hof2.update(ind)
    global_fixed = evo13.HallOfFame()
    for c, ind in local_hof2.best_by_complexity.items():
        evo13._rescore_individual_full_data(ind, X, y, 5, None)   # the fix
        global_fixed.update(ind)
    fixed_best = global_fixed.get_best_overall()

    print(f"    legacy global HoF best: loss={legacy_best.loss:.4f}, R²={legacy_best.r2:.4f}")
    print(f"    fixed  global HoF best: loss={fixed_best.loss:.4f}, R²={fixed_best.r2:.4f}")
    assert legacy_best.r2 > 0.9, "legacy path leaks the subset-overfit R² (demonstrates the bug)"
    assert fixed_best.r2 < 0.45, f"fixed path must report the true full-data R², got {fixed_best.r2}"
    ok("with the fix the global HoF reports the true full-data fit, not the subset overfit")
    print()


def test_no_batch_is_a_noop():
    print("4: with no batching the rescore is a faithful no-op (full == full)")
    _setup_globals()
    X, y = _data()
    ind = x0_model()
    ind.calculate_fitness(X, y, 5, batch_indices=None, use_cache=False)  # already full
    r2_before, loss_before, a_before = ind.r2, ind.loss, ind.affine_a
    evo13._rescore_individual_full_data(ind, X, y, 5, None)
    assert abs(ind.r2 - r2_before) < 1e-9 and abs(ind.loss - loss_before) < 1e-9
    assert abs(ind.affine_a - a_before) < 1e-9
    ok(f"full-data individual is unchanged by the rescore (R²={ind.r2:.4f})")
    print()


if __name__ == "__main__":
    test_bug_is_reproducible()
    test_rescore_makes_it_honest()
    test_admission_uses_full_data()
    test_no_batch_is_a_noop()
    print("ALL FULL-DATA-RESCORE TESTS PASSED ✓")
