"""
Regression tests for the parsimony mechanics in evo13.py.

Guards the `parsimony_fitness` contract and, in particular, that the MDL
parsimony model fixes the linear-parsimony pathology (the complexity term
swamping a small loss so a terrible-but-tiny model outranks a near-perfect-but-
larger one).

Runnable two ways:
    python test_parsimony.py        # prints a short report, exits non-zero on failure
    pytest test_parsimony.py        # standard test discovery
"""
import math
import warnings
warnings.filterwarnings("ignore")

import evo13


def _with_mode(mode, fn, **knobs):
    """Run fn() with PARSIMONY_MODE=mode (and optional MDL knobs), then restore."""
    saved = (evo13.PARSIMONY_MODE, evo13.PARSIMONY_STRENGTH,
             evo13.MDL_DATA_WEIGHT, evo13.MDL_COMPLEXITY_BITS)
    try:
        evo13.PARSIMONY_MODE = mode
        for k, v in knobs.items():
            setattr(evo13, k, v)
        return fn()
    finally:
        (evo13.PARSIMONY_MODE, evo13.PARSIMONY_STRENGTH,
         evo13.MDL_DATA_WEIGHT, evo13.MDL_COMPLEXITY_BITS) = saved


def test_linear_reduces_to_classic_formula():
    """linear mode must equal loss + k*complexity exactly (no freq adj)."""
    def check():
        evo13.PARSIMONY_STRENGTH = 0.01
        for loss, comp in [(0.5, 10.0), (1.8, 3.0), (0.001, 60.0)]:
            expect = loss + 0.01 * comp
            got = evo13.parsimony_fitness(loss, comp, 0.0)
            assert abs(got - expect) < 1e-12, (loss, comp, got, expect)
    _with_mode("linear", check)


def test_mdl_fixes_small_loss_pathology():
    """
    The headline upgrade: a near-perfect but larger model must outrank a
    terrible but tiny one.  Linear parsimony (at a meaningful strength) gets
    this WRONG once the loss is small; MDL gets it RIGHT.
    """
    near_perfect_big = (1e-4, 120.0)   # (loss, complexity)
    terrible_tiny    = (0.8,    6.0)

    # MDL: near-perfect must win (lower fitness).
    def mdl_check():
        f_good = evo13.parsimony_fitness(*near_perfect_big)
        f_bad  = evo13.parsimony_fitness(*terrible_tiny)
        assert f_good < f_bad, f"MDL picked the terrible-tiny model: {f_good} !< {f_bad}"
        return f_good, f_bad
    _with_mode("mdl", mdl_check, MDL_DATA_WEIGHT=2.0, MDL_COMPLEXITY_BITS=0.02)

    # Linear at the production strength exhibits the pathology (terrible wins) —
    # this documents *why* MDL is the upgrade, not a behaviour we want.
    def lin_check():
        evo13.PARSIMONY_STRENGTH = 0.01
        f_good = evo13.parsimony_fitness(*near_perfect_big)
        f_bad  = evo13.parsimony_fitness(*terrible_tiny)
        assert f_bad < f_good, "Linear unexpectedly avoided the pathology"
    _with_mode("linear", lin_check)


def test_mdl_monotonic_in_loss_and_complexity():
    """Lower loss ⇒ better fitness (fixed complexity); lower complexity ⇒
    better fitness (fixed loss).  Both must hold for a sane Occam objective."""
    def check():
        # Fixed complexity, decreasing loss → strictly decreasing fitness.
        prev = None
        for loss in [1.5, 0.8, 0.3, 0.05, 1e-3]:
            f = evo13.parsimony_fitness(loss, 20.0)
            if prev is not None:
                assert f < prev, f"not monotone in loss at loss={loss}"
            prev = f
        # Fixed loss, increasing complexity → non-decreasing fitness.
        prev = None
        for comp in [2.0, 10.0, 40.0, 100.0]:
            f = evo13.parsimony_fitness(0.2, comp)
            if prev is not None:
                assert f >= prev, f"complexity did not cost anything at comp={comp}"
            prev = f
    _with_mode("mdl", check, MDL_DATA_WEIGHT=2.0, MDL_COMPLEXITY_BITS=0.02)


def test_freq_adj_coefficient_is_clamped_nonnegative():
    """A strongly-negative frequency adjustment must not yield a negative
    complexity coefficient (which would reward unbounded growth)."""
    for mode, knobs in [("linear", {}), ("mdl", {"MDL_COMPLEXITY_BITS": 0.02})]:
        def check():
            # With a hugely negative adj the complexity term is clamped to 0, so
            # fitness must be independent of complexity.
            f0 = evo13.parsimony_fitness(0.3, 0.0, -10.0)
            f1 = evo13.parsimony_fitness(0.3, 80.0, -10.0)
            assert abs(f0 - f1) < 1e-12, f"{mode}: complexity coeff not clamped"
        _with_mode(mode, check, **knobs)


def test_mdl_loss_zero_is_finite():
    """A perfect-on-batch fit (loss == 0) must give a finite (large) reward."""
    def check():
        f = evo13.parsimony_fitness(0.0, 10.0)
        assert math.isfinite(f), f
        # And it must beat any positive-loss model of equal complexity.
        assert f < evo13.parsimony_fitness(1e-3, 10.0)
    _with_mode("mdl", check, MDL_DATA_WEIGHT=2.0, MDL_COMPLEXITY_BITS=0.02)


def test_freq_adj_modulates_complexity_cost():
    """Positive freq adj (over-crowded complexity) must raise the cost; negative
    (under-represented) must lower it — in both modes."""
    for mode, knobs in [("linear", {}), ("mdl", {"MDL_COMPLEXITY_BITS": 0.05})]:
        def check():
            base = evo13.parsimony_fitness(0.3, 50.0, 0.0)
            more = evo13.parsimony_fitness(0.3, 50.0, +0.02)
            less = evo13.parsimony_fitness(0.3, 50.0, -0.02)
            assert more > base > less, f"{mode}: freq adj wrong direction"
        _with_mode(mode, check, **knobs)


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    import sys
    failures = 0
    for t in _TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} passed")
    sys.exit(1 if failures else 0)
