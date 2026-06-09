"""Robustness tests for the predator–prey co-evolution hardening in evo13.

These target the two failure modes reported against the first implementation:

  PROBLEM 1 — once the hosts are near-perfect (especially with label noise) the
              predators latch onto the noise and throw a good model off kilter.
  PROBLEM 2 — the whole parasite population converges onto a single subset, so
              the hosts end up training on only `subset` rows and ignore the
              rest of the data.

and the three mechanisms added to fix them:

  • robust, *absolute* per-row hardness (kills noise-domination at the source)
  • a persistence-based NOISE GUARD (disengages from unlearnable / noisy rows)
  • competitive FITNESS SHARING (anti subset-collapse)
  • a uniform-random ANCHOR fraction in every batch (coverage + drift bound)

Run:  python test_coevo_robustness.py
"""
import numpy as np

import evo13


# --------------------------------------------------------------------------
# minimal fakes (a champion whose residual pattern we control per row)
# --------------------------------------------------------------------------
class _FakeTree:
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


def _champion_from_residual(y, resid):
    """A champion whose prediction is y+resid, looked up by integer row id in
    column 0 of X (so _row_hardness sees exactly the residual pattern we set)."""
    preds = (y + resid).astype(np.float64)
    def fn(X):
        ids = X[:, 0].astype(int)
        return preds[ids]
    return _FakeHoF(_FakeChampion(fn))


def _X_with_ids(n):
    X = np.zeros((n, 1), dtype=np.float64)
    X[:, 0] = np.arange(n)
    return X


def ok(name):
    print(f"  ✓ {name}")


# --------------------------------------------------------------------------
# 1.  Robust, absolute hardness  (root-cause fix for PROBLEM 1)
# --------------------------------------------------------------------------
def test_robust_hardness_bounded():
    print("1: robust per-row hardness (bounded; one outlier can't crush the rest)")
    n = 100
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, size=n) * 5.0      # target with a real spread
    resid = np.zeros(n)
    resid[0] = 1e6                                # one catastrophic noise row
    resid[1:6] = 0.5 * np.std(y)                  # a few genuinely-moderate rows
    co = evo13._PredatorPreyCoevolution(n_rows=n, subset=10, pop_size=8,
                                        mut_rate=0.3, virulence=1.0, seed=0)
    h = co._row_hardness(_X_with_ids(n),
                         y.reshape(-1, 1),
                         [_champion_from_residual(y, resid)],
                         out_types=[0])

    assert h is not None
    assert np.all(h >= 0.0) and np.all(h < 1.0), "hardness must live in [0,1)"
    ok("hardness is bounded in [0,1) for every row")

    assert h[0] > 0.99, f"the catastrophic-noise row should saturate near 1 (got {h[0]})"
    ok(f"the 1e6-residual outlier saturates (h[0]={h[0]:.4f}) instead of exploding")

    # The decisive contrast vs the old max-normalisation: the moderate rows keep
    # a real, non-vanishing signal even though a 1e6 outlier is present.  Under
    # err/max they would be ~ (0.5σ)² / (1e6)² ≈ 1e-13 — indistinguishable from 0.
    assert np.all(h[1:6] > 0.1), \
        f"moderate rows were crushed by the outlier: {h[1:6]}"
    assert np.all(h[6:] < 1e-9), "perfectly-fit rows should read ~0"
    ok(f"moderate rows survive the outlier (h[1:6]≈{np.mean(h[1:6]):.3f}, not ~0)")
    print()


def test_near_perfect_disengages():
    print("2: a near-perfect model yields a near-zero signal (disengagement)")
    n = 200
    rng = np.random.default_rng(1)
    y = rng.normal(0.0, 1.0, size=n) * 3.0
    X = _X_with_ids(n)
    co = evo13._PredatorPreyCoevolution(n_rows=n, subset=20, pop_size=16,
                                        mut_rate=0.3, virulence=1.0, seed=1)

    # near-perfect champion: residual is a tiny fraction of the target spread
    tiny = co._row_hardness(X, y.reshape(-1, 1),
                            [_champion_from_residual(y, 0.01 * np.std(y) * np.ones(n))],
                            out_types=[0])
    # badly-fit champion: residual on the order of the target spread
    bad = co._row_hardness(X, y.reshape(-1, 1),
                           [_champion_from_residual(y, 1.0 * np.std(y) * np.ones(n))],
                           out_types=[0])

    assert float(np.mean(tiny)) < 0.01, \
        f"near-perfect model should give ~0 mean hardness (got {np.mean(tiny):.4f})"
    assert float(np.mean(bad)) > 0.3, \
        f"badly-fit model should give a clearly-positive signal (got {np.mean(bad):.4f})"
    ok(f"mean hardness is absolute: near-perfect={np.mean(tiny):.4f} ≪ bad={np.mean(bad):.3f}")

    # Reduced-virulence reward tracks the absolute signal → predators get almost
    # nothing to chase on a good model (this is what stops noise-latching).
    best_tiny = max(co._score(p, tiny) for p in co.parasites)
    best_bad  = max(co._score(p, bad)  for p in co.parasites)
    assert best_tiny < 0.05 < best_bad, \
        f"predator reward should collapse on a good model ({best_tiny=:.4f}, {best_bad=:.4f})"
    ok(f"predator reward collapses when the model is good ({best_tiny:.4f} vs {best_bad:.3f})")
    print()


# --------------------------------------------------------------------------
# 3.  Persistence-based NOISE GUARD  (direct fix for PROBLEM 1)
# --------------------------------------------------------------------------
def test_noise_guard_unit():
    print("3: noise guard discounts persistently-hard rows, spares fresh ones")
    n = 50
    co = evo13._PredatorPreyCoevolution(n_rows=n, subset=10, pop_size=8,
                                        mut_rate=0.3, virulence=1.0, seed=2,
                                        noise_guard=True, guard_patience=5.0,
                                        guard_strength=0.9, guard_floor=0.05)
    hardness = np.zeros(n)
    hardness[0:5] = 0.9          # persistently-hard (trained a lot) → suspected noise
    hardness[5:10] = 0.9         # just-as-hard but FRESH (never trained) → keep
    co.hard_ema = hardness.copy()
    co.seen_count[0:5] = 50.0    # rows 0..4 have been force-fed many times
    # rows 5..9 keep seen_count == 0 (fresh)

    sel = co._selection_hardness(hardness)
    assert np.all(sel[5:10] > 0.89), f"fresh hard rows must be spared: {sel[5:10]}"
    assert np.all(sel[0:5] < 0.4), f"persistent rows must be discounted: {sel[0:5]}"
    assert sel[0:5].max() < sel[5:10].min(), "guard failed to separate noise from fresh"
    ok(f"persistent rows discounted ({sel[0:5].mean():.3f}) vs fresh kept ({sel[5:10].mean():.3f})")

    # floor: even a maximally-resistant row is never zeroed out
    co2 = evo13._PredatorPreyCoevolution(n_rows=n, subset=10, pop_size=8,
                                         mut_rate=0.3, virulence=1.0, seed=2,
                                         noise_guard=True, guard_patience=1.0,
                                         guard_strength=0.99, guard_floor=0.05)
    co2.hard_ema = np.ones(n)
    co2.seen_count[:] = 1e6
    sel2 = co2._selection_hardness(np.ones(n))
    assert np.all(sel2 >= 0.05 - 1e-9), f"guard floor violated: min={sel2.min()}"
    ok(f"guard never discounts below the floor (min sel={sel2.min():.3f})")

    # guard OFF reproduces the raw hardness exactly
    co.noise_guard = False
    assert np.allclose(co._selection_hardness(hardness), hardness)
    ok("guard OFF → selection hardness == raw hardness")
    print()


def test_noise_guard_disengages_over_time():
    print("4: with the guard ON the predator reward DECAYS on unlearnable noise")
    n, subset = 200, 20
    rng = np.random.default_rng(3)
    y = rng.normal(0.0, 1.0, size=n)
    noise_rows = np.arange(25)                       # 25 permanently-noisy rows
    resid = np.zeros(n)
    resid[noise_rows] = 1.5 * np.std(y)              # hard but bounded; never improves
    X = _X_with_ids(n)
    Y = y.reshape(-1, 1)
    champ = _champion_from_residual(y, resid)        # FIXED — models can't fix noise

    def run(noise_guard):
        co = evo13._PredatorPreyCoevolution(
            n_rows=n, subset=subset, pop_size=24, mut_rate=0.3, virulence=1.0,
            seed=7, anchor_frac=0.0, fitness_sharing=False, noise_guard=noise_guard,
            guard_patience=3.0, guard_strength=0.9)
        rewards = []
        for _ in range(80):
            # reward the predators would actually receive this chunk
            hardness = co._row_hardness(X, Y, [champ], [0])
            co._update_hardness_ema(hardness)
            sel = co._selection_hardness(hardness)
            rewards.append(max(co._score(p, sel) for p in co.parasites))
            co.next_batch(X, Y, [champ], [0])    # advance population + seen_count
        return np.array(rewards)

    r_on,  r_off  = run(noise_guard=True), run(noise_guard=False)
    # Measure from the PEAK (both runs first have to *find* the noise rows before
    # anything can disengage), then look at the settled tail.
    peak_on,  late_on  = float(r_on.max()),  float(r_on[-10:].mean())
    peak_off, late_off = float(r_off.max()), float(r_off[-10:].mean())
    print(f"    guard ON : reward peak={peak_on:.3f} → tail={late_on:.3f}")
    print(f"    guard OFF: reward peak={peak_off:.3f} → tail={late_off:.3f}")

    assert late_on < 0.6 * peak_on, \
        f"guard ON should disengage from the noise it found (peak={peak_on:.3f}, tail={late_on:.3f})"
    ok("guard ON: predator reward decays away from the noise peak (disengagement)")
    assert late_off > 0.85 * peak_off, \
        f"guard OFF should stay locked on the noise (peak={peak_off:.3f}, tail={late_off:.3f})"
    assert late_on < 0.5 * late_off, \
        f"guard ON should settle far below guard OFF ({late_on:.3f} vs {late_off:.3f})"
    ok(f"guard OFF stays locked on noise; ON settles far below it ({late_on:.3f} ≪ {late_off:.3f})")
    print()


# --------------------------------------------------------------------------
# 5.  Competitive FITNESS SHARING  (fix for PROBLEM 2)
# --------------------------------------------------------------------------
def _mean_pairwise_overlap(parasites):
    sets = [set(p.tolist()) for p in parasites]
    tot = cnt = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            tot += len(sets[i] & sets[j]); cnt += 1
    return tot / max(cnt, 1)


def test_fitness_sharing_prevents_collapse():
    print("5: fitness sharing stops the population collapsing onto one subset")
    n, subset, pop = 300, 40, 24
    rng = np.random.default_rng(4)
    y = rng.normal(0.0, 1.0, size=n)
    resid = np.zeros(n)
    C = np.arange(0, subset)         # ONE very-hard cluster, exactly `subset` rows:
    resid[C] = 50.0                  # the perfect attractor for population collapse
    X = _X_with_ids(n)
    Y = y.reshape(-1, 1)
    champ = _champion_from_residual(y, resid)

    def run(sharing):
        co = evo13._PredatorPreyCoevolution(
            n_rows=n, subset=subset, pop_size=pop, mut_rate=0.3, virulence=1.0,
            seed=11, anchor_frac=0.0, fitness_sharing=sharing, noise_guard=False)
        active_union = set()
        for _ in range(120):
            active_union.update(co.next_batch(X, Y, [champ], [0]).tolist())
        return (float(co._coverage().max()),
                _mean_pairwise_overlap(co.parasites),
                len(active_union))

    max_cov_off, overlap_off, union_off = run(sharing=False)
    max_cov_on,  overlap_on,  union_on  = run(sharing=True)
    print(f"    sharing OFF: peak row-coverage={max_cov_off:.0f}/{pop}, "
          f"mean pairwise overlap={overlap_off:.1f}/{subset}, "
          f"rows the hosts saw={union_off}")
    print(f"    sharing ON : peak row-coverage={max_cov_on:.0f}/{pop}, "
          f"mean pairwise overlap={overlap_on:.1f}/{subset}, "
          f"rows the hosts saw={union_on}")

    # Without sharing the predators pile onto the single hardest cluster: the
    # hottest row sits in almost every parasite, the parasites are near-identical,
    # and the hosts keep being fed essentially the SAME `subset` rows every chunk
    # (exactly PROBLEM 2).  Sharing flattens coverage, de-correlates the parasites,
    # and so feeds the hosts many more distinct rows over the run.
    assert max_cov_off > 0.7 * pop, \
        f"without sharing the population should pile up (peak coverage={max_cov_off})"
    assert max_cov_on < 0.8 * max_cov_off, \
        f"sharing should flatten peak coverage ({max_cov_off:.0f}→{max_cov_on:.0f})"
    ok(f"sharing flattens peak row-coverage ({max_cov_off:.0f}→{max_cov_on:.0f} of {pop})")

    assert overlap_on < 0.7 * overlap_off, \
        f"sharing should de-correlate the parasites ({overlap_off:.1f}→{overlap_on:.1f})"
    ok(f"sharing de-correlates the predators (overlap {overlap_off:.1f}→{overlap_on:.1f} of {subset})")

    # The headline for PROBLEM 2: how much of the data the hosts actually trained
    # on over the run.  Without sharing it stays a fraction of the dataset; with
    # sharing the de-correlated predators feed the hosts essentially everything.
    assert union_off < 0.6 * n, \
        f"without sharing the hosts should see well under the full dataset (saw {union_off}/{n})"
    assert union_on >= 0.9 * n, \
        f"with sharing the hosts should see nearly the whole dataset (saw {union_on}/{n})"
    ok(f"sharing feeds the hosts far more of the data ({union_off}/{n} → {union_on}/{n} rows)")
    print()


# --------------------------------------------------------------------------
# 6.  Random ANCHOR  (coverage + off-distribution-drift bound)
# --------------------------------------------------------------------------
def test_anchor_coverage_and_bound():
    print("6: the random anchor guarantees coverage and bounds adversarial share")
    n, subset = 400, 40
    rng = np.random.default_rng(5)
    y = rng.normal(0.0, 1.0, size=n)
    hard = np.arange(60)                       # a hard cluster larger than `subset`
    resid = np.zeros(n)
    resid[hard] = 20.0
    X = _X_with_ids(n)
    Y = y.reshape(-1, 1)
    champ = _champion_from_residual(y, resid)

    def run(anchor_frac):
        co = evo13._PredatorPreyCoevolution(
            n_rows=n, subset=subset, pop_size=24, mut_rate=0.3, virulence=1.0,
            seed=21, anchor_frac=anchor_frac, fitness_sharing=False,
            noise_guard=False)
        seen = set()
        for _ in range(80):
            b = co.next_batch(X, Y, [champ], [0])
            assert len(b) == subset and len(set(b.tolist())) == subset, \
                "batch must always be `subset` distinct rows"
            seen.update(b.tolist())
        return len(seen)

    cov0, cov25 = run(anchor_frac=0.0), run(anchor_frac=0.25)
    print(f"    rows the hosts ever trained on:  anchor 0% → {cov0},  anchor 25% → {cov25}")
    ok("active batch is always exactly `subset` distinct rows (size contract held)")
    assert cov25 > cov0, "the anchor should widen dataset coverage over time"
    assert cov25 > 150, \
        f"with a 25% anchor the run should cover far more than `subset` rows (got {cov25})"
    ok(f"anchor covers far more of the dataset over time ({cov0}→{cov25} of {n} rows)")

    # Structural guarantee: regardless of what the predator picks, the anchor
    # injects EXACTLY round(anchor_frac*subset) uniformly-drawn rows that are not
    # among the adversarial selections — this is the hard bound on how far an
    # adversarial batch can pull the hosts off the bulk distribution.
    co = evo13._PredatorPreyCoevolution(n_rows=n, subset=subset, pop_size=8,
                                        mut_rate=0.3, virulence=1.0, seed=21,
                                        anchor_frac=0.25)
    for _ in range(5):
        parasite = co.parasites[0]
        sel = np.zeros(n); sel[parasite] = np.arange(1, parasite.size + 1)  # ranked
        batch = co._assemble_batch(parasite, sel)
        n_anchor = int(round(0.25 * subset))
        adv = parasite[np.argsort(sel[parasite])[::-1][:subset - n_anchor]]
        anchor = set(batch.tolist()) - set(adv.tolist())
        assert len(anchor) == n_anchor, \
            f"anchor must inject exactly {n_anchor} non-adversarial rows (got {len(anchor)})"
        co._evolve_from(np.ones(co.pop_size), sel)   # churn the population a bit
    ok(f"every batch carries exactly {n_anchor} uniformly-drawn anchor rows (drift bound)")
    print()


def test_batch_size_contract_various():
    print("7: batch-size contract across anchor fractions and the degenerate case")
    n = 120
    rng = np.random.default_rng(6)
    y = rng.normal(size=n)
    resid = np.zeros(n)
    resid[:10] = 5.0
    X = _X_with_ids(n)
    Y = y.reshape(-1, 1)
    champ = _champion_from_residual(y, resid)

    for af in (0.0, 0.25, 0.5, 0.9):
        for subset in (8, 32, 64):
            co = evo13._PredatorPreyCoevolution(
                n_rows=n, subset=subset, pop_size=12, mut_rate=0.3, virulence=0.8,
                seed=99, anchor_frac=af)
            for _ in range(6):
                b = co.next_batch(X, Y, [champ], [0])
                assert len(b) == min(subset, n), f"size break: af={af} subset={subset}"
                assert len(set(b.tolist())) == len(b), "duplicate rows in batch"
                assert b.min() >= 0 and b.max() < n, "row index out of range"
    ok("batch is always `subset` distinct in-range rows (anchor∈{0,.25,.5,.9}, subset∈{8,32,64})")

    # subset >= n degenerates safely to full-data evaluation (None) upstream
    evo13.COEVOLUTION_ENABLED = True
    evo13.COEVO_CASE_SUBSET = n + 5
    evo13._COEVO_RUNTIME = None
    assert evo13._select_batch_indices(X, Y, [_FakeHoF(None)], [0], 0) is None
    evo13.COEVOLUTION_ENABLED = False
    evo13._COEVO_RUNTIME = None
    ok("subset ≥ dataset rows still degenerates to full-dataset evaluation")
    print()

class _MetricSpyIndividual:
    def __init__(self):
        self.complexity = 1
        self.loss = 0.0
        self.r2 = 1.0
        self.affine_fitted = True
        self.rows_seen = None

    def calculate_fitness(self, X, y, out_type, update_affine=True, target_grads=None):
        self.rows_seen = X.shape[0]
        self.loss = float(X.shape[0])
        self.r2 = -float(X.shape[0])
        self.affine_fitted = update_affine
        return self.loss


def test_full_data_hof_candidate_refits_subset_metrics():
    print("8: HoF admission re-scores subset candidates on full data")
    X = np.zeros((11, 2))
    Y = np.zeros((11, 1))
    subset_scored = _MetricSpyIndividual()
    subset_scored.loss = 0.0
    subset_scored.r2 = 1.0
    subset_scored.affine_fitted = True

    admitted = evo13._full_data_hof_candidate(subset_scored, X, Y[:, 0], 0)

    assert admitted is not subset_scored, "admission scoring must not mutate worker/local HoF objects"
    assert admitted.rows_seen == 11, "candidate should be scored against every training row"
    assert admitted.loss == 11.0 and admitted.r2 == -11.0, "stored metrics must be full-data metrics"
    assert subset_scored.loss == 0.0 and subset_scored.r2 == 1.0, "subset metrics should stay local"
    ok("global HoF admission uses a full-data, freshly-affine-refit copy")
    print()


if __name__ == "__main__":
    test_robust_hardness_bounded()
    test_near_perfect_disengages()
    test_noise_guard_unit()
    test_noise_guard_disengages_over_time()
    test_fitness_sharing_prevents_collapse()
    test_anchor_coverage_and_bound()
    test_batch_size_contract_various()
    test_full_data_hof_candidate_refits_subset_metrics()
    print("ALL ROBUSTNESS TESTS PASSED ✓")
