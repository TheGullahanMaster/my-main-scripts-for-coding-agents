"""Tests for the batch-RESOLUTION guards added to evo13's predator–prey
co-evolution (failure mode 5: resolution collapse).

The stress regime they target: COEVO_CASE_SUBSET forced to extremes (1 row, or
a few hundred rows on a big/complex dataset).  The workers fit the affine and
score entirely IN-SAMPLE on the chunk's batch, so a too-small batch cannot
resolve a genuinely better host from a lucky one — on ≤2 rows the affine alone
ties every host at ~0 loss — and the search degenerates to a random walk
("loses the resolution required to properly learn").

Mechanisms under test:

  • EVALUATION FLOOR    — the batch handed to the hosts is padded with anchor
    rows to ≥ min_eval_rows; the parasite stays a small adversarial spotlight.
  • RESOLUTION GOVERNOR — the batch→full generalisation gap of the candidates
    re-scored at global-HoF admission grows the batch while chunk winners keep
    failing the full data (subset overfit), shrinking back once honest.
  • STRATIFIED ANCHOR   — anchor rows drawn per class (classification) or per
    target-quantile bin (regression) so small batches stay representative.

Run:  python test_coevo_resolution.py
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
# 1.  evaluation floor — subset=1 still yields a batch selection can use
# --------------------------------------------------------------------------
def test_eval_floor():
    print("1: evaluation floor pads degenerate subsets (the subset=1 stress case)")
    n = 500
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, n) * 3.0
    resid = np.zeros(n); resid[:40] = 2.0 * np.std(y)
    X, Y = _X_with_ids(n), y.reshape(-1, 1)
    champ = _champion_from_residual(y, resid)

    co = evo13._PredatorPreyCoevolution(n_rows=n, subset=1, pop_size=8,
                                        mut_rate=0.3, virulence=0.75, seed=1,
                                        min_eval_rows=16)
    for _ in range(8):
        b = co.next_batch(X, Y, [champ], [0])
        assert len(b) == 16, f"subset=1 must be padded to the 16-row floor, got {len(b)}"
        assert len(set(b.tolist())) == len(b), "duplicate rows in padded batch"
        assert b.min() >= 0 and b.max() < n, "row out of range"
    ok("subset=1 → every batch padded to exactly the 16-row floor (distinct, in range)")

    # The parasites keep steering: their single adversarial row is in the batch.
    sel = co._selection_hardness(co._row_hardness(X, Y, [champ], [0]))
    scores = np.array([co._parasite_score(p, sel, co._coverage()) for p in co.parasites])
    best = co.parasites[int(np.argmax(scores))]
    batch = co._assemble_batch(best, sel)
    assert best[0] in set(batch.tolist()), "the parasite's adversarial row must be in the batch"
    ok("the winning parasite's adversarial row still leads the padded batch")

    # Floor is dataset-capped and legacy knobs reproduce the old contract.
    co_small = evo13._PredatorPreyCoevolution(n_rows=10, subset=2, pop_size=4,
                                              mut_rate=0.3, virulence=0.75, seed=2,
                                              min_eval_rows=64)
    b = co_small.next_batch(_X_with_ids(10), y[:10].reshape(-1, 1),
                            [_champion_from_residual(y[:10], np.zeros(10))], [0])
    assert len(b) == 10, "floor must cap at the dataset size"
    co_legacy = evo13._PredatorPreyCoevolution(n_rows=n, subset=4, pop_size=8,
                                               mut_rate=0.3, virulence=0.75, seed=3,
                                               min_eval_rows=0, res_max_factor=1.0,
                                               stratify=False)
    for _ in range(4):
        assert len(co_legacy.next_batch(X, Y, [champ], [0])) == 4
    ok("floor caps at the dataset; legacy knobs (floor 0, governor ×1) → exact subset")
    print()


# --------------------------------------------------------------------------
# 2.  resolution governor — gap grows the batch, honesty shrinks it
# --------------------------------------------------------------------------
def test_resolution_governor():
    print("2: the governor grows the batch on subset-overfit, shrinks when honest")
    n = 2000
    co = evo13._PredatorPreyCoevolution(n_rows=n, subset=32, pop_size=8,
                                        mut_rate=0.3, virulence=0.75, seed=4,
                                        min_eval_rows=16, res_max_factor=8.0,
                                        res_grow=1.5, res_shrink=0.8,
                                        res_gap_hi=0.35, res_gap_lo=0.15,
                                        res_gap_ema=0.5)
    assert co._effective_rows() == 32 and co.res_factor == 1.0
    ok("governor starts at ×1 (effective rows == subset)")

    # Chunk winners that ace the batch but fail the full data: gap → ~1.
    sizes = []
    for _ in range(12):
        co.observe_generalization(batch_loss=0.01, full_loss=1.0)   # overfit!
        co._update_resolution()
        sizes.append(co._effective_rows())
    assert co.res_factor > 7.0, f"sustained overfit must grow toward the cap ({co.res_factor:.2f})"
    assert sizes[-1] == 32 * 8, f"effective rows should reach subset×cap (got {sizes[-1]})"
    assert all(b >= a for a, b in zip(sizes, sizes[1:])), "growth must be monotone here"
    ok(f"sustained batch→full gap grows the batch 32 → {sizes[-1]} (×{co.res_factor:.1f}, capped)")

    # Honest chunks (batch ≈ full) decay it back toward the configured subset.
    for _ in range(40):
        co.observe_generalization(batch_loss=0.50, full_loss=0.52)
        co._update_resolution()
    assert co.res_factor == 1.0, f"honest batches must decay the factor to 1 ({co.res_factor:.2f})"
    assert co._effective_rows() == 32
    ok("honest batches decay the governor back to ×1 (subset-sized batches again)")

    # Evidence-gating: no observations → no movement either way.
    co.res_factor = 4.0
    for _ in range(10):
        co._update_resolution()
    assert co.res_factor == 4.0, "no observations must mean no governor movement"
    ok("no rescore observations → the governor holds (acts only on evidence)")

    # Proportional setpoint: AT the [lo, hi] midpoint the batch holds exactly —
    # the equilibrium "just big enough that the in-sample winner transfers";
    # in-band gaps above/below it nudge the factor up/down (no dead-band).
    co.res_factor = 2.0
    co.gap_ema    = None
    co.observe_generalization(0.75, 1.0)          # gap 0.25 == midpoint
    assert co._update_resolution() == 2.0, "at the setpoint the batch must hold"
    co.observe_generalization(0.70, 1.0)          # gap 0.30: in-band, above mid
    assert co._update_resolution() > 2.0, "above the setpoint must keep growing"
    ok("proportional control: holds at the transfer setpoint, no dead-band around it")

    # Gap math: scale-free and robust to junk.
    co.observe_generalization(float('nan'), 1.0)
    co.observe_generalization(1.0, float('inf'))
    co.observe_generalization(None, 1.0)
    assert co._gap_chunk_max is None, "non-finite observations must be ignored"
    co.observe_generalization(0.2, 0.1)      # full BETTER than batch → gap 0
    assert co._gap_chunk_max == 0.0
    co.observe_generalization(2e-6, 2.0)     # tiny-batch perfect, full bad → ~1
    assert co._gap_chunk_max > 0.99
    ok("gap is scale-free, clamps at [0,1], ignores non-finite inputs, keeps the max")
    print()


# --------------------------------------------------------------------------
# 2b.  governor anti-wind-up — res_factor never climbs past the point where the
#      batch already fills the dataset, so a cleared gap shrinks it right away
# --------------------------------------------------------------------------
def test_governor_antiwindup():
    print("2b: the governor doesn't wind res_factor up past full-data saturation")
    # res_max_factor (32) × base (32) = 1024 OVERSHOOTS the 600-row dataset, so
    # the batch saturates at full data once res_factor reaches 600/32 = 18.75.
    n = 600
    co = evo13._PredatorPreyCoevolution(n_rows=n, subset=32, pop_size=8,
                                        mut_rate=0.3, virulence=0.75, seed=4,
                                        min_eval_rows=32, res_max_factor=32.0,
                                        res_grow=1.5, res_shrink=0.8,
                                        res_gap_ema=0.5)
    sat = n / 32.0
    for _ in range(60):                         # sustained overfit → grow
        co.observe_generalization(batch_loss=0.01, full_loss=1.0)
        co._update_resolution()
    assert co._effective_rows() == n, "a big sustained gap must reach full data"
    assert co.res_factor <= sat + 1e-9, \
        f"res_factor must cap at the saturation factor {sat:.2f}, not wind past it " \
        f"toward res_max_factor (got {co.res_factor:.2f})"
    ok(f"res_factor capped at saturation ×{co.res_factor:.2f} (not the configured ×32)")

    # Gap clears: because the factor never wound up into the dead range above
    # saturation, the batch drops BELOW full data within a couple of chunks.
    dwell = 0
    for _ in range(15):
        co.observe_generalization(batch_loss=0.50, full_loss=0.50)   # gap 0
        co._update_resolution()
        if co._effective_rows() >= n:
            dwell += 1
        else:
            break
    assert dwell <= 2, \
        f"a cleared gap must shrink the batch within ~1 chunk, not after a long " \
        f"wind-up unwind (stuck at full data for {dwell} chunks)"
    ok(f"cleared gap un-pins the batch from full data in {dwell} chunk(s) (no wind-up lag)")

    # The cap is a strict no-op when res_max_factor×base already fits the dataset
    # (res_max_factor 8 × base 32 = 256 < 2000): the governor still reaches ×8.
    co2 = evo13._PredatorPreyCoevolution(n_rows=2000, subset=32, pop_size=8,
                                         mut_rate=0.3, virulence=0.75, seed=4,
                                         min_eval_rows=16, res_max_factor=8.0,
                                         res_grow=1.5, res_shrink=0.8,
                                         res_gap_ema=0.5)
    for _ in range(12):
        co2.observe_generalization(batch_loss=0.01, full_loss=1.0)
        co2._update_resolution()
    assert co2.res_factor > 7.0 and co2._effective_rows() == 256, \
        "configs that don't overshoot the dataset must be unaffected by the cap"
    ok("no-overshoot configs unchanged (cap inactive when the batch can't fill the data)")
    print()


# --------------------------------------------------------------------------
# 3.  stratified anchor — classification: minority class always represented
# --------------------------------------------------------------------------
def test_stratified_anchor_classification():
    print("3: stratified anchor keeps the minority class in every small batch")
    n = 2000
    rng = np.random.default_rng(5)
    y = np.zeros(n); y[:120] = 1.0                  # 6% minority
    rng.shuffle(y)
    X, Y = _X_with_ids(n), y.reshape(-1, 1)
    logits = np.where(y > 0.5, 0.2, -0.2)           # mediocre champion logits
    champ = _FakeHoF(_FakeChampion(lambda Xq, L=logits: L[Xq[:, 0].astype(int)]))

    def minority_hits(stratify, trials=40, subset=16):
        co = evo13._PredatorPreyCoevolution(n_rows=n, subset=subset, pop_size=8,
                                            mut_rate=0.3, virulence=0.75, seed=6,
                                            min_eval_rows=16, stratify=stratify)
        hits = 0
        for _ in range(trials):
            b = co.next_batch(X, Y, [champ], [6])
            if np.any(y[b] > 0.5):
                hits += 1
        return hits

    hits_on  = minority_hits(True)
    hits_off = minority_hits(False)
    print(f"    batches containing the 6% minority class: stratified {hits_on}/40, "
          f"uniform {hits_off}/40")
    assert hits_on == 40, "stratified anchor must carry the minority class in EVERY batch"
    assert hits_off < 40, "uniform anchor should miss the minority class sometimes (6% × 16 rows)"
    ok("stratified anchor: minority class present in 40/40 small batches (uniform misses)")
    print()


# --------------------------------------------------------------------------
# 4.  stratified anchor — regression: batches span the target's spread
# --------------------------------------------------------------------------
def test_stratified_anchor_regression():
    print("4: stratified anchor spans the regression target's quantile bins")
    n = 4000
    rng = np.random.default_rng(7)
    y = rng.normal(0.0, 1.0, n) * 4.0
    X, Y = _X_with_ids(n), y.reshape(-1, 1)
    champ = _champion_from_residual(y, np.full(n, 0.5 * np.std(y)))
    edges = np.quantile(y, np.linspace(0.0, 1.0, 9)[1:-1])

    co = evo13._PredatorPreyCoevolution(n_rows=n, subset=24, pop_size=8,
                                        mut_rate=0.3, virulence=0.75, seed=8,
                                        min_eval_rows=24, stratify=True)
    for _ in range(15):
        b = co.next_batch(X, Y, [champ], [0])
        bins = set(np.searchsorted(edges, y[b], side='right').tolist())
        assert len(bins) == 8, f"a 24-row stratified batch must span all 8 bins (got {len(bins)})"
    ok("every 24-row batch spans all 8 target-quantile bins (full y-spread preserved)")

    # Degenerate (constant) target → one stratum → behaves like uniform, no crash.
    yc = np.ones(50)
    coc = evo13._PredatorPreyCoevolution(n_rows=50, subset=8, pop_size=4,
                                         mut_rate=0.3, virulence=0.75, seed=9,
                                         min_eval_rows=8, stratify=True)
    b = coc.next_batch(_X_with_ids(50), yc.reshape(-1, 1),
                       [_champion_from_residual(yc, np.zeros(50))], [0])
    assert len(b) == 8 and len(set(b.tolist())) == 8
    ok("constant target degenerates to one stratum (uniform draw, no crash)")
    print()


# --------------------------------------------------------------------------
# 5.  the admission wrapper feeds the governor (and stays an honest rescore)
# --------------------------------------------------------------------------
def test_admission_wrapper():
    print("5: _coevo_admission_rescore = honest full-data rescore + governor feed")
    evo13.AFFINE_SCALING_ENABLED   = True
    evo13.PUSH_ENABLED             = False
    evo13.SOBOLEV_ENABLED          = False
    evo13.INSTANCE_REWEIGHT_ENABLED = False
    evo13._DIFFICULTY_ACTIVE       = None

    rng = np.random.default_rng(10)
    X = rng.normal(size=(3000, 3))
    y = X[:, 0] + X[:, 1] + X[:, 2]

    # The classic subset-overfit culprit: an x0-only model scored on 2 rows.
    eq = evo13.CGPEquation(n_features=3, max_nodes=4)
    eq.nodes = [evo13.CGPNode('add', 0, 0) for _ in range(4)]
    eq.out_idx = 0
    eq.update_active_nodes()
    ind = evo13.Individual(eq)
    idx = np.array([10, 20])
    ind.calculate_fitness(X[idx], y[idx], 5, batch_indices=None, use_cache=False)
    assert ind.loss < 1e-6, "precondition: the 2-row subset must look perfect"

    evo13.COEVOLUTION_ENABLED = True
    evo13._COEVO_RUNTIME = evo13._PredatorPreyCoevolution(
        n_rows=3000, subset=16, pop_size=8, mut_rate=0.3, virulence=0.75, seed=11)
    evo13._coevo_admission_rescore(ind, X, y, 5, None)

    assert ind.loss > 0.5 and 0.25 < ind.r2 < 0.45, \
        f"wrapper must still produce the honest full-data score (loss={ind.loss:.3f}, R²={ind.r2:.3f})"
    ok(f"rescore stays honest (R² {ind.r2:.3f} ≈ 1/3, the true x0-only fit)")
    assert evo13._COEVO_RUNTIME._gap_chunk_max is not None \
        and evo13._COEVO_RUNTIME._gap_chunk_max > 0.99, \
        "the wrapper must report the (huge) batch→full gap to the governor"
    ok(f"governor was fed the gap ({evo13._COEVO_RUNTIME._gap_chunk_max:.3f} ≈ 1: total subset overfit)")

    # Co-evolution off → the wrapper degrades to a plain rescore (no crash).
    evo13.COEVOLUTION_ENABLED = False
    evo13._COEVO_RUNTIME = None
    ind2 = evo13.Individual(eq)
    ind2.calculate_fitness(X[idx], y[idx], 5, batch_indices=None, use_cache=False)
    evo13._coevo_admission_rescore(ind2, X, y, 5, None)
    assert ind2.loss > 0.5
    ok("with co-evolution off the wrapper is a plain full-data rescore")
    print()


# --------------------------------------------------------------------------
# 6.  cold start honours the floor + stratification
# --------------------------------------------------------------------------
def test_cold_start():
    print("6: cold start (no champion) returns a representative floor-sized batch")
    n = 1000
    y = np.zeros(n); y[:100] = 1.0                  # 10% minority
    X, Y = _X_with_ids(n), y.reshape(-1, 1)
    co = evo13._PredatorPreyCoevolution(n_rows=n, subset=1, pop_size=8,
                                        mut_rate=0.3, virulence=0.75, seed=12,
                                        min_eval_rows=16, stratify=True)
    b = co.next_batch(X, Y, [_FakeHoF(None)], [6])   # no champion yet
    assert len(b) == 16 and len(set(b.tolist())) == 16
    assert np.any(y[b] > 0.5), "cold-start batch must already carry the minority class"
    ok("cold start: 16 distinct stratified rows (minority class included) at subset=1")
    print()


if __name__ == "__main__":
    test_eval_floor()
    test_resolution_governor()
    test_governor_antiwindup()
    test_stratified_anchor_classification()
    test_stratified_anchor_regression()
    test_admission_wrapper()
    test_cold_start()
    print("ALL RESOLUTION TESTS PASSED ✓")
