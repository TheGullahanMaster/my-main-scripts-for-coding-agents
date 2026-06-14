"""Tests for the progress-tracking + arms-race stabilisation added to evo13's
predator–prey co-evolution (failure mode 4: cycling / catastrophic forgetting).

These target the problem reported on many-variable (10+ input) targets: the
predators swing the batch onto a narrow region, the hosts overfit it and DROP a
variable that region doesn't need, then have to RELEARN it when the predators
swing away — the search oscillates instead of converging.

Mechanisms under test:

  • progress TRACKING — global host loss + short-term trend, a per-row best-ever
    fit, and a plateau-robust FORGETTING signal (mean per-row regression), all
    folded into a smoothed instability ∈ [0,1].  Always on (observational).
  • STABILISER — while the hosts are forgetting, WIDEN the uniform anchor (more
    breadth, so every variable stays represented) and damp predator virulence.
    Closed-loop simulation showed breadth — not re-injecting the specific dropped
    rows — is the lever that fixes the forgetting.  Gated by `stabilize`.

Run:  python test_coevo_stabilization.py
"""
import numpy as np

import evo13


# --------------------------------------------------------------------------
# fakes — a champion whose per-row residual pattern we control (and can MUTATE
# between chunks, to simulate the hosts learning then dropping a variable)
# --------------------------------------------------------------------------
class _MutableTree:
    """evaluate(X) looks each row up by its integer id in column 0 of X and
    returns y + resid for that row, so _row_hardness sees exactly the residual
    pattern we set.  `resid` is mutable so a test can change the fit over time."""
    def __init__(self, y):
        self.y = y.astype(np.float64)
        self.resid = np.zeros_like(self.y)
    def evaluate(self, X):
        ids = X[:, 0].astype(int)
        return (self.y + self.resid)[ids]


class _MutChampion:
    def __init__(self, y):
        self.tree = _MutableTree(y)
        self.affine_a = 1.0
        self.affine_b = 0.0


class _MutHoF:
    def __init__(self, champ):
        self._c = champ
    def get_best_overall(self):
        return self._c


def _X_ids(n):
    X = np.zeros((n, 1), dtype=np.float64)
    X[:, 0] = np.arange(n)
    return X


def ok(name):
    print(f"  ✓ {name}")


# --------------------------------------------------------------------------
# 1.  progress tracking + instability signal
# --------------------------------------------------------------------------
def test_progress_tracking_and_instability():
    print("1: global-loss + forgetting tracking drives instability")
    n = 300
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, n) * 4.0
    X, Y = _X_ids(n), y.reshape(-1, 1)
    champ = _MutChampion(y)
    hof = [_MutHoF(champ)]
    co = evo13._PredatorPreyCoevolution(n_rows=n, subset=30, pop_size=16,
                                        mut_rate=0.3, virulence=0.75, seed=1)

    # Phase A — well fit everywhere: instability ~0, no forgetting.
    for _ in range(15):
        co.next_batch(X, Y, hof, [0])
    inst_fit = co.instability
    assert co.global_loss is not None and co.history, "progress not recorded"
    assert inst_fit < 0.05, f"a well-fit, flat run should read ~0 instability ({inst_fit:.3f})"
    assert co.forget_frac < 1e-6, "no forgetting expected on a flat good fit"
    ok(f"flat/well-fit run → instability≈0 (={inst_fit:.3f}), forgetting≈0")

    # Phase B — the hosts drop a cluster (a dropped variable): forgetting rises,
    # so instability rises (even though the global loss is now ~flat-but-high).
    champ.tree.resid[50:90] = 4.0 * np.std(y)
    for _ in range(10):
        co.next_batch(X, Y, hof, [0])
    inst_regress = co.instability
    assert co.forget_frac > 0.0, "forgetting signal should fire on the dropped cluster"
    assert inst_regress > 0.3, f"forgetting should raise instability ({inst_regress:.3f})"
    ok(f"a dropped cluster raises forgetting→instability ({inst_fit:.3f} → {inst_regress:.3f})")

    # Phase C — the hosts recover (variable relearned): instability decays again.
    champ.tree.resid[:] = 0.0
    for _ in range(30):
        co.next_batch(X, Y, hof, [0])
    assert co.instability < 0.5 * inst_regress, \
        f"recovery should let instability decay ({inst_regress:.3f} → {co.instability:.3f})"
    ok(f"recovery → instability decays ({inst_regress:.3f} → {co.instability:.3f})")

    # stabilize=False: instability is still TRACKED, but the anchor/virulence are
    # never changed (the action is gated; the observation is not).  Run the same
    # fit→drop sequence so co2 actually experiences (and records) forgetting.
    co2 = evo13._PredatorPreyCoevolution(n_rows=n, subset=30, pop_size=16,
                                         mut_rate=0.3, virulence=0.75, seed=1,
                                         stabilize=False, anchor_frac=0.25)
    champ.tree.resid[:] = 0.0
    for _ in range(15):
        co2.next_batch(X, Y, hof, [0])           # fit everywhere first
    champ.tree.resid[50:90] = 4.0 * np.std(y)    # then drop the cluster
    for _ in range(10):
        co2.next_batch(X, Y, hof, [0])
    assert co2.instability > 0.0, "instability should still be tracked when stabilize=False"
    assert abs(co2._effective_anchor_frac() - 0.25) < 1e-9, \
        "stabilize=False must leave the anchor at the configured value"
    assert abs(co2._effective_virulence() - 0.75) < 1e-9, \
        "stabilize=False must leave virulence at the configured value"
    ok("stabilize=False → instability tracked but anchor/virulence untouched")
    print()


# --------------------------------------------------------------------------
# 2.  per-row best-ever fit + regression signal (noise does NOT count)
# --------------------------------------------------------------------------
def test_best_ever_and_regression():
    print("2: per-row best-ever fit + regression flags ONLY the dropped rows")
    n = 200
    rng = np.random.default_rng(2)
    y = rng.normal(0.0, 1.0, n) * 3.0
    X, Y = _X_ids(n), y.reshape(-1, 1)
    champ = _MutChampion(y)
    hof = [_MutHoF(champ)]
    co = evo13._PredatorPreyCoevolution(n_rows=n, subset=20, pop_size=12,
                                        mut_rate=0.3, virulence=1.0, seed=3,
                                        noise_guard=False)

    noise = np.arange(0, 20)                 # hard from the start (never fit)
    champ.tree.resid[noise] = 2.0 * np.std(y)
    learn = np.arange(100, 130)              # fit first, then dropped

    for _ in range(15):
        co.next_batch(X, Y, hof, [0])
    assert np.all(co.hard_best[learn] < 0.05), \
        "best-ever should record the good fit on the learnable cluster"
    ok("best-ever fit snaps DOWN fast (records the learnable cluster as ~solved)")

    champ.tree.resid[learn] = 3.0 * np.std(y)     # drop the variable
    co.next_batch(X, Y, hof, [0])
    reg = co._regression(co._row_hardness(X, Y, hof, [0]))
    assert reg is not None and np.all(reg[learn] > 0.3), \
        f"the just-dropped cluster must show strong regression ({reg[learn].mean():.3f})"
    ok(f"regression fires on the dropped cluster (mean={reg[learn].mean():.3f})")

    assert np.all(reg[noise] < 0.05), \
        f"never-fit (noise) rows must NOT read as regressed ({reg[noise].max():.3f})"
    ok("never-fit (noise) rows read ~0 regression (noise ≠ forgetting)")
    print()


# --------------------------------------------------------------------------
# 3.  the stabiliser widens the anchor as instability rises
# --------------------------------------------------------------------------
def test_adaptive_anchor():
    print("3: the stabiliser widens the uniform anchor as instability rises")
    n, subset = 400, 40
    co = evo13._PredatorPreyCoevolution(n_rows=n, subset=subset, pop_size=8,
                                        mut_rate=0.3, virulence=0.75, seed=4,
                                        anchor_frac=0.25, anchor_boost=0.30,
                                        adv_floor=0.30)
    co.instability = 0.0
    assert abs(co._effective_anchor_frac() - 0.25) < 1e-9, "inst 0 → configured anchor"
    co.instability = 0.5
    a_mid = co._effective_anchor_frac()
    co.instability = 1.0
    a_hi = co._effective_anchor_frac()
    assert 0.25 < a_mid < a_hi, f"anchor must widen with instability (0.25<{a_mid:.2f}<{a_hi:.2f})"
    # capped so the arms race keeps ≥ adv_floor of the batch
    assert a_hi <= 1.0 - co.adv_floor + 1e-9, f"anchor must respect adv_floor ({a_hi:.2f})"
    ok(f"anchor widens 0.25 → {a_mid:.2f} → {a_hi:.2f} (capped at 1−adv_floor={1-co.adv_floor:.2f})")

    # batch still always exactly `subset` distinct in-range rows, arms race ≥ floor
    parasite = co.parasites[0]
    sel = np.zeros(n); sel[parasite] = np.linspace(1.0, 0.5, subset)
    for inst in (0.0, 0.5, 1.0):
        co.instability = inst
        b = co._assemble_batch(parasite, sel)
        assert len(b) == subset and len(set(b.tolist())) == subset, "size contract"
        assert b.min() >= 0 and b.max() < n, "rows in range"
        kept = len(set(b.tolist()) & set(parasite.tolist()))
        assert kept >= int(co.adv_floor * subset) - 1, \
            f"arms-race floor broken at inst={inst}: kept {kept}/{subset}"
    ok("batch always `subset` distinct in-range rows; arms race ≥ adv_floor throughout")
    print()


# --------------------------------------------------------------------------
# 3b.  the adversarial spotlight never closes completely — even a 1-row subset
#      at max instability keeps a parasite row, so co-evolution can't silently
#      degrade into pure anchor sampling
# --------------------------------------------------------------------------
def test_adversarial_core_never_empty():
    print("3b: a tiny subset under max instability still carries an adversarial row")
    n = 200
    # At subset=1 the boosted anchor fraction (→ 1−adv_floor = 0.70) rounds to a
    # full anchor row, which used to leave n_adv=0: the predator population then
    # contributed NOTHING and the batch was indistinguishable from a random
    # anchor draw.  The spotlight must stay open (≥1 parasite row) regardless.
    for subset in (1, 2, 3):
        co = evo13._PredatorPreyCoevolution(
            n_rows=n, subset=subset, pop_size=8, mut_rate=0.3, virulence=0.75,
            seed=0, anchor_frac=0.25, anchor_boost=0.30, adv_floor=0.30,
            min_eval_rows=0, stratify=False)
        parasite = co.parasites[0]
        sel = np.zeros(n); sel[parasite] = np.linspace(1.0, 0.5, parasite.size)
        for inst in (0.0, 0.5, 1.0):
            co.instability = inst
            b = co._assemble_batch(parasite, sel)
            adv_rows = len(set(b.tolist()) & set(parasite.tolist()))
            assert adv_rows >= 1, \
                f"subset={subset} inst={inst}: adversarial spotlight closed " \
                f"(0 parasite rows in the batch)"
            assert len(b) == len(set(b.tolist())), "batch rows must stay distinct"
    ok("≥1 adversarial row at subset∈{1,2,3} across instability 0→1 (spotlight stays open)")
    print()


# --------------------------------------------------------------------------
# 4.  dormant when nothing is forgotten (legacy behaviour preserved)
# --------------------------------------------------------------------------
def test_stabilizer_dormant_when_healthy():
    print("4: no forgetting → instability≈0 → legacy adv/anchor split preserved")
    n, subset = 300, 40
    rng = np.random.default_rng(5)
    y = rng.normal(0.0, 1.0, n) * 3.0
    X, Y = _X_ids(n), y.reshape(-1, 1)
    # a FIXED champion (constant residual) — never improves, never regresses
    resid = np.zeros(n); resid[:30] = 5.0
    champ = _MutChampion(y); champ.tree.resid[:] = resid
    co = evo13._PredatorPreyCoevolution(n_rows=n, subset=subset, pop_size=8,
                                        mut_rate=0.3, virulence=0.75, seed=5,
                                        anchor_frac=0.25)
    for _ in range(20):
        b = co.next_batch(X, Y, [_MutHoF(champ)], [0])
        assert len(b) == subset and len(set(b.tolist())) == subset, "size contract"
    assert co.forget_frac < 1e-6 and co.instability < 1e-6, \
        "a fixed champion must not register forgetting/instability"
    assert abs(co._effective_anchor_frac() - 0.25) < 1e-9, \
        "with no forgetting the anchor must stay at the configured value"
    ok("fixed champion → no forgetting, anchor stays at the legacy 0.25 (dormant)")
    print()


# --------------------------------------------------------------------------
# 5.  the stabiliser damps the effective virulence under instability
# --------------------------------------------------------------------------
def test_effective_virulence_damping():
    print("5: the stabiliser damps effective virulence as instability rises")
    co = evo13._PredatorPreyCoevolution(n_rows=100, subset=10, pop_size=8,
                                        mut_rate=0.3, virulence=0.8, seed=6,
                                        virulence_damp=0.4)
    co.instability = 0.0
    assert abs(co._effective_virulence() - 0.8) < 1e-9, "no instability → full λ"
    co.instability = 1.0
    full_damp = co._effective_virulence()
    assert abs(full_damp - 0.8 * 0.6) < 1e-9, \
        f"full instability → λ cut by virulence_damp (got {full_damp:.3f})"
    co.instability = 0.5
    assert full_damp <= co._effective_virulence() <= 0.8, "λ should fall monotonically"
    ok(f"λ damped 0.80 → {full_damp:.2f} at full instability (calmer predators)")

    co_off = evo13._PredatorPreyCoevolution(n_rows=100, subset=10, pop_size=8,
                                            mut_rate=0.3, virulence=0.8, seed=6,
                                            stabilize=False)
    co_off.instability = 1.0
    assert abs(co_off._effective_virulence() - 0.8) < 1e-9, \
        "stabilize=False must leave λ at the configured value"
    ok("stabilize=False → λ never damped")
    print()


# --------------------------------------------------------------------------
# 6.  closed-loop: the stabiliser beats the bare arms race on a many-variable toy
# --------------------------------------------------------------------------
def _run_forgetting_toy(stabilize, seed, G=12, per=30, subset=40, chunks=260,
                        LEARN=0.3, DROP=0.45):
    """Faithful closed loop modelling the user's many-variable regime: G variable
    "groups", each fit in proportion to its share of the batch but with
    DIMINISHING returns (saturating at a few rows — extra rows on one group are
    wasted) and DROPPED fast when the batch ignores it.  An un-damped arms race
    over-concentrates on a few groups and the rest get dropped; breadth keeps them
    all represented.  Returns (mean loss, fraction of chunks all groups are fit)."""
    n = G * per
    grp = np.repeat(np.arange(G), per)
    thr, fsat = 1.0 / subset, 3.0 / subset
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, 1.0, n) * 3.0
    X, Y = _X_ids(n), y.reshape(-1, 1)
    m = np.full(G, 0.5)
    sd = float(np.std(y))
    champ = _MutChampion(y)
    hof = [_MutHoF(champ)]

    def sync():
        champ.tree.resid[:] = (1.0 - m[grp]) * 3.0 * sd
    sync()

    co = evo13._PredatorPreyCoevolution(
        n_rows=n, subset=subset, pop_size=24, mut_rate=0.3, virulence=0.75,
        seed=seed, anchor_frac=0.25, fitness_sharing=True, noise_guard=False,
        stabilize=stabilize)

    M = []
    for _ in range(chunks):
        b = co.next_batch(X, Y, hof, [0])
        frac = np.bincount(grp[b], minlength=G).astype(float) / subset
        gain = LEARN * np.minimum(1.0, frac / fsat)     # diminishing returns
        m = np.where(frac >= thr, m + gain * (1.0 - m), m - DROP * m)
        m = np.clip(m, 0.0, 1.0)
        sync()
        M.append(m.copy())
    M = np.array(M[-100:])
    return float((1.0 - M).mean()), float((M > 0.7).all(axis=1).mean())


def test_stabilizer_helps_many_variable():
    print("6: closed-loop — the stabiliser beats the bare arms race (10+ var regime)")
    off_loss, on_loss, off_fit, on_fit = [], [], [], []
    for seed in range(4):
        l0, f0 = _run_forgetting_toy(stabilize=False, seed=seed)
        l1, f1 = _run_forgetting_toy(stabilize=True,  seed=seed)
        off_loss.append(l0); on_loss.append(l1); off_fit.append(f0); on_fit.append(f1)
    mo, mn = float(np.mean(off_loss)), float(np.mean(on_loss))
    fo, fn = float(np.mean(off_fit)),  float(np.mean(on_fit))
    print(f"    bare arms race : mean-loss={mo:.3f}  all-vars-fit-frac={fo:.2f}")
    print(f"    + stabiliser   : mean-loss={mn:.3f}  all-vars-fit-frac={fn:.2f}")

    assert mn < mo - 0.02, \
        f"stabiliser should reach a clearly lower loss ({mn:.3f} vs {mo:.3f})"
    ok(f"lower final loss with the stabiliser ({mo:.3f} → {mn:.3f})")
    assert fn >= fo, \
        f"stabiliser should fit all variables at least as often ({fn:.2f} vs {fo:.2f})"
    ok(f"all variables fit simultaneously more often ({fo:.2f} → {fn:.2f})")
    print()


if __name__ == "__main__":
    test_progress_tracking_and_instability()
    test_best_ever_and_regression()
    test_adaptive_anchor()
    test_adversarial_core_never_empty()
    test_stabilizer_dormant_when_healthy()
    test_effective_virulence_damping()
    test_stabilizer_helps_many_variable()
    print("ALL STABILISATION TESTS PASSED ✓")
