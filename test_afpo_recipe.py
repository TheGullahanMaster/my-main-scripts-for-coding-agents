"""Regression tests for three AFPO-recipe items added to ``evolve_afpo``:

  4. Pareto tie-breaking on structural depth — when two individuals share an
     identical (age, fitness, complexity) profile the NSGA-II boundary trim now
     keeps the structurally shallower one (``_active_graph_depth`` +
     ``_trim_to_pareto_front_3obj`` lexsort tie-break).

  5a. Balanced low-depth native injections — the per-generation age-0 immigrant
      draws its node budget from a small low range (``_afpo_injection_budget``)
      instead of a single fixed size, so newcomers enter as small "underdogs"
      with room to scale up rather than as massive random trees.

  5b. ERC (Ephemeral Random Constant) dial-in for newly-formed Pareto-front
      members — fresh age-0 injects with random constants get a quick local
      constant optimisation (``_polish_young_frontier_constants``) so a good
      structure is not Pareto-dominated and culled before its constants can be
      tuned.

Requirements 1–3 (genotypic age tracking, the 3-objective criterion, and the
fast non-dominated sort itself) were already implemented; these tests cover the
gaps that were filled in.
"""
import os
import random

os.environ.setdefault("PYTHONWARNINGS", "ignore")
import numpy as np
import evo13 as e


def _const_tree(nf, feat, c):
    """A single 'const' node tree whose only constant is ``c``."""
    eq = e.CGPEquation(nf, 4, feat)
    eq.nodes = [e.CGPNode('const', 0, 0, c)]
    eq.out_idx = nf + 0
    eq.update_active_nodes()
    return eq


def _chain_tree(nf, feat, depth):
    """A '+' chain of the requested active-graph depth:
    node0 = x0 + x1 ; nodeK = node(K-1) + x2 ; out = last node."""
    eq = e.CGPEquation(nf, depth + 2, feat)
    eq.nodes = [e.CGPNode('+', 0, 1, 0.0)]          # idx nf+0, depth 1
    prev = nf + 0
    for k in range(1, depth):
        eq.nodes.append(e.CGPNode('+', prev, 2, 0.0))   # nodeK = prev + x2
        prev = nf + k
    eq.out_idx = prev
    eq.update_active_nodes()
    return eq


# ──────────────────────────────────────────────────────────────────────────
# Req 4 — structural-depth Pareto tie-break
# ──────────────────────────────────────────────────────────────────────────
def test_active_graph_depth():
    e.set_ops_mode(True); e._INIT_PHASE = False
    nf, feat = 3, ["x0", "x1", "x2"]

    # Output pointed straight at a raw feature → depth 0.
    eq0 = e.CGPEquation(nf, 4, feat)
    eq0.nodes = [e.CGPNode('+', 0, 1, 0.0)]
    eq0.out_idx = 0
    eq0.update_active_nodes()
    assert e._active_graph_depth(eq0) == 0

    for d in (1, 2, 3, 4, 5):
        assert e._active_graph_depth(_chain_tree(nf, feat, d)) == d, d

    assert e._active_graph_depth(None) == 0

    # Never crashes / always a non-negative int on real random trees.
    random.seed(0); np.random.seed(0)
    for _ in range(1000):
        d = e._active_graph_depth(e.random_cgp(nf, e.CGP_NODES, feat))
        assert isinstance(d, int) and d >= 0
    print("PASS test_active_graph_depth")


def test_depth_tiebreak_prefers_shallower():
    """Four individuals with an IDENTICAL (age, fitness, complexity) profile but
    increasing depth.  Trimming to 3 keeps the two crowding-boundary trees
    (shallowest + deepest) and, between the two equally-crowded interior trees,
    the depth tie-break must drop the deeper one."""
    e.set_ops_mode(True); e._INIT_PHASE = False
    nf, feat = 3, ["x0", "x1", "x2"]

    inds = []
    for d in (1, 2, 3, 4):
        t = _chain_tree(nf, feat, d)
        assert e._active_graph_depth(t) == d
        ind = e.Individual(t)
        ind.age = 5            # force an identical Pareto profile so depth is
        ind.fitness = 1.0      # the ONLY thing that separates the four
        ind.loss = 1.0
        ind.complexity = 10.0
        inds.append(ind)

    survivors = e._trim_to_pareto_front_3obj(inds, 3)
    surv_depths = sorted(e._active_graph_depth(s.tree) for s in survivors)
    assert len(survivors) == 3, len(survivors)
    assert surv_depths == [1, 2, 4], surv_depths
    assert 3 not in surv_depths, "deeper interior tree should lose the tie-break"
    print("PASS test_depth_tiebreak_prefers_shallower")


# ──────────────────────────────────────────────────────────────────────────
# Req 5a — balanced low-depth injection budget
# ──────────────────────────────────────────────────────────────────────────
def test_injection_budget_balanced_distribution():
    e.AFPO_BALANCED_INJECT_ENABLED = True
    random.seed(1)
    budgets = [e._afpo_injection_budget() for _ in range(500)]
    lo, hi = min(budgets), max(budgets)
    # Default CGP_NODES=50 → band [max(4,4), max(6,12)] = [4, 12].
    assert lo >= 4, lo
    assert hi <= max(6, e.CGP_NODES // 4), hi
    assert lo < hi, f"balanced injection should produce a spread, got {set(budgets)}"
    # Low-depth guarantee: random_cgp's structured target depth ≤ budget//2,
    # so the budget must stay well below the full node count.
    assert hi <= e.CGP_NODES // 2, hi

    # Flag off → the single legacy fixed budget.
    e.AFPO_BALANCED_INJECT_ENABLED = False
    fixed = {e._afpo_injection_budget() for _ in range(50)}
    assert fixed == {max(4, e.CGP_NODES // 4)}, fixed

    e.AFPO_BALANCED_INJECT_ENABLED = True   # restore default
    print("PASS test_injection_budget_balanced_distribution")


# ──────────────────────────────────────────────────────────────────────────
# Req 5b — ERC dial-in for newly-formed Pareto-front members
# ──────────────────────────────────────────────────────────────────────────
def test_erc_polish_dials_in_young_frontier_constants():
    e.set_ops_mode(True); e._INIT_PHASE = False
    e.AFPO_ERC_POLISH_ENABLED = True
    # Affine off so the tree's CONSTANT (not the analytic a·f+b wrapper) has to
    # carry the target's scale — this is what makes the random ERC matter.
    e.AFFINE_SCALING_ENABLED = False
    random.seed(0); np.random.seed(0)
    nf, feat = 2, ["x0", "x1"]
    X = np.random.uniform(-2.0, 2.0, size=(60, 2))
    y = np.full(X.shape[0], 5.0)

    # Young member: right STRUCTURE (a constant) but a bad random ERC (0.0).
    young = e.Individual(_const_tree(nf, feat, 0.0))
    young.age = 0
    young.calculate_fitness(X, y, 5)
    pre = young.loss
    assert pre > 1.0, pre

    # A few OLD, worse, more-complex junk members so `young` is unambiguously on
    # the first Pareto front (younger, lower loss, simpler).
    pop = [young]
    for _ in range(3):
        j = e.Individual(e.random_cgp(nf, e.CGP_NODES, feat))
        j.age = 100
        j.calculate_fitness(X, y, 5)
        j.loss = pre + 10.0
        j.fitness = j.loss
        j.complexity = young.complexity + 50.0
        pop.append(j)

    e._polish_young_frontier_constants(pop, X, y, 5)
    assert getattr(young, "_erc_polished", False) is True
    assert young.loss < pre * 0.5, (pre, young.loss)

    # Flag OFF → strict no-op (flag not stamped, loss untouched).
    e.AFPO_ERC_POLISH_ENABLED = False
    young2 = e.Individual(_const_tree(nf, feat, 0.0)); young2.age = 0
    young2.calculate_fitness(X, y, 5)
    pre2 = young2.loss
    e._polish_young_frontier_constants([young2], X, y, 5)
    assert not getattr(young2, "_erc_polished", False)
    assert abs(young2.loss - pre2) < 1e-12

    e.AFPO_ERC_POLISH_ENABLED = True        # restore defaults
    e.AFFINE_SCALING_ENABLED = True
    print("PASS test_erc_polish_dials_in_young_frontier_constants")


def test_erc_polish_respects_caps_and_age():
    e.set_ops_mode(True); e._INIT_PHASE = False
    e.AFPO_ERC_POLISH_ENABLED = True
    e.AFFINE_SCALING_ENABLED = False
    random.seed(2); np.random.seed(2)
    nf, feat = 2, ["x0", "x1"]
    X = np.random.uniform(-2.0, 2.0, size=(60, 2))
    y = np.full(X.shape[0], 4.0)

    # Five YOUNG front members → only MAX_MEMB are polished per call.
    pop = []
    for _ in range(5):
        m = e.Individual(_const_tree(nf, feat, 0.0)); m.age = 0
        m.calculate_fitness(X, y, 5)
        pop.append(m)
    e._polish_young_frontier_constants(pop, X, y, 5)
    n_polished = sum(bool(getattr(m, "_erc_polished", False)) for m in pop)
    assert n_polished == e.AFPO_ERC_POLISH_MAX_MEMB, n_polished

    # An over-age member is never polished even when it is on the front.
    old = e.Individual(_const_tree(nf, feat, 0.0))
    old.age = e.AFPO_ERC_POLISH_MAX_AGE + 5
    old.calculate_fitness(X, y, 5)
    e._polish_young_frontier_constants([old], X, y, 5)
    assert not getattr(old, "_erc_polished", False)

    e.AFFINE_SCALING_ENABLED = True          # restore default
    print("PASS test_erc_polish_respects_caps_and_age")


# ──────────────────────────────────────────────────────────────────────────
# End-to-end smoke: evolve_afpo runs with every new feature on
# ──────────────────────────────────────────────────────────────────────────
def test_end_to_end_evolve_afpo_recipe():
    e.set_ops_mode(True); e._INIT_PHASE = False
    e.AFPO_ERC_POLISH_ENABLED = True
    e.AFPO_BALANCED_INJECT_ENABLED = True
    e.AFFINE_SCALING_ENABLED = True
    random.seed(5); np.random.seed(5)
    nf, feat = 2, ["x0", "x1"]
    X = np.random.uniform(-2.0, 2.0, size=(120, 2))
    y = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1]   # constant-frequency: ERC matters

    pop = [e.Individual(e.random_cgp(nf, e.CGP_NODES, feat)) for _ in range(40)]
    for ind in pop:
        ind.calculate_fitness(X, y, 5)
    hof = e.HallOfFame(out_type=5)
    pop, stag = e.evolve_afpo(pop, X, y, 5, nf, feat, target_size=40,
                              n_generations=80, hof=hof, stag_counter=0,
                              ext_patience=60)

    assert len(pop) > 0
    assert len(set(id(i) for i in pop)) == len(pop), "no aliased individuals"
    assert all(getattr(i, "age", 0) >= 0 for i in pop)
    best = min(pop, key=lambda z: z.loss)
    assert np.isfinite(best.loss)
    n_polished = sum(bool(getattr(i, "_erc_polished", False)) for i in pop)
    print(f"  e2e: pool={len(pop)} best_loss={best.loss:.5f} "
          f"stag={stag} polished_survivors={n_polished}")
    print("PASS test_end_to_end_evolve_afpo_recipe")


if __name__ == "__main__":
    test_active_graph_depth()
    test_depth_tiebreak_prefers_shallower()
    test_injection_budget_balanced_distribution()
    test_erc_polish_dials_in_young_frontier_constants()
    test_erc_polish_respects_caps_and_age()
    test_end_to_end_evolve_afpo_recipe()
    print("\nALL AFPO RECIPE TESTS PASSED")
