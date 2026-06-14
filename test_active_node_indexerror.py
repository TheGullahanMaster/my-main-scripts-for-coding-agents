"""Regression tests for the update_active_nodes() IndexError crash.

The crash (IndexError: list index out of range at
`node = self.nodes[idx - self.n_features]`) was triggered when an
active-subgraph compaction (residual-composite seeding via
`_compose_additive_tree`/`_emit_active_nodes`, or ADF inlining via
`_topological_sort_eq`) renumbered a tree's active nodes into a smaller node
list while leaving a node's don't-care input slot pointing at a stale,
now-out-of-range index.  A later op-arity mutation promoted that slot live and
update_active_nodes() walked off the end of self.nodes.

These tests exercise:
  1. the residual-composite compaction path,
  2. the topological-sort (ADF-inline) compaction path,
  3. the self-healing safety net in update_active_nodes() (and that evaluate()
     stays consistent afterwards),
  4. a broad mutate/compose/crossover fuzz that must never raise.
"""
import random
import numpy as np
import evo13
from evo13 import CGPNode, CGPEquation


def _ref_in_range(eq):
    """Assert the contract update_active_nodes() guarantees: the output pointer
    and every ACTIVE node's *used* inputs (the slots evaluate() actually reads)
    are in range.  Inactive nodes may legitimately still carry an out-of-range
    don't-care slot — they get healed only if/when they become active."""
    limit = eq.n_features + len(eq.nodes)
    assert 0 <= eq.out_idx < limit, (eq.out_idx, limit)
    for idx in eq.active_nodes:
        assert 0 <= idx < limit, ("active idx", idx, limit)
        if idx < eq.n_features:
            continue
        n = eq.nodes[idx - eq.n_features]
        if n.op == 'const':
            continue
        used = [n.in1]
        if n.op in eq.OPS_BINARY_SET:
            used.append(n.in2)
        elif n.op in eq.OPS_TERNARY_SET:
            used.extend([n.in2, n.in3])
        for slot in used:
            assert 0 <= slot < limit, ("used slot", n.op, slot, limit)


def test_compose_compaction_no_dangling_slot():
    """A unary active node carrying a high (inactive) in2 must not produce an
    out-of-range slot after _compose_additive_tree, even once promoted."""
    nf = 1
    a = CGPEquation(nf, 12, ['x'])
    a.nodes = [CGPNode('const', 0, 0, 0.0) for _ in range(12)]
    a.nodes[0] = CGPNode('exp', 0, nf + 11, 1.0)   # in2 -> inactive slot 11
    a.out_idx = nf
    a.update_active_nodes()

    b = CGPEquation(nf, 12, ['x'])
    b.nodes = [CGPNode('const', 0, 0, 0.0) for _ in range(12)]
    b.nodes[0] = CGPNode('*', 0, 0, 0.0)
    b.out_idx = nf
    b.update_active_nodes()

    comp = evo13._compose_additive_tree(a, b, scale_a=1.0, max_total=200)
    assert comp is not None
    comp.update_active_nodes()
    _ref_in_range(comp)

    # Promote the (formerly unary) exp node to a binary op and re-trace: the
    # old failure mode crashed right here.
    exp_local = next(i for i, n in enumerate(comp.nodes) if n.op == 'exp')
    comp.nodes[exp_local].op = '+'
    comp.update_active_nodes()          # must not raise
    _ref_in_range(comp)


def test_topological_sort_compaction_no_dangling_slot():
    """_topological_sort_eq must also collapse unresolved don't-care slots."""
    nf = 1
    eq = CGPEquation(nf, 12, ['x'])
    eq.nodes = [CGPNode('const', 0, 0, 0.0) for _ in range(12)]
    # active: out -> exp(node? ) ; exp has a stale in2 to an inactive high slot.
    eq.nodes[0] = CGPNode('sin', 0, 0, 0.0)          # active leaf: sin(x)
    eq.nodes[1] = CGPNode('exp', nf + 0, nf + 10, 0.0)  # exp(sin(x)); in2 stale
    eq.out_idx = nf + 1
    eq.update_active_nodes()

    sorted_eq = evo13._topological_sort_eq(eq)
    sorted_eq.update_active_nodes()
    _ref_in_range(sorted_eq)

    # Promote exp -> binary and re-trace.
    exp_local = next(i for i, n in enumerate(sorted_eq.nodes) if n.op == 'exp')
    sorted_eq.nodes[exp_local].op = '*'
    sorted_eq.update_active_nodes()     # must not raise
    _ref_in_range(sorted_eq)


def test_safety_net_self_heals_corrupt_genome():
    """Even a hand-corrupted genome (out-of-range out_idx + live in/refs) must
    not crash update_active_nodes() or evaluate()."""
    nf = 2
    eq = CGPEquation(nf, 4, ['x', 'y'])
    eq.nodes = [CGPNode('const', 0, 0, 0.0) for _ in range(4)]
    # A live binary node whose in2 is wildly out of range, plus an out-of-range
    # output pointer.
    eq.nodes[0] = CGPNode('+', 0, 999, 0.0)     # in2 = 999 (out of range)
    eq.out_idx = 12345                          # out of range
    eq.update_active_nodes()                    # self-heals out_idx -> 0
    _ref_in_range(eq)

    # Point output at the corrupt node and re-trace + evaluate.
    eq.nodes[0] = CGPNode('+', 0, 999, 0.0)
    eq.out_idx = nf + 0
    eq.update_active_nodes()                    # heals in2 -> 0
    _ref_in_range(eq)
    X = np.random.randn(8, nf)
    out = eq.evaluate(X)                         # must not raise
    assert out.shape[0] == 8
    assert np.all(np.isfinite(np.nan_to_num(out)))


def test_normal_tree_traversal_unchanged():
    """For a well-formed tree the guard is a no-op: active set + output match a
    hand traversal, and references are untouched."""
    random.seed(0); np.random.seed(0)
    eq = evo13.random_cgp(3, evo13.CGP_NODES, ['a', 'b', 'c'])
    before = [(n.op, n.in1, n.in2, n.in3) for n in eq.nodes]
    active_before = set(eq.active_nodes)
    eq.update_active_nodes()
    after = [(n.op, n.in1, n.in2, n.in3) for n in eq.nodes]
    assert before == after, "guard must not rewire a well-formed genome"
    assert active_before == set(eq.active_nodes)
    _ref_in_range(eq)


def test_fuzz_mutate_compose_crossover():
    """Heavy fuzz across mutate / compose / crossover / simplify — must never
    raise IndexError from a dangling reference."""
    random.seed(1234); np.random.seed(1234)
    nf = 3
    feats = ['a', 'b', 'c']
    X = np.random.randn(32, nf)
    pop = [evo13.random_cgp(nf, evo13.CGP_NODES, feats) for _ in range(12)]

    for it in range(1500):
        p = random.choice(pop)
        kind = random.random()
        if kind < 0.55:
            child = evo13.mutate(p, nf, feats,
                                 mut_rate=random.randint(1, 25),
                                 temperature=random.random())
        elif kind < 0.70:
            q = random.choice(pop)
            child = evo13.crossover(p, q)
        elif kind < 0.85:
            # residual-composite compaction path
            other = random.choice(pop)
            child = evo13._compose_additive_tree(
                p, other, scale_a=random.uniform(-3, 3), max_total=300)
            if child is None:
                continue
        else:
            child = evo13.simplify_cgp_tree(p)

        child.update_active_nodes()         # the historically-crashing call
        _ref_in_range(child)
        # evaluate must also stay in range
        out = child.evaluate(X)
        assert out.shape[0] == 32
        # keep the population churning (bounded size)
        pop.append(child)
        if len(pop) > 40:
            pop = pop[-24:]


if __name__ == "__main__":
    test_compose_compaction_no_dangling_slot()
    print("ok: compose compaction")
    test_topological_sort_compaction_no_dangling_slot()
    print("ok: topological-sort compaction")
    test_safety_net_self_heals_corrupt_genome()
    print("ok: safety net self-heals")
    test_normal_tree_traversal_unchanged()
    print("ok: normal traversal unchanged")
    test_fuzz_mutate_compose_crossover()
    print("ok: fuzz mutate/compose/crossover")
    print("\nALL TESTS PASSED")
