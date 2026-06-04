"""
Regression tests for the adaptive operator-cost system in evo13.py.

`AdaptiveOperatorCosts` turns the hand-authored OP_COSTS table into a learned
one: operators the elite keep using get cheaper, ones they avoid get pricier.
These tests pin the safety contract that lets it be the default —

  * disabled (or un-observed) ⇒ the original static prices, byte-for-byte;
  * a useful operator really does get cheaper, an avoided one pricier;
  * every price stays clamped in a band around its prior;
  * the table mean is preserved (only RELATIVE prices move);
  * reset() restores the prior.

Runnable two ways:
    python test_adaptive_costs.py    # prints a short report, non-zero exit on failure
    pytest test_adaptive_costs.py    # standard test discovery
"""
import warnings
warnings.filterwarnings("ignore")

import evo13


# --------------------------------------------------------------------------- #
# Minimal stand-ins for an Individual / CGPEquation exposing exactly what the
# updater reads: tree.n_features, tree.active_nodes, tree.nodes[i].op,
# tree.update_active_nodes(), plus ind.fitness / ind.loss / ind.push_intrinsic.
# --------------------------------------------------------------------------- #
class _FakeNode:
    __slots__ = ("op",)
    def __init__(self, op):
        self.op = op


class _FakeTree:
    def __init__(self, ops, n_features=2):
        self.n_features = n_features
        self.nodes = [_FakeNode(op) for op in ops]
        self.active_nodes = [n_features + i for i in range(len(ops))]
    def update_active_nodes(self):
        pass


class _FakeInd:
    def __init__(self, ops, fitness, n_features=2):
        self.tree = _FakeTree(ops, n_features)
        self.fitness = float(fitness)
        self.loss = float(max(fitness, 1e-6))
        self.push_intrinsic = 0.0


def _make_population(elite_ops, loser_ops, n_each=10):
    """Elite (low fitness) use `elite_ops`; losers (high fitness) use `loser_ops`."""
    pop = []
    for i in range(n_each):
        pop.append(_FakeInd(list(elite_ops), fitness=0.1 + 0.001 * i))
    for i in range(n_each):
        pop.append(_FakeInd(list(loser_ops), fitness=5.0 + 0.001 * i))
    return pop


def _with_enabled(fn):
    saved = evo13.ADAPTIVE_COSTS_ENABLED
    try:
        evo13.ADAPTIVE_COSTS_ENABLED = True
        return fn()
    finally:
        evo13.ADAPTIVE_COSTS_ENABLED = saved


def _with_allowed(ops, fn):
    saved = list(evo13.ALLOWED_OPS)
    try:
        evo13.ALLOWED_OPS = list(ops)
        return fn()
    finally:
        evo13.ALLOWED_OPS = saved


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_disabled_recovers_static_table():
    """With the flag off, op_cost is exactly the authored OP_COSTS lookup."""
    saved = evo13.ADAPTIVE_COSTS_ENABLED
    try:
        evo13.ADAPTIVE_COSTS_ENABLED = False
        for op, c in evo13.OP_COSTS.items():
            assert evo13.op_cost(op) == c, op
        # Unknown operator falls back to the same default as before.
        assert evo13.op_cost("__nope__") == evo13.COST_OP_COMPLEX
    finally:
        evo13.ADAPTIVE_COSTS_ENABLED = saved


def test_fresh_table_equals_prior():
    """A just-constructed table prices every operator at its authored cost."""
    ac = evo13.AdaptiveOperatorCosts()
    for op, c in evo13.OP_COSTS.items():
        assert ac.effective_cost(op) == c, op
    # An operator unknown at construction is lazily adopted at its OP_COSTS value.
    assert ac.effective_cost("__unseen__") == evo13.COST_OP_COMPLEX


def test_small_population_is_noop():
    """Too few finite-fitness individuals ⇒ the table must not move."""
    def check():
        ac = evo13.AdaptiveOperatorCosts()
        before = dict(ac._cost)
        pop = [_FakeInd(["sin"], 1.0) for _ in range(3)]  # < ADAPTIVE_COST_MIN_POP
        assert ac.observe_and_update(pop) is False
        assert ac._cost == before
    _with_enabled(check)


def test_useful_operator_gets_cheaper_avoided_pricier():
    """The headline behaviour: an operator the elite lean on gets cheaper, while
    one only the losers use gets pricier — both staying inside the prior band."""
    def check():
        def inner():
            ac = evo13.AdaptiveOperatorCosts()
            c0_sin = ac._prior["sin"]
            c0_exp = ac._prior["exp"]
            pop = _make_population(elite_ops=["sin", "sin", "sin", "+"],
                                   loser_ops=["exp", "exp", "exp", "+"])
            moved = False
            for _ in range(12):
                moved = ac.observe_and_update(pop) or moved
            assert moved, "update never fired"
            assert ac.effective_cost("sin") < c0_sin, "useful op did not get cheaper"
            assert ac.effective_cost("exp") > c0_exp, "avoided op did not get pricier"
            # Band invariant holds for both.
            assert ac.effective_cost("sin") >= evo13.ADAPTIVE_COST_MIN_RATIO * c0_sin - 1e-9
            assert ac.effective_cost("exp") <= evo13.ADAPTIVE_COST_MAX_RATIO * c0_exp + 1e-9
        _with_allowed(["+", "-", "*", "/", "sin", "exp", "log", "const"], inner)
    _with_enabled(check)


def test_prices_stay_in_band():
    """Even under many aggressive same-direction updates nothing escapes the
    [MIN_RATIO, MAX_RATIO]·prior band (no collapse to 0, no blow-up)."""
    def check():
        def inner():
            ac = evo13.AdaptiveOperatorCosts()
            pop = _make_population(elite_ops=["sin"] * 8,
                                   loser_ops=["exp"] * 8)
            for _ in range(200):
                ac.observe_and_update(pop)
            lo, hi = evo13.ADAPTIVE_COST_MIN_RATIO, evo13.ADAPTIVE_COST_MAX_RATIO
            for op, c0 in ac._prior.items():
                if c0 > 0:
                    c = ac._cost[op]
                    assert lo * c0 - 1e-9 <= c <= hi * c0 + 1e-9, (op, c, c0)
        _with_allowed(["+", "-", "*", "/", "sin", "exp", "log", "const"], inner)
    _with_enabled(check)


def test_mean_preserved():
    """Only RELATIVE prices move: the mean over the in-play vocabulary stays
    pinned to the prior mean, so the global complexity scale is conserved."""
    def check():
        def inner():
            vocab = ["+", "*", "sin", "exp", "log"]
            ac = evo13.AdaptiveOperatorCosts()
            mean_prior = sum(ac._prior[o] for o in vocab) / len(vocab)
            pop = _make_population(elite_ops=["sin", "sin", "+"],
                                   loser_ops=["exp", "log", "*"])
            for _ in range(15):
                ac.observe_and_update(pop)
            mean_now = sum(ac._cost[o] for o in vocab) / len(vocab)
            assert abs(mean_now - mean_prior) <= 0.02 * mean_prior, (mean_now, mean_prior)
        _with_allowed(["+", "*", "sin", "exp", "log", "const"], inner)
    _with_enabled(check)


def test_reset_restores_prior():
    """reset() must return every price to its authored value."""
    def check():
        ac = evo13.AdaptiveOperatorCosts()
        pop = _make_population(elite_ops=["sin", "sin"], loser_ops=["exp", "exp"])
        for _ in range(5):
            ac.observe_and_update(pop)
        ac.reset()
        assert ac.n_updates == 0
        for op, c in ac._prior.items():
            assert ac._cost[op] == c, op
    _with_enabled(check)


def test_op_cost_routes_through_live_singleton():
    """op_cost reflects the live singleton's prices when enabled."""
    saved_flag = evo13.ADAPTIVE_COSTS_ENABLED
    saved_obj = evo13.ADAPTIVE_OP_COST
    try:
        evo13.ADAPTIVE_COSTS_ENABLED = True
        evo13.ADAPTIVE_OP_COST = evo13.AdaptiveOperatorCosts()
        evo13.ADAPTIVE_OP_COST._cost["sin"] = 999.0
        assert evo13.op_cost("sin") == 999.0
    finally:
        evo13.ADAPTIVE_COSTS_ENABLED = saved_flag
        evo13.ADAPTIVE_OP_COST = saved_obj


def test_summary_is_stringy():
    """summary() is a harmless human-readable digest in both states."""
    ac = evo13.AdaptiveOperatorCosts()
    assert isinstance(ac.summary(), str)
    def check():
        pop = _make_population(elite_ops=["sin", "sin"], loser_ops=["exp", "exp"])
        for _ in range(5):
            ac.observe_and_update(pop)
        assert isinstance(ac.summary(), str)
    _with_enabled(check)


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
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} passed")
    sys.exit(1 if failures else 0)
