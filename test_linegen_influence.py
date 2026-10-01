"""linegen_gui Sampler.influence: the occlusion attribution (each token replaced by random other tokens,
every position unchanged).

Run with the pytorch2 env:  python test_linegen_influence.py"""
import unittest

import torch

import linegen as lg
import linegen_gui as lgg
from linegenModel import MarkovBigramLM, ModernTransformer


def sampler_for(model, seq_len=64):
    s = lgg.Sampler()
    s.model = model.to(lg.DEVICE).eval()
    s.cfg = {"model_selection": -1, "seq_len": seq_len, "dataset_type": 0}
    s.vocab = None
    s.piece = lambda i: str(i)
    s.embedding = next(m for m in model.modules() if isinstance(m, torch.nn.Embedding))
    return s


def logp_after(model, ids, pos, target):
    with torch.no_grad():
        out = model(torch.tensor([ids], device=lg.DEVICE)).float()
    return float(torch.log_softmax(out[0, pos], -1)[target])


class OcclusionTests(unittest.TestCase):
    def test_bigram_only_the_current_token_matters(self):
        torch.manual_seed(0); V = 20
        m = MarkovBigramLM(V); torch.nn.init.normal_(m.transitions.weight)
        s = sampler_for(m)
        ids = torch.randint(0, V, (12,)).tolist(); pos = 7
        r = s.influence({"ids": ids, "position": pos})
        self.assertIsNone(r["occlusion_error"])
        occ, target = r["occlusion"], r["target"]
        for j in range(pos):  # the prediction after `pos` reads token `pos` only: replacing an earlier one changes nothing
            self.assertAlmostEqual(occ[j], 0.0, places=5)
        reps = r["occlusion_replacements"][pos]
        self.assertNotIn(ids[pos], reps)
        base = logp_after(m, ids, pos, target)
        expected = base - sum(logp_after(m, ids[:pos] + [t] + ids[pos + 1:], pos, target) for t in reps) / len(reps)
        self.assertAlmostEqual(occ[pos], expected, places=4)
        self.assertTrue(all(v == 0 for v in occ[pos + 1:]))

    def test_transformer_matches_rerunning_with_each_replacement(self):
        torch.manual_seed(0); V = 30
        m = ModernTransformer(V, 32, 2, 4)
        s = sampler_for(m)
        ids = torch.randint(0, V, (16,)).tolist(); pos = 11
        r = s.influence({"ids": ids, "position": pos})
        base = logp_after(m, ids, pos, r["target"])
        for j in (0, 5, 10, 11):
            reps = r["occlusion_replacements"][j]
            expected = base - sum(logp_after(m, ids[:j] + [t] + ids[j + 1:], pos, r["target"]) for t in reps) / len(reps)
            self.assertAlmostEqual(r["occlusion"][j], expected, places=4)
        self.assertEqual(len(r["ablation"]), len(r["occlusion"]))
        self.assertEqual(s.influence({"ids": ids, "position": pos})["occlusion"], r["occlusion"])  # same replacements every time

    def test_first_position_works(self):
        m = MarkovBigramLM(10); torch.nn.init.normal_(m.transitions.weight); s = sampler_for(m)
        r = s.influence({"ids": [1, 2, 3], "position": 0})
        self.assertIsNone(r["occlusion_error"]); self.assertNotEqual(r["occlusion"][0], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
