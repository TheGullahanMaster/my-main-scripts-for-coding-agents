"""inenc / outdec columns (mlpres_seq), the outexcat padding class, and the feature-token input attention.

Run with the pytorch2 env:  python test_seq_columns.py
The training tests run mlpRes6.main() on small generated datasets in a temporary directory."""
import json
import os
import random
import tempfile
import unittest

import torch  # before pandas: in the pytorch2 env pandas otherwise loads the system's older libstdc++
from torch import nn
import numpy as np
import pandas as pd

import mlpRes6 as M
import mlpres_seq as S

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ONES = "zero one two three four five six seven eight nine".split()
TEENS = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def words(n):
    if n < 10: return ONES[n]
    if n < 20: return TEENS[n - 10]
    if n < 100: return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + words(n % 100))


def train(csv, col_types, seq_params, steps, seq_bridge="mlp", hidden=(128,), lr=2e-3, batch=64, **kw):
    """mlpRes6.main() for `steps` steps on every row (no validation); returns an InteractiveSampler of the result."""
    vocab, imgs = {}, {}
    for c, t in col_types.items():
        if t != "i": M._setup_vocab_for_col(c, t, csv, ",", vocab, imgs, seq_params.get(c, {}).get("tokenizer", "char"))
    ins = [c for c, t in col_types.items() if "in" in t]; outs = [c for c, t in col_types.items() if "out" in t]
    ds, _, _ = M.prepare_train_val(csv, ",", ins, outs, col_types, vocab, imgs)
    stop = lambda info: info.get("step", 0) >= steps
    M.main(csv, ",", ins, outs, col_types, ds.vocabularies, ds.scalings, imgs, optimizer_choice="Adam",
           hidden_dims=list(hidden), batch_size=batch, activation_cls=nn.GELU, custom_lr=lr, progress_callback=stop,
           train_dataset=ds, seq_params=seq_params, seq_bridge=seq_bridge, **kw)
    return M.InteractiveSampler("model.pt", "config.json")


def predictions(sampler, df, in_cols, out_col):
    got = []
    for _, row in df.iterrows():
        outs = sampler.predict({c: row[c] for c in in_cols})["outputs"]
        got.append(next(o["value"] for o in outs if o["col"] == out_col))
    return got


class InTempDir(unittest.TestCase):
    def setUp(self):
        self.old = os.getcwd(); self.tmp = tempfile.TemporaryDirectory(); os.chdir(self.tmp.name)
        torch.manual_seed(0); random.seed(0); np.random.seed(0)

    def tearDown(self):
        os.chdir(self.old); self.tmp.cleanup()


class ModelTests(unittest.TestCase):
    def test_every_encoder_decoder_pair_runs_and_decoders_are_causal(self):
        vc = S.build_vocab(["hello world", "abc"], "char"); V = S.vocab_size(vc); Ti, To = 7, 6
        layout = [{"col": "o", "type": "out", "start": 0, "end": 1, "tgt_start": 0, "tgt_end": 1},
                  {"col": "t", "type": "outdec", "start": 1, "end": 1 + To * V, "tgt_start": 1, "tgt_end": 1 + To,
                   "num_classes": V, "max_len": To}]
        ct = {"n": "in", "s": "inenc", "o": "out", "t": "outdec"}
        mk = lambda i, o: nn.Sequential(nn.Linear(i, 32), nn.ReLU(), nn.Linear(32, o))
        x = torch.zeros(3, 1 + Ti, device=DEV); x[:, 1:5] = torch.randint(4, V, (3, 4)).float()
        y = torch.zeros(3, 1 + To, device=DEV); y[:, 1:4] = torch.randint(4, V, (3, 3)).float(); y[:, 4] = S.EOS
        for ea in S.ARCHS:
            for da in S.ARCHS:
                for att in (S.DEC_ATTENTION if da in S.ATTENTION_ARCHS else ["none"]):
                    with self.subTest(encoder=ea, decoder=da, attention=att):
                        sp = {"s": {"arch": ea, "dim": 32, "heads": 4}, "t": {"arch": da, "dim": 32, "heads": 4, "attention": att}}
                        m = S.SeqMLP([("n", 0, 1), ("s", 1, Ti)], ct, {"s": vc, "t": vc}, layout, sp, "mlp", mk).to(DEV)
                        tf = m(x, y); tf.sum().backward()
                        m.eval(); self.assertEqual(m(x).shape, tf.shape); m.train()
                        if da != "mlp":  # target token 3 feeds positions >= 4 only
                            y2 = y.clone(); y2[:, 4] = 7
                            a, b = (m(x, t)[:, 1:].view(3, To, V) for t in (y, y2))
                            self.assertTrue(torch.allclose(a[:, :4], b[:, :4], atol=1e-5))

    def test_stateful_generation_matches_prefix_rerun(self):
        """Per-type stepping (RNN / M2RNN state, KV cache, TCN window) must produce what re-running the prefix does."""
        vc = S.build_vocab(["hello world", "abc"], "char"); V = S.vocab_size(vc); Ti, To = 7, 20
        layout = [{"col": "t", "type": "outdec", "start": 0, "end": To * V, "tgt_start": 0, "tgt_end": To, "num_classes": V, "max_len": To}]
        ct = {"s": "inenc", "t": "outdec"}; mk = lambda i, o: nn.Sequential(nn.Linear(i, 32), nn.ReLU(), nn.Linear(32, o))
        x = torch.zeros(5, Ti, device=DEV); x[:, :5] = torch.randint(4, V, (5, 5)).float()
        for da in S.DECODERS:
            for att in (S.DEC_ATTENTION if da in S.ATTENTION_ARCHS else ["none"]):
                with self.subTest(decoder=da, attention=att):
                    sp = {"s": {"arch": "gru", "dim": 32}, "t": {"arch": da, "dim": 32, "heads": 4, "layers": 2, "attention": att}}
                    m = S.SeqMLP([("s", 0, Ti)], ct, {"s": vc, "t": vc}, layout, sp, "mlp", mk).to(DEV).eval()
                    head = m.decoders["t"].head
                    with torch.no_grad():
                        nn.init.normal_(head.weight, std=0.5); head.bias[S.EOS] = -1e4  # generate all 20 steps
                        m.set_generation(mode="prefix"); a = m(x); m.set_generation(mode="auto"); b = m(x)
                    self.assertTrue(torch.equal(a.view(5, To, V).argmax(-1), b.view(5, To, V).argmax(-1)))
                    # relative: the blocked <eos> logit sits near -1e4, where one float32 rounding step is ~1e-3
                    self.assertTrue(torch.allclose(a, b, rtol=1e-5, atol=1e-4))
                    if da == "tcn": self.assertLess(m.decoders["t"].core.receptive_field, To)  # the window really slides

    def test_sampling(self):
        lg = torch.tensor([[0.0, 1.0, 5.0, 2.0]]).repeat(2000, 1)
        self.assertTrue((S.sample_tokens(lg, 0.0) == 2).all())
        self.assertEqual(set(S.sample_tokens(lg, 1.0, top_k=2).tolist()), {2, 3})
        self.assertEqual(set(S.sample_tokens(lg, 1.0, top_p=0.5).tolist()), {2})  # token 2 alone has p > 0.5
        self.assertEqual(len(set(S.sample_tokens(lg, 5.0).tolist())), 4)
        with self.assertRaises(ValueError): S.check_generation({"top_p": 0})

    def test_all_mlp_is_a_plain_mlp_bridge(self):
        vc = S.build_vocab(["ab"], "char"); V = S.vocab_size(vc)
        layout = [{"col": "t", "type": "outdec", "start": 0, "end": 3 * V, "tgt_start": 0, "tgt_end": 3, "num_classes": V, "max_len": 3}]
        made = []
        mk = lambda i, o: made.append((i, o)) or nn.Linear(i, o)
        m = S.SeqMLP([("s", 0, 2)], {"s": "inenc", "t": "outdec"}, {"s": vc, "t": vc}, layout,
                     {"s": {"arch": "mlp", "dim": 8}, "t": {"arch": "mlp"}}, "latent", mk)
        self.assertFalse(m.needs_targets); self.assertEqual(m.bridge_kind, "mlp"); self.assertEqual(made, [(16, 3 * V)])

    def test_word_tokenizer_round_trip(self):
        v = S.build_vocab(["The cat sat.", "a dog"], "word")
        ids = S.encode_ids("The dog sat.", v, 6, True)
        self.assertEqual(ids[4], S.EOS); self.assertEqual(S.decode_ids(ids, v), "The dog sat .")
        self.assertEqual(S.decode_ids(S.encode_ids("zebra", v, 3, True), v), "<unk>")


class ExplainTests(unittest.TestCase):
    def test_attention_rows_sum_to_one_and_unused_inputs_get_no_attribution(self):
        vc = S.build_vocab(["hello world", "abc"], "char"); V = S.vocab_size(vc); Ti, To = 7, 8
        layout = [{"col": "o", "type": "out", "start": 0, "end": 1, "tgt_start": 0, "tgt_end": 1},
                  {"col": "t", "type": "outdec", "start": 1, "end": 1 + To * V, "tgt_start": 1, "tgt_end": 1 + To, "num_classes": V, "max_len": To}]
        ct = {"n": "in", "m": "in", "s": "inenc", "o": "out", "t": "outdec"}
        x = torch.zeros(1, 2 + Ti, device=DEV); x[0, :2] = torch.tensor([0.7, -0.4]); x[0, 2:6] = torch.randint(4, V, (4,)).float()
        for ea in ("gru", "transformer", "modern"):
            for da in ("gru", "m2rnn", "transformer", "tcn", "modern"):
                att = "multihead" if da in S.ATTENTION_ARCHS else "none"
                with self.subTest(encoder=ea, decoder=da):
                    sp = {"s": {"arch": ea, "dim": 32, "heads": 4}, "t": {"arch": da, "dim": 32, "heads": 4, "attention": att}}
                    m = S.SeqMLP([("n", 0, 1), ("m", 1, 1), ("s", 2, Ti)], ct, {"s": vc, "t": vc}, layout, sp, "latent", None).to(DEV)
                    with torch.no_grad(): m.bridge.weight[:, 1] = 0; m.decoders["t"].head.bias[S.EOS] = -1e4  # input "m" is cut off
                    ex = S.explain(m, x); steps = ex["decoders"]["t"]["attr"]
                    self.assertEqual(len(steps), To)
                    self.assertTrue(all(a["__numeric__"][1] == 0 for a in steps)); self.assertEqual(ex["regular"]["o"]["__numeric__"][1], 0)
                    self.assertTrue(any(a["__numeric__"][0] != 0 for a in steps))
                    self.assertTrue(ex["attention"])
                    for r in ex["attention"]: self.assertTrue(torch.allclose(r["weights"].sum(-1), torch.ones(1), atol=1e-4), r["module"])


class TextFitTests(unittest.TestCase):
    def test_edit_distance_alignment_and_scores(self):
        d, ops = M.levenshtein_ops(list("kitten"), list("sitting"))
        self.assertEqual(d, 3); self.assertEqual(sum(o != "match" for o, _, _ in ops), 3)
        stats, err = M.text_fit_stats([list("abc"), list("abcd"), list("ab")], [list("abc"), list("abd"), list("abx")])
        self.assertAlmostEqual(stats["exact"], 1 / 3); self.assertEqual(list(err), [0.0, 0.25, 0.5])
        self.assertAlmostEqual(stats["error_rate"], 2 / 9)
        ix = {l: i for i, l in enumerate(stats["labels"])}
        self.assertEqual(stats["confusion"][ix["c"]][ix["∅"]], 1)  # "c" missing from "abd"
        self.assertEqual(stats["confusion"][ix["∅"]][ix["x"]], 1)  # extra "x"
        self.assertEqual(stats["pos_acc"][:2], [1.0, 1.0])


class ColumnAndMlpOptionTests(InTempDir):
    def test_headerless_renamed_ranges_layer_and_output_activations_zero_layers(self):
        rng = np.random.default_rng(0); x = rng.uniform(0, 10, 400)
        with open("nohdr.csv", "w") as f: f.write("".join(f"{a:.4f},{'ab'[int(a) % 2]},{a * a:.4f}\n" for a in x))
        with self.assertRaises(ValueError): M.csv_format(False, {"col1": "y", "col3": "y"}, ["col1", "col2", "col3"])
        M.register_csv_format("nohdr.csv", M.csv_format(False, {"col1": "x", "col3": "y"}, ["col1", "col2", "col3"]))
        self.assertEqual(list(M.read_table("nohdr.csv", ",").columns), ["x", "col2", "y"]); self.assertEqual(len(M.read_table("nohdr.csv", ",")), 400)
        ct = {"x": "in", "col2": "i", "y": "out"}
        ds, vds, _ = M.prepare_train_val("nohdr.csv", ",", ["x"], ["y"], ct, {}, {}, val_frac=0.1, scale_ranges={"y": [0, 1]})
        sc = ds.scalings["y"]; self.assertEqual(sc["range"], [0.0, 1.0]); self.assertAlmostEqual(M.to_display_units(M.to_model_units(37.5, sc), sc), 37.5)
        t = [float(ds[i][1][0]) for i in range(len(ds))]; self.assertAlmostEqual(min(t), 0.0, places=5); self.assertAlmostEqual(max(t), 1.0, places=5)
        vl = torch.utils.data.DataLoader(vds, batch_size=64)

        def run(hidden, **kw):
            M.main("nohdr.csv", ",", ["x"], ["y"], ct, {}, ds.scalings, {}, hidden_dims=hidden, activation_cls=nn.ReLU, custom_lr=3e-3,
                   val_loader=vl, val_interval=100, progress_callback=lambda i: i.get("step", 0) >= 800, train_dataset=ds, **kw)
            smp = M.InteractiveSampler("model.pt", "config.json")
            return smp, [smp.predict({"x": v})["outputs"][0]["value"] for v in (1.0, 5.0, 9.0)]
        smp, preds = run([64, 32], layer_activations=[{"name": "Tanh", "params": {}}, {"name": "GELU", "params": {}}],
                         output_activation={"name": "Sigmoid", "params": {}})
        self.assertEqual([type(b.activation).__name__ for b in smp.model.blocks], ["Tanh", "GELU"])
        self.assertIsInstance(smp.model.output_act, nn.Sigmoid)
        for p, true in zip(preds, (1, 25, 81)): self.assertLess(abs(p - true), 6)
        with open("config.json") as f: cfg = json.load(f)
        self.assertEqual(cfg["csv_format"], {"header": False, "rename": {"col1": "x", "col3": "y"}})
        kwargs, _, _ = M.prepare_resume("config.json", "model.pt")
        self.assertEqual(kwargs["layer_activations"][1]["name"], "GELU"); self.assertEqual(len(kwargs["train_dataset"]), 360)
        smp, preds = run([])  # perceptron: a straight line through x
        self.assertEqual(len(smp.model.blocks), 0)
        self.assertAlmostEqual(preds[1] - preds[0], preds[2] - preds[1], delta=1e-3 * max(1.0, abs(preds[2])))

    def test_zero_layers_with_encoder_and_decoder(self):
        rows = sorted({"".join(random.choice("abc") for _ in range(random.randint(2, 4))) for _ in range(200)})
        pd.DataFrame({"src": rows, "tgt": [r[::-1] for r in rows]}).to_csv("z.csv", index=False)
        s = train("z.csv", {"src": "inenc", "tgt": "outdec"}, {"src": {"arch": "gru", "dim": 32}, "tgt": {"arch": "gru", "dim": 32}}, 50, hidden=())
        self.assertEqual(len(s.model.blocks), 0)
        self.assertIsInstance(s.predict({"src": "abc"})["outputs"][0]["value"], str)


class CliPromptTests(InTempDir):
    def test_new_cli_questions(self):
        from unittest import mock
        with open("h.csv", "w") as f: f.write("1,a,2\n3,b,4\n5,a,6\n")
        answers = iter(["n", "y", "x", "", "y"])  # no header row; rename col1 -> x, keep col2, col3 -> y
        with mock.patch("builtins.input", lambda *_: next(answers)):
            fmt = M.ask_csv_format("h.csv", ",")
        self.assertEqual(fmt, {"header": False, "rename": {"col1": "x", "col3": "y"}, "delimiter": ","})
        answers = iter(["y", "2", "", "y", "1", "y", "0 1", ""])  # per-layer: Tanh, global; sigmoid output; ranges
        with mock.patch("builtins.input", lambda *_: next(answers)):
            layers = M.ask_layer_activations([8, 8], nn.ReLU)
            out = M.ask_output_activation(["y"], {"y": "out"})
            ranges = M.ask_scale_ranges(["x", "y"], {"x": "in", "y": "out"})
        self.assertEqual([l["name"] for l in layers], ["Tanh", "ReLU"])
        self.assertEqual(out["name"], "Sigmoid"); self.assertEqual(ranges, {"x": [0.0, 1.0]})


class DelimiterTests(InTempDir):
    def test_headerless_tab_file_of_an_older_config_is_found(self):
        with open("d.tsv", "w") as f: f.write("david\tdavídek\nauto\tautíčko\n")
        ct = {"name": "inenc", "namecek": "outdec"}
        old_fmt = {"header": False, "rename": {"col1": "name", "col2": "namecek"}}  # saved before delimiters were recorded
        self.assertEqual(M.training_file_delimiter("d.tsv", ct, old_fmt), "\t")
        self.assertEqual(M.training_file_delimiter("d.tsv", ct, dict(old_fmt, delimiter=";")), ";")  # a recorded one wins
        with open("h.csv", "w") as f: f.write("a;b;c\n1;2;3\n")
        self.assertEqual(M.training_file_delimiter("h.csv", {"a": "in", "c": "out", "b": "i"}), ";")


class OutexcatTests(InTempDir):
    def test_last_character_is_a_valid_class_and_padding_is_class_zero(self):
        pd.DataFrame({"a": [1, 2, 3], "b": ["ab", "bc", "c"]}).to_csv("t.csv", index=False)
        voc = {}; M._setup_vocab_for_col("b", "outexcat", "t.csv", ",", voc, {})
        ct = {"a": "in", "b": "outexcat"}
        ds = M.CustomDataset("t.csv", ",", ["a"], ["b"], ct, voc, {}, {})
        lay, pdim, _ = M.build_output_layout(["b"], ct, ds.scalings, voc)
        self.assertEqual(lay[0]["num_classes"], len(voc["b"]) + 1)
        t = torch.stack([ds[i][1] for i in range(3)])
        crit = M.CombinedLoss(lay, column_weights=ds.compute_class_weights()); crit.calibrate(ds)
        self.assertTrue(torch.isfinite(crit(torch.zeros(3, pdim), t)))
        logits = torch.full((2, lay[0]["num_classes"]), -9.0); logits[0, voc["b"]["c"]] = 9; logits[1, 0] = 9  # "c" then padding
        inv = {v: k for k, v in voc["b"].items()}
        self.assertEqual("".join(inv.get(i, "") for i in logits.argmax(-1).tolist()), "c")


class InputAttentionTests(unittest.TestCase):
    def test_attention_compares_features_and_supports_any_width(self):
        for mode in ("basic", "cross"):
            with self.subTest(mode=mode):
                m = M.MLPO(5, [16], 1, nn.ReLU, input_attention_type=mode, num_heads=2)
                x = torch.randn(4, 5)
                self.assertTrue(torch.allclose(m.input_attn(x), x))  # zero-initialised update: starts as the plain MLP
                nn.init.normal_(m.input_attn.out.weight)
                a = m(x); m.input_attn.feature.data.normal_(); self.assertFalse(torch.allclose(a, m(x)))
                M.lsuv_init(m, [(x, torch.zeros(4, 1))], "cpu", verbose=False)
                self.assertEqual(m(x[0]).shape, (1,))


class TrainingTests(InTempDir):
    def test_reverse_strings_gru_with_attention(self):
        rows = ["".join(random.choice("abcde") for _ in range(random.randint(3, 6))) for _ in range(600)]
        df = pd.DataFrame({"src": rows, "tgt": [r[::-1] for r in rows]}).drop_duplicates("src"); df.to_csv("rev.csv", index=False)
        s = train("rev.csv", {"src": "inenc", "tgt": "outdec"},
                  {"src": {"arch": "gru", "dim": 128}, "tgt": {"arch": "gru", "dim": 128, "attention": "additive"}}, 1500)
        got = predictions(s, df.head(200), ["src"], "tgt")
        acc = np.mean([g == t for g, t in zip(got, df.head(200)["tgt"])]); print(f"reverse exact match {acc:.3f}")
        self.assertGreater(acc, 0.9)
        # Sequence view: generating the k-th output token should look at input position n-1-k (reversal)
        hits = total = 0
        for src in df.head(30)["src"]:
            v = s.sequence_view({"src": src}); cross = [m for m in v["maps"] if m["kind"] == "cross"][0]
            w = np.asarray(cross["w"]).mean(0); n = len(src)
            for k in range(min(n, len(w))): hits += int(np.argmax(w[k][:n]) == n - 1 - k); total += 1
            self.assertEqual(len(v["decoders"]["tgt"]["attr"]), len(v["decoders"]["tgt"]["tokens"]))
        print(f"reversal attention on the mirrored token: {hits / total:.3f}"); self.assertGreater(hits / total, 0.8)
        fit = s.data_stats()["fit"][0]; self.assertEqual(fit["kind"], "text"); self.assertGreater(fit["exact"], 0.9)

    def test_number_to_words_modern_decoder_and_resume(self):
        df = pd.DataFrame({"n": range(1000), "w": [words(n) for n in range(1000)]}); df.to_csv("num.csv", index=False)
        # the digits as text: a single scaled number would make "ones digit" a rapidly oscillating function
        s = train("num.csv", {"n": "inenc", "w": "outdec"},
                  {"n": {"arch": "gru", "dim": 64}, "w": {"arch": "modern", "dim": 128, "heads": 4, "tokenizer": "word"}}, 3000)
        sample = df.sample(200, random_state=0)
        acc = np.mean([g == t for g, t in zip(predictions(s, sample, ["n"], "w"), sample["w"])]); print(f"number->words exact match {acc:.3f}")
        self.assertGreater(acc, 0.9)
        with open("config.json") as f: cfg = json.load(f)
        self.assertEqual(cfg["seq_params"]["w"]["arch"], "modern")
        kwargs, _, _ = M.prepare_resume("config.json", "model.pt")
        self.assertEqual(kwargs["seq_params"]["w"]["tokenizer"], "word")

    def test_text_to_score_and_label_transformer_encoder(self):
        pos, neg, fill = ["good", "great", "fine"], ["bad", "awful", "poor"], ["the", "food", "was", "and", "service", "it"]
        rows = []
        for _ in range(1500):
            ws = [random.choice(pos + neg + fill) for _ in range(random.randint(3, 10))]
            sc = sum(w in pos for w in ws) - sum(w in neg for w in ws)
            rows.append((" ".join(ws), sc, "positive" if sc > 0 else "negative" if sc < 0 else "neutral"))
        df = pd.DataFrame(rows, columns=["review", "score", "label"]).drop_duplicates("review"); df.to_csv("rev.csv", index=False)
        s = train("rev.csv", {"review": "inenc", "score": "out", "label": "outlabcat"},
                  {"review": {"arch": "transformer", "dim": 64, "heads": 4, "tokenizer": "word"}}, 2000)
        sample = df.head(300)
        outs = [s.predict({"review": r})["outputs"] for r in sample["review"]]
        score = np.array([next(o["value"] for o in out if o["col"] == "score") for out in outs])
        label = [next(o["value"] for o in out if o["col"] == "label") for out in outs]
        r2 = 1 - np.sum((score - sample["score"]) ** 2) / np.sum((sample["score"] - sample["score"].mean()) ** 2)
        acc = np.mean([a == b for a, b in zip(label, sample["label"])]); print(f"text->score R2 {r2:.3f}, label acc {acc:.3f}")
        self.assertGreater(r2, 0.9); self.assertGreater(acc, 0.9)
        v = s.sequence_view({"review": "the food was great and good"})
        self.assertTrue(any(m["side"] == "encoder" and m["kind"] == "self" for m in v["maps"]))  # transformer encoder self-attention
        att = v["regular"]["score"]["review"]; toks = v["encoders"]["review"]["tokens"]
        self.assertEqual(len(att), len(toks))
        strongest = toks[int(np.argmax(np.abs(att)))]; print("score attribution peaks at", strongest)
        self.assertIn(strongest, ("great", "good"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
