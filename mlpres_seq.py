"""Sequence columns for mlpRes6: `inenc` (text read by an encoder) and `outdec` (text written by a decoder).

Data stays in mlpRes6's flat format: an `inenc` column fills `max_len` slots of the input vector with
token ids, an `outdec` column fills `max_len` target slots with token ids ending in <eos>, 0 = padding.
SeqMLP slices those ids back out, runs one encoder per `inenc` column, feeds the pooled encodings and
the ordinary inputs through a bridge (mlpRes6's MLPO, or a plain linear "latent" projection), and lets
one decoder per `outdec` column generate its text from the bridge's latent, optionally attending to
every encoder's per-token memory. Its output is the usual flat prediction vector, so an `outdec`
column's logits sit where an `outexcat` column's would (max_len x vocab).

Training passes the targets (teacher forcing); without targets the decoders generate greedily.
MLP encoder = shared token embeddings per position, flattened into the bridge. MLP decoder = the
bridge predicts all positions at once (non-causal, like outexcat). MLP on both sides therefore is the
plain MLPO, only with shared embeddings instead of per-position one-hots.
"""
import math
import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from linegenModel import RMSNorm, SwiGLU, RotaryEmbedding, apply_rotary_pos_emb, TCNBlock, M2RNNLayer

PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPECIALS = {"<pad>": PAD, "<unk>": UNK, "<bos>": BOS, "<eos>": EOS}
SEQ_TYPES = ("inenc", "outdec")
TOKENIZERS = ("char", "word")
ARCHS = {
    "mlp": "MLP (shared embeddings)", "rnn": "RNN (tanh, cuDNN)", "rnn_relu": "RNN (ReLU, cuDNN)",
    "gru": "GRU (cuDNN)", "lstm": "LSTM (cuDNN)", "m2rnn": "M2RNN (matrix-state RNN)",
    "transformer": "Transformer (original, 2017)", "tcn": "Temporal ConvNet", "modern": "Modern (LLaMA-style: RoPE, RMSNorm, SwiGLU)",
}
RNN_ARCHS = ("rnn", "rnn_relu", "gru", "lstm")
DEC_ATTENTION = {"none": "None", "additive": "Additive (Bahdanau score)", "multihead": "Multi-head"}
ATTENTION_ARCHS = RNN_ARCHS + ("m2rnn", "tcn")  # decoders whose cross-attention is optional
DEFAULT_PARAMS = {"tokenizer": "char", "arch": "gru", "dim": 128, "layers": 2, "heads": 4, "attention": "none"}
BRIDGES = {"mlp": "MLP (the network's hidden layers)", "latent": "Pure latent (one linear map)"}
MAX_WORD_VOCAB = 30000
_TOKENIZER_PREFIX = "<tokenizer:"  # vocabulary entry recording how the column was tokenized (id -1, never produced)


# ─────────────────────────── tokenizers / vocabularies ───────────────────────────
def tokenize(text, mode):
    text = str(text)
    if mode == "word": return re.findall(r"\w+|[^\w\s]", text)
    return list(text)


def build_vocab(texts, mode):
    if mode not in TOKENIZERS: raise ValueError(f"Unknown tokenizer {mode!r}; use one of {TOKENIZERS}")
    counts = Counter(tok for t in texts for tok in tokenize(t, mode))
    vocab = dict(SPECIALS); vocab[f"{_TOKENIZER_PREFIX}{mode}>"] = -1
    keep = counts.most_common(MAX_WORD_VOCAB) if mode == "word" else sorted(counts.items())
    next_id = len(SPECIALS)
    for tok, _ in keep:
        if tok not in vocab: vocab[tok] = next_id; next_id += 1
    return vocab


def tokenizer_of(vocab):
    for k in vocab:
        if k.startswith(_TOKENIZER_PREFIX): return k[len(_TOKENIZER_PREFIX):-1]
    return "char"


def vocab_size(vocab):
    return max(v for v in vocab.values()) + 1


def token_count(text, vocab, is_output):
    return len(tokenize(text, tokenizer_of(vocab))) + (1 if is_output else 0)


def encode_ids(text, vocab, max_len, is_output):
    """Token ids padded with PAD to max_len; outputs end in EOS (kept even when the text is truncated)."""
    ids = [vocab.get(t, UNK) for t in tokenize(text, tokenizer_of(vocab))]
    ids = ids[:max_len - 1] + [EOS] if is_output else ids[:max_len]
    return ids + [PAD] * (max_len - len(ids))


def decoded_tokens(ids, vocab):
    """Generated ids -> tokens up to <eos> (padding / <bos> skipped, <unk> kept as "<unk>")."""
    inv = {v: k for k, v in vocab.items() if v >= len(SPECIALS)}
    out = []
    for i in ids:
        i = int(i)
        if i == EOS: break
        if i in (PAD, BOS): continue
        out.append(inv.get(i, "<unk>"))
    return out


def decode_ids(ids, vocab):
    return ("" if tokenizer_of(vocab) == "char" else " ").join(decoded_tokens(ids, vocab))


# ─────────────────────────── explanations (attention maps, attributions) ───────────────────────────
_RECORDER = None


class AttentionRecorder:
    """While active, every attention in `model` records its weights (batch row 0, per head).
    nn.MultiheadAttention (original Transformer layers, multi-head cross-attention) is made to return
    per-head weights; RopeAttention and additive attention record themselves."""
    def __init__(self, model):
        self.model, self.records, self._patched = model, [], []
        self.names = {id(m): n for n, m in model.named_modules()}

    def add(self, module, weights):
        self.records.append({"module": self.names.get(id(module), "?"), "weights": weights[0].detach().float().cpu()})

    def __enter__(self):
        global _RECORDER
        _RECORDER = self
        for m in self.model.modules():
            if isinstance(m, nn.MultiheadAttention):
                orig = m.forward
                def fwd(*a, _orig=orig, _m=m, **kw):
                    kw["need_weights"] = True; kw["average_attn_weights"] = False; kw.pop("is_causal", None)
                    out, w = _orig(*a, **kw)
                    if w is not None: self.add(_m, w)
                    return out, w
                m.forward = fwd; self._patched.append(m)
        return self

    def __exit__(self, *exc):
        global _RECORDER
        _RECORDER = None
        for m in self._patched: del m.forward  # back to the class method


def explain(model, x):
    """Generated tokens with top-5 alternatives per step and all attention maps for one input row x (1, D):
    greedy generation, then one teacher-forced pass over the generated tokens (identical logits) while recording.
    Input attributions come from mlpRes6.occlusion_attribution, which works for every model type."""
    model.eval()
    with torch.no_grad(): gen_out = model(x)
    tgt_dim = max(e["tgt_end"] for e in model.output_layout)
    targets = x.new_zeros(1, tgt_dim); gen = {}
    for e, off, w, dcol in model.routes:
        if dcol is None: continue
        ids = gen_out[0, e["start"]:e["end"]].view(e["max_len"], e["num_classes"]).argmax(-1)
        n = int((ids == EOS).nonzero()[0, 0]) + 1 if (ids == EOS).any() else len(ids)
        ids[n:] = PAD; gen[dcol] = (ids, n)
        targets[0, e["tgt_start"]:e["tgt_end"]] = ids.float()
    # (grad mode on: PyTorch's encoder layers skip the attention module, and so the recording, on their no-grad fast path)
    with torch.enable_grad(), AttentionRecorder(model) as rec:
        out = (model(x, targets) if gen else model(x)).detach()
    decoders = {}
    for e, off, w, dcol in model.routes:
        if dcol is None: continue
        ids, n = gen[dcol]
        probs = out[0, e["start"]:e["end"]].view(e["max_len"], e["num_classes"]).float().softmax(-1)
        top = probs[:n].topk(min(5, probs.size(-1)), -1)
        decoders[dcol] = {"ids": ids[:n].tolist(), "prob": probs[torch.arange(n), ids[:n]].tolist(),
                          "top_ids": top.indices.tolist(), "top_p": top.values.tolist()}
    enc_ids = {c: x[0, s0:s0 + wd].round().long().clamp(0, V - 1).tolist() for c, s0, wd, V in model.enc_slots}
    return {"decoders": decoders, "enc_ids": enc_ids, "attention": rec.records}


def sample_tokens(logits, temperature=0.0, top_k=0, top_p=1.0):
    """Next token ids from logits (B, V): argmax at temperature 0, else temperature / top-k / nucleus (top-p) sampling."""
    if temperature <= 0: return logits.argmax(-1)
    lg = logits / temperature
    if top_k and 0 < top_k < lg.size(-1):
        kth = lg.topk(int(top_k), -1).values[:, -1:]
        lg = lg.masked_fill(lg < kth, float("-inf"))
    if top_p < 1.0:
        srt, idx = lg.sort(-1, descending=True)
        cum = srt.softmax(-1).cumsum(-1)
        drop = cum - srt.softmax(-1) >= top_p  # keep the smallest set whose probability reaches top_p (always >= 1 token)
        lg = lg.masked_fill(drop.scatter(1, idx, drop), float("-inf"))
    return torch.multinomial(lg.softmax(-1), 1)[:, 0]


def check_generation(g):
    g = {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "mode": "auto", **(g or {})}
    g["temperature"], g["top_k"], g["top_p"] = float(g["temperature"]), int(g["top_k"]), float(g["top_p"])
    if g["temperature"] < 0 or g["top_k"] < 0 or not 0 < g["top_p"] <= 1: raise ValueError("Need temperature >= 0, top-k >= 0, 0 < top-p <= 1.")
    if g["mode"] not in ("auto", "prefix"): raise ValueError("Generation mode is 'auto' or 'prefix'.")
    return g


def check_params(p, col, is_output):
    """Validated copy of one column's settings (missing keys take DEFAULT_PARAMS)."""
    p = {**DEFAULT_PARAMS, **(p or {})}
    for k in ("dim", "layers", "heads"): p[k] = int(p[k])
    if p["tokenizer"] not in TOKENIZERS: raise ValueError(f"{col}: unknown tokenizer {p['tokenizer']!r}")
    if p["arch"] not in ARCHS: raise ValueError(f"{col}: unknown architecture {p['arch']!r}; use one of {list(ARCHS)}")
    if p["attention"] not in DEC_ATTENTION: raise ValueError(f"{col}: unknown attention {p['attention']!r}")
    if p["dim"] < 2 or p["layers"] < 1 or p["heads"] < 1: raise ValueError(f"{col}: dim >= 2, layers >= 1 and heads >= 1 are required")
    uses_heads = p["arch"] in ("transformer", "modern") or (is_output and p["arch"] in ATTENTION_ARCHS and p["attention"] == "multihead")
    if uses_heads and p["dim"] % p["heads"]: raise ValueError(f"{col}: dim {p['dim']} must be divisible by heads {p['heads']}")
    if p["arch"] == "modern" and (p["dim"] // p["heads"]) % 2: raise ValueError(f"{col}: RoPE needs an even head size (dim / heads)")
    return p


# ─────────────────────────── shared pieces ───────────────────────────
def _valid_mask(ids):
    mask = ids != PAD
    mask[:, 0] = True  # an empty text still has one (padding) position to attend to / pool
    return mask


def _masked_mean(h, mask):
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


def _sinusoid(length, dim, device):
    pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)[:, :dim // 2]
    return pe


def _reverse_valid(x, lengths):
    """Reverse each row's first `lengths` steps in place of the whole padded sequence (padding stays at the end)."""
    T = x.size(1); t = torch.arange(T, device=x.device)[None, :]
    idx = torch.where(t < lengths[:, None], lengths[:, None] - 1 - t, t)
    return x.gather(1, idx[..., None].expand(-1, -1, x.size(-1)))


class RopeAttention(nn.Module):
    """Multi-head attention; RoPE on self-attention, none on cross-attention (positions of two different sequences)."""
    def __init__(self, dim, heads, rope):
        super().__init__()
        self.h, self.hd = heads, dim // heads
        self.q, self.kv, self.o = nn.Linear(dim, dim, bias=False), nn.Linear(dim, 2 * dim, bias=False), nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.hd) if rope else None

    def forward(self, x, mem=None, key_mask=None, causal=False):
        B, T, D = x.shape; mem = x if mem is None else mem; S = mem.size(1)
        q = self.q(x).view(B, T, self.h, self.hd)
        k, v = self.kv(mem).view(B, S, 2, self.h, self.hd).unbind(2)
        if self.rope is not None:
            cos, sin = self.rope(q, T); q, k = apply_rotary_pos_emb(q, k, cos, sin)
        allowed = None
        if key_mask is not None: allowed = key_mask[:, None, None, :].expand(B, 1, T, S)
        if causal:
            tri = torch.ones(T, S, dtype=torch.bool, device=x.device).tril()[None, None]
            allowed = tri if allowed is None else allowed & tri
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=allowed)
        if _RECORDER is not None:  # explanation pass: the same weights, computed explicitly
            sc = (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2)) / math.sqrt(self.hd)
            if allowed is not None: sc = sc.masked_fill(~allowed, float("-inf"))
            _RECORDER.add(self, sc.softmax(-1))
        return self.o(y.transpose(1, 2).reshape(B, T, D))

    def step(self, x_t, cache, mem=None, key_mask=None):
        """One new position x_t (B, D). Self-attention: cache = {"k", "v"} of all earlier positions (None at the start),
        the new key/value is appended (RoPE at its position). Cross-attention: the memory's keys/values are cached once."""
        B, D = x_t.shape
        q = self.q(x_t).view(B, 1, self.h, self.hd)
        if mem is None:  # causal self-attention: attend to every cached position plus itself
            k, v = self.kv(x_t).view(B, 1, 2, self.h, self.hd).unbind(2)
            pos = 0 if cache is None else cache["k"].size(1)
            if self.rope is not None:
                cos, sin = self.rope(q, 1, position_offset=pos); q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if cache is not None: k, v = torch.cat([cache["k"], k], 1), torch.cat([cache["v"], v], 1)
            cache, allowed = {"k": k, "v": v}, None
        else:
            if cache is None:
                k, v = self.kv(mem).view(B, mem.size(1), 2, self.h, self.hd).unbind(2); cache = {"k": k, "v": v}
            k, v = cache["k"], cache["v"]
            allowed = None if key_mask is None else key_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=allowed)
        return self.o(y.transpose(1, 2).reshape(B, D)), cache


class CrossAttention(nn.Module):
    """Attention from decoder states to the encoder memory, combined Luong-style: tanh(W [h; context])."""
    def __init__(self, dim, kind, heads):
        super().__init__(); self.kind = kind
        if kind == "additive":
            self.wq, self.wk, self.v = nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim), nn.Linear(dim, 1, bias=False)
        else:
            self.mha = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.combine = nn.Linear(2 * dim, dim)

    def forward(self, h, mem, mem_mask):
        if self.kind == "additive":
            score = self.v(torch.tanh(self.wq(h)[:, :, None] + self.wk(mem)[:, None])).squeeze(-1)  # (B, T, S)
            score = score.masked_fill(~mem_mask[:, None, :], float("-inf"))
            w = torch.softmax(score, -1); ctx = w @ mem
            if _RECORDER is not None: _RECORDER.add(self, w[:, None])
        else:
            ctx, _ = self.mha(h, mem, mem, key_padding_mask=~mem_mask, need_weights=False)
        return torch.tanh(self.combine(torch.cat([h, ctx], -1)))


# ─────────────────────────── encoders: ids (B, T) -> memory (B, T, dim), mask (B, T), pooled (B, pool_dim) ───────────────────────────
class MLPEncoder(nn.Module):
    def __init__(self, V, T, p):
        super().__init__(); d = p["dim"]
        self.emb = nn.Embedding(V, d, padding_idx=PAD); self.pos = nn.Parameter(torch.randn(T, d) * 0.02)
        self.dim, self.pool_dim = d, T * d

    def forward(self, ids):
        mask = _valid_mask(ids); e = self.emb(ids)
        return e + self.pos, mask, e.flatten(1)


class RNNEncoder(nn.Module):
    """Bidirectional cuDNN RNN / GRU / LSTM over the unpadded lengths (packed sequences)."""
    def __init__(self, V, T, p):
        super().__init__(); d = p["dim"]; kind = p["arch"]
        self.emb = nn.Embedding(V, d, padding_idx=PAD)
        cls = {"gru": nn.GRU, "lstm": nn.LSTM}.get(kind, nn.RNN)
        extra = {"nonlinearity": "relu" if kind == "rnn_relu" else "tanh"} if cls is nn.RNN else {}
        self.rnn = cls(d, d, p["layers"], batch_first=True, bidirectional=True, **extra)
        self.out, self.pool = nn.Linear(2 * d, d), nn.Linear(2 * d, d)
        self.dim = self.pool_dim = d

    def forward(self, ids):
        mask = _valid_mask(ids); lengths = mask.sum(1)
        packed = pack_padded_sequence(self.emb(ids), lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        if isinstance(h, tuple): h = h[0]
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
        return self.out(out), mask, self.pool(torch.cat([h[-2], h[-1]], -1))


class M2RNNEncoder(nn.Module):
    """Pre-norm residual blocks; each mixes a forward M2RNN pass and a pass over the reversed text, then SwiGLU."""
    def __init__(self, V, T, p):
        super().__init__(); d = p["dim"]; L = p["layers"]
        self.emb = nn.Embedding(V, d, padding_idx=PAD)
        self.fwd = nn.ModuleList(M2RNNLayer(d) for _ in range(L)); self.bwd = nn.ModuleList(M2RNNLayer(d) for _ in range(L))
        self.n1 = nn.ModuleList(RMSNorm(d) for _ in range(L)); self.n2 = nn.ModuleList(RMSNorm(d) for _ in range(L))
        self.ffn = nn.ModuleList(SwiGLU(d, int(8 * d / 3)) for _ in range(L)); self.norm = RMSNorm(d)
        self.dim = self.pool_dim = d

    def forward(self, ids):
        mask = _valid_mask(ids); lengths = mask.sum(1); x = self.emb(ids)
        for f, b, n1, n2, ffn in zip(self.fwd, self.bwd, self.n1, self.n2, self.ffn):
            h = n1(x)
            x = x + f(h)[0] + _reverse_valid(b(_reverse_valid(h, lengths))[0], lengths)
            x = x + ffn(n2(x))
        x = self.norm(x)
        return x, mask, _masked_mean(x, mask)


class TransformerEncoder(nn.Module):
    """The original encoder: scaled embeddings + sinusoidal positions, post-norm layers."""
    def __init__(self, V, T, p):
        super().__init__(); d = p["dim"]
        self.emb = nn.Embedding(V, d, padding_idx=PAD)
        layer = nn.TransformerEncoderLayer(d, p["heads"], 4 * d, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, p["layers"], enable_nested_tensor=False)
        self.dim = self.pool_dim = d

    def forward(self, ids):
        mask = _valid_mask(ids)
        x = self.emb(ids) * math.sqrt(self.dim) + _sinusoid(ids.size(1), self.dim, ids.device)
        x = self.enc(x, src_key_padding_mask=~mask)
        return x, mask, _masked_mean(x, mask)


class _ConvBlock(nn.Module):
    """Non-causal dilated residual conv block (the encoder side of a TCN)."""
    def __init__(self, d, dilation, k=3):
        super().__init__()
        self.c1 = nn.Conv1d(d, d, k, padding=dilation * (k - 1) // 2, dilation=dilation)
        self.c2 = nn.Conv1d(d, d, k, padding=dilation * (k - 1) // 2, dilation=dilation)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x):  # (B, T, d)
        h = self.n1(F.gelu(self.c1(x.transpose(1, 2)).transpose(1, 2)))
        h = self.n2(self.c2(h.transpose(1, 2)).transpose(1, 2))
        return F.gelu(x + h)


class TCNEncoder(nn.Module):
    def __init__(self, V, T, p):
        super().__init__(); d = p["dim"]
        self.emb = nn.Embedding(V, d, padding_idx=PAD)
        self.blocks = nn.ModuleList(_ConvBlock(d, 2 ** i) for i in range(p["layers"]))
        self.dim = self.pool_dim = d

    def forward(self, ids):
        mask = _valid_mask(ids); m = mask.unsqueeze(-1).to(torch.float32); x = self.emb(ids)
        for blk in self.blocks: x = blk(x) * m  # keep padding silent so it cannot leak into valid positions
        return x, mask, _masked_mean(x, mask)


class _ModernBlock(nn.Module):
    def __init__(self, d, heads, cross):
        super().__init__()
        self.n1, self.attn = RMSNorm(d), RopeAttention(d, heads, rope=True)
        self.nc, self.cross = (RMSNorm(d), RopeAttention(d, heads, rope=False)) if cross else (None, None)
        self.n2, self.ffn = RMSNorm(d), SwiGLU(d, int(8 * d / 3))

    def forward(self, x, key_mask=None, causal=False, mem=None, mem_mask=None):
        x = x + self.attn(self.n1(x), key_mask=key_mask, causal=causal)
        if self.cross is not None: x = x + self.cross(self.nc(x), mem, mem_mask)
        return x + self.ffn(self.n2(x))

    def step(self, x_t, cache, mem, mem_mask):
        cache = cache or {}
        a, cache_self = self.attn.step(self.n1(x_t), cache.get("self"))
        x_t = x_t + a
        c, cache_cross = self.cross.step(self.nc(x_t), cache.get("cross"), mem, mem_mask)
        x_t = x_t + c
        return x_t + self.ffn(self.n2(x_t)), {"self": cache_self, "cross": cache_cross}


class ModernEncoder(nn.Module):
    """LLaMA-style blocks without the causal mask (RoPE only encodes relative position, so it works both ways)."""
    def __init__(self, V, T, p):
        super().__init__(); d = p["dim"]
        self.emb = nn.Embedding(V, d, padding_idx=PAD)
        self.blocks = nn.ModuleList(_ModernBlock(d, p["heads"], cross=False) for _ in range(p["layers"]))
        self.norm = RMSNorm(d); self.dim = self.pool_dim = d

    def forward(self, ids):
        mask = _valid_mask(ids); x = self.emb(ids)
        for blk in self.blocks: x = blk(x, key_mask=mask)
        x = self.norm(x)
        return x, mask, _masked_mean(x, mask)


ENCODERS = {"mlp": MLPEncoder, "rnn": RNNEncoder, "rnn_relu": RNNEncoder, "gru": RNNEncoder, "lstm": RNNEncoder,
            "m2rnn": M2RNNEncoder, "transformer": TransformerEncoder, "tcn": TCNEncoder, "modern": ModernEncoder}


# ─────────────────────────── decoders: (x = embedded shifted targets + z, z, memory, mask) -> hidden (B, T, dim) ───────────────────────────
class _OptionalCross(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.cross = CrossAttention(p["dim"], p["attention"], p["heads"]) if p["attention"] != "none" else None

    def attend(self, h, mem, mem_mask):
        return h if self.cross is None else self.cross(h, mem, mem_mask)


class RNNDecoder(_OptionalCross):
    def __init__(self, p):
        super().__init__(p); d = p["dim"]; kind = p["arch"]; self.L = p["layers"]
        cls = {"gru": nn.GRU, "lstm": nn.LSTM}.get(kind, nn.RNN)
        extra = {"nonlinearity": "relu" if kind == "rnn_relu" else "tanh"} if cls is nn.RNN else {}
        self.rnn = cls(d, d, self.L, batch_first=True, **extra)
        self.init = nn.Linear(d, self.L * d); self.is_lstm = cls is nn.LSTM

    def _h0(self, z):
        h0 = torch.tanh(self.init(z)).view(z.size(0), self.L, -1).transpose(0, 1).contiguous()
        return (h0, torch.zeros_like(h0)) if self.is_lstm else h0

    def forward(self, x, z, mem, mem_mask):
        out, _ = self.rnn(x, self._h0(z))
        return self.attend(out, mem, mem_mask)

    def step(self, x_t, z, mem, mem_mask, state):
        """Stateful: carry the RNN hidden state, feed one token."""
        out, state = self.rnn(x_t[:, None], self._h0(z) if state is None else state)
        return self.attend(out, mem, mem_mask)[:, 0], state


class M2RNNDecoder(_OptionalCross):
    def __init__(self, p):
        super().__init__(p); d = p["dim"]; L = p["layers"]
        self.mix = nn.ModuleList(M2RNNLayer(d) for _ in range(L))
        self.n1 = nn.ModuleList(RMSNorm(d) for _ in range(L)); self.n2 = nn.ModuleList(RMSNorm(d) for _ in range(L))
        self.ffn = nn.ModuleList(SwiGLU(d, int(8 * d / 3)) for _ in range(L)); self.norm = RMSNorm(d)

    def forward(self, x, z, mem, mem_mask):
        for mix, n1, n2, ffn in zip(self.mix, self.n1, self.n2, self.ffn):
            x = x + mix(n1(x))[0]
            x = x + ffn(n2(x))
        return self.attend(self.norm(x), mem, mem_mask)

    def step(self, x_t, z, mem, mem_mask, state):
        """Stateful: each M2RNN layer carries its matrix state and conv buffer."""
        x, new = x_t[:, None], []
        for i, (mix, n1, n2, ffn) in enumerate(zip(self.mix, self.n1, self.n2, self.ffn)):
            y, st = mix(n1(x), None if state is None else state[i]); new.append(st)
            x = x + y
            x = x + ffn(n2(x))
        return self.attend(self.norm(x), mem, mem_mask)[:, 0], new


class TCNDecoder(_OptionalCross):
    """linegen's causal TCN blocks (dilations 1, 2, 4, ...)."""
    def __init__(self, p):
        super().__init__(p)
        self.blocks = nn.Sequential(*[TCNBlock(p["dim"], act_name="gelu", dilation=2 ** i) for i in range(p["layers"])])

    def forward(self, x, z, mem, mem_mask):
        return self.attend(self.blocks(x.transpose(1, 2)).transpose(1, 2), mem, mem_mask)

    @property
    def receptive_field(self):  # two causal convs per block, each seeing (kernel - 1) * dilation earlier steps
        return 1 + sum((c.conv.kernel_size[0] - 1) * c.conv.dilation[0] for blk in self.blocks for c in (blk.c1, blk.c2))

    def step(self, x_t, z, mem, mem_mask, state):
        """Sliding window: the newest output depends only on the last receptive_field inputs, so only those are re-run."""
        hist = x_t[:, None] if state is None else torch.cat([state, x_t[:, None]], 1)
        hist = hist[:, -self.receptive_field:]
        h = self.blocks(hist.transpose(1, 2)).transpose(1, 2)[:, -1:]
        return self.attend(h, mem, mem_mask)[:, 0], hist


class TransformerDecoder(nn.Module):
    """The original decoder: masked self-attention + cross-attention, post-norm."""
    def __init__(self, p):
        super().__init__(); d = p["dim"]; self.dim = d
        layer = nn.TransformerDecoderLayer(d, p["heads"], 4 * d, dropout=0.0, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, p["layers"])

    def forward(self, x, z, mem, mem_mask):
        T = x.size(1)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        x = x * math.sqrt(self.dim) + _sinusoid(T, self.dim, x.device)
        return self.dec(x, mem, tgt_mask=causal, memory_key_padding_mask=~mem_mask)

    def step(self, x_t, z, mem, mem_mask, state):
        """PyTorch's TransformerDecoder has no cache: re-run the prefix and keep the last position."""
        hist = x_t[:, None] if state is None else torch.cat([state, x_t[:, None]], 1)
        return self.forward(hist, z, mem, mem_mask)[:, -1], hist


class ModernDecoder(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.blocks = nn.ModuleList(_ModernBlock(p["dim"], p["heads"], cross=True) for _ in range(p["layers"]))
        self.norm = RMSNorm(p["dim"])

    def forward(self, x, z, mem, mem_mask):
        for blk in self.blocks: x = blk(x, causal=True, mem=mem, mem_mask=mem_mask)
        return self.norm(x)

    def step(self, x_t, z, mem, mem_mask, state):
        """KV cache: keys/values of earlier positions (and of the memory) are kept, only the new token is computed."""
        state = state or [None] * len(self.blocks); new = []
        for blk, st in zip(self.blocks, state):
            x_t, st = blk.step(x_t, st, mem, mem_mask); new.append(st)
        return self.norm(x_t), new


DECODERS = {"rnn": RNNDecoder, "rnn_relu": RNNDecoder, "gru": RNNDecoder, "lstm": RNNDecoder, "m2rnn": M2RNNDecoder,
            "transformer": TransformerDecoder, "tcn": TCNDecoder, "modern": ModernDecoder}


class SeqDecoderHead(nn.Module):
    """One outdec column: embeds the shifted targets, conditions them on the latent z, decodes, predicts tokens.
    Memory = every encoder's per-token memory (projected to this decoder's width) plus z as one extra token."""
    def __init__(self, V, T, p, enc_dims):
        super().__init__(); d = p["dim"]; self.T, self.V = T, V
        self.emb = nn.Embedding(V, d, padding_idx=PAD)
        self.mem_proj = nn.ModuleList(nn.Linear(e, d) for e in enc_dims)
        self.core = DECODERS[p["arch"]](p)
        self.head = nn.Linear(d, V)
        self.register_buffer("never", torch.tensor([PAD, BOS]), persistent=False)  # never generated mid-text

    def memory(self, z, encoded):
        mems = [proj(m) for proj, (m, _) in zip(self.mem_proj, encoded)] + [z[:, None]]
        masks = [mk for _, mk in encoded] + [torch.ones(z.size(0), 1, dtype=torch.bool, device=z.device)]
        return torch.cat(mems, 1), torch.cat(masks, 1)

    def logits(self, prev, z, mem, mem_mask):
        return self.head(self.core(self.emb(prev) + z[:, None], z, mem, mem_mask))

    def teacher(self, z, encoded, target):
        mem, mm = self.memory(z, encoded)
        prev = torch.cat([torch.full_like(target[:, :1], BOS), target[:, :-1]], 1)
        return self.logits(prev, z, mem, mm)

    @torch.no_grad()
    def generate(self, z, encoded, temperature=0.0, top_k=0, top_p=1.0, mode="auto"):
        """Token by token. mode "auto": the decoder's own step (RNN / M2RNN state, KV cache, TCN window);
        "prefix": re-run the whole prefix each step (reference; gives the same tokens).
        Returns logits (B, T, V); a sampled token that is not the most likely one gets its logit raised to
        just above the maximum, so argmax-based decoding reads the sampled text."""
        mem, mm = self.memory(z, encoded); B = z.size(0)
        tok = torch.full((B,), BOS, dtype=torch.long, device=z.device); prev = tok[:, None]
        pad_row = torch.full((self.V,), -1e4, device=z.device); pad_row[PAD] = 1e4
        out = pad_row.expand(B, self.T, self.V).clone(); done = torch.zeros(B, dtype=torch.bool, device=z.device)
        stateful = mode == "auto"; state = None
        for t in range(self.T):
            if stateful:
                h, state = self.core.step(self.emb(tok) + z, z, mem, mm, state)
                lg = self.head(h).float()
            else:
                lg = self.logits(prev, z, mem, mm)[:, -1].float()
            nxt = sample_tokens(lg.index_fill(1, self.never, float("-inf")), temperature, top_k, top_p)
            chosen, top = lg.gather(1, nxt[:, None]), lg.max(-1, keepdim=True).values
            lg = lg.scatter(1, nxt[:, None], torch.where(chosen < top, top + 1e-3, chosen))
            out[:, t] = torch.where(done[:, None], pad_row, lg)
            nxt = nxt.masked_fill(done, PAD)
            done |= nxt == EOS
            if bool(done.all()): break
            tok = nxt; prev = torch.cat([prev, nxt[:, None]], 1)
        return out


# ─────────────────────────── the whole model ───────────────────────────
class SeqMLP(nn.Module):
    """Encoders per inenc column -> bridge (MLPO or linear) -> decoders per outdec column.

    input_slots: [(col, start, width)] of every input column in the flat input vector.
    output_layout: mlpRes6's build_output_layout entries (start/end/tgt_start/tgt_end per output column).
    make_mlp(in_dim, out_dim): builds the MLPO bridge."""
    def __init__(self, input_slots, col_types, vocabularies, output_layout, seq_params, bridge, make_mlp, output_activation=None):
        """output_activation: activation class applied to the numeric ('out') outputs."""
        super().__init__()
        if bridge not in BRIDGES: raise ValueError(f"Unknown bridge {bridge!r}; use one of {list(BRIDGES)}")
        self.output_layout = output_layout
        self.pred_dim = max(e["end"] for e in output_layout)
        self.enc_slots, plain = [], []
        self.encoders = nn.ModuleDict()
        for col, start, width in input_slots:
            if col_types[col] == "inenc":
                p = check_params(seq_params.get(col), col, False)
                self.encoders[col] = ENCODERS[p["arch"]](vocab_size(vocabularies[col]), width, p)
                self.enc_slots.append((col, start, width, vocab_size(vocabularies[col])))
            else: plain.extend(range(start, start + width))
        self.register_buffer("plain_idx", torch.tensor(plain, dtype=torch.long), persistent=False)
        enc_dims = [self.encoders[c].dim for c, *_ in self.enc_slots]
        self.decoders = nn.ModuleDict(); self.routes = []; off = 0  # (entry, bridge offset, width, decoder col or None)
        for e in output_layout:
            if e["type"] == "outdec":
                p = check_params(seq_params.get(e["col"]), e["col"], True)
                if p["arch"] != "mlp":
                    self.decoders[e["col"]] = SeqDecoderHead(e["num_classes"], e["max_len"], p, enc_dims)
                    self.routes.append((e, off, p["dim"], e["col"])); off += p["dim"]; continue
            w = e["end"] - e["start"]; self.routes.append((e, off, w, None)); off += w
        bridge_in = len(plain) + sum(self.encoders[c].pool_dim for c, *_ in self.enc_slots)
        if bridge_in == 0: raise ValueError("The model has no inputs.")
        all_mlp = not len(self.decoders) and all(isinstance(m, MLPEncoder) for m in self.encoders.values())
        self.bridge_kind = "mlp" if all_mlp else bridge
        self.bridge = make_mlp(bridge_in, off) if self.bridge_kind == "mlp" else nn.Linear(bridge_in, off)
        self.needs_targets = len(self.decoders) > 0
        numeric = [e["start"] for e in output_layout if e["type"] == "out"]
        self.output_act = output_activation() if output_activation is not None and numeric else None
        self.register_buffer("output_act_idx", torch.tensor(numeric, dtype=torch.long), persistent=False)
        self.generation = check_generation(None)  # used whenever decoders run without targets

    def set_generation(self, **g):
        """temperature (0 = argmax), top_k (0 = off), top_p (1 = off), mode ("auto" per-type stepping / "prefix")."""
        self.generation = check_generation({**self.generation, **g})

    # mlpRes6 tooling (explorer neuron views, LSUV, routing losses) looks at the MLP part
    @property
    def blocks(self): return getattr(self.bridge, "blocks", [])
    @property
    def input_attn(self): return None
    def routing_loss(self): return self.bridge.routing_loss() if hasattr(self.bridge, "routing_loss") else self.plain_idx.new_zeros((), dtype=torch.float32)
    def routing_usage(self): return self.bridge.routing_usage() if hasattr(self.bridge, "routing_usage") else None
    def get_all_learned_parameters(self): return self.bridge.get_all_learned_parameters() if hasattr(self.bridge, "get_all_learned_parameters") else []

    def forward(self, x, targets=None):
        """targets (the flat target tensor) switches decoders to teacher forcing; without it they generate."""
        if x.dim() == 1: return self.forward(x.unsqueeze(0), None if targets is None else targets.unsqueeze(0)).squeeze(0)
        feats, encoded = [x[:, self.plain_idx]], []
        for col, start, width, V in self.enc_slots:
            ids = x[:, start:start + width].round().long().clamp(0, V - 1)
            mem, mask, pooled = self.encoders[col](ids)
            encoded.append((mem, mask)); feats.append(pooled)
        b = self.bridge(torch.cat(feats, 1))
        out = b.new_zeros(x.size(0), self.pred_dim)
        for e, off, w, dcol in self.routes:
            z = b[:, off:off + w]
            if dcol is None: out[:, e["start"]:e["end"]] = z; continue
            dec = self.decoders[dcol]
            if targets is not None:
                lg = dec.teacher(z, encoded, targets[:, e["tgt_start"]:e["tgt_end"]].round().long().clamp(0, dec.V - 1))
            else:
                lg = dec.generate(z, encoded, **self.generation)
            out[:, e["start"]:e["end"]] = lg.reshape(x.size(0), -1).to(out.dtype)
        if self.output_act is not None:
            out = out.index_copy(-1, self.output_act_idx, self.output_act(out.index_select(-1, self.output_act_idx)))
        return out


def model_forward(model, x, targets=None):
    """mlpRes6's single call site for training/validation forwards: teacher forcing when the model decodes."""
    return model(x, targets) if getattr(model, "needs_targets", False) and targets is not None else model(x)
