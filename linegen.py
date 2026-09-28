#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import csv
import time
import math
import copy
import itertools
import random
import shutil
import pathlib
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Set, Callable, FrozenSet
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lamb import *
import logging
import torch._dynamo
import torch._inductor.config
import struct
import re
import binascii
import hashlib
from tqdm import tqdm

# Enable TF32 for Matrix Multiplications (Linear Layers)
torch.backends.cuda.matmul.allow_tf32 = True

# Enable TF32 for Convolutions (if you use TCNs/CNNs)
torch.backends.cudnn.allow_tf32 = True

print(f"🚀 TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
# Debug flags - set to True only when debugging torch.compile issues
torch._dynamo.config.verbose = False
torch._inductor.config.debug = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "model.pt"
CONFIG_PATH = "textgen.json"
VALIDATION_CSV_PATH = "validation_metrics.csv"
BOS_TOKEN = "<BOS>"
# at top near BOS_TOKEN
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"

SEED = 1337
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# === ADD: tiktoken (optional) ===
try:
    import tiktoken
except Exception:
    tiktoken = None

def _ensure_tiktoken():
    if tiktoken is None:
        raise RuntimeError("tiktoken not installed. Run: pip install tiktoken")

def save_activation_capture_bin(
    path: str,
    tokens_text: List[str],
    captured: List[Optional[torch.Tensor]]
):
    """
    Write a compact binary file with per-timestep activations + decoded token.

    File format (little-endian):
      magic:   4 bytes  = b'ACTV'
      version: uint16   = 1
      L:       uint16   = num_layers
      S:       uint32   = steps (timesteps captured)
      B:       uint32   = batch size (always 1 for current UI)
      H_l...:  For l in [0..L-1]: uint32 hidden_size_l

      For each step s in [0..S-1]:
        Tlen:  uint16 = byte-length of decoded token text at step s (UTF-8)
        T:     bytes  = token text
        For each layer l:
          activations: H_l * float32  (row = layer l, step s, batch 0)

    Notes:
      - If a layer wasn't captured (None), we write H_l=0 and skip its data.
      - Batch is fixed to 1 in the current sampling UI.
    """
    # Infer steps and hidden sizes from captured tensors
    # captured[l]: Tensor [B, S, H_l] on CPU (as returned by .get_captured())
    L = len(captured)
    B = 1
    # Determine S as the maximum S found (missing layers -> treated as H=0)
    S = 0
    Hs = []
    for t in captured:
        if t is None:
            Hs.append(0)
            continue
        assert t.dim() == 3 and t.size(0) == 1, "Expect captured as [1, S, H]"
        Hs.append(int(t.size(2)))
        S = max(S, int(t.size(1)))

    # sanity: tokens_text should match S (if not, we clamp)
    S = min(S, len(tokens_text))

    with open(path, "wb") as f:
        # header
        f.write(b"ACTV")
        f.write(struct.pack("<H", 1))            # version
        f.write(struct.pack("<H", L))            # num layers
        f.write(struct.pack("<I", S))            # steps
        f.write(struct.pack("<I", B))            # batch size
        for h in Hs:
            f.write(struct.pack("<I", h))        # hidden size per layer

        # body
        for s in range(S):
            token_bytes = tokens_text[s].encode("utf-8", "ignore")
            f.write(struct.pack("<H", len(token_bytes)))
            f.write(token_bytes)
            for l in range(L):
                h = Hs[l]
                if h == 0:
                    continue
                # slice [1, S, H] -> [H] at step s
                step_vec = captured[l][0, s, :].contiguous().view(-1)
                f.write(struct.pack("<%sf" % h, *step_vec.tolist()))


def split_indices(n, frac=0.9):
    idx = torch.randperm(n)
    k = int(frac * n)
    return idx[:k], idx[k:]


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def readable_num(n):
    if n >= 1_000_000: return f"{n/1_000_000:.2f}M"
    if n >= 1_000: return f"{n/1_000:.2f}k"
    return str(n)

# ========= Tokenization =========
# ==== NEW HELPERS (put near other small utils) ====

BOLD_ON  = "\033[1m"
BOLD_OFF = "\033[0m"
def bold(s: str) -> str:
    return f"{BOLD_ON}{s}{BOLD_OFF}"

# ─── ANSI colour / CLI formatting ─────────────────────────────────────────────
_R  = "\033[0m"       # reset
_B  = "\033[1m"       # bold
_DIM= "\033[2m"       # dim
_CY = "\033[36m"      # cyan
_YL = "\033[1m"#"\033[33m"      # yellow
_GR = "\033[32m"      # green
_MG = "\033[35m"      # magenta
_RD = "\033[31m"      # red
_WH = "\033[39m"      # default foreground (adapts to light/dark terminals)
_BL = "\033[34m"      # blue

def _c(*parts) -> str:
    """Concatenate ANSI codes + text + reset."""
    return "".join(str(p) for p in parts) + _R

def cli_banner(title: str, subtitle: str = "", width: int = 64) -> None:
    bar = "═" * (width - 2)
    inner = width - 2
    t_pad = (inner - len(title)) // 2
    print(f"\n{_c(_CY, '╔' + bar + '╗')}")
    print(f"{_c(_CY, '║')}{' ' * t_pad}{_c(_B, _WH, title)}{' ' * (inner - t_pad - len(title))}{_c(_CY, '║')}")
    if subtitle:
        s_pad = (inner - len(subtitle)) // 2
        print(f"{_c(_CY, '║')}{' ' * s_pad}{_c(_DIM, subtitle)}{' ' * (inner - s_pad - len(subtitle))}{_c(_CY, '║')}")
    print(f"{_c(_CY, '╚' + bar + '╝')}\n")

def cli_section(title: str, width: int = 64) -> None:
    dash = "─" * (width - len(title) - 5)
    print(f"\n  {_c(_CY, _B, '┌─')} {_c(_WH, _B, title)} {_c(_CY, '─' * max(2, width - len(title) - 7) + '┐')}")

def cli_section_end(width: int = 64) -> None:
    print(f"  {_c(_CY, '└' + '─' * (width - 4) + '┘')}")

def cli_rule(width: int = 64) -> None:
    print(f"  {_c(_DIM, '─' * width)}")

def cli_opt(key, label: str, desc: str = "", kw: int = 4, lw: int = 24) -> None:
    """Print a numbered/keyed option row with an optional description."""
    k_str = _c(_B, _B, f"{str(key):>{kw}}")
    l_str = _c(_WH, f"  {label:<{lw}}")
    d_str = f"  {_c(_DIM, desc)}" if desc else ""
    print(f"  │ {k_str}{l_str}{d_str}")

def cli_blank_row() -> None:
    print(f"  │")

def cli_group(label: str, width: int = 60) -> None:
    """Print a group header row inside a box."""
    dash = "─" * max(2, width - len(label) - 4)
    print(f"  │  {_c(_DIM, '── ' + label + ' ' + dash)}")

def pinfo(msg: str) -> None:
    """Informational line (cyan bullet)."""
    print(f"  {_c(_CY, '·')} {msg}")

def pwarn(msg: str) -> None:
    """Warning line."""
    print(f"  {_c(_YL, '!')} {msg}")

def pok(msg: str) -> None:
    """Success/result line."""
    print(f"  {_c(_GR, '✓')} {msg}")

def prompt_label(msg: str, default=None) -> str:
    """Format a prompt label with optional default hint."""
    hint = f" {_c(_DIM, f'[{default}]')}" if default is not None else ""
    return f"  {_c(_YL, _B, '▸')} {_c(_WH, msg)}{hint} "

def print_model_menu() -> None:
    """Pretty-print the model selection menu."""
    W = 66
    bar = "─" * (W - 4)
    print(f"\n  {_c(_CY, _B, '┌─')} {_c(_WH, _B, 'Model Selection')} {_c(_CY, '─' * (W - 22) + '┐')}")

    # ``MODEL_SPECS`` is built after all model definitions, but is available by
    # the time this interactive function is called.  Keeping presentation data
    # on the specs prevents the menu from becoming a second registry.
    groups = []
    for group_name in MODEL_MENU_GROUP_ORDER:
        specs = [spec for spec in MODEL_SPECS.values() if spec.menu_group == group_name]
        if specs:
            groups.append((group_name, [
                (spec.id, spec.menu_label, spec.menu_description) for spec in specs
            ]))

    for gname, models in groups:
        cli_blank_row()
        cli_group(gname, W - 4)
        for mid, mname, mdesc in models:
            cli_opt(mid, mname, mdesc, kw=4, lw=26)

    cli_blank_row()
    print(f"  {_c(_CY, '└' + '─' * (W - 4) + '┘')}")


def print_megabyte_compatible_models() -> None:
    """Show the exact selections that have a validated hierarchy-stage core."""
    compatible = [
        f"{model_id} ({MODEL_SPECS[model_id].name})"
        for model_id in sorted(HIERARCHICAL_MODEL_MIXERS)
    ]
    print(f"  │  {_c(_DIM, 'MEGABYTE-compatible models:')}")
    print(f"  │  {_c(_GR, ', '.join(compatible))}")
    print(f"  │  {_c(_DIM, f'{MLP_MODEL_ID} (MLP) is a bottom-up fine-patch decoder core; an MLP encoder must use the matching fine stage.')}")

def ensure_filegen_clean():
    """Clear and recreate FileGen/."""
    out_dir = pathlib.Path("FileGen")
    if out_dir.exists():
        for p in out_dir.iterdir():
            try:
                if p.is_file() or p.is_symlink():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
            except Exception:
                pass
        try: out_dir.rmdir()
        except Exception: pass
    out_dir.mkdir(parents=True, exist_ok=True)

# ========= Tokenization =========
class BaseVocab:
    line_mode: bool = False
    bos_id: Optional[int] = None
    def encode(self, s): raise NotImplementedError
    def decode(self, ids): raise NotImplementedError
    @property
    def size(self): raise NotImplementedError

class CharVocab(BaseVocab):
    def __init__(self, texts: List[str], line_mode: bool):
        charset = set()
        for t in texts:
            charset.update(t)
        self.line_mode = line_mode

        self.tokens = sorted(list(charset))
        if line_mode:
            for sp in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN):
                if sp in self.tokens:
                    self.tokens.remove(sp)
            self.tokens = [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN] + self.tokens

        self.stoi = {ch: i for i, ch in enumerate(self.tokens)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        self.bos_id = self.stoi[BOS_TOKEN] if line_mode else None
        self.eos_id = self.stoi[EOS_TOKEN] if line_mode else None
        self.pad_id = self.stoi[PAD_TOKEN] if line_mode else None

    def encode(self, s: str) -> List[int]:
        if not self.line_mode:
            return [self.stoi[c] for c in s]
        # In line mode we allow literal `<BOS>` in input (optional), but we always append `<EOS>`
        out = []
        i = 0
        L = len(s)
        if s.startswith(BOS_TOKEN):
            out.append(self.bos_id)
            i += len(BOS_TOKEN)
        else:
            out.append(self.bos_id)
        while i < L:
            if s.startswith(BOS_TOKEN, i):
                out.append(self.bos_id); i += len(BOS_TOKEN)
            elif s.startswith(EOS_TOKEN, i):
                out.append(self.eos_id); i += len(EOS_TOKEN)
            else:
                out.append(self.stoi[s[i]]); i += 1
        out.append(self.eos_id)
        return out

    def decode(self, ids: List[int]) -> str:
        if not self.line_mode:
            return "".join(self.itos[i] for i in ids)
        parts = []
        for i in ids:
            if i == self.bos_id: parts.append(BOS_TOKEN)
            elif i == self.eos_id: parts.append(EOS_TOKEN)
            elif i == self.pad_id: parts.append("")  # hide PAD in decode
            else: parts.append(self.itos[i])
        return "".join(parts)

    @property
    def size(self): return len(self.tokens)
# === ADD: TiktokenVocab ===
class TiktokenVocab(BaseVocab):
    """
    Wrapper for tiktoken encodings. In line_mode we reserve three IDs above base vocab
    for BOS/EOS/PAD so sampling/stop logic works like other vocabs.
    
    NEW: scan_file() builds a filtered set of actually-used tokens from the dataset,
    so random prompt generation only picks tokens that actually exist in training data.
    """
    def __init__(self, encoding_name: str, line_mode: bool):
        _ensure_tiktoken()
        self.enc = tiktoken.get_encoding(encoding_name)
        self.line_mode = line_mode
        self.n_base = int(self.enc.n_vocab)
        self._active_tokens: Optional[List[int]] = None

        if line_mode:
            self.bos_id = self.n_base
            self.eos_id = self.n_base + 1
            self.pad_id = self.n_base + 2
            self._size = self.n_base + 3
        else:
            self.bos_id = None
            self.eos_id = None
            self.pad_id = None
            self._size = self.n_base

    def scan_file(self, txt_path: str, max_bytes: int = 8 * 1024 * 1024):
        """Scan the dataset and build a set of actually-used tiktoken IDs."""
        print(f"[TikToken] Scanning {txt_path} for active tokens (up to {max_bytes//1024//1024}MB)...")
        active = set()
        bytes_read = 0
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            while bytes_read < max_bytes:
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                bytes_read += len(chunk.encode("utf-8", "ignore"))
                ids = self.enc.encode(chunk)
                active.update(ids)
        self._active_tokens = sorted(active)
        print(f"[TikToken] Found {len(self._active_tokens)} unique tokens (of {self.n_base} total)")

    def get_random_token(self) -> int:
        """Return a random token from the active set if available."""
        if self._active_tokens and len(self._active_tokens) > 0:
            return random.choice(self._active_tokens)
        return random.randrange(self.n_base)

    def encode(self, s: str) -> List[int]:
        ids = self.enc.encode(s if isinstance(s, str) else str(s))
        if self.line_mode:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        if self.line_mode:
            ids = [i for i in ids if i not in (self.bos_id, self.eos_id, self.pad_id)]
        base_ids = [i for i in ids if 0 <= int(i) < self.n_base]
        return self.enc.decode(base_ids)

    @property
    def size(self): return self._size

    @property
    def tokens(self):
        if self._active_tokens is not None:
            return self._active_tokens
        return list(range(self._size))
    
    @property
    def active_token_count(self) -> int:
        if self._active_tokens is not None:
            return len(self._active_tokens)
        return self._size
# === Custom Efficient BPE ===
class CustomBPEVocab(BaseVocab):
    """
    Byte-level BPE tokenizer (like GPT-2) implemented in pure Python.
    Designed to train on a subset of data to avoid RAM issues on large files.
    """
    def __init__(self, vocab_path: str, line_mode: bool, expected_size: int = 4096):
        self.line_mode = line_mode
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.base_size = 256
        
        # Load if exists
        self.vocab_path = vocab_path
        if os.path.exists(vocab_path):
            self._load(vocab_path)
        else:
            print(f"[BPE] Vocab file {vocab_path} not found. Will train on data.")
            # We initialize empty, train() must be called externally
            
        # Determine IDs for specials
        if line_mode:
            self.bos_id = expected_size
            self.eos_id = expected_size + 1
            self.pad_id = expected_size + 2
            self._size = expected_size + 3
        else:
            self.bos_id = None
            self.eos_id = None
            self.pad_id = None
            self._size = expected_size

    def _get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_ids(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, txt_path: str, vocab_size: int):
        """
        Efficient BPE training with incremental pair count updates.
        Instead of rescanning the entire sequence each merge (O(n) per step = O(n*m) total),
        we update pair counts incrementally around each merge site (amortized O(1) per site).
        """
        print(f"[BPE] Training Custom BPE (target size: {vocab_size})...")
        
        MAX_BYTES = 8 * 1024 * 1024 
        with open(txt_path, "rb") as f:
            raw_bytes = f.read(MAX_BYTES)
        
        ids = list(raw_bytes)
        print(f"[BPE] Loaded {len(ids)/1024/1024:.2f} MB of sample data ({len(ids)} tokens).")
        
        num_merges = vocab_size - 256
        if num_merges <= 0:
            print("[BPE] Vocab size <= 256, no BPE training needed.")
            return

        # Build initial pair counts
        import collections as _collections
        pair_counts = _collections.Counter()
        for i in range(len(ids) - 1):
            pair_counts[(ids[i], ids[i+1])] += 1

        pbar = tqdm(total=num_merges, desc="BPE Merge")
        for merge_idx in range(num_merges):
            if not pair_counts:
                print(f"[BPE] No more pairs at vocab size {256+merge_idx}. Stopping.")
                break
            
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]
            
            if best_count < 2:
                print(f"[BPE] Best pair count={best_count} at step {merge_idx}. Stopping early.")
                break
            
            new_idx = 256 + merge_idx
            p0, p1 = best_pair
            
            self.merges[best_pair] = new_idx
            self.vocab[new_idx] = self.vocab[p0] + self.vocab[p1]
            
            # Apply merge with incremental pair count updates
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == p0 and ids[i+1] == p1:
                    # Subtract old neighbouring pairs
                    if new_ids:
                        prev = new_ids[-1]
                        old_left = (prev, p0)
                        if old_left in pair_counts:
                            pair_counts[old_left] -= 1
                            if pair_counts[old_left] <= 0:
                                del pair_counts[old_left]
                    
                    if best_pair in pair_counts:
                        pair_counts[best_pair] -= 1
                        if pair_counts[best_pair] <= 0:
                            del pair_counts[best_pair]
                    
                    if i + 2 < len(ids):
                        nxt = ids[i+2]
                        old_right = (p1, nxt)
                        if old_right in pair_counts:
                            pair_counts[old_right] -= 1
                            if pair_counts[old_right] <= 0:
                                del pair_counts[old_right]
                    
                    # Add new neighbouring pairs
                    if new_ids:
                        new_left = (new_ids[-1], new_idx)
                        pair_counts[new_left] = pair_counts.get(new_left, 0) + 1
                    
                    new_ids.append(new_idx)
                    
                    if i + 2 < len(ids):
                        nxt = ids[i+2]
                        new_right = (new_idx, nxt)
                        pair_counts[new_right] = pair_counts.get(new_right, 0) + 1
                    
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            
            ids = new_ids
            pbar.update(1)
            if merge_idx % 500 == 0:
                pbar.set_postfix(vocab=256+merge_idx, seq_len=len(ids), top_freq=best_count)
        
        pbar.close()
        print(f"[BPE] Training complete. Final vocab size: {256 + len(self.merges)}, seq compressed to {len(ids)}")
        self._save(self.vocab_path)

    def _save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Custom BPE Merges v1\n")
            sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
            for (p0, p1), idx in sorted_merges:
                f.write(f"{p0} {p1} {idx}\n")

    def _load(self, path):
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.strip().split()
                if len(parts) != 3: continue
                p0, p1, idx = int(parts[0]), int(parts[1]), int(parts[2])
                self.merges[(p0, p1)] = idx
                self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
    
    def encode(self, s) -> List[int]:
        if isinstance(s, str):
            s = s.encode("utf-8")
        
        if isinstance(s, (bytes, bytearray)):
            ids = list(s)
        elif isinstance(s, list):
            ids = s
        else:
            raise ValueError("BPE encode expects str, bytes, or list[int]")

        while len(ids) >= 2:
            candidates = {}
            for pair in zip(ids, ids[1:]):
                if pair in self.merges:
                    candidates[pair] = self.merges[pair]
            
            if not candidates:
                break
            
            best_pair = min(candidates, key=candidates.get)
            ids = self._merge_ids(ids, best_pair, self.merges[best_pair])
            
        if self.line_mode:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        if self.line_mode:
            ids = [i for i in ids if i not in (self.bos_id, self.eos_id, self.pad_id)]
        
        res = b""
        for i in ids:
            res += self.vocab.get(i, b"")
        return res.decode("utf-8", "ignore")

    @property
    def size(self):
        return self._size

    @property
    def tokens(self):
        return list(range(self._size))
class WordVocab(BaseVocab):
    """Whitespace-split word-level tokenizer (+ BOS/EOS/PAD in line mode)."""
    def __init__(self, lines: List[str], line_mode: bool):
        self.line_mode = line_mode
        words = set()
        for s in lines:
            words.update(s.split())
        self.tokens = sorted(list(words))
        if line_mode:
            for sp in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN):
                if sp in self.tokens:
                    self.tokens.remove(sp)
            self.tokens = [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN] + self.tokens
        self.stoi = {w:i for i,w in enumerate(self.tokens)}
        self.itos = {i:w for w,i in self.stoi.items()}
        self.bos_id = self.stoi[BOS_TOKEN] if line_mode else None
        self.eos_id = self.stoi[EOS_TOKEN] if line_mode else None
        self.pad_id = self.stoi[PAD_TOKEN] if line_mode else None

    def encode(self, s: str) -> List[int]:
        if not self.line_mode:
            return [self.stoi[w] for w in s.split()]

        if s.startswith(BOS_TOKEN):
            # keep BOS as the very first token, then split the rest
            rest = s[len(BOS_TOKEN):].lstrip()
            words = rest.split()
        else:
            words = s.split()
        return [self.bos_id] + [self.stoi[w] for w in words if w != EOS_TOKEN] + [self.eos_id]

    def decode(self, ids: List[int]) -> str:
        if not self.line_mode:
            return " ".join(self.itos[i] for i in ids)
        out = []
        for i in ids:
            if i == self.bos_id:
                # keep BOS literal; caller can hide it
                out.append(BOS_TOKEN)
            elif i == self.eos_id:
                out.append(EOS_TOKEN)
            elif i == self.pad_id:
                continue
            else:
                out.append(self.itos[i])
        return " ".join(out)

    @property
    def size(self): return len(self.tokens)

class BinaryVocab(BaseVocab):
    """Binary tokens {0,1} (+ BOS/EOS/PAD in line mode)."""
    def __init__(self, line_mode: bool):
        self.line_mode = line_mode
        self.tokens = [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, "0", "1"] if line_mode else ["0", "1"]
        self.stoi = {t:i for i,t in enumerate(self.tokens)}
        self.itos = {i:t for t,i in self.stoi.items()}
        self.bos_id = self.stoi[BOS_TOKEN] if line_mode else None
        self.eos_id = self.stoi[EOS_TOKEN] if line_mode else None
        self.pad_id = self.stoi[PAD_TOKEN] if line_mode else None

    def encode(self, s) -> List[int]:
        """Encode binary text, or expand each raw byte to its eight bits."""
        ids = []
        if isinstance(s, (bytes, bytearray, memoryview)):
            for byte in bytes(s):
                ids.extend(self.stoi[bit] for bit in f"{byte:08b}")
            return [self.bos_id] + ids + [self.eos_id] if self.line_mode else ids

        if not isinstance(s, str):
            raise TypeError("BinaryVocab.encode expects binary text or bytes")
        if self.line_mode and s.startswith(BOS_TOKEN):
            s = s[len(BOS_TOKEN):]
        for ch in s:
            if ch in ("0","1"):
                ids.append(self.stoi[ch])
        return [self.bos_id] + ids + [self.eos_id] if self.line_mode else ids

    def decode(self, ids: List[int]) -> str:
        if not self.line_mode:
            return "".join(self.itos[i] for i in ids)
        parts = []
        for i in ids:
            if i == self.bos_id:
                parts.append(BOS_TOKEN)
            elif i == self.eos_id:
                parts.append(EOS_TOKEN)
            elif i == self.pad_id:
                continue
            else:
                parts.append(self.itos[i])
        return "".join(parts)

    def to_bytes(self, ids: List[int]) -> bytes:
        """Pack generated bit tokens into bytes for FileGen output.

        A final incomplete byte is zero-padded on the right. Special line-mode
        tokens are omitted, so the result always contains only generated bits.
        """
        bits = [self.itos[token_id] for token_id in ids if self.itos.get(token_id) in ("0", "1")]
        if not bits:
            return bytes()
        padding = (-len(bits)) % 8
        bits.extend("0" for _ in range(padding))
        return bytes(int("".join(bits[index:index + 8]), 2) for index in range(0, len(bits), 8))

    @property
    def size(self): return len(self.tokens)

HEX_RE = re.compile(r'^(?:0x)?[0-9a-fA-F]+(?:\s+(?:0x)?[0-9a-fA-F]+)*$')

def _hex_to_bytes(s: str) -> bytes:
    # normalize: remove whitespace, allow optional 0x prefixes
    s = re.sub(r'\s+', '', s)
    s = re.sub(r'0x', '', s, flags=re.IGNORECASE)
    if len(s) % 2 == 1:
        raise ValueError(
            f"hex byte input must contain an even number of digits (received {len(s)})"
        )
    return bytes.fromhex(s)

class ByteVocab(BaseVocab):
    """0..255 bytes (+ BOS/EOS/PAD in line mode)."""
    def __init__(self, line_mode: bool):
        self.line_mode = line_mode
        self.bos_id = 256 if line_mode else None
        self.eos_id = 257 if line_mode else None
        self.pad_id = 258 if line_mode else None
        self._size = 259 if line_mode else 256

    def _with_line_specials(self, ids: List[int]) -> List[int]:
        return [self.bos_id] + ids + [self.eos_id] if self.line_mode else ids

    def encode(self, s) -> List[int]:
        """
        Accepts:
          - bytes/bytearray/memoryview → direct mapping
          - str → latin1 by default; BUT if it looks like hex (or starts with 'hex:'), parse as hex bytes
                   If line_mode and string starts with BOS_TOKEN, it's honored first.
          - int in [0,255] → single byte token
          - list/tuple of ints in [0,255] → sequence of byte tokens
        """
        # ints
        if isinstance(s, int):
            if not (0 <= s <= 255):
                raise ValueError("ByteVocab.encode int must be in [0,255]")
            ids = [s]
            return self._with_line_specials(ids)

        # list/tuple of ints
        if isinstance(s, (list, tuple)) and all(isinstance(x, int) for x in s):
            ids = [x for x in s if 0 <= x <= 255]
            return self._with_line_specials(ids)

        # bytes-like
        if isinstance(s, (bytes, bytearray, memoryview)):
            b = bytes(s)
            ids = list(b)
            return self._with_line_specials(ids)

        # strings (latin1 vs hex)
        if isinstance(s, str):
            # Handle BOS_TOKEN literally at the front in line_mode
            if self.line_mode and s.startswith(BOS_TOKEN):
                s = s[len(BOS_TOKEN):]

            # explicit "hex:" prefix OR looks like hex (with optional 0x and spaces)
            looks_hex = s.lower().startswith("hex:") or bool(HEX_RE.match(s))
            if looks_hex:
                if s.lower().startswith("hex:"):
                    s = s[4:].lstrip()
                try:
                    b = _hex_to_bytes(s)
                except ValueError:
                    raise ValueError("Invalid hex string for ByteVocab.encode")
                ids = list(b)
                return self._with_line_specials(ids)

            # fallback: latin1
            ids = list(s.encode("latin1", "ignore"))
            return self._with_line_specials(ids)

        raise TypeError("ByteVocab.encode expects bytes, str, int, or list[int]")

    def decode(self, ids: List[int]) -> str:
        if self.line_mode:
            ids = [i for i in ids if i not in (self.bos_id, self.eos_id, self.pad_id)]
        return bytes([i for i in ids if 0 <= i <= 255]).decode("latin1", "ignore")

    def to_bytes(self, ids: List[int]) -> bytes:
        if self.line_mode:
            ids = [i for i in ids if i not in (self.bos_id, self.eos_id, self.pad_id)]
        return bytes([i for i in ids if 0 <= i <= 255])

    @property
    def size(self): 
        return self._size

    @property
    def tokens(self):
        # for random sampling in callers that expect a tokens list
        return list(range(self._size))


# ========= Sequence-to-sequence line datasets =========
# Seq2seq mode rewrites a delimited file (CSV/TSV/...) into one example per
# line: the chosen input columns first, then the output columns, all joined by
# tabs.  Everything up to and including the tab after the last input column is
# the source.  It is fed to the model like a prompt but masked out of the loss,
# so training and validation score only the output columns and the final EOS.
SEQ2SEQ_JOINER = "\t"
SEQ2SEQ_DELIMITERS = {0: ",", 1: ":", 2: "\t"}


def _seq2seq_split_row(text: str, delimiter: str) -> List[str]:
    """Split one row; single-character delimiters honour CSV quoting."""
    if len(delimiter) == 1:
        return next(csv.reader([text], delimiter=delimiter), [])
    return text.split(delimiter)


def _seq2seq_rows(path: str, delimiter: str):
    """Yield the rows of a delimited file (surrogateescape keeps raw bytes intact)."""
    with open(path, "r", encoding="utf-8", errors="surrogateescape", newline="") as f:
        if len(delimiter) == 1:
            csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
            yield from csv.reader(f, delimiter=delimiter)
        else:
            for line in f:
                yield line.rstrip("\r\n").split(delimiter)


def seq2seq_prepared_path(spec: dict, source_path: str) -> str:
    """A prepared-file name unique to the source file's version and the column spec."""
    src = os.path.abspath(source_path)
    st = os.stat(src)
    key = json.dumps([src, st.st_size, st.st_mtime_ns, spec["delimiter"], spec["has_header"],
                      spec["input_cols"], spec["output_cols"]])
    p = pathlib.Path(src)
    return str(p.with_name(f"{p.stem}.seq2seq-{hashlib.sha1(key.encode()).hexdigest()[:10]}.txt"))


def prepare_seq2seq_dataset(spec: dict, source_path: Optional[str] = None,
                            out_path: Optional[str] = None) -> str:
    """Write the tab-joined, inputs-then-outputs line file for ``spec``; return its path.

    Rows are dropped (and counted) when their column count differs from the
    header, or when a kept field contains a tab or line break, because either
    would move the source/target boundary.
    """
    source_path = source_path or spec["source_path"]
    out_path = out_path or seq2seq_prepared_path(spec, source_path)
    if os.path.exists(out_path):
        pinfo(f"Seq2seq: using prepared dataset {out_path}")
        return out_path

    width = len(spec["columns"])
    keep = list(spec["input_cols"]) + list(spec["output_cols"])
    written = 0
    skipped = collections.Counter()
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", errors="surrogateescape", newline="\n") as out:
        for n, row in enumerate(tqdm(_seq2seq_rows(source_path, spec["delimiter"]), desc="Seq2seq rows")):
            if n == 0 and spec["has_header"]:
                continue
            if not any(field.strip() for field in row):
                skipped["blank row"] += 1
                continue
            if len(row) != width:
                skipped[f"not {width} columns"] += 1
                continue
            fields = [row[i] for i in keep]
            if any(ch in field for field in fields for ch in "\t\r\n"):
                skipped["tab or line break inside a field"] += 1
                continue
            out.write(SEQ2SEQ_JOINER.join(fields) + "\n")
            written += 1
    if written == 0:
        os.remove(tmp_path)
        raise ValueError(f"Seq2seq: no usable rows in {source_path} (skipped: {dict(skipped)})")
    os.replace(tmp_path, out_path)
    pok(f"Seq2seq: wrote {written:,} examples to {out_path}")
    for reason, count in skipped.items():
        pwarn(f"Seq2seq: skipped {count:,} rows ({reason})")
    return out_path


def ensure_seq2seq_dataset(cfg) -> None:
    """Recreate a seq2seq run's prepared file if it was deleted."""
    spec = cfg.get("seq2seq")
    if spec and not os.path.exists(cfg["dataset_path"]):
        pwarn(f"Seq2seq: {cfg['dataset_path']} is missing; rebuilding it from {spec['source_path']}")
        prepare_seq2seq_dataset(spec, out_path=cfg["dataset_path"])


def prompt_seq2seq_config(dataset_path: str) -> Optional[dict]:
    """Ask whether to train seq2seq and, if so, the delimiter and each column's role."""
    cli_section("Sequence-to-sequence", 64)
    print(f"  │  {_c(_DIM, 'Seq2seq reads a delimited file and trains the model to produce the')}")
    print(f"  │  {_c(_DIM, 'output columns from the input columns. Inputs are fed as a prompt')}")
    print(f"  │  {_c(_DIM, 'but excluded from the loss. Outputs are moved to the end of each line.')}")
    print(f"  │")
    if prompt_int("Seq2seq mode  (0=off 1=on)", valid={0, 1}, default=0) == 0:
        cli_section_end(64)
        return None

    print(f"  │")
    cli_opt(0, "Comma", ",")
    cli_opt(1, "Colon", ":")
    cli_opt(2, "Tab", "\\t")
    cli_opt(3, "Custom", "Any string; type \\t for a tab")
    print(f"  │")
    choice = prompt_int("Delimiter", valid={0, 1, 2, 3}, default=0)
    delimiter = SEQ2SEQ_DELIMITERS.get(choice, "")
    while not delimiter:
        # Read raw so a space (or other whitespace) delimiter survives.
        delimiter = input(prompt_label("Custom delimiter")).replace("\\t", "\t")
    has_header = prompt_int("First row is a header  (0=no 1=yes)", valid={0, 1}, default=1) == 1

    rows = _seq2seq_rows(dataset_path, delimiter)
    first, second = next(rows, None), next(rows, None)
    rows.close()
    if not first:
        raise ValueError(f"Seq2seq: {dataset_path} is empty")
    if len(first) < 2:
        raise ValueError(f"Seq2seq: delimiter {delimiter!r} finds only one column in the first row; "
                         "at least one input and one output column are needed")
    names = ([field.strip() or f"column {i}" for i, field in enumerate(first)] if has_header
             else [f"column {i}" for i in range(len(first))])
    example_row = second if has_header else first

    print(f"  │")
    print(f"  │  {_c(_DIM, f'{len(names)} columns. Type: 0 = ignore, 1 = input, 2 = output.')}")
    while True:
        roles = []
        for i, name in enumerate(names):
            example = example_row[i] if example_row and i < len(example_row) else ""
            if len(example) > 40:
                example = example[:40] + "…"
            print(f"  │  {_c(_CY, f'[{i}]')} {_c(_WH, name)}  {_c(_DIM, f'e.g. {example!r}')}")
            roles.append(prompt_int(f"Type of column {i}", valid={0, 1, 2},
                                    default=2 if i == len(names) - 1 else 1))
        input_cols = [i for i, r in enumerate(roles) if r == 1]
        output_cols = [i for i, r in enumerate(roles) if r == 2]
        if input_cols and output_cols:
            break
        pwarn("Choose at least one input column and one output column.")
    cli_section_end(64)
    return {
        "source_path": os.path.abspath(dataset_path),
        "delimiter": delimiter,
        "has_header": has_header,
        "columns": names,
        "input_cols": input_cols,
        "output_cols": output_cols,
    }


def _encode_line_body(vocab: BaseVocab, text) -> List[int]:
    """Encode text without the BOS/EOS that line-mode vocabularies add."""
    ids = vocab.encode(text)
    return ids[1:-1] if vocab.line_mode else ids


def encode_seq2seq_line(vocab: BaseVocab, line, input_count: int) -> Tuple[List[int], int]:
    """Encode a prepared seq2seq line as BOS + source + target + EOS.

    Source and target are tokenized separately so a subword tokenizer cannot
    merge across the boundary; prompting with only the source then reproduces
    the training token prefix exactly.  Returns ``(ids, source_len)``, where
    ``source_len`` counts the source tokens after BOS (their targets are masked).
    """
    joiner = SEQ2SEQ_JOINER.encode() if isinstance(line, (bytes, bytearray)) else SEQ2SEQ_JOINER
    cut = -1
    for _ in range(input_count):
        cut = line.find(joiner, cut + 1)
        if cut < 0:
            raise ValueError(f"Seq2seq line has fewer than {input_count} tab-separated "
                             f"input columns: {line[:80]!r}")
    source = _encode_line_body(vocab, line[:cut + 1])
    target = _encode_line_body(vocab, line[cut + 1:])
    return [vocab.bos_id] + source + target + [vocab.eos_id], len(source)


def encode_seq2seq_prompt(vocab: BaseVocab, spec: dict, text: str) -> List[int]:
    """BOS + source tokens for input columns typed with the dataset's delimiter."""
    fields = _seq2seq_split_row(text, spec["delimiter"])
    names = [spec["columns"][i] for i in spec["input_cols"]]
    if len(fields) != len(names):
        raise ValueError(f"Expected {len(names)} input column(s) ({', '.join(names)}) separated by "
                         f"{spec['delimiter']!r}; got {len(fields)}")
    source = SEQ2SEQ_JOINER.join(fields) + SEQ2SEQ_JOINER
    if isinstance(vocab, (ByteVocab, BinaryVocab)):
        source = source.encode("utf-8", "surrogateescape")
    return [vocab.bos_id] + _encode_line_body(vocab, source)


def seq2seq_visible(text: str) -> str:
    """Show the tab joiner between columns readably."""
    return text.replace(SEQ2SEQ_JOINER, " │ ")


# ========= Datasets =========
def pad_line_examples(examples: List[Tuple[List[int], int]], pad_id: int):
    """Next-token (x, y) tensors for encoded lines; each line's first
    ``masked`` targets (a seq2seq source) are set to PAD so the loss skips them."""
    width = max(1, max(len(ids) for ids, _ in examples) - 1)
    x = torch.full((len(examples), width), pad_id, dtype=torch.long)
    y = torch.full((len(examples), width), pad_id, dtype=torch.long)
    for row, (ids, masked) in enumerate(examples):
        if len(ids) < 2:
            continue
        seq = torch.tensor(ids, dtype=torch.long)
        n = len(ids) - 1
        x[row, :n] = seq[:-1]
        y[row, :n] = seq[1:]
        if masked:
            y[row, :min(masked, n)] = pad_id
    return x.to(DEVICE), y.to(DEVICE)


class ClassicCorpus:
    def __init__(self, text: str, vocab: CharVocab, seq_len: int):
        self.vocab = vocab
        self.ids = torch.tensor(vocab.encode(text), dtype=torch.long)
        self.seq_len = seq_len
    def get_batch(self, batch_size: int):
        L = len(self.ids) - (self.seq_len + 1)
        idx = torch.randint(0, max(1, L), (batch_size,))
        x = torch.stack([self.ids[i:i+self.seq_len] for i in idx])
        y = torch.stack([self.ids[i+1:i+self.seq_len+1] for i in idx])
        return x.to(DEVICE), y.to(DEVICE)

class LineDataset:
    def __init__(self, lines: List[str], vocab: BaseVocab):
        assert vocab.line_mode
        self.vocab = vocab
        # each line gets BOS ... EOS via vocab.encode
        enc = [vocab.encode(ln) for ln in lines]  # variable lengths, already BOS...EOS
        self.lines_enc = enc

        self.max_len = max(len(e) for e in enc)
        data = []
        for e in enc:
            pad_len = self.max_len - len(e)
            data.append(e + [vocab.pad_id]*pad_len)
        self.data = torch.tensor(data, dtype=torch.long)

    def get_batch(self, batch_size: int):
        idx = torch.randint(0, self.data.size(0), (batch_size,))
        x = self.data[idx, :-1]
        y = self.data[idx, 1:]
        # Important: we will ignore PAD in loss via ignore_index
        return x.to(DEVICE), y.to(DEVICE)


class LineDatasetSubset(LineDataset):
    def __init__(self, parent: LineDataset, rows: torch.Tensor):
        self.vocab = parent.vocab
        self.lines_enc = [parent.lines_enc[i] for i in rows.tolist()]
        self.max_len = max(len(e) for e in self.lines_enc) if self.lines_enc else 1
        # rebuild padded tensor for the classic random batching path
        data = []
        for e in self.lines_enc:
            pad_len = self.max_len - len(e)
            pad_id = self.vocab.pad_id
            data.append(e + [pad_id] * pad_len)
        self.data = torch.tensor(data, dtype=torch.long) if data else torch.empty(0, 1, dtype=torch.long)
from linegenModel import *

class LineTBPTTStream:
    """
    Maintains B independent streams over a set of encoded lines (each already has BOS at index 0).
    Windowed TBPTT: returns (x, y, reset_mask) each step with shape (B, W).

    - All streams start at BOS on the first step.
    - The last, shorter piece of every line is PAD-padded rather than discarded, so its EOS target is trained.
    - A stream is reset before its next line starts, and reset_mask marks that new-line window.

    `dataset` may be an IndexedLineDataset (or subset), avoiding a full in-memory
    materialization of a large line corpus. `lines_enc` remains supported for the
    legacy in-memory LineDataset.
    """
    def __init__(self, window: int, batch_size: int, bos_id: int, pad_id: int,
                 lines_enc: Optional[List[List[int]]] = None,
                 dataset: Optional['IndexedLineDataset'] = None):
        self.bos_id = bos_id
        self.pad_id = pad_id
        self.W = max(1, int(window))
        self.B = int(batch_size)

        self.dataset = dataset
        self.lines: List[torch.Tensor] = []
        if dataset is None:
            self.lines = [torch.tensor(line, dtype=torch.long) for line in (lines_enc or [])
                          if len(line) >= 2]
            if not self.lines:
                # Degenerate fallback: a complete empty line, BOS -> EOS.
                self.lines = [torch.tensor([bos_id, bos_id], dtype=torch.long)]
        elif len(dataset.offsets) == 0:
            raise ValueError("Line-mode TBPTT needs at least one non-empty indexed line")

        # Per-stream (line_idx, pos)
        self.line_idx = torch.zeros(self.B, dtype=torch.long)
        self.pos = torch.zeros(self.B, dtype=torch.long)
        self._init_streams()

        # An epoch in line mode means one pass over *tokens*, not one TBPTT
        # window per line.  The old training loop used ``num_lines / batch``
        # steps, which processes only the first window of an average line per
        # epoch.  For a 512-token line and a 64-token TBPTT window this made an
        # RNN receive roughly one eighth as much training signal as models that
        # train on whole lines.
        self.total_transitions = sum(
            max(0, self._get_line(i).numel() - 1)
            for i in range(len(self.dataset.offsets) if self.dataset is not None else len(self.lines))
        )

    def _pick_line(self) -> int:
        return random.randrange(len(self.dataset.offsets)) if self.dataset is not None \
            else random.randrange(len(self.lines))

    def _get_example(self, line_idx: int) -> Tuple[torch.Tensor, int]:
        """Encoded line and its count of masked leading targets (seq2seq source)."""
        if self.dataset is not None:
            ids, masked = self.dataset.get_encoded_example(line_idx)
            return torch.tensor(ids, dtype=torch.long), masked
        return self.lines[line_idx], 0

    def _get_line(self, line_idx: int) -> torch.Tensor:
        return self._get_example(line_idx)[0]

    def _init_streams(self):
        # All streams begin at BOS of a random line.
        for b in range(self.B):
            self.line_idx[b] = self._pick_line()
            self.pos[b] = 0

    def get_next(self, device):
        B = self.B; W = self.W
        x = torch.full((B, W), self.pad_id, dtype=torch.long)
        y = torch.full((B, W), self.pad_id, dtype=torch.long)
        reset = torch.zeros(B, dtype=torch.bool)

        for b in range(B):
            li = int(self.line_idx[b].item())
            line, masked = self._get_example(li)
            L = line.numel()
            p = int(self.pos[b].item())

            # The previous call consumed this line's final transition. Start a
            # fresh line now, resetting recurrent state before its BOS token.
            if p >= L - 1:
                li = self._pick_line()
                line, masked = self._get_example(li); L = line.numel()
                p = 0
                self.line_idx[b] = li
                self.pos[b] = 0
                reset[b] = True

            # Keep the final short chunk and pad it. In particular, EOS is a
            # valid target even when it falls before a full TBPTT window.
            n = min(W, L - p - 1)
            x[b, :n] = line[p : p + n]
            y[b, :n] = line[p + 1 : p + n + 1]
            # Seq2seq source targets (line positions < masked) carry no loss.
            if masked > p:
                y[b, :min(n, masked - p)] = self.pad_id

            # Advance
            self.pos[b] = p + n

        return x.to(device), y.to(device), reset.to(device)


# ==================================================
class MemmapClassicDataset:
    def __init__(self, txt_path: str, vocab: BaseVocab, seq_len: int, split_range=(0.0, 1.0)):
        self.seq_len = seq_len
        self.vocab = vocab
        src_path = pathlib.Path(txt_path)
        self.bin_path = str(src_path.with_name(f"{src_path.stem}.{self._cache_signature(vocab)}.bin"))
        
        cache_is_stale = (
            os.path.exists(self.bin_path)
            and os.path.getmtime(self.bin_path) < os.path.getmtime(txt_path)
        )
        if not os.path.exists(self.bin_path) or cache_is_stale:
            if cache_is_stale:
                print(f"[Dataset] Source changed; rebuilding {self.bin_path} ...")
            print(f"[Dataset] Pre-tokenizing {txt_path} -> {self.bin_path} ...")
            self._tokenize_and_save(txt_path)
        else:
            print(f"[Dataset] Found existing {self.bin_path}, loading...")

        self.dtype = np.uint16 if vocab.size <= 65536 else np.int32
        self.full_data = np.memmap(self.bin_path, dtype=self.dtype, mode='r')
        
        # === NEW: Slice the memmap logically ===
        total_len = len(self.full_data)
        if total_len < self.seq_len + 1:
            raise ValueError(
                f"Dataset has only {total_len} tokens, but seq_len={self.seq_len} "
                "requires at least seq_len + 1 tokens."
            )
        start_pct, end_pct = split_range
        self.start_idx = int(start_pct * total_len)
        self.end_idx = int(end_pct * total_len)
        
        # Safety: ensure we have at least one sequence
        if self.end_idx - self.start_idx <= seq_len + 1:
            print(f"[Dataset] Warning: Split {split_range} is too small! Using full.")
            self.start_idx = 0
            self.end_idx = total_len
            
        print(f"[Dataset] Loaded segment {split_range} ({readable_num(self.end_idx - self.start_idx)} tokens).")

    @staticmethod
    def _cache_signature(vocab: BaseVocab) -> str:
        payload = {
            "class": vocab.__class__.__name__,
            "size": int(vocab.size),
            "line_mode": bool(getattr(vocab, "line_mode", False)),
            "bos_id": getattr(vocab, "bos_id", None),
            "eos_id": getattr(vocab, "eos_id", None),
            "pad_id": getattr(vocab, "pad_id", None),
        }
        if isinstance(vocab, TiktokenVocab):
            payload["encoding"] = getattr(vocab.enc, "name", None)
            payload["n_base"] = vocab.n_base
        elif isinstance(vocab, CustomBPEVocab):
            payload["vocab_path"] = getattr(vocab, "vocab_path", "")
            payload["merges"] = sorted((str(k), v) for k, v in vocab.merges.items())
        elif hasattr(vocab, "tokens"):
            payload["tokens"] = list(getattr(vocab, "tokens"))
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return "tok-" + hashlib.sha256(raw).hexdigest()[:12]

    # ... (keep _tokenize_and_save exactly as it was) ...
    def _tokenize_and_save(self, txt_path):
        # [Use your existing code here, no changes needed]
        # Just ensure you copy the method from your file into this class
        dtype = np.uint16 if self.vocab.size <= 65536 else np.int32
        file_size = os.path.getsize(txt_path)
        temp_path = self.bin_path + ".tmp"
        CHUNK_SIZE = 1024 * 1024 
        
        # Byte and binary tokenizers must receive the original bytes.  Reading
        # through UTF-8 with ``errors='ignore'`` silently dropped arbitrary
        # binary data before it ever reached ByteVocab.
        raw_input = isinstance(self.vocab, (ByteVocab, BinaryVocab))
        input_kwargs = {} if raw_input else {"encoding": "utf-8", "errors": "ignore"}
        empty_buffer = b"" if raw_input else ""
        with open(temp_path, "wb") as f_out:
            with open(txt_path, "rb" if raw_input else "r", **input_kwargs) as f_in:
                with tqdm(total=file_size, unit="B", unit_scale=True, desc="Tokenizing") as pbar:
                    buffer = empty_buffer
                    while True:
                        chunk = f_in.read(CHUNK_SIZE)
                        if not chunk:
                            if buffer: 
                                ids = self.vocab.encode(buffer)
                                f_out.write(np.array(ids, dtype=dtype).tobytes())
                            break
                        buffer += chunk
                        pbar.update(len(chunk) if raw_input else len(chunk.encode('utf-8')))
                        
                        if hasattr(self.vocab, "enc") or isinstance(self.vocab, TiktokenVocab): 
                            last_nl = buffer.rfind('\n')
                            if last_nl != -1:
                                to_process = buffer[:last_nl+1]
                                buffer = buffer[last_nl+1:]
                                ids = self.vocab.encode(to_process)
                                f_out.write(np.array(ids, dtype=dtype).tobytes())
                            elif len(buffer) > 10 * CHUNK_SIZE:
                                ids = self.vocab.encode(buffer)
                                f_out.write(np.array(ids, dtype=dtype).tobytes())
                                buffer = empty_buffer
                        else:
                            ids = self.vocab.encode(buffer)
                            f_out.write(np.array(ids, dtype=dtype).tobytes())
                            buffer = empty_buffer

        if os.path.exists(self.bin_path): os.remove(self.bin_path)
        os.rename(temp_path, self.bin_path)

    def get_batch(self, batch_size: int):
        # Effective length of OUR slice
        slice_len = self.end_idx - self.start_idx
        high = slice_len - self.seq_len - 1
        
        if high <= 0: 
            return torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=DEVICE), \
                   torch.zeros(batch_size, self.seq_len, dtype=torch.long, device=DEVICE)
        
        # Pick random offsets relative to start_idx
        ix = torch.randint(0, high, (batch_size,)) + self.start_idx
        
        x_list = []
        y_list = []
        for i in ix:
            i_int = int(i)
            # Read from the memmap
            chunk = np.array(self.full_data[i_int : i_int + self.seq_len + 1], dtype=np.int64)
            x_list.append(torch.from_numpy(chunk[:-1]))
            y_list.append(torch.from_numpy(chunk[1:]))
            
        x = torch.stack(x_list).to(DEVICE)
        y = torch.stack(y_list).to(DEVICE)
        return x, y
    
    @property
    def ids(self):
        # Return only the slice if requested via property (for TBPTT).
        # This stays a memmap view: converting the whole split to int64 here
        # would load it into RAM at 8 bytes per token.  TBPTTClassicStream
        # converts one window at a time instead.
        return self.full_data[self.start_idx : self.end_idx]

class IndexedLineDataset:
    """
    Efficient Line-based dataset that indexes file offsets instead of loading lines.
    Allows random access to 10GB+ line-based files with minimal RAM.
    """
    def __init__(self, txt_path: str, vocab: BaseVocab, seq2seq_inputs: int = 0):
        self.path = txt_path
        self.vocab = vocab
        # >0: lines are prepared seq2seq examples with this many input columns.
        self.seq2seq_inputs = int(seq2seq_inputs)
        # FIX: Explicitly add .npy so numpy doesn't silently append it later
        self.index_path = str(pathlib.Path(txt_path).with_suffix(".idx.npy"))
        
        if not os.path.exists(self.index_path):
            print(f"[Dataset] Indexing lines in {txt_path} ...")
            self._build_index()
        else:
            print(f"[Dataset] Loading line index {self.index_path} ...")
        
        self.offsets = np.load(self.index_path)
        if len(self.offsets) == 0:
            raise ValueError("Line-mode dataset contains no non-empty lines.")
        print(f"[Dataset] Indexed {readable_num(len(self.offsets))} lines.")
        self.max_len = int(np.max(self.offsets[:, 1])) if len(self.offsets) > 0 else 0
        self.f = open(self.path, "rb")

    def _build_index(self):
        offsets = []
        cur_offset = 0
        seen_hashes = set()
        
        with open(self.path, "rb") as f:
            # We use tqdm for progress
            for line in tqdm(f, desc="Indexing"):
                length = len(line)
                
                # Strip for content check (handles \r\n, \n, whitespace)
                content = line.strip()
                
                # 1. Skip empty lines
                if len(content) == 0:
                    cur_offset += length
                    continue
                
                # 2. Skip duplicates using hash
                h = hash(content)
                if h in seen_hashes:
                    cur_offset += length
                    continue
                
                seen_hashes.add(h)
                offsets.append((cur_offset, length))
                cur_offset += length

        np.save(self.index_path, np.array(offsets, dtype=np.int64))

    def get_batch(self, batch_size: int):
        idx = np.random.randint(0, len(self.offsets), size=batch_size)
        examples = [self.get_encoded_example(int(i)) for i in idx]
        pad_id = self.vocab.pad_id if self.vocab.pad_id is not None else 0
        return pad_line_examples(examples, pad_id)

    def get_encoded_example(self, index: int) -> Tuple[List[int], int]:
        """Encoded line (with BOS/EOS) and how many leading targets to mask.

        The mask count is the seq2seq source length, or 0 for ordinary lines.
        """
        off, length = self.offsets[index]
        self.f.seek(int(off))
        line_bytes = self.f.read(int(length))

        if isinstance(self.vocab, (ByteVocab, BinaryVocab)):
            line = line_bytes.rstrip(b'\n\r')
        else:
            line = line_bytes.decode("utf-8", "ignore").rstrip('\n\r')
        if self.seq2seq_inputs:
            return encode_seq2seq_line(self.vocab, line, self.seq2seq_inputs)
        return self.vocab.encode(line), 0

    def get_encoded_line(self, index: int) -> List[int]:
        """Read and encode one indexed line, including its BOS/EOS markers."""
        return self.get_encoded_example(index)[0]
        
    def close(self):
        self.f.close()

class IndexedLineDatasetSubset(IndexedLineDataset):
    def __init__(self, parent: IndexedLineDataset, indices: np.ndarray):
        self.path = parent.path
        self.vocab = parent.vocab
        self.seq2seq_inputs = parent.seq2seq_inputs
        self.f = open(self.path, "rb")
        self.offsets = parent.offsets[indices]
        self.max_len = parent.max_len



# Update set of Scan models
ACT_MENU = activation_menu_text()
ACT_NAMES = activation_names()

ACTIVATION_DESCRIPTIONS = {
    "relu": "Fast, sparse positive activations",
    "gelu": "Transformer default; smooth Gaussian gate",
    "silu": "Smooth swish / sigmoid-weighted linear unit",
    "mish": "Smooth self-regularising activation",
    "swiglu": "SiLU-gated feed-forward block",
    "geglu": "GELU-gated feed-forward block",
    "miglu": "Mish-gated feed-forward block",
    "tanh": "Bounded symmetric activation",
    "sigmoid": "Bounded 0–1 activation",
    "elu": "Smooth negative branch",
    "lrelu": "Leaky ReLU (slope 0.2)",
    "leaky_relu": "Leaky ReLU (default slope)",
    "linear": "No non-linearity",
}
class TBPTTClassicStream:
    """
    Streaming TBPTT over a single long 1D tensor `ids`.
    Guarantees every (x,y) window has length exactly W, by choosing starts that have >= W+1 tokens left.
    One stream (b=0) always starts at position 0.
    """
    def __init__(self, ids, window: int, batch_size: int, total_len: int):
        # ``ids`` is a 1D torch tensor or a (memmap-backed) NumPy array.
        assert ids.ndim == 1, "ids must be 1D"
        self.ids = ids
        self.N = len(ids)
        self.W = max(1, int(window))
        self.B = int(batch_size)
        self.total_len = max(0, int(total_len))

        # If file is too short, clamp W so we can form (x,y)
        if self.N < self.W + 1:
            self.W = max(1, self.N - 1)

        # If a user provided a truncated segment smaller than W+1, clamp W
        if 0 < self.total_len < (self.W + 1):
            self.W = max(1, self.total_len - 1)

        # Per-stream position and segment end (exclusive)
        self.pos = torch.zeros(self.B, dtype=torch.long)
        self.seg_end = torch.zeros(self.B, dtype=torch.long)
        self._init_streams()

    def _new_start(self):
        """
        Choose a (start,end) such that end-start >= W+1.
        If total_len>0, we honor it (and we already clamped W accordingly).
        Otherwise choose start uniformly in [0, N-(W+1)] and end=N.
        """
        need = self.W + 1
        if self.total_len > 0:
            # segment length is fixed to total_len (>= need due to clamping)
            start_max = max(0, self.N - self.total_len)
            start = random.randrange(0, start_max + 1) if start_max > 0 else 0
            end = min(start + self.total_len, self.N)
        else:
            if self.N <= need:
                # degenerate but safe: whole file acts as a single window, W already clamped
                return 0, self.N
            start_max = self.N - need
            start = random.randrange(0, start_max + 1)
            end = self.N
        return start, end

    def _init_streams(self):
        # stream 0 starts at 0; ensure its segment has at least W+1 tokens
        self.pos[0] = 0
        if self.total_len > 0:
            self.seg_end[0] = min(self.total_len, self.N)
        else:
            self.seg_end[0] = self.N
        # if even stream 0's segment is too short, widen it safely
        if int(self.seg_end[0].item()) - int(self.pos[0].item()) < (self.W + 1):
            self.seg_end[0] = min(self.pos[0] + self.W + 1, self.N)

        # others random valid segments
        for b in range(1, self.B):
            s, e = self._new_start()
            self.pos[b] = s
            self.seg_end[b] = e

    def get_next(self, device):
        B, W = self.B, self.W
        x = torch.empty(B, W, dtype=torch.long)
        y = torch.empty(B, W, dtype=torch.long)
        reset = torch.zeros(B, dtype=torch.bool)

        need = W + 1
        for b in range(B):
            p = int(self.pos[b].item())
            e = int(self.seg_end[b].item())

            # If not enough room for a full window, resample a fresh valid segment and mark reset
            if (e - p) < need or (self.N - p) < need:
                s, ee = self._new_start()
                self.pos[b] = s
                self.seg_end[b] = ee
                p, e = s, ee
                reset[b] = True

            # Now guaranteed: e - p >= need
            window = self.ids[p : p + W + 1]
            if not torch.is_tensor(window):
                window = torch.from_numpy(np.asarray(window, dtype=np.int64))
            x[b] = window[:-1]
            y[b] = window[1:]
            self.pos[b] = p + W

            # If we exactly hit the boundary, next step will have to reset
            if (e - int(self.pos[b].item())) < need:
                reset[b] = True

        return x.to(device), y.to(device), reset.to(device)

def reset_rnn_state(state, reset_mask, model, msel):
    """Zero the hidden states for batch indices where reset_mask==True."""
    if state is None or reset_mask is None or reset_mask.numel() == 0:
        return state

    # --- BuiltinRNNWrapper: stacked 1-layer cores ---
    # GRU/RNN: state is List[Tensor] where each Tensor is (1, B, H)
    # LSTM:    state is List[Tuple[Tensor, Tensor]] where each is (1, B, H)
    if isinstance(model, BuiltinRNNWrapper):
        if model.mode == 'lstm':
            new_state = []
            for (h, c) in state:
                # shapes: (1, B, H); batch dimension is 1
                h[:, reset_mask, :] = 0
                c[:, reset_mask, :] = 0
                new_state.append((h, c))
            return new_state
        else:
            new_state = []
            for h in state:
                # shape: (1, B, H); batch dimension is 1
                h[:, reset_mask, :] = 0
                new_state.append(h)
            return new_state

    # --- CustomRNNWrapper with IndRNN/IndyGRU/JANET/LiquidRNN/ExtATanULSTM ---
    if isinstance(model, CustomRNNWrapper):
        # Every custom cell packs its state as (num_layers, B, H) tensors, or a
        # tuple of them for two-state cells (LSTM-style (h, c), UnICORNN (y, z)).
        if state is None:
            return None
        for part in (state if isinstance(state, tuple) else (state,)):
            part[:, reset_mask, :] = 0
        return state

    # --- xLSTM: list of dict states (per block) ---
    if isinstance(model, XlstmLM):
        if state is None: return None
        new_state = []
        for st in state:
            if st is None:
                new_state.append(None); continue
            st2 = {}
            for k,v in st.items():
                if v is None:
                    st2[k] = None
                elif torch.is_tensor(v) and v.dim() >= 2:
                    vv = v.clone()
                    # batch is dim 0 for these states
                    vv[reset_mask] = 0
                    st2[k] = vv
                else:
                    st2[k] = v
            new_state.append(st2)
        return new_state

    # --- ScanLM: dict/Tensor per block (batch is dim 0) ---
    if isinstance(model, ScanLM):
        if state is None:
            return None
        new_state = []
        for st in state:
            if st is None:
                new_state.append(None); continue
            if isinstance(st, dict):
                st2 = {}
                for k, v in st.items():
                    if torch.is_tensor(v) and v.dim() >= 2:
                        vv = v.clone()
                        vv[reset_mask] = 0
                        st2[k] = vv
                    else:
                        st2[k] = v
                new_state.append(st2)
            elif torch.is_tensor(st):
                st2 = st.clone()
                st2[reset_mask] = 0
                new_state.append(st2)
            else:
                new_state.append(st)
        return new_state

    # Unknown model: best-effort recursive zero on tensors assuming batch is first dim
    def _zero_any(x):
        if torch.is_tensor(x):
            if x.dim() >= 2:
                try:
                    x[reset_mask] = 0
                except Exception:
                    # fallback if batch is not leading dim
                    if x.dim() >= 3 and x.size(0) == 1:
                        x[:, reset_mask, :] = 0
            return x
        if isinstance(x, (list, tuple)):
            xs = [_zero_any(t) for t in x]
            return tuple(xs) if isinstance(x, tuple) else xs
        if isinstance(x, dict):
            return {k: _zero_any(v) for k, v in x.items()}
        return x

    return _zero_any(state)

def detach_state(state):
    """Detach hidden state from autograd graph (handles Tensor, tuple, list, dict, nested)."""
    if state is None:
        return None
    if torch.is_tensor(state):
        return state.detach()
    if isinstance(state, dict):
        return {k: detach_state(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        items = [detach_state(s) for s in state]
        return tuple(items) if isinstance(state, tuple) else items
    return state

def build_model(cfg, vocab_size):
    normalize_model_config(cfg)
    msel = cfg["model_selection"]
    try:
        spec = MODEL_SPECS[msel]
    except KeyError as exc:
        raise ValueError(f"Bad model selection: {msel}") from exc
    if spec.factory_key != msel:
        raise RuntimeError(f"Model registry factory mismatch for selection {msel}")
    embed = cfg["embed_dim"]
    layers = cfg["layer_count"]
    seq_len = cfg["seq_len"]
    heads = cfg.get("head_count", 4)
    if cfg["model_type"] in {MODEL_TYPE_MEGABYTE, MODEL_TYPE_MEGABYTE_BOTTOM_UP}:
        stage_mixer = (
            cfg.get("_legacy_megabyte_mixer")
            or cfg.get("megabyte_stage_mixers")
            or HIERARCHICAL_MODEL_MIXERS.get(msel)
        )
        if stage_mixer is None:
            raise ValueError(
                f"model selection {msel} ({spec.name}) has no MEGABYTE stage adapter"
            )
        stage_dims = cfg.get("megabyte_stage_dims")
        stage_depths = cfg.get("megabyte_stage_depths")
        stage_heads = cfg.get("megabyte_stage_heads")
        stage_seq_lens = cfg.get("megabyte_stage_seq_lens")
        stage_child_embed_dims = cfg.get("megabyte_stage_child_embed_dims")
        if stage_child_embed_dims is None:
            raise ValueError(
                "This MEGABYTE configuration predates bounded child embeddings and its checkpoint "
                "cannot be loaded safely. Start a new training run."
            )
        if not all(value is not None for value in (stage_dims, stage_depths, stage_heads, stage_seq_lens)):
            raise ValueError(
                "This MEGABYTE configuration is missing its per-stage topology. Start a new training run."
            )
        if cfg["model_type"] == MODEL_TYPE_MEGABYTE_BOTTOM_UP and cfg.get("megabyte_bottom_up_version") != 6:
            raise ValueError(
                "This bottom-up MEGABYTE checkpoint predates gated direct parent-context version 6 and "
                "cannot be loaded safely. Start a new training run."
            )
        selected_stage_mixers = (
            (stage_mixer,) if isinstance(stage_mixer, (str, int)) else tuple(stage_mixer)
        )
        for config_key in (
            "megabyte_bottom_up_encoder_stage_mixers",
            "megabyte_bottom_up_decoder_stage_mixers",
        ):
            mixers = cfg.get(config_key, ())
            selected_stage_mixers += (mixers,) if isinstance(mixers, (str, int)) else tuple(mixers)
        uses_builtin_rnn_stage = any(
            resolve_megabyte_stage_mixer(mixer) in {"rnn", "rnn_relu", "gru", "lstm"}
            for mixer in selected_stage_mixers
        )
        if uses_builtin_rnn_stage and cfg.get("megabyte_fused_rnn_version") != 2:
            raise ValueError(
                "This MEGABYTE RNN checkpoint predates parent-conditioned stage-state wiring "
                "and cannot be loaded safely. Start a new training run."
            )
        return MegaByteLM(
            vocab_size,
            stage_dims=stage_dims,
            stage_depths=stage_depths,
            stage_heads=stage_heads,
            stage_seq_lens=stage_seq_lens,
            stage_child_embed_dims=stage_child_embed_dims,
            stage_mixer=stage_mixer,
            hierarchy_mode=(
                "bottom_up" if cfg["model_type"] == MODEL_TYPE_MEGABYTE_BOTTOM_UP else "top_down"
            ),
            bottom_up_encoder_stage_mixer=cfg.get("megabyte_bottom_up_encoder_stage_mixers"),
            bottom_up_decoder_stage_mixer=cfg.get("megabyte_bottom_up_decoder_stage_mixers"),
            fused_rnn_norm_type=int(cfg.get("megabyte_fused_rnn_norm_type", 0)),
            fused_rnn_res_every=int(cfg.get("megabyte_fused_rnn_res_every", 0)),
            fused_rnn_res_type=int(cfg.get("megabyte_fused_rnn_res_type", 0)),
            fused_rnn_dropout=float(cfg.get("megabyte_fused_rnn_dropout", 0.0)),
        ).to(DEVICE)
    RNN_MAP = {
        509: "indrnn",
        513: "indygru",
        510: "janet",
        502: "atanulstm",
        601: "liquid",
        515: "mogrifier_lstm",
        506: "irnn",
        602: "unicornn",
        514: "indylstm",
        519: "lru",
        518: "rru",
        516: "mogrifier_gru",
        511: "exprnn",
        507: "ugrnn",
    }
    if msel in RNN_MAP:
        cell_type_str = RNN_MAP[msel]
        opts = cfg.get("rnn_cell_options", {})
        cell_kwargs = {
            "indrnn": {"activation": opts.get("indrnn_activation", "relu")},
            "indygru": {"relu_gates": bool(opts.get("relu_gates", False))},
            "indylstm": {"relu_gates": bool(opts.get("relu_gates", False))},
            "mogrifier_lstm": {"rounds": int(opts.get("mogrifier_rounds", 5))},
            "mogrifier_gru": {"rounds": int(opts.get("mogrifier_rounds", 5))},
            "unicornn": {"dt": float(opts.get("unicornn_dt", 0.1)), "alpha": float(opts.get("unicornn_alpha", 10.0))},
            "lru": {"highway": bool(opts.get("lru_highway", False))},
            "rru": {"middle_multiplier": float(opts.get("rru_middle_multiplier", 2.0)),
                    "dropout": float(opts.get("rru_dropout", 0.0))},
        }.get(cell_type_str, {})
        depth_options = {
            "use_norm": int(cfg.get("use_norm", 0)), "res_every": int(cfg.get("res_every", 0)),
            "res_type": int(cfg.get("res_type", 0)), "dropout": float(cfg.get("dropout", 0.0)),
            "use_multiplier": int(cfg.get("use_multiplier", 0)), "ffn": int(cfg.get("rnn_ffn", 0)),
        }
        return CustomRNNWrapper(cell_type_str, vocab_size, embed, cfg["layer_count"],
                                depth_options=depth_options, **cell_kwargs).to(DEVICE)
    if msel == 1009:  # Mamba-3
        return Mamba3LM(vocab_size, embed, layers).to(DEVICE)
    if msel == 1010:  # Mamba-3 MIMO
        return Mamba3LM(vocab_size, embed, layers, mimo_rank=int(cfg.get("mamba3_mimo_rank", 4))).to(DEVICE)
    if msel in (907, 908):  # ParaGRU / ParaLSTM (ParaRNN)
        return ParaRNNLM(vocab_size, embed, layers, kind="gru" if msel == 907 else "lstm",
                         newton_iters=int(cfg.get("pararnn_newton_iters", 3))).to(DEVICE)
    if msel == 520:  # M2RNN
        return M2RNNLM(vocab_size, embed, layers).to(DEVICE)
    if msel == 300:  # Original Transformer (Vaswani et al., 2017)
        return OriginalTransformerLM(
            vocab_size, embed, layers, heads, seq_len,
            act_name=cfg.get("activation_name", "relu"), dropout=float(cfg.get("dropout", 0.1)),
        ).to(DEVICE)
    if msel == 306:  # Trinity-style 2026 Transformer
        return TrinityTransformerLM(
            vocab_size, embed, layers, heads, seq_len,
            window=cfg.get("swa_window"), act_name=cfg.get("activation_name", "swiglu"),
        ).to(DEVICE)
    if msel == 517:  # SRU++
        return SRUppLM(vocab_size, embed, layers, max_cache=max(1, int(seq_len))).to(DEVICE)

    if msel == 505:
        return QRNNLM(vocab_size, embed, layers, kernel_size=int(cfg.get("qrnn_kernel_size", 2))).to(DEVICE)
    if msel == 508:
        return SRULM(vocab_size, embed, layers).to(DEVICE)
    if msel == 302:
        return SwitchMoELM(
            vocab_size, embed, layers, heads, seq_len,
            n_experts=int(cfg.get("moe_num_experts", 4)),
            ff_mult=int(cfg.get("ff_mult", 4)),
            dropout=float(cfg.get("dropout", 0.0)),
        ).to(DEVICE)
    if msel == 1201:
        return JambaLiteLM(vocab_size, embed, layers, heads).to(DEVICE)

    if msel == 1:
        # Basic MLP now uses rolling one-hot window input (MLPOG-style), needs seq_len
        return OneHotWindowMLPClassifier(
            vocab_size=vocab_size,
            seq_len=cfg["seq_len"],
            embed_dim=embed,
            n_layers=layers,
            act_name=cfg["activation_name"]
        ).to(DEVICE)

    if msel == 5:
        return ResidualMLPClassifier(vocab_size, embed, layers, cfg["activation_name"]).to(DEVICE)

    if msel == 500:
        return BuiltinRNNWrapper(
            vocab_size, embed, layers, 'rnn_tanh',
            tie_weights=bool(cfg.get("tie_weights", True)),
            use_norm=int(cfg.get("use_norm", 0)),
            res_every=int(cfg.get("res_every", 0)),
            res_type=int(cfg.get("res_type", 0)),
            dropout=float(cfg.get("dropout", 0.0)),
            use_multiplier=int(cfg.get("use_multiplier", 0))
        ).to(DEVICE)

    if msel == 504:
        return BuiltinRNNWrapper(
            vocab_size, embed, layers, 'rnn_relu',
            tie_weights=bool(cfg.get("tie_weights", True)),
            use_norm=int(cfg.get("use_norm", 0)),
            res_every=int(cfg.get("res_every", 0)),
            res_type=int(cfg.get("res_type", 0)),
            dropout=float(cfg.get("dropout", 0.0)),
            use_multiplier=int(cfg.get("use_multiplier", 0))
        ).to(DEVICE)

    if msel == 503:
        return BuiltinRNNWrapper(
            vocab_size, embed, layers, 'gru',
            tie_weights=bool(cfg.get("tie_weights", True)),
            use_norm=int(cfg.get("use_norm", 0)),
            res_every=int(cfg.get("res_every", 0)),
            res_type=int(cfg.get("res_type", 0)),
            dropout=float(cfg.get("dropout", 0.0)),
            use_multiplier=int(cfg.get("use_multiplier", 0))
        ).to(DEVICE)

    if msel == 501:
        return BuiltinRNNWrapper(
            vocab_size, embed, layers, 'lstm',
            tie_weights=bool(cfg.get("tie_weights", True)),
            use_norm=int(cfg.get("use_norm", 0)),
            res_every=int(cfg.get("res_every", 0)),
            res_type=int(cfg.get("res_type", 0)),
            dropout=float(cfg.get("dropout", 0.0)),
            use_multiplier=int(cfg.get("use_multiplier", 0))
        ).to(DEVICE)

    if msel == 102:
        return TemporalConvNet(vocab_size, embed, layers, act_name=cfg["activation_name"], k=3).to(DEVICE)
    if msel in (101, 100):
        return AutoregressiveConvLM(vocab_size, embed, layers, "wavenet" if msel == 101 else "pixelcnn").to(DEVICE)
    if msel == 104:
        return HyenaLM(vocab_size, embed, layers, seq_len).to(DEVICE)
    if msel == 103:
        return CausalConvNeXtLM(vocab_size, embed, layers).to(DEVICE)
    if msel == 208:
        return ToeplitzMLPMixerLM(vocab_size, embed, layers, seq_len).to(DEVICE)
    if msel == 209:
        return GrassmannMixerLM(vocab_size, embed, layers).to(DEVICE)
    if msel == 2:
        return NeuralNGramLM(vocab_size, embed, layers, int(cfg.get("ngram_context", 4))).to(DEVICE)
    if msel == 0:
        return MarkovBigramLM(vocab_size).to(DEVICE)
    if msel in (3, 4):
        return MaskedAutoregressiveMLP(vocab_size, embed, layers, seq_len, "nade" if msel == 3 else "made").to(DEVICE)
    if msel == 301:
        # GPT-2 style decoder-only LM
        cfg_gpt2 = GPT2Config(
            vocab_size=vocab_size,
            d_model=embed,
            n_layers=layers,
            n_heads=heads,
            max_seq_len=seq_len,
            ff_mult=int(cfg.get("ff_mult", 4)),
            dropout=float(cfg.get("dropout", 0.0)),
            attn_dropout=float(cfg.get("attn_dropout", cfg.get("dropout", 0.0))),
            bias=bool(cfg.get("bias", True)),
            tie_weights=bool(cfg.get("tie_weights", True)),
            use_flash=bool(cfg.get("use_flash", True)),
        )
        act_name = cfg.get("activation_name", "gelu")
        return GPT2ForLM(cfg_gpt2, act_name=act_name).to(DEVICE)
    if msel in (800, 801, 802):
        # Heads = cfg['head_count']; act = cfg['activation_name']
        # Mixed ratio (a:b) taken from cfg or defaults to 7:1
        a = int(cfg.get("xlstm_m_blocks", 7))
        b = int(cfg.get("xlstm_s_blocks", 1))
        kind = "s" if msel == 800 else ("m" if msel == 801 else "mix")
        # Paper-faithful xLSTM blocks (NX-AI xlstm): mLSTM blocks with sLSTM
        # blocks interleaved at the a:b ratio (xLSTM[7:1] by default).
        return XLSTMFullLM(vocab_size, embed, layers, num_heads=heads, kind=kind, m_to_s=(a, b)).to(DEVICE)
    if msel == 1006:  # Mamba (selective scan)
        return ScanLM(
            vocab_size=vocab_size,
            dim=embed,
            kind="mamba",
            n_blocks=cfg["layer_count"],
        ).to(DEVICE)
    if msel == 1007:  # Mamba selective SSM
        return ScanLM(vocab_size, embed, kind="mamba_ssm", n_blocks=layers).to(DEVICE)

    if msel == 900:  # minGRU (scan)
        return ScanLM(
            vocab_size=vocab_size,
            dim=embed,
            kind="mingru",
            n_blocks=cfg["layer_count"],
        ).to(DEVICE)

    if msel == 901:  # minLSTM (scan)
        return ScanLM(
            vocab_size=vocab_size,
            dim=embed,
            kind="minlstm",
            n_blocks=cfg["layer_count"],
        ).to(DEVICE)
    if msel == 1102:  # RWKV (scan)
        return ScanLM(
            vocab_size=vocab_size,
            dim=embed,
            kind="rwkv",
            n_blocks=cfg["layer_count"],
        ).to(DEVICE)
    if msel == 207:
        return HyperMixerLM(
            vocab_size=vocab_size,
            d_model=embed,
            n_layers=layers,
            d_hidden=int(cfg.get("hm_hidden", embed)),
            d_ff=int(cfg.get("hm_ff", 4*embed)),
            act_name=cfg.get("activation_name", "gelu"),
            max_seq_len=int(cfg.get("seq_len", 65536)),
            tie_hyper=bool(cfg.get("hm_tie", True)),
            dropout=float(cfg.get("dropout", 0.0)),
            n_heads=int(cfg.get("head_count", 4)),
            causal=bool(cfg.get("causal", True)),
        ).to(DEVICE)
    if msel == 1104:  # GateLoop (scan)
        return ScanLM(
            vocab_size=vocab_size,
            dim=embed,
            kind="gateloop",
            n_blocks=cfg["layer_count"],
        ).to(DEVICE)
    if msel == 200:  # gMLP
        return gMLPLanguageModel(vocab_size, embed, layers, embed*4, seq_len, act_name=cfg["activation_name"]).to(DEVICE)
    if msel == 201:  # aMLP
        return aMLPLanguageModel(vocab_size, embed, layers, embed*4, seq_len, d_attn=64, act_name=cfg["activation_name"]).to(DEVICE)

    if msel == 202: # Causal MLPMixer
        return CausalMLPMixer(vocab_size, embed, layers, seq_len, act_name=cfg["activation_name"]).to(DEVICE)
    
    if msel == 304: # Modern Transformer
        return ModernTransformer(vocab_size, embed, layers, heads, act_name=cfg["activation_name"]).to(DEVICE)
    if msel == 409:
        return SparseModernTransformerLM(
            vocab_size, embed, layers, heads, seq_len,
            local_window=int(cfg.get("sparse_local_window", min(512, max(1, seq_len)))),
            compression_block=int(cfg.get("sparse_compression_block", 32)),
            selected_blocks=int(cfg.get("sparse_selected_blocks", 16)),
            act_name=cfg.get("activation_name", "swiglu"),
        ).to(DEVICE)
    if msel == 1200: # Griffin
        return GriffinLM(vocab_size, embed, layers).to(DEVICE)

    if msel == 1101: # DeltaNet (fla-faithful)
        return LinearRecurrentLM(vocab_size, embed, layers, "deltanet").to(DEVICE)

    if msel == 1103: # RetNet (multi-scale retention, fla-faithful)
        return LinearRecurrentLM(vocab_size, embed, layers, "retnet").to(DEVICE)

    if msel == 1105: # HGRN
        return HGRN_LM(vocab_size, embed, layers).to(DEVICE)
        
    if msel == 902: # MinRNN (Generalized)
        return ScanLM(vocab_size, embed, kind="minrnn", n_blocks=layers, minrnn_act=cfg.get("minrnn_act", 0)).to(DEVICE)

    if msel == 903: # MinIndRNN
        return ScanLM(vocab_size, embed, kind="minindrnn", n_blocks=layers, minrnn_act=cfg.get("minrnn_act", 0)).to(DEVICE)

    if msel == 904: # MinJANET
        return ScanLM(vocab_size, embed, kind="minjanet", n_blocks=layers).to(DEVICE)

    if msel == 305: # KAN-Transformer
        return KAN_LM(vocab_size, embed, layers, n_heads=heads, act_name=cfg["activation_name"]).to(DEVICE)

    if msel == 1100: # Linear Transformer
        return LinearTransformerLM(vocab_size, embed, layers).to(DEVICE)

    if msel == 1004: # H3
        return H3LM(vocab_size, embed, layers).to(DEVICE)

    if msel == 303: # DCT-Former
        return DCTFormerLM(vocab_size, embed, layers, seq_len, act_name=cfg["activation_name"]).to(DEVICE)
    
    if msel == 905: # MinIndyGRU
        return ScanLM(vocab_size, embed, kind="minindygru", n_blocks=layers).to(DEVICE)

    if msel == 906: # MinIndyLSTM
        return ScanLM(vocab_size, embed, kind="minindylstm", n_blocks=layers).to(DEVICE)

    if msel == 407: # Recurrent Interface Model
        return RecurrentInterfaceLM(
            vocab_size,
            embed,
            layers,
            dropout=float(cfg.get("dropout", 0.0)),
            num_latents=int(cfg.get("rin_num_latents", 8)),
            num_heads=int(cfg.get("rin_num_heads", cfg.get("head_count", 4))),
        ).to(DEVICE)

    recurrent_kinds = {512: "nru", 600: "lmu", 603: "cfc"}
    if msel in recurrent_kinds:
        return StatefulCellLM(
            vocab_size, embed, layers, recurrent_kinds[msel],
            dropout=float(cfg.get("dropout", 0.0)), lmu_theta=seq_len,
        ).to(DEVICE)

    if msel == 1108:  # RWKV-7 "Goose"
        return RWKV7LM(vocab_size, embed, layers).to(DEVICE)
    modern_kinds = {1107: "gated_deltanet", 1008: "mamba2", 1106: "hgrn2"}
    if msel in modern_kinds:
        return LinearRecurrentLM(vocab_size, embed, layers, modern_kinds[msel]).to(DEVICE)
    if msel == 400:
        return TransformerXLLM(vocab_size, embed, layers, heads=heads, mem_len=int(cfg.get("txl_mem_len", 128))).to(DEVICE)
    if msel == 408:
        return TitansLM(vocab_size, embed, layers, heads=heads, memory_slots=int(cfg.get("titans_memory_slots", 32))).to(DEVICE)

    ssm_kinds = {1000: "s4", 1002: "s4d", 1003: "s5", 1001: "dss", 1005: "lru"}
    if msel in ssm_kinds:
        return StructuredSSMLM(vocab_size, embed, layers, ssm_kinds[msel]).to(DEVICE)
    memory_kinds = {401: "compressive", 406: "memorizing", 402: "knn", 405: "retro"}
    if msel in memory_kinds:
        return CausalMemoryLM(vocab_size, embed, layers, memory_kinds[msel]).to(DEVICE)
    if msel in (403, 404):
        return SparseCausalTransformerLM(vocab_size, embed, layers, heads, seq_len, "longformer" if msel == 403 else "bigbird").to(DEVICE)
    if msel in (700, 701):
        return LatentRecurrentLM(vocab_size, embed, layers, "vrnn" if msel == 700 else "srnn").to(DEVICE)

    mlp_variants = {206: "pnlp", 205: "dyna", 204: "wave", 203: "ccs"}
    if msel in mlp_variants:
        return CausalMLPFamilyLM(vocab_size, embed, layers, seq_len, mlp_variants[msel], dropout=float(cfg.get("dropout", 0.0)), act_name=cfg["activation_name"]).to(DEVICE)


    raise RuntimeError(f"Registered model {msel} has no construction branch")

@torch.no_grad()
def sample_step(logits, temperature=1.0, top_k=0, top_p=0.0,
                repetition_penalty=1.0, last_tokens=None):
    """
    Advanced sampling with top-k, top-p (nucleus), and repetition penalty.
    The penalty applies once to each distinct token in ``last_tokens``,
    also for greedy decoding (temperature 0).
    """
    # Apply repetition penalty
    if repetition_penalty != 1.0 and last_tokens is not None and len(last_tokens) > 0:
        logits = logits.clone()
        penalty_ids = torch.tensor(sorted(set(int(t) for t in last_tokens)), dtype=torch.long, device=logits.device)
        if logits.dim() == 1:
            for pid in penalty_ids:
                if logits[pid] > 0:
                    logits[pid] /= repetition_penalty
                else:
                    logits[pid] *= repetition_penalty
        else:
            for pid in penalty_ids:
                mask_pos = logits[:, pid] > 0
                logits[:, pid] = torch.where(mask_pos, logits[:, pid] / repetition_penalty,
                                              logits[:, pid] * repetition_penalty)

    if temperature <= 0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature
    
    # Top-k filtering
    if top_k > 0:
        v = logits.size(-1)
        top_k = min(top_k, v)
        topk_vals, _ = torch.topk(logits, top_k, dim=-1)
        threshold = topk_vals[..., -1:]
        logits = torch.where(logits < threshold, torch.full_like(logits, float('-inf')), logits)
    
    # Top-p (nucleus) filtering
    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float('-inf')
        logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)
    
    probs = F.softmax(logits, dim=-1)
    
    if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
        vocab_size = logits.size(-1)
        if logits.dim() > 1:
            return torch.randint(0, vocab_size, (logits.size(0),), device=logits.device)
        else:
            return torch.randint(0, vocab_size, (1,), device=logits.device).squeeze(-1)

    return torch.multinomial(probs, num_samples=1).squeeze(-1)
def sample_next(logits, cfg, history):
    """Sample the token after ``logits[:, -1]`` with the cfg's sampling settings.

    The repetition penalty (``_rep_penalty``) covers the distinct tokens among
    the last ``_rep_window`` (default 64) tokens of ``history``: prompt and
    generated text alike."""
    penalty = float(cfg.get("_rep_penalty", 1.0))
    recent = history[-max(1, int(cfg.get("_rep_window", 64))):] if penalty != 1.0 else None
    return sample_step(logits[:, -1, :], cfg.get("temperature", 1.0), top_k=cfg.get("_top_k", 0),
                       top_p=cfg.get("_top_p", 0.0), repetition_penalty=penalty,
                       last_tokens=recent).item()


@torch.no_grad()
def _init_scan_state_from_prompt(scan_model: ScanLM, idx_prompt: torch.Tensor):
    """
    Given a full prompt (B=1, T>=1), run the blocks' parallel path to
    initialize per-block states, then return that state list.
    """
    # FIX: Apply the embedding LayerNorm!
    # Without this, the hidden states are initialized with un-normalized embedding magnitudes,
    # causing immediate distribution shift and garbage output.
    x = scan_model.emb_ln(scan_model.embed(idx_prompt))
    
    states = []
    for b in scan_model.blocks:
        if isinstance(b, (ScanBlock_Mamba, RWKVBlock)):
            x, st = b.forward_seq(x, state=None)
        else:
            x, st_last = b.forward_seq(x, h0=None)  # st_last: (B,D)
            st = st_last
        states.append(st)
    return states

@torch.no_grad()
def generate_classic(model, cfg, vocab: CharVocab, prompt_ids: List[int], max_len: int, stream=True,
                     on_token=None):
    """on_token(token_id, raw_logits_row) after every sampled token (GUI); returning True stops."""
    msel = cfg["model_selection"]
    seq_len = cfg["seq_len"]
    model.eval()
    out_ids = list(prompt_ids)

    # === STATEFUL / MEGABYTE CACHE PATH ===
    uses_megabyte_cache = isinstance(model, MegaByteLM) and model.is_incremental
    if uses_megabyte_cache or ((msel in SCAN_MODEL_IDS or msel in RNN_MODEL_IDS) and (
        not is_bottom_up_megabyte(cfg) or getattr(model, "is_incremental", False)
    )):
        # [Logic preserved from your provided file, ensuring robustness]
        state = None
        
        # 1. Warmup / Init State
        # If we have a prompt, we scan all tokens *except* the last one to build state.
        if len(prompt_ids) > 1:
            # Context is everything up to the last token
            ctx_ids = prompt_ids[:-1]
            x_ctx = torch.tensor([ctx_ids], dtype=torch.long, device=DEVICE)
            
            if isinstance(model, ScanLM):
                state = _init_scan_state_from_prompt(model, x_ctx)
            else:
                _, state = model(x_ctx, None)
            
            cur = prompt_ids[-1]
        elif len(prompt_ids) == 1:
            cur = prompt_ids[0]
        else:
            cur = random.randrange(vocab.size)
            # If prompt was empty, we generated a token, so we should add it to out_ids
            if len(out_ids) == 0: out_ids.append(cur)

        # 2. Generation Loop
        for _ in range(max_len):
            x = torch.tensor([[cur]], dtype=torch.long, device=DEVICE)
            logits, state = model(x, state)
            raw = logits[0, -1].detach().float().clone() if on_token is not None else None
            nxt = sample_next(logits, cfg, out_ids)
            out_ids.append(nxt)
            if stream:
                sys.stdout.write(vocab.decode([nxt])); sys.stdout.flush()
            cur = nxt
            if on_token is not None and on_token(nxt, raw):
                break

    # === SLIDING WINDOW PATH (MLP / Transformers / Mixers) ===
    else:
        # 1. Initialize Context
        # Ensure we have a valid starting context window
        if len(prompt_ids) > 0:
            cur_ctx = prompt_ids[-seq_len:]
        else:
            # If empty prompt, seed with random token
            start_token = random.randrange(vocab.size)
            cur_ctx = [start_token]
            out_ids.append(start_token)
            if stream:
                sys.stdout.write(vocab.decode([start_token])); sys.stdout.flush()

        # 2. Generation Loop
        for _ in range(max_len):
            # Ensure context doesn't exceed seq_len (safety clip)
            cur_ctx = cur_ctx[-seq_len:]
            
            x = torch.tensor([cur_ctx], dtype=torch.long, device=DEVICE)
            logits = model(x)
            
            # Sample from the last position
            raw = logits[0, -1].detach().float().clone() if on_token is not None else None
            nxt = sample_next(logits, cfg, out_ids)
            out_ids.append(nxt)
            
            if stream:
                sys.stdout.write(vocab.decode([nxt])); sys.stdout.flush()
            
            # Slide window
            cur_ctx.append(nxt)
            if on_token is not None and on_token(nxt, raw):
                break

    if stream: print()
    return out_ids


@torch.no_grad()
def benchmark_megabyte_sampling_cache(model: MegaByteLM, prompt_ids: List[int], steps: int = 256):
    """Measure incremental MEGABYTE sampling against the former window replay.

    This is intentionally a callable utility rather than an interactive-menu
    option: it is useful for a loaded MEGABYTE checkpoint and keeps benchmark
    results tied to the exact model, device, and hierarchy being sampled.
    """
    if not isinstance(model, MegaByteLM) or not model.is_incremental:
        raise ValueError("benchmark requires an incremental MegaByteLM")
    if not prompt_ids:
        raise ValueError("benchmark requires at least one prompt token")
    model.eval()
    device = next(model.parameters()).device
    seed = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    token = seed[:, -1:]
    agreement_steps = min(steps, model.max_seq_len - seed.size(1))
    if agreement_steps < 1:
        raise ValueError("benchmark prompt must leave room for one in-window agreement step")

    def run_incremental(forced_tokens=None):
        _, cache = model(seed, None)
        logits_stream = []
        run_steps = len(forced_tokens) if forced_tokens is not None else steps
        current = forced_tokens[0] if forced_tokens is not None else token
        for step in range(run_steps):
            logits, cache = model(current, cache)
            logits_stream.append(logits[:, -1])
            current = (
                forced_tokens[step + 1] if forced_tokens is not None and step + 1 < run_steps
                else logits[:, -1].argmax(dim=-1, keepdim=True)
            )
        return torch.stack(logits_stream, dim=1), cache

    def run_replay(run_steps=steps, forced_tokens=None):
        history = seed
        current = token
        logits_stream, inputs = [], []
        for step in range(run_steps):
            history = torch.cat((history, current), dim=1)
            logits = model._forward_full(history[:, -model.max_seq_len:])
            logits_stream.append(logits[:, -1])
            inputs.append(current)
            current = (
                forced_tokens[step] if forced_tokens is not None
                else logits[:, -1].argmax(dim=-1, keepdim=True)
            )
        return torch.stack(logits_stream, dim=1), inputs

    # Compare a forced, in-window continuation so sampling cannot hide a
    # numerical divergence by taking different argmax branches.
    replay_logits, forced_tokens = run_replay(agreement_steps)
    incremental_logits, _ = run_incremental(forced_tokens)
    if not torch.allclose(incremental_logits, replay_logits, rtol=2e-5, atol=2e-5):
        max_error = (incremental_logits - replay_logits).abs().max().item()
        raise AssertionError(f"MEGABYTE cache logits diverged from replay (max error {max_error:.3e})")

    def measure(fn):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        result = fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            memory = torch.cuda.max_memory_allocated(device)
        else:
            memory = None
        return steps / (time.perf_counter() - started), memory, result

    incremental_rate, incremental_memory, (_, cache) = measure(run_incremental)
    replay_rate, replay_memory, _ = measure(run_replay)
    return {
        "incremental_tokens_per_second": incremental_rate,
        "replay_tokens_per_second": replay_rate,
        "speedup": incremental_rate / replay_rate if replay_rate else float("inf"),
        "incremental_peak_cuda_bytes": incremental_memory,
        "replay_peak_cuda_bytes": replay_memory,
        "cache_stages": model.num_stages,
        "replay_window_tokens": model.max_seq_len,
        "agreement_steps": agreement_steps,
    }


@torch.no_grad()
def generate_line_mode(model, cfg, vocab: BaseVocab, prompt_ids: List[int], limit_len: int,
                       return_stop_reason: bool = False, on_token=None):
    """on_token(token_id, raw_logits_row) after every sampled token (GUI); returning True stops."""
    msel = cfg["model_selection"]
    seq_len = cfg["seq_len"]
    bos = vocab.bos_id
    eos = getattr(vocab, "eos_id", None)

    model.eval()
    stop = False

    # ``vocab.encode`` appends EOS in line mode.  EOS terminates a training
    # example, so it must not be part of a prompt that we intend to continue.
    # Keep the returned IDs aligned with what the model actually consumed too;
    # previously the stale EOS was displayed between the prompt and generated
    # text.
    priming = list(prompt_ids)
    if eos is not None and priming and priming[-1] == eos:
        priming = priming[:-1]
    if not priming:
        priming = [bos]
    out_ids = list(priming)
    
    # === STATEFUL / MEGABYTE CACHE PATH ===
    uses_megabyte_cache = isinstance(model, MegaByteLM) and model.is_incremental
    if uses_megabyte_cache or ((msel in SCAN_MODEL_IDS or msel in RNN_MODEL_IDS) and (
        not is_bottom_up_megabyte(cfg) or getattr(model, "is_incremental", False)
    )):
        # Prime through the model's normal forward path.  This is deliberately
        # not split into a history pass plus a separate one-token pass: ScanLM
        # has distinct parallel (training) and sequential (eval) implementations
        # and a split path can leave its carried state out of sync with the logits
        # used during training.
        x = torch.tensor([priming], dtype=torch.long, device=DEVICE)
        logits, state = model(x, None)
        for _ in range(limit_len):
            raw = logits[0, -1].detach().float().clone() if on_token is not None else None
            nxt = sample_next(logits, cfg, out_ids)
            out_ids.append(nxt)
            if on_token is not None and on_token(nxt, raw) and not (eos is not None and nxt == eos):
                break
            if eos is not None and nxt == eos:
                stop = True; break
            x = torch.tensor([[nxt]], dtype=torch.long, device=DEVICE)
            logits, state = model(x, state)

    # === SLIDING WINDOW PATH (MLP / Transformers / Mixers) ===
    else:
        # 1. Initialize Context
        # If priming exists, take the last seq_len tokens.
        # If priming is empty (empty prompt), start with [BOS].
        cur_ctx = priming[-seq_len:]

        for _ in range(limit_len):
            # Ensure context doesn't exceed seq_len
            cur_ctx = cur_ctx[-seq_len:]
            
            x = torch.tensor([cur_ctx], dtype=torch.long, device=DEVICE)
            logits = model(x)
            
            # Sample
            raw = logits[0, -1].detach().float().clone() if on_token is not None else None
            nxt = sample_next(logits, cfg, out_ids)
            out_ids.append(nxt)
            
            # Slide window
            cur_ctx.append(nxt)
            
            if eos is not None and nxt == eos:
                if on_token is not None:
                    on_token(nxt, raw)
                stop = True; break
            if on_token is not None and on_token(nxt, raw):
                break

    # Remove the trailing EOS from the result if present (optional, standardizes output)
    if stop and len(out_ids) > 0 and eos is not None and out_ids[-1] == eos:
        out_ids = out_ids[:-1]
        
    if return_stop_reason:
        return out_ids, ("EOS" if stop else f"max_len={limit_len}")
    return out_ids

# ========= Training =========
def loss_metrics(total_nll: float, token_count: int, cfg, correct_tokens: Optional[int] = None) -> Dict[str, float]:
    """Convert token-weighted negative log-likelihood into comparable metrics."""
    nll = total_nll / max(1, token_count)
    bits_per_token = nll / math.log(2)
    metrics = {
        "nll": nll,
        "nats_per_token": nll,
        "bits_per_token": bits_per_token,
        "perplexity": math.exp(min(nll, 20)),
    }
    tokenizer_mode = int(cfg.get("tokenizer_mode", 1))
    if tokenizer_mode == -1:
        metrics["bits_per_byte"] = bits_per_token * 8
    elif tokenizer_mode == 0:
        # One byte-level token represents exactly one byte, so BPB is the
        # comparable headline metric and numerically equals BPC here.
        metrics["bpc"] = bits_per_token
        metrics["bits_per_byte"] = bits_per_token
    elif tokenizer_mode == 1:
        metrics["bpc"] = bits_per_token
    if correct_tokens is not None:
        metrics["accuracy"] = correct_tokens / max(1, token_count)
    return metrics


def format_loss_metrics(metrics: Dict[str, float], *, include_accuracy: bool = False) -> str:
    """Format the metrics appropriate to the active tokenizer without hiding NLL."""
    parts = [f"nll/nats {metrics['nll']:.4f}"]
    if "bits_per_byte" in metrics:
        parts.append(f"bpb {metrics['bits_per_byte']:.4f}")
    elif "bpc" in metrics:
        parts.append(f"bpc {metrics['bpc']:.4f}")
    else:
        parts.append(f"bits/tok {metrics['bits_per_token']:.4f}")
    parts.append(f"ppl {metrics['perplexity']:.2f}")
    if include_accuracy and "accuracy" in metrics:
        parts.append(f"acc {metrics['accuracy']:.1%}")
    return " | ".join(parts)


def append_validation_metrics(
    cfg, metrics: Dict[str, float], *, step: int, epoch: int, train_tokens: int,
) -> None:
    """Append one validation observation with stable, spreadsheet-friendly columns."""
    path = pathlib.Path(cfg.get("validation_csv_path", VALIDATION_CSV_PATH))
    fields = (
        "step", "epoch", "train_tokens", "valid_tokens", "nll", "nats_per_token", "bits_per_token",
        "bpc", "bits_per_byte", "perplexity", "accuracy",
    )
    new_file = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    if not new_file:
        with path.open(newline="", encoding="utf-8") as handle:
            existing_rows = list(csv.DictReader(handle))
            existing_fields = tuple(existing_rows[0].keys()) if existing_rows else ()
        if existing_fields != fields:
            # Upgrade the original ambiguous `tokens` column to `valid_tokens`
            # without losing prior observations. Old rows cannot recover exact
            # train-token totals because those were not recorded at the time.
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for old_row in existing_rows:
                    writer.writerow({
                        field: old_row.get(field, old_row.get("tokens", "") if field == "valid_tokens" else "")
                        for field in fields
                    })
            new_file = False
    row = {field: metrics.get(field, "") for field in fields}
    row.update({"step": step, "epoch": epoch, "train_tokens": train_tokens,
                "valid_tokens": metrics.get("tokens", "")})
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def training_steps_per_epoch(cfg, dataset, *, line_stream=None, tbptt_stream=None) -> int:
    """Return updates needed to traverse the configured training-token budget once."""
    batch_size = max(1, int(cfg["batch_size"]))
    if bool(cfg.get("use_tbptt", False)) and cfg["dataset_type"] == 1:
        return max(1, math.ceil(line_stream.total_transitions / max(1, batch_size * line_stream.W)))
    if bool(cfg.get("use_tbptt", False)):
        stream_tokens = tbptt_stream.total_len or max(0, tbptt_stream.N - 1)
        return max(1, math.ceil(stream_tokens / max(1, batch_size * tbptt_stream.W)))
    if cfg["dataset_type"] == 1:
        example_count = len(dataset.offsets) if hasattr(dataset, "offsets") else len(dataset.data)
        return max(1, math.ceil(example_count / batch_size))
    if hasattr(dataset, "start_idx") and hasattr(dataset, "end_idx"):
        corpus_tokens = max(0, dataset.end_idx - dataset.start_idx - 1)
        return max(1, math.ceil(corpus_tokens / max(1, batch_size * cfg["seq_len"])))
    example_count = len(dataset.data) if hasattr(dataset, "data") else batch_size
    return max(1, math.ceil(example_count / batch_size))


def train_loop(cfg, model, optimizer, dataset, valid_ds, vocab, line_mode, progress_callback=None):
    """progress_callback(event_dict) receives step/valid/sample/checkpoint
    events (GUI monitor); returning True stops and saves like Ctrl+C."""
    def notify(event):
        if progress_callback is not None and progress_callback(event):
            raise KeyboardInterrupt
    update_steps = max(0, int(cfg.get("iterations_done", 0)))
    iters = update_steps * max(1, int(cfg.get("grad_accum_steps", 1)))
    train_tokens_done = max(0, int(cfg.get("train_tokens_done", 0)))
    interval_tokens = 0; last_log = time.time(); losses = []
    pad_id = None
    if hasattr(vocab, "pad_id"): pad_id = vocab.pad_id
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id if pad_id is not None else -100)
    is_bottom_up = is_bottom_up_megabyte(cfg)
    is_scan = cfg["model_selection"] in SCAN_MODEL_IDS and not is_bottom_up

    # ---- Advanced training features ----
    use_amp, amp_dtype, scaler = amp_settings(cfg)
    grad_accum_steps = max(1, int(cfg.get("grad_accum_steps", 1)))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    log_interval = int(cfg.get("log_interval", 50))
    sample_interval = int(cfg.get("sample_interval", 500))
    val_interval = int(cfg.get("val_interval", 500))
    save_interval = int(cfg.get("save_interval", 10000))
    loss_window = max(10, log_interval)
    
    # Early stopping
    use_early_stop = bool(cfg.get("early_stopping", False))
    patience = int(cfg.get("patience", 10))
    best_valid_loss = float('inf')
    patience_counter = 0
    
    if use_amp: print(f"\u26a1 Mixed Precision (AMP) enabled ({str(amp_dtype).replace('torch.', '')})")
    if grad_accum_steps > 1: print(f"\U0001f4e6 Gradient accumulation: {grad_accum_steps} steps (effective batch = {cfg['batch_size'] * grad_accum_steps})")

    use_tbptt = bool(cfg.get("use_tbptt", False))

    bptt_window = int(cfg.get("bptt_window", 0)) or max(1, cfg["seq_len"])

    tbptt_stream = None
    line_stream = None
    if use_tbptt and cfg["dataset_type"] == 0:
        # dataset.ids may be a memmap view; the stream converts per window
        ids_ref = dataset.ids if hasattr(dataset, "ids") else dataset.data
        tbptt_stream = TBPTTClassicStream(
            ids_ref, window=bptt_window,
            batch_size=cfg["batch_size"], total_len=cfg.get("tbptt_total_len", 0))
    elif use_tbptt and cfg["dataset_type"] == 1:
        line_stream = LineTBPTTStream(
            dataset=dataset, window=bptt_window,
            batch_size=cfg["batch_size"], bos_id=vocab.bos_id,
            pad_id=vocab.pad_id)

    # LR schedule over optimizer updates (the factor is recomputed from the
    # update count, so a resumed run continues where its schedule left off).
    lr_schedule = str(cfg.get("lr_scheduler", "none") or "none")
    warmup_steps = max(0, int(cfg.get("warmup_steps", 0) or 0))
    base_lrs = []
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
        base_lrs.append(group["initial_lr"])
    total_updates = 1
    if lr_schedule != "none":
        batches = training_steps_per_epoch(
            cfg, dataset, line_stream=line_stream if use_tbptt else None,
            tbptt_stream=tbptt_stream if use_tbptt else None,
        )
        total_updates = max(1, math.ceil(batches / grad_accum_steps) * int(cfg["epoch_count"]))
        print(f"LR schedule: {lr_schedule}" + (f", {warmup_steps} warmup steps" if lr_schedule == "cosine_warmup" else "")
              + f", over {total_updates:,} updates")
        if all(base == 0.0 for base in base_lrs):
            pwarn("The initial learning rate is 0, so the schedule has nothing to scale and is ignored.")
        if total_updates > 10**9:
            pwarn("The epoch count is effectively unlimited, so the decay is spread over "
                  f"{total_updates:,} updates and the LR stays near its base value.")

    def apply_lr_schedule():
        if lr_schedule == "none":
            return
        factor = bench_lr_factor(lr_schedule, update_steps / total_updates, update_steps, warmup_steps)
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * factor

    print(f"Training on {DEVICE} ... (Ctrl+C to save & exit)")
    if update_steps:
        print(f"[Resume] Continuing from optimizer step {update_steps:,}.")
    try:
        rnn_state = None
        optimizer.zero_grad(set_to_none=True)
        epoch_count = int(cfg["epoch_count"])
        starting_epoch = 0
        for epoch in range(epoch_count):
            model.train()
            
            steps_per_epoch = training_steps_per_epoch(
                cfg, dataset, line_stream=line_stream if use_tbptt else None,
                tbptt_stream=tbptt_stream if use_tbptt else None,
            )

            if epoch == 0:
                starting_epoch = update_steps // steps_per_epoch
            if epoch < starting_epoch:
                continue
            remaining_steps = steps_per_epoch
            if epoch == starting_epoch:
                remaining_steps -= update_steps % steps_per_epoch
                if remaining_steps == 0:
                    continue

            for batch_index in range(remaining_steps):
                apply_lr_schedule()
                # ===== Batching =====
                if use_tbptt:
                    if cfg["dataset_type"] == 0:
                        x, y, reset_mask = tbptt_stream.get_next(DEVICE)
                    else:
                        x, y, reset_mask = line_stream.get_next(DEVICE)
                else:
                    x, y = dataset.get_batch(cfg["batch_size"])
                    reset_mask = None

                # ===== Forward / Backward =====
                # Individual numerically sensitive model operations can opt
                # out with their own autocast(enabled=False) guards.
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    if use_tbptt and (is_scan or (
                        cfg["model_selection"] in RNN_MODEL_IDS and not is_bottom_up
                    )):
                        # Detach and selective reset across TBPTT windows
                        # MEGABYTE normally trains through its vectorized full
                        # hierarchy.  TBPTT is the explicit exception: seed its
                        # sampler cache once so subsequent windows can carry it.
                        if rnn_state is None and isinstance(model, MegaByteLM) and model.is_incremental:
                            rnn_state = model.init_incremental_cache(x)
                        rnn_state = detach_state(rnn_state)
                        if rnn_state is not None and reset_mask is not None:
                            rnn_state = reset_rnn_state(rnn_state, reset_mask, model, cfg["model_selection"])
                        logits, rnn_state = model(x, rnn_state)
                    else:
                        if is_scan:
                            # Parallel training path (stateless)
                            logits, _ = model(x)
                        elif cfg["model_selection"] in RNN_MODEL_IDS and not is_bottom_up:
                            logits, rnn_state = model(x, None)
                        else:
                            out = model(x)
                            logits = out[0] if isinstance(out, tuple) else out

                    loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                    aux_loss = getattr(model, "aux_loss", None)
                    if aux_loss is not None:
                        loss = loss + float(cfg.get("moe_aux_loss_weight", 0.01)) * aux_loss

                # A seq2seq TBPTT window can lie wholly inside the masked
                # sources.  Its forward pass has advanced the carried state;
                # with no targets the mean loss is NaN, so skip the update.
                if pad_id is not None and not bool((y != pad_id).any()):
                    continue

                # NaN/Inf guard — skip the step entirely to avoid poisoning optimizer state
                if torch.isnan(loss) or torch.isinf(loss):
                    if iters == 0:
                        print(f"[WARNING] NaN/Inf loss at first step — model may be numerically unstable")
                    else:
                        print(f"[WARNING] NaN/Inf loss at iter {iters+1} — skipping step")
                    optimizer.zero_grad(set_to_none=True)
                    iters += 1
                    continue

                # Scale each micro-batch so one optimizer update has the same
                # gradient magnitude as a single large batch.  The setting was
                # previously collected in the UI but never applied.
                scaled_loss = loss / grad_accum_steps
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                is_update = (
                    (iters + 1) % grad_accum_steps == 0
                    or batch_index + 1 == remaining_steps
                )
                if is_update:
                    if hasattr(optimizer, "observe_loss"):   # HD optimizers' divergence guard
                        optimizer.observe_loss(loss.item())
                    if scaler is not None:
                        # Clipping must see true gradient magnitudes, not the
                        # values multiplied by GradScaler's dynamic scale.
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    update_steps += 1

                valid_targets = int((y != (pad_id if pad_id is not None else -100)).sum().item())
                iters += 1; losses.append(loss.item())
                train_tokens_done += valid_targets
                cfg["train_tokens_done"] = train_tokens_done
                interval_tokens += valid_targets
                if is_update and progress_callback is not None:
                    notify({
                        "event": "step", "step": update_steps, "epoch": epoch + 1,
                        "epochs": epoch_count, "batch": batch_index + 1,
                        "batches": remaining_steps, "loss": losses[-1],
                        "grad_norm": float(grad_norm), "lr": optimizer.param_groups[0]['lr'],
                        "tokens": train_tokens_done, "time": time.time(),
                    })
                if is_update and update_steps % log_interval == 0:
                    current_loss = losses[-1]
                    avg_loss = sum(losses[-loss_window:]) / min(loss_window, len(losses))
                    elapsed = max(1e-6, time.time() - last_log)
                    tok_s = interval_tokens / elapsed
                    it_s = log_interval / elapsed
                    lr_now = optimizer.param_groups[0]['lr']
                    metrics = loss_metrics(avg_loss, 1, cfg)
                    print(
                        f"[e{epoch+1} step {readable_num(update_steps)}] current loss {current_loss:.4f} | "
                        f"loss {avg_loss:.4f} (smooth/{loss_window}) | {format_loss_metrics(metrics)} | "
                        f"tokens {train_tokens_done:,} | tok/s {int(tok_s)} | "
                        f"it/s {it_s:.2f} | s/it {1 / it_s:.3f} | lr {lr_now:.2e}"
                    )
                    last_log = time.time(); interval_tokens = 0

                # ----- Validation (keeps your existing heuristics) -----
                want_valid = (valid_ds is not None)
                if is_update and want_valid and update_steps % val_interval == 0:
                    with schedule_free_eval(optimizer):
                        vloss = eval_valid_loss(
                            model, cfg, valid_ds, vocab,
                            line_mode=(cfg["dataset_type"]==1),
                            max_samples=1000, return_metrics=True,
                        )
                    if vloss is not None:
                        if isinstance(vloss, dict):
                            append_validation_metrics(
                                cfg, vloss, step=update_steps, epoch=epoch + 1,
                                train_tokens=train_tokens_done,
                            )
                            print(f"[valid @ {readable_num(update_steps)}] loss {vloss['nll']:.4f} | {format_loss_metrics(vloss, include_accuracy=True)} | valid tokens {int(vloss['tokens']):,}")
                        else:
                            print(f"[valid @ {readable_num(update_steps)}] loss {vloss:.4f}")
                        notify({"event": "valid", "step": update_steps, "epoch": epoch + 1,
                                "metrics": vloss if isinstance(vloss, dict) else {"nll": float(vloss)}})

                if is_update and update_steps % sample_interval == 0:
                    with torch.no_grad(), schedule_free_eval(optimizer):
                        samples = do_training_sample(cfg, model, vocab, line_mode, iteration=update_steps,
                                                     examples_ds=valid_ds or dataset)
                    notify({"event": "samples", "step": update_steps, "samples": samples or []})
                if is_update and update_steps % save_interval == 0:
                    cfg["iterations_done"] = update_steps
                    with schedule_free_eval(optimizer):
                        torch.save(model.state_dict(), CHECKPOINT_PATH)
                    save_json(CONFIG_PATH, cfg)
                    print(f"\n[checkpoint] saved {CHECKPOINT_PATH} + {CONFIG_PATH}")
                    notify({"event": "checkpoint", "step": update_steps})

        cfg["iterations_done"] = update_steps
        with schedule_free_eval(optimizer):
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        save_json(CONFIG_PATH, cfg)
        print(f"\n[checkpoint] saved {CHECKPOINT_PATH} + {CONFIG_PATH}")
    except KeyboardInterrupt:
        print("\n[interrupt] saving...")
        cfg["iterations_done"] = update_steps
        with schedule_free_eval(optimizer):
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        save_json(CONFIG_PATH, cfg)


def do_training_sample(cfg, model, vocab, line_mode, iteration=None, examples_ds=None):
    """Print training previews; also returns them as dicts for the GUI."""
    was_training = model.training
    original_temperature = cfg.get("temperature", 1.0)
    samples = []
    try:
        model.eval()
        if cfg.get("seq2seq") and examples_ds is not None and hasattr(examples_ds, "get_encoded_example"):
            # Seq2seq previews: prompt with a real example's inputs and show
            # the generated outputs next to the expected ones.
            cfg["temperature"] = float(cfg.get("train_sample_temperature", original_temperature))
            for i in range(int(cfg.get("train_sample_count", 1))):
                ids, source_len = examples_ds.get_encoded_example(random.randrange(len(examples_ds.offsets)))
                prompt_ids = ids[:1 + source_len]
                out, stop_reason = generate_line_mode(
                    model, cfg, vocab, prompt_ids, limit_len=cfg["seq_len"], return_stop_reason=True,
                )
                print(f"[sample t={cfg['temperature']} #{i+1}]  ({stop_reason})")
                print(f"  input    : {seq2seq_visible(vocab.decode(prompt_ids[1:]))}")
                print(f"  expected : {seq2seq_visible(vocab.decode(ids[1 + source_len:-1]))}")
                print(f"  generated: {seq2seq_visible(vocab.decode(out[len(prompt_ids):]))}\n")
                samples.append({
                    "temperature": cfg["temperature"], "stop_reason": stop_reason,
                    "input": vocab.decode(prompt_ids[1:]),
                    "expected": vocab.decode(ids[1 + source_len:-1]),
                    "text": vocab.decode(out[len(prompt_ids):]),
                })
            return samples
        tmode = int(cfg.get("tokenizer_mode", 1))
        byte_text = bool(cfg.get("byte_output_text", False))
        out_dir = pathlib.Path("FileGen")
        if tmode in {-1, 0}:
            out_dir.mkdir(parents=True, exist_ok=True)

        # Retrieve new settings
        sample_len = int(cfg.get("train_sample_len", 20000))
        sample_count = int(cfg.get("train_sample_count", 1))
        custom_prompt_str = cfg.get("train_sample_prompt", "")

        # Training previews are diagnostics, not a three-temperature benchmark.
        # A single configured temperature makes each output independently
        # interpretable and avoids the low-temperature sample looking like a
        # mysterious, consistently truncated "third line".
        temps = [float(cfg.get("train_sample_temperature", original_temperature))]

        # --- Helper to determine prompt IDs ---
        def get_train_prompt_ids():
            # 1. Line mode (Always starts with BOS)
            if line_mode:
                return [vocab.bos_id]
            
            # 2. Corpus mode with Custom Prompt
            if custom_prompt_str:
                if tmode == 0: # Byte mode -> Parse HEX
                    try:
                        # Remove spaces/0x and convert to bytes
                        clean_hex = custom_prompt_str.replace(" ", "").replace("0x", "")
                        raw_bytes = binascii.unhexlify(clean_hex)
                        return vocab.encode(raw_bytes)
                    except Exception as e:
                        print(f"[Warn] Invalid Hex prompt '{custom_prompt_str}': {e}. Using random.")
                        # Fallthrough to random
                else: # Text mode
                    return vocab.encode(custom_prompt_str)

            # 3. Corpus mode Random (use active tokens for tiktoken)
            if isinstance(vocab, TiktokenVocab):
                return [vocab.get_random_token()]
            start_id = random.randrange(vocab.size)
            return [start_id]

        for temp in temps:
            cfg["temperature"] = temp
            
            for i in range(sample_count):
                prompt_ids = get_train_prompt_ids()
                
                # Visual Logging
                if not line_mode:
                    if tmode in {-1, 0}:
                        p_vis = bytes(vocab.to_bytes(prompt_ids)).hex()
                    else:
                        p_vis = vocab.decode(prompt_ids)
                    if len(p_vis) > 40: p_vis = p_vis[:40] + "..."
                    print(f"[sample t={temp} #{i+1}] Prompt: {p_vis}")
                else:
                    print(f"[sample t={temp} #{i+1}]")

                # Generate
                stop_reason = None
                if line_mode:
                    out, stop_reason = generate_line_mode(
                        model, cfg, vocab, prompt_ids, limit_len=cfg["seq_len"],
                        return_stop_reason=True,
                    )
                    print(f"  [line sample ended: {stop_reason}]")
                else:
                    out = generate_classic(
                        model, cfg, vocab, prompt_ids, max_len=sample_len, stream=False
                    )

                # Output / Save
                if tmode in {-1, 0}: # Binary / byte mode
                    data = vocab.to_bytes(out) if hasattr(vocab, "to_bytes") else bytes()
                    
                    sample = {"temperature": temp, "stop_reason": stop_reason,
                              "prompt": bytes(vocab.to_bytes(prompt_ids)).hex() if not line_mode else "",
                              "hex": data[:4096].hex(), "bytes": len(data)}
                    samples.append(sample)
                    if byte_text:
                        text_rep = vocab.decode(out[1:] if line_mode else out)
                        print(f"{text_rep}\n")
                        sample["text"] = text_rep
                    else:
                        # Training samples run between checkpoints, so the
                        # persisted config can be thousands of updates stale.
                        iter_num = cfg.get("iterations_done", 0) if iteration is None else iteration
                        fname = f"train_iter{iter_num}_t{temp}_{i+1}.bin"
                        (out_dir / fname).write_bytes(data)
                        print(f"   -> Saved {fname} ({len(data)} bytes)")
                        sample["file"] = str(out_dir / fname)
                else: # Text mode
                    disp_ids = out[1:] if (line_mode and len(out) > 0) else out
                    text = vocab.decode(disp_ids)
                    print(f"{text}\n")
                    samples.append({"temperature": temp, "stop_reason": stop_reason,
                                    "prompt": "" if line_mode else vocab.decode(prompt_ids),
                                    "text": text})
        return samples

    finally:
        cfg["temperature"] = original_temperature
        if was_training:
            model.train()




# ========= Config / UI =========
@dataclass
class RunConfig:
    dataset_path: str
    dataset_type: int
    model_selection: int
    activation_name: str
    embed_dim: int
    head_count: int
    layer_count: int
    seq_len: int
    epoch_count: int
    batch_size: int
    learning_rate: float
    model_type: int = 0  # 0=flat, 1=top-down MEGABYTE, 2=bottom-up MEGABYTE
    temperature: float = 1.0
    iterations_done: int = 0
    train_tokens_done: int = 0
    vocab_tokens: Optional[List[str]] = None
    line_max_len: Optional[int] = None
    tokenizer_mode: int = 1  # -1=binary, 0=byte, 1=char, 2=word
    use_norm: int = 0     # 0=None, 1=BatchNorm, 2=LayerNorm, 3=RMSNorm
    res_every: int = 0    # 0 disables; otherwise every n layers
    res_type: int = 0     # 0=add, 1=concat(+proj), 2=ReZero scalar, 3=ReZero elementwise
    dropout: float = 0.0  # inter-layer dropout prob
    sparse_local_window: int = 512
    sparse_compression_block: int = 32
    sparse_selected_blocks: int = 16
    use_multiplier: int = 0 #
    train_sample_len: int = 200     # Length of generated samples during training (corpus mode)
    train_sample_count: int = 1     # Number of samples per temperature
    train_sample_prompt: str = ""   # Custom prompt string (Hex if byte mode, Text otherwise)
    def to_dict(self): return asdict(self)


def read_dataset(path: str, dataset_type: int, tokenizer_mode: int = 1):
    """
    Returns list of items:
      - for char/word/binary: list[str]
      - for byte tokenizer (0): list[bytes]  (raw)
    """
    if tokenizer_mode in {-1, 0}:
        # byte mode → read raw bytes
        if dataset_type == 0:
            with open(path, "rb") as f:
                return [f.read()]
        else:
            # line mode: split by b'\n' but keep raw bytes per line
            with open(path, "rb") as f:
                return [ln.rstrip(b"\n") for ln in f.readlines()]
    else:
        # text modes
        if dataset_type == 0:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return [f.read()]
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return [ln.rstrip("\n") for ln in f.readlines()]


def prompt_int(msg, valid=None, default=None, minimum=None):
    label = prompt_label(msg, default)
    while True:
        s = input(label).strip()
        if s == "" and default is not None: return default
        try:
            v = int(s)
            if (valid is None or v in valid) and (minimum is None or v >= minimum): return v
        except: pass
        constraint = f" in {valid}" if valid else ""
        if minimum is not None:
            constraint += f" of at least {minimum}"
        print(f"  {_c(_RD, '✗')} Please enter a valid integer{constraint}.")

def prompt_float(msg, default=None):
    label = prompt_label(msg, default)
    while True:
        s = input(label).strip()
        if s == "" and default is not None: return default
        try: return float(s)
        except: print(f"  {_c(_RD, '✗')} Please enter a valid number.")

def prompt_str(msg, default=None):
    label = prompt_label(msg, default)
    s = input(label).strip()
    if s == "" and default is not None: return default
    return s


def prompt_activation(default="gelu", model_name=None):
    """Present the shared activation menu with architecture-aware defaults."""
    names = activation_names()
    if default not in names:
        default = "gelu"

    cli_section("Activation Function", 64)
    if model_name:
        print(f"  │  {_c(_DIM, f'{model_name} default: {default.upper()}')}")
    print(f"  │  {_c(_DIM, 'Gated options use a true GLU feed-forward block where supported.')}")
    print(f"  │")
    for index, name in enumerate(names):
        cli_opt(index, name.upper(), ACTIVATION_DESCRIPTIONS.get(name, "Architecture-specific activation"), kw=3, lw=14)
    print(f"  │")
    choice = prompt_int("Activation", valid=set(range(len(names))), default=names.index(default))
    cli_section_end(64)
    return names[choice]

def build_config_new():
    # ── Dataset ────────────────────────────────────────────────────────────────
    cli_section("Dataset", 64)
    print(f"  │")
    dataset_path = prompt_str("Dataset file path")

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Dataset type:')}")
    cli_opt(0, "Standard (corpus)", "Continuous text stream — random sliding windows")
    cli_opt(1, "Line mode",         "One example per line — BOS/EOS padded sequences")
    print(f"  │")
    dataset_type = prompt_int("Dataset type", valid={0,1})

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Tokenizer:')}")
    cli_opt(-1, "Binary",   "Raw bytes interpreted as binary (no text decoding)")
    cli_opt( 0, "Byte",     "Byte-level 0–255, universal, no vocab building")
    cli_opt( 1, "Char",     "Character-level — vocab built from dataset characters")
    cli_opt( 2, "Word",     "Whitespace-split word tokens — vocab from dataset")
    cli_opt( 3, "Tiktoken", "GPT-4 cl100k_base BPE tokenizer (50k+ vocab)")
    cli_opt( 4, "BPE",      "Custom byte-pair encoding trained on your data")
    print(f"  │")
    tokenizer_mode = prompt_int("Tokenizer", valid={-1,0,1,2,3,4})

    vocab_size_bpe = 0
    if tokenizer_mode == 3:
        tke = prompt_str("Tiktoken encoding  (gpt2 / r50k_base / cl100k_base)", default="cl100k_base")
    elif tokenizer_mode == 4:
        vocab_size_bpe = prompt_int("BPE vocabulary size", default=4096)
    elif tokenizer_mode in {-1, 0}:
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Binary / byte output mode:')}")
        cli_opt(0, "Binary → FileGen/", "Write raw binary files (images, audio, etc.)")
        cli_opt(1, "Print tokens",      "Print byte text or raw bit tokens to the terminal")
        print(f"  │")
        byte_out = prompt_int("Output mode", valid={0,1}, default=0)
    cli_section_end(64)

    seq2seq = prompt_seq2seq_config(dataset_path) if dataset_type == 1 else None
    if seq2seq:
        # Train on the rearranged copy; the original file is left untouched.
        dataset_path = prepare_seq2seq_dataset(seq2seq)

    # ── Model ──────────────────────────────────────────────────────────────────
    print_model_menu()
    print_megabyte_compatible_models()

    # The training UI accepts exactly the same registered IDs as benchmarks.
    msel = prompt_int("Model #", valid=MODEL_IDS)

    print(f"  │")
    cli_opt(0, "Normal", "Use the selected model as a flat language model")
    if msel in HIERARCHICAL_MODEL_MIXERS:
        if msel != MLP_MODEL_ID:
            cli_opt(1, "MEGABYTE", "Use the selected processor at every hierarchy stage")
        cli_opt(2, "MEGABYTE-Bottom up", "Encode fine→coarse, then decode coarse→fine")
        model_type = prompt_int("Model type", valid=({0, 2} if msel == MLP_MODEL_ID else {0, 1, 2}), default=0)
    else:
        print(f"  │  {_c(_DIM, 'MEGABYTE is unavailable for this model; no stage adapter is registered.')}")
        model_type = MODEL_TYPE_NORMAL

    # MinRNN / MinIndRNN activation sub-menus
    minrnn_act = 0
    if msel == 902:
        print()
        cli_section("MinRNN Activation", 64)
        cli_opt(0, "Tanh",       "Original MinRNN formulation")
        cli_opt(1, "ReLU",       "Unbounded, sparse activations")
        cli_opt(2, "SiLU",       "Smooth gated linear unit")
        cli_opt(3, "GELU",       "Gaussian error linear unit")
        cli_opt(4, "Sigmoid",    "Bounded 0–1 gate-like activation")
        cli_opt(5, "g_act",      "Log-space scan (minGRU-style, numerically stable)")
        cli_section_end(64)
        minrnn_act = prompt_int("Activation", valid={0,1,2,3,4,5})

    if msel == 903:
        print()
        cli_section("MinIndRNN Activation", 64)
        print(f"  │  {_c(_DIM, 'Applied to the input projection inside the parallel IndRNN scan.')}")
        print(f"  │")
        _indrnn_acts = [
            (0,  "Tanh",            "Saturating, symmetric — original MinRNN"),
            (1,  "ReLU",            "Unbounded, sparse — fast but can explode"),
            (2,  "SiLU / Swish",    "Smooth gating, non-monotone — often best"),
            (3,  "PReLU (α=0.0)",   "Leaky ReLU with learned slope initialised at 0"),
            (4,  "PReLU (default)", "Leaky ReLU with learned slope initialised at 0.25"),
            (5,  "LReLU 0.2",       "Fixed leaky slope 0.2 — robust negative-side gradient"),
            (6,  "LReLU 0.01",      "Fixed leaky slope 0.01 — near-ReLU"),
            (7,  "GELU",            "Gaussian-weighted linear unit — Transformer-style"),
            (8,  "BentIdentity",    "(√(x²+1)−1)/2 + x — smooth near-linear"),
            (9,  "Sine",            "sin(x) — periodic, good for positional signals"),
            (10, "Cosine",          "cos(x) — periodic variant"),
            (11, "Snake",           "x + sin²(x) — periodic + monotone blend"),
            (12, "Stepping Sine",   "Quantised sine — discrete periodic steps"),
            (13, "Stepping Cosine", "Quantised cosine — discrete periodic steps"),
            (14, "Mish",            "x·tanh(softplus(x)) — very smooth, self-regularising"),
            (15, "Cone",            "Triangular bump function — local receptive field"),
            (16, "ReLU²",          "max(0,x)² — sparse and strictly positive"),
            (17, "g_act",           "Log-space scan gate (minGRU-style) — numerically stable"),
        ]
        for idx, name, desc in _indrnn_acts:
            cli_opt(idx, name, desc, kw=3, lw=22)
        cli_section_end(64)
        minrnn_act = prompt_int("Activation", valid=set(range(18)))

    # Every non-recurrent architecture now exposes its feed-forward activation.
    # Recurrent models retain their architecture-defined nonlinearities.
    activation_name = "gelu"
    if msel in NON_RNN_ACTIVATION_IDS:
        activation_name = prompt_activation(
            MODEL_DEFAULT_ACTIVATIONS[msel],
            MODEL_NAMES.get(msel, f"Model {msel}"),
        )

    # ── Architecture ───────────────────────────────────────────────────────────
    cli_section("Architecture", 64)
    target_params = None
    megabyte_stage_dims = None
    megabyte_stage_depths = None
    megabyte_stage_heads = None
    megabyte_stage_seq_lens = None
    megabyte_stage_child_embed_dims = None
    megabyte_stage_mixers = None
    megabyte_bottom_up_encoder_stage_mixers = None
    megabyte_bottom_up_decoder_stage_mixers = None
    if model_type in {MODEL_TYPE_MEGABYTE, MODEL_TYPE_MEGABYTE_BOTTOM_UP}:
        hierarchy_label = "MEGABYTE-Bottom up" if model_type == MODEL_TYPE_MEGABYTE_BOTTOM_UP else "MEGABYTE"
        print(f"  │  {_c(_DIM, f'{hierarchy_label} stage lengths multiply')}")
        print(f"  │  {_c(_DIM, 'into the token window used for training; each stage has its own width,')}")
        print(f"  │  {_c(_DIM, 'depth, and (where applicable) attention-head count.')}")
        if model_type == MODEL_TYPE_MEGABYTE_BOTTOM_UP:
            print(f"  │  {_c(_DIM, 'It first composes immediate child representations fine→coarse, then decodes.')}")
        print(f"  │")
        stage_count = prompt_int("MEGABYTE stage count  (2 or more)", minimum=2, default=2)
        window_mlp_selected = model_type == MODEL_TYPE_MEGABYTE_BOTTOM_UP and msel == MLP_MODEL_ID
        if window_mlp_selected:
            print(f"  │  {_c(_DIM, 'Window MLP is valid only at the fine decoder stage, so choose decoder cores per stage.')}")
            per_stage_cores = 1
        else:
            cli_opt(0, "Same core", "Use the selected model at every hierarchy stage")
            cli_opt(1, "Per-stage cores", "Select a compatible model ID for each stage")
            core_label = "Decoder/shared stage core selection" if model_type == MODEL_TYPE_MEGABYTE_BOTTOM_UP else "Stage core selection"
            per_stage_cores = prompt_int(core_label, valid={0, 1}, default=0)
        if per_stage_cores:
            print_megabyte_compatible_models()
            megabyte_stage_mixers = [
                prompt_int(
                    f"Core model ID for stage {stage + 1}",
                    valid=(
                        HIERARCHICAL_MODEL_MIXERS
                        if model_type != MODEL_TYPE_MEGABYTE_BOTTOM_UP or stage == stage_count - 1
                        else set(HIERARCHICAL_MODEL_MIXERS) - {MLP_MODEL_ID}
                    ),
                )
                for stage in range(stage_count)
            ]
        else:
            megabyte_stage_mixers = [msel] * stage_count
        if model_type == MODEL_TYPE_MEGABYTE_BOTTOM_UP:
            print(f"  │")
            print(f"  │  {_c(_DIM, 'Encoder/decoder cores default to the same schedule. Separate schedules let')}")
            print(f"  │  {_c(_DIM, 'the fine encoder and decoder use different sequence processors.')}")
            print(f"  │  {_c(_DIM, f'Window MLP (model {MLP_MODEL_ID}) may be the fine decoder stage; it emits')}")
            print(f"  │  {_c(_DIM, 'one complete fine patch before the next patch begins.')}")
            cli_opt(0, "Shared schedule", "Use the decoder schedule for the encoder too")
            cli_opt(1, "Separate encoder", "Choose an independent encoder core schedule")
            separate_encoder = prompt_int("Encoder/decoder mixer schedule", valid={0, 1}, default=0)
            if separate_encoder:
                if window_mlp_selected:
                    print(f"  │  {_c(_DIM, 'An MLP encoder also needs the fine decoder MLP, so choose encoder cores per stage.')}")
                    per_stage_encoder = 1
                else:
                    cli_opt(0, "Same encoder core", "Use the selected model at every encoder stage")
                    cli_opt(1, "Per-stage encoder cores", "Select a compatible model ID for each encoder stage")
                    per_stage_encoder = prompt_int("Encoder core selection", valid={0, 1}, default=0)
                if per_stage_encoder:
                    print_megabyte_compatible_models()
                    megabyte_bottom_up_encoder_stage_mixers = [
                        prompt_int(
                            f"Encoder model ID for stage {stage + 1}",
                            valid=(
                                HIERARCHICAL_MODEL_MIXERS
                                if (
                                    stage == stage_count - 1
                                    and megabyte_stage_mixers[-1] == MLP_MODEL_ID
                                )
                                else set(HIERARCHICAL_MODEL_MIXERS) - {MLP_MODEL_ID}
                            ),
                        )
                        for stage in range(stage_count)
                    ]
                else:
                    megabyte_bottom_up_encoder_stage_mixers = [msel] * stage_count
                megabyte_bottom_up_decoder_stage_mixers = list(megabyte_stage_mixers)
        dims, child_embed_dims, depths, heads_per_stage, lengths = [], [], [], [], []
        for stage in range(stage_count):
            label = f"Stage {stage + 1} {'(coarse)' if stage == 0 else '(fine)' if stage == stage_count - 1 else ''}".rstrip()
            print(f"  │")
            print(f"  │  {_c(_WH, _B, label)}")
            lengths.append(prompt_int("  Sequence length / groups", default=128 if stage == 0 else 4))
            stage_dim = prompt_int("  Hidden dimension", default=256 if stage == 0 else 128)
            dims.append(stage_dim)
            child_embed_dims.append(prompt_int(
                f"  Child token embedding dimension  (1–{stage_dim})",
                valid=range(1, stage_dim + 1),
                default=min(64, stage_dim),
            ))
            depths.append(prompt_int("  Layer count", default=2))
            if megabyte_mixer_uses_heads(megabyte_stage_mixers[stage]):
                heads_per_stage.append(prompt_int("  Attention head count", default=8))
            else:
                heads_per_stage.append(1)
        megabyte_stage_dims = dims
        megabyte_stage_depths = depths
        megabyte_stage_heads = heads_per_stage
        megabyte_stage_seq_lens = lengths
        megabyte_stage_child_embed_dims = child_embed_dims
        embed_dim = dims[0]
        layer_count = sum(depths)
        head_count = heads_per_stage[0]
    else:
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Embedding / hidden dim — size of every vector in the model.')}")
        print(f"  │  {_c(_DIM, 'Larger = more capacity, more VRAM, slower training.')}")
        cli_opt(0, "Fixed width",      "Enter the hidden dimension yourself")
        cli_opt(1, "Match parameters", "Pick the width that gives a target parameter count")
        if prompt_int("Model size", valid={0, 1}, default=0) == 1:
            target_params = int(prompt_float("Target parameters  (millions)", default=2.0) * 1e6)
            embed_dim = 256   # replaced once the vocabulary and window are known
        else:
            embed_dim = prompt_int("Embedding / hidden dim")

        head_count = 4
        if msel in ATTN_MODEL_IDS:
            print(f"  │")
            print(f"  │  {_c(_DIM, 'Head count — splits embed_dim into parallel attention heads.')}")
            print(f"  │  {_c(_DIM, 'Must divide embed_dim evenly. More heads = finer-grained attention.')}")
            head_count = prompt_int("Attention head count", default=4)

        print(f"  │")
        print(f"  │  {_c(_DIM, 'Layer count — number of stacked blocks / cells.')}")
        print(f"  │  {_c(_DIM, 'Deeper models learn longer-range patterns at higher compute cost.')}")
        layer_count = prompt_int("Layer count")

    ngram_context = 4
    if msel == 2:
        print(f"  │")
        print(f"  │  {_c(_DIM, 'N-gram context — preceding tokens used for each next-token prediction.')}")
        print(f"  │  {_c(_DIM, 'Stored as a flat [batch, time, context] window, never nested tensor dimensions.')}")
        print(f"  │  {_c(_DIM, 'Maximum 32 prevents excessive flattened MLP inputs and memory use.')}")
        ngram_context = prompt_int("Previous tokens / characters  (1–32)", valid=set(range(1, 33)), default=4)

    sparse_local_window = 512
    sparse_compression_block = 32
    sparse_selected_blocks = 16

    # Built-in RNN extras, including any RNN stages selected for MEGABYTE.
    use_norm = 0; res_every = 0; res_type = 0; rnn_dropout = 0.0; use_multiplier = 0
    rnn_ffn = 0; rnn_cell_options = {}
    custom_rnn_ids = (502, 506, 507, 509, 510, 511, 513, 514, 515, 516, 518, 519, 601, 602)
    megabyte_builtin_rnn = (
        model_type in {MODEL_TYPE_MEGABYTE, MODEL_TYPE_MEGABYTE_BOTTOM_UP}
        and any(resolve_megabyte_stage_mixer(mixer) in {"rnn", "rnn_relu", "gru", "lstm"}
                for mixer in (
                    list(megabyte_stage_mixers)
                    + list(megabyte_bottom_up_encoder_stage_mixers or ())
                    + list(megabyte_bottom_up_decoder_stage_mixers or ())
                ))
    )
    if msel in (500, 504, 503, 501, 407) + custom_rnn_ids or megabyte_builtin_rnn:
        print(f"  │")
        print(f"  │  {_c(_DIM, 'RNN structure options — extra stabilisation for step-by-step RNNs.')}")
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Normalisation — applied after each recurrent cell output:')}")
        cli_opt(0, "None",      "No normalisation — fastest, can be unstable in deep nets")
        cli_opt(1, "BatchNorm", "Normalise over the batch dim — sensitive to small batch sizes")
        cli_opt(2, "LayerNorm", "Normalise over the feature dim — robust, recommended default")
        cli_opt(3, "RMSNorm",   "LayerNorm without mean-centering — cheaper, similar quality")
        cli_opt(4, "TTanh",     "Tanh-based trainable normaliser — experimental")
        cli_opt(5, "ETTanh",    "Extended TTanh with elementwise scale — experimental")
        cli_opt(6, "DyT",       "Dynamic Tanh — replaces LayerNorm, no mean/var statistics")
        print(f"  │")
        use_norm = prompt_int("Norm type", valid={0,1,2,3,4,5,6}, default=0)
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Residual connections — skip connections between layers that help')}")
        print(f"  │  {_c(_DIM, 'gradients flow and allow training much deeper RNN stacks.')}")
        res_every = prompt_int("Residual every N layers  (0 = off)", default=0)
        if res_every > 0:
            print(f"  │")
            print(f"  │  {_c(_DIM, 'Residual type:')}")
            cli_opt(0, "Add",            "x = x + f(x)  — standard, zero overhead")
            cli_opt(1, "Concat + proj",  "x = proj([x, f(x)])  — richer but adds parameters")
            cli_opt(2, "ReZero scalar",  "x = x + α·f(x), α=0 init — very stable training start")
            cli_opt(3, "ReZero vector",  "x = x + α⊙f(x), per-dim α — most expressive ReZero")
            print(f"  │")
            res_type = prompt_int("Residual type", valid={0,1,2,3}, default=0)
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Dropout — randomly zeros inter-layer activations during training,')}")
        print(f"  │  {_c(_DIM, 'acting as regularisation. 0.1–0.3 for most tasks; 0 to disable.')}")
        rnn_dropout = prompt_float("Inter-layer dropout  (0.0 = off)", default=0.0)
        if not megabyte_builtin_rnn:
            print(f"  │")
            print(f"  │  {_c(_DIM, 'Output multiplier — a learnable gate on the final hidden→logit')}")
            print(f"  │  {_c(_DIM, 'projection. Can help calibrate output scale, especially early.')}")
            cli_opt(0, "Off",           "No multiplier — standard behaviour")
            cli_opt(1, "Scalar",        "One learnable scalar multiplies all outputs")
            cli_opt(2, "Per-dim vector","One learnable value per hidden dimension")
            print(f"  │")
            use_multiplier = prompt_int("Multiplier", default=0)
        if msel in custom_rnn_ids and model_type == MODEL_TYPE_NORMAL:
            print(f"  │")
            print(f"  │  {_c(_DIM, 'Post-layer feed-forward block after every recurrent layer:')}")
            cli_opt(0, "Off",    "Recurrent layers only")
            cli_opt(1, "SwiGLU", "Pre-norm SwiGLU FFN with residual")
            cli_opt(2, "ReGLU",  "Pre-norm ReGLU FFN with residual")
            cli_opt(3, "SiLU",   "Pre-norm SiLU MLP with residual (experimental)")
            rnn_ffn = prompt_int("Post-layer FFN", valid={0, 1, 2, 3}, default=0)
        if msel == 509:
            print(f"  │  {_c(_DIM, 'IndRNN activation: ReLU (paper-style, u in [0,1], |u|<=1) or tanh (legacy).')}")
            act = prompt_str("IndRNN activation  (relu/tanh)", default="relu").strip().lower()
            rnn_cell_options["indrnn_activation"] = "tanh" if act.startswith("t") else "relu"
        if msel in (513, 514):
            yn = prompt_str("Experimental ReLU gates instead of sigmoid  (y/n)", default="n").lower()
            rnn_cell_options["relu_gates"] = yn in ("y", "yes", "1")
        if msel in (515, 516):
            rnn_cell_options["mogrifier_rounds"] = prompt_int("Mogrifier rounds  (0 = plain LSTM/GRU)", default=5)
        if msel == 602:
            rnn_cell_options["unicornn_dt"] = prompt_float("UnICORNN time step dt", default=0.1)
            rnn_cell_options["unicornn_alpha"] = prompt_float("UnICORNN alpha (oscillator frequency)", default=10.0)
        if msel == 519:
            yn = prompt_str("LRU highway stacking (h~ = layer input for layers >= 2)  (y/n)", default="n").lower()
            rnn_cell_options["lru_highway"] = yn in ("y", "yes", "1")
        if msel == 518:
            rnn_cell_options["rru_middle_multiplier"] = prompt_float("RRU middle-layer size multiplier", default=2.0)
            rnn_cell_options["rru_dropout"] = prompt_float("RRU in-cell dropout", default=0.0)
    cli_section_end(64)

    # ── Sequence / Data ────────────────────────────────────────────────────────
    cli_section("Sequence & Validation", 64)
    print(f"  │")
    if dataset_type == 0:
        print(f"  │  {_c(_DIM, 'Sequence length — tokens per training window.')}")
        if model_type in {MODEL_TYPE_MEGABYTE, MODEL_TYPE_MEGABYTE_BOTTOM_UP}:
            seq_len = math.prod(megabyte_stage_seq_lens)
            print(f"  │  {_c(_DIM, f'MEGABYTE stage lengths set this automatically to {seq_len} tokens.')}")
        else:
            print(f"  │  {_c(_DIM, 'Longer = more context, more memory. Typical: 128–2048.')}")
            seq_len = prompt_int("Sequence length  (tokens per window)")
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Validation file — a separate held-out text file for measuring')}")
        print(f"  │  {_c(_DIM, 'generalisation. Leave blank to auto-split the training file.')}")
        classic_val_path = prompt_str("Validation file  (leave blank to split)", default="")
        if not classic_val_path:
            print(f"  │")
            print(f"  │  {_c(_DIM, 'Validation split — fraction of the file reserved for validation.')}")
            print(f"  │  {_c(_DIM, 'e.g. 0.1 = last 10% of the file. Set 0 to disable validation.')}")
            val_split = prompt_float("Validation split fraction  (0 = disable)", default=0.0)
        else:
            val_split = 0.0
    else:
        seq_len = 0; classic_val_path = ""
        print(f"  │  {_c(_DIM, 'Sequence length is set automatically from the longest line.')}")
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Validation split — fraction of lines held out for evaluation.')}")
        print(f"  │  {_c(_DIM, 'Lines are shuffled before splitting. Set 0 to use all for training.')}")
        val_split = prompt_float("Validation split fraction  (0 = disable)", default=0.0)
    cli_section_end(64)

    if msel == 409:
        sparse_default = min(512, max(1, seq_len))
        cli_section("Sparse Modern Transformer", 64)
        print(f"  │  {_c(_DIM, 'Local attention preserves recent detail; completed blocks provide')}")
        print(f"  │  {_c(_DIM, 'compressed global routing and selected raw-token retrieval.')}")
        sparse_local_window = prompt_int("Local causal window", default=sparse_default)
        sparse_compression_block = prompt_int("Compression block size", default=32)
        sparse_selected_blocks = prompt_int("Selected completed blocks", default=16)
        if min(sparse_local_window, sparse_compression_block, sparse_selected_blocks) < 1:
            raise ValueError("Sparse Modern Transformer settings must be positive")
        cli_section_end(64)

    # ── Training ───────────────────────────────────────────────────────────────
    cli_section("Training", 64)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Epoch count — full passes through the dataset.')}")
    print(f"  │  {_c(_DIM, 'One epoch = ceil(dataset_size / batch_size) gradient steps.')}")
    epoch_count   = prompt_int("Epoch count", default=5151515151515151)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Batch size — sequences processed per gradient step.')}")
    print(f"  │  {_c(_DIM, 'Larger batches = more stable gradients but more VRAM.')}")
    batch_size    = prompt_int("Batch size")
    cli_section_end(64)

    # ── Optimizer ──────────────────────────────────────────────────────────────
    optim_cfg = prompt_optimizer_config()
    learning_rate = optim_cfg["optim_params"].get("lr", 1.0)

    # ── Sampling ──────────────────────────────────────────────────────────────
    cli_section("Sampling", 64)

    print(f"  │")
    print(f"  │  {_c(_DIM, 'During-training sampling — periodically generates text so you can')}")
    print(f"  │  {_c(_DIM, 'watch the model improve without waiting for training to finish.')}")
    train_sample_count = prompt_int("Samples per training-sample call", default=2)
    train_sample_len   = 0
    train_sample_prompt = ""
    if dataset_type == 0:
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Sample length — how many tokens to generate each time.')}")
        train_sample_len = prompt_int("Sample length  (tokens)", default=200)
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Training prompt — text (or hex for byte mode) to seed each sample.')}")
        print(f"  │  {_c(_DIM, 'Leave blank to pick a random token from the vocab each time.')}")
        if tokenizer_mode == 0:
            train_sample_prompt = prompt_str("Training prompt  (hex, blank = random)", default="")
        else:
            train_sample_prompt = prompt_str("Training prompt  (text, blank = random)", default="")
    cli_section_end(64)

    # ── Advanced ───────────────────────────────────────────────────────────────
    cli_section("Advanced Training", 64)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Gradient accumulation — accumulates gradients over N mini-batches')}")
    print(f"  │  {_c(_DIM, 'before a weight update. Simulates a larger effective batch size')}")
    print(f"  │  {_c(_DIM, 'without extra VRAM. e.g. accum=4, batch=32 → effective batch 128.')}")
    grad_accum    = prompt_int("Gradient accumulation steps  (1 = off)", default=1)
    use_amp_val   = 0
    amp_dtype_val = "fp16"
    if DEVICE == "cuda":
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Automatic Mixed Precision (AMP) — runs forward pass in fp16 or bf16,')}")
        print(f"  │  {_c(_DIM, 'keeping master weights in float32. Halves VRAM and speeds up')}")
        print(f"  │  {_c(_DIM, 'matmuls on Ampere+ GPUs (RTX 30xx / A100 and newer).')}")
        use_amp_val = prompt_int("Mixed precision / AMP  (0=off 1=on)", valid={0,1}, default=0)
        if use_amp_val:
            amp_dtype_val = prompt_amp_dtype()

    print(f"  │")
    print(f"  │  {_c(_DIM, 'LR scheduler — adjusts learning rate over the course of training.')}")
    print(f"  │  {_c(_DIM, 'Has less impact with Prodigy (which self-tunes) but can still help.')}")
    cli_opt(0, "None",           "Fixed learning rate — Prodigy handles it already")
    cli_opt(1, "Cosine + warmup","Linear ramp for N steps, then cosine decay to 0")
    cli_opt(2, "Cosine",         "Cosine decay from step 0 — no warmup")
    cli_opt(3, "One-cycle",      "Ramps up then aggressively down — fast convergence")
    print(f"  │")
    sched_choice  = prompt_int("Scheduler", valid={0,1,2,3}, default=0)
    sched_map     = {0:"none", 1:"cosine_warmup", 2:"cosine", 3:"one_cycle"}
    lr_scheduler  = sched_map[sched_choice]
    warmup_steps  = 0
    if lr_scheduler == "cosine_warmup":
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Warmup steps — LR grows linearly from 0 to base LR over this many')}")
        print(f"  │  {_c(_DIM, 'steps. Helps stabilise early training. Typical: 100–500.')}")
        warmup_steps = prompt_int("Warmup steps", default=100)

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Early stopping — halts training if validation loss stops improving,')}")
    print(f"  │  {_c(_DIM, 'preventing overfitting. Requires a validation split or file.')}")
    early_stop    = prompt_int("Early stopping  (0=off 1=on)", valid={0,1}, default=0)
    patience_val  = 10
    if early_stop:
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Patience — number of validation checks without improvement before')}")
        print(f"  │  {_c(_DIM, 'training stops. Higher = more tolerance for temporary plateaus.')}")
        patience_val = prompt_int("Patience  (checks without improvement)", default=10)

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Logging / sampling / validation intervals — how often (in optimizer-update steps)')}")
    print(f"  │  {_c(_DIM, 'each event fires. Lower = more feedback, slightly more overhead.')}")
    log_interval        = prompt_int("Log loss every N update steps", default=10)
    sample_interval_val = prompt_int("Sample text every N update steps", default=500)
    val_interval_val    = prompt_int("Run validation every N update steps", default=500)
    use_compile_val     = prompt_int("Enable torch.compile  (0=off 1=on)", valid={0,1}, default=0)

    # TBPTT (recurrent / scan only)
    use_tbptt = False; bptt_window = 0; tbptt_total_len = 0
    if not is_bottom_up_megabyte({"model_type": model_type}) and (msel in RNN_MODEL_IDS or msel in SCAN_MODEL_IDS):
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Truncated BPTT (TBPTT) — instead of fitting one sequence per step,')}")
        print(f"  │  {_c(_DIM, 'the model processes a long stream and back-props through a short')}")
        print(f"  │  {_c(_DIM, 'window while carrying the hidden state forward. This lets RNNs')}")
        print(f"  │  {_c(_DIM, 'learn very long-range dependencies without exploding memory.')}")
        yn = prompt_str("Enable TBPTT  (y/n)", default="n").lower()
        use_tbptt = yn in ("y","yes","1")
        if use_tbptt:
            print(f"  │")
            print(f"  │  {_c(_DIM, 'BPTT window — tokens processed per gradient step.')}")
            print(f"  │  {_c(_DIM, 'Shorter = faster steps, shallower gradient signal.')}")
            if dataset_type == 1:
                bptt_window = prompt_int("BPTT window  (tokens per step, ≤ max line length)", default=64)
            else:
                bptt_window    = prompt_int("BPTT window  (tokens per step)", default=64)
                print(f"  │")
                print(f"  │  {_c(_DIM, 'Total TBPTT length — total tokens streamed before resetting state.')}")
                print(f"  │  {_c(_DIM, 'Set 0 to stream until EOF then wrap around.')}")
                tbptt_total_len = prompt_int("Total TBPTT length  (0 = until EOF)", default=0)
    cli_section_end(64)

    # ── Assemble config ────────────────────────────────────────────────────────
    cfg = RunConfig(
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        model_selection=msel,
        activation_name=activation_name,
        embed_dim=embed_dim,
        head_count=head_count,
        layer_count=layer_count,
        seq_len=seq_len,
        epoch_count=epoch_count,
        batch_size=batch_size,
        learning_rate=learning_rate,
        model_type=model_type,
        tokenizer_mode=tokenizer_mode,
        use_norm=use_norm,
        res_every=res_every,
        res_type=res_type,
        dropout=rnn_dropout,
        use_multiplier=use_multiplier,
        train_sample_len=train_sample_len,
        train_sample_count=train_sample_count,
        train_sample_prompt=train_sample_prompt,
    ).to_dict()

    cfg["model_id_scheme"]   = MODEL_ID_SCHEME
    cfg["line_seq_len_cap"]  = None   # line mode: window = longest line
    if seq2seq: cfg["seq2seq"] = seq2seq
    if classic_val_path: cfg["classic_val_path"] = classic_val_path
    cfg["val_split"]         = val_split
    cfg["use_tbptt"]         = use_tbptt
    cfg["bptt_window"]       = bptt_window
    cfg["tbptt_total_len"]   = tbptt_total_len
    cfg["minrnn_act"]        = minrnn_act
    cfg["rnn_ffn"]           = rnn_ffn
    cfg["rnn_cell_options"]  = rnn_cell_options
    cfg["grad_accum_steps"]  = grad_accum
    cfg["use_amp"]           = bool(use_amp_val)
    cfg["amp_dtype"]         = amp_dtype_val
    cfg["lr_scheduler"]      = lr_scheduler
    cfg["warmup_steps"]      = warmup_steps
    cfg["early_stopping"]    = bool(early_stop)
    cfg["patience"]          = patience_val
    cfg["log_interval"]      = log_interval
    cfg["sample_interval"]   = sample_interval_val
    cfg["val_interval"]      = val_interval_val
    cfg["use_compile"]       = bool(use_compile_val)
    cfg["ngram_context"]      = ngram_context
    if target_params:
        cfg["target_params"]  = target_params
    cfg["sparse_local_window"] = sparse_local_window
    cfg["sparse_compression_block"] = sparse_compression_block
    cfg["sparse_selected_blocks"] = sparse_selected_blocks
    if megabyte_stage_dims is not None:
        cfg["megabyte_stage_dims"] = megabyte_stage_dims
        cfg["megabyte_stage_depths"] = megabyte_stage_depths
        cfg["megabyte_stage_heads"] = megabyte_stage_heads
        cfg["megabyte_stage_seq_lens"] = megabyte_stage_seq_lens
        cfg["megabyte_stage_child_embed_dims"] = megabyte_stage_child_embed_dims
        cfg["megabyte_stage_mixers"] = megabyte_stage_mixers
        if megabyte_bottom_up_encoder_stage_mixers is not None:
            cfg["megabyte_bottom_up_encoder_stage_mixers"] = megabyte_bottom_up_encoder_stage_mixers
            cfg["megabyte_bottom_up_decoder_stage_mixers"] = megabyte_bottom_up_decoder_stage_mixers
        if megabyte_builtin_rnn:
            cfg["megabyte_fused_rnn_version"] = 2
            cfg["megabyte_fused_rnn_norm_type"] = use_norm
            cfg["megabyte_fused_rnn_res_every"] = res_every
            cfg["megabyte_fused_rnn_res_type"] = res_type
            cfg["megabyte_fused_rnn_dropout"] = rnn_dropout
        if model_type == MODEL_TYPE_MEGABYTE_BOTTOM_UP:
            cfg["megabyte_bottom_up_version"] = 6

    cfg["optimizer"]         = optim_cfg["optimizer"]
    cfg["optim_params"]      = optim_cfg["optim_params"]

    if tokenizer_mode == 3: cfg["tiktoken_encoding"] = tke
    if tokenizer_mode in {-1, 0}: cfg["byte_output_text"] = bool(byte_out == 1)
    if tokenizer_mode == 4: cfg["custom_bpe_size"]   = vocab_size_bpe
    return cfg


def validation_token_capacity(ds, line_mode: bool) -> Optional[int]:
    """Return the finite held-out target-token budget when the dataset exposes one."""
    if hasattr(ds, "start_idx") and hasattr(ds, "end_idx"):
        return max(0, int(ds.end_idx) - int(ds.start_idx) - 1)
    if line_mode and hasattr(ds, "offsets"):
        # Indexed line validation scores distinct lines without replacement,
        # so it cannot exceed the held-out set and needs no cap.
        return None
    if line_mode and hasattr(ds, "data"):
        pad_id = getattr(ds.vocab, "pad_id", None)
        return int((ds.data[:, 1:] != pad_id).sum().item()) if pad_id is not None else int(ds.data[:, 1:].numel())
    return None


@torch.no_grad()
def eval_valid_loss(model, cfg, ds, vocab, line_mode, max_samples=1000, seed=None, return_metrics=False):
    """Return mean token cross-entropy over a bounded dataset sample.

    ``reduction='sum'`` plus an explicit token count prevents short, padded
    line batches from receiving the same weight as full batches.  Supplying a
    seed makes repeated validation checks use the same random held-out sample,
    which is essential when selecting the best checkpoint by validation loss.
    """
    is_scan = cfg["model_selection"] in SCAN_MODEL_IDS

    pad_id = getattr(vocab, "pad_id", -100)
    if pad_id is None: pad_id = -100
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    token_capacity = validation_token_capacity(ds, line_mode)
    orig_training = model.training

    def forward_for_validation(x):
        # Incremental MEGABYTE models use their token-by-token sampler in
        # eval mode.  Validation scores independent full windows, so preserve
        # eval semantics while using the vectorized hierarchy used in training.
        if isinstance(model, MegaByteLM) and model.is_incremental:
            return model._forward_full(x)
        if is_scan or cfg["model_selection"] in RNN_MODEL_IDS:
            out = model(x)
            return out[0] if isinstance(out, tuple) else out
        return model(x)

    def accumulate(logits, targets):
        """Add only the remaining held-out tokens, never exceeding file capacity."""
        nonlocal total_loss, total_tokens, total_correct
        target_mask = targets != pad_id
        if token_capacity is not None:
            remaining = token_capacity - total_tokens
            if remaining <= 0:
                return False
            if int(target_mask.sum().item()) > remaining:
                keep = target_mask.flatten().nonzero().flatten()[:remaining]
                target_mask = torch.zeros_like(target_mask, dtype=torch.bool)
                target_mask.flatten()[keep] = True
        masked_targets = targets.masked_fill(~target_mask, pad_id)
        loss = criterion(logits.reshape(-1, logits.size(-1)), masked_targets.reshape(-1))
        total_loss += loss.item()
        total_tokens += int(target_mask.sum().item())
        total_correct += int(((logits.argmax(dim=-1) == targets) & target_mask).sum().item())
        return token_capacity is None or total_tokens < token_capacity

    # Dataset batchers use Python, NumPy, and Torch RNGs.  Preserve their state
    # so validation neither changes training batches nor varies across checks.
    rng_state = None
    if seed is not None:
        rng_state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    try:
        # Scan models: keep in train() mode to use the fast parallel scan path.
        # torch.no_grad() (from decorator) already disables autograd.
        # Switching to eval() forces some scan models into slow step-by-step mode.
        if is_scan:
            model.train()  # Keep parallel scan path active
        else:
            model.eval() 

        # 1. New IndexedLineDataset (Disk-based)
        if line_mode and hasattr(ds, 'offsets'):
            # Score up to max_samples distinct lines, each exactly once.  No
            # line can be counted twice however small the held-out set, so no
            # token cap (and no pass encoding the whole file) is needed.
            # Length-sorted batches keep padding, and so memory, near the
            # size of the lines actually in each batch.
            batch_size = max(1, min(32, cfg["batch_size"]))
            picked = random.sample(range(len(ds.offsets)), min(len(ds.offsets), max_samples))
            examples = sorted((ds.get_encoded_example(i) for i in picked), key=lambda e: len(e[0]))
            for start in range(0, len(examples), batch_size):
                x, y = pad_line_examples(examples[start:start + batch_size], pad_id)
                logits = forward_for_validation(x)
                accumulate(logits, y)

        # 2. Old LineDataset (Memory-based)
        elif line_mode and hasattr(ds, 'data'):
            n = ds.data.size(0)
            if n == 0: return None
            sample_count = min(n, max_samples)
            idx = torch.randperm(n)[:sample_count]
            BATCH = min(64, sample_count)
            for i in range(0, sample_count, BATCH):
                x = ds.data[idx][i:i+BATCH,:-1].to(DEVICE)
                y = ds.data[idx][i:i+BATCH,1:].to(DEVICE)
                
                logits = forward_for_validation(x)
                    
                if not accumulate(logits, y):
                    break
        
        # 3. Classic mode (Memmap or Standard)
        else:
            bs = min(64, cfg["batch_size"])
            steps = max(1, math.ceil(max_samples / bs))
            for _ in range(steps):
                x, y = ds.get_batch(bs)
                logits = forward_for_validation(x)
                
                if not accumulate(logits, y):
                    break

    finally:
        # Restore the caller's mode exactly.  In particular, validation of a
        # scan model temporarily switches to train mode for its parallel path.
        model.train(orig_training)
        if rng_state is not None:
            random.setstate(rng_state[0])
            np.random.set_state(rng_state[1])
            torch.set_rng_state(rng_state[2])
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

    if total_tokens == 0:
        return None
    metrics = loss_metrics(total_loss, total_tokens, cfg, total_correct)
    metrics["tokens"] = total_tokens
    return metrics if return_metrics else metrics["nll"]


def _restore_token_vocab(tokens, line_mode: bool, tokenizer_mode: int) -> BaseVocab:
    tokens = list(tokens)
    if tokenizer_mode == 2:
        v = WordVocab.__new__(WordVocab)
    else:
        v = CharVocab.__new__(CharVocab)

    if line_mode:
        body = [t for t in tokens if t not in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN)]
        tokens = [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN] + body

    v.line_mode = line_mode
    v.tokens = tokens
    v.stoi = {tok: i for i, tok in enumerate(v.tokens)}
    v.itos = {i: tok for tok, i in v.stoi.items()}
    v.bos_id = v.stoi.get(BOS_TOKEN) if line_mode else None
    v.eos_id = v.stoi.get(EOS_TOKEN) if line_mode else None
    v.pad_id = v.stoi.get(PAD_TOKEN) if line_mode else None
    return v


def load_or_make_vocab(cfg, txt_path: str, save_config: bool = True) -> BaseVocab:
    """Build or restore the vocabulary.  ``save_config=False`` is for callers
    whose ``cfg`` is a throwaway stub, not the run config in CONFIG_PATH."""
    if txt_path == cfg.get("dataset_path"):
        ensure_seq2seq_dataset(cfg)
    line_mode = (cfg["dataset_type"] == 1)
    tmode = int(cfg.get("tokenizer_mode", 1))

    # 1. Non-text tokenizers (Binary, Byte, Tiktoken) don't need saving/loading
    if tmode == -1: return BinaryVocab(line_mode)
    if tmode == 0: return ByteVocab(line_mode)
    if tmode == 3:
        v = TiktokenVocab(cfg.get("tiktoken_encoding", "cl100k_base"), line_mode)
        # Scan file for active tokens (for filtered random prompts)
        if os.path.exists(txt_path):
            v.scan_file(txt_path)
        return v
    if tmode == 4:
        # Save .bpe.vocab next to the text file
        vocab_file = str(pathlib.Path(txt_path).with_suffix(".bpe.vocab"))
        target_size = int(cfg.get("custom_bpe_size", 4096))
        
        bpe = CustomBPEVocab(vocab_file, line_mode, expected_size=target_size)
        
        # Train if empty (meaning file didn't exist)
        if len(bpe.merges) == 0:
            bpe.train(txt_path, target_size)
            
        return bpe

    # 2. Check if vocab is already in the config (this prevents recreation)
    if cfg.get("vocab_tokens") and len(cfg["vocab_tokens"]) > 0:
        print(f"[Vocab] Loading {len(cfg['vocab_tokens'])} tokens from config...")
        return _restore_token_vocab(cfg["vocab_tokens"], line_mode, tmode)

    # 3. If not found, scan the file to build it
    print(f"[Vocab] Scanning {txt_path} to build vocabulary...")
    unique_tokens = set()
    
    # Use the streaming read to avoid memory issues
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="Building Vocab"):
            if tmode == 1: unique_tokens.update(line)
            elif tmode == 2: unique_tokens.update(line.split())
    
    vocab_list = sorted(list(unique_tokens))
    print(f"[Vocab] Found {len(vocab_list)} unique tokens.")
    
    if tmode == 2:
        v = WordVocab(vocab_list, line_mode)
    else:
        v = CharVocab(vocab_list, line_mode)
    
    # 4. Save immediately to config
    cfg["vocab_tokens"] = v.tokens
    if save_config:
        save_json(CONFIG_PATH, cfg)
        print(f"[Vocab] Saved tokens to {CONFIG_PATH}")
    
    return v


def build_datasets(cfg, vocab):
    path = cfg["dataset_path"]
    dtype = cfg["dataset_type"]
    val_split = float(cfg.get("val_split", 0.0))
    
    if dtype == 0:
        # === Corpus Mode ===
        # If external validation file exists, use it and ignore split
        val_path = cfg.get("classic_val_path")
        if val_path and os.path.exists(val_path):
            print(f"[Dataset] Using explicit validation file: {val_path}")
            train_ds = MemmapClassicDataset(path, vocab, cfg["seq_len"], split_range=(0.0, 1.0))
            valid_ds = MemmapClassicDataset(val_path, vocab, cfg["seq_len"], split_range=(0.0, 1.0))
        elif val_split > 0.0:
            print(f"[Dataset] Splitting single file: {1.0-val_split:.0%} Train / {val_split:.0%} Valid")
            # Train gets 0.0 -> (1.0 - split)
            # Valid gets (1.0 - split) -> 1.0
            split_pt = 1.0 - val_split
            train_ds = MemmapClassicDataset(path, vocab, cfg["seq_len"], split_range=(0.0, split_pt))
            valid_ds = MemmapClassicDataset(path, vocab, cfg["seq_len"], split_range=(split_pt, 1.0))
        else:
            print("[Dataset] No validation split.")
            train_ds = MemmapClassicDataset(path, vocab, cfg["seq_len"], split_range=(0.0, 1.0))
            valid_ds = None
            
        return train_ds, valid_ds
    
    else:
        # === Line Mode ===
        spec = cfg.get("seq2seq")
        full_ds = IndexedLineDataset(path, vocab, seq2seq_inputs=len(spec["input_cols"]) if spec else 0)

        # Use existing indices/logic but parameterize the split
        n = len(full_ds.offsets)
        if val_split > 0.0:
            perm = np.random.permutation(n)
            cut = int((1.0 - val_split) * n)
            train_idx = perm[:cut]
            valid_idx = perm[cut:]
            
            print(f"[Dataset] Lines Split: {len(train_idx)} Train / {len(valid_idx)} Valid")
            train_ds = IndexedLineDatasetSubset(full_ds, train_idx)
            valid_ds = IndexedLineDatasetSubset(full_ds, valid_idx) if len(valid_idx) > 0 else None
            cfg["valid_examples"] = len(valid_idx)
        else:
            print(f"[Dataset] Using all {n} lines for training (no validation).")
            # Create a subset containing all indices to keep types consistent
            all_idx = np.arange(n)
            train_ds = IndexedLineDatasetSubset(full_ds, all_idx)
            valid_ds = None
            cfg["valid_examples"] = 0
        
        cfg["line_max_len"] = full_ds.max_len
        # The window is the longest line's byte length, which bounds its token
        # count for every tokenizer except Binary (8 tokens per byte).  Configs saved before the cap was
        # removed have no "line_seq_len_cap" key and keep the old 2048 limit,
        # because their checkpoints' position-dependent weights were sized to it.
        cap = cfg.get("line_seq_len_cap", 2048)
        cfg["seq_len"] = full_ds.max_len if cap is None else min(full_ds.max_len, cap)
        
        return train_ds, valid_ds



def resume_adjustments(cfg, _model):
    cli_section("Resume Adjustments", 64)
    print(f"  │  {_c(_DIM, 'Update training settings before continuing from the checkpoint.')}")
    print(f"  │  {_c(_DIM, 'Architecture and vocabulary cannot be changed on resume.')}")

    if cfg["model_selection"] in RNN_MODEL_IDS and not is_bottom_up_megabyte(cfg):
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Sequence length — RNNs allow changing this on resume because')}")
        print(f"  │  {_c(_DIM, 'the weights do not depend on a fixed context window size.')}")
        cfg["seq_len"] = prompt_int("New sequence length", default=cfg["seq_len"])

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Epoch count — total epochs for this continued run.')}")
    cfg["epoch_count"] = prompt_int("New epoch count")
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Batch size — can be changed freely on resume.')}")
    cfg["batch_size"] = prompt_int("New batch size")

    tmode = int(cfg.get("tokenizer_mode", 1))
    if tmode in (-1, 0, 3):  # binary / byte / tiktoken — vocab-size independent
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Dataset switch — only available for tokenizers whose vocab size')}")
        print(f"  │  {_c(_DIM, 'is fixed (binary, byte, tiktoken). Char/word/BPE cannot switch')}")
        print(f"  │  {_c(_DIM, 'datasets because the embedding matrix is tied to vocab size.')}")
        change_ds = prompt_str("Switch to a different dataset?  (y/n)", default="n").lower() in ("y", "yes", "1")
        if change_ds:
            new_path = prompt_str("New dataset path  (blank = keep current)", default=cfg["dataset_path"])
            cfg["_changed_dataset"] = False
            if new_path and new_path != cfg["dataset_path"]:
                cfg["dataset_path"] = new_path
                cfg["_changed_dataset"] = True
            print(f"  │")
            print(f"  │  {_c(_DIM, 'Dataset type — 0 = sliding-window corpus, 1 = per-line examples.')}")
            dt = prompt_str("New dataset type  (0=standard  1=line  blank=keep)", default="")
            if dt.strip() in ("0", "1"):
                cfg["dataset_type"] = int(dt.strip())
                cfg["_changed_dataset"] = True
    cli_section_end(64)
    return cfg


def retry_adjustments(cfg):
    """Tune a saved configuration while deliberately starting fresh weights."""
    cli_section("Retry Hyperparameters", 64)
    print(f"  │  {_c(_DIM, 'Architecture, tokenizer, datasets, and validation setup are retained.')}")
    print(f"  │  {_c(_DIM, 'Progress is reset; no saved model weights are loaded.')}")
    print(f"  │")
    cfg["epoch_count"] = prompt_int("Epoch count", default=cfg.get("epoch_count", 1))
    cfg["batch_size"] = prompt_int("Batch size", default=cfg.get("batch_size", 1))
    cfg["grad_accum_steps"] = prompt_int(
        "Gradient accumulation steps  (1 = off)", default=cfg.get("grad_accum_steps", 1),
    )
    if DEVICE == "cuda":
        cfg["use_amp"] = bool(prompt_int(
            "Mixed precision / AMP  (0=off 1=on)", valid={0, 1}, default=int(cfg.get("use_amp", False)),
        ))
        if cfg["use_amp"]:
            cfg["amp_dtype"] = prompt_amp_dtype(cfg.get("amp_dtype", "fp16"))
    cfg["log_interval"] = prompt_int("Log loss every N update steps", default=cfg.get("log_interval", 10))
    cfg["sample_interval"] = prompt_int("Sample every N update steps", default=cfg.get("sample_interval", 500))
    cfg["val_interval"] = prompt_int("Run validation every N update steps", default=cfg.get("val_interval", 500))
    cfg["save_interval"] = prompt_int("Save every N update steps", default=cfg.get("save_interval", 10_000))
    cli_section_end(64)
    optimizer_cfg = prompt_optimizer_config(cfg)
    cfg["optimizer"] = optimizer_cfg["optimizer"]
    cfg["optim_params"] = optimizer_cfg["optim_params"]
    cfg["learning_rate"] = optimizer_cfg["optim_params"].get("lr", cfg.get("learning_rate", 0.0))
    cfg["iterations_done"] = 0
    cfg["train_tokens_done"] = 0
    # Retry creates fresh weights, so it may safely adopt the current
    # bottom-up decoder topology; regular resume deliberately cannot.
    if cfg.get("model_type") == MODEL_TYPE_MEGABYTE_BOTTOM_UP:
        cfg["megabyte_bottom_up_version"] = 6
    selected_stage_mixers = cfg.get("megabyte_stage_mixers", ())
    if isinstance(selected_stage_mixers, (str, int)):
        selected_stage_mixers = (selected_stage_mixers,)
    for config_key in (
        "megabyte_bottom_up_encoder_stage_mixers",
        "megabyte_bottom_up_decoder_stage_mixers",
    ):
        mixers = cfg.get(config_key, ())
        selected_stage_mixers += (mixers,) if isinstance(mixers, (str, int)) else tuple(mixers)
    if any(resolve_megabyte_stage_mixer(mixer) in {"rnn", "rnn_relu", "gru", "lstm"}
           for mixer in selected_stage_mixers):
        cfg["megabyte_fused_rnn_version"] = 2
    return cfg


# ── Optimizer Registry ────────────────────────────────────────────────────────
# Each entry: { "name": str, "class": callable_or_str, "defaults": {param: value},
#               "params": [ { "key": str, "prompt": str, "type": "float"|"int", "default": value } ] }
# To add a new optimizer: append an entry here and it will appear in the menu automatically.

OPTIMIZER_REGISTRY = [
    {
        "name": "Prodigy",
        "class": "prodigy",
        "defaults": {"lr": 1.0},
        "params": [
            {"key": "lr",      "prompt": "Learning rate  (Prodigy auto-scales; 1.0 is usually fine)", "type": "float", "default": 1.0},
            {"key": "slice_p", "prompt": "Slice P  (gradient slicing factor)", "type": "int", "default": 8},
        ],
    },
    {
        "name": "Adam",
        "class": "adam",
        "defaults": {"lr": 3e-4},
        "params": [
            {"key": "lr",      "prompt": "Learning rate", "type": "float", "default": 3e-4},
            {"key": "betas",   "prompt": "Betas  (comma-separated, e.g. 0.9,0.999)", "type": "betas", "default": (0.9, 0.999)},
            {"key": "eps",     "prompt": "Epsilon", "type": "float", "default": 1e-8},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
        ],
    },
    {
        "name": "SGD",
        "class": "sgd",
        "defaults": {"lr": 1e-2},
        "params": [
            {"key": "lr",       "prompt": "Learning rate", "type": "float", "default": 1e-2},
            {"key": "momentum", "prompt": "Momentum", "type": "float", "default": 0.9},
            {"key": "dampening","prompt": "Dampening", "type": "float", "default": 0.0},
            {"key": "nesterov", "prompt": "Nesterov  (0=off 1=on)", "type": "bool", "default": False},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
        ],
    },
    {
        "name": "RMSprop",
        "class": "rmsprop",
        "defaults": {"lr": 1e-3},
        "params": [
            {"key": "lr",       "prompt": "Learning rate", "type": "float", "default": 1e-3},
            {"key": "alpha",    "prompt": "Alpha  (smoothing constant)", "type": "float", "default": 0.99},
            {"key": "eps",      "prompt": "Epsilon", "type": "float", "default": 1e-8},
            {"key": "momentum", "prompt": "Momentum", "type": "float", "default": 0.0},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
        ],
    },
    {
        "name": "Rprop",
        "class": "rprop",
        "defaults": {"lr": 1e-3},
        "params": [
            {"key": "lr",   "prompt": "Learning rate", "type": "float", "default": 1e-3},
            {"key": "etas", "prompt": "Etas  (comma-separated, e.g. 0.5,1.2)", "type": "betas", "default": (0.5, 1.2)},
        ],
    },
    {
        "name": "Adagrad",
        "class": "adagrad",
        "defaults": {"lr": 1e-2},
        "params": [
            {"key": "lr",                "prompt": "Learning rate", "type": "float", "default": 1e-2},
            {"key": "lr_decay",          "prompt": "LR decay", "type": "float", "default": 0.0},
            {"key": "eps",               "prompt": "Epsilon", "type": "float", "default": 1e-10},
            {"key": "weight_decay",      "prompt": "Weight decay", "type": "float", "default": 0.0},
        ],
    },
    {
        "name": "Adadelta",
        "class": "adadelta",
        "defaults": {"lr": 1.0},
        "params": [
            {"key": "lr",    "prompt": "Learning rate", "type": "float", "default": 1.0},
            {"key": "rho",   "prompt": "Rho  (decay rate)", "type": "float", "default": 0.9},
            {"key": "eps",   "prompt": "Epsilon", "type": "float", "default": 1e-6},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
        ],
    },
    {
        "name": "Lamb",
        "class": "lamb",
        "defaults": {"lr": 1e-3},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 1e-3},
            {"key": "betas", "prompt": "Betas", "type": "betas", "default": (0.9, 0.999)},
            {"key": "eps", "prompt": "Epsilon", "type": "float", "default": 1e-6},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
            {"key": "adam", "prompt": "Adam fallback mode (0=off 1=on)", "type": "bool", "default": False},
        ],
    },
    {
        "name": "GrokFastAdamW",
        "class": "grokfast_adamw",
        "defaults": {"lr": 1e-4},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 1e-4},
            {"key": "betas", "prompt": "Betas", "type": "betas", "default": (0.9, 0.99)},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
            {"key": "grokfast", "prompt": "Enable GrokFast (0=off 1=on)", "type": "bool", "default": True},
        ],
    },
    {
        "name": "CLion",
        "class": "clion",
        "defaults": {"lr": 1e-4},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 1e-4},
            {"key": "betas", "prompt": "Betas", "type": "betas", "default": (0.95, 0.98)},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
        ],
    },
    {
        "name": "nSGDA",
        "class": "nsgda",
        "defaults": {"lr": 5e-2},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 5e-2},
        ],
    },
    {
        "name": "LayerWise nSGDA",
        "class": "layerwise_nsgda",
        "defaults": {"lr": 5e-2},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 5e-2},
            {"key": "momentum", "prompt": "Momentum", "type": "float", "default": 0.9},
            {"key": "cautious", "prompt": "Use cautious mask (0=off 1=on)", "type": "bool", "default": True},
        ],
    },
    {
        "name": "Ada-nSGDA",
        "class": "ada_nsgda",
        "defaults": {"lr": 1e-4},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 1e-4},
            {"key": "betas", "prompt": "Betas", "type": "betas", "default": (0.0, 0.99)},
            {"key": "eps", "prompt": "Epsilon", "type": "float", "default": 1e-8},
        ],
    },
    {
        "name": "CAdamW",
        "class": "cadamw",
        "defaults": {"lr": 1e-3},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 1e-3},
            {"key": "betas", "prompt": "Betas", "type": "betas", "default": (0.9, 0.999)},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
        ],
    },
    {
        "name": "RCAdamW",
        "class": "rcadamw",
        "defaults": {"lr": 2e-3},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 2e-3},
            {"key": "betas", "prompt": "Betas", "type": "betas", "default": (0.9, 0.999)},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
            {"key": "rectify", "prompt": "Enable RAdam rectification (0=off 1=on)", "type": "bool", "default": True},
        ],
    },
    {
        "name": "RCAdamW2",
        "class": "rcadamw2",
        "defaults": {"lr": 2e-3},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 2e-3},
            {"key": "betas", "prompt": "Betas", "type": "betas", "default": (0.9, 0.999)},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
        ],
    },
    {
        "name": "ARCAdamW",
        "class": "arcadamw",
        "defaults": {"lr": 2e-3},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 2e-3},
            {"key": "betas", "prompt": "Betas", "type": "betas", "default": (0.9, 0.999)},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
            {"key": "rectify", "prompt": "Rectify", "type": "bool", "default": True},
            {"key": "use_gc", "prompt": "Gradient Centralization", "type": "bool", "default": True},
            {"key": "amsgrad", "prompt": "AMSGrad", "type": "bool", "default": True},
        ],
    },
    {
        "name": "CSGD",
        "class": "csgd",
        "defaults": {"lr": 1e-3},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 1e-3},
            {"key": "momentum", "prompt": "Momentum", "type": "float", "default": 0.9},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
        ],
    },
    {
        "name": "CRMSprop",
        "class": "crmsprop_c",
        "defaults": {"lr": 1e-2},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 1e-2},
            {"key": "alpha", "prompt": "Alpha", "type": "float", "default": 0.99},
            {"key": "momentum", "prompt": "Momentum", "type": "float", "default": 0.9},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
        ],
    },
    {
        "name": "CAdagrad",
        "class": "cadagrad",
        "defaults": {"lr": 1e-2},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 1e-2},
            {"key": "momentum", "prompt": "Momentum", "type": "float", "default": 0.9},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
        ],
    },
    {
        "name": "CAdadelta",
        "class": "cadadelta_c",
        "defaults": {"lr": 1.0},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 1.0},
            {"key": "rho", "prompt": "Rho", "type": "float", "default": 0.9},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
        ],
    },
    {
        "name": "CLamb",
        "class": "clamb",
        "defaults": {"lr": 1e-3},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 1e-3},
            {"key": "betas", "prompt": "Betas", "type": "betas", "default": (0.9, 0.999)},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
            {"key": "adam", "prompt": "Use Adam mode (0=off 1=on)", "type": "bool", "default": False},
        ],
    },
    {
        "name": "CAdadeltaM",
        "class": "cadadeltam",
        "defaults": {"lr": 1.0},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 1.0},
            {"key": "rho", "prompt": "Rho", "type": "float", "default": 0.9},
            {"key": "momentum", "prompt": "Momentum", "type": "float", "default": 0.9},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
        ],
    },
    {
        "name": "CSRprop",
        "class": "csrprop",
        "defaults": {"lr": 1e-2},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 1e-2},
            {"key": "betas", "prompt": "Betas", "type": "betas", "default": (0.9, 0.999)},
            {"key": "etas", "prompt": "Etas", "type": "betas", "default": (0.5, 1.2)},
        ],
    },
    {
        "name": "CRprop",
        "class": "crprop_c",
        "defaults": {"lr": 1e-2},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 1e-2},
            {"key": "etas", "prompt": "Etas", "type": "betas", "default": (0.5, 1.2)},
        ],
    },
    {
        "name": "CAdamW_v7",
        "class": "cadamw_v7",
        "defaults": {"lr": 1e-3},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 1e-3},
            {"key": "betas", "prompt": "Betas", "type": "betas", "default": (0.9, 0.999)},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
        ],
    },
    {
        "name": "CAdamax",
        "class": "cadamax",
        "defaults": {"lr": 2e-3},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 2e-3},
            {"key": "betas", "prompt": "Betas", "type": "betas", "default": (0.9, 0.999)},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
        ],
    },
    {
        "name": "Adan",
        "class": "adan",
        "defaults": {"lr": 1e-3},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 1e-3},
            {"key": "betas", "prompt": "Betas (3 values)", "type": "betas", "default": (0.98, 0.92, 0.99)},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
            {"key": "l1", "prompt": "Use L1 regularization (0=off 1=on)", "type": "bool", "default": False},
        ],
    },
    {
        "name": "ModernCLion",
        "class": "modern_clion",
        "defaults": {"lr": 1e-4},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 1e-4},
            {"key": "betas", "prompt": "Betas", "type": "betas", "default": (0.9, 0.99)},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
            {"key": "nu", "prompt": "CLion threshold nu", "type": "float", "default": 1e-15},
        ],
    },
    {
        "name": "EqualizedAdamW",
        "class": "equalized_adamw",
        "defaults": {"lr": 1e-3},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 1e-3},
            {"key": "betas", "prompt": "Betas", "type": "betas", "default": (0.9, 0.999)},
            {"key": "eps", "prompt": "Epsilon", "type": "float", "default": 1e-8},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
            {"key": "lr_multiplier", "prompt": "Equalized LR multiplier", "type": "float", "default": 1.0},
        ],
    },
    {
        "name": "Muon",
        "class": "muon",
        "defaults": {"lr": 4.2e-4},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 4.2e-4},
            {"key": "momentum", "prompt": "Momentum", "type": "float", "default": 0.95},
            {"key": "rank", "prompt": "Muon rank (0=full Muon)", "type": "int", "default": 0},
            {"key": "muon_all", "prompt": "MuonAll (all parameters)", "type": "bool", "default": False},
            {"key": "muon_all_reshape", "prompt": "MuonAll: use near-square vector reshape?", "type": "bool", "default": False},
            {"key": "cautious", "prompt": "Cautious updates (0=off 1=on)", "type": "bool", "default": False},
            {"key": "orthogonalization_backend", "prompt": "Backend (newton_schulz / polar_express)", "type": "backend", "default": "polar_express"},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.1},
            {"key": "newton_schulz_iter", "prompt": "Orthogonalization iterations", "type": "int", "default": 5},
            {"key": "adam_betas", "prompt": "AdamW fallback betas", "type": "betas", "default": (0.9, 0.999)},
            {"key": "adam_eps", "prompt": "AdamW fallback epsilon", "type": "float", "default": 1e-8},
            {"key": "foreach", "prompt": "Batch optimizer updates (0=off 1=on)", "type": "bool", "default": True},
            {"key": "ns_bfloat16", "prompt": "BF16 matrix iterations (0=off 1=on)", "type": "bool", "default": False},
        ],
    },
    {
        "name": "AdaMuon",
        "class": "adamuon",
        "defaults": {"lr": 4.2e-4},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 4.2e-4},
            {"key": "eps", "prompt": "AdaMuon epsilon", "type": "float", "default": 1e-8},
            {"key": "nesterov", "prompt": "Nesterov (off=paper Algorithm 1)", "type": "bool", "default": False},
            {"key": "momentum", "prompt": "Momentum", "type": "float", "default": 0.95},
            {"key": "rank", "prompt": "AdaMuon rank (0=paper, >0=approximate)", "type": "int", "default": 0},
            {"key": "muon_all", "prompt": "MuonAll (all parameters)", "type": "bool", "default": False},
            {"key": "muon_all_reshape", "prompt": "MuonAll: use near-square vector reshape?", "type": "bool", "default": False},
            {"key": "cautious", "prompt": "Cautious updates (0=off 1=on)", "type": "bool", "default": False},
            {"key": "orthogonalization_backend", "prompt": "Backend (newton_schulz / polar_express)", "type": "backend", "default": "polar_express"},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.1},
            {"key": "newton_schulz_iter", "prompt": "Orthogonalization iterations", "type": "int", "default": 5},
            {"key": "adam_betas", "prompt": "AdamW fallback betas", "type": "betas", "default": (0.9, 0.999)},
            {"key": "adam_eps", "prompt": "AdamW fallback epsilon", "type": "float", "default": 1e-8},
            {"key": "foreach", "prompt": "Batch optimizer updates (0=off 1=on)", "type": "bool", "default": True},
            {"key": "ns_bfloat16", "prompt": "BF16 matrix iterations (0=off 1=on)", "type": "bool", "default": False},
        ],
    },
    {
        "name": "NorMuon", "class": "normuon", "defaults": {"lr": 4.2e-4},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": 4.2e-4},
            {"key": "momentum", "prompt": "Momentum (beta1)", "type": "float", "default": .95},
            {"key": "beta2", "prompt": "Row variance decay (beta2)", "type": "float", "default": .95},
            {"key": "eps", "prompt": "NorMuon epsilon", "type": "float", "default": 1e-8},
            {"key": "rank", "prompt": "Rank (0=full, >0=approximate)", "type": "int", "default": 0},
            {"key": "muon_all", "prompt": "MuonAll (all parameters)", "type": "bool", "default": False},
            {"key": "muon_all_reshape", "prompt": "MuonAll: use near-square vector reshape?", "type": "bool", "default": False},
            {"key": "cautious", "prompt": "Cautious updates (0=off 1=on)", "type": "bool", "default": False},
            {"key": "orthogonalization_backend", "prompt": "Backend (newton_schulz / polar_express)", "type": "backend", "default": "polar_express"},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": .1},
            {"key": "newton_schulz_iter", "prompt": "Orthogonalization iterations", "type": "int", "default": 5},
            {"key": "adam_betas", "prompt": "AdamW fallback betas", "type": "betas", "default": (.9, .999)},
            {"key": "adam_eps", "prompt": "AdamW fallback epsilon", "type": "float", "default": 1e-8},
            {"key": "foreach", "prompt": "Batch optimizer updates (0=off 1=on)", "type": "bool", "default": True},
            {"key": "ns_bfloat16", "prompt": "BF16 matrix iterations (0=off 1=on)", "type": "bool", "default": False},
        ],
    },
    {
        "name": "AdaGO", "class": "adago", "defaults": {"lr": .05},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": .05},
            {"key": "adam_lr", "prompt": "AdamW fallback learning rate (embeddings, heads, vectors)", "type": "float", "default": 4.2e-4},
            {"key": "momentum", "prompt": "Momentum", "type": "float", "default": .95},
            {"key": "gamma", "prompt": "Gradient norm cap (gamma)", "type": "float", "default": 1.},
            {"key": "v0", "prompt": "Initial norm accumulator (v0)", "type": "float", "default": 1.},
            {"key": "eps", "prompt": "Minimum step (epsilon)", "type": "float", "default": 5e-4},
            {"key": "rank", "prompt": "Rank (0=full, >0=approximate)", "type": "int", "default": 0},
            {"key": "muon_all", "prompt": "MuonAll (all parameters)", "type": "bool", "default": False},
            {"key": "muon_all_reshape", "prompt": "MuonAll: use near-square vector reshape?", "type": "bool", "default": False},
            {"key": "cautious", "prompt": "Cautious updates (0=off 1=on)", "type": "bool", "default": False},
            {"key": "orthogonalization_backend", "prompt": "Backend (newton_schulz / polar_express)", "type": "backend", "default": "polar_express"},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.},
            {"key": "newton_schulz_iter", "prompt": "Orthogonalization iterations", "type": "int", "default": 5},
            {"key": "adam_betas", "prompt": "AdamW fallback betas", "type": "betas", "default": (.9, .95)},
            {"key": "adam_eps", "prompt": "AdamW fallback epsilon", "type": "float", "default": 1e-8},
            {"key": "foreach", "prompt": "Batch optimizer updates (0=off 1=on)", "type": "bool", "default": True},
            {"key": "ns_bfloat16", "prompt": "BF16 matrix iterations (0=off 1=on)", "type": "bool", "default": False},
        ],
    },
    {
        "name": "AdamGO", "class": "adamgo", "defaults": {"lr": .05},
        "params": [
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": .05},
            {"key": "adam_lr", "prompt": "AdamW fallback learning rate (embeddings, heads, vectors)", "type": "float", "default": 4.2e-4},
            {"key": "momentum", "prompt": "Momentum", "type": "float", "default": .95},
            {"key": "gamma", "prompt": "Gradient norm cap (gamma)", "type": "float", "default": 1.},
            {"key": "beta2", "prompt": "Norm variance decay (beta2)", "type": "float", "default": .999},
            {"key": "delta", "prompt": "Denominator stabilizer (delta)", "type": "float", "default": 1e-8},
            {"key": "min_step", "prompt": "Minimum step (0=disabled)", "type": "float", "default": 0.},
            {"key": "rank", "prompt": "Rank (0=full, >0=approximate)", "type": "int", "default": 0},
            {"key": "muon_all", "prompt": "MuonAll (all parameters)", "type": "bool", "default": False},
            {"key": "muon_all_reshape", "prompt": "MuonAll: use near-square vector reshape?", "type": "bool", "default": False},
            {"key": "cautious", "prompt": "Cautious updates (0=off 1=on)", "type": "bool", "default": False},
            {"key": "orthogonalization_backend", "prompt": "Backend (newton_schulz / polar_express)", "type": "backend", "default": "polar_express"},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.},
            {"key": "newton_schulz_iter", "prompt": "Orthogonalization iterations", "type": "int", "default": 5},
            {"key": "adam_betas", "prompt": "AdamW fallback betas", "type": "betas", "default": (.9, .95)},
            {"key": "adam_eps", "prompt": "AdamW fallback epsilon", "type": "float", "default": 1e-8},
            {"key": "foreach", "prompt": "Batch optimizer updates (0=off 1=on)", "type": "bool", "default": True},
            {"key": "ns_bfloat16", "prompt": "BF16 matrix iterations (0=off 1=on)", "type": "bool", "default": False},
        ],
    },
    {
        "name": "RMSGO", "class": "rmsgo", "defaults": {"lr": .05},
        "params": [
            {"key": "beta2", "prompt": "Norm variance decay (beta2)", "type": "float", "default": .99},
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": .05},
            {"key": "adam_lr", "prompt": "AdamW fallback learning rate (embeddings, heads, vectors)", "type": "float", "default": 4.2e-4},
            {"key": "momentum", "prompt": "Momentum", "type": "float", "default": .95},
            {"key": "gamma", "prompt": "Gradient norm cap (gamma)", "type": "float", "default": 1.},
            {"key": "v0", "prompt": "Initial norm accumulator (v0)", "type": "float", "default": 1.},
            {"key": "eps", "prompt": "Minimum step (epsilon)", "type": "float", "default": 5e-4},
            {"key": "rank", "prompt": "Rank (0=full, >0=approximate)", "type": "int", "default": 0},
            {"key": "muon_all", "prompt": "MuonAll (all parameters)", "type": "bool", "default": False},
            {"key": "muon_all_reshape", "prompt": "MuonAll: use near-square vector reshape?", "type": "bool", "default": False},
            {"key": "cautious", "prompt": "Cautious updates (0=off 1=on)", "type": "bool", "default": False},
            {"key": "orthogonalization_backend", "prompt": "Backend (newton_schulz / polar_express)", "type": "backend", "default": "polar_express"},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.},
            {"key": "newton_schulz_iter", "prompt": "Orthogonalization iterations", "type": "int", "default": 5},
            {"key": "adam_betas", "prompt": "AdamW fallback betas", "type": "betas", "default": (.9, .95)},
            {"key": "adam_eps", "prompt": "AdamW fallback epsilon", "type": "float", "default": 1e-8},
            {"key": "foreach", "prompt": "Batch optimizer updates (0=off 1=on)", "type": "bool", "default": True},
            {"key": "ns_bfloat16", "prompt": "BF16 matrix iterations (0=off 1=on)", "type": "bool", "default": False},
        ],
    },
    {
        "name": "AdaDeltaGO", "class": "adadeltago", "defaults": {"lr": .05},
        "params": [
            {"key": "rho", "prompt": "Averaging decay (rho)", "type": "float", "default": .9},
            {"key": "lr", "prompt": "Learning rate", "type": "float", "default": .05},
            {"key": "adam_lr", "prompt": "AdamW fallback learning rate (embeddings, heads, vectors)", "type": "float", "default": 4.2e-4},
            {"key": "momentum", "prompt": "Momentum", "type": "float", "default": .95},
            {"key": "gamma", "prompt": "Gradient norm cap (gamma)", "type": "float", "default": 1.},
            {"key": "eps", "prompt": "RMS stabilizer (epsilon)", "type": "float", "default": 1e-6},
            {"key": "rank", "prompt": "Rank (0=full, >0=approximate)", "type": "int", "default": 0},
            {"key": "muon_all", "prompt": "MuonAll (all parameters)", "type": "bool", "default": False},
            {"key": "muon_all_reshape", "prompt": "MuonAll: use near-square vector reshape?", "type": "bool", "default": False},
            {"key": "cautious", "prompt": "Cautious updates (0=off 1=on)", "type": "bool", "default": False},
            {"key": "orthogonalization_backend", "prompt": "Backend (newton_schulz / polar_express)", "type": "backend", "default": "polar_express"},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.},
            {"key": "newton_schulz_iter", "prompt": "Orthogonalization iterations", "type": "int", "default": 5},
            {"key": "adam_betas", "prompt": "AdamW fallback betas", "type": "betas", "default": (.9, .95)},
            {"key": "adam_eps", "prompt": "AdamW fallback epsilon", "type": "float", "default": 1e-8},
            {"key": "foreach", "prompt": "Batch optimizer updates (0=off 1=on)", "type": "bool", "default": True},
            {"key": "ns_bfloat16", "prompt": "BF16 matrix iterations (0=off 1=on)", "type": "bool", "default": False},
        ],
    },
    {
        "name": "RAdamScheduleFree",
        "class": "radam_schedulefree",
        "defaults": {"lr": 2.5e-3},
        "params": [
            {"key": "lr", "prompt": "Learning rate  (no warmup or scheduler needed)", "type": "float", "default": 2.5e-3},
            {"key": "betas", "prompt": "Betas", "type": "betas", "default": (0.9, 0.999)},
            {"key": "eps", "prompt": "Epsilon", "type": "float", "default": 1e-8},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": 0.0},
        ],
    },
    {
        "name": "AdamHD",
        "class": "adamhd",
        "defaults": {"lr": 1e-3},
        "params": [
            {"key": "lr", "prompt": "Initial learning rate  (adapted online by hypergradient descent; 0 = start at zero, needs a max LR)", "type": "float", "default": 1e-3},
            {"key": "max_lr", "prompt": "Max learning rate  (0 = 100x the initial LR)", "type": "float", "default": 0.0},
            {"key": "hyper_lr", "prompt": "Hypergradient step  (max log-LR change per step)", "type": "float", "default": 0.05},
            {"key": "normalize", "prompt": "Normalized hypergradient  (0 = original raw-cosine rule)", "type": "bool", "default": True},
            {"key": "anchor", "prompt": "Anchor  (pull towards the peak LR; higher decays less; 1.0 normalized, 0.02 raw)", "type": "float", "default": 1.0},
            {"key": "horizon", "prompt": "Horizon  (EMA of past updates in the hypergradient; 0 = paper's one step)", "type": "float", "default": 0.9},
            {"key": "betas", "prompt": "Betas  (comma-separated, e.g. 0.9,0.999)", "type": "betas", "default": (0.9, 0.999)},
            {"key": "eps", "prompt": "Epsilon", "type": "float", "default": 1e-8},
            {"key": "weight_decay", "prompt": "Weight decay  (decoupled)", "type": "float", "default": 0.0},
        ],
    },
    {
        "name": "MuonHD", "class": "muonhd", "defaults": {"lr": 4.2e-4},
        "params": [
            {"key": "lr", "prompt": "Initial learning rate  (adapted online by hypergradient descent; 0 = start at zero, needs a max LR)", "type": "float", "default": 4.2e-4},
            {"key": "hyper_lr", "prompt": "Hypergradient step  (max log-LR change per step)", "type": "float", "default": 0.05},
            {"key": "normalize", "prompt": "Normalized hypergradient  (0 = original raw-cosine rule)", "type": "bool", "default": True},
            {"key": "anchor", "prompt": "Anchor  (pull towards the peak LR; higher decays less; 1.0 normalized, 0.02 raw)", "type": "float", "default": 1.0},
            {"key": "max_lr", "prompt": "Max learning rate  (0 = 100x the initial LR)", "type": "float", "default": 0.0},
            {"key": "momentum", "prompt": "Momentum", "type": "float", "default": .95},
            {"key": "rank", "prompt": "Rank (0=full, >0=approximate)", "type": "int", "default": 0},
            {"key": "muon_all", "prompt": "MuonAll (all parameters)", "type": "bool", "default": False},
            {"key": "muon_all_reshape", "prompt": "MuonAll: use near-square vector reshape?", "type": "bool", "default": False},
            {"key": "cautious", "prompt": "Cautious updates (0=off 1=on)", "type": "bool", "default": False},
            {"key": "orthogonalization_backend", "prompt": "Backend (newton_schulz / polar_express)", "type": "backend", "default": "polar_express"},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": .1},
            {"key": "newton_schulz_iter", "prompt": "Orthogonalization iterations", "type": "int", "default": 5},
            {"key": "adam_betas", "prompt": "AdamW fallback betas", "type": "betas", "default": (.9, .999)},
            {"key": "adam_eps", "prompt": "AdamW fallback epsilon", "type": "float", "default": 1e-8},
            {"key": "foreach", "prompt": "Batch optimizer updates (0=off 1=on)", "type": "bool", "default": True},
            {"key": "ns_bfloat16", "prompt": "BF16 matrix iterations (0=off 1=on)", "type": "bool", "default": False},
        ],
    },
    {
        "name": "NorMuonHD", "class": "normuonhd", "defaults": {"lr": 4.2e-4},
        "params": [
            {"key": "lr", "prompt": "Initial learning rate  (adapted online by hypergradient descent; 0 = start at zero, needs a max LR)", "type": "float", "default": 4.2e-4},
            {"key": "hyper_lr", "prompt": "Hypergradient step  (max log-LR change per step)", "type": "float", "default": 0.05},
            {"key": "normalize", "prompt": "Normalized hypergradient  (0 = original raw-cosine rule)", "type": "bool", "default": True},
            {"key": "anchor", "prompt": "Anchor  (pull towards the peak LR; higher decays less; 1.0 normalized, 0.02 raw)", "type": "float", "default": 1.0},
            {"key": "max_lr", "prompt": "Max learning rate  (0 = 100x the initial LR)", "type": "float", "default": 0.0},
            {"key": "momentum", "prompt": "Momentum (beta1)", "type": "float", "default": .95},
            {"key": "beta2", "prompt": "Row variance decay (beta2)", "type": "float", "default": .95},
            {"key": "eps", "prompt": "NorMuon epsilon", "type": "float", "default": 1e-8},
            {"key": "rank", "prompt": "Rank (0=full, >0=approximate)", "type": "int", "default": 0},
            {"key": "muon_all", "prompt": "MuonAll (all parameters)", "type": "bool", "default": False},
            {"key": "muon_all_reshape", "prompt": "MuonAll: use near-square vector reshape?", "type": "bool", "default": False},
            {"key": "cautious", "prompt": "Cautious updates (0=off 1=on)", "type": "bool", "default": False},
            {"key": "orthogonalization_backend", "prompt": "Backend (newton_schulz / polar_express)", "type": "backend", "default": "polar_express"},
            {"key": "weight_decay", "prompt": "Weight decay", "type": "float", "default": .1},
            {"key": "newton_schulz_iter", "prompt": "Orthogonalization iterations", "type": "int", "default": 5},
            {"key": "adam_betas", "prompt": "AdamW fallback betas", "type": "betas", "default": (.9, .999)},
            {"key": "adam_eps", "prompt": "AdamW fallback epsilon", "type": "float", "default": 1e-8},
            {"key": "foreach", "prompt": "Batch optimizer updates (0=off 1=on)", "type": "bool", "default": True},
            {"key": "ns_bfloat16", "prompt": "BF16 matrix iterations (0=off 1=on)", "type": "bool", "default": False},
        ],
    },
]


def _get_optimizer_names():
    return [o["name"] for o in OPTIMIZER_REGISTRY]

def _parse_optim_param(raw_str, ptype, default):
    """Parse a single optimizer parameter from user input string."""
    if raw_str.strip() == "":
        return default
    if ptype == "float":
        return float(raw_str)
    if ptype == "int":
        return int(raw_str)
    if ptype == "bool":
        v = raw_str.strip().lower()
        return v in ("1", "true", "yes", "y")
    if ptype == "betas":
        parts = [float(x.strip()) for x in raw_str.split(",")]
        return tuple(parts)
    if ptype == "backend":
        backend = raw_str.strip().casefold()
        if backend not in ("newton_schulz", "polar_express"):
            raise ValueError("Backend must be newton_schulz or polar_express")
        return backend
    return raw_str

def prompt_optimizer_config(saved_config=None):
    """Interactive optimizer selection + per-optimizer param prompts.
    Returns dict: {"optimizer": str, "optim_params": {key: value, ...}}
    """
    cli_section("Optimizer", 64)
    print(f"  │")
    for i, entry in enumerate(OPTIMIZER_REGISTRY):
        desc = ", ".join(f"{p['key']}={p['default']}" for p in entry["params"][:3])
        cli_opt(i, entry["name"], desc)
    print(f"  │")
    saved_config = saved_config or {}
    saved_optimizer = saved_config.get("optimizer")
    default_choice = next(
        (index for index, candidate in enumerate(OPTIMIZER_REGISTRY)
         if candidate["class"] == saved_optimizer),
        24,
    )
    choice = prompt_int("Optimizer", valid=set(range(len(OPTIMIZER_REGISTRY))), default=default_choice)
    entry = OPTIMIZER_REGISTRY[choice]

    print(f"  │")
    opt_name = entry["name"]
    print(f"  │  {_c(_DIM, f'Configure {opt_name} parameters  (press Enter for default):')}")
    optim_params = {}
    saved_params = saved_config.get("optim_params", {}) if entry["class"] == saved_optimizer else {}
    for p in entry["params"]:
        default = saved_params.get(p["key"], p["default"])
        label = prompt_label(p["prompt"], default)
        raw = input(label).strip()
        optim_params[p["key"]] = _parse_optim_param(raw, p["type"], default)

    cli_section_end(64)
    return {"optimizer": entry["class"], "optim_params": optim_params}


def layerwise_param_groups(model: nn.Module):
    groups = []
    seen = set()
    for module in model.modules():
        params = []
        for p in module.parameters(recurse=False):
            if p.requires_grad and id(p) not in seen:
                params.append(p)
                seen.add(id(p))
        if params:
            groups.append({"params": params})
    leftover = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    if leftover:
        groups.append({"params": leftover})
    return groups or [{"params": [p for p in model.parameters() if p.requires_grad]}]


def linegen_muon_param_groups(model, optim_groups):
    """Keep token/position embeddings and vocabulary heads on AdamW.

    Preserve the existing decay groups; internal attention output projections
    remain Muon matrices. The shared helper handles tied embedding/head weights.
    """
    adamw_params = []
    for name, param in model.named_parameters():
        parts = name.split('.')
        if any('embed' in part or part in {'pos', 'head', 'lm_head'} for part in parts):
            adamw_params.append(param)
    root = model
    while hasattr(root, '_orig_mod'):
        root = root._orig_mod
    # A few LM families call their vocabulary projection `out`; the one-hot
    # MLP and NeuralNGramLM put it at the end of a top-level Sequential.
    if isinstance(getattr(root, 'out', None), nn.Linear):
        adamw_params.extend(root.out.parameters())
    if isinstance(getattr(root, 'mlp', None), nn.Sequential) and len(root.mlp):
        adamw_params.extend(root.mlp[-1].parameters())
    return muon_param_groups(model, adamw_params, optim_groups)


def build_optimizer(model, cfg):
    wd = cfg.get("optim_params", {}).get(
        "weight_decay", 0.1 if cfg.get("optimizer") in {"muon", "adamuon", "normuon", "muonhd", "normuonhd"} else 0.0)
    decay_params = []
    no_decay_params = []

    # Substrings to identify parameters that should NOT decay
    blacklist = [
        'embed', 'pos', 'tok',       # Embeddings (including GPT2 'tok')
        'norm', 'ln', 'gn',          # Normalization (LayerNorm, RMSNorm, GroupNorm)
        'alphas', 'betas',           # ReZero scalars
        'res_scale',                 # GatedMLP residual scale
        'gamma',                     # RetNet gammas
        'time_mix', 'w_tau',         # RWKV / Liquid parameters
        'log_A'                      # Mamba/Scan decay parameters
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(key in name for key in blacklist):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    print(f"[Optimizer] Decay: {len(decay_params)} tensors | No Decay: {len(no_decay_params)} tensors (Embeds/Norms/Gates)")

    optim_groups = [
        {'params': decay_params, 'weight_decay': wd},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    optimizer_key = cfg.get("optimizer", "prodigy")
    op = cfg.get("optim_params", {})

    if optimizer_key in {"muon", "adamuon", "normuon", "adago", "adamgo", "rmsgo", "adadeltago", "muonhd", "normuonhd"}:
        optimizer_cls = (AdaDeltaGO if optimizer_key == 'adadeltago' else RMSGO if optimizer_key == 'rmsgo' else AdamGO if optimizer_key == 'adamgo' else AdaGO if optimizer_key == 'adago' else NorMuon if optimizer_key == 'normuon'
                         else AdaMuon if optimizer_key == 'adamuon' else MuonHD if optimizer_key == 'muonhd'
                         else NorMuonHD if optimizer_key == 'normuonhd' else Muon)
        adaptive = ({'eps': op.get('eps', 1e-8), 'nesterov': op.get('nesterov', False)}
                    if optimizer_key == 'adamuon' else {})
        if optimizer_key in ('normuon', 'normuonhd'):
            adaptive = {'eps': op.get('eps', 1e-8), 'beta2': op.get('beta2', .95)}
        if optimizer_key in ('muonhd', 'normuonhd'):
            # Hypergradient LR (AdamHD rule); max_lr 0 = no absolute cap.
            normalize = bool(op.get('normalize', True))
            adaptive.update(hyper_lr=op.get('hyper_lr', 0.05), normalize=normalize,
                            anchor=op.get('anchor', 1.0 if normalize else 0.02),
                            max_lr=float(op.get('max_lr', 0.0) or 0.0) or None)
        if optimizer_key == 'adago':
            adaptive = {'eps': op.get('eps', 5e-4), 'gamma': op.get('gamma', 1.), 'v0': op.get('v0', 1.)}
        if optimizer_key == 'rmsgo':
            adaptive = {'eps': op.get('eps', 5e-4), 'gamma': op.get('gamma', 1.), 'v0': op.get('v0', 1.)}
            adaptive['beta2'] = op.get('beta2', .99)
        if optimizer_key == 'adadeltago':
            adaptive = {'rho': op.get('rho', .9), 'gamma': op.get('gamma', 1.), 'eps': op.get('eps', 1e-6)}
        if optimizer_key == 'adamgo':
            adaptive = {'beta2': op.get('beta2', .999), 'gamma': op.get('gamma', 1.),
                        'delta': op.get('delta', 1e-8), 'min_step': op.get('min_step', 0.)}
        optimizer = optimizer_cls(
            linegen_muon_param_groups(model, optim_groups),
            lr=op.get("lr", .05 if optimizer_key in {"adago", "adamgo", "rmsgo", "adadeltago"} else 4.2e-4), momentum=op.get("momentum", 0.95),
            weight_decay=wd, newton_schulz_iter=op.get("newton_schulz_iter", 5),
            adam_betas=tuple(op.get("adam_betas", (0.9, 0.95 if optimizer_key in {"adago", "adamgo", "rmsgo", "adadeltago"} else 0.999))),
            adam_eps=op.get("adam_eps", 1e-8), foreach=op.get("foreach", True),
            ns_bfloat16=op.get("ns_bfloat16", False),
            rank=op.get("rank", 0),
            cautious=op.get("cautious", False),
            muon_all=op.get("muon_all", False),
            muon_all_reshape=op.get("muon_all_reshape", False),
            orthogonalization_backend=op.get("orthogonalization_backend", "polar_express"),
            **adaptive,
        )
        # Separate AdamW-fallback LR; GO matrix LRs (0.05) are far too large
        # for Adam. Absent = legacy shared LR. MuonAll has no fallback groups.
        adam_lr = op.get("adam_lr")
        if adam_lr is not None and not op.get("muon_all", False):
            if isinstance(adam_lr, bool) or not math.isfinite(adam_lr) or adam_lr <= 0:
                raise ValueError(f"adam_lr must be a finite positive number, got {adam_lr!r}")
            for group in optimizer.param_groups:
                if not group["use_muon"]:
                    group["lr"] = adam_lr
        return optimizer
    elif optimizer_key == "prodigy":
        return Prodigy(optim_groups, lr=op.get("lr", 1.0), slice_p=op.get("slice_p", 8))
    elif optimizer_key == "adam":
        betas = op.get("betas", (0.9, 0.999))
        if isinstance(betas, list): betas = tuple(betas)
        return torch.optim.Adam(optim_groups, lr=op.get("lr", 3e-4),
                                betas=betas, eps=op.get("eps", 1e-8))
    elif optimizer_key == "adamhd":
        betas = op.get("betas", (0.9, 0.999))
        if isinstance(betas, list): betas = tuple(betas)
        max_lr = float(op.get("max_lr", 0.0) or 0.0)
        if float(op.get("lr", 1e-3)) == 0.0 and max_lr <= 0.0:
            raise ValueError("AdamHD: a zero initial learning rate needs a max learning rate > 0")
        # Runs saved before the option existed keep the original raw rule.
        normalize = bool(op.get("normalize", False))
        return AdamHD(optim_groups, lr=op.get("lr", 1e-3), hyper_lr=op.get("hyper_lr", 0.05),
                      anchor=op.get("anchor", 1.0 if normalize else 0.02), horizon=op.get("horizon", 0.9),
                      normalize=normalize,
                      betas=betas, eps=op.get("eps", 1e-8), max_lr=max_lr if max_lr > 0.0 else None)
    elif optimizer_key == "equalized_adamw":
        betas = op.get("betas", (0.9, 0.999))
        if isinstance(betas, list): betas = tuple(betas)
        return EqualizedAdamW(
            optim_groups,
            lr=op.get("lr", 1e-3),
            betas=betas,
            eps=op.get("eps", 1e-8),
            weight_decay=op.get("weight_decay", 0.0),
            lr_multiplier=op.get("lr_multiplier", 1.0),
        )
    elif optimizer_key == "sgd":
        return torch.optim.SGD(optim_groups, lr=op.get("lr", 1e-2),
                               momentum=op.get("momentum", 0.9),
                               dampening=op.get("dampening", 0.0),
                               nesterov=op.get("nesterov", False))
    elif optimizer_key == "rmsprop":
        return torch.optim.RMSprop(optim_groups, lr=op.get("lr", 1e-3),
                                   alpha=op.get("alpha", 0.99),
                                   eps=op.get("eps", 1e-8),
                                   momentum=op.get("momentum", 0.0))
    elif optimizer_key == "rprop":
        etas = op.get("etas", (0.5, 1.2))
        if isinstance(etas, list): etas = tuple(etas)
        return torch.optim.Rprop(optim_groups, lr=op.get("lr", 1e-3), etas=etas)
    elif optimizer_key == "adagrad":
        return torch.optim.Adagrad(optim_groups, lr=op.get("lr", 1e-2),
                                   lr_decay=op.get("lr_decay", 0.0),
                                   eps=op.get("eps", 1e-10))
    elif optimizer_key == "adadelta":
        return torch.optim.Adadelta(optim_groups, lr=op.get("lr", 1.0),
                                    rho=op.get("rho", 0.9),
                                    eps=op.get("eps", 1e-6))
    
    # === CUSTOM OPTIMIZERS ===
    elif optimizer_key == "lamb":
        betas = op.get("betas", (0.9, 0.999))
        if isinstance(betas, list): betas = tuple(betas)
        return Lamb(optim_groups, lr=op.get("lr", 1e-3), betas=betas,
                    eps=op.get("eps", 1e-6), adam=op.get("adam", False))
        
    elif optimizer_key == "grokfast_adamw":
        betas = op.get("betas", (0.9, 0.99))
        if isinstance(betas, list): betas = tuple(betas)
        return GrokFastAdamW(optim_groups, lr=op.get("lr", 1e-4), betas=betas, eps=op.get("eps", 1e-8), grokfast=op.get("grokfast", True))
        
    elif optimizer_key == "clion":
        betas = op.get("betas", (0.95, 0.98))
        if isinstance(betas, list): betas = tuple(betas)
        return CLion(optim_groups, lr=op.get("lr", 1e-4), betas=betas,
                     weight_decay=op.get("weight_decay", 0.0))

    elif optimizer_key == "modern_clion":
        betas = op.get("betas", (0.9, 0.99))
        if isinstance(betas, list): betas = tuple(betas)
        return ModernCLion(optim_groups, lr=op.get("lr", 1e-4), betas=betas,
                           weight_decay=op.get("weight_decay", 0.0),
                           nu=op.get("nu", 1e-15))

    elif optimizer_key == "nsgda":
        return NSGDA(model.parameters(), lr=op.get("lr", 5e-2))

    elif optimizer_key == "layerwise_nsgda":
        return NSGDA(
            layerwise_param_groups(model),
            lr=op.get("lr", 5e-2),
            momentum=op.get("momentum", 0.9),
            cautious=op.get("cautious", True),
        )

    elif optimizer_key == "ada_nsgda":
        betas = op.get("betas", (0.0, 0.99))
        if isinstance(betas, list): betas = tuple(betas)
        return AdaNSGDA(model.parameters(), lr=op.get("lr", 1e-4),
                        betas=betas, eps=op.get("eps", 1e-8))
	        
    elif optimizer_key == "cadamw":
        betas = op.get("betas", (0.9, 0.999))
        if isinstance(betas, list): betas = tuple(betas)
        return CAdamW(optim_groups, lr=op.get("lr", 1e-3), betas=betas, eps=op.get("eps", 1e-8))
        
    elif optimizer_key == "rcadamw":
        betas = op.get("betas", (0.9, 0.999))
        if isinstance(betas, list): betas = tuple(betas)
        return RCAdamW(optim_groups, lr=op.get("lr", 2e-3), betas=betas, eps=op.get("eps", 1e-8), rectify=op.get("rectify", True))
        
    elif optimizer_key == "rcadamw2":
        betas = op.get("betas", (0.9, 0.999))
        if isinstance(betas, list): betas = tuple(betas)
        return RCAdamW2(optim_groups, lr=op.get("lr", 2e-3), betas=betas, eps=op.get("eps", 1e-8))
        
    elif optimizer_key == "arcadamw":
        betas = op.get("betas", (0.9, 0.999))
        if isinstance(betas, list): betas = tuple(betas)
        return ARCAdamW(optim_groups, lr=op.get("lr", 2e-3), betas=betas, eps=op.get("eps", 1e-8),
                        rectify=op.get("rectify", True), use_gc=op.get("use_gc", True), amsgrad=op.get("amsgrad", True))
                        
    elif optimizer_key == "csgd":
        return CSGD(optim_groups, lr=op.get("lr", 1e-3), momentum=op.get("momentum", 0.9))
        
    elif optimizer_key == "crmsprop_c":
        return CRMSprop(optim_groups, lr=op.get("lr", 1e-2), alpha=op.get("alpha", 0.99), 
                        eps=op.get("eps", 1e-8), momentum=op.get("momentum", 0.9))
                        
    elif optimizer_key == "cadagrad":
        return CAdagrad(optim_groups, lr=op.get("lr", 1e-2), momentum=op.get("momentum", 0.9), eps=op.get("eps", 1e-10))
        
    elif optimizer_key == "cadadelta_c":
        return CAdadelta(optim_groups, lr=op.get("lr", 1.0), rho=op.get("rho", 0.9), eps=op.get("eps", 1e-6))
        
    elif optimizer_key == "clamb":
        betas = op.get("betas", (0.9, 0.999))
        if isinstance(betas, list): betas = tuple(betas)
        return CLamb(optim_groups, lr=op.get("lr", 1e-3), betas=betas, eps=op.get("eps", 1e-6), adam=op.get("adam", False))
        
    elif optimizer_key == "cadadeltam":
        return CAdadeltaM(optim_groups, lr=op.get("lr", 1.0), rho=op.get("rho", 0.9), momentum=op.get("momentum", 0.9), eps=op.get("eps", 1e-6))
        
    elif optimizer_key == "csrprop":
        betas = op.get("betas", (0.9, 0.999))
        if isinstance(betas, list): betas = tuple(betas)
        etas = op.get("etas", (0.5, 1.2))
        if isinstance(etas, list): etas = tuple(etas)
        return CSRprop(optim_groups, lr=op.get("lr", 1e-2), betas=betas, etas=etas)
        
    elif optimizer_key == "crprop_c":
        etas = op.get("etas", (0.5, 1.2))
        if isinstance(etas, list): etas = tuple(etas)
        return CRprop(optim_groups, lr=op.get("lr", 1e-2), etas=etas)
        
    elif optimizer_key == "cadamw_v7":
        betas = op.get("betas", (0.9, 0.999))
        if isinstance(betas, list): betas = tuple(betas)
        return CAdamW_v7(optim_groups, lr=op.get("lr", 1e-3), betas=betas, eps=op.get("eps", 1e-8))
        
    elif optimizer_key == "cadamax":
        betas = op.get("betas", (0.9, 0.999))
        if isinstance(betas, list): betas = tuple(betas)
        return CAdamax(optim_groups, lr=op.get("lr", 2e-3), betas=betas, eps=op.get("eps", 1e-8))
        
    elif optimizer_key == "adan":
        # Note: Adan expects 3 values for betas
        betas = op.get("betas", (0.98, 0.92, 0.99))
        if isinstance(betas, list): betas = tuple(betas)
        if op.get("l1", False) == 1 or op.get("l1", False) == "1":
            l1s=True
        else:
            l1s=False
        return Adan(
            optim_groups, 
            lr=op.get("lr", 1e-3), 
            betas=betas, 
            eps=op.get("eps", 1e-8), 
            l1=l1s,#op.get("l1", False)
        )
    elif optimizer_key == "radam_schedulefree":
        betas = op.get("betas", (0.9, 0.999))
        if isinstance(betas, list): betas = tuple(betas)
        optimizer = RAdamScheduleFree(optim_groups, lr=op.get("lr", 2.5e-3), betas=betas, eps=op.get("eps", 1e-8))
        optimizer.train()  # step() requires train mode; evaluation and saving swap to x
        return optimizer
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_key}")
AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def amp_settings(cfg):
    """Return (enabled, autocast dtype, GradScaler or None) for a config.

    ``amp_dtype`` defaults to fp16 so configs saved before the option existed
    keep their behaviour.  bf16 has fp32's exponent range and needs no loss
    scaling; it falls back to fp16 on GPUs without bf16 support.
    """
    use_amp = bool(cfg.get("use_amp", False)) and DEVICE == "cuda"
    name = str(cfg.get("amp_dtype", "fp16")).lower()
    if name not in AMP_DTYPES:
        raise ValueError(f"Unknown AMP dtype {name!r}; expected one of {sorted(AMP_DTYPES)}")
    if use_amp and name == "bf16" and not torch.cuda.is_bf16_supported():
        pwarn("bf16 AMP is not supported on this GPU; using fp16")
        name = "fp16"
    dtype = AMP_DTYPES[name]
    scaler = torch.amp.GradScaler("cuda") if use_amp and dtype == torch.float16 else None
    return use_amp, dtype, scaler


def prompt_amp_dtype(default="fp16"):
    """Ask for the autocast precision; returns 'fp16' or 'bf16'."""
    print(f"  │  {_c(_DIM, 'fp16 uses loss scaling; bf16 (Ampere+) keeps the fp32 exponent range and needs none.')}")
    choice = prompt_int("AMP precision  (0=fp16 1=bf16)", valid={0, 1}, default=int(default == "bf16"))
    return "bf16" if choice == 1 else "fp16"


def wrap_model_with_compile(model, cfg):
    """Best-effort family-aware compilation with a safe eager fallback.

    All zoo models must accept tensor inputs and return tensors or fixed nested
    tensor state.  Individual recurrent time loops may still graph-break under
    older PyTorch releases, so we deliberately do not require a full graph.
    """
    if not cfg.get("use_compile", False):
        return model
    # Inductor compiles lazily on the first forward.  A wrapping-time try/except
    # alone cannot catch unsupported Triton/CUDA kernel images, so ask Dynamo to
    # retain the eager graph whenever a backend compilation fails.
    torch._dynamo.config.suppress_errors = True
    msel = cfg["model_selection"]
    if msel in {301, 304, 305, 400, 300, 306}:
        family, mode, dynamic = "transformer", "max-autotune", True
    elif msel in SCAN_MODEL_IDS:
        family, mode, dynamic = "scan/SSM", "reduce-overhead", True
    elif msel in RNN_MODEL_IDS and not is_bottom_up_megabyte(cfg):
        family, mode, dynamic = "recurrent", "reduce-overhead", True
    else:
        family, mode, dynamic = "MLP/mixer", "reduce-overhead", True
    try:
        # `aot_eager` is the portable default.  Select `inductor` explicitly in
        # the saved config only on a CUDA/Triton combination known to support
        # the installed GPU; otherwise it can fail with an invalid kernel image.
        backend = cfg.get("compile_backend", "aot_eager")
        print(f"🚀 torch.compile: {family}, backend={backend}, mode={mode}, dynamic={dynamic}")
        kwargs = dict(backend=backend, dynamic=dynamic, fullgraph=False)
        # `mode` is an Inductor option; passing it to aot_eager itself causes
        # a deferred backend error on PyTorch 2.4.
        if backend == "inductor":
            kwargs["mode"] = mode
        return torch.compile(model, **kwargs)
    except Exception as exc:
        print(f"⚠️ torch.compile unavailable for {family}: {exc}; using eager mode")
        return model
# ========= NEW MODES =========

def run_interactive_chat(cfg, model, vocab):
    """Interactive multi-turn text generation."""
    model.eval()
    line_mode = (cfg["dataset_type"] == 1)

    cli_banner("Interactive Chat", "Multi-turn text generation session", width=64)
    W = 62
    print(f"  {_c(_CY, _B, '┌─')} {_c(_WH, _B, 'Runtime Commands')} {_c(_CY, '─' * (W - 24) + '┐')}")
    cli_opt("/temp N",  "Set temperature",      "e.g. /temp 0.8  — 0 = greedy, 1 = unmodified")
    cli_opt("/topk N",  "Set top-k",            "e.g. /topk 50   — keep N most likely tokens")
    cli_opt("/topp N",  "Set top-p",            "e.g. /topp 0.9  — nucleus sampling threshold")
    cli_opt("/rep N",   "Set rep. penalty",     "e.g. /rep 1.2   — penalises repeated tokens")
    cli_opt("/len N",   "Set generation length","e.g. /len 300   — max tokens per response")
    cli_opt("/help",    "Show this list",       "")
    cli_opt("/quit",    "Exit chat",            "Also Ctrl-C")
    cli_blank_row()
    print(f"  {_c(_CY, '└' + '─' * (W - 2) + '┘')}\n")

    temp = cfg.get("temperature", 1.0)
    top_k = 0; top_p = 0.0; rep_penalty = 1.0
    gen_len = 200 if not line_mode else cfg.get("seq_len", 512)
    
    while True:
        try:
            user_input = input("You> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat."); break
        if not user_input.strip(): continue
        
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()
            if cmd in ("/quit", "/exit"): break
            elif cmd == "/temp" and len(parts) > 1: temp = float(parts[1]); print(f"  Temperature={temp}"); continue
            elif cmd == "/topk" and len(parts) > 1: top_k = int(parts[1]); print(f"  Top-k={top_k}"); continue
            elif cmd == "/topp" and len(parts) > 1: top_p = float(parts[1]); print(f"  Top-p={top_p}"); continue
            elif cmd == "/rep" and len(parts) > 1: rep_penalty = float(parts[1]); print(f"  Rep penalty={rep_penalty}"); continue
            elif cmd == "/len" and len(parts) > 1: gen_len = int(parts[1]); print(f"  Gen length={gen_len}"); continue
            elif cmd == "/help": print("  /temp /topk /topp /rep /len /quit"); continue
            else: print(f"  Unknown: {cmd}"); continue
        
        cfg["temperature"] = temp
        cfg["_top_k"] = top_k; cfg["_top_p"] = top_p; cfg["_rep_penalty"] = rep_penalty
        
        if line_mode and cfg.get("seq2seq"):
            # Seq2seq: the user types the input columns; reply with the outputs.
            try:
                p_ids = encode_seq2seq_prompt(vocab, cfg["seq2seq"], user_input)
            except ValueError as exc:
                pwarn(str(exc)); continue
            out_ids = generate_line_mode(model, cfg, vocab, p_ids, limit_len=gen_len)
            print(f"Model> {seq2seq_visible(vocab.decode(out_ids[len(p_ids):]))}\n")
        elif line_mode:
            p_ids = vocab.encode(BOS_TOKEN + user_input)
            out_ids = generate_line_mode(model, cfg, vocab, p_ids, limit_len=gen_len)
            text = vocab.decode(out_ids[1:]) if len(out_ids) > 1 else ""
            print(f"Model> {text}\n")
        else:
            p_ids = vocab.encode(user_input)
            sys.stdout.write("Model> ")
            generate_classic(model, cfg, vocab, p_ids, max_len=gen_len, stream=True)
            print()


@torch.no_grad()
def run_perplexity_eval():
    """Comprehensive perplexity evaluation."""
    cli_banner("Perplexity Evaluation", "Measure how well the model predicts held-out text", width=64)
    if not os.path.exists(CONFIG_PATH):
        pwarn("No config found — train a model first."); return

    cfg = load_run_config()
    vocab = load_or_make_vocab(cfg, cfg["dataset_path"])
    model = build_model(cfg, vocab.size)
    model.to(DEVICE)
    pinfo(f"Loading checkpoint from {CHECKPOINT_PATH} …")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

    is_scan = cfg["model_selection"] in SCAN_MODEL_IDS
    if is_scan: model.train()
    else: model.eval()

    cli_section("Evaluation Settings", 64)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Evaluation file — the text file to measure perplexity on.')}")
    print(f"  │  {_c(_DIM, 'Should be held-out data the model has never seen during training.')}")
    print(f"  │  {_c(_DIM, 'Defaults to the training file if left blank.')}")
    if cfg.get("seq2seq"):
        print(f"  │  {_c(_DIM, 'Seq2seq: another file must have the same columns and delimiter;')}")
        print(f"  │  {_c(_DIM, 'only the output columns are scored.')}")
    eval_path = prompt_str("Evaluation file path", default=cfg["dataset_path"])

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Batch count — how many random batches to sample from the file.')}")
    print(f"  │  {_c(_DIM, 'More batches = more accurate estimate, more time. 100 is typical.')}")
    max_batches = prompt_int("Number of batches", default=100)

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Batch size — sequences per batch. Larger is faster but uses more')}")
    print(f"  │  {_c(_DIM, 'VRAM. Must fit in memory alongside the model.')}")
    batch_size = prompt_int("Batch size", default=cfg["batch_size"])
    cli_section_end(64)

    line_mode = (cfg["dataset_type"] == 1)
    pad_id = getattr(vocab, "pad_id", -100)
    if pad_id is None: pad_id = -100
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')

    if line_mode:
        spec = cfg.get("seq2seq")
        if spec and eval_path != cfg["dataset_path"]:
            # A different file is raw delimited data with the training columns.
            eval_path = prepare_seq2seq_dataset(spec, source_path=eval_path)
        eval_ds = IndexedLineDataset(eval_path, vocab, seq2seq_inputs=len(spec["input_cols"]) if spec else 0)
    else:
        eval_ds = MemmapClassicDataset(eval_path, vocab, cfg["seq_len"])

    total_loss = 0.0; total_tokens = 0
    n_batches = max_batches if max_batches > 0 else 100

    for _ in tqdm(range(n_batches), desc="Perplexity"):
        x, y = eval_ds.get_batch(batch_size)
        if is_scan or cfg["model_selection"] in RNN_MODEL_IDS:
            out = model(x); logits = out[0] if isinstance(out, tuple) else out
        else:
            logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        non_pad = (y != pad_id).sum().item() if pad_id >= 0 else y.numel()
        total_loss += loss.item(); total_tokens += non_pad

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 100))
    bpc = avg_loss / math.log(2)

    W = 62
    print()
    cli_section("Results", W)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Tokens evaluated :')}  {_c(_WH, readable_num(total_tokens))}")
    print(f"  │  {_c(_DIM, 'Cross-entropy loss:')}  {_c(_WH, f'{avg_loss:.4f}')}  {_c(_DIM, '(nats/token)')}")
    print(f"  │  {_c(_DIM, 'Perplexity        :')}  {_c(_WH, _B, f'{ppl:.2f}')}  {_c(_DIM, '(lower = better; random baseline ≈ vocab size)')}")
    print(f"  │  {_c(_DIM, 'Bits per char     :')}  {_c(_WH, f'{bpc:.4f}')}  {_c(_DIM, '(loss base-2; well-trained char models reach ~1.2)')}")
    cli_section_end(W)


def run_model_stats():
    """Display model statistics."""
    cli_banner("Model Statistics", "Parameter counts and layer breakdown", width=64)
    if not os.path.exists(CONFIG_PATH):
        pwarn("No config found — train a model first."); return
    cfg = load_run_config()
    vocab = load_or_make_vocab(cfg, cfg["dataset_path"])
    model = build_model(cfg, vocab.size)

    msel = cfg["model_selection"]
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen       = total_params - trainable

    W = 62
    cli_section("Model Overview", W)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Architecture   :')}  {_c(_WH, _B, MODEL_NAMES.get(msel, '?'))}")
    print(f"  │  {_c(_DIM, 'Embed dim      :')}  {_c(_WH, cfg['embed_dim'])}")
    print(f"  │  {_c(_DIM, 'Layers         :')}  {_c(_WH, cfg['layer_count'])}")
    print(f"  │  {_c(_DIM, 'Sequence len   :')}  {_c(_WH, cfg['seq_len'])}")
    print(f"  │  {_c(_DIM, 'Vocab size     :')}  {_c(_WH, vocab.size)}")
    print(f"  │  {_c(_DIM, 'Iterations done:')}  {_c(_WH, cfg.get('iterations_done', 0))}")
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Total params   :')}  {_c(_WH, _B, readable_num(total_params))}  {_c(_DIM, f'({total_params:,})')}")
    print(f"  │  {_c(_DIM, 'Trainable      :')}  {_c(_WH, readable_num(trainable))}  {_c(_DIM, f'({100*trainable/max(1,total_params):.1f}%)')}")
    if frozen:
        print(f"  │  {_c(_DIM, 'Frozen         :')}  {_c(_WH, readable_num(frozen))}")
    print(f"  │  {_c(_DIM, 'Size (fp32)    :')}  {_c(_WH, f'{total_params * 4 / 1024 / 1024:.1f} MB')}  {_c(_DIM, '(fp16 would be half that)')}")
    cli_section_end(W)

    layer_params = {}
    for name, param in model.named_parameters():
        top = name.split(".")[0]
        layer_params[top] = layer_params.get(top, 0) + param.numel()

    cli_section("Layer Breakdown", W)
    print(f"  │  {_c(_DIM, 'Grouped by top-level module name, sorted by parameter count.')}")
    print(f"  │")
    for name, count in sorted(layer_params.items(), key=lambda x: -x[1]):
        pct   = 100 * count / max(1, total_params)
        bar_w = int(pct / 2)            # 1 char per 2%
        bar   = _c(_CY, "█" * bar_w) + _c(_DIM, "░" * (50 - bar_w))
        print(f"  │  {_c(_WH, f'{name:<24}')} {_c(_DIM, readable_num(count)):>12}  {_c(_YL, f'{pct:5.1f}%')}  {bar}")
    cli_section_end(W)


def run_token_analysis():
    """Analyze token distribution in a dataset."""
    cli_banner("Token Analysis", "Vocabulary coverage, frequency, and entropy", width=64)

    cli_section("Settings", 64)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Dataset file to analyse. The first 10 MB will be read.')}")
    dataset_path = prompt_str("Dataset file path")
    if not os.path.exists(dataset_path):
        pwarn("File not found."); return

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Tokenizer — must match the one used during training for')}")
    print(f"  │  {_c(_DIM, 'the statistics to be meaningful.')}")
    cli_opt(-1, "Binary",   "Raw bytes as binary — 256 possible values")
    cli_opt( 0, "Byte",     "Byte-level 0–255 — universal, no text decoding needed")
    cli_opt( 1, "Char",     "Character-level — vocab from dataset characters")
    cli_opt( 2, "Word",     "Whitespace-split words")
    cli_opt( 3, "Tiktoken", "GPT-4 cl100k_base BPE (50k vocab)")
    cli_opt( 4, "BPE",      "Custom BPE trained on your data")
    print(f"  │")
    tokenizer_mode = prompt_int("Tokenizer", valid={-1,0,1,2,3,4})
    cfg_dummy = {"dataset_type": 0, "tokenizer_mode": tokenizer_mode, "custom_bpe_size": 4096}
    if tokenizer_mode == 3:
        cfg_dummy["tiktoken_encoding"] = prompt_str("Tiktoken encoding", default="cl100k_base")
    cli_section_end(64)

    vocab = load_or_make_vocab(cfg_dummy, dataset_path, save_config=False)

    max_read = 10 * 1024 * 1024
    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read(max_read)
    ids = vocab.encode(text)

    counter = collections.Counter(ids)
    unique = len(counter); total = len(ids)
    coverage = 100 * unique / max(1, vocab.size)
    compression = len(text.encode("utf-8")) / max(1, total * 2)

    W = 62
    cli_section("Statistics", W)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Total tokens  :')}  {_c(_WH, _B, readable_num(total))}")
    print(f"  │  {_c(_DIM, 'Unique tokens :')}  {_c(_WH, unique)}  {_c(_DIM, f'of {vocab.size} in vocab  ({coverage:.1f}% used)')}")
    print(f"  │  {_c(_DIM, 'Compression   :')}  {_c(_WH, f'{compression:.2f}x')}  {_c(_DIM, '(UTF-8 bytes / (tokens × 2)  — >1 = vocab saves space)')}")
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Top-N — show the most frequent token types.')}")
    print(f"  │  {_c(_DIM, 'High concentration in a few tokens = low entropy dataset.')}")
    top_n = prompt_int("Show top N tokens", default=20)
    cli_section_end(W)

    print()
    cli_section("Top Tokens", W)
    print(f"  │  {'Rank':<5}  {'Token':<32}  {'Count':>8}  {'Freq':>6}")
    cli_rule(W - 2)
    for rank, (tid, cnt) in enumerate(counter.most_common(top_n), 1):
        pct = 100 * cnt / total
        try: decoded = repr(vocab.decode([tid]))
        except: decoded = f"<id={tid}>"
        if len(decoded) > 30: decoded = decoded[:27] + "..."
        print(f"  │  {rank:<5}  {_c(_WH, f'{decoded:<32}')}  {cnt:>8,}  {_c(_YL, f'{pct:5.2f}%')}")
    cli_section_end(W)

    probs = np.array([c / total for c in counter.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    max_entropy = math.log2(unique)
    efficiency = 100 * entropy / max(1, max_entropy)

    print()
    cli_section("Entropy", W)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Shannon entropy  :')}  {_c(_WH, _B, f'{entropy:.4f}')} bits/token")
    print(f"  │  {_c(_DIM, 'Max possible     :')}  {_c(_WH, f'{max_entropy:.4f}')} bits/token  {_c(_DIM, '(uniform distribution)')}")
    print(f"  │  {_c(_DIM, 'Efficiency       :')}  {_c(_WH, f'{efficiency:.1f}%')}  {_c(_DIM, '(100% = perfectly uniform vocab usage)')}")
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Low entropy = vocabulary is dominated by a few tokens.')}")
    print(f"  │  {_c(_DIM, 'High entropy = tokens are spread more evenly.')}")
    cli_section_end(W)


def run_export():
    """Export model."""
    cli_banner("Export", "Save model in various portable formats", width=64)
    if not os.path.exists(CONFIG_PATH):
        pwarn("No config found — train a model first."); return
    cfg = load_run_config()
    vocab = load_or_make_vocab(cfg, cfg["dataset_path"])
    model = build_model(cfg, vocab.size)
    model.to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    cli_section("Export Format", 64)
    print(f"  │")
    cli_opt(0, "TorchScript (.pt)",    "Serialised traced graph — runs without Python source")
    cli_opt(1, "ONNX (.onnx)",         "Open standard — compatible with TensorRT, ONNX Runtime, etc.")
    cli_opt(2, "State dict (.pth)",    "Raw weight dictionary + JSON config — easiest to reload")
    cli_opt(3, "Quantised int8 (.pth)","Weights quantised to 8-bit — ~4× smaller file, slight quality loss")
    print(f"  │")
    cli_section_end(64)
    choice = prompt_int("Export format", valid={0,1,2,3})
    out_name = prompt_str("Output filename  (no extension)", default="exported_model")
    
    if choice == 2:
        torch.save(model.state_dict(), f"{out_name}.pth")
        save_json(f"{out_name}_config.json", cfg)
        print(f"Saved {out_name}.pth + config")
    elif choice == 3:
        sd = model.state_dict()
        q = {}
        for k, v in sd.items():
            if v.dtype in (torch.float32, torch.float16) and v.numel() > 100:
                scale = v.abs().max() / 127.0
                q[k] = {"data": (v / scale).to(torch.int8), "scale": scale}
            else: q[k] = v
        torch.save(q, f"{out_name}_int8.pth")
        orig = os.path.getsize(CHECKPOINT_PATH) / 1024 / 1024
        new = os.path.getsize(f"{out_name}_int8.pth") / 1024 / 1024
        print(f"Saved {out_name}_int8.pth ({orig:.1f}MB -> {new:.1f}MB)")
    elif choice == 0:
        try:
            dummy = torch.randint(0, vocab.size, (1, cfg["seq_len"]), device=DEVICE)
            if cfg["model_selection"] in RNN_MODEL_IDS and not is_bottom_up_megabyte(cfg):
                scripted = torch.jit.trace(model, (dummy, None))
            else:
                scripted = torch.jit.trace(model, dummy)
            scripted.save(f"{out_name}.pt")
            print(f"Saved TorchScript to {out_name}.pt")
        except Exception as e:
            print(f"TorchScript failed: {e}")
    elif choice == 1:
        try:
            dummy = torch.randint(0, vocab.size, (1, cfg["seq_len"]), device=DEVICE)
            torch.onnx.export(model, dummy, f"{out_name}.onnx",
                input_names=["input_ids"], output_names=["logits"])
            print(f"Saved ONNX to {out_name}.onnx")
        except Exception as e:
            print(f"ONNX failed: {e}")


def run_hyperparam_sweep():
    """Simple hyperparameter search."""
    cli_banner("Hyperparameter Sweep", "Grid search across embed / layers / lr / batch", width=64)

    cli_section("Dataset & Tokenizer", 64)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'All configs in the sweep share the same dataset and tokenizer.')}")
    dataset_path   = prompt_str("Dataset file path")
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Dataset type — 0 = sliding-window corpus, 1 = one example per line.')}")
    dataset_type   = prompt_int("Dataset type  (0=standard  1=line)", valid={0,1})
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Tokenizer — must be consistent across the whole sweep.')}")
    cli_opt(-1,"Binary"); cli_opt(0,"Byte"); cli_opt(1,"Char")
    cli_opt(2,"Word");    cli_opt(3,"Tiktoken"); cli_opt(4,"BPE")
    print(f"  │")
    tokenizer_mode = prompt_int("Tokenizer", valid={-1,0,1,2,3,4})
    cli_section_end(64)

    cfg_dummy = {"dataset_type": dataset_type, "tokenizer_mode": tokenizer_mode, "custom_bpe_size": 4096}
    if tokenizer_mode == 3: cfg_dummy["tiktoken_encoding"] = "cl100k_base"
    vocab = load_or_make_vocab(cfg_dummy, dataset_path, save_config=False)

    print_model_menu()
    msel = prompt_int("Model #", valid=MODEL_IDS)

    cli_section("Sweep Grid", 64)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Enter comma-separated values for each axis.')}")
    print(f"  │  {_c(_DIM, 'Every combination is tried — 2 dims × 2 lrs × 2 batches = 8 runs.')}")
    print(f"  │")

    if dataset_type == 0:
        print(f"  │  {_c(_DIM, 'Sequence length — fixed for all configs in this sweep.')}")
        seq_len = prompt_int("Sequence length", default=128)
    else:
        seq_len = 0

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Iterations per run — gradient steps each config trains for.')}")
    print(f"  │  {_c(_DIM, 'Keep low (200–1000) so the sweep completes in reasonable time.')}")
    iters_per = prompt_int("Iterations per run", default=500)

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Embedding dimensions to test  (comma-separated):')}")
    embed_dims   = [int(x) for x in prompt_str("Embed dims", default="256").split(",")]
    print(f"  │  {_c(_DIM, 'Layer counts to test  (comma-separated):')}")
    layer_counts = [int(x) for x in prompt_str("Layer counts", default="4").split(",")]
    print(f"  │  {_c(_DIM, 'Learning rates to test  (comma-separated, e.g. 1e-3,5e-4):')}")
    lrs          = [float(x) for x in prompt_str("Learning rates", default="1e-3").split(",")]
    print(f"  │  {_c(_DIM, 'Batch sizes to test  (comma-separated):')}")
    batch_sizes  = [int(x) for x in prompt_str("Batch sizes", default="32").split(",")]
    cli_section_end(64)

    cfg_data = {"dataset_path": dataset_path, "dataset_type": dataset_type, "seq_len": seq_len,
                "vocab_tokens": getattr(vocab, "tokens", None), "val_split": 0.1, "valid_examples": 0,
                "line_seq_len_cap": None}
    train_ds, valid_ds = build_datasets(cfg_data, vocab)
    if dataset_type == 1:
        cfg_data["seq_len"] = train_ds.max_len
        seq_len = cfg_data["seq_len"]

    configs = [{"embed_dim": e, "layer_count": l, "learning_rate": lr, "batch_size": b}
               for e in embed_dims for l in layer_counts for lr in lrs for b in batch_sizes]

    results = []
    print(f"\n  {_c(_GR, _B, '▸')} Running {_c(_WH, len(configs))} configurations…\n")
    for idx, hp in enumerate(configs, 1):
        cfg = cfg_data.copy()
        cfg.update({"model_selection": msel, "head_count": 4, "activation_name": "gelu",
                    "dropout": 0.0, "tokenizer_mode": tokenizer_mode, "seq_len": seq_len, **hp})
        desc = f"e={hp['embed_dim']} L={hp['layer_count']} lr={hp['learning_rate']:.0e} bs={hp['batch_size']}"
        try:
            model = build_model(cfg, vocab.size); model.to(DEVICE)
            opt   = build_optimizer(model, cfg)
            score, _, status = bench_train_loop(cfg, model, opt, train_ds, valid_ds, vocab,
                                                (dataset_type==1), iters_per, 0, True)
            results.append({"hp": hp, "score": score, "status": status})
            pok(f"[{idx}/{len(configs)}]  {_c(_WH, desc)}  →  {_c(_GR, f'{score:.5f}')}  {_c(_DIM, status)}")
            del model, opt; torch.cuda.empty_cache()
        except Exception as e:
            results.append({"hp": hp, "score": float("inf"), "status": str(e)})
            pwarn(f"[{idx}/{len(configs)}]  {_c(_WH, desc)}  →  {_c(_RD, 'CRASH:')} {e}")

    results.sort(key=lambda x: x["score"])
    W = 72
    print()
    cli_section("Sweep Results", W)
    hdr = f"  {'#':<4}  {'embed':>5}  {'layers':>6}  {'lr':>8}  {'batch':>5}  {'score':>10}  status"
    print(f"  │{_c(_DIM, hdr)}")
    cli_rule(W - 2)
    for i, r in enumerate(results, 1):
        hp      = r["hp"]
        score_s = f"{r['score']:.5f}" if r["score"] != float("inf") else "∞"
        score_c = _c(_GR, _B, score_s) if i == 1 else (_c(_RD, score_s) if r["score"] == float("inf") else score_s)
        rank_c  = _c(_YL, _B, f"{i:<4}") if i == 1 else f"{i:<4}"
        print(f"  │  {rank_c}  {hp['embed_dim']:>5}  {hp['layer_count']:>6}  {hp['learning_rate']:>8.1e}  {hp['batch_size']:>5}  {score_c:>10}  {_c(_DIM, r['status'])}")
    cli_section_end(W)
    if results and results[0]["score"] != float("inf"):
        b = results[0]["hp"]
        blr = f"{b['learning_rate']:.2e}"
        print(f"\n  {_c(_GR, '★')} Best config: embed={_c(_WH,b['embed_dim'])} layers={_c(_WH,b['layer_count'])} lr={_c(_WH,blr)} batch={_c(_WH,b['batch_size'])}\n")


def speed_test_model(cfg_t, vocab, warmup, measure):
    """Time forward + backward + optimizer steps on random tokens for one model.

    ``cfg_t`` carries model_selection, seq_len and batch_size.  Returns tok_s,
    params, ms_per_step and (on CUDA) peak_mem in bytes; raises on failure.
    """
    msel, seq_len, batch_size = cfg_t["model_selection"], cfg_t["seq_len"], cfg_t["batch_size"]
    model = opt = None
    try:
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = build_model(cfg_t, vocab.size); model.to(DEVICE); model.train()
        total_p = sum(p.numel() for p in model.parameters())
        opt = build_optimizer(model, cfg_t)
        dx = torch.randint(0, vocab.size, (batch_size, seq_len), device=DEVICE)
        dy = torch.randint(0, vocab.size, (batch_size, seq_len), device=DEVICE)
        crit = nn.CrossEntropyLoss()

        def step():
            if msel in RNN_MODEL_IDS and not is_bottom_up_megabyte(cfg_t): lg = model(dx, None)[0]
            elif msel in SCAN_MODEL_IDS: out = model(dx); lg = out[0] if isinstance(out, tuple) else out
            else:
                out = model(dx)
                lg = out[0] if isinstance(out, tuple) else out
            crit(lg.reshape(-1, lg.size(-1)), dy.reshape(-1)).backward(); opt.step(); opt.zero_grad(set_to_none=True)
        for _ in range(warmup):
            step()
        if DEVICE == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(measure):
            step()
        if DEVICE == "cuda": torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return {"tok_s": measure * batch_size * seq_len / elapsed, "params": total_p,
                "ms_per_step": 1000 * elapsed / max(1, measure),
                "peak_mem": torch.cuda.max_memory_allocated() if DEVICE == "cuda" else None}
    finally:
        del model, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_speed_benchmark():
    """Measure throughput (tokens/second) for models."""
    cli_banner("Speed Benchmark", "Forward + backward pass throughput in tokens / second", width=64)
    if not os.path.exists(CONFIG_PATH):
        pwarn("No config found — train a model first."); return
    cfg = load_run_config()
    vocab = load_or_make_vocab(cfg, cfg["dataset_path"])

    cli_section("Settings", 64)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Sequence length and batch size determine the tokens-per-step.')}")
    print(f"  │  {_c(_DIM, 'Use the same values you train with for a realistic comparison.')}")
    seq_len    = prompt_int("Sequence length", default=cfg["seq_len"])
    batch_size = prompt_int("Batch size",      default=cfg["batch_size"])
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Warmup steps — run these steps first to let CUDA / JIT settle.')}")
    print(f"  │  {_c(_DIM, 'Their time is discarded. 10 is usually enough.')}")
    warmup     = prompt_int("Warmup steps",    default=10)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Measure steps — the steps actually timed and averaged.')}")
    print(f"  │  {_c(_DIM, 'More steps = more stable result. 50 is a good default.')}")
    measure    = prompt_int("Measure steps",   default=50)
    cli_section_end(64)

    print_model_menu()
    speed_hint = "Enter comma-separated IDs, or 'all' to test every architecture."
    print(f"  {_c(_DIM, speed_hint)}\n")
    raw = prompt_str("Model IDs  (comma-separated or 'all')", default="all")
    if raw.lower() == "all":
        model_ids = sorted(MODEL_NAMES.keys())
    else:
        model_ids = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]

    results = []
    for msel in model_ids:
        cfg_t = cfg.copy()
        cfg_t["model_selection"] = msel
        cfg_t["seq_len"]         = seq_len
        cfg_t["batch_size"]      = batch_size
        name = MODEL_NAMES.get(msel, f"Model {msel}")
        try:
            r = speed_test_model(cfg_t, vocab, warmup, measure)
            tps, total_p = r["tok_s"], r["params"]
            results.append({"name": name, "tok_s": tps, "params": total_p})
            pok(f"{_c(_WH, f'{name:<42}')} {_c(_GR, _B, f'{tps:>10,.0f}')} tok/s  {_c(_DIM, readable_num(total_p) + ' params')}")
        except Exception as e:
            pwarn(f"{_c(_WH, f'{name:<42}')} {_c(_RD, 'FAILED:')} {e}")
            results.append({"name": name, "tok_s": 0, "params": 0})

    results.sort(key=lambda x: -x["tok_s"])
    W = 70
    print()
    cli_section("Speed Ranking  (fastest first)", W)
    hdr = f"  {'#':<4}  {'Model':<42}  {'tok/s':>12}  Params"
    print(f"  │{_c(_DIM, hdr)}")
    cli_rule(W - 2)
    for i, r in enumerate(results, 1):
        tps_s = f"{r['tok_s']:>12,.0f}" if r["tok_s"] else f"{'FAILED':>12}"
        tps_c = _c(_GR, _B, tps_s) if i == 1 else (_c(_RD, tps_s) if not r["tok_s"] else tps_s)
        rank_c = _c(_YL, _B, f"{i:<4}") if i == 1 else f"{i:<4}"
        print(f"  │  {rank_c}  {r['name']:<42}  {tps_c}  {_c(_DIM, readable_num(r['params']))}")
    cli_section_end(W)


def interactive_train():
    cli_banner("LineGen", "Neural Text Generation Framework", width=64)

    W = 62
    print(f"  {_c(_CY, _B, '┌─')} {_c(_WH, _B, 'Modes')} {_c(_CY, '─' * (W - 12) + '┐')}")
    cli_opt("0 / t", "Train",        "Train a new model or resume from checkpoint")
    cli_opt("1 / s", "Sample",       "Generate text from a saved checkpoint")
    cli_opt("2 / b", "Benchmark",    "Compare multiple architectures head-to-head")
    cli_opt("3 / c", "Chat",         "Interactive multi-turn generation session")
    cli_opt("4 / p", "Perplexity",   "Evaluate model perplexity on a text file")
    cli_opt("5 / m", "Stats",        "Print parameter counts and layer breakdown")
    cli_opt("6 / a", "Tokens",       "Analyse tokenization of a text file")
    cli_opt("7 / e", "Export",       "Export to TorchScript / ONNX / quantized int8")
    cli_opt("8 / h", "Sweep",        "Hyperparameter grid search over a single model")
    cli_opt("9 / v", "Speed",        "Measure throughput in tokens / second")
    cli_opt("g",     "GUI",          "Browser interface: train, sample, inspect activations")
    cli_blank_row()
    print(f"  {_c(_CY, '└' + '─' * (W - 2) + '┘')}\n")

    cont = prompt_str("Selection").lower().strip()
    
    if cont in ("gui", "g"):
        from linegen_gui import run_gui
        run_gui(port=prompt_int("Port", default=8767))
    elif cont in ("train","t","0"):
        # Training Mode
        resume = False
        retry = False
        if os.path.exists(CONFIG_PATH) and os.path.exists(CHECKPOINT_PATH):
            choice = prompt_str("Resume previous run?  (y = resume, r = retry config with fresh weights, n = new): ").lower()
            if choice == "y":
                resume = True
            elif choice == "r":
                retry = True

        if resume or retry:
            cfg = load_run_config()
            if retry:
                cfg = retry_adjustments(cfg)
            # 1. Load Vocab & Save it immediately to prevent sync issues
            vocab = load_or_make_vocab(cfg, cfg["dataset_path"])
            cfg["vocab_tokens"] = getattr(vocab, "tokens", None)
            save_json(CONFIG_PATH, cfg) 
            
            dataset, valid = build_datasets(cfg, vocab)
            model = build_model(cfg, vocab.size)
            model.to(DEVICE)
            #if torch.__version__.startswith("2."):
            #    model = wrap_model_with_compile(model, cfg)
            
            if resume:
                print(f"[Resume] Loading checkpoint {CHECKPOINT_PATH}...")
                model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
            else:
                print("[Retry] Starting fresh model weights from the saved configuration.")
        else:
            cfg = build_config_new()
            # 1. Load Vocab & Save it immediately
            vocab = load_or_make_vocab(cfg, cfg["dataset_path"])
            cfg["vocab_tokens"] = getattr(vocab, "tokens", None)
            save_json(CONFIG_PATH, cfg)

            dataset, valid = build_datasets(cfg, vocab)
            if cfg.get("target_params"):
                # The vocabulary and (line mode) window are known only now.
                dim, fitted = fit_width_to_params(cfg, vocab.size, cfg["target_params"], cfg.get("head_count", 4))
                cfg["embed_dim"] = dim
                pinfo(f"Width {dim} gives {readable_num(fitted or 0)} parameters "
                      f"(target {readable_num(cfg['target_params'])})")
                save_json(CONFIG_PATH, cfg)
            model = build_model(cfg, vocab.size)
            model.to(DEVICE)
            #if torch.__version__.startswith("2."):
            #    model = wrap_model_with_compile(model, cfg)
        
        model = wrap_model_with_compile(model, cfg)
        # Print model stats
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n  Model: {MODEL_NAMES.get(cfg['model_selection'], '?')} | Params: {readable_num(total_params)} | Vocab: {vocab.size}")
        
        model.to(DEVICE)
        opt = build_optimizer(model, cfg)
        
        train_loop(cfg, model, opt, dataset, valid, vocab, cfg["dataset_type"]==1)

    elif cont in ("sample","s","1"):
        # Sampling Mode
        if not os.path.exists(CONFIG_PATH):
            print("No config found. Train first.")
            return

        cfg = load_run_config()
        vocab = load_or_make_vocab(cfg, cfg["dataset_path"])
        
        # Build model with vocab size from config (now correctly reconstructed)
        model = build_model(cfg, vocab.size)
        model.to(DEVICE)
        #if torch.__version__.startswith("2."):
        #    model = wrap_model_with_compile(model, cfg)
        
        print(f"[Sample] Loading {CHECKPOINT_PATH}...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        model = wrap_model_with_compile(model, cfg)
        model.to(DEVICE)
        run_sampling_ui(cfg, model, vocab)

    elif cont in ("benchmark","b","2"):
        run_benchmark()
    
    elif cont in ("chat","c","3"):
        if not os.path.exists(CONFIG_PATH):
            print("No config found. Train first."); return
        cfg = load_run_config()
        vocab = load_or_make_vocab(cfg, cfg["dataset_path"])
        model = build_model(cfg, vocab.size); model.to(DEVICE)
        print(f"Loading {CHECKPOINT_PATH}...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        run_interactive_chat(cfg, model, vocab)
    
    elif cont in ("perplexity","p","4"):
        run_perplexity_eval()
    
    elif cont in ("stats","m","5"):
        run_model_stats()
    
    elif cont in ("tokens","a","6"):
        run_token_analysis()
    
    elif cont in ("export","e","7"):
        run_export()
    
    elif cont in ("sweep","h","8"):
        run_hyperparam_sweep()
    
    elif cont in ("speed","v","9"):
        run_speed_benchmark()
    
    else:
        print("Unknown choice.")


def get_prompt_batch(cfg, vocab: CharVocab):
    line_mode = (cfg["dataset_type"]==1)
    cli_section("Prompt", 64)
    print(f"  │  {_c(_DIM, 'The prompt seeds generation — the model continues from where it left off.')}")
    print(f"  │")
    cli_opt(0, "Random token",  "Pick a random token from the vocabulary as the seed")
    cli_opt(1, "'BEGIN'",       "Use the literal text 'BEGIN' as the prompt")
    cli_opt(2, "File",          "Load text prompts, or raw bytes / hex from a file in byte mode")
    cli_opt(3, "Custom",        "Type a prompt (hex in byte mode; use File for long byte prompts)")
    print(f"  │")
    cli_section_end(64)
    smode = prompt_int("Prompt mode", valid={0,1,2,3})
    prompts: List[str] = []
    if smode == 0:
        prompts = [BOS_TOKEN] if line_mode else [random.choice(vocab.tokens)]
    elif smode == 1:
        prompts = ["BEGIN"]
        if line_mode: prompts = [BOS_TOKEN + prompts[0]]
    elif smode == 2:
        f = prompt_str("Path to prompt file")
        if not os.path.exists(f): pwarn("File not found."); return []
        byte_mode = int(cfg.get("tokenizer_mode", 1)) in {-1, 0}
        if byte_mode and not line_mode:
            raw = pathlib.Path(f).read_bytes()
            # A file containing only ASCII hex is a convenient alternative to
            # pasting a long prompt; otherwise keep raw binary bytes untouched.
            try:
                text = raw.decode("ascii").strip()
            except UnicodeDecodeError:
                return [raw]
            if text.lower().startswith("hex:"):
                text = text[4:].lstrip()
            if HEX_RE.fullmatch(text):
                try:
                    return [_hex_to_bytes(text)]
                except ValueError as exc:
                    pwarn(f"Invalid hex prompt file: {exc}.")
                    return []
            return [raw]
        if line_mode:
            with open(f,"r",encoding="utf-8") as r: prompts = [BOS_TOKEN + ln.rstrip("\n") for ln in r.readlines()]
        else:
            with open(f,"r",encoding="utf-8") as r: prompts = [r.read()]
    else:
        p = prompt_str("Custom prompt")
        if not line_mode and int(cfg.get("tokenizer_mode", 1)) in {-1, 0}:
            try:
                return [_hex_to_bytes(p)]
            except ValueError as exc:
                pwarn(
                    f"Invalid hex prompt: {exc}. For long byte prompts, use "
                    "Prompt mode 2 with a raw .bmp or ASCII .hex file."
                )
                return []
        prompts = [BOS_TOKEN + p] if line_mode else [p]
    return prompts

def _visible_decode_prompt(vocab, ids: List[int], line_mode: bool) -> str:
    """Decode but hide BOS if line mode."""
    if hasattr(vocab, "bos_id") and line_mode and vocab.bos_id is not None:
        ids = [i for i in ids if i != vocab.bos_id]
    return vocab.decode(ids)

def run_sampling_ui(cfg, model, vocab):
    model.eval()
    if int(cfg.get("tokenizer_mode", 1)) in {-1, 0}:
        ensure_filegen_clean()

    cli_banner("Sample", "Generate text from the trained model", width=64)
    cli_section("Generation Settings", 64)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'How many independent samples to generate in this run.')}")
    count = prompt_int("Number of samples", default=1)

    line_mode = (cfg["dataset_type"]==1)
    byte_text = bool(cfg.get("byte_output_text", False))
    if not line_mode:
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Maximum tokens to generate per sample.')}")
        print(f"  │  {_c(_DIM, 'In line mode this is set automatically from the model config.')}")
        max_len = prompt_int("Max generated tokens", default=200)
    else:
        max_len = cfg["seq_len"]

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Temperature — scales the logits before sampling.')}")
    print(f"  │  {_c(_DIM, '  0   = greedy (always picks the top token, fully deterministic)')}")
    print(f"  │  {_c(_DIM, '  1.0 = unmodified distribution')}")
    print(f"  │  {_c(_DIM, '  >1  = more random / creative,  <1 = more focused / repetitive')}")
    temp = prompt_float("Temperature  (0 = greedy)", default=cfg.get("temperature", 1.0))
    cfg["temperature"] = temp

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Top-k filtering — keep only the k most likely tokens at each step.')}")
    print(f"  │  {_c(_DIM, 'Prevents sampling very unlikely tokens. 0 = disabled.')}")
    print(f"  │  {_c(_DIM, 'Typical values: 20–200.')}")
    top_k = prompt_int("Top-k  (0 = off)", default=0)

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Top-p (nucleus) — keep the smallest set of tokens whose cumulative')}")
    print(f"  │  {_c(_DIM, 'probability exceeds p, then sample from that set only.')}")
    print(f"  │  {_c(_DIM, 'Adapts dynamically to confidence. 0.0 = disabled. Typical: 0.9.')}")
    top_p = prompt_float("Top-p  (0.0 = off)", default=0.0)

    print(f"  │")
    print(f"  │  {_c(_DIM, 'Repetition penalty — divides logits of recently seen tokens,')}")
    print(f"  │  {_c(_DIM, 'making the model less likely to repeat itself. 1.0 = off.')}")
    print(f"  │  {_c(_DIM, 'Values 1.1–1.3 reduce loops without distorting output much.')}")
    rep_penalty = prompt_float("Repetition penalty  (1.0 = off)", default=1.0)
    cfg["_top_k"] = top_k
    cfg["_top_p"] = top_p
    cfg["_rep_penalty"] = rep_penalty

    want_capture = False
    if isinstance(model, BuiltinRNNWrapper):
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Activation capture — saves per-timestep hidden state vectors to')}")
        print(f"  │  {_c(_DIM, 'binary .bin files in FileGen/ for offline visualisation.')}")
        yn = prompt_str("Record per-timestep activations?  (y/n)", default="n").lower()
        want_capture = yn in ("y", "yes", "1")
    cli_section_end(64)

    seq2seq = cfg.get("seq2seq") if line_mode else None
    if seq2seq:
        inputs = ", ".join(seq2seq["columns"][c] for c in seq2seq["input_cols"])
        outputs_names = ", ".join(seq2seq["columns"][c] for c in seq2seq["output_cols"])
        pinfo(f"Seq2seq: give the input columns ({inputs}) separated by {seq2seq['delimiter']!r}; "
              f"the model writes {outputs_names}.")
    prompts = get_prompt_batch(cfg, vocab)
    if not prompts: return

    outputs = []
    tmode = int(cfg.get("tokenizer_mode",1))
    out_dir = pathlib.Path("FileGen")

    for i in range(count):
        prompt = prompts[i % len(prompts)]
        if seq2seq:
            try:
                p_ids = encode_seq2seq_prompt(vocab, seq2seq, prompt[len(BOS_TOKEN):]
                                              if prompt.startswith(BOS_TOKEN) else prompt)
            except ValueError as exc:
                pwarn(str(exc))
                continue
        else:
            p_ids = vocab.encode(prompt)

        if tmode in {-1, 0}:
            prompt_bytes = vocab.to_bytes(p_ids) if hasattr(vocab, "to_bytes") else bytes()
            print(bold(f"--- PROMPT (hex) ---\n{prompt_bytes.hex()}"))
        else:
            vis = _visible_decode_prompt(vocab, p_ids, line_mode)
            print(bold(f"--- PROMPT ---\n{seq2seq_visible(vis) if seq2seq else vis}"))

        # ===== NEW: begin capture (builtins only) =====
        if want_capture:
            model.start_capture()

        if line_mode:
            # Force stream=False if capturing, so we have the ids for saving
            out_ids = generate_line_mode(model, cfg, vocab, p_ids, limit_len=max_len)
            # Trim leading BOS for readable text output
            if tmode in {-1, 0}:
                data = vocab.to_bytes(out_ids) if hasattr(vocab, "to_bytes") else bytes()
                if byte_text:
                    print(f"--- sample {i+1} ---\n{vocab.decode(out_ids[1:]) if len(out_ids)>1 else ''}")
                else:
                    (out_dir / f"sample_{i+1:03d}").write_bytes(data)
                    print(f"[saved] FileGen/sample_{i+1:03d} ({len(data)} bytes)")
            else:
                text = vocab.decode(out_ids[1:]) if len(out_ids)>1 else ""
                outputs.append(seq2seq_visible(text) if seq2seq else text)
        else:
            # Classic
            if tmode in {-1, 0}:
                out_ids = generate_classic(model, cfg, vocab, p_ids, max_len=max_len, stream=False)
                if byte_text:
                    print(f"--- sample {i+1} ---\n{vocab.decode(out_ids)}")
                else:
                    data = vocab.to_bytes(out_ids) if hasattr(vocab, "to_bytes") else bytes()
                    (out_dir / f"sample_{i+1:03d}").write_bytes(data)
                    print(f"[saved] FileGen/sample_{i+1:03d} ({len(data)} bytes)")
            else:
                out_ids = generate_classic(model, cfg, vocab, p_ids, max_len=max_len, stream=True)
                sys.stdout.write(f"--- sample {i+1} ---\n")
                sys.stdout.write(vocab.decode(out_ids))
                sys.stdout.write("\n")

        # ===== NEW: finish + dump capture =====
        if want_capture:
            model.stop_capture()
            cap = model.get_captured()  # list of [1, S, H_l] or None

            # Build per-step token strings aligned to S steps (exclude prompt)
            # For both classic & line modes, sampling loops generate exactly K new tokens.
            # In out_ids, the number of newly generated tokens is:
            #   - classic: len(out_ids) - len(p_ids)
            #   - line:    len(out_ids) - len(p_ids)
            # Line generation removes the tokenizer-added terminal EOS before
            # priming, so its returned prefix is one token shorter than p_ids.
            prompt_len = len(p_ids)
            if line_mode and getattr(vocab, "eos_id", None) is not None and p_ids and p_ids[-1] == vocab.eos_id:
                prompt_len -= 1
            gen_only = out_ids[prompt_len:]
            # visible per-step tokens (hide BOS visually in line mode)
            step_tokens = []
            for tok_id in gen_only:
                if line_mode and hasattr(vocab, "bos_id") and tok_id == getattr(vocab, "bos_id", None):
                    step_tokens.append("")  # hide BOS
                else:
                    step_tokens.append(vocab.decode([tok_id]))

            # Ensure FileGen/ exists
            out_dir.mkdir(parents=True, exist_ok=True)
            bin_path = out_dir / f"activations_{i+1:03d}.bin"
            save_activation_capture_bin(str(bin_path), step_tokens, cap)
            print(f"[saved] {bin_path} (activations + tokens)")


    if line_mode and tmode != 0:
        print("==== BATCH OUTPUTS ====")
        for i,o in enumerate(outputs, 1):
            print(f"--- sample {i} ---\n{o}\n")


def train_for_iterations(cfg, model, optimizer, dataset, valid_ds, vocab, line_mode, iters_total, loss_window=100):
    criterion = nn.CrossEntropyLoss()
    model.train()
    iters = 0
    losses = []
    loss_window = max(10, int(cfg.get("log_interval", 1)))

    # simple minibatch fetcher (no TBPTT in benchmarks)
    def fetch():
        return dataset.get_batch(cfg["batch_size"])

    # Build a nice tqdm description
    name = None
    try:
        name = MODEL_NAMES.get(cfg["model_selection"], None)  # optional, if available
    except Exception:
        pass
    model_sel = cfg.get("model_selection", "?")
    desc = f"[bench] {name if name else f'model {model_sel}'}"


    msel = cfg["model_selection"]
    with tqdm(total=iters_total, desc=desc, ncols=100, leave=False) as pbar:
        while iters < iters_total:
            x, y = fetch()
            if msel in RNN_MODEL_IDS and not is_bottom_up_megabyte(cfg):
                logits = model(x, None)[0]
            elif msel in SCAN_MODEL_IDS:
                out = model(x); logits = out[0] if isinstance(out, tuple) else out
            else:
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                iters += 1; pbar.update(1)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if hasattr(optimizer, "observe_loss"):   # HD optimizers' divergence guard
                optimizer.observe_loss(loss.item())
            optimizer.step()

            # track losses
            l = float(loss.item())
            losses.append(l)
            if len(losses) > loss_window:
                losses.pop(0)

            # update progress bar with current + rolling avg loss
            avg = sum(losses) / max(1, len(losses))
            metrics = loss_metrics(avg, 1, cfg)
            pbar.set_postfix(current=f"{l:.5f}", loss=f"{avg:.5f}", metric=format_loss_metrics(metrics))
            pbar.update(1)

            iters += 1

    # Decide score
    use_valid = False
    if line_mode and valid_ds is not None and cfg.get("valid_examples", 0) > 1000:
        use_valid = True
    if (not line_mode) and valid_ds is not None and bool(cfg.get("classic_val_path", "")):
        use_valid = True

    if use_valid:
        vloss = eval_valid_loss(model, cfg, valid_ds, vocab, line_mode=line_mode, max_samples=1000)
        return float(vloss) if vloss is not None else (sum(losses) / max(1, len(losses)))
    else:
        return sum(losses) / max(1, len(losses))

# ==============================================================================
#  MODEL NAMES MAPPING
# ==============================================================================
# ==============================================================================
#  MODEL DEFINITIONS (User Spec)
# ==============================================================================

# Every selectable model.  IDs come in blocks of 100, one block per family
# (ID // 100 indexes MODEL_GROUPS), and run chronologically inside a block by
# MODEL_ORIGINS, so a newer model is appended to the end of its family
# without renumbering anything else.
MODEL_GROUPS = {
    0: "Classical baselines & MLPs",
    1: "Convolutional",
    2: "MLP & token mixers",
    3: "Transformers  (full attention)",
    4: "Sparse, long-context & memory Transformers",
    5: "Gated & vanilla RNNs  (sequential, stateful)",
    6: "Continuous-time & oscillator RNNs",
    7: "Latent-variable RNNs",
    8: "xLSTM",
    9: "Parallel-trained RNNs  (scan / Newton)",
    10: "State-space models",
    11: "Linear attention & gated linear RNNs",
    12: "Hybrids  (attention + recurrence)",
}

# (id, full name, menu label, menu description), ordered by ID.
MODEL_REGISTRY = (
    # ==== Classical baselines & MLPs ====
    (0, 'Markov bigram (trainable transition baseline)', 'Markov bigram', 'Trainable first-order token transition baseline'),
    (1, 'MLP (one-hot encoding window instead of embeddings, Mish activation)', 'MLP (one-hot window)', 'One-hot context window → feedforward, no embedding'),
    (2, 'Neural n-gram MLP (Bengio et al., flat bounded context)', 'Neural n-gram', 'Flat bounded previous-token context fed to an MLP'),
    (3, 'NADE (causal autoregressive density estimator)', 'NADE', 'Neural autoregressive density-estimation MLP'),
    (4, 'MADE (causal masked autoregressive estimator)', 'MADE', 'Masked autoencoder distribution-estimation MLP'),
    (5, 'Residual MLP (Transformer MLP blocks, no attention, Mish activation)', 'MLP (residual)', 'Transformer-style FF blocks, no attention or recurrence'),
    # ==== Convolutional ====
    (100, 'PixelCNN-style (causal masked convolution)', 'PixelCNN-style', 'Masked causal-convolution LM for token sequences'),
    (101, 'WaveNet (causal gated dilated convolution)', 'WaveNet', 'Gated causal dilated-convolution LM'),
    (102, 'Temporal ConvNet (causal dilated TCN, Mish activation)', 'Temporal ConvNet', 'Causal dilated 1-D TCN'),
    (103, 'Causal ConvNeXt-1D (large-kernel depthwise convolution)', 'Causal ConvNeXt-1D', 'Modern large-kernel causal depthwise convolution'),
    (104, 'Hyena (gated FFT long convolution)', 'Hyena', 'Gated FFT causal long-convolution LM'),
    # ==== MLP & token mixers ====
    (200, 'gMLP (causal spatial gating unit)', 'gMLP', 'Gated MLP with spatial gating unit (causal)'),
    (201, 'aMLP (causal gMLP with tiny attention)', 'aMLP', 'gMLP + tiny self-attention gate'),
    (202, 'MLP-Mixer (causal)', 'MLP-Mixer (causal)', 'Patch-style MLP mixer adapted for sequences'),
    (203, 'CCS token-mixing MLP (causal circulant)', 'CCS-MLP', 'Circulant channel-specific causal MLP'),
    (204, 'WaveMLP (causal phase-modulated mixer)', 'WaveMLP', 'Phase-modulated causal MLP mixer'),
    (205, 'DynaMixer (causal content-gated mixer)', 'DynaMixer', 'Content-gated causal MLP mixer'),
    (206, 'pNLP-Mixer (causal)', 'pNLP-Mixer', 'All-MLP NLP token/channel mixer (causal)'),
    (207, 'HyperMixer (hypernetwork token mixing)', 'HyperMixer', 'MLP-Mixer variant with hypernetwork token mixing'),
    (208, 'Toeplitz MLP Mixer (causal, experimental)', 'Toeplitz MLP Mixer', 'Experimental global causal FFT Toeplitz mixer'),
    (209, 'Grassmann-flow mixer (causal, experimental)', 'Grassmann mixer', 'Experimental causal local geometric pair mixer'),
    # ==== Transformers  (full attention) ====
    (300, 'Original Transformer (Vaswani et al. 2017, decoder-only)', 'Original Transformer', 'Vaswani 2017 decoder: post-LN, sinusoidal, ReLU FFN'),
    (301, 'GPT-2 decoder-only Transformer', 'GPT-2 Transformer', 'Decoder-only pre-LN transformer, learned positions'),
    (302, 'Switch Transformer (top-1 sparse MoE)', 'Switch Transformer', 'Top-1 routed sparse mixture-of-experts FFN'),
    (303, 'DCT-Former (spectral attention)', 'DCT-Former', 'DCT-based spectral attention transformer'),
    (304, 'Llama-3 style Transformer (RMSNorm, SwiGLU, RoPE)', 'Llama-3 Transformer', 'RMSNorm, SwiGLU, RoPE (Llama-3 style)'),
    (305, 'KAN-Transformer (Chebyshev KAN feed-forward)', 'KAN-Transformer', 'Transformer with Chebyshev KAN feed-forward'),
    (306, 'Trinity-style Transformer (2026 SOTA dense: gated GQA, QK-norm, 3:1 local/global)', 'Trinity Transformer 2026', 'GQA, QK-norm, gated attn, 3:1 SWA/global, sandwich'),
    # ==== Sparse, long-context & memory Transformers ====
    (400, 'Transformer-XL (segment-recurrent attention memory)', 'Transformer-XL', 'Segment recurrence with attention memory'),
    (401, 'Compressive Transformer', 'Compressive Transformer', 'Causal attention with compressed recurrent memory'),
    (402, 'kNN-LM (in-context recurrent memory)', 'kNN-LM', 'Causal nearest-neighbour recurrent memory (in-context)'),
    (403, 'Longformer (causal sparse attention)', 'Longformer (causal)', 'Sliding-window causal sparse-attention Transformer'),
    (404, 'BigBird (causal sparse attention)', 'BigBird (causal)', 'Block/global causal sparse-attention Transformer'),
    (405, 'RETRO-style (in-context recurrent retrieval memory)', 'RETRO-style', 'Causal retrieval-augmented recurrent memory (in-context)'),
    (406, 'Memorizing Transformer', 'Memorizing Transformer', 'Causal attention with recurrent key-value memory'),
    (407, 'Recurrent Interface Network (RIN)', 'Recurrent Interface', 'Stateful recurrent interface-gated latent LM'),
    (408, 'Titans (attention + neural long-term memory)', 'Titans', 'Short attention + learned long-term memory'),
    (409, 'Sparse Modern Transformer (NSA-inspired)', 'Sparse Transformer (NSA)', 'Local + compressed + selected-block sparse attention'),
    # ==== Gated & vanilla RNNs  (sequential, stateful) ====
    (500, 'Elman RNN (vanilla tanh)', 'RNN – Tanh', 'Vanilla Elman RNN with tanh activation'),
    (501, 'Long Short-Term Memory (LSTM)', 'LSTM', 'Long Short-Term Memory'),
    (502, 'ATanU-activated LSTM', 'ATanU-LSTM', 'LSTM with ArcTan unit activation'),
    (503, 'Gated Recurrent Unit (GRU)', 'GRU', 'Gated Recurrent Unit'),
    (504, 'Elman RNN (ReLU)', 'RNN – ReLU', 'Vanilla Elman RNN with ReLU activation'),
    (505, 'QRNN (causal convolution + f-pooling)', 'QRNN', 'Causal convolution + recurrent f-pooling'),
    (506, 'Intersection RNN (+RNN, Collins et al.)', 'Intersection RNN (+RNN)', 'Recurrent tanh gate + ReLU depth highway (Collins 2017)'),
    (507, 'UGRNN (update-gate RNN, Collins et al.)', 'UGRNN', 'Update-gate RNN: single coupled gate (Collins 2017)'),
    (508, 'SRU (Simple Recurrent Unit)', 'SRU', 'Simple Recurrent Unit: elementwise light recurrence'),
    (509, 'Independently Recurrent Neural Network (IndRNN)', 'IndRNN', 'Independently Recurrent NN (diagonal hidden-to-hidden)'),
    (510, 'JANET (forget-gate LSTM)', 'JANET', 'Forget-gate-only LSTM (simplified)'),
    (511, 'expRNN (orthogonal, matrix exponential)', 'expRNN', 'Orthogonal recurrence via matrix exponential, modReLU'),
    (512, 'NRU (Non-saturating Recurrent Unit)', 'NRU', 'Non-saturating additive-memory RNN'),
    (513, 'IndyGRU (independently recurrent GRU)', 'IndyGRU', 'GRU with diagonal (independent) recurrent weights'),
    (514, 'IndyLSTM (independently recurrent LSTM)', 'IndyLSTM', 'LSTM with diagonal (independent) recurrent weights'),
    (515, 'Mogrifier LSTM', 'Mogrifier LSTM', 'Context-modulated LSTM recurrent cell'),
    (516, 'Mogrifier GRU', 'Mogrifier GRU', 'Mogrifier input/state gating before a GRU step'),
    (517, 'SRU++ (SRU with causal attention)', 'SRU++', 'SRU with a light causal attention input (Lei 2021)'),
    (518, 'RRU (Residual Recurrent Unit)', 'RRU', 'Residual Recurrent Unit — gate-free ReZero recurrence'),
    (519, 'Light Recurrent Unit (LRU)', 'Light Recurrent Unit', 'Single forget-gate RNN, input-only candidate (LRU 2024)'),
    (520, 'M2RNN (nonlinear matrix-valued-state RNN, 2026 SOTA)', 'M2RNN', 'Nonlinear matrix-state RNN, sequential (Dao et al. 2026)'),
    # ==== Continuous-time & oscillator RNNs ====
    (600, 'LMU (Legendre Memory Unit)', 'LMU', 'Legendre-memory recurrent unit'),
    (601, 'Liquid Time-Constant network (LTC)', 'Liquid / LTC', 'Liquid Time-Constant Neural Network'),
    (602, 'UnICORNN (undamped oscillatory RNN)', 'UnICORNN', 'Undamped independent oscillators, symplectic Euler'),
    (603, 'CfC (Closed-form Continuous-time)', 'CfC', 'Closed-form continuous-time RNN'),
    # ==== Latent-variable RNNs ====
    (700, 'VRNN (variational recurrent neural network prior)', 'VRNN', 'Variational recurrent neural-network prior LM'),
    (701, 'SRNN (stochastic recurrent neural network prior)', 'SRNN', 'Stochastic recurrent neural-network prior LM'),
    # ==== xLSTM ====
    (800, 'xLSTM (sLSTM blocks only)', 'xLSTM – sLSTM blocks', 'Paper sLSTM blocks (conv, head GroupNorm, gated FFN)'),
    (801, 'xLSTM (mLSTM blocks only)', 'xLSTM – mLSTM blocks', 'Paper mLSTM blocks (up-proj, conv, skip, head norm)'),
    (802, 'xLSTM (full, xLSTM[a:b] mLSTM+sLSTM, 7:1)', 'xLSTM (full)', 'xLSTM[a:b]: mLSTM + sLSTM blocks, 7:1 default'),
    # ==== Parallel-scan minimal RNNs ====
    (900, 'minGRU (parallel-scan GRU)', 'minGRU', 'Parallelized minimal GRU (log-space scan)'),
    (901, 'minLSTM (parallel-scan LSTM)', 'minLSTM', 'Parallelized minimal LSTM (log-space scan)'),
    (902, 'MinRNN (parallel-scan vanilla RNN)', 'MinRNN ★', 'Parallelized vanilla RNN — multiple activation options'),
    (903, 'MinIndRNN (parallel-scan IndRNN)', 'MinIndRNN ★', 'Parallelized IndRNN — many activation choices'),
    (904, 'MinJANET (parallel-scan JANET)', 'MinJANET', 'Parallelized JANET forget-gate model'),
    (905, 'MinIndyGRU (parallel-scan IndyGRU)', 'MinIndyGRU', 'Parallelized IndyGRU (scan)'),
    (906, 'MinIndyLSTM (parallel-scan IndyLSTM)', 'MinIndyLSTM', 'Parallelized IndyLSTM (scan)'),
    (907, 'ParaGRU (ParaRNN: nonlinear GRU trained by parallel Newton)', 'ParaGRU (ParaRNN)', 'Nonlinear diagonal GRU, Newton + parallel scan (2025)'),
    (908, 'ParaLSTM (ParaRNN: nonlinear CIFG LSTM trained by parallel Newton)', 'ParaLSTM (ParaRNN)', 'Nonlinear peephole CIFG LSTM, Newton + 2x2 scan (2025)'),
    # ==== State-space models ====
    (1000, 'S4 (structured state-space model)', 'S4', 'Structured state-space reference recurrence'),
    (1001, 'DSS (diagonal state-space sequence model)', 'DSS', 'Diagonal state-space sequence model'),
    (1002, 'S4D (diagonal structured state-space model)', 'S4D', 'Diagonal structured state-space recurrence'),
    (1003, 'S5 (simplified state-space model)', 'S5', 'Simplified state-space sequence model'),
    (1004, 'H3 (Hungry Hungry Hippos)', 'H3', 'Hungry Hungry Hippos SSM'),
    (1005, 'LRU (linear recurrent unit)', 'LRU', 'Linear recurrent unit state-space model'),
    (1006, 'Mamba (selective scan)', 'Mamba', 'Selective state-space model (S6 scan)'),
    (1007, 'Mamba selective SSM (stage core)', 'Mamba SSM (stage core)', 'Selective SSM core without the Mamba block wrapper'),
    (1008, 'Mamba-2 (structured state-space duality)', 'Mamba-2', 'Structured state-space duality recurrence'),
    (1009, 'Mamba-3 SISO (trapezoidal + complex/RoPE)', 'Mamba-3 (SISO)', 'Trapezoidal, complex (RoPE) SSM, BC-norm (2026)'),
    (1010, 'Mamba-3 MIMO (rank 4, 2026 SOTA SSM)', 'Mamba-3 (MIMO)', 'Rank-4 multi-input multi-output Mamba-3 (2026 SOTA)'),
    # ==== Linear attention & gated linear RNNs ====
    (1100, 'Linear Transformer (recurrent form)', 'Linear Transformer', 'Linear attention recurrent form'),
    (1101, 'DeltaNet (delta-rule linear attention)', 'DeltaNet', 'Delta-rule linear recurrence'),
    (1102, 'RWKV-4 (scan)', 'RWKV-4', 'Receptance Weighted Key Value (scan)'),
    (1103, 'RetNet (multi-scale retention)', 'RetNet', 'Retentive network (multi-scale retention)'),
    (1104, 'GateLoop (scan)', 'GateLoop', 'Data-controlled linear recurrence'),
    (1105, 'HGRN (hierarchical gated recurrent network)', 'HGRN', 'Hierarchical Gated Recurrent Network'),
    (1106, 'HGRN2 (outer-product state expansion)', 'HGRN2', 'Outer-product gated recurrent memory'),
    (1107, 'Gated DeltaNet', 'Gated DeltaNet', 'Gated delta-rule matrix memory'),
    (1108, 'RWKV-7 Goose', 'RWKV-7 Goose', 'Dynamic state-evolution recurrence'),
    # ==== Hybrids  (attention + recurrence) ====
    (1200, 'Griffin (RG-LRU + local attention)', 'Griffin / RG-LRU', 'Real-gated linear recurrence + local attention'),
    (1201, 'Jamba-lite (1:3 attention/Mamba hybrid)', 'Jamba-lite', '1:3 attention/Mamba hybrid'),
)

# Saved configs record the ID scheme they use.  Anything without it predates
# the family-block renumbering (Sept 2026); LEGACY_MODEL_IDS maps those old
# IDs to the current ones and ``migrate_model_ids`` applies it on load.
MODEL_ID_SCHEME = 2
LEGACY_MODEL_IDS = {
    0: 1, 1: 5, 2: 500, 3: 504, 4: 503, 5: 501, 6: 509, 7: 513, 8: 502, 9: 102, 10: 301, 11: 800,
    12: 801, 13: 802, 14: 1006, 15: 900, 16: 901, 17: 1102, 18: 510, 19: 207, 20: 1104, 21: 200, 22:
    201, 23: 601, 24: 202, 25: 304, 26: 902, 27: 1200, 28: 1101, 29: 1103, 30: 1105, 32: 903, 33:
    904, 34: 305, 35: 1100, 36: 1004, 37: 303, 38: 905, 39: 906, 40: 407, 41: 1107, 42: 1008, 43:
    1106, 44: 400, 45: 408, 46: 1108, 47: 515, 48: 512, 49: 600, 50: 603, 51: 206, 52: 205, 53: 204,
    54: 203, 55: 101, 56: 100, 57: 3, 58: 4, 59: 1000, 60: 1002, 61: 1003, 62: 1001, 63: 1005, 64:
    401, 65: 406, 66: 403, 67: 404, 68: 402, 69: 405, 70: 700, 71: 701, 72: 104, 73: 103, 74: 208,
    75: 209, 76: 2, 77: 0, 90: 505, 91: 508, 92: 302, 93: 1201, 94: 409, 95: 1007, 96: 506, 97: 602,
    98: 514, 99: 519, 100: 518, 101: 517, 102: 516, 103: 511, 104: 1009
}
MLP_MODEL_ID = 1   # one-hot window MLP: the bottom-up MEGABYTE fine-patch decoder core

@dataclass(frozen=True)
class ModelSpec:
    """The single metadata contract for a selectable LineGen model.

    ``factory_key`` deliberately mirrors the persisted integer ID for now:
    construction remains in the long-standing ``build_model`` compatibility
    function, while all callers derive routing and UI behaviour from this
    record.  It gives the construction dispatcher a verified registry lookup
    without changing checkpoint/config serialization.
    """
    id: int
    name: str
    menu_group: str
    menu_label: str
    menu_description: str
    stateful: bool = False
    scan: bool = False
    attention: bool = False
    activation_default: Optional[str] = None
    megabyte_mixer: Optional[str] = None
    factory_key: Optional[int] = None


MODEL_MENU_GROUP_ORDER = tuple(MODEL_GROUPS.values())
_STATEFUL_MODEL_IDS = frozenset({
    400, 401, 402, 405, 406, 407, 408, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511,
    512, 513, 514, 515, 516, 517, 518, 519, 520, 600, 601, 602, 603, 700, 701, 800, 801, 802, 900,
    901, 902, 903, 904, 905, 906, 907, 908, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1008, 1009, 1010,
    1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1200, 1201
})
_SCAN_MODEL_IDS = frozenset({
    900, 901, 902, 903, 904, 905, 906, 1004, 1006, 1007, 1100, 1101, 1102, 1103, 1104, 1105, 1200,
    1201
})
_ATTENTION_MODEL_IDS = frozenset({
    207, 300, 301, 302, 304, 305, 306, 400, 403, 404, 407, 408, 409, 800, 801, 802, 1103, 1201
})
_ACTIVATION_DEFAULTS = {
    1: "mish", 5: "mish", 102: "mish", 200: "gelu", 201: "gelu", 202: "gelu", 203: "gelu", 204:
    "gelu", 205: "gelu", 206: "gelu", 207: "gelu", 300: "relu", 301: "gelu", 303: "swiglu", 304:
    "swiglu", 305: "mish", 306: "swiglu", 409: "swiglu"
}
MODEL_TYPE_NORMAL = 0
MODEL_TYPE_MEGABYTE = 1
MODEL_TYPE_MEGABYTE_BOTTOM_UP = 2


def is_bottom_up_megabyte(cfg) -> bool:
    """Whether this config needs the hourglass model's full-window interface."""
    return int(cfg.get("model_type", MODEL_TYPE_NORMAL)) == MODEL_TYPE_MEGABYTE_BOTTOM_UP

# A hierarchy needs a stage-level processor, not a complete LM with its own
# embeddings and vocabulary head.  These adapters deliberately cover only
# processors that MegaByteStageMixer can execute at every scale.
HIERARCHICAL_MODEL_MIXERS = {
    1: "mlp", 200: "gmlp", 201: "amlp", 202: "mlpmixer", 207: "hypermixer", 208: "toeplitz", 301:
    "gpt2", 304: "modern", 500: "rnn", 501: "lstm", 502: "atanulstm", 503: "gru", 504: "rnn_relu",
    505: "qrnn", 506: "irnn", 508: "sru", 509: "indrnn", 510: "janet", 511: "exprnn", 512: "nru",
    513: "indygru", 514: "indylstm", 515: "mogrifier_lstm", 516: "mogrifier_gru", 517: "srupp", 518:
    "rru", 519: "light_ru", 600: "lmu", 601: "liquid", 602: "unicornn", 603: "cfc", 800: "xlstm_s",
    801: "xlstm_m", 802: "xlstm", 900: "mingru", 901: "minlstm", 903: "minindrnn", 905:
    "minindygru", 906: "minindylstm", 1000: "s4", 1001: "dss", 1002: "s4d", 1003: "s5", 1005:
    "lru_ssm", 1006: "mamba", 1007: "mamba_ssm", 1008: "mamba2", 1009: "mamba3", 1101: "deltanet",
    1102: "rwkv", 1103: "retnet", 1106: "hgrn2", 1107: "gated_deltanet", 1108: "rwkv7", 520: "m2rnn"
}


@dataclass(frozen=True)
class HierarchyCoreCapability:
    """The core contract used by MEGABYTE routing and setup presentation."""
    mixer: str
    causal: bool = True
    incremental: bool = False
    uses_heads: bool = False


_INCREMENTAL_HIERARCHY_MIXERS = frozenset({
    "gru", "rnn", "rnn_relu", "lstm", "mingru", "minlstm", "minindrnn",
    "minindygru", "minindylstm", "mamba", "rwkv", "indrnn", "indygru",
    "janet", "atanulstm", "liquid", "mogrifier", "nru", "lmu", "cfc",
    "qrnn", "sru", "mamba_ssm", "deltanet", "gated_deltanet", "rwkv7",
    "modern", "transformer",
    "mogrifier_lstm", "mogrifier_gru", "irnn", "unicornn", "indylstm", "light_ru",
    "rru", "exprnn", "srupp", "mamba3", "m2rnn", "xlstm", "xlstm_m", "xlstm_s",
    "retnet", "mamba2", "hgrn2", "s4", "s4d", "s5", "dss", "lru_ssm",
})
_HEAD_HIERARCHY_MIXERS = frozenset({"transformer", "gpt2", "modern", "hypermixer", "xlstm", "xlstm_m", "xlstm_s"})
HIERARCHICAL_CORE_CAPABILITIES = {
    model_id: HierarchyCoreCapability(
        mixer=mixer,
        incremental=mixer in _INCREMENTAL_HIERARCHY_MIXERS,
        uses_heads=mixer in _HEAD_HIERARCHY_MIXERS,
    )
    for model_id, mixer in HIERARCHICAL_MODEL_MIXERS.items()
}
# Retired MEGABYTE-specific selections, in pre-renumbering IDs: the stage
# mixer and the (old) flat model ID each one now maps to.
LEGACY_MEGABYTE_MIXERS = {
    31: "transformer", 78: "gru", 79: "rnn", 80: "rnn_relu", 81: "lstm",
    82: "mingru", 83: "minlstm", 84: "mlpmixer", 85: "hypermixer", 86: "toeplitz",
    87: "minindrnn", 88: "minindygru", 89: "minindylstm",
}
LEGACY_MEGABYTE_MODEL_SELECTIONS = {
    31: 10, 78: 4, 79: 2, 80: 3, 81: 5, 82: 15, 83: 16, 84: 24,
    85: 19, 86: 74, 87: 32, 88: 38, 89: 39,
}


def megabyte_mixer_uses_heads(mixer: str) -> bool:
    if isinstance(mixer, int):
        mixer = HIERARCHICAL_MODEL_MIXERS.get(mixer)
    return mixer in _HEAD_HIERARCHY_MIXERS


def migrate_model_ids(cfg):
    """Translate a config saved before the family-block renumbering, in place.

    Configs carry ``model_id_scheme`` since; one without it uses the old IDs,
    including the retired MEGABYTE-specific selections.  Call this only on
    configs read from disk: in-memory configs already use current IDs."""
    if cfg.get("model_id_scheme") == MODEL_ID_SCHEME:
        return cfg
    selection = cfg.get("model_selection")
    if selection in LEGACY_MEGABYTE_MODEL_SELECTIONS:
        cfg["model_type"] = MODEL_TYPE_MEGABYTE
        cfg["_legacy_megabyte"] = True
        cfg["_legacy_megabyte_mixer"] = LEGACY_MEGABYTE_MIXERS[selection]
        selection = LEGACY_MEGABYTE_MODEL_SELECTIONS[selection]
    if selection is not None:
        if selection not in LEGACY_MODEL_IDS:
            raise ValueError(f"config names unknown legacy model ID {selection}")
        cfg["model_selection"] = LEGACY_MODEL_IDS[selection]
    for key in ("megabyte_stage_mixers", "megabyte_bottom_up_encoder_stage_mixers",
                "megabyte_bottom_up_decoder_stage_mixers"):
        mixers = cfg.get(key)
        if isinstance(mixers, int):
            cfg[key] = LEGACY_MODEL_IDS[mixers]
        elif isinstance(mixers, (list, tuple)):
            cfg[key] = [LEGACY_MODEL_IDS[m] if isinstance(m, int) else m for m in mixers]
    cfg["model_id_scheme"] = MODEL_ID_SCHEME
    return cfg


def load_run_config(path=None):
    """Read a saved run config, translating pre-renumbering model IDs."""
    return migrate_model_ids(load_json(CONFIG_PATH if path is None else path))


def normalize_model_config(cfg):
    """Validate the topology field (legacy MEGABYTE IDs are translated on load).

    Every config reaching here holds current IDs, so it is stamped with the
    scheme: saving it later can never make a reload re-translate it."""
    cfg.setdefault("model_id_scheme", MODEL_ID_SCHEME)
    cfg["model_type"] = int(cfg.get("model_type", MODEL_TYPE_NORMAL))
    if cfg["model_type"] not in {
        MODEL_TYPE_NORMAL, MODEL_TYPE_MEGABYTE, MODEL_TYPE_MEGABYTE_BOTTOM_UP,
    }:
        raise ValueError(f"unknown model type: {cfg['model_type']}")
MODEL_SPECS = {
    model_id: ModelSpec(
        id=model_id,
        name=name,
        menu_group=MODEL_GROUPS[model_id // 100],
        menu_label=label,
        menu_description=description,
        stateful=model_id in _STATEFUL_MODEL_IDS,
        scan=model_id in _SCAN_MODEL_IDS,
        attention=model_id in _ATTENTION_MODEL_IDS,
        activation_default=_ACTIVATION_DEFAULTS.get(model_id),
        megabyte_mixer=HIERARCHICAL_MODEL_MIXERS.get(model_id),
        factory_key=model_id,
    )
    for model_id, name, label, description in MODEL_REGISTRY
}
if len(MODEL_SPECS) != len(MODEL_REGISTRY):
    raise RuntimeError("MODEL_REGISTRY lists a model ID twice")
_unregistered = (_STATEFUL_MODEL_IDS | _SCAN_MODEL_IDS | _ATTENTION_MODEL_IDS
                 | set(_ACTIVATION_DEFAULTS) | set(HIERARCHICAL_MODEL_MIXERS)) - set(MODEL_SPECS)
if _unregistered or set(LEGACY_MODEL_IDS.values()) - set(MODEL_SPECS):
    raise RuntimeError(f"Model tables name unregistered IDs: {sorted(_unregistered)}")

# Compatibility aliases for external callers and saved-run tooling.  These are
# derived, so adding a model cannot make routing and menu data disagree.
MODEL_NAMES = {model_id: spec.name for model_id, spec in MODEL_SPECS.items()}
MODEL_IDS = frozenset(MODEL_SPECS)
RNN_MODEL_IDS = frozenset(spec.id for spec in MODEL_SPECS.values() if spec.stateful)
SCAN_MODEL_IDS = frozenset(spec.id for spec in MODEL_SPECS.values() if spec.scan)
ATTN_MODEL_IDS = frozenset(spec.id for spec in MODEL_SPECS.values() if spec.attention)
NON_RNN_ACTIVATION_IDS = frozenset(
    spec.id for spec in MODEL_SPECS.values() if spec.activation_default is not None
)
MODEL_DEFAULT_ACTIVATIONS = {
    spec.id: spec.activation_default for spec in MODEL_SPECS.values()
    if spec.activation_default is not None
}
MEGABYTE_MODEL_MIXERS = {
    spec.id: spec.megabyte_mixer for spec in MODEL_SPECS.values()
    if spec.megabyte_mixer is not None
}
MODEL_MENU = "\n".join(f"{model_id} - {MODEL_NAMES[model_id]}" for model_id in sorted(MODEL_IDS))

# Used to trigger the "RNN Specific Settings" menu (residuals, norms, etc)
# These are models that are likely implemented via the BuiltinRNNWrapper or similar loops
def build_config_benchmark():
    dataset_path = prompt_str("Dataset location (text file path): ")
    dataset_type = prompt_int("Dataset type: standard/0 or line/1? ", valid={0,1})
    tokenizer_mode = prompt_int("Tokenizer (-1=binary, 0=byte, 1=char, 2=word): ", valid={-1,0,1,2})

    activation_name = prompt_activation("mish", "MLP baseline")

    embed_dim  = prompt_int("Embed dim (also hidden dim / TCN channels / Transformer d_model): ")
    layer_count = prompt_int("Layer count: ")

    # This legacy helper builds the MLP baseline (MLP_MODEL_ID), so it has no heads.
    head_count = 4

    if dataset_type == 0:
        seq_len = prompt_int("Seq len (classic corpus): ")
        classic_val_path = prompt_str("Optional validation file path (enter to skip): ", default="")
    else:
        seq_len = 0
        classic_val_path = prompt_str("Optional validation file path (enter to skip): ", default="")

    batch_size  = prompt_int("Batch size: ")
    learning_rate = prompt_float("Learning rate (Adam): ")
    iters_total = prompt_int("Benchmark iteration count (total SGD steps per model): ")

    cfg = RunConfig(
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        model_selection=MLP_MODEL_ID,
        activation_name=activation_name,
        embed_dim=embed_dim,
        head_count=head_count,
        layer_count=layer_count,
        seq_len=seq_len,
        epoch_count=1,
        batch_size=batch_size,
        learning_rate=learning_rate,
        tokenizer_mode=tokenizer_mode
    ).to_dict()

    if classic_val_path:
        cfg["classic_val_path"] = classic_val_path

    cfg["use_tbptt"] = False
    cfg["bptt_window"] = 0
    cfg["tbptt_total_len"] = 0

    cfg["_bench_iters"] = iters_total
    return cfg


# ==============================================================================
#  BENCHMARKING SUITE (Updated)
# ==============================================================================

# Only these models consume the residual/norm/dropout/multiplier settings
# exposed by benchmark mode: the built-in cells and every CustomRNNWrapper cell
# (build_model's RNN_MAP).  Only the custom cells have the post-layer FFN.
BENCH_CUSTOM_RNN_IDS = frozenset({502, 506, 507, 509, 510, 511, 513, 514, 515, 516, 518, 519, 601, 602})
BENCH_RNN_STRUCTURE_IDS = frozenset({500, 501, 503, 504}) | BENCH_CUSTOM_RNN_IDS

# Group definitions.  IDs are deliberately explicit so the grouping remains
# readable, while groups 0/1 are derived from MODEL_NAMES and can never omit a
# newly registered model.
BENCH_GROUPS = {
    # Stateless / fixed-context models: MLPs, convolutions, attention, and mixers.
    2: [
        0, 1, 2, 3, 4, 5, 100, 101, 102, 103, 104, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
        300, 301, 302, 303, 304, 305, 306, 403, 404, 409
    ],

    # Conventional and continuous-time recurrent cells.
    3: [
        500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517,
        518, 519, 520, 600, 601, 602, 603, 700, 701
    ],

    # xLSTM, scan/SSM, recurrent-attention, and modern memory models.
    4: [
        400, 401, 402, 405, 406, 407, 408, 800, 801, 802, 900, 901, 902, 903, 904, 905, 906, 907, 908,
        1000,
        1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1100, 1101, 1102, 1103, 1104,
        1105, 1106, 1107, 1108, 1200, 1201
    ],

    # A compact, broadly representative comparison set.
    5: [
        0, 1, 2, 5, 102, 103, 104, 200, 201, 202, 203, 204, 205, 206, 208, 300, 301, 304, 305, 306,
        400, 407, 408, 409, 500, 501, 503, 504, 520, 900, 901, 902, 907, 1006, 1008, 1010, 1100, 1102,
        1103, 1107, 1108, 1200
    ],
}

# Keep the curated benchmark groups and the train menu in lockstep.  Groups may
# overlap conceptually, but together they must cover every registered model and
# may never contain an unknown selection ID.
_BENCH_GROUP_MODEL_IDS = set().union(*BENCH_GROUPS.values())
if _BENCH_GROUP_MODEL_IDS != set(MODEL_IDS):
    missing = sorted(set(MODEL_IDS) - _BENCH_GROUP_MODEL_IDS)
    unknown = sorted(_BENCH_GROUP_MODEL_IDS - set(MODEL_IDS))
    raise RuntimeError(f"Benchmark/train model registry mismatch: missing={missing}, unknown={unknown}")

# First public version (usually the arXiv preprint) of the idea each model
# implements, as (year, month).  Benchmarks run oldest-first, so a run replays
# the field's history; the month only orders models within a year.  Variants
# that add an existing recipe to an older cell (Min*, Mogrifier GRU) take the
# recipe's date.
MODEL_ORIGINS = {
    0: (1913, 1),       # Markov chain over letters (Markov)
    1: (1986, 10),      # backprop-trained MLP (Rumelhart, Hinton & Williams)
    500: (1990, 4),     # Elman RNN
    501: (1997, 11),    # LSTM (Hochreiter & Schmidhuber)
    502: (1997, 11),    # LSTM with an arctan unit; no separate source found
    2: (2000, 12),      # neural probabilistic LM (Bengio et al., NIPS 2000)
    3: (2011, 4),       # NADE
    503: (2014, 6),     # GRU
    4: (2015, 2),       # MADE
    504: (2015, 4),     # ReLU RNN (IRNN, Le et al.)
    700: (2015, 6),     # VRNN
    100: (2016, 1),     # PixelCNN (Pixel RNN paper)
    701: (2016, 5),     # SRNN (Fraccaro et al.)
    101: (2016, 9),     # WaveNet
    505: (2016, 11),    # QRNN
    506: (2016, 11),    # Intersection RNN (Collins et al.)
    507: (2016, 11),    # UGRNN (Collins et al., same paper as the +RNN)
    5: (2017, 6),       # Transformer feed-forward block
    300: (2017, 6),     # Transformer (Vaswani et al.), decoder-only
    508: (2017, 9),     # SRU
    509: (2018, 3),     # IndRNN
    102: (2018, 3),     # TCN (Bai et al.)
    510: (2018, 4),     # JANET
    511: (2019, 1),     # expRNN
    400: (2019, 1),     # Transformer-XL
    301: (2019, 2),     # GPT-2
    512: (2019, 2),     # NRU
    513: (2019, 3),     # IndyGRU (Gonnet & Deselaers)
    514: (2019, 3),     # IndyLSTM (Gonnet & Deselaers)
    515: (2019, 9),     # Mogrifier LSTM
    516: (2019, 9),     # Mogrifier GRU
    401: (2019, 11),    # Compressive Transformer
    402: (2019, 11),    # kNN-LM
    600: (2019, 12),    # LMU
    403: (2020, 4),     # Longformer
    601: (2020, 6),     # Liquid time-constant network
    1100: (2020, 6),    # Linear Transformer (Katharopoulos et al.)
    404: (2020, 7),     # BigBird
    302: (2021, 1),     # Switch Transformer
    1101: (2021, 2),    # DeltaNet (Schlag et al.)
    517: (2021, 2),     # SRU++
    602: (2021, 3),     # UnICORNN
    200: (2021, 5),     # gMLP
    201: (2021, 5),     # aMLP
    202: (2021, 5),     # MLP-Mixer
    603: (2021, 6),     # CfC
    203: (2021, 6),     # CCS token-mixing MLP (Yu et al.)
    518: (2021, 8),     # RRU (Zakovskis et al.)
    1000: (2021, 10),   # S4
    204: (2021, 11),    # Wave-MLP
    405: (2021, 12),    # RETRO
    205: (2022, 1),     # DynaMixer
    103: (2022, 1),     # ConvNeXt
    206: (2022, 2),     # pNLP-Mixer
    207: (2022, 3),     # HyperMixer
    303: (2022, 3),     # DCT-Former
    1001: (2022, 3),    # DSS
    406: (2022, 3),     # Memorizing Transformer
    1002: (2022, 6),    # S4D
    1003: (2022, 8),    # S5
    1004: (2022, 12),   # H3
    407: (2022, 12),    # RIN (Jabri et al.)
    104: (2023, 2),     # Hyena
    1005: (2023, 3),    # LRU (Orvieto et al.)
    1102: (2023, 5),    # RWKV (RWKV-4 paper)
    208: (2023, 5),     # Toeplitz neural network token mixing
    1103: (2023, 7),    # RetNet
    1104: (2023, 11),   # GateLoop
    1105: (2023, 11),   # HGRN
    1006: (2023, 12),   # Mamba
    1007: (2023, 12),   # Mamba selective SSM
    1200: (2024, 2),    # Griffin
    1201: (2024, 3),    # Jamba
    304: (2024, 4),     # Llama 3
    305: (2024, 4),     # KAN
    1106: (2024, 4),    # HGRN2
    800: (2024, 5),     # xLSTM
    801: (2024, 5),
    802: (2024, 5),
    1008: (2024, 5),    # Mamba-2
    519: (2024, 6),     # Light Recurrent Unit (Electronics 2024; month unknown)
    900: (2024, 10),    # minGRU / minLSTM ("Were RNNs All We Needed?")
    901: (2024, 10),
    902: (2024, 10),
    903: (2024, 10),
    904: (2024, 10),
    905: (2024, 10),
    906: (2024, 10),
    1107: (2024, 12),   # Gated DeltaNet
    408: (2024, 12),    # Titans
    409: (2025, 2),     # Native Sparse Attention
    1108: (2025, 3),    # RWKV-7 Goose
    907: (2025, 10),    # ParaRNN (Danieli et al., arXiv 2510.21450)
    908: (2025, 10),
    209: (2025, 12),    # experimental Grassmann-flow mixer; no source date found
    1009: (2026, 1),    # Mamba-3 (Lahoti et al., 2026)
    1010: (2026, 1),    # Mamba-3 MIMO (same paper as Mamba-3 SISO)
    306: (2026, 2),     # Arcee Trinity technical report
    520: (2026, 3),     # M2RNN (Mishra et al., arXiv 2603.14360)
}
if set(MODEL_ORIGINS) != set(MODEL_IDS):
    raise RuntimeError(
        f"Every model needs a MODEL_ORIGINS date: missing={sorted(set(MODEL_IDS) - set(MODEL_ORIGINS))}, "
        f"unknown={sorted(set(MODEL_ORIGINS) - set(MODEL_IDS))}"
    )

# ------------------------------------------------------------------ per-model options
_MINRNN_ACT_NAMES = ["tanh", "relu", "silu", "gelu", "sigmoid", "g_act"]
_MININDRNN_ACT_NAMES = [
    "tanh", "relu", "silu", "prelu0", "prelu", "lrelu0.2", "lrelu0.01", "gelu", "bentid",
    "sine", "cosine", "snake", "stepsine", "stepcos", "mish", "cone", "relu2", "g_act",
]


def _bench_opt(key, label, kind, default, choices=None, names=None, cell=False, short=None,
               lo=None, hi=None):
    """One benchmarkable model option.  ``cell`` options live in
    ``rnn_cell_options`` (read by build_model's custom RNN path); the others
    are top-level config keys.  ``kind`` is choice, bool, int or float."""
    if kind == "bool":
        choices = [False, True]
    return dict(key=key, label=label, kind=kind, default=default, choices=choices, names=names,
                cell=cell, short=short or key, lo=lo, hi=hi)


# The same options the training menu exposes, keyed by model ID.
BENCH_VARIANTS = {
    902: [_bench_opt("minrnn_act", "MinRNN activation", "choice", 0, list(range(6)), _MINRNN_ACT_NAMES,
                    short="act")],
    903: [_bench_opt("minrnn_act", "MinIndRNN activation", "choice", 0, list(range(18)), _MININDRNN_ACT_NAMES,
                    short="act")],
    509: [_bench_opt("indrnn_activation", "IndRNN activation", "choice", "relu", ["relu", "tanh"],
                   cell=True, short="act")],
    513: [_bench_opt("relu_gates", "ReLU gates instead of sigmoid", "bool", False, cell=True, short="relu_gates")],
    514: [_bench_opt("relu_gates", "ReLU gates instead of sigmoid", "bool", False, cell=True, short="relu_gates")],
    515: [_bench_opt("mogrifier_rounds", "Mogrifier rounds (0 = plain LSTM)", "int", 5, cell=True,
                    short="rounds", lo=0)],
    516: [_bench_opt("mogrifier_rounds", "Mogrifier rounds (0 = plain GRU)", "int", 5, cell=True,
                     short="rounds", lo=0)],
    602: [_bench_opt("unicornn_dt", "UnICORNN time step dt", "float", 0.1, cell=True, short="dt", lo=0.0),
         _bench_opt("unicornn_alpha", "UnICORNN alpha", "float", 10.0, cell=True, short="alpha")],
    519: [_bench_opt("lru_highway", "LRU highway stacking", "bool", False, cell=True, short="highway")],
    518: [_bench_opt("rru_middle_multiplier", "RRU middle-layer multiplier", "float", 2.0, cell=True,
                     short="mid", lo=0.0),
          _bench_opt("rru_dropout", "RRU in-cell dropout", "float", 0.0, cell=True, short="cell_drop",
                     lo=0.0, hi=1.0)],
    2: [_bench_opt("ngram_context", "N-gram context tokens", "int", 4, short="ctx", lo=1, hi=32)],
    409: [_bench_opt("sparse_local_window", "Sparse local window (auto = min(512, seq))", "int", None,
                    short="window", lo=1),
         _bench_opt("sparse_compression_block", "Sparse compression block", "int", 32, short="block", lo=1),
         _bench_opt("sparse_selected_blocks", "Sparse selected blocks", "int", 16, short="selected", lo=1)],
    407: [_bench_opt("rin_num_latents", "RIN latent count", "int", 8, short="latents", lo=1)],
    505: [_bench_opt("qrnn_kernel_size", "QRNN kernel size", "int", 2, short="kernel", lo=1)],
    302: [_bench_opt("moe_num_experts", "Switch experts", "int", 4, short="experts", lo=1)],
    400: [_bench_opt("txl_mem_len", "Transformer-XL memory length", "int", 128, short="mem", lo=1)],
    408: [_bench_opt("titans_memory_slots", "Titans memory slots", "int", 32, short="slots", lo=1)],
}


def _bench_value_text(opt, value):
    if value is None:
        return "auto"
    if opt["kind"] == "bool":
        return "on" if value else "off"
    if opt["names"]:
        return opt["names"][value]
    return f"{value:g}" if isinstance(value, float) else str(value)


def _bench_parse_values(opt, raw):
    """Parse one value, a comma list, or 'all'; raise ValueError when invalid."""
    raw = raw.strip()
    if not raw:
        return [opt["default"]]
    if raw.lower() == "all":
        if opt["choices"] is None:
            raise ValueError("'all' only works for options with a fixed set of choices")
        return list(opt["choices"])
    values = []
    for token in (t.strip() for t in raw.split(",")):
        if not token:
            continue
        if opt["kind"] == "bool":
            if token.lower() in {"1", "y", "yes", "on", "true"}:
                value = True
            elif token.lower() in {"0", "n", "no", "off", "false"}:
                value = False
            else:
                raise ValueError(f"{token!r} is not on/off")
        elif opt["kind"] == "choice":
            if opt["names"] and token.lower() in opt["names"]:
                value = opt["names"].index(token.lower())
            elif isinstance(opt["choices"][0], int) and token.lstrip("-").isdigit():
                value = int(token)
            else:
                value = token.lower()
            if value not in opt["choices"]:
                raise ValueError(f"{token!r} is not one of {opt['names'] or opt['choices']}")
        else:
            value = int(token) if opt["kind"] == "int" else float(token)
            if opt["lo"] is not None and value < opt["lo"]:
                raise ValueError(f"{token} is below the minimum {opt['lo']}")
            if opt["hi"] is not None and value > opt["hi"]:
                raise ValueError(f"{token} is above the maximum {opt['hi']}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("no value given")
    return values


def _prompt_bench_variants(base_ids, ultra):
    """Return {model_id: [overrides, ...]}; each overrides dict becomes one row."""
    present = [mid for mid in base_ids if mid in BENCH_VARIANTS]
    if not present:
        return {}
    chosen = {}
    customise = 0
    if not ultra:
        cli_section("Per-model Options", 64)
        print(f"  │  {_c(_DIM, f'{len(present)} selected models have their own options (activation, gates, …).')}")
        print(f"  │  {_c(_DIM, 'Enter one value, a comma list to benchmark each as its own row,')}")
        print(f"  │  {_c(_DIM, 'or all for every choice.')}")
        print(f"  │")
        customise = prompt_int("Use defaults (0) or customise / sweep (1)", valid={0, 1}, default=0)
    for mid in present:
        if customise:
            print(f"  │")
            print(f"  │  {_c(_WH, _B, MODEL_NAMES[mid])}")
        for opt in BENCH_VARIANTS[mid]:
            if ultra:
                values = list(opt["choices"]) if opt["kind"] in {"choice", "bool"} else [opt["default"]]
            elif not customise:
                values = [opt["default"]]
            else:
                if opt["names"]:
                    print(f"  │  {_c(_DIM, 'choices: ' + ', '.join(f'{i}={n}' for i, n in enumerate(opt['names'])))}")
                elif opt["kind"] == "choice":
                    print(f"  │  {_c(_DIM, 'choices: ' + ', '.join(map(str, opt['choices'])))}")
                while True:
                    raw = prompt_str(f"  {opt['label']}", default=_bench_value_text(opt, opt["default"]))
                    if raw == _bench_value_text(opt, opt["default"]):
                        raw = ""
                    try:
                        values = _bench_parse_values(opt, raw)
                        break
                    except ValueError as exc:
                        print(f"  {_c(_RD, '✗')} {exc}")
            chosen[(mid, opt["key"])] = values
    variants = {}
    for mid in present:
        opts = BENCH_VARIANTS[mid]
        rows = []
        for combo in itertools.product(*(chosen[(mid, o["key"])] for o in opts)):
            overrides, cell = {}, {}
            for opt, value in zip(opts, combo):
                if value is None:
                    continue
                (cell if opt["cell"] else overrides)[opt["key"]] = value
            if cell:
                overrides["rnn_cell_options"] = cell
            rows.append(overrides)
        variants[mid] = rows
    return variants


def _bench_variant_label(task):
    """Short label: choice and on/off options always, numbers when changed."""
    mid, overrides = task["id"], task["overrides"]
    values = dict(overrides.get("rnn_cell_options", {}))
    values.update({k: v for k, v in overrides.items() if k != "rnn_cell_options"})
    parts = [
        f"{opt['short']}={_bench_value_text(opt, values[opt['key']])}"
        for opt in BENCH_VARIANTS.get(mid, [])
        if opt["key"] in values
        and (opt["kind"] in {"choice", "bool"} or values[opt["key"]] != opt["default"])
    ]
    return f" [{', '.join(parts)}]" if parts else ""


def _bench_short_name(record, width=40):
    """Compact name: the acronym for long names such as 'Independently
    Recurrent Neural Network (IndRNN)', then the variant and structure tags."""
    base = MODEL_NAMES.get(record["id"], str(record["id"]))
    match = re.match(r"(.*?) \((.*?)\)", base)
    if match and len(match.group(1)) > 20 and len(match.group(2)) <= 16:
        short = match.group(2)
    else:
        short = base.split(" (")[0]
    tags = record["name"][len(base):].strip() if record["name"].startswith(base) else ""
    text = f"{short} {tags}".strip()
    return text if len(text) <= width else text[:width - 1] + "…"


def _prompt_bench_filters(base_ids):
    cli_section("Filters", 64)
    print(f"  │  {_c(_DIM, 'Years use each model’s first publication, e.g. 2015-2023, 2020-, -1999.')}")
    print(f"  │")
    while True:
        raw = prompt_str("Years  (blank = all)", default="").strip()
        match = re.fullmatch(r"(\d{4})?\s*(-)?\s*(\d{4})?", raw)
        if raw and match and (match.group(1) or match.group(3)):
            lo = int(match.group(1)) if match.group(1) else 0
            hi = int(match.group(3)) if match.group(3) else (9999 if match.group(2) else lo)
            if not match.group(1):
                lo = 0
            break
        if not raw:
            lo, hi = 0, 9999
            break
        print(f"  {_c(_RD, '✗')} Use a year or a range such as 2015-2023.")
    raw = prompt_str("Exclude model IDs  (comma-separated, blank = none)", default="")
    excluded = {int(x) for x in re.findall(r"\d+", raw)}
    kept = [mid for mid in base_ids if lo <= MODEL_ORIGINS[mid][0] <= hi and mid not in excluded]
    cli_section_end(64)
    if not kept:
        raise ValueError("The filters removed every model.")
    if len(kept) != len(base_ids):
        pinfo(f"{len(kept)} of {len(base_ids)} models kept by the filters")
    return kept


def get_bench_model_list():
    W = 64
    print(f"\n  {_c(_CY, _B, '┌─')} {_c(_WH, _B, 'Benchmark Group')} {_c(_CY, '─' * (W - 22) + '┐')}")
    cli_opt(0, "All models",         "Every architecture currently registered in the model zoo")
    cli_opt(1, "Ultra (all × options)", "All models; every activation / on-off option as its own row")
    cli_opt(2, "Fixed-context",      "MLPs, convolution, Transformers, and mixer variants")
    cli_opt(3, "Classic RNNs",        "Vanilla RNN, GRU/LSTM, IndRNN, LTC, Mogrifier, NRU…")
    cli_opt(4, "Modern recurrent",   "xLSTM, SSMs, scans, recurrent attention, and memory models")
    cli_opt(5, "Core comparison",    "Representative models across the current architecture families")
    cli_opt(6, "Custom list",         "Enter comma-separated model IDs manually")
    cli_blank_row()
    print(f"  {_c(_CY, '└' + '─' * (W - 2) + '┘')}\n")

    choice = prompt_int("Group", valid={0,1,2,3,4,5,6})
    if choice in [0, 1]:
        base_ids = sorted(MODEL_NAMES)
    elif choice in BENCH_GROUPS:
        base_ids = BENCH_GROUPS[choice]
    else:
        print_model_menu()
        raw = prompt_str("Model IDs  (comma-separated, e.g. 1,301,1006)")
        base_ids = list(dict.fromkeys(int(x.strip()) for x in raw.split(",") if x.strip().isdigit()))
        unknown = sorted(set(base_ids) - set(MODEL_IDS))
        if unknown:
            raise ValueError(f"Unknown model IDs: {unknown}. Choose IDs shown in the model menu.")

    base_ids = _prompt_bench_filters(base_ids)
    variants = _prompt_bench_variants(base_ids, ultra=(choice == 1))
    tasks = [
        {"id": mid, "overrides": overrides}
        for mid in base_ids for overrides in variants.get(mid, [{}])
    ]
    # Oldest idea first; a stable sort keeps each model's variants together.
    tasks.sort(key=lambda task: (MODEL_ORIGINS[task["id"]], task["id"]))
    return tasks


def _prompt_bench_hierarchy(tasks):
    """Optional MEGABYTE layout shared by every compatible model in the run."""
    compatible = [t for t in tasks if t["id"] in HIERARCHICAL_MODEL_MIXERS and t["id"] != MLP_MODEL_ID]
    if not compatible:
        return {"model_type": MODEL_TYPE_NORMAL}
    cli_section("Hierarchy", 64)
    cli_opt(0, "Flat", "Every model as an ordinary language model")
    cli_opt(1, "MEGABYTE", "Each compatible model at every hierarchy stage")
    cli_opt(2, "MEGABYTE-Bottom up", "Encode fine→coarse, then decode coarse→fine")
    print(f"  │")
    model_type = prompt_int("Model type", valid={0, 1, 2}, default=0)
    if model_type == MODEL_TYPE_NORMAL:
        cli_section_end(64)
        return {"model_type": MODEL_TYPE_NORMAL}
    skipped = len(tasks) - len(compatible)
    if skipped:
        pwarn(f"{skipped} selected rows have no MEGABYTE stage adapter and will be skipped")
    print(f"  │  {_c(_DIM, 'Stage lengths multiply into the training window; the benchmarked')}")
    print(f"  │  {_c(_DIM, 'model is the core at every stage.  Heads apply only to attention cores.')}")
    print(f"  │")
    stage_count = prompt_int("MEGABYTE stage count  (2 or more)", minimum=2, default=2)
    layout = {"model_type": model_type, "stage_seq_lens": [], "stage_dims": [],
              "stage_child_embed_dims": [], "stage_depths": [], "stage_heads": []}
    for stage in range(stage_count):
        label = f"Stage {stage + 1} {'(coarse)' if stage == 0 else '(fine)' if stage == stage_count - 1 else ''}".rstrip()
        print(f"  │")
        print(f"  │  {_c(_WH, _B, label)}")
        layout["stage_seq_lens"].append(prompt_int("  Sequence length / groups", default=128 if stage == 0 else 4))
        dim = prompt_int("  Hidden dimension", default=256 if stage == 0 else 128)
        layout["stage_dims"].append(dim)
        layout["stage_child_embed_dims"].append(prompt_int(
            f"  Child token embedding dimension  (1–{dim})", valid=range(1, dim + 1), default=min(64, dim)))
        layout["stage_depths"].append(prompt_int("  Layer count", default=2))
        layout["stage_heads"].append(prompt_int("  Attention head count", default=8))
    cli_section_end(64)
    return layout


# ------------------------------------------------------------------ settings
# Version 2: model IDs in family blocks (version-1 files are translated on load).
BENCH_SETTINGS_VERSION = 2
BENCH_RESULTS_ROOT = "benchmark_results"


def collect_bench_settings():
    """Ask every benchmark question once; the answers form a JSON preset."""
    s = {"version": BENCH_SETTINGS_VERSION}

    # ── Dataset & tokenizer ────────────────────────────────────────────────────
    cli_section("Dataset", 64)
    print(f"  │")
    s["dataset_path"] = prompt_str("Dataset file path")
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Dataset type:')}")
    cli_opt(0, "Standard (corpus)", "Sliding-window over a continuous text stream")
    cli_opt(1, "Line mode",         "One example per line with BOS/EOS padding")
    print(f"  │")
    s["dataset_type"] = prompt_int("Dataset type", valid={0,1})
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Tokenizer:')}")
    cli_opt(-1, "Binary"); cli_opt(0, "Byte"); cli_opt(1, "Char")
    cli_opt( 2, "Word");   cli_opt(3, "Tiktoken"); cli_opt(4, "BPE")
    print(f"  │")
    s["tokenizer_mode"] = prompt_int("Tokenizer", valid={-1,0,1,2,3,4})
    cli_section_end(64)
    s["seq2seq"] = prompt_seq2seq_config(s["dataset_path"]) if s["dataset_type"] == 1 else None

    # ── Models ─────────────────────────────────────────────────────────────────
    tasks = get_bench_model_list()
    s["hierarchy"] = _prompt_bench_hierarchy(tasks)
    hierarchical = s["hierarchy"]["model_type"] != MODEL_TYPE_NORMAL
    if hierarchical:
        tasks = [t for t in tasks if t["id"] in HIERARCHICAL_MODEL_MIXERS and t["id"] != MLP_MODEL_ID]
    s["tasks"] = tasks
    ids = {t["id"] for t in tasks}

    # Ask once per architecture, not once per variant.
    s["activations"] = {}
    activation_models = sorted(ids & NON_RNN_ACTIVATION_IDS)
    if activation_models:
        cli_section("Non-recurrent Activations", 64)
        print(f"  │  {_c(_DIM, 'Choose each architecture default or a user-defined activation.')}")
        print(f"  │")
        for mid in activation_models:
            default = MODEL_DEFAULT_ACTIVATIONS[mid]
            label = MODEL_NAMES.get(mid, f"Model {mid}")
            cli_opt(mid, label, f"default: {default.upper()}", kw=4, lw=26)
            mode = prompt_int("Use default (0) or customise (1)", valid={0, 1}, default=0)
            s["activations"][str(mid)] = default if mode == 0 else prompt_activation(default, label)
        cli_section_end(64)

    # ── Architecture ───────────────────────────────────────────────────────────
    cli_section("Architecture Defaults", 64)
    print(f"  │  {_c(_DIM, 'Applied to every model in the run.')}")
    print(f"  │")
    s["size_mode"], s["target_params"] = "fixed", 0
    s["embed_dim"], s["layer_count"], s["head_count"], s["seq_len"] = 256, 4, 4, 0
    if hierarchical:
        pinfo("Width, depth, heads and window come from the MEGABYTE stages")
    else:
        cli_opt(0, "Fixed width",     "Every model uses the same hidden dimension")
        cli_opt(1, "Match parameters", "Scale each model's hidden dimension to a parameter budget")
        print(f"  │")
        if prompt_int("Model size", valid={0, 1}, default=0) == 1:
            s["size_mode"] = "params"
            s["target_params"] = int(prompt_float("Target parameters  (millions)", default=2.0) * 1e6)
        else:
            s["embed_dim"] = prompt_int("Embedding / hidden dim", default=256)
        s["layer_count"] = prompt_int("Layer count", default=4)
        if ids & ATTN_MODEL_IDS:
            s["head_count"] = prompt_int("Attention / xLSTM head count", default=4)
        if s["dataset_type"] == 0:
            s["seq_len"] = prompt_int("Sequence length", default=128)
    s["batch_size"] = prompt_int("Batch size", default=32)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Stop each model after:')}")
    cli_opt(0, "Maximum steps",   "A fixed number of optimizer updates")
    cli_opt(1, "Maximum seconds", "A fixed wall-clock time budget")
    print(f"  │")
    if prompt_int("Benchmark limit", valid={0,1}, default=0) == 0:
        s["total_iters"] = prompt_int("Maximum steps per model", default=500, minimum=1)
        s["max_seconds"] = None
    else:
        s["total_iters"] = 0
        s["max_seconds"] = prompt_float("Maximum seconds per model", default=60.0)
        while s["max_seconds"] <= 0:
            print(f"  {_c(_RD, '✗')} Maximum seconds must be greater than zero.")
            s["max_seconds"] = prompt_float("Maximum seconds per model", default=60.0)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Set 0 to benchmark every model regardless of training speed.')}")
    s["min_iters_per_sec"] = prompt_float("Skip models below it/s  (0 = off)", default=0.0)
    while s["min_iters_per_sec"] < 0:
        print(f"  {_c(_RD, '✗')} The it/s threshold cannot be negative.")
        s["min_iters_per_sec"] = prompt_float("Skip models below it/s  (0 = off)", default=0.0)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Run these steps before measuring it/s, so CUDA/JIT/compile startup is ignored.')}")
    s["speed_warmup_steps"] = prompt_int("Throughput warm-up steps", default=5, minimum=0)
    cli_section_end(64)

    # ── Truncated BPTT ─────────────────────────────────────────────────────────
    s["line"] = {"sample_lines": 1}
    s["tbptt"] = {"enabled": False, "window": 0, "total_len": 0}
    line = s["dataset_type"] == 1
    cli_section("Line Mode & TBPTT" if line else "Truncated BPTT", 64)
    if line:
        print(f"  │  {_c(_DIM, 'Each example is one line with BOS/EOS; the window is the longest line.')}")
        print(f"  │  {_c(_DIM, 'TBPTT streams lines through recurrent and scan models in fixed windows,')}")
        print(f"  │  {_c(_DIM, 'carrying state and resetting it at each new line, as training does.')}")
    else:
        print(f"  │  {_c(_DIM, 'TBPTT streams the corpus through recurrent and scan models in fixed')}")
        print(f"  │  {_c(_DIM, 'windows, carrying state between steps, as training does.')}")
    print(f"  │  {_c(_DIM, 'Other models keep their normal batches.')}")
    print(f"  │")
    s["tbptt"]["enabled"] = prompt_str("Enable TBPTT  (y/n)", default="n").lower() in ("y", "yes", "1")
    if s["tbptt"]["enabled"]:
        s["tbptt"]["window"] = prompt_int(
            "BPTT window  (tokens per step" + (", ≤ max line length)" if line else ")"), default=64, minimum=1)
        if not line:
            print(f"  │  {_c(_DIM, 'Total TBPTT length — tokens streamed before every state is reset.')}")
            s["tbptt"]["total_len"] = prompt_int("Total TBPTT length  (0 = until end of data)", default=0, minimum=0)
    cli_section_end(64)

    # ── Performance ────────────────────────────────────────────────────────────
    perf = {"use_amp": False, "amp_dtype": "fp16", "use_compile": False}
    if DEVICE == "cuda":
        cli_section("Performance", 64)
        print(f"  │  {_c(_DIM, 'Applied to every model in the run.')}")
        print(f"  │")
        perf["use_amp"] = prompt_int("Mixed precision / AMP  (0=off 1=on)", valid={0, 1}, default=0) == 1
        if perf["use_amp"]:
            perf["amp_dtype"] = prompt_amp_dtype()
        print(f"  │")
        print(f"  │  {_c(_DIM, 'torch.compile — first steps of each model include compilation;')}")
        print(f"  │  {_c(_DIM, 'raise the warm-up steps so it/s is measured after it.')}")
        perf["use_compile"] = prompt_int("Enable torch.compile  (0=off 1=on)", valid={0, 1}, default=0) == 1
        if perf["use_compile"]:
            cli_opt(0, "aot_eager", "Portable default; traces graphs, no codegen")
            cli_opt(1, "inductor",  "Triton codegen; fastest when the GPU/CUDA stack supports it")
            print(f"  │")
            perf["compile_backend"] = ["aot_eager", "inductor"][
                prompt_int("Compile backend", valid={0, 1}, default=0)
            ]
        cli_section_end(64)
    s["perf"] = perf

    # ── Optimisation ───────────────────────────────────────────────────────────
    # The normal optimizer picker keeps every architecture on the same optimizer
    # and hyperparameters; its learning rate is the base for the options below.
    s["optim"] = prompt_optimizer_config()
    cli_section("Learning Rate & Clipping", 64)
    cli_opt(0, "Constant",        "The optimizer's learning rate throughout")
    cli_opt(1, "Cosine + warmup", "Linear ramp for N steps, then cosine decay to 0")
    cli_opt(2, "Cosine",          "Cosine decay from the start")
    cli_opt(3, "One-cycle",       "Ramp up over 30% of the run, then cosine down")
    print(f"  │  {_c(_DIM, 'With a time limit, progress is measured in elapsed time.')}")
    print(f"  │")
    s["lr_schedule"] = ["none", "cosine_warmup", "cosine", "one_cycle"][
        prompt_int("LR schedule", valid={0, 1, 2, 3}, default=0)]
    s["warmup_steps"] = (prompt_int("Warmup steps", default=100, minimum=0)
                         if s["lr_schedule"] == "cosine_warmup" else 0)
    s["grad_clip"] = prompt_float("Gradient clip norm  (0 = off)", default=1.0)
    print(f"  │")
    s["lr_multipliers"], s["lr_search"] = [1.0], None
    if "lr" not in s["optim"]["optim_params"]:
        pinfo(f"{s['optim']['optimizer']} has no learning rate; LR sweep unavailable")
        lr_mode = 0
    else:
        cli_opt(0, "Base LR",        "Every model uses the optimizer's learning rate")
        cli_opt(1, "Multipliers",    "Try listed multiples of the base LR, keep each model's best")
        cli_opt(2, "Search a range", "Golden-section search for each model's best LR in [min, max]")
        print(f"  │")
        lr_mode = prompt_int("Learning rate per model", valid={0, 1, 2}, default=0)
    if lr_mode == 2:
        print(f"  │  {_c(_DIM, 'Loss vs LR is U-shaped, so each step compares two LRs and drops the')}")
        print(f"  │  {_c(_DIM, 'worse side of the range (log scale): 38% narrower per extra run.')}")
        base = float(s["optim"]["optim_params"]["lr"])
        while True:
            lo = prompt_float("Minimum LR", default=base / 10)
            hi = prompt_float("Maximum LR", default=base * 10)
            if 0 < lo < hi:
                break
            print(f"  {_c(_RD, '✗')} Need 0 < minimum < maximum.")
        halvings = prompt_int("Range halvings  (precision; each halves the log range)", default=4, minimum=1)
        steps = lr_search_steps(halvings)
        width = (hi / lo) ** (0.618034 ** steps)
        pinfo(f"{steps + 2} runs per model; the best LR is found to within ×{width:.2f}")
        s["lr_search"] = {"min": lo, "max": hi, "halvings": halvings, "steps": steps}
    elif lr_mode == 1:
        print(f"  │  {_c(_DIM, 'Each model trains once per multiplier of the base LR and keeps its')}")
        print(f"  │  {_c(_DIM, 'best.  e.g. 0.3,1,3 triples the run time.')}")
        while True:
            raw = prompt_str("LR multipliers  (blank = base LR only)", default="")
            try:
                values = [float(x) for x in raw.split(",") if x.strip()] or [1.0]
                if all(v > 0 for v in values):
                    s["lr_multipliers"] = list(dict.fromkeys(values))
                    break
            except ValueError:
                pass
            print(f"  {_c(_RD, '✗')} Enter positive numbers separated by commas.")
    cli_section_end(64)

    # ── Recurrent structure ────────────────────────────────────────────────────
    s["rnn"] = {}
    structure_ids = ids & BENCH_RNN_STRUCTURE_IDS
    if structure_ids:
        rnn = s["rnn"]
        cli_section("RNN Structure Options", 64)
        print(f"  │  {_c(_DIM, f'Applied to the {len(structure_ids)} selected recurrent cells that support them')}")
        print(f"  │  {_c(_DIM, '(vanilla RNN, GRU, LSTM, IndRNN, JANET, LTC, Mogrifier, UnICORNN, …).')}")
        print(f"  │")
        rnn["res_every"] = prompt_int("Residual every N layers  (0 = off)", default=0, minimum=0)
        rnn["res_type"] = 0
        if rnn["res_every"] > 0:
            cli_opt(0, "Add"); cli_opt(1, "Concat+proj")
            cli_opt(2, "ReZero scalar"); cli_opt(3, "ReZero vector")
            print(f"  │")
            rnn["res_type"] = prompt_int("Residual type", default=0, valid={0,1,2,3})
        print(f"  │")
        print(f"  │  {_c(_DIM, 'Norm:')}")
        cli_opt(0,"None"); cli_opt(1,"BN"); cli_opt(2,"LN")
        cli_opt(3,"RMS");  cli_opt(4,"TTanh"); cli_opt(5,"ETTanh"); cli_opt(6,"DyT")
        print(f"  │")
        rnn["use_norm"] = prompt_int("Norm type", default=2, valid={0,1,2,3,4,5,6})
        rnn["dropout"] = prompt_float("Inter-layer dropout  (0.0 = off)", default=0.0)
        print(f"  │")
        cli_opt(0, "Off"); cli_opt(1, "Scalar"); cli_opt(2, "Per-dim vector")
        rnn["use_multiplier"] = prompt_int("Output multiplier", valid={0, 1, 2}, default=0)
        rnn["rnn_ffn"] = 0
        if ids & BENCH_CUSTOM_RNN_IDS:
            print(f"  │")
            print(f"  │  {_c(_DIM, 'Post-layer feed-forward block (custom cells only; not vanilla RNN/GRU/LSTM):')}")
            cli_opt(0, "Off"); cli_opt(1, "SwiGLU"); cli_opt(2, "ReGLU"); cli_opt(3, "SiLU")
            rnn["rnn_ffn"] = prompt_int("Post-layer FFN", valid={0, 1, 2, 3}, default=0)
        cli_section_end(64)

    # ── Evaluation ─────────────────────────────────────────────────────────────
    cli_section("Evaluation", 64)
    print(f"  │")
    s["nan_skip"] = prompt_int("Skip model on NaN / crash  (1=yes 0=no)", valid={0,1}, default=1) == 1
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Fitness metric:')}")
    cli_opt(0, "Sliding train loss",  "Rolling average of last 100 training steps")
    cli_opt(1, "Final train eval",    "Evaluate 1 000 random training samples at the end")
    cli_opt(2, "Validation loss",     "Best held-out loss tracked throughout training")
    print(f"  │")
    s["fitness_mode"] = prompt_int("Fitness metric", valid={0,1,2})
    s["val"] = {"val_split": 0.0}
    if s["fitness_mode"] == 2:
        if s["dataset_type"] == 0:
            vpath = prompt_str("Validation file  (blank to auto-split)", default="")
            if vpath:
                s["val"]["classic_val_path"] = vpath
            else:
                s["val"]["val_split"] = prompt_float("Validation split fraction", default=0.1)
        else:
            s["val"]["val_split"] = prompt_float("Validation split fraction", default=0.1)
        s["val"]["_val_freq"] = prompt_int("Validation frequency  (steps)", default=100, minimum=1)
        s["val"]["_val_samples"] = prompt_int("Validation samples per check", default=1000, minimum=1)
    print(f"  │")
    print(f"  │  {_c(_DIM, 'Seeds: each model trains this many times; the table shows mean ± spread.')}")
    s["seeds"] = prompt_int("Seeds per model", default=1, minimum=1)
    print(f"  │")
    if s["dataset_type"] == 1:
        s["sample_len"] = prompt_int("Max tokens per sampled line  (0 = no samples)", default=200, minimum=0)
        if s["sample_len"]:
            s["line"]["sample_lines"] = prompt_int("Lines sampled per model", default=3, minimum=1)
    else:
        s["sample_len"] = prompt_int("Sample tokens generated per model  (0 = off)", default=200, minimum=0)
    s["sample_temperature"] = (prompt_float("Sample temperature", default=0.8)
                               if s["sample_len"] else 0.8)
    print(f"  │")
    cli_opt(0, "Ranked",   "Best score first")
    cli_opt(1, "Timeline", "Oldest first, with records and improvement over the best so far")
    cli_opt(2, "Both")
    s["table_order"] = prompt_int("Results table", valid={0, 1, 2}, default=2)
    cli_section_end(64)
    return s


def _bench_legacy_id(model_id):
    if model_id not in LEGACY_MODEL_IDS:
        raise ValueError(f"Version-1 benchmark file names unknown model ID {model_id}")
    return LEGACY_MODEL_IDS[model_id]


def _bench_normalize_settings(s):
    """Undo JSON's changes: int dict keys and tuples.  Version-1 settings
    predate the model-ID renumbering and are translated in place."""
    if s.get("version") == 1:
        for task in s["tasks"]:
            task["id"] = _bench_legacy_id(task["id"])
        s["activations"] = {str(_bench_legacy_id(int(k))): v for k, v in s.get("activations", {}).items()}
        s["version"] = BENCH_SETTINGS_VERSION
    if s.get("version") != BENCH_SETTINGS_VERSION:
        raise ValueError(f"Unsupported benchmark settings version {s.get('version')!r}")
    line = s.setdefault("line", {})
    line.setdefault("sample_lines", 1)
    # Settings from before corpus TBPTT kept the line-mode switch under "line".
    s.setdefault("tbptt", {"enabled": bool(line.get("use_tbptt")), "window": int(line.get("bptt_window", 0)),
                           "total_len": 0})
    s.setdefault("lr_search", None)
    s.setdefault("seq2seq", None)
    s["optim"]["optim_params"] = {
        k: tuple(v) if isinstance(v, list) else v for k, v in s["optim"]["optim_params"].items()
    }
    unknown = sorted({t["id"] for t in s["tasks"]} - set(MODEL_IDS))
    if unknown:
        raise ValueError(f"Settings name unknown model IDs {unknown}")
    return s


def _bench_read_json(path):
    path = pathlib.Path(path).expanduser()
    if path.is_dir():
        path = path / "results.json"
    with open(path, "r", encoding="utf-8") as f:
        return path, json.load(f)


def _bench_task_key(task):
    return json.dumps([task["id"], task["overrides"]], sort_keys=True)


# ------------------------------------------------------------------ one model
def _bench_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _bench_task_cfg(s, task, cfg_base):
    mid, overrides = task["id"], task["overrides"]
    cfg = copy.deepcopy(cfg_base)
    cfg.update({
        "model_selection": mid,
        "embed_dim": s["embed_dim"],
        "layer_count": s["layer_count"],
        "head_count": s["head_count"],
        "batch_size": s["batch_size"],
        "learning_rate": s["optim"]["optim_params"].get("lr", 1e-3),
        "activation_name": s["activations"].get(str(mid), "gelu"),
        "dropout": 0.0,
        "tokenizer_mode": s["tokenizer_mode"],
        "optimizer": s["optim"]["optimizer"],
        "optim_params": dict(s["optim"]["optim_params"]),
        "temperature": s.get("sample_temperature", 0.8),
        "use_tbptt": s["tbptt"]["enabled"],
        "bptt_window": s["tbptt"]["window"],
        "tbptt_total_len": s["tbptt"]["total_len"],
        # Stable samples make final-train and repeated validation scores
        # directly comparable across checkpoints and models.
        "_bench_train_eval_seed": SEED + 10_001,
        "_bench_valid_seed": SEED + 10_002,
        **s["perf"],
    })
    rnn = s["rnn"]
    if mid in BENCH_RNN_STRUCTURE_IDS and rnn:
        cfg.update({k: rnn[k] for k in ("use_norm", "res_every", "res_type", "dropout", "use_multiplier")})
        if mid in BENCH_CUSTOM_RNN_IDS:
            cfg["rnn_ffn"] = rnn["rnn_ffn"]
    cfg["rnn_cell_options"] = dict(overrides.get("rnn_cell_options", {}))
    cfg.update({k: v for k, v in overrides.items() if k != "rnn_cell_options"})

    h = s["hierarchy"]
    if h["model_type"] != MODEL_TYPE_NORMAL:
        n = len(h["stage_dims"])
        heads = [hd if megabyte_mixer_uses_heads(mid) else 1 for hd in h["stage_heads"]]
        cfg.update({
            "model_type": h["model_type"],
            "megabyte_stage_mixers": [mid] * n,
            "megabyte_stage_dims": list(h["stage_dims"]),
            "megabyte_stage_depths": list(h["stage_depths"]),
            "megabyte_stage_heads": heads,
            "megabyte_stage_seq_lens": list(h["stage_seq_lens"]),
            "megabyte_stage_child_embed_dims": list(h["stage_child_embed_dims"]),
            "embed_dim": h["stage_dims"][0],
            "layer_count": sum(h["stage_depths"]),
            "head_count": heads[0],
            "seq_len": math.prod(h["stage_seq_lens"]),
        })
        if resolve_megabyte_stage_mixer(mid) in {"rnn", "rnn_relu", "gru", "lstm"}:
            cfg.update({
                "megabyte_fused_rnn_version": 2,
                "megabyte_fused_rnn_norm_type": rnn.get("use_norm", 0),
                "megabyte_fused_rnn_res_every": rnn.get("res_every", 0),
                "megabyte_fused_rnn_res_type": rnn.get("res_type", 0),
                "megabyte_fused_rnn_dropout": rnn.get("dropout", 0.0),
            })
        if h["model_type"] == MODEL_TYPE_MEGABYTE_BOTTOM_UP:
            cfg["megabyte_bottom_up_version"] = 6
    return cfg


def _bench_model_name(s, task, cfg):
    mid = task["id"]
    name = MODEL_NAMES.get(mid, f"Model {mid}") + _bench_variant_label(task)
    if mid in NON_RNN_ACTIVATION_IDS:
        name += f" [act={cfg['activation_name']}]"
    if mid in BENCH_RNN_STRUCTURE_IDS and s["rnn"]:
        norm_name = ["None","BN","LN","RMS","TT","ETT","DyT"][cfg.get("use_norm", 0)]
        res_str = f"|Res{cfg.get('res_every', 0)}" if cfg.get("res_every", 0) > 0 else ""
        ffn_str = f"|{['', 'SwiGLU', 'ReGLU', 'SiLU'][cfg.get('rnn_ffn', 0)]}" if cfg.get("rnn_ffn", 0) else ""
        name += f" [{norm_name}{res_str}{ffn_str}]"
    return name


def _bench_param_count(cfg, vocab_size):
    """Parameter count without allocating: build on the meta device, falling
    back to a real build for constructors that compute with real tensors."""
    global DEVICE
    saved = DEVICE
    try:
        DEVICE = "meta"
        with torch.device("meta"):
            model = build_model(copy.deepcopy(cfg), vocab_size)
        return sum(p.numel() for p in model.parameters())
    except Exception:
        pass
    finally:
        DEVICE = saved
    model = None
    try:
        model = build_model(copy.deepcopy(cfg), vocab_size)
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return None
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def fit_width_to_params(cfg, vocab_size, target, head_count):
    """Hidden size (a multiple of lcm(8, heads)) whose parameter count is
    closest to ``target``.  Returns (dim, params); params is None if no width
    could be built."""
    step = math.lcm(8, max(1, int(head_count)))
    counts = {}

    def count(k):
        if k not in counts:
            counts[k] = _bench_param_count(dict(cfg, embed_dim=k * step), vocab_size)
        return counts[k]

    lo, hi = 1, max(1, 8192 // step)
    while lo <= hi:
        mid = (lo + hi) // 2
        probe = mid
        while count(probe) is None and probe < min(hi, mid + 3):
            probe += 1
        if count(probe) is None:
            hi = mid - 1
        elif count(probe) < target:
            lo = probe + 1
        else:
            hi = mid - 1
    valid = [(abs(n - target), k) for k, n in counts.items() if n is not None]
    if not valid:
        return cfg["embed_dim"], None
    _, k = min(valid)
    return k * step, counts[k]


_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0     # 0.618…: range kept per search step


def lr_search_steps(halvings):
    """Golden-section steps giving at least ``halvings`` halvings of the range."""
    return max(1, math.ceil(halvings * math.log(2.0) / -math.log(_GOLDEN)))


def lr_golden_search(evaluate, lr_min, lr_max, steps):
    """Golden-section search for the lowest score over log(LR) in
    [lr_min, lr_max].  ``evaluate(lr)`` returns a dict with a ``score`` (inf
    for a failed run, which steers the search away from it).  Assumes loss is
    roughly U-shaped in log(LR); every LR tried is kept by the caller, so the
    reported best is the best actually trained, not an interpolation.
    Costs ``steps + 2`` evaluations."""
    a, b = math.log(lr_min), math.log(lr_max)
    c, d = b - _GOLDEN * (b - a), a + _GOLDEN * (b - a)
    fc, fd = evaluate(math.exp(c))["score"], evaluate(math.exp(d))["score"]
    for _ in range(steps):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN * (b - a)
            fc = evaluate(math.exp(c))["score"]
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN * (b - a)
            fd = evaluate(math.exp(d))["score"]


def bench_lr_factor(kind, progress, step, warmup_steps):
    """Multiplier on the base LR; ``progress`` runs 0→1 over the run."""
    p = min(max(progress, 0.0), 1.0)
    if kind == "cosine_warmup":
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 0.5 * (1.0 + math.cos(math.pi * p))
    if kind == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * p))
    if kind == "one_cycle":
        if p < 0.3:
            return 0.04 + 0.96 * p / 0.3
        return 0.5 * (1.0 + math.cos(math.pi * (p - 0.3) / 0.7))
    return 1.0


def bench_train_loop(cfg, model, optimizer, train_ds, valid_ds, vocab, line_mode,
                     total_iters, fitness_mode, nan_skip, *, max_seconds=None,
                     min_iters_per_sec=0.0, speed_warmup_steps=5, stats=None,
                     lr_schedule="none", warmup_steps=0, grad_clip=1.0):
    """
    Returns: (score, best_step, status_string)
    ``total_iters`` caps the run by steps.  Set ``max_seconds`` to cap it by
    elapsed wall-clock time instead.  A positive ``min_iters_per_sec`` skips a
    model after the warm-up throughput falls below that threshold.
    ``cfg["use_amp"]`` / ``cfg["amp_dtype"]`` enable autocast as in training.
    ``lr_schedule`` scales every param group's starting LR by bench_lr_factor;
    ``grad_clip`` <= 0 disables clipping.
    Pass a dict as ``stats`` to receive ``steps``, post-warm-up ``it_s``, and
    the ``curve`` / ``val_curve`` lists of (step, loss).
    """
    if total_iters <= 0 and (max_seconds is None or max_seconds <= 0):
        raise ValueError("Benchmark needs a positive maximum step count or time limit.")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("Benchmark time limit must be greater than zero.")
    if min_iters_per_sec < 0:
        raise ValueError("Minimum iterations per second cannot be negative.")
    if speed_warmup_steps < 0:
        raise ValueError("Speed-filter warm-up steps cannot be negative.")
    if fitness_mode == 2 and valid_ds is None:
        raise ValueError("Validation-loss benchmarking requires a validation dataset.")

    # === FIX START: Handle None pad_id explicitly ===
    pad_id = getattr(vocab, "pad_id", -100)
    if pad_id is None:
        pad_id = -100

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    # === FIX END ===
    model.train()
    use_amp, amp_dtype, scaler = amp_settings(cfg)
    if stats is None:
        stats = {}
    stats.update(steps=0, it_s=0.0, curve=[], val_curve=[])
    base_lrs = [group.get("lr") for group in optimizer.param_groups]
    curve_every = 1 if max_seconds is not None else max(1, total_iters // 200)

    losses = []
    best_valid_loss = float('inf')
    best_valid_step = 0

    # Validation settings from config
    val_freq = cfg.get("_val_freq", 100)
    if fitness_mode == 2 and val_freq <= 0:
        raise ValueError("Validation frequency must be greater than zero.")

    # Simple batch fetcher
    def get_batch(ds):
        if hasattr(ds, 'get_batch'): return ds.get_batch(cfg["batch_size"])
        # Fallback for old style datasets if any
        return ds[0], ds[1] # dummy

    # Progress bar
    pbar_desc = f"[bench] {MODEL_NAMES.get(cfg['model_selection'], cfg['model_selection'])}"
    if "minrnn_act" in cfg: pbar_desc += f" (act={cfg['minrnn_act']})"

    started_at = time.perf_counter()
    speed_started_at = started_at
    completed_steps = 0
    last_validation_step = 0
    pbar_total = None if max_seconds is not None else total_iters

    # TBPTT, exactly as train_loop streams it: only recurrent and scan models
    # carry state; every other model keeps its normal batches.
    eager = getattr(model, "_orig_mod", model)
    is_bottom_up = is_bottom_up_megabyte(cfg)
    msel = cfg["model_selection"]
    line_stream = None
    if cfg.get("use_tbptt") and (msel in SCAN_MODEL_IDS or (msel in RNN_MODEL_IDS and not is_bottom_up)):
        if line_mode:
            line_stream = LineTBPTTStream(
                dataset=train_ds, window=int(cfg["bptt_window"]), batch_size=cfg["batch_size"],
                bos_id=vocab.bos_id, pad_id=vocab.pad_id)
        else:
            line_stream = TBPTTClassicStream(
                train_ds.ids if hasattr(train_ds, "ids") else train_ds.data,
                window=int(cfg["bptt_window"]), batch_size=cfg["batch_size"],
                total_len=int(cfg.get("tbptt_total_len", 0)))
    stats["tbptt"] = line_stream is not None
    rnn_state = None
    with tqdm(total=pbar_total, desc=pbar_desc, leave=False) as pbar:
        while True:
            if total_iters > 0 and completed_steps >= total_iters:
                break
            if max_seconds is not None and completed_steps > 0:
                if time.perf_counter() - started_at >= max_seconds:
                    break

            i = completed_steps + 1
            try:
                if lr_schedule != "none":
                    progress = (completed_steps / total_iters if max_seconds is None
                                else (time.perf_counter() - started_at) / max_seconds)
                    factor = bench_lr_factor(lr_schedule, progress, completed_steps, warmup_steps)
                    for group, base in zip(optimizer.param_groups, base_lrs):
                        if base is not None:
                            group["lr"] = base * factor

                if line_stream is not None:
                    x, y, reset_mask = line_stream.get_next(DEVICE)
                else:
                    x, y = get_batch(train_ds)

                # Forward
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    if line_stream is not None:
                        if rnn_state is None and isinstance(eager, MegaByteLM) and eager.is_incremental:
                            rnn_state = eager.init_incremental_cache(x)
                        rnn_state = detach_state(rnn_state)
                        if rnn_state is not None and reset_mask is not None:
                            rnn_state = reset_rnn_state(rnn_state, reset_mask, eager, msel)
                        logits, rnn_state = model(x, rnn_state)
                    elif cfg["model_selection"] in RNN_MODEL_IDS and not is_bottom_up_megabyte(cfg):
                        out = model(x, None)
                        logits = out[0]
                    elif cfg["model_selection"] in SCAN_MODEL_IDS:
                        out = model(x) # ScanLM handles None state internally usually
                        logits = out[0] if isinstance(out, tuple) else out
                    else:
                        out = model(x)
                        logits = out[0] if isinstance(out, tuple) else out

                    loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                    aux_loss = getattr(model, "aux_loss", None)
                    if aux_loss is not None:
                        loss = loss + float(cfg.get("moe_aux_loss_weight", 0.01)) * aux_loss

                # A seq2seq TBPTT window wholly inside masked sources has no
                # targets (mean loss NaN); its forward already advanced the
                # carried state, so fetch the next window without a step.
                if not bool((y != pad_id).any()):
                    continue

                # NaN Check
                if torch.isnan(loss) or torch.isinf(loss):
                    if nan_skip:
                        return float('inf'), i, "NaN"
                    else:
                        # If not skipping, we must zero grad and maybe try to recover,
                        # but usually optimization is broken. We'll just log high loss.
                        loss = torch.tensor(100.0, device=DEVICE, requires_grad=True)

                optimizer.zero_grad(set_to_none=True)
                if hasattr(optimizer, "observe_loss"):   # HD optimizers' divergence guard
                    optimizer.observe_loss(loss.item())
                if scaler is not None:
                    scaler.scale(loss).backward()
                    # Clip true gradient magnitudes, not GradScaler-scaled ones.
                    scaler.unscale_(optimizer)
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                # Record
                l_val = loss.item()
                losses.append(l_val)
                if len(losses) > 100: losses.pop(0)

                # Update pbar
                avg_train = sum(losses)/len(losses)
                completed_steps = i
                if completed_steps % curve_every == 0:
                    stats["curve"].append((completed_steps, round(avg_train, 5)))
                    if len(stats["curve"]) > 400:
                        # Keep a time-limited run's curve bounded.
                        stats["curve"] = stats["curve"][1::2]
                        curve_every *= 2
                if completed_steps == speed_warmup_steps:
                    speed_started_at = time.perf_counter()
                measured_steps = completed_steps - speed_warmup_steps
                measured_elapsed = max(time.perf_counter() - speed_started_at, 1e-9)
                measured_iters_per_sec = measured_steps / measured_elapsed
                rate_label = "warming" if measured_steps <= 0 else f"{measured_iters_per_sec:.2f}"
                stats["steps"] = completed_steps
                if measured_steps > 0:
                    stats["it_s"] = measured_iters_per_sec
                pbar.set_postfix(loss=f"{l_val:.4f}", avg=f"{avg_train:.4f}",
                                 it_s=rate_label)
                pbar.update(1)

                if (min_iters_per_sec > 0 and measured_steps > 0
                        and measured_iters_per_sec < min_iters_per_sec):
                    return float("inf"), completed_steps, f"SLOW ({measured_iters_per_sec:.2f} it/s)"
                if stats.get("stop"):   # set by a GUI through BENCH_RUN_HOOK
                    return float("inf"), completed_steps, "STOPPED"

                # Validation Logic (Fitness Mode 2)
                if fitness_mode == 2 and valid_ds is not None:
                    if i % val_freq == 0:
                        with schedule_free_eval(optimizer):
                            vloss = eval_valid_loss(
                                model, cfg, valid_ds, vocab, line_mode,
                                max_samples=cfg.get("_val_samples", 1000),
                                seed=cfg.get("_bench_valid_seed"),
                            )
                        if vloss is not None:
                            last_validation_step = i
                            stats["val_curve"].append((i, round(vloss, 5)))
                            if vloss < best_valid_loss:
                                best_valid_loss = vloss
                                best_valid_step = i

            except RuntimeError as e:
                if "out of memory" in str(e):
                    return float('inf'), i, "OOM"
                if nan_skip:
                    return float('inf'), i, f"CRASH: {e}"
                raise e

    if losses and (not stats["curve"] or stats["curve"][-1][0] != completed_steps):
        stats["curve"].append((completed_steps, round(sum(losses) / len(losses), 5)))

    # === Final Scoring ===
    score = float('inf')

    if fitness_mode == 0: # Sliding Avg
        score = sum(losses) / max(1, len(losses))

    elif fitness_mode == 1: # Final Train Eval
        # Reuse the token-weighted evaluator.  In particular, line-mode batches
        # have different amounts of padding, so averaging ten batch means would
        # over-weight short examples.  It also preserves ScanLM's parallel path.
        with schedule_free_eval(optimizer):
            score = eval_valid_loss(
                model, cfg, train_ds, vocab, line_mode,
                max_samples=cfg.get("_final_eval_samples", 1000),
                seed=cfg.get("_bench_train_eval_seed"),
            )
        if score is None:
            return float("inf"), 0, "NO EVAL TOKENS"

    elif fitness_mode == 2: # Best Valid Loss
        # Perform one last check if the final completed step was not validated.
        if last_validation_step != completed_steps:
            with schedule_free_eval(optimizer):
                vloss = eval_valid_loss(
                    model, cfg, valid_ds, vocab, line_mode,
                    max_samples=cfg.get("_val_samples", 1000),
                    seed=cfg.get("_bench_valid_seed"),
                )
            if vloss is not None:
                stats["val_curve"].append((completed_steps, round(vloss, 5)))
            if vloss is not None and vloss < best_valid_loss:
                best_valid_loss = vloss
                best_valid_step = completed_steps

        score = best_valid_loss
        if score == float('inf'):
            return float("inf"), best_valid_step, "NO VALID TOKENS"

    return score, best_valid_step, "OK"


def _bench_sample(model, optimizer, cfg, vocab, prompt_ids, length, line_mode, lines=1):
    """Decoded continuation of the shared prompt (``lines`` generated lines in
    line mode), or a note if sampling fails."""
    _bench_seed(SEED)
    try:
        with torch.no_grad(), schedule_free_eval(optimizer):
            if line_mode and prompt_ids:
                # Seq2seq: the outputs generated for the shared held-out input.
                return "\n".join(
                    seq2seq_visible(vocab.decode(
                        generate_line_mode(model, cfg, vocab, list(prompt_ids), length)[len(prompt_ids):]))
                    for _ in range(max(1, lines))
                )
            if line_mode:
                return "\n".join(
                    vocab.decode(generate_line_mode(model, cfg, vocab, [vocab.bos_id], length))
                    for _ in range(max(1, lines))
                )
            ids = generate_classic(model, cfg, vocab, list(prompt_ids), length, stream=False)
            return vocab.decode(ids[len(prompt_ids):])
    except Exception as exc:
        return f"<sampling failed: {exc}>"
    finally:
        model.train()


# Optional observer for GUIs: BENCH_RUN_HOOK(run, stats) is called as each
# training run starts.  ``stats`` is the live dict bench_train_loop fills in
# (steps, it_s, curve, val_curve); setting stats["stop"] = True ends the run
# with status "STOPPED" at the next step.
BENCH_RUN_HOOK = None


def _bench_single_run(s, cfg, env, lr, seed):
    cfg = copy.deepcopy(cfg)
    if lr is not None:
        cfg["optim_params"]["lr"] = lr
        cfg["learning_rate"] = lr
    _bench_seed(seed)
    run = {"seed": seed, "lr": lr, "score": float("inf"), "best_step": -1, "status": "CRASH",
           "params": 0, "it_s": 0.0, "curve": [], "val_curve": [], "sample": ""}
    model = optimizer = None
    try:
        model = build_model(cfg, env["vocab"].size)
        model.to(DEVICE)
        run["params"] = sum(p.numel() for p in model.parameters())
        # Build the optimizer on the eager module; the compiled wrapper
        # shares its parameters.
        optimizer = build_optimizer(model, cfg)
        stats = {}
        if BENCH_RUN_HOOK is not None:
            BENCH_RUN_HOOK(run, stats)
        score, best_step, status = bench_train_loop(
            cfg, wrap_model_with_compile(model, cfg), optimizer, env["train_ds"], env["valid_ds"],
            env["vocab"], env["line_mode"], s["total_iters"], s["fitness_mode"], s["nan_skip"],
            max_seconds=s["max_seconds"], min_iters_per_sec=s["min_iters_per_sec"],
            speed_warmup_steps=s["speed_warmup_steps"], stats=stats,
            lr_schedule=s["lr_schedule"], warmup_steps=s["warmup_steps"], grad_clip=s["grad_clip"],
        )
        run["tbptt"] = stats.get("tbptt", False)
        run.update(score=score, best_step=best_step, status=status, it_s=stats.get("it_s", 0.0),
                   curve=stats.get("curve", []), val_curve=stats.get("val_curve", []))
        if status == "OK" and s["sample_len"] > 0:
            run["sample"] = _bench_sample(model, optimizer, cfg, env["vocab"], env["prompt_ids"],
                                          s["sample_len"], env["line_mode"], s["line"]["sample_lines"])
    except Exception as e:
        run["status"] = f"CRASH: {e}"
    finally:
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return run


def _bench_run_task(s, task, env):
    """Train one row over every LR multiplier and seed; keep the best LR."""
    cfg = _bench_task_cfg(s, task, env["cfg_base"])
    record = {
        "key": _bench_task_key(task), "id": task["id"], "overrides": task["overrides"],
        "year": MODEL_ORIGINS[task["id"]][0], "month": MODEL_ORIGINS[task["id"]][1],
        "name": _bench_model_name(s, task, cfg), "embed_dim": cfg["embed_dim"], "matched_params": None,
    }
    if s["size_mode"] == "params":
        record["embed_dim"], record["matched_params"] = fit_width_to_params(
            cfg, env["vocab"].size, s["target_params"], cfg["head_count"])
        cfg["embed_dim"] = record["embed_dim"]
    base_lr = cfg["optim_params"].get("lr")
    candidates = []

    def evaluate(lr, mult):
        runs = []
        for k in range(s["seeds"]):
            run = _bench_single_run(s, cfg, env, lr, SEED + k)
            runs.append(run)
            if run["status"] != "OK":
                break
        ok = all(r["status"] == "OK" for r in runs)
        scores = [r["score"] for r in runs] if ok else []
        candidates.append({
            "lr_mult": mult, "lr": lr, "runs": runs, "ok": ok,
            "score": float(np.mean(scores)) if ok else float("inf"),
            "score_std": float(np.std(scores)) if ok and len(scores) > 1 else 0.0,
        })
        return candidates[-1]

    if s.get("lr_search") and base_lr is not None:
        search = s["lr_search"]
        lr_golden_search(lambda lr: evaluate(lr, lr / base_lr), search["min"], search["max"], search["steps"])
    else:
        for mult in (s["lr_multipliers"] if base_lr is not None else [1.0]):
            first = evaluate(None if base_lr is None else base_lr * mult, mult)
            if first["runs"][0]["status"].startswith("SLOW"):
                break  # a different LR does not change the speed
    best = min(candidates, key=lambda c: (not c["ok"], c["score"]))
    runs = best["runs"]
    failed = next((r for r in runs if r["status"] != "OK"), None)
    lead = min(runs, key=lambda r: r["score"])
    record.update({
        "score": best["score"], "score_std": best["score_std"],
        "scores": [r["score"] for r in runs], "status": failed["status"] if failed else "OK",
        "best_step": lead["best_step"], "lr": best["lr"], "lr_mult": best["lr_mult"],
        "params": runs[0]["params"], "it_s": float(np.mean([r["it_s"] for r in runs])),
        "curve": lead["curve"], "val_curve": lead["val_curve"], "sample": lead["sample"],
        "tbptt": lead.get("tbptt", False),
        "lr_candidates": [{"lr": c["lr"], "lr_mult": c["lr_mult"], "score": c["score"], "ok": c["ok"],
                           "status": next((r["status"] for r in c["runs"] if r["status"] != "OK"), "OK")}
                          for c in candidates],
    })
    return record


# ------------------------------------------------------------------ outputs
def _bench_json_safe(value):
    """JSON has no infinity; failed scores are stored as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _bench_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bench_json_safe(v) for v in value]
    return value


def _bench_json_restore(record):
    for key in ("score", "score_std"):
        if record.get(key) is None:
            record[key] = float("inf") if key == "score" else 0.0
    record["scores"] = [float("inf") if v is None else v for v in record.get("scores", [])]
    return record


def _bench_fmt_score(r):
    if r["score"] == float("inf"):
        return "∞"
    if len(r.get("scores", [])) > 1:
        return f"{r['score']:.4f}±{r['score_std']:.3f}"
    return f"{r['score']:.5f}"


def _bench_save(run_dir, s, records, prompt_text=""):
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / "results.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_bench_json_safe({"settings": s, "records": records}), f, indent=1)
    os.replace(tmp, run_dir / "results.json")
    with open(run_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year", "model_id", "name", "score", "score_std", "seeds", "status", "best_step",
                    "params", "embed_dim", "lr", "lr_mult", "it_s"])
        for r in sorted(records, key=lambda r: (MODEL_ORIGINS[r["id"]], r["id"])):
            w.writerow([r["year"], r["id"], r["name"],
                        "" if r["score"] == float("inf") else f"{r['score']:.6f}",
                        f"{r['score_std']:.6f}", len(r["scores"]), r["status"], r["best_step"],
                        r["params"], r["embed_dim"], "" if r["lr"] is None else f"{r['lr']:.6g}",
                        r["lr_mult"], f"{r['it_s']:.3f}"])
    if s["sample_len"]:
        with open(run_dir / "samples.txt", "w", encoding="utf-8") as f:
            f.write(f"Prompt (shared by every model):\n{prompt_text}\n\n")
            for r in sorted(records, key=lambda r: (MODEL_ORIGINS[r["id"]], r["id"])):
                if r.get("sample"):
                    f.write(f"=== {r['year']}  {r['name']}  ({_bench_fmt_score(r)})\n{r['sample']}\n\n")
    try:
        _bench_plot(run_dir, s, records)
    except Exception as exc:  # plotting is a convenience; never lose the results for it
        pwarn(f"Plots not written: {exc}")


# Chart palette: a one-hue blue ramp for year (light = older, dark = newer)
# on a light chart surface with recessive grid and axis ink.
_BENCH_YEAR_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
_BENCH_SURFACE, _BENCH_GRID = "#fcfcfb", "#e1e0d9"
_BENCH_INK, _BENCH_INK2, _BENCH_MUTED = "#0b0b0b", "#52514e", "#898781"


def _bench_plot(run_dir, s, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    ok = [r for r in records if r["status"] == "OK"]
    if not ok:
        return
    cmap = LinearSegmentedColormap.from_list("bench_years", _BENCH_YEAR_RAMP)
    years = [r["year"] for r in ok]
    norm = Normalize(min(years), max(max(years), min(years) + 1))
    metric = ["Sliding train loss", "Final train eval", "Validation loss"][s["fitness_mode"]]

    def style(ax, title, xlabel, ylabel):
        ax.set_facecolor(_BENCH_SURFACE)
        ax.grid(True, color=_BENCH_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_BENCH_MUTED)
        ax.tick_params(colors=_BENCH_INK2, labelsize=9)
        ax.set_title(title, color=_BENCH_INK, fontsize=12, loc="left")
        ax.set_xlabel(xlabel, color=_BENCH_INK2)
        ax.set_ylabel(ylabel, color=_BENCH_INK2)

    curve_sets = [("curves.png", "curve", "Training loss (100-step average)")]
    if any(r.get("val_curve") for r in ok):
        curve_sets.append(("val_curves.png", "val_curve", "Validation loss"))
    for filename, key, ylabel in curve_sets:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=_BENCH_SURFACE)
        for r in sorted(ok, key=lambda r: r["year"]):
            pts = r.get(key) or []
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=cmap(norm(r["year"])),
                        linewidth=1.4, alpha=0.9)
        style(ax, f"{ylabel} per model, colored by year", "Step", ylabel)
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
        bar.set_label("Year first published", color=_BENCH_INK2)
        bar.ax.tick_params(colors=_BENCH_INK2, labelsize=9)
        bar.outline.set_visible(False)
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=130, facecolor=_BENCH_SURFACE)
        plt.close(fig)

    # Timeline: every model's score by publication date, and the best so far.
    order = sorted(ok, key=lambda r: (MODEL_ORIGINS[r["id"]], r["id"]))
    xs = [r["year"] + (r["month"] - 1) / 12 for r in order]
    ys = [r["score"] for r in order]
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=_BENCH_SURFACE)
    errs = [r["score_std"] for r in order]
    if any(errs):
        ax.errorbar(xs, ys, yerr=errs, fmt="none", ecolor=_BENCH_MUTED, elinewidth=1, capsize=0, zorder=1)
    ax.scatter(xs, ys, s=36, facecolors=_BENCH_SURFACE, edgecolors=_BENCH_MUTED, linewidths=1.2, zorder=2,
               label="Model")
    best, fx, fy, records_idx = float("inf"), [], [], []
    for i, (x, y) in enumerate(zip(xs, ys)):
        if y < best:
            best = y
            records_idx.append(i)
        fx.append(x); fy.append(best)
    ax.step(fx, fy, where="post", color="#184f95", linewidth=2, zorder=3, label="Best so far")
    ax.scatter([xs[i] for i in records_idx], [ys[i] for i in records_idx], s=56, color="#2a78d6",
               edgecolors=_BENCH_SURFACE, linewidths=2, zorder=4, label="New record")
    for i in records_idx:
        label = _bench_short_name(order[i], 34)
        ax.annotate(label, (xs[i], ys[i]), xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color=_BENCH_INK2)
    style(ax, f"{metric} by year of publication (lower is better)", "Year first published", metric)
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(_BENCH_INK2)
    fig.tight_layout()
    fig.savefig(run_dir / "timeline.png", dpi=130, facecolor=_BENCH_SURFACE)
    plt.close(fig)


def _bench_print_tables(s, records, perf_label):
    metric_lbl = ["Sliding Loss","Final Train Eval","Best Valid Loss"][s["fitness_mode"]]
    show_dim = s["size_mode"] == "params"
    show_lr = len(s["lr_multipliers"]) > 1 or bool(s.get("lr_search"))
    name_w = 52

    def cols(r):
        name = r["name"] if len(r["name"]) <= name_w else _bench_short_name(r, name_w)
        extra = ""
        if show_dim:
            extra += f"   {r['embed_dim']:>5}"
        if show_lr:
            extra += f"   {'-' if r['lr'] is None else format(r['lr'], '.2g'):>7}"
        par = f"{r['params'] / 1e6:>7.2f}M" if r["params"] else f"{'-':>8}"
        its = f"{r['it_s']:>8.2f}" if r["it_s"] else f"{'-':>8}"
        return name, extra, par, its

    extra_hdr = (f"   {'Dim':>5}" if show_dim else "") + (f"   {'LR':>7}" if show_lr else "")
    W = 120 + len(extra_hdr)

    if s["table_order"] in (0, 2):
        ranked = sorted(records, key=lambda x: (x["score"], 0 if x["status"] == "OK" else 1))
        print(f"\n")
        cli_banner(f"Benchmark Results  ·  {metric_lbl}  ·  {perf_label}", width=W)
        hdr = f"  {'Rank':<4}   {'Year':<4}   {'Model':<{name_w}}   {'Score':<15}{extra_hdr}   {'Params':>8}   {'it/s':>8}   Status"
        print(f"  {_c(_DIM, hdr)}")
        cli_rule(W - 4)
        for i, r in enumerate(ranked):
            name, extra, par, its = cols(r)
            score = _bench_fmt_score(r)
            rank_col = _c(_YL, _B, f"  {i+1:<4}") if i == 0 else f"  {i+1:<4}"
            name_col = _c(_WH, f"{name:<{name_w}}") if i == 0 else f"{name:<{name_w}}"
            scr_col = (_c(_GR, _B, f"{score:<15}") if i == 0 else
                       _c(_RD, f"{score:<15}") if r["status"] != "OK" else f"{score:<15}")
            stat_col = _c(_GR, r["status"]) if r["status"] == "OK" else _c(_RD, r["status"])
            print(f"  {rank_col}   {r['year']:<4}   {name_col}   {scr_col}{extra}   {par}   {its}   {stat_col}")
        cli_rule(W - 4)
        if ranked and ranked[0]["score"] != float('inf'):
            best = ranked[0]
            print(f"\n  {_c(_GR, '★')} Winner: {_c(_WH, _B, best['name'])}  ·  score {_c(_GR, _B, _bench_fmt_score(best))}\n")

    if s["table_order"] in (1, 2):
        timeline = sorted(records, key=lambda r: (MODEL_ORIGINS[r["id"]], r["id"]))
        print(f"\n")
        cli_banner(f"Timeline  ·  {metric_lbl}  ·  ★ = new best so far", width=W)
        hdr = f"  {'Year':<4}   {'Model':<{name_w}}   {'Score':<15}   {'vs best':>8}{extra_hdr}   {'Params':>8}   {'it/s':>8}   Status"
        print(f"  {_c(_DIM, hdr)}")
        cli_rule(W - 4)
        best = float("inf")
        for r in timeline:
            name, extra, par, its = cols(r)
            score = _bench_fmt_score(r)
            if r["status"] == "OK" and r["score"] < best:
                delta = "" if best == float("inf") else f"{r['score'] - best:+.4f}"
                best = r["score"]
                mark, score_col = _c(_YL, _B, "★"), _c(_GR, _B, f"{score:<15}")
            else:
                delta = "" if r["status"] != "OK" or best == float("inf") else f"{r['score'] - best:+.4f}"
                mark, score_col = " ", (_c(_RD, f"{score:<15}") if r["status"] != "OK" else f"{score:<15}")
            stat_col = _c(_GR, r["status"]) if r["status"] == "OK" else _c(_RD, r["status"])
            print(f"  {r['year']:<4} {mark} {name:<{name_w}}   {score_col}   {delta:>8}{extra}   {par}   {its}   {stat_col}")
        cli_rule(W - 4)


# ------------------------------------------------------------------ run
def _bench_prepare(s):
    """Vocabulary, datasets and the shared sampling prompt for a settings dict."""
    cfg_dummy = {"dataset_type": s["dataset_type"], "tokenizer_mode": s["tokenizer_mode"],
                 "custom_bpe_size": 4096}
    if s["tokenizer_mode"] == 3:
        cfg_dummy["tiktoken_encoding"] = "cl100k_base"
    seq2seq = s.get("seq2seq")
    data_path = prepare_seq2seq_dataset(seq2seq) if seq2seq else s["dataset_path"]
    vocab = load_or_make_vocab(cfg_dummy, data_path, save_config=False)

    seq_len = s["seq_len"]
    if s["hierarchy"]["model_type"] != MODEL_TYPE_NORMAL:
        seq_len = math.prod(s["hierarchy"]["stage_seq_lens"])
    cfg_base = {
        "dataset_path": data_path,
        "dataset_type": s["dataset_type"],
        "seq_len": seq_len,
        "vocab_tokens": getattr(vocab, "tokens", None),
        "val_split": 0.0,
        "valid_examples": 0,
        "line_seq_len_cap": None,
    }
    if seq2seq:
        cfg_base["seq2seq"] = seq2seq
    cfg_base.update(s["val"])
    train_ds, valid_ds = build_datasets(cfg_base, vocab)

    if s["fitness_mode"] == 2:
        if valid_ds is None:
            raise ValueError(
                "Validation-loss benchmarking requires a non-empty validation file or split."
            )
        if s["dataset_type"] == 1 and len(train_ds.offsets) == 0:
            raise ValueError("Validation split leaves no line examples for training.")
        # MemmapClassicDataset intentionally falls back to the full corpus when
        # a split cannot form one sequence.  That fallback is fine for ordinary
        # training, but it would make a benchmark validate on its training data.
        if (
            s["dataset_type"] == 0
            and cfg_base.get("val_split", 0.0) > 0.0
            and getattr(train_ds, "bin_path", None) == getattr(valid_ds, "bin_path", None)
            and train_ds.end_idx > valid_ds.start_idx
        ):
            raise ValueError(
                "Validation split is too small for the selected sequence length; "
                "choose a larger split, shorter sequence length, or a separate validation file."
            )

    line_mode = s["dataset_type"] == 1
    if line_mode:
        cfg_base["seq_len"] = train_ds.max_len
        pinfo(f"Auto-set seq_len → {cfg_base['seq_len']}")

    # One fixed prompt, from held-out text when there is any, for every sample.
    prompt_ids = []
    prompt_text = "<start of line>" if line_mode else ""
    if s["sample_len"] and seq2seq:
        # Every model gets the same held-out input; samples are its outputs.
        _bench_seed(SEED)
        ds = valid_ds or train_ds
        ids, source_len = ds.get_encoded_example(random.randrange(len(ds.offsets)))
        prompt_ids = ids[:1 + source_len]
        prompt_text = (f"input: {seq2seq_visible(vocab.decode(prompt_ids[1:]))}  "
                       f"expected: {seq2seq_visible(vocab.decode(ids[1 + source_len:-1]))}")
    elif s["sample_len"] and not line_mode:
        _bench_seed(SEED)
        x, _ = (valid_ds or train_ds).get_batch(1)
        prompt_ids = x[0, :max(1, min(64, cfg_base["seq_len"] // 2))].tolist()
        prompt_text = vocab.decode(prompt_ids)
    return {"vocab": vocab, "train_ds": train_ds, "valid_ds": valid_ds, "cfg_base": cfg_base,
            "line_mode": line_mode, "prompt_ids": prompt_ids, "prompt_text": prompt_text}


def run_benchmark():
    cli_banner("Benchmark", "Compare architectures on the same dataset", width=64)
    cli_section("Start", 64)
    cli_opt(0, "New benchmark", "Answer the setup questions")
    cli_opt(1, "Load preset",   "Replay settings saved from an earlier setup or run")
    cli_opt(2, "Resume run",    "Continue an interrupted run in its results folder")
    print(f"  │")
    start = prompt_int("Start", valid={0, 1, 2}, default=0)
    cli_section_end(64)

    records = []
    if start == 2:
        path, data = _bench_read_json(prompt_str("Results folder or results.json"))
        legacy_ids = data["settings"].get("version") == 1
        s = _bench_normalize_settings(data["settings"])
        records = [_bench_json_restore(r) for r in data.get("records", [])]
        if legacy_ids:
            for r in records:
                r["id"] = _bench_legacy_id(r["id"])
                r["key"] = _bench_task_key(r)
        run_dir = path.parent
        pinfo(f"Resuming {run_dir}: {len(records)} of {len(s['tasks'])} rows already done")
    else:
        if start == 1:
            _, data = _bench_read_json(prompt_str("Preset file (or a results folder)"))
            s = _bench_normalize_settings(data.get("settings", data))
        else:
            s = collect_bench_settings()
            preset = prompt_str("Save these settings as a preset  (file path, blank = skip)", default="")
            if preset:
                with open(pathlib.Path(preset).expanduser(), "w", encoding="utf-8") as f:
                    json.dump(s, f, indent=1)
                pok(f"Preset saved to {preset}")
        run_dir = pathlib.Path(BENCH_RESULTS_ROOT) / time.strftime("bench_%Y%m%d_%H%M%S")

    env = _bench_prepare(s)
    done = {r["key"] for r in records}
    tasks = s["tasks"]
    lr_tries = s["lr_search"]["steps"] + 2 if s.get("lr_search") else len(s["lr_multipliers"])
    runs_per_row = s["seeds"] * lr_tries
    metric_name = ["Sliding Loss","Final Train Eval","Best Valid Loss"][s["fitness_mode"]]
    limit_label = (
        f"{s['total_iters']} steps/model" if s["max_seconds"] is None
        else f"{s['max_seconds']:g} seconds/model"
    )
    threshold_label = (
        "no speed filter" if s["min_iters_per_sec"] == 0
        else f"skip below {s['min_iters_per_sec']:g} it/s after {s['speed_warmup_steps']} warm-up steps"
    )
    perf = s["perf"]
    perf_label = (
        (f"AMP {perf['amp_dtype']}" if perf["use_amp"] else "fp32")
        + (f" · compile ({perf['compile_backend']})" if perf["use_compile"] else "")
    )
    extras = []
    if s["hierarchy"]["model_type"] != MODEL_TYPE_NORMAL:
        extras.append("MEGABYTE" if s["hierarchy"]["model_type"] == MODEL_TYPE_MEGABYTE else "MEGABYTE bottom-up")
    if s["size_mode"] == "params":
        extras.append(f"~{s['target_params'] / 1e6:g}M params each")
    if runs_per_row > 1:
        extras.append(f"up to {runs_per_row} runs per row ({s['seeds']} seeds × {lr_tries} LRs)")
    if s.get("lr_search"):
        extras.append(f"LR search {s['lr_search']['min']:.2g}–{s['lr_search']['max']:.2g}")
    if s["lr_schedule"] != "none":
        extras.append(f"LR {s['lr_schedule']}")
    if s["tbptt"]["enabled"]:
        extras.append(f"TBPTT window {s['tbptt']['window']} (recurrent/scan models)")
    print(
        f"\n  {_c(_GR, _B, '▸')} Starting benchmark — {_c(_WH, len(tasks))} rows, oldest first"
        f" · metric: {_c(_YL, metric_name)} · {limit_label} · {threshold_label} · {perf_label}"
        + "".join(f" · {e}" for e in extras)
    )
    print(f"  {_c(_DIM, f'Results: {run_dir}')}\n")
    _bench_save(run_dir, s, records, env["prompt_text"])

    for n, task in enumerate(tasks, 1):
        key = _bench_task_key(task)
        if key in done:
            continue
        record = _bench_run_task(s, task, env)
        records.append(record)
        done.add(key)
        _bench_save(run_dir, s, records, env["prompt_text"])

        prefix = f"{_c(_DIM, f'[{n}/{len(tasks)}]')} {_c(_DIM, record['year'])}"
        detail = []
        if record["status"] == "OK" and s["fitness_mode"] == 2:
            detail.append(f"best @ step {record['best_step']}")
        if s["size_mode"] == "params":
            detail.append(f"dim {record['embed_dim']} · {record['params'] / 1e6:.2f}M params")
        if (len(s["lr_multipliers"]) > 1 or s.get("lr_search")) and record["lr"] is not None:
            detail.append(f"lr {record['lr']:.2g}")
        if record.get("tbptt"):
            detail.append("TBPTT")
        detail_str = f"  {_c(_DIM, ' · '.join(detail))}" if detail else ""
        if record["status"] == "OK":
            pok(f"{prefix}  {_c(_WH, record['name'])}  →  {_c(_GR, _B, _bench_fmt_score(record))}"
                f"{detail_str}  {_c(_DIM, 'OK')}")
            if record.get("sample"):
                preview = record["sample"].replace("\n", "⏎")[:110]
                print(f"      {_c(_DIM, '“' + preview + '”')}")
        else:
            pwarn(f"{prefix}  {_c(_WH, record['name'])}  →  {_c(_RD, _B, record['status'])}")

    _bench_print_tables(s, records, perf_label)
    written = ["results.json", "results.csv"] + (["samples.txt"] if s["sample_len"] else [])
    written += [p.name for p in sorted(run_dir.glob("*.png"))]
    pok(f"Saved to {run_dir}: {', '.join(written)}")


# Replace the old `run_benchmark` call in __main__ with this one.


if __name__ == "__main__":
    try:
        interactive_train()
    except Exception as e:
        print("Fatal error:", repr(e))
        raise
