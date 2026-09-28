import os
import sys
import json
import time
import math
import random
import pathlib
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from linegen_kernels import (
    kernels_available, indrnn_scan, janet_scan, slstm_scan, mlstm_chunkwise,
    indygru_scan, atanu_lstm_scan, rwkv4_wkv, rwkv4_wkv_reference,
    rwkv7_wkv, rwkv7_wkv_reference, selective_scan, selective_scan_reference,
    ltc_sensory, ltc_scan, indylstm_scan, indylstm_reference, unicornn_scan, unicornn_reference,
    sru_scan, sru_reference, irnn_scan, irnn_reference, lru_scan, lru_reference,
    exprnn_scan, exprnn_reference, rru_scan, rru_reference, mogrifier_scan, mogrifier_reference,
    mamba3_chunkwise, affine_scan, lmu_scan, rin_latent_scan, cfc_scan, nru_scan, latent_gru_scan,
    ugrnn_scan, ugrnn_reference, m2rnn_scan, m2rnn_reference, m2rnn_supported,
    ParaGRUCell, ParaLSTMCell, pararnn_solve, pararnn_reference,
)

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False
@torch.jit.script
def heinsen_associative_scan_log(log_coeffs: torch.Tensor, log_values: torch.Tensor, h0: Optional[torch.Tensor] = None):
    """
    Computes parallel scan in log-space.
    h_t = a_t * h_{t-1} + b_t  -->  log(h_t) = log_a + log(h_{t-1}) (associative)
    """
    # 1. Cumulative sum of log_coeffs (log_A)
    a_star = torch.cumsum(log_coeffs, dim=1)
    
    # 2. Log-Cumulative-Sum-Exp of the values corrected by A
    # log_values is log(b_t)
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    
    # 3. Combine
    log_h = a_star + log_h0_plus_b_star
    
    # 4. Handle initial state if present (h0 is strictly positive in this formulation)
    if h0 is not None:
        # Broadcasting h0 correction: log(h0) + A_t
        log_h0 = torch.log(torch.clamp(h0, min=1e-8)).unsqueeze(1)
        log_h = torch.logaddexp(log_h, a_star + log_h0)
        
    return torch.exp(log_h)

@torch.jit.script
def g_act(x: torch.Tensor):
    # The "g" activation from minGRU paper: linear for positive, sigmoid for negative
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

@torch.jit.script
def log_g_act(x: torch.Tensor):
    # Stable log(g(x))
    return torch.where(x >= 0, torch.log(F.relu(x) + 0.5), -F.softplus(-x))
def _torch_affine_scan(A: torch.Tensor, X: torch.Tensor, h0: Optional[torch.Tensor] = None):
    """
    Inclusive associative scan for ``h_t = A_t * h_{t-1} + X_t``.

    Each timestep is an affine transform ``(A_t, X_t)``.  Composition is
    associative: applying a left transform followed by a right transform gives
    ``(A_r * A_l, X_r + A_r * X_l)``.  The doubling loop therefore computes
    every prefix in ``ceil(log2(T))`` tensor stages rather than executing one
    recurrence step per token.  It is pure PyTorch, supports real and complex
    tensors, and remains differentiable.
    """
    if A.ndim != 3 or X.ndim != 3 or A.shape != X.shape:
        raise ValueError("A and X must have identical [batch, time, feature] shapes")
    if h0 is not None and h0.shape != (X.size(0), X.size(2)):
        raise ValueError("h0 must have shape [batch, feature]")

    _, T, _ = X.shape
    if T == 0:
        return X

    coeffs = A
    values = X
    offset = 1
    while offset < T:
        # Keep the right coefficient from the previous stage: it composes the
        # right prefix after the left prefix at this scan distance.
        right_coeffs = coeffs[:, offset:, :]
        coeffs = torch.cat(
            (coeffs[:, :offset, :], right_coeffs * coeffs[:, :-offset, :]),
            dim=1,
        )
        values = torch.cat(
            (values[:, :offset, :], values[:, offset:, :] + right_coeffs * values[:, :-offset, :]),
            dim=1,
        )
        offset *= 2

    return values if h0 is None else values + coeffs * h0.unsqueeze(1)


if _TRITON_AVAILABLE:
    @triton.jit
    def _affine_scan_combine(a_left, b_left, a_right, b_right):
        """Compose left then right affine transforms."""
        return a_right * a_left, b_right + a_right * b_left


    @triton.jit
    def _triton_affine_scan_kernel(
        a_ptr, x_ptr, h0_ptr, out_ptr,
        B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
        HAS_H0: tl.constexpr, BLOCK_T: tl.constexpr,
    ):
        # One program owns a complete recurrence for one (batch, feature)
        # stream. ``tl.associative_scan`` performs the parallel prefix inside
        # that program rather than issuing one kernel operation per timestep.
        program_id = tl.program_id(0)
        batch = program_id // D
        feature = program_id % D
        offsets = tl.arange(0, BLOCK_T)
        active = offsets < T
        ptrs = (batch * T + offsets) * D + feature

        coeffs = tl.load(a_ptr + ptrs, mask=active, other=1.0)
        values = tl.load(x_ptr + ptrs, mask=active, other=0.0)
        coeffs, values = tl.associative_scan(
            (coeffs, values), axis=0, combine_fn=_affine_scan_combine,
        )
        if HAS_H0:
            values += coeffs * tl.load(h0_ptr + batch * D + feature)
        tl.store(out_ptr + ptrs, values, mask=active)


def _triton_scan_eligible(A: torch.Tensor, X: torch.Tensor, h0: Optional[torch.Tensor]) -> bool:
    """Restrict the optional kernel to the currently validated kernel shape."""
    return False#(
        #_TRITON_AVAILABLE
        #and A.is_cuda
        #and X.is_cuda
        #and A.dtype == torch.float32
        #and X.dtype == torch.float32
        #and (h0 is None or (h0.is_cuda and h0.dtype == torch.float32))
        #and 0 < X.size(1) <= 1024
    #)


class _TritonAffineScan(torch.autograd.Function):
    """CUDA forward kernel with an exact, differentiable PyTorch backward."""
    @staticmethod
    def forward(ctx, A: torch.Tensor, X: torch.Tensor, h0: Optional[torch.Tensor]):
        B, T, D = X.shape
        out = torch.empty_like(X)
        block_t = 1 << (T - 1).bit_length()
        _triton_affine_scan_kernel[(B * D,)](
            A.contiguous(), X.contiguous(), h0.contiguous() if h0 is not None else X,
            out,
            B=B, T=T, D=D, HAS_H0=h0 is not None, BLOCK_T=block_t,
            num_warps=4,
        )
        ctx.has_h0 = h0 is not None
        ctx.save_for_backward(A, out, h0 if h0 is not None else X.new_empty(0))
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        A, out, saved_h0 = ctx.saved_tensors
        h0 = saved_h0 if ctx.has_h0 else None
        grad_A, grad_X, grad_h0 = _affine_scan_backward(A, out, grad_out, h0)
        return grad_A, grad_X, grad_h0


def _affine_scan_backward(
    A: torch.Tensor, out: torch.Tensor, grad_out: torch.Tensor, h0: Optional[torch.Tensor],
):
    """Exact VJP for the affine recurrence; shared by Triton backward tests."""
    with torch.no_grad():
        # Reverse-mode recurrence: r_t = g_t + conj(A_{t+1}) * r_{t+1}.
        # Reuse the mathematically identical tensor scan so CUDA forward is
        # accelerated without sacrificing exact autograd behaviour.
        reverse_coeffs = torch.cat(
            (torch.zeros_like(A[:, :1]), A[:, 1:].flip(1).conj()), dim=1,
        )
        reverse_states = _torch_affine_scan(reverse_coeffs, grad_out.flip(1))
        adjoint = reverse_states.flip(1)
        previous = torch.cat(
            ((torch.zeros_like(out[:, 0]) if h0 is None else h0).unsqueeze(1), out[:, :-1]),
            dim=1,
        )
        grad_A = adjoint * previous.conj()
        grad_X = adjoint
        grad_h0 = A[:, 0].conj() * adjoint[:, 0] if h0 is not None else None
    return grad_A, grad_X, grad_h0


def pscan_linear_jit(A: torch.Tensor, X: torch.Tensor, h0: Optional[torch.Tensor] = None):
    """Dispatch an affine scan to Triton on supported CUDA inputs, else PyTorch."""
    if A.ndim != 3 or X.ndim != 3 or A.shape != X.shape:
        raise ValueError("A and X must have identical [batch, time, feature] shapes")
    if h0 is not None and h0.shape != (X.size(0), X.size(2)):
        raise ValueError("h0 must have shape [batch, feature]")
    if _triton_scan_eligible(A, X, h0):
        return _TritonAffineScan.apply(A, X, h0)
    return _torch_affine_scan(A, X, h0)
# ========= Optional activations from lamb.py =========
class PostSDPAGate(nn.Module):
    """
    Implementation of the Gated Attention mechanism (G1 position).
    Applies a head-specific (elementwise) sigmoid gate to SDPA output.
    
    Paper: "Gated Attention for Large Language Models" (Qiu et al., 2025)
    Ref: [cite: 858, 1072]
    """
    def __init__(self, d_model):
        super().__init__()
        # The paper recommends elementwise gating (n x q x dk), which 
        # is equivalent to a linear layer projecting to d_model followed by sigmoid.
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        
        # Init: standard Xavier, bias 0 (starts roughly near 0.5 gating)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, x_input, y_attn):
        """
        Args:
            x_input: The input to the attention block (usually normalized). 
                     Used to compute the gate score[cite: 1086].
            y_attn:  The output of the SDPA (before Wo).
        """
        gate_score = torch.sigmoid(self.gate_proj(x_input))
        return y_attn * gate_score
def robust_log_scan(log_coeffs: torch.Tensor, log_values: torch.Tensor):
    """
    Computes h_t = a_t * h_{t-1} + x_t in log space.
    Stable parallel scan.
    """
    # log_coeffs = log(a)
    # log_values = log(x)
    
    # 1. Accumulate decay: A_t = prod(a_1...a_t) -> log_A = cumsum(log_a)
    log_A = torch.cumsum(log_coeffs, dim=1)
    
    # h_t = A_t * sum_{k<=t}(x_k / A_k), where A_t is the cumulative
    # product through timestep t.  Dividing by the *previous* cumulative
    # product would incorrectly multiply every input by one extra a_t.
    acc = torch.logcumsumexp(log_values - log_A, dim=1)
    
    # 3. Combine: h_t = A_t * acc_t
    log_h = log_A + acc
    
    # Safety Clamp to prevent float32 infinity (NaNs)
    return torch.exp(log_h.clamp(max=50.0))

def parallel_scan_split(A, X, h0: Optional[torch.Tensor]=None):
    """
    Handles x_t = a_t * h_{t-1} + x_t for SIGNED x_t using parallel log-scan.
    Splits X into Positive and Negative streams to allow log-space math.
    """
    # A: (B,T,D) in [0,1] (gates)
    # X: (B,T,D) real values (can be negative)
    
    # 1. Prepare Coefficients (Log Space)
    # Clamp A for stability (prevent log(0))
    log_a = torch.log(A.clamp(min=1e-6))
    
    # 2. Split Input into Pos/Neg streams
    x_pos = X.clamp(min=0)
    x_neg = -X.clamp(max=0)
    
    # Avoid log(0) by masking
    # (We use a tiny epsilon in log, but masked values won't contribute due to exp later)
    log_x_pos = torch.log(x_pos + 1e-12)
    log_x_neg = torch.log(x_neg + 1e-12)
    
    # 3. Handle Initial State h0
    # We fold h0 into the first timestep of the scan effectively
    if h0 is not None:
        h0_pos = h0.clamp(min=0)
        h0_neg = -h0.clamp(max=0)
        # We can't easily prepend to parallel scan without re-padding.
        # Simpler strategy: Add decaying h0 term explicitly at end.
        # h_t_total = h_scan_t + (A_1...A_t)*h0
        pass # handled below
        
    # 4. Run Parallel Scans
    h_pos = robust_log_scan(log_a, log_x_pos)
    h_neg = robust_log_scan(log_a, log_x_neg)
    
    h_out = h_pos - h_neg
    
    # 5. Add Initial State Decay
    if h0 is not None:
        # Decay chain: A_cum = cumprod(A)
        # term = h0 * A_cum
        # But we have log_a, so use exp(cumsum(log_a))
        A_cum = torch.exp(torch.cumsum(log_a, dim=1))
        h_out = h_out + h0.unsqueeze(1) * A_cum
        
    return h_out
# (TTanh / ATanU already used; we also import ASigU, atan_u, asig_u)
try:
    from lamb import TTanh, ATanU, ASigU, atan_u, asig_u
except Exception:
    class TTanh(nn.Module):
        def forward(self, x): return torch.tanh(1.25 * x)
    # Function fallbacks (used by ATanULSTM fallback); these must match lamb.py
    # exactly, or a missing lamb.py would silently change the model.
    def atan_u(x): return (2 / math.pi) * torch.atan((math.pi / 2) * x)
    def asig_u(x, k=2.0): return 0.5 * (1.0 + atan_u(k * x))
    class ATanU(nn.Module):
        def forward(self, x): return atan_u(x)
    class ASigU(nn.Module):
        def __init__(self, k=2.0): super().__init__(); self.k = k
        def forward(self, x): return asig_u(x, k=self.k)

# ========= Try to import custom recurrent cores from lstm.py =========
ExtIndRNN = None
ExtATanULSTM = None
try:
    from lstm import IndRNN as ExtIndRNN  # your file
    from lstm import ATanULSTM as ExtATanULSTM
except Exception:
    pass  # We'll provide robust fallbacks below so the script still runs.

# ========= Activations for MLPs =========
# ========= Activation registry (modular) =========
_ACT_REGISTRY = {}

def register_activation(name: str, factory):
    """factory: () -> nn.Module OR callable(x)->Tensor for functional acts"""
    _ACT_REGISTRY[name.lower()] = factory

class _SelfGatedActivation(nn.Module):
    """Shape-preserving gate for MLPs whose projection width is fixed.

    Full GLU feed-forward blocks are used where an architecture can choose its
    input/output projections (see ``SwiGLU``/``GEGLU``/``MiGLU`` below).  This
    variant lets legacy fixed-width MLP and convolution blocks expose the same
    activation choices without changing their parameter shapes.
    """
    def __init__(self, gate):
        super().__init__()
        self.gate = gate

    def forward(self, x):
        return x * self.gate(x)


def get_activation(name: str) -> nn.Module:
    n = (name or "linear").lower()
    if n not in _ACT_REGISTRY:
        raise ValueError(f"Unknown activation '{name}'. Available: {sorted(_ACT_REGISTRY.keys())}")
    return _ACT_REGISTRY[n]()
from lamb import *
# Built-ins + your customs
register_activation("linear", lambda: nn.Identity())
register_activation("sigmoid", lambda: nn.Sigmoid())
register_activation("tanh", lambda: nn.Tanh())
register_activation("relu",  lambda: nn.ReLU())
register_activation("lrelu", lambda: nn.LeakyReLU(0.2))
register_activation("leaky_relu", lambda: nn.LeakyReLU())
register_activation("elu", lambda: nn.ELU())
register_activation("gelu", lambda: nn.GELU())
register_activation("silu", lambda: nn.SiLU())
register_activation("swish", lambda: nn.SiLU())
register_activation("mish", lambda: nn.Mish())
register_activation("swiglu", lambda: _SelfGatedActivation(F.silu))
register_activation("geglu", lambda: _SelfGatedActivation(F.gelu))
register_activation("miglu", lambda: _SelfGatedActivation(F.mish))
register_activation("ttanh", lambda: TTanh())
register_activation("atanu", lambda: ATanU())
register_activation("sns", lambda: SNS())
register_activation("capsech", lambda: CapSech())
register_activation("salu", lambda: SALU())

# Menu is now dynamic:
def activation_menu_text() -> str:
    names = sorted(_ACT_REGISTRY.keys())
    lines = ["Choose activation (for MLPs/TCN/Transformer):"]
    for i, n in enumerate(names):
        lines.append(f"{i} - {n}")
    return "\n".join(lines)

def activation_names() -> list[str]:
    return sorted(_ACT_REGISTRY.keys())


# ========= MLPs =========
class MLPClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_layers, act_name):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        layers = []
        act = get_activation(act_name)
        for i in range(n_layers):
            layers.append(nn.Linear(embed_dim, embed_dim))
            layers.append(act if i < n_layers-1 else nn.Identity())
        self.net = nn.Sequential(*layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    def _forward_full(self, idx):
        x = self.embed(idx)              # (B,T,C)
        B,T,C = x.shape
        x = x.reshape(B*T, C)
        x = self.net(x)
        return self.lm_head(x).reshape(B,T,-1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
linegenModel.py - Extended with gMLP and aMLP implementations

This file implements gMLP and aMLP architectures based on the paper:
"Pay Attention to MLPs" by Liu et al. (2021)

Key adaptations for autoregressive modeling:
- Causal masking in spatial projections
- Support for variable sequence lengths
- Compatible with existing training infrastructure
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========= Positional Encodings =========
class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as used in the original Transformer."""
    def __init__(self, d_model: int, max_len: int = 65536):
        super().__init__()
        self.d_model = d_model
        # Precompute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe, persistent=False)
    
    def forward(self, T: int, device=None):
        """Return positional encodings for sequence length T."""
        if device is None:
            return self.pe[:T].unsqueeze(0)  # (1, T, d_model)
        return self.pe[:T].to(device).unsqueeze(0)


# ========= Baseline MLP Models (for reference) =========
class ResidualBlock(nn.Module):
    """Basic residual MLP block."""
    def __init__(self, dim, act_name):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.act = get_activation(act_name)
        self.lin2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        h = self.lin1(x)
        h = self.act(h)
        h = self.lin2(h)
        return self.norm(x + h)


class ResidualMLPClassifier(nn.Module):
    """Baseline residual MLP with sinusoidal positional encoding."""
    def __init__(self, vocab_size, embed_dim, n_layers, act_name):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = SinusoidalPositionalEncoding(embed_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(embed_dim, act_name) for _ in range(n_layers)])
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    
    def _forward_full(self, idx):
        x = self.embed(idx)  # (B, T, C)
        B, T, C = x.shape
        x = x + self.pos(T, device=idx.device)
        x = x.reshape(B * T, C)
        x = self.blocks(x)
        return self.lm_head(x).reshape(B, T, -1)

    def forward(self, idx):
        return self._forward_full(idx)


# ========= gMLP Implementation =========
class SpatialGatingUnit(nn.Module):
    """
    Spatial Gating Unit with causal masking for autoregressive modeling.
    
    Based on the paper "Pay Attention to MLPs" (Liu et al., 2021).
    The SGU performs spatial (cross-token) interactions using a learned linear projection
    combined with multiplicative gating.
    """
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn // 2)
        self.seq_len = seq_len
        
        # Spatial projection: projects along sequence dimension
        # For autoregressive tasks, we apply causal masking to the weight matrix
        self.spatial_proj = nn.Linear(seq_len, seq_len, bias=True)
        
        # Initialize bias to 1 as recommended in the paper
        # This ensures the block acts like a standard FFN at initialization
        nn.init.ones_(self.spatial_proj.bias)
        
        # Initialize weights to near-zero for training stability
        nn.init.normal_(self.spatial_proj.weight, mean=0.0, std=1e-6)
        
        # Register causal mask as buffer (won't be trained)
        # This masks the weight matrix to enforce causality
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("causal_mask", causal_mask, persistent=False)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, d_ffn)
        
        Returns:
            Tensor of shape (B, T, d_ffn//2)
        """
        B, T, C = x.shape
        
        # Check sequence length compatibility
        if T > self.seq_len:
            raise ValueError(
                f"Input sequence length ({T}) exceeds model's maximum sequence length ({self.seq_len}). "
                f"Please initialize the model with seq_len >= {T}."
            )
        
        # Split along channel dimension for gating
        u, v = x.chunk(2, dim=-1)  # each: (B, T, d_ffn/2)
        
        # Normalize v
        v = self.norm(v)
        
        # Spatial projection with causal masking
        # Transpose to (B, d_ffn/2, T) for projection
        v = v.transpose(1, 2)  # (B, d_ffn/2, T)
        
        # Apply causal mask to weight matrix during forward pass
        # This ensures position i can only see positions <= i
        W = self.spatial_proj.weight[:T, :T]  # Slice to current seq length
        W_masked = W * self.causal_mask[:T, :T]  # Apply causal mask
        b = self.spatial_proj.bias[:T] if T < self.seq_len else self.spatial_proj.bias
        
        # Manual linear projection with masked weights
        v = F.linear(v, W_masked, b)  # (B, d_ffn/2, T)
        
        v = v.transpose(1, 2)  # (B, T, d_ffn/2)
        
        # Multiplicative gating
        return u * v

    def step(self, x_t, state=None):
        """Run one causal SGU position from its normalized-v prefix cache."""
        if state is None:
            v_history = x_t.new_empty(x_t.size(0), 0, x_t.size(-1) // 2)
        else:
            v_history = state["v_history"]
        position = v_history.size(1)
        if position >= self.seq_len:
            raise ValueError(
                f"gMLP stage cache exceeded configured sequence length {self.seq_len}; "
                "restart the local hierarchy group before stepping again"
            )

        u, v = x_t.chunk(2, dim=-1)
        v_history = torch.cat((v_history, self.norm(v).unsqueeze(1)), dim=1)
        # At position p the causal projection can only use columns 0..p.
        row = self.spatial_proj.weight[position:position + 1, :position + 1]
        mixed_v = F.linear(v_history.transpose(1, 2), row, self.spatial_proj.bias[position:position + 1])
        return u * mixed_v.squeeze(-1), {"v_history": v_history}


class gMLPBlock(nn.Module):
    """
    A single gMLP block as described in 'Pay Attention to MLPs'.
    
    Structure:
    1. LayerNorm
    2. Channel expansion (linear projection)
    3. GELU activation
    4. Spatial Gating Unit (SGU)
    5. Channel projection back to original dimension
    6. Residual connection
    """
    def __init__(self, d_model, d_ffn, seq_len, act_name="gelu"):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn)
        self.activation = get_activation(act_name)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        self.channel_proj2 = nn.Linear(d_ffn // 2, d_model)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, d_model)
        
        Returns:
            Tensor of shape (B, T, d_model)
        """
        shortcut = x
        x = self.norm(x)
        x = self.channel_proj1(x)  # (B, T, d_ffn)
        x = self.activation(x)
        x = self.sgu(x)  # (B, T, d_ffn/2)
        x = self.channel_proj2(x)  # (B, T, d_model)
        return x + shortcut

    def step(self, x_t, state=None):
        shortcut = x_t
        x_t = self.activation(self.channel_proj1(self.norm(x_t)))
        x_t, state = self.sgu.step(x_t, state)
        return self.channel_proj2(x_t) + shortcut, state


class gMLPLanguageModel(nn.Module):
    """
    gMLP for autoregressive language modeling.
    
    Key features:
    - No explicit positional encodings (position info captured in spatial weights)
    - Causal masking for autoregressive generation
    - Multiplicative gating for spatial interactions
    
    **IMPORTANT**: The seq_len parameter MUST match the maximum sequence length 
    used during training! The spatial projection weights are fixed-size based on seq_len.
    
    Args:
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension (d_model)
        n_layers: Number of gMLP blocks
        d_ffn: Hidden dimension in feed-forward layers (typically 4 * embed_dim)
               MUST be even (will be split in half for gating)
        seq_len: Maximum sequence length - THIS MUST MATCH YOUR TRAINING SEQ_LEN!
    """
    def __init__(self, vocab_size, embed_dim, n_layers, d_ffn, seq_len, act_name="gelu"):
        super().__init__()
        
        # Validate d_ffn is even
        if d_ffn % 2 != 0:
            raise ValueError(f"d_ffn must be even (got {d_ffn}). It will be split in half for gating.")
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.seq_len = seq_len
        
        # Stack of gMLP blocks
        self.blocks = nn.ModuleList([
            gMLPBlock(embed_dim, d_ffn, seq_len, act_name=act_name) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    
    def forward_hidden(self, x):
        """Apply gMLP blocks to hidden vectors for hierarchy adapters."""
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, idx):
        """
        Args:
            idx: Token indices of shape (B, T) where T <= seq_len
        
        Returns:
            Logits of shape (B, T, vocab_size)
        """
        x = self.forward_hidden(self.embed(idx))
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits


# ========= aMLP Implementation (gMLP + Tiny Attention) =========
class TinyAttention(nn.Module):
    """
    Tiny single-head causal self-attention for aMLP.
    Updated with Post-SDPA Gating[cite: 858].
    """
    def __init__(self, d_model, d_attn=64):
        super().__init__()
        self.d_attn = d_attn
        self.qkv_proj = nn.Linear(d_model, 3 * d_attn)
        self.out_proj = nn.Linear(d_attn, d_model)
        self.scale = d_attn ** -0.5
        
        # === NEW: Post-SDPA Gate ===
        # Note: The gate acts on the attention dimension (d_attn) before projection,
        # or we can project the gate to d_attn. 
        # The paper applies G1 to the SDPA output (dim = heads * d_k). 
        # Here we gate the `d_attn` dimension.
        self.sdpa_gate = nn.Linear(d_model, d_attn) 
        nn.init.zeros_(self.sdpa_gate.bias)
        # ===========================

    def forward(self, x):
        # x is (B, T, d_model) - assumed to be normalized input [cite: 1086]
        B, T, C = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1) # (B, T, d_attn)
        
        # --- SDPA ---
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=True
            ).squeeze(1)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
        
        # === NEW: Apply Gating ===
        # Gate depends on input x [cite: 1195]
        # Y' = Y * sigmoid(XW)
        gate = torch.sigmoid(self.sdpa_gate(x)) # (B, T, d_attn)
        out = out * gate
        # =========================

        # Project back to d_model
        out = self.out_proj(out)
        return out

    def step(self, x_t, state=None):
        """Attend one item using exact causal K/V prefix caches."""
        q, k, v = self.qkv_proj(x_t).chunk(3, dim=-1)
        if state is None:
            k_history = x_t.new_empty(x_t.size(0), 0, self.d_attn)
            v_history = x_t.new_empty(x_t.size(0), 0, self.d_attn)
        else:
            k_history = state["keys"]
            v_history = state["values"]
        k_history = torch.cat((k_history, k.unsqueeze(1)), dim=1)
        v_history = torch.cat((v_history, v.unsqueeze(1)), dim=1)
        weights = torch.softmax(torch.einsum("bd,btd->bt", q, k_history) * self.scale, dim=-1)
        out = torch.einsum("bt,btd->bd", weights, v_history)
        out = out * torch.sigmoid(self.sdpa_gate(x_t))
        return self.out_proj(out), {"keys": k_history, "values": v_history}


class SpatialGatingUnitWithAttention(nn.Module):
    """
    SGU enhanced with tiny attention (for aMLP).
    
    This combines the spatial gating mechanism of gMLP with a tiny attention module.
    The attention is used to capture cross-sentence alignment patterns that pure
    spatial projection might miss.
    """
    def __init__(self, d_model, d_ffn, seq_len, d_attn=64, act_name="gelu"):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn // 2)
        self.seq_len = seq_len
        
        # Spatial projection (same as gMLP)
        self.spatial_proj = nn.Linear(seq_len, seq_len, bias=True)
        nn.init.ones_(self.spatial_proj.bias)
        nn.init.normal_(self.spatial_proj.weight, mean=0.0, std=1e-6)
        
        # Tiny attention module
        self.tiny_attn = TinyAttention(d_model, d_attn)
        # Projection to convert attention output to SGU dimension
        self.attn_proj = nn.Linear(d_model, d_ffn // 2)
        
        # Register causal mask for spatial projection
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("causal_mask", causal_mask, persistent=False)
    
    def forward(self, x, x_norm):
        """
        Args:
            x: Tensor of shape (B, T, d_ffn) - output from channel expansion
            x_norm: Tensor of shape (B, T, d_model) - normalized input (for attention)
        
        Returns:
            Tensor of shape (B, T, d_ffn//2)
        """
        B, T, C = x.shape
        
        # Check sequence length compatibility
        if T > self.seq_len:
            raise ValueError(
                f"Input sequence length ({T}) exceeds model's maximum sequence length ({self.seq_len}). "
                f"Please initialize the model with seq_len >= {T}."
            )
        
        # Standard SGU path
        u, v = x.chunk(2, dim=-1)  # each: (B, T, d_ffn/2)
        v = self.norm(v)
        
        # Spatial projection with causal masking
        v = v.transpose(1, 2)  # (B, d_ffn/2, T)
        
        # Apply causal mask to weight matrix
        W = self.spatial_proj.weight[:T, :T]
        W_masked = W * self.causal_mask[:T, :T]
        b = self.spatial_proj.bias[:T] if T < self.seq_len else self.spatial_proj.bias
        
        v = F.linear(v, W_masked, b)  # (B, d_ffn/2, T)
        v = v.transpose(1, 2)  # (B, T, d_ffn/2)
        
        # Add tiny attention contribution
        attn_out = self.tiny_attn(x_norm)  # (B, T, d_model)
        attn_contrib = self.attn_proj(attn_out)  # (B, T, d_ffn/2)
        v = v + attn_contrib
        
        return u * v

    def step(self, x_t, x_norm_t, state=None):
        if state is None:
            v_history = x_t.new_empty(x_t.size(0), 0, x_t.size(-1) // 2)
            attention_state = None
        else:
            v_history = state["v_history"]
            attention_state = state["attention"]
        position = v_history.size(1)
        if position >= self.seq_len:
            raise ValueError(
                f"aMLP stage cache exceeded configured sequence length {self.seq_len}; "
                "restart the local hierarchy group before stepping again"
            )

        u, v = x_t.chunk(2, dim=-1)
        v_history = torch.cat((v_history, self.norm(v).unsqueeze(1)), dim=1)
        row = self.spatial_proj.weight[position:position + 1, :position + 1]
        mixed_v = F.linear(v_history.transpose(1, 2), row, self.spatial_proj.bias[position:position + 1])
        attention, attention_state = self.tiny_attn.step(x_norm_t, attention_state)
        mixed_v = mixed_v.squeeze(-1) + self.attn_proj(attention)
        return u * mixed_v, {"v_history": v_history, "attention": attention_state}


class aMLPBlock(nn.Module):
    """
    aMLP block: gMLP + tiny attention.
    
    This hybrid architecture combines the efficiency of spatial gating with
    the flexibility of self-attention, using only a tiny attention module.
    """
    def __init__(self, d_model, d_ffn, seq_len, d_attn=64, act_name="gelu"):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn)
        self.activation = get_activation(act_name)
        self.sgu_with_attn = SpatialGatingUnitWithAttention(d_model, d_ffn, seq_len, d_attn, act_name=act_name)
        self.channel_proj2 = nn.Linear(d_ffn // 2, d_model)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, d_model)
        
        Returns:
            Tensor of shape (B, T, d_model)
        """
        shortcut = x
        x_norm = self.norm(x)
        x = self.channel_proj1(x_norm)
        x = self.activation(x)
        x = self.sgu_with_attn(x, x_norm)
        x = self.channel_proj2(x)
        return x + shortcut

    def step(self, x_t, state=None):
        shortcut = x_t
        x_norm_t = self.norm(x_t)
        x_t = self.activation(self.channel_proj1(x_norm_t))
        x_t, state = self.sgu_with_attn.step(x_t, x_norm_t, state)
        return self.channel_proj2(x_t) + shortcut, state


class aMLPLanguageModel(nn.Module):
    """
    aMLP for autoregressive language modeling (gMLP + tiny attention).
    
    This model enhances gMLP with small attention modules that help with
    cross-sentence alignment tasks. According to the paper, a single-head
    attention with dimension 64-128 is sufficient to close the gap with
    full Transformers on many NLP tasks.
    
    **IMPORTANT**: The seq_len parameter MUST match the maximum sequence length 
    used during training! The spatial projection weights are fixed-size based on seq_len.
    
    Args:
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension (d_model)
        n_layers: Number of aMLP blocks
        d_ffn: Hidden dimension in feed-forward layers
               MUST be even (will be split in half for gating)
        seq_len: Maximum sequence length - THIS MUST MATCH YOUR TRAINING SEQ_LEN!
        d_attn: Attention dimension (typically 64 or 128)
    """
    def __init__(self, vocab_size, embed_dim, n_layers, d_ffn, seq_len, d_attn=64, act_name="gelu"):
        super().__init__()
        
        # Validate d_ffn is even
        if d_ffn % 2 != 0:
            raise ValueError(f"d_ffn must be even (got {d_ffn}). It will be split in half for gating.")
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.seq_len = seq_len
        
        # Stack of aMLP blocks
        self.blocks = nn.ModuleList([
            aMLPBlock(embed_dim, d_ffn, seq_len, d_attn, act_name=act_name) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    
    def forward_hidden(self, x):
        """Apply aMLP blocks to hidden vectors for hierarchy adapters."""
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, idx):
        """
        Args:
            idx: Token indices of shape (B, T) where T <= seq_len
        
        Returns:
            Logits of shape (B, T, vocab_size)
        """
        x = self.forward_hidden(self.embed(idx))
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

class OneHotWindowMLPClassifier(nn.Module):
    """
    MLP that consumes a rolling window of one-hot tokens (no nn.Embedding).
    For a sequence length K=seq_len (a.k.a. block_size), we roll the input K times.
    Each roll inserts a special token at the start (id = vocab_size) and shifts the rest.
    We one-hot encode each rolled sequence and concatenate along features:
        input dim to first Linear = K * (vocab_size + 1)
    Output is per-position logits: (B, T, vocab_size)
    """
    def __init__(self, vocab_size: int, seq_len: int, embed_dim: int, n_layers: int, act_name: str):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len      # block_size
        self.input_dim = (vocab_size + 1) * seq_len  # (+1) for the special BOS/blank id
        layers = []

        # First layer: big one-hot window -> hidden
        layers.append(nn.Linear(self.input_dim, embed_dim))
        layers.append(get_activation(act_name))

        # Hidden layers
        for _ in range(max(0, n_layers - 1)):
            layers.append(nn.Linear(embed_dim, embed_dim))
            layers.append(get_activation(act_name))

        # Head to vocab
        layers.append(nn.Linear(embed_dim, vocab_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, idx: torch.Tensor):
        """
        idx: (B, T) long
        Returns logits: (B, T, vocab_size)
        """
        B, T_orig = idx.shape
    
        # If the current T is shorter than model's configured seq_len, left-pad with the special token
        if T_orig < self.seq_len:
            pad = idx.new_full((B, self.seq_len - T_orig), self.vocab_size)  # special id = vocab_size
            idx_work = torch.cat([pad, idx], dim=1)  # (B, T_pad)
        else:
            idx_work = idx
    
        # Build rolling one-hot window over exactly self.seq_len rolls
        cur = idx_work
        onehots = []
        for _ in range(self.seq_len):
            # (B, T_pad, V+1)
            oh = F.one_hot(cur.clamp_max(self.vocab_size), num_classes=self.vocab_size + 1).float()
            onehots.append(oh)
            cur = torch.roll(cur, shifts=1, dims=1)
            cur[:, 0] = self.vocab_size  # insert special token at the new front
    
        # Concatenate window features → (B, T_pad, seq_len*(V+1))
        x = torch.cat(onehots, dim=-1)
    
        # MLP: preserves time dimension
        logits = self.mlp(x)  # (B, T_pad, vocab)
    
        # If we padded, trim back to the original sequence length (right-aligned)
        if logits.size(1) != T_orig:
            logits = logits[:, -T_orig:, :]
    
        return logits


# ========= Builtin RNNs =========
import math
import torch
import torch.nn as nn
import torch.nn.init as init

class OGBuiltinRNNWrapper(nn.Module):
    def __init__(self, vocab_size, hidden, n_layers, mode, tie_weights=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.n_layers = n_layers
        self.mode = mode
        self.tie_weights = tie_weights

        self.embed = nn.Embedding(vocab_size, hidden)

        if mode == 'rnn_tanh':
            self.core = nn.RNN(hidden, hidden, num_layers=n_layers,
                               nonlinearity='tanh', batch_first=True)
        elif mode == 'rnn_relu':
            self.core = nn.RNN(hidden, hidden, num_layers=n_layers,
                               nonlinearity='relu', batch_first=True)
        elif mode == 'gru':
            self.core = nn.GRU(hidden, hidden, num_layers=n_layers, batch_first=True)
        elif mode == 'lstm':
            self.core = nn.LSTM(hidden, hidden, num_layers=n_layers, batch_first=True)
        else:
            raise ValueError("Unknown mode")

        self.lm_head = nn.Linear(hidden, vocab_size, bias=True)

        # init + optional tie
        self._init_parameters()
        if tie_weights:
            # requires embed_dim == hidden_dim used by lm_head
            if self.embed.weight.shape[1] != self.lm_head.in_features:
                raise ValueError("Cannot tie weights: embed dim != hidden dim")
            self.lm_head.weight = self.embed.weight  # weight tying

    def forward(self, idx, state=None):
        x = self.embed(idx)
        out, state = self.core(x, state)
        return self.lm_head(out), state

    # ---------- init helpers ----------
    @torch.no_grad()
    def _init_parameters(self):
        # Embedding: common choice is normal(0, 1/sqrt(hidden)) or uniform
        init.normal_(self.embed.weight, mean=0.0, std=1.0 / math.sqrt(self.hidden))

        # Output head bias: zero
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)

        # Initialize recurrent module per mode
        if isinstance(self.core, nn.RNN):
            if self.core.nonlinearity == 'tanh':
                self._init_rnn_tanh()
            else:
                self._init_rnn_relu()
        elif isinstance(self.core, nn.GRU):
            self._init_gru()
        elif isinstance(self.core, nn.LSTM):
            self._init_lstm()

    @torch.no_grad()
    def _init_rnn_tanh(self):
        gain = nn.init.calculate_gain('tanh')  # ~5/3
        for l in range(self.n_layers):
            w_ih = getattr(self.core, f'weight_ih_l{l}')
            w_hh = getattr(self.core, f'weight_hh_l{l}')
            b_ih = getattr(self.core, f'bias_ih_l{l}', None)
            b_hh = getattr(self.core, f'bias_hh_l{l}', None)

            init.xavier_uniform_(w_ih, gain=gain)
            init.orthogonal_(w_hh, gain=gain)
            if b_ih is not None: nn.init.zeros_(b_ih)
            if b_hh is not None: nn.init.zeros_(b_hh)

    @torch.no_grad()
    def _init_rnn_relu(self, rho: float = 0.97):
        """Scaled-identity recurrent init for long sequences."""
        relu_gain = nn.init.calculate_gain('relu')
    
        for l in range(self.n_layers):
            w_ih = getattr(self.core, f'weight_ih_l{l}')
            w_hh = getattr(self.core, f'weight_hh_l{l}')
            b_ih = getattr(self.core, f'bias_ih_l{l}', None)
            b_hh = getattr(self.core, f'bias_hh_l{l}', None)
    
            # Input: standard He init for ReLU
            nn.init.kaiming_uniform_(w_ih, a=0.0, nonlinearity='relu')
    
            # Recurrent: scaled identity
            hidden_size = w_hh.shape[0]
            w_hh.zero_()
            w_hh.view(hidden_size, hidden_size).copy_(torch.eye(hidden_size) * rho)
    
            # Biases: small positive to avoid dead ReLUs
            if b_ih is not None: nn.init.zeros_(b_ih)
            if b_hh is not None: b_hh.fill_(0.01)


    @torch.no_grad()
    def _init_gru(self):
        # PyTorch gate order: [reset, update, new] => chunks along dim 0
        for l in range(self.n_layers):
            w_ih = getattr(self.core, f'weight_ih_l{l}')
            w_hh = getattr(self.core, f'weight_hh_l{l}')
            b_ih = getattr(self.core, f'bias_ih_l{l}', None)
            b_hh = getattr(self.core, f'bias_hh_l{l}', None)

            # Input weights: Xavier per gate (sigmoid gates gain=1, tanh gate gain=tanh)
            r_ih, z_ih, n_ih = w_ih.chunk(3, dim=0)
            r_hh, z_hh, n_hh = w_hh.chunk(3, dim=0)

            init.xavier_uniform_(r_ih, gain=1.0)                    # reset (sigmoid)
            init.xavier_uniform_(z_ih, gain=1.0)                    # update (sigmoid)
            init.xavier_uniform_(n_ih, gain=nn.init.calculate_gain('tanh'))  # new (tanh)

            # Recurrent weights: orthogonal per gate
            init.orthogonal_(r_hh, gain=1.0)
            init.orthogonal_(z_hh, gain=1.0)
            init.orthogonal_(n_hh, gain=nn.init.calculate_gain('tanh'))

            if b_ih is not None: nn.init.zeros_(b_ih)
            if b_hh is not None: nn.init.zeros_(b_hh)

    @torch.no_grad()
    def _init_lstm(self):
        # PyTorch gate order: [ingate, forgetgate, cellgate, outgate]
        tanh_gain = nn.init.calculate_gain('tanh')
        for l in range(self.n_layers):
            w_ih = getattr(self.core, f'weight_ih_l{l}')
            w_hh = getattr(self.core, f'weight_hh_l{l}')
            b_ih = getattr(self.core, f'bias_ih_l{l}', None)
            b_hh = getattr(self.core, f'bias_hh_l{l}', None)

            i_ih, f_ih, g_ih, o_ih = w_ih.chunk(4, dim=0)
            i_hh, f_hh, g_hh, o_hh = w_hh.chunk(4, dim=0)

            # Input weights: Xavier (sigmoid gates gain=1, cell/tanh gate uses tanh gain)
            init.xavier_uniform_(i_ih, gain=1.0)
            init.xavier_uniform_(f_ih, gain=1.0)
            init.xavier_uniform_(o_ih, gain=1.0)
            init.xavier_uniform_(g_ih, gain=tanh_gain)

            # Recurrent weights: orthogonal gate-wise
            init.orthogonal_(i_hh, gain=1.0)
            init.orthogonal_(f_hh, gain=1.0)
            init.orthogonal_(o_hh, gain=1.0)
            init.orthogonal_(g_hh, gain=tanh_gain)

            # Biases: zero, then forget-gate bias trick
            if b_ih is not None:
                nn.init.zeros_(b_ih)
                # add +1.0 to forget gate (bias_ih slice)
                hidden = self.hidden
                b_ih[hidden:2*hidden].add_(1.0)
            if b_hh is not None:
                nn.init.zeros_(b_hh)


import math
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """RMSNorm over last dim: x * g / rms(x)"""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add_(self.eps).sqrt_()
        return x * (self.g / rms)

@torch.no_grad()
def initialize_native_rnn_core(core, tanh_spectral_radius=0.99, relu_identity_scale=1.0):
    """Initialize one native PyTorch recurrent layer for stable long sequences."""
    if isinstance(core, nn.RNN):
        if core.nonlinearity == "tanh":
            gain = nn.init.calculate_gain("tanh")
            nn.init.xavier_uniform_(core.weight_ih_l0, gain=gain)
            nn.init.orthogonal_(core.weight_hh_l0, gain=1.0)
            core.weight_hh_l0.mul_(tanh_spectral_radius)
        else:
            hidden = core.hidden_size
            core.weight_hh_l0.zero_()
            eye = torch.eye(hidden, device=core.weight_hh_l0.device, dtype=core.weight_hh_l0.dtype)
            core.weight_hh_l0.copy_(relu_identity_scale * eye)
            nn.init.kaiming_uniform_(core.weight_ih_l0, a=0.0, nonlinearity="relu")
            core.weight_ih_l0.mul_(1e-3)
        if core.bias:
            nn.init.zeros_(core.bias_ih_l0)
            nn.init.zeros_(core.bias_hh_l0)
        return
    if isinstance(core, nn.GRU):
        r_ih, z_ih, n_ih = core.weight_ih_l0.chunk(3, 0)
        r_hh, z_hh, n_hh = core.weight_hh_l0.chunk(3, 0)
        nn.init.xavier_uniform_(r_ih, gain=1.0)
        nn.init.xavier_uniform_(z_ih, gain=1.0)
        nn.init.xavier_uniform_(n_ih, gain=nn.init.calculate_gain("tanh"))
        nn.init.orthogonal_(r_hh, gain=1.0)
        nn.init.orthogonal_(z_hh, gain=1.0)
        nn.init.orthogonal_(n_hh, gain=nn.init.calculate_gain("tanh"))
        if core.bias:
            nn.init.zeros_(core.bias_ih_l0)
            nn.init.zeros_(core.bias_hh_l0)
            hidden = core.hidden_size
            core.bias_ih_l0[hidden:2 * hidden].add_(1.0)
        return
    if isinstance(core, nn.LSTM):
        i_ih, f_ih, g_ih, o_ih = core.weight_ih_l0.chunk(4, 0)
        i_hh, f_hh, g_hh, o_hh = core.weight_hh_l0.chunk(4, 0)
        tanh_gain = nn.init.calculate_gain("tanh")
        for weight in (i_ih, f_ih, o_ih):
            nn.init.xavier_uniform_(weight, gain=1.0)
        nn.init.xavier_uniform_(g_ih, gain=tanh_gain)
        for weight in (i_hh, f_hh, o_hh):
            nn.init.orthogonal_(weight, gain=1.0)
        nn.init.orthogonal_(g_hh, gain=tanh_gain)
        if core.bias:
            nn.init.zeros_(core.bias_ih_l0)
            nn.init.zeros_(core.bias_hh_l0)
            hidden = core.hidden_size
            core.bias_ih_l0[hidden:2 * hidden].add_(1.0)
        return
    raise TypeError(f"expected native RNN/GRU/LSTM, got {type(core).__name__}")


class BuiltinRNNWrapper(nn.Module):
    """
    Stack of num_layers separate 1-layer RNN/GRU/LSTM cores.
    Keeps cuDNN fast path; allows Norm/Dropout/Residuals between layers.
    """
    def __init__(self, vocab_size, hidden, num_layers, mode,
                 tie_weights=True,
                 use_norm=2,          # 0=None, 1=BatchNorm, 2=LayerNorm, 3=RMSNorm
                 res_every=2,         # 0 disables; otherwise every n layers
                 res_type=0,          # 0=add, 1=concat(+proj), 2=ReZero(scalar), 3=ReZero(elementwise)
                 dropout=0.0,         # inter-layer dropout prob
                 use_multiplier=0,    # 0=None, 1=scalar per-layer, 2=vector per-layer
                 # --- new: long-seq init knobs ---
                 tanh_spectral_radius=0.99,
                 relu_identity_scale=1.0,
                 # --- new: capture/visualizer ---
                 enable_capture=False):
        super().__init__()
        assert mode in ('rnn_tanh', 'rnn_relu', 'gru', 'lstm')
        assert use_norm in (0,1,2,3,4,5,6)
        assert res_type in (0,1,2,3)
        assert use_multiplier in (0,1,2)

        self.vocab_size = vocab_size
        self.hidden = hidden
        self.num_layers = num_layers
        self.mode = mode
        self.tie_weights = tie_weights
        self.use_norm = int(use_norm)
        self.res_every = int(res_every)
        self.res_type = int(res_type)
        self.dropout_p = float(dropout)
        self.use_multiplier = int(use_multiplier)

        # long-seq init knobs
        self.tanh_spectral_radius = float(tanh_spectral_radius)
        self.relu_identity_scale = float(relu_identity_scale)

        # capture buffers
        self._capture_enabled = bool(enable_capture)
        self._captured = None  

        self.embed = nn.Embedding(vocab_size, hidden)

        # Build 1-layer cores (cuDNN fast path)
        cores = []
        for _ in range(num_layers):
            if mode == 'rnn_tanh':
                core = nn.RNN(hidden, hidden, num_layers=1, nonlinearity='tanh', batch_first=True, dropout=0.0)
            elif mode == 'rnn_relu':
                core = nn.RNN(hidden, hidden, num_layers=1, nonlinearity='relu', batch_first=True, dropout=0.0)
            elif mode == 'gru':
                core = nn.GRU(hidden, hidden, num_layers=1, batch_first=True, dropout=0.0)
            else:
                core = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True, dropout=0.0)
            cores.append(core)
        self.cores = nn.ModuleList(cores)

        # Inter-layer normalization modules
        if self.use_norm == 0:
            self.norms = None
        elif self.use_norm == 1:
            self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(num_layers - 1)])
        elif self.use_norm == 2:
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers - 1)])
        elif self.use_norm == 3:  # RMSNorm
            self.norms = nn.ModuleList([RMSNorm(hidden) for _ in range(num_layers - 1)])
        elif self.use_norm == 4:  # TTanh
            self.norms = nn.ModuleList([TTanh() for _ in range(num_layers - 1)])
        elif self.use_norm == 5:  # ETTanh
            self.norms = nn.ModuleList([ETTanh(hidden) for _ in range(num_layers - 1)])
        elif self.use_norm == 6:  # DyT
            self.norms = nn.ModuleList([DyT(hidden) for _ in range(num_layers - 1)])

        # Inter-layer dropout
        self.drops = nn.ModuleList([nn.Dropout(self.dropout_p) for _ in range(num_layers - 1)]) if self.dropout_p > 0 else None

        # 4. Residual Handlers
        # === FIX IS HERE ===
        self.do_res = (self.res_every > 0)
        
        num_res_hops = 0
        if self.do_res:  # Guard against res_every=0
            for i in range(num_layers - 1):
                if ((i + 1) % self.res_every) == 0:
                    num_res_hops += 1

        self.res_mixers = None
        self.alphas = None
        self.betas = None

        if self.do_res and num_res_hops > 0:
            if self.res_type == 0:
                mixers = []
                for i in range(num_layers - 1):
                    if ((i + 1) % self.res_every) == 0:
                        mixers.append(nn.Linear(hidden, hidden))
                    else:
                        mixers.append(nn.Identity()) 
                self.res_mixers = nn.ModuleList(mixers)

            elif self.res_type == 1:
                mixers = []
                for i in range(num_layers - 1):
                    if ((i + 1) % self.res_every) == 0:
                        mixers.append(nn.Linear(hidden * 2, hidden))
                    else:
                        mixers.append(nn.Identity())
                self.res_mixers = nn.ModuleList(mixers)

            elif self.res_type in (2, 3):
                param_shape = 1 if self.res_type == 2 else hidden
                self.alphas = nn.ParameterList()
                self.betas = nn.ParameterList() 
                for i in range(num_layers - 1):
                    if ((i + 1) % self.res_every) == 0:
                        self.alphas.append(nn.Parameter(torch.zeros(param_shape)))
                        self.betas.append(nn.Parameter(torch.ones(param_shape)))

        # 5. Multipliers
        if self.use_multiplier == 0:
            self.multipliers = None
        elif self.use_multiplier == 1:
            self.multipliers = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(num_layers)])
        else:
            self.multipliers = nn.ParameterList([nn.Parameter(torch.ones(hidden)) for _ in range(num_layers)])

        self.lm_head = nn.Linear(hidden, vocab_size, bias=True)

        self._init_parameters()
        if tie_weights:
            if self.embed.weight.shape[1] != self.lm_head.in_features:
                raise ValueError("Cannot tie weights: embed dim != hidden dim")
            self.lm_head.weight = self.embed.weight

    # ... (Rest of the class methods: start_capture, stop_capture, get_captured, _maybe_capture, _zero_state, _apply_norm, _apply_residual, _apply_multiplier, forward, _init_parameters, etc. remain unchanged) ...
    # Be sure to include the rest of the methods below if copy-pasting!
    
    def start_capture(self):
        self._capture_enabled = True
        self._captured = [[] for _ in range(self.num_layers)]

    def stop_capture(self):
        self._capture_enabled = False

    @torch.no_grad()
    def get_captured(self):
        if self._captured is None: return None
        out = []
        for layer_list in self._captured:
            if len(layer_list) == 0:
                out.append(None)
            else:
                xs = [x.unsqueeze(1) if x.dim() == 2 else x for x in layer_list]
                stacked = torch.cat(xs, dim=1) 
                out.append(stacked)
        return out

    def _maybe_capture(self, li, y):
        if not self._capture_enabled: return
        if self._captured is None: self._captured = [[] for _ in range(self.num_layers)]
        y_last = y[:, -1, :].detach().to('cpu')
        self._captured[li].append(y_last)

    def _zero_state(self, B, device, dtype):
        if self.mode in ('gru', 'rnn_tanh', 'rnn_relu'):
            return [torch.zeros(1, B, self.hidden, device=device, dtype=dtype) for _ in range(self.num_layers)]
        else:  
            return [(torch.zeros(1, B, self.hidden, device=device, dtype=dtype),
                     torch.zeros(1, B, self.hidden, device=device, dtype=dtype)) for _ in range(self.num_layers)]

    def _apply_norm(self, li, y):
        if self.norms is None: return y
        if self.use_norm == 1:
            B, T, H = y.shape
            y2 = y.contiguous().view(B*T, H)
            y2 = self.norms[li](y2)
            return y2.view(B, T, H)
        else:
            return self.norms[li](y)

    def _apply_residual(self, li, y_in, y_out, res_idx):
        if not self.do_res: return y_out
        if ((li + 1) % self.res_every) != 0: return y_out

        if self.res_type == 0:
            mixed = self.res_mixers[li](y_out)
            return y_in + mixed
        elif self.res_type == 1:
            cat = torch.cat([y_out, y_in], dim=-1)
            return self.res_mixers[li](cat)
        elif self.res_type in (2, 3):
            alpha = self.alphas[res_idx]
            beta = self.betas[res_idx]
            return (y_out * alpha) + (y_in * beta)
        return y_out

    def _apply_multiplier(self, li, y):
        if self.multipliers is None: return y
        m = self.multipliers[li]
        if self.use_multiplier == 1:
            return y * m
        else:
            return y * m.view(1, 1, -1)

    def forward_hidden(self, x0, state=None):
        """Run the reusable recurrent core on hidden vectors, not token IDs."""
        B, T = x0.size(0), x0.size(1)

        if state is None:
            state = self._zero_state(B, x0.device, x0.dtype)

        new_state = []
        y = x0
        res_hop_count = 0

        for li, core in enumerate(self.cores):
            s_in = state[li]
            y = self._apply_multiplier(li, y)
            y_in = y
            
            if li > 0:
                y = self._apply_norm(li - 1, y)
            
            y, s_out = core(y, s_in) 

            if li < self.num_layers - 1:
                y = self._apply_residual(li, y_in, y, res_hop_count)
                if self.do_res and ((li + 1) % self.res_every) == 0:
                    res_hop_count += 1
                if self.drops is not None:
                    y = self.drops[li](y)

            self._maybe_capture(li, y)
            new_state.append(s_out)

        return y, new_state

    def forward(self, idx, state=None):
        hidden, new_state = self.forward_hidden(self.embed(idx), state)
        return self.lm_head(hidden), new_state

    # ... (Keep _init_parameters and the specific init methods as they were in your file) ...
    @torch.no_grad()
    def _init_parameters(self):
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0 / math.sqrt(self.hidden))
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)

        for core in self.cores:
            initialize_native_rnn_core(
                core, tanh_spectral_radius=self.tanh_spectral_radius,
                relu_identity_scale=self.relu_identity_scale,
            )

    @torch.no_grad()
    def _init_rnn_tanh_longseq(self, core: nn.RNN, spectral_radius: float = 0.99):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(core.weight_ih_l0, gain=gain)
        nn.init.orthogonal_(core.weight_hh_l0, gain=1.0)
        with torch.no_grad():
            core.weight_hh_l0.mul_(spectral_radius)
        if core.bias:
            nn.init.zeros_(core.bias_ih_l0); nn.init.zeros_(core.bias_hh_l0)

    @torch.no_grad()
    def _init_rnn_relu_longseq(self, core: nn.RNN, identity_scale: float = 1.0):
        H = core.hidden_size
        with torch.no_grad():
            core.weight_hh_l0.zero_()
            eye = torch.eye(H, device=core.weight_hh_l0.device, dtype=core.weight_hh_l0.dtype)
            core.weight_hh_l0[:H, :H].copy_(identity_scale * eye)
        nn.init.kaiming_uniform_(core.weight_ih_l0, a=0.0, nonlinearity='relu')
        core.weight_ih_l0.mul_(1e-3)  
        if core.bias:
            nn.init.zeros_(core.bias_ih_l0); nn.init.zeros_(core.bias_hh_l0)


class BuiltinRNNStage(nn.Module):
    """Hidden-vector native RNN stack with configurable inter-layer wiring.

    This is the MEGABYTE counterpart to ``BuiltinRNNWrapper``: it deliberately
    owns no embedding or language-model head, so it can process a stage's
    already-projected vectors in both full and incremental hierarchy paths.
    """
    def __init__(self, kind, dim, depth, norm_type=0, res_every=0, res_type=0,
                 dropout=0.0, tanh_spectral_radius=0.99, relu_identity_scale=1.0):
        super().__init__()
        if kind not in {"rnn", "rnn_relu", "gru", "lstm"}:
            raise ValueError(f"unknown native RNN stage kind: {kind}")
        if norm_type not in range(7):
            raise ValueError("MEGABYTE RNN norm type must be in 0..6")
        if res_type not in range(4):
            raise ValueError("MEGABYTE RNN residual type must be in 0..3")
        if depth < 1 or res_every < 0 or dropout < 0:
            raise ValueError("MEGABYTE RNN depth/residual interval/dropout must be non-negative")
        self.kind, self.dim, self.depth = kind, dim, depth
        self.norm_type = int(norm_type)
        self.res_every, self.res_type, self.dropout_p = int(res_every), int(res_type), float(dropout)
        self.do_res = self.res_every > 0

        def make_core():
            if kind == "rnn":
                return nn.RNN(dim, dim, 1, nonlinearity="tanh", batch_first=True)
            if kind == "rnn_relu":
                return nn.RNN(dim, dim, 1, nonlinearity="relu", batch_first=True)
            if kind == "gru":
                return nn.GRU(dim, dim, 1, batch_first=True)
            return nn.LSTM(dim, dim, 1, batch_first=True)

        self.cores = nn.ModuleList([make_core() for _ in range(depth)])
        for core in self.cores:
            initialize_native_rnn_core(core, tanh_spectral_radius, relu_identity_scale)

        # A hierarchy child group is independent of its siblings so all groups
        # can stay in the batch dimension.  It can nevertheless start from
        # its already-causal parent context rather than an all-zero recurrent
        # state.  Keep a projection per recurrent layer: native stacked cores
        # have separate hidden (and, for LSTM, cell) states at every layer.
        state_width = dim * (2 if kind == "lstm" else 1)
        self.parent_state_projs = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, state_width))
            for _ in range(depth)
        ])

        norm_factory = {
            1: lambda: nn.BatchNorm1d(dim), 2: lambda: nn.LayerNorm(dim),
            3: lambda: RMSNorm(dim), 4: TTanh, 5: lambda: ETTanh(dim), 6: lambda: DyT(dim),
        }.get(self.norm_type)
        self.norms = None if norm_factory is None else nn.ModuleList([
            norm_factory() for _ in range(depth - 1)
        ])
        self.drops = None if self.dropout_p == 0 else nn.ModuleList([
            nn.Dropout(self.dropout_p) for _ in range(depth - 1)
        ])

        residual_layers = [
            index for index in range(depth - 1) if self.do_res and (index + 1) % self.res_every == 0
        ]
        self.res_mixers = None
        self.alphas = None
        self.betas = None
        if self.res_type in {0, 1} and residual_layers:
            width = dim if self.res_type == 0 else dim * 2
            self.res_mixers = nn.ModuleList([
                nn.Linear(width, dim) if index in residual_layers else nn.Identity()
                for index in range(depth - 1)
            ])
        elif self.res_type in {2, 3} and residual_layers:
            shape = 1 if self.res_type == 2 else dim
            self.alphas = nn.ParameterList([nn.Parameter(torch.zeros(shape)) for _ in residual_layers])
            self.betas = nn.ParameterList([nn.Parameter(torch.ones(shape)) for _ in residual_layers])

    def _zero_state(self, batch, device, dtype):
        if self.kind == "lstm":
            return [(torch.zeros(1, batch, self.dim, device=device, dtype=dtype),
                     torch.zeros(1, batch, self.dim, device=device, dtype=dtype)) for _ in self.cores]
        return [torch.zeros(1, batch, self.dim, device=device, dtype=dtype) for _ in self.cores]

    def initial_state_from_parent(self, parent_context):
        """Project causal parent context into independent per-layer RNN states.

        This deliberately does not accept a preceding sibling state.  Groups
        remain parallel; the hierarchy supplies only the parent context that
        was already available before this child group began.
        """
        if parent_context is None:
            return None
        states = []
        for projection in self.parent_state_projs:
            projected = torch.tanh(projection(parent_context))
            if self.kind == "lstm":
                hidden, cell = projected.chunk(2, dim=-1)
                # ``chunk`` returns strided views, while cuDNN's LSTM path
                # requires both hx tensors to be contiguous.
                states.append((hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous()))
            else:
                states.append(projected.unsqueeze(0).contiguous())
        return states

    def _apply_norm(self, index, x):
        if self.norms is None:
            return x
        if self.norm_type == 1:
            batch, time, dim = x.shape
            return self.norms[index](x.reshape(batch * time, dim)).reshape(batch, time, dim)
        return self.norms[index](x)

    def _apply_residual(self, index, x_in, x_out, residual_index):
        if not self.do_res or (index + 1) % self.res_every:
            return x_out, residual_index
        if self.res_type == 0:
            x_out = x_in + self.res_mixers[index](x_out)
        elif self.res_type == 1:
            x_out = self.res_mixers[index](torch.cat((x_out, x_in), dim=-1))
        else:
            x_out = x_out * self.alphas[residual_index] + x_in * self.betas[residual_index]
        return x_out, residual_index + 1

    def forward(self, x, state=None):
        if state is None:
            state = self._zero_state(x.size(0), x.device, x.dtype)
        next_state, residual_index = [], 0
        for index, (core, layer_state) in enumerate(zip(self.cores, state)):
            x_in = x
            if index:
                x = self._apply_norm(index - 1, x)
            x, layer_state = core(x, layer_state)
            if index < self.depth - 1:
                x, residual_index = self._apply_residual(index, x_in, x, residual_index)
                if self.drops is not None:
                    x = self.drops[index](x)
            next_state.append(layer_state)
        return x, next_state

    def step(self, x_t, state=None):
        x, state = self(x_t.unsqueeze(1), state)
        return x[:, 0], state



# ========= Custom RNN-like wrappers =========
# ==============================================================================
# COMPILE-SAFE WRAPPER & RNN CORES (Jiri's Fix)
# ==============================================================================

class CustomRNNWrapper(nn.Module):
    """
    Compiler-Safe Wrapper.
    1. Initializes Embedding explicitly (Fixes AttributeError: embed).
    2. Maps string names to classes manually.
    3. Handles the input projection flow correctly.
    """
    # Cells whose classes take (input_size, hidden_size, num_layers, **options).
    CELLS = {
        "indrnn": lambda: IndRNN, "indygru": lambda: IndyGRU, "janet": lambda: JANET,
        "liquid": lambda: LiquidRNN, "atanulstm": lambda: ExtATanULSTM,
        "indylstm": lambda: IndyLSTM, "irnn": lambda: IntersectionRNN, "ugrnn": lambda: UGRNN,
        "unicornn": lambda: UnICORNN,
        "lru": lambda: LightRecurrentUnit, "rru": lambda: RRU, "exprnn": lambda: ExpRNN,
        "mogrifier_lstm": lambda: (lambda *a, **k: MogrifierRNN(*a, cell="lstm", **k)),
        "mogrifier_gru": lambda: (lambda *a, **k: MogrifierRNN(*a, cell="gru", **k)),
    }

    def __init__(self, cell_type, vocab_size, hidden_size, num_layers=1, depth_options=None, **kwargs):
        super().__init__()
        # 1. Define Embedding (Crucial!)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        
        # 2. Define Head
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        # 3. Map & Instantiate Core
        # Ensure we pass hidden_size as input_size because embedding dim == hidden dim
        c_type = cell_type.lower()
        if c_type not in self.CELLS:
            raise ValueError(f"Unknown or unavailable cell type: {cell_type}")
        cell_cls = self.CELLS[c_type]()

        # Depth-wise options (norm / residual / dropout / multiplier / FFN) wrap
        # single-layer cores; without them the plain multi-layer cell keeps its
        # original parameter layout (and checkpoint compatibility).
        depth_options = {k: v for k, v in (depth_options or {}).items() if v}
        if depth_options:
            self.rnn = DepthwiseRNNStack(
                lambda: cell_cls(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, **kwargs),
                hidden_size, num_layers, **depth_options,
            )
        else:
            self.rnn = cell_cls(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                **kwargs
            )

    def forward_hidden(self, x, state=None):
        """Run the selected custom recurrent cell stack on hidden vectors."""
        return self.rnn(x, state)

    def forward(self, idx, state=None):
        out, state = self.forward_hidden(self.embed(idx), state)
        return self.lm_head(out), state


class IndRNN(nn.Module):
    """
    Compiler-Safe IndRNN.
    - Uses module attributes instead of dicts (Fixes 'getitem' error).
    - Uses .size() instead of unpacking (Fixes AssertionError).
    - Pre-allocates output tensors (Fixes Graph Breaks).
    """
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, activation="relu"):
        super().__init__()
        if activation not in ("relu", "tanh"):
            raise ValueError("IndRNN activation must be 'relu' or 'tanh'")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Create a simple Module to hold params (Fixes 'FloatTensor is not Module')
            layer = nn.Module()
            layer.W_ih = nn.Linear(input_size if i == 0 else hidden_size, hidden_size, bias=bias)
            
            # Direct attribute assignment (safest for Dynamo).  ReLU follows the
            # IndRNN paper: u in [0, 1] with |u| kept <= 1 so units can hold long
            # memories without exploding; tanh keeps the former short-memory init.
            u_range = (0.0, 1.0) if activation == "relu" else (-0.5, 0.5)
            layer.u_hh = nn.Parameter(torch.empty(hidden_size).uniform_(*u_range))
            layer.b_hh = nn.Parameter(torch.zeros(hidden_size)) if bias else None
            
            self.layers.append(layer)
            
        self.act = nn.ReLU() if activation == "relu" else nn.Tanh()

    def _recurrent_weight(self, layer):
        """ReLU IndRNN constrains |u| <= 1; the clamp passes gradients through
        unchanged (a projected update), as clipping the weights would."""
        u = layer.u_hh
        if self.activation == "relu":
            u = u + (u.clamp(-1.0, 1.0) - u).detach()
        return u

    def forward(self, x, state=None):
        # 1. Safe Unpacking (Fixes Dynamo AssertionError)
        B = x.size(0)
        T = x.size(1)
        
        # 2. Handle None state inside compiled graph
        if state is None:
            state = [x.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]
        
        new_states = []
        layer_input = x
        
        for i, layer in enumerate(self.layers):
            h = state[i]
            # 4. Attribute Access (Fixes 'getitem' error)
            # Compute input projection for whole sequence
            preact = layer.W_ih(layer_input)
            if kernels_available(preact):
                # Fused Triton time loop (forward and backward).
                y = indrnn_scan(preact, self._recurrent_weight(layer), layer.b_hh, h, self.activation)
                new_states.append(y[:, -1])
                layer_input = y
                continue

            # 3. Pre-allocate output (Fixes Append Graph Break)
            y = torch.empty((B, T, self.hidden_size), device=x.device, dtype=x.dtype)
            u = self._recurrent_weight(layer)
            b = layer.b_hh if layer.b_hh is not None else 0.0

            # 5. Fused Loop
            for t in range(T):
                z = preact[:, t, :] + h * u + b
                h = self.act(z)
                y[:, t, :] = h
                
            new_states.append(h)
            layer_input = y
            
        return layer_input, torch.stack(new_states)


class IndyGRU(nn.Module):
    """Compiler-Safe IndyGRU"""
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0.0, relu_gates=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.relu_gates = bool(relu_gates)   # experimental: ReLU instead of sigmoid gates
        self.dropout = float(dropout)
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = nn.Module()
            in_sz = input_size if i == 0 else hidden_size
            
            # Input projections
            layer.W_gate = nn.Linear(in_sz, 2 * hidden_size, bias=bias)
            layer.W_cand = nn.Linear(in_sz, hidden_size, bias=bias)
            
            # Diagonal Recurrent weights (Direct Attributes)
            layer.u_gate = nn.Parameter(torch.ones(2 * hidden_size) * 0.5)
            layer.u_cand = nn.Parameter(torch.ones(hidden_size) * 0.5)
            
            self.layers.append(layer)
            
        self._drop = nn.Dropout(self.dropout)

    def forward(self, x, state=None):
        B = x.size(0)
        T = x.size(1)
        
        if state is None:
            state = [x.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]

        layer_input = x
        new_states = []
        
        for i, layer in enumerate(self.layers):
            h = state[i]
            y = torch.empty((B, T, self.hidden_size), device=x.device, dtype=x.dtype)
            
            # Attribute Access
            gate_in = layer.W_gate(layer_input)
            cand_in = layer.W_cand(layer_input)
            u_gate = layer.u_gate
            u_cand = layer.u_cand

            if kernels_available(gate_in):
                # Fused Triton time loop (forward and backward).
                y = indygru_scan(gate_in, cand_in, u_gate, u_cand, h, self.relu_gates)
                h = y[:, -1]
            else:
                for t in range(T):
                    # Fused Gate Logic
                    gates = (torch.relu if self.relu_gates else torch.sigmoid)(gate_in[:, t] + h.repeat(1, 2) * u_gate)
                    r, z = gates.chunk(2, dim=1)

                    h_tilde = torch.tanh(cand_in[:, t] + (r * h) * u_cand)
                    h = (1 - z) * h + z * h_tilde
                    y[:, t] = h
                
            new_states.append(h)
            if i != self.num_layers - 1 and self.dropout > 0:
                y = self._drop(y)
            layer_input = y
            
        return layer_input, torch.stack(new_states)


class JANET(nn.Module):
    """Compiler-Safe JANET"""
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0.0, beta=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.beta = beta
        self.dropout = dropout
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = nn.Module()
            in_sz = input_size if i == 0 else hidden_size
            
            layer.W_f = nn.Linear(in_sz, hidden_size, bias=bias)
            layer.W_c = nn.Linear(in_sz, hidden_size, bias=bias)
            layer.U_f = nn.Linear(hidden_size, hidden_size, bias=bias)
            layer.U_c = nn.Linear(hidden_size, hidden_size, bias=bias)
            
            # Chrono Init Logic (Simplified)
            nn.init.constant_(layer.W_f.bias, 1.0)
            
            self.layers.append(layer)
        self._drop = nn.Dropout(dropout)

    def forward(self, x, state=None):
        B = x.size(0)
        T = x.size(1)
        
        if state is None:
            state = [x.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]
            
        layer_in = x
        new_states = []
        
        for i, layer in enumerate(self.layers):
            c = state[i]
            # Input projections for the whole sequence, with the recurrent
            # biases folded in: s = W_f x + U_f c (+ biases), same for c~.
            px_f = layer.W_f(layer_in)
            px_c = layer.W_c(layer_in)
            if layer.U_f.bias is not None:
                px_f = px_f + layer.U_f.bias
                px_c = px_c + layer.U_c.bias

            if kernels_available(layer_in):
                u = torch.cat((layer.U_f.weight, layer.U_c.weight), dim=0)
                y = janet_scan(torch.cat((px_f, px_c), dim=-1), u, c, self.beta)
                c = y[:, -1]
            else:
                y = torch.empty((B, T, self.hidden_size), device=x.device, dtype=x.dtype)
                for t in range(T):
                    # JANET (van der Westhuizen & Lasenby, 2018):
                    # c = f*c + (1 - sigmoid(s - beta)) * tanh(c~)
                    s = px_f[:, t] + F.linear(c, layer.U_f.weight)
                    cand = torch.tanh(px_c[:, t] + F.linear(c, layer.U_c.weight))
                    c = torch.sigmoid(s) * c + (1.0 - torch.sigmoid(s - self.beta)) * cand
                    y[:, t] = c
                
            new_states.append(c)
            if i != self.num_layers - 1 and self.dropout > 0:
                y = self._drop(y)
            layer_in = y
            
        return layer_in, torch.stack(new_states)

# ========= Fallback implementations if lstm.py not present =========
# (We only define these if imports failed; matches your provided code.)
if ExtIndRNN is None:
    class ExtIndRNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity="tanh",
                     bias=True, batch_first=True, dropout=0.0, bidirectional=False,
                     activation_modules: Optional[List[nn.Module]] = None):
            super().__init__()
            if nonlinearity not in ("tanh","relu"): raise ValueError("nonlinearity must be 'tanh' or 'relu'")
            self.input_size = input_size; self.hidden_size = hidden_size
            self.num_layers = num_layers; self.bias = bias
            self.batch_first = batch_first; self.dropout = float(dropout)
            self.bidirectional = bidirectional; self.num_directions = 2 if bidirectional else 1
            for layer in range(num_layers):
                suffix = f"_l{layer}"
                in_features = input_size if layer==0 else hidden_size*self.num_directions
                self.register_parameter("weight_ih"+suffix, nn.Parameter(torch.empty(hidden_size, in_features)))
                self.register_parameter("weight_hh"+suffix, nn.Parameter(torch.empty(hidden_size)))
                if bias:
                    self.register_parameter("bias_ih"+suffix, nn.Parameter(torch.empty(hidden_size)))
                    self.register_parameter("bias_hh"+suffix, nn.Parameter(torch.empty(hidden_size)))
            if activation_modules is not None:
                assert len(activation_modules) == num_layers * self.num_directions
                self._activations = nn.ModuleList(activation_modules)
            else:
                acts = [CapSech() if nonlinearity=="tanh" else nn.ReLU() for _ in range(num_layers*self.num_directions)]
                self._activations = nn.ModuleList(acts)
            self.reset_parameters()
        def _p(self, name): return getattr(self, name)
        def _get_params(self, layer, direction):
            suffix = f"_l{layer}"
            W_ih = self._p("weight_ih"+suffix); u_hh = self._p("weight_hh"+suffix)
            b_ih = self._p("bias_ih"+suffix) if self.bias else None
            b_hh = self._p("bias_hh"+suffix) if self.bias else None
            act = self._activations[layer*self.num_directions+direction]
            return W_ih, u_hh, b_ih, b_hh, act
        def reset_parameters(self):
            for layer in range(self.num_layers):
                W_ih, u_hh, b_ih, b_hh, _ = self._get_params(layer, 0)
                nn.init.xavier_uniform_(W_ih)
                nn.init.uniform_(u_hh, -0.5, 0.5)
                if self.bias:
                    fan_in = W_ih.size(1); bound_b = 1/math.sqrt(fan_in) if fan_in>0 else 0
                    nn.init.uniform_(b_ih, -bound_b, bound_b); nn.init.uniform_(b_hh, -bound_b, bound_b)
        def flatten_parameters(self): return
        def forward(self, x, hx=None):
            if self.batch_first: x = x.transpose(0,1)  # (T,B,C)
            T,B,_ = x.shape
            if hx is None: hx = x.new_zeros(self.num_layers, B, self.hidden_size)
            out = x
            finals = []
            for layer in range(self.num_layers):
                W_ih, u_hh, b_ih, b_hh, act = self._get_params(layer, 0)
                pre = torch.matmul(out, W_ih.t())
                if b_ih is not None: pre = pre + b_ih
                h = hx[layer]
                ys = []
                for t in range(T):
                    z = pre[t] + h * u_hh
                    if b_hh is not None: z = z + b_hh
                    h = act(z); ys.append(h)
                y = torch.stack(ys, dim=0)
                finals.append(h.unsqueeze(0))
                if layer != self.num_layers-1 and self.training: y = F.dropout(y, p=0.0)
                out = y
            if self.batch_first: out = out.transpose(0,1)
            return out, torch.cat(finals, dim=0)

if ExtATanULSTM is None:
    class ExtATanULSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                     batch_first=True, dropout=0.0, forget_bias=1.0):
            super().__init__()
            self.input_size = input_size; self.hidden_size = hidden_size
            self.num_layers = num_layers; self.bias = bias
            self.batch_first = batch_first; self.dropout = float(dropout)
            self.forget_bias = float(forget_bias)
            self.layers = nn.ModuleList()
            in_sizes = [input_size] + [hidden_size]*(num_layers-1)
            for in_sz in in_sizes:
                mod = nn.Module()
                mod.W_ih = nn.Parameter(torch.empty(4*hidden_size, in_sz))
                mod.W_hh = nn.Parameter(torch.empty(4*hidden_size, hidden_size))
                if bias:
                    mod.b_ih = nn.Parameter(torch.zeros(4*hidden_size))
                    mod.b_hh = nn.Parameter(torch.zeros(4*hidden_size))
                else:
                    mod.register_parameter('b_ih', None); mod.register_parameter('b_hh', None)
                self.layers.append(mod)
            self._drop = nn.Dropout(self.dropout); self.reset_parameters()
        def reset_parameters(self):
            H = self.hidden_size
            for mod in self.layers:
                nn.init.xavier_uniform_(mod.W_ih); nn.init.orthogonal_(mod.W_hh)
                if self.bias:
                    nn.init.zeros_(mod.b_ih); nn.init.zeros_(mod.b_hh)
                    mod.b_ih.data[H:2*H].add_(self.forget_bias)  # forget gate bias
        def _layer_forward(self, xseq, h0, c0, mod):
            T,B,_ = xseq.shape; H = self.hidden_size
            if kernels_available(xseq):
                # Fused Triton time loop; both biases folded into the input side.
                px = F.linear(xseq, mod.W_ih, mod.b_ih)
                if mod.b_hh is not None:
                    px = px + mod.b_hh
                y, c = atanu_lstm_scan(px.transpose(0, 1), mod.W_hh, h0, c0)
                return y.transpose(0, 1), y[:, -1], c
            h = h0; c = c0; outs=[]
            for t in range(T):
                gates = F.linear(xseq[t], mod.W_ih, mod.b_ih) + F.linear(h, mod.W_hh, mod.b_hh)
                i_lin, f_lin, g_lin, o_lin = gates.chunk(4, dim=1)
                i = asig_u(i_lin, k=2.0); f = asig_u(f_lin, k=2.0)
                g = atan_u(g_lin);       o = asig_u(o_lin, k=2.0)
                c = f * c + i * g
                h = o * atan_u(c)
                outs.append(h)
            return torch.stack(outs, dim=0), h, c
        def forward(self, x, hx=None):
            if self.batch_first: x = x.transpose(0,1)
            T,B,_ = x.shape
            if hx is None:
                h0 = x.new_zeros(self.num_layers, B, self.hidden_size)
                c0 = x.new_zeros(self.num_layers, B, self.hidden_size)
            else:
                h0,c0 = hx
            layer_in = x; hn=[]; cn=[]
            for li,mod in enumerate(self.layers):
                y,hT,cT = self._layer_forward(layer_in, h0[li], c0[li], mod)
                if li != self.num_layers-1 and self.dropout>0 and self.training:
                    y = self._drop(y)
                layer_in = y; hn.append(hT); cn.append(cT)
            y = layer_in
            hn = torch.stack(hn, dim=0); cn = torch.stack(cn, dim=0)
            if self.batch_first: y = y.transpose(0,1)
            return y,(hn,cn)

# ========= TCN =========
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, c_in, c_out, k, dilation):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation)
        self.pad = pad

    def forward(self, x):
        y = self.conv(x)
        return y[:, :, :-self.pad] if self.pad > 0 else y


class TCNBlock(nn.Module):
    def __init__(self, channels, act_name="relu", k=3, dilation=1):
        super().__init__()
        self.c1 = CausalConv1d(channels, channels, k, dilation)
        self.c2 = CausalConv1d(channels, channels, k, dilation)
        self.n1 = nn.LayerNorm(channels)
        self.n2 = nn.LayerNorm(channels)
        self.act = get_activation(act_name)

    def forward(self, x):
        h = self.c1(x)
        # Conv1d outputs [B, C, L], but LayerNorm expects [B, L, C]
        h = h.transpose(1, 2)
        h = self.n1(self.act(h))
        h = h.transpose(1, 2)

        h = self.c2(h)
        h = h.transpose(1, 2)
        h = self.n2(h)
        h = h.transpose(1, 2)

        return self.act(x + h)


class TemporalConvNet(nn.Module):
    def __init__(self, vocab_size, channels, n_layers, act_name="relu", k=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, channels)
        self.blocks = nn.Sequential(
            *[TCNBlock(channels, act_name=act_name, k=k, dilation=2**i) for i in range(n_layers)]
        )
        self.head = nn.Linear(channels, vocab_size)

    def forward(self, idx):
        x = self.embed(idx).transpose(1, 2)  # [B, C, L]
        x = self.blocks(x)
        x = x.transpose(1, 2)  # back to [B, L, C]
        return self.head(x)



# ========= GPT (simple) =========

# ================= GPT-2 (drop-in) =================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPT2Config:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    max_seq_len: int
    ff_mult: int = 4
    dropout: float = 0.0
    attn_dropout: float = 0.0
    bias: bool = True
    tie_weights: bool = True
    use_flash: bool = False   # uses F.scaled_dot_product_attention if available

def _resolve_act(act_name: str):
    # Use your project's get_activation if present; else GELU
    ga = globals().get("get_activation", None)
    if ga is not None:
        try:
            act = ga(act_name)
            if isinstance(act, nn.Module):
                return act
            # if it returned a function, wrap it
            class _Fn(nn.Module):
                def forward(self, x): return act(x)
            return _Fn()
        except Exception:
            pass
    class _GELU(nn.Module):
        def forward(self, x): return F.gelu(x)
    return _GELU()

class MultiHeadCausalAttn(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0.0, resid_dropout=0.0, bias=True, use_flash=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.hd = d_model // n_heads
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")
        self.qkv = nn.Linear(d_model, 3*d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)
        
        # === NEW: Post-SDPA Gate [cite: 858] ===
        self.gate = PostSDPAGate(d_model)
        # =======================================

    def forward(self, x, past_kv=None):
        # x is the normalized input (from Block: self.attn(self.ln1(x)))
        B,T,C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.hd).transpose(1,2)
        q, k, v = qkv[:,0].transpose(1,2), qkv[:,1].transpose(1,2), qkv[:,2].transpose(1,2)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        present = (k, v)

        # Build a causal mask relative to the *absolute* query positions.  A
        # plain ``is_causal=True`` is only correct when there is no KV cache;
        # with a cache, a one-token query would otherwise attend only to the
        # first cached key instead of every previous token.
        Tq, Tk = q.size(-2), k.size(-2)
        past_len = Tk - Tq
        query_positions = torch.arange(Tq, device=x.device).unsqueeze(-1) + past_len
        key_positions = torch.arange(Tk, device=x.device).unsqueeze(0)
        causal = key_positions <= query_positions

        # SDPA Calculation
        if self.use_flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=causal,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            att = (q @ k.transpose(-2,-1)) / math.sqrt(self.hd)
            att = att.masked_fill(~causal, float("-inf"))
            w = F.softmax(att, dim=-1)
            w = self.attn_drop(w)
            y = w @ v

        # Reshape SDPA output
        y = y.transpose(1,2).contiguous().view(B, T, C)
        
        # === NEW: Apply Gating ===
        # Applied after SDPA, before output projection (Wo) [cite: 1068, 1079]
        # x is used as the gating input dependency
        y = self.gate(x, y) 
        # =========================

        y = self.resid_drop(self.proj(y))
        return y, present

class MLP(nn.Module):
    def __init__(self, d_model, ff_mult=4, dropout=0.0, bias=True, act: nn.Module = None):
        super().__init__()
        self.fc = nn.Linear(d_model, ff_mult*d_model, bias=bias)
        self.act = act if act is not None else _resolve_act("gelu")
        self.proj = nn.Linear(ff_mult*d_model, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.proj(x)
        return self.drop(x)

class Block(nn.Module):
    def __init__(self, cfg: GPT2Config, act: nn.Module):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadCausalAttn(cfg.d_model, cfg.n_heads, cfg.attn_dropout, cfg.dropout, cfg.bias, cfg.use_flash)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.ff_mult, cfg.dropout, cfg.bias, act)

    def forward(self, x, past_kv=None):
        a, present = self.attn(self.ln1(x), past_kv=past_kv)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, present

class GPT2Core(nn.Module):
    """Core GPT-2; returns (logits, presents)."""
    def __init__(self, cfg: GPT2Config, act_name: str = "gelu"):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        act = _resolve_act(act_name)
        self.blocks = nn.ModuleList([Block(cfg, act) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.head.weight = self.tok.weight

        self.apply(self._init_weights)
        # GPT-2 residual proj scaling
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*cfg.n_layers))

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward_hidden(self, x, past_kv=None):
        """Apply GPT-2 positional/block/output core to hidden vectors."""
        B, T, _ = x.shape
        if T > self.cfg.max_seq_len:
            x = x[:, -self.cfg.max_seq_len:]; T = x.size(1)
        # Keep cached keys/values within the learned position-embedding range
        # and assign new tokens their positions after the retained context.
        past_len = 0
        if past_kv is not None and past_kv[0] is not None:
            past_len = past_kv[0][0].size(2)
            keep = max(0, self.cfg.max_seq_len - T)
            if past_len > keep:
                past_kv = [
                    (k[:, :, -keep:, :], v[:, :, -keep:, :]) if keep > 0 else
                    (k[:, :, :0, :], v[:, :, :0, :])
                    for k, v in past_kv
                ]
                past_len = keep
        pos = torch.arange(past_len, past_len + T, device=x.device).unsqueeze(0)
        x = x + self.pos(pos)
        x = self.drop(x)

        presents = []
        for i, block in enumerate(self.blocks):
            pkv = None if past_kv is None else past_kv[i]
            x, present = block(x, past_kv=pkv)
            presents.append(present)

        x = self.ln_f(x)
        return x, presents

    def forward(self, idx, past_kv=None):
        hidden, presents = self.forward_hidden(self.tok(idx), past_kv=past_kv)
        return self.head(hidden), presents

class GPT2ForLM(nn.Module):
    """Thin wrapper: return logits only to match your training code."""
    def __init__(self, cfg: GPT2Config, act_name="gelu"):
        super().__init__()
        self.core = GPT2Core(cfg, act_name)

    def forward(self, idx):
        logits, _ = self.core(idx, past_kv=None)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, repetition_penalty=1.0):
        self.eval()
        past = [None]*len(self.core.blocks)
        
        # Keep track of the full sequence for output, but feed core only the new token
        full_idx = idx
        
        # Initial step: feed full context to prime 'past'
        logits, past = self.core(idx, past_kv=past)
        next_tok = self._sample_token(logits, temperature, top_k, repetition_penalty, full_idx)
        full_idx = torch.cat([full_idx, next_tok], dim=1)
        
        # Generation loop
        for _ in range(max_new_tokens - 1):
            # Learned absolute position embeddings cannot be safely shifted in
            # an existing KV cache.  Once the context is full, rebuild the
            # cache from the latest window so cached keys and positions remain
            # aligned instead of reusing the final position indefinitely.
            if full_idx.size(1) >= self.core.cfg.max_seq_len:
                logits, past = self.core(
                    full_idx[:, -self.core.cfg.max_seq_len:], past_kv=None)
            else:
                logits, past = self.core(next_tok, past_kv=past)
            
            next_tok = self._sample_token(logits, temperature, top_k, repetition_penalty, full_idx)
            full_idx = torch.cat([full_idx, next_tok], dim=1)
            
        return full_idx

    def _sample_token(self, logits, temperature, top_k, repetition_penalty, full_idx):
        logits = logits[:, -1, :]
        if repetition_penalty != 1.0:
            for b in range(logits.size(0)):
                logits[b, full_idx[b].unique()] /= repetition_penalty
        if temperature != 1.0:
            logits = logits / temperature
        if top_k is not None and top_k < logits.size(-1):
            v, _ = torch.topk(logits, top_k)
            thresh = v[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < thresh, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
class ScanBlock_GateLoop(nn.Module):
    def __init__(self, dim: int, mag_act: str = "sigmoid",
                 clamp_mag: float = 0.995, floor_mag: float = 1e-3, ln_eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm(dim, eps=ln_eps)

        self.Wq = nn.Linear(dim, dim, bias=True)
        self.Wk = nn.Linear(dim, dim, bias=True)
        self.Wv = nn.Linear(dim, dim, bias=True)
        self.Wmag = nn.Linear(dim, dim, bias=True)
        self.Wtheta = nn.Linear(dim, dim, bias=True)

        self.post = GatedMLP(dim, mult=4, act_name="gelu")

        self._mag_fn = torch.sigmoid if mag_act == "sigmoid" else torch.sigmoid
        self._clamp_mag_hi = float(clamp_mag)
        self._clamp_mag_lo = float(floor_mag)

        # Gentle init
        for W in (self.Wq, self.Wk, self.Wv, self.Wmag, self.Wtheta):
            nn.init.xavier_uniform_(W.weight); nn.init.zeros_(W.bias)

    @torch.amp.autocast("cuda", enabled=False)  # complex math in full precision
    def _complex_a(self, xhat: torch.Tensor) -> torch.Tensor:
        # xhat is real; compute in float32 → complex64
        xr = xhat.float()
        mag = self._mag_fn(self.Wmag(xr)).clamp(self._clamp_mag_lo, self._clamp_mag_hi)  # [lo, hi]
        theta = self.Wtheta(xr)
        # torch.polar expects (real, real) => complex
        a = torch.polar(mag, theta).to(torch.cfloat)  # complex64
        return a

    def _gate_triplets(self, xhat: torch.Tensor):
        xr = xhat.float()
        q = torch.sigmoid(self.Wq(xr))      # [0,1]
        k = self.Wk(xr)
        v = self.Wv(xr)
        return q, k, v

    @torch.amp.autocast("cuda", enabled=False)
    def forward_seq(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        x_norm = self.ln(x)
        
        # 1. Compute Parameters
        # Magnitude and Phase for A
        mag = torch.sigmoid(self.Wmag(x_norm))
        theta = self.Wtheta(x_norm)
        
        # Create Complex A
        # (B,T,D) complex64
        a_complex = torch.polar(mag, theta)
        
        # Q, K, V
        q = torch.sigmoid(self.Wq(x_norm))
        k = self.Wk(x_norm)
        v = self.Wv(x_norm)
        
        # Input to scan: K * V (Complex)
        # Note: In standard GateLoop, input is just (K*V) complexified? 
        # Actually usually it's Real K, Real V -> Complex KV via some transform, 
        # or just treated as real input to complex state. 
        # Let's treat K*V as the complex input b_t (pure real, imag=0)
        kv_complex = (k * v).to(a_complex.dtype)
        
        # 2. Parallel Scan (Linear, Complex)
        # h_t = a_t * h_{t-1} + kv_t
        s0 = h0.to(a_complex.dtype) if h0 is not None else None
        h_complex = parallel_scan_linear(a_complex, kv_complex, s0)
        
        # 3. Output
        # y = Re(q * h)
        y = (q * h_complex.real)
        
        out = x + self.post(y)
        return out, h_complex[:, -1, :]

    @torch.amp.autocast("cuda", enabled=False)
    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        x_tn = self.ln(x_t)
        
        mag = torch.sigmoid(self.Wmag(x_tn))
        theta = self.Wtheta(x_tn)
        a_t = torch.polar(mag, theta)
        
        q = torch.sigmoid(self.Wq(x_tn))
        k = self.Wk(x_tn)
        v = self.Wv(x_tn)
        kv_t = (k * v).to(a_t.dtype)
        
        h_prev_c = h_prev.to(a_t.dtype) if h_prev is not None else torch.zeros_like(a_t)
        
        h_t = a_t * h_prev_c + kv_t
        y = q * h_t.real
        
        out = x_t + self.post(y)
        return out, h_t


class BlockDiagLinear(nn.Module):
    """
    Block-diagonal linear: split last dim into Nh heads of size d_h, apply per-head Linear(d_h->d_h).
    Used for recurrent ('R*') matrices to realize head-wise memory mixing (no cross-head mixing).
    """
    def __init__(self, dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.dh = dim // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, self.dh, self.dh))
        self.bias = nn.Parameter(torch.empty(num_heads, self.dh)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        for h in range(self.num_heads):
            nn.init.xavier_uniform_(self.weight[h])
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) or (B, D)
        is_seq = (x.dim() == 3)
        if not is_seq: x = x.unsqueeze(1)
        B,T,D = x.shape
        Nh, dh = self.num_heads, self.dh
        xh = x.view(B, T, Nh, dh)
        # y[b,t,h,:] = xh @ W[h].T + b[h]
        y = torch.einsum('bt hd, hkd->bt hk', xh, self.weight.transpose(1,2))
        if self.bias is not None:
            y = y + self.bias.view(1,1,Nh,dh)
        y = y.reshape(B, T, D)
        if not is_seq: y = y.squeeze(1)
        return y


class sLSTMCore(nn.Module):
    """
    sLSTM with exponential input/forget, normalizer n, stabilizer m, and head-wise memory mixing.
    Equations (8)-(17). Heads implemented via block-diagonal recurrent matrices. 
    """
    def __init__(self, dim: int, num_heads: int = 1, phi: str = "tanh", forget_activation: str = "exp"):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.phi = getattr(torch, "tanh") if phi == "tanh" else torch.nn.functional.silu
        assert forget_activation in ("exp", "sigmoid")
        self.forget_activation = forget_activation

        # Input projections (W*)
        self.Wz = nn.Linear(dim, dim, bias=True)
        self.Wi = nn.Linear(dim, dim, bias=True)
        self.Wf = nn.Linear(dim, dim, bias=True)
        self.Wo = nn.Linear(dim, dim, bias=True)

        # Recurrent (R*): block-diagonal mixing within heads only
        self.Rz = BlockDiagLinear(dim, num_heads, bias=True)
        self.Ri = BlockDiagLinear(dim, num_heads, bias=True)
        self.Rf = BlockDiagLinear(dim, num_heads, bias=True)
        self.Ro = BlockDiagLinear(dim, num_heads, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in [self.Wz, self.Wi, self.Wf, self.Wo]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _gate_forget(self, x):
        return torch.exp(x) if self.forget_activation == "exp" else torch.sigmoid(x)

    def forward(self, x: torch.Tensor, state=None):
        """
        x: (B,T,D); state is a dict with keys ['h','c','n','m'] each (B,D)
        Returns y:(B,T,D), new_state
        """
        B,T,D = x.shape
        if state is None:
            h = x.new_zeros(B, D); c = x.new_zeros(B, D); n = x.new_zeros(B, D)
            m = x.new_zeros(B, D)  # stabilizer state (log-domain max tracker)
        else:
            h = state["h"]; c = state["c"]; n = state["n"]; m = state["m"]

        exp_forget = self.forget_activation == "exp"
        if kernels_available(x) and self.phi is torch.tanh:
            # Fused Triton time loop.  Input projections for every gate run as
            # one batch here; every bias (W* and R*) is folded in up front.
            def rbias(lin):
                return 0.0 if lin.bias is None else lin.bias.reshape(D)
            gin = torch.stack((
                self.Wz(x) + rbias(self.Rz), self.Wi(x) + rbias(self.Ri),
                self.Wf(x) + rbias(self.Rf), self.Wo(x) + rbias(self.Ro),
            ), dim=2)
            # BlockDiagLinear computes h_head @ W[head]; the kernel wants rows.
            r = torch.stack([lin.weight.transpose(1, 2) for lin in (self.Rz, self.Ri, self.Rf, self.Ro)])
            y, c, n, m = slstm_scan(gin, r, h, c, n, m, exp_forget)
            return y, {"h": y[:, -1], "c": c, "n": n, "m": m}

        ys = []
        for t in range(T):
            xt = x[:, t, :]

            z_tilde = self.Wz(xt) + self.Rz(h)
            i_tilde = self.Wi(xt) + self.Ri(h)
            f_tilde = self.Wf(xt) + self.Rf(h)
            o_tilde = self.Wo(xt) + self.Ro(h)

            z = self.phi(z_tilde)
            o = torch.sigmoid(o_tilde)                # (14)

            # Stabilizer m_t = max( log f + m_{t-1}, log i )  (15)
            # Work in log space directly: exp() then log() overflows to inf.
            logf = f_tilde if exp_forget else F.logsigmoid(f_tilde)
            logi = i_tilde                             # log(exp(i_tilde))
            m_new = torch.maximum(logf + m, logi)

            # Stabilized gates i', f'   (16,17)
            i_hat = torch.exp(i_tilde - m_new)
            f_hat = torch.exp(logf + m - m_new)

            # State updates (8,9)
            c = f_hat * c + i_hat * z
            n = f_hat * n + i_hat

            # Hidden (10)
            h_tilde = c / torch.clamp(n, min=1e-12)
            h = o * h_tilde

            m = m_new
            ys.append(h.unsqueeze(1))

        y = torch.cat(ys, dim=1)
        new_state = {"h": h, "c": c, "n": n, "m": m}
        return y, new_state


class mLSTMCore(nn.Module):
    """
    mLSTM with matrix memory C (per head), covariance update, normalizer n, stabilized gates.
    Equations (19)-(27).  As in the paper, the input and forget gates are one
    scalar per head, which is what makes the normalizer n^T q and the
    stabilizer consistent with the matrix memory.  Training and sampling both
    use the exact chunkwise-parallel form (``mlstm_chunkwise``).
    """
    def __init__(self, dim: int, num_heads: int = 1, forget_activation: str = "exp"):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.dh = dim // num_heads
        assert forget_activation in ("exp", "sigmoid")
        self.forget_activation = forget_activation

        # Projections (head-shared for simplicity; split per head by reshape)
        self.Wq = nn.Linear(dim, dim, bias=True)
        self.Wk = nn.Linear(dim, dim, bias=True)
        self.Wv = nn.Linear(dim, dim, bias=True)
        self.Wo = nn.Linear(dim, dim, bias=True)

        # Gates i, f: one input-dependent scalar per head (25), (26).
        self.Wi = nn.Linear(dim, num_heads, bias=True)
        self.Wf = nn.Linear(dim, num_heads, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in [self.Wq, self.Wk, self.Wv, self.Wo, self.Wi, self.Wf]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor, state=None):
        """
        x: (B,T,D)
        state: dict with 'C' (B,Nh,dh,dh), 'n' (B,Nh,dh), 'm' (B,Nh), stored
        in the stabilized scale.
        Returns y:(B,T,D), new_state
        """
        B, T, D = x.shape
        Nh, dh = self.num_heads, self.dh

        def heads(t):
            return t.view(B, T, Nh, dh).transpose(1, 2)

        q = heads(self.Wq(x))                            # (22)
        k = heads(self.Wk(x)) / math.sqrt(dh)            # (23)
        v = heads(self.Wv(x))                            # (24)
        o = torch.sigmoid(self.Wo(x))                    # (27)
        log_i = self.Wi(x).transpose(1, 2)               # (25), log of exp gate
        f_tilde = self.Wf(x).transpose(1, 2)             # (26)
        log_f = f_tilde if self.forget_activation == "exp" else F.logsigmoid(f_tilde)

        # (19)-(21) with stabilizer (15)-(17): C <- f C + i v k^T,
        # n <- f n + i k, h~ = C q / max(|n^T q|, 1) in the unstabilized scale.
        h_tilde, new_state = mlstm_chunkwise(q, k, v, log_i, log_f, state)
        y = o * h_tilde.transpose(1, 2).reshape(B, T, D)
        return y, new_state


# ======= GatedMLP (scaled residual) =======
class GatedMLP(nn.Module):
    """Simple gated MLP with a learnable residual scale (safer in deep stacks)."""
    def __init__(self, dim: int, mult: float = 4/3, act_name: str = "gelu"):
        super().__init__()
        hid = int(dim * mult)
        self.fc1  = nn.Linear(dim, hid)
        self.fc2  = nn.Linear(hid, dim)
        self.gate = nn.Linear(dim, dim)
        self.act  = get_activation(act_name)
        # Residual scale starts modest to prevent early amplification
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(x))          # gate in [0,1]
        y = self.fc2(self.act(self.fc1(x)))      # payload
        return self.res_scale * (y * g)          # scaled residual payload


class XBlock_sLSTM(nn.Module):
    """Pre-LN residual block with sLSTM core and post up-projection (gated MLP)."""
    def __init__(self, dim: int, num_heads: int, act_name: str):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.core = sLSTMCore(dim, num_heads=num_heads, phi="tanh", forget_activation="exp")
        self.post = GatedMLP(dim, mult=4, act_name=act_name)

    def forward(self, x, state):
        x_norm = self.ln(x)
        y, new_state = self.core(x_norm, state)
        y = self.post(y)
        return x + y, new_state


class XBlock_mLSTM(nn.Module):
    """Pre-LN residual block with pre up-projection, mLSTM in high-dim, and down-projection."""
    def __init__(self, dim: int, num_heads: int, up_mult: float, act_name: str):
        super().__init__()
        up = int(dim * up_mult)
        self.ln = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, up)
        self.core = mLSTMCore(up, num_heads=num_heads, forget_activation="exp")
        self.down = nn.Linear(up, dim)
        self.out_gate = nn.Linear(dim, dim)   # externalized component-wise output gate
        self.skip = nn.Parameter(torch.tensor(1.0))  # learnable skip (Fig. 11)
        self.act = get_activation(act_name)

    def forward(self, x, state):
        x_norm = self.ln(x)
        u = self.up(x_norm)
        u = self.act(u)
        y, new_state = self.core(u, state)
        y = self.down(y)
        # externalized output gate like Fig. 11
        y = torch.sigmoid(self.out_gate(x_norm)) * y
        return x * self.skip + y, new_state


class XlstmLM(nn.Module):
    """
    Full xLSTM language model: embedding -> stacked blocks -> head.
    Exposes 3 configs via build_model:
      - xLSTM_s: only sLSTM blocks
      - xLSTM_m: only mLSTM blocks
      - xLSTM_mix: mixed with ratio cfg['xlstm_m_to_s'] (e.g., 7:1)
    """
    def __init__(self, vocab_size: int, dim: int, n_blocks: int, num_heads: int,
                 act_name: str, kind: str = "mix", m_to_s: Tuple[int,int] = (7,1), up_mult_m: float = 2.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.kind = kind
        self.blocks = nn.ModuleList()
        if kind == "s":
            for _ in range(n_blocks):
                self.blocks.append(XBlock_sLSTM(dim, num_heads, act_name))
        elif kind == "m":
            for _ in range(n_blocks):
                self.blocks.append(XBlock_mLSTM(dim, num_heads, up_mult_m, act_name))
        else:
            a,b = m_to_s
            pattern = ["m"]*a + ["s"]*b
            for i in range(n_blocks):
                t = pattern[i % len(pattern)]
                if t == "m":
                    self.blocks.append(XBlock_mLSTM(dim, num_heads, up_mult_m, act_name))
                else:
                    self.blocks.append(XBlock_sLSTM(dim, num_heads, act_name))
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, idx: torch.Tensor, state=None):
        """
        idx: (B,T) -> logits: (B,T,V), state is list of per-block states
        """
        B,T = idx.shape
        x = self.embed(idx)
        if state is None:
            state = [None]*len(self.blocks)
        new_states = []
        for blk, st in zip(self.blocks, state):
            x, st_new = blk(x, st)
            new_states.append(st_new)
        x = self.ln(x)
        logits = self.head(x)
        return logits, new_states
# ========= Parallel-scan utilities (linear recurrences) =========
# (parallel_scan_linear defined below, after log-space scan)
# ========= Log-space parallel scan (stable) =========
def parallel_scan_log(log_coeffs: torch.Tensor, log_values: torch.Tensor):
    """
    Compute all h_t for recurrence h_t = a_t ⊙ h_{t-1} + b_t using logs.
      log_coeffs: (B,T,D)  = log(a_1..T)
      log_values: (B,T+1,D)= [log(h0), log(b_1..T)]
    Returns: h (B,T,D)
    """
    # prefix log-products of a_t over time
    logA = torch.cumsum(log_coeffs, dim=1)          # (B,T,D)
    logA_pad = F.pad(logA, (0,0,1,0))               # (B,T+1,D) with logA_0 = 0

    # h_t = exp( logA_t + logsumexp_{k=0..t}( log_values[k] - logA_k ) )
    acc = torch.logcumsumexp(log_values - logA_pad, dim=1)  # (B,T+1,D)
    log_h = logA_pad[:, 1:, :] + acc[:, 1:, :]
    
    # === SAFETY FIX ===
    # Clamp to prevent float32 overflow (e^88 is approx max float32)
    # We clamp to 50.0 to be safe (e^50 is ~5e21, plenty for a hidden state)
    log_h = torch.clamp(log_h, max=50.0)
    
    return torch.exp(log_h)

# ========= Paper's positive surrogate g and its log =========
def pos_surrogate_g(x: torch.Tensor):
    # g(x) = { x+0.5 if x>=0;  sigmoid(x) otherwise }
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

def log_g(x: torch.Tensor):
    # log g(x) = { log(x+0.5) if x>=0;  -softplus(-x) otherwise }
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))
class ScanBlock_minGRU(nn.Module):
    def __init__(self, dim: int, log_space: bool = True):
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm(dim)
        self.Wz = nn.Linear(dim, dim) # Gate
        self.Wh = nn.Linear(dim, dim) # Candidate
        self.post = GatedMLP(dim, mult=4, act_name="gelu")
        
        # Init
        nn.init.xavier_uniform_(self.Wz.weight)
        nn.init.xavier_uniform_(self.Wh.weight)
        nn.init.constant_(self.Wz.bias, -4.0)
        nn.init.zeros_(self.Wh.bias)

    def forward_seq(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        # x: (B, T, D)
        x_norm = self.ln(x)
        
        # 1. Projections
        z_raw = self.Wz(x_norm)
        h_tilde_raw = self.Wh(x_norm)
        
        # 2. Calculate Log-Coeffs and Log-Values for Heinsen Scan
        # We need (1 - z) in log space.
        # log(1 - sigmoid(z)) = log(sigmoid(-z)) = -softplus(z)
        log_coeffs = -F.softplus(z_raw) 
        
        # We need (z * h_tilde) in log space.
        # log(sigmoid(z)) = -softplus(-z_raw)
        # h_tilde must be positive for log-scan -> use g(h)
        log_z = -F.softplus(-z_raw)
        log_h_tilde = log_g_act(h_tilde_raw)
        log_values = log_z + log_h_tilde
        
        # 3. Parallel Scan.  a = 1 - z in (0, 1) and b = z g(h~) >= 0, so the
        # recurrence is stable in linear space: run it on the fused affine-scan
        # kernel (the log-space form is kept for CPU / non-Triton setups).
        if kernels_available(x):
            h_seq = affine_scan(torch.exp(log_coeffs), torch.exp(log_values), h0)
        else:
            h_seq = heinsen_associative_scan_log(log_coeffs, log_values, h0)
        
        # 4. Output
        out = x + self.post(h_seq)
        return out, h_seq[:, -1, :]

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        x_tn = self.ln(x_t)
        
        # 1. Projections
        z_raw = self.Wz(x_tn)
        h_tilde_raw = self.Wh(x_tn)
        
        # 2. Gate and Candidate
        z = torch.sigmoid(z_raw)
        h_tilde = g_act(h_tilde_raw) # Crucial: Match g_act from training
        
        # 3. GRU Update
        # h_t = (1-z) * h_{t-1} + z * h_tilde
        h = (1.0 - z) * h_prev + z * h_tilde
        
        out = x_t + self.post(h)
        return out, h


class ScanBlock_minLSTM(nn.Module):
    def __init__(self, dim: int, log_space: bool = True):
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm(dim)
        self.Wf = nn.Linear(dim, dim)
        self.Wi = nn.Linear(dim, dim)
        self.Wh = nn.Linear(dim, dim)
        self.post = GatedMLP(dim, mult=4, act_name="gelu")
        
        # Init
        nn.init.xavier_uniform_(self.Wf.weight); nn.init.zeros_(self.Wf.bias)
        nn.init.xavier_uniform_(self.Wi.weight); nn.init.zeros_(self.Wi.bias)
        nn.init.xavier_uniform_(self.Wh.weight); nn.init.zeros_(self.Wh.bias)
        # Bias f to be open initially
        with torch.no_grad():
            self.Wf.bias.fill_(4.0)
            self.Wi.bias.fill_( -4.0)

    def forward_seq(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        x_norm = self.ln(x)
        
        f_raw = self.Wf(x_norm)
        i_raw = self.Wi(x_norm)
        h_tilde_raw = self.Wh(x_norm)

        # Log-space Gate Normalization
        # log(f) = -softplus(-f_raw)
        # log(i) = -softplus(-i_raw)
        log_f = -F.softplus(-f_raw)
        log_i = -F.softplus(-i_raw)
        
        # Normalization denominator: log(exp(log_f) + exp(log_i))
        log_denom = torch.logaddexp(log_f, log_i)
        
        # Normalized logs: f' = f / (f+i) -> log(f') = log(f) - log_denom
        log_f_prime = log_f - log_denom
        log_i_prime = log_i - log_denom
        
        # Values: i' * h_tilde
        log_values = log_i_prime + log_g_act(h_tilde_raw)
        
        # Scan (fused linear-space affine scan on CUDA, as in minGRU above)
        if kernels_available(x):
            h_seq = affine_scan(torch.exp(log_f_prime), torch.exp(log_values), h0)
        else:
            h_seq = heinsen_associative_scan_log(log_f_prime, log_values, h0)
        
        out = x + self.post(h_seq)
        return out, h_seq[:, -1, :]

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        x_tn = self.ln(x_t)
        
        f = torch.sigmoid(self.Wf(x_tn))
        i = torch.sigmoid(self.Wi(x_tn))
        h_tilde = g_act(self.Wh(x_tn))
        
        # Normalize
        denom = f + i + 1e-8
        f_prime = f / denom
        i_prime = i / denom
        
        h = f_prime * h_prev + i_prime * h_tilde
        out = x_t + self.post(h)
        return out, h

# Add this helper
def parallel_scan_linear(a: torch.Tensor, b: torch.Tensor, h0: Optional[torch.Tensor] = None):
    """
    Linear scan: h_t = a_t * h_{t-1} + b_t.
    
    Args:
        a: (B, T, D) recurrence coefficients
        b: (B, T, D) input signals
        h0: (B, D) optional initial state
    Returns:
        h: (B, T, D) all hidden states
    """
    # The former cumprod/divide formulation silently produced incorrect output
    # when a coefficient was zero, negative (MinRNN), or complex (GateLoop),
    # and its long-sequence branch cannot clamp complex tensors.  The scripted
    # recurrence is exact for every supported dtype and still avoids Python
    # dispatch inside the timestep loop.
    return pscan_linear_jit(a, b, h0)
# ======= ScanBlock_Mamba (Mamba-1) =======
class ScanBlock_Mamba(nn.Module):
    """Mamba block (Gu & Dao, 2023), following state-spaces/mamba
    ``mamba_simple.Mamba`` and ``selective_scan_ref``:

    pre-RMSNorm residual; in_proj -> (x, z); causal depthwise conv + SiLU;
    x_proj -> (dt, B, C); delta = softplus(dt_proj(dt)); A = -exp(A_log)
    (S4D-real init); x_t = exp(delta A) x_{t-1} + delta B u;
    y = (C.x + D u) * SiLU(z); out_proj.  Initialization matches the
    reference (dt log-uniform in [dt_min, dt_max] through an inverse-softplus
    bias, uniform dt_proj weight, D = 1).

    State for TBPTT / sampling: {'s': (B, d_inner, d_state),
    'buf': (B, k-1, d_inner) raw pre-conv inputs}.
    """
    def __init__(self, dim: int, kernel_size: int = 4, expand: int = 2, d_state: int = 16,
                 dt_rank="auto", dt_min: float = 1e-3, dt_max: float = 0.1,
                 dt_init_floor: float = 1e-4):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("Mamba kernel_size must be at least 1")
        self.dim = dim
        self.d_inner = int(expand * dim)
        self.d_state = int(d_state)
        self.k = int(kernel_size)
        self.dt_rank = math.ceil(dim / 16) if dt_rank == "auto" else int(dt_rank)

        self.norm = RMSNorm(dim, eps=1e-5)
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)
        # Causal padding is applied explicitly so the conv can continue a stream.
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, self.k, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward_seq(self, x: torch.Tensor, state: Optional[dict] = None):
        B, T, _ = x.shape
        if T == 0:
            raise ValueError("Mamba forward_seq requires at least one timestep")
        state = state or {}
        x_in, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)       # (B, T, d_inner)

        # The conv needs exactly k-1 preceding raw inputs (zeros at a stream start).
        buf = state.get("buf")
        if buf is None:
            buf = x_in.new_zeros(B, self.k - 1, self.d_inner)
        elif buf.size(1) > self.k - 1:
            buf = buf[:, buf.size(1) - (self.k - 1):]
        elif buf.size(1) < self.k - 1:
            buf = F.pad(buf, (0, 0, self.k - 1 - buf.size(1), 0))
        conv_in = torch.cat((buf.to(x_in.dtype), x_in), dim=1)      # (B, k-1+T, d_inner)
        u = F.silu(self.conv1d(conv_in.transpose(1, 2)).transpose(1, 2))

        dt, Bm, Cm = self.x_proj(u).split((self.dt_rank, self.d_state, self.d_state), dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        s0 = state.get("s")
        if s0 is None:
            s0 = u.new_zeros(B, self.d_inner, self.d_state, dtype=torch.float32)
        if kernels_available(u):
            y, s = selective_scan(u, delta, A, Bm, Cm, self.D, s0)
        else:
            y, s = selective_scan_reference(u, delta, A, Bm, Cm, self.D, s0)
        out = self.out_proj(y * F.silu(z))
        next_buf = conv_in[:, T:] if self.k > 1 else conv_in[:, :0]
        return x + out, {"s": s, "buf": next_buf}

    def step(self, x_t: torch.Tensor, state: Optional[dict] = None):
        """Single-step inference.  x_t: (B, D)"""
        y, state = self.forward_seq(x_t.unsqueeze(1), state)
        return y[:, 0], state


class ScanBlock_MambaSSM(ScanBlock_Mamba):
    """Mamba selective-SSM stage core (model 95): the same Mamba-1 block as
    model 14, kept as its own model ID and state-dict namespace."""
    def __init__(self, dim: int, kernel_size: int = 4, expand: int = 2, d_state: int = 16):
        super().__init__(dim, kernel_size=kernel_size, expand=expand, d_state=d_state)


class RWKVBlock(nn.Module):
    """RWKV-4 block (Peng et al., 2023), following BlinkDL/RWKV-LM RWKV-v4
    ``src/model.py``: token-shifted time mixing through the stabilized WKV
    operator with a learned per-channel decay and bonus (``time_first``),
    then token-shifted squared-ReLU channel mixing with a receptance gate;
    each a pre-LayerNorm residual.  Initialization follows the reference
    ("fancy init" and RWKV_Init).

    State: {'aa', 'bb', 'pp'} (stabilized WKV numerator, denominator and
    running maximum) plus the previous normalized inputs 'x_att', 'x_ffn'.
    """
    def __init__(self, dim: int, layer_id: int = 0, n_layer: int = 1):
        super().__init__()
        self.dim = dim
        self.ln_time = nn.LayerNorm(dim)
        self.ln_chan = nn.LayerNorm(dim)
        ratio_0_to_1 = layer_id / max(1, n_layer - 1)
        ratio_1_to_almost0 = 1.0 - layer_id / max(1, n_layer)
        ddd = torch.arange(dim, dtype=torch.float32) / dim
        pos = torch.arange(dim, dtype=torch.float32) / max(1, dim - 1)
        with torch.no_grad():
            self.time_decay = nn.Parameter(-5 + 8 * pos ** (0.7 + 1.3 * ratio_0_to_1))
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(dim)], dtype=torch.float32) * 0.5
            self.time_first = nn.Parameter(torch.full((dim,), math.log(0.3)) + zigzag)
            self.time_mix_k = nn.Parameter(ddd.pow(ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(ddd.pow(ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(ddd.pow(0.5 * ratio_1_to_almost0))
            self.ffn_time_mix_k = nn.Parameter(ddd.pow(ratio_1_to_almost0))
            self.ffn_time_mix_r = nn.Parameter(ddd.pow(ratio_1_to_almost0))

        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.ffn_key = nn.Linear(dim, 4 * dim, bias=False)
        self.ffn_receptance = nn.Linear(dim, dim, bias=False)
        self.ffn_value = nn.Linear(4 * dim, dim, bias=False)

        # RWKV_Init: zero-init these projections, orthogonal elsewhere
        # (gain sqrt(out/in) when a layer widens).
        for lin in (self.key, self.receptance, self.output, self.ffn_value, self.ffn_receptance):
            nn.init.zeros_(lin.weight)
        nn.init.orthogonal_(self.value.weight)
        nn.init.orthogonal_(self.ffn_key.weight, gain=2.0)

    def init_state(self, x: torch.Tensor):
        """Fresh state for a batch like ``x`` (B, D)."""
        B, D = x.size(0), self.dim
        zeros = lambda: x.new_zeros(B, D, dtype=torch.float32)
        return {"aa": zeros(), "bb": zeros(), "pp": torch.full_like(zeros(), -1e38),
                "x_att": x.new_zeros(B, D), "x_ffn": x.new_zeros(B, D)}

    def forward_seq(self, x: torch.Tensor, state: Optional[dict] = None):
        B, T, D = x.shape
        state = state or self.init_state(x[:, 0])
        # Time mixing.
        xa = self.ln_time(x)
        shifted = torch.cat((state["x_att"].to(xa.dtype).unsqueeze(1), xa[:, :-1]), dim=1)
        k = self.key(xa * self.time_mix_k + shifted * (1 - self.time_mix_k))
        v = self.value(xa * self.time_mix_v + shifted * (1 - self.time_mix_v))
        r = self.receptance(xa * self.time_mix_r + shifted * (1 - self.time_mix_r))
        w = -torch.exp(self.time_decay.float())
        wkv_fn = rwkv4_wkv if kernels_available(k) else rwkv4_wkv_reference
        wkv, aa, bb, pp = wkv_fn(w, self.time_first.float(), k, v,
                                 state["aa"].float(), state["bb"].float(), state["pp"].float())
        x = x + self.output(torch.sigmoid(r) * wkv)
        # Channel mixing.
        xf = self.ln_chan(x)
        shifted = torch.cat((state["x_ffn"].to(xf.dtype).unsqueeze(1), xf[:, :-1]), dim=1)
        kf = torch.square(torch.relu(self.ffn_key(xf * self.ffn_time_mix_k + shifted * (1 - self.ffn_time_mix_k))))
        rf = torch.sigmoid(self.ffn_receptance(xf * self.ffn_time_mix_r + shifted * (1 - self.ffn_time_mix_r)))
        x = x + rf * self.ffn_value(kf)
        return x, {"aa": aa, "bb": bb, "pp": pp, "x_att": xa[:, -1], "x_ffn": xf[:, -1]}

    def step(self, x_t: torch.Tensor, state: Optional[dict] = None):
        y, state = self.forward_seq(x_t.unsqueeze(1), state)
        return y[:, 0], state


def _rwkv7_head_size(dim: int) -> int:
    """RWKV-7 uses 64-wide heads; smaller models fall back to the largest
    power-of-two head width that divides the model width."""
    size = 64
    while size > 1 and dim % size:
        size //= 2
    return size


def _rwkv7_lora_dim(dim: int, factor: float) -> int:
    return max(32, int(round(factor * dim ** 0.5 / 32) * 32))


class RWKV7TimeMix(nn.Module):
    """RWKV-7 "Goose" time mixing (BlinkDL/RWKV-LM RWKV-v7, RWKV_Tmix_x070):
    token-shifted r/w/k/v/a/g, data-dependent decay w = exp(-exp(w_raw))
    with w_raw soft-clamped below -0.5, in-context learning rate a, removal
    key kk, value residual to the first layer's v, per-head GroupNorm, the
    r.k bonus term, and an output gate.  Initialization follows the training
    reference (train_temp/src/model.py)."""
    def __init__(self, dim: int, layer_id: int, n_layer: int, head_size: Optional[int] = None):
        super().__init__()
        C = dim
        N = head_size or _rwkv7_head_size(C)
        H = C // N
        self.layer_id, self.n_head, self.head_size = layer_id, H, N
        ratio_0_to_1 = layer_id / max(1, n_layer - 1)
        ratio_1_to_almost0 = 1.0 - layer_id / max(1, n_layer)
        ddd = (torch.arange(C, dtype=torch.float32) / C).view(1, 1, C)
        n = torch.arange(C, dtype=torch.float32)
        linear = n / max(1, C - 1) - 0.5
        zigzag = ((n % N) - (N - 1) / 2) / max((N - 1) / 2, 1e-9)
        zigzag = zigzag * zigzag.abs()
        www = -6 + 6 * (n / max(1, C - 1)) ** (1 + ratio_0_to_1 ** 0.3)

        def ortho(rows, cols, scale):
            weight = torch.empty(rows, cols)
            gain = math.sqrt(rows / cols) if rows > cols else 1
            nn.init.orthogonal_(weight, gain=gain * scale)
            return weight

        with torch.no_grad():
            self.x_r = nn.Parameter(1.0 - ddd.pow(0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - ddd.pow(0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - ddd.pow(0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - ddd.pow(0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - ddd.pow(0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - ddd.pow(0.2 * ratio_1_to_almost0))
            d_w, d_a, d_v, d_g = (_rwkv7_lora_dim(C, f) for f in (2.5, 2.5, 1.7, 5.0))
            self.w1 = nn.Parameter(torch.zeros(C, d_w))
            self.w2 = nn.Parameter(ortho(d_w, C, 0.1))
            self.w0 = nn.Parameter((www + 0.5 + zigzag * 2.5).view(1, 1, C))
            self.a1 = nn.Parameter(torch.zeros(C, d_a))
            self.a2 = nn.Parameter(ortho(d_a, C, 0.1))
            self.a0 = nn.Parameter((torch.zeros(C) - 0.19 + zigzag * 0.3 + linear * 0.4).view(1, 1, C))
            self.v1 = nn.Parameter(torch.zeros(C, d_v))
            self.v2 = nn.Parameter(ortho(d_v, C, 0.1))
            self.v0 = nn.Parameter((torch.zeros(C) + 0.73 - linear * 0.4).view(1, 1, C))
            self.g1 = nn.Parameter(torch.zeros(C, d_g))
            self.g2 = nn.Parameter(ortho(d_g, C, 0.1))
            self.k_k = nn.Parameter((torch.zeros(C) + 0.71 - linear * 0.1).view(1, 1, C))
            self.k_a = nn.Parameter(torch.full((1, 1, C), 1.02))
            self.r_k = nn.Parameter(torch.full((H, N), -0.04))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)
        with torch.no_grad():
            bound = C ** -0.5
            self.receptance.weight.uniform_(-0.5 * bound, 0.5 * bound)
            self.key.weight.uniform_(-0.05 * bound, 0.05 * bound)
            self.value.weight.uniform_(-0.5 * bound, 0.5 * bound)
            self.output.weight.zero_()
            self.ln_x.weight.fill_(((1 + layer_id) / n_layer) ** 0.7)

    def forward(self, x, prev, v_first, S):
        B, T, C = x.shape
        H, N = self.n_head, self.head_size
        xx = torch.cat((prev.to(x.dtype).unsqueeze(1), x[:, :-1]), dim=1) - x
        xr, xw, xk = x + xx * self.x_r, x + xx * self.x_w, x + xx * self.x_k
        xv, xa, xg = x + xx * self.x_v, x + xx * self.x_a, x + xx * self.x_g

        r = self.receptance(xr)
        w_raw = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        w = -F.softplus(-w_raw) - 0.5                       # soft-clamp to (-inf, -0.5)
        decay = torch.exp(-torch.exp(w.float()))
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)   # in-context learning rate
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = F.normalize((k * self.k_k).view(B, T, H, N), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)
        wkv_fn = rwkv7_wkv if kernels_available(r) else rwkv7_wkv_reference
        y, S = wkv_fn(r.float(), decay, k.float(), v.float(), (-kk).float(), (kk * a).float(), S)
        y = self.ln_x(y.view(B * T, C)).view(B, T, C)
        bonus = (r.view(B, T, H, N) * k.view(B, T, H, N) * self.r_k).sum(dim=-1, keepdim=True)
        y = y + (bonus * v.view(B, T, H, N)).view(B, T, C)
        return self.output(y * g), v_first, S


class RWKV7ChannelMix(nn.Module):
    """RWKV-7 channel mixing: token shift, squared ReLU, no receptance."""
    def __init__(self, dim: int, layer_id: int, n_layer: int):
        super().__init__()
        ratio_1_to_almost0 = 1.0 - layer_id / max(1, n_layer)
        ddd = (torch.arange(dim, dtype=torch.float32) / dim).view(1, 1, dim)
        with torch.no_grad():
            self.x_k = nn.Parameter(1.0 - ddd.pow(ratio_1_to_almost0 ** 4))
        self.key = nn.Linear(dim, 4 * dim, bias=False)
        self.value = nn.Linear(4 * dim, dim, bias=False)
        with torch.no_grad():
            self.key.weight.uniform_(-0.5 * dim ** -0.5, 0.5 * dim ** -0.5)
            self.value.weight.zero_()

    def forward(self, x, prev):
        xx = torch.cat((prev.to(x.dtype).unsqueeze(1), x[:, :-1]), dim=1) - x
        return self.value(torch.relu(self.key(x + xx * self.x_k)) ** 2)


class RWKV7Block(nn.Module):
    """RWKV-7 block: (ln0 on layer 0), time mix and channel mix as
    pre-LayerNorm residuals.  State: {'x_att', 'x_ffn', 'S' (B, H, N, N)}."""
    def __init__(self, dim: int, layer_id: int = 0, n_layer: int = 1):
        super().__init__()
        self.dim, self.layer_id = dim, layer_id
        self.ln0 = nn.LayerNorm(dim) if layer_id == 0 else None
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.att = RWKV7TimeMix(dim, layer_id, n_layer)
        self.ffn = RWKV7ChannelMix(dim, layer_id, n_layer)

    def init_state(self, x: torch.Tensor):
        B, H, N = x.size(0), self.att.n_head, self.att.head_size
        return {"x_att": x.new_zeros(B, self.dim), "x_ffn": x.new_zeros(B, self.dim),
                "S": x.new_zeros(B, H, N, N, dtype=torch.float32)}

    def forward(self, x, v_first=None, state=None):
        state = state or self.init_state(x[:, 0])
        if self.ln0 is not None:
            x = self.ln0(x)
        xa = self.ln1(x)
        att, v_first, S = self.att(xa, state["x_att"], v_first, state["S"])
        x = x + att
        xf = self.ln2(x)
        x = x + self.ffn(xf, state["x_ffn"])
        return x, v_first, {"x_att": xa[:, -1], "x_ffn": xf[:, -1], "S": S}


class RWKV7Stack(nn.Module):
    """A stack of RWKV-7 blocks sharing the first layer's values (the value
    residual), with the ``forward_seq`` / ``step`` interface of a scan block."""
    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList([RWKV7Block(dim, i, depth) for i in range(depth)])

    def forward_seq(self, x, state=None):
        state = state or [None] * len(self.blocks)
        v_first, new_state = None, []
        for block, st in zip(self.blocks, state):
            x, v_first, st = block(x, v_first, st)
            new_state.append(st)
        return x, new_state

    def step(self, x_t, state=None):
        y, state = self.forward_seq(x_t.unsqueeze(1), state)
        return y[:, 0], state


class RWKV7LM(nn.Module):
    """RWKV-7 "Goose" language model (embedding -> RWKV-7 blocks -> head)."""
    def __init__(self, vocab_size: int, dim: int, depth: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.stack = RWKV7Stack(dim, depth)
        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        with torch.no_grad():
            nn.init.uniform_(self.embed.weight, -1e-4, 1e-4)
            gain = 0.5 * math.sqrt(vocab_size / dim) if vocab_size > dim else 0.5
            nn.init.orthogonal_(self.head.weight, gain=gain)

    def forward(self, idx, state=None):
        x, state = self.stack.forward_seq(self.embed(idx), state)
        return self.head(self.ln_out(x)), state


class ScanLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int, kind: str,
                 n_blocks: int = 2, mamba_kernel: int = 4, log_space: bool = True,
                 minrnn_act: int = 0): # Added minrnn_act arg
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.emb_ln = nn.LayerNorm(dim) 
        self.kind = kind
        self.blocks = nn.ModuleList()
        
        for layer_id in range(n_blocks):
            if kind == "mingru":
                self.blocks.append(ScanBlock_minGRU(dim, log_space=log_space))
            elif kind == "minrnn":
                # Use the new Generalized Block
                self.blocks.append(ScanBlock_MinRNN_Gen(dim, act_type=minrnn_act))
            elif kind == "minlstm":
                self.blocks.append(ScanBlock_minLSTM(dim, log_space=log_space))
            elif kind == "mamba":
                self.blocks.append(ScanBlock_Mamba(dim, kernel_size=mamba_kernel))
            elif kind == "mamba_ssm":
                self.blocks.append(ScanBlock_MambaSSM(dim, kernel_size=mamba_kernel))
            elif kind == "rwkv":
                self.blocks.append(RWKVBlock(dim, layer_id=layer_id, n_layer=n_blocks))
            elif kind == "gateloop":
                self.blocks.append(ScanBlock_GateLoop(dim))
            elif kind == "minindrnn":
                self.blocks.append(ScanBlock_MinIndRNN(dim, act_type=minrnn_act))
            elif kind == "minjanet":
                self.blocks.append(ScanBlock_MinJANET(dim))
            elif kind == "minindygru":  # <--- NEW
                self.blocks.append(ScanBlock_MinIndyGRU(dim, log_space=log_space))
            elif kind == "minindylstm": # <--- NEW
                self.blocks.append(ScanBlock_MinIndyLSTM(dim, log_space=log_space))
            else:
                raise ValueError(f"Unknown scan kind: {kind}")
                
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward_hidden(self, x: torch.Tensor, state=None):
        """Run the scan/SSM core on already embedded hidden vectors."""

        # === TRAINING ===
        # If state is provided, we are doing TBPTT and must thread initial states through.
        if self.training and (state is not None):
            new_states = []
            for b, st in zip(self.blocks, (state if isinstance(state, list) else [None]*len(self.blocks))):
                if isinstance(b, (ScanBlock_Mamba, ScanBlock_MambaSSM, RWKVBlock)):
                    x, st_new = b.forward_seq(x, state=st)
                else:
                    # minGRU / minLSTM: pass initial h0 = Tensor (B,D) or None
                    h0 = st if (torch.is_tensor(st) or st is None) else None
                    x, h_last = b.forward_seq(x, h0=h0)
                    st_new = h_last  # carry last hidden as state
                new_states.append(st_new)
            return self.ln(x), new_states

        # === TRAINING (no TBPTT) ===
        if self.training:
            st = None
            for b in self.blocks:
                if isinstance(b, (ScanBlock_Mamba, ScanBlock_MambaSSM, RWKVBlock)):
                    x, st = b.forward_seq(x, state=None)   # stateless
                else:
                    x, _ = b.forward_seq(x, h0=None)
            return self.ln(x), None

        # === EVAL / SEQUENTIAL ===
        if state is None:
            state = [None] * len(self.blocks)

        outs = []
        for t in range(x.size(1)):
            x_t = x[:, t, :]  # (B,D)
            new_states = []
            for b, st in zip(self.blocks, state):
                if isinstance(b, (ScanBlock_Mamba, ScanBlock_MambaSSM)):
                    x_t, st_new = b.step(x_t, st or {})
                elif isinstance(b, RWKVBlock):
                    x_t, st_new = b.step(x_t, st or b.init_state(x_t))
                else:
                    h_prev = st if st is not None else x_t.new_zeros(x_t.size(0), x_t.size(-1))
                    x_t, st_new = b.step(x_t, h_prev)
                new_states.append(st_new)
            state = new_states
            outs.append(x_t.unsqueeze(1))
        y = torch.cat(outs, dim=1)
        y = self.ln(y)
        return y, state

    def forward(self, idx: torch.Tensor, state=None):
        hidden, state = self.forward_hidden(self.emb_ln(self.embed(idx)), state)
        return self.head(hidden), state


class JanetRNN(nn.Module):
    """
    Multi-layer JANET (forget-gate-only LSTM), batch_first=True.

    Equations (JANET):
        f_t = σ(U_f h_{t-1} + W_f x_t + b_f)
        c~_t = tanh(U_c h_{t-1} + W_c x_t + b_c)
        c_t = f_t ⊙ c_{t-1} + (1 - σ(U_f h_{t-1} + W_f x_t + b_f - β)) ⊙ c~_t
        h_t = c_t

    State we carry per layer = c_t (shape (B, H)).
    Returned state = stacked (num_layers, B, H) tensor (like IndRNN).
    """
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=True, dropout=0.0, beta: float = 1.0,
                 chrono_Tmax: Optional[int] = None):
        super().__init__()
        assert batch_first, "This JanetRNN expects batch_first=True"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = float(dropout)
        self.beta = float(beta)
        self.chrono_Tmax = chrono_Tmax  # used for chrono init of forget biases

        self.layers = nn.ModuleList()
        in_sizes = [input_size] + [hidden_size]*(num_layers-1)
        for in_sz in in_sizes:
            mod = nn.Module()
            # f gate projections
            mod.W_f = nn.Linear(in_sz, hidden_size, bias=bias)
            mod.U_f = nn.Linear(hidden_size, hidden_size, bias=bias)
            # candidate c~ projections
            mod.W_c = nn.Linear(in_sz, hidden_size, bias=bias)
            mod.U_c = nn.Linear(hidden_size, hidden_size, bias=bias)
            self.layers.append(mod)

        self._drop = nn.Dropout(self.dropout)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        # Xavier-uniform for inputs; orthogonal for hidden like LSTM practice
        for li, mod in enumerate(self.layers):
            nn.init.xavier_uniform_(mod.W_f.weight); nn.init.xavier_uniform_(mod.W_c.weight)
            nn.init.orthogonal_(mod.U_f.weight);     nn.init.orthogonal_(mod.U_c.weight)
            if self.bias:
                # Start b_c at zero
                nn.init.zeros_(mod.W_c.bias); nn.init.zeros_(mod.U_c.bias)
                # Chrono init for forget gate bias b_f
                Tmax = self.chrono_Tmax if (self.chrono_Tmax is not None and self.chrono_Tmax >= 2) else 2
                # b_f ~ log(U[1, Tmax-1]); shape (H,)
                low = torch.ones(self.hidden_size)
                high = torch.full((self.hidden_size,), float(max(1, Tmax - 1)))
                bf = torch.log(torch.rand_like(low) * (high - low) + low)
                # We have two biases contributing to f preact: W_f.bias and U_f.bias; split bf across them.
                if self.bias:
                    nn.init.zeros_(mod.W_f.bias); nn.init.zeros_(mod.U_f.bias)
                    mod.W_f.bias.add_(0.5 * bf)
                    mod.U_f.bias.add_(0.5 * bf)

    def forward(self, x, state=None):
        # x: (B,T,in)
        B, T, _ = x.shape
        # state: (num_layers,B,H) with carried c_t per layer
        if state is None:
            c_list = [x.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]
        else:
            assert torch.is_tensor(state) and state.shape == (self.num_layers, B, self.hidden_size)
            c_list = [state[li] for li in range(self.num_layers)]

        layer_in = x
        new_c = []
        for li, mod in enumerate(self.layers):
            c = c_list[li]
            outs = []
            # precompute affine terms
            # Note: using explicit loop over T to keep it simple & consistent with other custom cores
            for t in range(T):
                xt = layer_in[:, t, :]
                f = torch.sigmoid(mod.W_f(xt) + mod.U_f(c))                      # f_t
                cand = torch.tanh(mod.W_c(xt) + mod.U_c(c))                      # c~_t
                # apply beta shift on the (1 - f) branch as in the paper (β=1 recommended)
                one_minus_f_beta = 1.0 - torch.sigmoid(mod.W_f(xt) + mod.U_f(c) - self.beta)
                c = f * c + one_minus_f_beta * cand
                outs.append(c.unsqueeze(1))
            y = torch.cat(outs, dim=1)  # (B,T,H)
            if li != self.num_layers-1 and self.training and self.dropout > 0.0:
                y = self._drop(y)
            layer_in = y
            new_c.append(c)
        # h_t = c_t
        out = layer_in
        new_state = torch.stack(new_c, dim=0)  # (L,B,H)
        return out, new_state
# ========= HyperMixer =========
class MultiHeadHyperMixing(nn.Module):
    def __init__(self, d_model, d_hidden, n_heads=4, tie_in_out=True, dropout=0.0, causal=True, act_name="gelu"):
        super().__init__()
        assert d_hidden % n_heads == 0, "d_hidden must be divisible by n_heads"
        self.d_model = int(d_model)
        self.d_hidden = int(d_hidden)
        self.n_heads = int(n_heads)
        self.d_head  = self.d_hidden // self.n_heads
        self.tie = bool(tie_in_out)
        self.causal = bool(causal)

        # Hypernets output per-token, per-head parameters
        out_dim = self.n_heads * self.d_head
        self.hyper_in  = nn.Sequential(nn.Linear(d_model, d_model), get_activation(act_name), nn.Linear(d_model, out_dim))
        self.hyper_out = nn.Sequential(nn.Linear(d_model, d_model), get_activation(act_name), nn.Linear(d_model, out_dim))

        self.out_proj = nn.Linear(self.n_heads * d_model, d_model)  # fuse heads
        self.ln_out = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape

        # Hyper weights per token → reshape to heads
        W1 = self.hyper_in(x).view(B, T, self.n_heads, self.d_head)            # (B, T, H, Dh)
        W2 = W1 if self.tie else self.hyper_out(x).view(B, T, self.n_heads, self.d_head)  # (B, T, H, Dh)

        # Shared values across heads (like V without per-head projection)
        # values: (B, D, T)
        values = x.transpose(1, 2)

        # Build per-head token kernels: K[b, h, τ, t]
        #   K_h[τ,t] = <W2[τ,h,:], W1[t,h,:]>
        # einsum over Dh
        # W1: (B, T, H, Dh); W2: (B, T, H, Dh) → K: (B, H, T(τ), T(t))
        K = torch.einsum('bthk,bshk->bhst', W1, W2)

        if self.causal:
            # causal mask over (τ, t): only t <= τ
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            # Use -inf for masked positions to zero them after softmax-like use; here it's linear mixing,
            # so we can just zero the masked positions directly:
            K = K * mask  # broadcast over B,H

        # Mix values with each head's kernel:
        # y_h[b, h, d, τ] = sum_t values[b, d, t] * K[b, h, τ, t]
        # → (B, H, D, T)
        y_heads = torch.einsum('bdt,bhst->bhds', values, K)

        # Reorder to (B, T, H*D) then fuse
        y_heads = y_heads.permute(0, 3, 1, 2).contiguous()   # (B, T, H, D)
        y_cat   = y_heads.view(B, T, self.n_heads * D)       # (B, T, H*D)
        y       = self.out_proj(y_cat)                       # (B, T, D)

        y = self.drop(y)
        return self.ln_out(y)



class _FeatureMLP(nn.Module):
    """Feature mixing MLP (token-wise) with your activation registry + gated style."""
    def __init__(self, d_model: int, d_ff: int, act_name: str = "gelu", dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = get_activation(act_name)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        z = self.fc2(self.act(self.fc1(self.ln(x))))
        return self.drop(z)

class HyperMixerBlock(nn.Module):
    """
    Pre-LN residual block:
      x = x + MultiHeadHyperMixing(x)
      x = x + FeatureMLP(x)
    """
    def __init__(self, d_model: int, d_hidden: int, d_ff: int, act_name: str = "gelu",
                 tie_hyper: bool = True, drop_token: float = 0.0, drop_ff: float = 0.0,
                 n_heads: int = 4, causal: bool = True):
        super().__init__()
        self.tmix = MultiHeadHyperMixing(
            d_model, d_hidden, n_heads=n_heads,
            tie_in_out=tie_hyper, dropout=drop_token, causal=causal, act_name=act_name
        )
        self.fmix = _FeatureMLP(d_model, d_ff, act_name=act_name, dropout=drop_ff)

    def forward(self, x):
        x = x + self.tmix(x)
        x = x + self.fmix(x)
        return x


class HyperMixerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 d_hidden: int = None, d_ff: int = None, act_name: str = "gelu",
                 max_seq_len: int = 65536, tie_hyper: bool = True, dropout: float = 0.0,
                 n_heads: int = 4, causal: bool = True):
        super().__init__()
        d_hidden = int(d_hidden or max(64, d_model))
        d_ff     = int(d_ff or (4 * d_model))
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        self.blocks = nn.ModuleList([
            HyperMixerBlock(d_model, d_hidden, d_ff, act_name=act_name,
                            tie_hyper=tie_hyper, drop_token=dropout, drop_ff=dropout,
                            n_heads=n_heads, causal=causal)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

    def forward_hidden(self, x: torch.Tensor):
        """Apply the positional HyperMixer core to hidden vectors."""
        T = x.size(1)
        if T > self.max_seq_len:
            x = x[:, -self.max_seq_len:]
            T = x.size(1)
        x = x + self.pos(T, device=x.device)
        for blk in self.blocks:
            x = blk(x)
        return self.ln(x)

    def forward(self, idx: torch.Tensor):
        return self.head(self.forward_hidden(self.tok(idx)))

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return self.weight * x_normed

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, bias=False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GEGLU(SwiGLU):
    """GELU-gated feed-forward network with SwiGLU-compatible dimensions."""
    def forward(self, x):
        return self.w3(F.gelu(self.w1(x)) * self.w2(x))


class MiGLU(SwiGLU):
    """Mish-gated feed-forward network with SwiGLU-compatible dimensions."""
    def forward(self, x):
        return self.w3(F.mish(self.w1(x)) * self.w2(x))


def make_feed_forward(dim, hidden_dim, act_name="gelu", bias=True, dropout=0.0):
    """Build a standard or true gated feed-forward projection."""
    name = (act_name or "gelu").lower()
    gated = {"swiglu": SwiGLU, "geglu": GEGLU, "miglu": MiGLU}
    if name in gated:
        return gated[name](dim, hidden_dim, bias=bias)
    return nn.Sequential(
        nn.Linear(dim, hidden_dim, bias=bias),
        get_activation(name),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim, bias=bias),
    )

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=65536):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self.cached_cos = None
        self.cached_sin = None

    def forward(self, x, seq_len=None, position_offset=0):
        # x: [B, T, n_heads, head_dim]
        if seq_len is None:
            seq_len = x.size(1)
        required_len = position_offset + seq_len
        if required_len > self.max_seq_len:
            raise ValueError(
                f"RoPE sequence length {required_len} exceeds configured maximum "
                f"{self.max_seq_len}."
            )

        cache_invalid = (
            self.cached_cos is None
            or self.cached_cos.size(1) < required_len
            or self.cached_cos.device != x.device
            or self.cached_cos.dtype != x.dtype
        )
        if cache_invalid:
            t = torch.arange(required_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cached_cos = emb.cos().to(dtype=x.dtype)[None, :, None, :]
            self.cached_sin = emb.sin().to(dtype=x.dtype)[None, :, None, :]

        return (
            self.cached_cos[:, position_offset:required_len],
            self.cached_sin[:, position_offset:required_len],
        )

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [B, T, H, D]
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# ==============================================================================
# 1. MinRNN (ScanBlock Variant)
# ==============================================================================
# ==============================================================================
# 1. MinRNN (ScanBlock Variant) - Tanh Edition
# ==============================================================================
class ScanBlock_MinRNN(nn.Module):
    """
    Minimal RNN compatible with parallel scan.
    Modified to use Tanh for the recurrence gate to improve gradient flow
    in deep networks ("punch through").
    
    Formulation:
    h_t = a_t * h_{t-1} + b_t
       where a_t = tanh(W_z x_t + bias)
             b_t = W_x x_t
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.ln = RMSNorm(dim)
        self.Wx = nn.Linear(dim, dim, bias=False)
        
        # Use bias=True for Wz to allow initializing near boundary (high memory)
        self.Wz = nn.Linear(dim, dim, bias=False) 
        self.out = nn.Linear(dim, dim, bias=False)
        
        # Init Wz bias to 2.0 so tanh(bias) ~= 0.96 (Long memory init)
        #nn.init.constant_(self.Wz.bias, 2.0)
        nn.init.xavier_uniform_(self.Wz.weight, gain=0.1) # Small random jitter

    def forward_seq(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        x_norm = self.ln(x)
        
        # Candidate / Input (b_t)
        b = self.Wx(x_norm)
        
        # Decay (a_t) using Tanh
        # Tanh allows a range of (-1, 1). 
        # Gradients are steeper (max 1.0) compared to sigmoid (max 0.25).
        a = torch.tanh(self.Wz(x_norm))
        
        # Linear Scan
        h = parallel_scan_linear(a, b, h0)
        
        return self.out(h + x), h[:, -1, :]

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        x_norm = self.ln(x_t)
        
        # Candidate
        b = self.Wx(x_norm)
        
        # Decay
        a = torch.tanh(self.Wz(x_norm))
        
        # Update
        h = a * h_prev + b
        
        return self.out(h + x_t), h


# ==============================================================================
# 2. Causal MLPMixer
# ==============================================================================
class CausalMixingBlock(nn.Module):
    """
    Autoregressive Mixer:
    1. Token Mixing (Time): Causal Masked Linear
    2. Channel Mixing (Feature): Standard MLP
    """
    def __init__(self, dim, seq_len, act_name="gelu"):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
        # Time Mixing: (B, D, T) -> (B, D, T) with causal mask
        self.time_mix = nn.Linear(seq_len, seq_len)
        self.register_buffer("causal_mask", torch.tril(torch.ones(seq_len, seq_len)))
        
        # Channel Mixing
        self.channel_mix = nn.Sequential(
            nn.Linear(dim, 4*dim),
            get_activation(act_name),
            nn.Linear(4*dim, dim)
        )

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        shortcut = x
        x = self.ln1(x)
        
        # Time mix: Transpose to (B, D, T)
        x = x.transpose(1, 2)
        
        # Apply masked linear manually for causality
        # y = x @ W.T + b
        # We need W to be masked. 
        # Ideally, we crop the weight matrix to T x T and mask it.
        W = self.time_mix.weight[:T, :T] * self.causal_mask[:T, :T]
        b = self.time_mix.bias[:T]
        
        x = F.linear(x, W, b)
        
        x = x.transpose(1, 2) # Back to (B, T, D)
        x = x + shortcut
        
        # Channel mix
        shortcut = x
        x = self.ln2(x)
        x = self.channel_mix(x)
        x = x + shortcut
        return x

    def step(self, x_t, state=None):
        """Run one Mixer item from the cached LayerNorm-1 prefix."""
        if state is None:
            normalized_history = x_t.new_empty(x_t.size(0), 0, x_t.size(-1))
        else:
            normalized_history = state["normalized_history"]
        position = normalized_history.size(1)
        if position >= self.time_mix.in_features:
            raise ValueError(
                f"Causal MLPMixer stage cache exceeded configured sequence length "
                f"{self.time_mix.in_features}; restart the local hierarchy group before stepping again"
            )

        normalized_history = torch.cat((normalized_history, self.ln1(x_t).unsqueeze(1)), dim=1)
        row = self.time_mix.weight[position:position + 1, :position + 1]
        mixed = F.linear(normalized_history.transpose(1, 2), row, self.time_mix.bias[position:position + 1])
        x_t = x_t + mixed.squeeze(-1)
        return x_t + self.channel_mix(self.ln2(x_t)), {"normalized_history": normalized_history}

class CausalMLPMixer(nn.Module):
    def __init__(self, vocab_size, dim, depth, seq_len, act_name="gelu"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([CausalMixingBlock(dim, seq_len, act_name=act_name) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.seq_len = seq_len

    def forward_hidden(self, x):
        if x.size(1) > self.seq_len:
            x = x[:, -self.seq_len:]
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

    def forward(self, idx):
        return self.head(self.forward_hidden(self.embed(idx)))

# ==============================================================================
# 3. Modern Transformer (Llama Style)
# ==============================================================================
class ModernAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads=None, max_seq_len=10000):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = dim // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires an even head_dim; got dim={dim}, n_heads={n_heads}, head_dim={self.head_dim}"
            )
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
        # === NEW: Post-SDPA Gate [cite: 858] ===
        self.gate = PostSDPAGate(dim)
        # =======================================

    def forward(self, x):
        # x is normalized input
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        cos, sin = self.rope(q, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # SDPA
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # === NEW: Apply Gating ===
        # Applied after SDPA, before output projection 
        out = self.gate(x, out)
        # =========================

        return self.wo(out)

    def step(self, x_t, state=None, cache_capacity=None):
        """Attend one normalized item using a fixed-capacity append-only KV cache."""
        if cache_capacity is None or cache_capacity < 1:
            raise ValueError("ModernAttention.step requires a positive cache_capacity")
        batch, dim = x_t.shape
        if state is None:
            state = {
                "k": x_t.new_empty(batch, self.n_heads, cache_capacity, self.head_dim),
                "v": x_t.new_empty(batch, self.n_heads, cache_capacity, self.head_dim),
                "length": 0,
            }
        length = state["length"]
        if length >= state["k"].size(2):
            raise ValueError("ModernAttention KV cache exceeded its stage capacity")

        q = self.wq(x_t).view(batch, 1, self.n_heads, self.head_dim)
        k = self.wk(x_t).view(batch, 1, self.n_kv_heads, self.head_dim)
        v = self.wv(x_t).view(batch, 1, self.n_kv_heads, self.head_dim)
        cos, sin = self.rope(q, 1, position_offset=length)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        state["k"][:, :, length].copy_(k[:, :, 0])
        state["v"][:, :, length].copy_(v[:, :, 0])
        length += 1
        state["length"] = length
        cached_k = state["k"][:, :, :length]
        cached_v = state["v"][:, :, :length]

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, cached_k, cached_v, is_causal=False)
        else:
            scores = torch.matmul(q, cached_k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            out = torch.matmul(scores.softmax(dim=-1), cached_v)
        out = out.transpose(1, 2).contiguous().view(batch, dim)
        return self.wo(self.gate(x_t, out)), state

class ModernTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, act_name="swiglu"):
        super().__init__()
        self.attn = ModernAttention(dim, n_heads, n_kv_heads)
        self.ffn = make_feed_forward(dim, int(dim * 2.68), act_name=act_name, bias=False)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    def step(self, x_t, state=None, cache_capacity=None):
        attn_out, attn_state = self.attn.step(
            self.norm1(x_t), None if state is None else state["attention"], cache_capacity
        )
        x_t = x_t + attn_out
        return x_t + self.ffn(self.norm2(x_t)), {"attention": attn_state}

class ModernTransformer(nn.Module):
    def __init__(self, vocab_size, dim, depth, n_heads, act_name="swiglu"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            ModernTransformerBlock(dim, n_heads, n_heads, act_name=act_name) for _ in range(depth)
        ])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward_hidden(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, idx):
        return self.head(self.forward_hidden(self.embed(idx)))

# ==============================================================================
# 4. Griffin (RG-LRU + Local Attn)
# ==============================================================================
class RGLRU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.input_gate = nn.Linear(dim, dim)
        self.recur_gate = nn.Linear(dim, dim)
        self.out_gate = nn.Linear(dim, dim)

    def forward(self, x, state=None):
        # x: (B,T,D)
        i = torch.sigmoid(self.input_gate(x))
        log_a = F.logsigmoid(self.recur_gate(x))
        u = i * x 
        a_lin = torch.exp(log_a)
        # Use JIT scan for stability
        h = pscan_linear_jit(a_lin, u, state)
        return h * torch.sigmoid(self.out_gate(x)), h[:, -1, :]

class GriffinBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMSNorm(dim)
        self.kernel_size = 4
        self.conv = nn.Conv1d(dim, dim, kernel_size=self.kernel_size, padding=0, groups=dim)
        self.rglru = RGLRU(dim)
        self.mlp = SwiGLU(dim, dim*4)

    def forward(self, x, state=None):
        # state is now a tuple: (rnn_hidden, conv_buffer)
        rnn_state = None
        conv_buf = None
        if state is not None:
            if isinstance(state, tuple):
                rnn_state, conv_buf = state
            else:
                rnn_state = state # legacy fallback

        shortcut = x
        x = self.norm(x)
        B, T, D = x.shape

        if conv_buf is not None:
            x_conv_in = torch.cat([conv_buf, x], dim=1)
        else:
            x_conv_in = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0)).transpose(1, 2)

        x_conv = self.conv(x_conv_in.transpose(1, 2)).transpose(1, 2)
        x_processed = F.mish(x_conv[:, -T:, :])

        if self.kernel_size > 1:
            raw_for_buffer = x if conv_buf is None else x_conv_in
            if raw_for_buffer.size(1) >= self.kernel_size - 1:
                new_conv_buf = raw_for_buffer[:, -(self.kernel_size - 1):, :]
            else:
                pad_len = self.kernel_size - 1 - raw_for_buffer.size(1)
                new_conv_buf = F.pad(raw_for_buffer, (0, 0, pad_len, 0))
        else:
            new_conv_buf = x.new_zeros(B, 0, D)
        
        # RG-LRU
        x_rnn, new_rnn_state = self.rglru(x_processed, rnn_state)
        
        x = x_rnn + shortcut
        x = x + self.mlp(self.norm(x))
        
        return x, (new_rnn_state, new_conv_buf)

class GriffinLM(nn.Module):
    def __init__(self, vocab_size, dim, depth):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([GriffinBlock(dim) for _ in range(depth)])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx, state=None):
        x = self.embed(idx)
        if state is None: state = [None]*len(self.blocks)
        new_states = []
        for blk, s in zip(self.blocks, state):
            x, ns = blk(x, s)
            new_states.append(ns)
        return self.head(self.norm(x)), new_states

# ==============================================================================
# 5. DeltaNet
# ==============================================================================
# ==============================================================================
# 5. DeltaNet - FIXED (Normalized Keys)
# ==============================================================================
class DeltaNetBlock(nn.Module):
    def __init__(self, dim, head_dim=64):
        super().__init__()
        self.dim = dim
        if dim < head_dim:
            self.head_dim = dim
            self.n_heads = 1
        else:
            self.n_heads = max(1, dim // head_dim)
            self.head_dim = dim // self.n_heads
            
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, dim*4)

    def forward(self, x, state=None):
        B, T, C = x.shape
        shortcut = x
        x_norm = self.norm(x)
        
        q = self.q(x_norm).view(B, T, self.n_heads, self.head_dim)
        k = self.k(x_norm).view(B, T, self.n_heads, self.head_dim)
        v = self.v(x_norm).view(B, T, self.n_heads, self.head_dim)
        beta = torch.sigmoid(self.beta(x_norm)).view(B, T, self.n_heads, self.head_dim)
        
        # === KEY NORMALIZATION FIX ===
        # L2 Normalize K to prevent explosion
        k = F.normalize(k, p=2, dim=-1)
        
        if state is None:
            state = torch.zeros(B, self.n_heads, self.head_dim, self.head_dim, device=x.device)
            
        outs = []
        H = state
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        beta = beta.transpose(1, 2)
        
        for t in range(T):
            qt, kt, vt, bt = q[:, :, t, :], k[:, :, t, :], v[:, :, t, :], beta[:, :, t, :]
            
            # Standard Delta Rule
            Rk = torch.matmul(H, kt.unsqueeze(-1)).squeeze(-1)
            diff = vt - Rk
            update = torch.matmul((bt * diff).unsqueeze(-1), kt.unsqueeze(-2))
            H = H + update
            
            ot = torch.matmul(H, qt.unsqueeze(-1)).squeeze(-1)
            outs.append(ot)
            
        y = torch.stack(outs, dim=2).transpose(1, 2).reshape(B, T, C)
        y = self.o(y)
        x = shortcut + y
        x = x + self.mlp(self.norm(x))
        return x, H

class DeltaNetLM(nn.Module):
    def __init__(self, vocab_size, dim, depth):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([DeltaNetBlock(dim) for _ in range(depth)])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx, state=None):
        x = self.embed(idx)
        if state is None: state = [None]*len(self.blocks)
        new_states = []
        for blk, s in zip(self.blocks, state):
            x, ns = blk(x, s)
            new_states.append(ns)
        return self.head(self.norm(x)), new_states


class ContentAddressedStageBlock(nn.Module):
    """Naive, exact delta-rule stage block for short MEGABYTE sequences."""
    def __init__(self, dim, kind):
        super().__init__()
        self.dim, self.kind = dim, kind
        self.norm = RMSNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.beta = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ff = SwiGLU(dim, dim * 4)

    def step(self, x, state=None):
        normalized = self.norm(x)
        q, k, v = (F.normalize(proj(normalized), dim=-1) for proj in (self.q, self.k, self.v))
        v = torch.tanh(v)
        state = x.new_zeros(x.size(0), self.dim, self.dim) if state is None else state
        beta = torch.sigmoid(self.beta(normalized))
        retrieved = torch.einsum("bij,bj->bi", state, k)
        update = beta * (v - retrieved)
        if self.kind in {"gated_deltanet", "rwkv7"}:
            state = torch.sigmoid(self.gate(normalized)).unsqueeze(-1) * state
        state = state + torch.einsum("bi,bj->bij", update, k)
        y = torch.einsum("bij,bj->bi", state, q)
        if self.kind == "rwkv7":
            y = torch.sigmoid(self.gate(normalized)) * y
        x = x + self.out(y)
        return x + self.ff(self.norm(x)), state

    def forward_seq(self, x, state=None):
        outputs = []
        for token in x.unbind(1):
            token, state = self.step(token, state)
            outputs.append(token)
        return torch.stack(outputs, 1), state

# ==============================================================================
# 6. RetNet (Simple Linear Retention)
# ==============================================================================
class RetNetBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.out_norm = nn.LayerNorm(dim)
        self.swiglu = SwiGLU(dim, dim*2)
        self.ln = RMSNorm(dim)
        gammas = 1.0 - 2.0 ** (-5.0 - torch.arange(n_heads).float())
        self.register_buffer("gammas", gammas)

    def forward(self, x, state=None):
        shortcut = x
        x_norm = self.ln(x)
        B, T, C = x.shape
        q = self.wq(x_norm).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x_norm).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x_norm).view(B, T, self.n_heads, self.head_dim)
        
        kv = torch.einsum('bthd,bthe->bthde', k, v).reshape(B, T, -1)
        gamma = self.gammas.view(1, 1, self.n_heads, 1, 1).expand(
            B, T, self.n_heads, self.head_dim, self.head_dim
        ).reshape(B, T, -1)
        
        if state is not None:
            state = state.reshape(B, -1)
        
        # Use JIT scan (handles vanishing gamma correctly without explosion)
        h = pscan_linear_jit(gamma, kv, state)
        
        h_mat = h.view(B, T, self.n_heads, self.head_dim, self.head_dim)
        out = torch.einsum('bthd,bthde->bthe', q, h_mat).flatten(2)
        out = self.out_norm(out)
        x = shortcut + self.wo(out)
        x = x + self.swiglu(self.ln(x))
        return x, h[:, -1, :].view(B, self.n_heads, self.head_dim, self.head_dim)


class RecurrentInterfaceBlock(nn.Module):
    """Autoregressive RIN block: data→latents read, latent compute, latents→data write.

    This mirrors the RIN routing structure from Jabri et al. while retaining
    causal language-model semantics.  A bank of latent interface tokens is the
    recurrent state; every token updates that bank through cross-attention,
    applies the bulk of the computation (latent self-attention + MLP), then
    reads the updated latents back through a second cross-attention operation.
    """
    def __init__(self, dim, num_latents=8, num_heads=4, dropout=0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError("RIN dim must be divisible by its attention heads")
        self.num_latents = num_latents
        # Learned latent interface tokens (not a zero recurrent vector) are a
        # defining part of RIN and let different slots specialize.
        self.initial_latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)
        self.data_norm = nn.LayerNorm(dim)
        self.read_norm = nn.LayerNorm(dim)
        self.latent_norm1 = nn.LayerNorm(dim)
        self.latent_norm2 = nn.LayerNorm(dim)
        self.write_norm = nn.LayerNorm(dim)
        self.read = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.latent_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.write = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.latent_mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim),
        )
        self.data_mlp = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state=None):
        B, T, D = x.shape
        if state is None:
            latents = self.initial_latents.expand(B, -1, -1)
        else:
            latents = state
        # Dropout on the single read weight and on the residual branches is not
        # in the fused kernel; the step loop below handles it.
        if kernels_available(x) and (self.dropout.p == 0.0 or not self.training):
            return self._forward_fused(x, latents)

        outputs = []
        for token in x.unbind(dim=1):
            data = token.unsqueeze(1)
            # Read: latent queries route the current causal data token inward.
            read, _ = self.read(self.read_norm(latents), self.data_norm(data), self.data_norm(data), need_weights=False)
            latents = latents + self.dropout(read)
            # Compute: global interaction happens only among the compact latents.
            q = self.latent_norm1(latents)
            update, _ = self.latent_attn(q, q, q, need_weights=False)
            latents = latents + self.dropout(update)
            latents = latents + self.dropout(self.latent_mlp(self.latent_norm2(latents)))
            # Write: the data token queries the processed latent interface.
            write, _ = self.write(self.data_norm(data), self.write_norm(latents), self.write_norm(latents), need_weights=False)
            data = data + self.dropout(write)
            outputs.append((data + self.dropout(self.data_mlp(data))).squeeze(1))

        return torch.stack(outputs, dim=1), latents

    def _forward_fused(self, x, latents):
        """The step loop above with only the latent update kept sequential."""
        B, T, D = x.shape
        data = self.data_norm(x)
        # Read: each token is the only key, so its attention weight is exactly
        # 1 and every latent receives out_proj(v_proj(token)).
        _, _, w_v = self.read.in_proj_weight.chunk(3)
        _, _, b_v = self.read.in_proj_bias.chunk(3)
        read = self.read.out_proj(F.linear(data, w_v, b_v))
        attn, mlp = self.latent_attn, self.latent_mlp
        seq = rin_latent_scan(
            read, latents, self.latent_norm1.weight, self.latent_norm1.bias,
            attn.in_proj_weight, attn.in_proj_bias, attn.out_proj.weight, attn.out_proj.bias,
            self.latent_norm2.weight, self.latent_norm2.bias,
            mlp[0].weight, mlp[0].bias, mlp[3].weight, mlp[3].bias,
            attn.num_heads, self.latent_norm1.eps,
        ).to(x.dtype)
        # Write: every token queries the latents it produced, all at once.
        keys = self.write_norm(seq).reshape(B * T, self.num_latents, D)
        write, _ = self.write(data.reshape(B * T, 1, D), keys, keys, need_weights=False)
        out = x + write.view(B, T, D)
        return out + self.data_mlp(out), seq[:, -1]


class RecurrentInterfaceLM(nn.Module):
    def __init__(self, vocab_size, dim, depth, dropout=0.0, num_latents=8, num_heads=None):
        super().__init__()
        if num_heads is None or dim % num_heads:
            num_heads = next((h for h in (8, 4, 2, 1) if dim % h == 0), 1)
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            RecurrentInterfaceBlock(dim, num_latents=num_latents, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx, state=None):
        x = self.embed(idx)
        if state is None:
            state = [None] * len(self.blocks)

        new_states = []
        for block, block_state in zip(self.blocks, state):
            x, new_state = block(x, block_state)
            new_states.append(new_state)

        return self.head(self.norm(x)), new_states


# ==============================================================================
# Additional recurrent and MLP zoo models
# ==============================================================================

class MogrifierLSTMCell(nn.Module):
    """Mogrifier-LSTM cell; alternating input/hidden modulation precedes LSTM."""
    def __init__(self, dim, rounds=5):
        super().__init__()
        self.rounds = rounds
        self.q = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range((rounds + 1) // 2)])
        self.r = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(rounds // 2)])
        self.cell = nn.LSTMCell(dim, dim)

    def forward(self, x, state):
        h, c = state
        qi = ri = 0
        for i in range(self.rounds):
            if i % 2 == 0:
                x = 2.0 * torch.sigmoid(self.q[qi](h)) * x; qi += 1
            else:
                h = 2.0 * torch.sigmoid(self.r[ri](x)) * h; ri += 1
        return self.cell(x, (h, c))


class LegendreDelayMemory(nn.Module):
    """Fixed, discrete Legendre delay system with ``(batch, units, order)`` state."""
    def __init__(self, theta: int, d_state: int = 12):
        super().__init__()
        if theta < 1 or d_state < 1:
            raise ValueError("LMU theta and d_state must be positive")
        self.theta, self.d_state = int(theta), int(d_state)
        q = torch.arange(d_state, dtype=torch.float64)
        i, j = torch.meshgrid(q, q, indexing="ij")
        scale = (2.0 * q + 1.0).unsqueeze(1) / theta
        a = torch.where(i < j, -torch.ones_like(i), (-1.0) ** (i - j + 1)) * scale
        b = ((-1.0) ** q * (2.0 * q + 1.0) / theta).unsqueeze(1)
        augmented = torch.zeros(d_state + 1, d_state + 1, dtype=torch.float64)
        augmented[:d_state, :d_state], augmented[:d_state, d_state:] = a, b
        discrete = torch.linalg.matrix_exp(augmented)
        self.register_buffer("A", discrete[:d_state, :d_state])
        self.register_buffer("B", discrete[:d_state, d_state])

    def step(self, u, state=None):
        if u.ndim == 1:
            u = u.unsqueeze(-1)
        if u.ndim != 2:
            raise ValueError("LMU drive must have shape (batch, units)")
        if state is None:
            state = u.new_zeros(u.size(0), u.size(1), self.d_state)
        return F.linear(state, self.A.to(dtype=state.dtype)) + u.unsqueeze(-1) * self.B.to(dtype=u.dtype)

    def forward(self, u, state=None):
        outputs = []
        for drive in u.unbind(1):
            state = self.step(drive, state)
            outputs.append(state)
        return torch.stack(outputs, 1), state


class StatefulCellLM(nn.Module):
    """Compile-friendly tensor-state wrapper for small BPTT recurrent cells."""
    def __init__(self, vocab_size, dim, depth, kind, dropout=0.0, lmu_theta=None,
                 lmu_units=None, lmu_order=None):
        super().__init__()
        self.kind, self.dim = kind, dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.cells = nn.ModuleList([
            MogrifierLSTMCell(dim) if kind == "mogrifier" else nn.GRUCell(dim, dim)
            for _ in range(depth)
        ])
        self.in_proj = nn.ModuleList([nn.Linear(dim, 3 * dim) for _ in range(depth)])
        self.rec_proj = nn.ModuleList([nn.Linear(dim, 3 * dim, bias=False) for _ in range(depth)])
        self.out = nn.Linear(dim, vocab_size)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        if kind == "lmu":
            # The canonical LMU memory is a discretized Legendre delay system,
            # not a bank of independent exponential decays.  One input drive
            # is encoded into `dim` Legendre coefficients over `lmu_theta`
            # token steps; the nonlinear hidden state reads those coefficients.
            theta = float(dim if lmu_theta is None else lmu_theta)
            if theta <= 0:
                raise ValueError("LMU memory horizon must be positive")
            self.lmu_units = int(dim if lmu_units is None else lmu_units)
            self.lmu_order = int(min(16, dim) if lmu_order is None else lmu_order)
            if self.lmu_units < 1 or self.lmu_order < 1:
                raise ValueError("LMU memory units and order must be positive")
            self.lmu_memory = LegendreDelayMemory(int(theta), d_state=self.lmu_order)
            memory_width = self.lmu_units * self.lmu_order
            self.lmu_drive = nn.ModuleList([
                nn.Linear(2 * dim + memory_width, self.lmu_units) for _ in range(depth)
            ])
            self.lmu_memory_read = nn.ModuleList([
                nn.Linear(memory_width, dim, bias=False) for _ in range(depth)
            ])
        if kind == "cfc":
            self.cfc_gate = nn.ModuleList([nn.Linear(2 * dim, dim) for _ in range(depth)])
            self.cfc_a = nn.ModuleList([nn.Linear(2 * dim, dim) for _ in range(depth)])
            self.cfc_b = nn.ModuleList([nn.Linear(2 * dim, dim) for _ in range(depth)])

    def _initial_state(self, batch, device, dtype):
        if self.kind == "mogrifier":
            z = torch.zeros(len(self.cells), batch, self.dim, device=device, dtype=dtype)
            return z, z.clone()
        if self.kind == "lmu":
            h = torch.zeros(len(self.cells), batch, self.dim, device=device, dtype=dtype)
            memory = torch.zeros(
                len(self.cells), batch, self.lmu_units, self.lmu_order,
                device=device, dtype=dtype,
            )
            return h, memory
        return torch.zeros(len(self.cells), batch, self.dim, device=device, dtype=dtype)

    def forward_hidden(self, x, state=None):
        """Run the recurrent cell stack on hidden vectors."""
        B, T, _ = x.shape
        if state is None:
            state = self._initial_state(B, x.device, x.dtype)
        if self.kind == "mogrifier":
            hs, cs = state
        elif self.kind == "lmu":
            hs, memories = state
            if kernels_available(x):
                return self._forward_lmu_fused(x, hs, memories)
        else:
            hs = state
            if self.kind in ("nru", "cfc") and kernels_available(x):
                return self._forward_cell_fused(x, hs)
        outputs = []
        for t in range(T):
            y = x[:, t]
            next_h, next_c, next_memories = [], [], []
            for layer, cell in enumerate(self.cells):
                h = hs[layer]
                if self.kind == "mogrifier":
                    h, c = cell(y, (h, cs[layer])); next_c.append(c)
                elif self.kind == "nru":
                    # Additive, non-saturating memory with normalized write/read directions.
                    a, w, r = (self.in_proj[layer](y) + self.rec_proj[layer](h)).chunk(3, -1)
                    write = F.normalize(w, dim=-1); read = F.normalize(r, dim=-1)
                    h = h + torch.tanh(a) * write - (h * read).sum(-1, keepdim=True) * read
                elif self.kind == "lmu":
                    memory = memories[layer]
                    drive = self.lmu_drive[layer](torch.cat((y, h, memory.flatten(1)), dim=-1))
                    memory = self.lmu_memory.step(drive, memory)
                    h = torch.tanh(
                        self.in_proj[layer](y)[..., :self.dim]
                        + self.rec_proj[layer](h)[..., :self.dim]
                        + self.lmu_memory_read[layer](memory.flatten(1))
                    )
                    next_memories.append(memory)
                elif self.kind == "cfc":
                    joined = torch.cat((y, h), -1)
                    gate = torch.sigmoid(self.cfc_gate[layer](joined))
                    h = gate * torch.tanh(self.cfc_a[layer](joined)) + (1.0 - gate) * torch.tanh(self.cfc_b[layer](joined)) * h
                else:
                    h = cell(y, h)
                y = self.dropout(h); next_h.append(h)
            hs = torch.stack(next_h)
            if self.kind == "mogrifier": cs = torch.stack(next_c)
            if self.kind == "lmu": memories = torch.stack(next_memories)
            outputs.append(y)
        state = (hs, cs) if self.kind == "mogrifier" else (hs, memories) if self.kind == "lmu" else hs
        return self.norm(torch.stack(outputs, 1)), state

    def _forward_cell_fused(self, x, hs):
        """NRU / CfC stack on the fused scans, layer-major like the LMU path."""
        D = self.dim
        y, next_h = x, []
        for layer in range(len(self.cells)):
            if self.kind == "nru":
                px, w = self.in_proj[layer](y), self.rec_proj[layer].weight
                h_seq, h_last = nru_scan(px, w, hs[layer])
            else:
                gate, a, b = self.cfc_gate[layer], self.cfc_a[layer], self.cfc_b[layer]
                w_all = torch.cat((gate.weight, a.weight, b.weight), 0)          # (3D, 2D) over [y, h]
                px = F.linear(y, w_all[:, :D], torch.cat((gate.bias, a.bias, b.bias)))
                h_seq, h_last = cfc_scan(px, w_all[:, D:], hs[layer])
            y = self.dropout(h_seq.to(x.dtype))
            next_h.append(h_last.to(hs.dtype))
        return self.norm(y), torch.stack(next_h)

    def _forward_lmu_fused(self, x, hs, memories):
        """Layer-major LMU stack on the fused scan: layer l at step t needs
        only layer l-1 at t and itself at t-1, so this equals the step loop."""
        D = self.dim
        y, next_h, next_m = x, [], []
        for layer in range(len(self.cells)):
            w_in, b_in = self.in_proj[layer].weight[:D], self.in_proj[layer].bias[:D]
            w_drive = self.lmu_drive[layer].weight
            h_seq, h_last, m_last = lmu_scan(
                F.linear(y, w_in, b_in), F.linear(y, w_drive[:, :D], self.lmu_drive[layer].bias),
                w_drive[:, D:2 * D], w_drive[:, 2 * D:], self.rec_proj[layer].weight[:D],
                self.lmu_memory_read[layer].weight, self.lmu_memory.A, self.lmu_memory.B,
                hs[layer], memories[layer],
            )
            y = self.dropout(h_seq.to(x.dtype))
            next_h.append(h_last.to(hs.dtype)); next_m.append(m_last.to(memories.dtype))
        return self.norm(y), (torch.stack(next_h), torch.stack(next_m))

    def forward(self, idx, state=None):
        hidden, state = self.forward_hidden(self.embed(idx), state)
        return self.out(hidden), state


class CausalMLPFamilyLM(nn.Module):
    """Fixed-context, strictly causal MLP family: pNLP, Dyna, Wave and CCS."""
    def __init__(self, vocab_size, dim, depth, seq_len, variant, dropout=0.0, act_name="gelu"):
        super().__init__()
        self.seq_len, self.variant = seq_len, variant
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)
        self.token_weights = nn.ParameterList([nn.Parameter(torch.randn(seq_len, seq_len) * 0.02) for _ in range(depth)])
        self.channel = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), make_feed_forward(dim, 4 * dim, act_name=act_name, dropout=dropout))
            for _ in range(depth)
        ])
        self.gates = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        x = self.embed(idx) + self.pos[:, :T]
        tril = torch.ones(T, T, device=x.device, dtype=x.dtype).tril()
        for weight, channel, gate in zip(self.token_weights, self.channel, self.gates):
            w = weight[:T, :T] * tril
            if self.variant == "ccs":
                # Shared cyclic offsets, restricted to the causal half-plane.
                w = torch.stack([torch.roll(w[i], i) for i in range(T)]) * tril
            mixed = torch.einsum("ij,bjd->bid", w, x)
            if self.variant == "dyna":
                # Content-dependent MLP gate; remains causal because `mixed` is.
                mixed = mixed * torch.sigmoid(gate(mixed))
            elif self.variant == "wave":
                mixed = mixed * torch.cos(gate(x)) + x * torch.sin(gate(x))
            x = x + mixed
            x = x + channel(x)
        return self.head(self.norm(x))


class ModernRecurrentLM(nn.Module):
    """Small, explicit-state implementations for Mamba-2, Gated DeltaNet,
    HGRN2 and RWKV-7-style dynamic state evolution."""
    def __init__(self, vocab_size, dim, depth, kind):
        super().__init__()
        self.kind, self.dim = kind, dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.in_proj = nn.ModuleList([nn.Linear(dim, 5 * dim) for _ in range(depth)])
        self.out_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx, state=None):
        x = self.embed(idx); B, T, D = x.shape
        matrix_state = self.kind in {"gated_delta", "hgrn2", "rwkv7"}
        if state is None:
            state = [x.new_zeros(B, D, D) if matrix_state else x.new_zeros(B, D) for _ in self.in_proj]
        next_states = []
        for layer, (proj, out) in enumerate(zip(self.in_proj, self.out_proj)):
            st = state[layer]; ys = []
            for token in x.unbind(1):
                q, k, v, gate, rate = proj(token).chunk(5, -1)
                if self.kind == "mamba2":
                    decay = torch.sigmoid(gate)
                    st = decay * st + (1.0 - decay) * torch.tanh(k) * torch.tanh(v)
                    y = torch.sigmoid(q) * st
                else:
                    q, k, v = F.normalize(q, dim=-1), F.normalize(k, dim=-1), torch.tanh(v)
                    beta = torch.sigmoid(rate)
                    retrieved = torch.einsum("bij,bj->bi", st, k)
                    if self.kind == "hgrn2":
                        st = torch.sigmoid(gate).unsqueeze(-1) * st + torch.einsum("bi,bj->bij", v, k)
                    else:
                        update = v - retrieved
                        st = torch.sigmoid(gate).unsqueeze(-1) * st + beta.unsqueeze(-1) * torch.einsum("bi,bj->bij", update, k)
                    y = torch.einsum("bij,bj->bi", st, q)
                ys.append(token + out(y))
            x = torch.stack(ys, 1); next_states.append(st)
        return self.head(self.norm(x)), next_states


class TransformerXLLM(nn.Module):
    """Segment-recurrent decoder with fixed-size tensor memory per layer."""
    def __init__(self, vocab_size, dim, depth, heads=4, mem_len=128):
        super().__init__()
        heads = next(h for h in (heads, 8, 4, 2, 1) if dim % h == 0)
        self.mem_len = mem_len; self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(depth)])
        self.ff = nn.ModuleList([nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim)) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, vocab_size)

    def forward(self, idx, state=None):
        x = self.embed(idx); B, T, D = x.shape
        if state is None: state = [x.new_zeros(B, 0, D) for _ in self.blocks]
        next_state = []
        for attn, ff, mem in zip(self.blocks, self.ff, state):
            kv = torch.cat((mem, x), 1); M = mem.size(1)
            mask = torch.ones(T, M + T, dtype=torch.bool, device=x.device).triu(M + 1)
            y, _ = attn(x, kv, kv, attn_mask=mask, need_weights=False)
            x = x + y; x = x + ff(x)
            # Detachment belongs at the TBPTT boundary in the training loop;
            # keeping forward tensor-only makes this path compile-friendly.
            next_state.append(kv[:, -self.mem_len:])
        return self.head(self.norm(x)), next_state


class TitansLM(nn.Module):
    """Causal short-term attention plus learned persistent neural memory."""
    def __init__(self, vocab_size, dim, depth, heads=4, memory_slots=32):
        super().__init__()
        heads = next(h for h in (heads, 8, 4, 2, 1) if dim % h == 0)
        self.memory_slots = memory_slots; self.embed = nn.Embedding(vocab_size, dim)
        self.attn = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(depth)])
        self.mem_read = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(depth)])
        self.update = nn.ModuleList([nn.Linear(2 * dim, dim) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, vocab_size)

    def forward(self, idx, state=None):
        x = self.embed(idx); B, T, D = x.shape
        if state is None: state = [x.new_zeros(B, self.memory_slots, D) for _ in self.attn]
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1); next_state = []
        for attn, read, update, memory in zip(self.attn, self.mem_read, self.update, state):
            short, _ = attn(x, x, x, attn_mask=causal, need_weights=False)
            long, _ = read(x, memory, memory, need_weights=False)
            x = x + short + long
            surprise = (x - long).mean(1, keepdim=True).expand(-1, self.memory_slots, -1)
            memory = memory + torch.tanh(update(torch.cat((memory, surprise), -1)))
            next_state.append(memory)
        return self.head(self.norm(x)), next_state

class RetNetLM(nn.Module):
    def __init__(self, vocab_size, dim, depth, n_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([RetNetBlock(dim, n_heads) for _ in range(depth)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx, state=None):
        x = self.embed(idx)
        if state is None: state = [None]*len(self.blocks)
        new_states = []
        for blk, s in zip(self.blocks, state):
            x, ns = blk(x, s)
            new_states.append(ns)
        return self.head(x), new_states

# ==============================================================================
# 7. HGRN - FIXED STABILITY
# ==============================================================================
class HGRNBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMSNorm(dim)
        self.i_gate = nn.Linear(dim, dim)
        self.f_gate = nn.Linear(dim, dim)
        self.g_gate = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, state=None):
        shortcut = x
        x = self.norm(x)
        i = torch.sigmoid(self.i_gate(x))
        g = torch.tanh(self.g_gate(x))
        f = torch.sigmoid(self.f_gate(x))
        u = i * g
        # Use JIT scan
        h = pscan_linear_jit(f, u, state)
        out = self.out(h)
        return shortcut + out, h[:, -1, :]

class HGRN_LM(nn.Module):
    def __init__(self, vocab_size, dim, depth):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([HGRNBlock(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx, state=None):
        x = self.embed(idx)
        if state is None: state = [None]*len(self.blocks)
        new_states = []
        for blk, s in zip(self.blocks, state):
            x, ns = blk(x, s)
            new_states.append(ns)
        return self.head(x), new_states

# ==============================================================================
# 8. Liquid Neural Network (LTC Cell Wrapper)
# ==============================================================================
class LTCLayer(nn.Module):
    """Liquid time-constant layer (Hasani et al., 2021), following the ncps
    ``LTCCell`` (mlech26l/ncps) run over a sequence: fully connected wiring
    (synapse polarities drawn from {-1, +1, +1} with seed 1111, recurrent
    synapses first), implicit softplus parameter constraints, affine input and
    output mappings, ``ode_unfolds`` fused semi-implicit solver steps per
    token and unit timespans.  State: the membrane potentials v (B, units).
    """
    _RANGES = {"gleak": (0.001, 1.0), "vleak": (-0.2, 0.2), "cm": (0.4, 0.6),
               "w": (0.001, 1.0), "sigma": (3.0, 8.0), "mu": (0.3, 0.8)}

    def __init__(self, input_size: int, units: int, ode_unfolds: int = 6,
                 epsilon: float = 1e-8, erev_init_seed: int = 1111):
        super().__init__()
        self.input_size, self.units = input_size, units
        self.ode_unfolds, self.epsilon = int(ode_unfolds), float(epsilon)

        def init(shape, name):
            low, high = self._RANGES[name]
            return nn.Parameter(torch.rand(*shape) * (high - low) + low)

        rng = np.random.default_rng(erev_init_seed)
        erev = rng.choice([-1, 1, 1], size=(units, units))
        sensory_erev = rng.choice([-1, 1, 1], size=(input_size, units))
        self.gleak = init((units,), "gleak")
        self.vleak = init((units,), "vleak")
        self.cm = init((units,), "cm")
        self.sigma = init((units, units), "sigma")
        self.mu = init((units, units), "mu")
        self.w = init((units, units), "w")
        self.erev = nn.Parameter(torch.as_tensor(erev, dtype=torch.float32))
        self.sensory_sigma = init((input_size, units), "sigma")
        self.sensory_mu = init((input_size, units), "mu")
        self.sensory_w = init((input_size, units), "w")
        self.sensory_erev = nn.Parameter(torch.as_tensor(sensory_erev, dtype=torch.float32))
        self.input_w = nn.Parameter(torch.ones(input_size))
        self.input_b = nn.Parameter(torch.zeros(input_size))
        self.output_w = nn.Parameter(torch.ones(units))
        self.output_b = nn.Parameter(torch.zeros(units))

    def _reference(self, xm, v, wp, swp, cm_t, gleak):
        """Direct transcription of ncps LTCCell._ode_solver, step by step."""
        outs = []
        for t in range(xm.size(1)):
            s_act = swp * torch.sigmoid(self.sensory_sigma * (xm[:, t, :, None] - self.sensory_mu))
            num_s = (s_act * self.sensory_erev).sum(1)
            den_s = s_act.sum(1)
            for _ in range(self.ode_unfolds):
                act = wp * torch.sigmoid(self.sigma * (v[:, :, None] - self.mu))
                numerator = cm_t * v + gleak * self.vleak + (act * self.erev).sum(1) + num_s
                denominator = cm_t + gleak + act.sum(1) + den_s
                v = numerator / (denominator + self.epsilon)
            outs.append(v)
        return torch.stack(outs, 1)

    def forward(self, x: torch.Tensor, v0: Optional[torch.Tensor] = None):
        B = x.size(0)
        v0 = x.new_zeros(B, self.units, dtype=torch.float32) if v0 is None else v0.float()
        xm = x.float() * self.input_w + self.input_b
        wp, swp = F.softplus(self.w), F.softplus(self.sensory_w)
        cm_t = F.softplus(self.cm) * self.ode_unfolds          # cm / (timespan / unfolds)
        gleak = F.softplus(self.gleak)
        if kernels_available(x):
            nums, dens = ltc_sensory(xm, swp, self.sensory_mu, self.sensory_sigma, self.sensory_erev)
            v_seq = ltc_scan(nums, dens, v0, wp, self.mu, self.sigma, self.erev, cm_t, gleak,
                             self.vleak, self.ode_unfolds, self.epsilon)
        else:
            v_seq = self._reference(xm, v0, wp, swp, cm_t, gleak)
        return v_seq * self.output_w + self.output_b, v_seq[:, -1]


class LiquidRNN(nn.Module):
    """Stack of LTC layers for the CustomRNNWrapper (state: (L, B, H))."""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super().__init__()
        if not batch_first:
            raise ValueError("LiquidRNN expects batch_first=True")
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.layers = nn.ModuleList([
            LTCLayer(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)
        ])

    def forward(self, x, state=None):
        final = []
        for i, layer in enumerate(self.layers):
            x, v = layer(x, None if state is None else state[i])
            final.append(v)
        return x, torch.stack(final, dim=0)

# ==============================================================================
# 9. MEGABYTE
# ==============================================================================
class MegaByteRMSNorm(nn.Module):
    """RMSNorm used by lucidrains' MEGABYTE implementation."""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


def _megabyte_token_shift(x):
    """RWKV-style half-channel one-token shift used in upstream MEGABYTE."""
    x, shifted = x.chunk(2, dim=-1)
    shifted = F.pad(shifted, (0, 0, 1, -1))
    return torch.cat((x, shifted), dim=-1)


class MegaByteAttention(nn.Module):
    """Causal pre-norm attention matching the non-flash upstream path."""
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.head_dim = dim_head
        inner_dim = heads * dim_head
        self.norm = MegaByteRMSNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        batch, seq, _ = x.shape
        x = self.norm(x)
        q = self.to_q(x).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        k = k.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Keep the model usable with the older PyTorch versions supported
            # by this project, which predate public SDPA.
            scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            causal_mask = torch.ones(seq, seq, device=x.device, dtype=torch.bool).tril_()
            scores = scores.masked_fill(~causal_mask, float("-inf"))
            out = torch.matmul(scores.softmax(dim=-1), v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.to_out(out)

    def step(self, x_t, state=None, cache_capacity=None):
        """Attend one item using an in-place, fixed-capacity KV cache."""
        if cache_capacity is None or cache_capacity < 1:
            raise ValueError("MegaByteAttention.step requires a positive cache_capacity")
        batch = x_t.size(0)
        if state is None:
            state = {
                "k": x_t.new_empty(batch, self.heads, cache_capacity, self.head_dim),
                "v": x_t.new_empty(batch, self.heads, cache_capacity, self.head_dim),
                "length": 0,
            }
        length = state["length"]
        if length >= state["k"].size(2):
            raise ValueError("MEGABYTE attention KV cache exceeded its stage capacity")

        x_t = self.norm(x_t)
        q = self.to_q(x_t).view(batch, self.heads, 1, self.head_dim)
        k, v = self.to_kv(x_t).chunk(2, dim=-1)
        k = k.view(batch, self.heads, 1, self.head_dim)
        v = v.view(batch, self.heads, 1, self.head_dim)
        state["k"][:, :, length].copy_(k[:, :, 0])
        state["v"][:, :, length].copy_(v[:, :, 0])
        length += 1
        state["length"] = length
        cached_k = state["k"][:, :, :length]
        cached_v = state["v"][:, :, :length]
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, cached_k, cached_v, is_causal=False)
        else:
            scores = torch.matmul(q, cached_k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            out = torch.matmul(scores.softmax(dim=-1), cached_v)
        return self.to_out(out.transpose(1, 2).contiguous().view(batch, -1)), state



class MegaByteFeedForward(nn.Module):
    """The upstream MEGABYTE MLP: RMSNorm → 4x GELU → linear."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            MegaByteRMSNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return self.net(x)


class MegaByteTransformer(nn.Module):
    """Token-shifted causal transformer used at each MEGABYTE scale."""
    def __init__(self, dim, depth, heads=8, dim_head=64):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList((MegaByteAttention(dim, heads, dim_head), MegaByteFeedForward(dim)))
            for _ in range(depth)
        ])
        self.norm = MegaByteRMSNorm(dim)

    def forward(self, x):
        for attention, feed_forward in self.layers:
            x = x + attention(_megabyte_token_shift(x))
            x = x + feed_forward(_megabyte_token_shift(x))
        return self.norm(x)

    def step(self, x_t, state=None, cache_capacity=None):
        """Match the token-shifted sequence path while retaining per-layer K/V."""
        if state is None:
            state = [None] * len(self.layers)
        next_state = []
        first_width = (x_t.size(-1) + 1) // 2
        for (attention, feed_forward), layer_state in zip(self.layers, state):
            if layer_state is None:
                previous_attention = x_t.new_zeros(x_t.size(0), x_t.size(1) - first_width)
                previous_feed_forward = x_t.new_zeros(x_t.size(0), x_t.size(1) - first_width)
                attention_state = None
            else:
                previous_attention = layer_state["attention_previous"]
                previous_feed_forward = layer_state["feed_forward_previous"]
                attention_state = layer_state["attention"]

            layer_input = x_t
            attention_input = torch.cat((layer_input[:, :first_width], previous_attention), dim=-1)
            attended, attention_state = attention.step(attention_input, attention_state, cache_capacity)
            x_t = layer_input + attended
            feed_forward_input = torch.cat((x_t[:, :first_width], previous_feed_forward), dim=-1)
            pre_feed_forward = x_t
            x_t = x_t + feed_forward(feed_forward_input)
            next_state.append({
                "attention": attention_state,
                "attention_previous": layer_input[:, first_width:],
                "feed_forward_previous": pre_feed_forward[:, first_width:],
            })
        return self.norm(x_t), next_state



# Keep the model-ID adapter map here so ``MegaByteLM`` can accept IDs directly
# without importing the CLI module (which would create a circular import).
# Keyed by linegen's model IDs (family blocks of 100; see linegen.MODEL_REGISTRY).
MEGABYTE_MODEL_ID_MIXERS = {
    1: "mlp", 200: "gmlp", 201: "amlp", 202: "mlpmixer", 207: "hypermixer", 208: "toeplitz", 301:
    "gpt2", 304: "modern", 500: "rnn", 501: "lstm", 502: "atanulstm", 503: "gru", 504: "rnn_relu",
    505: "qrnn", 506: "irnn", 508: "sru", 509: "indrnn", 510: "janet", 511: "exprnn", 512: "nru",
    513: "indygru", 514: "indylstm", 515: "mogrifier_lstm", 516: "mogrifier_gru", 517: "srupp", 518:
    "rru", 519: "light_ru", 600: "lmu", 601: "liquid", 602: "unicornn", 603: "cfc", 800: "xlstm_s",
    801: "xlstm_m", 802: "xlstm", 900: "mingru", 901: "minlstm", 903: "minindrnn", 905:
    "minindygru", 906: "minindylstm", 1000: "s4", 1001: "dss", 1002: "s4d", 1003: "s5", 1005:
    "lru_ssm", 1006: "mamba", 1007: "mamba_ssm", 1008: "mamba2", 1009: "mamba3", 1101: "deltanet",
    1102: "rwkv", 1103: "retnet", 1106: "hgrn2", 1107: "gated_deltanet", 1108: "rwkv7"
}
# Custom recurrent cells run through the same CustomRNNWrapper stacks as
# normal mode (MEGABYTE mixer name -> CustomRNNWrapper cell name).
MEGABYTE_CELL_MIXERS = {
    "indrnn": "indrnn", "indygru": "indygru", "janet": "janet",
    "atanulstm": "atanulstm", "liquid": "liquid",
    "indylstm": "indylstm", "irnn": "irnn", "unicornn": "unicornn",
    "light_ru": "lru", "rru": "rru", "exprnn": "exprnn",
    "mogrifier_lstm": "mogrifier_lstm", "mogrifier_gru": "mogrifier_gru",
}
# Stacks from complete LMs, used through their stateful ``forward_hidden``.
MEGABYTE_HIDDEN_LM_MIXERS = frozenset({
    "mogrifier", "nru", "lmu", "cfc", "qrnn", "sru",
    "srupp", "mamba3", "xlstm", "xlstm_m", "xlstm_s",
    "s4", "s4d", "s5", "dss", "lru_ssm",
})
MEGABYTE_LINEAR_STACK_MIXERS = frozenset({"deltanet", "gated_deltanet", "mamba2", "hgrn2", "retnet"})
INCREMENTAL_MEGABYTE_MIXERS = frozenset({
    "gru", "rnn", "rnn_relu", "lstm", "mingru", "minlstm", "minindrnn",
    "minindygru", "minindylstm", "mamba", "rwkv", "indrnn", "indygru",
    "janet", "atanulstm", "liquid", "mogrifier", "nru", "lmu", "cfc",
    "qrnn", "sru",
    "mamba_ssm",
    "deltanet", "gated_deltanet", "rwkv7",
    "modern", "transformer", "gpt2", "gmlp", "amlp", "mlpmixer",
}) | frozenset(MEGABYTE_CELL_MIXERS) | MEGABYTE_HIDDEN_LM_MIXERS | MEGABYTE_LINEAR_STACK_MIXERS


def resolve_megabyte_stage_mixer(mixer):
    """Resolve a registered stage model ID or retain its explicit mixer name."""
    if isinstance(mixer, int):
        try:
            return MEGABYTE_MODEL_ID_MIXERS[mixer]
        except KeyError as exc:
            raise ValueError(f"model ID {mixer} has no MEGABYTE stage adapter") from exc
    if not isinstance(mixer, str):
        raise TypeError("MEGABYTE stage mixers must be model IDs or strings")
    return mixer


class WindowMLPBlock(nn.Module):
    """A non-causal MLP-Mixer block over one complete hierarchy window.

    It is intentionally used only at the fine bottom-up patch boundary.  An
    MLP encoder, when selected, runs only after a complete patch is known;
    the decoder receives only parent-provided contexts that existed before
    that patch began.  Full token mixing is therefore causal at patch
    granularity while every patch output can be emitted in parallel.
    """
    def __init__(self, dim: int, seq_len: int):
        super().__init__()
        self.token_norm = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(seq_len, 4 * seq_len),
            nn.GELU(),
            nn.Linear(4 * seq_len, seq_len),
        )
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mlp(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channel_mlp(self.channel_norm(x))


_MEGABYTE_SCAN_MIXERS = frozenset({
    "mingru", "minlstm", "minindrnn", "minindygru", "minindylstm",
    "mamba", "mamba_ssm", "rwkv", "rwkv7",
}) | MEGABYTE_LINEAR_STACK_MIXERS


def _megabyte_hidden_lm(kind, dim, depth, heads, seq_len, lmu_theta=None):
    """Build a complete LM's stack for a stage; the stage owns embeddings
    and heads, so the LM's own vocabulary layers are dropped."""
    if kind in {"mogrifier", "nru", "lmu", "cfc"}:
        return StatefulCellLM(1, dim, depth, kind, lmu_theta=lmu_theta or seq_len)
    if kind == "qrnn":
        return QRNNLM(1, dim, depth)
    if kind == "sru":
        return SRULM(1, dim, depth)
    if kind == "srupp":
        lm = SRUppLM(1, dim, depth, max_cache=seq_len)
    elif kind == "mamba3":
        lm = Mamba3LM(1, dim, depth)
    elif kind in {"xlstm", "xlstm_m", "xlstm_s"}:
        lm = XLSTMFullLM(1, dim, depth, num_heads=heads,
                         kind={"xlstm": "mix", "xlstm_m": "m", "xlstm_s": "s"}[kind])
    else:
        lm = StructuredSSMLM(1, dim, depth, "lru" if kind == "lru_ssm" else kind)
    lm.embed = lm.head = None
    return lm


class MegaByteStageMixer(nn.Module):
    """Causal sequence processor shared by one MEGABYTE hierarchy stage."""
    def __init__(self, kind: str, dim: int, depth: int, heads: int, seq_len: int, lmu_theta=None,
                 rnn_norm_type=0, rnn_res_every=0, rnn_res_type=0, rnn_dropout=0.0):
        super().__init__()
        self.kind = resolve_megabyte_stage_mixer(kind)
        self.seq_len = seq_len
        kind = self.kind
        if kind == "transformer":
            self.layers = MegaByteTransformer(dim, depth, heads=heads)
        elif kind in {"gru", "rnn", "rnn_relu", "lstm"}:
            self.layers = BuiltinRNNStage(
                kind, dim, depth, norm_type=rnn_norm_type, res_every=rnn_res_every,
                res_type=rnn_res_type, dropout=rnn_dropout,
            )
        elif kind in _MEGABYTE_SCAN_MIXERS:
            block_cls = {
                "mingru": ScanBlock_minGRU,
                "minlstm": ScanBlock_minLSTM,
                "minindrnn": ScanBlock_MinIndRNN,
                "minindygru": ScanBlock_MinIndyGRU,
                "minindylstm": ScanBlock_MinIndyLSTM,
                "mamba": ScanBlock_Mamba,
                "mamba_ssm": ScanBlock_MambaSSM,
                "rwkv": RWKVBlock,
                "deltanet": lambda d: ContentAddressedStageBlock(d, "deltanet"),
                "gated_deltanet": lambda d: ContentAddressedStageBlock(d, "gated_deltanet"),
            }.get(kind)
            if kind == "rwkv7":
                # The value residual links every layer to the first, so the
                # whole RWKV-7 stack is one stage layer.
                self.layers = nn.ModuleList([RWKV7Stack(dim, depth)])
            elif kind in MEGABYTE_LINEAR_STACK_MIXERS:
                self.layers = nn.ModuleList([LinearRecurrentStack(dim, depth, kind)])
            elif kind == "rwkv":
                self.layers = nn.ModuleList([RWKVBlock(dim, i, depth) for i in range(depth)])
            else:
                self.layers = nn.ModuleList([
                    block_cls(dim, act_type=17) if kind == "minindrnn" else block_cls(dim)
                    for _ in range(depth)
                ])
        elif kind in MEGABYTE_CELL_MIXERS:
            # Reuse the exact hidden-vector cell stacks used by normal mode.
            self.layers = CustomRNNWrapper(MEGABYTE_CELL_MIXERS[kind], 1, dim, depth).rnn
        elif kind in MEGABYTE_HIDDEN_LM_MIXERS:
            self.layers = _megabyte_hidden_lm(kind, dim, depth, heads, seq_len, lmu_theta)
        elif kind == "mlp":
            self.layers = nn.ModuleList([WindowMLPBlock(dim, seq_len) for _ in range(depth)])
            self.output_norm = nn.LayerNorm(dim)
        elif kind == "gmlp":
            self.layers = nn.ModuleList([gMLPBlock(dim, dim * 4, seq_len) for _ in range(depth)])
            self.output_norm = nn.LayerNorm(dim)
        elif kind == "amlp":
            self.layers = nn.ModuleList([aMLPBlock(dim, dim * 4, seq_len) for _ in range(depth)])
            self.output_norm = nn.LayerNorm(dim)
        elif kind == "modern":
            self.layers = nn.ModuleList([ModernTransformerBlock(dim, heads, heads) for _ in range(depth)])
            self.output_norm = RMSNorm(dim)
        elif kind == "gpt2":
            self.layers = GPT2Core(GPT2Config(1, dim, depth, heads, seq_len, tie_weights=False))
        elif kind == "mlpmixer":
            self.layers = nn.ModuleList([CausalMixingBlock(dim, seq_len) for _ in range(depth)])
        elif kind == "hypermixer":
            hyper_hidden = math.ceil(max(64, dim) / heads) * heads
            self.layers = nn.ModuleList([
                HyperMixerBlock(dim, hyper_hidden, 4 * dim, n_heads=heads, causal=True)
                for _ in range(depth)
            ])
        elif kind == "toeplitz":
            self.layers = nn.ModuleList([_ToeplitzMixerBlock(dim, seq_len) for _ in range(depth)])
        else:
            raise ValueError(f"unknown MEGABYTE stage mixer: {kind}")

    def forward(self, x: torch.Tensor, state=None) -> torch.Tensor:
        if self.kind == "mlp" and x.size(1) != self.seq_len:
            raise ValueError(
                f"MEGABYTE window MLP expects exactly {self.seq_len} items, got {x.size(1)}"
            )
        if self.kind == "transformer":
            out = self.layers(x)
            return out[0] if isinstance(out, tuple) else out
        if self.kind in {"gru", "rnn", "rnn_relu", "lstm"}:
            out = self.layers(x, state)
            return out[0] if isinstance(out, tuple) else out
        if self.kind in MEGABYTE_CELL_MIXERS:
            return self.layers(x)[0]
        if self.kind in MEGABYTE_HIDDEN_LM_MIXERS:
            return self.layers.forward_hidden(x)[0]
        if self.kind == "gpt2":
            return self.layers.forward_hidden(x)[0]
        for layer in self.layers:
            if self.kind in _MEGABYTE_SCAN_MIXERS:
                x, _ = layer.forward_seq(x)
            else:
                x = layer(x)
        return self.output_norm(x) if hasattr(self, "output_norm") else x

    def initial_state_from_parent(self, parent_context):
        """Return a parent-conditioned native RNN state, when this core has one."""
        if self.kind in {"gru", "rnn", "rnn_relu", "lstm"}:
            return self.layers.initial_state_from_parent(parent_context)
        return None

    def step(self, x_t: torch.Tensor, state=None):
        """Process one item and return its output plus the carried state.

        Recurrent and scan processors carry their native state.  Finite-context
        processors instead retain their active local input window and replay it
        at each step.  This keeps sampling bounded while preserving the exact
        causal stage computation inside a configured hierarchy window.
        """
        if self.kind == "mlp":
            raise ValueError(
                "MEGABYTE window MLP runs one complete fine patch at a time; "
                "use the bottom-up patch sampler rather than stage.step()"
            )
        if self.kind == "transformer":
            return self.layers.step(x_t, state, self.seq_len)
        if self.kind == "modern":
            if state is None:
                state = [None] * len(self.layers)
            next_state = []
            for layer, layer_state in zip(self.layers, state):
                x_t, layer_state = layer.step(x_t, layer_state, self.seq_len)
                next_state.append(layer_state)
            return self.output_norm(x_t), next_state
        if self.kind == "gpt2":
            out, state = self.layers.forward_hidden(x_t.unsqueeze(1), past_kv=state)
            return out[:, 0], state
        if self.kind in {"gmlp", "amlp", "mlpmixer"}:
            if state is None:
                state = [None] * len(self.layers)
            next_state = []
            for layer, layer_state in zip(self.layers, state):
                x_t, layer_state = layer.step(x_t, layer_state)
                next_state.append(layer_state)
            return (self.output_norm(x_t) if hasattr(self, "output_norm") else x_t), next_state
        if self.kind in {"gru", "rnn", "rnn_relu", "lstm"}:
            return self.layers.step(x_t, state)
        if self.kind in MEGABYTE_CELL_MIXERS:
            out, state = self.layers(x_t.unsqueeze(1), state)
            return out[:, 0], state
        if self.kind in MEGABYTE_HIDDEN_LM_MIXERS:
            out, state = self.layers.forward_hidden(x_t.unsqueeze(1), state)
            return out[:, 0], state
        if self.kind in _MEGABYTE_SCAN_MIXERS:
            if state is None:
                state = [({} if self.kind in {"mamba", "mamba_ssm"} else layer.init_state(x_t)
                          if self.kind == "rwkv" else None if self.kind in MEGABYTE_LINEAR_STACK_MIXERS | {"rwkv7"}
                          else x_t.new_zeros(x_t.shape))
                         for layer in self.layers]
            next_state = []
            for layer, layer_state in zip(self.layers, state):
                x_t, layer_state = layer.step(x_t, layer_state)
                next_state.append(layer_state)
            return x_t, next_state
        if state is None:
            inputs = x_t.unsqueeze(1)
        else:
            inputs = torch.cat((state["inputs"], x_t.unsqueeze(1)), dim=1)
            inputs = inputs[:, -self.seq_len:]

        # Token-mixing MLPs have learned matrices sized to the configured
        # sequence length.  Replaying a fixed-size, right-padded window also
        # gives every mixer the same causal positions as the full-window path.
        window = x_t.new_zeros(x_t.size(0), self.seq_len, x_t.size(1))
        window[:, :inputs.size(1)] = inputs
        out = self.forward(window)
        return out[:, inputs.size(1) - 1], {"inputs": inputs}


class MegaByteLM(nn.Module):
    """Configurable hierarchical MEGABYTE for the flat token-stream interface.

    Stages are ordered coarse-to-fine.  ``stage_seq_lens`` defines how many
    child groups each stage contains, so its product is the model's maximum
    token window.  A stage processes a causal sequence of its groups, then its
    *previous* states are projected into the next finer stage.  This preserves
    autoregressive causality while allowing each stage to use its own width,
    depth, and attention-head count.
    """
    def __init__(
        self,
        vocab_size,
        dim=None,
        depth=None,
        patch_size=4,
        *,
        stage_dims=None,
        stage_depths=None,
        stage_heads=None,
        stage_seq_lens=None,
        stage_child_embed_dims=None,
        stage_mixer="transformer",
        hierarchy_mode="top_down",
        bottom_up_encoder_stage_mixer=None,
        bottom_up_decoder_stage_mixer=None,
        bottom_up_context_merge="concat",
        fused_rnn_norm_type=0,
        fused_rnn_res_every=0,
        fused_rnn_res_type=0,
        fused_rnn_dropout=0.0,
    ):
        super().__init__()
        # The scalar arguments remain supported for older saved configurations.
        if stage_dims is None:
            if dim is None or depth is None or dim < 4 or dim % 4:
                raise ValueError("legacy MEGABYTE embed_dim must be a multiple of 4")
            stage_dims = (dim, dim // 2)
            stage_depths = (max(1, depth // 2), max(1, depth - depth // 2))
            stage_heads = (8, 8)
            stage_seq_lens = (1024, patch_size)

        self.stage_dims = tuple(int(value) for value in stage_dims)
        self.stage_depths = tuple(int(value) for value in stage_depths)
        self.stage_heads = tuple(int(value) for value in stage_heads)
        self.stage_seq_lens = tuple(int(value) for value in stage_seq_lens)
        if stage_child_embed_dims is None:
            stage_child_embed_dims = tuple(min(64, stage_dim) for stage_dim in self.stage_dims)
        self.stage_child_embed_dims = tuple(int(value) for value in stage_child_embed_dims)
        if hierarchy_mode not in {"top_down", "bottom_up"}:
            raise ValueError("MEGABYTE hierarchy_mode must be 'top_down' or 'bottom_up'")
        if bottom_up_context_merge not in {"add", "norm_add", "concat", "norm_concat"}:
            raise ValueError("bottom_up_context_merge must be add, norm_add, concat, or norm_concat")
        self.hierarchy_mode = hierarchy_mode
        self.bottom_up_context_merge = bottom_up_context_merge
        self.num_stages = len(self.stage_dims)
        if self.num_stages < 2 or not (
            len(self.stage_depths) == len(self.stage_heads) == len(self.stage_seq_lens)
            == len(self.stage_child_embed_dims) == self.num_stages
        ):
            raise ValueError("MEGABYTE stage dims, child embedding dims, depths, heads, and sequence lengths must have the same length of at least 2")
        if any(value < 1 for values in (
            self.stage_dims, self.stage_child_embed_dims, self.stage_depths, self.stage_heads, self.stage_seq_lens
        ) for value in values):
            raise ValueError("MEGABYTE stage parameters must all be positive")
        if any(child_dim > stage_dim for child_dim, stage_dim in zip(self.stage_child_embed_dims, self.stage_dims)):
            raise ValueError("MEGABYTE child embedding dimensions cannot exceed their stage hidden dimensions")

        def normalize_stage_mixers(mixers, name):
            if isinstance(mixers, (str, int)):
                mixers = (mixers,) * self.num_stages
            elif isinstance(mixers, (tuple, list)) and len(mixers) == self.num_stages:
                mixers = tuple(mixers)
            else:
                raise ValueError(
                    f"MEGABYTE {name} must be a model ID/string or a sequence matching the stage count"
                )
            return tuple(resolve_megabyte_stage_mixer(mixer) for mixer in mixers)

        shared_stage_mixers = normalize_stage_mixers(stage_mixer, "stage_mixer")
        if hierarchy_mode == "top_down":
            if bottom_up_encoder_stage_mixer is not None or bottom_up_decoder_stage_mixer is not None:
                raise ValueError("separate encoder/decoder mixers require hierarchy_mode='bottom_up'")
            if "mlp" in shared_stage_mixers:
                raise ValueError("MEGABYTE window MLP requires hierarchy_mode='bottom_up'")
            self.stage_mixer = shared_stage_mixers
            self.bottom_up_encoder_stage_mixer = None
            self.bottom_up_decoder_stage_mixer = None
        else:
            self.bottom_up_encoder_stage_mixer = normalize_stage_mixers(
                shared_stage_mixers if bottom_up_encoder_stage_mixer is None else bottom_up_encoder_stage_mixer,
                "bottom_up_encoder_stage_mixer",
            )
            self.bottom_up_decoder_stage_mixer = normalize_stage_mixers(
                shared_stage_mixers if bottom_up_decoder_stage_mixer is None else bottom_up_decoder_stage_mixer,
                "bottom_up_decoder_stage_mixer",
            )
            # Keep the historical name as the decoder spelling for state-dict
            # and external-call compatibility with older bottom-up checkpoints.
            self.stage_mixer = self.bottom_up_decoder_stage_mixer
            fine_stage = self.num_stages - 1
            encoder_mlp_stages = [
                stage for stage, mixer in enumerate(self.bottom_up_encoder_stage_mixer)
                if mixer == "mlp"
            ]
            decoder_mlp_stages = [
                stage for stage, mixer in enumerate(self.bottom_up_decoder_stage_mixer)
                if mixer == "mlp"
            ]
            if decoder_mlp_stages and decoder_mlp_stages != [fine_stage]:
                raise ValueError(
                    "MEGABYTE window MLP may be selected only for the fine bottom-up decoder stage"
                )
            if encoder_mlp_stages and encoder_mlp_stages != [fine_stage]:
                raise ValueError("MEGABYTE window MLP may be selected only for the fine bottom-up encoder stage")
            if encoder_mlp_stages and not decoder_mlp_stages:
                raise ValueError(
                    "a fine bottom-up MLP encoder requires a fine MLP decoder so both sides advance by patches"
                )
        self.fused_rnn_norm_type = int(fused_rnn_norm_type)
        self.fused_rnn_res_every = int(fused_rnn_res_every)
        self.fused_rnn_res_type = int(fused_rnn_res_type)
        self.fused_rnn_dropout = float(fused_rnn_dropout)

        self.max_seq_len = math.prod(self.stage_seq_lens)
        self.stage_mixers = nn.ModuleList([
            MegaByteStageMixer(
                stage_mixer, stage_dim, stage_depth, stage_heads,
                # Both decoder topologies use a causal start slot.  In the
                # bottom-up hourglass this lets a parent's context condition
                # the first child group instead of arriving one level late.
                # A fine window MLP is the patch decoder exception: it maps
                # exactly the complete parent-context patch to all children.
                stage_len if (
                    self.hierarchy_mode == "bottom_up"
                    and stage_mixer == "mlp"
                    and stage_len == self.stage_seq_lens[-1]
                ) else stage_len + 1,
                lmu_theta=stage_len,
                rnn_norm_type=self.fused_rnn_norm_type,
                rnn_res_every=self.fused_rnn_res_every,
                rnn_res_type=self.fused_rnn_res_type,
                rnn_dropout=self.fused_rnn_dropout,
            )
            for stage_mixer, stage_dim, stage_depth, stage_heads, stage_len in zip(
                self.stage_mixer, self.stage_dims, self.stage_depths, self.stage_heads, self.stage_seq_lens
            )
        ])
        self.context_projs = nn.ModuleList([
            nn.Linear(self.stage_dims[stage], self.stage_seq_lens[stage + 1] * self.stage_dims[stage + 1])
            for stage in range(self.num_stages - 1)
        ])
        self.head = nn.Linear(self.stage_dims[-1], vocab_size)

        if self.hierarchy_mode == "top_down":
            # These modules flatten raw token groups, which is the original
            # pure top-down topology.  The bottom-up hourglass must not own
            # them: it composes child representations instead.
            self.stage_starts = nn.ParameterList([
                nn.Parameter(torch.randn(stage_dim)) for stage_dim in self.stage_dims
            ])
            self.stage_child_embeddings = nn.ModuleList([
                nn.Embedding(vocab_size, child_dim) for child_dim in self.stage_child_embed_dims
            ])
            self.stage_input_projs = nn.ModuleList()
            for stage, (stage_dim, child_dim) in enumerate(zip(self.stage_dims, self.stage_child_embed_dims)):
                child_width = math.prod(self.stage_seq_lens[stage + 1:])
                self.stage_input_projs.append(nn.Sequential(
                    nn.LayerNorm(child_width * child_dim),
                    nn.Linear(child_width * child_dim, stage_dim),
                    nn.LayerNorm(stage_dim),
                ))
        else:
            # Fine representations are composed one level at a time before
            # decoding.  A parent never flattens the raw token patch below its
            # immediate children.
            self.bottom_up_token_embedding = nn.Embedding(
                vocab_size, self.stage_child_embed_dims[-1]
            )
            self.bottom_up_encoder_input_projs = nn.ModuleList()
            for stage, stage_dim in enumerate(self.stage_dims):
                input_width = (
                    self.stage_child_embed_dims[-1]
                    if stage == self.num_stages - 1
                    else self.stage_seq_lens[stage + 1] * self.stage_dims[stage + 1]
                )
                self.bottom_up_encoder_input_projs.append(nn.Sequential(
                    nn.LayerNorm(input_width),
                    nn.Linear(input_width, stage_dim),
                    nn.LayerNorm(stage_dim),
                ))
            self.bottom_up_encoder_mixers = nn.ModuleList([
                MegaByteStageMixer(
                    stage_mixer, stage_dim, stage_depth, stage_heads, stage_len, lmu_theta=stage_len,
                    rnn_norm_type=self.fused_rnn_norm_type,
                    rnn_res_every=self.fused_rnn_res_every,
                    rnn_res_type=self.fused_rnn_res_type,
                    rnn_dropout=self.fused_rnn_dropout,
                )
                for stage_mixer, stage_dim, stage_depth, stage_heads, stage_len in zip(
                    self.bottom_up_encoder_stage_mixer, self.stage_dims, self.stage_depths,
                    self.stage_heads, self.stage_seq_lens
                )
            ])
            # The learned expansion supplies a different context to each
            # child.  In parallel, a shared direct parent lane gives every
            # child access to the same parent summary.  When widths match it
            # is an identity residual; otherwise it is the unavoidable
            # parent-to-child projection, gated so the expansion can refine
            # rather than overwrite it.
            self.bottom_up_context_skip_projs = nn.ModuleList([
                nn.Identity() if self.stage_dims[stage] == self.stage_dims[stage + 1]
                else nn.Linear(self.stage_dims[stage], self.stage_dims[stage + 1], bias=False)
                for stage in range(self.num_stages - 1)
            ])
            self.bottom_up_context_gates = nn.ModuleList([
                nn.Linear(self.stage_dims[stage], self.stage_dims[stage + 1])
                for stage in range(self.num_stages - 1)
            ])
            for gate in self.bottom_up_context_gates:
                nn.init.zeros_(gate.weight)
                nn.init.ones_(gate.bias)
            # Preserve each child encoder's within-group causal context on its
            # way back down the hourglass.  These lanes start disabled so the
            # initial optimization dynamics keep the previous decoder input
            # exactly.
            self.bottom_up_encoder_skips = nn.ModuleList([
                nn.Linear(self.stage_dims[stage + 1], self.stage_dims[stage + 1], bias=False)
                for stage in range(self.num_stages - 1)
            ])
            self.bottom_up_encoder_skip_gates = nn.ParameterList([
                nn.Parameter(torch.zeros(self.stage_dims[stage + 1]))
                for stage in range(self.num_stages - 1)
            ])
            # One decoder start vector per scale.  The root start establishes
            # the left edge of the stream; finer starts receive their first
            # parent context before any local child input is decoded.
            self.bottom_up_decoder_starts = nn.ParameterList([
                nn.Parameter(torch.randn(stage_dim)) for stage_dim in self.stage_dims
            ])
            # Create ablation-only modules after every baseline parameter so
            # same-seed additive and concat arms share identical initialization.
            self.bottom_up_context_norms = None
            self.bottom_up_context_scales = None
            if self.bottom_up_context_merge in {"norm_add", "norm_concat"}:
                self.bottom_up_context_norms = nn.ModuleList([
                    nn.LayerNorm(self.stage_dims[stage + 1])
                    for stage in range(self.num_stages - 1)
                ])
                self.bottom_up_context_scales = nn.ParameterList([
                    nn.Parameter(torch.ones(1))
                    for _ in range(self.num_stages - 1)
                ])
            self.bottom_up_context_merges = None
            if self.bottom_up_context_merge in {"concat", "norm_concat"}:
                self.bottom_up_context_merges = nn.ModuleList([
                    nn.Linear(2 * self.stage_dims[stage + 1], self.stage_dims[stage + 1])
                    for stage in range(self.num_stages - 1)
                ])
                for merge, stage in zip(self.bottom_up_context_merges, range(self.num_stages - 1)):
                    child_dim = self.stage_dims[stage + 1]
                    with torch.no_grad():
                        merge.weight.zero_()
                        eye = torch.eye(child_dim, device=merge.weight.device, dtype=merge.weight.dtype)
                        merge.weight[:, :child_dim].copy_(eye)
                        merge.weight[:, child_dim:].copy_(eye)
                        nn.init.zeros_(merge.bias)
        self._bottom_up_merge_metric_sums = None

    @property
    def is_incremental(self):
        if self.hierarchy_mode != "bottom_up":
            return all(mixer in INCREMENTAL_MEGABYTE_MIXERS for mixer in self.stage_mixer)
        fine_stage = self.num_stages - 1
        return all(
            mixer in INCREMENTAL_MEGABYTE_MIXERS
            or (self.uses_bottom_up_window_mlp and stage == fine_stage and mixer == "mlp")
            for stage, mixer in enumerate(self.bottom_up_encoder_stage_mixer)
        ) and all(
            mixer in INCREMENTAL_MEGABYTE_MIXERS
            or (stage == fine_stage and mixer == "mlp")
            for stage, mixer in enumerate(self.bottom_up_decoder_stage_mixer)
        )

    @property
    def uses_bottom_up_window_mlp(self):
        return (
            self.hierarchy_mode == "bottom_up"
            and self.bottom_up_decoder_stage_mixer[-1] == "mlp"
        )

    def _bottom_up_record_effective_rank(self, metric_stage: int, source: torch.Tensor,
                                         metric_name: str):
        """Accumulate squared-singular-value participation ratio for one source."""
        if self._bottom_up_merge_metric_sums is None or source.ndim < 3:
            return
        source = source.detach().float().reshape(-1, source.size(-1))
        singular_values = torch.linalg.svdvals(source)
        power = singular_values.square()
        effective_rank = power.sum().square() / power.square().sum().clamp_min(torch.finfo(power.dtype).eps)
        values = self._bottom_up_merge_metric_sums[metric_stage]
        if values is None:
            values = {}
        values[f"{metric_name}_sum"] = values.get(f"{metric_name}_sum", 0) + effective_rank
        values[f"{metric_name}_count"] = values.get(f"{metric_name}_count", 0) + 1
        values[f"{metric_name}_width"] = source.size(-1)
        self._bottom_up_merge_metric_sums[metric_stage] = values

    def _bottom_up_project_context(self, stage: int, parent_state: torch.Tensor) -> torch.Tensor:
        """Project one causal parent state into contexts for its child group.

        The slot-specific expansion preserves normal MEGABYTE fan-out.  The
        gated shared lane is a residual-style direct route from the parent;
        it introduces no sibling dependency, so all child groups remain
        parallel.  A narrower child width still imposes a rank bound only on
        that direct projection, not on the existence of the route.
        """
        # This is the exact decoder representation that feeds the fan-out
        # projection in the vectorized path.
        self._bottom_up_record_effective_rank(stage, parent_state, "source_effective_rank")
        next_len = self.stage_seq_lens[stage + 1]
        next_dim = self.stage_dims[stage + 1]
        expanded = self.context_projs[stage](parent_state).view(
            *parent_state.shape[:-1], next_len, next_dim
        )
        direct = self.bottom_up_context_skip_projs[stage](parent_state).unsqueeze(-2)
        gate = torch.sigmoid(self.bottom_up_context_gates[stage](parent_state)).unsqueeze(-2)
        return expanded + gate * direct

    def _bottom_up_decoder_local(self, parent_stage: int, encoder_input: torch.Tensor,
                                 encoder_state: torch.Tensor) -> torch.Tensor:
        """Add a zero-gated, same-position encoder skip to one child input."""
        skip = self.bottom_up_encoder_skips[parent_stage](encoder_state)
        gate = self.bottom_up_encoder_skip_gates[parent_stage]
        contribution = gate * skip
        if self._bottom_up_merge_metric_sums is not None:
            encoder_norms = encoder_input.detach().float().norm(dim=-1)
            contribution_norms = contribution.detach().float().norm(dim=-1)
            values = self._bottom_up_merge_metric_sums[parent_stage]
            if values is None:
                values = {}
            values["encoder_input_norm_sum"] = values.get("encoder_input_norm_sum", 0) + encoder_norms.sum()
            values["skip_contribution_norm_sum"] = values.get("skip_contribution_norm_sum", 0) + contribution_norms.sum()
            values["skip_norm_count"] = values.get("skip_norm_count", 0) + encoder_norms.numel()
            values["skip_gate_l2"] = gate.detach().float().norm()
            self._bottom_up_merge_metric_sums[parent_stage] = values
        return encoder_input + contribution

    def reset_bottom_up_merge_metrics(self):
        """Begin accumulating detached local/context norms for one or more forwards."""
        if self.hierarchy_mode != "bottom_up":
            raise ValueError("bottom-up merge metrics require hierarchy_mode='bottom_up'")
        self._bottom_up_merge_metric_sums = [None] * (self.num_stages - 1)

    def bottom_up_merge_metrics(self):
        """Return merge, source-rank, and skip-use metrics since the last reset."""
        if self._bottom_up_merge_metric_sums is None:
            return {}
        metrics = {}
        for stage, values in enumerate(self._bottom_up_merge_metric_sums):
            if values is None:
                continue
            result = {}
            if "local_sum" in values:
                local_norm = values["local_sum"] / values["count"]
                context_norm = values["context_sum"] / values["count"]
                result.update({
                    "local_norm": float(local_norm),
                    "context_norm": float(context_norm),
                    "context_to_local_ratio": float(context_norm / local_norm.clamp_min(torch.finfo(local_norm.dtype).eps)),
                })
            if "source_effective_rank_sum" in values:
                source_rank = values["source_effective_rank_sum"] / values["source_effective_rank_count"]
                result.update({
                    "source_effective_rank": float(source_rank),
                    "source_rank_fraction": float(source_rank / values["source_effective_rank_width"]),
                })
            if "encoder_effective_rank_sum" in values:
                encoder_rank = values["encoder_effective_rank_sum"] / values["encoder_effective_rank_count"]
                result.update({
                    "encoder_effective_rank": float(encoder_rank),
                    "encoder_rank_fraction": float(encoder_rank / values["encoder_effective_rank_width"]),
                })
            if "skip_contribution_norm_sum" in values:
                encoder_input_norm = values["encoder_input_norm_sum"] / values["skip_norm_count"]
                skip_contribution_norm = values["skip_contribution_norm_sum"] / values["skip_norm_count"]
                result.update({
                    "skip_contribution_norm": float(skip_contribution_norm),
                    "skip_to_encoder_input_ratio": float(
                        skip_contribution_norm / encoder_input_norm.clamp_min(torch.finfo(encoder_input_norm.dtype).eps)
                    ),
                    "skip_gate_l2": float(values["skip_gate_l2"]),
                })
            metrics[stage + 1] = result
        return metrics

    def _bottom_up_merge_decoder_sources(self, parent_stage: int, local: torch.Tensor,
                                         context: torch.Tensor) -> torch.Tensor:
        """Merge local child content and causal parent context for one decoder level."""
        if self._bottom_up_merge_metric_sums is not None:
            local_norms = local.detach().float().norm(dim=-1)
            context_norms = context.detach().float().norm(dim=-1)
            values = self._bottom_up_merge_metric_sums[parent_stage]
            if values is None:
                values = {}
            values["local_sum"] = values.get("local_sum", 0) + local_norms.sum()
            values["context_sum"] = values.get("context_sum", 0) + context_norms.sum()
            values["count"] = values.get("count", 0) + local_norms.numel()
            self._bottom_up_merge_metric_sums[parent_stage] = values
        if self.bottom_up_context_norms is not None:
            context = self.bottom_up_context_scales[parent_stage] * self.bottom_up_context_norms[parent_stage](context)
        if self.bottom_up_context_merges is not None:
            return self.bottom_up_context_merges[parent_stage](torch.cat((local, context), dim=-1))
        return local + context

    def _new_incremental_cache(self, batch_size: int, device, dtype):
        """Create the hierarchical sampler cache at the left edge of a stream."""
        if self.hierarchy_mode == "bottom_up":
            return self._new_bottom_up_incremental_cache(batch_size, device, dtype)
        cache = {
            # One recurrent state and one boundary counter per hierarchy stage.
            "stage_states": [None] * self.num_stages,
            "boundary_counters": [0] * self.num_stages,
            # Raw child groups accumulated until a parent stage can advance.
            "partial_child_token_buffers": [None] * self.num_stages,
            # Contexts supplied by a parent for its current child-stage group.
            "input_context_queues": [None] * self.num_stages,
            # Contexts produced after each stage's start / input step.
            "projected_parent_context_queues": [None] * (self.num_stages - 1),
            "batch_size": batch_size,
        }
        self._begin_incremental_stage(0, cache, None, device, dtype)
        return cache

    def _new_bottom_up_incremental_cache(self, batch_size: int, device, dtype):
        """State for streaming the bottom-up encoder/decoder hourglass."""
        cache = {
            "batch_size": batch_size,
            "encoder_states": [None] * self.num_stages,
            # Window-MLP fine encoders buffer a whole patch before composing
            # it; recurrent encoders leave these lists unused.
            "encoder_inputs": [[] for _ in range(self.num_stages)],
            "encoder_outputs": [[] for _ in range(self.num_stages)],
            "encoder_counters": [0] * self.num_stages,
            "decoder_states": [None] * self.num_stages,
            "decoder_counters": [0] * self.num_stages,
            "decoder_contexts": [None] * self.num_stages,
            # Fine window-MLP decoder outputs are sampled position-by-position
            # from one precomputed patch until the encoder completes it.
            "decoder_patch_outputs": [None] * self.num_stages,
            # A finite-context root restarts its window only once the next
            # token arrives, so the window's final prediction still uses the
            # completed window's last root output (as in the vectorized path).
            "root_window_restart_pending": False,
        }
        self._begin_bottom_up_root_window(cache, device, dtype)
        return cache

    def _begin_bottom_up_root_window(self, cache, device, dtype):
        """Seed the root decoder start slot exactly as a fresh window does."""
        batch_size = cache["batch_size"]
        cache["root_window_restart_pending"] = False
        cache["decoder_counters"][0] = 0
        root_start = self.bottom_up_decoder_starts[0].to(device=device, dtype=dtype).expand(batch_size, -1)
        attended, state = self.stage_mixers[0].step(root_start, None)
        # Keep each causal decoder input on a residual path.  A parent context
        # otherwise has to survive an entirely fresh child recurrent state
        # before it can reach the next hierarchy level, which chokes wide
        # parent-to-child fan-outs.
        attended = attended + root_start
        cache["decoder_states"][0] = state
        if self.num_stages > 1:
            contexts = self._bottom_up_project_context(0, attended)
            self._begin_bottom_up_decoder_group(1, contexts, cache)

    def _begin_bottom_up_decoder_group(self, stage: int, contexts: torch.Tensor, cache):
        """Prime a child decoder group so its first child sees parent context."""
        cache["decoder_contexts"][stage] = contexts
        cache["decoder_counters"][stage] = 0
        if self.uses_bottom_up_window_mlp and stage == self.num_stages - 1:
            decoded = self.stage_mixers[stage](contexts) + contexts
            cache["decoder_patch_outputs"][stage] = decoded
            cache["decoder_states"][stage] = None
            return
        start = self.bottom_up_decoder_starts[stage].to(dtype=contexts.dtype).expand(
            contexts.size(0), -1
        ) + contexts[:, 0]
        attended, state = self.stage_mixers[stage].step(
            start, self.stage_mixers[stage].initial_state_from_parent(contexts[:, 0])
        )
        attended = attended + start
        cache["decoder_states"][stage] = state
        if stage + 1 < self.num_stages:
            child_contexts = self._bottom_up_project_context(stage, attended)
            self._begin_bottom_up_decoder_group(stage + 1, child_contexts, cache)

    def init_incremental_cache(self, idx: torch.Tensor):
        """Initialize a sampler / TBPTT cache without consuming ``idx``."""
        if not self.is_incremental:
            raise ValueError("this MEGABYTE stage mixer has no incremental cache")
        return self._new_incremental_cache(idx.size(0), idx.device, self.head.weight.dtype)

    def _begin_incremental_stage(self, stage: int, cache, input_contexts, device, dtype):
        """Reset a child group, seed its start state, and seed its child."""
        batch_size = cache["batch_size"]
        start = self.stage_starts[stage].to(device=device, dtype=dtype).expand(batch_size, -1)
        # The first child group is conditioned from this start state.  Seed it
        # with the parent's first causal context so information from the
        # preceding parent group is available immediately, not one group late.
        if input_contexts is not None:
            start = start + input_contexts[:, 0].to(device=device, dtype=dtype)
        attended, state = self.stage_mixers[stage].step(
            start,
            self.stage_mixers[stage].initial_state_from_parent(
                None if input_contexts is None else input_contexts[:, 0]
            ),
        )
        cache["stage_states"][stage] = state
        cache["boundary_counters"][stage] = 0
        cache["partial_child_token_buffers"][stage] = None
        cache["input_context_queues"][stage] = input_contexts

        if stage + 1 < self.num_stages:
            next_len = self.stage_seq_lens[stage + 1]
            next_dim = self.stage_dims[stage + 1]
            contexts = self.context_projs[stage](attended).view(batch_size, next_len, next_dim)
            cache["projected_parent_context_queues"][stage] = contexts
            self._begin_incremental_stage(stage + 1, cache, contexts, device, dtype)

    def _consume_incremental_group(self, stage: int, tokens: torch.Tensor, cache):
        """Advance one stage after one complete group from its child stage."""
        batch_size = tokens.size(0)
        child_width = math.prod(self.stage_seq_lens[stage + 1:])
        if tokens.shape != (batch_size, child_width):
            raise ValueError(f"stage {stage} expected a child group of width {child_width}, got {tuple(tokens.shape)}")

        raw = self.stage_child_embeddings[stage](tokens).reshape(
            batch_size, child_width * self.stage_child_embed_dims[stage]
        )
        x_t = self.stage_input_projs[stage](raw)
        contexts = cache["input_context_queues"][stage]
        counter = cache["boundary_counters"][stage]
        if contexts is not None:
            x_t = x_t + contexts[:, counter]
        attended, state = self.stage_mixers[stage].step(x_t, cache["stage_states"][stage])
        cache["stage_states"][stage] = state
        cache["boundary_counters"][stage] = counter + 1

        if stage == self.num_stages - 1:
            result = self.head(attended)
        else:
            result = None

        # The root stream is intentionally unbounded: after its configured
        # training window recurrent cores keep their state.  Attention cores
        # instead restart at that boundary because their fixed KV storage is
        # deliberately limited to the configured local stage window.
        if stage == 0:
            if self.stage_mixers[stage].kind in {"transformer", "modern", "gpt2", "gmlp", "amlp", "mlpmixer"} and counter + 1 == self.stage_seq_lens[stage]:
                self._begin_incremental_stage(0, cache, None, tokens.device, raw.dtype)
                return result
            contexts = self.context_projs[stage](attended).view(
                batch_size, self.stage_seq_lens[1], self.stage_dims[1]
            )
            cache["projected_parent_context_queues"][stage] = contexts
            self._begin_incremental_stage(1, cache, contexts, tokens.device, raw.dtype)
            return result

        # A non-root stage may continue within the same parent group.  Its
        # newly attended state conditions the next child group; at the group
        # boundary the enclosing parent instead supplies a fresh start state.
        if stage + 1 < self.num_stages and cache["boundary_counters"][stage] < self.stage_seq_lens[stage]:
            next_len = self.stage_seq_lens[stage + 1]
            next_dim = self.stage_dims[stage + 1]
            contexts = self.context_projs[stage](attended).view(batch_size, next_len, next_dim)
            cache["projected_parent_context_queues"][stage] = contexts
            self._begin_incremental_stage(stage + 1, cache, contexts, tokens.device, raw.dtype)

        buffer = cache["partial_child_token_buffers"][stage]
        buffer = tokens if buffer is None else torch.cat((buffer, tokens), dim=1)
        cache["partial_child_token_buffers"][stage] = buffer
        if cache["boundary_counters"][stage] == self.stage_seq_lens[stage]:
            # A completed child-stage group becomes one raw input for its
            # parent.  The parent's newly projected context then starts this
            # stage's next group.
            self._consume_incremental_group(stage - 1, buffer, cache)
        return result

    def _incremental_forward(self, idx: torch.Tensor, cache=None):
        if self.hierarchy_mode == "bottom_up":
            return self._bottom_up_incremental_forward(idx, cache)
        if idx.ndim != 2:
            raise ValueError("MEGABYTE input must have shape [batch, time]")
        if cache is None:
            cache = self._new_incremental_cache(idx.size(0), idx.device, self.head.weight.dtype)
        if cache["batch_size"] != idx.size(0):
            raise ValueError("incremental MEGABYTE cache batch size does not match input")
        logits = []
        for token in idx.unbind(dim=1):
            value = self._consume_incremental_group(self.num_stages - 1, token.unsqueeze(1), cache)
            logits.append(value)
        return torch.stack(logits, dim=1) if logits else self.head.weight.new_empty(idx.size(0), 0, self.head.out_features), cache

    def _bottom_up_encoder_item(self, stage: int, x_t: torch.Tensor, cache,
                                defer_parent_emit: bool = False):
        """Consume one encoder item, recursively emitting completed parents."""
        attended, state = self.bottom_up_encoder_mixers[stage].step(
            x_t, cache["encoder_states"][stage]
        )
        # Preserve the projected child representation around each encoder
        # recurrence.  Root probes show that a stacked root RNN can erase
        # detail already present in this causal input before its decoder ever
        # receives it.
        attended = attended + x_t
        cache["encoder_states"][stage] = state
        cache["encoder_counters"][stage] += 1

        if stage == 0:
            # Root outputs feed only the root decoder; nothing reads a root
            # output list, so keeping one would grow for the whole stream.
            self._bottom_up_decoder_item(0, attended, cache)
            if (
                self.bottom_up_encoder_mixers[stage].kind in {"transformer", "modern", "gpt2", "gmlp", "amlp", "mlpmixer"}
                and cache["encoder_counters"][stage] == self.stage_seq_lens[stage]
            ):
                cache["encoder_states"][stage] = None
                cache["encoder_outputs"][stage] = []
                cache["encoder_counters"][stage] = 0
            return attended
        cache["encoder_outputs"][stage].append(attended)
        if (cache["encoder_counters"][stage] != self.stage_seq_lens[stage]
                or defer_parent_emit):
            return attended

        self._bottom_up_encoder_emit_parent(stage, cache)
        return attended

    def _bottom_up_encoder_emit_parent(self, stage: int, cache):
        """Propagate one completed encoder group after its decoder consumed it."""
        if stage <= 0 or cache["encoder_counters"][stage] != self.stage_seq_lens[stage]:
            raise ValueError("bottom-up encoder parent emission requires a completed non-root group")

        # A completed child sequence is the sole input to its parent encoder.
        child_states = torch.stack(cache["encoder_outputs"][stage], dim=1).reshape(
            cache["batch_size"], -1
        )
        cache["encoder_states"][stage] = None
        cache["encoder_outputs"][stage] = []
        cache["encoder_counters"][stage] = 0
        parent_input = self.bottom_up_encoder_input_projs[stage - 1](child_states)
        parent_stage = stage - 1
        if parent_stage == 0:
            self._bottom_up_encoder_item(parent_stage, parent_input, cache)
            return

        # The decoder needs this same-position encoder state for its skip, but
        # its parent must not advance until after the current decoder item is
        # consumed.  Otherwise a new parent context would reset this group's
        # decoder state one item too early.
        parent_state = self._bottom_up_encoder_item(
            parent_stage, parent_input, cache, defer_parent_emit=True
        )
        self._bottom_up_decoder_item(parent_stage, parent_input, cache, parent_state)
        if cache["encoder_counters"][parent_stage] == self.stage_seq_lens[parent_stage]:
            self._bottom_up_encoder_emit_parent(parent_stage, cache)

    def _bottom_up_decoder_item(self, stage: int, x_t: torch.Tensor, cache,
                                encoder_state: Optional[torch.Tensor] = None):
        """Decode one item and prime the following child group from its state."""
        counter = cache["decoder_counters"][stage]
        if stage:
            if encoder_state is None:
                raise ValueError("non-root bottom-up decoder requires its encoder state")
            local = self._bottom_up_decoder_local(stage - 1, x_t, encoder_state)
            context = cache["decoder_contexts"][stage][:, counter]
            x_t = self._bottom_up_merge_decoder_sources(stage - 1, local, context)
        attended, state = self.stage_mixers[stage].step(x_t, cache["decoder_states"][stage])
        attended = attended + x_t
        cache["decoder_states"][stage] = state
        cache["decoder_counters"][stage] = counter + 1

        if stage + 1 < self.num_stages:
            child_contexts = self._bottom_up_project_context(stage, attended)
            self._begin_bottom_up_decoder_group(stage + 1, child_contexts, cache)

        if cache["decoder_counters"][stage] == self.stage_seq_lens[stage]:
            # Local decoder sequences restart at their parent boundary; their
            # next start slot will be seeded by that parent's next context.
            # Recurrent root decoders intentionally carry state beyond a
            # window; finite attention cache stages must restart instead.
            if stage or self.stage_mixers[stage].kind in {"transformer", "modern", "gpt2", "gmlp", "amlp", "mlpmixer"}:
                cache["decoder_states"][stage] = None
                if stage == 0:
                    # The next window must begin from the root start slot,
                    # not from this window's final root output.
                    cache["root_window_restart_pending"] = True
            cache["decoder_counters"][stage] = 0
        return attended

    def _bottom_up_window_mlp_item(self, token: torch.Tensor, cache):
        """Consume one fine token and return the next slot of a patch decoder.

        The decoder patch is prepared from coarse parent contexts before any
        token in that patch is sampled.  The fine encoder only runs after all
        patch tokens are known, then supplies the parent encoder that prepares
        the following patch.  This is causal at patch boundaries and avoids
        feeding target fine-token embeddings to the MLP decoder.
        """
        stage = self.num_stages - 1
        patch_len = self.stage_seq_lens[stage]
        if cache["decoder_patch_outputs"][stage] is None:
            raise RuntimeError("bottom-up window MLP decoder was not primed")
        x_t = self.bottom_up_encoder_input_projs[stage](
            self.bottom_up_token_embedding(token)
        )
        encoder_mixer = self.bottom_up_encoder_mixers[stage]
        if encoder_mixer.kind == "mlp":
            cache["encoder_inputs"][stage].append(x_t)
            cache["encoder_counters"][stage] += 1
        else:
            encoder_state, encoder_cache = encoder_mixer.step(
                x_t, cache["encoder_states"][stage]
            )
            cache["encoder_states"][stage] = encoder_cache
            cache["encoder_outputs"][stage].append(encoder_state + x_t)
            cache["encoder_counters"][stage] += 1
        counter = cache["encoder_counters"][stage]
        if counter < patch_len:
            cache["decoder_counters"][stage] = counter
            # Slot j of a patch decoder predicts that patch's token j, so after
            # consuming tokens 0..j-1 the next prediction is slot j.
            return self.head(cache["decoder_patch_outputs"][stage][:, counter])

        if counter != patch_len:
            raise RuntimeError("bottom-up window MLP encoder patch counter overflow")
        if encoder_mixer.kind == "mlp":
            encoder_input = torch.stack(cache["encoder_inputs"][stage], dim=1)
            cache["encoder_inputs"][stage] = []
            encoder_state = encoder_mixer(encoder_input) + encoder_input
            cache["encoder_outputs"][stage] = list(encoder_state.unbind(dim=1))
        # Completing this patch causes the normal coarse encoder/decoder path
        # to prepare the next patch, whose slot 0 predicts the next token with
        # the completed patch already visible to its parent.
        self._bottom_up_encoder_emit_parent(stage, cache)
        if cache["decoder_patch_outputs"][stage] is None:
            raise RuntimeError("bottom-up window MLP decoder did not prepare the next patch")
        return self.head(cache["decoder_patch_outputs"][stage][:, 0])

    def _bottom_up_incremental_forward(self, idx: torch.Tensor, cache=None):
        if idx.ndim != 2:
            raise ValueError("MEGABYTE input must have shape [batch, time]")
        if cache is None:
            cache = self._new_bottom_up_incremental_cache(
                idx.size(0), idx.device, self.head.weight.dtype
            )
        if cache["batch_size"] != idx.size(0):
            raise ValueError("incremental MEGABYTE cache batch size does not match input")

        final_stage = self.num_stages - 1
        logits = []
        if self.uses_bottom_up_window_mlp:
            for token in idx.unbind(dim=1):
                if cache["root_window_restart_pending"]:
                    self._begin_bottom_up_root_window(cache, token.device, self.head.weight.dtype)
                logits.append(self._bottom_up_window_mlp_item(token, cache))
            return (
                torch.stack(logits, dim=1)
                if logits else self.head.weight.new_empty(idx.size(0), 0, self.head.out_features),
                cache,
            )
        for token in idx.unbind(dim=1):
            if cache["root_window_restart_pending"]:
                self._begin_bottom_up_root_window(cache, token.device, self.head.weight.dtype)
            encoder_input = self.bottom_up_encoder_input_projs[final_stage](
                self.bottom_up_token_embedding(token)
            )
            # Keep parent emission until after this decoder item: the decoder
            # consumes the matching encoder state through the new skip, while
            # preserving the old parent-context boundary ordering.
            encoder_state = self._bottom_up_encoder_item(
                final_stage, encoder_input, cache, defer_parent_emit=True
            )
            decoded = self._bottom_up_decoder_item(
                final_stage, encoder_input, cache, encoder_state
            )
            logits.append(self.head(decoded))

            if cache["encoder_counters"][final_stage] == self.stage_seq_lens[final_stage]:
                self._bottom_up_encoder_emit_parent(final_stage, cache)
            if cache["decoder_counters"][final_stage] == self.stage_seq_lens[final_stage]:
                cache["decoder_states"][final_stage] = None
                cache["decoder_counters"][final_stage] = 0

        return torch.stack(logits, dim=1) if logits else self.head.weight.new_empty(idx.size(0), 0, self.head.out_features), cache

    def _forward_full(self, idx):
        if self.hierarchy_mode == "bottom_up":
            return self._forward_bottom_up(idx)
        B, T = idx.shape
        if T == 0:
            return self.head.weight.new_empty(B, 0, self.head.out_features)
        if T > self.max_seq_len:
            raise ValueError(
                f"MEGABYTE input length {T} exceeds configured stage capacity {self.max_seq_len}"
            )

        padded = F.pad(idx, (0, self.max_seq_len - T), value=0)
        context = None
        final_attended = None
        for stage, (stage_dim, stage_len) in enumerate(zip(self.stage_dims, self.stage_seq_lens)):
            parent_count = math.prod(self.stage_seq_lens[:stage])
            child_width = math.prod(self.stage_seq_lens[stage + 1:])
            groups = padded.view(B, parent_count, stage_len, child_width)
            groups = groups.reshape(B * parent_count, stage_len, child_width)
            raw_tokens = self.stage_child_embeddings[stage](groups).reshape(
                B * parent_count, stage_len, child_width * self.stage_child_embed_dims[stage]
            )
            x = self.stage_input_projs[stage](raw_tokens)
            if context is not None:
                x = x + context
            start = self.stage_starts[stage].expand(x.size(0), 1, -1)
            # Seed the child-stage start token with its first parent context.
            # This is causal because ``context[:, 0]`` came from the preceding
            # parent state; the regular child input keeps that same context.
            if context is not None:
                start = start + context[:, :1]
            attended = self.stage_mixers[stage](
                torch.cat((start, x), dim=1),
                self.stage_mixers[stage].initial_state_from_parent(
                    None if context is None else context[:, 0]
                ),
            )

            if stage == self.num_stages - 1:
                final_attended = attended
                break

            # State j was computed without observing group j+1.  Expanding it
            # into that next group therefore gives the finer stage only prefix
            # context, never its current or future raw tokens.
            previous_states = attended[:, :-1]
            next_len = self.stage_seq_lens[stage + 1]
            next_dim = self.stage_dims[stage + 1]
            context = self.context_projs[stage](previous_states).view(
                B * parent_count * stage_len, next_len, next_dim
            )

        logits = self.head(final_attended[:, 1:]).view(B, self.max_seq_len, -1)
        return logits[:, :T]

    def _forward_bottom_up(self, idx):
        """Run the causal MEGABYTE encoder/decoder hourglass over one window."""
        B, T = idx.shape
        if T == 0:
            return self.head.weight.new_empty(B, 0, self.head.out_features)
        if T > self.max_seq_len:
            raise ValueError(
                f"MEGABYTE input length {T} exceeds configured stage capacity {self.max_seq_len}"
            )

        padded = F.pad(idx, (0, self.max_seq_len - T), value=0)
        encoder_inputs = [None] * self.num_stages
        encoder_states = [None] * self.num_stages

        # Fine → coarse.  Every non-fine encoder projection consumes only the
        # L immediate child representations from the preceding level.
        for stage in range(self.num_stages - 1, -1, -1):
            parent_count = math.prod(self.stage_seq_lens[:stage])
            stage_len = self.stage_seq_lens[stage]
            if stage == self.num_stages - 1:
                raw = self.bottom_up_token_embedding(
                    padded.view(B * parent_count, stage_len)
                )
            else:
                child_len = self.stage_seq_lens[stage + 1]
                child_dim = self.stage_dims[stage + 1]
                raw = encoder_states[stage + 1].reshape(
                    B * parent_count, stage_len, child_len * child_dim
                )
            x = self.bottom_up_encoder_input_projs[stage](raw)
            encoder_inputs[stage] = x
            encoder_states[stage] = self.bottom_up_encoder_mixers[stage](x) + x

        # Compare each parent encoder's causal representation with the decoder
        # source that will fan out to its children.  The final stage has no
        # child fan-out, so it has no corresponding decoder-source metric.
        for stage in range(self.num_stages - 1):
            self._bottom_up_record_effective_rank(
                stage, encoder_states[stage], "encoder_effective_rank"
            )

        # Coarse → fine.  Each decoder sequence has an explicit causal start
        # slot.  Its output conditions child group zero; output after child k
        # conditions child group k + 1.  This preserves causality while
        # allowing a root boundary to reach the first fine child immediately.
        root_start = self.bottom_up_decoder_starts[0].expand(B, 1, -1)
        decoder_input = torch.cat((root_start, encoder_states[0]), dim=1)
        decoded = self.stage_mixers[0](decoder_input) + decoder_input
        # A window-MLP patch predicts its own tokens, so the window's final
        # position needs the patch after this window.  Its contexts follow the
        # root's final output down each level's start slot, just as the
        # streamed cache prepares the next patch.
        tail_context = (
            self._bottom_up_project_context(0, decoded[:, -1])
            if self.uses_bottom_up_window_mlp else None
        )
        for stage in range(self.num_stages - 1):
            parent_count = math.prod(self.stage_seq_lens[:stage])
            stage_len = self.stage_seq_lens[stage]
            next_len = self.stage_seq_lens[stage + 1]
            next_dim = self.stage_dims[stage + 1]
            context = self._bottom_up_project_context(stage, decoded[:, :-1]).reshape(
                B * parent_count * stage_len, next_len, next_dim
            )
            if self.uses_bottom_up_window_mlp and stage + 1 == self.num_stages - 1:
                # Parent contexts are available before this fine patch begins.
                # The MLP can therefore mix every slot and emit the entire
                # patch without consuming its teacher-forced fine tokens.
                decoded = self.stage_mixers[stage + 1](context) + context
                tail_patch = self.stage_mixers[stage + 1](tail_context) + tail_context
                continue
            child_start = self.bottom_up_decoder_starts[stage + 1].expand(
                context.size(0), 1, -1
            ) + context[:, :1]
            local = self._bottom_up_decoder_local(
                stage, encoder_inputs[stage + 1], encoder_states[stage + 1]
            )
            decoder_input = torch.cat((
                child_start, self._bottom_up_merge_decoder_sources(stage, local, context),
            ), dim=1)
            decoded = self.stage_mixers[stage + 1](
                decoder_input,
                self.stage_mixers[stage + 1].initial_state_from_parent(context[:, 0]),
            ) + decoder_input
            if tail_context is not None:
                tail_start = self.bottom_up_decoder_starts[stage + 1].expand(B, 1, -1) + tail_context[:, :1]
                tail_decoded = self.stage_mixers[stage + 1](
                    tail_start,
                    self.stage_mixers[stage + 1].initial_state_from_parent(tail_context[:, 0]),
                ) + tail_start
                tail_context = self._bottom_up_project_context(stage + 1, tail_decoded[:, 0])

        if self.uses_bottom_up_window_mlp:
            # Slot j of a patch predicts that patch's token j, while output i
            # predicts token i + 1: drop the first slot and finish with the
            # following patch's slot 0.
            final_decoder_states = torch.cat(
                (decoded.reshape(B, self.max_seq_len, -1), tail_patch[:, :1]), dim=1
            )[:, 1:]
        else:
            final_decoder_states = decoded[:, 1:]
        logits = self.head(final_decoder_states).view(B, self.max_seq_len, -1)
        return logits[:, :T]

    def forward(self, idx, state=None):
        """Run a batch, or continue a true per-stage recurrent sampler cache."""
        if not self.is_incremental:
            return self._forward_full(idx)
        # The cache is a sampler optimization.  Training batches retain the
        # vectorized hierarchy used before it was added; otherwise every token
        # would be dispatched through Python and tiny recurrent kernels.
        if self.training and state is None:
            return self._forward_full(idx), None
        return self._incremental_forward(idx, state)
# ==============================================================================
# MinRNN Generalized (Multi-Activation)
# ==============================================================================
class ScanBlock_MinRNN_Gen(nn.Module):
    """
    MinRNN with configurable activation for the recurrence gate.
    h_t = a_t * h_{t-1} + b_t
    b_t = W_x x_t
    a_t = Activation(W_z x_t + bias)
    """
    def __init__(self, dim: int, act_type: int = 0):
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm(dim) # Standard LayerNorm for stability
        
        self.Wx = nn.Linear(dim, dim, bias=False)
        
        # FIXED: bias=True is required because you access self.Wz.bias in init logic below
        self.Wz = nn.Linear(dim, dim, bias=True) 
        
        self.out = nn.Linear(dim, dim, bias=False)
        self.act_type = act_type
        
        # Init Wz to produce values close to identity or stable decay initially
        if act_type == 4: # Sigmoid
            # Initialize bias to start with high retention (sigmoid(2.0) ~= 0.88)
            nn.init.constant_(self.Wz.bias, 2.0) 
            # Keep weights small so initially the gate is mostly controlled by bias
            nn.init.xavier_uniform_(self.Wz.weight, gain=0.01) 
            
        elif act_type == 0: # Tanh
             # MinRNN paper approach: Tanh generally decays (-1 to 1).
             # We want to avoid 0 (forgetting) or -1 (oscillation) initially.
             # Small weights ensure we stay in the linear region or near 0, 
             # but strictly speaking Tanh isn't great for "holding" memory compared to Sigmoid.
             nn.init.xavier_uniform_(self.Wz.weight, gain=0.1)
             nn.init.zeros_(self.Wz.bias)
        else:
            # ReLU, SiLU, GELU are unbounded positive. 
            # We want small initial values to avoid explosion (a_t > 1.0).
            nn.init.xavier_uniform_(self.Wz.weight, gain=0.01)
            nn.init.zeros_(self.Wz.bias)

    def _get_a(self, z):
        if self.act_type == 0: return torch.tanh(z)
        if self.act_type == 1: return F.relu(z)
        if self.act_type == 2: return F.silu(z)
        if self.act_type == 3: return F.gelu(z)
        if self.act_type == 4: return torch.sigmoid(z)
        return torch.sigmoid(z)

    def _recurrence_coeff(self, z):
        """Keep every non-log recurrence coefficient strictly inside (-1, 1)."""
        a = self._get_a(z)
        if self.act_type in (0, 4):
            return a
        # ReLU, SiLU, and GELU are unbounded.  Using them directly as a
        # recurrent multiplier makes h grow exponentially as soon as training
        # pushes a channel above one.  This smooth signed squash preserves the
        # selected activation's shape while making the affine scan contractive.
        return a / (1.0 + a.abs())

    def forward_seq(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        x_norm = self.ln(x)
        b = self.Wx(x_norm)
        
        # Calculate decay gate
        z = self.Wz(x_norm)
        #a = self._get_a(z)
        
        # Parallel Scan: h_t = a_t * h_{t-1} + b_t
        if self.act_type == 5:
            # g_act is positive but unbounded, so normalize it into (0, 1)
            # before using it as a recurrence coefficient.
            log_g_a = log_g_act(z)
            log_a = log_g_a - F.softplus(log_g_a)
            log_b = log_g_act(b) # Candidate must be positive for log scan
            h = heinsen_associative_scan_log(log_a, log_b, h0)
        else:
            # Linear Scan
            a = self._recurrence_coeff(z)
            h = pscan_linear_jit(a, b, h0)
        
        # FIXED: Transformer-style Residual
        # x + OutputProjection(Branch)
        # Allows 'x' to flow cleanly and 'self.out' to center the RNN signal.
        return x + self.out(h), h[:, -1, :]

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        x_norm = self.ln(x_t)
        b = self.Wx(x_norm)
        z = self.Wz(x_norm)
        
        if self.act_type == 5:
            # Match the normalized positive coefficient used by forward_seq.
            a = g_act(z)
            a = a / (1.0 + a)
            b = g_act(b)
        else:
            a = self._recurrence_coeff(z)
        
        h = a * h_prev + b
        
        # FIXED: Match forward_seq
        return x_t + self.out(h), h


# ==============================================================================
# MinIndRNN (Parallel Scan Compatible IndRNN) with Extended Activations
# ==============================================================================
class ScanBlock_MinIndRNN(nn.Module):
    """
    MinIndRNN: Independent RNN adapted for Parallel Scan with extended activation support.
    
    Structure: x + Linear(RNN(Norm(x)))
    This creates a stable residual block where the Linear layer centers the RNN output.
    """
    def __init__(self, dim: int, act_type: int = 0):
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm(dim) # Standard LayerNorm
        
        self.Wx = nn.Linear(dim, dim, bias=True)
        self.out = nn.Linear(dim, dim, bias=False)
        self.act_type = act_type
        
        # --- Handle Stateful Activations ---
        self.act_layer = None
        
        # PReLU (init 0.0)
        if self.act_type == 3:
            self.act_layer = nn.PReLU(num_parameters=dim, init=0.0)
        # PReLU (default init 0.25)
        elif self.act_type == 4:
            self.act_layer = nn.PReLU(num_parameters=dim, init=0.25)
        # Snake Activation parameter (alpha)
        elif self.act_type == 11:
            self.snake_alpha = nn.Parameter(torch.ones(dim))
        elif self.act_type == 17:
            pass # g_act handled functionally
        
        # --- Recurrent Weights ---
        # Static recurrent weight 'u'. 
        # Parameterized as sigmoid(p) to ensure stability in (0, 1) range
        self.u_param = nn.Parameter(torch.Tensor(dim))
        nn.init.uniform_(self.u_param, 2.0, 4.0) # Init high for long memory

    def _act(self, x):
        # 0: Tanh
        if self.act_type == 0: 
            return torch.tanh(x)
        # 1: ReLU
        if self.act_type == 1: 
            return F.relu(x)
        # 2: SiLU
        if self.act_type == 2: 
            return F.silu(x)
        
        # 3 & 4: PReLU (Stateful)
        if self.act_type in [3, 4]: 
            # If input is (B, T, D), transpose to (B, D, T) for PReLU
            if x.dim() == 3:
                return self.act_layer(x.transpose(1, 2)).transpose(1, 2)
            # If input is (B, D) (during step/inference), it works as is
            return self.act_layer(x)
        
        # 5: LeakyReLU (0.2)
        if self.act_type == 5: 
            return F.leaky_relu(x, negative_slope=0.2)
        # 6: LeakyReLU (0.01)
        if self.act_type == 6: 
            return F.leaky_relu(x, negative_slope=0.01)
        
        # 7: GELU
        if self.act_type == 7: 
            return F.gelu(x)
        
        # 8: BentIdentity
        if self.act_type == 8:
            return ((torch.sqrt(x.pow(2) + 1) - 1) / 2) + x
            
        # 9: Sine
        if self.act_type == 9: 
            return torch.sin(x)
        
        # 10: Cosine
        if self.act_type == 10: 
            return torch.cos(x)
            
        # 11: Snake
        if self.act_type == 11:
            return x + (1.0 / (self.snake_alpha + 1e-9)) * torch.pow(torch.sin(self.snake_alpha * x), 2)
            
        # 12: x + sin(x)
        if self.act_type == 12: 
            return x + torch.sin(x)
        
        # 13: x + cos(x)
        if self.act_type == 13: 
            return x + torch.cos(x)
            
        # 14: Mish
        if self.act_type == 14:
            return x * torch.tanh(F.softplus(x))
            
        # 15: Cone (Triangle)
        if self.act_type == 15:
            return 1.0 - torch.abs(x - 1)
            
        # 16: SquareReLU
        if self.act_type == 16: 
            return torch.square(F.relu(x))

        # 17: g_act (minGRU style)
        if self.act_type == 17:
            return g_act(x)

        return torch.tanh(x)

    def forward_seq(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        """
        Parallel forward pass (requires parallel_scan_linear).
        """
        B, T, D = x.shape
        x_norm = self.ln(x)
        
        if self.act_type == 17:
            # Log-Space Scan
            # b_t = g_act(Wx) -> log_b = log_g_act(Wx)
            log_b = log_g_act(self.Wx(x_norm))
            
            # a_t = sigmoid(u) -> log_a = logsigmoid(u)
            log_a = F.logsigmoid(self.u_param).view(1, 1, D).expand(B, T, D)
            
            h = heinsen_associative_scan_log(log_a, log_b, h0)
        else:
            # Linear Scan
            b = self._act(self.Wx(x_norm))
            # Recurrent Decay: a_t = Sigmoid(u) [Static across time]
            u = torch.sigmoid(self.u_param).view(1, 1, D).expand(B, T, D)
            h = parallel_scan_linear(u, b, h0)
        
        # FIXED: Transformer-style Residual
        # x + OutputProjection(Branch)
        # This keeps 'x' gradient flow clean (Identity Mapping)
        # and lets 'self.out' center the signal.
        return x + self.out(h), h[:, -1, :]

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        """
        Sequential inference step.
        """
        x_norm = self.ln(x_t)
        b = self._act(self.Wx(x_norm))
        u = torch.sigmoid(self.u_param)
        
        # h_t = u * h_{t-1} + b_t
        h = u * h_prev + b
        
        # FIXED: Match forward_seq
        return x_t + self.out(h), h

# ==============================================================================
# z (Parallel Scan Compatible JANET)
# ==============================================================================
class ScanBlock_MinJANET(nn.Module):
    """
    MinJANET adapted for Log-Space Parallel Scan (minGRU style).
    
    Changes from Standard JANET:
    1. Uses 'g_act' (linear/sigmoid hybrid) instead of 'tanh' for the candidate.
       This is necessary because heinsen_associative_scan_log assumes positive values.
    2. Computes the recurrence in log-space for stability over long sequences.
    
    Recurrence: h_t = f_t * h_{t-1} + (1 - f_t) * c_t
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm(dim)
        
        self.Wf = nn.Linear(dim, dim, bias=True) # Forget Gate
        self.Wc = nn.Linear(dim, dim, bias=True) # Candidate
        self.out = nn.Linear(dim, dim, bias=False)
        self.post = GatedMLP(dim, mult=4, act_name="gelu") # Optional: Match minGRU post-processing
        
        # Chrono Init: bias f to be open (1.0) initially
        # sigmoid(2.0) ~= 0.88, sigmoid(4.0) ~= 0.98
        nn.init.constant_(self.Wf.bias, 2.0) 
        nn.init.xavier_uniform_(self.Wc.weight)

    def forward_seq(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        x_norm = self.ln(x)
        
        # 1. Projections
        z_f = self.Wf(x_norm)
        z_c = self.Wc(x_norm)
        
        # 2. Log-Space Math
        # We need log(f) and log((1-f)*c)
        
        # log(f) where f = sigmoid(z_f)
        # log(sigmoid(x)) = -softplus(-x)
        log_f = -F.softplus(-z_f)
        
        # log(1-f) where f = sigmoid(z_f)
        # Identity: 1 - sigmoid(x) = sigmoid(-x)
        # log(sigmoid(-x)) = -softplus(x)
        log_1_minus_f = -F.softplus(z_f)
        
        # log(c) using stable log_g_act (from your file)
        log_c = log_g_act(z_c)
        
        # Combine input term: b_t = (1-f) * c
        # log(b_t) = log(1-f) + log(c)
        log_values = log_1_minus_f + log_c
        
        # 3. Parallel Scan (Heinsen Log Scan)
        h = heinsen_associative_scan_log(log_f, log_values, h0)
        
        # 4. Output (Residual + Post-MLP like minGRU)
        # Using 0.5 blending for residual stability
        y = x + self.out(h)
        # Optional: Add the GatedMLP post-layer if you want parity with minGRU block
        y = y + self.post(y)
        
        return y, h[:, -1, :]

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        x_norm = self.ln(x_t)
        
        # 1. Projections
        z_f = self.Wf(x_norm)
        z_c = self.Wc(x_norm)
        
        # 2. Activations (Linear Space)
        f = torch.sigmoid(z_f)
        c = g_act(z_c) # Use g_act to match log_g_act from forward_seq
        
        # 3. Recurrence
        # h_t = f * h_{t-1} + (1-f) * c
        h = f * h_prev + (1.0 - f) * c
        
        y = x_t + self.out(h)
        # if using post MLP:
        y = y + self.post(y)
        
        return y, h
# ==============================================================================
# 10. KAN-Transformer (Chebyshev Implementation)
# ==============================================================================
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree=4, act_name="mish"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        
        # Chebyshev coefficients
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1.0 / (input_dim * (degree + 1)))
        
        # Base linear activation (residual)
        self.base_linear = nn.Linear(input_dim, output_dim)
        self.act = get_activation(act_name)

    def forward(self, x):
        # x: (..., input_dim)
        # Normalize x to [-1, 1] for Chebyshev stability (using tanh)
        x_norm = torch.tanh(x)
        
        # Compute Chebyshev polynomials recursively
        # T_0(x) = 1, T_1(x) = x, T_n(x) = 2xT_{n-1} - T_{n-2}
        polys = [torch.ones_like(x_norm), x_norm]
        for i in range(2, self.degree + 1):
            polys.append(2 * x_norm * polys[-1] - polys[-2])
        
        # Stack: (..., input_dim, degree+1)
        poly_stack = torch.stack(polys, dim=-1)
        
        # y = Sum( c_ij * T_j(x_i) )
        # Contract: (...In, Deg) * (In, Out, Deg) -> (...Out)
        y = torch.einsum("...id,iod->...o", poly_stack, self.cheby_coeffs)
        
        # Add base linear transformation
        base = self.base_linear(self.act(x))
        return y + base

class KANBlock(nn.Module):
    def __init__(self, dim, n_heads=4, degree=3, act_name="mish"):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        # Replaces Standard Attention with a KAN-Mixer for this variant
        # or replaces FFN. Here we replace FFN with KAN and keep standard Attn.
        self.attn = ModernAttention(dim, n_heads=n_heads)
        self.kan_ffn = nn.Sequential(
            ChebyKANLayer(dim, dim * 2, degree=degree, act_name=act_name),
            nn.LayerNorm(dim * 2), # Norm inside KAN helps stability
            ChebyKANLayer(dim * 2, dim, degree=degree, act_name=act_name)
        )

    def forward(self, x):
        x = x + self.attn(self.ln(x))
        x = x + self.kan_ffn(self.ln(x))
        return x

class KAN_LM(nn.Module):
    def __init__(self, vocab_size, dim, depth, n_heads=4, act_name="mish"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([KANBlock(dim, n_heads=n_heads, act_name=act_name) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))

# ==============================================================================
# 11. Linear Transformer (Recurrent Form)
# ==============================================================================
class LinearAttentionBlock(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        
        self.norm = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim, dim*4)

    def feature_map(self, x):
        # Katharopoulos feature map: elu(x) + 1
        return F.elu(x) + 1.0

    def forward(self, x, state=None):
        B, T, C = x.shape
        shortcut = x
        x_norm = self.norm(x)
        
        q = self.wq(x_norm).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x_norm).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x_norm).view(B, T, self.n_heads, self.head_dim)
        
        Q = self.feature_map(q)
        K = self.feature_map(k)
        
        # KV calculation for recurrence: outer product K^T * V
        # shape: (B, T, H, D, 1) * (B, T, H, 1, D) -> (B, T, H, D, D)
        KV = torch.einsum("bthd,bthe->bthde", K, v)
        
        S_prev = None
        Z_prev = None
        if state is not None:
            if isinstance(state, dict):
                S_prev = state.get("S", None)
                Z_prev = state.get("Z", None)
            else:
                S_prev = state

        S = torch.cumsum(KV, dim=1)
        if S_prev is not None:
            S = S + S_prev.unsqueeze(1)

        K_sum = torch.cumsum(K, dim=1)
        if Z_prev is not None:
            K_sum = K_sum + Z_prev.unsqueeze(1)

        # Y_t = (Q_t * S_t) / (Q_t * Z_t)
        num = torch.einsum("bthd,bthde->bthe", Q, S)
        den = torch.einsum("bthd,bthd->bth", Q, K_sum).clamp(min=1e-4)
        
        y = num / den.unsqueeze(-1)
        y = y.reshape(B, T, C)
        y = self.wo(y)
        
        x = shortcut + y
        x = x + self.mlp(self.norm(x))
        
        # Return last state for generation
        last_state = {"S": S[:, -1, :, :, :], "Z": K_sum[:, -1, :, :]}
        return x, last_state

    def forward_seq(self, x, h0=None, state=None):
        # Alias for linegen compatibility
        st = state if state is not None else h0
        return self.forward(x, state=st)

class LinearTransformerLM(nn.Module):
    def __init__(self, vocab_size, dim, depth):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([LinearAttentionBlock(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx, state=None):
        x = self.embed(idx)
        if state is None: state = [None]*len(self.blocks)
        new_states = []
        for blk, s in zip(self.blocks, state):
            x, ns = blk(x, s)
            new_states.append(ns)
        return self.head(x), new_states

# ==============================================================================
# 12. H3 (Hungry Hungry Hippos) - Simplified
# ==============================================================================
class H3Block(nn.Module):
    def __init__(self, dim, head_dim=64):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        
        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Shift SSM (Causal convolution — padding on the left only)
        self.shift_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=0, groups=dim)
        
        # Diagonal SSM parameters
        self.dt_proj = nn.Linear(dim, dim)
        self.A_log = nn.Parameter(torch.log(torch.rand(dim) + 0.5)) # Decay
        
        self.mlp = SwiGLU(dim, dim*4)

    def forward(self, x, state=None):
        # State: tuple(ssm_state, conv_buffer)
        rnn_state, buf = (None, None)
        if state is not None: 
            if isinstance(state, tuple):
                rnn_state, buf = state
            else:
                rnn_state = state

        shortcut = x
        x = self.norm(x)
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 1. Shift-SSM on K (Causal Local Context — left-padded conv)
        if buf is None:
            # Training / Fresh — pad 2 zeros on the left for causal kernel_size=3
            k_T = k.transpose(1, 2)
            k_T_padded = F.pad(k_T, (2, 0))  # (left=2, right=0)
            k_shift = self.shift_conv(k_T_padded).transpose(1, 2)
            # Create buffer for next step (last 2 tokens)
            new_buf = k[:, -2:, :]
        else:
            # Step — prepend buffer for causal context
            k_cat = torch.cat([buf, k], dim=1)
            k_shift = self.shift_conv(k_cat.transpose(1, 2)).transpose(1, 2)[:, -x.shape[1]:, :]
            new_buf = k_cat[:, -2:, :]

        # 2. Diagonal SSM on V * K_shifted
        x_ssm = v * k_shift
        
        # Parameters
        dt = F.softplus(self.dt_proj(x_ssm))
        A = -torch.exp(self.A_log.clamp(max=5.0))
        D_decay = torch.exp((A * dt).clamp(min=-30.0, max=0.0))  # bounded decay in (0, 1]
        
        # Scan: h_t = D_decay * h_{t-1} + x_ssm
        h = parallel_scan_linear(D_decay, x_ssm, rnn_state)
        
        # 3. Output Gating
        y = q * h
        y = self.out_proj(y)
        
        x = shortcut + y
        x = x + self.mlp(self.norm(x))
        
        return x, (h[:, -1, :], new_buf)

    def forward_seq(self, x, h0=None, state=None):
        # Alias for linegen compatibility
        st = state if state is not None else h0
        return self.forward(x, state=st)

class H3LM(nn.Module):
    def __init__(self, vocab_size, dim, depth):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([H3Block(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx, state=None):
        x = self.embed(idx)
        if state is None: state = [None]*len(self.blocks)
        new_states = []
        for blk, s in zip(self.blocks, state):
            x, ns = blk(x, s)
            new_states.append(ns)
        return self.head(x), new_states

# ==============================================================================
# 13. DCT-Former (Discrete Cosine Transform Mixing)
# ==============================================================================
import torch.fft

class DCTMixingBlock(nn.Module):
    """
    Causal frequency-aware token mixing block.
    
    Original used FFT which is fundamentally non-causal (every output depends on
    all inputs). This replacement uses a causal linear projection with DCT-inspired
    initialization, preserving the frequency-domain character while being strictly causal.
    
    The weight matrix W is masked with a lower-triangular (causal) mask, and 
    initialized from a truncated DCT basis so the model starts with frequency-like
    mixing patterns that respect causality.
    """
    def __init__(self, dim, seq_len, act_name="swiglu"):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.ln = nn.LayerNorm(dim)
        
        # Causal mixing weight: (seq_len, seq_len) masked lower-triangular
        # Initialize from DCT basis (truncated to causal)
        W = torch.zeros(seq_len, seq_len)
        for k in range(seq_len):
            for n in range(k + 1):  # only causal entries (n <= k)
                W[k, n] = math.cos(math.pi * (2*n + 1) * k / (2 * seq_len))
        # Normalize rows
        row_norms = W.norm(dim=1, keepdim=True).clamp(min=1e-6)
        W = W / row_norms * 0.02  # scale down for stable init
        self.mix_weight = nn.Parameter(W)
        
        # Learnable per-channel frequency scaling
        self.channel_scale = nn.Parameter(torch.ones(dim) * 0.1)
        
        # Register causal mask as buffer (not a parameter)
        self.register_buffer('causal_mask', torch.tril(torch.ones(seq_len, seq_len)))
        
        self.mlp = make_feed_forward(dim, dim * 4, act_name=act_name, bias=False)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        shortcut = x
        x = self.ln(x)
        
        # Apply causal mask to weight matrix, then mix tokens
        T_eff = min(T, self.seq_len)
        W = self.mix_weight[:T_eff, :T_eff] * self.causal_mask[:T_eff, :T_eff]
        
        # Token mixing: (B, T, D) -> einsum with (T, T) -> (B, T, D)
        # Scale per channel
        x_mix = torch.einsum('ij,bjd->bid', W, x[:, :T_eff, :]) * self.channel_scale.unsqueeze(0).unsqueeze(0)
        
        if T > self.seq_len:
            # For sequences longer than seq_len, pass through unmixed (rare edge case)
            x_out = torch.cat([x_mix, x[:, T_eff:, :] * 0.0], dim=1)
        else:
            x_out = x_mix
        
        x = shortcut + x_out
        x = x + self.mlp(self.ln(x))
        return x

class DCTFormerLM(nn.Module):
    def __init__(self, vocab_size, dim, depth, seq_len, act_name="swiglu"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([DCTMixingBlock(dim, seq_len, act_name=act_name) for _ in range(depth)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)



class ScanBlock_MinIndyGRU(nn.Module):
    """
    MinIndyGRU: minGRU + Independent Recurrent Scaling (IndRNN).
    Recurrence: h_t = (u * (1-z_t)) * h_{t-1} + z_t * h_tilde
    
    The 'u' parameter allows the model to scale the memory state independently 
    of the gate, enabling better gradient flow and long-term memory (u ~ 1.0) 
    or rapid flushing (u < 1.0) per channel.
    """
    def __init__(self, dim: int, log_space: bool = True):
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm(dim)
        self.Wz = nn.Linear(dim, dim) # Gate
        self.Wh = nn.Linear(dim, dim) # Candidate
        
        # A bounded independent recurrent gain.  Keep the existing parameter
        # name and zero initialization so old checkpoints still load sensibly;
        # the offset centers a zero value at 0.99 rather than an unsafe gain 1.
        self.u_log = nn.Parameter(torch.zeros(dim))
        self._u_logit_offset = math.log(0.99 / 0.01)
        
        self.post = GatedMLP(dim, mult=4, act_name="gelu")
        
        # Init
        nn.init.xavier_uniform_(self.Wz.weight); nn.init.zeros_(self.Wz.bias)
        nn.init.xavier_uniform_(self.Wh.weight); nn.init.zeros_(self.Wh.bias)

    def forward_seq(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        x_norm = self.ln(x)
        
        z_raw = self.Wz(x_norm)
        h_tilde_raw = self.Wh(x_norm)
        
        # Log-Coeffs: log(u * (1-z)) = log(u) + log(1-z)
        # log(1 - sigmoid(z)) = -softplus(z)
        log_u = F.logsigmoid(self.u_log + self._u_logit_offset)
        log_coeffs = log_u - F.softplus(z_raw)
        
        # Log-Values: log(z * h_tilde)
        log_z = -F.softplus(-z_raw)
        log_h_tilde = log_g_act(h_tilde_raw)
        log_values = log_z + log_h_tilde
        
        h_seq = heinsen_associative_scan_log(log_coeffs, log_values, h0)
        
        out = x + self.post(h_seq)
        return out, h_seq[:, -1, :]

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        x_tn = self.ln(x_t)
        
        z = torch.sigmoid(self.Wz(x_tn))
        h_tilde = g_act(self.Wh(x_tn))
        
        # Linear-space bounded independent recurrent gain.
        u = torch.sigmoid(self.u_log + self._u_logit_offset)
        
        # h_t = u * (1-z) * h_{t-1} + z * h_tilde
        h = (u * (1.0 - z)) * h_prev + z * h_tilde
        
        out = x_t + self.post(h)
        return out, h


class ScanBlock_MinIndyLSTM(nn.Module):
    """
    MinIndyLSTM: minLSTM + Independent Recurrent Scaling.
    Recurrence: h_t = (u * f'_t) * h_{t-1} + i'_t * h_tilde
    
    Allows the 'forgetting' mechanic to be scaled by a learnable static vector 'u'.
    """
    def __init__(self, dim: int, log_space: bool = True):
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm(dim)
        self.Wf = nn.Linear(dim, dim)
        self.Wi = nn.Linear(dim, dim)
        self.Wh = nn.Linear(dim, dim)
        
        # Bounded independent recurrent gain; see MinIndyGRU above.
        self.u_log = nn.Parameter(torch.zeros(dim))
        self._u_logit_offset = math.log(0.99 / 0.01)
        
        self.post = GatedMLP(dim, mult=4, act_name="gelu")
        
        nn.init.xavier_uniform_(self.Wf.weight); nn.init.zeros_(self.Wf.bias)
        nn.init.xavier_uniform_(self.Wi.weight); nn.init.zeros_(self.Wi.bias)
        nn.init.xavier_uniform_(self.Wh.weight); nn.init.zeros_(self.Wh.bias)
        with torch.no_grad(): self.Wf.bias.fill_(1.0)

    def forward_seq(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        x_norm = self.ln(x)
        
        f_raw = self.Wf(x_norm)
        i_raw = self.Wi(x_norm)
        h_tilde_raw = self.Wh(x_norm)

        # 1. Normalize Gates (minLSTM logic)
        log_f = -F.softplus(-f_raw)
        log_i = -F.softplus(-i_raw)
        log_denom = torch.logaddexp(log_f, log_i)
        
        log_f_prime = log_f - log_denom
        log_i_prime = log_i - log_denom
        
        # 2. Inject Independent Scaling 'u' into decay
        # log(a_t) = log(u) + log(f')
        log_u = F.logsigmoid(self.u_log + self._u_logit_offset)
        log_coeffs = log_u + log_f_prime
        
        # log(b_t) = log(i') + log(h_tilde)
        log_values = log_i_prime + log_g_act(h_tilde_raw)
        
        h_seq = heinsen_associative_scan_log(log_coeffs, log_values, h0)
        
        out = x + self.post(h_seq)
        return out, h_seq[:, -1, :]

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        x_tn = self.ln(x_t)
        
        f = torch.sigmoid(self.Wf(x_tn))
        i = torch.sigmoid(self.Wi(x_tn))
        h_tilde = g_act(self.Wh(x_tn))
        
        denom = f + i + 1e-8
        f_prime = f / denom
        i_prime = i / denom
        
        u = torch.sigmoid(self.u_log + self._u_logit_offset)
        
        # h = u * f' * h_prev + i' * h_tilde
        h = (u * f_prime) * h_prev + i_prime * h_tilde
        
        out = x_t + self.post(h)
        return out, h


# ========= Additional autoregressive model families =========
class AutoregressiveConvLM(nn.Module):
    """Causal convolutional LMs: gated WaveNet or masked PixelCNN-style stack."""
    def __init__(self, vocab_size, dim, depth, kind="wavenet"):
        super().__init__()
        self.kind = kind
        self.embed = nn.Embedding(vocab_size, dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(dim, 2 * dim if kind == "wavenet" else dim, 3,
                      dilation=2 ** (i % 8), padding=0)
            for i in range(depth)
        ])
        self.residual = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, vocab_size)

    def forward(self, idx):
        x = self.embed(idx).transpose(1, 2)
        for conv, residual in zip(self.convs, self.residual):
            # Explicit left padding is causal; no output can observe a future token.
            y = conv(F.pad(x, (conv.dilation[0] * 2, 0))).transpose(1, 2)
            if self.kind == "wavenet":
                a, b = y.chunk(2, -1); y = torch.tanh(a) * torch.sigmoid(b)
            else:
                y = F.gelu(y)
            x = x + residual(y).transpose(1, 2)
        return self.head(self.norm(x.transpose(1, 2)))


class CausalFFTDepthwiseConv(nn.Module):
    """Learned depthwise causal convolution evaluated as a linear FFT convolution.

    The first filter coefficient is the current-token coefficient.  Cropping
    the linear convolution to ``T`` therefore preserves strict causality.
    """
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be at least one")
        self.max_seq_len = int(max_seq_len)
        self.filter = nn.Parameter(torch.randn(dim, self.max_seq_len) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("causal convolution expects [batch, time, feature]")
        _, time, _ = x.shape
        if time > self.max_seq_len:
            raise ValueError(f"context length {time} exceeds configured maximum {self.max_seq_len}")
        fft_size = 1 << (2 * time - 1).bit_length()
        signal = torch.fft.rfft(x.transpose(1, 2), n=fft_size)
        kernel = torch.fft.rfft(self.filter[:, :time], n=fft_size).unsqueeze(0)
        return torch.fft.irfft(signal * kernel, n=fft_size)[..., :time].transpose(1, 2)


class _CausalLongConvBlock(nn.Module):
    """Hyena-style gated causal long-convolution residual block."""
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, 3 * dim)
        self.long_conv = CausalFFTDepthwiseConv(dim, max_seq_len)
        self.out_proj = nn.Linear(dim, dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate, skip = self.in_proj(self.norm(x)).chunk(3, dim=-1)
        x = x + self.out_proj(self.long_conv(value * torch.sigmoid(gate)) * torch.sigmoid(skip))
        return x + self.ff(self.ff_norm(x))


class HyenaLM(nn.Module):
    """Attention-free, non-recurrent causal long-convolution language model."""
    def __init__(self, vocab_size: int, dim: int, depth: int, max_seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([_CausalLongConvBlock(dim, max_seq_len) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


class _CausalConvNeXtBlock(nn.Module):
    """ConvNeXt-style token block with explicit left-only depthwise padding."""
    def __init__(self, dim: int, kernel_size: int = 7):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("ConvNeXt kernel_size must be a positive odd integer")
        self.kernel_size = kernel_size
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=True)
        self.pointwise = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.layer_scale = nn.Parameter(torch.full((dim,), 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x).transpose(1, 2)
        y = self.depthwise(F.pad(y, (self.kernel_size - 1, 0))).transpose(1, 2)
        return x + self.layer_scale * self.pointwise(y)


class CausalConvNeXtLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int, depth: int, kernel_size: int = 7):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([_CausalConvNeXtBlock(dim, kernel_size) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


class _ToeplitzMixerBlock(nn.Module):
    """Causal lower-triangular Toeplitz token mixer plus pointwise MLP."""
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mixer = CausalFFTDepthwiseConv(dim, max_seq_len)
        self.mix_scale = nn.Parameter(torch.full((dim,), 1e-3))
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mix_scale * self.mixer(self.norm(x))
        return x + self.ff(self.ff_norm(x))


class ToeplitzMLPMixerLM(nn.Module):
    """Experimental global causal Toeplitz mixer implemented with FFTs."""
    def __init__(self, vocab_size: int, dim: int, depth: int, max_seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([_ToeplitzMixerBlock(dim, max_seq_len) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


class _GrassmannMixerBlock(nn.Module):
    """Experimental causal local pair mixer using rank-2 wedge features."""
    def __init__(self, dim: int, rank: int = 8, windows=(1, 2, 4, 8)):
        super().__init__()
        self.windows = tuple(windows)
        self.rank = rank
        self.norm = nn.LayerNorm(dim)
        self.pair = nn.Linear(dim, 4 * rank)
        self.out = nn.Linear(rank * len(self.windows), dim)
        self.gate = nn.Linear(dim, dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        u, v = self.pair(z).chunk(2, dim=-1)
        u, v = u.view(*u.shape[:-1], self.rank, 2), v.view(*v.shape[:-1], self.rank, 2)
        wedges = []
        for window in self.windows:
            previous = F.pad(v, (0, 0, 0, 0, window, 0))[:, :v.size(1)]
            wedges.append(u[..., 0] * previous[..., 1] - u[..., 1] * previous[..., 0])
        mixed = self.out(torch.cat(wedges, dim=-1))
        x = x + torch.sigmoid(self.gate(z)) * mixed
        return x + self.ff(self.ff_norm(x))


class GrassmannMixerLM(nn.Module):
    """Prototype causal Grassmann-flow-inspired local mixer; not a scan/RNN."""
    def __init__(self, vocab_size: int, dim: int, depth: int):
        super().__init__()
        rank = max(2, min(16, dim // 4))
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([_GrassmannMixerBlock(dim, rank=rank) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


class NeuralNGramLM(nn.Module):
    """Flat-context neural n-gram LM; context is [batch, time, n], never nested."""
    MAX_CONTEXT = 32

    def __init__(self, vocab_size: int, dim: int, depth: int, context_size: int = 4):
        super().__init__()
        if not 1 <= context_size <= self.MAX_CONTEXT:
            raise ValueError(f"n-gram context must be between 1 and {self.MAX_CONTEXT}")
        self.context_size = int(context_size)
        self.pad_id = vocab_size
        self.embed = nn.Embedding(vocab_size + 1, dim)
        layers = [nn.Linear(dim * self.context_size, 4 * dim), nn.GELU()]
        for _ in range(max(0, depth - 1)):
            layers.extend((nn.Linear(4 * dim, 4 * dim), nn.GELU()))
        layers.append(nn.Linear(4 * dim, vocab_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 2:
            raise ValueError("n-gram model expects [batch, time] token indices")
        padded = F.pad(idx, (self.context_size - 1, 0), value=self.pad_id)
        context = padded.unfold(1, self.context_size, 1)
        return self.mlp(self.embed(context).flatten(start_dim=2))


class MarkovBigramLM(nn.Module):
    """Trainable first-order Markov baseline: current token -> next-token logits."""
    def __init__(self, vocab_size: int):
        super().__init__()
        self.transitions = nn.Embedding(vocab_size, vocab_size)
        nn.init.zeros_(self.transitions.weight)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.transitions(idx)


class MaskedAutoregressiveMLP(nn.Module):
    """NADE/MADE-style causal prefix mixer; weights are lower triangular."""
    def __init__(self, vocab_size, dim, depth, seq_len, kind="made"):
        super().__init__()
        self.seq_len, self.kind = seq_len, kind
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.mix = nn.ParameterList([nn.Parameter(torch.randn(seq_len, seq_len) * .02) for _ in range(depth)])
        self.ff = nn.ModuleList([nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        if T > self.seq_len: raise ValueError("NADE/MADE context exceeds configured seq_len")
        x = self.embed(idx) + self.pos[:, :T]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=x.dtype), diagonal=-1)
        for mix, ff in zip(self.mix, self.ff):
            w = mix[:T, :T] * mask
            # NADE uses a normalized running prefix; MADE retains unconstrained masked weights.
            if self.kind == "nade": w = w / w.abs().sum(-1, keepdim=True).clamp_min(1.)
            x = x + torch.einsum("ij,bjd->bid", w, x)
            x = x + ff(x)
        return self.head(self.norm(x))


class DiagonalSSMLM(nn.Module):
    """Reference S4/S4D/S5/DSS/LRU diagonal state-space language models."""
    def __init__(self, vocab_size, dim, depth, kind):
        super().__init__()
        self.kind, self.dim = kind, dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.in_proj = nn.ModuleList([nn.Linear(dim, 2 * dim) for _ in range(depth)])
        self.out_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
        self.log_decay = nn.ParameterList([nn.Parameter(torch.zeros(dim)) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, vocab_size)

    def forward(self, idx, state=None):
        x = self.embed(idx); B, T, _ = x.shape
        if state is None: state = [x.new_zeros(B, self.dim) for _ in self.in_proj]
        next_state = []
        for layer, (proj, out, log_decay) in enumerate(zip(self.in_proj, self.out_proj, self.log_decay)):
            st, ys = state[layer], []
            for token in x.unbind(1):
                drive, gate = proj(token).chunk(2, -1)
                decay = torch.exp(-F.softplus(log_decay))
                if self.kind == "s5":
                    decay = decay * torch.sigmoid(gate)
                elif self.kind == "dss":
                    drive = drive * torch.sigmoid(gate)
                elif self.kind == "lru":
                    # Stable complex-like rotation represented with a real phase gate.
                    drive = torch.tanh(drive) * torch.cos(gate)
                st = decay * st + (1. - decay) * torch.tanh(drive)
                ys.append(token + out(st))
            x = torch.stack(ys, 1); next_state.append(st)
        return self.head(self.norm(x)), next_state


class CausalMemoryLM(nn.Module):
    """Causal memory/retrieval variants with tensor-only recurrent memory.

    ``compressive`` pools older states, ``memorizing``/``knn`` retrieve from a
    bounded key-value store, and ``retro`` performs chunk-level cross attention
    over prior chunks.  They are self-contained alternatives, not external
    corpus retrieval systems.
    """
    def __init__(self, vocab_size, dim, depth, kind, memory_slots=64):
        super().__init__()
        self.kind, self.dim, self.memory_slots = kind, dim, memory_slots
        self.embed = nn.Embedding(vocab_size, dim)
        self.q = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
        self.kv = nn.ModuleList([nn.Linear(dim, 2 * dim) for _ in range(depth)])
        self.out = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, vocab_size)

    # The memory only ever holds this layer's input values, never its outputs,
    # so the step loop is a windowed attention in disguise.  ``parallel``
    # computes it over blocks of queries; the loop remains as the reference.
    parallel = True
    _QUERY_BLOCK = 128

    def _compressed_prefix(self, u):
        """C_0 = u_0, C_k = (C_{k-1} + u_k) / 2: the compressive slot after
        folding in u_0..u_k, for every k."""
        if u.size(1) < 2:
            return u
        tail = u[:, 1:]
        if kernels_available(u):
            rest = affine_scan(torch.full_like(tail, 0.5), 0.5 * tail, u[:, 0]).to(u.dtype)
        else:
            c, outs = u[:, 0], []
            for step in tail.unbind(1):
                c = 0.5 * (c + step)
                outs.append(c)
            rest = torch.stack(outs, 1)
        return torch.cat((u[:, :1], rest), 1)

    def _layer_parallel(self, x, memory, q_proj, kv_proj, out_proj):
        B, T, D = x.shape
        S, compressive, knn = self.memory_slots, self.kind == "compressive", self.kind == "knn"
        q = q_proj(x)
        _, v = kv_proj(x).chunk(2, -1)
        u = torch.cat((memory.to(v.dtype), v), 1)            # every value ever appended, oldest first
        m0, N = memory.size(1), memory.size(1) + T
        prefix = self._compressed_prefix(u) if compressive and N > S else None
        reads = []
        for q0 in range(0, T, self._QUERY_BLOCK):
            q1 = min(T, q0 + self._QUERY_BLOCK)
            qb = q[:, q0:q1]
            n = m0 + torch.arange(q0, q1, device=x.device)[:, None]   # values appended before each query
            lo, hi = max(0, m0 + q0 - S), m0 + q1 - 1
            keys = u[:, lo:hi]
            j = torch.arange(lo, hi, device=x.device)[None, :]
            if compressive:
                allowed = (j < n) & ((n <= S) | (j >= n - S + 1))
            else:
                allowed = (j < n) & (j >= n - S)
            scores = torch.einsum("bqd,bkd->bqk", qb, keys) / math.sqrt(D)
            scores = scores.masked_fill(~allowed, float("-inf"))
            if compressive and prefix is not None:
                full = (n[:, 0] > S)
                slot = prefix[:, (n[:, 0] - S).clamp_min(0)]                  # (B, Q, D)
                extra = ((qb * slot).sum(-1) / math.sqrt(D)).masked_fill(~full[None, :], float("-inf"))
                scores = torch.cat((extra.unsqueeze(-1), scores), -1)
            has_any = torch.isfinite(scores).any(-1, keepdim=True)
            if scores.size(-1) == 0:
                reads.append(qb.new_zeros(qb.shape))
                continue
            if knn:
                pick = scores.argmax(-1)                                        # first maximum, oldest first
                read = torch.gather(keys, 1, pick.unsqueeze(-1).expand(-1, -1, D))
            else:
                weights = torch.where(has_any, scores, torch.zeros_like(scores)).softmax(-1)
                if compressive and prefix is not None:
                    read = weights[..., 0:1] * slot + torch.einsum("bqk,bkd->bqd", weights[..., 1:], keys)
                else:
                    read = torch.einsum("bqk,bkd->bqd", weights, keys)
            reads.append(torch.where(has_any, read, torch.zeros_like(read)))
        y = x + out_proj(torch.cat(reads, 1))
        if compressive and N > S:
            memory = torch.cat((prefix[:, N - S:N - S + 1], u[:, N - S + 1:]), 1)
        else:
            memory = u[:, max(0, N - S):]
        return y, memory

    def forward(self, idx, state=None):
        x = self.embed(idx); B, T, D = x.shape
        if state is None: state = [x.new_zeros(B, 0, D) for _ in self.q]
        if self.parallel:
            next_state = []
            for q_proj, kv_proj, out_proj, memory in zip(self.q, self.kv, self.out, state):
                x, memory = self._layer_parallel(x, memory, q_proj, kv_proj, out_proj)
                next_state.append(memory)
            return self.head(self.norm(x)), next_state
        next_state = []
        for q_proj, kv_proj, out_proj, memory in zip(self.q, self.kv, self.out, state):
            ys, additions = [], []
            for token in x.unbind(1):
                q = q_proj(token); k, v = kv_proj(token).chunk(2, -1)
                if memory.size(1):
                    scores = torch.einsum("bd,bmd->bm", q, memory) / math.sqrt(D)
                    if self.kind == "knn":
                        weights = F.one_hot(scores.argmax(-1), memory.size(1)).to(token.dtype)
                    else:
                        weights = scores.softmax(-1)
                    read = torch.einsum("bm,bmd->bd", weights, memory)
                else: read = torch.zeros_like(token)
                ys.append(token + out_proj(read)); additions.append(v)
                memory = torch.cat((memory, v.unsqueeze(1)), 1)
                if memory.size(1) > self.memory_slots:
                    if self.kind == "compressive":
                        memory = torch.cat(((memory[:, :2].mean(1, keepdim=True)), memory[:, 2:]), 1)
                    else: memory = memory[:, -self.memory_slots:]
            x = torch.stack(ys, 1); next_state.append(memory)
        return self.head(self.norm(x)), next_state


class SparseCausalTransformerLM(nn.Module):
    """Trainable fixed-context Longformer/BigBird causal sparse Transformers."""
    def __init__(self, vocab_size, dim, depth, heads, seq_len, kind):
        super().__init__()
        heads = next(h for h in (heads, 8, 4, 2, 1) if dim % h == 0)
        self.kind, self.seq_len = kind, seq_len
        self.window = min(64, max(1, seq_len))
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.empty(1, seq_len, dim))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(depth)])
        # Pre-norm keeps sparse-attention residual stacks trainable at depth.
        self.attn_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.ff_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)) for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        if T > self.seq_len: raise ValueError("sparse Transformer context exceeds configured seq_len")
        x = self.embed(idx) + self.pos[:, :T]
        q = torch.arange(T, device=x.device)[:, None]
        k = torch.arange(T, device=x.device)[None, :]
        # A query at q sees itself and the preceding `window - 1` positions only.
        allowed = (k <= q) & (k >= q - self.window + 1)
        if self.kind == "bigbird":
            # Every 16th *past key* is global.  This is causal even when a
            # global query attends, because future keys remain excluded.
            allowed = allowed | ((k % 16 == 0) & (k <= q))
        mask = ~allowed  # MultiheadAttention: True means "do not attend".
        for attn, attn_norm, ff_norm, ff in zip(self.blocks, self.attn_norm, self.ff_norm, self.ff):
            h = attn_norm(x)
            y, _ = attn(h, h, h, attn_mask=mask, need_weights=False)
            x = x + y
            x = x + ff(ff_norm(x))
        return self.head(self.norm(x))


class LatentRecurrentLM(nn.Module):
    """Prior-path VRNN/SRNN language model; CE training works without an ELBO."""
    def __init__(self, vocab_size, dim, depth, kind):
        super().__init__()
        self.kind, self.dim = kind, dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.cells = nn.ModuleList([nn.GRUCell(2 * dim, dim) for _ in range(depth)])
        self.prior = nn.ModuleList([nn.Linear(dim, 2 * dim) for _ in range(depth)])
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, idx, state=None):
        x = self.embed(idx); B, T, _ = x.shape
        if state is None: state = [x.new_zeros(B, self.dim) for _ in self.cells]
        if kernels_available(x):
            # Layer-major on the fused scan: layer n at step t needs only
            # layer n-1 at t and itself at t-1, so this equals the loop below.
            y, new_state = x, []
            for cell, prior, h0 in zip(self.cells, self.prior, state):
                w_ih = cell.weight_ih
                px = F.linear(y, w_ih[:, :self.dim], cell.bias_ih)
                y, h_last = latent_gru_scan(px, prior.weight, prior.bias, w_ih[:, self.dim:],
                                            cell.weight_hh, cell.bias_hh, h0, srnn=self.kind != "vrnn")
                y = y.to(x.dtype)
                new_state.append(h_last.to(h0.dtype))
            return self.out(y), new_state
        new_state, ys = list(state), []
        for token in x.unbind(1):
            y = token
            for n, (cell, prior) in enumerate(zip(self.cells, self.prior)):
                mu, log_scale = prior(new_state[n]).chunk(2, -1)
                # Use the prior mean at inference and training; a future ELBO can sample here.
                z = mu if self.kind == "vrnn" else torch.tanh(mu) * torch.sigmoid(-log_scale)
                new_state[n] = cell(torch.cat((y, z), -1), new_state[n]); y = new_state[n]
            ys.append(y)
        return self.out(torch.stack(ys, 1)), new_state


# ========= Compact implementations for the final model-zoo entries =========
class QRNNLM(nn.Module):
    """Causal QRNN with convolutional gates and f-pooling.

    This compact reference keeps QRNN's defining separation between parallel
    causal convolutions and lightweight recurrent pooling.  State is one cell
    tensor per layer, which also makes it compatible with LineGen TBPTT and
    incremental generation paths.
    """
    def __init__(self, vocab_size, dim, depth, kernel_size=2):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("QRNN kernel_size must be at least 1")
        self.dim, self.kernel_size = dim, int(kernel_size)
        self.embed = nn.Embedding(vocab_size, dim)
        self.gates = nn.ModuleList([
            nn.Conv1d(dim, 3 * dim, self.kernel_size, padding=0)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward_hidden(self, x, state=None):
        states = [None] * len(self.gates) if state is None else list(state)
        if len(states) != len(self.gates):
            raise ValueError("QRNN state must contain one (cell, input buffer) entry per layer")
        next_states = []
        k = self.kernel_size
        for gate_conv, previous in zip(self.gates, states):
            # State per layer: (cell, last k-1 layer inputs).  The causal
            # convolution needs those inputs to continue a stream exactly.
            if isinstance(previous, (tuple, list)):
                previous, buf = previous
            else:
                buf = None
            if buf is None:
                buf = x.new_zeros(x.size(0), k - 1, self.dim)
            conv_in = torch.cat((buf.to(x.dtype), x), dim=1)
            gates = gate_conv(conv_in.transpose(1, 2)).transpose(1, 2)
            z, forget, output = gates.chunk(3, dim=-1)
            z, forget, output = torch.tanh(z), torch.sigmoid(forget), torch.sigmoid(output)
            cell = x.new_zeros(x.size(0), self.dim) if previous is None else previous
            # f-pooling c_t = f c_{t-1} + (1 - f) z is linear in c: one affine scan.
            scan = affine_scan if kernels_available(forget) else pscan_linear_jit
            cells = scan(forget, (1.0 - forget) * z, cell.to(forget.dtype))
            x = output * cells
            next_states.append((cells[:, -1], conv_in[:, conv_in.size(1) - (k - 1):]))
        return self.norm(x), next_states

    def forward(self, idx, state=None):
        hidden, state = self.forward_hidden(self.embed(idx), state)
        return self.head(hidden), state


class SRULM(nn.Module):
    """Simple Recurrent Unit LM (Lei et al., 2018): per layer
    f = sigmoid(W_f x + v_f * c_{t-1} + b_f), r = sigmoid(W_r x + v_r * c_{t-1} + b_r),
    c = f c_{t-1} + (1 - f) W x, h = r c + (1 - r) alpha x with the paper's
    highway scaling alpha = sqrt(1 + 2 exp(b)) = sqrt(3) for a zero highway bias.
    State: one cell tensor per layer."""
    def __init__(self, vocab_size, dim, depth):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.projections = nn.ModuleList([nn.Linear(dim, 3 * dim) for _ in range(depth)])
        self.v_f = nn.ParameterList([nn.Parameter(torch.empty(dim).uniform_(-0.5, 0.5)) for _ in range(depth)])
        self.v_r = nn.ParameterList([nn.Parameter(torch.empty(dim).uniform_(-0.5, 0.5)) for _ in range(depth)])
        self.alpha = math.sqrt(3.0)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
    def forward_hidden(self, x, state=None):
        states = [None] * len(self.projections) if state is None else list(state)
        if len(states) != len(self.projections):
            raise ValueError("SRU state must contain one cell tensor per layer")
        next_states = []
        for projection, v_f, v_r, previous in zip(self.projections, self.v_f, self.v_r, states):
            u3 = projection(x)
            cell = x.new_zeros(x.size(0), self.dim) if previous is None else previous
            scan = sru_scan if kernels_available(u3) else sru_reference
            x, cell = scan(u3, x, v_f, v_r, cell, self.alpha)
            next_states.append(cell)
        return self.norm(x), next_states
    def forward(self, idx, state=None):
        hidden, state = self.forward_hidden(self.embed(idx), state)
        return self.head(hidden), state
class _SwitchFeedForward(nn.Module):
    """Top-1 routed expert bank; its auxiliary loss balances expert use."""
    def __init__(self, dim, n_experts, ff_mult=4, dropout=0.0):
        super().__init__()
        if n_experts < 1:
            raise ValueError("Switch MoE requires at least one expert")
        self.n_experts = int(n_experts)
        self.router = nn.Linear(dim, self.n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, ff_mult * dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(ff_mult * dim, dim),
            )
            for _ in range(self.n_experts)
        ])

    def forward(self, x):
        flat = x.reshape(-1, x.size(-1))
        probabilities = F.softmax(self.router(flat), dim=-1)
        weights, assignments = probabilities.max(dim=-1)
        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            chosen = assignments == expert_id
            if chosen.any():
                output[chosen] = expert(flat[chosen]) * weights[chosen].unsqueeze(-1)
        usage = F.one_hot(assignments, self.n_experts).to(probabilities.dtype).mean(dim=0)
        importance = probabilities.mean(dim=0)
        aux_loss = self.n_experts * torch.sum(usage * importance)
        return output.reshape_as(x), aux_loss


class _SwitchBlock(nn.Module):
    def __init__(self, dim, heads, n_experts, ff_mult, dropout):
        super().__init__()
        self.attn_norm, self.ff_norm = RMSNorm(dim), RMSNorm(dim)
        self.attn = ModernAttention(dim, heads, heads)
        self.experts = _SwitchFeedForward(dim, n_experts, ff_mult, dropout)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        update, aux_loss = self.experts(self.ff_norm(x))
        return x + update, aux_loss


class SwitchMoELM(nn.Module):
    """Causal Transformer with Switch-style top-1 feed-forward experts."""
    def __init__(self, vocab_size, dim, depth, heads, seq_len, n_experts=4, ff_mult=4, dropout=0.0):
        super().__init__()
        del seq_len  # Existing LineGen config compatibility; attention is length-flexible.
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            _SwitchBlock(dim, heads, n_experts, ff_mult, dropout) for _ in range(depth)
        ])
        self.norm, self.head = RMSNorm(dim), nn.Linear(dim, vocab_size, bias=False)
        self.aux_loss = None

    def forward(self, idx):
        x = self.embed(idx)
        aux_losses = []
        for block in self.blocks:
            x, aux_loss = block(x)
            aux_losses.append(aux_loss)
        self.aux_loss = torch.stack(aux_losses).mean() if aux_losses else x.new_zeros(())
        return self.head(self.norm(x))


class _JambaLiteAttention(nn.Module):
    """Causal attention with an append-only KV cache for chunked decoding."""
    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError("Jamba-lite dimension must be divisible by head count")
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, state=None):
        batch, time, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(tensor):
            return tensor.view(batch, time, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        old_k = None if state is None else state.get("k")
        old_v = None if state is None else state.get("v")
        key = k if old_k is None else torch.cat((old_k, k), dim=2)
        value = v if old_v is None else torch.cat((old_v, v), dim=2)
        previous = 0 if old_k is None else old_k.size(2)
        positions = torch.arange(time, device=x.device)
        key_positions = torch.arange(previous + time, device=x.device)
        mask = key_positions.unsqueeze(0) <= (previous + positions).unsqueeze(1)
        scores = torch.matmul(q, key.transpose(-2, -1)) * (self.head_dim ** -0.5)
        scores = scores.masked_fill(~mask.view(1, 1, time, previous + time), float("-inf"))
        output = torch.matmul(F.softmax(scores, dim=-1), value)
        output = output.transpose(1, 2).contiguous().view(batch, time, dim)
        return self.out(output), {"k": key, "v": value}


class JambaLiteLM(nn.Module):
    """Compact 1:3 causal attention/Mamba hybrid with continuation caches."""
    def __init__(self, vocab_size, dim, depth, heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            _JambaLiteAttention(dim, heads) if layer % 4 == 0 else ScanBlock_Mamba(dim)
            for layer in range(depth)
        ])
        self.norm, self.head = RMSNorm(dim), nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx, state=None):
        x = self.embed(idx)
        states = [None] * len(self.blocks) if state is None else list(state)
        if len(states) != len(self.blocks):
            raise ValueError("Jamba-lite state must contain one entry per layer")
        next_states = []
        for block, previous in zip(self.blocks, states):
            if isinstance(block, _JambaLiteAttention):
                update, next_state = block(self.norm(x), previous)
                x = x + update
            else:
                x, next_state = block.forward_seq(x, previous)
            next_states.append(next_state)
        return self.head(self.norm(x)), next_states


class HierarchicalSparseAttention(nn.Module):
    """Portable NSA-inspired causal attention with three sparse read paths.

    Query tokens read compressed summaries of completed blocks, raw tokens from
    their top-K scored completed blocks, and a dense local causal window.  The
    selection path is deliberately implemented with ordinary PyTorch gathers:
    it is a semantic/reference implementation, not a sparse-kernel benchmark.
    """
    def __init__(self, dim, heads, local_window, compression_block, selected_blocks,
                 max_seq_len=65536):
        super().__init__()
        if dim % heads:
            raise ValueError("Sparse Modern Transformer dimension must divide by head count")
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        if self.head_dim % 2:
            raise ValueError("Sparse Modern Transformer RoPE head dimension must be even")
        if min(local_window, compression_block, selected_blocks) < 1:
            raise ValueError("Sparse attention window, block size, and selected blocks must be positive")
        self.local_window = int(local_window)
        self.compression_block = int(compression_block)
        self.selected_blocks = int(selected_blocks)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        # One KV head (MQA) keeps the cache and sparse gathers compact.
        self.kv_proj = nn.Linear(dim, 2 * self.head_dim, bias=False)
        self.compress_key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.compress_value = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.out_proj = nn.Linear(3 * dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
        self.last_selected_blocks = None

    def _attend(self, query, keys, values):
        """Exact attention over the supplied, already-causal key set."""
        scores = torch.einsum("bhd,bhkd->bhk", query, keys) * (self.head_dim ** -0.5)
        return torch.einsum("bhk,bhkd->bhd", F.softmax(scores, dim=-1), values)

    def forward(self, x):
        batch, time, _ = x.shape
        query = self.q_proj(x).view(batch, time, self.heads, self.head_dim)
        key, value = self.kv_proj(x).chunk(2, dim=-1)
        key = key.view(batch, time, 1, self.head_dim)
        value = value.view(batch, time, 1, self.head_dim)
        cos, sin = self.rope(query, time)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2).expand(-1, self.heads, -1, -1)
        value = value.transpose(1, 2).expand(-1, self.heads, -1, -1)

        outputs = []
        self.last_selected_blocks = None
        block_size = self.compression_block
        for position in range(time):
            q_t = query[:, :, position, :]
            # Only blocks wholly before the query's current block can be
            # summarized/selected, so neither path can see a future token.
            complete_blocks = position // block_size
            compressed = q_t.new_zeros(batch, self.heads, self.head_dim)
            selected = q_t.new_zeros(batch, self.heads, self.head_dim)
            if complete_blocks:
                raw_keys = key[:, :, :complete_blocks * block_size, :]
                raw_values = value[:, :, :complete_blocks * block_size, :]
                key_blocks = raw_keys.reshape(batch, self.heads, complete_blocks, block_size, self.head_dim)
                value_blocks = raw_values.reshape(batch, self.heads, complete_blocks, block_size, self.head_dim)
                summary_keys = self.compress_key(key_blocks.mean(dim=3))
                summary_values = self.compress_value(value_blocks.mean(dim=3))
                compressed = self._attend(q_t, summary_keys, summary_values)

                route_scores = torch.einsum("bhd,bhnd->bhn", q_t, summary_keys)
                top_k = min(self.selected_blocks, complete_blocks)
                selected_indices = route_scores.topk(top_k, dim=-1).indices
                gathered_keys = torch.gather(
                    key_blocks, 2,
                    selected_indices[..., None, None].expand(-1, -1, -1, block_size, self.head_dim),
                ).flatten(2, 3)
                gathered_values = torch.gather(
                    value_blocks, 2,
                    selected_indices[..., None, None].expand(-1, -1, -1, block_size, self.head_dim),
                ).flatten(2, 3)
                selected = self._attend(q_t, gathered_keys, gathered_values)
                self.last_selected_blocks = selected_indices.detach()

            local_start = max(0, position - self.local_window + 1)
            local = self._attend(
                q_t,
                key[:, :, local_start:position + 1, :],
                value[:, :, local_start:position + 1, :],
            )
            outputs.append(torch.cat((compressed, selected, local), dim=-1).unsqueeze(2))
        output = torch.cat(outputs, dim=2).transpose(1, 2).reshape(batch, time, 3 * self.dim)
        return self.out_proj(output)


class SparseModernBlock(nn.Module):
    def __init__(self, dim, heads, local_window, compression_block, selected_blocks,
                 max_seq_len, act_name="swiglu"):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = HierarchicalSparseAttention(
            dim, heads, local_window, compression_block, selected_blocks, max_seq_len
        )
        self.ff_norm = RMSNorm(dim)
        self.ff = make_feed_forward(dim, int(dim * 2.68), act_name=act_name, bias=False)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.ff(self.ff_norm(x))


class SparseModernTransformerLM(nn.Module):
    """Modern causal decoder with portable hierarchical sparse attention.

    It follows NSA's compressed/selected/local hierarchy but intentionally does
    not claim the hardware speedups of DSA/FlashMLA sparse kernels.
    """
    def __init__(self, vocab_size, dim, depth, heads, seq_len, local_window=512,
                 compression_block=32, selected_blocks=16, act_name="swiglu"):
        super().__init__()
        max_seq_len = max(1, int(seq_len))
        self.seq_len = max_seq_len
        local_window = min(int(local_window), max_seq_len)
        compression_block = min(int(compression_block), max_seq_len)
        if min(local_window, compression_block, int(selected_blocks)) < 1:
            raise ValueError("Sparse Modern Transformer settings must be positive")
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            SparseModernBlock(
                dim, heads, local_window, compression_block, int(selected_blocks),
                max_seq_len, act_name,
            )
            for _ in range(depth)
        ])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx):
        if idx.size(1) > self.seq_len:
            raise ValueError("Sparse Modern Transformer context exceeds configured seq_len")
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


# ==============================================================================
# Custom recurrent cells (kernel-backed) and the depth-wise layer stack
# ==============================================================================
def _pack_layer_states(states):
    """Stack per-layer states (tensors (1,B,H)/(B,H) or tuples of them) into
    (L, B, H) tensors, the CustomRNNWrapper state convention."""
    first = states[0]
    as3 = lambda t: t if t.dim() == 3 else t.unsqueeze(0)
    if isinstance(first, (tuple, list)):
        return tuple(torch.cat([as3(s[k]) for s in states], 0) for k in range(len(first)))
    return torch.cat([as3(s) for s in states], 0)


def _layer_state(state, i):
    """Layer i's state from a packed (L,B,H) tensor / tuple, or a per-layer list."""
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(s[i] for s in state)
    return state[i]


class _LayeredCell(nn.Module):
    """Multi-layer recurrent cell.  Subclasses define ``_make_layer(in_size)``,
    ``_zero_state(x)`` (per-layer state for batch x) and
    ``_run_layer(layer, index, x, state) -> (y, state)``."""
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.input_size, self.hidden_size, self.num_layers = input_size, hidden_size, num_layers
        self.layers = nn.ModuleList([
            self._make_layer(input_size if i == 0 else hidden_size, i) for i in range(num_layers)
        ])

    def forward(self, x, state=None):
        outs = []
        for i, layer in enumerate(self.layers):
            st = _layer_state(state, i)
            if st is None:
                st = self._zero_state(x)
            x, st = self._run_layer(layer, i, x, st)
            outs.append(st)
        return x, _pack_layer_states(outs)


class IndyLSTM(_LayeredCell):
    """IndyLSTM (Gonnet & Deselaers, 2019): LSTM with diagonal recurrent
    weights.  ``relu_gates`` (experimental) replaces the sigmoid gates by ReLU.
    State per layer: (h, c)."""
    def __init__(self, input_size, hidden_size, num_layers=1, relu_gates=False):
        self.relu_gates = bool(relu_gates)
        super().__init__(input_size, hidden_size, num_layers)

    def _make_layer(self, in_size, index):
        layer = nn.Module()
        H = self.hidden_size
        layer.W = nn.Linear(in_size, 4 * H)
        layer.u = nn.Parameter(torch.empty(4 * H).uniform_(-1.0, 1.0))
        with torch.no_grad():
            layer.W.bias.zero_()
            layer.W.bias[H:2 * H].fill_(1.0)          # forget-gate bias
        return layer

    def _zero_state(self, x):
        z = x.new_zeros(x.size(0), self.hidden_size)
        return (z, z)

    def _run_layer(self, layer, index, x, state):
        gx = layer.W(x)
        scan = indylstm_scan if kernels_available(gx) else indylstm_reference
        y, c = scan(gx, layer.u, state[0], state[1], self.relu_gates)
        return y, (y[:, -1], c)


class IntersectionRNN(_LayeredCell):
    """Intersection RNN (+RNN; Collins et al., 2017): a coupled recurrent gate
    (GRU-like, tanh) and a coupled depth gate (highway, ReLU):
    y = g_y x + (1 - g_y) ReLU(.),  h = g_h h_{t-1} + (1 - g_h) tanh(.).
    The depth highway needs input width == hidden width."""
    def __init__(self, input_size, hidden_size, num_layers=1, gate_bias=1.0):
        if input_size != hidden_size:
            raise ValueError("Intersection RNN needs input width == hidden width (depth highway)")
        self.gate_bias = float(gate_bias)
        super().__init__(input_size, hidden_size, num_layers)

    def _make_layer(self, in_size, index):
        H = self.hidden_size
        layer = nn.Module()
        layer.Wx = nn.Linear(in_size, 4 * H)            # [y_in, h_in, g_y, g_h]
        layer.U = nn.Parameter(torch.empty(4 * H, H))
        nn.init.orthogonal_(layer.U)
        with torch.no_grad():
            layer.Wx.bias.zero_()
            layer.Wx.bias[2 * H:].fill_(self.gate_bias)
        return layer

    def _zero_state(self, x):
        return x.new_zeros(x.size(0), self.hidden_size)

    def _run_layer(self, layer, index, x, h):
        px = layer.Wx(x)
        scan = irnn_scan if kernels_available(px) else irnn_reference
        return scan(px, x, layer.U, h)


class UGRNN(_LayeredCell):
    """UGRNN (update-gate RNN; Collins, Sohl-Dickstein & Sussillo, 2017):
    c = tanh(W_c x + U_c h_{t-1} + b_c), g = sigmoid(W_g x + U_g h_{t-1} + b_g),
    h = g h_{t-1} + (1 - g) c.  The +RNN's recurrent half without the depth
    gate; initialised like IntersectionRNN (orthogonal U, gate bias 1)."""
    def __init__(self, input_size, hidden_size, num_layers=1, gate_bias=1.0):
        self.gate_bias = float(gate_bias)
        super().__init__(input_size, hidden_size, num_layers)

    def _make_layer(self, in_size, index):
        H = self.hidden_size
        layer = nn.Module()
        layer.Wx = nn.Linear(in_size, 2 * H)            # [c_in, g_in]
        layer.U = nn.Parameter(torch.empty(2 * H, H))
        nn.init.orthogonal_(layer.U)
        with torch.no_grad():
            layer.Wx.bias.zero_()
            layer.Wx.bias[H:].fill_(self.gate_bias)
        return layer

    def _zero_state(self, x):
        return x.new_zeros(x.size(0), self.hidden_size)

    def _run_layer(self, layer, index, x, h):
        px = layer.Wx(x)
        scan = ugrnn_scan if kernels_available(px) else ugrnn_reference
        y = scan(px, layer.U, h)
        return y, y[:, -1]


class UnICORNN(_LayeredCell):
    """UnICORNN (Rusch & Mishra, 2021): undamped independent oscillators,
    symplectic Euler, per-neuron time step dt * sigmoid(c).  Follows the
    official implementation (w ~ U(0, 1), c ~ U(-0.1, 0.1)).  State: (y, z)."""
    def __init__(self, input_size, hidden_size, num_layers=1, dt=0.1, alpha=10.0):
        self.dt, self.alpha = float(dt), float(alpha)
        super().__init__(input_size, hidden_size, num_layers)

    def _make_layer(self, in_size, index):
        layer = nn.Module()
        layer.V = nn.Linear(in_size, self.hidden_size)
        layer.w = nn.Parameter(torch.empty(self.hidden_size).uniform_(0.0, 1.0))
        layer.c = nn.Parameter(torch.empty(self.hidden_size).uniform_(-0.1, 0.1))
        return layer

    def _zero_state(self, x):
        z = x.new_zeros(x.size(0), self.hidden_size)
        return (z, z)

    def _run_layer(self, layer, index, x, state):
        vx = layer.V(x)
        step = self.dt * torch.sigmoid(layer.c)
        scan = unicornn_scan if kernels_available(vx) else unicornn_reference
        y, z = scan(vx, layer.w, step, state[0], state[1], self.alpha)
        return y, (y[:, -1], z)


class LightRecurrentUnit(_LayeredCell):
    """Light Recurrent Unit (Electronics 2024, 13, 3204):
    h~ = tanh(W_h x), f = sigmoid(U_f h_{t-1} + W_f x + b_f), h = (1-f) h + f h~.
    ``highway``: the paper's stacked-cell variant, h~ = x for layers >= 2."""
    def __init__(self, input_size, hidden_size, num_layers=1, highway=False):
        self.highway = bool(highway)
        super().__init__(input_size, hidden_size, num_layers)

    def _make_layer(self, in_size, index):
        H = self.hidden_size
        layer = nn.Module()
        layer.W_h = None if (self.highway and index > 0) else nn.Linear(in_size, H, bias=False)
        layer.W_f = nn.Linear(in_size, H)
        layer.U_f = nn.Linear(H, H, bias=False)
        return layer

    def _zero_state(self, x):
        return x.new_zeros(x.size(0), self.hidden_size)

    def _run_layer(self, layer, index, x, h):
        cand = x if layer.W_h is None else torch.tanh(layer.W_h(x))
        pf = layer.W_f(x)
        scan = lru_scan if kernels_available(pf) else lru_reference
        y = scan(pf, cand, layer.U_f.weight, h)
        return y, y[:, -1]


class RRU(_LayeredCell):
    """Residual Recurrent Unit (Zakovskis et al., 2021), following the official
    RRUCell: d = Dropout(ReLU(RMSNorm(W [x; h] + b_j))), [c; o] = W d + b,
    h = sigmoid(S) h + Z c (S with a 10x effective learning rate, Z zero-init),
    output o.  Middle width = round(multiplier * (input + hidden))."""
    def __init__(self, input_size, hidden_size, num_layers=1, middle_multiplier=2.0, dropout=0.0):
        self.middle_multiplier, self.cell_dropout = float(middle_multiplier), float(dropout)
        super().__init__(input_size, hidden_size, num_layers)

    def _make_layer(self, in_size, index):
        H = self.hidden_size
        G = max(1, round(self.middle_multiplier * (in_size + H)))
        layer = nn.Module()
        layer.J_x = nn.Linear(in_size, G)
        layer.J_h = nn.Linear(H, G, bias=False)
        layer.C = nn.Linear(G, H)
        layer.O = nn.Linear(G, H)
        for lin in (layer.J_x, layer.J_h, layer.C, layer.O):
            nn.init.xavier_uniform_(lin.weight)          # TF glorot default
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        s = torch.empty(H).uniform_(0.01, 0.99)
        layer.S = nn.Parameter(torch.log(s / (1 - s)) / 10.0)
        layer.Z = nn.Parameter(torch.zeros(H))
        return layer

    def _zero_state(self, x):
        h0 = x.new_zeros(x.size(0), self.hidden_size)
        h0[:, 0] = 0.25 * math.sqrt(self.hidden_size)    # avoids normalising all-zero inputs
        return h0

    def _run_layer(self, layer, index, x, h):
        pxj = layer.J_x(x)
        B, T, G = pxj.shape
        if self.training and self.cell_dropout > 0:
            mask = (torch.rand(B, T, G, device=x.device) >= self.cell_dropout).float() / (1 - self.cell_dropout)
        else:
            mask = pxj.new_ones(B, T, G, dtype=torch.float32)
        scan = rru_scan if kernels_available(pxj) else rru_reference
        d, h_last = scan(pxj, layer.J_h.weight, layer.C.weight, layer.C.bias, mask,
                         torch.sigmoid(10.0 * layer.S), layer.Z, h)
        return layer.O(d), h_last


class MogrifierRNN(_LayeredCell):
    """Mogrifier LSTM / GRU (Melis et al., 2020): ``rounds`` of mutual gating
    x <- 2 sigmoid(Q h) x, h <- 2 sigmoid(R x) h before each LSTM / GRU step.
    Full-rank Q, R.  State: (h, c) for the LSTM, h for the GRU."""
    def __init__(self, input_size, hidden_size, num_layers=1, rounds=5, cell="lstm"):
        if input_size != hidden_size:
            raise ValueError("Mogrifier cells here need input width == hidden width")
        self.rounds, self.cell = int(rounds), cell
        super().__init__(input_size, hidden_size, num_layers)

    def _make_layer(self, in_size, index):
        H = self.hidden_size
        G = (4 if self.cell == "lstm" else 3) * H
        k = H ** -0.5
        u = lambda *shape: nn.Parameter(torch.empty(*shape).uniform_(-k, k))
        layer = nn.Module()
        layer.Q = u((self.rounds + 1) // 2, H, H)
        layer.R = u(self.rounds // 2, H, H)
        layer.W, layer.U, layer.b_i = u(G, H), u(G, H), u(G)
        layer.b_h = u(G) if self.cell == "gru" else None
        return layer

    def _zero_state(self, x):
        z = x.new_zeros(x.size(0), self.hidden_size)
        return (z, z) if self.cell == "lstm" else z

    def _run_layer(self, layer, index, x, state):
        h, c = state if self.cell == "lstm" else (state, torch.zeros_like(state))
        scan = mogrifier_scan if kernels_available(x) else mogrifier_reference
        y, c = scan(x, layer.Q, layer.R, layer.W, layer.U, layer.b_i, layer.b_h, h, c, self.rounds, self.cell)
        return y, ((y[:, -1], c) if self.cell == "lstm" else y[:, -1])


class ExpRNN(_LayeredCell):
    """expRNN (Lezcano-Casado & Martinez-Rubio, 2019): orthogonal recurrent
    matrix W = expm(A - A^T), modReLU activation, Henaff initialization
    (2x2 rotation blocks with angles ~ U(-pi, pi))."""
    def _make_layer(self, in_size, index):
        H = self.hidden_size
        layer = nn.Module()
        layer.V = nn.Linear(in_size, H, bias=False)
        nn.init.kaiming_normal_(layer.V.weight, nonlinearity="relu")
        A = torch.zeros(H, H)
        idx = torch.arange(0, H - 1, 2)
        A[idx, idx + 1] = torch.empty(len(idx)).uniform_(-math.pi, math.pi) / 2
        layer.A = nn.Parameter(A)
        layer.b = nn.Parameter(torch.empty(H).uniform_(-0.01, 0.01))
        return layer

    def _zero_state(self, x):
        return x.new_zeros(x.size(0), self.hidden_size)

    def _run_layer(self, layer, index, x, h):
        W = torch.linalg.matrix_exp(layer.A - layer.A.t())
        px = layer.V(x)
        scan = exprnn_scan if kernels_available(px) else exprnn_reference
        y = scan(px, W, layer.b, h)
        return y, y[:, -1]


class _PostLayerFFN(nn.Module):
    """Pre-norm residual feed-forward block after a recurrent layer:
    kind 1 SwiGLU, 2 ReGLU, 3 SiLU MLP."""
    def __init__(self, dim, kind):
        super().__init__()
        self.kind = int(kind)
        self.norm = nn.LayerNorm(dim)
        hidden = int(8 * dim / 3) if self.kind in (1, 2) else 4 * dim
        self.up = nn.Linear(dim, hidden, bias=False)
        self.gate = nn.Linear(dim, hidden, bias=False) if self.kind in (1, 2) else None
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        y = self.norm(x)
        if self.kind == 1:
            y = F.silu(self.gate(y)) * self.up(y)
        elif self.kind == 2:
            y = F.relu(self.gate(y)) * self.up(y)
        else:
            y = F.silu(self.up(y))
        return x + self.down(y)


class DepthwiseRNNStack(nn.Module):
    """Single-layer recurrent cores with the built-in RNNs' depth-wise options
    (same order as BuiltinRNNWrapper: multiplier, norm from layer 2 on, core,
    then residual every N layers and dropout), plus an optional post-layer FFN
    (``ffn``: 0 off, 1 SwiGLU, 2 ReGLU, 3 SiLU).  State: packed (L, B, H)."""
    def __init__(self, make_core, hidden, num_layers, use_norm=0, res_every=0, res_type=0,
                 dropout=0.0, use_multiplier=0, ffn=0):
        super().__init__()
        self.hidden, self.num_layers = hidden, num_layers
        self.use_norm, self.res_every, self.res_type = int(use_norm), int(res_every), int(res_type)
        self.cores = nn.ModuleList([make_core() for _ in range(num_layers)])
        norm = {1: lambda: nn.BatchNorm1d(hidden), 2: lambda: nn.LayerNorm(hidden), 3: lambda: RMSNorm(hidden),
                4: TTanh, 5: lambda: ETTanh(hidden), 6: lambda: DyT(hidden)}.get(self.use_norm)
        self.norms = None if norm is None else nn.ModuleList([norm() for _ in range(num_layers - 1)])
        self.drops = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers - 1)]) if dropout > 0 else None
        hops = [i for i in range(num_layers - 1) if self.res_every > 0 and (i + 1) % self.res_every == 0]
        self.res_mixers = self.alphas = self.betas = None
        if hops and self.res_type in (0, 1):
            width = hidden if self.res_type == 0 else 2 * hidden
            self.res_mixers = nn.ModuleDict({str(i): nn.Linear(width, hidden) for i in hops})
        elif hops:
            shape = 1 if self.res_type == 2 else hidden
            self.alphas = nn.ParameterDict({str(i): nn.Parameter(torch.zeros(shape)) for i in hops})
            self.betas = nn.ParameterDict({str(i): nn.Parameter(torch.ones(shape)) for i in hops})
        self.multipliers = None if use_multiplier == 0 else nn.ParameterList([
            nn.Parameter(torch.ones(1 if use_multiplier == 1 else hidden)) for _ in range(num_layers)
        ])
        self.ffns = None if not ffn else nn.ModuleList([_PostLayerFFN(hidden, ffn) for _ in range(num_layers)])

    def _norm(self, i, y):
        if self.use_norm == 1:
            B, T, H = y.shape
            return self.norms[i](y.reshape(B * T, H)).view(B, T, H)
        return self.norms[i](y)

    def _residual(self, i, y_in, y_out):
        key = str(i)
        if self.res_mixers is not None and key in self.res_mixers:
            mix = self.res_mixers[key]
            return y_in + mix(y_out) if self.res_type == 0 else mix(torch.cat((y_out, y_in), -1))
        if self.alphas is not None and key in self.alphas:
            return y_out * self.alphas[key] + y_in * self.betas[key]
        return y_out

    def forward(self, x, state=None):
        outs = []
        y = x
        for i, core in enumerate(self.cores):
            if self.multipliers is not None:
                y = y * self.multipliers[i]
            y_in = y
            if i > 0 and self.norms is not None:
                y = self._norm(i - 1, y)
            st = _layer_state(state, i)
            if st is not None:
                st = tuple(s.unsqueeze(0) for s in st) if isinstance(st, tuple) else st.unsqueeze(0)
            y, st = core(y, st)
            if self.ffns is not None:
                y = self.ffns[i](y)
            if i < self.num_layers - 1:
                y = self._residual(i, y_in, y)
                if self.drops is not None:
                    y = self.drops[i](y)
            outs.append(st)
        return y, _pack_layer_states(outs)


class SRUppLM(nn.Module):
    """SRU++ (Lei, 2021): SRU layers whose recurrence input comes from a
    single-head causal attention sub-layer,
    Q = W_q x, K = W_k Q, V = W_v Q, A = softmax(Q K^T / sqrt(d')) V,
    U = W_o LayerNorm(Q + alpha A) (alpha zero-init), then the SRU recurrence
    with the highway on x.  State per layer: (c, keys, values) so attention can
    continue across windows (capped at ``max_cache`` positions)."""
    def __init__(self, vocab_size, dim, depth, attn_dim=None, max_cache=1024):
        super().__init__()
        d_att = attn_dim or max(16, dim // 4)
        self.dim, self.d_att, self.max_cache = dim, d_att, int(max_cache)
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            layer = nn.Module()
            layer.q = nn.Linear(dim, d_att, bias=False)
            layer.k = nn.Linear(d_att, d_att, bias=False)
            layer.v = nn.Linear(d_att, d_att, bias=False)
            layer.alpha = nn.Parameter(torch.zeros(1))
            layer.norm = nn.LayerNorm(d_att)
            layer.o = nn.Linear(d_att, 3 * dim)
            layer.v_f = nn.Parameter(torch.empty(dim).uniform_(-0.5, 0.5))
            layer.v_r = nn.Parameter(torch.empty(dim).uniform_(-0.5, 0.5))
            self.layers.append(layer)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.alpha_highway = math.sqrt(3.0)

    def forward_hidden(self, x, state=None):
        B, T, _ = x.shape
        new_state = []
        for i, layer in enumerate(self.layers):
            c0, k_prev, v_prev = (state[i] if state is not None and state[i] is not None
                                  else (x.new_zeros(B, self.dim), None, None))
            q = layer.q(x)
            k, v = layer.k(q), layer.v(q)
            if k_prev is not None and k_prev.size(1) > 0:
                k_all, v_all = torch.cat((k_prev.to(k.dtype), k), 1), torch.cat((v_prev.to(v.dtype), v), 1)
            else:
                k_all, v_all = k, v
            P = k_all.size(1) - T
            mask = torch.ones(T, P + T, dtype=torch.bool, device=x.device).tril(P)
            att = F.scaled_dot_product_attention(q.unsqueeze(1), k_all.unsqueeze(1), v_all.unsqueeze(1),
                                                 attn_mask=mask).squeeze(1)
            u3 = layer.o(layer.norm(q + layer.alpha * att))
            scan = sru_scan if kernels_available(u3) else sru_reference
            x, c = scan(u3, x, layer.v_f, layer.v_r, c0, self.alpha_highway)
            keep = self.max_cache
            new_state.append((c, k_all[:, -keep:], v_all[:, -keep:]))
        return self.norm(x), new_state

    def forward(self, idx, state=None):
        hidden, state = self.forward_hidden(self.embed(idx), state)
        return self.head(hidden), state


class Mamba3Mixer(nn.Module):
    """Mamba-3 SISO mixer (Lahoti et al., 2026), following state-spaces/mamba
    ``modules/mamba3.py``: one input projection to (z, x, B, C, dt, A, lambda,
    angles); data-dependent A = -heavy_tail(.) <= -A_floor; Delta =
    softplus(. + dt_bias); BC RMSNorm plus head-wise channel biases (init 1);
    data-dependent rotary on the first ``rope_fraction`` of the state (the
    complex-SSM "RoPE trick", angle += tanh(.) * pi * Delta); the
    exponential-trapezoidal recurrence; D skip; SiLU(z) output gate; no
    convolution.  State: (angle, S, k_prev, v_prev).

    ``mimo_rank`` R > 1 selects the MIMO variant (paper Section 3.3 and
    Appendix C; ``is_mimo`` in modules/mamba3.py): B and C gain a rank axis
    (shared across heads, biases per head and rank), and the per-head x, z
    and output are lifted to / reduced from R copies by learnable
    data-independent vectors mimo_x, mimo_z, mimo_o (init 1/R, 1, 1/R), so
        H_t = alpha H_{t-1} + beta B_{t-1} X_{t-1}^T + gamma B_t X_t^T,
        y = sum_r mimo_o_r * ((H_t^T C_{t,r} + D x_r) * SiLU(mimo_z_r z)).
    The state size is unchanged; decoding does R times the FLOPs."""
    def __init__(self, dim, d_state=64, expand=2, headdim=64, rope_fraction=0.5,
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, A_floor=1e-4, mimo_rank=1):
        super().__init__()
        self.d_inner = int(expand * dim)
        # Largest head width up to ``headdim`` dividing d_inner (unchanged
        # when d_inner is a multiple of headdim).
        self.headdim = math.gcd(headdim, self.d_inner)
        self.nheads = self.d_inner // self.headdim
        self.d_state, self.A_floor = d_state, A_floor
        self.mimo_rank = R = int(mimo_rank)
        rot = int(d_state * rope_fraction)
        rot -= rot % 2
        self.num_rope_angles = rot // 2
        self.in_proj = nn.Linear(dim, 2 * self.d_inner + 2 * d_state * R + 3 * self.nheads + self.num_rope_angles,
                                 bias=False)
        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        # SISO keeps its original (nheads, d_state) layout so saved checkpoints load.
        bias_shape = (self.nheads, d_state) if R == 1 else (self.nheads, R, d_state)
        self.B_bias = nn.Parameter(torch.ones(bias_shape))
        self.C_bias = nn.Parameter(torch.ones(bias_shape))
        self.B_norm = RMSNorm(d_state, eps=1e-5)
        self.C_norm = RMSNorm(d_state, eps=1e-5)
        self.D = nn.Parameter(torch.ones(self.nheads))
        if R > 1:
            self.mimo_x = nn.Parameter(torch.full((self.nheads, R, self.headdim), 1.0 / R))
            self.mimo_z = nn.Parameter(torch.ones(self.nheads, R, self.headdim))
            self.mimo_o = nn.Parameter(torch.full((self.nheads, R, self.headdim), 1.0 / R))
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def _rotate(self, t, cos, sin):
        pairs = t.view(*t.shape[:-1], -1, 2)
        a, b = pairs[..., 0], pairs[..., 1]
        pad = a.shape[-1] - cos.shape[-1]
        cos, sin = F.pad(cos, (0, pad), value=1.0), F.pad(sin, (0, pad), value=0.0)
        return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).view_as(t)

    def forward(self, u, state=None):
        Bsz, T, _ = u.shape
        H, P, N, R = self.nheads, self.headdim, self.d_state, self.mimo_rank
        ct = torch.promote_types(u.dtype, torch.float32)   # compute dtype (>= fp32)
        z, x, Bp, Cp, dd_dt, dd_A, trap, angles = torch.split(
            self.in_proj(u), [self.d_inner, self.d_inner, N * R, N * R, H, H, H, self.num_rope_angles], dim=-1)
        a = dd_A.to(ct)
        A = -(a.clamp_min(0) + torch.reciprocal(1 - a.clamp_max(0)))     # heavy-tail activation
        A = A.clamp(max=-self.A_floor)
        dt = F.softplus(dd_dt.to(ct) + self.dt_bias)                     # (B, T, H)
        adt = (A * dt).transpose(1, 2)
        if state is None:
            rank = () if R == 1 else (R,)
            ang0 = u.new_zeros(Bsz, H, self.num_rope_angles, dtype=ct)
            s0 = u.new_zeros(Bsz, H, P, N, dtype=ct)
            k0 = u.new_zeros(Bsz, H, *rank, N, dtype=ct)
            v0 = u.new_zeros(Bsz, H, *rank, P, dtype=ct)
        else:
            ang0, s0, k0, v0 = state
        # Cumulative data-dependent rotation angle per head, kept in [0, 2 pi).
        inc = (torch.tanh(angles.to(ct)) * math.pi).unsqueeze(2) * dt.unsqueeze(-1)   # (B, T, H, R)
        ang = ang0.unsqueeze(1) + torch.cumsum(inc, 1)
        ang = ang - 2 * math.pi * torch.floor(ang / (2 * math.pi))
        cos, sin = torch.cos(ang), torch.sin(ang)
        if R > 1:
            # (B, T, H, R, N): ranks share each head's rotation angle.
            cos, sin = cos.unsqueeze(3), sin.unsqueeze(3)
            q = self._rotate(self.C_norm(Cp.to(ct).view(Bsz, T, R, N)).unsqueeze(2) + self.C_bias, cos, sin)
            k = self._rotate(self.B_norm(Bp.to(ct).view(Bsz, T, R, N)).unsqueeze(2) + self.B_bias, cos, sin)
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            v = (x.to(ct).view(Bsz, T, H, 1, P) * self.mimo_x).transpose(1, 2)        # (B, H, T, R, P)
            y, (S, k_last, v_last) = mamba3_chunkwise(q, k, v, k0, v0, adt, dt.transpose(1, 2),
                                                      torch.sigmoid(trap.to(ct)).transpose(1, 2), s0)
            y = (y + self.D[None, :, None, None, None] * v).transpose(1, 2)             # (B, T, H, R, P)
            y = y * F.silu(z.to(ct).view(Bsz, T, H, 1, P) * self.mimo_z)
            y = (y * self.mimo_o).sum(3).reshape(Bsz, T, self.d_inner)
            return self.out_proj(y.to(u.dtype)), (ang[:, -1], S, k_last, v_last)
        q = self._rotate(self.C_norm(Cp.to(ct)).unsqueeze(2) + self.C_bias, cos, sin).transpose(1, 2)
        k = self._rotate(self.B_norm(Bp.to(ct)).unsqueeze(2) + self.B_bias, cos, sin).transpose(1, 2)
        v = x.to(ct).view(Bsz, T, H, P).transpose(1, 2)
        y, (S, k_last, v_last) = mamba3_chunkwise(q, k, v, k0, v0, adt, dt.transpose(1, 2),
                                                  torch.sigmoid(trap.to(ct)).transpose(1, 2), s0)
        y = y + self.D[None, :, None, None] * v
        y = y.transpose(1, 2).reshape(Bsz, T, self.d_inner) * F.silu(z.to(ct))
        return self.out_proj(y.to(u.dtype)), (ang[:, -1], S, k_last, v_last)


class Mamba3LM(nn.Module):
    """Mamba-3 language model: Llama-style pre-norm residual blocks alternating
    a Mamba-3 mixer and a SwiGLU feed-forward (as in the paper, Section 3.4).
    ``mimo_rank`` > 1 builds Mamba-3 MIMO; as in the paper (Appendix C) the
    SwiGLU width shrinks so the model keeps the SISO parameter count."""
    def __init__(self, vocab_size, dim, depth, d_state=64, headdim=64, mimo_rank=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.mixers = nn.ModuleList([
            Mamba3Mixer(dim, d_state=d_state, headdim=headdim, mimo_rank=mimo_rank) for _ in range(depth)
        ])
        self.mixer_norms = nn.ModuleList([RMSNorm(dim, eps=1e-5) for _ in range(depth)])
        self.ffn_norms = nn.ModuleList([RMSNorm(dim, eps=1e-5) for _ in range(depth)])
        hidden = int(8 * dim / 3)
        if mimo_rank > 1:
            siso = Mamba3Mixer(dim, d_state=d_state, headdim=headdim)
            extra = sum(p.numel() for p in self.mixers[0].parameters()) - sum(p.numel() for p in siso.parameters())
            hidden = max(1, hidden - round(extra / (3 * dim)))
        self.ffn_gate = nn.ModuleList([nn.Linear(dim, hidden, bias=False) for _ in range(depth)])
        self.ffn_up = nn.ModuleList([nn.Linear(dim, hidden, bias=False) for _ in range(depth)])
        self.ffn_down = nn.ModuleList([nn.Linear(hidden, dim, bias=False) for _ in range(depth)])
        self.norm = RMSNorm(dim, eps=1e-5)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward_hidden(self, x, state=None):
        new_state = []
        for i, mixer in enumerate(self.mixers):
            y, st = mixer(self.mixer_norms[i](x), None if state is None else state[i])
            x = x + y
            h = self.ffn_norms[i](x)
            x = x + self.ffn_down[i](F.silu(self.ffn_gate[i](h)) * self.ffn_up[i](h))
            new_state.append(st)
        return self.norm(x), new_state

    def forward(self, idx, state=None):
        hidden, state = self.forward_hidden(self.embed(idx), state)
        return self.head(hidden), state


# ==============================================================================
# xLSTM (Beck et al., 2024) — faithful blocks following NX-AI/xlstm
# ==============================================================================
def _xl_small_init_(w, dim):
    nn.init.normal_(w, 0.0, math.sqrt(2 / (5 * dim)))


def _xl_wang_init_(w, dim, num_blocks):
    nn.init.normal_(w, 0.0, 2 / num_blocks / math.sqrt(dim))


def _xl_up_dim(dim, factor, multiple=64):
    return int(math.ceil(factor * dim / multiple) * multiple)


class _XLLayerNorm(nn.Module):
    """xLSTM LayerNorm: residual weight (1 + w, w zero-init), no bias."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.zeros(dim)), eps

    def forward(self, x):
        return F.layer_norm(x, (x.shape[-1],), 1.0 + self.weight, None, self.eps)


class _XLHeadNorm(_XLLayerNorm):
    """MultiHeadLayerNorm: group norm with one group per head on (B, S, NH*DH)."""
    def __init__(self, dim, num_heads, eps=1e-5):
        super().__init__(dim, eps)
        self.num_heads = num_heads

    def forward(self, x):
        B, S, D = x.shape
        return F.group_norm(x.reshape(B * S, D), self.num_heads, 1.0 + self.weight, None, self.eps).view(B, S, D)


class _XLHeadwise(nn.Module):
    """LinearHeadwiseExpand (block-diagonal projection, no bias)."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, dim // num_heads, dim // num_heads))
        nn.init.normal_(self.weight, 0.0, math.sqrt(2 / 5 / self.weight.shape[-1]))

    def forward(self, x):
        shape = x.shape
        y = torch.einsum("...hd,hod->...ho", x.view(*shape[:-1], self.num_heads, -1), self.weight)
        return y.reshape(shape)


class _XLCausalConv(nn.Module):
    """Depthwise causal conv with a carried input buffer (last k-1 inputs)."""
    def __init__(self, dim, kernel_size=4):
        super().__init__()
        self.k = kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=True)

    def forward(self, x, buf=None):
        if buf is None:
            buf = x.new_zeros(x.size(0), self.k - 1, x.size(-1))
        inp = torch.cat((buf.to(x.dtype), x), 1)
        return self.conv(inp.transpose(1, 2)).transpose(1, 2), inp[:, inp.size(1) - (self.k - 1):]


class XLmLSTMLayer(nn.Module):
    """mLSTM layer of the xLSTM mLSTM block (official mLSTMLayer + mLSTMCell)."""
    def __init__(self, dim, num_blocks, num_heads=4, proj_factor=2.0, conv_kernel=4, qkv_blocksize=4):
        super().__init__()
        inner = _xl_up_dim(dim, proj_factor)
        self.inner, self.num_heads = inner, num_heads
        self.proj_up = nn.Linear(dim, 2 * inner, bias=False)
        self.q_proj, self.k_proj, self.v_proj = (_XLHeadwise(inner, inner // qkv_blocksize) for _ in range(3))
        self.conv = _XLCausalConv(inner, conv_kernel)
        self.igate = nn.Linear(3 * inner, num_heads)
        self.fgate = nn.Linear(3 * inner, num_heads)
        self.outnorm = _XLHeadNorm(inner, num_heads)
        self.learnable_skip = nn.Parameter(torch.ones(inner))
        self.proj_down = nn.Linear(inner, dim, bias=False)
        _xl_small_init_(self.proj_up.weight, dim)
        _xl_wang_init_(self.proj_down.weight, dim, num_blocks)
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            _xl_small_init_(proj.weight, dim)
        nn.init.zeros_(self.fgate.weight)
        with torch.no_grad():
            self.fgate.bias.copy_(torch.linspace(3.0, 6.0, num_heads))
        nn.init.zeros_(self.igate.weight)
        nn.init.normal_(self.igate.bias, 0.0, 0.1)

    def forward(self, x, state=None):
        B, S, _ = x.shape
        NH, DH = self.num_heads, self.inner // self.num_heads
        buf, cell_state = (None, None) if state is None else state
        x_m, z = self.proj_up(x).split(self.inner, dim=-1)
        x_conv, buf = self.conv(x_m, buf)
        x_conv = F.silu(x_conv)
        q, k, v = self.q_proj(x_conv), self.k_proj(x_conv), self.v_proj(x_m)
        gates_in = torch.cat((q, k, v), -1)
        heads = lambda t: t.view(B, S, NH, DH).transpose(1, 2)
        h, cell_state = mlstm_chunkwise(
            heads(q), heads(k) / math.sqrt(DH), heads(v),
            self.igate(gates_in).transpose(1, 2), F.logsigmoid(self.fgate(gates_in)).transpose(1, 2),
            cell_state, eps=1e-6)
        h = self.outnorm(h.transpose(1, 2).reshape(B, S, self.inner).to(x.dtype))
        h = (h + self.learnable_skip * x_conv) * F.silu(z)
        return self.proj_down(h), (buf, cell_state)


class XLsLSTMLayer(nn.Module):
    """sLSTM layer of the xLSTM sLSTM block (official sLSTMLayer + vanilla
    sLSTM cell: log-sigmoid forget, stabilized exponential input gate,
    zero-init recurrent kernel, power-law block-dependent forget bias)."""
    def __init__(self, dim, block_idx, num_blocks, num_heads=4, conv_kernel=4):
        super().__init__()
        self.dim, self.num_heads = dim, num_heads
        DH = dim // num_heads
        self.conv = _XLCausalConv(dim, conv_kernel)
        self.fgate, self.igate, self.zgate, self.ogate = (_XLHeadwise(dim, num_heads) for _ in range(4))
        for g in (self.fgate, self.igate, self.zgate, self.ogate):
            _xl_small_init_(g.weight, dim)
        # Recurrent kernel in the fused-kernel layout: (gate i,f,z,o, head, out, in).
        self.recurrent = nn.Parameter(torch.zeros(4, num_heads, DH, DH))
        bias = torch.zeros(4, num_heads, DH)
        ratio = block_idx / (num_blocks - 1) if num_blocks > 1 else 0.0
        bias[1] = -(-5.0 + 12.0 * (torch.arange(DH) / max(1, DH - 1)) ** (0.3 + 1.3 * ratio))
        self.bias = nn.Parameter(bias)
        self.group_norm = _XLHeadNorm(dim, num_heads)

    def forward(self, x, state=None):
        B, S, D = x.shape
        buf, cell = (None, None) if state is None else state
        x_conv, buf = self.conv(x, buf)
        x_conv = F.silu(x_conv)
        # The official layer passes (fgate(x_conv), igate(x_conv), zgate(x), ogate(x))
        # into the cell's (i, f, z, o) slots; kept as-is so weights correspond.
        gin = torch.stack((self.fgate(x_conv), self.igate(x_conv), self.zgate(x), self.ogate(x)), dim=2)
        gin = gin + self.bias.reshape(4, D)
        if cell is None:
            zeros = x.new_zeros(B, D, dtype=torch.float32)
            cell = (zeros, zeros, zeros, zeros)
        h0, c0, n0, m0 = cell
        if kernels_available(gin):
            # The fused kernel orders gates (z, i, f, o); parameters stay in the
            # official (i, f, z, o) order.
            order = [2, 0, 1, 3]
            y, c, n, m = slstm_scan(gin[:, :, order], self.recurrent[order], h0, c0, n0, m0, exp_forget=False)
        else:
            y, c, n, m = _xl_slstm_reference(gin, self.recurrent, h0, c0, n0, m0)
        out = self.group_norm(y.to(x.dtype))
        return out, (buf, (y[:, -1], c, n, m))


def _xl_slstm_reference(gin, r, h, c, n, m):
    """Loop form of the official vanilla sLSTM (log-sigmoid forget)."""
    B, T, _, D = gin.shape
    NH, DH = r.shape[1], r.shape[2]
    ys = []
    for t in range(T):
        rec = torch.einsum("ghok,bhk->bgho", r, h.view(B, NH, DH)).reshape(B, 4, D)
        p_i, p_f, p_z, p_o = (gin[:, t] + rec).unbind(1)
        logf = F.logsigmoid(p_f)
        m_new = torch.maximum(logf + m, p_i)
        i_hat, f_hat = torch.exp(p_i - m_new), torch.exp(logf + m - m_new)
        c = f_hat * c + i_hat * torch.tanh(p_z)
        n = f_hat * n + i_hat
        h = torch.sigmoid(p_o) * c / n.clamp_min(1e-12)
        m = m_new
        ys.append(h)
    return torch.stack(ys, 1), c, n, m


class XLGatedFeedForward(nn.Module):
    """xLSTM gated feed-forward (GELU gate, proj factor 1.3, no bias)."""
    def __init__(self, dim, num_blocks, proj_factor=1.3):
        super().__init__()
        self.up_dim = _xl_up_dim(dim, proj_factor)
        self.proj_up = nn.Linear(dim, 2 * self.up_dim, bias=False)
        self.proj_down = nn.Linear(self.up_dim, dim, bias=False)
        _xl_small_init_(self.proj_up.weight, dim)
        _xl_wang_init_(self.proj_down.weight, dim, num_blocks)

    def forward(self, x):
        gate, up = self.proj_up(x).split(self.up_dim, -1)
        return self.proj_down(F.gelu(gate) * up)


class XLSTMFullLM(nn.Module):
    """xLSTM language model (Beck et al., 2024; NX-AI xlstm): pre-LayerNorm
    residual blocks, mLSTM blocks with sLSTM blocks at ``slstm_at`` (the
    sLSTM blocks carry a gated feed-forward), a post-blocks LayerNorm, and
    small-init embedding / head.  kind: "mix" (xLSTM[a:b]), "m" or "s"."""
    def __init__(self, vocab_size, dim, depth, num_heads=4, kind="mix", m_to_s=(7, 1)):
        super().__init__()
        if kind == "s":
            slstm_at = set(range(depth))
        elif kind == "m":
            slstm_at = set()
        else:
            a, b = m_to_s
            slstm_at = {i for i in range(depth) if 1 <= i % (a + b) <= b}
        heads = next(h for h in (num_heads, 4, 2, 1) if dim % h == 0)
        self.embed = nn.Embedding(vocab_size, dim)
        self.norms, self.layers, self.ffn_norms, self.ffns = (nn.ModuleList() for _ in range(4))
        for i in range(depth):
            self.norms.append(_XLLayerNorm(dim))
            if i in slstm_at:
                self.layers.append(XLsLSTMLayer(dim, i, depth, num_heads=heads))
                self.ffn_norms.append(_XLLayerNorm(dim))
                self.ffns.append(XLGatedFeedForward(dim, depth))
            else:
                self.layers.append(XLmLSTMLayer(dim, depth, num_heads=heads))
                self.ffn_norms.append(nn.Identity())
                self.ffns.append(None)
        self.post_norm = _XLLayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        _xl_small_init_(self.embed.weight, dim)
        _xl_small_init_(self.head.weight, dim)

    def forward_hidden(self, x, state=None):
        new_state = []
        for i, (norm, layer) in enumerate(zip(self.norms, self.layers)):
            y, st = layer(norm(x), None if state is None else state[i])
            x = x + y
            if self.ffns[i] is not None:
                x = x + self.ffns[i](self.ffn_norms[i](x))
            new_state.append(st)
        return self.post_norm(x), new_state

    def forward(self, idx, state=None):
        hidden, state = self.forward_hidden(self.embed(idx), state)
        return self.head(hidden), state


# ==============================================================================
# Linear-recurrence family (DeltaNet, Gated DeltaNet, HGRN2, Mamba-2, RetNet)
# following flash-linear-attention / state-spaces reference layers.  All run
# on the rwkv7_wkv state kernel: S = S diag(w) + (S a) b^T + v k^T, y = S r.
# ==============================================================================
def _state_op(r, w, k, v, a, b, S):
    """Run the shared state recurrence (kernel when shapes allow).  Head sizes
    that are not powers of two are zero-padded up to one for the kernel: padded
    key channels have r = k = a = b = 0 and padded value rows v = 0, so their
    state stays zero and the sliced result is exact."""
    H, P, N = S.shape[-3], S.shape[-2], S.shape[-1]
    r, w, k, v, a, b, S = (t.float().contiguous() for t in (r, w, k, v, a, b, S))
    if not kernels_available(r):
        return rwkv7_wkv_reference(r, w, k, v, a, b, S)
    P2, N2 = 1 << (P - 1).bit_length(), 1 << (N - 1).bit_length()
    if (P2, N2) == (P, N):
        return rwkv7_wkv(r, w, k, v, a, b, S)
    B, T = r.shape[:2]
    pad = lambda t, n, n2: F.pad(t.view(B, T, H, n), (0, n2 - n)).reshape(B, T, H * n2)
    r, w, k, a, b = (pad(t, N, N2) for t in (r, w, k, a, b))
    y, S = rwkv7_wkv(r, w, k, pad(v, P, P2), a, b, F.pad(S, (0, N2 - N, 0, P2 - P)))
    return y.view(B, T, H, P2)[..., :P].reshape(B, T, H * P), S[..., :P, :N]


class _ShortConv(nn.Module):
    """Depthwise causal conv + optional SiLU with a carried input buffer."""
    def __init__(self, dim, kernel_size=4, bias=False, act=True):
        super().__init__()
        self.k, self.act = kernel_size, act
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=bias)

    def forward(self, x, buf=None):
        if buf is None:
            buf = x.new_zeros(x.size(0), self.k - 1, x.size(-1))
        inp = torch.cat((buf.to(x.dtype), x), 1)
        y = self.conv(inp.transpose(1, 2)).transpose(1, 2)
        return (F.silu(y) if self.act else y), inp[:, inp.size(1) - (self.k - 1):]


def _heads(t, H):
    B, T, C = t.shape
    return t.view(B, T, H, C // H)


class DeltaNetMixer(nn.Module):
    """DeltaNet (Yang et al., 2024; fla ``DeltaNet``): SiLU q/k/v with short
    convs, L2-normalized q/k, sigmoid beta, delta-rule state
    S <- S - beta (S k) k^T + beta v k^T, o = S q / sqrt(d), per-head RMSNorm.
    ``gated`` gives Gated DeltaNet (Yang et al., 2025; fla ``GatedDeltaNet``):
    Mamba-2 style decay exp(-exp(A_log) softplus(a + dt_bias)) on S and a
    gated RMSNorm output with v heads 2x wider."""
    def __init__(self, dim, gated=False, num_heads=None, head_dim=64):
        super().__init__()
        self.gated = gated
        if gated:
            H = num_heads or max(1, round(0.75 * dim / head_dim))
            dk, dv = head_dim, 2 * head_dim
        else:
            H = num_heads or next(h for h in (4, 2, 1) if dim % h == 0)
            dk = dv = dim // H
        self.H, self.dk, self.dv = H, dk, dv
        self.q_proj = nn.Linear(dim, H * dk, bias=False)
        self.k_proj = nn.Linear(dim, H * dk, bias=False)
        self.v_proj = nn.Linear(dim, H * dv, bias=False)
        self.b_proj = nn.Linear(dim, H, bias=False)
        self.q_conv, self.k_conv, self.v_conv = _ShortConv(H * dk), _ShortConv(H * dk), _ShortConv(H * dv)
        if gated:
            self.a_proj = nn.Linear(dim, H, bias=False)
            self.A_log = nn.Parameter(torch.log(torch.empty(H).uniform_(0, 16)))
            dt = torch.exp(torch.rand(H) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
            self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
            self.g_proj = nn.Linear(dim, H * dv, bias=False)
        self.o_norm = RMSNorm(dv, eps=1e-5)
        self.o_proj = nn.Linear(H * dv, dim, bias=False)

    def forward(self, x, state=None):
        B, T, _ = x.shape
        H, dk, dv = self.H, self.dk, self.dv
        bq, bk, bv, S = (None, None, None, None) if state is None else state
        # fla DeltaNet applies SiLU inside the short conv; its q/k then pass
        # through the kernel's L2 norm.
        q, bq = self.q_conv(self.q_proj(x), bq)
        k, bk = self.k_conv(self.k_proj(x), bk)
        v, bv = self.v_conv(self.v_proj(x), bv)
        q = F.normalize(_heads(q, H).float(), dim=-1) * dk ** -0.5
        k = F.normalize(_heads(k, H).float(), dim=-1)
        beta = torch.sigmoid(self.b_proj(x).float())                              # (B, T, H)
        if self.gated:
            g = -torch.exp(self.A_log.float()) * F.softplus(self.a_proj(x).float() + self.dt_bias)
            decay = torch.exp(g)
        else:
            decay = torch.ones_like(beta)
        w = decay.unsqueeze(-1).expand(B, T, H, dk)
        # S <- S decay - beta (S decay k) k^T + beta v k^T
        a = k
        b = -(beta * decay).unsqueeze(-1) * k
        kk = beta.unsqueeze(-1) * k
        if S is None:
            S = x.new_zeros(B, H, dv, dk, dtype=torch.float32)
        y, S = _state_op(q.reshape(B, T, H * dk), w.reshape(B, T, H * dk), kk.reshape(B, T, H * dk),
                         v.float(), a.reshape(B, T, H * dk), b.reshape(B, T, H * dk), S)
        y = self.o_norm(y.view(B, T, H, dv))
        if self.gated:
            y = y * F.silu(self.g_proj(x).float().view(B, T, H, dv))
        return self.o_proj(y.reshape(B, T, H * dv).to(x.dtype)), (bq, bk, bv, S)


class HGRN2Mixer(nn.Module):
    """HGRN2 (Qin et al., 2024; fla ``HGRN2Attention``): q = swish(W_q x),
    forget g = logsigmoid(W_f x) with the layer's learned lower bound,
    key = 1 - exp(g), gated-linear-attention state S = S diag(exp(g)) + v k^T,
    o = S q / sqrt(d_f), then RMSNorm and the output projection."""
    def __init__(self, dim, expand_ratio=128):
        super().__init__()
        ratio = min(expand_ratio, dim)
        self.H = max(1, dim // ratio)
        self.df, self.di = ratio, dim // self.H
        self.q_proj = nn.Linear(dim, self.H * self.df, bias=False)
        self.f_proj = nn.Linear(dim, self.H * self.df, bias=False)
        self.i_proj = nn.Linear(dim, dim, bias=False)
        self.g_norm = RMSNorm(dim, eps=1e-5)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, state=None, lower_bound=None):
        B, T, _ = x.shape
        q = F.silu(self.q_proj(x).float())
        g = F.logsigmoid(self.f_proj(x).float())
        if lower_bound is not None:
            g = torch.logaddexp(lower_bound.log(), torch.log1p(-lower_bound) + g)
        k = 1 - g.exp()
        S = x.new_zeros(B, self.H, self.di, self.df, dtype=torch.float32) if state is None else state
        zeros = torch.zeros_like(k)
        y, S = _state_op(q * self.df ** -0.5, g.exp(), k, self.i_proj(x).float(), zeros, zeros, S)
        return self.o_proj(self.g_norm(y).to(x.dtype)), S


class Mamba2Mixer(nn.Module):
    """Mamba-2 (Dao & Gu, 2024; fla / state-spaces ``Mamba2``): in_proj to
    (z, xBC, dt), causal conv + SiLU on xBC, A = -exp(A_log) (A ~ U(1, 16)),
    dt = softplus(. + dt_bias), SSD recurrence S = exp(dt A) S + dt x B^T,
    y = S C + D x, gated RMSNorm (norm after the SiLU(z) gate), out_proj."""
    def __init__(self, dim, d_state=64, head_dim=64, expand=2, conv_kernel=4):
        super().__init__()
        self.d_inner = expand * dim
        # Largest head width up to ``head_dim`` dividing d_inner (a power of
        # two, as the state kernel needs); d_inner % head_dim == 0 is unchanged.
        self.P = math.gcd(head_dim, self.d_inner)
        self.H, self.N = self.d_inner // self.P, d_state
        self.in_proj = nn.Linear(dim, 2 * self.d_inner + 2 * d_state + self.H, bias=False)
        self.conv = _ShortConv(self.d_inner + 2 * d_state, conv_kernel, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.empty(self.H).uniform_(1, 16)))
        dt = torch.exp(torch.rand(self.H) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.D = nn.Parameter(torch.ones(self.H))
        self.norm = RMSNorm(self.d_inner, eps=1e-5)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def forward(self, x, state=None):
        Bsz, T, _ = x.shape
        H, P, N = self.H, self.P, self.N
        buf, S = (None, None) if state is None else state
        z, xbc, dt = self.in_proj(x).split([self.d_inner, self.d_inner + 2 * N, H], -1)
        xbc, buf = self.conv(xbc, buf)
        xs, Bm, Cm = xbc.float().split([self.d_inner, N, N], -1)
        dt = F.softplus(dt.float() + self.dt_bias)                                 # (B, T, H)
        decay = torch.exp(dt * -torch.exp(self.A_log.float()))
        rep = lambda t: t.unsqueeze(2).expand(Bsz, T, H, N).reshape(Bsz, T, H * N)
        k = (dt.unsqueeze(-1) * Bm.unsqueeze(2)).reshape(Bsz, T, H * N)
        w = decay.unsqueeze(-1).expand(Bsz, T, H, N).reshape(Bsz, T, H * N)
        if S is None:
            S = x.new_zeros(Bsz, H, P, N, dtype=torch.float32)
        zeros = torch.zeros_like(k)
        y, S = _state_op(rep(Cm), w, k, xs, zeros, zeros, S)
        y = y + (self.D.repeat_interleave(P) * xs)
        y = self.norm(y * F.silu(z.float()))
        return self.out_proj(y.to(x.dtype)), (buf, S)


class RetNetMixer(nn.Module):
    """Multi-scale retention (Sun et al., 2023; fla ``MultiScaleRetention``):
    rotary q/k, per-head decay gamma_h = 1 - 2^(-5-h), S = gamma S + v k^T,
    o = S q / sqrt(d_k), per-head RMSNorm with a swish output gate."""
    def __init__(self, dim, num_heads=None, expand_v=2):
        super().__init__()
        H = num_heads or next(h for h in (8, 4, 2, 1) if dim % h == 0 and (dim // h) % 2 == 0)
        self.H, self.dk, self.dv = H, dim // H, expand_v * dim // H
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, H * self.dv, bias=False)
        self.g_proj = nn.Linear(dim, H * self.dv, bias=False)
        self.g_norm = RMSNorm(self.dv, eps=1e-5)
        self.o_proj = nn.Linear(H * self.dv, dim, bias=False)
        self.register_buffer("gamma", 1 - 2.0 ** (-5.0 - torch.arange(H, dtype=torch.float32)), persistent=False)
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, self.dk, 2).float() / self.dk)), persistent=False)

    def _rotary(self, t, pos):
        freqs = pos[:, None] * self.inv_freq[None, :]
        cos, sin = torch.cos(freqs)[None, :, None, :], torch.sin(freqs)[None, :, None, :]
        a, b = t.chunk(2, -1)
        return torch.cat((a * cos - b * sin, a * sin + b * cos), -1)

    def forward(self, x, state=None):
        B, T, _ = x.shape
        H = self.H
        offset, S = (0, None) if state is None else state
        pos = torch.arange(offset, offset + T, device=x.device, dtype=torch.float32)
        q = self._rotary(_heads(self.q_proj(x).float(), H), pos) * self.dk ** -0.5
        k = self._rotary(_heads(self.k_proj(x).float(), H), pos)
        w = self.gamma.view(1, 1, H, 1).expand(B, T, H, self.dk).reshape(B, T, H * self.dk)
        if S is None:
            S = x.new_zeros(B, H, self.dv, self.dk, dtype=torch.float32)
        zeros = torch.zeros_like(w)
        y, S = _state_op(q.reshape(B, T, -1), w, k.reshape(B, T, -1), self.v_proj(x).float(), zeros, zeros, S)
        y = self.g_norm(y.view(B, T, H, self.dv)) * F.silu(self.g_proj(x).float().view(B, T, H, self.dv))
        return self.o_proj(y.reshape(B, T, -1).to(x.dtype)), (offset + T, S)


class LinearRecurrentStack(nn.Module):
    """Pre-norm residual stack of a linear-recurrence mixer, each followed by a
    SwiGLU MLP (fla model layout; Mamba-2 blocks are mixer-only).  Exposes
    ``forward_seq`` / ``step`` for MEGABYTE stages."""
    def __init__(self, dim, depth, kind):
        super().__init__()
        self.kind, self.depth = kind, depth
        make = {"deltanet": lambda: DeltaNetMixer(dim), "gated_deltanet": lambda: DeltaNetMixer(dim, gated=True),
                "hgrn2": lambda: HGRN2Mixer(dim), "mamba2": lambda: Mamba2Mixer(dim),
                "retnet": lambda: RetNetMixer(dim)}[kind]
        self.mixers = nn.ModuleList([make() for _ in range(depth)])
        self.mixer_norms = nn.ModuleList([RMSNorm(dim, eps=1e-5) for _ in range(depth)])
        self.use_mlp = kind != "mamba2"
        if self.use_mlp:
            hidden = int(8 * dim / 3)
            self.mlp_norms = nn.ModuleList([RMSNorm(dim, eps=1e-5) for _ in range(depth)])
            self.mlp_gate = nn.ModuleList([nn.Linear(dim, hidden, bias=False) for _ in range(depth)])
            self.mlp_up = nn.ModuleList([nn.Linear(dim, hidden, bias=False) for _ in range(depth)])
            self.mlp_down = nn.ModuleList([nn.Linear(hidden, dim, bias=False) for _ in range(depth)])
        if kind == "hgrn2":
            self.lower_bounds = nn.Parameter(torch.zeros(depth, dim))

    def forward_seq(self, x, state=None):
        new_state = []
        if self.kind == "hgrn2":
            lb = self.lower_bounds.softmax(0)
            lb = lb.cumsum(0) - lb[0]
        for i, mixer in enumerate(self.mixers):
            st = None if state is None else state[i]
            h = self.mixer_norms[i](x)
            if self.kind == "hgrn2":
                y, st = mixer(h, st, lower_bound=lb[i] if i > 0 else None)
            else:
                y, st = mixer(h, st)
            x = x + y
            if self.use_mlp:
                h = self.mlp_norms[i](x)
                x = x + self.mlp_down[i](F.silu(self.mlp_gate[i](h)) * self.mlp_up[i](h))
            new_state.append(st)
        return x, new_state

    def step(self, x_t, state=None):
        y, state = self.forward_seq(x_t.unsqueeze(1), state)
        return y[:, 0], state


class LinearRecurrentLM(nn.Module):
    """Language model around ``LinearRecurrentStack``."""
    def __init__(self, vocab_size, dim, depth, kind):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.stack = LinearRecurrentStack(dim, depth, kind)
        self.norm = RMSNorm(dim, eps=1e-5)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward_hidden(self, x, state=None):
        x, state = self.stack.forward_seq(x, state)
        return self.norm(x), state

    def forward(self, idx, state=None):
        hidden, state = self.forward_hidden(self.embed(idx), state)
        return self.head(hidden), state


# ==============================================================================
# Structured SSMs: S4 (DPLR), S4D, DSS, S5, LRU.  Linear time-invariant layers,
# trained through an FFT convolution and carrying an exact recurrent state
# (the initial state's contribution is added in closed form).
# ==============================================================================
def _hippo_legs_dplr(N):
    """HiPPO-LegS in normal-plus-low-rank form (S4 ``nplr('legs')``), keeping
    the N/2 eigenvalues with positive imaginary part.  Returns (Lambda, P, B, V)."""
    q = torch.arange(N, dtype=torch.float64)
    r = torch.sqrt(2 * q + 1)
    A = -(torch.tril(r[:, None] * r[None, :]) - torch.diag(q))
    B = torch.sqrt(2 * q + 1)
    P = torch.sqrt(q + 0.5)
    S = A + P[:, None] * P[None, :]                   # normal (skew-symmetric + -1/2 I)
    lam_re = torch.mean(torch.diagonal(S)) * torch.ones(N, dtype=torch.float64)
    lam_im, V = torch.linalg.eigh(S * -1j)
    lam = lam_re + 1j * lam_im
    order = torch.argsort(lam.imag, descending=True)[: N // 2]
    V = V[:, order]
    return lam[order], V.conj().T @ P.to(V.dtype), V.conj().T @ B.to(V.dtype), V


def _fft_causal_conv(u, k):
    """y[..., t] = sum_s k[..., s] u[..., t - s] (complex-safe, zero history)."""
    L = u.shape[-1]
    n = 2 * L
    if torch.is_complex(u) or torch.is_complex(k):
        return torch.fft.ifft(torch.fft.fft(u, n=n) * torch.fft.fft(k, n=n), n=n)[..., :L]
    return torch.fft.irfft(torch.fft.rfft(u, n=n) * torch.fft.rfft(k, n=n), n=n)[..., :L]


class _DiagSISOSSM(nn.Module):
    """Per-channel diagonal SSM (S4D or DSS-exp): x_t = lam x_{t-1} + u_t,
    y_t = scale * Re(coef . x_t) + D u_t, then GELU and a 1x1 GLU output
    (S4D block).  State: (B, H, N) complex."""
    def __init__(self, H, N=64, kind="s4d", dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.kind, self.H = kind, H
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        if kind == "s4d":                                             # S4D-Lin, state N/2 (conj pairs)
            n = N // 2
            self.C = nn.Parameter(torch.view_as_real(torch.randn(H, n, dtype=torch.cfloat)))
            self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(H, n)))
            self.A_imag = nn.Parameter(math.pi * torch.arange(n).float().repeat(H, 1))
        else:                                                         # DSS-exp, skew-HiPPO init
            lam, _, _, _ = _hippo_legs_dplr(2 * N)
            self.Lambda_re = nn.Parameter(torch.log(-lam.real).float())
            self.Lambda_im = nn.Parameter(lam.imag.float())
            self.W = nn.Parameter(torch.view_as_real(torch.randn(H, N, dtype=torch.cfloat)))
        self.D = nn.Parameter(torch.randn(H))
        self.output_linear = nn.Linear(H, 2 * H)

    def _params(self):
        dt = torch.exp(self.log_dt)[:, None]
        if self.kind == "s4d":
            A = -torch.exp(self.log_A_real) + 1j * self.A_imag
            coef = torch.view_as_complex(self.C) * (torch.exp(dt * A) - 1.0) / A
            return torch.exp(dt * A), coef, 2.0
        A = (-torch.exp(self.Lambda_re) + 1j * self.Lambda_im)[None, :]
        coef = torch.view_as_complex(self.W) * (torch.exp(dt * A) - 1.0) / A
        return torch.exp(dt * A), coef, 1.0

    def forward(self, u, state=None):
        Bsz, T, H = u.shape
        lam, coef, scale = self._params()                             # (H, n)
        ut = u.float().transpose(1, 2)                                # (B, H, T)
        powers = lam[..., None] ** torch.arange(T, device=u.device)   # (H, n, T)
        k = scale * torch.einsum("hn,hnl->hl", coef, powers).real
        y = _fft_causal_conv(ut, k) + self.D[:, None] * ut
        if state is not None:
            y = y + scale * torch.einsum("hn,bhn,hnl->bhl", coef, state, powers * lam[..., None]).real
        # final state: lam^T x0 + sum_s lam^(T-1-s) u_s
        x_T = torch.einsum("hnl,bhl->bhn", powers.flip(-1).to(torch.cfloat), ut.to(torch.cfloat))
        if state is not None:
            x_T = x_T + state * lam ** T
        y = F.gelu(y).transpose(1, 2)
        return F.glu(self.output_linear(y), dim=-1).to(u.dtype), x_T


class _S4DPLR(nn.Module):
    """S4 (Gu et al., 2022): per-channel SISO SSM with state matrix
    A = Lambda - P P^* (HiPPO-LegS DPLR init, shared across channels, conj-
    symmetric half state), bilinear discretization with per-channel dt,
    y = 2 Re(C x) + D u, then GELU and a 1x1 GLU output.  The kernel
    C A_bar^l B_bar is evaluated by repeated squaring of the dense N/2 x N/2
    A_bar (the same kernel S4's Cauchy/Woodbury evaluation computes).  C is
    learned directly rather than through S4's C~ = C (I - A_bar^L)."""
    def __init__(self, H, N=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        lam, P, B, _ = _hippo_legs_dplr(N)
        n = N // 2
        self.H, self.n = H, n
        self.log_dt = nn.Parameter(torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.Lambda_re = nn.Parameter(torch.log(-lam.real).float())    # real part kept negative
        self.Lambda_im = nn.Parameter(lam.imag.float())
        self.P = nn.Parameter(torch.view_as_real(P.to(torch.cfloat)))
        self.B = nn.Parameter(torch.view_as_real(B.to(torch.cfloat)))
        C = torch.randn(H, n, dtype=torch.cfloat) * (0.5 ** 0.5)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.D = nn.Parameter(torch.randn(H))
        self.output_linear = nn.Linear(H, 2 * H)

    def _discrete(self):
        lam = -torch.exp(self.Lambda_re) + 1j * self.Lambda_im
        P, B = torch.view_as_complex(self.P), torch.view_as_complex(self.B)
        A = torch.diag(lam) - P[:, None] * P.conj()[None, :]           # (n, n)
        dt = torch.exp(self.log_dt)[:, None, None]                     # (H, 1, 1)
        I = torch.eye(self.n, dtype=A.dtype, device=A.device)
        left = torch.linalg.inv(I - dt / 2 * A)                        # (H, n, n)
        return left @ (I + dt / 2 * A), (left @ (dt * B[:, None]))[..., 0]   # A_bar, B_bar

    @staticmethod
    def _power_seq(Abar, v, T):
        """[A^0 v, A^1 v, ..., A^(T-1) v] by doubling: returns (..., n, T)."""
        seq = v[..., None]
        Ap = Abar
        while seq.shape[-1] < T:
            seq = torch.cat((seq, Ap @ seq), -1)
            Ap = Ap @ Ap
        return seq[..., :T]

    def forward(self, u, state=None):
        Bsz, T, H = u.shape
        Abar, Bbar = self._discrete()
        C = torch.view_as_complex(self.C)
        ut = u.float().transpose(1, 2)
        vecs = self._power_seq(Abar, Bbar, T)                          # (H, n, T)
        k = 2 * torch.einsum("hn,hnl->hl", C, vecs).real
        y = _fft_causal_conv(ut, k) + self.D[:, None] * ut
        # final state x_T = sum_s A^(T-1-s) B u_s  (+ A^T x0)
        x_T = torch.einsum("hnl,bhl->bhn", vecs.flip(-1), ut.to(torch.cfloat))
        if state is not None:
            svec = self._power_seq(Abar[None], torch.einsum("hij,bhj->bhi", Abar, state), T)   # (B, H, n, T)
            y = y + 2 * torch.einsum("hn,bhnl->bhl", C, svec).real
            x_T = x_T + svec[..., -1]                                   # A^T x0
        y = F.gelu(y).transpose(1, 2)
        return F.glu(self.output_linear(y), dim=-1).to(u.dtype), x_T


class _DiagMIMOSSM(nn.Module):
    """MIMO diagonal SSM.  kind "s5" (Smith et al., 2023): HiPPO-N init,
    conjugate-symmetric state (2 Re), per-state dt with ZOH,
    B_bar = (lam_bar - 1)/lam B~, y = 2 Re(C~ x) + D u, then GELU and the
    half-GLU output x * sigmoid(W x).  kind "lru" (Orvieto et al., 2023):
    lam = exp(-exp(nu) + i exp(theta)) on a ring, gamma = sqrt(1 - |lam|^2)
    input normalization, y = Re(C x) + D u, followed by a GLU.
    State: (B, P) complex."""
    def __init__(self, H, P=None, kind="s5", r_min=0.0, r_max=1.0, max_phase=2 * math.pi,
                 dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.kind, self.H = kind, H
        P = P or H
        if kind == "s5":
            lam, _, _, V = _hippo_legs_dplr(P)                         # HiPPO-N eigen-decomposition
            n = P // 2
            self.n = n
            self.Lambda_re = nn.Parameter(torch.log(-lam.real).float())
            self.Lambda_im = nn.Parameter(lam.imag.float())
            Vinv = torch.linalg.inv(V[:, :n].conj().T @ V[:, :n]) @ V[:, :n].conj().T   # (n, P)
            B = torch.randn(P, H, dtype=torch.float64) / math.sqrt(H)                   # lecun normal
            C = torch.randn(H, P, dtype=torch.float64) * (0.5 ** 0.5)
            self.B = nn.Parameter(torch.view_as_real((Vinv @ B.to(Vinv.dtype)).to(torch.cfloat)))
            self.C = nn.Parameter(torch.view_as_real((C.to(V.dtype) @ V[:, :n]).to(torch.cfloat)))
            self.log_dt = nn.Parameter(torch.rand(n) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
            self.out_gate = nn.Linear(H, H)
        else:
            n = P
            self.n = n
            u1, u2 = torch.rand(n), torch.rand(n)
            self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2)))
            self.theta_log = nn.Parameter(torch.log(max_phase * u2))
            lam_abs = torch.exp(-torch.exp(self.nu_log.detach()))
            self.gamma_log = nn.Parameter(torch.log(torch.sqrt(1 - lam_abs ** 2)))
            self.B = nn.Parameter(torch.view_as_real(torch.complex(torch.randn(n, H), torch.randn(n, H)) / math.sqrt(2 * H)))
            self.C = nn.Parameter(torch.view_as_real(torch.complex(torch.randn(H, n), torch.randn(H, n)) / math.sqrt(n)))
            self.mlp = nn.Linear(H, 2 * H)
        self.D = nn.Parameter(torch.randn(H))

    def _params(self):
        if self.kind == "s5":
            A = -torch.exp(self.Lambda_re) + 1j * self.Lambda_im
            lam = torch.exp(A * torch.exp(self.log_dt))
            Bbar = ((lam - 1) / A)[:, None] * torch.view_as_complex(self.B)
            return lam, Bbar, torch.view_as_complex(self.C), 2.0
        lam = torch.exp(-torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log))
        Bbar = torch.exp(self.gamma_log)[:, None] * torch.view_as_complex(self.B)
        return lam, Bbar, torch.view_as_complex(self.C), 1.0

    def forward(self, u, state=None):
        Bsz, T, H = u.shape
        lam, Bbar, C, scale = self._params()
        bu = torch.einsum("nh,bth->bnt", Bbar, u.float().to(torch.cfloat))            # (B, n, T)
        powers = lam[:, None] ** torch.arange(T, device=u.device)                     # (n, T)
        X = _fft_causal_conv(bu, powers)                                              # (B, n, T)
        if state is not None:
            X = X + state[..., None] * (powers * lam[:, None])
        y = scale * torch.einsum("hn,bnt->bth", C, X).real + self.D * u.float()
        if self.kind == "s5":
            y = F.gelu(y)
            y = y * torch.sigmoid(self.out_gate(y))
        else:
            y = F.glu(self.mlp(y), dim=-1)
        return y.to(u.dtype), X[..., -1]


class StructuredSSMLM(nn.Module):
    """Language model of pre-LayerNorm residual structured-SSM blocks."""
    def __init__(self, vocab_size, dim, depth, kind, d_state=64):
        super().__init__()
        make = {"s4": lambda: _S4DPLR(dim, d_state), "s4d": lambda: _DiagSISOSSM(dim, d_state, "s4d"),
                "dss": lambda: _DiagSISOSSM(dim, d_state, "dss"), "s5": lambda: _DiagMIMOSSM(dim, dim, "s5"),
                "lru": lambda: _DiagMIMOSSM(dim, dim, "lru")}[kind]
        self.embed = nn.Embedding(vocab_size, dim)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.layers = nn.ModuleList([make() for _ in range(depth)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, vocab_size)

    def forward_hidden(self, x, state=None):
        new_state = []
        for i, (norm, layer) in enumerate(zip(self.norms, self.layers)):
            y, st = layer(norm(x), None if state is None else state[i])
            x = x + y
            new_state.append(st)
        return self.norm(x), new_state

    def forward(self, idx, state=None):
        hidden, state = self.forward_hidden(self.embed(idx), state)
        return self.head(hidden), state


# ==============================================================================
# Original Transformer (Vaswani et al., 2017), decoder-only
# ==============================================================================
class _VaswaniDecoderLayer(nn.Module):
    """Masked multi-head self-attention and a position-wise FFN, each wrapped
    as LayerNorm(x + Dropout(sublayer(x))) (post-LN, Section 5.4)."""
    def __init__(self, dim, n_heads, ff_dim, act_name="relu", dropout=0.1):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"Original Transformer needs embed_dim ({dim}) divisible by head_count ({n_heads})")
        self.n_heads = n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn = make_feed_forward(dim, ff_dim, act_name=act_name, bias=True)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.drop, self.dropout = nn.Dropout(dropout), dropout

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=self.dropout if self.training else 0.0)
        x = self.norm1(x + self.drop(self.proj(att.transpose(1, 2).reshape(B, T, D))))
        return self.norm2(x + self.drop(self.ffn(x)))


class OriginalTransformerLM(nn.Module):
    """The 2017 Transformer's decoder stack used as a decoder-only LM (the
    encoder and cross-attention dropped): embeddings scaled by sqrt(d_model)
    plus fixed sinusoidal positions, post-LN layers, ReLU FFN with
    d_ff = 4 d_model, residual dropout 0.1, and the pre-softmax projection
    tied to the embedding matrix (Section 3.4)."""
    def __init__(self, vocab_size, dim, depth, n_heads, max_seq_len, act_name="relu", dropout=0.1):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embed.weight, 0.0, dim ** -0.5)
        self.pos = SinusoidalPositionalEncoding(dim, max_len=max(1, int(max_seq_len)))
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            _VaswaniDecoderLayer(dim, n_heads, 4 * dim, act_name=act_name, dropout=dropout) for _ in range(depth)
        ])

    def forward_hidden(self, x):
        x = self.drop(x * math.sqrt(self.dim) + self.pos(x.size(1), x.device).to(x.dtype))
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, idx):
        return F.linear(self.forward_hidden(self.embed(idx)), self.embed.weight)


# ==============================================================================
# 2026 dense Transformer (Arcee Trinity recipe, arXiv 2602.17004)
# ==============================================================================
class _TrinityAttention(nn.Module):
    """GQA with QK-norm (RMSNorm on each query / key head) and elementwise
    gated attention, o = W_O [SDPA(q, k, v) * sigmoid(W_G x)] (Qiu et al.,
    2025).  Local layers: RoPE + causal sliding window; global layers: NoPE,
    full causal attention."""
    def __init__(self, dim, n_heads, n_kv_heads, local, window, max_seq_len):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"Trinity Transformer needs embed_dim ({dim}) divisible by head_count ({n_heads})")
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, dim // n_heads
        self.local, self.window = local, window
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wg = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim, eps=1e-6), RMSNorm(self.head_dim, eps=1e-6)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max(1, int(max_seq_len))) if local else None

    def forward(self, x):
        B, T, D = x.shape
        q = self.q_norm(self.wq(x).view(B, T, self.n_heads, self.head_dim))
        k = self.k_norm(self.wk(x).view(B, T, self.n_kv_heads, self.head_dim))
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        if self.local:
            cos, sin = self.rope(q, T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        rep = self.n_heads // self.n_kv_heads
        q, k, v = q.transpose(1, 2), k.repeat_interleave(rep, 2).transpose(1, 2), v.repeat_interleave(rep, 2).transpose(1, 2)
        if self.local and self.window < T:
            i = torch.arange(T, device=x.device)
            mask = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < self.window)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, D) * torch.sigmoid(self.wg(x))
        return self.wo(out)


class _TrinityBlock(nn.Module):
    """Depth-scaled sandwich norm: x + RMSNorm2(M(RMSNorm1(x))) for the
    attention and the FFN, with the second gain initialised to 1/sqrt(L)."""
    def __init__(self, dim, n_heads, n_kv_heads, local, window, max_seq_len, depth, act_name):
        super().__init__()
        self.attn = _TrinityAttention(dim, n_heads, n_kv_heads, local, window, max_seq_len)
        self.ffn = make_feed_forward(dim, 3 * dim, act_name=act_name, bias=False)
        self.attn_in, self.attn_out = RMSNorm(dim, eps=1e-6), RMSNorm(dim, eps=1e-6)
        self.ffn_in, self.ffn_out = RMSNorm(dim, eps=1e-6), RMSNorm(dim, eps=1e-6)
        with torch.no_grad():
            self.attn_out.weight.fill_(depth ** -0.5)
            self.ffn_out.weight.fill_(depth ** -0.5)

    def forward(self, x):
        x = x + self.attn_out(self.attn(self.attn_in(x)))
        return x + self.ffn_out(self.ffn(self.ffn_in(x)))


class TrinityTransformerLM(nn.Module):
    """State-of-the-art (2026) dense decoder-only Transformer following the
    Arcee Trinity technical report (Feb 2026), whose attention recipe is shared
    by MiMo-V2-Flash and ZGCM-1: GQA (4 query heads per KV head where the
    head count allows), QK-norm, elementwise gated attention, a repeating
    3:1 local/global layer pattern (RoPE + sliding window of half the context
    locally, NoPE globally), depth-scaled sandwich RMSNorm, SwiGLU FFN of
    width 3 d, a final RMSNorm, and truncated-normal init with
    sigma = 0.5 / sqrt(d).  Trinity's MoE FFNs are replaced by the dense FFN
    its first layers use, so the model compares with the other dense LMs.
    The last layer is always global, so stacks shallower than four layers
    still see the whole context."""
    def __init__(self, vocab_size, dim, depth, n_heads, max_seq_len, window=None, act_name="swiglu"):
        super().__init__()
        n_kv = max(d for d in range(1, n_heads + 1) if n_heads % d == 0 and d <= max(1, n_heads // 4))
        window = max(1, int(window or max(1, int(max_seq_len) // 2)))
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            _TrinityBlock(dim, n_heads, n_kv, local=(i % 4 != 3 and i != depth - 1), window=window,
                          max_seq_len=max_seq_len, depth=depth, act_name=act_name)
            for i in range(depth)
        ])
        self.norm = RMSNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        sigma = 0.5 / math.sqrt(dim)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.trunc_normal_(module.weight, 0.0, sigma, -3 * sigma, 3 * sigma)

    def forward_hidden(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, idx):
        return self.head(self.forward_hidden(self.embed(idx)))


# ==============================================================================
# M2RNN (Mishra, Tan, Stoica, Gonzalez & Dao, 2026), arXiv 2603.14360
# ==============================================================================
class M2RNNLayer(nn.Module):
    """Matrix-to-Matrix RNN layer (paper Eqs. 14-23), multi-value form with
    K = 64, V = 16 (Appendix A): one query / key head shared by
    NH = ceil(dim / V) value heads, each with a K x V matrix state and a V x V
    transition W (identity init).  q, k, v: linear -> causal depthwise conv
    (k = 4) -> SiLU; forget gate f = (1 + exp(W_f x + beta))^-alpha per head;
    output gate SiLU(W_g x);
        Z_t = tanh(H_{t-1} W + k_t v_t^T),  H_t = f_t H_{t-1} + (1 - f_t) Z_t,
        y_t = H_t^T q_t + w_r * v_t,  o_t = W_o RMSNorm(y_t * g_t).
    The paper gives alpha ~ U(a_min, a_max) and beta ~ LogU(b_min, b_max)
    without the ranges; alpha ~ U(0.05, 1) and beta ~ LogU(1e-3, 1) (a spread
    of per-head memory lengths, f(0) from 0.5 to 0.97) are our choice, with
    alpha kept positive through a log parameter.  The paper's per-step
    clipping of dL/dH_t in BPTT is not applied; linegen's global gradient
    clipping covers training stability.  State: (H, conv buffer)."""
    def __init__(self, dim, key_dim=64, value_dim=16, conv_kernel=4):
        super().__init__()
        self.key_dim, self.value_dim = key_dim, value_dim
        self.n_heads = -(-dim // value_dim)
        inner = self.n_heads * value_dim
        self.qkv = nn.Linear(dim, 2 * key_dim + inner)
        self.conv = _XLCausalConv(2 * key_dim + inner, conv_kernel)
        self.f_proj = nn.Linear(dim, self.n_heads, bias=False)
        self.g_proj = nn.Linear(dim, inner, bias=False)
        self.log_alpha = nn.Parameter(torch.empty(self.n_heads).uniform_(0.05, 1.0).log())
        self.beta = nn.Parameter(torch.exp(torch.empty(self.n_heads).uniform_(math.log(1e-3), 0.0)))
        self.W = nn.Parameter(torch.eye(value_dim).repeat(self.n_heads, 1, 1))
        self.w_r = nn.Parameter(torch.ones(inner))
        self.norm = RMSNorm(inner, eps=1e-6)
        self.o_proj = nn.Linear(inner, dim, bias=False)

    def forward(self, x, state=None):
        B, T, _ = x.shape
        NH, Kd, Vd = self.n_heads, self.key_dim, self.value_dim
        h0, buf = (None, None) if state is None else state
        qkv, buf = self.conv(self.qkv(x), buf)
        q, k, v = F.silu(qkv).split((Kd, Kd, NH * Vd), dim=-1)
        v = v.reshape(B, T, NH, Vd)
        # log f = -alpha * softplus(W_f x + beta), computed in fp32.
        f = torch.exp(-self.log_alpha.exp() * F.softplus(self.f_proj(x).float() + self.beta))
        if h0 is None:
            h0 = x.new_zeros(B, NH, Kd, Vd, dtype=torch.float32)
        use_kernel = kernels_available(x) and m2rnn_supported(Kd, Vd)
        y, h = (m2rnn_scan if use_kernel else m2rnn_reference)(q, k, v, f, self.W, h0)
        y = y.to(x.dtype).reshape(B, T, NH * Vd) + self.w_r * v.reshape(B, T, NH * Vd)
        out = self.o_proj(self.norm(y * F.silu(self.g_proj(x))))
        return out, (h, buf)


class M2RNNLM(nn.Module):
    """M2RNN language model: pre-norm RMSNorm residual blocks alternating an
    M2RNN layer and a SwiGLU MLP, no positional embeddings (paper Figure 2
    and Section 5.2).  Training and sampling are both sequential in time."""
    def __init__(self, vocab_size, dim, depth, key_dim=64, value_dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.mixers = nn.ModuleList([M2RNNLayer(dim, key_dim, value_dim) for _ in range(depth)])
        self.mixer_norms = nn.ModuleList([RMSNorm(dim, eps=1e-6) for _ in range(depth)])
        self.ffn_norms = nn.ModuleList([RMSNorm(dim, eps=1e-6) for _ in range(depth)])
        self.ffns = nn.ModuleList([SwiGLU(dim, int(8 * dim / 3)) for _ in range(depth)])
        self.norm = RMSNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward_hidden(self, x, state=None):
        new_state = []
        for i, mixer in enumerate(self.mixers):
            y, st = mixer(self.mixer_norms[i](x), None if state is None else state[i])
            x = x + y
            x = x + self.ffns[i](self.ffn_norms[i](x))
            new_state.append(st)
        return self.norm(x), new_state

    def forward(self, idx, state=None):
        hidden, state = self.forward_hidden(self.embed(idx), state)
        return self.head(hidden), state


# ==============================================================================
# ParaRNN (Danieli et al., ICLR 2026, arXiv 2510.21450): ParaGRU / ParaLSTM
# ==============================================================================
class ParaRNNMixer(nn.Module):
    """ParaRNN block (paper App. C.1, Fig. 9): Linear -> short causal conv ->
    ParaGRU / ParaLSTM cell -> + learnable-scale input skip -> gated RMSNorm
    -> Linear.  The cell is multi-head: its input matrices B are block-diagonal
    over ``num_heads`` heads (4 in the paper), its state matrices diagonal, so
    training solves the whole sequence with 3 Newton iterations of parallel
    scans (linegen_kernels.pararnn_solve); a single step (decoding) runs the
    cell directly.  Init as in App. C.1: B Kaiming-uniform, diagonal a / peephole
    p truncated-normal(1/sqrt(d)), biases 0; a and p are clamped to [-0.5, 0.5]
    in the forward pass ("clip ... to a maximum of 0.5", which keeps Newton's
    3 iterations converged).  Mamba-inherited details the paper leaves
    implicit follow Mamba: conv kernel 4 with SiLU, gate SiLU(z) before the
    RMSNorm, cell width = model width.  State: (cell state, conv buffer)."""
    def __init__(self, dim, kind="gru", num_heads=4, conv_kernel=4, newton_iters=3):
        super().__init__()
        self.cell = ParaGRUCell if kind == "gru" else ParaLSTMCell
        self.heads = math.gcd(num_heads, dim)
        self.dim, self.newton_iters = dim, int(newton_iters)
        hd = dim // self.heads
        self.in_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.conv = _XLCausalConv(dim, conv_kernel)
        self.B = nn.Parameter(torch.empty(self.heads, hd, 3, hd).uniform_(-math.sqrt(3 / hd), math.sqrt(3 / hd)))
        self.b = nn.Parameter(torch.zeros(3, dim))
        self.a = nn.Parameter(nn.init.trunc_normal_(torch.empty(3, dim), 0.0, dim ** -0.5, -0.9, 0.9))
        self.p = (nn.Parameter(nn.init.trunc_normal_(torch.empty(2, dim), 0.0, dim ** -0.5, -0.9, 0.9))
                  if kind == "lstm" else None)
        self.scale = nn.Parameter(torch.ones(dim))
        self.norm = RMSNorm(dim, eps=1e-6)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, u, state=None):
        Bsz, T, D = u.shape
        h0, buf = (None, None) if state is None else state
        x, z = self.in_proj(u).chunk(2, dim=-1)
        x, buf = self.conv(x, buf)
        x = F.silu(x)
        xp = torch.einsum("bthi,hivj->btvhj", x.view(Bsz, T, self.heads, -1), self.B).reshape(Bsz, T, 3, D) + self.b
        params = [self.a.clamp(-0.5, 0.5)] + ([] if self.p is None else [self.p.clamp(-0.5, 0.5)])
        if h0 is None:
            h0 = u.new_zeros(Bsz, D, dtype=torch.float32) if self.p is None else u.new_zeros(Bsz, 2, D, dtype=torch.float32)
        if T == 1:
            states = pararnn_reference(self.cell, xp, h0, *params)
        else:
            states = pararnn_solve(self.cell, xp, h0, *params, newton_iters=self.newton_iters)
        y = states if self.p is None else states[..., 1, :]
        y = y.to(u.dtype) + self.scale * x
        out = self.out_proj(self.norm(y * F.silu(z)))
        return out, (states[:, -1], buf)


class ParaRNNLM(nn.Module):
    """ParaGRU / ParaLSTM language model: Transformer-style pre-norm residual
    blocks alternating a ParaRNN mixer and a SwiGLU MLP (paper App. C.1)."""
    def __init__(self, vocab_size, dim, depth, kind="gru", num_heads=4, newton_iters=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.mixers = nn.ModuleList([ParaRNNMixer(dim, kind, num_heads, newton_iters=newton_iters)
                                     for _ in range(depth)])
        self.mixer_norms = nn.ModuleList([RMSNorm(dim, eps=1e-6) for _ in range(depth)])
        self.ffn_norms = nn.ModuleList([RMSNorm(dim, eps=1e-6) for _ in range(depth)])
        self.ffns = nn.ModuleList([SwiGLU(dim, int(8 * dim / 3)) for _ in range(depth)])
        self.norm = RMSNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward_hidden(self, x, state=None):
        new_state = []
        for i, mixer in enumerate(self.mixers):
            y, st = mixer(self.mixer_norms[i](x), None if state is None else state[i])
            x = x + y
            x = x + self.ffns[i](self.ffn_norms[i](x))
            new_state.append(st)
        return self.norm(x), new_state

    def forward(self, idx, state=None):
        hidden, state = self.forward_hidden(self.embed(idx), state)
        return self.head(hidden), state
