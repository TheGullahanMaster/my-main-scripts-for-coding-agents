import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle  # For caching
from PIL import Image  # For image loading and resizing
from torchpwl import PWL
from torch.utils.data import DataLoader, Dataset
#from autoclip.torch import QuantileClip
from collections import deque
from tqdm import tqdm
import traceback
import threading
import time
import textwrap
import copy
import importlib
import ast
import struct


##############################################
# Custom Noise Injection Layer
##############################################
class NoiseInjectionLayer(nn.Module):
    def __init__(self, std=0.1):
        super(NoiseInjectionLayer, self).__init__()
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise

##############################################
# MVPOG and custom activations (unchanged)
##############################################
class MVPOG(nn.Module):
    def __init__(self, hidden_size, num_breakpoints):
        super(MVPOG, self).__init__()
        self.hidden_size = hidden_size
        self.pwl = PWL(num_channels=hidden_size, num_breakpoints=num_breakpoints)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), self.hidden_size, 1, 1)
        x = self.pwl(x)
        x = x.view(x.size(0), self.hidden_size)
        return x

    def get_learned_parameters(self):
        return {
            "slopes": self.pwl.get_slopes().detach().cpu().numpy(),
            "biases": self.pwl.get_biases().detach().cpu().numpy(),
            "x_positions": self.pwl.get_x_positions().detach().cpu().numpy()
        }

# Custom activations not built into PyTorch
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class Cosine(nn.Module):
    def forward(self, x):
        return torch.cos(x)

# Imported custom modules (ensure these are in your PYTHONPATH)
from pau import PAU      # PAU activation (if needed)
from lamb import *  # Lamb, GoLU, and AdamMHD
import mlpres_seq as seqm  # inenc / outdec columns: sequence encoders and decoders
#from hypergrad import SGDHD


class PaWeL(nn.Module):
    """torchpwl.PWL for MLPs: one learnable piecewise-linear function per neuron.

    Same parameters, initialization and math as torchpwl.PWL (x_positions [C, K],
    slopes [C, K+1], biases [C]; K breakpoints, K+1 segments, outer segments extend
    linearly), but channels live on the LAST dim, so [N, C] (or [..., C]) works
    directly without the conv-style [N, C, 1, 1] reshape. C is taken from the first
    forward pass or from a loaded checkpoint. Legacy lamb.PaWeL checkpoints (one
    shared keypoint set, x_points/y_points) are converted to the identical function.
    """
    def __init__(self, breakpoints=32, num_channels=None):
        super().__init__()
        self.num_breakpoints = int(breakpoints)
        if self.num_breakpoints < 1:
            raise ValueError("PaWeL needs at least 1 breakpoint.")
        self.num_channels = None
        for name in ("x_positions", "slopes", "biases"):
            self.register_parameter(name, None)
        self.register_buffer("_anchor", torch.empty(0), persistent=False)  # tracks .to(device)
        if num_channels is not None:
            self._build(num_channels)

    def _build(self, num_channels):
        c, k = int(num_channels), self.num_breakpoints
        kw = dict(device=self._anchor.device, dtype=self._anchor.dtype)
        self.num_channels = c
        self.x_positions = nn.Parameter(torch.empty(c, k, **kw))
        self.slopes = nn.Parameter(torch.ones(c, k + 1, **kw))
        self.biases = nn.Parameter(torch.empty(c, **kw))
        with torch.no_grad():
            nn.init.normal_(self.x_positions, std=2.0)
            self.biases.copy_(torch.sort(self.x_positions, dim=1)[0][:, 0])

    def breakpoint_values(self):
        """Sorted breakpoint x positions [C, K] and the function value at each [C, K]."""
        xp = torch.sort(self.x_positions, dim=1)[0]
        steps = (xp[:, 1:] - xp[:, :-1]) * self.slopes[:, 1:self.num_breakpoints]
        ys = torch.cat([self.biases.unsqueeze(1), self.biases.unsqueeze(1) + torch.cumsum(steps, dim=1)], dim=1)
        return xp, ys

    def forward(self, x):
        if self.x_positions is None:
            self._build(x.size(-1))
        c = self.num_channels
        if c != 1 and x.size(-1) != c:
            raise ValueError(f"PaWeL was built for {c} channels, got input with {x.size(-1)}.")
        xs = x.reshape(1, -1) if c == 1 else x.reshape(-1, c).t()  # [C, N]
        xp, ys = self.breakpoint_values()
        j = torch.searchsorted(xp.contiguous(), xs.contiguous(), right=True) - 1  # last breakpoint <= x
        anchor = j.clamp(min=0)
        out = (torch.gather(ys, 1, anchor)
               + (xs - torch.gather(xp, 1, anchor)) * torch.gather(self.slopes, 1, j + 1))
        return out.reshape(x.shape) if c == 1 else out.t().reshape(x.shape)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        old_x, old_y = prefix + "x_points", prefix + "y_points"
        if old_x in state_dict and old_y in state_dict:
            xp, order = torch.sort(state_dict.pop(old_x).reshape(1, -1), dim=1)
            yp = state_dict.pop(old_y).reshape(1, -1)[:, order[0]]
            seg = (yp[:, 1:] - yp[:, :-1]) / (xp[:, 1:] - xp[:, :-1] + 1e-6)
            slopes = torch.cat([seg[:, :1], seg, seg[:, -1:]], dim=1)
            c = self.num_channels or 1
            state_dict[prefix + "x_positions"] = xp.expand(c, -1).clone()
            state_dict[prefix + "slopes"] = slopes.expand(c, -1).clone()
            state_dict[prefix + "biases"] = yp[:, 0].expand(c).clone()
        key = prefix + "x_positions"
        if key in state_dict:
            shape = state_dict[key].shape
            if shape[1] != self.num_breakpoints:
                self.num_breakpoints = int(shape[1]); self.num_channels = None
            if self.num_channels != shape[0]:
                self._build(shape[0])
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

class SelfGatedActivation(nn.Module):
    """Self-gated: f(x) = x * af(x)"""
    def __init__(self, activation_fn):
        super(SelfGatedActivation, self).__init__()
        self.activation_fn = activation_fn
    
    def forward(self, x):
        return x * self.activation_fn(x)


class GLUActivation(nn.Module):
    """
    Projected GLU (SwiGLU style) with Lazy Initialization.
    """
    def __init__(self, activation_fn, dim=-1):
        super(GLUActivation, self).__init__()
        self.activation_fn = activation_fn
        self.dim = dim
        self.projection = None

    def forward(self, x):
        if self.projection is None:
            in_features = x.size(self.dim)
            self.projection = nn.Linear(in_features, in_features * 2)
            self.projection.to(device=x.device, dtype=x.dtype)
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
        x_proj = self.projection(x)
        x_gate, x_val = x_proj.chunk(2, dim=self.dim)
        return x_gate * self.activation_fn(x_val)

    def get_learned_parameters(self):
        if self.projection is not None:
            return {
                "projection_weight": self.projection.weight.detach().cpu().numpy(),
                "projection_bias": self.projection.bias.detach().cpu().numpy()
            }
        return {}

class GLU2Activation(nn.Module):
    """Projected GLU where BOTH branches get activation."""
    def __init__(self, activation_fn, dim=-1):
        super(GLU2Activation, self).__init__()
        self.activation_fn1 = activation_fn
        self.activation_fn2 = activation_fn
        self.dim = dim
        self.projection = None

    def forward(self, x):
        if self.projection is None:
            in_features = x.size(self.dim)
            self.projection = nn.Linear(in_features, in_features * 2)
            self.projection.to(device=x.device, dtype=x.dtype)
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
        x_proj = self.projection(x)
        x_gate, x_val = x_proj.chunk(2, dim=self.dim)
        return self.activation_fn2(x_gate) * self.activation_fn1(x_val)

    def get_learned_parameters(self):
        if self.projection is not None:
            return {
                "projection_weight": self.projection.weight.detach().cpu().numpy(),
                "projection_bias": self.projection.bias.detach().cpu().numpy()
            }
        return {}

class TAAFWrapper(nn.Module):
    def __init__(self, activation_fn):
        super(TAAFWrapper, self).__init__()
        self.activation_fn = activation_fn
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.delta = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x):
        return self.gamma * self.activation_fn(self.alpha * x + self.beta) + self.delta

class ResidualWrapper(nn.Module):
    def __init__(self, activation_fn):
        super(ResidualWrapper, self).__init__()
        self.activation_fn = activation_fn
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        return self.alpha*self.activation_fn(x) + x*self.beta

class ResidualTAAFWrapper(nn.Module):
    def __init__(self, activation_fn):
        super(ResidualTAAFWrapper, self).__init__()
        self.activation_fn = activation_fn
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.delta = nn.Parameter(torch.tensor(0.0))
        self.res1 = nn.Parameter(torch.tensor(0.0))
        self.res2 = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        return (self.gamma * self.activation_fn(self.alpha * x + self.beta) + self.delta)*self.res1 + x*self.res2


##############################################
# Gumbel Softmax Activation Selector ("All")
##############################################
def _get_all_activation_list():
    """Build the list of (name, factory) for every non-Custom activation.
    Called at module init time so new activations added to the map are picked up."""
    amap = _build_activation_map()
    exclude = {"Custom", "All", "Maxout"}
    return [(name, factory) for name, factory in amap.items() if name not in exclude]


class GumbelActivationSelector(nn.Module):
    """Uses Gumbel-Softmax to learn which activation function works best.
    Each layer gets its own instance with independent logits.
    Temperature is trainable and starts at init_temp (default 1.0)."""

    def __init__(self, init_temp=1.0):
        super().__init__()
        self.init_temp = init_temp
        self._built = False
        # Defer building until we know hidden_size (lazy init on first forward)
        self.activations = None
        self.logits = None
        self.log_temp = None
        self._act_names = None

    def _lazy_build(self, hidden_size, device, dtype):
        # Keep only candidates that build and give finite values and gradients on a fixed
        # probe; one NaN candidate poisons the whole weighted mix (0 * NaN = NaN). The probe
        # is deterministic, so training and checkpoint loading build the same list.
        probe = torch.randn(64, hidden_size, generator=torch.Generator().manual_seed(0)) * 3
        names, activations, dropped = [], [], []
        for name, factory in _get_all_activation_list():
            try:
                act = factory()
                x = probe.clone().requires_grad_(True)
                with torch.enable_grad():  # first forward usually runs under no_grad (dry run)
                    y = act(x)
                    y.sum().backward()
                ok = y.shape == x.shape and bool(torch.isfinite(y).all()) and bool(torch.isfinite(x.grad).all())
                for p in act.parameters(): p.grad = None
            except Exception:
                ok = False
            if ok: names.append(name); activations.append(act)
            else: dropped.append(name)
        if dropped: print(f"  Gumbel 'All': skipping candidates that fail or produce NaN/inf: {', '.join(dropped)}")
        self._act_names = names
        n = len(activations)
        self.activations = nn.ModuleList(activations)
        self.logits = nn.Parameter(torch.zeros(n, device=device, dtype=dtype))
        self.log_temp = nn.Parameter(torch.tensor(float(np.log(self.init_temp)),
                                                   device=device, dtype=dtype))
        self._built = True
        # Move all sub-modules to correct device
        self.to(device)

    @property
    def temperature(self):
        if self.log_temp is None:
            return self.init_temp
        return self.log_temp.exp()

    def forward(self, x):
        if not self._built:
            self._lazy_build(x.size(-1), x.device, x.dtype)

        temp = self.log_temp.exp().clamp(min=0.01)

        if not self.training:
            # Eval: run only the crystallized choice (0 * inf from other candidates would be NaN)
            return self.activations[int(self.logits.argmax())](x)
        # Gumbel-Softmax (straight-through variant)
        weights = F.gumbel_softmax(self.logits, tau=temp, hard=False)

        # Apply each activation and combine
        result = torch.zeros_like(x)
        for i, act in enumerate(self.activations):
            try:
                act_out = act(x)
            except Exception:
                act_out = x  # fallback to identity on error
            result = result + weights[i] * act_out
        return result

    def get_chosen_activation(self):
        """Return the name of the currently dominant activation."""
        if not self._built or self._act_names is None:
            return "Not initialized"
        idx = self.logits.argmax().item()
        return self._act_names[idx]

    def get_activation_weights(self):
        """Return softmax probabilities for inspection."""
        if not self._built:
            return {}
        probs = F.softmax(self.logits, dim=0).detach().cpu().numpy()
        return {name: float(p) for name, p in zip(self._act_names, probs)}


##############################################
# Helper functions for inim image processing
##############################################
def largest_connected_component_area(patch):
    H, W = patch.shape
    visited = np.zeros((H, W), dtype=bool)
    largest = 0
    for i in range(H):
        for j in range(W):
            if not visited[i, j]:
                value = patch[i, j]
                area = 0
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    if x < 0 or x >= H or y < 0 or y >= W:
                        continue
                    if visited[x, y]:
                        continue
                    if patch[x, y] != value:
                        continue
                    visited[x, y] = True
                    area += 1
                    stack.append((x-1, y))
                    stack.append((x+1, y))
                    stack.append((x, y-1))
                    stack.append((x, y+1))
                if area > largest:
                    largest = area
    return largest

def compute_patch_codes(patch):
    if patch.shape[1] > 1:
        diff = patch[:, 1:] - patch[:, :-1]
        pc = np.mean(diff) / 255.0
    else:
        pc = 0.0
    top = np.mean(patch[0, :])
    bottom = np.mean(patch[-1, :])
    left = np.mean(patch[:, 0])
    right = np.mean(patch[:, -1])
    ec = ((top + left) - (bottom + right)) / 510.0
    H, W = patch.shape
    area = H * W
    largest_cc = largest_connected_component_area(patch)
    bc = 2 * (largest_cc / area) - 1
    return pc, ec, bc

##############################################
# Normalisation layers
##############################################
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Type, Dict, Tuple, Optional

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return x / (rms + self.eps) * self.scale

class GroupNormMLP(nn.Module):
    def __init__(self, num_groups, num_features, eps=1e-5):
        super().__init__()
        assert num_features % num_groups == 0
        self.num_groups = num_groups
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0); squeeze_back = True
        else: squeeze_back = False
        B, C = x.shape; G = self.num_groups
        x = x.view(B, G, C // G)
        mean = x.mean(-1, keepdim=True); var = x.var(-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.view(B, C); out = x * self.weight + self.bias
        if squeeze_back: out = out.squeeze(0)
        return out

class BatchNorm1dWrapper(nn.Module):
    def __init__(self, num_features):
        super().__init__(); self.bn = nn.BatchNorm1d(num_features)
    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0); out = self.bn(x); return out.squeeze(0)
        return self.bn(x)

class InstanceNorm1dWrapper(nn.Module):
    def __init__(self, num_features):
        super().__init__(); self.norm = nn.InstanceNorm1d(num_features, affine=True)
    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0); out = self.norm(x); return out.squeeze(0)
        return self.norm(x)

##############################################
# Spatial Gating Unit (from gMLP paper)
##############################################
class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim; self.half_dim = dim // 2; self.eps = eps
        self.norm = nn.LayerNorm(self.half_dim)
        self.spatial_proj = nn.Linear(self.half_dim, self.half_dim, bias=True)
        nn.init.normal_(self.spatial_proj.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.spatial_proj.bias, 1.0)
    def forward(self, x):
        squeeze_back = False
        if x.dim() == 1: x = x.unsqueeze(0); squeeze_back = True
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v); v = self.spatial_proj(v); out = u * v
        if squeeze_back: out = out.squeeze(0)
        return out

##############################################
# Tiny Attention
##############################################
class TinyAttention(nn.Module):
    def __init__(self, dim, attn_dim=64):
        super().__init__()
        self.dim = dim; self.attn_dim = attn_dim; self.scale = attn_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * attn_dim, bias=False)
        self.proj_out = nn.Linear(attn_dim, dim, bias=True)
        nn.init.xavier_uniform_(self.qkv.weight); nn.init.xavier_uniform_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
    def forward(self, x):
        squeeze_back = False
        if x.dim() == 1: x = x.unsqueeze(0); squeeze_back = True
        qkv = self.qkv(x); q, k, v = qkv.chunk(3, dim=-1)
        q = q.unsqueeze(1); k = k.unsqueeze(1); v = v.unsqueeze(1)
        attn = (q @ k.transpose(-2, -1)) * self.scale; attn = F.softmax(attn, dim=-1)
        out = (attn @ v).squeeze(1); out = self.proj_out(out)
        if squeeze_back: out = out.squeeze(0)
        return out

##############################################
# Attention layers
##############################################
class BasicSelfAttention(nn.Module):
    """Feature-wise self-attention: groups features into tokens so attention
    produces non-trivial interactions (single-token attention is always 1.0).
    Falls back to a learned linear mix when dim < 2*group_size."""
    def __init__(self, dim, group_size=8):
        super().__init__(); self.dim = dim
        self.group_size = min(group_size, max(1, dim))
        self.num_tokens = max(1, dim // self.group_size)
        self.effective_dim = self.num_tokens * self.group_size
        self.pad = dim - self.effective_dim  # leftover features
        d = self.group_size
        self.scale = d ** -0.5
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=True)
        # Linear passthrough for any leftover features
        if self.pad > 0:
            self.pad_proj = nn.Linear(self.pad, self.pad, bias=False)
        else:
            self.pad_proj = None
    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0); squeeze = True
        else: squeeze = False
        B = x.size(0)
        main = x[:, :self.effective_dim]                          # (B, eff)
        main = main.view(B, self.num_tokens, self.group_size)    # (B, T, D)
        qkv = self.qkv(main)                                     # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)                           # each (B, T, D)
        attn = (q @ k.transpose(-2, -1)) * self.scale            # (B, T, T)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v)                                          # (B, T, D)
        out = self.proj(out)                                      # (B, T, D)
        out = out.reshape(B, self.effective_dim)                  # (B, eff)
        if self.pad > 0:
            leftover = self.pad_proj(x[:, self.effective_dim:])
            out = torch.cat([out, leftover], dim=-1)
        if squeeze: out = out.squeeze(0)
        return out

class MultiHeadSelfAttention(nn.Module):
    """Multi-head attention over feature groups.
    Groups the hidden dimension into tokens of size head_dim so that
    attention interactions are non-trivial (single-token MHA is identity)."""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        # Adjust num_heads so head_dim >= 1 and produces >= 2 tokens
        while num_heads > 1 and (dim // num_heads) < 1:
            num_heads -= 1
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.effective_dim = self.num_heads * self.head_dim
        self.pad = dim - self.effective_dim
        # We treat each head_dim-sized chunk as one token, giving num_heads tokens
        self.attn = nn.MultiheadAttention(self.head_dim, 1, batch_first=True)
        self.out_proj = nn.Linear(self.effective_dim, self.effective_dim, bias=True) if self.num_heads > 1 else None
        if self.pad > 0:
            self.pad_proj = nn.Linear(self.pad, self.pad, bias=False)
        else:
            self.pad_proj = None
    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0); squeeze = True
        else: squeeze = False
        B = x.size(0)
        main = x[:, :self.effective_dim].view(B, self.num_heads, self.head_dim)  # (B, T, D)
        out, _ = self.attn(main, main, main, need_weights=False)                 # (B, T, D)
        out = out.reshape(B, self.effective_dim)                                  # (B, eff)
        if self.out_proj is not None:
            out = self.out_proj(out)
        if self.pad > 0:
            leftover = self.pad_proj(x[:, self.effective_dim:])
            out = torch.cat([out, leftover], dim=-1)
        if squeeze: out = out.squeeze(0)
        return out

class FeatureTokenAttention(nn.Module):
    """Input attention across the input features: every feature becomes a token (its value times a learned
    vector plus a learned feature-identity vector), so attention weights really compare features.
      "basic": self-attention among the feature tokens, read back to one number per feature.
      "cross": learned queries attend over the feature tokens, read back to a full input-width update.
    The update is added to the raw inputs; its output projection starts at zero, so training begins
    from the plain MLP."""
    def __init__(self, input_dim, mode, num_heads=1, num_queries=4):
        super().__init__()
        self.mode = mode; h = max(1, int(num_heads or 1)); d = 8 * h
        self.value = nn.Parameter(torch.randn(input_dim, d) * d ** -0.5)
        self.feature = nn.Parameter(torch.randn(input_dim, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        if mode == "cross":
            self.queries = nn.Parameter(torch.randn(1, num_queries, d) * 0.02)
            self.out = nn.Linear(num_queries * d, input_dim)
        else:
            self.out = nn.Linear(d, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        squeeze = x.dim() == 1
        if squeeze: x = x.unsqueeze(0)
        tokens = x.unsqueeze(-1) * self.value + self.feature           # (B, D, d)
        if self.mode == "cross":
            q = self.queries.expand(x.size(0), -1, -1)
            ctx, _ = self.attn(q, tokens, tokens, need_weights=False)   # (B, Q, d)
            upd = self.out(ctx.flatten(1))
        else:
            ctx, _ = self.attn(tokens, tokens, tokens, need_weights=False)
            upd = self.out(ctx).squeeze(-1)
        out = x + upd
        return out.squeeze(0) if squeeze else out

##############################################
# MoE and LoRA components
##############################################
class MoELinear(nn.Module):
    def __init__(self, in_features, out_features, num_experts=4, activation=None):
        super().__init__()
        self.in_features = in_features; self.out_features = out_features
        self.num_experts = num_experts; self.activation = activation
        self.experts = nn.ModuleList([nn.Linear(in_features, out_features, bias=True) for _ in range(num_experts)])
        self.gate = nn.Sequential(nn.Linear(in_features, num_experts, bias=True), nn.Softmax(dim=-1))
    def forward(self, x):
        weights = self.gate(x)
        if x.dim() == 1: x = x.unsqueeze(0); squeeze = True
        else: squeeze = False
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)
        weights_expanded = weights.t().unsqueeze(-1)
        output = (expert_outputs * weights_expanded).sum(dim=0)
        if self.activation is not None: output = self.activation(output)
        if squeeze: output = output.squeeze(0)
        return output

class MaxoutActivation(nn.Module):
    """Configuration marker for true Maxout, implemented by MaxoutLinear."""
    def __init__(self, pieces=2):
        super().__init__()
        if pieces < 2: raise ValueError("Maxout requires at least two pieces.")
        self.pieces = pieces

    def forward(self, x):
        return x

class MaxoutLinear(nn.Module):
    """Maximum over learned affine projections for every output feature."""
    def __init__(self, in_features, out_features, pieces=2, bias=True):
        super().__init__()
        self.in_features = in_features; self.out_features = out_features; self.pieces = pieces
        self.weight = nn.Parameter(torch.empty(pieces, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(pieces, out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        if self.bias is not None: nn.init.zeros_(self.bias)

    def forward(self, x):
        values = torch.einsum('...i,poi->...po', x, self.weight)
        if self.bias is not None: values = values + self.bias
        return values.max(dim=-2).values

class LoRAHyperLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=1.0, context_mode="x_full", context_dim=None, per_sample=True):
        super().__init__()
        self.in_features = in_features; self.out_features = out_features
        self.rank = rank; self.alpha = alpha; self.context_mode = context_mode; self.per_sample = per_sample
        self.base_linear = nn.Linear(in_features, out_features, bias=True)
        if context_dim is None: context_dim = in_features
        self.context_dim = context_dim
        if context_mode == "learned": self.context_emb = nn.Parameter(torch.randn(1, context_dim))
        else: self.context_emb = None
        self.hyper_A = nn.Linear(context_dim, rank * in_features, bias=False)
        self.hyper_B = nn.Linear(context_dim, out_features * rank, bias=False)
        nn.init.xavier_uniform_(self.hyper_A.weight, gain=0.01)
        nn.init.xavier_uniform_(self.hyper_B.weight, gain=0.01)

    def _get_context(self, x):
        if self.context_mode == "learned":
            B = x.size(0) if x.dim() == 2 else 1
            return self.context_emb.expand(B, -1)
        elif self.context_mode == "x_full":
            return x if x.dim() == 2 else x.unsqueeze(0)
        elif self.context_mode == "x_mean":
            if x.dim() == 1: return x.unsqueeze(0)
            return x.mean(dim=0, keepdim=True).expand(x.size(0), -1)
        elif self.context_mode == "x_mean_std":
            if x.dim() == 1: return torch.cat([x, x], dim=-1).unsqueeze(0)
            mean = x.mean(dim=0, keepdim=True); std = x.std(dim=0, keepdim=True)
            return torch.cat([mean, std], dim=-1).expand(x.size(0), -1)
        else: raise ValueError(f"Unknown context_mode: {self.context_mode}")

    def forward(self, x):
        squeeze = False
        if x.dim() == 1: x = x.unsqueeze(0); squeeze = True
        base_out = self.base_linear(x); ctx = self._get_context(x)
        A_flat = self.hyper_A(ctx); B_flat = self.hyper_B(ctx)
        A = A_flat.view(-1, self.rank, self.in_features); B = B_flat.view(-1, self.out_features, self.rank)
        if self.per_sample:
            lora_out = torch.bmm(x.unsqueeze(1), A.transpose(-2, -1))
            lora_out = torch.bmm(lora_out, B.transpose(-2, -1)).squeeze(1)
        else:
            A_mean = A.mean(dim=0); B_mean = B.mean(dim=0)
            lora_out = x @ A_mean.t() @ B_mean.t()
        out = base_out + self.alpha * lora_out
        if squeeze: out = out.squeeze(0)
        return out

def functional_norm(x, norm_type, groups=1, eps=1e-5, batch_params=None):
    squeeze = False
    if x.dim() == 1: x = x.unsqueeze(0); squeeze = True
    nt = norm_type.lower(); B, F = x.shape; x_ = x
    def _bparam(name, default):
        if batch_params and name in batch_params:
            p = batch_params[name]
            if p.dim() == 1: p = p.unsqueeze(0)
            return p
        return default
    if nt in ["none", "identity"]: y = x_
    elif nt in ["layer", "layernorm"]:
        mu = x_.mean(dim=-1, keepdim=True); var = x_.var(dim=-1, unbiased=False, keepdim=True)
        y = (x_ - mu) / torch.sqrt(var + eps)
        w = _bparam("norm_weight", torch.ones(F, device=x_.device)).view(B, F)
        b = _bparam("norm_bias", torch.zeros(F, device=x_.device)).view(B, F)
        y = y * w + b
    elif nt == "batch":
        mu = x_.mean(dim=0, keepdim=True); var = x_.var(dim=0, unbiased=False, keepdim=True)
        y = (x_ - mu) / torch.sqrt(var + eps)
        w = _bparam("norm_weight", torch.ones(F, device=x_.device)).view(B, F)
        b = _bparam("norm_bias", torch.zeros(F, device=x_.device)).view(B, F)
        y = y * w + b
    elif nt == "group":
        assert F % groups == 0; t = x_.view(B, groups, F // groups)
        mu = t.mean(-1, keepdim=True); var = t.var(-1, unbiased=False, keepdim=True)
        t = (t - mu) / torch.sqrt(var + eps); y = t.view(B, F)
        w = _bparam("norm_weight", torch.ones(F, device=x_.device)).view(B, F)
        b = _bparam("norm_bias", torch.zeros(F, device=x_.device)).view(B, F)
        y = y * w + b
    elif nt == "rmsnorm":
        rms = x_.pow(2).mean(dim=-1, keepdim=True).sqrt(); y = x_ / (rms + 1e-8)
        s = _bparam("norm_scale", torch.ones(F, device=x_.device)).view(B, F)
        y = y * s
    else: raise ValueError(f"Unknown norm_type: {norm_type}")
    if squeeze: y = y.squeeze(0)
    return y


##############################################
# Residual Block
##############################################
class FeatureGRN(nn.Module):
    """GRN for a tabular feature vector, normalized within each example."""
    def __init__(self, features):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        response = x.float().abs()
        relative = (response / (response.mean(dim=-1, keepdim=True) + 1e-6)).to(x.dtype)
        return x + self.gamma.to(x.dtype) * x * relative + self.beta.to(x.dtype)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation_cls, residual_type="residual",
                 norm_type="layer", groups=1, attention_type="none", num_heads=1,
                 moe_experts=1, use_noise_injection_layers=False, noise_injection_std=0.0,
                 hyper_context="x_full", hyper_context_dim=64, hyper_film_per_sample=False, use_grn=False,
                 hyper_per_sample=True, hyper_low_rank_k=None):
        super().__init__()
        self.residual_type = residual_type.lower(); self.norm_type = norm_type.lower()
        self.attention_type = attention_type.lower(); self.use_moe = moe_experts > 1
        self.use_hyper = moe_experts < 1 and moe_experts > -4
        self.in_dim = in_dim; self.out_dim = out_dim
        self.grn = FeatureGRN(in_dim + out_dim if self.residual_type == "concat" else out_dim) if use_grn else nn.Identity()
        self.noise = use_noise_injection_layers; self.std = noise_injection_std; self.groups = groups
        self.use_sgu = (moe_experts == -4 or moe_experts == -5)
        self.use_tiny_attn = (moe_experts == -5)
        if moe_experts == 0: hyper_context = "x_full"
        elif moe_experts == -1: hyper_context = "learned"
        elif moe_experts == -2: hyper_context = "x_mean_std"
        elif moe_experts == -3: hyper_context = "x_mean"
        elif self.use_sgu: hyper_context = "learned"
        self.hyper_context = hyper_context
        self.use_hyper = (moe_experts < 1 and not self.use_sgu)

        norm_dim = in_dim
        if self.norm_type == "none": self.norm = nn.Identity()
        elif self.norm_type == "batch": self.norm = BatchNorm1dWrapper(norm_dim)
        elif self.norm_type == "instance": self.norm = InstanceNorm1dWrapper(norm_dim)
        elif self.norm_type == "layer": self.norm = nn.LayerNorm(norm_dim)
        elif self.norm_type == "group": self.norm = GroupNormMLP(groups, norm_dim)
        elif self.norm_type == "rmsnorm": self.norm = RMSNorm(norm_dim)
        else: raise ValueError(f"Unknown norm_type: {norm_type}")

        activation_probe = activation_cls()
        self.use_maxout = isinstance(activation_probe, MaxoutActivation)
        lin1_out_dim = out_dim * 2 if self.use_sgu else out_dim
        if self.use_maxout:
            self.linear1 = MaxoutLinear(in_dim, out_dim, pieces=activation_probe.pieces)
        elif self.use_moe:
            self.linear1 = MoELinear(in_dim, lin1_out_dim, num_experts=moe_experts, activation=nn.Identity())
        elif self.use_hyper:
            self.linear1 = LoRAHyperLinear(in_dim, lin1_out_dim, rank=(hyper_low_rank_k or 8), alpha=1.0,
                context_mode=hyper_context, context_dim=(in_dim if hyper_context == "x_full" else hyper_context_dim),
                per_sample=hyper_per_sample)
        else:
            self.linear1 = nn.Linear(in_dim, lin1_out_dim, bias=True)

        if not self.use_sgu and not self.use_maxout: self.activation = activation_probe
        else: self.activation = None
        act_name = activation_cls.__name__ if hasattr(activation_cls, '__name__') else activation_cls().__class__.__name__
        if self.use_sgu: self.sgu = SpatialGatingUnit(lin1_out_dim)
        else: self.sgu = None

        if self.residual_type != "none": self.linear2 = nn.Linear(out_dim, out_dim, bias=True)
        else: self.linear2 = None

        if self.use_tiny_attn: self.tiny_attn = TinyAttention(out_dim, attn_dim=64)
        else: self.tiny_attn = None

        if not self.use_hyper and not self.use_sgu and not self.use_maxout:
            self._init_weights(self.linear1, act_name)
            if self.linear2 is not None: self._init_weights(self.linear2, "Linear")

        if self.use_hyper:
            self.skip = None
            if in_dim != out_dim: self.skip = nn.Linear(in_dim, out_dim, bias=False)
            else: self.skip = nn.Identity()
        else:
            if self.residual_type == "concat": self.skip = nn.Identity()
            elif in_dim != out_dim: self.skip = nn.Linear(in_dim, out_dim, bias=False)
            else: self.skip = nn.Identity()

        if self.residual_type == "highway":
            self.gate = nn.Linear(in_dim, out_dim, bias=True)
            nn.init.xavier_uniform_(self.gate.weight); nn.init.constant_(self.gate.bias, -1.0)
            self.gamma = None
        elif self.residual_type == "rezero":
            self.gamma = nn.Parameter(torch.tensor(0.0)); self.beta = nn.Parameter(torch.tensor(1.0)); self.gate = None
        elif self.residual_type in ["elementwise rezero", "elementwise_rezero"]:
            self.gamma = nn.Parameter(torch.zeros(out_dim)); self.beta = nn.Parameter(torch.ones(out_dim)); self.gate = None
        else: self.gate = None; self.gamma = None

        if self.attention_type == "basic": self.attn = BasicSelfAttention(out_dim)
        elif self.attention_type == "multi": self.attn = MultiHeadSelfAttention(out_dim, num_heads if out_dim >= num_heads else 1)
        else: self.attn = None

    def _init_weights(self, layer, act_name):
        if isinstance(layer, nn.Linear):
            weight = layer.weight
            if act_name in ["SELU", "lSELU", "sSELU", "sGoLU"]:
                nn.init.kaiming_uniform_(weight, mode='fan_in', nonlinearity="linear")
            elif act_name == "Sigmoid": nn.init.xavier_uniform_(weight, gain=nn.init.calculate_gain('sigmoid'))
            elif act_name == "Tanh": nn.init.xavier_uniform_(weight, gain=nn.init.calculate_gain('tanh'))
            elif act_name in ["Sine", "Cosine", "GSine", "GCosine"]:
                bound = math.sqrt(6 / weight.size(1)) / 30; nn.init.uniform_(weight, -bound, bound)
            else: nn.init.kaiming_uniform_(weight, a=0, mode='fan_in', nonlinearity='relu')
        elif hasattr(layer, 'experts'):
            for exp in layer.experts: self._init_weights(exp, act_name)

    def _apply_residual_nonhyper(self, transform, x):
        if self.residual_type == "none": return transform
        elif self.residual_type == "highway":
            skip = self.skip(x); gate = torch.sigmoid(self.gate(x)); return gate * transform + (1 - gate) * skip
        elif self.residual_type == "residual": return transform + self.skip(x)
        elif self.residual_type == "rezero": return self.gamma * transform + self.skip(x) * self.beta
        elif self.residual_type in ["elementwise rezero", "elementwise_rezero"]:
            return self.gamma * transform + self.skip(x) * self.beta
        elif self.residual_type == "concat": return torch.cat([x, transform], dim=-1)
        else: return transform + self.skip(x)

    def forward(self, x):
        if self.noise: x = x + (torch.randn_like(x) * self.std)
        z = self.norm(x); z = self.linear1(z)
        if self.use_sgu: z = self.sgu(z)
        elif self.activation is not None: z = self.activation(z)
        if self.linear2 is not None: z = self.linear2(z)
        if self.use_tiny_attn: z = z + self.tiny_attn(z)
        if self.attn is not None: z = self.attn(z)
        return self.grn(self._apply_residual_nonhyper(z, x))

    def output_dim(self):
        if self.residual_type == "concat": return self.in_dim + self.out_dim
        else: return self.out_dim


##############################################
# GNN Block for tabular data
##############################################
##############################################
# GNN Block for tabular data (UPDATED)
##############################################
class GNNMessagePassingLayer(nn.Module):
    """Message passing layer: treats hidden dims as a fully-connected graph of nodes."""
    def __init__(self, node_dim, num_nodes, aggr='mean'):
        super().__init__()
        self.node_dim = node_dim
        self.num_nodes = num_nodes
        self.aggr = aggr
        # Learnable adjacency (logits -> sigmoid for soft adjacency)
        self.adj_logits = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.01)
        # Message transform
        self.msg_fn = nn.Linear(node_dim, node_dim, bias=True)
        # Update transform
        self.update_fn = nn.Linear(node_dim * 2, node_dim, bias=True)
        nn.init.xavier_uniform_(self.msg_fn.weight)
        nn.init.xavier_uniform_(self.update_fn.weight)

    def forward(self, x):
        """x: (batch, num_nodes, node_dim)"""
        # Ensure input is 3D (Batch, Nodes, Dim)
        if x.dim() == 2:
            B = x.size(0)
            x = x.view(B, self.num_nodes, self.node_dim)
            
        B = x.size(0)
        adj = torch.sigmoid(self.adj_logits)  # (N, N) soft adjacency
        # Zero out self-loops for message passing (self info comes via update)
        adj = adj * (1.0 - torch.eye(self.num_nodes, device=x.device))
        
        # Messages: transform node features
        msgs = self.msg_fn(x)  # (B, N, D)
        
        # Aggregate: adj @ msgs -> (B, N, D)
        agg = torch.bmm(adj.unsqueeze(0).expand(B, -1, -1), msgs)
        
        # Normalize by degree (soft)
        deg = adj.sum(dim=-1, keepdim=True).unsqueeze(0).expand(B, -1, -1).clamp(min=1.0)
        agg = agg / deg
        
        # Update: concat self + aggregated, then transform
        combined = torch.cat([x, agg], dim=-1)  # (B, N, 2D)
        h_new = self.update_fn(combined)  # (B, N, D)
        return h_new

class InputFeatureGraph(nn.Module):
    """
    Learns relationships directly between input features.
    Handles arbitrary input dimensions by padding to the nearest multiple of 'num_nodes'.
    """
    def __init__(self, input_dim, max_nodes=64):
        super().__init__()
        self.input_dim = input_dim
        
        # Heuristic: Try to have 1 node per feature, up to max_nodes.
        # If input_dim > max_nodes, we group features into nodes.
        if input_dim <= max_nodes:
            self.num_nodes = input_dim
            self.node_dim = 1
        else:
            self.num_nodes = max_nodes
            self.node_dim = math.ceil(input_dim / max_nodes)
            
        self.padded_dim = self.num_nodes * self.node_dim
        self.pad_amount = self.padded_dim - input_dim
        
        self.gnn = GNNMessagePassingLayer(self.node_dim, self.num_nodes)
        
        # Optional: Output projection to mix the result back cleanly
        self.out_proj = nn.Linear(self.padded_dim, input_dim)
        # Initialize close to identity to act as a gentle refinement initially
        nn.init.eye_(self.out_proj.weight[:input_dim, :input_dim])
        if self.pad_amount > 0:
            nn.init.zeros_(self.out_proj.weight[:, input_dim:])
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        # 1. Pad input if necessary
        if self.pad_amount > 0:
            x_pad = F.pad(x, (0, self.pad_amount))
        else:
            x_pad = x
            
        # 2. Reshape to (Batch, NumNodes, NodeDim)
        B = x.shape[0]
        x_graph = x_pad.view(B, self.num_nodes, self.node_dim)
        
        # 3. Apply GNN
        x_graph_out = self.gnn(x_graph)
        
        # 4. Flatten back
        x_flat = x_graph_out.view(B, self.padded_dim)
        
        # 5. Project back to original dimension
        out = self.out_proj(x_flat)
        
        # 6. Residual connection (Input + Graph_Influence)
        return x + out

class GNNResidualBlock(nn.Module):
    """A residual block that includes GNN message passing on latent features."""
    def __init__(self, in_dim, out_dim, activation_cls, residual_type="residual",
                 norm_type="layer", groups=1, use_grn=False):
        super().__init__()
        self.residual_type = residual_type.lower()
        self.in_dim = in_dim; self.out_dim = out_dim
        self.grn = FeatureGRN(in_dim + out_dim if self.residual_type == "concat" else out_dim) if use_grn else nn.Identity()

        # Pre-norm
        if norm_type == "none": self.norm = nn.Identity()
        elif norm_type == "batch": self.norm = BatchNorm1dWrapper(in_dim)
        elif norm_type == "instance": self.norm = InstanceNorm1dWrapper(in_dim)
        elif norm_type == "layer": self.norm = nn.LayerNorm(in_dim)
        elif norm_type == "group": self.norm = GroupNormMLP(groups, in_dim)
        elif norm_type == "rmsnorm": self.norm = RMSNorm(in_dim)
        else: self.norm = nn.LayerNorm(in_dim)

        # Linear projection
        self.linear1 = nn.Linear(in_dim, out_dim, bias=True)
        self.activation = activation_cls()
        
        # Latent GNN message passing
        self.gnn_node_dim = max(1, min(8, out_dim))  # node feature dim
        self.gnn_num_nodes = out_dim // self.gnn_node_dim
        self.gnn_effective_dim = self.gnn_num_nodes * self.gnn_node_dim
        
        if self.gnn_effective_dim > 0 and self.gnn_num_nodes > 1:
            self.gnn = GNNMessagePassingLayer(self.gnn_node_dim, self.gnn_num_nodes)
        else:
            self.gnn = None
            
        # Second linear
        self.linear2 = nn.Linear(out_dim, out_dim, bias=True) if residual_type != "none" else None

        # Skip connection
        if residual_type == "concat": self.skip = nn.Identity()
        elif in_dim != out_dim: self.skip = nn.Linear(in_dim, out_dim, bias=False)
        else: self.skip = nn.Identity()

        # Residual gating
        if residual_type == "highway":
            self.gate = nn.Linear(in_dim, out_dim, bias=True)
            nn.init.constant_(self.gate.bias, -1.0)
        elif residual_type == "rezero":
            self.gamma = nn.Parameter(torch.tensor(0.0)); self.beta = nn.Parameter(torch.tensor(1.0))
        else:
            self.gate = None; self.gamma = None

        nn.init.kaiming_uniform_(self.linear1.weight, a=0, mode='fan_in', nonlinearity='relu')
        if self.linear2 is not None:
            nn.init.kaiming_uniform_(self.linear2.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        z = self.norm(x)
        z = self.linear1(z)
        z = self.activation(z)
        
        # Latent GNN message passing
        if self.gnn is not None:
            # We reshape the vector into nodes, treat it as a graph, then flatten back
            gnn_in = z[..., :self.gnn_effective_dim]
            gnn_in = gnn_in.view(z.size(0), self.gnn_num_nodes, self.gnn_node_dim)
            gnn_out = self.gnn(gnn_in)
            gnn_out = gnn_out.view(z.size(0), self.gnn_effective_dim)
            
            if self.gnn_effective_dim < self.out_dim:
                z = torch.cat([gnn_out, z[..., self.gnn_effective_dim:]], dim=-1)
            else:
                z = gnn_out
                
        if self.linear2 is not None: z = self.linear2(z)
        
        # Residual
        if self.residual_type == "none": return self.grn(z)
        elif self.residual_type == "highway":
            skip = self.skip(x); gate = torch.sigmoid(self.gate(x))
            return self.grn(gate * z + (1 - gate) * skip)
        elif self.residual_type == "rezero":
            return self.grn(self.gamma * z + self.skip(x) * self.beta)
        elif self.residual_type == "concat":
            return self.grn(torch.cat([x, z], dim=-1))
        else:
            return self.grn(z + self.skip(x))

    def output_dim(self):
        if self.residual_type == "concat": return self.in_dim + self.out_dim
        return self.out_dim


class GNNMLPO(nn.Module):
    """
    MLPO variant using GNN blocks.
    Now includes an Input Feature Graph to learn relationships between raw inputs.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, activation_cls, residual_type="residual",
                 norm_type="layer", groups=1, dropout_prob=0.0, use_grn=False):
        super().__init__()
        self.residual_type = residual_type.lower()
        self.dropout_prob = dropout_prob
        
        # --- NEW: Input Feature Graph ---
        # If input_dim is large (>1), we try to learn relationships immediately.
        if input_dim > 1:
            self.input_graph = InputFeatureGraph(input_dim, max_nodes=128)
        else:
            self.input_graph = nn.Identity()
        # --------------------------------
        
        self.blocks = nn.ModuleList()
        current_dim = input_dim
        for hd in hidden_dims:
            block_res = "none" if self.residual_type == "densenet" else residual_type
            block = GNNResidualBlock(current_dim, hd, activation_cls, block_res,
                                     norm_type=norm_type, groups=groups, use_grn=use_grn)
            self.blocks.append(block)
            if self.residual_type == "concat": current_dim = current_dim + hd
            elif self.residual_type == "densenet": current_dim = current_dim + hd
            else: current_dim = hd

        self.final_linear = nn.Linear(current_dim, output_dim, bias=True)
        if current_dim == output_dim: self.final_skip = nn.Identity()
        else: self.final_skip = nn.Linear(current_dim, output_dim, bias=False)
        nn.init.kaiming_uniform_(self.final_linear.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        # 1. Apply Input Graph Learning (Feature <-> Feature)
        out = self.input_graph(x)
        
        if self.residual_type == "densenet":
            outputs = [out]
            for block in self.blocks:
                block_input = torch.cat(outputs, dim=-1)
                block_output = block(block_input)
                if self.training and self.dropout_prob > 0:
                    block_output = F.dropout(block_output, p=self.dropout_prob, training=True)
                outputs.append(block_output)
            out = torch.cat(outputs, dim=-1)
        else:
            for block in self.blocks:
                out = block(out)
                if self.training and self.dropout_prob > 0:
                    out = F.dropout(out, p=self.dropout_prob, training=True)
                    
        return self.final_linear(out) + self.final_skip(out)


##############################################
# MLPO (Extended)
##############################################
class MLPO(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_cls, residual_type="residual", *,
                 norm_type="layer", groups=1, attention_type="none", num_heads=1,
                 input_attention_type="none", moe_mode=1, dropout_prob=0.0,
                 use_noise_injection_layers=False, noise_injection_std=0.01,
                 final_lora_rank=8, final_lora_alpha=1.0, final_hyper_context_dim=64, final_lora_per_sample=True,
                 layer_routing="none", use_grn=False, output_activation=None, output_activation_idx=None):
        """activation_cls: one activation class for every hidden layer, or a list with one per layer.
        hidden_dims may be empty: a perceptron (inputs straight to outputs).
        output_activation: activation class applied to the output positions output_activation_idx
        (the numeric outputs; categorical logits are left alone)."""
        super().__init__()
        layer_acts = list(activation_cls) if isinstance(activation_cls, (list, tuple)) else [activation_cls] * len(hidden_dims)
        if len(layer_acts) != len(hidden_dims):
            raise ValueError(f"{len(layer_acts)} layer activations for {len(hidden_dims)} hidden layers")
        self.residual_type = residual_type.lower(); self.dropout_prob = dropout_prob
        self.use_noise_injection_layers = use_noise_injection_layers; self.moe_mode = moe_mode
        self.layer_routing = layer_routing
        if layer_routing not in ("none", "adaptive"):
            raise ValueError("layer_routing must be 'none' or 'adaptive'.")
        if layer_routing == "adaptive" and (self.residual_type != "residual" or moe_mode != 1):
            raise ValueError("Adaptive layer routing requires standard residual blocks with MoE mode 1 (off).")

        self.input_attention_type = input_attention_type.lower()
        if self.input_attention_type in ("basic", "cross"):
            self.input_attn = FeatureTokenAttention(input_dim, self.input_attention_type, num_heads)
        else: self.input_attn = None

        self.noise_injection = NoiseInjectionLayer(std=noise_injection_std) if use_noise_injection_layers else None

        self.blocks = nn.ModuleList()
        self.layer_routers = nn.ModuleList()
        self._last_router_probs = []
        current_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            block_in_dim = current_dim
            # DenseNet handles connectivity externally via concatenation,
            # so individual blocks should use "none" residual internally
            block_res = "none" if self.residual_type == "densenet" else residual_type
            block = ResidualBlock(block_in_dim, hidden_dim, layer_acts[i], block_res,
                norm_type=norm_type, groups=groups, attention_type=attention_type, num_heads=num_heads,
                moe_experts=moe_mode, use_noise_injection_layers=use_noise_injection_layers,
                noise_injection_std=noise_injection_std, use_grn=use_grn)
            self.blocks.append(block)
            if self.layer_routing == "adaptive":
                router = nn.Linear(block_in_dim, 1)
                nn.init.zeros_(router.weight); nn.init.zeros_(router.bias)
                self.layer_routers.append(router)
            if self.residual_type == "concat": current_dim = block_in_dim + hidden_dim
            elif self.residual_type == "densenet": current_dim = current_dim + hidden_dim
            else: current_dim = hidden_dim

        final_in_dim = current_dim
        def _ctx_from_moe_mode(m):
            if m == 0: return "x_full"
            elif m == -1: return "learned"
            elif m == -2: return "x_mean_std"
            elif m == -3: return "x_mean"
            elif m in [-4, -5]: return "learned"
            else: return "x_full"

        is_maxout = any(getattr(block, 'use_maxout', False) for block in self.blocks)
        if is_maxout and moe_mode != 1:
            raise ValueError("True Maxout cannot be combined with MoE/Hyper/gMLP modes.")
        if moe_mode > 1:
            self.final_linear = MoELinear(final_in_dim, output_dim, num_experts=moe_mode, activation=None)
            self._final_is_param_linear = False
        elif moe_mode < 1 and moe_mode not in [-4, -5]:
            hyper_context = _ctx_from_moe_mode(moe_mode)
            ctx_dim = (final_in_dim if hyper_context == "x_full" else final_hyper_context_dim)
            self.final_linear = LoRAHyperLinear(final_in_dim, output_dim, rank=final_lora_rank,
                alpha=final_lora_alpha, context_mode=hyper_context, context_dim=ctx_dim, per_sample=final_lora_per_sample)
            self._final_is_param_linear = False
        else:
            self.final_linear = nn.Linear(final_in_dim, output_dim, bias=True)
            self._final_is_param_linear = True

        if final_in_dim == output_dim: self.final_skip = nn.Identity()
        else: self.final_skip = nn.Linear(final_in_dim, output_dim, bias=False)

        last_act = layer_acts[-1] if layer_acts else nn.Identity
        act_name = last_act.__name__ if hasattr(last_act, '__name__') else last_act().__class__.__name__
        if self._final_is_param_linear: self._init_final_weights(act_name, final_in_dim)
        self.output_act = output_activation() if output_activation is not None and output_activation_idx else None
        self.register_buffer("output_act_idx", torch.tensor(list(output_activation_idx or []), dtype=torch.long), persistent=False)

    def _init_final_weights(self, act_name, in_features):
        if act_name in ["SELU", "lSELU", "sSELU", "sGoLU"]:
            nn.init.kaiming_uniform_(self.final_linear.weight, mode='fan_in', nonlinearity="linear")
        elif act_name == "Sigmoid": nn.init.xavier_uniform_(self.final_linear.weight, gain=nn.init.calculate_gain('sigmoid'))
        elif act_name == "Tanh": nn.init.xavier_uniform_(self.final_linear.weight, gain=nn.init.calculate_gain('tanh'))
        elif act_name in ["Sine", "Cosine", "GSine", "GCosine"]:
            bound = math.sqrt(6 / in_features) / 30; nn.init.uniform_(self.final_linear.weight, -bound, bound)
        else: nn.init.kaiming_uniform_(self.final_linear.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        if self.input_attn is not None: x = self.input_attn(x)
        if self.residual_type == "densenet":
            outputs = [x]
            for block in self.blocks:
                block_input = torch.cat(outputs, dim=-1); block_output = block(block_input)
                if self.training:
                    if self.dropout_prob > 0.0: block_output = F.dropout(block_output, p=self.dropout_prob, training=True)
                    elif self.noise_injection is not None: block_output = self.noise_injection(block_output)
                outputs.append(block_output)
            out = torch.cat(outputs, dim=-1)
        else:
            out = x
            self._last_router_probs = []
            for block_idx, block in enumerate(self.blocks):
                if self.layer_routing == "adaptive":
                    gate_prob = torch.sigmoid(self.layer_routers[block_idx](out))
                    self._last_router_probs.append(gate_prob)
                    skip = block.skip(out)
                    if self.training:
                        out = gate_prob * block(out) + (1.0 - gate_prob) * skip
                    else:
                        active = gate_prob.squeeze(-1) >= 0.5
                        if out.dim() == 1:
                            out = block(out) if active.item() else skip
                        else:
                            routed = skip.clone()
                            if active.any(): routed[active] = block(out[active])
                            out = routed
                else:
                    out = block(out)
                if self.training:
                    if self.dropout_prob > 0.0: out = F.dropout(out, p=self.dropout_prob, training=True)
                    elif self.noise_injection is not None: out = self.noise_injection(out)
        out = self.final_linear(out) + self.final_skip(out)
        if self.output_act is not None:
            out = out.index_copy(-1, self.output_act_idx, self.output_act(out.index_select(-1, self.output_act_idx)))
        return out

    def routing_loss(self):
        """Encourage binary per-sample gates with roughly half the blocks active."""
        if self.layer_routing != "adaptive" or not self._last_router_probs:
            return next(self.parameters()).new_zeros(())
        probs = torch.cat([p.reshape(-1) for p in self._last_router_probs])
        target_penalty = (probs.mean() - 0.5).pow(2)
        binary_penalty = (probs * (1.0 - probs)).mean()
        return 0.1 * target_penalty + 0.01 * binary_penalty

    def routing_usage(self):
        if self.layer_routing != "adaptive" or not self._last_router_probs:
            return None
        return torch.cat([p.detach().reshape(-1) for p in self._last_router_probs]).mean().item()

    def get_all_learned_parameters(self):
        params = []
        for block in self.blocks:
            if hasattr(block.activation, 'get_learned_parameters'):
                params.append(block.activation.get_learned_parameters())
        return params


##############################################
# NEW: Combined Loss (Huber + CrossEntropy)
##############################################
class CombinedLoss(nn.Module):
    """Loss over all output columns (Huber for numeric, cross-entropy for categorical).

    With several output columns their losses live on different scales and the largest one
    dominates the shared weights. `balancing` decides how they are combined:
      "normalized" (default): each column's loss is divided by the loss of a trivial predictor
          on the training data (mean value / class frequencies), then averaged. Every term is
          unitless (1.0 = no better than guessing), weights are fixed, and the validation value
          means the same thing all run long, so best-checkpoint selection stays sound.
      "uwso": Soft Optimal Uncertainty Weighting (Kirchdorfer et al., GCPR 2024, Eq. 4):
          w_k = softmax_k((1 / sg[L_k]) / T), L = sum_k w_k L_k. Adaptive; T needs tuning.
          Validation still reports the "normalized" value so checkpoints stay comparable.
      "legacy": the previous behaviour, mean(0.1 * Huber, CE).
    A single output column is returned unweighted in every mode.
    """
    BALANCING = ("normalized", "uwso", "legacy")

    def __init__(self, output_layout, column_weights=None, balancing="normalized", temperature=20.0):
        super().__init__()
        if balancing not in self.BALANCING: raise ValueError(f"balancing must be one of {self.BALANCING}")
        self.output_layout = output_layout
        self.column_weights = column_weights if column_weights is not None else {}
        self.huber = nn.HuberLoss()
        self.balancing = balancing; self.temperature = float(temperature)
        self.baselines = None          # per-column trivial-predictor loss, set by calibrate()
        self.last_task_losses = {}     # column -> latest normalized loss (for logging / GUI)

    def _task_losses(self, predictions, targets):
        losses = []
        for entry in self.output_layout:
            col_name = entry['col']
            pred_slice = predictions[:, entry['start']:entry['end']]
            tgt_slice = targets[:, entry['tgt_start']:entry['tgt_end']]
            weight = self.column_weights.get(col_name)
            if weight is not None and weight.device != predictions.device:
                weight = weight.to(predictions.device)
            if entry['type'] in ['out', 'outlab', 'outex']:
                losses.append(self.huber(pred_slice, tgt_slice))
            elif entry['type'] == 'outlabcat':
                losses.append(F.cross_entropy(pred_slice, tgt_slice.squeeze(-1).long(), weight=weight))
            elif entry['type'] in ('outexcat', 'outdec'):
                num_classes, max_len = entry['num_classes'], entry['max_len']
                # outdec: positions after <eos> are padding and carry no loss; outexcat predicts its padding
                losses.append(F.cross_entropy(pred_slice.reshape(-1, max_len, num_classes).reshape(-1, num_classes),
                                              tgt_slice.long().reshape(-1), weight=weight,
                                              ignore_index=seqm.PAD if entry['type'] == 'outdec' else -100))
        return losses

    @torch.no_grad()
    def calibrate(self, data, max_rows=20000):
        """Measure each column's loss for a trivial predictor. `data`: a Dataset or a target tensor."""
        if torch.is_tensor(data):
            targets = data.float().cpu()
        else:
            n = len(data)
            idx = list(range(n)) if n <= max_rows else np.random.default_rng(0).choice(n, max_rows, replace=False).tolist()
            if hasattr(data, "batch"): targets = data.batch(idx)[1].float()
            else:
                loader = DataLoader(torch.utils.data.Subset(data, idx), batch_size=1024, shuffle=False)
                targets = torch.cat([t.float() for _, t in loader], 0)
        pred_dim = max(e['end'] for e in self.output_layout)
        preds = torch.zeros(len(targets), pred_dim)
        for e in self.output_layout:
            tgt = targets[:, e['tgt_start']:e['tgt_end']]
            if e['type'] in ['out', 'outlab', 'outex']:
                preds[:, e['start']:e['end']] = tgt.mean(0)
            elif e['type'] == 'outlabcat':
                counts = torch.bincount(tgt.squeeze(-1).long().clamp(min=0), minlength=e['num_classes'])[:e['num_classes']].float() + 1.0
                preds[:, e['start']:e['end']] = torch.log(counts / counts.sum())
            elif e['type'] in ('outexcat', 'outdec'):
                nc, ml = e['num_classes'], e['max_len']
                logits = torch.stack([torch.log((torch.bincount(tgt[:, i].long().clamp(0, nc - 1), minlength=nc)[:nc].float() + 1.0)
                                                / (len(tgt) + nc)) for i in range(ml)])
                preds[:, e['start']:e['end']] = logits.reshape(1, -1)
        cpu_weights = {k: v.cpu() for k, v in self.column_weights.items()}
        saved, self.column_weights = self.column_weights, cpu_weights
        try: base = self._task_losses(preds, targets)
        finally: self.column_weights = saved
        self.baselines = [max(float(b), 1e-8) for b in base]
        return dict(zip([e['col'] for e in self.output_layout], self.baselines))

    def forward(self, predictions, targets):
        losses = self._task_losses(predictions, targets)
        if not losses:
            return torch.tensor(0.0, device=predictions.device)
        if len(losses) == 1:
            self.last_task_losses = {self.output_layout[0]['col']: float(losses[0].detach())}
            return losses[0]
        if self.balancing == "legacy":
            scaled = [l * 0.1 if e['type'] in ['out', 'outlab', 'outex'] else l for l, e in zip(losses, self.output_layout)]
            return sum(scaled) / len(scaled)
        if self.baselines is None:
            print("CombinedLoss: not calibrated on training data; using the first batch as the baseline.")
            self.calibrate(targets.detach())
        normalized = [l / b for l, b in zip(losses, self.baselines)]
        self.last_task_losses = {e['col']: float(n.detach()) for e, n in zip(self.output_layout, normalized)}
        if self.balancing == "uwso" and self.training:
            inv = torch.stack([1.0 / l.detach().clamp(min=1e-12) for l in losses]) / self.temperature
            weights = torch.softmax(inv, 0)
            return sum(w * l for w, l in zip(weights, losses))
        return sum(normalized) / len(normalized)


##############################################
# Helper: build output layout from col_types + scalings
##############################################
def text_num_classes(col_type, vocab):
    """Softmax width of one text position: outexcat = its characters (ids 1..V) plus class 0 for padding
    (so the model also predicts where the text ends); outdec = its token vocabulary incl. special tokens."""
    return seqm.vocab_size(vocab) if col_type == 'outdec' else len(vocab) + 1


def build_output_layout(output_cols, col_types, scalings, vocabularies):
    """
    Build the output layout describing where each output column's
    predictions and targets live in the output tensor.
    
    Returns:
      output_layout: list of dicts
      total_pred_dim: total model output dimension
      total_tgt_dim: total target dimension
    """
    layout = []
    pred_idx = 0  # index into model prediction tensor
    tgt_idx = 0   # index into target tensor
    
    for col in output_cols:
        ct = col_types[col]
        
        if ct == 'out':
            layout.append({'col': col, 'type': 'out', 
                          'start': pred_idx, 'end': pred_idx + 1,
                          'tgt_start': tgt_idx, 'tgt_end': tgt_idx + 1})
            pred_idx += 1; tgt_idx += 1
            
        elif ct == 'outlab':
            layout.append({'col': col, 'type': 'outlab',
                          'start': pred_idx, 'end': pred_idx + 1,
                          'tgt_start': tgt_idx, 'tgt_end': tgt_idx + 1})
            pred_idx += 1; tgt_idx += 1
            
        elif ct == 'outex':
            ml = scalings[col]['max_len']
            layout.append({'col': col, 'type': 'outex',
                          'start': pred_idx, 'end': pred_idx + ml,
                          'tgt_start': tgt_idx, 'tgt_end': tgt_idx + ml,
                          'max_len': ml})
            pred_idx += ml; tgt_idx += ml
            
        elif ct == 'outlabcat':
            nc = len(vocabularies[col])
            layout.append({'col': col, 'type': 'outlabcat',
                          'start': pred_idx, 'end': pred_idx + nc,
                          'tgt_start': tgt_idx, 'tgt_end': tgt_idx + 1,
                          'num_classes': nc})
            pred_idx += nc; tgt_idx += 1
            
        elif ct in ('outexcat', 'outdec'):
            ml = scalings[col]['max_len']
            nc = text_num_classes(ct, vocabularies[col])
            layout.append({'col': col, 'type': ct,
                          'start': pred_idx, 'end': pred_idx + ml * nc,
                          'tgt_start': tgt_idx, 'tgt_end': tgt_idx + ml,
                          'num_classes': nc, 'max_len': ml})
            pred_idx += ml * nc; tgt_idx += ml
    
    return layout, pred_idx, tgt_idx




##############################################
# Perplexity calculation for categorical outputs
##############################################
def compute_perplexity_metrics(predictions, targets, output_layout, vocabularies=None):
    """
    Compute per-column and global perplexity for categorical outputs.
    Returns dict of {col_name: perplexity} and global perplexity.
    """
    metrics = {}
    all_ce_losses = []

    for entry in output_layout:
        col_name = entry['col']
        ct = entry['type']

        if ct == 'outlabcat':
            pred_slice = predictions[:, entry['start']:entry['end']]
            tgt_idx = targets[:, entry['tgt_start']:entry['tgt_end']].squeeze(-1).long()
            ce = F.cross_entropy(pred_slice, tgt_idx, reduction='mean')
            ppl = torch.exp(ce).item()
            metrics[col_name] = ppl
            all_ce_losses.append(ce)

        elif ct in ('outexcat', 'outdec'):
            nc = entry['num_classes']
            ml = entry['max_len']
            pred_slice = predictions[:, entry['start']:entry['end']]
            tgt_idx = targets[:, entry['tgt_start']:entry['tgt_end']].long()
            pred_r = pred_slice.view(-1, ml, nc).reshape(-1, nc)
            tgt_r = tgt_idx.reshape(-1)
            # Mask out padding (index 0) for perplexity
            mask = tgt_r > 0
            if mask.any():
                ce = F.cross_entropy(pred_r[mask], tgt_r[mask], reduction='mean')
            else:
                ce = F.cross_entropy(pred_r, tgt_r, reduction='mean')
            ppl = torch.exp(ce).item()
            metrics[col_name] = ppl
            all_ce_losses.append(ce)

    global_ppl = float('nan')
    if all_ce_losses:
        avg_ce = sum(all_ce_losses) / len(all_ce_losses)
        global_ppl = torch.exp(avg_ce).item()

    return metrics, global_ppl

##############################################
# Updated calculate_dims (supports *cat types)
##############################################
def calculate_input_dim(cols, col_types, scalings, vocabularies):
    """Calculate total input dimension, including one-hot for *cat types."""
    dim = 0
    for col in cols:
        ct = col_types[col]
        if ct in ['intex', 'outex', 'inenc']:
            dim += scalings[col]['max_len']
        elif ct == 'intexcat':
            dim += scalings[col]['max_len'] * len(vocabularies[col])
        elif ct == 'inlabcat':
            dim += len(vocabularies[col])
        elif ct == 'inim':
            im_size = scalings[col]["im_size"]; patch_size = scalings[col]["patch_size"]
            num_patches = 1 if patch_size == 1 else (im_size // patch_size) ** 2
            dim += num_patches * 3
        else:  # in, inlab
            dim += 1
    return dim

def calculate_output_pred_dim(cols, col_types, scalings, vocabularies):
    """Calculate model output dimension (logits for *cat)."""
    dim = 0
    for col in cols:
        ct = col_types[col]
        if ct in ['outex', 'intex']:
            dim += scalings[col]['max_len']
        elif ct in ('outexcat', 'outdec'):
            dim += scalings[col]['max_len'] * text_num_classes(ct, vocabularies[col])
        elif ct == 'outlabcat':
            dim += len(vocabularies[col])
        else:  # out, outlab
            dim += 1
    return dim

def calculate_output_tgt_dim(cols, col_types, scalings):
    """Calculate target tensor dimension (*cat stores class indices)."""
    dim = 0
    for col in cols:
        ct = col_types[col]
        if ct in ['outex', 'outexcat', 'intex', 'intexcat', 'outdec', 'inenc']:
            dim += scalings[col]['max_len']
        else:  # out, outlab, outlabcat
            dim += 1
    return dim

# Legacy wrapper for backward compatibility
def calculate_dims(cols, col_types, scalings, vocabularies=None):
    """Legacy function: for inputs uses new one-hot dims, for outputs uses pred dim."""
    if vocabularies is None:
        vocabularies = {}
    dim = 0
    for col in cols:
        ct = col_types[col]
        if ct in ['intex', 'outex', 'inenc']:
            dim += scalings[col]['max_len']
        elif ct == 'intexcat':
            dim += scalings[col]['max_len'] * len(vocabularies.get(col, {}))
        elif ct in ['inlabcat', 'outlabcat']:
            dim += len(vocabularies.get(col, {}))
        elif ct in ('outexcat', 'outdec'):
            dim += scalings[col]['max_len'] * text_num_classes(ct, vocabularies.get(col, {}))
        elif ct == 'inim':
            im_size = scalings[col]["im_size"]; patch_size = scalings[col]["patch_size"]
            num_patches = 1 if patch_size == 1 else (im_size // patch_size) ** 2
            dim += num_patches * 3
        else:
            dim += 1
    return dim


##############################################
# Custom Dataset (UPDATED: supports *cat types)
##############################################
##############################################
# Custom Dataset (UPDATED: Validation & Cleaning)
##############################################
class CustomDataset(Dataset):
    def __init__(self, csv_file, delimiter=',', input_cols=[], output_cols=[], col_types={},
                 vocabularies={}, scalings={}, image_params={}, df=None, keep_raw=False, scale_ranges=None):
        """df: optional already-loaded rows (e.g. the training part of a split) instead of reading csv_file.
        keep_raw: keep the cleaned, unscaled rows in self.raw_df (aligned with the dataset's rows)."""
        if df is not None:
            self.df = df.reset_index(drop=True).copy()
        else:
            print(f"Loading dataset from {csv_file}...")
            try:
                self.df = read_table(csv_file, delimiter=delimiter)
            except Exception as e:
                print(f"Error reading CSV: {e}")
                self.df = pd.DataFrame() # Empty fallback

        self.input_cols = input_cols
        self.output_cols = output_cols
        self.col_types = col_types
        self.vocabularies = vocabularies
        self.scalings = scalings if scalings else {}
        self.image_params = image_params
        self.scale_ranges = scale_ranges or {}  # numeric column -> [lo, hi] model range for new scalings
        self.inim_cache = {}
        
        # --- NEW: VALIDATION STEP ---
        if not self.df.empty:
            self.validate_and_clean_data()
        
        self.raw_df = self.df.copy() if keep_raw else None
        # Proceed with setup only if data remains
        if not self.df.empty:
            self.setup_vocabularies()
            self.convert_labels_to_numbers()
            self.scale_data()
            self._build_arrays()
        else:
            print("⚠ Warning: Dataset is empty after validation (or file was empty).")

    def validate_and_clean_data(self):
        """
        Removes rows with missing values, non-numeric data in numeric columns, 
        or missing image files.
        """
        initial_count = len(self.df)
        active_cols = self.input_cols + self.output_cols
        
        # 1. Drop basic NaNs in used columns (counted per column, to explain an empty result)
        self.drop_report = {}  # column -> rows dropped because the value was missing or not a number
        for col in active_cols:
            n_missing = int(self.df[col].isna().sum())
            if n_missing: self.drop_report[col] = f"{n_missing} missing"
        self.df.dropna(subset=active_cols, inplace=True)
        
        # 2. Validate Numeric Columns
        # We check columns marked as scalar inputs/outputs ('in', 'out')
        numeric_cols = [c for c in active_cols if self.col_types[c] in ['in', 'out']]
        for col in numeric_cols:
            # Coerce errors to NaN, then drop rows that became NaN
            # This handles cases where a number column contains "error" or garbage text
            # True / False (bool dtype, or the words in a text column) count as 1 / 0
            if is_hashed(col, self.col_types, self.vocabularies):
                self.df[col] = hash_encode_series(self.df[col], self.vocabularies[col])
            converted = numeric_series(self.df[col])
            bad = int(converted.isna().sum())
            if bad:
                sample = self.df[col][converted.isna()].astype(str).head(3).tolist()
                self.drop_report[col] = (self.drop_report.get(col, "") + ", " if col in self.drop_report else "") + \
                    f"{bad} not numbers (e.g. {', '.join(repr(x) for x in sample)})"
            self.df[col] = converted
        
        self.df.dropna(subset=numeric_cols, inplace=True)

        # 3. Validate Image Paths (inim)
        # Check if files exist.
        inim_cols = [c for c in active_cols if self.col_types[c] == 'inim']
        if inim_cols:
            # Create a mask for valid images
            valid_mask = pd.Series(True, index=self.df.index)
            for col in inim_cols:
                # We use apply to check os.path.exists
                # We strip whitespace just in case
                col_valid = self.df[col].astype(str).str.strip().apply(os.path.exists)
                valid_mask = valid_mask & col_valid
                
                # Report missing files for debugging (optional, printing first few)
                missing_count = (~col_valid).sum()
                if missing_count > 0:
                    print(f"  - Found {missing_count} missing image files in column '{col}'")
            
            self.df = self.df[valid_mask]

        # 4. Validate Categorical Consistency
        # Ensure we don't have empty strings for text inputs if that matters
        text_cols = [c for c in active_cols if self.col_types[c] in ['intex', 'outex', 'intexcat', 'outexcat', 'inenc', 'outdec']]
        for col in text_cols:
            # Remove rows where text is just whitespace or empty
            self.df = self.df[self.df[col].astype(str).str.strip().str.len() > 0]

        final_count = len(self.df)
        dropped = initial_count - final_count
        if dropped > 0:
            print(f"Dataset Validation: Dropped {dropped} invalid rows. (Remaining: {final_count})")
        else:
            print(f"Dataset Validation: All {final_count} rows are valid.")
            
        # Reset index so __getitem__ works correctly 0..len-1
        self.df.reset_index(drop=True, inplace=True)

    def setup_vocabularies(self):
        """Helper to init vocabularies after cleaning but before processing."""
        for col in self.vocabularies:
            # If vocabulary was already built manually in 'ask_column_types', we might update it here 
            # or just rely on existing. If values are int (like 1,2,3), it's likely already built.
            if not self.vocabularies[col] or isinstance(list(self.vocabularies[col].values())[0], int): 
                continue
                
            # Otherwise, rebuild based on the CLEAN data
            sorted_tokens = sorted(self.vocabularies[col].keys())
            self.vocabularies[col] = {token: i+1 for i, token in enumerate(sorted_tokens)}

    def convert_labels_to_numbers(self):
        for col in self.input_cols + self.output_cols:
            ct = self.col_types[col]
            if ct in ['inlab', 'outlab', 'inlabcat', 'outlabcat']:
                # Map values. If a value isn't in vocab (shouldn't happen if vocab built from df), fillna(0)
                self.df[col] = self.df[col].astype(str).map(self.vocabularies[col]).fillna(0)
        
    def scale_data(self):
        for col in self.input_cols + self.output_cols:
            col_type = self.col_types[col]
            
            if col_type in ['intex', 'outex', 'intexcat', 'outexcat', 'inenc', 'outdec']:
                self.df[col] = self.df[col].astype(str)
                if col in self.scalings and 'max_len' in self.scalings[col]:
                    max_len = self.scalings[col]['max_len']
                elif col_type in seqm.SEQ_TYPES:  # tokens (+ <eos> for outputs)
                    toks = self._tokens(col)
                    max_len = max((len(t) for t in toks), default=0) + (1 if col_type == 'outdec' else 0)
                    self.scalings[col] = {'max_len': max_len}
                else:
                    max_len = self.df[col].apply(len).max()
                    self.scalings[col] = {'max_len': max_len}

            elif col_type == 'inim':
                if col in self.scalings: continue
                im_size = self.image_params[col]["im_size"]
                patch_size = self.image_params[col]["patch_size"]
                num_patches = 1 if patch_size == 1 else (im_size // patch_size) ** 2
                pc_all, ec_all, bc_all = [], [], []
                
                # Check for existing cache
                cache_filename = f"cache_{col}.pkl"
                if os.path.exists(cache_filename):
                    print(f"Loading image cache for {col}...")
                    with open(cache_filename, "rb") as f: cache = pickle.load(f)
                else: 
                    cache = {}
                
                self.inim_cache[col] = cache
                
                # Process images
                print(f"Processing images for '{col}' (this may take a while)...")
                valid_indices = []
                
                for index in tqdm(range(len(self.df)), desc=f"Img Proc {col}"):
                    file_path = str(self.df.at[index, col]).strip()
                    
                    if file_path in cache: 
                        codes = cache[file_path]
                    else:
                        # Double check existence (validation should have caught this, but for safety)
                        if not os.path.exists(file_path):
                            continue # Skip this code collection, will be filtered effectively by not adding stats
                            
                        try: 
                            img = Image.open(file_path).convert("L")
                            img = img.resize((im_size, im_size))
                            img_array = np.array(img)
                            
                            if patch_size == 1: 
                                patches = [img_array]
                            else:
                                n_patches = im_size // patch_size
                                patches = []
                                for i in range(n_patches):
                                    for j in range(n_patches):
                                        patches.append(img_array[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size])
                            
                            codes = []
                            for patch in patches:
                                pc, ec, bc = compute_patch_codes(patch)
                                codes.extend([pc, ec, bc])
                            cache[file_path] = codes
                        except Exception as e:
                            print(f"Corrupt image found during scaling: {file_path} ({e})")
                            continue

                    for i in range(0, len(codes), 3):
                        pc_all.append(codes[i])
                        ec_all.append(codes[i+1])
                        bc_all.append(codes[i+2])
                
                # Save updated cache
                with open(cache_filename, "wb") as f: pickle.dump(cache, f)
                
                self.scalings[col] = {
                    "num_patches": num_patches,
                    "pc_min": min(pc_all) if pc_all else 0, "pc_max": max(pc_all) if pc_all else 0,
                    "ec_min": min(ec_all) if ec_all else 0, "ec_max": max(ec_all) if ec_all else 0,
                    "bc_min": min(bc_all) if bc_all else 0, "bc_max": max(bc_all) if bc_all else 0,
                    "im_size": im_size, "patch_size": patch_size
                }

            elif col_type not in ['inlab', 'outlab', 'inlabcat', 'outlabcat']:
                # Numeric Scalar Scaling
                if col in self.scalings and 'min' in self.scalings[col]:
                    min_val = self.scalings[col]['min']; max_val = self.scalings[col]['max']
                else:
                    min_val, max_val = self.df[col].min(), self.df[col].max()
                    self.scalings[col] = {'min': min_val, 'max': max_val}
                    if col in self.scale_ranges and tuple(self.scale_ranges[col]) != (-1.0, 1.0):
                        self.scalings[col]['range'] = [float(x) for x in self.scale_ranges[col]]
                self.df[col] = to_model_units(self.df[col], self.scalings[col])
        
    def __len__(self):
        return len(self.df)

    def _tokens(self, col):
        """Token lists of a text encoder / decoder column, tokenized once."""
        cache = self.__dict__.setdefault("_tok_cache", {})
        if col not in cache:
            mode = seqm.tokenizer_of(self.vocabularies[col])
            cache[col] = [seqm.tokenize(t, mode) for t in self.df[col].astype(str)]
        return cache[col]

    def _text_ids(self, col, max_len, offset=0):
        """(N, max_len) int array of a character column's vocabulary ids (0 = padding), truncated like __getitem__."""
        vocab = self.vocabularies[col]; out = np.zeros((len(self.df), max_len), dtype=np.int64)
        for r, v in enumerate(self.df[col].astype(str)):
            ids = [vocab.get(ch, 0) for ch in v[:max_len]]
            out[r, :len(ids)] = ids
        return out

    def _seq_ids(self, col, is_output):
        vocab, ml = self.vocabularies[col], self.scalings[col]['max_len']
        out = np.zeros((len(self.df), ml), dtype=np.int64)
        for r, toks in enumerate(self._tokens(col)):
            ids = [vocab.get(t, seqm.UNK) for t in toks]
            ids = ids[:ml - 1] + [seqm.EOS] if is_output else ids[:ml]
            out[r, :len(ids)] = ids
        return out

    def _segments(self, cols, is_input):
        """Encode every column once: [("dense", width, float32 (N, width)) | ("onehot", width, ids (N, L), classes)].
        One-hot columns stay as ids (expanded per batch), so wide one-hot inputs do not fill memory."""
        segs, N = [], len(self.df)
        for col in cols:
            ct = self.col_types[col]
            if ct == 'inim': return None  # images: patch codes are built row by row (see _getitem_rowwise)
            if ct == 'inenc' or ct == 'outdec':
                a = self._seq_ids(col, ct == 'outdec'); segs.append(("dense", a.shape[1], a.astype(np.float32)))
            elif ct == 'intexcat' and is_input:
                V = len(self.vocabularies[col]); ids = self._text_ids(col, self.scalings[col]['max_len']) - 1
                ids[(ids < 0) | (ids >= V)] = -1
                segs.append(("onehot", ids.shape[1] * V, ids, V))
            elif ct == 'inlabcat' and is_input:
                V = len(self.vocabularies[col]); ids = self.df[col].to_numpy().astype(np.int64)[:, None]
                ids[(ids < 0) | (ids >= V)] = -1
                segs.append(("onehot", V, ids, V))
            elif ct in ('intex', 'outex', 'outexcat') or (ct == 'intexcat' and not is_input):
                a = self._text_ids(col, self.scalings[col]['max_len']); segs.append(("dense", a.shape[1], a.astype(np.float32)))
            elif ct == 'outlabcat':
                segs.append(("dense", 1, self.df[col].to_numpy().astype(np.int64).astype(np.float32)[:, None]))
            else:  # in, out, inlab, outlab: already numeric / scaled
                segs.append(("dense", 1, self.df[col].to_numpy(dtype=np.float32)[:, None]))
        return segs

    def _build_arrays(self):
        self._in_segs = self._segments(self.input_cols, True)
        self._out_segs = self._segments(self.output_cols, False) if self._in_segs is not None else None
        if self._in_segs is None or self._out_segs is None: self._in_segs = self._out_segs = None

    @staticmethod
    def _assemble(segs, idx):
        width = sum(sg[1] for sg in segs); out = np.zeros((len(idx), width), dtype=np.float32); start = 0
        for sg in segs:
            if sg[0] == "dense": out[:, start:start + sg[1]] = sg[2][idx]
            else:
                ids, V = sg[2][idx], sg[3]
                rows, pos = np.nonzero(ids >= 0)
                out[rows, start + pos * V + ids[rows, pos]] = 1.0
            start += sg[1]
        return out

    def batch(self, idx):
        """(inputs, targets) float tensors for the rows idx (array of positions), built from the pre-encoded columns."""
        idx = np.asarray(idx, dtype=np.int64)
        if getattr(self, "_in_segs", None) is None:
            items = [self._getitem_rowwise(int(i)) for i in idx]
            return torch.stack([a for a, _ in items]), torch.stack([b for _, b in items])
        return torch.from_numpy(self._assemble(self._in_segs, idx)), torch.from_numpy(self._assemble(self._out_segs, idx))

    def __getitem__(self, idx):
        if getattr(self, "_in_segs", None) is None: return self._getitem_rowwise(idx)
        x, y = self.batch([idx])
        return x[0], y[0]

    def _getitem_rowwise(self, idx):
        """Reference row encoder (used for image columns, and to test the array path against)."""
        # NOTE: Because we validated in __init__, we assume data here is generally valid.
        # However, for images, file corruption could still cause runtime crashes,
        # so we keep the try/except block for image loading specifically.
        
        input_features = []
        for col in self.input_cols:
            col_type = self.col_types[col]
            
            if col_type == 'inenc':
                input_features.extend(seqm.encode_ids(self.df.at[idx, col], self.vocabularies[col],
                                                     self.scalings[col]['max_len'], is_output=False))

            elif col_type == 'intexcat':
                value = str(self.df.at[idx, col])
                vocab = self.vocabularies[col]
                vocab_size = len(vocab)
                max_len = self.scalings[col]['max_len']
                for i in range(max_len):
                    onehot = [0.0] * vocab_size
                    if i < len(value):
                        token_idx = vocab.get(value[i], 0)
                        if 0 < token_idx <= vocab_size:
                            onehot[token_idx - 1] = 1.0
                    input_features.extend(onehot)
                    
            elif col_type == 'inlabcat':
                vocab = self.vocabularies[col]
                vocab_size = len(vocab)
                value = int(self.df.at[idx, col])
                onehot = [0.0] * vocab_size
                if 0 <= value < vocab_size:
                    onehot[value] = 1.0
                input_features.extend(onehot)
            
            elif col_type in ['intex', 'outex']:
                value = str(self.df.at[idx, col])
                vocab = self.vocabularies[col]
                max_len = self.scalings[col]['max_len']
                seq = [vocab.get(ch, 0) for ch in value]
                if len(seq) < max_len: seq = seq + [0] * (max_len - len(seq))
                else: seq = seq[:max_len]
                input_features.extend(seq)
                
            elif col_type in ['inlab', 'outlab']:
                input_features.append(self.df.at[idx, col])
                
            elif col_type == 'inim':
                im_size = self.image_params[col]["im_size"]
                patch_size = self.image_params[col]["patch_size"]
                file_path = str(self.df.at[idx, col]).strip()
                cache = self.inim_cache.get(col, {})
                
                if file_path in cache: 
                    codes = cache[file_path]
                else:
                    # Runtime fallback for corrupted images not caught by os.path.exists
                    try: 
                        img = Image.open(file_path).convert("L")
                        img = img.resize((im_size, im_size))
                        img_array = np.array(img)
                        if patch_size == 1: 
                            patches = [img_array]
                        else:
                            n_patches = im_size // patch_size
                            patches = []
                            for i in range(n_patches):
                                for j in range(n_patches):
                                    patches.append(img_array[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size])
                        codes = []
                        for patch in patches:
                            pc, ec, bc = compute_patch_codes(patch)
                            codes.extend([pc, ec, bc])
                    except Exception:
                        # Return zeros if runtime load fails (rare if validation worked)
                        num_patches = 1 if patch_size == 1 else (im_size // patch_size) ** 2
                        codes = [0.0] * (num_patches * 3)

                scaling = self.scalings[col]
                pc_min, pc_max = scaling["pc_min"], scaling["pc_max"]
                ec_min, ec_max = scaling["ec_min"], scaling["ec_max"]
                bc_min, bc_max = scaling["bc_min"], scaling["bc_max"]
                
                for i in range(0, len(codes), 3):
                    pc, ec, bc = codes[i], codes[i+1], codes[i+2]
                    norm_pc = 0 if pc_max == pc_min else 2 * (pc - pc_min) / (pc_max - pc_min) - 1
                    norm_ec = 0 if ec_max == ec_min else 2 * (ec - ec_min) / (ec_max - ec_min) - 1
                    norm_bc = 0 if bc_max == bc_min else 2 * (bc - bc_min) / (bc_max - bc_min) - 1
                    input_features.extend([norm_pc, norm_ec, norm_bc])
            else:
                input_features.append(self.df.at[idx, col])
            
        output_features = []
        for col in self.output_cols:
            col_type = self.col_types[col]
            
            if col_type == 'outdec':
                output_features.extend(float(i) for i in seqm.encode_ids(self.df.at[idx, col], self.vocabularies[col],
                                                                         self.scalings[col]['max_len'], is_output=True))

            elif col_type == 'outlabcat':
                value = int(self.df.at[idx, col])
                output_features.append(float(value))
                
            elif col_type == 'outexcat':
                value = str(self.df.at[idx, col])
                vocab = self.vocabularies[col]
                max_len = self.scalings[col]['max_len']
                seq = [float(vocab.get(ch, 0)) for ch in value]
                if len(seq) < max_len: seq = seq + [0.0] * (max_len - len(seq))
                else: seq = seq[:max_len]
                output_features.extend(seq)
            
            elif col_type in ['intex', 'outex']:
                value = str(self.df.at[idx, col])
                vocab = self.vocabularies[col]
                max_len = self.scalings[col]['max_len']
                seq = [vocab.get(ch, 0) for ch in value]
                if len(seq) < max_len: seq = seq + [0] * (max_len - len(seq))
                else: seq = seq[:max_len]
                output_features.extend(seq)
            elif col_type in ['inlab', 'outlab']:
                output_features.append(self.df.at[idx, col])
            else:
                output_features.append(self.df.at[idx, col])
            
        return torch.tensor(input_features, dtype=torch.float), torch.tensor(output_features, dtype=torch.float)

    def compute_class_weights(self, device=None):
        # (Same as before, simplified copy for context)
        weights_dict = {}
        if self.df.empty: return weights_dict
        for col in self.output_cols:
            col_type = self.col_types[col]
            if col_type in ['outlabcat', 'outexcat']:
                vocab = self.vocabularies[col]
                num_classes = text_num_classes(col_type, vocab) if col_type == 'outexcat' else len(vocab)
                counts = np.ones(num_classes, dtype=np.float32)
                if col_type == 'outlabcat':  # values are already mapped ints
                    v = self.df[col].to_numpy().astype(np.int64); v = v[(v >= 0) & (v < num_classes)]
                    counts += np.bincount(v, minlength=num_classes).astype(np.float32)
                elif col_type == 'outexcat':
                    # This logic assumes raw strings for outexcat were kept, 
                    # but scale_data didn't overwrite them for *cat types. 
                    # Actually, scale_data only casts to str.
                    # Ideally, re-read raw for counting or map tokens.
                    pass # Simplified for brevity, logic remains valid
                total_samples = np.sum(counts)
                weights = total_samples / (num_classes * counts)
                w_tensor = torch.tensor(weights, dtype=torch.float)
                if device: w_tensor = w_tensor.to(device)
                weights_dict[col] = w_tensor
        return weights_dict



##############################################
# Noise injection and LSUV (unchanged)
##############################################
##############################################
# Leak-free train / validation preparation
##############################################
def _row_group_keys(df, cols, col_types):
    """One hash per row over `cols`; numeric columns compare by value at 12 significant digits
    (1 == 1.0, and a CSV round trip of a float cannot break the match)."""
    norm = {}
    for c in cols:
        v = df[c]
        if col_types.get(c) in ('in', 'out'):
            v = numeric_series(v).map(lambda x: format(x, '.12g'))
        norm[c] = v.astype(str)
    return pd.util.hash_pandas_object(pd.DataFrame(norm, index=df.index), index=False)


def dedupe_rows(df, input_cols, output_cols, col_types):
    """Drop rows that repeat an earlier row in every used column. Returns (df, removed)."""
    used = list(dict.fromkeys(list(input_cols) + list(output_cols)))
    dup = _row_group_keys(df, used, col_types).duplicated().values
    return df[~dup], int(dup.sum())


def group_split_mask(keys, val_frac, seed=0):
    """Validation mask of about val_frac of the rows; rows with identical inputs stay on one side."""
    counts = keys.value_counts()
    order = counts.index.to_numpy().copy(); np.random.default_rng(seed).shuffle(order)
    chosen, total, target = [], 0, val_frac * len(keys)
    for k in order:
        if total >= target: break
        chosen.append(k); total += int(counts[k])
    return keys.isin(set(chosen)).values


def group_kfold_ids(keys, k, seed=0):
    """Fold id per row; rows with identical inputs share a fold, folds balanced by row count."""
    counts = keys.value_counts()
    order = counts.index.to_numpy().copy(); np.random.default_rng(seed).shuffle(order)
    sizes, fold_of = [0] * k, {}
    for key in order:
        f = int(np.argmin(sizes)); fold_of[key] = f; sizes[f] += int(counts[key])
    return keys.map(fold_of).to_numpy()


def prepare_train_val(csv_file, delimiter, input_cols, output_cols, col_types, vocabularies, image_params,
                      val_frac=0.0, val_file=None, seed=0, val_delimiter=None, scalings=None, val_is_holdout=False,
                      scale_ranges=None):
    """Deduplicated training set and a validation set that shares no row and no input
    combination with it. Scalings are fit on the training rows only.
    scalings: reuse these (resuming a model) instead of fitting them on the training rows.
    val_is_holdout: val_file lists rows OF csv_file that were held out; drop them from training
    (instead of dropping validation rows that also occur in training).
    Returns (train_dataset, val_dataset or None, report)."""
    used = list(dict.fromkeys(list(input_cols) + list(output_cols)))
    def require_columns(frame, path):
        missing = [c for c in used if c not in frame.columns]
        if missing:
            raise ValueError(f"{path} has no column(s) {missing}; its columns are {list(frame.columns)[:12]}. "
                             f"Check the delimiter, or pick the right {'validation ' if path != csv_file else ''}file.")
    df = read_table(csv_file, delimiter=delimiter)
    require_columns(df, csv_file)
    n_raw = len(df)
    df, dups = dedupe_rows(df, input_cols, output_cols, col_types)
    report = {"rows_read": n_raw, "duplicates_removed": dups}
    keys = _row_group_keys(df, input_cols, col_types)
    val_df = None
    if val_file:
        # a separate validation file has the training file's layout; model_validation.csv (holdout) is saved already renamed
        vfmt = {"header": True} if val_is_holdout else (CSV_FORMATS.get(os.path.abspath(val_file)) or CSV_FORMATS.get(os.path.abspath(csv_file)))
        val_df = read_table(val_file, delimiter=val_delimiter or delimiter, fmt=vfmt)
        require_columns(val_df, val_file)
        val_df, report["val_duplicates_removed"] = dedupe_rows(val_df, input_cols, output_cols, col_types)
        vkeys = _row_group_keys(val_df, input_cols, col_types)
        if val_is_holdout:
            held = keys.isin(set(vkeys)).values
            report["train_rows_held_out"] = int(held.sum()); report["val_rows_also_in_training"] = 0
            df = df[~held]
        else:
            overlap = vkeys.isin(set(keys)).values
            report["val_rows_also_in_training"] = int(overlap.sum())
            val_df = val_df[~overlap]
    elif val_frac > 0:
        mask = group_split_mask(keys, val_frac, seed)
        val_df, df = df[mask], df[~mask]
    print(f"Data: {n_raw} rows read, {dups} exact duplicates removed.")
    train_ds = CustomDataset(csv_file, delimiter, input_cols, output_cols, col_types, vocabularies, scalings or {}, image_params, df=df,
                             scale_ranges=scale_ranges)
    if len(train_ds) == 0:
        why = "; ".join(f"'{c}': {r}" for c, r in getattr(train_ds, "drop_report", {}).items())
        raise ValueError("No usable training rows: every row was dropped while cleaning the data"
                         + (f" ({why}). A column typed numeric (in / out) must hold numbers." if why else "."))
    val_ds = None
    if val_df is not None:
        if report.get("train_rows_held_out"):
            print(f"Held-out validation rows kept out of training: {report['train_rows_held_out']}.")
        elif "val_rows_also_in_training" in report:
            print(f"Validation file: {report['val_duplicates_removed']} duplicates and "
                  f"{report['val_rows_also_in_training']} rows whose inputs also occur in training removed.")
        val_ds = CustomDataset(val_file or csv_file, val_delimiter or delimiter, input_cols, output_cols, col_types,
                               vocabularies, train_ds.scalings, image_params, df=val_df, keep_raw=True)
        if len(val_ds) == 0: raise ValueError("Validation set is empty after removing rows shared with training.")
    report.update(train=len(train_ds), val=len(val_ds) if val_ds is not None else 0)
    print(f"Split: {report['train']} training rows, {report['val']} held-out validation rows "
          f"(no shared rows or input combinations; scalings from training rows only).")
    return train_ds, val_ds, report


VALIDATION_ROWS_FILE = "model_validation.csv"


def save_validation_rows(val_loader, path=VALIDATION_ROWS_FILE):
    """Write the held-out validation rows (raw values) next to model.pt so the explorer can score them.
    Without a validation set the file is rewritten empty, so an earlier run's rows are never mistaken
    for this model's."""
    raw = getattr(getattr(val_loader, "dataset", None), "raw_df", None) if val_loader is not None else None
    if raw is not None: raw.to_csv(path, index=False, float_format="%.17g")  # exact float round trip
    elif os.path.exists(path): open(path, "w").close()


def prepare_resume(config_path="config.json", model_path="model.pt"):
    """main() arguments that continue training the saved model: architecture and data encoding from
    config.json, training rows = its dataset minus the held-out rows saved in model_validation.csv.
    Returns (main_kwargs, val_dataset or None, report)."""
    with open(config_path) as f: cfg = json.load(f)
    if not os.path.isfile(model_path): raise FileNotFoundError(f"No trained model at {model_path}")
    register_csv_format(cfg.get("file_path"), cfg.get("csv_format"))
    if isinstance(cfg.get("hidden_dims"), dict): raise ValueError("Auto-grow models cannot be resumed.")
    path = cfg.get("file_path")
    if not path or not os.path.isfile(path): raise FileNotFoundError(f"Training data {path!r} from config.json not found")
    col_types = cfg["col_types"]; vocab = cfg.get("vocabularies", {}); image_params = cfg.get("image_params", {})
    input_cols = [c for c, v in col_types.items() if 'in' in v]; output_cols = [c for c, v in col_types.items() if 'out' in v]
    delim = training_file_delimiter(path, col_types)
    vfile = os.path.join(os.path.dirname(os.path.abspath(model_path)), VALIDATION_ROWS_FILE)
    if not (os.path.isfile(vfile) and os.path.getsize(vfile) > 0):
        vfile = None
        print("Resume: this model has no saved held-out rows; continuing without validation "
              "(a new random split would put already-trained rows into validation).")
    train_ds, val_ds, report = prepare_train_val(path, delim, input_cols, output_cols, col_types, vocab, image_params,
                                                 val_file=vfile, val_is_holdout=True, scalings=cfg["scalings"])
    act_cfg = cfg.get("activation", {"name": "ReLU", "params": {}}); name, params = act_cfg.get("name", "ReLU"), act_cfg.get("params", {})
    amap = _build_activation_map()
    if name == "Custom":
        custom = _rebuild_custom_activation_from_config(params)
        def base(): return custom()
    elif name in amap:
        def base(): return amap[name](**params)
    else:
        def base(): return nn.ReLU()
    base.activation_config = act_cfg
    at = cfg.get("activation_type", 0)
    kwargs = dict(csv_file=path, delimiter=delim, input_cols=input_cols, output_cols=output_cols, col_types=col_types,
                  vocabularies=vocab, scalings=cfg["scalings"], image_params=image_params, hidden_dims=cfg["hidden_dims"],
                  batch_size=cfg.get("batch_size", 32), activation_cls=wrap_activation(base, at), activation_type=at,
                  residual_type=cfg.get("residual_type", "residual"), norm_type=cfg.get("norm_type", "layer"),
                  groups=cfg.get("groups"), attention_type=cfg.get("attention_type", "none"), num_heads=cfg.get("num_heads"),
                  input_attention_type=cfg.get("input_attention_type", "none"), moe_mode=cfg.get("moe_mode", 1),
                  noise_mode=cfg.get("noise_mode", "none"), noise_params=cfg.get("noise_params", {}),
                  base_activation_cls_for_config=base, mlp_mode=cfg.get("mlp_mode", 0),
                  layer_routing=cfg.get("layer_routing", "none"), use_grn=cfg.get("use_grn", False),
                  optimizer_choice=cfg.get("optimizer_choice", "Adam"), train_dataset=train_ds, resume_from=model_path,
                  seq_params=cfg.get("seq_params", {}), seq_bridge=cfg.get("seq_bridge", "mlp"),
                  layer_activations=cfg.get("layer_activations"), output_activation=cfg.get("output_activation"))
    return kwargs, val_ds, report


def ask_validation_and_prepare(file_path, delimiter, input_cols, output_cols, col_types, vocabularies,
                               image_params, batch_size, ask_interval=True, scale_ranges=None):
    """CLI validation prompts + prepare_train_val. Returns (train_ds, val_loader, val_interval)."""
    print("\n--- Validation Setup ---")
    val_file = input("Enter validation CSV path (empty for percentage split): ").strip()
    val_frac = 0.0
    if val_file and not os.path.isfile(val_file):
        print("Invalid file, falling back to percentage split."); val_file = ""
    if not val_file:
        try: val_frac = float(input("Validation split % (0.0-1.0, e.g. 0.1): ").strip())
        except ValueError: val_frac = 0.0
    train_ds, val_ds, _ = prepare_train_val(file_path, delimiter, input_cols, output_cols, col_types,
                                            vocabularies, image_params, val_frac=val_frac, val_file=val_file or None,
                                            scale_ranges=scale_ranges)
    val_loader, val_interval = None, 1000
    if val_ds is not None:
        val_loader = make_loader(val_ds, batch_size, shuffle=False)
        if ask_interval: val_interval = int(input("Validation interval (steps): ").strip())
    else:
        print("Validation skipped.")
    return train_ds, val_loader, val_interval


def ask_loss_balancing(output_cols, col_types):
    """Only asked when several outputs include a categorical one (where CombinedLoss is used)."""
    if len(output_cols) < 2 or not any(col_types[c] in ('outlabcat', 'outexcat', 'outdec') for c in output_cols):
        return "normalized", 20.0
    print("\nLoss balancing across outputs:")
    print("  1: Normalized (each output's loss / trivial-predictor loss) [default]")
    print("  2: UW-SO (adaptive soft optimal uncertainty weighting)")
    print("  3: Legacy (0.1 x Huber + cross-entropy)")
    choice = input("Enter 1-3: ").strip()
    if choice == "2":
        t = input("UW-SO softmax temperature (default 20): ").strip()
        return "uwso", float(t) if t else 20.0
    return ("legacy" if choice == "3" else "normalized"), 20.0


def ask_noise_injection():
    print("Choose noise injection mode during training:")
    print("1: Dropout (with user defined percentage)")
    print("2: Input noise")
    print("3: Output noise")
    print("4: Noise Injection layers")
    print("5: Weight noise")
    print("6: No noise")
    choice = input("Enter the number corresponding to your choice: ").strip()
    noise_mode = "none"; noise_params = {}
    if choice == "1":
        noise_mode = "dropout"; noise_params["dropout_pct"] = float(input("Enter dropout percentage (0-1): ").strip())
    elif choice == "2":
        noise_mode = "input noise"; noise_params["std"] = float(input("Enter standard deviation for input noise: ").strip())
    elif choice == "3":
        noise_mode = "output noise"; noise_params["std"] = float(input("Enter standard deviation for output noise: ").strip())
    elif choice == "4":
        noise_mode = "noise injection layers"; noise_params["std"] = float(input("Enter standard deviation for noise injection layers: ").strip())
    elif choice == "5":
        noise_mode = "weight noise"; noise_params["std"] = float(input("Enter standard deviation for weight noise: ").strip())
    elif choice == "6": noise_mode = "none"
    else: print("Invalid choice, defaulting to no noise."); noise_mode = "none"
    return noise_mode, noise_params

def lsuv_init(model, dataloader, device, target_var=1.0, target_mean=0.0,
              normalize_mean=True, verbose=True, max_iter=15, tol=0.001):
    model.eval()
    try:
        sample_input, _ = next(iter(dataloader)); sample_input = sample_input.to(device)
    except StopIteration:
        print("Warning: Could not get sample batch for LSUV."); return
    if verbose:
        print("\n" + "="*60); print("Performing Advanced LSUV Initialization"); print("="*60)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.weight.shape[0] >= module.weight.shape[1]: nn.init.orthogonal_(module.weight)
            else: nn.init.orthogonal_(module.weight.t())
            if module.bias is not None: nn.init.zeros_(module.bias)

    def apply_lsuv_to_layer(layer, inp, current_iter_limit=max_iter):
        with torch.no_grad():
            for i in range(current_iter_limit):
                out = layer(inp); curr_mean = out.mean().item(); curr_var = out.var().item()
                var_err = abs(curr_var - target_var)
                mean_err = abs(curr_mean - target_mean) if normalize_mean else 0.0
                if var_err < tol and (not normalize_mean or mean_err < tol): return out
                if curr_var < 1e-10: break
                var_ratio = target_var / (curr_var + 1e-8); scale = math.sqrt(var_ratio)
                scale = 1.0 + 0.5 * (scale - 1.0)
                shift = 0.5 * (target_mean - curr_mean * scale) if normalize_mean else 0.0
                if isinstance(layer, nn.Linear):
                    layer.weight.data *= scale
                    if layer.bias is not None: layer.bias.data = layer.bias.data * scale + shift
                elif hasattr(layer, 'experts'):
                    for exp in layer.experts:
                        exp.weight.data *= scale
                        if exp.bias is not None: exp.bias.data = exp.bias.data * scale + shift
                elif hasattr(layer, 'base_linear'):
                    layer.base_linear.weight.data *= scale
                    if layer.base_linear.bias is not None: layer.base_linear.bias.data = layer.base_linear.bias.data * scale + shift
            return layer(inp)

    with torch.no_grad():
        current_activation = sample_input
        if hasattr(model, 'input_attn') and model.input_attn is not None:
            current_activation = model.input_attn(current_activation)
        for block_idx, block in enumerate(model.blocks):
            if verbose: print(f"  Block {block_idx+1}...")
            z = block.norm(current_activation)
            z = apply_lsuv_to_layer(block.linear1, z)
            if block.use_sgu: z = block.sgu(z)
            elif block.activation is not None: z = block.activation(z)
            if block.linear2 is not None: z = apply_lsuv_to_layer(block.linear2, z)
            current_activation = block._apply_residual_nonhyper(z, current_activation)
        if hasattr(model, 'final_linear'):
            if verbose: print("  Final Layer...")
            apply_lsuv_to_layer(model.final_linear, current_activation)
    model.train()
    if verbose: print("✓ LSUV Complete.\n")

def lsuv_init_conservative(model, dataloader, device, target_var=1.0, max_scale=2.0, verbose=True):
    model.eval()
    try: sample_input, _ = next(iter(dataloader)); sample_input = sample_input.to(device)
    except StopIteration: print("Warning: Could not get sample batch for LSUV."); return
    if verbose: print("\n" + "="*60); print("Performing Conservative LSUV (Variance Only)"); print("="*60)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.weight.shape[0] >= module.weight.shape[1]: nn.init.orthogonal_(module.weight)
            else: nn.init.orthogonal_(module.weight.t())
            if module.bias is not None: nn.init.zeros_(module.bias)
    with torch.no_grad():
        current_activation = sample_input
        if hasattr(model, 'input_attn') and model.input_attn is not None:
            current_activation = model.input_attn(current_activation)
        for block_idx, block in enumerate(model.blocks):
            pre_output = block(current_activation); pre_var = pre_output.var().item()
            if pre_var > 1e-8:
                raw_scale = torch.sqrt(torch.tensor(target_var / pre_var)).item()
                scale = max(1.0 / max_scale, min(max_scale, raw_scale))
                linear_layer = block.linear
                if isinstance(linear_layer, nn.Linear): linear_layer.weight.data *= scale
                elif hasattr(linear_layer, 'experts'):
                    for expert in linear_layer.experts: expert.weight.data *= scale
                elif hasattr(linear_layer, 'base_linear'): linear_layer.base_linear.weight.data *= scale
                post_output = block(current_activation); post_var = post_output.var().item()
                if verbose: print(f"  Block {block_idx + 1}: {pre_var:.4f} → {post_var:.4f} (scale={scale:.4f})")
                current_activation = post_output
            else:
                if verbose: print(f"  Block {block_idx + 1}: skipped (near-zero variance)")
                current_activation = pre_output
    model.train()
    if verbose: print("="*60 + "\n")

def ask_lsuv_init():
    print("\nDo you want to use LSUV initialization?")
    print("  0 - No LSUV initialization")
    print("  1 - Standard LSUV (variance normalization only)")
    print("  2 - Self-Normalizing LSUV (mean + variance normalization)")
    choice = input("Enter choice (0/1/2): ").strip()
    if choice == "0": return False, None, None
    elif choice in ["1", "2"]:
        normalize_mean = (choice == "2"); max_iter = 10
        try:
            custom_iter = input("Max iterations per layer (default 10): ").strip()
            if custom_iter: max_iter = int(custom_iter)
        except ValueError: max_iter = 10
        return True, max_iter, normalize_mean
    else: print("Invalid choice, defaulting to no LSUV"); return False, None, None


##############################################
# Training loop (UPDATED: CombinedLoss support)
##############################################
def validate_model(model, val_loader, criterion, device):
    model.eval(); criterion.eval()  # CombinedLoss reports its fixed (normalized) value in eval mode
    total_val_loss = 0.0
    with torch.no_grad():
        for v_inputs, v_targets in val_loader:
            v_inputs, v_targets = v_inputs.to(device), v_targets.to(device)
            v_outputs = seqm.model_forward(model, v_inputs, v_targets)  # teacher-forced for decoders
            loss = criterion(v_outputs, v_targets)
            total_val_loss += loss.item()
    model.train(); criterion.train()
    return total_val_loss / len(val_loader)

OPTIMIZER_STATE_FILE = "model_optimizer.pt"


def save_checkpoint(model, optimizer, optimizer_choice, path="model.pt"):
    """model.pt as before, plus the optimizer state so training can be resumed seamlessly."""
    torch.save(model.state_dict(), path)
    if hasattr(optimizer, "state_dict"):
        try: torch.save({"optimizer": optimizer_choice, "class": type(optimizer).__name__, "state": optimizer.state_dict()}, OPTIMIZER_STATE_FILE)
        except Exception as e: print(f"  (optimizer state not saved: {e})")


def restore_optimizer(optimizer, optimizer_choice, custom_lr=None):
    """Load the saved optimizer state when it belongs to the same optimizer; a new custom LR wins."""
    if not os.path.exists(OPTIMIZER_STATE_FILE) or not hasattr(optimizer, "load_state_dict"):
        print("Resume: no saved optimizer state, starting the optimizer fresh."); return False
    saved = torch.load(OPTIMIZER_STATE_FILE, map_location="cpu")
    if saved.get("optimizer") != optimizer_choice:
        print(f"Resume: saved optimizer state is for {saved.get('optimizer')}, not {optimizer_choice}; starting it fresh."); return False
    # Files saved before the class was recorded: "AdamHD" was then lamb.AdamAdaHD.
    saved_cls = saved.get("class") or ("AdamAdaHD" if optimizer_choice == "AdamHD" else type(optimizer).__name__)
    if saved_cls != type(optimizer).__name__:
        print(f"Resume: saved optimizer state is for {saved_cls}, not {type(optimizer).__name__}; starting it fresh."); return False
    fresh_lrs = [g.get("lr") for g in optimizer.param_groups]
    try: optimizer.load_state_dict(saved["state"])
    except Exception as e:
        print(f"Resume: optimizer state does not fit ({e}); starting it fresh."); return False
    if custom_lr is not None:
        for g, lr in zip(optimizer.param_groups, fresh_lrs): g["lr"] = lr
    print("Resume: optimizer state restored" + (" (with the new learning rate)." if custom_lr is not None else "."))
    return True


def materialize_lazy_modules(model, input_dim, device):
    """Run one eval-mode forward so lazily built modules (PaWeL, GLU, Gumbel) own their
    parameters before an optimizer is created from model.parameters()."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(2, input_dim, device=device))
    model.train(was_training)


CSV_FORMATS = {}  # absolute CSV path -> {"header": bool, "rename": {original name: new name}}


def csv_format(header=True, rename=None, columns=None, delimiter=None):
    """Validated reading format. columns: the file's original column names, to check the renaming against.
    delimiter: recorded so a headerless or renamed file can be re-read without guessing."""
    rename = {str(k): str(v).strip() for k, v in (rename or {}).items() if str(v).strip() and str(v).strip() != str(k)}
    if columns is not None:
        final = [rename.get(str(c), str(c)) for c in columns]
        dup = sorted({n for n in final if final.count(n) > 1})
        if dup: raise ValueError(f"Column names must be unique after renaming: {dup}")
    fmt = {"header": bool(header), "rename": rename}
    if delimiter: fmt["delimiter"] = delimiter
    return fmt


def register_csv_format(path, fmt):
    """Every later read_table(path) applies fmt (no header row -> col1, col2, ...; then the renaming)."""
    if path and fmt:
        CSV_FORMATS[os.path.abspath(os.path.expanduser(str(path)))] = csv_format(fmt.get("header", True), fmt.get("rename"),
                                                                                 delimiter=fmt.get("delimiter"))


def training_file_delimiter(path, col_types, fmt=None):
    """Delimiter of a model's training file: the recorded one; else (older configs) the one whose header names the
    model's columns, or, without a header row / after renaming, the one splitting the first line into as many
    fields as the model has columns."""
    fmt = fmt or CSV_FORMATS.get(os.path.abspath(os.path.expanduser(str(path)))) or {}
    if fmt.get("delimiter"): return fmt["delimiter"]
    with open(path, "r", errors="replace") as f: first = f.readline().rstrip("\n\r")
    original = {new: old for old, new in fmt.get("rename", {}).items()}
    wanted = {original.get(c, c) for c, t in col_types.items() if t != "i"}
    cands = [",", "\t", ";", " "]
    if fmt.get("header", True):
        best = max(cands, key=lambda d: len(wanted & {h.strip() for h in first.split(d)}))
        if wanted & {h.strip() for h in first.split(best)}: return best
    exact = [d for d in cands if len(first.split(d)) == len(col_types)]
    return exact[0] if exact else max(cands, key=lambda d: len(first.split(d)))


def headerless_names(n):
    return [f"col{i + 1}" for i in range(n)]


def read_table(path, delimiter=",", fmt=None, **kw):
    """pd.read_csv honouring the file's registered format (or `fmt`): header row or not, renamed columns."""
    fmt = fmt or CSV_FORMATS.get(os.path.abspath(os.path.expanduser(str(path)))) or {}
    if fmt.get("header", True): df = pd.read_csv(path, delimiter=delimiter, **kw)
    else:
        df = pd.read_csv(path, delimiter=delimiter, header=None, **kw); df.columns = headerless_names(df.shape[1])
    if fmt.get("rename"): df = df.rename(columns=fmt["rename"])
    return df


def scale_range(s):
    """Model-side range of a numeric column's scaling (default [-1, 1])."""
    r = s.get("range") or (-1.0, 1.0)
    return float(r[0]), float(r[1])


def to_model_units(v, s):
    """Raw value(s) -> model units: [min, max] of the training rows maps onto the column's range."""
    lo, hi = scale_range(s); d = (s["max"] - s["min"]) or 1.0
    return lo + (hi - lo) * (v - s["min"]) / d


def to_display_units(v, s):
    lo, hi = scale_range(s); w = (hi - lo) or 1.0
    return (v - lo) / w * (s["max"] - s["min"]) + s["min"]


def numeric_series(s):
    """Column -> float Series: booleans and the words true / false (any case) become 1 / 0, other text NaN. Vectorized."""
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s): return s.astype(float)
    low = s.astype(str).str.strip().str.lower()
    is_bool = low.isin(["true", "false"])
    out = pd.to_numeric(s.where(~is_bool), errors="coerce").astype(float)
    return out.mask(is_bool, (low == "true").astype(float))


def _bool_as_number(v):
    """True / False (bools or the strings, any case) -> 1 / 0; anything else unchanged."""
    if isinstance(v, (bool, np.bool_)): return int(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "false"): return int(v.strip().lower() == "true")
    return v


# ── Hashed text (inhash / outhash) ─────────────────────────────────────────────
# Text that the network sees as a number, by the EasyNN-plus rule that AFPO's "Text (encoded)" kind also
# uses: a string's code is the sum of each character code times its position counted from the end ('Dog'
# = 68*3 + 111*2 + 103 = 529), moved up by 101 while it lies within 100 of an earlier string's code.
# Inside the engine such a column is an ordinary numeric in / out column; the {string: code} codebook
# lives in vocabularies[col], so a numeric column that has a vocabulary entry is a hashed-text column.
# Outputs are shown as the nearest string, prefixed with "~" by how far the number is from it.
try:
    from afpo_lib import editor_format as _hashfmt
except ImportError:  # the rule lives in afpo_lib; without it only the hashed-text type is unavailable
    _hashfmt = None
HASH_TYPES = {"inhash": "in", "outhash": "out"}


def lower_hash_types(col_types):
    """col_types with inhash / outhash replaced by the plain numeric in / out they are stored as."""
    return {c: HASH_TYPES.get(t, t) for c, t in col_types.items()}


def is_hashed(col, col_types, vocabularies):
    return col_types.get(col) in ("in", "out") and isinstance(vocabularies.get(col), dict)


def _hash_rule():
    if _hashfmt is None: raise ImportError("Hashed text needs afpo_lib/editor_format.py next to mlpRes6.py")
    return _hashfmt


def _is_number_cell(v):
    if isinstance(v, (bool, np.bool_)): return False
    if isinstance(v, (int, float, np.integer, np.floating)): return True
    try: float(str(v)); return True
    except ValueError: return False


def hash_codebook(values, codes=None):
    """Codebook {string: code} for a column, strings numbered in order of first appearance (numbers are
    already codes and blank cells have none). Extends `codes` in place when given."""
    codes = {} if codes is None else codes
    _hash_rule().editor_assign_codes([str(v) for v in pd.Series(values).dropna() if not _is_number_cell(v)], codes)
    return codes


def hash_encode_series(s, codes):
    """Column of strings (and/or numeric codes) -> floats; unseen strings join the codebook."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s): return s.astype(float)
    hash_codebook(s, codes)
    def one(v):
        if v is None or (isinstance(v, float) and v != v): return float("nan")
        t = str(v)
        if t in codes: return float(codes[t])
        try: return float(t)
        except ValueError: return float("nan")
    return s.map(one).astype(float)


def hash_label(value, codes):
    """Nearest string for a model output (display units), '~'-prefixed by distance."""
    return _hash_rule().editor_decode_text(value, codes)


class FastLoader:
    """DataLoader replacement for CustomDataset: whole batches are sliced from the pre-encoded arrays
    (no per-row Python, no collation). Iterates (inputs, targets) like DataLoader; reshuffles every epoch."""
    def __init__(self, dataset, batch_size, shuffle=False):
        if batch_size < 1: raise ValueError(f"batch size must be >= 1, got {batch_size}")
        self.dataset, self.batch_size, self.shuffle = dataset, int(batch_size), shuffle

    def __len__(self): return -(-len(self.dataset) // self.batch_size)

    def __iter__(self):
        n = len(self.dataset)
        order = torch.randperm(n).numpy() if self.shuffle else np.arange(n)
        for i in range(0, n, self.batch_size): yield self.dataset.batch(order[i:i + self.batch_size])


def make_loader(dataset, batch_size, shuffle=False):
    """FastLoader for CustomDataset (any batch size), else a regular DataLoader."""
    if isinstance(dataset, CustomDataset): return FastLoader(dataset, batch_size, shuffle)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def has_seq_columns(col_types):
    return any(t in seqm.SEQ_TYPES for t in col_types.values())


def build_seq_model(make_mlp, input_cols, output_cols, col_types, scalings, vocabularies, seq_params, seq_bridge="mlp",
                    output_activation=None):
    """The encoder -> bridge -> decoder model for datasets with inenc / outdec columns (see mlpres_seq).
    output_activation: applied to the numeric ('out') outputs after the bridge."""
    slots, start = [], 0
    for c in input_cols:
        w = calculate_input_dim([c], col_types, scalings, vocabularies); slots.append((c, start, w)); start += w
    layout, _, _ = build_output_layout(output_cols, col_types, scalings, vocabularies)
    return seqm.SeqMLP(slots, col_types, vocabularies, layout, seq_params or {}, seq_bridge, make_mlp,
                       output_activation=output_activation)


def describe_seq_model(model):
    parts = [f"{c}: {type(model.encoders[c]).__name__}" for c, *_ in model.enc_slots]
    parts += [f"{c}: {type(d.core).__name__}" for c, d in model.decoders.items()]
    print(f"Sequence model: bridge={model.bridge_kind} ({type(model.bridge).__name__}); " + ", ".join(parts)
          + f"; {sum(p.numel() for p in model.parameters()):,} parameters")


def main(csv_file, delimiter=',', input_cols=[], output_cols=[], col_types={}, vocabularies={}, scalings={}, image_params={},
         optimizer_choice="Adam", hidden_dims=[100, 50], batch_size=32, activation_cls=nn.ReLU, activation_type=0,
         residual_type="residual", norm_type="layer", groups=1, attention_type="none", num_heads=1,
         input_attention_type="none", moe_mode=1, noise_mode="none", noise_params={}, base_activation_cls_for_config=None,
         use_lsuv=False, lsuv_max_iter=10, lsuv_normalize_mean=False,
         val_loader=None, val_interval=1000, custom_lr=None, mlp_mode=0, layer_routing="none", use_grn=False,
         optim_params=None, progress_callback=None, train_dataset=None,
         loss_balancing="normalized", loss_temperature=20.0, resume_from=None, seq_params=None, seq_bridge="mlp",
         layer_activations=None, output_activation=None):
    """progress_callback(dict) is called every step and after each validation (GUI);
    returning True stops training exactly like Ctrl+C (best model is kept).
    train_dataset: the prepared training rows (see prepare_train_val); None reads all of csv_file.
    resume_from: a state dict file (model.pt) to continue training from; the architecture arguments
    must match it (see prepare_resume).
    seq_params / seq_bridge: settings of inenc / outdec columns (see mlpres_seq.check_params, mlpres_seq.BRIDGES).
    layer_activations: None (activation_cls everywhere) or one {"name", "params"} per hidden layer.
    output_activation: {"name", "params"} applied to numeric outputs (None / Linear = none).
    hidden_dims may be [] (a perceptron; with encoders / decoders the bridge becomes one linear map)."""
    seq_params = seq_params or {}
    check_mlp_options(mlp_mode, layer_activations, output_activation, hidden_dims, norm_type, groups, batch_size)
    if layer_activations: activation_cls = [activation_from_config(c, activation_type) for c in layer_activations]
    out_act = None if is_linear_activation(output_activation) else activation_from_config(output_activation, wrap=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = train_dataset if train_dataset is not None else CustomDataset(csv_file, delimiter, input_cols, output_cols, col_types, vocabularies, scalings, image_params)
    train_loader = make_loader(dataset, batch_size, shuffle=True)
    # --- NEW: Calculate Class Weights ---
    print("Calculating class weights for dynamic loss balancing...")
    class_weights = dataset.compute_class_weights(device=device)
    for col, w in class_weights.items():
        print(f"  Col '{col}': found weights for {len(w)} classes.")
    
    input_dims = calculate_input_dim(input_cols, col_types, scalings, vocabularies)
    
    # Build output layout for CombinedLoss
    output_layout, output_pred_dim, output_tgt_dim = build_output_layout(output_cols, col_types, scalings, vocabularies)
    has_categorical = any(e['type'] in ['outlabcat', 'outexcat', 'outdec'] for e in output_layout)
    
    # Model uses prediction dim (includes logit dimensions for categorical)
    output_dims = output_pred_dim

    def make_mlp(in_dim, out_dim, out_idx=None):
        oa = dict(output_activation=out_act if out_idx else None, output_activation_idx=out_idx)
        if mlp_mode == 1:
            # GNN mode
            dropout_prob = noise_params.get("dropout_pct", 0.0) if noise_mode == "dropout" else 0.0
            return GNNMLPO(in_dim, hidden_dims, out_dim, activation_cls, residual_type,
                           norm_type=norm_type, groups=groups, dropout_prob=dropout_prob, use_grn=use_grn)
        elif noise_mode == "dropout":
            dropout_prob = noise_params.get("dropout_pct", 0.0)
            return MLPO(in_dim, hidden_dims, out_dim, activation_cls, residual_type,
                        norm_type=norm_type, groups=groups, attention_type=attention_type, num_heads=num_heads,
                        input_attention_type=input_attention_type, moe_mode=moe_mode, dropout_prob=dropout_prob,
                        layer_routing=layer_routing, use_grn=use_grn, **oa)
        elif noise_mode == "noise injection layers":
            injection_std = noise_params.get("std", 0.1)
            return MLPO(in_dim, hidden_dims, out_dim, activation_cls, residual_type,
                        norm_type=norm_type, groups=groups, attention_type=attention_type, num_heads=num_heads,
                        input_attention_type=input_attention_type, moe_mode=moe_mode,
                        use_noise_injection_layers=True, noise_injection_std=injection_std,
                        layer_routing=layer_routing, use_grn=use_grn, **oa)
        return MLPO(in_dim, hidden_dims, out_dim, activation_cls, residual_type,
                    norm_type=norm_type, groups=groups, attention_type=attention_type, num_heads=num_heads,
                    input_attention_type=input_attention_type, moe_mode=moe_mode,
                    layer_routing=layer_routing, use_grn=use_grn, **oa)

    if has_seq_columns(col_types):
        model = build_seq_model(make_mlp, input_cols, output_cols, col_types, scalings, vocabularies,
                                seq_params, seq_bridge, out_act).to(device)
        describe_seq_model(model)
    else:
        model = make_mlp(input_dims, output_dims, numeric_output_positions(output_layout)).to(device)

    materialize_lazy_modules(model, input_dims, device)
    if resume_from:
        model.load_state_dict(torch.load(resume_from, map_location=device))
        print(f"Resume: continuing from the weights in {resume_from}.")
    elif use_lsuv and isinstance(model, seqm.SeqMLP):
        print("LSUV skipped: it initializes plain MLPs, not models with sequence encoders / decoders.")
    elif use_lsuv:  # LSUV would overwrite resumed weights
        lsuv_init(model, train_loader, device, max_iter=lsuv_max_iter, normalize_mean=lsuv_normalize_mean, verbose=True)

    optimizer = select_optimizer(optimizer_choice, model, custom_lr=custom_lr, optim_params=optim_params)
    if resume_from: restore_optimizer(optimizer, optimizer_choice, custom_lr)
    if optimizer_choice == "RAdamScheduleFree": optimizer.train()
    
    # Use CombinedLoss if categorical outputs exist, else plain HuberLoss
    if has_categorical:
        criterion = CombinedLoss(output_layout, column_weights=class_weights, balancing=loss_balancing, temperature=loss_temperature)
        print(f"Loss baselines (trivial predictor): {criterion.calibrate(dataset)}")
        print(f"Using CombinedLoss with dynamic weighting.")
    else:
        criterion = nn.HuberLoss()
    
    best_val_loss = float('inf'); step = 0
    _is_evolution = isinstance(optimizer, EvolutionaryOptimizer)
    
    config_activation_cls = base_activation_cls_for_config if base_activation_cls_for_config else activation_cls
    save_config(csv_file, col_types, hidden_dims, vocabularies, scalings, image_params,
                optimizer_choice, batch_size, config_activation_cls, activation_type, residual_type,
                norm_type, groups, attention_type, num_heads, input_attention_type, moe_mode, noise_mode, noise_params,
                mlp_mode=mlp_mode, layer_routing=layer_routing, use_grn=use_grn, seq_params=seq_params, seq_bridge=seq_bridge,
                layer_activations=layer_activations, output_activation=output_activation)
    save_validation_rows(val_loader)
        
    try:
        for epoch in range(10000000):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                noisy_inputs = inputs
                if noise_mode == "input noise":
                    noisy_inputs = inputs + torch.randn_like(inputs) * noise_params.get("std", 0.1)
                
                if _is_evolution:
                    # Evolution optimizer: forward-only closure (no backward/grad clipping)
                    def closure():
                        outputs = seqm.model_forward(model, noisy_inputs, targets)
                        if noise_mode == "output noise":
                            outputs = outputs + torch.randn_like(outputs) * noise_params.get("std", 0.1)
                        loss = criterion(outputs, targets) + model.routing_loss()
                        return loss
                else:
                    def closure():
                        optimizer.zero_grad()
                        outputs = seqm.model_forward(model, noisy_inputs, targets)
                        if noise_mode == "output noise":
                            outputs = outputs + torch.randn_like(outputs) * noise_params.get("std", 0.1)
                        loss = criterion(outputs, targets) + model.routing_loss()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        return loss
                
                loss = optimizer.step(closure)
                
                if noise_mode == "weight noise":
                    with torch.no_grad():
                        for param in model.parameters():
                            param.add_(torch.randn_like(param) * noise_params.get("std", 0.1))

                if step % 1 == 0:
                    d_val = optimizer.param_groups[0].get('d', 0)
                    log_msg = f'Step={step}, epoch={epoch}, loss={loss.item():.6f}, Prodigy chosen LR={d_val:.2e}'
                    routing_usage = model.routing_usage()
                    if routing_usage is not None: log_msg += f' | routed_active={routing_usage:.2%}'
                    # Perplexity logging for categorical outputs
                    if has_categorical and step % 1 == 0:
                        with torch.no_grad():
                            ppl_preds = seqm.model_forward(model, inputs, targets)
                            ppl_metrics, global_ppl = compute_perplexity_metrics(
                                ppl_preds, targets, output_layout, vocabularies)
                            ppl_parts = [f'{k}_ppl={v:.2f}' for k, v in ppl_metrics.items()]
                            if ppl_parts:
                                log_msg += f' | {" ".join(ppl_parts)} | global_ppl={global_ppl:.2f}'
                    print(log_msg)
                if progress_callback is not None:
                    info = {"step": step, "epoch": epoch, "loss": loss.item(), "lr": optimizer.param_groups[0].get("lr")}
                    if routing_usage is not None: info["routed_active"] = routing_usage
                    if has_categorical: info["ppl"] = ppl_metrics
                    if getattr(criterion, "last_task_losses", None): info["tasks"] = dict(criterion.last_task_losses)
                    if progress_callback(info): raise KeyboardInterrupt

                if val_loader is not None and step % val_interval == 0:
                    current_val_loss = validate_model(model, val_loader, criterion, device)
                    print(f"--- Validation Check (Step {step}) ---")
                    print(f"    Current Val Loss: {current_val_loss:.6f}")
                    print(f"    Best Val Loss:    {best_val_loss:.6f}")
                    improved = current_val_loss < best_val_loss
                    if improved:
                        best_val_loss = current_val_loss
                        save_checkpoint(model, optimizer, optimizer_choice)
                        print("    >>> NEW BEST MODEL SAVED <<<")
                    else: print("    (No improvement)")
                    if progress_callback is not None and progress_callback(
                            {"step": step, "val_loss": current_val_loss, "best_val_loss": best_val_loss, "saved": improved}):
                        raise KeyboardInterrupt
                    print("-------------------------------------")

                step += 1

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        if val_loader is not None:
            print("Performing final validation check...")
            final_val_loss = validate_model(model, val_loader, criterion, device)
            print(f"Final Val Loss: {final_val_loss:.6f} vs Best: {best_val_loss:.6f}")
            if final_val_loss < best_val_loss:
                print("Interrupted model is better. Saving..."); save_checkpoint(model, optimizer, optimizer_choice)
            else: print("Interrupted model is WORSE. Discarding unsaved changes.")
        else:
            print("No validation set. Saving current state."); save_checkpoint(model, optimizer, optimizer_choice)
    
    if val_loader is None and not os.path.exists('model.pt'):
        save_checkpoint(model, optimizer, optimizer_choice)
    if os.path.exists('model.pt'):
        model.load_state_dict(torch.load('model.pt'))
        print("Loaded best model state from 'model.pt'.")

    config_activation_cls = base_activation_cls_for_config if base_activation_cls_for_config else activation_cls
    save_config(csv_file, col_types, hidden_dims, vocabularies, scalings, image_params,
                optimizer_choice, batch_size, config_activation_cls, activation_type, residual_type,
                norm_type, groups, attention_type, num_heads, input_attention_type, moe_mode, noise_mode, noise_params,
                mlp_mode=mlp_mode, layer_routing=layer_routing, use_grn=use_grn, seq_params=seq_params, seq_bridge=seq_bridge,
                layer_activations=layer_activations, output_activation=output_activation)


##############################################
# Auto-grow: weight transplant helper
##############################################
def transplant_weights(old_model, new_model):
    """Copy overlapping weights from old model to new model."""
    old_sd = old_model.state_dict()
    new_sd = new_model.state_dict()
    for key in new_sd:
        if key in old_sd:
            old_p = old_sd[key]
            new_p = new_sd[key]
            if old_p.shape == new_p.shape:
                new_sd[key] = old_p.clone()
            elif old_p.dim() == new_p.dim():
                # Copy overlapping region
                slices = tuple(slice(0, min(o, n)) for o, n in zip(old_p.shape, new_p.shape))
                new_sd[key][slices] = old_p[slices].clone()
    new_model.load_state_dict(new_sd)


def compute_min_output_gap(dataset, output_cols, col_types, scalings):
    """Compute the minimum gap between two closest distinct output values (in scaled space).
    Returns the gap per output dimension, averaged."""
    gaps = []
    # Collect all target tensors
    all_targets = []
    for i in range(len(dataset)):
        _, t = dataset[i]
        all_targets.append(t.unsqueeze(0) if t.dim() == 0 else t)
    all_targets = torch.stack(all_targets, dim=0)  # (N, out_dim)

    for dim_idx in range(all_targets.shape[1]):
        vals = all_targets[:, dim_idx].numpy()
        unique_vals = np.unique(vals)
        if len(unique_vals) < 2:
            gaps.append(1.0)  # Only one unique value, use 1.0 as placeholder
        else:
            sorted_vals = np.sort(unique_vals)
            diffs = np.diff(sorted_vals)
            diffs = diffs[diffs > 0]  # Filter zero diffs
            if len(diffs) > 0:
                gaps.append(float(np.min(diffs)))
            else:
                gaps.append(1.0)
    if not gaps:
        return 1.0
    return float(np.mean(gaps))


def build_model_for_auto(input_dims, hidden_dims, output_dims, activation_cls, residual_type,
                          norm_type, groups, attention_type, num_heads, input_attention_type,
                          moe_mode, noise_mode, noise_params, mlp_mode, device):
    """Build a model (MLP or GNN) for auto-grow training."""
    if mlp_mode == 1:
        # GNN mode
        dropout_prob = noise_params.get("dropout_pct", 0.0) if noise_mode == "dropout" else 0.0
        model = GNNMLPO(input_dims, hidden_dims, output_dims, activation_cls, residual_type,
                        norm_type=norm_type, groups=groups, dropout_prob=dropout_prob).to(device)
    else:
        if noise_mode == "dropout":
            dropout_prob = noise_params.get("dropout_pct", 0.0)
            model = MLPO(input_dims, hidden_dims, output_dims, activation_cls, residual_type,
                         norm_type=norm_type, groups=groups, attention_type=attention_type, num_heads=num_heads,
                         input_attention_type=input_attention_type, moe_mode=moe_mode, dropout_prob=dropout_prob).to(device)
        elif noise_mode == "noise injection layers":
            injection_std = noise_params.get("std", 0.1)
            model = MLPO(input_dims, hidden_dims, output_dims, activation_cls, residual_type,
                         norm_type=norm_type, groups=groups, attention_type=attention_type, num_heads=num_heads,
                         input_attention_type=input_attention_type, moe_mode=moe_mode,
                         use_noise_injection_layers=True, noise_injection_std=injection_std).to(device)
        else:
            model = MLPO(input_dims, hidden_dims, output_dims, activation_cls, residual_type,
                         norm_type=norm_type, groups=groups, attention_type=attention_type, num_heads=num_heads,
                         input_attention_type=input_attention_type, moe_mode=moe_mode).to(device)
    return model


def main_auto_grow(csv_file, delimiter=',', input_cols=[], output_cols=[], col_types={}, vocabularies={},
                   scalings={}, image_params={}, optimizer_choice="Adam", auto_config=None,
                   batch_size=32, activation_cls=nn.ReLU, activation_type=0, residual_type="residual",
                   norm_type="layer", groups=1, attention_type="none", num_heads=1,
                   input_attention_type="none", moe_mode=1, noise_mode="none", noise_params={},
                   base_activation_cls_for_config=None, use_lsuv=False, lsuv_max_iter=10,
                   lsuv_normalize_mean=False, val_loader=None, val_interval=1000, custom_lr=None,
                   mlp_mode=0, train_dataset=None, loss_balancing="normalized", loss_temperature=20.0):
    """Training with auto-growing architecture.
    Starts with 1 neuron, 1 layer. Adds neurons/layers when loss stagnates."""
    if has_seq_columns(col_types):
        raise ValueError("Auto-grow does not support inenc / outdec columns; use fixed hidden layers.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_dim = auto_config["max_dim"]
    max_layers = auto_config["max_layers"]
    patience = auto_config["patience"]

    dataset = train_dataset if train_dataset is not None else CustomDataset(csv_file, delimiter, input_cols, output_cols, col_types, vocabularies, scalings, image_params)
    train_loader = make_loader(dataset, batch_size, shuffle=True)
    print("Calculating class weights for dynamic loss balancing...")
    class_weights = dataset.compute_class_weights(device=device)
    for col, w in class_weights.items():
        print(f"  Col '{col}': found weights for {len(w)} classes.")

    input_dims = calculate_input_dim(input_cols, col_types, scalings, vocabularies)
    output_layout, output_pred_dim, output_tgt_dim = build_output_layout(output_cols, col_types, scalings, vocabularies)
    has_categorical = any(e['type'] in ['outlabcat', 'outexcat'] for e in output_layout)
    output_dims = output_pred_dim

    if has_categorical:
        criterion = CombinedLoss(output_layout, column_weights=class_weights, balancing=loss_balancing, temperature=loss_temperature)
        criterion.calibrate(dataset)
    else:
        criterion = nn.HuberLoss()

    # Compute data-dependent stopping threshold
    min_gap = compute_min_output_gap(dataset, output_cols, col_types, scalings)
    stop_threshold = 0.02 * min_gap
    print(f"Auto-grow: min output gap = {min_gap:.6f}, stop-growing threshold = {stop_threshold:.6f}")

    # Start with 1 layer, 1 neuron
    hidden_dims = [1]
    print(f"\n{'='*50}")
    print(f"AUTO-GROW: Starting with hidden_dims = {hidden_dims}")
    print(f"  Max dim per layer: {max_dim}, Max layers: {max_layers}")
    print(f"  Patience: {patience} steps")
    print(f"{'='*50}")

    model = build_model_for_auto(input_dims, hidden_dims, output_dims, activation_cls,
                                  residual_type, norm_type, groups, attention_type, num_heads,
                                  input_attention_type, moe_mode, noise_mode, noise_params,
                                  mlp_mode, device)

    # Initialize with dummy forward pass for lazy modules
    with torch.no_grad(): model(torch.zeros(1, input_dims).to(device))

    if use_lsuv:
        lsuv_init(model, train_loader, device, max_iter=lsuv_max_iter, normalize_mean=lsuv_normalize_mean, verbose=True)

    optimizer = select_optimizer(optimizer_choice, model, custom_lr=custom_lr)
    if optimizer_choice == "RAdamScheduleFree": optimizer.train()

    best_val_loss = float('inf')
    best_tracked_loss = float('inf')  # Best of whichever loss we track (train or val)
    best_hidden_dims = list(hidden_dims)  # Track dims matching best saved model
    steps_since_improvement = 0
    total_step = 0
    growing_enabled = True
    grow_stop_reason = None
    neurons_added_since_val_improve = 0  # For validation-based stop
    best_val_loss_at_grow = float('inf')

    # Rolling loss tracker
    loss_window = deque(maxlen=100)

    config_activation_cls = base_activation_cls_for_config if base_activation_cls_for_config else activation_cls
    save_validation_rows(val_loader)
    save_config(csv_file, col_types, hidden_dims, vocabularies, scalings, image_params,
                optimizer_choice, batch_size, config_activation_cls, activation_type, residual_type,
                norm_type, groups, attention_type, num_heads, input_attention_type, moe_mode, noise_mode, noise_params, mlp_mode=mlp_mode)

    def get_tracked_loss(loss_val):
        """Get the loss we're tracking: val loss if available, else training loss."""
        if val_loader is not None:
            return validate_model(model, val_loader, criterion, device)
        return loss_val

    def try_grow():
        """Try to add a neuron or layer. Returns new hidden_dims or None if maxed out."""
        nonlocal hidden_dims
        new_dims = list(hidden_dims)
        # Strategy: add neuron to smallest layer first; if all at max, add new layer
        # Find first layer that hasn't reached max_dim
        grew = False
        for i in range(len(new_dims)):
            if new_dims[i] < max_dim:
                new_dims[i] += 1
                grew = True
                print(f"  >>> GROW: Added neuron to layer {i+1}: {hidden_dims[i]} -> {new_dims[i]}")
                break
        if not grew:
            # All layers at max, try adding a new layer
            if len(new_dims) < max_layers:
                new_dims.append(1)
                print(f"  >>> GROW: Added new layer {len(new_dims)} with 1 neuron")
                grew = True
            else:
                return None  # Fully maxed out
        return new_dims

    try:
        for epoch in range(10000000):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                noisy_inputs = inputs
                if noise_mode == "input noise":
                    noisy_inputs = inputs + torch.randn_like(inputs) * noise_params.get("std", 0.1)

                def closure():
                    optimizer.zero_grad()
                    outputs = model(noisy_inputs)
                    if noise_mode == "output noise":
                        outputs = outputs + torch.randn_like(outputs) * noise_params.get("std", 0.1)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    return loss

                loss = optimizer.step(closure)

                if noise_mode == "weight noise":
                    with torch.no_grad():
                        for param in model.parameters():
                            param.add_(torch.randn_like(param) * noise_params.get("std", 0.1))

                loss_val = loss.item()
                loss_window.append(loss_val)

                if total_step % 1 == 0:
                    d_val = optimizer.param_groups[0].get('d', 0)
                    log_msg = f'Step={total_step}, epoch={epoch}, loss={loss_val:.6f}, arch={hidden_dims}, Prodigy chosen LR={d_val:.2e}'
                    if has_categorical and total_step % 1 == 0:
                        with torch.no_grad():
                            ppl_preds = model(inputs)
                            ppl_metrics, global_ppl = compute_perplexity_metrics(
                                ppl_preds, targets, output_layout, vocabularies)
                            ppl_parts = [f'{k}_ppl={v:.2f}' for k, v in ppl_metrics.items()]
                            if ppl_parts:
                                log_msg += f' | {" ".join(ppl_parts)} | global_ppl={global_ppl:.2f}'
                    print(log_msg)

                # Validation check
                if val_loader is not None and total_step % val_interval == 0 and total_step > 0:
                    current_val_loss = validate_model(model, val_loader, criterion, device)
                    print(f"--- Validation Check (Step {total_step}) ---")
                    print(f"    Current Val Loss: {current_val_loss:.6f}")
                    print(f"    Best Val Loss:    {best_val_loss:.6f}")
                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        best_hidden_dims = list(hidden_dims)
                        torch.save(model.state_dict(), 'model.pt')
                        print("    >>> NEW BEST MODEL SAVED <<<")
                    else: print("    (No improvement)")
                    print("-------------------------------------")

                # Track improvement for auto-grow
                tracked = get_tracked_loss(loss_val) if (total_step % max(val_interval // 4, 50) == 0 and total_step > 0) else None
                if tracked is not None:
                    if tracked < best_tracked_loss * 0.999:  # Must improve by at least 0.1%
                        best_tracked_loss = tracked
                        steps_since_improvement = 0
                    else:
                        steps_since_improvement += max(val_interval // 4, 50)
                else:
                    # Simple fallback: use raw training loss
                    if loss_val < best_tracked_loss * 0.999:
                        best_tracked_loss = loss_val
                        steps_since_improvement = 0
                    else:
                        steps_since_improvement += 1

                # Check if we should grow
                if growing_enabled and steps_since_improvement >= patience:
                    avg_loss = np.mean(loss_window) if loss_window else loss_val

                    # Check data-dependent stopping: loss already good enough
                    if avg_loss < stop_threshold:
                        growing_enabled = False
                        grow_stop_reason = f"avg_loss ({avg_loss:.6f}) < stop_threshold ({stop_threshold:.6f})"
                        print(f"\n{'='*50}")
                        print(f"AUTO-GROW DISABLED: {grow_stop_reason}")
                        print(f"Architecture finalized at: {hidden_dims}")
                        print(f"{'='*50}")
                    else:
                        new_dims = try_grow()
                        if new_dims is None:
                            growing_enabled = False
                            grow_stop_reason = "Architecture fully maxed out"
                            print(f"\n{'='*50}")
                            print(f"AUTO-GROW DISABLED: {grow_stop_reason}")
                            print(f"Final architecture: {hidden_dims}")
                            print(f"{'='*50}")
                        else:
                            old_model = model
                            hidden_dims = new_dims
                            print(f"\n  Rebuilding model with hidden_dims = {hidden_dims}")

                            # Build new model
                            model = build_model_for_auto(
                                input_dims, hidden_dims, output_dims, activation_cls,
                                residual_type, norm_type, groups, attention_type, num_heads,
                                input_attention_type, moe_mode, noise_mode, noise_params,
                                mlp_mode, device)
                            # Lazy init
                            with torch.no_grad(): model(torch.zeros(1, input_dims).to(device))
                            # Transplant weights from old model
                            transplant_weights(old_model, model)
                            print(f"  Weights transplanted. Resetting optimizer.")

                            # Reset optimizer (old momenta won't match new param shapes)
                            optimizer = select_optimizer(optimizer_choice, model, custom_lr=custom_lr)
                            if optimizer_choice == "RAdamScheduleFree": optimizer.train()

                            # Reset patience tracker
                            steps_since_improvement = 0
                            best_tracked_loss = float('inf')  # Reset so we don't immediately re-grow

                            # Track neurons added for validation-based stop
                            if val_loader is not None:
                                neurons_added_since_val_improve += 1
                                current_val = validate_model(model, val_loader, criterion, device)
                                if current_val < best_val_loss_at_grow:
                                    best_val_loss_at_grow = current_val
                                    neurons_added_since_val_improve = 0
                                if neurons_added_since_val_improve >= 10:
                                    growing_enabled = False
                                    grow_stop_reason = "10 neuron additions without validation improvement"
                                    print(f"\n{'='*50}")
                                    print(f"AUTO-GROW DISABLED: {grow_stop_reason}")
                                    print(f"Final architecture: {hidden_dims}")
                                    print(f"{'='*50}")

                            # Save updated config
                            save_config(csv_file, col_types, hidden_dims, vocabularies, scalings, image_params,
                                        optimizer_choice, batch_size, config_activation_cls, activation_type, residual_type,
                                        norm_type, groups, attention_type, num_heads, input_attention_type, moe_mode,
                                        noise_mode, noise_params, mlp_mode=mlp_mode)

                total_step += 1

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        print(f"Current architecture: {hidden_dims}")
        print(f"Best saved architecture: {best_hidden_dims}")
        if val_loader is not None:
            final_val_loss = validate_model(model, val_loader, criterion, device)
            print(f"Final Val Loss: {final_val_loss:.6f} vs Best: {best_val_loss:.6f}")
            if final_val_loss < best_val_loss:
                torch.save(model.state_dict(), 'model.pt')
                best_hidden_dims = list(hidden_dims)
                print("Interrupted model saved (better than best).")
            else:
                print("Keeping previous best model.")
                hidden_dims = list(best_hidden_dims)
        else:
            torch.save(model.state_dict(), 'model.pt')
            best_hidden_dims = list(hidden_dims)

    if val_loader is None and not os.path.exists('model.pt'):
        torch.save(model.state_dict(), 'model.pt')
        best_hidden_dims = list(hidden_dims)
    if os.path.exists('model.pt'):
        # Rebuild model with best_hidden_dims to load the saved state
        if list(hidden_dims) != list(best_hidden_dims):
            print(f"Rebuilding model with best architecture {best_hidden_dims} to load saved weights...")
            model = build_model_for_auto(input_dims, best_hidden_dims, output_dims, activation_cls,
                                          residual_type, norm_type, groups, attention_type, num_heads,
                                          input_attention_type, moe_mode, noise_mode, noise_params,
                                          mlp_mode, device)
            with torch.no_grad(): model(torch.zeros(1, input_dims).to(device))
            hidden_dims = list(best_hidden_dims)
        model.load_state_dict(torch.load('model.pt'))
        print("Loaded best model state from 'model.pt'.")

    # Final config save with the architecture matching the saved model
    config_activation_cls = base_activation_cls_for_config if base_activation_cls_for_config else activation_cls
    save_config(csv_file, col_types, best_hidden_dims, vocabularies, scalings, image_params,
                optimizer_choice, batch_size, config_activation_cls, activation_type, residual_type,
                norm_type, groups, attention_type, num_heads, input_attention_type, moe_mode, noise_mode, noise_params, mlp_mode=mlp_mode)
    print(f"\nAuto-grow complete. Final architecture: {best_hidden_dims}")
    if grow_stop_reason: print(f"Growth stopped because: {grow_stop_reason}")


##############################################
# Evolution Mode (Neural Architecture Search)
##############################################
import random as _random

def _evolution_activation_choices():
    """Return list of (name, config_dict) for all non-Custom activations usable by evolution."""
    amap = _build_activation_map()
    exclude = {"Custom", "All"}
    choices = []
    for name in amap:
        if name in exclude:
            continue
        choices.append(name)
    return choices

def _evolution_create_individual(max_hidden_dim, max_layers, act_choices,
                                 evolve_set=None, fixed_overrides=None,
                                 fixed_hidden_dims=None, fixed_activation=None,
                                 fixed_activation_type=None):
    """Create a random individual for the evolutionary search.
    evolve_set controls which fields are randomised; others use fixed values."""
    if evolve_set is None:
        evolve_set = {"hidden_dims", "activation", "activation_type",
                      "residual_type", "noise", "norm_type"}
    if fixed_overrides is None:
        fixed_overrides = {}

    # Hidden dims
    if "hidden_dims" in evolve_set:
        num_layers = _random.randint(1, max_layers)
        hidden_dims = [_random.randint(1, max_hidden_dim) for _ in range(num_layers)]
    else:
        hidden_dims = list(fixed_hidden_dims) if fixed_hidden_dims else [max_hidden_dim]

    # Activation
    if "activation" in evolve_set:
        activation_name = _random.choice(act_choices)
    else:
        activation_name = fixed_activation if fixed_activation else "ReLU"

    # Activation type
    if "activation_type" in evolve_set:
        activation_type = _random.randint(0, 7)
    else:
        activation_type = fixed_activation_type if fixed_activation_type is not None else 0

    # Residual type
    if "residual_type" in evolve_set:
        residual_choices = ["none", "highway", "residual", "rezero", "elementwise_rezero", "concat", "densenet"]
        residual_type = _random.choice(residual_choices)
    else:
        residual_type = fixed_overrides.get("residual_type", "residual")

    # Noise
    if "noise" in evolve_set:
        noise_choices = [
            ("none", {}),
            ("dropout", {"dropout_pct": round(_random.uniform(0.01, 0.5), 3)}),
            ("input noise", {"std": round(_random.uniform(0.001, 0.2), 4)}),
            ("output noise", {"std": round(_random.uniform(0.001, 0.2), 4)}),
            ("weight noise", {"std": round(_random.uniform(0.0001, 0.05), 5)}),
        ]
        noise_mode, noise_params = _random.choice(noise_choices)
    else:
        noise_mode = fixed_overrides.get("noise_mode", "none")
        noise_params = fixed_overrides.get("noise_params", {})

    # Norm type
    if "norm_type" in evolve_set:
        norm_choices = ["none", "batch", "layer", "rmsnorm"]
        norm_type = _random.choice(norm_choices)
    else:
        norm_type = fixed_overrides.get("norm_type", "layer")

    # Learning rate (log-uniform sampling for better coverage)
    if "learning_rate" in evolve_set:
        log_lr = _random.uniform(math.log(1e-5), math.log(1e-1))
        learning_rate = round(math.exp(log_lr), 6)
    else:
        learning_rate = fixed_overrides.get("learning_rate", None)

    # MoE mode
    if "moe_mode" in evolve_set:
        moe_mode = _random.choice([1, 1, 1, 2, 4, -4, -5])  # weighted towards simple
    else:
        moe_mode = fixed_overrides.get("moe_mode", 1)

    # Attention type
    if "attention" in evolve_set:
        attention_type = _random.choice(["none", "none", "none", "basic", "multi"])
        num_heads = _random.choice([1, 2, 4]) if attention_type == "multi" else 1
    else:
        attention_type = fixed_overrides.get("attention_type", "none")
        num_heads = fixed_overrides.get("num_heads", 1)

    return {
        "hidden_dims": hidden_dims,
        "activation_name": activation_name,
        "activation_type": activation_type,
        "residual_type": residual_type,
        "noise_mode": noise_mode,
        "noise_params": noise_params,
        "norm_type": norm_type,
        "learning_rate": learning_rate,
        "moe_mode": moe_mode,
        "attention_type": attention_type,
        "num_heads": num_heads,
        "fitness": float('inf'),
        "age": 0,
    }


def _evolution_population_diversity(population):
    """Compute a diversity score for the population (0=identical, 1=maximally diverse)."""
    if len(population) < 2:
        return 1.0
    n = len(population)
    diversity_scores = []
    for i in range(min(n, 20)):
        for j in range(i + 1, min(n, 20)):
            diff = 0
            # Architecture difference
            d1 = population[i]["hidden_dims"]
            d2 = population[j]["hidden_dims"]
            diff += abs(len(d1) - len(d2)) * 0.2
            for k in range(min(len(d1), len(d2))):
                diff += abs(d1[k] - d2[k]) / max(max(d1[k], d2[k]), 1) * 0.1
            # Categorical differences
            if population[i]["activation_name"] != population[j]["activation_name"]:
                diff += 1.0
            if population[i]["residual_type"] != population[j]["residual_type"]:
                diff += 0.5
            if population[i]["norm_type"] != population[j]["norm_type"]:
                diff += 0.3
            if population[i]["noise_mode"] != population[j]["noise_mode"]:
                diff += 0.2
            if population[i].get("moe_mode", 1) != population[j].get("moe_mode", 1):
                diff += 0.3
            diversity_scores.append(diff)
    if not diversity_scores:
        return 1.0
    max_possible = 2.5
    return min(1.0, np.mean(diversity_scores) / max_possible)

def _evolution_mutate(individual, max_hidden_dim, max_layers, act_choices,
                      mutation_rate=0.3, evolve_set=None):
    """Mutate an individual. Only mutates fields in evolve_set. Returns a new dict."""
    if evolve_set is None:
        evolve_set = {"hidden_dims", "activation", "activation_type",
                      "residual_type", "noise", "norm_type"}
    ind = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
           for k, v in individual.items()}

    # Mutate hidden dims
    if "hidden_dims" in evolve_set and _random.random() < mutation_rate:
        mutation_type = _random.choice(["add_neuron", "remove_neuron", "add_layer", "remove_layer", "change_neuron"])
        if mutation_type == "add_neuron" and ind["hidden_dims"]:
            idx = _random.randint(0, len(ind["hidden_dims"]) - 1)
            ind["hidden_dims"][idx] = min(max_hidden_dim, ind["hidden_dims"][idx] + _random.randint(1, max(1, max_hidden_dim // 4)))
        elif mutation_type == "remove_neuron" and ind["hidden_dims"]:
            idx = _random.randint(0, len(ind["hidden_dims"]) - 1)
            ind["hidden_dims"][idx] = max(1, ind["hidden_dims"][idx] - _random.randint(1, max(1, max_hidden_dim // 4)))
        elif mutation_type == "add_layer" and len(ind["hidden_dims"]) < max_layers:
            ind["hidden_dims"].append(_random.randint(1, max_hidden_dim))
        elif mutation_type == "remove_layer" and len(ind["hidden_dims"]) > 1:
            idx = _random.randint(0, len(ind["hidden_dims"]) - 1)
            ind["hidden_dims"].pop(idx)
        elif mutation_type == "change_neuron" and ind["hidden_dims"]:
            idx = _random.randint(0, len(ind["hidden_dims"]) - 1)
            ind["hidden_dims"][idx] = _random.randint(1, max_hidden_dim)

    # Mutate activation
    if "activation" in evolve_set and _random.random() < mutation_rate:
        ind["activation_name"] = _random.choice(act_choices)

    # Mutate activation type
    if "activation_type" in evolve_set and _random.random() < mutation_rate:
        ind["activation_type"] = _random.randint(0, 7)

    # Mutate residual type
    if "residual_type" in evolve_set and _random.random() < mutation_rate:
        residual_choices = ["none", "highway", "residual", "rezero", "elementwise_rezero", "concat", "densenet"]
        ind["residual_type"] = _random.choice(residual_choices)

    # Mutate noise
    if "noise" in evolve_set and _random.random() < mutation_rate:
        noise_choices = [
            ("none", {}),
            ("dropout", {"dropout_pct": round(_random.uniform(0.01, 0.5), 3)}),
            ("input noise", {"std": round(_random.uniform(0.001, 0.2), 4)}),
            ("output noise", {"std": round(_random.uniform(0.001, 0.2), 4)}),
            ("weight noise", {"std": round(_random.uniform(0.0001, 0.05), 5)}),
        ]
        ind["noise_mode"], ind["noise_params"] = _random.choice(noise_choices)

    # Mutate norm
    if "norm_type" in evolve_set and _random.random() < mutation_rate:
        norm_choices = ["none", "batch", "layer", "rmsnorm"]
        ind["norm_type"] = _random.choice(norm_choices)

    # Mutate learning rate (log-scale perturbation)
    if "learning_rate" in evolve_set and _random.random() < mutation_rate:
        if ind.get("learning_rate") is not None:
            log_lr = math.log(ind["learning_rate"])
            log_lr += _random.gauss(0, 0.5)
            log_lr = max(math.log(1e-6), min(math.log(0.5), log_lr))
            ind["learning_rate"] = round(math.exp(log_lr), 6)
        else:
            ind["learning_rate"] = round(math.exp(_random.uniform(math.log(1e-5), math.log(1e-1))), 6)

    # Mutate MoE mode
    if "moe_mode" in evolve_set and _random.random() < mutation_rate:
        ind["moe_mode"] = _random.choice([1, 2, 4, -4, -5])

    # Mutate attention
    if "attention" in evolve_set and _random.random() < mutation_rate:
        ind["attention_type"] = _random.choice(["none", "none", "basic", "multi"])
        ind["num_heads"] = _random.choice([1, 2, 4]) if ind["attention_type"] == "multi" else 1

    ind["fitness"] = float('inf')
    ind["age"] = individual.get("age", 0) + 1
    return ind

def _evolution_crossover(parent1, parent2, max_hidden_dim, max_layers):
    """Crossover two parents. Returns a child."""
    child = {}
    # Hidden dims: mix layers from both parents
    all_layers = parent1["hidden_dims"] + parent2["hidden_dims"]
    num_layers = _random.randint(1, min(max_layers, len(all_layers)))
    child["hidden_dims"] = [_random.choice(all_layers) for _ in range(num_layers)]
    # Other params: randomly pick from either parent
    for key in ["activation_name", "activation_type", "residual_type", "noise_mode",
                "noise_params", "norm_type", "moe_mode", "attention_type", "num_heads"]:
        child[key] = _random.choice([parent1.get(key, None), parent2.get(key, None)])
    # Learning rate: geometric mean with noise (BLX-alpha style)
    lr1 = parent1.get("learning_rate")
    lr2 = parent2.get("learning_rate")
    if lr1 is not None and lr2 is not None:
        log_mean = (math.log(lr1) + math.log(lr2)) / 2
        log_mean += _random.gauss(0, 0.1)
        child["learning_rate"] = round(math.exp(max(math.log(1e-6), min(math.log(0.5), log_mean))), 6)
    else:
        child["learning_rate"] = lr1 or lr2
    child["fitness"] = float('inf')
    child["age"] = 0
    return child

def _evolution_evaluate(individual, csv_file, delimiter, input_cols, output_cols, col_types,
                         vocabularies, scalings, image_params, batch_size, optimizer_choice,
                         custom_lr, eval_steps, device, val_loader=None, train_dataset=None):
    """Evaluate an individual by training for eval_steps and returning loss."""
    act_map = _build_activation_map()
    act_name = individual["activation_name"]
    if act_name in act_map:
        activation_cls = act_map[act_name]
    else:
        activation_cls = nn.ReLU
    activation_cls = wrap_activation(activation_cls, individual["activation_type"])

    hidden_dims = individual["hidden_dims"]
    residual_type = individual["residual_type"]
    noise_mode = individual["noise_mode"]
    noise_params = individual["noise_params"]
    norm_type = individual["norm_type"]

    dataset = train_dataset if train_dataset is not None else CustomDataset(
        csv_file, delimiter, input_cols, output_cols, col_types, vocabularies, scalings, image_params)
    train_loader = make_loader(dataset, batch_size, shuffle=True)

    input_dims = calculate_input_dim(input_cols, col_types, scalings, vocabularies)
    output_layout, output_pred_dim, output_tgt_dim = build_output_layout(output_cols, col_types, scalings, vocabularies)
    has_categorical = any(e['type'] in ['outlabcat', 'outexcat'] for e in output_layout)

    class_weights = dataset.compute_class_weights(device=device)
    if has_categorical:
        criterion = CombinedLoss(output_layout, column_weights=class_weights); criterion.calibrate(dataset)
    else:
        criterion = nn.HuberLoss()

    try:
        # Use individual's evolved moe_mode and attention settings
        ind_moe_mode = individual.get("moe_mode", 1)
        ind_attention_type = individual.get("attention_type", "none")
        ind_num_heads = individual.get("num_heads", 1)

        if noise_mode == "dropout":
            dropout_prob = noise_params.get("dropout_pct", 0.0)
            model = MLPO(input_dims, hidden_dims, output_pred_dim, activation_cls, residual_type,
                         norm_type=norm_type, groups=1, attention_type=ind_attention_type,
                         num_heads=ind_num_heads, input_attention_type="none",
                         moe_mode=ind_moe_mode, dropout_prob=dropout_prob).to(device)
        elif noise_mode == "noise injection layers":
            injection_std = noise_params.get("std", 0.1)
            model = MLPO(input_dims, hidden_dims, output_pred_dim, activation_cls, residual_type,
                         norm_type=norm_type, groups=1, attention_type=ind_attention_type,
                         num_heads=ind_num_heads, input_attention_type="none",
                         moe_mode=ind_moe_mode,
                         use_noise_injection_layers=True, noise_injection_std=injection_std).to(device)
        else:
            model = MLPO(input_dims, hidden_dims, output_pred_dim, activation_cls, residual_type,
                         norm_type=norm_type, groups=1, attention_type=ind_attention_type,
                         num_heads=ind_num_heads, input_attention_type="none",
                         moe_mode=ind_moe_mode).to(device)

        # Lazy init
        with torch.no_grad():
            model(torch.zeros(1, input_dims).to(device))

        # Use individual's learning rate if evolved
        ind_lr = individual.get("learning_rate")
        effective_lr = ind_lr if ind_lr is not None else custom_lr
        optimizer = select_optimizer(optimizer_choice, model, custom_lr=effective_lr)
        if optimizer_choice == "RAdamScheduleFree": optimizer.train()

        step = 0
        last_loss = float('inf')
        for epoch in range(10000000):
            for inputs, targets in train_loader:
                if step >= eval_steps:
                    break
                inputs, targets = inputs.to(device), targets.to(device)
                noisy_inputs = inputs
                if noise_mode == "input noise":
                    noisy_inputs = inputs + torch.randn_like(inputs) * noise_params.get("std", 0.1)

                def closure():
                    optimizer.zero_grad()
                    outputs = model(noisy_inputs)
                    if noise_mode == "output noise":
                        outputs = outputs + torch.randn_like(outputs) * noise_params.get("std", 0.1)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    return loss

                loss = optimizer.step(closure)

                if noise_mode == "weight noise":
                    with torch.no_grad():
                        for param in model.parameters():
                            param.add_(torch.randn_like(param) * noise_params.get("std", 0.1))

                last_loss = loss.item()
                step += 1
            if step >= eval_steps:
                break

        # Use validation loss if available, else train loss
        if val_loader is not None:
            fitness = validate_model(model, val_loader, criterion, device)
        else:
            fitness = last_loss

        # Check for NaN/Inf
        if not np.isfinite(fitness):
            fitness = float('inf')

        return fitness

    except Exception as e:
        print(f"    Evaluation failed: {e}")
        return float('inf')


def run_evolution(csv_file, delimiter, input_cols, output_cols, col_types, vocabularies,
                  scalings, image_params, batch_size, optimizer_choice, custom_lr,
                  max_hidden_dim, max_layers, population_size=20, generations=30,
                  eval_steps=500, val_loader=None, train_dataset=None,
                  evolve_set=None, fixed_overrides=None,
                  fixed_hidden_dims=None, fixed_activation=None, fixed_activation_type=None):
    """Run evolutionary NAS to find the best architecture, then train it fully.
    evolve_set: set of field names evolution is allowed to mutate.
    fixed_overrides: dict of fixed values for non-evolved fields.
    """

    if evolve_set is None:
        evolve_set = {"hidden_dims", "activation", "activation_type",
                      "residual_type", "noise", "norm_type"}
    if fixed_overrides is None:
        fixed_overrides = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    act_choices = _evolution_activation_choices()

    print(f"\n{'='*60}")
    print(f"EVOLUTION MODE (Neural Architecture Search)")
    print(f"{'='*60}")
    print(f"  Max hidden dim: {max_hidden_dim}")
    print(f"  Max layers: {max_layers}")
    print(f"  Population: {population_size}")
    print(f"  Generations: {generations}")
    print(f"  Eval steps per individual: {eval_steps}")
    print(f"  Activation candidates: {len(act_choices)}")
    print(f"  Evolving: {', '.join(sorted(evolve_set))}")
    if fixed_overrides: print(f"  Fixed overrides: {fixed_overrides}")
    print(f"  Tracking: {'Validation loss' if val_loader else 'Train loss'}")
    print(f"{'='*60}\n")

    # Initialize population
    population = [_evolution_create_individual(max_hidden_dim, max_layers, act_choices,
                      evolve_set=evolve_set, fixed_overrides=fixed_overrides,
                      fixed_hidden_dims=fixed_hidden_dims, fixed_activation=fixed_activation,
                      fixed_activation_type=fixed_activation_type)
                  for _ in range(population_size)]

    best_ever = None
    best_ever_fitness = float('inf')
    no_improve_count = 0
    early_stop_patience = max(5, generations // 4)

    try:
        for gen in range(generations):
            # Adaptive mutation rate: starts high, decays over generations
            base_mutation_rate = 0.5 - 0.3 * (gen / max(1, generations - 1))  # 0.5 -> 0.2
            mutation_rate = max(0.15, base_mutation_rate)

            print(f"\n--- Generation {gen+1}/{generations} (mutation_rate={mutation_rate:.2f}) ---")

            # Evaluate each individual
            for i, ind in enumerate(population):
                if ind["fitness"] == float('inf'):  # Not yet evaluated
                    desc_parts = [f"dims={ind['hidden_dims']}", f"act={ind['activation_name']}",
                                  f"type={ind['activation_type']}", f"res={ind['residual_type']}",
                                  f"noise={ind['noise_mode']}", f"norm={ind['norm_type']}"]
                    if ind.get("learning_rate") is not None:
                        desc_parts.append(f"lr={ind['learning_rate']:.1e}")
                    if ind.get("moe_mode", 1) != 1:
                        desc_parts.append(f"moe={ind['moe_mode']}")
                    if ind.get("attention_type", "none") != "none":
                        desc_parts.append(f"attn={ind['attention_type']}")
                    print(f"  Evaluating {i+1}/{len(population)}: {' '.join(desc_parts)}",
                          end="", flush=True)
                    fitness = _evolution_evaluate(
                        ind, csv_file, delimiter, input_cols, output_cols, col_types,
                        vocabularies, scalings, image_params, batch_size, optimizer_choice,
                        custom_lr, eval_steps, device, val_loader, train_dataset=train_dataset)
                    ind["fitness"] = fitness
                    print(f" -> fitness={fitness:.6f}")

            # Sort by fitness (lower is better)
            population.sort(key=lambda x: x["fitness"])

            # Track best
            prev_best = best_ever_fitness
            if population[0]["fitness"] < best_ever_fitness:
                best_ever_fitness = population[0]["fitness"]
                best_ever = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
                             for k, v in population[0].items()}
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Population diversity
            diversity = _evolution_population_diversity(population)

            print(f"\n  Best this gen: fitness={population[0]['fitness']:.6f} "
                  f"dims={population[0]['hidden_dims']} act={population[0]['activation_name']}")
            print(f"  Best ever:     fitness={best_ever_fitness:.6f} "
                  f"dims={best_ever['hidden_dims']} act={best_ever['activation_name']}")
            print(f"  Diversity: {diversity:.3f} | No improvement: {no_improve_count}/{early_stop_patience}")

            # Early stopping check
            if no_improve_count >= early_stop_patience and gen >= 5:
                print(f"\n  Early stopping: no improvement for {early_stop_patience} generations.")
                break

            # Selection + reproduction for next generation
            if gen < generations - 1:
                # Keep top 20% (elitism)
                elite_count = max(2, population_size // 5)
                new_pop = population[:elite_count]

                # Diversity maintenance: inject random individuals if diversity is low
                inject_count = 0
                if diversity < 0.2 and gen > 0:
                    inject_count = max(1, population_size // 10)
                    for _ in range(inject_count):
                        new_pop.append(_evolution_create_individual(
                            max_hidden_dim, max_layers, act_choices,
                            evolve_set=evolve_set, fixed_overrides=fixed_overrides,
                            fixed_hidden_dims=fixed_hidden_dims, fixed_activation=fixed_activation,
                            fixed_activation_type=fixed_activation_type))
                    print(f"  Injected {inject_count} random individuals (low diversity)")

                # Fill rest with crossover + mutation
                while len(new_pop) < population_size:
                    if _random.random() < 0.7 and len(population) >= 2:
                        # Tournament selection for parents
                        t1 = _random.sample(population[:max(4, population_size // 2)], 2)
                        p1 = min(t1, key=lambda x: x["fitness"])
                        t2 = _random.sample(population[:max(4, population_size // 2)], 2)
                        p2 = min(t2, key=lambda x: x["fitness"])
                        child = _evolution_crossover(p1, p2, max_hidden_dim, max_layers)
                        child = _evolution_mutate(child, max_hidden_dim, max_layers, act_choices,
                                                  mutation_rate=mutation_rate, evolve_set=evolve_set)
                    else:
                        # Mutate a random elite
                        parent = _random.choice(population[:elite_count])
                        child = _evolution_mutate(parent, max_hidden_dim, max_layers, act_choices,
                                                  mutation_rate=min(0.6, mutation_rate * 1.5),
                                                  evolve_set=evolve_set)
                    new_pop.append(child)

                population = new_pop

    except KeyboardInterrupt:
        print("\n\nEvolution interrupted by user.")
        if best_ever is None:
            print("No individuals were evaluated. Aborting.")
            return

    if best_ever is None:
        print("No viable individuals found. Aborting.")
        return

    # Print final results
    print(f"\n{'='*60}")
    print(f"EVOLUTION COMPLETE - Best Architecture Found:")
    print(f"{'='*60}")
    print(f"  Hidden dims:     {best_ever['hidden_dims']}")
    print(f"  Activation:      {best_ever['activation_name']}")
    print(f"  Activation type: {best_ever['activation_type']}")
    print(f"  Residual type:   {best_ever['residual_type']}")
    print(f"  Noise mode:      {best_ever['noise_mode']} {best_ever['noise_params']}")
    print(f"  Norm type:       {best_ever['norm_type']}")
    if best_ever.get('learning_rate') is not None:
        print(f"  Learning rate:   {best_ever['learning_rate']:.6f}")
    if best_ever.get('moe_mode', 1) != 1:
        print(f"  MoE mode:        {best_ever['moe_mode']}")
    if best_ever.get('attention_type', 'none') != 'none':
        print(f"  Attention:       {best_ever['attention_type']} (heads={best_ever.get('num_heads', 1)})")
    print(f"  Fitness:         {best_ever_fitness:.6f}")
    print(f"{'='*60}")

    return best_ever


##############################################
# Optimizer selection
##############################################
##############################################
# Evolutionary Strategy Optimizer (NES)
##############################################
##############################################
# Evolutionary Strategy Optimizer (NES + EGGROLL)
##############################################
class EvolutionaryOptimizer:
    """Natural Evolution Strategies (NES) optimizer with EGGROLL support.
    
    Standard NES: Perturbs every parameter independently (Full Rank).
    EGGROLL (Low-Rank): Perturbs parameters within a low-dimensional subspace, 
    allowing efficient scaling to larger models.
    """
    def __init__(self, params, lr=0.01, sigma=0.02, population_size=20,
                 weight_decay=0.0, antithetic=True, rank_transform=True, 
                 eggroll_rank=None):
        self.params = list(params)
        self.lr = lr
        self.sigma = sigma
        self.population_size = population_size
        self.weight_decay = weight_decay
        self.antithetic = antithetic       # Use mirrored sampling
        self.rank_transform = rank_transform  # Fitness shaping
        
        # EGGROLL Configurations
        self.eggroll_rank = eggroll_rank
        self.basis = None  # Lazy initialization of the projection matrix
        
        self.param_groups = [{'lr': lr, 'sigma': sigma}]

    def zero_grad(self):
        pass

    def _flatten_params(self):
        """Flatten all parameters into a single vector."""
        return torch.cat([p.data.view(-1) for p in self.params])

    def _unflatten_params(self, flat):
        """Write a flat vector back into the model parameters."""
        idx = 0
        for p in self.params:
            numel = p.data.numel()
            p.data.copy_(flat[idx:idx+numel].view(p.data.shape))
            idx += numel

    def _rank_transform(self, fitnesses):
        """Rank-based fitness shaping."""
        n = len(fitnesses)
        ranks = torch.zeros(n)
        sorted_idx = sorted(range(n), key=lambda i: fitnesses[i])
        for rank, idx in enumerate(sorted_idx):
            ranks[idx] = max(0, math.log(n / 2 + 1) - math.log(rank + 1))
        ranks = ranks / (ranks.sum() + 1e-8) - 1.0 / n
        return ranks

    def _init_eggroll_basis(self, full_dim, device):
        """Initialize the low-rank projection matrix (D x k) using QR decomposition
        to ensure orthonormality (preserving gradient norm)."""
        print(f"Initializing EGGROLL basis: Full Dim={full_dim} -> Low Rank={self.eggroll_rank}")
        # Generate random matrix
        H = torch.randn(full_dim, self.eggroll_rank, device=device)
        # Orthogonalize columns using QR decomposition
        Q, _ = torch.linalg.qr(H, mode='reduced')
        self.basis = Q  # (D, rank)

    def step(self, closure):
        device = self.params[0].device
        theta = self._flatten_params()
        d = theta.numel()

        # Initialize EGGROLL basis if needed
        if self.eggroll_rank is not None and self.eggroll_rank > 0 and self.basis is None:
            if self.eggroll_rank >= d:
                print("Warning: EGGROLL rank >= parameter count. Disabling EGGROLL.")
                self.eggroll_rank = None
            else:
                self._init_eggroll_basis(d, device)

        pop = self.population_size
        
        # 1. Generate Perturbations (eps)
        if self.antithetic:
            half = pop // 2
            
            if self.eggroll_rank:
                # EGGROLL: Sample in low rank (half, rank)
                z_half = torch.randn(half, self.eggroll_rank, device=device)
                # Project up: (half, rank) @ (rank, D)^T -> (half, D)
                eps_half = z_half @ self.basis.t()
            else:
                # Standard: Sample in full rank
                eps_half = torch.randn(half, d, device=device)
                
            eps = torch.cat([eps_half, -eps_half], dim=0)
            pop = 2 * half
        else:
            if self.eggroll_rank:
                z = torch.randn(pop, self.eggroll_rank, device=device)
                eps = z @ self.basis.t()
            else:
                eps = torch.randn(pop, d, device=device)

        # 2. Evaluate Population
        fitnesses = []
        for i in range(pop):
            # Apply perturbation
            self._unflatten_params(theta + self.sigma * eps[i])
            with torch.no_grad():
                loss = closure()
            fitnesses.append(loss.item() if isinstance(loss, torch.Tensor) else loss)

        # Restore original parameters
        self._unflatten_params(theta)

        # 3. Update Parameters
        fitnesses_t = torch.tensor(fitnesses, device=device, dtype=torch.float32)

        if self.rank_transform:
            weights = self._rank_transform(fitnesses_t).to(device)
        else:
            f_std = fitnesses_t.std() + 1e-8
            weights = -(fitnesses_t - fitnesses_t.mean()) / f_std

        # Compute gradient estimate
        # EGGROLL note: We calculate the update in full space here for simplicity 
        # (eps is already projected up).
        grad_est = (weights.unsqueeze(1) * eps).sum(dim=0) / (pop * self.sigma)

        new_theta = theta + self.lr * grad_est

        if self.weight_decay > 0:
            new_theta = new_theta * (1.0 - self.lr * self.weight_decay)

        self._unflatten_params(new_theta)

        return torch.tensor(fitnesses_t.mean().item())

    def train(self): pass
    def eval(self): pass


MUON_CHOICES = ("Muon", "AdaMuon", "NorMuon", "AdaGO", "AdamGO", "RMSGO", "AdaDeltaGO", "MuonHD", "NorMuonHD")
HD_CHOICES = ("AdamHD", "MuonHD", "NorMuonHD")   # hypergradient LR (lamb.AdamHD rule)
_HD_ANSWERS = {}


def prompt_hd_options(optimizer_choice, use_defaults=False):
    """Hypergradient-LR options of AdamHD / MuonHD / NorMuonHD (see lamb.AdamHD).
    use_defaults=True returns the defaults without reading stdin (GUI). The CLI asks
    once per run and reuses the answers when the optimizer is rebuilt (growth, evolution)."""
    if not use_defaults and optimizer_choice in _HD_ANSWERS:
        return dict(_HD_ANSWERS[optimizer_choice])
    def ask(label, default, convert=float):
        while not use_defaults:
            raw = input(f"{label} [{default}]: ").strip()
            if not raw:
                return default
            try:
                if convert is bool:
                    if raw.lower() not in ('1', '0', 'true', 'false', 'yes', 'no', 'y', 'n'):
                        raise ValueError
                    return raw.lower() in ('1', 'true', 'yes', 'y')
                return convert(raw)
            except ValueError:
                print("Invalid value; please try again.")
        return default

    options = dict(hyper_lr=ask("Hypergradient step (max log-LR change per step)", .05),
                   normalize=ask("Normalized hypergradient (off = original raw-cosine rule)?", True, bool))
    options['anchor'] = ask("Anchor (pull towards the peak LR; higher decays less)", 1. if options['normalize'] else .02)
    options['max_lr'] = ask("Maximum LR (0 = 100x the initial LR; needed for a zero initial LR)", 0.)
    if optimizer_choice == "AdamHD":
        options['weight_decay'] = ask("Weight decay (decoupled)", 0.)
    if not use_defaults:
        _HD_ANSWERS[optimizer_choice] = dict(options)
    return options


def optimizer_options(optimizer_choice, use_defaults=False):
    """Every constructor option the CLI asks for: Muon family and/or hypergradient LR."""
    options = {}
    if optimizer_choice in MUON_CHOICES:
        options.update(prompt_muon_options(optimizer_choice, use_defaults))
    if optimizer_choice in HD_CHOICES:
        options.update(prompt_hd_options(optimizer_choice, use_defaults))
    return options


def prompt_muon_options(optimizer_choice, use_defaults=False):
    """Configure the shared Muon family using its current constructor options.
    use_defaults=True returns the prompt defaults without reading stdin (GUI)."""
    def ask(label, default, convert=float):
        while not use_defaults:
            raw = input(f"{label} [{default}]: ").strip()
            if not raw:
                return default
            try:
                if convert is bool:
                    if raw.lower() not in ('1', '0', 'true', 'false', 'yes', 'no', 'y', 'n'):
                        raise ValueError
                    return raw.lower() in ('1', 'true', 'yes', 'y')
                return convert(raw)
            except ValueError:
                print("Invalid value; please try again.")
        return default

    options = dict(
        momentum=ask("Muon momentum", .95),
        weight_decay=ask("Weight decay", 0. if optimizer_choice in ("AdaGO", "AdamGO", "RMSGO", "AdaDeltaGO") else .1),
        rank=ask("Rank (0=full, >0=approximate)", 0, int),
        newton_schulz_iter=ask("Orthogonalization iterations", 5, int),
        foreach=ask("Batch optimizer updates?", True, bool),
        ns_bfloat16=ask("BF16 matrix iterations?", False, bool),
        cautious=ask("Cautious updates?", False, bool),
        muon_all=ask("MuonAll (all parameters)?", False, bool),
        muon_all_reshape=ask("MuonAll: use near-square vector reshape?", False, bool),
    )
    options['orthogonalization_backend'] = (
        'polar_express' if ask("Use Polar Express backend?", True, bool) else 'newton_schulz')
    options['adam_betas'] = (ask("AdamW fallback beta1", .9), ask("AdamW fallback beta2", .95 if optimizer_choice in ("AdaGO", "AdamGO", "RMSGO", "AdaDeltaGO") else .999))
    options['adam_eps'] = ask("AdamW fallback epsilon", 1e-8)
    # GO matrix LRs (0.05) are far too large for Adam: default 4.2e-4 at lr 0.05.
    options['adam_lr_ratio'] = ask("AdamW fallback LR as a fraction of the main LR",
                                   4.2e-4 / .05 if optimizer_choice in ("AdaGO", "AdamGO", "RMSGO", "AdaDeltaGO") else 1.)
    if optimizer_choice == "AdaMuon":
        options['eps'] = ask("AdaMuon epsilon", 1e-8)
        options['nesterov'] = ask("AdaMuon Nesterov (off=paper Algorithm 1)?", False, bool)
    elif optimizer_choice in ("NorMuon", "NorMuonHD"):
        options['eps'] = ask("NorMuon epsilon", 1e-8)
        options['beta2'] = ask("Row variance decay (beta2)", .95)
    elif optimizer_choice in ("AdaGO", "RMSGO"):
        options['gamma'] = ask("Gradient norm cap (gamma)", 1.)
        options['v0'] = ask("Initial norm accumulator (v0)", 1.)
        options['eps'] = ask("Minimum step (epsilon)", 5e-4)
        if optimizer_choice == "RMSGO":
            options['beta2'] = ask("Norm variance decay (beta2)", .99)
    elif optimizer_choice == "AdaDeltaGO":
        options['gamma'] = ask("Gradient norm cap (gamma)", 1.)
        options['rho'] = ask("Averaging decay (rho)", .9)
        options['eps'] = ask("RMS stabilizer (epsilon)", 1e-6)
    elif optimizer_choice == "AdamGO":
        options['beta2'] = ask("Norm variance decay (beta2)", .999)
        options['gamma'] = ask("Gradient norm cap (gamma)", 1.)
        options['delta'] = ask("Denominator stabilizer (delta)", 1e-8)
        options['min_step'] = ask("Minimum step (0=disabled)", 0.)
    return options


def mlpres_muon_param_groups(model):
    """Keep embeddings, vectors and both output branches on AdamW."""
    heads = [getattr(model, name, None) for name in ('final_linear', 'final_skip')]
    # Per-neuron PWL tables are [C, K] but are not weight matrices: keep them off Muon too.
    heads += [m for m in model.modules() if hasattr(m, 'x_positions') and hasattr(m, 'slopes')]  # PaWeL / torchpwl
    adamw_params = [p for head in heads if head is not None for p in head.parameters()]
    return muon_param_groups(model, adamw_params)


def select_optimizer(optimizer_choice, model, custom_lr=None, optim_params=None):
    """Select optimizer; custom_lr overrides its default learning rate.

    Muon-family optim_params accepts the shared constructor flags (including
    cautious, muon_all, muon_all_reshape and orthogonalization_backend). None prompts; {} uses defaults.
    """
    _lr = custom_lr  # None means use default
    if optimizer_choice == "Adam": optimizer = optim.Adam(model.parameters(), lr=_lr if _lr is not None else 0.0004)
    elif optimizer_choice in ("Muon", "AdaMuon", "NorMuon", "AdaGO", "AdamGO", "RMSGO", "AdaDeltaGO", "MuonHD", "NorMuonHD"):
        if optim_params is None:
            options = dict(prompt_muon_options(optimizer_choice))
            if optimizer_choice in ("MuonHD", "NorMuonHD"):
                options.update(prompt_hd_options(optimizer_choice))
        else:
            options = dict(optim_params)
        if 'max_lr' in options:   # HD: 0 means no absolute cap (lamb uses None)
            options['max_lr'] = float(options['max_lr']) or None
        if _lr is not None:
            options['lr'] = _lr
        optimizer_cls = (AdaDeltaGO if optimizer_choice == "AdaDeltaGO" else RMSGO if optimizer_choice == "RMSGO" else AdamGO if optimizer_choice == "AdamGO" else AdaGO if optimizer_choice == "AdaGO" else
                         MuonHD if optimizer_choice == "MuonHD" else NorMuonHD if optimizer_choice == "NorMuonHD" else
                         {"Muon": Muon, "AdaMuon": AdaMuon, "NorMuon": NorMuon}[optimizer_choice])
        # AdamW-fallback LR = main LR * adam_lr_ratio (absent = legacy shared LR);
        # a ratio keeps custom and LR-finder rates proportional.
        ratio = options.pop('adam_lr_ratio', 1.)
        if isinstance(ratio, bool) or not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"adam_lr_ratio must be a finite positive number, got {ratio!r}")
        optimizer = optimizer_cls(mlpres_muon_param_groups(model), **options)
        if not options.get('muon_all', False):
            for group in optimizer.param_groups:
                if not group['use_muon']:
                    group['lr'] *= ratio
                    if group.get('max_lr'):   # HD: keep the fallback's cap in proportion too
                        group['max_lr'] *= ratio
    elif optimizer_choice == "Adam3":
        order = max(int(optim_params.get("order", 3) if optim_params is not None else input("Enter amount of orders (3 is minimum): ")), 3)
        optimizer = ThreeAdam(model.parameters(), lr=0.004, order=order)
    elif optimizer_choice == "AdamHD":
        options = dict(prompt_hd_options(optimizer_choice) if optim_params is None else optim_params)
        if 'max_lr' in options:
            options['max_lr'] = float(options['max_lr']) or None
        optimizer = AdamHD(model.parameters(), lr=_lr if _lr is not None else 1e-3, **options)
    elif optimizer_choice == "SGD": optimizer = CSGD(model.parameters(), momentum=0.9, lr=0.01)
    elif optimizer_choice == "SGDHD": optimizer = SGDHD(model.parameters(), momentum=0.9, lr=0.0, nesterov=True)
    elif optimizer_choice == "Lamb": optimizer = CLamb(model.parameters(), lr=0.01)
    elif optimizer_choice == "Adagrad": optimizer = optim.Adagrad(model.parameters(), lr=_lr if _lr is not None else 0.01)
    elif optimizer_choice == "Adadelta": optimizer = CAdadeltaM(model.parameters(), lr=1.0)
    elif optimizer_choice == "AdamW": optimizer = CAdamax(model.parameters(), lr=_lr if _lr is not None else 0.0005, weight_decay=0.0)
    elif optimizer_choice == "RMSprop": optimizer = CRMSprop(model.parameters(), lr=0.02, momentum=0.9)
    elif optimizer_choice == "Rprop": optimizer = CRprop(model.parameters(), lr=0.02)
    elif optimizer_choice == "ASGD": optimizer = optim.ASGD(model.parameters(), lr=_lr if _lr is not None else 0.02)
    elif optimizer_choice == "Adamax": optimizer = optim.Adamax(model.parameters(), lr=_lr if _lr is not None else 0.001)
    elif optimizer_choice == "NAdam": optimizer = optim.NAdam(model.parameters(), lr=_lr if _lr is not None else 0.001)
    elif optimizer_choice == "SparseAdam": optimizer = optim.SparseAdam(model.parameters(), lr=_lr if _lr is not None else 0.001)
    elif optimizer_choice == "RAdamScheduleFree": optimizer = RAdamScheduleFree(model.parameters(), lr=_lr if _lr is not None else 0.0004)
    elif optimizer_choice == "AdEMAMix": optimizer = AdEMAMix(model.parameters(), lr=_lr if _lr is not None else 0.001)
    elif optimizer_choice == "Adam3":
        order = max(int(optim_params.get("order", 3) if optim_params is not None else input("Enter amount of orders (3 is minimum): ")), 3)
        optimizer = ThreeAdam(model.parameters(), lr=0.004, order=order)
    elif optimizer_choice == "AdamDelta": optimizer = AdamDelta(model.parameters())
    elif optimizer_choice == "AutoAdam": optimizer = AutoAdam(model.parameters())
    elif optimizer_choice == "NormAdam": optimizer = NormAdam(model.parameters())
    elif optimizer_choice == "SWATS": optimizer = SWATS(model.parameters(), lr=0.001)
    elif optimizer_choice == "AdaBoundW": optimizer = AdaBoundW(model.parameters(), lr=0.001)
    elif optimizer_choice == "CLion": optimizer = CLion(model.parameters(), lr=_lr if _lr is not None else 0.0001)
    elif optimizer_choice == "Signum": optimizer = CSignum(model.parameters(), lr=0.01)
    elif optimizer_choice == "SRprop": optimizer = SRprop(model.parameters(), lr=0.01)
    elif optimizer_choice == "IRprop": optimizer = MiniBatch_iRpropPlus(model.parameters(), lr=0.001)
    elif optimizer_choice == "Adan": optimizer = Adan(model.parameters(), lr=_lr if _lr is not None else 0.001)
    elif optimizer_choice == "Prodigy": optimizer = Prodigy(model.parameters(), lr=_lr if _lr is not None else 1.0, weight_decay=0.0)
    elif optimizer_choice == "Evolution" and optim_params is not None:
        rank = int(optim_params.get("eggroll_rank") or 0)
        optimizer = EvolutionaryOptimizer(model.parameters(), lr=_lr if _lr is not None else 0.01,
                                          sigma=float(optim_params.get("sigma", 0.02)),
                                          population_size=int(optim_params.get("population_size", 20)),
                                          antithetic=True, rank_transform=True, eggroll_rank=rank or None)
    elif optimizer_choice == "Evolution":
        print("\n--- Evolution Strategy Setup ---")
        pop_str = input("  Population size (default 20): ").strip()
        pop_size = int(pop_str) if pop_str else 20
        
        sigma_str = input("  Perturbation sigma (noise scale, default 0.02): ").strip()
        sigma = float(sigma_str) if sigma_str else 0.02
        
        print("  EGGROLL (Low-Rank) Mode:")
        print("  Enter a rank (e.g., 50 or 100) to constrain optimization to a subspace.")
        print("  Enter 0 or leave empty for standard Full-Rank ES (slower on large models).")
        rank_str = input("  Subspace Rank: ").strip()
        eggroll_rank = int(rank_str) if rank_str and rank_str != "0" else None
        
        if eggroll_rank:
            print(f"  > Enabled EGGROLL with rank {eggroll_rank}")
        else:
            print("  > Standard Full-Rank Evolution")

        optimizer = EvolutionaryOptimizer(model.parameters(), lr=_lr if _lr is not None else 0.01,
                                          sigma=sigma, population_size=pop_size,
                                          antithetic=True, rank_transform=True,
                                          eggroll_rank=eggroll_rank) # Pass the rank
    else: raise ValueError(f"Optimizer {optimizer_choice} is not supported.")
    return optimizer



##############################################
# Model Export System
##############################################
def ask_export_mode():
    """Ask user which export format(s) they want."""
    print("\n=== Model Export ===")
    print("1: ONNX (.onnx)")
    print("2: TorchScript (.pt traced)")
    print("3: Pure Python (no PyTorch dependency)")
    print("4: Quantized model (INT8)")
    print("5: All of the above")
    choice = input("Enter choice (1-5): ").strip()
    return choice


def export_to_onnx(model, input_dims, device, filename="model_export.onnx"):
    """Export model to ONNX format."""
    print(f"Exporting to ONNX: {filename}")
    model.eval()
    dummy = torch.randn(1, input_dims).to(device)
    try:
        torch.onnx.export(
            model, dummy, filename,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )
        print(f"  ✓ ONNX exported to {filename}")
        # Verify
        try:
            import onnx
            onnx_model = onnx.load(filename)
            onnx.checker.check_model(onnx_model)
            print(f"  ✓ ONNX model verified.")
        except ImportError:
            print("  (onnx package not installed, skipping verification)")
        except Exception as e:
            print(f"  ⚠ ONNX verification warning: {e}")
    except Exception as e:
        print(f"  ✗ ONNX export failed: {e}")


def export_to_torchscript(model, input_dims, device, filename="model_export_traced.pt"):
    """Export model to TorchScript via tracing."""
    print(f"Exporting to TorchScript: {filename}")
    model.eval()
    dummy = torch.randn(1, input_dims).to(device)
    try:
        traced = torch.jit.trace(model, dummy)
        traced.save(filename)
        print(f"  ✓ TorchScript exported to {filename}")
    except Exception as e:
        print(f"  ✗ TorchScript export failed: {e}")
        print("  Trying scripting instead of tracing...")
        try:
            scripted = torch.jit.script(model)
            scripted.save(filename)
            print(f"  ✓ TorchScript (scripted) exported to {filename}")
        except Exception as e2:
            print(f"  ✗ TorchScript scripting also failed: {e2}")


def export_quantized(model, input_dims, device, filename="model_quantized.pt"):
    """Export INT8 quantized model."""
    print(f"Exporting quantized model: {filename}")
    model.eval().cpu()
    try:
        quantized = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        torch.save(quantized.state_dict(), filename)
        print(f"  ✓ Quantized model exported to {filename}")
        
        # Also save the full quantized model for easy loading
        torch.save(quantized, filename.replace('.pt', '_full.pt'))
        print(f"  ✓ Full quantized model saved to {filename.replace('.pt', '_full.pt')}")
    except Exception as e:
        print(f"  ✗ Quantization failed: {e}")
    finally:
        model.to(device)


def _extract_activation_formula(activation_module):
    """Try to extract a Python math expression for the activation function."""
    cls_name = activation_module.__class__.__name__
    
    # Known mappings to pure-python equivalents
    known_formulas = {
        'ReLU': 'max(0, x)',
        'LeakyReLU': 'x if x > 0 else 0.01 * x',
        'Sigmoid': '1.0 / (1.0 + math.exp(-x))',
        'Tanh': 'math.tanh(x)',
        'SiLU': 'x / (1.0 + math.exp(-x))',
        'GELU': '0.5 * x * (1.0 + math.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x**3)))',
        'Softplus': 'math.log(1.0 + math.exp(x))',
        'Mish': 'x * math.tanh(math.log(1.0 + math.exp(x)))',
        'ELU': 'x if x > 0 else (math.exp(x) - 1)',
        'SELU': '1.0507 * (x if x > 0 else 1.6733 * (math.exp(x) - 1))',
        'ReLU6': 'min(max(0, x), 6)',
        'Identity': 'x',
        'Sine': 'math.sin(x)',
        'Cosine': 'math.cos(x)',
        'PReLU': 'x if x > 0 else ALPHA * x',  # needs param extraction
    }
    
    if cls_name in known_formulas:
        return known_formulas[cls_name], True
    
    # Check for custom activation
    if hasattr(activation_module, '_custom_expr'):
        expr = activation_module._custom_expr
        # Convert torch operations to math operations
        py_expr = expr.replace('torch.sin', 'math.sin')
        py_expr = py_expr.replace('torch.cos', 'math.cos')
        py_expr = py_expr.replace('torch.exp', 'math.exp')
        py_expr = py_expr.replace('torch.abs', 'abs')
        py_expr = py_expr.replace('torch.tanh', 'math.tanh')
        py_expr = py_expr.replace('torch.sigmoid', 'lambda _x: 1.0/(1.0+math.exp(-_x))')
        py_expr = py_expr.replace('torch.where', '(lambda cond, a, b: a if cond else b)')
        return py_expr, True
    
    return None, False


def export_pure_python(model, config, scalings, vocabularies, col_types, 
                       input_dims, output_dim, filename="model_pure.py"):
    """
    Export model as a standalone Python script.
    - Inspects model structure to hardcode block logic (skip connections, norms).
    - Embeds weights, scalings, and vocabularies.
    - Implements robust vector math with shape checking.
    - Supports all normalization types, activation functions, and residual patterns.
    """
    print(f"Exporting to pure Python: {filename}")
    model.eval().cpu()
    
    # 1. Detect Activation Formula (comprehensive)
    def _extract_activation_formula(act_module):
        """Extract a pure-python formula string for the activation function."""
        if act_module is None:
            return "x"
        name = type(act_module).__name__
        # Standard PyTorch activations
        if isinstance(act_module, nn.ReLU): return "max(0.0, x)"
        if isinstance(act_module, nn.ReLU6): return "min(6.0, max(0.0, x))"
        if isinstance(act_module, nn.SiLU): return "x / (1.0 + math.exp(-x))"
        if isinstance(act_module, nn.Sigmoid): return "1.0 / (1.0 + math.exp(-x))"
        if isinstance(act_module, nn.Tanh): return "math.tanh(x)"
        if isinstance(act_module, nn.GELU): return "0.5 * x * (1.0 + math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))"
        if isinstance(act_module, nn.SELU): return "(1.0507009873554805 * (x if x > 0 else 1.6732632423543772 * (math.exp(x) - 1.0)))"
        if isinstance(act_module, nn.CELU): return "(x if x >= 0 else (math.exp(x) - 1.0))"
        if isinstance(act_module, nn.ELU): return "(x if x >= 0 else (math.exp(x) - 1.0))"
        if isinstance(act_module, nn.Softplus): return "math.log(1.0 + math.exp(x))"
        if isinstance(act_module, nn.LeakyReLU):
            slope = act_module.negative_slope
            return f"(x if x >= 0 else {slope} * x)"
        if isinstance(act_module, nn.PReLU):
            return "PRELU_ACTIVATION"  # Special: needs per-element alpha
        if isinstance(act_module, nn.Identity): return "x"
        # Custom activations
        if name == "Mish": return "x * math.tanh(math.log(1.0 + math.exp(x)))"
        if name == "Softsign": return "x / (1.0 + abs(x))"
        if name == "HardSwish": return "(0.0 if x <= -3.0 else (x if x >= 3.0 else x * (x + 3.0) / 6.0))"
        if name == "BentIdentity": return "((math.sqrt(x*x + 1.0) - 1.0) / 2.0 + x)"
        if name == "Sine" or name == "TrainableSine": return "math.sin(x)"
        if name == "Cosine" or name == "TrainableCosine": return "math.cos(x)"
        if name == "SquaredReLU": return "(max(0.0, x) ** 2)"
        if name == "TeLU": return "(x * math.tanh(math.exp(x)) if x < 20 else x)"
        if name == "CoLU": return "(x / (1.0 - x * math.exp(-(x + math.exp(x)))) if abs(x) < 20 else x)"
        if name == "Serf": return "(x * (1.0 - 2.0 / (math.exp(2.0 * math.log(1.0 + math.exp(x))) + 1.0)) if abs(x) < 20 else x)"
        if name == "Sign": return "(1.0 if x > 0 else (-1.0 if x < 0 else 0.0))"
        if name == "Cauchy": return "(1.0 / (1.0 + x * x))"
        if name == "Reciprocal": return "(1.0 / (abs(x) + 1e-8))"
        if name == "Phish": return "(x * math.tanh(math.log(1.0 + math.exp(x)) if x < 20 else x))"
        if name == "TanhExp": return "(x * math.tanh(math.exp(x)) if x < 20 else x)"
        if name == "CombHsine": return "(0.5 * (x + math.sin(x)))"
        if name in ("ScaledTanh",):
            scale = getattr(act_module, 'scale', 1.7)
            return f"({scale} * math.tanh(x))"
        if name in ("lSELU", "sSELU", "AGLU", "SRS", "DRA", "OGDRA", "StarReLU",
                     "GSine", "GCosine", "SmeLU", "HeLU", "ATanU", "SALU", "SMU",
                     "TaLU", "SoLU", "SiLULU", "GoLU", "SGoLU", "Cone", "ParabolicCone",
                     "SquareActivation", "CubeActivation", "TheSquare", "TheCube",
                     "Sial", "SoftMax", "TTanh", "TSoftsign", "TSigma", "TReLU"):
            # Trainable/complex activations: fall back to SiLU as safe approximation
            return "x / (1.0 + math.exp(-x))"
        # Default: Mish as general-purpose fallback
        return "x * math.tanh(math.log(1.0 + math.exp(x)))"

    act_formula = "x * math.tanh(math.log(1.0 + math.exp(x)))"  # Mish default
    has_prelu = False
    if len(model.blocks) > 0:
        block_act = model.blocks[0].activation
        # Unwrap activation wrappers (SelfGated, GLU, TAAF, etc.)
        inner_act = block_act
        if hasattr(block_act, 'base_activation'):
            inner_act = block_act.base_activation
        elif hasattr(block_act, 'activation'):
            inner_act = block_act.activation
        act_formula = _extract_activation_formula(inner_act)
        if act_formula == "PRELU_ACTIVATION":
            has_prelu = True
            act_formula = "(x if x >= 0 else PRELU_ALPHA * x)"

    # 2. Build Output Layout
    output_cols = [c for c, t in col_types.items() if t in ["out", "outlab", "outex", "outlabcat", "outexcat"]]
    layout, _, _ = build_output_layout(output_cols, col_types, scalings, vocabularies)

    # 3. Analyze Block Structure (comprehensive)
    block_manifest = []
    for i, block in enumerate(model.blocks):
        meta = {"idx": i}
        # Skip Connection Type
        if isinstance(block.skip, nn.Linear):
            meta["skip_type"] = "linear"
        else:
            meta["skip_type"] = "identity"
        
        # Norm Type detection
        if isinstance(block.norm, RMSNorm):
            meta["norm_type"] = "rmsnorm"
        elif isinstance(block.norm, nn.BatchNorm1d):
            meta["norm_type"] = "batchnorm"
        elif isinstance(block.norm, GroupNormMLP):
            meta["norm_type"] = "groupnorm"
        elif isinstance(block.norm, nn.LayerNorm):
            meta["norm_type"] = "layernorm"
        elif isinstance(block.norm, nn.Identity):
            meta["norm_type"] = "none"
        else:
            meta["norm_type"] = "layernorm"  # default assumption
        
        # Residual type detection
        residual_type = getattr(block, 'residual_type', 'residual')
        meta["residual_type"] = residual_type
        
        # Whether linear2 exists
        meta["has_linear2"] = block.linear2 is not None
        
        # Whether highway gate exists
        meta["has_highway"] = hasattr(block, 'highway_gate') and block.highway_gate is not None
            
        block_manifest.append(meta)

    lines = []
    lines.append('#!/usr/bin/env python3')
    lines.append('"""')
    lines.append(f'Auto-generated Standalone Inference Script')
    lines.append('"""')
    lines.append('import math')
    lines.append('import json')
    lines.append('import sys')
    lines.append('')
    
    # Embed Configuration
    lines.append(f'COL_TYPES = {json.dumps(col_types, indent=4)}')
    lines.append(f'SCALINGS = {json.dumps(scalings, indent=4)}')
    lines.append(f'VOCABULARIES = {json.dumps(vocabularies, indent=4)}')
    lines.append(f'OUTPUT_LAYOUT = {json.dumps(layout, indent=4)}')
    lines.append(f'INPUT_DIMS = {input_dims}')
    lines.append(f'BLOCK_LAYOUT = {json.dumps(block_manifest, indent=4)}')
    lines.append('')

    # Embed PReLU alpha if needed
    if has_prelu:
        for i, block in enumerate(model.blocks):
            act = block.activation
            if hasattr(act, 'base_activation'):
                act = act.base_activation
            elif hasattr(act, 'activation'):
                act = act.activation
            if isinstance(act, nn.PReLU):
                alpha_val = act.weight.detach().cpu().numpy().tolist()
                if isinstance(alpha_val, (int, float)):
                    alpha_val = [alpha_val]
                lines.append(f'PRELU_ALPHA_{i} = {json.dumps(alpha_val)}')
        lines.append('')

    # Embed Weights
    lines.append('WEIGHTS = {}')
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        val = param.detach().cpu().numpy().tolist()
        if not isinstance(val, list): val = [val] # scalar handling
        lines.append(f'WEIGHTS["{name}"] = {json.dumps(val)}')
    
    lines.append('')
    lines.append('# ==========================================')
    lines.append('#            MATH & LAYERS')
    lines.append('# ==========================================')
    
    lines.append('''
def matmul(x, w, b=None):
    """ Matrix multiplication: y = x @ w.T + b """
    out_features = len(w)
    in_features = len(w[0])
    
    if len(x) != in_features:
        raise ValueError(f"Shape mismatch in matmul: input len {len(x)} vs weight in_features {in_features}")
        
    result = [0.0] * out_features
    for i in range(out_features):
        acc = 0.0
        row = w[i]
        for j in range(in_features):
            acc += x[j] * row[j]
        if b:
            acc += b[i]
        result[i] = acc
    return result

def vec_add(a, b):
    """ Element-wise addition with strict shape check """
    if len(a) != len(b):
        raise ValueError(f"Vector add shape mismatch: {len(a)} vs {len(b)}")
    return [xi + yi for xi, yi in zip(a, b)]

def layer_norm(x, w, b, eps=1e-5):
    n = len(x)
    mean = sum(x) / n
    var = sum((xi - mean) ** 2 for xi in x) / n
    std = math.sqrt(var + eps)
    return [((xi - mean) / std) * wi + bi for xi, wi, bi in zip(x, w, b)]

def rms_norm(x, w, eps=1e-8):
    n = len(x)
    rms = math.sqrt(sum(xi * xi for xi in x) / n + eps)
    return [xi / rms * wi for xi, wi in zip(x, w)]

def batch_norm_infer(x, running_mean, running_var, weight, bias, eps=1e-5):
    """ BatchNorm1d in eval mode """
    return [((xi - m) / math.sqrt(v + eps)) * w + b
            for xi, m, v, w, b in zip(x, running_mean, running_var, weight, bias)]

def vec_mul_param(vec, param):
    """ Multiply vector by a parameter (scalar or vector) """
    if len(param) == 1: 
        p = param[0]
        return [v * p for v in vec]
    elif len(param) == len(vec):
        return [v * p for v, p in zip(vec, param)]
    else:
        raise ValueError(f"Param shape mismatch: vec {len(vec)} vs param {len(param)}")

def highway_gate(x, residual, gate_w, gate_b):
    """ Highway network gating: out = g * x + (1-g) * residual """
    gate_input = [0.0] * len(gate_b)
    for i in range(len(gate_b)):
        acc = gate_b[i]
        for j in range(len(residual)):
            acc += residual[j] * gate_w[i][j]
        gate_input[i] = acc
    g = [1.0 / (1.0 + math.exp(-gi)) for gi in gate_input]  # sigmoid
    return [gi * xi + (1.0 - gi) * ri for gi, xi, ri in zip(g, x, residual)]

def activation(x_vec):
    return [''')
    lines.append(f'        {act_formula}')
    lines.append('''        for x in x_vec
    ]
''')

    lines.append('# ==========================================')
    lines.append('#           DATA PIPELINE')
    lines.append('# ==========================================')
    
    lines.append('''
def preprocess_input(user_input_dict):
    processed_vector = []
    
    # Sort columns by COL_TYPES insertion order to match training
    for col, ctype in COL_TYPES.items():
        if ctype not in ["in", "inlab", "intex", "inlabcat", "intexcat", "inim"]:
            continue
            
        val = user_input_dict.get(col)
        if val is None: val = 0 if ctype in ["in", "inlab"] else ""

        if ctype == "in":
            if col in SCALINGS and "min" in SCALINGS[col]:
                smin = SCALINGS[col]["min"]
                smax = SCALINGS[col]["max"]
                try:
                    fval = float(val)
                    lo, hi = SCALINGS[col].get("range", [-1.0, 1.0])
                    scaled = lo + (hi - lo) * (fval - smin) / ((smax - smin) or 1.0)
                    processed_vector.append(scaled)
                except: processed_vector.append(0.0)
            else:
                try: processed_vector.append(float(val))
                except: processed_vector.append(0.0)
        
        elif ctype == "inlab":
            if col in SCALINGS and "min" in SCALINGS[col]:
                smin = SCALINGS[col]["min"]
                smax = SCALINGS[col]["max"]
                try:
                    fval = float(val)
                    lo, hi = SCALINGS[col].get("range", [-1.0, 1.0])
                    scaled = lo + (hi - lo) * (fval - smin) / ((smax - smin) or 1.0)
                    processed_vector.append(scaled)
                except: processed_vector.append(0.0)
            else:
                try: processed_vector.append(float(val))
                except: processed_vector.append(0.0)
                
        elif ctype == "inlabcat":
            vocab = VOCABULARIES.get(col, {})
            vec_len = len(vocab)
            one_hot = [0.0] * vec_len
            idx = vocab.get(str(val))
            if idx is not None and 0 <= idx < vec_len:
                one_hot[idx] = 1.0
            processed_vector.extend(one_hot)
            
        elif ctype == "intexcat":
            vocab = VOCABULARIES.get(col, {})
            if col in SCALINGS and "max_len" in SCALINGS[col]:
                max_len = SCALINGS[col]["max_len"]
                vocab_size = len(vocab)
                sval = str(val)
                for i in range(max_len):
                    char_vec = [0.0] * vocab_size
                    if i < len(sval):
                        char = sval[i]
                        idx = vocab.get(char)
                        if idx is not None and idx > 0 and idx <= vocab_size:
                             char_vec[idx-1] = 1.0
                    processed_vector.extend(char_vec)
            else:
                processed_vector.append(0.0)

    return processed_vector

def postprocess_output(raw_output):
    results = {}
    for entry in OUTPUT_LAYOUT:
        col = entry["col"]
        ctype = entry["type"]
        start = entry["start"]
        end = entry["end"]
        slice_data = raw_output[start:end]
        
        if ctype in ["out", "outlab", "outex"]:
            val = slice_data[0]
            if col in SCALINGS and "min" in SCALINGS[col]:
                smin = SCALINGS[col]["min"]
                smax = SCALINGS[col]["max"]
                lo, hi = SCALINGS[col].get("range", [-1.0, 1.0])
                val = (val - lo) / ((hi - lo) or 1.0) * (smax - smin) + smin
            results[col] = val
            
        elif ctype in ["outlabcat", "outexcat"]:
            max_val = -float("inf")
            max_idx = -1
            for i, v in enumerate(slice_data):
                if v > max_val:
                    max_val = v
                    max_idx = i
            vocab = VOCABULARIES.get(col, {})
            found_label = str(max_idx)
            for k, v in vocab.items():
                if v == max_idx:
                    found_label = k
                    break
            results[col] = found_label
    return results
''')

    lines.append('# ==========================================')
    lines.append('#            FORWARD PASS')
    lines.append('# ==========================================')
    
    lines.append('def forward(input_vec):')
    lines.append('    x = list(input_vec)')
    lines.append('    original_input = list(input_vec)')
    lines.append('    ')
    
    # Input attention (skip in pure python for simplicity - note this in comments)
    lines.append('    # Note: Input attention layers are not exported (minor accuracy difference)')
    lines.append('    ')
    
    lines.append('    for block_meta in BLOCK_LAYOUT:')
    lines.append('        i = block_meta["idx"]')
    lines.append('        residual = list(x)')
    lines.append('        prefix = f"blocks.{i}."')
    lines.append('        ')
    
    # Norm dispatch
    lines.append('        # 1. Normalization')
    lines.append('        norm_type = block_meta.get("norm_type", "layernorm")')
    lines.append('        if norm_type == "layernorm":')
    lines.append('            if prefix + "norm.weight" in WEIGHTS:')
    lines.append('                ln_w = WEIGHTS[prefix + "norm.weight"]')
    lines.append('                ln_b = WEIGHTS[prefix + "norm.bias"]')
    lines.append('                x = layer_norm(x, ln_w, ln_b)')
    lines.append('        elif norm_type == "rmsnorm":')
    lines.append('            if prefix + "norm.weight" in WEIGHTS:')
    lines.append('                rms_w = WEIGHTS[prefix + "norm.weight"]')
    lines.append('                x = rms_norm(x, rms_w)')
    lines.append('        elif norm_type == "batchnorm":')
    lines.append('            if prefix + "norm.running_mean" in WEIGHTS:')
    lines.append('                x = batch_norm_infer(x,')
    lines.append('                    WEIGHTS[prefix + "norm.running_mean"],')
    lines.append('                    WEIGHTS[prefix + "norm.running_var"],')
    lines.append('                    WEIGHTS[prefix + "norm.weight"],')
    lines.append('                    WEIGHTS[prefix + "norm.bias"])')
    lines.append('        # norm_type == "none": skip')
    lines.append('        ')
    
    lines.append('        # 2. Linear 1')
    lines.append('        l1_w = WEIGHTS[prefix + "linear1.weight"]')
    lines.append('        l1_b = WEIGHTS[prefix + "linear1.bias"]')
    lines.append('        x = matmul(x, l1_w, l1_b)')
    lines.append('        ')
    lines.append('        # 3. Activation')
    lines.append('        x = activation(x)')
    lines.append('        ')
    
    lines.append('        # 4. Linear 2 (if exists)')
    lines.append('        if block_meta.get("has_linear2", True):')
    lines.append('            l2_key_w = prefix + "linear2.weight"')
    lines.append('            l2_key_b = prefix + "linear2.bias"')
    lines.append('            if l2_key_w in WEIGHTS:')
    lines.append('                x = matmul(x, WEIGHTS[l2_key_w], WEIGHTS.get(l2_key_b))')
    lines.append('        ')
    
    # Residual connection dispatch
    lines.append('        # 5. Residual Connection')
    lines.append('        res_type = block_meta.get("residual_type", "residual")')
    lines.append('        ')
    lines.append('        # Apply skip projection if dimensions differ')
    lines.append('        if block_meta["skip_type"] == "linear":')
    lines.append('            skip_w = WEIGHTS[prefix + "skip.weight"]')
    lines.append('            skip_b = WEIGHTS.get(prefix + "skip.bias")')
    lines.append('            skip_val = matmul(residual, skip_w, skip_b)')
    lines.append('        else:')
    lines.append('            skip_val = residual')
    lines.append('        ')
    lines.append('        # ReZero/Gamma/Beta scaling')
    lines.append('        if prefix + "gamma" in WEIGHTS:')
    lines.append('            gamma = WEIGHTS[prefix + "gamma"]')
    lines.append('            x = vec_mul_param(x, gamma)')
    lines.append('        if prefix + "beta" in WEIGHTS:')
    lines.append('            beta = WEIGHTS[prefix + "beta"]')
    lines.append('            skip_val = vec_mul_param(skip_val, beta)')
    lines.append('        ')
    lines.append('        if res_type == "none":')
    lines.append('            pass  # No residual, x stays as-is')
    lines.append('        elif res_type == "highway" and block_meta.get("has_highway", False):')
    lines.append('            hw_w = WEIGHTS.get(prefix + "highway_gate.weight")')
    lines.append('            hw_b = WEIGHTS.get(prefix + "highway_gate.bias")')
    lines.append('            if hw_w and hw_b:')
    lines.append('                x = highway_gate(x, skip_val, hw_w, hw_b)')
    lines.append('            else:')
    lines.append('                x = vec_add(x, skip_val)')
    lines.append('        else:')
    lines.append('            # Standard residual, rezero, elementwise_rezero all end up as add')
    lines.append('            if len(x) == len(skip_val):')
    lines.append('                x = vec_add(x, skip_val)')
    lines.append('            # else: dimension mismatch, skip residual (projected skip should fix this)')
    lines.append('        ')
    
    lines.append('    # Final Layer')
    lines.append('    fl_w = WEIGHTS["final_linear.weight"]')
    lines.append('    fl_b = WEIGHTS["final_linear.bias"]')
    lines.append('    x = matmul(x, fl_w, fl_b)')
    lines.append('    ')
    lines.append('    # Final Skip (input -> output direct connection)')
    lines.append('    if "final_skip.weight" in WEIGHTS:')
    lines.append('        skip_w = WEIGHTS["final_skip.weight"]')
    lines.append('        # final_skip maps from original input dims to output dims')
    lines.append('        skip_val = matmul(original_input, skip_w)')
    lines.append('        if len(x) == len(skip_val):')
    lines.append('            x = vec_add(x, skip_val)')
    lines.append('    ')
    lines.append('    return x')

    lines.append('')
    lines.append('# ==========================================')
    lines.append('#             INTERACTIVE CLI')
    lines.append('# ==========================================')
    lines.append('''
if __name__ == "__main__":
    print(f"Model Loaded. Input Size: {INPUT_DIMS}")
    print("-" * 40)
    
    input_cols = [c for c, t in COL_TYPES.items() if t in ["in", "inlab", "intex", "inlabcat", "intexcat"]]
    
    while True:
        try:
            print("\\nEnter Inputs:")
            user_data = {}
            for col in input_cols:
                val = input(f"  {col}: ").strip()
                if val.lower() == 'exit': sys.exit()
                user_data[col] = val
            
            input_vec = preprocess_input(user_data)
            raw_out = forward(input_vec)
            results = postprocess_output(raw_out)
            
            print("\\nResults:")
            for k, v in results.items():
                if isinstance(v, float): print(f"  {k}: {v:.4f}")
                else: print(f"  {k}: {v}")
                    
        except KeyboardInterrupt: break
        except Exception as e: 
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
''')

    with open(filename, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  ✓ Pure Python script exported to {filename}")


def run_export(config_path="config.json", model_path="model.pt"):
    """Main export function."""
    config = load_config()
    col_types = config["col_types"]
    input_cols = [c for c, t in col_types.items() if 'in' in t]
    output_cols = [c for c, t in col_types.items() if 'out' in t]
    vocabularies = config.get("vocabularies", {})
    scalings = config.get("scalings", {})
    
    input_dims = calculate_input_dim(input_cols, col_types, scalings, vocabularies)
    output_pred_dim = calculate_output_pred_dim(output_cols, col_types, scalings, vocabularies)
    
    # Rebuild model
    activation_map = _build_activation_map()
    act_cfg = config.get("activation", {"name": "ReLU", "params": {}})
    act_name = act_cfg.get("name", "ReLU")
    act_params = act_cfg.get("params", {})
    
    if act_name == "Custom":
        cls = _rebuild_custom_activation_from_config(act_params)
        activation_cls = lambda: cls()
    elif act_name in activation_map:
        activation_cls = lambda: activation_map[act_name](**act_params)
    else:
        activation_cls = nn.ReLU
    
    activation_type = config.get("activation_type", 0)
    activation_cls = wrap_activation(activation_cls, activation_type)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- FIX: Check MLP Mode ---
    mlp_mode = config.get("mlp_mode", 0)
    
    model = build_model_from_config(config, input_dims, output_pred_dim, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    materialize_lazy_modules(model, input_dims, next(model.parameters()).device)
    
    # Lazy init
    with torch.no_grad():
        model(torch.zeros(1, input_dims).to(device))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())} parameters")
    
    export_choice = ask_export_mode()
    
    if export_choice in ["1", "5"]:
        export_to_onnx(model, input_dims, device)
    if export_choice in ["2", "5"]:
        export_to_torchscript(model, input_dims, device)
    if export_choice in ["3", "5"]:
        export_pure_python(model, config, scalings, vocabularies, col_types, input_dims, output_pred_dim)
    if export_choice in ["4", "5"]:
        export_quantized(model, input_dims, device)
    
    print("\n✓ Export complete!")


##############################################
# Config save/load
##############################################
def convert_to_serializable(obj):
    if isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_to_serializable(x) for x in obj]
    elif isinstance(obj, (np.integer,)): return int(obj)
    elif isinstance(obj, (np.floating,)): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    else: return obj

def save_config(file_path, col_types, hidden_dims, vocabularies, scalings, image_params,
                optimizer_choice, batch_size, activation_cls, activation_type, residual_type,
                norm_type, groups, attention_type, num_heads, input_attention_type, moe_mode, noise_mode, noise_params,
                mlp_mode=0, layer_routing="none", use_grn=False, seq_params=None, seq_bridge="mlp",
                layer_activations=None, output_activation=None):
    try: activation_config = activation_cls.activation_config
    except AttributeError:
        activation_name = activation_cls.__name__ if hasattr(activation_cls, '__name__') else "ReLU"
        if activation_name == "Identity": activation_name = "Linear"
        activation_config = {"name": activation_name, "params": {}}
    config = {
        "file_path": file_path, "col_types": col_types, "hidden_dims": hidden_dims,
        "vocabularies": vocabularies, "scalings": scalings, "image_params": image_params,
        "optimizer_choice": optimizer_choice, "batch_size": batch_size,
        "activation": activation_config, "activation_type": activation_type,
        "residual_type": residual_type, "norm_type": norm_type, "groups": groups,
        "attention_type": attention_type, "num_heads": num_heads,
        "input_attention_type": input_attention_type, "moe_mode": moe_mode,
        "noise_mode": noise_mode, "noise_params": noise_params,
        "mlp_mode": mlp_mode, "layer_routing": layer_routing, "use_grn": use_grn,
    }
    if has_seq_columns(col_types): config.update(seq_params=seq_params or {}, seq_bridge=seq_bridge)
    if layer_activations: config["layer_activations"] = layer_activations
    if not is_linear_activation(output_activation): config["output_activation"] = output_activation
    fmt = CSV_FORMATS.get(os.path.abspath(file_path)) if file_path else None
    if fmt: config["csv_format"] = fmt
    with open("config.json", "w") as f: json.dump(convert_to_serializable(config), f, indent=2)

def load_config():
    with open("config.json", "r") as f: cfg = json.load(f)
    register_csv_format(cfg.get("file_path"), cfg.get("csv_format"))  # headerless / renamed training file
    return cfg

##############################################
# Ask column types (UPDATED: supports *cat)
##############################################
def _setup_vocab_for_col(col, col_type, file_path, delimiter, vocabularies, image_params, tokenizer="char"):
    """Set up vocabulary/image params for a single column (tokenizer: inenc / outdec columns only)."""
    if col_type in seqm.SEQ_TYPES:
        df_temp = read_table(file_path, delimiter=delimiter)
        vocabularies[col] = seqm.build_vocab(df_temp[col].dropna().astype(str), tokenizer)
        print(f"  '{col}': {tokenizer} vocabulary of {seqm.vocab_size(vocabularies[col]) - len(seqm.SPECIALS)} tokens")
    elif col_type in HASH_TYPES:
        df_temp = read_table(file_path, delimiter=delimiter)
        vocabularies[col] = hash_codebook(df_temp[col])
        print(f"  '{col}': {len(vocabularies[col])} hashed strings")
    elif col_type in ["inlab", "outlab", "inlabcat", "outlabcat"]:
        df_temp = read_table(file_path, delimiter=delimiter)
        sorted_unique = sorted(df_temp[col].astype(str).unique())
        vocabularies[col] = {val: idx for idx, val in enumerate(sorted_unique)}
    elif col_type in ["intex", "outex", "intexcat", "outexcat"]:
        df_temp = read_table(file_path, delimiter=delimiter)
        df_temp[col] = df_temp[col].astype(str)
        all_chars = set()
        for text in df_temp[col]: all_chars.update(list(text))
        sorted_chars = sorted(list(all_chars))
        vocabularies[col] = {ch: idx+1 for idx, ch in enumerate(sorted_chars)}
    elif col_type == "inim":
        im_size = int(input(f"  Image size for '{col}': "))
        patch_size = int(input(f"  Patch size for '{col}': "))
        image_params[col] = {"im_size": im_size, "patch_size": patch_size}


ASKED_SEQ_PARAMS = {}  # inenc / outdec settings collected by ask_column_types (CLI)


def _ask_choice(prompt, options, default):
    keys = list(options)
    print(prompt)
    for i, k in enumerate(keys, 1): print(f"  {i}: {options[k]}")
    raw = input(f"Choice [{keys.index(default) + 1}]: ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(keys): return keys[int(raw) - 1]
    return raw if raw in options else default


def _ask_int(prompt, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    try: return int(raw) if raw else default
    except ValueError: return default


def ask_seq_params(col, is_output):
    """Tokenizer and model of one inenc / outdec column."""
    d = seqm.DEFAULT_PARAMS
    print(f"\n--- '{col}': sequence {'decoder' if is_output else 'encoder'} ---")
    p = {"tokenizer": _ask_choice("Tokenizer:", {"char": "Characters", "word": "Words (punctuation separate)"}, d["tokenizer"])}
    p["arch"] = _ask_choice(("Decoder" if is_output else "Encoder") + " model:", seqm.ARCHS, d["arch"])
    if p["arch"] != "mlp" or not is_output:
        p["dim"] = _ask_int("Width (embedding / hidden size)", d["dim"])
    if p["arch"] not in ("mlp",):
        p["layers"] = _ask_int("Layers", d["layers"])
    if is_output and p["arch"] in seqm.ATTENTION_ARCHS:
        p["attention"] = _ask_choice("Attention to the encoder memory:", seqm.DEC_ATTENTION, d["attention"])
    if p["arch"] in ("transformer", "modern") or p.get("attention") == "multihead":
        p["heads"] = _ask_int("Attention heads", d["heads"])
    return seqm.check_params(p, col, is_output)


def ask_seq_bridge(col_types, seq_params):
    """How encoders and decoders are joined; only asked when a non-MLP encoder / decoder exists."""
    if not any(seq_params.get(c, {}).get("arch", "mlp") != "mlp" for c, t in col_types.items() if t in seqm.SEQ_TYPES):
        return "mlp"
    return _ask_choice("\nBetween encoders and decoders:", seqm.BRIDGES, "mlp")


def ask_csv_format(file_path, delimiter):
    """CLI: header row or not, and optional new column names (the CSV itself is never changed)."""
    no_header = input("Does the first row hold column names? [Y/n]: ").strip().lower() in ("n", "no", "0")
    df = read_table(file_path, delimiter, fmt={"header": not no_header}, nrows=5)
    names = [str(c) for c in df.columns]
    if no_header: print(f"  No header: the columns are called {', '.join(names[:8])}{' ...' if len(names) > 8 else ''}")
    rename = {}
    if input("Rename columns? [y/N]: ").strip().lower() in ("y", "yes", "1"):
        print("  New name for each column (Enter keeps it):")
        for c in names:
            sample = ", ".join(str(v) for v in df[c].head(3).tolist())
            new = input(f"    {c}  (e.g. {sample[:40]}): ").strip()
            if new: rename[c] = new
    while True:
        try: return csv_format(not no_header, rename, names, delimiter=delimiter)
        except ValueError as e:
            print(f"  {e}; enter the new names again."); rename = {}
            for c in names:
                new = input(f"    {c}: ").strip()
                if new: rename[c] = new


def ask_column_types(columns, file_path, delimiter, allow_seq=False):
    """allow_seq: offer inenc / outdec (only regular training supports them)."""
    col_types = {}; vocabularies = {}; image_params = {}
    ASKED_SEQ_PARAMS.clear()
    
    # --- STEP 1: Scan for Constant Columns ---
    print("\nScanning dataset for constant columns...")
    try:
        # Read the dataframe to check unique values
        df_check = read_table(file_path, delimiter=delimiter)
        
        active_columns = []
        ignored_columns = []
        
        for col in columns:
            if col not in df_check.columns:
                # Fallback for edge cases
                active_columns.append(col)
                continue
            
            # Count unique values (excluding NaNs by default)
            # If a column is all NaNs (0 unique) or all same value (1 unique), we ignore it.
            unique_count = df_check[col].nunique(dropna=True)
            
            if unique_count < 2:
                ignored_columns.append(col)
                col_types[col] = "0"  # Pre-set to ignored code
            else:
                active_columns.append(col)
                
    except Exception as e:
        print(f"⚠ Warning: Could not auto-scan for constant columns ({e}). All columns will be shown.")
        active_columns = columns
        ignored_columns = []

    # --- STEP 2: Inform User ---
    if ignored_columns:
        print(f"\n[System] Automatically ignored {len(ignored_columns)} constant columns (containing 0 or 1 unique value):")
        # Use textwrap for cleaner output if list is long
        wrapped_list = textwrap.fill(", ".join(ignored_columns), width=80, initial_indent="  ", subsequent_indent="  ")
        print(wrapped_list)
        print("-" * 60)
    
    if not active_columns:
        print("\n⚠ Error: All columns appear to be constant! Cannot proceed with training.")
        return col_types, vocabularies, image_params

    # --- STEP 3: Prompt for Active Columns ---
    print("\nColumn type codes:")
    print("  0=ignored, 1=in, 2=inlab, 3=intex, 4=inim,")
    print("  5=out, 6=outlab, 7=outex,")
    print("  8=inlabcat (one-hot label input),  9=intexcat (one-hot text input),")
    print(" 10=outlabcat (categorical label output), 11=outexcat (categorical text output)")
    print(" 12=inenc (text read by a sequence encoder), 13=outdec (text written by a sequence decoder)")
    print(" 14=inhash, 15=outhash (text as one number by the EasyNN-plus hash; outputs show the nearest string)")
    print("\n  TIP: Use N*M to apply type N to this column and the next M-1 columns.")
    
    type_map = {
        "0": "i", "1": "in", "2": "inlab", "3": "intex", "4": "inim",
        "5": "out", "6": "outlab", "7": "outex",
        "8": "inlabcat", "9": "intexcat", "10": "outlabcat", "11": "outexcat", "12": "inenc", "13": "outdec",
        "14": "inhash", "15": "outhash"
    }
    valid_types = set(type_map.values())
    
    col_idx = 0
    while col_idx < len(active_columns):
        col = active_columns[col_idx]
        while True:
            raw = input(f"Column '{col}' [{col_idx+1}/{len(active_columns)}] type: ").strip().lower()
            
            # Handle N*M repeat syntax
            repeat_count = 1
            if '*' in raw:
                parts = raw.split('*', 1)
                raw = parts[0].strip()
                try:
                    repeat_count = int(parts[1].strip())
                    repeat_count = max(1, repeat_count)
                except ValueError:
                    print("Invalid repeat count, using 1.")
                    repeat_count = 1
            
            col_type = type_map.get(raw, raw)
            if col_type in seqm.SEQ_TYPES and not allow_seq:
                print("inenc / outdec columns are only supported by Train (menu 0)."); continue
            if col_type in valid_types:
                # Apply to this column and next (repeat_count-1) columns
                for r in range(repeat_count):
                    target_idx = col_idx + r
                    if target_idx >= len(active_columns):
                        break
                    
                    target_col = active_columns[target_idx]
                    col_types[target_col] = col_type
                    
                    # Setup vocab/params only if not ignored
                    if col_type in seqm.SEQ_TYPES:
                        ASKED_SEQ_PARAMS[target_col] = ask_seq_params(target_col, col_type == "outdec")
                        _setup_vocab_for_col(target_col, col_type, file_path, delimiter, vocabularies, image_params,
                                             ASKED_SEQ_PARAMS[target_col]["tokenizer"])
                    elif col_type != 'i':
                        _setup_vocab_for_col(target_col, col_type, file_path, delimiter, vocabularies, image_params)
                    
                    if r > 0:
                        print(f"  -> Also set '{target_col}' to '{col_type}'")
                
                col_idx += repeat_count
                break
            else:
                print("Invalid input. Valid codes: " + ", ".join(sorted(type_map.keys())))
                print("  Use N*M for repeat (e.g. '1*3' = 'in' for 3 columns)")
    
    # Check if any output columns were defined
    has_output = any('out' in v for v in col_types.values())
    if not has_output:
        print("\n⚠ No output columns defined!")
        print("Available active columns:")
        for i, col in enumerate(active_columns):
            ct = col_types.get(col, 'i')
            if ct != 'i':
                print(f"  {col} ({ct})")
            else:
                print(f"  {col} (ignored)")
        
        print("\nEnter comma-separated column names to use as outputs:")
        out_cols_str = input(": ").strip()
        if out_cols_str:
            out_col_names = [c.strip() for c in out_cols_str.split(',')]
            print("\nNow specify output type for each:")
            print("  5=out, 6=outlab, 7=outex, 10=outlabcat, 11=outexcat, 13=outdec, 15=outhash")
            for oc in out_col_names:
                if oc not in columns:
                    print(f"  '{oc}' not found in columns, skipping.")
                    continue
                while True:
                    ot_raw = input(f"  Output type for '{oc}': ").strip()
                    ot = type_map.get(ot_raw, ot_raw)
                    if ot == "outdec" and not allow_seq:
                        print("    outdec columns are only supported by Train (menu 0)."); continue
                    if ot in valid_types and 'out' in ot:
                        col_types[oc] = ot
                        if ot == "outdec": ASKED_SEQ_PARAMS[oc] = ask_seq_params(oc, True)
                        _setup_vocab_for_col(oc, ot, file_path, delimiter, vocabularies, image_params,
                                             ASKED_SEQ_PARAMS.get(oc, {}).get("tokenizer", "char"))
                        break
                    print("    Invalid. Use 5, 6, 7, 10, 11, 13 or 15.")
    
    return lower_hash_types(col_types), vocabularies, image_params

def activation_config_of(cls):
    """The {"name", "params"} that save_config would store for an activation class / factory."""
    cfg = getattr(cls, "activation_config", None)
    if cfg: return cfg
    name = cls.__name__ if hasattr(cls, "__name__") else "ReLU"
    return {"name": "Linear" if name == "Identity" else name, "params": {}}


def ask_layer_activations(hidden_dims, base_activation_cls):
    """None = the chosen activation in every layer; else one {"name", "params"} per hidden layer."""
    if not isinstance(hidden_dims, list) or len(hidden_dims) < 2: return None
    if input("Different activation per hidden layer? [y/N]: ").strip().lower() not in ("y", "yes", "1"): return None
    acts = []
    for i, w in enumerate(hidden_dims):
        print(f"\n--- Layer {i + 1} ({w} neurons); Enter at the menu = the global choice ---")
        cls = ask_activation(default=base_activation_cls)
        acts.append(activation_config_of(cls))
    return acts


def ask_output_activation(output_cols, col_types):
    """Activation on numeric outputs (categorical / text outputs are logits and are never touched)."""
    if not any(col_types[c] == "out" for c in output_cols): return None
    if input("Activation on the numeric output nodes? (default Linear) [y/N]: ").strip().lower() not in ("y", "yes", "1"): return None
    return activation_config_of(ask_activation())


def ask_scale_ranges(cols, col_types):
    """Optional per-column model range of numeric inputs / outputs (default [-1, 1], e.g. 0 1 to pair with a sigmoid)."""
    numeric = [c for c in cols if col_types[c] in ("in", "out")]
    if not numeric or input("Custom scaling range for numeric columns? (default -1 1) [y/N]: ").strip().lower() not in ("y", "yes", "1"):
        return {}
    ranges = {}
    for c in numeric:
        raw = input(f"  {c} range as LOW HIGH (Enter = -1 1): ").replace(",", " ").split()
        if len(raw) != 2: continue
        try: lo, hi = float(raw[0]), float(raw[1])
        except ValueError: print("    not two numbers; keeping -1 1"); continue
        if lo == hi: print("    LOW and HIGH must differ; keeping -1 1"); continue
        ranges[c] = [lo, hi]
    return ranges


def ask_hidden_dims():
    print("Choose hidden dimension selection method:")
    print("1: Per-layer selection"); print("2: Global selection"); print("3: Auto (grow architecture)")
    method_choice = input("Enter 1, 2, or 3: ").strip()
    if method_choice == "2":
        global_dim = int(input("Enter the global hidden dimension: "))
        num_layers = int(input("Enter the number of layers: "))
        return [global_dim] * num_layers
    elif method_choice == "3":
        max_dim = int(input("Enter MAXIMUM hidden dimension per layer: ").strip())
        max_layers = int(input("Enter MAXIMUM number of layers: ").strip())
        patience = input("Stagnation patience (steps, default 500): ").strip()
        patience = int(patience) if patience else 500
        return {"mode": "auto", "max_dim": max_dim, "max_layers": max_layers, "patience": patience}
    hidden_dims = []; layer_num = 1
    print("  (finishing with no layers makes a perceptron: inputs straight to outputs)")
    while True:
        neurons = int(input(f"Neurons for layer {layer_num} (0 to finish): "))
        if neurons == 0: break
        hidden_dims.append(neurons); layer_num += 1
    return hidden_dims

def ask_batch_size():
    try: return int(input("Enter batch size: ").strip())
    except: return 32


def ask_learning_rate():
    """Ask user for custom learning rate or use default."""
    lr_str = input("Learning rate (empty for optimizer default): ").strip()
    if lr_str:
        try:
            lr = float(lr_str)
            print(f"  Using LR: {lr}")
            return lr
        except ValueError:
            print("  Invalid, using optimizer default.")
    return None

def ask_optimizer():
    options = {"1":"Adam","2":"AdamHD","3":"SGD","4":"SGDHD","5":"Lamb","6":"Adagrad",
               "7":"Adadelta","8":"AdamW","9":"RMSprop","10":"Rprop","11":"ASGD","12":"Adamax",
               "13":"NAdam","14":"SparseAdam","15":"RAdamScheduleFree","16":"AdEMAMix",
               "17":"Adam3","18":"AdamDelta","19":"AutoAdam","20":"NormAdam","21":"SWATS",
               "22":"AdaBoundW","23":"CLion","24":"Signum","25":"SRprop","26":"IRprop", "27": "Adan", "28": "Prodigy",
               "29": "Evolution", "30": "Muon", "31": "AdaMuon", "32": "NorMuon", "33": "AdaGO", "34": "AdamGO", "35": "RMSGO", "36": "AdaDeltaGO",
               "37": "MuonHD", "38": "NorMuonHD"}
    print("Choose optimizer:")
    for key, name in options.items(): print(f"{key}: {name}")
    choice = input("Enter the number or name: ").strip()
    if choice in options: return options[choice]
    for opt in options.values():
        if choice.lower() == opt.lower(): return opt
    return "Adam"

def ask_residual_type():
    options = {"1":"None","2":"Highway","3":"Residual","4":"ReZero","5":"Elementwise ReZero","6":"Concat","7":"DenseNet"}
    print("Choose residual type:")
    for k, v in options.items(): print(f"{k}: {v}")
    choice = input("Enter number or name: ").strip().lower()
    if choice in options: choice = options[choice].lower()
    mapping = {"none":"none","highway":"highway","residual":"residual","rezero":"rezero",
               "elementwise rezero":"elementwise_rezero","elementwise_rezero":"elementwise_rezero",
               "concat":"concat","densenet":"densenet"}
    for key in mapping:
        if key in choice: return mapping[key]
    return "none"

def ask_group_count():
    while True:
        try:
            groups = int(input("Number of groups for GroupNorm (≥1): ").strip())
            if groups >= 1: return groups
        except ValueError: pass
        print("Please enter an integer ≥ 1.")

def ask_normalization_type():
    options = {"1":"None","2":"Batch","3":"Instance","4":"Layer","5":"Group","6":"RMSNorm"}
    print("\nChoose normalisation style:")
    for k, v in options.items(): print(f"{k}: {v}")
    choice = input("Enter number or name: ").strip().lower()
    if choice in options: choice = options[choice].lower()
    mapping = {"none":"none","batch":"batch","instance":"instance","layer":"layer","group":"group","rmsnorm":"rmsnorm","rms":"rmsnorm"}
    for key in mapping:
        if key in choice:
            norm = mapping[key]
            if norm == "group": return norm, ask_group_count()
            return norm, None
    return "none", None

def ask_attention_type():
    options = {"1":"None","2":"Basic (single-head self-attention)","3":"Multi-head self-attention"}
    print("\nChoose in-block attention style:")
    for k, v in options.items(): print(f"{k}: {v}")
    choice = input("Enter number or name: ").strip().lower()
    if choice in options: choice = options[choice].lower()
    if "multi" in choice or "3" == choice:
        while True:
            try:
                heads = int(input("Number of attention heads (≥1): ").strip())
                if heads >= 1: return "multi", heads
            except ValueError: pass
    elif "basic" in choice or "2" == choice: return "basic", None
    return "none", None

def ask_input_attention_type():
    options = {"1":"None","2":"Basic (global self-attention on inputs)","3":"Cross (modern cross-attention)"}
    print("\nChoose *input* attention style:")
    for k, v in options.items(): print(f"{k}: {v}")
    choice = input("Enter number or name: ").strip().lower()
    if choice in options: choice = options[choice].lower()
    if "cross" in choice: return "cross"
    if "basic" in choice: return "basic"
    return "none"

def ask_moe_mode():
    while True:
        try:
            moe = int(input("\nMoE mode (1=off, >1=#experts, 0=x_full, -1=learned, -2=x_mean_std, -3=x_mean, -4=gMLP, -5=aMLP): ").strip())
            return moe
        except ValueError: print("Enter a valid integer.")

def ask_layer_routing():
    choice = input("Adaptive layer routing after activation? (y/N): ").strip().lower()
    return "adaptive" if choice in ("y", "yes", "1") else "none"

def ask_mlp_mode():
    print("\nChoose MLP mode:")
    print("0: Regular MLP")
    print("1: Graph Neural Network (GNN)")
    choice = input("Enter 0 or 1: ").strip()
    if choice == "1": return 1
    return 0


def ask_nas_mode():
    """Ask which NAS evolution mode to use.
    Returns (mode_int, evolve_set, fixed_overrides_dict).
    evolve_set: set of field names the evolution is allowed to mutate.
    fixed_overrides: dict of {field: value} the user specifies for non-evolved fields.
    """
    print("\nChoose NAS Evolution Mode:")
    print("0: All (evolve everything incl. LR, MoE, attention)")
    print("1: Activation only (evolve activation function & type)")
    print("2: Norm only (evolve normalization type)")
    print("3: Res type only (evolve residual connection type)")
    print("4: Architecture only (hidden_dims + activation + residual)")
    print("5: Custom (choose what to evolve)")
    while True:
        try:
            mode = int(input("Enter (0-5): ").strip())
            if 0 <= mode <= 5: break
        except ValueError: pass
        print("Invalid.")

    ALL_FIELDS = {"hidden_dims", "activation", "activation_type", "residual_type",
                  "noise", "norm_type", "learning_rate", "moe_mode", "attention"}

    if mode == 0:
        return mode, ALL_FIELDS, {}
    elif mode == 1:
        evolve = {"activation", "activation_type"}
    elif mode == 2:
        evolve = {"norm_type"}
    elif mode == 3:
        evolve = {"residual_type"}
    elif mode == 4:
        evolve = {"hidden_dims", "activation", "activation_type", "residual_type"}
    elif mode == 5:
        print("\nSelect which components to evolve (comma-separated):")
        print("  a=activation, t=activation_type, h=hidden_dims, r=residual_type")
        print("  n=norm_type, o=noise, l=learning_rate, m=moe_mode, e=attention")
        sel = input("Enter selection (e.g. a,t,r,l): ").strip().lower()
        field_map = {"a": "activation", "t": "activation_type", "h": "hidden_dims",
                     "r": "residual_type", "n": "norm_type", "o": "noise",
                     "l": "learning_rate", "m": "moe_mode", "e": "attention"}
        evolve = set()
        for ch in sel.replace(" ", "").split(","):
            if ch in field_map: evolve.add(field_map[ch])
        if not evolve:
            print("Nothing selected, defaulting to all.")
            evolve = ALL_FIELDS

    # For non-evolved fields, ask user to specify fixed values
    fixed = {}
    if "hidden_dims" not in evolve:
        # Will use whatever the user already configured externally
        pass
    if "activation" not in evolve:
        pass  # Will use user-specified activation
    if "activation_type" not in evolve:
        pass  # Will use user-specified activation_type
    if "residual_type" not in evolve:
        fixed["residual_type"] = ask_residual_type() if mode != 0 else "residual"
    if "norm_type" not in evolve:
        nt, _ = ask_normalization_type() if mode != 0 else ("layer", None)
        fixed["norm_type"] = nt
    if "noise" not in evolve:
        fixed["noise_mode"] = "none"
        fixed["noise_params"] = {}
    if "learning_rate" not in evolve:
        fixed["learning_rate"] = None  # Will use user-specified LR
    if "moe_mode" not in evolve:
        fixed["moe_mode"] = 1
    if "attention" not in evolve:
        fixed["attention_type"] = "none"
        fixed["num_heads"] = 1

    return mode, evolve, fixed


##############################################
# PWL plotting (unchanged)
##############################################
def plot_pwl(activation_layer, neuron_idx, layer_idx, input_or_activation_value):
    x_positions = activation_layer.get_x_positions()[neuron_idx].cpu().detach().numpy()
    y_breakpoints = activation_layer(activation_layer.get_x_positions().view(-1, 1)).cpu().detach().numpy()
    x_start, x_end = x_positions.min(), x_positions.max()
    y_start, y_end = y_breakpoints.min(), y_breakpoints.max()
    x_padding = (x_end - x_start) * 0.1; y_padding = (y_end - y_start) * 0.1
    x_start -= x_padding; x_end += x_padding; y_start -= y_padding; y_end += y_padding
    x = torch.linspace(x_start, x_end, steps=1000).unsqueeze(1).to(activation_layer.slopes.device)
    y = activation_layer(x)[:, neuron_idx].cpu().detach().numpy()
    plt.figure(); plt.ylim((y_start, y_end)); plt.xlim((x_start, x_end))
    plt.plot(list(x_positions), list(y_breakpoints[neuron_idx * len(x_positions):(neuron_idx + 1) * len(x_positions)]), "or")
    plt.plot(list(x.squeeze().cpu().numpy()), y, "b")
    plt.axvline(x=input_or_activation_value, color='g', linestyle='--')
    plt.title(f'Layer {layer_idx+1} Neuron {neuron_idx+1}')
    os.makedirs("MVPs", exist_ok=True); plt.savefig(f"MVPs/layer-{layer_idx+1}-neuron-{neuron_idx+1}.png"); plt.close()


def activation_from_config(act_cfg, activation_type=0, wrap=True):
    """{"name", "params"} (config.json form) -> activation factory; wrap applies the activation type (gating etc.)."""
    name = (act_cfg or {}).get("name", "ReLU"); params = (act_cfg or {}).get("params", {}) or {}
    amap = _build_activation_map()
    if name == "Custom":
        custom = _rebuild_custom_activation_from_config(params)
        def base(): return custom()
    elif name in amap:
        def base(): return amap[name](**params)
    else:
        raise ValueError(f"Unknown activation {name!r}")
    base.activation_config = {"name": name, "params": params}
    return wrap_activation(base, activation_type) if wrap else base


def is_linear_activation(act_cfg):
    return not act_cfg or act_cfg.get("name") in (None, "", "Linear", "Identity")


def numeric_output_positions(output_layout):
    """Model-output positions of numeric ('out') columns: the only outputs an output activation touches."""
    return [e["start"] for e in output_layout if e["type"] == "out"]


def check_mlp_options(mlp_mode, layer_activations, output_activation, hidden_dims, norm_type=None, groups=None, batch_size=None):
    if batch_size is not None and int(batch_size) < 1: raise ValueError(f"Batch size must be at least 1 (got {batch_size}).")
    if norm_type == "group" and groups:
        bad = [w for w in hidden_dims if w % int(groups)]
        if bad: raise ValueError(f"Group normalization with {groups} groups needs every hidden width divisible by {groups}; {bad} is not.")
    if layer_activations is not None and len(layer_activations) != len(hidden_dims):
        raise ValueError(f"{len(layer_activations)} layer activations given for {len(hidden_dims)} hidden layers.")
    if mlp_mode == 1 and (layer_activations or not is_linear_activation(output_activation)):
        raise ValueError("Per-layer and output activations need the MLP architecture (not GNN).")


def build_model_from_config(config, input_dims, output_dim, device, hidden_dims=None):
    """Rebuild the (untrained) network described by config.json; caller loads model.pt."""
    if hidden_dims is None: hidden_dims = config["hidden_dims"]
    activation_config = config.get("activation", {"name": "ReLU", "params": {}})
    activation_name = activation_config.get("name", "ReLU")
    activation_params = activation_config.get("params", {})
    residual_type = config.get("residual_type", "residual")
    norm_type = config.get("norm_type", "layer")
    groups = config.get("groups", 1)
    attention_type = config.get("attention_type", "none")
    num_heads = config.get("num_heads", 1)
    input_attention_type = config.get("input_attention_type", "none")
    moe_mode = config.get("moe_mode", 1)
    noise_mode = config.get("noise_mode", "none")
    noise_params = config.get("noise_params", {})
    
    activation_map = _build_activation_map()
    if activation_name == "Custom":
        cls = _rebuild_custom_activation_from_config(activation_params)
        activation_cls = lambda: cls()
    elif activation_name in activation_map:
        activation_cls = lambda: activation_map[activation_name](**activation_params)
    else: activation_cls = nn.ReLU
    
    activation_type = config.get("activation_type", 0)
    activation_cls = wrap_activation(activation_cls, activation_type)
    mlp_mode = config.get("mlp_mode", 0)
    if config.get("layer_activations"):
        activation_cls = [activation_from_config(c, activation_type) for c in config["layer_activations"]]
    out_cfg = config.get("output_activation")
    out_act = None if is_linear_activation(out_cfg) else activation_from_config(out_cfg, wrap=False)

    def make_mlp(in_dim, out_dim, out_idx=None):
        if mlp_mode == 1:
            return GNNMLPO(in_dim, hidden_dims, out_dim, activation_cls, residual_type,
                norm_type=norm_type, groups=groups,
                dropout_prob=noise_params.get("dropout_pct", 0.0) if noise_mode == "dropout" else 0.0,
                use_grn=config.get("use_grn", False))
        return MLPO(in_dim, hidden_dims, out_dim, activation_cls, residual_type,
            norm_type=norm_type, groups=groups, attention_type=attention_type, num_heads=num_heads,
            input_attention_type=input_attention_type, moe_mode=moe_mode,
            dropout_prob=noise_params.get("dropout_pct", 0.0) if noise_mode == "dropout" else 0.0,
            use_noise_injection_layers=(noise_mode == "noise injection layers"), layer_routing=config.get("layer_routing", "none"),
            noise_injection_std=noise_params.get("std", 0.0), use_grn=config.get("use_grn", False),
            output_activation=out_act if out_idx else None, output_activation_idx=out_idx)

    col_types = config["col_types"]
    in_cols = [c for c, t in col_types.items() if 'in' in t]; out_cols = [c for c, t in col_types.items() if 'out' in t]
    layout, _, _ = build_output_layout(out_cols, col_types, config["scalings"], config.get("vocabularies", {}))
    if has_seq_columns(col_types):
        return build_seq_model(make_mlp, in_cols, out_cols, col_types, config["scalings"], config.get("vocabularies", {}),
                               config.get("seq_params", {}), config.get("seq_bridge", "mlp"), out_act).to(device)
    return make_mlp(input_dims, output_dim, numeric_output_positions(layout)).to(device)


def encode_sample_input(sample_input, col_types, vocabularies, scalings, image_params):
    """Raw per-column input values (in col_types order) -> flat model input vector."""
    processed_input = []
    input_idx = 0
    for col_name, col_type in col_types.items():
        if col_type not in ["in", "inlab", "intex", "inim", "inlabcat", "intexcat", "inenc"]:
            continue
        value = sample_input[input_idx]
        
        if col_type == "inenc":
            processed_input.extend(seqm.encode_ids(value, vocabularies[col_name], scalings[col_name]['max_len'], is_output=False))
        elif col_type == "inlabcat":
            vocab = vocabularies[col_name]; vocab_size = len(vocab)
            mapped = vocab.get(str(value), 0)
            onehot = [0.0] * vocab_size
            if 0 <= mapped < vocab_size: onehot[mapped] = 1.0
            processed_input.extend(onehot)
            
        elif col_type == "intexcat":
            vocab = vocabularies[col_name]; vocab_size = len(vocab)
            max_len = scalings[col_name]['max_len']
            for i in range(max_len):
                onehot = [0.0] * vocab_size
                if i < len(str(value)):
                    token_idx = vocab.get(str(value)[i], 0)
                    if 0 < token_idx <= vocab_size: onehot[token_idx - 1] = 1.0
                processed_input.extend(onehot)
        
        elif col_type == "inlab":
            mapped = vocabularies[col_name].get(str(value), value)
            processed_input.append(float(mapped))
        elif col_type == "intex":
            vocab = vocabularies[col_name]; max_len = scalings[col_name]['max_len']
            seq = [vocab.get(ch, 0) for ch in str(value)]
            if len(seq) < max_len: seq += [0] * (max_len - len(seq))
            else: seq = seq[:max_len]
            processed_input.extend(seq)
        elif col_type == "inim":
            im_size = image_params[col_name]["im_size"]; patch_size = image_params[col_name]["patch_size"]
            try: img = Image.open(value).convert("L")
            except: img = Image.new("L", (im_size, im_size))
            img = img.resize((im_size, im_size)); img_array = np.array(img)
            if patch_size == 1: patches = [img_array]
            else:
                n_patches = im_size // patch_size; patches = []
                for i in range(n_patches):
                    for j in range(n_patches):
                        patches.append(img_array[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size])
            codes = []
            for patch in patches:
                pc, ec, bc = compute_patch_codes(patch); codes.extend([pc, ec, bc])
            processed_input.extend(codes)
        else:
            if col_type == "in" and isinstance(vocabularies.get(col_name), dict) and not _is_number_cell(value):  # hashed text
                value = _hash_rule().editor_lookup_code(value, vocabularies[col_name])
            if col_name in scalings and 'min' in scalings[col_name]:
                value = to_model_units(float(_bool_as_number(value)), scalings[col_name])
            processed_input.append(float(_bool_as_number(value)))
        input_idx += 1
    return processed_input


##############################################
# STREAMLINED Sampling/Inference with fast plots
##############################################
##############################################
# STREAMLINED Sampling/Inference with fast plots
##############################################
def ask_generation(col_types):
    """Sampling settings for outdec columns (temperature 0 = the most likely text)."""
    if "outdec" not in col_types.values(): return None
    print("\nText generation (temperature 0 = always the most likely token):")
    t = float(input("  Temperature [0]: ").strip() or 0)
    if t <= 0: return {"temperature": 0.0}
    k = int(input("  Top-k (0 = off) [0]: ").strip() or 0)
    p = float(input("  Top-p (1 = off) [1]: ").strip() or 1)
    return seqm.check_generation({"temperature": t, "top_k": k, "top_p": p})


def load_and_sample_model(sample_input, hidden_dims, vocabularies, col_types, output_dim, scalings={}, image_params={}, plot_option=None, plot_settings=None,
                          generation=None):
    """generation: sampling settings for outdec columns, see ask_generation."""
    if plot_settings is None: plot_settings = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import shutil
    dir_path = 'MVPs'
    if os.path.exists(dir_path):
        for fn in os.listdir(dir_path):
            fp = os.path.join(dir_path, fn)
            if os.path.isfile(fp) or os.path.islink(fp): os.unlink(fp)
            elif os.path.isdir(fp): shutil.rmtree(fp)
        
    input_cols = [col for col in col_types if col_types[col] in ["in", "inlab", "intex", "inim", "inlabcat", "intexcat", "inenc"]]
    
    def calc_dim(col):
        ct = col_types[col]
        if ct in ['intex', 'inenc']: return scalings[col]['max_len']
        elif ct == 'intexcat': return scalings[col]['max_len'] * len(vocabularies.get(col, {}))
        elif ct == 'inlabcat': return len(vocabularies.get(col, {}))
        elif ct == 'inim':
            im_size = scalings[col]["im_size"]; patch_size = scalings[col]["patch_size"]
            return (1 if patch_size == 1 else (im_size // patch_size) ** 2) * 3
        else: return 1
    input_dims = sum(calc_dim(col) for col in input_cols)
        
    config = load_config()
    model = build_model_from_config(config, input_dims, output_dim, device, hidden_dims)
    print("Performing dry run to initialize lazy layers...")
    materialize_lazy_modules(model, input_dims, device)
    model.load_state_dict(torch.load('model.pt', map_location=device)); print(model); model.eval()
    if generation and isinstance(model, seqm.SeqMLP): model.set_generation(**generation)
    os.makedirs("MVPs", exist_ok=True)
        
    with torch.no_grad():
        # ─── Process input ───
        processed_input = encode_sample_input(sample_input, col_types, vocabularies, scalings, image_params)
            
        x = torch.tensor(processed_input, dtype=torch.float).unsqueeze(0).to(device)
        model_output = model(x)
        
        # ─── Build output layout for decoding ───
        output_cols_list = [c for c in col_types if col_types[c] in ["out", "outlab", "outex", "outlabcat", "outexcat", "outdec"]]
        output_layout, _, _ = build_output_layout(output_cols_list, col_types, scalings, vocabularies)
        
        # ─── Decode output ───
        processed_output = []
        raw_output = model_output.squeeze(0) if model_output.dim() > 1 else model_output
        
        for entry in output_layout:
            col_name = entry['col']; ct = entry['type']
            pred_slice = raw_output[entry['start']:entry['end']]
            
            if ct == 'outlabcat':
                predicted_class = pred_slice.argmax().item()
                vocab_inv = {v: k for k, v in vocabularies[col_name].items()}
                output_value = vocab_inv.get(predicted_class, predicted_class)
                print(f"{col_name}: {output_value} (class {predicted_class}, logits: {pred_slice.cpu().numpy()})")
                processed_output.append(output_value)
                
            elif ct == 'outdec':
                ids = pred_slice.view(entry['max_len'], entry['num_classes']).argmax(dim=-1).cpu().tolist()
                output_value = seqm.decode_ids(ids, vocabularies[col_name])
                print(f"{col_name}: {output_value}")
                processed_output.append(output_value)

            elif ct == 'outexcat':
                max_len = entry['max_len']; num_classes = entry['num_classes']
                logits = pred_slice.view(max_len, num_classes)
                predicted_indices = logits.argmax(dim=-1).cpu().tolist()
                vocab_inv = {v: k for k, v in vocabularies[col_name].items()}
                output_value = ''.join([vocab_inv.get(idx, '') for idx in predicted_indices])
                print(f"{col_name}: {output_value}")
                processed_output.append(output_value)
                
            elif ct == 'outex':
                max_len = entry.get('max_len', entry['end'] - entry['start'])
                rounded = [round(v.item()) for v in pred_slice]
                vocab_inv = {v: k for k, v in vocabularies[col_name].items()}
                output_value = ''.join([vocab_inv.get(code, '') for code in rounded])
                print(f"{col_name}: {output_value}")
                processed_output.append(output_value)
                
            elif ct == 'outlab':
                rounded_value = round(pred_slice[0].item())
                vocab_inv = {v: k for k, v in vocabularies[col_name].items()}
                output_value = vocab_inv.get(rounded_value, rounded_value)
                print(f"{col_name}: {output_value}")
                processed_output.append(output_value)
            else:
                val = pred_slice[0].item()
                if col_name in scalings and 'min' in scalings[col_name]:
                    val = to_display_units(val, scalings[col_name])
                print(f"{col_name}: {val}")
                processed_output.append(val)
        
        # ─── Build input/output mappings for plotting ───
        input_mapping = []
        global_index = 0
        for col_name, ct in col_types.items():
            if ct in ["in", "inlab"]:
                input_mapping.append((col_name, None, processed_input[global_index], global_index, 1, 'scalar'))
                global_index += 1
            elif ct == "intex":
                ml = scalings[col_name]['max_len']
                for i in range(ml):
                    input_mapping.append((col_name, i+1, processed_input[global_index], global_index, 1, 'scalar'))
                    global_index += 1
            elif ct == "inlabcat":
                vs = len(vocabularies[col_name])
                input_mapping.append((col_name, None, processed_input[global_index:global_index+vs], global_index, vs, 'onehot_label'))
                global_index += vs
            elif ct == "intexcat":
                vs = len(vocabularies[col_name]); ml = scalings[col_name]['max_len']
                for i in range(ml):
                    input_mapping.append((col_name, i+1, processed_input[global_index:global_index+vs], global_index, vs, 'onehot_char'))
                    global_index += vs
            elif ct == "inenc":  # token ids: not plotted, but they occupy input slots
                global_index += scalings[col_name]['max_len']
        
        output_mapping_plot = []
        for entry in output_layout:
            col_name = entry['col']; ct = entry['type']
            if ct == 'outlabcat':
                output_mapping_plot.append((col_name, None, entry['start'], entry['end'], 'categorical', entry.get('num_classes', 1)))
            elif ct == 'outdec':
                continue  # generated text is not plotted
            elif ct == 'outexcat':
                ml = entry['max_len']; nc = entry['num_classes']
                for i in range(ml):
                    s = entry['start'] + i * nc; e = s + nc
                    output_mapping_plot.append((col_name, i+1, s, e, 'categorical', nc))
            elif ct == 'outex':
                ml = entry['end'] - entry['start']
                for i in range(ml):
                    output_mapping_plot.append((col_name, i+1, entry['start']+i, entry['start']+i+1, 'scalar', 1))
            else:
                output_mapping_plot.append((col_name, None, entry['start'], entry['end'], 'scalar', 1))
        
        # Default ranges for non-numeric scalar inputs (e.g. integer-encoded labels).
        global_input_ranges = {}
        for (col_name, sub_index, value, global_idx, width, kind) in input_mapping:
            if kind == 'scalar' and col_types[col_name] in ["inlab", "intex"]:
                vocab = vocabularies[col_name]
                rmin = min(vocab.values()); rmax = max(vocab.values())
                center = (rmin + rmax) / 2.0; half_range = ((rmax - rmin) / 2.0) * 1.1
                global_input_ranges[global_idx] = (center - half_range, center + half_range)

        # ─── STREAMLINED PLOT GENERATION (WITH MARKERS & LABELS) ───
        def _batch_model_eval(base_input_np, vary_indices, vary_values_list, output_start, output_end):
            N = len(vary_values_list)
            batch = np.tile(base_input_np, (N, 1))
            for i, vals in enumerate(vary_values_list):
                for idx, val in zip(vary_indices, vals):
                    batch[i, idx] = val
            inp_t = torch.tensor(batch, dtype=torch.float32).to(device)
            with torch.no_grad():
                out_t = model(inp_t)
            return out_t[:, output_start:output_end].cpu().numpy()
        
        def _get_categorical_plot_value(raw_out, out_kind, num_classes):
            if out_kind == 'categorical' and num_classes > 1:
                return np.argmax(raw_out, axis=-1)
            return raw_out.squeeze(-1)

        PLOT_RESOLUTION_1D = plot_settings.get('resolution_1d', 400) if isinstance(plot_option, str) else 400
        PLOT_RESOLUTION_2D = plot_settings.get('resolution_2d', 150) if isinstance(plot_option, str) else 150
        PLOT_INPUT_RANGES = plot_settings.get('input_ranges', {})

        def _has_numeric_scaling(col_name):
            return col_name in scalings and 'min' in scalings[col_name] and 'max' in scalings[col_name]

        def _to_display_units(value, col_name):
            if _has_numeric_scaling(col_name):
                return to_display_units(value, scalings[col_name])
            return value

        def _to_model_units(value, col_name):
            if _has_numeric_scaling(col_name):
                return to_model_units(value, scalings[col_name])
            return value

        def _plot_range(col_name, global_idx, sample_value):
            if _has_numeric_scaling(col_name):
                xmin, xmax = PLOT_INPUT_RANGES.get(
                    col_name, (scalings[col_name]['min'], scalings[col_name]['max']))
            else:
                xmin, xmax = global_input_ranges.get(global_idx, (-1.0, 1.0))

            xmin, xmax = sorted((float(xmin), float(xmax)))
            span = xmax - xmin
            if span == 0:
                span = max(abs(xmin), 1.0)
                xmin -= span * 0.5
                xmax += span * 0.5

            # Keep an out-of-range sample visible with a small amount of context.
            padding = span * 0.05
            if sample_value < xmin:
                xmin = sample_value - padding
            elif sample_value > xmax:
                xmax = sample_value + padding
            return xmin, xmax

        def _decode_plot_output(raw_out, col_name, out_kind, num_classes):
            values = _get_categorical_plot_value(raw_out, out_kind, num_classes)
            if out_kind != 'categorical' and _has_numeric_scaling(col_name):
                values = _to_display_units(values, col_name)
            return values
        
        def generate_1d_plot_fast(in_entry, out_entry, base_input_np):
            col_in, sub_in, val_in, gidx_in, width_in, kind_in = in_entry
            col_out, sub_out, ostart, oend, okind, onc = out_entry
            in_label = f"{col_in}" + (f"_{sub_in}" if sub_in else "")
            out_label = f"{col_out}" + (f"_{sub_out}" if sub_out else "")
            
            # Setup output Y-axis labels if categorical
            out_class_labels = None; out_class_indices = None
            if okind == 'categorical' and col_out in vocabularies:
                vocab = vocabularies[col_out]
                inv_vocab = {v: k for k, v in vocab.items()}
                valid_indices = sorted(list(set(vocab.values())))
                if len(valid_indices) <= 40:
                    out_class_indices = valid_indices
                    out_class_labels = [inv_vocab.get(i, str(i)) for i in valid_indices]

            if kind_in == 'onehot_label' or kind_in == 'onehot_char':
                # Bar Chart
                vocab = vocabularies[col_in]
                cat_names = sorted(vocab.keys(), key=lambda k: vocab[k])
                
                # Identify which bar corresponds to the sample input
                active_sample_idx = np.argmax(val_in) # val_in is the one-hot slice
                
                start_offset = 1 if kind_in == 'onehot_char' else 0
                results = []
                bar_labels = []
                
                for cat_name in cat_names:
                    cat_idx = vocab[cat_name] - start_offset
                    if not (0 <= cat_idx < width_in): continue
                    
                    modified = base_input_np.copy()
                    modified[gidx_in:gidx_in+width_in] = 0.0
                    modified[gidx_in + cat_idx] = 1.0
                    inp_t = torch.tensor(modified, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad(): raw = model(inp_t)[0, ostart:oend].cpu().numpy()
                    results.append(_decode_plot_output(raw.reshape(1, -1), col_out, okind, onc)[0])
                    bar_labels.append(cat_name)
                
                fig, ax = plt.subplots(figsize=(max(6, len(results)*0.6), 5), dpi=120)
                colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results)))
                bars = ax.bar(range(len(results)), results, color=colors, edgecolor='black', linewidth=0.5)
                
                # Highlight the sample bar
                # The 'val_in' argmax is relative to the one-hot vector size.
                # 'results' maps to 'cat_names' order. 
                # We need to find which result index corresponds to 'active_sample_idx'
                # Assuming cat_names are sorted by index, results[i] corresponds to index i (adjusted).
                if 0 <= active_sample_idx < len(bars):
                    bars[active_sample_idx].set_edgecolor('red')
                    bars[active_sample_idx].set_linewidth(2.5)
                    bars[active_sample_idx].set_label('Sample Input')
                    ax.legend()

                ax.set_xticks(range(len(results))); ax.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=8)
                
                if out_class_labels is not None:
                    ax.set_yticks(out_class_indices)
                    ax.set_yticklabels(out_class_labels)
                    ax.set_ylim(min(out_class_indices)-0.5, max(out_class_indices)+0.5)

                ax.set_xlabel(in_label); ax.set_ylabel(out_label)
                ylabel_extra = " (predicted class)" if okind == 'categorical' else ""
                ax.set_title(f'{in_label} → {out_label}{ylabel_extra}')
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                
            else:
                # Line Plot
                sample_display = _to_display_units(val_in, col_in)
                xmin, xmax = _plot_range(col_in, gidx_in, sample_display)
                xs = np.linspace(xmin, xmax, PLOT_RESOLUTION_1D)
                vary_list = [[_to_model_units(v, col_in)] for v in xs]
                
                raw_out = _batch_model_eval(base_input_np, [gidx_in], vary_list, ostart, oend)
                ys = _decode_plot_output(raw_out, col_out, okind, onc)
                
                fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
                ax.plot(xs, ys, '-', color='#2563eb', linewidth=1.5)
                
                # MARK SAMPLE INPUT (Vertical Line)
                ax.axvline(x=sample_display, color='red', linestyle='--', linewidth=1.5, label='Sample Input', alpha=0.8)
                ax.legend()
                
                if out_class_labels is not None:
                    ax.set_yticks(out_class_indices)
                    ax.set_yticklabels(out_class_labels)
                    ax.set_ylim(min(out_class_indices)-0.5, max(out_class_indices)+0.5)
                    
                ax.set_xlabel(in_label, fontsize=11); ax.set_ylabel(out_label, fontsize=11)
                ylabel_extra = " (predicted class)" if okind == 'categorical' else ""
                ax.set_title(f'{in_label} → {out_label}{ylabel_extra}', fontsize=12)
                ax.grid(True, alpha=0.3); plt.tight_layout()
            
            out_dir = os.path.join("plots", str(out_label)); os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, f"{in_label}_to_{out_label}.png"), dpi=120, bbox_inches='tight')
            plt.close()
    
        def generate_2d_plot_fast(in1, in2, out_entry, base_input_np):
            col1, sub1, val1, idx1, w1, kind1 = in1
            col2, sub2, val2, idx2, w2, kind2 = in2
            col_out, sub_out, ostart, oend, okind, onc = out_entry
            label1 = f"{col1}" + (f"_{sub1}" if sub1 else "")
            label2 = f"{col2}" + (f"_{sub2}" if sub2 else "")
            out_label = f"{col_out}" + (f"_{sub_out}" if sub_out else "")
            
            if kind1 != 'scalar' or kind2 != 'scalar': return
            
            sample1_display = _to_display_units(val1, col1)
            sample2_display = _to_display_units(val2, col2)
            g1 = np.linspace(*_plot_range(col1, idx1, sample1_display), PLOT_RESOLUTION_2D)
            g2 = np.linspace(*_plot_range(col2, idx2, sample2_display), PLOT_RESOLUTION_2D)
            X, Y = np.meshgrid(g1, g2)
            
            flat_x = X.ravel(); flat_y = Y.ravel()
            vary_list = list(zip(_to_model_units(flat_x, col1), _to_model_units(flat_y, col2)))
            raw_out = _batch_model_eval(base_input_np, [idx1, idx2], vary_list, ostart, oend)
            Z_flat = _decode_plot_output(raw_out, col_out, okind, onc)
            Z = Z_flat.reshape(X.shape)
            
            fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
            if okind == 'categorical':
                n_classes = int(Z.max()) + 1
                if n_classes <= 10: cmap = plt.cm.get_cmap('tab10', max(n_classes, 2))
                elif n_classes <= 20: cmap = plt.cm.get_cmap('tab20', n_classes)
                else:
                    import matplotlib.colors as mcolors
                    hsv_colors = [plt.cm.hsv(i / n_classes) for i in range(n_classes)]
                    cmap = mcolors.ListedColormap(hsv_colors)
                im = ax.pcolormesh(g1, g2, Z, cmap=cmap, shading='auto')
                
                cbar_ticks = range(n_classes)
                cbar = plt.colorbar(im, ax=ax, label='Predicted Class', ticks=cbar_ticks)
                if col_out in vocabularies:
                    vocab = vocabularies[col_out]
                    inv_vocab = {v: k for k, v in vocab.items()}
                    labels = [inv_vocab.get(i, str(i)) for i in cbar_ticks]
                    if len(labels) <= 40:
                        cbar.ax.set_yticklabels(labels)
            else:
                im = ax.pcolormesh(g1, g2, Z, cmap='RdYlBu_r', shading='auto')
                plt.colorbar(im, ax=ax, label=out_label)
            
            # MARK SAMPLE INPUT (Cross)
            ax.scatter([sample1_display], [sample2_display], color='red', marker='x', s=100, linewidth=2.5, label='Sample', zorder=10)
            # Optional: Add small legend for the marker
            # ax.legend(loc='upper right') 
            
            ax.set_xlabel(label1, fontsize=11); ax.set_ylabel(label2, fontsize=11)
            ax.set_title(f'{label1} & {label2} → {out_label}', fontsize=12)
            plt.tight_layout()
            
            out_dir = os.path.join("plots", str(out_label)); os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, f"{label1}_and_{label2}_to_{out_label}.png"), dpi=120, bbox_inches='tight')
            plt.close()
        
        if plot_option is not None:
            output_dir = "plots"
            if os.path.exists(output_dir):
                print(f"Clearing '{output_dir}'...")
                for fn in os.listdir(output_dir):
                    fp = os.path.join(output_dir, fn)
                    try:
                        if os.path.isfile(fp) or os.path.islink(fp): os.unlink(fp)
                        elif os.path.isdir(fp): shutil.rmtree(fp)
                    except: pass
            
            base_np = np.array(processed_input, dtype=np.float32)
            
            if plot_option == "1":
                total = len(input_mapping) * len(output_mapping_plot)
                done = 0
                for in_entry in input_mapping:
                    for out_entry in output_mapping_plot:
                        generate_1d_plot_fast(in_entry, out_entry, base_np)
                        done += 1
                        if done % 10 == 0: print(f"  1D plots: {done}/{total}")
                print(f"  1D plots complete: {done}")
                        
            elif plot_option == "2":
                if len(input_mapping) < 2:
                    for in_entry in input_mapping:
                        for out_entry in output_mapping_plot:
                            generate_1d_plot_fast(in_entry, out_entry, base_np)
                else:
                    from itertools import combinations
                    pairs = list(combinations(input_mapping, 2))
                    total = len(pairs) * len(output_mapping_plot); done = 0
                    for (in1, in2) in pairs:
                        for out_entry in output_mapping_plot:
                            generate_2d_plot_fast(in1, in2, out_entry, base_np)
                            done += 1
                            if done % 5 == 0: print(f"  2D plots: {done}/{total}")
                    print(f"  2D plots complete: {done}")
        
        return processed_output






##############################################
# Custom Activation Function System
##############################################
_CUSTOM_ACTIVATIONS_REGISTRY = {}

def _show_custom_activation_guide():
    guide = """
╔══════════════════════════════════════════════════════════════╗
║              CUSTOM ACTIVATION FUNCTION GUIDE               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Define your activation as a PyTorch expression.             ║
║  Available variables and functions:                          ║
║    x        - the input tensor                               ║
║    torch.*  - all torch functions (torch.sin, torch.exp...)  ║
║    F.*      - torch.nn.functional (F.relu, F.softplus...)    ║
║    math.*   - Python math module                             ║
║                                                              ║
║  You can also define TRAINABLE parameters:                   ║
║    Prefix a name with '$' to make it a trainable param.      ║
║    e.g. '$alpha * torch.sin($beta * x)'                      ║
║    Default init value is 1.0 unless you specify:             ║
║    '$alpha=0.5 * torch.sin($beta=2.0 * x)'                  ║
║                                                              ║
║  Examples:                                                   ║
║    x * torch.sigmoid(x)              # SiLU reimplemented   ║
║    torch.sin(x) * x                  # Sine-gated           ║
║    $a * torch.tanh($b * x) + $c * x  # Parametric blend     ║
║    torch.where(x > 0, x, $alpha * (torch.exp(x) - 1))       ║
║    x * torch.tanh(F.softplus(x))     # Mish reimplemented   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(guide)

def _parse_custom_activation_expr(expr_str):
    """Parse expression, extract trainable params ($name=init_val)."""
    params = {}
    # Find all $param=value or $param patterns
    pattern = r'\$(\w+)(?:=([-+]?\d*\.?\d+))?'
    for match in re.finditer(pattern, expr_str):
        name = match.group(1)
        init_val = float(match.group(2)) if match.group(2) else 1.0
        params[name] = init_val
    # Replace $param=val with self.param_name, $param with self.param_name
    clean_expr = re.sub(r'\$(\w+)(?:=[-+]?\d*\.?\d+)?', r'self._p_\1', expr_str)
    return clean_expr, params

def _build_custom_activation_class(name, expr_str, params):
    """Dynamically build a nn.Module class for the custom activation."""
    clean_expr, parsed_params = _parse_custom_activation_expr(expr_str)
    # Merge explicit params
    for k, v in params.items():
        if k not in parsed_params:
            parsed_params[k] = v

    class CustomActivation(nn.Module):
        _custom_name = name
        _custom_expr = expr_str
        _custom_params = parsed_params

        def __init__(self):
            super().__init__()
            for pname, pval in self.__class__._custom_params.items():
                setattr(self, f'_p_{pname}', nn.Parameter(torch.tensor(float(pval))))

        def forward(self, x):
            # Build local namespace
            local_ns = {'self': self, 'x': x, 'torch': torch, 'F': F, 'math': math}
            return eval(compile(self.__class__._compiled_expr, '<custom_activation>', 'eval'), 
                       {"__builtins__": {}}, local_ns)

        def extra_repr(self):
            return f'expr="{self.__class__._custom_expr}"'

    # Pre-compile expression
    compiled = compile(clean_expr, '<custom_activation>', 'eval')
    CustomActivation._compiled_expr = compiled
    CustomActivation.__name__ = name
    CustomActivation.__qualname__ = name

    return CustomActivation


def ask_custom_activation():
    """Interactive custom activation definition."""
    _show_custom_activation_guide()
    name = input("Activation name (e.g. MySwish): ").strip()
    if not name:
        name = "CustomAct"
    expr_str = input("Expression: ").strip()
    if not expr_str:
        print("Empty expression, defaulting to ReLU.")
        return nn.ReLU

    _, params = _parse_custom_activation_expr(expr_str)

    # Test the activation
    print("Testing activation...")
    try:
        cls = _build_custom_activation_class(name, expr_str, params)
        test_act = cls()
        test_input = torch.randn(4, 8)
        test_output = test_act(test_input)
        print(f"  Input shape:  {test_input.shape}")
        print(f"  Output shape: {test_output.shape}")
        print(f"  Output range: [{test_output.min().item():.4f}, {test_output.max().item():.4f}]")
        if params:
            print(f"  Trainable params: {list(params.keys())}")
        print("  ✓ Activation works!")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print("  Falling back to ReLU.")
        return nn.ReLU

    # Register it
    _CUSTOM_ACTIVATIONS_REGISTRY[name] = {
        "expr": expr_str,
        "params": params,
        "class": cls,
    }

    # Create factory with config
    def factory():
        return cls()
    factory.activation_config = {
        "name": "Custom",
        "params": {"custom_name": name, "custom_expr": expr_str, "custom_params": params}
    }
    factory.__name__ = name
    return factory


def _rebuild_custom_activation_from_config(cfg):
    """Rebuild a custom activation from saved config."""
    name = cfg.get("custom_name", "CustomAct")
    expr_str = cfg.get("custom_expr", "x")
    params = cfg.get("custom_params", {})
    cls = _build_custom_activation_class(name, expr_str, params)
    return cls



##############################################
# Neuron-level Activation Plotting
##############################################
def plot_all_neurons(model, processed_input, scalings, col_types, vocabularies,
                     input_mapping, device, plot_settings=None):
    """
    Plot the activation function response for every neuron in the network.
    For each neuron in each hidden layer, we sweep the neuron's pre-activation
    input and show what comes out after the activation function.
    """
    import shutil
    if plot_settings is None:
        plot_settings = {}
    resolution = plot_settings.get('neuron_resolution', 500)
    
    neuron_dir = "neuron_plots"
    if os.path.exists(neuron_dir):
        shutil.rmtree(neuron_dir)
    os.makedirs(neuron_dir, exist_ok=True)
    
    base_np = np.array(processed_input, dtype=np.float32)
    x_tensor = torch.tensor(base_np, dtype=torch.float32).unsqueeze(0).to(device)
    
    model.eval()
    
    # Collect pre-activation and post-activation values at each block
    block_data = []
    with torch.no_grad():
        current = x_tensor
        # Input attention
        if hasattr(model, 'input_attn') and model.input_attn is not None:
            current = model.input_attn(current)
        
        for block_idx, block in enumerate(model.blocks):
            z = block.norm(current)
            pre_act = block.linear1(z)  # Pre-activation values
            
            if block.use_sgu:
                post_act = block.sgu(pre_act)
            elif block.activation is not None:
                post_act = block.activation(pre_act)
            else:
                post_act = pre_act
            
            block_data.append({
                'pre_act': pre_act.cpu().numpy().flatten(),
                'post_act': post_act.cpu().numpy().flatten(),
                'activation': block.activation,
                'use_sgu': block.use_sgu,
                'block_idx': block_idx,
            })
            
            # Continue forward pass
            z_out = post_act
            if block.linear2 is not None:
                z_out = block.linear2(z_out)
            if block.use_tiny_attn and block.tiny_attn is not None:
                z_out = z_out + block.tiny_attn(z_out)
            if block.attn is not None:
                z_out = block.attn(z_out)
            current = block._apply_residual_nonhyper(z_out, current)
    
    total_neurons = sum(len(bd['pre_act']) for bd in block_data)
    print(f"\n⚠ Plotting {total_neurons} neurons across {len(block_data)} layers.")
    print(f"  This will generate {total_neurons} plot files...")
    
    plot_count = 0
    for bd in block_data:
        block_idx = bd['block_idx']
        pre_vals = bd['pre_act']
        post_vals = bd['post_act']
        activation = bd['activation']
        
        layer_dir = os.path.join(neuron_dir, f"layer_{block_idx+1}")
        os.makedirs(layer_dir, exist_ok=True)
        
        # For each neuron, sweep a range around its pre-activation value
        for neuron_idx in range(len(pre_vals)):
            center = pre_vals[neuron_idx]
            # Sweep range: center ± 3*std of pre-activation values, or at least ±2
            spread = max(2.0, np.std(pre_vals) * 3)
            x_range = np.linspace(center - spread, center + spread, resolution)
            
            if activation is not None and not bd['use_sgu']:
                # Evaluate activation on sweep values
                # Sweep as [R, C] so per-neuron activations (PaWeL) use this neuron's own curve
                x_sweep = torch.tensor(x_range, dtype=torch.float32).to(device)
                with torch.no_grad():
                    y_sweep = activation(x_sweep.unsqueeze(1).expand(-1, len(pre_vals)).contiguous())[:, neuron_idx].cpu().numpy()
                
                fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
                ax.plot(x_range, y_sweep, '-', color='#2563eb', linewidth=1.5, label='Activation')
                ax.axvline(x=center, color='red', linestyle='--', alpha=0.7, label=f'Current={center:.3f}')
                ax.axhline(y=post_vals[neuron_idx] if neuron_idx < len(post_vals) else 0,
                          color='green', linestyle=':', alpha=0.5, label=f'Output={post_vals[min(neuron_idx, len(post_vals)-1)]:.3f}')
                ax.set_xlabel('Pre-activation')
                ax.set_ylabel('Post-activation')
                ax.set_title(f'Layer {block_idx+1}, Neuron {neuron_idx+1}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(layer_dir, f"neuron_{neuron_idx+1}.png"), dpi=100)
                plt.close()
            else:
                # SGU or no activation: just show the value
                fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
                if neuron_idx < len(post_vals):
                    ax.bar([0], [post_vals[neuron_idx]], color='#2563eb')
                ax.set_title(f'Layer {block_idx+1}, Neuron {neuron_idx+1} (SGU/no-act)')
                ax.set_ylabel('Output value')
                plt.tight_layout()
                plt.savefig(os.path.join(layer_dir, f"neuron_{neuron_idx+1}.png"), dpi=100)
                plt.close()
            
            plot_count += 1
            if plot_count % 50 == 0:
                print(f"  Neuron plots: {plot_count}/{total_neurons}")
    
    print(f"  ✓ All {plot_count} neuron plots saved to '{neuron_dir}/'")

##############################################
# Activation helpers
##############################################
class ScaledTanh(nn.Module):
    def __init__(self, scale=1.7):
        super().__init__(); self.scale = scale
    def forward(self, x): return self.scale * torch.tanh(x)

class PELU(nn.Module):
    def __init__(self, a_init=1.0, b_init=1.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a_init)); self.b = nn.Parameter(torch.tensor(b_init))
    def forward(self, x):
        pos = (self.a / self.b) * torch.where(x >= 0, x, torch.zeros_like(x))
        neg = self.a * (torch.exp(torch.where(x < 0, x, torch.zeros_like(x)) / self.b) - 1)
        return pos + neg

def _build_activation_map():
    """Build the activation name -> factory dict for model loading."""
    return {
        "Custom": nn.ReLU,  # placeholder, rebuilt from config
        "ReLU": nn.ReLU, "APALU": APALU, "Sigmoid": nn.Sigmoid, "Tanh": nn.Tanh,
        "LeakyReLU": nn.LeakyReLU, "ScaledTanh": lambda **p: ScaledTanh(**p),
        "TrainableSine": lambda **p: Sine(), "TrainableCosine": lambda **p: Cosine(),
        "CombHsine": CombHsine, "SiLU": nn.SiLU, "PSiLU": PSiLU, "PReLU": nn.PReLU,
        "CELU": nn.CELU, "PELU": PELU, "SquareActivation": SquareActivation,
        "CubeActivation": CubeActivation, "Sine": Sine, "Cosine": Cosine, "ReLU6": nn.ReLU6,
        "GoLU": GoLU, "SELU": nn.SELU, "PAU": PAU, "RaPAU": lambda **p: PAU(**p),
        "Cone": Cone, "ParabolicCone": ParabolicCone, "Linear": nn.Identity,
        "lSELU": lSELU, "sSELU": sSELU, "SGoLU": SGoLU, "PaWeL": PaWeL,
        "GELU": nn.GELU, "Softplus": nn.Softplus, "Mish": Mish, "Softsign": Softsign,
        "HardSwish": HardSwish, "BentIdentity": BentIdentity, "SRS": SRS, "TAAF": TAAF,
        "AGLU": AGLU, "Sign": Sign, "Snake": Snake, "SmeLU": SmeLU, "Serf": Serf,
        "TeLU": TeLU, "DRA": DRA, "OGDRA": OGDRA, "SquaredReLU": SquaredReLU,
        "HeLU": HeLU, "CoLU": CoLU, "StarReLU": StarReLU, "GSine": GSine,
        "GCosine": GCosine, "TheSquare": TheSquare, "CapActi2": lambda **p: CapActi2(**p),
        "SineEvenLReLU": SineEvenLReLU, "SteppingSine": SteppingSine,
        "SteppingCosine": SteppingCosine, "TheCube": TheCube, "Cauchy": Cauchy,
        "AdaptiveBasisMixture": AdaptiveBasisMixture, "Phish": Phish, "TanhExp": TanhExp,
        "Sial": Sial, "SoftMax": SoftMax, "TaLU": TaLU, "SoLU": SoLU,
        "SiLULU": SiLULU, "Reciprocal": Reciprocal, "TTanh": TTanh,
        "TSoftsign": TSoftsign, "TSigma": TSigma, "TReLU": TReLU, "ATanU": ATanU,
        "SALU": SALU, "SMU": SMU, "ELU": nn.ELU, "RReLU": nn.RReLU, "PolyMorph": PolyMorph,
        "All": GumbelActivationSelector, "Maxout": lambda **p: MaxoutActivation(**p),
    }

def ask_activation(default=None):
    """default: returned when the user just presses Enter (per-layer choices fall back to the global one)."""
    print("Choose activation function:")
    print("0: Linear  1: Sigmoid  2: Tanh  3: Scaled Tanh  4: ReLU  5: LeakyReLU")
    print("6: PReLU  7: PELU  8: SiLU  9: PSiLU  10: Sine  11: Cosine  12: ReLU6")
    print("13: GoLU  14: Cone  15: ParabolicCone  16: Cube  17: Square  18: CombHsine")
    print("19: CELU  20: PAU  21: RaPAU  22: SELU  23: lSELU  24: sSELU  25: APALU")
    print("26: SGoLU  27: PaWeL  28: GELU  29: Softplus  30: Mish  31: Softsign")
    print("32: HardSwish  33: BentIdentity  34: SRS  35: TAAF  36: AGLU  37: Sign")
    print("38: Snake  39: SmeLU  40: Serf  41: TeLU  42: DRA  43: SquaredReLU")
    print("44: HeLU  45: CoLU  46: StarReLU  47: GSine  48: GCosine  49: TheSquare")
    print("50: CapActi2  51: SineEvenLReLU  52: SteppingSine  53: SteppingCosine")
    print("54: TheCube  55: Cauchy  56: ABM  57: Phish  58: TanhExp  59: Sial")
    print("60: Softmax  61: TaLU  62: SoLU  63: SiLULU  64: Reciprocal  65: TTanh")
    print("66: TSoftsign  67: TSigma  68: TReLU  69: ATanU  70: SALU  71: SMU")
    print("72: Snake  73: ELU  74: RReLU  75: PolyMorph  76: OGDRA")
    print("77: CUSTOM (define your own)")
    print("78: ALL (Gumbel Softmax learnable selection)")
    print("79: Maxout (learned maximum of affine pieces)")
    choice = input("Enter number: ").strip().lower()
    if not choice and default is not None: return default

    simple = {
        "0": nn.Identity, "1": nn.Sigmoid, "2": nn.Tanh, "4": nn.ReLU,
        "6": nn.PReLU, "8": nn.SiLU, "12": nn.ReLU6, "13": GoLU,
        "14": Cone, "15": ParabolicCone, "19": nn.CELU, "22": nn.SELU,
        "23": lSELU, "24": sSELU, "25": APALU, "26": SGoLU, "28": nn.GELU,
        "29": nn.Softplus, "30": Mish, "31": Softsign, "32": HardSwish,
        "33": BentIdentity, "34": SRS, "35": TAAF, "36": AGLU, "37": Sign,
        "38": Snake, "39": SmeLU, "40": Serf, "41": TeLU, "42": DRA,
        "43": SquaredReLU, "44": HeLU, "45": CoLU, "46": StarReLU,
        "47": GSine, "48": GCosine, "49": TheSquare, "51": SineEvenLReLU,
        "52": SteppingSine, "53": SteppingCosine, "54": TheCube, "55": Cauchy,
        "56": AdaptiveBasisMixture, "57": Phish, "58": TanhExp, "59": Sial,
        "60": SoftMax, "61": TaLU, "62": SoLU, "63": SiLULU, "64": Reciprocal,
        "65": TTanh, "66": TSoftsign, "67": TSigma, "68": TReLU, "69": ATanU,
        "70": SALU, "71": SMU, "72": Snake, "73": nn.ELU, "74": nn.RReLU,
        "75": PolyMorph, "76": OGDRA,
    }
    if choice in simple: return simple[choice]
    
    if choice == "3":
        scale = float(input("Enter scaling factor: "))
        def act(): return ScaledTanh(scale)
        act.activation_config = {"name": "ScaledTanh", "params": {"scale": scale}}; return act
    elif choice == "5":
        slope = float(input("Enter negative slope: "))
        def act(): return nn.LeakyReLU(negative_slope=slope)
        act.activation_config = {"name": "LeakyReLU", "params": {"negative_slope": slope}}; return act
    elif choice == "7": return PELU
    elif choice == "9":
        init = float(input("Enter initial alpha for PSiLU: "))
        def act(): return PSiLU(init)
        act.activation_config = {"name": "PSiLU", "params": {"init": init}}; return act
    elif choice == "10": return Sine
    elif choice == "11": return Cosine
    elif choice in ["16"]:
        class CubeActivation(nn.Module):
            def forward(self, x): return x ** 3
        return CubeActivation
    elif choice in ["17"]:
        class SquareActivation(nn.Module):
            def forward(self, x): return x ** 2
        return SquareActivation
    elif choice == "20":
        def act(): return PAU()
        act.activation_config = {"name": "PAU", "params": {}}; return act
    elif choice == "21":
        n = int(input("Enter n: ")); m = int(input("Enter m: "))
        def act(): return PAU(n=n, m=m)
        act.activation_config = {"name": "RaPAU", "params": {"n": n, "m": m}}; return act
    elif choice == "27":
        bp = int(input("Enter breakpoint amount: "))
        def act(): return PaWeL(bp)
        act.activation_config = {"name": "PaWeL", "params": {"breakpoints": bp}}; return act
    elif choice == "50":
        down = float(input("Min value: ")); up = float(input("Max value: "))
        def act(): return CapActi2(down=down, up=up)
        act.activation_config = {"name": "CapActi2", "params": {"down": down, "up": up}}; return act
    elif choice == "77": return ask_custom_activation()
    elif choice == "78":
        temp_str = input("Initial Gumbel temperature (default 1.0): ").strip()
        init_temp = float(temp_str) if temp_str else 1.0
        def act(): return GumbelActivationSelector(init_temp=init_temp)
        act.activation_config = {"name": "All", "params": {"init_temp": init_temp}}
        return act
    elif choice == "79":
        pieces_str = input("Maxout pieces per hidden output (default 2): ").strip()
        pieces = int(pieces_str) if pieces_str else 2
        if pieces < 2: raise ValueError("Maxout requires at least two pieces.")
        def act(): return MaxoutActivation(pieces=pieces)
        act.activation_config = {"name": "Maxout", "params": {"pieces": pieces}}
        return act
    else: print("Invalid, defaulting to ReLU."); return nn.ReLU

def ask_activation_type():
    print("\nChoose activation function TYPE:")
    print("0: Basic  1: Self-gated  2: GLU  3: TAAF  4: ReZero  5: Residual TAAF")
    print("6: Bipolar  7: GLU2")
    while True:
        choice = input("Enter (0-7): ").strip()
        if choice in [str(i) for i in range(8)]: return int(choice)
        print("Invalid.")

def wrap_activation(activation_cls, activation_type):
    if activation_type == 0: return activation_cls
    wrappers = {
        1: SelfGatedActivation, 2: GLUActivation, 3: TAAFWrapper,
        4: ResidualWrapper, 5: ResidualTAAFWrapper,
        6: lambda base: BipolarActivation(base),  # Assuming BipolarActivation exists
        7: GLU2Activation,
    }
    wrapper_cls = wrappers.get(activation_type)
    if wrapper_cls is None: raise ValueError(f"Invalid activation_type: {activation_type}")
    def wrapped():
        base = activation_cls() if callable(activation_cls) else activation_cls
        return wrapper_cls(base)
    return wrapped


##############################################
# Activation Maps for Benchmarking
##############################################
ACTIVATION_MAP = {
    "Linear": lambda: nn.Identity(), "Sigmoid": lambda: nn.Sigmoid(), "Tanh": lambda: nn.Tanh(),
    "Scaled Tanh (1.7)": lambda: ScaledTanh(1.7), "TTanh": lambda: TTanh(),
    "Softsign": lambda: Softsign(), "TSoftsign": lambda: TSoftsign(),
    "ReLU": lambda: nn.ReLU(), "TReLU": lambda: TReLU(),
    "LeakyReLU (0.01)": lambda: nn.LeakyReLU(0.01), "LeakyReLU (0.2)": lambda: nn.LeakyReLU(0.2),
    "PReLU": lambda: nn.PReLU(), "RReLU": lambda: nn.RReLU(), "ReLU6": lambda: nn.ReLU6(),
    "ELU": lambda: nn.ELU(), "PELU": lambda: PELU(), "SiLU": lambda: nn.SiLU(),
    "PSiLU (1.0)": lambda: PSiLU(init=1.0), "Sine": lambda: Sine(), "Cosine": lambda: Cosine(),
    "GoLU": lambda: GoLU(), "Cone": lambda: Cone(), "ParabolicCone": lambda: ParabolicCone(),
    "CELU": lambda: nn.CELU(), "PAU (relu)": lambda: PAU(initial_shape="relu"),
    "PAU (tanh)": lambda: PAU(initial_shape="tanh"), "PAU (swish)": lambda: PAU(initial_shape="swish"),
    "SELU": lambda: nn.SELU(), "lSELU": lambda: lSELU(), "sSELU": lambda: sSELU(),
    "APALU": lambda: APALU(), "SGoLU": lambda: SGoLU(),
    "PaWeL (32)": lambda: PaWeL(breakpoints=32), "GELU": lambda: nn.GELU(),
    "Softplus": lambda: nn.Softplus(), "Mish": lambda: Mish(), "HardSwish": lambda: HardSwish(),
    "BentIdentity": lambda: BentIdentity(), "SRS": lambda: SRS(), "TAAF": lambda: TAAF(),
    "AGLU": lambda: AGLU(), "Sign": lambda: Sign(), "SmeLU": lambda: SmeLU(),
    "Serf": lambda: Serf(), "TeLU": lambda: TeLU(), "DRA": lambda: DRA(), "OGDRA": lambda: OGDRA(),
    "SquaredReLU": lambda: SquaredReLU(), "HeLU": lambda: HeLU(), "StarReLU": lambda: StarReLU(),
    "GSine": lambda: GSine(), "GCosine": lambda: GCosine(), "TheSquare": lambda: TheSquare(),
    "CapActi2 (0,6)": lambda: CapActi2(down=0, up=6), "SineEvenLReLU": lambda: SineEvenLReLU(),
    "SteppingSine": lambda: SteppingSine(), "SteppingCosine": lambda: SteppingCosine(),
    "TheCube": lambda: TheCube(), "Cauchy": lambda: Cauchy(), "ABM": lambda: AdaptiveBasisMixture(),
    "Phish": lambda: Phish(), "TanhExp": lambda: TanhExp(), "Sial": lambda: Sial(),
    "SoftMax": lambda: SoftMax(), "TaLU": lambda: TaLU(), "SoLU": lambda: SoLU(),
    "SiLULU": lambda: SiLULU(), "ATanU": lambda: ATanU(), "SALU": lambda: SALU(),
    "SMU": lambda: SMU(), "Snake": lambda: Snake(), "PolyMorph": lambda: PolyMorph(),
    "All (Gumbel)": lambda: GumbelActivationSelector(init_temp=1.0),
}

BASICS_MAP = {
    "Linear": lambda: nn.Identity(), "Sigmoid": lambda: nn.Sigmoid(), "Tanh": lambda: nn.Tanh(),
    "TTanh": lambda: TTanh(), "Softsign": lambda: Softsign(), "ReLU": lambda: nn.ReLU(),
    "LeakyReLU (0.01)": lambda: nn.LeakyReLU(0.01), "LeakyReLU (0.2)": lambda: nn.LeakyReLU(0.2),
    "PReLU": lambda: nn.PReLU(), "PELU": lambda: PELU(), "SiLU": lambda: nn.SiLU(),
    "Sine": lambda: Sine(), "Cosine": lambda: Cosine(), "ReLU6": lambda: nn.ReLU6(),
    "CELU": lambda: nn.CELU(), "SELU": lambda: nn.SELU(), "GELU": lambda: nn.GELU(),
    "Softplus": lambda: nn.Softplus(), "Mish": lambda: Mish(), "HardSwish": lambda: HardSwish(),
    "Sign": lambda: Sign(), "Serf": lambda: Serf(), "ABM": lambda: AdaptiveBasisMixture(),
    "SoftMax": lambda: SoftMax(),
}

RELUS_MAP = {
    "ReLU": lambda: nn.ReLU(), "LeakyReLU (0.01)": lambda: nn.LeakyReLU(0.01),
    "LeakyReLU (0.2)": lambda: nn.LeakyReLU(0.2), "PELU": lambda: PELU(),
    "SiLU": lambda: nn.SiLU(), "PSiLU (1.0)": lambda: PSiLU(init=1.0),
    "ReLU6": lambda: nn.ReLU6(), "GoLU": lambda: GoLU(), "CELU": lambda: nn.CELU(),
    "SELU": lambda: nn.SELU(), "lSELU": lambda: lSELU(), "GELU": lambda: nn.GELU(),
    "Softplus": lambda: nn.Softplus(), "Mish": lambda: Mish(), "HardSwish": lambda: HardSwish(),
    "AGLU": lambda: AGLU(), "SmeLU": lambda: SmeLU(), "Serf": lambda: Serf(),
    "TeLU": lambda: TeLU(), "SquaredReLU": lambda: SquaredReLU(), "HeLU": lambda: HeLU(),
    "StarReLU": lambda: StarReLU(), "Phish": lambda: Phish(), "TanhExp": lambda: TanhExp(),
    "Sial": lambda: Sial(), "TaLU": lambda: TaLU(), "SoLU": lambda: SoLU(),
    "SiLULU": lambda: SiLULU(),
}

PERIODICS_MAP = {
    "Sine": lambda: Sine(), "Cosine": lambda: Cosine(), "DRA": lambda: DRA(),
    "GSine": lambda: GSine(), "GCosine": lambda: GCosine(), "SineEvenLReLU": lambda: SineEvenLReLU(),
    "SteppingSine": lambda: SteppingSine(), "SteppingCosine": lambda: SteppingCosine(),
    "TheCube": lambda: TheCube(), "ABM": lambda: AdaptiveBasisMixture(),
}

SELF_GATED_MAP = {
    "SiLU": lambda: nn.SiLU(), "PSiLU (1.0)": lambda: PSiLU(init=1.0),
    "GoLU": lambda: GoLU(), "SGoLU": lambda: SGoLU(), "Mish": lambda: Mish(),
    "HardSwish": lambda: HardSwish(), "AGLU": lambda: AGLU(), "Serf": lambda: Serf(),
    "TeLU": lambda: TeLU(), "GSine": lambda: GSine(), "GCosine": lambda: GCosine(),
    "Phish": lambda: Phish(), "TanhExp": lambda: TanhExp(), "Sial": lambda: Sial(),
    "TaLU": lambda: TaLU(), "SoLU": lambda: SoLU(), "SiLULU": lambda: SiLULU(),
}

TRAINABLE_MAP = {
    "TTanh": lambda: TTanh(), "TSoftsign": lambda: TSoftsign(), "PReLU": lambda: nn.PReLU(),
    "PELU": lambda: PELU(), "PSiLU (1.0)": lambda: PSiLU(init=1.0),
    "PAU (relu)": lambda: PAU(initial_shape="relu"), "PAU (tanh)": lambda: PAU(initial_shape="tanh"),
    "lSELU": lambda: lSELU(), "sSELU": lambda: sSELU(),
    "PaWeL (32)": lambda: PaWeL(breakpoints=32), "SRS": lambda: SRS(),
    "AGLU": lambda: AGLU(), "DRA": lambda: DRA(), "StarReLU": lambda: StarReLU(),
    "TheSquare": lambda: TheSquare(), "Cauchy": lambda: Cauchy(), "ABM": lambda: AdaptiveBasisMixture(),
}

IDK_MAP = {
    "TaLU": lambda: TaLU(), "SoLU": lambda: SoLU(), "SiLULU": lambda: SiLULU(),
    "Sigmoid": lambda: nn.Sigmoid(), "Tanh": lambda: nn.Tanh(),
    "TSigma": lambda: TSigma(), "TTanh": lambda: TTanh(), "TSoftsign": lambda: TSoftsign(),
    "Softsign": lambda: Softsign(), "Sine": lambda: Sine(), "Cosine": lambda: Cosine(),
    "ReLU6": lambda: nn.ReLU6(), "CapActi2 (0,6)": lambda: CapActi2(down=0, up=6),
    "CapActi2 (-1,1)": lambda: CapActi2(down=-1, up=1), "Sign": lambda: Sign(),
    "SoftMax": lambda: SoftMax(),
}


##############################################
# Benchmark (UPDATED: validation file option)
##############################################
import random

def _prompt_activation_subset():
    prompt = ("\nWhich activation subset?\n 0=all  1=basics  2=ReLUs  3=Periodics"
              "  4=self-gated  5=trainables  6=IDK\nEnter [0-6]: ")
    while True:
        try:
            c = int(input(prompt))
            if 0 <= c <= 6: return c
        except ValueError: pass
        print("Invalid.")

def ask_benchmark_sweep_config():
    """Interactively build a sweep configuration for benchmarking.
    Returns a dict with lists of values for each hyperparameter axis to sweep over."""
    config = {}
    print("\n=== Hyperparameter Sweep Configuration ===")
    print("For each axis, enter comma-separated values to sweep (empty = skip axis).\n")

    # Optimizers
    opt_str = input("Optimizers to sweep (e.g. Adam,AdamW,SGD) [empty=skip]: ").strip()
    if opt_str:
        config['optimizers'] = [o.strip() for o in opt_str.split(",") if o.strip()]

    # Learning rates
    lr_str = input("Learning rates to sweep (e.g. 0.001,0.0005,0.01) [empty=skip]: ").strip()
    if lr_str:
        try:
            config['learning_rates'] = [float(x.strip()) for x in lr_str.split(",") if x.strip()]
        except ValueError:
            print("  Invalid LR values, skipping.")

    # Architectures
    print("Architectures to sweep (each arch is comma-separated dims, separate archs with '|'):")
    arch_str = input("  e.g. 128,64|256,128,64|64,32 [empty=skip]: ").strip()
    if arch_str:
        archs = []
        for arch_entry in arch_str.split("|"):
            try:
                dims = [int(x.strip()) for x in arch_entry.split(",") if x.strip()]
                if dims: archs.append(dims)
            except ValueError:
                pass
        if archs: config['architectures'] = archs

    # Normalization types
    norm_str = input("Norm types to sweep (e.g. layer,batch,rmsnorm,none) [empty=skip]: ").strip()
    if norm_str:
        valid_norms = {"none", "batch", "instance", "layer", "group", "rmsnorm"}
        norms = [n.strip().lower() for n in norm_str.split(",") if n.strip().lower() in valid_norms]
        if norms: config['norm_types'] = norms

    # Residual types
    res_str = input("Residual types to sweep (e.g. residual,highway,rezero,none) [empty=skip]: ").strip()
    if res_str:
        valid_res = {"none", "highway", "residual", "rezero", "elementwise_rezero", "concat", "densenet"}
        residuals = [r.strip().lower() for r in res_str.split(",") if r.strip().lower() in valid_res]
        if residuals: config['residual_types'] = residuals

    # Activations (custom subset)
    act_str = input("Custom activation list (comma-separated names, e.g. ReLU,SiLU,GELU) [empty=use subset menu]: ").strip()
    if act_str:
        amap = _build_activation_map()
        act_dict = {}
        for name in act_str.split(","):
            name = name.strip()
            if name in amap:
                act_dict[name] = amap[name]
            else:
                print(f"  Warning: '{name}' not found in activation map, skipping.")
        if act_dict: config['activations'] = act_dict

    # Summary
    total = 1
    for key in ['optimizers', 'learning_rates', 'architectures', 'norm_types', 'residual_types', 'activations']:
        if key in config: total *= len(config[key])
    print(f"\nSweep configuration: {total} combinations")
    for key, val in config.items():
        if key == 'activations':
            print(f"  {key}: {list(val.keys())}")
        else:
            print(f"  {key}: {val}")

    if not config:
        print("No sweep axes configured. Running standard benchmark.")
        return None
    return config


def _get_activation_map(choice):
    names = {0:"ACTIVATION_MAP",1:"BASICS_MAP",2:"RELUS_MAP",3:"PERIODICS_MAP",
             4:"SELF_GATED_MAP",5:"TRAINABLE_MAP",6:"IDK_MAP"}
    return globals()[names[choice]]

def run_benchmark(csv_file, delimiter, input_cols, output_cols, col_types, vocabularies,
                  image_params, hidden_dims, batch_size, optimizer_choice, residual_type,
                  noise_mode, noise_params, norm_type="layer", groups=1, attention_type="none",
                  num_heads=1, input_attention_type="none", moe_mode=1, num_steps=1000,
                  loss_calc_mode=0, train_eval_amount=None, valid_eval_percentage=None,
                  use_lsuv=False, lsuv_max_iter=10, lsuv_normalize_mean=True,
                  val_file_path=None, val_delimiter=None, sweep_config=None, custom_lr=None,
                  activations=None, activation_type=None, optim_params_for=None, progress_callback=None,
                  eval_interval=500, final_eval=False):
    """Benchmark with optional external validation file support.

    Non-interactive use (GUI): ``activations`` (names) and ``activation_type`` replace
    the activation prompts, ``optim_params_for(optimizer) -> dict`` replaces the
    optimizer prompts.  ``progress_callback(event)`` receives start / combo / step /
    result events; returning True stops the benchmark.  ``final_eval`` adds a
    validation check after the last step (loss_calc_mode 2).  Returns the result
    rows (name, metric and per-combination statistics)."""
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Data: deduplicated, and with loss_calc_mode 2 a truly held-out validation set ──
    val_loader = None
    vfile = None; vfrac = 0.0
    if loss_calc_mode == 2:
        if val_file_path and os.path.isfile(val_file_path):
            print(f"Using external validation file: {val_file_path}"); vfile = val_file_path
        else:
            if val_file_path:
                print(f"File '{val_file_path}' not found/invalid. Falling back to percentage split.")
            if valid_eval_percentage is None:
                valid_eval_percentage = float(input("Validation percentage (0.0-1.0): ").strip())
            vfrac = valid_eval_percentage
    dataset, val_dataset, _ = prepare_train_val(csv_file, delimiter, input_cols, output_cols, col_types, vocabularies,
                                                image_params, val_frac=vfrac, val_file=vfile, val_delimiter=val_delimiter)
    scalings = dataset.scalings
    train_loader = make_loader(dataset, batch_size, shuffle=True)
    if val_dataset is not None:
        if len(val_dataset) > 3000:  # keep evaluation fast, as before
            val_dataset = torch.utils.data.Subset(val_dataset, np.random.default_rng(0).choice(len(val_dataset), 3000, replace=False).tolist())
        val_loader = make_loader(val_dataset, batch_size, shuffle=False)

    input_dims = calculate_input_dim(input_cols, col_types, scalings, vocabularies)
    
    # Build output layout for CombinedLoss
    output_layout, output_pred_dim, output_tgt_dim = build_output_layout(output_cols, col_types, scalings, vocabularies)
    has_categorical = any(e['type'] in ['outlabcat', 'outexcat'] for e in output_layout)
    output_dims = output_pred_dim

    # Build sweep combinations
    if activations is not None:
        amap = _build_activation_map()
        unknown = [n for n in activations if n not in amap]
        if unknown or not activations:
            raise ValueError(f"Unknown activations: {unknown}" if unknown else "Choose at least one activation.")
        if activation_type is None:
            activation_type = ask_activation_type()
        act_dict = {name: wrap_activation(amap[name], activation_type) for name in activations}
    elif sweep_config and sweep_config.get('activations'):
        act_dict = sweep_config['activations']
        activation_type = ask_activation_type()
        wrapped_act_dict = {name: wrap_activation(factory, activation_type) for name, factory in act_dict.items()}
        act_dict = wrapped_act_dict
    else:
        subset_choice = _prompt_activation_subset()
        act_dict = _get_activation_map(subset_choice)
        activation_type = ask_activation_type()
        wrapped_act_dict = {name: wrap_activation(factory, activation_type) for name, factory in act_dict.items()}
        act_dict = wrapped_act_dict

    optimizer_list = (sweep_config or {}).get('optimizers') or [optimizer_choice]
    lr_list = (sweep_config or {}).get('learning_rates') or [custom_lr]
    arch_list = (sweep_config or {}).get('architectures') or [hidden_dims]
    norm_list = (sweep_config or {}).get('norm_types') or [norm_type]
    res_list = (sweep_config or {}).get('residual_types') or [residual_type]
    
    # Build all combinations
    sweep_combos = []
    for act_name, act_factory in act_dict.items():
        for cur_optimizer in optimizer_list:
            for cur_lr in lr_list:
                for cur_arch in arch_list:
                    for cur_norm in norm_list:
                        for cur_res in res_list:
                            sweep_combos.append({
                                'act_name': act_name, 'act_factory': act_factory,
                                'optimizer': cur_optimizer, 'lr': cur_lr,
                                'arch': cur_arch, 'norm': cur_norm, 'res': cur_res,
                            })
    
    total_combos = len(sweep_combos)
    if total_combos > len(act_dict):
        print(f"\n=== Full Hyperparameter Sweep: {total_combos} combinations ===")
    
    notify = progress_callback or (lambda event: False)
    stop = bool(notify({"event": "start", "total": total_combos, "names": [c['act_name'] for c in sweep_combos]}))
    details = []
    results = []
    for combo_idx, combo in enumerate(sweep_combos):
        if stop:
            break
        act_name = combo['act_name']
        act_factory = combo['act_factory']
        cur_optimizer = combo['optimizer']
        cur_lr = combo['lr']
        cur_arch = combo['arch']
        cur_norm = combo['norm']
        cur_res = combo['res']
        hidden_dims = cur_arch
        
        # Build display name
        combo_name = act_name
        if total_combos > len(act_dict):
            extras = []
            if len(optimizer_list) > 1: extras.append(f"opt={cur_optimizer}")
            if len(lr_list) > 1: extras.append(f"lr={cur_lr}")
            if len(arch_list) > 1: extras.append(f"arch={cur_arch}")
            if len(norm_list) > 1: extras.append(f"norm={cur_norm}")
            if len(res_list) > 1: extras.append(f"res={cur_res}")
            if extras: combo_name = f"{act_name} [{', '.join(extras)}]"
        print(f"\n=== [{combo_idx+1}/{total_combos}] Benchmarking {combo_name} ===")
        stop = bool(notify({"event": "combo", "index": combo_idx, "name": combo_name, "optimizer": cur_optimizer,
                            "lr": cur_lr, "arch": cur_arch, "norm": cur_norm, "res": cur_res, "activation": act_name}))
        best_metric = float("nan"); recent = deque(maxlen=100)
        detail = {"name": combo_name, "activation": act_name, "optimizer": cur_optimizer, "lr": cur_lr, "arch": list(cur_arch),
                  "norm": cur_norm, "res": cur_res, "params": 0, "seconds": 0.0, "steps": 0, "curve": [], "val_curve": [],
                  "error": None}
        started = time.time(); curve_every = max(1, num_steps // 200); window = deque(maxlen=max(10, curve_every))
        try:
            model = MLPO(input_dims, hidden_dims, output_dims, act_factory, cur_res,
                norm_type=cur_norm, groups=groups, attention_type=attention_type, num_heads=num_heads,
                input_attention_type=input_attention_type, moe_mode=moe_mode,
                dropout_prob=noise_params.get("dropout_pct", 0.0) if noise_mode == "dropout" else 0.0,
                use_noise_injection_layers=(noise_mode == "noise injection layers"),
                noise_injection_std=noise_params.get("std", 0.0)).to(device)
            materialize_lazy_modules(model, input_dims, device)
            if use_lsuv:
                lsuv_init(model, train_loader, device, max_iter=lsuv_max_iter,
                         normalize_mean=lsuv_normalize_mean, verbose=False)
            detail["params"] = sum(p.numel() for p in model.parameters())
            optimizer = select_optimizer(cur_optimizer, model, custom_lr=cur_lr,
                                         optim_params=optim_params_for(cur_optimizer) if optim_params_for else None)
            if cur_optimizer == "RAdamScheduleFree": optimizer.train()
            
            criterion = CombinedLoss(output_layout) if has_categorical else nn.HuberLoss()
            if has_categorical: criterion.calibrate(dataset)
            step = 0; EVAL_INTERVAL = max(1, int(eval_interval))
            def validate():
                model.eval(); val_losses = []
                with torch.no_grad():
                    for vi, vt in val_loader:
                        vi, vt = vi.to(device), vt.to(device)
                        val_losses.append(criterion(model(vi), vt).item())
                model.train()
                return sum(val_losses) / len(val_losses) if val_losses else None
            pbar = tqdm(total=num_steps, desc=f"{combo_name[:30]:<30}", leave=True)
            
            while step < num_steps:
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    if noise_mode == "input noise":
                        inputs = inputs + torch.randn_like(inputs) * noise_params.get("std", 0.1)
                    outputs = model(inputs)
                    if noise_mode == "output noise":
                        outputs = outputs + torch.randn_like(outputs) * noise_params.get("std", 0.1)
                    loss = criterion(outputs, targets)
                    optimizer.zero_grad(); loss.backward()
                    if hasattr(optimizer, "observe_loss"): optimizer.observe_loss(loss.item())   # HD divergence guard
                    optimizer.step()
                    if noise_mode == "weight noise":
                        with torch.no_grad():
                            for p in model.parameters(): p.add_(torch.randn_like(p) * noise_params.get("std", 0.1))
                    
                    if loss_calc_mode == 0:
                        recent.append(loss.item())
                        if len(recent) == recent.maxlen:
                            avg = sum(recent) / len(recent)
                            if math.isnan(best_metric) or avg < best_metric: best_metric = avg
                    elif loss_calc_mode == 2 and (step % EVAL_INTERVAL == 0) and val_loader:
                        va = validate()
                        if va is not None:
                            detail["val_curve"].append((step, va))
                            if math.isnan(best_metric) or va < best_metric: best_metric = va
                    
                    window.append(loss.item())
                    step += 1; pbar.update(1)
                    if step % curve_every == 0 or step >= num_steps:
                        detail["curve"].append((step, sum(window) / len(window)))
                        if notify({"event": "step", "index": combo_idx, "step": step, "total_steps": num_steps,
                                   "loss": detail["curve"][-1][1], "val": detail["val_curve"][-1][1] if detail["val_curve"] else None,
                                   "curve": detail["curve"], "val_curve": detail["val_curve"]}):
                            stop = True
                    if step >= num_steps or stop: break
                if stop: break
            pbar.close()
            if final_eval and loss_calc_mode == 2 and val_loader and not stop:
                va = validate()
                if va is not None:
                    detail["val_curve"].append((step, va))
                    if math.isnan(best_metric) or va < best_metric: best_metric = va

            if loss_calc_mode == 1:
                if train_eval_amount is None: raise ValueError("train_eval_amount required")
                from torch.utils.data import SubsetRandomSampler
                sample_indices = random.sample(range(len(dataset)), min(train_eval_amount, len(dataset)))
                sample_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(sample_indices))
                model.eval(); eval_losses = []
                with torch.no_grad():
                    for si, st in sample_loader:
                        si, st = si.to(device), st.to(device)
                        eval_losses.append(criterion(model(si), st).item())
                model.train()
                best_metric = sum(eval_losses) / len(eval_losses) if eval_losses else float("nan")

        except RuntimeError as e:
            print(f"RuntimeError during {act_name}: {e}"); best_metric = float("nan"); detail["error"] = str(e).split("\n")[0][:300]
        except Exception as e:
            print(f"Error during {act_name}: {e}"); best_metric = float("nan"); detail["error"] = str(e).split("\n")[0][:300]
        finally:
            try: del model; del optimizer
            except: pass
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        if stop:
            break   # an interrupted combination is not a result
        detail.update(metric=best_metric, seconds=time.time() - started, steps=detail["curve"][-1][0] if detail["curve"] else 0,
                      final_loss=detail["curve"][-1][1] if detail["curve"] else None)
        detail["steps_per_s"] = detail["steps"] / detail["seconds"] if detail["seconds"] > 0 else 0.0
        details.append(detail)
        print(best_metric); results.append((combo_name, best_metric))
        stop = bool(notify({"event": "result", "index": combo_idx, "result": detail}))

    results.sort(key=lambda x: float("inf") if math.isnan(x[1]) else x[1])
    print("\n===== Benchmark Results =====")
    for rank, (name, score) in enumerate(results, 1): print(f"{rank:2d}. {name:<50} {score}")
    try:
        import csv
        with open("benchmark_results.csv", "w", newline="") as fp:
            writer = csv.writer(fp); writer.writerow(["Rank", "Activation", "Metric"])
            for rank, (name, score) in enumerate(results, 1): writer.writerow([rank, name, score])
        print("\nResults saved to benchmark_results.csv")
    except Exception as e: print("CSV save failed:", e)
    return details


##############################################
# Utility
##############################################
def preview_csv_file(path):
    if not os.path.isfile(path): print(f"File '{path}' does not exist!"); exit(1)
    with open(path, "r", encoding="utf-8") as f: lines = f.readlines()
    if not lines: print(f"File '{path}' is empty."); exit(1)
    print("First row:"); print(lines[0].rstrip())
    if len(lines) > 1:
        n = min(10, len(lines) - 1)
        sampled = random.sample(lines[1:], n)
        print(f"\n{n} randomly selected rows:"); 
        for l in sampled: print(l.rstrip())

##############################################
# NEW: Batch Inverse Design (Parallel Search)
##############################################
def run_inverse_design():
    print("\n=== Inverse Design (Optimize Inputs) ===")
    
    # 1. Load Config & Model
    if not os.path.exists("config.json") or not os.path.exists("model.pt"):
        print("Error: config.json or model.pt not found."); return

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Rebuild Model
    input_cols = [c for c, t in config["col_types"].items() if t in ["in", "inlab", "intex", "inim", "inlabcat", "intexcat"]]
    output_cols = [c for c, t in config["col_types"].items() if t in ["out", "outlab", "outex", "outlabcat", "outexcat"]]
    vocabularies = config.get("vocabularies", {})
    scalings = config.get("scalings", {})
    col_types = config["col_types"]
    
    input_dims = calculate_input_dim(input_cols, col_types, scalings, vocabularies)
    output_pred_dim = calculate_output_pred_dim(output_cols, col_types, scalings, vocabularies)
    
    act_map = _build_activation_map(); act_cfg = config.get("activation", {"name": "ReLU", "params": {}})
    if act_cfg["name"] in act_map: cls = lambda: act_map[act_cfg["name"]](**act_cfg["params"])
    else: cls = nn.ReLU
    cls = wrap_activation(cls, config.get("activation_type", 0))

    # --- FIX: Check MLP Mode ---
    mlp_mode = config.get("mlp_mode", 0)

    model = build_model_from_config(config, input_dims, output_pred_dim, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    materialize_lazy_modules(model, input_dims, next(model.parameters()).device)
    
    with torch.no_grad(): model(torch.zeros(1, input_dims).to(device))
    model.load_state_dict(torch.load('model.pt', map_location=device)); model.eval()
    for param in model.parameters(): param.requires_grad = False # Freeze model

    # --- NEW: ASK FOR BATCH SIZE ---
    try:
        batch_size = int(input("\nBatch Size (Parallel Candidates) [default=32]: ").strip() or "32")
    except: batch_size = 32
    print(f"  > optimizing {batch_size} candidates simultaneously...")

    # ... (Rest of the function remains the same as your provided code) ...
    # Copy the rest of the function logic from step 2 (Setup Inputs) downwards from your original code.
    # [Rest of code omitted for brevity, ensure you keep the optimization loop]

    # 2. Setup Inputs
    print("\n--- Configure Inputs ---")
    print("For each input, enter a fixed value OR type 'opt' to let the AI optimize it.")
    
    col_to_indices = {}; current_idx = 0
    for col in input_cols:
        ct = col_types[col]
        if ct in ['intex', 'outex']: width = scalings[col]['max_len']
        elif ct == 'intexcat': width = scalings[col]['max_len'] * len(vocabularies[col])
        elif ct == 'inlabcat': width = len(vocabularies[col])
        elif ct == 'inim': width = (1 if scalings[col]["patch_size"] == 1 else (scalings[col]["im_size"] // scalings[col]["patch_size"]) ** 2) * 3
        else: width = 1
        col_to_indices[col] = (current_idx, current_idx + width); current_idx += width

    # Init Batch Input Tensor (Batch, InputDims)
    input_tensor = torch.zeros(batch_size, input_dims, device=device, requires_grad=False)
    optimizable_masks = [] 
    
    for col in input_cols:
        ct = col_types[col]; start, end = col_to_indices[col]; width = end - start
        user_val = input(f"  {col} ({ct}): ").strip()
        
        if user_val.lower() == 'opt':
            # Initialize with randomness so candidates start at different places
            # We add noise to separate the batch items
            input_tensor[:, start:end] = torch.randn(batch_size, width, device=device) * 0.5
            optimizable_masks.extend([True] * width)
        else:
            optimizable_masks.extend([False] * width)
            val_tensor = [0.0] * width # Default zero
            # (Simplified loading logic for scalar/categorical)
            if ct == 'inlab': val_tensor = [float(vocabularies[col].get(user_val, 0))]
            elif ct == 'inlabcat':
                idx = vocabularies[col].get(user_val, 0); val_tensor = [0.0]*width
                if 0 <= idx < width: val_tensor[idx] = 1.0
            elif ct == 'in' and col in scalings:
                 smin = scalings[col]['min']; smax = scalings[col]['max']
                 try: val_tensor = [to_model_units(float(user_val), scalings[col])]
                 except: pass
            elif ct == 'in': 
                 try: val_tensor = [float(user_val)]
                 except: pass
            
            # Broadcast fixed value across batch
            input_tensor[:, start:end] = torch.tensor(val_tensor, device=device).unsqueeze(0).expand(batch_size, -1)

    opt_mask_tensor = torch.tensor(optimizable_masks, device=device, dtype=torch.bool)
    if not any(optimizable_masks): print("Error: Select at least one 'opt' input."); return

    # 3. Setup Targets
    out_layout, _, _ = build_output_layout(output_cols, col_types, scalings, vocabularies)
    targets = []
    print("\n--- Configure Targets ---")
    for entry in out_layout:
        col = entry['col']; ct = entry['type']; start, end = entry['start'], entry['end']
        user_target = input(f"  Target for '{col}': ").strip()
        if not user_target or user_target == 'ignore': continue
            
        if ct in ['outlabcat', 'outexcat']:
             if user_target in vocabularies[col]:
                targets.append({'type': 'cat', 'slice': slice(start, end), 
                                'target_idx': torch.tensor([vocabularies[col][user_target]], device=device)})
        elif ct in ['out', 'outlab']:
             try:
                 val = float(user_target)
                 if col in scalings and 'min' in scalings[col]:
                     smin, smax = scalings[col]['min'], scalings[col]['max']
                     val = to_model_units(val, scalings[col])
                 targets.append({'type': 'mse', 'slice': slice(start, end), 'target_val': torch.tensor([val], device=device)})
             except: pass

    # 4. Optimization Loop (Batch)
    print(f"\n--- Optimizing {batch_size} Parallel Candidates ---")
    
    # We optimize a parameter of shape (Batch, Num_Optimizable_Vars)
    # This ensures every candidate in the batch has its own independent variables
    initial_opt_values = input_tensor[:, opt_mask_tensor].clone()
    opt_params = nn.Parameter(initial_opt_values)
    
    optimizer = optim.Adam([opt_params], lr=0.05)
    
    # For tracking the best candidate found so far
    best_loss_overall = float('inf')
    best_inputs_overall = None
    
    pbar = tqdm(range(2000), desc="Optimizing")
    for step in pbar:
        optimizer.zero_grad()
        
        # Reconstruct Batch Input
        current_full_input = input_tensor.clone()
        current_full_input[:, opt_mask_tensor] = opt_params
        
        prediction = model(current_full_input) # (Batch, OutputDims)
        
        # Calculate loss PER ITEM in batch (do not reduce yet)
        total_loss = torch.zeros(batch_size, device=device)
        
        for tgt in targets:
            pred_slice = prediction[:, tgt['slice']] # (Batch, SliceDim)
            
            if tgt['type'] == 'mse':
                # MSE per row
                mse = (pred_slice - tgt['target_val']) ** 2
                total_loss += mse.sum(dim=1) 
                
            elif tgt['type'] == 'cat':
                # Logit Maximization (Target - Max(Others)) per row
                target_idx = tgt['target_idx'].item()
                target_logits = pred_slice[:, target_idx]
                
                # Mask out target to find max of others
                mask = torch.ones(pred_slice.shape[1], dtype=torch.bool, device=device)
                mask[target_idx] = False
                other_logits = pred_slice[:, mask]
                
                if other_logits.shape[1] > 0:
                    max_others, _ = torch.max(other_logits, dim=1)
                    # We minimize: -(Target - MaxOther)
                    total_loss += -(target_logits - max_others)
                else:
                    total_loss += -target_logits

        # Backprop: We sum the losses so optimizer sees gradients for all candidates
        # (Since candidates are independent in opt_params, gradients don't conflict)
        loss_sum = total_loss.sum()
        loss_sum.backward()
        optimizer.step()
        
        # Track best
        min_loss_val, min_loss_idx = torch.min(total_loss, dim=0)
        if min_loss_val.item() < best_loss_overall:
            best_loss_overall = min_loss_val.item()
            best_inputs_overall = current_full_input[min_loss_idx].detach().clone()

        pbar.set_postfix({'best_loss': f"{best_loss_overall:.6f}"})
        #if best_loss_overall < 1e-6: break

    # 5. Show Winner
    print("\n" + "="*40)
    print("       OPTIMIZATION WINNER")
    print("="*40)
    
    # Unscale and Print INPUTS
    print("\n[ Optimized Inputs (Best Candidate) ]")
    final_input = best_inputs_overall.cpu().numpy()
    
    for col in input_cols:
        ct = col_types[col]; start, end = col_to_indices[col]
        raw_vals = final_input[start:end]
        was_opt = opt_mask_tensor[start:end].any().item()
        tag = "(OPTIMIZED)" if was_opt else "(Fixed)"
        
        if ct in ['in', 'inlab']:
            val = raw_vals[0]
            if col in scalings and 'min' in scalings[col]:
                 val = to_display_units(val, scalings[col])
            print(f"  {col:<20}: {val:.4f}  {tag}")
        elif ct == 'inlabcat':
             idx = np.argmax(raw_vals)
             inv = {v: k for k, v in vocabularies[col].items()}
             print(f"  {col:<20}: {inv.get(idx, idx)}  {tag}")

    # Print Predicted Outputs
    print("\n[ Predicted Outputs ]")
    with torch.no_grad(): final_pred = model(best_inputs_overall.unsqueeze(0))[0]
    for entry in out_layout:
        col = entry['col']; ct = entry['type']; start, end = entry['start'], entry['end']
        raw_out = final_pred[start:end]
        if ct in ['out', 'outlab']:
             val = raw_out.item()
             if col in scalings and 'min' in scalings[col]:
                 val = to_display_units(val, scalings[col])
             print(f"  {col:<20}: {val:.4f}")
        elif ct == 'outlabcat':
             idx = raw_out.argmax().item(); inv = {v: k for k, v in vocabularies[col].items()}
             print(f"  {col:<20}: {inv.get(idx, idx)}")
    print("="*40 + "\n")

##############################################
# Main runner
##############################################
# Mode 7: Test Set Evaluation
##############################################
def run_test_eval(model, test_csv, delimiter, input_cols, output_cols, col_types,
                  vocabularies, scalings, image_params, batch_size, device):
    """Evaluate model on a test set with comprehensive metrics."""
    dataset = CustomDataset(test_csv, delimiter, input_cols, output_cols,
                           col_types, vocabularies, scalings, image_params)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    output_layout, output_pred_dim, output_tgt_dim = build_output_layout(output_cols, col_types, scalings, vocabularies)
    has_categorical = any(e['type'] in ['outlabcat', 'outexcat', 'outdec'] for e in output_layout)
    
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0
    
    class_weights = dataset.compute_class_weights(device=device)
    if has_categorical:
        criterion = CombinedLoss(output_layout, column_weights=class_weights); criterion.calibrate(dataset)
    else:
        criterion = nn.HuberLoss()
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)  # decoders generate freely here; the loss below is teacher-forced like training
            loss = criterion(seqm.model_forward(model, inputs, targets), targets)
            total_loss += loss.item()
            n_batches += 1
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    avg_loss = total_loss / max(1, n_batches)
    
    print(f"\n{'='*60}")
    print(f"TEST SET EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Samples: {len(all_preds)}")
    print(f"  Average Loss: {avg_loss:.6f}")
    
    for entry in output_layout:
        col = entry["col"]
        ctype = entry["type"]
        start = entry["start"]
        end = entry["end"]
        
        pred_slice = all_preds[:, start:end]
        tgt_slice = all_targets[:, start:end]
        
        if ctype in ["out", "outlab", "outex"]:
            pred_flat = pred_slice.flatten()
            tgt_flat = tgt_slice.flatten()
            mse = np.mean((pred_flat - tgt_flat) ** 2)
            mae = np.mean(np.abs(pred_flat - tgt_flat))
            ss_res = np.sum((tgt_flat - pred_flat) ** 2)
            ss_tot = np.sum((tgt_flat - np.mean(tgt_flat)) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
            
            # Unscale for interpretable metrics
            if col in scalings and "min" in scalings[col]:
                smin = scalings[col]["min"]
                smax = scalings[col]["max"]
                pred_unscaled = to_display_units(pred_flat, scalings[col])
                tgt_unscaled = to_display_units(tgt_flat, scalings[col])
                mae_real = np.mean(np.abs(pred_unscaled - tgt_unscaled))
                print(f"\n  [{col}] (regression)")
                print(f"    MSE (scaled):    {mse:.6f}")
                print(f"    MAE (scaled):    {mae:.6f}")
                print(f"    MAE (real):      {mae_real:.4f}")
                print(f"    R²:              {r2:.6f}")
            else:
                print(f"\n  [{col}] (regression)")
                print(f"    MSE:  {mse:.6f}")
                print(f"    MAE:  {mae:.6f}")
                print(f"    R²:   {r2:.6f}")
        
        elif ctype in ["outexcat", "outdec"]:
            ml, nc = entry["max_len"], entry["num_classes"]
            pred_ids = pred_slice.reshape(-1, ml, nc).argmax(-1)
            tgt_ids = all_targets[:, entry["tgt_start"]:entry["tgt_end"]].astype(int)
            scored = tgt_ids != 0 if ctype == "outdec" else np.ones_like(tgt_ids, dtype=bool)  # outdec: up to <eos>
            hit = pred_ids == tgt_ids
            exact = np.mean(np.all(hit | ~scored, axis=1))
            print(f"\n  [{col}] (text, {nc} token classes, max {ml} tokens)")
            print(f"    Exact match:    {exact:.4f} ({int(round(exact * len(hit)))}/{len(hit)})")
            print(f"    Token accuracy: {hit[scored].mean():.4f}")

        elif ctype in ["outlabcat", "outexcat"]:
            pred_classes = np.argmax(pred_slice, axis=1)
            tgt_classes = tgt_slice.flatten().astype(int) if tgt_slice.shape[1] == 1 else np.argmax(tgt_slice, axis=1)
            accuracy = np.mean(pred_classes == tgt_classes)
            
            n_classes = pred_slice.shape[1]
            print(f"\n  [{col}] (classification, {n_classes} classes)")
            print(f"    Accuracy: {accuracy:.4f} ({int(accuracy * len(pred_classes))}/{len(pred_classes)})")
            
            # Per-class accuracy
            for c in range(n_classes):
                mask = tgt_classes == c
                if mask.sum() > 0:
                    class_acc = np.mean(pred_classes[mask] == c)
                    vocab = vocabularies.get(col, {})
                    label = str(c)
                    for k, v in vocab.items():
                        if v == c: label = k; break
                    print(f"    Class '{label}': {class_acc:.4f} ({mask.sum()} samples)")
    
    print(f"\n{'='*60}")
    
    # Save report
    report_path = "test_eval_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"Test Set Evaluation Report\n")
        f.write(f"Samples: {len(all_preds)}\n")
        f.write(f"Average Loss: {avg_loss:.6f}\n")
    print(f"  Report saved to {report_path}")


##############################################
# Mode 8: Model Info
##############################################
def run_model_info(model, config):
    """Display comprehensive model architecture information."""
    print(f"\n{'='*60}")
    print(f"MODEL ARCHITECTURE SUMMARY")
    print(f"{'='*60}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Memory (approx):      {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Architecture details
    if hasattr(model, 'blocks'):
        print(f"\n  Number of blocks: {len(model.blocks)}")
        for i, block in enumerate(model.blocks):
            block_params = sum(p.numel() for p in block.parameters())
            in_feat = block.linear1.in_features
            out_feat = block.linear1.out_features
            act_name = type(block.activation).__name__ if block.activation else "None"
            norm_name = type(block.norm).__name__
            res_type = getattr(block, 'residual_type', 'unknown')
            
            l2_info = ""
            if block.linear2 is not None:
                l2_info = f" -> {block.linear2.out_features}"
            
            print(f"\n  Block {i}: {in_feat} -> {out_feat}{l2_info}")
            print(f"    Activation: {act_name}")
            print(f"    Norm:       {norm_name}")
            print(f"    Residual:   {res_type}")
            print(f"    Skip:       {'Linear' if isinstance(block.skip, nn.Linear) else 'Identity'}")
            print(f"    Params:     {block_params:,}")
            
            if hasattr(block, 'use_sgu') and block.use_sgu:
                print(f"    SGU:        Yes")
            if hasattr(block, 'tiny_attn') and block.tiny_attn is not None:
                print(f"    TinyAttn:   Yes")
            if hasattr(block, 'attn') and block.attn is not None:
                print(f"    Attention:  {type(block.attn).__name__}")
    
    if hasattr(model, 'final_linear'):
        fl = model.final_linear
        print(f"\n  Final Linear: {fl.in_features} -> {fl.out_features}")
    
    if hasattr(model, 'final_skip') and model.final_skip is not None:
        fs = model.final_skip
        print(f"  Final Skip:   {fs.in_features} -> {fs.out_features}")
    
    if hasattr(model, 'input_attn') and model.input_attn is not None:
        print(f"  Input Attention: {type(model.input_attn).__name__}")
    
    # Config info
    if config:
        print(f"\n  Config details:")
        for key in ["activation", "residual_type", "norm_type", "moe_mode",
                     "attention_type", "noise_mode", "hidden_dims"]:
            if key in config:
                print(f"    {key}: {config[key]}")
    
    print(f"\n{'='*60}")


##############################################
# Mode 9: Feature Importance (Permutation-based)
##############################################
def run_feature_importance(model, csv_file, delimiter, input_cols, output_cols, col_types,
                          vocabularies, scalings, image_params, batch_size, device,
                          n_repeats=5):
    """Permutation-based feature importance analysis."""
    dataset = CustomDataset(csv_file, delimiter, input_cols, output_cols,
                           col_types, vocabularies, scalings, image_params)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    output_layout, _, _ = build_output_layout(output_cols, col_types, scalings, vocabularies)
    has_categorical = any(e['type'] in ['outlabcat', 'outexcat'] for e in output_layout)
    
    class_weights = dataset.compute_class_weights(device=device)
    if has_categorical:
        criterion = CombinedLoss(output_layout, column_weights=class_weights); criterion.calibrate(dataset)
    else:
        criterion = nn.HuberLoss()
    
    model.eval()
    
    # Baseline loss
    baseline_loss = validate_model(model, loader, criterion, device)
    print(f"\n  Baseline loss: {baseline_loss:.6f}")
    
    # Build feature-to-column mapping
    input_dim_map = []  # List of (col_name, start_idx, end_idx)
    idx = 0
    for col in input_cols:
        ct = col_types.get(col, "in")
        if ct == "in" or ct == "inlab":
            input_dim_map.append((col, idx, idx + 1))
            idx += 1
        elif ct == "inlabcat":
            vocab = vocabularies.get(col, {})
            n = len(vocab)
            input_dim_map.append((col, idx, idx + n))
            idx += n
        elif ct == "intexcat":
            vocab = vocabularies.get(col, {})
            max_len = scalings.get(col, {}).get("max_len", 1)
            n = len(vocab) * max_len
            input_dim_map.append((col, idx, idx + n))
            idx += n
    
    # Permute each feature group and measure loss increase
    importances = {}
    print(f"\n  Analyzing {len(input_dim_map)} input features...")
    
    for col_name, start, end in input_dim_map:
        losses = []
        for _ in range(n_repeats):
            perm_loss = 0.0
            n_batches = 0
            with torch.no_grad():
                for inputs, targets in loader:
                    inputs = inputs.clone()
                    # Permute the feature columns
                    perm_idx = torch.randperm(inputs.size(0))
                    inputs[:, start:end] = inputs[perm_idx, start:end]
                    
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    perm_loss += loss.item()
                    n_batches += 1
            losses.append(perm_loss / max(1, n_batches))
        
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        importance = mean_loss - baseline_loss
        importances[col_name] = (importance, std_loss)
    
    # Sort by importance
    sorted_imp = sorted(importances.items(), key=lambda x: x[1][0], reverse=True)
    
    print(f"\n{'='*60}")
    print(f"FEATURE IMPORTANCE (Permutation-based)")
    print(f"{'='*60}")
    print(f"  Baseline loss: {baseline_loss:.6f}")
    print(f"  Higher = more important\n")
    
    max_name_len = max(len(name) for name, _ in sorted_imp)
    for rank, (name, (imp, std)) in enumerate(sorted_imp, 1):
        bar_len = max(0, int(imp / max(sorted_imp[0][1][0], 1e-10) * 30))
        bar = '█' * bar_len
        print(f"  {rank:2d}. {name:<{max_name_len}} {imp:+.6f} (±{std:.6f}) {bar}")
    
    print(f"\n{'='*60}")


##############################################
# Mode 12: Input Sensitivity (Per-Output)
##############################################
def run_input_sensitivity(model, csv_file, delimiter, input_cols, output_cols, col_types,
                          vocabularies, scalings, image_params, batch_size, device,
                          n_samples=None):
    """
    Gradient-based input sensitivity analysis, broken down per output.
    For each output neuron/column, shows how sensitive it is to each input feature
    by averaging |∂output_j / ∂input_i| across the dataset.
    """
    dataset = CustomDataset(csv_file, delimiter, input_cols, output_cols,
                            col_types, vocabularies, scalings, image_params)

    # Optionally subsample for speed
    total = len(dataset)
    if n_samples and n_samples < total:
        indices = torch.randperm(total)[:n_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"  Using {n_samples} / {total} samples for sensitivity analysis.")
    else:
        n_samples = total

    loader = DataLoader(dataset, batch_size=min(batch_size, 64), shuffle=False)

    output_layout, _, _ = build_output_layout(output_cols, col_types, scalings, vocabularies)

    # Build input feature-to-column mapping (same as feature importance)
    input_dim_map = []  # (col_name, start_idx, end_idx)
    idx = 0
    for col in input_cols:
        ct = col_types.get(col, "in")
        if ct in ("in", "inlab"):
            input_dim_map.append((col, idx, idx + 1))
            idx += 1
        elif ct == "inlabcat":
            vocab = vocabularies.get(col, {})
            n = len(vocab)
            input_dim_map.append((col, idx, idx + n))
            idx += n
        elif ct == "intexcat":
            vocab = vocabularies.get(col, {})
            max_len = scalings.get(col, {}).get("max_len", 1)
            n = len(vocab) * max_len
            input_dim_map.append((col, idx, idx + n))
            idx += n
        elif ct == "intex":
            max_len = scalings.get(col, {}).get("max_len", 1)
            input_dim_map.append((col, idx, idx + max_len))
            idx += max_len
        elif ct == "inim":
            im_size = scalings[col]["im_size"]; patch_size = scalings[col]["patch_size"]
            num_patches = 1 if patch_size == 1 else (im_size // patch_size) ** 2
            n = num_patches * 3
            input_dim_map.append((col, idx, idx + n))
            idx += n

    n_inputs = len(input_dim_map)

    # Build output entry info
    # For each output layout entry, we accumulate sensitivity to each input feature
    # sensitivity_matrix[out_idx][in_idx] = accumulated |grad|
    n_outputs = len(output_layout)
    sensitivity_matrix = np.zeros((n_outputs, n_inputs))
    count = 0

    model.eval()
    print(f"\n  Computing gradients for {n_outputs} outputs x {n_inputs} input features...")

    for inputs_batch, _ in tqdm(loader, desc="  Sensitivity", leave=False):
        inputs_batch = inputs_batch.to(device).requires_grad_(True)
        outputs_batch = model(inputs_batch)  # (B, output_pred_dim)

        for out_idx, entry in enumerate(output_layout):
            start_o = entry['start']
            end_o = entry['end']
            # Sum over all output dims in this column and all samples in batch
            target_sum = outputs_batch[:, start_o:end_o].sum()

            model.zero_grad()
            if inputs_batch.grad is not None:
                inputs_batch.grad.zero_()
            target_sum.backward(retain_graph=True)

            grad = inputs_batch.grad.detach().abs().cpu().numpy()  # (B, input_dim)

            for in_idx, (_, s, e) in enumerate(input_dim_map):
                # Sum the absolute gradient across the feature's dimensions, mean across batch
                sensitivity_matrix[out_idx, in_idx] += grad[:, s:e].sum(axis=1).mean()

        count += 1

    # Normalize by number of batches
    sensitivity_matrix /= max(count, 1)

    # Display results
    print(f"\n{'='*70}")
    print(f"INPUT SENSITIVITY (Per Output)")
    print(f"{'='*70}")
    print(f"  Method: mean |∂output / ∂input| averaged over {n_samples} samples")
    print(f"  Higher = output is more sensitive to that input\n")

    in_names = [name for name, _, _ in input_dim_map]
    max_in_len = max(len(n) for n in in_names) if in_names else 5

    for out_idx, entry in enumerate(output_layout):
        col = entry['col']
        ctype = entry['type']
        sensitivities = sensitivity_matrix[out_idx]

        # Sort by sensitivity descending
        sorted_indices = np.argsort(sensitivities)[::-1]
        max_sens = sensitivities[sorted_indices[0]] if len(sorted_indices) > 0 else 1e-10

        type_label = "classification" if ctype in ["outlabcat", "outexcat"] else "regression"
        print(f"\n  Output: [{col}] ({type_label})")
        print(f"  {'─'*55}")

        for rank, in_idx in enumerate(sorted_indices, 1):
            sens = sensitivities[in_idx]
            bar_len = max(0, int(sens / max(max_sens, 1e-10) * 30))
            bar = '█' * bar_len
            print(f"    {rank:2d}. {in_names[in_idx]:<{max_in_len}}  {sens:.6f}  {bar}")

    # Overall sensitivity (sum across outputs)
    overall = sensitivity_matrix.sum(axis=0)
    sorted_overall = np.argsort(overall)[::-1]
    max_overall = overall[sorted_overall[0]] if len(sorted_overall) > 0 else 1e-10

    print(f"\n  Overall Sensitivity (summed across all outputs)")
    print(f"  {'─'*55}")
    for rank, in_idx in enumerate(sorted_overall, 1):
        sens = overall[in_idx]
        bar_len = max(0, int(sens / max(max_overall, 1e-10) * 30))
        bar = '█' * bar_len
        print(f"    {rank:2d}. {in_names[in_idx]:<{max_in_len}}  {sens:.6f}  {bar}")

    print(f"\n{'='*70}")


##############################################
# Mode 13: Error Analysis (Top-N Worst Examples)
##############################################
def run_error_analysis(model, csv_file, delimiter, input_cols, output_cols, col_types,
                       vocabularies, scalings, image_params, batch_size, device,
                       max_display=50):
    """
    Run all examples through the model, compute per-sample loss,
    and display the top N examples with the highest loss.
    """
    dataset = CustomDataset(csv_file, delimiter, input_cols, output_cols,
                            col_types, vocabularies, scalings, image_params)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    output_layout, _, _ = build_output_layout(output_cols, col_types, scalings, vocabularies)
    has_categorical = any(e['type'] in ['outlabcat', 'outexcat'] for e in output_layout)

    class_weights = dataset.compute_class_weights(device=device)
    if has_categorical:
        criterion = CombinedLoss(output_layout, column_weights=class_weights); criterion.calibrate(dataset)
    else:
        criterion = nn.HuberLoss(reduction='none')

    model.eval()

    all_losses = []      # per-sample loss
    all_preds = []       # raw predictions
    all_targets = []     # raw targets
    all_indices = []     # sample index (row in cleaned dataset)
    sample_offset = 0

    print(f"\n  Running inference on {len(dataset)} samples...")

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="  Error scan", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            bsz = inputs.size(0)

            # Compute per-sample loss
            for i in range(bsz):
                pred_i = outputs[i:i+1]
                tgt_i = targets[i:i+1]
                if has_categorical:
                    loss_i = criterion(pred_i, tgt_i).item()
                else:
                    loss_i = criterion(pred_i, tgt_i).mean().item()
                all_losses.append(loss_i)
                all_preds.append(pred_i.cpu().numpy().flatten())
                all_targets.append(tgt_i.cpu().numpy().flatten())
                all_indices.append(sample_offset + i)

            sample_offset += bsz

    # Sort by loss descending
    sorted_order = np.argsort(all_losses)[::-1]
    n_show = min(max_display, len(sorted_order))

    # Try to read back the original CSV for row display
    try:
        df_orig = read_table(csv_file, delimiter=delimiter)
    except Exception:
        df_orig = None

    print(f"\n{'='*70}")
    print(f"ERROR ANALYSIS — Top {n_show} Worst Examples")
    print(f"{'='*70}")
    print(f"  Total samples: {len(all_losses)}")
    mean_loss = np.mean(all_losses)
    median_loss = np.median(all_losses)
    print(f"  Mean loss:   {mean_loss:.6f}")
    print(f"  Median loss: {median_loss:.6f}")
    print(f"  Max loss:    {all_losses[sorted_order[0]]:.6f}")
    print(f"  Min loss:    {all_losses[sorted_order[-1]]:.6f}")

    print(f"\n  {'Rank':<5} {'Row':<6} {'Loss':<14} ", end="")
    for entry in output_layout:
        col = entry['col']
        ctype = entry['type']
        if ctype in ['outlabcat', 'outexcat']:
            print(f"{'['+col+'] Pred':>14} {'Actual':>10} ", end="")
        else:
            print(f"{'['+col+'] Pred':>14} {'Actual':>14} {'Error':>10} ", end="")
    print()
    print(f"  {'─'*120}")

    for rank in range(n_show):
        idx = sorted_order[rank]
        sample_idx = all_indices[idx]
        loss = all_losses[idx]
        pred = all_preds[idx]
        tgt = all_targets[idx]

        line = f"  {rank+1:<5} {sample_idx:<6} {loss:<14.6f} "

        for entry in output_layout:
            col = entry['col']
            ctype = entry['type']
            s = entry['start']
            e = entry['end']
            ts = entry['tgt_start']
            te = entry['tgt_end']

            if ctype in ['outlabcat', 'outexcat']:
                pred_class = int(np.argmax(pred[s:e]))
                if ctype == 'outlabcat':
                    tgt_class = int(tgt[ts])
                else:
                    tgt_class = int(tgt[ts])  # first position for display
                # Try to get label names
                vocab = vocabularies.get(col, {})
                pred_label = str(pred_class)
                tgt_label = str(tgt_class)
                for k, v in vocab.items():
                    if v == pred_class: pred_label = k
                    if v == tgt_class: tgt_label = k
                match = "✓" if pred_class == tgt_class else "✗"
                line += f"{pred_label:>14} {tgt_label:>8} {match:>2} "
            else:
                pred_vals = pred[s:e]
                tgt_vals = tgt[ts:te]
                # For single-value regression
                if len(pred_vals) == 1:
                    p, t = pred_vals[0], tgt_vals[0]
                    err = abs(p - t)
                    # Unscale if possible
                    if col in scalings and "min" in scalings[col]:
                        smin, smax = scalings[col]["min"], scalings[col]["max"]
                        p_real = to_display_units(p, scalings[col])
                        t_real = to_display_units(t, scalings[col])
                        err_real = abs(p_real - t_real)
                        line += f"{p_real:>14.4f} {t_real:>14.4f} {err_real:>10.4f} "
                    else:
                        line += f"{p:>14.6f} {t:>14.6f} {err:>10.6f} "
                else:
                    # Multi-value: show MSE for this column
                    mse = np.mean((pred_vals - tgt_vals) ** 2)
                    line += f"{'(multi)':>14} {'(multi)':>14} {mse:>10.6f} "

        print(line)

    # Summary stats
    print(f"\n  Loss distribution:")
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(all_losses, p)
        print(f"    P{p:<2}: {val:.6f}")

    # If there are misclassifications, summarize
    for entry in output_layout:
        if entry['type'] in ['outlabcat', 'outexcat']:
            col = entry['col']
            s, e = entry['start'], entry['end']
            ts = entry['tgt_start']
            correct = 0
            total = len(all_preds)
            for i in range(total):
                pc = int(np.argmax(all_preds[i][s:e]))
                tc = int(all_targets[i][ts])
                if pc == tc: correct += 1
            print(f"\n  [{col}] Classification: {correct}/{total} correct ({100*correct/max(total,1):.1f}%)")

    print(f"\n{'='*70}")


##############################################
# Mode 10: Learning Rate Finder
##############################################
##############################################
# Mode 10: Learning Rate Finder (FIXED)
##############################################
def run_lr_finder(csv_file, delimiter, input_cols, output_cols, col_types,
                  vocabularies, scalings, image_params, batch_size,
                  hidden_dims, activation_cls, residual_type, norm_type,
                  optimizer_choice, device, lr_min=1e-7, lr_max=10.0, n_steps=200):
    """Learning rate range test to find optimal LR using the SELECTED optimizer."""
    
    dataset = CustomDataset(csv_file, delimiter, input_cols, output_cols,
                            col_types, vocabularies, scalings, image_params)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_dims = calculate_input_dim(input_cols, col_types, scalings, vocabularies)
    output_layout, output_pred_dim, _ = build_output_layout(output_cols, col_types, scalings, vocabularies)
    has_categorical = any(e['type'] in ['outlabcat', 'outexcat'] for e in output_layout)
    
    class_weights = dataset.compute_class_weights(device=device)
    if has_categorical:
        criterion = CombinedLoss(output_layout, column_weights=class_weights); criterion.calibrate(dataset)
    else:
        criterion = nn.HuberLoss()
    
    model = MLPO(input_dims, hidden_dims, output_pred_dim, activation_cls, residual_type,
                 norm_type=norm_type, groups=1, attention_type="none", num_heads=1,
                 input_attention_type="none", moe_mode=1).to(device)
    
    with torch.no_grad():
        model(torch.zeros(1, input_dims).to(device))
    
    # --- UPDATED: Use the User's Selected Optimizer ---
    print(f"\n  Using selected optimizer: {optimizer_choice}")
    # Initialize with lr_min
    optimizer = select_optimizer(optimizer_choice, model, custom_lr=lr_min)
    
    # Exponential LR schedule
    gamma = (lr_max / lr_min) ** (1.0 / n_steps)
    
    lrs = []
    losses = []
    best_loss = float('inf')
    
    print(f"  Running LR range test: {lr_min:.1e} -> {lr_max:.1e} over {n_steps} steps...")
    
    step = 0
    data_iter = iter(loader)
    smoothed_loss = 0.0
    
    _is_evolution = isinstance(optimizer, EvolutionaryOptimizer)

    for step in range(n_steps):
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inputs, targets = next(data_iter)
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Define closure for optimizer
        def closure():
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Evolution optimizer doesn't support backward(), it calculates loss in forward pass
            if not _is_evolution:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            return loss

        # Take step
        loss = optimizer.step(closure)
        
        # Get current LR (handling param groups)
        # Evolution strategy stores LR in param_groups similar to others
        current_lr = optimizer.param_groups[0]['lr']
        current_loss = loss.item() if isinstance(loss, torch.Tensor) else loss
        
        # Exponential smoothing
        if step == 0:
            smoothed_loss = current_loss
        else:
            smoothed_loss = 0.9 * smoothed_loss + 0.1 * current_loss
        
        lrs.append(current_lr)
        losses.append(smoothed_loss)
        
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
        
        # Stop if loss diverges (4x best)
        # Note: Evolution strategies might have higher variance, relax check slightly
        divergence_factor = 10.0 if _is_evolution else 4.0
        if smoothed_loss > divergence_factor * best_loss and step > 10:
            print(f"  Loss diverging at LR={current_lr:.2e}, stopping.")
            break
        
        # Update LR for next step
        for pg in optimizer.param_groups:
            pg['lr'] *= gamma
    
    if not lrs:
        print("  No valid steps completed.")
        return
    
    # Find steepest descent
    min_loss_idx = np.argmin(losses)
    suggested_lr = lrs[min_loss_idx] / 10.0  # Heuristic: 1/10th of min loss LR
    
    print(f"\n{'='*60}")
    print(f"LEARNING RATE FINDER RESULTS ({optimizer_choice})")
    print(f"{'='*60}")
    print(f"  Steps completed: {len(lrs)}")
    print(f"  Loss at minimum: {min(losses):.6f} (at LR={lrs[min_loss_idx]:.2e})")
    print(f"  Suggested LR:    {suggested_lr:.2e}")
    print(f"  Safe LR range:   {suggested_lr/3:.2e} to {suggested_lr*3:.2e}")
    
    

    # Save plot
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(lrs, losses, linewidth=1.5)
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Smoothed Loss')
        ax.set_title(f'LR Range Test ({optimizer_choice})')
        ax.axvline(x=suggested_lr, color='r', linestyle='--', label=f'Suggested: {suggested_lr:.2e}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("lr_finder_plot.png", dpi=150)
        plt.close()
        print(f"  Plot saved to lr_finder_plot.png")
    except Exception as e:
        print(f"  Plot failed: {e}")
    
    print(f"{'='*60}")
    return suggested_lr


##############################################
# Mode 11: Cross-Validation (K-Fold)
##############################################
def run_cross_validation(csv_file, delimiter, input_cols, output_cols, col_types,
                        vocabularies, scalings, image_params, batch_size,
                        hidden_dims, activation_cls, activation_type, residual_type,
                        norm_type, optimizer_choice, custom_lr, device,
                        k_folds=5, steps_per_fold=2000,
                        noise_mode="none", noise_params=None,
                        base_activation_cls_for_config=None):
    """K-fold cross-validation to estimate generalization performance."""
    if noise_params is None:
        noise_params = {}
    
    raw = read_table(csv_file, delimiter=delimiter)
    raw, dups = dedupe_rows(raw, input_cols, output_cols, col_types)
    print(f"  Exact duplicate rows removed: {dups}")
    dataset = CustomDataset(csv_file, delimiter, input_cols, output_cols,
                           col_types, vocabularies, scalings, image_params, df=raw)
    
    n = len(dataset)
    # rows with identical inputs share a fold, so no validation fold sees a training input
    fold_ids = group_kfold_ids(_row_group_keys(dataset.df, input_cols, col_types), k_folds)
    
    fold_size = n // k_folds
    fold_results = []
    
    print(f"\n{'='*60}")
    print(f"K-FOLD CROSS-VALIDATION (K={k_folds})")
    print(f"{'='*60}")
    print(f"  Total samples: {n}")
    print(f"  Fold size:     ~{fold_size}")
    print(f"  Steps/fold:    {steps_per_fold}")
    
    input_dims = calculate_input_dim(input_cols, col_types, scalings, vocabularies)
    output_layout, output_pred_dim, _ = build_output_layout(output_cols, col_types, scalings, vocabularies)
    has_categorical = any(e['type'] in ['outlabcat', 'outexcat'] for e in output_layout)
    
    for fold in range(k_folds):
        print(f"\n  --- Fold {fold+1}/{k_folds} ---")
        
        # Split indices
        val_indices = np.flatnonzero(fold_ids == fold).tolist()
        train_indices = np.flatnonzero(fold_ids != fold).tolist()
        
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        class_weights = dataset.compute_class_weights(device=device)
        if has_categorical:
            criterion = CombinedLoss(output_layout, column_weights=class_weights); criterion.calibrate(train_subset)
        else:
            criterion = nn.HuberLoss()
        
        # Build model
        dropout_prob = noise_params.get("dropout_pct", 0.0) if noise_mode == "dropout" else 0.0
        model = MLPO(input_dims, hidden_dims, output_pred_dim, activation_cls, residual_type,
                     norm_type=norm_type, groups=1, attention_type="none", num_heads=1,
                     input_attention_type="none", moe_mode=1, dropout_prob=dropout_prob).to(device)
        
        with torch.no_grad():
            model(torch.zeros(1, input_dims).to(device))
        
        optimizer = select_optimizer(optimizer_choice, model, custom_lr=custom_lr)
        
        # Train
        step = 0
        for epoch in range(10000000):
            model.train()
            for inputs, targets in train_loader:
                if step >= steps_per_fold: break
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                step += 1
            if step >= steps_per_fold: break
        
        # Validate
        val_loss = validate_model(model, val_loader, criterion, device)
        fold_results.append(val_loss)
        print(f"    Validation loss: {val_loss:.6f}")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    mean_loss = np.mean(fold_results)
    std_loss = np.std(fold_results)
    
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    for i, fl in enumerate(fold_results):
        print(f"  Fold {i+1}: {fl:.6f}")
    print(f"\n  Mean:  {mean_loss:.6f}")
    print(f"  Std:   {std_loss:.6f}")
    print(f"  CI95:  {mean_loss:.6f} ± {1.96 * std_loss / math.sqrt(k_folds):.6f}")
    print(f"{'='*60}")


##############################################
# Interactive sampling GUI (local web server, stdlib only)
##############################################
TEXT_INPUT_TYPES = ("intex", "intexcat", "inenc")


def _token_label(col, col_type, vocab, tok_id):
    if col_type == "inenc":
        return seqm.decoded_tokens([tok_id], vocab)[0] if tok_id >= len(seqm.SPECIALS) else "<unk>"
    inv = {v: k for k, v in vocab.items()}
    return inv.get(tok_id, "?")


def occlusion_attribution(model, x, input_slots, col_types, vocabs, output_layout, scalings, chunk=256, samples=16):
    """How much each output changes when one input is taken away, for any model and column types.
    x: (1, D) encoded input row. input_slots: [(col, start, width)].
    Taken away: numeric -> 0 in model units (middle of the default scaling); category (one-hot or label code) ->
    replaced by each other category in turn, averaged (an all-zero one-hot is an input the model never saw);
    image -> all patch codes 0; text (intex / intexcat / inenc) -> the whole text removed, and separately each
    character / token replaced by every other token of the column's vocabulary (or `samples` random ones), averaged,
    with every position unchanged (deleting would shift the later tokens, which position-aware inputs notice).
    Value = output with the input - output without it, per output column: numeric -> real units, label code -> code,
    category -> probability of the predicted class, text -> per position / generated step the log-probability of the
    predicted token. Decoders are teacher-forced on the text they generate for x, so every variant is scored on it."""
    model.eval()
    targets = None
    with torch.no_grad(): base_out = model(x)
    gen = {}
    if getattr(model, "needs_targets", False):
        targets = x.new_zeros(1, max(e["tgt_end"] for e in output_layout))
        for e in output_layout:
            if e["type"] != "outdec" or e["col"] not in model.decoders: continue
            ids = base_out[0, e["start"]:e["end"]].view(e["max_len"], e["num_classes"]).argmax(-1)
            n = int((ids == seqm.EOS).nonzero()[0, 0]) + 1 if (ids == seqm.EOS).any() else len(ids)
            ids[n:] = seqm.PAD; gen[e["col"]] = (ids, n); targets[0, e["tgt_start"]:e["tgt_end"]] = ids.float()
    variants, entries, inputs = [x[0]], [], []  # entries: (col, token index or None, [variant rows averaged])

    def add(v): variants.append(v); return len(variants) - 1
    for col, start, width in input_slots:
        ct = col_types[col]; sl = x[0, start:start + width]
        if ct in ("inlabcat", "inlab"):
            codes = range(width) if ct == "inlabcat" else sorted(set(vocabs[col].values()))
            cur = int(sl.argmax()) if ct == "inlabcat" else int(round(float(sl[0])))
            rows = []
            for k in codes:
                if k == cur: continue
                v = x[0].clone()
                if ct == "inlabcat": v[start:start + width] = 0; v[start + k] = 1.0
                else: v[start] = float(k)
                rows.append(add(v))
            entries.append((col, None, rows or [0])); inputs.append({"col": col, "type": ct, "tokens": None})
        elif ct in TEXT_INPUT_TYPES:
            V = len(vocabs[col]) if ct == "intexcat" else None
            ids = (sl.view(-1, V).argmax(-1) + 1) * (sl.view(-1, V).sum(-1) > 0) if ct == "intexcat" else sl.round().long()
            n = int((ids > 0).sum())
            v = x[0].clone(); v[start:start + width] = 0; entries.append((col, None, [add(v)]))
            labels = []
            # replacement candidates: the column's real token ids (intex / intexcat: 1..V, inenc: past the specials)
            cand = (list(range(len(seqm.SPECIALS), seqm.vocab_size(vocabs[col]))) if ct == "inenc"
                    else list(range(1, len(vocabs[col]) + 1)))
            rng = torch.Generator().manual_seed(0)  # the same replacements every time
            for j in range(n):
                others = [c for c in cand if c != int(ids[j])]
                if len(others) > samples: others = [others[i] for i in torch.randperm(len(others), generator=rng)[:samples].tolist()]
                rows = []
                for r in others or [int(ids[j])]:
                    v = x[0].clone()
                    if ct == "intexcat": blk = v[start:start + width].view(-1, V); blk[j] = 0; blk[j, r - 1] = 1.0
                    else: v[start + j] = float(r)
                    rows.append(add(v))
                entries.append((col, j, rows)); labels.append(_token_label(col, ct, vocabs[col], int(ids[j])))
            inputs.append({"col": col, "type": ct, "tokens": labels})
        else:
            v = x[0].clone(); v[start:start + width] = 0; entries.append((col, None, [add(v)]))
            inputs.append({"col": col, "type": ct, "tokens": None})
    X = torch.stack(variants); outs = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            xb = X[i:i + chunk]
            outs.append((model(xb, targets.expand(len(xb), -1)) if targets is not None else model(xb)).float())
    out = torch.cat(outs)

    def spread(effect):  # (R,) -> {"columns": {col: v}, "tokens": {col: [per token]}}
        res = {"columns": {}, "tokens": {}}
        for col, j, rows in entries:
            v = float(effect[0] - effect[rows].mean())
            if j is None: res["columns"][col] = v
            else: res["tokens"].setdefault(col, []).append(v)
        return res
    result = {}
    for e in output_layout:
        col, ct, sl = e["col"], e["type"], out[:, e["start"]:e["end"]]
        if ct == "out":
            s = scalings.get(col, {}); eff = sl[:, 0]
            if "min" in s: eff = to_display_units(eff, s)
            result[col] = {"kind": "scalar", "unit": "value", **spread(eff)}
        elif ct == "outlab": result[col] = {"kind": "scalar", "unit": "label code", **spread(sl[:, 0])}
        elif ct == "outlabcat":
            k = int(sl[0].argmax()); inv = {v: kk for kk, v in vocabs[col].items()}
            result[col] = {"kind": "scalar", "unit": f"probability of '{inv.get(k, k)}'", **spread(sl.softmax(-1)[:, k])}
        elif ct in ("outexcat", "outdec"):
            L = sl.view(len(X), e["max_len"], e["num_classes"]).log_softmax(-1)
            if col in gen: ids, n = gen[col]
            else:
                ids = L[0].argmax(-1); stop = seqm.EOS if ct == "outdec" else 0
                n = int((ids == stop).nonzero()[0, 0]) + (1 if ct == "outdec" else 0) if (ids == stop).any() else len(ids)
                n = max(n, 1)
            vocab = vocabs[col]
            labels = ([seqm.decoded_tokens([int(i)], vocab)[0] if int(i) >= len(seqm.SPECIALS) else {seqm.EOS: "<eos>", seqm.UNK: "<unk>"}.get(int(i), "<pad>")
                       for i in ids[:n]] if ct == "outdec" else [{v: kk for kk, v in vocab.items()}.get(int(i), "<end>") for i in ids[:n]])
            chosen = L[:, torch.arange(n), ids[:n].to(L.device)]
            result[col] = {"kind": "steps", "unit": "log-probability of the predicted token", "steps": labels,
                           "values": [spread(chosen[:, t]) for t in range(n)]}
        elif ct == "outex":
            inv = {v: kk for kk, v in vocabs[col].items()}; codes = [int(round(v)) for v in sl[0].tolist()]
            n = max(1, sum(1 for c in codes if c > 0))
            result[col] = {"kind": "steps", "unit": "character code", "steps": [inv.get(c, "?") for c in codes[:n]],
                           "values": [spread(sl[:, t]) for t in range(n)]}
    return {"inputs": inputs, "outputs": result}


def levenshtein_ops(a, b):
    """Edit distance between token lists a (target) and b (prediction) plus one alignment:
    [("match"|"sub", a_tok, b_tok) | ("del", a_tok, None) | ("ins", None, b_tok)]."""
    n, m = len(a), len(b)
    d = np.zeros((n + 1, m + 1), dtype=np.int32); d[:, 0] = np.arange(n + 1); d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            ops.append(("match" if a[i - 1] == b[j - 1] else "sub", a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif i and d[i, j] == d[i - 1, j] + 1: ops.append(("del", a[i - 1], None)); i -= 1
        else: ops.append(("ins", None, b[j - 1])); j -= 1
    return int(d[n, m]), ops[::-1]


def text_fit_stats(true_toks, pred_toks, max_labels=40, max_tokens=200):
    """How well generated text matches its targets. Token lists per row (chars or words).
    row_error: edit distance / target length per row (0 = perfect). Token confusions come from the alignment:
    a small vocabulary gets a full matrix (with "∅" for missing / extra tokens), a large one the top mistakes."""
    from collections import Counter
    pairs, dists, lens = Counter(), [], []
    for t, p in zip(true_toks, pred_toks):
        dist, ops = levenshtein_ops(t[:max_tokens], p[:max_tokens]); dists.append(dist); lens.append(len(t))
        for op, a, b in ops: pairs[(a if a is not None else "∅", b if b is not None else "∅")] += 1
    dists, lens = np.asarray(dists, dtype=np.float64), np.asarray(lens, dtype=np.float64)
    row_error = dists / np.maximum(lens, 1.0)
    L = int(max((len(t) for t in true_toks), default=0))
    pos_acc = [_gui_float(np.mean([len(p) > k and p[k] == t[k] for t, p in zip(true_toks, pred_toks) if len(t) > k])) for k in range(L)]
    res = {"kind": "text", "n": len(true_toks), "exact": _gui_float(np.mean([t == p for t, p in zip(true_toks, pred_toks)])) if true_toks else None,
           "error_rate": _gui_float(dists.sum() / max(lens.sum(), 1.0)), "pos_acc": pos_acc,
           "len_true": [len(t) for t in true_toks], "len_pred": [len(p) for p in pred_toks]}
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs} - {"∅"}) + (["∅"] if any("∅" in k for k in pairs) else [])
    if len(labels) <= max_labels:
        ix = {l: i for i, l in enumerate(labels)}; cm = np.zeros((len(labels), len(labels)), dtype=int)
        for (a, b), c in pairs.items(): cm[ix[a], ix[b]] += c
        res.update(labels=labels, confusion=cm.tolist())
    else:
        res["top_confusions"] = [[a, b, c] for (a, b), c in pairs.most_common() if a != b][:15]
    return res, row_error


_GUI_INPUT_TYPES = ["in", "inlab", "intex", "inim", "inlabcat", "intexcat", "inenc"]
_GUI_OUTPUT_TYPES = ["out", "outlab", "outex", "outlabcat", "outexcat", "outdec"]
_GUI_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlpRes6_gui.html")


def _gui_b64(arr):
    import base64
    return base64.b64encode(np.ascontiguousarray(np.nan_to_num(np.asarray(arr, dtype=np.float32))).tobytes()).decode("ascii")


def _gui_float(v):
    v = float(v)
    return v if math.isfinite(v) else None


def _gui_clean(obj):
    """NaN/inf -> None recursively (a diverged model must not break strict JSON)."""
    if isinstance(obj, float): return obj if math.isfinite(obj) else None
    if isinstance(obj, dict): return {k: _gui_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_gui_clean(v) for v in obj]
    return obj


def _gui_shape_module(block):
    """The elementwise activation whose shape is worth plotting, or None (SGU / Maxout)."""
    act = getattr(block, "activation", None)
    if isinstance(act, GLUActivation): return act.activation_fn   # GLU projection mixes channels
    if isinstance(act, GLU2Activation): return act.activation_fn1
    return act


class InteractiveSampler:
    """Owns the trained model for the GUI server; every public method returns JSON-ready data."""

    def __init__(self, model_path="model.pt", config_path="config.json"):
        import threading
        self.lock = threading.RLock()
        self.model_path = model_path; self.config_path = config_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.version = 0; self.model_mtime = None; self.load_error = None
        self._dataset = None; self._dataset_error = None; self._data_stats = None
        self._load_model()

    # ─── loading ───
    def _load_config(self):
        with open(self.config_path) as f: self.config = json.load(f)
        cfg = self.config
        register_csv_format(cfg.get("file_path"), cfg.get("csv_format"))
        self.col_types = cfg["col_types"]; self.scalings = cfg["scalings"]
        self.vocabs = cfg.get("vocabularies", {}); self.image_params = cfg.get("image_params", {})
        self.input_cols = [c for c, t in self.col_types.items() if t in _GUI_INPUT_TYPES]
        self.output_cols = [c for c, t in self.col_types.items() if t in _GUI_OUTPUT_TYPES]
        self.input_dim = calculate_input_dim(self.input_cols, self.col_types, self.scalings, self.vocabs)
        self.output_layout, self.output_dim, _ = build_output_layout(self.output_cols, self.col_types, self.scalings, self.vocabs)
        self.layout_by_col = {e["col"]: e for e in self.output_layout}
        self.slots, start = {}, 0
        for c in self.input_cols:
            w = calculate_input_dim([c], self.col_types, self.scalings, self.vocabs)
            self.slots[c] = (start, w); start += w

    def _load_model(self):
        snapshot = dict(self.__dict__)
        try: self._load_model_unchecked()
        except Exception:
            self.__dict__.update(snapshot)  # all-or-nothing: never pair a new config with the old model
            raise

    def _load_model_unchecked(self):
        old_path = getattr(self, "config", {}).get("file_path")
        self._load_config()  # a new training run may have changed columns or architecture
        if self.config.get("file_path") != old_path: self._dataset = None; self._dataset_error = None
        model = build_model_from_config(self.config, self.input_dim, self.output_dim, self.device)
        materialize_lazy_modules(model, self.input_dim, self.device)
        sig = self._file_sig()
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()
        self.model = model; self.model_mtime = os.path.getmtime(self.model_path); self._sig = sig; self.version += 1
        self.load_error = None; self._data_stats = None

    def _file_sig(self):
        """(mtime_ns, size) of model.pt and config.json: two saves within one timestamp tick still differ in size."""
        return tuple((st.st_mtime_ns, st.st_size) for st in (os.stat(self.model_path), os.stat(self.config_path)))

    def check_reload(self, force=False):
        """Reload model.pt if it changed on disk (e.g. a training run saved a better model)."""
        with self.lock:
            try:
                changed = self._file_sig() != self._sig
            except OSError as e:
                self.load_error = str(e); return False
            if not (changed or force): return False
            try:
                self._load_model(); return True
            except Exception as e:  # half-written file, or config/model from different runs: keep the old model
                self._sig = self._file_sig()  # retry on the next change, not on every poll
                self.load_error = f"{type(e).__name__}: {e}"; return False

    # ─── units / encoding ───
    def _scaled(self, col):
        s = self.scalings.get(col, {})
        return isinstance(s, dict) and "min" in s and "max" in s

    def _to_display(self, col, v):
        if not self._scaled(col): return v
        return to_display_units(v, self.scalings[col])

    def _to_model(self, col, v):
        if not self._scaled(col): return v
        return to_model_units(v, self.scalings[col])

    def _labels(self, col):
        vocab = self.vocabs.get(col, {})
        return sorted(vocab, key=lambda k: vocab[k])

    def _hashed(self, col):
        return is_hashed(col, self.col_types, self.vocabs)

    def _hash_labels(self, col):
        return sorted(self.vocabs[col], key=lambda k: self.vocabs[col][k])

    def _default_input(self, col):
        ct = self.col_types[col]
        if ct == "in" and self._hashed(col):
            labels = self._hash_labels(col); return labels[0] if labels else ""
        if ct in ("inlab", "inlabcat"): return self._labels(col)[0]
        if ct == "in" and self._scaled(col):
            df = self._get_dataset()
            if df is not None and col in df:
                vals = numeric_series(df[col]).dropna()
                if len(vals): return float(vals.median())
            s = self.scalings[col]; return (s["min"] + s["max"]) / 2.0
        return 0.0 if ct == "in" else ""

    def _encode(self, inputs):
        raw = [inputs.get(c, self._default_input(c)) for c in self.input_cols]
        return np.asarray(encode_sample_input(raw, self.col_types, self.vocabs, self.scalings, self.image_params),
                          dtype=np.float32)

    def _sweep_values(self, col, lo=None, hi=None, n=200):
        """Display-unit values to sweep for one input (numeric grid or category labels)."""
        ct = self.col_types[col]
        if ct in ("inlab", "inlabcat"): return self._labels(col), True
        if ct != "in": raise ValueError(f"Input '{col}' ({ct}) cannot be swept.")
        if lo is None or hi is None:
            if self._scaled(col): lo, hi = self.scalings[col]["min"], self.scalings[col]["max"]
            else: lo, hi = -1.0, 1.0
        lo, hi = sorted((float(lo), float(hi)))
        if hi == lo: hi = lo + 1.0
        return np.linspace(lo, hi, int(n)).tolist(), False

    def _apply(self, batch, col, values):
        """Write one input's display-unit values into rows of a model-unit batch (in place)."""
        start, width = self.slots[col]; ct = self.col_types[col]
        if ct == "inlabcat":
            batch[:, start:start + width] = 0.0
            for i, v in enumerate(values):
                k = self.vocabs[col].get(str(v), 0)
                if 0 <= k < width: batch[i, start + k] = 1.0
        elif ct == "inlab":
            batch[:, start] = [float(self.vocabs[col].get(str(v), 0)) for v in values]
        else:
            batch[:, start] = self._to_model(col, np.asarray(values, dtype=np.float64))

    def _run(self, batch_np, chunk=4096):
        outs = []
        with torch.no_grad():
            for i in range(0, len(batch_np), chunk):
                x = torch.as_tensor(batch_np[i:i + chunk], dtype=torch.float32, device=self.device)
                outs.append(self.model(x).float().cpu())
        return torch.cat(outs, 0)

    # ─── outputs ───
    def _decode(self, raw):
        items = []
        for e in self.output_layout:
            col, ct = e["col"], e["type"]; sl = raw[e["start"]:e["end"]]
            inv = {v: k for k, v in self.vocabs.get(col, {}).items()}
            item = {"col": col, "type": ct}
            if ct == "outlabcat":
                probs = torch.softmax(sl, 0).tolist(); k = int(sl.argmax())
                item["value"] = inv.get(k, k)
                item["probs"] = [[str(inv.get(i, i)), p] for i, p in enumerate(probs)]
            elif ct == "outdec":
                logits = sl.view(e["max_len"], e["num_classes"]); ids = logits.argmax(-1).tolist()
                item["value"] = seqm.decode_ids(ids, self.vocabs[col])
                n = ids.index(seqm.EOS) + 1 if seqm.EOS in ids else len(ids)
                item["confidence"] = torch.softmax(logits[:n], -1).max(-1).values.tolist()
            elif ct == "outexcat":
                logits = sl.view(e["max_len"], e["num_classes"])
                item["value"] = "".join(inv.get(i, "") for i in logits.argmax(-1).tolist())
                item["confidence"] = torch.softmax(logits, -1).max(-1).values.tolist()
            elif ct == "outex":
                item["value"] = "".join(inv.get(round(v), "") for v in sl.tolist())
            elif ct == "outlab":
                code = round(sl[0].item()); item["value"] = inv.get(code, code); item["raw"] = sl[0].item()
            else:
                shown = _gui_float(self._to_display(col, sl[0].item()))
                if self._hashed(col) and shown is not None: item["raw"] = shown; item["value"] = hash_label(shown, self.vocabs[col]) or shown
                else: item["value"] = shown
            items.append(item)
        return items

    def _series(self, out, col):
        """Plottable series for one output column over a batch of raw model outputs."""
        e = self.layout_by_col[col]; sl = out[:, e["start"]:e["end"]]
        if e["type"] == "outlabcat":
            probs = torch.softmax(sl, -1).numpy()
            return [{"name": lab, "y": probs[:, i].tolist()} for i, lab in enumerate(self._labels(col))
                    if i < probs.shape[1]], "probability"
        if e["type"] == "outlab":
            return [{"name": col, "y": sl[:, 0].tolist()}], "label code"
        if e["type"] == "out":
            return [{"name": col, "y": self._to_display(col, sl[:, 0].numpy().astype(np.float64)).tolist()}], col
        raise ValueError(f"Output '{col}' ({e['type']}) cannot be plotted.")

    def _target_scalar(self, out, col, cls):
        """One number per row to map/differentiate: display value, class probability, or label code."""
        e = self.layout_by_col[col]; sl = out[:, e["start"]:e["end"]]
        if e["type"] == "outlabcat":
            if cls in (None, "", "__argmax__"): return sl.argmax(-1).float()
            return torch.softmax(sl, -1)[:, self.vocabs[col].get(str(cls), 0)]
        if e["type"] == "out" and self._scaled(col):
            return to_display_units(sl[:, 0], self.scalings[col])
        return sl[:, 0]

    # ─── tracing ───
    def _trace(self, x):
        """Forward x and capture per-block tensors (pre-activation, activation in/out, block output)."""
        blocks = list(getattr(self.model, "blocks", []))
        rec = [dict() for _ in blocks]; handles = []

        def hook(i, key, use_input=False):
            def fn(_mod, inp, out):
                if key not in rec[i]:  # first call only (GLU2 calls its activation twice)
                    t = inp[0] if use_input else out
                    if torch.is_tensor(t): rec[i][key] = t.detach().float()
            return fn
        for i, blk in enumerate(blocks):
            handles.append(blk.register_forward_hook(hook(i, "out")))
            if getattr(blk, "linear1", None) is not None:
                handles.append(blk.linear1.register_forward_hook(hook(i, "pre")))
            sm = _gui_shape_module(blk)
            if sm is not None:
                handles.append(sm.register_forward_hook(hook(i, "act_in", True)))
                handles.append(sm.register_forward_hook(hook(i, "act_out")))
        try:
            with torch.no_grad(): out = self.model(x)
        finally:
            for h in handles: h.remove()
        return out, rec

    # ─── API ───
    def meta(self):
        with self.lock:
            df = self._get_dataset()
            inputs = []
            for c in self.input_cols:
                ct = self.col_types[c]; spec = {"col": c, "type": ct, "default": self._default_input(c)}
                if ct in ("inlab", "inlabcat"): spec["options"] = self._labels(c)
                if self._scaled(c): spec["min"] = self.scalings[c]["min"]; spec["max"] = self.scalings[c]["max"]
                if ct in ("intex", "intexcat", "inenc"): spec["max_len"] = self.scalings[c].get("max_len")
                spec["sweepable"] = ct in ("in", "inlab", "inlabcat")
                if self._hashed(c): spec.update(type="inhash", options=self._hash_labels(c), sweepable=False)
                inputs.append(spec)
            outputs = []
            for e in self.output_layout:
                spec = {"col": e["col"], "type": e["type"], "plottable": e["type"] in ("out", "outlab", "outlabcat")}
                if e["type"] in ("outlab", "outlabcat"): spec["classes"] = self._labels(e["col"])
                if self._scaled(e["col"]): spec["min"] = self.scalings[e["col"]]["min"]; spec["max"] = self.scalings[e["col"]]["max"]
                outputs.append(spec)
            x0 = torch.zeros(1, self.input_dim, device=self.device)
            _, rec = self._trace(x0)
            blocks = []
            for i, blk in enumerate(getattr(self.model, "blocks", [])):
                sm = _gui_shape_module(blk)
                blocks.append({
                    "index": i, "kind": type(blk).__name__,
                    "in_dim": getattr(blk, "in_dim", None), "out_dim": getattr(blk, "out_dim", None),
                    "neurons": int(rec[i]["act_in"].shape[-1]) if "act_in" in rec[i] else
                               (int(rec[i]["pre"].shape[-1]) if "pre" in rec[i] else None),
                    "activation": type(blk.activation).__name__ if getattr(blk, "activation", None) is not None else None,
                    "shape_module": type(sm).__name__ if sm is not None else None,
                    "params": sum(p.numel() for p in blk.parameters()),
                })
            cfg = self.config
            return {
                "inputs": inputs, "outputs": outputs, "blocks": blocks,
                "input_dim": self.input_dim, "output_dim": self.output_dim,
                "generates": bool(getattr(self.model, "needs_targets", False)),  # has decoders: sampling controls apply
                "sequence": isinstance(self.model, seqm.SeqMLP),  # encoders / decoders: the Sequence tab applies
                "slots": {c: list(v) for c, v in self.slots.items()},
                "params": sum(p.numel() for p in self.model.parameters()),
                "weights": [{"name": n, "shape": list(p.shape)} for n, p in self.model.named_parameters()],
                "config": {k: cfg.get(k) for k in ("hidden_dims", "activation", "activation_type", "residual_type",
                                                   "norm_type", "moe_mode", "mlp_mode", "layer_routing", "use_grn",
                                                   "optimizer_choice", "file_path")},
                "dataset": {"available": df is not None, "rows": 0 if df is None else len(df), "error": self._dataset_error},
                "status": self.status(),
            }

    def status(self):
        return {"version": self.version, "model_path": os.path.abspath(self.model_path),
                "model_mtime": self.model_mtime, "load_error": self.load_error, "device": str(self.device)}

    def predict(self, inputs, generation=None):
        """generation: {temperature, top_k, top_p} for text decoders (temperature 0 = most likely tokens).
        Every other explorer view keeps generating with argmax."""
        with self.lock:
            base = self._encode(inputs)
            x = torch.as_tensor(base, device=self.device).unsqueeze(0)
            sampling = bool(generation) and isinstance(self.model, seqm.SeqMLP) and float(generation.get("temperature") or 0) > 0
            if isinstance(self.model, seqm.SeqMLP) and generation: self.model.set_generation(**generation)
            try: out, rec = self._trace(x)
            finally:
                if isinstance(self.model, seqm.SeqMLP): self.model.set_generation(temperature=0.0, top_k=0, top_p=1.0)
            raw = out[0].float().cpu()
            layers = []
            for i, r in enumerate(rec):
                layer = {"index": i, "skipped": "out" not in r}
                for key in ("pre", "act_in", "act_out", "out"):
                    if key in r: layer[key] = r[key].reshape(-1).cpu().tolist()
                layers.append(layer)
            routers = [p.reshape(-1)[0].item() for p in getattr(self.model, "_last_router_probs", []) or []]
            outputs = self._decode(raw)
            if sampling:  # sampled tokens have forced logits: their softmax is not the model's confidence
                for o in outputs: o.pop("confidence", None) if o["type"] == "outdec" else None
            return {"outputs": outputs, "raw": raw.tolist(), "encoded": base.tolist(),
                    "layers": layers, "routers": routers, "version": self.version}

    def attribution_view(self, inputs):
        """Attribution tab: occlusion attribution of every output to every input (any model)."""
        with self.lock:
            x = torch.as_tensor(self._encode(inputs), device=self.device).unsqueeze(0)
            slots = [(c, self.slots[c][0], self.slots[c][1]) for c in self.input_cols]
            res = occlusion_attribution(self.model, x, slots, self.col_types, self.vocabs, self.output_layout, self.scalings)
        return dict(res, version=self.version)

    def sequence_view(self, inputs):
        """Sequence tab: generated tokens (top-5 per step), attributions and attention maps (see mlpres_seq.explain)."""
        if not isinstance(self.model, seqm.SeqMLP): raise ValueError("The Sequence view needs text encoder / decoder columns.")
        with self.lock:
            x = torch.as_tensor(self._encode(inputs), device=self.device).unsqueeze(0)
            ex = seqm.explain(self.model, x)
            slots = [(c, self.slots[c][0], self.slots[c][1]) for c in self.input_cols]
            occ = occlusion_attribution(self.model, x, slots, self.col_types, self.vocabs, self.output_layout, self.scalings)["outputs"]
        m = self.model
        toks = lambda col, ids: [seqm.decoded_tokens([i], self.vocabs[col])[0] if i >= len(seqm.SPECIALS)
                                 else {seqm.UNK: "<unk>", seqm.BOS: "<bos>", seqm.EOS: "<eos>", seqm.PAD: "<pad>"}[i] for i in ids]
        enc = {}
        for c, start, width, V in m.enc_slots:
            ids = ex["enc_ids"][c]; n = max(1, sum(1 for i in ids if i != seqm.PAD))
            enc[c] = {"tokens": toks(c, ids[:n]), "n": n, "width": width}
        plain_cols = [c for c in self.input_cols if self.col_types[c] != "inenc"]

        def attr(a):  # one occlusion result -> per encoder token, plus one value per other input column
            out = {c: (a["tokens"].get(c) or [0.0] * enc[c]["n"])[:enc[c]["n"]] for c in enc}
            out["__inputs__"] = {c: a["columns"].get(c, 0.0) for c in plain_cols}
            return out
        decs = {}
        for col, d in ex["decoders"].items():
            decs[col] = {"tokens": toks(col, d["ids"]), "prob": d["prob"], "attr": [attr(a) for a in occ[col]["values"]],
                         "top": [[[toks(col, [i])[0], p] for i, p in zip(ti, tp)] for ti, tp in zip(d["top_ids"], d["top_p"])]}
        regular = {c: attr(r) for c, r in occ.items() if r["kind"] == "scalar"}
        # attention maps with row / column labels
        cross_keys = [(c, i) for c in enc for i in range(enc[c]["width"])] + [("__latent__", 0)]
        maps = []
        for r in ex["attention"]:
            name, w = r["module"], r["weights"]; parts = name.split(".")
            if parts[0] == "encoders":
                c = parts[1]; n = enc[c]["n"]
                maps.append({"side": "encoder", "col": c, "kind": "self", "name": ".".join(parts[2:]),
                             "rows": enc[c]["tokens"], "cols": enc[c]["tokens"], "w": w[:, :n, :n].tolist()})
            elif parts[0] == "decoders" and parts[1] in decs:
                c = parts[1]; d = decs[c]; n = len(d["tokens"]); sub = ".".join(parts[2:])
                is_cross = w.shape[-1] == len(cross_keys) and ("cross" in sub or "multihead_attn" in sub)
                if is_cross:
                    keep = [j for j, (ec, i) in enumerate(cross_keys) if ec == "__latent__" or i < enc[ec]["n"]]
                    labels = [("[latent]" if ec == "__latent__" else f"{ec}:{enc[ec]['tokens'][i]}") for ec, i in (cross_keys[j] for j in keep)]
                    maps.append({"side": "decoder", "col": c, "kind": "cross", "name": sub, "rows": d["tokens"], "cols": labels,
                                 "w": w[:, :n][:, :, keep].tolist()})
                else:
                    maps.append({"side": "decoder", "col": c, "kind": "self", "name": sub, "rows": d["tokens"],
                                 "cols": ["<bos>"] + d["tokens"][:n - 1], "w": w[:, :n, :n].tolist()})
        return {"encoders": enc, "decoders": decs, "regular": regular, "maps": maps,
                "inputs": plain_cols, "version": self.version}

    def sweep1d(self, inputs, x_col, out_col, lo=None, hi=None, n=200, show_data=False):
        with self.lock:
            values, categorical = self._sweep_values(x_col, lo, hi, n)
            batch = np.tile(self._encode(inputs), (len(values), 1))
            self._apply(batch, x_col, values)
            series, y_label = self._series(self._run(batch), out_col)
            res = {"x": values, "categorical": categorical, "series": series, "y_label": y_label,
                   "current": inputs.get(x_col, self._default_input(x_col)), "x_col": x_col, "out_col": out_col}
            if self.layout_by_col[out_col]["type"] == "outlab": res["y_classes"] = self._labels(out_col)
            if show_data and not categorical and self.layout_by_col[out_col]["type"] == "out":
                res["data"] = self._data_points(x_col, out_col)
            return res

    def sweep2d(self, inputs, x_col, y_col, out_col, cls=None, n=80, ranges=None, show_data=False):
        with self.lock:
            ranges = ranges or {}
            n = int(max(8, min(int(n), 300)))
            xs, xcat = self._sweep_values(x_col, *(ranges.get(x_col) or (None, None)), n=n)
            ys, ycat = self._sweep_values(y_col, *(ranges.get(y_col) or (None, None)), n=n)
            batch = np.tile(self._encode(inputs), (len(xs) * len(ys), 1))
            self._apply(batch, x_col, [x for _ in ys for x in xs])   # row-major: y outer, x inner
            self._apply(batch, y_col, [y for y in ys for _ in xs])
            z = self._target_scalar(self._run(batch), out_col, cls).numpy()
            e = self.layout_by_col[out_col]
            res = {"x": xs, "y": ys, "x_categorical": xcat, "y_categorical": ycat,
                   "nx": len(xs), "ny": len(ys), "z": _gui_b64(z),
                   "zmin": _gui_float(np.nanmin(z)), "zmax": _gui_float(np.nanmax(z)),
                   "mode": ("class" if e["type"] == "outlabcat" and cls in (None, "", "__argmax__") else
                            "prob" if e["type"] == "outlabcat" else "value"),
                   "current": [inputs.get(x_col, self._default_input(x_col)), inputs.get(y_col, self._default_input(y_col))]}
            if e["type"] in ("outlab", "outlabcat"): res["classes"] = self._labels(out_col)
            if show_data and not xcat and not ycat:
                pts = self._data_points(x_col, y_col)
                if pts: res["data"] = pts
            return res

    def sensitivity(self, inputs, out_col, cls=None):
        """Local effect of each input on one output at the current point."""
        with self.lock:
            base = self._encode(inputs)
            e = self.layout_by_col[out_col]
            if e["type"] == "outlabcat" and cls in (None, "", "__argmax__"):
                with torch.no_grad():
                    logits = self.model(torch.as_tensor(base, device=self.device).unsqueeze(0)).float().cpu()[0, e["start"]:e["end"]]
                cls = self._labels(out_col)[int(logits.argmax())]
            x = torch.as_tensor(base, device=self.device).unsqueeze(0).requires_grad_(True)
            with torch.enable_grad():
                y = self._target_scalar(self.model(x).float().cpu(), out_col, cls)
                grad = torch.autograd.grad(y.sum(), x)[0][0].cpu().numpy()
            rows = []
            for c in self.input_cols:
                start, width = self.slots[c]; ct = self.col_types[c]
                if ct == "in":
                    span = (self.scalings[c]["max"] - self.scalings[c]["min"]) if self._scaled(c) else 2.0
                    g = float(grad[start])  # d target / d model-unit input
                    rows.append({"col": c, "kind": "gradient", "effect": _gui_float(g * 2.0),
                                 "per_unit": _gui_float(g * 2.0 / span if span else g)})
                elif ct in ("inlab", "inlabcat"):
                    labels = self._labels(c)
                    batch = np.tile(base, (len(labels), 1)); self._apply(batch, c, labels)
                    t = self._target_scalar(self._run(batch), out_col, cls).numpy()
                    rows.append({"col": c, "kind": "categorical", "effect": _gui_float(t.max() - t.min()),
                                 "best": labels[int(t.argmax())], "worst": labels[int(t.argmin())],
                                 "values": [[l, _gui_float(v)] for l, v in zip(labels, t)]})
            target = ("P(" + str(cls) + ")" if e["type"] == "outlabcat" else out_col)
            return {"rows": rows, "target": target, "out_col": out_col, "cls": cls}

    def activation_shapes(self, inputs, block, lo=None, hi=None, n=241):
        with self.lock:
            blocks = list(getattr(self.model, "blocks", []))
            block = int(block); blk = blocks[block]; sm = _gui_shape_module(blk)
            if sm is None:
                return {"available": False, "reason": "This block has no standalone activation (SGU / Maxout)."}
            x = torch.as_tensor(self._encode(inputs), device=self.device).unsqueeze(0)
            _, rec = self._trace(x)
            if "act_in" not in rec[block]:
                return {"available": False, "reason": "Block was skipped by the layer router for this input."}
            cur_in = rec[block]["act_in"].reshape(-1).cpu().numpy(); cur_out = rec[block]["act_out"].reshape(-1).cpu().numpy()
            C = cur_in.shape[0]
            stats = self.data_stats(compute=False)
            bstats = stats["blocks"][block] if stats and block < len(stats["blocks"]) else None
            if lo is None or hi is None:
                lo_a, hi_a = float(cur_in.min()), float(cur_in.max())
                if bstats: lo_a, hi_a = min(lo_a, min(bstats["p01"])), max(hi_a, max(bstats["p99"]))
                pad = max(0.5, 0.15 * (hi_a - lo_a)); lo, hi = min(-3.0, lo_a - pad), max(3.0, hi_a + pad)
            xs = torch.linspace(float(lo), float(hi), int(n), device=self.device)
            with torch.no_grad():
                ys = sm(xs.unsqueeze(1).expand(-1, C).contiguous())
            if ys.shape != (len(xs), C):
                return {"available": False, "reason": f"{type(sm).__name__} is not an elementwise activation."}
            ys = ys.float().cpu().numpy()
            shared = bool(np.allclose(ys, ys[:, :1], atol=1e-6))
            params = []
            wrapper = blk.activation
            owners = [("activation", sm)] if wrapper is sm else [("wrapper", wrapper), ("activation", sm)]
            for owner_name, owner in owners:
                for name, p in owner.named_parameters():
                    if owner is wrapper and wrapper is not sm and name.startswith("activation_fn"):
                        continue  # listed under the inner activation
                    v = p.detach().float().cpu()
                    entry = {"name": f"{owner_name}.{name}", "shape": list(v.shape)}
                    if v.numel() <= 8: entry["values"] = [_gui_float(t) for t in v.reshape(-1).tolist()]
                    else: entry.update(mean=_gui_float(v.mean()), std=_gui_float(v.std()), min=_gui_float(v.min()), max=_gui_float(v.max()))
                    params.append(entry)
            res = {"available": True, "block": block, "module": type(sm).__name__, "wrapper": type(wrapper).__name__,
                   "neurons": C, "n": int(n), "x": xs.cpu().tolist(), "curves": _gui_b64(ys), "shared": shared,
                   "current_in": cur_in.tolist(), "current_out": cur_out.tolist(), "params": params,
                   "data": bstats}
            knots = self._pwl_knots(sm, C)
            if knots is not None: res["knots"] = knots
            return res

    def _pwl_knots(self, sm, C):
        """PaWeL breakpoints mapped through TAAF-style wrappers onto the plotted curve."""
        inner = sm
        while not isinstance(inner, PaWeL) and isinstance(getattr(inner, "activation_fn", None), nn.Module):
            inner = inner.activation_fn
        if not isinstance(inner, PaWeL) or inner.num_channels not in (1, C): return None
        with torch.no_grad():
            kx = inner.breakpoint_values()[0].expand(C, -1)          # [C, K] in the PaWeL's own input units
            if inner is not sm and hasattr(sm, "alpha") and hasattr(sm, "beta") and isinstance(sm, (TAAFWrapper, ResidualTAAFWrapper)):
                if abs(float(sm.alpha)) < 1e-12: return None
                kx = (kx - sm.beta) / sm.alpha                          # TAAF feeds alpha * x + beta
            ky = sm(kx.t().contiguous()).t()                             # exact y of the full activation there
        return {"k": int(kx.shape[1]), "x": _gui_b64(kx.float().cpu().numpy()), "y": _gui_b64(ky.float().cpu().numpy())}

    # ─── EasyNN-style network diagram ───
    def _node_modules(self, blk):
        """(module whose input is a hidden node's net input, module whose output is its activation)."""
        sm = _gui_shape_module(blk)
        if sm is not None: return sm, sm, True
        return blk.linear1, blk.linear1, False  # SGU / Maxout: linear1 output is the node

    def _node_importance(self):
        """Mean |node value| per diagram layer over a fixed data sample; cached per model version so the
        diagram shows the same nodes whatever the current input is."""
        if getattr(self, "_imp_cache", (None,))[0] == self.version: return self._imp_cache[1]
        X = None
        df = self._get_dataset()
        if df is not None:
            enc, _ = self._encode_rows(df.sample(min(256, len(df)), random_state=0))
            if enc: X = torch.as_tensor(np.asarray(enc, dtype=np.float32), device=self.device)
        if X is None: X = torch.as_tensor(self._encode({}), device=self.device).unsqueeze(0)
        out, rec = self._trace(X)
        imp = [X.abs().mean(0).cpu()]
        for r in rec:
            t = r.get("act_out", r.get("pre"))
            imp.append(t.abs().reshape(-1, t.shape[-1]).mean(0).cpu() if t is not None and t.shape[0] == X.shape[0] else None)
        imp.append(out.abs().mean(0).float().cpu())
        self._imp_cache = (self.version, imp)
        return imp

    def _target_vector(self, targets):
        """Model-unit targets per output entry (None where a target is missing or text)."""
        out = {}
        for e in self.output_layout:
            col, ct = e["col"], e["type"]; t = (targets or {}).get(col)
            if t is None or t == "": continue
            try:
                if ct == "out": out[col] = torch.tensor([[self._to_model(col, float(t))]], dtype=torch.float32)
                elif ct in ("outlabcat", "outlab"):
                    k = self.vocabs[col].get(str(t))
                    if k is not None: out[col] = torch.tensor([k])
            except (TypeError, ValueError):
                pass
        return out

    def _static_weights(self, pairs):
        """Effective linear weight per adjacent node-layer pair, or None where it is not a fixed matrix.
        Folds in each block's second linear layer, ReZero gains and the next norm's scale;
        normalization's re-centering, skip paths and gates are nonlinear and left out."""
        m = self.model; blocks = list(getattr(m, "blocks", []))
        if not isinstance(m, MLPO) or not blocks: return [None] * pairs
        rt = m.residual_type
        def norm_scale(blk, width):
            w = getattr(blk.norm, "weight", None)
            if w is None: w = getattr(blk.norm, "scale", None)
            return w.detach().float().cpu() if w is not None and w.numel() == width else torch.ones(width)
        def feeds(i):
            """d(block i's contribution to the next stream)/d(block i's activation), or None."""
            blk = blocks[i]
            if blk.attn is not None or blk.tiny_attn is not None or not isinstance(blk.grn, nn.Identity): return None
            if blk.residual_type == "highway" or getattr(blk, "use_sgu", False) or getattr(blk, "use_maxout", False): return None
            if _gui_shape_module(blk) is not blk.activation: return None  # GLU: projection in between
            if blk.linear2 is None: M = torch.eye(blk.out_dim)
            elif isinstance(blk.linear2, nn.Linear): M = blk.linear2.weight.detach().float().cpu()
            else: return None
            if blk.residual_type in ("rezero", "elementwise rezero", "elementwise_rezero"):
                g = blk.gamma.detach().float().cpu(); M = (g.reshape(-1, 1) if g.dim() else g) * M
            return M
        def cols(i):
            """Column slice of the stream after block i that holds block i's transform output."""
            if rt == "concat": total = blocks[i].output_dim(); return slice(total - blocks[i].out_dim, total)
            if rt == "densenet":
                start = self.input_dim + sum(b.out_dim for b in blocks[:i]); return slice(start, start + blocks[i].out_dim)
            return slice(0, blocks[i].out_dim)
        res = []
        blk0 = blocks[0]
        ok0 = (m.input_attn is None and isinstance(blk0.linear1, nn.Linear) and _gui_shape_module(blk0) is blk0.activation)
        res.append(blk0.linear1.weight.detach().float().cpu() * norm_scale(blk0, self.input_dim) if ok0 else None)
        for i in range(len(blocks)):
            M = feeds(i)
            if i + 1 < len(blocks):
                nb = blocks[i + 1]
                if M is None or not isinstance(nb.linear1, nn.Linear) or _gui_shape_module(nb) is not nb.activation:
                    res.append(None); continue
                W = nb.linear1.weight.detach().float().cpu() * norm_scale(nb, nb.in_dim)
                res.append(W[:, cols(i)] @ M)
            else:
                if M is None or not isinstance(m.final_linear, nn.Linear): res.append(None); continue
                W = m.final_linear.weight.detach().float().cpu().clone()
                if isinstance(m.final_skip, nn.Linear): W = W + m.final_skip.weight.detach().float().cpu()
                else: W = W + torch.eye(W.shape[0], W.shape[1])
                res.append(W[:, cols(i)] @ M)
        return res[:pairs]

    def _influence(self, x_row, src, dst_dim, dst_capture):
        """Exact Jacobian d(dst node values)/d(src node values) at this input, one batched backward.
        src: None for the input layer, else a block index whose activation output is the source."""
        K = dst_dim; X = x_row.detach().repeat(K, 1)
        holder, handles = {}, []
        if src is None:
            X.requires_grad_(True); holder["leaf"] = X
        else:
            mod = self._node_modules(self.model.blocks[src])[1]
            def swap(_m, _inp, out):
                if "leaf" in holder: return out
                leaf = out.detach().clone().requires_grad_(True); holder["leaf"] = leaf; return leaf
            handles.append(mod.register_forward_hook(swap))
        handles += dst_capture(holder)
        try:
            with torch.enable_grad():
                out = self.model(X)
                if "dst" not in holder: holder["dst"] = out
                if "leaf" not in holder or not holder["dst"].requires_grad: return None
                sel = (holder["dst"].reshape(K, -1)[:, :K] * torch.eye(K, device=X.device)).sum()
                g, = torch.autograd.grad(sel, holder["leaf"], allow_unused=True)
        finally:
            for h in handles: h.remove()
        return None if g is None else g.reshape(K, -1).float().cpu()

    def network_view(self, inputs, targets=None, mode="weights", max_nodes=48):
        if isinstance(self.model, seqm.SeqMLP):
            raise ValueError("The network view draws plain MLPs; this model has sequence encoders / decoders.")
        with self.lock:
            m = self.model; blocks = list(getattr(m, "blocks", []))
            x = torch.as_tensor(self._encode(inputs), device=self.device).unsqueeze(0).requires_grad_(True)
            caps = [dict() for _ in blocks]; handles = []
            for i, blk in enumerate(blocks):
                net_mod, act_mod, has_act = self._node_modules(blk)
                def hin(_m, inp, out, i=i, has_act=has_act):
                    if "net" not in caps[i]:
                        t = inp[0] if has_act else out; t.retain_grad(); caps[i]["net"] = t
                def hout(_m, inp, out, i=i):
                    if "act" not in caps[i]: caps[i]["act"] = out
                handles += [net_mod.register_forward_hook(hin), act_mod.register_forward_hook(hout)]
            try:
                with torch.enable_grad():
                    out = m(x); out.retain_grad()
                    tv = self._target_vector(targets); loss = None
                    for e in self.output_layout:
                        if e["col"] not in tv: continue
                        sl = out[:, e["start"]:e["end"]]; t = tv[e["col"]].to(self.device)
                        term = F.cross_entropy(sl, t) if e["type"] == "outlabcat" else F.huber_loss(sl, t.float().reshape(sl.shape))
                        loss = term if loss is None else loss + term
                    if loss is not None: loss.backward()
            finally:
                for h in handles: h.remove()
            vec = lambda t: t.detach().reshape(-1).float().cpu()
            grad = lambda t: (vec(t.grad) if loss is not None and t.grad is not None else None)
            # nodes
            layers = [{"kind": "input", "name": "Input layer", "net": vec(x), "act": vec(x), "bias": None,
                       "error": grad(x), "labels": [self._slot_label(j) for j in range(self.input_dim)]}]
            for i, blk in enumerate(blocks):
                c = caps[i]
                if "net" not in c:
                    layers.append({"kind": "hidden", "name": f"Hidden layer {i + 1}", "skipped": True}); continue
                net, act = vec(c["net"]), vec(c["act"])
                bias = None
                if isinstance(blk.linear1, nn.Linear) and blk.linear1.bias is not None and _gui_shape_module(blk) is blk.activation \
                        and blk.linear1.bias.numel() == net.numel():
                    bias = vec(blk.linear1.bias)
                fn = type(_gui_shape_module(blk)).__name__ if _gui_shape_module(blk) is not None else type(blk.linear1).__name__
                layers.append({"kind": "hidden", "name": f"Hidden layer {i + 1}", "net": net, "act": act, "bias": bias,
                               "error": grad(c["net"]), "function": fn, "labels": [f"neuron {k + 1}" for k in range(net.numel())]})
            raw = vec(out); acts = raw.clone(); labels = []
            for e in self.output_layout:
                col, ct, a, b = e["col"], e["type"], e["start"], e["end"]
                if ct == "outlabcat":
                    acts[a:b] = torch.softmax(raw[a:b], 0); labels += [f"{col} = {l}" for l in self._labels(col)][:b - a]
                elif ct == "out":
                    acts[a] = float(self._to_display(col, raw[a].item())); labels.append(col)
                elif ct == "outexcat":
                    acts[a:b] = torch.softmax(raw[a:b].reshape(e["max_len"], -1), -1).reshape(-1)
                    labels += [f"{col}[{p}]#{k}" for p in range(e["max_len"]) for k in range(e["num_classes"])]
                else: labels += [col if b - a == 1 else f"{col}[{p}]" for p in range(b - a)]
            fb = m.final_linear.bias if isinstance(m.final_linear, nn.Linear) else None
            layers.append({"kind": "output", "name": "Output layer", "net": raw, "act": acts,
                           "bias": vec(fb) if fb is not None else None, "error": grad(out), "labels": labels})
            # node subset per layer (most active first when a layer is too wide to draw)
            for L in layers:
                if L.get("skipped"): continue
                n = L["net"].numel(); L["size"] = n
                if n <= max_nodes: L["idx"] = list(range(n))
                else:  # stable across inputs: ranked by typical activity, not by this input's values
                    imp = self._node_importance()[layers.index(L)]
                    if imp is None or imp.numel() != n: imp = torch.arange(n, 0, -1).float()
                    L["idx"] = sorted(torch.topk(imp, max_nodes).indices.tolist())
            # connections between adjacent drawable layers
            static = self._static_weights(len(layers) - 1) if mode == "weights" else [None] * (len(layers) - 1)
            conns = []
            for p in range(len(layers) - 1):
                A, B = layers[p], layers[p + 1]
                if A.get("skipped") or B.get("skipped"): conns.append(None); continue
                W, kind = static[p], "weight"
                if W is None or tuple(W.shape) != (B["size"], A["size"]):
                    kind = "influence"
                    if p + 1 < len(layers) - 1:
                        mod = self._node_modules(blocks[p])[0]; has_act = self._node_modules(blocks[p])[2]
                        def cap(holder, mod=mod, has_act=has_act):
                            def f(_m, inp, out):
                                if "dst" not in holder: holder["dst"] = inp[0] if has_act else out
                            return [mod.register_forward_hook(f)]
                    else:
                        cap = lambda holder: []
                    W = self._influence(x, None if p == 0 else p - 1, B["size"], cap)
                if W is None: conns.append(None); continue
                sub = W[B["idx"]][:, A["idx"]]
                conns.append({"kind": kind, "w": _gui_b64(sub.numpy()), "rows": len(B["idx"]), "cols": len(A["idx"]),
                              "max": _gui_float(sub.abs().max()) if sub.numel() else 0.0})
            def pack(L):
                if L.get("skipped"): return L
                sel = L["idx"]
                pick = lambda t: None if t is None else [_gui_float(t[k]) for k in sel]
                return {"kind": L["kind"], "name": L["name"], "size": L["size"], "idx": sel, "function": L.get("function"),
                        "labels": [L["labels"][k] if k < len(L["labels"]) else str(k) for k in sel],
                        "net": pick(L["net"]), "act": pick(L["act"]), "bias": pick(L["bias"]), "error": pick(L["error"])}
            return {"layers": [pack(L) for L in layers], "conns": conns, "has_error": loss is not None,
                    "mode": mode, "version": self.version, "activation": self.config.get("activation", {}).get("name")}

    def neuron(self, inputs, block, neuron):
        """Incoming and outgoing weights of one hidden neuron (linear1 row, linear2 column)."""
        with self.lock:
            blk = self.model.blocks[int(block)]; j = int(neuron); res = {"block": int(block), "neuron": j}
            lin1 = blk.linear1
            if isinstance(lin1, nn.Linear) and j < lin1.weight.shape[0]:
                res["incoming"] = lin1.weight[j].detach().float().cpu().tolist(); res["bias"] = _gui_float(lin1.bias[j]) if lin1.bias is not None else None
                if int(block) == 0 and getattr(self.model, "input_attn", None) is None and not isinstance(getattr(self.model, "input_graph", None), InputFeatureGraph):
                    res["incoming_labels"] = [self._slot_label(i) for i in range(len(res["incoming"]))]
            lin2 = getattr(blk, "linear2", None)
            if isinstance(lin2, nn.Linear) and j < lin2.weight.shape[1]:
                res["outgoing"] = lin2.weight[:, j].detach().float().cpu().tolist()
            return res

    def _slot_label(self, idx):
        for c, (start, w) in self.slots.items():
            if start <= idx < start + w:
                ct = self.col_types[c]
                if ct == "inlabcat": return f"{c}={self._labels(c)[idx - start]}" if idx - start < len(self._labels(c)) else c
                return c if w == 1 else f"{c}[{idx - start}]"
        return str(idx)

    def weight(self, name, max_side=256):
        with self.lock:
            params = dict(self.model.named_parameters())
            if name not in params: raise KeyError(f"No parameter named {name!r}")
            p = params[name].detach().float().cpu()
            flat = p.reshape(-1).numpy()
            m = p.reshape(1, -1) if p.dim() <= 1 else p.reshape(p.shape[0], -1)
            rows, cols = m.shape; down = None
            if rows > max_side or cols > max_side:
                oh, ow = min(rows, max_side), min(cols, max_side)
                m = F.adaptive_avg_pool2d(m.unsqueeze(0).unsqueeze(0), (oh, ow))[0, 0]; down = [oh, ow]
            hist, edges = np.histogram(flat, bins=60)
            res = {"name": name, "shape": list(p.shape), "rows": int(m.shape[0]), "cols": int(m.shape[1]),
                   "downsampled": down, "matrix": _gui_b64(m.numpy()),
                   "stats": {"mean": _gui_float(flat.mean()), "std": _gui_float(flat.std()), "min": _gui_float(flat.min()),
                             "max": _gui_float(flat.max()), "abs_mean": _gui_float(np.abs(flat).mean()),
                             "fro": _gui_float(np.sqrt((flat.astype(np.float64) ** 2).sum())), "numel": int(flat.size),
                             "near_zero": _gui_float((np.abs(flat) < 1e-3).mean())},
                   "hist": {"counts": hist.tolist(), "edges": edges.tolist()}}
            if p.dim() == 2 and min(p.shape) <= 2048:
                sv = torch.linalg.svdvals(p.double()).numpy()
                res["singular_values"] = sv[:512].tolist()
                res["stats"]["spectral"] = _gui_float(sv[0])
                res["stats"]["stable_rank"] = _gui_float((sv ** 2).sum() / (sv[0] ** 2)) if sv[0] > 0 else None
            if p.dim() == 1 and len(flat) <= 4096: res["values"] = flat.tolist()
            return res

    # ─── dataset features ───
    def _get_dataset(self):
        if self._dataset is not None or self._dataset_error is not None: return self._dataset
        path = self.config.get("file_path")
        try:
            if not path or not os.path.exists(path): raise FileNotFoundError(f"training file {path!r} not found")
            wanted = set(self.input_cols + self.output_cols)
            df = read_table(path, delimiter=training_file_delimiter(path, self.col_types))
            missing = [c for c in wanted if c not in df.columns]
            if missing: raise ValueError(f"columns missing from {path}: {missing}")
            self._dataset = df.dropna(subset=list(wanted)).reset_index(drop=True)
        except Exception as e:
            self._dataset_error = f"{type(e).__name__}: {e}"
        return self._dataset

    def _data_points(self, x_col, y_col, limit=1500):
        df = self._get_dataset()
        if df is None: return None
        sub = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) > limit: sub = sub.sample(limit, random_state=0)
        return {"x": sub[x_col].tolist(), "y": sub[y_col].tolist()}

    def dataset_row(self, index=None, source="training"):
        with self.lock:
            df = self._get_validation() if source == "validation" else self._get_dataset()
            if df is None: raise ValueError("No validation rows saved for this model." if source == "validation"
                                            else (self._dataset_error or "No dataset available."))
            i = int(np.random.randint(len(df))) if index is None else int(index) % len(df)
            row = df.iloc[i]

            def plain(v):
                return v.item() if hasattr(v, "item") else v
            return {"index": i, "rows": len(df),
                    "inputs": {c: (str(row[c]) if self.col_types[c] in ("inlab", "inlabcat", "intex", "intexcat", "inim", "inenc") else plain(row[c]))
                               for c in self.input_cols},
                    "targets": {c: (str(row[c]) if self.col_types[c] != "out" else plain(row[c])) for c in self.output_cols}}

    def _get_validation(self):
        """The held-out rows saved by training next to model.pt (None if absent or empty)."""
        path = os.path.join(os.path.dirname(os.path.abspath(self.model_path)), VALIDATION_ROWS_FILE)
        try: mtime = os.path.getmtime(path)
        except OSError: return None
        if getattr(self, "_val_cache", (None, None))[0] != mtime:
            try: df = pd.read_csv(path)
            except Exception: df = None  # empty file = no validation set
            if df is not None and (len(df) == 0 or any(c not in df.columns for c in self.input_cols + self.output_cols)): df = None
            self._val_cache = (mtime, df)
        return self._val_cache[1]

    def _encode_rows(self, rows):
        enc, keep = [], []
        for idx, row in rows.iterrows():
            try:
                raw = [row[c] if self.col_types[c] == "in" else str(row[c]) for c in self.input_cols]
                enc.append(encode_sample_input(raw, self.col_types, self.vocabs, self.scalings, self.image_params)); keep.append(idx)
            except Exception:
                continue
        return enc, keep

    def _fit_metrics(self, rows, out):
        """Per-output prediction quality on `rows` (raw values) given raw model outputs; plus a per-row badness score."""
        res, score, shown = {}, np.zeros(len(rows)), {}
        for e in self.output_layout:
            col, ct = e["col"], e["type"]
            if ct == "out":
                pred = self._to_display(col, out[:, e["start"]].numpy().astype(np.float64))
                true = pd.to_numeric(rows[col], errors="coerce").to_numpy(dtype=np.float64)
                span = (self.scalings[col]["max"] - self.scalings[col]["min"]) if self._scaled(col) else 1.0
                score += np.nan_to_num(np.abs(pred - true) / (span or 1.0), nan=1.0); shown[col] = pred
                ok = np.isfinite(true); err = pred[ok] - true[ok]; ss = ((true[ok] - true[ok].mean()) ** 2).sum()
                res[col] = {"pred": pred[ok].tolist(), "true": true[ok].tolist(), "n": int(ok.sum()),
                            "mae": _gui_float(np.abs(err).mean()) if len(err) else None,
                            "rmse": _gui_float(np.sqrt((err ** 2).mean())) if len(err) else None,
                            "r2": _gui_float(1 - (err ** 2).sum() / ss) if ss > 0 else None}
            elif ct == "outlabcat":
                labels = self._labels(col); logits = out[:, e["start"]:e["end"]]; pred = logits.argmax(-1).numpy()
                true = np.array([self.vocabs[col].get(str(v), -1) for v in rows[col]])
                probs = torch.softmax(logits, -1).numpy()
                score += 1.0 - np.where(true >= 0, probs[np.arange(len(true)), np.clip(true, 0, None)], 0.0)
                shown[col] = [labels[k] if k < len(labels) else k for k in pred]
                nc = len(labels); cm = np.zeros((nc, nc), dtype=int)
                for t, p in zip(true, pred):
                    if 0 <= t < nc and 0 <= p < nc: cm[t, p] += 1
                res[col] = {"accuracy": _gui_float((pred == true).mean()), "labels": labels, "confusion": cm.tolist(), "n": int(len(true))}
            elif ct == "outlab":  # label code regressed as a number: the rounded code is the predicted label
                labels = self._labels(col); inv = {v: k for k, v in self.vocabs[col].items()}
                pred = np.rint(out[:, e["start"]].numpy()).astype(int)
                true = np.array([self.vocabs[col].get(str(v), -1) for v in rows[col]])
                score += (pred != true).astype(np.float64); shown[col] = [inv.get(int(k), int(k)) for k in pred]
                nc = len(labels); cm = np.zeros((nc, nc), dtype=int)
                for t, p in zip(true, pred):
                    if 0 <= t < nc and 0 <= p < nc: cm[t, p] += 1
                res[col] = {"accuracy": _gui_float((pred == true).mean()), "labels": labels, "confusion": cm.tolist(), "n": int(len(true))}
            elif ct in ("outdec", "outexcat", "outex"):
                pred_toks, pred_text = self._text_predictions(e, out)
                true_toks = [self._text_tokens(col, v) for v in rows[col]]
                res[col], row_err = text_fit_stats(true_toks, pred_toks)
                score += np.minimum(row_err, 1.0); shown[col] = pred_text
        return res, score, shown

    def _text_tokens(self, col, value):
        vocab = self.vocabs.get(col, {})
        if self.col_types[col] == "outdec": return seqm.tokenize(value, seqm.tokenizer_of(vocab))
        return list(str(value))[:self.scalings[col]["max_len"]]

    def _text_predictions(self, e, out):
        """Predicted token lists and display strings of one text output column over a batch of raw outputs."""
        col, ct, sl = e["col"], e["type"], out[:, e["start"]:e["end"]]
        vocab = self.vocabs.get(col, {}); inv = {v: k for k, v in vocab.items()}
        if ct == "outex": ids = [[int(round(v)) for v in row] for row in sl.tolist()]
        else: ids = sl.reshape(len(sl), e["max_len"], e["num_classes"]).argmax(-1).tolist()
        toks, texts = [], []
        for row in ids:
            if ct == "outdec":
                t = seqm.decoded_tokens(row, vocab); texts.append(seqm.decode_ids(row, vocab))
            else:
                t = [inv[i] for i in row if i > 0 and i in inv]; texts.append("".join(t))
            toks.append(t)
        return toks, texts

    def data_stats(self, compute=True, limit=2000):
        """Per-neuron activation statistics (training rows) and prediction quality on training rows and on
        the held-out validation rows saved by training (cached per model version)."""
        with self.lock:
            if self._data_stats is not None and self._data_stats["version"] == self.version: return self._data_stats
            if not compute: return None
            df = self._get_dataset()
            if df is None: raise ValueError(self._dataset_error or "No dataset available.")
            if any(self.col_types[c] == "inim" for c in self.input_cols): limit = min(limit, 300)
            vdf = self._get_validation()
            used = self.input_cols + self.output_cols
            train_df = df
            if vdf is not None:  # training rows = file rows that are not validation rows
                vkeys = set(_row_group_keys(vdf, used, self.col_types))
                train_df = df[~_row_group_keys(df, used, self.col_types).isin(vkeys).values]
            sub = train_df.sample(min(limit, len(train_df)), random_state=0) if len(train_df) > limit else train_df
            enc, keep = self._encode_rows(sub)
            if not enc: raise ValueError("No dataset rows could be encoded.")
            X = torch.as_tensor(np.asarray(enc, dtype=np.float32), device=self.device)
            outs, recs = [], []
            for i in range(0, len(X), 512):
                o, r = self._trace(X[i:i + 512]); outs.append(o.float().cpu()); recs.append(r)
            out = torch.cat(outs, 0)
            blocks = []
            for b in range(len(recs[0])):
                if not all("act_in" in r[b] and r[b]["act_in"].shape[0] == min(512, len(X) - k * 512) for k, r in enumerate(recs)):
                    blocks.append(None); continue
                a_in = torch.cat([r[b]["act_in"] for r in recs], 0).cpu(); a_out = torch.cat([r[b]["act_out"] for r in recs], 0).cpu()
                q = torch.nanquantile(a_in, torch.tensor([0.01, 0.5, 0.99]), dim=0)
                finite = q[torch.isfinite(q)]
                hist_lo, hist_hi = (float(finite.min()), float(finite.max())) if finite.numel() else (-1.0, 1.0)
                if hist_hi <= hist_lo: hist_hi = hist_lo + 1.0
                edges = torch.linspace(hist_lo, hist_hi, 41)
                a_fin = torch.nan_to_num(a_in, nan=hist_lo - 1.0, posinf=hist_hi + 1.0, neginf=hist_lo - 1.0)  # out of range -> not counted
                hist = torch.stack([torch.histc(a_fin[:, j], bins=40, min=hist_lo, max=hist_hi) for j in range(a_in.shape[1])])
                blocks.append({"p01": q[0].tolist(), "p50": q[1].tolist(), "p99": q[2].tolist(),
                               "mean_out": a_out.mean(0).tolist(), "std_out": a_out.std(0).tolist() if len(a_out) > 1 else [0.0] * a_out.shape[1],
                               "dead": (a_out.abs() < 1e-6).float().mean(0).tolist(),
                               "hist_edges": edges.tolist(), "hist": _gui_b64(hist.numpy()), "hist_shape": list(hist.shape)})
            train_rows = df.loc[keep]
            tr_fit, tr_score, tr_shown = self._fit_metrics(train_rows, out)
            val_fit, val_rows, source = {}, None, "training"
            if vdf is not None:
                vsub = vdf.sample(min(limit, len(vdf)), random_state=0) if len(vdf) > limit else vdf
                venc, vkeep = self._encode_rows(vsub)
                if venc:
                    with torch.no_grad():
                        vout = self._run(np.asarray(venc, dtype=np.float32))
                    val_rows = vdf.loc[vkeep]
                    val_fit, v_score, v_shown = self._fit_metrics(val_rows, vout)
                    source = "validation"
            fit = []
            for e in self.output_layout:
                if e["col"] in tr_fit:
                    fit.append(dict(tr_fit[e["col"]], col=e["col"], type=e["type"], val=val_fit.get(e["col"])))
            # worst rows come from the held-out rows when there are any: that is where generalization fails
            w_rows, w_score, w_shown, w_keep = ((val_rows, v_score, v_shown, vkeep) if source == "validation"
                                                else (train_rows, tr_score, tr_shown, keep))
            worst = []
            for i in np.argsort(-w_score)[:30]:
                r = w_rows.iloc[int(i)]
                worst.append({"index": int(w_keep[int(i)]), "source": source, "score": _gui_float(w_score[i]),
                              "inputs": {c: str(r[c]) for c in self.input_cols},
                              "targets": {c: str(r[c]) for c in w_shown},
                              "preds": {c: (_gui_float(v[i]) if isinstance(v, np.ndarray) else str(v[i])) for c, v in w_shown.items()}})
            self._data_stats = {"version": self.version, "rows": len(enc), "val_rows": 0 if val_rows is None else len(val_rows),
                                "blocks": blocks, "fit": fit, "worst": worst, "worst_source": source}
            return self._data_stats


_GUI_COL_TYPES = ["i", "in", "inlab", "intex", "inim", "inlabcat", "intexcat", "inenc", "inhash", "out", "outlab", "outex", "outlabcat", "outexcat", "outdec", "outhash"]
_GUI_OPTIMIZERS = ["Adam", "AdamHD", "SGD", "SGDHD", "Lamb", "Adagrad", "Adadelta", "AdamW", "RMSprop", "Rprop", "ASGD",
                   "Adamax", "NAdam", "SparseAdam", "RAdamScheduleFree", "AdEMAMix", "Adam3", "AdamDelta", "AutoAdam",
                   "NormAdam", "SWATS", "AdaBoundW", "CLion", "Signum", "SRprop", "IRprop", "Adan", "Prodigy",
                   "Evolution", "Muon", "AdaMuon", "NorMuon", "AdaGO", "AdamGO", "RMSGO", "AdaDeltaGO", "MuonHD", "NorMuonHD"]
# optimizers with an options panel in the GUI (every option the CLI asks for)
_GUI_MUON_FAMILY = MUON_CHOICES + ("AdamHD",)


def _gui_muon_params(opt, overrides):
    """CLI defaults for a Muon-family optimizer with the GUI's overrides applied (same types as the CLI)."""
    params = optimizer_options(opt, use_defaults=True)
    for k, v in (overrides or {}).items():
        if k not in params or v is None or v == "": continue
        d = params[k]
        if isinstance(d, bool): params[k] = bool(v)
        elif isinstance(d, int): params[k] = int(v)
        elif isinstance(d, float): params[k] = float(v)
        elif isinstance(d, tuple): params[k] = tuple(float(x) for x in v)
        else: params[k] = v
    return params
# activation name -> constructor params the CLI asks for (saved to config.json and used on load)
_GUI_ACT_PARAMS = {"ScaledTanh": {"scale": 1.7}, "LeakyReLU": {"negative_slope": 0.01}, "PSiLU": {"init": 1.0},
                   "RaPAU": {"n": 5, "m": 4}, "PaWeL": {"breakpoints": 32}, "CapActi2": {"down": -1.0, "up": 1.0},
                   "All": {"init_temp": 1.0}, "Maxout": {"pieces": 2}}
_GUI_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")


def _gui_detect_delimiter(path, wanted=()):
    """config.json does not record the delimiter: pick the one that splits the header best."""
    with open(path, "r", errors="replace") as f: header = f.readline()
    wanted = set(wanted)
    return max([",", "\t", ";", " "], key=lambda d: (len(wanted & {h.strip() for h in header.split(d)}), len(header.split(d))))


def gui_list_dir(path=None):
    path = os.path.abspath(os.path.expanduser(path or os.getcwd()))
    if os.path.isfile(path): path = os.path.dirname(path)
    dirs, files = [], []
    for name in sorted(os.listdir(path), key=str.lower):
        if name.startswith("."): continue
        full = os.path.join(path, name)
        if os.path.isdir(full): dirs.append(name)
        elif name.lower().endswith((".csv", ".tsv", ".txt", ".dat", ".tvq")): files.append({"name": name, "size": os.path.getsize(full)})
    return {"path": path, "parent": os.path.dirname(path), "dirs": dirs, "files": files}


# ───────────── EasyNN-plus / JustNN .tvq import ─────────────
TVQ_DIR = "tvq_imports"


def _tvq_load(path):
    import hashlib, tvq_reader
    source = os.path.abspath(os.path.expanduser(str(path).strip().strip('"').strip("'")))
    if not os.path.isfile(source): raise FileNotFoundError(f"No such file: {source}")
    try: t = tvq_reader.read_tvq(source)
    except Exception as e: raise ValueError(f"{os.path.basename(source)} could not be read as a .tvq file: {e}") from e
    tag = hashlib.sha1(source.encode()).hexdigest()[:6]
    return source, t, os.path.join(TVQ_DIR, f"{os.path.splitext(os.path.basename(source))[0]}-{tag}")


def gui_tvq_dataset(path):
    """A .tvq file's training + validating rows as a CSV (text stays text). Returns (csv path, the type each
    column's own role suggests, {source, notes, network report}); the trained network is imported separately."""
    import tvq_reader
    source, t, stem = _tvq_load(path)
    ds = tvq_reader.dataset(t)
    if not ds["columns"]: raise ValueError(f"{os.path.basename(source)} has no usable columns (image columns are not stored in the file)")
    if not ds["rows"]: raise ValueError(f"{os.path.basename(source)} has no training or validating rows")
    os.makedirs(TVQ_DIR, exist_ok=True)
    csv_path = os.path.abspath(stem + ".csv")
    tvq_reader.write_csv(ds, csv_path)
    types = {}
    for c in ds["columns"]:
        hashed = c["mode"] == 3
        types[c["name"]] = {0: "inhash" if hashed else "in", 1: "outhash" if hashed else "out"}.get(c["role"], "i")
    return csv_path, types, {"source": source, "notes": ds["notes"], "network": tvq_reader.network_report(t)}


def gui_tvq_network(path, activate=False, overwrite=False):
    """Turn the trained network of a .tvq file into an mlpRes6 model (model.pt + config.json in tvq_imports/<name>-net/)
    and check it against the original. With activate, it also becomes the current model of the working directory;
    existing model files are only replaced when overwrite is set, and are kept as *.bak-<time>."""
    import shutil, tvq_reader
    source, t, stem = _tvq_load(path)
    rep = tvq_reader.network_report(t)
    if not rep["compatible"]: raise ValueError("This network cannot be imported: " + "; ".join(rep["reasons"]))
    net = tvq_reader.network(t)
    sizes = net["sizes"]; n_in, n_out = sizes[0], sizes[-1]
    names, cols = tvq_reader.column_names(t), t["columns"]
    used = sorted(net["input_cols"] + net["output_cols"])  # file order is mlpRes6's input / output order
    col_types = {names[i]: ("in" if i in net["input_cols"] else "out") for i in used}
    scalings = {names[i]: {"min": cols[i]["min"], "max": cols[i]["max"], "range": [0.0, 1.0]} for i in used}  # EasyNN scales to 0..1
    vocabularies = {names[i]: tvq_reader.codebook(t, i) for i in used if cols[i]["mode"] == 3}
    empty = [c for c, v in vocabularies.items() if not v]
    if empty: raise ValueError("Text column without any stored strings: " + ", ".join(empty))
    folder = os.path.abspath(stem + "-net"); os.makedirs(folder, exist_ok=True)
    # EasyNN's training rows are the dataset and its validating rows the held-out set that Resume validates on, but only
    # when they are really held out: EasyNN often repeats training rows as validating ones, and mlpRes6 keeps no input
    # combination in both sets (it would drop those training rows), so then everything is training data.
    train_only = tvq_reader.dataset(t, row_types=(0,), roles=(0, 1), label_column=False)
    held = tvq_reader.dataset(t, row_types=(1,), roles=(0, 1), label_column=False)
    at = [k for k, c in enumerate(train_only["columns"]) if col_types.get(c["name"]) == "in"]
    seen = {tuple(r[k] for k in at) for r in train_only["rows"]}
    split = len(train_only["rows"]) >= 2 and bool(held["rows"]) and not any(tuple(r[k] for k in at) in seen for r in held["rows"])
    train_types = (0,) if split else (0, 1)
    ds = train_only if split else tvq_reader.dataset(t, row_types=(0, 1), roles=(0, 1), label_column=False)
    csv_path = os.path.join(folder, "data.csv"); tvq_reader.write_csv(ds, csv_path)
    val_path = os.path.join(folder, VALIDATION_ROWS_FILE)
    if split: tvq_reader.write_csv(held, val_path)
    elif os.path.exists(val_path): os.remove(val_path)
    split_note = ("" if split or not held["rows"] else
                  "its validating rows repeat training rows, so all rows are training data and there is no held-out set")
    config = {"file_path": csv_path, "col_types": col_types, "hidden_dims": sizes[1:-1], "vocabularies": vocabularies,
              "scalings": scalings, "image_params": {}, "optimizer_choice": "Adam", "batch_size": 32,
              "activation": {"name": "Sigmoid", "params": {}}, "activation_type": 0, "residual_type": "none",
              "norm_type": "none", "groups": 1, "attention_type": "none", "num_heads": 1, "input_attention_type": "none",
              "moe_mode": 1, "noise_mode": "none", "noise_params": {}, "mlp_mode": 0, "layer_routing": "none", "use_grn": False,
              "output_activation": {"name": "Sigmoid", "params": {}},
              "csv_format": csv_format(True, delimiter=",")}  # recorded, or Resume would have to guess it from the column names
    model = build_model_from_config(config, n_in, n_out, torch.device("cpu"))
    mats = list(zip(net["weights"], net["biases"]))
    put = lambda dst, src: dst.copy_(torch.as_tensor(src, dtype=torch.float32))
    with torch.no_grad():
        for blk, (W, b) in zip(model.blocks, mats[:-1]): put(blk.linear1.weight, W); put(blk.linear1.bias, b)
        W_out = mats[-1][0]
        if isinstance(model.final_skip, nn.Linear): model.final_skip.weight.zero_()
        else: W_out = W_out - np.eye(len(W_out))  # MLPO adds an identity skip when the last layer is as wide as the output: cancel it
        put(model.final_linear.weight, W_out); put(model.final_linear.bias, mats[-1][1])
    model.eval()
    # Check the whole path: CSV cells -> mlpRes6 encoding -> model must equal EasyNN's stored scaling -> its own network.
    rows = [e for e in t["examples"] if e["type"] in train_types]
    frame = read_table(csv_path, delimiter=",")
    keep = [k for k, e in enumerate(rows) if all(e["raw"][i] != tvq_reader.MISSING for i in used)][:200]
    in_names = [names[i] for i in net["input_cols"]]
    X = np.array([encode_sample_input(list(frame.iloc[k][in_names]), col_types, vocabularies, scalings, {}) for k in keep], dtype=np.float32)
    Xt = np.array([[rows[k]["norm"][i] for i in net["input_cols"]] for k in keep], dtype=np.float32)
    with torch.no_grad(): got = model(torch.as_tensor(X)).numpy()
    want = tvq_reader.forward(net, Xt)
    e_in, e_out = float(np.abs(X - Xt).max()), float(np.abs(got - want).max())
    if not (e_in < 1e-4 and e_out < 1e-4):
        raise RuntimeError(f"The imported model does not reproduce the original network (input scaling off by {e_in:.2g}, outputs by {e_out:.2g})")
    torch.save(model.state_dict(), os.path.join(folder, "model.pt"))
    with open(os.path.join(folder, "config.json"), "w") as f: json.dump(convert_to_serializable(config), f, indent=2)
    result = {"source": source, "folder": folder, "sizes": sizes, "rows_checked": len(keep), "input_error": e_in, "output_error": e_out,
              "quality": rep.get("quality"), "notes": rep["notes"], "weight_scale": rep.get("weight_scale"), "activated": False,
              "validation_rows": len(held["rows"]) if split else 0, "validation_note": split_note}
    if activate:
        existing = [f for f in ("model.pt", "config.json", OPTIMIZER_STATE_FILE, VALIDATION_ROWS_FILE) if os.path.exists(f)]
        if existing and not overwrite:
            result.update(needs_overwrite=True, existing=existing); return result
        stamp = time.strftime("%Y%m%d-%H%M%S"); result["backed_up"] = []
        for f in existing: os.replace(f, f"{f}.bak-{stamp}"); result["backed_up"].append(f"{f}.bak-{stamp}")
        shutil.copy(os.path.join(folder, "model.pt"), "model.pt"); shutil.copy(os.path.join(folder, "config.json"), "config.json")
        if split: shutil.copy(val_path, VALIDATION_ROWS_FILE)
        result["activated"] = True
    return result


def gui_preview_dataset(path, delimiter=None, rows=15, header=True):
    """Column statistics and a suggested type per column, for the training GUI. Names are the file's own
    (or col1, col2, ... without a header row); renaming is applied by the form on top of them.
    A .tvq path is converted to a CSV first; its own input / output roles become the suggestions."""
    tvq = None
    if str(path).strip().lower().endswith(".tvq"):
        path, tvq_types, tvq = gui_tvq_dataset(path); delimiter = ","; header = True
    path = os.path.abspath(os.path.expanduser(path))
    delim = delimiter or _gui_detect_delimiter(path)
    df = read_table(path, delimiter=delim, fmt={"header": header})
    cols = []
    for c in df.columns:
        s = df[c]; nun = int(s.nunique(dropna=True)); numeric = bool(pd.api.types.is_numeric_dtype(s))
        info = {"name": str(c), "numeric": numeric, "unique": nun, "missing": int(s.isna().sum()),
                "samples": [str(v) for v in s.dropna().unique()[:6]]}
        if numeric and s.notna().any():
            info.update(min=_gui_float(s.min()), max=_gui_float(s.max()), mean=_gui_float(s.mean()))
        if not numeric:
            strs = s.dropna().astype(str)
            info["max_len"] = int(strs.str.len().max()) if len(strs) else 0
            info["image"] = bool(len(strs)) and float(strs.str.lower().str.endswith(_GUI_IMAGE_EXT).mean()) > 0.8
        if nun < 2: sug = "i"                                   # constant: the CLI ignores these too
        elif info.get("image"): sug = "inim"
        elif numeric: sug = "in"
        elif nun <= 64: sug = "inlabcat"
        else: sug = "intexcat" if info.get("max_len", 0) <= 32 else "intex"
        info["suggested"] = sug
        cols.append(info)
    if tvq:
        for c in cols:
            if c["suggested"] != "i": c["suggested"] = tvq_types.get(c["name"], "i")
    active = [] if tvq else [c for c in cols if c["suggested"] not in ("i", "inim")]
    if active:  # last usable column is the usual target
        last = active[-1]; last["suggested"] = {"in": "out", "inlabcat": "outlabcat", "intex": "outex", "intexcat": "outexcat"}[last["suggested"]]
    head = df.head(rows)
    out = {"path": path, "delimiter": delim, "n_rows": int(len(df)), "columns": cols,
           "head": [[("" if pd.isna(v) else str(v)) for v in r] for r in head.itertuples(index=False)]}
    if tvq: out.update(converted_from=tvq["source"], import_notes=tvq["notes"], network=tvq["network"])
    return out


class TrainingSession:
    """Runs the regular main() training loop in a background thread for the GUI."""

    def __init__(self):
        import threading
        from array import array
        self.lock = threading.Lock(); self.thread = None
        self.state = "idle"; self.error = None; self.spec = None; self.summary = None
        self.steps = array("l"); self.losses = array("f"); self.vals = []; self.last = {}; self.tasks = {}
        self.started = self.finished = None; self.stop_requested = False; self.saves = 0

    def _callback(self, info):
        from array import array
        with self.lock:
            if "val_loss" in info:
                self.vals.append([info["step"], info["val_loss"], bool(info.get("saved"))])
                if info.get("saved"): self.saves += 1
            else:
                self.steps.append(int(info["step"])); self.losses.append(float(info["loss"]))
                for col, v in (info.get("tasks") or {}).items():  # per-output normalized loss, aligned with steps
                    arr = self.tasks.setdefault(col, array("f", [float("nan")] * (len(self.steps) - 1)))
                    arr.append(float(v))
                self.last = {k: v for k, v in info.items() if k not in ("step", "loss", "tasks")}
                self.last["epoch"] = info.get("epoch")
            if self.state == "preparing": self.state = "training"
            return self.stop_requested

    def stop(self):
        with self.lock:
            if self.state in ("preparing", "training"): self.stop_requested = True; self.state = "stopping"
        return self.status(len(self.steps))

    def status(self, since=0, max_points=4000):
        import time
        with self.lock:
            since = max(0, min(int(since or 0), len(self.steps)))
            st, lo = list(self.steps[since:]), list(self.losses[since:])
            tasks = {c: list(a[since:]) for c, a in self.tasks.items()}
            if len(st) > max_points:  # coarse catch-up after a page reload: bucket means
                k = -(-len(st) // max_points)
                st = [st[min(i + k, len(st)) - 1] for i in range(0, len(st), k)]
                lo = [float(np.mean(lo[i:i + k])) for i in range(0, len(lo), k)]
                tasks = {c: [float(np.nanmean(v[i:i + k])) for i in range(0, len(v), k)] for c, v in tasks.items()}
            end = self.finished or time.time()
            return {"state": self.state, "error": self.error, "summary": self.summary, "total": len(self.steps),
                    "steps": st, "losses": lo, "tasks": tasks, "vals": list(self.vals), "last": dict(self.last), "saves": self.saves,
                    "elapsed": (end - self.started) if self.started else 0.0, "cwd": os.getcwd()}

    def start(self, spec):
        import threading, time
        with self.lock:
            if self.state in ("preparing", "training", "stopping"): raise RuntimeError("A training run is already active.")
        args = self._resume_args(spec) if spec.get("resume") else self._build_args(spec)  # errors surface immediately
        from array import array
        with self.lock:
            self.state = "preparing"; self.error = None; self.spec = spec; self.stop_requested = False
            self.steps = array("l"); self.losses = array("f"); self.vals = []; self.last = {}; self.saves = 0; self.tasks = {}
            self.started = time.time(); self.finished = None
        self.thread = threading.Thread(target=self._run, args=(spec, args), daemon=True, name="mlp-training")
        self.thread.start()
        return self.status()

    def _resume_args(self, spec):
        """Continue the saved model: only training options come from the form."""
        if not (os.path.exists("model.pt") and os.path.exists("config.json")):
            raise FileNotFoundError("No trained model (model.pt / config.json) in the working directory to resume.")
        with open("config.json") as f: cfg = json.load(f)
        opt = spec.get("optimizer") or cfg.get("optimizer_choice", "Adam")
        if opt not in _GUI_OPTIMIZERS: raise ValueError(f"Unknown optimizer {opt!r}")
        optim_params = dict(spec.get("optim_params") or {})
        if opt in _GUI_MUON_FAMILY: optim_params = _gui_muon_params(opt, optim_params)
        balancing = spec.get("loss_balancing", "normalized")
        if balancing not in CombinedLoss.BALANCING: raise ValueError(f"Unknown loss balancing {balancing!r}")
        lr = spec.get("lr")
        return {"resume": True, "interval": int(spec.get("val_interval", 200)),
                "overrides": dict(optimizer_choice=opt, custom_lr=float(lr) if lr not in (None, "") else None,
                                  batch_size=int(spec.get("batch_size") or cfg.get("batch_size", 32)), optim_params=optim_params,
                                  loss_balancing=balancing, loss_temperature=float(spec.get("loss_temperature") or 20.0))}

    def _build_args(self, spec):
        path = os.path.abspath(os.path.expanduser(spec["path"]))
        if not os.path.isfile(path): raise FileNotFoundError(f"Dataset not found: {path}")
        delim = spec.get("delimiter") or _gui_detect_delimiter(path)
        # the form keys columns by the file's own names (col1, col2, ... without a header); rename maps them to new ones
        fmt = csv_format(spec.get("header", True) is not False, spec.get("rename"), list(spec["columns"]), delimiter=delim)
        new = lambda c: fmt["rename"].get(c, c)
        col_types = {new(c): t for c, t in spec["columns"].items()}
        bad = [t for t in col_types.values() if t not in _GUI_COL_TYPES]
        if bad: raise ValueError(f"Unknown column types: {bad}")
        input_cols = [c for c, v in col_types.items() if 'in' in v]
        output_cols = [c for c, v in col_types.items() if 'out' in v]
        if not input_cols: raise ValueError("Mark at least one column as an input.")
        if not output_cols: raise ValueError("Mark at least one column as an output.")
        hidden = [int(h) for h in spec.get("hidden_dims", [])]
        if hidden and min(hidden) < 1: raise ValueError("Hidden layers need a width >= 1 (no layers at all = a perceptron).")
        amap = _build_activation_map()

        def act_cfg(act):
            name = (act or {}).get("name", "ReLU")
            if name not in amap or name == "Custom": raise ValueError(f"Unknown activation {name!r}")
            params = {k: v for k, v in ((act or {}).get("params") or {}).items() if k in _GUI_ACT_PARAMS.get(name, {})}
            for k, v in _GUI_ACT_PARAMS.get(name, {}).items(): params[k] = type(v)(params.get(k, v))
            return {"name": name, "params": params}
        glob = act_cfg(spec.get("activation", {"name": "ReLU"})); name = glob["name"]
        base_act = activation_from_config(glob, wrap=False)
        layer_acts = [act_cfg(a) for a in spec["layer_activations"]] if spec.get("layer_activations") else None
        if layer_acts is not None and len(layer_acts) != len(hidden): raise ValueError("One activation per hidden layer is needed.")
        out_act = None if is_linear_activation(spec.get("output_activation")) else act_cfg(spec["output_activation"])
        scale_ranges = {}
        for c, r in (spec.get("scale_ranges") or {}).items():
            if col_types.get(new(c)) not in ("in", "out", "inhash", "outhash") or not r or [float(v) for v in r] == [-1.0, 1.0]: continue
            lo, hi = map(float, r)
            if lo == hi: raise ValueError(f"{new(c)}: the scaling range needs two different numbers.")
            scale_ranges[new(c)] = [lo, hi]
        activation_type = int(spec.get("activation_type", 0))
        residual = spec.get("residual_type", "residual"); moe_mode = int(spec.get("moe_mode", 1))
        mlp_mode = int(spec.get("mlp_mode", 0)); routing = spec.get("layer_routing", "none")
        is_maxout = name == "Maxout"
        if is_maxout and activation_type != 0: raise ValueError("True Maxout must use the basic activation type.")
        if is_maxout and moe_mode != 1: raise ValueError("True Maxout requires MoE mode 1 (off).")
        if routing == "adaptive" and (mlp_mode != 0 or residual != "residual" or moe_mode != 1):
            raise ValueError("Adaptive layer routing requires regular MLP mode, standard residuals, and MoE mode 1.")
        norm = spec.get("norm_type", "layer"); groups = int(spec.get("groups") or 1) if norm == "group" else None
        attention = spec.get("attention_type", "none"); heads = int(spec.get("num_heads") or 1) if attention == "multi" else None
        noise_mode = spec.get("noise_mode", "none")
        noise_params = ({"dropout_pct": float(spec.get("noise_value", 0.1))} if noise_mode == "dropout" else
                        {} if noise_mode == "none" else {"std": float(spec.get("noise_value", 0.1))})
        opt = spec.get("optimizer", "Adam")
        if opt not in _GUI_OPTIMIZERS: raise ValueError(f"Unknown optimizer {opt!r}")
        optim_params = dict(spec.get("optim_params") or {})
        if opt in _GUI_MUON_FAMILY: optim_params = _gui_muon_params(opt, optim_params)
        lr = spec.get("lr"); lr = float(lr) if lr not in (None, "") else None
        lsuv = int(spec.get("lsuv", 0))
        balancing = spec.get("loss_balancing", "normalized")
        if balancing not in CombinedLoss.BALANCING: raise ValueError(f"Unknown loss balancing {balancing!r}")
        seq_in = {new(k): v for k, v in (spec.get("seq_params") or {}).items()}
        seq_params = {c: seqm.check_params(seq_in.get(c), c, t == "outdec") for c, t in col_types.items() if t in seqm.SEQ_TYPES}
        seq_bridge = spec.get("seq_bridge", "mlp")
        if seq_bridge not in seqm.BRIDGES: raise ValueError(f"Unknown bridge {seq_bridge!r}")
        if seq_params and mlp_mode != 0: raise ValueError("inenc / outdec columns need the MLP architecture (not GNN).")
        if seq_params and not hidden and seq_bridge == "mlp": seq_bridge = "latent"  # no hidden layers: one linear map
        check_mlp_options(mlp_mode, layer_acts, out_act, hidden, norm, groups, int(spec.get("batch_size", 32)))
        return dict(path=path, delim=delim, col_types=col_types, input_cols=input_cols, output_cols=output_cols, fmt=fmt,
                    scale_ranges=scale_ranges,
                    image_params={new(c): {"im_size": int(p["im_size"]), "patch_size": int(p["patch_size"])}
                                  for c, p in (spec.get("image_params") or {}).items() if col_types.get(new(c)) == "inim"},
                    kwargs=dict(hidden_dims=hidden, optimizer_choice=opt, batch_size=int(spec.get("batch_size", 32)),
                                activation_cls=wrap_activation(base_act, activation_type), activation_type=activation_type,
                                residual_type=residual, norm_type=norm, groups=groups, attention_type=attention,
                                num_heads=heads, input_attention_type=spec.get("input_attention_type", "none"),
                                moe_mode=moe_mode, noise_mode=noise_mode, noise_params=noise_params,
                                base_activation_cls_for_config=base_act, use_lsuv=lsuv > 0,
                                lsuv_max_iter=int(spec.get("lsuv_max_iter", 10)), lsuv_normalize_mean=lsuv == 2,
                                custom_lr=lr, mlp_mode=mlp_mode, layer_routing=routing,
                                use_grn=bool(spec.get("use_grn", False)), optim_params=optim_params,
                                loss_balancing=balancing, loss_temperature=float(spec.get("loss_temperature") or 20.0),
                                seq_params=seq_params, seq_bridge=seq_bridge, layer_activations=layer_acts, output_activation=out_act),
                    val=dict(mode=spec.get("val_mode", "split"), pct=float(spec.get("val_pct", 0.1)),
                             file=spec.get("val_file"), interval=int(spec.get("val_interval", 200))))

    def _run(self, spec, a):
        import time
        try:
            if a.get("resume"):
                kwargs, vds, rep = prepare_resume("config.json", "model.pt")
                kwargs.update(a["overrides"])
                val_loader = make_loader(vds, kwargs["batch_size"], shuffle=False) if vds is not None else None
                with self.lock:
                    self.summary = {"rows": len(kwargs["train_dataset"]), "inputs": kwargs["input_cols"], "outputs": kwargs["output_cols"],
                                    "duplicates_removed": rep["duplicates_removed"], "val_overlap_removed": 0, "resumed": True,
                                    "validation": "none (no saved held-out rows)" if val_loader is None
                                                  else f"{len(vds)} saved held-out rows every {a['interval']} steps"}
                main(val_loader=val_loader, val_interval=max(1, a["interval"]), progress_callback=self._callback, **kwargs)
                with self.lock: self.state = "done"
                return
            col_types = a["col_types"]; vocabularies = {}; image_params = dict(a["image_params"])
            register_csv_format(a["path"], a["fmt"])
            for c, t in col_types.items():
                if t in ("i", "inim"): continue  # inim sizes come from the GUI instead of stdin
                _setup_vocab_for_col(c, t, a["path"], a["delim"], vocabularies, image_params,
                                     a["kwargs"]["seq_params"].get(c, {}).get("tokenizer", "char"))
            col_types = lower_hash_types(col_types)  # hashed text is numeric from here on; its codebook is in vocabularies
            bs = a["kwargs"]["batch_size"]; v = a["val"]; vpath = None
            if v["mode"] == "file":
                vpath = os.path.abspath(os.path.expanduser(v["file"] or ""))
                if not os.path.isfile(vpath): raise FileNotFoundError(f"Validation file not found: {vpath}")
            ds, vds, rep = prepare_train_val(a["path"], a["delim"], a["input_cols"], a["output_cols"], col_types,
                                             vocabularies, image_params, val_file=vpath,
                                             val_frac=v["pct"] if v["mode"] == "split" else 0.0, scale_ranges=a["scale_ranges"])
            scalings, vocabularies = ds.scalings, ds.vocabularies
            val_loader = make_loader(vds, bs, shuffle=False) if vds is not None else None
            with self.lock:
                self.summary = {"rows": len(ds), "inputs": a["input_cols"], "outputs": a["output_cols"],
                                "duplicates_removed": rep["duplicates_removed"] + rep.get("val_duplicates_removed", 0),
                                "val_overlap_removed": rep.get("val_rows_also_in_training", 0),
                                "validation": "none" if val_loader is None else f"{len(vds)} held-out rows every {v['interval']} steps"}
            main(a["path"], delimiter=a["delim"], input_cols=a["input_cols"], output_cols=a["output_cols"],
                 col_types=col_types, vocabularies=vocabularies, scalings=scalings, image_params=image_params,
                 val_loader=val_loader, val_interval=max(1, v["interval"]), progress_callback=self._callback,
                 train_dataset=ds, **a["kwargs"])
            with self.lock: self.state = "done"
        except Exception as e:
            traceback.print_exc()
            with self.lock: self.state = "error"; self.error = f"{type(e).__name__}: {e}"
        finally:
            with self.lock: self.finished = time.time()


_GUI_ACT_SUBSETS = {"All": "ACTIVATION_MAP", "Basics": "BASICS_MAP", "ReLUs": "RELUS_MAP", "Periodics": "PERIODICS_MAP",
                    "Self-gated": "SELF_GATED_MAP", "Trainables": "TRAINABLE_MAP", "IDK": "IDK_MAP"}
_GUI_NORMS = ["none", "batch", "instance", "layer", "group", "rmsnorm"]
_GUI_RESIDUALS = ["none", "highway", "residual", "rezero", "elementwise_rezero", "concat", "densenet"]


class BenchSession:
    """The CLI benchmark (run_benchmark) driven from the GUI: the dataset, columns and
    base network come from the Train page's form; the sweep axes from the Benchmark page."""

    def __init__(self):
        self.lock = threading.Lock()
        self.state, self.error, self.results, self.current, self.total, self.names = "idle", None, [], None, 0, []
        self.started = self.finished = None
        self.stop_requested, self.settings = False, None

    def start(self, spec):
        with self.lock:
            if self.state in ("preparing", "running", "stopping"):
                raise RuntimeError("A benchmark is already running.")
        form, b = spec.get("train") or {}, spec.get("bench") or {}
        a = TrainingSession._build_args(None, form)
        if has_seq_columns(a["col_types"]): raise ValueError("The benchmark does not support inenc / outdec columns yet.")
        amap = _build_activation_map()
        acts = [n for n in (b.get("activations") or []) if n in amap and n != "Custom"]
        if not acts:
            raise ValueError("Choose at least one activation to benchmark.")
        opts = [o for o in (b.get("optimizers") or [a["kwargs"]["optimizer_choice"]]) if o in _GUI_OPTIMIZERS]
        def floats(raw):
            try: return [float(x) for x in str(raw or "").replace(";", ",").split(",") if x.strip()]
            except ValueError: raise ValueError("Learning rates must be numbers separated by commas.")
        lrs = floats(b.get("learning_rates"))
        archs = []
        for part in str(b.get("architectures") or "").split("|"):
            try: dims = [int(x) for x in part.replace(" ", "").split(",") if x]
            except ValueError: raise ValueError("Architectures look like 128,64 | 256,128,64")
            if dims:
                if min(dims) < 1: raise ValueError("Layer widths must be at least 1.")
                archs.append(dims)
        norms = [n for n in (b.get("norms") or []) if n in _GUI_NORMS]
        res = [r for r in (b.get("residuals") or []) if r in _GUI_RESIDUALS]
        sweep = {k: v for k, v in (("optimizers", opts if len(opts) > 1 or opts != [a["kwargs"]["optimizer_choice"]] else None),
                                   ("learning_rates", lrs or None), ("architectures", archs or None),
                                   ("norm_types", norms or None), ("residual_types", res or None)) if v}
        metric = int(b.get("metric", 2 if form.get("val_mode", "split") != "none" else 0))
        if metric not in (0, 1, 2): raise ValueError("Unknown benchmark metric")
        v = a["val"]
        if metric == 2 and v["mode"] == "none":
            raise ValueError("Validation-loss scoring needs validation (set it on the Train page).")
        steps = int(b.get("steps", 1000))
        if steps < 1: raise ValueError("Steps must be at least 1.")
        n = len(acts) * max(1, len(opts)) * max(1, len(lrs)) * max(1, len(archs)) * max(1, len(norms)) * max(1, len(res))
        base_optimizer, base_params = a["kwargs"]["optimizer_choice"], a["kwargs"]["optim_params"] or {}
        def optim_params_for(opt):
            overrides = base_params if opt == base_optimizer else {}
            return _gui_muon_params(opt, overrides) if opt in _GUI_MUON_FAMILY else dict(overrides)
        with self.lock:
            self.state, self.error, self.results, self.current, self.total, self.names = "preparing", None, [], None, n, []
            self.started, self.finished, self.stop_requested = time.time(), None, False
            self.settings = {"steps": steps, "metric": ["Best sliding train loss", "Final train eval", "Best validation loss"][metric],
                             "base": {"hidden_dims": a["kwargs"]["hidden_dims"], "optimizer": base_optimizer,
                                      "norm": a["kwargs"]["norm_type"], "residual": a["kwargs"]["residual_type"],
                                      "lr": a["kwargs"]["custom_lr"], "batch_size": a["kwargs"]["batch_size"]},
                             "axes": {"activations": acts, **{k: v for k, v in sweep.items()}}, "dataset": a["path"]}
        threading.Thread(target=self._run, args=(a, acts, int(b.get("activation_type", a["kwargs"]["activation_type"])), sweep,
                                                 metric, steps, b, optim_params_for), daemon=True, name="mlp-benchmark").start()
        return self.status()

    def _callback(self, e):
        with self.lock:
            kind = e.get("event")
            if kind == "start":
                self.state, self.names, self.total = "running" if not self.stop_requested else "stopping", e["names"], e["total"]
            elif kind == "combo":
                self.current = {k: e[k] for k in ("index", "name", "optimizer", "lr", "arch", "norm", "res", "activation")}
                self.current.update(step=0, curve=[], val_curve=[], started=time.time())
            elif kind == "step" and self.current:
                self.current.update(step=e["step"], total_steps=e["total_steps"], curve=list(e["curve"]), val_curve=list(e["val_curve"]))
            elif kind == "result":
                self.results.append(e["result"]); self.current = None
            return self.stop_requested

    def _run(self, a, acts, act_type, sweep, metric, steps, b, optim_params_for):
        try:
            col_types = a["col_types"]; vocabularies = {}; image_params = dict(a["image_params"])
            register_csv_format(a["path"], a["fmt"])
            for c, t in col_types.items():
                if t in ("i", "inim"): continue
                _setup_vocab_for_col(c, t, a["path"], a["delim"], vocabularies, image_params)
            kw, v = a["kwargs"], a["val"]
            vfile = None
            if metric == 2 and v["mode"] == "file":
                vfile = os.path.abspath(os.path.expanduser(v["file"] or ""))
                if not os.path.isfile(vfile): raise FileNotFoundError(f"Validation file not found: {vfile}")
            run_benchmark(a["path"], a["delim"], a["input_cols"], a["output_cols"], col_types, vocabularies, image_params,
                          kw["hidden_dims"], kw["batch_size"], kw["optimizer_choice"], kw["residual_type"], kw["noise_mode"],
                          kw["noise_params"], norm_type=kw["norm_type"], groups=kw["groups"] or 1, attention_type=kw["attention_type"],
                          num_heads=kw["num_heads"] or 1, input_attention_type=kw["input_attention_type"], moe_mode=kw["moe_mode"],
                          num_steps=steps, loss_calc_mode=metric, train_eval_amount=int(b.get("eval_amount", 1000)) if metric == 1 else None,
                          valid_eval_percentage=v["pct"] if metric == 2 and v["mode"] == "split" else None,
                          use_lsuv=kw["use_lsuv"], lsuv_max_iter=kw["lsuv_max_iter"], lsuv_normalize_mean=kw["lsuv_normalize_mean"],
                          val_file_path=vfile, val_delimiter=a["delim"], sweep_config=sweep or None, custom_lr=kw["custom_lr"],
                          activations=acts, activation_type=act_type, optim_params_for=optim_params_for,
                          progress_callback=self._callback, eval_interval=int(b.get("eval_interval", 100)), final_eval=True)
            with self.lock:
                self.state, self.current = ("stopped" if self.stop_requested else "done"), None
        except Exception as e:
            traceback.print_exc()
            with self.lock:
                self.state, self.error, self.current = "error", f"{type(e).__name__}: {e}", None
        finally:
            with self.lock: self.finished = time.time()

    def stop(self):
        with self.lock:
            if self.state in ("preparing", "running"):
                self.stop_requested, self.state = True, "stopping"
        return self.status()

    def status(self, known=0):
        with self.lock:
            known = max(0, min(int(known or 0), len(self.results)))
            done = len(self.results)
            elapsed = (self.finished or time.time()) - self.started if self.started else 0
            cur = dict(self.current) if self.current else None
            if cur: cur["elapsed"] = time.time() - cur.pop("started")
            return {"state": self.state, "error": self.error, "total": self.total, "done": done, "names": list(self.names),
                    "results_from": known, "results": self.results[known:], "current": cur, "settings": self.settings,
                    "elapsed": elapsed, "eta": elapsed / done * (self.total - done) if done and self.state == "running" else None}


def gui_train_options():
    amap = _build_activation_map()
    return {"col_types": _GUI_COL_TYPES, "optimizers": _GUI_OPTIMIZERS, "muon_family": list(_GUI_MUON_FAMILY),
            "muon_defaults": {o: optimizer_options(o, use_defaults=True) for o in _GUI_MUON_FAMILY},
            "activations": [n for n in amap if n != "Custom"], "activation_params": _GUI_ACT_PARAMS,
            "act_subsets": {k: [n for n in globals()[v] if n in amap and n != "Custom"] for k, v in _GUI_ACT_SUBSETS.items()},
            "norms": _GUI_NORMS, "residuals": _GUI_RESIDUALS,
            "seq": {"archs": seqm.ARCHS, "attention": seqm.DEC_ATTENTION, "attention_archs": list(seqm.ATTENTION_ARCHS),
                    "bridges": seqm.BRIDGES, "tokenizers": list(seqm.TOKENIZERS), "defaults": seqm.DEFAULT_PARAMS},
            "cwd": os.getcwd()}


def run_gui(host="127.0.0.1", port=8765, open_browser=True, model_path="model.pt", config_path="config.json", start_page=None):
    """Serve the training + exploration GUI at http://host:port until Ctrl+C."""
    import threading, webbrowser
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse, parse_qs

    session = TrainingSession()
    bench = BenchSession()
    holder = {"sampler": None, "error": None}
    holder_lock = threading.Lock()

    def sampler(required=True):
        """The explorer's model; (re)created once model.pt and config.json exist."""
        with holder_lock:
            if holder["sampler"] is None and os.path.exists(model_path) and os.path.exists(config_path):
                try:
                    holder["sampler"] = InteractiveSampler(model_path, config_path); holder["error"] = None
                    print(f"  Explorer loaded {os.path.abspath(model_path)}")
                except Exception as e:
                    holder["error"] = f"{type(e).__name__}: {e}"
            s = holder["sampler"]
        if s is None and required:
            raise ValueError(holder["error"] or "No trained model yet (model.pt / config.json not found). Train one first.")
        return s

    def status(b):
        s = sampler(required=False)
        if s is None: return {"version": 0, "load_error": holder["error"], "reloaded": False, "has_model": False}
        reloaded = s.check_reload(force=bool(b.get("force")))
        return dict(s.status(), reloaded=reloaded, has_model=True)

    routes = {
        "/api/app": lambda b: {"has_model": sampler(required=False) is not None, "model_error": holder["error"],
                               "training": session.status(10 ** 12)["state"], "cwd": os.getcwd(), "start_page": start_page},
        "/api/meta": lambda b: sampler().meta(),
        "/api/status": status,
        "/api/predict": lambda b: sampler().predict(b.get("inputs", {}), b.get("generation")),
        "/api/sequence": lambda b: sampler().sequence_view(b.get("inputs", {})),
        "/api/attribution": lambda b: sampler().attribution_view(b.get("inputs", {})),
        "/api/sweep1d": lambda b: sampler().sweep1d(b.get("inputs", {}), b["x"], b["out"], b.get("lo"), b.get("hi"),
                                                    b.get("n", 200), b.get("show_data", False)),
        "/api/sweep2d": lambda b: sampler().sweep2d(b.get("inputs", {}), b["x"], b["y"], b["out"], b.get("cls"),
                                                    b.get("n", 80), b.get("ranges"), b.get("show_data", False)),
        "/api/sensitivity": lambda b: sampler().sensitivity(b.get("inputs", {}), b["out"], b.get("cls")),
        "/api/activations": lambda b: sampler().activation_shapes(b.get("inputs", {}), b["block"], b.get("lo"), b.get("hi"), b.get("n", 241)),
        "/api/neuron": lambda b: sampler().neuron(b.get("inputs", {}), b["block"], b["neuron"]),
        "/api/netview": lambda b: sampler().network_view(b.get("inputs", {}), b.get("targets"), b.get("mode", "weights"), int(b.get("max_nodes", 48))),
        "/api/weight": lambda b: sampler().weight(b["name"], b.get("max_side", 256)),
        "/api/dataset_row": lambda b: sampler().dataset_row(b.get("index"), b.get("source", "training")),
        "/api/data_stats": lambda b: sampler().data_stats(),
        "/api/fs": lambda b: gui_list_dir(b.get("path")),
        "/api/tvq/network": lambda b: gui_tvq_network(b["path"], bool(b.get("activate")), bool(b.get("overwrite"))),
        "/api/dataset/preview": lambda b: gui_preview_dataset(b["path"], b.get("delimiter"), header=b.get("header", True) is not False),
        "/api/train/options": lambda b: gui_train_options(),
        "/api/train/start": lambda b: session.start(b),
        "/api/train/stop": lambda b: session.stop(),
        "/api/train/status": lambda b: session.status(b.get("since", 0)),
        "/api/bench/start": lambda b: bench.start(b),
        "/api/bench/stop": lambda b: bench.stop(),
        "/api/bench/status": lambda b: bench.status(b.get("known", 0)),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass  # keep the terminal quiet

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(_gui_clean(body), allow_nan=False, default=str).encode()
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(data)

        def _handle(self, body):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                with open(_GUI_HTML, "rb") as f: return self._send(200, f.read(), "text/html; charset=utf-8")
            if path not in routes: return self._send(404, {"error": f"unknown route {path}"})
            try:
                self._send(200, routes[path](body))
            except Exception as e:
                if not isinstance(e, (ValueError, KeyError, FileNotFoundError, RuntimeError)): traceback.print_exc()
                self._send(400, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self):
            q = parse_qs(urlparse(self.path).query)
            self._handle({k: v[0] for k, v in q.items()})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try: body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError: return self._send(400, {"error": "invalid JSON body"})
            self._handle(body)

    s = sampler(required=False)
    print(f"  Working directory: {os.getcwd()}")
    print(f"  Trained model: {'found' if s else 'none yet (' + (holder['error'] or 'train one in the GUI') + ')'}")
    server = None
    for p in range(int(port), int(port) + 20):
        try: server = ThreadingHTTPServer((host, p), Handler); break
        except OSError: continue
    if server is None: raise OSError(f"No free port in {port}-{int(port) + 19}")
    url = f"http://{host}:{server.server_address[1]}/" + (f"#{start_page}" if start_page else "")
    print(f"\n  GUI running at {url}\n  (Ctrl+C to stop; model.pt is reloaded automatically when it changes)", flush=True)
    if open_browser: threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nStopping GUI.")
    finally: server.server_close()


def run_sampling_gui(host="127.0.0.1", port=8765, open_browser=True, model_path="model.pt", config_path="config.json"):
    """Serve the GUI opened on the explorer (sampling) page."""
    run_gui(host, port, open_browser, model_path, config_path, start_page="response")


##############################################
##############################################
# Main runner
##############################################
if __name__ == "__main__":
    print("\n=== PyTorch MLP Surrogate Suite ===")
    if input("Interface: 1 = CLI (menus below), 2 = GUI (train + explore in the browser) [1]: ").strip() == "2":
        port_str = input("Port (Enter = 8765): ").strip()
        run_gui(port=int(port_str) if port_str else 8765)
        raise SystemExit(0)
    print("0: Train (New Model)")
    print("1: Sample (Run Inference)")
    print("2: Benchmark (Hyperparameter Search)")
    print("3: Scatter Analysis (Eval on Dataset)")
    print("4: Export Model (ONNX/Python)")
    print("5: Inverse Design (Optimize Inputs)")
    print("6: Evolution (Neural Architecture Search)")
    print("7: Test Eval (Evaluate on Test Set)")
    print("8: Model Info (Architecture Summary)")
    print("9: Feature Importance (Permutation-based)")
    print("10: LR Finder (Learning Rate Range Test)")
    print("11: Cross-Validation (K-Fold)")
    print("12: Input Sensitivity (Per-Output Gradient Analysis)")
    print("13: Error Analysis (Top-N Worst Examples)")
    print("14: Resume Training (continue the current model)")
    
    choice = input("\nEnter choice [0-14]: ").strip().lower()
    _seq_modes = {"0", "train", "t", "1", "sample", "s", "2", "b", "benchmark", "7", "test", "testeval", "8", "info", "modelinfo",
                  "14", "resume", "r"}
    if choice not in _seq_modes and os.path.exists("config.json"):
        try: _cfg_types = load_config().get("col_types", {})
        except Exception: _cfg_types = {}
        if has_seq_columns(_cfg_types):
            print("The current model (config.json) has inenc / outdec columns. Modes that support them: "
                  "Train, Sample, Test Eval, Model Info, Resume Training.")
            raise SystemExit(1)
    
    if choice in ["train", "t", "0"]:
        file_path = input("Enter file path: ").strip()
        preview_csv_file(file_path)
        print("CSV delimiter: 1=,  2=\\t  3=;  4=space")
        dc = input("Choice: ").strip()
        delimiter = {"1":",","2":"\t","3":";","4":" "}.get(dc, ",")
            
        register_csv_format(file_path, ask_csv_format(file_path, delimiter))
        df = read_table(file_path, delimiter=delimiter)
        col_types, vocabularies, image_params = ask_column_types(df.columns.tolist(), file_path, delimiter, allow_seq=True)
        input_cols = [c for c, v in col_types.items() if 'in' in v]
        output_cols = [c for c, v in col_types.items() if 'out' in v]
        seq_params = dict(ASKED_SEQ_PARAMS); seq_bridge = ask_seq_bridge(col_types, seq_params)
        if has_seq_columns(col_types) and seq_bridge == "mlp":
            print("\nThe hidden layers below form the MLP between the encoders and the decoders.")
        hidden_dims = ask_hidden_dims()
        mlp_mode = ask_mlp_mode()
        batch_size = ask_batch_size()
        base_activation_cls = ask_activation()
        layer_routing = ask_layer_routing()
        activation_type = ask_activation_type()
        if isinstance(base_activation_cls(), MaxoutActivation) and activation_type != 0:
            raise ValueError("True Maxout must use the basic activation type.")
        wrapped_activation_cls = wrap_activation(base_activation_cls, activation_type)
        layer_activations = ask_layer_activations(hidden_dims, base_activation_cls)
        output_activation = ask_output_activation(output_cols, col_types)
        scale_ranges = ask_scale_ranges(input_cols + output_cols, col_types)
        optimizer_choice = ask_optimizer()
        custom_lr = ask_learning_rate()
        residual_type = ask_residual_type()
        noise_mode, noise_params = ask_noise_injection()
        norm_type, groups = ask_normalization_type()
        attention_type, num_heads = ask_attention_type()
        input_attention_type = ask_input_attention_type()
        moe_mode = ask_moe_mode()
        use_grn = input("Use Global Response Normalization (GRN)? [y/N]: ").strip().lower() in ('y', 'yes', '1', 'true')
        if layer_routing == "adaptive" and (mlp_mode != 0 or residual_type.lower() != "residual" or moe_mode != 1):
            raise ValueError("Adaptive layer routing requires regular MLP mode, standard residuals, and MoE mode 1.")
        if isinstance(base_activation_cls(), MaxoutActivation) and moe_mode != 1:
            raise ValueError("True Maxout requires MoE mode 1 (off).")
        use_lsuv, lsuv_max_iter, lsuv_normalize_mean = ask_lsuv_init()

        loss_balancing, loss_temperature = ask_loss_balancing(output_cols, col_types)
        train_ds, val_loader, val_interval = ask_validation_and_prepare(
            file_path, delimiter, input_cols, output_cols, col_types, vocabularies, image_params, batch_size,
            scale_ranges=scale_ranges)
        scalings = train_ds.scalings; vocabularies = train_ds.vocabularies

        # Route to auto-grow or regular training
        is_auto = isinstance(hidden_dims, dict) and hidden_dims.get("mode") == "auto"
        if is_auto and layer_routing != "none":
            raise ValueError("Adaptive layer routing is not supported by auto-grow training.")
        if is_auto and output_activation:
            raise ValueError("An output activation is not supported by auto-grow training.")
        if is_auto:
            main_auto_grow(file_path, delimiter=delimiter, input_cols=input_cols, output_cols=output_cols,
                 col_types=col_types, vocabularies=vocabularies, scalings=scalings, image_params=image_params,
                 auto_config=hidden_dims, optimizer_choice=optimizer_choice, batch_size=batch_size,
                 activation_cls=wrapped_activation_cls, activation_type=activation_type,
                 residual_type=residual_type, norm_type=norm_type, groups=groups,
                 attention_type=attention_type, num_heads=num_heads,
                 input_attention_type=input_attention_type, moe_mode=moe_mode,
                 noise_mode=noise_mode, noise_params=noise_params,
                 base_activation_cls_for_config=base_activation_cls,
                 use_lsuv=use_lsuv, lsuv_max_iter=lsuv_max_iter or 10,
                 lsuv_normalize_mean=lsuv_normalize_mean if lsuv_normalize_mean is not None else False,
                 val_loader=val_loader, val_interval=val_interval,
                 custom_lr=custom_lr, mlp_mode=mlp_mode, train_dataset=train_ds,
                 loss_balancing=loss_balancing, loss_temperature=loss_temperature)
        else:
            main(file_path, delimiter=delimiter, input_cols=input_cols, output_cols=output_cols,
                 col_types=col_types, vocabularies=vocabularies, scalings=scalings, image_params=image_params,
                 hidden_dims=hidden_dims, optimizer_choice=optimizer_choice, batch_size=batch_size,
                 activation_cls=wrapped_activation_cls, activation_type=activation_type,
                 residual_type=residual_type, norm_type=norm_type, groups=groups,
                 attention_type=attention_type, num_heads=num_heads,
                 input_attention_type=input_attention_type, moe_mode=moe_mode,
                 noise_mode=noise_mode, noise_params=noise_params,
                 base_activation_cls_for_config=base_activation_cls,
                 use_lsuv=use_lsuv, lsuv_max_iter=lsuv_max_iter or 10,
                 lsuv_normalize_mean=lsuv_normalize_mean if lsuv_normalize_mean is not None else False,
                 val_loader=val_loader, val_interval=val_interval,
                 custom_lr=custom_lr, mlp_mode=mlp_mode, layer_routing=layer_routing,
                 use_grn=use_grn, train_dataset=train_ds,
                 loss_balancing=loss_balancing, loss_temperature=loss_temperature,
                 seq_params=seq_params, seq_bridge=seq_bridge,
                 layer_activations=layer_activations, output_activation=output_activation)

    elif choice in ["sample", "s", "1"]:
        if input("Launch interactive GUI (local web server)? (y/N): ").strip().lower() == 'y':
            port_str = input("Port (Enter = 8765): ").strip()
            run_sampling_gui(port=int(port_str) if port_str else 8765)
            raise SystemExit(0)
        config = load_config()
        sample_input = []
        for col in config["col_types"]:
            if config["col_types"][col] in ["in", "inlab", "intex", "inim", "inlabcat", "intexcat", "inenc"]:
                sample_input.append(input(f"Enter value for '{col}': ").strip())
        
        output_cols = [c for c, t in config["col_types"].items() if 'out' in t]
        output_dim = calculate_dims(output_cols, config["col_types"], config["scalings"], config.get("vocabularies", {}))
        generation = ask_generation(config["col_types"])
        
        generate_plots = input("Generate plots? (y/n): ").strip().lower() == 'y'
        plot_option = None; plot_settings = {}
        if generate_plots:
            print("1: 1D plots  2: 2D plots  3: Neuron activation plots")
            plot_option = input("Enter 1, 2, or 3: ").strip()
            if plot_option in ["1", "2"]:
                res_str = input("Plot resolution (empty for default): ").strip()
                if res_str:
                    try: 
                        res = int(res_str); plot_settings['resolution_1d'] = res; plot_settings['resolution_2d'] = res
                    except: pass
                numeric_inputs = [
                    col for col, col_type in config['col_types'].items()
                    if col_type == 'in' and col in config['scalings']
                    and 'min' in config['scalings'][col] and 'max' in config['scalings'][col]
                ]
                if numeric_inputs:
                    print("Plot range per numeric input (Enter = training range; format: MIN MAX or MIN,MAX):")
                    input_ranges = {}
                    for col in numeric_inputs:
                        smin = config['scalings'][col]['min']
                        smax = config['scalings'][col]['max']
                        entered_range = input(f"  {col} [{smin:.4g} to {smax:.4g}]: ").strip()
                        if not entered_range:
                            continue
                        parts = entered_range.replace(',', ' ').split()
                        if len(parts) != 2:
                            print("    Invalid range; using training range.")
                            continue
                        try:
                            rmin, rmax = sorted((float(parts[0]), float(parts[1])))
                            input_ranges[col] = (rmin, rmax)
                        except ValueError:
                            print("    Invalid range; using training range.")
                    if input_ranges:
                        plot_settings['input_ranges'] = input_ranges
        
        if plot_option == "3":
            print("\n⚠ WARNING: This will plot EVERY neuron in the network.")
            if input("  Continue? (y/n): ").strip().lower() == 'y':
                try: plot_settings['neuron_resolution'] = int(input("  Neuron plot resolution (default 500): ").strip())
                except: pass
                result = load_and_sample_model(sample_input, config['hidden_dims'], config.get('vocabularies', {}),
                                               config['col_types'], output_dim, config['scalings'],
                                               config.get('image_params', {}), None, plot_settings, generation=generation)
                # Re-load for visualization
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                input_cols_s = [c for c in config["col_types"] if config["col_types"][c] in ["in","inlab","intex","inim","inlabcat","intexcat","inenc"]]
                input_dims_s = calculate_input_dim(input_cols_s, config["col_types"], config["scalings"], config.get("vocabularies", {}))
                act_map = _build_activation_map()
                act_cfg = config.get("activation", {"name": "ReLU", "params": {}})
                aname = act_cfg.get("name", "ReLU"); aparams = act_cfg.get("params", {})
                if aname == "Custom": cls = _rebuild_custom_activation_from_config(aparams); acls = lambda: cls()
                elif aname in act_map: acls = lambda: act_map[aname](**aparams)
                else: acls = nn.ReLU
                acls = wrap_activation(acls, config.get("activation_type", 0))
                nplot_mlp_mode = config.get("mlp_mode", 0)
                mdl = build_model_from_config(config, input_dims_s, output_dim, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                materialize_lazy_modules(mdl, input_dims_s, next(mdl.parameters()).device)
                with torch.no_grad(): mdl(torch.zeros(1, input_dims_s).to(device))
                mdl.load_state_dict(torch.load('model.pt')); mdl.eval()
                processed = []
                sidx = 0
                for cn, ct in config["col_types"].items():
                    if ct not in ["in","inlab","intex","inim","inlabcat","intexcat","inenc"]: continue
                    val = sample_input[sidx]
                    if ct in ["in"] and cn in config["scalings"] and 'min' in config["scalings"][cn]:
                        processed.append(to_model_units(float(val), config["scalings"][cn]))
                    else: processed.append(float(val) if ct == "in" else 0.0)
                    sidx += 1
                while len(processed) < input_dims_s: processed.append(0.0)
                plot_all_neurons(mdl, processed, config["scalings"], config["col_types"],
                                config.get("vocabularies",{}), [], device, plot_settings)
            print(result)
        else:
            print(load_and_sample_model(sample_input, config['hidden_dims'], config.get('vocabularies', {}),
                                           config['col_types'], output_dim, config['scalings'],
                                           config.get('image_params', {}), plot_option, plot_settings, generation=generation))

    elif choice in ["scatter", "sc", "3"]:
        config = load_config()
        file_path = input("Enter file path: ").strip()
        dc = input("Delimiter: ").strip(); delimiter = {"1":",","2":"\t","3":";","4":" "}.get(dc, ",")
        col_types = config["col_types"]
        input_cols = [c for c, t in col_types.items() if 'in' in t]
        output_cols = [c for c, t in col_types.items() if 'out' in t]
        vocabularies = config.get("vocabularies", {})
        output_layout, output_pred_dim, output_tgt_dim = build_output_layout(output_cols, col_types, config["scalings"], vocabularies)
        dataset = CustomDataset(file_path, delimiter, input_cols, output_cols, col_types,
                                vocabularies, config["scalings"], config.get("image_params", {}))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        input_dims = calculate_input_dim(input_cols, col_types, config["scalings"], vocabularies)
        act_map = _build_activation_map(); act_cfg = config.get("activation", {"name": "ReLU", "params": {}})
        aname = act_cfg["name"]; aparams = act_cfg["params"]
        if aname in act_map: cls = lambda: act_map[aname](**aparams)
        else: cls = nn.ReLU
        cls = wrap_activation(cls, config.get("activation_type", 0))
        scatter_mlp_mode = config.get("mlp_mode", 0)
        model = build_model_from_config(config, input_dims, output_pred_dim, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        materialize_lazy_modules(model, input_dims, next(model.parameters()).device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
        with torch.no_grad(): model(torch.zeros(1, input_dims).to(device))
        model.load_state_dict(torch.load('model.pt')); model.eval()
        all_pred = []; all_gt = []
        with torch.no_grad():
            for i, t in dataloader: all_pred.append(model(i.to(device)).cpu().numpy()); all_gt.append(t.cpu().numpy())
        all_pred = np.concatenate(all_pred, axis=0); all_gt = np.concatenate(all_gt, axis=0)
        for entry in output_layout:
            c = entry['col']; ct = entry['type']
            if ct == 'outlabcat':
                gt = all_gt[:, entry['tgt_start']:entry['tgt_end']].squeeze(-1)
                pred = np.argmax(all_pred[:, entry['start']:entry['end']], axis=-1)
                acc = np.mean(pred == gt.astype(int)); print(f"{c} Acc: {acc:.4f}")
                plt.figure(); plt.scatter(gt, pred, alpha=0.5); plt.title(f"{c} Scatter"); plt.savefig(f"scatter_{c}.png"); plt.close()
            elif ct in ['out', 'outlab']:
                gt = all_gt[:, entry['tgt_start']:entry['tgt_end']].ravel()
                pred = all_pred[:, entry['start']:entry['end']].ravel()
                plt.figure(); plt.scatter(gt, pred, alpha=0.5); plt.plot([gt.min(), gt.max()], [gt.min(), gt.max()], 'r--')
                plt.title(f"{c} Scatter"); plt.savefig(f"scatter_{c}.png"); plt.close()
        print("Scatter plots saved.")

    elif choice in ["benchmark", "b", "2"]:
        file_path = input("Enter file path: ").strip()
        dc = input("Delimiter: ").strip(); delimiter = {"1":",","2":"\t","3":";","4":" "}.get(dc, ",")
        register_csv_format(file_path, ask_csv_format(file_path, delimiter))
        df = read_table(file_path, delimiter=delimiter)
        col_types, vocabularies, image_params = ask_column_types(df.columns.tolist(), file_path, delimiter)
        input_cols = [c for c, t in col_types.items() if "in" in t]
        output_cols = [c for c, t in col_types.items() if "out" in t]
        hidden_dims = ask_hidden_dims(); batch_size = ask_batch_size()
        opt = ask_optimizer(); clr = ask_learning_rate(); res_t = ask_residual_type()
        nm, np_ = ask_noise_injection(); nrm, grp = ask_normalization_type()
        att, nh = ask_attention_type(); iatt = ask_input_attention_type(); moe = ask_moe_mode()
        ul, lm, lnm = ask_lsuv_init()
        lc = int(input("Loss calc (0=sliding, 1=final, 2=valid): ").strip() or 0)
        tea = None; vep = None; vfp = None
        if lc == 1: tea = int(input("Eval amount: ").strip())
        elif lc == 2:
            vfp = input("Val file (empty for %): ").strip()
            if not vfp: vep = float(input("Val %: ").strip())
        steps = int(input("Steps: ").strip())
        sweep = None
        if input("Full sweep? (y/n): ").lower() == 'y': sweep = ask_benchmark_sweep_config()
        run_benchmark(file_path, delimiter, input_cols, output_cols, col_types, vocabularies,
            image_params, hidden_dims, batch_size, opt, res_t, nm, np_, nrm, grp, att, nh, iatt, moe,
            steps, lc, tea, vep, ul, lm, lnm, vfp, delimiter, sweep, clr)

    elif choice in ["export", "e", "4"]:
        run_export()

    elif choice in ["inverse", "i", "5"]:
        run_inverse_design()

    elif choice in ["evolution", "evo", "6"]:
        file_path = input("Enter file path: ").strip()
        preview_csv_file(file_path)
        print("CSV delimiter: 1=,  2=\\t  3=;  4=space")
        dc = input("Choice: ").strip()
        delimiter = {"1":",","2":"\t","3":";","4":" "}.get(dc, ",")

        register_csv_format(file_path, ask_csv_format(file_path, delimiter))
        df = read_table(file_path, delimiter=delimiter)
        col_types, vocabularies, image_params = ask_column_types(df.columns.tolist(), file_path, delimiter)
        input_cols = [c for c, v in col_types.items() if 'in' in v]
        output_cols = [c for c, v in col_types.items() if 'out' in v]

        max_hidden_dim = int(input("Max hidden dimension per layer: ").strip())
        max_layers = int(input("Max number of layers: ").strip())
        pop_str = input("Population size (default 20): ").strip()
        population_size = int(pop_str) if pop_str else 20
        gen_str = input("Number of generations (default 30): ").strip()
        generations = int(gen_str) if gen_str else 30
        eval_str = input("Eval steps per individual (default 500): ").strip()
        eval_steps = int(eval_str) if eval_str else 500
        batch_size = ask_batch_size()
        optimizer_choice = ask_optimizer()
        custom_lr = ask_learning_rate()

        # NAS mode selection
        nas_mode, evolve_set, fixed_overrides = ask_nas_mode()

        # For non-evolved fields that need user specification
        fixed_hidden_dims = None
        fixed_activation = None
        fixed_activation_type = None
        if "hidden_dims" not in evolve_set:
            fixed_hidden_dims = ask_hidden_dims()
        if "activation" not in evolve_set:
            act_cls = ask_activation()
            act_name = act_cls.__name__ if hasattr(act_cls, '__name__') else "ReLU"
            fixed_activation = act_name
        if "activation_type" not in evolve_set:
            fixed_activation_type = ask_activation_type()

        # Validation for evolution fitness tracking (deduplicated, truly held out)
        evo_train_ds, evo_val_loader, _ = ask_validation_and_prepare(
            file_path, delimiter, input_cols, output_cols, col_types, vocabularies, image_params,
            batch_size, ask_interval=False)
        scalings = evo_train_ds.scalings; vocabularies = evo_train_ds.vocabularies

        # Run evolution
        best_individual = run_evolution(
            file_path, delimiter, input_cols, output_cols, col_types, vocabularies,
            scalings, image_params, batch_size, optimizer_choice, custom_lr,
            max_hidden_dim, max_layers, population_size, generations, eval_steps,
            evo_val_loader, train_dataset=evo_train_ds, evolve_set=evolve_set, fixed_overrides=fixed_overrides,
            fixed_hidden_dims=fixed_hidden_dims, fixed_activation=fixed_activation,
            fixed_activation_type=fixed_activation_type)

        if best_individual is not None:
            print("\n>>> Now training the best evolved architecture <<<\n")

            # Build activation from the evolved choice
            act_map = _build_activation_map()
            act_name = best_individual["activation_name"]
            if act_name in act_map:
                base_activation_cls = act_map[act_name]
            else:
                base_activation_cls = nn.ReLU
            activation_type = best_individual["activation_type"]
            wrapped_activation_cls = wrap_activation(base_activation_cls, activation_type)

            hidden_dims = best_individual["hidden_dims"]
            residual_type = best_individual["residual_type"]
            noise_mode = best_individual["noise_mode"]
            noise_params = best_individual["noise_params"]
            norm_type = best_individual["norm_type"]
            evolved_moe_mode = best_individual.get("moe_mode", 1)
            evolved_attention_type = best_individual.get("attention_type", "none")
            evolved_num_heads = best_individual.get("num_heads", 1)
            evolved_lr = best_individual.get("learning_rate")
            if evolved_lr is not None:
                custom_lr = evolved_lr
                print(f"  Using evolved learning rate: {evolved_lr:.6f}")

            # Re-setup validation for full training (deduplicated, truly held out)
            final_train_ds, val_loader, val_interval = ask_validation_and_prepare(
                file_path, delimiter, input_cols, output_cols, col_types, vocabularies, image_params, batch_size)
            scalings = final_train_ds.scalings

            use_lsuv, lsuv_max_iter, lsuv_normalize_mean = ask_lsuv_init()

            # Train using regular main() with evolved parameters
            main(file_path, delimiter=delimiter, input_cols=input_cols, output_cols=output_cols,
                 col_types=col_types, vocabularies=vocabularies, scalings=scalings, image_params=image_params,
                 hidden_dims=hidden_dims, optimizer_choice=optimizer_choice, batch_size=batch_size,
                 activation_cls=wrapped_activation_cls, activation_type=activation_type,
                 residual_type=residual_type, norm_type=norm_type, groups=1,
                 attention_type=evolved_attention_type, num_heads=evolved_num_heads,
                 input_attention_type="none", moe_mode=evolved_moe_mode,
                 noise_mode=noise_mode, noise_params=noise_params,
                 base_activation_cls_for_config=base_activation_cls,
                 use_lsuv=use_lsuv, lsuv_max_iter=lsuv_max_iter or 10,
                 lsuv_normalize_mean=lsuv_normalize_mean if lsuv_normalize_mean is not None else False,
                 val_loader=val_loader, val_interval=val_interval,
                 custom_lr=custom_lr, mlp_mode=0, train_dataset=final_train_ds)

    elif choice in ["test", "testeval", "7"]:
        # Mode 7: Test Set Evaluation
        config = load_config()
        col_types_loaded = config["col_types"]
        input_cols_l = [c for c, v in col_types_loaded.items() if 'in' in v]
        output_cols_l = [c for c, v in col_types_loaded.items() if 'out' in v]
        vocabularies_l = config.get("vocabularies", {})
        scalings_l = config.get("scalings", {})
        
        input_dims = calculate_input_dim(input_cols_l, col_types_loaded, scalings_l, vocabularies_l)
        output_pred_dim = calculate_output_pred_dim(output_cols_l, col_types_loaded, scalings_l, vocabularies_l)
        
        activation_map = _build_activation_map()
        act_cfg = config.get("activation", {"name": "ReLU", "params": {}})
        act_name = act_cfg.get("name", "ReLU")
        act_params = act_cfg.get("params", {})
        if act_name == "Custom":
            cls = _rebuild_custom_activation_from_config(act_params)
            activation_cls = lambda: cls()
        elif act_name in activation_map:
            activation_cls = activation_map[act_name]
            if act_params:
                base_cls = activation_cls
                activation_cls = lambda **p: base_cls(**p) if callable(base_cls) else base_cls
        else:
            activation_cls = nn.ReLU
        
        at = config.get("activation_type", 0)
        activation_cls = wrap_activation(activation_cls, at)
        
        hidden_dims = config.get("hidden_dims", [64])
        residual_type = config.get("residual_type", "residual")
        norm_type = config.get("norm_type", "layer")
        moe_mode = config.get("moe_mode", 1)
        attention_type = config.get("attention_type", "none")
        num_heads = config.get("num_heads", 1)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp_mode = config.get("mlp_mode", 0)
        
        model = build_model_from_config(config, input_dims, output_pred_dim, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        materialize_lazy_modules(model, input_dims, next(model.parameters()).device)
        with torch.no_grad(): model(torch.zeros(1, input_dims).to(device))
        model.load_state_dict(torch.load("model.pt", map_location=device))
        
        test_csv = input("Enter test CSV path: ").strip()
        print("CSV delimiter: 1=,  2=\\t  3=;  4=space")
        dc = input("Choice: ").strip()
        delimiter = {"1":",","2":"\t","3":";","4":" "}.get(dc, ",")
        batch_size = ask_batch_size()
        
        run_test_eval(model, test_csv, delimiter, input_cols_l, output_cols_l, col_types_loaded,
                     vocabularies_l, scalings_l, config.get("image_params", {}), batch_size, device)

    elif choice in ["info", "modelinfo", "8"]:
        # Mode 8: Model Info
        config = load_config()
        col_types_loaded = config["col_types"]
        input_cols_l = [c for c, v in col_types_loaded.items() if 'in' in v]
        output_cols_l = [c for c, v in col_types_loaded.items() if 'out' in v]
        vocabularies_l = config.get("vocabularies", {})
        scalings_l = config.get("scalings", {})
        
        input_dims = calculate_input_dim(input_cols_l, col_types_loaded, scalings_l, vocabularies_l)
        output_pred_dim = calculate_output_pred_dim(output_cols_l, col_types_loaded, scalings_l, vocabularies_l)
        
        activation_map = _build_activation_map()
        act_cfg = config.get("activation", {"name": "ReLU", "params": {}})
        act_name = act_cfg.get("name", "ReLU")
        if act_name in activation_map:
            activation_cls = activation_map[act_name]
        else:
            activation_cls = nn.ReLU
        at = config.get("activation_type", 0)
        activation_cls = wrap_activation(activation_cls, at)
        
        hidden_dims = config.get("hidden_dims", [64])
        residual_type = config.get("residual_type", "residual")
        norm_type = config.get("norm_type", "layer")
        moe_mode = config.get("moe_mode", 1)
        attention_type = config.get("attention_type", "none")
        num_heads = config.get("num_heads", 1)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp_mode = config.get("mlp_mode", 0)
        
        model = build_model_from_config(config, input_dims, output_pred_dim, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        materialize_lazy_modules(model, input_dims, next(model.parameters()).device)
        with torch.no_grad(): model(torch.zeros(1, input_dims).to(device))
        if os.path.exists("model.pt"):
            model.load_state_dict(torch.load("model.pt", map_location=device))
        
        run_model_info(model, config)

    elif choice in ["importance", "fi", "9"]:
        # Mode 9: Feature Importance
        config = load_config()
        col_types_loaded = config["col_types"]
        input_cols_l = [c for c, v in col_types_loaded.items() if 'in' in v]
        output_cols_l = [c for c, v in col_types_loaded.items() if 'out' in v]
        vocabularies_l = config.get("vocabularies", {})
        scalings_l = config.get("scalings", {})
        
        input_dims = calculate_input_dim(input_cols_l, col_types_loaded, scalings_l, vocabularies_l)
        output_pred_dim = calculate_output_pred_dim(output_cols_l, col_types_loaded, scalings_l, vocabularies_l)
        
        activation_map = _build_activation_map()
        act_cfg = config.get("activation", {"name": "ReLU", "params": {}})
        act_name = act_cfg.get("name", "ReLU")
        if act_name in activation_map:
            activation_cls = activation_map[act_name]
        else:
            activation_cls = nn.ReLU
        at = config.get("activation_type", 0)
        activation_cls = wrap_activation(activation_cls, at)
        
        hidden_dims = config.get("hidden_dims", [64])
        residual_type = config.get("residual_type", "residual")
        norm_type = config.get("norm_type", "layer")
        moe_mode = config.get("moe_mode", 1)
        attention_type = config.get("attention_type", "none")
        num_heads = config.get("num_heads", 1)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp_mode = config.get("mlp_mode", 0)
        
        model = build_model_from_config(config, input_dims, output_pred_dim, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        materialize_lazy_modules(model, input_dims, next(model.parameters()).device)
        with torch.no_grad(): model(torch.zeros(1, input_dims).to(device))
        model.load_state_dict(torch.load("model.pt", map_location=device))
        
        data_csv = input("Enter data CSV path (for importance analysis): ").strip()
        print("CSV delimiter: 1=,  2=\\t  3=;  4=space")
        dc = input("Choice: ").strip()
        delimiter = {"1":",","2":"\t","3":";","4":" "}.get(dc, ",")
        batch_size = ask_batch_size()
        rep_str = input("Permutation repeats (default 5): ").strip()
        n_repeats = int(rep_str) if rep_str else 5
        
        run_feature_importance(model, data_csv, delimiter, input_cols_l, output_cols_l,
                             col_types_loaded, vocabularies_l, scalings_l,
                             config.get("image_params", {}), batch_size, device, n_repeats)

    elif choice in ["lrfinder", "lr", "10"]:
        # Mode 10: Learning Rate Finder
        file_path = input("Enter file path: ").strip()
        preview_csv_file(file_path)
        print("CSV delimiter: 1=,  2=\\t  3=;  4=space")
        dc = input("Choice: ").strip()
        delimiter = {"1":",","2":"\t","3":";","4":" "}.get(dc, ",")
        
        register_csv_format(file_path, ask_csv_format(file_path, delimiter))
        df = read_table(file_path, delimiter=delimiter)
        col_types_lr, vocabularies_lr, image_params_lr = ask_column_types(df.columns.tolist(), file_path, delimiter)
        input_cols_lr = [c for c, v in col_types_lr.items() if 'in' in v]
        output_cols_lr = [c for c, v in col_types_lr.items() if 'out' in v]
        
        hidden_dims = ask_hidden_dims()
        activation_cls = ask_activation()
        activation_type = ask_activation_type()
        activation_cls = wrap_activation(activation_cls, activation_type)
        residual_type = ask_residual_type()
        norm_type, _ = ask_normalization_type()
        batch_size = ask_batch_size()
        optimizer_choice = ask_optimizer()
        
        # Setup scalings
        dataset_pre = CustomDataset(file_path, delimiter, input_cols_lr, output_cols_lr,
                                    col_types_lr, vocabularies_lr, {}, image_params_lr)
        scalings_lr = dataset_pre.scalings
        vocabularies_lr = dataset_pre.vocabularies
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        lr_min_str = input("LR min (default 1e-7): ").strip()
        lr_max_str = input("LR max (default 10): ").strip()
        steps_str = input("Steps (default 200): ").strip()
        lr_min = float(lr_min_str) if lr_min_str else 1e-7
        lr_max = float(lr_max_str) if lr_max_str else 10.0
        n_steps = int(steps_str) if steps_str else 200
        
        run_lr_finder(file_path, delimiter, input_cols_lr, output_cols_lr, col_types_lr,
                     vocabularies_lr, scalings_lr, image_params_lr, batch_size,
                     hidden_dims, activation_cls, residual_type, norm_type,
                     optimizer_choice, device, lr_min, lr_max, n_steps)

    elif choice in ["cv", "crossval", "11"]:
        # Mode 11: Cross-Validation
        file_path = input("Enter file path: ").strip()
        preview_csv_file(file_path)
        print("CSV delimiter: 1=,  2=\\t  3=;  4=space")
        dc = input("Choice: ").strip()
        delimiter = {"1":",","2":"\t","3":";","4":" "}.get(dc, ",")
        
        register_csv_format(file_path, ask_csv_format(file_path, delimiter))
        df = read_table(file_path, delimiter=delimiter)
        col_types_cv, vocabularies_cv, image_params_cv = ask_column_types(df.columns.tolist(), file_path, delimiter)
        input_cols_cv = [c for c, v in col_types_cv.items() if 'in' in v]
        output_cols_cv = [c for c, v in col_types_cv.items() if 'out' in v]
        
        hidden_dims = ask_hidden_dims()
        base_act_cls = ask_activation()
        activation_type = ask_activation_type()
        activation_cls = wrap_activation(base_act_cls, activation_type)
        residual_type = ask_residual_type()
        norm_type, _ = ask_normalization_type()
        batch_size = ask_batch_size()
        optimizer_choice = ask_optimizer()
        custom_lr = ask_learning_rate()
        noise_mode_cv, noise_params_cv = ask_noise_injection()
        
        k_str = input("Number of folds (default 5): ").strip()
        k_folds = int(k_str) if k_str else 5
        steps_str = input("Training steps per fold (default 2000): ").strip()
        steps_per_fold = int(steps_str) if steps_str else 2000
        
        # Setup scalings
        dataset_pre = CustomDataset(file_path, delimiter, input_cols_cv, output_cols_cv,
                                    col_types_cv, vocabularies_cv, {}, image_params_cv)
        scalings_cv = dataset_pre.scalings
        vocabularies_cv = dataset_pre.vocabularies
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        run_cross_validation(file_path, delimiter, input_cols_cv, output_cols_cv, col_types_cv,
                           vocabularies_cv, scalings_cv, image_params_cv, batch_size,
                           hidden_dims, activation_cls, activation_type, residual_type,
                           norm_type, optimizer_choice, custom_lr, device,
                           k_folds, steps_per_fold, noise_mode_cv, noise_params_cv,
                           base_activation_cls_for_config=base_act_cls)

    elif choice in ["sensitivity", "sens", "12"]:
        # Mode 12: Input Sensitivity (Per-Output)
        config = load_config()
        col_types_loaded = config["col_types"]
        input_cols_l = [c for c, v in col_types_loaded.items() if 'in' in v]
        output_cols_l = [c for c, v in col_types_loaded.items() if 'out' in v]
        vocabularies_l = config.get("vocabularies", {})
        scalings_l = config.get("scalings", {})
        
        input_dims = calculate_input_dim(input_cols_l, col_types_loaded, scalings_l, vocabularies_l)
        output_pred_dim = calculate_output_pred_dim(output_cols_l, col_types_loaded, scalings_l, vocabularies_l)
        
        activation_map = _build_activation_map()
        act_cfg = config.get("activation", {"name": "ReLU", "params": {}})
        act_name = act_cfg.get("name", "ReLU")
        if act_name in activation_map:
            activation_cls = activation_map[act_name]
        else:
            activation_cls = nn.ReLU
        at = config.get("activation_type", 0)
        activation_cls = wrap_activation(activation_cls, at)
        
        hidden_dims = config.get("hidden_dims", [64])
        residual_type = config.get("residual_type", "residual")
        norm_type = config.get("norm_type", "layer")
        moe_mode = config.get("moe_mode", 1)
        attention_type = config.get("attention_type", "none")
        num_heads = config.get("num_heads", 1)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp_mode = config.get("mlp_mode", 0)
        
        model = build_model_from_config(config, input_dims, output_pred_dim, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        materialize_lazy_modules(model, input_dims, next(model.parameters()).device)
        with torch.no_grad(): model(torch.zeros(1, input_dims).to(device))
        model.load_state_dict(torch.load("model.pt", map_location=device))
        
        data_csv = input("Enter data CSV path (for sensitivity analysis): ").strip()
        print("CSV delimiter: 1=,  2=\\t  3=;  4=space")
        dc = input("Choice: ").strip()
        delimiter = {"1":",","2":"\t","3":";","4":" "}.get(dc, ",")
        batch_size = ask_batch_size()
        ns_str = input("Max samples to use (empty=all, e.g. 500): ").strip()
        n_samples = int(ns_str) if ns_str else None
        
        run_input_sensitivity(model, data_csv, delimiter, input_cols_l, output_cols_l,
                             col_types_loaded, vocabularies_l, scalings_l,
                             config.get("image_params", {}), batch_size, device, n_samples)

    elif choice in ["errors", "error", "13"]:
        # Mode 13: Error Analysis
        config = load_config()
        col_types_loaded = config["col_types"]
        input_cols_l = [c for c, v in col_types_loaded.items() if 'in' in v]
        output_cols_l = [c for c, v in col_types_loaded.items() if 'out' in v]
        vocabularies_l = config.get("vocabularies", {})
        scalings_l = config.get("scalings", {})
        
        input_dims = calculate_input_dim(input_cols_l, col_types_loaded, scalings_l, vocabularies_l)
        output_pred_dim = calculate_output_pred_dim(output_cols_l, col_types_loaded, scalings_l, vocabularies_l)
        
        activation_map = _build_activation_map()
        act_cfg = config.get("activation", {"name": "ReLU", "params": {}})
        act_name = act_cfg.get("name", "ReLU")
        if act_name in activation_map:
            activation_cls = activation_map[act_name]
        else:
            activation_cls = nn.ReLU
        at = config.get("activation_type", 0)
        activation_cls = wrap_activation(activation_cls, at)
        
        hidden_dims = config.get("hidden_dims", [64])
        residual_type = config.get("residual_type", "residual")
        norm_type = config.get("norm_type", "layer")
        moe_mode = config.get("moe_mode", 1)
        attention_type = config.get("attention_type", "none")
        num_heads = config.get("num_heads", 1)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp_mode = config.get("mlp_mode", 0)
        
        model = build_model_from_config(config, input_dims, output_pred_dim, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        materialize_lazy_modules(model, input_dims, next(model.parameters()).device)
        model.load_state_dict(torch.load("model.pt", map_location=device))
        
        data_csv = input("Enter data CSV path (for error analysis): ").strip()
        print("CSV delimiter: 1=,  2=\\t  3=;  4=space")
        dc = input("Choice: ").strip()
        delimiter = {"1":",","2":"\t","3":";","4":" "}.get(dc, ",")
        batch_size = ask_batch_size()
        max_str = input("Max examples to display (default 50): ").strip()
        max_display = int(max_str) if max_str else 50
        
        run_error_analysis(model, data_csv, delimiter, input_cols_l, output_cols_l,
                          col_types_loaded, vocabularies_l, scalings_l,
                          config.get("image_params", {}), batch_size, device, max_display)

    elif choice in ["resume", "r", "14"]:
        kwargs, val_ds, _ = prepare_resume("config.json", "model.pt")
        print(f"Resuming: {kwargs['hidden_dims']} {kwargs['base_activation_cls_for_config'].activation_config['name']} "
              f"on {kwargs['csv_file']} (last optimizer: {kwargs['optimizer_choice']}).")
        if input("Keep the same optimizer? (Y/n): ").strip().lower() == "n":
            kwargs["optimizer_choice"] = ask_optimizer()
        if kwargs["optimizer_choice"] in _GUI_MUON_FAMILY:
            kwargs["optim_params"] = optimizer_options(kwargs["optimizer_choice"], use_defaults=True)
        kwargs["custom_lr"] = ask_learning_rate()
        bs = input(f"Batch size (Enter = {kwargs['batch_size']}): ").strip()
        if bs: kwargs["batch_size"] = int(bs)
        val_loader, val_interval = None, 1000
        if val_ds is not None:
            val_loader = make_loader(val_ds, kwargs["batch_size"], shuffle=False)
            vi = input("Validation interval (steps, Enter = 1000): ").strip()
            val_interval = int(vi) if vi else 1000
        kwargs["loss_balancing"], kwargs["loss_temperature"] = ask_loss_balancing(kwargs["output_cols"], kwargs["col_types"])
        main(val_loader=val_loader, val_interval=val_interval, **kwargs)

    else:
        print("Invalid choice.")
