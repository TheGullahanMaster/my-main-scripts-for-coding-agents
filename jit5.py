import os
import sys
import glob
import math
import random
import signal
import copy
import shutil
import json
import hashlib
import fnmatch
import csv
import ctypes
import re
import subprocess
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from tqdm import tqdm
from lamb import *
from jit5_fcdm import FCDMTokenBlock, FCDMUNet, FCDMTimeEmbedding
from jit5_kernels import hyena_conv, hyena_direct_available
from torch.cuda.amp import autocast, GradScaler

# ==========================================
# Hardware Configuration
# ==========================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Auto-tune conv algorithms for fixed input sizes

SAVE_DIR = "JiTDiff_Flow"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# Global State
# ==========================================
interrupted = False
TRAINING_ACTIVE = False
SIGNAL_OWNER_PID = os.getpid()


def signal_handler(sig, frame):
    global interrupted
    # Forked workers can receive a terminal signal before their initializer runs.
    if os.getpid() != SIGNAL_OWNER_PID:
        return
    if TRAINING_ACTIVE:
        if not interrupted:
            print("\n⚠️ CTRL+C detected. Finishing current step, saving, and exiting...")
        interrupted = True
    else:
        print("\n🚫 CTRL+C detected. Exiting immediately.")
        sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def set_seed(seed):
    """Reproducibility helper."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cycle(dl):
    while True:
        for data in dl:
            yield data


def make_divisible(v, divisor=8):
    new_v = int((v + divisor / 2) // divisor * divisor)
    return max(divisor, new_v)


# ==========================================
# 0. Core Building Blocks
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x.float().pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(norm_x + self.eps).to(dtype=x.dtype)
        return self.scale * x_norm


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.w3(F.mish(self.w1(x)) * self.w2(x))


class SwiGLU_v2(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of=256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.mish(self.w1(x)) * self.w2(x))


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device).add_(keep_prob).floor_()
        return x.div(keep_prob) * random_tensor


# ==========================================
# 0.1 Positional Embeddings
# ==========================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, scale=1.0, theta=10000):
        super().__init__()
        self.dim = dim
        # `scale` lifts a continuous flow-time t in [0, 1] into a wider range
        # (e.g. ~[0, 1000]) so the high-frequency sinusoids carry real signal,
        # matching the effective resolution of DiT-style timestep embeddings.
        self.scale = scale
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=x.dtype) * -emb)
        emb = (x * self.scale)[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w):
    grid_h = torch.arange(grid_size_h, dtype=torch.float32)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size_h, grid_size_w)

    assert embed_dim % 2 == 0
    half = embed_dim // 2
    omega = torch.arange(half // 2, dtype=torch.float32)
    omega /= half / 2.
    omega = 1. / 10000 ** omega

    def _1d(pos):
        pos = pos.reshape(-1)
        out = torch.einsum('m,d->md', pos, omega)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    emb_h = _1d(grid[0])
    emb_w = _1d(grid[1])
    return torch.cat([emb_h, emb_w], dim=1)


class RotaryEmbedding2D(nn.Module):
    def __init__(self, dim, h_grid, w_grid, theta=10000.0):
        super().__init__()
        self.dim = dim
        freqs = self._get_freqs(h_grid, w_grid, dim, theta)
        self.register_buffer("freqs", freqs, persistent=False)

    def _get_freqs(self, h, w, dim, theta):
        dim_h = dim // 2
        dim_w = dim - dim_h

        freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2).float() / dim_h))
        freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2).float() / dim_w))

        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        angles_h = torch.einsum('hw, f -> hwf', grid_y.float(), freqs_h).reshape(-1, dim_h // 2)
        angles_w = torch.einsum('hw, f -> hwf', grid_x.float(), freqs_w).reshape(-1, dim_w // 2)
        return torch.cat([angles_h, angles_w], dim=-1)

    def forward(self, q, k):
        angles = self.freqs.to(q.device).unsqueeze(0).unsqueeze(0)
        cos = angles.cos()
        sin = angles.sin()
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)

    def _apply_rotary(self, t, cos, sin):
        d = t.shape[-1]
        t_paired = t.reshape(*t.shape[:-1], d // 2, 2)
        cos_s = cos.squeeze(-1) if cos.dim() > sin.dim() else cos
        sin_s = sin.squeeze(-1) if sin.dim() > cos.dim() else sin
        # Ensure shapes align
        if cos_s.shape[-1] != t_paired.shape[-2]:
            cos_s = cos.unsqueeze(-1)
            sin_s = sin.unsqueeze(-1)
        x, y = t_paired[..., 0], t_paired[..., 1]
        x_out = x * cos_s.squeeze(-1) - y * sin_s.squeeze(-1)
        y_out = x * sin_s.squeeze(-1) + y * cos_s.squeeze(-1)
        return torch.stack([x_out, y_out], dim=-1).flatten(-2)


# ==========================================
# 0.2 Modulation & EMA
# ==========================================

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


EMA_WARMUP_MODES = ('legacy', 'smooth', 'none')


class EMA(nn.Module):
    """Shadow weights with a selectable decay warmup.

    legacy: min(decay, (1+n)/(10+n)) forever (reaches 0.9999 near 90k steps).
    smooth: 1-decay is interpolated geometrically from 1 (copy) to 1-decay
            along a smoothstep, landing on `decay` at `warmup_steps` with zero
            slope, so there is no snap at the end of warmup.
    none:   constant decay from the first update.
    """

    def __init__(self, model, decay=0.9999, warmup_steps=2000, warmup='legacy'):
        super().__init__()
        if warmup not in EMA_WARMUP_MODES:
            raise ValueError(f"bad EMA warmup {warmup!r}")
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.warmup = warmup
        self.step_count = 0
        self.shadow = copy.deepcopy(model).float()
        self.shadow.requires_grad_(False)
        self.shadow.eval()
        self._pairs = None

    def get_decay(self):
        """Ramp up EMA decay during warmup to avoid copying random init."""
        if self.warmup == 'legacy':
            return min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        if self.warmup == 'none' or self.step_count >= self.warmup_steps:
            return self.decay
        s = self.step_count / max(1, self.warmup_steps)
        s = s * s * (3.0 - 2.0 * s)
        return 1.0 - (1.0 - self.decay) ** s

    def _build_pairs(self, model):
        """Cache matched tensor lists once so updates are single foreach calls."""
        s_param = dict(self.shadow.named_parameters())
        pairs = [(s_param[k], p) for k, p in model.named_parameters() if k in s_param]
        shadow_buffers = dict(self.shadow.named_buffers())
        buffers = [(shadow_buffers[k], b) for k, b in model.named_buffers()]
        self._pairs = ([s for s, _ in pairs], [p for _, p in pairs],
                       [s for s, _ in buffers], [b for _, b in buffers])

    def update(self, model):
        self.step_count += 1
        decay = self.get_decay()
        if self._pairs is None:
            self._build_pairs(model)
        shadow, params, shadow_buffers, buffers = self._pairs
        with torch.no_grad():
            if shadow:
                sources = [p if p.dtype == s.dtype else p.to(s.dtype) for s, p in zip(shadow, params)]
                torch._foreach_lerp_(shadow, sources, 1 - decay)
            for s, b in zip(shadow_buffers, buffers):
                s.copy_(b)

    def forward(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)


# ==========================================
# 0.3 Swin / Window Helpers
# ==========================================

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    # Integer division: exact for window-multiple maps, and float math on sizes
    # breaks torch.compile guards (PyTorch 2.4 'ToFloat' recompile loop).
    B = windows.shape[0] // (H * W // (window_size * window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class SwinWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., version='v1'):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.version = version

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        if version == 'v2':
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
            self.qkv = nn.Linear(dim, dim * 3, bias=False)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.version == 'v2':
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
            attn = (q @ k.transpose(-2, -1))
            attn = attn * torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size ** 2, self.window_size ** 2, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ==========================================
# 0.4 Channel Attention Helpers
# ==========================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.Mish(),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_feat, num_feat // squeeze_factor, bias=False),
            nn.Mish(),
            nn.Linear(num_feat // squeeze_factor, num_feat, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ==========================================
# 1. Attention Blocks
# ==========================================

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, h_patches=None, w_patches=None, qk_norm=False,
                 prenorm=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        # Callers that already normalize (and modulate) their input pass
        # prenorm=False, so adaLN shift/scale reach the projections intact.
        self.norm = RMSNorm(dim) if prenorm else nn.Identity()
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # qk-norm (one of the "Just Advanced" Transformer ingredients in JiT):
        # RMS-normalize the per-head query/key vectors before attention.
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(dim_head)
            self.k_norm = RMSNorm(dim_head)

        if h_patches is not None and w_patches is not None:
            self.rope = RotaryEmbedding2D(dim_head, h_patches, w_patches)
        else:
            self.rope = None

        self.use_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x):
        b, n, _ = x.shape
        x_norm = self.norm(x)

        qkv = self.to_qkv(x_norm).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, self.heads, -1).permute(0, 2, 1, 3), qkv)

        if self.qk_norm:
            q = self.q_norm(q).type_as(v)
            k = self.k_norm(k).type_as(v)

        if self.rope is not None:
            q, k = self.rope(q, k)

        if self.use_flash:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = dots.softmax(dim=-1)
            out = torch.matmul(attn, v)

        out = out.permute(0, 2, 1, 3).reshape(b, n, -1)
        return self.to_out(out)


class TinyAttention(nn.Module):
    def __init__(self, dim, d_out=64, output_dim=None):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, d_out * 3, bias=False)
        self.scale = d_out ** -0.5
        final_dim = output_dim if output_dim is not None else dim
        self.to_out = nn.Linear(d_out, final_dim)
        self.use_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x):
        b, n, c = x.shape
        x_norm = self.norm(x)
        qkv = self.to_qkv(x_norm).chunk(3, dim=-1)
        q, k, v = qkv

        if self.use_flash:
            q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = dots.softmax(dim=-1)
            out = torch.matmul(attn, v).unsqueeze(1)

        out = out.squeeze(1)
        return self.to_out(out)


class RelativeAttention(nn.Module):
    def __init__(self, dim, heads=8, h_patches=None, w_patches=None):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.h_patches = h_patches
        self.w_patches = w_patches

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * h_patches - 1) * (2 * w_patches - 1), heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        coords_h = torch.arange(h_patches)
        coords_w = torch.arange(w_patches)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += h_patches - 1
        relative_coords[:, :, 1] += w_patches - 1
        relative_coords[:, :, 0] *= 2 * w_patches - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class XCAttention(nn.Module):
    """Cross-Covariance Attention - O(N*D^2) instead of O(N^2*D)"""
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 4, 1)  # 3, B, H, D, N

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        return self.proj(out)


class LocalPatchInteraction(nn.Module):
    def __init__(self, dim, h_patches, w_patches):
        super().__init__()
        self.h, self.w = h_patches, w_patches
        self.dw_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.Mish(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
        )

    def forward(self, x):
        B, N, C = x.shape
        img = x.transpose(1, 2).view(B, C, self.h, self.w)
        return self.dw_conv(img).flatten(2).transpose(1, 2)


# ==========================================
# 2. MLP / Spatial Mixing Blocks
# ==========================================

class ConvMLP(nn.Module):
    """Large-kernel ConvMLP for spatial mixing."""
    def __init__(self, dim, mlp_dim, h_patches, w_patches):
        super().__init__()
        self.h, self.w = h_patches, w_patches
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.dwconv = nn.Conv2d(mlp_dim, mlp_dim, kernel_size=7, padding=3, groups=mlp_dim)
        self.act = nn.Mish()
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        x = self.fc1(x)
        x = x.transpose(1, 2).view(b, -1, self.h, self.w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        return self.fc2(x)


class SpatialGatingUnit_v2(nn.Module):
    def __init__(self, dim, seq_len, use_tiny_attn=False, input_dim=None):
        super().__init__()
        self.norm = RMSNorm(dim // 2)
        self.proj = nn.Linear(seq_len, seq_len)
        nn.init.constant_(self.proj.bias, 1.0)
        nn.init.constant_(self.proj.weight, 0.0)

        self.use_tiny_attn = use_tiny_attn
        if use_tiny_attn and input_dim is not None:
            self.tiny_attn = TinyAttention(input_dim, output_dim=dim // 2)

    def forward(self, x, gate_res=None):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.proj(v.transpose(1, 2)).transpose(1, 2)
        if self.use_tiny_attn and gate_res is not None:
            v = v + self.tiny_attn(gate_res)
        return u * F.mish(v)


class ConvSpatialGatingUnit(nn.Module):
    """Hybrid Local-Global gating: 7x7 DWConv + GAP."""
    def __init__(self, dim, h_patches, w_patches, use_tiny_attn=False, input_dim=None):
        super().__init__()
        self.h, self.w = h_patches, w_patches
        self.norm = RMSNorm(dim // 2)
        self.dwconv = nn.Conv2d(dim // 2, dim // 2, kernel_size=7, padding=3, groups=dim // 2)
        self.global_proj = nn.Conv2d(dim // 2, dim // 2, 1)

        self.use_tiny_attn = use_tiny_attn
        if use_tiny_attn and input_dim is not None:
            self.tiny_attn = TinyAttention(input_dim, output_dim=dim // 2)

    def forward(self, x, gate_res=None):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)

        B, N, C = v.shape
        v_img = v.transpose(1, 2).view(B, C, self.h, self.w)
        local_feat = self.dwconv(v_img)
        global_feat = self.global_proj(v_img.mean(dim=(2, 3), keepdim=True))
        v = (local_feat + global_feat).flatten(2).transpose(1, 2)

        if self.use_tiny_attn and gate_res is not None:
            v = v + self.tiny_attn(gate_res)
        return u * v


class CycleFC(nn.Module):
    """CycleMLP core: cyclically shift channel groups spatially."""
    def __init__(self, dim, h_patches, w_patches):
        super().__init__()
        self.h, self.w = h_patches, w_patches
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, h, w, c = x.shape
        c_split = c // 4
        x0, x1, x2, x3 = torch.split(x, [c_split, c_split, c_split, c - 3 * c_split], dim=-1)
        x0 = torch.roll(x0, shifts=-1, dims=1)
        x1 = torch.roll(x1, shifts=1, dims=1)
        x2 = torch.roll(x2, shifts=-1, dims=2)
        x3 = torch.roll(x3, shifts=1, dims=2)
        return self.proj(torch.cat([x0, x1, x2, x3], dim=-1))


class LKAModule(nn.Module):
    """Large Kernel Attention from VAN."""
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv1(self.conv_spatial(self.conv0(x)))
        return u * attn


# ==========================================
# 2.1 Axial Mixing Helpers
# ==========================================

class AxialTokenMixer(nn.Module):
    def __init__(self, h_patches, w_patches, dim):
        super().__init__()
        self.h, self.w = h_patches, w_patches
        self.mix_h = nn.Linear(h_patches, h_patches)
        self.mix_w = nn.Linear(w_patches, w_patches)
        self.act = nn.Mish()

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, self.h, self.w, C)
        x = self.act(self.mix_w(x.permute(0, 1, 3, 2)))  # B,H,C,W
        x = self.act(self.mix_h(x.permute(0, 3, 2, 1)))   # B,W,C,H
        return x.permute(0, 3, 1, 2).reshape(B, N, C)


class AxialSpatialGatingUnit(nn.Module):
    def __init__(self, dim, h_patches, w_patches):
        super().__init__()
        self.norm = RMSNorm(dim // 2)
        self.h, self.w = h_patches, w_patches
        self.proj_h = nn.Linear(h_patches, h_patches)
        self.proj_w = nn.Linear(w_patches, w_patches)
        for p in [self.proj_h, self.proj_w]:
            nn.init.constant_(p.bias, 1.0)
            nn.init.constant_(p.weight, 0.0)

    def forward(self, x, gate_res=None):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        B, N, C = v.shape
        v = v.view(B, self.h, self.w, C)
        v = self.proj_w(v.permute(0, 1, 3, 2))      # mix W
        v = self.proj_h(v.permute(0, 3, 2, 1))       # mix H
        v = v.permute(0, 3, 2, 1).reshape(B, N, C)
        return u * F.mish(v)


class Affine(nn.Module):
    """Learned affine transform (ResMLP style)."""
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.alpha + self.beta


class MDAttnTool(nn.Module):
    """Multi-Dimensional parallel axial mixing with gating."""
    def __init__(self, dim, h_patches, w_patches):
        super().__init__()
        self.h, self.w = h_patches, w_patches
        self.norm = RMSNorm(dim)
        self.proj_h = nn.Linear(h_patches, h_patches)
        self.proj_w = nn.Linear(w_patches, w_patches)
        self.fc_gate = nn.Linear(dim, dim)
        for p in [self.proj_h, self.proj_w]:
            nn.init.constant_(p.weight, 0)
            nn.init.constant_(p.bias, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = self.norm(x)
        gate = self.fc_gate(x)
        x = x.view(B, self.h, self.w, C)
        x_h = self.proj_h(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)  # mix H
        x_w = self.proj_w(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # mix W
        return (x_h + x_w).reshape(B, N, C) * F.silu(gate)


class HyperTokenMixer(nn.Module):
    def __init__(self, dim, num_patches, heads=1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.num_patches = num_patches
        assert dim % heads == 0
        self.head_dim = dim // heads
        self.norm = RMSNorm(dim)
        self.mlp_w1 = nn.Sequential(nn.Linear(self.head_dim, self.head_dim // 2), nn.Mish())
        self.mlp_w2 = nn.Linear(self.head_dim // 2, num_patches)

    def forward(self, x):
        h = self.norm(x)
        b, n, c = h.shape
        if self.heads > 1:
            h_in = h.view(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3).reshape(b * self.heads, n, self.head_dim)
        else:
            h_in = h
        w = F.softmax(self.mlp_w2(self.mlp_w1(h_in)), dim=-1)
        out = torch.bmm(w, h_in)
        if self.heads > 1:
            out = out.view(b, self.heads, n, self.head_dim).permute(0, 2, 1, 3).reshape(b, n, c)
        return out


class ConvHyperTokenMixer(nn.Module):
    def __init__(self, dim, num_patches, h_patches, w_patches, heads=1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.h, self.w = h_patches, w_patches
        self.num_patches = num_patches
        assert dim % heads == 0
        self.head_dim = dim // heads

        self.norm = RMSNorm(dim)
        self.local_mix = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.mlp_w1 = nn.Sequential(nn.Linear(self.head_dim, self.head_dim // 2), nn.Mish())
        self.mlp_w2 = nn.Linear(self.head_dim // 2, num_patches)

    def forward(self, x):
        B, N, C = x.shape
        x_img = x.transpose(1, 2).view(B, C, self.h, self.w)
        x = x + self.local_mix(x_img).flatten(2).transpose(1, 2)

        h = self.norm(x)
        if self.heads > 1:
            h_in = h.view(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3).reshape(B * self.heads, N, self.head_dim)
        else:
            h_in = h
        w = F.softmax(self.mlp_w2(self.mlp_w1(h_in)), dim=-1)
        out = torch.bmm(w, h_in)
        if self.heads > 1:
            out = out.view(B, self.heads, N, self.head_dim).permute(0, 2, 1, 3).reshape(B, N, C)
        return out


# ==========================================
# 3. Full Architecture Blocks (Isotropic)
# ==========================================

def _make_adaln(dim, n_params):
    """Helper to create AdaLN-Zero modulation layer."""
    mod = nn.Sequential(nn.Mish(), nn.Linear(dim, n_params * dim, bias=True))
    nn.init.constant_(mod[-1].weight, 0)
    nn.init.constant_(mod[-1].bias, 0)
    return mod


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, h_patches, w_patches,
                 use_adaln=False, use_conv_mlp=False, use_swiglu=True, qk_norm=False,
                 dropout=0.0, attn_double_norm=True):
        super().__init__()
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(dim)
        # attn_double_norm=True keeps the legacy second RMSNorm inside Attention.
        self.attn = Attention(dim, heads=heads, dim_head=64, h_patches=h_patches, w_patches=w_patches,
                              qk_norm=qk_norm, prenorm=attn_double_norm)
        self.norm2 = RMSNorm(dim)

        if use_conv_mlp:
            self.mlp = ConvMLP(dim, mlp_dim, h_patches, w_patches)
        elif use_swiglu:
            self.mlp = SwiGLU_v2(dim, mlp_dim)
        else:
            self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.Mish(), nn.Linear(mlp_dim, dim))

        # Dropout on the attention/MLP residual branches (paper applies this to
        # the middle half of blocks for the largest H/G models). nn.Dropout has
        # no parameters, so enabling it never changes the state_dict layout.
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        if self.use_adaln and t_emb is not None:
            shifts_scales = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = shifts_scales
            x = x + gate_msa.unsqueeze(1) * self.drop(self.attn(modulate(self.norm1(x), shift_msa, scale_msa)))
            x = x + gate_mlp.unsqueeze(1) * self.drop(self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        else:
            x = x + self.drop(self.attn(self.norm1(x)))
            x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class FullAttentionBlock(nn.Module):
    def __init__(self, dim, h, w, heads, use_adaln=False, qk_norm=False, attn_double_norm=True):
        super().__init__()
        self.use_adaln = use_adaln
        # attn_double_norm=True keeps the legacy second RMSNorm inside Attention.
        self.attn = Attention(dim, heads=heads, dim_head=64, h_patches=h, w_patches=w, qk_norm=qk_norm,
                              prenorm=attn_double_norm)
        self.norm = RMSNorm(dim)
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 3)

    def forward(self, x, t_emb=None):
        if self.use_adaln:
            shift, scale, gate = self.adaLN_modulation(t_emb).chunk(3, dim=1)
            x = x + gate.unsqueeze(1) * self.attn(modulate(self.norm(x), shift, scale))
        else:
            x = x + self.attn(self.norm(x))
        return x


class BaseMLPBlock(nn.Module):
    def __init__(self, dim, h_patch, w_patch):
        super().__init__()
        self.net = ConvMLP(dim, dim, h_patch, w_patch)

    def forward(self, x, t_emb=None):
        if t_emb is not None:
            x = x + t_emb.unsqueeze(1)
        return self.net(x)


class TransformerMLPBlock(nn.Module):
    def __init__(self, dim, mlp_dim, h_patches, w_patches, use_conv_mlp=False):
        super().__init__()
        self.norm = RMSNorm(dim)
        if use_conv_mlp:
            self.mlp = ConvMLP(dim, mlp_dim, h_patches, w_patches)
        else:
            self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.Mish(), nn.Linear(mlp_dim, dim))

    def forward(self, x, t_emb=None):
        return x + self.mlp(self.norm(x))


class TokenGRN(nn.Module):
    """FCDM-style global response normalization for B x token x channel features."""
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, channels))

    def forward(self, x):
        response = torch.linalg.vector_norm(x.float(), dim=1, keepdim=True)
        relative = (response / (response.mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
        return x + self.gamma.to(x.dtype) * x * relative + self.beta.to(x.dtype)


class GRNResidualBlock(nn.Module):
    """Apply optional GRN to a block's residual update without changing the block."""
    def __init__(self, block, dim):
        super().__init__()
        self.block = block
        self.grn = TokenGRN(dim)

    def forward(self, x, *args, **kwargs):
        return x + self.grn(self.block(x, *args, **kwargs) - x)


class gMLPBlock_v5(nn.Module):
    def __init__(self, dim, seq_len, expansion_factor=4, use_adaln=False, tiny_attn=False):
        super().__init__()
        inner_dim = make_divisible(int(dim * expansion_factor * (2 / 3)), 16) * 2
        self.use_adaln = use_adaln
        self.norm = RMSNorm(dim)
        self.proj_in = nn.Linear(dim, inner_dim)
        self.sgu = SpatialGatingUnit_v2(inner_dim, seq_len, use_tiny_attn=tiny_attn, input_dim=dim)
        self.proj_out = nn.Linear(inner_dim // 2, dim)
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 3)

    def forward(self, x, t_emb=None):
        if self.use_adaln and t_emb is not None:
            shift, scale, gate = self.adaLN_modulation(t_emb).chunk(3, dim=1)
            x_norm = modulate(self.norm(x), shift, scale)
            res = self.proj_out(self.sgu(self.proj_in(x_norm), gate_res=x_norm))
            return x + gate.unsqueeze(1) * res
        else:
            x_norm = self.norm(x)
            return x + self.proj_out(self.sgu(self.proj_in(x_norm), gate_res=x_norm))


class gMLPBlock_Conv(nn.Module):
    def __init__(self, dim, seq_len, h_patches, w_patches, expansion_factor=4, use_adaln=False, tiny_attn=False):
        super().__init__()
        self.use_adaln = use_adaln
        inner_dim = dim * expansion_factor
        self.norm = RMSNorm(dim)
        self.proj_in = nn.Linear(dim, inner_dim)
        self.sgu = ConvSpatialGatingUnit(inner_dim, h_patches, w_patches, use_tiny_attn=tiny_attn, input_dim=dim)
        self.proj_out = nn.Linear(inner_dim // 2, dim)
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 3)

    def forward(self, x, t_emb=None):
        if self.use_adaln:
            shift, scale, gate = self.adaLN_modulation(t_emb).chunk(3, dim=1)
            x_norm = modulate(self.norm(x), shift, scale)
            res = self.proj_out(self.sgu(self.proj_in(x_norm), gate_res=x_norm))
            return x + gate.unsqueeze(1) * res
        else:
            x_norm = self.norm(x)
            return x + self.proj_out(self.sgu(self.proj_in(x_norm), gate_res=x_norm))


class AxialgMLPBlock(nn.Module):
    def __init__(self, dim, h_patches, w_patches, expansion_factor=4, use_adaln=False, tiny_attn=False):
        super().__init__()
        inner_dim = make_divisible(int(dim * expansion_factor * (2 / 3)), 16) * 2
        self.use_adaln = use_adaln
        self.norm = RMSNorm(dim)
        self.proj_in = nn.Linear(dim, inner_dim)
        self.sgu = AxialSpatialGatingUnit(inner_dim, h_patches, w_patches)
        self.proj_out = nn.Linear(inner_dim // 2, dim)
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 3)

    def forward(self, x, t_emb=None):
        if self.use_adaln and t_emb is not None:
            shift, scale, gate = self.adaLN_modulation(t_emb).chunk(3, dim=1)
            x_norm = modulate(self.norm(x), shift, scale)
            res = self.proj_out(self.sgu(self.proj_in(x_norm)))
            return x + gate.unsqueeze(1) * res
        else:
            x_norm = self.norm(x)
            return x + self.proj_out(self.sgu(self.proj_in(x_norm)))


class AxialMixerBlock(nn.Module):
    def __init__(self, dim, h_patches, w_patches, token_dim, channel_dim, use_adaln=True):
        super().__init__()
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.token_mix = AxialTokenMixer(h_patches, w_patches, dim)
        self.channel_mlp = nn.Sequential(nn.Linear(dim, channel_dim), nn.Mish(), nn.Linear(channel_dim, dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        if self.use_adaln and t_emb is not None:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            x = x + s[2].unsqueeze(1) * self.token_mix(modulate(self.norm1(x), s[0], s[1]))
            x = x + s[5].unsqueeze(1) * self.channel_mlp(modulate(self.norm2(x), s[3], s[4]))
        else:
            x = x + self.token_mix(self.norm1(x))
            x = x + self.channel_mlp(self.norm2(x))
        return x


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, token_dim, channel_dim, use_adaln=True):
        super().__init__()
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.token_mlp = nn.Sequential(nn.Linear(num_patches, token_dim), nn.Mish(), nn.Linear(token_dim, num_patches))
        self.channel_mlp = nn.Sequential(nn.Linear(dim, channel_dim), nn.Mish(), nn.Linear(channel_dim, dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        if self.use_adaln and t_emb is not None:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            y = self.token_mlp(modulate(self.norm1(x), s[0], s[1]).transpose(1, 2)).transpose(1, 2)
            x = x + s[2].unsqueeze(1) * y
            y = self.channel_mlp(modulate(self.norm2(x), s[3], s[4]))
            x = x + s[5].unsqueeze(1) * y
        else:
            x = x + self.token_mlp(self.norm1(x).transpose(1, 2)).transpose(1, 2)
            x = x + self.channel_mlp(self.norm2(x))
        return x


class GatedMixerBlock(nn.Module):
    def __init__(self, dim, num_patches, token_dim, channel_dim, use_adaln=True):
        super().__init__()
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.token_proj_in = nn.Linear(num_patches, token_dim * 2)
        self.token_proj_out = nn.Linear(token_dim, num_patches)
        self.channel_proj_in = nn.Linear(dim, channel_dim * 2)
        self.channel_proj_out = nn.Linear(channel_dim, dim)
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        if self.use_adaln and t_emb is not None:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            # Gated Token
            y = modulate(self.norm1(x), s[0], s[1]).transpose(1, 2)
            y_u, y_v = self.token_proj_in(y).chunk(2, dim=-1)
            y = self.token_proj_out(y_u * F.mish(y_v)).transpose(1, 2)
            x = x + s[2].unsqueeze(1) * y
            # Gated Channel
            y = modulate(self.norm2(x), s[3], s[4])
            y_u, y_v = self.channel_proj_in(y).chunk(2, dim=-1)
            x = x + s[5].unsqueeze(1) * self.channel_proj_out(y_u * F.silu(y_v))
        else:
            y = self.norm1(x).transpose(1, 2)
            y_u, y_v = self.token_proj_in(y).chunk(2, dim=-1)
            x = x + self.token_proj_out(y_u * F.silu(y_v)).transpose(1, 2)
            y = self.norm2(x)
            y_u, y_v = self.channel_proj_in(y).chunk(2, dim=-1)
            x = x + self.channel_proj_out(y_u * F.silu(y_v))
        return x


class HyperMixerBlock(nn.Module):
    def __init__(self, dim, num_patches, heads=1, use_adaln=True):
        super().__init__()
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.token_mix = HyperTokenMixer(dim, num_patches, heads=heads)
        self.channel_mix = nn.Sequential(nn.Linear(dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        if self.use_adaln and t_emb is not None:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            x = x + s[2].unsqueeze(1) * self.token_mix(modulate(self.norm1(x), s[0], s[1]))
            x = x + s[5].unsqueeze(1) * self.channel_mix(modulate(self.norm2(x), s[3], s[4]))
        else:
            x = x + self.token_mix(self.norm1(x))
            x = x + self.channel_mix(self.norm2(x))
        return x


class HybridHyperBlock(nn.Module):
    def __init__(self, dim, num_patches, h_patches, w_patches, heads=1):
        super().__init__()
        self.token_mix = ConvHyperTokenMixer(dim, num_patches, h_patches, w_patches, heads=heads)
        self.norm2 = RMSNorm(dim)
        self.channel_mix = nn.Sequential(nn.Linear(dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim))

    def forward(self, x, t_emb=None):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(self.norm2(x))
        return x


class ConvMixerBlock(nn.Module):
    def __init__(self, dim, h_patches, w_patches, kernel_size=7):
        super().__init__()
        self.h, self.w = h_patches, w_patches
        self.norm1 = RMSNorm(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=dim, padding=kernel_size // 2)
        self.act = nn.Mish()
        self.norm2 = RMSNorm(dim)
        self.channel_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim))

    def forward(self, x, t_emb=None):
        residual = x
        x = self.norm1(x)
        B, N, C = x.shape
        x = self.act(self.dwconv(x.transpose(1, 2).view(B, C, self.h, self.w)))
        x = residual + x.flatten(2).transpose(1, 2)
        return x + self.channel_mlp(self.norm2(x))


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, h_patches, w_patches, drop_path=0., use_adaln=True):
        super().__init__()
        self.h, self.w = h_patches, w_patches
        self.use_adaln = use_adaln
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 3)
        else:
            self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x, t_emb=None):
        residual = x
        b, n, c = x.shape
        x = self.dwconv(x.transpose(1, 2).view(b, c, self.h, self.w)).view(b, c, n).transpose(1, 2)

        if self.use_adaln:
            if t_emb is None:
                raise ValueError('Conditioned ConvNeXt requires a timestep embedding')
            shift, scale, gate = self.adaLN_modulation(t_emb).chunk(3, dim=1)
            x = self.pwconv2(self.act(self.pwconv1(modulate(self.norm(x), shift, scale))))
            return residual + self.drop_path(gate.unsqueeze(1) * x)
        else:
            x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
            return residual + self.drop_path(self.gamma * x)


class ConvFormerBlock(nn.Module):
    def __init__(self, dim, mlp_dim, h_patches, w_patches):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.token_mix = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.Mish(), nn.Linear(mlp_dim, dim))
        self.h, self.w = h_patches, w_patches

    def forward(self, x, t_emb=None):
        residual = x
        x = self.norm1(x)
        b, n, c = x.shape
        x = self.token_mix(x.transpose(1, 2).view(b, c, self.h, self.w))
        x = residual + x.flatten(2).transpose(1, 2)
        return x + self.mlp(self.norm2(x))


class FourierMixerBlock(nn.Module):
    def __init__(self, dim, mlp_dim, h_patches, w_patches):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.Mish(), nn.Linear(mlp_dim, dim))
        self.h, self.w = h_patches, w_patches
        freq_h, freq_w = h_patches, w_patches // 2 + 1
        self.w_real = nn.Parameter(0.02 * torch.randn(freq_h, freq_w, 1))
        self.w_imag = nn.Parameter(0.02 * torch.randn(freq_h, freq_w, 1))

    def forward(self, x, t_emb=None):
        residual = x
        x = self.norm1(x)
        b, n, c = x.shape
        # FP32 FFT: cuFFT rejects BF16 and needs power-of-two sizes in FP16.
        x = x.view(b, self.h, self.w, c).float()
        x_f = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x_f = x_f * torch.complex(self.w_real.float(), self.w_imag.float())
        x = torch.fft.irfft2(x_f, s=(self.h, self.w), dim=(1, 2), norm="ortho")
        x = residual + x.view(b, n, c).to(residual.dtype)
        return x + self.mlp(self.norm2(x))


class RNNBlock(nn.Module):
    def __init__(self, dim, mlp_dim, h_patches, w_patches):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.h, self.w = h_patches, w_patches
        self.rnn_h = nn.GRU(dim, dim // 2, 1, batch_first=True, bidirectional=True)
        self.rnn_v = nn.GRU(dim, dim // 2, 1, batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.Mish(), nn.Linear(mlp_dim, dim))

    def forward(self, x, t_emb=None):
        b, n, c = x.shape
        residual = x
        x = self.norm1(x).view(b, self.h, self.w, c)
        # Horizontal sweep
        self.rnn_h.flatten_parameters()
        x_h, _ = self.rnn_h(x.reshape(b * self.h, self.w, c))
        x_h = x_h.view(b, self.h, self.w, c)
        # Vertical sweep
        x_v = x_h.permute(0, 2, 1, 3).reshape(b * self.w, self.h, c)
        self.rnn_v.flatten_parameters()
        x_v, _ = self.rnn_v(x_v)
        x = x_v.view(b, self.w, self.h, c).permute(0, 2, 1, 3).reshape(b, n, c)
        x = residual + x
        return x + self.mlp(self.norm2(x))


class LKABlock(nn.Module):
    def __init__(self, dim, mlp_dim, h_patches, w_patches, use_adaln=False):
        super().__init__()
        self.use_adaln = use_adaln
        self.h, self.w = h_patches, w_patches
        self.norm1 = RMSNorm(dim)
        self.lka = LKAModule(dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.Mish(), nn.Linear(mlp_dim, dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        b, n, c = x.shape
        if self.use_adaln:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            x_mod = modulate(self.norm1(x), s[0], s[1])
            out = self.lka(x_mod.transpose(1, 2).view(b, c, self.h, self.w)).flatten(2).transpose(1, 2)
            x = x + s[2].unsqueeze(1) * out
            x = x + s[5].unsqueeze(1) * self.mlp(modulate(self.norm2(x), s[3], s[4]))
        else:
            x_res = self.norm1(x)
            out = self.lka(x_res.transpose(1, 2).view(b, c, self.h, self.w)).flatten(2).transpose(1, 2)
            x = x + out
            x = x + self.mlp(self.norm2(x))
        return x


class XCiTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, h_patches, w_patches, use_adaln=False):
        super().__init__()
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(dim)
        self.xca = XCAttention(dim, heads=heads)
        self.lpi = LocalPatchInteraction(dim, h_patches, w_patches)
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.Mish(), nn.Linear(mlp_dim, dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        if self.use_adaln:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            x = x + s[2].unsqueeze(1) * self.xca(modulate(self.norm1(x), s[0], s[1]))
            x = x + self.lpi(self.norm1(x))
            x = x + s[5].unsqueeze(1) * self.mlp(modulate(self.norm2(x), s[3], s[4]))
        else:
            x = x + self.xca(self.norm1(x))
            x = x + self.lpi(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class CycleMLPBlock(nn.Module):
    def __init__(self, dim, mlp_dim, h_patches, w_patches, use_adaln=False):
        super().__init__()
        self.use_adaln = use_adaln
        self.h, self.w = h_patches, w_patches
        self.norm1 = RMSNorm(dim)
        self.cycle_fc = CycleFC(dim, h_patches, w_patches)
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.Mish(), nn.Linear(mlp_dim, dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        b, n, c = x.shape
        if self.use_adaln:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            x_norm = modulate(self.norm1(x), s[0], s[1])
            out = self.cycle_fc(x_norm.view(b, self.h, self.w, c)).view(b, n, c)
            x = x + s[2].unsqueeze(1) * out
            x = x + s[5].unsqueeze(1) * self.mlp(modulate(self.norm2(x), s[3], s[4]))
        else:
            x_norm = self.norm1(x)
            x = x + self.cycle_fc(x_norm.view(b, self.h, self.w, c)).view(b, n, c)
            x = x + self.mlp(self.norm2(x))
        return x


class ResMLPBlock(nn.Module):
    def __init__(self, dim, num_patches, mlp_ratio=4.0, use_adaln=False):
        super().__init__()
        self.use_adaln = use_adaln
        if not use_adaln:
            self.norm1 = Affine(dim)
            self.norm2 = Affine(dim)

        self.linear_tokens = nn.Linear(num_patches, num_patches)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))
        self.gamma1 = nn.Parameter(1e-4 * torch.ones(dim))
        self.gamma2 = nn.Parameter(1e-4 * torch.ones(dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        if self.use_adaln and t_emb is not None:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            res = self.linear_tokens(modulate(x, s[0], s[1]).transpose(1, 2)).transpose(1, 2)
            x = x + s[2].unsqueeze(1) * (self.gamma1 * res)
            res = self.mlp(modulate(x, s[3], s[4]))
            x = x + s[5].unsqueeze(1) * (self.gamma2 * res)
        else:
            y = self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2)
            x = x + self.gamma1 * y
            x = x + self.gamma2 * self.mlp(self.norm2(x))
        return x


class MDMLPBlock(nn.Module):
    def __init__(self, dim, h_patches, w_patches, mlp_ratio=4.0, use_adaln=False):
        super().__init__()
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.md_mix = MDAttnTool(dim, h_patches, w_patches)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Mish(), nn.Linear(hidden_dim, dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        if self.use_adaln and t_emb is not None:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            x = x + s[2].unsqueeze(1) * self.md_mix(modulate(self.norm1(x), s[0], s[1]))
            x = x + s[5].unsqueeze(1) * self.mlp(modulate(self.norm2(x), s[3], s[4]))
        else:
            x = x + self.md_mix(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class MixerAttnBlock(nn.Module):
    def __init__(self, dim, num_patches, token_dim, channel_dim, heads, use_adaln=True, attn_double_norm=True):
        super().__init__()
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.token_mlp = nn.Sequential(nn.Linear(num_patches, token_dim), nn.Mish(), nn.Linear(token_dim, num_patches))
        # attn_double_norm=True keeps the legacy second RMSNorm inside Attention.
        self.channel_attn = Attention(dim, heads=heads, dim_head=64, prenorm=attn_double_norm)
        self.channel_mlp = nn.Sequential(nn.Linear(dim, channel_dim), nn.Mish(), nn.Linear(channel_dim, dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        if self.use_adaln and t_emb is not None:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            y = self.token_mlp(modulate(self.norm1(x), s[0], s[1]).transpose(1, 2)).transpose(1, 2)
            x = x + s[2].unsqueeze(1) * y
            y = self.channel_mlp(self.channel_attn(modulate(self.norm2(x), s[3], s[4])))
            x = x + s[5].unsqueeze(1) * y
        else:
            x = x + self.token_mlp(self.norm1(x).transpose(1, 2)).transpose(1, 2)
            x = x + self.channel_mlp(self.channel_attn(self.norm2(x)))
        return x


class BiGSBlock(nn.Module):
    def __init__(self, dim, mlp_dim, h_patches, w_patches, use_adaln=False):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.use_adaln = use_adaln
        self.proj_in = nn.Linear(dim, 2 * dim)
        self.proj_out = nn.Linear(dim, dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.Mish(), nn.Linear(mlp_dim, dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        if self.use_adaln:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            x_norm = modulate(self.norm1(x), s[0], s[1])
        else:
            x_norm = self.norm1(x)

        x_tok, z = self.proj_in(x_norm).chunk(2, dim=-1)
        global_ctx = x_tok.mean(dim=1, keepdim=True)
        gate = F.silu(z)
        out = self.proj_out(x_tok * gate + global_ctx * (1 - gate))

        if self.use_adaln:
            x = x + s[2].unsqueeze(1) * out
            x = x + s[5].unsqueeze(1) * self.mlp(modulate(self.norm2(x), s[3], s[4]))
        else:
            x = x + out
            x = x + self.mlp(self.norm2(x))
        return x


# ==========================================
# 3.1 Hierarchical / Windowed Blocks
# ==========================================

class MBConvBlock(nn.Module):
    """CoAtNet MBConv block."""
    def __init__(self, dim, h_patches, w_patches, expansion=4, use_adaln=False):
        super().__init__()
        self.h, self.w = h_patches, w_patches
        self.use_adaln = use_adaln
        hidden_dim = int(dim * expansion)
        self.norm1 = RMSNorm(dim)
        self.expand_conv = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.se = SEBlock(hidden_dim, reduction=4)
        self.proj_conv = nn.Conv2d(hidden_dim, dim, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(dim)
        self.act = nn.Mish()
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 3)

    def forward(self, x, t_emb=None):
        residual = x
        b, n, c = x.shape
        if self.use_adaln:
            shift, scale, gate = self.adaLN_modulation(t_emb).chunk(3, dim=1)
            x = modulate(self.norm1(x), shift, scale)
        else:
            x = self.norm1(x)
        x = x.transpose(1, 2).view(b, c, self.h, self.w)
        x = self.act(self.bn1(self.expand_conv(x)))
        x = self.act(self.bn2(self.dw_conv(x)))
        x = self.se(x)
        x = self.bn3(self.proj_conv(x))
        x = x.flatten(2).transpose(1, 2)
        if self.use_adaln:
            return residual + gate.unsqueeze(1) * x
        return residual + x


class CoAtNetTransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, h_patches, w_patches, use_adaln=False):
        super().__init__()
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(dim)
        self.attn = RelativeAttention(dim, heads=heads, h_patches=h_patches, w_patches=w_patches)
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.Mish(), nn.Linear(mlp_dim, dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        if self.use_adaln:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            x = x + s[2].unsqueeze(1) * self.attn(modulate(self.norm1(x), s[0], s[1]))
            x = x + s[5].unsqueeze(1) * self.mlp(modulate(self.norm2(x), s[3], s[4]))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4., version='v1', time_dim=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.version = version
        self.time_proj = nn.Linear(time_dim, dim) if time_dim else None
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SwinWindowAttention(dim, window_size=window_size, num_heads=num_heads, version=version)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.Mish(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x, t_emb=None):
        if self.time_proj is not None and t_emb is not None:
            x = x + self.time_proj(F.silu(t_emb))[:, None, None, :]
        H, W = x.shape[1], x.shape[2]
        shortcut = x
        x_in = x if self.version == 'v2' else self.norm1(x)

        if self.shift_size > 0:
            shifted_x = torch.roll(x_in, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_in

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size ** 2, self.dim)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if self.version == 'v2':
            x = shortcut + self.norm1(x)
            x = x + self.norm2(self.mlp(x))
        else:
            x = shortcut + x
            x = x + self.mlp(self.norm2(x))
        return x


class SwinBlockAdapter(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim, version='v1'):
        super().__init__()
        self.match_dims = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.swin1 = SwinTransformerBlock(out_c, num_heads=4, window_size=8, shift_size=0, version=version, time_dim=time_emb_dim)
        self.swin2 = SwinTransformerBlock(out_c, num_heads=4, window_size=8, shift_size=4, version=version, time_dim=time_emb_dim)

    def forward(self, x, t_emb):
        x = self.match_dims(x)
        h = x.permute(0, 2, 3, 1)
        B, H, W, C = h.shape
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            h = F.pad(h, (0, 0, 0, pad_w, 0, pad_h))
        h = self.swin2(self.swin1(h, t_emb), t_emb)
        if pad_h > 0 or pad_w > 0:
            h = h[:, :H, :W, :]
        return h.permute(0, 3, 1, 2)


class HATBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4., window_size=8, shift_size=0,
                 h_patches=None, w_patches=None, use_adaln=False):
        super().__init__()
        self.dim = dim
        self.h, self.w = h_patches, w_patches
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn = SwinWindowAttention(dim, window_size=window_size, num_heads=heads, version='v1')
        self.cab = ChannelAttention(dim)
        self.mlp = ConvMLP(dim, int(dim * mlp_ratio), h_patches, w_patches)
        self.gamma1 = nn.Parameter(1e-6 * torch.ones(dim))
        self.gamma2 = nn.Parameter(1e-6 * torch.ones(dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        b, n, c = x.shape
        H, W = self.h, self.w

        if self.use_adaln:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            x_norm = modulate(self.norm1(x), s[0], s[1])
        else:
            x_norm = self.norm1(x)

        # Window Attention with optional shift
        x_img = x_norm.view(b, H, W, c)
        if self.shift_size > 0:
            x_img = torch.roll(x_img, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x_img = F.pad(x_img, (0, 0, 0, pad_w, 0, pad_h))
        x_windows = window_partition(x_img, self.window_size).view(-1, self.window_size ** 2, c)
        attn_windows = self.attn(x_windows).view(-1, self.window_size, self.window_size, c)
        x_img = window_reverse(attn_windows, self.window_size, H + pad_h, W + pad_w)
        if pad_h > 0 or pad_w > 0:
            x_img = x_img[:, :H, :W, :]
        if self.shift_size > 0:
            x_img = torch.roll(x_img, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x_attn = x_img.view(b, n, c)

        # Channel Attention
        x_cab = self.cab(x_norm.transpose(1, 2).view(b, c, H, W)).flatten(2).transpose(1, 2)
        combined = x_attn + x_cab

        if self.use_adaln:
            x = x + s[2].unsqueeze(1) * (self.gamma1 * combined)
            x = x + s[5].unsqueeze(1) * (self.gamma2 * self.mlp(modulate(self.norm2(x), s[3], s[4])))
        else:
            x = x + self.gamma1 * combined
            x = x + self.gamma2 * self.mlp(self.norm2(x))
        return x
class FocalModulation(nn.Module):
    def __init__(self, dim, focal_level=2, focal_window=7):
        super().__init__()
        self.dim = dim
        self.focal_level = focal_level
        self.focal_window = focal_window
        
        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1))
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True)
        self.act = nn.GELU()
        
        self.focal_layers = nn.ModuleList()
        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = 2 * k * self.focal_window + self.focal_window # Growing kernel
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, stride=1, 
                              groups=dim, padding=kernel_size//2, bias=False),
                    nn.GELU()
                )
            )
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x_img = x.transpose(1, 2).view(B, C, H, W)
        
        # 1. Projection
        x_proj = self.f(x).permute(0, 2, 1).view(B, -1, H, W)
        q, ctx, gates = torch.split(x_proj, [C, C, self.focal_level + 1], 1)
        
        # 2. Context Aggregation
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l:l+1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]
        
        # 3. Modulation
        x_out = q * self.h(ctx_all)
        x_out = x_out.flatten(2).transpose(1, 2)
        return self.proj(x_out)

class FocalNetBlock(nn.Module):
    def __init__(self, dim, mlp_dim, h_patches, w_patches, use_adaln=False):
        super().__init__()
        self.h, self.w = h_patches, w_patches
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(dim)
        self.modulation = FocalModulation(dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.Mish(), nn.Linear(mlp_dim, dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)
            
    def forward(self, x, t_emb=None):
        if self.use_adaln:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            x = x + s[2].unsqueeze(1) * self.modulation(modulate(self.norm1(x), s[0], s[1]), self.h, self.w)
            x = x + s[5].unsqueeze(1) * self.mlp(modulate(self.norm2(x), s[3], s[4]))
        else:
            x = x + self.modulation(self.norm1(x), self.h, self.w)
            x = x + self.mlp(self.norm2(x))
        return x

class GridAttention(nn.Module):
    """
    Grid Attention from MaxViT.
    Divides image into a grid and attends to pixels in the same relative position across windows.
    """
    def __init__(self, dim, heads, grid_size=(8, 8)):
        super().__init__()
        self.attn = Attention(dim, heads=heads)
        self.grid_h, self.grid_w = grid_size

    def forward(self, x, H, W):
        # x: B, N, C
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        
        # Partition into grid
        # 1. Reshape to (B, gh, g_sz_h, gw, g_sz_w, C)
        gh, gw = H // self.grid_h, W // self.grid_w
        x = x.view(B, gh, self.grid_h, gw, self.grid_w, C)
        
        # 2. Permute to (B, gh, gw, g_sz_h, g_sz_w, C) -> Grid becomes "Batch" dimension
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, gh * gw, C)
        
        # 3. Apply Attention
        x = self.attn(x)
        
        # 4. Reverse Partition
        x = x.view(B, self.grid_h, self.grid_w, gh, gw, C)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, N, C)
        return x

class MaxViTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, h_patches, w_patches, use_adaln=False):
        super().__init__()
        self.h, self.w = h_patches, w_patches
        self.use_adaln = use_adaln
        
        # Block Attention (Local) - We reuse your existing Attention
        # Note: In real MaxViT this is windowed, here we approximate with standard attn if small enough
        # or use your SwinWindowAttention. Let's use standard for simplicity of integration here, 
        # effectively making it a Dual-Axis block.
        self.block_norm = RMSNorm(dim)
        self.block_attn = Attention(dim, heads=heads)
        
        # Grid Attention (Global)
        self.grid_norm = RMSNorm(dim)
        self.grid_attn = GridAttention(dim, heads, grid_size=(8, 8)) # Fixed grid size 8x8
        
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.Mish(), nn.Linear(mlp_dim, dim))
        self.mlp_norm = RMSNorm(dim)
        
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 9) # Needs more gates for 3 sub-blocks

    def forward(self, x, t_emb=None):
        if self.use_adaln:
            s = self.adaLN_modulation(t_emb).chunk(9, dim=1)
            # Local
            x = x + s[2].unsqueeze(1) * self.block_attn(modulate(self.block_norm(x), s[0], s[1]))
            # Grid
            x = x + s[5].unsqueeze(1) * self.grid_attn(modulate(self.grid_norm(x), s[3], s[4]), self.h, self.w)
            # MLP
            x = x + s[8].unsqueeze(1) * self.mlp(modulate(self.mlp_norm(x), s[6], s[7]))
        else:
            x = x + self.block_attn(self.block_norm(x))
            x = x + self.grid_attn(self.grid_norm(x), self.h, self.w)
            x = x + self.mlp(self.mlp_norm(x))
        return x

try:  # linegen's fused Triton Mamba scan (same recurrence, hand-written backward)
    from linegen_kernels import selective_scan as _triton_selective_scan, kernels_available as _triton_kernels_ok
except Exception:  # pragma: no cover - missing Triton or module
    _triton_selective_scan = None


def fused_scan_available(tensor):
    return _triton_selective_scan is not None and _triton_kernels_ok(tensor)


@torch.compiler.disable
def fused_selective_scan(u, delta, A, Bm, Cm, D):
    """y_t = C_t . h_t + D u_t with h_t = exp(delta_t A) h_{t-1} + delta_t u_t B_t, h_0 = 0.

    u, delta: (B, L, D); A: (D, N); Bm, Cm: (B, L, N); D: (D,). Returns FP32
    (B, L, D). One Triton launch per direction instead of L Python steps;
    torch.compile treats it as an opaque call rather than unrolling L steps.
    """
    x0 = torch.zeros(u.shape[0], u.shape[2], A.shape[1], device=u.device, dtype=torch.float32)
    return _triton_selective_scan(u, delta, A, Bm, Cm, D, x0)[0]


class BiSSM(nn.Module):
    """
    Bidirectional State Space Model (Simplified Mamba-like for Vision).
    Pure PyTorch implementation of a Gated SSM.
    """
    def __init__(self, dim, d_state=16, expand=2, dt_rank="auto"):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        inner_dim = int(dim * expand)
        
        if dt_rank == "auto":
            dt_rank = math.ceil(dim / 16)
            
        self.in_proj = nn.Linear(dim, inner_dim * 2)
        
        # Discretization parameters
        self.x_proj = nn.Linear(inner_dim, dt_rank + d_state * 2)
        self.dt_proj = nn.Linear(dt_rank, inner_dim)

        # A and D parameters (structured state space)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(inner_dim, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(inner_dim))
        
        self.out_proj = nn.Linear(inner_dim, dim)
        self.act = nn.SiLU()

    def ssm_step(self, x):
        """Runs the SSM scan mechanism."""
        B, L, D = x.shape
        
        # Project x to parameters
        x_dbl = self.x_proj(x) # (B, L, dt_rank + 2*d_state)
        dt_rank = self.dt_proj.in_features
        d_state = self.d_state
        
        dt, B_param, C_param = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)) # (B, L, D)
        
        # Discretize A
        A = -torch.exp(self.A_log.float()) # (D, N)
        if fused_scan_available(x):
            return fused_selective_scan(x, dt, A, B_param, C_param, self.D.float())
        dA = torch.exp(torch.einsum("bld,dn->bldn", dt, A))
        dB = torch.einsum("bld,bln->bldn", dt, B_param)
        
        # Scan (Cumulative sum approximation for pure pytorch speed)
        # In real Mamba, this is a parallel associative scan. 
        # Here we use a simplified recurrence loop for compatibility.
        h = torch.zeros(B, D, d_state, device=x.device)
        y = []
        for t in range(L):
            h = h * dA[:, t] + x[:, t].unsqueeze(-1) * dB[:, t]
            y.append(torch.einsum("bdn,bln->bd", h, C_param[:, t].unsqueeze(1)).squeeze(1))
            
        y = torch.stack(y, dim=1) # (B, L, D)
        return y + x * self.D

    def forward(self, x):
        # x: B, N, C
        u, z = self.in_proj(x).chunk(2, dim=-1)
        
        # Bidirectional processing
        x_fwd = self.ssm_step(self.act(u))
        x_bwd = self.ssm_step(self.act(u).flip([1])).flip([1])
        
        out = x_fwd * F.silu(z) + x_bwd * F.silu(z)
        return self.out_proj(out)

class VisionMambaBlock(nn.Module):
    def __init__(self, dim, mlp_dim, h_patches, w_patches, use_adaln=False):
        super().__init__()
        self.use_adaln = use_adaln
        self.norm = RMSNorm(dim)
        self.ssm = BiSSM(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.Mish(), nn.Linear(mlp_dim, dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        if self.use_adaln:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            x = x + s[2].unsqueeze(1) * self.ssm(modulate(self.norm(x), s[0], s[1]))
            x = x + s[5].unsqueeze(1) * self.mlp(modulate(self.norm(x), s[3], s[4]))
        else:
            x = x + self.ssm(self.norm(x))
            x = x + self.mlp(self.norm(x))
        return x

# ==========================================
# 3.2 Reference-faithful HAT / MaxViT / Swin / Vim blocks (arch_version=2)
# ==========================================
# arch_version=1 keeps the legacy classes above so older checkpoints load and
# behave unchanged. These follow the official implementations:
#   HAT     https://github.com/XPixelGroup/HAT (hat/archs/hat_arch.py)
#   MaxViT  timm maxxvit.py (MbConvBlock, PartitionAttentionCl, MaxxVitBlock)
#   Swin    microsoft/Swin-Transformer (swin_transformer.py, swin_transformer_v2.py)
#   Vim     hustvl/Vim (mamba_simple.py bimamba "v2", models_mamba.py)
# adaLN-Zero, when enabled, modulates each pre-norm and gates each residual
# branch, as the other jit5 blocks do; without it the blocks match the originals.

def _block_norm(dim, use_adaln, eps=1e-6):
    """Reference LayerNorm; adaLN supplies shift/scale, so its norm has no affine.

    eps follows each reference: timm MaxViT 1e-6, HAT's nn.LayerNorm default 1e-5.
    """
    return nn.LayerNorm(dim, eps=eps, elementwise_affine=not use_adaln)


def shifted_window_mask(height, width, window, shift):
    """Swin/HAT SW-MSA mask: -100 between tokens from different pre-shift regions."""
    image = torch.zeros(1, height, width, 1)
    count = 0
    for hs in (slice(0, -window), slice(-window, -shift), slice(-shift, None)):
        for ws in (slice(0, -window), slice(-window, -shift), slice(-shift, None)):
            image[:, hs, ws, :] = count
            count += 1
    windows = window_partition(image, window).view(-1, window * window)
    mask = windows.unsqueeze(1) - windows.unsqueeze(2)
    return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)


def grid_partition(x, grid):
    """MaxViT grid partition: each window gathers grid x grid tokens strided across the map."""
    b, h, w, c = x.shape
    x = x.view(b, grid, h // grid, grid, w // grid, c)
    return x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, grid, grid, c)


def grid_reverse(windows, grid, h, w):
    c = windows.shape[-1]
    x = windows.view(-1, h // grid, w // grid, grid, grid, c)
    return x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, h, w, c)


def hat_window_size(h, w):
    """Largest HAT window (8 or 4) dividing both token-grid sides; OCAB needs it even."""
    for window in (8, 4):
        if h % window == 0 and w % window == 0:
            return window
    raise ValueError(f'HAT needs token-grid sides divisible by 4, got {h}x{w}')


class HATChannelAttentionBlock(nn.Module):
    """HAT CAB: 3x3 conv (C -> C/3), GELU, 3x3 conv, RCAN channel attention (squeeze 30)."""
    def __init__(self, dim, compress_ratio=3, squeeze_factor=30):
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(dim, max(1, dim // compress_ratio), 3, 1, 1), nn.GELU(),
            nn.Conv2d(max(1, dim // compress_ratio), dim, 3, 1, 1))
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, max(1, dim // squeeze_factor), 1), nn.ReLU(inplace=True),
            nn.Conv2d(max(1, dim // squeeze_factor), dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.cab(x)
        return x * self.attention(x)


class HABBlock(nn.Module):
    """HAT hybrid attention block: x + (S)W-MSA + 0.01 * CAB, then x + MLP."""
    def __init__(self, dim, heads, h, w, window, shift, mlp_ratio=2., conv_scale=0.01, use_adaln=False):
        super().__init__()
        self.h, self.w, self.window, self.shift = h, w, window, shift
        self.use_adaln = use_adaln
        self.norm1 = _block_norm(dim, use_adaln, eps=1e-5)
        self.attn = SwinWindowAttention(dim, window_size=window, num_heads=heads, version='v1')
        self.conv_scale = conv_scale
        self.conv_block = HATChannelAttentionBlock(dim)
        self.norm2 = _block_norm(dim, use_adaln, eps=1e-5)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.register_buffer('attn_mask', shifted_window_mask(h, w, window, shift) if shift else None,
                             persistent=False)
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        b, n, c = x.shape
        if self.use_adaln:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            y = modulate(self.norm1(x), s[0], s[1])
        else:
            y = self.norm1(x)
        image = y.view(b, self.h, self.w, c)
        conv_x = self.conv_block(image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(b, n, c)
        if self.shift:
            image = torch.roll(image, shifts=(-self.shift, -self.shift), dims=(1, 2))
        windows = window_partition(image, self.window).view(-1, self.window * self.window, c)
        windows = self.attn(windows, mask=self.attn_mask).view(-1, self.window, self.window, c)
        image = window_reverse(windows, self.window, self.h, self.w)
        if self.shift:
            image = torch.roll(image, shifts=(self.shift, self.shift), dims=(1, 2))
        update = image.reshape(b, n, c) + conv_x * self.conv_scale
        if self.use_adaln:
            x = x + s[2].unsqueeze(1) * update
            return x + s[5].unsqueeze(1) * self.mlp(modulate(self.norm2(x), s[3], s[4]))
        x = x + update
        return x + self.mlp(self.norm2(x))


class OCABBlock(nn.Module):
    """HAT overlapping cross-attention: window queries, (1 + overlap)x larger key windows."""
    def __init__(self, dim, heads, h, w, window, overlap_ratio=0.5, mlp_ratio=2., use_adaln=False):
        super().__init__()
        self.h, self.w, self.window, self.heads = h, w, window, heads
        self.ext = int(window * overlap_ratio) + window
        self.scale = (dim // heads) ** -0.5
        self.use_adaln = use_adaln
        self.norm1 = _block_norm(dim, use_adaln, eps=1e-5)
        self.qkv = nn.Linear(dim, dim * 3)
        self.unfold = nn.Unfold(kernel_size=(self.ext, self.ext), stride=window, padding=(self.ext - window) // 2)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((window + self.ext - 1) ** 2, heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        coords = lambda size: torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')).flatten(1)
        relative = (coords(self.ext)[:, None, :] - coords(window)[:, :, None]).permute(1, 2, 0).contiguous()
        relative += window - self.ext + 1
        relative[:, :, 0] *= window + self.ext - 1
        self.register_buffer('relative_position_index', relative.sum(-1), persistent=False)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = _block_norm(dim, use_adaln, eps=1e-5)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        b, n, c = x.shape
        if self.use_adaln:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            y = modulate(self.norm1(x), s[0], s[1])
        else:
            y = self.norm1(x)
        qkv = self.qkv(y).reshape(b, self.h, self.w, 3, c).permute(3, 0, 4, 1, 2)
        q = window_partition(qkv[0].permute(0, 2, 3, 1), self.window).view(-1, self.window * self.window, c)
        kv = self.unfold(torch.cat((qkv[1], qkv[2]), dim=1))
        kv = kv.view(b, 2, c, self.ext * self.ext, -1).permute(1, 0, 4, 3, 2).reshape(2, -1, self.ext * self.ext, c)
        d = c // self.heads
        q = q.reshape(q.shape[0], -1, self.heads, d).permute(0, 2, 1, 3)
        k, v = (t.reshape(t.shape[0], -1, self.heads, d).permute(0, 2, 1, 3) for t in kv)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window * self.window, self.ext * self.ext, -1).permute(2, 0, 1)
        attn = ((q * self.scale) @ k.transpose(-2, -1) + bias.unsqueeze(0)).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(-1, self.window, self.window, c)
        update = self.proj(window_reverse(out, self.window, self.h, self.w).reshape(b, n, c))
        if self.use_adaln:
            x = x + s[2].unsqueeze(1) * update
            return x + s[5].unsqueeze(1) * self.mlp(modulate(self.norm2(x), s[3], s[4]))
        x = x + update
        return x + self.mlp(self.norm2(x))


class HATGroupBlock(nn.Module):
    """One residual hybrid attention group (RHAG): HABs, OCAB, 3x3 conv, group residual.

    HABs alternate plain and shifted windows (shift = window // 2); like the
    reference, a grid no larger than the window uses one unshifted window.
    """
    def __init__(self, dim, heads, h, w, depth=2, use_adaln=False):
        super().__init__()
        self.h, self.w = h, w
        window = hat_window_size(h, w)
        shift = 0 if min(h, w) <= window else window // 2
        self.blocks = nn.ModuleList([
            HABBlock(dim, heads, h, w, window, shift if i % 2 else 0, use_adaln=use_adaln) for i in range(depth)
        ])
        self.overlap_attn = OCABBlock(dim, heads, h, w, window, use_adaln=use_adaln)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, t_emb=None):
        residual = x
        for block in self.blocks:
            x = block(x, t_emb)
        x = self.overlap_attn(x, t_emb)
        b, n, c = x.shape
        x = self.conv(x.transpose(1, 2).reshape(b, c, self.h, self.w)).flatten(2).transpose(1, 2)
        return x + residual


class MaxViTMBConv(nn.Module):
    """MaxViT MBConv: BN, 1x1 expand, BN+GELU, depthwise 3x3, BN+GELU, SE (0.25, SiLU), 1x1.

    BatchNorm follows the paper/timm; note it pools statistics across the
    batch's mixed noise levels.
    """
    def __init__(self, dim, h, w, expand_ratio=4., use_adaln=False):
        super().__init__()
        self.h, self.w, self.use_adaln = h, w, use_adaln
        mid = make_divisible(dim * expand_ratio)
        reduced = int(dim * 0.25)
        self.pre_norm = nn.BatchNorm2d(dim)
        self.conv1_1x1 = nn.Conv2d(dim, mid, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(mid)
        self.conv2_kxk = nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False)
        self.norm2 = nn.BatchNorm2d(mid)
        self.se_fc1 = nn.Conv2d(mid, reduced, 1)
        self.se_fc2 = nn.Conv2d(reduced, mid, 1)
        self.conv3_1x1 = nn.Conv2d(mid, dim, 1)
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 3)

    def forward(self, x, t_emb=None):
        b, n, c = x.shape
        image = x.transpose(1, 2).reshape(b, c, self.h, self.w)
        y = self.pre_norm(image)
        if self.use_adaln:
            shift, scale, gate = self.adaLN_modulation(t_emb).chunk(3, dim=1)
            y = y * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        y = F.gelu(self.norm1(self.conv1_1x1(y)))
        y = F.gelu(self.norm2(self.conv2_kxk(y)))
        y = y * torch.sigmoid(self.se_fc2(F.silu(self.se_fc1(y.mean((2, 3), keepdim=True)))))
        y = self.conv3_1x1(y).flatten(2).transpose(1, 2)
        return x + (gate.unsqueeze(1) * y if self.use_adaln else y)


class MaxViTPartitionAttention(nn.Module):
    """MaxViT block (local window) or grid (dilated global) attention + FFN, relative bias."""
    def __init__(self, dim, heads, h, w, partition, size=8, mlp_ratio=4., use_adaln=False):
        super().__init__()
        if partition not in ('block', 'grid'):
            raise ValueError(f'Unknown MaxViT partition {partition!r}')
        self.h, self.w, self.partition, self.size = h, w, partition, size
        self.use_adaln = use_adaln
        self.norm1 = _block_norm(dim, use_adaln)
        self.attn = SwinWindowAttention(dim, window_size=size, num_heads=heads, version='v1')
        self.norm2 = _block_norm(dim, use_adaln)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def _attend(self, y):
        b, n, c = y.shape
        image = y.view(b, self.h, self.w, c)
        split, merge = ((window_partition, window_reverse) if self.partition == 'block'
                        else (grid_partition, grid_reverse))
        windows = self.attn(split(image, self.size).view(-1, self.size * self.size, c))
        return merge(windows.view(-1, self.size, self.size, c), self.size, self.h, self.w).reshape(b, n, c)

    def forward(self, x, t_emb=None):
        if self.use_adaln:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            x = x + s[2].unsqueeze(1) * self._attend(modulate(self.norm1(x), s[0], s[1]))
            return x + s[5].unsqueeze(1) * self.mlp(modulate(self.norm2(x), s[3], s[4]))
        x = x + self._attend(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class MaxViTReferenceBlock(nn.Module):
    """MaxViT block: MBConv, block attention + FFN, grid attention + FFN (8x8 partitions)."""
    def __init__(self, dim, heads, h, w, use_adaln=False):
        super().__init__()
        self.conv = MaxViTMBConv(dim, h, w, use_adaln=use_adaln)
        self.attn_block = MaxViTPartitionAttention(dim, heads, h, w, 'block', use_adaln=use_adaln)
        self.attn_grid = MaxViTPartitionAttention(dim, heads, h, w, 'grid', use_adaln=use_adaln)

    def forward(self, x, t_emb=None):
        return self.attn_grid(self.attn_block(self.conv(x, t_emb), t_emb), t_emb)


class SwinV2WindowAttention(nn.Module):
    """Swin v2 window attention: scaled cosine attention and log-spaced continuous position bias."""
    def __init__(self, dim, window, heads):
        super().__init__()
        self.window, self.heads = window, heads
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((heads, 1, 1))))
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512), nn.ReLU(inplace=True), nn.Linear(512, heads, bias=False))
        offsets = torch.arange(-(window - 1), window, dtype=torch.float32)
        table = torch.stack(torch.meshgrid(offsets, offsets, indexing='ij')).permute(1, 2, 0).unsqueeze(0)
        table = table / max(window - 1, 1) * 8
        table = torch.sign(table) * torch.log2(table.abs() + 1.0) / math.log2(8)
        self.register_buffer('relative_coords_table', table, persistent=False)
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing='ij')).flatten(1)
        relative = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous() + (window - 1)
        relative[:, :, 0] *= 2 * window - 1
        self.register_buffer('relative_position_index', relative.sum(-1), persistent=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, n, c = x.shape
        bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, bias.to(x.dtype)).reshape(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        attn = attn * torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        table = self.cpb_mlp(self.relative_coords_table).view(-1, self.heads)
        position = table[self.relative_position_index.view(-1)].view(n, n, -1).permute(2, 0, 1)
        attn = attn + 16 * torch.sigmoid(position).unsqueeze(0)
        if mask is not None:
            windows = mask.shape[0]
            attn = attn.view(b // windows, windows, self.heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.heads, n, n)
        return self.proj((attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, n, c))


class SwinReferenceBlock(nn.Module):
    """Swin v1 (pre-norm) or v2 (res-post-norm) block with the reference shifted-window mask.

    Operates on B, H, W, C. As in the reference, a map no larger than the window
    uses one unshifted window; other sizes pad to a window multiple (mmdet style),
    with the mask built over the padded map.
    """
    def __init__(self, dim, heads, resolution, shift, version='v1', time_dim=None, window=8, mlp_ratio=4.):
        super().__init__()
        h, w = resolution
        self.version = version
        self.window = min(window, h, w)
        self.shift = 0 if min(h, w) <= window else shift
        self.pad = ((-h) % self.window, (-w) % self.window)
        self.time_proj = nn.Linear(time_dim, dim) if time_dim else None
        self.norm1 = nn.LayerNorm(dim)
        self.attn = (SwinWindowAttention(dim, window_size=self.window, num_heads=heads, version='v1')
                     if version == 'v1' else SwinV2WindowAttention(dim, self.window, heads))
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        mask = (shifted_window_mask(h + self.pad[0], w + self.pad[1], self.window, self.shift)
                if self.shift else None)
        self.register_buffer('attn_mask', mask, persistent=False)

    def _attend(self, x):
        b, h, w, c = x.shape
        if any(self.pad):
            x = F.pad(x, (0, 0, 0, self.pad[1], 0, self.pad[0]))
        if self.shift:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        hp, wp = x.shape[1:3]
        windows = window_partition(x, self.window).view(-1, self.window * self.window, c)
        windows = self.attn(windows, mask=self.attn_mask).view(-1, self.window, self.window, c)
        x = window_reverse(windows, self.window, hp, wp)
        if self.shift:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        return x[:, :h, :w, :].contiguous()

    def forward(self, x, t_emb=None):
        if self.time_proj is not None and t_emb is not None:
            x = x + self.time_proj(F.silu(t_emb))[:, None, None, :]
        if self.version == 'v2':
            x = x + self.norm1(self._attend(x))
            return x + self.norm2(self.mlp(x))
        x = x + self._attend(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class SwinReferenceAdapter(nn.Module):
    """Two Swin blocks (plain, then shifted) inside a ConvNet level, as NCHW in/out."""
    def __init__(self, in_c, out_c, time_emb_dim, resolution, version='v1', heads=4, window=8):
        super().__init__()
        self.match_dims = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.swin1 = SwinReferenceBlock(out_c, heads, resolution, 0, version, time_emb_dim, window)
        self.swin2 = SwinReferenceBlock(out_c, heads, resolution, window // 2, version, time_emb_dim, window)

    def forward(self, x, t_emb):
        x = self.match_dims(x).permute(0, 2, 3, 1)
        return self.swin2(self.swin1(x, t_emb), t_emb).permute(0, 3, 1, 2)


class BiMambaV2(nn.Module):
    """Vim bidirectional Mamba mixer (bimamba "v2", if_divide_out=True), pure PyTorch scan.

    Separate conv1d / x_proj / dt_proj / A / D per direction; the backward
    direction scans the flipped sequence. Initialisation follows Mamba/Vim:
    S4D-real A, forward dt bias = softplus^-1 of log-uniform [dt_min, dt_max],
    other Linear biases zero, out_proj Kaiming-uniform / sqrt(depth).
    The scan is sequential over tokens (no fused CUDA kernel), so it is slow.
    """
    def __init__(self, dim, depth=1, d_state=16, d_conv=4, expand=2, dt_min=0.001, dt_max=0.1,
                 dt_init_floor=1e-4):
        super().__init__()
        inner = int(expand * dim)
        self.inner, self.d_state = inner, d_state
        self.dt_rank = math.ceil(dim / 16)
        self.in_proj = nn.Linear(dim, inner * 2, bias=False)
        for suffix in ('', '_b'):
            setattr(self, 'conv1d' + suffix, nn.Conv1d(inner, inner, d_conv, groups=inner, padding=d_conv - 1))
            setattr(self, 'x_proj' + suffix, nn.Linear(inner, self.dt_rank + 2 * d_state, bias=False))
            setattr(self, 'dt_proj' + suffix, nn.Linear(self.dt_rank, inner))
            A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(inner, 1)
            setattr(self, 'A_log' if not suffix else 'A_b_log', nn.Parameter(torch.log(A)))
            setattr(self, 'D' + suffix, nn.Parameter(torch.ones(inner)))
        std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(torch.rand(inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            # Vim's _init_weights zeroes every other Linear bias, including dt_proj_b.
            self.dt_proj_b.bias.zero_()
        self.out_proj = nn.Linear(inner, dim, bias=False)
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.out_proj.weight /= math.sqrt(depth)

    def _direction(self, x, z, suffix):
        seq = x.shape[1]
        conv = getattr(self, 'conv1d' + suffix)
        u = F.silu(conv(x.transpose(1, 2))[..., :seq])                              # B, D, L
        dt, B, C = torch.split(getattr(self, 'x_proj' + suffix)(u.transpose(1, 2)),
                               [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_proj = getattr(self, 'dt_proj' + suffix)
        delta = F.softplus((dt @ dt_proj.weight.t()).float() + dt_proj.bias.float())  # B, L, D
        A = -torch.exp((self.A_log if not suffix else self.A_b_log).float())
        return u.float(), delta, A, B.float(), C.float(), getattr(self, 'D' + suffix).float(), z

    @staticmethod
    def _scan(u, delta, A, B, C, D, z):
        """selective_scan_ref: h_t = exp(delta A) h + delta B u; y = C h + D u; y * silu(z)."""
        if fused_scan_available(u):
            y = fused_selective_scan(u.transpose(1, 2), delta, A, B, C, D).transpose(1, 2)
            return y * F.silu(z.float())
        state = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
        ys = []
        for i in range(u.shape[2]):
            d = delta[:, i]                                                         # B, D
            state = torch.exp(d[..., None] * A) * state + (d * u[:, :, i])[..., None] * B[:, i, None, :]
            ys.append((state * C[:, i, None, :]).sum(-1))
        y = torch.stack(ys, dim=-1) + u * D[None, :, None]
        return y * F.silu(z.float())

    def forward(self, x):
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        forward = self._scan(*self._direction(x_in, z.transpose(1, 2), ''))
        backward = self._scan(*self._direction(x_in.flip(1), z.flip(1).transpose(1, 2), '_b'))
        out = (forward + backward.flip(-1)).transpose(1, 2) / 2
        return self.out_proj(out.to(x.dtype))


class VimReferenceBlock(nn.Module):
    """Vim block: x + BiMamba(norm(x)); no MLP. Vim uses RMSNorm (rms_norm=True)."""
    def __init__(self, dim, depth=1, use_adaln=False):
        super().__init__()
        self.use_adaln = use_adaln
        self.norm = RMSNorm(dim, eps=1e-5)
        self.mixer = BiMambaV2(dim, depth=depth)
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 3)

    def forward(self, x, t_emb=None):
        if self.use_adaln:
            shift, scale, gate = self.adaLN_modulation(t_emb).chunk(3, dim=1)
            return x + gate.unsqueeze(1) * self.mixer(modulate(self.norm(x), shift, scale))
        return x + self.mixer(self.norm(x))


class gnConv(nn.Module):
    """Recursive Gated Convolution from HorNet.
    Captures n-th order interactions: y = pwconv_out(p_n) where
    p_{i+1} = pws[i](p_i) * dwconv(q_i), starting from p_0 from a split."""
    def __init__(self, dim, order=5, kernel_size=7):
        super().__init__()
        self.order = order
        # Channel widths: [dim/2^(n-1), ..., dim/2, dim], summing to 2*dim - dims[0]
        self.dims = [dim // (2 ** (order - i - 1)) for i in range(order)]
        assert all(d > 0 for d in self.dims), f"dim={dim} too small for order={order}"

        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)
        sum_q = sum(self.dims)  # = 2*dim - dims[0]
        self.dwconv = nn.Conv2d(sum_q, sum_q, kernel_size,
                                padding=kernel_size // 2, groups=sum_q)
        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.pws = nn.ModuleList([
            nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)
        ])
        self.scale = 1.0 / order  # stabilize deep recursion

    def forward(self, x):
        # x: B, C, H, W
        fused = self.proj_in(x)
        pwa, qs = torch.split(fused, [self.dims[0], sum(self.dims)], dim=1)
        qs = self.dwconv(qs) * self.scale
        q_list = torch.split(qs, self.dims, dim=1)

        x = pwa * q_list[0]
        for i in range(self.order - 1):
            x = self.pws[i](x) * q_list[i + 1]
        return self.proj_out(x)


class HorNetBlock(nn.Module):
    def __init__(self, dim, h_patches, w_patches, order=5, mlp_ratio=4.0,
                 kernel_size=7, use_adaln=False):
        super().__init__()
        self.h, self.w = h_patches, w_patches
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.gnconv = gnConv(dim, order=order, kernel_size=kernel_size)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.Mish(), nn.Linear(hidden, dim))
        self.gamma1 = nn.Parameter(torch.ones(dim))
        self.gamma2 = nn.Parameter(torch.ones(dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        b, n, c = x.shape
        if self.use_adaln and t_emb is not None:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            x_norm = modulate(self.norm1(x), s[0], s[1])
            out = self.gnconv(x_norm.transpose(1, 2).view(b, c, self.h, self.w))
            out = out.flatten(2).transpose(1, 2)
            x = x + s[2].unsqueeze(1) * (self.gamma1 * out)
            x = x + s[5].unsqueeze(1) * (self.gamma2 * self.mlp(modulate(self.norm2(x), s[3], s[4])))
        else:
            out = self.gnconv(self.norm1(x).transpose(1, 2).view(b, c, self.h, self.w))
            x = x + self.gamma1 * out.flatten(2).transpose(1, 2)
            x = x + self.gamma2 * self.mlp(self.norm2(x))
        return x


class AFTSimple(nn.Module):
    """AFT-Simple: w=0. Reduces to Y = sigmoid(Q) ⊙ Σ softmax(K) ⊙ V. O(N) memory."""
    def __init__(self, dim):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        weighted = (k.softmax(dim=1) * v).sum(dim=1, keepdim=True)  # B, 1, C
        return self.proj(torch.sigmoid(q) * weighted)


class AFTFull(nn.Module):
    """AFT-Full: full N×N learned position bias. Memory grows with N²."""
    def __init__(self, dim, num_patches):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.w_bias = nn.Parameter(torch.zeros(num_patches, num_patches))
        nn.init.trunc_normal_(self.w_bias, std=0.02)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        # Numerically stable: subtract per-row max of bias and per-channel max of K
        exp_w = torch.exp(self.w_bias - self.w_bias.amax(dim=-1, keepdim=True))   # N, N
        exp_k = torch.exp(k - k.amax(dim=1, keepdim=True))                         # B, N, C
        num = torch.einsum('ts,bsc->btc', exp_w, exp_k * v)
        den = torch.einsum('ts,bsc->btc', exp_w, exp_k)
        return self.proj(torch.sigmoid(q) * num / (den + 1e-6))


class AFTLocal(nn.Module):
    """AFT-Local: bias only learned for relative positions within a 2D window.
    Outside window the bias is 0 (so K alone determines the contribution)."""
    def __init__(self, dim, h, w, window_size=7):
        super().__init__()
        self.h, self.w = h, w
        self.window_size = window_size
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        ws = window_size
        N = h * w
        positions = torch.stack(torch.meshgrid(
            torch.arange(h), torch.arange(w), indexing='ij'), dim=-1).view(-1, 2)
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)            # N, N, 2
        in_win = (diff[..., 0].abs() <= ws // 2) & (diff[..., 1].abs() <= ws // 2)
        rel_y = diff[..., 0].clamp(-(ws // 2), ws // 2) + ws // 2
        rel_x = diff[..., 1].clamp(-(ws // 2), ws // 2) + ws // 2
        rel_idx = rel_y * ws + rel_x
        self.register_buffer('rel_idx', rel_idx)
        self.register_buffer('in_win', in_win.float())
        self.bias_table = nn.Parameter(torch.zeros(ws * ws))
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        bias = self.bias_table[self.rel_idx] * self.in_win                # N, N
        exp_w = torch.exp(bias - bias.amax(dim=-1, keepdim=True))
        exp_k = torch.exp(k - k.amax(dim=1, keepdim=True))
        num = torch.einsum('ts,bsc->btc', exp_w, exp_k * v)
        den = torch.einsum('ts,bsc->btc', exp_w, exp_k)
        return self.proj(torch.sigmoid(q) * num / (den + 1e-6))


class AFTBlock(nn.Module):
    def __init__(self, dim, num_patches, h, w, mlp_dim, variant='full',
                 window_size=7, use_adaln=False):
        super().__init__()
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        if variant == 'simple':
            self.aft = AFTSimple(dim)
        elif variant == 'local':
            self.aft = AFTLocal(dim, h, w, window_size=window_size)
        else:
            self.aft = AFTFull(dim, num_patches)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.Mish(), nn.Linear(mlp_dim, dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        if self.use_adaln and t_emb is not None:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            x = x + s[2].unsqueeze(1) * self.aft(modulate(self.norm1(x), s[0], s[1]))
            x = x + s[5].unsqueeze(1) * self.mlp(modulate(self.norm2(x), s[3], s[4]))
        else:
            x = x + self.aft(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class HyenaFilter2D(nn.Module):
    """Generates `num_filters` long 2D conv filters from a positional encoding via MLP.

    Filters cover every relative offset (2h-1 x 2w-1), centred on offset zero,
    so each token sees the whole grid in all directions.
    """
    def __init__(self, dim, h, w, filter_dim=64, num_filters=2):
        super().__init__()
        self.dim, self.h, self.w = dim, 2 * h - 1, 2 * w - 1
        self.num_filters = num_filters

        ys = torch.linspace(-1, 1, self.h)
        xs = torch.linspace(-1, 1, self.w)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        n_freqs = max(filter_dim // 8, 4)
        freqs = math.pi * 2 ** torch.linspace(0, 4, n_freqs)  # π to 16π
        pe = torch.cat([
            torch.sin(gy[..., None] * freqs), torch.cos(gy[..., None] * freqs),
            torch.sin(gx[..., None] * freqs), torch.cos(gx[..., None] * freqs),
        ], dim=-1)
        self.register_buffer('pe', pe)

        self.implicit_mlp = nn.Sequential(
            nn.Linear(pe.shape[-1], filter_dim), nn.Mish(),
            nn.Linear(filter_dim, filter_dim), nn.Mish(),
            nn.Linear(filter_dim, num_filters * dim),
        )
        decay = torch.exp(-(gy ** 2 + gx ** 2) * 0.5)
        self.register_buffer('decay', decay)

    def forward(self):
        f = self.implicit_mlp(self.pe)                                  # h, w, num_filters*dim
        f = f.permute(2, 0, 1).contiguous().view(self.num_filters, self.dim, self.h, self.w)
        return f * self.decay[None, None]


class Hyena2D(nn.Module):
    """Hyena operator: order alternations of long-conv and multiplicative gate."""
    def __init__(self, dim, h, w, order=2, filter_dim=64):
        super().__init__()
        self.dim, self.h, self.w = dim, h, w
        self.order = order
        self.in_proj = nn.Linear(dim, dim * (order + 1))
        self.short_filter = nn.Conv2d(dim * (order + 1), dim * (order + 1),
                                      3, padding=1, groups=dim * (order + 1))
        self.filter_fn = HyenaFilter2D(dim, h, w, filter_dim=filter_dim, num_filters=order)
        self.out_proj = nn.Linear(dim, dim)

    @torch.compiler.disable
    def conv_fft(self, v, h):
        # Linear convolution with a centred (2H-1)x(2W-1) filter, keeping the
        # "same" region: every output sees offsets in all directions. FP32,
        # since half-precision cuFFT needs power-of-two sizes. Kept out of
        # torch.compile: Inductor (2.4) cannot lower the complex64 FFT tensors.
        H, W = self.h, self.w
        Hp, Wp = 3 * H - 2, 3 * W - 2
        v_f = torch.fft.rfft2(v.float(), s=(Hp, Wp), norm='ortho')
        h_f = torch.fft.rfft2(h.float(), s=(Hp, Wp), norm='ortho')
        out = torch.fft.irfft2(v_f * h_f, s=(Hp, Wp), norm='ortho')
        return out[..., H - 1:2 * H - 1, W - 1:2 * W - 1].to(v.dtype)

    def forward(self, x):
        B, N, C = x.shape
        H, W = self.h, self.w

        u = self.in_proj(x).transpose(1, 2).view(B, C * (self.order + 1), H, W)
        u = self.short_filter(u)
        u = u.view(B, self.order + 1, C, H, W)
        v = u[:, 0]

        filters = self.filter_fn()  # order, C, H, W
        direct = hyena_direct_available(v)  # Triton kernel; same result as conv_fft
        for i in range(self.order):
            v = (hyena_conv(v, filters[i]) if direct else self.conv_fft(v, filters[i])) * u[:, i + 1]

        return self.out_proj(v.flatten(2).transpose(1, 2))


class HyenaBlock(nn.Module):
    def __init__(self, dim, h_patches, w_patches, order=2, mlp_ratio=4.0, use_adaln=False):
        super().__init__()
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.hyena = Hyena2D(dim, h_patches, w_patches, order=order)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.Mish(), nn.Linear(hidden, dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        if self.use_adaln and t_emb is not None:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            x = x + s[2].unsqueeze(1) * self.hyena(modulate(self.norm1(x), s[0], s[1]))
            x = x + s[5].unsqueeze(1) * self.mlp(modulate(self.norm2(x), s[3], s[4]))
        else:
            x = x + self.hyena(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x

class NeighborhoodAttention(nn.Module):
    def __init__(self, dim, num_heads=4, kernel_size=7, h=None, w=None):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.scale = self.head_dim ** -0.5
        self.h, self.w = h, w
        self.pad = kernel_size // 2

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.rpb = nn.Parameter(torch.zeros(num_heads, kernel_size ** 2))
        nn.init.trunc_normal_(self.rpb, std=0.02)

    def forward(self, x):
        B, N, C = x.shape
        H, W = self.h, self.w
        kk = self.kernel_size

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, K, V = qkv.unbind(0)  # each: B, heads, N, head_dim

        # Reshape K, V to spatial and unfold k×k neighborhoods (replicate padding)
        def unfold_spatial(t):
            t = t.reshape(B, self.num_heads, H, W, self.head_dim)
            t = t.permute(0, 1, 4, 2, 3).reshape(B * self.num_heads, self.head_dim, H, W)
            t = F.pad(t, [self.pad] * 4, mode='replicate')
            t = F.unfold(t, kernel_size=kk)                                # B*heads, hd*k², N
            return t.view(B, self.num_heads, self.head_dim, kk * kk, N).permute(0, 1, 4, 3, 2)

        K_unf = unfold_spatial(K)  # B, heads, N, k², head_dim
        V_unf = unfold_spatial(V)

        attn = (q.unsqueeze(3) * self.scale) @ K_unf.transpose(-2, -1)     # B, heads, N, 1, k²
        attn = attn + self.rpb[None, :, None, None, :]
        attn = attn.softmax(dim=-1)

        out = (attn @ V_unf).squeeze(3)                                    # B, heads, N, head_dim
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(out)


class NATBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, h_patches, w_patches,
                 kernel_size=7, use_adaln=False):
        super().__init__()
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn = NeighborhoodAttention(dim, num_heads=heads, kernel_size=kernel_size,
                                          h=h_patches, w=w_patches)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.Mish(), nn.Linear(mlp_dim, dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        if self.use_adaln and t_emb is not None:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            x = x + s[2].unsqueeze(1) * self.attn(modulate(self.norm1(x), s[0], s[1]))
            x = x + s[5].unsqueeze(1) * self.mlp(modulate(self.norm2(x), s[3], s[4]))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x

class OutlookAttention(nn.Module):
    """VOLO Outlook: each token predicts a (k²×k²) attention to mix its k×k neighbors."""
    def __init__(self, dim, num_heads=1, kernel_size=3, h=None, w=None):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.scale = self.head_dim ** -0.5
        self.h, self.w = h, w
        self.pad = kernel_size // 2

        self.v = nn.Linear(dim, dim, bias=False)
        self.attn = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.proj = nn.Linear(dim, dim)

        # Cache the fold-overlap divisor (each interior pixel is covered k² times)
        with torch.no_grad():
            ones = torch.ones(1, 1, h, w)
            div = F.fold(F.unfold(ones, kernel_size=kernel_size, padding=self.pad),
                         output_size=(h, w), kernel_size=kernel_size, padding=self.pad)
        self.register_buffer('divisor', div)

    def forward(self, x):
        B, N, C = x.shape
        H, W = self.h, self.w
        kk = self.kernel_size

        # 1. V → unfold k×k neighborhoods
        v = self.v(x).transpose(1, 2).view(B, C, H, W)
        v_unf = F.unfold(v, kernel_size=kk, padding=self.pad)            # B, C*k², N
        v_unf = v_unf.view(B, self.num_heads, self.head_dim, kk * kk, N).permute(0, 1, 4, 3, 2)
        # B, heads, N, k², head_dim

        # 2. Predict attention from each pixel
        attn = self.attn(x).view(B, N, self.num_heads, kk * kk, kk * kk).permute(0, 2, 1, 3, 4)
        attn = (attn * self.scale).softmax(dim=-1)                       # B, heads, N, k², k²

        # 3. Apply and fold overlapping outputs back
        out = attn @ v_unf                                                # B, heads, N, k², head_dim
        out = out.permute(0, 1, 4, 3, 2).reshape(B, C * kk * kk, N)
        out = F.fold(out, output_size=(H, W), kernel_size=kk, padding=self.pad)
        out = out / self.divisor

        return self.proj(out.flatten(2).transpose(1, 2))


class VOLOBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, h_patches, w_patches,
                 kernel_size=3, use_adaln=False):
        super().__init__()
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.outlook = OutlookAttention(dim, num_heads=heads, kernel_size=kernel_size,
                                        h=h_patches, w=w_patches)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.Mish(), nn.Linear(mlp_dim, dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        if self.use_adaln and t_emb is not None:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            x = x + s[2].unsqueeze(1) * self.outlook(modulate(self.norm1(x), s[0], s[1]))
            x = x + s[5].unsqueeze(1) * self.mlp(modulate(self.norm2(x), s[3], s[4]))
        else:
            x = x + self.outlook(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class ASMLPBlock(nn.Module):
    """AS-MLP: axial shift along H and W, with channel groups shifted by varying offsets."""
    def __init__(self, dim, h_patches, w_patches, shift_size=5, mlp_ratio=4.0, use_adaln=False):
        super().__init__()
        self.h, self.w = h_patches, w_patches
        self.shift_size = shift_size
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        self.proj_in = nn.Linear(dim, dim)
        self.proj_h = nn.Linear(dim, dim)
        self.proj_w = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.act = nn.Mish()

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.Mish(), nn.Linear(hidden, dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def _shift(self, x_img, dim_idx):
        # x_img: B, C, H, W. Split channel-wise, roll each chunk by a different amount.
        chunks = list(torch.chunk(x_img, self.shift_size, dim=1))
        for i in range(len(chunks)):
            chunks[i] = torch.roll(chunks[i], shifts=i - self.shift_size // 2, dims=dim_idx)
        return torch.cat(chunks, dim=1)

    def axial_shift(self, x):
        b, n, c = x.shape
        x = self.act(self.proj_in(x))
        x_img = x.transpose(1, 2).view(b, c, self.h, self.w)
        x_w = self._shift(x_img, dim_idx=3).flatten(2).transpose(1, 2)
        x_h = self._shift(x_img, dim_idx=2).flatten(2).transpose(1, 2)
        return self.proj_out(self.act(self.proj_h(x_h)) + self.act(self.proj_w(x_w)))

    def forward(self, x, t_emb=None):
        if self.use_adaln and t_emb is not None:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            x = x + s[2].unsqueeze(1) * self.axial_shift(modulate(self.norm1(x), s[0], s[1]))
            x = x + s[5].unsqueeze(1) * self.mlp(modulate(self.norm2(x), s[3], s[4]))
        else:
            x = x + self.axial_shift(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class S2MLPBlock(nn.Module):
    """S2-MLP: split channels into 4 groups, each shifted in one of 4 cardinal directions."""
    def __init__(self, dim, h_patches, w_patches, mlp_ratio=4.0, use_adaln=False):
        super().__init__()
        assert dim % 4 == 0, "S2-MLP requires dim divisible by 4"
        self.h, self.w = h_patches, w_patches
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.Mish()
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.Mish(), nn.Linear(hidden, dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def _spatial_shift(self, x_img):
        # x_img: B, C, H, W
        b, c, h, w = x_img.shape
        c4 = c // 4
        out = torch.zeros_like(x_img)
        out[:, 0 * c4:1 * c4, :, 1:] = x_img[:, 0 * c4:1 * c4, :, :-1]   # right
        out[:, 1 * c4:2 * c4, :, :-1] = x_img[:, 1 * c4:2 * c4, :, 1:]   # left
        out[:, 2 * c4:3 * c4, 1:, :] = x_img[:, 2 * c4:3 * c4, :-1, :]   # down
        out[:, 3 * c4:4 * c4, :-1, :] = x_img[:, 3 * c4:4 * c4, 1:, :]   # up
        return out

    def token_mix(self, x):
        b, n, c = x.shape
        x = self.act(self.fc1(x))
        x_img = x.transpose(1, 2).view(b, c, self.h, self.w)
        x_img = self._spatial_shift(x_img)
        return self.fc2(x_img.flatten(2).transpose(1, 2))

    def forward(self, x, t_emb=None):
        if self.use_adaln and t_emb is not None:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            x = x + s[2].unsqueeze(1) * self.token_mix(modulate(self.norm1(x), s[0], s[1]))
            x = x + s[5].unsqueeze(1) * self.mlp(modulate(self.norm2(x), s[3], s[4]))
        else:
            x = x + self.token_mix(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class PermuteMLP(nn.Module):
    def __init__(self, dim, h, w, segment_dim=8):
        super().__init__()
        self.h, self.w = h, w
        segment_dim = min(segment_dim, dim)
        while dim % segment_dim != 0:
            segment_dim -= 1
        self.segment_dim = segment_dim
        self.segment_channels = dim // segment_dim

        self.mlp_h = nn.Linear(h * segment_dim, h * segment_dim)
        self.mlp_w = nn.Linear(w * segment_dim, w * segment_dim)
        self.mlp_c = nn.Linear(dim, dim)
        
        self.reweight = nn.Sequential(
            nn.Linear(dim, max(dim // 4, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(dim // 4, 1), dim * 3),
        )
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, t_emb=None):
        B, N, C = x.shape
        if N != self.h * self.w:
            raise ValueError(f"ViP expected {self.h * self.w} tokens, got {N}")
        x_img = x.reshape(B, self.h, self.w, C)
        
        x_h = x_img.reshape(B, self.h, self.w, self.segment_channels, self.segment_dim)
        x_h = x_h.permute(0, 2, 3, 1, 4).reshape(B, self.w, self.segment_channels, self.h * self.segment_dim)
        x_h = self.mlp_h(x_h)
        x_h = x_h.reshape(B, self.w, self.segment_channels, self.h, self.segment_dim)
        x_h = x_h.permute(0, 3, 1, 2, 4).reshape(B, self.h, self.w, C)

        x_w = x_img.reshape(B, self.h, self.w, self.segment_channels, self.segment_dim)
        x_w = x_w.permute(0, 1, 3, 2, 4).reshape(B, self.h, self.segment_channels, self.w * self.segment_dim)
        x_w = self.mlp_w(x_w)
        x_w = x_w.reshape(B, self.h, self.segment_channels, self.w, self.segment_dim)
        x_w = x_w.permute(0, 1, 3, 2, 4).reshape(B, self.h, self.w, C)

        x_c = self.mlp_c(x_img)
        
        # Reweighting Logic
        # Tweak: If you have t_emb, add it to the pool_feat
        pool_feat = x_img.mean(dim=(1, 2)) 
        if t_emb is not None:
            # Assumes t_emb is also 'dim' size
            pool_feat = pool_feat + t_emb 
            
        gate = self.reweight(pool_feat).reshape(B, 3, C).softmax(dim=1)
        
        # Weighted sum with 1,1 broadcasting
        out = (
            x_h * gate[:, 0:1, None, :] +
            x_w * gate[:, 1:2, None, :] +
            x_c * gate[:, 2:3, None, :]
        )
        
        return self.proj(out.reshape(B, N, C))


class ViPBlock(nn.Module):
    def __init__(self, dim, h_patches, w_patches, mlp_ratio=4.0, use_adaln=False):
        super().__init__()
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.permute = PermuteMLP(dim, h_patches, w_patches)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.Mish(), nn.Linear(hidden, dim))
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 6)

    def forward(self, x, t_emb=None):
        if self.use_adaln and t_emb is not None:
            s = self.adaLN_modulation(t_emb).chunk(6, dim=1)
            x = x + s[2].unsqueeze(1) * self.permute(modulate(self.norm1(x), s[0], s[1]), t_emb)
            x = x + s[5].unsqueeze(1) * self.mlp(modulate(self.norm2(x), s[3], s[4]))
        else:
            x = x + self.permute(self.norm1(x), t_emb)
            x = x + self.mlp(self.norm2(x))
        return x


# ==========================================
# 4. Main Model Architectures
# ==========================================

class RINMLP(nn.Module):
    """Pre-normalized residual feed-forward layer used throughout RIN."""
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.net = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.net(self.norm(x))


class RINLatentLayer(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.mlp = RINMLP(dim, dropout)

    def forward(self, latents):
        q = self.norm(latents)
        update, _ = self.attn(q, q, q, need_weights=False)
        return self.mlp(latents + update)


class RINTimeEmbedding(nn.Module):
    """Normalized sinusoidal features followed by the reference's SiLU MLP."""
    def __init__(self, dim, scale):
        super().__init__()
        self.feature_dim = dim // 4
        self.scale = scale
        self.proj = nn.Sequential(nn.Linear(dim // 4, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, time):
        # Pix2Seq's positional_encoding uses 10000**(-2*i/feature_dim),
        # unlike JiT's sinusoidal embedding with a (half_dim - 1) divisor.
        frequencies = 10000.0 ** (-torch.arange(0, self.feature_dim, 2,
                                                device=time.device, dtype=torch.float32) / self.feature_dim)
        angles = (time.float() * self.scale)[:, None] * frequencies[None, :]
        features = torch.cat((angles.sin(), angles.cos()), dim=-1)
        features = F.layer_norm(features, (self.feature_dim,), eps=0.0)
        return self.proj(features.to(dtype=self.proj[0].weight.dtype))


class RINBlock(nn.Module):
    """Read + MLP, K latent Transformer layers, write + MLP (Jabri et al., Alg. 3).

    Only queries are normalized in cross-attention. Interface and latent widths
    are independent; full self-attention is confined to the latent array.
    Reference: https://proceedings.mlr.press/v202/jabri23a/jabri23a.pdf
    """
    def __init__(self, dim, heads, dropout=0.0, latent_dim=None, num_layers=4):
        super().__init__()
        latent_dim = dim if latent_dim is None else latent_dim
        if heads < 1 or dim % heads or latent_dim % heads or num_layers < 1:
            raise ValueError('RIN widths must divide by heads, and num_layers must be positive')
        self.read_norm = nn.LayerNorm(latent_dim, eps=1e-6)
        self.read = nn.MultiheadAttention(latent_dim, heads, kdim=dim, vdim=dim, batch_first=True)
        self.read_mlp = RINMLP(latent_dim)
        self.compute = nn.ModuleList([
            RINLatentLayer(latent_dim, heads, dropout) for _ in range(num_layers)
        ])
        self.write_norm = nn.LayerNorm(dim, eps=1e-6)
        self.write = nn.MultiheadAttention(dim, heads, kdim=latent_dim, vdim=latent_dim, batch_first=True)
        self.write_mlp = RINMLP(dim)

    def forward(self, data, latents):
        read, _ = self.read(self.read_norm(latents), data, data, need_weights=False)
        latents = self.read_mlp(latents + read)
        for layer in self.compute:
            latents = layer(latents)
        write, _ = self.write(self.write_norm(data), latents, latents, need_weights=False)
        return self.write_mlp(data + write), latents


def conv_stem_geometry(img_size, grid_size, initial, maximum):
    """Plan small stride-1/2 convolutions, independently along each image axis.

    Non-dyadic grids use one final adaptive pool (input) / resize (output).
    The returned sizes also let transposed convolutions recover odd dimensions.
    """
    if len(img_size) != 2 or len(grid_size) != 2:
        raise ValueError('Conv-stem image and grid sizes must be (height, width) pairs')
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
           for v in (*img_size, *grid_size, initial, maximum)):
        raise ValueError('Conv-stem sizes and feature counts must be positive integers')
    if initial < 2 or maximum < initial:
        raise ValueError('Conv-stem needs 2 <= initial filters <= max features')
    if any(g > s for s, g in zip(img_size, grid_size)):
        raise ValueError('Conv-stem bottleneck grid cannot exceed the image size')
    sizes, strides, widths = [tuple(img_size)], [], []
    while True:
        stride = tuple(2 if s > g and (s + 1) // 2 >= g else 1
                       for s, g in zip(sizes[-1], grid_size))
        if stride == (1, 1) and strides:
            break
        strides.append(stride)
        widths.append(min(initial * 2 ** (len(strides) - 1), maximum))
        sizes.append(tuple((s + t - 1) // t for s, t in zip(sizes[-1], stride)))
        if stride == (1, 1):
            break
    return sizes, strides, widths


STEM_ACTIVATIONS = {
    0: 'Sigmoid', 1: 'Tanh', 3: 'ReLU', 4: 'SiLU', 5: 'GELU', 6: 'Mish',
    7: 'SwiGLU', 8: 'GEGLU', 9: 'MiGLU', 10: 'ReGLU',
}


class ChannelLayerNorm2d(nn.Module):
    """LayerNorm over channels at each pixel of an NCHW map (ConvNeXt-style).

    Unlike GroupNorm(1, C), which shares one mean/variance across the whole
    map, this rescales every pixel on its own, so no element can exceed about
    sqrt(C) times the per-pixel rms. Parameter names and shapes match
    GroupNorm(1, C), so earlier StemGLU checkpoints still load.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        xf = x.float()
        mean = xf.mean(1, keepdim=True)
        var = (xf - mean).pow(2).mean(1, keepdim=True)
        y = (xf - mean) * torch.rsqrt(var + self.eps)
        return (y * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)).to(x.dtype)


class StemGLU(nn.Module):
    """Gate NCHW feature maps: activation(gate) * value, halving channels.

    The preceding convolution learns separate gate/value projections, and the
    preceding GroupNorm(2) normalizes the two halves independently. Gate and
    value come from the same input, so a hot pixel is hot in both halves and
    the product squares it; over several stages the tails compound. A global
    GroupNorm fixes the map's rms but not those outliers, so the sub-norm here
    works per pixel, bounding each element before the next conv, the token
    projection, the RGB head, or a skip. See https://arxiv.org/abs/2002.05202
    for the GLU activation variants.
    """
    def __init__(self, activation, width):
        super().__init__()
        self.activation = activation
        self.norm = ChannelLayerNorm2d(width)

    def forward(self, x):
        gate, value = x.chunk(2, dim=1)
        return self.norm(self.activation(gate) * value)


def make_stem_activation(choice, width=2):
    """Return an activation and its required pre-activation channel multiplier.

    Feature-count settings describe post-activation widths. GLUs require twice
    that many convolution outputs, so they also increase parameters and work.
    `width` is the post-activation width (used by the GLU output sub-norm).
    """
    if isinstance(choice, bool) or not isinstance(choice, int) or choice not in STEM_ACTIVATIONS:
        raise ValueError(f'Stem activation must be one of {tuple(STEM_ACTIVATIONS)}')
    activations = {0: nn.Sigmoid, 1: nn.Tanh, 3: nn.ReLU, 4: nn.SiLU, 5: nn.GELU, 6: nn.Mish}
    if choice >= 7:
        gate = activations[{7: 4, 8: 5, 9: 6, 10: 3}[choice]]()
        return StemGLU(gate, width), 2
    return activations[choice](), 1


def make_stem_norm(width, expansion):
    """Pre-activation norm; GLUs get one group per half (gate, value)."""
    return nn.GroupNorm(expansion, width * expansion)


def scale_stem_widths(widths, activation, glu_scaling):
    """Apply the transformer-inspired 2/3 rule to post-gate stem widths.

    Round down, with a minimum of two channels for GroupNorm at a 1x1 grid.
    This is width scaling, not exact parameter matching for convolution stacks.
    Non-gated activations keep their original widths in either mode.
    """
    if isinstance(glu_scaling, bool) or not isinstance(glu_scaling, int) or glu_scaling not in (0, 1):
        raise ValueError('Stem GLU scaling must be 0=default or 1=two-thirds widths')
    make_stem_activation(activation)  # Validate direct constructor callers too.
    return [max(2, 2 * width // 3) for width in widths] if glu_scaling and activation >= 7 else widths


def validate_stem_skips(enabled, mode, input_grid=None, output_grid=None):
    if not isinstance(enabled, bool):
        raise ValueError('Stem skip connections must be boolean')
    if enabled and mode != 3:
        raise ValueError('Stem skip connections require both input and output stems')
    if enabled and input_grid != output_grid:
        raise ValueError('Stem skip connections require equal input and output bottleneck grids')


STEM_CONDITIONING = {
    0: 'none', 1: 'encoder only', 2: 'decoder only', 3: 'skips only',
    4: 'encoder and decoder only', 5: 'all stem parts',
}


def validate_stem_conditioning(mode, conv_stem, skips):
    if isinstance(mode, bool) or not isinstance(mode, int) or mode not in STEM_CONDITIONING:
        raise ValueError('stem_conditioning must be an integer from 0 to 5')
    if mode in (1, 4, 5) and not conv_stem & 1:
        raise ValueError('Encoder conditioning requires an input stem')
    if mode in (2, 4, 5) and not conv_stem & 2:
        raise ValueError('Decoder conditioning requires an output stem')
    if mode in (3, 5) and (conv_stem != 3 or not skips):
        raise ValueError('Skip conditioning requires both stems and stem_skips=true')


class StemConditioning(nn.Module):
    """Identity-initialized channel modulation using the class/time embedding."""
    def __init__(self, cond_dim, channels):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, conditioning):
        if conditioning is None:
            raise ValueError('Conditioned stems require a time/class embedding')
        scale, shift = self.proj(F.silu(conditioning)).to(x.dtype).chunk(2, dim=-1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class StemGRN(nn.Module):
    """FCDM-style response normalization for activated stem feature maps."""
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        response = torch.linalg.vector_norm(x.float(), dim=(2, 3), keepdim=True)
        relative = (response / (response.mean(1, keepdim=True) + 1e-6)).to(x.dtype)
        return x + self.gamma.to(x.dtype) * x * relative + self.beta.to(x.dtype)


class ConvStemInput(nn.Module):
    """3x3 conv / GroupNorm / selected activation, then a linear token projection.

    GroupNorm avoids batch statistics for small diffusion training batches.
    One feature-map cell becomes one token; patch sizes are replaced on this side.
    """
    def __init__(self, channels, dim, img_size, grid_size, initial, maximum, activation=4, glu_scaling=0,
                 cond_dim=None, condition_encoder=False, condition_skips=False, use_grn=False):
        super().__init__()
        _, strides, widths = conv_stem_geometry(img_size, grid_size, initial, maximum)
        widths = scale_stem_widths(widths, activation, glu_scaling)
        layers = []
        self.conditioning = nn.ModuleList()
        self.skip_conditioning = nn.ModuleList()
        self.grn = nn.ModuleList()
        for stride, width in zip(strides, widths):
            act, expansion = make_stem_activation(activation, width)
            layers.extend([nn.Conv2d(channels, width * expansion, 3, stride=stride, padding=1, bias=False),
                           make_stem_norm(width, expansion), act])
            if condition_encoder:
                self.conditioning.append(StemConditioning(cond_dim, width * expansion))
            if condition_skips:
                self.skip_conditioning.append(StemConditioning(cond_dim, width))
            if use_grn:
                self.grn.append(StemGRN(width))
            channels = width
        self.layers = nn.Sequential(*layers)
        self.grid_size = tuple(grid_size)
        self.proj = nn.Conv2d(channels, dim, 1)

    def forward(self, x, return_skips=False, conditioning=None):
        skips = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i % 3 == 1 and self.conditioning:
                x = self.conditioning[i // 3](x, conditioning)
            if i % 3 == 2 and return_skips:
                if self.grn:
                    x = self.grn[i // 3](x)
                # Branch-only modulation: never feed skip modulation back into the encoder.
                skip = self.skip_conditioning[i // 3](x, conditioning) if self.skip_conditioning else x
                skips.append(skip)
            elif i % 3 == 2 and self.grn:
                x = self.grn[i // 3](x)
        if x.shape[-2:] != self.grid_size:
            x = F.adaptive_avg_pool2d(x, self.grid_size)
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        return (tokens, skips) if return_skips else tokens


class ConvStemOutput(nn.Module):
    """Mirror the input geometry with transposed convolutions and a linear RGB head.

    The final projection has no normalization or activation: diffusion predictions
    must remain unbounded. Input and output stems do not share weights.
    Optional skips concatenate same-resolution encoder features before each
    transposed convolution, including the deepest encoder stage.
    """
    def __init__(self, channels, dim, img_size, grid_size, initial, maximum, activation=4,
                 glu_scaling=0, use_skips=False, cond_dim=None, condition_decoder=False, use_grn=False):
        super().__init__()
        self.sizes, strides, widths = conv_stem_geometry(img_size, grid_size, initial, maximum)
        widths = scale_stem_widths(widths, activation, glu_scaling)
        self.use_skips = use_skips
        self.grid_size = tuple(grid_size)
        act, expansion = make_stem_activation(activation, widths[-1])
        self.proj = nn.Sequential(nn.Conv2d(dim, widths[-1] * expansion, 1),
                                  make_stem_norm(widths[-1], expansion), act)
        self.conditioning = nn.ModuleList()
        self.grn = nn.ModuleList([StemGRN(widths[-1])]) if use_grn else nn.ModuleList()
        if condition_decoder:
            self.conditioning.append(StemConditioning(cond_dim, widths[-1] * expansion))
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in reversed(range(len(strides))):
            out_width = widths[max(0, i - 1)]
            act, expansion = make_stem_activation(activation, out_width)
            self.layers.append(nn.ConvTranspose2d(widths[i] * (2 if use_skips else 1), out_width * expansion, 3,
                                                  stride=strides[i], padding=1, bias=False))
            self.norms.append(nn.Sequential(make_stem_norm(out_width, expansion), act))
            if condition_decoder:
                self.conditioning.append(StemConditioning(cond_dim, out_width * expansion))
            if use_grn:
                self.grn.append(StemGRN(out_width))
        self.final_conv = nn.Conv2d(widths[0], channels, 3, padding=1)

    def forward(self, x, skips=None, conditioning=None):
        if self.use_skips and (skips is None or len(skips) != len(self.layers)):
            raise ValueError('Decoder requires one encoder skip per stem stage')
        if not self.use_skips and skips is not None:
            raise ValueError('Decoder skip connections are disabled')
        x = x.transpose(1, 2).reshape(x.shape[0], -1, *self.grid_size)
        x = self.proj[1](self.proj[0](x))
        if self.conditioning:
            x = self.conditioning[0](x, conditioning)
        x = self.proj[2](x)
        if self.grn:
            x = self.grn[0](x)
        if x.shape[-2:] != self.sizes[-1]:
            x = F.interpolate(x, size=self.sizes[-1], mode='bilinear', align_corners=False)
        for i, (layer, norm, size) in enumerate(zip(self.layers, self.norms, reversed(self.sizes[:-1]))):
            if self.use_skips:
                skip = skips[-1 - i]
                if skip.shape != x.shape:
                    raise ValueError('Encoder and decoder skip feature shapes must match')
                x = torch.cat((x, skip), dim=1)
            x = norm[0](layer(x, output_size=size))
            if self.conditioning:
                x = self.conditioning[i + 1](x, conditioning)
            x = norm[1](x)
            if self.grn:
                x = self.grn[i + 1](x)
        return self.final_conv(x)


class JiTModel(nn.Module):
    """Shared patch processor with optional input/output convolutional stems.

    conv_stem: 0=off (legacy layout), 1=input, 2=output, 3=both.
    stem_grid is (height, width) in tokens, with one token per stem feature cell.
    One-sided stems must match the ordinary, possibly overlapping patch grid;
    both stems replace patch geometry entirely. MaxViT retains its 8x8 constraint.
    """
    def __init__(self, img_size, patch_size, channels, dim, depth, heads,
                 model_type="jit", self_cond=None,
                 use_adaln=False, use_2d_pos_emb=False, use_conv_mlp=False,
                 bottleneck_dim=None, overlap_h=0, overlap_w=0, axial=False,
                 use_gradient_checkpointing=False,
                 use_qk_norm=False, use_final_adaln=False, time_scale=1.0,
                 bottleneck_act="none", dropout=0.0, rin_num_latents=256,
                 input_channels=None, class_count=0, class_cfg=False, cond_residual=False,
                 conv_stem=0, stem_initial=32, stem_max=256, stem_grid=None, stem_activation=4,
                 stem_glu_scaling=0, stem_skips=False, stem_conditioning=0, stem_grn=False,
                 rin_latent_dim=None, rin_layers_per_block=4, fcdm_mlp_ratio=3., use_grn=False,
                 attn_double_norm=True, arch_version=1, hat_group_depth=2):
        super().__init__()
        self.attn_double_norm = attn_double_norm
        if arch_version not in (1, 2):
            raise ValueError('arch_version must be 1 (legacy blocks) or 2 (reference-faithful blocks)')
        self.arch_version = arch_version
        self.hat_group_depth = hat_group_depth
        if model_type == 'fcdm_isotropic':
            use_adaln = use_final_adaln = True
        self.fcdm_mlp_ratio = fcdm_mlp_ratio
        self.use_grn = use_grn
        self_cond = model_type == 'rin' if self_cond is None else self_cond
        self.latent_self_cond_enabled = model_type == 'rin' and self_cond
        if model_type == 'rin':
            # RIN uses latent recurrence, learned positions and LayerNorm, not
            # JiT's optional pixel self-conditioning / adaLN / bottleneck.
            self_cond = False
            use_adaln = use_final_adaln = use_2d_pos_emb = False
            bottleneck_dim = None
        self.channels = channels
        self.input_channels = channels if input_channels is None else input_channels
        self.cond_residual = cond_residual and self.input_channels == 2 * channels
        self.self_cond = self_cond
        self.patch_size = patch_size
        self.depth = depth
        self.dropout = dropout
        self.use_adaln = use_adaln
        self.use_2d_pos_emb = use_2d_pos_emb
        self.use_conv_mlp = use_conv_mlp
        self.use_qk_norm = use_qk_norm
        self.model_type = model_type
        self.rin_num_latents = rin_num_latents
        self.rin_latent_dim = 2 * dim if rin_latent_dim is None else rin_latent_dim
        self.rin_layers_per_block = rin_layers_per_block
        self.axial = axial
        self.use_gradient_checkpointing = use_gradient_checkpointing
        if isinstance(conv_stem, bool) or not isinstance(conv_stem, int) or conv_stem not in range(4):
            raise ValueError('conv_stem must be 0=off, 1=input, 2=output, or 3=both')
        self.conv_stem = conv_stem
        validate_stem_skips(stem_skips, conv_stem)
        self.stem_skips = stem_skips
        validate_stem_conditioning(stem_conditioning, conv_stem, stem_skips)
        self.stem_conditioning = stem_conditioning
        if not isinstance(stem_grn, bool):
            raise ValueError('stem_grn must be boolean')
        if stem_grn and not conv_stem:
            raise ValueError('stem_grn requires a convolutional stem')
        self.stem_grn = stem_grn

        h_img, w_img = img_size
        p_h, p_w = patch_size

        # Overlap setup
        self.overlap_h = overlap_h
        self.overlap_w = overlap_w
        self.stride_h = p_h - overlap_h
        self.stride_w = p_w - overlap_w
        if conv_stem != 3:
            if (min(p_h, p_w) < 1 or min(overlap_h, overlap_w) < 0
                    or self.stride_h <= 0 or self.stride_w <= 0
                    or p_h > h_img or p_w > w_img
                    or (h_img - p_h) % self.stride_h or (w_img - p_w) % self.stride_w):
                raise ValueError("patch_size and overlap must tile img_size exactly")
            patch_grid = ((h_img - p_h) // self.stride_h + 1,
                          (w_img - p_w) // self.stride_w + 1)
        else:
            patch_grid = tuple(img_size)
        grid = patch_grid if stem_grid is None or not conv_stem else tuple(stem_grid)
        if conv_stem:
            conv_stem_geometry(img_size, grid, stem_initial, stem_max)
            if conv_stem != 3 and grid != patch_grid:
                raise ValueError('One-sided conv-stem grid must match the ordinary patch grid')
            if model_type == 'maxvit' and any(g % 8 for g in grid):
                raise ValueError('MaxViT stem grid height and width must be multiples of 8')
        self.h_patches, self.w_patches = grid
        if model_type == 'maxvit' and any(g % 8 for g in grid):
            raise ValueError(f'MaxViT needs token-grid sides divisible by 8 (8x8 block/grid partitions), got {grid}')
        self.num_patches = self.h_patches * self.w_patches
        print(f"│  Token grid: {self.h_patches}×{self.w_patches} ({self.num_patches} tokens, conv-stem {conv_stem})")

        if conv_stem != 3:
            self.unfold = nn.Unfold(kernel_size=patch_size, stride=(self.stride_h, self.stride_w))
            self.fold = nn.Fold(output_size=img_size, kernel_size=patch_size, stride=(self.stride_h, self.stride_w))

            with torch.no_grad():
                ones_img = torch.ones(1, channels, h_img, w_img)
                divisor = self.fold(self.unfold(ones_img))
                self.register_buffer('overlap_divisor', divisor)

        patch_dim = channels * p_h * p_w
        input_patch_dim = self.input_channels * p_h * p_w
        self.self_cond_patch_dim = patch_dim
        # Pixel self-conditioning predicts the target image, while Pix2Pix
        # additionally supplies a source image in the regular input channels.
        in_channels = input_patch_dim + (patch_dim if self_cond else 0)

        if conv_stem & 1:
            self.to_patch_embedding = ConvStemInput(
                self.input_channels + (channels if self_cond else 0), dim,
                img_size, grid, stem_initial, stem_max, stem_activation, stem_glu_scaling,
                self.rin_latent_dim if model_type == 'rin' else dim,
                stem_conditioning in (1, 4, 5), stem_conditioning in (3, 5), stem_grn)
        elif bottleneck_dim is not None and bottleneck_dim < dim:
            # Paper (Sec. 4.2 / Fig. 4): the bottleneck embedding is a pair of
            # *linear* layers -- a low-rank reparameterization, with no
            # nonlinearity between them. `bottleneck_act="mish"` is kept only for
            # backward compatibility with checkpoints trained with the older
            # nonlinear bottleneck (it also shifts the second Linear's index,
            # so the two variants have distinct, self-consistent state_dicts).
            embed = [nn.Linear(in_channels, bottleneck_dim)]
            if bottleneck_act == "mish":
                embed.append(nn.Mish())
            embed.append(nn.Linear(bottleneck_dim, dim))
            self.to_patch_embedding = nn.Sequential(*embed)
        else:
            self.to_patch_embedding = nn.Linear(in_channels, dim)

        if model_type == 'fcdm_isotropic':
            self.register_parameter('pos_embedding', None)
        elif use_2d_pos_emb:
            self.register_buffer('pos_embedding',
                get_2d_sincos_pos_embed(dim, self.h_patches, self.w_patches).reshape(1, self.num_patches, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)

        if model_type == 'rin':
            if self.rin_latent_dim < 16 or self.rin_latent_dim % 8:
                raise ValueError('RIN latent width must be a multiple of 8 and at least 16')
            self.time_mlp = RINTimeEmbedding(self.rin_latent_dim, time_scale)
        elif model_type == 'fcdm_isotropic':
            self.time_mlp = FCDMTimeEmbedding(dim, time_scale)
        else:
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim, scale=time_scale),
                nn.Linear(dim, dim * 4),
                nn.Mish(),
                nn.Linear(dim * 4, dim)
            )
        self.null_class_id = class_count if class_count > 0 and class_cfg else None
        cond_dim = self.rin_latent_dim if model_type == 'rin' else dim
        self.class_embedding = nn.Embedding(class_count + int(class_cfg), cond_dim) if class_count > 0 else None

        # Models that handle position internally (RoPE, convolutions, etc.)
        NO_POS_EMBED = {'jit', 'gmlp', 'oggmlp', 'amlp', 'ogamlp', 'mlpmixer',
                        'ogmlpmixer', 'pool', 'gru', 'convnext', 'fullattn',
                        'mixer_attn', 'lka', 'resmlp', 'mdmlp',
                        'hat', 'xcit', 'coatnet', 'hornet', 'aft_full', 'aft_local', 'hyena',
                        'nat', 'volo', 's2mlp', 'vip', 'fcdm_isotropic'}
        self.skip_pos_embed = model_type in NO_POS_EMBED

        if model_type == "rin":
            # The reference counts time/class tokens within the latent budget.
            learned_slots = rin_num_latents - 1 - int(class_count > 0)
            if learned_slots < 1:
                raise ValueError('RIN needs at least one learned slot in addition to time/class tokens')
            self.latent_tokens = nn.Parameter(torch.empty(1, learned_slots, self.rin_latent_dim))
            nn.init.trunc_normal_(self.latent_tokens, std=0.02, a=-0.04, b=0.04)
            nn.init.trunc_normal_(self.pos_embedding, std=0.02, a=-0.04, b=0.04)
            self.stem_norm = nn.LayerNorm(dim, eps=1e-6)
            if self.latent_self_cond_enabled:
                # MLP includes the residual: LN(previous + FFN(LN(previous))).
                self.latent_prev_proj = RINMLP(self.rin_latent_dim)
                self.latent_prev_norm = nn.LayerNorm(self.rin_latent_dim, eps=1e-6)
                nn.init.zeros_(self.latent_prev_norm.weight)
                nn.init.zeros_(self.latent_prev_norm.bias)
            self.rin_blocks = nn.ModuleList([
                RINBlock(dim, heads, dropout=dropout, latent_dim=self.rin_latent_dim,
                         num_layers=rin_layers_per_block) for _ in range(depth)
            ])
            self.layers = nn.ModuleList()
            self.supports_latent_self_conditioning = self.latent_self_cond_enabled
        else:
            self.layers = nn.ModuleList()
            for i in range(depth):
                block = self._make_block(model_type, dim, heads, i)
                if use_grn and model_type in GRN_TOKEN_MODELS:
                    block = GRNResidualBlock(block, dim)
                self.layers.append(block)
            self.supports_latent_self_conditioning = False

        self.norm = (nn.LayerNorm(dim, eps=1e-6, elementwise_affine=model_type != 'fcdm_isotropic')
                     if model_type in {'rin', 'convnext', 'fcdm_isotropic'} else RMSNorm(dim))
        # DiT-style final layer: optionally modulate the output head with
        # adaLN-Zero before projecting back to pixels (zero-init => identity
        # at start, so it never disrupts early training).
        self.use_final_adaln_active = use_final_adaln and use_adaln
        if self.use_final_adaln_active:
            self.final_adaLN = nn.Sequential(nn.SiLU() if model_type == 'fcdm_isotropic' else nn.Mish(),
                                             nn.Linear(dim, 2 * dim, bias=True))
            nn.init.constant_(self.final_adaLN[-1].weight, 0)
            nn.init.constant_(self.final_adaLN[-1].bias, 0)
        self.to_pixels = (ConvStemOutput(channels, dim, img_size, grid, stem_initial, stem_max,
                                        stem_activation, stem_glu_scaling, stem_skips,
                                        self.rin_latent_dim if model_type == 'rin' else dim,
                                        stem_conditioning in (2, 4, 5), stem_grn)
                          if conv_stem & 2 else nn.Linear(dim, patch_dim))
        if model_type == 'fcdm_isotropic':
            output = self.to_pixels.final_conv if conv_stem & 2 else self.to_pixels
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def _make_block(self, model_type, dim, heads, layer_idx):
        hp, wp, np_ = self.h_patches, self.w_patches, self.num_patches
        adaln = self.use_adaln

        if model_type == "jit":
            # Paper applies dropout to the middle half of the blocks.
            mid_lo, mid_hi = self.depth // 4, self.depth - self.depth // 4
            drop = self.dropout if (mid_lo <= layer_idx < mid_hi) else 0.0
            return TransformerBlock(dim, heads, dim * 4, hp, wp, use_adaln=adaln,
                                    use_conv_mlp=self.use_conv_mlp, qk_norm=self.use_qk_norm,
                                    dropout=drop, attn_double_norm=self.attn_double_norm)
        elif model_type == "oggmlp":
            if self.axial:
                return AxialgMLPBlock(dim, hp, wp, expansion_factor=4, use_adaln=adaln)
            return gMLPBlock_v5(dim, np_, expansion_factor=4, use_adaln=adaln, tiny_attn=False)
        elif model_type == "ogamlp":
            if self.axial:
                return AxialgMLPBlock(dim, hp, wp, expansion_factor=4, use_adaln=adaln)
            return gMLPBlock_v5(dim, np_, expansion_factor=4, use_adaln=adaln, tiny_attn=True)
        elif model_type == "basemlp":
            return BaseMLPBlock(dim, hp, wp)
        elif model_type == "mlp":
            return TransformerMLPBlock(dim, dim * 4, hp, wp, use_conv_mlp=False)
        elif model_type == "mlpmixer":
            return ConvMixerBlock(dim, hp, wp)
        elif model_type == "hypermixer":
            return HybridHyperBlock(dim, np_, hp, wp, heads=heads)
        elif model_type == "ogmlpmixer":
            if self.axial:
                return AxialMixerBlock(dim, hp, wp, dim // 2, dim * 4, use_adaln=adaln)
            return MixerBlock(dim, np_, dim // 2, dim * 4, use_adaln=adaln)
        elif model_type == "oghypermixer":
            return HyperMixerBlock(dim, np_, heads=1, use_adaln=adaln)
        elif model_type == "fcdm_isotropic":
            return FCDMTokenBlock(dim, hp, wp, self.fcdm_mlp_ratio)
        elif model_type == "convnext":
            return ConvNeXtBlock(dim, hp, wp, drop_path=0.0, use_adaln=adaln)
        elif model_type == "fullattn":
            return FullAttentionBlock(dim, hp, wp, heads=heads, use_adaln=adaln, qk_norm=self.use_qk_norm,
                                      attn_double_norm=self.attn_double_norm)
        elif model_type == "pool":
            return ConvFormerBlock(dim, dim * 4, hp, wp)
        elif model_type == "fourier":
            return FourierMixerBlock(dim, dim * 4, hp, wp)
        elif model_type == "gru":
            return RNNBlock(dim, dim * 4, hp, wp)
        elif model_type == "lka":
            return LKABlock(dim, dim * 4, hp, wp, use_adaln=adaln)
        elif model_type == "xcit":
            return XCiTBlock(dim, heads, dim * 4, hp, wp, use_adaln=adaln)
        elif model_type == "hat" and self.arch_version == 2:
            return HATGroupBlock(dim, heads, hp, wp, depth=self.hat_group_depth, use_adaln=adaln)
        elif model_type == "hat":
            ws = hp
            shift = ws // 2 if (layer_idx % 2 != 0) else 0
            return HATBlock(dim, heads, window_size=ws, shift_size=shift, h_patches=hp, w_patches=wp, use_adaln=adaln)
        elif model_type == "cyclemlp":
            return CycleMLPBlock(dim, dim * 4, hp, wp, use_adaln=adaln)
        elif model_type == "resmlp":
            return ResMLPBlock(dim, np_, mlp_ratio=4.0, use_adaln=adaln)
        elif model_type == "mdmlp":
            return MDMLPBlock(dim, hp, wp, mlp_ratio=4.0, use_adaln=adaln)
        elif model_type == "bigs":
            return BiGSBlock(dim, dim * 4, hp, wp, use_adaln=adaln)
        elif model_type == "coatnet":
            # Alternate MBConv and RelativeTransformer blocks
            if layer_idx % 2 == 0:
                return MBConvBlock(dim, hp, wp, expansion=4, use_adaln=adaln)
            else:
                return CoAtNetTransformerBlock(dim, heads, dim * 4, hp, wp, use_adaln=adaln)
        elif model_type == "gatedmlpmixer":
            return GatedMixerBlock(dim, np_, dim // 2, dim * 4, use_adaln=adaln)
        elif model_type == "mixer_attn":
            return MixerAttnBlock(dim, np_, dim // 2, dim * 4, heads, use_adaln=adaln,
                                  attn_double_norm=self.attn_double_norm)
        elif model_type == "gmlp":
            return gMLPBlock_Conv(dim, np_, hp, wp, use_adaln=adaln, tiny_attn=False)
        elif model_type == "amlp":
            return gMLPBlock_Conv(dim, np_, hp, wp, use_adaln=adaln, tiny_attn=True)
        elif model_type == "vim" and self.arch_version == 2:
            return VimReferenceBlock(dim, depth=self.depth, use_adaln=adaln)
        elif model_type == "vim":
            return VisionMambaBlock(dim, dim * 4, hp, wp, use_adaln=adaln)
        elif model_type == "maxvit" and self.arch_version == 2:
            return MaxViTReferenceBlock(dim, heads, hp, wp, use_adaln=adaln)
        elif model_type == "maxvit":
            return MaxViTBlock(dim, heads, dim * 4, hp, wp, use_adaln=adaln)
        elif model_type == "focal":
            return FocalNetBlock(dim, dim * 4, hp, wp, use_adaln=adaln)
        elif model_type == "hornet":
            return HorNetBlock(dim, hp, wp, order=5, mlp_ratio=4.0, use_adaln=adaln)
        elif model_type == "aft_full":
            return AFTBlock(dim, np_, hp, wp, dim * 4, variant='full', use_adaln=adaln)
        elif model_type == "aft_simple":
            return AFTBlock(dim, np_, hp, wp, dim * 4, variant='simple', use_adaln=adaln)
        elif model_type == "aft_local":
            return AFTBlock(dim, np_, hp, wp, dim * 4, variant='local', window_size=7, use_adaln=adaln)
        elif model_type == "hyena":
            return HyenaBlock(dim, hp, wp, order=2, mlp_ratio=4.0, use_adaln=adaln)
        elif model_type == "nat":
            return NATBlock(dim, heads, dim * 4, hp, wp, kernel_size=7, use_adaln=adaln)
        elif model_type == "volo":
            return VOLOBlock(dim, heads, dim * 4, hp, wp, kernel_size=3, use_adaln=adaln)
        elif model_type == "asmlp":
            return ASMLPBlock(dim, hp, wp, shift_size=5, mlp_ratio=4.0, use_adaln=adaln)
        elif model_type == "s2mlp":
            return S2MLPBlock(dim, hp, wp, mlp_ratio=4.0, use_adaln=adaln)
        elif model_type == "vip":
            return ViPBlock(dim, hp, wp, mlp_ratio=4.0, use_adaln=adaln)
        else:
            return gMLPBlock_Conv(dim, np_, hp, wp, use_adaln=adaln, tiny_attn=True)

    def forward(self, x, time, x_self_cond=None, latent_self_cond=None,
                return_latents=False, class_labels=None):
        b, c, h, w = x.shape
        # Pix2Pix input is [noisy target, conditioning image].
        cond = x[:, self.channels:] if self.cond_residual else None

        t = self.time_mlp(time)
        class_emb = None
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("Class-conditional model requires class labels")
            class_emb = self.class_embedding(class_labels)
            if self.model_type != 'rin':
                t = t + class_emb
        stem_context = t + class_emb if self.model_type == 'rin' and class_emb is not None else t

        # 1. Patch extraction via unfold
        if self.conv_stem & 1:
            if self.self_cond:
                if x_self_cond is None:
                    x_self_cond = x.new_zeros(b, self.channels, h, w)
                x = torch.cat((x, x_self_cond), dim=1)
            x_patches = x
        else:
            x_patches = self.unfold(x).transpose(1, 2)  # B, N, patch_dim
            if self.self_cond:
                if x_self_cond is None:
                    x_self_cond_patches = x_patches.new_zeros(
                        x_patches.shape[0], x_patches.shape[1], self.self_cond_patch_dim,
                    )
                else:
                    x_self_cond_patches = self.unfold(x_self_cond).transpose(1, 2)
                x_patches = torch.cat((x_patches, x_self_cond_patches), dim=-1)

        # 2. Embedding
        skips = None
        if self.stem_skips:
            x, skips = self.to_patch_embedding(x_patches, return_skips=True, conditioning=stem_context)
        else:
            x = (self.to_patch_embedding(x_patches, conditioning=stem_context) if self.conv_stem & 1
                 else self.to_patch_embedding(x_patches))

        if self.model_type == 'rin':
            x = self.stem_norm(x)

        # 3. Positional embedding
        if not self.skip_pos_embed:
            x = x + self.pos_embedding

        # 4. Time embedding. Blocks without an adaLN path always need it here.
        if ((not self.use_adaln or self.model_type in INPUT_TIME_MODELS)
                and self.model_type not in ["basemlp", "rin"]):
            x = x + t.unsqueeze(1)

        # 5. Blocks. RIN conditions learned latent slots on time and on the
        # previous reverse-diffusion pass, then routes data ↔ latents repeatedly.
        latents = None
        if self.model_type == "rin":
            tokens = [self.latent_tokens.expand(b, -1, -1), t.unsqueeze(1)]
            if class_emb is not None:
                tokens.append(class_emb.unsqueeze(1))
            latents = torch.cat(tokens, dim=1)
            if self.latent_self_cond_enabled:
                if latent_self_cond is None:
                    latent_self_cond = torch.zeros_like(latents)
                if latent_self_cond.shape != latents.shape:
                    raise ValueError('Previous RIN latents must match batch, slot count and latent width')
                latent_self_cond = latent_self_cond.detach().to(device=latents.device, dtype=latents.dtype)
                latents = latents + self.latent_prev_norm(self.latent_prev_proj(latent_self_cond))
            for layer in self.rin_blocks:
                if self.use_gradient_checkpointing and self.training:
                    x, latents = torch.utils.checkpoint.checkpoint(layer, x, latents, use_reentrant=False)
                else:
                    x, latents = layer(x, latents)
        else:
            for layer in self.layers:
                if self.model_type == "mlp":
                    x = x + t.unsqueeze(1)
                if self.use_gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(layer, x, t, use_reentrant=False)
                else:
                    x = layer(x, t_emb=t)

        # 6. Output
        if self.model_type != "basemlp":
            x = self.norm(x)
            if self.use_final_adaln_active:
                shift, scale = self.final_adaLN(t).chunk(2, dim=1)
                x = modulate(x, shift, scale)
        x = (self.to_pixels(x, skips=skips, conditioning=stem_context) if self.conv_stem & 2
             else self.to_pixels(x))

        # 7. Fold back with overlap normalization
        if not self.conv_stem & 2:
            x = self.fold(x.transpose(1, 2))
            x = x / self.overlap_divisor
        if cond is not None:
            x = x + cond
        return (x, latents) if return_latents else x


# ==========================================
# 5. ConvNet Architectures (UNet / EncDec)
# ==========================================

class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim, groups=8, dropout=0.0):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_c), nn.Mish(), nn.Conv2d(in_c, out_c, 3, padding=1))
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, out_c), nn.Mish(), nn.Dropout(dropout), nn.Conv2d(out_c, out_c, 3, padding=1))
        self.res_conv = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.time_proj = nn.Linear(time_emb_dim, out_c)

    def forward(self, x, t_emb):
        h = self.block1(x)
        if self.time_proj is not None:
            h = h + self.time_proj(F.mish(t_emb))[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class ConvNetModel(nn.Module):
    def __init__(self, img_size, channels, dim, fmap_max, bottleneck_res,
                 model_type="unet", num_res_blocks=1, time_scale=1.0,
                 input_channels=None, class_count=0, class_cfg=False, cond_residual=False, arch_version=1):
        super().__init__()
        if arch_version not in (1, 2):
            raise ValueError('arch_version must be 1 (legacy blocks) or 2 (reference-faithful blocks)')
        self.model_type = model_type
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.cond_residual = cond_residual and input_channels == 2 * channels
        self.input_channels = channels if input_channels is None else input_channels
        self.num_levels = int(math.log2(img_size[0]) - math.log2(bottleneck_res))

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim, scale=time_scale), nn.Linear(dim, time_dim), nn.Mish(), nn.Linear(time_dim, time_dim))
        self.null_class_id = class_count if class_count > 0 and class_cfg else None
        self.class_embedding = nn.Embedding(class_count + int(class_cfg), time_dim) if class_count > 0 else None

        self.init_conv = nn.Conv2d(self.input_channels, dim, 3, padding=1)

        def swin(in_c, out_c, level):
            # arch_version 2: reference shifted-window masks (and Swin v2 log-CPB attention).
            if arch_version == 2:
                resolution = (img_size[0] >> level, img_size[1] >> level)
                return SwinReferenceAdapter(in_c, out_c, time_dim, resolution, version=model_type.split('_')[1])
            return SwinBlockAdapter(in_c, out_c, time_dim, version=model_type.split('_')[1])

        self.downs = nn.ModuleList()
        dims = [dim]
        curr_dim = dim

        for level in range(self.num_levels):
            out_dim = make_divisible(min(curr_dim * 2, fmap_max), 8)
            layers = nn.ModuleList()
            for __ in range(num_res_blocks):
                if "swin" in model_type:
                    layers.append(swin(curr_dim, curr_dim, level))
                else:
                    layers.append(ResnetBlock(curr_dim, curr_dim, time_dim))
            self.downs.append(nn.ModuleList([layers, nn.Conv2d(curr_dim, out_dim, 4, stride=2, padding=1)]))
            dims.append(out_dim)
            curr_dim = out_dim

        mid_dim = dims[-1]
        if "swin" in model_type:
            self.mid_block1 = swin(mid_dim, mid_dim, self.num_levels)
            self.mid_block2 = swin(mid_dim, mid_dim, self.num_levels)
        else:
            self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_dim)
            self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_dim)
        self.mid_attn = Attention(mid_dim, heads=8, dim_head=64)

        self.ups = nn.ModuleList()
        is_skip = model_type == "unet" or "swin" in model_type
        for i in reversed(range(self.num_levels)):
            in_dim, out_dim = dims[i + 1], dims[i]
            layers = nn.ModuleList()
            for j in range(num_res_blocks):
                res_in = (in_dim + out_dim if is_skip else in_dim) if j == 0 else out_dim
                if "swin" in model_type:
                    layers.append(swin(res_in, out_dim, i))
                else:
                    layers.append(ResnetBlock(res_in, out_dim, time_dim))
            self.ups.append(nn.ModuleList([nn.ConvTranspose2d(in_dim, in_dim, 2, stride=2), layers]))

        self.is_skip = is_skip
        self.final_norm = nn.GroupNorm(8, dim)
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def forward(self, x, time, x_self_cond=None, class_labels=None):
        cond = x[:, self.channels:] if self.cond_residual else None
        t = self.time_mlp(time)
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("Class-conditional model requires class labels")
            t = t + self.class_embedding(class_labels)
        x = self.init_conv(x)
        skips = []

        for layers, downsample in self.downs:
            for block in layers:
                x = block(x, t)
            if self.is_skip:
                skips.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(b, -1, c)
        x_flat = x_flat + self.mid_attn(x_flat)
        x = x_flat.reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = self.mid_block2(x, t)

        for upsample, layers in self.ups:
            x = upsample(x)
            if self.is_skip:
                x = torch.cat((x, skips.pop()), dim=1)
            for block in layers:
                x = block(x, t)

        x = self.final_conv(self.final_norm(x))
        return x + cond if cond is not None else x


class HierMLPStage(nn.Module):
    """Refine every non-overlapping parent patch into a local 2x2 child grid."""
    def __init__(self, patch_dim, parent_dim, time_dim, hidden_dim, out_dim, layer_count, num_children=4):
        super().__init__()
        if layer_count < 1:
            raise ValueError("HierMLP layer_count must be at least 1")

        input_dim = patch_dim + parent_dim + time_dim + 2
        layers = []
        current_dim = input_dim
        for _ in range(layer_count):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.Mish()])
            current_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_children * out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, patch, coords, time_emb, parent=None):
        batch, num_patches, _ = patch.shape
        inputs = [patch, coords.expand(batch, -1, -1), time_emb.unsqueeze(1).expand(-1, num_patches, -1)]
        if parent is not None:
            inputs.append(parent)
        return self.mlp(torch.cat(inputs, dim=-1))


def _make_hier_global_block(mixer_type, dim, grid_size, heads, attn_double_norm=True):
    """Build one processor for HierMLP's fine input-token grid."""
    token_count = grid_size * grid_size
    if mixer_type == 'jit':
        return TransformerBlock(dim, heads, dim * 4, grid_size, grid_size,
                                use_adaln=True, qk_norm=True, attn_double_norm=attn_double_norm)
    if mixer_type == 'mlpmixer':
        return MixerBlock(dim, token_count, max(dim // 2, 1), dim * 4, use_adaln=True)
    if mixer_type == 'gmlp':
        return gMLPBlock_v5(dim, token_count, expansion_factor=4, use_adaln=True, tiny_attn=False)
    if mixer_type == 'convnext':
        return ConvNeXtBlock(dim, grid_size, grid_size, drop_path=0.0, use_adaln=True)
    if mixer_type == 'vip':
        return ViPBlock(dim, grid_size, grid_size, mlp_ratio=4.0, use_adaln=True)
    raise ValueError(f"Unknown HierMLP global mixer {mixer_type!r}")


class HierMLPModel(nn.Module):
    """Coarse-to-fine image model with a global input grid and local refinement.

    A selected processor first mixes a fine input grid. A learned 2D merge then
    produces a possibly smaller coarse output grid; every following 2x2 stage
    is local, conditioned on its parent feature, position, timestep, and the
    matching raw-image patch. The processor has independent width and depth;
    equal grids plus matching widths/depths retain the legacy MLP-Mixer
    architecture and checkpoint layout.

    Conv-stem input replaces global patch embedding at input_grid_size; output
    replaces the local refinement tree with a decoder from output_grid_size.
    The existing nested square-grid requirements apply in every stem mode.
    """
    def __init__(self, img_size, channels, dim, fmap_max, layer_count,
                 initial_grid_size=4, time_scale=1.0, input_channels=None,
                 class_count=0, class_cfg=False, input_grid_size=None,
                 output_grid_size=None, global_mixer='mlpmixer', global_heads=4,
                 global_dim=None, global_depth=None, cond_residual=False,
                 conv_stem=0, stem_initial=32, stem_max=256, stem_activation=4,
                 stem_glu_scaling=0, stem_skips=False, stem_conditioning=0, stem_grn=False,
                 attn_double_norm=True):
        super().__init__()
        if isinstance(conv_stem, bool) or not isinstance(conv_stem, int) or conv_stem not in range(4):
            raise ValueError('conv_stem must be 0=off, 1=input, 2=output, or 3=both')
        self.conv_stem = conv_stem
        h_img, w_img = img_size
        self.cond_residual = cond_residual and input_channels == 2 * channels
        output_grid_size = initial_grid_size if output_grid_size is None else output_grid_size
        input_grid_size = output_grid_size if input_grid_size is None else input_grid_size
        validate_stem_skips(stem_skips, conv_stem, input_grid_size, output_grid_size)
        self.stem_skips = stem_skips
        validate_stem_conditioning(stem_conditioning, conv_stem, stem_skips)
        self.stem_conditioning = stem_conditioning
        if not isinstance(stem_grn, bool):
            raise ValueError('stem_grn must be boolean')
        if stem_grn and not conv_stem:
            raise ValueError('stem_grn requires a convolutional stem')
        self.stem_grn = stem_grn
        global_dim = dim if global_dim is None else global_dim
        global_depth = layer_count if global_depth is None else global_depth
        if input_grid_size < 1 or output_grid_size < 1:
            raise ValueError("HierMLP input and output grid sizes must be at least 1")
        if global_dim < 1 or global_depth < 1 or global_heads < 1:
            raise ValueError("HierMLP global dimension, depth, and head count must be at least 1")
        if input_grid_size < output_grid_size or input_grid_size % output_grid_size:
            raise ValueError("HierMLP input grid size must be an integer multiple of output grid size")
        if (h_img < input_grid_size or w_img < input_grid_size
                or h_img % input_grid_size or w_img % input_grid_size):
            raise ValueError("HierMLP image height and width must be divisible by input_grid_size")

        h_steps = int(math.log2(h_img // output_grid_size)) if h_img > output_grid_size else 0
        w_steps = int(math.log2(w_img // output_grid_size)) if w_img > output_grid_size else 0
        if (h_steps != w_steps or output_grid_size * (2 ** h_steps) != h_img
                or output_grid_size * (2 ** w_steps) != w_img):
            raise ValueError("HierMLP requires each image dimension to equal output_grid_size * 2^n with the same n")

        self.channels = channels
        self.input_channels = channels if input_channels is None else input_channels
        # Keep `initial_grid_size` as the output-grid alias for old callers and
        # checkpoints, while making the input representation independently settable.
        self.initial_grid_size = output_grid_size
        self.input_grid_size = input_grid_size
        self.output_grid_size = output_grid_size
        self.global_mixer = global_mixer
        self.global_heads = global_heads
        self.global_dim = global_dim
        self.global_depth = global_depth
        self.input_to_output_ratio = input_grid_size // output_grid_size
        self.num_refinements = 0 if conv_stem & 2 else h_steps
        self.layer_count = layer_count
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim, scale=time_scale), nn.Linear(dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim))
        # The local refinement tree retains `dim`; project its time/class
        # conditioning only when the independently sized global processor needs it.
        self.global_time_proj = nn.Identity() if global_dim == dim else nn.Linear(dim, global_dim)
        self.null_class_id = class_count if class_count > 0 and class_cfg else None
        self.class_embedding = nn.Embedding(class_count + int(class_cfg), dim) if class_count > 0 else None

        # The selected processor sees finer input patches before a learned 2D
        # merge establishes the coarse grid used by local refinements.
        input_token_count = input_grid_size * input_grid_size
        input_patch_dim = self.input_channels * (h_img // input_grid_size) * (w_img // input_grid_size)
        self.coarse_patch_embedding = (ConvStemInput(
            self.input_channels, global_dim, img_size, (input_grid_size, input_grid_size),
            stem_initial, stem_max, stem_activation, stem_glu_scaling, dim,
            stem_conditioning in (1, 4, 5), stem_conditioning in (3, 5), stem_grn) if conv_stem & 1 else nn.Linear(input_patch_dim, global_dim))
        self.coarse_pos_embedding = nn.Parameter(torch.randn(1, input_token_count, global_dim) * 0.02)
        self.coarse_mixer = nn.ModuleList([
            _make_hier_global_block(global_mixer, global_dim, input_grid_size, global_heads,
                                    attn_double_norm=attn_double_norm)
            for _ in range(global_depth)
        ])
        merge_dim = self.input_to_output_ratio ** 2 * global_dim
        self.input_to_output = (nn.Identity() if self.input_to_output_ratio == 1 and global_dim == dim
                                else nn.Linear(merge_dim, dim))
        self.draft_to_pixels = (ConvStemOutput(
            channels, dim, img_size, (output_grid_size, output_grid_size), stem_initial, stem_max,
            stem_activation, stem_glu_scaling, stem_skips, dim, stem_conditioning in (2, 4, 5), stem_grn)
            if conv_stem & 2 else (nn.Linear(dim, channels) if self.num_refinements == 0 else None))
        self.refinement_stages = nn.ModuleList()
        parent_dim = dim
        for level in range(self.num_refinements):
            parent_grid = output_grid_size * (2 ** level)
            patch_dim = self.input_channels * (h_img // parent_grid) * (w_img // parent_grid)
            is_last = level == self.num_refinements - 1
            out_dim = channels if is_last else make_divisible(min(parent_dim * 2, fmap_max), 8)
            self.refinement_stages.append(
                HierMLPStage(patch_dim, parent_dim, dim, out_dim if is_last else parent_dim, out_dim, layer_count))
            parent_dim = out_dim

    @staticmethod
    def _coords(grid_h, grid_w, device, dtype):
        y = torch.linspace(-1, 1, grid_h, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, grid_w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        return torch.stack((xx, yy), dim=-1).reshape(1, grid_h * grid_w, 2)

    @staticmethod
    def _local_patches(x, grid_size):
        h, w = x.shape[-2:]
        patch_h, patch_w = h // grid_size, w // grid_size
        return F.unfold(x, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w)).transpose(1, 2)

    @staticmethod
    def _expand_children(children, grid_size):
        batch, _, child_count, channels = children.shape
        if child_count != 4:
            raise ValueError("HierMLP refinement stages must emit exactly four children per parent patch")
        return (children.view(batch, grid_size, grid_size, 2, 2, channels)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(batch, grid_size * 2, grid_size * 2, channels))

    def _merge_to_output_grid(self, features):
        """Group fine-grid tokens spatially and project each group to one root token."""
        if isinstance(self.input_to_output, nn.Identity):
            return features
        batch, _, dim = features.shape
        ratio = self.input_to_output_ratio
        grid = self.output_grid_size
        grouped = (features.view(batch, grid, ratio, grid, ratio, dim)
                   .permute(0, 1, 3, 2, 4, 5)
                   .reshape(batch, grid * grid, ratio * ratio * dim))
        return self.input_to_output(grouped)

    def forward(self, x, time, x_self_cond=None, class_labels=None):
        batch = x.shape[0]
        cond = x[:, self.channels:] if self.cond_residual else None
        time_emb = self.time_mlp(time)
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("Class-conditional model requires class labels")
            time_emb = time_emb + self.class_embedding(class_labels)
        skips = None
        if self.stem_skips:
            features, skips = self.coarse_patch_embedding(x, return_skips=True, conditioning=time_emb)
        else:
            features = (self.coarse_patch_embedding(x, conditioning=time_emb) if self.conv_stem & 1
                        else self.coarse_patch_embedding(self._local_patches(x, self.input_grid_size)))
        features = features + self.coarse_pos_embedding
        global_time_emb = self.global_time_proj(time_emb)
        for mixer_block in self.coarse_mixer:
            features = mixer_block(features, t_emb=global_time_emb)
        grid_size = self.output_grid_size
        features = self._merge_to_output_grid(features)
        if self.conv_stem & 2:
            output = self.draft_to_pixels(features, skips=skips, conditioning=time_emb)
            return output + cond if cond is not None else output

        for stage in self.refinement_stages:
            patches = self._local_patches(x, grid_size)
            coords = self._coords(grid_size, grid_size, x.device, x.dtype)
            children = stage(patches, coords, time_emb, features)
            features = self._expand_children(children.view(batch, grid_size * grid_size, 4, -1), grid_size)
            grid_size *= 2
            features = features.reshape(batch, grid_size * grid_size, -1)

        if grid_size == self.output_grid_size:
            # An image already at the initial grid has no refinement stages, so turn its draft directly
            # into pixels while preserving the same local MLP semantics.
            features = self.draft_to_pixels(features)
        output = features.view(batch, grid_size, grid_size, self.channels).permute(0, 3, 1, 2)
        return output + cond if cond is not None else output


# ==========================================
# 6. Flow Matching Wrapper
# ==========================================

class FlowMatchingWrapper(nn.Module):
    """Rectified-flow / flow-matching wrapper for the JiT formulation.

    Implements all nine (prediction-space x loss-space) combinations of Tab. 1
    in "Back to Basics: Let Denoising Generative Models Denoise". The default
    (pred_mode="x", loss_mode="v") is the paper's final algorithm: predict the
    clean image directly and train it under a v-loss (Alg. 1 / Tab. 1(3)(a)).

    Faithfulness notes:
      * The default linear schedule is z_t = t*x + (1-t)*eps.  The optional
        RIN schedule instead uses the paper's sigmoidal signal power.
      * The network's direct output is interpreted as x, eps, or v; the other
        two quantities are recovered analytically (Tab. 1). (1-t) and t are
        clamped by `t_clip` (paper default 0.05) wherever they sit in a
        denominator, so the loss stays 0 at a perfect prediction even near the
        endpoints.
      * Noise magnitude scales with H/256 at higher resolution to roughly hold
        the SNR fixed (paper: eps ~ N(0, (H/256)^2 I) at 512 / 1024).
      * Sampling integrates dz/dt = v -- the paper's "generator space": whatever
        the network predicts is first mapped to a velocity, then stepped.
    """

    def __init__(self, model, pred_mode="x", loss_mode="v",
                 t_loc=-0.8, t_scale=0.8, t_clip=0.05, x_clip="none",
                 noise_schedule="linear", self_cond_prob=None, class_dropout_prob=0.0):
        super().__init__()
        if self_cond_prob is None:
            self_cond_prob = 0.9 if getattr(model, 'supports_latent_self_conditioning', False) else 0.5
        assert pred_mode in ("x", "eps", "v"), f"bad pred_mode {pred_mode!r}"
        assert loss_mode in ("x", "eps", "v"), f"bad loss_mode {loss_mode!r}"
        assert x_clip in ("none", "static", "dynamic"), f"bad x_clip {x_clip!r}"
        assert noise_schedule in ("linear", "rin_sigmoid"), f"bad noise schedule {noise_schedule!r}"
        assert 0.0 <= self_cond_prob <= 1.0, f"bad self-conditioning probability {self_cond_prob!r}"
        self.model = model
        self.pred_mode = pred_mode
        self.loss_mode = loss_mode
        self.loc = t_loc
        self.scale = t_scale
        self.t_clip = t_clip
        self.x_clip = x_clip
        self.noise_schedule = noise_schedule
        self.self_cond_prob = self_cond_prob
        if not 0.0 <= class_dropout_prob <= 1.0:
            raise ValueError("class_dropout_prob must be between 0 and 1")
        if class_dropout_prob > 0 and getattr(model, 'null_class_id', None) is None:
            raise ValueError("Class dropout requires a model built with class_cfg=True")
        self.class_dropout_prob = class_dropout_prob

    def _coefficients(self, t_img):
        """Return signal/noise coefficients and their derivatives w.r.t. progress.

        `rin_sigmoid` is the paper's normalized sigmoid signal-power schedule,
        with start=-3, end=3 and temperature tau=0.9.  Progress runs from
        noise (0) to data (1), the reverse of the paper's noising time.
        """
        if self.noise_schedule == "linear":
            return t_img, 1.0 - t_img, torch.ones_like(t_img), -torch.ones_like(t_img)
        t_img = t_img.float()
        start, end, tau = -3.0, 3.0, 0.9
        v_start = torch.sigmoid(t_img.new_tensor(start / tau))
        v_end = torch.sigmoid(t_img.new_tensor(end / tau))
        w = (end - (end - start) * t_img) / tau
        s_w = torch.sigmoid(w)
        gamma = ((-s_w + v_end) / (v_end - v_start)).clamp(torch.finfo(torch.float32).eps, 1.0 - torch.finfo(torch.float32).eps)
        dgamma = s_w * (1.0 - s_w) * ((end - start) / tau) / (v_end - v_start)
        signal, noise = gamma.sqrt(), (1.0 - gamma).sqrt()
        return signal, noise, 0.5 * dgamma / signal, -0.5 * dgamma / noise

    def _model_time(self, progress, t_img):
        # The official RIN denoiser is conditioned on gamma (signal power),
        # rather than the normalized noising-time coordinate.
        if self.noise_schedule == "rin_sigmoid":
            signal, _, _, _ = self._coefficients(t_img)
            return signal.square().flatten(1).squeeze(-1)
        return progress

    def get_noise_scale(self, h):
        return h / 256.0 if h > 256 else 1.0

    def _convert(self, src, z_t, t_img, clip=None, omt_clip=None):
        """Interpret `src` as the quantity named by self.pred_mode and return the
        triple (x, eps, v) via the relations of Tab. 1, clamping (1-t)/t by
        `clip` (default self.t_clip) wherever they appear in a denominator.
        `omt_clip` overrides the clamp on (1-t) alone (default: `clip`)."""
        if self.noise_schedule == "rin_sigmoid":
            signal, noise, dsignal, dnoise = self._coefficients(t_img)
            if self.pred_mode == "x":
                x = src
                eps = (z_t - signal * x) / noise.clamp(min=1e-5)
                v = dsignal * x + dnoise * eps
            elif self.pred_mode == "eps":
                eps = src
                x = (z_t - noise * eps) / signal.clamp(min=1e-5)
                v = dsignal * x + dnoise * eps
            else:
                v = src
                determinant = signal * dnoise - dsignal * noise
                x = (z_t * dnoise - noise * v) / determinant
                eps = (signal * v - dsignal * z_t) / determinant
            return x, eps, v
        clip = self.t_clip if clip is None else clip
        omt_clip = clip if omt_clip is None else omt_clip
        omt = (1.0 - t_img).clamp(min=omt_clip)   # (1 - t), clamped
        t_s = t_img.clamp(min=clip)           # t, clamped
        if self.pred_mode == "x":
            x = src
            eps = (z_t - t_img * x) / omt
            v = (x - z_t) / omt
        elif self.pred_mode == "eps":
            eps = src
            x = (z_t - (1.0 - t_img) * eps) / t_s
            v = (z_t - eps) / t_s
        else:  # "v"
            v = src
            x = z_t + (1.0 - t_img) * v
            eps = z_t - t_img * v
        return x, eps, v

    def _clip_x(self, x):
        """Optional clipping of the predicted clean image (off by default)."""
        if self.x_clip == "static":
            return x.clamp(-1.0, 1.0)
        if self.x_clip == "dynamic":
            b = x.shape[0]
            s = torch.quantile(x.detach().abs().flatten(1), 0.995, dim=1).clamp(min=1.0)
            s = s.view(b, *([1] * (x.dim() - 1)))
            return x.clamp(-s, s) / s
        return x

    @staticmethod
    def _model_input(z_t, condition):
        """Append a paired Pix2Pix source image without noising it."""
        if condition is None:
            return z_t
        if condition.shape != z_t.shape:
            raise ValueError(
                f"Pix2Pix condition shape {tuple(condition.shape)} must match target shape {tuple(z_t.shape)}"
            )
        return torch.cat((z_t, condition.to(device=z_t.device, dtype=z_t.dtype)), dim=1)

    def _class_labels(self, class_labels, batch_size, device):
        if class_labels is None:
            null_id = getattr(self.model, 'null_class_id', None)
            if null_id is not None:
                return torch.full((batch_size,), null_id, device=device, dtype=torch.long)
            return None
        class_labels = class_labels.to(device=device, dtype=torch.long)
        if class_labels.shape != (batch_size,):
            raise ValueError(f"Class labels must have shape ({batch_size},), got {tuple(class_labels.shape)}")
        return class_labels

    def p_losses(self, x_start, condition=None, class_labels=None):
        b, c, h, w = x_start.shape
        device = x_start.device
        model_dtype = next(self.model.parameters()).dtype
        if x_start.dtype != model_dtype:
            x_start = x_start.to(dtype=model_dtype)
        class_labels = self._class_labels(class_labels, b, device)

        # Drop once per example so preliminary and main self-conditioning
        # passes see the SAME label. The last embedding row is the null class.
        if self.training and self.class_dropout_prob > 0 and class_labels is not None:
            drop = torch.rand(b, device=device) < self.class_dropout_prob
            class_labels = torch.where(drop, self.model.null_class_id, class_labels)

        noise_scale = self.get_noise_scale(h)
        epsilon = torch.randn_like(x_start) * noise_scale

        # Logit-normal time sampling: logit(t) ~ N(loc, scale^2)
        s = torch.randn(b, device=device, dtype=torch.float32) * self.scale + self.loc
        t = torch.sigmoid(s)
        t_img = t.view(b, 1, 1, 1)

        signal, noise, dsignal, dnoise = self._coefficients(t_img)
        z_t = signal * x_start + noise * epsilon
        model_input = self._model_input(z_t, condition)

        # RIN's latent self-conditioning feeds the previous denoising pass's
        # latent slots into the current one, rather than concatenating pixels.
        latent_self_cond = None
        model_time = self._model_time(t, t_img).to(dtype=model_dtype)
        # Self-conditioning coin flips use the CPU generator: a CUDA draw in an
        # `if` would force a host sync on every forward pass.
        if getattr(self.model, "supports_latent_self_conditioning", False):
            # Algorithm 1: sometimes train from zero context, as at the first
            # sampling step. The preliminary pass uses the SAME noisy input,
            # time and dropped labels, with no gradient through its latents.
            if self.self_cond_prob > 0 and torch.rand(()).item() < self.self_cond_prob:
                with torch.no_grad():
                    _, latent_self_cond = self.model(
                        model_input.to(dtype=model_dtype), model_time, return_latents=True, class_labels=class_labels,
                    )
                latent_self_cond = latent_self_cond.detach()
            net_out = self.model(
                model_input.to(dtype=model_dtype), model_time, latent_self_cond=latent_self_cond, class_labels=class_labels,
            )
        else:
            # Self-conditioning uses a detached preliminary clean-image estimate
            # on a fraction of training examples. The second pass receives that
            # estimate through JiTModel's x_self_cond input without extending the
            # backward graph through the preliminary prediction.
            x_self_cond = None
            if getattr(self.model, "self_cond", False) and torch.rand(()).item() < self.self_cond_prob:
                with torch.no_grad():
                    preliminary_out = self.model(model_input.to(dtype=model_dtype), model_time, class_labels=class_labels)
                    x_self_cond, _, _ = self._convert(preliminary_out, z_t, t_img)
                x_self_cond = x_self_cond.detach().to(dtype=model_dtype)
            net_out = self.model(
                model_input.to(dtype=model_dtype), model_time, x_self_cond=x_self_cond, class_labels=class_labels,
            )

        # Derive the predicted triple and the ground-truth triple with the SAME
        # clamped transforms. Sharing the transform guarantees the loss is 0 at a
        # perfect prediction in every space, including inside the clamped region
        # (the source of the original v-target mismatch near t -> 1).
        true_velocity = dsignal * x_start + dnoise * epsilon
        true_src = {"x": x_start, "eps": epsilon,
                    "v": true_velocity if self.noise_schedule == "rin_sigmoid" else x_start - epsilon}[self.pred_mode]
        px, pe, pv = self._convert(net_out.float(), z_t, t_img)
        tx, te, tv = self._convert(true_src, z_t, t_img)
        pred, target = {"x": (px, tx), "eps": (pe, te), "v": (pv, tv)}[self.loss_mode]

        # Keep the reduction in FP32 so full-BF16 training does not accumulate
        # its loss in BF16 precision.
        return F.mse_loss(pred.float(), target.float())

    @torch.no_grad()
    def sample(self, shape, steps=50, solver="heun", condition=None, class_labels=None,
               guidance_scale=1.0, initial_noise=None):
        """Sample with optional CFG: unconditional + scale * (conditional - unconditional).

        Scale 1 preserves ordinary conditional sampling and its single forward
        pass. Guidance requires training with a null class (class dropout).
        Conditional/unconditional self-conditioning histories stay separate.
        """
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")
        apply_guidance = class_labels is not None and guidance_scale != 1.0
        if apply_guidance and getattr(self.model, 'null_class_id', None) is None:
            raise ValueError("This checkpoint has no null class; use guidance_scale=1")
        b, c, h, w = shape
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        class_labels = self._class_labels(class_labels, b, device)
        if condition is not None:
            condition = condition.to(device=device, dtype=dtype)
            if tuple(condition.shape) != tuple(shape):
                raise ValueError(
                    f"Pix2Pix condition shape {tuple(condition.shape)} must match sample shape {tuple(shape)}"
                )

        if steps < 1 or solver not in {"heun", "euler"}:
            raise ValueError("Sampling requires positive steps and an euler/heun solver")
        if initial_noise is None:
            z = torch.randn(shape, device=device, dtype=torch.float32) * self.get_noise_scale(h)
        else:
            if tuple(initial_noise.shape) != tuple(shape):
                raise ValueError('Initial noise must match the requested sample shape')
            z = initial_noise.to(device=device, dtype=torch.float32).clone() * self.get_noise_scale(h)
        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float32)

        cond_state = (None, None)
        uncond_state = (None, None)
        null_labels = self._class_labels(None, b, device) if apply_guidance else None

        def velocity(z_in, t_scalar, latent_prev, x_self_cond_prev, labels):
            t_vec = torch.full((b,), t_scalar, device=device, dtype=torch.float32)
            t_img = t_vec.view(b, 1, 1, 1)
            model_time = self._model_time(t_vec, t_img).to(dtype=dtype)
            model_input = self._model_input(z_in, condition)
            if getattr(self.model, "supports_latent_self_conditioning", False):
                net_out, latents = self.model(
                    model_input.to(dtype=dtype), model_time, latent_self_cond=latent_prev,
                    return_latents=True, class_labels=labels,
                )
            else:
                net_out = self.model(
                    model_input.to(dtype=dtype), model_time,
                    x_self_cond=None if x_self_cond_prev is None else x_self_cond_prev.to(dtype=dtype),
                    class_labels=labels,
                )
                latents = None
            # Use a tiny (1-t) clamp at inference so the final steps fully
            # denoise: 1-t >= 1/steps at every evaluated time, so it never
            # binds. A 0.05 clamp there left residual noise above 20 steps.
            # The t clamp stays at 0.05 for eps-prediction at t=0.
            x_pred, _, v_pred = self._convert(net_out.float(), z_in, t_img, clip=0.05, omt_clip=1e-5)
            return v_pred, latents, x_pred

        def guided_velocity(z_in, t_scalar, cond_prev, uncond_prev):
            v_pred, latents, x_pred = velocity(z_in, t_scalar, *cond_prev, class_labels)
            next_cond = (latents, self._clip_x(x_pred) if apply_guidance else x_pred)
            next_uncond = (None, None)
            if apply_guidance:
                u_v, u_latents, u_x = velocity(z_in, t_scalar, *uncond_prev, null_labels)
                next_uncond = (u_latents, self._clip_x(u_x))
                v_pred = u_v + guidance_scale * (v_pred - u_v)
                x_pred = u_x + guidance_scale * (x_pred - u_x)
            # Apply optional clipping AFTER combining the predictions, in the
            # same clean-image space regardless of x/eps/v prediction mode.
            if self.x_clip != "none":
                t_img = torch.full((b, 1, 1, 1), t_scalar, device=device, dtype=torch.float32)
                x_pred = self._clip_x(x_pred)
                if self.noise_schedule == "rin_sigmoid":
                    signal, noise, dsignal, dnoise = self._coefficients(t_img)
                    eps_pred = (z_in - signal * x_pred) / noise.clamp(min=1e-5)
                    v_pred = dsignal * x_pred + dnoise * eps_pred
                else:
                    v_pred = (x_pred - z_in) / (1.0 - t_img).clamp(min=1e-5)
                if not apply_guidance:
                    next_cond = (latents, x_pred)
            return v_pred, next_cond, next_uncond

        # Heun (default) or Euler integration of dz/dt = v from t=0 to t=1.
        for i in range(steps):
            t0 = timesteps[i].item()
            t1 = timesteps[i + 1].item()
            dt = t1 - t0
            d0, next_cond, next_uncond = guided_velocity(z, t0, cond_state, uncond_state)
            if solver == "euler" or i == steps - 1:
                z = z + dt * d0
            else:
                d1, next_cond, next_uncond = guided_velocity(
                    z + dt * d0, t1, next_cond, next_uncond)
                z = z + dt * 0.5 * (d0 + d1)
            cond_state, uncond_state = next_cond, next_uncond

        return z


# ==========================================
# 7. Data Pipeline
# ==========================================

IMAGE_EXTENSIONS = ('jpg', 'jpeg', 'png', 'webp', 'bmp')


def find_image_paths(folder, ext=IMAGE_EXTENSIONS):
    """Return unique image paths recursively, in a stable order."""
    extensions = {'.' + extension.lstrip('.').casefold() for extension in ext}
    paths = [path for path in glob.glob(os.path.join(folder, '**', '*'), recursive=True)
             if os.path.splitext(path)[1].casefold() in extensions and os.path.isfile(path)]
    return sorted(set(paths))


def discover_class_names(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(
        entry.name for entry in os.scandir(folder)
        if entry.is_dir() and find_image_paths(entry.path)
    )


def parse_size_range(value, label='size'):
    """Parse `N`, `N-M`, `N,M`, or `N M` into an inclusive integer range."""
    if isinstance(value, int):
        values = [value]
    elif isinstance(value, (tuple, list)):
        values = list(value)
    else:
        if str(value).strip().startswith('-'):
            raise ValueError(f'{label} must be at least 1; native -1 applies only to source resize dimensions')
        values = str(value).replace('-', ' ').replace(',', ' ').split()
    if len(values) not in {1, 2}:
        raise ValueError(f"{label} must be an integer or min-max range")
    try:
        values = [int(item) for item in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer or min-max range") from exc
    if any(item < 1 for item in values):
        raise ValueError(f"{label} must be at least 1")
    return (values[0], values[0]) if len(values) == 1 else tuple(sorted(values))


class CropAwareTransform:
    """Native-or-resized canvas, aspect-preserving variable crop, fixed output size."""
    def __init__(self, resize_size, crop_size, output_size=None, augment=False,
                 vertical_flip=False, rotate90=False):
        self.resize_size = tuple(resize_size)  # H/W; -1 retains that native dimension.
        self.crop_h = parse_size_range(crop_size[0], 'crop height')
        self.crop_w = parse_size_range(crop_size[1], 'crop width')
        if self.crop_w[0] * self.crop_h[1] != self.crop_w[1] * self.crop_h[0]:
            raise ValueError('Crop width and height ranges must preserve one aspect ratio')
        unit = math.gcd(self.crop_h[0], self.crop_w[0])
        self._crop_unit = (self.crop_h[0] // unit, self.crop_w[0] // unit)
        self._crop_scale_range = (unit, self.crop_h[1] // self._crop_unit[0])
        self.output_size = tuple(output_size or (self.crop_h[1], self.crop_w[1]))
        self.augment = augment
        self.vertical_flip = vertical_flip
        self.rotate90 = rotate90 and self.output_size[0] == self.output_size[1]
        self.resample = getattr(Image.Resampling, 'LANCZOS', Image.LANCZOS)

    def canvas_size(self, image):
        height = image.height if self.resize_size[0] == -1 else self.resize_size[0]
        width = image.width if self.resize_size[1] == -1 else self.resize_size[1]
        if height < 1 or width < 1:
            raise ValueError('Image resize dimensions must be positive or -1')
        # Native images smaller than the requested crop are uniformly enlarged,
        # avoiding padded pixels and preserving their aspect ratio.
        scale = max(1.0, self.crop_h[1] / height, self.crop_w[1] / width)
        return max(self.crop_h[1], round(height * scale)), max(self.crop_w[1], round(width * scale))

    def _crop_box(self, canvas_size):
        canvas_h, canvas_w = canvas_size
        scale = random.randint(*self._crop_scale_range)
        crop_h = self._crop_unit[0] * scale
        crop_w = self._crop_unit[1] * scale
        left = random.randint(0, canvas_w - crop_w) if canvas_w > crop_w else 0
        top = random.randint(0, canvas_h - crop_h) if canvas_h > crop_h else 0
        return left, top, left + crop_w, top + crop_h

    def _spatial_state(self):
        return {
            'hflip': self.augment and random.random() < 0.5,
            'vflip': self.vertical_flip and random.random() < 0.5,
            'rotations': random.randrange(4) if self.rotate90 else 0,
        }

    def prepare(self, image, crop_box=None, spatial_state=None, canvas_size=None):
        canvas_size = self.canvas_size(image) if canvas_size is None else canvas_size
        if image.size != (canvas_size[1], canvas_size[0]):
            image = image.resize((canvas_size[1], canvas_size[0]), self.resample)
        crop_box = self._crop_box(canvas_size) if crop_box is None else crop_box
        image = image.crop(crop_box)
        if image.size != (self.output_size[1], self.output_size[0]):
            image = image.resize((self.output_size[1], self.output_size[0]), self.resample)
        spatial_state = self._spatial_state() if spatial_state is None else spatial_state
        if spatial_state['hflip']:
            image = image.transpose(getattr(Image.Transpose, 'FLIP_LEFT_RIGHT', Image.FLIP_LEFT_RIGHT))
        if spatial_state['vflip']:
            image = image.transpose(getattr(Image.Transpose, 'FLIP_TOP_BOTTOM', Image.FLIP_TOP_BOTTOM))
        for _ in range(spatial_state['rotations']):
            image = image.transpose(getattr(Image.Transpose, 'ROTATE_90', Image.ROTATE_90))
        return image, crop_box, spatial_state

    @staticmethod
    def tensor(image):
        return T.ToTensor()(image) * 2 - 1


class ImageDataset(Dataset):
    def __init__(self, folder, image_size, crop_size=None, output_size=None, channels=3, augment=False,
                 vertical_flip=False, rotate90=False,
                 ext=IMAGE_EXTENSIONS):
        super().__init__()
        self.channels = channels
        self.paths = find_image_paths(folder, ext)
        print(f"│  Dataset images: {len(self.paths):,}")

        self.transform = CropAwareTransform(image_size, crop_size or image_size, output_size, augment, vertical_flip, rotate90)
        self._mode_map = {3: 'RGB', 1: 'L', 4: 'RGBA'}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Try up to 50 times to load a valid image to prevent infinite hanging
        # in the rare case that the entire directory is corrupted.
        for _ in range(50):
            try:
                img = Image.open(self.paths[index]).convert(self._mode_map.get(self.channels, 'RGB'))
                image, _, _ = self.transform.prepare(img)
                return self.transform.tensor(image)
            except Exception:
                # Skip this image and randomly select another one
                index = random.randint(0, len(self.paths) - 1)
        
        # If it fails 50 times in a row, surface the error
        raise RuntimeError("Dataset seems heavily corrupted; failed to load 50 consecutive random images.")


class ClassImageDataset(Dataset):
    """Image-folder classification layout: root/class_name/image.ext."""
    def __init__(self, folder, image_size, crop_size=None, output_size=None, channels=3, augment=False,
                 vertical_flip=False, rotate90=False, class_names=None):
        super().__init__()
        self.channels = channels
        self.class_names = list(class_names) if class_names is not None else discover_class_names(folder)
        self.samples = []
        for class_index, class_name in enumerate(self.class_names):
            for path in find_image_paths(os.path.join(folder, class_name)):
                self.samples.append((path, class_index))
        self.samples.sort()
        print(f"│  Class-conditional images: {len(self.samples):,} across {len(self.class_names):,} classes")
        self.transform = CropAwareTransform(image_size, crop_size or image_size, output_size, augment, vertical_flip, rotate90)
        self._mode = {3: 'RGB', 1: 'L', 4: 'RGBA'}.get(channels, 'RGB')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        selected_class = self.samples[index][1]
        for _ in range(50):
            path, class_index = self.samples[index]
            try:
                with Image.open(path) as img:
                    image, _, _ = self.transform.prepare(img.convert(self._mode))
                    return self.transform.tensor(image), class_index
            except Exception:
                index = random.choice([i for i, (_, label) in enumerate(self.samples)
                                       if label == selected_class])
        raise RuntimeError("Class dataset seems heavily corrupted; failed to load 50 consecutive images.")


class PairedImageDataset(Dataset):
    """Paired Pix2Pix layout: root/A/relative/name.ext ↔ root/B/relative/name.ext."""
    def __init__(self, folder, image_size, crop_size=None, output_size=None, channels=3, augment=False,
                 vertical_flip=False, rotate90=False, direction="a_to_b"):
        super().__init__()
        if direction not in {'a_to_b', 'b_to_a'}:
            raise ValueError(f"Unknown Pix2Pix direction {direction!r}")
        self.a_root = os.path.join(folder, 'A')
        self.b_root = os.path.join(folder, 'B')
        self.direction = direction
        self.channels = channels
        self.transform = CropAwareTransform(image_size, crop_size or image_size, output_size, augment, vertical_flip, rotate90)
        self._mode = {3: 'RGB', 1: 'L', 4: 'RGBA'}.get(channels, 'RGB')

        def keyed_paths(root):
            result = {}
            duplicates = 0
            for path in find_image_paths(root):
                relative = os.path.splitext(os.path.relpath(path, root))[0].replace(os.sep, '/').casefold()
                if relative in result:
                    duplicates += 1
                    continue
                result[relative] = path
            return result, duplicates

        a_paths, a_duplicates = keyed_paths(self.a_root)
        b_paths, b_duplicates = keyed_paths(self.b_root)
        keys = sorted(set(a_paths).intersection(b_paths))
        self.pairs = [(a_paths[key], b_paths[key]) for key in keys]
        discarded = (len(a_paths) - len(keys) + a_duplicates + len(b_paths) - len(keys) + b_duplicates)
        print(f"│  Pix2Pix pairs: {len(self.pairs):,}")
        if discarded:
            print(f"│  Pix2Pix discarded unmatched images: {discarded:,}")

    def __len__(self):
        return len(self.pairs)

    def _transform_pair(self, image_a, image_b):
        canvas_size = self.transform.canvas_size(image_a)
        image_a, crop_box, spatial_state = self.transform.prepare(image_a, canvas_size=canvas_size)
        image_b, _, _ = self.transform.prepare(image_b, crop_box=crop_box,
                                                spatial_state=spatial_state, canvas_size=canvas_size)
        return self.transform.tensor(image_a), self.transform.tensor(image_b)

    def __getitem__(self, index):
        for _ in range(50):
            a_path, b_path = self.pairs[index]
            try:
                with Image.open(a_path) as image_a, Image.open(b_path) as image_b:
                    a, b = self._transform_pair(image_a.convert(self._mode), image_b.convert(self._mode))
                return (b, a) if self.direction == 'a_to_b' else (a, b)
            except Exception:
                index = random.randrange(len(self.pairs))
        raise RuntimeError("Pix2Pix dataset seems heavily corrupted; failed to load 50 consecutive pairs.")


class CombinedPairedImageDataset(Dataset):
    """Pix2Pix pairs stored as one image: A is the left half, B the right half."""
    def __init__(self, folder, image_size, crop_size=None, output_size=None, channels=3, augment=False,
                 vertical_flip=False, rotate90=False, direction="a_to_b"):
        if direction not in {'a_to_b', 'b_to_a'}:
            raise ValueError(f"Unknown Pix2Pix direction {direction!r}")
        self.paths = find_image_paths(folder)
        self.direction = direction
        self.transform = CropAwareTransform(image_size, crop_size or image_size, output_size, augment, vertical_flip, rotate90)
        self._mode = {3: 'RGB', 1: 'L', 4: 'RGBA'}.get(channels, 'RGB')
        print(f"│  Combined Pix2Pix images: {len(self.paths):,}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        for _ in range(50):
            try:
                with Image.open(self.paths[index]) as image:
                    image = image.convert(self._mode)
                    midpoint = image.width // 2
                    if midpoint == 0:
                        raise ValueError('combined pair is narrower than two pixels')
                    image_a = image.crop((0, 0, midpoint, image.height))
                    image_b = image.crop((midpoint, 0, image.width, image.height))
                    a, b = PairedImageDataset._transform_pair(self, image_a, image_b)
                return (b, a) if self.direction == 'a_to_b' else (a, b)
            except Exception:
                index = random.randrange(len(self.paths))
        raise RuntimeError("Combined Pix2Pix dataset seems heavily corrupted; failed to load 50 consecutive pairs.")


class SyntheticPix2PixDataset(Dataset):
    """Generate an A-side degradation from each clean source image on demand."""
    MODES = {'colorify', 'superresolution', 'deblur', 'restoration', 'inpaint', 'outpaint',
             'affine', 'noise', 'hue', 'grayscale', 'sharpen', 'quantize', 'glitch',
             'colorize', 'film', 'invert', 'solarize', 'autocanny'}
    EFFECT_MODES = MODES - {'restoration', 'autocanny'}
    DEFAULT_PARAMS = {
        'superres_min_size': 4, 'superres_max_size': 32,
        'blur_min_radius': .4, 'blur_max_radius': 3.0,
        'noise_min_sigma': 6., 'noise_max_sigma': 38.,
        'affine_shear': .22, 'affine_translate_fraction': .08,
        'hue_max_shift': 80,
        'sharpen_min_radius': 1., 'sharpen_max_radius': 4.,
        'sharpen_min_percent': 180, 'sharpen_max_percent': 450,
        'quantize_min_colors': 2, 'quantize_max_colors': 256,
        'glitch_min_bands': 2, 'glitch_max_bands': 10,
        'glitch_max_band_fraction': 1 / 16, 'glitch_max_offset_fraction': 1 / 8,
        'film_min_marks': 3, 'film_max_marks': 14,
        'solarize_min_threshold': 48, 'solarize_max_threshold': 208,
        'cutout_min_fraction': 1 / 8, 'cutout_max_fraction': 3 / 4,
        'restoration_probability': .26,
        'autocanny_sigma': .33,
    }

    def __init__(self, folder, image_size, crop_size=None, output_size=None, channels=3, augment=False,
                 vertical_flip=False, rotate90=False, synthetic_mode='colorify', synthetic_params=None):
        self.synthetic_modes = self.parse_modes(synthetic_mode)
        if channels != 3:
            raise ValueError('Synthetic Pix2Pix modes require 3 RGB channels')
        self.paths = find_image_paths(folder)
        self.params = dict(self.DEFAULT_PARAMS)
        self.params.update(synthetic_params or {})
        if 'autocanny' in self.synthetic_modes:
            sigma = self.params['autocanny_sigma']
            if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or not .01 <= sigma <= 1.:
                raise ValueError('autocanny_sigma must be a number between 0.01 and 1.0')
            # Check the optional converter dependency before the image retry loop.
            from make_canny_lineart_pairs import auto_canny_lineart
        self.transform = CropAwareTransform(image_size, crop_size or image_size, output_size,
                                            augment, vertical_flip, rotate90)
        self._mode = {3: 'RGB', 1: 'L', 4: 'RGBA'}.get(channels, 'RGB')
        print(f"│  Synthetic Pix2Pix images ({', '.join(self.synthetic_modes)}): {len(self.paths):,}")

    @classmethod
    def parse_modes(cls, value):
        modes = [item.strip().lower() for item in (value.split(',') if isinstance(value, str) else value) if item.strip()]
        if not modes or any(mode not in cls.MODES for mode in modes):
            raise ValueError(f"Unknown synthetic Pix2Pix mode(s) {value!r}")
        return list(dict.fromkeys(modes))

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def _large_enough(image, image_size):
        """Require native dimensions at least as large as the pre-crop target (H, W)."""
        return image.width >= image_size[1] and image.height >= image_size[0]

    def _resolution_drop(self, image):
        width, height = image.size
        max_w = min(width, int(self.params['superres_max_size']))
        max_h = min(height, int(self.params['superres_max_size']))
        low_w = random.randint(min(int(self.params['superres_min_size']), max_w), max_w)
        low_h = random.randint(min(int(self.params['superres_min_size']), max_h), max_h)
        downsample = getattr(Image.Resampling, 'LANCZOS', Image.LANCZOS)
        upsample = getattr(Image.Resampling, 'BICUBIC', Image.BICUBIC)
        return image.resize((low_w, low_h), downsample).resize((width, height), upsample)

    def _add_noise(self, image):
        array = np.asarray(image.convert('RGB')).astype(np.int16)
        array += np.random.normal(0, random.uniform(self.params['noise_min_sigma'], self.params['noise_max_sigma']), array.shape).astype(np.int16)
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), 'RGB')

    def _hue_shift(self, image):
        hsv = np.asarray(image.convert('HSV')).copy()
        shift = int(self.params['hue_max_shift'])
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + random.randint(-shift, shift)) % 256
        return Image.fromarray(hsv, 'HSV').convert('RGB')

    def _glitch(self, image):
        image = image.copy()
        width, height = image.size
        for _ in range(random.randint(int(self.params['glitch_min_bands']), int(self.params['glitch_max_bands']))):
            top = random.randrange(height)
            band_h = random.randint(1, max(1, round(height * self.params['glitch_max_band_fraction'])))
            offset = random.randint(-max(1, round(width * self.params['glitch_max_offset_fraction'])), max(1, round(width * self.params['glitch_max_offset_fraction'])))
            band = image.crop((0, top, width, min(height, top + band_h)))
            image.paste(band, (offset, top))
        return image

    def _film_artifacts(self, image):
        image = image.copy()
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for _ in range(random.randint(int(self.params['film_min_marks']), int(self.params['film_max_marks']))):
            x, y = random.randrange(width), random.randrange(height)
            radius = random.randint(1, max(1, min(width, height) // 40))
            color = random.choice([(40, 30, 20), (240, 225, 190), (120, 90, 60)])
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        for _ in range(random.randint(1, 4)):
            points = [(random.randrange(width), random.randrange(height))]
            for __ in range(random.randint(4, 12)):
                px, py = points[-1]
                points.append((max(0, min(width - 1, px + random.randint(-8, 8))),
                               max(0, min(height - 1, py + random.randint(-8, 8)))))
            draw.line(points, fill=(50, 35, 25), width=1)
        return image

    def _cutout(self, image, keep_only=False):
        image = image.copy()
        width, height = image.size
        side = random.randint(max(4, round(min(width, height) * self.params['cutout_min_fraction'])),
                              max(4, round(min(width, height) * self.params['cutout_max_fraction'])))
        side = min(side, width, height)
        left = random.randint(0, width - side)
        top = random.randint(0, height - side)
        fill = tuple(random.randint(0, 255) for _ in range(3))
        if keep_only:
            canvas = Image.new('RGB', image.size, fill)
            canvas.paste(image.crop((left, top, left + side, top + side)), (left, top))
            return canvas
        ImageDraw.Draw(image).rectangle((left, top, left + side, top + side), fill=fill)
        return image

    def _apply_mode(self, image, mode):
        if mode == 'autocanny':
            from make_canny_lineart_pairs import auto_canny_lineart
            return auto_canny_lineart(image, self.params['autocanny_sigma'])
        if mode == 'colorify' or mode == 'grayscale': return image.convert('L').convert('RGB')
        if mode == 'superresolution': return self._resolution_drop(image)
        if mode == 'deblur': return image.filter(ImageFilter.GaussianBlur(random.uniform(self.params['blur_min_radius'], self.params['blur_max_radius'])))
        if mode == 'noise': return self._add_noise(image)
        if mode == 'hue': return self._hue_shift(image)
        if mode == 'affine':
            shear, translate = self.params['affine_shear'], self.params['affine_translate_fraction']
            return image.transform(image.size, Image.Transform.AFFINE,
                (1, random.uniform(-shear, shear), random.uniform(-translate, translate) * image.width,
                 random.uniform(-shear, shear), 1, random.uniform(-translate, translate) * image.height),
                resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))
        if mode == 'sharpen': return image.filter(ImageFilter.UnsharpMask(radius=random.uniform(self.params['sharpen_min_radius'], self.params['sharpen_max_radius']), percent=random.randint(int(self.params['sharpen_min_percent']), int(self.params['sharpen_max_percent'])), threshold=1))
        if mode == 'quantize': return image.quantize(colors=random.randint(int(self.params['quantize_min_colors']), int(self.params['quantize_max_colors']))).convert('RGB')
        if mode == 'glitch': return self._glitch(image)
        if mode == 'colorize': return ImageOps.colorize(image.convert('L'), (55, 30, 12), (245, 224, 175))
        if mode == 'film': return self._film_artifacts(image)
        if mode == 'invert': return ImageOps.invert(image.convert('RGB'))
        if mode == 'solarize': return ImageOps.solarize(image.convert('RGB'), threshold=random.randint(int(self.params['solarize_min_threshold']), int(self.params['solarize_max_threshold'])))
        if mode == 'inpaint': return self._cutout(image)
        if mode == 'outpaint': return self._cutout(image, keep_only=True)
        raise ValueError(f"Unknown synthetic effect {mode!r}")

    def _restore_corruption(self, image):
        modes = sorted(self.EFFECT_MODES)
        selected = [mode for mode in modes if random.random() < self.params['restoration_probability']]
        for mode in selected or [random.choice(modes)]:
            image = self._apply_mode(image, mode)
        return image

    def _condition(self, clean):
        for mode in self.synthetic_modes:
            clean = self._restore_corruption(clean) if mode == 'restoration' else self._apply_mode(clean, mode)
        return clean

    def __getitem__(self, index):
        for _ in range(50):
            try:
                with Image.open(self.paths[index]) as image:
                    clean, _, _ = self.transform.prepare(image.convert(self._mode))
                # Synthetic degradations currently operate in RGB; retain the
                # clean target's requested channel mode when returning tensors.
                if clean.mode != 'RGB':
                    clean = clean.convert('RGB')
                condition = self._condition(clean.copy())
                return self.transform.tensor(clean), self.transform.tensor(condition)
            except Exception:
                index = random.randrange(len(self.paths))
        raise RuntimeError("Synthetic Pix2Pix dataset seems heavily corrupted; failed to load 50 consecutive images.")


SYNTHETIC_SOURCE_IDS = {
    '2': 'colorify', '3': 'superresolution', '4': 'deblur', '5': 'restoration',
    '6': 'inpaint', '7': 'outpaint', '8': 'affine', '9': 'noise', '10': 'hue',
    '11': 'grayscale', '12': 'sharpen', '13': 'quantize', '14': 'glitch',
    '15': 'colorize', '16': 'film', '17': 'invert', '18': 'solarize', '19': 'autocanny',
}


def parse_synthetic_source(value):
    """Resolve one or comma-separated synthetic source IDs/names."""
    items = [item.strip().lower() for item in str(value).split(',') if item.strip()]
    if len(items) == 1 and items[0] in {'0', 'folders'}:
        return 'folders'
    if len(items) == 1 and items[0] in {'1', 'combined'}:
        return 'combined'
    aliases = {'superres': 'superresolution', 'gray': 'grayscale', 'restore': 'restoration'}
    modes = [aliases.get(SYNTHETIC_SOURCE_IDS.get(item, item), SYNTHETIC_SOURCE_IDS.get(item, item)) for item in items]
    return ','.join(SyntheticPix2PixDataset.parse_modes(modes))


def prompt_synthetic_params(modes):
    """Ask only for controls used by selected effects (restoration enables all)."""
    active = set(modes.split(','))
    if 'restoration' in active:
        active |= SyntheticPix2PixDataset.EFFECT_MODES
    p = {}
    ask = lambda key, label, default, cast: p.__setitem__(key, get_input(label, default, cast))
    if 'autocanny' in active:
        ask('autocanny_sigma', 'Auto-Canny sensitivity sigma (0.01-1.0)', .33, float)
    if 'superresolution' in active:
        ask('superres_min_size', 'Superresolution minimum low-res side', 4, int)
        ask('superres_max_size', 'Superresolution maximum low-res side', 32, int)
    if 'deblur' in active:
        ask('blur_min_radius', 'Blur minimum radius', .4, float); ask('blur_max_radius', 'Blur maximum radius', 3., float)
    if 'noise' in active:
        ask('noise_min_sigma', 'Noise minimum sigma', 6., float); ask('noise_max_sigma', 'Noise maximum sigma', 38., float)
    if 'affine' in active:
        ask('affine_shear', 'Affine maximum shear', .22, float); ask('affine_translate_fraction', 'Affine maximum translation fraction', .08, float)
    if 'hue' in active:
        ask('hue_max_shift', 'Hue maximum shift (0-255)', 80, int)
    if 'sharpen' in active:
        ask('sharpen_min_radius', 'Sharpen minimum radius', 1., float); ask('sharpen_max_radius', 'Sharpen maximum radius', 4., float)
        ask('sharpen_min_percent', 'Sharpen minimum percent', 180, int); ask('sharpen_max_percent', 'Sharpen maximum percent', 450, int)
    if 'quantize' in active:
        ask('quantize_min_colors', 'Quantize minimum colors', 2, int); ask('quantize_max_colors', 'Quantize maximum colors', 256, int)
    if 'glitch' in active:
        ask('glitch_min_bands', 'Glitch minimum bands', 2, int); ask('glitch_max_bands', 'Glitch maximum bands', 10, int)
        ask('glitch_max_band_fraction', 'Glitch maximum band-height fraction', 1 / 16, float)
        ask('glitch_max_offset_fraction', 'Glitch maximum horizontal-offset fraction', 1 / 8, float)
    if 'film' in active:
        ask('film_min_marks', 'Film minimum marks', 3, int); ask('film_max_marks', 'Film maximum marks', 14, int)
    if 'solarize' in active:
        ask('solarize_min_threshold', 'Solarize minimum threshold', 48, int); ask('solarize_max_threshold', 'Solarize maximum threshold', 208, int)
    if active & {'inpaint', 'outpaint'}:
        ask('cutout_min_fraction', 'Cutout minimum side fraction', 1 / 8, float); ask('cutout_max_fraction', 'Cutout maximum side fraction', 3 / 4, float)
    if 'restoration' in active:
        ask('restoration_probability', 'Per-effect restoration probability', .26, float)
    return p


def build_dataset(cfg):
    mode = cfg.get('conditioning_mode', 'unconditional')
    common = dict(
        image_size=(cfg.get('resize_height', cfg['height']), cfg.get('resize_width', cfg['width'])),
        crop_size=((cfg.get('crop_height_min', cfg['height']), cfg.get('crop_height_max', cfg['height'])),
                   (cfg.get('crop_width_min', cfg['width']), cfg.get('crop_width_max', cfg['width']))),
        output_size=(cfg['height'], cfg['width']), channels=cfg['channels'], augment=cfg['use_flip'],
        vertical_flip=cfg.get('use_vflip', False), rotate90=cfg.get('use_rot90', False),
    )
    if mode == 'class':
        return ClassImageDataset(cfg['dataset_path'], class_names=cfg['class_names'], **common)
    if mode == 'pix2pix':
        source_mode = cfg.get('pix2pix_source_mode', 'folders')
        if source_mode == 'combined':
            return CombinedPairedImageDataset(cfg['dataset_path'], direction=cfg['pix2pix_direction'], **common)
        if all(item in SyntheticPix2PixDataset.MODES for item in source_mode.split(',')):
            return SyntheticPix2PixDataset(cfg['dataset_path'], synthetic_mode=source_mode,
                                           synthetic_params=cfg.get('synthetic_params'), **common)
        return PairedImageDataset(cfg['dataset_path'], direction=cfg['pix2pix_direction'], **common)
    return ImageDataset(cfg['dataset_path'], **common)


def dataset_items(dataset):
    """Return the physical example list, including labels or both sides of pairs."""
    for key in ('pairs', 'samples', 'paths'):
        if hasattr(dataset, key):
            return key, getattr(dataset, key)
    raise ValueError('Dataset has no supported example list')


def dataset_subset(dataset, indices):
    # A torch Subset would let the dataset's corrupt-image fallback escape the split.
    result = copy.deepcopy(dataset)
    key, items = dataset_items(dataset)
    setattr(result, key, [items[i] for i in indices])
    return result


def dataset_files(dataset):
    _, items = dataset_items(dataset)
    return [os.path.realpath(p) for item in items
            for p in (item if isinstance(item, (tuple, list)) else [item]) if isinstance(p, str)]


def build_training_datasets(cfg):
    """Create physically separate splits; preserve paired examples and class IDs."""
    train = build_dataset(cfg)
    val = None
    if cfg['validation_path']:
        val_cfg = dict(cfg, dataset_path=cfg['validation_path'])
        if cfg['conditioning_mode'] == 'class':
            unknown = set(discover_class_names(cfg['validation_path'])) - set(cfg['class_names'])
            if unknown:
                raise ConfigError(f'Unknown validation classes: {sorted(unknown)}')
        val = build_dataset(val_cfg)
        if set(dataset_files(train)) & set(dataset_files(val)):
            raise ConfigError('Training and validation folders contain overlapping files')
    elif cfg['validation_percent'] > 0:
        _, items = dataset_items(train)
        rng = random.Random(cfg['validation_seed'])
        groups = {}
        for i, item in enumerate(items):
            label = item[1] if cfg['conditioning_mode'] == 'class' else None
            groups.setdefault(label, []).append(i)
        held_out = []
        for indices in groups.values():
            rng.shuffle(indices)
            count = min(len(indices) - 1, max(1, round(len(indices) * cfg['validation_percent'] / 100)))
            held_out.extend(indices[:count])
        if not held_out:
            raise ConfigError('Not enough examples for validation (need at least two per splittable class)')
        selected = set(held_out)
        val = dataset_subset(train, sorted(selected))
        train = dataset_subset(train, [i for i in range(len(items)) if i not in selected])
    if not len(train) or (val is not None and not len(val)):
        raise ConfigError('Training and enabled validation datasets must be nonempty')
    if val is not None:
        # Spatial randomness and synthetic corruptions are fixed per example below.
        val.transform.augment = False
        val.transform.vertical_flip = False
        val.transform.rotate90 = False
    return train, val


class SeededDataset(Dataset):
    """Make augmentation independent of worker scheduling, prefetch, and model RNG."""
    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        epoch, index = key[:2] if isinstance(key, tuple) else (0, key)
        occurrence = key[2] if isinstance(key, tuple) and len(key) == 3 else 0
        state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        seed = (self.seed + epoch * 1000000007 + index * 1000003
                + occurrence * 1000000000039) % (2 ** 63 - 1)
        try:
            random.seed(seed)
            np.random.seed(seed % 2 ** 32)
            # Do not initialize CUDA in forked data workers.
            torch.random.default_generator.manual_seed(seed)
            return self.dataset[index]
        finally:
            random.setstate(state[0])
            np.random.set_state(state[1])
            torch.set_rng_state(state[2])


class StreamBatchSampler:
    def __init__(self, stream):
        self.stream = stream

    def __iter__(self):
        s = self.stream
        epoch, position = s.epoch, s.position
        if s.class_groups:
            rng = random.Random((s.seed + epoch) % (2 ** 63 - 1))
            pools = {label: [] for label in s.class_groups}
            keys = []
            # Every complete round visits each class once. Each class cycles
            # through shuffled images without replacement before refilling.
            for _ in range(s.epoch_size // len(s.class_groups)):
                labels = list(s.class_groups)
                rng.shuffle(labels)
                for label in labels:
                    if not pools[label]:
                        pools[label] = list(s.class_groups[label])
                        rng.shuffle(pools[label])
                    keys.append((epoch, pools[label].pop(), len(keys)))
            for start in range(position, len(keys), s.batch_size):
                yield keys[start:start + s.batch_size]
            return
        order = torch.randperm(len(s.dataset), generator=torch.Generator().manual_seed(
            (s.seed + epoch) % (2 ** 63 - 1))).tolist()
        for start in range(position, len(order), s.batch_size):
            yield [(epoch, i) for i in order[start:start + s.batch_size]]


def initialize_data_worker(worker_id):
    """Only the parent handles terminal Ctrl+C and owns checkpoint/shutdown."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


@contextmanager
def training_session(stream):
    """Keep the graceful signal handler active through saving and worker cleanup."""
    global TRAINING_ACTIVE
    TRAINING_ACTIVE = True
    try:
        yield
    finally:
        try:
            stream.close()
        finally:
            TRAINING_ACTIVE = False


class TrainingStream:
    """Checkpoint consumed examples, never the sampler's prefetched position."""
    def __init__(self, dataset, cfg, state=None):
        self.dataset = dataset
        self.batch_size = cfg['batch_size']
        self.seed = state['seed'] if state else (cfg['seed'] or random.randrange(2 ** 63 - 1))
        self.epoch = self.position = 0
        self.class_groups = {}
        if cfg['conditioning_mode'] == 'class' and cfg.get('class_sampling', 'uniform') == 'uniform':
            for index, (_, label) in enumerate(dataset.samples):
                self.class_groups.setdefault(label, []).append(index)
        self.epoch_size = (len(self.class_groups) * max(map(len, self.class_groups.values()))
                           if self.class_groups else len(dataset))
        manifest = [(p, os.stat(p).st_size, os.stat(p).st_mtime_ns) for p in dataset_files(dataset)]
        data_keys = ('conditioning_mode', 'class_names', 'pix2pix_source_mode', 'pix2pix_direction',
                     'synthetic_params', 'channels', 'width', 'height', 'resize_width', 'resize_height',
                     'crop_width_min', 'crop_width_max', 'crop_height_min', 'crop_height_max',
                     'use_flip', 'use_vflip', 'use_rot90', 'validation_percent', 'validation_seed')
        data_settings = {k: cfg.get(k) for k in data_keys}
        if self.class_groups:
            data_settings['class_sampling'] = 'uniform_v1'
        self.fingerprint = stable_digest([manifest, data_settings])
        if state:
            if state['fingerprint'] != self.fingerprint:
                raise ConfigError('Dataset or transforms changed. Set reset_data_stream=true to start a new stream.')
            self.seed, self.epoch, self.position = state['seed'], state['epoch'], state['position']
            if not 0 <= self.position <= self.epoch_size or self.epoch < 0:
                raise ConfigError('Invalid saved data-stream position')
        workers = cfg['num_workers']
        self.loader = DataLoader(SeededDataset(dataset, self.seed), batch_sampler=StreamBatchSampler(self),
                                 num_workers=workers, pin_memory=torch.cuda.is_available(),
                                 persistent_workers=workers > 0,
                                 worker_init_fn=initialize_data_worker,
                                 generator=torch.Generator().manual_seed(self.seed),
                                 **({'prefetch_factor': 2} if workers else {}))
        self.iterator = None
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise RuntimeError('Training data stream is closed')
        if self.position == self.epoch_size:
            self.epoch += 1
            self.position = 0
            self.iterator = None
        if self.iterator is None:
            self.iterator = iter(self.loader)
        batch = next(self.iterator)
        self.position += min(self.batch_size, self.epoch_size - self.position)
        return batch

    def state_dict(self):
        return dict(seed=self.seed, epoch=self.epoch, position=self.position, fingerprint=self.fingerprint)

    def close(self):
        """Release persistent workers while Python/PyTorch are still fully alive."""
        if self.closed:
            return
        previous_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            iterator = self.iterator if self.iterator is not None else getattr(self.loader, '_iterator', None)
            # PyTorch 2.4 has no public DataLoader.close(). Its iterator shutdown
            # closes the pin-memory thread, signals workers, and joins them.
            shutdown = getattr(iterator, '_shutdown_workers', None)
            if shutdown is not None:
                shutdown()
            self.iterator = None
            self.loader._iterator = None
            self.closed = True
        finally:
            signal.signal(signal.SIGINT, previous_handler)


def prepare_training_batch(batch, cfg, device):
    condition = labels = None
    if cfg['conditioning_mode'] == 'class':
        data, labels = batch
        labels = labels.to(device)
    elif cfg['conditioning_mode'] == 'pix2pix':
        data, condition = batch
    else:
        data = batch
    dtype = torch.bfloat16 if cfg['full_bf16'] else torch.float32
    data = data.to(device=device, dtype=dtype)
    if condition is not None:
        condition = condition.to(device=device, dtype=dtype)
    return data, condition, labels


def accumulated_backward(flow, batches, cfg, scaler, amp_enabled, amp_dtype):
    """Weight unequal microbatches by example count; one optimizer update follows.

    Returns the weighted loss as a device tensor so the caller can read it in
    the same host sync as the gradient norm. A non-finite loss propagates into
    the result (and the gradients), and the caller rejects the whole update.
    """
    sizes = [len(b if cfg['conditioning_mode'] == 'unconditional' else b[0]) for b in batches]
    total = sum(sizes)
    value = None
    device = next(flow.parameters()).device
    for batch, size in zip(batches, sizes):
        data, condition, labels = prepare_training_batch(batch, cfg, device)
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
            loss = flow.p_losses(data, condition=condition, class_labels=labels)
        weighted = loss.detach().float() * (size / total)
        value = weighted if value is None else value + weighted
        scaler.scale(loss * (size / total)).backward()
    return value


@torch.no_grad()
def validation_loss(model, dataset, cfg, amp_enabled=False, amp_dtype=torch.float16):
    """Fixed examples/corruptions/noise/times; restore all module modes and RNGs."""
    modes = [(m, m.training) for m in model.modules()]
    try:
        with preview_randomness(cfg['validation_seed']):
            model.eval()
            flow = FlowMatchingWrapper(model, pred_mode=cfg['pred_mode'], loss_mode=cfg['loss_mode'],
                                       t_loc=cfg['t_mu'], t_scale=cfg['t_sigma'], x_clip=cfg['x_clip'],
                                       noise_schedule=cfg['noise_schedule'],
                                       self_cond_prob=cfg['self_cond_prob'], class_dropout_prob=0).eval()
            loader = DataLoader(SeededDataset(dataset, cfg['validation_seed']),
                                batch_size=cfg['validation_batch_size'], shuffle=False, num_workers=0,
                                generator=torch.Generator().manual_seed(cfg['validation_seed']))
            total = count = 0
            for i, batch in enumerate(loader):
                if cfg['validation_max_batches'] and i >= cfg['validation_max_batches']:
                    break
                data, condition, labels = prepare_training_batch(batch, cfg, next(model.parameters()).device)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                    loss = flow.p_losses(data, condition=condition, class_labels=labels)
                total += loss.float().item() * len(data)
                count += len(data)
            return total / count
    finally:
        for module, training in modes:
            module.training = training


def configure_trainable(model, cfg):
    """Freeze before optimizer construction; same policy is reapplied on continue."""
    policy = cfg['finetune_policy']
    patterns = cfg['trainable_modules']
    if policy == 'last':
        if isinstance(model, FCDMUNet):
            container, head = 'stages', ['output_conv', 'final_norm', 'final_modulation', 'final_conv']
        elif isinstance(model, ConvNetModel):
            container, head = 'ups', ['final_norm', 'final_conv']
        elif isinstance(model, HierMLPModel):
            container, head = 'refinement_stages', ['to_pixels', 'draft_to_pixels']
            if not len(model.refinement_stages):
                container = 'coarse_mixer'
        else:
            container = 'rin_blocks' if model.model_type == 'rin' else 'layers'
            head = ['to_pixels', 'norm', 'final_adaLN']
        blocks = getattr(model, container)
        n = cfg['finetune_last_blocks']
        if n > len(blocks):
            raise ConfigError(f'Requested {n} blocks, but {container} has {len(blocks)}')
        patterns = head + [f'{container}.{i}' for i in range(len(blocks) - n, len(blocks))]
    for name, param in model.named_parameters():
        param.requires_grad_(policy == 'all' or any(
            name == p or name.startswith(p + '.') or fnmatch.fnmatchcase(name, p) for p in patterns))
    if not any(p.requires_grad for p in model.parameters()):
        raise ConfigError('Freeze policy selected no trainable parameters')
    # Frozen dropout and normalization modules should not change behavior/buffers.
    for module in model.modules():
        params = list(module.parameters())
        if params and not any(p.requires_grad for p in params):
            module.eval()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'│  Trainable parameters: {trainable:,} / {total:,} ({policy})')


def align_grad_strides(params):
    """Give each gradient its parameter's strides (a zero-copy view).

    cuDNN returns depthwise / 1x1 conv weight gradients whose strides differ
    from the weight's only on size-1 dimensions: the same memory layout, but
    foreach optimizer kernels demand identical strides and otherwise fall back
    to one launch per parameter for the whole parameter list.
    """
    for p in params:
        g = p.grad
        if g is None or g.stride() == p.stride():
            continue
        if p.is_contiguous() and g.is_contiguous():
            p.grad = g.as_strided(g.shape, p.stride())
        else:
            p.grad = torch.empty_strided(p.shape, p.stride(), dtype=g.dtype, device=g.device).copy_(g)


def atomic_torch_save(value, path):
    temporary = str(path) + '.tmp'
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def update_best(model, cfg, score, step, best, path, metric, optimizer=None):
    if not cfg['save_best'] or not math.isfinite(score) or score >= best.get('score', float('inf')):
        return best
    result = dict(best, score=score, step=step, metric=metric)
    with schedule_free_eval(optimizer):
        atomic_torch_save(dict(model=model.state_dict(), config=copy.deepcopy(cfg), best=result), path)
    return result


def load_config(path):
    if str(path).lower().endswith('.json'):
        with open(path, encoding='utf-8') as handle:
            cfg = json.load(handle)
    else:
        cfg = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(cfg, dict):
        raise ConfigError('Configuration must be an object')
    return cfg


def stable_digest(value):
    """Equivalent JSON numbers (33 and 33.0) must not invalidate a resume."""
    normalized = json.loads(json.dumps(value),
                            parse_float=lambda v: int(float(v)) if float(v).is_integer() else float(v))
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()


def save_config(cfg, path):
    temporary = str(path) + '.tmp'
    try:
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(cfg, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def edit_config(cfg):
    """Retry starts fresh with saved defaults; edit any field using JSON values."""
    cfg = copy.deepcopy(cfg)
    print(json.dumps(cfg, indent=2, sort_keys=True))
    print('│  Edit key=value (JSON values; bare strings allowed), or Enter to train. Nested keys: synthetic_params.noise_min_sigma')
    while True:
        entry = input('│  Config edit: ').strip()
        if not entry:
            return cfg
        if '=' not in entry:
            print('│  Use key=value, for example dim=128 or model_type=rin')
            continue
        key, raw = (s.strip() for s in entry.split('=', 1))
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        keys = key.split('.')
        target = cfg
        for part in keys[:-1]:
            if not isinstance(target.get(part), dict):
                break
            target = target[part]
        else:
            target[keys[-1]] = value
            continue
        print('│  Parent setting must be an existing object')


def prompt_training_features(cfg):
    cfg['grad_accum_steps'] = get_number('Gradient accumulation batches per update', cfg['grad_accum_steps'], int, 1)
    cfg['validation_path'] = get_input('Validation folder (blank=use percentage/off)', cfg['validation_path'])
    if cfg['validation_path'] == '-':
        cfg['validation_path'] = ''
    cfg['validation_percent'] = (0.0 if cfg['validation_path'] else get_number(
        'Validation percentage (0=off)', cfg['validation_percent'], float, 0, 99))
    if cfg['validation_path'] or cfg['validation_percent']:
        cfg['validation_every'] = get_number('Validate every N updates', cfg['validation_every'], int, 1)
        cfg['validation_batch_size'] = get_number('Validation batch size', cfg['validation_batch_size'], int, 1)
        cfg['validation_max_batches'] = get_number('Validation batch limit (0=all)', cfg['validation_max_batches'], int, 0)
        cfg['validation_seed'] = get_number('Validation split/evaluation seed', cfg['validation_seed'], int, 0)
    cfg['save_best'] = get_input('Save best weights?', cfg['save_best'], bool)
    if not (cfg['validation_path'] or cfg['validation_percent']):
        cfg['best_training_loss'] = get_input('Use smoothed training loss for best weights?', cfg['best_training_loss'], bool)


def conditioning_model_kwargs(cfg):
    mode = cfg.get('conditioning_mode', 'unconditional')
    return {
        'input_channels': cfg['channels'] * 2 if mode == 'pix2pix' else cfg['channels'],
        'cond_residual': mode == 'pix2pix' and cfg.get('cond_residual', False),
        'class_count': len(cfg.get('class_names', [])) if mode == 'class' else 0,
        # Missing key means an old checkpoint: preserve its embedding shape.
        'class_cfg': mode == 'class' and cfg.get('class_dropout_prob', 0.0) > 0,
    }


# ==========================================
# 8. Learning Rate Schedule
# ==========================================

def report_training_spike(output_dir, report, *, kind, value, limit, loss,
                          completed_steps, optimizer, total_skips, consecutive_skips):
    """Keep rejected-batch diagnostics visible and append them to the run log.

    Logging failure must not interrupt checkpointing or change spike decisions.
    """
    reason = 'finite spike' if math.isfinite(value) else 'nonfinite'
    lrs = ','.join(f"{group['lr']:.6g}" for group in optimizer.param_groups)
    message = (f"spike guard: after {completed_steps} updates, skipped {kind} {value:.6g} "
               f"({reason}; limit {limit:.6g}) · loss {loss:.6g} · lr [{lrs}] "
               f"· total {total_skips} · consecutive {consecutive_skips}")
    report(message)
    try:
        with open(os.path.join(output_dir, 'spike_events.log'), 'a', encoding='utf-8') as log:
            log.write(message + '\n')
    except OSError as exc:
        report(f"Could not save spike diagnostics: {exc}")


class CosineWarmupScheduler:
    """Linear warmup, then either hold constant (paper default) or cosine-decay."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1, mode="constant",
                 resume_step=None, min_lrs=None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.mode = mode
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.resume_step = resume_step
        self.min_lrs = min_lrs or [lr * min_lr_ratio for lr in self.base_lrs]

    def step(self, current_step):
        if self.mode == 'cosine' and self.resume_step is not None:
            progress = min(1.0, max(0.0, (current_step - self.resume_step) /
                                    max(1, self.total_steps - 1 - self.resume_step)))
            for pg, start, end in zip(self.optimizer.param_groups, self.base_lrs, self.min_lrs):
                pg['lr'] = end + (start - end) * .5 * (1 + math.cos(math.pi * progress))
            return
        if current_step < self.warmup_steps:
            scale = current_step / max(1, self.warmup_steps)
        elif self.mode == "cosine":
            progress = min(1.0, max(0.0, (current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)))
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))
        else:  # "constant" (paper: constant LR after warmup)
            scale = 1.0
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale


# ==========================================
# 9. Sampling Utilities
# ==========================================

def make_sample_grid(images, nrow=4):
    """Create a grid image from a batch of [-1, 1] tensors."""
    images = ((images.clamp(-1, 1) + 1) * 0.5).float()  # to [0, 1]
    from torchvision.utils import make_grid
    return T.ToPILImage()(make_grid(images, nrow=nrow, padding=2))


def tensor_to_pil_image(image):
    """Convert one model image in [-1, 1] into a CPU PIL image."""
    return T.ToPILImage()(((image.detach().cpu().clamp(-1, 1) + 1) * 0.5).float())


def make_combined_pix2pix_image(source, generated, direction):
    """Return a combined pair with the dataset convention A-left and B-right."""
    source_image = tensor_to_pil_image(source)
    generated_image = tensor_to_pil_image(generated)
    if direction == 'a_to_b':
        image_a, image_b = source_image, generated_image
    elif direction == 'b_to_a':
        image_a, image_b = generated_image, source_image
    else:
        raise ValueError(f"Unknown Pix2Pix direction {direction!r}")
    combined = Image.new(image_a.mode, (image_a.width + image_b.width, image_a.height))
    combined.paste(image_a, (0, 0))
    combined.paste(image_b, (image_a.width, 0))
    return combined


def make_unique_output_dir(parent, stem):
    """Create a non-overwriting numbered subdirectory and return its path."""
    index = 0
    while True:
        name = stem if index == 0 else f"{stem}_{index}"
        path = os.path.join(parent, name)
        try:
            os.makedirs(path)
            return path
        except FileExistsError:
            index += 1


def capture_rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state.get('cuda'):
        for device, rng in enumerate(state['cuda'][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(rng.cpu(), device)


@contextmanager
def preview_randomness(seed):
    """Keep preview selection, augmentation and sampling out of training RNGs."""
    state = capture_rng_state()
    try:
        set_seed(seed)
        yield
    finally:
        restore_rng_state(state)


def preview_class_ids(class_count, count, seed):
    """Shuffle complete class cycles; never repeat before covering all classes."""
    if class_count < 1 or count < 1:
        raise ValueError('Class and preview counts must be positive')
    rng = random.Random(seed)
    labels = []
    while len(labels) < count:
        labels.extend(rng.sample(range(class_count), min(class_count, count - len(labels))))
    return labels


def caption_image(image, caption):
    """Place a small, wrapped caption below the image, leaving pixels visible."""
    image = image.convert('RGB')
    if image.width < 128:
        image = image.resize((128, max(1, round(image.height * 128 / image.width))), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    lines, line = [], ''
    for word in str(caption).split():
        if line and draw.textbbox((0, 0), line + ' ' + word)[2] > image.width - 12:
            lines.append(line)
            line = ''
        for char in (' ' if line else '') + word:
            if line and draw.textbbox((0, 0), line + char)[2] > image.width - 12:
                lines.append(line)
                line = ''
            line += char
    lines.append(line)
    tile = Image.new('RGB', (image.width, image.height + 8 + 14 * len(lines)), '#202020')
    tile.paste(image, (0, 0))
    draw = ImageDraw.Draw(tile)
    for i, line in enumerate(lines):
        draw.text((6, image.height + 4 + i * 14), line, fill='#dddddd')
    return tile


def make_preview_grid(samples, labels=None, sources=None, targets=None):
    """Class captions or source/generated/target triptychs, one group per sample."""
    panels = []
    for i, sample in enumerate(samples):
        if sources is not None:
            tiles = [caption_image(tensor_to_pil_image(image), label)
                     for image, label in ((sources[i], 'Source'), (sample, 'Generated'), (targets[i], 'Target'))]
            panel = Image.new('RGB', (sum(t.width for t in tiles) + 4, max(t.height for t in tiles)), '#202020')
            x = 0
            for tile in tiles:
                panel.paste(tile, (x, 0))
                x += tile.width + 2
        else:
            panel = caption_image(tensor_to_pil_image(sample), labels[i] if labels is not None else f'Sample {i + 1}')
        panels.append(panel)
    columns = math.ceil(math.sqrt(len(panels)))
    width, height = max(p.width for p in panels), max(p.height for p in panels)
    grid = Image.new('RGB', (columns * (width + 4) + 4,
                            math.ceil(len(panels) / columns) * (height + 4) + 4), '#303030')
    for i, panel in enumerate(panels):
        grid.paste(panel, (4 + (i % columns) * (width + 4), 4 + (i // columns) * (height + 4)))
    return grid


def training_preview(model, dataset, cfg, output_path, amp_enabled=False, amp_dtype=torch.float16):
    """Generate the full fixed panel in bounded chunks; restore RNG and mode on failure."""
    states = [(module, module.training) for module in model.modules()]
    seed = cfg.get('preview_seed', 42)
    try:
        with preview_randomness(seed):
            model.eval()
            flow = FlowMatchingWrapper(model, pred_mode=cfg['pred_mode'], loss_mode=cfg['loss_mode'],
                                       x_clip=cfg['x_clip'], noise_schedule=cfg['noise_schedule'])
            count = cfg['num_sample_images']
            chunk_size = cfg.get('preview_batch_size', cfg['batch_size'])
            device = next(model.parameters()).device
            labels = (preview_class_ids(len(cfg['class_names']), count, seed)
                      if cfg['conditioning_mode'] == 'class' else None)
            rng = random.Random(seed)
            indices = []
            if cfg['conditioning_mode'] == 'pix2pix':
                if len(dataset) == 0:
                    raise ValueError('Cannot preview an empty paired dataset')
                while len(indices) < count:
                    indices.extend(rng.sample(range(len(dataset)), min(len(dataset), count - len(indices))))
            generated, sources, targets = [], [], []
            for start in range(0, count, chunk_size):
                end = min(count, start + chunk_size)
                condition = None
                if indices:
                    pairs = []
                    for slot in range(start, end):
                        # Fix each crop/degradation independently of chunk size.
                        set_seed(seed + slot)
                        pairs.append(dataset[indices[slot]])
                    targets.extend(pair[0].cpu() for pair in pairs)
                    sources.extend(pair[1].cpu() for pair in pairs)
                    condition = torch.stack([pair[1] for pair in pairs]).to(device)
                shape = (end - start, cfg['channels'], cfg['height'], cfg['width'])
                noise = torch.stack([torch.randn(shape[1:], generator=torch.Generator().manual_seed(seed + i))
                                     for i in range(start, end)])
                classes = torch.tensor(labels[start:end], device=device) if labels is not None else None
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                    out = flow.sample(shape, steps=cfg['sampling_steps'], condition=condition,
                                      class_labels=classes, guidance_scale=cfg['guidance_scale'], initial_noise=noise)
                if not torch.isfinite(out).all():
                    raise ValueError('Preview produced nonfinite pixels')
                generated.extend(out.cpu())
            captions = [cfg['class_names'][label] for label in labels] if labels is not None else None
            grid = make_preview_grid(generated, captions, sources if indices else None, targets if indices else None)
            grid.save(output_path)
    finally:
        for module, training in states:
            module.training = training


def safe_training_preview(model, dataset, cfg, output_path, report, **kwargs):
    """Optional previews must not prevent training or its checkpoint saves."""
    try:
        training_preview(model, dataset, cfg, output_path, **kwargs)
    except Exception as exc:
        oom = isinstance(exc, torch.cuda.OutOfMemoryError)
        report(f'Preview skipped ({type(exc).__name__}): {exc}')
    else:
        report(f'✓ Preview saved: {output_path}')
        return True
    # The failed preview stack has unwound before releasing unused CUDA blocks.
    if oom:
        torch.cuda.empty_cache()
    return False


def pix2pix_source_canvas(path, cfg):
    """Load a Pix2Pix source at the pixel scale the model was trained at.

    Training resizes each image onto its canvas (resize_*; -1 = native, enlarged
    to at least the largest crop), crops a window and resizes that window to the
    model size. Inference reuses the canvas and applies the same window-to-model
    scale, using the training crop closest to the model size (no rescale when the
    crop range includes it, as in the default resize == crop == model setup).
    The result is at least model-sized; frame it with pix2pix_tile_corners.
    """
    height, width = cfg['height'], cfg['width']
    crop_h = (cfg.get('crop_height_min', height), cfg.get('crop_height_max', height))
    crop_w = (cfg.get('crop_width_min', width), cfg.get('crop_width_max', width))
    transform = CropAwareTransform((cfg.get('resize_height', height), cfg.get('resize_width', width)),
                                   (crop_h, crop_w), output_size=(height, width))
    mode = {3: 'RGB', 1: 'L', 4: 'RGBA'}.get(cfg['channels'], 'RGB')
    with Image.open(path) as image:
        image = image.convert(mode)
        canvas_h, canvas_w = transform.canvas_size(image)
        if image.size != (canvas_w, canvas_h):
            image = image.resize((canvas_w, canvas_h), transform.resample)
    window_h = min(max(height, crop_h[0]), crop_h[1])
    window_w = min(max(width, crop_w[0]), crop_w[1])
    target = (max(height, round(canvas_h * height / window_h)), max(width, round(canvas_w * width / window_w)))
    if image.size != (target[1], target[0]):
        image = image.resize((target[1], target[0]), transform.resample)
    return transform.tensor(image)


def pix2pix_tile_corners(canvas_size, tile_size, framing='tile'):
    """Top-left corners of model-sized windows: one centred window, or tiles covering
    the canvas (the last row/column is aligned to the edge and overlaps its neighbour)."""
    (canvas_h, canvas_w), (tile_h, tile_w) = canvas_size, tile_size
    if framing == 'center':
        return [((canvas_h - tile_h) // 2, (canvas_w - tile_w) // 2)]
    if framing != 'tile':
        raise ValueError(f"Unknown Pix2Pix framing {framing!r}")

    def starts(total, size):
        values = list(range(0, total - size + 1, size))
        return values + [total - size] if values[-1] + size < total else values
    return [(top, left) for top in starts(canvas_h, tile_h) for left in starts(canvas_w, tile_w)]


def stitch_tiles(tiles, corners, canvas_size):
    """Place generated tiles on the canvas, averaging where edge tiles overlap."""
    channels = tiles[0].shape[0]
    total = torch.zeros(channels, *canvas_size)
    weight = torch.zeros(1, *canvas_size)
    for tile, (top, left) in zip(tiles, corners):
        h, w = tile.shape[1:]
        total[:, top:top + h, left:left + w] += tile.float().cpu()
        weight[:, top:top + h, left:left + w] += 1
    return total / weight


def save_checkpoint(model, optimizer, ema, step, path, max_keep=3, *, scaler=None, cfg=None, guard=None,
                    stream=None, best=None, scheduler=None):
    """Save checkpoint and rotate old ones."""
    # state_dict() aliases live tensors, so schedule-free x must stay loaded until written.
    with schedule_free_eval(optimizer):
        save_dict = {
            'step': step,
            'completed_steps': step,
            'rng': capture_rng_state(),
            'config': copy.deepcopy(cfg),
            'guard': guard,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'data_stream': stream.state_dict() if stream is not None else None,
            'best': best or {},
            'scheduler': ({k: v for k, v in vars(scheduler).items() if k != 'optimizer'}
                          if scheduler is not None else None),
        }
        if ema:
            save_dict['ema'] = ema.shadow.state_dict()
            save_dict['ema_step_count'] = ema.step_count
        if scaler is not None:
            save_dict['scaler'] = scaler.state_dict()
        atomic_torch_save(save_dict, path)

    # Numbered backups
    backup_path = path.replace('.pt', f'_step0.pt')
    shutil.copy2(path, backup_path)

    # Rotate old backups
    backups = sorted(glob.glob(path.replace('.pt', '_step*.pt')), key=os.path.getmtime)
    while len(backups) > max_keep:
        os.remove(backups.pop(0))


# ==========================================
# 10. CLI & Main
# ==========================================

UI_WIDTH = 76


def _ui_text(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def ui_rule(char="─"):
    print(char * UI_WIDTH)


def ui_header(title, subtitle=None):
    print()
    ui_rule("═")
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    ui_rule("═")


def ui_section(title):
    print(f"\n┌─ {title} " + "─" * max(0, UI_WIDTH - len(title) - 4))


def ui_key_value(label, value):
    print(f"│  {label:<22} {_ui_text(value)}")


def ui_menu(items, columns=2):
    """Print numbered menu items in compact, terminal-friendly columns."""
    entries = [f"{key:>2}  {description}" for key, description in items]
    rows = math.ceil(len(entries) / columns)
    column_width = max(len(entry) for entry in entries) + 3
    for row in range(rows):
        line = ""
        for column in range(columns):
            index = row + column * rows
            if index < len(entries):
                line += entries[index].ljust(column_width)
        print(f"│  {line.rstrip()}")


def ui_config_summary(cfg, parameter_count=None):
    ui_header("Run summary", "Reviewing the configuration before work begins")
    ui_section("Model")
    ui_key_value("Architecture", cfg['model_type'])
    ui_key_value("Image", f"{cfg['width']}×{cfg['height']} · {cfg['channels']} channel(s)")
    if cfg['model_type'] in {'fcdm_unet', 'fcdm_isotropic'}:
        ui_key_value('FCDM width / depth', f"{cfg['dim']} / {cfg['depth']}")
        ui_key_value('FCDM expansion', cfg['fcdm_mlp_ratio'])
    if cfg['model_type'] == 'rin':
        ui_key_value("RIN widths", f"interface {cfg['dim']} · latents {cfg['rin_latent_dim']}")
        ui_key_value("RIN routing", f"{cfg['depth']} blocks · {cfg['rin_layers_per_block']} latent layers/block")
        ui_key_value("RIN latent slots", f"{cfg['rin_num_latents']} including conditioning tokens")
        ui_key_value("Latent self-conditioning", cfg['self_cond_prob'] if cfg['self_cond'] else 'off')
    if cfg.get('conv_stem', 0):
        ui_key_value("Conv-stem", {1: 'input only', 2: 'output only', 3: 'both'}[cfg['conv_stem']])
        if cfg['model_type'] == 'hiermlp':
            ui_key_value("Stem grids", f"input {cfg['hier_input_grid_size']}² · output {cfg['hier_output_grid_size']}²")
        else:
            ui_key_value("Stem token grid", f"{cfg['stem_width']}×{cfg['stem_height']}")
        ui_key_value("Stem features", f"{cfg['stem_initial']} initial · {cfg['stem_max']} maximum")
        ui_key_value("Stem activation", STEM_ACTIVATIONS[cfg.get('stem_activation', 4)])
        if cfg.get('stem_activation', 4) >= 7:
            ui_key_value("Stem GLU scaling", '2/3 widths' if cfg.get('stem_glu_scaling', 0) else 'default widths')
        ui_key_value("Stem skips", 'on (concatenate)' if cfg.get('stem_skips', False) else 'off')
        ui_key_value("Stem conditioning", STEM_CONDITIONING[cfg.get('stem_conditioning', 0)])
        ui_key_value("Stem GRN", 'on' if cfg.get('stem_grn', False) else 'off')
    if cfg['model_type'] == 'hiermlp':
        ui_key_value(
            "HierMLP grids",
            f"input {cfg.get('hier_input_grid_size', cfg['initial_grid_size'])}×{cfg.get('hier_input_grid_size', cfg['initial_grid_size'])}"
            f" → root {cfg.get('hier_output_grid_size', cfg['initial_grid_size'])}×{cfg.get('hier_output_grid_size', cfg['initial_grid_size'])}",
        )
        ui_key_value("Hier processor", cfg.get('hier_global_mixer', 'mlpmixer'))
        ui_key_value(
            "Hier processor size",
            f"width {cfg.get('hier_global_dim', cfg['dim'])} · {cfg.get('hier_global_depth', cfg['depth'])} layers",
        )
        if cfg.get('hier_global_mixer') == 'jit':
            ui_key_value("Hier JiT heads", cfg.get('hier_global_heads', 4))
    source_size = (cfg.get('resize_width', cfg['width']), cfg.get('resize_height', cfg['height']))
    crop_range = (cfg.get('crop_width_min', cfg['width']), cfg.get('crop_width_max', cfg['width']),
                  cfg.get('crop_height_min', cfg['height']), cfg.get('crop_height_max', cfg['height']))
    if -1 in source_size:
        ui_key_value("Source sizing", f"native dimension(s) → crop {crop_range[0]}–{crop_range[1]}×{crop_range[2]}–{crop_range[3]}")
    elif crop_range != (cfg['width'], cfg['width'], cfg['height'], cfg['height']):
        ui_key_value("Variable crop", f"{crop_range[0]}–{crop_range[1]}×{crop_range[2]}–{crop_range[3]} → {cfg['width']}×{cfg['height']}")
    elif source_size != (cfg['width'], cfg['height']):
        ui_key_value("Random crop", f"{cfg['width']}×{cfg['height']} from {cfg['resize_width']}×{cfg['resize_height']}")
    ui_key_value("Parameters", f"{parameter_count:,}" if parameter_count is not None else "building…")
    ui_key_value("Objective", f"{cfg['pred_mode']}-pred / {cfg['loss_mode']}-loss")
    condition = cfg.get('conditioning_mode', 'unconditional')
    if condition == 'class':
        condition = f"class ({len(cfg.get('class_names', []))} classes)"
    elif condition == 'pix2pix':
        condition = f"Pix2Pix {cfg.get('pix2pix_source_mode', 'folders')} · {cfg.get('pix2pix_direction', 'a_to_b')}"
    ui_key_value("Conditioning", condition)
    ui_section("Training")
    ui_key_value("Optimizer", f"{cfg['optimizer_type']} · lr {cfg['lr']:.2e}")
    ui_key_value("Batch / steps", f"{cfg['batch_size']} / {cfg['steps']:,}")
    ui_key_value("Spike guard", "on" if cfg.get('use_spike_guard', True) else "off")
    augmentations = []
    if cfg.get('use_flip', False): augmentations.append('horizontal flip')
    if cfg.get('use_vflip', False): augmentations.append('vertical flip')
    if cfg.get('use_rot90', False): augmentations.append('90° rotations')
    if augmentations:
        ui_key_value("Augmentation", ', '.join(augmentations))
    if cfg.get('full_bf16', False):
        precision = "full BF16 (experimental)"
    elif cfg['use_amp']:
        precision = f"AMP {cfg.get('amp_dtype', 'fp16').upper()}"
    else:
        precision = "full FP32"
    ui_key_value("Precision", precision)
    ui_key_value("Samples / checkpoint", f"every {cfg['sample_every']:,} / {cfg['save_every']:,} steps")
    ui_section("Paths")
    ui_key_value("Dataset", cfg['dataset_path'])
    ui_key_value("Output", SAVE_DIR)
    ui_rule()


def get_input(prompt, default=None, cast_type=str):
    default_text = _ui_text(default) if default is not None else "required"
    user_val = input(f"│  {prompt} [{default_text}]: ").strip()
    if user_val == "":
        # Fall through to casting the default too, so that e.g. a bool prompt
        # with default "0" yields False (not the truthy string "0"). Without
        # this, accepting the displayed default silently enabled every
        # off-by-default toggle (self-cond, grad-ckpt, conv-mlp, ...).
        if default is None:
            return None
        user_val = str(default)
    if cast_type == bool:
        return user_val.lower() in ['1', 'yes', 'true', 'y', 'on']
    return cast_type(user_val)


def get_number(prompt, default, cast_type, minimum=0, maximum=None):
    """Reprompt invalid resume settings before mutating optimizer state."""
    while True:
        try:
            value = get_input(prompt, default, cast_type)
            if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
                raise ValueError('outside the allowed range')
            return value
        except (TypeError, ValueError):
            print(f"│  Enter a finite number >= {minimum}" +
                  (f" and <= {maximum}." if maximum is not None else '.'))


def prompt_stem_activation():
    choices = ', '.join(f'{key}={name}' for key, name in STEM_ACTIVATIONS.items())
    while True:
        choice = get_number(f'Stem activation [{choices}]', 4, int, 0, 10)
        if choice in STEM_ACTIVATIONS:
            return choice
        print('│  Option 2 is unused; select one of the listed activations.')


def prompt_stem_extras(cfg):
    cfg['stem_glu_scaling'] = (get_number('Stem GLU scaling [0=default, 1=Transformer 2/3 widths]', 0, int, 0, 1)
                               if cfg['stem_activation'] >= 7 else 0)
    cfg['stem_skips'] = (get_input('U-Net-style stem skip connections?', False, bool)
                         if cfg['conv_stem'] == 3 else False)
    cfg['stem_grn'] = get_input('Use FCDM Global Response Normalization in stem stages?', False, bool)
    if cfg['stem_skips']:
        print('│  Skips require matching input/output bottleneck grids.')
    for value, label in STEM_CONDITIONING.items():
        print(f'│  {value}: {label}')
    while True:
        mode = get_number('Stem class/time conditioning', 0, int, 0, 5)
        try:
            validate_stem_conditioning(mode, cfg['conv_stem'], cfg['stem_skips'])
        except ValueError as exc:
            print(f'│  {exc}')
        else:
            cfg['stem_conditioning'] = mode
            break


def prompt_resume_settings(cfg, optimizer, completed_steps):
    """Keep architecture and optimizer type; edit compatible runtime settings."""
    ui_section('Resume settings')
    extra = get_number('Additional successful training steps',
                       max(1, cfg['steps'] - completed_steps), int)
    cfg['steps'] = completed_steps + extra
    cfg['batch_size'] = get_number('Batch size', cfg['batch_size'], int, 1)
    adaptive = cfg['optimizer_type'] in {'paper_adamhd', 'prodigy', 'radam_schedulefree'}
    cosine = (cfg['lr_schedule'] == 'cosine' and not adaptive
              and completed_steps >= cfg['warmup_steps'])
    cfg.setdefault('cosine_min_lr', cfg['lr'] * .1)
    default_lr = optimizer.param_groups[0]['lr'] if adaptive or cosine else cfg['lr']
    cfg['lr'] = get_number('Resume learning rate (cosine continues from here)' if cosine else 'Learning rate',
                           default_lr, float)
    apply_muon_group_lrs(optimizer, cfg)
    if get_input('Edit optimizer parameters?', False, bool):
        # Only scalar hyperparameters and beta coefficients are editable. Rank,
        # parameter grouping, optimizer type and tensor state stay compatible.
        bounds = {'weight_decay': (0, None), 'momentum': (0, .999999),
                  'eps': (1e-16, None), 'alpha': (0, .999999),
                  'hyper_lr': (0, None), 'nu': (0, None), 'beta2': (0, .999999)}
        if cfg['optimizer_type'] == 'adadeltago':
            bounds.update(gamma=(1e-16, None), rho=(0, .999999))
        if cfg['optimizer_type'] == 'rmsgo':
            bounds.update(gamma=(1e-16, None), v0=(1e-16, None))
        if cfg['optimizer_type'] == 'adamgo':
            bounds.update(gamma=(1e-16, None), delta=(1e-16, None), min_step=(0, None))
        for key, (minimum, maximum) in bounds.items():
            groups = [g for g in optimizer.param_groups if key in g]
            if groups:
                value = get_number(key, groups[0][key], float, minimum, maximum)
                for group in groups:
                    group[key] = value
        groups = [g for g in optimizer.param_groups if 'betas' in g]
        if groups:
            betas = tuple(get_number(f'Beta {i + 1}', value, float, 0, .999999)
                          for i, value in enumerate(groups[0]['betas']))
            for group in groups:
                group['betas'] = betas
        if cfg['optimizer_type'] in {'muon', 'adamuon', 'normuon', 'adago', 'adamgo', 'rmsgo', 'adadeltago'}:
            backend = optimizer.param_groups[0].get('orthogonalization_backend', 'polar_express')
            cfg['muon_backend'] = ('polar_express' if get_input('Use Polar Express backend?',
                                   backend == 'polar_express', bool) else 'newton_schulz')
            cfg['cautious'] = get_input('Cautious updates?',
                                       optimizer.param_groups[0].get('cautious', False), bool)
            for group in optimizer.param_groups:
                group['orthogonalization_backend'] = cfg['muon_backend']
                group['cautious'] = cfg['cautious']
            if not cfg.get('muon_all', False) and cfg['lr'] > 0:
                fallback = get_number('AdamW fallback learning rate (embeddings, heads, norms, biases)',
                                      cfg['lr'] * cfg.get('muon_adam_lr_ratio', 1.0), float, 1e-12)
                cfg['muon_adam_lr_ratio'] = fallback / cfg['lr']
                apply_muon_group_lrs(optimizer, cfg)
    if get_input('Edit training and preview settings?', False, bool):
        for key, label, minimum in (
                ('grad_clip', 'Gradient clip norm (0=off)', 0),
                ('sample_every', 'Sample every N successful steps', 1),
                ('save_every', 'Save every N successful steps', 1),
                ('sampling_steps', 'Preview solver steps', 1),
                ('preview_batch_size', 'Preview batch size', 1),
                ('preview_seed', 'Fixed preview seed', 0),
                ('num_sample_images', 'Preview image count', 1)):
            cfg[key] = get_number(label, cfg[key], float if key == 'grad_clip' else int, minimum)
        if cfg['conditioning_mode'] == 'class' and cfg['class_dropout_prob'] > 0:
            cfg['guidance_scale'] = get_number(
                'Preview CFG strength (1=unguided)', cfg['guidance_scale'], float, 0)
        cfg['compile_mode'] = prompt_compile_mode(cfg['compile_mode'])
        for key, label in (('use_spike_guard', 'Reject finite spikes?'),
                           ('use_flip', 'Random horizontal flips?'),
                           ('use_vflip', 'Random vertical flips?'),
                           ('use_rot90', 'Random square-image rotations?')):
            cfg[key] = get_input(label, cfg[key], bool)


MODEL_TYPES = {
    '1': ("jit", "JiT (Transformer)"),
    '2': ("oggmlp", "gMLP (Basic)"),
    '3': ("ogamlp", "aMLP (gMLP + Tiny Attention)"),
    '4': ("basemlp", "BaseMLP (Non-residual)"),
    '5': ("mlp", "MLP (Transformer FF Only)"),
    '6': ("encdec", "EncDec (ConvNet)"),
    '7': ("unet", "UNET (ConvNet + Skips)"),
    '8': ("ogmlpmixer", "MLPMixer (Original)"),
    '9': ("oghypermixer", "HyperMixer (Original)"),
    '10': ("convnext", "ConvNeXt"),
    '11': ("fullattn", "FullAttention (No MLP)"),
    '12': ("pool", "ConvFormer (DWConv Token Mix)"),
    '13': ("fourier", "Fourier Mixer"),
    '14': ("gru", "Bi-GRU (Spatial Sweep)"),
    '15': ("lka", "LKA (Visual Attention Network)"),
    '16': ("xcit", "XCiT (Cross-Covariance)"),
    '17': ("swin_v1", "Swin Transformer v1"),
    '18': ("swin_v2", "Swin Transformer v2"),
    '19': ("bigs", "BiGS (Gated SSM)"),
    '20': ("hat", "HAT (Hybrid Attention)"),
    '21': ("gmlp", "Conv-gMLP"),
    '22': ("amlp", "Conv-aMLP"),
    '23': ("mlpmixer", "ConvMixer"),
    '24': ("hypermixer", "ConvHyperMixer"),
    '25': ("coatnet", "CoAtNet (MBConv + Relative Transformer)"),
    '26': ("mixer_attn", "MLPMixer (With Attention)"),
    '27': ("gatedmlpmixer", "MLPMixer (SwiGLU Gated)"),
    '28': ("cyclemlp", "CycleMLP (Spatial Shifting)"),
    '29': ("resmlp", "ResMLP (Affine + Linear Spatial)"),
    '30': ("mdmlp", "MDMLP (Parallel Axial + Gating)"),
    '31': ("vim", "Vision Mamba (Bi-SSM)"),
    '32': ("maxvit", "MaxViT (Block + Grid Attn)"),
    '33': ("focal", "FocalNet (Focal Modulation)"),
    '34': ("hornet",     "HorNet (Recursive Gated Conv)"),
    '35': ("aft_full",   "AFT-Full (Learned N×N bias)"),
    '36': ("aft_simple", "AFT-Simple (No position)"),
    '37': ("aft_local",  "AFT-Local (Windowed bias)"),
    '38': ("hyena",      "Hyena 2D (Implicit Long Conv + Gating)"),
    '39': ("nat",        "NAT (Neighborhood Attention)"),
    '40': ("volo",       "VOLO (Outlook Attention)"),
    '41': ("asmlp",      "AS-MLP (Axial Shift)"),
    '42': ("s2mlp",      "S2-MLP (4-Direction Shift)"),
    '43': ("vip",        "ViP (Vision Permutator)"),
    '44': ("rin",        "RIN (Recurrent Interface Network)"),
    '45': ("hiermlp",    "HierMLP (Local 4x4-to-Pixel Refinement)"),
    '46': ("fcdm_unet",  "FCDM-UNet"),
    '47': ("fcdm_isotropic", "FCDM-Isotropic"),
}

# Blocks that ignore t_emb; they receive time/class at the input even with adaLN.
INPUT_TIME_MODELS = {'pool', 'fourier', 'gru', 'mlpmixer', 'hypermixer'}
GRN_TOKEN_MODELS = {'jit', 'oggmlp', 'ogamlp', 'ogmlpmixer', 'gmlp', 'amlp', 'mlpmixer', 'vip'}
IS_CONV = {'encdec', 'unet', 'swin_v1', 'swin_v2'}
IS_HIERARCHICAL = {'hiermlp'}

NEEDS_HEADS = {'jit', 'fullattn', 'xcit', 'mixer_attn', 'hat', 'coatnet', 'nat', 'volo', 'rin'}
HYPER_HEADS = {'hypermixer', 'oghypermixer'}
AXIAL_MODELS = {'ogmlpmixer', 'oggmlp', 'ogamlp'}


COMPILE_MODES = ('off', 'default', 'max-autotune-no-cudagraphs')


def cuda_driver_version():
    """Driver CUDA version as 1000*major + 10*minor (11070 = 11.7), or None."""
    try:
        version = ctypes.c_int()
        if ctypes.CDLL('libcuda.so.1').cuDriverGetVersion(ctypes.byref(version)) == 0:
            return version.value
    except (OSError, AttributeError):
        pass
    return None


def ptxas_version(path):
    try:
        text = subprocess.run([path, '--version'], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r'release (\d+)\.(\d+)', text)
    return int(match[1]) * 1000 + int(match[2]) * 10 if match else None


def configure_triton_ptxas():
    """Point Triton at a ptxas the installed driver can load.

    Triton bundles a CUDA 12 ptxas. Under an older driver (e.g. 11.7) its
    kernels fail with "device kernel image is invalid"; a system ptxas no
    newer than the driver makes Triton emit PTX that driver accepts.
    Returns False only when a workaround is needed and none was found.
    """
    if os.environ.get('TRITON_PTXAS_PATH'):
        return True
    driver = cuda_driver_version()
    if driver is None or driver >= 12000:
        return True
    roots = [os.environ.get('CUDA_HOME'), os.environ.get('CUDA_PATH'), '/usr/local/cuda',
             *sorted(glob.glob('/usr/local/cuda-*'), reverse=True)]
    paths = [os.path.join(r, 'bin', 'ptxas') for r in roots if r] + [shutil.which('ptxas')]
    for path in dict.fromkeys(p for p in paths if p and os.path.isfile(p)):
        version = ptxas_version(path)
        if version is not None and version <= driver:
            os.environ['TRITON_PTXAS_PATH'] = path
            print(f"│  torch.compile: driver CUDA {driver // 1000}.{driver % 1000 // 10}; using {path}")
            return True
    print(f"│  torch.compile: driver CUDA {driver // 1000}.{driver % 1000 // 10} needs a CUDA "
          f"<= that ptxas (set TRITON_PTXAS_PATH); none found")
    return False


def maybe_compile(model, cfg, device):
    """Compile `model` in place (state_dict keys stay unprefixed); eager on failure."""
    mode = cfg.get('compile_mode', 'off')
    if mode == 'off':
        return
    if torch.device(device).type != 'cuda' or not configure_triton_ptxas():
        print("│  torch.compile: unavailable here; running eager")
        return
    import torch._dynamo as dynamo
    import torch._inductor.config as inductor_config
    # A backend failure falls back to eager for that frame instead of
    # killing the run; the first steps are slow while kernels compile.
    dynamo.config.suppress_errors = True
    # Each compile worker is a separate Python process holding its own torch
    # import; one per CPU core exhausts RAM alongside data-loader workers.
    if 'TORCHINDUCTOR_COMPILE_THREADS' not in os.environ:
        inductor_config.compile_threads = min(4, os.cpu_count() or 1)
    model.compile(mode=None if mode == 'default' else mode)
    print(f"│  torch.compile: {mode} (first steps compile kernels)")


def verify_compiled_gradients(model, flow, cfg, device, amp_enabled, amp_dtype, compiled_calls=3):
    """Compare compiled and eager gradients once; fall back to eager on mismatch.

    Under an older driver with the ptxas workaround, some autotuned Triton
    kernels have produced gradients 100-1000x too large in BF16 (seen on
    Swin v2, depending on shapes), while the loss looked normal. The compiled
    path runs several times so the kernels the autotuner settles on are the
    ones checked. A synthetic batch of the real shape is used; RNG state,
    buffers (e.g. BatchNorm statistics) and gradients are restored afterwards.
    Returns True when the compiled model is kept.
    """
    if getattr(model, '_compiled_call_impl', None) is None:
        return False
    dtype = torch.bfloat16 if cfg['full_bf16'] else torch.float32
    shape = (cfg['batch_size'], cfg['channels'], cfg['height'], cfg['width'])
    buffers = {k: b.detach().clone() for k, b in model.named_buffers()}
    rng = capture_rng_state()
    try:
        generator = torch.Generator(device=device).manual_seed(0)
        data = torch.rand(shape, device=device, generator=generator).mul(2).sub(1).to(dtype)
        condition = (torch.rand(shape, device=device, generator=generator).mul(2).sub(1).to(dtype)
                     if cfg['conditioning_mode'] == 'pix2pix' else None)
        labels = (torch.randint(len(cfg['class_names']), (shape[0],), device=device, generator=generator)
                  if cfg['conditioning_mode'] == 'class' else None)

        def gradient_norm():
            model.zero_grad(set_to_none=True)
            set_seed(0)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                loss = flow.p_losses(data, condition=condition, class_labels=labels)
            loss.float().backward()
            grads = [p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None]
            return torch.stack(grads).norm().item() if grads else 0.0

        compiled_impl = model._compiled_call_impl
        model._compiled_call_impl = None
        try:
            eager = gradient_norm()
        finally:
            model._compiled_call_impl = compiled_impl
        compiled = [gradient_norm() for _ in range(compiled_calls)]
    finally:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for k, b in model.named_buffers():
                b.copy_(buffers[k])
        restore_rng_state(rng)
    if not math.isfinite(eager) or eager == 0:
        print("│  torch.compile check: eager gradients unusable here; keeping compiled model")
        return True
    worst = max(max(c / eager, eager / max(c, 1e-30)) if math.isfinite(c) else float('inf') for c in compiled)
    if worst <= 2.0:
        print(f"│  torch.compile check: gradients match eager (worst norm ratio {worst:.3f})")
        return True
    print(f"│  ⚠️ torch.compile check FAILED: compiled gradient norm differs from eager by {worst:.3g}x.")
    print("│     Training eager instead. A driver with CUDA >= 12 is the real fix.")
    model._compiled_call_impl = None
    return False


def prompt_compile_mode(default):
    choice = get_input(f"torch.compile [{'/'.join(COMPILE_MODES)}]", default, str).lower()
    return choice if choice in COMPILE_MODES else default


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def clear_old_images(save_dir):
    """Deletes all image files in the save directory, leaving .pt files intact."""
    print(f"│  Clearing old samples from {save_dir}…")
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        for fpath in glob.glob(os.path.join(save_dir, ext)):
            try:
                os.remove(fpath)
            except Exception as e:
                print(f"│  ⚠ Could not remove {fpath}: {e}")


MODERN_CLION_DEFAULT_NU = 1e-15
OLD_MODERN_CLION_DEFAULT_NU = 1e-6

OPTIMIZER_TYPES = {
    '1': ('adam', 'Adam', 5e-4),
    '2': ('adan', 'Adan', 5e-4),
    '3': ('sgd', 'SGD', 1e-2),
    '4': ('nsgda', 'nSGDA', 5e-4),
    '5': ('layerwise_nsgda', 'LayerWise nSGDA', 5e-4),
    '6': ('ada_nsgda', 'Ada-nSGDA', 5e-4),
    '7': ('clion', 'CLion', 1e-4),
    '8': ('lamb', 'Lamb', 1e-3),
    '9': ('rmsprop', 'RMSProp', 1e-4),
    '10': ('adagrad', 'AdaGrad', 1e-2),
    '11': ('adadelta', 'Adadelta', 1.0),
    '12': ('prodigy', 'Prodigy', 1.0),
    '13': ('paper_adamhd', 'PaperAdamHD', 1e-3),
    '14': ('modern_clion', 'ModernCLion', 1e-4),
    '15': ('equalized_adamw', 'EqualizedAdamW', 1e-3),
    '16': ('muon', 'Muon', 4.2e-4),
    '17': ('adamuon', 'AdaMuon', 4.2e-4),
    '18': ('normuon', 'NorMuon', 4.2e-4),
    '19': ('adago', 'AdaGO', .05),
    '20': ('adamgo', 'AdamGO', .05),
    '21': ('rmsgo', 'RMSGO', .05),
    '22': ('adadeltago', 'AdaDeltaGO', .05),
    '23': ('radam_schedulefree', 'RAdamScheduleFree', 2.5e-3),
}


def get_optimizer_choice(opt_in):
    if opt_in in OPTIMIZER_TYPES:
        return OPTIMIZER_TYPES[opt_in]
    for opt_type, desc, default_lr in OPTIMIZER_TYPES.values():
        if opt_in.lower() in {opt_type.lower(), desc.lower()}:
            return opt_type, desc, default_lr
    return opt_in, opt_in, 5e-4


class PaperAdamHD(torch.optim.Optimizer):
    """Adam-HD using the additive hypergradient rule from Baydin et al. (ICLR 2018)."""
    def __init__(self, params, lr=1e-3, hyper_lr=1e-10, betas=(0.9, 0.999), eps=1e-8,
                 min_lr=1e-5, max_lr=1e-2):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if hyper_lr < 0.0:
            raise ValueError(f"Invalid hypergradient learning rate: {hyper_lr}")
        if min_lr < 0.0:
            raise ValueError(f"Invalid minimum learning rate: {min_lr}")
        if max_lr <= 0.0:
            raise ValueError(f"Invalid maximum learning rate: {max_lr}")
        if min_lr > max_lr:
            raise ValueError(f"Minimum learning rate {min_lr} is greater than maximum learning rate {max_lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, hyper_lr=hyper_lr, betas=betas, eps=eps,
                        min_lr=min_lr, max_lr=max_lr, last_hypergrad=0.0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            hypergrad = 0.0
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("PaperAdamHD does not support sparse gradients")
                state = self.state[p]
                if 'prev_d_update_d_lr' in state:
                    hypergrad += torch.sum(grad * state['prev_d_update_d_lr']).item()

            group['last_hypergrad'] = hypergrad
            group['lr'] -= group['hyper_lr'] * hypergrad
            group['lr'] = min(max(group['lr'], group['min_lr']), group['max_lr'])

            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                exp_avg_hat = exp_avg / (1 - beta1 ** step)
                exp_avg_sq_hat = exp_avg_sq / (1 - beta2 ** step)
                d_update_d_lr = -exp_avg_hat / (exp_avg_sq_hat.sqrt().add(eps))

                p.add_(d_update_d_lr, alpha=lr)
                state['prev_d_update_d_lr'] = d_update_d_lr.clone()

        return loss


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
            groups.append({'params': params})
    leftover = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    if leftover:
        groups.append({'params': leftover})
    return groups or [{'params': [p for p in model.parameters() if p.requires_grad]}]


def jit_muon_param_groups(model):
    """Use AdamW for learned image/time/class embeddings and pixel heads."""
    adamw_params = []
    for name, param in model.named_parameters():
        parts = name.split('.')
        if any('embedding' in part or part in {
                'time_mlp', 'to_pixels', 'draft_to_pixels', 'final_conv',
        } for part in parts):
            adamw_params.append(param)
    root = model
    while hasattr(root, '_orig_mod'):
        root = root._orig_mod
    # HierMLP's final refinement emits pixels directly, without a named head.
    stages = getattr(root, 'refinement_stages', None)
    if stages is not None and len(stages):
        adamw_params.extend(stages[-1].mlp[-1].parameters())
    return muon_param_groups(model, adamw_params,
                             param_groups=[{'params': [p for p in model.parameters() if p.requires_grad]}])


MUON_FAMILY = {'muon', 'adamuon', 'normuon', 'adago', 'adamgo', 'rmsgo', 'adadeltago'}
GO_FAMILY = {'adago', 'adamgo', 'rmsgo', 'adadeltago'}
# Suggested AdamW-fallback LR; GO matrix LRs (0.05) are far too large for Adam.
MUON_FALLBACK_LR = 4.2e-4


def muon_group_lr_scale(group, cfg):
    """LR multiplier for one param group: AdamW-fallback groups use the ratio.

    With muon_all the fallback groups follow the matrix rule, so they keep the
    main LR. Other optimizers' groups have no use_muon key and are unscaled.
    """
    if group.get('use_muon', True) or cfg.get('muon_all', False):
        return 1.0
    return cfg.get('muon_adam_lr_ratio', 1.0)


def apply_muon_group_lrs(optimizer, cfg):
    for group in optimizer.param_groups:
        group['lr'] = cfg['lr'] * muon_group_lr_scale(group, cfg)


def build_optimizer(model: nn.Module, cfg: dict):
    opt_type = cfg.get('optimizer_type', 'adan')
    lr = cfg['lr']
    trainable = [p for p in model.parameters() if p.requires_grad]
    if opt_type in {'muon', 'adamuon', 'normuon', 'adago', 'adamgo', 'rmsgo', 'adadeltago'}:
        optimizer_cls = (AdaDeltaGO if opt_type == 'adadeltago' else RMSGO if opt_type == 'rmsgo' else AdamGO if opt_type == 'adamgo' else AdaGO if opt_type == 'adago' else NorMuon if opt_type == 'normuon'
                         else AdaMuon if opt_type == 'adamuon' else Muon)
        adaptive = ({'eps': cfg.get('adamuon_eps', 1e-8),
                     'nesterov': cfg.get('adamuon_nesterov', False)} if opt_type == 'adamuon' else {})
        if opt_type == 'normuon':
            adaptive = {'eps': cfg.get('normuon_eps', 1e-8), 'beta2': cfg.get('normuon_beta2', .95)}
        if opt_type == 'adago':
            adaptive = {'eps': cfg.get('adago_eps', 5e-4), 'gamma': cfg.get('adago_gamma', 1.),
                        'v0': cfg.get('adago_v0', 1.)}
        if opt_type == 'rmsgo':
            adaptive = {'eps': cfg.get('rmsgo_eps', 5e-4), 'gamma': cfg.get('rmsgo_gamma', 1.),
                        'v0': cfg.get('rmsgo_v0', 1.)}
            adaptive['beta2'] = cfg.get('rmsgo_beta2', .99)
        if opt_type == 'adadeltago':
            adaptive = {'rho': cfg.get('adadeltago_rho', .9), 'gamma': cfg.get('adadeltago_gamma', 1.),
                        'eps': cfg.get('adadeltago_eps', 1e-6)}
        if opt_type == 'adamgo':
            adaptive = {'beta2': cfg.get('adamgo_beta2', .999), 'gamma': cfg.get('adamgo_gamma', 1.),
                        'delta': cfg.get('adamgo_delta', 1e-8), 'min_step': cfg.get('adamgo_min_step', 0.)}
        optimizer = optimizer_cls(
            jit_muon_param_groups(model), lr=lr,
            momentum=cfg.get('muon_momentum', 0.95),
            weight_decay=cfg.get('muon_weight_decay', 0. if opt_type in {'adago', 'adamgo', 'rmsgo', 'adadeltago'} else 0.1),
            newton_schulz_iter=cfg.get('muon_ns_steps', 5),
            adam_betas=tuple(cfg.get('muon_adam_betas', (0.9, 0.95 if opt_type in {'adago', 'adamgo', 'rmsgo', 'adadeltago'} else 0.999))),
            adam_eps=cfg.get('muon_adam_eps', 1e-8),
            foreach=cfg.get('muon_foreach', True),
            ns_bfloat16=cfg.get('muon_ns_bfloat16', False),
            rank=cfg.get('muon_rank', 0),
            cautious=cfg.get('cautious', False),
            muon_all=cfg.get('muon_all', False),
            muon_all_reshape=cfg.get('muon_all_reshape', False),
            orthogonalization_backend=cfg.get('muon_backend', 'polar_express'),
            **adaptive,
        )
        apply_muon_group_lrs(optimizer, cfg)
        return optimizer
    if opt_type == 'adam':
        return torch.optim.Adam(trainable, lr=lr, betas=(0.0, 0.99))
    if opt_type == 'adan':
        return Adan(trainable, lr=lr)
    if opt_type == 'sgd':
        return torch.optim.SGD(trainable, lr=lr, momentum=0.9)
    if opt_type == 'rmsprop':
        return torch.optim.RMSprop(trainable, lr=lr, alpha=0.99, eps=1e-8, momentum=0.0)
    if opt_type == 'adagrad':
        return torch.optim.Adagrad(trainable, lr=lr)
    if opt_type == 'adadelta':
        return torch.optim.Adadelta(trainable, lr=lr)
    if opt_type == 'prodigy':
        return Prodigy(
            trainable, lr=lr,
            betas=(0.9, 0.99),
            weight_decay=0.01,
            decouple=True,
            use_bias_correction=True,
            safeguard_warmup=True,
            d_coef=0.5,
            growth_rate=1.02,
            d_upper_limit=1e-2,
            max_dlr=1e-2,
            update_clip_norm=1.0,
        )
    if opt_type == 'paper_adamhd':
        return PaperAdamHD(
            trainable, lr=lr,
            hyper_lr=cfg.get('hyper_lr', 1e-10),
            min_lr=cfg.get('hd_min_lr', 1e-5),
            max_lr=cfg.get('hd_max_lr', 1e-2),
        )
    if opt_type == 'nsgda':
        return NSGDA(trainable, lr=lr)
    if opt_type == 'layerwise_nsgda':
        return NSGDA(
            layerwise_param_groups(model),
            lr=lr,
            momentum=cfg.get('layerwise_nsgda_momentum', 0.9),
            cautious=cfg.get('layerwise_nsgda_cautious', True),
        )
    if opt_type == 'ada_nsgda':
        return AdaNSGDA(trainable, lr=lr, betas=(0.0, 0.99))
    if opt_type == 'clion':
        return CLion(
            trainable,
            lr=lr,
            betas=(0.95, 0.98),
            weight_decay=0.0,
            rescale_mask=cfg.get('clion_rescale_mask', False),
        )
    if opt_type == 'modern_clion':
        return ModernCLion(
            trainable,
            lr=lr,
            betas=(0.9, 0.99),
            weight_decay=cfg.get('modern_clion_weight_decay', 0.0),
            nu=cfg.get('modern_clion_nu', MODERN_CLION_DEFAULT_NU),
        )
    if opt_type == 'lamb':
        return Lamb(trainable, lr=lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0)
    if opt_type == 'equalized_adamw':
        return EqualizedAdamW(trainable, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    if opt_type == 'radam_schedulefree':
        # Warmup and averaging are internal; the training loop skips the LR scheduler.
        return RAdamScheduleFree(trainable, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=cfg.get('schedulefree_weight_decay', 0.0))
    raise ValueError(f"Unknown optimizer type: {opt_type}")


def apply_config_defaults(cfg):
    defaults = {
        'grad_accum_steps': 1, 'num_workers': 4,
        'validation_path': '', 'validation_percent': 0.0, 'validation_seed': 42,
        'validation_every': 250, 'validation_batch_size': cfg.get('batch_size', 4),
        'validation_max_batches': 0, 'save_best': True, 'best_training_loss': False,
        'reset_data_stream': False, 'finetune_policy': 'all', 'finetune_last_blocks': 1,
        'trainable_modules': [],
        # Old runs keep the (1+n)/(10+n) EMA ramp; new runs are prompted.
        'ema_warmup': 'legacy', 'ema_warmup_steps': 10000,
        'compile_mode': 'off',
        'use_adaln': False, 'use_ema': False, 'use_flip': False, 'use_vflip': False, 'use_rot90': False,
        'use_2d_pos_emb': False, 'use_conv_mlp': False, 'use_grn': False,
        'self_cond': cfg.get('model_type') == 'rin',
        'bottleneck_dim': None, 'fmap_max': 512, 'bottleneck_res': 8,
        'conv_stem': 0, 'stem_initial': 32, 'stem_max': 256, 'stem_activation': 4,
        'stem_glu_scaling': 0, 'stem_skips': False, 'stem_conditioning': 0, 'stem_grn': False,
        'overlap_h': 0, 'overlap_w': 0, 'axial': False,
        'use_amp': True, 'grad_clip': 1.0, 'use_spike_guard': True, 'use_grad_ckpt': False,
        'amp_dtype': 'fp16', 'full_bf16': False,
        'warmup_steps': 1000, 'seed': 42, 'sample_every': 250,
        'save_every': 5000, 'num_sample_images': 4,
        'pred_mode': 'x', 'loss_mode': 'v', 'x_clip': 'none',
        't_mu': -0.8, 't_sigma': 0.8, 'lr_schedule': 'constant',
        'noise_schedule': 'linear', 'rin_num_latents': 256,
        'rin_latent_dim': 2 * cfg['dim'], 'rin_layers_per_block': 4,
        'use_qk_norm': False, 'use_final_adaln': False, 'time_scale': 1.0,
        'initial_grid_size': 4,
        'hier_global_mixer': 'mlpmixer', 'hier_global_heads': 4,
        'bottleneck_act': 'mish', 'dropout': 0.0, 'optimizer_type': 'adan',
        # Old JiT/FullAttention/MixerAttn/HierMLP-JiT checkpoints normalize twice before attention.
        'attn_double_norm': True,
        # 1 = legacy HAT/MaxViT/Swin/Vim blocks; new runs use 2 (reference-faithful).
        'arch_version': 1, 'hat_group_depth': 2,
        # Muon-family AdamW-fallback LR = lr * ratio; 1.0 is the legacy shared LR.
        'muon_adam_lr_ratio': 1.0,
        'fcdm_mlp_ratio': 3.0,
        'hyper_lr': 1e-10, 'hd_min_lr': 1e-5, 'hd_max_lr': 1e-2,
        'modern_clion_nu': MODERN_CLION_DEFAULT_NU, 'modern_clion_weight_decay': 0.0,
        'clion_rescale_mask': False,
        'layerwise_nsgda_momentum': 0.9, 'layerwise_nsgda_cautious': True,
        'self_cond_prob': 0.9 if cfg.get('model_type') == 'rin' else 0.5,
        'conditioning_mode': 'unconditional', 'class_names': [], 'pix2pix_direction': 'a_to_b',
        'class_sampling': 'uniform',
        'pix2pix_source_mode': 'folders',
        'cond_residual': False,
        'synthetic_params': {}, 'crop_output_policy': 'max',
        # Old class checkpoints retain their original embedding table.
        'class_dropout_prob': 0.0, 'guidance_scale': 1.0,
        # Finite-spike protection. These conservative defaults only act
        # after a baseline of accepted updates has been observed.
        'spike_guard_warmup': 100, 'spike_loss_factor': 6.0,
        'spike_grad_factor': 8.0, 'spike_ema_decay': 0.98,
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    for axis, overlap_key in (('height', 'overlap_h'), ('width', 'overlap_w')):
        patch = cfg.get('p_' + axis, 1)
        grid = (cfg[axis] - patch) // max(1, patch - cfg[overlap_key]) + 1
        cfg.setdefault('stem_' + axis, max(1, grid))
    cfg.setdefault('preview_seed', cfg.get('seed', 42) or 42)
    cfg.setdefault('preview_batch_size', cfg.get('batch_size', 4))
    # Old checkpoints stored only the model's image size. Treat that as
    # both the source resize size and the inactive crop size.
    cfg.setdefault('resize_width', cfg['width'])
    cfg.setdefault('resize_height', cfg['height'])
    cfg.setdefault('crop_width', cfg['width'])
    cfg.setdefault('crop_height', cfg['height'])
    cfg.setdefault('crop_width_min', cfg['width'])
    cfg.setdefault('crop_width_max', cfg['width'])
    cfg.setdefault('crop_height_min', cfg['height'])
    cfg.setdefault('crop_height_max', cfg['height'])
    # Legacy HierMLP used one grid for input tokenization and refinement.
    cfg.setdefault('hier_input_grid_size', cfg['initial_grid_size'])
    cfg.setdefault('hier_output_grid_size', cfg['initial_grid_size'])
    cfg.setdefault('hier_global_dim', cfg['dim'])
    cfg.setdefault('hier_global_depth', cfg['depth'])
    if (cfg.get('optimizer_type') == 'modern_clion'
            and cfg.get('modern_clion_nu') == OLD_MODERN_CLION_DEFAULT_NU):
        cfg['modern_clion_nu'] = MODERN_CLION_DEFAULT_NU
        print(f"│  Updated ModernCLion ν from {OLD_MODERN_CLION_DEFAULT_NU:g} to {MODERN_CLION_DEFAULT_NU:g}")
    if cfg.get('optimizer_type') == 'paper_adamhd' and cfg.get('hyper_lr') == 1e-7:
        cfg['hyper_lr'] = 1e-10
        print("│  Updated PaperAdamHD beta from legacy 1e-7 to 1e-10")
    if cfg.get('amp_dtype') not in {'fp16', 'bf16'}:
        cfg['amp_dtype'] = 'fp16'



class ConfigError(ValueError):
    pass


def validate_fresh_run(cfg):
    """Checks for runs that start from new weights; saved checkpoints stay usable."""
    if (cfg.get('conditioning_mode') == 'pix2pix' and cfg.get('cond_residual', False)
            and cfg.get('pred_mode') == 'eps'):
        raise ConfigError("cond_residual adds the source image to the network output, which is a noise "
                          "estimate under pred_mode='eps'; use pred_mode 'x' or 'v', or disable cond_residual")


def validate_config(cfg):
    """Check runtime ranges and model/data geometry before allocating weights."""
    def number(key, minimum=0, maximum=None, integer=False):
        value = cfg.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or (integer and not isinstance(value, int))
                or value < minimum or (maximum is not None and value > maximum)):
            raise ConfigError(f'{key} must be a finite {"integer" if integer else "number"} '
                              f'>= {minimum}' + (f' and <= {maximum}' if maximum is not None else ''))
        return value

    for key in ('width', 'height', 'dim', 'batch_size', 'sample_every', 'save_every',
                'num_sample_images', 'preview_batch_size', 'sampling_steps',
                'grad_accum_steps', 'validation_every', 'validation_batch_size', 'finetune_last_blocks'):
        number(key, 1, integer=True)
    for key in ('steps', 'warmup_steps', 'seed', 'preview_seed', 'spike_guard_warmup', 'ema_warmup_steps',
                'validation_seed', 'validation_max_batches', 'num_workers'):
        number(key, 0, integer=True)
    number('validation_percent', 0, 99)
    if not isinstance(cfg['validation_path'], str):
        raise ConfigError('validation_path must be a string')
    if cfg['validation_path'] and cfg['validation_percent']:
        raise ConfigError('Choose a validation folder OR a percentage, not both')
    if cfg['finetune_policy'] not in {'all', 'last', 'modules'}:
        raise ConfigError('finetune_policy must be all, last, or modules')
    if (not isinstance(cfg['trainable_modules'], list)
            or any(not isinstance(p, str) or not p for p in cfg['trainable_modules'])):
        raise ConfigError('trainable_modules must be a list of nonempty module names/patterns')
    for key, value in cfg.items():
        if (key.startswith('use_') or key in {'save_best', 'best_training_loss', 'reset_data_stream',
                'full_bf16', 'self_cond', 'axial', 'stem_skips', 'cond_residual', 'cautious', 'attn_double_norm',
                'muon_all', 'muon_all_reshape', 'muon_foreach', 'muon_ns_bfloat16'}):
            if not isinstance(value, bool):
                raise ConfigError(f'{key} must be a JSON boolean (true/false), not {value!r}')
    for key in ('lr', 'grad_clip', 't_sigma', 'guidance_scale'):
        number(key)
    number('t_mu', -float('inf'))
    number('arch_version', 1, 2, integer=True)
    number('hat_group_depth', 1, integer=True)
    number('time_scale', 1e-12)
    for key in ('dropout', 'self_cond_prob', 'class_dropout_prob', 'spike_ema_decay'):
        number(key, 0, 1)
    for key in ('spike_loss_factor', 'spike_grad_factor'):
        number(key, 1e-12)
    for key, values in (('channels', (1, 3, 4)), ('pred_mode', ('x', 'eps', 'v')),
                        ('loss_mode', ('x', 'eps', 'v')), ('x_clip', ('none', 'static', 'dynamic')),
                        ('lr_schedule', ('constant', 'cosine')),
                        ('ema_warmup', EMA_WARMUP_MODES),
                        ('compile_mode', COMPILE_MODES),
                        ('noise_schedule', ('linear', 'rin_sigmoid')),
                        ('conditioning_mode', ('unconditional', 'class', 'pix2pix')),
                        ('class_sampling', ('uniform', 'natural')),
                        ('amp_dtype', ('fp16', 'bf16')),
                        ('optimizer_type', tuple(v[0] for v in OPTIMIZER_TYPES.values())),
                        ('model_type', tuple(v[0] for v in MODEL_TYPES.values()))):
        if cfg.get(key) not in values:
            raise ConfigError(f'{key} must be one of {values}')
    if cfg['full_bf16'] and cfg['use_amp']:
        raise ConfigError('Full BF16 and AMP cannot both be enabled')
    if cfg['optimizer_type'] in {'muon', 'adamuon', 'normuon', 'adago', 'adamgo', 'rmsgo', 'adadeltago'}:
        if not isinstance(cfg.get('muon_all', False), bool):
            raise ConfigError('muon_all must be boolean')
        if not isinstance(cfg.get('muon_all_reshape', False), bool):
            raise ConfigError('muon_all_reshape must be boolean')
        if cfg.get('muon_backend', 'polar_express') not in ('newton_schulz', 'polar_express'):
            raise ConfigError('muon_backend must be newton_schulz or polar_express')
        ratio = cfg.get('muon_adam_lr_ratio', 1.0)
        if (isinstance(ratio, bool) or not isinstance(ratio, (int, float))
                or not math.isfinite(ratio) or ratio <= 0):
            raise ConfigError('muon_adam_lr_ratio must be a finite positive number')
    if cfg['conditioning_mode'] == 'class' and not cfg['class_names']:
        raise ConfigError('Class conditioning requires at least one class')
    for axis in ('height', 'width'):
        value = cfg['resize_' + axis]
        if value != -1:
            number('resize_' + axis, 1, integer=True)
        lo = number('crop_' + axis + '_min', 1, integer=True)
        hi = number('crop_' + axis + '_max', 1, integer=True)
        if lo > hi:
            raise ConfigError(f'Crop {axis} minimum exceeds maximum')
    if cfg['crop_width_min'] * cfg['crop_height_max'] != cfg['crop_width_max'] * cfg['crop_height_min']:
        raise ConfigError('Crop ranges must preserve aspect ratio')
    h, w, dim = cfg['height'], cfg['width'], cfg['dim']
    if dim < 4 or dim % 2:
        raise ConfigError('Model width must be even and at least 4 for time embeddings')
    model_type = cfg['model_type']
    stem_mode = number('conv_stem', 0, 3, integer=True)
    number('stem_glu_scaling', 0, 1, integer=True)
    number('stem_conditioning', 0, 5, integer=True)
    if not isinstance(cfg['stem_grn'], bool):
        raise ConfigError('stem_grn must be boolean')
    if cfg['stem_grn'] and not stem_mode:
        raise ConfigError('stem_grn requires a convolutional stem')
    try:
        validate_stem_conditioning(cfg['stem_conditioning'], stem_mode, cfg['stem_skips'])
        validate_stem_skips(cfg['stem_skips'], stem_mode,
                            cfg['hier_input_grid_size'] if model_type in IS_HIERARCHICAL else None,
                            cfg['hier_output_grid_size'] if model_type in IS_HIERARCHICAL else None)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if stem_mode:
        if model_type in IS_CONV or model_type == 'fcdm_unet':
            raise ConfigError('Conv-stem is available for patch models only')
        initial = number('stem_initial', 2, integer=True)
        maximum = number('stem_max', initial, integer=True)
        if number('stem_activation', 0, 10, integer=True) not in STEM_ACTIVATIONS:
            raise ConfigError(f'Stem activation must be one of {tuple(STEM_ACTIVATIONS)}')
        if model_type in IS_HIERARCHICAL:
            grid = (number('hier_input_grid_size', 1, integer=True),) * 2
        else:
            grid = (number('stem_height', 1, integer=True), number('stem_width', 1, integer=True))
        try:
            conv_stem_geometry((h, w), grid, initial, maximum)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if model_type == 'maxvit' and any(g % 8 for g in grid):
            raise ConfigError('MaxViT stem grid height and width must be multiples of 8')
    if model_type in {'fcdm_unet', 'fcdm_isotropic'}:
        number('fcdm_mlp_ratio', 1)
    if model_type == 'fcdm_unet':
        number('depth', 1, integer=True)
        if h % 4 or w % 4:
            raise ConfigError('FCDM-UNet image dimensions must be divisible by 4')
        if cfg['self_cond']:
            raise ConfigError('FCDM-UNet does not support pixel self-conditioning')
    elif model_type in IS_CONV:
        bottleneck = number('bottleneck_res', 1, integer=True)
        number('fmap_max', 1, integer=True)
        if bottleneck > h or dim % 8:
            raise ConfigError('ConvNet needs bottleneck_res <= image height and dim divisible by 8')
        levels = int(math.log2(h) - math.log2(bottleneck))
        divisor = 2 ** levels
        if h % divisor or w % divisor or min(h, w) < divisor:
            raise ConfigError(f'ConvNet image height and width must be multiples of {divisor}')
    elif model_type in IS_HIERARCHICAL:
        for key in ('depth', 'fmap_max', 'hier_input_grid_size', 'hier_output_grid_size',
                    'hier_global_dim', 'hier_global_depth', 'hier_global_heads'):
            number(key, 1, integer=True)
        source, root = cfg['hier_input_grid_size'], cfg['hier_output_grid_size']
        scale = h // root
        if source % root or h % source or w % source or h != w or h % root or scale < 1 or scale & (scale - 1):
            raise ConfigError('HierMLP needs nested grids and equal power-of-two refinements to the image size')
        if cfg['hier_global_mixer'] not in ('jit', 'mlpmixer', 'gmlp', 'convnext', 'vip'):
            raise ConfigError('Unknown HierMLP global mixer')
        # The JiT global mixer projects to heads * 64 independently of global_dim.
    else:
        number('depth', 1, integer=True)
        heads = number('heads', 1, integer=True)
        for axis, overlap_key in (('height', 'overlap_h'), ('width', 'overlap_w')):
            if stem_mode == 3:
                continue  # Both stems replace patch extraction and folding entirely.
            patch = number('p_' + axis, 1, integer=True)
            overlap = number(overlap_key, 0, integer=True)
            if patch > cfg[axis] or overlap >= patch or (cfg[axis] - patch) % (patch - overlap):
                raise ConfigError(f'Patch {axis} and overlap must tile image {axis} exactly')
            if stem_mode and cfg['stem_' + axis] != (cfg[axis] - patch) // (patch - overlap) + 1:
                raise ConfigError(f'One-sided conv-stem {axis} must match the ordinary patch grid')
        # These Attention users project to heads * 64; they do not split dim.
        split_width_models = (NEEDS_HEADS | HYPER_HEADS) - {'jit', 'fullattn', 'mixer_attn'}
        if model_type in split_width_models and dim % heads:
            raise ConfigError('Model width must be divisible by head count')
        if cfg['use_2d_pos_emb'] and dim % 4:
            raise ConfigError('2D positional embedding width must be divisible by 4')
        if model_type == 'rin':
            number('rin_num_latents', 3 if cfg['conditioning_mode'] == 'class' else 2, integer=True)
            latent_dim = number('rin_latent_dim', 16, integer=True)
            number('rin_layers_per_block', 1, integer=True)
            if latent_dim % 8 or latent_dim % heads:
                raise ConfigError('RIN latent width must be divisible by 8 and by head count')
        if cfg['bottleneck_dim'] is not None:
            number('bottleneck_dim', 1, integer=True)


def build_model(cfg, device):
    validate_config(cfg)
    if cfg['model_type'] == 'fcdm_unet':
        model = FCDMUNet(
            channels=cfg['channels'], dim=cfg['dim'], depth=cfg['depth'],
            ratio=cfg['fcdm_mlp_ratio'], time_scale=cfg['time_scale'],
            use_gradient_checkpointing=cfg['use_grad_ckpt'],
            **conditioning_model_kwargs(cfg),
        ).to(device)
    elif cfg['model_type'] in IS_CONV:
        model = ConvNetModel(
            img_size=(cfg['height'], cfg['width']),
            channels=cfg['channels'], dim=cfg['dim'],
            fmap_max=cfg['fmap_max'], bottleneck_res=cfg['bottleneck_res'],
            model_type=cfg['model_type'], time_scale=cfg['time_scale'],
            arch_version=cfg['arch_version'],
            **conditioning_model_kwargs(cfg),
        ).to(device)
    elif cfg['model_type'] in IS_HIERARCHICAL:
        model = HierMLPModel(
            img_size=(cfg['height'], cfg['width']), channels=cfg['channels'],
            dim=cfg['dim'], fmap_max=cfg['fmap_max'], layer_count=cfg['depth'],
            initial_grid_size=cfg['initial_grid_size'],
            input_grid_size=cfg['hier_input_grid_size'],
            output_grid_size=cfg['hier_output_grid_size'],
            global_mixer=cfg['hier_global_mixer'], global_heads=cfg['hier_global_heads'],
            global_dim=cfg['hier_global_dim'], global_depth=cfg['hier_global_depth'],
            conv_stem=cfg['conv_stem'], stem_initial=cfg['stem_initial'], stem_max=cfg['stem_max'],
            stem_activation=cfg['stem_activation'],
            stem_glu_scaling=cfg['stem_glu_scaling'], stem_skips=cfg['stem_skips'],
            stem_conditioning=cfg['stem_conditioning'], stem_grn=cfg['stem_grn'],
            time_scale=cfg['time_scale'], attn_double_norm=cfg['attn_double_norm'],
            **conditioning_model_kwargs(cfg),
        ).to(device)
    else:
        model = JiTModel(
            img_size=(cfg['height'], cfg['width']),
            patch_size=(cfg['p_height'], cfg['p_width']),
            channels=cfg['channels'], dim=cfg['dim'],
            depth=cfg['depth'], heads=cfg['heads'],
            model_type=cfg['model_type'],
            self_cond=cfg['self_cond'],
            use_adaln=cfg['use_adaln'],
            use_2d_pos_emb=cfg['use_2d_pos_emb'],
            use_conv_mlp=cfg['use_conv_mlp'],
            bottleneck_dim=cfg['bottleneck_dim'],
            overlap_h=cfg['overlap_h'], overlap_w=cfg['overlap_w'],
            axial=cfg['axial'],
            use_gradient_checkpointing=cfg['use_grad_ckpt'],
            use_qk_norm=cfg['use_qk_norm'],
            use_final_adaln=cfg['use_final_adaln'],
            use_grn=cfg['use_grn'],
            time_scale=cfg['time_scale'],
            bottleneck_act=cfg['bottleneck_act'],
            attn_double_norm=cfg['attn_double_norm'],
            arch_version=cfg['arch_version'], hat_group_depth=cfg['hat_group_depth'],
            dropout=cfg['dropout'],
            rin_num_latents=cfg['rin_num_latents'],
            rin_latent_dim=cfg['rin_latent_dim'], rin_layers_per_block=cfg['rin_layers_per_block'],
            fcdm_mlp_ratio=cfg['fcdm_mlp_ratio'],
            conv_stem=cfg['conv_stem'], stem_initial=cfg['stem_initial'], stem_max=cfg['stem_max'],
            stem_activation=cfg['stem_activation'],
            stem_glu_scaling=cfg['stem_glu_scaling'], stem_skips=cfg['stem_skips'],
            stem_conditioning=cfg['stem_conditioning'], stem_grn=cfg['stem_grn'],
            stem_grid=(cfg['stem_height'], cfg['stem_width']),
            **conditioning_model_kwargs(cfg),
        ).to(device)

    if cfg['full_bf16']:
        model = model.to(dtype=torch.bfloat16)

    return model


def main():
    global interrupted
    interrupted = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = SAVE_DIR
    model_path = os.path.join(checkpoint_dir, "model.pt")
    config_path = os.path.join(checkpoint_dir, "config.json")
    previous_config = (config_path if os.path.exists(config_path)
                       else os.path.join(checkpoint_dir, 'config.pt'))
    ui_header("JiT Diffusion Studio", "Image generation training and sampling")
    ui_key_value("Device", device)
    ui_key_value("TF32", torch.backends.cuda.matmul.allow_tf32)
    ui_key_value("Output", SAVE_DIR)
    ui_rule()
    mode_in = input("│  Mode [0/train, 1/sample, 2/retry, 3/continue, 4/finetune]: ").lower().strip()

    if mode_in not in ['t', '0', 'train', 's', '1', 'sample', '2', 'retry',
                       '3', 'continue', '4', 'finetune']:
        return
    output_dir = make_unique_output_dir(checkpoint_dir, 'run')

    if mode_in not in ['s', '1', 'sample']:
        cfg = {}
        loaded = False
        prepared = False
        fine_checkpoint = None
        ckpt = {}
        ui_section("Training setup")
        continuing = mode_in in ['3', 'continue']
        if mode_in in ['t', '0', 'train', '2', 'retry']:
            prepared_path = get_input('Prepared config path (blank=interactive)', '')
            if prepared_path:
                cfg = load_config(prepared_path)
                prepared = True
            elif mode_in in ['2', 'retry']:
                cfg = load_config(previous_config)
                apply_config_defaults(cfg)
                cfg = edit_config(cfg)
                prepared = True
            else:
                continuing = get_input('Continue training?', False, bool)
        if continuing:
            if not os.path.exists(model_path):
                raise FileNotFoundError('Cannot resume: model.pt is missing')
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            cfg = copy.deepcopy(ckpt.get('config') or load_config(previous_config))
            loaded = True
            legacy_class_sampling = (cfg.get('conditioning_mode') == 'class'
                                     and 'class_sampling' not in cfg)
            old_dataset = cfg['dataset_path']
            cfg['dataset_path'] = get_input('Dataset location', old_dataset)
            cfg['reset_data_stream'] = (os.path.realpath(cfg['dataset_path']) != os.path.realpath(old_dataset)
                                        or get_input('Reset data stream (changed files/transforms)?', False, bool))
            if legacy_class_sampling:
                print('│  Enabling uniform class sampling; restarting the data stream with existing weights/optimizer.')
                cfg['reset_data_stream'] = True
        elif mode_in in ['4', 'finetune']:
            source_path = get_input('Source checkpoint', model_path)
            fine_checkpoint = torch.load(source_path, map_location='cpu', weights_only=False)
            cfg = copy.deepcopy(fine_checkpoint.get('config') or load_config(previous_config))
            apply_config_defaults(cfg)
            cfg['dataset_path'] = get_input('Fine-tuning dataset', cfg['dataset_path'])
            cfg['lr'] = get_number('Fine-tuning learning rate', cfg['lr'] * .1, float, 0)
            cfg['steps'] = get_number('Fine-tuning update count', cfg['steps'], int, 1)
            cfg['warmup_steps'] = get_number('Fine-tuning warmup updates', min(100, cfg['steps']), int, 0)
            cfg['finetune_policy'] = get_input('Trainable parameters [all/last/modules]', 'all')
            if cfg['finetune_policy'] == 'last':
                cfg['finetune_last_blocks'] = get_number('Last blocks to train (plus output layers)', 1, int, 1)
            elif cfg['finetune_policy'] == 'modules':
                print('│  Module names will be listed before selecting trainable modules.')
            cfg['finetune_weights'] = get_input('Source weights [ema/model]', 'ema' if 'ema' in fine_checkpoint else 'model')
            if cfg['finetune_weights'] not in {'ema', 'model'} or cfg['finetune_weights'] not in fine_checkpoint:
                raise ConfigError('Requested source weights are unavailable')
            prompt_training_features(cfg)
            prepared = True

        if not loaded and not prepared:
            ui_section("Dataset")
            cfg['dataset_path'] = get_input("Dataset location", "./data")
            conditioning_in = get_input("Dataset format [0=unconditional, 1=class folders, 2=Pix2Pix]", "0", str).lower()
            cfg['conditioning_mode'] = {
                '0': 'unconditional', 'unconditional': 'unconditional',
                '1': 'class', 'class': 'class', 'conditional': 'class',
                '2': 'pix2pix', 'pix2pix': 'pix2pix',
            }.get(conditioning_in, 'unconditional')
            cfg['class_names'] = []
            cfg['pix2pix_direction'] = 'a_to_b'
            cfg['pix2pix_source_mode'] = 'folders'
            if cfg['conditioning_mode'] == 'class':
                cfg['class_names'] = discover_class_names(cfg['dataset_path'])
                if not cfg['class_names']:
                    print("│  ✕ No class folders containing images found. Exiting.")
                    return
                print(f"│  Found classes: {', '.join(cfg['class_names'])}")
                cfg['class_dropout_prob'] = get_input("Class dropout for guidance (0=off)", 0.1, float)
                cfg['guidance_scale'] = (get_input("Class guidance strength (1=unguided)", 3.0, float)
                                         if cfg['class_dropout_prob'] > 0 else 1.0)
            elif cfg['conditioning_mode'] == 'pix2pix':
                cfg['cond_residual'] = get_input("Add the conditioning image as an output residual?", False, bool)
                source_in = get_input(
                    "Pix2Pix source IDs (comma-combine: 0=folders, 1=combined, 2=colorify, 3=superres, 4=deblur, 5=restoration, 6=inpaint, 7=outpaint, 8=affine, 9=noise, 10=hue, 11=gray, 12=sharpen, 13=quantize, 14=glitch, 15=colorize, 16=film, 17=invert, 18=solarize, 19=autocanny)",
                    "0", str,
                ).lower()
                try:
                    cfg['pix2pix_source_mode'] = parse_synthetic_source(source_in)
                except ValueError as exc:
                    print(f"│  ✕ {exc}. Exiting.")
                    return
                cfg['synthetic_params'] = (prompt_synthetic_params(cfg['pix2pix_source_mode'])
                                           if cfg['pix2pix_source_mode'] not in {'folders', 'combined'} else {})
                if cfg['pix2pix_source_mode'] == 'folders':
                    if not (os.path.isdir(os.path.join(cfg['dataset_path'], 'A'))
                            and os.path.isdir(os.path.join(cfg['dataset_path'], 'B'))):
                        print("│  ✕ Folder Pix2Pix datasets need A and B folders. Exiting.")
                        return
                elif not find_image_paths(cfg['dataset_path']):
                    print("│  ✕ No images found for this Pix2Pix source mode. Exiting.")
                    return
                if cfg['pix2pix_source_mode'] in {'folders', 'combined'}:
                    direction_in = get_input("Pix2Pix direction [0=A→B, 1=B→A]", "0", str).lower()
                    cfg['pix2pix_direction'] = {
                        '0': 'a_to_b', 'a2b': 'a_to_b', 'a_to_b': 'a_to_b',
                        '1': 'b_to_a', 'b2a': 'b_to_a', 'b_to_a': 'b_to_a',
                    }.get(direction_in, 'a_to_b')
            cfg['channels'] = get_input("Channel count", 3, int)
            cfg['resize_width'] = get_input("Source image width (-1=keep native width)", 64, int)
            cfg['resize_height'] = get_input("Source image height (-1=keep native height)", 64, int)
            crop_required = -1 in {cfg['resize_width'], cfg['resize_height']}
            crop_w_default = None if crop_required else str(cfg['resize_width'])
            crop_h_default = None if crop_required else str(cfg['resize_height'])
            try:
                crop_w = parse_size_range(get_input("Crop width [N, N-M, N,M, or N M]", crop_w_default, str), 'crop width')
                crop_h = parse_size_range(get_input("Crop height [N, N-M, N,M, or N M]", crop_h_default, str), 'crop height')
            except ValueError as exc:
                print(f"│  ✕ {exc}; a crop is required for native-size images. Exiting.")
                return
            if crop_w[0] * crop_h[1] != crop_w[1] * crop_h[0]:
                print("│  ✕ Crop ranges must preserve one aspect ratio. Exiting.")
                return
            cfg['crop_width_min'], cfg['crop_width_max'] = crop_w
            cfg['crop_height_min'], cfg['crop_height_max'] = crop_h
            varying_crop = crop_w[0] != crop_w[1] or crop_h[0] != crop_h[1]
            crop_output = get_input("Variable crop output size [max/min]", "max", str).lower() if varying_crop else "max"
            use_min_crop = crop_output == 'min'
            cfg['crop_output_policy'] = 'min' if use_min_crop else 'max'
            cfg['width'] = crop_w[0] if use_min_crop else crop_w[1]
            cfg['height'] = crop_h[0] if use_min_crop else crop_h[1]
            cfg['crop_width'], cfg['crop_height'] = cfg['width'], cfg['height']
            print(f"│  Effective model/crop resolution: {cfg['width']}×{cfg['height']}")
            if (cfg['conditioning_mode'] == 'pix2pix'
                    and all(item in SyntheticPix2PixDataset.MODES for item in cfg['pix2pix_source_mode'].split(','))
                    and cfg['channels'] != 3):
                print("│  Synthetic Pix2Pix modes use RGB effects; forcing channel count to 3.")
                cfg['channels'] = 3

            ui_section("Model selection")
            ui_menu([(k, desc) for k, (_, desc) in sorted(MODEL_TYPES.items(), key=lambda x: int(x[0]))])

            m_type_in = get_input("Model type", "1", str)
            # Match by number or name
            if m_type_in in MODEL_TYPES:
                cfg['model_type'] = MODEL_TYPES[m_type_in][0]
            else:
                # Try matching by name
                matched = [v[0] for v in MODEL_TYPES.values() if v[0] == m_type_in]
                cfg['model_type'] = matched[0] if matched else "jit"

            is_conv = cfg['model_type'] in IS_CONV
            is_hierarchical = cfg['model_type'] in IS_HIERARCHICAL

            if cfg['model_type'] == 'fcdm_unet':
                cfg['dim'] = get_number('FCDM base channels (even)', 32, int, 4)
                cfg['depth'] = get_number('FCDM base blocks (L,2L,4L,2L,L)', 2, int, 1)
                cfg['fcdm_mlp_ratio'] = get_number('FCDM channel expansion ratio', 3.0, float, 1)
                cfg.update(conv_stem=0, stem_conditioning=0, stem_skips=False, stem_grn=False,
                           use_adaln=True, self_cond=False, self_cond_prob=0.,
                           p_width=1, p_height=1, heads=1, dropout=0.)
                print('│  FCDM-UNet uses its own conditioned encoder/decoder at full, half and quarter resolution.')
            elif not is_conv and not is_hierarchical:
                cfg['conv_stem'] = get_number("Conv-stem [0=off, 1=input only, 2=output only, 3=both]", 0, int, 0, 3)
                if cfg['conv_stem'] != 3:
                    cfg['p_width'] = get_number("Patch width", 8, int, 1)
                    cfg['p_height'] = get_number("Patch height", 8, int, 1)
                    cfg['overlap_w'] = get_input("Patch Width Overlap (pixels)", 0, int)
                    cfg['overlap_h'] = get_input("Patch Height Overlap (pixels)", 0, int)
                else:
                    cfg.update(p_width=1, p_height=1, overlap_w=0, overlap_h=0)
                if cfg['conv_stem']:
                    cfg['stem_initial'] = get_number("Stem initial filters", 32, int, 2)
                    cfg['stem_max'] = get_number("Stem max features", max(256, cfg['stem_initial']), int, cfg['stem_initial'])
                    cfg['stem_activation'] = prompt_stem_activation()
                    prompt_stem_extras(cfg)
                    if cfg['conv_stem'] != 3:
                        print("│  Stem bottleneck must match the ordinary patch grid on the other side.")
                    for axis, overlap_key in (('width', 'overlap_w'), ('height', 'overlap_h')):
                        patch = cfg['p_' + axis]
                        default_grid = ((cfg[axis] - patch) // max(1, patch - cfg[overlap_key]) + 1
                                        if cfg['conv_stem'] != 3 else min(8, cfg[axis]))
                        cfg['stem_' + axis] = get_number(f"Stem bottleneck grid {axis}", max(1, default_grid), int, 1, cfg[axis])

                raw_dim = get_input("Model width (dim)", 256, int)
                cfg['dim'] = make_divisible(raw_dim, 8)
                if cfg['dim'] != raw_dim:
                    print(f"│  Rounded width to {cfg['dim']}")

                cfg['depth'] = get_number(
                    "RIN routing blocks" if cfg['model_type'] == 'rin' else "Model depth",
                    6 if cfg['model_type'] == 'rin' else 4, int, 1)

                cfg['heads'] = 1
                if cfg['model_type'] in NEEDS_HEADS:
                    cfg['heads'] = get_number("Attention head count", 4, int, 1)
                elif cfg['model_type'] in HYPER_HEADS:
                    cfg['heads'] = get_number("HyperMixer head count", 1, int, 1)

                cfg['axial'] = False
                if cfg['model_type'] in AXIAL_MODELS:
                    cfg['axial'] = get_input("Use Axial Mixing?", "0", bool)

                is_rin = cfg['model_type'] == 'rin'
                is_fcdm = cfg['model_type'] == 'fcdm_isotropic'
                cfg['use_adaln'] = True if is_fcdm else False if is_rin else get_input("Use AdaLN-Zero?", "1", bool)
                cfg['use_2d_pos_emb'] = False if is_rin or is_fcdm else get_input("Use 2D Sinusoidal Pos Emb?", "1", bool)
                # RIN uses GELU feed-forward layers and learned positions.
                cfg['use_conv_mlp'] = False if is_rin or is_fcdm else get_input("Use Conv-MLP (off = paper SwiGLU)?", "0", bool)
                cfg['self_cond'] = get_input("Latent self-conditioning?" if is_rin else "Self-conditioning?",
                                             "1" if is_rin else "0", bool)
                self_cond_prob = (get_input(
                    "Self-conditioning probability (extra denoising pass)", 0.9 if is_rin else 0.5, float)
                    if cfg['self_cond'] else 0.0)
                cfg['self_cond_prob'] = min(max(self_cond_prob, 0.0), 1.0)

                # "Just Advanced" Transformer ingredients (paper Sec. 4.4 / Tab. 4).
                cfg['use_qk_norm'] = False if is_rin or is_fcdm else get_input("Use QK-Norm (Just-Advanced)?", "1", bool)
                cfg['use_final_adaln'] = True if is_fcdm else False if is_rin else get_input("Use final adaLN-Zero head (DiT)?", "1", bool)
                cfg['use_grn'] = (get_input("Use Global Response Normalization (GRN)?", "0", bool)
                                  if cfg['model_type'] in GRN_TOKEN_MODELS else False)
                # Dropout on the middle-half of blocks (paper: 0 for B/L, 0.2 for H/G).
                cfg['dropout'] = 0.0 if is_fcdm else get_input("Dropout (latent processing, 0=off)" if is_rin
                                           else "Dropout (middle-half blocks, 0=off)", 0.0, float)
                if is_rin:
                    min_slots = 3 if cfg['conditioning_mode'] == 'class' else 2
                    cfg['rin_num_latents'] = get_number("RIN slots (including time/class tokens)", 256, int, min_slots)
                    cfg['rin_latent_dim'] = get_number("RIN latent width (multiple of 8 and heads)", 2 * cfg['dim'], int, 16)
                    cfg['rin_layers_per_block'] = get_number("RIN latent layers per routing block", 4, int, 1)
                else:
                    cfg['rin_num_latents'] = 256

                use_bn = (get_input("Use Bottleneck Patch Embed?", "1", bool)
                          if not is_rin and not cfg['conv_stem'] & 1 else False)
                cfg['bottleneck_dim'] = get_number("Bottleneck dim", 128, int, 1) if use_bn else None

                if cfg['model_type'] == 'fcdm_isotropic':
                    cfg['fcdm_mlp_ratio'] = get_number('FCDM channel expansion ratio', 3.0, float, 1)
                    cfg['use_adaln'] = cfg['use_final_adaln'] = True
                    print('│  FCDM blocks and final normalization always use class/time conditioning.')
                cfg['fmap_max'] = 0
                cfg['bottleneck_res'] = 0
            elif is_conv:
                raw_dim = get_input("Initial Filter Count (dim)", 64, int)
                cfg['dim'] = make_divisible(raw_dim, 8)
                if cfg['dim'] != raw_dim:
                    print(f"│  Rounded width to {cfg['dim']}")
                cfg['fmap_max'] = get_number("Max Filter Count", 512, int, 1)
                cfg['bottleneck_res'] = get_number("Bottleneck Resolution", 8, int, 1)
                for k in ['p_width', 'p_height', 'depth', 'heads']:
                    cfg[k] = 0
                for k in ['use_adaln', 'use_2d_pos_emb', 'use_conv_mlp', 'use_grn', 'self_cond', 'axial',
                          'use_qk_norm', 'use_final_adaln']:
                    cfg[k] = False
                cfg['self_cond_prob'] = 0.0
                cfg['bottleneck_dim'] = None
                cfg['overlap_h'] = 0
                cfg['overlap_w'] = 0
                cfg['dropout'] = 0.0
            else:
                raw_dim = get_input("Initial Feature Width (dim)", 64, int)
                cfg['dim'] = make_divisible(raw_dim, 8)
                if cfg['dim'] != raw_dim:
                    print(f"│  Rounded width to {cfg['dim']}")
                cfg['fmap_max'] = get_number("Max Feature Width", 512, int, 1)
                cfg['depth'] = get_number("MLP layers per hierarchy stage", 2, int, 1)
                cfg['conv_stem'] = get_number("Conv-stem [0=off, 1=input only, 2=output only, 3=both]", 0, int, 0, 3)
                if cfg['conv_stem']:
                    cfg['stem_initial'] = get_number("Stem initial filters", 32, int, 2)
                    cfg['stem_max'] = get_number("Stem max features", max(256, cfg['stem_initial']), int, cfg['stem_initial'])
                    cfg['stem_activation'] = prompt_stem_activation()
                    prompt_stem_extras(cfg)
                    print("│  Input stem uses the global-input grid; output stem decodes the root grid.")
                cfg['hier_input_grid_size'] = get_input(
                    "Global-input patch grid size (N means N×N)", 8, int)
                cfg['hier_output_grid_size'] = get_input(
                    "Local-refinement root grid size (must divide input grid)",
                    cfg['hier_input_grid_size'] if cfg.get('stem_skips', False) else 4, int)
                input_grid, output_grid = cfg['hier_input_grid_size'], cfg['hier_output_grid_size']
                output_scale_h = cfg['height'] // output_grid if output_grid else 0
                output_scale_w = cfg['width'] // output_grid if output_grid else 0
                valid_output_scale = (output_scale_h > 0 and output_scale_w > 0
                                      and output_scale_h == output_scale_w
                                      and output_scale_h & (output_scale_h - 1) == 0)
                if (input_grid < 1 or output_grid < 1 or input_grid < output_grid or input_grid % output_grid
                        or cfg['height'] % input_grid or cfg['width'] % input_grid
                        or not valid_output_scale):
                    print("│  ✕ HierMLP needs an input grid divisible by the root grid; "
                          "the root grid must reach both image dimensions through equal 2×2 refinements. Exiting.")
                    return
                # Retain the old field as the output-grid alias so tooling that
                # reads legacy configs still sees the refinement root.
                cfg['initial_grid_size'] = cfg['hier_output_grid_size']
                mixer_in = get_input(
                    "HierMLP global processor [0=JiT, 1=MLP-Mixer, 2=gMLP, 3=ConvNeXt, 4=ViP]",
                    "1", str).lower()
                cfg['hier_global_mixer'] = {
                    '0': 'jit', 'jit': 'jit',
                    '1': 'mlpmixer', 'mlpmixer': 'mlpmixer', 'mixer': 'mlpmixer',
                    '2': 'gmlp', 'gmlp': 'gmlp',
                    '3': 'convnext', 'convnext': 'convnext',
                    '4': 'vip', 'vip': 'vip',
                }.get(mixer_in, 'mlpmixer')
                raw_global_dim = get_input("HierMLP global processor width", cfg['dim'], int)
                cfg['hier_global_dim'] = make_divisible(raw_global_dim, 8)
                if cfg['hier_global_dim'] != raw_global_dim:
                    print(f"│  Rounded global processor width to {cfg['hier_global_dim']}")
                cfg['hier_global_depth'] = get_input(
                    "HierMLP global processor layer count", cfg['depth'], int)
                cfg['hier_global_heads'] = (get_input("HierMLP JiT attention head count", 4, int)
                                            if cfg['hier_global_mixer'] == 'jit' else 4)
                if cfg['hier_global_depth'] < 1 or cfg['hier_global_heads'] < 1:
                    print("│  ✕ HierMLP global processor depth and JiT head count must be at least 1. Exiting.")
                    return
                for k in ['p_width', 'p_height', 'heads']:
                    cfg[k] = 0
                for k in ['use_adaln', 'use_2d_pos_emb', 'use_conv_mlp', 'use_grn', 'self_cond', 'axial',
                          'use_qk_norm', 'use_final_adaln']:
                    cfg[k] = False
                cfg['self_cond_prob'] = 0.0
                cfg['bottleneck_dim'] = None
                cfg['bottleneck_res'] = 0
                cfg['overlap_h'] = 0
                cfg['overlap_w'] = 0
                cfg['dropout'] = 0.0
                cfg['rin_num_latents'] = 64

            ui_section("Diffusion objective")
            # Applies to every model type.
            # Paper's final algorithm = x-prediction trained with a v-loss
            # (Tab. 1(3)(a)). Other spaces are exposed to reproduce Tab. 1/2/3.
            pm = get_input("Network prediction space [x/eps/v]", "x", str).lower()
            cfg['pred_mode'] = pm if pm in ("x", "eps", "v") else "x"
            lm = get_input("Loss space [x/eps/v]", "v", str).lower()
            cfg['loss_mode'] = lm if lm in ("x", "eps", "v") else "v"
            cfg['t_mu'] = get_input("Logit-normal time mean mu (more negative = more noise)", -0.8, float)
            cfg['t_sigma'] = get_input("Logit-normal time std sigma", 0.8, float)
            ns = get_input("Noise schedule [linear/rin_sigmoid]", "linear", str).lower()
            cfg['noise_schedule'] = ns if ns in ("linear", "rin_sigmoid") else "linear"
            xc = get_input("Predicted-x clipping at sampling [none/static/dynamic]", "none", str).lower()
            cfg['x_clip'] = xc if xc in ("none", "static", "dynamic") else "none"
            ls = get_input("LR schedule after warmup [constant/cosine]", "constant", str).lower()
            cfg['lr_schedule'] = ls if ls in ("constant", "cosine") else "constant"
            cfg['time_scale'] = get_input("Time-embedding scale (t in [0,1] -> [0,scale])", 1000.0, float)
            # Paper's bottleneck is a purely-linear low-rank reparameterization.
            cfg['bottleneck_act'] = "none"
            # One pre-attention norm, so adaLN modulation is not re-normalized.
            cfg['attn_double_norm'] = False
            # Reference-faithful HAT/MaxViT/Swin/Vim blocks.
            cfg['arch_version'] = 2
            if cfg['model_type'] == 'hat':
                cfg['hat_group_depth'] = get_number("HAT blocks per residual group (paper: 6)", 2, int, 1)

            cfg['sampling_steps'] = get_number("Heun Sampling steps", 50, int, 1)
            cfg['batch_size'] = get_number("Batch size", 16, int, 1)
            ui_section("Optimizer")
            ui_menu([(k, f"{desc} · default lr {default_lr:g}")
                     for k, (_, desc, default_lr) in sorted(OPTIMIZER_TYPES.items(), key=lambda x: int(x[0]))])
            opt_in = get_input("Optimizer", "2", str)
            opt_type, opt_desc, default_lr = get_optimizer_choice(opt_in)
            cfg['optimizer_type'] = opt_type
            cfg['lr'] = get_input(f"Learning rate for {opt_desc}", default_lr, float)
            if opt_type in {'muon', 'adamuon', 'normuon', 'adago', 'adamgo', 'rmsgo', 'adadeltago'}:
                cfg['cautious'] = get_input("Cautious updates?", False, bool)
                cfg['muon_all'] = get_input("MuonAll (all parameters)?", False, bool)
                cfg['muon_all_reshape'] = get_input("MuonAll: use near-square vector reshape?", False, bool)
                if cfg['muon_all'] or cfg['lr'] <= 0:
                    cfg['muon_adam_lr_ratio'] = 1.0
                else:
                    fallback = get_number("AdamW fallback learning rate (embeddings, heads, norms, biases)",
                                          MUON_FALLBACK_LR if opt_type in GO_FAMILY else cfg['lr'], float, 1e-12)
                    cfg['muon_adam_lr_ratio'] = fallback / cfg['lr']
                cfg['muon_momentum'] = get_input("Muon momentum", 0.95, float)
                cfg['muon_rank'] = get_input("Muon rank (0=full Muon)", 0, int)
                cfg['muon_weight_decay'] = get_input("Muon weight decay", 0. if opt_type in {'adago', 'adamgo', 'rmsgo', 'adadeltago'} else 0.1, float)
                cfg['muon_backend'] = ('polar_express' if get_input("Use Polar Express backend?", True, bool)
                                       else 'newton_schulz')
                cfg['muon_ns_steps'] = get_number("Orthogonalization iterations", 5, int, 1)
                cfg['muon_foreach'] = get_input("Batch optimizer updates?", "1", bool)
                cfg['muon_ns_bfloat16'] = get_input("BF16 matrix iterations?", "0", bool)
                if opt_type == 'adamuon':
                    cfg['adamuon_eps'] = get_number("AdaMuon epsilon", 1e-8, float, 1e-16)
                    cfg['adamuon_nesterov'] = get_input("AdaMuon Nesterov (off=paper Algorithm 1)?", False, bool)
                if opt_type == 'normuon':
                    cfg['normuon_beta2'] = get_number("NorMuon variance decay (beta2)", .95, float, 0, .999999)
                    cfg['normuon_eps'] = get_number("NorMuon epsilon", 1e-8, float, 1e-16)
                if opt_type == 'adago':
                    cfg['adago_gamma'] = get_number("AdaGO gradient norm cap (gamma)", 1., float, 1e-16)
                    cfg['adago_v0'] = get_number("AdaGO initial norm accumulator (v0)", 1., float, 1e-16)
                    cfg['adago_eps'] = get_number("AdaGO minimum step (epsilon)", 5e-4, float, 1e-16)
                if opt_type == 'rmsgo':
                    cfg['rmsgo_gamma'] = get_number("RMSGO gradient norm cap (gamma)", 1., float, 1e-16)
                    cfg['rmsgo_v0'] = get_number("RMSGO initial norm accumulator (v0)", 1., float, 1e-16)
                    cfg['rmsgo_eps'] = get_number("RMSGO minimum step (epsilon)", 5e-4, float, 1e-16)
                    cfg['rmsgo_beta2'] = get_number("RMSGO norm variance decay (beta2)", .99, float, 0, .999999)
                if opt_type == 'adadeltago':
                    cfg['adadeltago_gamma'] = get_number("AdaDeltaGO gradient norm cap (gamma)", 1., float, 1e-16)
                    cfg['adadeltago_rho'] = get_number("AdaDeltaGO averaging decay (rho)", .9, float, 0, .999999)
                    cfg['adadeltago_eps'] = get_number("AdaDeltaGO RMS stabilizer (epsilon)", 1e-6, float, 1e-16)
                if opt_type == 'adamgo':
                    cfg['adamgo_beta2'] = get_number("AdamGO norm variance decay (beta2)", .999, float, 0, .999999)
                    cfg['adamgo_gamma'] = get_number("AdamGO gradient norm cap (gamma)", 1., float, 1e-16)
                    cfg['adamgo_delta'] = get_number("AdamGO denominator stabilizer (delta)", 1e-8, float, 1e-16)
                    cfg['adamgo_min_step'] = get_number("AdamGO minimum step (0=disabled)", 0., float, 0)
            if opt_type == 'paper_adamhd':
                cfg['hyper_lr'] = get_input("Hypergradient learning rate beta", 1e-10, float)
                cfg['hd_min_lr'] = get_input("PaperAdamHD minimum LR", 1e-5, float)
                cfg['hd_max_lr'] = get_input("PaperAdamHD maximum LR", 1e-2, float)
            if opt_type == 'modern_clion':
                cfg['modern_clion_nu'] = get_input("ModernCLion threshold nu", MODERN_CLION_DEFAULT_NU, float)
                cfg['modern_clion_weight_decay'] = get_input("ModernCLion weight decay", 0.0, float)
            if opt_type == 'clion':
                cfg['clion_rescale_mask'] = get_input(
                    "CLion paper mask rescaling?", "0", bool,
                )
            if opt_type == 'layerwise_nsgda':
                cfg['layerwise_nsgda_momentum'] = get_input("LayerWise nSGDA momentum", 0.9, float)
                cfg['layerwise_nsgda_cautious'] = get_input("LayerWise nSGDA cautious mask?", "1", bool)
            ui_section("Runtime")
            cfg['steps'] = get_number("Step count", 100000, int)
            cfg['use_ema'] = get_input("Use EMA?", "1", bool)
            if cfg['use_ema']:
                warmup = get_input("EMA warmup [smooth/legacy/none]", "smooth", str).lower()
                cfg['ema_warmup'] = warmup if warmup in EMA_WARMUP_MODES else 'smooth'
                if cfg['ema_warmup'] == 'smooth':
                    cfg['ema_warmup_steps'] = get_number("EMA warmup steps (decay reaches 0.9999)", 10000, int, 1)
            cfg['use_flip'] = get_input("Use RandFlip Augmentation?", "1", bool)
            cfg['use_vflip'] = get_input("Use random vertical flip?", "0", bool)
            cfg['use_rot90'] = get_input("Use random 90° rotations (square outputs only)?", "0", bool)
            cfg['use_amp'] = get_input("Use Mixed Precision (AMP)?", "1", bool)
            amp_dtype = get_input("AMP dtype [fp16/bf16]", "fp16", str).lower() if cfg['use_amp'] else "fp16"
            cfg['amp_dtype'] = amp_dtype if amp_dtype in {"fp16", "bf16"} else "fp16"
            cfg['full_bf16'] = get_input("Use full BF16 weights and inputs? (experimental)", "0", bool)
            if cfg['full_bf16']:
                # Full BF16 is a separate mode: do not combine it with AMP's
                # mixed FP32/BF16 operator policy.
                cfg['use_amp'] = False
            cfg['grad_clip'] = get_input("Gradient clip norm (0=off)", 1.0, float)
            cfg['use_spike_guard'] = get_input("Skip destructive loss/gradient spikes?", "1", bool)
            cfg['use_grad_ckpt'] = get_input("Use Gradient Checkpointing?", "0", bool)
            cfg['compile_mode'] = prompt_compile_mode('off')
            cfg['warmup_steps'] = get_input("LR Warmup steps", 1000, int)
            cfg['seed'] = get_input("Random seed (0=random)", 42, int)
            cfg['sample_every'] = get_number("Sample every N steps", 250, int, 1)
            cfg['save_every'] = get_number("Save checkpoint every N steps", 5000, int, 1)
            cfg['num_sample_images'] = get_number("Number of sample images per grid", 4, int, 1)
            cfg['preview_batch_size'] = get_number("Preview batch size", cfg['batch_size'], int, 1)
            cfg['preview_seed'] = get_number("Fixed preview seed", cfg['seed'] or 42, int)

            apply_config_defaults(cfg)
            prompt_training_features(cfg)
            validate_config(cfg)

            # Optional dataset resize & cache
            if get_input("Resize and cache dataset?", "0", bool):
                if cfg['resize_width'] == -1 or cfg['resize_height'] == -1:
                    print("│  ⚠ Native-size loading cannot be pre-resized into a cache; continuing without cache.")
                    cfg['dataset_path'] = cfg['dataset_path']
                else:
                    original_path = cfg['dataset_path']
                    resized_path = os.path.join(original_path, "resized")
                    files = [
                        path for path in find_image_paths(original_path)
                        if os.path.relpath(path, original_path).split(os.sep)[0] != 'resized'
                    ]
                    if files:
                        if os.path.exists(resized_path):
                            shutil.rmtree(resized_path)
                        os.makedirs(resized_path)
                        resample = getattr(Image.Resampling, 'LANCZOS', Image.LANCZOS)
                        mode = {3: 'RGB', 1: 'L', 4: 'RGBA'}.get(cfg['channels'], 'RGB')
                        for fpath in tqdm(files, desc="Resizing"):
                            try:
                                with Image.open(fpath) as img:
                                    if cfg['conditioning_mode'] == 'pix2pix' and cfg.get('pix2pix_source_mode') == 'combined':
                                        resize_size = (2 * cfg['resize_width'], cfg['resize_height'])
                                    else:
                                        resize_size = (cfg['resize_width'], cfg['resize_height'])
                                    img = img.convert(mode).resize(resize_size, resample)
                                    relative = os.path.relpath(fpath, original_path)
                                    base_dest = os.path.join(resized_path, os.path.splitext(relative)[0] + ".png")
                                    dest = base_dest
                                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                                    counter = 1
                                    while os.path.exists(dest):
                                        stem, extension = os.path.splitext(base_dest)
                                        dest = f"{stem}_{counter}{extension}"
                                        counter += 1
                                    img.save(dest)
                            except Exception:
                                pass
                        cfg['dataset_path'] = resized_path

            # Clean old samples
            #for f in glob.glob(os.path.join(output_dir, "sample_*.png")):
            #    os.remove(f)

        # --- Apply defaults for legacy configs ---
        # NOTE: architecture-changing additions (qk_norm, final adaLN, time
        # scaling) default to their LEGACY (off / 1.0) values here so that
        # resuming an old checkpoint rebuilds the exact original architecture.
        # New configs created above already carry the paper-faithful values.
        apply_config_defaults(cfg)
        if not loaded and fine_checkpoint is None:
            cfg['finetune_policy'] = 'all'
        if cfg['conditioning_mode'] == 'class':
            names = discover_class_names(cfg['dataset_path'])
            if loaded or fine_checkpoint is not None:
                unknown = set(names) - set(cfg['class_names'])
                if unknown:
                    raise ConfigError(f'New class labels require explicit transfer mapping: {sorted(unknown)}')
            else:
                cfg['class_names'] = names
                if not names:
                    raise ConfigError('No class folders found')
        if not loaded and fine_checkpoint is None:
            validate_fresh_run(cfg)
        validate_config(cfg)

        bf16_supported = (device.type == 'cuda'
                          and getattr(torch.cuda, 'is_bf16_supported', lambda: False)())
        if cfg['full_bf16'] and not bf16_supported:
            raise RuntimeError("Full BF16 requires a CUDA device with native BF16 support")
        if cfg['use_amp'] and cfg['amp_dtype'] == 'bf16' and not bf16_supported:
            print("│  ⚠ BF16 AMP is unavailable; falling back to FP16 AMP")
            cfg['amp_dtype'] = 'fp16'

        # Seed
        if cfg['seed'] > 0:
            set_seed(cfg['seed'])

        # --- Build Model ---
        model = build_model(cfg, device)
        if fine_checkpoint is not None:
            model.load_state_dict(fine_checkpoint[cfg['finetune_weights']], strict=True)
            if cfg['finetune_policy'] == 'modules':
                print('\n'.join(name for name, module in model.named_modules() if name and list(module.parameters())))
                cfg['trainable_modules'] = [p.strip() for p in get_input(
                    'Trainable module names/patterns (comma separated)', '').split(',') if p.strip()]
        configure_trainable(model, cfg)

        ui_config_summary(cfg, parameter_count=count_parameters(model))
        ui_section("Diffusion")
        ui_key_value("Time sampling", f"logit-normal μ={cfg['t_mu']} σ={cfg['t_sigma']}")
        ui_key_value("Schedule / x clipping", f"{cfg['noise_schedule']} / {cfg['x_clip']}")
        ui_rule()

        ema = (EMA(model, warmup_steps=cfg['ema_warmup_steps'], warmup=cfg['ema_warmup'])
               if cfg['use_ema'] else None)
        maybe_compile(model, cfg, device)  # after EMA: the shadow copy stays eager
        flow_model = FlowMatchingWrapper(
            model, pred_mode=cfg['pred_mode'], loss_mode=cfg['loss_mode'],
            t_loc=cfg['t_mu'], t_scale=cfg['t_sigma'], x_clip=cfg['x_clip'], noise_schedule=cfg['noise_schedule'],
            self_cond_prob=cfg['self_cond_prob'],
            class_dropout_prob=cfg['class_dropout_prob'],
        ).to(device)
        optimizer = build_optimizer(model, cfg)

        scheduler = CosineWarmupScheduler(
            optimizer, warmup_steps=cfg['warmup_steps'],
            total_steps=cfg['steps'], min_lr_ratio=0.1, mode=cfg['lr_schedule'],
        )

        start_step = 0
        if loaded and os.path.exists(model_path):
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            if 'completed_steps' not in ckpt and cfg['optimizer_type'] == 'clion':
                for group in optimizer.param_groups:
                    group['betas'] = (0.95, 0.98)
                    group['weight_decay'] = 0.0
                    group['rescale_mask'] = cfg.get('clion_rescale_mask', False)
                    group['foreach'] = True
            if 'completed_steps' not in ckpt and cfg['optimizer_type'] == 'modern_clion':
                for group in optimizer.param_groups:
                    group['nu'] = cfg.get('modern_clion_nu', MODERN_CLION_DEFAULT_NU)
                    group['weight_decay'] = cfg.get('modern_clion_weight_decay', group.get('weight_decay', 0.0))
            if 'completed_steps' not in ckpt and cfg['optimizer_type'] == 'layerwise_nsgda':
                for group in optimizer.param_groups:
                    group['momentum'] = cfg.get('layerwise_nsgda_momentum', 0.9)
                    group['cautious'] = cfg.get('layerwise_nsgda_cautious', True)
            if ema and 'ema' in ckpt:
                ema.shadow.load_state_dict(ckpt['ema'])
            start_step = ckpt.get('completed_steps', ckpt.get('step', -1) + 1)
            if ema:
                ema.step_count = ckpt.get('ema_step_count', start_step)
            old_schedule = (cfg['steps'], cfg['lr'], cfg['warmup_steps'], cfg['lr_schedule'],
                            cfg.get('muon_adam_lr_ratio', 1.0))
            old_current_lr = optimizer.param_groups[0]['lr']
            prompt_resume_settings(cfg, optimizer, start_step)
            prompt_training_features(cfg)
            if (ckpt.get('scheduler') and cfg['steps'] == old_schedule[0]
                    and cfg['lr'] == old_current_lr and cfg['lr_schedule'] == 'cosine'):
                cfg['lr'] = old_schedule[1]
            validate_config(cfg)
            scheduler = CosineWarmupScheduler(optimizer, cfg['warmup_steps'], cfg['steps'],
                                              min_lr_ratio=0.1, mode=cfg['lr_schedule'],
                                              resume_step=(start_step if cfg['lr_schedule'] == 'cosine'
                                                           and start_step >= cfg['warmup_steps'] else None),
                                              min_lrs=[min(cfg['cosine_min_lr'] * muon_group_lr_scale(group, cfg),
                                                           group['lr'])
                                                       for group in optimizer.param_groups])
            if (ckpt.get('scheduler') and old_schedule ==
                    (cfg['steps'], cfg['lr'], cfg['warmup_steps'], cfg['lr_schedule'],
                     cfg.get('muon_adam_lr_ratio', 1.0))):
                for key, value in ckpt['scheduler'].items():
                    setattr(scheduler, key, value)
            restore_rng_state(ckpt.get('rng'))
            print(f"│  Resuming checkpoint at step {start_step:,}")

        ds, validation_ds = build_training_datasets(cfg)
        reset_stream = cfg['reset_data_stream']
        stream_state = ckpt.get('data_stream') if loaded and not reset_stream else None
        if loaded and not stream_state:
            print('│  Starting a new data stream (reset requested or legacy checkpoint).')
        dl_iter = TrainingStream(ds, cfg, stream_state)
        if cfg['conditioning_mode'] == 'class':
            print(f"│  Class sampling: {cfg['class_sampling']} · {dl_iter.epoch_size:,} examples per epoch")
        cfg['reset_data_stream'] = False
        best = ckpt.get('best', {}) if loaded and not reset_stream else {}
        evaluation_keys = ('pred_mode', 'loss_mode', 'noise_schedule', 't_mu', 't_sigma', 'self_cond_prob',
                           'use_ema', 'validation_seed', 'validation_batch_size', 'validation_max_batches',
                           'best_training_loss', 'grad_accum_steps', 'batch_size')
        evaluation_files = ([(p, os.stat(p).st_size, os.stat(p).st_mtime_ns)
                             for p in dataset_files(validation_ds)] if validation_ds is not None else [])
        best_context = stable_digest([dl_iter.fingerprint, evaluation_files,
                                      {k: cfg[k] for k in evaluation_keys}])
        if best.get('context') != best_context:
            best = {'context': best_context}
        best_path = os.path.join(checkpoint_dir, 'best.pt')
        save_config(cfg, config_path)

        amp_enabled = cfg['use_amp'] and device.type == 'cuda'
        amp_dtype = torch.bfloat16 if cfg['amp_dtype'] == 'bf16' else torch.float16
        # Gradient scaling prevents FP16 underflow; BF16 has FP32-like range.
        scaler = GradScaler(enabled=amp_enabled and cfg['amp_dtype'] == 'fp16')

        if loaded and ckpt.get('scaler'):
            scaler.load_state_dict(ckpt['scaler'])
        if is_schedule_free(optimizer):
            optimizer.train()  # checkpoints hold x; training steps run at y
        if cfg['compile_mode'] != 'off':
            verify_compiled_gradients(model, flow_model, cfg, device, amp_enabled, amp_dtype)
        parameter_list = [p for p in model.parameters() if p.requires_grad]  # built once, walked every step

        with training_session(dl_iter):
            # Loss tracking
            loss_ema = 0.0
            loss_ema_decay = 0.99
            grad_norm_ema = 0.0
            spike_guard_safe_steps = 0
            spike_guard_skips = 0

            ui_header("Training", "Press Ctrl+C to finish the current step, save, and exit")
            completed_steps = start_step
            consecutive_skips = 0
            if loaded and not reset_stream and ckpt.get('guard'):
                state = ckpt['guard']
                loss_ema = state['loss_ema']
                grad_norm_ema = state['grad_norm_ema']
                spike_guard_safe_steps = state['safe_steps']
            pbar = tqdm(initial=start_step, total=cfg['steps'],
                        desc="Training", unit="step", dynamic_ncols=True)
            pbar.write(f"Spike diagnostics: {os.path.join(output_dir, 'spike_events.log')}")
            while completed_steps < cfg['steps']:
                step = completed_steps
                if interrupted:
                    break

                batches = [next(dl_iter) for _ in range(cfg['grad_accum_steps'])]

                # LR schedule (linear warmup, then constant per the paper unless
                # cfg['lr_schedule'] == 'cosine'). PaperAdamHD and Prodigy adapt
                # their step sizes internally, so do not overwrite them.
                if cfg['optimizer_type'] not in {'paper_adamhd', 'prodigy', 'radam_schedulefree'}:
                    scheduler.step(step)

                optimizer.zero_grad(set_to_none=True)
                loss_tensor = accumulated_backward(flow_model, batches, cfg, scaler, amp_enabled, amp_dtype)
                del batches

                # Unscale before measuring the real global norm. clip_grad_norm_
                # returns the norm before clipping, so it can detect finite but
                # destructive jumps even when ordinary clipping is enabled.
                # Loss and norm then reach the host in ONE sync per update.
                scaler.unscale_(optimizer)
                max_norm = cfg['grad_clip'] if cfg['grad_clip'] > 0 else float('inf')
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                loss_val, grad_norm_val = torch.stack(
                    (loss_tensor, grad_norm.detach().float().to(loss_tensor.device))).tolist()

                # A finite loss can still be catastrophically larger than the
                # recent flow-matching distribution. Skip it before it reaches
                # the optimizer or momentum buffers.
                loss_limit = max(loss_ema * cfg['spike_loss_factor'], 1e-6)
                if (not math.isfinite(loss_val) or (cfg['use_spike_guard']
                        and spike_guard_safe_steps >= cfg['spike_guard_warmup']
                        and loss_val > loss_limit)):
                    optimizer.zero_grad(set_to_none=True)
                    # unscale_ already ran; update() resets it (and backs off
                    # the fp16 scale when the gradients overflowed).
                    scaler.update()
                    spike_guard_skips += 1
                    consecutive_skips += 1
                    report_training_spike(
                        output_dir, pbar.write, kind='loss', value=loss_val, limit=loss_limit,
                        loss=loss_val, completed_steps=completed_steps, optimizer=optimizer,
                        total_skips=spike_guard_skips, consecutive_skips=consecutive_skips)
                    if consecutive_skips >= 100:
                        pbar.write("Stopping after 100 consecutive rejected batches; saving progress.")
                        break
                    continue

                grad_limit = max(grad_norm_ema * cfg['spike_grad_factor'], 1e-6)
                if (not math.isfinite(grad_norm_val) or (cfg['use_spike_guard']
                        and spike_guard_safe_steps >= cfg['spike_guard_warmup']
                        and grad_norm_val > grad_limit)):
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    spike_guard_skips += 1
                    consecutive_skips += 1
                    report_training_spike(
                        output_dir, pbar.write, kind='grad', value=grad_norm_val, limit=grad_limit,
                        loss=loss_val, completed_steps=completed_steps, optimizer=optimizer,
                        total_skips=spike_guard_skips, consecutive_skips=consecutive_skips)
                    if consecutive_skips >= 100:
                        pbar.write("Stopping after 100 consecutive rejected batches; saving progress.")
                        break
                    continue

                align_grad_strides(parameter_list)
                scaler.step(optimizer)
                scaler.update()
                consecutive_skips = 0
                completed_steps += 1
                pbar.update(1)

                if ema:
                    ema.update(model)

                # Track accepted updates only; rejected spikes cannot inflate the
                # guard's own baseline and make it permissive.
                loss_ema = loss_ema * loss_ema_decay + loss_val * (1 - loss_ema_decay) if spike_guard_safe_steps else loss_val
                grad_norm_ema = (grad_norm_ema * cfg['spike_ema_decay'] + grad_norm_val * (1 - cfg['spike_ema_decay'])
                                 if spike_guard_safe_steps else grad_norm_val)
                spike_guard_safe_steps += 1
                current_lr = optimizer.param_groups[0].get('last_dlr', optimizer.param_groups[0]['lr'])
                desc = f"loss {loss_val:.4f} · avg {loss_ema:.4f} · lr {current_lr:.2e}"
                if cfg['optimizer_type'] == 'paper_adamhd':
                    last_hypergrad = optimizer.param_groups[0].get('last_hypergrad', 0.0)
                    desc += f" | HD: {last_hypergrad:.2e}"
                elif cfg['optimizer_type'] == 'prodigy':
                    d = optimizer.param_groups[0].get('d', 0.0)
                    desc += f" | D: {d:.2e}"
                elif cfg['optimizer_type'] == 'modern_clion':
                    sign_fraction = optimizer.param_groups[0].get('last_sign_fraction', None)
                    min_c = optimizer.param_groups[0].get('last_min_nonzero_abs', None)
                    if sign_fraction is None:
                        used_sign = optimizer.param_groups[0].get('last_used_sign', None)
                        mode = "sign" if used_sign else "raw"
                    elif sign_fraction >= 0.999999:
                        mode = "sign"
                    elif sign_fraction <= 1e-12:
                        mode = "raw"
                    else:
                        mode = f"mixed {sign_fraction:.1%} sign"
                    if min_c is not None:
                        desc += f" | MCLion: {mode} min|c|={min_c:.1e}"
                pbar.set_postfix_str(desc, refresh=False)

                step = completed_steps
                val_score = None
                if validation_ds is not None and (step % cfg['validation_every'] == 0 or step == cfg['steps']):
                    evaluation_model = ema.shadow if ema else model
                    with schedule_free_eval(optimizer):
                        val_score = validation_loss(evaluation_model, validation_ds, cfg, amp_enabled, amp_dtype)
                    pbar.write(f'Validation loss at {step}: {val_score:.6g}')
                    best = update_best(evaluation_model, cfg, val_score, step, best, best_path, 'validation_loss',
                                       optimizer=optimizer)
                elif validation_ds is None and cfg['best_training_loss']:
                    # This metric came from the training model, not its EMA shadow.
                    best = update_best(model, cfg, loss_ema, step, best, best_path, 'training_loss_ema',
                                       optimizer=optimizer)
                metrics_path = os.path.join(output_dir, 'metrics.csv')
                new_metrics = not os.path.exists(metrics_path)
                with open(metrics_path, 'a', newline='', encoding='utf-8') as handle:
                    writer = csv.writer(handle)
                    if new_metrics:
                        writer.writerow(['step', 'train_loss', 'train_loss_ema', 'validation_loss', 'lr', 'grad_norm'])
                    writer.writerow([step, loss_val, loss_ema, val_score, current_lr, grad_norm_val])
                # Optional previews use fixed examples and never interrupt checkpointing.
                if step > 0 and step % cfg['sample_every'] == 0:
                    with schedule_free_eval(optimizer):
                        safe_training_preview(ema.shadow if ema else model, validation_ds if validation_ds is not None else ds, cfg,
                                              f"{output_dir}/sample_{step}.png", pbar.write,
                                              amp_enabled=amp_enabled, amp_dtype=amp_dtype)

                # --- Checkpointing ---
                if step > 0 and step % cfg['save_every'] == 0:
                    save_checkpoint(model, optimizer, ema, completed_steps, model_path, max_keep=1,
                                scaler=scaler, cfg=cfg, stream=dl_iter, best=best, scheduler=scheduler, guard={'loss_ema': loss_ema,
                                'grad_norm_ema': grad_norm_ema, 'safe_steps': spike_guard_safe_steps})
                    pbar.write(f"  ✓ Checkpoint saved at step {step:,}")

            # Final save
            save_checkpoint(model, optimizer, ema, completed_steps, model_path, max_keep=1,
                                scaler=scaler, cfg=cfg, stream=dl_iter, best=best, scheduler=scheduler, guard={'loss_ema': loss_ema,
                                'grad_norm_ema': grad_norm_ema, 'safe_steps': spike_guard_safe_steps})
            pbar.close()
        ui_header("Training complete" if completed_steps >= cfg['steps'] else "Training stopped",
                  f"Completed {completed_steps:,} updates · Final checkpoint: {model_path}")

    elif mode_in in ['s', '1', 'sample']:
        ui_section("Sampling setup")
        model_path = get_input('Checkpoint to sample (model.pt or best.pt)', model_path)
        if not os.path.exists(model_path):
            print("│  No configuration found. Train a model first.")
            return
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        cfg = ckpt.get('config') or load_config(previous_config)

        apply_config_defaults(cfg)

        bf16_supported = (device.type == 'cuda'
                          and getattr(torch.cuda, 'is_bf16_supported', lambda: False)())
        if cfg['full_bf16'] and not bf16_supported:
            raise RuntimeError("Full BF16 checkpoint sampling requires CUDA BF16 support")
        if cfg['use_amp'] and cfg['amp_dtype'] == 'bf16' and not bf16_supported:
            print("│  ⚠ BF16 AMP is unavailable; sampling with FP16 AMP")
            cfg['amp_dtype'] = 'fp16'

        steps = get_number("Heun Steps", cfg.get('sampling_steps', 50), int, 1)
        if cfg['conditioning_mode'] == 'class' and cfg['class_dropout_prob'] > 0:
            cfg['guidance_scale'] = get_input(
                "Class guidance strength (1=unguided)", cfg['guidance_scale'], float)
        seed = get_input("Seed (0=random)", 0, int)
        if cfg['conditioning_mode'] == 'unconditional':
            count = get_number("Image count", 4, int, 1)
            batch_size = get_number("Batch size for sampling", min(count, 4), int, 1)
        else:
            count = 0
            batch_size = get_number("Batch size for sampling", 4, int, 1)
        # Allow overriding predicted-x clipping at inference without retraining.
        xc = get_input("Predicted-x clipping [none/static/dynamic]", cfg['x_clip'], str).lower()
        cfg['x_clip'] = xc if xc in ("none", "static", "dynamic") else cfg['x_clip']

        #for f in glob.glob(os.path.join(output_dir, "generated_*.png")):
        #    os.remove(f)

        if seed > 0:
            set_seed(seed)

        # Build model (FIX: pass all config params including overlap/axial)
        model = build_model(cfg, device)

        sampling_description = (f"Generating {count} image(s) with {steps} Heun steps"
                                if cfg['conditioning_mode'] == 'unconditional'
                                else f"Interactive {cfg['conditioning_mode']} sampling with {steps} Heun steps")
        ui_header("Sampling", sampling_description)
        ui_key_value("Model", cfg['model_type'])
        ui_key_value("Resolution", f"{cfg['width']}×{cfg['height']}")
        ui_key_value("Batch size", batch_size)
        ui_rule()
        if cfg['use_ema'] and 'ema' in ckpt:
            print("│  Loading EMA weights…")
            model.load_state_dict(ckpt['ema'])
        else:
            model.load_state_dict(ckpt['model'])

        model.eval()
        maybe_compile(model, cfg, device)
        flow_model = FlowMatchingWrapper(
            model, pred_mode=cfg['pred_mode'], loss_mode=cfg['loss_mode'],
            t_loc=cfg['t_mu'], t_scale=cfg['t_sigma'], x_clip=cfg['x_clip'], noise_schedule=cfg['noise_schedule'],
            self_cond_prob=cfg['self_cond_prob'],
            class_dropout_prob=cfg['class_dropout_prob'],
        ).to(device)

        amp_enabled = cfg['use_amp'] and device.type == 'cuda'
        amp_dtype = torch.bfloat16 if cfg['amp_dtype'] == 'bf16' else torch.float16

        generated = 0

        def generate(batch_condition=None, batch_classes=None):
            """Sample model-sized images in chunks of batch_size."""
            total = batch_condition.shape[0] if batch_condition is not None else batch_classes.shape[0]
            outputs = []
            for start in range(0, total, max(1, batch_size)):
                end = min(start + max(1, batch_size), total)
                condition_chunk = batch_condition[start:end] if batch_condition is not None else None
                class_chunk = batch_classes[start:end] if batch_classes is not None else None
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                    outputs.extend(flow_model.sample(
                        (end - start, cfg['channels'], cfg['height'], cfg['width']), steps=steps,
                        condition=condition_chunk, class_labels=class_chunk,
                        guidance_scale=cfg['guidance_scale'],
                    ))
            return outputs

        def save_generated(image, source=None, combined_output_dir=None):
            nonlocal generated
            if combined_output_dir is not None:
                combined = make_combined_pix2pix_image(source, image, cfg['pix2pix_direction'])
                output_path = os.path.join(combined_output_dir, f"generated_{generated:06d}.png")
                combined.save(output_path)
                print(f"│  ✓ Saved combined pair: {output_path}")
            else:
                tensor_to_pil_image(image).save(f"{output_dir}/generated_{generated}.png")
                print(f"│  ✓ Saved generated_{generated}.png")
            generated += 1

        def sample_and_save(batch_classes):
            for image in generate(batch_classes=batch_classes):
                save_generated(image)

        if cfg['conditioning_mode'] == 'class':
            class_lookup = {name.casefold(): index for index, name in enumerate(cfg['class_names'])}
            print("│  Enter class, class*count, blank for a random class, or END to finish.")
            while True:
                request = input("│  Class: ").strip()
                if request.casefold() == 'end':
                    break
                name, amount = request, 1
                if '*' in request:
                    requested_name, requested_amount = request.rsplit('*', 1)
                    try:
                        amount = max(1, int(requested_amount))
                        name = requested_name.strip()
                    except ValueError:
                        print("│  Invalid count; generating one random class image.")
                        name, amount = '', 1
                class_index = class_lookup.get(name.casefold())
                if class_index is None:
                    if name:
                        print(f"│  Unknown class {name!r}; using a random class.")
                    class_index = random.randrange(len(cfg['class_names']))
                class_batch = torch.full((amount,), class_index, device=device, dtype=torch.long)
                sample_and_save(batch_classes=class_batch)
        elif cfg['conditioning_mode'] == 'pix2pix':
            framing = get_input("Inputs larger than the model at training scale: tile or center",
                                "tile", str).lower()
            if framing not in ('tile', 'center'):
                print("│  Unknown framing; using tile.")
                framing = 'tile'
            print("│  Enter an image file or folder; blank ends this sampling session.")
            while True:
                requested_path = input("│  Input path: ").strip()
                if not requested_path:
                    break
                if os.path.isdir(requested_path):
                    paths = find_image_paths(requested_path)
                    if not paths:
                        print("│  No valid image files found in that folder.")
                        continue
                    combined_output_dir = None
                    if get_input("Generate a combined A|B dataset from this folder?", False, bool):
                        combined_output_dir = make_unique_output_dir(output_dir, 'combined_pairs')
                        print(f"│  Combined pairs will use A-left / B-right: {combined_output_dir}")
                elif os.path.isfile(requested_path):
                    paths = [requested_path]
                    combined_output_dir = None
                else:
                    print("│  Path is not a readable file or folder.")
                    continue
                for path in paths:
                    # Match the training scale; frame larger canvases as tiles or one centre crop.
                    try:
                        source = pix2pix_source_canvas(path, cfg)
                    except (OSError, ValueError) as exc:
                        print(f"│  Skipping invalid image {path}: {exc}")
                        continue
                    canvas = tuple(source.shape[1:])
                    corners = pix2pix_tile_corners(canvas, (cfg['height'], cfg['width']), framing)
                    tiles = torch.stack([source[:, top:top + cfg['height'], left:left + cfg['width']]
                                         for top, left in corners])
                    outputs = generate(batch_condition=tiles.to(device))
                    if framing == 'center':
                        source, result = tiles[0], outputs[0]
                    else:
                        result = stitch_tiles(outputs, corners, canvas)
                    if len(corners) > 1 or canvas != (cfg['height'], cfg['width']):
                        print(f"│  {os.path.basename(path)}: {canvas[1]}×{canvas[0]} at training scale, "
                              f"{len(corners)} {framing} window(s)")
                    save_generated(result, source, combined_output_dir)
        else:
            while generated < count:
                bs = min(max(1, batch_size), count - generated)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                    out = flow_model.sample((bs, cfg['channels'], cfg['height'], cfg['width']), steps=steps)
                for image in out:
                    img = T.ToPILImage()(((image.cpu().clamp(-1, 1) + 1) * 0.5).float())
                    img.save(f"{output_dir}/generated_{generated}.png")
                    print(f"│  ✓ Saved generated_{generated}.png")
                    generated += 1

        # Preserve the existing grid convention for count-based unconditional sampling.
        if cfg['conditioning_mode'] == 'unconditional' and count > 1:
            all_imgs = []
            for i in range(count):
                all_imgs.append(T.ToTensor()(Image.open(f"{output_dir}/generated_{i}.png")))
            from torchvision.utils import make_grid
            grid = T.ToPILImage()(make_grid(torch.stack(all_imgs), nrow=int(math.ceil(math.sqrt(count))), padding=2))
            grid.save(f"{output_dir}/generated_grid.png")
            print(f"│  ✓ Saved generated_grid.png")

        ui_header("Sampling complete", f"Images saved in {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as exc:
        print(f"│  Invalid configuration: {exc}")
