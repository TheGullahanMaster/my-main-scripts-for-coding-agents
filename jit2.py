import os
import sys
import glob
import math
import random
import signal
import copy
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from lamb import *
from torch.cuda.amp import autocast, GradScaler

# ==========================================
# Hardware Configuration
# ==========================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Auto-tune conv algorithms for fixed input sizes
print(f"🚀 TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")

SAVE_DIR = "JiTDiff_Flow"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# Global State
# ==========================================
interrupted = False
TRAINING_ACTIVE = False


def signal_handler(sig, frame):
    global interrupted
    if TRAINING_ACTIVE:
        print("\n⚠️ CTRL+C detected. Finishing current step, saving, and exiting...")
        interrupted = True
    else:
        print("\n🚫 CTRL+C detected. Exiting immediately.")
        sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def set_seed(seed):
    """Reproducibility helper."""
    random.seed(seed)
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
        x_norm = x * torch.rsqrt(norm_x + self.eps)
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
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
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


class EMA(nn.Module):
    def __init__(self, model, decay=0.9999, warmup_steps=2000):
        super().__init__()
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.shadow = copy.deepcopy(model)
        self.shadow.requires_grad_(False)
        self.shadow.eval()

    def get_decay(self):
        """Ramp up EMA decay during warmup to avoid copying random init."""
        if self.step_count < self.warmup_steps:
            return min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        return self.decay

    def update(self, model):
        self.step_count += 1
        decay = self.get_decay()
        with torch.no_grad():
            m_param = dict(model.named_parameters())
            s_param = dict(self.shadow.named_parameters())
            for key in m_param:
                if key in s_param:
                    s_param[key].data.lerp_(m_param[key].data, 1 - decay)

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
    B = int(windows.shape[0] / (H * W / window_size / window_size))
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
    def __init__(self, dim, heads=8, dim_head=64, h_patches=None, w_patches=None, qk_norm=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = RMSNorm(dim)
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
                 dropout=0.0):
        super().__init__()
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads=heads, dim_head=64, h_patches=h_patches, w_patches=w_patches, qk_norm=qk_norm)
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
    def __init__(self, dim, h, w, heads, use_adaln=False, qk_norm=False):
        super().__init__()
        self.use_adaln = use_adaln
        self.attn = Attention(dim, heads=heads, dim_head=64, h_patches=h, w_patches=w, qk_norm=qk_norm)
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
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = RMSNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.Mish()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        if use_adaln:
            self.adaLN_modulation = _make_adaln(dim, 3)
        else:
            self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x, t_emb=None):
        residual = x
        b, n, c = x.shape
        x = self.dwconv(x.transpose(1, 2).view(b, c, self.h, self.w)).view(b, c, n).transpose(1, 2)

        if self.use_adaln and t_emb is not None:
            shift, scale, gate = self.adaLN_modulation(t_emb).chunk(3, dim=1)
            x = self.pwconv2(self.act(self.pwconv1(modulate(self.norm(x), shift, scale))))
            return residual + gate.unsqueeze(1) * x
        else:
            x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
            return residual + self.gamma * x


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
        x = x.view(b, self.h, self.w, c)
        x_f = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x_f = x_f * torch.complex(self.w_real, self.w_imag)
        x = torch.fft.irfft2(x_f, s=(self.h, self.w), dim=(1, 2), norm="ortho")
        x = residual + x.view(b, n, c)
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
    def __init__(self, dim, num_patches, token_dim, channel_dim, heads, use_adaln=True):
        super().__init__()
        self.use_adaln = use_adaln
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.token_mlp = nn.Sequential(nn.Linear(num_patches, token_dim), nn.Mish(), nn.Linear(token_dim, num_patches))
        self.channel_attn = Attention(dim, heads=heads, dim_head=64)
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
    """Generates `num_filters` long 2D conv filters from a positional encoding via MLP."""
    def __init__(self, dim, h, w, filter_dim=64, num_filters=2):
        super().__init__()
        self.dim, self.h, self.w = dim, h, w
        self.num_filters = num_filters

        ys = torch.linspace(-1, 1, h)
        xs = torch.linspace(-1, 1, w)
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

    def conv_fft(self, v, h):
        # Linear convolution via zero-padding to 2× then crop
        Hp, Wp = 2 * self.h, 2 * self.w
        v_pad = F.pad(v, (0, self.w, 0, self.h))
        h_pad = F.pad(h, (0, self.w, 0, self.h))
        v_f = torch.fft.rfft2(v_pad, norm='ortho')
        h_f = torch.fft.rfft2(h_pad, norm='ortho')
        out = torch.fft.irfft2(v_f * h_f, s=(Hp, Wp), norm='ortho')
        return out[..., :self.h, :self.w]

    def forward(self, x):
        B, N, C = x.shape
        H, W = self.h, self.w

        u = self.in_proj(x).transpose(1, 2).view(B, C * (self.order + 1), H, W)
        u = self.short_filter(u)
        u = u.view(B, self.order + 1, C, H, W)
        v = u[:, 0]

        filters = self.filter_fn()  # order, C, H, W
        for i in range(self.order):
            v = self.conv_fft(v, filters[i]) * u[:, i + 1]

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
    def __init__(self, dim, h, w):
        super().__init__()
        self.h, self.w = h, w
        self.norm = RMSNorm(dim) # Vital for generative stability
        
        self.mlp_h = nn.Linear(h, h)
        self.mlp_w = nn.Linear(w, w)
        self.mlp_c = nn.Linear(dim, dim)
        
        self.reweight = nn.Sequential(
            nn.Linear(dim, dim // 4), 
            nn.Mish(),
            nn.Linear(dim // 4, dim * 3),
        )
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, t_emb=None):
        B, N, C = x.shape
        res = x # Keep a residual for the whole block
        
        x = self.norm(x)
        x_img = x.reshape(B, self.h, self.w, C)
        
        # Spatial Mixing
        x_h = self.mlp_h(x_img.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_w = self.mlp_w(x_img.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
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
        
        # Final projection + residual
        return res + self.proj(out.reshape(B, N, C))


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

class JiTModel(nn.Module):
    def __init__(self, img_size, patch_size, channels, dim, depth, heads,
                 model_type="jit", self_cond=False,
                 use_adaln=False, use_2d_pos_emb=False, use_conv_mlp=False,
                 bottleneck_dim=None, overlap_h=0, overlap_w=0, axial=False,
                 use_gradient_checkpointing=False,
                 use_qk_norm=False, use_final_adaln=False, time_scale=1.0,
                 bottleneck_act="none", dropout=0.0):
        super().__init__()
        self.channels = channels
        self.self_cond = self_cond
        self.patch_size = patch_size
        self.depth = depth
        self.dropout = dropout
        self.use_adaln = use_adaln
        self.use_2d_pos_emb = use_2d_pos_emb
        self.use_conv_mlp = use_conv_mlp
        self.use_qk_norm = use_qk_norm
        self.model_type = model_type
        self.axial = axial
        self.use_gradient_checkpointing = use_gradient_checkpointing

        h_img, w_img = img_size
        p_h, p_w = patch_size

        # Overlap setup
        self.overlap_h = overlap_h
        self.overlap_w = overlap_w
        self.stride_h = p_h - overlap_h
        self.stride_w = p_w - overlap_w
        assert self.stride_h > 0 and self.stride_w > 0, "Overlap must be smaller than patch size"

        self.h_patches = (h_img - p_h) // self.stride_h + 1
        self.w_patches = (w_img - p_w) // self.stride_w + 1
        self.num_patches = self.h_patches * self.w_patches
        print(f"Grid: {self.h_patches}x{self.w_patches} = {self.num_patches} patches. Overlap: H={overlap_h}, W={overlap_w}")

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=(self.stride_h, self.stride_w))
        self.fold = nn.Fold(output_size=img_size, kernel_size=patch_size, stride=(self.stride_h, self.stride_w))

        with torch.no_grad():
            ones_img = torch.ones(1, channels, h_img, w_img)
            divisor = self.fold(self.unfold(ones_img))
            self.register_buffer('overlap_divisor', divisor)

        patch_dim = channels * p_h * p_w
        in_channels = patch_dim * 2 if self_cond else patch_dim

        if bottleneck_dim is not None and bottleneck_dim < dim:
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

        if use_2d_pos_emb:
            self.register_buffer('pos_embedding',
                get_2d_sincos_pos_embed(dim, self.h_patches, self.w_patches).reshape(1, self.num_patches, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim, scale=time_scale),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim)
        )

        # Models that handle position internally (RoPE, convolutions, etc.)
        NO_POS_EMBED = {'jit', 'gmlp', 'oggmlp', 'amlp', 'ogamlp', 'mlpmixer',
                        'ogmlpmixer', 'pool', 'gru', 'convnext', 'fullattn',
                        'mixer_attn', 'cyclemlp', 'lka', 'resmlp', 'mdmlp',
                        'hat', 'xcit', 'coatnet', 'bigs', 'hornet', 'aft_full', 'aft_local', 'hyena',
                        'nat', 'volo', 'asmlp', 's2mlp', 'vip'}
        self.skip_pos_embed = model_type in NO_POS_EMBED

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(self._make_block(model_type, dim, heads, i))

        self.norm = RMSNorm(dim)
        # DiT-style final layer: optionally modulate the output head with
        # adaLN-Zero before projecting back to pixels (zero-init => identity
        # at start, so it never disrupts early training).
        self.use_final_adaln_active = use_final_adaln and use_adaln
        if self.use_final_adaln_active:
            self.final_adaLN = nn.Sequential(nn.Mish(), nn.Linear(dim, 2 * dim, bias=True))
            nn.init.constant_(self.final_adaLN[-1].weight, 0)
            nn.init.constant_(self.final_adaLN[-1].bias, 0)
        self.to_pixels = nn.Linear(dim, patch_dim)

    def _make_block(self, model_type, dim, heads, layer_idx):
        hp, wp, np_ = self.h_patches, self.w_patches, self.num_patches
        adaln = self.use_adaln

        if model_type == "jit":
            # Paper applies dropout to the middle half of the blocks.
            mid_lo, mid_hi = self.depth // 4, self.depth - self.depth // 4
            drop = self.dropout if (mid_lo <= layer_idx < mid_hi) else 0.0
            return TransformerBlock(dim, heads, dim * 4, hp, wp, use_adaln=adaln,
                                    use_conv_mlp=self.use_conv_mlp, qk_norm=self.use_qk_norm,
                                    dropout=drop)
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
        elif model_type == "convnext":
            return ConvNeXtBlock(dim, hp, wp, drop_path=0.0)
        elif model_type == "fullattn":
            return FullAttentionBlock(dim, hp, wp, heads=heads, use_adaln=adaln, qk_norm=self.use_qk_norm)
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
            return MixerAttnBlock(dim, np_, dim // 2, dim * 4, heads, use_adaln=adaln)
        elif model_type == "gmlp":
            return gMLPBlock_Conv(dim, np_, hp, wp, use_adaln=adaln, tiny_attn=False)
        elif model_type == "amlp":
            return gMLPBlock_Conv(dim, np_, hp, wp, use_adaln=adaln, tiny_attn=True)
        elif model_type == "vim":
            return VisionMambaBlock(dim, dim * 4, hp, wp, use_adaln=adaln)
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

    def forward(self, x, time, x_self_cond=None):
        b, c, h, w = x.shape

        # 1. Patch extraction via unfold
        x_patches = self.unfold(x).transpose(1, 2)  # B, N, patch_dim

        if self.self_cond:
            if x_self_cond is None:
                x_self_cond_patches = torch.zeros_like(x_patches)
            else:
                x_self_cond_patches = self.unfold(x_self_cond).transpose(1, 2)
            x_patches = torch.cat((x_patches, x_self_cond_patches), dim=-1)

        # 2. Embedding
        x = self.to_patch_embedding(x_patches)

        # 3. Positional embedding
        if not self.skip_pos_embed:
            x = x + self.pos_embedding

        # 4. Time embedding
        t = self.time_mlp(time)
        if not self.use_adaln and self.model_type not in ["basemlp"]:
            x = x + t.unsqueeze(1)

        # 5. Blocks (with optional gradient checkpointing)
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, t, use_reentrant=False)
            elif self.model_type == "basemlp":
                x = layer(x, t_emb=t)
            elif self.model_type == "mlp":
                x = x + t.unsqueeze(1)
                x = layer(x)
            else:
                x = layer(x, t_emb=t)

        # 6. Output
        if self.model_type != "basemlp":
            x = self.norm(x)
            if self.use_final_adaln_active:
                shift, scale = self.final_adaLN(t).chunk(2, dim=1)
                x = modulate(x, shift, scale)
        x = self.to_pixels(x)

        # 7. Fold back with overlap normalization
        x = self.fold(x.transpose(1, 2))
        return x / self.overlap_divisor


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
                 model_type="unet", num_res_blocks=1, time_scale=1.0):
        super().__init__()
        self.model_type = model_type
        self.num_res_blocks = num_res_blocks
        self.num_levels = int(math.log2(img_size[0]) - math.log2(bottleneck_res))

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim, scale=time_scale), nn.Linear(dim, time_dim), nn.Mish(), nn.Linear(time_dim, time_dim))

        self.init_conv = nn.Conv2d(channels, dim, 3, padding=1)
        self.downs = nn.ModuleList()
        dims = [dim]
        curr_dim = dim

        for _ in range(self.num_levels):
            out_dim = make_divisible(min(curr_dim * 2, fmap_max), 8)
            layers = nn.ModuleList()
            for __ in range(num_res_blocks):
                if "swin" in model_type:
                    layers.append(SwinBlockAdapter(curr_dim, curr_dim, time_dim, version=model_type.split('_')[1]))
                else:
                    layers.append(ResnetBlock(curr_dim, curr_dim, time_dim))
            self.downs.append(nn.ModuleList([layers, nn.Conv2d(curr_dim, out_dim, 4, stride=2, padding=1)]))
            dims.append(out_dim)
            curr_dim = out_dim

        mid_dim = dims[-1]
        if "swin" in model_type:
            v = model_type.split('_')[1]
            self.mid_block1 = SwinBlockAdapter(mid_dim, mid_dim, time_dim, version=v)
            self.mid_block2 = SwinBlockAdapter(mid_dim, mid_dim, time_dim, version=v)
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
                    layers.append(SwinBlockAdapter(res_in, out_dim, time_dim, version=model_type.split('_')[1]))
                else:
                    layers.append(ResnetBlock(res_in, out_dim, time_dim))
            self.ups.append(nn.ModuleList([nn.ConvTranspose2d(in_dim, in_dim, 2, stride=2), layers]))

        self.is_skip = is_skip
        self.final_norm = nn.GroupNorm(8, dim)
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def forward(self, x, time, x_self_cond=None):
        t = self.time_mlp(time)
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

        return self.final_conv(self.final_norm(x))


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
      * Linear schedule z_t = t*x + (1-t)*eps, with t ~ logit-Normal(mu, sigma).
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
                 t_loc=-0.8, t_scale=0.8, t_clip=0.05, x_clip="none"):
        super().__init__()
        assert pred_mode in ("x", "eps", "v"), f"bad pred_mode {pred_mode!r}"
        assert loss_mode in ("x", "eps", "v"), f"bad loss_mode {loss_mode!r}"
        assert x_clip in ("none", "static", "dynamic"), f"bad x_clip {x_clip!r}"
        self.model = model
        self.pred_mode = pred_mode
        self.loss_mode = loss_mode
        self.loc = t_loc
        self.scale = t_scale
        self.t_clip = t_clip
        self.x_clip = x_clip

    def get_noise_scale(self, h):
        return h / 256.0 if h > 256 else 1.0

    def _convert(self, src, z_t, t_img, clip=None):
        """Interpret `src` as the quantity named by self.pred_mode and return the
        triple (x, eps, v) via the relations of Tab. 1, clamping (1-t)/t by
        `clip` (default self.t_clip) wherever they appear in a denominator."""
        clip = self.t_clip if clip is None else clip
        omt = (1.0 - t_img).clamp(min=clip)   # (1 - t), clamped
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

    def p_losses(self, x_start):
        b, c, h, w = x_start.shape
        device = x_start.device

        noise_scale = self.get_noise_scale(h)
        epsilon = torch.randn_like(x_start) * noise_scale

        # Logit-normal time sampling: logit(t) ~ N(loc, scale^2)
        s = torch.randn(b, device=device) * self.scale + self.loc
        t = torch.sigmoid(s)
        t_img = t.view(b, 1, 1, 1)

        # Linear (rectified-flow) interpolation
        z_t = t_img * x_start + (1.0 - t_img) * epsilon

        net_out = self.model(z_t, t)

        # Derive the predicted triple and the ground-truth triple with the SAME
        # clamped transforms. Sharing the transform guarantees the loss is 0 at a
        # perfect prediction in every space, including inside the clamped region
        # (the source of the original v-target mismatch near t -> 1).
        true_src = {"x": x_start, "eps": epsilon, "v": x_start - epsilon}[self.pred_mode]
        px, pe, pv = self._convert(net_out, z_t, t_img)
        tx, te, tv = self._convert(true_src, z_t, t_img)
        pred, target = {"x": (px, tx), "eps": (pe, te), "v": (pv, tv)}[self.loss_mode]

        return F.mse_loss(pred, target)

    @torch.no_grad()
    def sample(self, shape, steps=50, solver="heun"):
        b, c, h, w = shape
        device = next(self.model.parameters()).device

        z = torch.randn(shape, device=device) * self.get_noise_scale(h)
        timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device)

        def velocity(z_in, t_scalar):
            t_vec = torch.full((b,), t_scalar, device=device)
            t_img = t_vec.view(b, 1, 1, 1)
            net_out = self.model(z_in, t_vec)
            # Use a tiny clamp at inference so the final step fully denoises
            # (training's larger t_clip only exists to bound the loss weight).
            x_pred, _, v_pred = self._convert(net_out, z_in, t_img, clip=1e-5)
            if self.x_clip != "none":
                x_pred = self._clip_x(x_pred)
                v_pred = (x_pred - z_in) / (1.0 - t_img).clamp(min=1e-5)
            return v_pred

        # Heun (default) or Euler integration of dz/dt = v from t=0 to t=1.
        for i in range(steps):
            t0 = timesteps[i].item()
            t1 = timesteps[i + 1].item()
            dt = t1 - t0
            d0 = velocity(z, t0)
            if solver == "euler" or i == steps - 1:
                z = z + dt * d0
            else:
                d1 = velocity(z + dt * d0, t1)
                z = z + dt * 0.5 * (d0 + d1)

        return z


# ==========================================
# 7. Data Pipeline
# ==========================================

# ==========================================
# 7. Data Pipeline
# ==========================================

class ImageDataset(Dataset):
    def __init__(self, folder, image_size, channels=3, augment=False,
                 ext=('jpg', 'jpeg', 'png', 'webp', 'bmp')):
        super().__init__()
        self.channels = channels
        self.paths = []
        for e in ext:
            self.paths.extend(glob.glob(f'{folder}/**/*.{e}', recursive=True))
            self.paths.extend(glob.glob(f'{folder}/*.{e}'))
        self.paths = sorted(set(self.paths))
        print(f"Found {len(self.paths)} images.")

        transforms_list = [T.Resize(image_size), T.CenterCrop(image_size)]
        if augment:
            transforms_list.append(T.RandomHorizontalFlip())
        transforms_list.extend([T.ToTensor(), T.Lambda(lambda t: (t * 2) - 1)])
        self.transform = T.Compose(transforms_list)
        self._mode_map = {3: 'RGB', 1: 'L', 4: 'RGBA'}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Try up to 50 times to load a valid image to prevent infinite hanging
        # in the rare case that the entire directory is corrupted.
        for _ in range(50):
            try:
                img = Image.open(self.paths[index]).convert(self._mode_map.get(self.channels, 'RGB'))
                return self.transform(img)
            except Exception:
                # Skip this image and randomly select another one
                index = random.randint(0, len(self.paths) - 1)
        
        # If it fails 50 times in a row, surface the error
        raise RuntimeError("Dataset seems heavily corrupted; failed to load 50 consecutive random images.")


# ==========================================
# 8. Learning Rate Schedule
# ==========================================

class CosineWarmupScheduler:
    """Linear warmup, then either hold constant (paper default) or cosine-decay."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1, mode="constant"):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.mode = mode
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, current_step):
        if current_step < self.warmup_steps:
            scale = current_step / max(1, self.warmup_steps)
        elif self.mode == "cosine":
            progress = (current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
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
    images = (images.clamp(-1, 1) + 1) * 0.5  # to [0, 1]
    from torchvision.utils import make_grid
    return T.ToPILImage()(make_grid(images, nrow=nrow, padding=2))


def save_checkpoint(model, optimizer, ema, step, path, max_keep=3):
    """Save checkpoint and rotate old ones."""
    save_dict = {
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if ema:
        save_dict['ema'] = ema.shadow.state_dict()
    torch.save(save_dict, path)

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

def get_input(prompt, default=None, cast_type=str):
    user_val = input(f"{prompt} (default: {default}): ").strip()
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
}

IS_CONV = {'encdec', 'unet', 'swin_v1', 'swin_v2'}

NEEDS_HEADS = {'jit', 'fullattn', 'xcit', 'mixer_attn', 'hat', 'coatnet', 'nat', 'volo'}
HYPER_HEADS = {'hypermixer', 'oghypermixer'}
AXIAL_MODELS = {'ogmlpmixer', 'oggmlp', 'ogamlp'}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def clear_old_images(save_dir):
    """Deletes all image files in the save directory, leaving .pt files intact."""
    print(f"🧹 Clearing old samples and generated images from {save_dir}...")
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        for fpath in glob.glob(os.path.join(save_dir, ext)):
            try:
                os.remove(fpath)
            except Exception as e:
                print(f"⚠️ Could not remove {fpath}: {e}")
def main():
    mode_in = input("Select mode: Train (t/0) or Sample (s/1): ").lower().strip()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(SAVE_DIR, "model.pt")
    config_path = os.path.join(SAVE_DIR, "config.pt")
    clear_old_images(SAVE_DIR)

    if mode_in in ['t', '0', 'train']:
        cfg = {}
        loaded = False
        if get_input("Continue training?", "0", bool) and os.path.exists(config_path):
            cfg = torch.load(config_path, weights_only=False)
            loaded = True

        if not loaded:
            cfg['dataset_path'] = get_input("Dataset location", "./data")
            cfg['channels'] = get_input("Channel count", 3, int)
            cfg['width'] = get_input("Image width", 64, int)
            cfg['height'] = get_input("Image height", 64, int)

            print("\n--- Model Selection ---")
            for k, (_, desc) in sorted(MODEL_TYPES.items(), key=lambda x: int(x[0])):
                print(f"  {k}: {desc}")

            m_type_in = get_input("Model type", "1", str)
            # Match by number or name
            if m_type_in in MODEL_TYPES:
                cfg['model_type'] = MODEL_TYPES[m_type_in][0]
            else:
                # Try matching by name
                matched = [v[0] for v in MODEL_TYPES.values() if v[0] == m_type_in]
                cfg['model_type'] = matched[0] if matched else "jit"

            is_conv = cfg['model_type'] in IS_CONV

            if not is_conv:
                cfg['p_width'] = get_input("Patch width", 8, int)
                cfg['p_height'] = get_input("Patch height", 8, int)
                cfg['overlap_w'] = get_input("Patch Width Overlap (pixels)", 0, int)
                cfg['overlap_h'] = get_input("Patch Height Overlap (pixels)", 0, int)

                raw_dim = get_input("Model width (dim)", 256, int)
                cfg['dim'] = make_divisible(raw_dim, 8)
                if cfg['dim'] != raw_dim:
                    print(f"  → Rounded dim to {cfg['dim']}")

                cfg['depth'] = get_input("Model depth", 4, int)

                cfg['heads'] = 1
                if cfg['model_type'] in NEEDS_HEADS:
                    cfg['heads'] = get_input("Attention head count", 4, int)
                elif cfg['model_type'] in HYPER_HEADS:
                    cfg['heads'] = get_input("HyperMixer head count", 1, int)

                cfg['axial'] = False
                if cfg['model_type'] in AXIAL_MODELS:
                    cfg['axial'] = get_input("Use Axial Mixing?", "0", bool)

                cfg['use_adaln'] = get_input("Use AdaLN-Zero?", "1", bool)
                cfg['use_2d_pos_emb'] = get_input("Use 2D Sinusoidal Pos Emb?", "1", bool)
                # Paper's JiT uses a SwiGLU MLP; Conv-MLP is an optional non-paper
                # token-mixing variant, so default it off.
                cfg['use_conv_mlp'] = get_input("Use Conv-MLP (off = paper SwiGLU)?", "0", bool)
                cfg['self_cond'] = get_input("Self-conditioning?", "0", bool)

                # "Just Advanced" Transformer ingredients (paper Sec. 4.4 / Tab. 4).
                cfg['use_qk_norm'] = get_input("Use QK-Norm (Just-Advanced)?", "1", bool)
                cfg['use_final_adaln'] = get_input("Use final adaLN-Zero head (DiT)?", "1", bool)
                # Dropout on the middle-half of blocks (paper: 0 for B/L, 0.2 for H/G).
                cfg['dropout'] = get_input("Dropout (middle-half blocks, 0=off)", 0.0, float)

                use_bn = get_input("Use Bottleneck Patch Embed?", "1", bool)
                cfg['bottleneck_dim'] = get_input("Bottleneck dim", 128, int) if use_bn else None

                cfg['fmap_max'] = 0
                cfg['bottleneck_res'] = 0
            else:
                raw_dim = get_input("Initial Filter Count (dim)", 64, int)
                cfg['dim'] = make_divisible(raw_dim, 8)
                if cfg['dim'] != raw_dim:
                    print(f"  → Rounded dim to {cfg['dim']}")
                cfg['fmap_max'] = get_input("Max Filter Count", 512, int)
                cfg['bottleneck_res'] = get_input("Bottleneck Resolution", 8, int)
                for k in ['p_width', 'p_height', 'depth', 'heads']:
                    cfg[k] = 0
                for k in ['use_adaln', 'use_2d_pos_emb', 'use_conv_mlp', 'self_cond', 'axial',
                          'use_qk_norm', 'use_final_adaln']:
                    cfg[k] = False
                cfg['bottleneck_dim'] = None
                cfg['overlap_h'] = 0
                cfg['overlap_w'] = 0
                cfg['dropout'] = 0.0

            # --- Diffusion objective (applies to every model type) ---
            # Paper's final algorithm = x-prediction trained with a v-loss
            # (Tab. 1(3)(a)). Other spaces are exposed to reproduce Tab. 1/2/3.
            print("\n--- Diffusion objective (Tab. 1 of the JiT paper) ---")
            pm = get_input("Network prediction space [x/eps/v]", "x", str).lower()
            cfg['pred_mode'] = pm if pm in ("x", "eps", "v") else "x"
            lm = get_input("Loss space [x/eps/v]", "v", str).lower()
            cfg['loss_mode'] = lm if lm in ("x", "eps", "v") else "v"
            cfg['t_mu'] = get_input("Logit-normal time mean mu (more negative = more noise)", -0.8, float)
            cfg['t_sigma'] = get_input("Logit-normal time std sigma", 0.8, float)
            xc = get_input("Predicted-x clipping at sampling [none/static/dynamic]", "none", str).lower()
            cfg['x_clip'] = xc if xc in ("none", "static", "dynamic") else "none"
            ls = get_input("LR schedule after warmup [constant/cosine]", "constant", str).lower()
            cfg['lr_schedule'] = ls if ls in ("constant", "cosine") else "constant"
            cfg['time_scale'] = get_input("Time-embedding scale (t in [0,1] -> [0,scale])", 1000.0, float)
            # Paper's bottleneck is a purely-linear low-rank reparameterization.
            cfg['bottleneck_act'] = "none"

            cfg['sampling_steps'] = get_input("Heun Sampling steps", 50, int)
            cfg['batch_size'] = get_input("Batch size", 16, int)
            cfg['lr'] = get_input("Learning rate", 5e-4, float)
            cfg['steps'] = get_input("Step count", 100000, int)
            cfg['use_ema'] = get_input("Use EMA?", "1", bool)
            cfg['use_flip'] = get_input("Use RandFlip Augmentation?", "1", bool)
            cfg['use_amp'] = get_input("Use Mixed Precision (AMP)?", "1", bool)
            cfg['grad_clip'] = get_input("Gradient clip norm (0=off)", 1.0, float)
            cfg['use_grad_ckpt'] = get_input("Use Gradient Checkpointing?", "0", bool)
            cfg['warmup_steps'] = get_input("LR Warmup steps", 1000, int)
            cfg['seed'] = get_input("Random seed (0=random)", 42, int)
            cfg['sample_every'] = get_input("Sample every N steps", 250, int)
            cfg['save_every'] = get_input("Save checkpoint every N steps", 5000, int)
            cfg['num_sample_images'] = get_input("Number of sample images per grid", 4, int)

            # Optional dataset resize & cache
            if get_input("Resize and cache dataset?", "0", bool):
                original_path = cfg['dataset_path']
                resized_path = os.path.join(original_path, "resized")
                exts = ['jpg', 'jpeg', 'png', 'webp', 'bmp']
                files = []
                for e in exts:
                    files.extend(glob.glob(f'{original_path}/**/*.{e}', recursive=True))
                    files.extend(glob.glob(f'{original_path}/*.{e}'))
                files = [f for f in set(files) if "resized" not in f]
                if files:
                    if os.path.exists(resized_path):
                        shutil.rmtree(resized_path)
                    os.makedirs(resized_path)
                    resample = getattr(Image.Resampling, 'LANCZOS', Image.LANCZOS)
                    mode = {3: 'RGB', 1: 'L', 4: 'RGBA'}.get(cfg['channels'], 'RGB')
                    for fpath in tqdm(files, desc="Resizing"):
                        try:
                            with Image.open(fpath) as img:
                                img = img.convert(mode).resize((cfg['width'], cfg['height']), resample)
                                base = os.path.splitext(os.path.basename(fpath))[0]
                                dest = os.path.join(resized_path, f"{base}.png")
                                counter = 1
                                while os.path.exists(dest):
                                    dest = os.path.join(resized_path, f"{base}_{counter}.png")
                                    counter += 1
                                img.save(dest)
                        except Exception:
                            pass
                    cfg['dataset_path'] = resized_path

            # Clean old samples
            #for f in glob.glob(os.path.join(SAVE_DIR, "sample_*.png")):
            #    os.remove(f)

            torch.save(cfg, config_path)

        # --- Apply defaults for legacy configs ---
        # NOTE: architecture-changing additions (qk_norm, final adaLN, time
        # scaling) default to their LEGACY (off / 1.0) values here so that
        # resuming an old checkpoint rebuilds the exact original architecture.
        # New configs created above already carry the paper-faithful values.
        defaults = {
            'use_adaln': False, 'use_ema': False, 'use_flip': False,
            'use_2d_pos_emb': False, 'use_conv_mlp': False, 'self_cond': False,
            'bottleneck_dim': None, 'fmap_max': 512, 'bottleneck_res': 8,
            'overlap_h': 0, 'overlap_w': 0, 'axial': False,
            'use_amp': True, 'grad_clip': 1.0, 'use_grad_ckpt': False,
            'warmup_steps': 1000, 'seed': 42, 'sample_every': 250,
            'save_every': 5000, 'num_sample_images': 4,
            'pred_mode': 'x', 'loss_mode': 'v', 'x_clip': 'none',
            't_mu': -0.8, 't_sigma': 0.8, 'lr_schedule': 'constant',
            'use_qk_norm': False, 'use_final_adaln': False, 'time_scale': 1.0,
            'bottleneck_act': 'mish', 'dropout': 0.0,
        }
        for k, v in defaults.items():
            cfg.setdefault(k, v)

        # Seed
        if cfg['seed'] > 0:
            set_seed(cfg['seed'])

        # --- Build Model ---
        if cfg['model_type'] in IS_CONV:
            model = ConvNetModel(
                img_size=(cfg['height'], cfg['width']),
                channels=cfg['channels'], dim=cfg['dim'],
                fmap_max=cfg['fmap_max'], bottleneck_res=cfg['bottleneck_res'],
                model_type=cfg['model_type'], time_scale=cfg['time_scale'],
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
                time_scale=cfg['time_scale'],
                bottleneck_act=cfg['bottleneck_act'],
                dropout=cfg['dropout'],
            ).to(device)

        print(f"📊 Model: {cfg['model_type']} | Parameters: {count_parameters(model):,}")
        print(f"🎯 Objective: {cfg['pred_mode']}-pred / {cfg['loss_mode']}-loss | "
              f"t~logitN(mu={cfg['t_mu']}, sigma={cfg['t_sigma']}) | x_clip={cfg['x_clip']}")

        ema = EMA(model) if cfg['use_ema'] else None
        flow_model = FlowMatchingWrapper(
            model, pred_mode=cfg['pred_mode'], loss_mode=cfg['loss_mode'],
            t_loc=cfg['t_mu'], t_scale=cfg['t_sigma'], x_clip=cfg['x_clip'],
        ).to(device)
        optimizer = Adan(model.parameters(), lr=cfg['lr'])#Prodigy(model.parameters(), lr=cfg['lr'], weight_decay=0.0, betas=(0.9, 0.95), slice_p=11)#CAdamax(model.parameters(), lr=cfg['lr'], weight_decay=0.05, betas=(0.9, 0.95))

        scheduler = CosineWarmupScheduler(
            optimizer, warmup_steps=cfg['warmup_steps'],
            total_steps=cfg['steps'], min_lr_ratio=0.1, mode=cfg['lr_schedule'],
        )

        start_step = 0
        if loaded and os.path.exists(model_path):
            ckpt = torch.load(model_path, weights_only=False)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            if ema and 'ema' in ckpt:
                ema.shadow.load_state_dict(ckpt['ema'])
            start_step = ckpt.get('step', 0)
            print(f"▶ Resuming from step {start_step}")

        ds = ImageDataset(cfg['dataset_path'], (cfg['height'], cfg['width']),
                          channels=cfg['channels'], augment=cfg['use_flip'])
        if len(ds) == 0:
            print("❌ No images found. Exiting.")
            return
        dl = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=min(4, os.cpu_count() or 1),
                        pin_memory=True, prefetch_factor=2,
                        drop_last=True, persistent_workers=True)
        dl_iter = cycle(dl)

        scaler = GradScaler(enabled=cfg['use_amp'])

        global TRAINING_ACTIVE
        TRAINING_ACTIVE = True

        # Loss tracking
        loss_ema = 0.0
        loss_ema_decay = 0.99

        pbar = tqdm(range(start_step, cfg['steps']), initial=start_step, total=cfg['steps'])
        for step in pbar:
            if interrupted:
                break

            data = next(dl_iter).to(device)

            # LR schedule (linear warmup, then constant per the paper unless
            # cfg['lr_schedule'] == 'cosine')
            scheduler.step(step)

            # Forward pass with optional AMP
            with torch.amp.autocast('cuda', enabled=cfg['use_amp']):
                loss = flow_model.p_losses(data)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # Gradient clipping
            if cfg['grad_clip'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])

            scaler.step(optimizer)
            scaler.update()

            if ema:
                ema.update(model)

            # Smoothed loss tracking
            loss_val = loss.item()
            loss_ema = loss_ema * loss_ema_decay + loss_val * (1 - loss_ema_decay) if step > start_step else loss_val
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_description(f"Loss: {loss_val:.4f} | Avg: {loss_ema:.4f} | LR: {current_lr:.2e}")

            # --- Sampling ---
            if step > 0 and step % cfg['sample_every'] == 0:
                inference_model = ema.shadow if ema else model
                inference_model.eval()
                temp_flow = FlowMatchingWrapper(
                    inference_model, pred_mode=cfg['pred_mode'], loss_mode=cfg['loss_mode'],
                    t_loc=cfg['t_mu'], t_scale=cfg['t_sigma'], x_clip=cfg['x_clip'],
                ).to(device)
                n_samples = cfg['num_sample_images']
                samples = temp_flow.sample(
                    (n_samples, cfg['channels'], cfg['height'], cfg['width']),
                    steps=cfg['sampling_steps']
                )
                grid = make_sample_grid(samples.cpu(), nrow=int(math.ceil(math.sqrt(n_samples))))
                grid.save(f"{SAVE_DIR}/sample_{step}.png")
                inference_model.train()

            # --- Checkpointing ---
            if step > 0 and step % cfg['save_every'] == 0:
                save_checkpoint(model, optimizer, ema, step, model_path, max_keep=1)

        # Final save
        save_checkpoint(model, optimizer, ema, step, model_path, max_keep=1)
        TRAINING_ACTIVE = False
        print("✅ Training complete.")

    elif mode_in in ['s', '1', 'sample']:
        if not os.path.exists(config_path):
            print("No config found. Train first.")
            return
        cfg = torch.load(config_path, weights_only=False)

        defaults = {
            'use_adaln': False, 'use_ema': False, 'use_flip': False,
            'use_2d_pos_emb': False, 'use_conv_mlp': False, 'self_cond': False,
            'bottleneck_dim': None, 'fmap_max': 512, 'bottleneck_res': 8,
            'overlap_h': 0, 'overlap_w': 0, 'axial': False,
            'pred_mode': 'x', 'loss_mode': 'v', 'x_clip': 'none',
            't_mu': -0.8, 't_sigma': 0.8, 'use_qk_norm': False,
            'use_final_adaln': False, 'time_scale': 1.0, 'bottleneck_act': 'mish',
            'dropout': 0.0,
        }
        for k, v in defaults.items():
            cfg.setdefault(k, v)

        count = get_input("Image count", 4, int)
        steps = get_input("Heun Steps", cfg.get('sampling_steps', 50), int)
        seed = get_input("Seed (0=random)", 0, int)
        batch_size = get_input("Batch size for sampling", min(count, 4), int)
        # Allow overriding predicted-x clipping at inference without retraining.
        xc = get_input("Predicted-x clipping [none/static/dynamic]", cfg['x_clip'], str).lower()
        cfg['x_clip'] = xc if xc in ("none", "static", "dynamic") else cfg['x_clip']

        #for f in glob.glob(os.path.join(SAVE_DIR, "generated_*.png")):
        #    os.remove(f)

        if seed > 0:
            set_seed(seed)

        # Build model (FIX: pass all config params including overlap/axial)
        if cfg['model_type'] in IS_CONV:
            model = ConvNetModel(
                img_size=(cfg['height'], cfg['width']),
                channels=cfg['channels'], dim=cfg['dim'],
                fmap_max=cfg['fmap_max'], bottleneck_res=cfg['bottleneck_res'],
                model_type=cfg['model_type'], time_scale=cfg['time_scale'],
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
                overlap_h=cfg['overlap_h'],
                overlap_w=cfg['overlap_w'],
                axial=cfg['axial'],
                use_qk_norm=cfg['use_qk_norm'],
                use_final_adaln=cfg['use_final_adaln'],
                time_scale=cfg['time_scale'],
                bottleneck_act=cfg['bottleneck_act'],
            ).to(device)

        ckpt = torch.load(model_path, weights_only=False)
        if cfg['use_ema'] and 'ema' in ckpt:
            print("Loading EMA weights...")
            model.load_state_dict(ckpt['ema'])
        else:
            model.load_state_dict(ckpt['model'])

        model.eval()
        flow_model = FlowMatchingWrapper(
            model, pred_mode=cfg['pred_mode'], loss_mode=cfg['loss_mode'],
            t_loc=cfg['t_mu'], t_scale=cfg['t_sigma'], x_clip=cfg['x_clip'],
        ).to(device)

        generated = 0
        while generated < count:
            bs = min(batch_size, count - generated)
            out = flow_model.sample((bs, cfg['channels'], cfg['height'], cfg['width']), steps=steps)
            for j in range(bs):
                img = T.ToPILImage()((out[j].cpu().clamp(-1, 1) + 1) * 0.5)
                img.save(f"{SAVE_DIR}/generated_{generated}.png")
                generated += 1
                print(f"  Saved generated_{generated - 1}.png")

        # Also save a grid
        if count > 1:
            all_imgs = []
            for i in range(count):
                all_imgs.append(T.ToTensor()(Image.open(f"{SAVE_DIR}/generated_{i}.png")))
            from torchvision.utils import make_grid
            grid = T.ToPILImage()(make_grid(torch.stack(all_imgs), nrow=int(math.ceil(math.sqrt(count))), padding=2))
            grid.save(f"{SAVE_DIR}/generated_grid.png")
            print(f"  Saved generated_grid.png")

        print("✅ Sampling complete.")


if __name__ == "__main__":
    main()
