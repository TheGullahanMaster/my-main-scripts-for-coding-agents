"""Pixel-space adaptations of FCDM (arXiv:2603.09408).

Architecture reference: https://github.com/star-kwon/FCDM
No VAE or learned-variance head: JiT's wrapper supplies the training objective,
label dropout and classifier-free guidance. Isotropic blocks are also usable in
JiT's configurable patch/stem shell.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from jit5_kernels import fcdm_gelu_grn, fcdm_kernels_available, fcdm_ln_modulate


class FCDMGRN(nn.Module):
    """Spatial response normalization, with FP32 reductions under mixed precision."""
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        response = torch.linalg.vector_norm(x, dim=(2, 3), keepdim=True,
                                            dtype=torch.promote_types(x.dtype, torch.float32))
        relative = (response / (response.mean(1, keepdim=True) + 1e-6)).to(x.dtype)
        return x + self.gamma.to(x.dtype) * x * relative + self.beta.to(x.dtype)

    def forward_nhwc(self, x):
        """Same equation on B, H, W, C; x*(1 + gamma*r) + beta as one fused kernel."""
        response = torch.linalg.vector_norm(x, dim=(1, 2), keepdim=True,
                                            dtype=torch.promote_types(x.dtype, torch.float32))
        relative = (response / (response.mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
        gamma = self.gamma.view(1, 1, 1, -1).to(x.dtype)
        return torch.addcmul(self.beta.view(1, 1, 1, -1).to(x.dtype), x, 1 + gamma * relative)


class FCDMNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=False)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def modulation(channels, count):
    result = nn.Sequential(nn.SiLU(), nn.Linear(channels, count * channels))
    nn.init.zeros_(result[-1].weight)
    nn.init.zeros_(result[-1].bias)
    return result


class FCDMBlock(nn.Module):
    def __init__(self, channels, ratio=3.):
        super().__init__()
        expanded = int(channels * ratio)
        self.dwconv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = FCDMNorm(channels)
        self.expand = nn.Conv2d(channels, expanded, 1)
        self.grn = FCDMGRN(expanded)
        self.contract = nn.Conv2d(expanded, channels, 1)
        self.adaLN_modulation = modulation(channels, 3)

    def forward(self, x, context):
        # Channels-last: the depthwise conv takes cuDNN's NHWC path, the
        # permute to B, H, W, C is a free view, and the 1x1 convs run as
        # linears on it (same weights, so checkpoints are unchanged).
        # Elementwise chains are fused with addcmul; this is memory-bound.
        shift, scale, gate = self.adaLN_modulation(context).to(x.dtype).chunk(3, -1)
        h = self.dwconv(x.contiguous(memory_format=torch.channels_last)).permute(0, 2, 3, 1)
        if fcdm_kernels_available(h):
            # Eager CUDA: fused Triton LayerNorm+modulation and GELU+GRN
            # (jit5_kernels). torch.compile fuses the PyTorch path itself.
            h = fcdm_ln_modulate(h, shift, scale, self.norm.norm.eps)
            h = fcdm_gelu_grn(F.linear(h, self.expand.weight.flatten(1), self.expand.bias),
                              self.grn.gamma, self.grn.beta)
        else:
            h = F.layer_norm(h, h.shape[-1:], eps=self.norm.norm.eps)
            h = torch.addcmul(shift[:, None, None, :], h, 1 + scale[:, None, None, :])
            h = self.grn.forward_nhwc(F.gelu(F.linear(h, self.expand.weight.flatten(1), self.expand.bias)))
        h = F.linear(h, self.contract.weight.flatten(1), self.contract.bias)
        return torch.addcmul(x, gate[:, :, None, None], h.permute(0, 3, 1, 2))


class FCDMTokenBlock(FCDMBlock):
    def __init__(self, dim, height, width, ratio=3.):
        super().__init__(dim, ratio)
        self.height, self.width = height, width

    def forward(self, x, t_emb=None):
        if t_emb is None:
            raise ValueError('FCDM requires timestep conditioning')
        # B, N, C tokens are already channels-last: both reshapes are views.
        b, n, c = x.shape
        image = x.reshape(b, self.height, self.width, c).permute(0, 3, 1, 2)
        return super().forward(image, t_emb).permute(0, 2, 3, 1).reshape(b, n, c)


class FCDMTimeEmbedding(nn.Module):
    def __init__(self, channels, time_scale):
        super().__init__()
        self.time_scale = time_scale
        self.mlp = nn.Sequential(nn.Linear(256, channels), nn.SiLU(), nn.Linear(channels, channels))

    def forward(self, t):
        frequencies = torch.exp(-math.log(10000) * torch.arange(128, device=t.device).float() / 128)
        angles = t.float()[:, None] * self.time_scale * frequencies[None]
        return self.mlp(torch.cat((angles.cos(), angles.sin()), -1).to(self.mlp[0].weight.dtype))


class FCDMUNet(nn.Module):
    """Three resolutions; widths C/2C/4C and stage depths L/2L/4L/2L/L."""
    def __init__(self, channels, dim, depth, ratio=3., time_scale=1000.,
                 input_channels=None, class_count=0, class_cfg=False,
                 cond_residual=False, use_gradient_checkpointing=False):
        super().__init__()
        self.model_type = 'fcdm_unet'
        self.channels = channels
        self.input_channels = input_channels or channels
        self.cond_residual = cond_residual and self.input_channels == 2 * channels
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.null_class_id = class_count if class_count and class_cfg else None
        widths = [dim, 2 * dim, 4 * dim]
        self.time_embeddings = nn.ModuleList([FCDMTimeEmbedding(c, time_scale) for c in widths])
        self.class_embeddings = nn.ModuleList([
            nn.Embedding(class_count + int(class_cfg), c) for c in widths
        ]) if class_count else nn.ModuleList()
        self.input_embedding = nn.Conv2d(self.input_channels, dim, 3, padding=1)
        self.stages = nn.ModuleList([
            nn.ModuleList([FCDMBlock(c, ratio) for _ in range(n)])
            for c, n in zip([dim, 2*dim, 4*dim, 2*dim, dim],
                            [depth, 2*depth, 4*depth, 2*depth, depth])
        ])
        self.down = nn.ModuleList([
            nn.Sequential(nn.Conv2d(a, b // 4, 3, padding=1, bias=False), nn.PixelUnshuffle(2))
            for a, b in zip(widths, widths[1:])
        ])
        self.up = nn.ModuleList([
            nn.Sequential(nn.Conv2d(a, b * 4, 3, padding=1, bias=False), nn.PixelShuffle(2))
            for a, b in [(4*dim, 2*dim), (2*dim, dim)]
        ])
        self.merge = nn.ModuleList([nn.Conv2d(4*dim, 2*dim, 1), nn.Conv2d(2*dim, dim, 1)])
        self.output_conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.final_norm = FCDMNorm(dim)
        self.final_modulation = modulation(dim, 2)
        self.final_conv = nn.Conv2d(dim, channels, 3, padding=1)
        self.apply(self._initialize)
        for m in self.modules():
            if isinstance(m, FCDMBlock):
                nn.init.zeros_(m.adaLN_modulation[-1].weight)
                nn.init.zeros_(m.adaLN_modulation[-1].bias)
        for embedding in self.class_embeddings:
            nn.init.normal_(embedding.weight, std=.02)
        for embedding in self.time_embeddings:
            for m in (embedding.mlp[0], embedding.mlp[2]):
                nn.init.normal_(m.weight, std=.02)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    @staticmethod
    def _initialize(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, time, x_self_cond=None, class_labels=None):
        if x.shape[-2] % 4 or x.shape[-1] % 4:
            raise ValueError('FCDM-UNet image dimensions must be divisible by 4')
        contexts = [embedding(time) for embedding in self.time_embeddings]
        if self.class_embeddings:
            if class_labels is None:
                raise ValueError('Class-conditioned FCDM requires class labels')
            contexts = [c + emb(class_labels) for c, emb in zip(contexts, self.class_embeddings)]
        residual = x[:, self.channels:] if self.cond_residual else None
        h = self.input_embedding(x).contiguous(memory_format=torch.channels_last)
        skips = []
        for level, blocks in enumerate(self.stages):
            if level in (1, 2):
                h = self.down[level - 1](h)
            elif level in (3, 4):
                h = self.merge[level - 3](torch.cat((self.up[level - 3](h), skips.pop()), 1))
            context = contexts[min(level, 4 - level)]
            for block in blocks:
                h = (checkpoint(block, h, context, use_reentrant=False)
                     if self.use_gradient_checkpointing and self.training else block(h, context))
            if level < 2:
                skips.append(h)
        h = self.final_norm(self.output_conv(h))
        shift, scale = self.final_modulation(contexts[0]).to(h.dtype).chunk(2, -1)
        y = self.final_conv(h * (1 + scale[:, :, None, None]) + shift[:, :, None, None])
        return y + residual if residual is not None else y
