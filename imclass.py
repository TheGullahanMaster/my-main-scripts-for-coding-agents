"""Interactive classifier, transfer learning, CAM and activation maximization.

See IMCLASS.md for the input contract and architecture/visualization semantics.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import secrets
import signal
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# Configure this before importing torch or probing CUDA: cudaGetDeviceCount
# can initialize the driver before torch.cuda._lazy_init sets its own default.
# Honor an explicit setting supplied by the caller.
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')

import torch
from PIL import Image, ImageDraw, ImageOps
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.models.efficientnet import MBConv, MBConvConfig
from torchvision.ops import StochasticDepth, SqueezeExcitation


IMAGE_EXTENSIONS = {'.bmp', '.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff', '.gif'}
MODEL_NAMES = ['Basic ConvNet', 'ResNet', 'EfficientNet family', 'Basic ViT',
               'DeiT III style ViT', 'MLP-Mixer', 'gMLP', 'aMLP',
               'Hierarchical ConvNeXt', 'Isotropic ConvNeXt', 'PatchRNN', 'ViP (Vision Permutator)']
ISOTROPIC = {3, 4, 5, 6, 7, 9, 10, 11}
OPTIMIZER_LRS = (0.01, 0.001, 0.00042, 0.0001, 0.0025)
DREAM_SETUPS = ('direct pixels', 'FFT')
DREAM_OPTIMIZERS = ('Adam', 'L-BFGS', 'normalized gradient ascent')
AUGMENTS = {
    1: 'horizontal flip', 2: 'vertical flip', 3: 'rotation (15 degrees)',
    4: 'affine translate/scale/shear', 5: 'perspective', 6: 'color jitter',
    7: 'grayscale', 8: 'Gaussian blur', 9: 'solarize', 10: 'posterize',
    11: 'autocontrast', 12: 'equalize', 13: 'invert', 14: 'RandAugment',
    15: 'TrivialAugmentWide', 16: 'AugMix', 17: 'random erasing',
    18: 'Gaussian noise', 19: 'MixUp', 20: 'CutMix',
}


@dataclass
class Config:
    channels: int = 3
    imres: int = 32
    crop: int = 32
    model: int = 0
    min_dim: int = 32
    max_dim: int = 128
    patch: int = 4
    dim: int = 128
    depth: int = 4
    heads: int = 4
    rnn_cell: str = 'gru'
    use_grn: bool = False
    head_dims: list[int] = field(default_factory=lambda: [128])
    activation: int = 7
    slope: float = 0.01

    @property
    def size(self):
        return self.crop if self.imres == -1 else min(self.imres, self.crop)

    def validate(self):
        if self.channels not in (1, 2, 3, 4):
            raise ValueError('Channel count must be 1, 2, 3 or 4 (L/LA/RGB/RGBA).')
        if self.imres != -1 and self.imres < 1:
            raise ValueError('Image resolution must be -1 or positive.')
        if self.crop < 1 or self.model not in range(len(MODEL_NAMES)):
            raise ValueError('Invalid crop size or model ID.')
        if self.min_dim < 2 or self.max_dim < self.min_dim:
            raise ValueError('Stage widths require 2 <= minimum <= maximum.')
        if self.dim < 2 or self.depth < 1 or self.patch < 1:
            raise ValueError('Hidden width >= 2, depth >= 1 and patch >= 1 required.')
        if self.model in ISOTROPIC and (self.patch > self.size or self.size % self.patch):
            raise ValueError('Effective image/crop size must be divisible by patch size.')
        if self.model in (3, 4) and (self.heads < 1 or self.dim % self.heads):
            raise ValueError('ViT hidden width must be divisible by head count.')
        if self.model == 10 and self.rnn_cell not in ('rnn', 'gru', 'lstm'):
            raise ValueError('PatchRNN cell must be rnn, gru or lstm.')
        if self.model == 11 and self.dim % (self.size // self.patch):
            raise ValueError('ViP width must be divisible by the patch grid side.')
        if any(d < 1 for d in self.head_dims) or self.activation not in range(10):
            raise ValueError('Invalid MLP head configuration.')
        if not math.isfinite(self.slope) or self.slope < 0:
            raise ValueError('LeakyReLU slope must be finite and nonnegative.')


def seed_everything(seed):
    seed = seed or secrets.randbelow(2**31 - 1) + 1
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed


def image_files(root):
    root = Path(root).expanduser().resolve()
    if root.is_file():
        if root.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f'Unsupported image extension: {root}')
        return [root]
    if not root.is_dir():
        raise ValueError(f'Image path does not exist: {root}')
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def scan_classes(root, classes=None):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'Data directory does not exist: {root}')
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    training = classes is None
    classes = [p.name for p in folders] if training else list(classes)
    if training and len(classes) < 2:
        raise ValueError('Data must contain at least two class subdirectories.')
    records = []
    for folder in folders:
        if folder.name not in classes:
            raise ValueError(f'Unknown validation class: {folder.name}')
        paths = image_files(folder)
        if training and not paths:
            raise ValueError(f'Class has no supported images: {folder}')
        records.extend((p.resolve(), classes.index(folder.name)) for p in paths)
    if not records:
        raise ValueError('No supported images found.')
    return records, classes


DATASET_FORMAT = 'imclass-dataset'


def load_dataset_json(path, classes=None):
    """Records from a dataset JSON: {"format": "imclass-dataset", "version": 1, "classes": [...],
    "root": optional base folder, "items": [{"path": ..., "class": ...}, ...]}.
    Relative paths resolve against "root", else the JSON file's folder. Items without a class
    are unlabeled and skipped; missing files are skipped with a warning."""
    path = Path(path).expanduser().resolve()
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read dataset JSON {path}: {exc}') from exc
    if not isinstance(data, dict) or data.get('format') != DATASET_FORMAT or not isinstance(data.get('items'), list):
        raise ValueError(f'{path} is not an imclass dataset JSON (format "{DATASET_FORMAT}").')
    root = Path(data.get('root') or path.parent).expanduser()
    root = (path.parent / root).resolve() if not root.is_absolute() else root.resolve()
    training = classes is None
    names = [str(c) for c in data.get('classes') or []]
    classes = names if training else list(classes)
    if training and (len(classes) < 2 or len(set(classes)) != len(classes)):
        raise ValueError('Dataset JSON must list at least two distinct classes.')
    records, missing, seen = [], 0, set()
    for item in data['items']:
        label = item.get('class') if isinstance(item, dict) else None
        if label in (None, ''):
            continue
        if str(label) not in classes:
            raise ValueError(f'Unknown {"validation " if not training else ""}class in {path}: {label}')
        p = Path(str(item['path'])).expanduser()
        p = (p if p.is_absolute() else root / p).resolve()
        if p.suffix.lower() not in IMAGE_EXTENSIONS or not p.is_file():
            missing += 1
            continue
        if p in seen:
            continue
        seen.add(p)
        records.append((p, classes.index(str(label))))
    if missing:
        print(f'Warning: {missing} dataset entries point to missing or unsupported files; skipped.', flush=True)
    if training:
        empty = [c for i, c in enumerate(classes) if not any(y == i for _, y in records)]
        if empty:
            raise ValueError(f'Classes without images in {path.name}: {empty}')
    if not records:
        raise ValueError('No labeled images found.')
    return records, classes


def save_dataset_json(path, classes, items, root=None):
    """Write a dataset JSON; items are (path, class name or None). Paths under root are stored relative."""
    path = Path(path).expanduser().resolve()
    root = Path(root).expanduser().resolve() if root else path.parent
    out = []
    for p, label in items:
        p = Path(p).expanduser().resolve()
        try:
            stored = p.relative_to(root).as_posix()
        except ValueError:
            stored = str(p)
        out.append({'path': stored, 'class': label})
    payload = {'format': DATASET_FORMAT, 'version': 1, 'classes': list(classes),
               'root': str(root) if root != path.parent else '', 'items': out}
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=1))
    temporary.replace(path)
    return path


def load_records(source, classes=None):
    """A class-subfolder directory or a dataset .json file."""
    source = Path(source).expanduser()
    if source.suffix.lower() == '.json':
        return load_dataset_json(source, classes)
    return scan_classes(source, classes)


def split_records(records, fraction, seed):
    if not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError('Validation fraction must satisfy 0 <= fraction < 1.')
    if fraction == 0:
        return list(records), []
    rng = random.Random(seed)
    train, val = [], []
    for label in sorted({y for _, y in records}):
        group = [r for r in records if r[1] == label]
        if len(group) < 2:
            raise ValueError('A positive validation fraction requires two images per class.')
        rng.shuffle(group)
        n = min(len(group) - 1, max(1, round(len(group) * fraction)))
        val.extend(group[:n])
        train.extend(group[n:])
    return train, val


def parse_augments(raw):
    raw = raw.strip()
    if raw in ('', '0'):
        return []
    if raw == '-1':
        return list(AUGMENTS)
    try:
        values = sorted({int(x.strip()) for x in raw.split(',')})
    except ValueError as exc:
        raise ValueError('Enter comma-separated augmentation IDs.') from exc
    if not values or any(v not in AUGMENTS for v in values):
        raise ValueError('Use IDs 1–20; 0 and -1 must be used alone.')
    return values


class Preprocess:
    def __init__(self, cfg, training=False, augments=()):
        cfg.validate()
        self.cfg, self.training = cfg, training
        self.augments = set(augments) if training else set()
        geometric = {
            1: T.RandomHorizontalFlip(), 2: T.RandomVerticalFlip(),
            3: T.RandomRotation(15),
            4: T.RandomAffine(0, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10),
            5: T.RandomPerspective(distortion_scale=0.2, p=0.5),
        }
        photo = {
            6: T.ColorJitter(0.3, 0.3, 0.3, 0.05), 7: T.RandomGrayscale(0.2),
            8: T.RandomApply([T.GaussianBlur(3, (0.1, 1.5))], p=0.3),
            9: T.RandomSolarize(128), 10: T.RandomPosterize(4),
            11: T.RandomAutocontrast(), 12: T.RandomEqualize(), 13: T.RandomInvert(0.1),
        }
        self.geometric = [v for k, v in geometric.items() if k in self.augments]
        self.photo = [v for k, v in photo.items() if k in self.augments]
        # Automatic policies include geometry. Replay their random choices on
        # alpha, using the same channel layout to keep policy RNG consumption equal.
        auto = {14: T.RandAugment(), 15: T.TrivialAugmentWide(), 16: T.AugMix()}
        self.auto = [v for k, v in auto.items() if k in self.augments]

    def __call__(self, image):
        c = self.cfg
        image = ImageOps.exif_transpose(image).convert({1: 'L', 2: 'LA', 3: 'RGB', 4: 'RGBA'}[c.channels])
        if c.imres != -1:
            image = TF.resize(image, [c.imres, c.imres])
        elif min(image.size) < c.crop:
            ratio = c.crop / min(image.size)
            image = TF.resize(image, [math.ceil(image.height * ratio), math.ceil(image.width * ratio)])
        if c.imres == -1 or c.crop < c.imres:
            if self.training:
                top = random.randrange(image.height - c.crop + 1)
                left = random.randrange(image.width - c.crop + 1)
                image = TF.crop(image, top, left, c.crop, c.crop)
            else:
                image = TF.center_crop(image, [c.crop, c.crop])
        for operation in self.geometric:
            image = operation(image)
        alpha = image.getchannel('A') if c.channels in (2, 4) else None
        color = image.convert('L' if c.channels < 3 else 'RGB')
        for operation in self.photo:
            color = operation(color)
        for operation in self.auto:
            before = torch.get_rng_state()
            color = operation(color)
            if alpha is not None:
                after = torch.get_rng_state()
                torch.set_rng_state(before)
                alpha = operation(alpha.convert('L' if c.channels < 3 else 'RGB')).convert('L')
                torch.set_rng_state(after)
        if alpha is not None:
            color.putalpha(alpha)
        x = TF.pil_to_tensor(color).float().div(255)
        if 17 in self.augments:
            x = T.RandomErasing(p=0.3, value=0.5)(x)
        if 18 in self.augments and torch.rand(()).item() < 0.5:
            x = (x + torch.randn_like(x) * 0.04).clamp(0, 1)
        return x.mul(2).sub(1)


class ImageDataset(Dataset):
    def __init__(self, records, transform):
        self.records, self.transform = records, transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        try:
            with Image.open(path) as image:
                x = self.transform(image)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f'Cannot load training image {path}: {exc}') from exc
        return x, label


def balanced_sampler(records, classes, seed):
    counts = torch.bincount(torch.tensor([y for _, y in records]), minlength=len(classes))
    if (counts == 0).any():
        raise ValueError('Every class must have training images.')
    class_weights = counts.double().reciprocal()
    weights = class_weights[torch.tensor([y for _, y in records])]
    return (WeightedRandomSampler(weights, len(records), replacement=True,
                                  generator=torch.Generator().manual_seed(seed)),
            counts.tolist(), class_weights.tolist())


class ChannelNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def conv_unit(a, b, stride=1):
    return nn.Sequential(nn.Conv2d(a, b, 3, stride, 1, bias=False),
                         nn.GroupNorm(1, b), nn.SiLU())


class Residual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, bias=False),
                                  nn.GroupNorm(1, dim), nn.ReLU(),
                                  nn.Conv2d(dim, dim, 3, padding=1, bias=False),
                                  nn.GroupNorm(1, dim))
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.body(x))


class ConvNextBlock(nn.Module):
    def __init__(self, dim, drop=0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = ChannelNorm(dim)
        self.expand = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.project = nn.Conv2d(4 * dim, dim, 1)
        self.scale = nn.Parameter(torch.full((dim,), 1e-6))
        self.drop = StochasticDepth(drop, 'row')

    def forward(self, x):
        return x + self.drop(self.project(self.act(self.expand(self.norm(self.depthwise(x))))) * self.scale[None, :, None, None])


class ImageGRN(nn.Module):
    """FCDM-style channel response normalization for channel-first feature maps."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        response = torch.linalg.vector_norm(x.float(), dim=(2, 3), keepdim=True)
        relative = (response / (response.mean(1, keepdim=True) + 1e-6)).to(x.dtype)
        return x + self.gamma.to(x.dtype) * x * relative + self.beta.to(x.dtype)


class Hierarchical(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dims = [cfg.min_dim]
        while dims[-1] < cfg.max_dim:
            dims.append(min(dims[-1] * 2, cfg.max_dim))
        layers, previous = [], cfg.channels
        for i, dim in enumerate(dims):
            if cfg.model == 8:
                layers += [nn.Conv2d(previous, dim, 3, 1 if i == 0 else 2, 1), ChannelNorm(dim)]
                layers += [ConvNextBlock(dim, 0.05) for _ in range(2)]
            else:
                layers.append(conv_unit(previous, dim, 1 if i == 0 else 2))
                for _ in range(2):
                    if cfg.model == 0:
                        layers.append(conv_unit(dim, dim))
                    elif cfg.model == 1:
                        layers.append(Residual(dim))
                    else:
                        config = MBConvConfig(6, 3, 1, dim, dim, 1)
                        # Preserve the exact requested stage widths rather than
                        # MBConvConfig's automatic multiple-of-eight rounding.
                        config.input_channels = config.out_channels = dim
                        layers.append(MBConv(config, 0.05, lambda d: nn.GroupNorm(1, d)))
            previous = dim
        self.layers = nn.Sequential(*layers)
        self.grn = ImageGRN(dims[-1]) if cfg.use_grn else nn.Identity()
        self.features = nn.Identity()
        self.out_dim = dims[-1]

    def forward(self, x):
        return self.features(self.grn(self.layers(x))).mean((2, 3))


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim=None, out_dim=None):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim or dim // heads
        self.qkv = nn.Linear(dim, 3 * heads * self.head_dim)
        self.proj = nn.Linear(heads * self.head_dim, out_dim or dim)
        self.capture = False
        self.last_attention = None

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        if self.capture:
            weights = (q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)).softmax(-1)
            self.last_attention = weights.detach().cpu()
            y = weights @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(b, n, -1))


def feedforward(dim, hidden):
    return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, modern=False):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = Attention(dim, heads)
        self.mlp = feedforward(dim, 4 * dim)
        self.scale1 = nn.Parameter(torch.full((dim,), 1e-4)) if modern else 1.0
        self.scale2 = nn.Parameter(torch.full((dim,), 1e-4)) if modern else 1.0
        self.drop = StochasticDepth(0.1 if modern else 0.0, 'row')

    def forward(self, x):
        x = x + self.drop(self.attention(self.norm1(x)) * self.scale1)
        return x + self.drop(self.mlp(self.norm2(x)) * self.scale2)


class TokenGRN(nn.Module):
    """FCDM-style channel response normalization across a patch-token grid."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        response = torch.linalg.vector_norm(x.float(), dim=1, keepdim=True)
        relative = (response / (response.mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
        return x + self.gamma.to(x.dtype) * x * relative + self.beta.to(x.dtype)


class GRNResidualBlock(nn.Module):
    """Normalize a block's update; zero initialization preserves its initial mapping."""
    def __init__(self, block, dim, channels_first=False):
        super().__init__()
        self.block = block
        self.grn = ImageGRN(dim) if channels_first else TokenGRN(dim)

    def forward(self, x):
        return x + self.grn(self.block(x) - x)


class MixerBlock(nn.Module):
    def __init__(self, dim, tokens):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.token_mlp = feedforward(tokens, max(8, tokens // 2))
        self.channel_mlp = feedforward(dim, 4 * dim)
        self.capture, self.last_response = False, None

    def forward(self, x):
        mixed = self.token_mlp(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        if self.capture:
            self.last_response = mixed.detach().abs().mean(-1).cpu()
        x = x + mixed
        return x + self.channel_mlp(self.norm2(x))


class GatedBlock(nn.Module):
    def __init__(self, dim, tokens, attention=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.gate_norm = nn.LayerNorm(2 * dim)
        self.spatial = nn.Linear(tokens, tokens)
        nn.init.normal_(self.spatial.weight, std=1e-3)
        nn.init.ones_(self.spatial.bias)
        self.attention = Attention(dim, 1, head_dim=64, out_dim=2 * dim) if attention else None
        self.project = nn.Linear(2 * dim, dim)
        self.capture, self.last_gate = False, None

    def forward(self, x):
        z = self.norm(x)
        u, v = self.act(self.expand(z)).chunk(2, -1)
        gate = self.spatial(self.gate_norm(v).transpose(1, 2)).transpose(1, 2)
        if self.attention is not None:
            gate = gate + self.attention(z)
        if self.capture:
            self.last_gate = gate.detach().abs().mean(-1).cpu()
        return x + self.project(u * gate)


class PatchRecurrent(nn.Module):
    """Raster-order patch recurrence using a fused PyTorch sequence module."""
    def __init__(self, dim, depth, cell):
        super().__init__()
        self.sequence = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[cell](
            dim, dim, num_layers=depth, batch_first=True)

    def forward(self, x):
        # cuDNN inference RNNs do not retain the reserve space for input gradients.
        # Keep the fused training path; use the native backend for explanations/dreams.
        if not self.training and torch.is_grad_enabled() and x.requires_grad:
            with torch.backends.cudnn.flags(enabled=False):
                return self.sequence(x)[0]
        return self.sequence(x)[0]


class PermutatorBlock(nn.Module):
    """Weighted height/width/channel mixing on a fixed square patch grid."""
    def __init__(self, dim, grid):
        super().__init__()
        self.grid = grid
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.height, self.width, self.channel = (nn.Linear(dim, dim) for _ in range(3))
        self.reweight = nn.Sequential(nn.Linear(dim, max(1, dim // 4)), nn.GELU(),
                                      nn.Linear(max(1, dim // 4), 3 * dim))
        self.project = nn.Linear(dim, dim)
        self.mlp = feedforward(dim, 4 * dim)

    def forward(self, x):
        b, _, c = x.shape
        g = self.grid
        z = self.norm1(x).reshape(b, g, g, g, c // g)
        h = z.permute(0, 3, 2, 1, 4).reshape(b, g, g, c)
        h = self.height(h).reshape(b, g, g, g, c // g).permute(0, 3, 2, 1, 4)
        w = z.permute(0, 1, 3, 2, 4).reshape(b, g, g, c)
        w = self.width(w).reshape(b, g, g, g, c // g).permute(0, 1, 3, 2, 4)
        branches = torch.stack((h.reshape(b, g*g, c), w.reshape(b, g*g, c),
                                self.channel(z.reshape(b, g*g, c))), dim=2)
        weights = self.reweight(branches.sum(2).mean(1))
        weights = weights.reshape(b, c, 3).transpose(1, 2).softmax(1)
        x = x + self.project((branches * weights[:, None]).sum(2))
        return x + self.mlp(self.norm2(x))


class Isotropic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.grid = cfg.size // cfg.patch
        self.patch = nn.Conv2d(cfg.channels, cfg.dim, cfg.patch, cfg.patch)
        self.conv = cfg.model == 9
        tokens = self.grid**2
        self.position = nn.Parameter(torch.randn(1, tokens, cfg.dim) * 0.02) if cfg.model in (3, 4) else None
        blocks = []
        for _ in range(cfg.depth):
            if cfg.model in (3, 4):
                blocks.append(TransformerBlock(cfg.dim, cfg.heads, cfg.model == 4))
            elif cfg.model == 5:
                blocks.append(MixerBlock(cfg.dim, tokens))
            elif cfg.model in (6, 7):
                blocks.append(GatedBlock(cfg.dim, tokens, cfg.model == 7))
            elif cfg.model == 10:
                blocks.append(PatchRecurrent(cfg.dim, cfg.depth, cfg.rnn_cell))
                if cfg.use_grn:
                    blocks[-1] = GRNResidualBlock(blocks[-1], cfg.dim)
                break
            elif cfg.model == 11:
                blocks.append(PermutatorBlock(cfg.dim, self.grid))
            else:
                blocks.append(ConvNextBlock(cfg.dim, 0.05))
            if cfg.use_grn:
                blocks[-1] = GRNResidualBlock(blocks[-1], cfg.dim, channels_first=self.conv)
        self.blocks = nn.Sequential(*blocks)
        self.norm = ChannelNorm(cfg.dim) if self.conv else nn.LayerNorm(cfg.dim)
        self.features = nn.Identity()
        self.out_dim = cfg.dim

    def forward(self, x):
        x = self.patch(x)
        if x.shape[-2:] != (self.grid, self.grid):
            raise ValueError('Input size differs from the checkpoint preprocessing size.')
        if not self.conv:
            x = x.flatten(2).transpose(1, 2)
            if self.position is not None:
                x = x + self.position
        x = self.features(self.norm(self.blocks(x)))
        return x.mean((2, 3)) if self.conv else x.mean(1)


class SwiGLU(nn.Module):
    def forward(self, x):
        value, gate = x.chunk(2, dim=-1)
        return value * F.silu(gate)


class HiddenLayer(nn.Module):
    def __init__(self, a, b, activation, slope):
        super().__init__()
        self.linear = nn.Linear(a, b * (2 if activation == 9 else 1))
        self.activation = [nn.Identity, nn.Sigmoid, nn.Tanh, nn.ReLU,
                           lambda: nn.LeakyReLU(slope), nn.PReLU, nn.GELU,
                           nn.SiLU, nn.Mish, SwiGLU][activation]()
        self.out_dim = b

    def forward(self, x):
        return self.activation(self.linear(x))


class Head(nn.Module):
    def __init__(self, dim, cfg, classes):
        super().__init__()
        hidden = []
        for width in cfg.head_dims:
            hidden.append(HiddenLayer(dim, width, cfg.activation, cfg.slope))
            dim = width
        self.hidden = nn.Sequential(*hidden)
        self.logits = nn.Linear(dim, classes)

    def forward(self, x):
        return self.logits(self.hidden(x))


class Classifier(nn.Module):
    def __init__(self, cfg, classes):
        super().__init__()
        cfg.validate()
        self.cfg = copy.deepcopy(cfg)
        self.encoder = Isotropic(cfg) if cfg.model in ISOTROPIC else Hierarchical(cfg)
        self.head = Head(self.encoder.out_dim, cfg, classes)
        self.frozen = False

    def forward(self, x):
        return self.head(self.encoder(x))

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def replace_head(self, classes, head_dims, activation=7, slope=0.01, freeze=True):
        self.cfg.head_dims = list(head_dims)
        self.cfg.activation, self.cfg.slope = activation, slope
        self.cfg.validate()
        self.head = Head(self.encoder.out_dim, self.cfg, classes).to(next(self.encoder.parameters()).device)
        self.frozen = freeze
        self.encoder.requires_grad_(not freeze)
        self.train(self.training)


def make_optimizer(model, choice, lr=None):
    if choice not in range(len(OPTIMIZER_LRS)):
        raise ValueError('Optimizer ID must be 0–4.')
    lr = OPTIMIZER_LRS[choice] if lr is None else lr
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('Optimizer learning rate must be finite and positive.')
    params = [p for p in model.parameters() if p.requires_grad]
    if choice == 0:
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if choice == 1:
        return torch.optim.Adam(params, lr=lr)
    if choice == 2:
        from lamb import Muon
        matrix, fallback = [], []
        for name, p in model.named_parameters():
            if p.requires_grad:
                target = fallback if p.ndim < 2 or name.startswith('head.') or name.endswith('position') else matrix
                target.append(p)
        groups = []
        if matrix:
            groups.append({'params': matrix, 'use_muon': True})
        if fallback:
            groups.append({'params': fallback, 'use_muon': False})
        return Muon(groups, lr=lr)
    if choice == 3:
        from lamb import CLion
        return CLion(params, lr=lr)
    if choice == 4:
        from lamb import RAdamScheduleFree
        optimizer = RAdamScheduleFree(params, lr=lr)
        optimizer.train()  # step() requires train mode; evaluation and saving swap to x
        return optimizer
    raise ValueError('Optimizer ID must be 0–4.')


def mix_batch(x, y, classes, augments):
    targets = F.one_hot(y, classes).to(x.dtype)
    if 19 in augments:
        lam = random.betavariate(0.4, 0.4)
        order = torch.randperm(len(x), device=x.device)
        x = lam * x + (1 - lam) * x[order]
        targets = lam * targets + (1 - lam) * targets[order]
    if 20 in augments:
        lam = random.betavariate(1.0, 1.0)
        order = torch.randperm(len(x), device=x.device)
        h, w = x.shape[-2:]
        ch, cw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        cy, cx = random.randrange(h), random.randrange(w)
        y0, y1 = max(0, cy - ch // 2), min(h, cy + (ch + 1) // 2)
        x0, x1 = max(0, cx - cw // 2), min(w, cx + (cw + 1) // 2)
        x = x.clone()
        x[:, :, y0:y1, x0:x1] = x[order, :, y0:y1, x0:x1]
        lam = 1 - (y1 - y0) * (x1 - x0) / (h * w)
        targets = lam * targets + (1 - lam) * targets[order]
    return x, targets


@torch.no_grad()
def evaluate(model, loader, device, classes, check_stop=None):
    model.eval()
    total, loss_sum, correct = 0, 0.0, 0
    counts, hits = torch.zeros(classes), torch.zeros(classes)
    for x, y in loader:
        if check_stop is not None:
            check_stop()
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction='sum').item()
        predictions = logits.argmax(1)
        correct += (predictions == y).sum().item()
        total += len(y)
        counts += torch.bincount(y.cpu(), minlength=classes)
        hits += torch.bincount(y[predictions == y].cpu(), minlength=classes)
    if not total:
        raise ValueError('Validation loader is empty.')
    return {'loss': loss_sum / total, 'accuracy': correct / total,
            'balanced_accuracy': (hits[counts > 0] / counts[counts > 0]).mean().item()}


def new_run(root, prefix='run'):
    root = Path(root).expanduser().resolve()
    run = root / f'{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}_{secrets.token_hex(2)}'
    run.mkdir(parents=True, exist_ok=False)
    return run


@contextmanager
def schedule_free_eval(optimizer):
    """RAdamScheduleFree trains at y; evaluate and save its averaged x."""
    swap = optimizer is not None and optimizer.param_groups[0].get('train_mode', False)
    if swap:
        optimizer.eval()
    try:
        yield
    finally:
        if swap:
            optimizer.train()


def save_checkpoint(path, model, classes, optimizer=None, **metadata):
    # state_dict() aliases live tensors, so x must stay loaded until written.
    with schedule_free_eval(optimizer):
        payload = {'format_version': 1, 'config': asdict(model.cfg), 'classes': list(classes),
                   'model_state': model.state_dict(), 'frozen': model.frozen, **metadata}
        if optimizer is not None:
            payload['optimizer_state'] = optimizer.state_dict()
        temporary = Path(str(path) + '.tmp')
        torch.save(payload, temporary)
    temporary.replace(path)


def resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        for name in ('interrupt.pt', 'best.pt', 'last.pt'):
            if (path / name).is_file():
                return path / name
        runs = sorted(p for p in path.iterdir() if p.is_dir() and
                      any((p / name).is_file() for name in ('interrupt.pt', 'last.pt')))
        if runs:
            return resolve_checkpoint(runs[-1])
    raise ValueError(f'No classifier checkpoint found at {path}')


def load_checkpoint(path, device='cpu'):
    path = resolve_checkpoint(path)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('format_version') != 1:
        raise ValueError('Unsupported classifier checkpoint format.')
    classes = saved['classes']
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError('Checkpoint class mapping is invalid.')
    model = Classifier(Config(**saved['config']), len(classes))
    model.load_state_dict(saved['model_state'], strict=True)
    model.frozen = saved.get('frozen', False)
    model.encoder.requires_grad_(not model.frozen)
    return model.to(device).eval(), classes, saved


@contextmanager
def save_on_interrupt(run, model, classes, optimizer, options, progress):
    """Defer terminal SIGINT to a safe boundary, then save and propagate exit."""
    requested = False
    def request_stop(signum, frame):
        nonlocal requested
        requested = True

    def check_stop():
        if requested:
            raise KeyboardInterrupt

    main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGINT, request_stop) if main_thread else None
    try:
        yield check_stop
        check_stop()
    except KeyboardInterrupt:
        path = run / 'interrupt.pt'
        print(f'\nStopping training. Saving {path} ...', flush=True)
        save_checkpoint(path, model, classes, optimizer, interrupted=True,
                        epoch=progress['epoch'], progress=dict(progress), training=options)
        print(f'Interrupt checkpoint saved: {path}', flush=True)
        raise
    finally:
        if main_thread:
            signal.signal(signal.SIGINT, previous)


def train_model(model, classes, train_records, val_records, *, batch_size=64,
                epochs=10, optimizer_id=1, seed=1, augments=(), save_dir='ImClass', device='cpu', lr=None,
                progress_callback=None, optimizer_state=None):
    """progress_callback(dict) is called at start, after every batch and after every epoch (GUI);
    returning True stops like Ctrl+C: interrupt.pt is saved at the same safe boundary.
    optimizer_state: continue from a saved optimizer state (lr, if given, overrides its rate)."""
    if batch_size < 1 or epochs < 1:
        raise ValueError('Batch size and epoch count must be positive.')
    if {p.resolve() for p, _ in train_records} & {p.resolve() for p, _ in val_records}:
        raise ValueError('Training and validation contain overlapping image paths.')
    model.to(device)
    sampler, counts, weights = balanced_sampler(train_records, classes, seed)
    loader = DataLoader(ImageDataset(train_records, Preprocess(model.cfg, True, augments)),
                        batch_size=batch_size, sampler=sampler, num_workers=0,
                        pin_memory=torch.device(device).type == 'cuda')
    val_loader = (DataLoader(ImageDataset(val_records, Preprocess(model.cfg)), batch_size=batch_size)
                  if val_records else None)
    optimizer = make_optimizer(model, optimizer_id, lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        if lr is not None:
            for group in optimizer.param_groups:
                group['lr'] = lr
    run = new_run(save_dir)
    options = dict(seed=seed, augments=list(augments), batch_size=batch_size, epochs=epochs,
                   optimizer_id=optimizer_id, lr=optimizer.param_groups[0]['lr'],
                   class_counts=counts, class_weights=weights)
    (run / 'config.json').write_text(json.dumps({'config': asdict(model.cfg), 'classes': classes,
                                               **options}, indent=2))
    with (run / 'split.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['split', 'path', 'class'])
        for split, records in [('train', train_records), ('validation', val_records)]:
            writer.writerows((split, str(p), classes[y]) for p, y in records)
    print(f'Run: {run}\nClass counts: {dict(zip(classes, counts))}', flush=True)
    notify = progress_callback or (lambda info: False)
    best = float('inf')
    progress = {'epoch': 0, 'completed_batches': 0, 'phase': 'starting'}
    with save_on_interrupt(run, model, classes, optimizer, options, progress) as check_stop, \
            (run / 'history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_loss', 'train_accuracy',
                                                   'val_loss', 'val_accuracy', 'val_balanced_accuracy'])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            progress.update(epoch=epoch, completed_batches=0, phase='training')
            check_stop()
            model.train()
            loss_sum, credit, total = 0.0, 0.0, 0
            for batch, (x, y) in enumerate(loader, 1):
                check_stop()
                x, y = x.to(device), y.to(device)
                x, targets = mix_batch(x, y, len(classes), augments)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, targets)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite training loss at epoch {epoch}, batch {batch}.')
                loss.backward()
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0,
                                         error_if_nonfinite=True)
                optimizer.step()
                loss_sum += loss.item() * len(y)
                credit += targets.gather(1, logits.argmax(1, keepdim=True)).sum().item()
                total += len(y)
                progress['completed_batches'] = batch
                if notify({'event': 'batch', 'epoch': epoch, 'batch': batch, 'batches': len(loader),
                           'loss': loss.item(), 'run': str(run)}):
                    raise KeyboardInterrupt
                check_stop()
                if batch % 100 == 0:
                    print(f'  epoch {epoch}: {batch}/{len(loader)} batches, loss {loss_sum/total:.4f}', flush=True)
            row = {'epoch': epoch, 'train_loss': loss_sum / total, 'train_accuracy': credit / total}
            if val_loader:
                progress['phase'] = 'validation'
                with schedule_free_eval(optimizer):
                    metrics = evaluate(model, val_loader, device, len(classes), check_stop)
                row.update({f'val_{k}': v for k, v in metrics.items()})
            check_stop()
            score = row.get('val_loss', row['train_loss'])
            if not math.isfinite(score):
                raise RuntimeError('Nonfinite epoch loss; checkpoint was not saved.')
            writer.writerow(row)
            stream.flush()
            metadata = dict(epoch=epoch, metrics=row, training=options)
            progress['phase'] = 'checkpoint'
            save_checkpoint(run / 'last.pt', model, classes, optimizer, **metadata)
            if score < best:
                best = score
                save_checkpoint(run / 'best.pt', model, classes, optimizer, **metadata)
            check_stop()
            print(' | '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()), flush=True)
            if notify({'event': 'epoch', 'epoch': epoch, 'epochs': epochs, 'row': row, 'run': str(run),
                       'best': score <= best, 'model': model}):
                raise KeyboardInterrupt
    if optimizer.param_groups[0].get('train_mode', False):
        optimizer.eval()  # return the averaged schedule-free weights that were saved
    return run


def unit_scale(x):
    x = x.detach().float().cpu()
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-12)


def display_image(x):
    x = x.detach().cpu().float()
    if x.ndim == 4:
        x = x[0]
    x = ((x + 1) / 2).clamp(0, 1)
    if x.shape[0] in (1, 2):
        x = x[:1].repeat(3, 1, 1)
    else:
        x = x[:3]
    return TF.to_pil_image(x)


def save_pair(x, heatmap, path):
    original = display_image(x)
    heatmap = F.interpolate(unit_scale(heatmap)[None, None], size=(original.height, original.width),
                            mode='bilinear', align_corners=False)[0, 0]
    # Blue -> yellow -> red, independently scaled for each map.
    rgb = torch.stack([(1.5 - (4 * heatmap - 3).abs()).clamp(0, 1),
                       (1.5 - (4 * heatmap - 2).abs()).clamp(0, 1),
                       (1.5 - (4 * heatmap - 1).abs()).clamp(0, 1)])
    overlay = Image.blend(original, TF.to_pil_image(rgb), 0.5)
    pair = Image.new('RGB', (original.width * 2, original.height))
    pair.paste(original, (0, 0))
    pair.paste(overlay, (original.width, 0))
    pair.save(path)


def attention_rollout(attentions):
    """Head-mean attention with residual identity, composed in forward order."""
    joint = None
    for attention in attentions:
        matrix = attention.detach().float().cpu().mean(0)
        matrix = matrix + torch.eye(matrix.shape[-1])
        matrix = matrix / matrix.sum(-1, keepdim=True)
        joint = matrix if joint is None else matrix @ joint
    if joint is None:
        raise ValueError('Attention rollout requires at least one attention layer.')
    # These ViTs pool all output tokens rather than using a CLS token.
    return joint.mean(0)


def explain(model, x, target=None):
    """Grad-CAM on final spatial features; maps returned separately by meaning."""
    model.eval()
    x = x.detach().requires_grad_(True)
    features, extra, handles, capturing = [], {}, [], []
    handles.append(model.encoder.features.register_forward_hook(lambda m, i, o: features.append(o)))
    for name, module in model.encoder.named_modules():
        if isinstance(module, (Attention, GatedBlock, MixerBlock)):
            capturing.append((name, module, module.capture))
            module.capture = True
        if isinstance(module, SqueezeExcitation):
            def collect(m, inputs, output, key=name):
                extra[f'{key}_se_feature_energy'] = output.detach().abs().mean(1)[0].cpu()
            handles.append(module.register_forward_hook(collect))
    try:
        logits = model(x)
        target = logits.argmax(1).item() if target is None else target
        gradient, input_gradient = torch.autograd.grad(logits[0, target], (features[0], x))
        extra['input_saliency'] = input_gradient[0].detach().abs().mean(0).cpu()
        feat = features[0]
        if feat.ndim == 4:
            cam = (gradient.mean((2, 3), keepdim=True) * feat).sum(1)[0].relu()
        else:
            cam = (gradient.mean(1, keepdim=True) * feat).sum(-1)[0].relu()
            cam = cam.reshape(model.encoder.grid, model.encoder.grid)
        grid = getattr(model.encoder, 'grid', None)
        if model.cfg.model in (3, 4):
            extra['attention_rollout'] = attention_rollout(
                [module.last_attention[0] for _, module, _ in capturing
                 if isinstance(module, Attention)]).reshape(grid, grid)
        for name, module, _ in capturing:
            if isinstance(module, Attention):
                for h, weights in enumerate(module.last_attention[0]):
                    extra[f'{name}_attention_head_{h}'] = weights.mean(0).reshape(grid, grid)
            elif isinstance(module, GatedBlock):
                extra[f'{name}_gate_magnitude'] = module.last_gate[0].reshape(grid, grid)
            else:
                extra[f'{name}_token_mixer_response'] = module.last_response[0].reshape(grid, grid)
        return logits.detach(), cam.detach(), extra
    finally:
        for handle in handles:
            handle.remove()
        for _, module, previous in capturing:
            module.capture = previous
            for attr in ('last_attention', 'last_gate', 'last_response'):
                if hasattr(module, attr):
                    setattr(module, attr, None)


def safe_name(name):
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in str(name))[:160]


def save_class_comparison(model, x, classes, logits, count, path):
    """Top-k class CAM panels sharing one intensity scale, with probabilities."""
    probs = logits.softmax(-1)[0]
    indices = probs.topk(min(count, len(classes))).indices.tolist()
    cams = [explain(model, x, target=i)[1].cpu() for i in indices]
    scale = torch.stack(cams).max().clamp_min(1e-12)
    original = display_image(x).resize((192, 192))
    sheet = Image.new('RGB', (192 * (len(indices) + 1), 232), 'white')
    sheet.paste(original, (0, 40))
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), 'Input / shared CAM scale', fill='black')
    for col, (index, cam) in enumerate(zip(indices, cams), 1):
        heat = F.interpolate((cam / scale)[None, None], (192, 192), mode='bilinear',
                             align_corners=False)[0, 0].clamp(0, 1)
        rgb = torch.stack((heat, torch.zeros_like(heat), 1 - heat))
        sheet.paste(Image.blend(original, TF.to_pil_image(rgb), 0.5), (192 * col, 40))
        label = f'{index}: {classes[index]}'
        draw.text((192 * col + 4, 4), label.encode('ascii', 'replace').decode()[:28], fill='black')
        draw.text((192 * col + 4, 20), f'{probs[index].item():.2%}', fill='black')
    sheet.save(path)


def feature_layers(model):
    """Named outputs with meaningful spatial channels or final head units."""
    layers = {}
    activations = (nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.GELU, nn.SiLU, nn.Mish,
                   nn.Sigmoid, nn.Tanh, SwiGLU)
    for name, module in model.named_modules():
        if name.startswith('encoder.'):
            if isinstance(module, nn.Conv2d):
                layers[name] = (module, 'convolution output (before activation)')
            elif isinstance(module, activations) and hasattr(model.encoder, 'layers'):
                layers[name] = (module, 'activation output')
            elif name == 'encoder.features' or (name.startswith('encoder.blocks.') and name.count('.') == 2):
                layers[name] = (module, 'encoder feature output')
        elif isinstance(module, HiddenLayer) or name == 'head.logits':
            layers[name] = (module, 'head unit output')
    return layers


def signed_feature_image(values, scale):
    """Blue = negative, white = zero, red = positive; retain constant maps."""
    z = (values.detach().float().cpu() / max(scale, 1e-12)).clamp(-1, 1)
    rgb = torch.stack((1 + z.clamp(max=0), 1 - z.abs(), 1 - z.clamp(min=0)))
    return TF.to_pil_image(rgb)


def write_feature_sheets(maps, entries, folder, name, kind, writer, total_units):
    """Stream up to 64 labeled maps per page, with a shared symmetric scale."""
    scale = maps.abs().max().item()
    sheet, draw = None, None
    count = len(entries)
    for index, (unit, input_channel) in enumerate(entries):
        slot = index % 64
        filename = f'{safe_name(name)}_{index // 64 + 1:04d}.png'
        if slot == 0:
            on_page = min(64, count - index)
            sheet = Image.new('RGB', (min(8, max(4, on_page)) * 112, 64 + math.ceil(on_page / 8) * 112), 'white')
            draw = ImageDraw.Draw(sheet)
            draw.text((4, 2), name, fill='black')
            draw.text((4, 17), kind, fill='black')
            draw.text((4, 32), f'blue -{scale:.3g} / white 0 / red +{scale:.3g}', fill='black')
            draw.text((4, 47), f'{maps.shape[-2]}x{maps.shape[-1]} maps; page {index // 64 + 1}', fill='black')
        values = maps[index]
        left, top = (slot % 8) * 112, 64 + (slot // 8) * 112
        label = f'unit {unit}' if input_channel == '' else f'out {unit} in {input_channel}'
        draw.text((left + 3, top + 2), label, fill='black')
        ratio = 88 / max(values.shape)
        thumb = signed_feature_image(values, scale).resize(
            (max(1, round(values.shape[1] * ratio)), max(1, round(values.shape[0] * ratio))),
            Image.Resampling.NEAREST)
        sheet.paste(thumb, (left + 4, top + 20))
        writer.writerow([filename, slot, name, kind, unit, input_channel,
                         maps.shape[-2], maps.shape[-1], values.min().item(), values.max().item(),
                         values.mean().item(), scale, total_units])
        if slot == 63 or index == count - 1:
            sheet.save(folder / filename)


FEATURE_FIELDS = ['file', 'tile', 'layer', 'kind', 'unit', 'input_channel', 'height', 'width',
                  'minimum', 'maximum', 'mean', 'scale', 'total_layer_units']


def check_unit_limit(max_units):
    if not isinstance(max_units, int) or max_units < 0:
        raise ValueError('Maximum feature units must be a nonnegative integer (0 = all).')


def save_kernels(model, folder, max_units=0):
    """Export every input-channel slice of selected Conv2d output filters."""
    check_unit_limit(max_units)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / 'manifest.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(FEATURE_FIELDS)
        for name, module in model.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            n = min(max_units or module.out_channels, module.out_channels)
            weights = module.weight[:n].detach().float().cpu()
            inputs_per_group = weights.shape[1]
            outputs_per_group = module.out_channels // module.groups
            entries = [(out, (out // outputs_per_group) * inputs_per_group + channel)
                       for out in range(n) for channel in range(inputs_per_group)]
            write_feature_sheets(weights.flatten(0, 1), entries, folder, name,
                                 'kernel weights (bias excluded)', writer, module.out_channels)
    return folder


def save_feature_maps(model, x, folder, max_units=0):
    """Export unit responses for one image without retaining all layer outputs."""
    check_unit_limit(max_units)
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError('Feature visualization requires a single [1,C,H,W] image.')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    display_image(x).save(folder / 'input.png')
    handles = []
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with (folder / 'manifest.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(FEATURE_FIELDS)
            def capture(module, inputs, output, name, kind):
                if output.ndim == 4:
                    maps = output[0]
                elif output.ndim == 3:
                    grid = model.encoder.grid
                    maps = output[0].transpose(0, 1).reshape(-1, grid, grid)
                elif output.ndim == 2:
                    maps = output[0, :, None, None]
                else:
                    raise ValueError(f'Unsupported feature shape for {name}: {tuple(output.shape)}')
                total = maps.shape[0]
                n = min(max_units or total, total)
                maps = maps[:n].detach().float().cpu()
                write_feature_sheets(maps, [(i, '') for i in range(n)], folder, name, kind, writer, total)
            for name, (module, kind) in feature_layers(model).items():
                handles.append(module.register_forward_hook(
                    lambda m, i, o, name=name, kind=kind: capture(m, i, o, name, kind)))
            with torch.no_grad():
                logits = model(x)
        return logits
    finally:
        for handle in handles:
            handle.remove()
        for module, mode in modes:
            module.training = mode


def sample(model, classes, source, plot=False, output_root='samples', device='cpu', compare_classes=0,
           kernel_plots=False, feature_maps=False, max_units=0):
    check_unit_limit(max_units)
    if compare_classes < 0:
        raise ValueError('Class comparison count must be nonnegative.')
    paths = image_files(source)
    if not paths:
        raise ValueError('No supported images in the sample folder.')
    model.to(device).eval()
    transform = Preprocess(model.cfg)
    run = new_run(output_root, 'sample')
    if kernel_plots:
        save_kernels(model, run / 'kernels', max_units)
    with (run / 'predictions.csv').open('w', newline='') as out, (run / 'errors.csv').open('w', newline='') as err:
        writer, errors = csv.writer(out), csv.writer(err)
        writer.writerow(['path', 'prediction', 'confidence'] + [f'p:{c}' for c in classes])
        errors.writerow(['path', 'error'])
        successes, failures = 0, 0
        for path in paths:
            try:
                with Image.open(path) as image:
                    x = transform(image).unsqueeze(0).to(device)
            except (OSError, ValueError) as exc:
                errors.writerow([str(path), str(exc)])
                failures += 1
                print(f'Skipped {path}: {exc}', flush=True)
                continue
            stem = safe_name(path.stem) + '_' + hashlib.sha256(str(path).encode()).hexdigest()[:12]
            if feature_maps:
                logits = save_feature_maps(model, x, run / 'features' / stem, max_units)
            if plot:
                logits, cam, maps = explain(model, x)
            elif not feature_maps:
                with torch.no_grad():
                    logits = model(x)
            probs = logits.softmax(-1)[0].cpu()
            prediction = int(probs.argmax())
            writer.writerow([str(path), classes[prediction], probs[prediction].item()] + probs.tolist())
            if plot:
                save_pair(x, cam, run / f'{stem}_CAM_{safe_name(classes[prediction])}.png')
                for name, heatmap in maps.items():
                    save_pair(x, heatmap, run / f'{stem}_{safe_name(name)}.png')
                if compare_classes:
                    save_class_comparison(model, x, classes, logits, compare_classes,
                                          run / f'{stem}_class_comparison.png')
            successes += 1
            print(f'{path.name}: {classes[prediction]} ({probs[prediction]:.2%})', flush=True)
        print(f'Saved {successes} predictions; {failures} unreadable images. Output: {run}', flush=True)
    return run


@dataclass(frozen=True)
class DreamTarget:
    module: str
    unit: int
    axis: int
    name: str


def dream_targets(model, classes, scope):
    if scope not in (0, 1, 2):
        raise ValueError('DeepDream scope must be 0, 1 or 2.')
    targets = []
    if scope == 0:
        for name, module in model.encoder.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                count = module.out_channels if isinstance(module, nn.Conv2d) else module.out_features
                axis = 1 if isinstance(module, nn.Conv2d) else -1
                targets.extend(DreamTarget(f'encoder.{name}', i, axis,
                                           f'encoder_{name}_unit_{i:05d}') for i in range(count))
            elif isinstance(module, PatchRecurrent):
                targets.extend(DreamTarget(f'encoder.{name}', i, -1,
                                           f'encoder_{name}_recurrent_unit_{i:05d}')
                               for i in range(module.sequence.hidden_size))
    if scope in (0, 1):
        for layer, module in enumerate(model.head.hidden):
            targets.extend(DreamTarget(f'head.hidden.{layer}', i, -1,
                                       f'head_layer_{layer}_neuron_{i:05d}') for i in range(module.out_dim))
    targets.extend(DreamTarget('head.logits', i, -1, f'class_{i:05d}_{safe_name(name)}')
                   for i, name in enumerate(classes))
    return targets


def validate_dream_choices(setup, dream_optimizer):
    if setup not in range(len(DREAM_SETUPS)):
        raise ValueError('DeepDream setup must be 0=direct pixels or 1=FFT.')
    if dream_optimizer not in range(len(DREAM_OPTIMIZERS)):
        raise ValueError('DeepDream optimizer must be 0=Adam, 1=L-BFGS or 2=normalized gradient ascent.')


class DreamImage(nn.Module):
    """Image or Fourier coordinates; render bounded pixels for every objective call."""
    def __init__(self, pixels, setup):
        super().__init__()
        validate_dream_choices(setup, 0)
        self.setup = setup
        self.size = pixels.shape[-2:]
        self.values = nn.Parameter(self.encode(pixels.detach()).clone().contiguous())

    def encode(self, pixels):
        if self.setup == 0:
            return pixels
        return torch.view_as_real(torch.fft.rfft2(pixels, norm='ortho'))

    def forward(self):
        pixels = self.values if self.setup == 0 else torch.fft.irfft2(
            torch.view_as_complex(self.values), s=self.size, norm='ortho')
        return pixels.clamp(0, 1)

    def project(self):
        # Keep first-order updates within bounds in either coordinate system.
        with torch.no_grad():
            self.values.copy_(self.encode(self()))


def dream_one(model, target, steps=100, lr=0.05, starter=None, noise=0.0,
              setup=0, dream_optimizer=0, suppress_other_units=False, step_callback=None):
    """Maximize a target, optionally suppressing other outputs in its layer.
    step_callback(step, image, activation) after each step (GUI); returning True stops early."""
    if steps < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError('DeepDream requires positive steps and finite positive learning rate.')
    if not math.isfinite(noise) or not 0 <= noise <= 1:
        raise ValueError('Starter noise must be finite and between 0 and 1.')
    if not isinstance(suppress_other_units, bool):
        raise ValueError('suppress_other_units must be boolean.')
    validate_dream_choices(setup, dream_optimizer)
    model.eval()
    device = next(model.parameters()).device
    shape = (1, model.cfg.channels, model.cfg.size, model.cfg.size)
    if starter is None:
        initial_pixels = torch.rand(*shape, device=device) * 0.2 + 0.4
    else:
        if tuple(starter.shape) != shape or not torch.isfinite(starter).all() or \
                starter.min() < -1 or starter.max() > 1:
            raise ValueError('Starter must be a finite preprocessed [1,C,H,W] tensor in [-1,1].')
        initial_pixels = (starter.detach().to(device) + 1) / 2
        if noise:
            initial_pixels = initial_pixels + noise * (torch.rand_like(initial_pixels) * 2 - 1)
        initial_pixels = initial_pixels.clamp(0, 1)
    image = DreamImage(initial_pixels, setup)
    variable = image.values
    if dream_optimizer == 0:
        optimizer = torch.optim.Adam([variable], lr=lr)
    elif dream_optimizer == 1:
        optimizer = torch.optim.LBFGS([variable], lr=lr, max_iter=1,
                                      history_size=10, line_search_fn='strong_wolfe')
    module = dict(model.named_modules())[target.module]
    values = []
    def capture(m, inputs, output):
        target_output = output.select(target.axis, target.unit)
        objective = target_output.mean()
        if suppress_other_units and output.shape[target.axis] > 1:
            other_mean = ((output.sum(dim=target.axis) - target_output)
                          / (output.shape[target.axis] - 1)).mean()
            objective = objective - other_mean
        values.append(objective)
    handle = module.register_forward_hook(capture)
    try:
        initial = None
        def closure():
            nonlocal initial
            variable.grad = None
            values.clear()
            pixels = image()
            model(pixels * 2 - 1)
            activation = values[-1]
            if initial is None:
                initial = activation.item()
            tv = pixels.new_zeros(())
            if pixels.shape[-1] > 1:
                tv = (pixels[:, :, 1:, :] - pixels[:, :, :-1, :]).square().mean()
                tv = tv + (pixels[:, :, :, 1:] - pixels[:, :, :, :-1]).square().mean()
            loss = -activation + 0.01 * tv + 0.001 * (pixels - 0.5).square().mean()
            gradient, = torch.autograd.grad(loss, variable)
            if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
                raise RuntimeError(f'Nonfinite DeepDream objective or gradient: {target.name}')
            variable.grad = gradient.contiguous()
            return loss
        for step in range(steps):
            if dream_optimizer == 1:
                # Do not project parameters during/after line search: that would
                # invalidate its displacement history. The closure clamps pixels.
                optimizer.step(closure)
            else:
                closure()
                if dream_optimizer == 0:
                    optimizer.step()
                else:
                    # Descent on negative activation equals regularized ascent.
                    with torch.no_grad():
                        gradient = variable.grad
                        # Scale first to avoid overflow when squaring large gradients.
                        scaled = gradient / gradient.abs().max().clamp_min(1e-12)
                        direction = scaled / scaled.square().mean().sqrt().clamp_min(1e-12)
                        variable.add_(direction, alpha=-lr)
                image.project()
            if not torch.isfinite(variable).all():
                raise RuntimeError(f'Nonfinite DeepDream parameters: {target.name}')
            if step_callback is not None and step_callback(step + 1, image, values[-1].item() if values else None):
                break
        values.clear()
        with torch.no_grad():
            pixels = image()
            model(pixels * 2 - 1)
        if not torch.isfinite(values[-1]):
            raise RuntimeError(f'Nonfinite final DeepDream activation: {target.name}')
        return pixels.detach() * 2 - 1, initial, values[-1].item()
    finally:
        handle.remove()


def deepdream(model, classes, scope=2, steps=100, lr=0.05, output_root='DeepDream', targets=None,
              starter=None, inits=1, noise=0.1, setup=0, dream_optimizer=0,
              suppress_other_units=False):
    validate_dream_choices(setup, dream_optimizer)
    if not isinstance(inits, int) or inits < 1:
        raise ValueError('DeepDream initialization count must be a positive integer.')
    if steps < 1 or not math.isfinite(lr) or lr <= 0 or not math.isfinite(noise) or not 0 <= noise <= 1:
        raise ValueError('Require positive steps/LR and starter noise in [0,1].')
    if not isinstance(suppress_other_units, bool):
        raise ValueError('suppress_other_units must be boolean.')
    starter_path = str(Path(starter).expanduser().resolve()) if starter is not None else ''
    initial = None
    if starter_path:
        with Image.open(starter_path) as image:
            initial = Preprocess(model.cfg)(image).unsqueeze(0)
    targets = dream_targets(model, classes, scope) if targets is None else targets
    run = new_run(output_root, 'dream')
    (run / 'config.json').write_text(json.dumps(dict(starter=starter_path, inits=inits, noise=noise,
                                                   steps=steps, lr=lr, seed=torch.initial_seed(),
                                                   setup=setup, setup_name=DREAM_SETUPS[setup],
                                                   dream_optimizer=dream_optimizer,
                                                   suppress_other_units=suppress_other_units,
                                                   optimizer_name=DREAM_OPTIMIZERS[dream_optimizer]), indent=2))
    print(f'Optimizing {len(targets)} units x {inits} initializations, {steps} steps each. Output: {run}', flush=True)
    with (run / 'manifest.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['file', 'module', 'unit', 'axis', 'initial_activation', 'final_activation',
                         'initialization', 'starter', 'noise', 'setup', 'dream_optimizer',
                         'suppress_other_units'])
        for index, target in enumerate(targets, 1):
            for init in range(inits):
                # First run preserves the starter exactly; subsequent runs perturb it.
                jitter = noise if initial is not None and init > 0 else 0.0
                x, start, end = dream_one(model, target, steps, lr, initial, jitter, setup,
                                          dream_optimizer, suppress_other_units)
                name = f'{index:07d}_{safe_name(target.name)}_init_{init + 1:03d}.png'
                # Preserve LA/RGBA channels in optimized images.
                TF.to_pil_image(((x[0].cpu() + 1) / 2).clamp(0, 1)).save(run / name)
                writer.writerow([name, target.module, target.unit, target.axis, start, end,
                                 init + 1, starter_path, jitter, setup, dream_optimizer,
                                 suppress_other_units])
                stream.flush()
                print(f'[{index}/{len(targets)}, init {init + 1}/{inits}] {target.name}: {start:.5g} -> {end:.5g}', flush=True)
    return run


def ask(label, default=None, parse=str, valid=lambda x: True):
    while True:
        raw = input(f'{label}' + (f' [{default}]' if default is not None else '') + ': ').strip()
        try:
            value = parse(raw if raw else str(default)) if (raw or default is not None) else parse('')
            if not valid(value):
                raise ValueError('Value is outside the allowed range.')
            return value
        except (ValueError, OSError) as exc:
            print(f'  {exc} Please try again.')


def yes_no(raw):
    if raw.lower() in ('y', 'yes', '1'):
        return True
    if raw.lower() in ('n', 'no', '0'):
        return False
    raise ValueError('Enter yes or no.')


def prompt_head(cfg):
    layers = ask('MLP hidden layer count (0 = linear classifier)', 1, int, lambda x: x >= 0)
    if layers:
        def widths(raw):
            dims = [int(x.strip()) for x in raw.split(',')]
            if len(dims) == 1:
                dims *= layers
            if len(dims) != layers or min(dims) < 1:
                raise ValueError('Enter one positive width or one per hidden layer.')
            return dims
        cfg.head_dims = ask('MLP hidden widths (one or comma-separated)', 128, widths)
        print('Activation: 0=none 1=Sigmoid 2=Tanh 3=ReLU 4=LeakyReLU 5=PReLU 6=GELU 7=SiLU 8=Mish 9=SwiGLU')
        cfg.activation = ask('Head activation', 7, int, lambda x: x in range(10))
        if cfg.activation == 4:
            cfg.slope = ask('LeakyReLU slope', 0.01, float, lambda x: math.isfinite(x) and x >= 0)
    else:
        cfg.head_dims = []
    return cfg


def prompt_config():
    cfg = Config()
    cfg.channels = ask('Image channels (1=L, 2=LA, 3=RGB, 4=RGBA)', 3, int, lambda x: x in (1, 2, 3, 4))
    cfg.imres = ask('Image resolution (-1 = original)', 32, int, lambda x: x == -1 or x > 0)
    cfg.crop = ask('Crop size', 256 if cfg.imres == -1 else cfg.imres, int, lambda x: x > 0)
    print('\n'.join(f'{i} = {name}' for i, name in enumerate(MODEL_NAMES)))
    cfg.model = ask('Model type', 0, int, lambda x: x in range(len(MODEL_NAMES)))
    if cfg.model in ISOTROPIC:
        default_patch = 4 if cfg.size % 4 == 0 else 1
        cfg.patch = ask('Patch size (must divide effective crop/image size)', default_patch, int,
                        lambda x: x > 0 and x <= cfg.size and cfg.size % x == 0)
        divisor = cfg.size // cfg.patch if cfg.model == 11 else 1
        cfg.dim = ask(f'Hidden width (multiple of {divisor})', max(divisor, math.ceil(128 / divisor) * divisor),
                      int, lambda x: x >= 2 and x % divisor == 0)
        cfg.depth = ask('Encoder layer count', 4, int, lambda x: x > 0)
        if cfg.model in (3, 4):
            cfg.heads = ask('Attention head count (must divide width)', 4 if cfg.dim % 4 == 0 else 1,
                            int, lambda x: x > 0 and cfg.dim % x == 0)
        elif cfg.model == 7:
            print('aMLP uses one attention head with dimension 64 per block.')
        elif cfg.model == 10:
            cfg.rnn_cell = ask('Recurrent cell: rnn, gru, lstm', 'gru', str.lower,
                               lambda x: x in ('rnn', 'gru', 'lstm'))
        cfg.use_grn = ask('Use Global Response Normalization (GRN)', False, bool)
    else:
        cfg.min_dim = ask('Minimum feature width', 32, int, lambda x: x >= 2)
        cfg.max_dim = ask('Maximum feature width', max(128, cfg.min_dim), int, lambda x: x >= cfg.min_dim)
    return prompt_head(cfg)


def prompt_training(model, classes, records, device):
    print('Augmentations: empty/0=none, -1=all, or comma-separated IDs\n' +
          '\n'.join(f'{i}={name}' for i, name in AUGMENTS.items()))
    augments = ask('Augmentations', '', parse_augments)
    def validation_path(raw):
        if not raw:
            return []
        validation, _ = load_records(raw, classes)
        if {p for p, _ in records} & {p for p, _ in validation}:
            raise ValueError('Validation overlaps the training image paths.')
        return validation
    validation = ask('Validation folder or dataset .json (empty = percentage split)', '', validation_path)
    fraction = 0.0
    if not validation:
        def check_fraction(raw):
            f = float(raw)
            split_records(records, f, 1)  # Validate per-class feasibility before training.
            return f
        fraction = ask('Validation fraction [0,1), 0 disables', 0.1, check_fraction)
    batch_size = ask('Batch size', 64, int, lambda x: x > 0)
    epochs = ask('Epoch count', 10, int, lambda x: x > 0)
    optimizer_id = ask('Optimizer: 0=SGD+momentum 1=Adam 2=Muon 3=CLion 4=RAdamScheduleFree', 1, int,
                       lambda x: x in range(len(OPTIMIZER_LRS)))
    lr = ask('Optimizer learning rate', OPTIMIZER_LRS[optimizer_id], float,
             lambda x: math.isfinite(x) and x > 0)
    seed = seed_everything(ask('Seed (0 = random)', 0, int, lambda x: 0 <= x < 2**63))
    save_dir = ask('Save directory', 'ImClass')
    print(f'Seed: {seed}', flush=True)
    if fraction:
        records, validation = split_records(records, fraction, seed)
    # Initialization must occur after the seed prompt for reproducible runs.
    if isinstance(model, Config):
        model = Classifier(model, len(classes))
    else:
        # Recreate only the head after seeding; retain the loaded encoder.
        model.replace_head(len(classes), model.cfg.head_dims, model.cfg.activation,
                           model.cfg.slope, model.frozen)
    return train_model(model, classes, records, validation, batch_size=batch_size, epochs=epochs,
                       optimizer_id=optimizer_id, lr=lr, seed=seed, augments=augments,
                       save_dir=save_dir, device=device)


def main():
    aliases = {key: mode for mode, keys in enumerate([('0', 'p', 'pretrain'), ('1', 'f', 'finetune'),
                                                     ('2', 's', 'sample'), ('3', 'd', 'deepdream'),
                                                     ('4', 'g', 'gui')]) for key in keys}
    def mode_parser(raw):
        if raw.lower() not in aliases:
            raise ValueError('Choose 0/p/pretrain, 1/f/finetune, 2/s/sample, 3/d/deepdream or 4/g/gui.')
        return aliases[raw.lower()]
    mode = ask('Mode: 0/p/pretrain, 1/f/finetune, 2/s/sample, 3/d/deepdream, 4/g/gui (browser)', 'p', mode_parser)
    if mode == 4:
        from imclass_gui import run_gui
        port = ask('Port', 8766, int, lambda x: 0 < x < 65536)
        run_gui(port=port)
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)
    if mode == 0:
        records, classes = ask('Data path (class subfolders or dataset .json)', 'CIFAR10', load_records)
        cfg = prompt_config()
        run = prompt_training(cfg, classes, records, device)
    else:
        checkpoint = ask('Checkpoint file or directory', 'ImClass', resolve_checkpoint)
        model, classes, _ = load_checkpoint(checkpoint, device)
        print(f'Loaded {MODEL_NAMES[model.cfg.model]}, {len(classes)} classes, input {model.cfg.channels}x{model.cfg.size}x{model.cfg.size}')
        if mode == 1:
            records, classes = ask('New image data path (class subfolders or dataset .json)', 'CIFAR10', load_records)
            freeze = ask('Freeze encoder?', 'yes', yes_no)
            cfg = prompt_head(copy.deepcopy(model.cfg))
            model.replace_head(len(classes), cfg.head_dims, cfg.activation, cfg.slope, freeze)
            run = prompt_training(model, classes, records, device)
        elif mode == 2:
            plot = ask('Save CAM, saliency and attention/gating/rollout maps?', 'yes', yes_no)
            compare = ask('Compare top classes (0 disables)', min(3, len(classes)), int,
                          lambda x: 0 <= x <= len(classes)) if plot else 0
            features = ask('Feature plots: 0=off, 1=kernels, 2=unit activations, 3=both', 0,
                           int, lambda x: x in range(4))
            max_units = ask('Maximum units per layer (0=all; kernels include every input slice)', 0,
                            int, lambda x: x >= 0) if features else 0
            if features:
                print('Feature plots are paginated, 64 tiles per sheet; all units can produce many files.')
            def sample_path(raw):
                if not image_files(raw):
                    raise ValueError('No images found.')
                return Path(raw).expanduser().resolve()
            source = ask('Image file or directory', None, sample_path)
            run = sample(model, classes, source, plot, device=device, compare_classes=compare,
                         kernel_plots=features in (1, 3), feature_maps=features in (2, 3), max_units=max_units)
        else:
            scope = ask('DeepDream: 0=encoder+head+classes, 1=head+classes, 2=classes', 2,
                         int, lambda x: x in (0, 1, 2))
            suppress_other_units = ask('Suppress all other units in each target layer?', False, yes_no)
            setup = ask('Dreaming setup: 0=direct pixels, 1=FFT and inverse FFT', 0,
                         int, lambda x: x in range(len(DREAM_SETUPS)))
            dream_optimizer = ask('DeepDream optimizer: 0=Adam, 1=L-BFGS, 2=Gradient Ascent with Normalized Gradients',
                                   0, int, lambda x: x in range(len(DREAM_OPTIMIZERS)))
            steps = ask('Optimization steps per unit', 100, int, lambda x: x > 0)
            lr = ask('Dream learning rate', 1.0 if dream_optimizer == 1 else 0.05, float,
                     lambda x: math.isfinite(x) and x > 0)
            def starter_path(raw):
                if not raw:
                    return None
                path = Path(raw).expanduser().resolve()
                with Image.open(path) as image:
                    Preprocess(model.cfg)(image)
                return path
            starter = ask('Starter image (empty = noise)', '', starter_path)
            inits = ask('Initializations per target', 1, int, lambda x: x > 0)
            noise = ask('Starter noise amplitude for extra initializations [0,1]', 0.1, float,
                        lambda x: math.isfinite(x) and 0 <= x <= 1) if starter and inits > 1 else 0.1
            seed = seed_everything(ask('Seed (0 = random)', 0, int, lambda x: 0 <= x < 2**63))
            print(f'Seed: {seed}')
            run = deepdream(model, classes, scope, steps, lr, starter=starter, inits=inits, noise=noise,
                            setup=setup, dream_optimizer=dream_optimizer,
                            suppress_other_units=suppress_other_units)
    print(f'Finished: {run}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\nStopped. Completed checkpoint and visualization files have been retained.')
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(f'Error: {exc}') from exc
