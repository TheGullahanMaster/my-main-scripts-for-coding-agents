#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Architecture GAN Trainer with:
- MLP (CIPS-style coordinate-based generation)
- CREPS (column-row entangled pixel synthesis)
- Deconv (Original DCGAN-style)
- ViT (Vision Transformer)
- GRU (Autoregressive column generation)
- Swin (StyleSwin transformer-based)
- HyperMixer LoRA projections for patch-based models
- ETAct, GroupNorm, Self-attention, Residual connections
- Multiple loss functions and optimizers
"""

import json
import math
import os
import random
import sys
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Set
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils as vutils, datasets as tvdatasets
from PIL import Image
import torch.nn.utils.spectral_norm as spectral_norm
from lamb import *
# =====================
#  ETAct
# =====================
class ETAct(nn.Module):
    """Per‑channel trainable α/β/δ/γ wrapper around an inner activation."""
    def __init__(self, channels: int, activation: nn.Module):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.delt  = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.act   = activation
    def forward(self, x):
        return (self.act((x * self.alpha) + self.delt) * self.gamma) + self.beta

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)
class SineEvenLReLU(nn.Module):
    """Odd channels use sine, even channels use LeakyReLU."""
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x):
        # Split into odd and even channels
        odd_channels = x[:, 1::2]  # indices 1, 3, 5, ...
        even_channels = x[:, 0::2]  # indices 0, 2, 4, ...
        
        # Apply activations
        odd_activated = torch.sin(odd_channels)
        even_activated = F.leaky_relu(even_channels, self.negative_slope)
        
        # Interleave back
        output = torch.zeros_like(x)
        output[:, 1::2] = odd_activated
        output[:, 0::2] = even_activated
        
        return output
# =====================
#  HyperMixer LoRA
# =====================
class HyperMixerLoRA(nn.Module):
    """LoRA-based adaptive linear projection conditioned on input features."""
    def __init__(self, in_features: int, out_features: int, condition_dim: int, rank: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Base linear projection (no activation)
        self.base_linear = nn.Linear(in_features, out_features, bias=True)
        
        # LoRA branches: conditioner -> low-rank matrices
        self.lora_down = nn.Linear(condition_dim, in_features * rank, bias=False)
        self.lora_up = nn.Linear(condition_dim, rank * out_features, bias=False)
        
        # ETAct(Tanh) for the conditioner pathway
        self.condition_act = ETActFlat(condition_dim, nn.Tanh())
        
        # Scale factor for LoRA
        self.lora_scale = nn.Parameter(torch.ones(1) * 0.01)
        self.base_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x, condition):
        """
        x: (batch, in_features) - input to be projected
        condition: (batch, condition_dim) - conditioning signal
        """
        # Base projection
        base_out = self.base_linear(x)
        
        # Generate LoRA matrices from condition
        cond_act = self.condition_act(condition)
        A = self.lora_down(cond_act).view(-1, self.in_features, self.rank)  # (B, in, rank)
        B = self.lora_up(cond_act).view(-1, self.rank, self.out_features)   # (B, rank, out)
        
        # Apply LoRA: x @ A @ B
        lora_out = torch.bmm(x.unsqueeze(1), A)  # (B, 1, rank)
        lora_out = torch.bmm(lora_out, B).squeeze(1)  # (B, out)
        
        return base_out*self.base_scale + self.lora_scale*lora_out


class ETActFlat(nn.Module):
    """ETAct for flat tensors (batch, features)."""
    def __init__(self, channels: int, activation: nn.Module):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels))
        self.beta  = nn.Parameter(torch.zeros(1, channels))
        self.delt  = nn.Parameter(torch.zeros(1, channels))
        self.gamma = nn.Parameter(torch.ones(1, channels))
        self.act   = activation
    def forward(self, x):
        return (self.act((x * self.alpha) + self.delt) * self.gamma) + self.beta


# =====================
#  Helpers
# =====================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
GAN_DIR = "GAN"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def is_power_of_two(n: int) -> bool:
    return (n > 0) and (n & (n - 1) == 0)

def best_gn_groups(ch: int) -> int:
    for g in (8, 7, 6, 5, 4, 3, 2, 1):
        if ch % g == 0:
            return g
    return 1

def clear_gan_images():
    ensure_dir(GAN_DIR)
    for fn in os.listdir(GAN_DIR):
        if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tif', '.tiff')):
            try:
                os.remove(os.path.join(GAN_DIR, fn))
            except Exception:
                pass
def check_and_manage_disk_space(threshold_mb=100):
    """Check free disk space and clear images if below threshold."""
    try:
        stat = shutil.disk_usage(GAN_DIR)
        free_mb = stat.free / (1024 * 1024)
        
        if free_mb < threshold_mb:
            print(f"\n[WARNING] Low disk space: {free_mb:.1f} MB free. Clearing images from {GAN_DIR}...")
            clear_gan_images()
            print("Images cleared. Continuing training...")
            return True
        return False
    except Exception as e:
        print(f"[WARNING] Could not check disk space: {e}")
        return False
def save_config(cfg: dict):
    ensure_dir(GAN_DIR)
    with open(os.path.join(GAN_DIR, 'GD.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)

def load_config() -> dict:
    with open(os.path.join(GAN_DIR, 'GD.json'), 'r', encoding='utf-8') as f:
        return json.load(f)

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        w = getattr(m, 'weight_orig', None)
        if w is None:
            w = m.weight
        nn.init.normal_(w, 0.0, 0.02)
        if getattr(m, 'bias', None) is not None:
            nn.init.zeros_(m.bias)

# =====================
#  Dataset utilities
# =====================
class ImageFolderFlat(Dataset):
    def __init__(self, root: str, image_size: int, channels: int):
        super().__init__()
        self.paths: List[str] = []
        for base, _, files in os.walk(root):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                    self.paths.append(os.path.join(base, fn))
        if not self.paths:
            raise FileNotFoundError(f"No image files found under: {root}")
        self.image_size = image_size
        self.channels = channels
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Lambda(lambda img: convert_mode(img, channels)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * channels, [0.5] * channels)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        im = self.transform(Image.open(p).convert({1:'L',3:'RGB',4:'RGBA'}[self.channels]))
        return im

def convert_mode(img: Image.Image, channels: int) -> Image.Image:
    if channels == 1:
        return img.convert('L')
    if channels == 3:
        return img.convert('RGB')
    if channels == 4:
        return img.convert('RGBA')
    raise ValueError('channels must be 1, 3, or 4')
# Add after the existing gradient_penalty function (around line 1142):
def preresize_dataset(dataset_path: str, target_size: int, channels: int):
    """Pre-resize all images in dataset to target size and save to resized subfolder."""
    resized_dir = os.path.join(dataset_path, "resized")
    
    # Clear and create resized directory
    if os.path.exists(resized_dir):
        print(f"Clearing existing resized directory: {resized_dir}")
        shutil.rmtree(resized_dir)
    
    os.makedirs(resized_dir, exist_ok=True)
    
    # Collect all image files
    image_files = []
    for base, _, files in os.walk(dataset_path):
        # Skip the resized directory itself
        if "resized" in base:
            continue
        for fn in files:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                image_files.append(os.path.join(base, fn))
    
    if not image_files:
        print("No images found to resize!")
        return dataset_path
    
    print(f"Found {len(image_files)} images to resize to {target_size}x{target_size}...")
    
    # Process images
    for i, img_path in enumerate(image_files):
        try:
            # Load image
            img = Image.open(img_path)
            
            # Convert to appropriate mode
            img = convert_mode(img, channels)
            
            # Resize
            img_resized = img.resize((target_size, target_size), Image.BICUBIC)
            
            # Save to resized directory with same filename
            filename = os.path.basename(img_path)
            save_path = os.path.join(resized_dir, filename)
            
            # Handle potential filename conflicts by adding numbers
            if os.path.exists(save_path):
                base_name, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(save_path):
                    save_path = os.path.join(resized_dir, f"{base_name}_{counter}{ext}")
                    counter += 1
            
            # Save with appropriate format
            if channels == 4:
                img_resized.save(save_path, format='PNG')  # PNG supports RGBA
            else:
                img_resized.save(save_path)
            
            if (i + 1) % 100 == 0 or i == 0:
                print(f"Processed {i + 1}/{len(image_files)} images...")
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"Successfully resized {len(image_files)} images to {resized_dir}")
    return resized_dir
def gradient_penalty_r1(D, real, device):
    """R1 gradient penalty on real data."""
    real_copy = real.detach().requires_grad_(True)
    d_real = D(real_copy)
    grads = torch.autograd.grad(
        outputs=d_real.sum(),
        inputs=real_copy,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grads = grads.view(grads.size(0), -1)
    r1_penalty = (grads ** 2).sum(dim=1).mean()
    return r1_penalty

def gradient_penalty_r2(D, fake, device):
    """R2 gradient penalty on fake data."""
    fake_copy = fake.detach().requires_grad_(True)
    d_fake = D(fake_copy)
    grads = torch.autograd.grad(
        outputs=d_fake.sum(),
        inputs=fake_copy,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grads = grads.reshape(grads.size(0), -1)
    r2_penalty = (grads ** 2).sum(dim=1).mean()
    return r2_penalty

def gradient_penalty_r3(D, real, fake, device):
    real = real.to(device)
    fake = fake.to(device)

    real.requires_grad_(True)
    fake.requires_grad_(True)

    real_out = D(real)
    fake_out = D(fake)

    grad_real = torch.autograd.grad(
        outputs=real_out.sum(), inputs=real, create_graph=True
    )[0]
    grad_fake = torch.autograd.grad(
        outputs=fake_out.sum(), inputs=fake, create_graph=True
    )[0]

    diff = grad_real - grad_fake
    r3 = diff.view(diff.size(0), -1).pow(2).sum(1).mean()
    return r3


def make_loader(dataset_name_or_path: str, image_size: int, channels: int, batch_size: int, num_workers: int = 2):
    name = dataset_name_or_path.strip().lower()
    builtin = name in {"cifar10", "cifar100", "mnist", "fashionmnist"}
    
    if builtin:
        if name == 'cifar10':
            ds = tvdatasets.CIFAR10(root='./data', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.Resize(32, interpolation=transforms.InterpolationMode.BICUBIC),
                                        transforms.CenterCrop(32),
                                        transforms.Lambda(lambda img: convert_mode(img, 3)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5]*3, [0.5]*3)
                                    ]))
            resolved_size, resolved_ch = 32, 3
        elif name == 'cifar100':
            ds = tvdatasets.CIFAR100(root='./data', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.Resize(32, interpolation=transforms.InterpolationMode.BICUBIC),
                                         transforms.CenterCrop(32),
                                         transforms.Lambda(lambda img: convert_mode(img, 3)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5]*3, [0.5]*3)
                                     ]))
            resolved_size, resolved_ch = 32, 3
        elif name == 'mnist':
            ds = tvdatasets.MNIST(root='./data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.Resize(32, interpolation=transforms.InterpolationMode.BICUBIC),
                                       transforms.CenterCrop(32),
                                       transforms.Lambda(lambda img: convert_mode(img, 1)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5], [0.5])
                                   ]))
            resolved_size, resolved_ch = 32, 1
        else:
            ds = tvdatasets.FashionMNIST(root='./data', train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.Resize(32, interpolation=transforms.InterpolationMode.BICUBIC),
                                             transforms.CenterCrop(32),
                                             transforms.Lambda(lambda img: convert_mode(img, 1)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5])
                                         ]))
            resolved_size, resolved_ch = 32, 1
    else:
        if not os.path.isdir(dataset_name_or_path):
            raise NotADirectoryError(f"Dataset path is not a directory: {dataset_name_or_path}")
        ds = ImageFolderFlat(dataset_name_or_path, image_size=image_size, channels=channels)
        resolved_size, resolved_ch = image_size, channels
    
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return loader, resolved_size, resolved_ch

# =====================
#  MLP Generator (CIPS-style)
# =====================


import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchLoRAHyperMixer(nn.Module):
    """HyperMixer that generates LoRA weights for each patch location."""
    def __init__(self, z_dim, pos_dim, in_channels, out_channels, rank=8):
        super().__init__()
        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # HyperMixer network conditioned on z + positional embeddings
        input_dim = z_dim + pos_dim
        hidden_dim = 256
        
        self.hypernet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Generate low-rank matrices A and B
        # LoRA weight = B @ A, shape: (out_channels, in_channels)
        self.to_lora_a = nn.Linear(hidden_dim, in_channels * rank)
        self.to_lora_b = nn.Linear(hidden_dim, rank * out_channels)
        
        # Learnable scale for LoRA contribution
        self.lora_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.base_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x, z, pos_embeds):
        """
        Apply patch-specific LoRA transformations.
        
        Args:
            x: (B, in_channels, H, W) - features at grid resolution
            z: (B, z_dim) - raw latent code
            pos_embeds: (H, W, pos_dim) - position embeddings for each grid location
            
        Returns:
            (B, out_channels, H, W) - transformed features
        """
        B, C, H, W = x.shape
        
        # Expand z and position embeddings to all spatial locations
        z_expanded = z.unsqueeze(1).unsqueeze(2).expand(B, H, W, -1)  # (B, H, W, z_dim)
        pos_expanded = pos_embeds.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, pos_dim)
        
        # Concatenate z and position embeddings
        hyper_input = torch.cat([z_expanded, pos_expanded], dim=-1)  # (B, H, W, z_dim+pos_dim)
        hyper_input = hyper_input.reshape(B * H * W, -1)
        
        # Generate LoRA parameters for each spatial location
        features = self.hypernet(hyper_input)  # (B*H*W, hidden_dim)
        
        lora_a = self.to_lora_a(features).view(B, H, W, self.in_channels, self.rank)
        lora_b = self.to_lora_b(features).view(B, H, W, self.rank, self.out_channels)
        
        # Apply LoRA transformation per spatial location
        # Rearrange x for batch matrix multiplication
        x_permuted = x.permute(0, 2, 3, 1)  # (B, H, W, in_channels)
        x_unsqueezed = x_permuted.unsqueeze(-2)  # (B, H, W, 1, in_channels)
        
        # Apply: x @ A @ B for each location
        temp = torch.matmul(x_unsqueezed, lora_a)  # (B, H, W, 1, rank)
        out = torch.matmul(temp, lora_b)  # (B, H, W, 1, out_channels)
        out = out.squeeze(-2)  # (B, H, W, out_channels)
        
        # Permute back to (B, C, H, W) format
        out = out.permute(0, 3, 1, 2)  # (B, out_channels, H, W)
        
        # Scale the LoRA contribution
        out = out * self.lora_scale
        
        return out


class ModulatedConv1x1(nn.Module):
    """1x1 convolution with weight modulation (StyleGAN-style)."""
    def __init__(self, in_channels, out_channels, style_dim, demodulate=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.demodulate = demodulate
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.style_transform = nn.Linear(style_dim, in_channels)
        
        # Initialize
        nn.init.kaiming_normal_(self.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
    
    def forward(self, x, style):
        batch, in_c, h, w = x.shape
        
        # Get modulation from style
        s = self.style_transform(style).view(batch, 1, in_c, 1, 1)  # (B, 1, in_c, 1, 1)
        
        # Modulate weights
        weight = self.weight.unsqueeze(0) * s  # (B, out_c, in_c, 1, 1)
        
        # Demodulation
        if self.demodulate:
            d = torch.rsqrt((weight ** 2).sum(dim=[2, 3, 4], keepdim=True) + 1e-8)
            weight = weight * d
        
        # Reshape for group convolution
        weight = weight.view(batch * self.out_channels, in_c, 1, 1)
        x = x.view(1, batch * in_c, h, w)
        
        # Apply convolution
        out = F.conv2d(x, weight, padding=0, groups=batch)
        out = out.view(batch, self.out_channels, h, w)
        out = out + self.bias.view(1, -1, 1, 1)
        
        return out




class MLPGenerator(nn.Module):
    """CIPS-style generator with coordinate-based synthesis supporting patch-based generation.
    
    When patch_size > 1, uses a LoRA HyperMixer to generate patch-specific weights
    conditioned on raw z values and positional embeddings.
    """
    def __init__(self, zdim: int, img_ch: int, hidden_dim: int, num_layers: int, 
                 img_size: int, patch_size: int = 1, lora_rank: int = 8):
        super().__init__()
        self.zdim = zdim
        self.img_ch = img_ch
        self.img_size = img_size
        self.patch_size = max(1, patch_size)  # Ensure at least 1
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lora_rank = lora_rank
        self.use_patch_lora = self.patch_size > 1  # Enable LoRA for patches
        
        # Calculate grid size (number of patches or pixels)
        if self.patch_size > 1:
            assert img_size % patch_size == 0, f"Image size {img_size} must be divisible by patch size {patch_size}"
            self.grid_size = img_size // patch_size
        else:
            self.grid_size = img_size
        
        # Fourier features dimension
        self.fourier_dim = 256
        self.coord_embed_dim = 256
        self.total_pos_dim = self.fourier_dim + self.coord_embed_dim
        
        # Mapping network: z -> w (style vector)
        mapping_layers = []
        for i in range(4):
            in_dim = zdim if i == 0 else hidden_dim
            mapping_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU()
            ])
        self.mapping_network = nn.Sequential(*mapping_layers)
        self.style_dim = hidden_dim
        self.rev = nn.Linear(zdim, hidden_dim)
        
        # Fourier features (learnable)
        self.fourier_matrix = nn.Parameter(torch.randn(2, self.fourier_dim // 2))
        
        # Coordinate embeddings (learnable per grid position - patches or pixels)
        self.coord_embeddings = nn.Parameter(
            torch.randn(self.grid_size, self.grid_size, self.coord_embed_dim)
        )
        
        # Initial projection from positional encoding
        self.input_proj = nn.Conv2d(self.total_pos_dim, hidden_dim, 1)
        
        # Modulated 1x1 conv layers (operate on patch/pixel grid)
        self.mod_convs = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        # LoRA HyperMixers for patch-based generation
        if self.use_patch_lora:
            self.lora_mixers = nn.ModuleList()
        
        for i in range(num_layers):
            in_ch = hidden_dim
            out_ch = hidden_dim
            
            self.mod_convs.append(
                ModulatedConv1x1(in_ch, out_ch, self.style_dim, demodulate=True)
            )
            self.activations.append(SineEvenLReLU(0.2))
            
            # Add LoRA HyperMixer for each layer when using patches
            if self.use_patch_lora:
                self.lora_mixers.append(
                    PatchLoRAHyperMixer(
                        z_dim=zdim,
                        pos_dim=self.coord_embed_dim,  # Use coord embeddings for position
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        rank=lora_rank
                    )
                )
        
        # Final to RGB
        out_channels = img_ch * (self.patch_size ** 2) if self.patch_size > 1 else img_ch
        self.to_rgb = nn.Conv2d(hidden_dim, out_channels, 1)
        
        # LoRA HyperMixer for final RGB layer when using patches
        if self.use_patch_lora:
            self.lora_to_rgb = PatchLoRAHyperMixer(
                z_dim=zdim,
                pos_dim=self.coord_embed_dim,
                in_channels=hidden_dim,
                out_channels=out_channels,
                rank=lora_rank
            )
    
    def get_positional_encoding(self, batch, device):
        """Generate positional encoding at grid resolution (patches or pixels)."""
        # Normalized coordinates for the grid
        coords_y = torch.linspace(-1, 1, self.grid_size, device=device)
        coords_x = torch.linspace(-1, 1, self.grid_size, device=device)
        grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing='ij')
        coords = torch.stack([grid_x, grid_y], dim=0)  # (2, grid_size, grid_size)
        
        # Fourier features: sin/cos(B @ coords)
        coords_flat = coords.view(2, -1)  # (2, grid_size²)
        fourier_input = self.fourier_matrix.T @ coords_flat  # (fourier_dim//2, grid_size²)
        fourier_features = torch.cat([
            torch.sin(fourier_input),
            torch.cos(fourier_input)
        ], dim=0)  # (fourier_dim, grid_size²)
        fourier_features = fourier_features.view(self.fourier_dim, self.grid_size, self.grid_size)
        
        # Coordinate embeddings (learnable per grid position)
        coord_embed = self.coord_embeddings.permute(2, 0, 1)  # (coord_embed_dim, grid_size, grid_size)
        
        # Concatenate
        pos_encoding = torch.cat([fourier_features, coord_embed], dim=0)  # (total_pos_dim, grid_size, grid_size)
        pos_encoding = pos_encoding.unsqueeze(0).expand(batch, -1, -1, -1)
        
        return pos_encoding
    
    def forward(self, z):
        """Generate image from latent code.
        
        When patch_size > 1, uses LoRA HyperMixer to generate patch-specific
        transformations conditioned on z and positional embeddings.
        """
        batch = z.size(0)
        device = z.device
        
        # Map latent to style
        w = self.rev(z) * 2
        
        # Get positional encoding at grid resolution
        pos_enc = self.get_positional_encoding(batch, device)  # (B, total_pos_dim, grid_size, grid_size)
        
        # Initial projection
        x = self.input_proj(pos_enc)  # (B, hidden_dim, grid_size, grid_size)
        
        # Get coordinate embeddings for LoRA conditioning (H, W, coord_embed_dim)
        coord_embeds = self.coord_embeddings  # (grid_size, grid_size, coord_embed_dim)
        
        # Process through modulated 1x1 convs with optional LoRA HyperMixer
        for i, (mod_conv, activation) in enumerate(zip(self.mod_convs, self.activations)):
            # Standard modulated convolution
            x = mod_conv(x, w)
            
            # Add patch-specific LoRA transformation when using patches
            if self.use_patch_lora:
                lora_output = self.lora_mixers[i](x, z, coord_embeds)
                x = x + lora_output  # Residual connection
            
            x = activation(x)
        
        # Final RGB conversion
        img = self.to_rgb(x)
        
        # Add LoRA transformation for RGB layer when using patches
        if self.use_patch_lora:
            lora_rgb = self.lora_to_rgb(x, z, coord_embeds)
            img = img + lora_rgb
        
        # If patch_size > 1, rearrange patches to form the full image
        # img shape: (B, img_ch * patch_size², grid_size, grid_size)
        # pixel_shuffle converts to: (B, img_ch, img_size, img_size)
        if self.patch_size > 1:
            img = F.pixel_shuffle(img, self.patch_size)
        
        return img




class CREPSGenerator(nn.Module):
    """Column-Row Entangled Pixel Synthesis generator.

    This is a compact CREPS-inspired generator option for this trainer.  It avoids
    spatial convolutions in the synthesis trunk by producing separate thick row
    and column feature encodings, composing them with a dot product, and fusing
    layer-wise composed maps with lightweight 1x1 decoder blocks.
    """
    def __init__(self, zdim: int, img_ch: int, hidden_dim: int, num_layers: int,
                 img_size: int, thickness: int = 8, decoder_depth: int = 4):
        super().__init__()
        self.zdim = zdim
        self.img_ch = img_ch
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.img_size = img_size
        self.thickness = max(1, thickness)
        self.coord_dim = 128
        self.embed_dim = 128
        self.pos_dim = self.coord_dim + self.embed_dim

        self.mapping_network = nn.Sequential(
            nn.Linear(zdim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.style_dim = hidden_dim

        # Separate row/column Fourier bases and learnable position embeddings.
        self.row_fourier = nn.Parameter(torch.randn(1, self.coord_dim // 2))
        self.col_fourier = nn.Parameter(torch.randn(1, self.coord_dim // 2))
        self.row_embeddings = nn.Parameter(torch.randn(img_size, self.embed_dim) * 0.02)
        self.col_embeddings = nn.Parameter(torch.randn(img_size, self.embed_dim) * 0.02)

        self.row_input = nn.Conv2d(self.pos_dim, hidden_dim, 1)
        self.col_input = nn.Conv2d(self.pos_dim, hidden_dim, 1)
        self.row_blocks = nn.ModuleList()
        self.col_blocks = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(num_layers):
            self.row_blocks.append(ModulatedConv1x1(hidden_dim, hidden_dim, self.style_dim, demodulate=True))
            self.col_blocks.append(ModulatedConv1x1(hidden_dim, hidden_dim, self.style_dim, demodulate=True))
            layers = []
            dec_hidden = max(32, hidden_dim // 2)
            in_ch = hidden_dim
            for _j in range(max(1, decoder_depth - 1)):
                layers.extend([nn.Conv2d(in_ch, dec_hidden, 1), nn.LeakyReLU(0.2, inplace=True)])
                in_ch = dec_hidden
            layers.append(nn.Conv2d(in_ch, hidden_dim, 1))
            self.decoders.append(nn.Sequential(*layers))

        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, max(64, hidden_dim // 2), 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(max(64, hidden_dim // 2), max(32, hidden_dim // 4), 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(max(32, hidden_dim // 4), img_ch, 1),
        )

    def _axis_encoding(self, axis: str, batch: int, device: torch.device) -> torch.Tensor:
        coords = torch.linspace(-1, 1, self.img_size, device=device).unsqueeze(1)
        if axis == 'row':
            fourier_matrix = self.row_fourier
            learned = self.row_embeddings
        else:
            fourier_matrix = self.col_fourier
            learned = self.col_embeddings
        fourier_input = coords @ fourier_matrix
        fourier = torch.cat([torch.sin(fourier_input), torch.cos(fourier_input)], dim=1)
        pos = torch.cat([fourier, learned.to(device)], dim=1)  # (R, pos_dim)
        pos = pos.transpose(0, 1).view(1, self.pos_dim, self.img_size, 1)
        pos = pos.expand(batch, -1, -1, self.thickness)
        return pos

    def _compose(self, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        # row/col are (B, C, R, D).  Output is (B, C, R, R).
        return torch.einsum('bchd,bcwd->bchw', row, col) / math.sqrt(float(self.thickness))

    def forward(self, z):
        batch = z.size(0)
        device = z.device
        w = self.mapping_network(z)
        row = self.row_input(self._axis_encoding('row', batch, device))
        col = self.col_input(self._axis_encoding('col', batch, device))

        fused = None
        for row_block, col_block, decoder in zip(self.row_blocks, self.col_blocks, self.decoders):
            row = SineEvenLReLU(0.2)(row_block(row, w))
            col = SineEvenLReLU(0.2)(col_block(col, w))
            composed = self._compose(row, col)
            fused = composed if fused is None else decoder(fused) + composed

        img = self.refine(fused)
        return img






# =====================
#  ViT Generator
# =====================
# =====================
#  ViT Generator with L2 Attention
# =====================
class SelfModulatedLayerNorm(nn.Module):
    """Self-Modulated LayerNorm from ViTGAN (Eq. 13 in paper)."""
    def __init__(self, normalized_shape: int, latent_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        # Affine parameters computed from latent w
        self.gamma_mlp = nn.Linear(latent_dim, normalized_shape)
        self.beta_mlp = nn.Linear(latent_dim, normalized_shape)
        
    def forward(self, x, w):
        """
        Args:
            x: (B, N, D) or (B, D) features to normalize
            w: (B, latent_dim) latent code
        """
        normalized = self.norm(x)
        gamma = self.gamma_mlp(w)
        beta = self.beta_mlp(w)
        
        # Broadcast for sequence dimension if needed
        if x.dim() == 3:  # (B, N, D)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            
        return gamma * normalized + beta


class ViTGANTransformerBlock(nn.Module):
    """Transformer block with Self-Modulated LayerNorm and L2 Attention."""
    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int, use_conv_proj: bool = True):
        super().__init__()
        self.sln1 = SelfModulatedLayerNorm(hidden_dim, hidden_dim)
        # Replace MultiheadAttention with L2SelfAttention
        self.attn = L2SelfAttention(hidden_dim, num_heads, use_conv_proj=use_conv_proj)
        self.sln2 = SelfModulatedLayerNorm(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim, bias=False),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim, bias=False)
        )
        
    def forward(self, x, w, h=None, w_spatial=None):
        """
        Args:
            x: (B, N, D) token sequence
            w: (B, D) latent code
            h: height of patch grid (for conv projection)
            w_spatial: width of patch grid (for conv projection)
        """
        # h'ℓ = MSA(SLN(hℓ-1, w)) + hℓ-1
        h_prime = self.sln1(x, w)
        h_prime = self.attn(h_prime, h=h, w=w_spatial)  # L2 attention with conv projection
        h_prime = h_prime + x
        
        # hℓ = MLP(SLN(h'ℓ, w)) + h'ℓ
        h_out = self.sln2(h_prime, w)
        h_out = self.mlp(h_out)
        h_out = h_out + h_prime
        
        return h_out


class ViTGenerator(nn.Module):
    """ViTGAN Generator with L2 attention and direct patch generation."""
    def __init__(self, zdim: int, img_ch: int, hidden_dim: int, num_heads: int, 
                 num_layers: int, img_size: int, patch_size: int, overlap: int = 0,
                 use_conv_proj: bool = True):
        super().__init__()
        self.zdim = zdim
        self.img_ch = img_ch
        self.img_size = img_size
        self.patch_size = patch_size
        self.overlap = overlap
        self.hidden_dim = hidden_dim
        self.use_conv_proj = use_conv_proj
        
        # Calculate stride and number of patches (same as discriminator)
        self.stride = patch_size - overlap
        self.patches_per_side = (img_size - patch_size) // self.stride + 1
        self.num_patches = self.patches_per_side ** 2
        
        # Mapping network: z -> w (3-layer MLP from paper)
        self.mapping_network = nn.Sequential(
            nn.Linear(zdim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rev = nn.Linear(zdim, hidden_dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim) * 0.02)
        
        # Transformer blocks with Self-Modulated LayerNorm and L2 Attention
        self.transformer_blocks = nn.ModuleList([
            ViTGANTransformerBlock(hidden_dim, num_heads, hidden_dim * 4, use_conv_proj=use_conv_proj)
            for _ in range(num_layers)
        ])
        
        # Final SLN
        self.final_sln = SelfModulatedLayerNorm(hidden_dim, hidden_dim)
        
        # Direct patch generation
        patch_pixels = patch_size * patch_size * img_ch
        self.patch_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, patch_pixels, bias=False)
        )
        
        self.out_act = nn.Identity()  # nn.Tanh()
        
    def forward(self, z):
        batch = z.size(0)
        
        # Mapping network: z -> w
        w = self.rev(z) * 2  # self.mapping_network(z)
        
        # Initialize with positional embeddings
        tokens = self.pos_embed.expand(batch, -1, -1)  # (B, N, hidden_dim)
        
        # Apply transformer blocks with SLN and L2 attention
        # Pass spatial dimensions for convolutional projection
        h = w_spatial = self.patches_per_side if self.use_conv_proj else None
        
        for block in self.transformer_blocks:
            tokens = block(tokens, w, h=h, w_spatial=w_spatial)
        
        # Final SLN
        tokens = self.final_sln(tokens, w)  # (B, N, hidden_dim)
        
        # Generate patches directly from tokens
        patches = self.patch_generator(tokens)  # (B, N, patch_size^2 * img_ch)
        
        # Reshape: (B, N, P*P*C) -> (B, N, C*P*P)
        # Then transpose for F.fold: (B, C*P*P, N)
        patches = patches.transpose(1, 2)  # (B, patch_size^2 * img_ch, N)
        
        # Use F.fold to reconstruct image from overlapping patches
        # F.fold handles overlapping regions by summing contributions
        img = F.fold(
            patches,
            output_size=(self.img_size, self.img_size),
            kernel_size=self.patch_size,
            stride=self.stride
        )  # (B, C, H, W)
        
        # When patches overlap, F.fold sums the overlapping regions
        # We need to divide by the number of overlaps to get the average
        if self.overlap > 0:
            # Create a ones tensor to count overlaps
            ones = torch.ones_like(patches)
            overlap_count = F.fold(
                ones,
                output_size=(self.img_size, self.img_size),
                kernel_size=self.patch_size,
                stride=self.stride
            )  # (B, C, H, W)
            
            # Average the overlapping regions
            img = img / overlap_count
        
        img = self.out_act(img)
        
        return img


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math

class ImprovedSpectralNorm(nn.Module):
    """Improved Spectral Normalization from ViTGAN (Eq. 7)"""
    def __init__(self, module):
        super().__init__()
        self.module = spectral_norm(module)
        # Store initial spectral norm
        with torch.no_grad():
            weight = self.module.weight
            u = weight.new_empty(weight.size(0)).normal_(0, 1)
            v = weight.new_empty(weight.size(1)).normal_(0, 1)
            for _ in range(10):  # Power iteration
                v = F.normalize(torch.mv(weight.t(), u), dim=0)
                u = F.normalize(torch.mv(weight, v), dim=0)
            sigma_init = torch.dot(u, torch.mv(weight, v))
            self.register_buffer('sigma_init', sigma_init)
    
    def forward(self, *args, **kwargs):
        # Apply standard spectral norm, then scale by initial spectral norm
        return self.module(*args, **kwargs) * self.sigma_init

class L2SelfAttention(nn.Module):
    """L2 attention with convolutional projection (ViTGAN Section 4.1)"""
    def __init__(self, hidden_dim, num_heads, use_conv_proj=True):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_conv_proj = use_conv_proj
        
        # Tied weights for query and key (Wq = Wk)
        self.qk_proj = ImprovedSpectralNorm(nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.v_proj = ImprovedSpectralNorm(nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.out_proj = ImprovedSpectralNorm(nn.Linear(hidden_dim, hidden_dim, bias=False))
        
        # Convolutional projection (3x3 conv for Q, K, V)
        if use_conv_proj:
            self.conv_q = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
            self.conv_k = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
            self.conv_v = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
    
    def forward(self, x, h=None, w=None):
        B, N, C = x.shape
        
        # If using conv projection, need spatial dimensions
        if self.use_conv_proj and h is not None and w is not None:
            # Reshape to spatial format
            x_spatial = x.transpose(1, 2).reshape(B, C, h, w)
            
            # Apply convolutions
            q_spatial = self.conv_q(x_spatial)
            k_spatial = self.conv_k(x_spatial)
            v_spatial = self.conv_v(x_spatial)
            
            # Reshape back
            q = q_spatial.reshape(B, C, N).transpose(1, 2)
            k = k_spatial.reshape(B, C, N).transpose(1, 2)
            v = v_spatial.reshape(B, C, N).transpose(1, 2)
        else:
            q = k = x
            v = x
        
        # Apply tied projection for Q and K
        q = self.qk_proj(q).reshape(B, N, self.num_heads, self.head_dim)
        k = self.qk_proj(k).reshape(B, N, self.num_heads, self.head_dim)  # Same weights!
        v = self.v_proj(v).reshape(B, N, self.num_heads, self.head_dim)
        
        # Transpose for attention: (B, num_heads, N, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # L2 attention: -||q - k||^2 = -||q||^2 - ||k||^2 + 2<q,k>
        # Compute pairwise L2 distances
        q_norm = (q ** 2).sum(dim=-1, keepdim=True)  # (B, H, N, 1)
        k_norm = (k ** 2).sum(dim=-1, keepdim=True).transpose(-2, -1)  # (B, H, 1, N)
        qk_dot = torch.matmul(q, k.transpose(-2, -1))  # (B, H, N, N)
        
        # Distance = ||q||^2 + ||k||^2 - 2<q,k>
        dist = q_norm + k_norm - 2 * qk_dot
        
        # Attention weights: softmax(-dist / sqrt(d))
        attn = F.softmax(-dist * self.scale, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        out = self.out_proj(out)
        return out

class ViTGANTransformerBlocka(nn.Module):
    """Transformer block with L2 attention and improved normalization"""
    def __init__(self, hidden_dim, num_heads, use_conv_proj=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = L2SelfAttention(hidden_dim, num_heads, use_conv_proj)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # MLP with ISN
        self.mlp = nn.Sequential(
            ImprovedSpectralNorm(nn.Linear(hidden_dim, hidden_dim * 4, bias=False)),
            nn.GELU(),
            ImprovedSpectralNorm(nn.Linear(hidden_dim * 4, hidden_dim, bias=False))
        )
    
    def forward(self, x, h=None, w=None):
        # Pre-norm architecture
        x = x + self.attn(self.ln1(x), h, w)
        x = x + self.mlp(self.ln2(x))
        return x

class ViTDiscriminator(nn.Module):
    """Vision Transformer discriminator with ViTGAN improvements."""
    def __init__(self, img_ch: int, hidden_dim: int, num_heads: int, 
                 num_layers: int, img_size: int, patch_size: int, overlap: int):
        super().__init__()
        self.img_ch = img_ch
        self.img_size = img_size
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        
        # Calculate number of patches with overlap
        self.num_patches = ((img_size - patch_size) // self.stride + 1) ** 2
        self.spatial_size = int(math.sqrt(self.num_patches))
        
        patch_dim = img_ch * patch_size * patch_size
        
        # Patch embedding with ISN
        self.patch_embed = ImprovedSpectralNorm(nn.Linear(patch_dim, hidden_dim, bias=False))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim) * 0.02)
        
        # Transformer blocks with L2 attention and conv projection
        self.transformer = nn.ModuleList([
            ViTGANTransformerBlocka(hidden_dim, num_heads, use_conv_proj=True) for _ in range(num_layers)
        ])
        
        # Final norm
        self.ln_final = nn.LayerNorm(hidden_dim)
        
        # Classification head with ISN
        self.classifier = ImprovedSpectralNorm(nn.Linear(hidden_dim, 1, bias=False))
        
    def forward(self, x):
        batch = x.size(0)
        
        # Extract overlapping patches
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.stride)
        patches = patches.transpose(1, 2)  # (B, num_patches, patch_dim)
        
        # Embed patches
        tokens = self.patch_embed(patches)
        tokens = tokens + self.pos_embed
        
        # Apply transformer blocks
        for block in self.transformer:
            tokens = block(tokens, h=self.spatial_size, w=self.spatial_size)
        
        # Final normalization
        tokens = self.ln_final(tokens)
        
        # Global average pooling
        pooled = tokens.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled).squeeze(-1)
        
        return logits


# =====================
#  GRU Generator
# =====================
class GRUGenerator(nn.Module):
    """Autoregressive GRU generator producing column-by-column."""
    def __init__(self, zdim: int, img_ch: int, hidden_dim: int, num_layers: int, img_size: int):
        super().__init__()
        self.zdim = zdim
        self.img_ch = img_ch
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU processes one column at a time
        # Input: previous column (img_ch * img_size) + latent (zdim)
        input_dim = img_ch * img_size + zdim
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, img_ch * img_size)
        self.out_act = ETActFlat(img_ch * img_size, nn.Tanh())
        
    def forward(self, z):
        batch = z.size(0)
        device = z.device
        
        # Initialize with zeros for first column
        prev_col = torch.zeros(batch, self.img_ch * self.img_size, device=device)
        
        columns = []
        h = None
        
        for i in range(self.img_size):
            # Concatenate previous column with latent
            gru_input = torch.cat([prev_col, z], dim=-1).unsqueeze(1)  # (B, 1, input_dim)
            
            # GRU step
            out, h = self.gru(gru_input, h)  # out: (B, 1, hidden_dim)
            
            # Generate column
            col = self.fc_out(out.squeeze(1))  # (B, img_ch * img_size)
            col = self.out_act(col)
            
            columns.append(col)
            prev_col = col
        
        # Stack columns to form image
        columns = torch.stack(columns, dim=-1)  # (B, img_ch * img_size, img_size)
        img = columns.view(batch, self.img_ch, self.img_size, self.img_size)
        
        return img


class GRUDiscriminator(nn.Module):
    """Bidirectional GRU discriminator scanning image columns."""
    def __init__(self, img_ch: int, hidden_dim: int, num_layers: int, img_size: int):
        super().__init__()
        self.img_ch = img_ch
        self.img_size = img_size
        
        # Input: one column (img_ch * img_size)
        input_dim = img_ch * img_size
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, 
                         batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        batch = x.size(0)
        
        # Reshape to columns
        columns = x.view(batch, self.img_ch * self.img_size, self.img_size)
        columns = columns.transpose(1, 2)  # (B, img_size, img_ch * img_size)
        
        # Process with bidirectional GRU
        out, _ = self.gru(columns)  # (B, img_size, hidden_dim * 2)
        
        # Aggregate (mean pooling over columns)
        pooled = out.mean(dim=1)  # (B, hidden_dim * 2)
        logits = self.classifier(pooled).squeeze(-1)
        
        return logits


# =====================
#  Swin Models (StyleSwin-based)
# =====================
class SwinBlock(nn.Module):
    """Swin Transformer block with double attention and style injection."""
    def __init__(self, dim: int, num_heads: int, window_size: int, shift: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift = shift
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Multi-head attention (we'll implement double attention)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def window_partition(self, x, h, w):
        """Partition into non-overlapping windows."""
        b, hw, c = x.shape
        x = x.view(b, h, w, c)
        
        # If spatial dims are smaller than window size, use entire feature map as one window
        if h < self.window_size or w < self.window_size:
            return x.view(b, h * w, c)
        
        if self.shift:
            # Shifted window partition
            x = torch.roll(x, shifts=(-self.window_size // 2, -self.window_size // 2), dims=(1, 2))
        
        # Partition into windows
        num_h = h // self.window_size
        num_w = w // self.window_size
        x = x.view(b, num_h, self.window_size, num_w, self.window_size, c)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(b * num_h * num_w, self.window_size * self.window_size, c)
        
        return x
    
    def window_reverse(self, x, h, w):
        """Reverse window partition."""
        # If spatial dims are smaller than window size, already in correct shape
        if h < self.window_size or w < self.window_size:
            b = x.shape[0]
            return x.view(b, h * w, self.dim)
        
        num_h = h // self.window_size
        num_w = w // self.window_size
        b = x.shape[0] // (num_h * num_w)
        
        x = x.view(b, num_h, num_w, self.window_size, self.window_size, self.dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(b, h, w, self.dim)
        
        if self.shift:
            # Reverse shift
            x = torch.roll(x, shifts=(self.window_size // 2, self.window_size // 2), dims=(1, 2))
        
        x = x.view(b, h * w, self.dim)
        return x
    
    def forward(self, x, h, w):
        """
        x: (B, H*W, C)
        h, w: spatial dimensions
        """
        shortcut = x
        
        # Norm + attention
        x = self.norm1(x)
        
        # Window partition
        x_windows = self.window_partition(x, h, w)
        
        # Attention within windows
        attn_out, _ = self.attn(x_windows, x_windows, x_windows)
        
        # Reverse windows
        x = self.window_reverse(attn_out, h, w)
        
        # Residual
        x = shortcut + x
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class StyleSwinGenerator(nn.Module):
    """StyleSwin generator using Swin transformer blocks with style injection."""
    def __init__(self, zdim: int, img_ch: int, g_last_hidden: int, fmap_max: int, 
                 img_size: int, window_size: int = 8):
        super().__init__()
        assert is_power_of_two(img_size) and img_size >= 8
        self.zdim = zdim
        self.img_size = img_size
        self.img_ch = img_ch
        self.window_size = window_size
        
        # Calculate number of upsampling stages
        s0 = 4  # Start from 4x4
        ups = int(math.log2(img_size)) - 2  # Number of 2x upsamples
        
        # Calculate channel dimensions (decreasing with resolution)
        start_ch = min(fmap_max, g_last_hidden * (2 ** ups))
        chs = [start_ch]
        for _ in range(ups):
            chs.append(max(g_last_hidden, chs[-1] // 2))
        
        # Style mapping network
        self.style_net = nn.Sequential(
            nn.Linear(zdim, zdim),
            nn.SiLU(),
            nn.Linear(zdim, zdim),
            nn.SiLU(),
            nn.Linear(zdim, zdim),
            nn.SiLU(),
            nn.Linear(zdim, zdim)
        )
        
        # Constant input
        self.const_input = nn.Parameter(torch.randn(1, chs[0], s0, s0))
        
        # Stages (2 blocks per resolution)
        self.stages = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        # Sinusoidal positional encoding parameters
        self.spe_params = nn.ParameterList()
        
        cur_res = s0
        for i in range(len(chs)):
            ch = chs[i]
            num_heads = max(1, ch // 64)
            
            # Two Swin blocks per resolution
            stage_blocks = nn.ModuleList([
                SwinBlock(ch, num_heads, window_size, shift=False),
                SwinBlock(ch, num_heads, window_size, shift=True)
            ])
            
            # AdaIN parameters for each block
            stage_adain = nn.ModuleList([
                nn.Linear(zdim, ch * 2),  # scale and bias
                nn.Linear(zdim, ch * 2)
            ])
            
            self.stages.append(nn.ModuleDict({
                'blocks': stage_blocks,
                'adain': stage_adain
            }))
            
            # Sinusoidal positional encoding
            #self.spe_params.append(nn.Parameter(torch.randn(1, cur_res * cur_res, ch) * 0.02))
            
            # To RGB
            self.to_rgbs.append(nn.Conv2d(ch, img_ch, 1, 1, 0))
            
            # Upsampler (if not last stage)
            if i < len(chs) - 1:
                self.upsamplers.append(nn.Sequential(
                    #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.ConvTranspose2d(ch, chs[i + 1], 4, 2, 1),
                    #nn.PReLU(chs[i + 1]),
                ))
                cur_res *= 2
            else:
                self.upsamplers.append(None)
        
        self.out_act = nn.Identity()#Tanh()
    
    def forward(self, z):
        batch = z.size(0)
        
        # Map to style space
        style_w = z*2#self.style_net(z)
        
        # Start from constant input
        x = self.const_input.expand(batch, -1, -1, -1)
        
        cur_res = 4
        for i, stage in enumerate(self.stages):
            # Reshape to sequence
            _, c, h, w = x.shape
            x_seq = x.view(batch, c, h * w).transpose(1, 2)  # (B, H*W, C)
            
            # Add sinusoidal positional encoding
            #x_seq = x_seq + self.spe_params[i]
            
            # Apply Swin blocks with AdaIN
            for block, adain_layer in zip(stage['blocks'], stage['adain']):
                # AdaIN
                style_params = adain_layer(style_w).unsqueeze(1)  # (B, 1, 2C)
                scale, bias = style_params.chunk(2, dim=-1)  # Each (B, 1, C)
                
                # Normalize
                mean = x_seq.mean(dim=-1, keepdim=True)
                std = x_seq.std(dim=-1, keepdim=True) + 1e-8
                x_seq = (x_seq - mean) / std
                
                # Apply style
                x_seq = x_seq * (scale + 1.0) + bias
                
                # Swin block
                x_seq = block(x_seq, h, w)
            
            # Reshape back to spatial
            x = x_seq.transpose(1, 2).view(batch, -1, h, w)
            
            # Upsample for next stage (if not last)
            if self.upsamplers[i] is not None:
                x = self.upsamplers[i](x)
                cur_res *= 2
        
        # Final to RGB
        img = self.to_rgbs[-1](x)
        img = self.out_act(img)
        
        return img


class Blur(nn.Module):
    """Blur layer using [1, 3, 3, 1] kernel (StyleGAN2)"""
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([1, 3, 3, 1], dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel[None, None, :, :])
        
    def forward(self, x):
        # Apply blur per channel
        n, c, h, w = x.shape
        x = x.view(n * c, 1, h, w)
        # Pad to maintain size
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        x = F.conv2d(x, self.kernel)
        return x.view(n, c, h, w)


class SwinDiscriminator(nn.Module):
    """Swin-based discriminator with optional from_rgb skip connections."""
    def __init__(self, img_ch: int, d_first_hidden: int, fmap_max: int, img_size: int,
                 window_size: int = 8, from_rgb_res: Optional[List[int]] = None, 
                 use_blur: bool = False):
        super().__init__()
        assert is_power_of_two(img_size) and img_size >= 8
        
        self.img_size = img_size
        self.window_size = window_size
        self.use_from_rgb = from_rgb_res is not None and len(from_rgb_res) > 0
        self.from_rgb_res = set(from_rgb_res or [])
        self.use_blur = use_blur
        
        # Calculate downsampling stages
        s0 = 4
        downs = int(math.log2(img_size)) - 2
        
        # Calculate channel progression
        chs = [d_first_hidden]
        for _ in range(downs - 1):
            chs.append(min(fmap_max, chs[-1] * 2))
        last_ch = min(fmap_max, chs[-1] * 2) if downs > 0 else d_first_hidden
        
        # From RGB layer for full resolution
        self.from_rgb = nn.Conv2d(img_ch, d_first_hidden, 1, 1, 0)
        
        # Build stages
        self.stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        cur_res = img_size
        for i, ch in enumerate(chs):
            num_heads = max(1, ch // 64)
            
            # Two Swin blocks per resolution
            stage_blocks = nn.ModuleList([
                SwinBlock(ch, num_heads, window_size, shift=False),
                SwinBlock(ch, num_heads, window_size, shift=True)
            ])
            
            self.stages.append(stage_blocks)
            
            # Downsampler (if not last stage)
            if i < len(chs) - 1:
                self.downsamplers.append(nn.Sequential(
                    nn.Conv2d(ch, chs[i + 1], 3, 1, 1),
                    nn.AvgPool2d(2)
                ))
                cur_res //= 2
            else:
                self.downsamplers.append(None)
        
        # Build from_rgb layers for skip connections
        self.from_rgb_layers = nn.ModuleDict()
        if self.use_from_rgb:
            res_to_channels = {}
            cur_res = img_size // 2  # Skip first resolution
            for i, ch in enumerate(chs):
                res_to_channels[cur_res] = ch
                if i < len(chs) - 1:
                    cur_res //= 2
            
            for res in self.from_rgb_res:
                if res < img_size and res >= 4 and res in res_to_channels:
                    target_ch = res_to_channels[res]
                    layers = []
                    if self.use_blur:
                        layers.append(Blur())
                    layers.append(nn.Conv2d(img_ch, target_ch, 1, 1, 0))
                    self.from_rgb_layers[str(res)] = nn.Sequential(*layers)
        
        # Final layers
        self.final_conv = nn.Conv2d(chs[-1], last_ch, 3, 1, 1)
        self.final_act = nn.SiLU()
        self.out = spectral_norm(nn.Linear(last_ch, 1))
    
    def forward(self, img):
        original_img = img
        batch = img.size(0)
        
        # Initial from_rgb
        x = self.from_rgb(img)
        
        cur_res = self.img_size
        for i, (stage, downsampler) in enumerate(zip(self.stages, self.downsamplers)):
            # Apply from_rgb skip if applicable (before processing this resolution)
            if self.use_from_rgb and i > 0 and cur_res in self.from_rgb_res:
                res_key = str(cur_res)
                if res_key in self.from_rgb_layers:
                    downsampled = F.interpolate(original_img, size=(cur_res, cur_res),
                                               mode='bilinear', align_corners=False)
                    rgb_path = self.from_rgb_layers[res_key](downsampled)
                    x = x + rgb_path
            
            # Reshape to sequence for Swin blocks
            _, c, h, w = x.shape
            x_seq = x.view(batch, c, h * w).transpose(1, 2)  # (B, H*W, C)
            
            # Apply Swin blocks
            for block in stage:
                x_seq = block(x_seq, h, w)
            
            # Reshape back to spatial
            x = x_seq.transpose(1, 2).view(batch, c, h, w)
            
            # Downsample for next stage
            if downsampler is not None:
                x = downsampler(x)
                cur_res //= 2
        
        # Final layers
        x = self.final_conv(x)
        x = self.final_act(x)
        x = x.mean(dim=[2, 3])  # Global average pooling
        x = self.out(x)
        
        return x.view(-1)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Optional, Literal


def build_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """Static 2D sine-cosine positional embeddings."""
    dim_y = embed_dim // 2
    dim_x = embed_dim - dim_y

    def _pe_1d(length: int, dim: int) -> torch.Tensor:
        pos = torch.arange(length).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    pe_y = _pe_1d(grid_h, dim_y).unsqueeze(1).expand(grid_h, grid_w, dim_y)
    pe_x = _pe_1d(grid_w, dim_x).unsqueeze(0).expand(grid_h, grid_w, dim_x)
    pe = torch.cat([pe_y, pe_x], dim=-1).view(grid_h * grid_w, embed_dim)
    return pe


class EfficientMixerBlock(nn.Module):
    """
    Efficient MLP-Mixer block with multiple token-mixing strategies.
    
    Strategies:
    - 'global': Original global token mixing (slow for large N)
    - 'local': Windowed local token mixing (fast, O(W²) per window)
    - 'axial': Separate mixing along H and W dimensions (very fast, O(H+W))
    - 'axial_local': Windowed axial mixing (ultra-fast for very long sequences)
    
    Args:
        num_tokens: Total sequence length (or height*width for 2D)
        hidden_dim: Channel dimension
        token_mlp_dim: Hidden width for token-mixing MLP
        channel_mlp_dim: Hidden width for channel-mixing MLP
        token_mixing: Strategy for token mixing
        window_size: Window size for local token mixing (if applicable)
        grid_h, grid_w: Grid dimensions for axial mixing (if applicable)
        sn: Whether to use spectral normalization
    """
    def __init__(
        self, 
        num_tokens: int, 
        hidden_dim: int,
        token_mlp_dim: int, 
        channel_mlp_dim: int,
        token_mixing: Literal['global', 'local', 'axial', 'axial_local'] = 'axial',
        window_size: int = 8,
        grid_h: Optional[int] = None,
        grid_w: Optional[int] = None,
        sn: bool = True
    ):
        super().__init__()
        
        self.token_mixing = token_mixing
        self.window_size = window_size
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_tokens = num_tokens
        
        # Pre-normalization layers
        self.norm_tok = nn.LayerNorm(hidden_dim)
        self.norm_chn = nn.LayerNorm(hidden_dim)
        self.alpha1 = nn.Parameter(torch.Tensor([1.0]))
        self.beta1 = nn.Parameter(torch.Tensor([1.0]))
        self.alpha2 = nn.Parameter(torch.Tensor([1.0]))
        self.beta2 = nn.Parameter(torch.Tensor([1.0]))
        
        # Token-mixing MLP based on strategy
        if token_mixing == 'global':
            # Original: single MLP across all tokens
            self.token_mlp = self._build_mlp(num_tokens, token_mlp_dim, num_tokens, sn)
            
        elif token_mixing == 'local':
            # Windowed: MLP per window
            self.token_mlp = self._build_mlp(window_size, token_mlp_dim, window_size, sn)
            
        elif token_mixing == 'axial':
            # Separate MLPs for height and width dimensions
            assert grid_h is not None and grid_w is not None, \
                "grid_h and grid_w required for axial mixing"
            self.token_mlp_h = self._build_mlp(grid_h, token_mlp_dim // 2, grid_h, sn)
            self.token_mlp_w = self._build_mlp(grid_w, token_mlp_dim // 2, grid_w, sn)
            
        elif token_mixing == 'axial_local':
            # Windowed axial: local windows along each axis
            self.token_mlp_h = self._build_mlp(window_size, token_mlp_dim // 2, window_size, sn)
            self.token_mlp_w = self._build_mlp(window_size, token_mlp_dim // 2, window_size, sn)
        
        # Channel-mixing MLP (same for all strategies)
        self.channel_mlp = self._build_mlp(hidden_dim, channel_mlp_dim, hidden_dim, sn)
    
    def _build_mlp(self, in_dim: int, hidden_dim: int, out_dim: int, sn: bool):
        """Helper to build MLP with optional spectral norm."""
        linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        linear2 = nn.Linear(hidden_dim, out_dim, bias=False)
        
        if sn:
            linear1 = spectral_norm(linear1)
            linear2 = spectral_norm(linear2)
        
        return nn.Sequential(linear1, nn.GELU(), linear2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) - batch, num_tokens, channels
        Returns:
            (B, N, C)
        """
        # Token-mixing
        identity = x
        y = self.norm_tok(x)
        
        if self.token_mixing == 'global':
            y = self._global_token_mix(y)
        elif self.token_mixing == 'local':
            y = self._local_token_mix(y)
        elif self.token_mixing == 'axial':
            y = self._axial_token_mix(y)
        elif self.token_mixing == 'axial_local':
            y = self._axial_local_token_mix(y)
        
        x = identity*self.beta2 + y*self.alpha2  # Skip connection
        
        # Channel-mixing
        y = self.norm_chn(x)
        y = self.channel_mlp(y)
        x = identity*self.beta2 + y*self.alpha2  # Skip connection
        
        return x
    
    def _global_token_mix(self, x: torch.Tensor) -> torch.Tensor:
        """Original global token mixing: O(N²)"""
        # (B, N, C) -> (B, C, N) -> MLP -> (B, C, N) -> (B, N, C)
        x = x.transpose(1, 2)
        x = self.token_mlp(x)
        x = x.transpose(1, 2)
        return x
    
    def _local_token_mix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Windowed token mixing: O(W²) per window, much faster for large N.
        Partitions sequence into non-overlapping windows and mixes within each.
        """
        B, N, C = x.shape
        W = self.window_size
        
        # Pad if necessary
        pad_n = (W - N % W) % W
        if pad_n > 0:
            x = F.pad(x, (0, 0, 0, pad_n))
        
        # Reshape to windows: (B, N, C) -> (B, num_windows, W, C)
        N_padded = x.shape[1]
        num_windows = N_padded // W
        x = x.view(B, num_windows, W, C)
        
        # Apply token mixing within each window
        # (B, num_windows, W, C) -> (B*num_windows, W, C)
        x = x.view(B * num_windows, W, C)
        x = x.transpose(1, 2)  # (B*num_windows, C, W)
        x = self.token_mlp(x)
        x = x.transpose(1, 2)  # (B*num_windows, W, C)
        
        # Reshape back
        x = x.view(B, num_windows, W, C)
        x = x.view(B, N_padded, C)
        
        # Remove padding
        if pad_n > 0:
            x = x[:, :N, :]
        
        return x
    
    def _axial_token_mix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Axial/separable token mixing: O(H+W) instead of O(H*W).
        Mixes tokens separately along height and width dimensions.
        Much faster than global mixing for 2D sequences.
        """
        B, N, C = x.shape
        H, W = self.grid_h, self.grid_w
        assert N == H * W, f"Sequence length {N} must equal grid_h * grid_w = {H * W}"
        
        # Reshape to 2D grid: (B, H*W, C) -> (B, H, W, C)
        x = x.view(B, H, W, C)
        
        # Mix along height dimension (for each column)
        # (B, H, W, C) -> (B*W, H, C) -> (B*W, C, H) -> MLP -> (B*W, C, H) -> (B*W, H, C)
        x_h = x.permute(0, 2, 1, 3).contiguous().view(B * W, H, C)
        x_h = x_h.transpose(1, 2)
        x_h = self.token_mlp_h(x_h)
        x_h = x_h.transpose(1, 2)
        x_h = x_h.view(B, W, H, C).permute(0, 2, 1, 3)
        
        # Mix along width dimension (for each row)
        # (B, H, W, C) -> (B*H, W, C) -> (B*H, C, W) -> MLP -> (B*H, C, W) -> (B*H, W, C)
        x_w = x.view(B * H, W, C)
        x_w = x_w.transpose(1, 2)
        x_w = self.token_mlp_w(x_w)
        x_w = x_w.transpose(1, 2)
        x_w = x_w.view(B, H, W, C)
        
        # Combine both directions
        x = x_h + x_w
        
        # Reshape back to sequence
        x = x.reshape(B, N, C)
        
        return x
    
    def _axial_local_token_mix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Windowed axial mixing: O(W) per window along each axis.
        Ultra-fast for very long sequences. Combines windowing with axial mixing.
        """
        B, N, C = x.shape
        H, W_grid = self.grid_h, self.grid_w
        W = self.window_size
        
        assert N == H * W_grid, f"Sequence length {N} must equal grid_h * grid_w = {H * W_grid}"
        
        # Reshape to 2D grid
        x = x.view(B, H, W_grid, C)
        
        # Windowed mixing along height
        x_h = self._window_mix_along_dim(x, dim=1, window_size=W, mlp=self.token_mlp_h)
        
        # Windowed mixing along width
        x_w = self._window_mix_along_dim(x, dim=2, window_size=W, mlp=self.token_mlp_w)
        
        # Combine
        x = x_h + x_w
        
        # Reshape back
        x = x.view(B, N, C)
        
        return x
    
    def _window_mix_along_dim(self, x: torch.Tensor, dim: int, window_size: int, mlp: nn.Module) -> torch.Tensor:
        """Helper for windowed mixing along a specific dimension."""
        B, H, W, C = x.shape
        
        if dim == 1:  # Mix along height
            length = H
            x = x.permute(0, 2, 1, 3)  # (B, W, H, C)
        else:  # Mix along width
            length = W
            x = x.permute(0, 1, 2, 3)  # (B, H, W, C)
        
        # Pad if necessary
        pad_len = (window_size - length % window_size) % window_size
        if pad_len > 0:
            if dim == 1:
                x = F.pad(x, (0, 0, 0, pad_len, 0, 0))
            else:
                x = F.pad(x, (0, 0, 0, pad_len))
        
        # Get dimensions after padding
        if dim == 1:
            B, W, H_pad, C = x.shape
            num_windows = H_pad // window_size
            x = x.view(B, W, num_windows, window_size, C)
            x = x.permute(0, 1, 2, 4, 3).contiguous()  # (B, W, num_windows, C, window_size)
            x = x.view(B * W * num_windows, C, window_size)
        else:
            B, H, W_pad, C = x.shape
            num_windows = W_pad // window_size
            x = x.view(B, H, num_windows, window_size, C)
            x = x.permute(0, 1, 2, 4, 3).contiguous()  # (B, H, num_windows, C, window_size)
            x = x.view(B * H * num_windows, C, window_size)
        
        # Apply MLP
        x = mlp(x)
        
        # Reshape back
        if dim == 1:
            x = x.view(B, W, num_windows, C, window_size)
            x = x.permute(0, 1, 2, 4, 3).contiguous()  # (B, W, num_windows, window_size, C)
            x = x.view(B, W, H_pad, C)
            if pad_len > 0:
                x = x[:, :, :H, :]
            x = x.permute(0, 2, 1, 3)  # (B, H, W, C)
        else:
            x = x.view(B, H, num_windows, C, window_size)
            x = x.permute(0, 1, 2, 4, 3).contiguous()  # (B, H, num_windows, window_size, C)
            x = x.view(B, H, W_pad, C)
            if pad_len > 0:
                x = x[:, :, :W, :]
        
        return x


class MLPDiscriminator(nn.Module):
    """
    Efficient MLP-Mixer discriminator with faster token mixing strategies.
    
    Args:
        img_ch: Number of input image channels
        hidden_dim: Hidden dimension
        num_layers: Number of Mixer blocks
        img_size: Input image size
        patch_size: Size of each patch
        token_mlp_dim: Hidden width for token-mixing MLP
        channel_mlp_dim: Hidden width for channel-mixing MLP
        token_mixing: Strategy for token mixing ('global', 'local', 'axial', 'axial_local')
        window_size: Window size for local token mixing
        use_cls_token: Whether to use [CLS] token
        use_pos_embed: Whether to use positional embeddings
        overlap: Pixel overlap between patches
    """
    def __init__(
        self, 
        img_ch: int, 
        hidden_dim: int, 
        num_layers: int,
        img_size: int, 
        patch_size: int,
        token_mlp_dim: int = None, 
        channel_mlp_dim: int = None,
        token_mixing: Literal['global', 'local', 'axial', 'axial_local'] = 'axial',
        window_size: int = 8,
        use_cls_token: bool = True, 
        use_pos_embed: bool = True,
        overlap: int = 0
    ):
        super().__init__()
        assert overlap >= 0 and overlap < patch_size
        
        self.img_ch = img_ch
        self.img_size = img_size
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.hidden_dim = hidden_dim
        self.use_cls_token = use_cls_token
        self.use_pos_embed = use_pos_embed
        self.token_mixing = token_mixing

        # Calculate number of patches
        self.num_patches_h = (img_size - patch_size) // self.stride + 1
        self.num_patches_w = (img_size - patch_size) // self.stride + 1
        self.num_patches = self.num_patches_h * self.num_patches_w

        patch_dim = img_ch * patch_size * patch_size

        # Patch embedding with spectral norm
        self.patch_embed = spectral_norm(nn.Linear(patch_dim, hidden_dim, bias=False))

        # Optional [CLS] token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            seq_tokens = self.num_patches + 1
        else:
            self.register_parameter("cls_token", None)
            seq_tokens = self.num_patches

        # Positional embeddings
        if use_pos_embed:
            pe = build_2d_sincos_pos_embed(hidden_dim, self.num_patches_h, self.num_patches_w)
            if use_cls_token:
                cls_pe = torch.zeros(1, hidden_dim)
                pe = torch.cat([cls_pe, pe], dim=0)
            self.register_buffer("pos_embed", pe.unsqueeze(0), persistent=False)
        else:
            self.register_buffer("pos_embed", None, persistent=False)

        # Default MLP dimensions
        token_mlp_dim = token_mlp_dim or max(256, seq_tokens // 4)
        channel_mlp_dim = channel_mlp_dim or (hidden_dim * 4)
        
        # Stack of Efficient Mixer blocks
        self.mixer = nn.Sequential(*[
            EfficientMixerBlock(
                seq_tokens if use_cls_token else self.num_patches,
                hidden_dim, 
                token_mlp_dim, 
                channel_mlp_dim,
                token_mixing=token_mixing if not use_cls_token else 'global',  # Use global if CLS token
                window_size=window_size,
                grid_h=self.num_patches_h if not use_cls_token else None,
                grid_w=self.num_patches_w if not use_cls_token else None,
                sn=True
            )
            for _ in range(num_layers)
        ])

        # Output head
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.head = spectral_norm(nn.Linear(hidden_dim, 1, bias=False))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size

        # Extract patches
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.stride)
        patches = patches.transpose(1, 2)
        
        # Project patches
        x_tokens = self.patch_embed(patches)

        # Add [CLS] token
        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            x_tokens = torch.cat([cls, x_tokens], dim=1)

        # Add positional embeddings
        if self.use_pos_embed and self.pos_embed is not None:
            x_tokens = x_tokens + self.pos_embed

        # Apply Mixer blocks
        x_tokens = self.mixer(x_tokens)

        # Global pooling
        if self.use_cls_token:
            feat = x_tokens[:, 0, :]
        else:
            feat = x_tokens.mean(dim=1)
        
        feat = self.norm_out(feat)
        logits = self.head(feat).squeeze(-1)
        
        return logits


class MLPMixerGenerator(nn.Module):
    """
    Efficient MLP-Mixer Generator with faster token mixing strategies.
    
    Args:
        zdim: Latent dimension
        img_ch: Number of image channels
        hidden_dim: Hidden dimension
        num_layers: Number of Mixer blocks
        img_size: Output image size
        patch_size: Size of each patch
        token_mlp_dim: Hidden width for token-mixing MLP
        channel_mlp_dim: Hidden width for channel-mixing MLP
        token_mixing: Strategy for token mixing
        window_size: Window size for local token mixing
        overlap: Pixel overlap between patches
    """
    def __init__(
        self, 
        img_ch: int, 
        hidden_dim: int, 
        num_layers: int,
        img_size: int, 
        patch_size: int, 
        zdim: int,
        token_mlp_dim: int = None, 
        channel_mlp_dim: int = None,
        token_mixing: Literal['global', 'local', 'axial', 'axial_local'] = 'axial',
        window_size: int = 8,
        overlap: int = 0
    ):
        super().__init__()
        self.zdim = zdim
        self.img_ch = img_ch
        self.img_size = img_size
        self.patch_size = patch_size
        self.overlap = overlap
        self.hidden_dim = hidden_dim
        self.token_mixing = token_mixing
        
        # Calculate patches
        self.stride = patch_size - overlap
        self.num_patches_h = (img_size - patch_size) // self.stride + 1
        self.num_patches_w = (img_size - patch_size) // self.stride + 1
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Project latent to tokens
        self.latent_proj = nn.Linear(zdim, self.num_patches * hidden_dim, bias=False)
        
        # Positional embeddings
        pe = build_2d_sincos_pos_embed(hidden_dim, self.num_patches_h, self.num_patches_w)
        self.register_buffer("pos_embed", pe.unsqueeze(0), persistent=False)
        
        # Default MLP dimensions
        token_mlp_dim = token_mlp_dim or max(256, self.num_patches // 4)
        channel_mlp_dim = channel_mlp_dim or (hidden_dim * 4)
        
        # Stack of Efficient Mixer blocks
        self.mixer = nn.Sequential(*[
            EfficientMixerBlock(
                self.num_patches, 
                hidden_dim, 
                token_mlp_dim, 
                channel_mlp_dim,
                token_mixing=token_mixing,
                window_size=window_size,
                grid_h=self.num_patches_h,
                grid_w=self.num_patches_w,
                sn=False
            )
            for _ in range(num_layers)
        ])
        
        # Output
        self.norm_out = nn.LayerNorm(hidden_dim)
        
        patch_dim = patch_size * patch_size * img_ch
        self.patch_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, patch_dim, bias=False)
        )
        
        self.out_act = nn.Identity()
        
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, zdim) - latent codes
        Returns:
            (B, img_ch, img_size, img_size) - generated images
        """
        B = z.size(0)
        
        # Project latent to tokens
        x_tokens = self.latent_proj(z)
        x_tokens = x_tokens.view(B, self.num_patches, self.hidden_dim)
        
        # Add positional embeddings
        if self.pos_embed is not None:
            x_tokens = x_tokens + self.pos_embed
        
        # Apply Mixer blocks
        x_tokens = self.mixer(x_tokens)
        
        # Normalize
        x_tokens = self.norm_out(x_tokens)
        
        # Generate patches
        patches = self.patch_head(x_tokens)
        patches = patches.transpose(1, 2)
        
        # Reconstruct image
        img = F.fold(
            patches,
            output_size=(self.img_size, self.img_size),
            kernel_size=self.patch_size,
            stride=self.stride
        )
        
        # Handle overlapping regions
        if self.overlap > 0:
            ones = torch.ones_like(patches)
            overlap_count = F.fold(
                ones,
                output_size=(self.img_size, self.img_size),
                kernel_size=self.patch_size,
                stride=self.stride
            )
            img = img / (overlap_count + 1e-8)
        
        img = self.out_act(img)
        
        return img
# =====================
#  Original GAN (MLP-based, Goodfellow et al. 2014)
# =====================
# =====================
#  R3GAN Architecture (Config E from paper)
# =====================

# =====================
#  Enhanced R3GAN Architecture with from_rgb, attention, and adaptive normalization
# =====================

class AdaptiveGroupNorm(nn.Module):
    """Adaptive Group Normalization modulated by latent z (AdaGN from diffusion models)."""
    def __init__(self, num_groups: int, num_channels: int, latent_dim: int):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, affine=False)
        # Affine parameters predicted from latent z
        self.style = nn.Linear(latent_dim, num_channels * 2)
        
    def forward(self, x, z):
        """
        x: (B, C, H, W) feature maps
        z: (B, latent_dim) latent code
        """
        # Normalize
        normalized = self.gn(x)
        
        # Get adaptive scale and bias from z
        style = self.style(z)  # (B, C*2)
        scale, bias = style.chunk(2, dim=1)  # Each (B, C)
        
        # Reshape for broadcasting
        scale = scale.view(-1, self.gn.num_channels, 1, 1)
        bias = bias.view(-1, self.gn.num_channels, 1, 1)
        
        return scale * normalized + bias


class R3GANResBlock(nn.Module):
    """
    Enhanced R3GAN residual block with:
    - Optional attention
    - Optional adaptive normalization (for generator only)
    """
    def __init__(self, channels: int, groups: int = 16, expansion: float = 2.0,
                 attn: Optional[nn.Module] = None, use_adaptive_norm: bool = False,
                 latent_dim: int = 128):
        super().__init__()
        self.channels = channels
        self.attn = attn
        self.use_adaptive_norm = use_adaptive_norm
        expanded_channels = int(channels * expansion)
        
        # Calculate group size
        actual_groups = min(groups, expanded_channels)
        while expanded_channels % actual_groups != 0 and actual_groups > 1:
            actual_groups -= 1
        
        # Inverted bottleneck layers
        self.conv1 = nn.Conv2d(channels, expanded_channels, 1, 1, 0, bias=True)
        self.act1 = nn.SiLU()#nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(expanded_channels, expanded_channels, 3, 1, 1,
                              groups=actual_groups, bias=True)
        self.act2 = nn.SiLU()#nn.LeakyReLU(0.2, inplace=True)
        
        self.conv3 = nn.Conv2d(expanded_channels, channels, 1, 1, 0, bias=False)
        
        # Optional adaptive normalization (only for generator)
        if use_adaptive_norm:
            num_groups = best_gn_groups(channels)
            self.ada_norm = AdaptiveGroupNorm(num_groups, channels, latent_dim)
        else:
            self.ada_norm = None
        
        # Fix-up initialization
        nn.init.zeros_(self.conv3.weight)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.ones(1))
    
    def forward(self, x, z=None):
        """
        x: (B, C, H, W) input
        z: (B, latent_dim) latent code (only needed if using adaptive norm)
        """
        identity = x
        
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        
        # Add residual
        out = out*self.alpha + identity*self.beta
        
        # Apply adaptive normalization if enabled
        if self.ada_norm is not None and z is not None:
            out = self.ada_norm(out, z)
        
        # Apply attention if present
        if self.attn is not None:
            out = self.attn(out)
        
        return out


class R3GANGenerator(nn.Module):
    """
    Enhanced R3GAN Generator with:
    1. Optional attention at specified resolutions
    2. Optional adaptive normalization modulated by z
    """
    def __init__(self, zdim: int, img_ch: int, img_size: int,
                 base_channels: int = 32, max_channels: int = 512,
                 attn_type: int = 0, attn_res: Set[int] = None,
                 attn_heads_map: Optional[dict] = None,
                 use_adaptive_norm: bool = False):
        super().__init__()
        assert is_power_of_two(img_size) and img_size >= 8
        
        self.zdim = zdim
        self.img_ch = img_ch
        self.img_size = img_size
        self.use_adaptive_norm = use_adaptive_norm
        
        attn_res = attn_res or set()
        attn_heads_map = attn_heads_map or {}
        
        # Calculate resolution stages
        num_stages = int(math.log2(img_size)) - 2  # Start from 4×4
        
        # Calculate channel progression (decreases with resolution)
        channels = []
        for i in range(num_stages + 1):
            ch = min(max_channels, base_channels * (2 ** (num_stages - i)))
            channels.append(ch)
        
        # 4×4 basis layer
        self.basis = nn.Parameter(torch.randn(1, channels[0], 4, 4))
        self.z_to_bias = nn.Linear(zdim, channels[0])
        
        # Build stages
        self.stages = nn.ModuleList()
        
        current_res = 4
        for stage_idx in range(num_stages):
            in_ch = channels[stage_idx]
            out_ch = channels[stage_idx + 1]
            
            stage = nn.ModuleDict()
            
            # Transition layer: upsample + optional channel change
            if in_ch != out_ch:
                stage['transition'] = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=True)
                )
            else:
                stage['transition'] = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            
            current_res *= 2
            
            # Create attention modules if specified for this resolution
            attn1 = None
            attn2 = None
            if current_res in attn_res:
                heads = attn_heads_map.get(current_res)
                attn1 = make_attn(attn_type, out_ch, heads=heads)
                attn2 = make_attn(attn_type, out_ch, heads=heads)
            
            # Two residual blocks per stage with optional attention and adaptive norm
            stage['block1'] = R3GANResBlock(out_ch, groups=16, expansion=2.0,
                                           attn=attn1,
                                           use_adaptive_norm=use_adaptive_norm,
                                           latent_dim=zdim)
            stage['block2'] = R3GANResBlock(out_ch, groups=16, expansion=2.0,
                                           attn=attn2,
                                           use_adaptive_norm=use_adaptive_norm,
                                           latent_dim=zdim)
            
            self.stages.append(stage)
        
        # Final output layer
        self.to_rgb = nn.Conv2d(channels[-1], img_ch, 3, 1, 1, bias=True)
        
        # Apply fix-up initialization
        self._fixup_init(num_stages * 2)
    
    def _fixup_init(self, num_blocks):
        """Fix-up initialization."""
        scale = num_blocks ** (-0.25)
        
        for module in self.modules():
            if isinstance(module, R3GANResBlock):
                nn.init.normal_(module.conv1.weight, 0, 0.02 * scale)
                nn.init.normal_(module.conv2.weight, 0, 0.02 * scale)
    
    def forward(self, z):
        batch = z.size(0)
        
        # Start from learnable basis modulated by z
        x = self.basis.expand(batch, -1, -1, -1)
        z_bias = self.z_to_bias(z).view(batch, -1, 1, 1)
        x = x + z_bias
        
        # Process through stages
        for stage in self.stages:
            x = stage['transition'](x)
            # Pass z to blocks if using adaptive normalization
            if self.use_adaptive_norm:
                x = stage['block1'](x, z)
                x = stage['block2'](x, z)
            else:
                x = stage['block1'](x)
                x = stage['block2'](x)
        
        # Final RGB output
        x = self.to_rgb(x)
        
        return x


class R3GANDiscriminator(nn.Module):
    """
    Enhanced R3GAN Discriminator with:
    1. Optional from_rgb skip connections at specified resolutions
    2. Optional attention at specified resolutions
    3. Optional blur for from_rgb layers
    """
    def __init__(self, img_ch: int, img_size: int,
                 base_channels: int = 32, max_channels: int = 512,
                 attn_type: int = 0, attn_res: Set[int] = None,
                 attn_heads_map: Optional[dict] = None,
                 from_rgb_res: Optional[List[int]] = None,
                 use_blur: bool = False):
        super().__init__()
        assert is_power_of_two(img_size) and img_size >= 8
        
        self.img_size = img_size
        self.use_from_rgb = from_rgb_res is not None and len(from_rgb_res) > 0
        self.from_rgb_res = set(from_rgb_res or [])
        self.use_blur = use_blur
        
        attn_res = attn_res or set()
        attn_heads_map = attn_heads_map or {}
        
        # Calculate resolution stages
        num_stages = int(math.log2(img_size)) - 2  # Down to 4×4
        
        # Calculate channel progression (increases with depth)
        channels = []
        for i in range(num_stages + 1):
            ch = min(max_channels, base_channels * (2 ** i))
            channels.append(ch)
        channels.reverse()  # Reverse for discriminator
        
        # Initial from_rgb
        from_rgb_layers = []
        if self.use_blur:
            from_rgb_layers.append(Blur())
        from_rgb_layers.append(nn.Conv2d(img_ch, channels[0], 3, 1, 1, bias=True))
        from_rgb_layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.from_rgb = nn.Sequential(*from_rgb_layers)
        
        # Build stages
        self.stages = nn.ModuleList()
        
        current_res = img_size
        for stage_idx in range(num_stages):
            in_ch = channels[stage_idx]
            out_ch = channels[stage_idx + 1]
            
            stage = nn.ModuleDict()
            
            # Create attention modules if specified for this resolution
            attn1 = None
            attn2 = None
            if current_res in attn_res:
                heads = attn_heads_map.get(current_res)
                attn1 = make_attn(attn_type, in_ch, heads=heads)
                attn2 = make_attn(attn_type, in_ch, heads=heads)
            
            # Two residual blocks per stage
            stage['block1'] = R3GANResBlock(in_ch, groups=16, expansion=2.0,
                                           attn=attn1, use_adaptive_norm=False)
            stage['block2'] = R3GANResBlock(in_ch, groups=16, expansion=2.0,
                                           attn=attn2, use_adaptive_norm=False)
            
            # Transition layer: downsample + optional channel change
            if in_ch != out_ch:
                stage['transition'] = nn.Sequential(
                    nn.AvgPool2d(2),
                    nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=True)
                )
            else:
                stage['transition'] = nn.AvgPool2d(2)
            
            self.stages.append(stage)
            current_res //= 2
        
        # Build from_rgb layers for skip connections
        self.from_rgb_layers = nn.ModuleDict()
        if self.use_from_rgb:
            # Map resolutions to channel counts
            res_to_channels = {}
            cur_res = img_size // 2  # Skip first resolution (handled by main from_rgb)
            for i, ch in enumerate(channels):
                res_to_channels[cur_res] = ch
                if i < len(channels) - 1:
                    cur_res //= 2
            
            # Create from_rgb for each valid resolution
            for res in self.from_rgb_res:
                if res < img_size and res >= 8 and res in res_to_channels:
                    target_ch = res_to_channels[res]
                    layers = []
                    if self.use_blur:
                        layers.append(Blur())
                    layers.append(nn.Conv2d(img_ch, target_ch, 1, 1, 0, bias=True))
                    layers.append(nn.LeakyReLU(0.2, inplace=True))
                    self.from_rgb_layers[str(res)] = nn.Sequential(*layers)
        
        # Final classifier head
        final_ch = channels[-1]
        self.final_conv = nn.Conv2d(final_ch, final_ch, 4, 1, 0, groups=final_ch, bias=True)
        self.final_act = nn.LeakyReLU(0.2, inplace=True)
        self.classifier = spectral_norm(nn.Linear(final_ch, 1, bias=False))
        
        # Apply fix-up initialization
        self._fixup_init(num_stages * 2)
    
    def _fixup_init(self, num_blocks):
        """Fix-up initialization."""
        scale = num_blocks ** (-0.25)
        
        for module in self.modules():
            if isinstance(module, R3GANResBlock):
                nn.init.normal_(module.conv1.weight, 0, 0.02 * scale)
                nn.init.normal_(module.conv2.weight, 0, 0.02 * scale)
    
    def forward(self, x):
        # Keep reference to original input for from_rgb skip connections
        original_input = x
        current_res = self.img_size
        
        # Initial from_rgb
        x = self.from_rgb(x)
        
        # Process through stages with optional from_rgb skip connections
        for stage_idx, stage in enumerate(self.stages):
            # Apply from_rgb skip connection if applicable (before processing this resolution)
            if self.use_from_rgb and stage_idx > 0 and current_res in self.from_rgb_res:
                res_key = str(current_res)
                if res_key in self.from_rgb_layers:
                    # Downsample original input to current resolution
                    downsampled = F.interpolate(original_input,
                                               size=(current_res, current_res),
                                               mode='bilinear', align_corners=False)
                    # Apply from_rgb and add to main path
                    rgb_path = self.from_rgb_layers[res_key](downsampled)
                    x = x + rgb_path
            
            # Process through blocks
            x = stage['block1'](x)
            x = stage['block2'](x)
            x = stage['transition'](x)
            current_res //= 2
        
        # Final classifier
        x = self.final_conv(x)  # (B, C, 1, 1)
        x = self.final_act(x)
        x = x.view(x.size(0), -1)  # (B, C)
        x = self.classifier(x)  # (B, 1)
        
        return x.view(-1)

# =====================
#  Basic MLP GAN
# =====================

class MLPGANGenerator(nn.Module):
    """
    Simple MLP Generator.
    Pure feedforward network with no convolutions.
    
    Args:
        zdim: Latent dimension
        img_ch: Output image channels
        img_size: Output image size
        hidden_dims: List of hidden layer dimensions
        activation: Activation function to use
    """
    def __init__(self, zdim: int, img_ch: int, img_size: int,
                 hidden_dims: List[int] = None, activation: str = 'relu'):
        super().__init__()
        self.zdim = zdim
        self.img_ch = img_ch
        self.img_size = img_size
        self.output_dim = img_ch * img_size * img_size
        
        if hidden_dims is None:
            # Default: 3 hidden layers with decreasing size
            hidden_dims = [256, 512, 1024]
        
        # Choose activation
        if activation == 'relu':
            act = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            act = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            act = nn.Tanh()
        elif activation == 'gelu':
            act = nn.GELU()
        else:
            act = nn.ReLU(inplace=True)
        
        # Build network
        layers = []
        in_dim = zdim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                act,
                nn.Dropout(0.2)  # Dropout for regularization
            ])
            in_dim = h_dim
        
        # Final layer to output
        layers.append(nn.Linear(in_dim, self.output_dim))
        layers.append(nn.Tanh())  # Normalize to [-1, 1]
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, z):
        batch = z.size(0)
        x = self.net(z)
        x = x.view(batch, self.img_ch, self.img_size, self.img_size)
        return x


class MLPGANDiscriminator(nn.Module):
    """
    Simple MLP Discriminator.
    Pure feedforward network with no convolutions.
    
    Args:
        img_ch: Input image channels
        img_size: Input image size
        hidden_dims: List of hidden layer dimensions
        activation: Activation function to use
        use_spectral_norm: Whether to use spectral normalization
    """
    def __init__(self, img_ch: int, img_size: int,
                 hidden_dims: List[int] = None, activation: str = 'leakyrelu',
                 use_spectral_norm: bool = False):
        super().__init__()
        self.img_ch = img_ch
        self.img_size = img_size
        self.input_dim = img_ch * img_size * img_size
        
        if hidden_dims is None:
            # Default: 3 hidden layers with decreasing size
            hidden_dims = [1024, 512, 256]
        
        # Choose activation
        if activation == 'relu':
            act = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            act = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            act = nn.Tanh()
        elif activation == 'gelu':
            act = nn.GELU()
        else:
            act = nn.LeakyReLU(0.2, inplace=True)
        
        # Build network
        layers = []
        in_dim = self.input_dim
        
        for h_dim in hidden_dims:
            linear = nn.Linear(in_dim, h_dim)
            if use_spectral_norm:
                linear = spectral_norm(linear)
            
            layers.extend([
                linear,
                act,
                nn.Dropout(0.3)  # More aggressive dropout for discriminator
            ])
            in_dim = h_dim
        
        # Final classification layer
        final_linear = nn.Linear(in_dim, 1)
        if use_spectral_norm:
            final_linear = spectral_norm(final_linear)
        layers.append(final_linear)
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Get actual weight (might be wrapped by spectral_norm)
            w = getattr(m, 'weight_orig', None)
            if w is None:
                w = m.weight
            nn.init.normal_(w, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        batch = x.size(0)
        x = x.view(batch, -1)  # Flatten
        x = self.net(x)
        return x.view(-1)
# =====================
#  Deconv Models (Original)
# =====================
class SelfAttention2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.q = nn.Conv1d(channels, channels, 1)
        self.k = nn.Conv1d(channels, channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.scale = channels ** -0.5
        self.norm_in = nn.GroupNorm(1, channels)
        self.norm_out = nn.GroupNorm(1, channels)

    def forward(self, x):
        n, c, h, w = x.shape
        residual = x
        x = self.norm_in(x)
        flat = x.view(n, c, h * w)
        q = self.q(flat)
        k = self.k(flat)
        v = self.v(flat)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2))
        out = self.proj(out)
        out = out.view(n, c, h, w)
        return self.norm_out(out + residual)


class MultiHeadSelfAttention2D(nn.Module):
    def __init__(self, channels: int, heads: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.heads = heads if heads is not None else max(1, min(8, channels // 64))
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=self.heads, batch_first=False)
        self.norm_in = nn.GroupNorm(1, channels)
        self.norm_seq = nn.GroupNorm(best_gn_groups(channels), channels)
        self.norm_out = nn.GroupNorm(1, channels)

    def forward(self, x):
        n, c, h, w = x.shape
        residual = x
        x = self.norm_in(x)
        seq = x.view(n, c, h * w).permute(2, 0, 1)
        seq = self.norm_seq(seq.permute(1, 2, 0)).permute(2, 0, 1)
        out, _ = self.mha(seq, seq, seq, need_weights=False)
        out = out.permute(1, 2, 0).contiguous().view(n, c, h, w)
        return self.norm_out(out + residual)


ACT_NAMES = {
    0: 'linear', 1: 'sigmoid', 2: 'tanh', 3: 'relu', 4: 'softsign', 
    5: 'silu', 6: 'prelu', 7: 'gelu', 8: 'sine', 9: "sinelrelu", 10: "smu"
}


def make_base_act(act_id: int, channels: int, is_discriminator: bool) -> nn.Module:
    if act_id == 0:
        return nn.Identity()
    if act_id == 1:
        return nn.Sigmoid()
    if act_id == 2:
        return nn.Tanh()
    if act_id == 3:
        return nn.LeakyReLU(0.2, inplace=True) if is_discriminator else nn.ReLU(inplace=True)
    if act_id == 4:
        return nn.Softsign()
    if act_id == 5:
        return nn.SiLU(inplace=True)
    if act_id == 6:
        return nn.PReLU(num_parameters=channels)
    if act_id == 7:
        return nn.GELU()
    if act_id == 8:
        return Sine()
    if act_id == 9:
        return SineEvenLReLU(0.2)
    if act_id == 10:
        return SMU()
    raise ValueError('Invalid activation id')


class SelfGated(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
    def forward(self, x):
        y = self.base(x)
        return y * torch.sigmoid(x)


class ETActWrap(nn.Module):
    def __init__(self, channels: int, base: nn.Module):
        super().__init__()
        self.act = ETAct(channels, base)
    def forward(self, x):
        return self.act(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act_id: int, act_type: int, residual: bool, attn: Optional[nn.Module], add_noise: bool):
        super().__init__()
        self.add_noise = add_noise
        self.residual = residual
        self.act_id = act_id
        self.act_type = act_type
        self.noise_mul = nn.Parameter(torch.zeros(1))

        if act_type == 2:
            conv_out = out_ch * 2
            self.conv = nn.ConvTranspose2d(in_ch, conv_out, 4, 2, 1)
            self.gn = nn.GroupNorm(best_gn_groups(conv_out), conv_out)
            self.base_act = make_base_act(act_id, out_ch, is_discriminator=False)
        else:
            self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
            self.gn = nn.GroupNorm(best_gn_groups(out_ch), out_ch)
            base = make_base_act(act_id, out_ch, is_discriminator=False)
            if act_type == 1:
                self.act = SelfGated(base)
            elif act_type == 3:
                self.act = ETActWrap(out_ch, base)
            else:
                self.act = base

        self.attn = attn

        if residual:
            self.skip = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_ch, out_ch, 1, 1, 0)
            )
        else:
            self.skip = None

    def forward(self, x):
        residual = x
        if self.add_noise:
            x = x + (torch.randn_like(x) * F.softplus(self.noise_mul))
        x = self.conv(x)
        if self.act_type == 2:
            x = self.gn(x)
            a, b = x.chunk(2, dim=1)
            x = self.base_act(a) * torch.sigmoid(b)
        else:
            x = self.gn(x)
            x = self.act(x)
        if self.attn is not None:
            x = self.attn(x)
        if self.residual:
            x = x*0.5 + self.skip(residual)*0.5
        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act_id: int, act_type: int, residual: bool, attn: Optional[nn.Module], add_noise: bool, is_first: bool):
        super().__init__()
        self.add_noise = add_noise
        self.residual = residual
        self.is_first = is_first
        self.act_id = act_id
        self.act_type = act_type

        if act_type == 2:
            conv_out = out_ch * 2
            self.conv = nn.Conv2d(in_ch, conv_out, 4, 2, 1)
            self.gn = nn.GroupNorm(best_gn_groups(conv_out), conv_out)
            self.base_act = make_base_act(act_id, conv_out // 2, is_discriminator=True)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 4, 2, 1)
            self.gn = nn.GroupNorm(best_gn_groups(out_ch), out_ch)
            base = make_base_act(act_id, out_ch, is_discriminator=True)
            if act_type == 1:
                self.act = SelfGated(base)
            elif act_type == 3:
                self.act = ETActWrap(out_ch, base)
            else:
                self.act = base

        self.attn = attn

        if residual:
            self.skip = nn.Sequential(
                nn.AvgPool2d(2),
                nn.Conv2d(in_ch, out_ch, 1, 1, 0)
            )
        else:
            self.skip = None

    def forward(self, x):
        residual = x
        if self.add_noise and not self.is_first:
            x = x + torch.randn_like(x)
        x = self.conv(x)
        if self.act_type == 2:
            x = self.gn(x)
            a, b = x.chunk(2, dim=1)
            x = self.base_act(a) * torch.sigmoid(b)
        else:
            x = self.gn(x)
            x = self.act(x)
        if self.attn is not None:
            x = self.attn(x)
        if self.residual:
            x = x + self.skip(residual)
        return x


class DeconvGenerator(nn.Module):
    def __init__(self, zdim: int, img_ch: int, g_last_hidden: int, fmap_max: int, img_size: int,
                 act_id: int, act_type: int, residual: bool,
                 attn_type: int, attn_res: Set[int], add_noise: bool, attn_heads_map: Optional[dict] = None):
        super().__init__()
        assert is_power_of_two(img_size) and img_size >= 8
        self.zdim = zdim
        self.img_size = img_size
        self.add_noise = add_noise
        self.attn_heads_map = attn_heads_map or {}
        self.add_noise = add_noise

        s0 = 8
        ups = int(math.log2(img_size)) - 3
        start_ch = min(fmap_max, g_last_hidden * (2 ** ups))
        chs = [start_ch]
        for _ in range(ups):
            chs.append(max(g_last_hidden, chs[-1] // 2))
        chs[-1] = g_last_hidden

        self.fc = nn.Linear(zdim, chs[0] * s0 * s0)
        self.blocks = nn.ModuleList()

        cur_res = s0
        in_ch = chs[0]
        for i in range(len(chs) - 1):
            out_ch = chs[i + 1]
            cur_res *= 2
            heads = self.attn_heads_map.get(cur_res) if (cur_res in attn_res) else None
            attn = make_attn(attn_type, out_ch, heads=heads) if cur_res in attn_res else None
            self.blocks.append(UpBlock(in_ch, out_ch, act_id, act_type, residual, attn, add_noise))
            in_ch = out_ch

        self.to_img = nn.Conv2d(in_ch, img_ch, 3, 1, 1)
        self.out_act = ETActWrap(img_ch, nn.Tanh())

    def forward(self, z):
        z = z + torch.randn_like(z) if self.add_noise == True else z
        x = self.fc(z).view(z.size(0), -1, 8, 8)
        for blk in self.blocks:
            x = blk(x)
        x = self.to_img(x)
        x = self.out_act(x)
        return x


class DeconvDiscriminator(nn.Module):
    def __init__(self, img_ch: int, d_first_hidden: int, fmap_max: int, img_size: int,
                 act_id: int, act_type: int, residual: bool,
                 attn_type: int, attn_res: Set[int], add_noise: bool, 
                 attn_heads_map: Optional[dict] = None,
                 from_rgb_res: Optional[List[int]] = None, use_blur: bool = False):
        super().__init__()
        assert is_power_of_two(img_size) and img_size >= 8

        s0 = 8
        downs = int(math.log2(img_size)) - 3
        self.attn_heads_map = attn_heads_map or {}
        self.img_size = img_size
        self.use_from_rgb = from_rgb_res is not None and len(from_rgb_res) > 0
        self.from_rgb_res = set(from_rgb_res or [])
        self.use_blur = use_blur
        
        # Build channel progression
        chs = [d_first_hidden]
        for _ in range(downs - 1):
            chs.append(min(fmap_max, chs[-1] * 2))
        last_ch = min(fmap_max, chs[-1] * 2) if downs > 0 else d_first_hidden

        # Main path blocks
        blocks = []
        in_ch = img_ch
        cur_res = img_size
        
        # If using from_rgb, first block takes d_first_hidden channels
        # (from_rgb will convert img_ch -> d_first_hidden)
        if self.use_from_rgb and img_size in self.from_rgb_res:
            in_ch = d_first_hidden
        
        for i, out_ch in enumerate(chs):
            next_res = cur_res // 2
            if next_res in attn_res:
                heads = self.attn_heads_map.get(next_res)
                attn = make_attn(attn_type, out_ch, heads=heads)
            else:
                attn = None
            blocks.append(DownBlock(in_ch, out_ch, act_id, act_type, residual, attn, 
                                   add_noise, is_first=(i == 0)))
            in_ch = out_ch
            cur_res //= 2

        self.blocks = nn.ModuleList(blocks)
        
        # Build from_rgb layers for each specified resolution
        self.from_rgb_layers = nn.ModuleDict()
        if self.use_from_rgb:
            # Get channel count at each resolution
            res_to_channels = {}
            cur_res = img_size
            cur_ch = d_first_hidden
            res_to_channels[cur_res] = cur_ch
            
            for ch in chs:
                cur_res //= 2
                res_to_channels[cur_res] = ch
            
            # Create from_rgb for each valid resolution (skip image_size itself)
            for res in self.from_rgb_res:
                if res < img_size and res >= 8 and res in res_to_channels:
                    target_ch = res_to_channels[res]
                    layers = []
                    if self.use_blur:
                        layers.append(Blur())
                    layers.append(nn.Conv2d(img_ch, target_ch, 1, 1, 0))
                    self.from_rgb_layers[str(res)] = nn.Sequential(*layers)

        self.final_conv = nn.Conv2d(in_ch, last_ch, 3, 1, 1)
        self.final_act = make_base_act(act_id, last_ch, is_discriminator=True) if act_type == 0 else (
            SelfGated(make_base_act(act_id, last_ch, True)) if act_type == 1 else (
                ETActWrap(last_ch, make_base_act(act_id, last_ch, True)) if act_type == 3 else make_base_act(act_id, last_ch, True)
            )
        )
        self.final_gn = nn.GroupNorm(best_gn_groups(last_ch), last_ch)
        self.out = spectral_norm(nn.Linear(last_ch, 1))

    def forward(self, x):
        # Keep reference to original input for from_rgb skip connections
        original_input = x
        cur_res = self.img_size
        
        # Process through blocks with from_rgb skip connections
        for i, blk in enumerate(self.blocks):
            # Check if we should add from_rgb at this resolution
            # Apply BEFORE processing the block at this resolution
            if self.use_from_rgb and i > 0 and cur_res in self.from_rgb_res:
                res_key = str(cur_res)
                if res_key in self.from_rgb_layers:
                    # Downsample original input image to current resolution
                    downsampled = F.interpolate(original_input, 
                                            size=(cur_res, cur_res), 
                                            mode='area')
                    # Apply from_rgb to get features at this resolution
                    rgb_path = self.from_rgb_layers[res_key](downsampled)
                    # Blend with main path (StyleGAN2 uses addition)
                    x = x + rgb_path
            
            x = blk(x)
            cur_res //= 2
        
        # Final layers
        x = self.final_conv(x)
        x = self.final_gn(x)
        x = self.final_act(x)
        x = x.mean(dim=[2, 3])
        x = self.out(x)
        return x.view(-1)

# Add this section after the existing imports and before the model definitions

# =====================
#  LadaGAN Components
# =====================

class LinearAdditiveAttention(nn.Module):
    """Linear additive attention mechanism from LadaGAN."""
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        
        # Learnable weight vector for computing attention weights
        self.w = nn.Parameter(torch.randn(num_heads, self.head_dim))
        
    def forward(self, x):
        """
        x: (B, N, C)
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention weights: α_i = exp(w^T q_i / √d) / Σ exp(w^T q_j / √d)
        # q: (B, H, N, D), w: (H, D)
        attn_logits = torch.einsum('bhnd,hd->bhn', q, self.w) * self.scale  # (B, H, N)
        alpha = F.softmax(attn_logits, dim=-1)  # (B, H, N)
        
        # Compute global vector: g = Σ α_i q_i
        g = torch.einsum('bhn,bhnd->bhd', alpha, q)  # (B, H, D)
        
        # Element-wise with keys: p_i = g ⊙ k_i
        g_expanded = g.unsqueeze(2)  # (B, H, 1, D)
        p = g_expanded * k  # (B, H, N, D)
        
        # Element-wise with values: r_i = p_i ⊙ v_i
        out = p * v  # (B, H, N, D)
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        return out


class SelfModulatedLayerNorm(nn.Module):
    """Self-Modulated LayerNorm from ViTGAN paper."""
    def __init__(self, normalized_shape: int, latent_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.gamma_mlp = nn.Linear(latent_dim, normalized_shape)
        self.beta_mlp = nn.Linear(latent_dim, normalized_shape)
        
    def forward(self, x, z):
        """
        x: (B, N, D) or (B, D)
        z: (B, latent_dim)
        """
        normalized = self.norm(x)
        gamma = self.gamma_mlp(z)
        beta = self.beta_mlp(z)
        
        if x.dim() == 3:  # (B, N, D)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            
        return gamma * normalized + beta


class LadaformerGenerator(nn.Module):
    """Ladaformer block for generator with SLN and no MLP residual."""
    def __init__(self, dim: int, num_heads: int, mlp_dim: int, latent_dim: int):
        super().__init__()
        self.sln1 = SelfModulatedLayerNorm(dim, latent_dim)
        self.attn = LinearAdditiveAttention(dim, num_heads)
        self.sln2 = SelfModulatedLayerNorm(dim, latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        
    def forward(self, x, z):
        """
        x: (B, N, D)
        z: (B, latent_dim)
        """
        # Attention with residual, no MLP residual as per paper
        h_prime = self.sln1(x, z)
        h_prime = self.attn(h_prime) + x
        
        # MLP without residual (key difference from discriminator)
        h = self.sln2(h_prime, z)
        h = self.mlp(h)
        
        return h


class LadaformerDiscriminator(nn.Module):
    """Ladaformer block for discriminator with standard LN and MLP residual."""
    def __init__(self, dim: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = LinearAdditiveAttention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        
    def forward(self, x):
        """x: (B, N, D)"""
        # Attention with residual
        h_prime = self.ln1(x)
        h_prime = self.attn(h_prime) + x
        
        # MLP with residual (key difference from generator)
        h = self.ln2(h_prime)
        h = self.mlp(h) + h_prime
        
        return h


class LadaGenerator(nn.Module):
    """LadaGAN Generator based on linear additive attention."""
    def __init__(self, zdim: int, img_ch: int, img_size: int, num_heads: int = 4, mlp_dim: int = 512):
        super().__init__()
        self.zdim = zdim
        self.img_ch = img_ch
        self.img_size = img_size
        
        # Embedding dimensions for each stage
        self.emb_8 = 1024
        self.emb_16 = 256
        self.emb_32 = 64
        
        # Initial projection: z -> 8x8 feature map
        self.fc = nn.Linear(zdim, self.emb_8 * 8 * 8)
        
        # Positional embeddings
        self.pos_embed_8 = nn.Parameter(torch.randn(1, 64, self.emb_8) * 0.02)
        self.pos_embed_16 = nn.Parameter(torch.randn(1, 256, self.emb_16) * 0.02)
        self.pos_embed_32 = nn.Parameter(torch.randn(1, 1024, self.emb_32) * 0.02)
        
        # Ladaformer blocks
        self.block_8 = LadaformerGenerator(self.emb_8, num_heads, mlp_dim, zdim)
        self.block_16 = LadaformerGenerator(self.emb_16, num_heads, mlp_dim, zdim)
        self.block_32 = LadaformerGenerator(self.emb_32, num_heads, mlp_dim, zdim)
        
        # Local Embedding Expansion (LEE): PixelShuffle + Conv
        self.lee_8_to_16 = nn.Sequential(
            nn.Conv2d(self.emb_8, self.emb_16 * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(self.emb_16, self.emb_16, 3, 1, 1)
        )
        self.lee_16_to_32 = nn.Sequential(
            nn.Conv2d(self.emb_16, self.emb_32 * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(self.emb_32, self.emb_32, 3, 1, 1)
        )
        
        # Final output layers
        if img_size == 32:
            # Pixel-level generation for 32x32
            self.final_mlp = nn.Linear(self.emb_32, mlp_dim)
            self.final_sln = SelfModulatedLayerNorm(mlp_dim, zdim)
            self.to_rgb = nn.Conv2d(mlp_dim, img_ch, 3, 1, 1)
        else:
            # Patch generation or additional upsampling for higher resolutions
            up_blocks = []
            in_ch = self.emb_32
            current_res = 32
            while current_res < img_size:
                out_ch = max(32, in_ch // 2)
                up_blocks.append(nn.Conv2d(in_ch, out_ch * 4, 3, 1, 1))
                up_blocks.append(nn.PixelShuffle(2))
                up_blocks.append(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
                up_blocks.append(nn.GELU())
                in_ch = out_ch
                current_res *= 2
            self.upsampler = nn.Sequential(*up_blocks) if up_blocks else nn.Identity()
            self.to_rgb = nn.Conv2d(in_ch, img_ch, 3, 1, 1)
        
        self.out_act = nn.Identity()
        
    def forward(self, z):
        B = z.size(0)
        
        # Initial 8x8 feature map
        h = self.fc(z).view(B, self.emb_8, 8, 8)
        h = h.flatten(2).transpose(1, 2)  # (B, 64, emb_8)
        h = h + self.pos_embed_8
        h = self.block_8(h, z)
        h = h.transpose(1, 2).view(B, self.emb_8, 8, 8)
        
        # 8x8 -> 16x16
        h = self.lee_8_to_16(h)  # (B, emb_16, 16, 16)
        h = h.flatten(2).transpose(1, 2)  # (B, 256, emb_16)
        h = h + self.pos_embed_16
        h = self.block_16(h, z)
        h = h.transpose(1, 2).view(B, self.emb_16, 16, 16)
        
        # 16x16 -> 32x32
        h = self.lee_16_to_32(h)  # (B, emb_32, 32, 32)
        h = h.flatten(2).transpose(1, 2)  # (B, 1024, emb_32)
        h = h + self.pos_embed_32
        h = self.block_32(h, z)
        h = h.transpose(1, 2).view(B, self.emb_32, 32, 32)
        
        # Generate final image
        if self.img_size == 32:
            # For 32x32, use MLP + SLN
            h_seq = h.flatten(2).transpose(1, 2)  # (B, 1024, emb_32)
            h_seq = self.final_mlp(h_seq)
            h_seq = self.final_sln(h_seq, z)
            h = h_seq.transpose(1, 2).view(B, -1, 32, 32)
        else:
            # For higher resolutions, use upsampler
            h = self.upsampler(h)
        
        img = self.to_rgb(h)
        img = self.out_act(img)
        
        return img


class LadaDiscriminator(nn.Module):
    """LadaGAN Discriminator combining FastGAN-style conv blocks with Ladaformer."""
    def __init__(self, img_ch: int, img_size: int, d_hidden: int = 64, num_heads: int = 4, mlp_dim: int = 512):
        super().__init__()
        self.img_size = img_size
        
        # Initial convolutional feature extractor (FastGAN-style)
        self.from_rgb = nn.Sequential(
            nn.Conv2d(img_ch, d_hidden, 3, 1, 1),
            nn.SiLU()
        )
        
        # Residual blocks with downsampling
        self.res_blocks = nn.ModuleList()
        in_ch = d_hidden
        current_res = img_size
        
        while current_res > 32:
            out_ch = min(512, in_ch * 2)
            self.res_blocks.append(nn.ModuleDict({
                'conv1': nn.Conv2d(in_ch, out_ch, 3, 2, 1),
                'bn1': nn.BatchNorm2d(out_ch),
                'act1': nn.SiLU(),
                'conv2': nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                'bn2': nn.BatchNorm2d(out_ch),
                'act2': nn.SiLU(),
                'skip': nn.Sequential(
                    nn.AvgPool2d(2),
                    nn.Conv2d(in_ch, out_ch, 1, 1, 0)
                )
            }))
            in_ch = out_ch
            current_res //= 2
        
        # Ladaformer at 32x32 resolution
        self.lada_dim = min(256, in_ch)
        if in_ch != self.lada_dim:
            self.to_lada = nn.Conv2d(in_ch, self.lada_dim, 1, 1, 0)
        else:
            self.to_lada = nn.Identity()
            
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, self.lada_dim) * 0.02)
        self.ladaformer = LadaformerDiscriminator(self.lada_dim, num_heads, mlp_dim)
        
        # Fix: Changed to properly downsample to 1x1
        # 32x32 -> 16x16 -> 8x8 -> 4x4 -> 1x1
        self.final_convs = nn.Sequential(
            nn.Conv2d(self.lada_dim, self.lada_dim * 2, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(self.lada_dim * 2, self.lada_dim * 2, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(self.lada_dim * 2, self.lada_dim * 2, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(self.lada_dim * 2, 512, 4, 1, 0),  # 4x4 -> 1x1
            nn.SiLU()
        )
        
        self.out = spectral_norm(nn.Linear(512, 1))
        
    def forward(self, x):
        B = x.size(0)
        
        # Initial conv
        h = self.from_rgb(x)
        
        # Residual blocks with downsampling
        for block in self.res_blocks:
            identity = h
            h = block['conv1'](h)
            h = block['bn1'](h)
            h = block['act1'](h)
            h = block['conv2'](h)
            h = block['bn2'](h)
            h = block['act2'](h)
            h = h + block['skip'](identity)
        
        # Ladaformer processing at 32x32
        h = self.to_lada(h)  # (B, lada_dim, 32, 32)
        h = h.flatten(2).transpose(1, 2)  # (B, 1024, lada_dim)
        h = h + self.pos_embed
        h = self.ladaformer(h)
        h = h.transpose(1, 2).view(B, self.lada_dim, 32, 32)
        
        # Final downsampling and classification
        h = self.final_convs(h)  # (B, 512, 1, 1)
        h = h.reshape(B, -1)
        logits = self.out(h).squeeze(-1)
        
        return logits

# =====================
#  2D Positional Embedding for MLPs
# =====================
# =====================
#  2D Positional Embedding for MLPs - Enhanced
# =====================
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import math


class PositionalEmbedding2D(nn.Module):
    """
    Enhanced 2D positional embeddings for patch-based models.
    Supports multiple embedding types: sinusoidal_2d, static_2d, trainable_2d, and none.
    """
    def __init__(self, num_patches_h: int, num_patches_w: int, hidden_dim: int, 
                 embed_type: str = 'trainable_2d'):
        """
        Args:
            num_patches_h: Number of patches in height
            num_patches_w: Number of patches in width
            hidden_dim: Dimensionality of hidden features
            embed_type: Type of positional embedding
                - 'sinusoidal_2d': 2D sinusoidal positional embeddings (static, not learned)
                - 'static_2d': Static 2D Gaussian random embeddings (frozen)
                - 'trainable_2d': Learnable 2D positional embeddings
                - 'none': No positional embeddings (identity function)
        """
        super().__init__()
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.hidden_dim = hidden_dim
        self.embed_type = embed_type
        
        if embed_type == 'sinusoidal_2d':
            # 2D sinusoidal positional embeddings (not learned)
            pos_embed = self._get_sinusoidal_embeddings_2d(num_patches_h, num_patches_w, hidden_dim)
            self.register_buffer('pos_embed', pos_embed)
            
        elif embed_type == 'static_2d':
            # Static 2D embeddings initialized with Gaussian distribution (frozen)
            pos_embed = torch.randn(1, num_patches_h * num_patches_w, hidden_dim) * 0.02
            self.register_buffer('pos_embed', pos_embed)
            
        elif embed_type == 'trainable_2d':
            # Learnable position embeddings (original behavior)
            self.pos_embed = nn.Parameter(
                torch.randn(1, num_patches_h * num_patches_w, hidden_dim) * 0.02
            )
            
        elif embed_type == 'none':
            # No positional embeddings
            self.pos_embed = None
            
        else:
            raise ValueError(f"Unknown embed_type: {embed_type}. Choose from "
                           f"['sinusoidal_2d', 'static_2d', 'trainable_2d', 'none']")
    
    def _get_sinusoidal_embeddings_2d(self, h: int, w: int, d: int):
        """
        Generate 2D sinusoidal positional embeddings.
        
        Args:
            h: Height (number of patches)
            w: Width (number of patches)
            d: Dimensionality
            
        Returns:
            pos_embed: (1, h*w, d)
        """
        assert d % 4 == 0, f"hidden_dim must be divisible by 4 for sinusoidal_2d, got {d}"
        
        # Split dimensions: half for height, half for width
        d_h = d // 2
        d_w = d // 2
        
        # Generate position indices
        pos_h = torch.arange(h, dtype=torch.float32)
        pos_w = torch.arange(w, dtype=torch.float32)
        
        # Create frequency scales
        div_term_h = torch.exp(torch.arange(0, d_h, 2, dtype=torch.float32) * 
                               -(math.log(10000.0) / d_h))
        div_term_w = torch.exp(torch.arange(0, d_w, 2, dtype=torch.float32) * 
                               -(math.log(10000.0) / d_w))
        
        # Compute sinusoidal embeddings for height dimension
        pos_embed_h = torch.zeros(h, d_h)
        pos_embed_h[:, 0::2] = torch.sin(pos_h.unsqueeze(1) * div_term_h)
        pos_embed_h[:, 1::2] = torch.cos(pos_h.unsqueeze(1) * div_term_h)
        
        # Compute sinusoidal embeddings for width dimension
        pos_embed_w = torch.zeros(w, d_w)
        pos_embed_w[:, 0::2] = torch.sin(pos_w.unsqueeze(1) * div_term_w)
        pos_embed_w[:, 1::2] = torch.cos(pos_w.unsqueeze(1) * div_term_w)
        
        # Combine height and width embeddings
        # Create meshgrid for 2D positions
        pos_embed_2d = torch.zeros(h, w, d)
        for i in range(h):
            for j in range(w):
                pos_embed_2d[i, j, :d_h] = pos_embed_h[i]
                pos_embed_2d[i, j, d_h:] = pos_embed_w[j]
        
        # Reshape to (1, h*w, d)
        pos_embed_2d = pos_embed_2d.view(1, h * w, d)
        
        return pos_embed_2d
        
    def forward(self, x):
        """
        Args:
            x: (B, N, C) where N = num_patches_h * num_patches_w
        Returns:
            x + pos_embed or x (depending on embed_type)
        """
        if self.embed_type == 'none':
            return x
        return x + self.pos_embed


# =====================
#  Projection Modules
# =====================
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class MultiheadLinearProj(nn.Module):
    """Multiheaded linear projection - splits features into heads and applies separate projections."""
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, sn: bool = False):
        super().__init__()
        assert in_dim % num_heads == 0, f"in_dim ({in_dim}) must be divisible by num_heads ({num_heads})"
        assert out_dim % num_heads == 0, f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})"
        
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.head_in_dim = in_dim // num_heads
        self.head_out_dim = out_dim // num_heads
        
        # Create separate linear layers for each head
        self.head_projs = nn.ModuleList()
        for _ in range(num_heads):
            proj = nn.Linear(self.head_in_dim, self.head_out_dim)
            if sn:
                proj = spectral_norm(proj)
            self.head_projs.append(proj)
    
    def forward(self, x):
        """
        Args:
            x: (B, ..., in_dim) or (B, N, in_dim)
        Returns:
            out: (B, ..., out_dim) or (B, N, out_dim)
        """
        # Split into heads along feature dimension
        *batch_dims, feat_dim = x.shape
        x = x.view(*batch_dims, self.num_heads, self.head_in_dim)
        
        # Apply each head's projection
        head_outputs = []
        for i in range(self.num_heads):
            head_out = self.head_projs[i](x[..., i, :])
            head_outputs.append(head_out)
        
        # Concatenate heads back together
        out = torch.cat(head_outputs, dim=-1)
        return out


class MultiheadConvProj(nn.Module):
    """Multiheaded convolutional projection - splits channels into heads and applies separate convolutions."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, 
                 num_heads: int = 4, padding: int = 0, sn: bool = False):
        super().__init__()
        assert in_ch % num_heads == 0, f"in_ch ({in_ch}) must be divisible by num_heads ({num_heads})"
        assert out_ch % num_heads == 0, f"out_ch ({out_ch}) must be divisible by num_heads ({num_heads})"
        
        self.num_heads = num_heads
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.head_in_ch = in_ch // num_heads
        self.head_out_ch = out_ch // num_heads
        
        # Create separate conv layers for each head
        self.head_convs = nn.ModuleList()
        for _ in range(num_heads):
            conv = nn.Conv2d(self.head_in_ch, self.head_out_ch, 
                           kernel_size=kernel_size, stride=stride, 
                           padding=padding, bias=False)
            if sn:
                conv = spectral_norm(conv)
            self.head_convs.append(conv)
    
    def forward(self, x):
        """
        Args:
            x: (B, in_ch, H, W)
        Returns:
            out: (B, out_ch, H', W')
        """
        B, C, H, W = x.shape
        # Split channels into heads
        x = x.view(B, self.num_heads, self.head_in_ch, H, W)
        
        # Apply each head's convolution
        head_outputs = []
        for i in range(self.num_heads):
            head_out = self.head_convs[i](x[:, i])
            head_outputs.append(head_out)
        
        # Concatenate heads back together along channel dimension
        out = torch.cat(head_outputs, dim=1)
        return out


class HyperMixerLinearProj(nn.Module):
    """
    HyperMixer-based linear projection with LoRA decomposition.
    A baseline weight is learned, and a hypernetwork produces low-rank weight updates.
    
    Memory savings: Instead of generating (out_dim * in_dim) parameters,
    generates (out_dim * rank + rank * in_dim) where rank << min(out_dim, in_dim)
    """
    def __init__(self, in_dim: int, out_dim: int, cond_dim: int, 
                 lora_rank: int = 32, hidden_ratio: int = 2, sn: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cond_dim = cond_dim
        self.lora_rank = lora_rank
        
        # Baseline learnable weights
        self.baseline_weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.baseline_bias = nn.Parameter(torch.zeros(out_dim))
        
        # Hypernetwork to produce LOW-RANK weight updates
        hidden_dim = cond_dim * hidden_ratio
        
        # Generate LoRA A matrix: (rank, in_dim)
        self.hyper_net_A = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, lora_rank * in_dim)
        )
        
        # Generate LoRA B matrix: (out_dim, rank)
        self.hyper_net_B = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim * lora_rank)
        )
        
        if sn:
            # Apply spectral norm to hypernetworks
            self.hyper_net_A[0] = spectral_norm(self.hyper_net_A[0])
            self.hyper_net_A[2] = spectral_norm(self.hyper_net_A[2])
            self.hyper_net_B[0] = spectral_norm(self.hyper_net_B[0])
            self.hyper_net_B[2] = spectral_norm(self.hyper_net_B[2])
    
    def forward(self, x, cond):
        """
        Args:
            x: (B, ..., in_dim) - input to project
            cond: (B, cond_dim) or (B, N, cond_dim) - conditioning input for hypernetwork
        Returns:
            out: (B, ..., out_dim)
        """
        # If cond has sequence dimension, pool it
        if cond.dim() == 3:  # (B, N, cond_dim)
            cond = cond.mean(dim=1)  # (B, cond_dim)
        
        B = x.size(0)
        
        # Generate low-rank matrices from hypernetwork
        lora_A = self.hyper_net_A(cond)  # (B, rank * in_dim)
        lora_A = lora_A.view(B, self.lora_rank, self.in_dim)  # (B, rank, in_dim)
        
        lora_B = self.hyper_net_B(cond)  # (B, out_dim * rank)
        lora_B = lora_B.view(B, self.out_dim, self.lora_rank)  # (B, out_dim, rank)
        
        # Compute low-rank weight delta: B @ A = (out_dim, rank) @ (rank, in_dim) = (out_dim, in_dim)
        weight_delta = torch.bmm(lora_B, lora_A)  # (B, out_dim, in_dim)
        
        # Combine baseline and delta with small scaling factor
        weights = self.baseline_weight.unsqueeze(0) + 0.1 * weight_delta  # (B, out_dim, in_dim)
        
        # Apply projection with batch-specific weights
        orig_shape = x.shape
        x_flat = x.view(B, -1, self.in_dim)  # (B, N, in_dim)
        
        # Batch matrix multiply: (B, N, in_dim) @ (B, in_dim, out_dim) = (B, N, out_dim)
        out = torch.bmm(x_flat, weights.transpose(1, 2))
        out = out + self.baseline_bias.unsqueeze(0).unsqueeze(0)
        
        # Reshape back to original shape
        out = out.view(*orig_shape[:-1], self.out_dim)
        
        return out


class HyperMixerConvProj(nn.Module):
    """
    HyperMixer-based convolutional projection with LoRA decomposition.
    A baseline kernel is learned, and a hypernetwork produces low-rank kernel updates.
    
    For convolutions, we treat the kernel as a matrix of shape:
    (out_ch, in_ch * kernel_size * kernel_size) and apply LoRA decomposition.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int,
                 cond_ch: int, lora_rank: int = 32, hidden_ratio: int = 2, 
                 padding: int = 0, sn: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.cond_ch = cond_ch
        self.lora_rank = lora_rank
        
        # Baseline learnable kernel
        self.baseline_weight = nn.Parameter(
            torch.randn(out_ch, in_ch, kernel_size, kernel_size) * 0.02
        )
        
        # For LoRA decomposition, treat kernel as matrix of shape (out_ch, in_ch*K*K)
        kernel_in_dim = in_ch * kernel_size * kernel_size
        
        # Hypernetwork conditioning
        cond_dim = cond_ch * kernel_size * kernel_size
        hidden_dim = cond_dim * hidden_ratio
        
        # Generate LoRA A matrix: (rank, in_ch*K*K)
        self.hyper_net_A = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),#nn.GELU(),
            nn.Linear(hidden_dim, lora_rank * kernel_in_dim)
        )
        
        # Generate LoRA B matrix: (out_ch, rank)
        self.hyper_net_B = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),#nn.GELU(),
            nn.Linear(hidden_dim, out_ch * lora_rank)
        )
        
        if sn:
            self.hyper_net_A[0] = spectral_norm(self.hyper_net_A[0])
            self.hyper_net_A[2] = spectral_norm(self.hyper_net_A[2])
            self.hyper_net_B[0] = spectral_norm(self.hyper_net_B[0])
            self.hyper_net_B[2] = spectral_norm(self.hyper_net_B[2])
    
    def forward(self, x):
        """
        Args:
            x: (B, in_ch, H, W)
        Returns:
            out: (B, out_ch, H', W')
        """
        B, C, H, W = x.shape
        kernel_in_dim = self.in_ch * self.kernel_size * self.kernel_size
        
        # Extract conditioning from input patches
        cond_patches = nn.functional.unfold(x, kernel_size=self.kernel_size, 
                                           stride=self.stride, padding=self.padding)
        cond = cond_patches.mean(dim=2).view(B, -1)  # (B, C*K*K)
        
        # Generate low-rank matrices from hypernetwork
        lora_A = self.hyper_net_A(cond)  # (B, rank * kernel_in_dim)
        lora_A = lora_A.view(B, self.lora_rank, kernel_in_dim)  # (B, rank, in_ch*K*K)
        
        lora_B = self.hyper_net_B(cond)  # (B, out_ch * rank)
        lora_B = lora_B.view(B, self.out_ch, self.lora_rank)  # (B, out_ch, rank)
        
        # Compute low-rank kernel delta: B @ A
        kernel_delta = torch.bmm(lora_B, lora_A)  # (B, out_ch, in_ch*K*K)
        kernel_delta = kernel_delta.view(B, self.out_ch, self.in_ch, 
                                        self.kernel_size, self.kernel_size)
        
        # Combine baseline and delta (use mean across batch for stability)
        kernel_delta_mean = kernel_delta.mean(dim=0)
        kernel = self.baseline_weight + 0.1 * kernel_delta_mean
        
        # Apply convolution with updated kernel
        out = nn.functional.conv2d(x, kernel, stride=self.stride, padding=self.padding)
        
        return out


class MultiheadConvTransposeProj(nn.Module):
    """Multiheaded transposed convolution for upsampling in generator."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int,
                 num_heads: int = 4, padding: int = 0, sn: bool = False):
        super().__init__()
        assert in_ch % num_heads == 0
        assert out_ch % num_heads == 0
        
        self.num_heads = num_heads
        self.head_in_ch = in_ch // num_heads
        self.head_out_ch = out_ch // num_heads
        
        self.head_convs = nn.ModuleList()
        for _ in range(num_heads):
            conv = nn.ConvTranspose2d(self.head_in_ch, self.head_out_ch,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding, bias=False)
            if sn:
                conv = spectral_norm(conv)
            self.head_convs.append(conv)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.num_heads, self.head_in_ch, H, W)
        
        head_outputs = []
        for i in range(self.num_heads):
            head_out = self.head_convs[i](x[:, i])
            head_outputs.append(head_out)
        
        out = torch.cat(head_outputs, dim=1)
        return out


class HyperMixerConvTransposeProj(nn.Module):
    """
    HyperMixer-based transposed convolutional projection with LoRA decomposition.
    A baseline kernel is learned, and a hypernetwork produces low-rank kernel updates.
    
    For transposed convolutions, we treat the kernel as a matrix of shape:
    (in_ch, out_ch * kernel_size * kernel_size) and apply LoRA decomposition.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int,
                 cond_ch: int, lora_rank: int = 32, hidden_ratio: int = 2, 
                 padding: int = 0, sn: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.cond_ch = cond_ch
        self.lora_rank = lora_rank
        
        # Baseline learnable kernel for ConvTranspose2d
        # Note: ConvTranspose2d weight shape is (in_channels, out_channels, kernel_h, kernel_w)
        self.baseline_weight = nn.Parameter(
            torch.randn(in_ch, out_ch, kernel_size, kernel_size) * 0.02
        )
        
        # For LoRA decomposition, treat kernel as matrix of shape (in_ch, out_ch*K*K)
        kernel_out_dim = out_ch * kernel_size * kernel_size
        
        # Hypernetwork conditioning - condition on spatial features
        # Average pool the input features to get conditioning signal
        cond_dim = cond_ch
        hidden_dim = cond_dim * hidden_ratio
        
        # Generate LoRA A matrix: (rank, out_ch*K*K)
        self.hyper_net_A = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),#nn.GELU(),
            nn.Linear(hidden_dim, lora_rank * kernel_out_dim)
        )
        
        # Generate LoRA B matrix: (in_ch, rank)
        self.hyper_net_B = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),#nn.GELU(),
            nn.Linear(hidden_dim, in_ch * lora_rank)
        )
        
        if sn:
            self.hyper_net_A[0] = spectral_norm(self.hyper_net_A[0])
            self.hyper_net_A[2] = spectral_norm(self.hyper_net_A[2])
            self.hyper_net_B[0] = spectral_norm(self.hyper_net_B[0])
            self.hyper_net_B[2] = spectral_norm(self.hyper_net_B[2])
    
    def forward(self, x):
        """
        Args:
            x: (B, in_ch, H, W) - input feature map
        Returns:
            out: (B, out_ch, H', W') - upsampled output
        """
        B, C, H, W = x.shape
        kernel_out_dim = self.out_ch * self.kernel_size * self.kernel_size
        
        # Extract conditioning from input features via global average pooling
        cond = torch.mean(x, dim=[2, 3])  # (B, in_ch)
        
        # Generate low-rank matrices from hypernetwork
        lora_A = self.hyper_net_A(cond)  # (B, rank * kernel_out_dim)
        lora_A = lora_A.view(B, self.lora_rank, kernel_out_dim)  # (B, rank, out_ch*K*K)
        
        lora_B = self.hyper_net_B(cond)  # (B, in_ch * rank)
        lora_B = lora_B.view(B, self.in_ch, self.lora_rank)  # (B, in_ch, rank)
        
        # Compute low-rank kernel delta: B @ A
        kernel_delta = torch.bmm(lora_B, lora_A)  # (B, in_ch, out_ch*K*K)
        kernel_delta = kernel_delta.view(B, self.in_ch, self.out_ch, 
                                        self.kernel_size, self.kernel_size)
        
        # Combine baseline and delta (use mean across batch for stability)
        kernel_delta_mean = kernel_delta.mean(dim=0)
        kernel = self.baseline_weight + 0.1 * kernel_delta_mean
        
        # Apply transposed convolution with updated kernel
        out = nn.functional.conv_transpose2d(x, kernel, stride=self.stride, padding=self.padding)
        
        return out


# =====================
#  PatchMLP Models with Extended Projection Options
# =====================
class SimpleMLP(nn.Module):
    """Simple MLP block without spatial gating - just feedforward with skip connection."""
    def __init__(self, dim: int, mlp_ratio: int = 1, dropout: float = 0.0, use_norm: bool = True, sn: bool = False):
        super().__init__()
        hidden_dim = dim * mlp_ratio
        
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        if sn == False:
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                Sial(),#nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)
            )
        else:
            self.mlp = nn.Sequential(
                spectral_norm(nn.Linear(dim, hidden_dim)),
                Sial(),#nn.SiLU(),
                nn.Dropout(dropout),
                spectral_norm(nn.Linear(hidden_dim, dim)),
                nn.Dropout(dropout)
            )
        
        # Learnable skip connection weights
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        """x: (B, N, C)"""
        shortcut = x
        x = self.norm(x)
        x = self.mlp(x)
        return self.alpha * x + self.beta * shortcut

class LatentProjShared(nn.Module):
    """
    Projects z -> a single content token, then broadcasts to all patches.
    Optional learned per-patch bias (very small) can be enabled to add a tiny, explicit positional nudge.
    """
    def __init__(self, zdim, hidden_dim, num_patches, use_pos_bias: bool = False, pos_bias_init_std: float = 0.01):
        super().__init__()
        self.shared = nn.Linear(zdim, hidden_dim)
        self.use_pos_bias = use_pos_bias
        if use_pos_bias:
            self.pos_bias = nn.Parameter(torch.randn(num_patches, hidden_dim) * pos_bias_init_std)

    def forward(self, z, num_patches):
        B = z.size(0)
        token = self.shared(z)                             # (B, C)
        x = token[:, None, :].expand(B, num_patches, -1)   # (B, N, C)
        if self.use_pos_bias:
            x = x + self.pos_bias[None, :, :]              # broadcast (1, N, C)
        return x


class PosFiLM(nn.Module):
    """
    Fixed 2D pos codes -> gamma,beta; both are modulated by z each sample.
    This breaks symmetry (needs pos) and injects diversity (from z).
    """
    def __init__(self, H, W, hidden_dim, pos_dim=32, zdim=128):
        super().__init__()
        self.H, self.W = H, W
        # fixed sinusoidal pos (or learnable; sinusoid keeps params tiny)
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij'
        )
        bands = [xx, yy]
        for k in (1,2,4,8):  # 4 bands -> pos_dim ~ 2 + 4*4 = 18; tweak as you like
            bands += [torch.sin(k*xx), torch.cos(k*xx), torch.sin(k*yy), torch.cos(k*yy)]
        pe = torch.stack(bands, dim=-1)                # (H,W,F)
        self.register_buffer('pos', pe.view(H*W, -1))  # (N, F)

        Fpos = self.pos.size(-1)
        self.pos_to_ab = nn.Linear(Fpos, 2*hidden_dim)
        self.z_to_gate = nn.Linear(zdim, 2*hidden_dim)  # modulates gamma,beta per sample

    def forward(self, x, z):
        # x: (B, N, C), z: (B, zdim)
        B, N, C = x.shape
        ab = self.pos_to_ab(self.pos)                  # (N, 2C) -> (gamma_pos, beta_pos)
        a_pos, b_pos = ab.chunk(2, dim=-1)             # (N,C), (N,C)
        a_pos = a_pos.unsqueeze(0).expand(B, -1, -1)   # (B,N,C)
        b_pos = b_pos.unsqueeze(0).expand(B, -1, -1)

        gz = self.z_to_gate(z).unsqueeze(1)            # (B,1,2C)
        a_z, b_z = gz.chunk(2, dim=-1)                 # (B,1,C) each

        gamma = 1 + a_pos * torch.tanh(a_z)            # pos provides structure; z scales it per-sample
        beta  = b_pos + b_z                            # z shifts content globally per-sample
        return x * gamma + beta
class GlobalMLP(nn.Module):
    """
    Global MLP that processes all patches at once rather than per-patch.
    Takes (B, N, C) -> flattens to (B, N*C) -> MLP -> (B, N*C) -> reshape to (B, N, C)
    This allows the MLP to mix information across all patches simultaneously.
    """
    def __init__(self, num_patches: int, dim: int, mlp_ratio: int = 1, dropout: float = 0.0, use_norm: bool = True, sn: bool = False):
        super().__init__()
        self.num_patches = num_patches
        self.dim = dim
        total_dim = num_patches * dim
        hidden_dim = total_dim * mlp_ratio
        
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        if sn == False:
            self.mlp = nn.Sequential(
                nn.Linear(total_dim, hidden_dim),
                Sial(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, total_dim),
                nn.Dropout(dropout)
            )
        else:
            self.mlp = nn.Sequential(
                spectral_norm(nn.Linear(total_dim, hidden_dim)),
                Sial(),
                nn.Dropout(dropout),
                spectral_norm(nn.Linear(hidden_dim, total_dim)),
                nn.Dropout(dropout)
            )
        
        # Learnable skip connection weights
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        """x: (B, N, C)"""
        B, N, C = x.shape
        shortcut = x
        
        # Normalize per-patch, then flatten
        x = self.norm(x)  # (B, N, C)
        x = x.view(B, -1)  # (B, N*C)
        
        # Global MLP
        x = self.mlp(x)  # (B, N*C)
        
        # Reshape back
        x = x.view(B, N, C)  # (B, N, C)
        
        return self.alpha * x + self.beta * shortcut

class PatchMLPGenerator(nn.Module):
    """
    Patch-based MLP generator with selectable latent projection:
      - 'per_patch' (original): z -> (N*C) then view to (B,N,C) [implicit positional info]
      - 'shared' (Option A):   z -> C then broadcast to all patches (no implicit pos info)
    
    And selectable MLP mode:
      - global_mlp=False (default): Each patch token processed independently with shared weights
      - global_mlp=True: Single MLP processes all patches at once (flattened to N*C)
    """
    def __init__(self, zdim: int, img_ch: int, hidden_dim: int, num_layers: int,
                 img_size: int, patch_size: int, overlap: int = 0, mlp_ratio: int = 4,
                 hierarchical: bool = True, min_patch_size: int = 2, use_pos_embed: bool = False,
                 proj_type: str = 'linear', num_heads: int = 4, hypermixer_hidden_ratio: int = 4,
                 latent_proj_mode: str = 'shared',  # options: 'shared', 'per_patch'
                 use_pos_bias: bool = False,        # optional tiny per-patch bias for 'shared'
                 global_mlp: bool = False           # NEW: use global MLP instead of per-patch
                 ):
        super().__init__()
        self.zdim = zdim
        self.img_ch = img_ch
        self.img_size = img_size
        self.patch_size = patch_size
        self.overlap = overlap
        self.hidden_dim = hidden_dim
        self.hierarchical = hierarchical
        self.min_patch_size = min_patch_size
        self.use_pos_embed = use_pos_embed
        self.proj_type = proj_type
        self.num_heads = num_heads
        self.latent_proj_mode = latent_proj_mode
        self.global_mlp = global_mlp

        # grid
        self.stride = patch_size - overlap
        assert self.patch_size > 0 and self.stride > 0, "patch_size and stride must be > 0"
        assert (img_size - patch_size) % self.stride == 0, "img_size, patch_size, overlap produce non-integer grid"
        self.patches_per_side = (img_size - patch_size) // self.stride + 1
        self.num_patches = self.patches_per_side ** 2
        if use_pos_bias == True:
            self.use_film = False
        else:
            self.use_film = True
        # latent projection
        if latent_proj_mode == 'per_patch':
            self.latent_proj = nn.Linear(zdim, self.num_patches * hidden_dim)
            self.latent_is_per_patch = True
        elif latent_proj_mode == 'shared':
            self.latent_proj = LatentProjShared(zdim, hidden_dim, self.num_patches, use_pos_bias=use_pos_bias)
            if self.use_film == True:
                self.pos_film = PosFiLM(self.patches_per_side, self.patches_per_side, hidden_dim, 32, zdim)
            self.latent_is_per_patch = False
        else:
            raise ValueError("latent_proj_mode must be 'per_patch' or 'shared'")

        # positional embeddings hook (kept as Identity for now)
        self.pos_embed = nn.Identity()

        # MLP trunk - choose between per-patch or global mode
        if global_mlp:
            self.blocks = nn.ModuleList([
                GlobalMLP(self.num_patches, hidden_dim, mlp_ratio, dropout=0.0, use_norm=True)
                for _ in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                SimpleMLP(hidden_dim, mlp_ratio, dropout=0.0, use_norm=True)
                for _ in range(num_layers)
            ])
        self.norm = nn.LayerNorm(hidden_dim)

        # heads
        self._build_patch_head(proj_type, num_heads, hypermixer_hidden_ratio)
        self.out_act = nn.Identity()  # typical GAN range [-1,1]; change if your D expects 0..1

        
    def _build_patch_head(self, proj_type: str, num_heads: int, hypermixer_hidden_ratio: int):
        """Build the appropriate patch head based on projection type."""
        if proj_type == 'linear':
            self.patch_head = nn.Linear(self.hidden_dim, self.img_ch * self.patch_size * self.patch_size)
            self.use_linear_head = True
            
        elif proj_type == 'multihead_linear':
            out_dim = self.img_ch * self.patch_size * self.patch_size
            self.patch_head = MultiheadLinearProj(self.hidden_dim, out_dim, num_heads, sn=False)
            self.use_linear_head = True
            
        elif proj_type == 'hypermixer_linear':
            out_dim = self.img_ch * self.patch_size * self.patch_size
            # Condition on the normalized patch tokens from last layer
            self.patch_head = HyperMixerLinearProj(
                self.hidden_dim, out_dim, self.hidden_dim, 
                lora_rank=32,  # Default rank of 32
                hidden_ratio=hypermixer_hidden_ratio, sn=False
            )
            self.use_linear_head = True
            self.use_hypermixer = True
            
        elif proj_type == 'conv':
            if self.hierarchical:
                from .gmlp_models import HierarchicalPatchReconstruct  # Assuming this exists
                self.patch_head = HierarchicalPatchReconstruct(
                    self.hidden_dim, 
                    self.img_ch, 
                    self.patch_size,
                    self.min_patch_size,
                    self.stride
                )
            else:
                self.patch_head = nn.ConvTranspose2d(
                    self.hidden_dim, 
                    self.img_ch, 
                    kernel_size=self.patch_size,
                    stride=self.stride,
                    padding=0,
                    bias=False
                )
            self.use_linear_head = False
            
        elif proj_type == 'multihead_conv':
            if self.hierarchical:
                # Not implemented for hierarchical
                raise NotImplementedError("Multihead conv not supported with hierarchical reconstruction")
            self.patch_head = MultiheadConvTransposeProj(
                self.hidden_dim,
                self.img_ch,
                kernel_size=self.patch_size,
                stride=self.stride,
                num_heads=num_heads,
                padding=0,
                sn=False
            )
            self.use_linear_head = False
            
        elif proj_type == 'hypermixer_conv':
            if self.hierarchical:
                raise NotImplementedError("HyperMixer conv not supported with hierarchical reconstruction")
            
            # HyperMixer transposed convolution with LoRA (rank=32)
            self.patch_head = HyperMixerConvTransposeProj(
                self.hidden_dim,
                self.img_ch,
                kernel_size=self.patch_size,
                stride=self.stride,
                cond_ch=self.hidden_dim,
                lora_rank=32,  # Default rank of 32
                hidden_ratio=hypermixer_hidden_ratio,
                padding=0,
                sn=False
            )
            self.use_linear_head = False
            
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")
        
    def forward(self, z):
        B = z.size(0)

        # Latent → tokens (B, N, C)
        if self.latent_is_per_patch:
            x = self.latent_proj(z).view(B, self.num_patches, self.hidden_dim)
        else:
            x = self.latent_proj(z, self.num_patches)  # shared content broadcast
            if self.use_film == True:
                x = self.pos_film(x, z)                     # (B, N, C)

        # (Optional) explicit pos-emb if you later want it
        #x = self.pos_embed(x)

        # MLP trunk (either per-patch SimpleMLP or global GlobalMLP)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Linear head per patch
        patches = self.patch_head(x)  # (B, N, C*P*P)
        patches = patches.view(B, self.num_patches, self.img_ch, self.patch_size, self.patch_size)

        # Assemble image (no overlap for patch==stride; for overlaps, prefer F.fold)
        if self.overlap == 0:
            H = W = self.patches_per_side
            img = patches.view(B, H, W, self.img_ch, self.patch_size, self.patch_size)\
                         .permute(0, 3, 1, 4, 2, 5).contiguous()\
                         .view(B, self.img_ch, H*self.patch_size, W*self.patch_size)
        else:
            # Overlap-safe assembly
            img = self._patches_to_image_fold(patches)

        return img
    
    def _patches_to_image(self, patches):
        """
        patches: (B, N, C, P, P)
        returns: (B, C, H, W)
        """
        B, N, C, P, _ = patches.shape
        H = W = self.img_size
        stride = self.stride
    
        # Arrange for F.fold: (B, C*P*P, N)
        patches_col = patches.view(B, N, C*P*P).transpose(1, 2).contiguous()
    
        # Place patches
        img = F.fold(
            patches_col,
            output_size=(H, W),
            kernel_size=(P, P),
            stride=(stride, stride)
        )
    
        # Overlap count via folding an all-ones "patch" tensor
        ones = patches.new_ones(B, N, C*P*P)
        ones = ones.transpose(1, 2).contiguous()
        count = F.fold(
            ones,
            output_size=(H, W),
            kernel_size=(P, P),
            stride=(stride, stride)
        )
    
        # Safe normalize
        img = img / (count + 1e-8)
        return img


# =====================
#  Enhanced Discriminator with Multi-Headed Attention Option
# =====================
class PatchMLPDiscriminator(nn.Module):
    def __init__(self, img_ch: int, hidden_dim: int, num_layers: int,
                 img_size: int, patch_size: int, overlap: int = 0, mlp_ratio: int = 4,
                 hierarchical: bool = True, min_patch_size: int = 2, 
                 pos_embed_type: str = 'trainable_2d',  # NEW: 'sinusoidal_2d', 'static_2d', 'trainable_2d', 'none'
                 proj_type: str = 'linear', num_heads: int = 4, hypermixer_hidden_ratio: int = 2,
                 use_attention_pooling: bool = True, pool_num_heads: int = 8,
                 pool_query_type: str = 'gated',  # 'learned', 'global', 'gated', 'multi'
                 use_mha_after_proj: bool = False,  # NEW: Multi-headed attention after projection
                 mha_num_heads: int = 8,  # NEW: Number of heads for post-projection MHA
                 mha_dropout: float = 0.0):  # NEW: Dropout for MHA
        """
        Enhanced PatchMLPDiscriminator with multiple positional embedding options
        and optional multi-headed attention after projection.
        
        Args:
            pos_embed_type: Type of positional embedding to use:
                - 'sinusoidal_2d': 2D sinusoidal positional embeddings (static)
                - 'static_2d': Static 2D Gaussian embeddings (frozen)
                - 'trainable_2d': Learnable 2D positional embeddings (default)
                - 'none': No positional embeddings
            use_mha_after_proj: If True, adds multi-headed attention between projection and first MLP layer
            mha_num_heads: Number of attention heads for post-projection MHA
            mha_dropout: Dropout rate for the MHA layer
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.overlap = overlap
        self.hierarchical = hierarchical
        self.min_patch_size = min_patch_size
        self.pos_embed_type = pos_embed_type
        self.proj_type = proj_type
        self.num_heads = num_heads
        self.use_attention_pooling = use_attention_pooling
        self.pool_query_type = pool_query_type
        self.use_mha_after_proj = use_mha_after_proj
        
        # Calculate patches
        self.stride = patch_size - overlap
        self.patches_per_side = (img_size - patch_size) // self.stride + 1
        self.num_patches = self.patches_per_side ** 2
        
        # Patch embedding based on projection type
        self._build_patch_embed(proj_type, num_heads, hypermixer_hidden_ratio, img_ch, hidden_dim)
        
        # Enhanced positional embeddings with multiple types
        self.pos_embed = PositionalEmbedding2D(
            self.patches_per_side,
            self.patches_per_side,
            hidden_dim,
            embed_type=pos_embed_type
        )
        
        # NEW: Optional multi-headed attention after projection
        if use_mha_after_proj:
            self.post_proj_mha = nn.MultiheadAttention(
                hidden_dim,
                num_heads=mha_num_heads,
                dropout=mha_dropout,
                batch_first=True
            )
            self.mha_norm = nn.LayerNorm(hidden_dim)
        else:
            self.post_proj_mha = None
        
        # Stack of simple MLP blocks with spectral norm
        self.blocks = nn.ModuleList([
            SimpleMLP(hidden_dim, mlp_ratio, dropout=0.0, use_norm=True, sn=True)
            for _ in range(num_layers)
        ])
        
        # Final layers
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Attention-based pooling with different query types
        if use_attention_pooling:
            self._build_attention_pooling(pool_query_type, hidden_dim, pool_num_heads)
        
        self.head = spectral_norm(nn.Linear(hidden_dim, 1))
    
    def _build_attention_pooling(self, query_type: str, hidden_dim: int, pool_num_heads: int):
        """Build attention pooling components based on query type."""
        if query_type == 'learned':
            # Original: fixed learnable query
            self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            self.query_proj = None
            
        elif query_type == 'global':
            # Query from global average of patches
            self.pool_query = None
            self.query_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
        elif query_type == 'gated':
            # Query from gated combination of global statistics
            self.pool_query = None
            self.query_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),  # mean + max
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
        elif query_type == 'multi':
            # Multiple learned queries that get selected/combined based on input
            self.num_query_slots = 4
            self.pool_query = nn.Parameter(torch.randn(1, self.num_query_slots, hidden_dim) * 0.02)
            self.query_selector = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, self.num_query_slots),
                nn.Softmax(dim=-1)
            )
            
        else:
            raise ValueError(f"Unknown pool_query_type: {query_type}")
        
        # Multi-head attention for pooling
        self.pool_attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=pool_num_heads,
            batch_first=True,
            dropout=0.0
        )
    
    def _build_patch_embed(self, proj_type: str, num_heads: int, 
                          hypermixer_hidden_ratio: int, img_ch: int, hidden_dim: int):
        """Build the appropriate patch embedding based on projection type."""
        self.use_hypermixer = False
        
        if proj_type == 'conv':
            if self.hierarchical:
                # Note: HierarchicalPatchEmbed needs to be imported from your models
                # Keeping the reference but commenting out the import
                # from .gmlp_models import HierarchicalPatchEmbed
                # self.patch_embed = HierarchicalPatchEmbed(...)
                raise NotImplementedError("HierarchicalPatchEmbed needs to be imported")
            else:
                self.patch_embed = nn.Conv2d(
                    img_ch,
                    hidden_dim,
                    kernel_size=self.patch_size,
                    stride=self.stride,
                    padding=0
                )
            self.use_linear_embed = False
            
        elif proj_type == 'linear':
            self.patch_to_embed = nn.Linear(img_ch * self.patch_size * self.patch_size, hidden_dim)
            self.use_linear_embed = True
            
        elif proj_type == 'multihead_linear':
            in_dim = img_ch * self.patch_size * self.patch_size
            self.patch_to_embed = MultiheadLinearProj(in_dim, hidden_dim, num_heads, sn=False)
            self.use_linear_embed = True
            
        elif proj_type == 'multihead_conv':
            if self.hierarchical:
                raise NotImplementedError("Multihead conv not supported with hierarchical embedding")
            self.patch_embed = MultiheadConvProj(
                img_ch,
                hidden_dim,
                kernel_size=self.patch_size,
                stride=self.stride,
                num_heads=num_heads,
                padding=0,
                sn=False
            )
            self.use_linear_embed = False
            
        elif proj_type == 'hypermixer_linear':
            in_dim = img_ch * self.patch_size * self.patch_size
            self.patch_to_embed = HyperMixerLinearProj(
                in_dim, hidden_dim, in_dim,
                hidden_ratio=hypermixer_hidden_ratio, sn=False
            )
            self.use_linear_embed = True
            self.use_hypermixer = True
            
        elif proj_type == 'hypermixer_conv':
            if self.hierarchical:
                raise NotImplementedError("HyperMixer conv not supported with hierarchical embedding")
            self.patch_embed = HyperMixerConvProj(
                img_ch,
                hidden_dim,
                kernel_size=self.patch_size,
                stride=self.stride,
                cond_ch=img_ch,
                hidden_ratio=hypermixer_hidden_ratio,
                padding=0,
                sn=False
            )
            self.use_linear_embed = False
            self.use_hypermixer = True
            
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")
        
    def forward(self, x):
        B = x.size(0)
        
        # Extract patches
        if self.use_linear_embed:
            patches = nn.functional.unfold(
                x, 
                kernel_size=self.patch_size,
                stride=self.stride
            )
            patches = patches.transpose(1, 2)
            
            if self.use_hypermixer:
                x = self.patch_to_embed(patches, cond=patches)
            else:
                x = self.patch_to_embed(patches)
        else:
            if self.use_hypermixer:
                x = self.patch_embed(x, cond=x)
            else:
                x = self.patch_embed(x)
            x = x.flatten(2).transpose(1, 2)
        
        # Add positional embeddings
        x = self.pos_embed(x)
        
        # NEW: Optional multi-headed attention after projection and pos embedding
        if self.use_mha_after_proj:
            # Self-attention across patches
            attn_out, _ = self.post_proj_mha(x, x, x)
            x = x + attn_out  # Residual connection
            x = self.mha_norm(x)
        
        # Apply MLP blocks
        for block in self.blocks:
            x = block(x)
        
        # Norm
        x = self.norm(x)
        
        # Attention-based pooling with content-aware query
        if self.use_attention_pooling:
            query = self._generate_query(x, B)  # Content-aware query generation
            
            # Attend to all patches
            pooled, _ = self.pool_attention(query, x, x)
            x = pooled.squeeze(1) if pooled.size(1) == 1 else pooled.mean(dim=1)
        else:
            x = x.mean(dim=1)
        
        # Final classification
        logits = self.head(x).squeeze(-1)
        
        return logits
    
    def _generate_query(self, x, B):
        """Generate query based on pool_query_type."""
        if self.pool_query_type == 'learned':
            # Fixed learned query
            return self.pool_query.expand(B, -1, -1)
        
        elif self.pool_query_type == 'global':
            # Query from global average
            global_feat = x.mean(dim=1, keepdim=True)  # (B, 1, hidden_dim)
            return self.query_proj(global_feat)
        
        elif self.pool_query_type == 'gated':
            # Query from both mean and max pooling
            mean_feat = x.mean(dim=1)  # (B, hidden_dim)
            max_feat = x.max(dim=1)[0]  # (B, hidden_dim)
            combined = torch.cat([mean_feat, max_feat], dim=-1)  # (B, 2*hidden_dim)
            query = self.query_proj(combined).unsqueeze(1)  # (B, 1, hidden_dim)
            return query
        
        elif self.pool_query_type == 'multi':
            # Select and combine multiple queries based on input
            global_feat = x.mean(dim=1)  # (B, hidden_dim)
            weights = self.query_selector(global_feat)  # (B, num_query_slots)
            
            # Weighted combination of query slots
            queries = self.pool_query.expand(B, -1, -1)  # (B, num_query_slots, hidden_dim)
            query = torch.einsum('bn,bnh->bh', weights, queries).unsqueeze(1)  # (B, 1, hidden_dim)
            return query

# =====================
#  gMLP Models (Spatial Gating Unit)
# =====================
# =====================================================================
# AXIAL GATING FOR gMLP
# =====================================================================
class HierarchicalPatchEmbed(nn.Module):
    """
    Hierarchical patch embedding: progressively merge from small patches to target size.
    
    Example: min_patch_size=8, target_patch_size=32
    - Stage 1: 8×8 patches → (B, N_large, hidden_dim//4)
    - Stage 2: Merge 2×2 → (B, N_medium, hidden_dim//2)
    - Stage 3: Merge 2×2 → (B, N_target, hidden_dim)
    """
    def __init__(self, img_ch: int, hidden_dim: int, patch_size: int, 
                 min_patch_size: int, stride: int, sn: bool = False):
        super().__init__()
        assert patch_size >= min_patch_size, "patch_size must be >= min_patch_size"
        assert patch_size % min_patch_size == 0, "patch_size must be divisible by min_patch_size"
        
        self.patch_size = patch_size
        self.min_patch_size = min_patch_size
        self.stride = stride
        
        # Calculate number of merge stages needed
        ratio = patch_size // min_patch_size
        self.num_stages = int(math.log2(ratio)) + 1 if ratio > 1 else 1
        
        self.stages = nn.ModuleList()
        
        current_patch_size = min_patch_size
        current_channels = img_ch
        
        # Build stages progressively
        for stage_idx in range(self.num_stages):
            if stage_idx == 0:
                # First stage: extract smallest patches
                stage_hidden_dim = hidden_dim // (2 ** (self.num_stages - 1))
                conv = nn.Conv2d(
                    current_channels,
                    stage_hidden_dim,
                    kernel_size=min_patch_size,
                    stride=min_patch_size,  # Non-overlapping at first stage
                    padding=0
                )
                if sn:
                    conv = spectral_norm(conv)
                self.stages.append(conv)
                current_channels = stage_hidden_dim
            else:
                # Subsequent stages: merge 2×2 patches with stride 2
                stage_hidden_dim = hidden_dim // (2 ** (self.num_stages - 1 - stage_idx))
                conv = nn.Conv2d(
                    current_channels,
                    stage_hidden_dim,
                    kernel_size=2,
                    stride=2,
                    padding=0
                )
                if sn:
                    conv = spectral_norm(conv)
                self.stages.append(nn.Sequential(
                    nn.SiLU(),
                    conv
                ))
                current_channels = stage_hidden_dim
                current_patch_size *= 2
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: (B, hidden_dim, H_out, W_out)
        """
        for stage in self.stages:
            x = stage(x)
        return x


class HierarchicalPatchReconstruct(nn.Module):
    """
    Hierarchical patch reconstruction: progressively expand from features to target image.
    
    Inverse of HierarchicalPatchEmbed.
    """
    def __init__(self, hidden_dim: int, img_ch: int, patch_size: int,
                 min_patch_size: int, stride: int):
        super().__init__()
        assert patch_size >= min_patch_size, "patch_size must be >= min_patch_size"
        assert patch_size % min_patch_size == 0, "patch_size must be divisible by min_patch_size"
        
        self.patch_size = patch_size
        self.min_patch_size = min_patch_size
        self.stride = stride
        
        # Calculate number of stages needed
        ratio = patch_size // min_patch_size
        self.num_stages = int(math.log2(ratio)) + 1 if ratio > 1 else 1
        
        self.stages = nn.ModuleList()
        
        # Build stages in reverse order (expanding)
        for stage_idx in range(self.num_stages):
            if stage_idx == self.num_stages - 1:
                # Last stage: final upsampling to image
                stage_input_dim = hidden_dim // (2 ** stage_idx)
                deconv = nn.ConvTranspose2d(
                    stage_input_dim,
                    img_ch,
                    kernel_size=min_patch_size,
                    stride=min_patch_size,
                    padding=0,
                    bias=False
                )
                self.stages.append(deconv)
            else:
                # Earlier stages: upsample 2x
                stage_input_dim = hidden_dim // (2 ** stage_idx)
                stage_output_dim = hidden_dim // (2 ** (stage_idx + 1))
                deconv = nn.ConvTranspose2d(
                    stage_input_dim,
                    stage_output_dim,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=False
                )
                self.stages.append(nn.Sequential(
                    deconv,
                    nn.SiLU()
                ))
    
    def forward(self, x):
        """
        x: (B, hidden_dim, H, W)
        Returns: (B, img_ch, H_out, W_out)
        """
        for stage in self.stages:
            x = stage(x)
        return x


# =====================
#  Spatial Gating Units
# =====================
# =====================
#  gMLP Models (Spatial Gating Unit)
# =====================
class SpatialGatingUnit(nn.Module):
    """Spatial Gating Unit from gMLP paper."""
    def __init__(self, dim: int, seq_len: int, sn: bool):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        
        # Split into two halves for gating
        self.norm = nn.LayerNorm(dim // 2)
        
        # Spatial projection (across sequence dimension)
        if sn == True:
            self.spatial_proj = spectral_norm(nn.Linear(seq_len, seq_len))
        else:
            self.spatial_proj = nn.Linear(seq_len, seq_len)
        
        # Initialize spatial projection to identity-like
        nn.init.zeros_(self.spatial_proj.weight)
        nn.init.ones_(self.spatial_proj.bias)
    
    def forward(self, x):
        """
        x: (B, N, C) where N is sequence length, C is channels
        """
        # Split into two halves
        u, v = x.chunk(2, dim=-1)  # Each: (B, N, C/2)
        
        # Normalize v
        v = self.norm(v)
        
        # Spatial projection: transpose to (B, C/2, N), project, transpose back
        v = v.transpose(1, 2)  # (B, C/2, N)
        v = self.spatial_proj(v)  # (B, C/2, N)
        v = v.transpose(1, 2)  # (B, N, C/2)
        
        # Element-wise gating
        return u * v


class gMLPBlock(nn.Module):
    """gMLP block with Spatial Gating Unit."""
    def __init__(self, dim: int, seq_len: int, mlp_ratio: int = 4, sn: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        if sn == True:
            self.channel_proj1 = spectral_norm(nn.Linear(dim, dim * mlp_ratio))
        else:
            self.channel_proj1 = nn.Linear(dim, dim * mlp_ratio)
        self.activation = nn.SiLU()
        self.sgu = SpatialGatingUnit(dim * mlp_ratio, seq_len, sn)
        if sn == True:
            self.channel_proj2 = spectral_norm(nn.Linear(dim * mlp_ratio // 2, dim))
        else:
            self.channel_proj2 = nn.Linear(dim * mlp_ratio // 2, dim)
        
    def forward(self, x):
        """
        x: (B, N, C)
        """
        shortcut = x
        
        # Norm
        x = self.norm(x)
        
        # Channel expansion
        x = self.channel_proj1(x)
        x = self.activation(x)
        
        # Spatial Gating Unit (reduces channels by half)
        x = self.sgu(x)
        
        # Channel reduction
        x = self.channel_proj2(x)
        
        # Residual
        return x + shortcut


class gMLPGenerator(nn.Module):
    """gMLP Generator for image generation."""
    def __init__(self, zdim: int, img_ch: int, hidden_dim: int, num_layers: int,
                 img_size: int, patch_h: int, patch_w: int, overlap: int = 0, mlp_ratio: int = 6):
        super().__init__()
        self.zdim = zdim
        self.img_ch = img_ch
        self.img_size = img_size
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.overlap = overlap
        self.hidden_dim = hidden_dim
        
        # Calculate patches
        self.stride_h = patch_h - overlap
        self.stride_w = patch_w - overlap
        num_patches_h = (img_size - patch_h) // self.stride_h + 1
        num_patches_w = (img_size - patch_w) // self.stride_w + 1
        self.num_patches = num_patches_h * num_patches_w
        
        # Initial projection from latent
        self.latent_proj = nn.Linear(zdim, self.num_patches * hidden_dim)
        
        # Stack of gMLP blocks
        self.blocks = nn.ModuleList([
            gMLPBlock(hidden_dim, self.num_patches, mlp_ratio)
            for _ in range(num_layers)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Patch generator: each patch becomes patch_h * patch_w * img_ch pixels
        patch_dim = patch_h * patch_w * img_ch
        self.patch_head = nn.Linear(hidden_dim, patch_dim, bias=False)
        
        self.out_act = nn.Identity()
        
    def forward(self, z):
        B = z.size(0)
        
        # Project latent to patch tokens
        x = self.latent_proj(z)  # (B, num_patches * hidden_dim)
        x = x.view(B, self.num_patches, self.hidden_dim)  # (B, N, C)
        
        # Apply gMLP blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm
        x = self.norm(x)
        
        # Generate patches
        patches = self.patch_head(x)  # (B, N, patch_dim)
        
        # Reshape for F.fold: (B, patch_dim, N)
        patches = patches.transpose(1, 2)
        
        # Reconstruct image from patches
        img = F.fold(
            patches,
            output_size=(self.img_size, self.img_size),
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.stride_h, self.stride_w)
        )
        
        # Handle overlaps by averaging
        if self.overlap > 0:
            ones = torch.ones_like(patches)
            overlap_count = F.fold(
                ones,
                output_size=(self.img_size, self.img_size),
                kernel_size=(self.patch_h, self.patch_w),
                stride=(self.stride_h, self.stride_w)
            )
            img = img / (overlap_count + 1e-8)
        
        img = self.out_act(img)
        
        return img


class gMLPDiscriminator(nn.Module):
    """gMLP Discriminator for image classification."""
    def __init__(self, img_ch: int, hidden_dim: int, num_layers: int,
                 img_size: int, patch_h: int, patch_w: int, overlap: int = 0, mlp_ratio: int = 4):
        super().__init__()
        self.img_size = img_size
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.overlap = overlap
        
        # Calculate patches
        self.stride_h = patch_h - overlap
        self.stride_w = patch_w - overlap
        num_patches_h = (img_size - patch_h) // self.stride_h + 1
        num_patches_w = (img_size - patch_w) // self.stride_w + 1
        self.num_patches = num_patches_h * num_patches_w
        
        patch_dim = img_ch * patch_h * patch_w
        
        # Patch embedding with spectral norm
        self.patch_embed = spectral_norm(nn.Linear(patch_dim, hidden_dim))
        
        # Stack of gMLP blocks
        self.blocks = nn.ModuleList([
            gMLPBlock(hidden_dim, self.num_patches, mlp_ratio, sn=True)
            for _ in range(num_layers)
        ])
        
        # Final norm and head
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = spectral_norm(nn.Linear(hidden_dim, 1))
        
    def forward(self, x):
        B = x.size(0)
        
        # Extract patches
        patches = F.unfold(x, kernel_size=(self.patch_h, self.patch_w), stride=(self.stride_h, self.stride_w))
        patches = patches.transpose(1, 2)  # (B, N, patch_dim)
        
        # Embed patches
        x = self.patch_embed(patches)  # (B, N, hidden_dim)
        
        # Apply gMLP blocks
        for block in self.blocks:
            x = block(x)
        
        # Norm
        x = self.norm(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (B, hidden_dim)
        
        # Classification
        logits = self.head(x).squeeze(-1)
        
        return logits

def make_attn(attn_type: int, channels: int, heads: Optional[int] = None) -> Optional[nn.Module]:
    if attn_type == 1:
        return SelfAttention2D(channels)
    if attn_type == 2:
        h = heads if heads and channels % heads == 0 else max(1, min(8, channels // 64))
        return MultiHeadSelfAttention2D(channels, heads=h)
    return None


# =====================
#  Losses
# =====================
class GANLosses:
    # === persistent EMA buffers ===
    _ema_initialized = False
    _ema_mr = None
    _ema_mf = None
    @staticmethod
    def dcgan_d(d_real, d_fake):
        bce = nn.functional.binary_cross_entropy_with_logits
        ones = torch.ones_like(d_real)
        zeros = torch.zeros_like(d_fake)
        return bce(d_real, ones) + bce(d_fake, zeros)

    @staticmethod
    def dcgan_g(d_fake):
        bce = nn.functional.binary_cross_entropy_with_logits
        ones = torch.ones_like(d_fake)
        return bce(d_fake, ones)

    @staticmethod
    def lsgan_d(d_real, d_fake):
        mse = nn.MSELoss()
        ones = torch.ones_like(d_real)
        zeros = torch.zeros_like(d_fake)
        return mse(d_real, ones) + mse(d_fake, zeros)

    @staticmethod
    def lsgan_g(d_fake):
        mse = nn.MSELoss()
        ones = torch.ones_like(d_fake)
        return mse(d_fake, ones)

    @staticmethod
    def hbgan_d(d_real, d_fake):
        mse = nn.HuberLoss()
        ones = torch.ones_like(d_real)
        zeros = torch.zeros_like(d_fake)
        return mse(d_real, ones) + mse(d_fake, zeros)

    @staticmethod
    def hbgan_g(d_fake):
        mse = nn.HuberLoss()
        ones = torch.ones_like(d_fake)
        return mse(d_fake, ones) * 0.5

    @staticmethod
    def logistic_d(d_real, d_fake):
        return F.softplus(-d_real).mean() + F.softplus(d_fake).mean()

    @staticmethod
    def logistic_g(d_fake):
        return F.softplus(-d_fake).mean()

    @staticmethod
    def hinge_d(d_real, d_fake):
        return torch.clamp((1. - d_real), min=0.0).mean() + torch.clamp((1. + d_fake), min=0.0).mean()

    @staticmethod
    def hinge_g(d_fake):
        return (-d_fake).mean()

    @staticmethod
    def wgan_d(d_real, d_fake):
        return -(d_real.mean() - d_fake.mean())

    @staticmethod
    def wgan_g(d_fake):
        return -d_fake.mean()
    
    @staticmethod
    def dagan_d(d_real, d_fake, alpha=0.2, margin=1.0):
        """DAGAN Discriminator Loss"""
        real_loss = torch.clamp((margin - d_real), min=0.0).mean()
        fake_confidence = torch.sigmoid(d_fake).mean()
        adaptive_margin = margin * (1.0 + alpha * torch.clamp(0.5 - fake_confidence, min=0.0))
        fake_loss = torch.clamp((adaptive_margin + d_fake), min=0.0).mean()
        return real_loss + fake_loss
    
    @staticmethod
    def dagan_g(d_fake, fake_samples=None, beta=0.1, diversity_mode='cosine'):
        """DAGAN Generator Loss"""
        adversarial_loss = -d_fake.mean()
        diversity_loss = 0.0
        
        if fake_samples is not None and fake_samples.shape[0] > 1:
            batch_size = fake_samples.shape[0]
            flat_samples = fake_samples.view(batch_size, -1)
            
            if diversity_mode == 'cosine':
                normalized = F.normalize(flat_samples, p=2, dim=1)
                similarity_matrix = torch.mm(normalized, normalized.t())
                mask = ~torch.eye(batch_size, dtype=torch.bool, device=fake_samples.device)
                similarities = similarity_matrix[mask]
                diversity_loss = (similarities ** 2).mean()
                
            elif diversity_mode == 'l2':
                dists_sq = torch.cdist(flat_samples, flat_samples, p=2) ** 2
                mask = ~torch.eye(batch_size, dtype=torch.bool, device=fake_samples.device)
                dists_sq = dists_sq[mask]
                diversity_loss = torch.exp(-dists_sq / flat_samples.shape[1]).mean()
                
            elif diversity_mode == 'learned':
                feature_variance = flat_samples.var(dim=0).mean()
                diversity_loss = -torch.log(feature_variance + 1e-8)
        
        total_loss = adversarial_loss + beta * diversity_loss
        return total_loss

    @staticmethod
    def _ra_pairs(
        d_real: torch.Tensor,
        d_fake: torch.Tensor,
        detach_fake_mean: bool = False,
        detach_real_mean: bool = False,
        use_ema: bool = False,
        ema_decay: float = 0.9,
    ):
        """Produce the relativistic-average logits with optional detach and EMA smoothing."""
        device = d_real.device

        # Initialize EMA buffers
        if not GANLosses._ema_initialized or GANLosses._ema_mr is None or GANLosses._ema_mf is None:
            GANLosses._ema_mr = d_real.detach().mean().to(device)
            GANLosses._ema_mf = d_fake.detach().mean().to(device)
            GANLosses._ema_initialized = True

        # Compute means
        mean_real = d_real.mean()
        mean_fake = d_fake.mean()

        # Apply detach if requested
        if detach_real_mean:
            mean_real = mean_real.detach()
        if detach_fake_mean:
            mean_fake = mean_fake.detach()

        # EMA smoothing (helps stability with small batch or irregular D)
        if use_ema:
            GANLosses._ema_mr = ema_decay * GANLosses._ema_mr + (1 - ema_decay) * mean_real.detach()
            GANLosses._ema_mf = ema_decay * GANLosses._ema_mf + (1 - ema_decay) * mean_fake.detach()
            mean_real = GANLosses._ema_mr
            mean_fake = GANLosses._ema_mf

        # Compute relativistic logits
        r = d_real - mean_fake
        f = d_fake - mean_real
        return r, f

    @staticmethod
    def rasgan_d(d_real, d_fake):
        r, f = GANLosses._ra_pairs(d_real, d_fake)
        bce = nn.functional.binary_cross_entropy_with_logits
        ones_r = torch.ones_like(r)
        zeros_f = torch.zeros_like(f)
        return bce(r, ones_r) + bce(f, zeros_f)

    @staticmethod
    def rasgan_g(d_real, d_fake):
        r, f = GANLosses._ra_pairs(d_real, d_fake)
        bce = nn.functional.binary_cross_entropy_with_logits
        zeros_r = torch.zeros_like(r)
        ones_f = torch.ones_like(f)
        return bce(r, zeros_r) + bce(f, ones_f)

    @staticmethod
    def ralsgan_d(d_real, d_fake):
        r, f = GANLosses._ra_pairs(d_real, d_fake)
        mse = nn.MSELoss()
        return mse(r, torch.ones_like(r)) + mse(f, torch.zeros_like(f))

    @staticmethod
    def ralsgan_g(d_real, d_fake):
        r, f = GANLosses._ra_pairs(d_real, d_fake)
        mse = nn.MSELoss()
        return mse(r, torch.zeros_like(r)) + mse(f, torch.ones_like(f))

    @staticmethod
    def rahbgan_d(d_real, d_fake):
        r, f = GANLosses._ra_pairs(d_real, d_fake)
        mse = nn.HuberLoss()
        return mse(r, torch.ones_like(r)) + mse(f, torch.zeros_like(f))

    @staticmethod
    def rahbgan_g(d_real, d_fake):
        r, f = GANLosses._ra_pairs(d_real, d_fake)
        mse = nn.HuberLoss()
        return mse(r, torch.zeros_like(r)) + mse(f, torch.ones_like(f))

    @staticmethod
    def rahinge_d(d_real, d_fake):
        r, f = GANLosses._ra_pairs(d_real, d_fake)
        return F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()

    @staticmethod
    def rahinge_g(d_real, d_fake):
        r, f = GANLosses._ra_pairs(d_real, d_fake)
        return F.relu(1.0 + r).mean() + F.relu(1.0 - f).mean()

    @staticmethod
    def ralogistic_d(d_real, d_fake):
        r, f = GANLosses._ra_pairs(d_real, d_fake)
        #loss = -r.mean() + f.mean()
        return F.softplus(-r).mean() + F.softplus(f).mean()

    @staticmethod
    def ralogistic_g(d_real, d_fake):
        r, f = GANLosses._ra_pairs(d_real, d_fake)
        #loss = -f.mean()
        return F.softplus(r).mean() + F.softplus(-f).mean()
    # Add to the GANLosses class (around line 1120, after the existing relativistic losses):
    
    @staticmethod
    def r3gan_d(d_real, d_fake):
        """
        R3GAN Discriminator Loss (RpGAN).
        Uses direct pairing: f(D(fake) - D(real)) where f(t) = -log(1 + e^(-t))
        This is equivalent to: BCE(D(fake) - D(real), 0)
        """
        diff = d_fake - d_real
        return F.softplus(diff).mean()
    
    @staticmethod
    def r3gan_g(d_fake, d_real=None):
        """
        R3GAN Generator Loss (RpGAN).
        Generator wants to maximize D(fake) - D(real)
        This is equivalent to: BCE(D(fake) - D(real), 1)
        Note: d_real parameter is optional for compatibility but required for R3GAN
        """
        if d_real is None:
            # Fallback if d_real not provided (shouldn't happen in R3GAN)
            return -d_fake.mean()
        diff = d_fake - d_real
        return F.softplus(-diff).mean()


def gradient_penalty(D, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    inter = alpha * real + (1 - alpha) * fake
    inter.requires_grad_(True)
    d_inter = D(inter)
    grads = torch.autograd.grad(outputs=d_inter, inputs=inter,
                                grad_outputs=torch.ones_like(d_inter),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
    grads = grads.view(grads.size(0), -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


# =====================
#  Config
# =====================
@dataclass
class TrainConfig:
    mode: str = 'train'
    dataset: str = ''
    image_size: int = 64
    channels: int = 3
    g_type: str = 'deconv'
    d_type: str = 'deconv'
    g_filters: int = 64
    d_filters: int = 64
    g_hidden_dim: int = 256
    d_hidden_dim: int = 256
    g_num_layers: int = 4
    d_num_layers: int = 4
    g_num_heads: int = 8
    d_num_heads: int = 8
    patch_size: int = 8
    fmap_max: int = 512
    attn_type: int = 0
    attn_res: List[int] = None
    attn_heads: List[int] = None
    residual_type: int = 0
    zdim: int = 128
    g_random: int = 0
    d_random: int = 0
    loss_type: int = 1
    activation: int = 3
    act_type: int = 0
    epochs: int = 1
    batch_size: int = 64
    step: int = 0
    epoch_done: int = 0
    relativistic: int = 0
    from_rgb_res: List[int] = None
    use_blur: int = 0
    overlap: int = 8
    r3gan_lazy_reg: int = 1  # Apply R1+R2 every N iterations (0 = every iteration)
    use_adaptive_norm: int = 0  # 0 = off, 1 = on (for R3GAN generator)
    use_hierarchy: int = 0  # 0 = off, 1 = on (for gMLP GAN)
    min_patch_size: int = 0  # Minimum patch size (for gMLP GAN)
    latent_mode: str = "per_patch"
    use_film: bool = True
    creps_thickness: int = 8

    def to_json(self):
        d = asdict(self)
        d['attn_res'] = self.attn_res or []
        d['attn_heads'] = self.attn_heads or []
        d['from_rgb_res'] = self.from_rgb_res or []
        return d


# =====================
#  CLI prompts
# =====================
def ask_choice(prompt: str, choices: List[str]) -> str:
    choices_lower = [c.lower() for c in choices]
    while True:
        ans = input(prompt).strip().lower()
        for i, c in enumerate(choices_lower):
            if ans == c:
                return choices[i]
        print(f"Invalid option. Choose one of: {', '.join(choices)}")


def ask_int(prompt: str, cond=lambda x: True, err="Invalid number.") -> int:
    while True:
        try:
            v = int(input(prompt).strip())
            if cond(v):
                return v
        except Exception:
            pass
        print(err)


def ask_str(prompt: str, cond=lambda s: True, err="Invalid input.") -> str:
    while True:
        s = input(prompt).strip()
        if cond(s):
            return s
        print(err)


def ask_attn_resolutions(img_size: int) -> List[int]:
    while True:
        txt = input("Attention resolutions (comma separated, powers of 2 between 8 and image size). Example: 8,16,32\n> ").strip()
        try:
            res = [int(x) for x in txt.split(',') if x.strip()]
            if not res:
                print("Provide at least one resolution.")
                continue
            ok = all(is_power_of_two(x) and 8 <= x <= img_size for x in res)
            if not ok:
                print("All resolutions must be powers of 2 between 8 and image size.")
                continue
            return sorted(list(set(res)))
        except Exception:
            print("Invalid format. Use comma separated integers.")


def ask_heads_for_resolutions(res_list: List[int]) -> List[int]:
    heads = []
    for r in res_list:
        h = ask_int(f"Multihead attention: number of heads at resolution {r}? ",
                    cond=lambda v: v > 0,
                    err="Heads must be a positive integer.")
        heads.append(h)
    return heads


# =====================
#  Model Building
# =====================
def build_models(cfg: TrainConfig, device: torch.device):
    attn_res_set = set(cfg.attn_res or [])
    residual = (cfg.residual_type == 1)
    heads_map = {res: (cfg.attn_heads or [])[i] for i, res in enumerate(cfg.attn_res or [])}
    
    # Build Generator
    if cfg.g_type == 'mlp':
        G = MLPGenerator(cfg.zdim, cfg.channels, cfg.g_hidden_dim, cfg.g_num_layers, 
                        cfg.image_size, cfg.patch_size).to(device)
    elif cfg.g_type == 'mlpgan':  # NEW: Basic MLP GAN
        G = MLPGANGenerator(cfg.zdim, cfg.channels, cfg.image_size,
                           hidden_dims=[cfg.g_hidden_dim] * cfg.g_num_layers,
                           activation='leakyrelu').to(device)
    elif cfg.g_type == 'r3gan':  # NEW: R3GAN architecture
#        # Use defaults from paper: base_channels scales with resolution
#        if cfg.image_size <= 32:
#            base_ch = 128
#        elif cfg.image_size <= 64:
#            base_ch = 96
#        elif cfg.image_size <= 128:
#            base_ch = 64
#        else:
#            base_ch = 48
        G = R3GANGenerator(cfg.zdim, cfg.channels, cfg.image_size,
                          base_channels=cfg.g_filters, max_channels=cfg.fmap_max, attn_type=cfg.attn_type, attn_res=attn_res_set, attn_heads_map=heads_map, use_adaptive_norm=getattr(cfg, 'use_adaptive_norm', False)).to(device)
    elif cfg.g_type == 'creps':
        G = CREPSGenerator(cfg.zdim, cfg.channels, cfg.g_hidden_dim, cfg.g_num_layers,
                          cfg.image_size, thickness=getattr(cfg, 'creps_thickness', 8)).to(device)
    elif cfg.g_type == 'vit':
        G = ViTGenerator(cfg.zdim, cfg.channels, cfg.g_hidden_dim, cfg.g_num_heads,
                        cfg.g_num_layers, cfg.image_size, cfg.patch_size, cfg.overlap).to(device)
    elif cfg.g_type == 'gru':
        G = GRUGenerator(cfg.zdim, cfg.channels, cfg.g_hidden_dim, cfg.g_num_layers,
                        cfg.image_size).to(device)
    elif cfg.g_type == 'swin':
        G = StyleSwinGenerator(zdim=cfg.zdim, img_ch=cfg.channels, g_last_hidden=cfg.g_filters,
                              fmap_max=cfg.fmap_max, img_size=cfg.image_size).to(device)
    elif cfg.g_type == 'lada':
        G = LadaGenerator(cfg.zdim, cfg.channels, cfg.image_size, 
                         num_heads=cfg.g_num_heads, mlp_dim=512).to(device)
    elif cfg.g_type == 'gmlp':
        G = gMLPGenerator(cfg.zdim, cfg.channels, cfg.g_hidden_dim, cfg.g_num_layers,
                         cfg.image_size, patch_h=cfg.patch_size, patch_w=cfg.overlap, overlap=0).to(device)
    elif cfg.g_type == 'mlpmixer':
        G = MLPMixerGenerator(cfg.channels, cfg.g_hidden_dim, cfg.g_num_layers,
                             cfg.image_size, cfg.patch_size, cfg.zdim,
                             overlap=cfg.overlap).to(device)
    elif cfg.g_type == 'patchmlp':  # NEW
        G = PatchMLPGenerator(cfg.zdim, cfg.channels, cfg.g_hidden_dim, cfg.g_num_layers,
                            cfg.image_size, cfg.patch_size, cfg.overlap, 4, 
                            False if cfg.use_hierarchy == 0 else True, cfg.min_patch_size, latent_proj_mode=cfg.latent_mode, use_pos_bias = cfg.use_film).to(device)
    else:  # deconv
        G = DeconvGenerator(zdim=cfg.zdim, img_ch=cfg.channels, g_last_hidden=cfg.g_filters, 
                           fmap_max=cfg.fmap_max, img_size=cfg.image_size, act_id=cfg.activation, 
                           act_type=cfg.act_type, residual=residual, attn_type=cfg.attn_type, 
                           attn_res=attn_res_set, add_noise=(cfg.g_random == 1), 
                           attn_heads_map=heads_map).to(device)
    
    # Build Discriminator
    if cfg.d_type == 'mlp':
        D = MLPDiscriminator(cfg.channels, cfg.d_hidden_dim, cfg.d_num_layers,
                            cfg.image_size, cfg.patch_size, overlap=cfg.overlap).to(device)
    elif cfg.d_type == 'mlpgan':  # NEW: Basic MLP GAN
        D = MLPGANDiscriminator(cfg.channels, cfg.image_size,
                               hidden_dims=[cfg.d_hidden_dim] * cfg.d_num_layers,
                               activation='leakyrelu',
                               use_spectral_norm=True).to(device)
    elif cfg.d_type == 'r3gan':  # NEW: R3GAN architecture
#        if cfg.image_size <= 32:
#            base_ch = 128
#        elif cfg.image_size <= 64:
#            base_ch = 96
#        elif cfg.image_size <= 128:
#            base_ch = 64
#        else:
#            base_ch = 48
        D = R3GANDiscriminator(cfg.channels, cfg.image_size,
                              base_channels=cfg.d_filters, max_channels=cfg.fmap_max, attn_type=cfg.attn_type, attn_res=attn_res_set, attn_heads_map=heads_map, from_rgb_res=getattr(cfg, 'from_rgb_res', None), use_blur=getattr(cfg, 'use_blur', False)).to(device)
    elif cfg.d_type == 'vit':
        D = ViTDiscriminator(cfg.channels, cfg.d_hidden_dim, cfg.d_num_heads,
                            cfg.d_num_layers, cfg.image_size, cfg.patch_size, cfg.overlap).to(device)
    elif cfg.d_type == 'gru':
        D = GRUDiscriminator(cfg.channels, cfg.d_hidden_dim, cfg.d_num_layers,
                            cfg.image_size).to(device)
    elif cfg.d_type == 'swin':
        D = SwinDiscriminator(img_ch=cfg.channels, d_first_hidden=cfg.d_filters,
                             fmap_max=cfg.fmap_max, img_size=cfg.image_size,
                             from_rgb_res=getattr(cfg, 'from_rgb_res', None),
                             use_blur=getattr(cfg, 'use_blur', False)).to(device)
    elif cfg.d_type == 'lada':
        D = LadaDiscriminator(cfg.channels, cfg.image_size, 
                             d_hidden=cfg.d_filters, num_heads=cfg.d_num_heads, 
                             mlp_dim=512).to(device)
    elif cfg.d_type == 'gmlp':
        D = gMLPDiscriminator(cfg.channels, cfg.d_hidden_dim, cfg.d_num_layers,
                             cfg.image_size, patch_h=cfg.patch_size, patch_w=cfg.overlap, overlap=0).to(device)
    elif cfg.d_type == 'patchmlp':  # NEW
        D = PatchMLPDiscriminator(cfg.channels, cfg.d_hidden_dim, cfg.d_num_layers,
                                 cfg.image_size, cfg.patch_size, cfg.overlap, 4,
                                 False if cfg.use_hierarchy == 0 else True, cfg.min_patch_size).to(device)
    else:  # deconv
        D = DeconvDiscriminator(img_ch=cfg.channels, d_first_hidden=cfg.d_filters, 
                               fmap_max=cfg.fmap_max, img_size=cfg.image_size, 
                               act_id=cfg.activation, act_type=cfg.act_type, residual=residual,
                               attn_type=cfg.attn_type, attn_res=attn_res_set, 
                               add_noise=(cfg.d_random == 1), attn_heads_map=heads_map,
                               from_rgb_res=getattr(cfg, 'from_rgb_res', None),
                               use_blur=getattr(cfg, 'use_blur', False)).to(device)
    
    G.apply(init_weights)
    D.apply(init_weights)
    return G, D


def sample_images(G: nn.Module, cfg: TrainConfig, step: int, device: torch.device, count: int = 16, prefix: str = "sample"):
    ensure_dir(GAN_DIR)
    with torch.no_grad():
        if cfg.g_random == 1:
            z = torch.zeros(count, cfg.zdim, device=device)
        else:
            z = torch.randn(count, cfg.zdim, device=device)
        imgs = G(z)
        grid = vutils.make_grid((imgs.clamp(-1, 1) + 1) / 2.0, nrow=min(8, int(math.sqrt(count))))
        vutils.save_image(grid, os.path.join(GAN_DIR, f"{prefix}_{step:07d}.png"))


def save_ckpt(G, D, cfg: TrainConfig):
    ensure_dir(GAN_DIR)
    torch.save(G.state_dict(), os.path.join(GAN_DIR, 'G.pth'))
    torch.save(D.state_dict(), os.path.join(GAN_DIR, 'D.pth'))
    save_config(cfg.to_json())


def load_ckpt(cfg: TrainConfig, device: torch.device):
    G, D = build_models(cfg, device)
    g_p = os.path.join(GAN_DIR, 'G.pth')
    d_p = os.path.join(GAN_DIR, 'D.pth')
    if not (os.path.isfile(g_p) and os.path.isfile(d_p) and os.path.isfile(os.path.join(GAN_DIR, 'GD.json'))):
        raise FileNotFoundError("Missing checkpoints or GD.json in ./GAN")
    G.load_state_dict(torch.load(g_p, map_location=device))
    D.load_state_dict(torch.load(d_p, map_location=device))
    return G, D

from lamb import Adan
def make_optimizers(G, D, cfg):
    gtype, dtype = cfg.g_type, cfg.d_type
    optG = Adan(G.parameters(), lr=1e-3, betas=(0.0, 0.5, 0.99))
    optD = Adan(D.parameters(), lr=1e-3, betas=(0.0, 0.5, 0.99))
    return optG, optD


import torch
from torch.nn.utils import clip_grad_norm_

def train_loop(cfg: TrainConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg.mode == 'continue':
        saved = load_config()
        cfg_loaded = TrainConfig(**{**saved})
        cfg_loaded.mode = 'train'
        cfg_loaded.dataset = cfg.dataset
        cfg_loaded.epochs = cfg.epochs
        cfg_loaded.batch_size = cfg.batch_size
        cfg = cfg_loaded
        G, D = load_ckpt(cfg, device)
    else:
        G, D = build_models(cfg, device)

    optG, optD = make_optimizers(G, D, cfg)

    loader, resolved_size, resolved_ch = make_loader(cfg.dataset, cfg.image_size, cfg.channels, cfg.batch_size)
    cfg.image_size, cfg.channels = resolved_size, resolved_ch

    loss_id = cfg.loss_type
    gp_lambda = ((cfg.image_size**2) / cfg.batch_size) * 0.05#((cfg.image_size / cfg.batch_size)**2) * 0.05#275#10.0
    #clip_value = 0.1  # gradient clipping threshold

    try:
        for epoch in range(cfg.epoch_done, cfg.epoch_done + cfg.epochs):
            for i, real in enumerate(loader):
                real = real.to(device)
                bs = real.size(0)

                # === Train D ===
                optD.zero_grad(set_to_none=True)
                with torch.no_grad():
                    z = torch.zeros(bs, cfg.zdim, device=device) if cfg.g_random == 1 else torch.randn(bs, cfg.zdim, device=device)
                    fake = G(z).detach()
                d_real = D(real)
                d_fake = D(fake)

                # Discriminator loss
                if loss_id == 7:  # R3GAN
                    d_loss = GANLosses.r3gan_d(d_real, d_fake)
                    #if cfg.r3gan_lazy_reg == 0 or cfg.step % cfg.r3gan_lazy_reg == 0:
                    # R3GAN REQUIRES both R1 and R2 for stability
                elif cfg.relativistic == 1:
                    if      loss_id == 0:  # DCGAN (BCE) -> RaSGAN
                        d_loss = GANLosses.rasgan_d(d_real, d_fake)
                    elif    loss_id == 1:  # LSGAN -> RaLSGAN
                        d_loss = GANLosses.ralsgan_d(d_real, d_fake)
                    elif    loss_id == 3:  # Logistic (softplus) -> Ra-Logistic
                        d_loss = GANLosses.ralogistic_d(d_real, d_fake)
                    elif    loss_id == 4:  # Hinge -> RaHinge
                        d_loss = GANLosses.rahinge_d(d_real, d_fake)
                    elif    loss_id == 5:  # LSGAN -> RaLSGAN
                        d_loss = GANLosses.rahbgan_d(d_real, d_fake)
                    else:
                        if loss_id == 2:
                            d_loss = GANLosses.wgan_d(d_real, d_fake)
                            gp = gradient_penalty(D, real, fake, device) * gp_lambda
                            d_loss = d_loss + gp
                        elif loss_id == 6:
                            d_loss = GANLosses.dagan_d(d_real, d_fake)
                        else:
                            raise ValueError('Unknown loss type')
                else:
                    if loss_id == 0:
                        d_loss = GANLosses.dcgan_d(d_real, d_fake)
                    elif loss_id == 1:
                        d_loss = GANLosses.lsgan_d(d_real, d_fake)
                    elif loss_id == 2:
                        d_loss = GANLosses.wgan_d(d_real, d_fake)
                        gp = gradient_penalty(D, real, fake, device) * gp_lambda
                        d_loss = d_loss + gp
                    elif loss_id == 3:
                        d_loss = GANLosses.logistic_d(d_real, d_fake)
                    elif loss_id == 4:
                        d_loss = GANLosses.hinge_d(d_real, d_fake)
                    elif loss_id == 5:
                        d_loss = GANLosses.hbgan_d(d_real, d_fake)
                    elif loss_id == 6:
                        d_loss = GANLosses.dagan_d(d_real, d_fake)
                    else:
                        raise ValueError('Unknown loss type')
                r1 = gradient_penalty_r1(D, real, device) * gp_lambda
                r2 = gradient_penalty_r2(D, fake, device) * gp_lambda
                #r3 = gradient_penalty_r3(D, real, fake, device) * gp_lambda
                d_loss = d_loss + r1 + r2# + r3

                d_loss.backward()
                #clip_grad_norm_(D.parameters(), max_norm=clip_value)  # <-- Clip D gradients
                optD.step()

                # === Train G ===
                optG.zero_grad(set_to_none=True)
                z = torch.zeros(bs, cfg.zdim, device=device) if cfg.g_random == 1 else torch.randn(bs, cfg.zdim, device=device)
                gen = G(z)
                d_fake_for_g = D(gen)
                d_real_det = d_real.detach()

                # Generator loss
                # For R3GAN, we need d_real for the generator too
                if loss_id == 7:  # R3GAN
                    d_real_for_g = d_real_det#D(real.detach())  # Detach to prevent backprop through D on real
                    g_loss = GANLosses.r3gan_g(d_fake_for_g, d_real_for_g)
                elif cfg.relativistic == 1:
                    if      loss_id == 0:
                        g_loss = GANLosses.rasgan_g(d_real_det, d_fake_for_g)
                    elif    loss_id == 1:
                        g_loss = GANLosses.ralsgan_g(d_real_det, d_fake_for_g)
                    elif    loss_id == 3:
                        g_loss = GANLosses.ralogistic_g(d_real_det, d_fake_for_g)
                    elif    loss_id == 4:
                        g_loss = GANLosses.rahinge_g(d_real_det, d_fake_for_g)
                    elif    loss_id == 5:
                        g_loss = GANLosses.rahbgan_g(d_real_det, d_fake_for_g)
                    else:
                        if loss_id == 2:
                            g_loss = GANLosses.wgan_g(d_fake_for_g)
                        elif loss_id == 6:
                            g_loss = GANLosses.dagan_g(d_fake_for_g)
                        else:
                            raise ValueError('Unknown loss type')
                else:
                    if loss_id == 0:
                        g_loss = GANLosses.dcgan_g(d_fake_for_g)
                    elif loss_id == 1:
                        g_loss = GANLosses.lsgan_g(d_fake_for_g)
                    elif loss_id == 2:
                        g_loss = GANLosses.wgan_g(d_fake_for_g)
                    elif loss_id == 3:
                        g_loss = GANLosses.logistic_g(d_fake_for_g)
                    elif loss_id == 4:
                        g_loss = GANLosses.hinge_g(d_fake_for_g)
                    elif loss_id == 5:
                        g_loss = GANLosses.hbgan_g(d_fake_for_g)
                    elif loss_id == 6:
                        g_loss = GANLosses.dagan_g(d_fake_for_g)
                    else:
                        raise ValueError('Unknown loss type')

                g_loss.backward()
                #clip_grad_norm_(G.parameters(), max_norm=clip_value)  # <-- Clip G gradients
                optG.step()

                cfg.step += 1

                if cfg.step % 50 == 0:
                    try:
                        check_and_manage_disk_space()
                        sample_images(G, cfg, cfg.step, device, count=9, prefix='tick')
                    except OSError as e:
                        if 'No space left on device' in str(e) or e.errno == 28:
                            print(f"\n[ERROR] No space left on device! Clearing images...")
                            clear_gan_images()
                            try:
                                sample_images(G, cfg, cfg.step, device, count=9, prefix='tick')
                            except:
                                print("[ERROR] Still no space. Skipping sample generation.")
                        else:
                            raise

                if cfg.step % 100 == 0:
                    cfg.epoch_done = epoch
                    save_ckpt(G, D, cfg)

                print(f"Epoch {epoch+1}/{cfg.epoch_done+cfg.epochs} | Step {cfg.step} | D: {d_loss.item():.4f} | G: {g_loss.item():.4f}")

            cfg.epoch_done = epoch + 1
            save_ckpt(G, D, cfg)

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Saving checkpoint and sampling...")
        save_ckpt(G, D, cfg)
        try:
            check_and_manage_disk_space()
            sample_images(G, cfg, cfg.step, device, count=16, prefix='interrupt')
        except OSError as e:
            if 'No space left on device' in str(e) or e.errno == 28:
                print(f"\n[ERROR] No space left on device! Clearing images...")
                clear_gan_images()
                try:
                    sample_images(G, cfg, cfg.step, device, count=16, prefix='interrupt')
                except:
                    print("[ERROR] Still no space. Skipping sample generation.")
            else:
                print(f"[ERROR] Could not save interrupt samples: {e}")
        print("Saved.")



def run_sample_mode(count: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg_d = load_config()
    cfg = TrainConfig(**cfg_d)
    clear_gan_images()
    G, _ = load_ckpt(cfg, device)
    print(G)
    with torch.no_grad():
        for i in range(count):
            z = torch.zeros(1, cfg.zdim, device=device) if cfg.g_random == 1 else torch.randn(1, cfg.zdim, device=device)
            img = G(z)
            vutils.save_image((img.clamp(-1, 1) + 1) / 2.0, os.path.join(GAN_DIR, f"sample_{i:06d}.png"))
    print(f"Generated {count} images in {GAN_DIR}/")


# =====================
#  Interactive main
# =====================
def main():
    print("Modes: train/t/0, continue/c/1, sample/s/2")
    mode = ask_choice(
        "> ",
        ["train", "t", "0", "continue", "c", "1", "sample", "s", "2"]
    )
    if mode in {"train", "t", "0"}:
        cfg = TrainConfig(mode='train')
        ds = ask_str("Dataset path or one of: CIFAR10, CIFAR100, MNIST, FashionMNIST\n> ")
        builtin = ds.strip().lower() in {"cifar10", "cifar100", "mnist", "fashionmnist"}
        cfg.dataset = ds

        if not builtin:
            img_size = ask_int("Image size (power of two, min 8): ",
                               cond=lambda v: v >= 8 and is_power_of_two(v),
                               err="Must be a power of two >= 8")
            ch = ask_int("Channels 1=grayscale, 3=RGB, 4=RGBA: ",
                         cond=lambda v: v in {1, 3, 4}, err="Must be 1, 3 or 4")
            cfg.image_size, cfg.channels = img_size, ch
        else:
            if ds.strip().lower() in {"cifar10", "cifar100"}:
                cfg.image_size, cfg.channels = 32, 3
            else:
                cfg.image_size, cfg.channels = 32, 1
            print(f"Using built-in defaults: size={cfg.image_size}, channels={cfg.channels}")

        # In main function, update the generator type prompt:
        print("\nGenerator type:")
        print("  CIPS-Style MLP(0)")
        print("  Basic Transposed Convnet(1)")
        print("  ViT-GAN G(2)")
        print("  Gated Recurrent Unit(3)")
        print("  StyleSwin(4)")
        print("  Lada(5)")
        print("  gMLP(6)")
        print("  MLP-Mixer(7)")
        print("  R3GAN Architecture(8)")
        print("  Basic MLP GAN(9)")
        print("  PatchMLP (Simplified gMLP)(10)")
        print("  CREPS Column-Row Generator(11)")
        
        cfg.g_type = ask_choice("G type> ", 
            ["mlp", "deconv", "vit", "gru", "swin", "lada", "gmlp", "mlpmixer", "r3gan", "mlpgan", "patchmlp", "creps",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]).lower()
        
        type_map = {
            "0": "mlp", "1": "deconv", "2": "vit", "3": "gru", "4": "swin",
            "5": "lada", "6": "gmlp", "7": "mlpmixer", "8": "r3gan", "9": "mlpgan", "10": "patchmlp", "11": "creps"
        }
        cfg.g_type = type_map.get(cfg.g_type, cfg.g_type)
        
        # Similarly for discriminator:
        print("\nDiscriminator type:")
        print("  MLP-Mixer(0)")
        print("  Basic DCGAN D(1)")
        print("  ViT-GAN D(2)")
        print("  Gated Recurrent Unit(3)")
        print("  Basic Swin(4)")
        print("  Lada(5)")
        print("  gMLP(6)")
        print("  R3GAN Architecture(7)")
        print("  Basic MLP GAN(8)")
        print("  PatchMLP (Simplified gMLP)(9)")  # NEW
        
        cfg.d_type = ask_choice("D type> ",
            ["mlp", "deconv", "vit", "gru", "swin", "lada", "gmlp", "r3gan", "mlpgan", "patchmlp",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]).lower()
        
        type_map_d = {
            "0": "mlp", "1": "deconv", "2": "vit", "3": "gru", "4": "swin",
            "5": "lada", "6": "gmlp", "7": "r3gan", "8": "mlpgan", "9": "patchmlp"
        }
        cfg.d_type = type_map_d.get(cfg.d_type, cfg.d_type)
        
        # Update the needs_patch check:
        needs_patch = cfg.g_type in {'mlp', 'vit', 'gmlp', 'mlpmixer', 'patchmlp'} or cfg.d_type in {'mlp', 'vit', 'gmlp', 'patchmlp'}

        if needs_patch:
            cfg.patch_size = ask_int("Patch size (must divide image size evenly): ",
                                    cond=lambda v: v > 0 and cfg.image_size % v == 0,
                                    err=f"Must be positive and divide {cfg.image_size}")
        if cfg.d_type in {'vit', 'mlp', 'gmlp', 'patchmlp'} or cfg.g_type in {'vit', 'gmlp', 'mlpmixer', 'patchmlp'}:
            cfg.overlap = ask_int("Patch overlap: ",
                                    cond=lambda v: v < cfg.image_size and v > -1,
                                    err=f"Must be positive and smaller than {cfg.patch_size}")
        if cfg.d_type in {'gmlp', 'patchmlp'} or cfg.g_type in {'gmlp', 'patchmlp'}:
            cfg.use_hierarchy = ask_int("Shall we use patch hierarchy (0 shuts it off, any other value enables it): ")
            cfg.min_patch_size = 8 if cfg.use_hierarchy == 0 else ask_int("Minimum patch size: ")
        if cfg.g_type in {'patchmlp'}:
            cfg.latent_mode = ask_int("Use per_patch (0, the default option, heavier) or shared (1, more lightweight, CIPS-like) or 2 (shared as well, but no FiLM))? >")
            if cfg.latent_mode == 1:
                cfg.latent_mode = "shared"
                cfg.use_film = False
            elif cfg.latent_mode == 2:
                cfg.latent_mode = "shared"
                cfg.use_film = True
            else:
                cfg.latent_mode = "per_patch"
            print(cfg.latent_mode)

        # Configure Generator parameters
        if cfg.g_type in ['deconv', 'swin', 'r3gan']:
            cfg.g_filters = ask_int("G filter count (last hidden channels): ", cond=lambda v: v > 0)
        else:
            cfg.g_hidden_dim = ask_int("G hidden dimension: ", cond=lambda v: v > 0)
            cfg.g_num_layers = ask_int("G number of layers: ", cond=lambda v: v > 0)
            if cfg.g_type == 'creps':
                cfg.creps_thickness = ask_int("CREPS thickness D (e.g. 8): ", cond=lambda v: v > 0)
            if cfg.g_type == 'vit' or cfg.g_type == "lada":
                cfg.g_num_heads = ask_int("G number of attention heads: ", cond=lambda v: v > 0)

        # Configure Discriminator parameters
        if cfg.d_type in ['deconv', 'swin', 'r3gan']:
            cfg.d_filters = ask_int("D filter count (first hidden channels): ", cond=lambda v: v > 0)
        
            
            # Ask for from_rgb resolutions (for both deconv and swin)
            print("\nStyleGAN2-style from_rgb skip connections:")
            print(f"Enter resolutions < {cfg.image_size} (comma-separated, powers of 2).")
            print("Example: 32,16,8  or  0 (for no from_rgb)")
            from_rgb_input = input("> ").strip()
            try:
                from_rgb_list = [int(x.strip()) for x in from_rgb_input.split(',') if x.strip()]
                # Filter valid resolutions (< image_size, >= 8, power of 2)
                valid_from_rgb = [r for r in from_rgb_list 
                                 if is_power_of_two(r) and 8 <= r < cfg.image_size]
                cfg.from_rgb_res = valid_from_rgb if valid_from_rgb else []
                if valid_from_rgb:
                    print(f"Using from_rgb at resolutions: {valid_from_rgb}")
                else:
                    print("No valid from_rgb resolutions. Using standard discriminator.")
            except:
                cfg.from_rgb_res = []
                print("Invalid input. Using standard discriminator.")
            
            # Ask for blur kernels (only if using from_rgb)
            if cfg.from_rgb_res:
                cfg.use_blur = ask_int("Use blur kernels for from_rgb? (0=no, 1=yes): ", 
                                      cond=lambda v: v in {0, 1})
            else:
                cfg.use_blur = 0
        else:
            # For non-deconv/non-swin discriminators, ask for hidden_dim and num_layers
            cfg.d_hidden_dim = ask_int("D hidden dimension: ", cond=lambda v: v > 0)
            cfg.d_num_layers = ask_int("D number of layers: ", cond=lambda v: v > 0)
            if cfg.d_type == 'vit' or cfg.d_type == "lada":
                cfg.d_num_heads = ask_int("D number of attention heads: ", cond=lambda v: v > 0)

        # Deconv/Swin-specific settings (fmap_max and optionally attention for deconv)
        if cfg.g_type in {'deconv', 'swin', 'r3gan'} or cfg.d_type in {'deconv', 'swin', 'r3gan'}:
            cfg.fmap_max = ask_int("Max feature maps (cap): ", cond=lambda v: v > 0)

            # Attention/residual/activation only for deconv (not for swin)
            if cfg.g_type in ['deconv', 'r3gan'] or cfg.d_type in ['deconv', 'r3gan']:
                cfg.attn_type = ask_int("Attention type (0 none, 1 basic, 2 multihead): ", cond=lambda v: v in {0, 1, 2})
                if cfg.attn_type != 0:
                    cfg.attn_res = ask_attn_resolutions(cfg.image_size)
                    if cfg.attn_type == 2:
                        cfg.attn_heads = ask_heads_for_resolutions(cfg.attn_res)
                    else:
                        cfg.attn_heads = []
                else:
                    cfg.attn_res = []
                    cfg.attn_heads = []

                cfg.residual_type = ask_int("Residual type (0 none, 1 addition): ", cond=lambda v: v in {0, 1})
            else:
                # Swin doesn't need these settings
                cfg.attn_res = []
                cfg.attn_heads = []
                cfg.residual_type = 0
        else:
            cfg.attn_res = []
            cfg.attn_heads = []
        if cfg.g_type == 'r3gan':
            cfg.use_adaptive_norm = ask_int(
                "Use adaptive normalization (z-modulated)? 0=no, 1=yes\n"
                "(Paper mentions this improves FID but wasn't tested)\n> ",
                cond=lambda v: v in {0, 1}
            )
        cfg.zdim = ask_int("Latent dim (>0): ", cond=lambda v: v > 0)

        cfg.g_random = ask_int("G_random (0 standard latent->image, 1 zero-latent + noise in blocks): ", cond=lambda v: v in {0, 1})
        cfg.d_random = ask_int("D_random (0 normal, 1 noise injection in blocks): ", cond=lambda v: v in {0, 1})

        print("Loss: 0=DCGAN, 1=LSGAN, 2=WGAN-GP, 3=Logistic, 4=Hinge, 5=HuberGAN, 6=DAGAN, 7=R3GAN")
        cfg.loss_type = ask_int("> ", cond=lambda v: v in {0, 1, 2, 3, 4, 5, 6, 7})
        if cfg.loss_type == 7:
            cfg.relativistic = 0  # R3GAN doesn't use the relativistic average approach
            print("Note: R3GAN uses RpGAN loss with R1+R2 gradient penalties (as per the paper)")
        else:
            cfg.relativistic = ask_int("Relativistic (Ra*)? 0=no, 1=yes\n> ", cond=lambda v: v in {0, 1})

        # Only ask for activation settings if using Deconv
        if cfg.g_type == 'deconv' or cfg.d_type == 'deconv':
            print("Activation (hidden): 0=Linear, 1=Sigmoid, 2=Tanh, 3=ReLU (D uses LeakyReLU), 4=Softsign, 5=SiLU, 6=PReLU, 7=GELU, 8=Sine, 9=SineEvenLReLU, 10=Smooth Maxout Unit")
            cfg.activation = ask_int("> ", cond=lambda v: v in {0,1,2,3,4,5,6,7,8,9,10})

            print("Activation type: 0=normal, 1=self-gated, 2=GLU, 3=ETAct")
            cfg.act_type = ask_int("> ", cond=lambda v: v in {0,1,2,3})
        else:
            # Default values for non-deconv models
            cfg.activation = 3  # ReLU
            cfg.act_type = 0  # normal

        cfg.epochs = ask_int("Epoch count: ", cond=lambda v: v > 0)
        cfg.batch_size = ask_int("Batch size: ", cond=lambda v: v > 0)

        # Ask about pre-resizing for custom datasets
        if not builtin:
            print("\nPre-resize dataset? This will create a 'resized' subfolder with all images resized.")
            preresize_choice = ask_choice("Pre-resize images? (yes/y/1 or no/n/0)\n> ", 
                                         ["yes", "y", "1", "no", "n", "0"])
            if preresize_choice in ["yes", "y", "1"]:
                print("Pre-resizing dataset...")
                cfg.dataset = preresize_dataset(cfg.dataset, cfg.image_size, cfg.channels)
                print(f"Dataset path updated to: {cfg.dataset}")

        save_config(cfg.to_json())
        clear_gan_images()
        train_loop(cfg)

    elif mode in {"continue", "c", "1"}:
        saved = load_config()
        cfg = TrainConfig(**saved)
        cfg.mode = 'continue'
        ds = ask_str("Dataset path or one of: CIFAR10, CIFAR100, MNIST, FashionMNIST\n> ")
        cfg.dataset = ds
        cfg.epochs = ask_int("Extra epochs to train: ", cond=lambda v: v > 0)
        cfg.batch_size = ask_int("Batch size: ", cond=lambda v: v > 0)
        
        # Ask about pre-resizing for custom datasets
        builtin = ds.strip().lower() in {"cifar10", "cifar100", "mnist", "fashionmnist"}
        if not builtin:
            print("\nPre-resize dataset? This will create a 'resized' subfolder with all images resized.")
            preresize_choice = ask_choice("Pre-resize images? (yes/y/1 or no/n/0)\n> ", 
                                         ["yes", "y", "1", "no", "n", "0"])
            if preresize_choice in ["yes", "y", "1"]:
                print("Pre-resizing dataset...")
                cfg.dataset = preresize_dataset(cfg.dataset, cfg.image_size, cfg.channels)
                print(f"Dataset path updated to: {cfg.dataset}")
        
        clear_gan_images()
        train_loop(cfg)

    else:  # sample
        cnt = ask_int("How many images to sample? ", cond=lambda v: v > 0)
        clear_gan_images()
        run_sample_mode(cnt)


if __name__ == '__main__':
    main()
