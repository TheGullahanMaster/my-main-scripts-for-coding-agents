"""Browser GUI for jit5.py: training setup, live training monitor, sampling and run history.

Start with `python jit5.py` and choose mode 5/g/gui, or run `python jit5_gui.py [port]`.
The server listens on 127.0.0.1 only and runs everything in this process.  Like the CLI it
trains into a checkpoint folder (JiTDiff_Flow by default: model.pt, best.pt, config.json and
numbered run*/ output folders).  Training runs on the main thread, so Ctrl+C in the terminal
still finishes the current step, saves and stops the run, exactly as in the CLI.
"""
from __future__ import annotations

import base64
import copy
import csv
import io
import json
import math
import os
import queue
import random
import sys
import threading
import time
import traceback
from pathlib import Path

import torch
from PIL import Image

import jit5 as jit

HTML = Path(__file__).with_name('jit5_gui.html')
IMAGE_SUFFIXES = {'.' + e for e in jit.IMAGE_EXTENSIONS}
MUON_FAMILY = {'muon', 'adamuon', 'normuon', 'adago', 'adamgo', 'rmsgo', 'adadeltago', 'muonhd', 'normuonhd'}
ADAPTIVE_LR = {'paper_adamhd', 'prodigy', 'radam_schedulefree'}
# Scalar optimizer hyperparameters editable on resume, with the CLI's bounds.
RESUME_BOUNDS = {'weight_decay': (0, None), 'momentum': (0, .999999), 'eps': (1e-16, None), 'alpha': (0, .999999),
                 'hyper_lr': (0, None), 'nu': (0, None), 'beta2': (0, .999999), 'anchor': (0, None)}
RESUME_EXTRA_BOUNDS = {'adadeltago': {'gamma': (1e-16, None), 'rho': (0, .999999)},
                       'rmsgo': {'gamma': (1e-16, None), 'v0': (1e-16, None)},
                       'adamgo': {'gamma': (1e-16, None), 'delta': (1e-16, None), 'min_step': (0, None)}}
TRAINING_SETTINGS = (('grad_clip', 0, float), ('sample_every', 1, int), ('save_every', 1, int),
                     ('sampling_steps', 1, int), ('preview_batch_size', 1, int), ('preview_seed', 0, int),
                     ('num_sample_images', 1, int))
TRAINING_FLAGS = ('use_spike_guard', 'use_flip', 'use_vflip', 'use_rot90')

# Defaults of the interactive new-run prompts (jit5.main), as one flat form.
FORM_DEFAULTS = {
    'dataset_path': './data', 'conditioning_mode': 'unconditional',
    'class_dropout_prob': 0.1, 'guidance_scale': 3.0, 'class_sampling': 'uniform',
    'cond_residual': False, 'pix2pix_sources': ['folders'], 'pix2pix_direction': 'a_to_b', 'synthetic_params': {},
    'channels': 3, 'resize_width': 64, 'resize_height': 64, 'crop_w': '64', 'crop_h': '64', 'crop_output': 'max',
    'model_type': 'jit', 'dim': 256, 'depth': None, 'heads': 4, 'hyper_heads': 1, 'axial': False,
    'conv_stem': 0, 'p_width': 8, 'p_height': 8, 'overlap_w': 0, 'overlap_h': 0,
    'stem_initial': 32, 'stem_max': 256, 'stem_activation': 4, 'stem_glu_scaling': 0, 'stem_skips': False,
    'stem_grn': False, 'stem_conditioning': 0, 'stem_width': None, 'stem_height': None,
    'use_adaln': True, 'use_2d_pos_emb': True, 'use_conv_mlp': False, 'self_cond': None, 'self_cond_prob': None,
    'use_qk_norm': True, 'use_final_adaln': True, 'use_grn': False, 'dropout': 0.0,
    'rin_num_latents': 256, 'rin_latent_dim': None, 'rin_layers_per_block': 4,
    'use_bottleneck': True, 'bottleneck_dim': 128, 'fcdm_mlp_ratio': 3.0, 'fcdm_dim': 32, 'fcdm_depth': 2,
    'conv_dim': 64, 'fmap_max': 512, 'bottleneck_res': 8,
    'hier_dim': 64, 'hier_fmap_max': 512, 'hier_depth': 2, 'hier_input_grid_size': 8, 'hier_output_grid_size': 4,
    'hier_global_mixer': 'mlpmixer', 'hier_global_dim': None, 'hier_global_depth': None, 'hier_global_heads': 4,
    'pred_mode': 'x', 'loss_mode': 'v', 't_mu': -0.8, 't_sigma': 0.8, 'noise_schedule': 'linear',
    'x_clip': 'none', 'lr_schedule': 'constant', 'time_scale': 1000.0, 'hat_group_depth': 2,
    'sampling_steps': 50, 'batch_size': 16,
    'optimizer_type': 'adan', 'lr': None, 'cautious': False, 'muon_all': False, 'muon_all_reshape': False,
    'muon_fallback_lr': None, 'muon_momentum': 0.95, 'muon_rank': 0, 'muon_weight_decay': None,
    'muon_backend': 'polar_express', 'muon_ns_steps': 5, 'muon_foreach': True, 'muon_ns_bfloat16': False,
    'adamuon_eps': 1e-8, 'adamuon_nesterov': False, 'normuon_beta2': .95, 'normuon_eps': 1e-8,
    'adago_gamma': 1., 'adago_v0': 1., 'adago_eps': 5e-4,
    'rmsgo_gamma': 1., 'rmsgo_v0': 1., 'rmsgo_eps': 5e-4, 'rmsgo_beta2': .99,
    'adadeltago_gamma': 1., 'adadeltago_rho': .9, 'adadeltago_eps': 1e-6,
    'adamgo_beta2': .999, 'adamgo_gamma': 1., 'adamgo_delta': 1e-8, 'adamgo_min_step': 0.,
    'hg_hyper_lr': .05, 'hg_normalize': True, 'hg_anchor': None, 'hg_max_lr': 0., 'hg_weight_decay': 0.,
    'hyper_lr': 1e-10, 'hd_min_lr': 1e-5, 'hd_max_lr': 1e-2,
    'modern_clion_nu': jit.MODERN_CLION_DEFAULT_NU, 'modern_clion_weight_decay': 0.0, 'clion_rescale_mask': False,
    'layerwise_nsgda_momentum': 0.9, 'layerwise_nsgda_cautious': True,
    'steps': 100000, 'use_ema': True, 'ema_warmup': 'smooth', 'ema_warmup_steps': 10000,
    'use_flip': True, 'use_vflip': False, 'use_rot90': False, 'use_amp': True, 'amp_dtype': 'fp16',
    'full_bf16': False, 'grad_clip': 1.0, 'use_spike_guard': True, 'use_grad_ckpt': False,
    'compile_mode': 'off', 'warmup_steps': 1000, 'seed': 42, 'sample_every': 250, 'save_every': 5000,
    'num_sample_images': 4, 'preview_batch_size': None, 'preview_seed': None,
    'grad_accum_steps': 1, 'validation_path': '', 'validation_percent': 0.0, 'validation_every': 250,
    'validation_batch_size': None, 'validation_max_batches': 0, 'validation_seed': 42,
    'save_best': True, 'best_training_loss': False, 'num_workers': 4, 'resize_cache': False,
}
FEATURE_KEYS = ('grad_accum_steps', 'validation_path', 'validation_percent', 'validation_every',
                'validation_batch_size', 'validation_max_batches', 'validation_seed', 'save_best', 'best_training_loss')


# ───────────────────────── helpers ─────────────────────────
def clean(obj):
    """NaN/inf -> None, tuples/sets -> lists, tensors -> lists, recursively (strict JSON)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [clean(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if torch.is_tensor(obj):
        return clean(obj.tolist())
    return obj


def data_url(image, fmt='PNG'):
    buf = io.BytesIO()
    image.save(buf, fmt)
    return f'data:image/{fmt.lower()};base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    return bool(value)


def num(form, key, cast=float, minimum=None, maximum=None, default=None):
    """form[key] (or default when blank) as a finite int/float within [minimum, maximum]."""
    raw = form.get(key)
    if raw in (None, ''):
        raw = default
    if raw is None:
        raise ValueError(f'{key} is required')
    try:
        if isinstance(raw, bool):
            raise TypeError
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{key} must be a number, not {raw!r}') from exc
    if not math.isfinite(value):
        raise ValueError(f'{key} must be finite')
    if cast is int:
        if value != int(value):
            raise ValueError(f'{key} must be an integer')
        value = int(value)
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f'{key} must be in [{minimum}, {"∞" if maximum is None else maximum}]')
    return value


def list_dir(path=None, kinds=('image', 'json', 'checkpoint')):
    path = Path(path or os.getcwd()).expanduser().resolve()
    if path.is_file():
        path = path.parent
    dirs, files = [], []
    for entry in sorted(path.iterdir(), key=lambda p: p.name.lower()):
        if entry.name.startswith('.'):
            continue
        if entry.is_dir():
            dirs.append(entry.name)
            continue
        suffix = entry.suffix.lower()
        kind = ('image' if suffix in IMAGE_SUFFIXES else 'json' if suffix == '.json'
                else 'checkpoint' if suffix == '.pt' else None)
        if kind in kinds:
            files.append({'name': entry.name, 'kind': kind, 'size': entry.stat().st_size})
    return {'path': str(path), 'parent': str(path.parent), 'dirs': dirs, 'files': files}


def read_image_file(path, max_side=None):
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
        raise ValueError(f'Not an image file: {path}')
    with Image.open(path) as image:
        image.load()
        if max_side:
            image.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        image.convert('RGB' if image.mode not in ('RGB', 'L') else image.mode).save(buf, 'PNG')
    return buf.getvalue()


def bf16_supported():
    return torch.cuda.is_available() and getattr(torch.cuda, 'is_bf16_supported', lambda: False)()


def options():
    family = lambda s: sorted(s)
    return clean({
        'models': [{'id': k, 'key': key, 'name': name} for k, (key, name) in
                   sorted(jit.MODEL_TYPES.items(), key=lambda x: int(x[0]))],
        'is_conv': family(jit.IS_CONV), 'is_hier': family(jit.IS_HIERARCHICAL),
        'needs_heads': family(jit.NEEDS_HEADS), 'hyper_heads': family(jit.HYPER_HEADS),
        'axial_models': family(jit.AXIAL_MODELS), 'grn_models': family(jit.GRN_TOKEN_MODELS),
        'optimizers': [{'id': k, 'key': key, 'name': name, 'lr': lr} for k, (key, name, lr) in
                       sorted(jit.OPTIMIZER_TYPES.items(), key=lambda x: int(x[0]))],
        'muon_family': sorted(MUON_FAMILY), 'go_family': sorted(jit.GO_FAMILY), 'muon_fallback_lr': jit.MUON_FALLBACK_LR,
        'stem_activations': [[k, v] for k, v in jit.STEM_ACTIVATIONS.items()],
        'stem_conditioning': [[k, v] for k, v in jit.STEM_CONDITIONING.items()],
        'synthetic_sources': [['folders', 'A/B folders'], ['combined', 'combined A|B images']] +
                             [[name, name] for _, name in sorted(jit.SYNTHETIC_SOURCE_IDS.items(), key=lambda x: int(x[0]))],
        'synthetic_defaults': jit.SyntheticPix2PixDataset.DEFAULT_PARAMS,
        'effect_modes': sorted(jit.SyntheticPix2PixDataset.EFFECT_MODES),
        'synthetic_params': [[mode, list(names)] for mode, names in SYNTHETIC_PARAMS],
        'ema_warmups': list(jit.EMA_WARMUP_MODES), 'compile_modes': list(jit.COMPILE_MODES),
        'defaults': FORM_DEFAULTS, 'save_dir': jit.SAVE_DIR, 'cwd': os.getcwd(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'bf16': bf16_supported(),
    })


# ───────────────────────── datasets ─────────────────────────
def inspect_dataset(path, mode='unconditional', samples=6):
    """What a dataset folder holds for the chosen format (counts, classes, A/B folders, native sizes)."""
    root = Path(path).expanduser()
    if not root.is_dir():
        raise ValueError(f'Not a folder: {root.resolve()}')
    images = jit.find_image_paths(str(root))
    out = {'path': str(root.resolve()), 'images': len(images)}
    sizes = {}
    for p in images[:40]:
        try:
            with Image.open(p) as im:
                sizes[f'{im.width}×{im.height}'] = sizes.get(f'{im.width}×{im.height}', 0) + 1
        except OSError:
            continue
    out['sizes'] = sorted(sizes.items(), key=lambda kv: -kv[1])[:5]
    out['classes'] = [{'name': c, 'count': len(jit.find_image_paths(str(root / c)))}
                      for c in jit.discover_class_names(str(root))]
    out['has_ab'] = (root / 'A').is_dir() and (root / 'B').is_dir()
    if out['has_ab']:
        out['a_count'] = len(jit.find_image_paths(str(root / 'A')))
        out['b_count'] = len(jit.find_image_paths(str(root / 'B')))
    out['resized_cache'] = (root / 'resized').is_dir()
    pick = random.Random(0).sample(images, min(samples, len(images))) if images else []
    out['samples'] = [str(p) for p in pick]
    warnings = []
    if mode == 'class' and not out['classes']:
        warnings.append('No class subfolders containing images.')
    if mode == 'pix2pix_folders' and not out['has_ab']:
        warnings.append('Folder Pix2Pix datasets need A and B subfolders.')
    if not images:
        warnings.append('No images found (jpg, jpeg, png, webp, bmp).')
    out['warnings'] = warnings
    return out


def synthetic_preview(form, count=6):
    """A few (source, target) pairs exactly as the synthetic Pix2Pix dataset produces them."""
    cfg, _ = config_from_form(form)
    if cfg['conditioning_mode'] != 'pix2pix':
        raise ValueError('Previews are for Pix2Pix datasets.')
    ds = jit.build_dataset(cfg)
    rng = random.Random(int(form.get('preview_nonce', 0) or 0))
    out = []
    for i in rng.sample(range(len(ds)), min(count, len(ds))):
        target, source = ds[i][0], ds[i][1]
        out.append({'source': data_url(jit.tensor_to_pil_image(source)), 'target': data_url(jit.tensor_to_pil_image(target))})
    return {'pairs': out, 'direction': cfg['pix2pix_direction']}


# ───────────────────────── config ─────────────────────────
def config_from_form(f):
    """The new-run prompts of jit5.main as one function: flat GUI form -> validated config."""
    f = {**FORM_DEFAULTS, **{k: v for k, v in f.items() if v is not None}}
    cfg = {'dataset_path': str(f['dataset_path']).strip()}
    if not cfg['dataset_path']:
        raise ValueError('Choose a dataset folder.')
    mode = f['conditioning_mode']
    if mode not in ('unconditional', 'class', 'pix2pix'):
        raise ValueError('Unknown dataset format.')
    cfg.update(conditioning_mode=mode, class_names=[], pix2pix_direction='a_to_b', pix2pix_source_mode='folders')
    if mode == 'class':
        cfg['class_names'] = jit.discover_class_names(cfg['dataset_path'])
        if not cfg['class_names']:
            raise ValueError('No class folders containing images found.')
        cfg['class_dropout_prob'] = num(f, 'class_dropout_prob', float, 0, 1)
        cfg['guidance_scale'] = num(f, 'guidance_scale', float, 0) if cfg['class_dropout_prob'] > 0 else 1.0
        cfg['class_sampling'] = f['class_sampling'] if f['class_sampling'] in ('uniform', 'natural') else 'uniform'
    elif mode == 'pix2pix':
        cfg['cond_residual'] = to_bool(f['cond_residual'])
        sources = f['pix2pix_sources']
        cfg['pix2pix_source_mode'] = jit.parse_synthetic_source(','.join(sources) if isinstance(sources, list) else sources)
        if cfg['pix2pix_source_mode'] not in {'folders', 'combined'}:
            active = set(cfg['pix2pix_source_mode'].split(','))
            if 'restoration' in active:
                active |= jit.SyntheticPix2PixDataset.EFFECT_MODES
            wanted = synthetic_param_keys(active)
            defaults = jit.SyntheticPix2PixDataset.DEFAULT_PARAMS
            given = f.get('synthetic_params') or {}
            cfg['synthetic_params'] = {k: num(given, k, float if isinstance(defaults[k], float) else int,
                                              default=defaults[k]) for k in wanted}
        else:
            cfg['synthetic_params'] = {}
        if cfg['pix2pix_source_mode'] == 'folders':
            if not (os.path.isdir(os.path.join(cfg['dataset_path'], 'A')) and os.path.isdir(os.path.join(cfg['dataset_path'], 'B'))):
                raise ValueError('Folder Pix2Pix datasets need A and B folders.')
        elif not jit.find_image_paths(cfg['dataset_path']):
            raise ValueError('No images found for this Pix2Pix source mode.')
        if cfg['pix2pix_source_mode'] in {'folders', 'combined'}:
            cfg['pix2pix_direction'] = f['pix2pix_direction'] if f['pix2pix_direction'] in ('a_to_b', 'b_to_a') else 'a_to_b'
    cfg['channels'] = num(f, 'channels', int)
    cfg['resize_width'] = num(f, 'resize_width', int)
    cfg['resize_height'] = num(f, 'resize_height', int)
    crop_w = jit.parse_size_range(str(f['crop_w']).strip() or cfg['resize_width'], 'crop width')
    crop_h = jit.parse_size_range(str(f['crop_h']).strip() or cfg['resize_height'], 'crop height')
    if crop_w[0] * crop_h[1] != crop_w[1] * crop_h[0]:
        raise ValueError('Crop ranges must preserve one aspect ratio.')
    cfg['crop_width_min'], cfg['crop_width_max'] = crop_w
    cfg['crop_height_min'], cfg['crop_height_max'] = crop_h
    varying = crop_w[0] != crop_w[1] or crop_h[0] != crop_h[1]
    use_min = varying and f['crop_output'] == 'min'
    cfg['crop_output_policy'] = 'min' if use_min else 'max'
    cfg['width'] = crop_w[0] if use_min else crop_w[1]
    cfg['height'] = crop_h[0] if use_min else crop_h[1]
    cfg['crop_width'], cfg['crop_height'] = cfg['width'], cfg['height']
    notes = []
    if (mode == 'pix2pix' and all(i in jit.SyntheticPix2PixDataset.MODES for i in cfg['pix2pix_source_mode'].split(','))
            and cfg['channels'] != 3):
        notes.append('Synthetic Pix2Pix modes use RGB effects; channel count forced to 3.')
        cfg['channels'] = 3

    mt = f['model_type']
    if mt not in {v[0] for v in jit.MODEL_TYPES.values()}:
        raise ValueError(f'Unknown model type {mt!r}')
    cfg['model_type'] = mt
    is_conv, is_hier = mt in jit.IS_CONV, mt in jit.IS_HIERARCHICAL

    def rounded(key, label):
        raw = num(f, key, int, 1)
        value = jit.make_divisible(raw, 8)
        if value != raw:
            notes.append(f'{label} rounded to {value}.')
        return value

    def stem(grid_default=None):
        cfg['stem_initial'] = num(f, 'stem_initial', int, 2)
        cfg['stem_max'] = num(f, 'stem_max', int, cfg['stem_initial'])
        cfg['stem_activation'] = num(f, 'stem_activation', int)
        if cfg['stem_activation'] not in jit.STEM_ACTIVATIONS:
            raise ValueError('Unknown stem activation.')
        cfg['stem_glu_scaling'] = num(f, 'stem_glu_scaling', int, 0, 1) if cfg['stem_activation'] >= 7 else 0
        cfg['stem_skips'] = to_bool(f['stem_skips']) if cfg['conv_stem'] == 3 else False
        cfg['stem_grn'] = to_bool(f['stem_grn'])
        cfg['stem_conditioning'] = num(f, 'stem_conditioning', int, 0, 5)
        jit.validate_stem_conditioning(cfg['stem_conditioning'], cfg['conv_stem'], cfg['stem_skips'])

    if mt == 'fcdm_unet':
        cfg['dim'] = num(f, 'fcdm_dim', int, 4)
        cfg['depth'] = num(f, 'fcdm_depth', int, 1)
        cfg['fcdm_mlp_ratio'] = num(f, 'fcdm_mlp_ratio', float, 1)
        cfg.update(conv_stem=0, stem_conditioning=0, stem_skips=False, stem_grn=False, use_adaln=True,
                   self_cond=False, self_cond_prob=0., p_width=1, p_height=1, heads=1, dropout=0.)
    elif not is_conv and not is_hier:
        cfg['conv_stem'] = num(f, 'conv_stem', int, 0, 3)
        if cfg['conv_stem'] != 3:
            cfg['p_width'] = num(f, 'p_width', int, 1)
            cfg['p_height'] = num(f, 'p_height', int, 1)
            cfg['overlap_w'] = num(f, 'overlap_w', int, 0)
            cfg['overlap_h'] = num(f, 'overlap_h', int, 0)
        else:
            cfg.update(p_width=1, p_height=1, overlap_w=0, overlap_h=0)
        if cfg['conv_stem']:
            stem()
            for axis, overlap_key in (('width', 'overlap_w'), ('height', 'overlap_h')):
                patch = cfg['p_' + axis]
                default_grid = ((cfg[axis] - patch) // max(1, patch - cfg[overlap_key]) + 1
                                if cfg['conv_stem'] != 3 else min(8, cfg[axis]))
                cfg['stem_' + axis] = num(f, 'stem_' + axis, int, 1, cfg[axis], default=max(1, default_grid))
        cfg['dim'] = rounded('dim', 'Model width')
        cfg['depth'] = num(f, 'depth', int, 1, default=6 if mt == 'rin' else 4)
        cfg['heads'] = 1
        if mt in jit.NEEDS_HEADS:
            cfg['heads'] = num(f, 'heads', int, 1)
        elif mt in jit.HYPER_HEADS:
            cfg['heads'] = num(f, 'hyper_heads', int, 1)
        cfg['axial'] = to_bool(f['axial']) if mt in jit.AXIAL_MODELS else False
        is_rin, is_fcdm = mt == 'rin', mt == 'fcdm_isotropic'
        cfg['use_adaln'] = True if is_fcdm else False if is_rin else to_bool(f['use_adaln'])
        cfg['use_2d_pos_emb'] = False if is_rin or is_fcdm else to_bool(f['use_2d_pos_emb'])
        cfg['use_conv_mlp'] = False if is_rin or is_fcdm else to_bool(f['use_conv_mlp'])
        cfg['self_cond'] = to_bool(is_rin if f['self_cond'] is None else f['self_cond'])
        prob = num(f, 'self_cond_prob', float, default=0.9 if is_rin else 0.5) if cfg['self_cond'] else 0.0
        cfg['self_cond_prob'] = min(max(prob, 0.0), 1.0)
        cfg['use_qk_norm'] = False if is_rin or is_fcdm else to_bool(f['use_qk_norm'])
        cfg['use_final_adaln'] = True if is_fcdm else False if is_rin else to_bool(f['use_final_adaln'])
        cfg['use_grn'] = to_bool(f['use_grn']) if mt in jit.GRN_TOKEN_MODELS else False
        cfg['dropout'] = 0.0 if is_fcdm else num(f, 'dropout', float, 0, 1)
        if is_rin:
            cfg['rin_num_latents'] = num(f, 'rin_num_latents', int, 3 if mode == 'class' else 2)
            cfg['rin_latent_dim'] = num(f, 'rin_latent_dim', int, 16, default=2 * cfg['dim'])
            cfg['rin_layers_per_block'] = num(f, 'rin_layers_per_block', int, 1)
        else:
            cfg['rin_num_latents'] = 256
        use_bn = to_bool(f['use_bottleneck']) if not is_rin and not cfg['conv_stem'] & 1 else False
        cfg['bottleneck_dim'] = num(f, 'bottleneck_dim', int, 1) if use_bn else None
        if is_fcdm:
            cfg['fcdm_mlp_ratio'] = num(f, 'fcdm_mlp_ratio', float, 1)
            cfg['use_adaln'] = cfg['use_final_adaln'] = True
        cfg['fmap_max'] = 0
        cfg['bottleneck_res'] = 0
    elif is_conv:
        cfg['dim'] = rounded('conv_dim', 'Initial filter count')
        cfg['fmap_max'] = num(f, 'fmap_max', int, 1)
        cfg['bottleneck_res'] = num(f, 'bottleneck_res', int, 1)
        for k in ['p_width', 'p_height', 'depth', 'heads']:
            cfg[k] = 0
        for k in ['use_adaln', 'use_2d_pos_emb', 'use_conv_mlp', 'use_grn', 'self_cond', 'axial',
                  'use_qk_norm', 'use_final_adaln']:
            cfg[k] = False
        cfg.update(self_cond_prob=0.0, bottleneck_dim=None, overlap_h=0, overlap_w=0, dropout=0.0)
    else:
        cfg['dim'] = rounded('hier_dim', 'Initial feature width')
        cfg['fmap_max'] = num(f, 'hier_fmap_max', int, 1)
        cfg['depth'] = num(f, 'hier_depth', int, 1)
        cfg['conv_stem'] = num(f, 'conv_stem', int, 0, 3)
        if cfg['conv_stem']:
            stem()
        cfg['hier_input_grid_size'] = num(f, 'hier_input_grid_size', int, 1)
        cfg['hier_output_grid_size'] = num(f, 'hier_output_grid_size', int, 1)
        input_grid, output_grid = cfg['hier_input_grid_size'], cfg['hier_output_grid_size']
        scale_h, scale_w = cfg['height'] // output_grid, cfg['width'] // output_grid
        valid = scale_h > 0 and scale_w > 0 and scale_h == scale_w and scale_h & (scale_h - 1) == 0
        if (input_grid < output_grid or input_grid % output_grid or cfg['height'] % input_grid
                or cfg['width'] % input_grid or not valid):
            raise ValueError('HierMLP needs an input grid divisible by the root grid; the root grid must reach '
                             'both image dimensions through equal 2×2 refinements.')
        cfg['initial_grid_size'] = output_grid
        cfg['hier_global_mixer'] = f['hier_global_mixer'] if f['hier_global_mixer'] in (
            'jit', 'mlpmixer', 'gmlp', 'convnext', 'vip') else 'mlpmixer'
        raw = num(f, 'hier_global_dim', int, 1, default=cfg['dim'])
        cfg['hier_global_dim'] = jit.make_divisible(raw, 8)
        if cfg['hier_global_dim'] != raw:
            notes.append(f'Global processor width rounded to {cfg["hier_global_dim"]}.')
        cfg['hier_global_depth'] = num(f, 'hier_global_depth', int, 1, default=cfg['depth'])
        cfg['hier_global_heads'] = num(f, 'hier_global_heads', int, 1) if cfg['hier_global_mixer'] == 'jit' else 4
        for k in ['p_width', 'p_height', 'heads']:
            cfg[k] = 0
        for k in ['use_adaln', 'use_2d_pos_emb', 'use_conv_mlp', 'use_grn', 'self_cond', 'axial',
                  'use_qk_norm', 'use_final_adaln']:
            cfg[k] = False
        cfg.update(self_cond_prob=0.0, bottleneck_dim=None, bottleneck_res=0, overlap_h=0, overlap_w=0,
                   dropout=0.0, rin_num_latents=64)

    for key in ('pred_mode', 'loss_mode'):
        cfg[key] = f[key] if f[key] in ('x', 'eps', 'v') else FORM_DEFAULTS[key]
    cfg['t_mu'] = num(f, 't_mu', float)
    cfg['t_sigma'] = num(f, 't_sigma', float)
    cfg['noise_schedule'] = f['noise_schedule'] if f['noise_schedule'] in ('linear', 'rin_sigmoid') else 'linear'
    cfg['x_clip'] = f['x_clip'] if f['x_clip'] in ('none', 'static', 'dynamic') else 'none'
    cfg['lr_schedule'] = f['lr_schedule'] if f['lr_schedule'] in ('constant', 'cosine') else 'constant'
    cfg['time_scale'] = num(f, 'time_scale', float)
    cfg.update(bottleneck_act='none', attn_double_norm=False, arch_version=2)
    if mt == 'hat':
        cfg['hat_group_depth'] = num(f, 'hat_group_depth', int, 1)
    cfg['sampling_steps'] = num(f, 'sampling_steps', int, 1)
    cfg['batch_size'] = num(f, 'batch_size', int, 1)

    opt = f['optimizer_type']
    match = [v for v in jit.OPTIMIZER_TYPES.values() if v[0] == opt]
    if not match:
        raise ValueError(f'Unknown optimizer {opt!r}')
    cfg['optimizer_type'] = opt
    cfg['lr'] = num(f, 'lr', float, default=match[0][2])
    if opt in MUON_FAMILY:
        cfg['cautious'] = to_bool(f['cautious'])
        cfg['muon_all'] = to_bool(f['muon_all'])
        cfg['muon_all_reshape'] = to_bool(f['muon_all_reshape'])
        if cfg['muon_all'] or cfg['lr'] <= 0:
            cfg['muon_adam_lr_ratio'] = 1.0
        else:
            fallback = num(f, 'muon_fallback_lr', float, 1e-12,
                           default=jit.MUON_FALLBACK_LR if opt in jit.GO_FAMILY else cfg['lr'])
            cfg['muon_adam_lr_ratio'] = fallback / cfg['lr']
        cfg['muon_momentum'] = num(f, 'muon_momentum', float)
        cfg['muon_rank'] = num(f, 'muon_rank', int)
        cfg['muon_weight_decay'] = num(f, 'muon_weight_decay', float, default=0. if opt in jit.GO_FAMILY else 0.1)
        cfg['muon_backend'] = 'newton_schulz' if f['muon_backend'] == 'newton_schulz' else 'polar_express'
        cfg['muon_ns_steps'] = num(f, 'muon_ns_steps', int, 1)
        cfg['muon_foreach'] = to_bool(f['muon_foreach'])
        cfg['muon_ns_bfloat16'] = to_bool(f['muon_ns_bfloat16'])
        if opt == 'adamuon':
            cfg['adamuon_eps'] = num(f, 'adamuon_eps', float, 1e-16)
            cfg['adamuon_nesterov'] = to_bool(f['adamuon_nesterov'])
        if opt in ('normuon', 'normuonhd'):
            cfg['normuon_beta2'] = num(f, 'normuon_beta2', float, 0, .999999)
            cfg['normuon_eps'] = num(f, 'normuon_eps', float, 1e-16)
        for prefix, keys in (('adago', ('gamma', 'v0', 'eps')), ('rmsgo', ('gamma', 'v0', 'eps', 'beta2')),
                             ('adadeltago', ('gamma', 'rho', 'eps')), ('adamgo', ('beta2', 'gamma', 'delta', 'min_step'))):
            if opt == prefix:
                for k in keys:
                    lo = 0 if k in ('beta2', 'rho', 'min_step') else 1e-16
                    hi = .999999 if k in ('beta2', 'rho') else None
                    cfg[f'{prefix}_{k}'] = num(f, f'{prefix}_{k}', float, lo, hi)
    if opt in ('adamhd', 'muonhd', 'normuonhd'):
        cfg['hg_hyper_lr'] = num(f, 'hg_hyper_lr', float, 0)
        cfg['hg_normalize'] = to_bool(f['hg_normalize'])
        cfg['hg_anchor'] = num(f, 'hg_anchor', float, 0, default=1. if cfg['hg_normalize'] else .02)
        cfg['hg_max_lr'] = num(f, 'hg_max_lr', float, 0)
        if cfg['lr'] <= 0 and cfg['hg_max_lr'] <= 0:
            raise ValueError('A zero initial learning rate needs a maximum LR to set the scale.')
        if opt == 'adamhd':
            cfg['hg_weight_decay'] = num(f, 'hg_weight_decay', float, 0)
    if opt == 'paper_adamhd':
        for k in ('hyper_lr', 'hd_min_lr', 'hd_max_lr'):
            cfg[k] = num(f, k, float)
    if opt == 'modern_clion':
        cfg['modern_clion_nu'] = num(f, 'modern_clion_nu', float)
        cfg['modern_clion_weight_decay'] = num(f, 'modern_clion_weight_decay', float)
    if opt == 'clion':
        cfg['clion_rescale_mask'] = to_bool(f['clion_rescale_mask'])
    if opt == 'layerwise_nsgda':
        cfg['layerwise_nsgda_momentum'] = num(f, 'layerwise_nsgda_momentum', float)
        cfg['layerwise_nsgda_cautious'] = to_bool(f['layerwise_nsgda_cautious'])

    cfg['steps'] = num(f, 'steps', int, 0)
    cfg['use_ema'] = to_bool(f['use_ema'])
    if cfg['use_ema']:
        cfg['ema_warmup'] = f['ema_warmup'] if f['ema_warmup'] in jit.EMA_WARMUP_MODES else 'smooth'
        if cfg['ema_warmup'] == 'smooth':
            cfg['ema_warmup_steps'] = num(f, 'ema_warmup_steps', int, 1)
    for k in ('use_flip', 'use_vflip', 'use_rot90', 'use_amp', 'full_bf16', 'use_spike_guard', 'use_grad_ckpt'):
        cfg[k] = to_bool(f[k])
    cfg['amp_dtype'] = f['amp_dtype'] if cfg['use_amp'] and f['amp_dtype'] in ('fp16', 'bf16') else 'fp16'
    if cfg['full_bf16']:
        cfg['use_amp'] = False
    cfg['grad_clip'] = num(f, 'grad_clip', float)
    cfg['compile_mode'] = f['compile_mode'] if f['compile_mode'] in jit.COMPILE_MODES else 'off'
    cfg['warmup_steps'] = num(f, 'warmup_steps', int)
    cfg['seed'] = num(f, 'seed', int)
    cfg['sample_every'] = num(f, 'sample_every', int, 1)
    cfg['save_every'] = num(f, 'save_every', int, 1)
    cfg['num_sample_images'] = num(f, 'num_sample_images', int, 1)
    cfg['preview_batch_size'] = num(f, 'preview_batch_size', int, 1, default=cfg['batch_size'])
    cfg['preview_seed'] = num(f, 'preview_seed', int, default=cfg['seed'] or 42)
    cfg['num_workers'] = num(f, 'num_workers', int, 0)
    jit.apply_config_defaults(cfg)
    apply_features(cfg, f)
    jit.validate_config(cfg)
    jit.validate_fresh_run(cfg)
    return cfg, notes


# Effect -> the controls jit5.prompt_synthetic_params asks for.
SYNTHETIC_PARAMS = (('autocanny', ('autocanny_sigma',)), ('superresolution', ('superres_min_size', 'superres_max_size')),
                    ('deblur', ('blur_min_radius', 'blur_max_radius')), ('noise', ('noise_min_sigma', 'noise_max_sigma')),
                    ('affine', ('affine_shear', 'affine_translate_fraction')), ('hue', ('hue_max_shift',)),
                    ('sharpen', ('sharpen_min_radius', 'sharpen_max_radius', 'sharpen_min_percent', 'sharpen_max_percent')),
                    ('quantize', ('quantize_min_colors', 'quantize_max_colors')),
                    ('glitch', ('glitch_min_bands', 'glitch_max_bands', 'glitch_max_band_fraction', 'glitch_max_offset_fraction')),
                    ('film', ('film_min_marks', 'film_max_marks')),
                    ('solarize', ('solarize_min_threshold', 'solarize_max_threshold')),
                    ('inpaint', ('cutout_min_fraction', 'cutout_max_fraction')),
                    ('outpaint', ('cutout_min_fraction', 'cutout_max_fraction')),
                    ('restoration', ('restoration_probability',)))


def synthetic_param_keys(active):
    """The synthetic-effect controls jit5.prompt_synthetic_params asks for, in its order."""
    keys = []
    for mode, names in SYNTHETIC_PARAMS:
        if mode in active:
            keys.extend(n for n in names if n not in keys)
    return keys


def apply_features(cfg, f):
    """jit5.prompt_training_features from a form (validation, accumulation, best weights)."""
    f = {k: v for k, v in f.items() if v not in (None,)}
    cfg['grad_accum_steps'] = num(f, 'grad_accum_steps', int, 1, default=cfg['grad_accum_steps'])
    path = str(f.get('validation_path', cfg['validation_path']) or '').strip()
    cfg['validation_path'] = '' if path == '-' else path
    cfg['validation_percent'] = (0.0 if cfg['validation_path'] else
                                 num(f, 'validation_percent', float, 0, 99, default=cfg['validation_percent']))
    if cfg['validation_path'] or cfg['validation_percent']:
        cfg['validation_every'] = num(f, 'validation_every', int, 1, default=cfg['validation_every'])
        cfg['validation_batch_size'] = num(f, 'validation_batch_size', int, 1, default=cfg['validation_batch_size'])
        cfg['validation_max_batches'] = num(f, 'validation_max_batches', int, 0, default=cfg['validation_max_batches'])
        cfg['validation_seed'] = num(f, 'validation_seed', int, 0, default=cfg['validation_seed'])
    cfg['save_best'] = to_bool(f.get('save_best', cfg['save_best']))
    if not (cfg['validation_path'] or cfg['validation_percent']):
        cfg['best_training_loss'] = to_bool(f.get('best_training_loss', cfg['best_training_loss']))


def config_from_json(text_or_obj):
    """A prepared or retried configuration, exactly as the CLI would use it."""
    cfg = json.loads(text_or_obj) if isinstance(text_or_obj, str) else copy.deepcopy(text_or_obj)
    if not isinstance(cfg, dict):
        raise ValueError('Configuration must be a JSON object')
    jit.apply_config_defaults(cfg)
    jit.validate_config(cfg)
    jit.validate_fresh_run(cfg)
    return cfg


def model_summary(cfg):
    """Parameter count and token geometry, without allocating weights."""
    cfg = copy.deepcopy(cfg)
    jit.apply_config_defaults(cfg)
    try:
        with torch.device('meta'):
            model = jit.build_model(cfg, 'meta')
    except Exception:
        model = jit.build_model(cfg, 'cpu')
    params = sum(p.numel() for p in model.parameters())
    out = {'params': params, 'model': cfg['model_type'], 'size': f"{cfg['width']}×{cfg['height']}"}
    mt = cfg['model_type']
    if mt not in jit.IS_CONV and mt not in jit.IS_HIERARCHICAL and mt != 'fcdm_unet' and cfg.get('conv_stem', 0) != 3:
        gh = (cfg['height'] - cfg['p_height']) // max(1, cfg['p_height'] - cfg['overlap_h']) + 1
        gw = (cfg['width'] - cfg['p_width']) // max(1, cfg['p_width'] - cfg['overlap_w']) + 1
        out['tokens'] = f'{gw}×{gh} = {gw * gh}'
    elif cfg.get('conv_stem', 0) == 3 and mt not in jit.IS_HIERARCHICAL:
        out['tokens'] = f"{cfg['stem_width']}×{cfg['stem_height']} = {cfg['stem_width'] * cfg['stem_height']}"
    return out


def check_config(spec):
    if spec.get('config') is not None:
        cfg, notes = config_from_json(spec['config']), []
    else:
        cfg, notes = config_from_form(spec.get('form') or {})
    return clean({'config': cfg, 'notes': notes, 'summary': model_summary(cfg)})


def previous_config(checkpoint_dir):
    """The config retry mode starts from (config.json, else the legacy config.pt)."""
    root = Path(checkpoint_dir or jit.SAVE_DIR)
    for name in ('config.json', 'config.pt'):
        if (root / name).is_file():
            cfg = jit.load_config(str(root / name))
            jit.apply_config_defaults(cfg)
            return clean({'path': str((root / name).resolve()), 'config': cfg})
    raise ValueError(f'No config.json or config.pt in {root.resolve()}')


def load_config_file(path):
    cfg = jit.load_config(str(Path(path).expanduser()))
    return clean({'path': str(Path(path).expanduser().resolve()), 'config': cfg})


def save_config_file(path, config):
    target = Path(path).expanduser().resolve()
    if target.suffix.lower() != '.json':
        raise ValueError('Config file name must end in .json')
    target.parent.mkdir(parents=True, exist_ok=True)
    jit.save_config(config, str(target))
    return {'path': str(target)}


def checkpoint_info(path):
    """Everything the continue / fine-tune / sample forms need from one checkpoint."""
    path = Path(path).expanduser()
    if path.is_dir():
        path = path / 'model.pt'
    if not path.is_file():
        raise ValueError(f'No checkpoint at {path.resolve()}')
    ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
    cfg = copy.deepcopy(ckpt.get('config') or jit.load_config(str(path.parent / 'config.json')))
    legacy_class_sampling = cfg.get('conditioning_mode') == 'class' and 'class_sampling' not in cfg
    jit.apply_config_defaults(cfg)
    completed = ckpt.get('completed_steps', ckpt.get('step', -1) + 1) if 'optimizer' in ckpt else None
    group = {}
    if isinstance(ckpt.get('optimizer'), dict) and ckpt['optimizer'].get('param_groups'):
        group = {k: v for k, v in ckpt['optimizer']['param_groups'][0].items()
                 if k != 'params' and (isinstance(v, (int, float, str, bool)) or k == 'betas')}
    editable = dict(RESUME_BOUNDS, **RESUME_EXTRA_BOUNDS.get(cfg.get('optimizer_type'), {}))
    resume_lr = cfg['lr']
    if completed is not None and group:
        adaptive = cfg['optimizer_type'] in ADAPTIVE_LR
        cosine = cfg['lr_schedule'] == 'cosine' and not adaptive and completed >= cfg['warmup_steps']
        if adaptive or cosine:
            resume_lr = group.get('lr', cfg['lr'])
    info = {'path': str(path.resolve()), 'config': cfg, 'completed_steps': completed,
            'kind': 'training checkpoint' if completed is not None else 'weights only (best.pt)',
            'has_ema': 'ema' in ckpt, 'weights': [k for k in ('ema', 'model') if k in ckpt],
            'best': ckpt.get('best'), 'legacy_class_sampling': legacy_class_sampling,
            'optimizer_group': {k: v for k, v in group.items() if k in editable or k in ('betas', 'lr')},
            'optimizer_bounds': {k: list(v) for k, v in editable.items() if k in group},
            'resume_lr': resume_lr, 'size_mb': path.stat().st_size / 2 ** 20}
    return clean(info)


def checkpoint_modules(path):
    """Module names with parameters (for finetune_policy='modules')."""
    info = checkpoint_info(path)
    cfg = info['config']
    try:
        with torch.device('meta'):
            model = jit.build_model(cfg, 'meta')
    except Exception:
        model = jit.build_model(cfg, 'cpu')
    rows = [{'name': name, 'params': sum(p.numel() for p in module.parameters()),
             'type': type(module).__name__, 'depth': name.count('.')}
            for name, module in model.named_modules() if name and any(True for _ in module.parameters())]
    return {'modules': rows}


# ───────────────────────── training ─────────────────────────
def bucket_mean(rows, max_points):
    """At most max_points rows: each bucket keeps its last x and the mean of its other columns."""
    if len(rows) <= max_points:
        return [list(r) for r in rows]
    k = -(-len(rows) // max_points)
    out = []
    for i in range(0, len(rows), k):
        chunk = rows[i:i + k]
        row = [chunk[-1][0]]
        for j in range(1, len(chunk[0])):
            values = [r[j] for r in chunk if r[j] is not None and math.isfinite(r[j])]
            row.append(sum(values) / len(values) if values else None)
        out.append(row)
    return out


class Trainer:
    """One training run at a time, executed on the main thread (see run_gui)."""
    MAX_LOG = 2000

    def __init__(self, jobs):
        self.jobs, self.lock = jobs, threading.Lock()
        self.reset()

    def reset(self):
        self.state, self.error, self.detail, self.stop_requested = 'idle', None, '', False
        self.points = []   # (step, loss, loss_ema, lr, grad_norm)
        self.val = []      # (step, validation loss)
        self.logs, self.previews, self.info = [], [], {}
        self.started = self.finished = None
        self.run_dir = None
        self.completed = self.total = 0
        self.skips = 0
        self.best = None

    def status(self, since=0, log_since=0, max_points=4000):
        with self.lock:
            since = max(0, min(int(since or 0), len(self.points)))
            tail = bucket_mean(self.points[since:], max_points)
            end = self.finished or time.time()
            log_since = max(0, min(int(log_since or 0), len(self.logs)))
            return clean({
                'state': self.state, 'error': self.error, 'detail': self.detail, 'total_points': len(self.points),
                'points': tail, 'val': list(self.val), 'logs': self.logs[log_since:], 'log_total': len(self.logs),
                'previews': list(self.previews), 'info': self.info, 'run_dir': self.run_dir,
                'completed': self.completed, 'total': self.total, 'skips': self.skips, 'best': self.best,
                'elapsed': end - self.started if self.started else 0, 'stop_requested': self.stop_requested})

    def stop(self):
        with self.lock:
            if self.state in ('preparing', 'training'):
                self.stop_requested, self.state = True, 'stopping'
        return self.status(10 ** 12, 10 ** 12)

    def _set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def _log(self, message, kind='log'):
        with self.lock:
            self.logs.append({'t': time.time(), 'kind': kind, 'message': str(message)})
            del self.logs[:-self.MAX_LOG]

    def start(self, spec):
        with self.lock:
            if self.state in ('queued', 'preparing', 'training', 'stopping'):
                raise RuntimeError('A training run is already active.')
        job = self._prepare(spec)  # quick validation errors reach the page immediately
        with self.lock:
            self.reset()
            self.state, self.started, self.detail = 'queued', time.time(), 'waiting for the main thread'
        self.jobs.put(lambda: self._run(job))
        return self.status()

    @staticmethod
    def _prepare(spec):
        mode = spec.get('mode', 'new')
        checkpoint_dir = str(Path(spec.get('checkpoint_dir') or jit.SAVE_DIR).expanduser())
        job = {'mode': mode, 'checkpoint_dir': checkpoint_dir, 'resize_cache': False}
        if mode == 'new':
            if spec.get('config') is not None:
                job['cfg'] = config_from_json(spec['config'])
            else:
                job['cfg'], _ = config_from_form(spec.get('form') or {})
                job['resize_cache'] = to_bool((spec.get('form') or {}).get('resize_cache', False))
        elif mode == 'continue':
            path = Path(checkpoint_dir) / 'model.pt'
            if not path.is_file():
                raise ValueError(f'Cannot resume: {path.resolve()} is missing')
            job['resume'] = spec.get('resume') or {}
            job['dataset_path'] = spec.get('dataset_path')
            job['reset_stream'] = to_bool(spec.get('reset_stream', False))
        elif mode == 'finetune':
            if not spec.get('source'):
                raise ValueError('Choose the source checkpoint to fine-tune.')
            job['finetune'] = spec
        else:
            raise ValueError('mode must be new, continue or finetune')
        return job

    def _progress(self, event):
        kind = event['event']
        with self.lock:
            if kind == 'start':
                self.info = clean({k: v for k, v in event.items() if k != 'event'})
                self.info['config'] = clean(copy.deepcopy(event['config']))
                self.completed, self.total = event['start_step'], event['total']
                if self.state in ('preparing', 'queued'):
                    self.state = 'training'
                self.detail = ''
            elif kind == 'step':
                self.points.append((event['step'], event['loss'], event['loss_ema'], event['lr'], event['grad_norm']))
                self.completed, self.skips, self.best = event['step'], event['skips'], event.get('best')
                if event.get('validation_loss') is not None:
                    self.val.append((event['step'], event['validation_loss']))
            elif kind == 'preview':
                self.previews.append({'step': event['step'], 'path': event['path']})
            elif kind == 'saved':
                self.detail = f"final checkpoint {event['path']}"
            stop = self.stop_requested
        if kind in ('log', 'spike', 'checkpoint'):
            self._log(event['message'].strip(), kind)
        return stop

    def _run(self, job):
        """Runs on the main thread: TrainingStream and Ctrl+C handling need signals there."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        jit.interrupted = False
        self._set(state='preparing', detail='loading')
        try:
            checkpoint_dir = job['checkpoint_dir']
            os.makedirs(checkpoint_dir, exist_ok=True)
            loaded, ckpt, fine_checkpoint, choose_modules, edit_resume = False, {}, None, None, None
            if job['mode'] == 'new':
                cfg = job['cfg']
                if job['resize_cache']:
                    self._set(detail='resizing dataset into a cache')
                    jit.resize_and_cache_dataset(cfg)
                    self._log(f"Training from resized cache {cfg['dataset_path']}")
            elif job['mode'] == 'continue':
                ckpt = torch.load(os.path.join(checkpoint_dir, 'model.pt'), map_location='cpu', weights_only=False)
                previous = os.path.join(checkpoint_dir, 'config.json')
                previous = previous if os.path.exists(previous) else os.path.join(checkpoint_dir, 'config.pt')
                cfg = copy.deepcopy(ckpt.get('config') or jit.load_config(previous))
                loaded = True
                legacy_class_sampling = cfg.get('conditioning_mode') == 'class' and 'class_sampling' not in cfg
                old_dataset = cfg['dataset_path']
                cfg['dataset_path'] = job['dataset_path'] or old_dataset
                cfg['reset_data_stream'] = (os.path.realpath(cfg['dataset_path']) != os.path.realpath(old_dataset)
                                            or job['reset_stream'])
                if legacy_class_sampling:
                    self._log('Enabling uniform class sampling; restarting the data stream with existing weights/optimizer.')
                    cfg['reset_data_stream'] = True
                resume = job['resume']
                edit_resume = lambda c, optimizer, step: apply_resume_edits(c, optimizer, step, resume)
            else:
                spec = job['finetune']
                fine_checkpoint = torch.load(str(Path(spec['source']).expanduser()), map_location='cpu', weights_only=False)
                previous = os.path.join(checkpoint_dir, 'config.json')
                cfg = copy.deepcopy(fine_checkpoint.get('config') or jit.load_config(previous))
                jit.apply_config_defaults(cfg)
                cfg['dataset_path'] = spec.get('dataset_path') or cfg['dataset_path']
                cfg['lr'] = num(spec, 'lr', float, 0, default=cfg['lr'] * .1)
                cfg['steps'] = num(spec, 'steps', int, 1, default=cfg['steps'])
                cfg['warmup_steps'] = num(spec, 'warmup_steps', int, 0, default=min(100, cfg['steps']))
                cfg['finetune_policy'] = spec.get('policy') or 'all'
                if cfg['finetune_policy'] == 'last':
                    cfg['finetune_last_blocks'] = num(spec, 'last_blocks', int, 1, default=1)
                modules = [str(m).strip() for m in spec.get('modules') or [] if str(m).strip()]
                choose_modules = lambda model: modules
                cfg['finetune_weights'] = spec.get('weights') or ('ema' if 'ema' in fine_checkpoint else 'model')
                if cfg['finetune_weights'] not in {'ema', 'model'} or cfg['finetune_weights'] not in fine_checkpoint:
                    raise jit.ConfigError('Requested source weights are unavailable')
                apply_features(cfg, spec.get('features') or {})
            output_dir = jit.make_unique_output_dir(checkpoint_dir, 'run')
            self._set(run_dir=output_dir, detail='building model')
            self._log(f'Output folder {output_dir}')
            if cfg.get('seed', 0) <= 0:
                self._log('Seed 0: the run is not reproducible.')
            jit.run_training(cfg, device, checkpoint_dir=checkpoint_dir, output_dir=output_dir, loaded=loaded,
                             ckpt=ckpt, fine_checkpoint=fine_checkpoint, choose_modules=choose_modules,
                             edit_resume=edit_resume, progress=self._progress)
            with self.lock:
                stopped = self.stop_requested or jit.interrupted or self.completed < self.total
                self.state = 'stopped' if stopped else 'done'
        except (KeyboardInterrupt, SystemExit):
            # Ctrl+C before the training loop started: like the CLI, quit (the GUI with it).
            self._set(state='stopped', detail='interrupted from the terminal')
            raise
        except Exception as exc:
            traceback.print_exc()
            self._set(state='error', error=f'{type(exc).__name__}: {exc}')
            self._log(f'{type(exc).__name__}: {exc}', 'error')
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        finally:
            jit.interrupted = False
            self._set(finished=time.time())


def apply_resume_edits(cfg, optimizer, completed_steps, r):
    """jit5.prompt_resume_settings + prompt_training_features, from the continue form."""
    cfg['steps'] = completed_steps + num(r, 'extra_steps', int, 0, default=max(1, cfg['steps'] - completed_steps))
    cfg['batch_size'] = num(r, 'batch_size', int, 1, default=cfg['batch_size'])
    adaptive = cfg['optimizer_type'] in ADAPTIVE_LR
    cosine = cfg['lr_schedule'] == 'cosine' and not adaptive and completed_steps >= cfg['warmup_steps']
    cfg.setdefault('cosine_min_lr', cfg['lr'] * .1)
    default_lr = optimizer.param_groups[0]['lr'] if adaptive or cosine else cfg['lr']
    cfg['lr'] = num(r, 'lr', float, default=default_lr)
    jit.apply_muon_group_lrs(optimizer, cfg)
    edits = r.get('optimizer') or {}
    if edits:
        bounds = dict(RESUME_BOUNDS, **RESUME_EXTRA_BOUNDS.get(cfg['optimizer_type'], {}))
        for key, (lo, hi) in bounds.items():
            groups = [g for g in optimizer.param_groups if key in g]
            if groups and edits.get(key) not in (None, ''):
                value = num(edits, key, float, lo, hi)
                for group in groups:
                    group[key] = value
        groups = [g for g in optimizer.param_groups if 'betas' in g]
        if groups and edits.get('betas'):
            betas = tuple(float(b) for b in edits['betas'])
            if len(betas) != len(groups[0]['betas']) or any(not 0 <= b <= .999999 for b in betas):
                raise ValueError('Betas must stay in [0, 0.999999] and keep their count')
            for group in groups:
                group['betas'] = betas
        if cfg['optimizer_type'] in MUON_FAMILY:
            if edits.get('muon_backend') in ('polar_express', 'newton_schulz'):
                cfg['muon_backend'] = edits['muon_backend']
            if 'cautious' in edits:
                cfg['cautious'] = to_bool(edits['cautious'])
            for group in optimizer.param_groups:
                group['orthogonalization_backend'] = cfg.get('muon_backend', group.get('orthogonalization_backend'))
                group['cautious'] = cfg.get('cautious', group.get('cautious', False))
            if not cfg.get('muon_all', False) and cfg['lr'] > 0 and edits.get('fallback_lr') not in (None, ''):
                cfg['muon_adam_lr_ratio'] = num(edits, 'fallback_lr', float, 1e-12) / cfg['lr']
                jit.apply_muon_group_lrs(optimizer, cfg)
    settings = r.get('settings') or {}
    for key, minimum, cast in TRAINING_SETTINGS:
        if settings.get(key) not in (None, ''):
            cfg[key] = num(settings, key, cast, minimum)
    if cfg['conditioning_mode'] == 'class' and cfg['class_dropout_prob'] > 0 and settings.get('guidance_scale') not in (None, ''):
        cfg['guidance_scale'] = num(settings, 'guidance_scale', float, 0)
    if settings.get('compile_mode') in jit.COMPILE_MODES:
        cfg['compile_mode'] = settings['compile_mode']
    for key in TRAINING_FLAGS:
        if key in settings:
            cfg[key] = to_bool(settings[key])
    apply_features(cfg, r.get('features') or {})


# ───────────────────────── sampling ─────────────────────────
class Sampler:
    """Loaded checkpoint + background sampling jobs with live progress."""

    def __init__(self):
        self.lock = threading.Lock()
        self.model = self.cfg = self.key = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state, self.error, self.done, self.total, self.results, self.trajectory = 'idle', None, 0, 0, [], []
        self.output_dir, self.stop_flag, self.thread, self.grid, self.seed = None, False, None, None, None

    def load(self, path):
        path = Path(path).expanduser()
        if path.is_dir():
            path = path / 'model.pt'
        key = (str(path.resolve()), path.stat().st_mtime_ns)
        if self.key != key:
            ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
            cfg = ckpt.get('config') or jit.load_config(str(path.parent / 'config.json'))
            cfg = copy.deepcopy(cfg)
            jit.apply_config_defaults(cfg)
            with torch.random.fork_rng(devices=[]):  # weight init must not advance the global CPU RNG
                model = jit.build_model(cfg, self.device)
            use_ema = cfg['use_ema'] and 'ema' in ckpt
            model.load_state_dict(ckpt['ema'] if use_ema else ckpt['model'])
            model.eval()
            self.model, self.cfg, self.key, self.weights = model, cfg, key, 'EMA' if use_ema else 'model'
        return self.info()

    def info(self):
        cfg = self.cfg
        return clean({'checkpoint': self.key[0], 'weights': self.weights, 'config': cfg,
                      'conditioning': cfg['conditioning_mode'], 'classes': cfg['class_names'],
                      'guidance_available': cfg['conditioning_mode'] == 'class' and cfg['class_dropout_prob'] > 0,
                      'size': [cfg['width'], cfg['height']], 'channels': cfg['channels'],
                      'params': sum(p.numel() for p in self.model.parameters()), 'device': str(self.device)})

    def status(self, since=0):
        with self.lock:
            since = max(0, min(int(since or 0), len(self.results)))
            return clean({'state': self.state, 'error': self.error, 'done': self.done, 'total': self.total,
                          'results': self.results[since:], 'result_total': len(self.results),
                          'trajectory': list(self.trajectory), 'output_dir': self.output_dir, 'grid': self.grid,
                          'seed': self.seed, 'checkpoint': self.key[0] if self.key else None})

    def stop(self):
        with self.lock:
            self.stop_flag = True
        return self.status(10 ** 9)

    def start(self, spec):
        with self.lock:
            if self.state == 'running':
                raise RuntimeError('Sampling is already running.')
        self.load(spec['checkpoint'])
        cfg = copy.deepcopy(self.cfg)
        steps = num(spec, 'steps', int, 1, default=cfg.get('sampling_steps', 50))
        seed = num(spec, 'seed', int, 0, default=0) or random.SystemRandom().randrange(1, 2 ** 31)
        batch_size = num(spec, 'batch_size', int, 1, default=4)
        if cfg['conditioning_mode'] == 'class' and cfg['class_dropout_prob'] > 0:
            cfg['guidance_scale'] = num(spec, 'guidance_scale', float, 0, default=cfg['guidance_scale'])
        if spec.get('x_clip') in ('none', 'static', 'dynamic'):
            cfg['x_clip'] = spec['x_clip']
        bf16 = bf16_supported()
        if cfg['full_bf16'] and not bf16:
            raise RuntimeError('Full BF16 checkpoint sampling requires CUDA BF16 support')
        if cfg['use_amp'] and cfg['amp_dtype'] == 'bf16' and not bf16:
            cfg['amp_dtype'] = 'fp16'
        mode = cfg['conditioning_mode']
        requests = []  # (kind, payload)
        if mode == 'unconditional':
            requests = [('noise', num(spec, 'count', int, 1, default=4))]
        elif mode == 'class':
            for item in spec.get('classes') or []:
                n = num(item, 'count', int, 0, default=1)
                index = item.get('index')
                if n:
                    requests.append(('class', (None if index in (None, '', -1) else int(index), n)))
            if not requests:
                raise ValueError('Choose at least one class and count.')
        else:
            framing = spec.get('framing') if spec.get('framing') in ('tile', 'center') else 'tile'
            paths = []
            for p in spec.get('inputs') or []:
                p = str(p).strip()
                if os.path.isdir(p):
                    paths.extend(jit.find_image_paths(p))
                elif os.path.isfile(p):
                    paths.append(p)
                else:
                    raise ValueError(f'Not a readable file or folder: {p}')
            if not paths:
                raise ValueError('Choose input images or a folder.')
            requests = [('pix2pix', (paths, framing, to_bool(spec.get('combined', False))))]
        output_dir = jit.make_unique_output_dir(str(Path(self.key[0]).parent), 'run')
        total = sum(r[1] if r[0] == 'noise' else r[1][1] if r[0] == 'class' else len(r[1][0]) for r in requests)
        with self.lock:
            self.state, self.error, self.done, self.total, self.results = 'running', None, 0, total, []
            self.trajectory, self.output_dir, self.stop_flag, self.grid = [], output_dir, False, None
            self.seed = seed
        args = (cfg, steps, seed, batch_size, requests, output_dir, to_bool(spec.get('trajectory', True)))
        self.thread = threading.Thread(target=self._run, args=args, daemon=True, name='jit5-sample')
        self.thread.start()
        return self.status()

    def _run(self, cfg, steps, seed, batch_size, requests, output_dir, want_trajectory):
        # Private generators: sampling while a run trains must not touch the run's global RNG state.
        noise_rng, class_rng = torch.Generator().manual_seed(seed), random.Random(seed)
        try:
            flow = jit.FlowMatchingWrapper(
                self.model, pred_mode=cfg['pred_mode'], loss_mode=cfg['loss_mode'], t_loc=cfg['t_mu'],
                t_scale=cfg['t_sigma'], x_clip=cfg['x_clip'], noise_schedule=cfg['noise_schedule'],
                self_cond_prob=cfg['self_cond_prob'], class_dropout_prob=cfg['class_dropout_prob']).to(self.device)
            amp_enabled = cfg['use_amp'] and self.device.type == 'cuda'
            amp_dtype = torch.bfloat16 if cfg['amp_dtype'] == 'bf16' else torch.float16
            shape_of = lambda n: (n, cfg['channels'], cfg['height'], cfg['width'])
            marks = set(round(i * steps / 10) for i in range(1, 11)) | {1}
            first = [True]

            def callback(done, total, z, x_pred):
                if self.stop_flag:
                    raise InterruptedError('stopped')
                if want_trajectory and first[0] and done in marks and x_pred is not None:
                    image = data_url(jit.tensor_to_pil_image(x_pred[0].float()))
                    with self.lock:
                        self.trajectory.append({'step': done, 'steps': total, 'image': image})

            def generate(n, condition=None, classes=None):
                outputs = []
                for start in range(0, n, batch_size):
                    end = min(n, start + batch_size)
                    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                        out = flow.sample(shape_of(end - start), steps=steps,
                                          condition=None if condition is None else condition[start:end],
                                          class_labels=None if classes is None else classes[start:end],
                                          guidance_scale=cfg['guidance_scale'], step_callback=callback,
                                          initial_noise=torch.randn(shape_of(end - start), generator=noise_rng))
                    first[0] = False
                    outputs.extend(out)
                return outputs

            def publish(image, label, pil=None, extra=None):
                index = self.done
                pil = pil or jit.tensor_to_pil_image(image)
                path = os.path.join(output_dir, f'generated_{index}.png')
                pil.save(path)
                item = {'index': index, 'label': label, 'path': path, 'image': data_url(pil)}
                item.update(extra or {})
                with self.lock:
                    self.results.append(item)
                    self.done += 1

            for kind, payload in requests:
                if kind == 'noise':
                    for start in range(0, payload, batch_size):
                        for image in generate(min(batch_size, payload - start)):
                            publish(image, f'sample {self.done + 1}')
                    if payload > 1:
                        from torchvision.utils import make_grid
                        tensors = [(torch.as_tensor(jit.T.ToTensor()(Image.open(r['path'])))) for r in self.results]
                        grid = jit.T.ToPILImage()(make_grid(torch.stack(tensors), nrow=int(math.ceil(math.sqrt(len(tensors)))), padding=2))
                        grid_path = os.path.join(output_dir, 'generated_grid.png')
                        grid.save(grid_path)
                        with self.lock:
                            self.grid = grid_path
                elif kind == 'class':
                    index, n = payload
                    labels = [class_rng.randrange(len(cfg['class_names'])) if index is None else index for _ in range(n)]
                    classes = torch.tensor(labels, device=self.device, dtype=torch.long)
                    for image, label in zip(generate(n, classes=classes), labels):
                        publish(image, cfg['class_names'][label])
                else:
                    paths, framing, combined = payload
                    combined_dir = jit.make_unique_output_dir(output_dir, 'combined_pairs') if combined else None
                    for path in paths:
                        try:
                            source = jit.pix2pix_source_canvas(path, cfg)
                        except (OSError, ValueError) as exc:
                            with self.lock:
                                self.results.append({'index': -1, 'label': os.path.basename(path), 'error': str(exc)})
                                self.done += 1
                            continue
                        canvas = tuple(source.shape[1:])
                        corners = jit.pix2pix_tile_corners(canvas, (cfg['height'], cfg['width']), framing)
                        tiles = torch.stack([source[:, t:t + cfg['height'], l:l + cfg['width']] for t, l in corners])
                        outputs = generate(len(tiles), condition=tiles.to(self.device))
                        if framing == 'center':
                            source, result = tiles[0], outputs[0]
                        else:
                            result = jit.stitch_tiles(outputs, corners, canvas)
                        extra = {'source': data_url(jit.tensor_to_pil_image(source)), 'windows': len(corners)}
                        if combined_dir is not None:
                            pair = jit.make_combined_pix2pix_image(source, result, cfg['pix2pix_direction'])
                            pair.save(os.path.join(combined_dir, f'generated_{self.done:06d}.png'))
                        publish(result, os.path.basename(path), extra=extra)
            with self.lock:
                self.state = 'done'
        except InterruptedError:
            with self.lock:
                self.state = 'stopped'
        except Exception as exc:
            traceback.print_exc()
            with self.lock:
                self.state, self.error = 'error', f'{type(exc).__name__}: {exc}'
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()


# ───────────────────────── run history ─────────────────────────
def read_metrics(path, max_points=4000):
    rows = []
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append([float(row[k]) if row.get(k) not in (None, '') else None for k in
                             ('step', 'train_loss', 'train_loss_ema', 'validation_loss', 'lr', 'grad_norm')])
            except ValueError:
                continue
    val = [(r[0], r[3]) for r in rows if r[3] is not None]
    rows = bucket_mean([[r[0], r[1], r[2], r[4], r[5]] for r in rows], max_points)
    return {'points': rows, 'val': val}


def list_runs(checkpoint_dir=None):
    root = Path(checkpoint_dir or jit.SAVE_DIR).expanduser().resolve()
    out = {'root': str(root), 'runs': [], 'checkpoints': []}
    if not root.is_dir():
        return out
    for name in ('model.pt', 'best.pt'):
        if (root / name).is_file():
            out['checkpoints'].append({'name': name, 'path': str(root / name), 'mtime': (root / name).stat().st_mtime,
                                       'size_mb': (root / name).stat().st_size / 2 ** 20})
    for run in sorted((p for p in root.iterdir() if p.is_dir() and p.name.startswith('run')),
                      key=lambda p: p.stat().st_mtime, reverse=True):
        previews = sorted(run.glob('sample_*.png'), key=lambda p: int(p.stem.split('_')[1]) if p.stem.split('_')[1].isdigit() else 0)
        generated = sorted(run.glob('generated_*.png'))
        metrics = run / 'metrics.csv'
        info = {'name': run.name, 'path': str(run), 'mtime': run.stat().st_mtime, 'previews': len(previews),
                'generated': len(generated), 'has_metrics': metrics.is_file(),
                'spikes': len((run / 'spike_events.log').read_text(encoding='utf-8').splitlines())
                if (run / 'spike_events.log').is_file() else 0}
        if metrics.is_file():
            with open(metrics, 'rb') as handle:
                handle.seek(0, 2)
                size = handle.tell()
                handle.seek(max(0, size - 4096))
                last = handle.read().decode('utf-8', 'replace').strip().splitlines()[-1].split(',')
            if last and last[0].isdigit():
                info['last_step'] = int(last[0])
                try:
                    info['last_loss_ema'] = float(last[2])
                except (IndexError, ValueError):
                    pass
        if not (previews or generated or metrics.is_file()):
            continue
        out['runs'].append(info)
    return clean(out)


def run_details(path):
    run = Path(path).expanduser().resolve()
    if not run.is_dir():
        raise ValueError(f'Not a run folder: {run}')
    out = {'path': str(run), 'name': run.name}
    if (run / 'metrics.csv').is_file():
        out.update(read_metrics(run / 'metrics.csv'))
    previews = []
    for p in run.glob('sample_*.png'):
        step = p.stem.split('_', 1)[1]
        if step.isdigit():
            previews.append({'step': int(step), 'path': str(p)})
    out['previews'] = sorted(previews, key=lambda x: x['step'])
    out['generated'] = [str(p) for p in sorted(run.glob('generated_*.png'), key=lambda p: (len(p.name), p.name))][:400]
    log = run / 'spike_events.log'
    out['spikes'] = log.read_text(encoding='utf-8').splitlines()[-200:] if log.is_file() else []
    return clean(out)


# ───────────────────────── server ─────────────────────────
def run_gui(host='127.0.0.1', port=8768, open_browser=True):
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    jobs = queue.Queue()
    trainer, sampler = Trainer(jobs), Sampler()
    routes = {
        '/api/options': lambda b: options(),
        '/api/fs': lambda b: list_dir(b.get('path'), tuple(b.get('kinds') or ('image', 'json', 'checkpoint'))),
        '/api/dataset/inspect': lambda b: inspect_dataset(b['path'], b.get('mode', 'unconditional')),
        '/api/dataset/synthetic_preview': lambda b: synthetic_preview(b.get('form') or {}),
        '/api/config/check': check_config,
        '/api/config/previous': lambda b: previous_config(b.get('checkpoint_dir')),
        '/api/config/load': lambda b: load_config_file(b['path']),
        '/api/config/save': lambda b: save_config_file(b['path'], b['config']),
        '/api/checkpoint/info': lambda b: checkpoint_info(b['path']),
        '/api/checkpoint/modules': lambda b: checkpoint_modules(b['path']),
        '/api/train/start': trainer.start,
        '/api/train/stop': lambda b: trainer.stop(),
        '/api/train/status': lambda b: trainer.status(b.get('since', 0), b.get('log_since', 0)),
        '/api/sample/load': lambda b: sampler.load(b['path']),
        '/api/sample/start': sampler.start,
        '/api/sample/stop': lambda b: sampler.stop(),
        '/api/sample/status': lambda b: sampler.status(b.get('since', 0)),
        '/api/runs': lambda b: list_runs(b.get('checkpoint_dir')),
        '/api/run': lambda b: run_details(b['path']),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype='application/json'):
            data = body if isinstance(body, bytes) else json.dumps(clean(body), allow_nan=False, default=str).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            # Written images never change (sample_N.png, generated_N.png), so let the browser keep them.
            self.send_header('Cache-Control', 'private, max-age=600' if ctype.startswith('image/') else 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def _handle(self, body):
            path = urlparse(self.path).path
            if path in ('/', '/index.html'):
                return self._send(200, HTML.read_bytes(), 'text/html; charset=utf-8')
            try:
                if path == '/api/image':
                    size = int(body.get('size') or 0) or None
                    return self._send(200, read_image_file(body['path'], size), 'image/png')
                if path not in routes:
                    return self._send(404, {'error': f'unknown route {path}'})
                self._send(200, routes[path](body))
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser dropped the request (page closed or image replaced); nobody to answer
            except Exception as exc:
                if not isinstance(exc, (ValueError, KeyError, FileNotFoundError, RuntimeError)):
                    traceback.print_exc()
                try:
                    self._send(400, {'error': f'{type(exc).__name__}: {exc}'})
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_GET(self):
            self._handle({k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()})

        def do_POST(self):
            n = int(self.headers.get('Content-Length') or 0)
            try:
                body = json.loads(self.rfile.read(n) or b'{}')
            except ValueError:
                return self._send(400, {'error': 'invalid JSON body'})
            self._handle(body)

    server = None
    for p in range(int(port), int(port) + 20):
        try:
            server = ThreadingHTTPServer((host, p), Handler)
            break
        except OSError:
            continue
    if server is None:
        raise OSError(f'No free port in {port}-{int(port) + 19}')
    server.daemon_threads = True
    url = f'http://{host}:{server.server_address[1]}/'
    threading.Thread(target=server.serve_forever, daemon=True, name='jit5-gui-http').start()
    print(f'JiT5 GUI running at {url}  (Ctrl+C: stop the active training run, or quit when idle; '
          f'working directory {os.getcwd()})', flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        while True:  # training jobs run here, on the main thread
            try:
                job = jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            job()
    except (KeyboardInterrupt, SystemExit):
        print('\nStopping GUI.')
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    run_gui(port=int(sys.argv[1]) if len(sys.argv) > 1 else 8768)
