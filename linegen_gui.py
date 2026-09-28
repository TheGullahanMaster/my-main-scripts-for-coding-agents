"""Browser GUI for linegen.py: training setup and monitor, sampling and model inspection.

Start with `python linegen.py` and choose `g`, or run `python linegen_gui.py`.
The server listens on 127.0.0.1 only and runs everything in this process.  Like the
CLI it trains into the working directory (textgen.json + model.pt).
"""
from __future__ import annotations

import base64
import collections
import shutil
import subprocess
import math
import os
import random
import sys
import threading
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import linegen as lg

HTML = Path(__file__).with_name('linegen_gui.html')
MAX_ANALYZE_T = 512        # tokens per teacher-forced analysis
MAX_LAYER_DIM = 2048       # units kept per captured layer
MAX_ATTN_MAPS = 32         # attention maps kept per analysis
MAX_STATE_UNITS = 1024     # units kept per recurrent-state tensor
RNN_EXTRA_IDS = frozenset((500, 504, 503, 501, 407, 502, 506, 507, 509, 510, 511, 513, 514, 515, 516, 518, 519, 601, 602))
CUSTOM_RNN_IDS = frozenset((502, 506, 507, 509, 510, 511, 513, 514, 515, 516, 518, 519, 601, 602))
MINRNN_ACTS = ['Tanh', 'ReLU', 'SiLU', 'GELU', 'Sigmoid', 'g_act (log-space scan)']
MININDRNN_ACTS = ['Tanh', 'ReLU', 'SiLU / Swish', 'PReLU (α=0)', 'PReLU (0.25)', 'LReLU 0.2', 'LReLU 0.01', 'GELU',
                  'BentIdentity', 'Sine', 'Cosine', 'Snake', 'Stepping sine', 'Stepping cosine', 'Mish', 'Cone',
                  'ReLU²', 'g_act (log-space scan)']


# ───────────────────────── helpers ─────────────────────────
def clean(obj):
    """NaN/inf -> None, tuples -> lists, tensors -> lists, recursively (strict JSON)."""
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
    if callable(obj):
        return getattr(obj, '__name__', str(obj))
    return obj


def b64f32(t):
    t = torch.nan_to_num(t.detach().float().cpu().contiguous())
    return base64.b64encode(t.numpy().tobytes()).decode('ascii')


def list_dir(path=None):
    path = Path(path or os.getcwd()).expanduser().resolve()
    if path.is_file():
        path = path.parent
    dirs, files = [], []
    for entry in sorted(path.iterdir(), key=lambda p: p.name.lower()):
        if entry.name.startswith('.'):
            continue
        try:
            if entry.is_dir():
                dirs.append(entry.name)
                continue
            suffix = entry.suffix.lower()
            kind = 'json' if suffix == '.json' else 'checkpoint' if suffix in ('.pt', '.pth') else 'file'
            files.append({'name': entry.name, 'kind': kind, 'size': entry.stat().st_size})
        except OSError:
            continue
    return {'path': str(path), 'parent': str(path.parent), 'dirs': dirs, 'files': files}


PICK_FILTERS = {
    'json': [('Run config (*.json)', '*.json')],
    'checkpoint': [('Checkpoint (*.pt *.pth)', '*.pt *.pth')],
}


def native_pick(kind='file', start=None, title='Choose a file'):
    """Open the desktop's own file dialog on this machine and return the chosen path
    (the browser cannot reveal real paths, and uploading would copy the file)."""
    start = Path(str(start or os.getcwd())).expanduser()
    if start.is_file():
        start = start.parent
    if not start.is_dir():
        start = Path(os.getcwd())
    filters = PICK_FILTERS.get(kind, [])
    if shutil.which('zenity'):
        cmd = ['zenity', '--file-selection', f'--title={title}', f'--filename={start}/']
        for label, pattern in filters:
            cmd.append(f'--file-filter={label} | {pattern}')
        if filters:
            cmd.append('--file-filter=All files | *')
    else:
        script = ('import sys, tkinter as tk\nfrom tkinter import filedialog\nr = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)\n'
                  'p = filedialog.askopenfilename(title=sys.argv[1], initialdir=sys.argv[2], filetypes=%r)\nprint(p or "")' %
                  ([(l, pat) for l, pat in filters] + [('All files', '*')]))
        cmd = [sys.executable, '-c', script, title, str(start)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except FileNotFoundError as exc:
        raise RuntimeError(f'No desktop file dialog available ({exc}); use Browse or Upload instead.')
    path = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ''
    if not path:
        if out.returncode not in (0, 1):
            raise RuntimeError(f'The file dialog failed: {out.stderr.strip()[:300] or out.returncode}. Use Browse or Upload instead.')
        return {'cancelled': True}
    return {'path': os.path.abspath(path)}


def save_upload(name, stream, length, folder='uploads', chunk=1 << 20):
    """Stream an uploaded file into <cwd>/uploads without holding it in memory."""
    name = os.path.basename(str(name or 'upload.bin')).strip() or 'upload.bin'
    dest_dir = Path(os.getcwd()) / folder
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / name
    stem, suffix, n = dest.stem, dest.suffix, 1
    while dest.exists():
        if dest.stat().st_size == length:      # the same upload again: reuse it (and its token caches)
            stream.read(length)
            return {'path': str(dest), 'size': length, 'reused': True}
        dest = dest_dir / f'{stem}-{n}{suffix}'
        n += 1
    tmp = dest.with_suffix(dest.suffix + '.part')
    left = length
    with tmp.open('wb') as f:
        while left > 0:
            data = stream.read(min(chunk, left))
            if not data:
                break
            f.write(data)
            left -= len(data)
    if left:
        tmp.unlink(missing_ok=True)
        raise RuntimeError('Upload interrupted')
    tmp.replace(dest)
    return {'path': str(dest), 'size': length, 'reused': False}


class ConsoleTee:
    """stdout replacement: everything still reaches the terminal; lines printed by
    registered threads are also kept for the browser console."""

    def __init__(self, real):
        self.real, self.lines, self.partial = real, collections.deque(maxlen=3000), {}
        self.threads, self.count, self.lock = set(), 0, threading.Lock()

    def write(self, s):
        n = self.real.write(s)
        tid = threading.get_ident()
        if tid in self.threads and s:
            with self.lock:
                text = self.partial.get(tid, '') + s.replace('\r', '\n')
                *done, rest = text.split('\n')
                self.partial[tid] = rest
                for line in done:
                    if line.strip():
                        self.lines.append(line)
                        self.count += 1
        return n

    def flush(self):
        self.real.flush()

    def __getattr__(self, name):
        return getattr(self.real, name)

    def tail(self, since):
        with self.lock:
            since = max(0, int(since or 0))
            first = self.count - len(self.lines)
            return self.count, list(self.lines)[max(0, since - first):]


CONSOLE = None


def console():
    global CONSOLE
    if CONSOLE is None:
        CONSOLE = sys.stdout if isinstance(sys.stdout, ConsoleTee) else ConsoleTee(sys.stdout)
        sys.stdout = CONSOLE
    return CONSOLE


def run_summary(config_path=None, checkpoint_path=None):
    """What `Resume` would continue, or None."""
    config_path = config_path or lg.CONFIG_PATH
    checkpoint_path = checkpoint_path or lg.CHECKPOINT_PATH
    if not (os.path.exists(config_path) and os.path.exists(checkpoint_path)):
        return None
    try:
        cfg = lg.load_run_config(config_path)
    except Exception as exc:
        return {'error': f'{type(exc).__name__}: {exc}'}
    msel = cfg.get('model_selection')
    keep = ('dataset_path', 'dataset_type', 'tokenizer_mode', 'model_type', 'embed_dim', 'layer_count', 'head_count',
            'seq_len', 'epoch_count', 'batch_size', 'iterations_done', 'train_tokens_done', 'optimizer', 'optim_params',
            'grad_accum_steps', 'use_amp', 'amp_dtype', 'log_interval', 'sample_interval', 'val_interval',
            'save_interval', 'train_sample_count', 'train_sample_len', 'train_sample_prompt', 'train_sample_temperature',
            'val_split', 'classic_val_path', 'use_tbptt', 'bptt_window', 'activation_name', 'seq2seq',
            'lr_scheduler', 'warmup_steps')
    out = {k: cfg.get(k) for k in keep}
    out.update(model=msel, model_name=lg.MODEL_NAMES.get(msel, f'model {msel}'), config=os.path.abspath(config_path),
               checkpoint=os.path.abspath(checkpoint_path), saved=os.path.getmtime(checkpoint_path),
               stateful=msel in lg.RNN_MODEL_IDS, scan=msel in lg.SCAN_MODEL_IDS)
    return out


def options():
    models = []
    for spec in sorted(lg.MODEL_SPECS.values(), key=lambda s: s.id):
        models.append({
            'id': spec.id, 'name': spec.name, 'label': spec.menu_label, 'desc': spec.menu_description,
            'group': spec.menu_group, 'stateful': spec.stateful, 'scan': spec.scan, 'attention': spec.attention,
            'activation': spec.activation_default, 'megabyte': spec.megabyte_mixer,
            'stage_heads': bool(spec.megabyte_mixer) and lg.megabyte_mixer_uses_heads(spec.megabyte_mixer),
            'rnn_extras': spec.id in RNN_EXTRA_IDS, 'custom_rnn': spec.id in CUSTOM_RNN_IDS,
        })
    optimizers = [{'name': o['name'], 'class': o['class'], 'params': o['params']} for o in lg.OPTIMIZER_REGISTRY]
    acts = [{'name': a, 'desc': lg.ACTIVATION_DESCRIPTIONS.get(a, '')} for a in lg.ACT_NAMES]
    return {
        'device': lg.DEVICE, 'cuda': lg.DEVICE == 'cuda', 'cwd': os.getcwd(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'bf16': bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        'models': models, 'groups': list(lg.MODEL_MENU_GROUP_ORDER), 'activations': acts,
        'optimizers': optimizers, 'default_optimizer': 24 if len(optimizers) > 24 else 0,
        'minrnn_acts': MINRNN_ACTS, 'minindrnn_acts': MININDRNN_ACTS,
        'mlp_model_id': lg.MLP_MODEL_ID, 'run': run_summary(),
        'files': {'config': os.path.abspath(lg.CONFIG_PATH), 'checkpoint': os.path.abspath(lg.CHECKPOINT_PATH)},
    }


# ───────────────────────── datasets ─────────────────────────
def inspect_dataset(path, delimiter=None, has_header=True, scan_limit=256 << 20):
    p = Path(str(path)).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f'{p} is not a file')
    size = p.stat().st_size
    lengths, lines, scanned = [], 0, 0
    with p.open('rb') as f:
        head = f.read(4 << 20)
        f.seek(0)
        for raw in f:
            lengths.append(len(raw.rstrip(b'\r\n')))
            lines += 1
            scanned += len(raw)
            if scanned >= scan_limit:
                break
    complete = scanned >= size
    nul = head[:1 << 16].count(0)
    text = head.decode('utf-8', 'replace')
    bad = text.count('�')
    binary = nul > 0 or (len(text) and bad / max(1, len(text)) > 0.02)
    counts = collections.Counter(text)
    words = len(text.split())
    ratio = size / max(1, len(head))
    chars_est = int(len(text) * ratio)
    hist = []
    if lengths:
        hi = max(lengths)
        nb = min(30, max(1, hi))
        width = max(1, math.ceil((hi + 1) / nb))
        hist = [0] * math.ceil((hi + 1) / width)
        for n in lengths:
            hist[n // width] += 1
        hist = {'width': width, 'counts': hist}
    srt = sorted(lengths)
    q = (lambda f: srt[min(len(srt) - 1, int(f * len(srt)))]) if srt else (lambda f: 0)
    out = {
        'path': str(p.resolve()), 'size': size, 'binary': bool(binary), 'lines': lines, 'lines_complete': complete,
        'scanned_bytes': min(scanned, size),
        'line_len': {'max': max(lengths) if lengths else 0, 'mean': sum(lengths) / len(lengths) if lengths else 0,
                     'median': q(0.5), 'p95': q(0.95), 'empty': sum(1 for n in lengths if n == 0)},
        'line_hist': hist, 'charset': len(counts),
        'top_chars': [[c, n] for c, n in counts.most_common(48)],
        'estimates': {'chars': chars_est, 'words': int(words * ratio), 'bytes': size},
        'preview': (head[:1024].hex() if binary else '\n'.join(text.splitlines()[:40])[:6000]),
        'bpe_vocab': os.path.exists(p.with_suffix('.bpe.vocab')),
    }
    if delimiter:
        rows = lg._seq2seq_rows(str(p), delimiter)
        first, examples = next(rows, None), []
        for _ in range(5):
            r = next(rows, None)
            if r is None:
                break
            examples.append(r)
        rows.close()
        if first:
            names = ([f.strip() or f'column {i}' for i, f in enumerate(first)] if has_header
                     else [f'column {i}' for i in range(len(first))])
            if not has_header:
                examples = [first] + examples[:4]
            out['columns'] = {'names': names, 'examples': examples}
    return out


# ───────────────────────── config from the form ─────────────────────────
def _int(f, key, default=None, lo=None, hi=None):
    v = f.get(key, default)
    if v in (None, ''):
        if default is None:
            raise ValueError(f'{key} is required')
        v = default
    v = int(float(v))
    if lo is not None and v < lo:
        raise ValueError(f'{key} must be ≥ {lo}')
    if hi is not None and v > hi:
        raise ValueError(f'{key} must be ≤ {hi}')
    return v


def _float(f, key, default=None, lo=None):
    v = f.get(key, default)
    if v in (None, ''):
        if default is None:
            raise ValueError(f'{key} is required')
        v = default
    v = float(v)
    if lo is not None and v < lo:
        raise ValueError(f'{key} must be ≥ {lo}')
    return v


SCHEDULERS = ('none', 'cosine_warmup', 'cosine', 'one_cycle')


def _scheduler(name):
    name = str(name or 'none')
    if name not in SCHEDULERS:
        raise ValueError(f'Unknown LR scheduler {name}')
    return name


def optimizer_from_form(index, values):
    index = int(index)
    if not 0 <= index < len(lg.OPTIMIZER_REGISTRY):
        raise ValueError('unknown optimizer')
    entry = lg.OPTIMIZER_REGISTRY[index]
    params = {}
    for p in entry['params']:
        raw = (values or {}).get(p['key'])
        if raw is None or raw == '':
            params[p['key']] = p['default']
        elif p['type'] == 'betas':
            params[p['key']] = tuple(float(x) for x in (raw if isinstance(raw, (list, tuple)) else str(raw).split(',')))
        elif p['type'] == 'bool':
            params[p['key']] = raw if isinstance(raw, bool) else str(raw).strip().lower() in ('1', 'true', 'yes', 'y')
        else:
            params[p['key']] = lg._parse_optim_param(str(raw), p['type'], p['default'])
    return {'optimizer': entry['class'], 'optim_params': params}


def seq2seq_from_form(dataset_path, s):
    delimiter = str(s.get('delimiter') or '').replace('\\t', '\t')
    if not delimiter:
        raise ValueError('Seq2seq needs a delimiter')
    has_header = bool(s.get('has_header', True))
    rows = lg._seq2seq_rows(dataset_path, delimiter)
    first = next(rows, None)
    rows.close()
    if not first or len(first) < 2:
        raise ValueError(f'Seq2seq: delimiter {delimiter!r} finds fewer than two columns in the first row')
    names = ([f.strip() or f'column {i}' for i, f in enumerate(first)] if has_header
             else [f'column {i}' for i in range(len(first))])
    roles = [int(r) for r in s.get('roles') or []]
    if len(roles) != len(names):
        roles = [2 if i == len(names) - 1 else 1 for i in range(len(names))]
    input_cols = [i for i, r in enumerate(roles) if r == 1]
    output_cols = [i for i, r in enumerate(roles) if r == 2]
    if not input_cols or not output_cols:
        raise ValueError('Seq2seq: choose at least one input column and one output column')
    return {'source_path': os.path.abspath(dataset_path), 'delimiter': delimiter, 'has_header': has_header,
            'columns': names, 'input_cols': input_cols, 'output_cols': output_cols}


def config_from_form(f, prepare=True):
    """Build the same run config build_config_new() would from the CLI answers."""
    dataset_path = str(Path(str(f.get('dataset_path') or '')).expanduser())
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f'Dataset file not found: {dataset_path or "(empty)"}')
    dataset_type = _int(f, 'dataset_type', 0, 0, 1)
    tokenizer_mode = _int(f, 'tokenizer_mode', 1, -1, 4)
    seq2seq = None
    if dataset_type == 1 and f.get('seq2seq_on'):
        seq2seq = seq2seq_from_form(dataset_path, f.get('seq2seq') or {})
        if prepare:
            dataset_path = lg.prepare_seq2seq_dataset(seq2seq)

    msel = _int(f, 'model')
    if msel not in lg.MODEL_IDS:
        raise ValueError(f'Unknown model {msel}')
    model_type = _int(f, 'model_type', 0, 0, 2) if msel in lg.HIERARCHICAL_MODEL_MIXERS else lg.MODEL_TYPE_NORMAL
    if msel == lg.MLP_MODEL_ID and model_type == lg.MODEL_TYPE_MEGABYTE:
        raise ValueError('The window MLP supports only Normal or MEGABYTE-Bottom up')
    minrnn_act = _int(f, 'minrnn_act', 0, 0, 5) if msel == 902 else _int(f, 'minrnn_act', 0, 0, 17) if msel == 903 else 0
    activation_name = 'gelu'
    if msel in lg.NON_RNN_ACTIVATION_IDS:
        activation_name = str(f.get('activation') or lg.MODEL_DEFAULT_ACTIVATIONS[msel])
        if activation_name not in lg.ACT_NAMES:
            raise ValueError(f'Unknown activation {activation_name}')

    megabyte = model_type in {lg.MODEL_TYPE_MEGABYTE, lg.MODEL_TYPE_MEGABYTE_BOTTOM_UP}
    target_params = None
    mb = {}
    enc_mixers = dec_mixers = None
    if megabyte:
        stages = f.get('stages') or []
        if len(stages) < 2:
            raise ValueError('MEGABYTE needs at least two stages')
        bottom_up = model_type == lg.MODEL_TYPE_MEGABYTE_BOTTOM_UP
        allowed = set(lg.HIERARCHICAL_MODEL_MIXERS)
        mixers = []
        for i, st in enumerate(stages):
            m = int(st.get('mixer') or msel) if f.get('per_stage_cores') else msel
            fine = i == len(stages) - 1
            if m not in allowed or (m == lg.MLP_MODEL_ID and (not bottom_up or not fine)):
                raise ValueError(f'Stage {i + 1}: model {m} cannot be a stage core here')
            mixers.append(m)
        if bottom_up and f.get('separate_encoder'):
            enc_mixers = []
            for i, st in enumerate(stages):
                m = int(st.get('enc_mixer') or msel)
                fine = i == len(stages) - 1
                if m not in allowed or (m == lg.MLP_MODEL_ID and not (fine and mixers[-1] == lg.MLP_MODEL_ID)):
                    raise ValueError(f'Stage {i + 1}: model {m} cannot be an encoder core here')
                enc_mixers.append(m)
            dec_mixers = list(mixers)
        dims, child, depths, heads, lens = [], [], [], [], []
        for i, st in enumerate(stages):
            lens.append(_int(st, 'seq_len', 4, 1))
            d = _int(st, 'dim', 128, 1)
            dims.append(d)
            child.append(_int(st, 'child_dim', min(64, d), 1, d))
            depths.append(_int(st, 'depth', 2, 1))
            heads.append(_int(st, 'heads', 8, 1) if lg.megabyte_mixer_uses_heads(mixers[i]) else 1)
        mb = dict(megabyte_stage_dims=dims, megabyte_stage_depths=depths, megabyte_stage_heads=heads,
                  megabyte_stage_seq_lens=lens, megabyte_stage_child_embed_dims=child, megabyte_stage_mixers=mixers)
        embed_dim, layer_count, head_count = dims[0], sum(depths), heads[0]
    else:
        if f.get('size_mode') == 'target':
            target_params = int(_float(f, 'target_millions', 2.0, 0.0001) * 1e6)
            embed_dim = 256
        else:
            embed_dim = _int(f, 'embed_dim', 256, 1)
        head_count = _int(f, 'head_count', 4, 1) if msel in lg.ATTN_MODEL_IDS else 4
        if msel in lg.ATTN_MODEL_IDS and not target_params and embed_dim % head_count:
            raise ValueError(f'Width {embed_dim} is not divisible by {head_count} heads')
        layer_count = _int(f, 'layer_count', 2, 1)
    ngram_context = _int(f, 'ngram_context', 4, 1, 32) if msel == 2 else 4

    use_norm = res_every = res_type = use_multiplier = rnn_ffn = 0
    rnn_dropout, cell = 0.0, {}
    builtin_rnn = megabyte and any(
        lg.resolve_megabyte_stage_mixer(m) in {'rnn', 'rnn_relu', 'gru', 'lstm'}
        for m in list(mb.get('megabyte_stage_mixers', [])) + list(enc_mixers or []) + list(dec_mixers or []))
    if msel in RNN_EXTRA_IDS or builtin_rnn:
        use_norm = _int(f, 'use_norm', 0, 0, 6)
        res_every = _int(f, 'res_every', 0, 0)
        res_type = _int(f, 'res_type', 0, 0, 3) if res_every > 0 else 0
        rnn_dropout = _float(f, 'dropout', 0.0, 0.0)
        if not builtin_rnn:
            use_multiplier = _int(f, 'use_multiplier', 0, 0, 2)
        if msel in CUSTOM_RNN_IDS and model_type == lg.MODEL_TYPE_NORMAL:
            rnn_ffn = _int(f, 'rnn_ffn', 0, 0, 3)
        if msel == 509:
            cell['indrnn_activation'] = 'tanh' if str(f.get('indrnn_activation', 'relu')).startswith('t') else 'relu'
        if msel in (513, 514):
            cell['relu_gates'] = bool(f.get('relu_gates', False))
        if msel in (515, 516):
            cell['mogrifier_rounds'] = _int(f, 'mogrifier_rounds', 5, 0)
        if msel == 602:
            cell['unicornn_dt'] = _float(f, 'unicornn_dt', 0.1)
            cell['unicornn_alpha'] = _float(f, 'unicornn_alpha', 10.0)
        if msel == 519:
            cell['lru_highway'] = bool(f.get('lru_highway', False))
        if msel == 518:
            cell['rru_middle_multiplier'] = _float(f, 'rru_middle_multiplier', 2.0)
            cell['rru_dropout'] = _float(f, 'rru_dropout', 0.0, 0.0)

    classic_val_path, val_split = '', 0.0
    if dataset_type == 0:
        seq_len = math.prod(mb['megabyte_stage_seq_lens']) if megabyte else _int(f, 'seq_len', 256, 1)
        if f.get('val_mode') == 'file':
            classic_val_path = str(Path(str(f.get('val_path') or '')).expanduser())
            if not os.path.isfile(classic_val_path):
                raise FileNotFoundError(f'Validation file not found: {classic_val_path}')
        elif f.get('val_mode') == 'split':
            val_split = _float(f, 'val_split', 0.1, 0.0)
    else:
        seq_len = 0
        if f.get('val_mode') in ('split', 'file'):
            val_split = _float(f, 'val_split', 0.1, 0.0)
    if not 0.0 <= val_split < 1.0:
        raise ValueError('Validation split must be in [0, 1)')

    sparse = (512, 32, 16)
    if msel == 409:
        sparse = (_int(f, 'sparse_local_window', min(512, max(1, seq_len)), 1),
                  _int(f, 'sparse_compression_block', 32, 1), _int(f, 'sparse_selected_blocks', 16, 1))

    optim = optimizer_from_form(f.get('optimizer', 24), f.get('optim_params'))
    use_tbptt, bptt_window, tbptt_total = False, 0, 0
    if not lg.is_bottom_up_megabyte({'model_type': model_type}) and (msel in lg.RNN_MODEL_IDS or msel in lg.SCAN_MODEL_IDS):
        use_tbptt = bool(f.get('use_tbptt', False))
        if use_tbptt:
            bptt_window = _int(f, 'bptt_window', 64, 1)
            tbptt_total = _int(f, 'tbptt_total_len', 0, 0) if dataset_type == 0 else 0

    cfg = lg.RunConfig(
        dataset_path=dataset_path, dataset_type=dataset_type, model_selection=msel, activation_name=activation_name,
        embed_dim=embed_dim, head_count=head_count, layer_count=layer_count, seq_len=seq_len,
        epoch_count=_int(f, 'epoch_count', 10, 1), batch_size=_int(f, 'batch_size', 32, 1),
        learning_rate=optim['optim_params'].get('lr', 1.0), model_type=model_type, tokenizer_mode=tokenizer_mode,
        use_norm=use_norm, res_every=res_every, res_type=res_type, dropout=rnn_dropout, use_multiplier=use_multiplier,
        train_sample_len=_int(f, 'train_sample_len', 200, 1) if dataset_type == 0 else 0,
        train_sample_count=_int(f, 'train_sample_count', 2, 0),
        train_sample_prompt=str(f.get('train_sample_prompt') or '') if dataset_type == 0 else '',
    ).to_dict()
    cfg['model_id_scheme'] = lg.MODEL_ID_SCHEME
    cfg['line_seq_len_cap'] = None
    if seq2seq:
        cfg['seq2seq'] = seq2seq
    if classic_val_path:
        cfg['classic_val_path'] = classic_val_path
    cuda = lg.DEVICE == 'cuda'
    cfg.update(
        val_split=val_split, use_tbptt=use_tbptt, bptt_window=bptt_window, tbptt_total_len=tbptt_total,
        minrnn_act=minrnn_act, rnn_ffn=rnn_ffn, rnn_cell_options=cell,
        grad_accum_steps=_int(f, 'grad_accum_steps', 1, 1),
        use_amp=bool(f.get('use_amp')) and cuda, amp_dtype='bf16' if f.get('amp_dtype') == 'bf16' else 'fp16',
        lr_scheduler=_scheduler(f.get('lr_scheduler')), warmup_steps=_int(f, 'warmup_steps', 0, 0),
        early_stopping=False, patience=10,
        log_interval=_int(f, 'log_interval', 10, 1), sample_interval=_int(f, 'sample_interval', 500, 1),
        val_interval=_int(f, 'val_interval', 500, 1), save_interval=_int(f, 'save_interval', 10000, 1),
        use_compile=bool(f.get('use_compile')), ngram_context=ngram_context,
        sparse_local_window=sparse[0], sparse_compression_block=sparse[1], sparse_selected_blocks=sparse[2],
        max_grad_norm=_float(f, 'max_grad_norm', 1.0, 0.0),
        train_sample_temperature=_float(f, 'train_sample_temperature', 1.0, 0.0),
    )
    if target_params:
        cfg['target_params'] = target_params
    if megabyte:
        cfg.update(mb)
        if enc_mixers is not None:
            cfg['megabyte_bottom_up_encoder_stage_mixers'] = enc_mixers
            cfg['megabyte_bottom_up_decoder_stage_mixers'] = dec_mixers
        if builtin_rnn:
            cfg.update(megabyte_fused_rnn_version=2, megabyte_fused_rnn_norm_type=use_norm,
                       megabyte_fused_rnn_res_every=res_every, megabyte_fused_rnn_res_type=res_type,
                       megabyte_fused_rnn_dropout=rnn_dropout)
        if model_type == lg.MODEL_TYPE_MEGABYTE_BOTTOM_UP:
            cfg['megabyte_bottom_up_version'] = 6
    cfg['optimizer'] = optim['optimizer']
    cfg['optim_params'] = optim['optim_params']
    if tokenizer_mode == 3:
        cfg['tiktoken_encoding'] = str(f.get('tiktoken_encoding') or 'cl100k_base')
    if tokenizer_mode in {-1, 0}:
        cfg['byte_output_text'] = bool(f.get('byte_output_text', False))
    if tokenizer_mode == 4:
        cfg['custom_bpe_size'] = _int(f, 'custom_bpe_size', 4096, 16)
    return cfg


def check_config(form):
    """Parameter count and derived sizes without training (vocab is built / loaded as training would)."""
    cfg = config_from_form(form, prepare=False)
    notes = []
    if cfg['dataset_type'] == 1:
        info = inspect_dataset(cfg['dataset_path'])
        cfg['seq_len'] = max(1, info['line_len']['max'] + 2)
        if cfg['tokenizer_mode'] == -1:
            cfg['seq_len'] = cfg['seq_len'] * 8
        notes.append(f"line mode: window ≈ longest line ({info['line_len']['max']} bytes) + BOS/EOS")
    if cfg['tokenizer_mode'] == 4 and not os.path.exists(Path(cfg['dataset_path']).with_suffix('.bpe.vocab')):
        vocab_size = None
    else:
        if cfg.get('seq2seq'):
            notes.append('vocabulary estimated from the source file')
        stub = dict(cfg)
        vocab = lg.load_or_make_vocab(stub, cfg['dataset_path'], save_config=False)
        vocab_size = vocab.size
    if vocab_size is None:
        notes.append('BPE vocabulary is trained when the run starts; size assumed')
        vocab_size = int(cfg.get('custom_bpe_size', 4096))
    if cfg.get('target_params'):
        dim, fitted = lg.fit_width_to_params(cfg, vocab_size, cfg['target_params'], cfg.get('head_count', 4))
        cfg['embed_dim'] = dim
        notes.append(f'width {dim} fits the target')
    params = lg._bench_param_count(cfg, vocab_size)
    return {'params': params, 'vocab_size': vocab_size, 'embed_dim': cfg['embed_dim'], 'seq_len': cfg['seq_len'],
            'layer_count': cfg['layer_count'], 'notes': notes}


# ───────────────────────── training ─────────────────────────
class TrainingSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.reset()

    def reset(self):
        self.state, self.error, self.detail, self.info = 'idle', None, '', None
        self.points = []   # (step, loss, grad_norm, lr, tokens, time, epoch)
        self.valid, self.samples, self.checkpoints = [], [], []
        self.started = self.finished = None
        self.stop_requested, self.progress, self.console_start = False, {}, 0

    def status(self, since=0, console_since=None, max_points=4000):
        with self.lock:
            since = max(0, min(int(since or 0), len(self.points)))
            tail = self.points[since:]
            if len(tail) > max_points:
                k = -(-len(tail) // max_points)
                tail = [tail[min(i + k, len(tail)) - 1][:1] + tuple(
                    sum(p[j] for p in tail[i:i + k]) / len(tail[i:i + k]) for j in (1, 2)) + tail[min(i + k, len(tail)) - 1][3:]
                    for i in range(0, len(tail), k)]
            end = self.finished or time.time()
            out = {'state': self.state, 'error': self.error, 'detail': self.detail, 'info': self.info,
                   'total': len(self.points), 'points': [list(p) for p in tail], 'valid': list(self.valid),
                   'samples': list(self.samples[-40:]), 'checkpoints': list(self.checkpoints), 'progress': dict(self.progress),
                   'elapsed': end - self.started if self.started else 0, 'run': run_summary()}
        if console_since is not None:
            count, lines = console().tail(max(int(console_since), self.console_start))
            out['console'], out['console_total'] = lines[-400:], count
        return clean(out)

    def stop(self):
        with self.lock:
            if self.state in ('preparing', 'training'):
                self.stop_requested, self.state = True, 'stopping'
        return self.status(10 ** 12)

    def start(self, spec):
        with self.lock:
            if self.state in ('preparing', 'training', 'stopping'):
                raise RuntimeError('A training run is already active.')
        mode = spec.get('mode', 'new')
        if mode == 'new':
            config_from_form(spec.get('config') or {}, prepare=False)   # validate before starting the thread
        elif not (os.path.exists(lg.CONFIG_PATH) and os.path.exists(lg.CHECKPOINT_PATH)):
            raise FileNotFoundError(f'{mode} needs {lg.CONFIG_PATH} and {lg.CHECKPOINT_PATH} in {os.getcwd()}')
        with self.lock:
            self.reset()
            self.state, self.started = 'preparing', time.time()
            self.console_start = console().count
        self.thread = threading.Thread(target=self._run, args=(spec,), daemon=True, name='linegen-training')
        self.thread.start()
        return self.status()

    def _set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def _event(self, e):
        kind = e.get('event')
        with self.lock:
            if kind == 'step':
                self.points.append((e['step'], e['loss'], e['grad_norm'], e['lr'], e['tokens'], e['time'], e['epoch']))
                self.progress = {k: e[k] for k in ('step', 'epoch', 'epochs', 'batch', 'batches')}
            elif kind == 'valid':
                self.valid.append({'step': e['step'], 'epoch': e['epoch'], **{k: v for k, v in e['metrics'].items()}})
            elif kind == 'samples':
                for s in e['samples']:
                    self.samples.append({'step': e['step'], **s})
                del self.samples[:-200]
            elif kind == 'checkpoint':
                self.checkpoints.append({'step': e['step'], 'time': time.time()})
            return self.stop_requested

    @staticmethod
    def _overrides(cfg, o):
        for key in ('epoch_count', 'batch_size', 'grad_accum_steps', 'log_interval', 'sample_interval', 'val_interval',
                    'save_interval', 'train_sample_count', 'train_sample_len'):
            if o.get(key) not in (None, ''):
                cfg[key] = max(0 if key == 'train_sample_count' else 1, int(float(o[key])))
        for key in ('train_sample_temperature', 'max_grad_norm'):
            if o.get(key) not in (None, ''):
                cfg[key] = float(o[key])
        if o.get('lr_scheduler') not in (None, ''):
            cfg['lr_scheduler'] = _scheduler(o['lr_scheduler'])
        if o.get('warmup_steps') not in (None, ''):
            cfg['warmup_steps'] = max(0, int(float(o['warmup_steps'])))
        if 'train_sample_prompt' in o and cfg['dataset_type'] == 0:
            cfg['train_sample_prompt'] = str(o['train_sample_prompt'] or '')
        if lg.DEVICE == 'cuda' and 'use_amp' in o:
            cfg['use_amp'] = bool(o['use_amp'])
            cfg['amp_dtype'] = 'bf16' if o.get('amp_dtype') == 'bf16' else 'fp16'

    def _run(self, spec):
        tee = console()
        tee.threads.add(threading.get_ident())
        model = opt = None
        mode = spec.get('mode', 'new')
        try:
            if mode == 'new':
                self._set(detail='preparing the dataset')
                cfg = config_from_form(spec.get('config') or {})
            else:
                cfg = lg.load_run_config()
                o = spec.get('overrides') or {}
                self._overrides(cfg, o)
                if mode == 'resume':
                    if cfg['model_selection'] in lg.RNN_MODEL_IDS and not lg.is_bottom_up_megabyte(cfg) and o.get('seq_len'):
                        cfg['seq_len'] = int(o['seq_len'])
                else:   # retry: fresh weights, may change the optimizer (mirrors retry_adjustments)
                    if o.get('optimizer') not in (None, ''):
                        optim = optimizer_from_form(o['optimizer'], o.get('optim_params'))
                        cfg['optimizer'], cfg['optim_params'] = optim['optimizer'], optim['optim_params']
                        cfg['learning_rate'] = optim['optim_params'].get('lr', cfg.get('learning_rate', 0.0))
                    cfg['iterations_done'] = 0
                    cfg['train_tokens_done'] = 0
                    if cfg.get('model_type') == lg.MODEL_TYPE_MEGABYTE_BOTTOM_UP:
                        cfg['megabyte_bottom_up_version'] = 6
                    mixers = cfg.get('megabyte_stage_mixers', ())
                    mixers = (mixers,) if isinstance(mixers, (str, int)) else tuple(mixers)
                    for key in ('megabyte_bottom_up_encoder_stage_mixers', 'megabyte_bottom_up_decoder_stage_mixers'):
                        m = cfg.get(key, ())
                        mixers += (m,) if isinstance(m, (str, int)) else tuple(m)
                    if any(lg.resolve_megabyte_stage_mixer(m) in {'rnn', 'rnn_relu', 'gru', 'lstm'} for m in mixers):
                        cfg['megabyte_fused_rnn_version'] = 2
            self._set(detail='building the vocabulary')
            vocab = lg.load_or_make_vocab(cfg, cfg['dataset_path'])
            cfg['vocab_tokens'] = getattr(vocab, 'tokens', None)
            lg.save_json(lg.CONFIG_PATH, cfg)
            self._set(detail='indexing the dataset')
            dataset, valid = lg.build_datasets(cfg, vocab)
            if mode == 'new' and cfg.get('target_params'):
                self._set(detail='fitting the width to the parameter target')
                dim, fitted = lg.fit_width_to_params(cfg, vocab.size, cfg['target_params'], cfg.get('head_count', 4))
                cfg['embed_dim'] = dim
                lg.save_json(lg.CONFIG_PATH, cfg)
            self._set(detail='building the model')
            model = lg.build_model(cfg, vocab.size)
            model.to(lg.DEVICE)
            if mode == 'resume':
                model.load_state_dict(_strip_compile_prefix(torch.load(lg.CHECKPOINT_PATH, map_location=lg.DEVICE)))
            model = lg.wrap_model_with_compile(model, cfg)
            opt = lg.build_optimizer(model, cfg)
            line_mode = cfg['dataset_type'] == 1
            steps_per_epoch = None
            try:
                steps_per_epoch = lg.training_steps_per_epoch(cfg, dataset)
            except Exception:
                pass
            info = {
                'mode': mode, 'model': cfg['model_selection'], 'model_name': lg.MODEL_NAMES.get(cfg['model_selection']),
                'params': sum(p.numel() for p in model.parameters()), 'vocab_size': vocab.size,
                'seq_len': cfg['seq_len'], 'embed_dim': cfg['embed_dim'], 'layer_count': cfg['layer_count'],
                'batch_size': cfg['batch_size'], 'grad_accum_steps': cfg.get('grad_accum_steps', 1),
                'epoch_count': cfg['epoch_count'], 'steps_per_epoch': steps_per_epoch,
                'start_step': int(cfg.get('iterations_done', 0)), 'line_mode': line_mode,
                'dataset': cfg['dataset_path'], 'train_size': _ds_size(dataset), 'valid_size': _ds_size(valid) if valid is not None else None,
                'tokenizer_mode': cfg.get('tokenizer_mode'), 'optimizer': cfg.get('optimizer'),
                'device': lg.DEVICE, 'amp': cfg.get('amp_dtype') if cfg.get('use_amp') else None,
                'intervals': {k: cfg.get(k) for k in ('log_interval', 'sample_interval', 'val_interval', 'save_interval')},
                'lr_scheduler': cfg.get('lr_scheduler', 'none'), 'warmup_steps': cfg.get('warmup_steps', 0),
            }
            self._set(info=info, state='training' if not self.stop_requested else 'stopping', detail='')
            lg.train_loop(cfg, model, opt, dataset, valid, vocab, line_mode, progress_callback=self._event)
            self._set(state='stopped' if self.stop_requested else 'done', finished=time.time(),
                      detail=f'saved {lg.CHECKPOINT_PATH} + {lg.CONFIG_PATH}')
        except BaseException as exc:
            traceback.print_exc()
            self._set(state='error', error=f'{type(exc).__name__}: {exc}', finished=time.time())
        finally:
            tee.threads.discard(threading.get_ident())
            del model, opt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _ds_size(ds):
    if ds is None:
        return None
    for attr in ('indices', 'offsets'):
        v = getattr(ds, attr, None)
        if v is not None:
            return {'examples': len(v)}
    n = getattr(ds, 'n_tokens', None) or getattr(ds, 'length', None)
    if n is None:
        ids = getattr(ds, 'ids', None)
        n = len(ids) if ids is not None else None
    return {'tokens': int(n)} if n is not None else None


def _strip_compile_prefix(state):
    if any(k.startswith('_orig_mod.') for k in state):
        return {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v for k, v in state.items()}
    return state


# ───────────────────────── capture machinery ─────────────────────────
def _first_tensor(obj):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (tuple, list)):
        for o in obj:
            t = _first_tensor(o)
            if t is not None:
                return t
    if isinstance(obj, dict):
        for o in obj.values():
            t = _first_tensor(o)
            if t is not None:
                return t
    return None


def _rows(t):
    """A captured activation as [rows, units] (batch 1: rows are positions)."""
    t = t.detach()
    if t.dim() == 0:
        return None
    if t.dim() == 1:
        return t[None]
    return t.reshape(-1, t.shape[-1])


class Capture:
    """Records layer outputs (forward hooks) and attention weights (patched softmax /
    SDPA / MultiheadAttention) for one forward pass run on the calling thread."""

    def __init__(self, layers, model, seq_len, attention=True):
        self.layers, self.model, self.T, self.want_attn = layers, model, seq_len, attention
        self.outputs, self.maps, self.current, self.busy = {}, [], 'model', 0
        self.depth = collections.Counter()
        self.handles, self.patched = [], []
        self.tid = threading.get_ident()

    def _mine(self):
        return threading.get_ident() == self.tid and not self.busy

    def __enter__(self):
        for name, mod in self.layers:
            self.handles.append(mod.register_forward_pre_hook(lambda m, i, name=name: setattr(self, 'current', name)))
            self.handles.append(mod.register_forward_hook(lambda m, i, o, name=name: self._layer(name, o)))
            # recurrent / scan blocks are often driven through step() or
            # forward_seq() at inference, which bypasses forward hooks
            for meth in ('step', 'forward_seq'):
                orig = getattr(type(mod), meth, None)
                if orig is None or meth in vars(mod):
                    continue
                bound = getattr(mod, meth)

                def wrapped(*args, _bound=bound, _name=name, **kw):
                    self.current = _name
                    self.depth[_name] += 1
                    try:
                        out = _bound(*args, **kw)
                    finally:
                        self.depth[_name] -= 1
                    if self.depth[_name] == 0:
                        self._layer(_name, out, 'method')
                    return out
                setattr(mod, meth, wrapped)
                self.patched.append((None, mod, meth))
        if self.want_attn:
            self._patch_attention()
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        for owner, attr, orig in reversed(self.patched):
            if owner is None:
                delattr(attr, orig)     # instance attribute shadowing a method
            else:
                setattr(owner, attr, orig)
        return False

    def _layer(self, name, out, source='forward'):
        if threading.get_ident() != self.tid or (source == 'forward' and self.depth[name]):
            return
        t = _first_tensor(out)
        if t is None or not t.is_floating_point():
            return
        r = _rows(t)
        if r is None:
            return
        self.outputs.setdefault((name, source), []).append(r[:, :MAX_LAYER_DIM].float().cpu().to(torch.float16))

    def _record(self, w, kind):
        if len(self.maps) >= MAX_ATTN_MAPS or w.dim() < 2:
            return
        w = w.detach()
        if w.dim() == 2:
            w = w[None]
        else:
            w = w.reshape(-1, *w.shape[-3:])[0] if w.dim() >= 4 else w
        tq, tk = w.shape[-2], w.shape[-1]
        if tq < 2 and tk < 2 or tq > 1024 or tk > 1024:
            return
        self.maps.append({'layer': self.current, 'kind': kind, 'w': w[:16].float().cpu().to(torch.float16),
                          'heads': int(w.shape[0])})

    def _patch_attention(self):
        cap = self
        sdpa = getattr(F, 'scaled_dot_product_attention', None)
        if sdpa is not None:
            def sdpa_rec(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, *args, **kw):
                out = sdpa(q, k, v, attn_mask, dropout_p, is_causal, *args, **kw)
                if cap._mine():
                    cap.busy += 1
                    try:
                        scale = kw.get('scale') or 1.0 / math.sqrt(q.shape[-1])
                        s = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
                        if is_causal:
                            tq, tk = s.shape[-2], s.shape[-1]
                            s = s.masked_fill(~torch.ones(tq, tk, dtype=torch.bool, device=s.device).tril(), float('-inf'))
                        if attn_mask is not None:
                            s = s.masked_fill(~attn_mask, float('-inf')) if attn_mask.dtype == torch.bool else s + attn_mask.float()
                        cap._record(torch.softmax(s, -1), 'sdpa')
                    finally:
                        cap.busy -= 1
                return out
            self.patched.append((F, 'scaled_dot_product_attention', sdpa))
            F.scaled_dot_product_attention = sdpa_rec

        def softmax_wrap(fn, kind):
            def rec(input, *args, **kw):
                out = fn(input, *args, **kw)
                if cap._mine() and torch.is_tensor(out) and out.dim() >= 3:
                    dim = kw.get('dim', args[0] if args else None)
                    if (dim is None or dim in (-1, out.dim() - 1)) and out.shape[-1] > 1 and out.shape[-2] > 1 and \
                            cap.T in (out.shape[-1], out.shape[-2]):
                        cap._record(out, kind)
                return out
            return rec
        for owner, attr in ((F, 'softmax'), (torch, 'softmax'), (torch.Tensor, 'softmax')):
            orig = getattr(owner, attr)
            self.patched.append((owner, attr, orig))
            setattr(owner, attr, softmax_wrap(orig, 'softmax'))
        for mod in self.model.modules():
            if isinstance(mod, nn.MultiheadAttention):
                orig = mod.forward

                def fwd(*args, _orig=orig, **kw):
                    if not cap._mine():
                        return _orig(*args, **kw)
                    wanted = kw.get('need_weights', True)
                    avg = kw.get('average_attn_weights', True)
                    kw.update(need_weights=True, average_attn_weights=False)
                    cap.busy += 1
                    try:
                        out, w = _orig(*args, **kw)
                    finally:
                        cap.busy -= 1
                    if w is not None:
                        cap._record(w, 'multihead')
                    return out, (None if not wanted else w.mean(1) if avg and w is not None and w.dim() == 4 else w)
                mod.forward = fwd
                self.patched.append((None, mod, 'forward'))


def find_token_embedding(model, vocab_size):
    embs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Embedding)]
    for n, m in embs:
        if m.num_embeddings == vocab_size:
            return n, m
    bigger = [(n, m) for n, m in embs if m.num_embeddings >= vocab_size]
    return bigger[0] if bigger else (None, None)


def find_layers(model, vocab_size):
    """(name, module) pairs worth recording: the token embedding, then the blocks of
    the largest ModuleList (or the top-level children when there is none)."""
    best, best_n = None, 0
    for name, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and len(m) > 0 and not all(isinstance(c, nn.Embedding) for c in m):
            n = sum(p.numel() for p in m.parameters())
            if n > best_n:
                best, best_n = (name, m), n
    out = []
    ename, emb = find_token_embedding(model, vocab_size)
    if emb is not None:
        out.append((ename or 'embedding', emb))
    if best is not None:
        out += [(f'{best[0]}.{i}', c) for i, c in enumerate(best[1])]
    else:
        for name, c in model.named_children():
            if c is emb or not any(True for _ in c.parameters()):
                continue
            out.append((name, c))
    return out


def _state_leaves(state, prefix='state'):
    if torch.is_tensor(state):
        yield prefix, state
    elif isinstance(state, (tuple, list)):
        for i, s in enumerate(state):
            yield from _state_leaves(s, f'{prefix}[{i}]')
    elif isinstance(state, dict):
        for k, s in state.items():
            yield from _state_leaves(s, f'{prefix}.{k}')


# ───────────────────────── sampler / inspector ─────────────────────────
class Sampler:
    def __init__(self):
        self.lock = threading.RLock()      # guards the model (one forward at a time)
        self.model = self.cfg = self.vocab = None
        self.job, self.job_id, self.analysis = None, 0, None
        self.source = None

    # ---- model ----
    def load(self, config_path=None, checkpoint_path=None):
        config_path = config_path or lg.CONFIG_PATH
        checkpoint_path = checkpoint_path or lg.CHECKPOINT_PATH
        for p in (config_path, checkpoint_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f'{p} not found')
        cfg = lg.load_run_config(config_path)
        vocab = lg.load_or_make_vocab(cfg, cfg['dataset_path'], save_config=False)
        model = lg.build_model(cfg, vocab.size)
        model.to(lg.DEVICE)
        model.load_state_dict(_strip_compile_prefix(torch.load(checkpoint_path, map_location=lg.DEVICE)))
        model.eval()
        with self.lock:
            if self.job and self.job['state'] == 'running':
                self.job['stop'] = True
            self.model, self.cfg, self.vocab, self.analysis = model, cfg, vocab, None
            self.source = {'config': os.path.abspath(config_path), 'checkpoint': os.path.abspath(checkpoint_path),
                           'saved': os.path.getmtime(checkpoint_path)}
            self.layers = find_layers(model, vocab.size)
            self.embedding = find_token_embedding(model, vocab.size)[1]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.info()

    def require(self):
        if self.model is None:
            raise RuntimeError('Load a model first.')

    @property
    def line_mode(self):
        return self.cfg['dataset_type'] == 1

    @property
    def stateful_path(self):
        cfg, model, msel = self.cfg, self.model, self.cfg['model_selection']
        if isinstance(model, lg.MegaByteLM) and model.is_incremental:
            return True
        return (msel in lg.SCAN_MODEL_IDS or msel in lg.RNN_MODEL_IDS) and (
            not lg.is_bottom_up_megabyte(cfg) or getattr(model, 'is_incremental', False))

    def info(self):
        if self.model is None:
            return {'loaded': False, 'run': run_summary()}
        cfg, v, msel = self.cfg, self.vocab, self.cfg['model_selection']
        tm = int(cfg.get('tokenizer_mode', 1))
        return clean({
            'loaded': True, 'source': self.source, 'model': msel, 'model_name': lg.MODEL_NAMES.get(msel),
            'params': sum(p.numel() for p in self.model.parameters()), 'vocab_size': v.size,
            'tokenizer_mode': tm, 'byte_mode': tm in (-1, 0), 'byte_output_text': bool(cfg.get('byte_output_text')),
            'line_mode': self.line_mode, 'seq_len': cfg['seq_len'], 'embed_dim': cfg['embed_dim'],
            'layer_count': cfg['layer_count'], 'model_type': cfg.get('model_type', 0),
            'stateful': self.stateful_path, 'attention': msel in lg.ATTN_MODEL_IDS,
            'iterations_done': cfg.get('iterations_done', 0), 'train_tokens_done': cfg.get('train_tokens_done', 0),
            'seq2seq': cfg.get('seq2seq'), 'word_tokens': tm == 2, 'layers': [n for n, _ in self.layers],
            'embedding': self.embedding is not None, 'dataset': cfg.get('dataset_path'),
            'max_analyze': min(MAX_ANALYZE_T, cfg['seq_len']) if not self.stateful_path else MAX_ANALYZE_T,
            'train_sample_prompt': cfg.get('train_sample_prompt', ''),
        })

    # ---- tokens ----
    def piece(self, i):
        v = self.vocab
        if self.line_mode:
            if i == v.bos_id:
                return '⟨BOS⟩'
            if i == getattr(v, 'eos_id', None):
                return '⟨EOS⟩'
            if i == getattr(v, 'pad_id', None):
                return '⟨PAD⟩'
        if isinstance(v, lg.ByteVocab):
            return chr(i) if 32 <= i < 127 or i in (9, 10) else f'\\x{i:02x}'
        try:
            return v.decode([i])
        except Exception:
            return f'⟨{i}⟩'

    def text_of(self, ids):
        v = self.vocab
        if hasattr(v, 'to_bytes') and isinstance(v, (lg.ByteVocab, lg.BinaryVocab)):
            if isinstance(v, lg.BinaryVocab) and not self.cfg.get('byte_output_text'):
                return v.decode([i for i in ids if i not in (v.bos_id, v.eos_id, v.pad_id)])
            return v.to_bytes(ids).decode('utf-8', 'replace')
        if self.line_mode:
            ids = [i for i in ids if i not in (v.bos_id, getattr(v, 'eos_id', None), getattr(v, 'pad_id', None))]
        return v.decode(ids)

    def encode_prompt(self, spec):
        """-> (ids, warnings).  kind: text | hex | file (base64 bytes)."""
        self.require()
        v, cfg = self.vocab, self.cfg
        kind, text = spec.get('kind', 'text'), spec.get('text') or ''
        data = base64.b64decode(spec['data']) if kind == 'file' and spec.get('data') else b''
        tm, warnings = int(cfg.get('tokenizer_mode', 1)), []
        if kind == 'file' and tm not in (-1, 0):
            text = data.decode('utf-8', 'replace')
        if self.line_mode and cfg.get('seq2seq'):
            if kind == 'file':
                text = data.decode('utf-8', 'replace').splitlines()[0] if data else ''
            ids = lg.encode_seq2seq_prompt(v, cfg['seq2seq'], text)
            return ids, warnings
        if tm in (-1, 0):
            raw = lg._hex_to_bytes(text) if kind == 'hex' else data if kind == 'file' else text.encode('utf-8')
            ids = v.encode(raw)
        else:
            if kind == 'hex':
                raise ValueError('Hex prompts are for the byte and binary tokenizers.')
            if tm in (1, 2) and hasattr(v, 'stoi'):
                if tm == 1:
                    missing = sorted({c for c in text if c not in v.stoi})
                    text = ''.join(c for c in text if c in v.stoi)
                else:
                    missing = sorted({w for w in text.split() if w not in v.stoi})
                    text = ' '.join(w for w in text.split() if w in v.stoi)
                if missing:
                    shown = ''.join(missing[:30]) if tm == 1 else ' '.join(missing[:12])
                    warnings.append(f'{len(missing)} unknown {"characters" if tm == 1 else "words"} dropped: {shown!r}')
            ids = v.encode(lg.BOS_TOKEN + text if self.line_mode else text)
        if self.line_mode and ids and ids[-1] == getattr(v, 'eos_id', None):
            ids = ids[:-1]
        return list(ids), warnings

    def tokenize(self, spec):
        ids, warnings = self.encode_prompt(spec)
        return {'ids': ids, 'pieces': [self.piece(i) for i in ids], 'warnings': warnings}

    def _token_record(self, tok, raw):
        probs = torch.softmax(raw.float(), -1)
        logp = torch.log2(probs.clamp_min(1e-30))
        top = torch.topk(probs, min(5, probs.numel()))
        return {'id': int(tok), 't': self.piece(int(tok)), 'p': float(probs[tok]),
                'h': float(-(probs * logp).sum()),
                'alt': [[self.piece(int(i)), float(p)] for p, i in zip(top.values.tolist(), top.indices.tolist())]}

    # ---- streaming sampling ----
    def start_sample(self, spec):
        self.require()
        with self.lock:
            if self.job and self.job['state'] == 'running':
                raise RuntimeError('Sampling is already running.')
        ids, warnings = self.encode_prompt(spec.get('prompt') or {})
        count = max(1, min(64, int(spec.get('count', 1))))
        default_len = self.cfg['seq_len'] if self.line_mode else 200
        max_tokens = max(1, min(100_000, int(spec.get('max_tokens') or default_len)))
        cfg = dict(self.cfg)
        cfg.update(temperature=float(spec.get('temperature', 1.0)), _top_k=int(spec.get('top_k', 0) or 0),
                   _top_p=float(spec.get('top_p', 0.0) or 0.0),
                   _rep_penalty=max(1e-3, float(spec.get('rep_penalty', 1.0) or 1.0)),
                   _rep_window=max(1, int(spec.get('rep_window', 64) or 64)))
        self.job_id += 1
        job = {'id': self.job_id, 'state': 'running', 'stop': False, 'error': None, 'warnings': warnings,
               'samples': [], 'count': count, 'max_tokens': max_tokens, 'started': time.time()}
        self.job = job
        threading.Thread(target=self._sample_run, args=(job, cfg, ids, count, max_tokens), daemon=True,
                         name='linegen-sampling').start()
        return self.sample_status({'id': job['id']})

    def _sample_run(self, job, cfg, ids, count, max_tokens):
        try:
            model, vocab = self.model, self.vocab
            for i in range(count):
                if job['stop']:
                    break
                s = {'prompt_ids': list(ids), 'prompt': [self.piece(t) for t in ids], 'tokens': [], 'done': False,
                     'stop_reason': None, 'started': time.time()}
                job['samples'].append(s)

                def on_token(tok, raw, s=s):
                    s['tokens'].append(self._token_record(tok, raw))
                    return job['stop']
                with self.lock:
                    if self.line_mode:
                        out, reason = lg.generate_line_mode(model, cfg, vocab, list(ids), limit_len=max_tokens,
                                                            return_stop_reason=True, on_token=on_token)
                    else:
                        out = lg.generate_classic(model, cfg, vocab, list(ids), max_len=max_tokens, stream=False,
                                                  on_token=on_token)
                        reason = f'max_len={max_tokens}'
                if job['stop'] and len(s['tokens']) < max_tokens and reason != 'EOS':
                    reason = 'stopped'
                gen = len(s['tokens'])
                if self.line_mode and reason == 'EOS':
                    gen -= 1
                s['prompt_ids'] = out[:len(out) - gen] if gen else list(out)
                s['prompt'] = [self.piece(t) for t in s['prompt_ids']]
                s['ids'] = list(out) + ([vocab.eos_id] if self.line_mode and reason == 'EOS' else [])
                s['text'] = self.text_of(out[len(s['prompt_ids']):])
                if int(cfg.get('tokenizer_mode', 1)) in (-1, 0):
                    s['bytes_hex'] = vocab.to_bytes(out).hex()
                s['stop_reason'], s['done'], s['elapsed'] = reason, True, time.time() - s['started']
            job['state'] = 'stopped' if job['stop'] else 'done'
        except Exception as exc:
            traceback.print_exc()
            job['state'], job['error'] = 'error', f'{type(exc).__name__}: {exc}'

    def sample_status(self, body):
        job = self.job
        if job is None or (body.get('id') and int(body['id']) != job['id']):
            return {'state': 'idle', 'samples': []}
        known = body.get('known') or []
        samples = []
        for i, s in enumerate(list(job['samples'])):
            k = int(known[i]) if i < len(known) else 0
            d = {kk: vv for kk, vv in s.items() if kk != 'tokens'}
            d['from'], d['tokens'] = k, s['tokens'][k:]
            samples.append(d)
        return clean({'id': job['id'], 'state': job['state'], 'error': job['error'], 'warnings': job['warnings'],
                      'count': job['count'], 'max_tokens': job['max_tokens'], 'samples': samples})

    def stop_sample(self):
        if self.job:
            self.job['stop'] = True
        return {'ok': True}

    # ---- analysis ----
    def _window(self, ids):
        ids = [int(i) for i in ids]
        if not ids:
            raise ValueError('Nothing to analyze: the text has no tokens.')
        cap = MAX_ANALYZE_T if self.stateful_path else min(MAX_ANALYZE_T, max(1, int(self.cfg['seq_len'])))
        return ids[-cap:], max(0, len(ids) - cap)

    def _forward(self, x):
        out = self.model(x, None) if self.stateful_path else self.model(x)
        return out[0] if isinstance(out, tuple) else out

    def analyze(self, body):
        self.require()
        if body.get('ids'):
            ids = body['ids']
        else:
            ids = self.encode_prompt(body.get('prompt') or {})[0]
        ids, offset = self._window(ids)
        T = len(ids)
        x = torch.tensor([ids], dtype=torch.long, device=lg.DEVICE)
        with self.lock, torch.no_grad():
            self.model.eval()
            with Capture(self.layers, self.model, T, attention=bool(body.get('attention', True))) as cap:
                logits = self._forward(x)
            logits = logits.float()
            rows = logits.reshape(-1, logits.shape[-1])[-T:] if logits.dim() == 3 else logits.reshape(-1, logits.shape[-1])
            probs = torch.softmax(rows, -1)
            ent = -(probs * torch.log2(probs.clamp_min(1e-30))).sum(-1)
            nxt = torch.tensor(ids[1:] + [0], device=probs.device)
            p_next = probs.gather(1, nxt[:, None])[:, 0]
            top = torch.topk(probs, min(5, probs.shape[-1]), -1)
            states = self._state_trace(ids, rows[-1]) if self.stateful_path and body.get('states', True) else None
        positions = []
        for t in range(T):
            positions.append({'p_next': float(p_next[t]) if t + 1 < T else None, 'h': float(ent[t]),
                              'top': [[self.piece(int(i)), float(p), int(i)] for p, i in zip(top.values[t].tolist(), top.indices[t].tolist())]})
        layers = []
        for name, _ in self.layers:
            chunks = cap.outputs.get((name, 'forward')) or cap.outputs.get((name, 'method'))
            if not chunks:
                continue
            if len(chunks) > 1 and all(c.shape[0] == 1 for c in chunks):
                r = torch.cat(chunks, 0)       # a cell run once per time step
            else:
                r = chunks[0]
            rf = r.float()
            layers.append({'name': name, 'rows': int(r.shape[0]), 'dim': int(r.shape[1]), 'calls': len(chunks),
                           'rms': rf.pow(2).mean(1).sqrt().tolist(), 'mean': float(rf.mean()), 'std': float(rf.std()) if rf.numel() > 1 else 0.0,
                           '_data': r})
        maps = [{'index': i, 'layer': m['layer'], 'kind': m['kind'], 'heads': m['heads'], 'kept_heads': int(m['w'].shape[0]),
                 'tq': int(m['w'].shape[-2]), 'tk': int(m['w'].shape[-1]), '_w': m['w']} for i, m in enumerate(cap.maps)]
        self.analysis = {'ids': ids, 'layers': {l['name']: l for l in layers}, 'maps': maps, 'states': states}
        return clean({
            'ids': ids, 'pieces': [self.piece(i) for i in ids], 'offset': offset, 'positions': positions,
            'nll_bits': [(-math.log2(max(p['p_next'], 1e-30))) if p['p_next'] is not None else None for p in positions],
            'layers': [{k: v for k, v in l.items() if not k.startswith('_')} for l in layers],
            'attention': [{k: v for k, v in m.items() if not k.startswith('_')} for m in maps],
            'attention_truncated': len(cap.maps) >= MAX_ATTN_MAPS,
            'states': None if states is None else {k: v for k, v in states.items() if k != 'leaves'} | {
                'leaves': [{k: v for k, v in l.items() if not k.startswith('_')} for l in states['leaves']]},
            'stateful': self.stateful_path, 'embedding': self.embedding is not None,
        })

    def _state_trace(self, ids, parallel_last):
        """Run the model one token at a time and record every tensor in its carried state."""
        leaves, dropped = {}, set()
        state, logits = None, None
        for t, tok in enumerate(ids):
            x = torch.tensor([[tok]], dtype=torch.long, device=lg.DEVICE)
            out = self.model(x, state)
            if not isinstance(out, tuple) or len(out) < 2:
                return {'error': 'this model does not return a recurrent state', 'leaves': []}
            logits, state = out[0], out[1]
            for name, ten in _state_leaves(state):
                if name in dropped or not ten.is_floating_point():
                    continue
                flat = ten.detach().reshape(-1)
                rec = leaves.get(name)
                if rec is None:
                    n = flat.numel()
                    idx = None if n <= MAX_STATE_UNITS * 2 else torch.linspace(0, n - 1, MAX_STATE_UNITS, device=flat.device).long()
                    rec = leaves[name] = {'name': name, 'shape': list(ten.shape), 'numel': n, 'idx': idx, 'frames': [], 'first': t}
                elif flat.numel() != rec['numel']:
                    dropped.add(name)       # growing cache (e.g. attention keys): not a fixed-size state
                    del leaves[name]
                    continue
                v = flat if rec['idx'] is None else flat[rec['idx']]
                rec['frames'].append(v.float().cpu().to(torch.float16))
        out_leaves = []
        for rec in leaves.values():
            if len(rec['frames']) != len(ids) - rec['first']:
                continue
            data = torch.stack(rec['frames'])
            f = data.float()
            out_leaves.append({'name': rec['name'], 'shape': rec['shape'], 'units': int(data.shape[1]),
                               'subsampled': rec['idx'] is not None, 'first': rec['first'],
                               'rms': f.pow(2).mean(1).sqrt().tolist(),
                               'delta': [0.0] + (f[1:] - f[:-1]).pow(2).mean(1).sqrt().tolist(), '_data': data})
        consistency = None
        if logits is not None:
            a = torch.softmax(logits.float().reshape(-1, logits.shape[-1])[-1], -1)
            b = torch.softmax(parallel_last.float(), -1)
            consistency = float((a - b).abs().sum() / 2)   # total variation distance
        return {'leaves': out_leaves, 'dropped': sorted(dropped), 'consistency_tv': consistency}

    def _require_analysis(self):
        if not self.analysis:
            raise RuntimeError('Run an analysis first.')
        return self.analysis

    @staticmethod
    def _pick_units(data, units, order):
        f = data.float()
        n = f.shape[1]
        units = max(1, min(int(units or 128), 512, n))
        if order == 'index':
            idx = torch.arange(n)[:units]
        elif order == 'mean':
            idx = torch.argsort(f.mean(0).abs(), descending=True)[:units]
        else:
            idx = torch.argsort(f.var(0, unbiased=False) if f.shape[0] > 1 else f.abs()[0], descending=True)[:units]
        z = f[:, idx]
        return {'rows': int(z.shape[0]), 'units': [int(i) for i in idx], 'z': b64f32(z),
                'lo': float(z.min()), 'hi': float(z.max()), 'absmax': float(z.abs().max())}

    def layer_detail(self, body):
        a = self._require_analysis()
        layer = a['layers'].get(body.get('layer'))
        if layer is None:
            raise KeyError(f"no captured output for layer {body.get('layer')!r}")
        return self._pick_units(layer['_data'], body.get('units', 128), body.get('order', 'variance'))

    def position_units(self, body):
        """Strongest units of every layer at one position (what fires for this token)."""
        a = self._require_analysis()
        pos, k = int(body['position']), int(body.get('k', 8))
        out = []
        for name, l in a['layers'].items():
            d = l['_data'].float()
            if pos >= d.shape[0]:
                continue
            row = d[pos]
            z = (row - d.mean(0)) / (d.std(0) + 1e-6) if d.shape[0] > 1 else row
            top = torch.topk(z.abs(), min(k, z.numel()))
            out.append({'layer': name, 'units': [[int(i), float(row[i]), float(z[i])] for i in top.indices.tolist()]})
        return {'position': pos, 'layers': out}

    def unit_values(self, body):
        """One unit of a captured layer (or recorded state tensor) at every position."""
        a = self._require_analysis()
        if body.get('leaf'):
            leaf = next((l for l in (a.get('states') or {}).get('leaves', []) if l['name'] == body['leaf']), None)
            if leaf is None:
                raise KeyError(f"no recorded state {body['leaf']!r}")
            data, first = leaf['_data'], leaf['first']
        else:
            layer = a['layers'].get(body.get('layer'))
            if layer is None:
                raise KeyError(f"no captured output for layer {body.get('layer')!r}")
            data, first = layer['_data'], 0
        unit = int(body['unit'])
        if not 0 <= unit < data.shape[1]:
            raise ValueError(f'unit must be 0–{data.shape[1] - 1}')
        v = data[:, unit].float()
        return {'unit': unit, 'first': first, 'values': v.tolist(), 'absmax': float(v.abs().max()),
                'mean': float(v.mean()), 'std': float(v.std()) if v.numel() > 1 else 0.0}

    def attention_detail(self, body):
        a = self._require_analysis()
        m = a['maps'][int(body['index'])]
        w = m['_w'].float()
        head = int(body.get('head', -1))
        z = w.mean(0) if head < 0 or head >= w.shape[0] else w[head]
        return {'tq': int(z.shape[0]), 'tk': int(z.shape[1]), 'z': b64f32(z), 'head': head,
                'head_entropy': [float(-(h * torch.log2(h.clamp_min(1e-30))).sum(-1).mean()) for h in w]}

    def state_detail(self, body):
        a = self._require_analysis()
        states = a.get('states') or {}
        for leaf in states.get('leaves', []):
            if leaf['name'] == body.get('leaf'):
                return self._pick_units(leaf['_data'], body.get('units', 128), body.get('order', 'variance'))
        raise KeyError(f"no recorded state {body.get('leaf')!r}")

    def influence(self, body):
        """Which earlier tokens raised the log-probability of the token after `position`
        (or of `target`)?  ablation: swap one token's embedding for the mean embedding
        and measure the drop (batched forwards); gradient: gradient × input on the
        embeddings (one backward pass; degenerate when a norm follows the embedding)."""
        self.require()
        emb = self.embedding
        if emb is None:
            raise RuntimeError('This model has no token embedding table to attribute to.')
        ids, _ = self._window(body.get('ids') or self._require_analysis()['ids'])
        pos = int(body['position'])
        if not 0 <= pos < len(ids):
            raise ValueError('position outside the analyzed text')
        target, T = body.get('target'), len(ids)
        x = torch.tensor([ids], dtype=torch.long, device=lg.DEVICE)
        box, grad_x_input, grad_norm, grad_error = {}, None, None, None

        def keep(m, i, o):
            box.setdefault('e', o)
        with self.lock:
            handle = emb.register_forward_hook(keep)
            try:
                # cuDNN RNN kernels have no backward in eval mode; the native path does
                with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
                    logits = self._forward(x).float()
                    logp = torch.log_softmax(logits.reshape(-1, logits.shape[-1])[-T:][pos], -1)
                    if target is None:
                        target = ids[pos + 1] if pos + 1 < T else int(logp.argmax())
                    target = int(target)
                    e = box.get('e')
                    if e is None or e.dim() != 3 or e.shape[0] != 1 or e.shape[1] != T:
                        raise RuntimeError('Token attribution needs a [1, T, D] token-embedding output; this '
                                           f'architecture produces {tuple(e.shape) if e is not None else "none"}.')
                    try:
                        g, = torch.autograd.grad(logp[target], e)
                        grad_x_input = (g * e).sum(-1)[0].detach().float()
                        grad_norm = g[0].norm(dim=-1).detach().float()
                    except RuntimeError as exc:
                        grad_error = str(exc).split('\n')[0]
            finally:
                handle.remove()
            base_logp = float(logp[target])
            baseline = emb.weight.detach().mean(0)
            ablation = torch.zeros(T)
            chunk = 32 if T <= 256 else 16
            with torch.no_grad():
                for start in range(0, pos + 1, chunk):
                    ps = torch.arange(start, min(pos + 1, start + chunk))
                    n = len(ps)

                    def swap(m, i, o, ps=ps, n=n):
                        o = o.clone()
                        o[torch.arange(n, device=o.device), ps.to(o.device)] = baseline.to(o.dtype)
                        return o
                    h = emb.register_forward_hook(swap)
                    try:
                        out = self._forward(x.expand(n, T).contiguous()).float()
                    finally:
                        h.remove()
                    lp = torch.log_softmax(out.reshape(n, -1, out.shape[-1])[:, -T:][:, pos], -1)[:, target]
                    ablation[ps] = (base_logp - lp).cpu()
        degenerate = grad_x_input is not None and float(grad_x_input.abs().max()) < 1e-4 * max(1e-12, float(grad_norm.max()) * float(e.detach().norm(dim=-1).max()))
        return clean({'position': pos, 'target': target, 'target_piece': self.piece(target), 'logp': base_logp,
                      'ablation': ablation.tolist(), 'grad_x_input': grad_x_input.tolist() if grad_x_input is not None else None,
                      'grad_norm': grad_norm.tolist() if grad_norm is not None else None, 'grad_error': grad_error,
                      'grad_degenerate': degenerate})



# ───────────────────────── benchmark ─────────────────────────
BENCH_GROUP_NAMES = {0: 'All models', 1: 'Ultra (all × options)', 2: 'Fixed-context', 3: 'Classic RNNs',
                     4: 'Modern recurrent', 5: 'Core comparison'}
FITNESS_NAMES = ['Sliding train loss', 'Final train eval', 'Best validation loss']


def bench_options():
    groups = []
    for gid, name in BENCH_GROUP_NAMES.items():
        ids = sorted(lg.MODEL_NAMES) if gid in (0, 1) else list(lg.BENCH_GROUPS[gid])
        groups.append({'id': gid, 'name': name, 'ids': ids, 'ultra': gid == 1})
    origins = {mid: list(lg.MODEL_ORIGINS[mid]) for mid in lg.MODEL_IDS}
    variants = {str(mid): [{k: v for k, v in opt.items()} for opt in opts] for mid, opts in lg.BENCH_VARIANTS.items()}
    return clean({
        'groups': groups, 'origins': origins, 'variants': variants, 'fitness': FITNESS_NAMES,
        'structure_ids': sorted(lg.BENCH_RNN_STRUCTURE_IDS), 'custom_rnn_ids': sorted(lg.BENCH_CUSTOM_RNN_IDS),
        'hierarchical_ids': sorted(set(lg.HIERARCHICAL_MODEL_MIXERS) - {lg.MLP_MODEL_ID}),
        'results_root': os.path.abspath(lg.BENCH_RESULTS_ROOT),
    })


def bench_tasks(ids, variant_text=None, ultra=False, hierarchical=False):
    """The CLI's row list: one row per model and per combination of its variant values
    (variant_text[mid][key] uses the CLI syntax: one value, a comma list or 'all')."""
    import itertools
    variant_text = variant_text or {}
    ids = [int(i) for i in dict.fromkeys(ids)]
    unknown = sorted(set(ids) - set(lg.MODEL_IDS))
    if unknown:
        raise ValueError(f'Unknown model IDs: {unknown}')
    if hierarchical:
        ids = [i for i in ids if i in lg.HIERARCHICAL_MODEL_MIXERS and i != lg.MLP_MODEL_ID]
    if not ids:
        raise ValueError('Choose at least one model' + (' with a MEGABYTE stage adapter.' if hierarchical else '.'))
    tasks = []
    for mid in ids:
        opts = lg.BENCH_VARIANTS.get(mid)
        if not opts:
            tasks.append({'id': mid, 'overrides': {}})
            continue
        choices = []
        for opt in opts:
            if ultra:
                values = list(opt['choices']) if opt['kind'] in {'choice', 'bool'} else [opt['default']]
            else:
                raw = str((variant_text.get(str(mid)) or {}).get(opt['key'], '') or '').strip()
                try:
                    values = lg._bench_parse_values(opt, raw)
                except ValueError as exc:
                    raise ValueError(f"{lg.MODEL_NAMES[mid]} · {opt['label']}: {exc}")
            choices.append(values)
        for combo in itertools.product(*choices):
            overrides, cell = {}, {}
            for opt, value in zip(opts, combo):
                if value is None:
                    continue
                (cell if opt['cell'] else overrides)[opt['key']] = value
            if cell:
                overrides['rnn_cell_options'] = cell
            tasks.append({'id': mid, 'overrides': overrides})
    tasks.sort(key=lambda t: (lg.MODEL_ORIGINS[t['id']], t['id']))
    return tasks


def bench_settings_from_form(f):
    """The settings dict collect_bench_settings() builds from the CLI answers."""
    import json as _json
    s = {'version': lg.BENCH_SETTINGS_VERSION}
    path = str(Path(str(f.get('dataset_path') or '')).expanduser())
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Dataset file not found: {path or "(empty)"}')
    s['dataset_path'] = path
    s['dataset_type'] = _int(f, 'dataset_type', 0, 0, 1)
    s['tokenizer_mode'] = _int(f, 'tokenizer_mode', 1, -1, 4)
    s['seq2seq'] = seq2seq_from_form(path, f.get('seq2seq') or {}) if s['dataset_type'] == 1 and f.get('seq2seq_on') else None
    model_type = _int(f, 'model_type', 0, 0, 2)
    if model_type:
        stages = f.get('stages') or []
        if len(stages) < 2:
            raise ValueError('MEGABYTE needs at least two stages')
        h = {'model_type': model_type, 'stage_seq_lens': [], 'stage_dims': [], 'stage_child_embed_dims': [],
             'stage_depths': [], 'stage_heads': []}
        for st in stages:
            d = _int(st, 'dim', 128, 1)
            h['stage_seq_lens'].append(_int(st, 'seq_len', 4, 1)); h['stage_dims'].append(d)
            h['stage_child_embed_dims'].append(_int(st, 'child_dim', min(64, d), 1, d))
            h['stage_depths'].append(_int(st, 'depth', 2, 1)); h['stage_heads'].append(_int(st, 'heads', 8, 1))
        s['hierarchy'] = h
    else:
        s['hierarchy'] = {'model_type': 0}
    s['tasks'] = bench_tasks(f.get('models') or [], f.get('variants'), bool(f.get('ultra')), bool(model_type))
    ids = {t['id'] for t in s['tasks']}
    acts = f.get('activations') or {}
    s['activations'] = {}
    for mid in sorted(ids & lg.NON_RNN_ACTIVATION_IDS):
        a = acts.get(str(mid)) or lg.MODEL_DEFAULT_ACTIVATIONS[mid]
        if a not in lg.ACT_NAMES:
            raise ValueError(f'Unknown activation {a}')
        s['activations'][str(mid)] = a
    s['size_mode'], s['target_params'] = 'fixed', 0
    s['embed_dim'], s['layer_count'], s['head_count'], s['seq_len'] = 256, 4, 4, 0
    if not model_type:
        if f.get('size_mode') == 'params':
            s['size_mode'] = 'params'
            s['target_params'] = int(_float(f, 'target_millions', 2.0, 0.0001) * 1e6)
        else:
            s['embed_dim'] = _int(f, 'embed_dim', 256, 1)
        s['layer_count'] = _int(f, 'layer_count', 4, 1)
        s['head_count'] = _int(f, 'head_count', 4, 1)
        if s['dataset_type'] == 0:
            s['seq_len'] = _int(f, 'seq_len', 128, 1)
    s['batch_size'] = _int(f, 'batch_size', 32, 1)
    if f.get('limit_mode') == 'seconds':
        s['total_iters'], s['max_seconds'] = 0, _float(f, 'max_seconds', 60.0, 1e-6)
    else:
        s['total_iters'], s['max_seconds'] = _int(f, 'total_iters', 500, 1), None
    s['min_iters_per_sec'] = _float(f, 'min_iters_per_sec', 0.0, 0.0)
    s['speed_warmup_steps'] = _int(f, 'speed_warmup_steps', 5, 0)
    s['line'] = {'sample_lines': _int(f, 'sample_lines', 3, 1) if s['dataset_type'] == 1 else 1}
    tb = bool(f.get('use_tbptt'))
    s['tbptt'] = {'enabled': tb, 'window': _int(f, 'bptt_window', 64, 1) if tb else 0,
                  'total_len': _int(f, 'tbptt_total_len', 0, 0) if tb and s['dataset_type'] == 0 else 0}
    cuda = lg.DEVICE == 'cuda'
    perf = {'use_amp': bool(f.get('use_amp')) and cuda, 'amp_dtype': 'bf16' if f.get('amp_dtype') == 'bf16' else 'fp16',
            'use_compile': bool(f.get('use_compile')) and cuda}
    if perf['use_compile']:
        perf['compile_backend'] = 'inductor' if f.get('compile_backend') == 'inductor' else 'aot_eager'
    s['perf'] = perf
    s['optim'] = optimizer_from_form(f.get('optimizer', 24), f.get('optim_params'))
    s['lr_schedule'] = _scheduler(f.get('lr_scheduler'))
    s['warmup_steps'] = _int(f, 'warmup_steps', 100, 0) if s['lr_schedule'] == 'cosine_warmup' else 0
    s['grad_clip'] = _float(f, 'grad_clip', 1.0, 0.0)
    s['lr_multipliers'], s['lr_search'] = [1.0], None
    lr_mode = f.get('lr_mode', 'single') if 'lr' in s['optim']['optim_params'] else 'single'
    if lr_mode == 'search':
        base = float(s['optim']['optim_params']['lr'] or 1e-3)
        lo, hi = _float(f, 'lr_min', base / 10, 1e-12), _float(f, 'lr_max', base * 10, 1e-12)
        if not lo < hi:
            raise ValueError('LR search needs minimum < maximum')
        halvings = _int(f, 'lr_halvings', 4, 1)
        s['lr_search'] = {'min': lo, 'max': hi, 'halvings': halvings, 'steps': lg.lr_search_steps(halvings)}
    elif lr_mode == 'multipliers':
        try:
            values = [float(x) for x in str(f.get('lr_multipliers') or '').split(',') if x.strip()] or [1.0]
        except ValueError:
            raise ValueError('LR multipliers must be numbers separated by commas')
        if not all(v > 0 for v in values):
            raise ValueError('LR multipliers must be positive')
        s['lr_multipliers'] = list(dict.fromkeys(values))
    s['rnn'] = {}
    if ids & lg.BENCH_RNN_STRUCTURE_IDS:
        res_every = _int(f, 'res_every', 0, 0)
        s['rnn'] = {'res_every': res_every, 'res_type': _int(f, 'res_type', 0, 0, 3) if res_every else 0,
                    'use_norm': _int(f, 'use_norm', 2, 0, 6), 'dropout': _float(f, 'dropout', 0.0, 0.0),
                    'use_multiplier': _int(f, 'use_multiplier', 0, 0, 2),
                    'rnn_ffn': _int(f, 'rnn_ffn', 0, 0, 3) if ids & lg.BENCH_CUSTOM_RNN_IDS else 0}
    s['nan_skip'] = bool(f.get('nan_skip', True))
    s['fitness_mode'] = _int(f, 'fitness_mode', 2, 0, 2)
    s['val'] = {'val_split': 0.0}
    if s['fitness_mode'] == 2:
        if s['dataset_type'] == 0 and f.get('val_mode') == 'file':
            vp = str(Path(str(f.get('val_path') or '')).expanduser())
            if not os.path.isfile(vp):
                raise FileNotFoundError(f'Validation file not found: {vp}')
            s['val']['classic_val_path'] = vp
        else:
            s['val']['val_split'] = _float(f, 'val_split', 0.1, 0.0)
            if not 0 < s['val']['val_split'] < 1:
                raise ValueError('The validation split must be in (0, 1) for validation-loss scoring')
        s['val']['_val_freq'] = _int(f, 'val_freq', 100, 1)
        s['val']['_val_samples'] = _int(f, 'val_samples', 1000, 1)
    s['seeds'] = _int(f, 'seeds', 1, 1)
    s['sample_len'] = _int(f, 'sample_len', 200, 0)
    s['sample_temperature'] = _float(f, 'sample_temperature', 0.8, 0.0) if s['sample_len'] else 0.8
    s['table_order'] = _int(f, 'table_order', 2, 0, 2)
    # a JSON round trip gives exactly what a preset file or results.json would load as
    return lg._bench_normalize_settings(_json.loads(_json.dumps(lg._bench_json_safe(s))))


def bench_record_view(r):
    keep = ('key', 'id', 'name', 'year', 'month', 'score', 'score_std', 'scores', 'status', 'best_step', 'lr', 'lr_mult',
            'params', 'embed_dim', 'matched_params', 'it_s', 'curve', 'val_curve', 'sample', 'tbptt', 'lr_candidates', 'overrides')
    out = {k: r.get(k) for k in keep}
    spec = lg.MODEL_SPECS.get(r['id'])
    if spec is not None:
        out.update(family=spec.menu_group, short=lg._bench_short_name(r), stateful=spec.stateful, attention=spec.attention)
    return out


def bench_summary(s):
    lr_tries = s['lr_search']['steps'] + 2 if s.get('lr_search') else len(s['lr_multipliers'])
    return {'rows': len(s['tasks']), 'runs_per_row': s['seeds'] * lr_tries, 'metric': FITNESS_NAMES[s['fitness_mode']],
            'fitness_mode': s['fitness_mode'], 'limit': (f"{s['total_iters']} steps/model" if s['max_seconds'] is None
                                                          else f"{s['max_seconds']:g} s/model"),
            'dataset': s['dataset_path'], 'optimizer': s['optim']['optimizer'], 'batch_size': s['batch_size'],
            'seq_len': s['seq_len'], 'size_mode': s['size_mode'], 'target_params': s['target_params'],
            'embed_dim': s['embed_dim'], 'layer_count': s['layer_count'], 'seeds': s['seeds'],
            'lr_mode': 'search' if s.get('lr_search') else 'multipliers' if len(s['lr_multipliers']) > 1 else 'single',
            'model_type': s['hierarchy']['model_type'], 'sample_len': s['sample_len']}


def bench_list(root=None):
    root = Path(root or lg.BENCH_RESULTS_ROOT)
    out = []
    if root.is_dir():
        for d in sorted(root.iterdir(), reverse=True):
            f = d / 'results.json'
            if f.is_file():
                try:
                    import json as _json
                    data = _json.loads(f.read_text(encoding='utf-8'))
                    n = len(data.get('settings', {}).get('tasks', []))
                    out.append({'path': str(d.resolve()), 'name': d.name, 'rows': n, 'done': len(data.get('records', [])),
                                'dataset': data.get('settings', {}).get('dataset_path'), 'modified': f.stat().st_mtime})
                except Exception as exc:
                    out.append({'path': str(d.resolve()), 'name': d.name, 'error': str(exc)})
    return {'root': str(root.resolve()), 'runs': out[:200]}


def bench_load(path):
    """A finished (or interrupted) benchmark folder, for viewing."""
    p, data = lg._bench_read_json(path)
    s = lg._bench_normalize_settings(data['settings'])
    records = [lg._bench_json_restore(r) for r in data.get('records', [])]
    return clean({'path': str(p.parent), 'settings': s, 'summary': bench_summary(s),
                  'records': [bench_record_view(r) for r in records],
                  'prompt': (p.parent / 'samples.txt').read_text(encoding='utf-8').split('\n\n')[0] if (p.parent / 'samples.txt').exists() else ''})


class BenchmarkSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.reset()

    def reset(self):
        self.state, self.error, self.detail = 'idle', None, ''
        self.s, self.records, self.run_dir, self.prompt = None, [], None, ''
        self.current, self.live, self.row_times = None, None, []
        self.started = self.finished = None
        self.stop_requested, self.console_start = False, 0

    def start(self, spec):
        with self.lock:
            if self.state in ('preparing', 'running', 'stopping'):
                raise RuntimeError('A benchmark is already running.')
        mode = spec.get('mode', 'new')
        records = []
        if mode == 'resume':
            path, data = lg._bench_read_json(spec['path'])
            legacy = data['settings'].get('version') == 1
            s = lg._bench_normalize_settings(data['settings'])
            records = [lg._bench_json_restore(r) for r in data.get('records', [])]
            if legacy:
                for r in records:
                    r['id'] = lg._bench_legacy_id(r['id'])
                    r['key'] = lg._bench_task_key(r)
            run_dir = path.parent
        else:
            if mode == 'preset':
                _, data = lg._bench_read_json(spec['path'])
                s = lg._bench_normalize_settings(data.get('settings', data))
            else:
                s = bench_settings_from_form(spec.get('config') or {})
            run_dir = Path(lg.BENCH_RESULTS_ROOT) / time.strftime('bench_%Y%m%d_%H%M%S')
        with self.lock:
            self.reset()
            self.s, self.records, self.run_dir = s, records, run_dir
            self.state, self.started = 'preparing', time.time()
            self.console_start = console().count
        self.thread = threading.Thread(target=self._run, daemon=True, name='linegen-benchmark')
        self.thread.start()
        return self.status()

    def stop(self):
        with self.lock:
            if self.state in ('preparing', 'running'):
                self.stop_requested, self.state = True, 'stopping'
                if self.live:
                    self.live['stats']['stop'] = True
        return self.status()

    def _hook(self, run, stats):
        with self.lock:
            if self.stop_requested:
                stats['stop'] = True
            if self.current is not None:
                self.current['runs_started'] += 1
            self.live = {'run': run, 'stats': stats, 'started': time.time()}

    def _run(self):
        tee = console()
        tee.threads.add(threading.get_ident())
        lg.BENCH_RUN_HOOK = self._hook
        try:
            s = self.s
            self._set(detail='preparing the vocabulary and datasets')
            env = lg._bench_prepare(s)
            self._set(prompt=env['prompt_text'], state='running' if not self.stop_requested else 'stopping', detail='')
            lg._bench_save(self.run_dir, s, self.records, env['prompt_text'])
            done = {r['key'] for r in self.records}
            for n, task in enumerate(s['tasks']):
                if self.stop_requested:
                    break
                key = lg._bench_task_key(task)
                if key in done:
                    continue
                cfg = lg._bench_task_cfg(s, task, env['cfg_base'])
                with self.lock:
                    self.current = {'index': n, 'id': task['id'], 'name': lg._bench_model_name(s, task, cfg),
                                    'year': lg.MODEL_ORIGINS[task['id']][0], 'started': time.time(), 'runs_started': 0}
                    self.live = None
                t0 = time.time()
                record = lg._bench_run_task(s, task, env)
                stopped = any(c['status'] == 'STOPPED' for c in record.get('lr_candidates', [])) or record['status'] == 'STOPPED'
                if stopped and self.stop_requested:
                    break   # an interrupted row is not a result; resuming reruns it
                with self.lock:
                    self.records.append(record)
                    self.row_times.append(time.time() - t0)
                done.add(key)
                lg._bench_save(self.run_dir, s, self.records, env['prompt_text'])
            self._set(state='stopped' if self.stop_requested else 'done', finished=time.time(), current=None, live=None,
                      detail=f'saved {self.run_dir}')
        except BaseException as exc:
            traceback.print_exc()
            self._set(state='error', error=f'{type(exc).__name__}: {exc}', finished=time.time(), current=None, live=None)
        finally:
            lg.BENCH_RUN_HOOK = None
            tee.threads.discard(threading.get_ident())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def status(self, known=0, console_since=None):
        with self.lock:
            s = self.s
            out = {'state': self.state, 'error': self.error, 'detail': self.detail,
                   'run_dir': str(self.run_dir.resolve()) if self.run_dir else None, 'prompt': self.prompt,
                   'summary': bench_summary(s) if s else None, 'total_rows': len(s['tasks']) if s else 0,
                   'done_rows': len(self.records), 'elapsed': (self.finished or time.time()) - self.started if self.started else 0}
            known = max(0, int(known or 0))
            out['records_from'] = min(known, len(self.records))
            out['records'] = [bench_record_view(r) for r in self.records[known:]]
            if self.row_times and s:
                remaining = len(s['tasks']) - len(self.records)
                out['eta'] = remaining * (sum(self.row_times) / len(self.row_times))
            cur = dict(self.current) if self.current else None
            if cur and self.live:
                st = self.live['stats']
                cur.update(seed=self.live['run'].get('seed'), lr=self.live['run'].get('lr'),
                           params=self.live['run'].get('params'), steps=st.get('steps', 0), it_s=st.get('it_s', 0.0),
                           curve=list(st.get('curve', [])), val_curve=list(st.get('val_curve', [])),
                           run_elapsed=time.time() - self.live['started'])
            if cur:
                cur['elapsed'] = time.time() - cur['started']
            out['current'] = cur
        if console_since is not None:
            count, lines = console().tail(max(int(console_since), self.console_start))
            out['console'], out['console_total'] = lines[-300:], count
        return clean(out)


class SpeedSession:
    """linegen's Speed mode: training-step throughput per model on random tokens,
    with the saved run's config (textgen.json) as the template, as in the CLI."""

    def __init__(self):
        self.lock = threading.Lock()
        self.state, self.error, self.results, self.current, self.total = 'idle', None, [], None, 0
        self.stop_requested, self.settings = False, None

    def start(self, spec):
        with self.lock:
            if self.state == 'running':
                raise RuntimeError('A speed test is already running.')
        if not os.path.exists(lg.CONFIG_PATH):
            raise FileNotFoundError(f'The speed test uses the saved run config ({lg.CONFIG_PATH}); train a model first.')
        cfg = lg.load_run_config()
        ids = [int(i) for i in spec.get('models') or []]
        unknown = sorted(set(ids) - set(lg.MODEL_IDS))
        if unknown or not ids:
            raise ValueError(f'Unknown model IDs: {unknown}' if unknown else 'Choose at least one model.')
        settings = {'seq_len': _int(spec, 'seq_len', cfg['seq_len'], 1), 'batch_size': _int(spec, 'batch_size', cfg['batch_size'], 1),
                    'warmup': _int(spec, 'warmup', 10, 0), 'measure': _int(spec, 'measure', 50, 1)}
        with self.lock:
            self.state, self.error, self.results, self.current = 'running', None, [], None
            self.total, self.stop_requested, self.settings = len(ids), False, dict(settings, config=cfg.get('dataset_path'))
        threading.Thread(target=self._run, args=(cfg, ids, settings), daemon=True, name='linegen-speed').start()
        return self.status()

    def _run(self, cfg, ids, st):
        try:
            vocab = lg.load_or_make_vocab(cfg, cfg['dataset_path'], save_config=False)
            for msel in ids:
                if self.stop_requested:
                    break
                name = lg.MODEL_NAMES.get(msel, f'Model {msel}')
                with self.lock:
                    self.current = name
                cfg_t = dict(cfg, model_selection=msel, seq_len=st['seq_len'], batch_size=st['batch_size'])
                row = {'id': msel, 'name': name, 'family': lg.MODEL_SPECS[msel].menu_group, 'year': lg.MODEL_ORIGINS[msel][0]}
                try:
                    row.update(lg.speed_test_model(cfg_t, vocab, st['warmup'], st['measure']), status='OK')
                except Exception as exc:
                    row.update(status=f'FAILED: {exc}'.split('\n')[0][:200], tok_s=0, params=0)
                with self.lock:
                    self.results.append(row)
            with self.lock:
                self.state, self.current = ('stopped' if self.stop_requested else 'done'), None
        except Exception as exc:
            traceback.print_exc()
            with self.lock:
                self.state, self.error, self.current = 'error', f'{type(exc).__name__}: {exc}', None

    def stop(self):
        self.stop_requested = True
        return self.status()

    def status(self):
        with self.lock:
            return clean({'state': self.state, 'error': self.error, 'results': list(self.results), 'current': self.current,
                          'total': self.total, 'settings': self.settings})


def bench_check(form):
    """Rows and run count for a form, without starting anything."""
    s = bench_settings_from_form(form)
    return clean({'summary': bench_summary(s), 'tasks': [{'id': t['id'], 'name': lg.MODEL_NAMES[t['id']] + lg._bench_variant_label(t),
                                                          'year': lg.MODEL_ORIGINS[t['id']][0]} for t in s['tasks']]})


def bench_save_preset(form, path):
    s = bench_settings_from_form(form)
    import json as _json
    p = Path(str(path)).expanduser()
    if p.is_dir():
        p = p / 'benchmark_preset.json'
    p.write_text(_json.dumps(lg._bench_json_safe(s), indent=1), encoding='utf-8')
    return {'path': str(p.resolve())}


# ───────────────────────── server ─────────────────────────
def run_gui(host='127.0.0.1', port=8767, open_browser=True):
    import json
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    console()
    session, sampler, bench, speed = TrainingSession(), Sampler(), BenchmarkSession(), SpeedSession()
    routes = {
        '/api/options': lambda b: options(),
        '/api/fs': lambda b: list_dir(b.get('path')),
        '/api/pick': lambda b: native_pick(b.get('kind', 'file'), b.get('start'), b.get('title') or 'Choose a file'),
        '/api/dataset/inspect': lambda b: inspect_dataset(b['path'], b.get('delimiter'), b.get('has_header', True)),
        '/api/train/check': lambda b: check_config(b.get('config') or {}),
        '/api/train/start': lambda b: session.start(b),
        '/api/train/stop': lambda b: session.stop(),
        '/api/train/status': lambda b: session.status(b.get('since', 0), b.get('console_since')),
        '/api/model/load': lambda b: sampler.load(b.get('config'), b.get('checkpoint')),
        '/api/model/info': lambda b: sampler.info(),
        '/api/tokenize': lambda b: sampler.tokenize(b.get('prompt') or {}),
        '/api/sample/start': lambda b: sampler.start_sample(b),
        '/api/sample/status': lambda b: sampler.sample_status(b),
        '/api/sample/stop': lambda b: sampler.stop_sample(),
        '/api/analyze': lambda b: sampler.analyze(b),
        '/api/analyze/layer': lambda b: sampler.layer_detail(b),
        '/api/analyze/position': lambda b: sampler.position_units(b),
        '/api/analyze/attention': lambda b: sampler.attention_detail(b),
        '/api/analyze/unit': lambda b: sampler.unit_values(b),
        '/api/analyze/state': lambda b: sampler.state_detail(b),
        '/api/influence': lambda b: sampler.influence(b),
        '/api/bench/options': lambda b: bench_options(),
        '/api/bench/check': lambda b: bench_check(b.get('config') or {}),
        '/api/bench/preset': lambda b: bench_save_preset(b.get('config') or {}, b['path']),
        '/api/bench/start': lambda b: bench.start(b),
        '/api/bench/stop': lambda b: bench.stop(),
        '/api/bench/status': lambda b: bench.status(b.get('known', 0), b.get('console_since')),
        '/api/bench/list': lambda b: bench_list(b.get('root')),
        '/api/speed/start': lambda b: speed.start(b),
        '/api/speed/stop': lambda b: speed.stop(),
        '/api/speed/status': lambda b: speed.status(),
        '/api/bench/load': lambda b: bench_load(b['path']),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype='application/json'):
            data = body if isinstance(body, bytes) else json.dumps(clean(body), allow_nan=False, default=str).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def _handle(self, body):
            path = urlparse(self.path).path
            if path in ('/', '/index.html'):
                return self._send(200, HTML.read_bytes(), 'text/html; charset=utf-8')
            try:
                if path not in routes:
                    return self._send(404, {'error': f'unknown route {path}'})
                self._send(200, routes[path](body))
            except Exception as exc:
                if not isinstance(exc, (ValueError, KeyError, FileNotFoundError, RuntimeError)):
                    traceback.print_exc()
                self._send(400, {'error': f'{type(exc).__name__}: {exc}'})

        def do_GET(self):
            self._handle({k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()})

        def do_POST(self):
            n = int(self.headers.get('Content-Length') or 0)
            url = urlparse(self.path)
            if url.path == '/api/upload':
                try:
                    name = parse_qs(url.query).get('name', ['upload.bin'])[0]
                    return self._send(200, save_upload(name, self.rfile, n))
                except Exception as exc:
                    return self._send(400, {'error': f'{type(exc).__name__}: {exc}'})
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
    url = f'http://{host}:{server.server_address[1]}/'
    print(f'LineGen GUI running at {url}  (Ctrl+C to stop; working directory {os.getcwd()})', flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopping GUI.')
    finally:
        session.stop()
        server.server_close()


if __name__ == '__main__':
    run_gui()
