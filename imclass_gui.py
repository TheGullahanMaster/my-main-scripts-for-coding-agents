"""Browser GUI for imclass.py: dataset builder, training monitor and model explorer.

Start with `python imclass.py` and choose mode 4/g/gui, or `python imclass_gui.py`.
The server listens on 127.0.0.1 only; everything runs in this process.
"""
from __future__ import annotations

import os

os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')  # before torch is imported (see IMCLASS.md)

import base64
import csv
import hashlib
import io
import json
import math
import threading
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

import imclass as ic

HTML = Path(__file__).with_name('imclass_gui.html')
DEFAULT_SAVE_DIR = 'ImClass'


# ───────────────────────── helpers ─────────────────────────
def clean(obj):
    """NaN/inf -> None and tensors -> lists, recursively (strict JSON)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if torch.is_tensor(obj):
        return clean(obj.tolist())
    return obj


def b64f32(t):
    t = torch.nan_to_num(t.detach().float().cpu().contiguous())
    return base64.b64encode(t.numpy().tobytes()).decode('ascii')


def data_url(image, fmt='PNG'):
    buf = io.BytesIO()
    image.save(buf, fmt)
    return f'data:image/{fmt.lower()};base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def open_image(src):
    """{"path": file} or {"data": data URL} -> loaded PIL image."""
    if not isinstance(src, dict):
        raise ValueError('Image source must be {"path": ...} or {"data": ...}.')
    if src.get('data'):
        raw = src['data'].split(',', 1)[-1]
        image = Image.open(io.BytesIO(base64.b64decode(raw)))
    elif src.get('path'):
        path = Path(src['path']).expanduser().resolve()
        if path.suffix.lower() not in ic.IMAGE_EXTENSIONS:
            raise ValueError(f'Not a supported image: {path}')
        image = Image.open(path)
    else:
        raise ValueError('No image given.')
    image.load()
    return image


def file_hash(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, 'rb') as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                return h.hexdigest()
            h.update(block)


class Thumbs:
    """Small LRU cache of JPEG thumbnails keyed by (path, size, mtime)."""
    def __init__(self, capacity=4000):
        self.cache, self.capacity, self.lock = OrderedDict(), capacity, threading.Lock()

    def get(self, path, size):
        path = Path(path).expanduser().resolve()
        if path.suffix.lower() not in ic.IMAGE_EXTENSIONS or not path.is_file():
            raise ValueError('Not an image file.')
        size = max(16, min(int(size), 512))
        key = (str(path), size, path.stat().st_mtime_ns)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image.thumbnail((size, size))
            if image.mode not in ('RGB', 'L'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                image = image.convert('RGBA')
                background.paste(image, mask=image.getchannel('A'))
                image = background
            buf = io.BytesIO()
            image.save(buf, 'JPEG', quality=82)
        data = buf.getvalue()
        with self.lock:
            self.cache[key] = data
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return data


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
        kind = ('image' if suffix in ic.IMAGE_EXTENSIONS else 'json' if suffix == '.json'
                else 'checkpoint' if suffix == '.pt' else None)
        if kind in kinds:
            files.append({'name': entry.name, 'kind': kind, 'size': entry.stat().st_size})
    return {'path': str(path), 'parent': str(path.parent), 'dirs': dirs, 'files': files}


# ───────────────────────── datasets ─────────────────────────
def inspect_source(path):
    """What a training source contains: class folders, a dataset JSON, or unlabeled images."""
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == '.json':
        data = json.loads(path.read_text())
        if data.get('format') != ic.DATASET_FORMAT:
            raise ValueError(f'{path.name} is not an imclass dataset JSON.')
        records, classes = ic.load_dataset_json(path)
        unlabeled = sum(1 for item in data['items'] if isinstance(item, dict) and item.get('class') in (None, ''))
        missing = sum(1 for item in data['items'] if isinstance(item, dict) and item.get('class') not in (None, '')) - len(records)
        kind = 'json'
    elif path.is_dir():
        try:
            records, classes = ic.scan_classes(path)
            kind, unlabeled, missing = 'folders', 0, 0
        except ValueError as exc:
            images = ic.image_files(path)
            if images:
                return {'path': str(path), 'kind': 'unlabeled', 'images': len(images),
                        'message': f'{len(images)} images but no usable class subfolders ({exc}). '
                                   'Label them on the Dataset page to create a dataset JSON.'}
            raise
    else:
        raise ValueError(f'{path} is neither a folder nor a dataset .json')
    counts = [0] * len(classes)
    samples = {c: [] for c in classes}
    for p, y in records:
        counts[y] += 1
        if len(samples[classes[y]]) < 6:
            samples[classes[y]].append(str(p))
    return {'path': str(path), 'kind': kind, 'classes': classes, 'counts': counts, 'total': len(records),
            'unlabeled': unlabeled, 'missing': max(0, missing), 'samples': samples}


def builder_scan(folder):
    """All images under a folder, with the subfolder each sits in (a label suggestion)."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'Not a folder: {root}')
    images = ic.image_files(root)
    items = []
    for p in images:
        rel = p.relative_to(root)
        items.append({'path': str(p), 'rel': rel.as_posix(), 'folder': rel.parts[0] if len(rel.parts) > 1 else ''})
    suggested = sorted({i['folder'] for i in items if i['folder']})
    return {'root': str(root), 'items': items, 'suggested_classes': suggested}


def builder_load(path):
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text())
    if data.get('format') != ic.DATASET_FORMAT:
        raise ValueError(f'{path.name} is not an imclass dataset JSON.')
    root = Path(data.get('root') or path.parent).expanduser()
    root = (path.parent / root).resolve() if not root.is_absolute() else root.resolve()
    items = []
    for item in data['items']:
        p = Path(str(item['path'])).expanduser()
        p = (p if p.is_absolute() else root / p).resolve()
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            rel = str(p)
        items.append({'path': str(p), 'rel': rel, 'class': item.get('class'), 'missing': not p.is_file(),
                      'folder': Path(rel).parts[0] if len(Path(rel).parts) > 1 else ''})
    return {'root': str(root), 'path': str(path), 'classes': [str(c) for c in data.get('classes', [])], 'items': items}


def builder_save(path, classes, items, root=None):
    classes = [str(c).strip() for c in classes]
    if len(set(classes)) != len(classes) or any(not c for c in classes):
        raise ValueError('Class names must be unique and non-empty.')
    pairs = []
    for item in items:
        label = item.get('class')
        if label not in (None, '') and label not in classes:
            raise ValueError(f'Item labeled with unknown class {label!r}')
        pairs.append((item['path'], label if label not in (None, '') else None))
    target = Path(path).expanduser().resolve()
    if target.suffix.lower() != '.json':
        raise ValueError('Dataset file name must end in .json')
    ic.save_dataset_json(target, classes, pairs, root)
    labeled = sum(1 for _, label in pairs if label)
    return {'path': str(target), 'items': len(pairs), 'labeled': labeled}


def duplicate_groups(paths):
    """Groups of paths whose file contents are identical."""
    by_hash = {}
    for p in paths:
        try:
            by_hash.setdefault(file_hash(p), []).append(p)
        except OSError:
            continue
    return [group for group in by_hash.values() if len(group) > 1]


def dedupe_records(records, report, key='duplicates_removed'):
    """Keep the first of every set of byte-identical images; returns (records, hashes)."""
    kept, hashes, seen = [], [], set()
    for p, y in records:
        h = file_hash(p)
        if h in seen:
            report[key] = report.get(key, 0) + 1
            continue
        seen.add(h)
        kept.append((p, y))
        hashes.append(h)
    return kept, hashes


def read_split(run_dir, classes):
    """Validation records recorded in a run's split.csv (for continuing without leaking)."""
    split = Path(run_dir) / 'split.csv'
    if not split.is_file():
        return None
    val = []
    with split.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['split'] == 'validation' and row['class'] in classes:
                val.append((Path(row['path']), classes.index(row['class'])))
    return val


# ───────────────────────── training ─────────────────────────
class TrainingSession:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.reset()

    def reset(self):
        self.state, self.error, self.detail = 'idle', None, ''
        self.losses, self.epochs, self.summary, self.probe = [], [], None, None
        self.run_dir, self.started, self.finished = None, None, None
        self.stop_requested, self.batches_per_epoch, self.epoch_count = False, 0, 0

    def status(self, since=0, max_points=4000):
        with self.lock:
            since = max(0, min(int(since or 0), len(self.losses)))
            tail = self.losses[since:]
            if len(tail) > max_points:
                k = -(-len(tail) // max_points)
                tail = [(tail[min(i + k, len(tail)) - 1][0], sum(v for _, v in tail[i:i + k]) / len(tail[i:i + k]))
                        for i in range(0, len(tail), k)]
            end = self.finished or time.time()
            return clean({'state': self.state, 'error': self.error, 'detail': self.detail, 'total': len(self.losses),
                          'steps': [s for s, _ in tail], 'losses': [v for _, v in tail], 'epochs': list(self.epochs),
                          'summary': self.summary, 'probe': self.probe, 'run': self.run_dir,
                          'batches_per_epoch': self.batches_per_epoch, 'epoch_count': self.epoch_count,
                          'elapsed': end - self.started if self.started else 0})

    def stop(self):
        with self.lock:
            if self.state in ('preparing', 'training'):
                self.stop_requested, self.state = True, 'stopping'
        return self.status(10 ** 12)

    def start(self, spec):
        with self.lock:
            if self.state in ('preparing', 'training', 'stopping'):
                raise RuntimeError('A training run is already active.')
        options = self._options(spec)
        with self.lock:
            self.reset()
            self.state, self.started = 'preparing', time.time()
        self.thread = threading.Thread(target=self._run, args=(spec, options), daemon=True, name='imclass-training')
        self.thread.start()
        return self.status()

    @staticmethod
    def _options(spec):
        """Validate everything that does not need the data (errors reach the GUI immediately)."""
        mode = spec.get('mode', 'new')
        if mode not in ('new', 'finetune', 'continue'):
            raise ValueError('mode must be new, finetune or continue')
        if not spec.get('source') and mode != 'continue':
            raise ValueError('Choose a training source (class folders or a dataset .json).')
        optimizer_id = int(spec.get('optimizer_id', 1))
        if optimizer_id not in range(len(ic.OPTIMIZER_LRS)):
            raise ValueError('Unknown optimizer.')
        lr = spec.get('lr')
        lr = float(lr) if lr not in (None, '') else ic.OPTIMIZER_LRS[optimizer_id]
        hd = ic.hd_options(spec.get('hd')) if optimizer_id in ic.HD_OPTIMIZERS else None
        if not math.isfinite(lr) or lr < 0 or (lr == 0 and not (hd and hd['max_lr'])):
            raise ValueError('The learning rate must be positive (0 only for the HD optimizers with a max LR).')
        augments = sorted({int(a) for a in spec.get('augments', [])})
        if any(a not in ic.AUGMENTS for a in augments):
            raise ValueError('Unknown augmentation id.')
        cfg = None
        if mode == 'new':
            fields = {k: v for k, v in (spec.get('config') or {}).items() if k in ic.Config.__dataclass_fields__}
            cfg = ic.Config(**fields)
            cfg.validate()
        fraction = float(spec.get('val_fraction', 0.1) or 0)
        if not 0 <= fraction < 1:
            raise ValueError('Validation fraction must be in [0, 1).')
        return dict(mode=mode, cfg=cfg, optimizer_id=optimizer_id, lr=lr, hd=spec.get('hd') if hd else None,
                    augments=augments, fraction=fraction,
                    batch_size=int(spec.get('batch_size', 64)), epochs=int(spec.get('epochs', 10)),
                    seed=int(spec.get('seed', 0) or 0), save_dir=spec.get('save_dir') or DEFAULT_SAVE_DIR,
                    dedupe=bool(spec.get('dedupe', True)))

    def _set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def _run(self, spec, o):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        report = {}
        try:
            seed = ic.seed_everything(o['seed'])
            report['seed'] = seed
            optimizer_state, checkpoint = None, spec.get('checkpoint')
            if o['mode'] == 'continue':
                self._set(detail='loading checkpoint')
                model, classes, saved = ic.load_checkpoint(checkpoint, device)
                optimizer_state = saved.get('optimizer_state')
                training = saved.get('training', {})
                if int(training.get('optimizer_id', o['optimizer_id'])) != o['optimizer_id']:
                    optimizer_state = None  # a different optimizer starts fresh
                source = spec.get('source')
                run_dir = ic.resolve_checkpoint(checkpoint).parent
                if source:
                    records, _ = ic.load_records(source, classes)
                else:  # the run's own images, as recorded in its split.csv
                    if not (run_dir / 'split.csv').is_file():
                        raise ValueError('This run has no split.csv; choose the training source to continue on.')
                    records = []
                    with (run_dir / 'split.csv').open(newline='') as stream:
                        for row in csv.DictReader(stream):
                            if row['class'] in classes:
                                records.append((Path(row['path']), classes.index(row['class'])))
            else:
                self._set(detail='reading dataset')
                records, classes = ic.load_records(spec['source'])
                if o['mode'] == 'finetune':
                    model, _, _ = ic.load_checkpoint(checkpoint, device)
                    head = spec.get('head') or {}
                    model.replace_head(len(classes), [int(d) for d in head.get('dims', model.cfg.head_dims)],
                                       int(head.get('activation', model.cfg.activation)),
                                       float(head.get('slope', model.cfg.slope)), bool(spec.get('freeze', True)))
                else:
                    model = ic.Classifier(o['cfg'], len(classes))
            records = [(Path(p).resolve(), y) for p, y in records]
            report['images_read'] = len(records)
            if o['dedupe']:
                self._set(detail=f'checking {len(records)} images for duplicates')
                records, hashes = dedupe_records(records, report)
            else:
                hashes = None
            # validation: truly held out (no shared file and no identical content)
            val = []
            if o['mode'] == 'continue' and read_split(run_dir, classes):
                val = [(Path(p).resolve(), y) for p, y in read_split(run_dir, classes)]
                report['validation'] = 'the run\'s recorded validation split'
            elif spec.get('val_source'):
                val, _ = ic.load_records(spec['val_source'], classes)
                val = [(Path(p).resolve(), y) for p, y in val]
                report['validation'] = 'separate source'
            elif o['fraction'] > 0:
                records, val = ic.split_records(records, o['fraction'], seed)
                report['validation'] = f'{o["fraction"]:.0%} held out per class'
            if val:
                val_paths = {p for p, _ in val}
                val_hashes = {file_hash(p) for p, _ in val} if o['dedupe'] else set()
                before = len(records)
                records = [(p, y) for p, y in records if p not in val_paths and
                           not (o['dedupe'] and file_hash(p) in val_hashes)]
                report['train_removed_for_validation'] = before - len(records)
            counts = [0] * len(classes)
            for _, y in records:
                counts[y] += 1
            summary = dict(report, classes=classes, train=len(records), val=len(val), counts=counts,
                           model=ic.MODEL_NAMES[model.cfg.model], device=str(device),
                           params=sum(p.numel() for p in model.parameters()))
            self._set(summary=summary, epoch_count=o['epochs'], detail='')
            probe = self._probe_source(spec, val or records)

            def callback(info):
                with self.lock:
                    if info['event'] == 'batch':
                        self.batches_per_epoch = info['batches']
                        step = (info['epoch'] - 1) * info['batches'] + info['batch']
                        self.losses.append((step, info['loss']))
                        self.run_dir = info['run']
                        if self.state == 'preparing':
                            self.state = 'training'
                    elif info['event'] == 'epoch':
                        self.epochs.append(dict(info['row'], best=info['best']))
                    stop = self.stop_requested
                if info['event'] == 'epoch' and probe is not None:
                    try:
                        result = explain_image(info['model'], classes, probe, top=3)
                        with self.lock:
                            self.probe = dict(result, epoch=info['epoch'])
                    except Exception:  # a probe failure must never stop training
                        traceback.print_exc()
                return stop
            run = ic.train_model(model, classes, records, val, batch_size=o['batch_size'], epochs=o['epochs'],
                                 optimizer_id=o['optimizer_id'], seed=seed, augments=o['augments'],
                                 save_dir=o['save_dir'], device=device, lr=o['lr'],
                                 progress_callback=callback, optimizer_state=optimizer_state, hd=o['hd'])
            self._set(state='done', run_dir=str(run))
        except KeyboardInterrupt:
            self._set(state='stopped', detail='stopped; interrupt.pt saved in the run folder')
        except Exception as exc:
            traceback.print_exc()
            self._set(state='error', error=f'{type(exc).__name__}: {exc}')
        finally:
            self._set(finished=time.time())

    @staticmethod
    def _probe_source(spec, records):
        if spec.get('probe'):
            return {'path': spec['probe']}
        return {'path': str(records[0][0])} if records else None


# ───────────────────────── exploring ─────────────────────────
def to_input(model, src):
    image = open_image(src)
    x = ic.Preprocess(model.cfg)(image).unsqueeze(0)
    return x.to(next(model.parameters()).device)


def map_payload(t):
    t = t.detach().float().cpu()
    if t.ndim == 1:
        t = t[None]
    return {'h': t.shape[-2], 'w': t.shape[-1], 'data': b64f32(t)}


def explain_image(model, classes, src, target=None, top=5, maps=False):
    """Prediction + Grad-CAM (and optionally every other map explain() provides) for one image."""
    was_training = model.training
    x = to_input(model, src)
    try:
        logits, cam, extra = ic.explain(model, x, target)
    finally:
        model.train(was_training)
    probs = logits.softmax(-1)[0].cpu()
    order = probs.argsort(descending=True).tolist()
    target = int(order[0]) if target is None else int(target)
    out = {'image': data_url(ic.display_image(x)), 'probs': [[i, classes[i], probs[i].item()] for i in order[:top]],
           'all_probs': probs.tolist(), 'prediction': int(order[0]), 'target': target, 'cam': map_payload(cam)}
    if maps:
        out['maps'] = {name: map_payload(m) for name, m in extra.items()}
    return out


class DreamJob:
    def __init__(self):
        self.lock, self.thread = threading.Lock(), None
        self.state, self.error, self.image, self.step, self.steps, self.values, self.stop_flag = 'idle', None, None, 0, 0, [], False

    def status(self):
        with self.lock:
            return clean({'state': self.state, 'error': self.error, 'image': self.image, 'step': self.step,
                          'steps': self.steps, 'values': list(self.values)})

    def stop(self):
        with self.lock:
            self.stop_flag = True
        return self.status()

    def start(self, model, target, steps, lr, setup, optimizer, starter, suppress, lock):
        with self.lock:
            if self.state == 'running':
                raise RuntimeError('A dream is already running.')
            self.state, self.error, self.image, self.step, self.steps, self.values, self.stop_flag = 'running', None, None, 0, steps, [], False
        every = max(1, steps // 60)

        def callback(step, image, value):
            with self.lock:
                self.step = step
                if value is not None:
                    self.values.append([step, value])
                if step % every == 0 or step == steps:
                    with torch.no_grad():
                        self.image = data_url(ic.display_image(image() * 2 - 1))
                return self.stop_flag

        def work():
            try:
                with lock:
                    x, start, end = ic.dream_one(model, target, steps, lr, starter, 0.0, setup, optimizer,
                                                 suppress, step_callback=callback)
                with self.lock:
                    self.image = data_url(ic.display_image(x))
                    self.state = 'done'
            except Exception as exc:
                traceback.print_exc()
                with self.lock:
                    self.state, self.error = 'error', f'{type(exc).__name__}: {exc}'
        self.thread = threading.Thread(target=work, daemon=True, name='imclass-dream')
        self.thread.start()
        return self.status()


class Explorer:
    def __init__(self):
        self.lock = threading.RLock()
        self.model = self.classes = self.checkpoint = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dream = DreamJob()

    def require(self):
        if self.model is None:
            raise ValueError('Load a checkpoint first.')
        return self.model

    def load(self, path):
        with self.lock:
            model, classes, saved = ic.load_checkpoint(path, self.device)
            self.model, self.classes = model, classes
            self.checkpoint = str(ic.resolve_checkpoint(path))
            return self.info(saved)

    def info(self, saved=None):
        m = self.require()
        cfg = m.cfg
        layers = [{'name': name, 'kind': kind} for name, (_, kind) in ic.feature_layers(m).items()]
        convs = [name for name, mod in m.named_modules() if isinstance(mod, torch.nn.Conv2d)]
        return clean({'checkpoint': self.checkpoint, 'classes': self.classes, 'model': ic.MODEL_NAMES[cfg.model],
                      'config': ic.asdict(cfg), 'layers': layers, 'convs': convs, 'device': str(self.device),
                      'params': sum(p.numel() for p in m.parameters()),
                      'metrics': (saved or {}).get('metrics'), 'epoch': (saved or {}).get('epoch'),
                      'has_split': (Path(self.checkpoint).parent / 'split.csv').is_file()})

    def predict(self, src, target=None):
        with self.lock:
            return explain_image(self.require(), self.classes, src, target, top=len(self.classes), maps=True)

    def compare(self, src, count=4):
        with self.lock:
            m = self.require()
            first = explain_image(m, self.classes, src, top=count)
            panels = [{'class': i, 'name': name, 'prob': p, 'cam': explain_image(m, self.classes, src, i)['cam']}
                      for i, name, p in first['probs']]
            return {'image': first['image'], 'panels': panels}

    def features(self, src, layer, max_units=64):
        with self.lock:
            m = self.require()
            layers = ic.feature_layers(m)
            if layer not in layers:
                raise ValueError(f'Unknown layer {layer}')
            module, kind = layers[layer]
            x = to_input(m, src)
            got = {}
            handle = module.register_forward_hook(lambda mod, i, o: got.setdefault('o', o))
            try:
                with torch.no_grad():
                    m.eval()
                    logits = m(x)
            finally:
                handle.remove()
            out = got['o']
            if out.ndim == 4:
                maps = out[0]
            elif out.ndim == 3:
                grid = m.encoder.grid
                maps = out[0].transpose(0, 1).reshape(-1, grid, grid)
            else:
                maps = out[0].reshape(-1)[:, None, None]
            total = maps.shape[0]
            strength = maps.abs().mean((1, 2))
            order = strength.argsort(descending=True)[:max_units] if max_units and max_units < total else torch.arange(total)
            maps = maps[order].float()
            if maps.shape[-1] > 64:  # keep payloads small; maps are displayed as thumbnails
                maps = F.adaptive_avg_pool2d(maps[None], (64, 64))[0]
            return {'layer': layer, 'kind': kind, 'total': total, 'units': order.tolist(), 'n': maps.shape[0],
                    'h': maps.shape[1], 'w': maps.shape[2], 'data': b64f32(maps), 'scale': maps.abs().max().item(),
                    'means': maps.mean((1, 2)).tolist(), 'prediction': self.classes[int(logits.argmax())]}

    def kernels(self, layer, max_units=64):
        with self.lock:
            m = self.require()
            module = dict(m.named_modules()).get(layer)
            if not isinstance(module, torch.nn.Conv2d):
                raise ValueError(f'{layer} is not a convolution')
            w = module.weight[:max_units].detach().float().cpu()
            rgb = w.shape[1] == 3
            tiles = w if rgb else w.mean(1, keepdim=True)
            return {'layer': layer, 'n': tiles.shape[0], 'c': tiles.shape[1], 'k': tiles.shape[-1],
                    'total': module.out_channels, 'in_channels': module.in_channels, 'rgb': rgb,
                    'data': b64f32(tiles), 'scale': tiles.abs().max().item()}

    def occlusion(self, src, target=None, patch=None, stride=None):
        """Probability drop when each square of the image is covered with mid-gray."""
        with self.lock:
            m = self.require()
            x = to_input(m, src)
            size = x.shape[-1]
            patch = int(patch or max(1, size // 6))
            stride = int(stride or max(1, patch // 2))
            with torch.no_grad():
                m.eval()
                base = m(x).softmax(-1)[0]
                target = int(base.argmax()) if target is None else int(target)
                spots = [(t, l) for t in range(0, size - patch + 1, stride) for l in range(0, size - patch + 1, stride)]
                drops = []
                for i in range(0, len(spots), 64):
                    batch = x.repeat(len(spots[i:i + 64]), 1, 1, 1)
                    for j, (t, l) in enumerate(spots[i:i + 64]):
                        batch[j, :, t:t + patch, l:l + patch] = 0.0
                    drops.append(base[target] - m(batch).softmax(-1)[:, target])
                drops = torch.cat(drops).cpu()
            n = (size - patch) // stride + 1
            heat = torch.zeros(size, size)
            count = torch.zeros(size, size)
            for d, (t, l) in zip(drops, spots):
                heat[t:t + patch, l:l + patch] += d
                count[t:t + patch, l:l + patch] += 1
            heat = heat / count.clamp_min(1)
            return {'target': target, 'name': self.classes[target], 'base': base[target].item(), 'patch': patch,
                    'stride': stride, 'grid': n, 'map': map_payload(heat), 'image': data_url(ic.display_image(x))}

    def dream_targets(self):
        m = self.require()
        classes = [{'module': 'head.logits', 'unit': i, 'axis': -1, 'label': f'class {c}'} for i, c in enumerate(self.classes)]
        head = [{'module': f'head.hidden.{l}', 'unit': i, 'axis': -1, 'label': f'head layer {l} neuron {i}'}
                for l, mod in enumerate(m.head.hidden) for i in range(mod.out_dim)]
        encoder = []
        for name, module in m.encoder.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                encoder.append({'module': f'encoder.{name}', 'count': module.out_channels, 'axis': 1})
            elif isinstance(module, torch.nn.Linear):
                encoder.append({'module': f'encoder.{name}', 'count': module.out_features, 'axis': -1})
        return {'classes': classes, 'head': head, 'encoder': encoder}

    def start_dream(self, spec):
        m = self.require()
        target = ic.DreamTarget(spec['module'], int(spec['unit']), int(spec.get('axis', -1)), 'gui')
        if target.module not in dict(m.named_modules()):
            raise ValueError(f'Unknown module {target.module}')
        steps, lr = int(spec.get('steps', 100)), float(spec.get('lr', 0.05))
        setup, optimizer = int(spec.get('setup', 0)), int(spec.get('optimizer', 0))
        ic.validate_dream_choices(setup, optimizer)
        starter = to_input(m, spec['starter']) if spec.get('starter') else None
        return self.dream.start(m, target, steps, lr, setup, optimizer, starter, bool(spec.get('suppress', False)), self.lock)

    def evaluate(self, source=None, limit=3000):
        """Confusion matrix and misclassified images on the run's validation split (or a given source)."""
        with self.lock:
            m = self.require()
            if source:
                records, _ = ic.load_records(source, self.classes)
                label = str(source)
            else:
                records = read_split(Path(self.checkpoint).parent, self.classes)
                if not records:
                    raise ValueError('This run recorded no validation split; choose a folder or dataset .json to evaluate.')
                label = 'the run\'s validation split'
            records = records[:limit]
            loader = torch.utils.data.DataLoader(ic.ImageDataset(records, ic.Preprocess(m.cfg)), batch_size=64)
            k = len(self.classes)
            confusion = torch.zeros(k, k, dtype=torch.long)
            wrong, index = [], 0
            m.eval()
            with torch.no_grad():
                for x, y in loader:
                    probs = m(x.to(self.device)).softmax(-1).cpu()
                    pred = probs.argmax(1)
                    for j in range(len(y)):
                        confusion[y[j], pred[j]] += 1
                        if pred[j] != y[j]:
                            wrong.append({'path': str(records[index + j][0]), 'true': self.classes[y[j]],
                                          'pred': self.classes[pred[j]], 'confidence': probs[j, pred[j]].item(),
                                          'true_prob': probs[j, y[j]].item()})
                    index += len(y)
            wrong.sort(key=lambda w: -w['confidence'])
            per_class = (confusion.diag().float() / confusion.sum(1).clamp_min(1)).tolist()
            return clean({'source': label, 'n': index, 'accuracy': confusion.diag().sum().item() / max(1, index),
                          'balanced_accuracy': sum(per_class) / k, 'per_class': per_class,
                          'confusion': confusion.tolist(), 'classes': self.classes, 'wrong': wrong[:200]})

    def random_image(self, split='validation'):
        m = self.require()
        records = read_split(Path(self.checkpoint).parent, self.classes) if split == 'validation' else None
        if not records:
            raise ValueError('No recorded validation images for this run.')
        p, y = records[int(torch.randint(len(records), ()))]
        return {'path': str(p), 'label': self.classes[y]}


def list_runs(save_dir=DEFAULT_SAVE_DIR):
    root = Path(save_dir).expanduser().resolve()
    if not root.is_dir():
        return {'root': str(root), 'runs': []}
    runs = []
    for run in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        checkpoints = [name for name in ('best.pt', 'last.pt', 'interrupt.pt') if (run / name).is_file()]
        if not checkpoints:
            continue
        info = {'path': str(run), 'name': run.name, 'checkpoints': checkpoints, 'mtime': run.stat().st_mtime}
        try:
            cfg = json.loads((run / 'config.json').read_text())
            info.update(model=ic.MODEL_NAMES[cfg['config']['model']], classes=len(cfg.get('classes', [])),
                        size=cfg['config'].get('crop'), epochs=cfg.get('epochs'))
        except (OSError, ValueError, KeyError, IndexError):
            pass
        history = run / 'history.csv'
        if history.is_file():
            rows = list(csv.DictReader(history.open(newline='')))
            if rows:
                last = rows[-1]
                info['last'] = {k: float(v) for k, v in last.items() if v not in ('', None)}
                vals = [float(r['val_accuracy']) for r in rows if r.get('val_accuracy')]
                if vals:
                    info['best_val_accuracy'] = max(vals)
        runs.append(info)
    return {'root': str(root), 'runs': runs}


def augment_preview(spec, count=8):
    """What training batches look like: one image through the chosen augmentations, several times."""
    fields = {k: v for k, v in (spec.get('config') or {}).items() if k in ic.Config.__dataclass_fields__}
    cfg = ic.Config(**fields)
    cfg.validate()
    augments = sorted({int(a) for a in spec.get('augments', [])})
    image = open_image({'path': spec['path']})
    transform = ic.Preprocess(cfg, True, augments)
    return {'images': [data_url(ic.display_image(transform(image))) for _ in range(count)],
            'note': 'MixUp and CutMix mix pairs inside a batch and are not shown here.' if {19, 20} & set(augments) else ''}


def options():
    return {'models': ic.MODEL_NAMES, 'isotropic': sorted(ic.ISOTROPIC), 'augments': ic.AUGMENTS,
            'optimizers': ['SGD + momentum', 'Adam', 'Muon', 'CLion', 'RAdamScheduleFree', 'AdamHD', 'MuonHD', 'NorMuonHD'],
            'hd_optimizers': list(ic.HD_OPTIMIZERS), 'hd_defaults': dict(ic.HD_DEFAULTS),
            'optimizer_lrs': list(ic.OPTIMIZER_LRS), 'defaults': ic.asdict(ic.Config()),
            'activations': ['none', 'Sigmoid', 'Tanh', 'ReLU', 'LeakyReLU', 'PReLU', 'GELU', 'SiLU', 'Mish', 'SwiGLU'],
            'dream_setups': list(ic.DREAM_SETUPS), 'dream_optimizers': list(ic.DREAM_OPTIMIZERS),
            'cwd': os.getcwd(), 'device': 'cuda' if torch.cuda.is_available() else 'cpu'}


# ───────────────────────── server ─────────────────────────
def run_gui(host='127.0.0.1', port=8766, open_browser=True):
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    thumbs, session, explorer = Thumbs(), TrainingSession(), Explorer()
    routes = {
        '/api/options': lambda b: options(),
        '/api/fs': lambda b: list_dir(b.get('path')),
        '/api/source/inspect': lambda b: inspect_source(b['path']),
        '/api/builder/scan': lambda b: builder_scan(b['folder']),
        '/api/builder/load': lambda b: builder_load(b['path']),
        '/api/builder/save': lambda b: builder_save(b['path'], b['classes'], b['items'], b.get('root')),
        '/api/builder/duplicates': lambda b: {'groups': duplicate_groups(b['paths'])},
        '/api/train/start': lambda b: session.start(b),
        '/api/train/stop': lambda b: session.stop(),
        '/api/train/status': lambda b: session.status(b.get('since', 0)),
        '/api/train/augment_preview': lambda b: augment_preview(b),
        '/api/runs': lambda b: list_runs(b.get('save_dir') or DEFAULT_SAVE_DIR),
        '/api/model/load': lambda b: explorer.load(b['path']),
        '/api/model/info': lambda b: explorer.info(),
        '/api/predict': lambda b: explorer.predict(b['image'], b.get('target')),
        '/api/compare': lambda b: explorer.compare(b['image'], int(b.get('count', 4))),
        '/api/features': lambda b: explorer.features(b['image'], b['layer'], int(b.get('max_units', 64))),
        '/api/kernels': lambda b: explorer.kernels(b['layer'], int(b.get('max_units', 64))),
        '/api/occlusion': lambda b: explorer.occlusion(b['image'], b.get('target'), b.get('patch'), b.get('stride')),
        '/api/dream/targets': lambda b: explorer.dream_targets(),
        '/api/dream/start': lambda b: explorer.start_dream(b),
        '/api/dream/status': lambda b: explorer.dream.status(),
        '/api/dream/stop': lambda b: explorer.dream.stop(),
        '/api/evaluate': lambda b: explorer.evaluate(b.get('source')),
        '/api/random_image': lambda b: explorer.random_image(b.get('split', 'validation')),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype='application/json'):
            data = body if isinstance(body, bytes) else json.dumps(clean(body), allow_nan=False, default=str).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store' if ctype == 'application/json' else 'max-age=300')
            self.end_headers()
            self.wfile.write(data)

        def _handle(self, body):
            path = urlparse(self.path).path
            if path in ('/', '/index.html'):
                return self._send(200, HTML.read_bytes(), 'text/html; charset=utf-8')
            try:
                if path == '/api/thumb':
                    return self._send(200, thumbs.get(body['path'], body.get('size', 128)), 'image/jpeg')
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
    print(f'ImClass GUI running at {url}  (Ctrl+C to stop; working directory {os.getcwd()})', flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopping GUI.')
    finally:
        server.server_close()


if __name__ == '__main__':
    run_gui()
