"""Training workflow regressions. Run with the pytorch2 environment."""
import contextlib
import copy
import io
import json
import random
import os
import signal
import subprocess
import sys
import select
import time
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
import torch
from torch import nn

import jit5 as jit


def config(**updates):
    cfg = dict(width=4, height=4, channels=3, dim=16, depth=2, heads=2,
               p_width=2, p_height=2, model_type='jit', batch_size=2,
               steps=4, lr=.001, optimizer_type='adam', dataset_path='unused',
               sampling_steps=2, use_amp=False, use_ema=True, warmup_steps=0,
               sample_every=100, save_every=100, num_workers=0, dropout=.1)
    cfg.update(updates)
    jit.apply_config_defaults(cfg)
    return cfg


def images(root, n=9):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        values = np.random.default_rng(i).integers(0, 256, (8, 8, 3), dtype=np.uint8)
        Image.fromarray(values).save(root / f'{i:02}.png')


class MeanLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.3))

    def p_losses(self, data, **kwargs):
        return (data * self.weight - 1).square().mean()


class TrainingTests(unittest.TestCase):
    def test_optional_token_grn_for_flat_backbones(self):
        grn = jit.TokenGRN(4)
        x = torch.randn(2, 3, 4)
        torch.testing.assert_close(grn(x), x)
        with torch.no_grad():
            grn.gamma.fill_(.5)
            grn.beta.fill_(.1)
        response = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        expected = x + .5 * x * response / (response.mean(-1, keepdim=True) + 1e-6) + .1
        torch.testing.assert_close(grn(x), expected)

        for kind in sorted(jit.GRN_TOKEN_MODELS):
            with self.subTest(model_type=kind):
                cfg = config(model_type=kind, depth=1, use_grn=True, dropout=0.)
                jit.validate_config(cfg)
                with contextlib.redirect_stdout(io.StringIO()):
                    model = jit.build_model(cfg, 'cpu')
                self.assertTrue(all(isinstance(block, jit.GRNResidualBlock) for block in model.layers))
                output = model(torch.randn(2, 3, 4, 4), torch.rand(2))
                output.square().mean().backward()
                for block in model.layers:
                    self.assertTrue(torch.isfinite(block.grn.gamma.grad).all())
                    self.assertTrue(torch.isfinite(block.grn.beta.grad).all())

    def test_fixed_attention_head_width_validation_and_forward(self):
        cases = [dict(model_type='jit', dim=256, heads=128)]
        cases += [dict(model_type=kind, dim=16, heads=3)
                  for kind in ('jit', 'fullattn', 'mixer_attn')]
        cases.append(dict(model_type='hiermlp', dim=16, hier_global_mixer='jit',
                          hier_global_dim=16, hier_global_heads=3,
                          hier_global_depth=1, hier_input_grid_size=2,
                          hier_output_grid_size=2))
        for updates in cases:
            with self.subTest(**updates):
                cfg = config(depth=1, dropout=0., **updates)
                jit.validate_config(cfg)
                with contextlib.redirect_stdout(io.StringIO()):
                    model = jit.build_model(cfg, 'cpu')
                attention = [m for m in model.modules() if isinstance(m, jit.Attention)]
                self.assertTrue(attention)
                for layer in attention:
                    self.assertEqual(layer.to_qkv.out_features, 3 * layer.heads * 64)
                    self.assertEqual(layer.to_out.in_features, layer.heads * 64)
                y = model(torch.randn(1, 3, 4, 4), torch.tensor([.5]))
                self.assertEqual(y.shape, (1, 3, 4, 4))
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(y).all())
                for layer in attention:
                    self.assertTrue(torch.isfinite(layer.to_qkv.weight.grad).all())
        for kind in ('rin', 'xcit', 'nat', 'volo'):
            with self.subTest(split_width_model=kind), self.assertRaises(jit.ConfigError):
                jit.validate_config(config(model_type=kind, dim=16, heads=3))

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_boolean_defaults_and_json_types(self):
        for default in (False, 0, '0', 'false', 'off'):
            with mock.patch('builtins.input', return_value=''):
                self.assertIs(jit.get_input('toggle', default, bool), False)
        for default in (True, 1, '1', 'true'):
            with mock.patch('builtins.input', return_value=''):
                self.assertIs(jit.get_input('toggle', default, bool), True)
        with self.assertRaises(jit.ConfigError):
            jit.validate_config(config(use_flip='0'))
        with self.assertRaises(jit.ConfigError):
            jit.validate_config(config(validation_path='val', validation_percent=20))

    def test_config_roundtrip_retry_and_legacy(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            jit.save_config(cfg, path)
            self.assertEqual(jit.load_config(path), cfg)
            legacy = Path(directory) / 'config.pt'
            torch.save(cfg, legacy)
            self.assertEqual(jit.load_config(legacy), cfg)
        with mock.patch('builtins.input', side_effect=['dim=32', 'model_type=rin',
                'use_ema=false', 'synthetic_params.noise_min_sigma=2', '']), \
                contextlib.redirect_stdout(io.StringIO()):
            edited = jit.edit_config(cfg)
        self.assertEqual(edited['dim'], 32)
        self.assertEqual(edited['model_type'], 'rin')
        self.assertFalse(edited['use_ema'])
        self.assertEqual(edited['synthetic_params']['noise_min_sigma'], 2)
        self.assertEqual(cfg['dim'], 16)
        example = jit.load_config(Path(__file__).with_name('jit5_config.example.json'))
        jit.apply_config_defaults(example)
        jit.validate_config(example)

    def test_separate_splits_and_corrupt_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            images(directory)
            cfg = config(dataset_path=directory, validation_percent=33)
            train, val = jit.build_training_datasets(cfg)
            self.assertEqual((len(train), len(val)), (6, 3))
            self.assertFalse(set(train.paths) & set(val.paths))
            again, held = jit.build_training_datasets(cfg)
            self.assertEqual(train.paths, again.paths)
            self.assertEqual(val.paths, held.paths)
            for path in val.paths:
                Path(path).write_bytes(b'broken')
            with self.assertRaises(RuntimeError):
                val[0]  # Must not recover by reading a training image.

    def test_class_pair_and_explicit_folder_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in ('a', 'b'):
                images(root / label, 4)
            cfg = config(dataset_path=str(root), conditioning_mode='class',
                         class_names=['a', 'b'], validation_percent=25)
            train, val = jit.build_training_datasets(cfg)
            self.assertEqual([label for _, label in val.samples], [0, 1])
            self.assertEqual(len(train), 6)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ('train', 'val'):
                for side in ('A', 'B'):
                    images(root / split / side, 4)
            cfg = config(dataset_path=str(root / 'train'), validation_path=str(root / 'val'),
                         conditioning_mode='pix2pix')
            train, val = jit.build_training_datasets(cfg)
            self.assertEqual((len(train), len(val)), (4, 4))
            cfg.update(validation_path='', validation_percent=25)
            train, val = jit.build_training_datasets(cfg)
            self.assertEqual((len(train), len(val)), (3, 1))
            self.assertFalse(set(train.pairs) & set(val.pairs))
            cfg.update(validation_path=cfg['dataset_path'], validation_percent=0)
            with self.assertRaises(jit.ConfigError):
                jit.build_training_datasets(cfg)

    def test_autocanny_pairs_and_settings(self):
        self.assertEqual(jit.parse_synthetic_source('19'), 'autocanny')
        self.assertEqual(jit.parse_synthetic_source('autocanny'), 'autocanny')
        self.assertNotIn('autocanny', jit.SyntheticPix2PixDataset.EFFECT_MODES)
        with tempfile.TemporaryDirectory() as directory:
            pixels = np.full((32, 32, 3), 255, dtype=np.uint8)
            pixels[8:24, 8:24] = 0
            Image.fromarray(pixels).save(Path(directory) / 'square.png')
            cfg = config(dataset_path=directory, width=32, height=32,
                         conditioning_mode='pix2pix', pix2pix_source_mode='autocanny')
            dataset = jit.build_dataset(cfg)
            target, condition = dataset[0]
            expected = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 127.5 - 1
            torch.testing.assert_close(target, expected)
            self.assertEqual(condition.shape, target.shape)
            self.assertEqual(set(condition.unique().tolist()), {-1., 1.})
            torch.testing.assert_close(condition[0], condition[1])
            torch.testing.assert_close(condition[1], condition[2])
            self.assertEqual(condition[0, 16, 16].item(), 1.)
            self.assertTrue((condition[0, 7:10, 10:22] == -1).any())
            self.assertEqual(condition[0, 0, 0].item(), 1.)
            for sigma in (0, 1.1, float('nan'), True, '0.33'):
                with self.subTest(sigma=sigma), self.assertRaisesRegex(ValueError, 'autocanny_sigma'):
                    jit.build_dataset(dict(cfg, synthetic_params={'autocanny_sigma': sigma}))
        with mock.patch('builtins.input', return_value=''):
            self.assertEqual(jit.prompt_synthetic_params('autocanny'), {'autocanny_sigma': .33})

    def test_exact_stream_with_prefetch_and_synthetic_randomness(self):
        with tempfile.TemporaryDirectory() as directory:
            images(directory)
            cfg = config(dataset_path=directory, conditioning_mode='pix2pix',
                         pix2pix_source_mode='noise,hue', use_flip=True, use_vflip=True,
                         resize_width=8, resize_height=8, num_workers=2)
            ds = jit.build_dataset(cfg)
            stream = jit.TrainingStream(ds, cfg)
            next(stream)
            state = stream.state_dict()
            expected = [next(stream) for _ in range(8)]  # Cross epoch boundaries.
            resumed = jit.TrainingStream(ds, dict(cfg, num_workers=0), state)
            for batch in expected:
                actual = next(resumed)
                for a, b in zip(actual, batch):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            images(directory, 10)
            with self.assertRaises(jit.ConfigError):
                jit.TrainingStream(jit.build_dataset(cfg), cfg, state)
            stream.close()
            resumed.close()

    def test_uniform_classes_coverage_splits_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, count in [('a', 1), ('b', 4), ('c', 10)]:
                images(root / label, count)
            cfg = config(dataset_path=directory, conditioning_mode='class',
                         class_names=['a', 'b', 'c', 'empty'], validation_percent=25,
                         batch_size=5, num_workers=2, use_flip=True, resize_width=8, resize_height=8)
            train, val = jit.build_training_datasets(cfg)
            stream = jit.TrainingStream(train, cfg)
            self.addCleanup(stream.close)
            self.assertEqual(stream.epoch_size, 24)
            keys = [key for batch in jit.StreamBatchSampler(stream) for key in batch]
            labels = [train.samples[key[1]][1] for key in keys]
            self.assertEqual([labels.count(i) for i in range(3)], [8, 8, 8])
            for start in range(0, len(labels), 3):
                self.assertEqual(set(labels[start:start + 3]), {0, 1, 2})
            self.assertEqual({key[1] for key in keys}, set(range(len(train))))
            self.assertFalse(set(jit.dataset_files(train)) & set(jit.dataset_files(val)))
            rare = [key for key in keys if train.samples[key[1]][1] == 0]
            seeded = jit.SeededDataset(train, stream.seed)
            self.assertTrue(any(not torch.equal(seeded[rare[0]][0], seeded[key][0])
                                for key in rare[1:]))
            for _ in range(3):
                next(stream)
            state = stream.state_dict()
            self.assertGreater(state['position'], len(train))
            expected = [next(stream) for _ in range(14)]  # Includes short batches and epoch boundaries.
            resumed = jit.TrainingStream(train, dict(cfg, num_workers=0), state)
            self.addCleanup(resumed.close)
            for batch in expected:
                for actual, reference in zip(next(resumed), batch):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            with self.assertRaises(jit.ConfigError):
                jit.TrainingStream(train, dict(cfg, class_sampling='natural'), state)
            natural = jit.TrainingStream(train, dict(cfg, class_sampling='natural', num_workers=0))
            self.addCleanup(natural.close)
            order = [key[1] for batch in jit.StreamBatchSampler(natural) for key in batch]
            self.assertEqual(sorted(order), list(range(len(train))))

    def test_class_fallback_preserves_label(self):
        with tempfile.TemporaryDirectory() as directory:
            for label in ('a', 'b'):
                images(Path(directory) / label, 2)
            ds = jit.build_dataset(config(dataset_path=directory, conditioning_mode='class',
                                          class_names=['a', 'b']))
            Path(ds.samples[0][0]).write_bytes(b'broken')
            with mock.patch.object(jit.random, 'choice', side_effect=lambda choices: choices[-1]):
                self.assertEqual(ds[0][1], 0)
                Path(ds.samples[1][0]).write_bytes(b'broken')
                with self.assertRaises(RuntimeError):
                    ds[0]
            with self.assertRaises(jit.ConfigError):
                jit.validate_config(config(class_sampling='invalid'))

    def test_accumulation_weights_short_batches(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            cfg = config(conditioning_mode='unconditional')
            model = MeanLoss().to(device)
            reference = copy.deepcopy(model)
            data = torch.arange(1, 6, dtype=torch.float32).reshape(5, 1)
            scaler = torch.cuda.amp.GradScaler(enabled=device == 'cuda')
            score = jit.accumulated_backward(model, [data[:3], data[3:]], cfg,
                                             scaler, device == 'cuda', torch.float16)
            opt = torch.optim.SGD(model.parameters(), lr=.01)
            scaler.unscale_(opt)
            loss = reference.p_losses(data.to(device))
            loss.backward()
            torch.testing.assert_close(model.weight.grad, reference.weight.grad)
            self.assertAlmostEqual(float(score), loss.item(), places=5)

    def test_validation_repeatable_and_rng_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            images(directory)
            cfg = config(dataset_path=directory, validation_percent=33, self_cond=True,
                         use_flip=True, validation_batch_size=2)
            _, val = jit.build_training_datasets(cfg)
            model = jit.build_model(cfg, 'cpu')
            model.layers[0].eval()
            modes = [m.training for m in model.modules()]
            jit.set_seed(123)
            rng = jit.capture_rng_state()
            a = jit.validation_loss(model, val, cfg)
            actual = (random.random(), np.random.rand(), torch.rand(2))
            jit.restore_rng_state(rng)
            expected = (random.random(), np.random.rand(), torch.rand(2))
            self.assertEqual(actual[:2], expected[:2])
            torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
            b = jit.validation_loss(model, val, cfg)
            self.assertEqual(a, b)
            self.assertEqual(modes, [m.training for m in model.modules()])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_amp_accumulation_validation_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images(root / 'data', 6)
            cfg = config(dataset_path=str(root / 'data'), validation_percent=33,
                         use_amp=True, grad_accum_steps=2)
            train, val = jit.build_training_datasets(cfg)
            for dtype in (torch.float16, torch.bfloat16):
                model = jit.build_model(cfg, 'cuda')
                optimizer = jit.build_optimizer(model, cfg)
                scaler = torch.cuda.amp.GradScaler(enabled=dtype == torch.float16, init_scale=16)
                stream = jit.TrainingStream(train, cfg)
                flow = jit.FlowMatchingWrapper(model)
                value = jit.accumulated_backward(flow, [next(stream), next(stream)], cfg,
                                                 scaler, True, dtype)
                self.assertTrue(np.isfinite(float(value)))
                scaler.unscale_(optimizer)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                scaler.step(optimizer)
                scaler.update()
                score = jit.validation_loss(model, val, cfg, True, dtype)
                self.assertTrue(np.isfinite(score))
                path = str(root / 'model.pt')
                jit.save_checkpoint(model, optimizer, None, 1, path, cfg=cfg, scaler=scaler, stream=stream)
                saved = torch.load(path, map_location='cpu', weights_only=False)
                resumed = torch.cuda.amp.GradScaler(enabled=dtype == torch.float16)
                resumed.load_state_dict(saved['scaler'])
                self.assertEqual(resumed.state_dict(), scaler.state_dict())
                self.assertEqual(saved['data_stream'], stream.state_dict())
                stream.close()

    def test_stream_cleanup_on_exception_and_repeated_close(self):
        with tempfile.TemporaryDirectory() as directory:
            images(directory, 4)
            cfg = config(dataset_path=directory, num_workers=2)
            stream = jit.TrainingStream(jit.build_dataset(cfg), cfg)
            with self.assertRaisesRegex(ValueError, 'training failed'):
                with jit.training_session(stream):
                    next(stream)
                    workers = list(stream.iterator._workers)
                    iterator = stream.iterator
                    shutdown = stream.iterator._shutdown_workers

                    def interrupted_shutdown():
                        iterator._shutdown_workers = shutdown
                        # An extra Ctrl+C during cleanup must not abort the join.
                        os.kill(os.getpid(), signal.SIGINT)
                        shutdown()

                    stream.iterator._shutdown_workers = interrupted_shutdown
                    raise ValueError('training failed')
            self.assertFalse(jit.TRAINING_ACTIVE)
            self.assertTrue(stream.closed)
            self.assertTrue(all(not worker.is_alive() for worker in workers))
            stream.close()
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                next(stream)

    @unittest.skipUnless(hasattr(os, 'killpg'), 'POSIX process group signals required')
    def test_ctrl_c_process_group_saves_and_exits_with_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.Popen(
                [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--shutdown-probe', directory],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
            lines = []
            try:
                deadline = time.monotonic() + 40
                while time.monotonic() < deadline:
                    ready, _, _ = select.select([process.stdout], [], [], 1)
                    if not ready:
                        continue
                    line = process.stdout.readline()
                    lines.append(line)
                    if 'READY_FOR_SIGINT' in line:
                        break
                    if process.poll() is not None:
                        self.fail('Probe exited before training: ' + ''.join(lines) + process.stderr.read())
                else:
                    self.fail('Probe did not reach training: ' + ''.join(lines))
                os.killpg(process.pid, signal.SIGINT)
                os.killpg(process.pid, signal.SIGINT)
                output, errors = process.communicate(timeout=30)
                output = ''.join(lines) + output
                self.assertEqual(process.returncode, 0, output + errors)
                self.assertIn('WORKERS_CLOSED checkpoint=1', output)
                self.assertEqual(output.count('CTRL+C detected'), 1, output)
                self.assertNotIn('Exception ignored', errors)
                self.assertNotIn('Traceback', errors)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()

    def test_freezing_and_optimizer_exclusion(self):
        for kind, updates, prefix in [
            ('jit', {}, 'layers.1.'),
            ('rin', {'rin_num_latents': 4, 'rin_latent_dim': 32}, 'rin_blocks.1.'),
            ('unet', {'width': 8, 'height': 8, 'bottleneck_res': 2, 'fmap_max': 32}, 'ups.1.'),
            ('hiermlp', {'width': 8, 'height': 8, 'hier_input_grid_size': 2,
                         'hier_output_grid_size': 2, 'fmap_max': 32}, 'refinement_stages.1.'),
        ]:
            cfg = config(model_type=kind, finetune_policy='last', **updates)
            model = jit.build_model(cfg, 'cpu')
            jit.configure_trainable(model, cfg)
            trainable = {name for name, p in model.named_parameters() if p.requires_grad}
            self.assertTrue(any(name.startswith(prefix) for name in trainable), (kind, trainable))
            for opt_type in ('adam', 'muon', 'layerwise_nsgda'):
                optimizer = jit.build_optimizer(model, dict(cfg, optimizer_type=opt_type))
                self.assertEqual({id(p) for g in optimizer.param_groups for p in g['params']},
                                 {id(p) for p in model.parameters() if p.requires_grad})
            frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
            optimizer = jit.build_optimizer(model, cfg)
            model(torch.randn(2, 3, cfg['height'], cfg['width']), torch.ones(2) * .5).square().mean().backward()
            optimizer.step()
            for name, p in model.named_parameters():
                if name in frozen:
                    torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)

    def test_best_is_finite_improvement_only_and_weights_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'best.pt'
            model = MeanLoss()
            cfg = config()
            best = jit.update_best(model, cfg, 2., 1, {}, path, 'validation_loss')
            for value in (3., float('nan'), float('inf')):
                self.assertEqual(jit.update_best(model, cfg, value, 2, best, path, 'validation_loss'), best)
            best = jit.update_best(model, cfg, 1., 3, best, path, 'validation_loss')
            saved = torch.load(path, weights_only=False)
            self.assertNotIn('optimizer', saved)
            self.assertEqual(saved['best']['step'], 3)
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def run_main(self, directory, answers, stop_after=None):
        calls = 0
        backward = jit.accumulated_backward

        def wrapped(*args, **kwargs):
            nonlocal calls
            result = backward(*args, **kwargs)
            calls += 1
            if calls == stop_after:
                jit.interrupted = True
            return result

        def answer(prompt):
            for key, value in answers.items():
                if key in prompt:
                    return value
            if 'Config edit:' in prompt:
                return ''
            return ''

        with mock.patch.object(jit, 'SAVE_DIR', str(directory)), \
                mock.patch.object(torch.cuda, 'is_available', return_value=False), \
                mock.patch('builtins.input', side_effect=answer), \
                mock.patch.object(jit, 'accumulated_backward', side_effect=wrapped), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            jit.main()
        return torch.load(Path(directory) / 'model.pt', map_location='cpu', weights_only=False)

    def test_main_exact_resume_with_validation_accumulation_and_dropout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images(root / 'data')
            cfg = config(dataset_path=str(root / 'data'), validation_percent=33,
                         validation_every=1, grad_accum_steps=2, use_flip=True,
                         lr_schedule='cosine', resize_width=8, resize_height=8)
            path = root / 'prepared.json'
            jit.save_config(cfg, path)
            for name in ('full', 'resume'):
                (root / name).mkdir()
            full = self.run_main(root / 'full', {'Mode [': 'train', 'Prepared config': str(path)})
            interrupted = self.run_main(root / 'resume', {'Mode [': 'train', 'Prepared config': str(path)}, 2)
            self.assertEqual(interrupted['completed_steps'], 2)
            resumed = self.run_main(root / 'resume', {'Mode [': 'continue'})
            self.assertEqual(resumed['completed_steps'], 4)
            for key in full['model']:
                torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)
            for key in full['ema']:
                torch.testing.assert_close(full['ema'][key], resumed['ema'][key], rtol=0, atol=0)
            self.assertEqual(full['data_stream'], resumed['data_stream'])
            self.assertEqual(full['best'], resumed['best'])
            self.assertTrue((root / 'resume' / 'best.pt').exists())
            self.assertTrue((root / 'resume' / 'config.json').exists())

    def test_main_legacy_class_stream_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images(root / 'data' / 'a', 1)
            images(root / 'data' / 'b', 5)
            cfg = config(dataset_path=str(root / 'data'), conditioning_mode='class',
                         class_names=['a', 'b'], class_sampling='natural', steps=1)
            path = root / 'prepared.json'
            jit.save_config(cfg, path)
            run = root / 'run'
            run.mkdir()
            original = self.run_main(run, {'Mode [': 'train', 'Prepared config': str(path)})
            del original['config']['class_sampling']
            torch.save(original, run / 'model.pt')
            migrated = self.run_main(run, {'Mode [': 'continue'})
            self.assertEqual(migrated['config']['class_sampling'], 'uniform')
            self.assertEqual(migrated['completed_steps'], 2)
            self.assertNotEqual(original['data_stream']['fingerprint'], migrated['data_stream']['fingerprint'])
            self.assertEqual(migrated['data_stream']['position'], 2)

    def test_main_retry_finetune_and_changed_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images(root / 'data', 4)
            images(root / 'other', 4)
            run = root / 'run'
            run.mkdir()
            cfg = config(dataset_path=str(root / 'data'), steps=1, best_training_loss=True)
            path = root / 'prepared.json'
            jit.save_config(cfg, path)
            initial = self.run_main(run, {'Mode [': 'train', 'Prepared config': str(path)})
            fine = self.run_main(run, {'Mode [': 'finetune', 'Trainable parameters': 'last'})
            self.assertEqual(fine['completed_steps'], 1)
            self.assertEqual(fine['config']['finetune_policy'], 'last')
            for name in initial['ema']:
                if not name.startswith(('layers.1.', 'to_pixels.', 'norm.', 'final_adaLN.')):
                    torch.testing.assert_close(initial['ema'][name], fine['model'][name], rtol=0, atol=0)
            retry = self.run_main(run, {'Mode [': '2'})
            self.assertEqual(retry['completed_steps'], 1)
            self.assertEqual(retry['config']['finetune_policy'], 'all')
            continued = self.run_main(run, {'Mode [': 'continue', 'Dataset location': str(root / 'other')})
            self.assertEqual(continued['completed_steps'], 2)
            self.assertNotEqual(retry['data_stream']['fingerprint'], continued['data_stream']['fingerprint'])
            self.assertEqual(continued['data_stream']['position'], 2)

    def test_prepared_configs_class_and_pix2pix_formats(self):
        for mode in ('class', 'folders', 'combined', 'noise'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                if mode in ('class', 'folders'):
                    for label in ('A', 'B'):
                        images(root / 'data' / label, 4)
                else:
                    images(root / 'data', 4)
                cfg = config(dataset_path=str(root / 'data'), steps=1, validation_percent=25,
                             conditioning_mode='class' if mode == 'class' else 'pix2pix',
                             pix2pix_source_mode='folders' if mode == 'class' else mode)
                path = root / 'prepared.json'
                jit.save_config(cfg, path)
                run = root / 'run'
                run.mkdir()
                saved = self.run_main(run, {'Mode [': 'train', 'Prepared config': str(path)})
                self.assertEqual(saved['completed_steps'], 1)
                self.assertTrue((run / 'best.pt').exists())
                if mode == 'class':
                    self.assertEqual(saved['config']['class_names'], ['A', 'B'])


def shutdown_probe(directory):
    """Run the actual main/checkpoint path in a disposable terminal process group."""
    torch.set_num_threads(1)
    root = Path(directory)
    images(root / 'data', 8)
    run = root / 'run'
    run.mkdir()
    cfg = config(dataset_path=str(root / 'data'), num_workers=2, steps=100)
    prepared = root / 'prepared.json'
    jit.save_config(cfg, prepared)
    workers = []
    next_batch = jit.TrainingStream.__next__
    backward = jit.accumulated_backward

    def observed_next(stream):
        batch = next_batch(stream)
        workers[:] = stream.iterator._workers
        return batch

    def waiting_backward(*args, **kwargs):
        print('READY_FOR_SIGINT', flush=True)
        while not jit.interrupted:
            time.sleep(.01)
        return backward(*args, **kwargs)

    answers = iter(['train', str(prepared)])
    with mock.patch.object(jit, 'SAVE_DIR', str(run)), \
            mock.patch('builtins.input', side_effect=lambda _: next(answers)), \
            mock.patch.object(torch.cuda, 'is_available', return_value=False), \
            mock.patch.object(jit.TrainingStream, '__next__', observed_next), \
            mock.patch.object(jit, 'accumulated_backward', waiting_backward):
        jit.main()
    saved = torch.load(run / 'model.pt', map_location='cpu', weights_only=False)
    assert saved['completed_steps'] == 1
    assert all(not worker.is_alive() for worker in workers)
    print('WORKERS_CLOSED checkpoint=1', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--shutdown-probe':
        shutdown_probe(sys.argv[2])
    else:
        unittest.main()
