"""jit5_gui: the form builds the same config as the CLI prompts; training, resume and sampling work."""
import contextlib
import io
import os
import queue
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

import jit5 as jit
import jit5_gui as gui


def images(folder, count=8, size=16):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(len(str(folder)))
    for i in range(count):
        Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(folder / f'{i}.png')


def cli_config(save_dir, answers):
    """Run jit5.main's new-run prompts with the given answers; return the config it would train."""
    captured = {}

    def answer(prompt):
        for key, value in answers.items():
            if key in prompt:
                return value
        return ''

    def fake_run(cfg, *args, **kwargs):
        captured['cfg'] = cfg
        return 0

    with mock.patch.object(jit, 'SAVE_DIR', str(save_dir)), \
            mock.patch.object(torch.cuda, 'is_available', return_value=False), \
            mock.patch('builtins.input', side_effect=answer), \
            mock.patch.object(jit, 'run_training', side_effect=fake_run), \
            contextlib.redirect_stdout(io.StringIO()):
        jit.main()
    return captured['cfg']


class FormParityTests(unittest.TestCase):
    """Each case: CLI prompt answers and the equivalent GUI form fields; everything else is defaults."""
    CASES = {
        'jit': ({}, {}),
        'unet': ({'Model type': '7', 'Bottleneck Resolution': '8'}, {'model_type': 'unet', 'bottleneck_res': 8}),
        'hiermlp': ({'Model type': '45'}, {'model_type': 'hiermlp'}),
        'rin': ({'Model type': '44'}, {'model_type': 'rin'}),
        'fcdm_unet': ({'Model type': '46'}, {'model_type': 'fcdm_unet'}),
        'fcdm_isotropic': ({'Model type': '47'}, {'model_type': 'fcdm_isotropic'}),
        'conv_stem_both': ({'Conv-stem [': '3', 'Stem class/time': '5', 'U-Net-style': 'yes'},
                           {'conv_stem': 3, 'stem_conditioning': 5, 'stem_skips': True}),
        'muon': ({'Optimizer': '16', 'MuonAll (all': 'no'}, {'optimizer_type': 'muon'}),
        'adago': ({'Optimizer': '19'}, {'optimizer_type': 'adago'}),
        'adamhd': ({'Optimizer': '24'}, {'optimizer_type': 'adamhd'}),
        'cosine_ema_legacy': ({'LR schedule': 'cosine', 'EMA warmup': 'legacy', 'Validation percentage': '25'},
                              {'lr_schedule': 'cosine', 'ema_warmup': 'legacy', 'validation_percent': 25}),
        'class': ({'Dataset format': '1'}, {'conditioning_mode': 'class'}),
        'pix2pix_noise': ({'Dataset format': '2', 'Pix2Pix source IDs': '9,3'},
                          {'conditioning_mode': 'pix2pix', 'pix2pix_sources': ['noise', 'superresolution']}),
        'pix2pix_folders': ({'Dataset format': '2', 'Pix2Pix direction': '1'},
                            {'conditioning_mode': 'pix2pix', 'pix2pix_sources': ['folders'], 'pix2pix_direction': 'b_to_a'}),
        'crop_range': ({'Source image width': '32', 'Source image height': '32', 'Crop width': '8-16',
                        'Crop height': '8-16', 'Variable crop output': 'min', 'Patch width': '4', 'Patch height': '4'},
                       {'resize_width': 32, 'resize_height': 32, 'crop_w': '8-16', 'crop_h': '8-16',
                        'crop_output': 'min', 'p_width': 4, 'p_height': 4}),
    }

    def test_form_matches_cli_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images(root / 'data' / 'x')
            images(root / 'data' / 'y')
            images(root / 'data' / 'A')
            images(root / 'data' / 'B')
            base_cli = {'Mode [': 'train', 'Dataset location': str(root / 'data'),
                        'Source image width': '16', 'Source image height': '16'}
            base_form = {'dataset_path': str(root / 'data'), 'resize_width': 16, 'resize_height': 16,
                         'crop_w': '', 'crop_h': ''}
            for name, (cli_answers, form) in self.CASES.items():
                with self.subTest(name):
                    save = root / name
                    save.mkdir()
                    expected = cli_config(save, {**base_cli, **cli_answers})
                    got, _ = gui.config_from_form({**base_form, **form})
                    jit.apply_config_defaults(got)
                    self.assertEqual(got, expected)


class SessionTests(unittest.TestCase):
    def form(self, root, **extra):
        return dict(dataset_path=str(root / 'data'), resize_width=16, resize_height=16, crop_w='16', crop_h='16',
                    dim=32, depth=1, heads=2, p_width=4, p_height=4, batch_size=4, steps=4, lr=1e-3,
                    optimizer_type='adam', use_amp=False, sample_every=2, save_every=2, sampling_steps=2,
                    num_sample_images=2, warmup_steps=0, validation_percent=25, validation_every=2,
                    num_workers=0) | extra

    def run_jobs(self, trainer, jobs, spec):
        trainer.start(spec)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            while not jobs.empty():
                jobs.get()()
        return trainer.status()

    def test_train_stop_continue_and_sample(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(torch.cuda, 'is_available', return_value=False):
            root = Path(directory)
            images(root / 'data')
            jobs = queue.Queue()
            trainer = gui.Trainer(jobs)
            ck = str(root / 'ck')
            status = self.run_jobs(trainer, jobs, {'mode': 'new', 'form': self.form(root), 'checkpoint_dir': ck})
            self.assertEqual((status['state'], status['completed'], len(status['points'])), ('done', 4, 4))
            self.assertEqual([p['step'] for p in status['previews']], [2, 4])
            self.assertEqual([v[0] for v in status['val']], [2, 4])

            info = gui.checkpoint_info(ck)
            self.assertEqual(info['completed_steps'], 4)
            original = trainer._progress
            trainer._progress = lambda event: original(event) or (event['event'] == 'step' and event['step'] >= 6)
            status = self.run_jobs(trainer, jobs, {'mode': 'continue', 'checkpoint_dir': ck,
                                                   'resume': {'extra_steps': 10, 'optimizer': {'weight_decay': .01}}})
            self.assertEqual((status['state'], status['completed'], status['total']), ('stopped', 6, 14))
            saved = torch.load(Path(ck) / 'model.pt', map_location='cpu', weights_only=False)
            self.assertEqual(saved['completed_steps'], 6)
            self.assertEqual(saved['optimizer']['param_groups'][0]['weight_decay'], .01)

            before = (torch.get_rng_state().clone(), random.getstate())
            sampler = gui.Sampler()
            sampler.start({'checkpoint': ck, 'steps': 3, 'count': 3, 'batch_size': 2, 'seed': 7})
            sampler.thread.join()
            first = sampler.status()
            self.assertEqual((first['state'], first['done']), ('done', 3))
            self.assertTrue(torch.equal(before[0], torch.get_rng_state()))
            self.assertEqual(before[1], random.getstate())
            self.assertTrue(os.path.isfile(first['grid']))
            sampler.start({'checkpoint': ck, 'steps': 3, 'count': 3, 'batch_size': 3, 'seed': 7})
            sampler.thread.join()
            second = sampler.status()
            for a, b in zip(first['results'], second['results']):  # same seed, other batch size: same images
                self.assertTrue(np.array_equal(np.asarray(Image.open(a['path'])), np.asarray(Image.open(b['path']))))

    def test_invalid_form_is_rejected_before_queueing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images(root / 'data')
            jobs = queue.Queue()
            trainer = gui.Trainer(jobs)
            with self.assertRaises(ValueError):
                trainer.start({'mode': 'new', 'form': self.form(root, p_width=5)})
            self.assertTrue(jobs.empty())
            self.assertEqual(trainer.status()['state'], 'idle')


if __name__ == '__main__':
    unittest.main()
