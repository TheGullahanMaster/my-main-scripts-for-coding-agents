"""RIN architecture and recurrence regressions against jit5.py.

Run: python -B -m unittest -v test_jit5_rin
Extract definitions to avoid importing the interactive app and optional datasets.
Reference: Jabri et al., Algorithms 1/3 and google-research/pix2seq architectures/tape.py.
"""
import ast
import contextlib
import copy
import io
import math
from pathlib import Path
import types
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint


def load_source(path):
    names = {
        'RINMLP', 'RINLatentLayer', 'RINTimeEmbedding', 'RINBlock', 'JiTModel',
        'SinusoidalPosEmb', 'RMSNorm', 'FlowMatchingWrapper', 'ConvMixerBlock',
        'get_2d_sincos_pos_embed', 'modulate', 'ConvStemInput', 'ConvStemOutput',
        'conv_stem_geometry', 'validate_stem_skips', 'scale_stem_widths',
        'make_stem_activation', 'make_stem_norm', 'StemGLU', 'ChannelLayerNorm2d',
        'apply_config_defaults', 'ConfigError', 'validate_config', 'build_model',
        'conditioning_model_kwargs',
        'StemConditioning', 'validate_stem_conditioning',
    }
    constants = {
        'STEM_ACTIVATIONS', 'MODEL_TYPES', 'IS_CONV', 'IS_HIERARCHICAL',
        'STEM_CONDITIONING', 'EMA_WARMUP_MODES', 'COMPILE_MODES',
        'NEEDS_HEADS', 'HYPER_HEADS', 'OPTIMIZER_TYPES',
        'MODERN_CLION_DEFAULT_NU', 'OLD_MODERN_CLION_DEFAULT_NU',
    }
    tree = ast.parse(Path(path).read_text())
    nodes = [node for node in tree.body
             if (isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names)
             or (isinstance(node, ast.Assign) and any(
                 isinstance(t, ast.Name) and t.id in constants for t in node.targets))]
    scope = dict(torch=torch, nn=nn, F=F, math=math)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return types.SimpleNamespace(**scope)


jit = load_source(Path(__file__).with_name('jit5.py'))


def config(**updates):
    cfg = dict(width=8, height=8, channels=3, dim=16, depth=2, heads=2,
               p_height=2, p_width=2, model_type='rin', rin_num_latents=6,
               rin_latent_dim=32, rin_layers_per_block=2,
               batch_size=2, steps=2, lr=.0002, optimizer_type='adam',
               sampling_steps=3, use_amp=False)
    cfg.update(updates)
    jit.apply_config_defaults(cfg)
    return cfg


def build(**updates):
    with contextlib.redirect_stdout(io.StringIO()):
        return jit.build_model(config(**updates), 'cpu')


class RINTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(123)

    def test_read_compute_write_order_and_attention_inputs(self):
        block = jit.RINBlock(16, 2, latent_dim=32, num_layers=3)
        data, latents = torch.randn(2, 12, 16), torch.randn(2, 6, 32)
        events, inputs = [], {}
        handles = []
        for name, module in [('read', block.read), ('read_mlp', block.read_mlp),
                             *[(f'compute{i}', m) for i, m in enumerate(block.compute)],
                             ('write', block.write), ('write_mlp', block.write_mlp)]:
            def record(mod, args, name=name):
                events.append(name)
                inputs[name] = tuple(a.detach().clone() for a in args)
            handles.append(module.register_forward_pre_hook(record))
        output, state = block(data, latents)
        for handle in handles:
            handle.remove()
        self.assertEqual(events, ['read', 'read_mlp', 'compute0', 'compute1', 'compute2', 'write', 'write_mlp'])
        torch.testing.assert_close(inputs['read'][1], data)
        torch.testing.assert_close(inputs['read'][2], data)
        torch.testing.assert_close(inputs['read'][0], F.layer_norm(latents, (32,), eps=1e-6))
        torch.testing.assert_close(inputs['write'][1], state)
        torch.testing.assert_close(inputs['write'][0], F.layer_norm(data, (16,), eps=1e-6))
        self.assertEqual(output.shape, data.shape)
        self.assertEqual(state.shape, latents.shape)
        (output.square().mean() + state.square().mean()).backward()
        for module in [block.read_mlp, block.write_mlp, *block.compute]:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))

    def test_condition_tokens_and_normalized_interface(self):
        model = build(conditioning_mode='class', class_names=['a', 'b'], class_dropout_prob=.1)
        image, time, labels = torch.randn(2, 3, 8, 8), torch.tensor([.2, .7]), torch.tensor([0, 1])
        captured = []
        handle = model.rin_blocks[0].register_forward_pre_hook(
            lambda mod, args: captured.append(tuple(a.detach().clone() for a in args)))
        output, state = model(image, time, class_labels=labels, return_latents=True)
        handle.remove()
        interface, latents = captured[0]
        tokens = model.to_patch_embedding(model.unfold(image).transpose(1, 2))
        torch.testing.assert_close(interface, F.layer_norm(tokens, (16,), eps=1e-6) + model.pos_embedding)
        self.assertEqual(model.latent_tokens.shape, (1, 4, 32))
        torch.testing.assert_close(latents[:, :4], model.latent_tokens.expand(2, -1, -1))
        torch.testing.assert_close(latents[:, -2], model.time_mlp(time))
        torch.testing.assert_close(latents[:, -1], model.class_embedding(labels))
        self.assertEqual(state.shape, (2, 6, 32))
        self.assertEqual(output.shape, image.shape)
        self.assertIsInstance(model.norm, nn.LayerNorm)
        self.assertFalse(model.self_cond)  # No extra pixel channels for RIN.

    def test_time_embedding_reference_frequencies_and_normalization(self):
        embedding = jit.RINTimeEmbedding(32, scale=1000)
        time = torch.tensor([0., .25, 1.])
        angles = (time * 1000)[:, None] * torch.tensor([1., .1, .01, .001])[None]
        features = torch.cat((angles.sin(), angles.cos()), -1)
        mean = features.mean(-1, keepdim=True)
        variance = (features - mean).square().mean(-1, keepdim=True)
        expected = embedding.proj((features - mean) / variance.sqrt())
        torch.testing.assert_close(embedding(time), expected)

    def test_previous_latents_zero_init_residual_and_stop_gradient(self):
        model = build()
        image, time = torch.randn(2, 3, 8, 8), torch.tensor([.2, .7])
        previous = torch.randn(2, 6, 32, requires_grad=True)
        baseline = model(image, time)
        torch.testing.assert_close(model(image, time, latent_self_cond=previous), baseline)
        with torch.no_grad():
            model.latent_prev_norm.weight.fill_(1)
            # A zero FFN must retain the previous state through its residual.
            for p in model.latent_prev_proj.net.parameters():
                p.zero_()
        captured = []
        handle = model.rin_blocks[0].register_forward_pre_hook(
            lambda mod, args: captured.append(args[1].detach().clone()))
        output = model(image, time, latent_self_cond=previous)
        handle.remove()
        initial = torch.cat([model.latent_tokens.expand(2, -1, -1), model.time_mlp(time)[:, None]], 1)
        torch.testing.assert_close(captured[0], initial + F.layer_norm(previous, (32,), eps=1e-6))
        output.square().mean().backward()
        self.assertIsNone(previous.grad)
        self.assertGreater(model.latent_prev_norm.weight.grad.abs().sum(), 0)
        # Missing context follows the SAME transform applied to explicit zeros.
        torch.testing.assert_close(model(image, time), model(image, time, latent_self_cond=torch.zeros_like(previous)))

    def test_probability_and_labels_shared_between_passes(self):
        image, labels = torch.randn(2, 3, 8, 8), torch.tensor([0, 1])
        for probability, expected_calls in [(0., 1), (1., 2)]:
            model = build(conditioning_mode='class', class_names=['a', 'b'], class_dropout_prob=1.)
            flow = jit.FlowMatchingWrapper(model, self_cond_prob=probability, class_dropout_prob=1.)
            calls = []
            def record(mod, args, kwargs):
                calls.append((torch.is_grad_enabled(), args, kwargs.copy()))
            handle = model.register_forward_pre_hook(record, with_kwargs=True)
            loss = flow.p_losses(image, class_labels=labels)
            handle.remove()
            self.assertEqual(len(calls), expected_calls)
            self.assertTrue(calls[-1][0])
            self.assertTrue(torch.equal(calls[-1][2]['class_labels'], torch.full_like(labels, 2)))
            if probability == 1:
                self.assertFalse(calls[0][0])
                self.assertFalse(calls[-1][2]['latent_self_cond'].requires_grad)
                torch.testing.assert_close(calls[0][1][0], calls[1][1][0])
                torch.testing.assert_close(calls[0][1][1], calls[1][1][1])
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
        # Exercise both sides of an intermediate probability without flaky statistics.
        for draw, count in [(.1, 2), (.95, 1)]:
            model = build()
            with mock.patch.object(torch, 'rand', return_value=torch.tensor(draw)), \
                    mock.patch.object(model, 'forward', wraps=model.forward) as forward:
                jit.FlowMatchingWrapper(model, self_cond_prob=.9).p_losses(image)
                self.assertEqual(forward.call_count, count)

    def test_disabled_self_conditioning_and_default_rate(self):
        model = build(self_cond=False)
        self.assertFalse(model.supports_latent_self_conditioning)
        with mock.patch.object(model, 'forward', wraps=model.forward) as forward:
            jit.FlowMatchingWrapper(model, self_cond_prob=1).p_losses(torch.randn(2, 3, 8, 8))
            self.assertEqual(forward.call_count, 1)
            self.assertIsNone(forward.call_args.kwargs.get('x_self_cond'))
        self.assertEqual(jit.FlowMatchingWrapper(build()).self_cond_prob, .9)

    def test_sampling_carries_separate_conditional_states(self):
        model = build(conditioning_mode='class', class_names=['a', 'b'], class_dropout_prob=.1)
        model.eval()
        calls, outputs = [], []
        def record(mod, args, kwargs, result):
            calls.append(kwargs.get('latent_self_cond'))
            outputs.append(result[1])
        handle = model.register_forward_hook(record, with_kwargs=True)
        sample = jit.FlowMatchingWrapper(model).sample((1, 3, 8, 8), steps=3,
                    class_labels=torch.tensor([0]), guidance_scale=2)
        handle.remove()
        self.assertTrue(torch.isfinite(sample).all())
        self.assertEqual(len(calls), 10)  # Two branches, Heun + final Euler.
        self.assertIsNone(calls[0])
        self.assertIsNone(calls[1])
        for i in range(2, len(calls)):
            torch.testing.assert_close(calls[i], outputs[i-2])

    def test_checkpointing_preserves_outputs_and_gradients(self):
        plain = build()
        checkpointed = copy.deepcopy(plain)
        checkpointed.use_gradient_checkpointing = True
        x, time = torch.randn(2, 3, 8, 8), torch.tensor([.1, .8])
        a, b = plain(x, time), checkpointed(x, time)
        torch.testing.assert_close(a, b)
        a.square().mean().backward()
        b.square().mean().backward()
        for p, q in zip(plain.parameters(), checkpointed.parameters()):
            if p.grad is None:
                self.assertIsNone(q.grad)
            else:
                torch.testing.assert_close(p.grad, q.grad)

    def test_training_sampling_stems_precision_and_checkpoint_roundtrip(self):
        for mode, skips in [(0, False), (1, False), (2, False), (3, False), (3, True)]:
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(stem=mode, skips=skips, dtype=dtype):
                    model = build(conv_stem=mode, stem_skips=skips, stem_initial=4, stem_max=8,
                                  stem_activation=7, conditioning_mode='pix2pix').to(dtype=dtype)
                    for schedule in ('linear', 'rin_sigmoid'):
                        flow = jit.FlowMatchingWrapper(model, noise_schedule=schedule, self_cond_prob=1)
                        data, condition = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
                        loss = flow.p_losses(data, condition=condition)
                        loss.backward()
                        self.assertTrue(torch.isfinite(loss))
                        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                        model.eval()
                        sample = flow.sample(data.shape, steps=3, condition=condition)
                        self.assertEqual(sample.shape, data.shape)
                        self.assertTrue(torch.isfinite(sample).all())
                        model.train()
                        model.zero_grad(set_to_none=True)
        cfg = config()
        model = build().eval()
        buffer = io.BytesIO()
        torch.save({'cfg': cfg, 'model': model.state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        with contextlib.redirect_stdout(io.StringIO()):
            rebuilt = jit.build_model(saved['cfg'], 'cpu').eval()
        rebuilt.load_state_dict(saved['model'])
        image, time = torch.randn(2, 3, 8, 8), torch.tensor([.1, .5])
        torch.testing.assert_close(model(image, time), rebuilt(image, time), rtol=0, atol=0)

    def test_config_validation(self):
        cfg = config()
        self.assertTrue(cfg['self_cond'])
        self.assertEqual(cfg['self_cond_prob'], .9)
        for updates in [dict(rin_latent_dim=18), dict(rin_latent_dim=0),
                        dict(rin_layers_per_block=0), dict(rin_num_latents=1),
                        dict(rin_latent_dim=40, heads=4, dim=18)]:
            with self.subTest(updates=updates), self.assertRaises(jit.ConfigError):
                jit.validate_config(config(**updates))
        other = config(model_type='mlpmixer')
        self.assertFalse(other['self_cond'])
        self.assertEqual(other['self_cond_prob'], .5)


if __name__ == '__main__':
    unittest.main()
