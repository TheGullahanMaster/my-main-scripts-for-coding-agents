import copy
import os
import unittest
from unittest import mock

import torch
from torch import nn

import jit5 as jit


class OracleX(nn.Module):
    """Always predicts the true clean image, isolating solver error."""

    def __init__(self, target):
        super().__init__()
        self.target = target
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, z, t, **kwargs):
        return self.target.clone()


class SamplerTests(unittest.TestCase):
    def test_perfect_x_prediction_fully_denoises_at_any_step_count(self):
        torch.manual_seed(0)
        target = torch.rand(2, 3, 8, 8) * 2 - 1
        flow = jit.FlowMatchingWrapper(OracleX(target), pred_mode='x', self_cond_prob=0.0)
        for steps in (1, 10, 20, 50, 100):
            for solver in ('euler', 'heun'):
                out = flow.sample(target.shape, steps=steps, solver=solver)
                self.assertLess((out - target).abs().max().item(), 1e-4, (steps, solver))

    def test_training_clamp_is_unchanged(self):
        flow = jit.FlowMatchingWrapper(OracleX(torch.zeros(1)), pred_mode='x')
        z, x, t = torch.ones(1, 1, 1, 1), torch.zeros(1, 1, 1, 1), torch.full((1, 1, 1, 1), 0.99)
        _, _, v = flow._convert(x, z, t)
        torch.testing.assert_close(v, (x - z) / 0.05)


class EMATests(unittest.TestCase):
    def test_legacy_ramp_is_unchanged(self):
        ema = jit.EMA(nn.Linear(2, 2))
        for n in (0, 10, 1000, 10 ** 6):
            ema.step_count = n
            self.assertEqual(ema.get_decay(), min(0.9999, (1 + n) / (10 + n)))

    def test_smooth_warmup_is_monotone_and_lands_on_decay(self):
        ema = jit.EMA(nn.Linear(2, 2), warmup_steps=1000, warmup='smooth')
        decays = []
        for n in range(0, 1101):
            ema.step_count = n
            decays.append(ema.get_decay())
        self.assertEqual(decays[0], 0.0)
        self.assertTrue(all(b >= a for a, b in zip(decays, decays[1:])))
        self.assertAlmostEqual(decays[1000], 0.9999, places=12)
        # No snap: the last warmup increment is far smaller than the first ones.
        self.assertLess(decays[1000] - decays[999], 1e-6)
        self.assertEqual(decays[1100], 0.9999)

    def test_none_is_constant(self):
        ema = jit.EMA(nn.Linear(2, 2), warmup='none')
        ema.step_count = 1
        self.assertEqual(ema.get_decay(), 0.9999)

    def test_foreach_update_matches_reference_loop(self):
        torch.manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)).to(dtype)
            ema = jit.EMA(model, warmup='none', decay=0.9)
            reference = copy.deepcopy(ema.shadow)
            for _ in range(3):
                with torch.no_grad():
                    for p in model.parameters():
                        p.add_(torch.randn_like(p))
                    model[1].running_mean.add_(1)
                ema.update(model)
                with torch.no_grad():
                    for s, p in zip(reference.parameters(), model.parameters()):
                        s.lerp_(p.float(), 0.1)
                    for s, b in zip(reference.buffers(), model.buffers()):
                        s.copy_(b)
            for a, b in zip(ema.shadow.state_dict().values(), reference.state_dict().values()):
                torch.testing.assert_close(a, b)

    def test_rejects_unknown_warmup(self):
        with self.assertRaises(ValueError):
            jit.EMA(nn.Linear(1, 1), warmup='bogus')


@unittest.skipUnless(torch.cuda.is_available() and jit.fused_scan_available(torch.zeros(1, device='cuda'))
                     if torch.cuda.is_available() else False, 'Triton scan kernel needs CUDA')
class FusedScanTests(unittest.TestCase):
    def test_vim_scans_match_python_loop(self):
        real = jit.fused_scan_available
        for make in (lambda: jit.BiMambaV2(32, depth=2), lambda: jit.BiSSM(32)):
            torch.manual_seed(0)
            base = make().cuda()
            x = torch.randn(2, 40, 32, device='cuda')
            results = []
            for available in (lambda t: False, real):
                model = copy.deepcopy(base)
                with mock.patch.object(jit, 'fused_scan_available', available):
                    xx = x.clone().requires_grad_()
                    out = model(xx)
                    out.square().mean().backward()
                results.append((out, xx.grad, [p.grad for p in model.parameters()]))
            (lo, lx, lg), (fo, fx, fg) = results
            torch.testing.assert_close(fo, lo, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(fx, lx, rtol=1e-3, atol=1e-5)
            for a, b in zip(fg, lg):
                torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-5)


@unittest.skipUnless(torch.cuda.is_available() and jit.hyena_direct_available(torch.zeros(1, 1, 2, 2, device='cuda'))
                     if torch.cuda.is_available() else False, 'Triton Hyena kernel needs CUDA')
class HyenaKernelTests(unittest.TestCase):
    def test_direct_conv_matches_fft(self):
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for B, C, H, W in ((1, 4, 3, 3), (3, 8, 5, 7), (17, 4, 16, 16)):
                torch.manual_seed(0)
                hyena = jit.Hyena2D(C, H, W).cuda()
                v = torch.randn(B, C, H, W, device='cuda', requires_grad=True)
                f = torch.randn(C, 2 * H - 1, 2 * W - 1, device='cuda', requires_grad=True)
                g = torch.randn(B, C, H, W, device='cuda')
                expected = hyena.conv_fft(v, f)
                expected_v, expected_f = torch.autograd.grad(expected, (v, f), g)
                actual = jit.hyena_conv(v, f)
                actual_v, actual_f = torch.autograd.grad(actual, (v, f), g)
                for a, e in ((actual, expected), (actual_v, expected_v), (actual_f, expected_f)):
                    torch.testing.assert_close(a, e, rtol=1e-4, atol=1e-5)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32


@unittest.skipUnless(torch.cuda.is_available() and jit.hyena_direct_available(torch.zeros(1, 1, 2, 2, device='cuda'))
                     if torch.cuda.is_available() else False, 'Triton FCDM kernels need CUDA')
class FCDMKernelTests(unittest.TestCase):
    def test_fused_block_matches_pytorch_path(self):
        import jit5_fcdm
        tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        try:
            for B, C, H, W in ((2, 32, 8, 8), (3, 48, 5, 7), (2, 16, 24, 30)):  # 720 rows spans 2 programs
                torch.manual_seed(0)
                block = jit5_fcdm.FCDMBlock(C).cuda()
                for p in block.parameters():
                    nn.init.normal_(p, std=0.1)  # non-zero adaLN and GRN
                x = torch.randn(B, C, H, W, device='cuda')
                context = torch.randn(B, C, device='cuda')
                results = []
                for available in ((lambda h: False), jit5_fcdm.fcdm_kernels_available):
                    model = copy.deepcopy(block)
                    xx = x.clone().requires_grad_()
                    with mock.patch.object(jit5_fcdm, 'fcdm_kernels_available', available):
                        out = model(xx, context)
                    out.square().mean().backward()
                    results.append([out, xx.grad] + [p.grad for p in model.parameters()])
                for a, e in zip(results[1], results[0]):
                    torch.testing.assert_close(a, e, rtol=1e-4, atol=1e-6)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32


class GradStrideTests(unittest.TestCase):
    def test_gradients_take_parameter_strides_without_changing_values(self):
        param = nn.Parameter(torch.randn(8, 1, 7, 7))
        param.grad = torch.randn(8, 7, 7, 1).permute(0, 3, 1, 2)  # size-1 dim stride differs
        other = nn.Parameter(torch.randn(3, 4))
        other.grad = torch.randn(4, 3).t()  # genuinely transposed layout
        expected = [param.grad.clone(), other.grad.clone()]
        self.assertNotEqual(param.grad.stride(), param.stride())
        jit.align_grad_strides([param, other])
        for p, e in zip((param, other), expected):
            self.assertEqual(p.grad.stride(), p.stride())
            torch.testing.assert_close(p.grad, e, rtol=0, atol=0)


class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, factor):
        ctx.factor = factor
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.factor, None


class CompiledGradientCheckTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(width=4, height=4, channels=3, dim=16, depth=2, heads=2, p_width=2, p_height=2,
                        model_type='unet', batch_size=2, steps=1, lr=1e-3, optimizer_type='adam',
                        dataset_path='x', sampling_steps=2, use_amp=False, compile_mode='default',
                        bottleneck_res=2, fmap_max=16)
        jit.apply_config_defaults(self.cfg)
        torch.manual_seed(0)
        self.model = jit.build_model(self.cfg, 'cpu')
        self.flow = jit.FlowMatchingWrapper(self.model)

    def fake_compile(self, factor):
        model = self.model
        model._compiled_call_impl = lambda *a, **k: ScaleGrad.apply(model._call_impl(*a, **k), factor)

    def check(self):
        with mock.patch('builtins.print'):
            return jit.verify_compiled_gradients(self.model, self.flow, self.cfg, torch.device('cpu'),
                                                 False, torch.float16)

    def test_matching_gradients_keep_compiled_model(self):
        self.fake_compile(1.0)
        self.assertTrue(self.check())
        self.assertIsNotNone(self.model._compiled_call_impl)

    def test_corrupted_gradients_fall_back_to_eager(self):
        self.fake_compile(1000.0)
        self.assertFalse(self.check())
        self.assertIsNone(self.model._compiled_call_impl)

    def test_state_is_restored(self):
        self.fake_compile(1.0)
        buffers = {k: b.clone() for k, b in self.model.named_buffers()}
        rng = torch.get_rng_state()
        self.check()
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for k, b in self.model.named_buffers():
            torch.testing.assert_close(b, buffers[k], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_uncompiled_model_is_skipped(self):
        self.assertFalse(self.check())


class CompileSetupTests(unittest.TestCase):
    def ptxas(self, versions, driver, env=None):
        with mock.patch.dict(os.environ, env or {}, clear=True), \
                mock.patch.object(jit, 'cuda_driver_version', return_value=driver), \
                mock.patch.object(jit.glob, 'glob', return_value=['/usr/local/cuda-12.1', '/usr/local/cuda-11.7']), \
                mock.patch.object(jit.shutil, 'which', return_value=None), \
                mock.patch.object(jit.os.path, 'isfile', side_effect=lambda p: p in versions), \
                mock.patch.object(jit, 'ptxas_version', side_effect=versions.get), \
                mock.patch('builtins.print'):
            ok = jit.configure_triton_ptxas()
            return ok, os.environ.get('TRITON_PTXAS_PATH')

    def test_old_driver_picks_ptxas_not_newer_than_driver(self):
        versions = {'/usr/local/cuda-12.1/bin/ptxas': 12010, '/usr/local/cuda-11.7/bin/ptxas': 11070}
        self.assertEqual(self.ptxas(versions, 11070), (True, '/usr/local/cuda-11.7/bin/ptxas'))

    def test_old_driver_without_matching_ptxas_reports_failure(self):
        self.assertEqual(self.ptxas({'/usr/local/cuda-12.1/bin/ptxas': 12010}, 11070), (False, None))

    def test_new_driver_or_explicit_path_is_left_alone(self):
        self.assertEqual(self.ptxas({'/usr/local/cuda-11.7/bin/ptxas': 11070}, 12020), (True, None))
        self.assertEqual(self.ptxas({}, 11070, {'TRITON_PTXAS_PATH': '/x'}), (True, '/x'))

    def test_ptxas_version_parse(self):
        result = mock.Mock(stdout='Cuda compilation tools, release 11.7, V11.7.64')
        with mock.patch.object(jit.subprocess, 'run', return_value=result):
            self.assertEqual(jit.ptxas_version('ptxas'), 11070)

    def test_compile_mode_validated_and_off_by_default(self):
        cfg = dict(width=4, height=4, channels=3, dim=16, depth=2, heads=2, p_width=2, p_height=2,
                   model_type='jit', batch_size=2, steps=1, lr=1e-3, optimizer_type='adam',
                   dataset_path='x', sampling_steps=2, use_amp=False)
        jit.apply_config_defaults(cfg)
        self.assertEqual(cfg['compile_mode'], 'off')
        self.assertEqual(cfg['ema_warmup'], 'legacy')
        jit.validate_config(cfg)
        with self.assertRaises(jit.ConfigError):
            jit.validate_config(dict(cfg, compile_mode='fast'))
        model = nn.Linear(1, 1)
        jit.maybe_compile(model, cfg, 'cuda')  # off: untouched
        self.assertIsNone(getattr(model, '_compiled_call_impl', None))

    def test_enabled_compile_path(self):
        cfg = {'compile_mode': 'max-autotune-no-cudagraphs'}
        model = nn.Linear(1, 1)
        with mock.patch.object(jit, 'configure_triton_ptxas', return_value=True), \
                mock.patch.object(model, 'compile') as compile_, mock.patch('builtins.print'):
            jit.maybe_compile(model, cfg, torch.device('cuda'))
            compile_.assert_called_once_with(mode='max-autotune-no-cudagraphs')
            jit.maybe_compile(model, dict(cfg, compile_mode='default'), 'cuda')
            compile_.assert_called_with(mode=None)
        with mock.patch.object(jit, 'configure_triton_ptxas', return_value=False), \
                mock.patch.object(model, 'compile') as compile_, mock.patch('builtins.print'):
            jit.maybe_compile(model, cfg, 'cuda')  # no usable ptxas: stays eager
            jit.maybe_compile(model, cfg, torch.device('cpu'))
            compile_.assert_not_called()


if __name__ == '__main__':
    unittest.main()
