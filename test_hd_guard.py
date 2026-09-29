"""Regression checks for the divergence guard shared by AdamHD, MuonHD and NorMuonHD.

A noisy but healthy loss used to trip the loss-based rollback over and over;
each rollback lowered a permanent LR ceiling that also overrode the LR floor,
so the LR ratcheted down to ~1e-8 and below.

Run with: python -B -m unittest -v test_hd_guard
"""
import contextlib
import io
import math
import random
import unittest

import torch

from lamb import AdamHD, MuonHD, NorMuonHD, muon_param_groups

BASE_LR = 1e-3
FLOOR = 1e-2 * BASE_LR   # default lower multiplier bound without min_lr


def controller(seed=0):
    """AdamHD on one dummy tensor; the tests drive its LR controller directly."""
    random.seed(seed)
    return AdamHD([torch.nn.Parameter(torch.zeros(4))], lr=BASE_LR)


def feed(opt, steps, loss_fn, gsq_fn=lambda: 1.0, finite=True):
    """Push synthetic loss / hypergradient statistics through the controller."""
    lrs = []
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(steps):
            opt.observe_loss(loss_fn())
            dot = random.gauss(0, 0.3) if finite else float('nan')
            opt._hd_adapt(dot, gsq_fn(), 1.0)
            lrs.append(opt.param_groups[0]['lr'])
    return lrs


def noisy(level=1.0, sigma=0.35):
    return lambda: level * math.exp(random.gauss(0, sigma))


class ControllerTests(unittest.TestCase):
    def test_noisy_stationary_loss_never_rolls_back(self):
        # Diffusion-style per-batch noise (±35%) plus rare 10x gradient-norm spikes.
        opt = controller()
        lrs = feed(opt, 5000, noisy(), lambda: 100.0 if random.random() < 0.01 else 1.0)
        group = opt.param_groups[0]
        self.assertEqual(group.get('hd_rollbacks', 0), 0)
        self.assertNotIn('hd_ceiling', group)
        self.assertGreaterEqual(min(lrs), FLOOR * (1 - 1e-9))

    def test_real_divergence_still_rolls_back(self):
        opt = controller()
        feed(opt, 300, noisy(1.0, 0.1))
        feed(opt, 30, noisy(3.0, 0.1))
        group = opt.param_groups[0]
        self.assertGreaterEqual(group.get('hd_rollbacks', 0), 1)
        self.assertIn('hd_ceiling', group)

    def test_persistent_loss_shift_is_accepted_not_ratcheted(self):
        # The loss moves to a new level for good (e.g. a data change): after
        # GUARD_RETRIES rollbacks from one snapshot it becomes the new reference.
        opt = controller()
        feed(opt, 300, noisy(1.0, 0.1))
        lrs = feed(opt, 3000, noisy(3.0, 0.1))
        group = opt.param_groups[0]
        self.assertEqual(group['hd_rollbacks'], AdamHD.GUARD_RETRIES)
        self.assertGreaterEqual(min(lrs), FLOOR * (1 - 1e-9))

    def test_ceiling_is_lifted_after_healthy_steps(self):
        opt = controller()
        feed(opt, 300, noisy(1.0, 0.1))
        feed(opt, 30, noisy(3.0, 0.1))
        self.assertIn('hd_ceiling', opt.param_groups[0])
        feed(opt, 400, noisy(1.0, 0.1))
        self.assertNotIn('hd_ceiling', opt.param_groups[0])

    def test_non_finite_storm_stays_above_floor(self):
        opt = controller()
        feed(opt, 300, noisy(1.0, 0.1))
        lrs = feed(opt, 500, noisy(1.0, 0.1), finite=False)
        self.assertGreaterEqual(opt.param_groups[0]['hd_rollbacks'], 5)
        self.assertGreaterEqual(min(lrs), FLOOR * (1 - 1e-9))

    def test_collapsed_checkpoint_recovers(self):
        # State saved by the old guard: a ceiling far below the floor, no hd_failed.
        opt = controller()
        feed(opt, 100, noisy(1.0, 0.1))
        opt.param_groups[0]['hd_ceiling'] = 1e-5
        lrs = feed(opt, 100, noisy(1.0, 0.1))
        self.assertGreaterEqual(min(lrs), FLOOR * (1 - 1e-9))
        self.assertNotIn('hd_ceiling', opt.param_groups[0])


def train_noisy(make_opt, steps=3000, batch=8, seed=0):
    """Small regression MLP on heavily label-noised data: every batch loss is noisy.

    Returns (clean loss before, clean loss after, lowest LR seen, rollbacks).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    teacher = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
    model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                                torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = make_opt(model)
    x_eval = torch.randn(2048, 16)
    with torch.no_grad():
        y_eval = teacher(x_eval)

    def clean_loss():
        with torch.no_grad():
            return torch.nn.functional.mse_loss(model(x_eval), y_eval).item()

    before, lowest = clean_loss(), math.inf
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(steps):
            x = torch.randn(batch, 16)
            with torch.no_grad():
                y = teacher(x) + 0.5 * torch.randn(batch, 1)
            loss = torch.nn.functional.mse_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.observe_loss(loss.item())
            opt.step()
            lowest = min(lowest, min(g['lr'] / g['hd_base'] for g in opt.param_groups))
    return before, clean_loss(), lowest, opt.param_groups[0].get('hd_rollbacks', 0)


OPTIMIZERS = {
    'AdamHD': lambda m: AdamHD(m.parameters(), lr=BASE_LR),
    'MuonHD': lambda m: MuonHD(muon_param_groups(m), lr=4.2e-4),
    'NorMuonHD': lambda m: NorMuonHD(muon_param_groups(m), lr=4.2e-4),
}


class NoisyTrainingTests(unittest.TestCase):
    def test_noisy_training_keeps_learning(self):
        for name, make in OPTIMIZERS.items():
            with self.subTest(name):
                before, after, lowest, rollbacks = train_noisy(make)
                self.assertLess(after, 0.5 * before, f"{name} stopped learning")
                self.assertGreaterEqual(lowest, 1e-2 * (1 - 1e-9), f"{name} LR fell below its floor")
                self.assertLessEqual(rollbacks, 1, f"{name} rolled back a healthy run")


if __name__ == '__main__':
    unittest.main()
