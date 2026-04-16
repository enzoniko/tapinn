from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp_common.trainers import _soft_ao_step


class _ToyTapinn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, 1, bias=False)
        self.generator = nn.Linear(1, 1, bias=False)

    def loss(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        pred = self.generator(latent)
        return torch.mean((pred - target) ** 2)


def _exp_avg_snapshot(optimizer: torch.optim.Adam, parameters: list[torch.nn.Parameter]) -> list[torch.Tensor | None]:
    snapshots: list[torch.Tensor | None] = []
    for parameter in parameters:
        state = optimizer.state.get(parameter, {})
        exp_avg = state.get("exp_avg")
        snapshots.append(None if exp_avg is None else exp_avg.detach().clone())
    return snapshots


class SoftAOFixTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = _ToyTapinn()
        self.enc_optimizer = torch.optim.Adam(self.model.encoder.parameters(), lr=0.05)
        self.gen_optimizer = torch.optim.Adam(self.model.generator.parameters(), lr=0.05)
        self.x = torch.tensor([[1.0], [2.0], [3.0]])
        self.target = torch.tensor([[0.5], [1.0], [1.5]])
        self.encoder_params = list(self.model.encoder.parameters())
        self.generator_params = list(self.model.generator.parameters())

    def _step(self, *, enc_focus: bool) -> None:
        loss = self.model.loss(self.x, self.target)
        _soft_ao_step(
            loss,
            self.enc_optimizer,
            self.gen_optimizer,
            self.encoder_params,
            self.generator_params,
            enc_focus=enc_focus,
        )

    def test_encoder_focus_does_not_change_generator_adam_momentum(self) -> None:
        self._step(enc_focus=False)
        before = _exp_avg_snapshot(self.gen_optimizer, self.generator_params)
        self._step(enc_focus=True)
        after = _exp_avg_snapshot(self.gen_optimizer, self.generator_params)
        self.assertEqual(len(before), len(after))
        for prev, curr in zip(before, after):
            self.assertIsNotNone(prev)
            self.assertTrue(torch.equal(prev, curr))

    def test_generator_focus_does_not_change_encoder_adam_momentum(self) -> None:
        self._step(enc_focus=True)
        before = _exp_avg_snapshot(self.enc_optimizer, self.encoder_params)
        self._step(enc_focus=False)
        after = _exp_avg_snapshot(self.enc_optimizer, self.encoder_params)
        self.assertEqual(len(before), len(after))
        for prev, curr in zip(before, after):
            self.assertIsNotNone(prev)
            self.assertTrue(torch.equal(prev, curr))

    def test_alternating_updates_leave_inactive_optimizer_state_unchanged_each_iteration(self) -> None:
        self._step(enc_focus=True)
        self._step(enc_focus=False)
        prev_enc = _exp_avg_snapshot(self.enc_optimizer, self.encoder_params)
        prev_gen = _exp_avg_snapshot(self.gen_optimizer, self.generator_params)

        for step in range(10):
            enc_focus = step % 2 == 0
            self._step(enc_focus=enc_focus)
            curr_enc = _exp_avg_snapshot(self.enc_optimizer, self.encoder_params)
            curr_gen = _exp_avg_snapshot(self.gen_optimizer, self.generator_params)

            if enc_focus:
                for prev, curr in zip(prev_gen, curr_gen):
                    self.assertTrue(torch.equal(prev, curr))
            else:
                for prev, curr in zip(prev_enc, curr_enc):
                    self.assertTrue(torch.equal(prev, curr))

            prev_enc = curr_enc
            prev_gen = curr_gen


if __name__ == "__main__":
    unittest.main()
