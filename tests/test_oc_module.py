from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp_common.models import ObservationConditioner


class ObservationConditionerTest(unittest.TestCase):
    def test_forward_returns_projected_final_hidden_state(self) -> None:
        torch.manual_seed(0)
        model = ObservationConditioner(state_dim=3, lstm_hidden=5, latent_dim=4)
        obs_window = torch.randn(2, 6, 3)

        latent = model(obs_window)
        _, (hidden, _) = model.lstm(obs_window)
        expected = model.projection(hidden[-1])

        self.assertEqual(tuple(latent.shape), (2, 4))
        self.assertTrue(torch.allclose(latent, expected, atol=1e-6, rtol=1e-6))

    def test_forward_supports_backward(self) -> None:
        torch.manual_seed(1)
        model = ObservationConditioner(state_dim=2, lstm_hidden=4, latent_dim=3)
        obs_window = torch.randn(3, 5, 2, requires_grad=True)

        loss = model(obs_window).pow(2).mean()
        loss.backward()

        self.assertIsNotNone(obs_window.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
