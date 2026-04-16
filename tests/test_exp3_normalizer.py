from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_common.experiments import _predict_exp3_model
from exp_common.trainers import StateNormalizer


class _FakeTapinn:
    def eval(self):
        return self

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.zeros(observations.shape[0], 1, dtype=observations.dtype)

    def decode(self, coords: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros(coords.shape[0], 1, dtype=coords.dtype, device=coords.device)


class Exp3NormalizerTest(unittest.TestCase):
    def test_predict_denormalizes_tapinn_outputs(self) -> None:
        model = _FakeTapinn()
        observations = torch.zeros(2, 3, 1)
        coords = torch.zeros(2, 4, 1)
        params = torch.zeros(2)
        state_normalizer = StateNormalizer(state_mins=torch.tensor([10.0]), state_maxs=torch.tensor([20.0]))

        preds, _ = _predict_exp3_model(
            "tapinn",
            model,
            observations,
            coords,
            params,
            grid_size=4,
            device=torch.device("cpu"),
            state_normalizer=state_normalizer,
        )

        self.assertEqual(preds.shape, (2, 4, 1))
        self.assertTrue(torch.allclose(torch.tensor(preds), torch.full((2, 4, 1), 15.0)))

    def test_predict_requires_state_normalizer(self) -> None:
        model = _FakeTapinn()
        observations = torch.zeros(1, 2, 1)
        coords = torch.zeros(1, 3, 1)
        params = torch.zeros(1)

        with self.assertRaisesRegex(ValueError, "state_normalizer"):
            _predict_exp3_model(
                "tapinn",
                model,
                observations,
                coords,
                params,
                grid_size=3,
                device=torch.device("cpu"),
                state_normalizer=None,
            )


if __name__ == "__main__":
    unittest.main()
