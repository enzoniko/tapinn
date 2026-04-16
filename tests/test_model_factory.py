from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp_common.models import create_model


def _assert_has_gradients(test_case: unittest.TestCase, model: torch.nn.Module) -> None:
    test_case.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


class ModelFactoryTest(unittest.TestCase):
    def test_create_model_rejects_unknown_family(self) -> None:
        with self.assertRaises(ValueError):
            create_model("UnknownModel", coord_dim=1, output_dim=1)

    def test_all_ten_model_configs_support_forward_and_backward(self) -> None:
        torch.manual_seed(0)
        batch_size = 3
        obs_window = torch.randn(batch_size, 5, 3, requires_grad=True)
        coords = torch.randn(batch_size, 2, requires_grad=True)
        grid = torch.randn(batch_size, 9, 1, requires_grad=True)
        scalar_param = torch.randn(batch_size, requires_grad=True)
        branch_input = torch.randn(batch_size, 15, requires_grad=True)

        configs = [
            ("StandardPINN", False, {"coord_dim": 2, "output_dim": 1, "hidden_dim": 8}),
            ("StandardPINN", True, {"coord_dim": 2, "output_dim": 1, "hidden_dim": 8, "state_dim": 3, "lstm_hidden": 6, "latent_dim": 4}),
            ("HyperPINN", False, {"coord_dim": 2, "output_dim": 1, "hidden_dim": 8}),
            ("HyperPINN", True, {"coord_dim": 2, "output_dim": 1, "hidden_dim": 8, "state_dim": 3, "lstm_hidden": 6, "latent_dim": 4}),
            ("HyperLRPINN", False, {"coord_dim": 2, "output_dim": 1, "hidden_dim": 8, "rank": 2}),
            ("HyperLRPINN", True, {"coord_dim": 2, "output_dim": 1, "hidden_dim": 8, "rank": 2, "state_dim": 3, "lstm_hidden": 6, "latent_dim": 4}),
            ("DeepONet", False, {"branch_input_dim": 15, "coord_dim": 2, "output_dim": 1, "hidden_dim": 8, "basis_dim": 4}),
            ("DeepONet", True, {"coord_dim": 2, "output_dim": 1, "hidden_dim": 8, "basis_dim": 4, "state_dim": 3, "lstm_hidden": 6, "latent_dim": 4}),
            ("FNO", False, {"branch_input_dim": 15, "grid_size": 9, "output_dim": 1, "width": 8, "modes": 4}),
            ("FNO", True, {"grid_size": 9, "output_dim": 1, "width": 8, "modes": 4, "state_dim": 3, "lstm_hidden": 6, "latent_dim": 4}),
        ]

        for family, with_oc, config in configs:
            with self.subTest(family=family, with_oc=with_oc):
                model = create_model(family, with_oc=with_oc, **config)

                if family == "DeepONet" and not with_oc:
                    pred = model(branch_input, coords)
                    self.assertEqual(tuple(pred.shape), (batch_size, 1))
                elif family == "FNO" and not with_oc:
                    pred = model(branch_input, grid)
                    self.assertEqual(tuple(pred.shape), (batch_size, 9, 1))
                elif family == "FNO" and with_oc:
                    pred = model(grid, obs_window)
                    self.assertEqual(tuple(pred.shape), (batch_size, 9, 1))
                else:
                    model_input = coords if family != "StandardPINN" or with_oc else coords
                    conditioning = obs_window if with_oc else scalar_param
                    pred = model(model_input, conditioning)
                    self.assertEqual(tuple(pred.shape), (batch_size, 1))

                loss = pred.pow(2).mean()
                loss.backward()
                _assert_has_gradients(self, model)

                self.assertEqual(hasattr(model, "observation_conditioner"), with_oc)
                if family == "FNO":
                    self.assertFalse(getattr(model, "has_physics_loss"))
                else:
                    self.assertTrue(getattr(model, "has_physics_loss", True))


if __name__ == "__main__":
    unittest.main()
