from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp_common.models import create_model
from exp_common.trainers import _soft_ao_step, _triplet_loss, train_direct_model


class OCAwareTrainingTest(unittest.TestCase):
    device: torch.device = torch.device("cpu")
    observations: torch.Tensor = torch.empty(0)
    coords: torch.Tensor = torch.empty(0)
    targets: torch.Tensor = torch.empty(0)
    params: torch.Tensor = torch.empty(0)
    trajectory_ids: torch.Tensor = torch.empty(0, dtype=torch.long)

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = torch.device("cpu")
        self.observations = torch.randn(4, 5, 2)
        self.coords = torch.linspace(-1.0, 1.0, 6).view(1, 6, 1).repeat(4, 1, 1)
        self.targets = torch.randn(4, 6, 1)
        self.params = torch.tensor([0.2, 0.2, 0.5, 0.5], dtype=torch.float32)
        self.trajectory_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    def test_oc_physics_models_use_triplet_and_soft_ao(self) -> None:
        model = create_model(
            "StandardPINN",
            with_oc=True,
            coord_dim=1,
            output_dim=1,
            hidden_dim=8,
            state_dim=2,
            lstm_hidden=6,
            latent_dim=4,
        )

        with patch("exp_common.trainers.compute_ode_residual", side_effect=lambda *args, **kwargs: args[2][:, 0] * 0.0), \
             patch("exp_common.trainers._triplet_loss", wraps=_triplet_loss) as triplet_mock, \
             patch("exp_common.trainers._soft_ao_step", wraps=_soft_ao_step) as soft_ao_mock:
            result = train_direct_model(
                model,
                problem_name="duffing",
                model_kind="standardpinn",
                observations=self.observations,
                coords=self.coords,
                targets=self.targets,
                params=self.params,
                device=self.device,
                epochs=1,
                batch_size=4,
                lr=1e-3,
                use_lra=False,
                beta_metric=0.2,
                alternating=True,
                ao_warmup_epochs=0,
                trajectory_ids=self.trajectory_ids,
            )

        self.assertEqual(result.epochs_trained, 1)
        self.assertGreater(triplet_mock.call_count, 0)
        self.assertGreater(soft_ao_mock.call_count, 0)
        self.assertIn("metric_loss", result.history[0])

    def test_fno_oc_uses_data_plus_triplet_without_physics_or_soft_ao(self) -> None:
        model = create_model(
            "FNO",
            with_oc=True,
            grid_size=6,
            output_dim=1,
            width=8,
            modes=4,
            state_dim=2,
            lstm_hidden=6,
            latent_dim=3,
        )

        with patch("exp_common.trainers.compute_ode_residual") as ode_mock, \
             patch("exp_common.trainers.compute_pde_residual") as pde_mock, \
             patch("exp_common.trainers._triplet_loss", wraps=_triplet_loss) as triplet_mock, \
             patch("exp_common.trainers._soft_ao_step", wraps=_soft_ao_step) as soft_ao_mock:
            result = train_direct_model(
                model,
                problem_name="allen_cahn",
                model_kind="fno",
                observations=self.observations,
                coords=self.coords,
                targets=self.targets,
                params=self.params,
                device=self.device,
                epochs=1,
                batch_size=4,
                lr=1e-3,
                use_lra=False,
                beta_metric=0.2,
                alternating=True,
                ao_warmup_epochs=0,
                trajectory_ids=self.trajectory_ids,
            )

        self.assertEqual(result.epochs_trained, 1)
        self.assertGreater(triplet_mock.call_count, 0)
        self.assertEqual(soft_ao_mock.call_count, 0)
        self.assertEqual(ode_mock.call_count, 0)
        self.assertEqual(pde_mock.call_count, 0)
        self.assertEqual(result.history[0]["physics_mse"], 0.0)
        self.assertIn("metric_loss", result.history[0])

    def test_non_oc_models_skip_triplet_and_soft_ao(self) -> None:
        model = create_model("StandardPINN", with_oc=False, coord_dim=1, output_dim=1, hidden_dim=8)

        with patch("exp_common.trainers.compute_ode_residual", side_effect=lambda *args, **kwargs: args[2][:, 0] * 0.0), \
             patch("exp_common.trainers._triplet_loss", wraps=_triplet_loss) as triplet_mock, \
             patch("exp_common.trainers._soft_ao_step", wraps=_soft_ao_step) as soft_ao_mock:
            result = train_direct_model(
                model,
                problem_name="duffing",
                model_kind="standardpinn",
                observations=self.observations,
                coords=self.coords,
                targets=self.targets,
                params=self.params,
                device=self.device,
                epochs=1,
                batch_size=4,
                lr=1e-3,
                use_lra=False,
                alternating=True,
                ao_warmup_epochs=0,
            )

        self.assertEqual(result.epochs_trained, 1)
        self.assertEqual(triplet_mock.call_count, 0)
        self.assertEqual(soft_ao_mock.call_count, 0)
        self.assertNotIn("metric_loss", result.history[0])


if __name__ == "__main__":
    unittest.main()
