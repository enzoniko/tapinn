from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp_common.trainers import _triplet_loss


class TripletFixTest(unittest.TestCase):
    def test_duplicate_parameter_values_do_not_create_false_positive_pairs(self) -> None:
        latent = torch.tensor([
            [0.0, 0.0],
            [0.2, 0.0],
            [1.0, 0.0],
            [1.2, 0.0],
        ])
        params = torch.tensor([1.0, 1.0, 2.0, 2.0])
        trajectory_ids = torch.tensor([0, 1, 2, 3])

        loss = _triplet_loss(latent, params, trajectory_ids)

        self.assertEqual(float(loss.item()), 0.0)

    def test_same_trajectory_id_is_positive_even_if_parameter_values_differ(self) -> None:
        latent = torch.tensor([
            [0.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [0.5, 0.0],
        ])
        params = torch.tensor([1.0, 9.0, 3.0, 4.0])
        trajectory_ids = torch.tensor([7, 7, 8, 9])

        loss = _triplet_loss(latent, params, trajectory_ids)

        self.assertGreater(float(loss.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
