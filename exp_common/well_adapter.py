# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false, reportAny=false, reportExplicitAny=false, reportUnannotatedClassAttribute=false, reportImplicitStringConcatenation=false, reportMissingTypeStubs=false
from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any
import sys

import torch

from .trainers import CoordNormalizer, StateNormalizer


def _load_residual_classes() -> dict[str, type]:
    physics_root = Path(__file__).resolve().parents[2] / "PI-WELL" / "the_well"
    if str(physics_root) not in sys.path:
        sys.path.insert(0, str(physics_root))

    residuals = importlib.import_module("the_well.physics.residuals")

    return {
        "shear_flow": residuals.ShearFlowResidual,
        "euler_multi_quadrants": residuals.EulerMultiQuadrantsResidual,
        "planet_swe": residuals.PlanetSWEResidual,
        "planetswe": residuals.PlanetSWEResidual,
        "mhd": residuals.MHDResidual,
        "active_matter": residuals.ActiveMatterResidual,
        "viscoelastic_instability": residuals.ViscoelasticInstabilityResidual,
        "helmholtz_staircase": residuals.HelmholtzStaircaseResidual,
    }


RESIDUAL_REGISTRY = _load_residual_classes()


@dataclass(frozen=True)
class _GridShape:
    time: int
    x: int
    y: int
    channels: int

    @property
    def num_points(self) -> int:
        return self.time * self.x * self.y


class WellAdapter:
    """Bridge between PI-WELL grid data and tapinn point-based pipeline."""

    def __init__(
        self,
        dataset_name: str,
        data_tensor: torch.Tensor | None = None,
        max_trajectories: int | None = None,
    ):
        dataset_key = dataset_name.lower()
        if dataset_key not in RESIDUAL_REGISTRY:
            raise ValueError(f"Unsupported Well dataset: {dataset_name}")

        loaded = data_tensor if data_tensor is not None else self._load_dataset(dataset_key, max_trajectories)
        if loaded.dim() != 5:
            raise ValueError("WellAdapter expects data_tensor with shape (N_traj, T, X, Y, C)")

        if max_trajectories is not None:
            loaded = loaded[:max_trajectories]

        self.dataset_name: str = dataset_key
        self.data_tensor: torch.Tensor = loaded.detach().clone().to(dtype=torch.float32)
        self.grid_shape: _GridShape = _GridShape(*self.data_tensor.shape[1:])
        self.residual_evaluator = RESIDUAL_REGISTRY[self.dataset_name]()

    def get_point_cloud(self, traj_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        trajectory = self._get_trajectory(traj_idx)
        coords = self._base_coords(device=trajectory.device)
        states = trajectory.reshape(-1, self.grid_shape.channels)
        return coords, states

    def get_observation_window(self, traj_idx: int, window_size: int) -> torch.Tensor:
        trajectory = self._get_trajectory(traj_idx)
        clamped = max(1, min(int(window_size), self.grid_shape.time))
        return trajectory[:clamped].mean(dim=(1, 2))

    def get_fno_grid(self, traj_idx: int) -> torch.Tensor:
        trajectory = self._get_trajectory(traj_idx)
        return trajectory.reshape(-1, self.grid_shape.channels).transpose(0, 1).unsqueeze(0)

    def compute_physics_residual(self, traj_idx: int, predictions: torch.Tensor) -> dict[str, torch.Tensor]:
        reference = self._get_trajectory(traj_idx)
        pred_grid = self._reshape_predictions(predictions, reference)
        residuals = self.residual_evaluator(
            pred_grid.unsqueeze(0),
            reference.unsqueeze(0),
            metadata=None,
            grid_data=self._grid_data(reference.device),
        )
        return {name: value.reshape(()) for name, value in residuals.items()}

    def create_normalizers(self) -> tuple[CoordNormalizer, StateNormalizer]:
        coord_norm = CoordNormalizer.from_coords(self._base_coords())
        state_norm = StateNormalizer.from_targets(self.data_tensor.reshape(-1, self.grid_shape.channels))
        return coord_norm, state_norm

    def _get_trajectory(self, traj_idx: int) -> torch.Tensor:
        if traj_idx < 0 or traj_idx >= self.data_tensor.shape[0]:
            raise IndexError(f"Trajectory index {traj_idx} out of range for {self.data_tensor.shape[0]} trajectories")
        return self.data_tensor[traj_idx]

    def _base_coords(self, device: torch.device | None = None) -> torch.Tensor:
        t = torch.arange(self.grid_shape.time, dtype=torch.float32, device=device)
        x = torch.arange(self.grid_shape.x, dtype=torch.float32, device=device)
        y = torch.arange(self.grid_shape.y, dtype=torch.float32, device=device)
        mesh_t, mesh_x, mesh_y = torch.meshgrid(t, x, y, indexing="ij")
        return torch.stack((mesh_t, mesh_x, mesh_y), dim=-1).reshape(-1, 3)

    def _reshape_predictions(self, predictions: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        expected_points = self.grid_shape.num_points
        expected_channels = self.grid_shape.channels

        if predictions.dim() == 3 and predictions.shape[0] == 1:
            predictions = predictions.squeeze(0)
            if predictions.shape[0] == expected_channels:
                predictions = predictions.transpose(0, 1)
        elif predictions.dim() == 2 and predictions.shape[0] == expected_channels and predictions.shape[1] == expected_points:
            predictions = predictions.transpose(0, 1)

        if predictions.shape != (expected_points, expected_channels):
            raise ValueError(
                f"Predictions must have shape ({expected_points}, {expected_channels}), "
                f"(1, {expected_channels}, {expected_points}), or ({expected_channels}, {expected_points})"
            )

        return predictions.reshape_as(reference)

    def _grid_data(self, device: torch.device) -> dict[str, torch.Tensor]:
        x = torch.arange(self.grid_shape.x, dtype=torch.float32, device=device)
        y = torch.arange(self.grid_shape.y, dtype=torch.float32, device=device)
        mesh_x, mesh_y = torch.meshgrid(x, y, indexing="ij")
        space_grid = torch.stack((mesh_x, mesh_y), dim=-1)
        time_grid = torch.arange(self.grid_shape.time, dtype=torch.float32, device=device)
        return {"space_grid": space_grid, "output_time_grid": time_grid}

    def _load_dataset(self, dataset_name: str, max_trajectories: int | None) -> torch.Tensor:
        try:
            from the_well.data import WellDataset  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - optional path
            raise ImportError(
                "Loading Well datasets requires the_well to be installed or data_tensor to be provided"
            ) from exc

        dataset = WellDataset(
            well_base_path="hf://datasets/polymathic-ai/",
            well_dataset_name=dataset_name,
            well_split_name="train",
        )

        limit = max_trajectories or len(dataset)
        trajectories: list[torch.Tensor] = []
        for idx in range(min(limit, len(dataset))):
            sample = dataset[idx]
            field = self._extract_field_tensor(sample)
            trajectories.append(field.to(dtype=torch.float32))

        if not trajectories:
            raise ValueError(f"No trajectories loaded for dataset {dataset_name}")

        return torch.stack(trajectories, dim=0)

    def _extract_field_tensor(self, sample: Any) -> torch.Tensor:
        if isinstance(sample, torch.Tensor):
            tensor = sample
        elif isinstance(sample, dict):
            for key in ("targets", "field", "fields", "data", "output"):
                value = sample.get(key)
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break
            else:
                raise ValueError("Unable to locate tensor field in Well sample")
        else:  # pragma: no cover - optional path
            raise ValueError(f"Unsupported Well sample type: {type(sample)!r}")

        if tensor.dim() == 4:
            return tensor
        if tensor.dim() == 5 and tensor.shape[0] == 1:
            return tensor.squeeze(0)
        raise ValueError("Loaded Well sample must reshape to (T, X, Y, C)")
