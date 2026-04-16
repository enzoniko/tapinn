# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp_common.trainers import CoordNormalizer, StateNormalizer
from exp_common.well_adapter import WellAdapter


@pytest.fixture
def mock_data() -> torch.Tensor:
    _ = torch.manual_seed(42)
    return torch.randn(2, 10, 8, 8, 4)


@pytest.fixture
def adapter(mock_data: torch.Tensor) -> WellAdapter:
    return WellAdapter("shear_flow", data_tensor=mock_data)


@pytest.fixture
def structured_data() -> torch.Tensor:
    values = torch.arange(2 * 3 * 2 * 2 * 4, dtype=torch.float32)
    return values.reshape(2, 3, 2, 2, 4)


def test_get_point_cloud_shapes(adapter: WellAdapter) -> None:
    coords, states = adapter.get_point_cloud(0)
    assert coords.shape == (10 * 8 * 8, 3)
    assert states.shape == (10 * 8 * 8, 4)


def test_get_point_cloud_values_finite(adapter: WellAdapter) -> None:
    coords, states = adapter.get_point_cloud(1)
    assert torch.isfinite(coords).all()
    assert torch.isfinite(states).all()


def test_get_point_cloud_preserves_flatten_order(structured_data: torch.Tensor) -> None:
    adapter = WellAdapter("shear_flow", data_tensor=structured_data)
    coords, states = adapter.get_point_cloud(0)
    expected_coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(coords[:5], expected_coords)
    assert torch.equal(states[0], structured_data[0, 0, 0, 0])
    assert torch.equal(states[1], structured_data[0, 0, 0, 1])
    assert torch.equal(states[4], structured_data[0, 1, 0, 0])


def test_get_observation_window_shape(adapter: WellAdapter) -> None:
    window = adapter.get_observation_window(0, window_size=4)
    assert window.shape == (4, 4)


def test_observation_window_matches_spatial_average(structured_data: torch.Tensor) -> None:
    adapter = WellAdapter("shear_flow", data_tensor=structured_data)
    window = adapter.get_observation_window(0, window_size=2)
    expected = structured_data[0, :2].mean(dim=(1, 2))
    assert torch.allclose(window, expected)


def test_get_fno_grid_shape(adapter: WellAdapter) -> None:
    grid = adapter.get_fno_grid(0)
    assert grid.shape == (1, 4, 10 * 8 * 8)


def test_fno_grid_is_3d(adapter: WellAdapter) -> None:
    grid = adapter.get_fno_grid(0)
    assert grid.dim() == 3


def test_fno_grid_flattens_to_channel_first_sequence(structured_data: torch.Tensor) -> None:
    adapter = WellAdapter("shear_flow", data_tensor=structured_data)
    grid = adapter.get_fno_grid(0)
    expected = structured_data[0].reshape(-1, 4).transpose(0, 1).unsqueeze(0)
    assert torch.equal(grid, expected)


def test_create_normalizers_invertible(adapter: WellAdapter, mock_data: torch.Tensor) -> None:
    coord_norm, state_norm = adapter.create_normalizers()
    assert isinstance(coord_norm, CoordNormalizer)
    assert isinstance(state_norm, StateNormalizer)

    coords, states = adapter.get_point_cloud(0)
    assert torch.allclose(coord_norm.denormalize(coord_norm.normalize(coords)), coords, atol=1e-5)
    assert torch.allclose(state_norm.denormalize(state_norm.normalize(states)), states, atol=1e-5)

    flat_states = mock_data.reshape(-1, mock_data.shape[-1])
    assert torch.allclose(state_norm.state_mins, flat_states.min(dim=0).values)
    assert torch.allclose(state_norm.state_maxs, flat_states.max(dim=0).values)


def test_compute_physics_residual_returns_dict(adapter: WellAdapter, mock_data: torch.Tensor) -> None:
    predictions = mock_data[0].reshape(-1, mock_data.shape[-1])
    residuals = adapter.compute_physics_residual(0, predictions)
    assert isinstance(residuals, dict)
    assert residuals
    assert all(isinstance(value, torch.Tensor) for value in residuals.values())
    assert all(value.ndim == 0 for value in residuals.values())
    assert all(torch.isfinite(value) for value in residuals.values())


def test_invalid_traj_idx_raises(adapter: WellAdapter) -> None:
    with pytest.raises(IndexError):
        adapter.get_point_cloud(5)


def test_observation_window_size_clamped(adapter: WellAdapter) -> None:
    window = adapter.get_observation_window(0, window_size=999)
    assert window.shape == (10, 4)


def test_unknown_dataset_name_raises(mock_data: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        WellAdapter("unknown_dataset", data_tensor=mock_data)


def test_compute_physics_residual_rejects_bad_prediction_shape(adapter: WellAdapter) -> None:
    with pytest.raises(ValueError):
        adapter.compute_physics_residual(0, torch.randn(9, 4))
