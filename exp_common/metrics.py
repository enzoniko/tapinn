from __future__ import annotations

import statistics
import time
from typing import Iterable

import numpy as np
import torch


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    values = list(float(v) for v in values)
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def timed_call(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return out, elapsed


def _dx(values: np.ndarray, step: float, axis: int) -> np.ndarray:
    return np.gradient(values, step, axis=axis, edge_order=2)


def numerical_ode_residual(
    problem_name: str,
    times: np.ndarray,
    states: np.ndarray,
    param: float,
    metadata: dict[str, np.ndarray] | None = None,
) -> float:
    dt = float(times[1] - times[0])
    derivatives = _dx(states, dt, axis=0)
    if problem_name == "duffing":
        x = states[:, 0]
        v = states[:, 1]
        rhs = np.stack([v, param * np.cos(times) - 0.3 * v + x - x**3], axis=1)
    elif problem_name == "lorenz":
        x = states[:, 0]
        y = states[:, 1]
        z = states[:, 2]
        rhs = np.stack([10.0 * (y - x), x * (param - z) - y, x * y - (8.0 / 3.0) * z], axis=1)
    elif problem_name == "kuramoto":
        diffs = states[:, :, None] - states[:, None, :]
        interaction = np.sin(diffs).sum(axis=2) / states.shape[1]
        if metadata is None or "natural_frequencies" not in metadata:
            raise ValueError("Kuramoto residual requires per-sample natural frequencies.")
        base = np.asarray(metadata["natural_frequencies"], dtype=np.float32)[None, :]
        rhs = base + param * interaction
    else:
        raise ValueError(f"Unsupported ODE problem: {problem_name}")
    return float(np.mean((derivatives - rhs) ** 2))


def _periodic_dx(field: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2.0 * dx)


def _periodic_dxx(field: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(field, -1, axis=1) - 2.0 * field + np.roll(field, 1, axis=1)) / (dx * dx)


def _periodic_dxxxx(field: np.ndarray, dx: float) -> np.ndarray:
    return (
        np.roll(field, -2, axis=1)
        - 4.0 * np.roll(field, -1, axis=1)
        + 6.0 * field
        - 4.0 * np.roll(field, 1, axis=1)
        + np.roll(field, 2, axis=1)
    ) / (dx**4)


def numerical_pde_residual(
    problem_name: str,
    times: np.ndarray,
    space: np.ndarray,
    field: np.ndarray,
    param: float,
    boundary: str = "periodic",
) -> float:
    dt = float(times[1] - times[0])
    dx = float(space[1] - space[0])
    u_t = _dx(field, dt, axis=0)
    if boundary == "periodic":
        u_x = _periodic_dx(field, dx)
        u_xx = _periodic_dxx(field, dx)
    else:
        u_x = _dx(field, dx, axis=1)
        u_xx = _dx(u_x, dx, axis=1)

    if problem_name == "allen_cahn":
        residual = u_t - (0.08**2 * u_xx + param * (field - field**3))
    elif problem_name == "burgers":
        residual = u_t + field * u_x - param * u_xx
    elif problem_name == "kuramoto_sivashinsky":
        u_xxxx = _periodic_dxxxx(field, dx) if boundary == "periodic" else _dx(_dx(u_xx, dx, axis=1), dx, axis=1)
        residual = u_t + field * u_x + u_xx + param * u_xxxx
    else:
        raise ValueError(f"Unsupported PDE problem: {problem_name}")
    return float(np.mean(residual**2))


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
