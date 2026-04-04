from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from scipy.integrate import solve_ivp


@dataclass
class ODESystemData:
    name: str
    param_name: str
    times: np.ndarray
    params: np.ndarray
    states: np.ndarray
    metadata: dict[str, object] | None = None


@dataclass
class PDESystemData:
    name: str
    param_name: str
    times: np.ndarray
    space: np.ndarray
    params: np.ndarray
    fields: np.ndarray
    metadata: dict[str, object] | None = None


def duffing_rhs_np(t: float, state: np.ndarray, forcing_amp: float) -> list[float]:
    x, v = state
    delta = 0.3
    alpha = -1.0
    beta = 1.0
    omega = 1.0
    return [v, forcing_amp * np.cos(omega * t) - delta * v - alpha * x - beta * x**3]


def lorenz_rhs_np(t: float, state: np.ndarray, rho: float) -> list[float]:
    x, y, z = state
    sigma = 10.0
    beta = 8.0 / 3.0
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def kuramoto_rhs_np(
    t: float,
    phases: np.ndarray,
    coupling_strength: float,
    natural_frequencies: np.ndarray,
) -> np.ndarray:
    del t
    phase_diffs = phases[None, :] - phases[:, None]
    coupling_term = np.sin(phase_diffs).sum(axis=1) / len(phases)
    return natural_frequencies + coupling_strength * coupling_term


def _solve_trajectory(
    rhs: Callable[[float, np.ndarray], np.ndarray | list[float]],
    y0: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    solution = solve_ivp(rhs, (float(times[0]), float(times[-1])), y0, t_eval=times, max_step=0.05)
    if solution.status != 0:
        raise RuntimeError("ODE solver failed to converge.")
    return solution.y.T


def generate_duffing_dataset(
    forcing_values: list[float],
    num_trajectories: int,
    num_points: int,
    t_span: tuple[float, float],
    seed: int,
) -> ODESystemData:
    rng = np.random.default_rng(seed)
    times = np.linspace(t_span[0], t_span[1], num_points, dtype=np.float32)
    states = []
    params = []
    for forcing_amp in forcing_values:
        for _ in range(num_trajectories):
            y0 = rng.uniform(low=-1.0, high=1.0, size=2)
            traj = _solve_trajectory(lambda t, y: duffing_rhs_np(t, y, forcing_amp), y0, times)
            states.append(traj.astype(np.float32))
            params.append(float(forcing_amp))
    return ODESystemData("duffing", "forcing_amp", times, np.asarray(params, dtype=np.float32), np.asarray(states))


def generate_lorenz_dataset(
    rho_values: list[float],
    num_trajectories: int,
    num_points: int,
    t_span: tuple[float, float],
    seed: int,
) -> ODESystemData:
    rng = np.random.default_rng(seed)
    times = np.linspace(t_span[0], t_span[1], num_points, dtype=np.float32)
    states = []
    params = []
    for rho in rho_values:
        for _ in range(num_trajectories):
            y0 = rng.uniform(low=-10.0, high=10.0, size=3)
            traj = _solve_trajectory(lambda t, y: lorenz_rhs_np(t, y, rho), y0, times)
            states.append(traj.astype(np.float32))
            params.append(float(rho))
    return ODESystemData("lorenz", "rho", times, np.asarray(params, dtype=np.float32), np.asarray(states))


def generate_kuramoto_dataset(
    coupling_values: list[float],
    num_trajectories: int,
    num_points: int,
    t_span: tuple[float, float],
    num_oscillators: int,
    seed: int,
) -> ODESystemData:
    rng = np.random.default_rng(seed)
    times = np.linspace(t_span[0], t_span[1], num_points, dtype=np.float32)
    states = []
    params = []
    natural_frequencies = []
    for coupling_strength in coupling_values:
        for _ in range(num_trajectories):
            y0 = rng.uniform(low=-np.pi, high=np.pi, size=num_oscillators)
            frequencies = np.sort(rng.normal(loc=0.0, scale=0.35, size=num_oscillators))
            traj = _solve_trajectory(
                lambda t, y: kuramoto_rhs_np(t, y, coupling_strength, frequencies),
                y0,
                times,
            )
            states.append(traj.astype(np.float32))
            params.append(float(coupling_strength))
            natural_frequencies.append(frequencies.astype(np.float32))
    return ODESystemData(
        "kuramoto",
        "coupling_strength",
        times,
        np.asarray(params, dtype=np.float32),
        np.asarray(states),
        metadata={"natural_frequencies": np.asarray(natural_frequencies, dtype=np.float32)},
    )


def _periodic_dx(u: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1)) / (2.0 * dx)


def _periodic_dxx(u: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(u, -1, axis=-1) - 2.0 * u + np.roll(u, 1, axis=-1)) / (dx * dx)


def _periodic_dxxxx(u: np.ndarray, dx: float) -> np.ndarray:
    return (
        np.roll(u, -2, axis=-1)
        - 4.0 * np.roll(u, -1, axis=-1)
        + 6.0 * u
        - 4.0 * np.roll(u, 1, axis=-1)
        + np.roll(u, 2, axis=-1)
    ) / (dx**4)


def _fourier_wavenumbers(nx: int, length: float) -> np.ndarray:
    return 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)


def _spectral_dx(u: np.ndarray, wave_numbers: np.ndarray) -> np.ndarray:
    return np.fft.ifft(1j * wave_numbers * np.fft.fft(u)).real


def _spectral_dxx(u: np.ndarray, wave_numbers: np.ndarray) -> np.ndarray:
    return np.fft.ifft(-(wave_numbers**2) * np.fft.fft(u)).real


def _spectral_dxxxx(u: np.ndarray, wave_numbers: np.ndarray) -> np.ndarray:
    return np.fft.ifft((wave_numbers**4) * np.fft.fft(u)).real


def _random_periodic_profile(space: np.ndarray, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    profile = (
        0.6 * np.sin(space)
        + 0.35 * np.sin(2.0 * space + rng.uniform(0.0, np.pi))
        + 0.15 * np.cos(3.0 * space + rng.uniform(0.0, np.pi))
    )
    profile += 0.05 * rng.normal(size=space.shape)
    return scale * profile.astype(np.float32)


def simulate_allen_cahn(reaction: float, nx: int, nt: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    length = 2.0 * np.pi
    space = np.linspace(0.0, length, nx, endpoint=False, dtype=np.float32)
    times = np.linspace(0.0, 1.0, nt, dtype=np.float32)
    wave_numbers = _fourier_wavenumbers(nx, length)
    dt = float(times[1] - times[0]) / 8.0
    epsilon = 0.08
    field = np.zeros((nt, nx), dtype=np.float32)
    field[0] = _random_periodic_profile(space, rng, scale=0.9)
    for i in range(nt - 1):
        u = field[i].astype(np.float64)
        for _ in range(8):
            nonlinear_hat = np.fft.fft(reaction * (u - u**3))
            u_hat = np.fft.fft(u)
            updated_hat = (u_hat + dt * nonlinear_hat) / (1.0 + dt * epsilon**2 * (wave_numbers**2))
            u = np.fft.ifft(updated_hat).real
        field[i + 1] = u.astype(np.float32)
    return times, space, field


def simulate_burgers(viscosity: float, nx: int, nt: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    length = 2.0 * np.pi
    space = np.linspace(0.0, length, nx, endpoint=False, dtype=np.float32)
    times = np.linspace(0.0, 1.0, nt, dtype=np.float32)
    wave_numbers = _fourier_wavenumbers(nx, length)
    dt = float(times[1] - times[0]) / 10.0
    field = np.zeros((nt, nx), dtype=np.float32)
    field[0] = _random_periodic_profile(space, rng, scale=0.8)
    for i in range(nt - 1):
        u = field[i].astype(np.float64)
        for _ in range(10):
            ux = _spectral_dx(u, wave_numbers)
            nonlinear_hat = np.fft.fft(-u * ux)
            u_hat = np.fft.fft(u)
            updated_hat = (u_hat + dt * nonlinear_hat) / (1.0 + dt * viscosity * (wave_numbers**2))
            u = np.fft.ifft(updated_hat).real
        field[i + 1] = u.astype(np.float32)
    return times, space, field


def simulate_kuramoto_sivashinsky(diffusion: float, nx: int, nt: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    length = 16.0 * np.pi
    space = np.linspace(0.0, length, nx, endpoint=False, dtype=np.float32)
    times = np.linspace(0.0, 1.2, nt, dtype=np.float32)
    steps_per_output = 8
    dt = float(times[1] - times[0]) / steps_per_output
    wave_numbers = (2.0 * np.pi / length) * np.fft.fftfreq(nx, d=1.0 / nx)
    linear_operator = wave_numbers**2 - diffusion * wave_numbers**4
    g = -0.5j * wave_numbers

    m = 16
    roots = np.exp(1j * np.pi * (np.arange(1, m + 1) - 0.5) / m)
    lr = dt * linear_operator[:, None] + roots[None, :]
    e = np.exp(dt * linear_operator)
    e2 = np.exp(dt * linear_operator / 2.0)
    q = dt * np.real(np.mean((np.exp(lr / 2.0) - 1.0) / lr, axis=1))
    f1 = dt * np.real(np.mean((-4.0 - lr + np.exp(lr) * (4.0 - 3.0 * lr + lr**2)) / (lr**3), axis=1))
    f2 = dt * np.real(np.mean((2.0 + lr + np.exp(lr) * (-2.0 + lr)) / (lr**3), axis=1))
    f3 = dt * np.real(np.mean((-4.0 - 3.0 * lr - lr**2 + np.exp(lr) * (4.0 - lr)) / (lr**3), axis=1))

    u0 = (
        0.15 * np.cos(space / 8.0)
        + 0.1 * np.sin(space / 16.0 + rng.uniform(0.0, np.pi))
        + 0.05 * np.cos(space / 4.0 + rng.uniform(0.0, np.pi))
    ).astype(np.float64)
    v = np.fft.fft(u0)
    field = np.zeros((nt, nx), dtype=np.float32)
    field[0] = u0.astype(np.float32)
    for out_idx in range(1, nt):
        for _ in range(steps_per_output):
            u = np.fft.ifft(v).real
            nv = g * np.fft.fft(u**2)
            a = e2 * v + q * nv
            ua = np.fft.ifft(a).real
            na = g * np.fft.fft(ua**2)
            b = e2 * v + q * na
            ub = np.fft.ifft(b).real
            nb = g * np.fft.fft(ub**2)
            c = e2 * a + q * (2.0 * nb - nv)
            uc = np.fft.ifft(c).real
            nc = g * np.fft.fft(uc**2)
            v = e * v + nv * f1 + 2.0 * (na + nb) * f2 + nc * f3
        field[out_idx] = np.fft.ifft(v).real.astype(np.float32)
    return times, space, field


def generate_allen_cahn_dataset(reaction_values: list[float], num_samples: int, nx: int, nt: int, seed: int) -> PDESystemData:
    fields = []
    params = []
    times = None
    space = None
    for offset, reaction in enumerate(reaction_values):
        for sample_idx in range(num_samples):
            times, space, field = simulate_allen_cahn(reaction, nx, nt, seed + 37 * offset + sample_idx)
            fields.append(field)
            params.append(float(reaction))
    return PDESystemData(
        "allen_cahn",
        "reaction",
        times,
        space,
        np.asarray(params, dtype=np.float32),
        np.asarray(fields),
        metadata={"boundary": "periodic"},
    )


def generate_burgers_dataset(viscosity_values: list[float], num_samples: int, nx: int, nt: int, seed: int) -> PDESystemData:
    fields = []
    params = []
    times = None
    space = None
    for offset, viscosity in enumerate(viscosity_values):
        for sample_idx in range(num_samples):
            times, space, field = simulate_burgers(viscosity, nx, nt, seed + 41 * offset + sample_idx)
            fields.append(field)
            params.append(float(viscosity))
    return PDESystemData(
        "burgers",
        "viscosity",
        times,
        space,
        np.asarray(params, dtype=np.float32),
        np.asarray(fields),
        metadata={"boundary": "periodic"},
    )


def generate_kuramoto_sivashinsky_dataset(
    diffusion_values: list[float],
    num_samples: int,
    nx: int,
    nt: int,
    seed: int,
) -> PDESystemData:
    fields = []
    params = []
    times = None
    space = None
    for offset, diffusion in enumerate(diffusion_values):
        for sample_idx in range(num_samples):
            times, space, field = simulate_kuramoto_sivashinsky(diffusion, nx, nt, seed + 43 * offset + sample_idx)
            fields.append(field)
            params.append(float(diffusion))
    return PDESystemData(
        "kuramoto_sivashinsky",
        "diffusion",
        times,
        space,
        np.asarray(params, dtype=np.float32),
        np.asarray(fields),
        metadata={"boundary": "periodic"},
    )


def ode_rhs_torch(
    problem_name: str,
    t: torch.Tensor,
    y: torch.Tensor,
    param: torch.Tensor,
    metadata: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if problem_name == "duffing":
        x = y[:, 0]
        v = y[:, 1]
        rhs = torch.stack(
            [
                v,
                param * torch.cos(t) - 0.3 * v + x - x**3,
            ],
            dim=1,
        )
        return rhs

    if problem_name == "lorenz":
        x = y[:, 0]
        yy = y[:, 1]
        z = y[:, 2]
        sigma = 10.0
        beta = 8.0 / 3.0
        rhs = torch.stack(
            [
                sigma * (yy - x),
                x * (param - z) - yy,
                x * yy - beta * z,
            ],
            dim=1,
        )
        return rhs

    if problem_name == "kuramoto":
        phase_diffs = y.unsqueeze(2) - y.unsqueeze(1)
        interaction = torch.sin(phase_diffs).sum(dim=2) / y.shape[1]
        if metadata is None or "natural_frequencies" not in metadata:
            raise ValueError("Kuramoto residual requires per-sample natural frequencies.")
        base_frequencies = metadata["natural_frequencies"]
        return base_frequencies + param.unsqueeze(1) * interaction

    raise ValueError(f"Unsupported ODE problem: {problem_name}")


def _gradient(values: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    gradient = torch.autograd.grad(
        values,
        inputs,
        grad_outputs=torch.ones_like(values),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]
    return gradient


def compute_ode_residual(
    problem_name: str,
    coords: torch.Tensor,
    y_pred: torch.Tensor,
    param: torch.Tensor,
    metadata: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if coords.dim() == 1:
        coords = coords.unsqueeze(1)
    t = coords[:, 0]
    derivatives = []
    for dim in range(y_pred.shape[1]):
        derivatives.append(_gradient(y_pred[:, dim], coords)[:, 0])
    dy_dt = torch.stack(derivatives, dim=1)
    rhs = ode_rhs_torch(problem_name, t, y_pred, param, metadata=metadata)
    return dy_dt - rhs


def compute_pde_residual(problem_name: str, coords: torch.Tensor, u_pred: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    grads = _gradient(u_pred, coords)
    u_t = grads[:, 0]
    u_x = grads[:, 1]
    u_xx = _gradient(u_x, coords)[:, 1]

    if problem_name == "allen_cahn":
        epsilon = 0.08
        return u_t - (epsilon**2 * u_xx + param * (u_pred - u_pred**3))

    if problem_name == "burgers":
        return u_t + u_pred * u_x - param * u_xx

    if problem_name == "kuramoto_sivashinsky":
        u_xxxx = _gradient(_gradient(u_xx, coords)[:, 1], coords)[:, 1]
        return u_t + u_pred * u_x + u_xx + param * u_xxxx

    raise ValueError(f"Unsupported PDE problem: {problem_name}")
