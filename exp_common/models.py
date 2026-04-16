from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, depth: int = 3, activation: str = "tanh"):
        super().__init__()
        if activation == "relu":
            act = nn.ReLU
        else:
            act = nn.Tanh

        layers: list[nn.Module] = []
        last_dim = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(last_dim, hidden_dim), act()])
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ObservationConditioner(nn.Module):
    def __init__(self, state_dim: int, lstm_hidden: int, latent_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
        self.projection = nn.Linear(lstm_hidden, latent_dim)

    def forward(self, obs_window: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(obs_window)
        return self.projection(hidden[-1])


class StandardPINN(nn.Module):
    has_physics_loss = True

    def __init__(self, coord_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.model = MLP(coord_dim + 1, output_dim, hidden_dim=hidden_dim, depth=3, activation="tanh")

    def forward(self, coords: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        if coords.dim() == 1:
            coords = coords.unsqueeze(1)
        if param.dim() == 1:
            param = param.unsqueeze(1)
        return self.model(torch.cat([coords, param], dim=1))


class StandardPINN_OC(nn.Module):
    has_physics_loss = True

    def __init__(self, coord_dim: int, output_dim: int, state_dim: int, lstm_hidden: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.observation_conditioner = ObservationConditioner(state_dim=state_dim, lstm_hidden=lstm_hidden, latent_dim=latent_dim)
        self.encoder = self.observation_conditioner
        self.model = MLP(coord_dim + latent_dim, output_dim, hidden_dim=hidden_dim, depth=3, activation="tanh")
        self.generator = self.model

    def encode(self, obs_window: torch.Tensor) -> torch.Tensor:
        return self.observation_conditioner(obs_window)

    def decode(self, coords: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if coords.dim() == 1:
            coords = coords.unsqueeze(1)
        return self.model(torch.cat([coords, latent], dim=-1))

    def forward(self, coords: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        latent = self.observation_conditioner(param)
        return self.decode(coords, latent)


class HyperPINN(nn.Module):
    has_physics_loss = True

    def __init__(self, coord_dim: int, output_dim: int, hidden_dim: int = 64, conditioning_dim: int = 1):
        super().__init__()
        self.coord_dim = coord_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.conditioning_dim = conditioning_dim
        self.n_params = (
            (coord_dim * hidden_dim)
            + hidden_dim
            + (hidden_dim * hidden_dim)
            + hidden_dim
            + (hidden_dim * output_dim)
            + output_dim
        )
        self.hyper = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_params),
        )

    def _unpack(self, vector: torch.Tensor) -> tuple[torch.Tensor, ...]:
        idx = 0
        w1_size = self.coord_dim * self.hidden_dim
        w1 = vector[:, idx : idx + w1_size].reshape(-1, self.coord_dim, self.hidden_dim)
        idx += w1_size
        b1 = vector[:, idx : idx + self.hidden_dim]
        idx += self.hidden_dim
        w2_size = self.hidden_dim * self.hidden_dim
        w2 = vector[:, idx : idx + w2_size].reshape(-1, self.hidden_dim, self.hidden_dim)
        idx += w2_size
        b2 = vector[:, idx : idx + self.hidden_dim]
        idx += self.hidden_dim
        w3_size = self.hidden_dim * self.output_dim
        w3 = vector[:, idx : idx + w3_size].reshape(-1, self.hidden_dim, self.output_dim)
        idx += w3_size
        b3 = vector[:, idx : idx + self.output_dim]
        return w1, b1, w2, b2, w3, b3

    def forward(self, coords: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        if coords.dim() == 1:
            coords = coords.unsqueeze(1)
        if param.dim() == 1:
            param = param.unsqueeze(1)
        generated = self.hyper(param)
        w1, b1, w2, b2, w3, b3 = self._unpack(generated)
        x = torch.bmm(coords.unsqueeze(1), w1).squeeze(1) + b1
        x = torch.tanh(x)
        x = torch.bmm(x.unsqueeze(1), w2).squeeze(1) + b2
        x = torch.tanh(x)
        x = torch.bmm(x.unsqueeze(1), w3).squeeze(1) + b3
        return x


class HyperPINN_OC(HyperPINN):
    has_physics_loss = True

    def __init__(self, coord_dim: int, output_dim: int, state_dim: int, lstm_hidden: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__(coord_dim=coord_dim, output_dim=output_dim, hidden_dim=hidden_dim, conditioning_dim=latent_dim)
        self.observation_conditioner = ObservationConditioner(state_dim=state_dim, lstm_hidden=lstm_hidden, latent_dim=latent_dim)
        self.encoder = self.observation_conditioner
        self.generator = self.hyper

    def encode(self, obs_window: torch.Tensor) -> torch.Tensor:
        return self.observation_conditioner(obs_window)

    def decode(self, coords: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return super().forward(coords, latent)

    def forward(self, coords: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        latent = self.observation_conditioner(param)
        return self.decode(coords, latent)


class HyperLRPINN(nn.Module):
    has_physics_loss = True

    def __init__(self, coord_dim: int, output_dim: int, hidden_dim: int = 64, rank: int = 4, conditioning_dim: int = 1):
        super().__init__()
        self.coord_dim = coord_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.conditioning_dim = conditioning_dim

        self.layer1 = nn.Linear(coord_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

        self.n_ab_params = (
            rank * (hidden_dim + coord_dim)
            + rank * (hidden_dim + hidden_dim)
            + rank * (output_dim + hidden_dim)
        )

        self.hyper = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.n_ab_params),
        )

    def _unpack(self, vector: torch.Tensor) -> tuple[torch.Tensor, ...]:
        idx = 0
        s1 = self.rank * self.coord_dim
        a1 = vector[:, idx : idx + s1].reshape(-1, self.rank, self.coord_dim)
        idx += s1
        s2 = self.hidden_dim * self.rank
        b1 = vector[:, idx : idx + s2].reshape(-1, self.hidden_dim, self.rank)
        idx += s2

        s3 = self.rank * self.hidden_dim
        a2 = vector[:, idx : idx + s3].reshape(-1, self.rank, self.hidden_dim)
        idx += s3
        s4 = self.hidden_dim * self.rank
        b2 = vector[:, idx : idx + s4].reshape(-1, self.hidden_dim, self.rank)
        idx += s4

        s5 = self.rank * self.hidden_dim
        a3 = vector[:, idx : idx + s5].reshape(-1, self.rank, self.hidden_dim)
        idx += s5
        s6 = self.output_dim * self.rank
        b3 = vector[:, idx : idx + s6].reshape(-1, self.output_dim, self.rank)
        return a1, b1, a2, b2, a3, b3

    def forward(self, coords: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        if coords.dim() == 1:
            coords = coords.unsqueeze(1)
        if param.dim() == 1:
            param = param.unsqueeze(1)

        generated = self.hyper(param)
        a1, b1, a2, b2, a3, b3 = self._unpack(generated)

        dw1 = torch.bmm(b1, a1)
        dw2 = torch.bmm(b2, a2)
        dw3 = torch.bmm(b3, a3)

        w1 = self.layer1.weight.unsqueeze(0) + dw1
        w2 = self.layer2.weight.unsqueeze(0) + dw2
        w3 = self.layer3.weight.unsqueeze(0) + dw3

        x = torch.bmm(coords.unsqueeze(1), w1.transpose(1, 2)).squeeze(1) + self.layer1.bias
        x = torch.tanh(x)
        x = torch.bmm(x.unsqueeze(1), w2.transpose(1, 2)).squeeze(1) + self.layer2.bias
        x = torch.tanh(x)
        x = torch.bmm(x.unsqueeze(1), w3.transpose(1, 2)).squeeze(1) + self.layer3.bias
        return x


class HyperLRPINN_OC(HyperLRPINN):
    has_physics_loss = True

    def __init__(self, coord_dim: int, output_dim: int, state_dim: int, lstm_hidden: int, latent_dim: int, hidden_dim: int = 64, rank: int = 4):
        super().__init__(coord_dim=coord_dim, output_dim=output_dim, hidden_dim=hidden_dim, rank=rank, conditioning_dim=latent_dim)
        self.observation_conditioner = ObservationConditioner(state_dim=state_dim, lstm_hidden=lstm_hidden, latent_dim=latent_dim)
        self.encoder = self.observation_conditioner
        self.generator = nn.ModuleList([self.hyper, self.layer1, self.layer2, self.layer3])

    def encode(self, obs_window: torch.Tensor) -> torch.Tensor:
        return self.observation_conditioner(obs_window)

    def decode(self, coords: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return super().forward(coords, latent)

    def forward(self, coords: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        latent = self.observation_conditioner(param)
        return self.decode(coords, latent)


class DeepONet(nn.Module):
    has_physics_loss = True

    def __init__(self, branch_input_dim: int, coord_dim: int, output_dim: int, hidden_dim: int = 64, basis_dim: int = 32):
        super().__init__()
        self.branch = MLP(branch_input_dim, basis_dim * output_dim, hidden_dim=hidden_dim, depth=2, activation="relu")
        self.trunk = MLP(coord_dim, basis_dim * output_dim, hidden_dim=hidden_dim, depth=2, activation="tanh")
        self.output_dim = output_dim
        self.basis_dim = basis_dim

    def forward(self, branch_input: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        if coords.dim() == 1:
            coords = coords.unsqueeze(1)
        branch = self.branch(branch_input).reshape(-1, self.output_dim, self.basis_dim)
        trunk = self.trunk(coords).reshape(-1, self.output_dim, self.basis_dim)
        return (branch * trunk).sum(dim=-1)


class DeepONet_OC(nn.Module):
    has_physics_loss = True

    def __init__(self, coord_dim: int, output_dim: int, state_dim: int, lstm_hidden: int, latent_dim: int, hidden_dim: int = 64, basis_dim: int = 32):
        super().__init__()
        self.observation_conditioner = ObservationConditioner(state_dim=state_dim, lstm_hidden=lstm_hidden, latent_dim=latent_dim)
        self.encoder = self.observation_conditioner
        self.branch = MLP(latent_dim, basis_dim * output_dim, hidden_dim=hidden_dim, depth=2, activation="relu")
        self.trunk = MLP(coord_dim, basis_dim * output_dim, hidden_dim=hidden_dim, depth=2, activation="tanh")
        self.generator = nn.ModuleList([self.branch, self.trunk])
        self.output_dim = output_dim
        self.basis_dim = basis_dim

    def encode(self, obs_window: torch.Tensor) -> torch.Tensor:
        return self.observation_conditioner(obs_window)

    def decode(self, coords: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if coords.dim() == 1:
            coords = coords.unsqueeze(1)
        branch_out = self.branch(latent).reshape(-1, self.output_dim, self.basis_dim)
        trunk_out = self.trunk(coords).reshape(-1, self.output_dim, self.basis_dim)
        return (branch_out * trunk_out).sum(dim=-1)

    def forward(self, coords: torch.Tensor, obs_window: torch.Tensor) -> torch.Tensor:
        latent = self.observation_conditioner(obs_window)
        return self.decode(coords, latent)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / max(in_channels * out_channels, 1)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, grid_size = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(batch_size, self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :modes] = torch.einsum("bim,iom->bom", x_ft[:, :, :modes], self.weight[:, :, :modes])
        return torch.fft.irfft(out_ft, n=grid_size, dim=-1)


class LightweightFNO1d(nn.Module):
    has_physics_loss = False

    def __init__(self, branch_input_dim: int, grid_size: int, output_dim: int = 1, width: int = 32, modes: int = 12):
        super().__init__()
        self.grid_size = grid_size
        self.output_dim = output_dim
        self.width = width
        self.branch_proj = nn.Linear(branch_input_dim, width)
        self.coord_proj = nn.Linear(1, width)
        self.spec1 = SpectralConv1d(width, width, modes)
        self.spec2 = SpectralConv1d(width, width, modes)
        self.pointwise1 = nn.Conv1d(width, width, kernel_size=1)
        self.pointwise2 = nn.Conv1d(width, width, kernel_size=1)
        self.readout = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width, output_dim, kernel_size=1),
        )

    def forward(self, branch_input: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        if grid.dim() == 1:
            grid = grid.unsqueeze(1)
        branch = self.branch_proj(branch_input).unsqueeze(-1).expand(-1, self.width, self.grid_size)
        coord = self.coord_proj(grid).transpose(1, 2)
        x = branch + coord
        x = F.gelu(self.spec1(x) + self.pointwise1(x))
        x = F.gelu(self.spec2(x) + self.pointwise2(x))
        return self.readout(x).transpose(1, 2)


class FNO_OC(nn.Module):
    has_physics_loss = False

    def __init__(self, grid_size: int, state_dim: int, lstm_hidden: int, latent_dim: int, output_dim: int = 1, width: int = 32, modes: int = 12):
        super().__init__()
        self.grid_size = grid_size
        self.width = width
        self.output_dim = output_dim
        self.observation_conditioner = ObservationConditioner(state_dim=state_dim, lstm_hidden=lstm_hidden, latent_dim=latent_dim)
        self.encoder = self.observation_conditioner
        self.grid_proj = nn.Linear(1 + latent_dim, width)
        self.spec1 = SpectralConv1d(width, width, modes)
        self.spec2 = SpectralConv1d(width, width, modes)
        self.pointwise1 = nn.Conv1d(width, width, kernel_size=1)
        self.pointwise2 = nn.Conv1d(width, width, kernel_size=1)
        self.readout = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width, output_dim, kernel_size=1),
        )
        self.generator = nn.ModuleList([self.grid_proj, self.spec1, self.spec2, self.pointwise1, self.pointwise2, self.readout])

    def encode(self, obs_window: torch.Tensor) -> torch.Tensor:
        return self.observation_conditioner(obs_window)

    def decode(self, grid: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if grid.dim() == 1:
            grid = grid.unsqueeze(1)
        expanded_latent = latent.unsqueeze(1).expand(-1, grid.shape[1], -1)
        x = torch.cat([grid, expanded_latent], dim=-1)
        x = self.grid_proj(x).transpose(1, 2)
        x = F.gelu(self.spec1(x) + self.pointwise1(x))
        x = F.gelu(self.spec2(x) + self.pointwise2(x))
        return self.readout(x).transpose(1, 2)

    def forward(self, grid: torch.Tensor, obs_window: torch.Tensor) -> torch.Tensor:
        latent = self.observation_conditioner(obs_window)
        return self.decode(grid, latent)


@dataclass
class ModelBundle:
    name: str
    model: nn.Module


SMALL_OC_CONFIG = {"hidden_dim": 64, "latent_dim": 16}
LARGE_OC_CONFIG = {"hidden_dim": 128, "latent_dim": 32}
HYPERPINN_MATCHED_HIDDEN_DIM = 64


def create_model(family: str, with_oc: bool = False, **config) -> nn.Module:
    if family == "StandardPINN":
        return StandardPINN_OC(**config) if with_oc else StandardPINN(**config)
    if family == "HyperPINN":
        return HyperPINN_OC(**config) if with_oc else HyperPINN(**config)
    if family == "HyperLRPINN":
        return HyperLRPINN_OC(**config) if with_oc else HyperLRPINN(**config)
    if family == "DeepONet":
        return DeepONet_OC(**config) if with_oc else DeepONet(**config)
    if family == "FNO":
        return FNO_OC(**config) if with_oc else LightweightFNO1d(**config)
    raise ValueError(f"Unknown model family: {family}")


def build_tapinn(obs_dim: int, coord_dim: int, output_dim: int, large: bool = False) -> StandardPINN_OC:
    config = LARGE_OC_CONFIG if large else SMALL_OC_CONFIG
    return StandardPINN_OC(
        coord_dim=coord_dim,
        output_dim=output_dim,
        state_dim=obs_dim,
        lstm_hidden=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"],
    )


def build_capacity_matched_hyperpinn(coord_dim: int, output_dim: int) -> HyperPINN:
    return HyperPINN(coord_dim=coord_dim, output_dim=output_dim, hidden_dim=HYPERPINN_MATCHED_HIDDEN_DIM)


def build_low_rank_hyperpinn(coord_dim: int, output_dim: int, hidden_dim: int = 64, rank: int = 4) -> HyperLRPINN:
    return HyperLRPINN(coord_dim=coord_dim, output_dim=output_dim, hidden_dim=hidden_dim, rank=rank)
