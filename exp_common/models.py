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


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(sequence)
        return self.proj(hidden[-1])


class CoordinateGenerator(nn.Module):
    def __init__(self, coord_dim: int, latent_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.model = MLP(coord_dim + latent_dim, output_dim, hidden_dim=hidden_dim, depth=3, activation="tanh")

    def forward(self, coords: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if coords.dim() == 1:
            coords = coords.unsqueeze(1)
        return self.model(torch.cat([coords, latent], dim=1))


class TAPINN(nn.Module):
    def __init__(self, obs_dim: int, coord_dim: int, output_dim: int, hidden_dim: int = 32, latent_dim: int = 8):
        super().__init__()
        self.encoder = SequenceEncoder(obs_dim, hidden_dim, latent_dim)
        self.generator = CoordinateGenerator(coord_dim, latent_dim, output_dim, hidden_dim)
        self.latent_dim = latent_dim
        self.coord_dim = coord_dim
        self.output_dim = output_dim

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations)

    def decode(self, coords: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.generator(coords, latent)

    def forward(self, observations: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(observations)
        preds = self.decode(coords, latent)
        return preds, latent


class StandardPINN(nn.Module):
    def __init__(self, coord_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.model = MLP(coord_dim + 1, output_dim, hidden_dim=hidden_dim, depth=3, activation="tanh")

    def forward(self, coords: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        if coords.dim() == 1:
            coords = coords.unsqueeze(1)
        if param.dim() == 1:
            param = param.unsqueeze(1)
        return self.model(torch.cat([coords, param], dim=1))


class HyperPINN(nn.Module):
    def __init__(self, coord_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.coord_dim = coord_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_params = (
            (coord_dim * hidden_dim)
            + hidden_dim
            + (hidden_dim * hidden_dim)
            + hidden_dim
            + (hidden_dim * output_dim)
            + output_dim
        )
        self.hyper = nn.Sequential(
            nn.Linear(1, hidden_dim),
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


class HyperLRPINN(nn.Module):
    def __init__(self, coord_dim: int, output_dim: int, hidden_dim: int = 64, rank: int = 4):
        super().__init__()
        self.coord_dim = coord_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.rank = rank

        # Static base weights and biases for the trunk
        self.layer1 = nn.Linear(coord_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

        self.n_ab_params = (
            rank * (hidden_dim + coord_dim) +
            rank * (hidden_dim + hidden_dim) +
            rank * (output_dim + hidden_dim)
        )

        self.hyper = nn.Sequential(
            nn.Linear(1, hidden_dim),
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
        idx += s6

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


class DeepONet(nn.Module):
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


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / max(in_channels * out_channels, 1)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, grid_size = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x_ft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, :modes] = torch.einsum("bim,iom->bom", x_ft[:, :, :modes], self.weight[:, :, :modes])
        return torch.fft.irfft(out_ft, n=grid_size, dim=-1)


class LightweightFNO1d(nn.Module):
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
        out = self.readout(x).transpose(1, 2)
        return out


@dataclass
class ModelBundle:
    name: str
    model: nn.Module


SMALL_TAPINN_CONFIG = {"hidden_dim": 32, "latent_dim": 8}
LARGE_TAPINN_CONFIG = {"hidden_dim": 72, "latent_dim": 8}
HYPERPINN_MATCHED_HIDDEN_DIM = 32


def build_tapinn(obs_dim: int, coord_dim: int, output_dim: int, large: bool = False) -> TAPINN:
    config = LARGE_TAPINN_CONFIG if large else SMALL_TAPINN_CONFIG
    hidden_dim = config["hidden_dim"]
    latent_dim = config["latent_dim"]
    return TAPINN(obs_dim=obs_dim, coord_dim=coord_dim, output_dim=output_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)


def build_capacity_matched_hyperpinn(coord_dim: int, output_dim: int) -> HyperPINN:
    return HyperPINN(coord_dim=coord_dim, output_dim=output_dim, hidden_dim=HYPERPINN_MATCHED_HIDDEN_DIM)


def build_low_rank_hyperpinn(coord_dim: int, output_dim: int, hidden_dim: int = 64, rank: int = 4) -> HyperLRPINN:
    return HyperLRPINN(coord_dim=coord_dim, output_dim=output_dim, hidden_dim=hidden_dim, rank=rank)

