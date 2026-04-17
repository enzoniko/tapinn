from __future__ import annotations

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnusedCallResult=false

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp_common.models import create_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = PROJECT_ROOT / ".sisyphus" / "evidence"
SUMMARY_PATH = EVIDENCE_DIR / "task-13-metrics-summary.txt"

os.makedirs(EVIDENCE_DIR, exist_ok=True)


@dataclass(frozen=True)
class SmokeCase:
    name: str
    family: str
    with_oc: bool
    config: dict[str, int]


CASES = [
    SmokeCase("StandardPINN", "StandardPINN", False, {"coord_dim": 1, "output_dim": 1, "hidden_dim": 8}),
    SmokeCase(
        "StandardPINN_OC",
        "StandardPINN",
        True,
        {"coord_dim": 1, "output_dim": 1, "hidden_dim": 8, "state_dim": 2, "lstm_hidden": 6, "latent_dim": 4},
    ),
    SmokeCase("HyperPINN", "HyperPINN", False, {"coord_dim": 1, "output_dim": 1, "hidden_dim": 8}),
    SmokeCase(
        "HyperPINN_OC",
        "HyperPINN",
        True,
        {"coord_dim": 1, "output_dim": 1, "hidden_dim": 8, "state_dim": 2, "lstm_hidden": 6, "latent_dim": 4},
    ),
    SmokeCase("HyperLRPINN", "HyperLRPINN", False, {"coord_dim": 1, "output_dim": 1, "hidden_dim": 8, "rank": 2}),
    SmokeCase(
        "HyperLRPINN_OC",
        "HyperLRPINN",
        True,
        {"coord_dim": 1, "output_dim": 1, "hidden_dim": 8, "rank": 2, "state_dim": 2, "lstm_hidden": 6, "latent_dim": 4},
    ),
    SmokeCase(
        "DeepONet",
        "DeepONet",
        False,
        {"branch_input_dim": 6, "coord_dim": 1, "output_dim": 1, "hidden_dim": 8, "basis_dim": 4},
    ),
    SmokeCase(
        "DeepONet_OC",
        "DeepONet",
        True,
        {"coord_dim": 1, "output_dim": 1, "hidden_dim": 8, "basis_dim": 4, "state_dim": 2, "lstm_hidden": 6, "latent_dim": 4},
    ),
    SmokeCase(
        "FNO",
        "FNO",
        False,
        {"branch_input_dim": 2, "grid_size": 64, "output_dim": 1, "width": 8, "modes": 4},
    ),
    SmokeCase(
        "FNO_OC",
        "FNO",
        True,
        {"grid_size": 64, "output_dim": 1, "width": 8, "modes": 4, "state_dim": 2, "lstm_hidden": 6, "latent_dim": 4},
    ),
]

CASE_ORDER = [case.name for case in CASES]
RESULTS: dict[str, dict[str, object]] = {}


def _write_summary() -> None:
    lines = ["name | params | forward OK | backward OK | OC detected"]
    for name in CASE_ORDER:
        if name not in RESULTS:
            continue
        row = RESULTS[name]
        lines.append(
            f"{name} | {row['params']} | {row['forward_ok']} | {row['backward_ok']} | {row['oc_detected']}"
        )
    _ = SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_inputs(case: SmokeCase) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
    if case.family == "FNO":
        grid = torch.randn(1, 64, 1)
        if case.with_oc:
            obs_window = torch.randn(1, 10, 2)
            return (grid, obs_window), (1, 64, 1)
        branch_input = torch.randn(1, 2)
        return (branch_input, grid), (1, 64, 1)

    coords = torch.randn(8, 1)
    if case.family == "DeepONet" and not case.with_oc:
        branch_input = torch.randn(8, 6)
        return (branch_input, coords), (8, 1)

    if case.with_oc:
        obs_window = torch.randn(8, 10, 2)
        return (coords, obs_window), (8, 1)

    scalar_param = torch.randn(8, 1)
    return (coords, scalar_param), (8, 1)


@pytest.mark.parametrize("case", CASES, ids=CASE_ORDER)
def test_smoke_all_model_configs(case: SmokeCase) -> None:
    _ = torch.manual_seed(0)
    model = create_model(case.family, with_oc=case.with_oc, **case.config)
    params = sum(parameter.numel() for parameter in model.parameters())

    assert params > 0
    assert hasattr(model, "observation_conditioner") is case.with_oc

    inputs, expected_shape = _make_inputs(case)
    output = cast(torch.Tensor, model(*inputs))

    assert tuple(output.shape) == expected_shape
    assert bool(output.isfinite().all().item())

    loss = output.sum()
    loss.backward()

    finite_grads: list[bool] = []
    for parameter in model.parameters():
        if parameter.grad is not None:
            finite_grads.append(bool(parameter.grad.isfinite().all().item()))

    assert finite_grads
    assert all(finite_grads)

    RESULTS[case.name] = {
        "params": params,
        "forward_ok": True,
        "backward_ok": True,
        "oc_detected": hasattr(model, "observation_conditioner"),
    }
    _write_summary()
