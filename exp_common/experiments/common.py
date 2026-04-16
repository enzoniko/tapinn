from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections.abc import Iterable
from types import MethodType
from pathlib import Path
from typing import cast
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..device import get_best_device
from ..io_utils import prepare_run_dir, write_csv, write_json
from ..metrics import mean_std, mse, numerical_ode_residual, numerical_pde_residual
from ..models import (
    DeepONet,
    LightweightFNO1d,
    StandardPINN,
    build_capacity_matched_hyperpinn,
    build_low_rank_hyperpinn,
    build_tapinn,
    count_parameters,
)
from ..plotting import configure_plotting, get_model_color, moving_average, save_figure
from ..problems import (
    generate_allen_cahn_dataset,
    generate_burgers_dataset,
    generate_duffing_dataset,
    generate_kuramoto_dataset,
    generate_kuramoto_sivashinsky_dataset,
    generate_lorenz_dataset,
)
from ..repro import set_global_seed
from ..trainers import (
    CallbackConfig,
    CoordNormalizer,
    StateNormalizer,
    ValBundle,
    predict_direct,
    predict_fno,
    predict_tapinn,
    prepare_ode_tensors,
    prepare_pde_tensors,
    refit_normalizers_on_physical_split,
    refit_normalizers_on_split,
    train_direct_model,
    train_fno_model,
    train_tapinn,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback only
    def tqdm(iterable=None, **kwargs):
        return iterable


_EXP_EPOCHS_ENV = "TAPINN_EXPERIMENT_EPOCHS"
_EXP_ALL_CONFIGS_ENV = "TAPINN_EXPERIMENT_ALL_CONFIGS"


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-root", default="./neurips_results", help="Directory for metrics and figures.")
    parser.add_argument("--device", default="auto", help="Device override: auto, cpu, cuda, or mps.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny configuration for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--epochs", type=int, default=1000, help="Optional training epoch override for compatible experiments.")
    parser.add_argument("--all-configs", action="store_true", help="Run all model configurations even in smoke-test mode where supported.")

    original_parse_args = parser.parse_args

    def _parse_args_with_env(
        self: argparse.ArgumentParser,
        args: Iterable[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        raw_args = list(sys.argv[1:] if args is None else args)
        parsed = cast(argparse.Namespace, original_parse_args(args=args, namespace=namespace))
        if "--epochs" in raw_args:
            os.environ[_EXP_EPOCHS_ENV] = str(parsed.epochs)
        if "--all-configs" in raw_args:
            os.environ[_EXP_ALL_CONFIGS_ENV] = "1" if parsed.all_configs else "0"
        return parsed

    parser.parse_args = MethodType(_parse_args_with_env, parser)
    return parser


def _split_indices(num_samples: int, seed: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_samples)
    split = max(1, int(0.8 * num_samples))
    split = min(split, num_samples - 1) if num_samples > 1 else 1
    return perm[:split], perm[split:]


def _split_indices_three_way(num_samples: int, seed: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_samples)
    if num_samples < 3:
        train = perm[:1]
        val = perm[:1]
        test = perm[1:] if num_samples > 1 else perm[:1]
        return train, val, test

    train_end = max(1, int(0.7 * num_samples))
    val_end = max(train_end + 1, int(0.85 * num_samples))
    val_end = min(val_end, num_samples - 1)
    return perm[:train_end], perm[train_end:val_end], perm[val_end:]


def _subset_tensors(indices: np.ndarray[Any, Any], *tensors: torch.Tensor) -> list[torch.Tensor]:
    idx = torch.tensor(indices, dtype=torch.long)
    return [tensor[idx] for tensor in tensors]


def _subset_optional_tensor(indices: np.ndarray[Any, Any], tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    idx = torch.tensor(indices, dtype=torch.long)
    return tensor[idx]


def _aggregate_ode_metrics(
    problem_name: str,
    times: np.ndarray[Any, Any],
    predictions: np.ndarray[Any, Any],
    truth: np.ndarray[Any, Any],
    params: torch.Tensor,
    ode_metadata: torch.Tensor | None = None,
) -> tuple[float, float]:
    predictions = np.asarray(predictions)
    truth = np.asarray(truth)
    residuals = []
    for sample_idx, param in enumerate(params):
        metadata = None
        if ode_metadata is not None:
            metadata = {"natural_frequencies": ode_metadata[sample_idx].cpu().numpy()}
        residuals.append(numerical_ode_residual(problem_name, times, predictions[sample_idx], float(param.item()), metadata=metadata))
    return mse(predictions, truth), float(np.mean(residuals))


def _aggregate_pde_metrics(
    problem_name: str,
    times: np.ndarray[Any, Any],
    space: np.ndarray[Any, Any],
    predictions: np.ndarray[Any, Any],
    truth: np.ndarray[Any, Any],
    params: torch.Tensor,
    boundary: str,
) -> tuple[float, float]:
    predictions = np.asarray(predictions)
    truth = np.asarray(truth)
    nt = len(times)
    nx = len(space)
    pred_fields = predictions[:, :, 0].reshape(predictions.shape[0], nt, nx)
    residuals = [
        numerical_pde_residual(problem_name, times, space, pred_fields[sample_idx], float(params[sample_idx].item()), boundary=boundary)
        for sample_idx in range(pred_fields.shape[0])
    ]
    return mse(predictions, truth), float(np.mean(residuals))


def _forecast_mse(predictions: np.ndarray[Any, Any], targets: torch.Tensor, observed_steps: int) -> float:
    predictions = np.asarray(predictions)
    target_np = targets.cpu().numpy()
    if observed_steps >= target_np.shape[1]:
        return mse(predictions, target_np)
    return mse(predictions[:, observed_steps:, :], target_np[:, observed_steps:, :])


def _phase_plot(problem_name: str, truth: np.ndarray[Any, Any], pred: np.ndarray[Any, Any], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    if problem_name == "lorenz":
        ax.plot(truth[:, 0], truth[:, 2], label="Ground Truth", linewidth=2.0)
        ax.plot(pred[:, 0], pred[:, 2], label="TAPINN", linewidth=2.0, linestyle="--")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
    elif problem_name == "kuramoto":
        ax.plot(np.unwrap(truth[:, 0]), np.unwrap(truth[:, 1]), label="Ground Truth", linewidth=2.0)
        ax.plot(np.unwrap(pred[:, 0]), np.unwrap(pred[:, 1]), label="TAPINN", linewidth=2.0, linestyle="--")
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
    else:
        ax.plot(truth[:, 0], truth[:, 1], label="Ground Truth", linewidth=2.0)
        ax.plot(pred[:, 0], pred[:, 1], label="TAPINN", linewidth=2.0, linestyle="--")
        ax.set_xlabel("x")
        ax.set_ylabel("v")
    ax.set_title(title)
    ax.legend(loc="best")
    save_figure(fig, path)


def _heatmap_triptych(exact: np.ndarray[Any, Any], pred: np.ndarray[Any, Any], title: str, path: Path) -> None:
    error = np.abs(exact - pred)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    panels = [("Exact", exact), ("Predicted", pred), ("Absolute Error", error)]
    for ax, (panel_title, panel) in zip(axes, panels):
        im = ax.imshow(panel, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(panel_title)
        ax.set_xlabel("Space index")
        ax.set_ylabel("Time index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    save_figure(fig, path)


def _line_plot(x_values, y_values, x_label: str, y_label: str, title: str, path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', linewidth=2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(path)
    plt.close()


def _multi_line_plot(rows: list[dict[str, object]], x_key: str, y_key: str, hue_key: str, title: str, path: Path):
    plt.figure(figsize=(8, 6))
    groups = {}
    for r in rows:
        hue = r[hue_key]
        if hue not in groups:
            groups[hue] = {"x": [], "y": []}
        groups[hue]["x"].append(r[x_key])
        groups[hue]["y"].append(r[y_key])

    for hue, data in groups.items():
        # Sort by x
        idx = np.argsort(data["x"])
        x_sorted = np.array(data["x"])[idx]
        y_sorted = np.array(data["y"])[idx]
        plt.plot(x_sorted, y_sorted, marker='o', label=str(hue).upper())

    plt.xlabel(x_key.replace('_', ' ').title())
    plt.ylabel(y_key.replace('_', ' ').title())
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(path)
    plt.close()


def _scatter_plot(points: list[dict[str, float]], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for point in points:
        ax.scatter(point["inference_ms"], point["physics_residual"], s=70)
        ax.annotate(point["name"], (point["inference_ms"], point["physics_residual"]), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Inference Time (ms / sample)")
    ax.set_ylabel("Physics Residual")
    ax.set_title(title)
    save_figure(fig, path)


def _bar_plot(labels: list[str], values: list[float], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.bar(labels, values)
    ax.set_ylabel("Generalization Gap (MSE)")
    ax.set_title(title)
    save_figure(fig, path)


def _spectrum_plot(records: list[dict[str, object]], title: str, path: Path, problem_name: str | None = None) -> None:
    """NTK spectrum plot — one averaged line per model, always with legend."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    plot_records = records if problem_name is None else [r for r in records if r.get("problem") == problem_name]

    # Aggregate: for each (model, epoch), average eigenvalues across seeds
    from collections import defaultdict
    model_epoch_eigs: dict[str, list[np.ndarray[Any, Any]]] = defaultdict(list)
    for record in plot_records:
        key = str(record["model"])
        model_epoch_eigs[key].append(np.asarray(record["eigenvalues"], dtype=np.float64))

    # Determine a stable colour cycle
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "^", "D", "v", "P"]
    for idx, (model_name, eig_list) in enumerate(sorted(model_epoch_eigs.items())):
        # Pad to same length then take median across seed/epoch records
        max_len = max(len(e) for e in eig_list)
        stacked = np.array([np.pad(e, (0, max_len - len(e)), constant_values=np.nan) for e in eig_list])
        median_eigs = np.nanmedian(stacked, axis=0)
        ranks = np.arange(1, len(median_eigs) + 1)
        col = get_model_color(model_name)
        ax.plot(ranks, median_eigs, marker="o", linewidth=1.6, color=col,
                label=model_name, markevery=max(1, len(ranks)//8))

    ax.set_xlabel("Eigenvalue Rank")
    ax.set_ylabel("NTK Eigenvalue (median)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    save_figure(fig, path)


def _condition_plot(records: list[dict[str, object]], title: str, path: Path, problem_name: str | None = None) -> None:
    """Jacobian condition number plot — mean ± std shaded band, one per model."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    plot_records = records if problem_name is None else [r for r in records if r.get("problem") == problem_name]

    from collections import defaultdict
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "^", "D", "v", "P"]

    # Group by model then epoch: collect condition numbers across seeds
    model_epoch_data: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in plot_records:
        epoch_obj = r.get("epoch", 0.0)
        cond_obj = r.get("condition_number", 0.0)
        epoch_value = float(epoch_obj) if isinstance(epoch_obj, (int, float, str)) else 0.0
        cond_value = float(cond_obj) if isinstance(cond_obj, (int, float, str)) else 0.0
        model_epoch_data[str(r.get("model", ""))][int(epoch_value)].append(cond_value)

    for idx, model_name in enumerate(sorted(model_epoch_data.keys())):
        epoch_data = model_epoch_data[model_name]
        epochs_sorted = sorted(epoch_data.keys())
        means = np.array([np.mean(epoch_data[e]) for e in epochs_sorted])
        stds  = np.array([np.std(epoch_data[e])  for e in epochs_sorted])
        
        # Smooth the trajectory for visual stability in reports
        means_smoothed = moving_average(means, window=3)
        # Adjust epochs to match smoothed length
        epochs_trimmed = epochs_sorted[len(epochs_sorted) - len(means_smoothed):]
        
        col = get_model_color(model_name)
        ax.plot(epochs_trimmed, means_smoothed, marker="o", linewidth=1.8, color=col,
                label=model_name, markevery=max(1, len(epochs_trimmed)//6))
        
        # Match standard deviation shading to smoothed mean length
        stds_trimmed = stds[len(stds) - len(means_smoothed):]
        ax.fill_between(epochs_trimmed,
                        np.maximum(means_smoothed - stds_trimmed, 1e-12),
                        means_smoothed + stds_trimmed,
                        color=col, alpha=0.15)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Jacobian Condition Number (smoothed mean ± std)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    save_figure(fig, path)


def _final_conditioning_summary_plot(records: list[dict[str, object]], path: Path) -> None:
    """Summary bar plot: Problem vs Mean Final Condition Number (log scale)."""
    problems = sorted(str(r["problem"]) for r in records if "problem" in r)
    if not problems:
        return

    models = sorted(str(r["model"]) for r in records)
    max_epoch = max(
        int(float(r["epoch"])) if isinstance(r["epoch"], (int, float, str)) else 0
        for r in records
    )

    n_models = len(models)
    n_problems = len(problems)
    bar_w = 0.7 / max(n_models, 1)
    # Add a visible gap between problem groups
    group_spacing = n_models * bar_w + 0.4
    group_centers = np.arange(n_problems) * group_spacing

    fig_w = max(8.0, group_spacing * n_problems + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, model_name in enumerate(models):
        final_conds = []
        for prob in problems:
            subset = [
                r["condition_number"] for r in records
                if r["problem"] == prob and r["model"] == model_name and r["epoch"] == max_epoch
            ]
            final_conds.append(float(np.mean(np.asarray(subset, dtype=np.float64))) if subset else 1.0)

        offsets = group_centers + (i - (n_models - 1) / 2.0) * bar_w
        col = get_model_color(model_name)
        bars = ax.bar(offsets, final_conds, bar_w * 0.9,
                      label=model_name, color=col)
        # Annotate bar tops
        for bar, val in zip(bars, final_conds):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                f"{val:.1e}", ha="center", va="bottom", fontsize=6, rotation=45,
            )

    ax.set_ylabel("Final Condition Number (mean)")
    ax.set_yscale("log")
    ax.set_title("Optimization Stability Across Case Studies")
    ax.set_xticks(group_centers)
    ax.set_xticklabels(problems, rotation=25, ha="right", fontsize=9)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    save_figure(fig, path)


def _measure_inference_ms(fn, sample_count: int, *args) -> tuple[np.ndarray[Any, Any], float]:
    start = time.perf_counter()
    output = fn(*args)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(sample_count, 1)
    return output, elapsed_ms


def _tapinn_predict_numpy(model, obs, coords, device, state_normalizer: StateNormalizer | None = None) -> np.ndarray[Any, Any]:
    return predict_tapinn(model, obs, coords, device, state_normalizer=state_normalizer).detach().cpu().numpy()


def _direct_predict_numpy(model, kind: str, obs, coords, params, device, state_normalizer: StateNormalizer | None = None) -> np.ndarray[Any, Any]:
    return predict_direct(model, kind, obs, coords, params, device, state_normalizer=state_normalizer).detach().cpu().numpy()


def _fno_predict_numpy(model, obs, points: int, device):
    return predict_fno(model, obs, points, device)
