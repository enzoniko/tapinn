from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .device import get_best_device
from .io_utils import prepare_run_dir, write_csv, write_json
from .metrics import mean_std, mse, numerical_ode_residual, numerical_pde_residual
from .models import (
    DeepONet,
    LightweightFNO1d,
    StandardPINN,
    build_capacity_matched_hyperpinn,
    build_low_rank_hyperpinn,
    build_tapinn,
    count_parameters,
)
from .plotting import configure_plotting, save_figure
from .problems import (
    generate_allen_cahn_dataset,
    generate_burgers_dataset,
    generate_duffing_dataset,
    generate_kuramoto_dataset,
    generate_kuramoto_sivashinsky_dataset,
    generate_lorenz_dataset,
)
from .repro import set_global_seed
from .trainers import (
    CallbackConfig,
    CoordNormalizer,
    StateNormalizer,
    ValBundle,
    predict_direct,
    predict_fno,
    predict_tapinn,
    prepare_ode_tensors,
    prepare_pde_tensors,
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


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-root", default="./neurips_results", help="Directory for metrics and figures.")
    parser.add_argument("--device", default="auto", help="Device override: auto, cpu, cuda, or mps.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny configuration for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    return parser


def _split_indices(num_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_samples)
    split = max(1, int(0.8 * num_samples))
    split = min(split, num_samples - 1) if num_samples > 1 else 1
    return perm[:split], perm[split:]


def _split_indices_three_way(num_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _subset_tensors(indices: np.ndarray, *tensors: torch.Tensor) -> list[torch.Tensor]:
    idx = torch.tensor(indices, dtype=torch.long)
    return [tensor[idx] for tensor in tensors]


def _subset_optional_tensor(indices: np.ndarray, tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    idx = torch.tensor(indices, dtype=torch.long)
    return tensor[idx]


def _aggregate_ode_metrics(
    problem_name: str,
    times: np.ndarray,
    predictions: np.ndarray,
    truth: np.ndarray,
    params: torch.Tensor,
    ode_metadata: torch.Tensor | None = None,
) -> tuple[float, float]:
    residuals = []
    for sample_idx, param in enumerate(params):
        metadata = None
        if ode_metadata is not None:
            metadata = {"natural_frequencies": ode_metadata[sample_idx].cpu().numpy()}
        residuals.append(numerical_ode_residual(problem_name, times, predictions[sample_idx], float(param.item()), metadata=metadata))
    return mse(predictions, truth), float(np.mean(residuals))


def _aggregate_pde_metrics(
    problem_name: str,
    times: np.ndarray,
    space: np.ndarray,
    predictions: np.ndarray,
    truth: np.ndarray,
    params: torch.Tensor,
    boundary: str,
) -> tuple[float, float]:
    nt = len(times)
    nx = len(space)
    pred_fields = predictions[:, :, 0].reshape(predictions.shape[0], nt, nx)
    residuals = [
        numerical_pde_residual(problem_name, times, space, pred_fields[sample_idx], float(params[sample_idx].item()), boundary=boundary)
        for sample_idx in range(pred_fields.shape[0])
    ]
    return mse(predictions, truth), float(np.mean(residuals))


def _forecast_mse(predictions: np.ndarray, targets: torch.Tensor, observed_steps: int) -> float:
    target_np = targets.cpu().numpy()
    if observed_steps >= target_np.shape[1]:
        return mse(predictions, target_np)
    return mse(predictions[:, observed_steps:, :], target_np[:, observed_steps:, :])


def _phase_plot(problem_name: str, truth: np.ndarray, pred: np.ndarray, title: str, path: Path) -> None:
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


def _heatmap_triptych(exact: np.ndarray, pred: np.ndarray, title: str, path: Path) -> None:
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


def _multi_line_plot(rows: list[dict], x_key: str, y_key: str, hue_key: str, title: str, path: Path):
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
    model_epoch_eigs: dict[str, list[np.ndarray]] = defaultdict(list)
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
        col = prop_cycle[idx % len(prop_cycle)]
        mk = markers[idx % len(markers)]
        ax.plot(ranks, median_eigs, marker=mk, linewidth=1.6, color=col,
                label=model_name, markevery=max(1, len(ranks)//8))

    ax.set_xlabel("Eigenvalue Rank")
    ax.set_ylabel("NTK Eigenvalue (median)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, framealpha=0.7)
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
        model_epoch_data[str(r["model"])][int(r["epoch"])].append(float(r["condition_number"]))

    for idx, model_name in enumerate(sorted(model_epoch_data.keys())):
        epoch_data = model_epoch_data[model_name]
        epochs_sorted = sorted(epoch_data.keys())
        means = np.array([np.mean(epoch_data[e]) for e in epochs_sorted])
        stds  = np.array([np.std(epoch_data[e])  for e in epochs_sorted])
        col = prop_cycle[idx % len(prop_cycle)]
        mk  = markers[idx % len(markers)]
        ax.plot(epochs_sorted, means, marker=mk, linewidth=1.8, color=col,
                label=model_name, markevery=max(1, len(epochs_sorted)//6))
        ax.fill_between(epochs_sorted,
                        np.maximum(means - stds, 1e-12),
                        means + stds,
                        color=col, alpha=0.18)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Jacobian Condition Number (mean ± std)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, framealpha=0.7)
    save_figure(fig, path)


def _final_conditioning_summary_plot(records: list[dict[str, object]], path: Path) -> None:
    """Summary bar plot: Problem vs Mean Final Condition Number (log scale)."""
    problems = sorted({r["problem"] for r in records if "problem" in r})
    if not problems:
        return

    models = sorted({r["model"] for r in records})
    max_epoch = max(r["epoch"] for r in records)

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
            final_conds.append(float(np.mean(subset)) if subset else 1.0)

        offsets = group_centers + (i - (n_models - 1) / 2.0) * bar_w
        bars = ax.bar(offsets, final_conds, bar_w * 0.9,
                      label=model_name, color=prop_cycle[i % len(prop_cycle)])
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
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    fig.tight_layout()
    save_figure(fig, path)


def _measure_inference_ms(fn, sample_count: int, *args) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    output = fn(*args)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(sample_count, 1)
    return output, elapsed_ms


def _tapinn_predict_numpy(model, obs, coords, device) -> np.ndarray:
    return predict_tapinn(model, obs, coords, device).detach().cpu().numpy()


def _direct_predict_numpy(model, kind: str, obs, coords, params, device) -> np.ndarray:
    return predict_direct(model, kind, obs, coords, params, device).detach().cpu().numpy()


def _fno_predict_numpy(model, obs, points: int, device) -> np.ndarray:
    return predict_fno(model, obs, points, device).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Experiment 1 — ODE Chaos Suite (multi-baseline, comprehensive)
# ---------------------------------------------------------------------------

# Model specifications for Exp 1: name -> (family, comparison_group)
_EXP1_MODELS = {
    "tapinn":       ("tapinn",   "physics_trained"),
    "standard_pinn":("standard", "physics_trained"),
    "hyperpinn":    ("hyper",    "physics_trained"),
    "deeponet":     ("deeponet", "physics_trained"),
    "fno":          ("fno",      "supervised_only"),
}


def _build_ode_models(
    obs_dim: int,
    obs_steps: int,
    output_dim: int,
    num_points: int,
    smoke_test: bool,
) -> dict[str, object]:
    """Build all Exp-1 models for a given ODE problem dimensionality."""
    branch_input_dim = obs_steps * obs_dim
    fno_width = 8 if smoke_test else 32
    return {
        "tapinn":        build_tapinn(obs_dim=obs_dim, coord_dim=1, output_dim=output_dim, large=False),
        "standard_pinn": StandardPINN(coord_dim=1, output_dim=output_dim, hidden_dim=64),
        "hyperpinn":     build_capacity_matched_hyperpinn(coord_dim=1, output_dim=output_dim),
        "deeponet":      DeepONet(branch_input_dim=branch_input_dim, coord_dim=1, output_dim=output_dim, hidden_dim=64, basis_dim=32),
        "fno":           LightweightFNO1d(branch_input_dim=branch_input_dim, grid_size=num_points, output_dim=output_dim, width=fno_width, modes=4 if smoke_test else 12),
    }


def _train_ode_model(
    model_name: str,
    model,
    problem_name: str,
    train_obs: torch.Tensor,
    train_coords: torch.Tensor,
    train_targets: torch.Tensor,
    train_params: torch.Tensor,
    train_ode_metadata: torch.Tensor | None,
    val_bundle: ValBundle | None,
    callbacks: CallbackConfig | None,
    device: torch.device,
    max_epochs: int,
    batch_size: int,
    seed: int,
    coord_normalizer: CoordNormalizer | None = None,
    state_normalizer: StateNormalizer | None = None,
):
    """Train a single model, applying identical callbacks for fairness."""
    desc = f"{model_name[:8]}-{problem_name[:6]}-s{seed}"
    family = _EXP1_MODELS[model_name][0]
    if family == "tapinn":
        return train_tapinn(
            model,
            problem_name=problem_name,
            observations=train_obs,
            coords=train_coords,
            targets=train_targets,
            params=train_params,
            device=device,
            ode_metadata=train_ode_metadata,
            epochs=max_epochs,
            batch_size=batch_size,
            progress_desc=desc,
            val_bundle=val_bundle,
            callbacks=callbacks,
            coord_normalizer=coord_normalizer,
            state_normalizer=state_normalizer,
        )
    elif family == "fno":
        fno_val: ValBundle | None = None
        if val_bundle is not None:
            fno_val = ValBundle(
                observations=val_bundle.observations,
                coords=val_bundle.coords,
                targets=val_bundle.targets,
                params=val_bundle.params,
            )
        return train_fno_model(
            model,
            observations=train_obs,
            targets=train_targets,
            device=device,
            epochs=max_epochs,
            batch_size=batch_size,
            progress_desc=desc,
            val_bundle=fno_val,
            callbacks=callbacks,
        )
    else:
        return train_direct_model(
            model,
            problem_name=problem_name,
            model_kind=family,
            observations=train_obs,
            coords=train_coords,
            targets=train_targets,
            params=train_params,
            device=device,
            ode_metadata=train_ode_metadata,
            epochs=max_epochs,
            batch_size=batch_size,
            progress_desc=desc,
            val_bundle=val_bundle,
            callbacks=callbacks,
            coord_normalizer=coord_normalizer,
            state_normalizer=state_normalizer,
        )


def _predict_ode_model(
    model_name: str,
    model,
    test_obs: torch.Tensor,
    test_coords: torch.Tensor,
    test_params: torch.Tensor,
    num_points: int,
    device: torch.device,
    state_normalizer: StateNormalizer | None = None,
) -> tuple[np.ndarray, float]:
    """Run inference for a model and return (predictions_numpy, inference_ms)."""
    family = _EXP1_MODELS[model_name][0]
    if family == "tapinn":
        return _measure_inference_ms(
            lambda *a: predict_tapinn(*a, state_normalizer=state_normalizer).detach().cpu().numpy(),
            test_obs.shape[0],
            model, test_obs, test_coords, device,
        )
    elif family == "fno":
        return _measure_inference_ms(
            _fno_predict_numpy, test_obs.shape[0],
            model, test_obs, num_points, device,
        )
    else:
        return _measure_inference_ms(
            lambda *a: predict_direct(*a, state_normalizer=state_normalizer).detach().cpu().numpy(),
            test_obs.shape[0],
            model, family, test_obs, test_coords, test_params, device,
        )


def _run_ode_seed_all_models(
    problem_name: str,
    param_values: list[float],
    seed: int,
    smoke_test: bool,
    device: torch.device,
    max_epochs: int,
    callbacks: CallbackConfig,
) -> dict[str, dict]:
    """Train and evaluate ALL Exp-1 models on a single ODE problem/seed.

    Returns a mapping  model_name -> metrics_dict  containing data_mse,
    physics_residual, param_count, epochs_trained, seconds_per_epoch,
    inference_ms, comparison_group, and the first test trajectory for
    phase-space plots.

    Scientific design notes
    -----------------------
    * Three-way train/val/test split (70/15/15) with the same indices for ALL
      models, ensuring no model has seen the test set.
    * HyperPINN is trained on data with duplicated parameter values (multiple
      trajectories per parameter).  It cannot distinguish trajectories with
      identical parameters — this is the intended comparison: observation-
      conditioning (TAPINN, DeepONet) vs parameter-only conditioning
      (StandardPINN, HyperPINN).
    * The same CallbackConfig (EarlyStopping + ReduceLROnPlateau) is applied
      to every model.  Only the validation data differs (wrapped in ValBundle).
    """
    # ------------------------------------------------------------------
    # 1. Generate data  (larger datasets vs original exp_1)
    # ------------------------------------------------------------------
    num_points = 48 if smoke_test else 200
    if problem_name == "duffing":
        data = generate_duffing_dataset(
            param_values,
            num_trajectories=1 if smoke_test else 20,
            num_points=num_points,
            t_span=(0.0, 16.0),
            seed=seed,
        )
    elif problem_name == "lorenz":
        data = generate_lorenz_dataset(
            param_values,
            num_trajectories=1 if smoke_test else 12,
            num_points=num_points,
            t_span=(0.0, 4.5),
            seed=seed,
        )
    elif problem_name == "kuramoto":
        data = generate_kuramoto_dataset(
            param_values,
            num_trajectories=1 if smoke_test else 12,
            num_points=num_points,
            t_span=(0.0, 10.0),
            num_oscillators=5,
            seed=seed,
        )
    else:
        raise ValueError(problem_name)

    obs_steps = 8 if smoke_test else max(24, int(0.15 * data.states.shape[1]))
    observations, coords, targets, params, ode_metadata, coord_norm, state_norm = prepare_ode_tensors(data, observation_steps=obs_steps)

    # ------------------------------------------------------------------
    # 2. Three-way split (70 / 15 / 15)
    # ------------------------------------------------------------------
    train_idx, val_idx, test_idx = _split_indices_three_way(len(params), seed)
    train_obs, train_coords, train_targets, train_params = _subset_tensors(train_idx, observations, coords, targets, params)
    val_obs,   val_coords,   val_targets,   val_params   = _subset_tensors(val_idx,   observations, coords, targets, params)
    test_obs,  test_coords,  test_targets,  test_params  = _subset_tensors(test_idx,  observations, coords, targets, params)
    train_ode_meta = _subset_optional_tensor(train_idx, ode_metadata)
    val_ode_meta   = _subset_optional_tensor(val_idx,   ode_metadata)
    test_ode_meta  = _subset_optional_tensor(test_idx,  ode_metadata)

    # Refit normalizers on training split only (no data leakage)
    coord_norm, state_norm = refit_normalizers_on_split(train_obs, train_coords, train_targets)
    # Re-apply normalisation on all splits using training-fit normalizers
    train_targets = state_norm.normalize(state_norm.denormalize(train_targets))  # no-op after refit
    val_targets   = state_norm.normalize(state_norm.denormalize(val_targets))
    test_targets  = state_norm.normalize(state_norm.denormalize(test_targets))

    val_bundle = ValBundle(
        observations=val_obs,
        coords=val_coords,
        targets=val_targets,
        params=val_params,
        ode_metadata=val_ode_meta,
    )
    active_val_bundle: ValBundle | None = val_bundle if val_obs.shape[0] > 0 else None
    active_callbacks: CallbackConfig | None = callbacks if val_obs.shape[0] > 0 else None

    # ------------------------------------------------------------------
    # 3. Build models
    # ------------------------------------------------------------------
    obs_dim    = observations.shape[-1]
    output_dim = targets.shape[-1]
    models = _build_ode_models(obs_dim, obs_steps, output_dim, num_points, smoke_test)
    batch_size = 4

    # ------------------------------------------------------------------
    # 4. Train + evaluate each model
    # ------------------------------------------------------------------
    results: dict[str, dict] = {}
    # Denormalise test targets for ground-truth metric evaluation
    truth_norm = test_targets.cpu().numpy()  # normalised, for loss checks
    truth = state_norm.denormalize(test_targets).cpu().numpy()  # physical space

    for model_name, model in tqdm(models.items(), desc=f"{problem_name[:6]}/models", leave=False):
        train_result = _train_ode_model(
            model_name, model, problem_name,
            train_obs, train_coords, train_targets, train_params, train_ode_meta,
            active_val_bundle, active_callbacks, device, max_epochs, batch_size, seed,
            coord_normalizer=coord_norm, state_normalizer=state_norm,
        )
        predictions, inference_ms = _predict_ode_model(
            model_name, model, test_obs, test_coords, test_params, num_points, device,
            state_normalizer=state_norm,
        )
        # predictions are now in physical (denormalised) space
        data_mse, physics_residual = _aggregate_ode_metrics(
            problem_name, data.times, predictions, truth, test_params, test_ode_meta,
        )
        _, comparison_group = _EXP1_MODELS[model_name]
        results[model_name] = {
            "data_mse":           data_mse,
            "physics_residual":   physics_residual,
            "param_count":        count_parameters(model),
            "epochs_trained":     train_result.epochs_trained,
            "best_val_loss":      train_result.best_val_loss,
            "seconds_per_epoch":  train_result.seconds_per_epoch,
            "inference_ms":       inference_ms,
            "comparison_group":   comparison_group,
            # first test sample for phase-space plots
            "prediction":         predictions[0],
            "truth":              truth[0],
            "param":              float(test_params[0].item()),
            "times":              data.times,
        }

    return results


# ---------------------------------------------------------------------------
# Exp-1 figure helpers
# ---------------------------------------------------------------------------

def _multi_model_phase_plot(
    problem_name: str,
    model_results: dict[str, dict],
    title: str,
    path: Path,
) -> None:
    """One subplot per model showing ground-truth vs prediction in phase space."""
    model_names = list(model_results.keys())
    ncols = min(3, len(model_names))
    nrows = math.ceil(len(model_names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for ax, model_name in zip(axes_flat, model_names):
        info = model_results[model_name]
        truth = info["truth"]
        pred  = info["prediction"]
        if problem_name == "lorenz":
            ax.plot(truth[:, 0], truth[:, 2], label="GT",   linewidth=1.6)
            ax.plot(pred[:, 0],  pred[:, 2],  label="Pred", linewidth=1.6, linestyle="--")
            ax.set_xlabel("x"); ax.set_ylabel("z")
        elif problem_name == "kuramoto":
            ax.plot(np.unwrap(truth[:, 0]), np.unwrap(truth[:, 1]), label="GT",   linewidth=1.6)
            ax.plot(np.unwrap(pred[:, 0]),  np.unwrap(pred[:, 1]),  label="Pred", linewidth=1.6, linestyle="--")
            ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")
        else:
            ax.plot(truth[:, 0], truth[:, 1], label="GT",   linewidth=1.6)
            ax.plot(pred[:, 0],  pred[:, 1],  label="Pred", linewidth=1.6, linestyle="--")
            ax.set_xlabel("x"); ax.set_ylabel("v")
        mse_val = info["data_mse"]
        ax.set_title(f"{model_name}\nMSE={mse_val:.3g}")
        ax.legend(loc="best", fontsize=7)

    for ax in axes_flat[len(model_names):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    save_figure(fig, path)


def _metrics_bar_chart(
    model_names: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    path: Path,
    comparison_groups: list[str] | None = None,
) -> None:
    """Bar chart of a metric across models, coloured by comparison group."""
    colors = []
    if comparison_groups:
        palette = {"physics_trained": "#4C72B0", "supervised_only": "#DD8452"}
        colors = [palette.get(g, "grey") for g in comparison_groups]
    else:
        colors = ["#4C72B0"] * len(model_names)
    fig, ax = plt.subplots(figsize=(max(5.0, 1.5 * len(model_names)), 3.8))
    bars = ax.bar(model_names, values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    if comparison_groups:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#4C72B0", label="physics_trained"),
            Patch(facecolor="#DD8452", label="supervised_only"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    save_figure(fig, path)


def _param_vs_residual_scatter(
    model_results_flat: list[dict],
    title: str,
    path: Path,
) -> None:
    """Scatter: param count (x) vs physics residual (y), one point per model."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    palette = {"physics_trained": "#4C72B0", "supervised_only": "#DD8452"}
    for row in model_results_flat:
        color = palette.get(row.get("comparison_group", ""), "grey")
        ax.scatter(row["param_count"], row["physics_residual"], s=80, color=color)
        ax.annotate(row["model_name"], (row["param_count"], row["physics_residual"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Parameter Count")
    ax.set_ylabel("Physics Residual")
    ax.set_title(title)
    save_figure(fig, path)


# ---------------------------------------------------------------------------
# Experiment 1 entry point
# ---------------------------------------------------------------------------

def run_exp_1_ode_chaos_suite(output_root: str, device_name: str, smoke_test: bool, seed: int) -> dict[str, object]:
    """ODE Chaos Suite — comprehensive multi-baseline benchmark.

    Scientific role
    ---------------
    This experiment serves as the foundational ODE benchmark for the paper:

    1. **Grounding in the original domain.**  The Duffing oscillator was the
       workshop paper's central case study.  Providing systematic results here
       anchors the method's lineage.
    2. **Challenge gradient for baselines.**  Three systems of increasing
       difficulty (Duffing → Kuramoto → Lorenz) expose how each method's
       failure mode differs under chaos.  A diagnostic failure is scientifically
       useful: Lorenz showing universal difficulty supports Exp 5's
       optimization analysis rather than contradicting the method's value.
    3. **ODE-specific comparative performance.**  Exp 3 covers Duffing + a
       PDE; this experiment is the only place that tests all baselines on
       Lorenz and Kuramoto.
    4. **Initial-condition sensitivity.**  Multiple trajectories per parameter
       value test whether observation-conditioning (TAPINN, DeepONet) can
       distinguish trajectories that share the same parameter but diverge from
       different initial conditions.  HyperPINN and StandardPINN cannot.

    Baselines
    ---------
    * TAPINN            — proposed method, physics_trained
    * StandardPINN      — param-conditioned MLP, physics_trained
    * HyperPINN         — hypernetwork, physics_trained  (param-only, ~40K params)
    * DeepONet          — operator network, physics_trained
    * FNO               — Fourier operator, supervised_only

    Callbacks (identical for ALL models)
    -------------------------------------
    * EarlyStopping: patience=15
    * ReduceLROnPlateau: patience=8, factor=0.5, min_lr=1e-6
    """
    configure_plotting()
    device = get_best_device(device_name)
    run_dir = prepare_run_dir(output_root, "exp_1_ode_chaos_suite")
    seeds = [seed] if smoke_test else [seed + offset for offset in range(5)]
    configs = {
        "duffing":  [0.24, 0.38, 0.52] if smoke_test else np.linspace(0.2, 0.55, 7).tolist(),
        "lorenz":   [20.0, 24.74, 32.0] if smoke_test else np.linspace(18.0, 35.0, 7).tolist(),
        "kuramoto": [0.2,  1.0,   2.0]  if smoke_test else np.linspace(0.2, 3.0,  7).tolist(),
    }
    max_epochs = 4 if smoke_test else 150
    callbacks = CallbackConfig(
        early_stopping_patience=2 if smoke_test else 15,
        reduce_lr_patience=1 if smoke_test else 8,
        reduce_lr_factor=0.5,
        min_lr=1e-6,
    )
    model_names = list(_EXP1_MODELS.keys())

    # Rows for per-seed-model-problem CSV
    seed_rows: list[dict] = []
    # Aggregated per problem+model
    problem_model_summaries: list[dict] = []
    # Representative phase-space data (first seed only)
    phase_data: dict[str, dict[str, dict]] = {}  # problem -> model -> info

    for problem_name, param_values in tqdm(configs.items(), desc="ODE systems", leave=False):
        per_model_seed_data: dict[str, list[dict]] = {m: [] for m in model_names}

        for local_seed in tqdm(seeds, desc=f"{problem_name} seeds", leave=False):
            set_global_seed(local_seed)
            results = _run_ode_seed_all_models(
                problem_name, param_values, local_seed, smoke_test, device,
                max_epochs, callbacks,
            )
            # Store phase-space data from first seed
            if local_seed == seeds[0]:
                phase_data[problem_name] = results

            for model_name, info in results.items():
                per_model_seed_data[model_name].append(info)
                seed_rows.append({
                    "problem":          problem_name,
                    "seed":             local_seed,
                    "model":            model_name,
                    "comparison_group": info["comparison_group"],
                    "data_mse":         info["data_mse"],
                    "physics_residual": info["physics_residual"],
                    "param_count":      info["param_count"],
                    "epochs_trained":   info["epochs_trained"],
                    "best_val_loss":    info["best_val_loss"],
                    "seconds_per_epoch":info["seconds_per_epoch"],
                    "inference_ms":     info["inference_ms"],
                })

        # Aggregate across seeds per model
        for model_name in model_names:
            rows = per_model_seed_data[model_name]
            if not rows:
                continue
            data_mean, data_std   = mean_std(r["data_mse"]         for r in rows)
            phys_mean, phys_std   = mean_std(r["physics_residual"]  for r in rows)
            ep_mean,   ep_std     = mean_std(r["epochs_trained"]     for r in rows)
            spe_mean,  spe_std    = mean_std(r["seconds_per_epoch"]  for r in rows)
            inf_mean,  inf_std    = mean_std(r["inference_ms"]       for r in rows)
            problem_model_summaries.append({
                "problem":                   problem_name,
                "model":                     model_name,
                "comparison_group":          rows[0]["comparison_group"],
                "param_count":               rows[0]["param_count"],
                "data_mse_mean":             data_mean,
                "data_mse_std":              data_std,
                "physics_residual_mean":     phys_mean,
                "physics_residual_std":      phys_std,
                "epochs_trained_mean":       ep_mean,
                "epochs_trained_std":        ep_std,
                "seconds_per_epoch_mean":    spe_mean,
                "inference_ms_mean":         inf_mean,
            })

        # Phase-space comparison figure (all models, first seed)
        _multi_model_phase_plot(
            problem_name,
            phase_data[problem_name],
            f"{problem_name.title()} — All Models Phase-Space (seed {seeds[0]})",
            run_dir / "figures" / f"{problem_name}_model_comparison.pdf",
        )
        # Keep legacy single-model phase plot for backward compat with tests
        tapinn_info = phase_data[problem_name].get("tapinn", {})
        if tapinn_info:
            _phase_plot(
                problem_name,
                tapinn_info["truth"],
                tapinn_info["prediction"],
                f"{problem_name.title()} Phase-Space — TAPINN",
                run_dir / "figures" / f"{problem_name}_phase_space.pdf",
            )

        # Per-problem metric bar charts
        prob_rows = [r for r in problem_model_summaries if r["problem"] == problem_name]
        _metrics_bar_chart(
            [r["model"] for r in prob_rows],
            [r["data_mse_mean"] for r in prob_rows],
            "Data MSE",
            f"{problem_name.title()} — Data MSE by Model",
            run_dir / "figures" / f"{problem_name}_data_mse_bar.pdf",
            comparison_groups=[r["comparison_group"] for r in prob_rows],
        )
        _metrics_bar_chart(
            [r["model"] for r in prob_rows],
            [r["physics_residual_mean"] for r in prob_rows],
            "Physics Residual",
            f"{problem_name.title()} — Physics Residual by Model",
            run_dir / "figures" / f"{problem_name}_physics_residual_bar.pdf",
            comparison_groups=[r["comparison_group"] for r in prob_rows],
        )

    # Global param-count vs residual scatter (Duffing, as representative)
    duffing_rows = [r for r in problem_model_summaries if r["problem"] == "duffing"]
    if duffing_rows:
        scatter_input = [
            {
                "model_name":       r["model"],
                "param_count":      r["param_count"],
                "physics_residual": r["physics_residual_mean"],
                "comparison_group": r["comparison_group"],
            }
            for r in duffing_rows
        ]
        _param_vs_residual_scatter(
            scatter_input,
            "Duffing — Parameter Count vs Physics Residual",
            run_dir / "figures" / "duffing_pareto_params_vs_residual.pdf",
        )

    # Legacy summary_table.csv (TAPINN-only for backward compatibility)
    tapinn_summaries = [
        {
            "problem":              r["problem"],
            "data_mse_mean":        r["data_mse_mean"],
            "data_mse_std":         r["data_mse_std"],
            "physics_residual_mean":r["physics_residual_mean"],
            "physics_residual_std": r["physics_residual_std"],
        }
        for r in problem_model_summaries
        if r["model"] == "tapinn"
    ]

    write_csv(run_dir / "tables" / "seed_metrics.csv", seed_rows)
    write_csv(run_dir / "tables" / "model_summary.csv", problem_model_summaries)
    write_csv(run_dir / "tables" / "summary_table.csv", tapinn_summaries)  # legacy
    payload = {
        "device":        str(device),
        "smoke_test":    smoke_test,
        "max_epochs":    max_epochs,
        "callbacks":     {
            "early_stopping_patience": callbacks.early_stopping_patience,
            "reduce_lr_patience":      callbacks.reduce_lr_patience,
            "reduce_lr_factor":        callbacks.reduce_lr_factor,
            "min_lr":                  callbacks.min_lr,
        },
        "models":        model_names,
        "comparison_groups": {m: _EXP1_MODELS[m][1] for m in model_names},
        "summary":       problem_model_summaries,
    }
    write_json(run_dir / "results.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Experiment 4 — Sensitivity & Robustness (multi-baseline comparative sweep)
# ---------------------------------------------------------------------------

# Model specifications for Exp 4: focuses on the observation-conditioned winners
_EXP4_MODELS = {
    "tapinn": ("tapinn", "physics_trained"),
    "fno":    ("fno",    "supervised_only"),
}


def _eval_robustness_grid_point(
    model_name: str,
    problem_name: str,
    train_tuple: tuple,  # (obs, coords, targets, params, meta)
    val_tuple: tuple,
    test_tuple: tuple,
    noise_sigma: float,
    device: torch.device,
    max_epochs: int,
    callbacks: CallbackConfig,
    seed: int,
) -> dict:
    """Evaluate a single model at one grid point (noise/window)."""
    (t_obs, t_coords, t_targets, t_params, t_meta, t_sn, t_cn) = train_tuple
    (v_obs, v_coords, v_targets, v_params, v_meta, v_sn, v_cn) = val_tuple
    (s_obs, s_coords, s_targets, s_params, s_meta, s_sn, s_cn) = test_tuple


    # 1. Apply observational noise (relative to training std)
    obs_std = float(t_obs.std().item()) if t_obs.numel() else 1.0
    noise_abs = noise_sigma * obs_std
    
    noisy_t_obs = t_obs + noise_abs * torch.randn_like(t_obs)
    noisy_v_obs = v_obs + noise_abs * torch.randn_like(v_obs)
    noisy_s_obs = s_obs + noise_abs * torch.randn_like(s_obs)

    # 2. Build model
    obs_dim = t_obs.shape[-1]
    coord_dim = t_coords.shape[-1]
    output_dim = t_targets.shape[-1]
    branch_dim = t_obs.shape[1] * t_obs.shape[2]
    grid_size = t_targets.shape[1]

    if model_name == "tapinn":
        model = build_tapinn(obs_dim=obs_dim, coord_dim=coord_dim, output_dim=output_dim, large=False)
    else:
        model = LightweightFNO1d(branch_input_dim=branch_dim, grid_size=grid_size, output_dim=output_dim, width=32, modes=12)

    val_bundle = ValBundle(noisy_v_obs, v_coords, v_targets, v_params, v_meta)

    # 3. Train
    family = _EXP4_MODELS[model_name][0]
    # coords are already normalised by prepare_*_tensors; state_norm provided by caller
    state_norm_arg = t_sn
    coord_norm_arg = t_cn

    if family == "tapinn":
        train_result = train_tapinn(
            model, problem_name, noisy_t_obs, t_coords, t_targets, t_params, device,
            ode_metadata=t_meta, epochs=max_epochs, batch_size=4, progress_desc=f"{model_name}-noise{noise_sigma}",
            val_bundle=val_bundle, callbacks=callbacks,
            coord_normalizer=coord_norm_arg, state_normalizer=state_norm_arg,
        )
        preds, inference_ms = _measure_inference_ms(
            lambda *a: predict_tapinn(*a, state_normalizer=state_norm_arg).detach().cpu().numpy(),
            noisy_s_obs.shape[0], model, noisy_s_obs, s_coords, device,
        )

    else:
        train_result = train_fno_model(
            model, noisy_t_obs, t_targets, device, epochs=max_epochs, batch_size=4,
            progress_desc=f"{model_name}-noise{noise_sigma}", val_bundle=val_bundle, callbacks=callbacks,
        )
        preds, inference_ms = _measure_inference_ms(_fno_predict_numpy, noisy_s_obs.shape[0], model, noisy_s_obs, grid_size, device)

    # 4. Aggregate metrics (forecast-style: focus on unobserved tail)
    obs_steps = t_obs.shape[1]
    test_truth = s_targets.cpu().numpy()
    if problem_name == "allen_cahn":
        # AC is fixed size in this suite: nt=36, nx=48
        data_mse, physics = _aggregate_pde_metrics(problem_name, np.linspace(0, 1, 36), np.linspace(-1, 1, 48), preds, test_truth, s_params, "periodic")
    else:
        # ODE: Duffing (T=16), Kuramoto (T=10). We infer or assume based on problem_name.
        t_final = 16.0 if problem_name == "duffing" else 10.0
        data_mse, physics = _aggregate_ode_metrics(problem_name, np.linspace(0, t_final, grid_size), preds, test_truth, s_params, s_meta)


    forecast_error = float(mse(preds[:, obs_steps:, :], test_truth[:, obs_steps:, :]))

    return {
        "model":             model_name,
        "noise_sigma":       noise_sigma,
        "data_mse":          data_mse,
        "physics_residual":  physics,
        "forecast_error":    forecast_error,
        "epochs_trained":    train_result.epochs_trained,
        "best_val_loss":     train_result.best_val_loss,
    }


def run_exp_4_sensitivity_and_robustness(output_root: str, device_name: str, smoke_test: bool, seed: int) -> dict[str, object]:
    """Experiment 4 — Sensitivity & Robustness Suite.

    Scientific Role
    ---------------
    * Demonstrates TAPINN's robustness to sparse and noisy observations compared
      to supervised baselines (FNO).
    * Tests the "forecast boundary" — how small an observation window is required
      to maintain valid physics and low error.
    * Evaluates robustness across both ODE (Duffing) and PDE (Allen-Cahn) regimes.
    """
    configure_plotting()
    device = get_best_device(device_name)
    run_dir = prepare_run_dir(output_root, "exp_4_sensitivity_and_robustness")

    # Shared training config for all sweep points
    max_epochs = 4 if smoke_test else 100
    callbacks = CallbackConfig(
        early_stopping_patience=2 if smoke_test else 15,
        reduce_lr_patience=1 if smoke_test else 8,
        reduce_lr_factor=0.5,
        min_lr=1e-6,
    )
    seeds = [seed] if smoke_test else [seed + offset for offset in range(5)]
    model_names = list(_EXP4_MODELS.keys())

    # Sweeps configuration
    noise_levels = [0.0, 0.1, 0.25] if smoke_test else [0.0, 0.05, 0.1, 0.25, 0.5]
    window_fractions = [0.1, 0.2] if smoke_test else [0.05, 0.1, 0.2]

    noise_rows = []
    window_rows = []

    # 1. Task Generation helpers
    def get_data_duffing(s):
        p = [0.24, 0.38, 0.52] if smoke_test else np.linspace(0.2, 0.55, 7).tolist()
        d = generate_duffing_dataset(p, num_trajectories=1 if smoke_test else 10, num_points=160, t_span=(0.0, 16.0), seed=s)
        return prepare_ode_tensors(d, observation_steps=16)  # returns 7-tuple

    def get_data_allen(s):
        p = [0.8, 1.1] if smoke_test else [0.75, 0.9, 1.05, 1.2]
        d = generate_allen_cahn_dataset(p, num_samples=1 if smoke_test else 8, nx=48, nt=36, seed=s + 11)
        return prepare_pde_tensors(d, observation_steps=4) # Fixed ~10% window for noise sweep

    # 2. RUN NOISE SWEEP
    for problem_name, data_fn in [("duffing", get_data_duffing), ("allen_cahn", get_data_allen)]:
        for noise in tqdm(noise_levels, desc=f"Noise Sweep: {problem_name}", leave=False):
            for model_name in model_names:
                seed_metrics = []
                for local_seed in tqdm(seeds, desc=f"Seeds (noise={noise})", leave=False):
                    set_global_seed(local_seed)
                    tensors = data_fn(local_seed)
                    obs, crds, trgs, prms = tensors[:4]
                    meta_raw = tensors[4] if len(tensors) == 7 else None

                    train_idx, val_idx, test_idx = _split_indices_three_way(len(prms), local_seed)

                    # 2.2 Re-fit normalizers on training split for rigor
                    t_obs, t_crds, t_trgs, t_prms = _subset_tensors(train_idx, obs, crds, trgs, prms)
                    c_norm, s_norm = refit_normalizers_on_split(t_obs, t_crds, t_trgs)

                    t_tup = (t_obs, t_crds, t_trgs, t_prms, _subset_optional_tensor(train_idx, meta_raw), s_norm, c_norm)
                    v_tup = tuple(_subset_tensors(val_idx, obs, crds, trgs, prms)) + (_subset_optional_tensor(val_idx, meta_raw), s_norm, c_norm)
                    s_tup = tuple(_subset_tensors(test_idx, obs, crds, trgs, prms)) + (_subset_optional_tensor(test_idx, meta_raw), s_norm, c_norm)

                    res = _eval_robustness_grid_point(model_name, problem_name, t_tup, v_tup, s_tup, noise, device, max_epochs, callbacks, local_seed)

                    seed_metrics.append(res)
                
                # Aggregate across seeds
                d_mean, d_std = mean_std(r["data_mse"] for r in seed_metrics)
                p_mean, p_std = mean_std(r["physics_residual"] for r in seed_metrics)
                f_mean, f_std = mean_std(r["forecast_error"] for r in seed_metrics)
                
                noise_rows.append({
                    "problem": problem_name,
                    "model": model_name,
                    "noise_sigma": noise,
                    "data_mse_mean": d_mean,
                    "data_mse_std": d_std,
                    "physics_residual_mean": p_mean,
                    "physics_residual_std": p_std,
                    "forecast_error_mean": f_mean,
                    "forecast_error_std": f_std,
                })

    # 3. RUN WINDOW SWEEP (Fixed Noise = 0.0)
    for problem_name, data_fn_base in [("duffing", get_data_duffing), ("allen_cahn", get_data_allen)]:
        for frac in tqdm(window_fractions, desc=f"Window Sweep: {problem_name}", leave=False):
            # Recalculate obs_steps for this fraction
            total_steps = 160 if problem_name == "duffing" else 36
            obs_steps = max(2, int(frac * total_steps))

            for model_name in model_names:
                seed_metrics = []
                for local_seed in tqdm(seeds, desc=f"Seeds (window={frac})", leave=False):
                    set_global_seed(local_seed)
                    # Need to regenerate data to apply different obs_steps
                    if problem_name == "duffing":
                        p = [0.24, 0.38, 0.52] if smoke_test else np.linspace(0.2, 0.55, 7).tolist()
                        d = generate_duffing_dataset(p, num_trajectories=1 if smoke_test else 10, num_points=160, t_span=(0.0, 16.0), seed=local_seed)
                        obs, crds, trgs, prms, meta, _cn, _sn = prepare_ode_tensors(d, observation_steps=obs_steps)
                    else:
                        p = [0.8, 1.1] if smoke_test else [0.75, 0.9, 1.05, 1.2]
                        d = generate_allen_cahn_dataset(p, num_samples=1 if smoke_test else 8, nx=48, nt=36, seed=local_seed + 11)
                        obs, crds, trgs, prms, _cn, _sn = prepare_pde_tensors(d, observation_steps=obs_steps)
                        meta = None
                    train_idx, val_idx, test_idx = _split_indices_three_way(len(prms), local_seed)
                    
                    # 3.2 Re-fit normalizers on training split
                    t_obs, t_crds, t_trgs, t_prms = _subset_tensors(train_idx, obs, crds, trgs, prms)
                    c_norm, s_norm = refit_normalizers_on_split(t_obs, t_crds, t_trgs)

                    t_tup = (t_obs, t_crds, t_trgs, t_prms, _subset_optional_tensor(train_idx, meta), s_norm, c_norm)
                    v_tup = tuple(_subset_tensors(val_idx, obs, crds, trgs, prms)) + (_subset_optional_tensor(val_idx, meta), s_norm, c_norm)
                    s_tup = tuple(_subset_tensors(test_idx, obs, crds, trgs, prms)) + (_subset_optional_tensor(test_idx, meta), s_norm, c_norm)

                    res = _eval_robustness_grid_point(model_name, problem_name, t_tup, v_tup, s_tup, 0.0, device, max_epochs, callbacks, local_seed)

                    res["window_fraction"] = frac
                    seed_metrics.append(res)
                
                f_mean, f_std = mean_std(r["forecast_error"] for r in seed_metrics)
                window_rows.append({
                    "problem": problem_name,
                    "model": model_name,
                    "window_fraction": frac,
                    "observed_steps": obs_steps,
                    "forecast_error_mean": f_mean,
                    "forecast_error_std": f_std,
                })

    # Save results
    run_dir = Path(run_dir) # Ensure Path object
    write_csv(run_dir / "tables" / "noise_sweep.csv", noise_rows)
    write_csv(run_dir / "tables" / "window_sweep.csv", window_rows)

    # Plot Noise Robustness: TAPINN vs FNO
    for prob in ["duffing", "allen_cahn"]:
        subset = [r for r in noise_rows if r["problem"] == prob]
        _multi_line_plot(
            subset, "noise_sigma", "forecast_error_mean", "model",
            f"{prob.title()} Robustness: Forecast Error vs Obs Noise",
            run_dir / "figures" / f"{prob}_noise_robustness.pdf"
        )
        
        subset_w = [r for r in window_rows if r["problem"] == prob]
        _multi_line_plot(
            subset_w, "window_fraction", "forecast_error_mean", "model",
            f"{prob.title()} Sensitivity: Forecast Error vs Window Size",
            run_dir / "figures" / f"{prob}_window_sensitivity.pdf"
        )

    payload = {
        "device":      str(device),
        "smoke_test":  smoke_test,
        "max_epochs":  max_epochs,
        "noise_sweep": noise_rows,
        "window_sweep": window_rows,
    }
    write_json(run_dir / "results.json", payload)
    return payload


def _finite_ntk_and_condition(model: nn.Module, model_kind: str, observations: torch.Tensor, coords: torch.Tensor, params: torch.Tensor, device: torch.device):
    """Compute the NTK spectrum and Jacobian condition number for a small batch.
    
    Robustness Fix: Ensures all model parameters and input tensors are pinned to 
    the same compute device to avoid 'RuntimeError: Input and parameter tensors 
    are not at the same device'.
    """
    diag_samples = min(4, observations.shape[0])
    points_per_sample = max(1, min(6, coords.shape[1]))
    jac_rows = []
    
    # Ensure model is on the correct device and in training mode.
    # RNNs (LSTM) often require training mode for backward/grad operations in cuDNN.
    model.to(device)
    model.train() 
    
    with torch.enable_grad():
        if model_kind == "tapinn":
            # 1. Capture Latent Representations (Encoder path)
            obs_batch = observations[:diag_samples].to(device)
            latent_batch = model.encode(obs_batch) # type: ignore
            
            # Identify parameters that contribute to the NTK
            target_params = [p for p in model.parameters() if p.requires_grad]
            
            for sample_idx in range(diag_samples):
                sample_coords = coords[sample_idx, :points_per_sample, :].to(device)
                sample_latent = latent_batch[sample_idx : sample_idx + 1]
                
                for idx in range(sample_coords.shape[0]):
                    coord = sample_coords[idx : idx + 1].detach().clone().requires_grad_(True)
                    pred = model.decode(coord, sample_latent) # type: ignore
                    scalar = pred[0, 0]
                    
                    grads = torch.autograd.grad(
                        scalar, target_params, 
                        retain_graph=True, create_graph=False, allow_unused=True
                    )
                    # Filter out None grads (for parameters not in the compute graph of this specific scalar)
                    grad_vec = torch.cat([g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1) 
                                        for g, p in zip(grads, target_params)])
                    jac_rows.append(grad_vec)
        else:
            # 2. Standard PINN path
            target_params = [p for p in model.parameters() if p.requires_grad]
            for sample_idx in range(diag_samples):
                sample_coords = coords[sample_idx, :points_per_sample, :].to(device)
                sample_param = params[sample_idx : sample_idx + 1].to(device)
                
                for idx in range(sample_coords.shape[0]):
                    coord = sample_coords[idx : idx + 1].detach().clone().requires_grad_(True)
                    pred = model(coord, sample_param)
                    scalar = pred[0, 0]
                    
                    grads = torch.autograd.grad(
                        scalar, target_params, 
                        retain_graph=True, create_graph=False, allow_unused=True
                    )
                    grad_vec = torch.cat([g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1) 
                                        for g, p in zip(grads, target_params)])
                    jac_rows.append(grad_vec)

    if not jac_rows:
        return [1e-10], 1.0

    jacobian = torch.stack(jac_rows)
    ntk = jacobian @ jacobian.T
    
    # Eigenvalues of NTK (S^2)
    eigvals = torch.linalg.eigvalsh(ntk).detach().cpu().numpy()
    eigvals = np.sort(np.clip(eigvals.real, 1e-12, None))[::-1]
    
    # Jacobian Condition Number (S_max / S_min)
    singular_vals = torch.linalg.svdvals(jacobian).detach().cpu().numpy()
    singular_vals = np.clip(singular_vals, 1e-12, None)
    condition_number = float(singular_vals.max() / singular_vals.min())
    
    from exp_common.trainers import lipschitz_estimate
    lipschitz = 0.0
    if model_kind == "tapinn":
        obs_b = observations[:diag_samples].to(device)
        lipschitz = float(lipschitz_estimate(model.encoder, obs_b))
        
    return eigvals.tolist(), condition_number, lipschitz



def run_exp_5_theoretical_optimization_landscape(output_root: str, device_name: str, smoke_test: bool, seed: int) -> dict[str, object]:
    """Experiment 5 — Theoretical Optimization Landscape.

    Scientific Role
    ---------------
    * Analyzes the Neural Tangent Kernel (NTK) and Jacobian condition numbers.
    * Directly compares Alternating Optimization (AO) vs. Joint Optimization.
    * Demonstrates stability across a difficulty gradient (ODEs to chaotic PDEs).
    """
    configure_plotting()
    device = get_best_device(device_name)
    run_dir = prepare_run_dir(output_root, "exp_5_theoretical_optimization_landscape")
    
    seeds = [seed] if smoke_test else [seed + offset for offset in range(3)] # Reduced seeds for multi-prob
    problems = ["duffing", "kuramoto", "lorenz", "allen_cahn", "kuramoto_sivashinsky"]
    
    total_epochs = 20 if smoke_test else 400
    checkpoint_step = 10 if smoke_test else 100
    
    callbacks = CallbackConfig(
        early_stopping_patience=total_epochs,
        reduce_lr_patience=total_epochs // 10,
        reduce_lr_factor=0.5,
        min_lr=1e-6,
    )

    spectrum_records = []
    condition_records = []
    seed_rows = []

    for problem_name in tqdm(problems, desc="Problems"):
        fig_subdir = run_dir / "figures" / problem_name
        fig_subdir.mkdir(parents=True, exist_ok=True)
        
        for local_seed in tqdm(seeds, desc=f"Seeds-{problem_name}", leave=False):
            set_global_seed(local_seed)
            
            # 1. Data Generation — coords and targets are normalised to [-1, 1]
            if problem_name == "duffing":
                data = generate_duffing_dataset([0.3, 0.5], 1, 160, (0.0, 10.0), local_seed)
                obs, coords, targets, params, meta, coord_norm, state_norm = prepare_ode_tensors(data, observation_steps=20)
            elif problem_name == "kuramoto":
                data = generate_kuramoto_dataset([2.0, 4.0], 1, 160, (0.0, 10.0), 16, local_seed)
                obs, coords, targets, params, meta, coord_norm, state_norm = prepare_ode_tensors(data, observation_steps=20)
            elif problem_name == "lorenz":
                data = generate_lorenz_dataset([18.0, 24.74], 1, 160, (0.0, 6.0), local_seed)
                obs, coords, targets, params, meta, coord_norm, state_norm = prepare_ode_tensors(data, observation_steps=20)
            elif problem_name == "allen_cahn":
                data = generate_allen_cahn_dataset([1.0], 1, 64, 48, local_seed)
                obs, coords, targets, params, coord_norm, state_norm = prepare_pde_tensors(data, observation_steps=8)
                meta = None
            elif problem_name == "kuramoto_sivashinsky":
                data = generate_kuramoto_sivashinsky_dataset([1.0], 1, 64, 48, local_seed)
                obs, coords, targets, params, coord_norm, state_norm = prepare_pde_tensors(data, observation_steps=8)
                meta = None
            else:
                continue

            train_idx, _, test_idx = _split_indices_three_way(len(params), local_seed)
            train_obs, train_coords, train_targets, train_params = _subset_tensors(train_idx, obs, coords, targets, params)
            test_obs,  test_coords,  test_targets,  test_params  = _subset_tensors(test_idx,  obs, coords, targets, params)
            # Refit normalizers on training split to prevent data leakage
            coord_norm, state_norm = refit_normalizers_on_split(train_obs, train_coords, train_targets)
            train_meta = _subset_optional_tensor(train_idx, meta)
            test_meta  = _subset_optional_tensor(test_idx,  meta)

            models = {
                "tapinn_ao":    ("tapinn", build_tapinn(obs_dim=obs.shape[-1], coord_dim=coords.shape[-1], output_dim=targets.shape[-1], large=False), True, None),
                "tapinn_joint": ("tapinn", build_tapinn(obs_dim=obs.shape[-1], coord_dim=coords.shape[-1], output_dim=targets.shape[-1], large=False), False, None),
                "tapinn_config":("tapinn", build_tapinn(obs_dim=obs.shape[-1], coord_dim=coords.shape[-1], output_dim=targets.shape[-1], large=False), False, "config"),
                "tapinn_soap":  ("tapinn", build_tapinn(obs_dim=obs.shape[-1], coord_dim=coords.shape[-1], output_dim=targets.shape[-1], large=False), False, "soap"),
                "standard_pinn":("direct", StandardPINN(coord_dim=coords.shape[-1], output_dim=targets.shape[-1], hidden_dim=64), None, None),
            }

            for model_name, (family, model, alternating, opt_type) in models.items():
                model.to(device)
                use_config = (opt_type == "config")
                use_soap = (opt_type == "soap")

                # Landscape Captures: checkpoint every `checkpoint_step` epochs
                for epoch in range(0, total_epochs + 1, checkpoint_step):
                    if epoch > 0:
                        desc = f"{problem_name[:3]}-{model_name[:4]}-ep{epoch}-s{local_seed}"
                        if family == "tapinn":
                            train_tapinn(
                                model, problem_name, train_obs, train_coords, train_targets, train_params,
                                device, ode_metadata=train_meta, epochs=checkpoint_step, batch_size=4,
                                alternating=bool(alternating), progress_desc=desc, val_bundle=None, callbacks=callbacks,
                                use_config=use_config, use_soap=use_soap,
                                coord_normalizer=coord_norm, state_normalizer=state_norm,
                            )
                        else:
                            train_direct_model(
                                model, problem_name, "standard", train_obs, train_coords, train_targets, train_params,
                                device, ode_metadata=train_meta, epochs=checkpoint_step, batch_size=4,
                                progress_desc=desc, val_bundle=None, callbacks=callbacks,
                                use_config=use_config, use_soap=use_soap,
                                coord_normalizer=coord_norm, state_normalizer=state_norm,
                            )

                    eigvals, condition_number, lipschitz = _finite_ntk_and_condition(
                        model, "tapinn" if family == "tapinn" else "direct",
                        train_obs, train_coords, train_params, device
                    )
                    spectrum_records.append({"problem": problem_name, "model": model_name, "epoch": epoch, "eigenvalues": eigvals[:24]})
                    condition_records.append({"problem": problem_name, "model": model_name, "epoch": epoch, "condition_number": condition_number, "lipschitz": lipschitz})

                # Final Accuracy Stats — denormalise predictions for evaluation
                if family == "tapinn":
                    preds = predict_tapinn(model, test_obs, test_coords, device, state_normalizer=state_norm).cpu().numpy()
                else:
                    preds = predict_direct(model, "standard", test_obs, test_coords, test_params, device, state_normalizer=state_norm).cpu().numpy()

                # Ground truth in physical (denormalised) space
                truth_phys = state_norm.denormalize(test_targets).cpu().numpy()
                if problem_name in ["duffing", "kuramoto", "lorenz"]:
                    d_mse, p_res = _aggregate_ode_metrics(problem_name, data.times, preds, truth_phys, test_params, test_meta)
                else:
                    boundary = str(data.metadata.get("boundary", "periodic")) if data.metadata else "periodic"
                    d_mse, p_res = _aggregate_pde_metrics(problem_name, data.times, data.space, preds, truth_phys, test_params, boundary)

                seed_rows.append({"problem": problem_name, "seed": local_seed, "model": model_name, "data_mse": d_mse, "physics_residual": p_res})

        # Per-problem plots
        _spectrum_plot(spectrum_records, f"NTK {problem_name.title()} Over Training", fig_subdir / "ntk_spectrum.pdf", problem_name)
        _condition_plot(condition_records, f"Conditioning {problem_name.title()} Over Training", fig_subdir / "condition.pdf", problem_name)

    # Aggregates and Global Reporting
    _final_conditioning_summary_plot(condition_records, run_dir / "figures" / "global_conditioning_summary.pdf")
    write_csv(run_dir / "tables" / "seed_summary.csv", seed_rows)
    write_json(run_dir / "results.json", {
        "device": str(device), "max_epochs": total_epochs,
        "seed_summary": seed_rows, "spectra": spectrum_records, "conditioning": condition_records
    })
    return {"problems": problems}


# ---------------------------------------------------------------------------
# Experiment 2 — PDE Spatiotemporal Suite (multi-baseline, comprehensive)
# ---------------------------------------------------------------------------

# Model specifications for Exp 2: name -> (family, comparison_group)
_EXP2_MODELS = {
    "tapinn":        ("tapinn",   "physics_trained"),
    "standard_pinn": ("standard", "physics_trained"),
    "hyperpinn":     ("hyper",    "physics_trained"),
    "deeponet":      ("deeponet", "physics_trained"),
    "fno":           ("fno",      "supervised_only"),
}


def _build_pde_models(
    obs_steps: int,
    nx: int,
    nt: int,
    smoke_test: bool,
) -> dict[str, object]:
    """Build all Exp-2 models for a given PDE problem grid.

    TAPINN / DeepONet / FNO observe `obs_steps` spatial snapshots of size `nx`.
    StandardPINN / HyperPINN receive only the PDE scalar parameter — they cannot
    see the initial field.  This is the intentional IC-sensitivity comparison.
    """
    branch_input_dim = obs_steps * nx     # flattened obs window
    grid_size = nt * nx                   # flattened spatiotemporal domain
    fno_width = 8 if smoke_test else 32
    return {
        "tapinn":        build_tapinn(obs_dim=nx, coord_dim=2, output_dim=1, large=False),
        "standard_pinn": StandardPINN(coord_dim=2, output_dim=1, hidden_dim=64),
        "hyperpinn":     build_capacity_matched_hyperpinn(coord_dim=2, output_dim=1),
        "deeponet":      DeepONet(branch_input_dim=branch_input_dim, coord_dim=2, output_dim=1, hidden_dim=64, basis_dim=32),
        "fno":           LightweightFNO1d(branch_input_dim=branch_input_dim, grid_size=grid_size, output_dim=1, width=fno_width, modes=4 if smoke_test else 12),
    }


def _train_pde_model(
    model_name: str,
    model,
    problem_name: str,
    train_obs: torch.Tensor,
    train_coords: torch.Tensor,
    train_targets: torch.Tensor,
    train_params: torch.Tensor,
    val_bundle: "ValBundle | None",
    callbacks: "CallbackConfig | None",
    device: torch.device,
    max_epochs: int,
    batch_size: int,
    seed: int,
    coord_normalizer: CoordNormalizer | None = None,
    state_normalizer: StateNormalizer | None = None,
    max_phys_points: int = 64,
):
    """Train a single PDE model, applying identical callbacks for fairness."""
    desc = f"{model_name[:8]}-{problem_name[:8]}-s{seed}"
    family = _EXP2_MODELS[model_name][0]
    if family == "tapinn":
        return train_tapinn(
            model,
            problem_name=problem_name,
            observations=train_obs,
            coords=train_coords,
            targets=train_targets,
            params=train_params,
            device=device,
            ode_metadata=None,
            epochs=max_epochs,
            batch_size=batch_size,
            progress_desc=desc,
            val_bundle=val_bundle,
            callbacks=callbacks,
            coord_normalizer=coord_normalizer,
            state_normalizer=state_normalizer,
            max_phys_points=max_phys_points,
        )
    elif family == "fno":
        fno_val: ValBundle | None = None
        if val_bundle is not None:
            fno_val = ValBundle(
                observations=val_bundle.observations,
                coords=val_bundle.coords,
                targets=val_bundle.targets,
                params=val_bundle.params,
            )
        return train_fno_model(
            model,
            observations=train_obs,
            targets=train_targets,
            device=device,
            epochs=max_epochs,
            batch_size=batch_size,
            progress_desc=desc,
            val_bundle=fno_val,
            callbacks=callbacks,
        )
    else:
        return train_direct_model(
            model,
            problem_name=problem_name,
            model_kind=family,
            observations=train_obs,
            coords=train_coords,
            targets=train_targets,
            params=train_params,
            device=device,
            ode_metadata=None,
            epochs=max_epochs,
            batch_size=batch_size,
            progress_desc=desc,
            val_bundle=val_bundle,
            callbacks=callbacks,
            coord_normalizer=coord_normalizer,
            state_normalizer=state_normalizer,
            max_phys_points=max_phys_points,
        )



def _predict_pde_model(
    model_name: str,
    model,
    test_obs: torch.Tensor,
    test_coords: torch.Tensor,
    test_params: torch.Tensor,
    grid_size: int,
    device: torch.device,
    state_normalizer: StateNormalizer | None = None,
) -> tuple[np.ndarray, float]:
    """Run inference for a PDE model; return (predictions_numpy, inference_ms)."""
    family = _EXP2_MODELS[model_name][0]
    if family == "tapinn":
        return _measure_inference_ms(
            lambda *a: predict_tapinn(*a, state_normalizer=state_normalizer).detach().cpu().numpy(),
            test_obs.shape[0],
            model, test_obs, test_coords, device,
        )
    elif family == "fno":
        return _measure_inference_ms(
            _fno_predict_numpy, test_obs.shape[0],
            model, test_obs, grid_size, device,
        )
    else:
        return _measure_inference_ms(
            lambda *a: predict_direct(*a, state_normalizer=state_normalizer).detach().cpu().numpy(),
            test_obs.shape[0],
            model, family, test_obs, test_coords, test_params, device,
        )


def _run_pde_seed_all_models(
    problem_name: str,
    param_values: list[float],
    seed: int,
    smoke_test: bool,
    device: torch.device,
    max_epochs: int,
    callbacks: CallbackConfig,
) -> dict[str, dict]:
    """Train and evaluate ALL Exp-2 models on one PDE problem/seed.

    Returns a mapping  model_name -> metrics_dict  containing data_mse,
    physics_residual, param_count, epochs_trained, seconds_per_epoch,
    inference_ms, comparison_group, and the first test field (prediction,
    truth) for heatmap visualisation.

    Scientific design notes
    -----------------------
    * Three-way train/val/test split (70/15/15) shared identically across all
      models — no model sees the test set during training or validation.
    * HyperPINN and StandardPINN are parameter-only conditioned: they receive
      the PDE scalar but NOT the initial field snapshot.  TAPINN and DeepONet
      see an observation window of `obs_steps` time snapshots.  This is the
      direct PDE analogue of the IC-sensitivity finding from Exp 1 Kuramoto.
    * Same CallbackConfig (EarlyStopping + ReduceLROnPlateau) for all models.
    """
    # ------------------------------------------------------------------
    # 1. Generate data
    # ------------------------------------------------------------------
    num_samples_per_param = 1 if smoke_test else 6
    # KS is 4th-order PDE: autograd is O(N^4), so keep grid tiny in smoke mode
    if smoke_test and problem_name == "kuramoto_sivashinsky":
        nx, nt = 8, 6   # 48 physics points — manageable even on CPU
    else:
        nx = 20 if smoke_test else 64
        nt = 14 if smoke_test else 48

    if problem_name == "allen_cahn":
        data = generate_allen_cahn_dataset(param_values, num_samples=num_samples_per_param, nx=nx, nt=nt, seed=seed)
    elif problem_name == "burgers":
        data = generate_burgers_dataset(param_values, num_samples=num_samples_per_param, nx=nx, nt=nt, seed=seed)
    elif problem_name == "kuramoto_sivashinsky":
        data = generate_kuramoto_sivashinsky_dataset(param_values, num_samples=num_samples_per_param, nx=nx, nt=nt, seed=seed)
    else:
        raise ValueError(problem_name)

    obs_steps = 2 if (smoke_test and problem_name == "kuramoto_sivashinsky") else (4 if smoke_test else 8)

    observations, coords, targets, params, coord_norm, state_norm = prepare_pde_tensors(data, observation_steps=obs_steps)

    # ------------------------------------------------------------------
    # 2. Three-way split (70 / 15 / 15)
    # ------------------------------------------------------------------
    train_idx, val_idx, test_idx = _split_indices_three_way(len(params), seed)
    train_obs, train_coords, train_targets, train_params = _subset_tensors(train_idx, observations, coords, targets, params)
    val_obs,   val_coords,   val_targets,   val_params   = _subset_tensors(val_idx,   observations, coords, targets, params)
    test_obs,  test_coords,  test_targets,  test_params  = _subset_tensors(test_idx,  observations, coords, targets, params)

    # Refit normalizers on training split only (no data leakage)
    coord_norm, state_norm = refit_normalizers_on_split(train_obs, train_coords, train_targets)

    val_bundle = ValBundle(
        observations=val_obs,
        coords=val_coords,
        targets=val_targets,
        params=val_params,
    )
    active_val_bundle: ValBundle | None = val_bundle if val_obs.shape[0] > 0 else None
    active_callbacks: CallbackConfig | None = callbacks if val_obs.shape[0] > 0 else None

    # ------------------------------------------------------------------
    # 3. Build models
    # ------------------------------------------------------------------
    actual_nx = len(data.space)
    actual_nt = len(data.times)
    grid_size = actual_nt * actual_nx
    models = _build_pde_models(obs_steps, actual_nx, actual_nt, smoke_test)
    batch_size = 4
    boundary = str(data.metadata.get("boundary", "periodic")) if data.metadata else "periodic"

    # ------------------------------------------------------------------
    # 4. Train + evaluate each model
    # ------------------------------------------------------------------
    results: dict[str, dict] = {}
    truth = state_norm.denormalize(test_targets).cpu().numpy()  # physical space

    for model_name, model in tqdm(models.items(), desc=f"{problem_name[:8]}/models", leave=False):
        # KS with 4th-order autograd is expensive: cap physics batch size in smoke mode
        ks_smoke = smoke_test and problem_name == "kuramoto_sivashinsky"
        train_result = _train_pde_model(
            model_name, model, problem_name,
            train_obs, train_coords, train_targets, train_params,
            active_val_bundle, active_callbacks, device, max_epochs, batch_size, seed,
            coord_normalizer=coord_norm, state_normalizer=state_norm,
            max_phys_points=8 if ks_smoke else 64,
        )
        predictions, inference_ms = _predict_pde_model(
            model_name, model, test_obs, test_coords, test_params, grid_size, device,
            state_normalizer=state_norm,
        )
        # predictions are denormalised (physical space)
        data_mse, physics_residual = _aggregate_pde_metrics(
            problem_name, data.times, data.space, predictions, truth, test_params, boundary,
        )
        _, comparison_group = _EXP2_MODELS[model_name]
        # Heatmap data from first test sample (physical space)
        pred_field = predictions[0, :, 0].reshape(actual_nt, actual_nx)
        truth_field = truth[0, :, 0].reshape(actual_nt, actual_nx)
        results[model_name] = {
            "data_mse":          data_mse,
            "physics_residual":  physics_residual,
            "param_count":       count_parameters(model),
            "epochs_trained":    train_result.epochs_trained,
            "best_val_loss":     train_result.best_val_loss,
            "seconds_per_epoch": train_result.seconds_per_epoch,
            "inference_ms":      inference_ms,
            "comparison_group":  comparison_group,
            "prediction":        pred_field,
            "truth":             truth_field,
        }

    return results


# ---------------------------------------------------------------------------
# Experiment 2 entry point
# ---------------------------------------------------------------------------

def run_exp_2_pde_spatiotemporal_suite(output_root: str, device_name: str, smoke_test: bool, seed: int) -> dict[str, object]:
    """PDE Spatiotemporal Suite — comprehensive multi-baseline benchmark.

    Scientific role
    ---------------
    1. **IC-sensitivity on PDEs.**  Multiple samples per PDE parameter test
       whether observation-conditioning (TAPINN, DeepONet) generalises across
       field profiles that share the same parameter but differ in initial
       conditions.  StandardPINN and HyperPINN receive only the PDE scalar —
       they cannot distinguish between different IC-induced trajectories.
    2. **Difficulty gradient.**  Allen-Cahn (smooth) → Burgers (shock formation)
       → Kuramoto-Sivashinsky (spatiotemporal chaos) provides increasing challenge,
       the PDE analogue of Exp 1's Duffing → Kuramoto → Lorenz gradient.
    3. **Physics compliance for 2-D fields.**  This experiment is the only place
       in the suite that reports PDE residuals for ALL baselines.

    Baselines (identical to Exp 1)
    --------------------------------
    * TAPINN           — observation-conditioned, physics_trained
    * StandardPINN     — param-conditioned MLP, physics_trained
    * HyperPINN        — hypernetwork, physics_trained
    * DeepONet         — operator network, physics_trained
    * FNO              — Fourier operator, supervised_only

    Callbacks (identical for ALL models)
    -------------------------------------
    * EarlyStopping: patience=15
    * ReduceLROnPlateau: patience=8, factor=0.5, min_lr=1e-6
    """
    configure_plotting()
    device = get_best_device(device_name)
    run_dir = prepare_run_dir(output_root, "exp_2_pde_spatiotemporal_suite")
    seeds = [seed] if smoke_test else [seed + offset for offset in range(5)]
    configs = {
        "allen_cahn":           [0.8, 1.1]   if smoke_test else [0.75, 0.9, 1.05, 1.2],
        "burgers":              [0.02, 0.05] if smoke_test else [0.015, 0.03, 0.05, 0.08],
        "kuramoto_sivashinsky": [0.8, 1.0]   if smoke_test else [0.75, 0.9, 1.05, 1.2],
    }
    max_epochs = 4 if smoke_test else 100
    callbacks = CallbackConfig(
        early_stopping_patience=2 if smoke_test else 15,
        reduce_lr_patience=1 if smoke_test else 8,
        reduce_lr_factor=0.5,
        min_lr=1e-6,
    )
    model_names = list(_EXP2_MODELS.keys())

    seed_rows: list[dict] = []
    problem_model_summaries: list[dict] = []
    heatmap_data: dict[str, dict] = {}  # problem -> TAPINN info (first seed)

    for problem_name, param_values in tqdm(configs.items(), desc="PDE systems", leave=False):
        per_model_seed_data: dict[str, list[dict]] = {m: [] for m in model_names}

        for local_seed in tqdm(seeds, desc=f"{problem_name} seeds", leave=False):
            set_global_seed(local_seed)
            results = _run_pde_seed_all_models(
                problem_name, param_values, local_seed, smoke_test, device,
                max_epochs, callbacks,
            )
            if local_seed == seeds[0] and "tapinn" in results:
                heatmap_data[problem_name] = results["tapinn"]

            for model_name, info in results.items():
                per_model_seed_data[model_name].append(info)
                seed_rows.append({
                    "problem":          problem_name,
                    "seed":             local_seed,
                    "model":            model_name,
                    "comparison_group": info["comparison_group"],
                    "data_mse":         info["data_mse"],
                    "physics_residual": info["physics_residual"],
                    "param_count":      info["param_count"],
                    "epochs_trained":   info["epochs_trained"],
                    "best_val_loss":    info["best_val_loss"],
                    "seconds_per_epoch":info["seconds_per_epoch"],
                    "inference_ms":     info["inference_ms"],
                })

        # Aggregate across seeds per model
        for model_name in model_names:
            rows = per_model_seed_data[model_name]
            if not rows:
                continue
            data_mean, data_std = mean_std(r["data_mse"]          for r in rows)
            phys_mean, phys_std = mean_std(r["physics_residual"]   for r in rows)
            ep_mean,   ep_std   = mean_std(r["epochs_trained"]     for r in rows)
            spe_mean,  spe_std  = mean_std(r["seconds_per_epoch"]  for r in rows)
            inf_mean,  inf_std  = mean_std(r["inference_ms"]       for r in rows)
            problem_model_summaries.append({
                "problem":               problem_name,
                "model":                 model_name,
                "comparison_group":      rows[0]["comparison_group"],
                "param_count":           rows[0]["param_count"],
                "data_mse_mean":         data_mean,
                "data_mse_std":          data_std,
                "physics_residual_mean": phys_mean,
                "physics_residual_std":  phys_std,
                "epochs_trained_mean":   ep_mean,
                "epochs_trained_std":    ep_std,
                "seconds_per_epoch_mean":spe_mean,
                "inference_ms_mean":     inf_mean,
            })

        # Heatmap triptych (first seed, TAPINN prediction)
        if problem_name in heatmap_data:
            hd = heatmap_data[problem_name]
            _heatmap_triptych(
                hd["truth"], hd["prediction"],
                f"{problem_name.replace('_', ' ').title()} Heatmaps — TAPINN",
                run_dir / "figures" / f"{problem_name}_heatmap_triptych.pdf",
            )

        # Per-problem metric bar charts (all models)
        prob_rows = [r for r in problem_model_summaries if r["problem"] == problem_name]
        _metrics_bar_chart(
            [r["model"] for r in prob_rows],
            [r["data_mse_mean"] for r in prob_rows],
            "Data MSE",
            f"{problem_name.replace('_', ' ').title()} — Data MSE by Model",
            run_dir / "figures" / f"{problem_name}_data_mse_bar.pdf",
            comparison_groups=[r["comparison_group"] for r in prob_rows],
        )
        _metrics_bar_chart(
            [r["model"] for r in prob_rows],
            [r["physics_residual_mean"] for r in prob_rows],
            "Physics Residual",
            f"{problem_name.replace('_', ' ').title()} — Physics Residual by Model",
            run_dir / "figures" / f"{problem_name}_physics_residual_bar.pdf",
            comparison_groups=[r["comparison_group"] for r in prob_rows],
        )

    # Legacy summary.csv (TAPINN-only for backward compat with tests)
    tapinn_summaries = [
        {
            "problem":               r["problem"],
            "data_mse_mean":         r["data_mse_mean"],
            "data_mse_std":          r["data_mse_std"],
            "physics_residual_mean": r["physics_residual_mean"],
            "physics_residual_std":  r["physics_residual_std"],
        }
        for r in problem_model_summaries if r["model"] == "tapinn"
    ]
    write_csv(run_dir / "tables" / "seed_metrics.csv", seed_rows)
    write_csv(run_dir / "tables" / "model_summary.csv", problem_model_summaries)
    write_csv(run_dir / "tables" / "summary.csv", tapinn_summaries)  # legacy
    payload = {
        "device":            str(device),
        "smoke_test":        smoke_test,
        "max_epochs":        max_epochs,
        "callbacks": {
            "early_stopping_patience": callbacks.early_stopping_patience,
            "reduce_lr_patience":      callbacks.reduce_lr_patience,
            "reduce_lr_factor":        callbacks.reduce_lr_factor,
            "min_lr":                  callbacks.min_lr,
        },
        "models":            model_names,
        "comparison_groups": {m: _EXP2_MODELS[m][1] for m in model_names},
        "summary":           problem_model_summaries,
    }
    write_json(run_dir / "results.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Experiment 3 — SOTA Baselines & Capacity (multi-baseline, comprehensive)
# ---------------------------------------------------------------------------

# Model specifications for Exp 3: name -> (family, comparison_group)
_EXP3_MODELS = {
    "tapinn":        ("tapinn",       "physics_trained"),
    "tapinn_large":  ("tapinn_large", "physics_trained"),
    "hyperpinn":     ("hyper",        "physics_trained"),
    "hyper_lr_pinn": ("hyper",        "physics_trained"),
    "deeponet":      ("deeponet",     "physics_trained"),
    "fno":           ("fno",          "supervised_only"),
}


def _build_exp3_models(
    obs_dim: int,
    coord_dim: int,
    output_dim: int,
    branch_dim: int,
    grid_size: int,
    smoke_test: bool,
) -> dict[str, object]:
    """Build all Exp-3 models for a given problem's dimensions.

    Uses the same architecture families as Exp 1/2, adding the `tapinn_large`
    variant to justify scaling claims.
    """
    fno_width = 8 if smoke_test else 32
    return {
        "tapinn":        build_tapinn(obs_dim=obs_dim, coord_dim=coord_dim, output_dim=output_dim, large=False),
        "tapinn_large":  build_tapinn(obs_dim=obs_dim, coord_dim=coord_dim, output_dim=output_dim, large=True),
        "hyperpinn":     build_capacity_matched_hyperpinn(coord_dim=coord_dim, output_dim=output_dim),
        "hyper_lr_pinn": build_low_rank_hyperpinn(coord_dim=coord_dim, output_dim=output_dim, hidden_dim=64, rank=4),
        "deeponet":      DeepONet(branch_input_dim=branch_dim, coord_dim=coord_dim, output_dim=output_dim, hidden_dim=64, basis_dim=32),
        "fno":           LightweightFNO1d(branch_input_dim=branch_dim, grid_size=grid_size, output_dim=output_dim, width=fno_width, modes=4 if smoke_test else 12),
    }


def _train_exp3_model(
    model_name: str,
    model: any,
    problem_name: str,
    train_obs: torch.Tensor,
    train_coords: torch.Tensor,
    train_targets: torch.Tensor,
    train_params: torch.Tensor,
    train_ode_meta: torch.Tensor | None,
    val_bundle: ValBundle | None,
    callbacks: CallbackConfig | None,
    device: torch.device,
    max_epochs: int,
    batch_size: int,
    seed: int,
):
    """Dispatch training for any Exp-3 model variant."""
    desc = f"{model_name[:8]}-{problem_name[:8]}-s{seed}"
    family = _EXP3_MODELS[model_name][0]
    if family in ("tapinn", "tapinn_large"):
        return train_tapinn(
            model, problem_name, train_obs, train_coords, train_targets, train_params,
            device, ode_metadata=train_ode_meta, epochs=max_epochs, batch_size=batch_size,
            progress_desc=desc, val_bundle=val_bundle, callbacks=callbacks,
        )
    elif family == "fno":
        fno_val: ValBundle | None = None
        if val_bundle is not None:
            fno_val = ValBundle(observations=val_bundle.observations, coords=val_bundle.coords, targets=val_bundle.targets, params=val_bundle.params)
        return train_fno_model(
            model, train_obs, train_targets, device, epochs=max_epochs, batch_size=batch_size,
            progress_desc=desc, val_bundle=fno_val, callbacks=callbacks,
        )
    else:
        # direct family: standard, hyper, deeponet
        return train_direct_model(
            model, problem_name, family, train_obs, train_coords, train_targets, train_params,
            device, ode_metadata=train_ode_meta, epochs=max_epochs, batch_size=batch_size,
            progress_desc=desc, val_bundle=val_bundle, callbacks=callbacks,
        )


def _predict_exp3_model(
    model_name: str,
    model: any,
    test_obs: torch.Tensor,
    test_coords: torch.Tensor,
    test_params: torch.Tensor,
    grid_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Run inference for Exp-3 models; consistent with Exp 1 and 2."""
    family = _EXP3_MODELS[model_name][0]
    if family in ("tapinn", "tapinn_large"):
        return _measure_inference_ms(_tapinn_predict_numpy, test_obs.shape[0], model, test_obs, test_coords, device)
    elif family == "fno":
        return _measure_inference_ms(_fno_predict_numpy, test_obs.shape[0], model, test_obs, grid_size, device)
    else:
        return _measure_inference_ms(_direct_predict_numpy, test_obs.shape[0], model, family, test_obs, test_coords, test_params, device)


def _eval_model_on_dataset(
    model_name: str,
    task_name: str,
    system_data: any,
    observations: torch.Tensor,
    coords: torch.Tensor,
    targets: torch.Tensor,
    params: torch.Tensor,
    ode_metadata: torch.Tensor | None,
    problem_name: str,
    device: torch.device,
    smoke_test: bool,
    seed: int,
    max_epochs: int,
    callbacks: CallbackConfig,
) -> dict:
    """Standardized evaluation for Experiment 3.

    Scientific Role
    ---------------
    * Implements the 70/15/15 three-way split required for fair callback triggers.
    * Reuses the multi-baseline dispatch infrastructure established for Exp 1/2.
    * Calculates the 'generalization_gap' (MSE_test - MSE_train) to detect overfitting
      in high-capacity models (HyperPINN, TAPINN-Large).
    """
    # ------------------------------------------------------------------
    # 1. Three-way split (70 / 15 / 15)
    # ------------------------------------------------------------------
    train_idx, val_idx, test_idx = _split_indices_three_way(len(params), seed)
    train_obs, train_coords, train_targets, train_params = _subset_tensors(train_idx, observations, coords, targets, params)
    val_obs,   val_coords,   val_targets,   val_params   = _subset_tensors(val_idx,   observations, coords, targets, params)
    test_obs,  test_coords,  test_targets,  test_params  = _subset_tensors(test_idx,  observations, coords, targets, params)

    train_ode_meta = _subset_optional_tensor(train_idx, ode_metadata)
    val_ode_meta   = _subset_optional_tensor(val_idx,   ode_metadata)
    test_ode_meta  = _subset_optional_tensor(test_idx,  ode_metadata)

    val_bundle = ValBundle(val_obs, val_coords, val_targets, val_params, val_ode_meta)
    active_val_bundle = val_bundle if val_obs.shape[0] > 0 else None
    active_callbacks = callbacks   if val_obs.shape[0] > 0 else None

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    obs_dim = observations.shape[-1]
    coord_dim = coords.shape[-1]
    output_dim = targets.shape[-1]
    branch_dim = train_obs.shape[1] * train_obs.shape[2]
    grid_size = targets.shape[1]

    models = _build_exp3_models(obs_dim, coord_dim, output_dim, branch_dim, grid_size, smoke_test)
    model = models[model_name]

    # ------------------------------------------------------------------
    # 3. Train + eval
    # ------------------------------------------------------------------
    train_result = _train_exp3_model(
        model_name, model, problem_name,
        train_obs, train_coords, train_targets, train_params, train_ode_meta,
        active_val_bundle, active_callbacks, device, max_epochs, batch_size=4, seed=seed,
    )

    preds, inference_ms = _predict_exp3_model(model_name, model, test_obs, test_coords, test_params, grid_size, device)

    # Generalization gap check (using a separate call to train_pred for simplicity)
    family = _EXP3_MODELS[model_name][0]
    if family in ("tapinn", "tapinn_large"):
        train_pred = _tapinn_predict_numpy(model, train_obs, train_coords, device)
    elif family == "fno":
        train_pred = _fno_predict_numpy(model, train_obs, grid_size, device)
    else:
        train_pred = _direct_predict_numpy(model, family, train_obs, train_coords, train_params, device)

    train_truth = train_targets.cpu().numpy()
    test_truth = test_targets.cpu().numpy()
    gen_gap = float(mse(preds, test_truth) - mse(train_pred, train_truth))

    if task_name == "duffing":
        data_mse, physics = _aggregate_ode_metrics(problem_name, system_data.times, preds, test_truth, test_params, test_ode_meta)
    else:
        boundary = str(system_data.metadata.get("boundary", "periodic")) if system_data.metadata else "periodic"
        data_mse, physics = _aggregate_pde_metrics(problem_name, system_data.times, system_data.space, preds, test_truth, test_params, boundary)

    return {
        "task":              task_name,
        "model_name":        model_name,
        "data_mse":          data_mse,
        "physics_residual":  physics,
        "seconds_per_epoch": train_result.seconds_per_epoch,
        "epochs_trained":    train_result.epochs_trained,
        "best_val_loss":     train_result.best_val_loss,
        "inference_ms":      inference_ms,
        "task_param_count":  count_parameters(model),
        "generalization_gap": gen_gap,
        "comparison_group":  _EXP3_MODELS[model_name][1],
    }


def run_exp_3_sota_baselines_and_capacity(output_root: str, device_name: str, smoke_test: bool, seed: int) -> dict[str, object]:
    """Experiment 3 — SOTA Baselines & Capacity.

    Scientific Role
    ---------------
    * Benchmarks model performance on one ODE (Duffing) and one PDE (Allen-Cahn)
      across diverse architectures.
    * Explicitly compares `tapinn` (standard) against `tapinn_large` (scaled)
      to demonstrate architectural capacity benefits.
    * Highlights the "Pareto Frontier" of physics consistency vs. inference speed.
    * Exposes generalization gaps in high-capacity models.

    The experiment now uses identical CallbackConfig and 3-way data splits to
    ensure that claims about SOTA performance are based on fair, well-converged metrics.
    """
    configure_plotting()
    device = get_best_device(device_name)
    run_dir = prepare_run_dir(output_root, "exp_3_sota_baselines_and_capacity")

    max_epochs = 4 if smoke_test else 100
    callbacks = CallbackConfig(
        early_stopping_patience=2 if smoke_test else 15,
        reduce_lr_patience=1 if smoke_test else 8,
        reduce_lr_factor=0.5,
        min_lr=1e-6,
    )
    seeds = [seed] if smoke_test else [seed + offset for offset in range(5)]
    model_names = list(_EXP3_MODELS.keys())

    per_task_seed_metrics = []
    summary_rows = []
    pareto_rows = []
    gap_labels = []
    gap_values = []

    # ------------------------------------------------------------------
    # Pre-calculate canonical parameter counts (Duffing baseline)
    # ------------------------------------------------------------------
    canonical_counts = {
        "tapinn":        count_parameters(build_tapinn(obs_dim=2, coord_dim=1, output_dim=2, large=False)),
        "tapinn_large":  count_parameters(build_tapinn(obs_dim=2, coord_dim=1, output_dim=2, large=True)),
        "hyperpinn":     count_parameters(build_capacity_matched_hyperpinn(coord_dim=1, output_dim=2)),
        "hyper_lr_pinn": count_parameters(build_low_rank_hyperpinn(coord_dim=1, output_dim=2, hidden_dim=64, rank=4)),
        "deeponet":      count_parameters(DeepONet(branch_input_dim=20*2, coord_dim=1, output_dim=2, hidden_dim=64, basis_dim=32)),
        "fno":           count_parameters(LightweightFNO1d(branch_input_dim=20*2, grid_size=160, output_dim=2, width=32, modes=12)),
    }

    for model_name in model_names:
        seed_summaries = []
        for local_seed in tqdm(seeds, desc=f"{model_name} seeds", leave=False):
            set_global_seed(local_seed)

            # Duffing: Increase task consistency with 10 trajectories instead of 1
            duff_params = [0.24, 0.38, 0.52] if smoke_test else np.linspace(0.2, 0.55, 7).tolist()
            duff_data = generate_duffing_dataset(duff_params, num_trajectories=1 if smoke_test else 10, num_points=56 if smoke_test else 160, t_span=(0.0, 16.0), seed=local_seed)
            duff_obs, duff_coords, duff_targets, duff_params_t, duff_meta, _dcn, _dsn = prepare_ode_tensors(duff_data, observation_steps=8 if smoke_test else 20)

            # Allen-Cahn: Increase samples from 1 to 8 instead of 1
            allen_params = [0.8, 1.1] if smoke_test else [0.75, 0.9, 1.05, 1.2]
            allen_data = generate_allen_cahn_dataset(allen_params, num_samples=1 if smoke_test else 8, nx=16 if smoke_test else 48, nt=16 if smoke_test else 36, seed=local_seed + 11)
            allen_obs, allen_coords, allen_targets, allen_params_t, _acn, _asn = prepare_pde_tensors(allen_data, observation_steps=4 if smoke_test else 8)

            m_duffing = _eval_model_on_dataset(model_name, "duffing", duff_data, duff_obs, duff_coords, duff_targets, duff_params_t, duff_meta, "duffing", device, smoke_test, local_seed, max_epochs, callbacks)
            m_allen   = _eval_model_on_dataset(model_name, "allen_cahn", allen_data, allen_obs, allen_coords, allen_targets, allen_params_t, None, "allen_cahn", device, smoke_test, local_seed + 101, max_epochs, callbacks)

            per_task_seed_metrics.extend([{"seed": local_seed, **m_duffing}, {"seed": local_seed, **m_allen}])

            seed_summaries.append({
                "data_mse":          float(np.mean([m_duffing["data_mse"], m_allen["data_mse"]])),
                "physics_residual":  float(np.mean([m_duffing["physics_residual"], m_allen["physics_residual"]])),
                "seconds_per_epoch": float(np.mean([m_duffing["seconds_per_epoch"], m_allen["seconds_per_epoch"]])),
                "inference_ms":      float(np.mean([m_duffing["inference_ms"], m_allen["inference_ms"]])),
                "gen_gap":           float(np.mean([m_duffing["generalization_gap"], m_allen["generalization_gap"]])),
                "task_params":       max(m_duffing["task_param_count"], m_allen["task_param_count"]),
                "epochs_trained":    float(np.mean([m_duffing["epochs_trained"], m_allen["epochs_trained"]])),
            })

        # Aggregate summary across seeds
        data_m, data_s = mean_std(s["data_mse"]         for s in seed_summaries)
        phys_m, phys_s = mean_std(s["physics_residual"] for s in seed_summaries)
        ep_m, ep_s     = mean_std(s["epochs_trained"]    for s in seed_summaries)
        inf_m, inf_s   = mean_std(s["inference_ms"]      for s in seed_summaries)
        gap_m, gap_s   = mean_std(s["gen_gap"]           for s in seed_summaries)

        summary_rows.append({
            "model_name":              model_name,
            "comparison_group":        _EXP3_MODELS[model_name][1],
            "param_count":             canonical_counts[model_name],
            "task_param_count_max":    max(s["task_params"] for s in seed_summaries),
            "data_mse_mean":           data_m,
            "data_mse_std":            data_s,
            "physics_residual_mean":    phys_m,
            "physics_residual_std":     phys_s,
            "epochs_trained_mean":     ep_m,
            "epochs_trained_std":      ep_s,
            "inference_ms_mean":       inf_m,
            "inference_ms_std":        inf_s,
            "generalization_gap_mean":  gap_m,
            "generalization_gap_std":   gap_s,
        })

        if _EXP3_MODELS[model_name][1] == "physics_trained":
            pareto_rows.append({"name": model_name, "physics_residual": phys_m, "inference_ms": inf_m})

        if model_name in ("tapinn_large", "hyperpinn"):
            gap_labels.append(model_name)
            gap_values.append(gap_m)

    write_csv(run_dir / "tables" / "per_task_seed_metrics.csv", per_task_seed_metrics)
    write_csv(run_dir / "tables" / "capacity_benchmark.csv", summary_rows)
    write_csv(run_dir / "tables" / "model_summary.csv", summary_rows)  # standardized name

    _scatter_plot(pareto_rows, "Physics-Trained Models: Inference vs Physics Residual", run_dir / "figures" / "pareto_frontier.pdf")
    _bar_plot(gap_labels, gap_values, "Generalization Gap: TAPINN-Large vs HyperPINN", run_dir / "figures" / "overfitting_gap_bar_chart.pdf")

    payload = {
        "device":            str(device),
        "smoke_test":        smoke_test,
        "max_epochs":        max_epochs,
        "callbacks": {
            "early_stopping_patience": callbacks.early_stopping_patience,
            "reduce_lr_patience":      callbacks.reduce_lr_patience,
        },
        "per_task_seed_metrics": per_task_seed_metrics,
        "summary":               summary_rows,
    }
    write_json(run_dir / "results.json", payload)
    return payload
