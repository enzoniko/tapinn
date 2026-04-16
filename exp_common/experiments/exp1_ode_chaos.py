from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from ..device import get_best_device
from ..io_utils import prepare_run_dir, write_csv, write_json
from ..metrics import mean_std
from ..models import (
    DeepONet,
    LightweightFNO1d,
    StandardPINN,
    build_capacity_matched_hyperpinn,
    build_tapinn,
    count_parameters,
)
from ..plotting import configure_plotting, save_figure
from ..problems import generate_duffing_dataset, generate_kuramoto_dataset, generate_lorenz_dataset
from ..repro import set_global_seed
from ..trainers import (
    CallbackConfig,
    CoordNormalizer,
    StateNormalizer,
    ValBundle,
    predict_direct,
    predict_tapinn,
    prepare_ode_tensors,
    refit_normalizers_on_physical_split,
    train_direct_model,
    train_fno_model,
    train_tapinn,
)
from .common import (
    tqdm,
    _aggregate_ode_metrics,
    _fno_predict_numpy,
    _measure_inference_ms,
    _phase_plot,
    _split_indices_three_way,
    _subset_optional_tensor,
    _subset_tensors,
)


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
) -> dict[str, nn.Module]:
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
    model: nn.Module,
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
    model: nn.Module,
    test_obs: torch.Tensor,
    test_coords: torch.Tensor,
    test_params: torch.Tensor,
    num_points: int,
    device: torch.device,
    state_normalizer: StateNormalizer | None = None,
 ) -> tuple[np.ndarray[Any, Any], float]:
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
) -> dict[str, dict[str, Any]]:
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
            num_trajectories=1 if smoke_test else 100,
            num_points=num_points,
            t_span=(0.0, 16.0),
            seed=seed,
        )
    elif problem_name == "lorenz":
        data = generate_lorenz_dataset(
            param_values,
            num_trajectories=1 if smoke_test else 60,
            num_points=num_points,
            t_span=(0.0, 4.5),
            seed=seed,
        )
    elif problem_name == "kuramoto":
        data = generate_kuramoto_dataset(
            param_values,
            num_trajectories=1 if smoke_test else 60,
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
    (
        coord_norm, state_norm,
        train_targets, val_targets, test_targets,
        train_obs, val_obs, test_obs,
    ) = refit_normalizers_on_physical_split(
        original_coord_norm=coord_norm,
        original_state_norm=state_norm,
        train_coords_normed=train_coords,
        train_targets_normed=train_targets,
        train_obs_normed=train_obs,
        val_targets_normed=val_targets,
        val_obs_normed=val_obs,
        test_targets_normed=test_targets,
        test_obs_normed=test_obs,
    )

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
    results: dict[str, dict[str, Any]] = {}
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
    model_results: dict[str, dict[str, Any]],
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
    model_results_flat: list[dict[str, Any]],
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
    max_epochs = 4 if smoke_test else 10000
    callbacks = CallbackConfig(
        early_stopping_patience=2 if smoke_test else 50,
        reduce_lr_patience=1 if smoke_test else 20,
        reduce_lr_factor=0.5,
        min_lr=1e-6,
    )
    model_names = list(_EXP1_MODELS.keys())

    # Rows for per-seed-model-problem CSV
    seed_rows: list[dict[str, Any]] = []
    # Aggregated per problem+model
    problem_model_summaries: list[dict[str, Any]] = []
    # Representative phase-space data (first seed only)
    phase_data: dict[str, dict[str, dict[str, Any]]] = {}  # problem -> model -> info

    for problem_name, param_values in tqdm(configs.items(), desc="ODE systems", leave=False):
        per_model_seed_data: dict[str, list[dict[str, Any]]] = {m: [] for m in model_names}

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
    payload: dict[str, object] = {
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
