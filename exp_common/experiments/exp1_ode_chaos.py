from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from ..device import get_best_device
from ..io_utils import prepare_run_dir, write_csv, write_json
from ..metrics import compute_disambiguation_score, compute_relative_l2_error, mean_std
from ..models import count_parameters, create_model
from ..plotting import configure_plotting, save_figure
from ..problems import generate_duffing_dataset, generate_kuramoto_dataset, generate_lorenz_dataset
from ..repro import set_global_seed
from ..trainers import (
    CallbackConfig,
    CoordNormalizer,
    StateNormalizer,
    ValBundle,
    predict_direct,
    prepare_ode_tensors,
    refit_normalizers_on_physical_split,
    train_direct_model,
    train_fno_model,
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


_EXP_EPOCHS_ENV = "TAPINN_EXPERIMENT_EPOCHS"
_EXP_ALL_CONFIGS_ENV = "TAPINN_EXPERIMENT_ALL_CONFIGS"

_FAMILY_ORDER = ["StandardPINN", "HyperPINN", "HyperLRPINN", "DeepONet", "FNO"]

_FULL_MODEL_SPECS = [
    {"name": "StandardPINN", "family": "StandardPINN", "with_oc": False},
    {"name": "StandardPINN_OC", "family": "StandardPINN", "with_oc": True},
    {"name": "HyperPINN", "family": "HyperPINN", "with_oc": False},
    {"name": "HyperPINN_OC", "family": "HyperPINN", "with_oc": True},
    {"name": "HyperLRPINN", "family": "HyperLRPINN", "with_oc": False},
    {"name": "HyperLRPINN_OC", "family": "HyperLRPINN", "with_oc": True},
    {"name": "DeepONet", "family": "DeepONet", "with_oc": False},
    {"name": "DeepONet_OC", "family": "DeepONet", "with_oc": True},
    {"name": "FNO", "family": "FNO", "with_oc": False},
    {"name": "FNO_OC", "family": "FNO", "with_oc": True},
]

_SMOKE_MODEL_SPECS = _FULL_MODEL_SPECS[:2]


def _env_flag(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    return int(raw)


def _resolve_run_all_configs(smoke_test: bool) -> bool:
    if not smoke_test:
        return True
    return bool(_env_flag(_EXP_ALL_CONFIGS_ENV))


def _resolve_max_epochs(smoke_test: bool) -> int:
    cli_override = _env_int(_EXP_EPOCHS_ENV)
    if cli_override is not None:
        return cli_override
    return 4 if smoke_test else 1000


def _selected_model_specs(smoke_test: bool, all_configs: bool) -> list[dict[str, Any]]:
    if smoke_test and not all_configs:
        return [dict(spec) for spec in _SMOKE_MODEL_SPECS]
    return [dict(spec) for spec in _FULL_MODEL_SPECS]


def _model_training_group(family: str) -> str:
    return "supervised_only" if family == "FNO" else "physics_trained"


def _oc_variant_label(with_oc: bool) -> str:
    return "with_oc" if with_oc else "without_oc"


def _legacy_tapinn_name(model_name: str) -> str:
    return "tapinn" if model_name == "StandardPINN_OC" else model_name


def _build_model_configs(
    obs_dim: int,
    obs_steps: int,
    output_dim: int,
    num_points: int,
    smoke_test: bool,
    model_specs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    branch_input_dim = obs_steps * obs_dim
    fno_width = 8 if smoke_test else 32
    fno_modes = 4 if smoke_test else 12
    oc_shared = {"state_dim": obs_dim, "lstm_hidden": 64, "latent_dim": 16}

    family_configs = {
        "StandardPINN": {
            False: {"coord_dim": 1, "output_dim": output_dim, "hidden_dim": 64},
            True: {"coord_dim": 1, "output_dim": output_dim, "hidden_dim": 64, **oc_shared},
        },
        "HyperPINN": {
            False: {"coord_dim": 1, "output_dim": output_dim, "hidden_dim": 64},
            True: {"coord_dim": 1, "output_dim": output_dim, "hidden_dim": 64, **oc_shared},
        },
        "HyperLRPINN": {
            False: {"coord_dim": 1, "output_dim": output_dim, "hidden_dim": 64, "rank": 4},
            True: {"coord_dim": 1, "output_dim": output_dim, "hidden_dim": 64, "rank": 4, **oc_shared},
        },
        "DeepONet": {
            False: {
                "branch_input_dim": branch_input_dim,
                "coord_dim": 1,
                "output_dim": output_dim,
                "hidden_dim": 64,
                "basis_dim": 32,
            },
            True: {"coord_dim": 1, "output_dim": output_dim, "hidden_dim": 64, "basis_dim": 32, **oc_shared},
        },
        "FNO": {
            False: {
                "branch_input_dim": branch_input_dim,
                "grid_size": num_points,
                "output_dim": output_dim,
                "width": fno_width,
                "modes": fno_modes,
            },
            True: {
                "grid_size": num_points,
                "output_dim": output_dim,
                "width": fno_width,
                "modes": fno_modes,
                **oc_shared,
            },
        },
    }

    built: dict[str, dict[str, Any]] = {}
    for spec in model_specs:
        family = str(spec["family"])
        with_oc = bool(spec["with_oc"])
        config = dict(family_configs[family][with_oc])
        model = create_model(family, with_oc=with_oc, **config)
        built[str(spec["name"])] = {
            "family": family,
            "with_oc": with_oc,
            "training_group": _model_training_group(family),
            "oc_variant": _oc_variant_label(with_oc),
            "config": config,
            "model": model,
        }
    return built


def _build_trajectory_ids(params: torch.Tensor) -> torch.Tensor:
    local_counts: dict[float, int] = {}
    labels: list[int] = []
    for value in params.detach().cpu().tolist():
        key = round(float(value), 8)
        labels.append(local_counts.get(key, 0))
        local_counts[key] = local_counts.get(key, 0) + 1
    return torch.tensor(labels, dtype=torch.long)


def _train_ode_model(
    model_name: str,
    spec: dict[str, Any],
    problem_name: str,
    train_obs: torch.Tensor,
    train_coords: torch.Tensor,
    train_targets: torch.Tensor,
    train_params: torch.Tensor,
    train_trajectory_ids: torch.Tensor,
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
    model = spec["model"]
    family = str(spec["family"])
    has_oc = hasattr(model, "observation_conditioner")
    desc = f"{model_name[:12]}-{problem_name[:6]}-s{seed}"

    if family == "FNO" and not has_oc:
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

    return train_direct_model(
        model,
        problem_name=problem_name,
        model_kind=family.lower(),
        observations=train_obs,
        coords=train_coords,
        targets=train_targets,
        params=train_params,
        device=device,
        ode_metadata=train_ode_metadata,
        epochs=max_epochs,
        batch_size=batch_size,
        progress_desc=desc,
        trajectory_ids=train_trajectory_ids,
        val_bundle=val_bundle,
        callbacks=callbacks,
        coord_normalizer=coord_normalizer,
        state_normalizer=state_normalizer,
    )


@torch.no_grad()
def _predict_oc_model_numpy(
    model: nn.Module,
    family: str,
    observations: torch.Tensor,
    coords: torch.Tensor,
    device: torch.device,
    state_normalizer: StateNormalizer | None = None,
) -> np.ndarray[Any, Any]:
    model_ref = cast(Any, model)
    model_ref.eval()
    obs = observations.to(device)
    coord_batch = coords.to(device)
    latent = model_ref.observation_conditioner(obs)
    if family == "FNO":
        pred_norm = model_ref.decode(coord_batch, latent)
    else:
        expanded_latent = latent.unsqueeze(1).expand(-1, coord_batch.shape[1], -1).reshape(-1, latent.shape[-1])
        pred_norm = model_ref.decode(coord_batch.reshape(-1, coord_batch.shape[-1]), expanded_latent)
        pred_norm = pred_norm.reshape(obs.shape[0], coord_batch.shape[1], -1)
    if state_normalizer is not None:
        pred_norm = state_normalizer.denormalize(pred_norm)
    return pred_norm.detach().cpu().numpy()


def _predict_ode_model(
    spec: dict[str, Any],
    test_obs: torch.Tensor,
    test_coords: torch.Tensor,
    test_params: torch.Tensor,
    num_points: int,
    device: torch.device,
    state_normalizer: StateNormalizer | None = None,
) -> tuple[np.ndarray[Any, Any], float]:
    model = spec["model"]
    family = str(spec["family"])
    has_oc = hasattr(model, "observation_conditioner")
    if family == "FNO" and not has_oc:
        return _measure_inference_ms(_fno_predict_numpy, test_obs.shape[0], model, test_obs, num_points, device)
    if has_oc:
        return _measure_inference_ms(
            _predict_oc_model_numpy,
            test_obs.shape[0],
            model,
            family,
            test_obs,
            test_coords,
            device,
            state_normalizer,
        )
    return _measure_inference_ms(
        lambda *a: predict_direct(*a, state_normalizer=state_normalizer).detach().cpu().numpy(),
        test_obs.shape[0],
        model,
        family.lower(),
        test_obs,
        test_coords,
        test_params,
        device,
    )


def _compute_disambiguation_metric(
    spec: dict[str, Any],
    observations: torch.Tensor,
    trajectory_ids: torch.Tensor,
    device: torch.device,
) -> float | None:
    model = cast(Any, spec["model"])
    if not hasattr(model, "observation_conditioner"):
        return None
    model.eval()
    with torch.no_grad():
        embeddings = model.observation_conditioner(observations.to(device)).detach().cpu().numpy()
    return compute_disambiguation_score(embeddings, trajectory_ids.detach().cpu().tolist())


def _run_ode_seed_all_models(
    problem_name: str,
    param_values: list[float],
    seed: int,
    smoke_test: bool,
    device: torch.device,
    max_epochs: int,
    callbacks: CallbackConfig,
    all_configs: bool,
) -> dict[str, dict[str, Any]]:
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
    trajectory_ids = _build_trajectory_ids(params)

    train_idx, val_idx, test_idx = _split_indices_three_way(len(params), seed)
    train_obs, train_coords, train_targets, train_params = _subset_tensors(train_idx, observations, coords, targets, params)
    val_obs, val_coords, val_targets, val_params = _subset_tensors(val_idx, observations, coords, targets, params)
    test_obs, test_coords, test_targets, test_params = _subset_tensors(test_idx, observations, coords, targets, params)
    train_trajectory_ids = _subset_tensors(train_idx, trajectory_ids)[0]
    test_trajectory_ids = _subset_tensors(test_idx, trajectory_ids)[0]
    train_ode_meta = _subset_optional_tensor(train_idx, ode_metadata)
    val_ode_meta = _subset_optional_tensor(val_idx, ode_metadata)
    test_ode_meta = _subset_optional_tensor(test_idx, ode_metadata)

    (
        coord_norm,
        state_norm,
        train_targets,
        val_targets,
        test_targets,
        train_obs,
        val_obs,
        test_obs,
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

    obs_dim = observations.shape[-1]
    output_dim = targets.shape[-1]
    model_specs = _selected_model_specs(smoke_test, all_configs)
    models = _build_model_configs(obs_dim, obs_steps, output_dim, num_points, smoke_test, model_specs)
    batch_size = 4

    results: dict[str, dict[str, Any]] = {}
    truth = state_norm.denormalize(test_targets).cpu().numpy()

    for model_name in tqdm(list(models.keys()), desc=f"{problem_name[:6]}/models", leave=False):
        spec = models[model_name]
        train_result = _train_ode_model(
            model_name,
            spec,
            problem_name,
            train_obs,
            train_coords,
            train_targets,
            train_params,
            train_trajectory_ids,
            train_ode_meta,
            active_val_bundle,
            active_callbacks,
            device,
            max_epochs,
            batch_size,
            seed,
            coord_normalizer=coord_norm,
            state_normalizer=state_norm,
        )
        predictions, inference_ms = _predict_ode_model(
            spec,
            test_obs,
            test_coords,
            test_params,
            num_points,
            device,
            state_normalizer=state_norm,
        )
        data_mse, physics_residual = _aggregate_ode_metrics(
            problem_name,
            data.times,
            predictions,
            truth,
            test_params,
            test_ode_meta,
        )
        relative_l2_error = compute_relative_l2_error(predictions, truth)
        disambiguation_score = _compute_disambiguation_metric(spec, test_obs, test_trajectory_ids, device)
        results[model_name] = {
            "family": spec["family"],
            "with_oc": spec["with_oc"],
            "comparison_group": spec["training_group"],
            "oc_variant": spec["oc_variant"],
            "data_mse": data_mse,
            "relative_l2_error": relative_l2_error,
            "physics_residual": physics_residual,
            "disambiguation_score": disambiguation_score,
            "param_count": count_parameters(spec["model"]),
            "epochs_trained": train_result.epochs_trained,
            "best_val_loss": train_result.best_val_loss,
            "seconds_per_epoch": train_result.seconds_per_epoch,
            "inference_ms": inference_ms,
            "prediction": predictions[0],
            "truth": truth[0],
            "param": float(test_params[0].item()),
            "times": data.times,
        }

    return results


def _multi_model_phase_plot(
    problem_name: str,
    model_results: dict[str, dict[str, Any]],
    title: str,
    path: Path,
) -> None:
    model_names = list(model_results.keys())
    ncols = min(3, len(model_names))
    nrows = math.ceil(len(model_names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for ax, model_name in zip(axes_flat, model_names):
        info = model_results[model_name]
        truth = info["truth"]
        pred = info["prediction"]
        if problem_name == "lorenz":
            ax.plot(truth[:, 0], truth[:, 2], label="GT", linewidth=1.6)
            ax.plot(pred[:, 0], pred[:, 2], label="Pred", linewidth=1.6, linestyle="--")
            ax.set_xlabel("x")
            ax.set_ylabel("z")
        elif problem_name == "kuramoto":
            ax.plot(np.unwrap(truth[:, 0]), np.unwrap(truth[:, 1]), label="GT", linewidth=1.6)
            ax.plot(np.unwrap(pred[:, 0]), np.unwrap(pred[:, 1]), label="Pred", linewidth=1.6, linestyle="--")
            ax.set_xlabel(r"$\theta_1$")
            ax.set_ylabel(r"$\theta_2$")
        else:
            ax.plot(truth[:, 0], truth[:, 1], label="GT", linewidth=1.6)
            ax.plot(pred[:, 0], pred[:, 1], label="Pred", linewidth=1.6, linestyle="--")
            ax.set_xlabel("x")
            ax.set_ylabel("v")
        ax.set_title(f"{model_name}\nrelL2={info['relative_l2_error']:.3g}")
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
    colors = []
    if comparison_groups:
        palette = {
            "physics_trained": "#4C72B0",
            "supervised_only": "#DD8452",
            "with_oc": "#55A868",
            "without_oc": "#C44E52",
        }
        colors = [palette.get(group, "grey") for group in comparison_groups]
    else:
        colors = ["#4C72B0"] * len(model_names)
    fig, ax = plt.subplots(figsize=(max(5.0, 1.5 * len(model_names)), 3.8))
    ax.bar(model_names, values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    save_figure(fig, path)


def _param_vs_residual_scatter(
    model_results_flat: list[dict[str, Any]],
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    palette = {"physics_trained": "#4C72B0", "supervised_only": "#DD8452"}
    for row in model_results_flat:
        color = palette.get(row.get("comparison_group", ""), "grey")
        ax.scatter(row["param_count"], row["physics_residual"], s=80, color=color)
        ax.annotate(row["model_name"], (row["param_count"], row["physics_residual"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Parameter Count")
    ax.set_ylabel("Physics Residual")
    ax.set_title(title)
    save_figure(fig, path)


def _build_oc_benefit_rows(problem_model_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_problem_family: dict[tuple[str, str], dict[bool, dict[str, Any]]] = {}
    for row in problem_model_summaries:
        key = (str(row["problem"]), str(row["family"]))
        by_problem_family.setdefault(key, {})[bool(row["with_oc"])] = row

    rows: list[dict[str, Any]] = []
    for (problem, family) in sorted(by_problem_family.keys(), key=lambda item: (_FAMILY_ORDER.index(item[1]), item[0])):
        pair = by_problem_family[(problem, family)]
        if False not in pair or True not in pair:
            continue
        no_oc = pair[False]
        with_oc = pair[True]
        rows.append(
            {
                "problem": problem,
                "family": family,
                "oc_model": with_oc["model"],
                "baseline_model": no_oc["model"],
                "relative_l2_error_delta": no_oc["relative_l2_error_mean"] - with_oc["relative_l2_error_mean"],
                "data_mse_delta": no_oc["data_mse_mean"] - with_oc["data_mse_mean"],
                "physics_residual_delta": no_oc["physics_residual_mean"] - with_oc["physics_residual_mean"],
                "oc_disambiguation_score_mean": with_oc["disambiguation_score_mean"],
            }
        )
    return rows


def run_exp_1_ode_chaos_suite(output_root: str, device_name: str, smoke_test: bool, seed: int) -> dict[str, object]:
    configure_plotting()
    device = get_best_device(device_name)
    run_dir = prepare_run_dir(output_root, "exp_1_ode_chaos_suite")
    seeds = [seed] if smoke_test else [seed + offset for offset in range(5)]
    all_configs = _resolve_run_all_configs(smoke_test)
    model_specs = _selected_model_specs(smoke_test, all_configs)
    model_names = [str(spec["name"]) for spec in model_specs]

    configs = {
        "duffing": [0.24, 0.38, 0.52] if smoke_test else np.linspace(0.2, 0.55, 7).tolist(),
        "lorenz": [20.0, 24.74, 32.0] if smoke_test else np.linspace(18.0, 35.0, 7).tolist(),
        "kuramoto": [0.2, 1.0, 2.0] if smoke_test else np.linspace(0.2, 3.0, 7).tolist(),
    }
    max_epochs = _resolve_max_epochs(smoke_test)
    callbacks = CallbackConfig(
        early_stopping_patience=2 if smoke_test else 50,
        reduce_lr_patience=1 if smoke_test else 20,
        reduce_lr_factor=0.5,
        min_lr=1e-6,
    )

    seed_rows: list[dict[str, Any]] = []
    problem_model_summaries: list[dict[str, Any]] = []
    phase_data: dict[str, dict[str, dict[str, Any]]] = {}

    for problem_name, param_values in tqdm(configs.items(), desc="ODE systems", leave=False):
        per_model_seed_data: dict[str, list[dict[str, Any]]] = {name: [] for name in model_names}

        for local_seed in tqdm(seeds, desc=f"{problem_name} seeds", leave=False):
            set_global_seed(local_seed)
            results = _run_ode_seed_all_models(
                problem_name,
                param_values,
                local_seed,
                smoke_test,
                device,
                max_epochs,
                callbacks,
                all_configs,
            )
            if local_seed == seeds[0]:
                phase_data[problem_name] = results

            for model_name, info in results.items():
                per_model_seed_data[model_name].append(info)
                seed_rows.append(
                    {
                        "problem": problem_name,
                        "seed": local_seed,
                        "model": model_name,
                        "family": info["family"],
                        "with_oc": info["with_oc"],
                        "comparison_group": info["comparison_group"],
                        "oc_variant": info["oc_variant"],
                        "data_mse": info["data_mse"],
                        "relative_l2_error": info["relative_l2_error"],
                        "physics_residual": info["physics_residual"],
                        "disambiguation_score": info["disambiguation_score"],
                        "param_count": info["param_count"],
                        "epochs_trained": info["epochs_trained"],
                        "best_val_loss": info["best_val_loss"],
                        "seconds_per_epoch": info["seconds_per_epoch"],
                        "inference_ms": info["inference_ms"],
                    }
                )

        for model_name in model_names:
            rows = per_model_seed_data[model_name]
            if not rows:
                continue
            data_mean, data_std = mean_std(row["data_mse"] for row in rows)
            rel_mean, rel_std = mean_std(row["relative_l2_error"] for row in rows)
            phys_mean, phys_std = mean_std(row["physics_residual"] for row in rows)
            ep_mean, ep_std = mean_std(row["epochs_trained"] for row in rows)
            spe_mean, _ = mean_std(row["seconds_per_epoch"] for row in rows)
            inf_mean, _ = mean_std(row["inference_ms"] for row in rows)
            dis_vals = [float(row["disambiguation_score"]) for row in rows if row["disambiguation_score"] is not None]
            dis_mean, dis_std = mean_std(dis_vals) if dis_vals else (0.0, 0.0)
            problem_model_summaries.append(
                {
                    "problem": problem_name,
                    "model": model_name,
                    "family": rows[0]["family"],
                    "with_oc": rows[0]["with_oc"],
                    "comparison_group": rows[0]["comparison_group"],
                    "oc_variant": rows[0]["oc_variant"],
                    "param_count": rows[0]["param_count"],
                    "data_mse_mean": data_mean,
                    "data_mse_std": data_std,
                    "relative_l2_error_mean": rel_mean,
                    "relative_l2_error_std": rel_std,
                    "physics_residual_mean": phys_mean,
                    "physics_residual_std": phys_std,
                    "disambiguation_score_mean": dis_mean if rows[0]["with_oc"] else None,
                    "disambiguation_score_std": dis_std if rows[0]["with_oc"] else None,
                    "epochs_trained_mean": ep_mean,
                    "epochs_trained_std": ep_std,
                    "seconds_per_epoch_mean": spe_mean,
                    "inference_ms_mean": inf_mean,
                }
            )

        _multi_model_phase_plot(
            problem_name,
            phase_data[problem_name],
            f"{problem_name.title()} — Model Comparison (seed {seeds[0]})",
            run_dir / "figures" / f"{problem_name}_model_comparison.pdf",
        )
        tapinn_info = phase_data[problem_name].get("StandardPINN_OC", {})
        if tapinn_info:
            _phase_plot(
                problem_name,
                tapinn_info["truth"],
                tapinn_info["prediction"],
                f"{problem_name.title()} Phase-Space — StandardPINN_OC",
                run_dir / "figures" / f"{problem_name}_phase_space.pdf",
            )

        prob_rows = [row for row in problem_model_summaries if row["problem"] == problem_name]
        _metrics_bar_chart(
            [row["model"] for row in prob_rows],
            [row["data_mse_mean"] for row in prob_rows],
            "Data MSE",
            f"{problem_name.title()} — Data MSE by Model",
            run_dir / "figures" / f"{problem_name}_data_mse_bar.pdf",
            comparison_groups=[row["comparison_group"] for row in prob_rows],
        )
        _metrics_bar_chart(
            [row["model"] for row in prob_rows],
            [row["relative_l2_error_mean"] for row in prob_rows],
            "Relative L2 Error",
            f"{problem_name.title()} — Relative L2 by Model",
            run_dir / "figures" / f"{problem_name}_relative_l2_bar.pdf",
            comparison_groups=[row["oc_variant"] for row in prob_rows],
        )
        _metrics_bar_chart(
            [row["model"] for row in prob_rows],
            [row["physics_residual_mean"] for row in prob_rows],
            "Physics Residual",
            f"{problem_name.title()} — Physics Residual by Model",
            run_dir / "figures" / f"{problem_name}_physics_residual_bar.pdf",
            comparison_groups=[row["comparison_group"] for row in prob_rows],
        )

    duffing_rows = [row for row in problem_model_summaries if row["problem"] == "duffing"]
    if duffing_rows:
        _param_vs_residual_scatter(
            [
                {
                    "model_name": row["model"],
                    "param_count": row["param_count"],
                    "physics_residual": row["physics_residual_mean"],
                    "comparison_group": row["comparison_group"],
                }
                for row in duffing_rows
            ],
            "Duffing — Parameter Count vs Physics Residual",
            run_dir / "figures" / "duffing_pareto_params_vs_residual.pdf",
        )

    oc_benefit_rows = _build_oc_benefit_rows(problem_model_summaries)
    tapinn_summaries = [
        {
            "problem": row["problem"],
            "data_mse_mean": row["data_mse_mean"],
            "data_mse_std": row["data_mse_std"],
            "physics_residual_mean": row["physics_residual_mean"],
            "physics_residual_std": row["physics_residual_std"],
            "relative_l2_error_mean": row["relative_l2_error_mean"],
            "relative_l2_error_std": row["relative_l2_error_std"],
        }
        for row in problem_model_summaries
        if row["model"] == "StandardPINN_OC"
    ]

    write_csv(run_dir / "tables" / "seed_metrics.csv", seed_rows)
    write_csv(run_dir / "tables" / "model_summary.csv", problem_model_summaries)
    write_csv(run_dir / "tables" / "oc_benefit_summary.csv", oc_benefit_rows)
    write_csv(run_dir / "tables" / "summary_table.csv", tapinn_summaries)

    payload: dict[str, object] = {
        "device": str(device),
        "smoke_test": smoke_test,
        "all_configs": all_configs,
        "max_epochs": max_epochs,
        "callbacks": {
            "early_stopping_patience": callbacks.early_stopping_patience,
            "reduce_lr_patience": callbacks.reduce_lr_patience,
            "reduce_lr_factor": callbacks.reduce_lr_factor,
            "min_lr": callbacks.min_lr,
        },
        "models": model_names,
        "legacy_models": [_legacy_tapinn_name(name) for name in model_names],
        "summary": problem_model_summaries,
        "oc_benefit": oc_benefit_rows,
    }
    write_json(run_dir / "results.json", payload)
    return payload
