from __future__ import annotations
# pyright: reportArgumentType=false, reportOptionalMemberAccess=false, reportCallIssue=false

import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from ..device import get_best_device
from ..io_utils import prepare_run_dir, write_csv, write_json
from ..metrics import compute_disambiguation_score, compute_relative_l2_error, mean_std
from ..models import count_parameters, create_model
from ..plotting import configure_plotting
from ..problems import generate_allen_cahn_dataset, generate_duffing_dataset
from ..repro import set_global_seed
from ..trainers import (
    CallbackConfig,
    CoordNormalizer,
    StateNormalizer,
    ValBundle,
    predict_direct,
    prepare_ode_tensors,
    prepare_pde_tensors,
    refit_normalizers_on_physical_split,
    train_direct_model,
    train_fno_model,
)
from .common import (
    tqdm,
    _aggregate_ode_metrics,
    _aggregate_pde_metrics,
    _fno_predict_numpy,
    _measure_inference_ms,
    _multi_line_plot,
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


def _build_model_configs(
    obs_dim: int,
    obs_steps: int,
    output_dim: int,
    num_points: int,
    coord_dim: int,
    smoke_test: bool,
    model_specs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    branch_input_dim = obs_steps * obs_dim
    fno_width = 8 if smoke_test else 32
    fno_modes = 4 if smoke_test else 12
    oc_shared = {"state_dim": obs_dim, "lstm_hidden": 64, "latent_dim": 16}

    family_configs = {
        "StandardPINN": {
            False: {"coord_dim": coord_dim, "output_dim": output_dim, "hidden_dim": 64},
            True: {"coord_dim": coord_dim, "output_dim": output_dim, "hidden_dim": 64, **oc_shared},
        },
        "HyperPINN": {
            False: {"coord_dim": coord_dim, "output_dim": output_dim, "hidden_dim": 64},
            True: {"coord_dim": coord_dim, "output_dim": output_dim, "hidden_dim": 64, **oc_shared},
        },
        "HyperLRPINN": {
            False: {"coord_dim": coord_dim, "output_dim": output_dim, "hidden_dim": 64, "rank": 4},
            True: {"coord_dim": coord_dim, "output_dim": output_dim, "hidden_dim": 64, "rank": 4, **oc_shared},
        },
        "DeepONet": {
            False: {
                "branch_input_dim": branch_input_dim,
                "coord_dim": coord_dim,
                "output_dim": output_dim,
                "hidden_dim": 64,
                "basis_dim": 32,
            },
            True: {"coord_dim": coord_dim, "output_dim": output_dim, "hidden_dim": 64, "basis_dim": 32, **oc_shared},
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
        built[str(spec["name"])] = {
            "family": family,
            "with_oc": with_oc,
            "training_group": _model_training_group(family),
            "oc_variant": _oc_variant_label(with_oc),
            "config": config,
            "model": create_model(family, with_oc=with_oc, **config),
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


def _build_fno_grid_like(coords: torch.Tensor) -> torch.Tensor:
    point_count = coords.shape[1]
    grid = torch.linspace(0.0, 1.0, point_count, dtype=coords.dtype, device=coords.device)
    return grid.unsqueeze(0).unsqueeze(-1).expand(coords.shape[0], -1, -1).clone()


def _train_sensitivity_model(
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
    max_phys_points: int = 64,
):
    model = spec["model"]
    family = str(spec["family"])
    has_oc = hasattr(model, "observation_conditioner")
    desc = f"{model_name[:12]}-{problem_name[:8]}-s{seed}"

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

    active_train_coords = train_coords
    active_val_bundle = val_bundle
    if family == "FNO" and has_oc and train_coords.shape[-1] > 1:
        active_train_coords = _build_fno_grid_like(train_coords)
        if val_bundle is not None:
            active_val_bundle = ValBundle(
                observations=val_bundle.observations,
                coords=_build_fno_grid_like(val_bundle.coords),
                targets=val_bundle.targets,
                params=val_bundle.params,
                ode_metadata=val_bundle.ode_metadata,
            )

    return train_direct_model(
        model,
        problem_name=problem_name,
        model_kind=family.lower(),
        observations=train_obs,
        coords=active_train_coords,
        targets=train_targets,
        params=train_params,
        device=device,
        ode_metadata=train_ode_metadata,
        epochs=max_epochs,
        batch_size=batch_size,
        progress_desc=desc,
        trajectory_ids=train_trajectory_ids,
        val_bundle=active_val_bundle,
        callbacks=callbacks,
        coord_normalizer=coord_normalizer,
        state_normalizer=state_normalizer,
        max_phys_points=max_phys_points,
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
        if coord_batch.shape[-1] > 1:
            coord_batch = _build_fno_grid_like(coord_batch)
        pred_norm = model_ref.decode(coord_batch, latent)
    else:
        expanded_latent = latent.unsqueeze(1).expand(-1, coord_batch.shape[1], -1).reshape(-1, latent.shape[-1])
        pred_norm = model_ref.decode(coord_batch.reshape(-1, coord_batch.shape[-1]), expanded_latent)
        pred_norm = pred_norm.reshape(obs.shape[0], coord_batch.shape[1], -1)
    if state_normalizer is not None:
        pred_norm = state_normalizer.denormalize(pred_norm)
    return pred_norm.detach().cpu().numpy()


def _predict_model(
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


def _apply_observation_noise(observations: torch.Tensor, noise_sigma: float, reference_std: float) -> torch.Tensor:
    if noise_sigma <= 0.0:
        return observations.clone()
    return observations + noise_sigma * reference_std * torch.randn_like(observations)


def _compute_forecast_error(
    predictions: np.ndarray[Any, Any],
    truth: np.ndarray[Any, Any],
    observed_steps: int,
    problem_kind: str,
    obs_dim: int,
) -> float:
    observed_points = observed_steps if problem_kind == "ode" else observed_steps * obs_dim
    pred = np.asarray(predictions)
    target = np.asarray(truth)
    if observed_points >= target.shape[1]:
        return float(np.mean((pred - target) ** 2))
    return float(np.mean((pred[:, observed_points:, :] - target[:, observed_points:, :]) ** 2))


def _generate_problem_data(problem_name: str, smoke_test: bool, seed: int):
    if problem_name == "duffing":
        return generate_duffing_dataset(
            [0.24, 0.38, 0.52] if smoke_test else np.linspace(0.2, 0.55, 7).tolist(),
            num_trajectories=1 if smoke_test else 100,
            num_points=48 if smoke_test else 200,
            t_span=(0.0, 16.0),
            seed=seed,
        )
    if problem_name == "allen_cahn":
        nx = 20 if smoke_test else 64
        nt = 14 if smoke_test else 48
        return generate_allen_cahn_dataset(
            [0.8, 1.1] if smoke_test else [0.75, 0.9, 1.05, 1.2],
            num_samples=1 if smoke_test else 24,
            nx=nx,
            nt=nt,
            seed=seed,
        )
    raise ValueError(problem_name)


def _default_observation_steps(problem_name: str, smoke_test: bool, data: Any) -> int:
    if problem_name == "duffing":
        return 8 if smoke_test else max(24, int(0.15 * data.states.shape[1]))
    if problem_name == "allen_cahn":
        return 4 if smoke_test else 8
    raise ValueError(problem_name)


def _prepare_problem_bundle(problem_name: str, smoke_test: bool, seed: int, observation_steps: int) -> dict[str, Any]:
    data = _generate_problem_data(problem_name, smoke_test, seed)
    if problem_name == "duffing":
        observations, coords, targets, params, ode_metadata, coord_norm, state_norm = prepare_ode_tensors(
            data,
            observation_steps=observation_steps,
        )
        problem_kind = "ode"
        boundary = None
    else:
        observations, coords, targets, params, coord_norm, state_norm = prepare_pde_tensors(
            data,
            observation_steps=observation_steps,
        )
        ode_metadata = None
        problem_kind = "pde"
        boundary = str(data.metadata.get("boundary", "periodic")) if data.metadata else "periodic"

    trajectory_ids = _build_trajectory_ids(params)
    train_idx, val_idx, test_idx = _split_indices_three_way(len(params), seed)
    train_obs, train_coords, train_targets, train_params = _subset_tensors(train_idx, observations, coords, targets, params)
    val_obs, val_coords, val_targets, val_params = _subset_tensors(val_idx, observations, coords, targets, params)
    test_obs, test_coords, test_targets, test_params = _subset_tensors(test_idx, observations, coords, targets, params)
    train_trajectory_ids = _subset_tensors(train_idx, trajectory_ids)[0]
    test_trajectory_ids = _subset_tensors(test_idx, trajectory_ids)[0]
    train_ode_metadata = _subset_optional_tensor(train_idx, ode_metadata)
    val_ode_metadata = _subset_optional_tensor(val_idx, ode_metadata)
    test_ode_metadata = _subset_optional_tensor(test_idx, ode_metadata)

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

    return {
        "problem_name": problem_name,
        "problem_kind": problem_kind,
        "data": data,
        "boundary": boundary,
        "observation_steps": observation_steps,
        "obs_dim": int(observations.shape[-1]),
        "coord_dim": int(coords.shape[-1]),
        "output_dim": int(targets.shape[-1]),
        "num_points": int(targets.shape[1]),
        "train_obs": train_obs,
        "train_coords": train_coords,
        "train_targets": train_targets,
        "train_params": train_params,
        "train_trajectory_ids": train_trajectory_ids,
        "train_ode_metadata": train_ode_metadata,
        "val_obs": val_obs,
        "val_coords": val_coords,
        "val_targets": val_targets,
        "val_params": val_params,
        "val_ode_metadata": val_ode_metadata,
        "test_obs": test_obs,
        "test_coords": test_coords,
        "test_targets": test_targets,
        "test_params": test_params,
        "test_trajectory_ids": test_trajectory_ids,
        "test_ode_metadata": test_ode_metadata,
        "coord_normalizer": coord_norm,
        "state_normalizer": state_norm,
    }


def _run_problem_condition(
    problem_name: str,
    seed: int,
    smoke_test: bool,
    device: torch.device,
    max_epochs: int,
    callbacks: CallbackConfig,
    model_specs: list[dict[str, Any]],
    noise_sigma: float,
    observation_steps: int,
) -> dict[str, dict[str, Any]]:
    bundle = _prepare_problem_bundle(problem_name, smoke_test, seed, observation_steps)
    train_obs = cast(torch.Tensor, bundle["train_obs"])
    val_obs = cast(torch.Tensor, bundle["val_obs"])
    test_obs = cast(torch.Tensor, bundle["test_obs"])
    reference_std = float(train_obs.std().item()) if train_obs.numel() else 1.0
    noisy_train_obs = _apply_observation_noise(train_obs, noise_sigma, reference_std)
    noisy_val_obs = _apply_observation_noise(val_obs, noise_sigma, reference_std)
    noisy_test_obs = _apply_observation_noise(test_obs, noise_sigma, reference_std)

    val_bundle = ValBundle(
        observations=noisy_val_obs,
        coords=cast(torch.Tensor, bundle["val_coords"]),
        targets=cast(torch.Tensor, bundle["val_targets"]),
        params=cast(torch.Tensor, bundle["val_params"]),
        ode_metadata=cast(torch.Tensor | None, bundle["val_ode_metadata"]),
    )
    active_val_bundle: ValBundle | None = val_bundle if noisy_val_obs.shape[0] > 0 else None
    active_callbacks: CallbackConfig | None = callbacks if noisy_val_obs.shape[0] > 0 else None

    models = _build_model_configs(
        obs_dim=int(bundle["obs_dim"]),
        obs_steps=observation_steps,
        output_dim=int(bundle["output_dim"]),
        num_points=int(bundle["num_points"]),
        coord_dim=int(bundle["coord_dim"]),
        smoke_test=smoke_test,
        model_specs=model_specs,
    )
    batch_size = 4
    truth = cast(StateNormalizer, bundle["state_normalizer"]).denormalize(cast(torch.Tensor, bundle["test_targets"])).cpu().numpy()
    data = bundle["data"]

    results: dict[str, dict[str, Any]] = {}
    for model_name in tqdm(list(models.keys()), desc=f"{problem_name[:8]}/models", leave=False):
        spec = models[model_name]
        train_result = _train_sensitivity_model(
            model_name,
            spec,
            problem_name,
            noisy_train_obs,
            cast(torch.Tensor, bundle["train_coords"]),
            cast(torch.Tensor, bundle["train_targets"]),
            cast(torch.Tensor, bundle["train_params"]),
            cast(torch.Tensor, bundle["train_trajectory_ids"]),
            cast(torch.Tensor | None, bundle["train_ode_metadata"]),
            active_val_bundle,
            active_callbacks,
            device,
            max_epochs,
            batch_size,
            seed,
            coord_normalizer=cast(CoordNormalizer, bundle["coord_normalizer"]),
            state_normalizer=cast(StateNormalizer, bundle["state_normalizer"]),
            max_phys_points=32 if smoke_test else 64,
        )
        predictions, inference_ms = _predict_model(
            spec,
            noisy_test_obs,
            cast(torch.Tensor, bundle["test_coords"]),
            cast(torch.Tensor, bundle["test_params"]),
            int(bundle["num_points"]),
            device,
            state_normalizer=cast(StateNormalizer | None, bundle["state_normalizer"]),
        )

        if bundle["problem_kind"] == "ode":
            data_mse, physics_residual = _aggregate_ode_metrics(
                problem_name,
                data.times,
                predictions,
                truth,
                cast(torch.Tensor, bundle["test_params"]),
                cast(torch.Tensor | None, bundle["test_ode_metadata"]),
            )
        else:
            data_mse, physics_residual = _aggregate_pde_metrics(
                problem_name,
                data.times,
                data.space,
                predictions,
                truth,
                cast(torch.Tensor, bundle["test_params"]),
                str(bundle["boundary"]),
            )

        relative_l2_error = compute_relative_l2_error(predictions, truth)
        forecast_error = _compute_forecast_error(
            predictions,
            truth,
            observation_steps,
            str(bundle["problem_kind"]),
            int(bundle["obs_dim"]),
        )
        disambiguation_score = _compute_disambiguation_metric(spec, noisy_test_obs, cast(torch.Tensor, bundle["test_trajectory_ids"]), device)
        results[model_name] = {
            "problem": problem_name,
            "family": spec["family"],
            "with_oc": spec["with_oc"],
            "comparison_group": spec["training_group"],
            "oc_variant": spec["oc_variant"],
            "noise_sigma": noise_sigma,
            "observation_steps": observation_steps,
            "data_mse": data_mse,
            "relative_l2_error": relative_l2_error,
            "physics_residual": physics_residual,
            "forecast_error": forecast_error,
            "disambiguation_score": disambiguation_score,
            "param_count": count_parameters(spec["model"]),
            "epochs_trained": train_result.epochs_trained,
            "best_val_loss": train_result.best_val_loss,
            "seconds_per_epoch": train_result.seconds_per_epoch,
            "inference_ms": inference_ms,
        }
    return results


def _aggregate_condition_rows(
    rows: list[dict[str, Any]],
    problem_name: str,
    model_specs: list[dict[str, Any]],
    control_key: str,
    control_value: float,
    observation_steps: int | None = None,
) -> list[dict[str, Any]]:
    aggregated: list[dict[str, Any]] = []
    model_order = [str(spec["name"]) for spec in model_specs]
    for model_name in model_order:
        model_rows = [row for row in rows if row["model"] == model_name]
        if not model_rows:
            continue
        data_mean, data_std = mean_std(row["data_mse"] for row in model_rows)
        rel_mean, rel_std = mean_std(row["relative_l2_error"] for row in model_rows)
        phys_mean, phys_std = mean_std(row["physics_residual"] for row in model_rows)
        forecast_mean, forecast_std = mean_std(row["forecast_error"] for row in model_rows)
        epochs_mean, epochs_std = mean_std(row["epochs_trained"] for row in model_rows)
        seconds_mean, _ = mean_std(row["seconds_per_epoch"] for row in model_rows)
        inference_mean, _ = mean_std(row["inference_ms"] for row in model_rows)
        dis_vals = [float(row["disambiguation_score"]) for row in model_rows if row["disambiguation_score"] is not None]
        dis_mean, dis_std = mean_std(dis_vals) if dis_vals else (0.0, 0.0)
        first = model_rows[0]
        aggregated_row = {
            "problem": problem_name,
            "model": model_name,
            "family": first["family"],
            "with_oc": first["with_oc"],
            "comparison_group": first["comparison_group"],
            "oc_variant": first["oc_variant"],
            control_key: control_value,
            "data_mse_mean": data_mean,
            "data_mse_std": data_std,
            "relative_l2_error_mean": rel_mean,
            "relative_l2_error_std": rel_std,
            "physics_residual_mean": phys_mean,
            "physics_residual_std": phys_std,
            "forecast_error_mean": forecast_mean,
            "forecast_error_std": forecast_std,
            "disambiguation_score_mean": dis_mean if first["with_oc"] else None,
            "disambiguation_score_std": dis_std if first["with_oc"] else None,
            "param_count": first["param_count"],
            "epochs_trained_mean": epochs_mean,
            "epochs_trained_std": epochs_std,
            "seconds_per_epoch_mean": seconds_mean,
            "inference_ms_mean": inference_mean,
        }
        if observation_steps is not None:
            aggregated_row["observation_steps"] = observation_steps
        aggregated.append(aggregated_row)
    return aggregated


def _build_oc_noise_benefit_rows(noise_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, str], dict[bool, dict[str, Any]]] = {}
    for row in noise_rows:
        key = (str(row["problem"]), float(row["noise_sigma"]), str(row["family"]))
        grouped.setdefault(key, {})[bool(row["with_oc"])] = row

    rows: list[dict[str, Any]] = []
    for problem, noise_sigma, family in sorted(grouped.keys(), key=lambda item: (item[0], item[1], _FAMILY_ORDER.index(item[2]))):
        pair = grouped[(problem, noise_sigma, family)]
        if False not in pair or True not in pair:
            continue
        baseline = pair[False]
        oc_row = pair[True]
        rows.append(
            {
                "problem": problem,
                "noise_sigma": noise_sigma,
                "family": family,
                "baseline_model": baseline["model"],
                "oc_model": oc_row["model"],
                "relative_l2_error_delta": baseline["relative_l2_error_mean"] - oc_row["relative_l2_error_mean"],
                "data_mse_delta": baseline["data_mse_mean"] - oc_row["data_mse_mean"],
                "physics_residual_delta": baseline["physics_residual_mean"] - oc_row["physics_residual_mean"],
                "forecast_error_delta": baseline["forecast_error_mean"] - oc_row["forecast_error_mean"],
                "oc_disambiguation_score_mean": oc_row["disambiguation_score_mean"],
            }
        )
    return rows


def run_exp_4_sensitivity_and_robustness(output_root: str, device_name: str, smoke_test: bool, seed: int) -> dict[str, object]:
    configure_plotting()
    device = get_best_device(device_name)
    run_dir = prepare_run_dir(output_root, "exp_4_sensitivity_and_robustness")
    seeds = [seed] if smoke_test else [seed + offset for offset in range(5)]
    all_configs = _resolve_run_all_configs(smoke_test)
    model_specs = _selected_model_specs(smoke_test, all_configs)
    model_names = [str(spec["name"]) for spec in model_specs]
    max_epochs = _resolve_max_epochs(smoke_test)
    callbacks = CallbackConfig(
        early_stopping_patience=2 if smoke_test else 50,
        reduce_lr_patience=1 if smoke_test else 20,
        reduce_lr_factor=0.5,
        min_lr=1e-6,
    )

    noise_levels = [0.0, 0.1, 0.25] if smoke_test else [0.0, 0.05, 0.1, 0.25, 0.5]
    window_fractions = [0.1, 0.2] if smoke_test else [0.05, 0.1, 0.15, 0.2, 0.3]
    problem_names = ["duffing", "allen_cahn"]

    noise_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    for problem_name in tqdm(problem_names, desc="Noise sweep", leave=False):
        preview_data = _generate_problem_data(problem_name, smoke_test, seed)
        default_obs_steps = _default_observation_steps(problem_name, smoke_test, preview_data)
        for noise_sigma in tqdm(noise_levels, desc=f"{problem_name[:8]} noise", leave=False):
            seed_rows: list[dict[str, Any]] = []
            for local_seed in tqdm(seeds, desc=f"{problem_name[:8]} n={noise_sigma}", leave=False):
                set_global_seed(local_seed)
                results = _run_problem_condition(
                    problem_name,
                    local_seed,
                    smoke_test,
                    device,
                    max_epochs,
                    callbacks,
                    model_specs,
                    noise_sigma=noise_sigma,
                    observation_steps=default_obs_steps,
                )
                for model_name, info in results.items():
                    seed_rows.append({"model": model_name, **info})
            noise_rows.extend(
                _aggregate_condition_rows(
                    seed_rows,
                    problem_name,
                    model_specs,
                    control_key="noise_sigma",
                    control_value=float(noise_sigma),
                    observation_steps=default_obs_steps,
                )
            )

    for problem_name in tqdm(problem_names, desc="Window sweep", leave=False):
        preview_data = _generate_problem_data(problem_name, smoke_test, seed)
        total_steps = int(len(preview_data.times))
        for window_fraction in tqdm(window_fractions, desc=f"{problem_name[:8]} window", leave=False):
            observation_steps = max(2, int(round(window_fraction * total_steps)))
            seed_rows: list[dict[str, Any]] = []
            for local_seed in tqdm(seeds, desc=f"{problem_name[:8]} w={window_fraction}", leave=False):
                set_global_seed(local_seed)
                results = _run_problem_condition(
                    problem_name,
                    local_seed,
                    smoke_test,
                    device,
                    max_epochs,
                    callbacks,
                    model_specs,
                    noise_sigma=0.0,
                    observation_steps=observation_steps,
                )
                for model_name, info in results.items():
                    seed_rows.append({"model": model_name, **info})
            window_rows.extend(
                _aggregate_condition_rows(
                    seed_rows,
                    problem_name,
                    model_specs,
                    control_key="window_fraction",
                    control_value=float(window_fraction),
                    observation_steps=observation_steps,
                )
            )

    oc_noise_benefit_rows = _build_oc_noise_benefit_rows(noise_rows)

    write_csv(Path(run_dir) / "tables" / "noise_sweep.csv", noise_rows)
    write_csv(Path(run_dir) / "tables" / "window_sweep.csv", window_rows)
    write_csv(Path(run_dir) / "tables" / "oc_noise_benefit.csv", oc_noise_benefit_rows)

    for problem_name in problem_names:
        _multi_line_plot(
            [row for row in noise_rows if row["problem"] == problem_name],
            "noise_sigma",
            "forecast_error_mean",
            "model",
            f"{problem_name.replace('_', ' ').title()} Noise Robustness",
            Path(run_dir) / "figures" / f"{problem_name}_noise_robustness.pdf",
        )
        _multi_line_plot(
            [row for row in window_rows if row["problem"] == problem_name],
            "window_fraction",
            "forecast_error_mean",
            "model",
            f"{problem_name.replace('_', ' ').title()} Window Sensitivity",
            Path(run_dir) / "figures" / f"{problem_name}_window_sensitivity.pdf",
        )

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
        "noise_sweep": noise_rows,
        "window_sweep": window_rows,
        "oc_noise_benefit": oc_noise_benefit_rows,
    }
    write_json(Path(run_dir) / "results.json", payload)
    return payload
