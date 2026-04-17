from __future__ import annotations
# pyright: reportArgumentType=false, reportOptionalMemberAccess=false, reportCallIssue=false

import os
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from ..device import get_best_device
from ..io_utils import prepare_run_dir, write_csv, write_json
from ..metrics import compute_disambiguation_score, compute_relative_l2_error, mean_std, mse
from ..models import count_parameters, create_model
from ..plotting import configure_plotting, save_figure
from ..problems import generate_allen_cahn_dataset, generate_duffing_dataset
from ..repro import set_global_seed
from ..trainers import (
    CallbackConfig,
    CoordNormalizer,
    StateNormalizer,
    ValBundle,
    predict_direct,
    predict_fno,
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
    _measure_inference_ms,
    _save_plot_data,
    _split_indices_three_way,
    _subset_optional_tensor,
    _subset_tensors,
    _tapinn_predict_numpy,
)
from .exp1_ode_chaos import _metrics_bar_chart


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

_LEGACY_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "tapinn": {"family": "StandardPINN", "with_oc": True, "legacy_predict": "tapinn"},
    "tapinn_large": {
        "family": "StandardPINN",
        "with_oc": True,
        "legacy_predict": "tapinn",
        "config_overrides": {"hidden_dim": 128, "lstm_hidden": 128, "latent_dim": 32},
    },
    "hyperpinn": {"family": "HyperPINN", "with_oc": False},
    "hyper_lr_pinn": {"family": "HyperLRPINN", "with_oc": False},
    "deeponet": {"family": "DeepONet", "with_oc": False},
    "fno": {"family": "FNO", "with_oc": False},
}

_DUPLICATE_PARAM_REJECT_MODELS = {"hyperpinn", "hyper_lr_pinn"}


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
    return 4 if smoke_test else 100


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


def _build_family_configs(
    observation_feature_dim: int,
    obs_steps: int,
    coord_dim: int,
    output_dim: int,
    num_points: int,
    smoke_test: bool,
) -> dict[str, dict[bool, dict[str, Any]]]:
    branch_input_dim = obs_steps * observation_feature_dim
    fno_width = 8 if smoke_test else 32
    fno_modes = 4 if smoke_test else 12
    oc_shared = {"state_dim": observation_feature_dim, "lstm_hidden": 64, "latent_dim": 16}

    return {
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


def _build_named_model_spec(
    model_name: str,
    observation_feature_dim: int,
    obs_steps: int,
    coord_dim: int,
    output_dim: int,
    num_points: int,
    smoke_test: bool,
) -> dict[str, Any]:
    family_configs = _build_family_configs(
        observation_feature_dim=observation_feature_dim,
        obs_steps=obs_steps,
        coord_dim=coord_dim,
        output_dim=output_dim,
        num_points=num_points,
        smoke_test=smoke_test,
    )
    legacy_spec = _LEGACY_MODEL_SPECS.get(model_name)
    if legacy_spec is not None:
        family = str(legacy_spec["family"])
        with_oc = bool(legacy_spec["with_oc"])
        config = dict(family_configs[family][with_oc])
        config.update(cast(dict[str, Any], legacy_spec.get("config_overrides", {})))
        return {
            "name": model_name,
            "family": family,
            "with_oc": with_oc,
            "training_group": _model_training_group(family),
            "oc_variant": _oc_variant_label(with_oc),
            "config": config,
            "legacy_predict": legacy_spec.get("legacy_predict"),
            "model": create_model(family, with_oc=with_oc, **config),
        }

    for spec in _FULL_MODEL_SPECS:
        if str(spec["name"]) != model_name:
            continue
        family = str(spec["family"])
        with_oc = bool(spec["with_oc"])
        config = dict(family_configs[family][with_oc])
        return {
            "name": model_name,
            "family": family,
            "with_oc": with_oc,
            "training_group": _model_training_group(family),
            "oc_variant": _oc_variant_label(with_oc),
            "config": config,
            "legacy_predict": None,
            "model": create_model(family, with_oc=with_oc, **config),
        }

    raise ValueError(f"Unknown Exp3 model name: {model_name}")


def _build_model_configs(
    observation_feature_dim: int,
    obs_steps: int,
    coord_dim: int,
    output_dim: int,
    num_points: int,
    smoke_test: bool,
    model_specs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    built: dict[str, dict[str, Any]] = {}
    for spec in model_specs:
        name = str(spec["name"])
        built[name] = _build_named_model_spec(
            name,
            observation_feature_dim=observation_feature_dim,
            obs_steps=obs_steps,
            coord_dim=coord_dim,
            output_dim=output_dim,
            num_points=num_points,
            smoke_test=smoke_test,
        )
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


def _reject_duplicate_parameter_tasks(model_name: str, params: torch.Tensor) -> None:
    if model_name not in _DUPLICATE_PARAM_REJECT_MODELS:
        return
    rounded = [round(float(value), 8) for value in params.detach().cpu().tolist()]
    if len(set(rounded)) != len(rounded):
        raise ValueError(f"{model_name} does not support duplicate parameter tasks in Exp3.")


def _train_exp3_model(
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
    desc = f"{model_name[:12]}-{problem_name[:10]}-s{seed}"

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

    active_train_coords = _build_fno_grid_like(train_coords) if family == "FNO" and has_oc and train_coords.shape[-1] > 1 else train_coords
    active_val_bundle = val_bundle
    if val_bundle is not None and family == "FNO" and has_oc and val_bundle.coords.shape[-1] > 1:
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
        max_phys_points=64,
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


def _predict_new_model_numpy(
    spec: dict[str, Any],
    observations: torch.Tensor,
    coords: torch.Tensor,
    params: torch.Tensor,
    grid_size: int,
    device: torch.device,
    state_normalizer: StateNormalizer | None = None,
) -> np.ndarray[Any, Any]:
    family = str(spec["family"])
    model = spec["model"]
    has_oc = hasattr(model, "observation_conditioner")
    if family == "FNO" and not has_oc:
        pred_norm = predict_fno(model, observations, grid_size, device)
        if state_normalizer is not None:
            pred_norm = state_normalizer.denormalize(pred_norm)
        return pred_norm.detach().cpu().numpy()
    if has_oc:
        return _predict_oc_model_numpy(model, family, observations, coords, device, state_normalizer=state_normalizer)
    return predict_direct(
        model,
        family.lower(),
        observations,
        coords,
        params,
        device,
        state_normalizer=state_normalizer,
    ).detach().cpu().numpy()


def _predict_exp3_numpy(
    model_name: str,
    model: Any,
    test_obs: torch.Tensor,
    test_coords: torch.Tensor,
    test_params: torch.Tensor,
    grid_size: int,
    device: torch.device,
    state_normalizer: StateNormalizer | None = None,
) -> np.ndarray[Any, Any]:
    if state_normalizer is None:
        raise ValueError("Exp 3 predictions require state_normalizer so outputs can be denormalized.")

    legacy_spec = _LEGACY_MODEL_SPECS.get(model_name)
    if legacy_spec is not None:
        legacy_predict = legacy_spec.get("legacy_predict")
        family = str(legacy_spec["family"])
        if legacy_predict == "tapinn":
            return _tapinn_predict_numpy(model, test_obs, test_coords, device, state_normalizer=state_normalizer)
        if family == "FNO":
            pred_norm = predict_fno(model, test_obs, grid_size, device)
            return state_normalizer.denormalize(pred_norm).detach().cpu().numpy()
        return predict_direct(
            model,
            family.lower(),
            test_obs,
            test_coords,
            test_params,
            device,
            state_normalizer=state_normalizer,
        ).detach().cpu().numpy()

    spec = {
        "family": "",
        "model": model,
    }
    for candidate in _FULL_MODEL_SPECS:
        if str(candidate["name"]) == model_name:
            spec["family"] = str(candidate["family"])
            break
    if not spec["family"]:
        raise ValueError(f"Unknown Exp3 model name: {model_name}")
    return _predict_new_model_numpy(spec, test_obs, test_coords, test_params, grid_size, device, state_normalizer=state_normalizer)


def _predict_exp3_model(
    model_name: str,
    model: Any,
    test_obs: torch.Tensor,
    test_coords: torch.Tensor,
    test_params: torch.Tensor,
    grid_size: int,
    device: torch.device,
    state_normalizer: StateNormalizer | None = None,
) -> tuple[np.ndarray[Any, Any], float]:
    return _measure_inference_ms(
        _predict_exp3_numpy,
        test_obs.shape[0],
        model_name,
        model,
        test_obs,
        test_coords,
        test_params,
        grid_size,
        device,
        state_normalizer,
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


def _eval_model_on_dataset(
    model_name: str,
    task_name: str,
    system_data: Any,
    observations: torch.Tensor,
    coords: torch.Tensor,
    targets: torch.Tensor,
    params: torch.Tensor,
    ode_metadata: torch.Tensor | None,
    problem_name: str,
    device: torch.device,
    smoke_test: bool,
    seed: int,
    max_epochs: int | None = None,
    callbacks: CallbackConfig | None = None,
    coord_normalizer: CoordNormalizer | None = None,
    state_normalizer: StateNormalizer | None = None,
) -> dict[str, object]:
    _reject_duplicate_parameter_tasks(model_name, params)

    if coord_normalizer is None or state_normalizer is None:
        raise ValueError("Exp 3 evaluation requires coord_normalizer and state_normalizer.")

    if max_epochs is None:
        max_epochs = _resolve_max_epochs(smoke_test)
    if callbacks is None:
        callbacks = CallbackConfig(
            early_stopping_patience=2 if smoke_test else 15,
            reduce_lr_patience=1 if smoke_test else 8,
            reduce_lr_factor=0.5,
            min_lr=1e-6,
        )

    train_idx, val_idx, test_idx = _split_indices_three_way(len(params), seed)
    train_obs, train_coords, train_targets, train_params = _subset_tensors(train_idx, observations, coords, targets, params)
    val_obs, val_coords, val_targets, val_params = _subset_tensors(val_idx, observations, coords, targets, params)
    test_obs, test_coords, test_targets, test_params = _subset_tensors(test_idx, observations, coords, targets, params)

    trajectory_ids = _build_trajectory_ids(params)
    train_trajectory_ids = _subset_tensors(train_idx, trajectory_ids)[0]
    test_trajectory_ids = _subset_tensors(test_idx, trajectory_ids)[0]
    train_ode_meta = _subset_optional_tensor(train_idx, ode_metadata)
    val_ode_meta = _subset_optional_tensor(val_idx, ode_metadata)
    test_ode_meta = _subset_optional_tensor(test_idx, ode_metadata)

    (
        coord_normalizer,
        state_normalizer,
        train_targets,
        val_targets,
        test_targets,
        train_obs,
        val_obs,
        test_obs,
    ) = refit_normalizers_on_physical_split(
        original_coord_norm=coord_normalizer,
        original_state_norm=state_normalizer,
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

    obs_steps = observations.shape[1]
    observation_feature_dim = observations.shape[-1]
    coord_dim = coords.shape[-1]
    output_dim = targets.shape[-1]
    num_points = targets.shape[1]

    spec = _build_named_model_spec(
        model_name,
        observation_feature_dim=observation_feature_dim,
        obs_steps=obs_steps,
        coord_dim=coord_dim,
        output_dim=output_dim,
        num_points=num_points,
        smoke_test=smoke_test,
    )
    train_result = _train_exp3_model(
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
        batch_size=4,
        seed=seed,
        coord_normalizer=coord_normalizer,
        state_normalizer=state_normalizer,
    )

    predictions, inference_ms = _predict_exp3_model(
        model_name,
        spec["model"],
        test_obs,
        test_coords,
        test_params,
        num_points,
        device,
        state_normalizer=state_normalizer,
    )
    train_predictions = _predict_exp3_numpy(
        model_name,
        spec["model"],
        train_obs,
        train_coords,
        train_params,
        num_points,
        device,
        state_normalizer=state_normalizer,
    )

    train_truth = state_normalizer.denormalize(train_targets).cpu().numpy()
    test_truth = state_normalizer.denormalize(test_targets).cpu().numpy()
    train_data_mse = mse(train_predictions, train_truth)
    generalization_gap = float(mse(predictions, test_truth) - train_data_mse)

    if coord_dim == 1:
        data_mse, physics_residual = _aggregate_ode_metrics(
            problem_name,
            system_data.times,
            predictions,
            test_truth,
            test_params,
            test_ode_meta,
        )
    else:
        boundary = str(system_data.metadata.get("boundary", "periodic")) if system_data.metadata else "periodic"
        data_mse, physics_residual = _aggregate_pde_metrics(
            problem_name,
            system_data.times,
            system_data.space,
            predictions,
            test_truth,
            test_params,
            boundary,
        )

    relative_l2_error = compute_relative_l2_error(predictions, test_truth)
    disambiguation_score = _compute_disambiguation_metric(spec, test_obs, test_trajectory_ids, device)
    param_count = count_parameters(cast(nn.Module, spec["model"]))

    return {
        "task": task_name,
        "problem": task_name,
        "model_name": model_name,
        "model": model_name,
        "family": spec["family"],
        "with_oc": spec["with_oc"],
        "comparison_group": spec["training_group"],
        "oc_variant": spec["oc_variant"],
        "data_mse": data_mse,
        "train_data_mse": train_data_mse,
        "relative_l2_error": relative_l2_error,
        "physics_residual": physics_residual,
        "disambiguation_score": disambiguation_score,
        "seconds_per_epoch": train_result.seconds_per_epoch,
        "epochs_trained": train_result.epochs_trained,
        "best_val_loss": train_result.best_val_loss,
        "inference_ms": inference_ms,
        "param_count": param_count,
        "task_param_count": param_count,
        "generalization_gap": generalization_gap,
    }


def _build_oc_benefit_rows(problem_model_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_problem_family: dict[tuple[str, str], dict[bool, dict[str, Any]]] = {}
    for row in problem_model_summaries:
        key = (str(row["problem"]), str(row["family"]))
        by_problem_family.setdefault(key, {})[bool(row["with_oc"])] = row

    rows: list[dict[str, Any]] = []
    for problem, family in sorted(by_problem_family.keys(), key=lambda item: (_FAMILY_ORDER.index(item[1]), item[0])):
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
                "param_count_delta": with_oc["param_count"] - no_oc["param_count"],
                "relative_l2_error_delta": no_oc["relative_l2_error_mean"] - with_oc["relative_l2_error_mean"],
                "data_mse_delta": no_oc["data_mse_mean"] - with_oc["data_mse_mean"],
                "physics_residual_delta": no_oc["physics_residual_mean"] - with_oc["physics_residual_mean"],
                "generalization_gap_delta": no_oc["generalization_gap_mean"] - with_oc["generalization_gap_mean"],
                "oc_disambiguation_score_mean": with_oc["disambiguation_score_mean"],
            }
        )
    return rows


def _param_vs_residual_scatter(model_rows: list[dict[str, Any]], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    palette = {"with_oc": "#55A868", "without_oc": "#C44E52"}
    for row in model_rows:
        color = palette.get(str(row.get("oc_variant", "")), "#4C72B0")
        ax.scatter(row["param_count"], row["physics_residual_mean"], s=80, color=color)
        ax.annotate(row["model"], (row["param_count"], row["physics_residual_mean"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Parameter Count")
    ax.set_ylabel("Physics Residual")
    ax.set_title(title)
    save_figure(fig, path)
    _save_plot_data(path, {"title": title, "rows": model_rows})


def _prepare_duffing_dataset_for_exp3(smoke_test: bool, seed: int):
    param_values = [0.24, 0.38, 0.52] if smoke_test else np.linspace(0.2, 0.55, 7).tolist()
    data = generate_duffing_dataset(
        param_values,
        num_trajectories=1 if smoke_test else 10,
        num_points=56 if smoke_test else 160,
        t_span=(0.0, 16.0),
        seed=seed,
    )
    tensors = prepare_ode_tensors(data, observation_steps=8 if smoke_test else 20)
    return data, tensors


def _prepare_allen_cahn_dataset_for_exp3(smoke_test: bool, seed: int):
    param_values = [0.8, 1.1] if smoke_test else [0.75, 0.9, 1.05, 1.2]
    data = generate_allen_cahn_dataset(
        param_values,
        num_samples=1 if smoke_test else 8,
        nx=16 if smoke_test else 48,
        nt=16 if smoke_test else 36,
        seed=seed,
    )
    tensors = prepare_pde_tensors(data, observation_steps=4 if smoke_test else 8)
    return data, tensors


def run_exp_3_sota_baselines_and_capacity(output_root: str, device_name: str, smoke_test: bool, seed: int) -> dict[str, object]:
    configure_plotting()
    device = get_best_device(device_name)
    run_dir = prepare_run_dir(output_root, "exp_3_sota_baselines_and_capacity")
    seeds = [seed] if smoke_test else [seed + offset for offset in range(5)]
    all_configs = _resolve_run_all_configs(smoke_test)
    max_epochs = _resolve_max_epochs(smoke_test)
    callbacks = CallbackConfig(
        early_stopping_patience=2 if smoke_test else 15,
        reduce_lr_patience=1 if smoke_test else 8,
        reduce_lr_factor=0.5,
        min_lr=1e-6,
    )

    model_specs = _selected_model_specs(smoke_test, all_configs)
    model_names = [str(spec["name"]) for spec in model_specs]
    problems = ["duffing", "allen_cahn"]

    per_task_seed_metrics: list[dict[str, Any]] = []
    problem_model_summaries: list[dict[str, Any]] = []

    for problem_name in tqdm(problems, desc="Exp3 tasks", leave=False):
        per_model_seed_data: dict[str, list[dict[str, Any]]] = {name: [] for name in model_names}

        for local_seed in tqdm(seeds, desc=f"{problem_name} seeds", leave=False):
            set_global_seed(local_seed)
            if problem_name == "duffing":
                system_data, prepared = _prepare_duffing_dataset_for_exp3(smoke_test, local_seed)
                observations, coords, targets, params, ode_metadata, coord_normalizer, state_normalizer = prepared
            else:
                system_data, prepared = _prepare_allen_cahn_dataset_for_exp3(smoke_test, local_seed + 11)
                observations, coords, targets, params, coord_normalizer, state_normalizer = prepared
                ode_metadata = None

            for model_name in tqdm(model_names, desc=f"{problem_name[:8]}/models", leave=False):
                result = _eval_model_on_dataset(
                    model_name,
                    problem_name,
                    system_data,
                    observations,
                    coords,
                    targets,
                    params,
                    ode_metadata,
                    problem_name,
                    device,
                    smoke_test,
                    local_seed,
                    max_epochs=max_epochs,
                    callbacks=callbacks,
                    coord_normalizer=coord_normalizer,
                    state_normalizer=state_normalizer,
                )
                per_model_seed_data[model_name].append(result)
                per_task_seed_metrics.append({"seed": local_seed, **result})

        for model_name in model_names:
            rows = per_model_seed_data[model_name]
            if not rows:
                continue
            data_mean, data_std = mean_std(row["data_mse"] for row in rows)
            train_mean, train_std = mean_std(row["train_data_mse"] for row in rows)
            rel_mean, rel_std = mean_std(row["relative_l2_error"] for row in rows)
            phys_mean, phys_std = mean_std(row["physics_residual"] for row in rows)
            gap_mean, gap_std = mean_std(row["generalization_gap"] for row in rows)
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
                    "train_data_mse_mean": train_mean,
                    "train_data_mse_std": train_std,
                    "data_mse_mean": data_mean,
                    "data_mse_std": data_std,
                    "relative_l2_error_mean": rel_mean,
                    "relative_l2_error_std": rel_std,
                    "physics_residual_mean": phys_mean,
                    "physics_residual_std": phys_std,
                    "generalization_gap_mean": gap_mean,
                    "generalization_gap_std": gap_std,
                    "disambiguation_score_mean": dis_mean if rows[0]["with_oc"] else None,
                    "disambiguation_score_std": dis_std if rows[0]["with_oc"] else None,
                    "epochs_trained_mean": ep_mean,
                    "epochs_trained_std": ep_std,
                    "seconds_per_epoch_mean": spe_mean,
                    "inference_ms_mean": inf_mean,
                }
            )

    capacity_benchmark: list[dict[str, Any]] = []
    for model_name in model_names:
        rows = [row for row in per_task_seed_metrics if str(row["model"]) == model_name]
        if not rows:
            continue
        data_mean, data_std = mean_std(row["data_mse"] for row in rows)
        rel_mean, rel_std = mean_std(row["relative_l2_error"] for row in rows)
        phys_mean, phys_std = mean_std(row["physics_residual"] for row in rows)
        gap_mean, gap_std = mean_std(row["generalization_gap"] for row in rows)
        ep_mean, ep_std = mean_std(row["epochs_trained"] for row in rows)
        spe_mean, _ = mean_std(row["seconds_per_epoch"] for row in rows)
        inf_mean, _ = mean_std(row["inference_ms"] for row in rows)
        capacity_benchmark.append(
            {
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
                "generalization_gap_mean": gap_mean,
                "generalization_gap_std": gap_std,
                "epochs_trained_mean": ep_mean,
                "epochs_trained_std": ep_std,
                "seconds_per_epoch_mean": spe_mean,
                "inference_ms_mean": inf_mean,
            }
        )

    oc_benefit_rows = _build_oc_benefit_rows(problem_model_summaries)

    _param_vs_residual_scatter(
        [row for row in capacity_benchmark if row["comparison_group"] == "physics_trained"],
        "Exp3 Pareto Frontier — Parameter Count vs Physics Residual",
        run_dir / "figures" / "pareto_frontier.pdf",
    )
    _metrics_bar_chart(
        [str(row["model"]) for row in capacity_benchmark],
        [float(row["generalization_gap_mean"]) for row in capacity_benchmark],
        "Generalization Gap (test MSE - train MSE)",
        "Exp3 Overfitting Gap by Model",
        run_dir / "figures" / "overfitting_gap_bar_chart.pdf",
        comparison_groups=[str(row["oc_variant"]) for row in capacity_benchmark],
    )

    write_csv(run_dir / "tables" / "per_task_seed_metrics.csv", per_task_seed_metrics)
    write_csv(run_dir / "tables" / "capacity_benchmark.csv", capacity_benchmark)
    write_csv(run_dir / "tables" / "model_summary.csv", problem_model_summaries)
    write_csv(run_dir / "tables" / "oc_benefit_summary.csv", oc_benefit_rows)

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
        "per_task_seed_metrics": per_task_seed_metrics,
        "capacity_benchmark": capacity_benchmark,
        "summary": problem_model_summaries,
        "oc_benefit": oc_benefit_rows,
    }
    write_json(run_dir / "results.json", payload)
    return payload
