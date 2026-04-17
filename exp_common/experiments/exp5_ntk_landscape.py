from __future__ import annotations
# pyright: reportUnusedImport=false, reportPrivateUsage=false, reportPrivateImportUsage=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportUntypedFunctionDecorator=false, reportUnusedCallResult=false

import os
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from ..device import get_best_device
from ..io_utils import prepare_run_dir, write_csv, write_json
from ..metrics import compute_relative_l2_error, mean_std
from ..models import count_parameters, create_model
from ..plotting import configure_plotting
from ..problems import generate_duffing_dataset, generate_kuramoto_dataset, generate_lorenz_dataset
from ..repro import set_global_seed
from ..trainers import (
    CallbackConfig,
    CoordNormalizer,
    StateNormalizer,
    ValBundle,
    lipschitz_estimate,
    predict_direct,
    predict_fno,
    prepare_ode_tensors,
    refit_normalizers_on_physical_split,
    train_direct_model,
    train_fno_model,
)
from .common import (
    tqdm,
    _aggregate_ode_metrics,
    _condition_plot,
    _final_conditioning_summary_plot,
    _measure_inference_ms,
    _spectrum_plot,
    _split_indices_three_way,
    _subset_optional_tensor,
    _subset_tensors,
)


_EXP_EPOCHS_ENV = "TAPINN_EXPERIMENT_EPOCHS"
_EXP_ALL_CONFIGS_ENV = "TAPINN_EXPERIMENT_ALL_CONFIGS"

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
    return 20 if smoke_test else 400


def _resolve_checkpoint_step(smoke_test: bool, total_epochs: int) -> int:
    if smoke_test:
        return min(10, total_epochs)
    return min(100, total_epochs)


def _selected_model_specs(smoke_test: bool, all_configs: bool) -> list[dict[str, Any]]:
    if smoke_test and not all_configs:
        return [dict(spec) for spec in _SMOKE_MODEL_SPECS]
    return [dict(spec) for spec in _FULL_MODEL_SPECS]


def _oc_variant_label(with_oc: bool) -> str:
    return "with_oc" if with_oc else "without_oc"


def _comparison_group(family: str) -> str:
    return "supervised_only" if family == "FNO" else "physics_trained"


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
        built[str(spec["name"])] = {
            "base_name": str(spec["name"]),
            "family": family,
            "with_oc": with_oc,
            "comparison_group": _comparison_group(family),
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


def _is_soft_ao_eligible(spec: dict[str, Any]) -> bool:
    model = cast(nn.Module, spec["model"])
    return bool(spec["with_oc"]) and bool(getattr(model, "has_physics_loss", True))


def _build_run_specs(model_specs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    run_specs: list[dict[str, Any]] = []
    for base_name, spec in model_specs.items():
        if _is_soft_ao_eligible(spec):
            run_specs.append({**spec, "run_name": f"{base_name}_ao", "alternating": True, "training_mode": "ao"})
            run_specs.append({**spec, "run_name": f"{base_name}_joint", "alternating": False, "training_mode": "joint"})
        else:
            run_specs.append({**spec, "run_name": base_name, "alternating": False, "training_mode": "standard"})
    return run_specs


def _ntk_note(family: str, with_oc: bool) -> tuple[bool, str]:
    if family == "FNO" and not with_oc:
        return False, "Skipped NTK/Jacobian: FNO baseline forward pass is grid-based and not pointwise-compatible."
    if family == "FNO" and with_oc:
        return True, "FNO_OC NTK uses observation_conditioner-only Jacobian diagnostics through decode."
    return True, "Full model NTK/Jacobian diagnostics."


def _finite_ntk_and_condition(
    model: nn.Module,
    model_kind: str,
    observations: torch.Tensor,
    coords: torch.Tensor,
    params: torch.Tensor,
    device: torch.device,
) -> tuple[list[float], float, float]:
    diag_samples = min(4, observations.shape[0])
    points_per_sample = max(1, min(6, coords.shape[1]))
    jac_rows: list[torch.Tensor] = []

    model.to(device)
    model.train()

    if model_kind == "fno" and not hasattr(model, "observation_conditioner"):
        return [1e-12], float("nan"), 0.0

    with torch.enable_grad():
        if hasattr(model, "observation_conditioner"):
            model_ref = cast(Any, model)
            obs_batch = observations[:diag_samples].to(device)
            latent_batch = model_ref.observation_conditioner(obs_batch)

            if model_kind == "fno":
                target_params = [p for p in model_ref.observation_conditioner.parameters() if p.requires_grad]
                for sample_idx in range(diag_samples):
                    sample_grid = coords[sample_idx : sample_idx + 1, :points_per_sample, :1].to(device)
                    sample_latent = latent_batch[sample_idx : sample_idx + 1]
                    pred = model_ref.decode(sample_grid, sample_latent)
                    scalar = pred[0, 0, 0]
                    grads = torch.autograd.grad(
                        scalar,
                        target_params,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    )
                    grad_vec = torch.cat(
                        [g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1) for g, p in zip(grads, target_params)]
                    )
                    jac_rows.append(grad_vec)
            else:
                target_params = [p for p in model.parameters() if p.requires_grad]
                for sample_idx in range(diag_samples):
                    sample_coords = coords[sample_idx, :points_per_sample, :].to(device)
                    sample_latent = latent_batch[sample_idx : sample_idx + 1].expand(sample_coords.shape[0], -1)
                    for idx in range(sample_coords.shape[0]):
                        coord = sample_coords[idx : idx + 1].detach().clone().requires_grad_(True)
                        pred = model_ref.decode(coord, sample_latent[idx : idx + 1])
                        scalar = pred[0, 0]
                        grads = torch.autograd.grad(
                            scalar,
                            target_params,
                            retain_graph=True,
                            create_graph=False,
                            allow_unused=True,
                        )
                        grad_vec = torch.cat(
                            [g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1) for g, p in zip(grads, target_params)]
                        )
                        jac_rows.append(grad_vec)
        else:
            if model_kind == "deeponet":
                branch_input = observations[:diag_samples].reshape(diag_samples, -1).to(device)
                target_params = [p for p in model.parameters() if p.requires_grad]
                for sample_idx in range(diag_samples):
                    sample_coords = coords[sample_idx, :points_per_sample, :].to(device)
                    sample_branch = branch_input[sample_idx : sample_idx + 1]
                    for idx in range(sample_coords.shape[0]):
                        coord = sample_coords[idx : idx + 1].detach().clone().requires_grad_(True)
                        pred = model(sample_branch, coord)
                        scalar = pred[0, 0]
                        grads = torch.autograd.grad(
                            scalar,
                            target_params,
                            retain_graph=True,
                            create_graph=False,
                            allow_unused=True,
                        )
                        grad_vec = torch.cat(
                            [g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1) for g, p in zip(grads, target_params)]
                        )
                        jac_rows.append(grad_vec)
            else:
                target_params = [p for p in model.parameters() if p.requires_grad]
                for sample_idx in range(diag_samples):
                    sample_coords = coords[sample_idx, :points_per_sample, :].to(device)
                    sample_param = params[sample_idx : sample_idx + 1].to(device)
                    for idx in range(sample_coords.shape[0]):
                        coord = sample_coords[idx : idx + 1].detach().clone().requires_grad_(True)
                        pred = model(coord, sample_param)
                        scalar = pred[0, 0]
                        grads = torch.autograd.grad(
                            scalar,
                            target_params,
                            retain_graph=True,
                            create_graph=False,
                            allow_unused=True,
                        )
                        grad_vec = torch.cat(
                            [g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1) for g, p in zip(grads, target_params)]
                        )
                        jac_rows.append(grad_vec)

    if not jac_rows:
        return [1e-12], 1.0, 0.0

    jacobian = torch.stack(jac_rows)
    ntk = jacobian @ jacobian.T
    eigvals = torch.linalg.eigvalsh(ntk).detach().cpu().numpy()
    eigvals = np.sort(np.clip(eigvals.real, 1e-12, None))[::-1]

    singular_vals = torch.linalg.svdvals(jacobian).detach().cpu().numpy()
    singular_vals = np.clip(singular_vals, 1e-12, None)
    condition_number = float(singular_vals.max() / singular_vals.min())

    lipschitz = 0.0
    if hasattr(model, "observation_conditioner"):
        obs_batch = observations[:diag_samples].to(device)
        lipschitz = float(lipschitz_estimate(cast(Any, model).observation_conditioner, obs_batch))

    return eigvals.tolist(), condition_number, lipschitz


def _train_model(
    run_spec: dict[str, Any],
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
    epochs: int,
    batch_size: int,
    progress_desc: str,
    coord_normalizer: CoordNormalizer,
    state_normalizer: StateNormalizer,
):
    model = cast(nn.Module, run_spec["model"])
    family = str(run_spec["family"])
    has_oc = hasattr(model, "observation_conditioner")

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
            epochs=epochs,
            batch_size=batch_size,
            progress_desc=progress_desc,
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
        epochs=epochs,
        batch_size=batch_size,
        progress_desc=progress_desc,
        alternating=bool(run_spec["alternating"]),
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
        pred_norm = model_ref.decode(coord_batch[..., :1], latent)
    else:
        expanded_latent = latent.unsqueeze(1).expand(-1, coord_batch.shape[1], -1).reshape(-1, latent.shape[-1])
        pred_norm = model_ref.decode(coord_batch.reshape(-1, coord_batch.shape[-1]), expanded_latent)
        pred_norm = pred_norm.reshape(obs.shape[0], coord_batch.shape[1], -1)
    if state_normalizer is not None:
        pred_norm = state_normalizer.denormalize(pred_norm)
    return pred_norm.detach().cpu().numpy()


def _predict_model(
    run_spec: dict[str, Any],
    test_obs: torch.Tensor,
    test_coords: torch.Tensor,
    test_params: torch.Tensor,
    device: torch.device,
    state_normalizer: StateNormalizer,
) -> tuple[np.ndarray[Any, Any], float]:
    model = cast(nn.Module, run_spec["model"])
    family = str(run_spec["family"])
    has_oc = hasattr(model, "observation_conditioner")
    if family == "FNO" and not has_oc:
        return _measure_inference_ms(
            lambda *a: state_normalizer.denormalize(predict_fno(*a)).detach().cpu().numpy(),
            test_obs.shape[0],
            model,
            test_obs,
            test_coords.shape[1],
            device,
        )
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


def _prepare_problem_data(problem_name: str, seed: int, smoke_test: bool):
    num_points = 32 if smoke_test else 160
    num_trajectories = 2 if smoke_test else 12

    if problem_name == "duffing":
        data = generate_duffing_dataset(
            [0.24, 0.38, 0.52] if smoke_test else np.linspace(0.2, 0.55, 5).tolist(),
            num_trajectories=num_trajectories,
            num_points=num_points,
            t_span=(0.0, 12.0),
            seed=seed,
        )
    elif problem_name == "lorenz":
        data = generate_lorenz_dataset(
            [20.0, 24.74, 30.0] if smoke_test else np.linspace(18.0, 32.0, 5).tolist(),
            num_trajectories=num_trajectories,
            num_points=num_points,
            t_span=(0.0, 4.0),
            seed=seed,
        )
    elif problem_name == "kuramoto":
        data = generate_kuramoto_dataset(
            [0.8, 1.6, 2.4] if smoke_test else np.linspace(0.5, 2.5, 5).tolist(),
            num_trajectories=num_trajectories,
            num_points=num_points,
            t_span=(0.0, 8.0),
            num_oscillators=5,
            seed=seed,
        )
    else:
        raise ValueError(problem_name)

    obs_steps = 8 if smoke_test else max(20, int(0.15 * data.states.shape[1]))
    return data, prepare_ode_tensors(data, observation_steps=obs_steps), obs_steps, num_points


def _aggregate_problem_model_summary(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in seed_rows:
        grouped.setdefault((str(row["problem"]), str(row["model"])), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (problem, model_name), rows in sorted(grouped.items()):
        rel_mean, rel_std = mean_std(float(row["relative_l2_error"]) for row in rows)
        mse_mean, mse_std = mean_std(float(row["data_mse"]) for row in rows)
        phys_mean, phys_std = mean_std(float(row["physics_residual"]) for row in rows)
        cond_values = [float(row["final_condition_number"]) for row in rows if row["final_condition_number"] is not None]
        lip_values = [float(row["final_lipschitz"]) for row in rows if row["final_lipschitz"] is not None]
        cond_mean, cond_std = mean_std(cond_values) if cond_values else (0.0, 0.0)
        lip_mean, lip_std = mean_std(lip_values) if lip_values else (0.0, 0.0)
        inf_mean, inf_std = mean_std(float(row["inference_ms"]) for row in rows)

        summary_rows.append(
            {
                "problem": problem,
                "model": model_name,
                "family": rows[0]["family"],
                "base_model": rows[0]["base_model"],
                "with_oc": rows[0]["with_oc"],
                "oc_variant": rows[0]["oc_variant"],
                "comparison_group": rows[0]["comparison_group"],
                "training_mode": rows[0]["training_mode"],
                "soft_ao_eligible": rows[0]["soft_ao_eligible"],
                "ntk_supported": rows[0]["ntk_supported"],
                "ntk_note": rows[0]["ntk_note"],
                "relative_l2_error_mean": rel_mean,
                "relative_l2_error_std": rel_std,
                "data_mse_mean": mse_mean,
                "data_mse_std": mse_std,
                "physics_residual_mean": phys_mean,
                "physics_residual_std": phys_std,
                "final_condition_number_mean": cond_mean if cond_values else None,
                "final_condition_number_std": cond_std if cond_values else None,
                "final_lipschitz_mean": lip_mean if lip_values else None,
                "final_lipschitz_std": lip_std if lip_values else None,
                "inference_ms_mean": inf_mean,
                "inference_ms_std": inf_std,
                "param_count": rows[0]["param_count"],
            }
        )
    return summary_rows


def _build_soft_ao_summary(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in summary_rows:
        if not bool(row["soft_ao_eligible"]):
            continue
        grouped.setdefault((str(row["problem"]), str(row["base_model"])), {})[str(row["training_mode"])] = row

    ao_rows: list[dict[str, Any]] = []
    for (problem, base_model), pair in sorted(grouped.items()):
        if "ao" not in pair or "joint" not in pair:
            continue
        ao = pair["ao"]
        joint = pair["joint"]
        ao_rows.append(
            {
                "problem": problem,
                "base_model": base_model,
                "family": ao["family"],
                "ao_model": ao["model"],
                "joint_model": joint["model"],
                "ao_condition_number_mean": ao["final_condition_number_mean"],
                "joint_condition_number_mean": joint["final_condition_number_mean"],
                "condition_number_delta": (
                    float(joint["final_condition_number_mean"]) - float(ao["final_condition_number_mean"])
                    if ao["final_condition_number_mean"] is not None and joint["final_condition_number_mean"] is not None
                    else None
                ),
                "ao_relative_l2_error_mean": ao["relative_l2_error_mean"],
                "joint_relative_l2_error_mean": joint["relative_l2_error_mean"],
            }
        )
    return ao_rows


def run_exp_5_theoretical_optimization_landscape(output_root: str, device_name: str, smoke_test: bool, seed: int) -> dict[str, object]:
    configure_plotting()
    device = get_best_device(device_name)
    run_dir = prepare_run_dir(output_root, "exp_5_theoretical_optimization_landscape")

    seeds = [seed] if smoke_test else [seed + offset for offset in range(3)]
    problems = ["duffing"] if smoke_test else ["duffing", "kuramoto", "lorenz"]
    all_configs = _resolve_run_all_configs(smoke_test)
    total_epochs = _resolve_max_epochs(smoke_test)
    checkpoint_step = _resolve_checkpoint_step(smoke_test, total_epochs)

    callbacks = CallbackConfig(
        early_stopping_patience=total_epochs,
        reduce_lr_patience=max(1, checkpoint_step),
        reduce_lr_factor=0.5,
        min_lr=1e-6,
    )

    spectrum_records: list[dict[str, Any]] = []
    condition_records: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    limitation_rows: list[dict[str, Any]] = []

    selected_specs = _selected_model_specs(smoke_test, all_configs)

    for problem_name in tqdm(problems, desc="Optimization problems", leave=False):
        fig_subdir = run_dir / "figures" / problem_name
        fig_subdir.mkdir(parents=True, exist_ok=True)

        for local_seed in tqdm(seeds, desc=f"{problem_name} seeds", leave=False):
            set_global_seed(local_seed)
            data, prepared, obs_steps, num_points = _prepare_problem_data(problem_name, local_seed, smoke_test)
            observations, coords, targets, params, ode_metadata, coord_norm, state_norm = prepared
            trajectory_ids = _build_trajectory_ids(params)

            train_idx, val_idx, test_idx = _split_indices_three_way(len(params), local_seed)
            train_obs, train_coords, train_targets, train_params = _subset_tensors(train_idx, observations, coords, targets, params)
            val_obs, val_coords, val_targets, val_params = _subset_tensors(val_idx, observations, coords, targets, params)
            test_obs, test_coords, test_targets, test_params = _subset_tensors(test_idx, observations, coords, targets, params)
            train_trajectory_ids = _subset_tensors(train_idx, trajectory_ids)[0]
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

            model_specs = _build_model_configs(
                obs_dim=observations.shape[-1],
                obs_steps=obs_steps,
                output_dim=targets.shape[-1],
                num_points=num_points,
                smoke_test=smoke_test,
                model_specs=selected_specs,
            )
            run_specs = _build_run_specs(model_specs)
            truth = state_norm.denormalize(test_targets).cpu().numpy()

            for run_spec in tqdm(run_specs, desc=f"{problem_name} models", leave=False):
                run_name = str(run_spec["run_name"])
                family = str(run_spec["family"])
                ntk_supported, ntk_note = _ntk_note(family, bool(run_spec["with_oc"]))
                epochs_trained_total = 0
                last_train_result = None
                final_condition_number: float | None = None
                final_lipschitz: float | None = None

                if not ntk_supported:
                    limitation_rows.append(
                        {
                            "problem": problem_name,
                            "seed": local_seed,
                            "model": run_name,
                            "family": family,
                            "with_oc": run_spec["with_oc"],
                            "note": ntk_note,
                        }
                    )

                for epoch in range(0, total_epochs + 1, checkpoint_step):
                    if epoch > 0:
                        remaining = min(checkpoint_step, total_epochs - epochs_trained_total)
                        if remaining > 0:
                            last_train_result = _train_model(
                                run_spec,
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
                                remaining,
                                batch_size=4,
                                progress_desc=f"{problem_name[:4]}-{run_name[:14]}-s{local_seed}-e{epoch}",
                                coord_normalizer=coord_norm,
                                state_normalizer=state_norm,
                            )
                            epochs_trained_total += int(last_train_result.epochs_trained)

                    if ntk_supported:
                        eigvals, condition_number, lipschitz = _finite_ntk_and_condition(
                            cast(nn.Module, run_spec["model"]),
                            family.lower(),
                            train_obs,
                            train_coords,
                            train_params,
                            device,
                        )
                        final_condition_number = condition_number
                        final_lipschitz = lipschitz
                        spectrum_records.append(
                            {
                                "problem": problem_name,
                                "seed": local_seed,
                                "model": run_name,
                                "epoch": epoch,
                                "eigenvalues": eigvals[:24],
                                "family": family,
                                "with_oc": run_spec["with_oc"],
                                "training_mode": run_spec["training_mode"],
                            }
                        )
                        condition_records.append(
                            {
                                "problem": problem_name,
                                "seed": local_seed,
                                "model": run_name,
                                "epoch": epoch,
                                "condition_number": condition_number,
                                "lipschitz": lipschitz,
                                "family": family,
                                "with_oc": run_spec["with_oc"],
                                "training_mode": run_spec["training_mode"],
                            }
                        )

                predictions, inference_ms = _predict_model(
                    run_spec,
                    test_obs,
                    test_coords,
                    test_params,
                    device,
                    state_norm,
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
                param_count = count_parameters(cast(nn.Module, run_spec["model"]))
                seed_rows.append(
                    {
                        "problem": problem_name,
                        "seed": local_seed,
                        "model": run_name,
                        "base_model": run_spec["base_name"],
                        "family": family,
                        "with_oc": run_spec["with_oc"],
                        "oc_variant": run_spec["oc_variant"],
                        "comparison_group": run_spec["comparison_group"],
                        "training_mode": run_spec["training_mode"],
                        "soft_ao_eligible": _is_soft_ao_eligible(run_spec),
                        "alternating": run_spec["alternating"],
                        "ntk_supported": ntk_supported,
                        "ntk_note": ntk_note,
                        "data_mse": data_mse,
                        "physics_residual": physics_residual,
                        "relative_l2_error": relative_l2_error,
                        "final_condition_number": final_condition_number,
                        "final_lipschitz": final_lipschitz,
                        "epochs_trained": epochs_trained_total,
                        "best_val_loss": None if last_train_result is None else last_train_result.best_val_loss,
                        "seconds_per_epoch": None if last_train_result is None else last_train_result.seconds_per_epoch,
                        "inference_ms": inference_ms,
                        "param_count": param_count,
                    }
                )

        _spectrum_plot(
            spectrum_records,
            f"{problem_name.title()} NTK Spectrum Over Training",
            fig_subdir / "ntk_spectrum.pdf",
            problem_name,
        )
        _condition_plot(
            condition_records,
            f"{problem_name.title()} Jacobian Conditioning Over Training",
            fig_subdir / "condition_number.pdf",
            problem_name,
        )

    _final_conditioning_summary_plot(condition_records, run_dir / "figures" / "global_conditioning_summary.pdf")

    summary_rows = _aggregate_problem_model_summary(seed_rows)
    soft_ao_rows = _build_soft_ao_summary(summary_rows)

    write_csv(run_dir / "tables" / "seed_summary.csv", seed_rows)
    write_csv(run_dir / "tables" / "model_summary.csv", summary_rows)
    write_csv(run_dir / "tables" / "soft_ao_summary.csv", soft_ao_rows)
    write_csv(run_dir / "tables" / "limitations.csv", limitation_rows)
    write_json(
        run_dir / "results.json",
        {
            "device": str(device),
            "smoke_test": smoke_test,
            "all_configs": all_configs,
            "max_epochs": total_epochs,
            "checkpoint_step": checkpoint_step,
            "problems": problems,
            "model_specs": selected_specs,
            "seed_summary": seed_rows,
            "model_summary": summary_rows,
            "soft_ao_summary": soft_ao_rows,
            "limitations": limitation_rows,
        },
    )
    return {
        "device": str(device),
        "smoke_test": smoke_test,
        "all_configs": all_configs,
        "max_epochs": total_epochs,
        "checkpoint_step": checkpoint_step,
        "problems": problems,
        "models": [str(spec["name"]) for spec in selected_specs],
        "seed_summary": seed_rows,
        "model_summary": summary_rows,
        "soft_ao_summary": soft_ao_rows,
    }
