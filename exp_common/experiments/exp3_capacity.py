from __future__ import annotations
# pyright: reportArgumentType=false, reportOptionalMemberAccess=false, reportCallIssue=false

from typing import Any

import numpy as np
import torch

from ..device import get_best_device
from ..io_utils import prepare_run_dir, write_csv, write_json
from ..metrics import mean_std, mse
from ..models import (
    DeepONet,
    LightweightFNO1d,
    build_capacity_matched_hyperpinn,
    build_low_rank_hyperpinn,
    build_tapinn,
    count_parameters,
)
from ..plotting import configure_plotting
from ..problems import generate_allen_cahn_dataset, generate_duffing_dataset
from ..repro import set_global_seed
from ..trainers import (
    CallbackConfig,
    CoordNormalizer,
    StateNormalizer,
    ValBundle,
    predict_fno,
    prepare_ode_tensors,
    prepare_pde_tensors,
    train_direct_model,
    train_fno_model,
    train_tapinn,
)
from .common import (
    tqdm,
    _aggregate_ode_metrics,
    _aggregate_pde_metrics,
    _bar_plot,
    _direct_predict_numpy,
    _fno_predict_numpy,
    _measure_inference_ms,
    _scatter_plot,
    _split_indices_three_way,
    _subset_optional_tensor,
    _subset_tensors,
    _tapinn_predict_numpy,
)


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
) -> dict[str, Any]:
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
    model: Any,
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
    coord_normalizer: CoordNormalizer | None = None,
    state_normalizer: StateNormalizer | None = None,
):
    """Dispatch training for any Exp-3 model variant."""
    desc = f"{model_name[:8]}-{problem_name[:8]}-s{seed}"
    family = _EXP3_MODELS[model_name][0]
    if family in ("tapinn", "tapinn_large"):
        if coord_normalizer is None or state_normalizer is None:
            raise ValueError("Exp 3 TAPINN training requires coord_normalizer and state_normalizer.")
        return train_tapinn(
            model, problem_name, train_obs, train_coords, train_targets, train_params,
            device, ode_metadata=train_ode_meta, epochs=max_epochs, batch_size=batch_size,
            progress_desc=desc, val_bundle=val_bundle, callbacks=callbacks,
            coord_normalizer=coord_normalizer, state_normalizer=state_normalizer,
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
        if coord_normalizer is None or state_normalizer is None:
            raise ValueError("Exp 3 direct-model training requires coord_normalizer and state_normalizer.")
        # direct family: standard, hyper, deeponet
        return train_direct_model(
            model, problem_name, family, train_obs, train_coords, train_targets, train_params,
            device, ode_metadata=train_ode_meta, epochs=max_epochs, batch_size=batch_size,
            progress_desc=desc, val_bundle=val_bundle, callbacks=callbacks,
            coord_normalizer=coord_normalizer, state_normalizer=state_normalizer,
        )


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
    """Run inference for Exp-3 models; consistent with Exp 1 and 2."""
    if state_normalizer is None:
        raise ValueError("Exp 3 predictions require state_normalizer so outputs can be denormalized.")
    family = _EXP3_MODELS[model_name][0]
    if family in ("tapinn", "tapinn_large"):
        return _measure_inference_ms(
            lambda *a: _tapinn_predict_numpy(*a, state_normalizer=state_normalizer),
            test_obs.shape[0], model, test_obs, test_coords, device,
        )
    elif family == "fno":
        return _measure_inference_ms(
            lambda *a: state_normalizer.denormalize(predict_fno(*a)).detach().cpu().numpy(),
            test_obs.shape[0], model, test_obs, grid_size, device,
        )
    else:
        return _measure_inference_ms(
            lambda *a: _direct_predict_numpy(*a, state_normalizer=state_normalizer),
            test_obs.shape[0], model, family, test_obs, test_coords, test_params, device,
        )


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
    """Standardized evaluation for Experiment 3.

    Scientific Role
    ---------------
    * Implements the 70/15/15 three-way split required for fair callback triggers.
    * Reuses the multi-baseline dispatch infrastructure established for Exp 1/2.
    * Calculates the 'generalization_gap' (MSE_test - MSE_train) to detect overfitting
      in high-capacity models (HyperPINN, TAPINN-Large).
    """
    if max_epochs is None:
        max_epochs = 4 if smoke_test else 100
    if callbacks is None:
        callbacks = CallbackConfig(
            early_stopping_patience=2 if smoke_test else 15,
            reduce_lr_patience=1 if smoke_test else 8,
            reduce_lr_factor=0.5,
            min_lr=1e-6,
        )
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
        coord_normalizer=coord_normalizer, state_normalizer=state_normalizer,
    )

    preds, inference_ms = _predict_exp3_model(
        model_name, model, test_obs, test_coords, test_params, grid_size, device,
        state_normalizer=state_normalizer,
    )

    # Generalization gap check (using a separate call to train_pred for simplicity)
    family = _EXP3_MODELS[model_name][0]
    if family in ("tapinn", "tapinn_large"):
        train_pred = _tapinn_predict_numpy(model, train_obs, train_coords, device, state_normalizer=state_normalizer)
    elif family == "fno":
        train_pred = _fno_predict_numpy(model, train_obs, grid_size, device)
    else:
        train_pred = _direct_predict_numpy(model, family, train_obs, train_coords, train_params, device, state_normalizer=state_normalizer)

    train_truth = state_normalizer.denormalize(train_targets).cpu().numpy()
    test_truth = state_normalizer.denormalize(test_targets).cpu().numpy()
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

    per_task_seed_metrics: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    pareto_rows: list[dict[str, float | str]] = []
    gap_labels: list[str] = []
    gap_values: list[float] = []

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
        seed_summaries: list[dict[str, float]] = []
        for local_seed in tqdm(seeds, desc=f"{model_name} seeds", leave=False):
            set_global_seed(local_seed)

            # Duffing: Increase task consistency with 10 trajectories instead of 1
            duff_params = [0.24, 0.38, 0.52] if smoke_test else np.linspace(0.2, 0.55, 7).tolist()
            duff_data = generate_duffing_dataset(duff_params, num_trajectories=1 if smoke_test else 10, num_points=56 if smoke_test else 160, t_span=(0.0, 16.0), seed=local_seed)
            duff_obs, duff_coords, duff_targets, duff_params_t, duff_meta, duff_coord_norm, duff_state_norm = prepare_ode_tensors(duff_data, observation_steps=8 if smoke_test else 20)

            # Allen-Cahn: Increase samples from 1 to 8 instead of 1
            allen_params = [0.8, 1.1] if smoke_test else [0.75, 0.9, 1.05, 1.2]
            allen_data = generate_allen_cahn_dataset(allen_params, num_samples=1 if smoke_test else 8, nx=16 if smoke_test else 48, nt=16 if smoke_test else 36, seed=local_seed + 11)
            allen_obs, allen_coords, allen_targets, allen_params_t, allen_coord_norm, allen_state_norm = prepare_pde_tensors(allen_data, observation_steps=4 if smoke_test else 8)

            m_duffing = _eval_model_on_dataset(
                model_name, "duffing", duff_data, duff_obs, duff_coords, duff_targets, duff_params_t, duff_meta, "duffing",
                device, smoke_test, local_seed, max_epochs, callbacks,
                coord_normalizer=duff_coord_norm, state_normalizer=duff_state_norm,
            )
            m_allen   = _eval_model_on_dataset(
                model_name, "allen_cahn", allen_data, allen_obs, allen_coords, allen_targets, allen_params_t, None, "allen_cahn",
                device, smoke_test, local_seed + 101, max_epochs, callbacks,
                coord_normalizer=allen_coord_norm, state_normalizer=allen_state_norm,
            )

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

    payload: dict[str, object] = {
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
