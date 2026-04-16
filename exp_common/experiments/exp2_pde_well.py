from __future__ import annotations
# pyright: reportArgumentType=false

from typing import Any

import numpy as np
import torch

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
from ..plotting import configure_plotting
from ..problems import (
    generate_allen_cahn_dataset,
    generate_burgers_dataset,
    generate_kuramoto_sivashinsky_dataset,
)
from ..repro import set_global_seed
from ..trainers import (
    CallbackConfig,
    CoordNormalizer,
    StateNormalizer,
    ValBundle,
    predict_direct,
    predict_tapinn,
    prepare_pde_tensors,
    refit_normalizers_on_physical_split,
    train_direct_model,
    train_fno_model,
    train_tapinn,
)
from .common import tqdm, _aggregate_pde_metrics, _fno_predict_numpy, _heatmap_triptych, _measure_inference_ms, _split_indices_three_way, _subset_tensors
from .exp1_ode_chaos import _metrics_bar_chart


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
) -> dict[str, Any]:
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
) -> tuple[np.ndarray[Any, Any], float]:
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
) -> dict[str, dict[str, object]]:
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
    num_samples_per_param = 1 if smoke_test else 24
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
    results: dict[str, dict[str, object]] = {}
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
    max_epochs = 4 if smoke_test else 10000
    callbacks = CallbackConfig(
        early_stopping_patience=2 if smoke_test else 50,
        reduce_lr_patience=1 if smoke_test else 20,
        reduce_lr_factor=0.5,
        min_lr=1e-6,
    )
    model_names = list(_EXP2_MODELS.keys())

    seed_rows: list[dict[str, object]] = []
    problem_model_summaries: list[dict[str, object]] = []
    heatmap_data: dict[str, dict[str, object]] = {}  # problem -> TAPINN info (first seed)

    for problem_name, param_values in tqdm(configs.items(), desc="PDE systems", leave=False):
        per_model_seed_data: dict[str, list[dict[str, object]]] = {m: [] for m in model_names}

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
    payload: dict[str, object] = {
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
