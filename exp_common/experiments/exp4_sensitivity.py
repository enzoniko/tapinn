from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..device import get_best_device
from ..io_utils import prepare_run_dir, write_csv, write_json
from ..metrics import mean_std, mse
from ..models import LightweightFNO1d, build_tapinn
from ..plotting import configure_plotting
from ..problems import generate_allen_cahn_dataset, generate_duffing_dataset
from ..repro import set_global_seed
from ..trainers import CallbackConfig, ValBundle, predict_tapinn, prepare_ode_tensors, prepare_pde_tensors, refit_normalizers_on_physical_split, train_fno_model, train_tapinn
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

RobustnessTuple = tuple[Any, ...]


_EXP4_MODELS = {
    "tapinn": ("tapinn", "physics_trained"),
    "fno":    ("fno",    "supervised_only"),
}


def _eval_robustness_grid_point(
    model_name: str,
    problem_name: str,
    train_tuple: RobustnessTuple,  # (obs, coords, targets, params, meta)
    val_tuple: RobustnessTuple,
    test_tuple: RobustnessTuple,
    noise_sigma: float,
    device: torch.device,
    max_epochs: int,
    callbacks: CallbackConfig,
    seed: int,
) -> dict[str, Any]:
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
                    if len(tensors) == 7:
                        # ODE case: (obs, coords, targets, params, metadata, cn, sn)
                        meta_raw, orig_cn, orig_sn = tensors[4], tensors[5], tensors[6]
                    else:
                        # PDE case: (obs, coords, targets, params, cn, sn)
                        meta_raw, orig_cn, orig_sn = None, tensors[4], tensors[5]

                    train_idx, val_idx, test_idx = _split_indices_three_way(len(prms), local_seed)
                    t_obs, t_crds, t_trgs, t_prms = _subset_tensors(train_idx, obs, crds, trgs, prms)
                    v_obs, v_crds, v_trgs, v_prms = _subset_tensors(val_idx, obs, crds, trgs, prms)
                    s_obs, s_crds, s_trgs, s_prms = _subset_tensors(test_idx, obs, crds, trgs, prms)
                    (
                        c_norm, s_norm,
                        t_trgs, v_trgs, s_trgs,
                        t_obs, v_obs_new, s_obs_new,
                    ) = refit_normalizers_on_physical_split(
                        original_coord_norm=orig_cn,
                        original_state_norm=orig_sn,
                        train_coords_normed=t_crds,
                        train_targets_normed=t_trgs,
                        train_obs_normed=t_obs,
                        val_targets_normed=v_trgs,
                        val_obs_normed=v_obs,
                        test_targets_normed=s_trgs,
                        test_obs_normed=s_obs,
                    )

                    t_tup = (t_obs, t_crds, t_trgs, t_prms, _subset_optional_tensor(train_idx, meta_raw), s_norm, c_norm)
                    v_tup = (v_obs_new, v_crds, v_trgs, v_prms, _subset_optional_tensor(val_idx, meta_raw), s_norm, c_norm)
                    s_tup = (s_obs_new, s_crds, s_trgs, s_prms, _subset_optional_tensor(test_idx, meta_raw), s_norm, c_norm)

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
                    v_obs, v_crds, v_trgs, v_prms = _subset_tensors(val_idx, obs, crds, trgs, prms)
                    s_obs, s_crds, s_trgs, s_prms = _subset_tensors(test_idx, obs, crds, trgs, prms)
                    
                    (
                        c_norm, s_norm,
                        t_trgs, v_trgs, s_trgs,
                        t_obs, v_obs_new, s_obs_new,
                    ) = refit_normalizers_on_physical_split(
                        original_coord_norm=_cn,
                        original_state_norm=_sn,
                        train_coords_normed=t_crds,
                        train_targets_normed=t_trgs,
                        train_obs_normed=t_obs,
                        val_targets_normed=v_trgs,
                        val_obs_normed=v_obs,
                        test_targets_normed=s_trgs,
                        test_obs_normed=s_obs,
                    )

                    t_tup = (t_obs, t_crds, t_trgs, t_prms, _subset_optional_tensor(train_idx, meta), s_norm, c_norm)
                    v_tup = (v_obs_new, v_crds, v_trgs, v_prms, _subset_optional_tensor(val_idx, meta), s_norm, c_norm)
                    s_tup = (s_obs_new, s_crds, s_trgs, s_prms, _subset_optional_tensor(test_idx, meta), s_norm, c_norm)

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

    payload: dict[str, object] = {
        "device":      str(device),
        "smoke_test":  smoke_test,
        "max_epochs":  max_epochs,
        "noise_sweep": noise_rows,
        "window_sweep": window_rows,
    }
    write_json(run_dir / "results.json", payload)
    return payload
