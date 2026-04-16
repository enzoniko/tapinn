from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

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
from .common import (
    _aggregate_ode_metrics,
    _aggregate_pde_metrics,
    _condition_plot,
    _final_conditioning_summary_plot,
    _split_indices_three_way,
    _spectrum_plot,
    _subset_optional_tensor,
    _subset_tensors,
    tqdm,
)


def _finite_ntk_and_condition(model: torch.nn.Module, model_kind: str, observations: torch.Tensor, coords: torch.Tensor, params: torch.Tensor, device: torch.device):
    """Compute the NTK spectrum and Jacobian condition number for a small batch.

    Robustness Fix: Ensures all model parameters and input tensors are pinned to
    the same compute device to avoid 'RuntimeError: Input and parameter tensors
    are not at the same device'.
    """
    diag_samples = min(4, observations.shape[0])
    points_per_sample = max(1, min(6, coords.shape[1]))
    jac_rows = []

    model.to(device)
    model.train()

    with torch.enable_grad():
        if model_kind == "tapinn":
            obs_batch = observations[:diag_samples].to(device)
            latent_batch = model.encode(obs_batch)  # type: ignore

            target_params = [p for p in model.parameters() if p.requires_grad]

            for sample_idx in range(diag_samples):
                sample_coords = coords[sample_idx, :points_per_sample, :].to(device)
                sample_latent = latent_batch[sample_idx: sample_idx + 1]

                for idx in range(sample_coords.shape[0]):
                    coord = sample_coords[idx: idx + 1].detach().clone().requires_grad_(True)
                    pred = model.decode(coord, sample_latent)  # type: ignore
                    scalar = pred[0, 0]

                    grads = torch.autograd.grad(
                        scalar, target_params,
                        retain_graph=True, create_graph=False, allow_unused=True
                    )
                    grad_vec = torch.cat([g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                                          for g, p in zip(grads, target_params)])
                    jac_rows.append(grad_vec)
        else:
            target_params = [p for p in model.parameters() if p.requires_grad]
            for sample_idx in range(diag_samples):
                sample_coords = coords[sample_idx, :points_per_sample, :].to(device)
                sample_param = params[sample_idx: sample_idx + 1].to(device)

                for idx in range(sample_coords.shape[0]):
                    coord = sample_coords[idx: idx + 1].detach().clone().requires_grad_(True)
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

    eigvals = torch.linalg.eigvalsh(ntk).detach().cpu().numpy()
    eigvals = np.sort(np.clip(eigvals.real, 1e-12, None))[::-1]

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

    seeds = [seed] if smoke_test else [seed + offset for offset in range(3)]
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
            test_obs, test_coords, test_targets, test_params = _subset_tensors(test_idx, obs, coords, targets, params)
            (
                coord_norm, state_norm,
                train_targets, _, test_targets,
                train_obs, _, test_obs,
            ) = refit_normalizers_on_physical_split(
                original_coord_norm=coord_norm,
                original_state_norm=state_norm,
                train_coords_normed=train_coords,
                train_targets_normed=train_targets,
                train_obs_normed=train_obs,
                val_targets_normed=torch.zeros_like(train_targets[0:1]),
                val_obs_normed=torch.zeros_like(train_obs[0:1]),
                test_targets_normed=test_targets,
                test_obs_normed=test_obs,
            )
            train_meta = _subset_optional_tensor(train_idx, meta)
            test_meta = _subset_optional_tensor(test_idx, meta)

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

                if family == "tapinn":
                    preds = predict_tapinn(model, test_obs, test_coords, device, state_normalizer=state_norm).cpu().numpy()
                else:
                    preds = predict_direct(model, "standard", test_obs, test_coords, test_params, device, state_normalizer=state_norm).cpu().numpy()

                truth_phys = state_norm.denormalize(test_targets).cpu().numpy()
                if problem_name in ["duffing", "kuramoto", "lorenz"]:
                    d_mse, p_res = _aggregate_ode_metrics(problem_name, data.times, preds, truth_phys, test_params, test_meta)
                else:
                    boundary = str(data.metadata.get("boundary", "periodic")) if data.metadata else "periodic"
                    d_mse, p_res = _aggregate_pde_metrics(problem_name, data.times, data.space, preds, truth_phys, test_params, boundary)

                seed_rows.append({"problem": problem_name, "seed": local_seed, "model": model_name, "data_mse": d_mse, "physics_residual": p_res})

        _spectrum_plot(spectrum_records, f"NTK {problem_name.title()} Over Training", fig_subdir / "ntk_spectrum.pdf", problem_name)
        _condition_plot(condition_records, f"Conditioning {problem_name.title()} Over Training", fig_subdir / "condition.pdf", problem_name)

    _final_conditioning_summary_plot(condition_records, run_dir / "figures" / "global_conditioning_summary.pdf")
    write_csv(run_dir / "tables" / "seed_summary.csv", seed_rows)
    write_csv(run_dir / "tables" / "summary.csv", seed_rows)
    write_json(run_dir / "results.json", {
        "device": str(device), "max_epochs": total_epochs,
        "seed_summary": seed_rows, "spectra": spectrum_records, "conditioning": condition_records
    })
    return {"problems": problems}
