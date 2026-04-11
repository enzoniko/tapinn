from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from exp_common.experiments import _eval_model_on_dataset
from exp_common.device import get_best_device
from exp_common.metrics import numerical_ode_residual, numerical_pde_residual
from exp_common.models import DeepONet, HyperPINN, LightweightFNO1d, build_capacity_matched_hyperpinn, build_tapinn, count_parameters
from exp_common.problems import compute_ode_residual, compute_pde_residual, generate_allen_cahn_dataset, generate_duffing_dataset, generate_kuramoto_dataset
from exp_common.repro import set_global_seed
from exp_common.trainers import (
    CallbackConfig,
    EarlyStopping,
    ValBundle,
    prepare_ode_tensors,
    prepare_pde_tensors,
    train_tapinn,
    train_direct_model,
)
from exp_common.models import StandardPINN


REPO_ROOT = Path(__file__).resolve().parents[1]


class ExperimentSuiteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_root = Path(tempfile.mkdtemp(prefix="neurips_suite_tests_"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_root, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Utility / infrastructure                                             #
    # ------------------------------------------------------------------ #

    def test_device_and_seed_helpers(self) -> None:
        set_global_seed(7)
        device = get_best_device("cpu")
        self.assertEqual(str(device), "cpu")

    def test_model_forward_passes(self) -> None:
        tapinn = build_tapinn(obs_dim=2, coord_dim=1, output_dim=2, large=False)
        obs = torch.randn(2, 5, 2)
        coords = torch.randn(2, 12, 1)
        tapinn_out, latent = tapinn(obs, coords[:, 0, :])
        self.assertEqual(tapinn_out.shape, (2, 2))
        self.assertEqual(latent.shape[0], 2)

        hyper = HyperPINN(coord_dim=1, output_dim=2, hidden_dim=16)
        hyper_out = hyper(torch.randn(2, 1), torch.randn(2))
        self.assertEqual(hyper_out.shape, (2, 2))

        deeponet = DeepONet(branch_input_dim=10, coord_dim=2, output_dim=1, hidden_dim=16, basis_dim=8)
        deep_out = deeponet(torch.randn(3, 10), torch.randn(3, 2))
        self.assertEqual(deep_out.shape, (3, 1))

        fno = LightweightFNO1d(branch_input_dim=10, grid_size=12, output_dim=1, width=8, modes=4)
        fno_out = fno(torch.randn(2, 10), torch.randn(2, 12, 1))
        self.assertEqual(fno_out.shape, (2, 12, 1))

    def test_capacity_targets_are_reasonable(self) -> None:
        small_tapinn = count_parameters(build_tapinn(obs_dim=2, coord_dim=1, output_dim=2, large=False))
        large_tapinn = count_parameters(build_tapinn(obs_dim=2, coord_dim=1, output_dim=2, large=True))
        hyper = count_parameters(build_capacity_matched_hyperpinn(coord_dim=1, output_dim=2))
        self.assertTrue(7500 <= small_tapinn <= 9500)
        self.assertTrue(36000 <= large_tapinn <= 43000)
        self.assertTrue(36000 <= hyper <= 43000)

    # ------------------------------------------------------------------ #
    # Callback unit tests (Exp-1 specific)                                 #
    # ------------------------------------------------------------------ #

    def test_early_stopping_triggers_correctly(self) -> None:
        """EarlyStopping should trigger after exactly patience epochs without improvement."""
        es = EarlyStopping(patience=3, min_delta=1e-4)
        # Epoch 0: improvement
        self.assertFalse(es.step(1.0))
        # Epochs 1-3: no improvement (patience = 3 → triggers on step 4)
        self.assertFalse(es.step(1.1))
        self.assertFalse(es.step(1.1))
        triggered = es.step(1.1)
        self.assertTrue(triggered)
        self.assertTrue(es.triggered)

    def test_early_stopping_resets_on_improvement(self) -> None:
        """Counter should reset when loss improves, delaying trigger."""
        es = EarlyStopping(patience=2)
        es.step(1.0)   # best = 1.0, counter = 0
        es.step(1.1)   # no improvement, counter = 1
        es.step(0.9)   # improvement!  counter resets to 0
        self.assertFalse(es.triggered)
        self.assertEqual(es.counter, 0)

    def test_val_bundle_callbacks_reduce_epochs(self) -> None:
        """With EarlyStopping active, training stops before max_epochs when loss plateaus."""
        set_global_seed(42)
        device = torch.device("cpu")
        # Tiny dataset: 6 trajectories of a Duffing oscillator
        data = generate_duffing_dataset([0.3, 0.5], num_trajectories=3, num_points=24, t_span=(0.0, 4.0), seed=42)
        obs, coords, targets, params, ode_meta, coord_norm, state_norm = prepare_ode_tensors(data, observation_steps=6)
        # 80% train, 20% val (no test split needed here)
        n_val = max(1, int(0.2 * len(params)))
        train_obs, val_obs = obs[:-n_val], obs[-n_val:]
        train_coords, val_coords = coords[:-n_val], coords[-n_val:]
        train_targets, val_targets = targets[:-n_val], targets[-n_val:]
        train_params, val_params = params[:-n_val], params[-n_val:]

        val_bundle = ValBundle(
            observations=val_obs,
            coords=val_coords,
            targets=val_targets,
            params=val_params,
        )
        # Use very low patience so the test is fast (ceiling 20 epochs, stops much earlier)
        callbacks = CallbackConfig(early_stopping_patience=2, reduce_lr_patience=1)
        model = build_tapinn(obs_dim=2, coord_dim=1, output_dim=2, large=False)
        result = train_tapinn(
            model,
            problem_name="duffing",
            observations=train_obs,
            coords=train_coords,
            targets=train_targets,
            params=train_params,
            device=device,
            epochs=20,  # ceiling -- EarlyStopping should fire before this
            batch_size=4,
            val_bundle=val_bundle,
            callbacks=callbacks,
        )
        # With patience=2, training must stop well before 20 epochs
        self.assertLess(result.epochs_trained, 20)
        self.assertGreater(result.epochs_trained, 0)

    def test_train_result_has_epochs_trained(self) -> None:
        """TrainResult should always carry epochs_trained field."""
        set_global_seed(1)
        device = torch.device("cpu")
        data = generate_duffing_dataset([0.3], num_trajectories=2, num_points=16, t_span=(0.0, 2.0), seed=1)
        obs, coords, targets, params, _, coord_norm, state_norm = prepare_ode_tensors(data, observation_steps=4)
        model = build_tapinn(obs_dim=2, coord_dim=1, output_dim=2, large=False)
        result = train_tapinn(model, "duffing", obs, coords, targets, params,
                              device=device, epochs=3, batch_size=2)
        self.assertEqual(result.epochs_trained, 3)

    def test_coords_are_normalized_to_minus1_1(self) -> None:
        """Coordinates returned by prepare_ode_tensors must be in [-1, 1] to prevent Tanh saturation."""
        data = generate_duffing_dataset([0.3], num_trajectories=2, num_points=200, t_span=(0.0, 16.0), seed=7)
        _, coords, _, _, _, coord_norm, state_norm = prepare_ode_tensors(data, observation_steps=4)
        self.assertAlmostEqual(float(coords.min()), -1.0, places=4,
                               msg="Min coordinate should be -1.0 after normalization")
        self.assertAlmostEqual(float(coords.max()),  1.0, places=4,
                               msg="Max coordinate should be +1.0 after normalization")

    def test_state_normalizer_round_trips_lorenz_scale(self) -> None:
        """StateNormalizer must exactly round-trip Lorenz-scale states (high magnitude)."""
        from exp_common.trainers import StateNormalizer
        y_raw = torch.randn(50, 3) * 20  # Lorenz scale O(20)
        sn = StateNormalizer.from_targets(y_raw)
        y_norm = sn.normalize(y_raw)
        y_back = sn.denormalize(y_norm)
        round_trip_err = float((y_back - y_raw).abs().max())
        self.assertLess(round_trip_err, 1e-5, msg="StateNormalizer round-trip error too large")
        norm_min, norm_max = float(y_norm.min()), float(y_norm.max())
        self.assertAlmostEqual(norm_min, -1.0, places=4)
        self.assertAlmostEqual(norm_max,  1.0, places=4)

    # ------------------------------------------------------------------ #
    # Physics / residual correctness                                       #
    # ------------------------------------------------------------------ #

    def test_residual_functions_are_finite(self) -> None:
        coords_ode = torch.linspace(0.0, 1.0, 8).unsqueeze(1).requires_grad_(True)
        y_pred = torch.cat([coords_ode, coords_ode**2], dim=1)
        ode_residual = compute_ode_residual("duffing", coords_ode, y_pred, torch.full((8,), 0.3))
        self.assertTrue(torch.isfinite(ode_residual).all())

        coords_pde = torch.randn(10, 2, requires_grad=True)
        u_pred = (coords_pde[:, 0] + coords_pde[:, 1] ** 2)
        pde_residual = compute_pde_residual("allen_cahn", coords_pde, u_pred, torch.full((10,), 1.0))
        self.assertTrue(torch.isfinite(pde_residual).all())

    def test_kuramoto_metadata_is_preserved_and_used(self) -> None:
        data = generate_kuramoto_dataset([1.0], num_trajectories=1, num_points=96, t_span=(0.0, 10.0), num_oscillators=5, seed=11)
        observations, coords, targets, params, ode_metadata, coord_norm, state_norm = prepare_ode_tensors(data, observation_steps=12)
        self.assertIsNotNone(ode_metadata)
        self.assertEqual(tuple(ode_metadata.shape), (1, 5))
        self.assertTrue(torch.allclose(ode_metadata[0], torch.tensor(data.metadata["natural_frequencies"][0], dtype=torch.float32)))

        residual = numerical_ode_residual(
            "kuramoto",
            data.times,
            data.states[0],
            float(data.params[0]),
            metadata={"natural_frequencies": data.metadata["natural_frequencies"][0]},
        )
        self.assertLess(residual, 0.75)

        with self.assertRaises(ValueError):
            numerical_ode_residual("kuramoto", data.times, data.states[0], float(data.params[0]))

    def test_numerical_pde_residual_depends_on_true_grid(self) -> None:
        times = torch.linspace(0.0, 1.0, 12).numpy()
        space = torch.linspace(0.0, 2.0 * torch.pi, 48).numpy()
        mesh_t, mesh_x = torch.meshgrid(torch.tensor(times), torch.tensor(space), indexing="ij")
        field = torch.sin(mesh_x).numpy()
        correct = numerical_pde_residual("burgers", times, space, field, param=0.05, boundary="periodic")
        wrong = numerical_pde_residual("burgers", times, torch.linspace(-1.0, 1.0, 48).numpy(), field, param=0.05, boundary="periodic")
        self.assertGreater(abs(correct - wrong), 1e-3)

    # ------------------------------------------------------------------ #
    # Exp-3 baseline check (HyperPINN still rejects duplicates there)      #
    # ------------------------------------------------------------------ #

    def test_param_only_baselines_reject_duplicate_parameter_tasks(self) -> None:
        """_eval_model_on_dataset (Exp 3) still rejects HyperPINN with duplicate params."""
        data = generate_duffing_dataset([0.3], num_trajectories=3, num_points=24, t_span=(0.0, 4.0), seed=5)
        observations, coords, targets, params, ode_metadata, coord_norm, state_norm = prepare_ode_tensors(data, observation_steps=6)
        with self.assertRaises(ValueError):
            _eval_model_on_dataset(
                "hyperpinn",
                "duffing",
                data,
                observations,
                coords,
                targets,
                params,
                ode_metadata,
                "duffing",
                torch.device("cpu"),
                True,
                5,
            )

    def test_prepare_pde_tensors_preserve_dataset_grid(self) -> None:
        data = generate_allen_cahn_dataset([0.9], num_samples=1, nx=20, nt=12, seed=3)
        _, coords, _, _, coord_norm, state_norm = prepare_pde_tensors(data, observation_steps=4)
        coord_sample = coords[0].cpu().numpy()
        # With normalization, both dims should have exactly nt and nx distinct values
        self.assertEqual(len(set(coord_sample[:, 0].tolist())), len(data.times))
        self.assertEqual(len(set(coord_sample[:, 1].tolist())), len(data.space))
        # Normalized to [-1, 1]
        self.assertAlmostEqual(float(coord_sample[:, 0].max()),  1.0, places=4)
        self.assertAlmostEqual(float(coord_sample[:, 0].min()), -1.0, places=4)
        self.assertAlmostEqual(float(coord_sample[:, 1].max()),  1.0, places=4)
        self.assertAlmostEqual(float(coord_sample[:, 1].min()), -1.0, places=4)

    # ------------------------------------------------------------------ #
    # Smoke-run all scripts                                                #
    # ------------------------------------------------------------------ #

    def test_all_scripts_smoke_run_and_emit_outputs(self) -> None:
        script_expectations = {
            "exp_1_ode_chaos_suite.py": [
                "exp_1_ode_chaos_suite/results.json",
                # legacy single-model tables/figures still present
                "exp_1_ode_chaos_suite/tables/summary_table.csv",
                "exp_1_ode_chaos_suite/tables/seed_metrics.csv",
                "exp_1_ode_chaos_suite/tables/model_summary.csv",
                "exp_1_ode_chaos_suite/figures/duffing_phase_space.pdf",
                # new multi-model figures
                "exp_1_ode_chaos_suite/figures/duffing_model_comparison.pdf",
                "exp_1_ode_chaos_suite/figures/duffing_data_mse_bar.pdf",
                "exp_1_ode_chaos_suite/figures/duffing_physics_residual_bar.pdf",
            ],
            "exp_2_pde_spatiotemporal_suite.py": [
                "exp_2_pde_spatiotemporal_suite/results.json",
                "exp_2_pde_spatiotemporal_suite/tables/summary.csv",
                "exp_2_pde_spatiotemporal_suite/tables/seed_metrics.csv",
                "exp_2_pde_spatiotemporal_suite/figures/allen_cahn_heatmap_triptych.pdf",
            ],
            "exp_3_sota_baselines_and_capacity.py": [
                "exp_3_sota_baselines_and_capacity/results.json",
                "exp_3_sota_baselines_and_capacity/tables/capacity_benchmark.csv",
                "exp_3_sota_baselines_and_capacity/tables/per_task_seed_metrics.csv",
                "exp_3_sota_baselines_and_capacity/figures/pareto_frontier.pdf",
            ],
            "exp_4_sensitivity_and_robustness.py": [
                "exp_4_sensitivity_and_robustness/results.json",
                "exp_4_sensitivity_and_robustness/tables/noise_sweep.csv",
                "exp_4_sensitivity_and_robustness/figures/error_vs_noise.pdf",
            ],
            "exp_5_theoretical_optimization_landscape.py": [
                "exp_5_theoretical_optimization_landscape/results.json",
                "exp_5_theoretical_optimization_landscape/tables/summary.csv",
                "exp_5_theoretical_optimization_landscape/tables/seed_summary.csv",
                "exp_5_theoretical_optimization_landscape/figures/ntk_spectrum.pdf",
            ],
        }

        for script, expected_paths in script_expectations.items():
            output_root = self.temp_root / script.replace(".py", "")
            result = subprocess.run(
                ["python", script, "--smoke-test", "--device", "cpu", "--output-root", str(output_root)],
                cwd=REPO_ROOT,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(
                result.returncode, 0,
                msg=f"{script} exited with code {result.returncode}.\nSTDERR:\n{result.stderr[-2000:]}"
            )
            for rel_path in expected_paths:
                self.assertTrue(
                    (output_root / rel_path).exists(),
                    msg=f"Missing output {rel_path} for {script}"
                )


if __name__ == "__main__":
    unittest.main()
