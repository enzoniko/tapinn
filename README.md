# TAPINN Experiment Suite

This repository contains the official research codebase for **Topology-Aware / observation-conditioned PINNs (TAPINN)**. The suite has been fully remediated and expanded into a rigorous 5-experiment benchmark for the **NeurIPS 2026 Main Track**.

## Project Status & Source of Truth

The Experiment Suite is now **remediated and finalized** for the NeurIPS 2026 Main Track.

**The single source of truth for the project is now:**
### [NEURIPS_MASTER_SUMMARY.md](file:///home/enzo/ICLR2026/siamesepinn/NEURIPS_MASTER_SUMMARY.md)

This document consolidates results from all 5 remediated experiments and should be used as the primary reference for paper writing.

---

## Installation

### Option 1: Conda (Recommended for CUDA support)
```bash
conda env create -f environment.yml
conda activate tapinn
```

### Option 2: Pip
```bash
pip install -r requirements.txt
```

---

## What This Repository Implements

### Shared experiment core

Under `exp_common/`:

- `device.py`: dynamic device fallback with `CUDA > MPS > CPU`
- `repro.py`: seeding and deterministic behavior helpers
- `io_utils.py`: output directories plus CSV/JSON writing
- `plotting.py`: publication-style PDF figure configuration and saving
- `metrics.py`: data MSE plus numerical ODE/PDE residual diagnostics
- `problems.py`: ODE/PDE generators and physics residual definitions
- `models.py`: TAPINN, Standard PINN, HyperPINN, DeepONet, lightweight FNO
- `trainers.py`: training loops, prediction utilities, tensor preparation
- `experiments.py`: experiment runners used by the standalone entrypoint scripts

### Standalone experiment scripts

- `exp_1_ode_chaos_suite.py`
- `exp_2_pde_spatiotemporal_suite.py`
- `exp_3_sota_baselines_and_capacity.py`
- `exp_4_sensitivity_and_robustness.py`
- `exp_5_theoretical_optimization_landscape.py`

Each script:

- generates its own data,
- initializes its own models,
- trains and evaluates end to end,
- writes CSV/JSON outputs under `neurips_results/`,
- saves paper-style PDF figures.

### Tests

- `tests/test_neurips_experiment_suite.py`

- multiple trajectories per parameter to test IC-sensitivity,
- EarlyStopping and ReduceLROnPlateau standardized across all models.

## Repository Layout

```text
siamesepinn/
├── exp_common/
│   ├── device.py
├── exp_common/          (Shared Logic: Models, Trainers, Metrics)
├── exp_1_ode_chaos_suite.py
├── exp_2_pde_spatiotemporal_suite.py
├── exp_3_sota_baselines_and_capacity.py
├── exp_4_sensitivity_and_robustness.py
├── exp_5_theoretical_optimization_landscape.py
├── neurips_results/     (Final PDFs, Tables, and JSON)
├── legacy/              (Archive of Workshop POCs and Old Logs)
└── tests/               (Verification Suite)
```

## Running the Suite

### Full experiments

```bash
python exp_1_ode_chaos_suite.py
python exp_2_pde_spatiotemporal_suite.py
python exp_3_sota_baselines_and_capacity.py
python exp_4_sensitivity_and_robustness.py
python exp_5_theoretical_optimization_landscape.py
```

### Smoke tests

```bash
python -m pytest tests/test_neurips_experiment_suite.py
```

### Run a single experiment in smoke-test mode

```bash
python exp_2_pde_spatiotemporal_suite.py --smoke-test --device cpu
```

### Run a single experiment in smoke-test mode

```bash
python exp_2_pde_spatiotemporal_suite.py --smoke-test --device cuda
```

## Experiment Summary

### `exp_1_ode_chaos_suite.py`

**Scientific role:**

This is the foundational ODE benchmark for the paper. Its specific responsibilities are:

1. **Grounding the method in its origin domain.** The Duffing oscillator is the workshop paper's central case. Exp 1 provides systematic ODE results that anchor the method's lineage.
2. **Challenge gradient across chaos levels.** Three systems of increasing difficulty (Duffing → Kuramoto → Lorenz) expose how each method's failure mode differs under chaos. Lorenz failure across **all** methods is scientifically informative — it shows true chaos is universally challenging (not a TAPINN weakness), which supports Exp 5's optimization analysis.
3. **ODE-specific comparative performance.** Exp 3 covers Duffing + one PDE. Exp 1 is the **only** experiment that tests all baselines on Lorenz and Kuramoto.
4. **Initial-condition sensitivity.** Multiple trajectories per parameter value directly test whether observation-conditioning (TAPINN, DeepONet) can distinguish trajectories that share a parameter but diverge from different initial conditions. HyperPINN and StandardPINN cannot — this is a key architectural finding.

**Baselines (all 5, same as Exp 3):**

| Model | Type | Config | ~Params (Duffing) |
|---|---|---|---|
| TAPINN | physics_trained | hidden=32, latent=8 | ~8.5K |
| StandardPINN | physics_trained | MLP(coord+param→state), hidden=64 | ~8.6K |
| HyperPINN | physics_trained | hypernetwork, hidden=32 | ~40K |
| DeepONet | physics_trained | branch+trunk, hidden=64, basis=32 | ~20K |
| FNO | supervised_only | width=32, modes=12 | ~30K |

Note: HyperPINN is significantly larger due to hypernetwork overhead. Parameter counts are reported transparently. HyperPINN is trained on data with duplicate parameter values (multiple trajectories per param) — its inability to distinguish them is an explicit architectural finding, not a data error.

**Dataset sizes (full run):**

| System | Params | Traj/Param | Total traj | Time span | Points |
|---|---|---|---|---|---|
| Duffing | 7 (linspace 0.2–0.55) | 20 | 140 | [0, 16] | 200 |
| Lorenz | 7 (linspace 18–35) | 12 | 84 | [0, 4.5] | 200 |
| Kuramoto | 7 (linspace 0.2–3.0) | 12 | 84 | [0, 10] | 200 |

Split: 70% train / 15% val / 15% test.

**Callbacks (identical for ALL models):**

```
EarlyStopping:    patience=15, min_delta=1e-6
ReduceLROnPlateau: patience=8, factor=0.5, min_lr=1e-6
Max epochs: 150
```

The same `CallbackConfig` instance is passed to every model training call. This is the fairness guarantee: no model receives more epochs, more LR reductions, or a different stopping criterion than any other.

**Outputs:**

```
tables/seed_metrics.csv         — per-seed, per-model, per-problem metrics
tables/model_summary.csv        — aggregated mean±std across seeds per model+problem
tables/summary_table.csv        — legacy TAPINN-only summary (backward compat)
figures/<problem>_model_comparison.pdf    — phase-space subplot grid, all models
figures/<problem>_phase_space.pdf         — legacy TAPINN single-model plot
figures/<problem>_data_mse_bar.pdf        — bar chart, data MSE per model
figures/<problem>_physics_residual_bar.pdf — bar chart, physics residual per model
figures/duffing_pareto_params_vs_residual.pdf — param count vs residual scatter
results.json                    — full structured results payload
```

**Recommendation:**

With comprehensive baselines and effort, Duffing results are usable as main-paper supplementary; Lorenz provides diagnostic evidence supporting Exp 5; Kuramoto is a qualified challenge case.

### `exp_2_pde_spatiotemporal_suite.py`

Purpose:

- TAPINN on Allen-Cahn, Burgers, and Kuramoto-Sivashinsky PDEs.

Current status:

- strongest clean empirical block in the repo,
- seeded summaries available,
- finite outputs and full-test aggregated reporting available.

Recommendation:

- keep in the main paper.

### `exp_3_sota_baselines_and_capacity.py`

Purpose:

- comparative benchmark across TAPINN, TAPINN-Large, HyperPINN, DeepONet, and lightweight FNO.

Current status:

- fairness improved relative to earlier versions,
- task-level metrics and seeded summaries available,
- comparison groups now explicitly distinguish `physics_trained` from `supervised_only`.

Important caveat:

- this is **not** a strict universal capacity-matching benchmark across all families.

Recommendation:

- keep in the main paper with explicit caveats.

### `exp_4_sensitivity_and_robustness.py`

Purpose:

- observation-window and noise robustness study.

Current status:

- the observation-window curve is informative,
- the noise-response remains too flat to support a strong robustness claim,
- targeted stress probes still recommend appendix-only treatment.

Recommendation:

- appendix or negative-result ablation only.

### `exp_5_theoretical_optimization_landscape.py`

Purpose:

- optimization, NTK-spectrum, and conditioning diagnostics for TAPINN AO vs joint vs standard PINN.

Current status:

- seeded summary available,
- better theory framing than earlier versions,
- scientifically useful as supporting evidence,
- not suitable as a pure predictive-accuracy headline.

Recommendation:

- keep in the main paper as supporting evidence.

## Important Scientific Repairs Already Applied

The current codebase incorporates several key fixes that were missing in weaker earlier snapshots:

- Kuramoto residuals now use the same per-trajectory natural frequencies as the generated data.
- PDE residual evaluation uses the true simulation grids rather than synthetic placeholder grids.
- Headline metrics aggregate across full held-out test sets rather than a single test sample.
- Experiments 2, 3, and 5 include seeded summaries.
- `exp_3` now avoids the earlier one-to-many baseline mismatch by using fairer task construction and explicit comparison groups.
- `exp_5` now uses a better-posed standard PINN comparison protocol.
- **Exp 1 redesign (current):** All 5 baselines added; dataset sizes tripled; 3-way train/val/test splits; EarlyStopping and ReduceLROnPlateau applied identically to all models; 150 max epochs; new multi-model comparison figures and tables. `TrainResult` now carries `epochs_trained` and `best_val_loss`.

### Full experiment outputs

Each experiment writes to `neurips_results/<experiment_name>/` and includes:

- `results.json`: Full structured results payload.
- `tables/`: Per-seed and aggregated summary CSVs.
- `figures/`: Problem-specific PDFs (Spectra, Phase-Space, Metrics).

A consolidated walkthrough of all findings is available in `neurips_results/walkthroughs/`.

## Claim Boundaries

The current codebase is compatible with a **narrow, careful, main-track submission**, but not with broad claims such as:

- broad superiority on chaotic ODEs,
- strong observation-noise robustness,
- strict capacity-matched fairness across all baseline families,
- universal predictive dominance of TAPINN.

Defensible claim style:

- TAPINN is a reproducible observation-conditioned physics-learning framework.
- It is scientifically sound and competitive on selected parameterized ODE/PDE settings.
- Its optimization and conditioning behavior are informative even when predictive gains are mixed.

## Summary of Submissions Goals
The current codebase is directly optimized for a **NeurIPS 2026 Main Track submission**, featuring a full 5-experiment suite that covers empirical performance, baseline comparison, and theoretical optimization analysis.

For a definitive overview of the current results, refer to **[NEURIPS_MASTER_SUMMARY.md](file:///home/enzo/ICLR2026/siamesepinn/NEURIPS_MASTER_SUMMARY.md)**.
