# NEURIPS 2026 MASTER SUMMARY: TAPINN Evaluation

This document provides the canonical results for the **Topology-Aware PINN (TAPINN)** remediated experiment suite (Experiments 1–5), replacing all previous intermediate summaries and legacy documentation.

## Strategic Overview (NeurIPS Main Track)
The submission narrative is centered on **Observation-Conditioned Physics Learning**. We demonstrate that TAPINN is a robust operator-learning framework that outperforms standard and parameter-only PINNs in spatiotemporal operator tasks, with a specific theoretical advantage in optimization stability provided by **Alternating Optimization (AO)**.

---

## 1. Experiment Results Summary

### Exp 1: ODE Chaos Suite (Duffing, Kuramoto, Lorenz)
- **Finding**: TAPINN successfully handles early-stage chaos (Duffing, Kuramoto) with high fidelity. In extreme chaos (Lorenz), TAPINN models fail gracefully alongside all baselines, exposing the universal difficulty of the benchmark.
- **Arch. Advantage**: TAPINN uses the observation window to distinguish divergent trajectories sharing parameters, which Standard and HyperPINNs cannot do.

### Exp 2: PDE Spatiotemporal Suite (Allen-Cahn, Burgers, KS)
- **Finding**: TAPINN's strongest performance is in **Spatiotemporal PDEs**. It consistently achieves 1–2 orders of magnitude lower MSE than Standard PINNs by anchoring logic in snapshots.
- **Status**: Headline empirical result for the Main Track.

### Exp 3: SOTA Baselines & Capacity Matching
- **Finding**: TAPINN remains competitive with **DeepONet** and **FNO** while maintaining physical consistency (via the PDE loss) that FNO lacks.
- **Parameter Capacity**: TAPINN models are typically more efficient (10k–30k params) than large Baselines.

### Exp 4: Robustness Stress Probes
- **Finding**: TAPINN is robust to the **placement of the observation window**, but sensitivity to uniform noise remains a secondary finding (to be presented in the Appendix).

### Exp 5: Theoretical Optimization Landscape (NTK Analysis)
- **The "AO" Stabilizer**: Alternating Optimization keeps the Jacobian condition number stable (**~10^4 to 10^7**) across all 5 case studies.
- **Joint Optimization Failure**: Attempting to train the TAPINN split architecture jointly causes a condition number explosion (**~10^{10}**) in chaotic systems. This is the paper's primary theoretical justification for AO.

---

## 2. Directory Navigation

| Category | Location | Purpose |
| :--- | :--- | :--- |
| **New Results** | `neurips_results/` | Final systematic results (JSON, Tables, PDFs). |
| **PDF Figures** | `neurips_results/exp_X/figures/` | Sub-directories per problem for easy reference. |
| **Detailed Walkthroughs** | `neurips_results/walkthroughs/` | Consolidated narratives for each experiment. |
| **Paper Draft** | `PAPER_NARRATIVE_DRAFT.md` | Strategic framing and drafting. |
| **Legacy Archives** | `legacy/` | Outdated Feb 2026 documents and old POCs. |

---

> [!IMPORTANT]
> **Source of Truth Check**: All documents dated before **March 25, 2026** (found in `legacy/docs/`) have been archived and should NOT be used for reference.
