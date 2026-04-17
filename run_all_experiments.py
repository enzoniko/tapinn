"""Unified runner for all 5 NeurIPS experiments.

Usage:
    python run_all_experiments.py --quick --device cpu    # Quick validation
    python run_all_experiments.py --device cuda            # Full run on CUDA
"""
from __future__ import annotations

import argparse
import sys
import time

from exp_common.experiments import (
    run_exp_1_ode_chaos_suite,
    run_exp_2_pde_spatiotemporal_suite,
    run_exp_3_sota_baselines_and_capacity,
    run_exp_4_sensitivity_and_robustness,
    run_exp_5_theoretical_optimization_landscape,
)


_EXPERIMENTS = [
    ("Exp 1: ODE Chaos Suite", run_exp_1_ode_chaos_suite),
    ("Exp 2: PDE Spatiotemporal Suite", run_exp_2_pde_spatiotemporal_suite),
    ("Exp 3: Baselines & Capacity", run_exp_3_sota_baselines_and_capacity),
    ("Exp 4: Sensitivity & Robustness", run_exp_4_sensitivity_and_robustness),
    ("Exp 5: Optimization Landscape", run_exp_5_theoretical_optimization_landscape),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all 5 NeurIPS experiments.")
    parser.add_argument("--quick", action="store_true", help="Quick validation: smoke-test mode.")
    parser.add_argument("--output-root", default="./neurips_results", help="Output directory.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    args = parser.parse_args()

    smoke_test = args.quick
    failed: list[str] = []
    total_start = time.time()

    for name, run_fn in _EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        start = time.time()
        try:
            run_fn(
                output_root=args.output_root,
                device_name=args.device,
                smoke_test=smoke_test,
                seed=args.seed,
            )
            elapsed = time.time() - start
            print(f"  [OK] {name} completed in {elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.time() - start
            print(f"  [FAIL] {name} FAILED after {elapsed:.1f}s: {exc}")
            failed.append(name)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  All experiments finished in {total_elapsed:.1f}s")
    if failed:
        print(f"  FAILED ({len(failed)}/{len(_EXPERIMENTS)}):")
        for name in failed:
            print(f"    - {name}")
        return 1
    print(f"  ALL PASSED ({len(_EXPERIMENTS)}/{len(_EXPERIMENTS)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
