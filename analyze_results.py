import json
import pandas as pd
from pathlib import Path

def load_results(path):
    p = Path(path)
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return {}

print("\n--- EXP 1: ODE Chaos Suite ---")
r1 = load_results("neurips_results/exp_1_ode_chaos_suite/results.json")
if r1 and "summary" in r1:
    df1 = pd.DataFrame(r1["summary"])
    print(df1.groupby(["problem", "model"])[["data_mse_mean", "physics_residual_mean"]].mean().to_string())

print("\n--- EXP 2: PDE Spatiotemporal Suite ---")
r2 = load_results("neurips_results/exp_2_pde_spatiotemporal_suite/results.json")
if r2 and "summary" in r2:
    df2 = pd.DataFrame(r2["summary"])
    print(df2.groupby(["problem", "model"])[["data_mse_mean", "physics_residual_mean"]].mean().to_string())

print("\n--- EXP 3: Baselines & Capacity ---")
r3 = load_results("neurips_results/exp_3_sota_baselines_and_capacity/results.json")
if r3 and "summary" in r3:
    df3 = pd.DataFrame(r3["summary"])
    print(df3[["model_name", "param_count", "data_mse_mean", "physics_residual_mean"]].to_string())

print("\n--- EXP 4: Sensitivity & Robustness ---")
r4 = load_results("neurips_results/exp_4_sensitivity_and_robustness/results.json")
if r4 and "summary" in r4:
    if isinstance(r4["summary"], list):
        df4 = pd.DataFrame(r4["summary"])
        print(df4.to_string())
    elif isinstance(r4["summary"], dict):
        for k, v in r4["summary"].items():
            print(f"Condition: {k}")
            print(pd.DataFrame(v).to_string())

print("\n--- EXP 5: Theoretical Optimization Landscape ---")
r5 = load_results("neurips_results/exp_5_theoretical_optimization_landscape/results.json")
if r5 and "conditioning" in r5:
    df5 = pd.DataFrame(r5["conditioning"])
    print(df5.groupby("model")[["cond_final", "ntk_max_final", "lipschitz_final"]].mean().to_string())
