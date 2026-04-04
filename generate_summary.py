import json
from pathlib import Path

def load_json(p):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {p}: {e}")
        return {}

def format_scientific(val):
    if val is None: return "N/A"
    try:
        if abs(val) < 1e-2 or abs(val) > 1e4:
            return f"{val:.4e}"
        return f"{val:.4f}"
    except:
        return str(val)

d1 = load_json("neurips_results/exp_1_ode_chaos_suite/results.json")
d2 = load_json("neurips_results/exp_2_pde_spatiotemporal_suite/results.json")
d3 = load_json("neurips_results/exp_3_sota_baselines_and_capacity/results.json")
d4 = load_json("neurips_results/exp_4_sensitivity_and_robustness/results.json")
d5 = load_json("neurips_results/exp_5_theoretical_optimization_landscape/results.json")

output_sections = ["\n\n## 3. Quantitative Results Appendix (Auto-Generated)\n"]

# --- Exp 1: ODE Chaos Suite ---
if "summary" in d1:
    lines = ["#### Exp 1: ODE Chaos Suite Metrics", 
             "| Problem | Model | Data MSE | Physics Residual |",
             "|:---|:---|:---|:---|"]
    for r in d1["summary"]:
        lines.append(f"| {r['problem'].title()} | {r['model']} | {format_scientific(r['data_mse_mean'])} | {format_scientific(r['physics_residual_mean'])} |")
    output_sections.append("\n".join(lines))

# --- Exp 2: PDE Spatiotemporal Suite ---
if "summary" in d2:
    lines = ["#### Exp 2: PDE Spatiotemporal Suite Metrics",
             "| Problem | Model | Data MSE | Physics Residual |",
             "|:---|:---|:---|:---|"]
    for r in d2["summary"]:
        lines.append(f"| {r['problem'].title()} | {r['model']} | {format_scientific(r['data_mse_mean'])} | {format_scientific(r['physics_residual_mean'])} |")
    output_sections.append("\n".join(lines))

# --- Exp 3: Capacity & Baselines ---
if "summary" in d3:
    lines = ["#### Exp 3: Capacity & Baselines Metrics",
             "| Model | Params | Data MSE | Physics Residual | Gen. Gap |",
             "|:---|:---|:---|:---|:---|"]
    for r in d3["summary"]:
        m_name = r.get('model_name', r.get('model', 'N/A'))
        lines.append(f"| {m_name} | {r.get('param_count', 'N/A')} | {format_scientific(r.get('data_mse_mean'))} | {format_scientific(r.get('physics_residual_mean'))} | {format_scientific(r.get('generalization_gap_mean'))} |")
    output_sections.append("\n".join(lines))

# --- Exp 4: Robustness (Window Sweep) ---
if "window_sweep" in d4:
    lines = ["#### Exp 4: Robustness Metrics (Window Sweep)",
             "| Window Size | Problem | Model | Forecast Error |",
             "|:---|:---|:---|:---|"]
    for r in d4["window_sweep"]:
        lines.append(f"| {r.get('observed_steps', 'N/A')} | {r.get('problem', 'N/A')} | {r.get('model', 'N/A')} | {format_scientific(r.get('forecast_error_mean'))} |")
    output_sections.append("\n".join(lines))

# --- Exp 5: Optimization Landscape ---
if "conditioning" in d5:
    lines = ["#### Exp 5: Theoretical Optimization Landscape",
             "| Model | Mean κ(J) | Max NTK λ | Mean Lipschitz L |",
             "|:---|:---|:---|:---|"]
    
    models = sorted(list(set(x["model"] for x in d5["conditioning"])))
    
    # Pre-map spectra for easy access
    model_to_ntk = {}
    if "spectra" in d5:
        for s in d5["spectra"]:
            m = s["model"]
            if m not in model_to_ntk: model_to_ntk[m] = []
            if s.get("eigenvalues"):
                model_to_ntk[m].append(max(s["eigenvalues"]))

    for model in models:
        m_conds = [x["condition_number"] for x in d5["conditioning"] if x["model"] == model and x.get("condition_number") is not None]
        m_lips = [x["lipschitz"] for x in d5["conditioning"] if x["model"] == model and x.get("lipschitz") is not None]
        
        avg_cond = sum(m_conds)/len(m_conds) if m_conds else 0
        avg_lip = sum(m_lips)/len(m_lips) if m_lips else 0
        
        ntk_list = model_to_ntk.get(model, [])
        max_ntk = sum(ntk_list)/len(ntk_list) if ntk_list else 0
        
        lines.append(f"| {model} | {format_scientific(avg_cond)} | {format_scientific(max_ntk)} | {format_scientific(avg_lip)} |")
    output_sections.append("\n".join(lines))

# Write to file
p = Path("NEURIPS_MASTER_SUMMARY.md")
content = p.read_text()

if "## 3. Quantitative Results Appendix" in content:
    content = content.split("## 3. Quantitative Results Appendix")[0].strip()

new_content = content + "\n" + "\n\n".join(output_sections)
p.write_text(new_content)

print(f"Master Summary successfully updated at {p.absolute()}")
