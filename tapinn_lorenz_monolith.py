import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm
import time
from dataclasses import dataclass, field

# =============================================================================
# 1. Physics: Lorenz System
# =============================================================================

def lorenz_rhs_np(t, state, rho):
    x, y, z = state
    sigma = 10.0
    beta = 8.0 / 3.0
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

def lorenz_rhs_torch(t, y, rho):
    x = y[:, 0]
    yy = y[:, 1]
    z = y[:, 2]
    sigma = 10.0
    beta = 8.0 / 3.0
    rhs = torch.stack([
        sigma * (yy - x),
        x * (rho - z) - yy,
        x * yy - beta * z
    ], dim=1)
    return rhs

def _gradient(values, inputs):
    grad = torch.autograd.grad(
        values, inputs, grad_outputs=torch.ones_like(values),
        create_graph=True, retain_graph=True, allow_unused=True
    )[0]
    if grad is None:
        return torch.zeros_like(inputs)
    return grad

def compute_lorenz_residual(coords, y_pred, rho, coord_norm, state_norm):
    # 1. Denormalise time
    t_raw = coord_norm.denormalize(coords)[:, 0]
    t_scale = float(coord_norm.coord_scales[0].item())
    
    # 2. Compute dy/dt_norm
    derivatives = []
    for d in range(y_pred.shape[1]):
        derivatives.append(_gradient(y_pred[:, d], coords)[:, 0])
    dy_dt_norm = torch.stack(derivatives, dim=1)
    
    # 3. Chain rule for physical derivative
    dy_dt_raw = dy_dt_norm * t_scale
    
    # 4. Denormalise state for RHS evaluation
    y_raw = state_norm.denormalize(y_pred)
    y_raw = torch.clamp(y_raw, -1000, 1000) # Safety
    
    # 5. Eval RHS
    rhs_raw = lorenz_rhs_torch(t_raw, y_raw, rho)
    
    # 6. Renormalise RHS for consistent residual magnitude
    state_scale = (state_norm.state_maxs - state_norm.state_mins).clamp(min=1e-6) / 2.0
    rhs_norm = rhs_raw / state_scale.to(rhs_raw.device)
    
    return dy_dt_raw - rhs_norm

# =============================================================================
# 2. Model: TAPINN
# =============================================================================

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, depth=3):
        super().__init__()
        layers = []
        curr = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(curr, hidden_dim))
            layers.append(nn.Tanh())
            curr = hidden_dim
        layers.append(nn.Linear(curr, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class SequenceEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.proj(h[-1])

class TAPINN(nn.Module):
    def __init__(self, obs_dim, coord_dim, output_dim, hidden_dim=128, latent_dim=16):
        super().__init__()
        self.encoder = SequenceEncoder(obs_dim, 64, latent_dim)
        # Fourier Features: map coord to a higher-dim periodic basis
        self.B = nn.Parameter(torch.randn(coord_dim, 32) * 10.0, requires_grad=False)
        self.decoder = MLP(64 + latent_dim, output_dim, hidden_dim, depth=4)

    def encode(self, obs):
        return self.encoder(obs)

    def decode(self, coords, latent):
        # Apply Fourier features: [sin(2pi*B*t), cos(2pi*B*t)]
        proj = coords @ self.B
        coords_f = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        
        # Expand latent to match coords if needed
        if latent.dim() == 2 and coords.dim() == 3:
            latent = latent.unsqueeze(1).expand(-1, coords.shape[1], -1)
        
        # Flatten for MLP
        orig_shape = coords.shape
        inp = torch.cat([coords_f, latent], dim=-1)
        out = self.decoder(inp.reshape(-1, inp.shape[-1]))
        return out.reshape(orig_shape[0], orig_shape[1], -1)

    def forward(self, obs, coords):
        latent = self.encode(obs)
        preds = self.decode(coords, latent)
        return preds, latent

# =============================================================================
# 3. Normalization
# =============================================================================

class CoordNormalizer:
    def __init__(self, t_min, t_max):
        self.t_min, self.t_max = t_min, t_max
        self.coord_scales = torch.tensor([2.0 / (t_max - t_min + 1e-8)])
    def normalize(self, t):
        self.coord_scales = self.coord_scales.to(t.device)
        return (t - self.t_min) * self.coord_scales - 1.0
    def denormalize(self, tn):
        self.coord_scales = self.coord_scales.to(tn.device)
        return (tn + 1.0) / self.coord_scales + self.t_min

class StateNormalizer:
    def __init__(self, states):
        self.state_mins = torch.min(states, dim=0)[0]
        self.state_maxs = torch.max(states, dim=0)[0]
    def normalize(self, y):
        self.state_mins = self.state_mins.to(y.device)
        self.state_maxs = self.state_maxs.to(y.device)
        scale = (self.state_maxs - self.state_mins).clamp(min=1e-8) / 2.0
        return (y - self.state_mins) / scale - 1.0
    def denormalize(self, yn):
        self.state_mins = self.state_mins.to(yn.device)
        self.state_maxs = self.state_maxs.to(yn.device)
        scale = (self.state_maxs - self.state_mins).clamp(min=1e-8) / 2.0
        return (yn + 1.0) * scale + self.state_mins

# =============================================================================
# 4. Data Generation
# =============================================================================

def get_lorenz_data(rho_values, num_traj, nt, t_span, seed=42):
    rng = np.random.default_rng(seed)
    t_eval = np.linspace(t_span[0], t_span[1], nt)
    all_states, all_params = [], []
    for rho in rho_values:
        for _ in range(num_traj):
            y0 = rng.uniform(-15, 15, size=3)
            sol = solve_ivp(lorenz_rhs_np, t_span, y0, t_eval=t_eval, args=(rho,), method='RK45', atol=1e-8, rtol=1e-8)
            all_states.append(sol.y.T)
            all_params.append(rho)
    return t_eval, np.array(all_params), np.array(all_states)

# =============================================================================
# 5. Training Logic (LRA)
# =============================================================================

def train_monolith(model, train_data, val_data, coord_norm, state_norm, epochs=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    alpha_phys = 0.1 # LRA weight (start small to avoid origin-lock)
    
    obs_t, params_t, states_t = [torch.from_numpy(x).float().to(device) for x in train_data]
    v_obs_t, v_params_t, v_states_t = [torch.from_numpy(x).float().to(device) for x in val_data]
    
    # Observations: use first 8 steps
    train_obs = states_t[:, :8, :]
    val_obs = v_states_t[:, :8, :]
    
    # Coords: all time steps normalized
    t_points = torch.from_numpy(np.linspace(0, 16, states_t.shape[1])).float().to(device)
    coords_n = coord_norm.normalize(t_points.unsqueeze(1)).unsqueeze(0).expand(states_t.shape[0], -1, -1)
    v_coords_n = coord_norm.normalize(t_points.unsqueeze(1)).unsqueeze(0).expand(v_states_t.shape[0], -1, -1)
    
    # Targets: normalized states
    targets_n = state_norm.normalize(states_t.reshape(-1, 3)).reshape(states_t.shape)
    v_targets_n = state_norm.normalize(v_states_t.reshape(-1, 3)).reshape(v_states_t.shape)

    best_val = float('inf')
    patience = 50
    counter = 0

    pbar = tqdm(range(epochs))
    for ep in pbar:
        model.train()
        optim.zero_grad()
        
        # Forward
        preds_n, latent = model(train_obs, coords_n)
        data_loss = F.mse_loss(preds_n, targets_n)
        
        # Physics Residual
        # Sample subset for physics to stay fast
        idx = torch.randperm(coords_n.shape[1])[:64]
        c_phys = coords_n[:, idx, :].detach().requires_grad_(True)
        # We need latent expanded for physics
        L_phys = latent.clone()
        p_phys = model.decode(c_phys, L_phys)
        
        res = compute_lorenz_residual(c_phys.reshape(-1, 1), p_phys.reshape(-1, 3), params_t.repeat_interleave(len(idx)), coord_norm, state_norm)
        phys_loss = torch.mean(res**2)
        
        # LRA weight update (simplified)
        if ep % 50 == 0:
            # compute gradients
            model.zero_grad()
            data_loss.backward(retain_graph=True)
            g_data = [p.grad.norm() for p in model.decoder.parameters() if p.grad is not None]
            g_data = sum(g_data) / len(g_data) if g_data else 1.0
            
            model.zero_grad()
            phys_loss.backward(retain_graph=True)
            g_phys = [p.grad.norm() for p in model.decoder.parameters() if p.grad is not None]
            g_phys = sum(g_phys) / len(g_phys) if g_phys else 1.0
            
            alpha_phys = 0.9 * alpha_phys + 0.1 * (g_data / (g_phys + 1e-8))
            optim.zero_grad()

        # Determined phase: warmup = data only
        is_warmup = (ep < 200)
        
        # Dedicated Initial Condition (IC) loss
        ic_loss = F.mse_loss(preds_n[:, 0, :], targets_n[:, 0, :])
        
        loss_p = alpha_phys.detach() * phys_loss if not is_warmup else torch.tensor(0.0).to(device)
        loss = data_loss + loss_p + 10.0 * ic_loss
        loss.backward()
        optim.step()
        
        # Validation
        if ep % 20 == 0:
            model.eval()
            with torch.no_grad():
                v_preds_n, _ = model(val_obs, v_coords_n)
                val_mse = F.mse_loss(v_preds_n, v_targets_n).item()
                if val_mse < best_val:
                    best_val = val_mse
                    counter = 0
                else:
                    counter += 1
                if counter > patience: break
        
        pbar.set_postfix({"L": f"{loss.item():.2e}", "V": f"{val_mse:.2e}", "A": f"{alpha_phys.item():.2f}"})

# =============================================================================
# 6. Execution & Plotting
# =============================================================================

if __name__ == "__main__":
    print("Generating Lorenz Data...")
    t_eval, params, states = get_lorenz_data(rho_values=[21.0, 28.0, 35.0], num_traj=20, nt=200, t_span=(0, 16))
    
    # Split
    idx = np.random.permutation(len(params))
    tr, vl = idx[:int(0.7*len(idx))], idx[int(0.7*len(idx)):]
    train_data = (states[tr, :8, :], params[tr], states[tr]) # Note: using full state for normalizer
    val_data = (states[vl, :8, :], params[vl], states[vl])
    
    # State-wide stats for normalizer
    state_norm = StateNormalizer(torch.from_numpy(states.reshape(-1, 3)).float())
    coord_norm = CoordNormalizer(0, 16)
    
    print("Initializing Model...")
    model = TAPINN(obs_dim=3, coord_dim=1, output_dim=3, hidden_dim=64, latent_dim=16)
    
    print("Training...")
    train_monolith(model, (states[tr], params[tr], states[tr]), (states[vl], params[vl], states[vl]), coord_norm, state_norm, epochs=5000)
    
    # Evaluate on a specific parameter
    model.eval()
    test_rho = 28.0
    _, _, test_states = get_lorenz_data([test_rho], 1, 400, (0, 16), seed=123)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs = torch.from_numpy(test_states[:, :8, :]).float().to(device)
    t_full = torch.linspace(0, 16, 400).unsqueeze(1).to(device)
    coords_n = coord_norm.normalize(t_full).unsqueeze(0)
    
    with torch.no_grad():
        pred_n, _ = model(obs, coords_n)
        pred = state_norm.denormalize(pred_n[0]).cpu().numpy()
        truth = test_states[0]
        
    # Plotting
    fig = plt.figure(figsize=(12, 5))
    
    # X vs Z (Butterfly)
    ax1 = fig.add_subplot(121)
    ax1.plot(truth[:, 0], truth[:, 2], 'k', alpha=0.3, label="Ground Truth")
    ax1.plot(pred[:, 0], pred[:, 2], 'r--', label="TAPINN")
    ax1.set_title(f"Lorenz Phase Space (X-Z) - rho={test_rho}")
    ax1.set_xlabel("X"); ax1.set_ylabel("Z")
    ax1.legend()
    
    # X-Y-Z 3D
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(truth[:, 0], truth[:, 1], truth[:, 2], 'k', alpha=0.3)
    ax2.plot(pred[:, 0], pred[:, 1], pred[:, 2], 'r--')
    ax2.set_title("3D Attractor")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    
    plt.tight_layout()
    plt.savefig("tapinn_lorenz_monolith.png", dpi=300)
    print("Saved results to tapinn_lorenz_monolith.png")
