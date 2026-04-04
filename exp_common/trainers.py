from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap, jacrev
from torch.utils.data import DataLoader, TensorDataset

from .problems import compute_ode_residual, compute_pde_residual

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback only
    def tqdm(iterable=None, **kwargs):
        return iterable


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    history: list[dict[str, float]]
    seconds_per_epoch: float
    epochs_trained: int = 0
    best_val_loss: float = float("inf")
    ntk_eigenvalues: list[dict] = field(default_factory=list)
    jacobian_conditions: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Callback infrastructure (shared across all model families for fairness)
# ---------------------------------------------------------------------------

@dataclass
class EarlyStopping:
    """Stops training when validation loss does not improve for *patience* epochs.

    A single shared instance must be created fresh for each model so that
    different training runs do not share state.  Call :meth:`reset` if you
    want to reuse the object.
    """
    patience: int = 15
    min_delta: float = 1e-6
    best_loss: float = field(default_factory=lambda: float("inf"))
    counter: int = 0
    triggered: bool = False

    def step(self, val_loss: float) -> bool:
        """Return True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered

    def reset(self) -> None:
        self.best_loss = float("inf")
        self.counter = 0
        self.triggered = False


@dataclass
class ValBundle:
    """Typed container for validation data passed to training loops."""
    observations: torch.Tensor
    coords: torch.Tensor
    targets: torch.Tensor
    params: torch.Tensor
    ode_metadata: torch.Tensor | None = None


@dataclass
class CallbackConfig:
    """Shared callback hyper-parameters applied **identically** to all models.

    Using a single shared instance ensures experimental fairness: every model
    is given the same budget, patience, and LR-decay schedule.

    Parameters
    ----------
    early_stopping_patience:
        Epochs without improvement before stopping.  0 = disabled.
    reduce_lr_patience:
        Epochs without improvement before reducing LR.  0 = disabled.
    reduce_lr_factor:
        Multiplicative factor applied to LR on each reduction.
    min_lr:
        Minimum LR below which ReduceLROnPlateau will not go.
    """
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 8
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6


# ---------------------------------------------------------------------------
# Tensor preparation utilities
# ---------------------------------------------------------------------------

def prepare_ode_tensors(
    system_data,
    observation_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    observations = torch.tensor(system_data.states[:, :observation_steps, :], dtype=torch.float32)
    num_samples, num_steps, _ = system_data.states.shape
    coords = np.tile(system_data.times[None, :, None], (num_samples, 1, 1))
    targets = torch.tensor(system_data.states, dtype=torch.float32)
    params = torch.tensor(system_data.params, dtype=torch.float32)
    ode_metadata = None
    if system_data.metadata and "natural_frequencies" in system_data.metadata:
        ode_metadata = torch.tensor(system_data.metadata["natural_frequencies"], dtype=torch.float32)
    return observations, torch.tensor(coords, dtype=torch.float32), targets, params, ode_metadata


def prepare_pde_tensors(system_data, observation_steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    observations = torch.tensor(system_data.fields[:, :observation_steps, :], dtype=torch.float32)
    mesh_t, mesh_x = np.meshgrid(system_data.times, system_data.space, indexing="ij")
    flat_coords = np.stack([mesh_t.reshape(-1), mesh_x.reshape(-1)], axis=1).astype(np.float32)
    num_samples = system_data.fields.shape[0]
    coords = np.tile(flat_coords[None, :, :], (num_samples, 1, 1))
    targets = torch.tensor(system_data.fields.reshape(num_samples, -1, 1), dtype=torch.float32)
    params = torch.tensor(system_data.params, dtype=torch.float32)
    return observations, torch.tensor(coords, dtype=torch.float32), targets, params


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def compute_ntk_spectrum(model, coords):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def fmodel(params, buffers, x):
        return functional_call(model, (params, buffers), (x.unsqueeze(0),))

    jacobian_fn = vmap(jacrev(fmodel, argnums=0), in_dims=(None, None, 0))
    jac_dict = jacobian_fn(params, buffers, coords)

    J = torch.cat([j.flatten(start_dim=1) for j in jac_dict.values()], dim=1)

    K = J @ J.T
    eigvals = torch.linalg.eigvalsh(K)
    return eigvals

def compute_jacobian_condition(residual_fn, model, coords):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    jac_fn = jacrev(residual_fn, argnums=0)
    jac_dict = jac_fn(params, buffers, coords)

    J = torch.cat([j.flatten(start_dim=1) for j in jac_dict.values()], dim=1)

    s = torch.linalg.svdvals(J)
    return s.max() / max(s.min().item(), 1e-12)

def lipschitz_estimate(encoder, x):
    from torch.autograd.functional import jacobian
    # Must use train() to allow cuDNN LSTM backward passes
    encoder.train()
    
    norms = []
    # Use standard jacobian in a loop to avoid cuDNN LSTM flatten_parameters bug under torch.func
    for i in range(x.shape[0]):
        x_i = x[i:i+1]
        def f(inp):
            return encoder(inp)
        
        J = jacobian(f, x_i, vectorize=False)
        J = J.view(J.shape[1], -1)  # Reshape to (out_dim, flattened_in_dim)
        norm = torch.linalg.matrix_norm(J, ord=2)
        norms.append(norm)
        
    if not norms:
        return torch.tensor(0.0, device=x.device)
        
    return torch.stack(norms).max()

class ConFIGOptimizer:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def config_update(self, g1, g2):
        dot = torch.dot(g1, g2)
        if dot >= 0:
            return g1 + g2
        correction = (-dot / max(torch.norm(g2).item() ** 2, 1e-12)) * g2
        return g1 + correction

    def step(self, loss1, loss2, model_parameters):
        params = [p for p in model_parameters if p.requires_grad]
        g1 = torch.autograd.grad(loss1, params, retain_graph=True, allow_unused=True)
        g2 = torch.autograd.grad(loss2, params, allow_unused=True)
        
        for p, g1_n, g2_n in zip(params, g1, g2):
            if g1_n is None and g2_n is None:
                continue
            if g1_n is None:
                p.grad = g2_n.clone()
            elif g2_n is None:
                p.grad = g1_n.clone()
            else:
                p.grad = self.config_update(g1_n.reshape(-1), g2_n.reshape(-1)).view_as(p)

        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


def _triplet_loss(latent: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    if latent.shape[0] < 3:
        return torch.tensor(0.0, device=latent.device)
    total = torch.tensor(0.0, device=latent.device)
    count = 0
    for idx in range(latent.shape[0]):
        diffs = torch.abs(params - params[idx])
        pos_candidates = torch.where((diffs < 1e-6) & (torch.arange(latent.shape[0], device=latent.device) != idx))[0]
        neg_candidates = torch.where(diffs > torch.median(diffs))[0]
        if len(pos_candidates) == 0 or len(neg_candidates) == 0:
            continue
        pos = latent[pos_candidates[0]]
        neg = latent[neg_candidates[-1]]
        pos_dist = torch.norm(latent[idx] - pos, p=2)
        neg_dist = torch.norm(latent[idx] - neg, p=2)
        margin = torch.abs(params[idx] - params[neg_candidates[-1]])
        total = total + torch.relu(pos_dist - neg_dist + margin)
        count += 1
    if count == 0:
        return torch.tensor(0.0, device=latent.device)
    return total / count


def _sample_coord_subset(coords: torch.Tensor, targets: torch.Tensor, max_points: int) -> tuple[torch.Tensor, torch.Tensor]:
    total_points = coords.shape[1]
    if total_points <= max_points:
        return coords, targets
    idx = torch.randperm(total_points, device=coords.device)[:max_points]
    return coords[:, idx, :], targets[:, idx, :]


# ---------------------------------------------------------------------------
# Validation-loss helpers (used by callbacks, run under torch.no_grad)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_val_loss_tapinn(
    model,
    val_bundle: ValBundle,
    device: torch.device,
    max_data_points: int = 128,
) -> float:
    model.eval()
    obs = val_bundle.observations.to(device)
    if obs.shape[0] == 0:
        model.train()
        return float("inf")
    coords = val_bundle.coords.to(device)
    targets = val_bundle.targets.to(device)
    latent = model.encode(obs)
    data_coords, data_targets = _sample_coord_subset(coords, targets, max_data_points)
    flat_coords = data_coords.reshape(-1, coords.shape[-1])
    flat_targets = data_targets.reshape(-1, targets.shape[-1])
    exp_latent = latent.unsqueeze(1).expand(-1, data_coords.shape[1], -1).reshape(-1, latent.shape[-1])
    pred = model.decode(flat_coords, exp_latent)
    loss = float(F.mse_loss(pred, flat_targets).item())
    model.train()
    return loss


@torch.no_grad()
def _compute_val_loss_direct(
    model,
    model_kind: str,
    val_bundle: ValBundle,
    device: torch.device,
    max_data_points: int = 128,
) -> float:
    model.eval()
    obs = val_bundle.observations.to(device)
    if obs.shape[0] == 0:
        model.train()
        return float("inf")
    coords = val_bundle.coords.to(device)
    targets = val_bundle.targets.to(device)
    params = val_bundle.params.to(device)
    data_coords, data_targets = _sample_coord_subset(coords, targets, max_data_points)
    flat_coords = data_coords.reshape(-1, coords.shape[-1])
    flat_targets = data_targets.reshape(-1, targets.shape[-1])
    repeated_params = params.unsqueeze(1).expand(-1, data_coords.shape[1]).reshape(-1)
    if model_kind == "deeponet":
        branch_input = obs.reshape(obs.shape[0], -1)
        repeated_branch = branch_input.unsqueeze(1).expand(-1, data_coords.shape[1], -1).reshape(
            branch_input.shape[0] * data_coords.shape[1], -1
        )
        pred = model(repeated_branch, flat_coords)
    else:
        pred = model(flat_coords, repeated_params)
    loss = float(F.mse_loss(pred, flat_targets).item())
    model.train()
    return loss


@torch.no_grad()
def _compute_val_loss_fno(
    model,
    val_bundle: ValBundle,
    device: torch.device,
) -> float:
    model.eval()
    obs = val_bundle.observations.to(device)
    if obs.shape[0] == 0:
        model.train()
        return float("inf")
    targets = val_bundle.targets.to(device)
    flat_obs = obs.reshape(obs.shape[0], -1)
    num_steps = targets.shape[1]
    grid = torch.linspace(0.0, 1.0, num_steps, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
    pred = model(flat_obs, grid.expand(flat_obs.shape[0], -1, -1))
    loss = float(F.mse_loss(pred, targets).item())
    model.train()
    return loss


# ---------------------------------------------------------------------------
# TAPINN training loop
# ---------------------------------------------------------------------------

def train_tapinn(
    model,
    problem_name: str,
    observations: torch.Tensor,
    coords: torch.Tensor,
    targets: torch.Tensor,
    params: torch.Tensor,
    device: torch.device,
    ode_metadata: torch.Tensor | None = None,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    alpha_physics: float = 0.1,
    beta_metric: float = 0.1,
    alternating: bool = True,
    interaction_frequency: int = 4,
    max_data_points: int = 128,
    max_phys_points: int = 64,
    progress_desc: str | None = None,
    use_config: bool = False,
    use_soap: bool = False,
    # --- callback arguments (all optional; default = disabled) ---
    val_bundle: ValBundle | None = None,
    callbacks: CallbackConfig | None = None,
) -> TrainResult:
    dataset_tensors = [observations, coords, targets, params]
    if ode_metadata is not None:
        dataset_tensors.append(ode_metadata)
    dataset = TensorDataset(*dataset_tensors)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    model = model.to(device)
    
    if use_soap:
        from .soap import SOAP
        optimizer_encoder = SOAP(model.encoder.parameters(), lr=lr)
        optimizer_generator = SOAP(model.generator.parameters(), lr=lr)
        optimizer_joint = SOAP(model.parameters(), lr=lr * 0.75)
    else:
        optimizer_encoder = torch.optim.Adam(model.encoder.parameters(), lr=lr)
        optimizer_generator = torch.optim.Adam(model.generator.parameters(), lr=lr)
        optimizer_joint = torch.optim.Adam(model.parameters(), lr=lr * 0.75)
        
    config_opt_joint = ConFIGOptimizer(optimizer_joint) if use_config else None

    history = []
    total_time = 0.0

    # Build callbacks for each sub-optimizer
    early_stopper: EarlyStopping | None = None
    schedulers: list = []
    if val_bundle is not None and callbacks is not None:
        if callbacks.early_stopping_patience > 0:
            early_stopper = EarlyStopping(patience=callbacks.early_stopping_patience)
        if callbacks.reduce_lr_patience > 0:
            for opt in [optimizer_encoder, optimizer_generator, optimizer_joint]:
                schedulers.append(
                    torch.optim.lr_scheduler.ReduceLROnPlateau(
                        opt,
                        mode="min",
                        factor=callbacks.reduce_lr_factor,
                        patience=callbacks.reduce_lr_patience,
                        min_lr=callbacks.min_lr,
                    )
                )

    best_val_loss = float("inf")
    epochs_trained = 0
    epoch_iter = tqdm(range(epochs), desc=progress_desc or f"TAPINN[{problem_name}]", leave=False)
    for epoch in epoch_iter:
        epoch_start = time.perf_counter()
        running = {"loss": 0.0, "data_mse": 0.0, "physics_mse": 0.0, "metric_loss": 0.0}
        steps = 0
        for step_idx, batch in enumerate(loader):
            if ode_metadata is not None:
                obs_batch, coord_batch, target_batch, param_batch, ode_meta_batch = batch
                ode_meta_batch = ode_meta_batch.to(device)
            else:
                obs_batch, coord_batch, target_batch, param_batch = batch
                ode_meta_batch = None
            obs_batch = obs_batch.to(device)
            coord_batch = coord_batch.to(device)
            target_batch = target_batch.to(device)
            param_batch = param_batch.to(device)
            phase = "joint"
            if alternating:
                global_step = epoch * len(loader) + step_idx + 1
                if global_step % interaction_frequency == 0:
                    phase = "joint"
                else:
                    phase = "encoder" if global_step % 2 == 1 else "generator"

            data_coords, data_targets = _sample_coord_subset(coord_batch, target_batch, max_data_points)
            flat_data_coords = data_coords.reshape(-1, coord_batch.shape[-1])
            flat_data_targets = data_targets.reshape(-1, target_batch.shape[-1])

            if phase == "encoder":
                optimizer_encoder.zero_grad()
                latent = model.encode(obs_batch)
                expanded_latent = latent.unsqueeze(1).expand(-1, data_coords.shape[1], -1).reshape(-1, latent.shape[-1])
                pred = model.decode(flat_data_coords, expanded_latent)
                data_loss = torch.mean((pred - flat_data_targets) ** 2)
                metric_loss = _triplet_loss(latent, param_batch)
                physics_loss = torch.tensor(0.0, device=device)
                loss = data_loss + beta_metric * metric_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
                optimizer_encoder.step()
            elif phase == "generator":
                optimizer_generator.zero_grad()
                with torch.no_grad():
                    latent = model.encode(obs_batch)
                phys_coords, _ = _sample_coord_subset(coord_batch, target_batch, max_phys_points)
                flat_phys_coords = phys_coords.reshape(-1, coord_batch.shape[-1]).detach().clone().requires_grad_(True)
                expanded_latent = latent.unsqueeze(1).expand(-1, data_coords.shape[1], -1).reshape(-1, latent.shape[-1])
                pred = model.decode(flat_data_coords, expanded_latent)
                data_loss = torch.mean((pred - flat_data_targets) ** 2)
                expanded_phys_latent = latent.unsqueeze(1).expand(-1, phys_coords.shape[1], -1).reshape(-1, latent.shape[-1])
                phys_param = param_batch.unsqueeze(1).expand(-1, phys_coords.shape[1]).reshape(-1)
                phys_pred = model.decode(flat_phys_coords, expanded_phys_latent)
                if flat_phys_coords.shape[1] == 1:
                    phys_metadata = None
                    if ode_meta_batch is not None:
                        phys_metadata = {
                            "natural_frequencies": ode_meta_batch.unsqueeze(1)
                            .expand(-1, phys_coords.shape[1], -1)
                            .reshape(-1, ode_meta_batch.shape[-1])
                        }
                    residual = compute_ode_residual(problem_name, flat_phys_coords, phys_pred, phys_param, metadata=phys_metadata)
                else:
                    residual = compute_pde_residual(problem_name, flat_phys_coords, phys_pred[:, 0], phys_param)
                physics_loss = torch.mean(residual**2)
                metric_loss = torch.tensor(0.0, device=device)
                loss = data_loss + alpha_physics * physics_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
                optimizer_generator.step()
            else:
                optimizer_joint.zero_grad()
                latent = model.encode(obs_batch)
                expanded_latent = latent.unsqueeze(1).expand(-1, data_coords.shape[1], -1).reshape(-1, latent.shape[-1])
                pred = model.decode(flat_data_coords, expanded_latent)
                data_loss = torch.mean((pred - flat_data_targets) ** 2)
                metric_loss = _triplet_loss(latent, param_batch)
                phys_coords, _ = _sample_coord_subset(coord_batch, target_batch, max_phys_points)
                flat_phys_coords = phys_coords.reshape(-1, coord_batch.shape[-1]).detach().clone().requires_grad_(True)
                expanded_phys_latent = latent.unsqueeze(1).expand(-1, phys_coords.shape[1], -1).reshape(-1, latent.shape[-1])
                phys_param = param_batch.unsqueeze(1).expand(-1, phys_coords.shape[1]).reshape(-1)
                phys_pred = model.decode(flat_phys_coords, expanded_phys_latent)
                if flat_phys_coords.shape[1] == 1:
                    phys_metadata = None
                    if ode_meta_batch is not None:
                        phys_metadata = {
                            "natural_frequencies": ode_meta_batch.unsqueeze(1)
                            .expand(-1, phys_coords.shape[1], -1)
                            .reshape(-1, ode_meta_batch.shape[-1])
                        }
                    residual = compute_ode_residual(problem_name, flat_phys_coords, phys_pred, phys_param, metadata=phys_metadata)
                else:
                    residual = compute_pde_residual(problem_name, flat_phys_coords, phys_pred[:, 0], phys_param)
                physics_loss = torch.mean(residual**2)
                
                loss = data_loss + alpha_physics * physics_loss + beta_metric * metric_loss
                if use_config:
                    config_opt_joint.step(data_loss, alpha_physics * physics_loss + beta_metric * metric_loss, model.parameters())
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer_joint.step()

            running["loss"] += float(loss.item())
            running["data_mse"] += float(data_loss.item())
            running["physics_mse"] += float(physics_loss.item())
            running["metric_loss"] += float(metric_loss.item())
            steps += 1

        total_time += time.perf_counter() - epoch_start
        epoch_summary = {key: value / max(steps, 1) for key, value in running.items()}
        history.append(epoch_summary)
        epochs_trained = epoch + 1
        if hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(loss=f'{epoch_summary["loss"]:.4f}', data=f'{epoch_summary["data_mse"]:.4f}', phys=f'{epoch_summary["physics_mse"]:.4f}')

        # --- callbacks ---
        if val_bundle is not None and callbacks is not None:
            val_loss = _compute_val_loss_tapinn(model, val_bundle, device, max_data_points)
            epoch_summary["val_data_mse"] = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            for sched in schedulers:
                sched.step(val_loss)
            if early_stopper is not None and early_stopper.step(val_loss):
                break

    return TrainResult(
        history=history,
        seconds_per_epoch=total_time / max(epochs_trained, 1),
        epochs_trained=epochs_trained,
        best_val_loss=best_val_loss,
    )


# ---------------------------------------------------------------------------
# Direct-model training loop (StandardPINN, HyperPINN, DeepONet)
# ---------------------------------------------------------------------------

def train_direct_model(
    model,
    problem_name: str,
    model_kind: str,
    observations: torch.Tensor,
    coords: torch.Tensor,
    targets: torch.Tensor,
    params: torch.Tensor,
    device: torch.device,
    ode_metadata: torch.Tensor | None = None,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    alpha_physics: float = 0.1,
    max_data_points: int = 128,
    max_phys_points: int = 64,
    progress_desc: str | None = None,
    use_config: bool = False,
    use_soap: bool = False,
    # --- callback arguments ---
    val_bundle: ValBundle | None = None,
    callbacks: CallbackConfig | None = None,
) -> TrainResult:
    dataset_tensors = [observations, coords, targets, params]
    if ode_metadata is not None:
        dataset_tensors.append(ode_metadata)
    dataset = TensorDataset(*dataset_tensors)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    model = model.to(device)
    
    if use_soap:
        from .soap import SOAP
        optimizer = SOAP(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    config_opt = ConFIGOptimizer(optimizer) if use_config else None
    history = []
    total_time = 0.0

    early_stopper: EarlyStopping | None = None
    schedulers: list = []
    if val_bundle is not None and callbacks is not None:
        if callbacks.early_stopping_patience > 0:
            early_stopper = EarlyStopping(patience=callbacks.early_stopping_patience)
        if callbacks.reduce_lr_patience > 0:
            schedulers.append(
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=callbacks.reduce_lr_factor,
                    patience=callbacks.reduce_lr_patience,
                    min_lr=callbacks.min_lr,
                )
            )

    best_val_loss = float("inf")
    epochs_trained = 0
    epoch_iter = tqdm(range(epochs), desc=progress_desc or f"{model_kind}[{problem_name}]", leave=False)
    for _ in epoch_iter:
        epoch_start = time.perf_counter()
        running = {"loss": 0.0, "data_mse": 0.0, "physics_mse": 0.0}
        steps = 0
        for batch in loader:
            if ode_metadata is not None:
                obs_batch, coord_batch, target_batch, param_batch, ode_meta_batch = batch
                ode_meta_batch = ode_meta_batch.to(device)
            else:
                obs_batch, coord_batch, target_batch, param_batch = batch
                ode_meta_batch = None
            obs_batch = obs_batch.to(device)
            coord_batch = coord_batch.to(device)
            target_batch = target_batch.to(device)
            param_batch = param_batch.to(device)
            optimizer.zero_grad()

            data_coords, data_targets = _sample_coord_subset(coord_batch, target_batch, max_data_points)
            flat_data_coords = data_coords.reshape(-1, coord_batch.shape[-1])
            flat_data_targets = data_targets.reshape(-1, target_batch.shape[-1])
            repeated_params = param_batch.unsqueeze(1).expand(-1, data_coords.shape[1]).reshape(-1)

            if model_kind == "deeponet":
                branch_input = obs_batch.reshape(obs_batch.shape[0], -1)
                repeated_branch = branch_input.unsqueeze(1).expand(-1, data_coords.shape[1], -1).reshape(branch_input.shape[0] * data_coords.shape[1], -1)
                pred = model(repeated_branch, flat_data_coords)
            else:
                pred = model(flat_data_coords, repeated_params)
            data_loss = torch.mean((pred - flat_data_targets) ** 2)

            phys_coords, _ = _sample_coord_subset(coord_batch, target_batch, max_phys_points)
            flat_phys_coords = phys_coords.reshape(-1, coord_batch.shape[-1]).detach().clone().requires_grad_(True)
            phys_params = param_batch.unsqueeze(1).expand(-1, phys_coords.shape[1]).reshape(-1)
            if model_kind == "deeponet":
                branch_input = obs_batch.reshape(obs_batch.shape[0], -1)
                repeated_branch = branch_input.unsqueeze(1).expand(-1, phys_coords.shape[1], -1).reshape(branch_input.shape[0] * phys_coords.shape[1], -1)
                phys_pred = model(repeated_branch, flat_phys_coords)
            else:
                phys_pred = model(flat_phys_coords, phys_params)

            if flat_phys_coords.shape[1] == 1:
                phys_metadata = None
                if ode_meta_batch is not None:
                    phys_metadata = {
                        "natural_frequencies": ode_meta_batch.unsqueeze(1)
                        .expand(-1, phys_coords.shape[1], -1)
                        .reshape(-1, ode_meta_batch.shape[-1])
                    }
                residual = compute_ode_residual(problem_name, flat_phys_coords, phys_pred, phys_params, metadata=phys_metadata)
            else:
                residual = compute_pde_residual(problem_name, flat_phys_coords, phys_pred[:, 0], phys_params)
            physics_loss = torch.mean(residual**2)
            loss = data_loss + alpha_physics * physics_loss
            
            if use_config:
                config_opt.step(data_loss, alpha_physics * physics_loss, model.parameters())
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            running["loss"] += float(loss.item())
            running["data_mse"] += float(data_loss.item())
            running["physics_mse"] += float(physics_loss.item())
            steps += 1

        total_time += time.perf_counter() - epoch_start
        epoch_summary = {key: value / max(steps, 1) for key, value in running.items()}
        history.append(epoch_summary)
        epochs_trained += 1
        if hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(loss=f'{epoch_summary["loss"]:.4f}', data=f'{epoch_summary["data_mse"]:.4f}', phys=f'{epoch_summary["physics_mse"]:.4f}')

        # --- callbacks ---
        if val_bundle is not None and callbacks is not None:
            val_loss = _compute_val_loss_direct(model, model_kind, val_bundle, device, max_data_points)
            epoch_summary["val_data_mse"] = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            for sched in schedulers:
                sched.step(val_loss)
            if early_stopper is not None and early_stopper.step(val_loss):
                break

    return TrainResult(
        history=history,
        seconds_per_epoch=total_time / max(epochs_trained, 1),
        epochs_trained=epochs_trained,
        best_val_loss=best_val_loss,
    )


# ---------------------------------------------------------------------------
# FNO training loop (supervised-only)
# ---------------------------------------------------------------------------

def train_fno_model(
    model,
    observations: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    progress_desc: str | None = None,
    # --- callback arguments ---
    val_bundle: ValBundle | None = None,
    callbacks: CallbackConfig | None = None,
) -> TrainResult:
    flat_obs = observations.reshape(observations.shape[0], -1)
    dataset = TensorDataset(flat_obs, targets)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    grid = torch.linspace(0.0, 1.0, targets.shape[1], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
    history = []
    total_time = 0.0

    early_stopper: EarlyStopping | None = None
    schedulers: list = []
    if val_bundle is not None and callbacks is not None:
        if callbacks.early_stopping_patience > 0:
            early_stopper = EarlyStopping(patience=callbacks.early_stopping_patience)
        if callbacks.reduce_lr_patience > 0:
            schedulers.append(
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=callbacks.reduce_lr_factor,
                    patience=callbacks.reduce_lr_patience,
                    min_lr=callbacks.min_lr,
                )
            )

    best_val_loss = float("inf")
    epochs_trained = 0
    epoch_iter = tqdm(range(epochs), desc=progress_desc or "FNO", leave=False)
    for _ in epoch_iter:
        epoch_start = time.perf_counter()
        running = {"loss": 0.0}
        steps = 0
        for obs_batch, target_batch in loader:
            obs_batch = obs_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad()
            repeated_grid = grid.expand(obs_batch.shape[0], -1, -1)
            pred = model(obs_batch, repeated_grid)
            loss = torch.mean((pred - target_batch) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running["loss"] += float(loss.item())
            steps += 1
        total_time += time.perf_counter() - epoch_start
        epoch_summary = {"loss": running["loss"] / max(steps, 1)}
        history.append(epoch_summary)
        epochs_trained += 1
        if hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(loss=f'{epoch_summary["loss"]:.4f}')

        # --- callbacks ---
        if val_bundle is not None and callbacks is not None:
            val_loss = _compute_val_loss_fno(model, val_bundle, device)
            epoch_summary["val_data_mse"] = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            for sched in schedulers:
                sched.step(val_loss)
            if early_stopper is not None and early_stopper.step(val_loss):
                break

    return TrainResult(
        history=history,
        seconds_per_epoch=total_time / max(epochs_trained, 1),
        epochs_trained=epochs_trained,
        best_val_loss=best_val_loss,
    )


# ---------------------------------------------------------------------------
# Prediction utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_tapinn(model, observations: torch.Tensor, coords: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    latent = model.encode(observations.to(device))
    expanded_latent = latent.unsqueeze(1).expand(-1, coords.shape[1], -1).reshape(-1, latent.shape[-1])
    pred = model.decode(coords.to(device).reshape(-1, coords.shape[-1]), expanded_latent)
    return pred.reshape(observations.shape[0], coords.shape[1], -1)


@torch.no_grad()
def predict_direct(model, model_kind: str, observations: torch.Tensor, coords: torch.Tensor, params: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    flat_coords = coords.to(device).reshape(-1, coords.shape[-1])
    repeated_params = params.to(device).unsqueeze(1).expand(-1, coords.shape[1]).reshape(-1)
    if model_kind == "deeponet":
        branch_input = observations.reshape(observations.shape[0], -1).to(device)
        repeated_branch = branch_input.unsqueeze(1).expand(-1, coords.shape[1], -1).reshape(-1, branch_input.shape[-1])
        pred = model(repeated_branch, flat_coords)
    else:
        pred = model(flat_coords, repeated_params)
    return pred.reshape(observations.shape[0], coords.shape[1], -1)


@torch.no_grad()
def predict_fno(model, observations: torch.Tensor, target_points: int, device: torch.device) -> torch.Tensor:
    model.eval()
    branch_input = observations.reshape(observations.shape[0], -1).to(device)
    grid = torch.linspace(0.0, 1.0, target_points, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
    pred = model(branch_input, grid.expand(branch_input.shape[0], -1, -1))
    return pred
