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
    ntk_eigenvalues: list[dict[str, float]] = field(default_factory=list)
    jacobian_conditions: list[dict[str, float]] = field(default_factory=list)


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
# Coordinate and State Normalizers
# ---------------------------------------------------------------------------

@dataclass
class CoordNormalizer:
    """MinMax normalizer that maps each coordinate dimension to [-1, 1].

    ``coord_scales`` stores the factor ``2 / (c_max - c_min)`` for each
    coordinate dimension.  This is the chain-rule multiplier needed to convert
    an autodiff derivative w.r.t. the *normalised* coordinate back to the
    derivative w.r.t. the *raw* coordinate::

        d/dt_raw = (d/dt_norm) * coord_scale[0]
    """
    coord_mins: torch.Tensor    # shape (coord_dim,)
    coord_maxs: torch.Tensor    # shape (coord_dim,)
    coord_scales: torch.Tensor  # 2/(max-min) per dim, shape (coord_dim,)

    @classmethod
    def from_coords(cls, flat_coords: torch.Tensor) -> "CoordNormalizer":
        """Build from a 2-D tensor of shape (N, coord_dim)."""
        mins = flat_coords.min(dim=0).values
        maxs = flat_coords.max(dim=0).values
        ranges = (maxs - mins).clamp(min=1e-6)
        scales = 2.0 / ranges
        return cls(coord_mins=mins, coord_maxs=maxs, coord_scales=scales)

    def normalize(self, coords: torch.Tensor) -> torch.Tensor:
        """Map coords from raw space to [-1, 1].  Works on any shape (…, coord_dim)."""
        mins = self.coord_mins.to(coords.device)
        maxs = self.coord_maxs.to(coords.device)
        ranges = (maxs - mins).clamp(min=1e-6)
        return 2.0 * (coords - mins) / ranges - 1.0

    def denormalize(self, coords: torch.Tensor) -> torch.Tensor:
        """Map coords from [-1, 1] back to raw space."""
        mins = self.coord_mins.to(coords.device)
        maxs = self.coord_maxs.to(coords.device)
        ranges = (maxs - mins).clamp(min=1e-6)
        return (coords + 1.0) * ranges / 2.0 + mins


@dataclass
class StateNormalizer:
    """MinMax normalizer for state/target tensors (output space).

    Maps each state dimension from ``[state_min, state_max]`` to ``[-1, 1]``.
    The inverse is used to un-normalise predictions before evaluation.
    """
    state_mins: torch.Tensor   # shape (state_dim,)
    state_maxs: torch.Tensor   # shape (state_dim,)

    @classmethod
    def from_targets(cls, targets: torch.Tensor) -> "StateNormalizer":
        """Build from a tensor of shape (N, T, state_dim) or (N*T, state_dim)."""
        flat = targets.reshape(-1, targets.shape[-1])
        mins = flat.min(dim=0).values
        maxs = flat.max(dim=0).values
        return cls(state_mins=mins, state_maxs=maxs)

    def normalize(self, states: torch.Tensor) -> torch.Tensor:
        mins = self.state_mins.to(states.device)
        maxs = self.state_maxs.to(states.device)
        ranges = (maxs - mins).clamp(min=1e-6)
        return 2.0 * (states - mins) / ranges - 1.0

    def denormalize(self, states: torch.Tensor) -> torch.Tensor:
        mins = self.state_mins.to(states.device)
        maxs = self.state_maxs.to(states.device)
        ranges = (maxs - mins).clamp(min=1e-6)
        return (states + 1.0) * ranges / 2.0 + mins


# ---------------------------------------------------------------------------
# Tensor preparation utilities
# ---------------------------------------------------------------------------

def prepare_ode_tensors(
    system_data,
    observation_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, CoordNormalizer, StateNormalizer]:
    """Prepare ODE tensors with coordinate and state normalisation.

    Returns normalised coordinates in ``[-1, 1]`` and normalised targets in
    ``[-1, 1]``.  The returned ``CoordNormalizer`` and ``StateNormalizer``
    carry the inverse transforms needed for physics residual scaling and
    plotting.
    """
    raw_states = system_data.states  # (N, T, state_dim)
    num_samples, num_steps, _ = raw_states.shape

    raw_targets = torch.tensor(raw_states, dtype=torch.float32)  # (N, T, state_dim)
    state_norm = StateNormalizer.from_targets(raw_targets)
    targets = state_norm.normalize(raw_targets)  # (N, T, state_dim) in [-1, 1]

    # Build normalised observations from the first `observation_steps` time steps
    raw_obs = torch.tensor(raw_states[:, :observation_steps, :], dtype=torch.float32)
    observations = state_norm.normalize(raw_obs)

    # Raw coordinate grid: shape (N, T, 1)
    raw_coords_flat = torch.tensor(system_data.times[:, None], dtype=torch.float32)  # (T, 1)

    # Fit normalizers on training data (caller can refit on train split only)
    coord_norm = CoordNormalizer.from_coords(raw_coords_flat)
    norm_times = coord_norm.normalize(raw_coords_flat)  # (T, 1) in [-1, 1]
    coords = norm_times.unsqueeze(0).expand(num_samples, -1, -1).clone()  # (N, T, 1)

    params = torch.tensor(system_data.params, dtype=torch.float32)
    ode_metadata = None
    if system_data.metadata and "natural_frequencies" in system_data.metadata:
        ode_metadata = torch.tensor(system_data.metadata["natural_frequencies"], dtype=torch.float32)
    return observations, coords, targets, params, ode_metadata, coord_norm, state_norm


def refit_normalizers_on_physical_split(
    original_coord_norm: CoordNormalizer,
    original_state_norm: StateNormalizer,
    train_coords_normed: torch.Tensor,
    train_targets_normed: torch.Tensor,
    train_obs_normed: torch.Tensor,
    val_targets_normed: torch.Tensor,
    val_obs_normed: torch.Tensor,
    test_targets_normed: torch.Tensor,
    test_obs_normed: torch.Tensor,
) -> tuple[
    CoordNormalizer, StateNormalizer,
    torch.Tensor, torch.Tensor, torch.Tensor,  # re-normed targets (train, val, test)
    torch.Tensor, torch.Tensor, torch.Tensor,  # re-normed obs (train, val, test)
]:
    """Correctly refit normalizers by first mapping back to physical space.

    BUG PREVENTION: refit_normalizers_on_split must receive PHYSICAL-SPACE
    tensors.  Because prepare_ode_tensors already normalizes everything, this
    helper reverses that normalization using the original normalizers before
    refitting on the training partition only.
    """
    # 1. Denormalize to physical space using the CORRECT original normalizers
    train_targets_phys = original_state_norm.denormalize(train_targets_normed)
    val_targets_phys   = original_state_norm.denormalize(val_targets_normed)
    test_targets_phys  = original_state_norm.denormalize(test_targets_normed)
    
    train_obs_phys     = original_state_norm.denormalize(train_obs_normed)
    val_obs_phys       = original_state_norm.denormalize(val_obs_normed)
    test_obs_phys      = original_state_norm.denormalize(test_obs_normed)

    # 2. Refit state_norm on training targets IN PHYSICAL SPACE
    new_state_norm = StateNormalizer.from_targets(train_targets_phys)

    # 3. Coord_norm is inherited unchanged: it was built from raw system.times
    # (all ODE/PDE samples share the same coordinate grid, so there is no
    # leakage from using the full-dataset coord_norm)
    new_coord_norm = original_coord_norm

    # 4. Re-normalize all splits with the new training-based state_norm
    train_targets_out = new_state_norm.normalize(train_targets_phys)
    val_targets_out   = new_state_norm.normalize(val_targets_phys)
    test_targets_out  = new_state_norm.normalize(test_targets_phys)
    
    train_obs_out     = new_state_norm.normalize(train_obs_phys)
    val_obs_out       = new_state_norm.normalize(val_obs_phys)
    test_obs_out      = new_state_norm.normalize(test_obs_phys)

    return (
        new_coord_norm, new_state_norm,
        train_targets_out, val_targets_out, test_targets_out,
        train_obs_out, val_obs_out, test_obs_out,
    )


def refit_normalizers_on_split(
    train_obs: torch.Tensor,
    train_coords: torch.Tensor,
    train_targets: torch.Tensor,
) -> tuple[CoordNormalizer, StateNormalizer]:
    """Re-fit normalizers from training-split tensors to avoid data leakage.

    WARNING: This function expects PHYSICAL-SPACE (un-normalized) tensors.
    Passing already-normalized [-1, 1] tensors will produce wrong coord_scales
    (≈1.0 instead of 2/T) causing silent physics residual corruption.

    For the correct usage pattern with pre-normalized tensors from
    prepare_ode_tensors / prepare_pde_tensors, use
    refit_normalizers_on_physical_split() instead.
    """
    # Refit coord normalizer from the training slice
    flat_coords = train_coords[0].reshape(-1, train_coords.shape[-1])
    
    # Safety check: if coords appear already normalized to [-1, 1],
    # the resulting coord_scales will be ≈1.0 (wrong for any T > 2).
    coord_range = float((flat_coords.max() - flat_coords.min()).item())
    if coord_range < 2.1 and flat_coords.min().item() > -1.1:
         raise ValueError(
            "refit_normalizers_on_split received coords that appear to be "
            "already normalized to [-1, 1] (range={:.3f}, min={:.3f}). "
            "This will produce coord_scales ≈ 1.0 instead of the correct "
            "2/(t_max - t_min), silently corrupting all physics residuals. "
            "Use refit_normalizers_on_physical_split() instead.".format(
                coord_range, float(flat_coords.min().item()))
        )

    coord_norm = CoordNormalizer.from_coords(flat_coords)
    # Refit state normalizer from training targets
    state_norm = StateNormalizer.from_targets(train_targets)
    return coord_norm, state_norm


def prepare_pde_tensors(
    system_data,
    observation_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, CoordNormalizer, StateNormalizer]:
    """Prepare PDE tensors with coordinate and state normalisation.

    Observation window: first ``observation_steps`` time snapshots of the field.
    Coordinates: (t, x) pairs normalised to ``[-1, 1]`` in both dimensions.
    Targets: field values normalised to ``[-1, 1]``.
    """
    raw_fields = system_data.fields  # (N, nt, nx)
    num_samples = raw_fields.shape[0]

    raw_targets = torch.tensor(raw_fields.reshape(num_samples, -1, 1), dtype=torch.float32)
    state_norm = StateNormalizer.from_targets(raw_targets)
    targets = state_norm.normalize(raw_targets)

    raw_obs = torch.tensor(raw_fields[:, :observation_steps, :], dtype=torch.float32)
    observations = state_norm.normalize(raw_obs)

    mesh_t, mesh_x = np.meshgrid(system_data.times, system_data.space, indexing="ij")
    flat_coords = torch.tensor(
        np.stack([mesh_t.reshape(-1), mesh_x.reshape(-1)], axis=1).astype(np.float32)
    )  # (nt*nx, 2)

    coord_norm = CoordNormalizer.from_coords(flat_coords)
    norm_flat = coord_norm.normalize(flat_coords)  # (nt*nx, 2)
    coords = norm_flat.unsqueeze(0).expand(num_samples, -1, -1).clone()  # (N, nt*nx, 2)

    params = torch.tensor(system_data.params, dtype=torch.float32)
    return observations, coords, targets, params, coord_norm, state_norm


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

    jacobian_matrix = torch.cat([j.flatten(start_dim=1) for j in jac_dict[0].values()], dim=1)

    s = torch.linalg.svdvals(jacobian_matrix)
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
        
        jac = jacobian(f, x_i, vectorize=False)
        jac = jac.view(jac.shape[1], -1)  # Reshape to (out_dim, flattened_in_dim)
        norm = torch.linalg.matrix_norm(jac, ord=2)
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


def _soft_ao_step(
    loss: torch.Tensor,
    enc_optimizer,
    gen_optimizer,
    encoder_parameters,
    generator_parameters,
    *,
    enc_focus: bool,
    max_grad_norm: float = 1.0,
) -> None:
    encoder_parameters = list(encoder_parameters)
    generator_parameters = list(generator_parameters)
    active_optimizer = enc_optimizer if enc_focus else gen_optimizer
    inactive_optimizer = gen_optimizer if enc_focus else enc_optimizer
    active_parameters = encoder_parameters if enc_focus else generator_parameters
    inactive_parameters = generator_parameters if enc_focus else encoder_parameters

    active_optimizer.zero_grad()
    inactive_optimizer.zero_grad()
    loss.backward()

    for parameter in inactive_parameters:
        parameter.grad = None

    torch.nn.utils.clip_grad_norm_(active_parameters, max_grad_norm)
    active_optimizer.step()


def _triplet_loss(latent: torch.Tensor, params: torch.Tensor, trajectory_ids: torch.Tensor) -> torch.Tensor:
    if latent.shape[0] < 3:
        return latent.sum() * 0.0
    total = latent.sum() * 0.0

    count = 0
    for idx in range(latent.shape[0]):
        sample_indices = torch.arange(latent.shape[0], device=latent.device)
        pos_candidates = torch.where((trajectory_ids == trajectory_ids[idx]) & (sample_indices != idx))[0]
        neg_candidates = torch.where(trajectory_ids != trajectory_ids[idx])[0]
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
        return latent.sum() * 0.0
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
    has_oc = hasattr(model, "observation_conditioner")
    if has_oc:
        if model_kind == "fno":
            pred = model(data_coords, obs)
        else:
            latent = model.observation_conditioner(obs)
            repeated_latent = latent.unsqueeze(1).expand(-1, data_coords.shape[1], -1).reshape(
                latent.shape[0] * data_coords.shape[1], -1
            )
            pred = model.decode(flat_coords, repeated_latent)
    else:
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
    interaction_frequency: int = 5,
    ao_warmup_epochs: int = 10,
    max_data_points: int = 512,
    max_phys_points: int = 256,
    progress_desc: str | None = None,
    use_config: bool = False,
    use_soap: bool = False,
    use_lra: bool = True,
    coord_normalizer: CoordNormalizer | None = None,
    state_normalizer: StateNormalizer | None = None,
    trajectory_ids: torch.Tensor | None = None,
    # --- callback arguments (all optional; default = disabled) ---
    val_bundle: ValBundle | None = None,
    callbacks: CallbackConfig | None = None,
) -> TrainResult:
    # -----------------------------------------------------------------------
    # Chain-rule physics scale: dy/dt_raw = (dy/dt_norm) * coord_scale[0]
    # When coords are normalised to [-1,1], autograd gives d/dt_norm, so we
    # must multiply the residual by coord_scale to recover d/dt_raw.
    # For t_norm ∈ [-1,1] mapping raw t ∈ [t0, t1]: scale = 2/(t1-t0).
    # -----------------------------------------------------------------------
    phys_t_scale = 1.0
    if coord_normalizer is not None and coord_normalizer.coord_scales.numel() >= 1:
        phys_t_scale = float(coord_normalizer.coord_scales[0].item())
    # State normalizer scale: d(y_norm)/dy_raw = 2/(y_max-y_min)
    # Residual in norm space = (dy_norm/dt_raw - f_norm), we need to divide
    # both sides by state_scale to compare to the true unnorm residual, OR
    # simply compute the residual in normalised space consistently.
    # We train purely in normalised space and rescale for reporting.

    if trajectory_ids is None:
        trajectory_ids = torch.arange(observations.shape[0], dtype=torch.long)
    else:
        trajectory_ids = trajectory_ids.to(dtype=torch.long)

    dataset_tensors = [observations, coords, targets, params, trajectory_ids]
    if ode_metadata is not None:
        dataset_tensors.append(ode_metadata)
    dataset = TensorDataset(*dataset_tensors)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    model = model.to(device)

    if use_soap:
        from .soap import SOAP
        optimizer_joint = SOAP(model.parameters(), lr=lr)
    else:
        optimizer_joint = torch.optim.Adam(model.parameters(), lr=lr)

    config_opt_joint = ConFIGOptimizer(optimizer_joint) if use_config else None

    # Soft-AO: separate param groups for encoder vs generator within optimizer_joint
    # We use two Adam optimizers but NEVER hard-freeze either — both always update.
    # During AO warm-up phase we use pure joint. After warm-up, we bias the LR
    # per-component using gradient conflict detection.
    if use_soap:
        from .soap import SOAP
        _opt_enc = SOAP(model.encoder.parameters(), lr=lr)
        _opt_gen = SOAP(model.generator.parameters(), lr=lr)
    else:
        _opt_enc = torch.optim.Adam(model.encoder.parameters(), lr=lr)
        _opt_gen = torch.optim.Adam(model.generator.parameters(), lr=lr)

    history = []
    total_time = 0.0
    alpha_phys_ema = alpha_physics  # LRA-adapted weight (EMA)

    # Build callbacks
    early_stopper: EarlyStopping | None = None
    schedulers: list[torch.optim.lr_scheduler.ReduceLROnPlateau] = []
    if val_bundle is not None and callbacks is not None:
        if callbacks.early_stopping_patience > 0:
            early_stopper = EarlyStopping(patience=callbacks.early_stopping_patience)
        if callbacks.reduce_lr_patience > 0:
            for opt in [optimizer_joint, _opt_enc, _opt_gen]:
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
    for epoch in (epoch_iter if epoch_iter is not None else range(epochs)):
        epoch_start = time.perf_counter()
        running = {"loss": 0.0, "data_mse": 0.0, "physics_mse": 0.0, "metric_loss": 0.0}
        steps = 0

        # Determine AO phase: warm-up = joint only; after = soft-alternating
        in_warmup = (not alternating) or (epoch < ao_warmup_epochs)

        for step_idx, batch in enumerate(loader):
            if ode_metadata is not None:
                obs_batch, coord_batch, target_batch, param_batch, trajectory_batch, ode_meta_batch = batch
                ode_meta_batch = ode_meta_batch.to(device)
            else:
                obs_batch, coord_batch, target_batch, param_batch, trajectory_batch = batch
                ode_meta_batch = None
            obs_batch     = obs_batch.to(device)
            coord_batch   = coord_batch.to(device)
            target_batch  = target_batch.to(device)
            param_batch   = param_batch.to(device)
            trajectory_batch = trajectory_batch.to(device)

            data_coords, data_targets = _sample_coord_subset(coord_batch, target_batch, max_data_points)
            flat_data_coords = data_coords.reshape(-1, coord_batch.shape[-1])
            flat_data_targets = data_targets.reshape(-1, target_batch.shape[-1])

            # ------------------------------------------------------------------
            # Forward pass (shared)
            # ------------------------------------------------------------------
            latent = model.encode(obs_batch)
            assert latent is not None
            expanded_latent = latent.unsqueeze(1).expand(-1, data_coords.shape[1], -1).reshape(-1, latent.shape[-1])
            pred = model.decode(flat_data_coords, expanded_latent)
            data_loss = torch.mean((pred - flat_data_targets) ** 2)
            metric_loss = _triplet_loss(latent, param_batch, trajectory_batch)

            phys_coords, _ = _sample_coord_subset(coord_batch, target_batch, max_phys_points)
            flat_phys_coords = phys_coords.reshape(-1, coord_batch.shape[-1]).detach().clone().requires_grad_(True)
            expanded_phys_latent = latent.unsqueeze(1).expand(-1, phys_coords.shape[1], -1).reshape(-1, latent.shape[-1])
            phys_param = param_batch.unsqueeze(1).expand(-1, phys_coords.shape[1]).reshape(-1)
            phys_pred = model.decode(flat_phys_coords, expanded_phys_latent)

            if flat_phys_coords.shape[-1] == 1:
                phys_metadata = None
                if ode_meta_batch is not None:
                    phys_metadata = {
                        "natural_frequencies": ode_meta_batch.unsqueeze(1)
                        .expand(-1, phys_coords.shape[1], -1)
                        .reshape(-1, ode_meta_batch.shape[-1])
                    }
                # phys_pred is in normalised state-space; compute_ode_residual
                # computes dy/dt_norm.  Scale by phys_t_scale to get dy/dt_raw.
                residual_raw = compute_ode_residual(
                    problem_name, flat_phys_coords, phys_pred, phys_param,
                    metadata=phys_metadata, coord_normalizer=coord_normalizer,
                    state_normalizer=state_normalizer,
                )
            else:
                residual_raw = compute_pde_residual(
                    problem_name, flat_phys_coords, phys_pred[:, 0], phys_param,
                    coord_scales=coord_normalizer.coord_scales if coord_normalizer is not None else None,
                    state_normalizer=state_normalizer,
                )
            physics_loss = torch.mean(residual_raw ** 2)

            # ------------------------------------------------------------------
            # LRA: dynamically adapt alpha_physics (Wang et al. 2021)
            # ------------------------------------------------------------------
            if use_lra and physics_loss.item() > 0.0:
                # Compute gradient norms OUTSIDE torch.no_grad() so the second-order
                # graph (built by _gradient with create_graph=True inside physics_loss)
                # is fully traversable.
                g_data = torch.autograd.grad(
                    data_loss, list(model.parameters()),
                    retain_graph=True, allow_unused=True, create_graph=False,
                )
                g_phys = torch.autograd.grad(
                    physics_loss, list(model.parameters()),
                    retain_graph=True, allow_unused=True, create_graph=False,
                )
                
                with torch.no_grad():
                    norm_d = torch.sqrt(sum((g.detach().norm() ** 2 for g in g_data if g is not None), torch.tensor(0.0, device=device))).clamp(min=1e-8)
                    norm_p = torch.sqrt(sum((g.detach().norm() ** 2 for g in g_phys if g is not None), torch.tensor(0.0, device=device))).clamp(min=1e-8)
                    lambda_hat = float((norm_d / norm_p).item())
                    # Clamp lambda_hat *before* EMA to prevent singular exploding gradients
                    lambda_hat = float(np.clip(lambda_hat, 1e-3, 100.0))
                
                # EMA update: smooth so alpha doesn't oscillate
                alpha_phys_ema = 0.9 * alpha_phys_ema + 0.1 * lambda_hat
                # Final clamp for redundancy
                alpha_phys_ema = float(np.clip(alpha_phys_ema, 1e-3, 10.0))
                # Explicitly delete gradient references to prevent accidental graph retention
                del g_data, g_phys

            eff_alpha = alpha_phys_ema
            loss = data_loss + eff_alpha * physics_loss + beta_metric * metric_loss

            # ------------------------------------------------------------------
            # Backward + update
            # ------------------------------------------------------------------
            if in_warmup or use_config:
                optimizer_joint.zero_grad()
                if use_config:
                    assert config_opt_joint is not None
                    config_opt_joint.step(
                        data_loss + beta_metric * metric_loss,
                        eff_alpha * physics_loss,
                        model.parameters(),
                    )
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer_joint.step()
            else:
                # Soft-AO: alternate encoder/generator focus while keeping the
                # inactive optimizer's Adam state untouched.
                global_step = epoch * len(loader) + step_idx + 1
                enc_focus = (global_step // interaction_frequency) % 2 == 0
                _soft_ao_step(
                    loss,
                    _opt_enc,
                    _opt_gen,
                    model.encoder.parameters(),
                    model.generator.parameters(),
                    enc_focus=enc_focus,
                )

            running["loss"]        += float(loss.item())
            running["data_mse"]    += float(data_loss.item())
            running["physics_mse"] += float(physics_loss.item())
            running["metric_loss"] += float(metric_loss.item())
            steps += 1

        total_time += time.perf_counter() - epoch_start
        epoch_summary = {key: value / max(steps, 1) for key, value in running.items()}
        epoch_summary["alpha_physics"] = alpha_phys_ema
        history.append(epoch_summary)
        epochs_trained = epoch + 1
        if hasattr(epoch_iter, "set_postfix"):
            assert epoch_iter is not None
            epoch_iter.set_postfix(
                loss=f'{epoch_summary["loss"]:.4f}',
                data=f'{epoch_summary["data_mse"]:.4f}',
                phys=f'{epoch_summary["physics_mse"]:.4f}',
                α=f'{alpha_phys_ema:.3f}',
            )

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
    beta_metric: float = 0.1,
    alternating: bool = True,
    interaction_frequency: int = 5,
    ao_warmup_epochs: int = 10,
    max_data_points: int = 512,
    max_phys_points: int = 256,
    progress_desc: str | None = None,
    use_config: bool = False,
    use_soap: bool = False,
    use_lra: bool = True,
    coord_normalizer: CoordNormalizer | None = None,
    state_normalizer: StateNormalizer | None = None,
    trajectory_ids: torch.Tensor | None = None,
    # --- callback arguments ---
    val_bundle: ValBundle | None = None,
    callbacks: CallbackConfig | None = None,
) -> TrainResult:
    phys_t_scale = 1.0
    if coord_normalizer is not None and coord_normalizer.coord_scales.numel() >= 1:
        phys_t_scale = float(coord_normalizer.coord_scales[0].item())

    has_oc = hasattr(model, "observation_conditioner")
    has_physics = getattr(model, "has_physics_loss", True)
    if trajectory_ids is None:
        trajectory_ids = torch.arange(observations.shape[0], dtype=torch.long)
    else:
        trajectory_ids = trajectory_ids.to(dtype=torch.long)

    dataset_tensors = [observations, coords, targets, params, trajectory_ids]
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

    _opt_enc = None
    _opt_gen = None
    if has_oc and has_physics:
        encoder_parameters = list(model.observation_conditioner.parameters())
        generator_parameters = list(model.generator.parameters())
        if use_soap:
            from .soap import SOAP
            _opt_enc = SOAP(encoder_parameters, lr=lr)
            _opt_gen = SOAP(generator_parameters, lr=lr)
        else:
            _opt_enc = torch.optim.Adam(encoder_parameters, lr=lr)
            _opt_gen = torch.optim.Adam(generator_parameters, lr=lr)

    config_opt = ConFIGOptimizer(optimizer) if use_config else None
    history = []
    total_time = 0.0
    alpha_phys_ema = alpha_physics

    early_stopper: EarlyStopping | None = None
    schedulers: list[torch.optim.lr_scheduler.ReduceLROnPlateau] = []
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
    for epoch in (epoch_iter if epoch_iter is not None else range(epochs)):
        epoch_start = time.perf_counter()
        running = {"loss": 0.0, "data_mse": 0.0, "physics_mse": 0.0}
        if has_oc:
            running["metric_loss"] = 0.0
        steps = 0
        in_warmup = (not alternating) or (epoch < ao_warmup_epochs)
        for step_idx, batch in enumerate(loader):
            latent: torch.Tensor | None = None
            if ode_metadata is not None:
                obs_batch, coord_batch, target_batch, param_batch, trajectory_batch, ode_meta_batch = batch
                ode_meta_batch = ode_meta_batch.to(device)
            else:
                obs_batch, coord_batch, target_batch, param_batch, trajectory_batch = batch
                ode_meta_batch = None
            obs_batch     = obs_batch.to(device)
            coord_batch   = coord_batch.to(device)
            target_batch  = target_batch.to(device)
            param_batch   = param_batch.to(device)
            trajectory_batch = trajectory_batch.to(device)
            optimizer.zero_grad()

            data_coords, data_targets = _sample_coord_subset(coord_batch, target_batch, max_data_points)
            flat_data_coords = data_coords.reshape(-1, coord_batch.shape[-1])
            flat_data_targets = data_targets.reshape(-1, target_batch.shape[-1])
            if has_oc:
                latent = model.observation_conditioner(obs_batch)
                assert latent is not None
                metric_loss = _triplet_loss(latent, param_batch, trajectory_batch)
                if model_kind == "fno":
                    pred = model.decode(data_coords, latent)
                    flat_data_targets = data_targets
                else:
                    repeated_latent = latent.unsqueeze(1).expand(-1, data_coords.shape[1], -1).reshape(
                        latent.shape[0] * data_coords.shape[1], -1
                    )
                    pred = model.decode(flat_data_coords, repeated_latent)
            else:
                metric_loss = data_coords.sum() * 0.0
                repeated_params = param_batch.unsqueeze(1).expand(-1, data_coords.shape[1]).reshape(-1)
                if model_kind == "deeponet":
                    branch_input = obs_batch.reshape(obs_batch.shape[0], -1)
                    repeated_branch = branch_input.unsqueeze(1).expand(-1, data_coords.shape[1], -1).reshape(
                        branch_input.shape[0] * data_coords.shape[1], -1)
                    pred = model(repeated_branch, flat_data_coords)
                elif model_kind == "fno":
                    branch_input = obs_batch.reshape(obs_batch.shape[0], -1)
                    pred = model(branch_input, data_coords)
                    flat_data_targets = data_targets
                else:
                    pred = model(flat_data_coords, repeated_params)
            data_loss = torch.mean((pred - flat_data_targets) ** 2)

            if has_physics:
                phys_coords, _ = _sample_coord_subset(coord_batch, target_batch, max_phys_points)
                flat_phys_coords = phys_coords.reshape(-1, coord_batch.shape[-1]).detach().clone().requires_grad_(True)
                phys_params = param_batch.unsqueeze(1).expand(-1, phys_coords.shape[1]).reshape(-1)
                if has_oc:
                    assert latent is not None
                    repeated_latent = latent.unsqueeze(1).expand(-1, phys_coords.shape[1], -1).reshape(
                        latent.shape[0] * phys_coords.shape[1], -1
                    )
                    phys_pred = model.decode(flat_phys_coords, repeated_latent)
                elif model_kind == "deeponet":
                    branch_input = obs_batch.reshape(obs_batch.shape[0], -1)
                    repeated_branch = branch_input.unsqueeze(1).expand(-1, phys_coords.shape[1], -1).reshape(
                        branch_input.shape[0] * phys_coords.shape[1], -1)
                    phys_pred = model(repeated_branch, flat_phys_coords)
                else:
                    phys_pred = model(flat_phys_coords, phys_params)

                if flat_phys_coords.shape[-1] == 1:
                    phys_metadata = None
                    if ode_meta_batch is not None:
                        phys_metadata = {
                            "natural_frequencies": ode_meta_batch.unsqueeze(1)
                            .expand(-1, phys_coords.shape[1], -1)
                            .reshape(-1, ode_meta_batch.shape[-1])
                        }
                    residual = compute_ode_residual(
                        problem_name, flat_phys_coords, phys_pred, phys_params,
                        metadata=phys_metadata, coord_normalizer=coord_normalizer,
                        state_normalizer=state_normalizer,
                    )
                else:
                    residual = compute_pde_residual(
                        problem_name, flat_phys_coords, phys_pred[:, 0], phys_params,
                        coord_scales=coord_normalizer.coord_scales if coord_normalizer is not None else None,
                        state_normalizer=state_normalizer,
                    )
                physics_loss = torch.mean(residual ** 2)
            else:
                physics_loss = data_loss * 0.0

            if has_physics and use_lra and physics_loss.item() > 0.0:
                # Compute gradient norms OUTSIDE torch.no_grad()
                g_data = torch.autograd.grad(
                    data_loss, list(model.parameters()),
                    retain_graph=True, allow_unused=True, create_graph=False,
                )
                g_phys = torch.autograd.grad(
                    physics_loss, list(model.parameters()),
                    retain_graph=True, allow_unused=True, create_graph=False,
                )
                
                with torch.no_grad():
                    norm_d = torch.sqrt(sum((g.detach().norm() ** 2 for g in g_data if g is not None), torch.tensor(0.0, device=device))).clamp(min=1e-8)
                    norm_p = torch.sqrt(sum((g.detach().norm() ** 2 for g in g_phys if g is not None), torch.tensor(0.0, device=device))).clamp(min=1e-8)
                    lambda_hat = float((norm_d / norm_p).item())
                    lambda_hat = float(np.clip(lambda_hat, 1e-3, 100.0))
                
                alpha_phys_ema = 0.9 * alpha_phys_ema + 0.1 * lambda_hat
                alpha_phys_ema = float(np.clip(alpha_phys_ema, 1e-3, 10.0))
                del g_data, g_phys

            eff_alpha = alpha_phys_ema if has_physics else 0.0
            if has_oc and has_physics:
                loss = data_loss + eff_alpha * physics_loss + beta_metric * metric_loss
            elif has_oc:
                loss = data_loss + beta_metric * metric_loss
            else:
                loss = data_loss + eff_alpha * physics_loss

            if use_config:
                assert config_opt is not None
                config_opt.step(data_loss, eff_alpha * physics_loss, model.parameters())
            elif has_oc and has_physics and not in_warmup:
                global_step = epoch * len(loader) + step_idx + 1
                enc_focus = (global_step // interaction_frequency) % 2 == 0
                assert _opt_enc is not None and _opt_gen is not None
                _soft_ao_step(
                    loss,
                    _opt_enc,
                    _opt_gen,
                    model.observation_conditioner.parameters(),
                    model.generator.parameters(),
                    enc_focus=enc_focus,
                )
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            running["loss"]        += float(loss.item())
            running["data_mse"]    += float(data_loss.item())
            running["physics_mse"] += float(physics_loss.item())
            if has_oc:
                running["metric_loss"] += float(metric_loss.item())
            steps += 1

        total_time += time.perf_counter() - epoch_start
        epoch_summary = {key: value / max(steps, 1) for key, value in running.items()}
        epoch_summary["alpha_physics"] = alpha_phys_ema
        history.append(epoch_summary)
        epochs_trained += 1
        if hasattr(epoch_iter, "set_postfix"):
            assert epoch_iter is not None
            epoch_iter.set_postfix(
                loss=f'{epoch_summary["loss"]:.4f}',
                data=f'{epoch_summary["data_mse"]:.4f}',
                phys=f'{epoch_summary["physics_mse"]:.4f}',
                α=f'{alpha_phys_ema:.3f}',
            )

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
    schedulers: list[torch.optim.lr_scheduler.ReduceLROnPlateau] = []
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
    for _ in (epoch_iter if epoch_iter is not None else range(epochs)):
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
            assert epoch_iter is not None
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
def predict_tapinn(
    model,
    observations: torch.Tensor,
    coords: torch.Tensor,
    device: torch.device,
    state_normalizer: StateNormalizer | None = None,
) -> torch.Tensor:
    """Run TAPINN inference.  ``coords`` should already be normalised.

    Returns predictions in the original (un-normalised) state space when
    ``state_normalizer`` is provided; otherwise returns the raw network output.
    """
    model.eval()
    latent = model.encode(observations.to(device))
    expanded_latent = latent.unsqueeze(1).expand(-1, coords.shape[1], -1).reshape(-1, latent.shape[-1])
    pred_norm = model.decode(coords.to(device).reshape(-1, coords.shape[-1]), expanded_latent)
    pred_norm = pred_norm.reshape(observations.shape[0], coords.shape[1], -1)
    if state_normalizer is not None:
        return state_normalizer.denormalize(pred_norm)
    return pred_norm


@torch.no_grad()
def predict_direct(
    model,
    model_kind: str,
    observations: torch.Tensor,
    coords: torch.Tensor,
    params: torch.Tensor,
    device: torch.device,
    state_normalizer: StateNormalizer | None = None,
) -> torch.Tensor:
    """Run direct-model inference.  ``coords`` should already be normalised.

    Returns predictions in original state space when ``state_normalizer`` given.
    """
    model.eval()
    flat_coords = coords.to(device).reshape(-1, coords.shape[-1])
    repeated_params = params.to(device).unsqueeze(1).expand(-1, coords.shape[1]).reshape(-1)
    if model_kind == "deeponet":
        branch_input = observations.reshape(observations.shape[0], -1).to(device)
        repeated_branch = branch_input.unsqueeze(1).expand(-1, coords.shape[1], -1).reshape(-1, branch_input.shape[-1])
        pred_norm = model(repeated_branch, flat_coords)
    else:
        pred_norm = model(flat_coords, repeated_params)
    pred_norm = pred_norm.reshape(observations.shape[0], coords.shape[1], -1)
    if state_normalizer is not None:
        return state_normalizer.denormalize(pred_norm)
    return pred_norm


@torch.no_grad()
def predict_fno(model, observations: torch.Tensor, target_points: int, device: torch.device) -> torch.Tensor:
    """FNO inference.  Grid is always normalised internally to [0, 1]."""
    model.eval()
    branch_input = observations.reshape(observations.shape[0], -1).to(device)
    grid = torch.linspace(0.0, 1.0, target_points, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
    pred = model(branch_input, grid.expand(branch_input.shape[0], -1, -1))
    return pred
