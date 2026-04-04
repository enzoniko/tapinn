from __future__ import annotations

import torch


def get_best_device(preferred: str | None = None) -> torch.device:
    """Return the best available device with CUDA > MPS > CPU fallback."""
    if preferred and preferred != "auto":
        return torch.device(preferred)

    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
