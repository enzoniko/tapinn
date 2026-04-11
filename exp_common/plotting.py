from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Standardized color palette for NeurIPS submission consistency
MODEL_PALETTE = {
    "tapinn":        "#4C72B0",  # Muted Blue
    "tapinn_ao":     "#4C72B0",
    "tapinn_joint":  "#55A868",  # Muted Green
    "tapinn_config": "#C44E52",  # Muted Red
    "tapinn_soap":   "#8172B2",  # Muted Purple
    "standard_pinn": "#DD8452",  # Muted Orange
    "hyperpinn":     "#8172B2",
    "hyper_lr_pinn": "#76A5AF",  # Muted Cyan
    "deeponet":      "#937860",  # Muted Brown
    "fno":           "#DD8452",
}


def get_model_color(model_name: str) -> str:
    """Return a consistent color for a given model name."""
    if model_name in MODEL_PALETTE:
        return MODEL_PALETTE[model_name]
    # Fallback to root name (e.g. tapinn_large -> tapinn)
    for key in MODEL_PALETTE:
        if model_name.startswith(key):
            return MODEL_PALETTE[key]
    return "#8c8c8c"  # Neutral grey fallback


def moving_average(data: np.ndarray | list[float], window: int = 5) -> np.ndarray:
    """Smooth erratic training or diagnostic curves."""
    x = np.asarray(data)
    if len(x) <= window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.format": "pdf",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )


def save_figure(fig: plt.Figure, path: str | Path, tight: bool = True) -> None:
    """Save figure with consistent NeurIPS settings."""
    if tight:
        fig.tight_layout()
    # Use bbox_inches='tight' by default to ensure legend isn't cut off if placed outside
    fig.savefig(Path(path), format="pdf", bbox_inches="tight")
    plt.close(fig)
