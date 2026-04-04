from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


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


def save_figure(fig: plt.Figure, path: str | Path) -> None:
    fig.tight_layout()
    fig.savefig(Path(path), format="pdf", bbox_inches="tight")
    plt.close(fig)
