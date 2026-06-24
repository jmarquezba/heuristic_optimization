"""Curvas de convergencia (mejor f por iteración) y boxplots comparativos."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.config import FIGURES_DIR


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_convergence(histories: dict[str, list[float]], fn: str, dim: int,
                     out_dir: Path | None = None) -> Path:
    """Grafica varias curvas de convergencia (una por algoritmo)."""
    out_dir = Path(out_dir or FIGURES_DIR)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for alg, h in histories.items():
        ax.plot(np.arange(len(h)), h, label=alg, lw=1.8)
    ax.set_yscale("symlog")
    ax.set_xlabel("Iteración / generación")
    ax.set_ylabel("Mejor f hasta la iteración")
    ax.set_title(f"Convergencia · {fn} {dim}D")
    ax.legend()
    ax.grid(alpha=0.3)
    path = out_dir / f"convergence_{fn}_{dim}d.png"
    _save(fig, path)
    return path


def boxplot_final_values(df: pd.DataFrame, fn: str, dim: int,
                         out_dir: Path | None = None) -> Path:
    """Boxplot del valor final por algoritmo para una función/dim."""
    out_dir = Path(out_dir or FIGURES_DIR)
    sub = df[(df.function == fn) & (df.dim == dim)]
    algs = sorted(sub.algorithm.unique())
    data = [sub[sub.algorithm == a]["f_best"].to_numpy() for a in algs]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(data, labels=algs, showfliers=True)
    ax.set_ylabel("Valor final f(x*)")
    ax.set_title(f"Distribución del valor final · {fn} {dim}D")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3)
    path = out_dir / f"boxplot_{fn}_{dim}d.png"
    _save(fig, path)
    return path
