"""Histogramas de resultados de optimización numérica (Parte 1, numeral 1).

Genera, para cada combinación función/dim/algoritmo y grupo de n corridas:
- histograma del valor final de la función objetivo,
- histograma del número de evaluaciones (NFE),
- histograma de la solución final por coordenada.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..utils.config import FIGURES_DIR


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def histogram_final_values(df: pd.DataFrame, fn: str, dim: int, alg: str,
                           n_group: int, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir or FIGURES_DIR)
    sub = df[(df.function == fn) & (df.dim == dim) & (df.algorithm == alg)
             & (df.n_runs_group == n_group)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sub["f_best"], bins=30, color="#3b6", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Valor final f(x*)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"{alg} · {fn} {dim}D · n={n_group}\nValor final de la función objetivo")
    ax.grid(alpha=0.3)
    path = out_dir / f"{alg}_{fn}_{dim}d_n{n_group}_final_values.png"
    _save(fig, path)
    return path


def histogram_evaluations(df: pd.DataFrame, fn: str, dim: int, alg: str,
                          n_group: int, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir or FIGURES_DIR)
    sub = df[(df.function == fn) & (df.dim == dim) & (df.algorithm == alg)
             & (df.n_runs_group == n_group)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sub["n_evals"], bins=30, color="#36b", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Número de evaluaciones de f (NFE)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"{alg} · {fn} {dim}D · n={n_group}\nNúmero de evaluaciones")
    ax.grid(alpha=0.3)
    path = out_dir / f"{alg}_{fn}_{dim}d_n{n_group}_evaluations.png"
    _save(fig, path)
    return path


def histogram_solution_coords(df: pd.DataFrame, fn: str, dim: int, alg: str,
                              n_group: int, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir or FIGURES_DIR)
    sub = df[(df.function == fn) & (df.dim == dim) & (df.algorithm == alg)
             & (df.n_runs_group == n_group)]
    coord_cols = [c for c in sub.columns if c.startswith("x") and c[1:].isdigit()]
    fig, axes = plt.subplots(1, len(coord_cols), figsize=(4 * len(coord_cols), 4),
                             squeeze=False)
    for k, c in enumerate(coord_cols):
        ax = axes[0][k]
        ax.hist(sub[c].dropna(), bins=30, color="#b63", edgecolor="black", alpha=0.8)
        ax.set_xlabel(f"x[{c[1:]}] final")
        ax.set_ylabel("Frecuencia")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{alg} · {fn} {dim}D · n={n_group} · Solución final por coordenada")
    path = out_dir / f"{alg}_{fn}_{dim}d_n{n_group}_solution_coords.png"
    _save(fig, path)
    return path


def generate_gd_histograms(df: pd.DataFrame, out_dir: Path | None = None) -> list[Path]:
    """Genera los histogramas requeridos para el descenso por gradiente."""
    paths: list[Path] = []
    gd = df[df.algorithm == "gradient_descent"]
    for (fn, dim, n_group), _ in gd.groupby(["function", "dim", "n_runs_group"]):
        paths.append(histogram_final_values(df, fn, dim, "gradient_descent", n_group, out_dir))
        paths.append(histogram_evaluations(df, fn, dim, "gradient_descent", n_group, out_dir))
        paths.append(histogram_solution_coords(df, fn, dim, "gradient_descent", n_group, out_dir))
    return paths
