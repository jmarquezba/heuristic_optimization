"""Estructuras de resultado y métricas para optimización numérica."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class OptResult:
    """Resultado de una corrida de optimización."""
    algorithm: str
    function: str
    dim: int
    x_best: np.ndarray
    f_best: float
    n_evals: int
    n_iter: int
    seed: int
    runtime_s: float
    converged: bool = False
    history: list[float] = field(default_factory=list)   # mejor f por iteración
    trajectory: list[np.ndarray] = field(default_factory=list)  # para animaciones

    def to_row(self) -> dict:
        """Fila plana para CSV (sin trayectoria/historial)."""
        row = {
            "algorithm": self.algorithm,
            "function": self.function,
            "dim": self.dim,
            "f_best": self.f_best,
            "n_evals": self.n_evals,
            "n_iter": self.n_iter,
            "seed": self.seed,
            "runtime_s": self.runtime_s,
            "converged": self.converged,
        }
        for i, xi in enumerate(np.atleast_1d(self.x_best)):
            row[f"x{i}"] = float(xi)
        return row


def summarize(df: pd.DataFrame, success_tol: dict | None = None) -> pd.DataFrame:
    """Construye la tabla resumen por (function, dim, algorithm).

    Incluye media/mediana/desv/mejor/peor del valor final, media/mediana de NFE,
    tasa de éxito (si se da ``success_tol`` por función) y tiempo medio.
    """
    success_tol = success_tol or {}
    rows = []
    for (fn, dim, alg), g in df.groupby(["function", "dim", "algorithm"]):
        f = g["f_best"].to_numpy()
        tol = success_tol.get(fn)
        if tol is not None:
            # f_opt se resta fuera; aquí se asume distancia al óptimo ya en f_best
            success_rate = float(np.mean(g["gap"].to_numpy() < tol)) if "gap" in g else np.nan
        else:
            success_rate = np.nan
        rows.append({
            "function": fn,
            "dim": dim,
            "algorithm": alg,
            "n_runs": len(g),
            "f_mean": float(np.mean(f)),
            "f_median": float(np.median(f)),
            "f_std": float(np.std(f)),
            "f_best": float(np.min(f)),
            "f_worst": float(np.max(f)),
            "evals_mean": float(np.mean(g["n_evals"])),
            "evals_median": float(np.median(g["n_evals"])),
            "success_rate": success_rate,
            "runtime_mean_s": float(np.mean(g["runtime_s"])),
            "seed_base": int(g["seed"].iloc[0]),
        })
    return pd.DataFrame(rows).sort_values(["function", "dim", "algorithm"])
