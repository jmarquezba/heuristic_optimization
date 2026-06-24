"""Orquestador de experimentos de optimización numérica (Parte 1).

Ejecuta:
- Descenso por gradiente con n corridas (n = 100, 500, 1000 en modo full).
- EA, PSO y DE con >= 30 corridas independientes.

Persiste ``numerical_results.csv`` (una fila por corrida) y
``numerical_summary.csv`` (agregados). Las semillas se derivan de una base para
reproducibilidad (``utils/seed.derive_seed``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .differential_evolution import differential_evolution
from .evolutionary import evolutionary_algorithm
from .functions import ALL_FUNCTIONS, get_function
from .gradient_descent import gradient_descent
from .metrics import OptResult, summarize
from .pso import particle_swarm
from ..utils.config import RESULTS_DIR
from ..utils.io import save_df
from ..utils.logging import get_logger
from ..utils.seed import derive_seed

logger = get_logger("experiment_runner")

DEFAULT_DIMS = (2, 3)


def _gap(fn: str, dim: int, f_best: float, f_opt: float) -> float:
    return abs(f_best - f_opt)


def run_gradient_descent(cfg: dict) -> list[dict]:
    """Corridas de GD para todas las funciones, dims y valores de n."""
    rows: list[dict] = []
    gd_cfg = cfg["gradient_descent"]
    n_runs_list = gd_cfg["n_runs"]
    base_seed = cfg["seed_base"]
    for fn in cfg.get("functions", ALL_FUNCTIONS):
        for dim in cfg.get("dims", DEFAULT_DIMS):
            spec = get_function(fn, dim)
            lr = gd_cfg.get("lr_per_function", {}).get(fn, gd_cfg["lr"])
            for n_runs in n_runs_list:
                for r in range(n_runs):
                    seed = derive_seed(base_seed + 100, r)
                    res = gradient_descent(
                        spec, seed=seed, lr=lr,
                        max_iter=gd_cfg["max_iter"],
                        grad_tol=gd_cfg["grad_tol"],
                        f_tol=gd_cfg["f_tol"],
                    )
                    row = res.to_row()
                    row["n_runs_group"] = n_runs
                    row["gap"] = _gap(fn, dim, res.f_best, spec.f_opt)
                    rows.append(row)
                logger.info("GD %s d=%d n=%d listo", fn, dim, n_runs)
    return rows


_HEURISTICS = {
    "evolutionary": evolutionary_algorithm,
    "pso": particle_swarm,
    "differential_evolution": differential_evolution,
}

# Offsets deterministas por algoritmo (evita hash() no reproducible).
_ALG_SEED_OFFSET = {
    "evolutionary": 200,
    "pso": 300,
    "differential_evolution": 400,
}


def run_heuristics(cfg: dict) -> list[dict]:
    """Corridas de EA, PSO y DE (>= 30 por combinación en full)."""
    rows: list[dict] = []
    base_seed = cfg["seed_base"]
    n_runs = cfg["heuristics"]["n_runs"]
    for fn in cfg.get("functions", ALL_FUNCTIONS):
        for dim in cfg.get("dims", DEFAULT_DIMS):
            spec = get_function(fn, dim)
            for alg_name, alg_fn in _HEURISTICS.items():
                params = dict(cfg["heuristics"].get(alg_name, {}))
                for r in range(n_runs):
                    seed = derive_seed(base_seed + _ALG_SEED_OFFSET[alg_name], r)
                    res: OptResult = alg_fn(spec, seed=seed, **params)
                    row = res.to_row()
                    row["n_runs_group"] = n_runs
                    row["gap"] = _gap(fn, dim, res.f_best, spec.f_opt)
                    rows.append(row)
                logger.info("%s %s d=%d (%d corridas) listo", alg_name, fn, dim, n_runs)
    return rows


def run_all(cfg: dict, out_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta GD + heurísticos y guarda CSVs de resultados y resumen."""
    out_dir = Path(out_dir or RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = run_gradient_descent(cfg) + run_heuristics(cfg)
    df = pd.DataFrame(rows)
    save_df(df, out_dir / "numerical_results.csv")

    success_tol = cfg.get("success_tol", {})
    summary = summarize(df, success_tol=success_tol)
    save_df(summary, out_dir / "numerical_summary.csv")
    logger.info("Resultados guardados en %s", out_dir)
    return df, summary
