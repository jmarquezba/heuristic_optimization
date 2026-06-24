"""Generador de resultados por etapas (para entornos con límite de tiempo).

Escribe CSVs parciales que luego se combinan, de modo que cada etapa quepa en una
ejecución corta. Utilitario de demostración; el experimento completo se corre con
`python run_all.py --mode full`.

Uso:
    python scripts/gen_stage.py gd
    python scripts/gen_stage.py heur 2
    python scripts/gen_stage.py heur 3
    python scripts/gen_stage.py combine
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimization.functions import ALL_FUNCTIONS, get_function
from src.optimization.gradient_descent import gradient_descent
from src.optimization.evolutionary import evolutionary_algorithm
from src.optimization.pso import particle_swarm
from src.optimization.differential_evolution import differential_evolution
from src.optimization.metrics import summarize
from src.utils.config import RESULTS_DIR, load_config, ensure_dirs
from src.utils.seed import derive_seed

ensure_dirs()
PART = RESULTS_DIR / "_parts"
PART.mkdir(parents=True, exist_ok=True)

GD_NRUNS = [30, 60]
GD_MAX_ITER = 200
HEUR_NRUNS = 10
SEED = 1983

_ALGS = {"evolutionary": evolutionary_algorithm, "pso": particle_swarm,
         "differential_evolution": differential_evolution}
_OFF = {"evolutionary": 200, "pso": 300, "differential_evolution": 400}


def _gap(fb, fo):
    return abs(fb - fo)


def stage_gd():
    raw = load_config("numerical_experiments.yaml")["fast"]["gradient_descent"]
    rows = []
    for fn in ALL_FUNCTIONS:
        for dim in (2, 3):
            spec = get_function(fn, dim)
            lr = raw.get("lr_per_function", {}).get(fn, raw["lr"])
            for n in GD_NRUNS:
                for r in range(n):
                    res = gradient_descent(spec, seed=derive_seed(SEED + 100, r),
                                           lr=lr, max_iter=GD_MAX_ITER,
                                           grad_tol=raw["grad_tol"], f_tol=raw["f_tol"])
                    row = res.to_row()
                    row["n_runs_group"] = n
                    row["gap"] = _gap(res.f_best, spec.f_opt)
                    rows.append(row)
    pd.DataFrame(rows).to_csv(PART / "gd.csv", index=False)
    print("GD listo: %d filas" % len(rows))


def stage_heur(dim):
    raw = load_config("numerical_experiments.yaml")["fast"]["heuristics"]
    rows = []
    for fn in ALL_FUNCTIONS:
        spec = get_function(fn, dim)
        for alg, fnc in _ALGS.items():
            params = dict(raw.get(alg, {}))
            for r in range(HEUR_NRUNS):
                res = fnc(spec, seed=derive_seed(SEED + _OFF[alg], r), **params)
                row = res.to_row()
                row["n_runs_group"] = HEUR_NRUNS
                row["gap"] = _gap(res.f_best, spec.f_opt)
                rows.append(row)
    pd.DataFrame(rows).to_csv(PART / ("heur%d.csv" % dim), index=False)
    print("Heuristicos %dD listo: %d filas" % (dim, len(rows)))


def stage_combine():
    parts = [pd.read_csv(p) for p in sorted(PART.glob("*.csv"))]
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(RESULTS_DIR / "numerical_results.csv", index=False)
    tol = load_config("numerical_experiments.yaml").get("success_tol", {})
    summary = summarize(df, success_tol=tol)
    summary.to_csv(RESULTS_DIR / "numerical_summary.csv", index=False)
    print("Combinado: %d filas" % len(df))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gd":
        stage_gd()
    elif cmd == "heur":
        stage_heur(int(sys.argv[2]))
    elif cmd == "combine":
        stage_combine()
