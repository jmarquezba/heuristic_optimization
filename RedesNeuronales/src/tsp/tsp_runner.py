"""Orquestador del TSP de Francia (Parte 2).

Para cada escenario de valor hora (bajo, medio, alto) corre ACO y GA, guarda
resultados, rutas y resumen, y genera mapas/animaciones de la mejor solución.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .ant_colony import ant_colony_tsp
from .cost_model import CostParams, Vehicle, build_cost_matrices, route_breakdown, save_base_matrices
from .data_loader import build_dataset
from .genetic_algorithm import genetic_algorithm_tsp
from ..utils.config import GIFS_DIR, RESULTS_DIR
from ..utils.io import save_df
from ..utils.logging import get_logger
from ..visualization.maps import animate_route_evolution, plot_route, plot_route_comparison

logger = get_logger("tsp_runner")


def run_tsp(cfg: dict, make_animations: bool = True) -> dict:
    """Ejecuta el TSP completo para todos los escenarios de valor hora."""
    df = build_dataset(write=True)
    vehicle = Vehicle(**cfg.get("vehicle", {}))
    params = CostParams(**cfg.get("cost_params", {}))
    scenarios = cfg["valor_hora_scenarios"]   # {"bajo": 15, "medio": 30, "alto": 50}
    seed = cfg["seed_base"]
    aco_cfg = cfg.get("aco", {})
    ga_cfg = cfg.get("ga", {})

    rows, route_rows = [], []
    best_overall = {}

    for scen_name, valor_hora in scenarios.items():
        mats = build_cost_matrices(df, vehicle, params, valor_hora)
        if scen_name == cfg.get("base_scenario", "medio"):
            save_base_matrices(df, mats)
        cost = mats["total"]

        aco = ant_colony_tsp(cost, seed=seed, **aco_cfg)
        ga = genetic_algorithm_tsp(cost, seed=seed, **ga_cfg)

        for algo_name, res, n_it, best_it in (
            ("aco", aco, aco.n_iter, aco.best_iter),
            ("ga", ga, ga.n_gen, ga.best_gen),
        ):
            bd = route_breakdown(res.best_route, mats)
            rows.append({
                "algorithm": algo_name, "escenario": scen_name, "valor_hora": valor_hora,
                "cost_total": bd["cost_total"], "distance_total_km": bd["distance_total_km"],
                "time_total_h": bd["time_total_h"], "toll_total": bd["toll_total"],
                "fuel_total": bd["fuel_total"],
                "cost_time": bd["time_total_h"] * valor_hora,
                "n_iter": n_it, "best_iter": best_it,
                "runtime_s": res.runtime_s, "seed": res.seed,
            })
            route_rows.append({
                "algorithm": algo_name, "escenario": scen_name,
                "route_prefecturas": " -> ".join(df["prefectura"].to_numpy()[res.best_route]),
                "route_indices": ",".join(map(str, res.best_route)),
                "cost_total": bd["cost_total"],
            })
        logger.info("Escenario %s: ACO=%.0f€ GA=%.0f€", scen_name,
                    aco.best_cost, ga.best_cost)
        best_overall[scen_name] = {"aco": aco, "ga": ga, "df": df}

    results = pd.DataFrame(rows)
    save_df(results, RESULTS_DIR / "tsp_results.csv")
    save_df(pd.DataFrame(route_rows), RESULTS_DIR / "tsp_best_routes.csv")

    summary = (results.groupby(["algorithm", "escenario"])
               .agg(cost_total=("cost_total", "mean"),
                    distance_total_km=("distance_total_km", "mean"),
                    runtime_s=("runtime_s", "mean")).reset_index())
    save_df(summary, RESULTS_DIR / "tsp_summary.csv")

    # Visualizaciones para el escenario base
    base = cfg.get("base_scenario", "medio")
    b = best_overall[base]
    plot_route(b["df"], b["aco"].best_route, "Mejor ruta ACO (Francia continental)",
               color="#d33")
    plot_route(b["df"], b["ga"].best_route, "Mejor ruta GA (Francia continental)",
               color="#36b")
    plot_route_comparison(b["df"], b["aco"].best_route, b["ga"].best_route)
    if make_animations:
        animate_route_evolution(b["df"], b["aco"].best_route_per_iter,
                                b["aco"].history, algorithm="aco")
        animate_route_evolution(b["df"], b["ga"].best_route_per_gen,
                                b["ga"].history, algorithm="ga")

    return {"results": results, "summary": summary, "best": best_overall}
