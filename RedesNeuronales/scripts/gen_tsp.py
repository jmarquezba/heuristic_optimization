"""Generador de resultados del TSP por escenarios (entornos con límite de tiempo).

Uso:
    python scripts/gen_tsp.py bajo
    python scripts/gen_tsp.py medio
    python scripts/gen_tsp.py alto
    python scripts/gen_tsp.py combine     # combina CSVs + mapas del escenario medio
    python scripts/gen_tsp.py gifs        # animaciones de la mejor ruta (medio)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tsp.ant_colony import ant_colony_tsp
from src.tsp.cost_model import (CostParams, Vehicle, build_cost_matrices,
                                route_breakdown, save_base_matrices)
from src.tsp.data_loader import build_dataset
from src.tsp.genetic_algorithm import genetic_algorithm_tsp
from src.utils.config import RESULTS_DIR, load_config
from src.visualization.maps import (animate_route_evolution, plot_route,
                                     plot_route_comparison)

PART = RESULTS_DIR / "_tsp_parts"
PART.mkdir(parents=True, exist_ok=True)

RAW = load_config("tsp_experiments.yaml")
CFG = {k: v for k, v in RAW.items() if k not in ("fast", "full")}
CFG.update(RAW["fast"])
SCEN = CFG["valor_hora_scenarios"]


def _setup():
    df = build_dataset(write=True)
    veh = Vehicle(**CFG.get("vehicle", {}))
    par = CostParams(**CFG.get("cost_params", {}))
    return df, veh, par


def stage_scenario(name: str):
    df, veh, par = _setup()
    valor_hora = SCEN[name]
    mats = build_cost_matrices(df, veh, par, valor_hora)
    if name == CFG.get("base_scenario", "medio"):
        save_base_matrices(df, mats)
    cost = mats["total"]
    aco = ant_colony_tsp(cost, seed=CFG["seed_base"], **CFG["aco"])
    ga = genetic_algorithm_tsp(cost, seed=CFG["seed_base"], **CFG["ga"])

    rows, routes = [], []
    for algo, res, nit, bit in (("aco", aco, aco.n_iter, aco.best_iter),
                                ("ga", ga, ga.n_gen, ga.best_gen)):
        bd = route_breakdown(res.best_route, mats)
        rows.append({"algorithm": algo, "escenario": name, "valor_hora": valor_hora,
                     "cost_total": bd["cost_total"], "distance_total_km": bd["distance_total_km"],
                     "time_total_h": bd["time_total_h"], "toll_total": bd["toll_total"],
                     "fuel_total": bd["fuel_total"], "cost_time": bd["time_total_h"] * valor_hora,
                     "n_iter": nit, "best_iter": bit, "runtime_s": res.runtime_s, "seed": res.seed})
        routes.append({"algorithm": algo, "escenario": name,
                       "route_prefecturas": " -> ".join(df["prefectura"].to_numpy()[res.best_route]),
                       "route_indices": ",".join(map(str, res.best_route)),
                       "cost_total": bd["cost_total"]})
    pd.DataFrame(rows).to_csv(PART / f"res_{name}.csv", index=False)
    pd.DataFrame(routes).to_csv(PART / f"route_{name}.csv", index=False)
    if name == CFG.get("base_scenario", "medio"):
        with open(PART / "base_obj.pkl", "wb") as fh:
            pickle.dump({"aco": aco, "ga": ga}, fh)
    print(f"{name}: ACO={aco.best_cost:.0f}€  GA={ga.best_cost:.0f}€")


def stage_combine():
    res = pd.concat([pd.read_csv(p) for p in sorted(PART.glob("res_*.csv"))], ignore_index=True)
    res.to_csv(RESULTS_DIR / "tsp_results.csv", index=False)
    rt = pd.concat([pd.read_csv(p) for p in sorted(PART.glob("route_*.csv"))], ignore_index=True)
    rt.to_csv(RESULTS_DIR / "tsp_best_routes.csv", index=False)
    summ = (res.groupby(["algorithm", "escenario"])
            .agg(cost_total=("cost_total", "mean"),
                 distance_total_km=("distance_total_km", "mean"),
                 runtime_s=("runtime_s", "mean")).reset_index())
    summ.to_csv(RESULTS_DIR / "tsp_summary.csv", index=False)
    # Mapas del escenario base
    df, _, _ = _setup()
    with open(PART / "base_obj.pkl", "rb") as fh:
        b = pickle.load(fh)
    plot_route(df, b["aco"].best_route, "Mejor ruta ACO (Francia continental)", color="#d33")
    plot_route(df, b["ga"].best_route, "Mejor ruta GA (Francia continental)", color="#36b")
    plot_route_comparison(df, b["aco"].best_route, b["ga"].best_route)
    print("Combinado + mapas listos")
    print(summ.to_string(index=False))


def stage_gifs():
    df, _, _ = _setup()
    with open(PART / "base_obj.pkl", "rb") as fh:
        b = pickle.load(fh)
    animate_route_evolution(df, b["aco"].best_route_per_iter, b["aco"].history,
                            algorithm="aco", max_frames=60)
    animate_route_evolution(df, b["ga"].best_route_per_gen, b["ga"].history,
                            algorithm="ga", max_frames=60)
    print("Gifs TSP listos")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd in SCEN:
        stage_scenario(cmd)
    elif cmd == "combine":
        stage_combine()
    elif cmd == "gifs":
        stage_gifs()
