"""Pipeline completo del proyecto (Fase 18).

Ejecuta, en orden: preparación de datos, experimentos numéricos, histogramas,
animaciones numéricas, TSP, mapas/animaciones TSP y tablas resumen.

Uso:
    python run_all.py --mode fast    # validación rápida (por defecto)
    python run_all.py --mode full    # cumple los requisitos del enunciado
    python run_all.py --only numeric # solo Parte 1
    python run_all.py --only tsp      # solo Parte 2
    python run_all.py --no-animations # omite gifs (más rápido)

El modo full corre GD con n = 100, 500 y 1000 (requisito) y >= 30 corridas
heurísticas. Puede tardar; el modo fast valida el código en minutos.
"""
from __future__ import annotations

import argparse
import time

from src.optimization import experiment_runner
from src.optimization.functions import get_function
from src.optimization.gradient_descent import gradient_descent
from src.optimization.pso import particle_swarm
from src.tsp import tsp_runner
from src.utils.config import GIFS_DIR, load_config, ensure_dirs
from src.utils.logging import get_logger
from src.visualization.animations import animate_gradient_descent, animate_population
from src.visualization.convergence_plots import boxplot_final_values
from src.visualization.histograms import generate_gd_histograms

logger = get_logger("run_all")


def _merge(cfg: dict, mode: str) -> dict:
    """Aplana el perfil (fast/full) sobre las claves comunes."""
    merged = {k: v for k, v in cfg.items() if k not in ("fast", "full")}
    merged.update(cfg[mode])
    return merged


def run_numeric(mode: str, make_animations: bool) -> None:
    raw = load_config("numerical_experiments.yaml")
    cfg = _merge(raw, mode)
    logger.info("=== Parte 1: optimización numérica (modo %s) ===", mode)
    df, summary = experiment_runner.run_all(cfg)

    logger.info("Generando histogramas de GD...")
    generate_gd_histograms(df)
    for fn in cfg["functions"]:
        for dim in cfg["dims"]:
            try:
                boxplot_final_values(df, fn, dim)
            except Exception as e:  # pragma: no cover
                logger.warning("Boxplot %s %dD falló: %s", fn, dim, e)

    if make_animations:
        logger.info("Generando animaciones (Rastrigin 2D)...")
        spec = get_function("rastrigin", 2)
        gd = gradient_descent(spec, seed=7, lr=1e-3, max_iter=800, record_trajectory=True)
        animate_gradient_descent(spec, gd.trajectory, gd.history)
        pso = particle_swarm(spec, seed=7, n_particles=30,
                             max_iter=cfg["heuristics"]["pso"]["max_iter"],
                             record_trajectory=True)
        animate_population(spec, pso.trajectory, pso.history, algorithm="pso")
    logger.info("Parte 1 completada. Resumen:\n%s", summary.to_string(index=False))


def run_tsp_part(mode: str, make_animations: bool) -> None:
    raw = load_config("tsp_experiments.yaml")
    cfg = _merge(raw, mode)
    logger.info("=== Parte 2: TSP Francia (modo %s) ===", mode)
    out = tsp_runner.run_tsp(cfg, make_animations=make_animations)
    logger.info("Parte 2 completada. Resumen:\n%s", out["summary"].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline de optimización IRNA 2026-01")
    ap.add_argument("--mode", choices=["fast", "full"], default="fast")
    ap.add_argument("--only", choices=["numeric", "tsp"], default=None)
    ap.add_argument("--no-animations", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    make_anim = not args.no_animations
    t0 = time.perf_counter()

    if args.only in (None, "numeric"):
        run_numeric(args.mode, make_anim)
    if args.only in (None, "tsp"):
        run_tsp_part(args.mode, make_anim)

    logger.info("Pipeline finalizado en %.1f s. Gifs en %s", time.perf_counter() - t0, GIFS_DIR)


if __name__ == "__main__":
    main()
