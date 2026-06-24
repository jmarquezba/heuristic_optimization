"""Mapas estáticos y animación de rutas TSP sobre Francia continental.

Si ``geopandas``/``contextily`` no están disponibles, se usa una dispersión de
lat/lon (proyección equirectangular simple) con marco aproximado de Francia.
La visualización principal es sobre Francia (decisión D5, SPEC §6).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from ..utils.config import FIGURES_DIR, GIFS_DIR


def _setup_ax(ax, df: pd.DataFrame, title: str) -> None:
    ax.scatter(df["lon"], df["lat"], c="#222", s=14, zorder=3)
    ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud")
    ax.set_title(title)
    ax.set_aspect(1.0 / np.cos(np.radians(df["lat"].mean())))
    ax.grid(alpha=0.3)


def _route_xy(df: pd.DataFrame, route: list[int]) -> tuple[np.ndarray, np.ndarray]:
    idx = route + [route[0]]
    return df["lon"].to_numpy()[idx], df["lat"].to_numpy()[idx]


def plot_route(df: pd.DataFrame, route: list[int], title: str,
               color: str = "#d33", out_path: Path | None = None) -> Path:
    """Mapa estático de una ruta cerrada."""
    out_path = Path(out_path or FIGURES_DIR / "tsp_route.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    _setup_ax(ax, df, title)
    x, y = _route_xy(df, route)
    ax.plot(x, y, "-", color=color, lw=1.2, zorder=2)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def plot_route_comparison(df: pd.DataFrame, route_aco: list[int], route_ga: list[int],
                          out_path: Path | None = None) -> Path:
    """Comparación lado a lado de las rutas ACO y GA."""
    out_path = Path(out_path or FIGURES_DIR / "tsp_routes_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, route, name, col in ((axes[0], route_aco, "ACO", "#d33"),
                                 (axes[1], route_ga, "GA", "#36b")):
        _setup_ax(ax, df, f"Mejor ruta {name}")
        x, y = _route_xy(df, route)
        ax.plot(x, y, "-", color=col, lw=1.2, zorder=2)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def animate_route_evolution(df: pd.DataFrame, routes_per_iter: list[list[int]],
                            history: list[float], algorithm: str = "aco",
                            fps: int = 10, max_frames: int = 120,
                            out_path: Path | None = None) -> Path:
    """Anima la evolución de la mejor ruta por iteración/generación."""
    out_path = Path(out_path or GIFS_DIR / f"tsp_france_best_route_{algorithm}.gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    step = max(1, len(routes_per_iter) // max_frames)
    routes = routes_per_iter[::step]
    hist = history[::step]
    col = "#d33" if algorithm == "aco" else "#36b"

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6.2))
    _setup_ax(ax, df, f"Evolución mejor ruta · {algorithm.upper()} (Francia)")
    (line,) = ax.plot([], [], "-", color=col, lw=1.2, zorder=2)
    ax2.set_title("Mejor costo por iteración")
    ax2.set_xlabel("iteración"); ax2.set_ylabel("costo (€)"); ax2.grid(alpha=0.3)
    (curve,) = ax2.plot([], [], "-", color=col)

    def update(i):
        x, y = _route_xy(df, routes[i])
        line.set_data(x, y)
        curve.set_data(np.arange(i + 1), hist[: i + 1])
        ax2.relim(); ax2.autoscale_view()
        return line, curve

    anim = FuncAnimation(fig, update, frames=len(routes), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
