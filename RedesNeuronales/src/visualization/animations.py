"""Animaciones del proceso de optimización (Parte 1, numeral 3).

- GD: curvas de nivel + trayectoria del punto + valor de f por iteración.
- Heurístico (PSO/EA): población moviéndose + mejor punto + valor global.

Se generan GIFs con ``matplotlib.animation.PillowWriter`` (no requiere
imagemagick). Si el GIF es muy pesado se submuestrean los frames.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from ..optimization.functions import FunctionSpec
from ..utils.config import GIFS_DIR


def _contour_grid(spec: FunctionSpec, n: int = 200):
    lb, ub = spec.lower, spec.upper
    xs = np.linspace(lb[0], ub[0], n)
    ys = np.linspace(lb[1], ub[1], n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    if spec.dim == 3:  # fijar x3=0 para visualizar el corte 2D
        pts = np.column_stack([pts, np.zeros(len(pts))])
    Z = spec.func(pts).reshape(X.shape)
    return X, Y, Z


def animate_gradient_descent(spec: FunctionSpec, trajectory: list[np.ndarray],
                             history: list[float], fps: int = 12,
                             max_frames: int = 120,
                             out_path: Path | None = None) -> Path:
    """Anima la trayectoria del GD sobre las curvas de nivel."""
    out_path = Path(out_path or GIFS_DIR / f"gd_{spec.name}_{spec.dim}d.gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traj = np.array(trajectory)
    step = max(1, len(traj) // max_frames)
    traj, hist = traj[::step], history[::step]

    X, Y, Z = _contour_grid(spec)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax.contour(X, Y, Z, levels=30, cmap="viridis")
    ax.set_title(f"Descenso por gradiente · {spec.name} {spec.dim}D")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    (line,) = ax.plot([], [], "-r", lw=1.5)
    (pt,) = ax.plot([], [], "ro", ms=7)
    ax2.set_title("Valor de f por iteración")
    ax2.set_xlabel("iteración"); ax2.set_ylabel("f")
    ax2.set_yscale("symlog"); ax2.grid(alpha=0.3)
    (curve,) = ax2.plot([], [], "-b")

    def update(i):
        line.set_data(traj[: i + 1, 0], traj[: i + 1, 1])
        pt.set_data([traj[i, 0]], [traj[i, 1]])
        curve.set_data(np.arange(i + 1), hist[: i + 1])
        ax2.relim(); ax2.autoscale_view()
        return line, pt, curve

    anim = FuncAnimation(fig, update, frames=len(traj), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def animate_population(spec: FunctionSpec, pop_history: list[np.ndarray],
                       best_history: list[float], algorithm: str = "pso",
                       fps: int = 12, max_frames: int = 120,
                       out_path: Path | None = None) -> Path:
    """Anima una población/enjambre sobre las curvas de nivel."""
    out_path = Path(out_path or GIFS_DIR / f"{algorithm}_{spec.name}_{spec.dim}d.gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    step = max(1, len(pop_history) // max_frames)
    pops = pop_history[::step]
    hist = best_history[::step]

    X, Y, Z = _contour_grid(spec)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax.contour(X, Y, Z, levels=30, cmap="viridis")
    ax.set_title(f"{algorithm.upper()} · {spec.name} {spec.dim}D")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    scat = ax.scatter([], [], c="red", s=20)
    ax2.set_title("Mejor valor global por iteración")
    ax2.set_xlabel("iteración"); ax2.set_ylabel("f")
    ax2.set_yscale("symlog"); ax2.grid(alpha=0.3)
    (curve,) = ax2.plot([], [], "-b")

    def update(i):
        p = pops[i]
        scat.set_offsets(p[:, :2])
        curve.set_data(np.arange(i + 1), hist[: i + 1])
        ax2.relim(); ax2.autoscale_view()
        return scat, curve

    anim = FuncAnimation(fig, update, frames=len(pops), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
