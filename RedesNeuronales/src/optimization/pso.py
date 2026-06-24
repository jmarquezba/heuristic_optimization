"""Optimización por enjambre de partículas (PSO) desde cero (NumPy).

Convención de hiperparámetros alineada con PySwarms / material de clase
(``IRNA202402_PSO.ipynb``): inercia ``w``, coeficiente cognitivo ``c1``, social
``c2``. Incluye límites de posición y velocidad y conteo de evaluaciones.
"""
from __future__ import annotations

import time

import numpy as np

from .functions import CountingFunction, FunctionSpec
from .metrics import OptResult
from ..utils.seed import make_rng


def particle_swarm(
    spec: FunctionSpec,
    seed: int,
    n_particles: int = 40,
    max_iter: int = 200,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    vmax_frac: float = 0.2,
    record_trajectory: bool = False,
) -> OptResult:
    """Minimiza ``spec`` con PSO (GlobalBest).

    ``vmax_frac`` limita la velocidad a una fracción del ancho del dominio.
    """
    rng = make_rng(seed)
    f = CountingFunction(spec)
    lb, ub = spec.lower, spec.upper
    span = ub - lb
    d = spec.dim
    vmax = vmax_frac * span

    pos = rng.uniform(lb, ub, size=(n_particles, d))
    vel = rng.uniform(-vmax, vmax, size=(n_particles, d))
    fit = f(pos)

    pbest = pos.copy()
    pbest_fit = fit.copy()
    g_idx = int(np.argmin(pbest_fit))
    gbest = pbest[g_idx].copy()
    gbest_fit = float(pbest_fit[g_idx])

    history: list[float] = [gbest_fit]
    trajectory: list[np.ndarray] = [pos.copy()] if record_trajectory else []
    t0 = time.perf_counter()

    for _ in range(max_iter):
        r1, r2 = rng.random((n_particles, d)), rng.random((n_particles, d))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        vel = np.clip(vel, -vmax, vmax)
        pos = np.clip(pos + vel, lb, ub)
        fit = f(pos)

        improved = fit < pbest_fit
        pbest[improved] = pos[improved]
        pbest_fit[improved] = fit[improved]
        g_idx = int(np.argmin(pbest_fit))
        if pbest_fit[g_idx] < gbest_fit:
            gbest = pbest[g_idx].copy()
            gbest_fit = float(pbest_fit[g_idx])

        history.append(gbest_fit)
        if record_trajectory:
            trajectory.append(pos.copy())

    runtime = time.perf_counter() - t0
    return OptResult(
        algorithm="pso",
        function=spec.name,
        dim=spec.dim,
        x_best=gbest,
        f_best=gbest_fit,
        n_evals=f.n_evals,
        n_iter=max_iter,
        seed=seed,
        runtime_s=runtime,
        converged=True,
        history=history,
        trajectory=trajectory,
    )
