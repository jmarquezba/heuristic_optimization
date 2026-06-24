"""Descenso por gradiente desde cero (NumPy).

Características (SPEC §8):
- condición inicial aleatoria dentro del dominio (semilla controlada),
- tasa de aprendizaje y máximo de iteraciones configurables,
- criterio de parada por norma del gradiente y por cambio pequeño en f,
- *clipping*/proyección al dominio (caja),
- conteo estricto de evaluaciones (via ``CountingFunction``),
- registro de trayectoria y de convergencia.
"""
from __future__ import annotations

import time

import numpy as np

from .functions import CountingFunction, FunctionSpec
from .gradients import central_difference_gradient
from .metrics import OptResult
from ..utils.seed import make_rng


def random_point(spec: FunctionSpec, rng: np.random.Generator) -> np.ndarray:
    """Punto aleatorio uniforme dentro del dominio de la función."""
    return rng.uniform(spec.lower, spec.upper)


def clip_to_domain(x: np.ndarray, spec: FunctionSpec) -> np.ndarray:
    """Proyecta x a la caja del dominio."""
    return np.clip(x, spec.lower, spec.upper)


def gradient_descent(
    spec: FunctionSpec,
    seed: int,
    lr: float = 1e-3,
    max_iter: int = 2000,
    grad_tol: float = 1e-6,
    f_tol: float = 1e-10,
    record_trajectory: bool = False,
) -> OptResult:
    """Optimiza ``spec`` con descenso por gradiente.

    Parameters
    ----------
    spec : FunctionSpec
        Función a optimizar (define dominio y dimensión).
    seed : int
        Semilla para la condición inicial.
    lr : float
        Tasa de aprendizaje (paso).
    max_iter : int
        Máximo de iteraciones.
    grad_tol : float
        Parada si ``||grad|| < grad_tol``.
    f_tol : float
        Parada si ``|f_{k} - f_{k-1}| < f_tol``.
    record_trajectory : bool
        Si guarda la trayectoria (para animaciones).
    """
    rng = make_rng(seed)
    f = CountingFunction(spec)
    x = random_point(spec, rng)
    fx = float(f(x))

    history = [fx]
    trajectory = [x.copy()] if record_trajectory else []
    converged = False
    t0 = time.perf_counter()

    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        grad = central_difference_gradient(f, x)
        gnorm = float(np.linalg.norm(grad))
        if gnorm < grad_tol:
            converged = True
            break
        x = clip_to_domain(x - lr * grad, spec)
        fx_new = float(f(x))
        history.append(fx_new)
        if record_trajectory:
            trajectory.append(x.copy())
        if abs(fx_new - fx) < f_tol:
            converged = True
            fx = fx_new
            break
        fx = fx_new

    runtime = time.perf_counter() - t0
    return OptResult(
        algorithm="gradient_descent",
        function=spec.name,
        dim=spec.dim,
        x_best=x,
        f_best=fx,
        n_evals=f.n_evals,
        n_iter=n_iter,
        seed=seed,
        runtime_s=runtime,
        converged=converged,
        history=history,
        trajectory=trajectory,
    )
