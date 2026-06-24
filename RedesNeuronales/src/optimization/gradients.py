"""Gradientes por diferencias finitas centrales y analíticos validados.

Decisión D3 (SPEC §6): el descenso por gradiente usa diferencias finitas
centrales para tratar las seis funciones de forma uniforme. Cada evaluación
adicional de la función objetivo se cuenta en el NFE porque se invoca la
``CountingFunction``. Se proveen además gradientes analíticos para Rosenbrock y
Rastrigin como verificación cruzada (tests).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def central_difference_gradient(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    h_rel: float = 1e-5,
) -> np.ndarray:
    """Gradiente por diferencias finitas centrales.

    Usa paso adaptativo ``h = h_rel * (1 + |x_i|)``. Cuesta ``2d`` evaluaciones
    de ``f`` (que deben ser una ``CountingFunction`` para contar el NFE).
    """
    x = np.asarray(x, dtype=float)
    d = x.size
    grad = np.zeros(d)
    for i in range(d):
        h = h_rel * (1.0 + abs(x[i]))
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        grad[i] = (float(f(xp)) - float(f(xm))) / (2.0 * h)
    return grad


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Gradiente analítico de Rosenbrock (verificación cruzada)."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    grad[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
    grad[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
    return grad


def rastrigin_grad(x: np.ndarray) -> np.ndarray:
    """Gradiente analítico de Rastrigin (verificación cruzada)."""
    x = np.asarray(x, dtype=float)
    return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)
