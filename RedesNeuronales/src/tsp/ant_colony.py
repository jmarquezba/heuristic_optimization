"""Colonias de hormigas (ACO) para el TSP (Parte 2).

Convención de parámetros alineada con el material de clase
(``IRNABI202601 ant_colony.ipynb``, implementación de J. Berroa): ``alpha``,
``beta``, ``evaporation_rate``, ``intensification`` (Q), número de hormigas.

Componentes: feromonas, visibilidad η = 1/costo, evaporación, depósito de
feromona, varias hormigas por iteración, ciclo cerrado (retorno al origen) y
registro de la mejor ruta por iteración.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..utils.seed import make_rng


@dataclass
class ACOResult:
    best_route: list[int]
    best_cost: float
    history: list[float]               # mejor costo por iteración
    best_route_per_iter: list[list[int]]
    n_iter: int
    seed: int
    runtime_s: float
    best_iter: int


def _route_cost(route: np.ndarray, cost: np.ndarray) -> float:
    idx = np.append(route, route[0])
    return float(cost[idx[:-1], idx[1:]].sum())


def ant_colony_tsp(
    cost: np.ndarray,
    seed: int,
    n_ants: int = 30,
    n_iter: int = 200,
    alpha: float = 1.0,
    beta: float = 4.0,
    evaporation_rate: float = 0.1,
    intensification: float = 1.0,
    q0: float = 0.0,
) -> ACOResult:
    """Resuelve el TSP con ACO (Ant System con intensificación del mejor).

    Parameters
    ----------
    cost : (N, N) ndarray
        Matriz de costos simétrica con diagonal cero.
    alpha, beta : float
        Pesos de feromona y visibilidad.
    evaporation_rate : float
        Tasa de evaporación de feromona (rho).
    intensification : float
        Constante de depósito Q.
    q0 : float
        Probabilidad de explotación greedy (regla pseudo-aleatoria, 0 = AS clásico).
    """
    rng = make_rng(seed)
    n = cost.shape[0]
    eta = np.zeros_like(cost)
    nz = cost > 0
    eta[nz] = 1.0 / cost[nz]
    tau = np.ones((n, n))

    best_route: list[int] = list(range(n))
    best_cost = np.inf
    best_iter = 0
    history: list[float] = []
    best_route_per_iter: list[list[int]] = []
    t0 = time.perf_counter()

    for it in range(n_iter):
        iter_best_route, iter_best_cost = None, np.inf
        for _ in range(n_ants):
            start = int(rng.integers(0, n))
            unvisited = set(range(n))
            unvisited.remove(start)
            route = [start]
            current = start
            while unvisited:
                js = np.fromiter(unvisited, dtype=int)
                w = (tau[current, js] ** alpha) * (eta[current, js] ** beta)
                if not np.any(w > 0):
                    w = np.ones_like(w)
                if q0 > 0 and rng.random() < q0:
                    nxt = int(js[np.argmax(w)])
                else:
                    p = w / w.sum()
                    nxt = int(rng.choice(js, p=p))
                route.append(nxt)
                unvisited.remove(nxt)
                current = nxt
            c = _route_cost(np.array(route), cost)
            if c < iter_best_cost:
                iter_best_cost, iter_best_route = c, route

        # Evaporación
        tau *= (1.0 - evaporation_rate)
        # Depósito del mejor de la iteración (elitista)
        idx = np.array(iter_best_route + [iter_best_route[0]])
        deposit = intensification / iter_best_cost
        for a, b in zip(idx[:-1], idx[1:]):
            tau[a, b] += deposit
            tau[b, a] += deposit

        if iter_best_cost < best_cost:
            best_cost, best_route, best_iter = iter_best_cost, iter_best_route, it
        history.append(best_cost)
        best_route_per_iter.append(list(best_route))

    runtime = time.perf_counter() - t0
    return ACOResult(best_route, float(best_cost), history, best_route_per_iter,
                     n_iter, seed, runtime, best_iter)
