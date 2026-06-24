"""Algoritmo genético para el TSP (Parte 2).

Representación por permutaciones de las 96 ciudades. Inicialización aleatoria,
selección por torneo, cruce de orden (OX), mutación por inversión y elitismo.
Cada ruta es válida (todas las ciudades una vez) y el ciclo se cierra al evaluar.
Registra la mejor ruta por generación.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..utils.seed import make_rng


@dataclass
class GAResult:
    best_route: list[int]
    best_cost: float
    history: list[float]
    best_route_per_gen: list[list[int]]
    n_gen: int
    seed: int
    runtime_s: float
    best_gen: int


def _route_cost(route: np.ndarray, cost: np.ndarray) -> float:
    idx = np.append(route, route[0])
    return float(cost[idx[:-1], idx[1:]].sum())


def _ox(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Order Crossover (OX) para permutaciones."""
    n = p1.size
    a, b = sorted(rng.choice(n, size=2, replace=False))
    child = -np.ones(n, dtype=int)
    child[a:b + 1] = p1[a:b + 1]
    fill = [g for g in p2 if g not in set(p1[a:b + 1])]
    pos = [i for i in range(n) if child[i] == -1]
    for i, g in zip(pos, fill):
        child[i] = g
    return child


def _inversion(route: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    a, b = sorted(rng.choice(route.size, size=2, replace=False))
    r = route.copy()
    r[a:b + 1] = r[a:b + 1][::-1]
    return r


def _tournament(fit: np.ndarray, k: int, rng: np.random.Generator) -> int:
    cand = rng.integers(0, fit.size, size=k)
    return int(cand[np.argmin(fit[cand])])


def genetic_algorithm_tsp(
    cost: np.ndarray,
    seed: int,
    pop_size: int = 200,
    n_gen: int = 400,
    elite_frac: float = 0.05,
    mutation_rate: float = 0.2,
    tournament_k: int = 5,
) -> GAResult:
    """Resuelve el TSP con un algoritmo genético (permutaciones + OX)."""
    rng = make_rng(seed)
    n = cost.shape[0]
    pop = np.array([rng.permutation(n) for _ in range(pop_size)])
    fit = np.array([_route_cost(ind, cost) for ind in pop])
    n_elite = max(1, int(round(elite_frac * pop_size)))

    history: list[float] = []
    best_per_gen: list[list[int]] = []
    best_route = pop[int(np.argmin(fit))].copy()
    best_cost = float(np.min(fit))
    best_gen = 0
    t0 = time.perf_counter()

    for g in range(n_gen):
        order = np.argsort(fit)
        pop, fit = pop[order], fit[order]
        if fit[0] < best_cost:
            best_cost, best_route, best_gen = float(fit[0]), pop[0].copy(), g
        history.append(best_cost)
        best_per_gen.append(best_route.tolist())

        new_pop = [pop[i].copy() for i in range(n_elite)]
        while len(new_pop) < pop_size:
            i = _tournament(fit, tournament_k, rng)
            j = _tournament(fit, tournament_k, rng)
            child = _ox(pop[i], pop[j], rng)
            if rng.random() < mutation_rate:
                child = _inversion(child, rng)
            new_pop.append(child)
        pop = np.array(new_pop)
        fit = np.array([_route_cost(ind, cost) for ind in pop])

    order = np.argsort(fit)
    if fit[order[0]] < best_cost:
        best_cost, best_route, best_gen = float(fit[order[0]]), pop[order[0]].copy(), n_gen
    runtime = time.perf_counter() - t0
    return GAResult(best_route.tolist(), best_cost, history, best_per_gen,
                    n_gen, seed, runtime, best_gen)
