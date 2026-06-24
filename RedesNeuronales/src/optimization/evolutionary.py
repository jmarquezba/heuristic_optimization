"""Algoritmo evolutivo desde cero (NumPy).

Convenciones alineadas con el material de clase
(``IRNA202601_prep_Algoritmos_Evolutivos.ipynb``): elitismo y fracción de
mutación. Aquí se añade selección por torneo y cruce aritmético, con conteo
estricto de evaluaciones (SPEC §8).
"""
from __future__ import annotations

import time

import numpy as np

from .functions import CountingFunction, FunctionSpec
from .metrics import OptResult
from ..utils.seed import make_rng


def _tournament(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    cand = rng.integers(0, fitness.size, size=k)
    return int(cand[np.argmin(fitness[cand])])


def evolutionary_algorithm(
    spec: FunctionSpec,
    seed: int,
    pop_size: int = 50,
    max_gen: int = 200,
    elite_frac: float = 0.1,
    mutation_rate: float = 0.2,
    mutation_sigma: float = 0.1,
    tournament_k: int = 3,
    crossover: str = "arithmetic",
    record_trajectory: bool = False,
) -> OptResult:
    """Minimiza ``spec`` con un algoritmo evolutivo.

    Selección por torneo, cruce aritmético o uniforme, mutación gaussiana y
    elitismo. ``mutation_sigma`` es relativo al ancho del dominio.
    """
    rng = make_rng(seed)
    f = CountingFunction(spec)
    lb, ub = spec.lower, spec.upper
    span = ub - lb
    d = spec.dim

    pop = rng.uniform(lb, ub, size=(pop_size, d))
    fit = f(pop)
    n_elite = max(1, int(round(elite_frac * pop_size)))

    history: list[float] = []
    trajectory: list[np.ndarray] = []
    t0 = time.perf_counter()

    for _ in range(max_gen):
        order = np.argsort(fit)
        pop, fit = pop[order], fit[order]
        history.append(float(fit[0]))
        if record_trajectory:
            trajectory.append(pop.copy())

        new_pop = [pop[i].copy() for i in range(n_elite)]
        while len(new_pop) < pop_size:
            i = _tournament(fit, tournament_k, rng)
            j = _tournament(fit, tournament_k, rng)
            p1, p2 = pop[i], pop[j]
            if crossover == "uniform":
                mask = rng.random(d) < 0.5
                child = np.where(mask, p1, p2)
            else:  # aritmético
                a = rng.random()
                child = a * p1 + (1 - a) * p2
            # mutación gaussiana
            mut_mask = rng.random(d) < mutation_rate
            child = child + mut_mask * rng.normal(0.0, mutation_sigma * span, size=d)
            new_pop.append(np.clip(child, lb, ub))

        pop = np.array(new_pop)
        fit = f(pop)

    order = np.argsort(fit)
    pop, fit = pop[order], fit[order]
    history.append(float(fit[0]))
    runtime = time.perf_counter() - t0

    return OptResult(
        algorithm="evolutionary",
        function=spec.name,
        dim=spec.dim,
        x_best=pop[0],
        f_best=float(fit[0]),
        n_evals=f.n_evals,
        n_iter=max_gen,
        seed=seed,
        runtime_s=runtime,
        converged=True,
        history=history,
        trajectory=trajectory,
    )
