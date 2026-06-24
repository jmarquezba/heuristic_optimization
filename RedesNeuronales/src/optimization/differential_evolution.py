"""Evolución diferencial (DE/rand/1/bin) desde cero (NumPy).

Estrategia clásica de Storn & Price: mutación con factor ``F``, cruce binomial
con tasa ``CR`` y selección por supervivencia (greedy). Conteo de evaluaciones
según SPEC §8.
"""
from __future__ import annotations

import time

import numpy as np

from .functions import CountingFunction, FunctionSpec
from .metrics import OptResult
from ..utils.seed import make_rng


def differential_evolution(
    spec: FunctionSpec,
    seed: int,
    pop_size: int = 40,
    max_gen: int = 200,
    F: float = 0.7,
    CR: float = 0.9,
    record_trajectory: bool = False,
) -> OptResult:
    """Minimiza ``spec`` con DE/rand/1/bin."""
    rng = make_rng(seed)
    f = CountingFunction(spec)
    lb, ub = spec.lower, spec.upper
    d = spec.dim

    pop = rng.uniform(lb, ub, size=(pop_size, d))
    fit = f(pop)
    history: list[float] = [float(np.min(fit))]
    trajectory: list[np.ndarray] = [pop.copy()] if record_trajectory else []
    t0 = time.perf_counter()

    idx_all = np.arange(pop_size)
    for _ in range(max_gen):
        for i in range(pop_size):
            choices = idx_all[idx_all != i]
            r1, r2, r3 = rng.choice(choices, size=3, replace=False)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            mutant = np.clip(mutant, lb, ub)
            cross = rng.random(d) < CR
            if not np.any(cross):
                cross[rng.integers(0, d)] = True
            trial = np.where(cross, mutant, pop[i])
            f_trial = float(f(trial))
            if f_trial <= fit[i]:
                pop[i] = trial
                fit[i] = f_trial
        history.append(float(np.min(fit)))
        if record_trajectory:
            trajectory.append(pop.copy())

    best = int(np.argmin(fit))
    runtime = time.perf_counter() - t0
    return OptResult(
        algorithm="differential_evolution",
        function=spec.name,
        dim=spec.dim,
        x_best=pop[best],
        f_best=float(fit[best]),
        n_evals=f.n_evals,
        n_iter=max_gen,
        seed=seed,
        runtime_s=runtime,
        converged=True,
        history=history,
        trajectory=trajectory,
    )
