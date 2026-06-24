"""Pruebas de los algoritmos de optimización numérica."""
import numpy as np

from src.optimization.differential_evolution import differential_evolution
from src.optimization.evolutionary import evolutionary_algorithm
from src.optimization.functions import get_function
from src.optimization.gradient_descent import gradient_descent
from src.optimization.pso import particle_swarm


def test_gd_respects_domain_and_counts():
    spec = get_function("rastrigin", 2)
    res = gradient_descent(spec, seed=1, lr=1e-3, max_iter=100)
    assert np.all(res.x_best >= spec.lower - 1e-9)
    assert np.all(res.x_best <= spec.upper + 1e-9)
    assert res.n_evals > 0
    assert np.isfinite(res.f_best)


def test_gd_finds_rosenbrock_basin():
    # Desde varias semillas, al menos una corrida debe reducir mucho f.
    spec = get_function("rosenbrock", 2)
    best = min(gradient_descent(spec, seed=s, lr=1e-3, max_iter=2000).f_best
               for s in range(5))
    assert best < 10.0


def test_pso_converges_rastrigin():
    spec = get_function("rastrigin", 2)
    res = particle_swarm(spec, seed=3, n_particles=40, max_iter=200)
    assert res.f_best < 5.0
    assert np.all(res.x_best >= spec.lower) and np.all(res.x_best <= spec.upper)


def test_de_converges_sphere_like():
    spec = get_function("griewank", 2)
    res = differential_evolution(spec, seed=2, pop_size=40, max_gen=200)
    assert res.f_best < 0.5


def test_ea_runs_and_bounds():
    spec = get_function("six_hump_camel", 2)
    res = evolutionary_algorithm(spec, seed=7, pop_size=50, max_gen=150)
    assert res.f_best < 0.0   # el mínimo es -1.0316
    assert np.all(res.x_best >= spec.lower) and np.all(res.x_best <= spec.upper)


def test_history_monotone_best():
    spec = get_function("rastrigin", 2)
    res = particle_swarm(spec, seed=1, n_particles=30, max_iter=50)
    h = np.array(res.history)
    assert np.all(np.diff(h) <= 1e-9)   # mejor global no empeora
