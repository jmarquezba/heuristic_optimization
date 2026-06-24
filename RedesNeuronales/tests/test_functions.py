"""Pruebas de las funciones objetivo en sus óptimos conocidos."""
import numpy as np
import pytest

from src.optimization.functions import ALL_FUNCTIONS, CountingFunction, get_function
from src.optimization.gradients import (
    central_difference_gradient, rastrigin_grad, rosenbrock_grad,
)


@pytest.mark.parametrize("name,dim", [(n, d) for n in ALL_FUNCTIONS for d in (2, 3)])
def test_optimum_value(name, dim):
    spec = get_function(name, dim)
    f_at_opt = float(spec.func(spec.optimum))
    assert abs(f_at_opt - spec.f_opt) < 1e-3, (name, dim, f_at_opt, spec.f_opt)


def test_rosenbrock_min_is_ones():
    spec = get_function("rosenbrock", 3)
    assert abs(float(spec.func(np.ones(3)))) < 1e-9


def test_goldstein_price_2d_value():
    spec = get_function("goldstein_price", 2)
    assert abs(float(spec.func(np.array([0.0, -1.0]))) - 3.0) < 1e-6


def test_six_hump_camel_two_optima():
    spec = get_function("six_hump_camel", 2)
    f1 = float(spec.func(np.array([0.0898, -0.7126])))
    f2 = float(spec.func(np.array([-0.0898, 0.7126])))
    assert abs(f1 - f2) < 1e-3
    assert abs(f1 - (-1.0316)) < 1e-3


def test_3d_extension_preserves_optimum():
    # f_3D(x1,x2,0) == f_2D(x1,x2)
    for name in ("goldstein_price", "six_hump_camel"):
        s2 = get_function(name, 2)
        s3 = get_function(name, 3)
        x2 = np.array([0.5, -0.3])
        x3 = np.array([0.5, -0.3, 0.0])
        assert abs(float(s2.func(x2)) - float(s3.func(x3))) < 1e-9


def test_counting_function_increments():
    spec = get_function("rastrigin", 2)
    f = CountingFunction(spec)
    f(np.zeros(2))
    f(np.ones((5, 2)))
    assert f.n_evals == 6
    f.reset()
    assert f.n_evals == 0


def test_no_nan_inf():
    spec = get_function("griewank", 3)
    vals = spec.func(np.random.default_rng(0).uniform(-600, 600, size=(100, 3)))
    assert np.all(np.isfinite(vals))


def test_finite_difference_matches_analytic():
    spec = get_function("rosenbrock", 3)
    f = CountingFunction(spec)
    x = np.array([0.3, -0.5, 1.2])
    g_fd = central_difference_gradient(f, x)
    assert np.allclose(g_fd, rosenbrock_grad(x), atol=1e-3)

    spec_r = get_function("rastrigin", 2)
    fr = CountingFunction(spec_r)
    xr = np.array([0.4, -0.2])
    assert np.allclose(central_difference_gradient(fr, xr), rastrigin_grad(xr), atol=1e-3)
