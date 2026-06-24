"""Funciones de prueba para optimización numérica.

Implementa seis funciones canónicas (Rosenbrock, Rastrigin, Schwefel, Griewank,
Goldstein-Price y *six-hump camel*) con:

- dominio estándar por dimensión,
- óptimo global y valor mínimo conocidos,
- implementación vectorizada en NumPy,
- contador estricto de evaluaciones de la función objetivo,
- validación de entrada,
- *docstring* con fórmula, dominio y fuente.

Goldstein-Price y *six-hump camel* son funciones canónicas de 2 variables. Para
soportar 3D se usa la extensión documentada (SPEC §7, decisión D2):

    f_ext(x1, x2, x3) = f_2D(x1, x2) + x3**2

que conserva la función original en x3 = 0 y mantiene el óptimo principal.

Fuentes:
- Rosenbrock: https://en.wikipedia.org/wiki/Rosenbrock_function
- Rastrigin:  https://en.wikipedia.org/wiki/Rastrigin_function
- Goldstein-Price: https://www.sfu.ca/~ssurjano/goldpr.html
- Six-hump camel:  https://www.sfu.ca/~ssurjano/camel6.html
- Test functions: https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

ArrayLike = np.ndarray


# --------------------------------------------------------------------------- #
# Implementaciones puras (sin conteo). Aceptan x de forma (d,) o (n, d).
# --------------------------------------------------------------------------- #
def _as_2d(x: ArrayLike) -> tuple[np.ndarray, bool]:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(1, -1), True
    return x, False


def rosenbrock(x: ArrayLike) -> np.ndarray:
    r"""Función de Rosenbrock (valle curvo, unimodal en d=2).

    f(x) = sum_{i=1}^{d-1} [100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2].
    Mínimo f=0 en x=(1,...,1). Dominio típico [-5, 10]^d.
    """
    x, scalar = _as_2d(x)
    val = np.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1.0 - x[:, :-1]) ** 2, axis=1)
    return val[0] if scalar else val


def rastrigin(x: ArrayLike) -> np.ndarray:
    r"""Función de Rastrigin (altamente multimodal).

    f(x) = 10 d + sum_{i=1}^d [x_i^2 - 10 cos(2 pi x_i)].
    Mínimo f=0 en x=0. Dominio [-5.12, 5.12]^d.
    """
    x, scalar = _as_2d(x)
    d = x.shape[1]
    val = 10.0 * d + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x), axis=1)
    return val[0] if scalar else val


def schwefel(x: ArrayLike) -> np.ndarray:
    r"""Función de Schwefel (multimodal, óptimo lejos del centro).

    f(x) = 418.9829 d - sum_{i=1}^d x_i sin(sqrt(|x_i|)).
    Mínimo f≈0 en x_i = 420.9687. Dominio [-500, 500]^d.
    """
    x, scalar = _as_2d(x)
    d = x.shape[1]
    val = 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)
    return val[0] if scalar else val


def griewank(x: ArrayLike) -> np.ndarray:
    r"""Función de Griewank (multimodal, muchos mínimos locales).

    f(x) = 1 + sum x_i^2/4000 - prod cos(x_i/sqrt(i)).
    Mínimo f=0 en x=0. Dominio [-600, 600]^d.
    """
    x, scalar = _as_2d(x)
    d = x.shape[1]
    i = np.arange(1, d + 1)
    s = np.sum(x ** 2, axis=1) / 4000.0
    p = np.prod(np.cos(x / np.sqrt(i)), axis=1)
    val = 1.0 + s - p
    return val[0] if scalar else val


def _goldstein_price_2d(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[:, 0], x[:, 1]
    a = 1 + (x1 + x2 + 1) ** 2 * (
        19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2)
    b = 30 + (2 * x1 - 3 * x2) ** 2 * (
        18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2)
    return a * b


def goldstein_price(x: ArrayLike) -> np.ndarray:
    r"""Función de Goldstein-Price (canónica 2D; extensión 3D documentada).

    En 2D: mínimo f=3 en (0, -1), dominio [-2, 2]^2.
    En 3D (D2): f_3D = f_2D(x1,x2) + x3^2, x3 en [-2, 2]; mismo óptimo en x3=0.
    Fuente: https://www.sfu.ca/~ssurjano/goldpr.html
    """
    x, scalar = _as_2d(x)
    if x.shape[1] not in (2, 3):
        raise ValueError("Goldstein-Price soporta d=2 o d=3 (extensión).")
    val = _goldstein_price_2d(x[:, :2])
    if x.shape[1] == 3:
        val = val + x[:, 2] ** 2
    return val[0] if scalar else val


def _six_hump_camel_2d(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[:, 0], x[:, 1]
    return ((4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2
            + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2)


def six_hump_camel(x: ArrayLike) -> np.ndarray:
    r"""Función *six-hump camel* (canónica 2D; extensión 3D documentada).

    f(x) = (4 - 2.1 x1^2 + x1^4/3) x1^2 + x1 x2 + (-4 + 4 x2^2) x2^2.
    En 2D: mínimo f≈-1.0316 en (0.0898, -0.7126) y (-0.0898, 0.7126).
    Dominio x1 en [-3,3], x2 en [-2,2].
    En 3D (D2): f_3D = f_2D + x3^2, x3 en [-3, 3]; mismo óptimo en x3=0.
    Fuente: https://www.sfu.ca/~ssurjano/camel6.html
    """
    x, scalar = _as_2d(x)
    if x.shape[1] not in (2, 3):
        raise ValueError("Six-hump camel soporta d=2 o d=3 (extensión).")
    val = _six_hump_camel_2d(x[:, :2])
    if x.shape[1] == 3:
        val = val + x[:, 2] ** 2
    return val[0] if scalar else val


# --------------------------------------------------------------------------- #
# Metadatos y wrapper con conteo de evaluaciones
# --------------------------------------------------------------------------- #
@dataclass
class FunctionSpec:
    """Especificación de una función de prueba para una dimensión dada."""
    name: str
    func: Callable[[ArrayLike], np.ndarray]
    dim: int
    bounds: np.ndarray            # forma (dim, 2): [low, high] por coordenada
    optimum: np.ndarray           # x* (uno de los óptimos)
    f_opt: float                  # f(x*)
    multimodal: bool

    @property
    def lower(self) -> np.ndarray:
        return self.bounds[:, 0]

    @property
    def upper(self) -> np.ndarray:
        return self.bounds[:, 1]


class CountingFunction:
    """Envuelve una función objetivo y cuenta evaluaciones (NFE).

    Una llamada con una matriz (n, d) cuenta como ``n`` evaluaciones; con un
    vector (d,) cuenta como 1.
    """

    def __init__(self, spec: FunctionSpec):
        self.spec = spec
        self.n_evals = 0

    def __call__(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = 1 if x.ndim == 1 else x.shape[0]
        out = self.spec.func(x)
        if not np.all(np.isfinite(out)):
            raise FloatingPointError(f"Valor no finito en {self.spec.name}")
        self.n_evals += n
        return out

    def reset(self) -> None:
        self.n_evals = 0


def _box(low: float, high: float, dim: int) -> np.ndarray:
    return np.tile([low, high], (dim, 1)).astype(float)


def _make_spec(name: str, dim: int) -> FunctionSpec:
    """Construye la especificación de ``name`` en dimensión ``dim``."""
    if name == "rosenbrock":
        return FunctionSpec(name, rosenbrock, dim, _box(-5, 10, dim),
                            np.ones(dim), 0.0, multimodal=False)
    if name == "rastrigin":
        return FunctionSpec(name, rastrigin, dim, _box(-5.12, 5.12, dim),
                            np.zeros(dim), 0.0, multimodal=True)
    if name == "schwefel":
        return FunctionSpec(name, schwefel, dim, _box(-500, 500, dim),
                            np.full(dim, 420.9687), 0.0, multimodal=True)
    if name == "griewank":
        return FunctionSpec(name, griewank, dim, _box(-600, 600, dim),
                            np.zeros(dim), 0.0, multimodal=True)
    if name == "goldstein_price":
        if dim == 2:
            b = np.array([[-2, 2], [-2, 2]], dtype=float)
            return FunctionSpec(name, goldstein_price, 2, b,
                                np.array([0.0, -1.0]), 3.0, multimodal=True)
        if dim == 3:
            b = np.array([[-2, 2], [-2, 2], [-2, 2]], dtype=float)
            return FunctionSpec(name, goldstein_price, 3, b,
                                np.array([0.0, -1.0, 0.0]), 3.0, multimodal=True)
        raise ValueError("goldstein_price solo d=2 o d=3.")
    if name == "six_hump_camel":
        if dim == 2:
            b = np.array([[-3, 3], [-2, 2]], dtype=float)
            return FunctionSpec(name, six_hump_camel, 2, b,
                                np.array([0.0898, -0.7126]), -1.0316284535, True)
        if dim == 3:
            b = np.array([[-3, 3], [-2, 2], [-3, 3]], dtype=float)
            return FunctionSpec(name, six_hump_camel, 3, b,
                                np.array([0.0898, -0.7126, 0.0]), -1.0316284535, True)
        raise ValueError("six_hump_camel solo d=2 o d=3.")
    raise KeyError(f"Función desconocida: {name}")


ALL_FUNCTIONS = (
    "rosenbrock", "rastrigin", "schwefel", "griewank",
    "goldstein_price", "six_hump_camel",
)


def get_function(name: str, dim: int) -> FunctionSpec:
    """Devuelve la especificación de ``name`` en dimensión ``dim``."""
    if name not in ALL_FUNCTIONS:
        raise KeyError(f"Función desconocida: {name}. Opciones: {ALL_FUNCTIONS}")
    return _make_spec(name, dim)
