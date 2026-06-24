"""Manejo reproducible de semillas."""
from __future__ import annotations

import numpy as np


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Crea un generador de NumPy con semilla controlada.

    Parameters
    ----------
    seed : int | None
        Semilla. Si es ``None`` el generador es no determinista.

    Returns
    -------
    numpy.random.Generator
    """
    return np.random.default_rng(seed)


def derive_seed(base_seed: int, run_index: int) -> int:
    """Deriva una semilla determinista por corrida a partir de una base.

    Garantiza independencia entre corridas y reproducibilidad global.
    """
    return int((np.uint64(base_seed) * np.uint64(1_000_003) + np.uint64(run_index + 1))
               % np.uint64(2**32 - 1))
