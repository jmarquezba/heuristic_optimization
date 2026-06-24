"""Utilidades de entrada/salida (CSV)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def append_rows(rows: list[dict], path: str | Path) -> None:
    """Añade filas a un CSV, creándolo con encabezado si no existe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def save_df(df: pd.DataFrame, path: str | Path) -> None:
    """Guarda un DataFrame como CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
