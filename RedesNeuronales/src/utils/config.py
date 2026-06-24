"""Carga de configuración YAML y resolución de rutas del proyecto."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Raíz del proyecto = dos niveles por encima de este archivo (src/utils/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ASSETS_DIR = PROJECT_ROOT / "assets"
FIGURES_DIR = ASSETS_DIR / "figures"
GIFS_DIR = ASSETS_DIR / "gifs"
TABLES_DIR = ASSETS_DIR / "tables"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_config(path: str | Path) -> dict[str, Any]:
    """Carga un archivo YAML como diccionario."""
    path = Path(path)
    if not path.is_absolute():
        path = CONFIG_DIR / path
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dirs() -> None:
    """Crea las carpetas de salida si no existen."""
    for d in (RESULTS_DIR, RAW_DIR, PROCESSED_DIR, FIGURES_DIR, GIFS_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)
