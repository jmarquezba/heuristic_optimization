"""Modelo de costos del TSP (Parte 2).

Costo total entre ciudad i y j (SPEC §4, fórmula base):

    costo_total_ij = tiempo_horas_ij * valor_hora
                   + distancia_km_ij * costo_combustible_km
                   + peaje_estimado_ij

Componentes:
- Distancia por carretera ≈ haversine × factor de detour (S2).
- Tiempo = distancia_carretera / velocidad media (S3).
- Combustible = distancia_km × (consumo_L_100km/100) × precio_L.
- Peaje = distancia_km × frac_autopista × tarifa_peaje_por_km (S4, estimación).

Todas las matrices son simétricas con diagonal cero (S6).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.config import PROCESSED_DIR

EARTH_RADIUS_KM = 6371.0088


@dataclass
class Vehicle:
    """Vehículo del recorrido (escenario base: Renault Clio V gasolina)."""
    marca: str = "Renault"
    modelo: str = "Clio V 1.0 TCe"
    combustible: str = "Gasolina (SP95-E10)"
    consumo_l_100km: float = 5.3        # WLTP mixto (orden de magnitud)
    precio_combustible_l: float = 1.75  # €/L (orden de magnitud 2024-2025, Francia)
    fuente: str = ("Ficha técnica Renault Clio (WLTP) y precios medios de "
                   "carburantes en Francia; valores como estimación documentada (S5).")

    @property
    def costo_combustible_km(self) -> float:
        """Costo de combustible por km (€/km)."""
        return (self.consumo_l_100km / 100.0) * self.precio_combustible_l


@dataclass
class CostParams:
    """Parámetros del modelo de costos."""
    road_factor: float = 1.30      # S2: detour haversine -> carretera
    avg_speed_kmh: float = 90.0    # S3: velocidad media efectiva
    frac_autopista: float = 0.70   # S4: fracción de autopista de peaje
    toll_eur_per_km: float = 0.092 # S4: tarifa de peaje por km de autopista


def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Matriz de distancias *haversine* (km) entre coordenadas."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_cost_matrices(df: pd.DataFrame, vehicle: Vehicle, params: CostParams,
                        valor_hora: float) -> dict[str, np.ndarray]:
    """Construye las matrices de distancia, tiempo, peaje y costo total.

    Parameters
    ----------
    df : DataFrame
        Capitales con columnas ``lat`` y ``lon``.
    vehicle : Vehicle
    params : CostParams
    valor_hora : float
        Valor de la hora del vendedor (€/h).

    Returns
    -------
    dict con claves: distance_km, time_h, toll, fuel, total.
    """
    lat = df["lat"].to_numpy(float)
    lon = df["lon"].to_numpy(float)
    hav = haversine_matrix(lat, lon)
    dist = hav * params.road_factor
    np.fill_diagonal(dist, 0.0)

    time_h = dist / params.avg_speed_kmh
    toll = dist * params.frac_autopista * params.toll_eur_per_km
    fuel = dist * vehicle.costo_combustible_km

    total = time_h * valor_hora + fuel + toll
    np.fill_diagonal(total, 0.0)

    return {"distance_km": dist, "time_h": time_h, "toll": toll,
            "fuel": fuel, "total": total}


def route_breakdown(route: list[int], matrices: dict[str, np.ndarray]) -> dict:
    """Descompone el costo de una ruta cerrada en sus componentes."""
    idx = np.array(route + [route[0]])
    a, b = idx[:-1], idx[1:]
    return {
        "distance_total_km": float(matrices["distance_km"][a, b].sum()),
        "time_total_h": float(matrices["time_h"][a, b].sum()),
        "toll_total": float(matrices["toll"][a, b].sum()),
        "fuel_total": float(matrices["fuel"][a, b].sum()),
        "cost_total": float(matrices["total"][a, b].sum()),
    }


def save_base_matrices(df: pd.DataFrame, matrices: dict[str, np.ndarray],
                       out_dir: Path | None = None) -> None:
    """Guarda las matrices base como CSV (etiquetadas por prefectura)."""
    out_dir = Path(out_dir or PROCESSED_DIR)
    labels = df["prefectura"].tolist()
    mapping = {"distance_km": "tsp_distance_matrix.csv",
               "time_h": "tsp_time_matrix.csv",
               "toll": "tsp_toll_matrix.csv",
               "total": "tsp_total_cost_matrix_base.csv"}
    for key, fname in mapping.items():
        pd.DataFrame(matrices[key], index=labels, columns=labels).to_csv(out_dir / fname)
