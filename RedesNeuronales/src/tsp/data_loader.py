"""Carga y validación del dataset de las 96 capitales departamentales de la
Francia continental (metropolitana, sin territorios de ultramar).

Francia metropolitana tiene 96 departamentos: los numerados 01–95 (sin el 20,
que se divide en 2A Corse-du-Sud y 2B Haute-Corse). La capital usada es la
prefectura de cada departamento.

Fuente de coordenadas: datos geográficos públicos de las prefecturas
(Wikipedia / OpenStreetMap / IGN). Precisión ~0.01–0.05° (supuesto S1, SPEC §4),
suficiente para distancias inter-ciudad. Las coordenadas se conservan como
constante embebida (fuente única) y se materializan en CSV con ``build_dataset``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.config import PROCESSED_DIR, RAW_DIR
from ..utils.logging import get_logger

logger = get_logger("tsp.data_loader")

# (codigo, departamento, prefectura, lat, lon)
DEPARTMENTS: list[tuple[str, str, str, float, float]] = [
    ("01", "Ain", "Bourg-en-Bresse", 46.205, 5.225),
    ("02", "Aisne", "Laon", 49.564, 3.624),
    ("03", "Allier", "Moulins", 46.567, 3.333),
    ("04", "Alpes-de-Haute-Provence", "Digne-les-Bains", 44.092, 6.236),
    ("05", "Hautes-Alpes", "Gap", 44.559, 6.079),
    ("06", "Alpes-Maritimes", "Nice", 43.700, 7.268),
    ("07", "Ardèche", "Privas", 44.735, 4.599),
    ("08", "Ardennes", "Charleville-Mézières", 49.771, 4.716),
    ("09", "Ariège", "Foix", 42.964, 1.605),
    ("10", "Aube", "Troyes", 48.297, 4.074),
    ("11", "Aude", "Carcassonne", 43.213, 2.349),
    ("12", "Aveyron", "Rodez", 44.349, 2.575),
    ("13", "Bouches-du-Rhône", "Marseille", 43.297, 5.370),
    ("14", "Calvados", "Caen", 49.183, -0.370),
    ("15", "Cantal", "Aurillac", 44.926, 2.444),
    ("16", "Charente", "Angoulême", 45.650, 0.160),
    ("17", "Charente-Maritime", "La Rochelle", 46.160, -1.151),
    ("18", "Cher", "Bourges", 47.081, 2.399),
    ("19", "Corrèze", "Tulle", 45.267, 1.768),
    ("2A", "Corse-du-Sud", "Ajaccio", 41.919, 8.738),
    ("2B", "Haute-Corse", "Bastia", 42.697, 9.450),
    ("21", "Côte-d'Or", "Dijon", 47.322, 5.041),
    ("22", "Côtes-d'Armor", "Saint-Brieuc", 48.514, -2.765),
    ("23", "Creuse", "Guéret", 46.171, 1.870),
    ("24", "Dordogne", "Périgueux", 45.184, 0.721),
    ("25", "Doubs", "Besançon", 47.238, 6.024),
    ("26", "Drôme", "Valence", 44.933, 4.892),
    ("27", "Eure", "Évreux", 49.027, 1.151),
    ("28", "Eure-et-Loir", "Chartres", 48.444, 1.489),
    ("29", "Finistère", "Quimper", 47.996, -4.097),
    ("30", "Gard", "Nîmes", 43.837, 4.360),
    ("31", "Haute-Garonne", "Toulouse", 43.604, 1.444),
    ("32", "Gers", "Auch", 43.646, 0.586),
    ("33", "Gironde", "Bordeaux", 44.838, -0.579),
    ("34", "Hérault", "Montpellier", 43.611, 3.877),
    ("35", "Ille-et-Vilaine", "Rennes", 48.117, -1.677),
    ("36", "Indre", "Châteauroux", 46.811, 1.690),
    ("37", "Indre-et-Loire", "Tours", 47.394, 0.689),
    ("38", "Isère", "Grenoble", 45.188, 5.724),
    ("39", "Jura", "Lons-le-Saunier", 46.674, 5.554),
    ("40", "Landes", "Mont-de-Marsan", 43.890, -0.500),
    ("41", "Loir-et-Cher", "Blois", 47.586, 1.336),
    ("42", "Loire", "Saint-Étienne", 45.439, 4.387),
    ("43", "Haute-Loire", "Le Puy-en-Velay", 45.043, 3.885),
    ("44", "Loire-Atlantique", "Nantes", 47.218, -1.554),
    ("45", "Loiret", "Orléans", 47.902, 1.909),
    ("46", "Lot", "Cahors", 44.448, 1.441),
    ("47", "Lot-et-Garonne", "Agen", 44.203, 0.616),
    ("48", "Lozère", "Mende", 44.518, 3.500),
    ("49", "Maine-et-Loire", "Angers", 47.471, -0.551),
    ("50", "Manche", "Saint-Lô", 49.116, -1.090),
    ("51", "Marne", "Châlons-en-Champagne", 48.957, 4.365),
    ("52", "Haute-Marne", "Chaumont", 48.111, 5.139),
    ("53", "Mayenne", "Laval", 48.070, -0.770),
    ("54", "Meurthe-et-Moselle", "Nancy", 48.692, 6.184),
    ("55", "Meuse", "Bar-le-Duc", 48.771, 5.160),
    ("56", "Morbihan", "Vannes", 47.658, -2.760),
    ("57", "Moselle", "Metz", 49.119, 6.176),
    ("58", "Nièvre", "Nevers", 46.990, 3.163),
    ("59", "Nord", "Lille", 50.629, 3.057),
    ("60", "Oise", "Beauvais", 49.430, 2.081),
    ("61", "Orne", "Alençon", 48.430, 0.093),
    ("62", "Pas-de-Calais", "Arras", 50.291, 2.778),
    ("63", "Puy-de-Dôme", "Clermont-Ferrand", 45.777, 3.087),
    ("64", "Pyrénées-Atlantiques", "Pau", 43.295, -0.370),
    ("65", "Hautes-Pyrénées", "Tarbes", 43.233, 0.078),
    ("66", "Pyrénées-Orientales", "Perpignan", 42.698, 2.896),
    ("67", "Bas-Rhin", "Strasbourg", 48.573, 7.752),
    ("68", "Haut-Rhin", "Colmar", 48.079, 7.358),
    ("69", "Rhône", "Lyon", 45.764, 4.835),
    ("70", "Haute-Saône", "Vesoul", 47.622, 6.154),
    ("71", "Saône-et-Loire", "Mâcon", 46.307, 4.828),
    ("72", "Sarthe", "Le Mans", 48.007, 0.200),
    ("73", "Savoie", "Chambéry", 45.564, 5.918),
    ("74", "Haute-Savoie", "Annecy", 45.899, 6.129),
    ("75", "Paris", "Paris", 48.857, 2.351),
    ("76", "Seine-Maritime", "Rouen", 49.443, 1.099),
    ("77", "Seine-et-Marne", "Melun", 48.540, 2.660),
    ("78", "Yvelines", "Versailles", 48.804, 2.130),
    ("79", "Deux-Sèvres", "Niort", 46.324, -0.464),
    ("80", "Somme", "Amiens", 49.895, 2.302),
    ("81", "Tarn", "Albi", 43.928, 2.148),
    ("82", "Tarn-et-Garonne", "Montauban", 44.018, 1.355),
    ("83", "Var", "Toulon", 43.124, 5.928),
    ("84", "Vaucluse", "Avignon", 43.949, 4.806),
    ("85", "Vendée", "La Roche-sur-Yon", 46.670, -1.427),
    ("86", "Vienne", "Poitiers", 46.580, 0.340),
    ("87", "Haute-Vienne", "Limoges", 45.833, 1.262),
    ("88", "Vosges", "Épinal", 48.174, 6.449),
    ("89", "Yonne", "Auxerre", 47.798, 3.573),
    ("90", "Territoire de Belfort", "Belfort", 47.638, 6.864),
    ("91", "Essonne", "Évry", 48.629, 2.441),
    ("92", "Hauts-de-Seine", "Nanterre", 48.892, 2.206),
    ("93", "Seine-Saint-Denis", "Bobigny", 48.906, 2.451),
    ("94", "Val-de-Marne", "Créteil", 48.790, 2.455),
    ("95", "Val-d'Oise", "Cergy", 49.036, 2.060),
]

COLUMNS = ["codigo", "departamento", "prefectura", "lat", "lon"]
SOURCE = ("Datos geográficos públicos de prefecturas (Wikipedia/OpenStreetMap/IGN). "
          "Precisión ~0.01-0.05 grados (supuesto S1, SPEC).")


def build_dataset(write: bool = True) -> pd.DataFrame:
    """Construye el DataFrame de capitales y opcionalmente escribe los CSV."""
    df = pd.DataFrame(DEPARTMENTS, columns=COLUMNS)
    df["fuente"] = SOURCE
    df["notas_limpieza"] = (
        "Corcega dividida en 2A/2B (no existe 20). Coordenadas de la prefectura "
        "(centro urbano). Sin territorios de ultramar."
    )
    validate(df)
    if write:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DIR / "france_departments_sources.csv", index=False)
        df[COLUMNS].to_csv(PROCESSED_DIR / "france_96_capitals.csv", index=False)
        logger.info("Dataset Francia escrito (%d ciudades).", len(df))
    return df


def validate(df: pd.DataFrame) -> None:
    """Valida el dataset (96 ciudades, sin duplicados, coords en Francia)."""
    assert len(df) == 96, f"Se esperan 96 ciudades, hay {len(df)}"
    assert df["codigo"].nunique() == 96, "Códigos de departamento duplicados"
    assert df["prefectura"].nunique() == 96, "Prefecturas duplicadas"
    assert df["lat"].between(41.0, 51.5).all(), "Latitud fuera de Francia continental"
    assert df["lon"].between(-5.5, 10.0).all(), "Longitud fuera de Francia continental"


def load_capitals() -> pd.DataFrame:
    """Carga el CSV procesado; lo construye si no existe."""
    path = PROCESSED_DIR / "france_96_capitals.csv"
    if not path.exists():
        return build_dataset(write=True)[COLUMNS]
    df = pd.read_csv(path, dtype={"codigo": str})
    validate(df)
    return df


if __name__ == "__main__":
    build_dataset(write=True)
