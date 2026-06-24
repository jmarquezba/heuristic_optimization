"""Pruebas del TSP de Francia (datos, costos y algoritmos)."""
import numpy as np

from src.tsp.ant_colony import ant_colony_tsp
from src.tsp.cost_model import CostParams, Vehicle, build_cost_matrices, route_breakdown
from src.tsp.data_loader import build_dataset, validate
from src.tsp.genetic_algorithm import genetic_algorithm_tsp


def _matrices():
    df = build_dataset(write=False)
    mats = build_cost_matrices(df, Vehicle(), CostParams(), valor_hora=30.0)
    return df, mats


def test_dataset_has_96_cities():
    df = build_dataset(write=False)
    validate(df)
    assert len(df) == 96
    assert df["codigo"].nunique() == 96


def test_cost_matrix_shape_and_diagonal():
    _, mats = _matrices()
    cost = mats["total"]
    assert cost.shape == (96, 96)
    assert np.allclose(np.diag(cost), 0.0)
    assert np.allclose(cost, cost.T)          # simetría
    assert np.all(np.isfinite(cost))


def test_vehicle_fuel_cost_positive():
    v = Vehicle()
    assert v.costo_combustible_km > 0


def test_aco_route_valid():
    _, mats = _matrices()
    res = ant_colony_tsp(mats["total"], seed=1, n_ants=10, n_iter=15)
    assert sorted(res.best_route) == list(range(96))   # 96 ciudades, sin repetir
    assert res.best_cost > 0


def test_ga_route_valid():
    _, mats = _matrices()
    res = genetic_algorithm_tsp(mats["total"], seed=1, pop_size=40, n_gen=20)
    assert sorted(res.best_route) == list(range(96))
    assert len(set(res.best_route)) == 96


def test_route_breakdown_consistency():
    _, mats = _matrices()
    route = list(range(96))
    bd = route_breakdown(route, mats)
    assert bd["cost_total"] > 0
    assert bd["distance_total_km"] > 0
    # cost_total ~= tiempo*valor + fuel + toll (valor_hora=30)
    approx = bd["time_total_h"] * 30.0 + bd["fuel_total"] + bd["toll_total"]
    assert abs(approx - bd["cost_total"]) < 1e-6
