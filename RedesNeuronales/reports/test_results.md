# Evidencia de pruebas

## Cómo ejecutar

```bash
pytest -q
```

Las pruebas están en `tests/` y cubren las validaciones de la Fase 12 del
enunciado. El archivo `conftest.py` añade la raíz del proyecto a `sys.path` para
los imports `src.*`.

## Cobertura de pruebas

### `tests/test_functions.py`
- Cada función alcanza su valor mínimo conocido en el óptimo (las seis, 2D y 3D).
- Rosenbrock = 0 en el vector de unos; Goldstein-Price = 3 en (0, -1);
  *six-hump camel* ≈ -1.0316 en sus dos óptimos.
- La extensión 3D `f_2D + x3²` coincide con la 2D en `x3=0`.
- El contador de evaluaciones incrementa correctamente (vector = 1, matriz n×d = n).
- Ausencia de NaN/Inf en evaluaciones aleatorias.
- El gradiente por diferencias finitas coincide con el analítico (Rosenbrock,
  Rastrigin).

### `tests/test_algorithms.py`
- GD respeta el dominio (*clipping*) y cuenta evaluaciones.
- GD reduce sustancialmente Rosenbrock en al menos una corrida.
- PSO y DE convergen por debajo de umbrales razonables en funciones multimodales.
- EA encuentra valores negativos en *six-hump camel* (cuyo mínimo es -1.0316).
- El historial del mejor valor global es monótono no creciente.

### `tests/test_tsp.py`
- El dataset tiene exactamente 96 ciudades, sin duplicados, coords en Francia.
- La matriz de costos es 96×96, simétrica y con diagonal cero.
- El costo de combustible por km del vehículo es positivo.
- Las rutas de ACO y GA contienen las 96 ciudades exactamente una vez.
- La descomposición de costo de una ruta es consistente con la fórmula.

## Resultado de la última ejecución

Ejecutado correctamente. **31 pruebas pasaron** (0 fallos):

```
$ pytest -q
...............................                                          [100%]
31 passed
```

Todas las pruebas de funciones, algoritmos y TSP pasan. El experimento numérico
y el TSP se ejecutaron y produjeron resultados reales en `data/results/`
(`numerical_results.csv` con 1 440 corridas, `numerical_summary.csv`,
`tsp_results.csv`, `tsp_summary.csv`), además de 84 figuras en `assets/figures/`
y 4 GIFs en `assets/gifs/`.

**Criterio de aceptación:** todas las pruebas deben pasar (`N passed`). Si alguna
falla por un umbral estocástico ajustado, revisar la semilla o el presupuesto en
`configs/` (los umbrales se eligieron holgados, pero los heurísticos son
estocásticos).
