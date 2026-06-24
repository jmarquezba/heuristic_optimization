# Optimización: gradiente, heurísticos bioinspirados y TSP (IRNA 2026-01)

Proyecto del **Trabajo 1** del curso *Introducción a Redes Neuronales y
Algoritmos Bioinspirados* (Universidad Nacional de Colombia). Compara métodos de
**descenso por gradiente** frente a **heurísticos bioinspirados** (algoritmo
evolutivo, PSO, evolución diferencial) sobre seis funciones de prueba, y resuelve
un **TSP** sobre las 96 capitales departamentales de la Francia continental con
**colonias de hormigas (ACO)** y **algoritmo genético (GA)**.

> Desarrollado con metodología **SDD** (Specification Driven Development): ver
> `SPEC.md` antes que el código.

## Descripción

- **Parte 1 — Optimización numérica.** GD desde cero con condición inicial
  aleatoria repetido n = 100/500/1000; EA, PSO y DE con ≥30 corridas;
  histogramas de solución/valor/NFE; animaciones de GD y de un heurístico.
- **Parte 2 — Optimización combinatoria.** TSP de Francia con modelo de costos
  (tiempo del vendedor + combustible + peajes), ACO y GA, tres escenarios de
  valor hora, mapas y animación de la mejor ruta.

## Requisitos

- Python ≥ 3.10. Dependencias en `requirements.txt`
  (NumPy, pandas, SciPy, matplotlib, PyYAML, imageio, pytest).
- Opcional para mapas con fondo geográfico: `geopandas`, `contextily`
  (si faltan, se usa dispersión lat/lon; el resultado es válido).

## Instalación

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Estructura de carpetas

```
src/optimization/   funciones, gradientes, GD, EA, PSO, DE, runner, métricas
src/tsp/            datos Francia, modelo de costos, ACO, GA, runner, mapas
src/visualization/  histogramas, convergencia, animaciones, mapas
src/utils/          config, semillas, io, logging
configs/            numerical_experiments.yaml, tsp_experiments.yaml
data/               raw/ processed/ results/
assets/             figures/ gifs/ tables/
reports/            blog_post.md, technical_report.md, prompts_ia.md, ...
tests/              test_functions.py, test_algorithms.py, test_tsp.py
notebooks/          01_exploration, 02_numerical_results, 03_tsp_results
docs/               contexto_material_clase.md, decisiones_tecnicas.md
run_all.py          pipeline completo (modos fast/full)
```

## Cómo ejecutar las pruebas

```bash
pytest
```

## Cómo correr los experimentos

```bash
# Validación rápida (minutos):
python run_all.py --mode fast

# Cumple los requisitos del enunciado (GD n=100/500/1000, >=30 corridas):
python run_all.py --mode full

# Solo una parte:
python run_all.py --only numeric --mode full
python run_all.py --only tsp --mode full

# Sin animaciones (más rápido):
python run_all.py --mode full --no-animations
```

### Construir solo los datos de Francia
```bash
python -m src.tsp.data_loader
```

## Cómo generar gráficos y gifs

`run_all.py` genera automáticamente histogramas (`assets/figures/`), boxplots,
mapas y animaciones (`assets/gifs/`). Las animaciones requeridas:
- `assets/gifs/gd_rastrigin_2d.gif` — descenso por gradiente.
- `assets/gifs/pso_rastrigin_2d.gif` — PSO.
- `assets/gifs/tsp_france_best_route_aco.gif`, `..._ga.gif` — mejor ruta.

## Cómo regenerar el reporte

El reporte (`reports/blog_post.md`) referencia figuras y las tablas resumen
(`data/results/*.csv`). Tras correr `run_all.py --mode full`, las figuras y CSVs
se actualizan y el reporte queda consistente. Las tablas pueden reconstruirse
desde `numerical_summary.csv` y `tsp_summary.csv`.

## Resultados (archivos)

- `data/results/numerical_results.csv`, `numerical_summary.csv`
- `data/results/tsp_results.csv`, `tsp_best_routes.csv`, `tsp_summary.csv`
- `data/processed/france_96_capitals.csv` y matrices `tsp_*_matrix*.csv`

## Repositorio Git

> **URL del repositorio:** https://github.com/jmarquezba/heuristic_optimization

## Autores

Equipo IRNA 2026-01. Ver `reports/contribution_video_template.md` para los aportes
individuales.

## Licencia

MIT. Ver `LICENSE`.
