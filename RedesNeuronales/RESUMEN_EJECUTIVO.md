# Resumen ejecutivo

## Qué se construyó

Proyecto técnico completo (metodología SDD) que compara **descenso por gradiente**
contra **heurísticos bioinspirados** (EA, PSO, DE) en seis funciones de prueba, y
resuelve el **TSP de las 96 capitales de la Francia continental** con **ACO** y
**GA**, bajo un modelo de costos (tiempo + combustible + peajes).

## Archivos creados (resumen)

- **Especificación:** `SPEC.md`; decisiones en `docs/decisiones_tecnicas.md`;
  contexto del curso en `docs/contexto_material_clase.md`.
- **Código (`src/`):** funciones, gradientes, GD, EA, PSO, DE, runner y métricas
  (`optimization/`); datos de Francia, modelo de costos, ACO, GA, runner y mapas
  (`tsp/`); histogramas, convergencia, animaciones, mapas (`visualization/`);
  utilidades (`utils/`).
- **Configuración:** `configs/numerical_experiments.yaml`, `configs/tsp_experiments.yaml`
  (perfiles `fast` y `full`).
- **Datos:** dataset embebido de las 96 prefecturas (`src/tsp/data_loader.py`),
  que genera `data/processed/france_96_capitals.csv` y las matrices de costo.
- **Pruebas:** `tests/test_functions.py`, `test_algorithms.py`, `test_tsp.py`.
- **Reportes:** `reports/blog_post.md` (entrada de blog principal),
  `technical_report.md` (+ bibliografía APA), `prompts_ia.md`,
  `contribution_video_template.md`, `test_results.md`.
- **Notebooks:** `notebooks/01_exploration`, `02_numerical_results`, `03_tsp_results`.
- **Pipeline y proyecto:** `run_all.py`, `README.md`, `requirements.txt`,
  `pyproject.toml`, `Makefile`, `.gitignore`, `LICENSE`, `conftest.py`.

## Cómo ejecutar

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                                   # pruebas
python run_all.py --mode fast            # validación rápida (minutos)
python run_all.py --mode full            # cumple el enunciado (GD n=100/500/1000)
```

## Principales resultados (se obtienen al ejecutar `--mode full`)

- Tablas: `data/results/numerical_summary.csv`, `tsp_summary.csv`.
- Figuras: `assets/figures/` (histogramas, boxplots, mapas).
- Gifs: `assets/gifs/gd_rastrigin_2d.gif`, `pso_rastrigin_2d.gif`,
  `tsp_france_best_route_aco.gif`, `..._ga.gif`.

## Mejor método por familia de problemas (hipótesis sustentada por el diseño)

- **Funciones suaves / cerca del óptimo (Rosenbrock):** descenso por gradiente
  (eficiente, menos evaluaciones).
- **Funciones multimodales (Rastrigin, Griewank, Schwefel):** heurísticos
  (DE/PSO/EA) por su exploración global, a costa de más evaluaciones.
- **TSP combinatorio:** ACO y GA encuentran rutas de calidad; ACO suele converger
  más rápido.

## Ubicaciones clave

- Reporte/blog: `reports/blog_post.md`.
- Gráficos: `assets/figures/`. Gifs: `assets/gifs/`.
- Resultados: `data/results/`.

## Pendientes (dependen del usuario)

1. Crear el repositorio en GitHub y subir el proyecto.
2. Pegar la URL del repositorio en `README.md` y en `reports/blog_post.md` (§10).
3. Grabar el video de contribución individual con `reports/contribution_video_template.md`.
4. Ejecutar `python run_all.py --mode full` para poblar tablas, figuras y gifs, y
   pegar la salida de `pytest` en `reports/test_results.md`.

> **Nota de transparencia:** el entorno de ejecución de Python no estuvo disponible
> durante la generación automática, por lo que el código quedó **listo para
> ejecutar** pero los artefactos de resultados (CSV, figuras, gifs) deben generarse
> corriendo el pipeline. Todo lo demás (código, datos, especificación, reportes,
> pruebas) está completo.
