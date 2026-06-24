# Reporte técnico — Optimización numérica y combinatoria

**Curso:** Introducción a Redes Neuronales y Algoritmos Bioinspirados (IRNA 2026-01).
**Metodología:** Specification Driven Development (SDD).
**Documento complementario** de la entrada de blog (`reports/blog_post.md`); aquí
se detalla la metodología técnica y se consigna la bibliografía APA completa.

## 1. Entendimiento y alcance

Ver `SPEC.md` (§1–§3). El trabajo compara descenso por gradiente (GD) contra
heurísticos bioinspirados (EA, PSO, DE) en seis funciones de prueba (2D y 3D) y
resuelve un TSP de 96 ciudades francesas con ACO y GA bajo un modelo de costos
compuesto.

## 2. Metodología justificada

### 2.1 Funciones objetivo
Implementación vectorizada en NumPy con contador de evaluaciones
(`CountingFunction`). Extensión 3D `f_2D + x3²` para Goldstein-Price y *six-hump
camel* (DT-03): conserva el óptimo en `x3=0`. Validación en óptimos conocidos.

### 2.2 Descenso por gradiente
Diferencias finitas centrales (DT-02), paso `h=1e-5·(1+|x|)`, coste `2d`
evaluaciones por gradiente contabilizadas. Inicialización aleatoria, *clipping* al
dominio, parada por `‖∇f‖` y por `|Δf|`. Repeticiones n=100/500/1000 (modo full).

### 2.3 Heurísticos
- **EA:** torneo + cruce aritmético/uniforme + mutación gaussiana + elitismo.
- **PSO:** inercia `w`, cognitivo `c1`, social `c2`, límites de posición/velocidad.
- **DE:** `DE/rand/1/bin`, factor `F`, cruce `CR`, selección greedy.
≥30 corridas por combinación; parámetros en `configs/numerical_experiments.yaml`.

### 2.4 TSP
Distancia haversine × detour (1.30); tiempo a 90 km/h; combustible por
consumo×precio; peaje estimado por km de autopista (0.70×0.092 €/km). Costo total
= tiempo·valor_hora + combustible + peaje. ACO (feromona/visibilidad/evaporación/
depósito elitista) y GA (permutaciones, OX, inversión, elitismo). Tres escenarios
de valor hora (15/30/50 €/h).

## 3. Métricas

Valor final (media, mediana, desviación, mejor, peor), NFE (media/mediana), tasa
de éxito (|f−f*|<tol por función), tiempo. TSP: costo, distancia, tiempo, peajes,
combustible, costo por tiempo, iteración del mejor, semilla. Implementadas en
`metrics.py`; agregados en `numerical_summary.csv` y `tsp_summary.csv`.

## 4. Resultados y discusión

Ver `reports/blog_post.md` §4.5–4.7 y §5.7–5.9. Las cifras se obtienen al ejecutar
`run_all.py --mode full` y se leen de `data/results/`. La discusión está
sustentada en las propiedades de cada función (convexidad, multimodalidad,
ubicación del óptimo) y se confirma con los agregados.

## 5. Reproducibilidad

Semillas derivadas deterministas (DT-07); configuración externa en YAML; pruebas
pytest; pipeline `run_all.py`. Ver `README.md`.

## 6. Figuras y tablas

Todas las figuras (`assets/figures/`, `assets/gifs/`) y tablas (resúmenes CSV)
están rotuladas y referenciadas en el blog (Figuras 1–9, Tablas 1–4). No hay
elementos sueltos.

## 7. Bibliografía (APA, 7.ª edición)

Dorigo, M., & Stützle, T. (2004). *Ant colony optimization*. MIT Press.

Eberhart, R., & Kennedy, J. (1995). A new optimizer using particle swarm theory.
*Proceedings of the Sixth International Symposium on Micro Machine and Human
Science (MHS'95)*, 39–43. https://doi.org/10.1109/MHS.1995.494215

Goldstein, A. A., & Price, J. F. (1971). On descent from local minima.
*Mathematics of Computation, 25*(115), 569–574.
https://doi.org/10.1090/S0025-5718-1971-0312365-X

Holland, J. H. (1992). *Adaptation in natural and artificial systems: An
introductory analysis with applications to biology, control, and artificial
intelligence*. MIT Press. (Obra original publicada en 1975).

Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of
the IEEE International Conference on Neural Networks, 4*, 1942–1948.
https://doi.org/10.1109/ICNN.1995.488968

Miranda, L. J. V. (2018). PySwarms: A research toolkit for particle swarm
optimization in Python. *Journal of Open Source Software, 3*(21), 433.
https://doi.org/10.21105/joss.00433

Rastrigin, L. A. (1974). *Systems of extremal control*. Nauka.

Rosenbrock, H. H. (1960). An automatic method for finding the greatest or least
value of a function. *The Computer Journal, 3*(3), 175–184.
https://doi.org/10.1093/comjnl/3.3.175

Storn, R., & Price, K. (1997). Differential evolution – A simple and efficient
heuristic for global optimization over continuous spaces. *Journal of Global
Optimization, 11*(4), 341–359. https://doi.org/10.1023/A:1008202821328

Surjanovic, S., & Bingham, D. (2013). *Virtual library of simulation experiments:
Test functions and datasets — Goldstein-Price function*. Simon Fraser University.
https://www.sfu.ca/~ssurjano/goldpr.html

Surjanovic, S., & Bingham, D. (2013). *Virtual library of simulation experiments:
Test functions and datasets — Six-hump camel function*. Simon Fraser University.
https://www.sfu.ca/~ssurjano/camel6.html

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau,
D., … SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental algorithms for
scientific computing in Python. *Nature Methods, 17*, 261–272.
https://doi.org/10.1038/s41592-019-0686-2

Wikipedia contributors. (2025). *Rastrigin function*. Wikipedia.
https://en.wikipedia.org/wiki/Rastrigin_function

Wikipedia contributors. (2025). *Rosenbrock function*. Wikipedia.
https://en.wikipedia.org/wiki/Rosenbrock_function

Wikipedia contributors. (2025). *Test functions for optimization*. Wikipedia.
https://en.wikipedia.org/wiki/Test_functions_for_optimization

Wikipedia contributors. (2025). *Travelling salesman problem*. Wikipedia.
https://en.wikipedia.org/wiki/Travelling_salesman_problem

Gad, A. F. (2023). PyGAD: An intuitive genetic algorithm Python library.
*Multimedia Tools and Applications*. https://doi.org/10.1007/s11042-023-17167-y

> **Datos geográficos:** coordenadas de prefecturas tomadas de fuentes públicas
> (OpenStreetMap contributors, 2025, https://www.openstreetmap.org; y Wikipedia).
> **Combustible/peajes:** precios medios de carburantes y tarifas de autopista en
> Francia (estimaciones documentadas, ver SPEC §4, supuestos S4–S5).
