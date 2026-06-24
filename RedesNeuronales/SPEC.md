# SPEC.md — Especificación técnica (SDD)

**Proyecto:** Comparación de métodos de optimización basados en gradiente,
heurísticos bioinspirados y combinatorios.
**Curso:** Introducción a Redes Neuronales y Algoritmos Bioinspirados (IRNA 2026-01).
**Modo de trabajo:** Specification Driven Development (SDD). Esta especificación
se redactó **antes** de la implementación y guía todo el desarrollo.

---

## 1. Entendimiento del problema

El trabajo tiene dos partes:

- **Parte 1 — Optimización numérica.** Optimizar seis funciones de prueba
  (Rosenbrock, Rastrigin, Schwefel, Griewank, Goldstein-Price, *six-hump camel*)
  en 2D y 3D mediante (a) descenso por gradiente (GD) con condición inicial
  aleatoria repetido n = 100, 500, 1000 veces, y (b) tres métodos heurísticos:
  algoritmo evolutivo (EA), optimización por enjambre de partículas (PSO) y
  evolución diferencial (DE). Se comparan calidad de la solución y número de
  evaluaciones de la función objetivo (NFE).
- **Parte 2 — Optimización combinatoria.** Resolver un TSP sobre las capitales
  de los 96 departamentos de la Francia continental con colonias de hormigas
  (ACO) y algoritmo genético (GA), con un costo de viaje compuesto
  (valor del tiempo del vendedor + combustible + peajes). Se estudia el valor
  hora del vendedor en tres escenarios.

El hilo conductor con el curso es la **optimización**: el GD es el motor de
entrenamiento de redes neuronales, y los métodos bioinspirados son alternativas
para superficies no convexas y problemas combinatorios.

## 2. Alcance funcional

1. Seis funciones objetivo con dominio, óptimo conocido y conteo de evaluaciones.
2. GD desde cero (NumPy) con inicialización aleatoria, semilla, criterios de
   parada y registro de trayectoria.
3. EA, PSO y DE desde cero (NumPy).
4. Runner de experimentos numéricos que produce CSV de resultados y resumen.
5. Histogramas (solución final, valor final, NFE) y curvas de convergencia.
6. Dos animaciones (GD y heurístico) en una función 2D.
7. Dataset de 96 capitales francesas con coordenadas y fuentes.
8. Modelo de costos (tiempo, combustible, peajes) y matrices 96×96.
9. ACO y GA para TSP con tres escenarios de valor hora.
10. Mapas estáticos y animación de la mejor ruta sobre Francia.
11. Pruebas pytest, reporte/blog, bibliografía APA, reporte de uso de IA,
    plantilla de video, README y pipeline `run_all.py` con modos fast/full.

## 3. Alcance técnico

- **Lenguaje:** Python ≥ 3.10. **Dependencias núcleo:** NumPy, pandas,
  matplotlib, SciPy, PyYAML, pytest, imageio (gifs). `geopandas`/`contextily`
  son **opcionales**; si no están, los mapas usan dispersión de lat/lon con
  contorno del hexágono francés aproximado.
- **Reproducibilidad:** semillas controladas (`utils/seed.py`), configs en YAML.
- **Estilo:** funciones pequeñas, *type hints*, *docstrings*, *logging*,
  resultados persistidos en disco.

## 4. Supuestos

- **S1.** Coordenadas de las prefecturas: datos geográficos públicos
  (Wikipedia / OpenStreetMap / IGN), precisión ~0.01°. Suficiente para distancias
  inter-ciudad de decenas a cientos de km.
- **S2.** Distancia por carretera ≈ distancia *haversine* × factor de detour
  `f_road = 1.30` (valor documentado típico para redes viales europeas).
- **S3.** Velocidad media efectiva puerta a puerta: 90 km/h (mezcla autopista/
  nacional). Tiempo = distancia_carretera / velocidad.
- **S4.** Peajes: modelo lineal por km de autopista. Se asume que una fracción
  `frac_autopista = 0.70` del recorrido es autopista de peaje, con tarifa
  `toll_eur_per_km ≈ 0.092 €/km` (orden de magnitud de las autopistas francesas).
  **Es una estimación documentada**, no una matriz oficial de peajes por tramo.
- **S5.** Vehículo base: **Renault Clio V 1.0 TCe (gasolina)**, consumo mixto
  ≈ 5.3 L/100 km (WLTP), precio gasolina SP95-E10 ≈ 1.75 €/L (orden de magnitud
  2024–2025 en Francia). Ver `configs/tsp_experiments.yaml`.
- **S6.** El TSP es simétrico (costo i→j = costo j→i) y se cierra volviendo al
  origen.
- **S7.** Para GD se usan **gradientes por diferencias finitas centrales**
  (decisión D3), contando cada evaluación adicional en el NFE.

## 5. Ambigüedades detectadas

- **A1 — Mapa de México vs. ruta de Francia.** El enunciado pide la ruta por
  Francia continental pero la animación "en el mapa de México". **Inconsistencia.**
  Resolución: la visualización **principal** se hace sobre Francia (coherente con
  las coordenadas reales). Se documenta la inconsistencia y, opcionalmente, se
  genera un anexo experimental claramente rotulado. (Ver D5.)
- **A2 — 3D en funciones 2D canónicas.** Goldstein-Price y *six-hump camel* son
  funciones de 2 variables. Resolución: extensión documentada (ver §7 y D2).
- **A3 — "Tasa de éxito".** No se define umbral. Resolución: éxito si
  `f_final - f_opt < tol_exito` con `tol_exito` por función (D4).
- **A4 — Número de evaluaciones objetivo en heurísticos.** Se cuenta cada llamada
  a `f`, incluida la evaluación de la población inicial.

## 6. Decisiones metodológicas

- **D1.** Implementación propia en NumPy de todos los algoritmos para permitir
  el conteo estricto de evaluaciones y cumplir "desde cero".
- **D2.** Extensión a 3D para funciones canónicas 2D:
  `f_ext(x1,x2,x3) = f_2D(x1,x2) + x3²`. Conserva la función original en
  `x3 = 0` y mantiene el óptimo principal (el término `x3²` se minimiza en 0),
  permitiendo comparar el comportamiento en 3D sin alterar el óptimo.
- **D3.** Gradiente por **diferencias finitas centrales** con paso
  `h = 1e-5·(1+|x|)`; coste `2d` evaluaciones por gradiente, todas contadas.
  Se elige sobre gradientes analíticos para tratar las seis funciones de forma
  uniforme y validada. (El módulo `gradients.py` también ofrece analíticos
  validados para Rosenbrock/Rastrigin como verificación cruzada.)
- **D4.** Tolerancias de éxito por función en `configs/numerical_experiments.yaml`.
- **D5.** Animación TSP principal sobre Francia; México solo como anexo
  experimental rotulado (no resultado principal).
- **D6.** GD con *clipping* al dominio en cada paso (proyección a la caja).
- **D7.** Para comparar GD vs. heurísticos de forma justa se reporta NFE además
  del valor final; se hacen múltiples corridas independientes.

## 7. Funciones objetivo y dominios

| Función | Dominio por dimensión | Óptimo (x*) | f(x*) |
|---|---|---|---|
| Rosenbrock | [-5, 10] | (1,…,1) | 0 |
| Rastrigin | [-5.12, 5.12] | (0,…,0) | 0 |
| Schwefel | [-500, 500] | (420.9687,…) | ≈ 0 |
| Griewank | [-600, 600] | (0,…,0) | 0 |
| Goldstein-Price (2D) | [-2, 2]² | (0, -1) | 3 |
| Six-hump camel (2D) | x1∈[-3,3], x2∈[-2,2] | (0.0898,-0.7126) y (-0.0898,0.7126) | ≈ -1.0316 |

**Extensión 3D (D2):** Goldstein-Price 3D usa `x3∈[-2,2]`; six-hump camel 3D usa
`x3∈[-3,3]`; en ambos `f_3D = f_2D + x3²`, con óptimo en `x3=0` y mismo `f(x*)`.
Las demás funciones son de dimensión arbitraria y se evalúan en 2D y 3D.

## 8. Algoritmos a implementar

- **GD** (`gradient_descent.py`): paso fijo, criterios de parada por `‖∇f‖` y por
  `|Δf|`, máx. iteraciones, *clipping*, registro de trayectoria y convergencia.
- **EA** (`evolutionary.py`): selección por torneo, cruce aritmético/uniforme,
  mutación gaussiana, elitismo.
- **PSO** (`pso.py`): posición/velocidad, `b_i` y `g`, inercia `w`, cognitivo
  `c1`, social `c2`, límites de posición y velocidad.
- **DE** (`differential_evolution.py`): estrategia `DE/rand/1/bin`, factor `F`,
  tasa de cruce `CR`, selección por supervivencia.
- **ACO** (`tsp/ant_colony.py`): feromonas, visibilidad `η = 1/costo`,
  evaporación, depósito `Q`, varias hormigas, ciclo cerrado.
- **GA-TSP** (`tsp/genetic_algorithm.py`): permutaciones, torneo, cruce OX,
  mutación por inversión/swap, elitismo.

## 9. Métricas de evaluación

- Valor final de `f` (media, mediana, desv. estándar, mejor, peor).
- NFE (media, mediana).
- Tasa de éxito (según tol por función).
- Tiempo de ejecución por corrida.
- Para TSP: costo total, distancia, tiempo, peajes, combustible, costo por
  tiempo, iteración/generación del mejor, semilla.

## 10. Protocolo experimental

- **GD:** por cada función × dim ∈ {2,3} × n ∈ {100, 500, 1000}: n corridas con
  inicialización aleatoria; se guardan solución, valor, NFE, convergencia,
  semilla, tiempo.
- **Heurísticos:** por cada función × dim × algoritmo ∈ {EA, PSO, DE}:
  ≥ 30 corridas independientes (config). Presupuesto de NFE comparable entre
  heurísticos.
- **TSP:** ACO y GA × escenario hora ∈ {bajo, medio, alto}; semillas fijas;
  registro de la mejor ruta por iteración/generación.
- **Modos:** `fast` (pocas corridas/iteraciones, para validar el código) y
  `full` (cumple los requisitos: GD con n=100/500/1000, ≥30 corridas heurísticas).

## 11. Formato de resultados

- `data/results/numerical_results.csv` — una fila por corrida.
- `data/results/numerical_summary.csv` — agregados por función/dim/algoritmo.
- `data/results/tsp_results.csv`, `tsp_best_routes.csv`, `tsp_summary.csv`.
- Figuras en `assets/figures/`, gifs en `assets/gifs/`, tablas en `assets/tables/`.

## 12. Estructura esperada de carpetas

```
src/optimization/{functions,gradients,gradient_descent,evolutionary,pso,
                  differential_evolution,experiment_runner,metrics}.py
src/tsp/{data_loader,cost_model,ant_colony,genetic_algorithm,route_plotter,tsp_runner}.py
src/visualization/{histograms,convergence_plots,animations,maps}.py
src/utils/{config,seed,io,logging}.py
configs/{numerical_experiments,tsp_experiments}.yaml
data/{raw,processed,results}/
reports/{blog_post,technical_report,prompts_ia,contribution_video_template}.md
assets/{figures,gifs,tables}/
tests/{test_functions,test_algorithms,test_tsp}.py
notebooks/{01_exploration,02_numerical_results,03_tsp_results}.ipynb
docs/{contexto_material_clase,decisiones_tecnicas}.md
README.md  requirements.txt  pyproject.toml  .gitignore  run_all.py  Makefile  LICENSE
```

## 13. Criterios de aceptación

Ver Fase 19 del enunciado. Resumido: SPEC completo; código modular; pruebas
pytest; CSV de resultados; tablas resumen; histogramas; gifs; mapas; reporte/blog;
bibliografía APA; reporte de prompts; plantilla de contribución; README; guía de
ejecución; evidencia de pruebas. El reporte debe permitir entender qué se hizo,
por qué, qué algoritmos se compararon, resultados, mejor método por caso, costo
en evaluaciones y cómo reproducir.

## 14. Riesgos técnicos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Costo computacional del modo full (GD n=1000 × 6 funciones × 2 dims). | Modo `fast` para validación; vectorización NumPy; `n_jobs` opcional. |
| Schwefel/Griewank: GD se atasca en mínimos locales. | Es el resultado esperado y forma parte de la discusión; se reporta. |
| Falta de `geopandas`. | *Fallback* a dispersión lat/lon con marco aproximado de Francia. |
| Datos de peajes no oficiales. | Modelo estimado documentado (S4), marcado como estimación. |
| Gifs pesados. | Submuestreo de frames y opción MP4. |

## 15. Plan de validación

- Pruebas unitarias de funciones en sus óptimos conocidos (tolerancias).
- Verificación del conteo de evaluaciones (monótono creciente).
- Ausencia de NaN/Inf en soluciones.
- TSP: rutas con 96 ciudades, sin repetición (salvo retorno), matrices 96×96 con
  diagonal cero y simetría.
- Verificación cruzada gradiente analítico vs. diferencias finitas.

## 16. Entregables finales

SPEC.md, código en `src/`, configs, datos `data/`, resultados CSV, figuras, gifs,
mapas, `reports/*` (blog, técnico, prompts IA, plantilla video), README, pruebas
y evidencia (`reports/test_results.md`), `run_all.py`. Pendientes del usuario:
crear el repositorio GitHub, pegar su URL y grabar el video de contribución.
