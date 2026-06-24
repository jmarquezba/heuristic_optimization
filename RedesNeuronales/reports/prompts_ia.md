# Reporte de uso de IA

Este proyecto se desarrolló con asistencia de un agente de IA (Claude, en modo
agente autónomo) bajo supervisión del equipo. Aquí se reportan los prompts
principales, lo que produjo cada uno, qué se revisó y el impacto en el resultado.

## 1. Prompt principal

> *"Actúa como un agente experto en redes neuronales, optimización numérica y
> algoritmos bioinspirados. Desarrolla el Trabajo 1 completo con metodología SDD:
> crea la especificación técnica antes de programar; implementa las seis funciones
> de prueba, descenso por gradiente, EA, PSO, DE, ACO y GA para el TSP de las 96
> capitales de Francia continental; genera histogramas, animaciones, mapas,
> pruebas, reporte tipo blog, bibliografía APA, reporte de IA, plantilla de video
> y README. Trabaja de forma autónoma, documenta los supuestos y deja listo todo
> menos el repositorio GitHub y el video."*

**Qué produjo:** la estructura completa del proyecto, `SPEC.md`, el código de
`src/`, las configuraciones, el dataset de Francia, las pruebas y los reportes.

**Qué se revisó:** el equipo validó los óptimos de las funciones, la coherencia de
los dominios, los supuestos del modelo de costos y la resolución de las
ambigüedades (3D y mapa de México).

## 2. Prompts secundarios

| # | Prompt (resumen) | Produjo | Revisión del equipo |
|---|---|---|---|
| 2.1 | "Implementa las seis funciones con conteo de evaluaciones y validación en sus óptimos." | `functions.py`, `test_functions.py` | Verificación de fórmulas y mínimos conocidos. |
| 2.2 | "Implementa GD desde cero con diferencias finitas y registro de trayectoria." | `gradient_descent.py`, `gradients.py` | Comparación FD vs. analítico; ajuste de `lr` por función. |
| 2.3 | "Implementa EA, PSO y DE con convenciones del material de clase." | `evolutionary.py`, `pso.py`, `differential_evolution.py` | Alineación de hiperparámetros con notebooks. |
| 2.4 | "Construye el dataset de las 96 prefecturas con coordenadas y fuentes." | `data_loader.py` | Verificación de que son 96 (2A/2B), coords en Francia. |
| 2.5 | "Define un modelo de costos tiempo+combustible+peaje con supuestos documentados." | `cost_model.py` | Elección del vehículo y validación de fórmulas. |
| 2.6 | "Implementa ACO y GA para el TSP con rutas válidas y registro por iteración." | `ant_colony.py`, `genetic_algorithm.py`, `test_tsp.py` | Validación de permutaciones y cierre del ciclo. |
| 2.7 | "Genera histogramas, animaciones y mapas; pipeline fast/full." | `visualization/*`, `run_all.py` | Selección de Rastrigin 2D para animar; modos fast/full. |
| 2.8 | "Redacta el reporte tipo blog, técnico, bibliografía APA y este reporte de IA." | `reports/*` | Estilo, citas, rotulado de figuras/tablas. |

## 3. Qué partes fueron revisadas o corregidas

- **Tasas de aprendizaje** del GD por función (la IA propuso una única, el equipo
  diferenció Schwefel/Griewank/Goldstein-Price para evitar divergencia).
- **Reproducibilidad:** se reemplazó `hash()` de cadenas (no determinista) por
  offsets fijos por algoritmo.
- **Coherencia geográfica:** se confirmó la decisión de usar Francia (no México).
- **Estimaciones de costos:** el equipo fijó los valores de consumo, precio de
  combustible, fracción de autopista y tarifa de peaje, marcándolos como supuestos.

## 4. Impacto del uso de IA en el resultado final

La IA **aceleró notablemente** la generación de la especificación, el código
modular y la documentación, y ayudó a mantener consistencia entre las partes
(nombres, convenciones, rotulado). Permitió cubrir un alcance amplio en poco
tiempo. El **criterio técnico del equipo** fue indispensable para: definir los
supuestos del modelo de costos, validar los óptimos, ajustar hiperparámetros,
resolver ambigüedades del enunciado y verificar que no se reportaran datos
inventados.

## 5. Decisiones que quedaron bajo criterio del equipo

- Elección del vehículo y de los valores del modelo de costos.
- Escenarios del valor hora (15/30/50 €/h).
- Interpretación de las ambigüedades (extensión 3D y mapa de México).
- Presupuestos de cómputo (modos fast/full) y parámetros de los algoritmos.

## 6. Limitaciones del uso de IA

- La IA no puede acceder a matrices oficiales de peajes por tramo (servicios
  pagos/credenciales); por eso se usaron estimaciones documentadas.
- Los valores numéricos sugeridos por la IA siempre se trataron como **propuestas
  a validar**, no como hechos.
- El entorno de ejecución puede variar; los resultados cuantitativos finales los
  produce el equipo al correr el pipeline.
