# Plantilla — Video de contribución individual

Cada integrante graba un video **en primera persona** describiendo sus aportes
específicos al trabajo. Esta plantilla es editable: reemplaza los campos entre
`<...>` y borra lo que no aplique.

## Instrucciones para grabar

- **Duración sugerida:** 2 a 4 minutos por estudiante.
- **Formato:** cámara activa (opcional), audio claro; puede mostrarse la pantalla
  con el código/figuras al hablar de un aporte concreto.
- **Contenido:** habla en primera persona ("yo hice…", "me encargué de…"). Sé
  concreto: menciona archivos, funciones o figuras específicas.
- **Estructura recomendada:** (1) qué hiciste, (2) cómo lo hiciste, (3) qué
  validaste, (4) qué aprendiste.
- **Entrega:** sube el video y pega el enlace en la tabla de abajo.

## Guion base (rellenar por estudiante)

### Estudiante: `<Nombre completo>`

> "Hola, soy `<nombre>`. En este trabajo **yo contribuí a** `<área principal>`.
>
> **Me encargué de** `<tarea concreta, p. ej. implementar el descenso por
> gradiente en src/optimization/gradient_descent.py>`.
>
> **Validé** `<qué verificaste, p. ej. que las funciones alcanzan su óptimo
> conocido en tests/test_functions.py>`.
>
> **Aprendí que** `<aprendizaje, p. ej. que el GD es muy sensible a la
> inicialización en funciones multimodales como Rastrigin>`."

- **Enlace al video:** `<URL>`

---

## Ejemplos de aportes concretos (para inspirarse)

- **Programación del descenso por gradiente** (`gradient_descent.py`,
  `gradients.py`): condición inicial aleatoria, criterios de parada, conteo de
  evaluaciones.
- **Implementación de PSO** (`pso.py`): inercia y coeficientes cognitivo/social,
  límites de velocidad.
- **Implementación del algoritmo evolutivo / evolución diferencial**
  (`evolutionary.py`, `differential_evolution.py`).
- **Implementación de colonias de hormigas** (`tsp/ant_colony.py`): feromonas,
  visibilidad, evaporación, depósito.
- **Algoritmo genético para el TSP** (`tsp/genetic_algorithm.py`): cruce OX,
  mutación por inversión.
- **Construcción del dataset de Francia** (`tsp/data_loader.py`): 96 prefecturas,
  coordenadas y validación.
- **Modelo de costos** (`tsp/cost_model.py`): tiempo, combustible, peajes;
  escenarios de valor hora.
- **Generación de gráficos y gifs** (`visualization/*`): histogramas, mapas,
  animaciones.
- **Redacción del reporte** (`reports/blog_post.md`, `technical_report.md`):
  metodología, discusión, bibliografía APA.
- **Validación de resultados** (`tests/`, `reports/test_results.md`).
- **Organización del repositorio** (estructura, `run_all.py`, README).

## Tabla resumen de contribuciones

| Estudiante | Aportes principales | Enlace al video |
|---|---|---|
| `<Nombre 1>` | `<…>` | `<URL>` |
| `<Nombre 2>` | `<…>` | `<URL>` |
| `<Nombre 3>` | `<…>` | `<URL>` |
