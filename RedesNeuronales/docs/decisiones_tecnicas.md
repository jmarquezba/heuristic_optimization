# Decisiones técnicas

Registro vivo de decisiones (Architecture Decision Records simplificados). Cada
decisión documenta contexto, opción elegida y justificación. Complementa `SPEC.md`.

## DT-01 — Implementación propia en NumPy en lugar de librerías
**Contexto:** el curso usa PySwarms (PSO), PyGad (EA) y el repo de Berroa (ACO).
**Decisión:** reimplementar todo en NumPy.
**Justificación:** el enunciado pide implementar los métodos y exige conteo
estricto de evaluaciones de la función objetivo, que las librerías no exponen de
forma uniforme. Las librerías se citan como referencia conceptual.

## DT-02 — Gradiente por diferencias finitas centrales
**Decisión:** GD usa diferencias finitas centrales con paso `h=1e-5·(1+|x|)`.
**Justificación:** trata las seis funciones de forma homogénea sin derivar a
mano cada una; cada gradiente cuesta `2d` evaluaciones que se cuentan en el NFE.
Se validan contra gradientes analíticos de Rosenbrock y Rastrigin (tests).
**Consecuencia:** el NFE del GD es alto (≈ `2d` por iteración), lo que es
relevante para la comparación con heurísticos.

## DT-03 — Extensión a 3D de funciones canónicas 2D
**Decisión:** `f_3D(x1,x2,x3) = f_2D(x1,x2) + x3²` para Goldstein-Price y
*six-hump camel*.
**Justificación:** son funciones de 2 variables; el término `x3²` es convexo,
se minimiza en `x3=0` y no altera el óptimo principal. Permite estudiar el
efecto de la dimensión de forma controlada.

## DT-04 — Tasas de aprendizaje por función
**Decisión:** `lr` específico para Schwefel (1e-1), Griewank (5e-1) y
Goldstein-Price (1e-5) en `configs/`.
**Justificación:** los dominios y escalas de gradiente difieren en órdenes de
magnitud (Goldstein-Price tiene gradientes enormes cerca de los bordes;
Schwefel/Griewank tienen dominios muy amplios). Un `lr` único diverge.

## DT-05 — Modelo de costos del TSP
**Decisión:** costo compuesto tiempo+combustible+peaje con haversine×detour.
**Justificación:** no hay matriz oficial de peajes por tramo accesible sin
servicios pagos/credenciales. Se documenta como estimación (S2–S5) con fórmulas
y fuentes; se evita inventar datos puntuales.

## DT-06 — Animación TSP sobre Francia (no México)
**Decisión:** la visualización principal es sobre Francia; México solo como
anexo experimental rotulado, si se genera.
**Justificación:** las coordenadas del problema son francesas; un mapa de México
sería incoherente con los datos. Inconsistencia A1 documentada en el reporte.

## DT-07 — Semillas derivadas deterministas
**Decisión:** `derive_seed(base, run_index)` y offsets fijos por algoritmo.
**Justificación:** reproducibilidad total sin colisiones entre corridas; se evita
`hash()` de cadenas (no determinista entre procesos de Python).

## DT-08 — Animaciones con PillowWriter
**Decisión:** usar `matplotlib.animation.PillowWriter` en vez de imagemagick.
**Justificación:** evita dependencias del sistema (el notebook de clase requería
`apt install imagemagick`); funciona en cualquier entorno con matplotlib.

## DT-09 — Modos fast / full
**Decisión:** dos perfiles en los YAML.
**Justificación:** el modo full (GD n=100/500/1000 × 6 funciones × 2 dims +
heurísticos ×30 + TSP 96 ciudades) es costoso; el modo fast valida el pipeline
en minutos. El modo full cumple los requisitos del enunciado.
