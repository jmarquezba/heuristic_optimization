# Contexto del material de clase

Documento de la **Fase 1** del Trabajo 1. Inventario del material del curso
*Introducción a Redes Neuronales y Algoritmos Bioinspirados* (IRNA 2026-01,
Universidad Nacional de Colombia, prof. Juan David Ospina Arango) y su relación
con este trabajo.

> Nota: todos los archivos originales del material de clase se conservan sin
> modificación en `MaterialClase/`. Este documento solo los referencia.

## 1. Materiales revisados y tema central

| Archivo | Tema central | Relación con este trabajo |
|---|---|---|
| `Introducción método de descenso por gradiente para optimización.pdf` | Fundamentos del descenso por gradiente (GD) para optimización. | Soporte teórico directo de la Parte 1, numeral 1 (GD desde cero). |
| `IRNA202601 Perceptrón y descenso por gradiente.pdf` | GD aplicado al entrenamiento del perceptrón. | Conecta GD con el contexto de redes neuronales del curso. |
| `IRNA202601 El Perceptrón.pdf` | Modelo del perceptrón. | Marco conceptual; motivación de la optimización en RNA. |
| `IRNA202501 Optimización de RNA.pdf` | Optimización de redes neuronales (variantes de GD, *learning rate*, etc.). | Justifica criterios de parada, tasa de aprendizaje y *clipping* en `gradient_descent.py`. |
| `IRNA202601_prep_Algoritmos_Evolutivos.ipynb` (+ copia `(2)`) | Algoritmo evolutivo desde cero (`pob_inicial`, `mutacion`, `cruzamiento`, `next_gen`, elitismo) y uso de **PyGad**. Incluye Rastrigin y el problema de la mochila. | Plantilla de diseño para `evolutionary.py` y `tsp/genetic_algorithm.py`. Se reutilizan las convenciones de elitismo y mutación. |
| `IRNA202402_PSO.ipynb` | PSO con **PySwarms** (`GlobalBestPSO`, opciones `c1`, `c2`, `w`), animaciones con `plot_contour`. | Plantilla de diseño para `pso.py` y para las animaciones (Fase 8). Define convención de hiperparámetros. |
| `IRNA202601 PSO.pdf` | Teoría de PSO (posición, velocidad, mejor personal/global, inercia). | Soporte teórico de `pso.py`. |
| `IRNABI202601 ant_colony.ipynb` | TSP sintético con colonias de hormigas (implementación de J. Berroa: `ants`, `evaporation_rate`, `intensification`, `alpha`, `beta`, `choose_best`), uso de `scipy.spatial.distance_matrix`. | Plantilla de diseño para `tsp/ant_colony.py`. Convención de parámetros e idea de actividades con 100/500/1000 ciudades. |
| `Optimización combinatoria con hormigas.pdf` | Teoría de ACO (feromonas, visibilidad, evaporación, depósito). | Soporte teórico de `tsp/ant_colony.py`. |
| `IRNA202601 Knapsack Problem.pdf` | Problema de la mochila (optimización combinatoria 0/1 y k/n). | Marco de optimización combinatoria; refuerza la representación por permutaciones/enteros del GA. |
| `IRNA202601 Intro PyTorch.pdf`, `IRNA202601_PyTorch_Productización.ipynb`, `IRNA202601 Productización.pdf` | PyTorch y productización/estructura de proyectos. | Inspira la **estructura modular** del repositorio (separación lógica/experimentos/visualización), configs externas y reproducibilidad. |
| `IRNA202601_clasificación_con_un_perceptron_con_pytorch.ipynb`, `IRNA_202501_regresion_lineal_pytorch.ipynb`, `IRNA202501 S08 Regresión con Pytorch.pdf` | Regresión/clasificación con PyTorch (GD en la práctica). | Refuerzo del rol del GD en RNA; no se usa código directamente. |
| `IRNA202601 Regresión SoftMax.pdf`, `IRNA2025_Clasificación_con_MNIST.ipynb` | Clasificación multiclase. | Contexto; no usado directamente. |
| `IRNA202601 CNN.pdf` y notebooks de CNN | Redes convolucionales. | Contexto del curso; fuera del alcance de este trabajo. |
| `IRNA202606 Autoeconders.pdf`, `IRNA202601_ilustracionAutoencoders.ipynb` | Autoencoders. | Contexto; fuera de alcance. |
| `IRNA202601 Sistemas de Recomendación.pdf`, `IRNA202601_DLrecommendingsystem*.ipynb`, `ncf_model_500epochs_*.pth` | Sistemas de recomendación. | Contexto; fuera de alcance. |
| `RNN-LSTM.pdf`, `RNN_Generacion_Nombres.ipynb`, `Prediccion_acciones_*LSTM.ipynb`, `AAPL_*.csv`, `nombres_dinosaurios.txt` | Redes recurrentes / LSTM. | Contexto; fuera de alcance. |
| `trabaj2 Redes.pdf`, `Resumen_Curso_IRNA_2026.pdf` | Enunciado del trabajo 2 y resumen del curso. | Referencia administrativa. |

## 2. Materiales que sirven como soporte directo

- **Descenso por gradiente:** `Introducción método de descenso por gradiente...pdf`,
  `Perceptrón y descenso por gradiente.pdf`, `Optimización de RNA.pdf`.
- **Optimización de redes neuronales:** `Optimización de RNA.pdf`, notebooks PyTorch.
- **PSO:** `IRNA202402_PSO.ipynb`, `IRNA202601 PSO.pdf`.
- **Algoritmos evolutivos:** `IRNA202601_prep_Algoritmos_Evolutivos.ipynb`.
- **Colonias de hormigas:** `IRNABI202601 ant_colony.ipynb`, `Optimización combinatoria con hormigas.pdf`.
- **Problemas combinatorios:** `Knapsack Problem.pdf`, notebook ACO.
- **Productización / estructura de código:** `Productización.pdf`, `IRNA202601_PyTorch_Productización.ipynb`.
- **NumPy/pandas/matplotlib/notebooks:** transversal a todos los notebooks.

## 3. Decisiones tomadas a partir del material

1. **Convenciones de hiperparámetros alineadas con el curso.** El EA usa
   fracción de elitismo y de mutación (como `next_gen` del notebook). El PSO usa
   `w`, `c1`, `c2` (como PySwarms). El ACO usa `alpha`, `beta`,
   `evaporation_rate`, `intensification` (como la implementación de Berroa).
2. **Implementación propia en NumPy.** El curso usa librerías (PySwarms, PyGad,
   repo de Berroa). Para cumplir el requisito de "implementar desde cero" y el
   conteo estricto de evaluaciones, se reimplementan los algoritmos en NumPy,
   conservando las convenciones anteriores. Se citan las librerías como
   referencia conceptual en la bibliografía.
3. **Matriz de distancias con `scipy.spatial.distance_matrix`** como en el
   notebook de hormigas, pero generalizada a una matriz de costos compuesta
   (tiempo + combustible + peaje) para el problema real de Francia.
4. **Animaciones** siguiendo el patrón de los notebooks (curvas de nivel +
   posiciones por iteración para PSO; trayectoria para GD), implementadas con
   `matplotlib.animation` para no depender de `imagemagick`/Colab.
5. **Estructura modular** inspirada en el material de productización: separación
   entre lógica (`src/`), configuración (`configs/`), datos (`data/`),
   resultados y reportes.

Ver `docs/decisiones_tecnicas.md` y `SPEC.md` para el detalle metodológico.
