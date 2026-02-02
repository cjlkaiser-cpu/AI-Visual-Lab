# Perceptrón Viviente — Clasificador Lineal 2D Interactivo

> Simulación interactiva del perceptrón de Rosenblatt: la neurona artificial más simple que aprende a trazar una frontera de decisión lineal en el plano. Observa cómo los pesos se ajustan epoch tras epoch, escucha la disonancia convertirse en consonancia, y descubre por qué el XOR rompió los sueños de una generación de investigadores.

---

## Tabla de Contenidos

1. [Introducción](#1-introduccion)
2. [Conceptos Fundamentales](#2-conceptos-fundamentales)
3. [La Interfaz](#3-la-interfaz)
4. [Controles Interactivos](#4-controles-interactivos)
5. [Las Matemáticas](#5-las-matematicas)
6. [Sonificación](#6-sonificacion)
7. [Guía Paso a Paso](#7-guia-paso-a-paso)
8. [Conceptos Avanzados](#8-conceptos-avanzados)
9. [Ejercicios](#9-ejercicios)
10. [Glosario](#10-glosario)
11. [Referencias](#11-referencias)

---

## 1. Introducción

En 1958, Frank Rosenblatt presentó el **perceptrón**: un algoritmo inspirado en la neurona biológica capaz de aprender a clasificar patrones. Era la primera vez que una máquina podía *aprender de los datos* sin ser programada explícitamente para cada caso.

Esta simulación te permite experimentar con un perceptrón de dos entradas en tiempo real. Puedes colocar puntos en un plano 2D, asignarles clases, y observar cómo el algoritmo ajusta sus pesos para encontrar una **frontera de decisión** — una línea recta que separa las dos clases.

El perceptrón es el átomo de las redes neuronales modernas. Entenderlo profundamente es el primer paso para comprender arquitecturas con millones de parámetros. Esta simulación hace visible lo que normalmente ocurre en silencio dentro de un `model.fit()`.

### ¿Para quién es esta guía?

- Estudiantes que se inician en machine learning
- Programadores curiosos sobre cómo funcionan las redes neuronales por dentro
- Cualquiera que quiera *ver* y *escuchar* cómo aprende una neurona artificial

---

## 2. Conceptos Fundamentales

### 2.1 La neurona artificial

El perceptrón toma dos valores de entrada $x_1$ y $x_2$, los multiplica por sus respectivos **pesos** $w_1$ y $w_2$, suma un **sesgo** (bias) $b$, y aplica una función de activación escalón:

$$y = \text{sign}(w_1 x_1 + w_2 x_2 + b)$$

Si la suma ponderada es positiva o cero, la salida es **clase 1** (rojo). Si es negativa, es **clase 0** (azul).

### 2.2 La frontera de decisión

La ecuación $w_1 x_1 + w_2 x_2 + b = 0$ define una **línea recta** en el plano. Esta línea es la frontera de decisión: todos los puntos de un lado se clasifican como clase 0, y los del otro como clase 1.

La línea amarilla que ves en la simulación *es* esta ecuación hecha visible. El **vector de pesos** $\mathbf{w} = (w_1, w_2)$ es perpendicular a la frontera y apunta hacia la región de clase 1.

### 2.3 Separabilidad lineal

Un conjunto de datos es **linealmente separable** si existe una línea recta (en 2D) o un hiperplano (en dimensiones superiores) que separa perfectamente las dos clases. El perceptrón solo puede resolver problemas linealmente separables.

### 2.4 El vector de pesos

El vector $\mathbf{w} = (w_1, w_2)$ tiene una interpretación geométrica directa:

- Su **dirección** determina la orientación de la frontera de decisión
- Su **magnitud** $\|\mathbf{w}\| = \sqrt{w_1^2 + w_2^2}$ indica la "confianza" del clasificador
- El **sesgo** $b$ desplaza la frontera paralelamente, alejándola del origen

---

## 3. La Interfaz

### 3.1 Estructura general

La simulación se divide en dos áreas principales:

| Área | Ubicación | Función |
|------|-----------|---------|
| **Canvas principal** | Izquierda (área grande) | Plano 2D con puntos, frontera de decisión, regiones coloreadas |
| **Panel de controles** | Derecha (360px) | Ecuación, dataset, parámetros, entrenamiento, audio, estadísticas |

### 3.2 El canvas

El canvas muestra un plano cartesiano con ejes $x_1$ (horizontal) y $x_2$ (vertical), ambos en el rango $[-1, 1]$. Los elementos visuales son:

- **Fondo degradado**: azul (clase 0) a rojo (clase 1), mostrando la región de decisión continua calculada con una sigmoide suave sobre la salida cruda del perceptrón
- **Línea amarilla**: la frontera de decisión $w_1 x_1 + w_2 x_2 + b = 0$
- **Flecha punteada**: el vector de pesos $\mathbf{w}$, perpendicular a la frontera
- **Puntos azules**: datos de clase 0
- **Puntos rojos**: datos de clase 1
- **Halo dorado pulsante**: puntos mal clasificados
- **Cruz amarilla**: marca sobre puntos con predicción incorrecta
- **Badge verde "CONVERGIDO"**: aparece cuando accuracy = 100%
- **Aviso rojo**: aparece tras 200+ epochs sin convergencia

### 3.3 El header

Muestra en tiempo real: número de epoch actual, accuracy (%) y error rate.

### 3.4 Ecuación en el canvas

En la esquina superior izquierda del canvas se muestra la ecuación actual de la frontera en forma numérica: por ejemplo, `0.45x₁ + 0.72x₂ - 0.31 = 0`.

---

## 4. Controles Interactivos

### 4.1 Dataset

**Selector de clase**: dos botones para elegir qué clase se asignará al siguiente punto que coloques — Clase 0 (azul) o Clase 1 (rojo).

**Presets de datos**: menú desplegable con configuraciones predefinidas:

| Preset | Descripción | ¿Separable? |
|--------|-------------|-------------|
| **Lineal** | Puntos separados por $y > 0.3x + 0.1$ | Sí |
| **Diagonal** | Puntos separados por $x + y > 0$ | Sí |
| **Vertical** | Puntos separados por $x > 0.2$ | Sí |
| **Clusters** | Dos grupos compactos en esquinas opuestas | Sí |
| **XOR** | Patrón exclusivo-or: $(x>0) \neq (y>0)$ | **No** |
| **Limpiar** | Elimina todos los puntos | — |

**Interacción con el mouse**:

- **Click izquierdo**: añadir un punto de la clase seleccionada
- **Arrastrar**: mover un punto existente
- **Click derecho**: eliminar el punto más cercano
- **Hover**: muestra tooltip con coordenadas, clase real, predicción y salida cruda

### 4.2 Parámetros

| Control | Rango | Default | Efecto |
|---------|-------|---------|--------|
| Learning rate ($\eta$) | 0.001 – 1.0 | 0.100 | Magnitud de los ajustes de peso por paso |
| $w_1$ | -3.0 – 3.0 | 0.00 | Peso de la entrada $x_1$ |
| $w_2$ | -3.0 – 3.0 | 0.00 | Peso de la entrada $x_2$ |
| bias ($b$) | -3.0 – 3.0 | 0.00 | Desplazamiento de la frontera |

Los sliders de $w_1$, $w_2$ y $b$ se actualizan en tiempo real durante el entrenamiento. También puedes moverlos manualmente para experimentar con diferentes configuraciones de pesos.

### 4.3 Entrenamiento

| Botón | Acción |
|-------|--------|
| **Entrenar** | Inicia/detiene el entrenamiento automático |
| **1 Paso** | Ejecuta una actualización sobre un punto aleatorio |
| **1 Epoch** | Recorre todos los puntos una vez (orden aleatorio) |
| **Reset** | Reinicia los pesos a valores aleatorios pequeños |

**Velocidad**: slider de 1 a 50 epochs por segundo.

**Gráfica de loss**: miniatura que muestra el historial del error (rojo) y accuracy (verde) durante las últimas 200 epochs.

### 4.4 Audio

| Control | Descripción |
|---------|-------------|
| **Audio ON/OFF** | Activa o desactiva la sonificación |
| **Volumen** | 0% – 100% |
| **Forma de onda** | Sine, Triangle, Square, Sawtooth |

### 4.5 Valores en tiempo real

Panel de estadísticas que muestra: Epoch, Accuracy, Error rate, $w_1$, $w_2$, bias, $\|\mathbf{w}\|$ (norma del vector de pesos), número total de puntos, y número de puntos mal clasificados.

---

## 5. Las Matemáticas

### 5.1 Función de decisión

El perceptrón computa una **combinación lineal** de las entradas:

$$z = w_1 x_1 + w_2 x_2 + b = \mathbf{w}^T \mathbf{x} + b$$

La predicción se obtiene aplicando la función escalón (sign):

$$\hat{y} = \begin{cases} 1 & \text{si } z \geq 0 \\ 0 & \text{si } z < 0 \end{cases}$$

### 5.2 Regla de aprendizaje

Para cada punto $(x_1^{(i)}, x_2^{(i)})$ con etiqueta $t^{(i)}$ y predicción $\hat{y}^{(i)}$, los pesos se actualizan según:

$$w_j \leftarrow w_j + \eta \cdot (t^{(i)} - \hat{y}^{(i)}) \cdot x_j^{(i)}$$

$$b \leftarrow b + \eta \cdot (t^{(i)} - \hat{y}^{(i)})$$

donde $\eta$ es el **learning rate** (tasa de aprendizaje). Observa que:

- Si la predicción es correcta ($t = \hat{y}$), el factor $(t - \hat{y}) = 0$ y no hay actualización
- Si predice 0 cuando debería ser 1, $(t - \hat{y}) = 1$: los pesos se incrementan en la dirección del punto
- Si predice 1 cuando debería ser 0, $(t - \hat{y}) = -1$: los pesos se decrementan

### 5.3 Geometría de la frontera

La frontera de decisión es la recta definida por:

$$w_1 x_1 + w_2 x_2 + b = 0$$

Se puede reescribir en forma explícita (cuando $w_2 \neq 0$):

$$x_2 = -\frac{w_1}{w_2} x_1 - \frac{b}{w_2}$$

La **pendiente** es $-w_1/w_2$ y la **ordenada al origen** es $-b/w_2$.

La **distancia** del origen a la frontera es:

$$d = \frac{|b|}{\|\mathbf{w}\|} = \frac{|b|}{\sqrt{w_1^2 + w_2^2}}$$

### 5.4 Teorema de convergencia del perceptrón

**Teorema** (Rosenblatt, 1962; Novikoff, 1963): Si el conjunto de datos es linealmente separable con margen $\gamma > 0$ y todos los puntos satisfacen $\|\mathbf{x}\| \leq R$, entonces el algoritmo del perceptrón converge en a lo sumo:

$$k \leq \left(\frac{R}{\gamma}\right)^2$$

actualizaciones, donde $\gamma$ es el margen geométrico máximo (la mayor distancia mínima entre los puntos y cualquier hiperplano separador).

Este teorema garantiza que si una solución existe, el perceptrón *la encontrará* en un número finito de pasos. Sin embargo, **no dice nada** sobre cuántos epochs serán necesarios — solo acota el número total de actualizaciones de peso.

### 5.5 La limitación XOR

El XOR (o exclusivo) asigna clase 1 cuando exactamente una de las dos entradas es positiva:

| $x_1 > 0$ | $x_2 > 0$ | XOR |
|:----------:|:----------:|:---:|
| No | No | 0 |
| Sí | No | 1 |
| No | Sí | 1 |
| Sí | Sí | 0 |

No existe ninguna línea recta que separe las clases 0 y 1 del XOR. Esto se puede demostrar formalmente: para separar (0,0) y (1,1) de (1,0) y (0,1), necesitaríamos simultáneamente $w_1 + w_2 + b < 0$ y $w_1 + b \geq 0$ y $w_2 + b \geq 0$ y $b < 0$, lo cual es una contradicción.

Minsky y Papert demostraron esta limitación formalmente en *Perceptrons* (1969), lo que contribuyó al primer "invierno de la IA".

### 5.6 Visualización de la región de decisión

La simulación muestra un degradado continuo en lugar de una frontera binaria abrupta. Internamente se computa una **sigmoide suavizada** sobre la salida cruda:

$$\sigma(z) = \frac{1}{1 + e^{-4z}}$$

El factor 4 comprime la transición. Los colores interpolan entre azul (clase 0, $\sigma \approx 0$) y rojo (clase 1, $\sigma \approx 1$), haciendo visible la "confianza" del clasificador en cada región del plano.

---

## 6. Sonificación

La simulación traduce el estado del entrenamiento a sonido, creando un canal sensorial adicional para comprender el aprendizaje.

### 6.1 Mapeo error → intervalo musical

| Error | Sonido | Intervalo | Significado |
|-------|--------|-----------|-------------|
| $> 50\%$ | Disonante | Tritono (C–F#) | El perceptrón clasifica peor que el azar |
| $20\%$–$50\%$ | Tenso | Tercera menor (C–Eb) | Mejorando, pero con errores significativos |
| $1\%$–$20\%$ | Armónico | Quinta justa (C–G) | Casi convergido |
| $0\%$ | Consonante | Acorde mayor (C–E–G–C') | **Convergencia total** |

La nota base es $C_4 = 261.63$ Hz. Los intervalos se calculan con temperamento igual: semitono = $2^{1/12}$.

### 6.2 Otros sonidos

- **Tick de actualización** (880 Hz, 40ms): suena con cada ajuste de peso
- **Nota grave de error** (110 Hz, 120ms): punto mal clasificado durante paso individual
- **Acorde de convergencia** (C mayor completo, 800ms): se reproduce una sola vez al alcanzar 100% accuracy
- **Click de punto nuevo**: $C_5$ (523 Hz) para clase 0, $E_4$ (330 Hz) para clase 1

### 6.3 Formas de onda

| Forma | Carácter |
|-------|----------|
| **Sine** | Suave, puro, fundamental |
| **Triangle** | Ligeramente brillante, armónicos impares |
| **Square** | Hueco, robótico, rico en armónicos impares |
| **Sawtooth** | Brillante, zumbante, todos los armónicos |

Cada forma de onda pasa por un filtro lowpass (frecuencia de corte = $4f$, $Q = 0.7$) para evitar sonidos estridentes.

---

## 7. Guía Paso a Paso

### Experimento 1: Clasificación lineal básica

1. **Carga el preset "Lineal"** desde el menú desplegable
2. Observa los 40 puntos distribuidos con la frontera teórica $y = 0.3x + 0.1$
3. Los pesos se inicializan aleatoriamente — la línea amarilla estará en una posición arbitraria
4. **Haz click en "Entrenar"** y observa cómo la frontera se ajusta epoch tras epoch
5. Nota cómo los halos dorados (puntos mal clasificados) van desapareciendo
6. Cuando el badge "CONVERGIDO" aparezca, el perceptrón habrá encontrado una frontera válida
7. Observa los valores finales de $w_1$, $w_2$ y $b$ en el panel de estadísticas

### Experimento 2: Efecto del learning rate

1. Carga el preset "Diagonal"
2. Fija $\eta = 0.001$ (mínimo) y entrena. Observa cómo la convergencia es **lenta pero estable**
3. Reset y fija $\eta = 1.0$ (máximo). La frontera **salta violentamente** entre posiciones
4. Prueba $\eta = 0.1$ (valor medio): buen balance entre velocidad y estabilidad
5. Compara las curvas de loss en el mini-gráfico para cada valor

### Experimento 3: La imposibilidad del XOR

1. **Carga el preset "XOR"**
2. Inicia el entrenamiento automático
3. Observa cómo la frontera oscila sin encontrar una posición estable
4. Después de 200 epochs, aparecerá el aviso rojo: "¿No linealmente separable?"
5. El accuracy nunca superará ~75% (solo puede clasificar correctamente 3 de los 4 cuadrantes)
6. **Esto no es un bug** — es una limitación fundamental del perceptrón de una sola capa

### Experimento 4: Construcción manual de la frontera

1. Selecciona "Limpiar todo" para empezar en blanco
2. Coloca 5 puntos azules (clase 0) en la parte inferior y 5 rojos (clase 1) en la superior
3. **Sin entrenar**, mueve manualmente los sliders de $w_1$, $w_2$ y $b$
4. Intenta posicionar la frontera amarilla de forma que separe los dos grupos
5. Observa cómo $w_2$ controla la inclinación y $b$ el desplazamiento vertical
6. Cuando lo hayas logrado manualmente, verifica con "Entrenar" que el algoritmo llega a una solución similar

### Experimento 5: Sensibilidad a outliers

1. Carga el preset "Clusters"
2. Entrena hasta convergencia
3. Ahora coloca un punto rojo (clase 1) **dentro** del cluster azul
4. Observa cómo la frontera se reajusta drásticamente para intentar clasificar correctamente ese único punto
5. Si el punto está muy profundo dentro del cluster opuesto, puede causar que otros puntos se clasifiquen mal
6. Esto ilustra la **sensibilidad** del perceptrón a datos ruidosos o anómalos

---

## 8. Conceptos Avanzados

### 8.1 Del perceptrón al MLP

La limitación del XOR se resuelve añadiendo **capas ocultas**. Un perceptrón multicapa (MLP) con una capa oculta de 2 neuronas puede resolver XOR:

$$h_1 = \text{sign}(x_1 + x_2 - 0.5), \quad h_2 = \text{sign}(x_1 + x_2 - 1.5)$$
$$y = \text{sign}(h_1 - h_2 - 0.5)$$

La primera neurona oculta detecta "al menos uno positivo" y la segunda "ambos positivos". La neurona de salida computa la diferencia: XOR.

### 8.2 Margen y SVM

El perceptrón encuentra *alguna* frontera separadora, pero no necesariamente la mejor. Las **máquinas de vectores de soporte** (SVM) buscan la frontera con el **margen máximo** — la mayor distancia entre la frontera y los puntos más cercanos de cada clase.

### 8.3 Regla delta vs. regla del perceptrón

La regla del perceptrón usa la predicción discreta ($\hat{y} \in \{0, 1\}$). La **regla delta** (Widrow-Hoff) usa la salida cruda continua $z$:

$$w_j \leftarrow w_j + \eta \cdot (t^{(i)} - z^{(i)}) \cdot x_j^{(i)}$$

La regla delta minimiza el error cuadrático medio y converge incluso para datos no separables (al mejor hiperplano en sentido de mínimos cuadrados).

### 8.4 Contexto histórico

- **1943**: McCulloch y Pitts proponen el modelo de neurona artificial
- **1958**: Rosenblatt construye el Mark I Perceptron en el Cornell Aeronautical Laboratory
- **1960**: Widrow y Hoff desarrollan ADALINE con la regla delta
- **1969**: Minsky y Papert publican *Perceptrons*, demostrando limitaciones fundamentales
- **1969–1986**: "Invierno de la IA" — reducción drástica en la investigación de redes neuronales
- **1986**: Rumelhart, Hinton y Williams popularizan la retropropagación, superando la limitación de una sola capa

### 8.5 Interpretación geométrica de la actualización

Cuando un punto $\mathbf{x}^{(i)}$ es mal clasificado, la actualización $\mathbf{w} \leftarrow \mathbf{w} + \eta(t - \hat{y})\mathbf{x}^{(i)}$ tiene un efecto geométrico claro:

- Si $t = 1$ y $\hat{y} = 0$: el vector de pesos **rota hacia** el punto, incluyéndolo en la región positiva
- Si $t = 0$ y $\hat{y} = 1$: el vector de pesos **rota alejándose** del punto, excluyéndolo de la región positiva

La magnitud de la rotación es proporcional a $\eta$ y a la distancia del punto al origen.

---

## 9. Ejercicios

### Ejercicio 1: Frontera vertical

Configura manualmente los pesos para que la frontera de decisión sea una línea **perfectamente vertical** en $x_1 = 0$. ¿Qué valores de $w_1$, $w_2$ y $b$ necesitas? Verifica tu respuesta cargando el preset "Vertical" y comparando con la solución del entrenamiento automático.

*Pista*: si $w_2 = 0$, la ecuación se reduce a $w_1 x_1 + b = 0$.

### Ejercicio 2: Velocidad de convergencia

Carga el preset "Clusters" tres veces, usando $\eta = 0.01$, $\eta = 0.1$ y $\eta = 0.5$. Para cada caso, anota:
- Número de epochs hasta convergencia
- Forma de la curva de loss (suave vs. errática)
- Posición final de la frontera

¿Qué valor de $\eta$ da la convergencia más rápida? ¿Y la más estable?

### Ejercicio 3: Capacidad máxima

Empieza con un plano vacío. Añade puntos alternando entre clase 0 y clase 1, cada vez más cerca entre sí. ¿Cuántos puntos puedes añadir antes de que el perceptrón ya no pueda separarlos? ¿Depende de la distribución espacial?

### Ejercicio 4: Demostración geométrica del XOR

Carga el preset XOR. En una hoja de papel, dibuja los cuatro cuadrantes con sus clases asignadas. Intenta trazar *una sola línea recta* que separe los puntos rojos de los azules. Explica formalmente por qué es imposible usando un sistema de inecuaciones.

### Ejercicio 5: Peso como dirección

Con el preset "Diagonal" convergido, observa la dirección del vector de pesos (flecha punteada). ¿Es perpendicular a la frontera? Mide los ángulos visualmente. Ahora calcula: si $w_1 = a$ y $w_2 = b$, ¿cuál es el ángulo que forma $\mathbf{w}$ con el eje $x_1$? Usa $\theta = \arctan(w_2/w_1)$ y verifica que coincide con lo que ves.

### Ejercicio 6: Reconstrucción de la ecuación

Entrena el perceptrón con el preset "Lineal" hasta convergencia. Lee los valores de $w_1$, $w_2$ y $b$ del panel de estadísticas. Escribe la ecuación de la frontera en forma $x_2 = mx_1 + n$. ¿Es similar a la ecuación original del preset ($x_2 = 0.3x_1 + 0.1$)? ¿Por qué puede ser diferente?

---

## 10. Glosario

| Término | Definición |
|---------|------------|
| **Perceptrón** | Neurona artificial que computa $y = \text{sign}(\mathbf{w}^T\mathbf{x} + b)$ y aprende mediante actualizaciones de pesos |
| **Peso** ($w_i$) | Coeficiente que multiplica cada entrada, determinando su importancia relativa |
| **Sesgo** (bias, $b$) | Término independiente que desplaza la frontera de decisión respecto al origen |
| **Frontera de decisión** | Recta (2D) o hiperplano (nD) donde $\mathbf{w}^T\mathbf{x} + b = 0$; separa las regiones de clase |
| **Learning rate** ($\eta$) | Tasa de aprendizaje que controla la magnitud de cada actualización de pesos |
| **Epoch** | Una pasada completa por todos los puntos del dataset de entrenamiento |
| **Accuracy** | Fracción de puntos correctamente clasificados: $\text{acc} = \frac{\text{correctos}}{N}$ |
| **Error rate** | Complemento del accuracy: $1 - \text{acc}$ |
| **Convergencia** | Estado donde todos los puntos se clasifican correctamente (accuracy = 100%) |
| **Separabilidad lineal** | Propiedad de un dataset donde existe un hiperplano que separa perfectamente las clases |
| **XOR** | Función lógica "o exclusivo"; ejemplo canónico de problema no linealmente separable |
| **Función escalón** (sign) | Función de activación que devuelve 1 si $z \geq 0$ y 0 si $z < 0$ |
| **Vector de pesos** ($\mathbf{w}$) | Vector $(w_1, w_2)$ que determina la orientación de la frontera; perpendicular a ella |
| **Combinación lineal** | Expresión $\sum_i w_i x_i + b$; la operación fundamental del perceptrón |
| **Margen** ($\gamma$) | Distancia mínima entre la frontera de decisión y el punto más cercano del dataset |
| **Regla del perceptrón** | Algoritmo de actualización: $w_i \leftarrow w_i + \eta(t-\hat{y})x_i$ |
| **Disonancia** | En la sonificación, intervalos musicales tensos que representan error alto |
| **Consonancia** | En la sonificación, intervalos armónicos que representan error bajo o convergencia |

---

## 11. Referencias

1. **Rosenblatt, F.** (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain." *Psychological Review*, 65(6), 386–408.

2. **Minsky, M. & Papert, S.** (1969). *Perceptrons: An Introduction to Computational Geometry*. MIT Press.

3. **Novikoff, A. B. J.** (1963). "On Convergence Proofs for Perceptrons." *Proceedings of the Symposium on the Mathematical Theory of Automata*, 12, 615–622.

4. **Widrow, B. & Hoff, M. E.** (1960). "Adaptive Switching Circuits." *IRE WESCON Convention Record*, Part 4, 96–104.

5. **Rumelhart, D. E., Hinton, G. E. & Williams, R. J.** (1986). "Learning Representations by Back-propagating Errors." *Nature*, 323(6088), 533–536.

6. **McCulloch, W. S. & Pitts, W.** (1943). "A Logical Calculus of Ideas Immanent in Nervous Activity." *Bulletin of Mathematical Biophysics*, 5(4), 115–133.

7. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*, Chapter 4. Springer.

8. **Goodfellow, I., Bengio, Y. & Courville, A.** (2016). *Deep Learning*, Chapter 6. MIT Press. Disponible en [deeplearningbook.org](https://www.deeplearningbook.org/).
