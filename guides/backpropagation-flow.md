# Backpropagation Flow — Visualización del Flujo de Gradientes

> Simulación interactiva de la retropropagación en redes neuronales multicapa. Observa en tiempo real cómo la información fluye hacia adelante (forward pass) y los gradientes fluyen hacia atrás (backward pass), coloreando cada conexión y neurona según su contribución al error. Diagnostica vanishing y exploding gradients, congela capas, y experimenta con activaciones.

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

La **retropropagación** (backpropagation) es el algoritmo que permite que las redes neuronales aprendan. Publicado por Rumelhart, Hinton y Williams en 1986, resolvió el problema que había paralizado el campo desde 1969: ¿cómo ajustar los pesos de las capas ocultas, que no tienen una señal directa de error?

La respuesta es la **regla de la cadena** del cálculo diferencial. El error en la salida se propaga hacia atrás capa por capa, asignando a cada peso una "responsabilidad" proporcional a su contribución al error. Esta simulación hace visible ese flujo invisible: partículas verdes fluyen hacia adelante (activaciones) y partículas naranjas fluyen hacia atrás (gradientes).

Puedes construir redes de diferentes arquitecturas, entrenarlas en tres problemas de clasificación (XOR, círculo, espiral), y observar fenómenos como el desvanecimiento de gradientes en capas profundas con sigmoid.

---

## 2. Conceptos Fundamentales

### 2.1 Forward pass

En el **forward pass**, la información fluye de la capa de entrada a la salida. Para cada neurona $j$ en la capa $l$:

1. Se calcula la **pre-activación**: $z_j^{(l)} = \sum_i w_{ij}^{(l)} a_i^{(l-1)} + b_j^{(l)}$
2. Se aplica la **función de activación**: $a_j^{(l)} = \sigma(z_j^{(l)})$

### 2.2 Backward pass

En el **backward pass**, los gradientes fluyen de la salida a la entrada. Para cada peso $w_{ij}^{(l)}$, el gradiente se descompone en tres factores:

$$\frac{\partial L}{\partial w_{ij}^{(l)}} = \frac{\partial L}{\partial a_j^{(l)}} \cdot \sigma'(z_j^{(l)}) \cdot a_i^{(l-1)}$$

- $\partial L / \partial a_j^{(l)}$: cómo la loss cambia con la activación (propagado desde capas posteriores)
- $\sigma'(z_j^{(l)})$: derivada de la activación (sensibilidad de la neurona)
- $a_i^{(l-1)}$: activación de la neurona de origen (señal entrante)

### 2.3 La regla de la cadena

El gradiente se transmite entre capas multiplicando factores. Para una red de $L$ capas, el gradiente respecto a un peso en la capa 1 involucra un producto de $L$ derivadas — una por cada capa que el error debe atravesar. Este producto puede crecer o encogerse exponencialmente.

### 2.4 Actualización de pesos

Cada peso se actualiza proporcionalmente a su gradiente:

$$w_{ij} \leftarrow w_{ij} - \eta \frac{\partial L}{\partial w_{ij}}$$

---

## 3. La Interfaz

### 3.1 Estructura general

| Área | Ubicación | Función |
|------|-----------|---------|
| **Canvas principal** | Izquierda | Diagrama de red con neuronas, conexiones y partículas de flujo |
| **Tooltip** | Sobre neurona hovereada | Muestra activación, pre-activación, gradiente |
| **Badge de fase** | Esquina inf-izq del canvas | Indica "Forward Pass" (verde) o "Backward Pass" (naranja) |
| **Panel de controles** | Derecha (360px) | Ecuación, problema, arquitectura, controles, métricas |

### 3.2 El diagrama de red

La visualización principal muestra la red neuronal como un grafo:

- **Neuronas** (círculos): brillo proporcional a la activación. Un valor alto = neurona brillante
- **Conexiones** (curvas Bézier): azul = peso positivo, rojo = peso negativo. Grosor proporcional a $|w|$
- **Barras de gradiente**: debajo de cada neurona, muestran la magnitud del gradiente. Verde = saludable, amarillo = vanishing, rojo = exploding
- **Etiquetas de capa**: Input, Hidden 1, Hidden 2, Output
- **Partículas de flujo**: puntos verdes (forward) y naranjas (backward) que recorren las conexiones

### 3.3 Preview de región de decisión

Miniatura en el panel derecho que muestra la frontera de decisión actual de la red: azul = clase 0, rojo = clase 1, con los puntos del dataset superpuestos. Se actualiza en cada epoch.

### 3.4 Panel de métricas

- **Loss (MSE)**: error cuadrático medio
- **Accuracy**: porcentaje de clasificaciones correctas
- **Max/Min gradiente**: rango de gradientes en toda la red
- **Diagnóstico**: HEALTHY (verde), VANISHING (amarillo) o EXPLODING (rojo)
- **Gráfica de loss**: historial de las últimas 300 epochs
- **Barras de gradiente por capa**: visualización rápida de la distribución de gradientes

---

## 4. Controles Interactivos

### 4.1 Problema

| Problema | Datos | Complejidad |
|----------|-------|-------------|
| **XOR** | 4 puntos: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0 | Mínima pero no lineal |
| **Círculo** | 50 puntos: clase 1 si $x^2+y^2 < 0.5$ | Frontera circular |
| **Espiral** | 120 puntos: dos espirales entrelazadas | Muy compleja |

### 4.2 Arquitectura

Menú desplegable con cuatro opciones:

| Arquitectura | Capas | Parámetros aprox. |
|-------------|-------|-------------------|
| 2→4→4→1 | 2 ocultas | ~37 |
| 2→8→8→1 | 2 ocultas | ~97 |
| 2→4→4→4→1 | 3 ocultas | ~53 |
| 2→3→3→1 (minimal) | 2 ocultas | ~22 |

### 4.3 Hiperparámetros

| Control | Rango | Default | Descripción |
|---------|-------|---------|-------------|
| Learning rate ($\eta$) | 0.01 – 2.0 | 0.50 | Tamaño del paso de actualización |
| Activación | sigmoid, ReLU, tanh | sigmoid | Función de activación para capas ocultas |

La capa de salida siempre usa sigmoid (para producir valores en $[0,1]$).

### 4.4 Congelar capas

Botones L1, L2, L3... permiten **congelar** capas individuales. Una capa congelada:
- No actualiza sus pesos durante el backward pass
- Se muestra con borde azul en las neuronas
- Tiene un indicador ❄ en el canvas

Esto permite simular **transfer learning** y observar el efecto de entrenar solo subconjuntos de la red.

### 4.5 Control de entrenamiento

| Botón | Acción |
|-------|--------|
| **Entrenar/Detener** | Inicia o pausa el entrenamiento continuo |
| **Paso animado** | Ejecuta un forward+backward con partículas de flujo visibles (~3.5s) |
| **10 Epochs** | Ejecuta 10 epochs rápidamente |
| **Reset** | Reinicializa los pesos con Xavier/Glorot: $w \sim \mathcal{N}(0, \sqrt{2/(n_{in}+n_{out})})$ |

**Velocidad**: 1 a 100 epochs por segundo.

### 4.6 Interacción con el canvas

- **Hover** sobre una neurona: tooltip con activación, pre-activación y gradiente
- **Click** en una neurona: selecciona y muestra panel detallado persistente con valores numéricos

---

## 5. Las Matemáticas

### 5.1 Funciones de activación

La simulación ofrece tres funciones de activación para las capas ocultas:

**Sigmoid:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1 - \sigma(z))$$

Rango: $(0, 1)$. Derivada máxima: $0.25$ (en $z = 0$). Causa vanishing gradients porque $\sigma' < 0.25$ siempre.

**ReLU:**
$$\text{ReLU}(z) = \max(0, z), \quad \text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}$$

Rango: $[0, \infty)$. Derivada: 0 o 1. No causa vanishing gradients en neuronas activas, pero las "neuronas muertas" (salida siempre 0) no reciben gradiente.

**Tanh:**
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad \tanh'(z) = 1 - \tanh^2(z)$$

Rango: $(-1, 1)$. Derivada máxima: $1$ (en $z = 0$). Mejor que sigmoid para vanishing gradients, pero aún puede ocurrir para $|z|$ grande.

### 5.2 Función de pérdida

La simulación usa **MSE** (Mean Squared Error):

$$L = \frac{1}{N}\sum_{n=1}^{N}(\hat{y}^{(n)} - y^{(n)})^2$$

Para la neurona de salida (sigmoid), la derivada del error es:

$$\frac{\partial L}{\partial a_j^{(L)}} = 2(a_j^{(L)} - y_j) \cdot \sigma'(z_j^{(L)})$$

### 5.3 Propagación de deltas

El algoritmo define un "delta" $\delta_j^{(l)}$ para cada neurona:

$$\delta_j^{(L)} = (a_j^{(L)} - y_j) \cdot \sigma'(z_j^{(L)}) \quad \text{(capa de salida)}$$

$$\delta_j^{(l)} = \left(\sum_k w_{jk}^{(l+1)} \delta_k^{(l+1)}\right) \cdot \sigma'(z_j^{(l)}) \quad \text{(capas ocultas)}$$

Los gradientes de los pesos se obtienen directamente:

$$\frac{\partial L}{\partial w_{ij}^{(l)}} = \delta_j^{(l)} \cdot a_i^{(l-1)}$$

### 5.4 Inicialización Xavier/Glorot

La red se inicializa con pesos muestreados de:

$$w \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$$

Esta escala mantiene las varianzas de activaciones y gradientes aproximadamente constantes a lo largo de las capas, previniendo saturación temprana.

### 5.5 Diagnóstico de gradientes

La simulación clasifica el estado de los gradientes:

- **HEALTHY**: todos los gradientes en rango razonable
- **VANISHING**: gradiente mínimo $< 10^{-6}$ después de 20 epochs
- **EXPLODING**: gradiente máximo $> 10$

---

## 6. Sonificación

### 6.1 Forward pass

Cada capa produce una nota ascendente durante el forward pass animado:

$$f_l = 440 \cdot 2^{(48 + 5l - 69)/12} \text{ Hz}$$

Las capas tempranas suenan más graves, las tardías más agudas — representando el flujo de información de entrada a salida.

### 6.2 Backward pass

Las notas descienden durante el backward pass:

$$f_l = 440 \cdot 2^{(72 - 5l - 69)/12} \text{ Hz}$$

El volumen es proporcional a la magnitud media del gradiente de la capa. Las capas con gradientes desvanecidos suenan más débiles — puedes *escuchar* el vanishing gradient como un silencio progresivo en las capas profundas.

### 6.3 Throttling

Los sonidos se limitan a un mínimo de 50ms entre notas para evitar saturación.

---

## 7. Guía Paso a Paso

### Experimento 1: Forward y backward animados

1. Mantén la configuración por defecto: XOR, 2→4→4→1, sigmoid
2. Haz click en **"Paso animado"**
3. Observa las partículas verdes fluyendo de izquierda a derecha (forward pass)
4. Tras ~1.5s, las partículas naranjas fluyen de derecha a izquierda (backward pass)
5. Nota cómo las conexiones más gruesas tienen más partículas
6. Repite varias veces y observa los valores de activación cambiar en cada neurona

### Experimento 2: Entrenamiento del XOR

1. Con la configuración por defecto, haz click en **"Entrenar"**
2. Observa la gráfica de loss descender
3. La preview de decisión mostrará gradualmente las cuatro regiones del XOR
4. El loss debería bajar de ~0.25 a <0.01 en ~200-500 epochs
5. Nota cómo las barras de gradiente se hacen más pequeñas a medida que el loss decrece

### Experimento 3: Vanishing gradients con sigmoid

1. Selecciona arquitectura **2→4→4→4→1** (3 capas ocultas)
2. Mantén activación **sigmoid**
3. Entrena durante 50+ epochs
4. Observa las barras de gradiente: L1 (primera capa) será mucho más pequeña que L3
5. El diagnóstico debería mostrar **VANISHING** en amarillo
6. Las neuronas de las primeras capas apenas cambian — los gradientes se multiplican por $\sigma'(z) < 0.25$ en cada capa

### Experimento 4: ReLU al rescate

1. Desde el estado anterior, cambia la activación a **ReLU**
2. Entrena de nuevo
3. Las barras de gradiente ahora deberían ser más uniformes entre capas
4. El diagnóstico debería mostrar **HEALTHY** en verde
5. El loss desciende más rápido — ReLU permite que los gradientes fluyan sin atenuarse

### Experimento 5: Congelar y descongelar capas

1. Problema: Círculo, arquitectura 2→4→4→1
2. Entrena hasta que el loss baje significativamente (~100 epochs)
3. **Congela L1** (primera capa oculta) haciendo click en el botón
4. Continúa entrenando — solo L2 se actualiza
5. ¿El loss sigue bajando? ¿Más lento?
6. Descongela L1 y congela L2. Entrena. ¿Cuál capa es más importante?

### Experimento 6: La espiral

1. Selecciona problema **Espiral** y arquitectura **2→8→8→1**
2. Usa activación **ReLU** y learning rate **0.5**
3. Entrena por 500+ epochs
4. Observa la preview de decisión: ¿logra separar las dos espirales?
5. Prueba con **tanh** — ¿es mejor o peor que ReLU para este problema?
6. Prueba la arquitectura **2→4→4→4→1** — ¿la profundidad ayuda con las espirales?

---

## 8. Conceptos Avanzados

### 8.1 ¿Por qué sigmoid causa vanishing gradients?

La derivada de sigmoid es $\sigma'(z) = \sigma(z)(1-\sigma(z))$, con máximo $0.25$ en $z=0$. En una red de $L$ capas, el gradiente respecto a la primera capa incluye un producto:

$$\prod_{l=1}^{L} \sigma'(z^{(l)}) \leq 0.25^L$$

Para $L = 4$: $0.25^4 = 0.0039$. El gradiente de la primera capa es ~256 veces menor que el de la última.

### 8.2 Batch vs. online training

La simulación usa **online training** (actualiza pesos después de cada muestra, en orden aleatorio). En la práctica, se usa **mini-batch training**:

- **Online** (batch=1): ruidoso, actualizaciones frecuentes
- **Mini-batch** (batch=32-256): balance entre ruido y eficiencia
- **Full batch** (batch=N): gradiente exacto, pero lento por epoch

### 8.3 Transfer learning y fine-tuning

Congelar capas simula el concepto de **transfer learning**: las primeras capas aprenden features genéricos (bordes, texturas) que se reutilizan, y solo las últimas capas se reentrenan para la tarea específica. En esta simulación, puedes experimentar con qué capas son más importantes para cada problema.

### 8.4 Regla de la cadena computacional

Backpropagation es una aplicación eficiente de la regla de la cadena que evita recalcular derivadas parciales redundantes. Recorre el grafo computacional una sola vez de salida a entrada, almacenando los resultados intermedios. La complejidad es $O(W)$ donde $W$ es el número de pesos — el mismo orden que el forward pass.

### 8.5 Modos de fallo

- **Gradientes desvanecidos**: capas profundas + sigmoid. Síntoma: primeras capas no aprenden
- **Gradientes explosivos**: learning rate alto + pesos grandes. Síntoma: loss explota a NaN
- **Neuronas muertas**: ReLU con gradientes negativos persistentes. Síntoma: neurona siempre outputs 0
- **Pérdida estancada**: red demasiado pequeña para el problema. Síntoma: loss no baja de cierto umbral

---

## 9. Ejercicios

### Ejercicio 1: Contando multiplicaciones

Para la arquitectura 2→4→4→1, calcula manualmente cuántas multiplicaciones se necesitan en un forward pass y en un backward pass para una muestra. ¿Son iguales? Verifica contando las conexiones en el diagrama.

### Ejercicio 2: Sigmoid vs. ReLU en profundidad

Entrena la red 2→4→4→4→1 en el problema Círculo durante exactamente 200 epochs con sigmoid, y luego repite con ReLU (mismo learning rate, reset entre ambos). Compara:
- Loss final
- Barras de gradiente de L1
- ¿Cuántas epochs tardó cada una en bajar de loss = 0.1?

### Ejercicio 3: La arquitectura mínima para XOR

El XOR requiere mínimo 2 neuronas ocultas para resolverse. Usa la arquitectura **2→3→3→1** y entrena. ¿Converge siempre? ¿Qué ocurre si reduces a 2→2→1? Intenta múltiples resets.

### Ejercicio 4: Congelación selectiva

Con el problema Espiral y arquitectura 2→8→8→1:
1. Entrena 100 epochs completo
2. Congela L1, entrena 100 más
3. Reset, entrena 100 epochs con L2 congelada desde el inicio
¿En qué caso el rendimiento final es peor? ¿Qué capa es más crítica para las features de bajo nivel?

### Ejercicio 5: Diagnosticando la explosión

Fija learning rate $\eta = 2.0$ y usa sigmoid en 2→4→4→1 con XOR. Entrena y observa. ¿Cuántos epochs hasta que aparezca el diagnóstico EXPLODING? ¿Qué le ocurre al loss? Baja $\eta$ a 1.0 y repite. ¿Hay un umbral de $\eta$ donde el entrenamiento pasa de estable a inestable?

### Ejercicio 6: Visualizando la regla de la cadena

Usa "Paso animado" en la red 2→4→4→1 con XOR. Después de un paso, haz click en cada neurona de la capa oculta 1 y anota su gradiente. Haz lo mismo para la capa oculta 2. ¿Los gradientes de la capa 1 son consistentemente menores que los de la capa 2? Calcula el ratio medio. ¿Es cercano a 0.25 (la derivada máxima de sigmoid)?

---

## 10. Glosario

| Término | Definición |
|---------|------------|
| **Backpropagation** | Algoritmo que calcula gradientes de la loss respecto a cada peso usando la regla de la cadena |
| **Forward pass** | Fase donde la entrada se transforma capa por capa hasta producir la salida |
| **Backward pass** | Fase donde los gradientes se propagan de la salida hacia la entrada |
| **Gradiente** ($\partial L/\partial w$) | Derivada parcial de la loss respecto a un peso; indica dirección y magnitud de ajuste |
| **Regla de la cadena** | $\frac{\partial f}{\partial x} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x}$ — permite componer derivadas |
| **Delta** ($\delta_j$) | Señal de error local de una neurona; producto del error propagado y la derivada de activación |
| **Pre-activación** ($z$) | Suma ponderada antes de aplicar la función de activación: $z = \sum w_i a_i + b$ |
| **Activación** ($a$) | Salida de la neurona después de aplicar la función de activación: $a = \sigma(z)$ |
| **Vanishing gradients** | Problema donde los gradientes se encogen exponencialmente en capas profundas |
| **Exploding gradients** | Problema donde los gradientes crecen sin control, desestabilizando el entrenamiento |
| **Neurona muerta** | Neurona con ReLU que siempre produce 0 y no recibe gradiente |
| **Xavier/Glorot** | Esquema de inicialización de pesos que mantiene varianzas constantes entre capas |
| **MSE** | Mean Squared Error: $\frac{1}{N}\sum(y - \hat{y})^2$ |
| **Congelar capa** | Impedir la actualización de pesos de una capa durante el entrenamiento |
| **Transfer learning** | Técnica que reutiliza pesos pre-entrenados, congelando las primeras capas |
| **Derivada de activación** ($\sigma'$) | Factor multiplicativo en la regla de la cadena; controla cuánto gradiente pasa por la neurona |
| **Región de decisión** | Partición del espacio de entrada en regiones según la clase predicha por la red |
| **Grafo computacional** | Representación de las operaciones como un grafo dirigido; backprop lo recorre en reversa |

---

## 11. Referencias

1. **Rumelhart, D. E., Hinton, G. E. & Williams, R. J.** (1986). "Learning Representations by Back-propagating Errors." *Nature*, 323(6088), 533–536.

2. **Linnainmaa, S.** (1970). "The Representation of the Cumulative Rounding Error of an Algorithm as a Taylor Expansion of the Local Rounding Errors." *Master's Thesis*, University of Helsinki.

3. **Werbos, P. J.** (1974). "Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences." *PhD Thesis*, Harvard University.

4. **Glorot, X. & Bengio, Y.** (2010). "Understanding the Difficulty of Training Deep Feedforward Neural Networks." *Proceedings of AISTATS 2010*.

5. **Hochreiter, S.** (1991). "Untersuchungen zu dynamischen neuronalen Netzen." *Diploma Thesis*, TU München. (Primera descripción formal del vanishing gradient problem.)

6. **Goodfellow, I., Bengio, Y. & Courville, A.** (2016). *Deep Learning*, Chapter 6: Deep Feedforward Networks. MIT Press.

7. **Nielsen, M.** (2015). *Neural Networks and Deep Learning*, Chapter 2: How the backpropagation algorithm works. [neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com/chap2.html).
