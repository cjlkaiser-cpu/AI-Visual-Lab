# Gradient Descent Landscape — Explorador de Superficies de Pérdida

> Simulación interactiva del descenso por gradiente sobre superficies de pérdida 2D. Coloca partículas en paisajes matemáticos — desde cuencos cuadráticos hasta la función de Rosenbrock — y observa cómo SGD, Momentum, RMSProp y Adam navegan valles, escapan puntos de silla, y convergen (o no) hacia los mínimos.

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

El **descenso por gradiente** es el motor que impulsa el aprendizaje de casi todas las redes neuronales modernas. La idea es elegantemente simple: dado un "paisaje" definido por una función de pérdida, queremos encontrar el punto más bajo — el mínimo. En cada paso, calculamos la dirección de máxima subida (el gradiente) y nos movemos en la dirección opuesta.

Esta simulación te permite visualizar ese proceso en superficies 2D. Puedes colocar múltiples **partículas** (hasta 8) en diferentes posiciones del paisaje y observar cómo cada una desciende siguiendo el gradiente. Puedes alternar entre cuatro optimizadores distintos y cinco funciones de pérdida, viendo en tiempo real cómo las decisiones algorítmicas afectan la trayectoria de convergencia.

El descenso por gradiente en 2D es conceptualmente idéntico al entrenamiento de una red neuronal con millones de parámetros — solo que en dimensiones que podemos ver.

---

## 2. Conceptos Fundamentales

### 2.1 La función de pérdida

Una **función de pérdida** $L(\boldsymbol{\theta})$ mide qué tan lejos está un modelo de la solución ideal. Los parámetros del modelo $\boldsymbol{\theta} = (\theta_1, \theta_2)$ definen un punto en el "paisaje". El objetivo del entrenamiento es encontrar los valores de $\boldsymbol{\theta}$ que minimizan $L$.

### 2.2 El gradiente

El **gradiente** $\nabla L(\boldsymbol{\theta})$ es un vector que apunta en la dirección de máximo crecimiento de $L$:

$$\nabla L(\boldsymbol{\theta}) = \left(\frac{\partial L}{\partial \theta_1}, \frac{\partial L}{\partial \theta_2}\right)$$

Su magnitud $\|\nabla L\|$ indica cuán empinada es la superficie en ese punto. En un mínimo, el gradiente es cero (o muy cercano a cero).

### 2.3 La regla de actualización

La regla fundamental del descenso por gradiente es:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla L(\boldsymbol{\theta}_t)$$

donde $\eta$ es el **learning rate**. El signo negativo invierte la dirección: nos movemos *cuesta abajo*.

### 2.4 Mínimos locales vs. globales

Un **mínimo local** es un punto donde $L$ es menor que en todos los puntos cercanos, pero no necesariamente el más bajo de toda la superficie. El **mínimo global** es el punto más bajo absoluto. Las superficies no convexas (como Rastrigin) tienen muchos mínimos locales que pueden "atrapar" al optimizador.

---

## 3. La Interfaz

### 3.1 Estructura general

| Área | Ubicación | Función |
|------|-----------|---------|
| **Canvas principal** | Izquierda | Mapa de contornos 2D o superficie 3D isométrica |
| **Botones de vista** | Esquina sup-izq del canvas | Alternar entre Contour 2D y Surface 3D |
| **Tooltip** | Sigue al mouse | Muestra coordenadas, loss y gradiente bajo el cursor |
| **Badge de convergencia** | Centro inferior del canvas | Aparece cuando todas las partículas han convergido |
| **Panel de controles** | Derecha (360px) | Ecuación, controles, estadísticas, explicaciones |

### 3.2 Vista Contour 2D

El mapa de contornos muestra la superficie de pérdida vista desde arriba. Los colores representan la altura (loss):

- **Azul oscuro/teal**: regiones de loss bajo (valles, mínimos)
- **Ámbar/rojo**: regiones de loss alto (cimas, mesetas)
- **Líneas blancas**: iso-curvas de nivel constante (18 niveles)
- **Cruceta verde**: posición del mínimo global conocido
- **Flechas blancas** (opcional): campo vectorial del gradiente negativo

### 3.3 Vista Surface 3D

Proyección isométrica de la superficie con iluminación Phong. Las partículas se muestran sobre la superficie con sus trazas. Útil para apreciar la geometría de valles y puntos de silla.

### 3.4 Las partículas

Cada partícula es un "optimizador independiente" que desciende por la superficie:

- **Punto coloreado**: posición actual (roja, azul, verde, ámbar, violeta, cyan, rosa, lima)
- **Estela**: trayectoria recorrida (últimos 200 pasos), con transparencia progresiva
- **Flecha**: dirección del gradiente negativo (dirección de descenso)
- **Anillo verde**: indica convergencia ($\|\nabla L\| < 0.005$)

---

## 4. Controles Interactivos

### 4.1 Función de pérdida

Menú desplegable con cinco superficies:

| Función | Ecuación | Mínimo | Dificultad |
|---------|----------|--------|------------|
| **Quadratic Bowl** | $L = \theta_1^2 + \theta_2^2$ | $(0, 0)$ | Fácil — convexo |
| **Rosenbrock** | $L = (1-\theta_1)^2 + 100(\theta_2-\theta_1^2)^2$ | $(1, 1)$ | Difícil — valle curvado |
| **Rastrigin** | $L = 20 + \sum(\theta_i^2 - 10\cos(2\pi\theta_i))$ | $(0, 0)$ | Muy difícil — multi-modal |
| **Saddle Point** | $L = \theta_1^2 - \theta_2^2$ | No hay | Punto de silla en origen |
| **Beale** | $(1.5-\theta_1+\theta_1\theta_2)^2 + \ldots$ | $(3, 0.5)$ | Difícil — asimétrica |

**Checkbox "Mostrar campo vectorial"**: superpone flechas que muestran la dirección del gradiente negativo en una cuadrícula de $20 \times 20$ puntos.

### 4.2 Optimizador

| Optimizador | Parámetros | Descripción |
|-------------|------------|-------------|
| **SGD** | $\eta$ | Descenso puro por gradiente |
| **SGD + Momentum** | $\eta$, $\beta$ | Acumula velocidad como inercia física |
| **RMSProp** | $\eta$ | Learning rate adaptativo por dimensión |
| **Adam** | $\eta$, $\beta_1$ | Combina momentum + learning rate adaptativo |

### 4.3 Parámetros

| Control | Rango | Default | Escala |
|---------|-------|---------|--------|
| Learning rate ($\eta$) | $10^{-4}$ – $10^{0}$ | 0.01 | Logarítmica |
| Momentum ($\beta$) | 0.00 – 0.99 | 0.90 | Lineal |

El slider de learning rate usa escala logarítmica: cada posición del slider corresponde a una potencia de 10, cubriendo cuatro órdenes de magnitud.

### 4.4 Control de ejecución

| Botón | Acción |
|-------|--------|
| **Iniciar/Pausar** | Activa/pausa la optimización continua |
| **1 Paso** | Ejecuta una sola actualización para todas las partículas |
| **10 Pasos** | Ejecuta 10 actualizaciones |
| **Reset** | Elimina todas las partículas y reinicia el estado |

**Velocidad**: 1 a 100 steps por segundo.

**Interacción con el canvas**: click para colocar una nueva partícula en esa posición. En vista 3D, la partícula se coloca en una posición aleatoria.

### 4.5 Panel de estadísticas

Muestra para la primera partícula (P1): número de partículas, step global, mejor loss alcanzado, loss actual, coordenadas $(\theta_1, \theta_2)$, magnitud del gradiente $|\nabla L|$, y gráfica del historial de loss (escala logarítmica).

### 4.6 Panel de partículas

Lista de todas las partículas activas con su color, nombre, loss actual y número de pasos. Un checkmark indica convergencia.

---

## 5. Las Matemáticas

### 5.1 SGD (Stochastic Gradient Descent)

La actualización más simple:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla L(\boldsymbol{\theta}_t)$$

Es eficiente pero puede ser lento en valles estrechos (oscila de lado a lado) y no tiene mecanismo para escapar mínimos locales poco profundos.

### 5.2 SGD con Momentum

Introduce una "velocidad" $\mathbf{v}$ que acumula gradientes pasados:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} - \eta \nabla L(\boldsymbol{\theta}_t)$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \mathbf{v}_t$$

El parámetro $\beta \in [0, 1)$ controla cuánta "inercia" acumula la partícula. Con $\beta = 0.9$, la partícula mantiene el 90% de su velocidad anterior — como una bola rodando cuesta abajo que gana impulso.

### 5.3 RMSProp

Adapta el learning rate por cada dimensión dividiendo por la raíz cuadrada de la media exponencial de gradientes al cuadrado:

$$s_t = \beta_2 s_{t-1} + (1 - \beta_2)(\nabla L)^2$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{s_t} + \epsilon} \nabla L(\boldsymbol{\theta}_t)$$

En dimensiones donde el gradiente es consistentemente grande, $s$ crece y el paso efectivo se reduce. En dimensiones con gradiente pequeño, el paso se amplifica. Esto equilibra la velocidad de descenso en todas las direcciones.

### 5.4 Adam (Adaptive Moment Estimation)

Combina las ideas de Momentum y RMSProp con **corrección de sesgo**:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)\nabla L$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2)(\nabla L)^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Los términos $\hat{m}_t$ y $\hat{v}_t$ corrigen el sesgo hacia cero que tienen las medias exponenciales en los primeros pasos (cuando $\beta^t$ es grande).

### 5.5 Propiedades de las superficies

**Convexidad**: una función $L$ es convexa si $L(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha L(\mathbf{x}) + (1-\alpha)L(\mathbf{y})$ para todo $\alpha \in [0,1]$. Las funciones convexas tienen un único mínimo global. Solo Quadratic Bowl es convexa en esta simulación.

**Número de condición**: para la función cuadrática $L = a\theta_1^2 + b\theta_2^2$, el número de condición es $\kappa = \max(a,b)/\min(a,b)$. Cuanto mayor es $\kappa$, más "alargados" son los contornos y más oscila SGD. La Rosenbrock tiene $\kappa$ efectivo muy alto en el valle.

**Punto de silla**: un punto donde $\nabla L = 0$ pero no es un mínimo ni un máximo. La matriz Hessiana tiene eigenvalores de signos opuestos. En dimensiones altas, los puntos de silla son mucho más comunes que los mínimos locales.

---

## 6. Sonificación

### 6.1 Sonido del descenso

Durante la optimización, cada paso produce un sonido cuya frecuencia depende del valor de loss:

$$f = 180 + \min(\log(L+1) \cdot 80, \, 600) \text{ Hz}$$

- **Loss alto**: frecuencia alta (~780 Hz), más agudo
- **Loss bajo**: frecuencia baja (~180 Hz), más grave
- **Volumen**: proporcional a la magnitud del gradiente (más empinado = más fuerte)

Los sonidos se throttlean a un máximo de 10 por segundo para evitar saturación.

### 6.2 Acorde de convergencia

Cuando una partícula converge ($\|\nabla L\| < 0.005$), se reproduce un acorde de Do mayor ascendente: $C_4$–$E_4$–$G_4$–$C_5$ (261–330–392–523 Hz), cada nota separada por 60ms.

### 6.3 Formas de onda disponibles

- **Sine**: suave, indica descenso estable
- **Triangle**: ligeramente más brillante, útil para distinguir frecuencias
- **Sawtooth**: timbre buzzy, hace más evidentes los cambios de frecuencia

---

## 7. Guía Paso a Paso

### Experimento 1: El cuenco cuadrático

1. Selecciona **Quadratic Bowl** y optimizador **SGD**
2. Haz click en una esquina del canvas para colocar una partícula lejos del centro
3. Click en **Iniciar** — observa cómo desciende en línea recta hacia $(0,0)$
4. Prueba con $\eta = 0.5$: la partícula converge rápido
5. Ahora $\eta = 0.001$: convergencia mucho más lenta
6. Coloca una segunda partícula en otra esquina. Ambas convergen al mismo punto
7. Activa el campo vectorial para ver las flechas apuntando al centro

### Experimento 2: El valle de Rosenbrock

1. Cambia a **Rosenbrock** y coloca una partícula en $(-1.5, -1.5)$
2. Con **SGD**, observa cómo baja rápidamente al valle pero luego avanza muy lentamente a lo largo de él
3. Cambia a **SGD + Momentum** ($\beta = 0.9$): la partícula gana impulso y recorre el valle más rápido
4. Prueba **Adam**: convergencia más directa, menos oscilaciones
5. Coloca varias partículas para comparar trayectorias desde diferentes puntos de inicio

### Experimento 3: Mínimos locales en Rastrigin

1. Selecciona **Rastrigin** — nota los múltiples "pozos" en el mapa de contornos
2. Coloca 5-6 partículas en diferentes posiciones con **SGD**
3. Observa cómo la mayoría quedan atrapadas en mínimos locales, no en el global $(0,0)$
4. Repite con **Momentum** y $\eta$ más alto: algunas partículas logran saltar entre mínimos
5. Esto demuestra por qué encontrar el mínimo global es un problema difícil

### Experimento 4: El punto de silla

1. Selecciona **Saddle Point** ($L = \theta_1^2 - \theta_2^2$)
2. Coloca una partícula exactamente en el centro — se queda atrapada (gradiente cercano a cero)
3. Coloca una ligeramente desplazada: se aleja del punto de silla
4. Con **Momentum**, las partículas cruzan más rápidamente la región de silla
5. Observa la vista 3D para ver la geometría de "silla de montar"

### Experimento 5: Comparación de optimizadores

1. Elige **Rosenbrock**
2. Coloca una partícula roja con **SGD**. Anota cuántos pasos hasta loss < 0.01
3. Reset. Coloca una con **Momentum**. Compara
4. Reset. Prueba **RMSProp**
5. Reset. Prueba **Adam**
6. ¿Cuál converge más rápido? ¿Cuál tiene la trayectoria más suave?

---

## 8. Conceptos Avanzados

### 8.1 Learning rate scheduling

En la práctica, el learning rate no se mantiene constante. Estrategias comunes:

- **Step decay**: reducir $\eta$ por un factor cada N epochs
- **Cosine annealing**: $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t/T))$
- **Warmup**: empezar con $\eta$ muy bajo y aumentar linealmente durante los primeros pasos

### 8.2 Gradient clipping

La simulación implementa **gradient clipping**: si $\|\nabla L\| > 100$, el gradiente se recorta a magnitud 100. Esto previene la "explosión de gradientes" que ocurre en superficies con pendientes extremas (como Beale lejos del mínimo). También limita el tamaño del paso a 1 unidad.

### 8.3 Paisajes de pérdida reales

Las superficies de esta simulación tienen 2 parámetros. Las redes neuronales reales tienen millones. En dimensiones altas:

- Los **mínimos locales** son raros; los **puntos de silla** son ubicuos
- La mayoría de los mínimos locales tienen loss muy similar al global (el paisaje es "benigno")
- Las direcciones de alta curvatura son pocas; la mayoría son "planas" — el descenso ocurre en un subespacio de baja dimensión

### 8.4 Convergencia y tasa de convergencia

Para funciones convexas con constante de Lipschitz $\ell$ del gradiente, SGD con $\eta \leq 1/\ell$ garantiza:

$$L(\boldsymbol{\theta}_T) - L(\boldsymbol{\theta}^*) \leq \frac{\|\boldsymbol{\theta}_0 - \boldsymbol{\theta}^*\|^2}{2\eta T}$$

Es decir, convergencia $O(1/T)$. Con funciones fuertemente convexas (como Quadratic Bowl), la tasa mejora a convergencia lineal $O(\kappa^T)$ donde $\kappa < 1$.

### 8.5 Visualización logarítmica

La simulación aplica $\log(L+1)$ para la visualización de contornos y colores. Esto comprime el rango dinámico de funciones como Rosenbrock (donde $L$ varía de 0 a $>10^4$) y hace visibles los detalles cerca del mínimo.

---

## 9. Ejercicios

### Ejercicio 1: Learning rate óptimo

Para la función **Quadratic Bowl**, encuentra el learning rate más alto que permita convergencia estable con SGD. ¿Qué ocurre cuando $\eta > 1$? ¿Y cuando $\eta = 1$ exactamente? Relaciona tu observación con el hecho de que el gradiente de $L = \theta^2$ es $2\theta$.

### Ejercicio 2: Momentum y oscilaciones

Usa la función **Quadratic Bowl** con un learning rate deliberadamente alto ($\eta = 0.3$). Primero, observa las oscilaciones con SGD puro. Luego activa Momentum con $\beta = 0.9$. ¿Mejora o empeora la situación? Prueba $\beta = 0.5$. ¿Por qué valores altos de momentum pueden amplificar las oscilaciones?

### Ejercicio 3: Adam vs. SGD en Rosenbrock

Coloca una partícula en $(-1.5, 1.5)$ en la función **Rosenbrock**. Ejecuta 500 pasos con SGD ($\eta = 0.001$) y anota el loss final. Repite con Adam ($\eta = 0.001$). ¿Cuál está más cerca del mínimo $(1,1)$? ¿Qué papel juega la adaptación del learning rate por dimensión?

### Ejercicio 4: Escape de mínimos locales

En **Rastrigin**, ¿puedes encontrar una combinación de optimizador, learning rate y momentum que permita a una partícula escapar de un mínimo local y llegar al global? Experimenta con al menos 3 configuraciones y documenta los resultados.

### Ejercicio 5: Geometría del punto de silla

En la función **Saddle Point**, activa el campo vectorial. Observa las flechas alrededor del origen. ¿En qué direcciones apuntan las flechas *hacia* el origen? ¿En cuáles *se alejan*? Relaciona esto con los eigenvalores de la Hessiana $H = \begin{pmatrix} 2 & 0 \\ 0 & -2 \end{pmatrix}$.

### Ejercicio 6: Trayectorias de Beale

En la función **Beale**, coloca 4 partículas en las cuatro esquinas del espacio con **Adam** ($\eta = 0.01$). ¿Cuántas llegan al mínimo en $(3, 0.5)$? ¿Cuántas divergen o se quedan atrapadas? ¿Cambia el resultado con $\eta = 0.001$?

---

## 10. Glosario

| Término | Definición |
|---------|------------|
| **Función de pérdida** ($L$) | Función que mide el error del modelo; el objetivo es minimizarla |
| **Gradiente** ($\nabla L$) | Vector de derivadas parciales que apunta en la dirección de máximo crecimiento |
| **Learning rate** ($\eta$) | Tamaño del paso en la dirección del gradiente negativo; controla velocidad vs. estabilidad |
| **SGD** | Stochastic Gradient Descent — actualización directa $\theta \leftarrow \theta - \eta\nabla L$ |
| **Momentum** ($\beta$) | Acumulación de velocidad usando gradientes pasados; ayuda a cruzar valles |
| **RMSProp** | Optimizador que adapta $\eta$ por dimensión usando media de gradientes al cuadrado |
| **Adam** | Optimizador que combina momentum y learning rate adaptativo con corrección de sesgo |
| **Mínimo local** | Punto donde $L$ es menor que en su entorno, pero no necesariamente el más bajo globalmente |
| **Mínimo global** | Punto con el valor más bajo de $L$ en todo el dominio |
| **Punto de silla** | Punto con $\nabla L = 0$ donde la Hessiana tiene eigenvalores positivos y negativos |
| **Convergencia** | Estado donde $\|\nabla L\|$ es suficientemente pequeño (< 0.005 en esta simulación) |
| **Mapa de contornos** | Visualización 2D de una superficie donde líneas conectan puntos de igual valor |
| **Campo vectorial** | Representación visual del gradiente negativo en cada punto del espacio |
| **Gradient clipping** | Técnica que recorta el gradiente si su magnitud supera un umbral, previniendo explosiones |
| **Convexidad** | Propiedad de una función que garantiza un único mínimo global |
| **Número de condición** ($\kappa$) | Ratio entre curvatura máxima y mínima; valores altos indican optimización difícil |
| **Corrección de sesgo** | Ajuste en Adam que compensa la inicialización en cero de las medias exponenciales |
| **Hessiana** | Matriz de segundas derivadas; sus eigenvalores indican la curvatura de la superficie |

---

## 11. Referencias

1. **Cauchy, A.** (1847). "Méthode générale pour la résolution des systèmes d'équations simultanées." *Comptes Rendus de l'Académie des Sciences*, 25, 536–538.

2. **Robbins, H. & Monro, S.** (1951). "A Stochastic Approximation Method." *The Annals of Mathematical Statistics*, 22(3), 400–407.

3. **Polyak, B. T.** (1964). "Some methods of speeding up the convergence of iteration methods." *USSR Computational Mathematics and Mathematical Physics*, 4(5), 1–17.

4. **Hinton, G. E.** (2012). "Neural Networks for Machine Learning — Lecture 6a: Overview of mini-batch gradient descent." *Coursera*.

5. **Kingma, D. P. & Ba, J.** (2015). "Adam: A Method for Stochastic Optimization." *Proceedings of ICLR 2015*. arXiv:1412.6980.

6. **Rosenbrock, H. H.** (1960). "An Automatic Method for Finding the Greatest or Least Value of a Function." *The Computer Journal*, 3(3), 175–184.

7. **Goodfellow, I., Bengio, Y. & Courville, A.** (2016). *Deep Learning*, Chapter 8: Optimization for Training Deep Models. MIT Press.

8. **Ruder, S.** (2016). "An Overview of Gradient Descent Optimization Algorithms." arXiv:1609.04747.
