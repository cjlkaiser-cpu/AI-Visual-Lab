# GAN Duel — El Duelo Adversarial

> Simulación interactiva de una GAN (Generative Adversarial Network) donde un generador y un discriminador compiten en 2D. Elige una distribución objetivo (anillo, clusters, espiral, grid, gaussiana), entrena paso a paso, y observa cómo los puntos generados convergen hacia la distribución real. Detecta mode collapse, ajusta learning rates, y visualiza el campo de decisión del discriminador.

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

Una **GAN** (Generative Adversarial Network) es un sistema de dos redes neuronales que compiten entre sí como en un juego:

- El **Generador** (G) intenta crear datos falsos que parezcan reales
- El **Discriminador** (D) intenta distinguir datos reales de falsos

Esta competencia impulsa a ambas redes a mejorar: G genera datos cada vez más realistas, D se vuelve cada vez mejor detectando falsificaciones. En el equilibrio ideal, G produce datos indistinguibles de los reales.

Esta simulación trabaja en 2D: la distribución "real" son puntos en formas geométricas (anillo, clusters, espiral), y el generador aprende a producir puntos que imiten esa distribución. Puedes observar en tiempo real cómo los puntos generados (rojos) convergen hacia los puntos reales (verdes), mientras el heatmap de fondo muestra dónde el discriminador cree que hay datos reales (azul) o falsos.

---

## 2. Conceptos Fundamentales

### 2.1 El juego minimax

La GAN se formula como un juego de suma cero:

$$\min_G \max_D \; \mathbb{E}_{x \sim p_\text{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

- $D(x)$: probabilidad de que $x$ sea real (según el discriminador)
- $G(z)$: dato generado a partir del ruido $z$
- $p_\text{data}$: distribución de datos reales
- $p_z$: distribución del ruido de entrada (típicamente gaussiana)

### 2.2 El generador

El generador toma **ruido aleatorio** $z \sim \mathcal{N}(0, I)$ y lo transforma en puntos 2D:

$$G: \mathbb{R}^{d_z} \to \mathbb{R}^2$$

Su objetivo es que $D(G(z))$ sea lo más cercano posible a 1 (engañar al discriminador).

### 2.3 El discriminador

El discriminador recibe un punto 2D y produce la probabilidad de que sea real:

$$D: \mathbb{R}^2 \to [0, 1]$$

Se entrena con datos reales (etiqueta 1) y datos generados (etiqueta 0).

### 2.4 Equilibrio de Nash

El estado ideal es el **equilibrio de Nash**: G produce datos con la misma distribución que los reales, y D no puede distinguirlos ($D(x) = 0.5$ para todo $x$). En la práctica, alcanzar este equilibrio es difícil, y las GANs suelen oscilar.

---

## 3. La Interfaz

### 3.1 Estructura general

| Área | Ubicación | Función |
|------|-----------|---------|
| **Canvas principal** | Izquierda | Puntos reales (verdes), generados (rojos), heatmap del discriminador |
| **Badge overlay** | Esquina sup. izq. | Estado ("Pulsa Entrenar", "Entrenando...") |
| **Panel de controles** | Derecha (360px) | Entrenamiento, configuración, loss chart, estado, audio |

### 3.2 Visualización del canvas

- **Puntos verdes** ($\bullet$): muestras de la distribución real
- **Puntos rojos** ($\bullet$): muestras generadas por G
- **Heatmap de fondo**: campo de decisión de D
  - Azul intenso = D cree que hay datos reales ($D(x) \to 1$)
  - Oscuro/negro = D cree que es espacio vacío ($D(x) \to 0$)
- **Convergencia**: cuando los puntos rojos se superponen con los verdes, G ha aprendido

### 3.3 Header

Muestra el paso de entrenamiento actual, G loss y D loss.

---

## 4. Controles Interactivos

### 4.1 Entrenamiento

| Control | Función |
|---------|---------|
| **Entrenar** | Inicia/pausa el entrenamiento continuo |
| **+1 Paso** | Un solo paso de entrenamiento |
| **+100** | 100 pasos de una vez |
| **Reset** | Reinicia pesos y entrenamiento |

### 4.2 Configuración

| Parámetro | Rango | Descripción |
|-----------|-------|-------------|
| **Distribución** | Anillo, Dos clusters, Espiral, Grid, Gaussiana | Forma de la distribución real |
| **LR Generador** | $10^{-4}$ a $10^{-1}$ (log) | Learning rate del generador |
| **LR Discriminador** | $10^{-4}$ a $10^{-1}$ (log) | Learning rate del discriminador |
| **Puntos por batch** | 16–256 (paso 16) | Tamaño del mini-batch |

### 4.3 Gráfico de loss

Un mini-gráfico muestra la evolución temporal de:
- **G loss** (rojo): pérdida del generador
- **D loss** (azul): pérdida del discriminador

### 4.4 Estado

| Métrica | Descripción |
|---------|-------------|
| **Paso** | Número de iteración actual |
| **G loss** | Pérdida del generador |
| **D loss** | Pérdida del discriminador |
| **D(real) medio** | Probabilidad media asignada a datos reales |
| **D(fake) medio** | Probabilidad media asignada a datos generados |
| **Mode collapse** | Detector automático: Sí/No |

### 4.5 Audio

- **Volumen**: 0-100%
- **Audio ON/OFF**: toggle

---

## 5. Las Matemáticas

### 5.1 Loss del discriminador

$$L_D = -\frac{1}{m}\sum_{i=1}^{m}\left[\log D(x^{(i)}) + \log(1 - D(G(z^{(i)})))\right]$$

El discriminador quiere **maximizar** esta expresión: dar probabilidades altas a datos reales y bajas a datos falsos.

### 5.2 Loss del generador

$$L_G = -\frac{1}{m}\sum_{i=1}^{m}\log D(G(z^{(i)}))$$

El generador quiere **minimizar** esta expresión: hacer que el discriminador asigne probabilidades altas a sus datos falsos.

En la práctica se usa $-\log D(G(z))$ en lugar de $\log(1 - D(G(z)))$ porque proporciona gradientes más fuertes al inicio del entrenamiento (truco de Goodfellow).

### 5.3 Entrenamiento alternante

En cada paso:
1. **Entrenar D**: generar batch de datos reales y falsos, calcular $L_D$, actualizar pesos de D
2. **Entrenar G**: generar batch de datos falsos, calcular $L_G$, actualizar pesos de G

### 5.4 Distribuciones implementadas

| Distribución | Generación |
|-------------|------------|
| **Anillo** | $(r\cos\theta, r\sin\theta)$ con $\theta \sim U[0, 2\pi]$, $r$ con ruido gaussiano |
| **Dos clusters** | Mezcla de dos gaussianas centradas en $(\pm c, 0)$ |
| **Espiral** | $(\theta\cos\theta, \theta\sin\theta)$ con $\theta \in [0, 4\pi]$ y ruido |
| **Grid** | Puntos en rejilla regular con ruido gaussiano |
| **Gaussiana** | $\mathcal{N}(0, \sigma^2 I)$ |

### 5.5 Detección de mode collapse

La simulación detecta mode collapse cuando la varianza de los puntos generados cae por debajo de un umbral:

$$\text{var}(G(z)) < \epsilon \cdot \text{var}(x_\text{real})$$

Mode collapse ocurre cuando G produce puntos en una sola región, ignorando la diversidad de la distribución real.

---

## 6. Sonificación

### 6.1 Mapeo audio

| Evento | Sonido | Parámetro mapeado |
|--------|--------|-------------------|
| **Paso de entrenamiento** | Pulso rítmico | Velocidad del entrenamiento |
| **G loss bajando** | Tono ascendente suave | El generador mejora |
| **D loss bajando** | Tono descendente | El discriminador mejora |
| **Equilibrio** | Armonía | G loss ≈ D loss |
| **Mode collapse** | Tono monótono repetitivo | Varianza baja en puntos generados |
| **Convergencia** | Acorde resuelto | Distribuciones superpuestas |

### 6.2 Interpretación

El "duelo" audible: cuando G y D están equilibrados, el sonido es armónico. Cuando uno domina, hay disonancia. Mode collapse produce un sonido repetitivo y sin variación.

---

## 7. Guía Paso a Paso

### 7.1 Primera exploración

1. Selecciona la distribución **Anillo**
2. Pulsa **Entrenar** y observa los puntos rojos converger hacia el anillo verde
3. Mira el heatmap: D empieza coloreando todo en azul (cree que todo es real), luego refina
4. Observa las losses: al principio D loss baja rápido, luego G empieza a mejorar
5. Pausa cuando D(real) ≈ 0.5 y D(fake) ≈ 0.5 — has alcanzado un cuasi-equilibrio

### 7.2 Comparar distribuciones

1. Prueba cada distribución: Anillo → Clusters → Espiral → Grid → Gaussiana
2. La **Gaussiana** es la más fácil (unimodal)
3. La **Espiral** es la más difícil (estructura continua compleja)
4. Los **Clusters** pueden causar mode collapse (G solo aprende un cluster)

### 7.3 Provocar mode collapse

1. Selecciona **Dos clusters**
2. Pon LR Generador = 0.1 (muy alto) y LR Discriminador = 0.001 (bajo)
3. Entrena: el generador aprende un solo cluster y se queda ahí
4. La métrica "Mode collapse" debería cambiar a "Sí"
5. Reset y usa LR más balanceados para evitarlo

### 7.4 Balance de learning rates

1. Con la distribución **Anillo**:
   - LR G = 0.01, LR D = 0.01 → entrenamiento estable
   - LR G = 0.1, LR D = 0.001 → G domina, mode collapse probable
   - LR G = 0.001, LR D = 0.1 → D domina, G no aprende
2. Observa cómo el balance de LR es crucial para la estabilidad

---

## 8. Conceptos Avanzados

### 8.1 Teorema de convergencia de Goodfellow

Si G y D tienen capacidad infinita y D se entrena hasta el óptimo en cada paso, G converge a $p_\text{data}$. El discriminador óptimo es:

$$D^*(x) = \frac{p_\text{data}(x)}{p_\text{data}(x) + p_G(x)}$$

En el equilibrio: $p_G = p_\text{data}$ y $D^*(x) = 0.5$ para todo $x$.

### 8.2 Wasserstein GAN (WGAN)

La GAN original sufre de inestabilidad de entrenamiento. La WGAN reemplaza la divergencia de Jensen-Shannon con la **distancia de Wasserstein**:

$$W(p_r, p_g) = \inf_{\gamma \in \Pi(p_r, p_g)} \mathbb{E}_{(x,y) \sim \gamma}[\|x - y\|]$$

Esto proporciona gradientes más estables y evita el problema de "vanishing gradients" cuando D es demasiado bueno.

### 8.3 Mode collapse en detalle

Mode collapse viene en dos formas:
- **Parcial**: G produce datos en solo algunas modas de la distribución (ej: 1 de 2 clusters)
- **Total**: G produce un único punto o una región diminuta

Soluciones: mini-batch discrimination, unrolled GANs, spectral normalization.

### 8.4 GANs condicionales

Las GANs condicionales añaden información de clase $y$:

$$G(z, y) \to x, \quad D(x, y) \to [0, 1]$$

Permiten generar datos específicos: "genera un perro" o "convierte día a noche".

### 8.5 Aplicaciones modernas

- **Generación de imágenes**: StyleGAN, DALL-E (con variantes)
- **Transferencia de estilo**: CycleGAN convierte caballos en cebras
- **Super-resolución**: SRGAN mejora la resolución de imágenes
- **Datos sintéticos**: generar datos de entrenamiento donde hay escasez
- **Drug discovery**: generar moléculas candidatas

---

## 9. Ejercicios

### Ejercicio 1: Convergencia en anillo
Entrena con distribución **Anillo** y learning rates por defecto (0.01). ¿En cuántos pasos los puntos rojos cubren razonablemente el anillo? ¿D(real) y D(fake) convergen a 0.5?

### Ejercicio 2: Mode collapse controlado
Con **Dos clusters**, intenta provocar mode collapse ajustando los learning rates. Documenta la combinación de LR G y LR D que causa collapse. Luego encuentra una combinación que lo evite. ¿Cuál es la relación crítica LR_G/LR_D?

### Ejercicio 3: Espiral difícil
La distribución **Espiral** tiene estructura continua no convexa. Entrena hasta 500 pasos. ¿G captura la estructura espiral completa o solo partes? ¿Ayuda aumentar el batch size? Compara batch 16 vs batch 256.

### Ejercicio 4: Lectura del heatmap
Con distribución **Grid**, entrena hasta que el heatmap de D se estabilice. ¿Cuántos "picos" azules muestra el heatmap? ¿Coinciden con las posiciones del grid? ¿Qué pasa entre los picos (en el espacio vacío)?

### Ejercicio 5: Análisis de las losses
Entrena con distribución **Gaussiana** durante 300 pasos. Dibuja (en papel) la evolución de G loss y D loss. ¿Hay un patrón oscilatorio? ¿Las losses convergen a algún valor? En el equilibrio teórico, $L_G = L_D = \log 2 \approx 0.693$. ¿Qué tan cerca llegas?

### Ejercicio 6: Batch size y estabilidad
Fija la distribución **Anillo** y LR = 0.01 para ambos. Entrena 200 pasos con batch 16, luego reset y 200 pasos con batch 256. Compara la suavidad de las curvas de loss y la calidad de los puntos generados. ¿Por qué batches más grandes producen entrenamiento más estable?

---

## 10. Glosario

| Término | Definición |
|---------|------------|
| **GAN** | Generative Adversarial Network: dos redes que compiten (generador vs discriminador) |
| **Generador (G)** | Red que transforma ruido aleatorio en datos sintéticos |
| **Discriminador (D)** | Red que clasifica datos como reales o generados |
| **Minimax** | Formulación de juego: G minimiza lo que D maximiza |
| **Equilibrio de Nash** | Estado donde ningún jugador puede mejorar unilateralmente |
| **Mode collapse** | Fallo donde G produce muestras con poca diversidad |
| **Distribución latente** | Distribución del ruido de entrada del generador (típicamente gaussiana) |
| **Loss del generador** | $-\log D(G(z))$: penalización cuando D detecta los datos falsos |
| **Loss del discriminador** | $-[\log D(x) + \log(1-D(G(z)))]$: penalización por errores de clasificación |
| **D(real)** | Probabilidad media que D asigna a datos reales (idealmente → 0.5) |
| **D(fake)** | Probabilidad media que D asigna a datos generados (idealmente → 0.5) |
| **Learning rate** | Tasa de aprendizaje: magnitud de cada actualización de pesos |
| **Batch size** | Número de muestras procesadas por paso de entrenamiento |
| **Heatmap** | Mapa de calor que visualiza el campo de decisión de D |
| **Wasserstein** | Distancia alternativa que mejora la estabilidad del entrenamiento |
| **Condicional** | GAN que recibe información adicional (clase, texto) para generar datos específicos |
| **Spectral normalization** | Técnica de regularización para estabilizar el entrenamiento de D |
| **Convergencia** | Estado donde la distribución generada coincide con la real |

---

## 11. Referencias

1. Goodfellow, I. et al. (2014). "Generative Adversarial Nets." *NeurIPS*.
2. Arjovsky, M., Chintala, S. & Bottou, L. (2017). "Wasserstein GAN." *ICML*.
3. Radford, A., Metz, L. & Chintala, S. (2016). "Unsupervised Representation Learning with Deep Convolutional GANs." *ICLR*.
4. Karras, T. et al. (2019). "A Style-Based Generator Architecture for GANs." *CVPR*.
5. Salimans, T. et al. (2016). "Improved Techniques for Training GANs." *NeurIPS*.
6. Goodfellow, I. (2016). "NIPS 2016 Tutorial: Generative Adversarial Networks." *arXiv:1701.00160*.
