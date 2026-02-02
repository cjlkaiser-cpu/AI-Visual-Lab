# Autoencoder Cathedral — Compresión y Reconstrucción de Dígitos

> Simulación interactiva de un autoencoder que aprende a comprimir dígitos escritos a mano en un espacio latente de pocas dimensiones y reconstruirlos. Dibuja dígitos, observa cómo la red los comprime a 2 coordenadas, interpola entre ellos, y genera "fantasías" desde el espacio latente.

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

Un **autoencoder** es una red neuronal que aprende a comprimir datos. Tiene forma de reloj de arena: un **encoder** reduce la dimensionalidad de la entrada, un **cuello de botella** (bottleneck) fuerza una representación comprimida, y un **decoder** intenta reconstruir la entrada original a partir de esa representación.

Esta simulación trabaja con dígitos escritos a mano de $8 \times 8$ píxeles (64 dimensiones). La red los comprime a un **espacio latente** de tan solo 2 dimensiones — una compresión de 32:1 — y luego intenta reconstruirlos. Puedes ver la reconstrucción en tiempo real, explorar el espacio latente 2D donde cada dígito se convierte en un punto, interpolar entre dígitos, y generar dígitos "imaginarios" que la red nunca ha visto.

La metáfora de la "catedral" evoca las bóvedas que comprimen fuerzas en estructuras esbeltas: información amplia que se canaliza a través de un punto estrecho.

---

## 2. Conceptos Fundamentales

### 2.1 La arquitectura encoder-decoder

El autoencoder consiste en dos redes simétricas:

**Encoder**: $64 \to 32 \to n_\text{bottleneck}$

$$\mathbf{z} = f_\text{enc}(\mathbf{x})$$

**Decoder**: $n_\text{bottleneck} \to 32 \to 64$

$$\hat{\mathbf{x}} = f_\text{dec}(\mathbf{z})$$

### 2.2 La función de pérdida

El objetivo es minimizar la **error de reconstrucción**:

$$L = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 = \sum_{i=1}^{64}(x_i - \hat{x}_i)^2$$

La red aprende a preservar la información más importante en el bottleneck, descartando el ruido y los detalles irrelevantes.

### 2.3 El espacio latente

El **espacio latente** es el espacio de baja dimensionalidad donde vive la representación comprimida $\mathbf{z}$. Cuando el bottleneck tiene 2 dimensiones, cada dígito se convierte en un punto $(z_1, z_2)$ que podemos visualizar en un plano.

Los dígitos similares tienden a agruparse en regiones cercanas del espacio latente. La interpolación entre dos puntos produce transiciones suaves entre dígitos.

### 2.4 Ratio de compresión

$$\text{ratio} = \frac{\text{dim entrada}}{\text{dim bottleneck}} = \frac{64}{n_\text{bottleneck}}$$

Con bottleneck = 2: ratio 32:1. Con bottleneck = 16: ratio 4:1.

---

## 3. La Interfaz

### 3.1 Estructura general

| Área | Ubicación | Función |
|------|-----------|---------|
| **Canvas principal** | Izquierda | Diagrama de red con flujo de datos, entrada/salida visualizadas |
| **Espacio latente 2D** | Overlay en canvas | Mapa de los dígitos en el plano latente (si bottleneck = 2) |
| **Panel de controles** | Derecha (360px) | Arquitectura, entrenamiento, dibujo, interpolación, métricas |

### 3.2 Visualización de la red

El canvas muestra:

- **Entrada**: cuadrícula $8 \times 8$ del dígito original
- **Encoder**: capas que reducen dimensionalidad (64 → 32 → bottleneck)
- **Bottleneck**: representación comprimida como vector de $n$ números
- **Decoder**: capas que expanden (bottleneck → 32 → 64)
- **Salida**: cuadrícula $8 \times 8$ de la reconstrucción $\hat{x}$

### 3.3 Espacio latente 2D

Cuando el bottleneck es 2D, se muestra un mapa donde:
- Cada punto coloreado representa un dígito del dataset
- Los colores corresponden a la clase (0-9)
- Clusters bien definidos indican buena separación
- Se puede hacer click para "navegar" el espacio y generar dígitos

---

## 4. Controles Interactivos

### 4.1 Arquitectura

| Control | Rango | Default | Descripción |
|---------|-------|---------|-------------|
| Bottleneck | 2 – 16 dims | 2 | Dimensionalidad del espacio latente |
| Learning rate | $10^{-4}$ – $10^{-1}$ | 0.01 | Escala logarítmica |

La arquitectura completa se muestra como: $64 \to 32 \to n \to 32 \to 64$.

### 4.2 Entrenamiento

| Botón | Acción |
|-------|--------|
| **Entrenar** | Inicia/detiene el entrenamiento continuo |
| **+50 Epochs** | Ejecuta 50 epochs rápidamente |
| **Reset** | Reinicializa todos los pesos |

### 4.3 Input: dígitos

**Botones 0-9**: carga un dígito pre-definido del dataset (estilo sklearn digits, $8 \times 8$, valores 0-16).

**Cuadrícula de dibujo**: permite dibujar un dígito manualmente:
- **Click**: pintar píxel
- **Shift + click**: borrar píxel

### 4.4 Espacio latente

| Control | Descripción |
|---------|-------------|
| **Interpolación** | Slider 0.0 – 1.0: mezcla entre punto A y punto B |
| **Set A / Set B** | Fija el dígito actual como extremo de interpolación |
| **Fantasía** | Genera un dígito aleatorio desde el espacio latente |
| **Mostrar espacio latente 2D** | Toggle para la visualización del mapa latente |

### 4.5 Métricas

- **Epoch**: número de pasadas por el dataset
- **Loss (MSE)**: error cuadrático medio de reconstrucción
- **Bottleneck dim**: dimensionalidad actual
- **Parámetros**: número total de pesos en la red
- **Ratio compresión**: dim entrada / dim bottleneck

---

## 5. Las Matemáticas

### 5.1 Forward pass del encoder

Para una entrada $\mathbf{x} \in \mathbb{R}^{64}$:

$$\mathbf{h}_1 = \sigma(W_1 \mathbf{x} + \mathbf{b}_1) \in \mathbb{R}^{32}$$
$$\mathbf{z} = \sigma(W_2 \mathbf{h}_1 + \mathbf{b}_2) \in \mathbb{R}^{n}$$

### 5.2 Forward pass del decoder

$$\mathbf{h}_3 = \sigma(W_3 \mathbf{z} + \mathbf{b}_3) \in \mathbb{R}^{32}$$
$$\hat{\mathbf{x}} = \sigma(W_4 \mathbf{h}_3 + \mathbf{b}_4) \in \mathbb{R}^{64}$$

### 5.3 Entrenamiento

La función de pérdida MSE se minimiza por backpropagation:

$$L = \frac{1}{N}\sum_{n=1}^{N}\|\mathbf{x}^{(n)} - \hat{\mathbf{x}}^{(n)}\|^2$$

Los gradientes fluyen desde la salida, a través del decoder, por el bottleneck, y hasta el encoder.

### 5.4 Interpolación lineal

Dados dos dígitos con representaciones latentes $\mathbf{z}_A$ y $\mathbf{z}_B$, la interpolación es:

$$\mathbf{z}_t = (1 - t)\mathbf{z}_A + t\mathbf{z}_B, \quad t \in [0, 1]$$

El dígito reconstruido $\hat{\mathbf{x}}_t = f_\text{dec}(\mathbf{z}_t)$ muestra una transición suave entre A y B.

### 5.5 Generación (fantasía)

Un punto aleatorio $\mathbf{z}_\text{random}$ en el espacio latente se decodifica:

$$\hat{\mathbf{x}} = f_\text{dec}(\mathbf{z}_\text{random})$$

Si $\mathbf{z}_\text{random}$ cae en una región donde el autoencoder ha visto datos de entrenamiento, la reconstrucción será un dígito reconocible. Si cae fuera, será ruido o un "artefacto".

### 5.6 Relación con PCA

Un autoencoder lineal (sin funciones de activación) con bottleneck de dimensión $k$ aprende exactamente las **$k$ componentes principales** del dataset. El autoencoder no lineal es una generalización de PCA que puede capturar estructuras más complejas.

---

## 6. Sonificación

### 6.1 Sonido del entrenamiento

- **Cada epoch**: nota cuya frecuencia es inversamente proporcional al loss. Loss alto = grave; loss bajo = agudo.
- **Convergencia**: acorde consonante cuando el loss cae por debajo de un umbral.

### 6.2 Sonido del espacio latente

- **Interpolación**: las coordenadas $z_1$ y $z_2$ modulan la frecuencia y el volumen de dos osciladores, creando un sonido que cambia mientras deslizas.
- **Fantasía**: acorde breve aleatorio al generar un nuevo punto.

### 6.3 Error de reconstrucción

El sonido refleja la calidad de la reconstrucción: cuando el MSE es bajo, el sonido es limpio y consonante; cuando es alto, es disonante.

---

## 7. Guía Paso a Paso

### Experimento 1: Entrenamiento básico

1. Mantén la configuración por defecto (bottleneck = 2)
2. Click en **Entrenar** y observa el loss descender en la gráfica
3. Selecciona diferentes dígitos (0-9) y observa la reconstrucción mejorar
4. Tras ~100 epochs, los dígitos deberían ser reconocibles en la salida
5. Compara la entrada original con la reconstrucción: ¿qué detalles se pierden?

### Experimento 2: Explorar el espacio latente

1. Tras entrenar ~200 epochs con bottleneck = 2
2. Activa "Mostrar espacio latente 2D"
3. Observa cómo los dígitos se agrupan en clusters por clase
4. Carga el dígito "3" y luego el "8" — nota que están cerca (ambos son redondeados)
5. "1" y "7" también deberían estar cerca (ambos son angulares)

### Experimento 3: Interpolación entre dígitos

1. Carga el dígito "0", click en **Set A**
2. Carga el dígito "8", click en **Set B**
3. Desliza **Interpolación** de 0.0 a 1.0
4. Observa la transición gradual: el "0" se transforma suavemente en "8"
5. Prueba interpolaciones más extremas: "1" ↔ "0", "3" ↔ "5"

### Experimento 4: Fantasías del decoder

1. Con la red entrenada, click en **Fantasía** varias veces
2. Observa los dígitos generados: algunos serán reconocibles, otros serán híbridos
3. Los mejores resultados ocurren cuando el punto aleatorio cae dentro de un cluster

### Experimento 5: Efecto del bottleneck

1. Entrena con bottleneck = 2 durante 200 epochs. Anota el loss final
2. Reset, cambia bottleneck = 8 y entrena 200 epochs. Compara el loss
3. Reset, bottleneck = 16 y 200 epochs
4. ¿Cómo mejora la reconstrucción con más dimensiones?
5. ¿A partir de qué dimensión la mejora es marginal?

### Experimento 6: Dibujar tu propio dígito

1. Usa la cuadrícula de dibujo para crear un dígito a mano
2. Observa la reconstrucción: ¿la red reconoce tu estilo?
3. Dibuja un dígito "ambiguo" (entre 3 y 8, por ejemplo)
4. Observa dónde cae en el espacio latente: ¿entre los clusters de 3 y 8?

---

## 8. Conceptos Avanzados

### 8.1 Autoencoder variacional (VAE)

El VAE añade una restricción: el espacio latente debe seguir una distribución gaussiana $\mathcal{N}(0, I)$. En lugar de un punto $\mathbf{z}$, el encoder produce $\boldsymbol{\mu}$ y $\boldsymbol{\sigma}^2$:

$$L_\text{VAE} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 + \text{KL}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

Esto produce un espacio latente continuo y suave, ideal para generación.

### 8.2 Autoencoders para reducción de dimensionalidad

Los autoencoders son una alternativa no lineal a PCA y t-SNE. Son especialmente útiles cuando la estructura de los datos es intrínsecamente no lineal (como dígitos manuscritos que viven en una variedad de baja dimensión).

### 8.3 Denoising autoencoders

Un **denoising autoencoder** recibe una entrada corrupta (con ruido añadido) y debe reconstruir la entrada limpia original. Esto fuerza a la red a aprender representaciones más robustas y menos propensas al overfitting.

### 8.4 Autoencoders en la práctica

| Aplicación | Tipo de autoencoder |
|-----------|-------------------|
| Compresión de imágenes | Convolucional profundo |
| Detección de anomalías | Reconstrucción con alto MSE = anomalía |
| Generación de imágenes | VAE, VQ-VAE |
| Aprendizaje de representaciones | Pre-entrenamiento para downstream tasks |
| Eliminación de ruido | Denoising autoencoder |

### 8.5 El manifold hypothesis

Los datos reales de alta dimensión (imágenes, texto) tienden a vivir cerca de **variedades** (manifolds) de baja dimensión. Los dígitos $8 \times 8$ tienen 64 dimensiones pero su variedad intrínseca es mucho menor. El autoencoder aprende una parametrización de esta variedad.

---

## 9. Ejercicios

### Ejercicio 1: Compresión mínima

¿Cuál es el bottleneck mínimo que permite reconstruir los 10 dígitos de forma reconocible? Prueba bottleneck = 2, 3, 4, 8 y anota el loss final tras 300 epochs en cada caso. ¿Hay un "codo" claro en la curva loss vs. dimensiones?

### Ejercicio 2: Mapa del espacio latente

Con bottleneck = 2 y red entrenada, carga cada dígito (0-9) y anota sus coordenadas $(z_1, z_2)$ del espacio latente. Dibuja un mapa en papel. ¿Qué dígitos son vecinos? ¿La organización tiene sentido visual (dígitos similares juntos)?

### Ejercicio 3: Interpolación imposible

Interpola entre el dígito "1" y el "0". ¿Hay un punto intermedio que parezca un "7"? ¿O la transición pasa por formas que no son dígitos reconocibles? ¿Qué dice esto sobre la geometría del espacio latente?

### Ejercicio 4: Robustez ante ruido

Dibuja un dígito limpio (ej. "5") y observa la reconstrucción. Luego añade píxeles aleatorios de ruido. ¿La reconstrucción sigue siendo un "5" limpio? Si sí, el autoencoder ha aprendido a filtrar ruido.

### Ejercicio 5: Fantasías con sentido

Con la red entrenada, genera 20 fantasías. ¿Cuántas son dígitos reconocibles? ¿Cuántas son híbridos o ruido? ¿Aumenta la proporción de dígitos reconocibles con más epochs de entrenamiento?

### Ejercicio 6: Contando parámetros

Para la arquitectura 64→32→2→32→64, calcula el número total de parámetros (pesos + biases). Verifica con el valor mostrado en métricas. ¿Cuántos parámetros tiene la versión con bottleneck = 16?

---

## 10. Glosario

| Término | Definición |
|---------|------------|
| **Autoencoder** | Red neuronal que aprende a comprimir y reconstruir datos, con forma de reloj de arena |
| **Encoder** | Parte de la red que comprime la entrada al espacio latente |
| **Decoder** | Parte de la red que reconstruye la entrada desde el espacio latente |
| **Bottleneck** | Capa intermedia de dimensionalidad reducida que fuerza la compresión |
| **Espacio latente** | Espacio de baja dimensión donde viven las representaciones comprimidas |
| **Código latente** ($\mathbf{z}$) | Vector comprimido de un dato en el espacio latente |
| **Error de reconstrucción** | $\|\mathbf{x} - \hat{\mathbf{x}}\|^2$; mide la calidad de la reconstrucción |
| **Interpolación** | Mezcla lineal entre dos códigos latentes: $\mathbf{z}_t = (1-t)\mathbf{z}_A + t\mathbf{z}_B$ |
| **Fantasía** | Dígito generado decodificando un punto aleatorio del espacio latente |
| **Ratio de compresión** | $\text{dim entrada} / \text{dim bottleneck}$ |
| **VAE** | Variational Autoencoder; impone distribución gaussiana en el espacio latente |
| **Manifold** | Variedad de baja dimensión donde viven los datos reales |
| **Denoising autoencoder** | Variante que recibe entrada con ruido y debe reconstruir la limpia |
| **PCA** | Principal Component Analysis; caso lineal especial del autoencoder |
| **Cluster** | Grupo de puntos cercanos en el espacio latente que corresponden a la misma clase |
| **Reconstrucción** ($\hat{\mathbf{x}}$) | Salida del decoder; intento de replicar la entrada original |
| **Representación** | Transformación aprendida de los datos que captura sus características esenciales |
| **Compresión con pérdida** | Reducción de dimensionalidad donde parte de la información se descarta |

---

## 11. Referencias

1. **Rumelhart, D. E., Hinton, G. E. & Williams, R. J.** (1986). "Learning Internal Representations by Error Propagation." *Parallel Distributed Processing*, Vol. 1, MIT Press.

2. **Hinton, G. E. & Salakhutdinov, R. R.** (2006). "Reducing the Dimensionality of Data with Neural Networks." *Science*, 313(5786), 504–507.

3. **Kingma, D. P. & Welling, M.** (2014). "Auto-Encoding Variational Bayes." *Proceedings of ICLR 2014*. arXiv:1312.6114.

4. **Vincent, P., Larochelle, H., Bengio, Y. & Manzagol, P.-A.** (2008). "Extracting and Composing Robust Features with Denoising Autoencoders." *Proceedings of ICML 2008*.

5. **Bengio, Y., Courville, A. & Vincent, P.** (2013). "Representation Learning: A Review and New Perspectives." *IEEE Transactions on PAMI*, 35(8), 1798–1828.

6. **Goodfellow, I., Bengio, Y. & Courville, A.** (2016). *Deep Learning*, Chapter 14: Autoencoders. MIT Press.
