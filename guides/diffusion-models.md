# Diffusion Models — De Ruido a Arte

> Simulación interactiva del proceso de difusión: observa cómo una imagen limpia se destruye progresivamente con ruido gaussiano (forward process) y cómo una red neuronal aprende a reconstruirla paso a paso (reverse process). Controla el schedule de ruido, el número de pasos, y explora la magia detrás de DALL-E, Stable Diffusion y Midjourney.

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

Los **modelos de difusión** son la tecnología detrás de la revolución en generación de imágenes. Su principio es sorprendentemente simple: si sabes cómo destruir algo (añadir ruido), puedes aprender a reconstruirlo (quitar ruido).

El proceso tiene dos fases:
1. **Forward** (destrucción): se añade ruido gaussiano progresivamente a una imagen hasta convertirla en ruido puro
2. **Reverse** (creación): una red neuronal aprende a revertir cada paso, reconstruyendo la imagen desde el ruido

Esta simulación trabaja con imágenes de 16×16 píxeles — lo suficientemente pequeñas para ver cada píxel, pero lo suficientemente complejas para apreciar cómo el ruido destruye y se revierte la estructura.

---

## 2. Conceptos Fundamentales

### 2.1 El proceso forward

Dado una imagen limpia $x_0$, el proceso forward añade ruido en $T$ pasos:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}\, x_{t-1}, \beta_t I)$$

donde $\beta_t$ es el **schedule de ruido** que controla cuánto ruido se añade en cada paso.

### 2.2 La fórmula cerrada

Gracias a las propiedades de la gaussiana, podemos saltar directamente al paso $t$ sin calcular todos los intermedios:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

donde $\alpha_t = 1 - \beta_t$ y $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ es el producto acumulado.

### 2.3 El proceso reverse

El objetivo es aprender una red $\epsilon_\theta(x_t, t)$ que predice el ruido añadido:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

La media se computa como:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

### 2.4 Imágenes objetivo

La simulación ofrece 4 targets de 16×16:
- **Dígito 3** — patrón curvo
- **Cruz** — líneas rectas ortogonales
- **Círculo** — simetría radial
- **Corazón** — curva paramétrica $(x^2 + y^2 - 1)^3 - x^2 y^3 < 0$

---

## 3. La Interfaz

### 3.1 Canvas principal

El canvas muestra la imagen actual como una cuadrícula de 16×16 píxeles:

- **Píxeles brillantes**: valor alto (parte de la imagen)
- **Píxeles oscuros**: valor bajo (fondo)
- **Ruido**: valores intermedios aleatorios
- **Transición visual**: se ve cómo la estructura emerge o desaparece paso a paso

Se muestran además gráficas del schedule de ruido ($\beta_t$ y $\bar{\alpha}_t$) y la señal vs. ruido a lo largo de los pasos.

### 3.2 Panel lateral

- **Ecuación**: $x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon$
- **Selector de target**: dígito 3, cruz, círculo, corazón
- **Sliders**: pasos totales $T$, paso actual $t$
- **Selector de schedule**: lineal o coseno
- **Métricas**: paso actual, varianza del ruido, SNR (dB), entropía

### 3.3 Barra de estado

Muestra "Paso: t/T" y la varianza del ruido actual.

---

## 4. Controles Interactivos

### 4.1 Selección de imagen

Dropdown con 4 targets. Cada target tiene estructura geométrica diferente, lo que afecta cómo el ruido interactúa con la señal.

### 4.2 Parámetros de difusión

| Control | Rango | Efecto |
|---------|-------|--------|
| Pasos T | 10 – 100 | Total de pasos de difusión |
| Paso actual | 0 – T | Posición en la cadena (0 = limpia, T = ruido puro) |
| Schedule | Lineal / Coseno | Cómo se distribuye el ruido a lo largo de los pasos |

### 4.3 Botones de acción

| Botón | Acción |
|-------|--------|
| Denoise (Reverse) | Anima el proceso inverso: de ruido a imagen |
| Añadir Ruido (Forward) | Anima el proceso forward: de imagen a ruido |
| Reset | Restaura la imagen limpia en paso 0 |

### 4.4 Audio

- **Volumen** (0–100%)
- **Silenciar** — desactiva audio

---

## 5. Las Matemáticas

### 5.1 Schedule lineal

El schedule lineal distribuye $\beta_t$ uniformemente:

$$\beta_t = \beta_{\min} + \frac{t-1}{T-1}(\beta_{\max} - \beta_{\min})$$

Típicamente $\beta_{\min} = 10^{-4}$ y $\beta_{\max} = 0.02$. Esto destruye la imagen rápidamente al inicio.

### 5.2 Schedule coseno

Propuesto por Nichol & Dhariwal (2021), preserva más estructura al inicio:

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

donde $s = 0.008$ es un offset para evitar $\beta_t$ demasiado pequeño en $t = 0$.

### 5.3 Signal-to-Noise Ratio (SNR)

El SNR en el paso $t$ mide cuánta señal queda respecto al ruido:

$$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$$

En decibeles: $\text{SNR}_{\text{dB}}(t) = 10 \log_{10} \text{SNR}(t)$. A $t = 0$, SNR = $\infty$ (imagen pura). A $t = T$, SNR $\approx 0$ (ruido puro).

### 5.4 Función de pérdida

La red de denoising se entrena minimizando:

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

Es decir: predice el ruido que se añadió, y minimiza el error cuadrático medio.

### 5.5 Varianza y entropía

La simulación calcula en tiempo real:
- **Varianza del ruido**: $\sigma^2_{\text{noise}} = 1 - \bar{\alpha}_t$
- **Entropía**: medida de desorden en la imagen, calculada sobre el histograma de valores de píxel

---

## 6. Sonificación

### 6.1 Diseño de audio

El audio mapea el estado de la difusión a sonido:

- **Proceso forward**: frecuencia descendente (más grave = más ruido)
- **Proceso reverse**: frecuencia ascendente (más agudo = más señal)
- **Cada paso**: click percusivo cuyo timbre cambia con el SNR
- **Filtro lowpass**: frecuencia de corte proporcional a $\bar{\alpha}_t$ (más señal = más brillo)

### 6.2 Mapeo

| Estado | Frecuencia | Filtro LP | Tipo |
|--------|-----------|----------|------|
| $t = 0$ (limpia) | 800 Hz | 3000 Hz | sine |
| $t = T/2$ (parcial) | 400 Hz | 1500 Hz | sine |
| $t = T$ (ruido) | 150 Hz | 500 Hz | noise-like |

---

## 7. Guía Paso a Paso

### Paso 1: Observar la imagen limpia

1. Selecciona "Dígito 3" como target
2. El canvas muestra la imagen limpia (paso $t = 0$)
3. Observa la estructura clara: píxeles brillantes forman el dígito

### Paso 2: Proceso forward paso a paso

1. Usa el slider de "Paso actual" para mover de 0 a T gradualmente
2. Observa cómo el ruido corrompe la imagen progresivamente
3. Nota el punto donde el dígito deja de ser reconocible

### Paso 3: Forward animado

1. Pulsa **Añadir Ruido (Forward)**
2. La animación avanza de $t = 0$ a $t = T$
3. Observa las métricas: SNR cae, varianza sube, entropía aumenta

### Paso 4: Reverse (denoising)

1. Con la imagen en ruido puro, pulsa **Denoise (Reverse)**
2. Observa cómo la estructura emerge paso a paso
3. Primero aparecen las formas globales, luego los detalles

### Paso 5: Comparar schedules

1. Con schedule lineal, ejecuta forward + reverse
2. Cambia a schedule coseno y repite
3. Nota cómo el coseno preserva más estructura en los primeros pasos

### Paso 6: Cambiar targets

1. Prueba con cruz, círculo y corazón
2. Observa que formas con más simetría son más fáciles de reconstruir
3. El corazón (curva suave) vs. la cruz (ángulos rectos) muestran diferencias

---

## 8. Conceptos Avanzados

### 8.1 Denoising Diffusion Probabilistic Models (DDPM)

El framework original de Ho et al. (2020) fija la varianza del reverse process y entrena solo la media. La pérdida simplificada es equivalente a predecir el ruido $\epsilon$.

### 8.2 DDIM (Denoising Diffusion Implicit Models)

Song et al. (2021) mostraron que se puede hacer el sampling **determinista** y con menos pasos, usando la misma red entrenada:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-1}}\, \epsilon_\theta$$

### 8.3 Guidance (Classifier-Free)

Para generar imágenes condicionadas en texto, se usa:

$$\tilde{\epsilon}_\theta = \epsilon_\theta(x_t, \varnothing) + w \cdot [\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)]$$

donde $c$ es el prompt de texto y $w$ es la escala de guidance. $w > 1$ amplifica la señal del texto, produciendo imágenes más fieles al prompt.

### 8.4 Latent Diffusion (Stable Diffusion)

En lugar de difundir en el espacio de píxeles, se trabaja en el espacio latente de un autoencoder:

$$z = \text{Encoder}(x), \quad \text{difusión sobre } z, \quad x = \text{Decoder}(\hat{z})$$

Esto reduce la dimensionalidad y el costo computacional dramáticamente.

---

## 9. Ejercicios

### Ejercicio 1: Punto de no retorno
Para cada target, encuentra el paso $t^*$ donde la imagen deja de ser reconocible visualmente (usa el slider). ¿Es $t^*$ similar para todos los targets? Registra el SNR en ese punto.

### Ejercicio 2: Lineal vs. coseno
Fija $T = 50$. Para el dígito 3, compara el SNR en $t = 10$ entre schedule lineal y coseno. ¿Cuál preserva más señal en pasos tempranos? Calcula $\bar{\alpha}_{10}$ para ambos schedules.

### Ejercicio 3: Efecto de T
Ejecuta el reverse process con $T = 10$, $T = 50$ y $T = 100$ para el mismo target. ¿La calidad de reconstrucción mejora con más pasos? ¿Hay rendimientos decrecientes?

### Ejercicio 4: Simetría y reconstrucción
Compara la reconstrucción (reverse) del círculo vs. el dígito 3. ¿Cuál se reconstruye mejor con menos pasos ($T = 10$)? Relaciona con la complejidad geométrica del target.

### Ejercicio 5: Calcular $\bar{\alpha}_t$
Para el schedule lineal con $\beta_{\min} = 0.0001$, $\beta_{\max} = 0.02$ y $T = 50$, calcula manualmente $\beta_1$, $\beta_{25}$ y $\beta_{50}$. Luego calcula $\bar{\alpha}_{50}$ (el producto acumulado). ¿Cuánta señal queda?

### Ejercicio 6: SNR y percepción
Registra el SNR (dB) en pasos $t = 0, T/4, T/2, 3T/4, T$ para cualquier target con schedule lineal. Grafica SNR vs. $t$. ¿La curva es lineal? ¿Qué forma tiene?

---

## 10. Glosario

| Término | Definición |
|---------|-----------|
| Difusión | Proceso estocástico de añadir ruido gaussiano progresivamente |
| Forward process | Fase de destrucción: imagen → ruido |
| Reverse process | Fase de creación: ruido → imagen |
| $\beta_t$ (beta) | Varianza del ruido añadido en el paso $t$ |
| $\alpha_t$ | $1 - \beta_t$, fracción de señal retenida en un paso |
| $\bar{\alpha}_t$ | Producto acumulado $\prod \alpha_s$, señal total retenida hasta $t$ |
| Schedule | Función que define cómo $\beta_t$ varía con $t$ |
| SNR | Signal-to-Noise Ratio: $\bar{\alpha}_t / (1 - \bar{\alpha}_t)$ |
| Denoiser | Red neuronal que predice el ruido para revertir un paso |
| $\epsilon$ (epsilon) | Ruido gaussiano estándar $\mathcal{N}(0, I)$ |
| DDPM | Denoising Diffusion Probabilistic Model (Ho et al., 2020) |
| DDIM | Variante determinista que permite sampling con menos pasos |
| Guidance | Técnica para condicionar la generación en texto u otra señal |
| Latent Diffusion | Difusión en el espacio latente de un autoencoder |
| Entropía | Medida de desorden en la distribución de píxeles |
| Varianza | Dispersión de los valores de píxel respecto a la media |
| Ruido gaussiano | Ruido con distribución normal (campana de Gauss) |
| Markov chain | Cadena donde cada paso depende solo del estado anterior |

---

## 11. Referencias

1. Ho, J., Jain, A. & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.
2. Nichol, A. & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models*. ICML.
3. Song, J., Meng, C. & Ermon, S. (2021). *Denoising Diffusion Implicit Models*. ICLR.
4. Rombach, R., et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR.
5. Lilian Weng (2021). *What are Diffusion Models?*. lilianweng.github.io.
