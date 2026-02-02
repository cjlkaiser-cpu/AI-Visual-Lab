# Activation Function Orchestra — 8 Funciones en Paralelo

> Simulación interactiva que muestra 8 funciones de activación procesando la misma señal de entrada en tiempo real. Observa cómo Sigmoid satura, ReLU corta, GELU filtra suavemente y Mish se auto-regula — todo mientras escuchas la orquesta sonora que producen.

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

Las **funciones de activación** son la fuente de no-linealidad en las redes neuronales. Sin ellas, una red de cualquier profundidad sería equivalente a una sola transformación lineal — incapaz de aprender patrones complejos como bordes en imágenes, sentimiento en texto, o estructura en datos.

Esta simulación muestra **8 funciones de activación** procesando simultáneamente la misma señal de entrada. Cada función tiene su propia celda tipo osciloscopio, donde puedes observar en tiempo real cómo transforma la señal, dónde satura, dónde corta, y cómo se comporta su derivada. Cambiando la amplitud de la entrada, puedes provocar saturación masiva en Sigmoid y Tanh mientras ReLU y GELU continúan operando.

La metáfora de la "orquesta" es literal: cada función produce un sonido diferente, y el conjunto forma un paisaje sonoro que cambia con la señal de entrada.

---

## 2. Conceptos Fundamentales

### 2.1 ¿Por qué necesitamos no-linealidad?

Sin funciones de activación, una red de $L$ capas computa:

$$f(\mathbf{x}) = W_L \cdot W_{L-1} \cdots W_1 \cdot \mathbf{x} = W_{\text{total}} \cdot \mathbf{x}$$

El producto de matrices lineales es otra matriz lineal. No importa cuántas capas apiles: la red solo puede representar transformaciones lineales. Las funciones de activación $\sigma$ rompen esta limitación:

$$f(\mathbf{x}) = \sigma(W_L \cdot \sigma(W_{L-1} \cdots \sigma(W_1 \cdot \mathbf{x})))$$

### 2.2 Propiedades deseables

Una buena función de activación debería:

- Ser **no lineal** (de lo contrario no aporta nada)
- Tener **derivada no nula** en un rango amplio (para que los gradientes fluyan)
- Ser **computacionalmente eficiente** (se evalúa millones de veces)
- Producir **salidas centradas en cero** o cercanas (facilita la optimización)

### 2.3 La derivada importa

En backpropagation, el gradiente que fluye a través de una neurona se multiplica por $\sigma'(z)$. Si la derivada es:

- **Siempre $< 1$**: los gradientes se desvanecen exponencialmente (vanishing)
- **Exactamente 0**: la neurona no recibe señal de error (zona muerta)
- **Siempre 1**: los gradientes fluyen perfectamente (ideal)

---

## 3. La Interfaz

### 3.1 Estructura general

| Área | Ubicación | Función |
|------|-----------|---------|
| **Banda de entrada** | Parte superior del canvas | Señal de entrada como osciloscopio horizontal |
| **Cuadrícula 4×2** | Área principal del canvas | 8 celdas, una por función de activación |
| **Tooltip** | Sobre celda hovereada | Salida, derivada, estado de saturación |
| **Panel de controles** | Derecha (360px) | Señal, visualización, audio, leyenda |

### 3.2 La banda de entrada

Una tira horizontal dorada muestra la señal de entrada actual (seno, cuadrada, etc.) en formato osciloscopio: el tiempo fluye de izquierda a derecha, con un marcador en el valor actual.

### 3.3 Las 8 celdas

Cada celda muestra:

- **Línea sólida** (color de la función): salida $\sigma(x(t))$ a lo largo del tiempo
- **Línea punteada** (si está activada): derivada $\sigma'(x(t))$
- **Relleno bajo curva** (si está activado): área sombreada entre la curva y el eje
- **Nombre de la función**: esquina superior izquierda
- **Resaltado al hover**: borde iluminado en la celda activa

### 3.4 Código de colores

| Función | Color |
|---------|-------|
| Sigmoid | Rojo (#ef4444) |
| Tanh | Azul (#3b82f6) |
| ReLU | Verde (#22c55e) |
| Leaky ReLU | Ámbar (#f59e0b) |
| ELU | Violeta (#a855f7) |
| Swish | Cyan (#06b6d4) |
| GELU | Rosa (#ec4899) |
| Mish | Lima (#84cc16) |

---

## 4. Controles Interactivos

### 4.1 Señal de entrada

| Tipo | Forma | Uso didáctico |
|------|-------|---------------|
| **Seno** | Onda sinusoidal suave | Caso base: transición gradual entre regiones |
| **Cuadrada** | Saltos entre $+A$ y $-A$ | Testea respuesta a transiciones abruptas |
| **Triángulo** | Rampa lineal simétrica | Entrada linealmente creciente y decreciente |
| **Rampa** (sawtooth) | Rampa ascendente con reset | Muestra comportamiento asimétrico |
| **Ruido** | Valores aleatorios | Simula entradas ruidosas realistas |
| **Chirp** | Seno con frecuencia creciente | Muestra comportamiento a diferentes velocidades |

**Presets de amplitud**:
- **Explosión** ($A = 10$): satura las funciones acotadas
- **Suave** ($A = 0.5$): todas las funciones se comportan casi linealmente
- **Normal** ($A = 2$): rango típico de pre-activaciones

### 4.2 Parámetros

| Control | Rango | Default | Efecto |
|---------|-------|---------|--------|
| Frecuencia | 0.1 – 5.0 Hz | 1.0 | Velocidad de oscilación de la entrada |
| Amplitud | 0.1 – 10.0 | 2.0 | Rango de la señal; amplitudes altas causan saturación |
| Velocidad | 0.1 – 5.0x | 1.0 | Escala temporal de la simulación |

### 4.3 Opciones de visualización

| Opción | Descripción |
|--------|-------------|
| **Mostrar derivadas** | Superpone la derivada $\sigma'(x)$ como línea punteada |
| **Mostrar curva estática** | Muestra $\sigma(x)$ como función estática (no temporal) |
| **Relleno bajo curva** | Sombrea el área entre la curva y el eje horizontal |

### 4.4 Hover sobre celdas

Al pasar el mouse sobre una celda, el tooltip muestra:
- Valor de salida actual
- Valor de la derivada actual
- Estado: "Activo" o "Saturado" (derivada < 0.05 cuando $|x| > 1$)
- Descripción breve de la función

---

## 5. Las Matemáticas

### 5.1 Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}, \quad \sigma'(x) = \sigma(x)(1 - \sigma(x))$$

- **Rango**: $(0, 1)$
- **Derivada máxima**: $0.25$ en $x = 0$
- **Problema**: satura para $|x| > 4$, causando vanishing gradients
- **Uso**: capa de salida en clasificación binaria, gates en LSTM

### 5.2 Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}, \quad \tanh'(x) = 1 - \tanh^2(x)$$

- **Rango**: $(-1, 1)$
- **Derivada máxima**: $1$ en $x = 0$
- **Ventaja sobre sigmoid**: centrada en cero, derivada más alta
- **Problema**: aún satura para $|x| > 3$
- **Uso**: celdas LSTM, redes recurrentes clásicas

### 5.3 ReLU (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x), \quad \text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$$

- **Rango**: $[0, \infty)$
- **Derivada**: constante 1 o 0
- **Ventaja**: no satura para $x > 0$, computacionalmente trivial
- **Problema**: "neuronas muertas" — si una neurona entra en $x < 0$ permanentemente, deja de aprender
- **Uso**: estándar en redes convolucionales y feed-forward

### 5.4 Leaky ReLU

$$\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ 0.01x & x \leq 0 \end{cases}$$

- **Rango**: $(-\infty, \infty)$
- **Derivada**: $1$ o $0.01$ (nunca exactamente 0)
- **Ventaja**: elimina el problema de neuronas muertas
- **Uso**: alternativa robusta a ReLU

### 5.5 ELU (Exponential Linear Unit)

$$\text{ELU}(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}, \quad \alpha = 1$$

$$\text{ELU}'(x) = \begin{cases} 1 & x > 0 \\ \alpha e^x & x \leq 0 \end{cases}$$

- **Rango**: $(-\alpha, \infty)$
- **Ventaja**: suave en $x = 0$, salidas con media cercana a cero
- **Uso**: redes donde se busca convergencia más rápida

### 5.6 Swish

$$\text{Swish}(x) = x \cdot \sigma(x)$$

$$\text{Swish}'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x))$$

- **Rango**: $\approx (-0.28, \infty)$
- **Propiedad**: suave, no monótona (tiene un mínimo negativo cerca de $x \approx -1.28$)
- **Ventaja**: auto-gated — la sigmoid actúa como compuerta que deja pasar más o menos señal
- **Uso**: redes de búsqueda de arquitectura (NAS), EfficientNet

### 5.7 GELU (Gaussian Error Linear Unit)

$$\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)$$

- **Rango**: $\approx (-0.17, \infty)$
- **Propiedad**: suave, no monótona, basada en la función de distribución gaussiana $\Phi$
- **Interpretación**: la probabilidad de que una variable gaussiana sea menor que $x$ pondera la entrada
- **Uso**: **estándar en Transformers** (BERT, GPT, ViT)

### 5.8 Mish

$$\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))$$

- **Rango**: $\approx (-0.31, \infty)$
- **Propiedad**: suave, no monótona, auto-regularizante
- **Ventaja**: gradientes que fluyen mejor que Swish en redes muy profundas
- **Uso**: YOLOv4, redes de detección de objetos

---

## 6. Sonificación

### 6.1 La orquesta

Cada función de activación tiene asignado un **oscilador continuo** con frecuencia y timbre distintos:

| Función | Frecuencia base | Forma de onda | Posición estéreo |
|---------|----------------|---------------|-----------------|
| Sigmoid | 110 Hz | Sine | Izquierda |
| Tanh | 220 Hz | Sine | Centro-izq |
| ReLU | 330 Hz | Sawtooth | Centro-izq |
| Leaky ReLU | 440 Hz | Sawtooth | Centro |
| ELU | 550 Hz | Triangle | Centro |
| Swish | 660 Hz | Triangle | Centro-der |
| GELU | 770 Hz | Triangle | Centro-der |
| Mish | 880 Hz | Triangle | Derecha |

### 6.2 Modulación por salida

El **volumen** de cada oscilador es proporcional a $|\sigma(x)|$: funciones con salida alta suenan más fuerte, las que están en zona muerta se silencian.

### 6.3 Detuning por saturación

Cuando la derivada es baja ($\sigma'(x) < 0.1$), el oscilador se **desafina** hasta 25 centésimas. Esto crea una sensación de "disonancia" audible que representa la saturación — puedes *escuchar* cuándo una función deja de ser útil para el gradiente.

---

## 7. Guía Paso a Paso

### Experimento 1: Comparación visual básica

1. Mantén la configuración por defecto (seno, amplitud 2.0)
2. Observa las 8 celdas simultáneamente
3. **Sigmoid** y **Tanh** producen curvas suaves y acotadas
4. **ReLU** corta la mitad negativa — la onda se "rectifica"
5. **Swish**, **GELU** y **Mish** son intermedias: suaves pero no acotadas
6. Activa "Mostrar derivadas" para ver dónde $\sigma'$ es alta y dónde cae a cero

### Experimento 2: Saturación con amplitud alta

1. Click en **"Explosión"** ($A = 10$)
2. Observa Sigmoid: se convierte en una onda cuadrada (saturación total en 0 y 1)
3. Tanh hace lo mismo: satura en -1 y 1
4. ReLU y Leaky ReLU **no saturan** — la salida crece linealmente
5. Mira las derivadas punteadas: en Sigmoid y Tanh, $\sigma'$ es prácticamente 0 excepto en transiciones
6. **Conclusión**: con señales grandes, Sigmoid y Tanh no transmiten gradientes

### Experimento 3: Régimen lineal

1. Click en **"Suave"** ($A = 0.5$)
2. Todas las funciones producen salidas muy similares — casi lineales
3. Las derivadas son altas y estables
4. **Conclusión**: cerca del origen, todas las activaciones se comportan de forma parecida. Las diferencias aparecen con señales fuertes.

### Experimento 4: Señales abruptas

1. Selecciona señal **"Cuadrada"** con amplitud normal
2. La señal de entrada salta entre $+2$ y $-2$
3. Sigmoid y Tanh suavizan los saltos (tienen inercia)
4. ReLU produce una onda cuadrada rectificada (solo la parte positiva)
5. Observa la derivada de ReLU: salta entre 0 y 1 instantáneamente

### Experimento 5: GELU vs. ReLU

1. Señal seno, amplitud 2.0
2. Compara la celda de ReLU (verde) con GELU (rosa)
3. ReLU corta abruptamente en $x = 0$ — la derivada es discontinua
4. GELU tiene una transición suave — permite un pequeño rango negativo
5. Activa las derivadas: la de ReLU salta de 0 a 1; la de GELU es una curva suave
6. **Esto explica por qué GELU funciona mejor en Transformers**: gradientes más suaves

---

## 8. Conceptos Avanzados

### 8.1 La muerte de neuronas ReLU

Cuando una neurona con ReLU recibe consistentemente entradas negativas (porque $w \cdot x + b < 0$ para todos los datos del batch), su salida es 0 y su gradiente es 0. No recibe señal de error y nunca se recupera. Esto se llama el problema de **"dying ReLU"**.

Soluciones: Leaky ReLU, ELU, o inicialización cuidadosa de pesos.

### 8.2 La "no-monotonicidad" de Swish, GELU y Mish

Estas tres funciones tienen un comportamiento peculiar: para $x$ ligeramente negativo, la salida es negativa y *menor* que la entrada. Esto crea un "valle" que actúa como regularización implícita — penaliza ligeramente las activaciones pequeñas pero negativas.

El mínimo de Swish está en $x \approx -1.28$ con valor $\approx -0.28$.

### 8.3 La aproximación de GELU

GELU se define como $x \cdot \Phi(x)$ donde $\Phi$ es la CDF de la distribución normal estándar. La implementación usa la aproximación:

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)$$

Esta aproximación es extremadamente precisa (error < $10^{-4}$) y más rápida que evaluar la función de error.

### 8.4 Funciones de activación en diferentes arquitecturas

| Arquitectura | Activación típica | Razón |
|-------------|-------------------|-------|
| CNN (ResNet, VGG) | ReLU | Simple, rápida, no satura |
| Transformers (GPT, BERT) | GELU | Suave, mejor rendimiento empírico |
| LSTM / GRU | Sigmoid (gates) + Tanh (estado) | Gates necesitan $[0,1]$; estado centrado en cero |
| GANs | Leaky ReLU | Evita neuronas muertas en el discriminador |
| Detection (YOLO) | Mish / SiLU | Mejor gradiente en redes muy profundas |

### 8.5 Softplus como precursor

$$\text{Softplus}(x) = \ln(1 + e^x)$$

Softplus es una versión suave de ReLU. Es la base de Mish ($x \cdot \tanh(\text{softplus}(x))$). Aunque teóricamente elegante, en la práctica ReLU suele funcionar mejor por su simplicidad y derivada exacta.

---

## 9. Ejercicios

### Ejercicio 1: Clasificando saturación

Con señal seno y amplitud 5.0, clasifica las 8 funciones en tres categorías: "satura completamente", "satura parcialmente" y "no satura". Verifica tu clasificación observando las derivadas.

### Ejercicio 2: Reconstrucción de la entrada

¿Desde qué función podrías reconstruir la señal de entrada original? Con amplitud normal, compara las salidas de ReLU y Leaky ReLU. ¿Cuál preserva más información? ¿Es posible reconstruir $x$ a partir de $\text{ReLU}(x)$? ¿Y a partir de $\text{LeakyReLU}(x)$?

### Ejercicio 3: Sensibilidad a la amplitud

Varía la amplitud de 0.1 a 10 en incrementos de 0.5. Para cada valor, anota si Sigmoid está: (a) en régimen lineal, (b) parcialmente saturada, (c) totalmente saturada. ¿En qué rango de amplitud cambia el comportamiento?

### Ejercicio 4: GELU manual

La fórmula de GELU usa $\Phi(x) \approx 0.5(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$. Calcula $\text{GELU}(-1)$, $\text{GELU}(0)$ y $\text{GELU}(2)$ a mano. Verifica tus resultados observando los valores en el tooltip.

### Ejercicio 5: Escuchando la saturación

Activa el audio y usa señal seno con amplitud normal. Gradualmente sube la amplitud a 10. ¿En qué punto empiezas a notar la desafinación (detuning) en los osciladores? ¿Cuáles funciones se desafinan primero? ¿Cuáles mantienen el tono?

### Ejercicio 6: Derivada máxima

Para cada una de las 8 funciones, observa la derivada con señal seno y amplitud 2.0. ¿Cuál tiene la derivada máxima más alta? ¿Cuál tiene la derivada más estable (menos variación)? Ordena las 8 funciones de "mejor para gradientes" a "peor para gradientes".

---

## 10. Glosario

| Término | Definición |
|---------|------------|
| **Función de activación** | Función no lineal $\sigma(z)$ aplicada a la pre-activación de cada neurona |
| **No-linealidad** | Propiedad que permite a las redes neuronales representar funciones complejas |
| **Saturación** | Estado donde $\|\sigma'(z)\| \approx 0$; los gradientes no fluyen a través de la neurona |
| **Zona muerta** | Región de la entrada donde la salida y la derivada son cero (ej. $x < 0$ en ReLU) |
| **Neurona muerta** | Neurona que siempre produce 0 y no se recupera; problema de ReLU |
| **Sigmoid** | $\sigma(x) = 1/(1+e^{-x})$; acota a $(0,1)$, satura |
| **Tanh** | $\tanh(x)$; centrada en cero, acota a $(-1,1)$, satura |
| **ReLU** | $\max(0,x)$; simple, no satura para $x > 0$, tiene zona muerta |
| **Leaky ReLU** | $\max(0.01x, x)$; ReLU con pendiente pequeña en $x < 0$ |
| **ELU** | Exponential Linear Unit; suave en $x = 0$, media cercana a cero |
| **Swish** | $x \cdot \sigma(x)$; auto-gated, no monótona, suave |
| **GELU** | $x \cdot \Phi(x)$; estándar en Transformers, suave probabilística |
| **Mish** | $x \cdot \tanh(\text{softplus}(x))$; auto-regularizante, no monótona |
| **Derivada de activación** ($\sigma'$) | Factor por el que se multiplica el gradiente al atravesar la neurona |
| **Detuning** | Desafinación del oscilador proporcional a la saturación |
| **Softplus** | $\ln(1 + e^x)$; versión suave de ReLU, base de Mish |
| **Auto-gating** | Propiedad de Swish/GELU donde la función controla su propia magnitud |
| **CDF gaussiana** ($\Phi$) | Función de distribución acumulada de la normal estándar; base de GELU |

---

## 11. Referencias

1. **Nair, V. & Hinton, G. E.** (2010). "Rectified Linear Units Improve Restricted Boltzmann Machines." *Proceedings of ICML 2010*.

2. **Clevert, D.-A., Unterthiner, T. & Hochreiter, S.** (2016). "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)." *Proceedings of ICLR 2016*. arXiv:1511.07289.

3. **Ramachandran, P., Zoph, B. & Le, Q. V.** (2018). "Searching for Activation Functions." *ICLR 2018 Workshop*. arXiv:1710.05941.

4. **Hendrycks, D. & Gimpel, K.** (2016). "Gaussian Error Linear Units (GELUs)." arXiv:1606.08415.

5. **Misra, D.** (2019). "Mish: A Self Regularized Non-Monotonic Activation Function." arXiv:1908.08681.

6. **Glorot, X., Bordes, A. & Bengio, Y.** (2011). "Deep Sparse Rectifier Neural Networks." *Proceedings of AISTATS 2011*.

7. **Goodfellow, I., Bengio, Y. & Courville, A.** (2016). *Deep Learning*, Section 6.3: Hidden Units. MIT Press.

8. **Maas, A. L., Hannun, A. Y. & Ng, A. Y.** (2013). "Rectifier Nonlinearities Improve Neural Network Acoustic Models." *ICML Workshop on Deep Learning for Audio*.
