# Self-Attention Mechanism — El Mecanismo que Revolucionó la IA

> Simulación interactiva del mecanismo de self-attention paso a paso. Observa cómo cada token de una secuencia genera sus vectores Query, Key y Value, cómo se calculan los scores de atención, y cómo el softmax produce los pesos que determinan a quién "atiende" cada palabra.

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

En 2017, el paper *"Attention Is All You Need"* de Vaswani et al. introdujo el **Transformer**, una arquitectura que reemplazó las redes recurrentes con un mecanismo aparentemente simple: cada token en una secuencia puede "mirar" directamente a todos los demás tokens y decidir cuánta atención prestarle a cada uno.

Este mecanismo — el **self-attention** — es la piedra angular de GPT, BERT, y prácticamente todos los modelos de lenguaje modernos. Su poder radica en que captura dependencias de largo alcance sin las limitaciones secuenciales de las RNNs.

Esta simulación descompone el self-attention en **7 pasos** visuales, desde los embeddings de entrada hasta la salida ponderada, mostrando exactamente cómo cada token decide a quién atender.

---

## 2. Conceptos Fundamentales

### 2.1 De secuencias a matrices

Cada token (palabra) se representa como un **vector de embedding** $x_i \in \mathbb{R}^{d}$. Para una secuencia de $n$ tokens, apilamos los vectores en una matriz:

$$X = [x_1, x_2, \ldots, x_n]^T \in \mathbb{R}^{n \times d}$$

### 2.2 Las tres proyecciones: Q, K, V

El self-attention proyecta cada embedding en tres espacios diferentes mediante matrices de pesos aprendidas:

**Query** (pregunta) — lo que el token busca:

$$Q = X W_Q, \quad W_Q \in \mathbb{R}^{d \times d_k}$$

**Key** (clave) — lo que el token ofrece como identificador:

$$K = X W_K, \quad W_K \in \mathbb{R}^{d \times d_k}$$

**Value** (valor) — la información que el token aporta:

$$V = X W_V, \quad W_V \in \mathbb{R}^{d \times d_v}$$

### 2.3 La intuición

Piensa en una biblioteca: el **Query** es tu pregunta, los **Keys** son las etiquetas de los libros, y los **Values** son los contenidos. Buscas los libros cuyas etiquetas más coincidan con tu pregunta, y lees proporcionalmente más de esos libros.

### 2.4 La fórmula completa

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

---

## 3. La Interfaz

### 3.1 Canvas principal

El canvas muestra los **7 pasos del self-attention** de forma secuencial:

1. **Input Embeddings** — vectores de entrada como barras coloreadas
2. **Proyección Q** — transformación a queries
3. **Proyección K** — transformación a keys
4. **$QK^T$** — matriz de compatibilidad
5. **$/\sqrt{d_k}$** — escalado
6. **Softmax** — normalización a probabilidades
7. **$\times V$** — output ponderado

Cada paso se ilumina al activarse. Los tokens aparecen como columnas, y las conexiones de atención se muestran como líneas cuyo grosor refleja el peso.

### 3.2 Panel lateral

- **Ecuación** destacada: $\text{Attention}(Q,K,V) = \text{softmax}(QK^T / \sqrt{d_k})V$
- **Indicador de paso**: 7 puntos navegables (inactivo/activo/completado)
- **Métricas**: dimensión $d_k$, entropía de atención, máximo peso, token query actual
- **Controles**: siguiente paso, animar, reset, selector de oración

### 3.3 Barra de estado

Muestra el paso actual (e.g., "Paso: 3/7") y el token query seleccionado.

---

## 4. Controles Interactivos

### 4.1 Secuencia de entrada

Selector con tres oraciones predefinidas:
- *"el gato se sienta en la mesa"* — 7 tokens, relaciones sintácticas claras
- *"la red neuronal aprende rapido"* — 5 tokens, vocabulario técnico
- *"atencion es todo lo que necesitas"* — 7 tokens, referencia al paper original

### 4.2 Navegación por pasos

- **Siguiente Paso** — avanza al siguiente de los 7 pasos
- **Animar** — reproduce todos los pasos automáticamente con transiciones
- **Reset** — vuelve al paso 1 con la secuencia actual
- **Step dots** — click en cualquier punto para saltar a ese paso

### 4.3 Configuración

| Control | Rango | Efecto |
|---------|-------|--------|
| Temperatura | 0.1 – 5.0 | Controla la "nitidez" del softmax. $T < 1$ concentra atención; $T > 1$ la distribuye |
| Causal Mask (GPT) | On/Off | Aplica máscara triangular: token $i$ solo atiende a posiciones $\leq i$ |
| Mostrar Valores | On/Off | Muestra/oculta los valores numéricos en las matrices |

### 4.4 Audio

- **Volumen** (0–100%) — controla la intensidad de la sonificación
- **Silenciar** — desactiva todo el audio

---

## 5. Las Matemáticas

### 5.1 Producto punto escalado

El score de compatibilidad entre los tokens $i$ y $j$ es:

$$e_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}$$

donde $q_i$ es la query del token $i$ y $k_j$ es la key del token $j$. La división por $\sqrt{d_k}$ (en la simulación, $d_k = 8$) evita que los productos punto crezcan demasiado con la dimensionalidad, lo que saturaría el softmax.

### 5.2 ¿Por qué $\sqrt{d_k}$?

Si $q$ y $k$ tienen componentes con media 0 y varianza 1, entonces:

$$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$

La desviación estándar es $\sqrt{d_k}$. Dividir por ella normaliza la varianza a 1, manteniendo los gradientes del softmax en un rango útil.

### 5.3 Softmax con temperatura

$$\alpha_{ij} = \frac{\exp(e_{ij} / T)}{\sum_{l=1}^{n} \exp(e_{il} / T)}$$

Con $T = 1$ (estándar), los pesos reflejan directamente las compatibilidades. Con $T \to 0$, la distribución se vuelve one-hot (atención concentrada). Con $T \to \infty$, la distribución se vuelve uniforme.

### 5.4 Máscara causal

Para modelos autoregresivos (GPT), el token en posición $i$ no debe "ver" posiciones futuras. Se aplica:

$$e_{ij} = \begin{cases} \frac{q_i \cdot k_j}{\sqrt{d_k}} & \text{si } j \leq i \\ -\infty & \text{si } j > i \end{cases}$$

El $-\infty$ se convierte en 0 después del softmax, bloqueando efectivamente la atención al futuro.

### 5.5 Entropía de atención

La simulación calcula la entropía de la distribución de atención para cada token:

$$H(\alpha_i) = -\sum_{j=1}^{n} \alpha_{ij} \log \alpha_{ij}$$

Entropía baja indica atención concentrada (el token "sabe" a quién mirar). Entropía alta indica atención dispersa.

---

## 6. Sonificación

### 6.1 Diseño de audio

La simulación usa un sintetizador con filtro lowpass para representar eventos:

- **Cambio de paso**: tono ascendente proporcional al número de paso
- **Pesos de atención**: cada conexión genera una nota cuya frecuencia y volumen reflejan el peso $\alpha_{ij}$
- **Temperatura**: temperaturas bajas producen sonidos más agudos y definidos; temperaturas altas generan texturas más difusas

### 6.2 Mapeo

| Evento | Frecuencia | Duración | Tipo de onda |
|--------|-----------|----------|-------------|
| Avance de paso | 300 + paso × 80 Hz | 0.15s | sine |
| Peso de atención | 200 + $\alpha_{ij}$ × 600 Hz | 0.3s | sine + lowpass |
| Reset | 200 Hz descendente | 0.2s | triangle |

---

## 7. Guía Paso a Paso

### Paso 1: Observa los embeddings

1. Selecciona *"el gato se sienta en la mesa"*
2. El paso 1 muestra los vectores de embedding como columnas coloreadas
3. Cada token tiene un vector de $d$ dimensiones (representado visualmente)

### Paso 2: Proyecciones Q

1. Avanza al paso 2
2. Los embeddings se transforman en **queries** mediante $W_Q$
3. Observa cómo los vectores cambian de forma: ahora representan "preguntas"

### Paso 3: Proyecciones K

1. Avanza al paso 3
2. Simultáneamente se generan los **keys** mediante $W_K$
3. Cada token ahora tiene un identificador de lo que "ofrece"

### Paso 4: Matriz $QK^T$

1. Avanza al paso 4 — este es el paso clave
2. Se calcula el producto punto entre cada par (query, key)
3. La matriz resultante muestra compatibilidades: valores altos = alta afinidad

### Paso 5: Escalado

1. Avanza al paso 5
2. Todos los scores se dividen por $\sqrt{d_k} = \sqrt{8} \approx 2.83$
3. Los valores se comprimen, preparándolos para el softmax

### Paso 6: Softmax

1. Avanza al paso 6
2. Cada fila se normaliza a probabilidades que suman 1
3. Activa la **Causal Mask** para ver cómo se enmascara el futuro

### Paso 7: Output ponderado

1. Avanza al paso 7
2. Los pesos de atención ponderan los **values** de cada token
3. El resultado es un nuevo vector por token que incorpora información contextual

### Paso 8: Experimenta con temperatura

1. Baja la temperatura a 0.2: la atención se concentra en 1-2 tokens
2. Sube la temperatura a 3.0: la atención se distribuye casi uniformemente
3. Observa cómo la entropía cambia con la temperatura

---

## 8. Conceptos Avanzados

### 8.1 Multi-Head Attention

En la práctica, el Transformer no usa una sola cabeza de atención sino $h$ cabezas en paralelo:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

donde $\text{head}_i = \text{Attention}(Q W_Q^i, K W_K^i, V W_V^i)$. Cada cabeza puede especializarse en diferentes tipos de relaciones (sintácticas, semánticas, posicionales).

### 8.2 Complejidad computacional

El self-attention tiene complejidad $O(n^2 d)$ donde $n$ es la longitud de la secuencia. Esto limita la longitud máxima de contexto y ha motivado variantes como:

- **Sparse Attention** (reducir conexiones)
- **Linear Attention** (aproximar softmax)
- **Flash Attention** (optimizar memoria GPU)

### 8.3 Cross-Attention

En modelos encoder-decoder (traducción), las queries vienen del decoder y las keys/values del encoder:

$$\text{CrossAttn} = \text{softmax}\!\left(\frac{Q_{\text{dec}} K_{\text{enc}}^T}{\sqrt{d_k}}\right) V_{\text{enc}}$$

Esto permite que el decoder "mire" selectivamente la entrada mientras genera la salida.

### 8.4 Encodings posicionales

El self-attention es invariante al orden (la operación es la misma sin importar la posición). Los **positional encodings** inyectan información de posición:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}), \quad PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

---

## 9. Ejercicios

### Ejercicio 1: Atención e identidad
Selecciona cada oración y avanza al paso 6. ¿El token "el"/"la" atiende más al sustantivo que le sigue? ¿Por qué tiene sentido lingüísticamente?

### Ejercicio 2: Efecto de la temperatura
Para la oración *"el gato se sienta en la mesa"*, registra el peso máximo de atención del token "gato" con $T = 0.5$, $T = 1.0$ y $T = 3.0$. Grafica los tres valores y describe la tendencia.

### Ejercicio 3: Máscara causal
Activa la causal mask y observa el paso 6. ¿Cuántos pesos de atención se anulan para el último token vs. el primer token? Explica la asimetría.

### Ejercicio 4: Entropía como indicador
Compara la entropía de atención del primer token vs. el cuarto token en cada oración. ¿Cuál tiene mayor entropía y por qué? Relaciona con la cantidad de contexto disponible (con mask causal activada).

### Ejercicio 5: Dimensionalidad
La simulación usa $d_k = 8$. Si se usara $d_k = 64$ (como en GPT-2), ¿por cuánto cambiaría el factor de escalado $\sqrt{d_k}$? ¿Qué efecto tendría en la "nitidez" de la atención?

### Ejercicio 6: Diseño de cabezas
Si un Transformer tiene $d = 512$ y usa $h = 8$ cabezas, ¿cuál es $d_k$ por cabeza? Si duplicamos $h$ a 16, ¿qué trade-off se introduce entre especialización y expresividad por cabeza?

---

## 10. Glosario

| Término | Definición |
|---------|-----------|
| Self-Attention | Mecanismo donde cada token atiende a todos los tokens de la misma secuencia |
| Query (Q) | Vector que representa "lo que busca" un token |
| Key (K) | Vector que representa "lo que ofrece" un token como identificador |
| Value (V) | Vector con la información que un token contribuye al output |
| Score | Producto punto entre query y key antes de normalizar |
| Escalado | División por $\sqrt{d_k}$ para estabilizar gradientes del softmax |
| Softmax | Función que convierte scores en probabilidades (suman 1) |
| Peso de atención ($\alpha_{ij}$) | Probabilidad de que token $i$ atienda a token $j$ |
| Temperatura | Parámetro que controla la nitidez de la distribución de atención |
| Máscara causal | Restricción que impide atender a posiciones futuras (modelos autoregresivos) |
| Multi-Head Attention | Múltiples cabezas de atención en paralelo, cada una con sus propias $W_Q, W_K, W_V$ |
| Cross-Attention | Atención donde Q viene de una secuencia y K, V de otra |
| Positional Encoding | Vector sumado al embedding para codificar la posición del token |
| Entropía | Medida de dispersión de la distribución de atención |
| $d_k$ | Dimensionalidad de los vectores query y key |
| $d_v$ | Dimensionalidad de los vectores value |
| Transformer | Arquitectura basada enteramente en mecanismos de atención |
| Autoregresivo | Modelo que genera tokens uno a uno, condicionado en los anteriores |

---

## 11. Referencias

1. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
2. Phuong, M. & Hutter, M. (2022). *Formal Algorithms for Transformers*. arXiv:2207.09238.
3. Dao, T., et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention*. NeurIPS.
4. Jay Alammar (2018). *The Illustrated Transformer*. jalammar.github.io.
5. Lilian Weng (2023). *Attention? Attention!*. lilianweng.github.io.
