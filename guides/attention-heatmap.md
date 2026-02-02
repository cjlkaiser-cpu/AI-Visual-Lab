# Attention Heatmap — Visualizando Quién Mira a Quién

> Simulación interactiva del mecanismo de atención multi-head (Scaled Dot-Product Attention). Escribe una frase, configura el número de heads y la dimensión de las keys, y observa en un heatmap cómo cada token distribuye su atención sobre los demás. Explora cómo la temperatura del softmax concentra o dispersa la atención.

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

El **mecanismo de atención** es el corazón de los Transformers, la arquitectura que impulsa GPT, BERT, y la mayoría de modelos de lenguaje modernos. Su idea central es simple pero poderosa: en lugar de procesar una secuencia paso a paso (como una LSTM), la atención permite que **cada token mire directamente a todos los demás tokens** y decida a cuáles prestar atención.

Esta simulación calcula la atención multi-head para una frase que tú escribes. Genera pesos aleatorios para las matrices Q, K, V (Query, Key, Value), computa los scores de atención, y los visualiza como un **heatmap**: una matriz donde la intensidad de cada celda indica cuánta atención presta el token de la fila al token de la columna.

Puedes cambiar el número de heads, la dimensión de las keys ($d_k$), y la **temperatura** del softmax para ver cómo afectan la distribución de atención.

---

## 2. Conceptos Fundamentales

### 2.1 Query, Key, Value

Cada token genera tres vectores a partir de su embedding $x$:

$$Q = xW^Q, \quad K = xW^K, \quad V = xW^V$$

- **Query** ($Q$): "¿Qué estoy buscando?"
- **Key** ($K$): "¿Qué ofrezco?"
- **Value** ($V$): "¿Cuál es mi contenido?"

La atención es un mecanismo de **recuperación asociativa**: cada Query busca las Keys más compatibles y recupera sus Values.

### 2.2 Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

1. **Scores**: $S = QK^T$ — producto punto entre queries y keys
2. **Escalado**: dividir por $\sqrt{d_k}$ para evitar que los scores sean demasiado grandes
3. **Softmax**: convertir scores en probabilidades (cada fila suma 1)
4. **Salida**: promedio ponderado de los values

### 2.3 Multi-Head Attention

En lugar de un solo conjunto de Q, K, V, se usan $h$ "heads" en paralelo:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

donde $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$.

Cada head puede aprender a atender a diferentes tipos de relaciones simultáneamente.

### 2.4 ¿Por qué escalar por $\sqrt{d_k}$?

Si $Q$ y $K$ tienen componentes i.i.d. con media 0 y varianza 1, el producto punto $Q \cdot K$ tiene media 0 y varianza $d_k$. Al dividir por $\sqrt{d_k}$, la varianza se normaliza a 1, evitando que el softmax se sature en valores extremos.

---

## 3. La Interfaz

### 3.1 Estructura general

| Área | Ubicación | Función |
|------|-----------|---------|
| **Canvas principal** | Izquierda | Heatmap de atención + líneas de conexión entre tokens |
| **Tooltip** | Flotante | Score de atención al pasar el mouse sobre una celda |
| **Panel de controles** | Derecha (360px) | Frase, configuración, heads, audio, métricas |

### 3.2 El heatmap

La visualización principal es una **matriz cuadrada** de $n \times n$ (donde $n$ = número de tokens):

- **Filas**: tokens que "preguntan" (queries)
- **Columnas**: tokens que "responden" (keys)
- **Color**: intensidad proporcional al peso de atención (más brillante = más atención)
- **Líneas de atención**: arcos que conectan tokens, con grosor proporcional al peso

### 3.3 Métricas en header

- **Heads**: número de heads activos
- **Tokens**: longitud de la frase
- **$d_k$**: dimensión de cada head

---

## 4. Controles Interactivos

### 4.1 Frase

| Control | Tipo | Función |
|---------|------|---------|
| **Campo de texto** | Input | Frase a analizar |
| **Calcular** | Botón | Recomputa la atención con la frase actual |
| **Gato** | Preset | "el gato se sienta en la alfombra" |
| **Negación** | Preset | "yo no creo que el pueda venir hoy" |
| **Referencia** | Preset | "la reina mando al rey a dormir" |
| **Inglés** | Preset | "the cat sat on the mat near me" |

### 4.2 Configuración

| Parámetro | Rango | Descripción |
|-----------|-------|-------------|
| **Número de heads** | 1–8 | Cantidad de heads de atención en paralelo |
| **Dimensión $d_k$** | 4–32 (paso 4) | Dimensionalidad de cada head |
| **Temperatura** | 0.1–5.0 | Controla la "agudeza" del softmax |

### 4.3 Attention head

- **Tabs numerados**: selecciona qué head visualizar (H0, H1, ..., H7)
- **Mostrar valores numéricos**: checkbox para superponer los pesos sobre el heatmap
- **Líneas de atención**: checkbox para mostrar/ocultar los arcos entre tokens

### 4.4 Audio y métricas

- **Volumen**: 0-100%
- **Audio ON/OFF**: toggle de sonificación
- **Métricas**: tokens, heads, $d_k$, $d_\text{model}$ (= heads × $d_k$), entropía media, máxima atención

---

## 5. Las Matemáticas

### 5.1 Dimensiones del modelo

$$d_\text{model} = h \times d_k$$

Con 4 heads y $d_k = 8$: $d_\text{model} = 32$. Cada token se representa como un vector de dimensión 32, dividido en 4 sub-espacios de 8 dimensiones.

### 5.2 Scores de atención con temperatura

La simulación incorpora un parámetro de temperatura $T$:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k} \cdot T}\right)V$$

- $T < 1$: atención **concentrada** (sharp) — pocos tokens reciben casi toda la atención
- $T = 1$: comportamiento estándar
- $T > 1$: atención **dispersa** (flat) — la atención se distribuye más uniformemente

### 5.3 Entropía de la atención

La **entropía** mide cuán dispersa está la distribución de atención de un token:

$$H_i = -\sum_{j=1}^{n} \alpha_{ij} \log_2 \alpha_{ij}$$

donde $\alpha_{ij}$ es el peso de atención del token $i$ al token $j$.

- **Entropía baja**: atención concentrada en pocos tokens
- **Entropía alta** (máximo $\log_2 n$): atención uniforme sobre todos los tokens

### 5.4 Softmax y estabilidad numérica

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Para evitar overflow numérico, se resta el máximo:

$$\text{softmax}(z)_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$

### 5.5 Concatenación multi-head

Si cada head produce una salida de dimensión $d_v = d_k$, la concatenación tiene dimensión $h \cdot d_v$, y la proyección $W^O$ la reduce de vuelta a $d_\text{model}$:

$$W^O \in \mathbb{R}^{(h \cdot d_v) \times d_\text{model}}$$

---

## 6. Sonificación

### 6.1 Mapeo audio

| Evento | Sonido | Parámetro mapeado |
|--------|--------|-------------------|
| **Cálculo de atención** | Secuencia de notas | Una nota por token, frecuencia según su posición |
| **Peso de atención alto** | Nota más fuerte y aguda | Volumen y pitch proporcionales al peso $\alpha_{ij}$ |
| **Cambio de head** | Cambio de timbre | Cada head tiene un timbre distinto |
| **Temperatura baja** | Sonido puntual | Pocas notas dominantes |
| **Temperatura alta** | Sonido difuso | Muchas notas suaves simultáneas |

### 6.2 Interpretación auditiva

Cuando un token concentra su atención en uno solo (temperatura baja), escucharás una nota clara y definida. Con temperatura alta, el sonido se vuelve un coro difuso de muchas notas suaves.

---

## 7. Guía Paso a Paso

### 7.1 Primera exploración

1. Deja la frase por defecto "el gato se sienta en la alfombra"
2. Pulsa **Calcular**
3. Observa el heatmap: cada fila es un token query, cada columna un token key
4. Pasa el mouse sobre las celdas para ver los valores exactos
5. Nota qué tokens reciben más atención (columnas más brillantes)

### 7.2 Explorar multi-head

1. Haz clic en los tabs H0, H1, H2, H3
2. Compara los patrones: cada head atiende a relaciones diferentes
3. Un head puede atender posiciones cercanas, otro puede conectar sujeto-verbo
4. Nota cómo "se" y "sienta" pueden tener alta atención mutua en algún head

### 7.3 Experimentar con temperatura

1. Pon temperatura = 0.1: observa cómo el heatmap se vuelve casi binario (una celda brillante por fila)
2. Pon temperatura = 5.0: el heatmap se vuelve casi uniforme (todas las celdas similares)
3. Vuelve a temperatura = 1.0: balance entre concentración y cobertura
4. Compara la entropía media en cada caso

### 7.4 Variar dimensiones

1. Reduce $d_k$ a 4: representación muy comprimida, atención menos discriminativa
2. Aumenta $d_k$ a 32: más capacidad para distinguir relaciones
3. Pon 1 head con $d_k$ = 32 vs 8 heads con $d_k$ = 4: ¿cuál produce patrones más interesantes?

---

## 8. Conceptos Avanzados

### 8.1 Self-attention vs Cross-attention

En **self-attention** (lo que muestra esta simulación), Q, K, V provienen de la misma secuencia. En **cross-attention** (usado en encoders-decoders), Q viene de una secuencia y K, V de otra — por ejemplo, el decoder atendiendo al encoder en traducción.

### 8.2 Causal masking

En modelos generativos (GPT), se aplica una máscara triangular para que cada token solo pueda atender a tokens anteriores:

$$\text{mask}_{ij} = \begin{cases} 0 & \text{si } j \leq i \\ -\infty & \text{si } j > i \end{cases}$$

Esto previene que el modelo "mire el futuro" durante la generación.

### 8.3 Positional encoding

La atención pura es **invariante al orden** (no distingue "el gato come" de "come gato el"). El positional encoding inyecta información de posición:

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

### 8.4 Complejidad computacional

La atención tiene complejidad $O(n^2 d)$ donde $n$ es la longitud de la secuencia. Esto la hace costosa para secuencias largas. Variantes como Linformer, Performer, y Flash Attention reducen esta complejidad.

### 8.5 Interpretabilidad

Los pesos de atención se usan frecuentemente para "explicar" decisiones del modelo, pero esto es controverso. Jain & Wallace (2019) mostraron que los pesos de atención no siempre correlacionan con la importancia de los tokens para la predicción. Los mapas de atención son informativos pero no son explicaciones causales.

---

## 9. Ejercicios

### Ejercicio 1: Sujeto-verbo
Escribe la frase "la reina mando al rey a dormir". Explora los heads buscando uno donde "reina" y "mando" tengan alta atención mutua. ¿Qué head captura mejor la relación sujeto-verbo?

### Ejercicio 2: Temperatura extrema
Con la frase "el gato se sienta en la alfombra", calcula la atención con temperatura 0.1 y con temperatura 5.0. Calcula la entropía teórica máxima para 7 tokens ($\log_2 7 \approx 2.81$). ¿Qué tan cerca está la entropía de cada caso del máximo?

### Ejercicio 3: Un solo head vs muchos
Compara 1 head con $d_k = 32$ ($d_\text{model} = 32$) contra 8 heads con $d_k = 4$ ($d_\text{model} = 32$). Ambos tienen el mismo $d_\text{model}$. ¿Cuál produce patrones de atención más variados? ¿Por qué el multi-head es preferido en la práctica?

### Ejercicio 4: Auto-atención
¿Qué token de la diagonal del heatmap tiene el mayor valor? La diagonal representa la auto-atención (un token atendiéndose a sí mismo). ¿Hay heads donde la diagonal es dominante y otros donde no?

### Ejercicio 5: Frase bilingüe
Prueba la frase en inglés "the cat sat on the mat near me". Compara los patrones de atención con la versión en español. ¿Hay similitudes en la estructura de atención a pesar del cambio de idioma?

### Ejercicio 6: Dimensión y discriminación
Fija 4 heads y varía $d_k$ de 4 a 32. Para cada valor, observa la distribución de atención de "gato" hacia los demás tokens. ¿A partir de qué $d_k$ la atención se vuelve más selectiva (concentrada en pocos tokens)?

---

## 10. Glosario

| Término | Definición |
|---------|------------|
| **Atención** | Mecanismo que permite a cada token ponderar la importancia de todos los demás tokens |
| **Query (Q)** | Vector que representa "qué busca" un token |
| **Key (K)** | Vector que representa "qué ofrece" un token |
| **Value (V)** | Vector que contiene el "contenido" de un token |
| **Head** | Una instancia independiente de atención con sus propias matrices Q, K, V |
| **Multi-Head Attention** | Múltiples heads en paralelo, concatenados y proyectados |
| **$d_k$** | Dimensión de los vectores Query y Key en cada head |
| **$d_\text{model}$** | Dimensión total del modelo ($= h \times d_k$) |
| **Softmax** | Función que convierte scores en probabilidades normalizadas (suman 1) |
| **Temperatura** | Parámetro que controla la "agudeza" del softmax ($T < 1$: sharp, $T > 1$: flat) |
| **Heatmap** | Representación matricial donde el color indica la magnitud de cada valor |
| **Score de atención** | Producto punto entre Query y Key: $s_{ij} = q_i \cdot k_j$ |
| **Entropía** | Medida de dispersión de una distribución: $H = -\sum p_i \log_2 p_i$ |
| **Self-attention** | Atención donde Q, K, V provienen de la misma secuencia |
| **Cross-attention** | Atención donde Q viene de una secuencia y K, V de otra |
| **Causal masking** | Máscara que impide atender a tokens futuros |
| **Positional encoding** | Vectores sinusoidales que inyectan información de posición |
| **Transformer** | Arquitectura de red neuronal basada enteramente en mecanismos de atención |

---

## 11. Referencias

1. Vaswani, A. et al. (2017). "Attention Is All You Need." *NeurIPS*.
2. Bahdanau, D., Cho, K. & Bengio, Y. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate." *ICLR*.
3. Jain, S. & Wallace, B.C. (2019). "Attention is not Explanation." *NAACL*.
4. Clark, K. et al. (2019). "What Does BERT Look At? An Analysis of BERT's Attention." *BlackboxNLP Workshop*.
5. Vig, J. (2019). "A Multiscale Visualization of Attention in the Transformer Model." *ACL Demo*.
6. Dao, T. et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS*.
