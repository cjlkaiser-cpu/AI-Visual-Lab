# Transformer Anatomy — Anatomía de un Transformer

> Simulación interactiva de un bloque Transformer completo. Selecciona una frase, ejecuta un forward pass paso a paso, y observa cómo cada token fluye a través de embedding, positional encoding, multi-head attention, feed-forward network, conexiones residuales y layer normalization. Visualiza las activaciones en cada componente y sigue el procesamiento desde texto hasta representaciones vectoriales.

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

El **Transformer** es la arquitectura que revolucionó el procesamiento de lenguaje natural y que impulsa modelos como GPT, BERT, y Claude. A diferencia de las RNNs que procesan tokens uno a uno, el Transformer procesa todos los tokens **en paralelo** usando el mecanismo de atención.

Esta simulación implementa un micro-Transformer con 2 capas y 2 heads de atención, dimensión $d = 32$, y un vocabulario de 20 palabras. Puedes ejecutar un **forward pass** completo o avanzar paso a paso por cada componente:

1. **Embedding**: palabra → vector
2. **Positional Encoding**: inyectar información de posición
3. **Multi-Head Attention** (×2 capas): tokens se comunican entre sí
4. **Feed-Forward Network** (×2 capas): procesamiento individual por token
5. **Output**: representación final

Cada componente se anima y visualiza en el canvas, mostrando las activaciones como barras de calor.

---

## 2. Conceptos Fundamentales

### 2.1 El flujo completo

$$\text{Output} = \text{LayerNorm}(x + \text{FFN}(\text{LayerNorm}(x + \text{Attention}(x))))$$

Este patrón se repite $N$ veces (en esta simulación, $N = 2$).

### 2.2 Embedding

Cada palabra se mapea a un vector de dimensión $d$ usando una tabla de embedding:

$$e_\text{token} = W_E[\text{token\_id}] \in \mathbb{R}^d$$

donde $W_E \in \mathbb{R}^{|\text{vocab}| \times d}$ es la matriz de embedding.

### 2.3 Positional Encoding

El Transformer no tiene noción intrínseca del orden. Las codificaciones posicionales sinusoidales inyectan esta información:

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

Se suman al embedding: $x = e_\text{token} + PE_{pos}$.

### 2.4 Multi-Head Attention

$$\text{MultiHead}(x) = \text{Concat}(\text{head}_1, \text{head}_2) W^O$$

$$\text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

Con 2 heads y $d = 32$: cada head opera en un subespacio de dimensión $d_k = 16$.

### 2.5 Feed-Forward Network

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

Típicamente, la dimensión interna es $4d$ (en esta simulación: $4 \times 32 = 128$). Esto da capacidad de procesamiento no lineal a cada token individualmente.

### 2.6 Conexiones residuales + Layer Normalization

$$x' = \text{LayerNorm}(x + \text{Sublayer}(x))$$

- **Residual** ($x + \text{Sublayer}(x)$): permite que los gradientes fluyan directamente
- **LayerNorm**: normaliza las activaciones para estabilizar el entrenamiento

---

## 3. La Interfaz

### 3.1 Estructura general

| Área | Ubicación | Función |
|------|-----------|---------|
| **Canvas principal** | Izquierda | Diagrama de bloques del Transformer con flujo animado |
| **Tooltip** | Flotante | Detalles del componente al pasar el mouse |
| **Panel de controles** | Derecha (360px) | Entrada, audio, componente activo, explicaciones |

### 3.2 Diagrama del canvas

El canvas muestra el Transformer como una torre vertical de bloques:

1. **Embedding** (base): tokens convertidos en vectores
2. **+ Positional Encoding**: vectores sinusoidales sumados
3. **Capa 1**: Attention → Add & Norm → FFN → Add & Norm
4. **Capa 2**: mismo patrón
5. **Output** (tope): representaciones finales

Cada componente se ilumina cuando está activo, y las activaciones se visualizan como barras de calor dentro de cada bloque. Partículas fluyen entre componentes durante la animación.

### 3.3 Header

Muestra el paso actual del forward pass y el número de tokens.

---

## 4. Controles Interactivos

### 4.1 Entrada

| Control | Función |
|---------|---------|
| **Selector de frase** | 4 frases predefinidas en español |
| **Forward Pass** | Ejecuta todo el forward pass con animación |
| **Paso a Paso** | Avanza un componente a la vez |
| **Reset** | Vuelve al estado inicial |

Frases disponibles:
- "el gato se sienta"
- "la red aprende rapido"
- "el modelo genera texto"
- "atencion es todo"

### 4.2 Audio

- **Volumen**: 0-100%
- **Silenciar**: toggle de audio

### 4.3 Componente activo

| Métrica | Descripción |
|---------|-------------|
| **Componente** | Nombre del componente activo (Embedding, PosEnc, Attention L0, FFN L0, etc.) |
| **Dimensión** | $d = 32$ |
| **Heads** | 2 |
| **Capas** | 2 |
| **Gráfico de activaciones** | Barras mostrando las activaciones del componente activo |

---

## 5. Las Matemáticas

### 5.1 Dimensiones del modelo

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| $d$ | 32 | Dimensión del modelo |
| $h$ | 2 | Número de heads |
| $d_k$ | 16 | Dimensión por head ($d / h$) |
| $d_{ff}$ | 128 | Dimensión interna del FFN ($4d$) |
| $N$ | 2 | Número de capas |
| $|\text{vocab}|$ | 20 | Tamaño del vocabulario |

### 5.2 Cómputo del attention score

Para cada head $i$, la query $q$, key $k$ y value $v$ de cada token se computan:

$$q_j = x_j W_i^Q, \quad k_j = x_j W_i^K, \quad v_j = x_j W_i^V$$

Los scores entre todos los pares de tokens:

$$\alpha_{jl} = \frac{\exp(q_j \cdot k_l / \sqrt{d_k})}{\sum_m \exp(q_j \cdot k_m / \sqrt{d_k})}$$

La salida para el token $j$:

$$\text{head}_i(j) = \sum_l \alpha_{jl} \cdot v_l$$

### 5.3 Layer Normalization

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

donde $\mu = \frac{1}{d}\sum_i x_i$, $\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$, y $\gamma, \beta$ son parámetros aprendidos.

### 5.4 Conteo de parámetros

| Componente | Parámetros |
|-----------|------------|
| **Embedding** | $20 \times 32 = 640$ |
| **Attention (por capa)** | $4 \times 32 \times 32 = 4096$ ($W^Q, W^K, W^V, W^O$) |
| **FFN (por capa)** | $32 \times 128 + 128 + 128 \times 32 + 32 = 8352$ |
| **LayerNorm (4×)** | $4 \times 2 \times 32 = 256$ |
| **Total** | $\approx 25,400$ |

### 5.5 Positional Encoding: propiedades

Las codificaciones sinusoidales tienen propiedades geométricas útiles:

$$PE_{pos+k} \cdot PE_{pos} = f(k)$$

El producto punto entre dos posiciones depende solo de su **distancia relativa**, no de sus posiciones absolutas. Esto permite al modelo generalizar a longitudes de secuencia no vistas.

---

## 6. Sonificación

### 6.1 Mapeo audio

| Evento | Sonido | Parámetro mapeado |
|--------|--------|-------------------|
| **Embedding** | Nota base por token | Frecuencia según la posición del token en el vocabulario |
| **Positional Encoding** | Superposición sinusoidal | Las frecuencias del PE se mapean a audio |
| **Attention** | Acordes | Tokens con alta atención mutua suenan juntos |
| **FFN** | Transformación tímbrica | El timbre cambia según la magnitud de las activaciones |
| **LayerNorm** | Normalización de volumen | El volumen se estabiliza |
| **Avance de capa** | Cambio de octava | Cada capa suena en un registro más alto |

### 6.2 Interpretación

El forward pass suena como una progresión musical: comienza con notas aisladas (embedding), se enriquece con armonías (attention), pasa por transformaciones (FFN), y culmina con las representaciones finales.

---

## 7. Guía Paso a Paso

### 7.1 Forward pass completo

1. Selecciona "el gato se sienta"
2. Pulsa **Forward Pass**
3. Observa la animación: cada componente se ilumina secuencialmente
4. Nota cómo las partículas fluyen de abajo hacia arriba
5. Al final, observa las activaciones de salida en el gráfico

### 7.2 Paso a paso

1. Selecciona una frase y pulsa **Paso a Paso**
2. **Embedding**: cada token se convierte en un vector (barras en el gráfico)
3. **PosEnc**: los vectores se modifican ligeramente (ondas sinusoidales sumadas)
4. **Attention L0**: los tokens se "comunican" — las activaciones cambian según el contexto
5. **FFN L0**: cada token se transforma individualmente
6. **Attention L1, FFN L1**: segunda capa refina las representaciones
7. **Output**: representación final

### 7.3 Comparar frases

1. Ejecuta "el gato se sienta" paso a paso, anotando las activaciones del token "gato"
2. Reset, ejecuta "la red aprende rapido", observa las activaciones de "red"
3. ¿Cómo difieren las representaciones después de la atención? Cada token se contextualiza diferente según su contexto

### 7.4 Observar la atención

1. En el paso de Attention, el gráfico de activaciones muestra cómo los tokens se influencian mutuamente
2. Nota si "gato" atiende más a "el" (artículo) o a "sienta" (verbo)
3. En la segunda capa de atención, las relaciones pueden ser más abstractas

---

## 8. Conceptos Avanzados

### 8.1 Pre-norm vs Post-norm

Hay dos variantes de dónde colocar LayerNorm:

- **Post-norm** (original): $\text{LN}(x + \text{Sublayer}(x))$
- **Pre-norm** (moderna): $x + \text{Sublayer}(\text{LN}(x))$

Pre-norm es más estable para entrenar redes profundas y es la variante usada en GPT-2/3.

### 8.2 Encoder vs Decoder

El Transformer original tiene dos partes:
- **Encoder**: procesa la entrada completa con self-attention bidireccional (BERT)
- **Decoder**: genera la salida token a token con atención causal + cross-attention (GPT usa solo decoder)

Esta simulación modela un encoder: todos los tokens se ven entre sí.

### 8.3 Escalamiento

Los Transformers escalan de forma predecible (leyes de escalado):

$$L \propto N^{-\alpha}$$

donde $L$ es la loss, $N$ es el número de parámetros, y $\alpha \approx 0.076$ para modelos de lenguaje. Duplicar parámetros reduce la loss de forma consistente.

### 8.4 Rotary Position Embeddings (RoPE)

Las codificaciones posicionales sinusoidales clásicas se suman al embedding. RoPE (usado en LLaMA, Mistral) en cambio **rota** los vectores Q y K:

$$\text{RoPE}(x, pos) = R_\theta(pos) \cdot x$$

Esto codifica posiciones relativas directamente en los productos punto de atención.

### 8.5 KV-Cache

Durante la generación, los vectores K y V de tokens anteriores no cambian. El **KV-cache** los almacena para no recalcularlos, reduciendo la complejidad de generación de $O(n^2)$ a $O(n)$ por token.

---

## 9. Ejercicios

### Ejercicio 1: Embedding vs Output
Compara las activaciones después del embedding con las del output final para la frase "el gato se sienta". ¿Qué tan diferentes son? El embedding es una representación estática; el output incorpora contexto de toda la frase.

### Ejercicio 2: Efecto del positional encoding
Ejecuta paso a paso y compara las activaciones antes y después de sumar el positional encoding. ¿Se notan las ondas sinusoidales? ¿Son más pronunciadas para posiciones tempranas o tardías?

### Ejercicio 3: Una capa vs dos
Anota las activaciones después de la capa 1 y después de la capa 2. ¿La segunda capa cambia mucho las representaciones? ¿O las refina sutilmente? En redes reales con 32+ capas, las capas profundas hacen cambios cada vez más pequeños (gracias a las conexiones residuales).

### Ejercicio 4: Conteo de operaciones
Calcula el número de multiplicaciones en un forward pass para 4 tokens con $d = 32$ y 2 heads. Pista: la atención requiere $O(n^2 d)$ y el FFN requiere $O(n \cdot d \cdot d_{ff})$. ¿Cuál es el cuello de botella para secuencias largas?

### Ejercicio 5: Frases diferentes, mismas palabras
Compara "el modelo genera texto" con "atencion es todo". ¿Cómo difieren las representaciones finales? ¿El token "modelo" en una frase tiene la misma representación que "modelo" en otra frase?

### Ejercicio 6: LayerNorm en acción
Observa el gráfico de activaciones antes y después de cada LayerNorm. ¿Las activaciones se centran (media ≈ 0) y normalizan (varianza ≈ 1)? ¿Por qué esto es importante para la estabilidad del entrenamiento?

---

## 10. Glosario

| Término | Definición |
|---------|------------|
| **Transformer** | Arquitectura basada en atención que procesa tokens en paralelo |
| **Embedding** | Tabla que convierte tokens (palabras) en vectores de dimensión $d$ |
| **Positional Encoding** | Vectores sinusoidales que inyectan información de posición en los embeddings |
| **Self-Attention** | Mecanismo donde cada token atiende a todos los demás tokens de la misma secuencia |
| **Multi-Head** | Múltiples instancias de atención en paralelo, cada una en un subespacio |
| **FFN** | Feed-Forward Network: dos capas lineales con ReLU, aplicada por token |
| **LayerNorm** | Normalización que centra y escala las activaciones por token |
| **Conexión residual** | Atajo $x + f(x)$ que permite flujo directo de gradientes |
| **$d_\text{model}$** | Dimensión del modelo (tamaño de los vectores de representación) |
| **$d_k$** | Dimensión por head de atención ($d_\text{model} / h$) |
| **$d_{ff}$** | Dimensión interna del FFN (típicamente $4 \times d_\text{model}$) |
| **Forward pass** | Propagación de la entrada a través de todas las capas hasta la salida |
| **Pre-norm** | Variante donde LayerNorm se aplica antes de la sublayer |
| **Post-norm** | Variante original donde LayerNorm se aplica después de la sublayer |
| **Encoder** | Parte del Transformer con atención bidireccional (ve todos los tokens) |
| **Decoder** | Parte del Transformer con atención causal (solo ve tokens previos) |
| **RoPE** | Rotary Position Embeddings: codificación posicional por rotación de vectores |
| **KV-Cache** | Almacenamiento de Keys y Values previos para generación eficiente |

---

## 11. Referencias

1. Vaswani, A. et al. (2017). "Attention Is All You Need." *NeurIPS*.
2. Devlin, J. et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." *NAACL*.
3. Radford, A. et al. (2019). "Language Models are Unsupervised Multitask Learners." *OpenAI Technical Report*.
4. Kaplan, J. et al. (2020). "Scaling Laws for Neural Language Models." *arXiv:2001.08361*.
5. Elhage, N. et al. (2021). "A Mathematical Framework for Transformer Circuits." *Anthropic*.
6. Su, J. et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv:2104.09864*.
