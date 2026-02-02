# LSTM Memory Cells — El Río de la Memoria

> Simulación interactiva de una célula LSTM (Long Short-Term Memory) que procesa secuencias de caracteres paso a paso. Observa cómo las tres compuertas — forget, input y output — controlan el flujo de información, manteniendo memoria a largo plazo mientras decide qué recordar y qué olvidar.

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

Las **redes neuronales recurrentes** (RNN) procesan datos secuenciales — texto, audio, series temporales — pasando información de un paso al siguiente. Pero las RNN simples sufren el problema del **vanishing gradient**: la información se diluye exponencialmente con cada paso, y la red "olvida" lo que vio hace más de unos pocos pasos.

La **LSTM** (Long Short-Term Memory), propuesta por Hochreiter y Schmidhuber en 1997, resuelve este problema con un diseño elegante: un **cell state** (estado celular) que fluye horizontalmente a través del tiempo como un río, y tres **compuertas** que regulan qué información entra, se mantiene y sale.

Esta simulación procesa secuencias de caracteres (como "abcabcabc") paso a paso, mostrando en tiempo real cómo cada compuerta se activa, cómo evoluciona el cell state, y cómo la red aprende a predecir el siguiente carácter.

---

## 2. Conceptos Fundamentales

### 2.1 El problema de la memoria en RNNs

Una RNN simple tiene un único estado oculto $h_t$ que se actualiza en cada paso:

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

Cada multiplicación por $W_h$ escala el gradiente. Después de $n$ pasos, el gradiente se multiplica por $W_h^n$: si los eigenvalores de $W_h$ son menores que 1, el gradiente desaparece; si son mayores que 1, explota.

### 2.2 La arquitectura LSTM

La LSTM reemplaza la actualización simple con cuatro operaciones:

**Forget gate** — qué olvidar del cell state anterior:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Input gate** — qué nueva información guardar:

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**Candidato** — nueva información propuesta:

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Cell state** — actualización de la memoria:

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Output gate** — qué parte de la memoria usar:

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**Hidden state** — la salida:

$$h_t = o_t \odot \tanh(C_t)$$

### 2.3 Las tres compuertas

Cada compuerta usa una **sigmoide** $\sigma$ que produce valores entre 0 y 1:

- **0** = bloquear completamente (cerrada)
- **1** = dejar pasar todo (abierta)
- **Valores intermedios** = filtrado parcial

El operador $\odot$ es la multiplicación elemento a elemento (Hadamard product).

### 2.4 El cell state como autopista

El cell state $C_t$ es la clave de la LSTM. Fluye de paso a paso con solo dos operaciones lineales (multiplicar por $f_t$, sumar $i_t \odot \tilde{C}_t$), lo que permite que los gradientes fluyan sin degradarse durante cientos de pasos.

---

## 3. La Interfaz

### 3.1 Estructura general

| Área | Ubicación | Función |
|------|-----------|---------|
| **Canvas principal** | Izquierda | Diagrama de la célula LSTM con flujo de datos animado |
| **Badge overlay** | Esquina sup. izq. | Estado actual ("Introduce una secuencia", "Procesando paso 3/9") |
| **Panel de controles** | Derecha (360px) | Secuencia, control, compuertas forzadas, audio, métricas |

### 3.2 Visualización del canvas

El canvas muestra un diagrama esquemático de la célula LSTM:

- **Cell state** (línea superior amarilla): flujo horizontal de la memoria
- **Forget gate** (rojo): punto de multiplicación que borra información
- **Input gate** (verde): punto de adición que escribe nueva información
- **Output gate** (azul): filtro que selecciona qué sale al hidden state
- **Partículas animadas**: fluyen por las conexiones mostrando la dirección de los datos
- **Barras de activación**: intensidad de cada compuerta en el paso actual

### 3.3 Métricas en header

El header muestra paso actual, total de pasos y la predicción del carácter siguiente.

---

## 4. Controles Interactivos

### 4.1 Secuencia

| Control | Tipo | Función |
|---------|------|---------|
| **Campo de texto** | Input | Secuencia de caracteres a procesar (máx. 20) |
| **Repetitiva** | Preset | "abcabcabc" — patrón cíclico predecible |
| **Pares** | Preset | "aabbccaabb" — repetición por parejas |
| **Secuencial** | Preset | "abcdefgh" — orden alfabético sin repetición |
| **Palíndromo** | Preset | "abbaabba" — patrón simétrico |
| **Monótona** | Preset | "aaaaaaaaaa" — sin variación |
| **Aleatoria** | Preset | Secuencia aleatoria de caracteres |

### 4.2 Control de ejecución

| Control | Función |
|---------|---------|
| **Procesar** | Ejecuta toda la secuencia de una vez |
| **Paso a paso** | Avanza un carácter |
| **Animar** | Reproducción automática con pausa entre pasos |
| **Unidades LSTM** | Slider 4-16 (paso 4): número de unidades en la capa oculta |

### 4.3 Forzar compuertas

Tres sliders permiten anular el valor aprendido de cada compuerta:

| Compuerta | Rango | Efecto de valor forzado |
|-----------|-------|------------------------|
| **Forget gate** | -1 (auto) a 100 | 0 = borrar toda la memoria; 100 = retener todo |
| **Input gate** | -1 (auto) a 100 | 0 = no guardar nada nuevo; 100 = guardar todo |
| **Output gate** | -1 (auto) a 100 | 0 = no emitir nada; 100 = emitir todo |

El valor -1 significa "automático": la red decide el valor con sus pesos aprendidos.

### 4.4 Audio y métricas

- **Volumen**: 0-100%
- **Audio ON/OFF**: toggle de sonificación
- **Métricas**: paso actual, carácter, predicción, |cell state|, |hidden|, valores de cada compuerta con barras visuales

---

## 5. Las Matemáticas

### 5.1 Codificación one-hot

Cada carácter se convierte en un vector one-hot de tamaño $|\text{vocab}| = 26$ (alfabeto):

$$x_t = e_k \in \mathbb{R}^{26}, \quad \text{donde } k = \text{índice del carácter}$$

### 5.2 Concatenación de entrada

Las compuertas reciben la concatenación del hidden state anterior y la entrada actual:

$$[h_{t-1}, x_t] \in \mathbb{R}^{n_h + 26}$$

donde $n_h$ es el número de unidades LSTM (4, 8, 12 o 16).

### 5.3 Inicialización de pesos

Los pesos se inicializan con distribución gaussiana escalada:

$$W_{ij} \sim \mathcal{N}\left(0, \frac{1}{n_h + 26}\right)$$

El **bias del forget gate** se inicializa a 1 (en lugar de 0) para que la red comience recordando por defecto — un truco práctico descubierto por Jozefowicz et al. (2015).

### 5.4 Predicción

La capa de salida convierte el hidden state en probabilidades sobre el vocabulario:

$$P(\text{siguiente} = k) = \text{softmax}(W_y h_t + b_y)_k$$

$$\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

### 5.5 Normas como diagnóstico

La simulación muestra $|C_t|$ y $|h_t|$ (normas L2 de los vectores):

$$|C_t| = \sqrt{\sum_i C_{t,i}^2}$$

Si $|C_t|$ crece sin control, la memoria se satura. Si colapsa a 0, la red ha olvidado todo.

---

## 6. Sonificación

### 6.1 Mapeo audio

La simulación sonifica el procesamiento de cada paso:

| Evento | Sonido | Parámetro mapeado |
|--------|--------|-------------------|
| **Forget gate** | Nota grave | Frecuencia proporcional a $f_t$ (más grave = más olvido) |
| **Input gate** | Nota media | Frecuencia proporcional a $i_t$ |
| **Output gate** | Nota aguda | Frecuencia proporcional a $o_t$ |
| **Predicción correcta** | Acorde consonante | Armonía cuando la predicción coincide |
| **Predicción incorrecta** | Tono disonante | Frecuencia desafinada |

### 6.2 Interpretación auditiva

- **Secuencias predecibles** (repetitivas): el audio se vuelve más armónico conforme la red aprende el patrón
- **Secuencias aleatorias**: sonidos más caóticos e irregulares
- **Forget gate cerrada** (forzada a 0): silencios o notas muy graves
- **Cell state saturado**: frecuencias que alcanzan el límite superior

---

## 7. Guía Paso a Paso

### 7.1 Primera exploración

1. Deja la secuencia por defecto "abcabcabc"
2. Pulsa **Paso a paso** repetidamente
3. Observa cómo las barras de forget/input/output cambian en cada paso
4. Nota cómo $|C_t|$ crece gradualmente — la red está acumulando memoria
5. Compara la predicción con el carácter real del siguiente paso

### 7.2 Comparar patrones

1. Prueba **Repetitiva** → la red aprende rápido (predicciones correctas tras pocos pasos)
2. Prueba **Monótona** → forget gate alta, input gate baja (no necesita nueva info)
3. Prueba **Aleatoria** → predicciones pobres, compuertas erráticas
4. Compara los valores de $|C_t|$ entre las tres secuencias

### 7.3 Experimentar con compuertas forzadas

1. Fuerza **Forget gate = 0**: observa cómo $|C_t|$ cae a cero — sin memoria
2. Fuerza **Forget gate = 100**: $|C_t|$ crece sin parar — memoria infinita
3. Fuerza **Input gate = 0**: la red no puede guardar información nueva
4. Fuerza **Output gate = 0**: la red tiene memoria interna pero no puede usarla para predecir

### 7.4 Variar unidades

1. Cambia a 4 unidades → menos capacidad, predicciones peores en secuencias complejas
2. Cambia a 16 unidades → más capacidad, mejor con patrones largos
3. Observa cómo $|h_t|$ escala con el número de unidades

---

## 8. Conceptos Avanzados

### 8.1 Vanishing gradient: por qué funciona la LSTM

En una RNN simple, el gradiente en el paso $t$ respecto al paso $t-k$ es:

$$\frac{\partial L_t}{\partial h_{t-k}} = \prod_{i=t-k}^{t-1} W_h^T \cdot \text{diag}(\sigma'(z_i))$$

Este producto tiende a 0 exponencialmente. En una LSTM, el gradiente a través del cell state es:

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

Si $f_t \approx 1$, el gradiente fluye sin atenuación. La LSTM aprende cuándo olvidar ($f_t \to 0$) y cuándo recordar ($f_t \to 1$), controlando activamente el flujo del gradiente.

### 8.2 GRU: la alternativa simplificada

La **Gated Recurrent Unit** (Cho et al., 2014) combina forget e input gate en una sola "update gate" $z_t$:

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

Menos parámetros, rendimiento similar en muchas tareas.

### 8.3 Peephole connections

Variante de Gers & Schmidhuber (2000) donde las compuertas también miran el cell state:

$$f_t = \sigma(W_f \cdot [C_{t-1}, h_{t-1}, x_t] + b_f)$$

Permite a la red basar sus decisiones de compuerta en el contenido actual de la memoria.

### 8.4 Bidirectional LSTM

Procesa la secuencia en ambas direcciones y concatena los hidden states:

$$\overrightarrow{h_t}, \overleftarrow{h_t} \to h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$$

Cada posición tiene contexto del pasado y del futuro.

### 8.5 LSTM vs Transformers

Los Transformers han reemplazado a las LSTM en muchas tareas de NLP porque procesan toda la secuencia en paralelo (atención) en lugar de paso a paso. Sin embargo, las LSTM siguen siendo relevantes en streaming de datos, dispositivos con recursos limitados y series temporales en tiempo real.

---

## 9. Ejercicios

### Ejercicio 1: Detector de patrones
Usa la secuencia "abcabcabc". Procésala paso a paso y anota en qué paso la predicción se vuelve correcta por primera vez. ¿Cuántos pasos necesita la red para "aprender" el patrón cíclico de período 3?

### Ejercicio 2: Memoria forzada
Fuerza el forget gate a 0 y procesa "abcabcabc". ¿Qué ocurre con las predicciones? Ahora fuerza el forget gate a 100. ¿Mejora o empeora? Explica por qué la red necesita un balance entre recordar y olvidar.

### Ejercicio 3: Capacidad vs complejidad
Compara el rendimiento con 4 unidades vs 16 unidades en la secuencia "abbaabba" (palíndromo). ¿Cuántas unidades necesita la red para capturar la simetría del palíndromo? ¿Hay un punto de rendimientos decrecientes?

### Ejercicio 4: Secuencia monótona
Procesa "aaaaaaaaaa" y observa las compuertas. ¿Qué valores toman forget, input y output gate? Interpreta: ¿por qué el forget gate es alto y el input gate bajo cuando la secuencia no cambia?

### Ejercicio 5: Output gate aislada
Fuerza input gate = 100 y forget gate = 100 (máxima memoria). Ahora varía el output gate de 0 a 100 mientras procesas una secuencia. ¿Cómo cambia la predicción? El output gate es como una válvula de lectura: ¿qué pasa si la memoria está llena pero no se puede leer?

### Ejercicio 6: Aleatoriedad y predicción
Genera 5 secuencias aleatorias y compara las predicciones. ¿Por qué una secuencia verdaderamente aleatoria es inherentemente impredecible? Relaciona esto con la entropía de Shannon: $H = -\sum p_i \log_2 p_i$. ¿Cuál sería la entropía máxima para un vocabulario de 26 caracteres?

---

## 10. Glosario

| Término | Definición |
|---------|------------|
| **LSTM** | Long Short-Term Memory: tipo de RNN con compuertas que controlan el flujo de información |
| **Cell state** | Vector de memoria que fluye horizontalmente a través de los pasos temporales |
| **Hidden state** | Salida de la célula LSTM en cada paso, usada para predicciones |
| **Forget gate** | Compuerta sigmoide que decide qué parte del cell state anterior borrar |
| **Input gate** | Compuerta sigmoide que decide qué nueva información guardar en el cell state |
| **Output gate** | Compuerta sigmoide que decide qué parte del cell state emitir como hidden state |
| **Candidato** | Valor propuesto para añadir al cell state, generado con tanh |
| **Sigmoide** | Función $\sigma(x) = 1/(1+e^{-x})$ que comprime valores al rango $(0,1)$ |
| **Hadamard product** | Multiplicación elemento a elemento de dos vectores: $(a \odot b)_i = a_i \cdot b_i$ |
| **One-hot encoding** | Representación de un carácter como vector con un 1 y el resto 0s |
| **Vanishing gradient** | Problema donde los gradientes se atenúan exponencialmente en redes profundas/recurrentes |
| **RNN** | Red Neuronal Recurrente: arquitectura que procesa secuencias con estado oculto |
| **GRU** | Gated Recurrent Unit: variante simplificada de LSTM con 2 compuertas |
| **Softmax** | Función que convierte un vector de scores en probabilidades que suman 1 |
| **Peephole** | Variante LSTM donde las compuertas también observan el cell state |
| **Bidirectional** | LSTM que procesa la secuencia en ambas direcciones |
| **Norma L2** | $\|v\| = \sqrt{\sum v_i^2}$: magnitud de un vector |
| **Secuencia** | Serie ordenada de tokens (caracteres, palabras, valores) procesada paso a paso |

---

## 11. Referencias

1. Hochreiter, S. & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*, 9(8), 1735-1780.
2. Gers, F., Schmidhuber, J. & Cummins, F. (2000). "Learning to Forget: Continual Prediction with LSTM." *Neural Computation*, 12(10), 2451-2471.
3. Cho, K. et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." *EMNLP*.
4. Jozefowicz, R., Zaremba, W. & Sutskever, I. (2015). "An Empirical Exploration of Recurrent Network Architectures." *ICML*.
5. Olah, C. (2015). "Understanding LSTM Networks." *colah's blog*.
6. Greff, K. et al. (2017). "LSTM: A Search Space Odyssey." *IEEE TNNLS*, 28(10), 2222-2232.
