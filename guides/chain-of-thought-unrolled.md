# Chain of Thought Unrolled — El Árbol del Razonamiento

> Simulación interactiva del razonamiento paso a paso (Chain of Thought) en modelos de lenguaje. Compara cómo un LLM resuelve problemas descomponiendo su razonamiento en pasos intermedios versus dando una respuesta directa. Visualiza el árbol de razonamiento con confianza por nodo, backtracking y alucinaciones.

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

Los **modelos de lenguaje** (LLMs) como GPT-4 o Claude son capaces de resolver problemas complejos, pero su rendimiento depende dramáticamente de *cómo* se les pide que razonen. Pedirle a un modelo que dé una respuesta directa ("¿Cuánto es 23 × 17?") produce resultados diferentes a pedirle que **piense paso a paso**.

El **Chain of Thought** (CoT), introducido por Wei et al. (2022), es una técnica de prompting que hace explícito el razonamiento intermedio. En lugar de un salto directo de pregunta a respuesta, el modelo genera una cadena de pasos lógicos.

Esta simulación visualiza el proceso como un **árbol de razonamiento**: cada nodo es un paso con su confianza, los errores se muestran como backtracking (nodos rojos), y el modo de alucinación demuestra cómo los razonamientos pueden parecer lógicos pero contener errores sutiles.

---

## 2. Conceptos Fundamentales

### 2.1 Razonamiento implícito vs. explícito

Un LLM sin CoT genera $P(\text{respuesta} | \text{pregunta})$ directamente. Con CoT, genera:

$$P(\text{respuesta} | \text{pregunta}) = \sum_{\text{cadena}} P(\text{respuesta} | \text{cadena}, \text{pregunta}) \cdot P(\text{cadena} | \text{pregunta})$$

Al marginalizar sobre cadenas de razonamiento, el modelo puede "explorar" caminos lógicos intermedios que mejoran la probabilidad de llegar a la respuesta correcta.

### 2.2 La inequalidad clave

La observación empírica fundamental es:

$$P(\text{respuesta correcta} | \text{CoT}) > P(\text{respuesta correcta} | \text{directa})$$

Esta desigualdad se amplifica con la complejidad del problema: para aritmética simple la diferencia es pequeña, pero para razonamiento multi-paso se vuelve enorme.

### 2.3 Tipos de razonamiento

La simulación incluye 5 tipos de problemas:

| Tipo | Ejemplo | Pasos CoT típicos |
|------|---------|-------------------|
| Aritmética | 23 × 17 | 5-7 (descomponer, multiplicar, sumar) |
| Lógica | Silogismo | 3-4 (premisas, deducción) |
| Secuencia | 2, 6, 18, 54, ? | 3-4 (patrón, regla, aplicar) |
| Analogía | Día:Noche :: Calor:? | 2-3 (relación, aplicar) |
| Multi-paso | Problema verbal | 5-8 (parsear, modelar, resolver) |

### 2.4 Confianza por paso

Cada nodo del árbol tiene una **confianza** $c \in [0, 1]$ que refleja cuán seguro está el modelo de ese paso intermedio. Visualmente se codifica con color: verde = alta confianza, amarillo = media, rojo = baja (posible error o backtracking).

---

## 3. La Interfaz

### 3.1 Canvas principal

El canvas muestra el **árbol de razonamiento**:

- **Nodos circulares**: cada paso de razonamiento, coloreados por confianza
- **Aristas**: conexiones entre pasos (padre → hijo)
- **Nodos rojos**: backtracking — pasos descartados por error detectado
- **Nodo final**: respuesta, resaltado con borde brillante
- **Comparación**: si se usa respuesta directa, aparece un solo nodo versus el árbol completo

### 3.2 Panel lateral

- **Ecuación**: $P(\text{answer} | \text{CoT}) > P(\text{answer} | \text{direct})$
- **Selector de problema**: 5 problemas predefinidos
- **Botones**: Resolver con CoT, Respuesta Directa, Reset
- **Toggles**: Modo Alucinación, Mostrar Confianza
- **Comparación**: tabla con pasos, confianza y corrección para ambos métodos

### 3.3 Barra de estado

Muestra el conteo de nodos activos y la confianza general del razonamiento.

---

## 4. Controles Interactivos

### 4.1 Selector de problema

Dropdown con 5 problemas de dificultad creciente:
- **Aritmética**: 23 × 17 = ? — requiere descomposición
- **Lógica**: Todos A son B. X es A. ¿X es B? — silogismo clásico
- **Secuencia**: 2, 6, 18, 54, ? — progresión geométrica
- **Analogía**: Día:Noche :: Calor:? — relación de opuestos
- **Multi-paso**: Si tengo 5 manzanas... — problema verbal compuesto

### 4.2 Modos de resolución

| Botón | Acción |
|-------|--------|
| Resolver con CoT | Genera el árbol de razonamiento paso a paso con animación |
| Respuesta Directa | Genera un único nodo con la respuesta inmediata |
| Reset | Limpia el canvas y las métricas |

### 4.3 Configuración

| Control | Efecto |
|---------|--------|
| Modo Alucinación | Introduce errores sutiles en los pasos intermedios: cálculos incorrectos que "parecen" lógicos |
| Mostrar Confianza | Muestra/oculta los valores numéricos de confianza en cada nodo |

### 4.4 Métricas

La sección de comparación muestra lado a lado:
- Número de pasos CoT
- Confianza CoT vs. Directa
- Si cada método llegó a la respuesta correcta

---

## 5. Las Matemáticas

### 5.1 Probabilidad condicional con cadena

Para un problema con respuesta $a$ y cadena de razonamiento $c_1, c_2, \ldots, c_n$:

$$P(a | x) = P(a | c_n, x) \prod_{i=1}^{n} P(c_i | c_{i-1}, \ldots, c_1, x)$$

Cada paso condiciona en todos los anteriores, creando una cadena de dependencias.

### 5.2 Confianza acumulada

La confianza total de una cadena es aproximadamente:

$$C_{\text{total}} = \prod_{i=1}^{n} c_i$$

donde $c_i$ es la confianza del paso $i$. Esto implica que una cadena larga con pasos de confianza 0.9 cada uno tiene confianza total $0.9^n$, que decrece exponencialmente.

### 5.3 Self-consistency

Una mejora sobre CoT es generar $k$ cadenas de razonamiento y tomar la respuesta mayoritaria:

$$a^* = \arg\max_a \sum_{j=1}^{k} \mathbb{1}[a_j = a]$$

Esto es equivalente a un ensemble de razonamientos: si 7 de 10 cadenas llegan a "391", esa es probablemente la respuesta correcta.

### 5.4 Alucinaciones en cadena

Una alucinación ocurre cuando un paso $c_i$ es incorrecto pero tiene alta confianza aparente. El error se propaga:

$$P(a_{\text{correcta}} | c_1, \ldots, c_i^{\text{error}}, \ldots, c_n) \approx 0$$

Un solo paso erróneo puede invalidar toda la cadena. La simulación demuestra esto visualmente con nodos de alta confianza pero contenido incorrecto.

---

## 6. Sonificación

### 6.1 Diseño de audio

Cada evento del razonamiento produce un sonido:

- **Nuevo nodo**: nota ascendente cuya frecuencia refleja la confianza
- **Backtracking**: tono descendente en rojo
- **Respuesta final**: acorde mayor si correcta, menor si incorrecta
- **Alucinación**: nota con ligero detune (desafinación intencional)

### 6.2 Mapeo de frecuencias

| Evento | Frecuencia | Duración | Tipo |
|--------|-----------|----------|------|
| Nodo normal | 300 + $c \times 400$ Hz | 0.2s | sine |
| Backtrack | 600 → 200 Hz (glide) | 0.3s | sawtooth |
| Respuesta correcta | Acorde C-E-G | 0.5s | sine |
| Respuesta incorrecta | 200 Hz | 0.4s | triangle |
| Alucinación | freq ± 8 Hz | 0.25s | sine (detuned) |

---

## 7. Guía Paso a Paso

### Paso 1: Resolver con CoT

1. Selecciona el problema *"Aritmética: 23 × 17"*
2. Pulsa **Resolver con CoT**
3. Observa cómo el árbol crece: "Descomponer 23 × 17" → "23 × 10 = 230" → "23 × 7 = ?" → ...
4. Cada nodo muestra su texto y confianza coloreada

### Paso 2: Comparar con respuesta directa

1. Pulsa **Reset**, luego **Respuesta Directa**
2. Aparece un único nodo con la respuesta
3. Compara las métricas: ¿es correcta? ¿Con qué confianza?
4. Para aritmética simple, ambos métodos pueden acertar, pero CoT es más fiable

### Paso 3: Escalar la dificultad

1. Cambia a *"Multi-paso: Si tengo 5 manzanas..."*
2. Resuelve con CoT: observa 5-8 pasos con confianzas variables
3. Resuelve con Directa: la confianza baja notablemente
4. Este es el régimen donde CoT brilla

### Paso 4: Explorar alucinaciones

1. Activa **Modo Alucinación**
2. Resuelve el problema aritmético con CoT
3. Observa que un paso intermedio tiene confianza alta pero contiene un error (e.g., "20 × 7 = 160" en vez de 140)
4. El error se propaga: la respuesta final es incorrecta a pesar del "razonamiento"

### Paso 5: Lógica y backtracking

1. Selecciona el problema de lógica (silogismo)
2. Resuelve con CoT
3. Observa si hay nodos rojos de backtracking donde el modelo corrigió un razonamiento

---

## 8. Conceptos Avanzados

### 8.1 Tree of Thought

Una extensión de CoT donde el modelo explora múltiples caminos de razonamiento en paralelo, podando ramas poco prometedoras:

$$\text{ToT}: \text{BFS/DFS sobre el espacio de razonamientos}$$

En lugar de una cadena lineal, se genera un árbol completo y se selecciona el mejor camino.

### 8.2 Chain of Thought Faithful vs. Post-hoc

Un debate activo: ¿los pasos de CoT reflejan el razonamiento *real* del modelo, o son racionalizaciones post-hoc? Estudios muestran que modificar los pasos intermedios a veces no cambia la respuesta, sugiriendo que el modelo ya "sabe" la respuesta y genera el razonamiento como justificación.

### 8.3 CoT como computación implícita

Cada token generado en la cadena CoT actúa como un "paso de computación" adicional. Un modelo con $L$ layers de Transformer y $n$ tokens de CoT tiene efectivamente $L \times n$ pasos de procesamiento, versus solo $L$ sin CoT. Esto explica por qué CoT ayuda: da al modelo más "tiempo para pensar".

### 8.4 Verificación y auto-corrección

Técnicas modernas combinan CoT con verificadores:

1. Generar cadena de razonamiento
2. Verificar cada paso con un modelo verificador
3. Si un paso falla, regenerar desde ese punto

Esto mitiga las alucinaciones al introducir un mecanismo de feedback.

---

## 9. Ejercicios

### Ejercicio 1: Longitud óptima de cadena
Resuelve cada uno de los 5 problemas con CoT. Registra el número de pasos y si la respuesta fue correcta. ¿Hay correlación entre el número de pasos y la dificultad del problema?

### Ejercicio 2: Tasa de error con alucinaciones
Con el modo alucinación activado, resuelve el problema aritmético 10 veces con CoT. ¿En cuántas ocasiones la respuesta final fue incorrecta? Calcula la tasa de error y compárala con la tasa sin alucinaciones.

### Ejercicio 3: Confianza acumulada
Para el problema multi-paso, anota la confianza de cada nodo de la cadena CoT. Calcula $C_{\text{total}} = \prod c_i$. ¿El producto explica la confianza final mostrada? ¿Qué pasa si un solo paso tiene $c_i = 0.5$?

### Ejercicio 4: CoT vs. Directa por tipo
Completa una tabla comparando CoT vs. Directa para cada tipo de problema (aritmética, lógica, secuencia, analogía, multi-paso). ¿Para qué tipos la diferencia es más pronunciada?

### Ejercicio 5: Self-consistency manual
Resuelve el problema aritmético con CoT 5 veces. Si obtienes resultados diferentes, ¿cuál elegirías por votación mayoritaria? Compara con la respuesta verdadera.

### Ejercicio 6: Detectar la alucinación
Activa el modo alucinación y resuelve un problema. Sin mirar la respuesta correcta, intenta identificar visualmente qué nodo contiene el error basándote solo en los textos de cada paso. ¿Es fácil o difícil?

---

## 10. Glosario

| Término | Definición |
|---------|-----------|
| Chain of Thought (CoT) | Técnica de prompting que hace explícito el razonamiento paso a paso |
| Prompting | Técnica de formular instrucciones para guiar la salida de un LLM |
| LLM | Large Language Model — modelo de lenguaje de gran escala |
| Razonamiento multi-paso | Problemas que requieren múltiples operaciones lógicas secuenciales |
| Backtracking | Retroceder en la cadena de razonamiento al detectar un error |
| Alucinación | Paso de razonamiento que parece correcto pero contiene errores factuales |
| Confianza | Probabilidad asignada a cada paso del razonamiento (0 a 1) |
| Self-consistency | Generar múltiples cadenas CoT y elegir la respuesta mayoritaria |
| Tree of Thought (ToT) | Extensión de CoT que explora múltiples caminos en paralelo |
| Zero-shot CoT | Usar CoT sin ejemplos, solo con "Let's think step by step" |
| Few-shot CoT | Incluir ejemplos resueltos paso a paso en el prompt |
| Verificador | Modelo que evalúa la corrección de cada paso de razonamiento |
| Token | Unidad mínima de texto procesada por el LLM |
| Marginalización | Sumar sobre todas las cadenas posibles para obtener $P(\text{respuesta})$ |
| Ensemble | Combinar múltiples predicciones para mejorar la precisión |
| Descomposición | Dividir un problema complejo en subproblemas más simples |
| Scratchpad | Espacio de trabajo intermedio donde el modelo "escribe" su razonamiento |
| Faithfulness | Grado en que los pasos CoT reflejan el proceso real del modelo |

---

## 11. Referencias

1. Wei, J., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS.
2. Wang, X., et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning*. ICLR.
3. Yao, S., et al. (2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*. NeurIPS.
4. Kojima, T., et al. (2022). *Large Language Models are Zero-Shot Reasoners*. NeurIPS.
5. Turpin, M., et al. (2023). *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting*. NeurIPS.
