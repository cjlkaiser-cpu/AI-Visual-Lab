# Mixture of Experts — Especialización y Routing Disperso

> Simulación interactiva de una capa Mixture of Experts (MoE). Un router aprendido decide qué expertos procesan cada entrada, logrando capacidad masiva con costo computacional reducido. Haz click en el canvas para añadir puntos, observa cómo se asignan a expertos, y experimenta con Top-K, temperatura y balance de carga.

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

¿Cómo construyes un modelo de 1.8 trillones de parámetros (como Switch Transformer) sin necesitar 1.8 trillones de FLOPs por predicción? La respuesta es **Mixture of Experts** (MoE): en lugar de usar *todos* los parámetros para cada entrada, un **router** selecciona un subconjunto de redes especializadas (**expertos**) que procesan esa entrada específica.

Esto permite escalar la capacidad del modelo (más parámetros = más conocimiento) sin escalar proporcionalmente el costo computacional. GPT-4, Mixtral, y muchos modelos modernos usan arquitecturas MoE.

Esta simulación visualiza el routing en 2D: puntos de entrada se asignan a 2, 4 u 8 expertos, y puedes observar cómo el router aprende a particionar el espacio.

---

## 2. Conceptos Fundamentales

### 2.1 La idea central

Una capa MoE reemplaza una red feed-forward monolítica con $E$ redes especializadas (expertos) y un router que decide cuáles usar:

$$y = \sum_{i=1}^{E} g(x)_i \cdot E_i(x)$$

donde $g(x)_i$ es el peso del router para el experto $i$ y $E_i(x)$ es la salida del experto $i$.

### 2.2 Routing disperso

La clave es que el router es **disperso**: para cada entrada, solo $K$ de los $E$ expertos tienen peso no-cero. Con Top-1, solo un experto procesa cada entrada. Con Top-2, dos expertos comparten el trabajo.

$$g(x) = \text{TopK}\!\left(\text{softmax}\!\left(\frac{W_g \cdot x}{T}\right)\right)$$

### 2.3 Balance de carga

Si el router envía todo a un solo experto, los demás no se entrenan. El **load balance loss** penaliza distribuciones desiguales:

$$\mathcal{L}_{\text{balance}} = E \cdot \sum_{i=1}^{E} f_i \cdot p_i$$

donde $f_i$ es la fracción de tokens asignados al experto $i$ y $p_i$ es la probabilidad promedio del router para el experto $i$.

### 2.4 Visualización 2D

La simulación usa un espacio 2D donde:
- Cada punto $(x_1, x_2)$ es una entrada
- Cada experto tiene un color y una región del espacio
- El router colorea los puntos según a qué experto(s) se asignan

---

## 3. La Interfaz

### 3.1 Canvas principal

El canvas muestra:

- **Fondo coloreado**: regiones de Voronoi definidas por los expertos
- **Puntos**: entradas añadidas por el usuario, coloreados según el experto asignado
- **Centros de expertos**: puntos grandes que definen la posición de cada experto
- **Líneas de routing**: conexiones entre puntos y sus expertos asignados (con Top-2)
- **Regiones de decisión**: fronteras donde el router cambia de un experto a otro

### 3.2 Panel lateral

- **Ecuación**: $y = \Sigma_i g(x)_i \cdot E_i(x), \; g(x) = \text{TopK}(\text{softmax}(W_g \cdot x))$
- **Slider de expertos**: 2, 4, u 8 expertos
- **Botones Top-K**: Top-1 o Top-2
- **Sliders**: temperatura, load balance loss
- **Barras de carga**: porcentaje de inputs asignados a cada experto
- **Métricas**: puntos, pasos, router loss, expert loss, total loss

### 3.3 Barra de estado

Muestra el número de puntos, expertos activos y modo Top-K.

---

## 4. Controles Interactivos

### 4.1 Entrada interactiva

- **Click en canvas**: añade un punto de entrada en esa posición
- **Random Points**: genera puntos aleatorios
- **Clear Points**: borra todos los puntos

### 4.2 Configuración del router

| Control | Rango | Default | Efecto |
|---------|-------|---------|--------|
| Número de expertos | 2 / 4 / 8 | 4 | Cuántas redes especializadas disponibles |
| Top-K | 1 / 2 | 1 | Cuántos expertos procesan cada entrada |
| Temperatura | 0.05 – 3.0 | 1.0 | Nitidez del routing. Baja = decisiones duras; alta = suaves |
| Load Balance | 0 – 1.0 | 0.10 | Peso de la penalización de balance de carga |

### 4.3 Simulación

| Botón | Acción |
|-------|--------|
| Train Step | Ejecuta un paso de entrenamiento del router |
| Auto-Train | Entrena continuamente con animación |
| Clear Points | Borra los puntos de entrada |
| Random Points | Genera puntos distribuidos aleatoriamente |

### 4.4 Expert Load Balance

Barras horizontales coloreadas que muestran qué fracción de los puntos se asigna a cada experto. El ideal es distribución uniforme (100%/$E$ cada uno). Se muestra la **entropía del balance** y la **carga máxima**.

---

## 5. Las Matemáticas

### 5.1 El router

El router es una red lineal con softmax y temperatura:

$$g(x) = \text{softmax}\!\left(\frac{W_g \cdot x}{T}\right) \in \mathbb{R}^E$$

donde $W_g \in \mathbb{R}^{E \times d}$ son los pesos del router y $T$ es la temperatura.

### 5.2 Selección Top-K

Solo los $K$ expertos con mayor peso de routing se activan:

$$g(x)_i^{\text{sparse}} = \begin{cases} g(x)_i / \sum_{j \in \text{TopK}} g(x)_j & \text{si } i \in \text{TopK}(g(x)) \\ 0 & \text{en caso contrario} \end{cases}$$

Los pesos se renormalizan para sumar 1 entre los expertos activos.

### 5.3 Salida combinada

Con Top-K disperso:

$$y = \sum_{i \in \text{TopK}} g(x)_i^{\text{sparse}} \cdot E_i(x)$$

Solo se computan las salidas de $K$ expertos, no de los $E$ totales.

### 5.4 Loss de balance de carga

Para evitar colapso (todos los inputs a un experto):

$$\mathcal{L}_{\text{balance}} = E \cdot \sum_{i=1}^{E} f_i \cdot p_i$$

donde:
- $f_i = \frac{1}{N}\sum_{n=1}^{N} \mathbb{1}[i \in \text{TopK}(g(x_n))]$ (fracción de tokens asignados)
- $p_i = \frac{1}{N}\sum_{n=1}^{N} g(x_n)_i$ (probabilidad promedio)

### 5.5 Entropía del balance

$$H_{\text{balance}} = -\sum_{i=1}^{E} \hat{f}_i \log \hat{f}_i$$

donde $\hat{f}_i$ es la carga normalizada. Máxima entropía = balance perfecto. Entropía baja = desbalance.

---

## 6. Sonificación

### 6.1 Diseño de audio

Cada experto tiene una nota y timbre distintivos:

| Experto | Color | Nota | Tipo de onda |
|---------|-------|------|-------------|
| 1 (rojo) | #EF4444 | C4 (261 Hz) | sine |
| 2 (verde) | #22C55E | E4 (329 Hz) | triangle |
| 3 (azul) | #3B82F6 | G4 (392 Hz) | square |
| 4 (amarillo) | #EAB308 | C5 (523 Hz) | sawtooth |
| 5-8 | Colores adicionales | D5-A5 | Alternando |

### 6.2 Mapeo

| Evento | Sonido |
|--------|--------|
| Nuevo punto | Nota del experto asignado |
| Top-2 routing | Acorde de dos notas (los dos expertos) |
| Train step | Click percusivo |
| Balance mejorado | Tono ascendente sutil |

---

## 7. Guía Paso a Paso

### Paso 1: Crear puntos

1. Haz click en diferentes zonas del canvas para añadir puntos
2. Observa cómo cada punto se colorea según el experto asignado
3. Los puntos cercanos tienden a ir al mismo experto

### Paso 2: Observar el routing

1. Con 4 expertos y Top-1, cada punto va a exactamente un experto
2. Las regiones de color muestran la partición del espacio
3. Revisa las barras de carga: ¿están balanceadas?

### Paso 3: Top-1 vs. Top-2

1. Cambia a Top-2
2. Los puntos en fronteras entre regiones ahora mezclan colores
3. Las barras de carga se redistribuyen (cada punto cuenta para 2 expertos)

### Paso 4: Efecto de la temperatura

1. Con $T = 0.1$ (muy baja): routing duro, fronteras nítidas
2. Con $T = 3.0$ (alta): routing suave, fronteras difusas
3. Con $T$ muy baja y Top-1: comportamiento casi como clustering

### Paso 5: Load balance

1. Pon load balance en 0: los puntos pueden colapsar a un solo experto
2. Sube a 0.5: el entrenamiento fuerza distribución más uniforme
3. Entrena con **Auto-Train** y observa cómo las barras se equilibran

### Paso 6: Escalar expertos

1. Prueba con 2, 4 y 8 expertos
2. Con más expertos, las regiones son más pequeñas y especializadas
3. Pero el balance de carga se vuelve más difícil de mantener

---

## 8. Conceptos Avanzados

### 8.1 Switch Transformer

Fedus et al. (2022) propusieron usar Top-1 routing (solo 1 experto por token), maximizando la eficiencia:

$$\text{FLOPs}_{\text{MoE}} \approx \frac{\text{FLOPs}_{\text{dense}}}{E/K}$$

Con $E = 64$ y $K = 1$, se necesita ~1/64 del cómputo de un modelo denso equivalente.

### 8.2 Expert Choice routing

En lugar de que cada token elija su experto, cada experto elige sus tokens:

$$\text{tokens}_i = \text{TopK}_{\text{tokens}}(g_i(\text{all tokens}))$$

Esto garantiza balance perfecto pero permite que algunos tokens no sean procesados.

### 8.3 Auxiliary losses

Además del load balance, se usan:
- **Router z-loss**: penaliza logits grandes del router para estabilidad
- **Importance loss**: asegura que todos los expertos sean "importantes" en promedio

### 8.4 MoE en la práctica: Mixtral

Mixtral 8×7B (Mistral AI) usa 8 expertos de 7B cada uno con Top-2 routing. Total: 46.7B parámetros, pero solo ~13B activos por token. Rendimiento comparable a modelos densos de 70B.

---

## 9. Ejercicios

### Ejercicio 1: Regiones de Voronoi
Con 4 expertos y Top-1, crea puntos distribuidos uniformemente (usa Random Points). ¿Las regiones de decisión se parecen a un diagrama de Voronoi? ¿Por qué?

### Ejercicio 2: Colapso de routing
Fija load balance en 0 y temperatura en 0.1. Entrena 50 pasos. ¿Cuántos expertos tienen carga >5%? ¿Ocurre el colapso? Ahora sube load balance a 0.5 y repite.

### Ejercicio 3: Temperatura y especialización
Con 4 expertos y Top-1, compara $T = 0.1$ vs. $T = 2.0$. Añade un punto exactamente en la frontera entre dos regiones. ¿Cómo cambia la asignación del router?

### Ejercicio 4: Top-1 vs. Top-2 cuantitativo
Genera 100 puntos aleatorios. Con Top-1, registra la entropía del balance. Cambia a Top-2 (sin reentrenar) y registra de nuevo. ¿Cuál tiene mejor balance? ¿Por qué?

### Ejercicio 5: Eficiencia computacional
Si un modelo denso tiene $P$ parámetros y usamos MoE con $E = 8$ expertos y Top-2, ¿qué fracción de los parámetros se activa por token? Si un forward pass denso cuesta $C$ FLOPs, ¿cuántos FLOPs cuesta el MoE (aproximadamente)?

### Ejercicio 6: Diseñar regiones
Intenta colocar puntos estratégicamente para que cada experto tenga exactamente el 25% de la carga (con 4 expertos). ¿Es posible sin entrenamiento, solo con la posición de los puntos?

---

## 10. Glosario

| Término | Definición |
|---------|-----------|
| Mixture of Experts (MoE) | Arquitectura donde un router selecciona subconjuntos de expertos para cada entrada |
| Experto | Red neuronal especializada que procesa un subconjunto de entradas |
| Router (Gating) | Red que decide qué expertos procesan cada entrada |
| Top-K | Seleccionar los $K$ expertos con mayor peso de routing |
| Sparse routing | Solo un subconjunto de expertos se activa por entrada |
| Dense routing | Todos los expertos procesan cada entrada (menos eficiente) |
| Load balance | Distribución equitativa de entradas entre expertos |
| Load balance loss | Penalización que evita el colapso a pocos expertos |
| Temperatura | Controla la nitidez de la decisión del router |
| Colapso de routing | Cuando todas las entradas se asignan al mismo experto |
| Entropía del balance | Medida de cuán uniforme es la distribución de carga |
| Expert Choice | Variante donde los expertos eligen sus tokens |
| Capacidad | Factor que limita cuántos tokens puede recibir cada experto |
| Switch Transformer | MoE con Top-1 routing para máxima eficiencia |
| Mixtral | Modelo de Mistral AI que usa MoE con 8×7B expertos |
| FLOPs | Floating-point operations — medida de costo computacional |
| Softmax | Función que normaliza los pesos del router a probabilidades |
| Voronoi | Partición del espacio donde cada punto se asigna al centro más cercano |

---

## 11. Referencias

1. Shazeer, N., et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR.
2. Fedus, W., Zoph, B. & Shazeer, N. (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR.
3. Lepikhin, D., et al. (2021). *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*. ICLR.
4. Jiang, A., et al. (2024). *Mixtral of Experts*. arXiv:2401.04088.
5. Zhou, Y., et al. (2022). *Mixture-of-Experts with Expert Choice Routing*. NeurIPS.
