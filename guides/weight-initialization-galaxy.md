# Weight Initialization Galaxy — Propagación de Varianza en Redes Profundas

> Simulación interactiva que visualiza cómo diferentes esquemas de inicialización de pesos afectan la propagación de señales a través de redes neuronales profundas. Observa en tiempo real cómo las activaciones explotan, se desvanecen o se mantienen estables capa por capa — representadas como histogramas galácticos con partículas estelares.

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

Antes de que una red neuronal aprenda cualquier cosa, sus pesos deben recibir **valores iniciales**. Esta decisión, que parece trivial, determina si el entrenamiento será exitoso o fallará catastróficamente. Pesos demasiado grandes causan que las señales exploten exponencialmente; pesos demasiado pequeños hacen que colapsen a cero.

Esta simulación te permite ver ese fenómeno directamente. Construyes una red de hasta 30 capas y 256 neuronas por capa, seleccionas un método de inicialización y una función de activación, y observas cómo un batch de datos se transforma al pasar por cada capa. Los histogramas de activaciones de cada capa se muestran como "galaxias" — distribuciones de puntos que pueden expandirse, comprimirse o mantenerse estables.

La clave está en una ecuación simple: $\text{Var}[a_l] = n_{in} \cdot \text{Var}[w] \cdot \text{Var}[a_{l-1}]$. Si el producto $n_{in} \cdot \text{Var}[w] \neq 1$, la varianza cambia exponencialmente con la profundidad.

---

## 2. Conceptos Fundamentales

### 2.1 El problema de la inicialización

En una capa densa, la pre-activación de la neurona $j$ es:

$$z_j = \sum_{i=1}^{n} w_{ij} a_i + b_j$$

Si los pesos $w_{ij}$ y las activaciones $a_i$ son independientes con media cero:

$$\text{Var}[z_j] = n_{in} \cdot \text{Var}[w] \cdot \text{Var}[a]$$

Para que la varianza se mantenga estable capa a capa, necesitamos:

$$n_{in} \cdot \text{Var}[w] = 1$$

### 2.2 Explosión y desvanecimiento

Si $n_{in} \cdot \text{Var}[w] > 1$, la varianza crece exponencialmente:

$$\text{Var}[a_L] = (n \cdot \text{Var}[w])^L \cdot \text{Var}[a_0] \to \infty$$

Si $n_{in} \cdot \text{Var}[w] < 1$, la varianza decrece exponencialmente:

$$\text{Var}[a_L] = (n \cdot \text{Var}[w])^L \cdot \text{Var}[a_0] \to 0$$

En una red de 30 capas, incluso un factor de $1.5$ por capa resulta en $1.5^{30} \approx 191000$ — explosión total.

### 2.3 Los tres esquemas clásicos

| Método | Varianza | Diseñado para |
|--------|----------|---------------|
| Xavier/Glorot | $\frac{2}{n_{in} + n_{out}}$ | Sigmoid, Tanh |
| He/Kaiming | $\frac{2}{n_{in}}$ | ReLU |
| LeCun | $\frac{1}{n_{in}}$ | SELU |

---

## 3. La Interfaz

### 3.1 Estructura general

| Área | Ubicación | Función |
|------|-----------|---------|
| **Canvas principal** | Izquierda | Histogramas de activaciones por capa como "galaxias" |
| **Tooltip** | Sobre histograma hovereado | Varianza, media, min/max de la capa |
| **Badge de estado** | Canvas | Indica "Estable", "Vanishing" o "Exploding" |
| **Panel de controles** | Derecha (360px) | Inicialización, activación, arquitectura, métricas |

### 3.2 Los histogramas galácticos

Cada capa se representa como un histograma vertical de las activaciones del batch:

- **Histograma ancho y centrado**: distribución saludable (varianza ≈ 1)
- **Histograma que colapsa a una línea**: señal desvanecida (varianza → 0)
- **Histograma que se desborda**: señal explosiva (varianza → ∞)
- **Partículas estelares**: puntos individuales del batch que flotan alrededor del histograma
- **Glow**: resplandor en los histogramas proporcional a la varianza

### 3.3 Gráfica de varianza por capa

Miniatura que muestra $\log(\text{Var})$ en función del número de capa. Una línea horizontal indica estabilidad; una línea descendente indica vanishing; ascendente indica explosión.

### 3.4 Diagnóstico

Panel con: estado general, varianza de entrada, varianza de salida, ratio salida/entrada, presencia de NaN, varianza media, y número de capas estables.

---

## 4. Controles Interactivos

### 4.1 Método de inicialización

| Método | Distribución | Descripción |
|--------|-------------|-------------|
| **Xavier/Glorot** | $\mathcal{N}(0, 2/(n_{in}+n_{out}))$ | Equilibra forward y backward pass |
| **He/Kaiming** | $\mathcal{N}(0, 2/n_{in})$ | Compensa la "muerte" de mitad de ReLU |
| **LeCun** | $\mathcal{N}(0, 1/n_{in})$ | Variante conservadora para SELU |
| **Normal(0, 1)** | $\mathcal{N}(0, 1)$ | Demasiado grande — explosión garantizada |
| **Normal(0, 0.01)** | $\mathcal{N}(0, 0.0001)$ | Demasiado pequeño — colapso garantizado |
| **Zeros** | $w = 0$ | Sin ruptura de simetría — la red no aprende |

### 4.2 Función de activación

| Activación | Efecto en varianza |
|------------|-------------------|
| **ReLU** | Elimina valores negativos → reduce varianza a la mitad |
| **Sigmoid** | Comprime a $(0,1)$ → reduce varianza drásticamente |
| **Tanh** | Comprime a $(-1,1)$ → reduce varianza |
| **Lineal** | Sin efecto → solo la inicialización importa |

### 4.3 Arquitectura

| Control | Rango | Default |
|---------|-------|---------|
| Profundidad | 2 – 30 capas | 10 |
| Ancho | 8 – 256 neuronas | 64 |
| Batch size | 1 – 128 | 32 |

### 4.4 Control de propagación

| Botón | Acción |
|-------|--------|
| **Propagar** | Inicializa pesos y propaga un batch completo instantáneamente |
| **Animar** | Propaga capa por capa con animación visual |
| **Comparar Todo** | Ejecuta todas las combinaciones de inicialización + activación |

### 4.5 Opciones visuales

- **Partículas estelares**: muestra puntos individuales del batch flotando alrededor del histograma
- **Glow en histogramas**: añade un efecto de resplandor proporcional a la energía de cada capa

---

## 5. Las Matemáticas

### 5.1 Derivación de Xavier/Glorot

Para una capa con $n_{in}$ entradas y $n_{out}$ salidas, queremos mantener la varianza en el forward pass:

$$\text{Var}[z] = n_{in} \cdot \text{Var}[w] \cdot \text{Var}[a] \implies \text{Var}[w] = \frac{1}{n_{in}}$$

Y también en el backward pass:

$$\text{Var}[\delta] = n_{out} \cdot \text{Var}[w] \cdot \text{Var}[\delta'] \implies \text{Var}[w] = \frac{1}{n_{out}}$$

Como no podemos satisfacer ambas simultáneamente, Xavier usa el **promedio**:

$$\text{Var}[w] = \frac{2}{n_{in} + n_{out}}$$

### 5.2 Derivación de He/Kaiming

ReLU elimina todos los valores negativos, reduciendo la varianza a la mitad:

$$\text{Var}[\text{ReLU}(z)] = \frac{1}{2}\text{Var}[z]$$

Para compensar, He duplica la varianza de Xavier (solo para forward):

$$\text{Var}[w] = \frac{2}{n_{in}}$$

### 5.3 Factor de escala por capa

Para activación lineal, el factor de escala por capa es:

$$\alpha = n_{in} \cdot \text{Var}[w]$$

| Método | $\alpha$ (con $n_{in} = 64$) |
|--------|------|
| Xavier | $64 \cdot 2/128 = 1.0$ |
| He | $64 \cdot 2/64 = 2.0$ |
| LeCun | $64 \cdot 1/64 = 1.0$ |
| Normal(0,1) | $64 \cdot 1 = 64$ |
| Normal(0,0.01) | $64 \cdot 0.0001 = 0.0064$ |

Con 10 capas: Normal(0,1) → $64^{10} \approx 10^{18}$ (NaN). Normal(0,0.01) → $0.0064^{10} \approx 10^{-22}$ (cero).

### 5.4 Efecto de la función de activación

La varianza después de aplicar la activación $\sigma$ es:

$$\text{Var}[\sigma(z)] = \mathbb{E}[\sigma(z)^2] - \mathbb{E}[\sigma(z)]^2$$

Para Sigmoid con $z \sim \mathcal{N}(0, 1)$: $\text{Var}[\sigma(z)] \approx 0.05$. Esto significa que sigmoid aplasta la varianza a $1/20$ de su valor original **en cada capa**.

### 5.5 El problema de simetría con zeros

Si todos los pesos son cero, todas las neuronas de una capa computan la misma función. Sus gradientes son idénticos, se actualizan de forma idéntica, y **nunca se diferencian**. La red tiene efectivamente una sola neurona por capa.

---

## 6. Sonificación

### 6.1 Nota por capa

Cada capa produce una nota cuya frecuencia depende de la varianza:

- **Varianza estable** ($\approx 1$): nota media ($\sim 440$ Hz), armónica
- **Varianza alta** (explosión): nota aguda, volumen creciente
- **Varianza baja** (vanishing): nota grave, volumen decreciente
- **NaN/Infinito**: silencio abrupto (la señal murió)

### 6.2 Secuencia de propagación

Durante la animación, las notas suenan en secuencia (una por capa), creando una melodía ascendente (explosión), descendente (vanishing) o plana (estable).

### 6.3 Acorde de diagnóstico

Al completar la propagación:
- **Estable**: acorde mayor consonante
- **Vanishing**: secuencia descendente melancólica
- **Exploding**: cluster disonante ascendente

---

## 7. Guía Paso a Paso

### Experimento 1: Xavier + ReLU (el error clásico)

1. Selecciona **Xavier** y **ReLU**, profundidad 10, ancho 64
2. Click en **Propagar**
3. Observa los histogramas: la varianza debería decrecer ligeramente
4. Xavier está diseñado para sigmoid/tanh, no para ReLU
5. Cambia a **He** y propaga de nuevo: los histogramas se mantienen más estables
6. **Conclusión**: usar el esquema correcto para cada activación importa

### Experimento 2: Normal(0,1) — la explosión

1. Selecciona **Normal(0, 1)** y **Lineal**, profundidad 10
2. Propaga — las últimas capas tendrán varianza astronómica
3. La gráfica de varianza será una línea ascendente exponencial
4. Con profundidad 20, el diagnóstico mostrará NaN
5. $n_{in} \cdot \text{Var}[w] = 64 \cdot 1 = 64$ → la varianza se multiplica por 64 en cada capa

### Experimento 3: Normal(0, 0.01) — el colapso

1. Selecciona **Normal(0, 0.01)** y **Lineal**, profundidad 10
2. Propaga — los histogramas colapsan rápidamente a una línea vertical en cero
3. Con sigmoid, es aún peor: las activaciones se concentran en $\sigma(0) = 0.5$ (un valor constante)
4. **Conclusión**: pesos demasiado pequeños matan las señales

### Experimento 4: La profundidad amplifica

1. Usa **Xavier + Sigmoid**, profundidad 5 → razonablemente estable
2. Sube a profundidad 15 → se nota el vanishing
3. Profundidad 30 → colapso total
4. Repite con **He + ReLU** profundidad 30 → mucho más estable
5. **Conclusión**: la profundidad amplifica cualquier desequilibrio

### Experimento 5: Comparar Todo

1. Click en **Comparar Todo**
2. La simulación muestra una cuadrícula con todas las combinaciones de inicialización × activación
3. Identifica las combinaciones "verdes" (estables) y "rojas" (inestables)
4. He + ReLU y Xavier + Tanh deberían ser las más estables

### Experimento 6: El efecto del ancho

1. Fija He + ReLU, profundidad 10
2. Prueba ancho 8, luego 64, luego 256
3. Observa la gráfica de varianza: ¿el ancho afecta la estabilidad?
4. La fórmula de He ajusta $\text{Var}[w] = 2/n_{in}$, compensando el ancho
5. Sin este ajuste (Normal(0,1)), redes más anchas explotan más rápido

---

## 8. Conceptos Avanzados

### 8.1 Inicialización ortogonal

Además de los métodos basados en varianza, se puede inicializar con matrices **ortogonales** ($W^T W = I$). Esto preserva exactamente las normas de los vectores y es óptimo para redes lineales profundas. La descomposición QR de una matriz gaussiana aleatoria produce una inicialización ortogonal.

### 8.2 Batch Normalization como solución

**Batch Normalization** (Ioffe & Szegedy, 2015) normaliza las activaciones en cada capa a media 0 y varianza 1 durante el entrenamiento. Esto hace que la red sea mucho menos sensible a la inicialización — pero no la hace irrelevante (la inicialización aún afecta la velocidad de convergencia).

### 8.3 Layer Normalization y alternativas

| Técnica | Normaliza sobre | Uso principal |
|---------|-----------------|---------------|
| Batch Norm | Batch × Features | CNNs |
| Layer Norm | Features | Transformers |
| Group Norm | Grupos de features | CNNs con batch pequeño |
| RMS Norm | RMS de features | LLMs modernos |

### 8.4 Residual connections y el problema de profundidad

Las **conexiones residuales** ($a_l = f(a_{l-1}) + a_{l-1}$) de ResNet mitigan el vanishing gradient al crear un "atajo" para las señales. La inicialización sigue siendo importante — He se diseñó específicamente pensando en ResNets.

### 8.5 Data-dependent initialization

Métodos modernos como **LSUV** (Layer-Sequential Unit Variance) inicializan los pesos iterativamente, propagando datos reales y ajustando la escala de cada capa hasta que la varianza sea exactamente 1. Esto es más robusto que los métodos analíticos cuando las suposiciones (distribución gaussiana, independencia) no se cumplen.

---

## 9. Ejercicios

### Ejercicio 1: Cálculo de varianza

Para una red de ancho 32 con Normal(0, 1) y activación lineal, calcula $\text{Var}[a_5]$ si $\text{Var}[a_0] = 1$. Verifica tu cálculo configurando la simulación con esos parámetros y leyendo la varianza de la capa 5.

### Ejercicio 2: Diseña tu propio esquema

Si tu red tiene capas de ancho variable (32, 64, 128, 64, 32), ¿qué varianza de pesos usarías en cada capa para mantener la varianza constante con activación lineal? Compara con lo que He y Xavier producen.

### Ejercicio 3: Sigmoid vs. ReLU con Xavier

Fija Xavier, profundidad 20, ancho 64. Propaga con Sigmoid y anota la varianza de las capas 5, 10, 15 y 20. Repite con ReLU. ¿Para cuál función Xavier mantiene mejor la varianza? ¿Por qué?

### Ejercicio 4: El punto de quiebre

Con Normal(0, 1) y activación lineal, ¿cuál es la profundidad máxima antes de obtener NaN? Prueba con ancho 8, 64 y 256. ¿Redes más estrechas aguantan más capas o menos?

### Ejercicio 5: Zeros y simetría

Inicializa con Zeros y cualquier activación, profundidad 5. Observa los histogramas: ¿todas las neuronas tienen el mismo valor? Intenta entrenar (en la simulación de Backpropagation Flow) una red inicializada con zeros. ¿Qué ocurre con el loss?

### Ejercicio 6: Efecto del batch size

Fija He + ReLU, profundidad 15. Propaga con batch size 1, 32 y 128. ¿Cambia el diagnóstico? ¿La varianza estimada es más ruidosa con batch size pequeño? Explica por qué.

---

## 10. Glosario

| Término | Definición |
|---------|------------|
| **Inicialización de pesos** | Asignación de valores iniciales a los parámetros de la red antes del entrenamiento |
| **Varianza** ($\text{Var}$) | Medida de dispersión: $\text{Var}[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$ |
| **Xavier/Glorot** | Inicialización con $\text{Var}[w] = 2/(n_{in}+n_{out})$; óptima para sigmoid/tanh |
| **He/Kaiming** | Inicialización con $\text{Var}[w] = 2/n_{in}$; óptima para ReLU |
| **LeCun** | Inicialización con $\text{Var}[w] = 1/n_{in}$; variante para SELU |
| **Explosión de activaciones** | Varianza que crece exponencialmente con la profundidad |
| **Desvanecimiento de activaciones** | Varianza que decrece exponencialmente con la profundidad |
| **Ruptura de simetría** | Necesidad de que los pesos iniciales sean diferentes para que las neuronas se especialicen |
| **NaN** | Not a Number; ocurre cuando los valores exceden el rango del punto flotante |
| **Batch size** | Número de muestras propagadas simultáneamente para estimar la distribución |
| **Factor de escala por capa** ($\alpha$) | $n_{in} \cdot \text{Var}[w]$; debe ser $\approx 1$ para estabilidad |
| **Batch Normalization** | Técnica que normaliza activaciones a media 0 y varianza 1 en cada capa |
| **Conexión residual** | Atajo que suma la entrada a la salida: $a_l = f(a_{l-1}) + a_{l-1}$ |
| **Inicialización ortogonal** | Pesos como matrices ortogonales que preservan normas exactamente |
| **Histograma de activaciones** | Distribución de los valores de salida de una capa para un batch dado |
| **Diagnóstico** | Clasificación del estado de propagación: Estable, Vanishing o Exploding |
| **Profundidad** | Número de capas de la red; amplifica cualquier desequilibrio en la varianza |
| **Partículas estelares** | Representación visual de puntos individuales del batch en el histograma |

---

## 11. Referencias

1. **Glorot, X. & Bengio, Y.** (2010). "Understanding the Difficulty of Training Deep Feedforward Neural Networks." *Proceedings of AISTATS 2010*.

2. **He, K., Zhang, X., Ren, S. & Sun, J.** (2015). "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." *ICCV 2015*. arXiv:1502.01852.

3. **LeCun, Y., Bottou, L., Orr, G. B. & Müller, K.-R.** (1998). "Efficient BackProp." *Neural Networks: Tricks of the Trade*, Springer, 9–50.

4. **Ioffe, S. & Szegedy, C.** (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." *Proceedings of ICML 2015*. arXiv:1502.03167.

5. **Saxe, A. M., McClelland, J. L. & Ganguli, S.** (2014). "Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks." *ICLR 2014*. arXiv:1312.6120.

6. **Mishkin, D. & Matas, J.** (2016). "All You Need is a Good Init." *ICLR 2016*. arXiv:1511.06422.

7. **Goodfellow, I., Bengio, Y. & Courville, A.** (2016). *Deep Learning*, Section 8.4: Parameter Initialization Strategies. MIT Press.
