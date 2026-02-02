# Superposition — Dentro de la Caja Negra

> Simulación interactiva del fenómeno de superposición en redes neuronales: cómo un modelo con $N$ dimensiones puede codificar $M > N$ features usando representaciones superpuestas. Basada en el trabajo de Anthropic sobre interpretabilidad mecanicista, esta herramienta muestra la transición entre representación dedicada y superposición.

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

¿Cómo puede una red neuronal con 512 neuronas en una capa representar miles de conceptos? La respuesta es **superposición**: las features se codifican en direcciones del espacio que no son ortogonales, compartiendo las mismas dimensiones.

Este fenómeno, estudiado intensamente por Anthropic en su programa de interpretabilidad mecanicista, es clave para entender *qué sabe* una red neuronal y *cómo lo organiza internamente*. Si $M > N$ (más features que dimensiones), el modelo debe comprimir, y la interferencia resultante es el precio de la superposición.

Esta simulación implementa el **toy model** de superposición: un autoencoder lineal $\hat{x} = W^T W x$ que intenta reconstruir $M$ features usando $N$ dimensiones. Con la sparsity y la importancia relativa de las features, emerge un diagrama de fase que muestra cuándo la superposición es ventajosa.

---

## 2. Conceptos Fundamentales

### 2.1 Features y dimensiones

En una red neuronal, cada capa tiene $N$ neuronas (dimensiones). Las **features** son los conceptos que el modelo necesita representar: "es un gato", "es azul", "sentimiento positivo", etc. Si hay $M$ features y $M \leq N$, cada feature puede tener su propia neurona dedicada. Pero típicamente $M \gg N$.

### 2.2 El toy model

El modelo más simple de superposición es un autoencoder lineal:

$$\hat{x} = W^T W x$$

donde $x \in \mathbb{R}^M$ es el vector de features (sparse), $W \in \mathbb{R}^{N \times M}$ es la matriz de encoding ($M$ features → $N$ dimensiones), y $W^T$ es la matriz de decoding.

### 2.3 Sparsity

La superposición solo funciona si las features son **sparse**: la mayoría están inactivas (cero) en cualquier momento. Si todas las features estuvieran activas simultáneamente, la interferencia sería destructiva.

$$P(\text{feature}_i \neq 0) = 1 - S$$

donde $S$ es la sparsity (0 = todas activas, 1 = ninguna activa).

### 2.4 El trade-off

El modelo enfrenta una tensión:
- **Representación dedicada**: una dimensión por feature, sin interferencia pero limitado a $N$ features
- **Superposición**: más features pero con interferencia entre ellas
- La sparsity determina cuál estrategia es óptima

---

## 3. La Interfaz

### 3.1 Vista Geométrica

El canvas muestra las columnas de $W$ como vectores en 2D (cuando $N = 2$):

- **Vectores coloreados**: cada feature es un vector que apunta en una dirección
- **Longitud del vector**: importancia de la feature (columnas más largas = features mejor representadas)
- **Ángulos entre vectores**: interferencia — vectores cercanos interfieren más
- **Vectores unitarios**: referencia para los ejes ortogonales

Cuando $M \leq N$, los vectores son ortogonales (sin interferencia). Cuando $M > N$, empiezan a "apilarse", creando superposición.

### 3.2 Vista Diagrama de Fase

Muestra la transición entre representación dedicada y superposición como función de la sparsity y la ratio $M/N$:

- **Eje X**: sparsity ($S$)
- **Eje Y**: ratio $M/N$
- **Color**: grado de superposición (frío = dedicado, cálido = superpuesto)
- **Punto actual**: posición de los parámetros actuales

### 3.3 Panel lateral

- **Ecuación**: $\hat{x} = W^T W x, \; L = \|x - \hat{x}\|^2 + \lambda \cdot \text{sparsity}$
- **Sliders**: N (dimensiones), M (features), sparsity, lambda
- **Feature toggles**: botones para activar/desactivar features individuales
- **Métricas**: ratio M/N, loss total, reconstruction loss, sparsity loss, interferencia, features representadas

---

## 4. Controles Interactivos

### 4.1 Parámetros del modelo

| Control | Rango | Default | Efecto |
|---------|-------|---------|--------|
| N (dimensiones) | 2 – 10 | 2 | Dimensiones del espacio de representación |
| M (features) | 1 – 20 | 3 | Número de features a codificar |
| Sparsity (S) | 0 – 1.0 | 0.90 | Fracción de features inactivas en promedio |
| Lambda ($\lambda$) | 0 – 1.0 | 0.10 | Penalización de sparsity en la pérdida |

### 4.2 Vistas

| Botón | Vista |
|-------|-------|
| Geométrica | Vectores de features en el espacio $N$-dimensional (proyectado a 2D) |
| Diagrama de Fase | Mapa de transición entre representación y superposición |

### 4.3 Acciones

| Botón | Acción |
|-------|--------|
| Optimizar W | Ejecuta optimización de la matriz $W$ minimizando la pérdida |
| Reset | Reinicia $W$ a valores aleatorios |
| Sonificar | Reproduce las features como sonido |

### 4.4 Feature Toggles

Botones numerados y coloreados para cada feature. Al activar/desactivar una feature:
- El vector correspondiente se resalta o atenúa
- La métrica de interferencia se recalcula
- Se puede explorar qué pasa al activar features que interfieren

---

## 5. Las Matemáticas

### 5.1 Función de pérdida

$$\mathcal{L} = \mathbb{E}_x \left[ \|x - W^T W x\|^2 \right] + \lambda \cdot \text{regularización}$$

El primer término mide la calidad de reconstrucción. El segundo penaliza la complejidad.

### 5.2 Reconstrucción expandida

Para un vector de features $x$ con feature $i$ activa:

$$\hat{x}_i = (W^T W)_{ii} \cdot x_i + \sum_{j \neq i} (W^T W)_{ij} \cdot x_j$$

El primer término es la **auto-reconstrucción**. El segundo es la **interferencia** con otras features activas.

### 5.3 Interferencia

La interferencia entre features $i$ y $j$ es:

$$I_{ij} = (W^T W)_{ij} = w_i^T w_j$$

donde $w_i$ es la columna $i$ de $W$. Si $w_i \perp w_j$, no hay interferencia. Si son paralelos, la interferencia es máxima.

### 5.4 Superposición y ángulos

En 2D con $M = 3$ features, los vectores óptimos se distribuyen a 120° entre sí (como un triángulo equilátero), minimizando la interferencia mutua:

$$\cos\theta_{ij} = w_i^T w_j / (\|w_i\| \|w_j\|)$$

Para $M$ features en $N$ dimensiones, el ángulo máximo entre pares es limitado por la **simplex bound**.

### 5.5 El rol de la sparsity

La pérdida esperada de interferencia para el par $(i, j)$ es proporcional a $P(x_i \neq 0 \text{ y } x_j \neq 0)$. Con sparsity $S$:

$$P(x_i \neq 0 \text{ y } x_j \neq 0) = (1-S)^2$$

Con $S = 0.95$: $P = 0.0025$. La interferencia ocurre solo 0.25% del tiempo, haciendo la superposición viable.

### 5.6 Número efectivo de features

El modelo puede representar efectivamente hasta:

$$M_{\text{eff}} \approx \frac{N}{(1-S)^2}$$

features con interferencia tolerable. Con $N = 512$ y $S = 0.99$: $M_{\text{eff}} \approx 5{,}120{,}000$.

---

## 6. Sonificación

### 6.1 Diseño de audio

Cada feature se mapea a un sonido:

- **Frecuencia**: proporcional al índice de la feature (features bajas = graves)
- **Volumen**: proporcional a la norma del vector $\|w_i\|$ (features bien representadas suenan más fuerte)
- **Interferencia**: cuando dos features interfieren, se produce un *beating* (pulsación entre frecuencias cercanas)

### 6.2 Mapeo

| Evento | Sonido |
|--------|--------|
| Feature $i$ activa | Nota: 200 + $i \times 50$ Hz, vol $\propto \|w_i\|$ |
| Optimización paso | Click a 440 Hz |
| Convergencia | Acorde consonante |
| Interferencia alta | Beating entre notas cercanas |

---

## 7. Guía Paso a Paso

### Paso 1: Sin superposición ($M \leq N$)

1. Fija $N = 2$, $M = 2$
2. Optimiza $W$
3. Observa que los dos vectores son ortogonales (perpendiculares)
4. Loss de reconstrucción ≈ 0, interferencia ≈ 0

### Paso 2: Inicio de superposición ($M = 3, N = 2$)

1. Sube $M$ a 3 (con $N = 2$)
2. Optimiza $W$
3. Los vectores ya no pueden ser ortogonales — se distribuyen a ~120°
4. La interferencia sube ligeramente pero la reconstrucción sigue siendo buena (con sparsity alta)

### Paso 3: El rol de la sparsity

1. Con $M = 5$, $N = 2$, fija $S = 0.5$ y optimiza
2. Observa que algunos vectores son cortos (features "sacrificadas")
3. Ahora sube $S$ a 0.95 y optimiza de nuevo
4. Todos los vectores son más largos — la superposición es viable con sparsity alta

### Paso 4: Diagrama de fase

1. Cambia a vista **Diagrama de Fase**
2. Observa la frontera entre representación dedicada (frío) y superposición (cálido)
3. Mueve el slider de sparsity: el punto se desplaza horizontalmente
4. Mueve M/N: el punto se desplaza verticalmente

### Paso 5: Feature toggles

1. Vuelve a vista geométrica con $M = 5$, $N = 2$
2. Activa solo 2 features no adyacentes — baja interferencia
3. Activa 2 features adyacentes — alta interferencia
4. Esto demuestra por qué la sparsity importa: features interferentes raramente están activas juntas

### Paso 6: Escalar

1. Sube $N$ a 5 y $M$ a 20
2. Optimiza W
3. La vista 2D es una proyección, pero las métricas muestran que muchas features se representan exitosamente

---

## 8. Conceptos Avanzados

### 8.1 Polisemantismo

Cuando las features se superponen, una sola neurona puede activarse por múltiples features no relacionadas. Esto se llama **polisemantismo**: la neurona 42 podría activarse ante "puentes", "bacterias" y "acentos franceses" porque estas features comparten la misma dirección.

### 8.2 Sparse Autoencoders (SAEs)

Para descomponer las representaciones superpuestas en features interpretables, Anthropic entrena **sparse autoencoders**:

$$h = \text{ReLU}(W_{\text{enc}} x + b_{\text{enc}})$$
$$\hat{x} = W_{\text{dec}} h + b_{\text{dec}}$$

con penalización L1 en $h$ para forzar sparsity. Las columnas de $W_{\text{dec}}$ corresponden a features interpretables.

### 8.3 Transición de fase geométrica

La transición de representación dedicada a superposición es un fenómeno de umbral:

- Para $M/N < 1$: representación dedicada siempre gana
- Para $M/N$ ligeramente $> 1$: depende de la sparsity
- Para $M/N \gg 1$: superposición es inevitable si queremos representar todas las features

### 8.4 Implicaciones para la seguridad

La superposición dificulta la interpretabilidad: no podemos simplemente mirar neuronas individuales para entender qué sabe un modelo. Los **dictionary learning methods** (SAEs, etc.) intentan encontrar las "features verdaderas" escondidas en la superposición.

---

## 9. Ejercicios

### Ejercicio 1: Umbral de superposición
Con $N = 2$ y sparsity $S = 0.9$, incrementa $M$ de 1 a 10. Para cada valor, optimiza y registra la interferencia media. ¿En qué valor de $M$ la interferencia empieza a crecer significativamente?

### Ejercicio 2: Ángulos óptimos
Con $N = 2$, $M = 3$, optimiza W. Mide (visualmente) los ángulos entre los 3 vectores. ¿Son aproximadamente iguales? ¿Cuánto mide cada uno? Compara con el valor teórico de $360°/3 = 120°$.

### Ejercicio 3: Sparsity como factor habilitante
Fija $N = 2$, $M = 8$. Optimiza con $S = 0.5$ y luego con $S = 0.99$. ¿Cuántas features se "representan" (norma del vector > 0.5) en cada caso? Calcula $M_{\text{eff}} = N/(1-S)^2$ para cada caso.

### Ejercicio 4: Pérdida de reconstrucción
Con $N = 2$, $M = 5$, $S = 0.9$, optimiza y registra el reconstruction loss. Ahora sube $N$ a 5 (manteniendo $M = 5$). ¿Cuánto baja el loss? ¿Es cero?

### Ejercicio 5: Explorar el diagrama de fase
Mueve el punto en el diagrama de fase a la esquina (alta sparsity, alto M/N). ¿El color indica superposición o representación? Ahora muévelo a (baja sparsity, alto M/N). ¿Qué cambia?

### Ejercicio 6: Interferencia selectiva
Con $M = 6$, $N = 2$, optimiza W. Usando los feature toggles, encuentra el par de features con mayor interferencia (vectores más paralelos). ¿Activar ambas simultáneamente causa un error de reconstrucción visible?

---

## 10. Glosario

| Término | Definición |
|---------|-----------|
| Superposición | Codificar más features que dimensiones usando direcciones no ortogonales |
| Feature | Concepto o propiedad que el modelo necesita representar |
| Dimensión | Neurona o componente del espacio de representación |
| Sparsity | Fracción de features que están inactivas en un momento dado |
| Interferencia | Error causado por features que comparten las mismas dimensiones |
| Polisemantismo | Una neurona se activa ante múltiples features no relacionadas |
| Monosemantismo | Una neurona se activa ante una sola feature interpretable |
| Toy model | Modelo simplificado para estudiar un fenómeno aislado |
| Autoencoder lineal | Modelo $\hat{x} = W^T W x$ que reconstruye la entrada |
| Reconstruction loss | $\|x - \hat{x}\|^2$ — error de reconstrucción |
| Sparse Autoencoder (SAE) | Autoencoder con penalización L1 para encontrar features interpretables |
| Diagrama de fase | Mapa que muestra transiciones entre regímenes (dedicado vs. superpuesto) |
| Norma del vector | $\|w_i\|$ — longitud de la columna de W, mide cuán bien se representa una feature |
| Ortogonalidad | Vectores perpendiculares ($w_i^T w_j = 0$), sin interferencia |
| Simplex bound | Límite geométrico del ángulo mínimo entre $M$ vectores en $N$ dimensiones |
| Dictionary learning | Método para encontrar features interpretables en representaciones superpuestas |
| Interpretabilidad | Capacidad de entender qué ha aprendido un modelo |
| Interpretabilidad mecanicista | Programa de investigación para entender modelos a nivel de circuitos |

---

## 11. Referencias

1. Elhage, N., et al. (2022). *Toy Models of Superposition*. Anthropic Research.
2. Bricken, T., et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*. Anthropic.
3. Cunningham, H., et al. (2023). *Sparse Autoencoders Find Highly Interpretable Features in Language Models*. ICLR.
4. Templeton, A., et al. (2024). *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*. Anthropic.
5. Olah, C. (2023). *Mechanistic Interpretability, Variables, and the Importance of Interpretable Bases*. Transformer Circuits Thread.
