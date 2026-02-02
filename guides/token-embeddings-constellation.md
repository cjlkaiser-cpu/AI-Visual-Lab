# Token Embeddings Constellation — Constelaciones en el Espacio Semántico

> Simulación interactiva de word embeddings donde ~200 palabras en español se representan como vectores de 50 dimensiones, proyectados a 2D mediante PCA. Explora cómo palabras similares forman clusters (constelaciones), busca vecinos cercanos, y realiza aritmética vectorial semántica: rey − hombre + mujer ≈ reina.

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

Las computadoras no entienden palabras directamente — necesitan números. Los **word embeddings** son la solución: asignan a cada palabra un vector de números de alta dimensionalidad donde la **distancia** entre vectores refleja la **similitud semántica**.

"Gato" y "perro" están cerca porque comparten propiedades semánticas (son animales, mascotas). "Gato" y "justicia" están lejos porque no comparten contextos.

Lo más sorprendente es la **aritmética vectorial**: las relaciones semánticas se codifican como **direcciones** en el espacio. La dirección "masculino → femenino" es consistente: $\vec{\text{rey}} - \vec{\text{hombre}} + \vec{\text{mujer}} \approx \vec{\text{reina}}$.

Esta simulación genera embeddings sintéticos para ~200 palabras en español organizadas en 12 campos semánticos (animales, colores, verbos, países, familia, emociones...). Los vectores de 50 dimensiones se proyectan a 2D mediante PCA para visualizarlos como una constelación donde puedes explorar clusters, buscar vecinos, y realizar analogías.

---

## 2. Conceptos Fundamentales

### 2.1 De palabras a vectores

Un embedding es una función que asigna cada palabra a un vector denso:

$$f: \text{Vocabulario} \to \mathbb{R}^d$$

donde $d$ es la dimensión del embedding (en esta simulación, $d = 50$).

A diferencia de one-hot encoding ($d = |\text{vocab}|$, sparse), los embeddings son densos y de dimensión fija, capturando semántica en cada componente.

### 2.2 Similitud coseno

La similitud entre dos palabras se mide con el **coseno** del ángulo entre sus vectores:

$$\text{sim}(a, b) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \cdot \|\vec{b}\|}$$

- $\text{sim} = 1$: vectores paralelos (máxima similitud)
- $\text{sim} = 0$: vectores ortogonales (sin relación)
- $\text{sim} = -1$: vectores opuestos (antónimos en algunos espacios)

### 2.3 Clusters semánticos

Palabras del mismo campo semántico tienden a agruparse:

- **Animales**: gato, perro, león, águila → cluster
- **Colores**: rojo, azul, verde → cluster
- **Emociones**: feliz, triste, miedo → cluster

Estos clusters emergen naturalmente cuando los embeddings se entrenan con texto real, porque palabras similares aparecen en contextos similares.

### 2.4 Aritmética vectorial

Las relaciones semánticas se codifican como direcciones constantes:

$$\vec{\text{rey}} - \vec{\text{hombre}} + \vec{\text{mujer}} \approx \vec{\text{reina}}$$

La dirección $\vec{\text{hombre}} \to \vec{\text{mujer}}$ (género) es la misma que $\vec{\text{rey}} \to \vec{\text{reina}}$.

---

## 3. La Interfaz

### 3.1 Estructura general

| Área | Ubicación | Función |
|------|-----------|---------|
| **Canvas principal** | Izquierda | Mapa 2D de palabras como estrellas, coloreadas por campo semántico |
| **Tooltip** | Flotante | Nombre de la palabra y campo semántico al pasar el mouse |
| **Panel de controles** | Derecha (360px) | Búsqueda, aritmética, visualización, audio, vecinos |

### 3.2 La constelación

El canvas muestra las ~200 palabras como puntos en un espacio 2D:

- **Color**: cada campo semántico tiene un color distinto
- **Etiquetas**: nombres de las palabras junto a cada punto
- **Conexiones**: líneas a los K vecinos más cercanos de la palabra seleccionada
- **Resultado de analogía**: la palabra resultado se resalta con un estilo diferente (verde)
- **Zoom y pan**: navega por el espacio con el mouse

### 3.3 Header

Muestra el número total de palabras y la proyección ($50D \to 2D$).

---

## 4. Controles Interactivos

### 4.1 Buscar palabra

| Control | Función |
|---------|---------|
| **Campo de texto** | Escribe una palabra para localizarla en el mapa |
| **Resultado** | Muestra si la palabra existe y su campo semántico |

### 4.2 Aritmética vectorial

| Control | Función |
|---------|---------|
| **Input A** | Primera palabra (ej: "rey") |
| **Input B** | Palabra a restar (ej: "hombre") |
| **Input C** | Palabra a sumar (ej: "mujer") |
| **Calcular Analogía** | Computa A − B + C y encuentra la palabra más cercana al resultado |
| **Resultado** | Muestra la palabra encontrada con estilo visual |

### 4.3 Visualización

| Parámetro | Rango | Descripción |
|-----------|-------|-------------|
| **Vecinos K** | 1–15 | Número de vecinos más cercanos a mostrar |
| **Zoom** | 0.5×–3.0× | Nivel de zoom del mapa |
| **Campo semántico** | Todos, Animales, Colores, Verbos, Números, Países, Familia, Emociones | Filtrar por campo |

### 4.4 Audio

- **Volumen**: 0-100%
- **Silenciar**: toggle

### 4.5 Vecinos cercanos

| Métrica | Descripción |
|---------|-------------|
| **Palabra seleccionada** | La palabra actualmente seleccionada (clic en el mapa) |
| **Lista de vecinos** | K vecinos más cercanos con su distancia |

---

## 5. Las Matemáticas

### 5.1 PCA: de 50D a 2D

**Principal Component Analysis** encuentra las direcciones de máxima varianza en los datos:

1. Centrar los datos: $\bar{x} = \frac{1}{n}\sum_i x_i$, $\tilde{x}_i = x_i - \bar{x}$
2. Calcular la matriz de covarianza: $\Sigma = \frac{1}{n}\tilde{X}^T\tilde{X}$
3. Encontrar los eigenvectores $v_1, v_2$ de los dos mayores eigenvalores
4. Proyectar: $z_i = [v_1^T \tilde{x}_i, \; v_2^T \tilde{x}_i]$

La proyección 2D preserva las dos dimensiones con mayor varianza, pero pierde información de las otras 48 dimensiones. Palabras que aparecen lejos en 2D podrían estar cerca en 50D (y viceversa).

### 5.2 Distancia euclidiana

$$d(a, b) = \|\vec{a} - \vec{b}\| = \sqrt{\sum_{i=1}^{50}(a_i - b_i)^2}$$

Se usa para encontrar los K vecinos más cercanos (K-Nearest Neighbors).

### 5.3 Construcción de los embeddings (sintéticos)

Esta simulación genera embeddings sintéticos que preservan propiedades clave:

1. Cada campo semántico tiene un **centro** $c_f \in \mathbb{R}^{50}$ (vector aleatorio)
2. Campos similares tienen centros más cercanos (animales ↔ naturaleza, familia ↔ emociones)
3. Cada palabra tiene un vector: $\vec{w} = c_f + \epsilon$, donde $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ es ruido

$$\text{sim}_\text{intra-campo} > \text{sim}_\text{inter-campo}$$

### 5.4 Aritmética vectorial: geometría

La analogía $A - B + C \approx D$ funciona cuando existe un **paralelogramo** en el espacio:

$$\vec{A} - \vec{B} \approx \vec{D} - \vec{C}$$

Es decir, la relación entre A y B es la misma que entre D y C. La simulación busca:

$$D^* = \arg\min_{w \in \text{vocab}} \|\vec{w} - (\vec{A} - \vec{B} + \vec{C})\|$$

excluyendo A, B y C del resultado.

### 5.5 Hipótesis distribucional

La base teórica de los embeddings: "You shall know a word by the company it keeps" (Firth, 1957). Palabras que aparecen en contextos similares tienen significados similares:

- "El **gato** duerme en el sofá" y "El **perro** duerme en el sofá" → gato ≈ perro

Algoritmos como Word2Vec y GloVe formalizan esta hipótesis con funciones objetivo específicas.

---

## 6. Sonificación

### 6.1 Mapeo audio

| Evento | Sonido | Parámetro mapeado |
|--------|--------|-------------------|
| **Seleccionar palabra** | Nota | Frecuencia según la posición $y$ en el mapa 2D |
| **Vecinos cercanos** | Notas en cascada | Una nota por vecino, frecuencia según distancia |
| **Analogía exitosa** | Acorde mayor | La palabra resultado suena como resolución armónica |
| **Analogía fallida** | Tono plano | No se encontró una analogía clara |
| **Cambio de campo** | Cambio de timbre | Cada campo semántico tiene un timbre asociado |

### 6.2 Interpretación

Al explorar la constelación, el audio da una sensación de "paisaje semántico": clusters cercanos suenan armónicamente similares, y saltar entre campos distantes produce contrastes tímbricos.

---

## 7. Guía Paso a Paso

### 7.1 Explorar la constelación

1. Observa el mapa 2D: las palabras forman clusters visibles
2. Identifica los campos semánticos por color
3. Haz clic en una palabra (ej: "gato") para seleccionarla
4. Observa los K vecinos más cercanos conectados con líneas
5. La lista de vecinos muestra las palabras más similares con su distancia

### 7.2 Buscar y filtrar

1. Escribe "león" en el buscador — se resalta en el mapa
2. Selecciona "Animales" en el filtro de campo — solo se muestran los animales
3. Nota cómo los animales forman un cluster compacto
4. Cambia a "Colores" — un cluster diferente en otra región del espacio
5. Vuelve a "Todos" para ver la constelación completa

### 7.3 Aritmética vectorial

1. Escribe: A = "rey", B = "hombre", C = "mujer"
2. Pulsa **Calcular Analogía**
3. Resultado esperado: "reina" (o la palabra más cercana al vector resultante)
4. Prueba: A = "padre", B = "hombre", C = "mujer" → ¿"madre"?
5. Experimenta con otras analogías

### 7.4 Ajustar K vecinos

1. Con "gato" seleccionado, pon K = 1: solo el vecino más cercano
2. Aumenta K a 15: observa cómo el "barrio" semántico se expande
3. Con K alto, ¿empiezan a aparecer palabras de otros campos?
4. ¿A partir de qué K se "rompe" la frontera del campo semántico?

---

## 8. Conceptos Avanzados

### 8.1 Word2Vec: Skip-gram y CBOW

**Word2Vec** (Mikolov et al., 2013) entrena embeddings con dos arquitecturas:

- **Skip-gram**: dada una palabra, predice las palabras del contexto
- **CBOW**: dado el contexto, predice la palabra central

Objetivo Skip-gram:

$$\max \sum_{t} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t)$$

$$P(w_O | w_I) = \frac{\exp(\vec{w_O} \cdot \vec{w_I})}{\sum_{w \in V} \exp(\vec{w} \cdot \vec{w_I})}$$

### 8.2 GloVe: Global Vectors

**GloVe** (Pennington et al., 2014) factoriza la matriz de co-ocurrencias:

$$J = \sum_{i,j=1}^{|V|} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

donde $X_{ij}$ es cuántas veces la palabra $i$ aparece en el contexto de $j$.

### 8.3 Subword embeddings (BPE)

Los modelos modernos no usan word embeddings sino **subword** embeddings (Byte Pair Encoding):

- "desconocida" → ["des", "conoc", "ida"]
- Permite manejar palabras fuera del vocabulario
- Usado en GPT, BERT, y la mayoría de LLMs actuales

### 8.4 Embeddings contextuales

Los embeddings estáticos (Word2Vec, GloVe) asignan **un único vector** por palabra. Los embeddings contextuales (BERT, GPT) generan vectores que **cambian según el contexto**:

- "banco" (financiero) → vector cercano a "dinero"
- "banco" (mueble) → vector cercano a "parque"

### 8.5 Limitaciones de la proyección 2D

PCA preserva solo 2 de 50 dimensiones. Esto causa:

- **Falsas vecindades**: palabras que parecen cercanas en 2D pueden estar lejos en 50D
- **Falsas lejanías**: palabras distantes en 2D pueden ser vecinas en 50D
- **Clusters solapados**: clusters que son separables en 50D se pueden superponer en 2D

Alternativas: t-SNE (mejor para preservar estructura local, pero distorsiona distancias globales), UMAP (balance entre local y global).

### 8.6 Sesgo en embeddings

Los embeddings heredan los sesgos del texto de entrenamiento. Por ejemplo, "doctor" puede estar más cerca de "hombre" que de "mujer" si los datos de entrenamiento reflejan ese sesgo. Debiasing es un área activa de investigación.

---

## 9. Ejercicios

### Ejercicio 1: Explorando campos
Selecciona palabras de 3 campos diferentes (animales, colores, emociones). Para cada una, anota sus 5 vecinos más cercanos. ¿Todos los vecinos pertenecen al mismo campo? ¿Hay "intrusos" de otros campos?

### Ejercicio 2: Frontera semántica
Selecciona "agua" y aumenta K de 1 a 15. ¿En qué punto aparecen vecinos de un campo diferente al de "agua"? "Agua" pertenece a naturaleza pero también a comida — ¿aparecen vecinos de ambos campos?

### Ejercicio 3: Analogías de género
Prueba las siguientes analogías y anota el resultado:
- padre − hombre + mujer = ?
- hermano − hombre + mujer = ?
- actor − hombre + mujer = ?
- maestro − hombre + mujer = ?
¿Son todos correctos? ¿Cuáles fallan y por qué?

### Ejercicio 4: Analogías de campo
Inventa 3 analogías basadas en relaciones distintas al género:
- país − capital + capital = ? (ej: españa − madrid + paris = ?)
- animal − tierra + agua = ? (ej: gato − tierra + agua = ?)
Pruébalas en la simulación. ¿Funcionan? ¿Por qué algunas analogías son más fáciles que otras?

### Ejercicio 5: PCA y distorsión
Encuentra dos palabras que parecen cercanas en el mapa 2D pero tienen una distancia alta en la lista de vecinos (no aparecen como vecinos mutuos). Esto demuestra la distorsión de la proyección. ¿Por qué PCA puede crear estas ilusiones?

### Ejercicio 6: Simetría de similitud
Selecciona "gato" y mira si "perro" está entre sus vecinos. Luego selecciona "perro" y verifica si "gato" está entre sus vecinos. ¿La relación de vecindad es simétrica? Matemáticamente, $d(a,b) = d(b,a)$, pero ¿los K-NN son simétricos?

---

## 10. Glosario

| Término | Definición |
|---------|------------|
| **Word embedding** | Representación de una palabra como vector denso en $\mathbb{R}^d$ |
| **Similitud coseno** | Medida de similitud basada en el ángulo entre vectores: $\cos(\theta) = \frac{a \cdot b}{\|a\|\|b\|}$ |
| **PCA** | Análisis de Componentes Principales: proyección que preserva la máxima varianza |
| **Campo semántico** | Grupo de palabras relacionadas por significado (animales, colores, etc.) |
| **Cluster** | Agrupación de puntos cercanos en el espacio de embeddings |
| **K-NN** | K vecinos más cercanos: las K palabras con menor distancia al vector dado |
| **Aritmética vectorial** | Operaciones $A - B + C$ que capturan relaciones semánticas como analogías |
| **Word2Vec** | Algoritmo que entrena embeddings prediciendo contexto (Skip-gram) o palabra central (CBOW) |
| **GloVe** | Global Vectors: embeddings basados en factorización de la matriz de co-ocurrencia |
| **One-hot** | Representación sparse donde cada palabra es un vector con un solo 1 |
| **Distancia euclidiana** | $\|a - b\| = \sqrt{\sum(a_i - b_i)^2}$: distancia geométrica entre vectores |
| **Embedding contextual** | Vector que cambia según el contexto de la palabra (BERT, GPT) |
| **Subword** | Unidad sub-léxica usada en tokenización moderna (BPE) |
| **t-SNE** | Técnica de visualización no lineal que preserva estructura local |
| **UMAP** | Técnica de reducción dimensional que balancea estructura local y global |
| **Hipótesis distribucional** | Palabras con contextos similares tienen significados similares |
| **Sesgo** | Prejuicios sociales codificados en los embeddings por los datos de entrenamiento |
| **Dimensionalidad** | Número de componentes del vector de embedding ($d = 50$ en esta simulación) |

---

## 11. Referencias

1. Mikolov, T. et al. (2013). "Efficient Estimation of Word Representations in Vector Space." *ICLR Workshop*.
2. Mikolov, T. et al. (2013). "Distributed Representations of Words and Phrases and their Compositionality." *NeurIPS*.
3. Pennington, J., Socher, R. & Manning, C.D. (2014). "GloVe: Global Vectors for Word Representation." *EMNLP*.
4. Bojanowski, P. et al. (2017). "Enriching Word Vectors with Subword Information." *TACL*.
5. Bolukbasi, T. et al. (2016). "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings." *NeurIPS*.
6. van der Maaten, L. & Hinton, G. (2008). "Visualizing Data using t-SNE." *JMLR*, 9, 2579-2605.
