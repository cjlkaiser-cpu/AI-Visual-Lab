# CNN Feature Detectives — Detectives de Características Visuales

> Simulación interactiva de una red neuronal convolucional (CNN) que clasifica dígitos escritos a mano. Dibuja un dígito en el lienzo de 8×8, observa cómo los filtros convolucionales 3×3 detectan bordes y patrones, visualiza los feature maps en cada capa, y sigue el flujo de información desde píxeles crudos hasta la clasificación final.

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

Una **red neuronal convolucional** (CNN) es la arquitectura estándar para procesar imágenes. En lugar de conectar cada píxel con cada neurona (lo que sería costosísimo), una CNN aplica **filtros pequeños** (típicamente 3×3) que se deslizan sobre la imagen detectando patrones locales: bordes, esquinas, texturas.

Esta simulación implementa una CNN compacta con dos capas convolucionales y una capa fully-connected que clasifica dígitos de $8 \times 8$ píxeles. Puedes dibujar dígitos con el mouse, cargar ejemplos predefinidos (0-9), y observar en tiempo real:

- Los **filtros 3×3** aprendidos por cada capa (qué patrones buscan)
- Los **feature maps** resultantes (dónde se activan esos patrones)
- La **clasificación final** con barras de confianza para cada dígito

Los filtros de la capa 1 detectan bordes simples; los de la capa 2 combinan esos bordes en formas más complejas. Esta jerarquía de features es la esencia del deep learning visual.

---

## 2. Conceptos Fundamentales

### 2.1 La operación de convolución

Un filtro $F$ de tamaño $k \times k$ se desplaza sobre la imagen $I$:

$$(F * I)(x, y) = \sum_{i=0}^{k-1} \sum_{j=0}^{k-1} F(i, j) \cdot I(x+i, y+j)$$

Cada posición produce un solo número: la suma de productos elemento a elemento entre el filtro y la región de la imagen que cubre.

### 2.2 Filtros como detectores

Cada filtro se especializa en detectar un patrón:

| Filtro ejemplo (3×3) | Detecta |
|----------------------|---------|
| `[-1, 0, 1; -1, 0, 1; -1, 0, 1]` | Bordes verticales |
| `[-1, -1, -1; 0, 0, 0; 1, 1, 1]` | Bordes horizontales |
| `[0, -1, 0; -1, 4, -1; 0, -1, 0]` | Puntos/esquinas |

Los filtros no se diseñan manualmente: la red los **aprende** durante el entrenamiento para maximizar la precisión de clasificación.

### 2.3 Feature maps

El resultado de aplicar un filtro a toda la imagen es un **feature map**: una imagen más pequeña donde cada píxel indica cuánto se activó el filtro en esa posición. Si la capa tiene $n$ filtros, produce $n$ feature maps.

### 2.4 Jerarquía de features

- **Capa 1** (6 filtros): detecta features de bajo nivel (bordes, gradientes)
- **Capa 2** (4 filtros): combina features de la capa 1 en patrones de nivel medio (esquinas, curvas, intersecciones)
- **Capa FC**: combina toda la información espacial para la clasificación

---

## 3. La Interfaz

### 3.1 Estructura general

| Área | Ubicación | Función |
|------|-----------|---------|
| **Canvas principal** | Izquierda | Diagrama del flujo CNN: input → conv1 → conv2 → FC → clasificación |
| **Tooltip** | Flotante | Valor de activación al pasar el mouse |
| **Panel de controles** | Derecha (360px) | Dibujo, filtros, audio, barras de clasificación |

### 3.2 Visualización del flujo

El canvas muestra la arquitectura de izquierda a derecha:

1. **Input 8×8**: la imagen dibujada, ampliada para visibilidad
2. **Conv1 feature maps**: 6 mapas de $6 \times 6$ (sin padding)
3. **Conv2 feature maps**: 4 mapas de $4 \times 4$
4. **Barras de clasificación**: 10 barras (dígitos 0-9) con la predicción resaltada
5. **Flechas de conexión**: muestran el flujo de datos entre capas
6. **Filtros 3×3**: visualización opcional de los kernels aprendidos

### 3.3 Header

Muestra la predicción actual y el nivel de confianza.

---

## 4. Controles Interactivos

### 4.1 Input (8×8)

| Control | Función |
|---------|---------|
| **Lienzo de dibujo** | Canvas 160×160px mapeado a 8×8 píxeles |
| **Botones 0-9** | Cargar dígitos predefinidos del dataset |
| **Limpiar** | Borra el lienzo |
| **Shift + click** | Modo borrar (en lugar de dibujar) |

### 4.2 Filtros

| Control | Rango | Función |
|---------|-------|---------|
| **Capa activa** | -1 (todas), 0, 1 | Qué capa visualizar en detalle |
| **Mostrar filtros 3×3** | Checkbox | Visualizar los kernels convolucionales |
| **Mostrar pre-activación** | Checkbox | Ver valores antes de ReLU |

### 4.3 Audio y clasificación

- **Volumen**: 0-100%
- **Audio ON/OFF**: toggle
- **Barras de clasificación**: 10 barras horizontales mostrando la probabilidad para cada dígito (0-9)

---

## 5. Las Matemáticas

### 5.1 Arquitectura de la red

```
Input: 8×8×1
  ↓ Conv1: 6 filtros 3×3, stride 1, no padding
  ↓ ReLU
Feature maps: 6×6×6
  ↓ Conv2: 4 filtros 3×3, stride 1, no padding
  ↓ ReLU
Feature maps: 4×4×4
  ↓ Flatten: 64 valores
  ↓ FC: 64 → 10
  ↓ Softmax
Output: 10 probabilidades
```

### 5.2 Dimensiones de salida

Para una imagen de tamaño $W$ con filtro de tamaño $k$, stride $s$ y padding $p$:

$$W_\text{out} = \frac{W - k + 2p}{s} + 1$$

Con $W=8$, $k=3$, $s=1$, $p=0$: $W_\text{out} = 6$. Luego $W=6$, $k=3$: $W_\text{out} = 4$.

### 5.3 Número de parámetros

| Capa | Parámetros |
|------|-----------|
| **Conv1** | $6 \times (3 \times 3 \times 1 + 1) = 60$ |
| **Conv2** | $4 \times (3 \times 3 \times 6 + 1) = 220$ |
| **FC** | $(4 \times 4 \times 4 + 1) \times 10 = 650$ |
| **Total** | **930** |

### 5.4 ReLU como no-linealidad

$$\text{ReLU}(x) = \max(0, x)$$

Sin la no-linealidad, apilar capas convolucionales sería equivalente a una sola convolución (por linealidad). ReLU introduce la capacidad de representar funciones no lineales.

### 5.5 Softmax para clasificación

$$P(\text{dígito} = k) = \frac{e^{z_k}}{\sum_{j=0}^{9} e^{z_j}}$$

donde $z$ es la salida de la capa FC (logits).

### 5.6 Cross-entropy loss

$$L = -\sum_{k=0}^{9} y_k \log P_k = -\log P_\text{correcto}$$

donde $y$ es el vector one-hot del dígito correcto.

---

## 6. Sonificación

### 6.1 Mapeo audio

| Evento | Sonido | Parámetro mapeado |
|--------|--------|-------------------|
| **Trazo de dibujo** | Click suave | Intensidad del píxel dibujado |
| **Activación de filtro** | Nota por filtro | Magnitud de la activación máxima |
| **Clasificación** | Acorde | Las 3 clases más probables como notas |
| **Confianza alta** | Nota pura | Frecuencia clara y definida |
| **Confianza baja** | Ruido | Múltiples frecuencias simultáneas |

### 6.2 Interpretación

Cuando dibujas un dígito claro (como el "1"), escucharás un sonido definido al clasificar. Dígitos ambiguos producen sonidos más complejos y disonantes, reflejando la incertidumbre del modelo.

---

## 7. Guía Paso a Paso

### 7.1 Clasificación básica

1. Haz clic en el botón **1** para cargar un dígito predefinido
2. Observa los feature maps de la capa 1: los filtros detectan los bordes del trazo vertical
3. Observa los feature maps de la capa 2: combinan esos bordes
4. Mira las barras de clasificación: el "1" debería tener la barra más larga
5. Repite con el **0** y nota cómo los filtros responden diferente a las curvas

### 7.2 Explorar filtros

1. Activa **Mostrar filtros 3×3**
2. Selecciona **Capa activa = 0**: observa los 6 filtros de la primera capa
3. Identifica patrones: ¿cuáles detectan bordes horizontales? ¿verticales? ¿diagonales?
4. Cambia a **Capa activa = 1**: los filtros son más complejos (3×3×6 canales)

### 7.3 Dibujar propio

1. Pulsa **Limpiar**
2. Dibuja un dígito con el mouse (el lienzo mapea a 8×8)
3. Observa cómo la clasificación cambia en tiempo real mientras dibujas
4. Intenta dibujar un dígito ambiguo (entre "3" y "8") y observa las barras

### 7.4 Pre-activación vs post-activación

1. Activa **Mostrar pre-activación**
2. Nota los valores negativos (mostrados en otro color)
3. Desactiva: ReLU convierte todos los negativos en 0
4. Compara: ¿cuánta información se pierde con ReLU?

---

## 8. Conceptos Avanzados

### 8.1 Equivarianza traslacional

La convolución es **equivariante a traslaciones**: si desplazas la imagen, los feature maps se desplazan igual. Si dibujas un "1" a la izquierda o a la derecha del lienzo, los mismos filtros se activan, solo cambia la posición de la activación.

### 8.2 Compartición de parámetros

El mismo filtro 3×3 se aplica en todas las posiciones de la imagen. Esto reduce drásticamente el número de parámetros: en vez de $8 \times 8 \times 6 \times 6 = 2304$ conexiones, solo 9 pesos por filtro. La compartición de parámetros también actúa como regularización.

### 8.3 Receptive field

Cada neurona de la capa 2 "ve" una región de $5 \times 5$ de la imagen original (3×3 de la capa 2, más el contexto 3×3 de la capa 1). Este es el **receptive field**. Redes más profundas tienen receptive fields más grandes, capturando patrones más globales.

### 8.4 Max Pooling

Aunque esta simulación no usa pooling (la imagen ya es pequeña), las CNNs típicas intercalan capas de **max pooling** que reducen la resolución espacial a la mitad tomando el máximo de cada ventana $2 \times 2$:

$$\text{MaxPool}_{2\times 2}(x) = \max(x_{00}, x_{01}, x_{10}, x_{11})$$

Esto da invariancia parcial a pequeñas traslaciones y reduce el cómputo.

### 8.5 De CNNs a Vision Transformers

Los Vision Transformers (ViT) dividen la imagen en parches (como "tokens") y aplican atención. A diferencia de las CNNs, no tienen sesgo inductivo de localidad ni equivariancia traslacional built-in, pero con suficientes datos superan a las CNNs en muchos benchmarks.

---

## 9. Ejercicios

### Ejercicio 1: Dígitos similares
Carga el dígito "3" y luego el "8". Compara los feature maps de la capa 1. ¿Qué filtros se activan de forma similar? ¿Cuáles son diferentes? ¿Cómo distingue la red un 3 de un 8?

### Ejercicio 2: El poder de los bordes
Dibuja un cuadrado (contorno del "0") y un cuadrado relleno. Compara los feature maps. ¿Se activan los mismos filtros? ¿La clasificación cambia? Esto ilustra que las CNNs inicialmente detectan bordes, no regiones rellenas.

### Ejercicio 3: Conteo de parámetros
Calcula manualmente el número total de parámetros de la red (incluyendo biases). Verifica: Conv1 tiene $6 \times (3 \times 3 + 1) = 60$ parámetros, Conv2 tiene $4 \times (3 \times 3 \times 6 + 1) = 220$, FC tiene $(64 + 1) \times 10 = 650$. ¿Cuántos parámetros tiene una red fully-connected equivalente (64 inputs → 10 outputs con una capa oculta de 64)?

### Ejercicio 4: Efecto de borrar
Dibuja un "7" claro. Usa Shift+click para borrar la barra horizontal superior. ¿Cómo cambia la clasificación? ¿A qué dígito se parece un 7 sin la barra superior? Esto muestra qué features son críticas para la clasificación.

### Ejercicio 5: ReLU vs pre-activación
Con un dígito cargado, activa y desactiva "Mostrar pre-activación". Cuenta cuántos valores negativos hay en los feature maps de la capa 1. ¿Qué porcentaje de activaciones mata ReLU? Relaciona esto con el problema de "dying ReLU".

### Ejercicio 6: Receptive field
Dibuja un solo píxel brillante en el centro del lienzo 8×8. ¿Cuántas posiciones del feature map de capa 1 se activan? ¿Y del feature map de capa 2? Esto visualiza el receptive field: el área de la imagen que influye en cada neurona.

---

## 10. Glosario

| Término | Definición |
|---------|------------|
| **CNN** | Red Neuronal Convolucional: arquitectura que usa filtros locales para procesar imágenes |
| **Convolución** | Operación que aplica un filtro deslizante sobre una imagen, produciendo un feature map |
| **Filtro/Kernel** | Matriz pequeña de pesos (típicamente 3×3) que detecta un patrón local |
| **Feature map** | Resultado de aplicar un filtro a una imagen: mapa de activaciones |
| **Stride** | Paso del desplazamiento del filtro (stride 1 = un píxel a la vez) |
| **Padding** | Píxeles añadidos al borde de la imagen para controlar el tamaño de salida |
| **ReLU** | $\max(0, x)$: función de activación que elimina valores negativos |
| **Max Pooling** | Reducción espacial tomando el máximo de cada ventana |
| **Fully Connected** | Capa donde cada neurona está conectada a todas las de la capa anterior |
| **Softmax** | Convierte logits en probabilidades que suman 1 |
| **Receptive field** | Región de la imagen original que influye en una neurona de una capa profunda |
| **Equivarianza** | Si la entrada se traslada, la salida se traslada de la misma manera |
| **Feature hierarchy** | Capas tempranas detectan features simples, capas profundas features complejas |
| **One-hot** | Vector con un 1 en la posición del dígito correcto y 0s en el resto |
| **Logits** | Salida cruda de la capa final antes de aplicar softmax |
| **Cross-entropy** | Función de pérdida estándar para clasificación multiclase |
| **Parámetros compartidos** | Mismo filtro aplicado en todas las posiciones (reduce parámetros) |
| **Pre-activación** | Valor de la neurona antes de aplicar la función de activación |

---

## 11. Referencias

1. LeCun, Y. et al. (1998). "Gradient-Based Learning Applied to Document Recognition." *Proceedings of the IEEE*, 86(11), 2278-2324.
2. Krizhevsky, A., Sutskever, I. & Hinton, G. (2012). "ImageNet Classification with Deep Convolutional Neural Networks." *NeurIPS*.
3. Zeiler, M.D. & Fergus, R. (2014). "Visualizing and Understanding Convolutional Networks." *ECCV*.
4. He, K. et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.
5. Dosovitskiy, A. et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR*.
6. Olah, C. et al. (2017). "Feature Visualization." *Distill*.
