# Knowledge Distillation — Maestro y Alumno

> Simulación interactiva de destilación de conocimiento: una red grande (Teacher) transfiere su conocimiento a una red pequeña (Student) usando soft labels suavizadas con temperatura. Observa cómo el alumno aprende no solo *qué* clasificar, sino *cómo piensa* el maestro sobre las relaciones entre clases.

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

Entrenar un modelo grande produce excelentes resultados, pero desplegarlo en producción (un teléfono móvil, un navegador, un microcontrolador) puede ser prohibitivo por su tamaño y consumo energético. La **destilación de conocimiento**, propuesta por Hinton et al. (2015), ofrece una solución elegante: entrenar una red pequeña para que imite el comportamiento de la red grande.

La clave está en las **soft labels**: en lugar de enseñar al alumno solo la respuesta correcta ("esto es un 3"), le enseñamos la distribución completa del maestro ("esto es 90% un 3, 5% un 8, 3% un 5..."). Estas probabilidades suaves contienen información rica sobre las relaciones entre clases.

Esta simulación muestra el proceso para clasificación de dígitos (0-9), donde un Teacher grande produce soft labels que un Student pequeño aprende a imitar.

---

## 2. Conceptos Fundamentales

### 2.1 El problema de compresión

Un Teacher con millones de parámetros codifica patrones complejos:

$$P_T(y | x) = \text{softmax}(z_T)$$

donde $z_T$ son los logits del Teacher. Queremos un Student con muchos menos parámetros que produzca:

$$P_S(y | x) \approx P_T(y | x)$$

### 2.2 Temperatura y suavizado

Aplicar softmax con **temperatura** $T > 1$ suaviza la distribución:

$$\sigma(z_i; T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

Con $T = 1$: distribución original (peaked). Con $T = 5$: distribución más uniforme, revelando las relaciones entre clases. Con $T = 20$: casi uniforme.

### 2.3 El conocimiento "oscuro"

Cuando el Teacher dice que un "3" tiene 5% de probabilidad de ser un "8", está comunicando **dark knowledge**: el 3 se parece al 8 más que al 7. Esta información se pierde con hard labels (1-hot) pero se preserva con soft labels.

### 2.4 Clasificación de dígitos

La simulación usa una versión simplificada de dígitos 8×8 (estilo sklearn digits). Cada dígito tiene:
- 64 features de entrada (8×8 píxeles)
- 10 clases de salida (dígitos 0-9)
- El Teacher es una red más grande; el Student es más compacta

---

## 3. La Interfaz

### 3.1 Canvas principal

El canvas muestra tres zonas:

- **Izquierda**: el dígito de entrada como imagen 8×8
- **Centro**: distribuciones de probabilidad del Teacher (barras azules) y Student (barras verdes) para las 10 clases
- **Derecha**: visualización de la pérdida y convergencia del entrenamiento

Las barras muestran cómo las soft labels del Teacher difieren de un 1-hot encoding, y cómo el Student converge hacia las distribuciones del Teacher.

### 3.2 Panel lateral

- **Ecuación**: $L = \alpha \cdot KL(\sigma(z_T/T) \| \sigma(z_S/T)) \cdot T^2 + (1-\alpha) \cdot CE(y, \sigma(z_S))$
- **Selector de dígito**: 0-9
- **Sliders**: temperatura $T$, peso $\alpha$
- **Toggle**: usar soft labels
- **Métricas**: época, KL divergencia, accuracy Teacher/Student, loss total

### 3.3 Barra de estado

Muestra la época actual y la divergencia KL entre Teacher y Student.

---

## 4. Controles Interactivos

### 4.1 Entrada

Selector de dígito (0-9): elige qué dígito clasificar. Cada dígito produce una distribución de soft labels diferente.

### 4.2 Parámetros de destilación

| Control | Rango | Default | Efecto |
|---------|-------|---------|--------|
| Temperatura $T$ | 1 – 20 | 5 | Suaviza las distribuciones. $T > 1$ revela más dark knowledge |
| Peso $\alpha$ | 0 – 1 | 0.70 | Balance entre soft loss ($\alpha$) y hard loss ($1 - \alpha$) |
| Soft labels | On/Off | On | Si Off, solo usa hard labels (CE con etiqueta real) |

### 4.3 Entrenamiento

| Botón | Acción |
|-------|--------|
| Entrenar Alumno | Inicia entrenamiento continuo con animación |
| 1 Paso | Ejecuta una sola época de entrenamiento |
| Reset | Reinicia pesos del Student |

### 4.4 Audio

- **Volumen** (0–100%)
- **Silenciar**

---

## 5. Las Matemáticas

### 5.1 Función de pérdida de destilación

La pérdida total combina dos componentes:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{soft}} + (1 - \alpha) \cdot \mathcal{L}_{\text{hard}}$$

### 5.2 Soft loss (KL divergencia)

$$\mathcal{L}_{\text{soft}} = T^2 \cdot \text{KL}\!\left(\sigma(z_T / T) \,\|\, \sigma(z_S / T)\right)$$

$$= T^2 \sum_{i=1}^{C} \sigma(z_{T,i} / T) \cdot \log \frac{\sigma(z_{T,i} / T)}{\sigma(z_{S,i} / T)}$$

El factor $T^2$ compensa que los gradientes de la softmax suavizada son $1/T^2$ más pequeños.

### 5.3 Hard loss (Cross-Entropy)

$$\mathcal{L}_{\text{hard}} = -\sum_{i=1}^{C} y_i \cdot \log \sigma(z_{S,i})$$

donde $y$ es la etiqueta one-hot verdadera.

### 5.4 Gradientes del Student

El gradiente de la soft loss respecto a los logits del Student $z_S$ es:

$$\frac{\partial \mathcal{L}_{\text{soft}}}{\partial z_{S,i}} = T \left( \sigma(z_{S,i}/T) - \sigma(z_{T,i}/T) \right)$$

Cuando $T$ es grande, $\sigma(z/T) \approx \frac{1}{C} + \frac{z}{CT}$, y el gradiente se simplifica a:

$$\frac{\partial \mathcal{L}_{\text{soft}}}{\partial z_{S,i}} \approx \frac{1}{C}(z_{S,i} - z_{T,i})$$

Es decir, a temperatura alta, el Student simplemente trata de igualar los logits del Teacher.

### 5.5 KL Divergencia

La divergencia KL mide cuán diferente es la distribución del Student respecto al Teacher:

$$\text{KL}(P_T \| P_S) = \sum_i P_T(i) \log \frac{P_T(i)}{P_S(i)} \geq 0$$

$\text{KL} = 0$ cuando las distribuciones son idénticas. La simulación muestra este valor en tiempo real durante el entrenamiento.

---

## 6. Sonificación

### 6.1 Diseño de audio

La simulación usa armónicos para representar las distribuciones:

- **Teacher**: acorde cuyas frecuencias son proporcionales a las probabilidades de cada clase
- **Student**: mismo esquema, pero evoluciona con el entrenamiento
- **Convergencia**: cuando el Student se acerca al Teacher, los sonidos convergen
- **KL divergencia**: nota baja cuyo volumen es proporcional a la KL — se apaga al converger

### 6.2 Mapeo

| Evento | Frecuencia | Tipo |
|--------|-----------|------|
| Clase $i$ (Teacher) | 200 + $i \times 80$ Hz, vol $\propto P_T(i)$ | sine |
| Clase $i$ (Student) | 200 + $i \times 80$ Hz, vol $\propto P_S(i)$ | triangle |
| Época completada | 440 Hz click | sine |
| KL baja (<0.01) | Acorde de resolución | sine |

---

## 7. Guía Paso a Paso

### Paso 1: Observar el Teacher

1. Selecciona dígito 3
2. Observa la distribución del Teacher: alta probabilidad en "3", algo en "8", "5", "9"
3. Estas relaciones son el "dark knowledge"

### Paso 2: Efecto de la temperatura

1. Con $T = 1$: la distribución es peaked (90%+ en la clase correcta)
2. Sube $T$ a 5: las probabilidades se redistribuyen, mostrando relaciones
3. Sube $T$ a 20: distribución casi uniforme — demasiado suave

### Paso 3: Entrenar con soft labels

1. Fija $T = 5$, $\alpha = 0.7$
2. Pulsa **Entrenar Alumno**
3. Observa cómo las barras del Student convergen hacia las del Teacher
4. La KL divergencia baja progresivamente

### Paso 4: Solo hard labels

1. Reset, desactiva **Usar soft labels** (o pon $\alpha = 0$)
2. Entrena de nuevo
3. El Student aprende la clasificación correcta pero no las relaciones entre clases
4. Compara la KL: será más alta que con soft labels

### Paso 5: Explorar el peso α

1. Con $\alpha = 1.0$: solo soft loss — el Student imita al Teacher perfectamente
2. Con $\alpha = 0.5$: balance — buena clasificación y buena imitación
3. Con $\alpha = 0$: solo hard loss — no hay destilación

### Paso 6: Diferentes dígitos

1. Prueba cada dígito (0-9)
2. Observa cuáles tienen más dark knowledge (distribuciones más ricas)
3. El dígito "1" tiene poca ambigüedad; el "8" tiene mucha

---

## 8. Conceptos Avanzados

### 8.1 ¿Por qué funciona?

El Teacher comprime su conocimiento en la distribución de logits. Entrenar con hard labels solo da 1 bit de información por ejemplo (la clase correcta). Entrenar con soft labels da $\log_2 C$ bits (la distribución completa).

### 8.2 Feature-based distillation

Además de las predicciones finales, se puede hacer que el Student imite las representaciones internas del Teacher:

$$\mathcal{L}_{\text{feature}} = \|f_S(x) - \phi(f_T(x))\|^2$$

donde $\phi$ es una transformación que ajusta dimensiones.

### 8.3 Self-distillation

Sorprendentemente, un modelo puede destilarse a sí mismo: entrenar una copia del mismo modelo con sus propias soft labels. Esto mejora la generalización, sugiriendo que la destilación actúa como regularización.

### 8.4 Destilación en LLMs

La destilación se usa masivamente en LLMs:
- GPT-4 → modelos más pequeños
- Modelos open-source entrenados con outputs de modelos propietarios
- On-device models (LLMs en teléfono) necesitan destilación agresiva

---

## 9. Ejercicios

### Ejercicio 1: Temperatura óptima
Entrena el Student con $T = 1, 2, 5, 10, 20$ (fija $\alpha = 0.7$) durante 50 épocas cada uno. Registra la KL final y el accuracy del Student. ¿Cuál temperatura produce el mejor Student?

### Ejercicio 2: Dark knowledge del 8
Selecciona el dígito 8 con $T = 5$. Lista las 3 clases con mayor probabilidad después de "8". ¿Tiene sentido visual? (¿El 8 se parece al 0, al 3, al 9?)

### Ejercicio 3: Información en soft labels
Para el dígito 3 con $T = 1$ y $T = 5$, calcula la entropía $H = -\sum p_i \log p_i$ de la distribución del Teacher. ¿Cuánta más información (bits) contiene la distribución suavizada?

### Ejercicio 4: α como trade-off
Entrena con $\alpha = 0, 0.25, 0.5, 0.75, 1.0$. Para cada valor, registra accuracy y KL. Grafica ambas métricas vs. $\alpha$. ¿Hay un $\alpha$ óptimo?

### Ejercicio 5: Sin soft labels
Compara entrenar con soft labels ($\alpha = 0.7, T = 5$) vs. sin soft labels ($\alpha = 0$). Después de 50 épocas, prueba ambos Students en un dígito ambiguo (como "9"). ¿Cuál produce distribuciones más parecidas al Teacher?

### Ejercicio 6: Factor $T^2$
Entrena con $T = 10$ y observa la magnitud de la soft loss. Si removieras el factor $T^2$ de la fórmula, ¿esperarías gradientes más grandes o más pequeños? ¿Qué efecto tendría en la velocidad de convergencia?

---

## 10. Glosario

| Término | Definición |
|---------|-----------|
| Knowledge Distillation | Transferir conocimiento de un modelo grande (Teacher) a uno pequeño (Student) |
| Teacher | Red neuronal grande y precisa que sirve como fuente de conocimiento |
| Student | Red neuronal pequeña que aprende a imitar al Teacher |
| Soft labels | Distribución de probabilidad suavizada con temperatura $T > 1$ |
| Hard labels | Etiquetas one-hot (la clase verdadera con probabilidad 1) |
| Temperatura ($T$) | Parámetro que suaviza la distribución softmax |
| Logits ($z$) | Salidas de la red antes del softmax |
| Dark knowledge | Información sobre relaciones entre clases, contenida en las soft labels |
| KL Divergencia | Medida de distancia entre dos distribuciones de probabilidad |
| Cross-Entropy | Función de pérdida para clasificación con etiquetas verdaderas |
| Alfa ($\alpha$) | Peso que balancea soft loss vs. hard loss |
| Softmax | Función que convierte logits en probabilidades |
| Época | Una pasada completa por el dataset de entrenamiento |
| Accuracy | Fracción de ejemplos clasificados correctamente |
| Feature distillation | Destilación usando representaciones internas, no solo predicciones |
| Self-distillation | Un modelo se destila a sí mismo para mejorar generalización |
| Compresión | Reducir el tamaño de un modelo manteniendo rendimiento |
| On-device | Ejecución del modelo en dispositivos con recursos limitados |

---

## 11. Referencias

1. Hinton, G., Vinyals, O. & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. NIPS Workshop.
2. Gou, J., et al. (2021). *Knowledge Distillation: A Survey*. IJCV.
3. Beyer, L., et al. (2022). *Knowledge Distillation: A Good Teacher is Patient and Consistent*. CVPR.
4. Touvron, H., et al. (2021). *Training data-efficient image transformers & distillation through attention*. ICML.
5. Tang, R., et al. (2019). *Distilling Task-Specific Knowledge from BERT into Simple Neural Networks*. arXiv.
