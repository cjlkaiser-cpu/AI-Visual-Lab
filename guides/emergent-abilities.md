# Emergent Abilities — Cuando la Escala lo Cambia Todo

> Simulación interactiva de las habilidades emergentes en modelos de lenguaje. Desliza la escala de parámetros de 10M a 1T y observa cómo tareas que parecían imposibles se desbloquean abruptamente al cruzar un umbral crítico — una transición de fase en la inteligencia artificial.

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

En 2022, investigadores de Google descubrieron algo sorprendente: ciertas habilidades en modelos de lenguaje no mejoran gradualmente al escalar el modelo, sino que **aparecen abruptamente** al cruzar un umbral de tamaño. Un modelo de 1B de parámetros no puede hacer aritmética de 3 dígitos en absoluto, pero un modelo de 100B la resuelve con facilidad.

Este fenómeno — llamado **habilidades emergentes** — se asemeja a las **transiciones de fase** en física: el agua no se congela gradualmente, hay un punto crítico (0°C) donde el cambio es abrupto.

Esta simulación permite explorar cómo el rendimiento en 5 tareas diferentes cambia con la escala del modelo, visualizando las curvas sigmoidales de transición y las leyes de escala que gobiernan la pérdida del modelo.

---

## 2. Conceptos Fundamentales

### 2.1 Leyes de escala (Scaling Laws)

Kaplan et al. (2020) descubrieron que la pérdida de un LLM sigue una ley de potencia:

$$L(N) = \left(\frac{N_c}{N}\right)^\alpha$$

donde $N$ son los parámetros del modelo, $N_c$ es una constante crítica, y $\alpha$ es el exponente de escala. Más parámetros = menos pérdida, de forma predecible.

### 2.2 De pérdida a habilidad

Pero el **rendimiento en tareas** no es una función suave de la pérdida. Una tarea puede requerir un "umbral de capacidad" — un nivel mínimo de entendimiento antes de que el modelo pueda resolver esa tarea en absoluto.

$$\text{Rendimiento}(N) = \frac{1}{1 + \exp(-k \cdot (\log N - \log N_{\text{threshold}}))}$$

Esta sigmoide en escala logarítmica produce la transición de fase observada.

### 2.3 Las 5 tareas

| Tarea | Umbral aprox. | Descripción |
|-------|--------------|-------------|
| Aritmética (3 dígitos) | ~10B | Multiplicar números de 3 dígitos |
| Traducción | ~1B | Traducir entre idiomas |
| Razonamiento lógico | ~50B | Resolver silogismos y puzzles |
| Código (HumanEval) | ~100B | Generar código funcional |
| Razonamiento matemático | ~200B | Resolver problemas de matemáticas |

### 2.4 Modelos de referencia

La simulación muestra curvas para múltiples "familias" de modelos, reflejando que diferentes arquitecturas alcanzan las transiciones en puntos ligeramente diferentes.

---

## 3. La Interfaz

### 3.1 Canvas principal

El canvas muestra un gráfico con:

- **Eje X**: parámetros del modelo (escala logarítmica, de 10M a 1T)
- **Eje Y**: rendimiento en la tarea (0% a 100%)
- **Curvas sigmoidales**: una por tarea, cada una con su color y umbral
- **Línea vertical**: posición actual del slider de escala
- **Puntos de modelo**: marcadores para modelos específicos (GPT-2, GPT-3, etc.)
- **Zona de transición**: sombreado alrededor del umbral

### 3.2 Panel lateral

- **Ecuación**: $L(N) = (N_c/N)^\alpha$ (Chinchilla scaling law)
- **Slider de escala**: parámetros de 10M a 1T (logarítmico)
- **Selector de tarea**: individual o todas simultáneamente
- **Leyenda de modelos**: tags coloreados para cada familia
- **Métricas**: parámetros actuales, tarea, rendimiento, punto de transición

### 3.3 Barra de estado

Muestra la escala actual y el rendimiento en la tarea seleccionada.

---

## 4. Controles Interactivos

### 4.1 Escala del modelo

El slider principal controla los parámetros en escala logarítmica:

$$N = 10^{7 + 5 \cdot p / 100}$$

donde $p \in [0, 100]$ es la posición del slider. Esto cubre desde 10M ($10^7$) hasta ~1T ($10^{12}$).

### 4.2 Selector de tarea

| Opción | Qué muestra |
|--------|------------|
| Todas las tareas | Las 5 curvas simultáneamente |
| Aritmética | Solo la curva de aritmética de 3 dígitos |
| Traducción | Solo la curva de traducción |
| Razonamiento Lógico | Solo razonamiento lógico |
| Código (HumanEval) | Solo generación de código |
| Razonamiento Matemático | Solo razonamiento matemático |

### 4.3 Animación

- **Animar Escala** — recorre el slider de izquierda a derecha automáticamente
- **Reset** — vuelve al inicio

### 4.4 Audio

- **Volumen** (0–100%) y botón de silenciar

---

## 5. Las Matemáticas

### 5.1 Ley de escala de Kaplan

La pérdida del modelo (cross-entropy en el conjunto de test) sigue:

$$L(N) = \left(\frac{N_c}{N}\right)^\alpha + L_\infty$$

donde $L_\infty$ es la pérdida irreducible (entropía del lenguaje). Empíricamente, $\alpha \approx 0.076$ para modelos de lenguaje.

### 5.2 Chinchilla scaling

Hoffmann et al. (2022) demostraron que el entrenamiento óptimo requiere escalar datos y modelo proporcionalmente:

$$N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}$$

donde $C$ es el presupuesto de cómputo y $D$ es el número de tokens de entrenamiento. Esto implica que muchos modelos previos estaban *sub-entrenados*.

### 5.3 Función de transición (sigmoide)

Para cada tarea $i$, el rendimiento sigue una sigmoide en log-escala:

$$\text{Perf}_i(N) = \frac{1}{1 + \exp\left(-k_i \cdot (\log_{10} N - \log_{10} N_i^*)\right)}$$

donde $N_i^*$ es el umbral de transición y $k_i$ controla la pendiente (cuán abrupta es la transición).

### 5.4 Rendimiento aleatorio vs. perfecto

Para $N \ll N_i^*$: $\text{Perf}_i \approx 0$ (rendimiento aleatorio)
Para $N \gg N_i^*$: $\text{Perf}_i \approx 1$ (rendimiento cercano a perfecto)

La transición ocurre en un rango relativamente estrecho de escala — típicamente un orden de magnitud.

### 5.5 Relación entre pérdida y emergencia

La pérdida baja suavemente, pero el rendimiento en tareas salta:

$$\text{Perf}_i = f_i(L(N))$$

donde $f_i$ es una función escalón suavizada. Pequeñas reducciones en $L$ pueden desbloquear habilidades enteras cuando se cruza un umbral interno.

---

## 6. Sonificación

### 6.1 Diseño de audio

La simulación usa sonido para representar la escala y las transiciones:

- **Slider de escala**: nota continua cuya frecuencia sube con los parámetros
- **Cruce de umbral**: acorde mayor que se dispara cuando una tarea pasa del 50% de rendimiento
- **Animación**: barrido de frecuencia ascendente durante la animación

### 6.2 Mapeo

| Evento | Frecuencia | Tipo |
|--------|-----------|------|
| Escala baja (<1B) | 200 Hz | sine (grave) |
| Escala media (1-100B) | 400 Hz | sine |
| Escala alta (>100B) | 800 Hz | sine (agudo) |
| Transición de fase | Acorde C-E-G-C | sine × 4 |

---

## 7. Guía Paso a Paso

### Paso 1: Explorar la escala completa

1. Selecciona "Todas las tareas"
2. Mueve el slider lentamente de izquierda a derecha
3. Observa cómo las curvas permanecen en 0 y luego saltan a ~100%
4. Nota que cada tarea tiene un umbral diferente

### Paso 2: Identificar los umbrales

1. Selecciona "Aritmética"
2. Encuentra el punto exacto donde el rendimiento cruza 50%
3. Anota los parámetros en ese punto
4. Repite para cada tarea individual

### Paso 3: Orden de emergencia

1. Vuelve a "Todas las tareas"
2. Usa la animación automática
3. Registra el orden en que las tareas se "desbloquean"
4. ¿Traducción emerge antes que aritmética? ¿Código antes que matemáticas?

### Paso 4: Comparar modelos

1. Observa las etiquetas de modelo en la leyenda
2. Cada familia tiene curvas ligeramente desplazadas
3. Identifica qué familia alcanza la aritmética con menos parámetros

### Paso 5: La ley de escala de la pérdida

1. Observa la curva de pérdida (si se muestra como referencia)
2. Note que es una línea recta en escala log-log
3. Pero el rendimiento en tareas no lo es — la emergencia es no lineal

---

## 8. Conceptos Avanzados

### 8.1 ¿Son reales las emergencias?

Schaeffer et al. (2023) argumentaron que las habilidades "emergentes" podrían ser un artefacto de las métricas utilizadas. Si se usan métricas continuas en lugar de binarias (correcto/incorrecto), las transiciones pueden parecer más graduales.

### 8.2 Compute-optimal scaling

La visión Chinchilla dice que no basta escalar parámetros — también hay que escalar datos:

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty$$

donde $D$ es el número de tokens y $\alpha \approx \beta \approx 0.34$.

### 8.3 Emergencia como transición de fase

En física estadística, las transiciones de fase ocurren cuando un parámetro de orden cruza un punto crítico. La analogía:

| Física | LLMs |
|--------|------|
| Temperatura | $1/\log N$ (inverso de escala) |
| Magnetización | Rendimiento en tarea |
| Punto crítico | $N^*$ (umbral de emergencia) |

### 8.4 Implicaciones para la seguridad de IA

Si las habilidades emergen impredeciblemente al escalar, podríamos encontrar habilidades peligrosas que aparecen sin advertencia previa en modelos más grandes. Esto motiva la investigación en **evaluación proactiva** de modelos.

---

## 9. Ejercicios

### Ejercicio 1: Tabla de umbrales
Para cada una de las 5 tareas, usa el slider para encontrar el punto de transición (rendimiento = 50%). Construye una tabla con: tarea, umbral $N^*$, y $\log_{10}(N^*)$.

### Ejercicio 2: Ancho de transición
Para la tarea de aritmética, encuentra los puntos donde el rendimiento es 10% y 90%. ¿Cuántos órdenes de magnitud separan estos puntos? Esto mide cuán "abrupta" es la transición.

### Ejercicio 3: Ley de escala
Usando la fórmula $L(N) = (N_c/N)^{0.076}$ con $N_c = 10^{13}$, calcula $L$ para $N = 10^9$, $10^{10}$ y $10^{11}$. ¿Cuánto disminuye $L$ por cada orden de magnitud?

### Ejercicio 4: Chinchilla
Si tienes un presupuesto de cómputo fijo $C$, y $N_{\text{opt}} \propto C^{0.5}$, ¿cuánto más cómputo necesitas para duplicar el número de parámetros óptimos? Si el cómputo cuesta \$1M para 10B, ¿cuánto costaría entrenar un modelo de 40B óptimamente?

### Ejercicio 5: Orden predecible
¿El orden de emergencia de tareas (traducción → aritmética → lógica → código → matemáticas) tiene sentido intuitivamente? Argumenta por qué ciertas tareas requieren más escala que otras.

### Ejercicio 6: Debate sobre emergencia
Lee el resumen del paper de Schaeffer et al. Si las emergencias fueran artefactos de métricas, ¿cómo cambiaría la interpretación de esta simulación? ¿Las transiciones se suavizarían o desaparecerían?

---

## 10. Glosario

| Término | Definición |
|---------|-----------|
| Habilidad emergente | Capacidad que aparece abruptamente al escalar un modelo, ausente en modelos pequeños |
| Transición de fase | Cambio abrupto en una propiedad al cruzar un umbral crítico |
| Scaling law | Relación matemática entre tamaño del modelo y su rendimiento |
| Parámetros (N) | Número de pesos entrenables en el modelo |
| Pérdida (Loss) | Cross-entropy del modelo en datos de test; mide calidad general |
| Chinchilla | Estudio de DeepMind sobre entrenamiento óptimo: datos ∝ parámetros |
| Compute-optimal | Modelo entrenado con la proporción óptima de datos para su tamaño |
| Sigmoide | Función en forma de S que modela la transición del rendimiento |
| Umbral ($N^*$) | Número de parámetros donde el rendimiento en una tarea cruza 50% |
| Pendiente ($k$) | Cuán abrupta es la transición de fase |
| Log-escala | Escala logarítmica donde cada marca representa un orden de magnitud |
| Benchmark | Conjunto de tareas estandarizado para evaluar modelos |
| HumanEval | Benchmark de generación de código de OpenAI |
| Cross-entropy | Función de pérdida estándar para modelos de lenguaje |
| Familia de modelos | Serie de modelos con la misma arquitectura pero diferente tamaño |
| Exponente de escala ($\alpha$) | Pendiente en log-log de la ley de escala |
| Entrenamiento sub-óptimo | Modelo con menos datos de los que necesita para su tamaño |
| Evaluación proactiva | Testear modelos por habilidades peligrosas antes de desplegarlos |

---

## 11. Referencias

1. Wei, J., et al. (2022). *Emergent Abilities of Large Language Models*. TMLR.
2. Kaplan, J., et al. (2020). *Scaling Laws for Neural Language Models*. arXiv:2001.08361.
3. Hoffmann, J., et al. (2022). *Training Compute-Optimal Large Language Models (Chinchilla)*. arXiv:2203.15556.
4. Schaeffer, R., et al. (2023). *Are Emergent Abilities of Large Language Models a Mirage?*. NeurIPS.
5. Ganguli, D., et al. (2022). *Predictability and Surprise in Large Generative Models*. FAccT.
