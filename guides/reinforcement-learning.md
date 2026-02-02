# Reinforcement Learning — Grid World: Aprender de la Experiencia

> Simulación interactiva de aprendizaje por refuerzo en un mundo de cuadrícula 8×8. Un agente aprende a navegar entre muros, trampas y recompensas para alcanzar la meta, usando Q-Learning o SARSA. Observa cómo los Q-values evolucionan, cómo la política óptima emerge episodio tras episodio, y cómo el balance exploración-explotación determina el éxito.

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

El **aprendizaje por refuerzo** (RL) es el paradigma donde un agente aprende a tomar decisiones interactuando con un entorno. No hay un dataset de entrenamiento con respuestas correctas — solo señales de **recompensa** que indican cuán buena fue cada acción.

Este es el paradigma detrás de AlphaGo (DeepMind, 2016), la robótica autónoma, y el fine-tuning de LLMs con RLHF (Reinforcement Learning from Human Feedback).

Esta simulación presenta un **Grid World** de 8×8: un entorno simple pero completo donde un agente debe aprender a navegar desde la esquina superior izquierda hasta la meta en la esquina inferior derecha, evitando trampas y recolectando recompensas. El agente no conoce el mapa — debe descubrirlo por exploración.

---

## 2. Conceptos Fundamentales

### 2.1 El framework MDP

Un **Proceso de Decisión de Markov** (MDP) se define por la tupla $(S, A, P, R, \gamma)$:

- $S$: conjunto de estados (celdas del grid)
- $A$: conjunto de acciones ($\uparrow, \downarrow, \leftarrow, \rightarrow$)
- $P(s' | s, a)$: probabilidad de transición
- $R(s, a)$: recompensa por tomar acción $a$ en estado $s$
- $\gamma \in [0, 1]$: factor de descuento temporal

### 2.2 Q-values

El **Q-value** $Q(s, a)$ estima la recompensa total esperada de tomar acción $a$ en estado $s$ y seguir la política óptima después:

$$Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' | s, a) \max_{a'} Q^*(s', a')$$

### 2.3 Política

La **política** $\pi(s)$ determina qué acción tomar en cada estado. La política óptima es:

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

En la simulación, las flechas en cada celda muestran la dirección de la acción con mayor Q-value.

### 2.4 El grid world

| Celda | Símbolo | Recompensa |
|-------|---------|-----------|
| Vacía | - | $-0.01$ (pequeño costo por paso) |
| Muro | Bloque gris | No transitable |
| Trampa | Rojo | $-1.0$ (penalización fuerte) |
| Recompensa | Amarillo | $+0.5$ |
| Meta | Verde | $+1.0$ (episodio termina) |

---

## 3. La Interfaz

### 3.1 Canvas principal

El canvas muestra el grid 8×8 con:

- **Celdas coloreadas**: vacías (oscuro), muros (gris), trampas (rojo), recompensas (amarillo), meta (verde)
- **Flechas**: en cada celda, la dirección de la mejor acción según los Q-values actuales
- **Intensidad de color**: refleja el Q-value máximo — celdas con Q-values altos brillan más
- **Agente**: círculo azul que se mueve durante la ejecución
- **Camino**: rastro del agente durante la ejecución (trazado en azul)

### 3.2 Panel lateral

- **Ecuación**: $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \cdot \max_{a'} Q(s',a') - Q(s,a)]$
- **Botones de entrenamiento**: Train 1 Ep, Train 100, Train 1000
- **Correr Agente**: ejecuta la política actual visualmente
- **Sliders**: epsilon, gamma, alpha
- **Selector de algoritmo**: Q-Learning / SARSA
- **Gráfico de recompensa**: historial de reward por episodio

### 3.3 Barra de estado

Muestra el episodio actual y la recompensa acumulada.

---

## 4. Controles Interactivos

### 4.1 Entrenamiento

| Botón | Acción |
|-------|--------|
| Train 1 Ep | Ejecuta 1 episodio de entrenamiento (invisible, actualiza Q) |
| Train 100 | Ejecuta 100 episodios rápidamente |
| Train 1000 | Ejecuta 1000 episodios |
| Correr Agente | Visualiza al agente siguiendo la política actual |
| Reset Q | Reinicia todos los Q-values a 0 |
| Reset Grid | Restaura el mapa original |

### 4.2 Hiperparámetros

| Parámetro | Rango | Default | Efecto |
|-----------|-------|---------|--------|
| Epsilon ($\epsilon$) | 0 – 1.0 | 0.30 | Probabilidad de exploración aleatoria |
| Gamma ($\gamma$) | 0 – 1.0 | 0.95 | Descuento temporal: cuánto valora el futuro |
| Alpha ($\alpha$) | 0.01 – 1.0 | 0.10 | Learning rate: velocidad de actualización de Q |

### 4.3 Algoritmo

- **Q-Learning** (off-policy): actualiza con el máximo Q del siguiente estado
- **SARSA** (on-policy): actualiza con la acción realmente tomada

### 4.4 Estadísticas

- Episodios totales entrenados
- Reward promedio (últimos 200 episodios)
- Número de éxitos (llegó a la meta)
- Gráfico de reward acumulado por episodio

---

## 5. Las Matemáticas

### 5.1 Actualización Q-Learning

Q-Learning es un algoritmo **off-policy**: la regla de actualización usa la mejor acción posible en el siguiente estado, independientemente de qué acción se tomó realmente:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

El término entre corchetes es el **TD error** (temporal difference error):

$$\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)$$

### 5.2 Actualización SARSA

SARSA es **on-policy**: usa la acción que realmente se tomó en el siguiente estado:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$

donde $a'$ es la acción elegida por la política actual (incluyendo exploración).

### 5.3 Política $\epsilon$-greedy

Para balancear exploración y explotación:

$$a = \begin{cases} \text{acción aleatoria} & \text{con probabilidad } \epsilon \\ \arg\max_a Q(s, a) & \text{con probabilidad } 1 - \epsilon \end{cases}$$

### 5.4 Factor de descuento

El valor de un estado es la suma descontada de recompensas futuras:

$$V(s) = \sum_{t=0}^{\infty} \gamma^t r_{t+1}$$

Con $\gamma = 0.95$, una recompensa de $+1$ en 10 pasos vale $0.95^{10} \approx 0.60$ hoy. Con $\gamma = 0.5$, vale solo $0.5^{10} \approx 0.001$.

### 5.5 Convergencia

Q-Learning converge a $Q^*$ bajo condiciones:
1. Cada par $(s, a)$ se visita infinitas veces
2. El learning rate decrece: $\sum \alpha_t = \infty$, $\sum \alpha_t^2 < \infty$
3. Las recompensas están acotadas

En la práctica, un $\alpha$ fijo pequeño funciona bien para entornos simples.

---

## 6. Sonificación

### 6.1 Diseño de audio

El audio representa eventos del agente:

- **Paso del agente**: click cuya frecuencia refleja el Q-value de la celda visitada
- **Trampa**: tono grave descendente (penalización)
- **Recompensa**: nota aguda ascendente
- **Meta alcanzada**: acorde de victoria
- **Episodio fallido**: tono bajo

### 6.2 Mapeo

| Evento | Frecuencia | Duración | Tipo |
|--------|-----------|----------|------|
| Paso normal | 300 + Q × 400 Hz | 0.08s | triangle |
| Trampa | 200 → 100 Hz | 0.3s | sawtooth |
| Recompensa | 500 → 700 Hz | 0.2s | sine |
| Meta | Acorde C-E-G | 0.5s | sine |
| Exploración (ε) | 250 Hz (detuned) | 0.05s | square |

---

## 7. Guía Paso a Paso

### Paso 1: Explorar el entorno

1. Observa el grid: identifica muros (gris), trampas (rojo), recompensas (amarillo), meta (verde)
2. Nota que el agente empieza en (0,0) y la meta está en (7,7)
3. Los Q-values están todos en 0 — el agente no sabe nada

### Paso 2: Primer entrenamiento

1. Pulsa **Train 1 Ep** con $\epsilon = 0.3$
2. Observa que las flechas apenas cambian — un solo episodio no es suficiente
3. Revisa la recompensa del episodio en la gráfica

### Paso 3: Entrenamiento significativo

1. Pulsa **Train 100**
2. Las flechas empiezan a apuntar hacia la meta
3. Las celdas cercanas a la meta tienen Q-values más altos (más brillantes)
4. El reward promedio sube en el gráfico

### Paso 4: Convergencia

1. Pulsa **Train 1000**
2. Las flechas forman caminos claros evitando trampas
3. Pulsa **Correr Agente** para ver la política en acción
4. El agente debería llegar a la meta eficientemente

### Paso 5: Q-Learning vs. SARSA

1. Resetea Q, cambia a **SARSA**, entrena 1000 episodios
2. Compara las políticas: SARSA tiende a ser más conservador (evita pasar cerca de trampas)
3. Q-Learning puede aprender caminos más cortos que pasan cerca de trampas

### Paso 6: Experimenta con parámetros

1. Cambia $\epsilon = 0$ (sin exploración) y entrena — ¿converge?
2. Cambia $\gamma = 0.1$ (muy miope) — ¿qué política emerge?
3. Cambia $\alpha = 0.9$ (learning rate alto) — ¿es estable?

---

## 8. Conceptos Avanzados

### 8.1 On-policy vs. Off-policy

**Q-Learning** (off-policy) aprende la política óptima independientemente de cómo explora. **SARSA** (on-policy) aprende el valor de la política que realmente sigue, incluyendo la exploración.

Consecuencia práctica: con $\epsilon > 0$, SARSA aprende a evitar caminos donde una acción exploratoria podría ser catastrófica (e.g., celda adyacente a una trampa).

### 8.2 Dilema exploración-explotación

Con $\epsilon = 0$: el agente explota lo que sabe, pero puede quedar atrapado en óptimos locales (nunca descubre caminos mejores). Con $\epsilon = 1$: el agente explora todo pero nunca aprovecha lo aprendido.

Estrategia común: $\epsilon$-decay, donde $\epsilon$ decrece con el tiempo.

### 8.3 Deep Q-Networks (DQN)

Para entornos con espacios de estado grandes (imágenes), se reemplaza la tabla Q con una red neuronal:

$$Q(s, a; \theta) \approx Q^*(s, a)$$

DQN (Mnih et al., 2015) combinó Q-Learning con redes profundas para jugar Atari directamente desde píxeles.

### 8.4 RLHF y LLMs

El mismo principio de RL se usa para alinear LLMs con preferencias humanas:

$$\text{reward}(s, a) = \text{modelo\_de\_reward}(\text{prompt}, \text{respuesta})$$

PPO (Proximal Policy Optimization) optimiza la política del LLM para maximizar esta recompensa.

---

## 9. Ejercicios

### Ejercicio 1: Entrenamiento mínimo
¿Cuántos episodios necesita Q-Learning para que el agente alcance la meta consistentemente (>80% de éxito en los últimos 50 episodios)? Prueba con $\epsilon = 0.3$, $\gamma = 0.95$, $\alpha = 0.1$.

### Ejercicio 2: Impacto de gamma
Entrena 1000 episodios con $\gamma = 0.1$, $\gamma = 0.5$, $\gamma = 0.99$. ¿Qué política emerge con cada valor? ¿Un gamma bajo hace que el agente tome atajos peligrosos o rodeos seguros?

### Ejercicio 3: Q-Learning vs. SARSA cuantitativo
Entrena 1000 episodios con cada algoritmo (mismos hiperparámetros). Compara: reward promedio final, tasa de éxito, y longitud promedio del camino. ¿Cuál es "mejor"?

### Ejercicio 4: El papel de alpha
Con $\alpha = 0.01$, $\alpha = 0.1$ y $\alpha = 0.5$, entrena 500 episodios cada uno. ¿Cuál converge más rápido? ¿Cuál es más estable? Grafica el reward promedio para cada caso.

### Ejercicio 5: Exploración pura
Fija $\epsilon = 1.0$ (totalmente aleatorio) y corre el agente 100 veces. ¿Qué fracción de las veces alcanza la meta? Calcula la probabilidad teórica de llegar en un camino aleatorio de 200 pasos.

### Ejercicio 6: Diseño de recompensas
¿Qué pasaría si cambias la recompensa de la trampa de $-1$ a $-0.1$? ¿Y si el costo por paso fuera $-0.1$ en vez de $-0.01$? Razona sobre cómo el diseño de recompensas afecta la política resultante.

---

## 10. Glosario

| Término | Definición |
|---------|-----------|
| Reinforcement Learning | Paradigma donde un agente aprende mediante interacción y recompensas |
| Agente | Entidad que toma decisiones (elige acciones) |
| Entorno | Sistema con el que el agente interactúa |
| Estado ($s$) | Configuración actual del entorno |
| Acción ($a$) | Decisión del agente ($\uparrow, \downarrow, \leftarrow, \rightarrow$) |
| Recompensa ($r$) | Señal numérica de retroalimentación |
| Q-value | Estimación de recompensa futura para un par estado-acción |
| Política ($\pi$) | Regla que mapea estados a acciones |
| Epsilon ($\epsilon$) | Probabilidad de tomar una acción aleatoria (exploración) |
| Gamma ($\gamma$) | Factor de descuento: cuánto se valora la recompensa futura |
| Alpha ($\alpha$) | Learning rate: velocidad de actualización de Q |
| TD error | Diferencia entre recompensa esperada y obtenida |
| Q-Learning | Algoritmo off-policy que actualiza con $\max Q(s', a')$ |
| SARSA | Algoritmo on-policy que actualiza con $Q(s', a')$ real |
| Episodio | Secuencia completa desde el inicio hasta meta o fallo |
| $\epsilon$-greedy | Política que explora con probabilidad $\epsilon$ |
| MDP | Proceso de Decisión de Markov: formalismo del RL |
| Convergencia | Cuando los Q-values se estabilizan en sus valores óptimos |

---

## 11. Referencias

1. Sutton, R. & Barto, A. (2018). *Reinforcement Learning: An Introduction*. 2nd ed. MIT Press.
2. Watkins, C. & Dayan, P. (1992). *Q-Learning*. Machine Learning, 8(3-4).
3. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
4. Ouyang, L., et al. (2022). *Training language models to follow instructions with human feedback*. NeurIPS.
5. Silver, D., et al. (2016). *Mastering the game of Go with deep neural networks and tree search*. Nature.
