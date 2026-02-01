# AI Visual Lab - Documentación Técnica y Prompt de Automatización

## Descripción

El **AI Visual Lab** es el primer laboratorio completo de IA explicable visual y sonoramente. **20 simulaciones** que van desde el perceptrón hasta arquitecturas especulativas, todas interactivas y con sonificación. Cada simulación convierte conceptos abstractos de inteligencia artificial en experiencias sensoriales: se ve, se escucha y se manipula el aprendizaje automático.

**Filosofía:** Explicabilidad radical. No basta con ver la red — hay que escuchar el gradiente desvanecerse, sentir la fricción del learning rate, oír la disonancia del error.

## Stack Tecnológico

### Core
- **HTML5** - Archivos autocontenidos, sin bundler
- **CSS3** - Tema oscuro, variables CSS custom properties
- **JavaScript ES6+** - Vanilla, sin frameworks ni TensorFlow.js
- **Google Fonts: Inter + JetBrains Mono** - Tipografía

### Rendering
- **Canvas 2D** - Renderizado principal (mayoría de simulaciones)
- **Three.js** (CDN) - Visualizaciones 3D (Gradient Landscape, Embeddings, Transformer)

### Audio
- **Web Audio API** - Síntesis nativa (lazy init, iOS/Safari compatible)
- **ADSR Envelopes** - Modelado de notas
- **Síntesis FM/Aditiva** - Timbres complejos
- **Filtros** - Lowpass, highpass, bandpass
- **Gestión de polifonía** - Límite 64 voces

### AI/ML
- **Redes neuronales en JS puro** - Forward/backward pass implementados from scratch
- **Datos pre-entrenados** - Embedidos como arrays JSON en el HTML cuando necesario
- **Modelos toy** - Redes pequeñas que entrenan en tiempo real en el browser

## Estructura

```
AI/
└── AI Visual Lab/
    ├── index.html                          # Landing con 20 cards animadas
    ├── CLAUDE.md                           # Este archivo
    ├── guides/                             # Guías pedagógicas
    │
    │── Módulo 1: Fundamentos
    ├── perceptron-viviente.html            # 01 - Perceptrón con sonificación
    ├── gradient-descent-landscape.html     # 02 - Superficie 3D de pérdida
    ├── backpropagation-flow.html           # 03 - Flujo de gradientes
    ├── activation-function-orchestra.html  # 04 - 8 funciones en paralelo
    ├── weight-initialization-galaxy.html   # 05 - Distribución de pesos
    │
    │── Módulo 2: Arquitecturas Clásicas
    ├── autoencoder-cathedral.html          # 06 - Compresión como arquitectura
    ├── lstm-memory-cells.html              # 07 - Neuronas con memoria
    ├── attention-heatmap.html              # 08 - Donde mira la red
    ├── cnn-feature-detectives.html         # 09 - Corteza visual artificial
    ├── gan-duel.html                       # 10 - Generador vs Discriminador
    │
    │── Módulo 3: Transformers y LLMs
    ├── transformer-anatomy.html            # 11 - El corazón de GPT
    ├── token-embeddings-constellation.html # 12 - Espacio semántico
    ├── self-attention-mechanism.html       # 13 - QKV en detalle
    ├── chain-of-thought-unrolled.html      # 14 - Razonamiento paso a paso
    ├── emergent-abilities.html             # 15 - Transiciones de fase
    │
    │── Módulo 4: Fronteras
    ├── diffusion-models.html              # 16 - De ruido a arte
    ├── reinforcement-learning.html        # 17 - Grid World
    ├── knowledge-distillation.html        # 18 - Maestro y alumno
    ├── mixture-of-experts.html            # 19 - El comité
    └── superposition-black-box.html       # 20 - Dentro de la caja negra
```

## Paleta de Colores

### Color principal de disciplina
- **AI Visual Lab:** `#ef4444` (Rojo)
- Variable CSS: `--accent: #ef4444`

### Colores por módulo (subcategorías en index.html)
| Módulo | Color | Hex | Uso |
|--------|-------|-----|-----|
| Fundamentos | Rojo | `#ef4444` | Perceptrón, gradientes, pesos |
| Arquitecturas | Ámbar | `#f59e0b` | Autoencoder, LSTM, CNN, GAN |
| Transformers | Violeta | `#8b5cf6` | Attention, embeddings, CoT |
| Fronteras | Cyan | `#06b6d4` | Diffusion, RL, MoE, superposición |

## Convenciones

### Nomenclatura
- **Idioma UI**: Español (títulos, labels, breadcrumbs, ecuaciones)
- **Idioma código**: Inglés (variables, funciones, comentarios técnicos)
- **Archivos**: kebab-case (`perceptron-viviente.html`)
- **Variables**: camelCase (`learningRate`, `audioContext`)
- **Constantes**: UPPER_CASE o CONFIG (`MAX_VOICES`, `CONFIG.volume`)
- **Clases**: PascalCase (`Perceptron`, `NeuralNetwork`, `Particle`)

### Breadcrumbs
```
EigenLab / AI Visual Lab / [Nombre Simulación]
```

### Ecuaciones
- Font: Times New Roman / serif, italic
- Background: `rgba(accent, 0.1)`
- Border-left: 3px solid accent

## Patrón de Simulación

### Estructura HTML
```html
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>[Nombre] - AI Visual Lab - EigenLab</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --bg-primary: #030712;
            --bg-secondary: #0f172a;
            --bg-panel: #1e293b;
            --accent: #ef4444;
            --accent-dim: rgba(239, 68, 68, 0.15);
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --border: #1e293b;
        }
        body {
            font-family: 'Inter', system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            overflow: hidden;
            height: 100vh;
        }
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.75rem 1.5rem;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            z-index: 10;
        }
        .header h1 { font-size: 1rem; font-weight: 600; }
        .breadcrumb { font-size: 0.8rem; color: var(--text-secondary); }
        .breadcrumb a { color: var(--accent); text-decoration: none; }
        .main {
            display: flex;
            height: calc(100vh - 49px);
        }
        .canvas-container {
            flex: 1;
            position: relative;
            overflow: hidden;
        }
        #canvas {
            width: 100%;
            height: 100%;
            display: block;
        }
        .controls {
            width: 340px;
            background: var(--bg-secondary);
            border-left: 1px solid var(--border);
            padding: 1.25rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .control-section {
            background: var(--bg-panel);
            border-radius: 12px;
            padding: 1rem;
        }
        .control-section h3 {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 0.75rem;
        }
        .equation-box {
            background: var(--accent-dim);
            border-left: 3px solid var(--accent);
            padding: 0.75rem 1rem;
            border-radius: 0 8px 8px 0;
            font-family: 'Times New Roman', serif;
            font-style: italic;
            font-size: 1.1rem;
            color: var(--text-primary);
            line-height: 1.6;
        }
        .slider-group { margin-bottom: 0.75rem; }
        .slider-group label {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            margin-bottom: 0.25rem;
            color: var(--text-secondary);
        }
        .slider-group label span {
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent);
        }
        input[type="range"] {
            width: 100%;
            height: 4px;
            -webkit-appearance: none;
            background: var(--bg-primary);
            border-radius: 2px;
            outline: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            background: var(--accent);
            border-radius: 50%;
            cursor: pointer;
        }
        .btn {
            padding: 0.5rem 1rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--bg-panel);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn:hover { border-color: var(--accent); color: var(--accent); }
        .btn-accent {
            background: var(--accent);
            color: white;
            border-color: var(--accent);
        }
        .btn-accent:hover { opacity: 0.9; color: white; }
        .btn-group { display: flex; gap: 0.5rem; flex-wrap: wrap; }
        .stat {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            padding: 0.25rem 0;
        }
        .stat-value {
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent);
        }
        .mono { font-family: 'JetBrains Mono', monospace; }
        select {
            width: 100%;
            padding: 0.5rem;
            background: var(--bg-primary);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-size: 0.8rem;
        }
    </style>
</head>
<body>
    <header class="header">
        <div>
            <h1>[Nombre de la Simulación]</h1>
            <nav class="breadcrumb">
                <a href="../../_portal/index.html">EigenLab</a> /
                <a href="index.html">AI Visual Lab</a> /
                [Nombre]
            </nav>
        </div>
        <div class="header-stats mono" style="font-size: 0.75rem; color: var(--text-secondary);">
            <!-- Live stats -->
        </div>
    </header>
    <main class="main">
        <div class="canvas-container">
            <canvas id="canvas"></canvas>
        </div>
        <aside class="controls">
            <!-- Equation box -->
            <div class="equation-box">[Ecuación principal]</div>

            <!-- Control sections -->
            <div class="control-section">
                <h3>Parámetros</h3>
                <!-- Sliders, buttons -->
            </div>

            <div class="control-section">
                <h3>Audio</h3>
                <!-- Volume, waveform, mute -->
            </div>

            <div class="control-section">
                <h3>Valores</h3>
                <!-- Computed stats -->
            </div>
        </aside>
    </main>
    <script>
        // === CONFIG ===
        const CONFIG = {
            // Simulation params
            running: true,
            // Audio params
            volume: 0.3,
            audioEnabled: true,
            waveform: 'sine'
        };

        // === CANVAS SETUP ===
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let W, H, dpr;

        function resizeCanvas() {
            dpr = window.devicePixelRatio || 1;
            const rect = canvas.parentElement.getBoundingClientRect();
            W = rect.width;
            H = rect.height;
            canvas.width = W * dpr;
            canvas.height = H * dpr;
            canvas.style.width = W + 'px';
            canvas.style.height = H + 'px';
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        // === AUDIO (Lazy Init) ===
        let audioContext = null;
        let masterGain = null;

        function initAudio() {
            if (audioContext) return;
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            masterGain = audioContext.createGain();
            masterGain.gain.value = CONFIG.volume;
            masterGain.connect(audioContext.destination);
        }

        function playNote(frequency, duration, volume = 0.3) {
            if (!audioContext || !CONFIG.audioEnabled) return;
            const now = audioContext.currentTime;
            const osc = audioContext.createOscillator();
            osc.type = CONFIG.waveform;
            osc.frequency.value = frequency;
            const env = audioContext.createGain();
            env.gain.setValueAtTime(0, now);
            env.gain.linearRampToValueAtTime(volume, now + 0.01);
            env.gain.exponentialRampToValueAtTime(0.001, now + duration);
            osc.connect(env);
            env.connect(masterGain);
            osc.start(now);
            osc.stop(now + duration + 0.05);
        }

        // === STATE ===
        let state = {};

        function init() {
            // Initialize simulation state
        }

        // === PHYSICS / ML ===
        function update(dt) {
            if (!CONFIG.running) return;
            // Simulation logic
        }

        // === RENDERING ===
        function draw() {
            ctx.clearRect(0, 0, W, H);
            // Draw simulation
        }

        // === LOOP ===
        let lastTime = 0;
        function loop(time) {
            const dt = Math.min((time - lastTime) / 1000, 0.1);
            lastTime = time;
            update(dt);
            draw();
            requestAnimationFrame(loop);
        }

        // === CONTROLS ===
        // Event listeners for sliders, buttons, etc.

        // === INIT ===
        init();
        requestAnimationFrame(loop);
    </script>
</body>
</html>
```

## Patrón de Audio para AI Lab

### Sonificación del Error
```javascript
// Error alto = disonancia, Error bajo = consonancia
function sonifyError(error) {
    if (!audioContext || !CONFIG.audioEnabled) return;
    const baseFreq = 440;
    // Tritono (disonancia máxima) cuando error = 1
    // Octava (consonancia) cuando error = 0
    const interval = error * 6 + (1 - error) * 12; // semitones
    const freq2 = baseFreq * Math.pow(2, interval / 12);
    playNote(baseFreq, 0.2, 0.15);
    playNote(freq2, 0.2, 0.15);
}
```

### Sonificación de Gradientes
```javascript
// Gradiente grande = nota fuerte, gradiente pequeño = nota suave
// Vanishing gradient = fade to silence
function sonifyGradient(gradientMagnitude, layerIndex, totalLayers) {
    if (!audioContext || !CONFIG.audioEnabled) return;
    const baseNote = 60 + (totalLayers - layerIndex) * 5; // MIDI
    const freq = 440 * Math.pow(2, (baseNote - 69) / 12);
    const vol = Math.min(0.4, gradientMagnitude * 2);
    if (vol > 0.005) {
        playNote(freq, 0.15, vol);
    }
}
```

### Sonificación de Convergencia
```javascript
// Acorde que se resuelve progresivamente
function sonifyConvergence(progress) { // 0 to 1
    if (!audioContext || !CONFIG.audioEnabled) return;
    const root = 261.63; // C4
    if (progress < 0.3) {
        // Diminished chord (tension)
        playNote(root, 0.3, 0.1);
        playNote(root * Math.pow(2, 3/12), 0.3, 0.1);
        playNote(root * Math.pow(2, 6/12), 0.3, 0.1);
    } else if (progress < 0.7) {
        // Dominant 7th (moving toward resolution)
        playNote(root, 0.3, 0.1);
        playNote(root * Math.pow(2, 4/12), 0.3, 0.1);
        playNote(root * Math.pow(2, 7/12), 0.3, 0.1);
    } else {
        // Major chord (resolved)
        playNote(root, 0.4, 0.15);
        playNote(root * Math.pow(2, 4/12), 0.4, 0.12);
        playNote(root * Math.pow(2, 7/12), 0.4, 0.12);
    }
}
```

## Especificación de las 20 Simulaciones

### MÓDULO 1: FUNDAMENTOS (Sims 1-5)

#### 1. Perceptrón Viviente (`perceptron-viviente.html`)

**Concepto:** El nacimiento del aprendizaje. Un perceptrón que clasifica puntos en 2D con sonificación del proceso de aprendizaje.

**Modelo:** Perceptrón real — `y = sign(w1*x1 + w2*x2 + b)`, entrenamiento con regla de aprendizaje.

**Visualización:**
- Plano 2D con puntos rojos/azules (clases 0/1)
- Línea de decisión que pivotea y se traslada en tiempo real
- Pesos w1, w2, b como sliders que "empujan" la línea
- Superficie de error como fondo (gradiente de color = error local)
- Punto misclasificado parpadea en dorado

**Interacción:**
- Arrastrar puntos para modificar dataset
- Click para añadir puntos (toggle clase)
- Modo paso a paso: un ejemplo a la vez
- Modo automático: entrena continuamente
- Presets: lineal, diagonal, circular (no separable)

**Sonificación:**
- Error alto → intervalo de tritono (disonancia)
- Error bajo → intervalo de octava (consonancia)
- Cada actualización de peso → click percusivo
- Convergencia completa → acorde mayor resuelto
- Misclasificación → nota grave de advertencia

**Ecuación:** `y = sign(w₁x₁ + w₂x₂ + b)`

**Complejidad:** ~600-800 líneas

---

#### 2. Gradient Descent Landscape (`gradient-descent-landscape.html`)

**Concepto:** Topografía del aprendizaje. Ver cómo los optimizadores navegan superficies de pérdida.

**Modelo:** Funciones de pérdida reales renderizadas como superficies 3D. Partículas que descienden con diferentes algoritmos.

**Visualización (Three.js):**
- Superficie 3D de función de pérdida con color = altura
- Partícula (esfera) que desciende dejando trail
- Múltiples partículas = múltiples inicializaciones
- Vectores de gradiente como flechas en la superficie
- Vista alternativa: contour map 2D (isolíneas)

**Funciones de pérdida:**
- Quadratic bowl (caso fácil)
- Rosenbrock (banana valley)
- Rastrigin (muchos mínimos locales)
- Saddle point (punto de silla)
- Beale (asimétrica)

**Optimizadores:**
- SGD básico
- SGD + Momentum
- Adam
- RMSProp

**Interacción:**
- Slider learning rate (0.0001 - 1.0)
- Slider momentum (0 - 0.99)
- Click en superficie para colocar nueva partícula
- Toggle entre vista 3D y contour 2D
- Selector de función y optimizador

**Sonificación:**
- Altura de pérdida → pitch (más alto = más agudo = más tenso)
- Velocidad de descenso → tempo
- Oscilaciones → vibrato (inestabilidad audible)
- Mínimo local atrapado → nota sostenida monótona
- Mínimo global alcanzado → resolución armónica

**Ecuación:** `θ_{t+1} = θ_t - η∇L(θ_t)`

**Complejidad:** ~800-1000 líneas

---

#### 3. Backpropagation Flow (`backpropagation-flow.html`)

**Concepto:** El río del gradiente. Visualizar cómo fluyen señales y gradientes a través de una red neuronal.

**Modelo:** Red neuronal real de 4 capas (2-4-4-1) entrenándose en XOR o función no lineal. Forward y backward pass calculados matemáticamente.

**Visualización:**
- Red como grafo: nodos = neuronas, aristas = pesos
- Onda de color forward: input → output (azul a verde)
- Onda de color backward: error → gradientes (rojo a naranja)
- Grosor de aristas = |peso|
- Color de aristas = signo (azul positivo, rojo negativo)
- Nodos brillan proporcionalmente a su activación
- Gradientes como partículas que fluyen por las aristas

**Interacción:**
- Click en neurona: ver su responsabilidad en el error
- Congelar capa: desactivar entrenamiento de una capa
- Inyectar ruido en gradientes
- Cambiar función de activación por capa (ReLU, Sigmoid, Tanh)
- Modo paso a paso: un forward+backward a la vez

**Sonificación:**
- Forward pass → arpeggio ascendente (capa a capa)
- Backward pass → arpeggio descendente
- Gradientes grandes → notas fuertes (ff)
- Vanishing gradient → fade to silence (pp→ppp→silence)
- Exploding gradient → crescendo + clipping

**Ecuación:** `∂L/∂w_ij = ∂L/∂a_j · ∂a_j/∂z_j · ∂z_j/∂w_ij`

**Complejidad:** ~900-1100 líneas

---

#### 4. Activation Function Orchestra (`activation-function-orchestra.html`)

**Concepto:** La personalidad de las neuronas. Cómo diferentes funciones de activación transforman señales.

**Modelo:** 8 funciones de activación procesando la misma señal de entrada en paralelo. Implementación matemática exacta.

**Funciones:**
1. Sigmoid: σ(x) = 1/(1+e^(-x))
2. Tanh: tanh(x)
3. ReLU: max(0, x)
4. Leaky ReLU: max(0.01x, x)
5. ELU: x if x>0, α(e^x-1) if x≤0
6. Swish: x·σ(x)
7. GELU: x·Φ(x)
8. Mish: x·tanh(softplus(x))

**Visualización:**
- 8 canales tipo osciloscopio en paralelo (4x2 grid)
- Señal input (onda configurable) entrando por la izquierda
- Output de cada función como traza de osciloscopio
- Derivada superpuesta en color más tenue
- Zona de "saturación" marcada en rojo semitransparente
- "Dead zone" de ReLU marcada en gris

**Interacción:**
- Slider de frecuencia del input (0.1 - 10 Hz)
- Slider de amplitud (0.1 - 10)
- Tipo de input: seno, cuadrada, triángulo, rampa, ruido
- Toggle derivada on/off
- Modo "explosión": input enorme para ver comportamiento límite

**Sonificación:**
- Cada función = timbre distinto (osc type varía)
- Sigmoid/Tanh = sine (suave)
- ReLU = sawtooth (áspero, rectificado)
- GELU/Mish = triangle (intermedio)
- Saturación = clipping audible (distorsión)
- Derivada = modula filtro (cutoff frequency)

**Ecuación panel:** Cada función muestra su ecuación al lado

**Complejidad:** ~700-900 líneas

---

#### 5. Weight Initialization Galaxy (`weight-initialization-galaxy.html`)

**Concepto:** El big bang de los pesos. Cómo arrancar una red neuronal correctamente.

**Modelo:** Red profunda de 10 capas (64 neuronas cada una). Forward pass de señal aleatoria mostrando propagación de varianza.

**Estrategias:**
1. Xavier/Glorot: σ² = 2/(n_in + n_out)
2. He/Kaiming: σ² = 2/n_in
3. LeCun: σ² = 1/n_in
4. Normal(0, 1): sin ajuste
5. Normal(0, 0.01): muy pequeño
6. Zeros: todo cero

**Visualización:**
- Histograma 3D de activaciones por capa (10 histogramas apilados)
- Eje X = valor de activación
- Eje Y = capa (1-10)
- Eje Z = frecuencia
- Color del histograma: verde = varianza estable, rojo = explosión, azul = colapso
- Varianza numérica de cada capa como línea superpuesta

**Interacción:**
- Selector de estrategia de inicialización
- Selector de función de activación (ReLU, Sigmoid, Tanh)
- Botón "Propagar": envía batch de datos aleatorios y muestra propagación
- Slider de profundidad (2-20 capas)
- Slider de ancho (8-128 neuronas)
- Botón "Animar": propagación capa a capa con delay

**Sonificación:**
- Varianza estable = acorde sostenido (consonante)
- Colapso a cero = unísono monótono que se apaga
- Explosión = ruido blanco creciente
- Cada capa = nota en arpeggio (si la varianza se mantiene, el arpeggio suena bien)
- NaN/Infinity = ruido de estática + silencio abrupto

**Ecuación:** `Var[a_l] = Var[a_{l-1}] · n · Var[w]`

**Complejidad:** ~700-800 líneas

---

### MÓDULO 2: ARQUITECTURAS CLÁSICAS (Sims 6-10)

#### 6. Autoencoder Cathedral (`autoencoder-cathedral.html`)

**Concepto:** Compresión como arquitectura. Red en reloj de arena que aprende representaciones.

**Modelo toy:** Autoencoder sobre dígitos 8x8 (sklearn digits embebidos como JSON). Capas: 64→32→bottleneck→32→64. Bottleneck ajustable (2-16).

**Visualización:**
- Red en forma de reloj de arena (encoder→bottleneck→decoder)
- Input original (8x8 grid) a la izquierda
- Bottleneck como puntos en espacio 2D (cuando dim=2)
- Output reconstruido a la derecha
- Flujo de activaciones animado through las capas

**Interacción:**
- Dibujar dígito en canvas de input
- Ajustar tamaño de bottleneck (2, 4, 8, 16)
- Interpolar entre dos dígitos en espacio latente (slider)
- "Fantasía": click en espacio latente para decodificar punto aleatorio
- Botón entrenar: ve loss descender en tiempo real

**Sonificación:**
- Input = melodía de 8 notas (fila por fila del dígito)
- Bottleneck = melodía comprimida (2-16 notas)
- Output = reconstrucción de la melodía
- Error = disonancia entre input y output melodies
- Interpolación = glissando entre dos temas

**Ecuación:** `z = f_enc(x), x̂ = f_dec(z), L = ||x - x̂||²`

---

#### 7. LSTM Memory Cells (`lstm-memory-cells.html`)

**Concepto:** Neuronas con memoria. Cómo las LSTM recuerdan y olvidan.

**Modelo toy:** LSTM de 16 unidades procesando secuencias de caracteres (ej: "abcabc" → predice siguiente). Forward pass completo con gates visibles.

**Visualización:**
- Célula LSTM desplegada en el tiempo (unrolled, ~8 pasos)
- Gates coloreados: Forget (rojo), Input (verde), Output (azul)
- Cell state como río horizontal (grosor = magnitud)
- Hidden state como línea de puntos debajo
- Valores de gate [0,1] como barras al lado de cada compuerta

**Interacción:**
- Escribir secuencia de caracteres, ver procesamiento paso a paso
- Forzar forget gate a 0 (memoria perfecta) o 1 (amnesia)
- Forzar input gate
- Ver qué caracteres la red "recuerda" más
- Presets de secuencias: repetitiva, aleatoria, palindrómica

**Sonificación:**
- Cell state = nota grave continua (bajo pedal, memoria larga)
- Hidden state = melodía que cambia (output en cada paso)
- Forget gate abierto = decay de reverb (olvido audible)
- Input gate abierto = ataque de nota nueva (memoria nueva)
- Patrón repetitivo reconocido = motivo musical que se estabiliza

**Ecuación:** `f_t = σ(W_f·[h_{t-1}, x_t] + b_f)`

---

#### 8. Attention Heatmap (`attention-heatmap.html`)

**Concepto:** Dónde mira la red. Visualización de mecanismo de atención.

**Modelo toy:** Attention de una capa sobre pares de frases cortas (~10 tokens). Q, K, V calculados con matrices de peso pequeñas.

**Visualización:**
- Secuencia input arriba (tokens como cajas)
- Matriz de attention como mapa de calor NxN
- Líneas animadas conectando tokens con peso proporcional al grosor
- Multi-head: tabs para ver cada head por separado
- Softmax distribution como barras por cada query token

**Interacción:**
- Escribir frase (o elegir preset)
- Cambiar número de heads (1, 2, 4, 8)
- Click en token query → highlight sus pesos de atención
- Slider de temperatura del softmax (sharp vs flat)
- Toggle: mostrar valores numéricos en celdas

**Sonificación:**
- Cada token = nota (pitch por posición)
- Attention weight = volumen de conexión
- Multi-head = polifonía (cada head un instrumento/timbre)
- Softmax sharp = nota dominante clara
- Softmax flat = cluster disonante (incertidumbre)

**Ecuación:** `Attention(Q,K,V) = softmax(QK^T / √d_k)V`

---

#### 9. CNN Feature Detectives (`cnn-feature-detectives.html`)

**Concepto:** La corteza visual de la máquina. Qué "ve" una CNN en cada capa.

**Modelo toy:** CNN de 3 capas (Conv3x3→ReLU→Pool)×3 → FC → 10 clases. Sobre dígitos 28x28 (subset MNIST embebido como datos). Pesos pre-entrenados embebidos.

**Visualización:**
- Imagen input (28x28) a la izquierda
- Feature maps de cada capa como grids de mini-imágenes
- Filtros 3x3 ampliados (ver patterns de bordes/texturas)
- Receptive field: al hover sobre feature map, highlight zona de input
- Barra de clasificación final (10 clases, probabilidades)

**Interacción:**
- Dibujar dígito en canvas de input (28x28)
- Click en feature map → zoom + ver filtro que la genera
- Ocultar partes de imagen (occlusión) → ver cambio en clasificación
- Toggle entre filtros aprendidos y filtros clásicos (Sobel, Gaussian)
- Selector de capa para inspeccionar

**Sonificación:**
- Capa 1 (bordes) = staccato, alta frecuencia, seco
- Capa 2 (texturas) = ritmo medio, armónicos
- Capa 3 (objetos) = notas graves, sostenidas
- Clasificación alta confianza = acorde mayor resuelto
- Confusión = cluster disonante
- Occlusión = nota que desaparece

**Ecuación:** `(f * g)(x,y) = ΣΣ f(i,j)·g(x-i, y-j)`

---

#### 10. GAN Duel (`gan-duel.html`)

**Concepto:** Generador vs Discriminador. Competencia creativa.

**Modelo toy:** GAN sobre distribuciones 2D (no imágenes). Generador: ruido 1D → puntos 2D. Discriminador: punto 2D → real/falso. Distribuciones target: anillo, dos clusters, espiral.

**Visualización:**
- Canvas dividido: real data (puntos azules) + generated (puntos rojos)
- Discriminador boundary como mapa de calor de fondo
- Gráfico de losses en tiempo real (G loss vs D loss)
- Histograma de scores del discriminador (bimodal cuando funciona bien)
- Modo "mode collapse" visible (todos los puntos rojos convergen a un punto)

**Interacción:**
- Ajustar learning rates relativos de G y D
- Selector de distribución target (ring, clusters, spiral, grid)
- Pause/step para ver evolución lenta
- Botón "force collapse" para ver mode collapse
- Reset con diferentes inicializaciones

**Sonificación:**
- G loss bajando = melodía ascendente (G mejorando)
- D loss bajando = percusión firme (D clasificando bien)
- Equilibrio Nash = groove estable (ritmo + melodía balanceados)
- Mode collapse = loop monótono (una sola nota repetida)
- Inestabilidad = tempo errático

**Ecuación:** `min_G max_D V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]`

---

### MÓDULO 3: TRANSFORMERS Y LLMs (Sims 11-15)

#### 11. Transformer Anatomy (`transformer-anatomy.html`)

**Concepto:** El corazón de GPT. Arquitectura completa navegable.

**Modelo:** Visualización arquitectural + micro-transformer de 2 capas, 2 heads, vocab ~50 palabras. Forward pass real sobre frases cortas.

**Visualización (Three.js + Canvas):**
- Diagrama arquitectural completo y navegable
- Embeddings posicionales como ondas senoidales superpuestas
- Multi-head attention como streams paralelos
- Feed-forward como bloques de procesamiento
- Layer norm como filtros estabilizadores
- Residual connections como bypass highways

**Interacción:**
- Escribir frase, ver flujo token por token
- Zoom en cada componente (click expande detalle)
- Toggle: vista completa vs vista por capa
- Comparar "base" (2 capas) vs "grande" (esquemático)
- Hover sobre componente = tooltip con explicación

**Sonificación:**
- Embedding = tema musical base (nota por token)
- Attention = variaciones contrapuntísticas
- FFN = desarrollo armónico (transformación de timbre)
- Residual = eco/delay del tema original
- Layer norm = compresión de dinámica

**Ecuación:** `Output = LayerNorm(x + Attention(x)) → LayerNorm(x + FFN(x))`

---

#### 12. Token Embeddings Constellation (`token-embeddings-constellation.html`)

**Concepto:** El espacio semántico. Donde las palabras se convierten en vectores.

**Modelo:** ~500 embeddings pre-computados (GloVe o similar) de 50 dimensiones, proyectados a 2D/3D con PCA/t-SNE. Aritmética vectorial real.

**Visualización:**
- Scatter plot 3D (Three.js) de embeddings proyectados
- Palabras como puntos con etiquetas
- Clusters coloreados por campo semántico (animales, colores, verbos...)
- Vectores de analogía como flechas (rey - hombre + mujer = reina)
- Vecinos más cercanos conectados con líneas tenues

**Interacción:**
- Buscar palabra → zoom + highlight + vecinos
- Aritmética: input A - B + C = ? con resultado visual
- Toggle PCA vs t-SNE vs UMAP (pre-computados)
- Slider de vecinos a mostrar (1-20)
- Selector de campo semántico para filtrar

**Sonificación:**
- Distancia euclídea entre palabras = intervalo musical
- Palabras cercanas = consonancia
- Palabras lejanas = disonancia
- Movimiento en el espacio = glissando
- Analogía exitosa = cadencia armónica (IV-V-I)

**Ecuación:** `king - man + woman ≈ queen`

---

#### 13. Self-Attention Mechanism (`self-attention-mechanism.html`)

**Concepto:** Cada token habla con todos. El mecanismo QKV paso a paso.

**Modelo:** Self-attention real sobre secuencia de 8 tokens. Matrices Q, K, V computadas paso a paso. Visualización de cada operación matricial.

**Visualización:**
- Secuencia de tokens arriba
- Paso 1: Input embeddings (vectores como barras)
- Paso 2: Proyección Q, K, V (tres matrices animadas)
- Paso 3: Q·K^T (heatmap de compatibilidades)
- Paso 4: Escalar por √d_k
- Paso 5: Softmax (normalización)
- Paso 6: Multiplicar por V (output)
- Paso 7: Masking causal (triángulo inferior)

**Interacción:**
- Avanzar paso a paso con botón "Next Step"
- Click en celda de attention → ver cálculo exacto
- Slider de temperatura (sharp/flat softmax)
- Toggle causal mask (GPT) vs bidireccional (BERT)
- Elegir secuencia o escribir tokens

**Sonificación:**
- Q·K^T = disonancia/consonancia por compatibilidad
- Softmax = selección de "melodía principal" (la nota más fuerte)
- V = orquesta que acompaña según atención
- Causal mask = silencio en posiciones futuras
- Cada paso = sección musical que se acumula

**Ecuación:** `Attention(Q,K,V) = softmax(QK^T/√d_k)V`

---

#### 14. Chain of Thought Unrolled (`chain-of-thought-unrolled.html`)

**Concepto:** Razonamiento paso a paso. Cómo "piensa" un LLM.

**Modelo:** Visualización conceptual con datos pre-generados. Árboles de razonamiento para problemas matemáticos y lógicos simples.

**Visualización:**
- Problema arriba (pregunta matemática o lógica)
- Árbol de razonamiento expandiéndose nodo por nodo
- Cada nodo = paso de razonamiento con confianza (color)
- Backtracking visible como nodos grises/rojos
- Barra de confianza acumulada
- Comparación: respuesta directa (1 nodo) vs CoT (N nodos)

**Interacción:**
- Selector de problema (aritmética, lógica, analogía)
- Toggle CoT on/off → comparar accuracy
- Click en nodo → ver razonamiento interno
- Slider "tokens de pensamiento" (más = más preciso, hasta un punto)
- Modo alucinación: ver errores de razonamiento marcados

**Sonificación:**
- Cada paso = compás musical
- Alta confianza = notas claras, consonantes
- Baja confianza = notas tensas, disonantes
- Backtracking = modulación a tonalidad distante + retorno
- Respuesta final correcta = cadencia perfecta
- Respuesta incorrecta = cadencia rota (deceptive cadence)

**Ecuación:** `P(answer|CoT) > P(answer|direct)`

---

#### 15. Emergent Abilities Phase Transition (`emergent-abilities.html`)

**Concepto:** Cuando la escala lo cambia todo. Transiciones de fase en LLMs.

**Modelo:** Visualización de datos reales publicados (scaling laws, benchmarks). No entrena nada. Gráficos interactivos con datos reales.

**Visualización:**
- Gráfico principal: eje X = parámetros (log), eje Y = rendimiento en benchmark
- Curvas para diferentes tareas (aritmética, traducción, razonamiento)
- Puntos críticos marcados donde emergen nuevas habilidades
- Comparar modelos: GPT-2 (1.5B), GPT-3 (175B), GPT-4, Claude, Llama
- Scaling laws como líneas de tendencia (power laws)

**Interacción:**
- Slider de escala (10M → 1T parámetros)
- Selector de benchmark/tarea
- Toggle entre modelos
- Zoom en transiciones de fase
- Animación: ver emergencia progresiva

**Sonificación:**
- Modelo pequeño = melodía simple, monofónica, 1 instrumento
- Modelo mediano = polifonía básica, 2-3 instrumentos
- Modelo grande = orquesta completa con estructura
- Transición de fase = cambio abrupto de textura musical
- Scaling law = crescendo gradual

**Ecuación:** `L(N) = (N_c/N)^α (Chinchilla scaling law)`

---

### MÓDULO 4: FRONTERAS (Sims 16-20)

#### 16. Diffusion Models: De Ruido a Arte (`diffusion-models.html`)

**Concepto:** El proceso de denoising. Cómo emerge orden del caos.

**Modelo toy:** Diffusion sobre imágenes 16x16 (o 32x32). Forward process: añadir ruido gaussiano en T pasos. Reverse: denoiser simple (pequeña red). Pesos pre-entrenados embebidos.

**Visualización:**
- Timeline horizontal: T pasos de izquierda (ruido puro) a derecha (imagen limpia)
- Imagen principal grande mostrando paso actual
- Histograma de distribución de pixels por paso
- Gráfico de varianza del ruido (schedule lineal/coseno)
- Forward (añadir ruido) vs Reverse (quitar ruido) como dos direcciones

**Interacción:**
- Slider de pasos (1-50)
- Botón play: animar denoising completo
- Click en cualquier paso de la timeline → saltar
- Selector de schedule: lineal vs coseno vs exponencial
- Selector de imagen target: dígito, patrón geométrico
- Toggle: ver forward process (destrucción) vs reverse (creación)

**Sonificación:**
- Ruido puro = ruido blanco (white noise)
- Denoising = filtro lowpass que se abre progresivamente
- Imagen clara = melodía limpia con armónicos definidos
- Schedule lineal = transición gradual
- Schedule coseno = transición más suave al principio y al final
- Cada paso = nota que se aclara (de ruidosa a pura)

**Ecuación:** `x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε`

---

#### 17. Reinforcement Learning: Grid World (`reinforcement-learning.html`)

**Concepto:** Aprender de la experiencia. Un agente que navega un mundo.

**Modelo:** Q-Learning real sobre grid 8x8. Agente, obstáculos, recompensas, meta. Tabla Q actualizada en tiempo real.

**Visualización:**
- Grid 8x8 con tiles de colores
- Agente como entidad que se mueve
- Obstáculos (paredes, trampas)
- Recompensas como estrellas/gemas
- Q-values como mapa de calor superpuesto (4 direcciones por celda)
- Flechas de política óptima por celda
- Gráfico de reward acumulado por episodio

**Interacción:**
- Click para colocar/quitar paredes
- Click derecho para colocar recompensas
- Sliders: epsilon (exploración), gamma (descuento), alpha (learning rate)
- Botones: train 1 episodio, train 100, train 1000
- Toggle: mostrar Q-values vs política vs ambos
- Selector de algoritmo: Q-Learning vs SARSA

**Sonificación:**
- Paso del agente = nota rítmica (beat)
- Recompensa = acorde mayor (campana)
- Trampa = acorde menor (caída)
- Exploración (random) = notas aleatorias
- Explotación (greedy) = melodía aprendida estable
- Meta alcanzada = fanfarria breve
- Episodio sin meta = fade out

**Ecuación:** `Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]`

---

#### 18. Knowledge Distillation: Maestro y Alumno (`knowledge-distillation.html`)

**Concepto:** Transferencia de conocimiento. Una red grande enseña a una pequeña.

**Modelo toy:** Teacher: red 64→32→16→10 (pre-entrenada, pesos embebidos). Student: red 64→8→10. Entrenamiento del student con soft labels del teacher.

**Visualización:**
- Dos redes lado a lado: Teacher (grande) y Student (pequeña)
- Mismo input → comparar outputs como barras de probabilidad
- Soft labels: distribuciones suaves (temperatura alta) vs hard labels (one-hot)
- Gráfico de divergencia KL entre teacher y student
- Accuracy de ambas redes en tiempo real
- Flujo de "conocimiento" del teacher al student (partículas animadas)

**Interacción:**
- Slider de temperatura T (1-20)
- Selector de input (dígito para clasificar)
- Botón entrenar: ver student acercarse al teacher
- Toggle: entrenar con hard labels vs soft labels
- Comparar accuracy final de cada método
- Slider de peso α entre soft loss y hard loss

**Sonificación:**
- Teacher = orquesta completa (muchos armónicos)
- Student (inicio) = nota simple, pobre
- Student (entrenado) = se enriquece, acercándose al timbre del teacher
- Divergencia KL alta = disonancia entre ambos
- Divergencia KL baja = consonancia
- Temperatura alta = sonido difuso/reverb
- Temperatura baja = sonido seco/staccato

**Ecuación:** `L = α·KL(σ(z_T/T) || σ(z_S/T))·T² + (1-α)·CE(y, σ(z_S))`

---

#### 19. Mixture of Experts: El Comité (`mixture-of-experts.html`)

**Concepto:** División del trabajo. Expertos especializados con un router inteligente.

**Modelo toy:** 4 expertos (redes pequeñas de 2 capas) + router (softmax gate). Cada experto especializado en un cuadrante del espacio de input. Top-K routing (K=1 o K=2).

**Visualización:**
- Router central como nodo grande
- N expertos como nodos alrededor (color por especialización)
- Tokens de input llegan al router, flechas dirigidas a expertos seleccionados
- Grosor de flecha = peso de routing
- Heatmap del router: qué regiones del input van a qué experto
- Load balancing: barras mostrando cuánto trabajo tiene cada experto
- Output combinado ponderado

**Interacción:**
- Slider número de expertos (2, 4, 8)
- Selector Top-K (1 o 2)
- Click en espacio de input: ver a qué experto se enruta
- Toggle load balancing loss on/off
- Forzar routing a un experto específico
- Visualizar fronteras de decisión del router

**Sonificación:**
- Cada experto = instrumento distinto (piano, strings, flute, brass)
- Router = director que asigna solos
- Token enrutado = instrumento asignado toca nota
- Top-1 = solo, Top-2 = dueto
- Load balancing malo = un instrumento domina (monotonía)
- Load balancing bueno = ensemble equilibrado (polifonía rica)

**Ecuación:** `y = Σᵢ g(x)ᵢ · Eᵢ(x), g(x) = TopK(softmax(W_g·x))`

---

#### 20. Superposition: Dentro de la Caja Negra (`superposition-black-box.html`)

**Concepto:** Polisemántica e interpretabilidad mecánica. Cómo las neuronas codifican más features de las que "deberían".

**Modelo:** Demostración de superposición: red con N neuronas que codifica M > N features. Toy model de Anthropic: features como vectores en espacio de dimensión N, con sparsity variable.

**Visualización:**
- Espacio 2D/3D representando neuronas (dimensiones del modelo)
- Features como vectores/direcciones en ese espacio
- Cuando M ≤ N: cada feature tiene su dimensión propia (sin interferencia)
- Cuando M > N: features se superponen (ángulos entre vectores)
- Interferencia visible como solapamiento de colores
- Sparsity slider: más sparse → más features caben sin interferencia
- Diagrama de fase: ejes (M/N ratio) vs (sparsity)

**Interacción:**
- Slider N (dimensiones del modelo): 2-10
- Slider M (features a codificar): 1-20
- Slider sparsity (0-1): qué fracción de features están activas
- Activar/desactivar features individuales
- Ver interferencia entre features específicas
- Toggle: vista geométrica vs diagrama de fase

**Sonificación:**
- Features individuales = notas puras (sin superposición)
- Superposición = acordes (múltiples features en una neurona)
- Interferencia destructiva = beating (batimiento audible entre frecuencias cercanas)
- Sparsity alta = notas claras, bien separadas
- Sparsity baja = cluster denso, disonante
- Transición de fase = cambio abrupto de textura sonora

**Ecuación:** `x̂ = W^T W x, L = ||x - x̂||² + λ·sparsity`

---

## Index.html Pattern

El `index.html` sigue el patrón exacto de Math Visual Lab:

- **Grid responsive**: 1/2/3 columnas
- **Cards** con canvas preview animado (140px), tag de módulo, título, descripción, ecuación, botón
- **Previews**: Mini-simulaciones ejecutándose en cada card
- **Header**: "AI Visual Lab" + "La inteligencia hecha visible y audible"
- **Estadísticas**: 20 simulaciones, 4 módulos
- **Footer**: "AI Visual Lab · La inteligencia hecha visible y audible"

### Colores de tags por módulo
```
Fundamentos:   Red    #ef4444
Arquitecturas: Amber  #f59e0b
Transformers:  Violet #8b5cf6
Fronteras:     Cyan   #06b6d4
```

## Workflow de Ejecución

### Orden de construcción
1. **Batch 1** (Módulo 1): Sims 1-5, luego index.html parcial
2. **Batch 2** (Módulo 2): Sims 6-10, actualizar index.html
3. **Batch 3** (Módulo 3): Sims 11-15, actualizar index.html
4. **Batch 4** (Módulo 4): Sims 16-20, index.html final

### Para cada simulación
1. Crear archivo HTML autocontenido
2. Implementar visualización completa
3. Implementar toda la interactividad
4. Implementar sonificación
5. Verificar responsive y DPR
6. Verificar lazy audio init (iOS compatible)

---

**Color de disciplina:** `#ef4444` (Rojo)
**Total:** 20 simulaciones · 4 módulos · ~15,000-20,000 líneas estimadas
**Última actualización:** 2026-02-01
