# AI Visual Lab

**El primer laboratorio completo de IA explicable visual y sonoramente.**

20 simulaciones interactivas que convierten conceptos abstractos de inteligencia artificial en experiencias sensoriales: se ve, se escucha y se manipula el aprendizaje automático. Desde el perceptrón hasta arquitecturas especulativas, todo ejecutándose en tiempo real en tu navegador.

> *No basta con ver la red — hay que escuchar el gradiente desvanecerse, sentir la fricción del learning rate, oír la disonancia del error.*

## Características

- **20 simulaciones interactivas** con controles en tiempo real (sliders, botones, selección de datos)
- **Sonificación completa** — cada simulación mapea sus variables a audio mediante Web Audio API (frecuencia, volumen, timbre, ritmo)
- **Sin instalación** — archivos HTML autocontenidos, abre en cualquier navegador moderno
- **Sin dependencias de servidor** — todo corre en el cliente, incluyendo el entrenamiento de redes neuronales
- **Redes neuronales from scratch** — forward/backward pass implementados en JavaScript puro
- **Tutorial pedagógico** con guías detalladas para cada simulación, glosario de 156 términos y rutas de aprendizaje
- **Accesible** — diseñado para enseñar IA a personas sin formación técnica previa

## Simulaciones

### Módulo 1: Fundamentos

| # | Simulación | Descripción |
|---|-----------|-------------|
| 01 | [Perceptrón Viviente](perceptron-viviente.html) | Neurona artificial con frontera de decisión interactiva y sonificación del error |
| 02 | [Gradient Descent Landscape](gradient-descent-landscape.html) | Superficie 3D de pérdida con optimización en tiempo real (Three.js) |
| 03 | [Backpropagation Flow](backpropagation-flow.html) | Visualización del flujo de gradientes capa a capa |
| 04 | [Activation Function Orchestra](activation-function-orchestra.html) | 8 funciones de activación en paralelo con sonificación diferenciada |
| 05 | [Weight Initialization Galaxy](weight-initialization-galaxy.html) | Distribución de pesos como partículas en el espacio |

### Módulo 2: Arquitecturas Clásicas

| # | Simulación | Descripción |
|---|-----------|-------------|
| 06 | [Autoencoder Cathedral](autoencoder-cathedral.html) | Compresión y reconstrucción como arquitectura visual |
| 07 | [LSTM Memory Cells](lstm-memory-cells.html) | Neuronas con memoria: gates de entrada, olvido y salida |
| 08 | [Attention Heatmap](attention-heatmap.html) | Mapas de calor mostrando dónde mira la red |
| 09 | [CNN Feature Detectives](cnn-feature-detectives.html) | Corteza visual artificial: detección jerárquica de patrones |
| 10 | [GAN Duel](gan-duel.html) | Duelo en tiempo real entre Generador y Discriminador |

### Módulo 3: Transformers y LLMs

| # | Simulación | Descripción |
|---|-----------|-------------|
| 11 | [Transformer Anatomy](transformer-anatomy.html) | Arquitectura completa de un Transformer con flujo de datos (Three.js) |
| 12 | [Token Embeddings Constellation](token-embeddings-constellation.html) | Espacio semántico 3D de embeddings (Three.js) |
| 13 | [Self-Attention Mechanism](self-attention-mechanism.html) | Mecanismo Query-Key-Value en detalle |
| 14 | [Chain of Thought Unrolled](chain-of-thought-unrolled.html) | Razonamiento paso a paso desenrollado |
| 15 | [Emergent Abilities](emergent-abilities.html) | Transiciones de fase y capacidades emergentes por escala |

### Módulo 4: Fronteras

| # | Simulación | Descripción |
|---|-----------|-------------|
| 16 | [Diffusion Models](diffusion-models.html) | Proceso de difusión: de ruido gaussiano a imagen |
| 17 | [Reinforcement Learning](reinforcement-learning.html) | Grid World con Q-learning y exploración vs explotación |
| 18 | [Knowledge Distillation](knowledge-distillation.html) | Transferencia de conocimiento de red maestra a alumna |
| 19 | [Mixture of Experts](mixture-of-experts.html) | Router dinámico distribuyendo tokens entre expertos |
| 20 | [Superposition & Black Box](superposition-black-box.html) | Representaciones superpuestas dentro de la caja negra |

## Guías y Tutorial

El directorio `guides/` contiene material pedagógico completo:

- **20 guías individuales** (`.md` + `.html`) — una por simulación, con matemáticas detalladas, tablas de sonificación, glosario y ejercicios
- **[Tutorial Completo](guides/tutorial-completo.html)** — recorrido integrado por las 20 simulaciones con:
  - Rutas de aprendizaje (Fundamental, Practitioner, Researcher)
  - Deep Dives colapsables con ecuaciones y mapeo de audio
  - Mapa conceptual interactivo SVG
  - Mega-glosario unificado (156 términos)
  - 47 ejercicios con niveles de dificultad

## Uso

No requiere instalación, servidor ni build. Solo un navegador moderno.

```bash
# Clonar el repositorio
git clone https://github.com/cjlkaiser-cpu/AI-Visual-Lab.git

# Abrir la landing page
open "AI Visual Lab/index.html"

# O abrir cualquier simulación individual
open "AI Visual Lab/perceptron-viviente.html"
```

Alternativa: descarga el ZIP y abre `index.html` directamente en tu navegador.

### Controles comunes

- **Sliders** — ajustan parámetros en tiempo real (learning rate, epochs, neuronas, etc.)
- **Play/Pause** — detener y reanudar la simulación
- **Reset** — reiniciar al estado inicial
- **Audio** — activar/desactivar sonificación, ajustar volumen y tipo de onda

## Stack Tecnológico

| Capa | Tecnología |
|------|-----------|
| Estructura | HTML5 autocontenido (sin bundler) |
| Estilos | CSS3 con custom properties, tema oscuro |
| Lógica | JavaScript ES6+ vanilla (sin frameworks) |
| Renderizado | Canvas 2D + Three.js (CDN) para vistas 3D |
| Audio | Web Audio API (síntesis FM/aditiva, ADSR, filtros) |
| ML | Redes neuronales implementadas from scratch en JS |
| Landing | Tailwind CSS (CDN) solo en `index.html` |
| Tipografía | Inter + JetBrains Mono (Google Fonts) |

## Requisitos del Navegador

- Chrome 80+, Firefox 78+, Safari 14+, Edge 80+
- JavaScript habilitado
- Web Audio API (para sonificación)
- WebGL (para simulaciones 3D con Three.js: #02, #11, #12)
- Resolución recomendada: 1280x720 o superior

## Estructura del Proyecto

```
AI Visual Lab/
├── index.html                          # Landing con 20 cards animadas
├── README.md                           # Este archivo
├── CLAUDE.md                           # Documentación técnica interna
├── perceptron-viviente.html            # Simulación 01
├── gradient-descent-landscape.html     # Simulación 02
├── ...                                 # Simulaciones 03-19
├── superposition-black-box.html        # Simulación 20
└── guides/
    ├── tutorial-completo.html          # Tutorial integrado
    ├── perceptron-viviente.md          # Guía 01
    ├── perceptron-viviente.html        # Guía 01 (HTML)
    ├── ...                             # Guías 02-19
    ├── superposition-black-box.md      # Guía 20
    └── superposition-black-box.html    # Guía 20 (HTML)
```

## Licencia

Parte del proyecto [EigenLab](https://github.com/cjlkaiser-cpu).
