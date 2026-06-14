# ⬡ NeuralForge

**A neural network engine built entirely from scratch in TypeScript — no TensorFlow, no PyTorch, no ML libraries.**

Implements backpropagation, the Adam optimizer, He/Xavier initialization, and real-time training visualization in Vue 3 — all from first principles.

![Vue 3](https://img.shields.io/badge/Vue-3.5-42b883?logo=vue.js)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178c6?logo=typescript)
![Zero ML Dependencies](https://img.shields.io/badge/ML_Dependencies-Zero-purple)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## 🧠 What makes this hard

Most developers use ML frameworks as black boxes. NeuralForge implements every layer from first principles:

- **Matrix algebra** — `Tensor` class with `Float32Array`, matmul, transpose, broadcasting
- **Backpropagation** — exact chain-rule gradient computation per layer
- **Adam optimizer** — bias-corrected first and second moment estimates
- **He / Xavier initialization** — correct weight scaling per activation function
- **Binary cross-entropy loss** — numerically stable with ε clamping
- **Four datasets** — Spiral (hardest), Circles, XOR, Blobs

---

## ✨ Features

### Engine (`src/engine/`)
| File | Purpose |
|---|---|
| `Tensor.ts` | Row-major Float32Array matrix — matmul, transpose, broadcasting add, Hadamard product, Box-Muller randn |
| `Activation.ts` | ReLU, Leaky ReLU, Sigmoid, Tanh, Linear — forward + analytic derivative |
| `Layer.ts` | Dense layer — forward pass, backward pass (full chain rule), SGD + Adam update |
| `Network.ts` | Multi-layer perceptron — forward, BCE loss, backprop, step, reset |

### Visualization
| Component | What it shows |
|---|---|
| **Decision Boundary** | Canvas grid — each pixel = network output probability. Live data overlay |
| **Network Graph** | SVG — weights as edges (blue=+, red=−, thickness=magnitude). Updates every 10 epochs |
| **Training Chart** | Dual SVG line chart — loss (red) and accuracy (green) over time |
| **Control Panel** | Add/remove layers, tune units, switch activation, change optimizer, control speed |

---

## 🔬 The Maths

### Forward pass (per layer)
```
Z = X @ W + b     // linear transform
A = σ(Z)          // activation
```

### Backward pass (chain rule)
```
dZ = dA ⊙ σ'(Z)           // element-wise with activation derivative
dW = Xᵀ @ dZ / m          // weight gradient
db = colSum(dZ) / m        // bias gradient
dX = dZ @ Wᵀ              // upstream gradient
```

### Adam optimizer
```
m_t = β₁·m_{t-1} + (1-β₁)·g          // first moment
v_t = β₂·v_{t-1} + (1-β₂)·g²         // second moment
m̂ = m_t / (1-β₁ᵗ)                    // bias correction
v̂ = v_t / (1-β₂ᵗ)
w ← w - α · m̂ / (√v̂ + ε)
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/One-Of-The-Ekhandes/neuralforge.git
cd neuralforge
npm install
npm run dev
```

Open `http://localhost:5173` — click **▶ Train** and watch the decision boundary form in real time.

### Try this:
1. Select **Spiral** dataset (hardest — requires depth)
2. Add 2–3 layers of 16 units with ReLU
3. Use **Adam** optimizer at `lr = 0.003`
4. Watch the boundary learn the spiral shape over ~500 epochs

---

## 📁 Architecture

```
src/
├── engine/
│   ├── Tensor.ts        # Linear algebra primitives
│   ├── Activation.ts    # σ and σ' for each activation type
│   ├── Layer.ts         # DenseLayer with full forward/backward
│   └── Network.ts       # MLP — training step, inference, reset
├── datasets/
│   └── index.ts         # Spiral, Circles, XOR, Blobs generators
├── composables/
│   └── useTrainer.ts    # RAF-based reactive training loop
├── components/
│   ├── DecisionBoundary.vue  # Canvas decision boundary renderer
│   ├── NetworkGraph.vue      # SVG weight visualization
│   ├── TrainingChart.vue     # SVG loss/accuracy chart
│   └── ControlPanel.vue      # Hyperparameter UI
└── App.vue
```

---

## 📄 License

MIT — built by [Rujuta Ekhande](https://github.com/One-Of-The-Ekhandes)
