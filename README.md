<div align="center">

# ⬡ NeuralForge

**A neural network engine built entirely from scratch in TypeScript — no TensorFlow, no PyTorch, no autograd, no math libraries.**

Backpropagation, the Adam optimizer, L2 regularization, inverted dropout, mini-batch training, He/Xavier initialization and real-time training visualization — all implemented from first principles and verified by a finite-difference gradient check.

![Vue 3](https://img.shields.io/badge/Vue-3.5-42b883?logo=vue.js&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-6.x-3178c6?logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-8-646cff?logo=vite&logoColor=white)
![Vitest](https://img.shields.io/badge/tested_with-Vitest-6da13f?logo=vitest&logoColor=white)
![Zero ML Dependencies](https://img.shields.io/badge/ML_dependencies-zero-7c3aed)
![License](https://img.shields.io/badge/License-MIT-blue)

</div>

---

## 📚 Table of Contents

- [Why this is hard](#-why-this-is-hard)
- [Features](#-features)
- [Live demo walkthrough](#-live-demo-walkthrough)
- [The mathematics](#-the-mathematics)
- [Correctness: the gradient check](#-correctness-the-gradient-check)
- [Getting started](#-getting-started)
- [Scripts](#-scripts)
- [Architecture](#-architecture)
- [Design decisions](#-design-decisions)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 🧠 Why this is hard

Most developers use deep-learning frameworks as black boxes. NeuralForge re-derives and implements every moving part by hand:

- **Linear algebra** — a `Tensor` class over `Float32Array` with matmul, transpose, broadcasting add and Hadamard product.
- **Backpropagation** — exact, per-layer chain-rule gradients with no automatic differentiation.
- **Optimizers** — SGD and Adam (bias-corrected first/second moments), both with optional L2 weight decay.
- **Regularization** — inverted dropout on hidden layers and L2 weight decay.
- **Initialization** — He scaling for ReLU-family activations, Xavier for the rest.
- **Numerically stable BCE** — binary cross-entropy with ε-clamping and the exact `(p − y)` output gradient.
- **Reproducibility** — a single seed drives weight init, data generation, the train/val split and mini-batch shuffling.

And the whole thing is wired to a real-time Vue 3 visualization so you can _watch_ the decision boundary form.

---

## ✨ Features

### Engine (`src/engine/`)

| File            | Responsibility                                                                                                 |
| --------------- | -------------------------------------------------------------------------------------------------------------- |
| `Tensor.ts`     | Row-major `Float32Array` matrix — matmul, transpose, broadcasting add, Hadamard, column-sum, seedable `randn`. |
| `Rng.ts`        | Seedable PRNG (mulberry32) with uniform, Gaussian (Box–Muller) and Fisher–Yates shuffle.                       |
| `Activation.ts` | ReLU, Leaky ReLU, Sigmoid, Tanh, Linear — forward + analytic derivative.                                       |
| `Layer.ts`      | Dense layer — forward/backward, inverted dropout, L2 weight decay, SGD + Adam updates.                         |
| `Network.ts`    | MLP — forward, BCE loss, backprop, mini-batch step, evaluation, reproducible reset.                            |

### Training

- **Mini-batch gradient descent** with a configurable batch size (full-batch when the size ≥ dataset).
- **Learning-rate scheduler** — per-epoch exponential decay (`lr · decay^epoch`).
- **Train / validation split** with reproducible shuffling, so you can see generalization, not just memorization.
- **L2 regularization** and **dropout** to combat overfitting on the spiral/XOR tasks.
- **Correct epoch accounting** — one epoch = one full pass over the training set.

### Visualization (`src/components/`)

| Component             | What it shows                                                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Decision Boundary** | Canvas probability field, computed with a _single batched forward pass_ over a 100×100 lattice and upscaled. Live data overlay. |
| **Network Graph**     | SVG neurons + weight edges (blue = +, red = −, thickness = magnitude).                                                          |
| **Training Chart**    | Loss & accuracy over epochs — **solid = train, dashed = validation**.                                                           |
| **Control Panel**     | Edit architecture, optimizer, learning rate, LR decay, L2, dropout, batch size, validation split, seed and speed.               |

### Quality

- **17 unit tests** including a **finite-difference gradient check**.
- **ESLint + Prettier** (flat config) and a **GitHub Actions CI** pipeline (lint → format → typecheck → test → build).
- Strict TypeScript with `erasableSyntaxOnly`, `noUnusedLocals` and `noUnusedParameters`.

---

## 🎮 Live demo walkthrough

1. Select the **🌀 Spiral** dataset (the hardest — it needs depth).
2. Build **2–3 hidden layers of 16 units** with ReLU.
3. Choose **Adam** at `lr ≈ 0.03` with a small **LR decay** of `0.999`.
4. Set a **20% validation split** and press **▶ Train**.
5. Watch the decision boundary wind into the spiral while the dashed validation curves track the solid training ones. If the dashed lines lag behind, add **dropout** or **L2** to close the gap.

Every run is reproducible: fix the **seed** and you get identical weights, data and batches.

---

## 🔬 The mathematics

### Forward pass (per layer)

```
Z = X · W + b          // affine transform
A = σ(Z)               // activation
A = dropout(A)         // inverted dropout (training only, hidden layers)
```

### Backward pass (chain rule)

```
dZ = dA ⊙ σ'(Z)        // hidden layers
dW = Xᵀ · dZ / m  + λ·W // weight gradient (+ L2 decay)
db = colSum(dZ) / m     // bias gradient
dX = dZ · Wᵀ            // gradient passed upstream
```

### Output layer — BCE + sigmoid

For a sigmoid output with binary cross-entropy, the gradient w.r.t. the **pre-activation** collapses to a clean form:

```
∂L/∂Z_out = (p − y)          // returned raw by the loss
dW_out     = X_outᵀ · (p − y) / m
```

The activation derivative is applied **exactly once** (via a dedicated `backwardFromDZ` path on the output layer) and the `1/m` batch averaging happens **exactly once** inside the layer. Applying either twice is the classic "it still kind of trains, but the gradients are wrong" bug — see below.

### Adam optimizer

```
m_t = β₁·m_{t-1} + (1-β₁)·g
v_t = β₂·v_{t-1} + (1-β₂)·g²
m̂   = m_t / (1-β₁ᵗ)          // bias correction
v̂   = v_t / (1-β₂ᵗ)
w  ← w - α · m̂ / (√v̂ + ε)
```

---

## ✅ Correctness: the gradient check

Hand-written backprop is easy to get subtly wrong, so the engine is validated against numerical gradients. For every weight `w`, the analytic gradient from backprop is compared to a central finite difference of the loss:

```
∂L/∂w ≈ ( L(w + ε) − L(w − ε) ) / 2ε
```

The test asserts that the **norm-based relative error** between analytic and numeric gradients is below `1e-2`:

```
‖ grad_analytic − grad_numeric ‖
────────────────────────────────  < 0.01
‖ grad_analytic ‖ + ‖ grad_numeric ‖
```

This check earns its keep: it caught a real double-`1/m` scaling bug during development (Adam had hidden it, because Adam is invariant to constant gradient scaling — the gradient check is not).

```bash
npm test
# ✓ Gradient check (analytic vs finite difference)
# Test Files 1 passed (1)
#      Tests 17 passed (17)
```

---

## 🚀 Getting started

```bash
git clone https://github.com/One-Of-The-Ekhandes/neuralforge.git
cd neuralforge
npm install
npm run dev
```

Open the printed local URL (default `http://localhost:5173`), pick a dataset and press **▶ Train**.

> **Requirements:** Node.js 20+ and npm.

---

## 🛠 Scripts

| Script                 | Description                                      |
| ---------------------- | ------------------------------------------------ |
| `npm run dev`          | Start the Vite dev server.                       |
| `npm run build`        | Type-check (`vue-tsc`) and build for production. |
| `npm run preview`      | Preview the production build.                    |
| `npm test`             | Run the Vitest suite once.                       |
| `npm run test:watch`   | Run tests in watch mode.                         |
| `npm run typecheck`    | Type-check the whole project.                    |
| `npm run lint`         | Lint with ESLint.                                |
| `npm run lint:fix`     | Lint and auto-fix.                               |
| `npm run format`       | Format with Prettier.                            |
| `npm run format:check` | Verify formatting (used in CI).                  |

---

## 📁 Architecture

```
src/
├── engine/
│   ├── Tensor.ts        # Linear-algebra primitives (Float32Array)
│   ├── Rng.ts           # Seedable PRNG (uniform · gaussian · shuffle)
│   ├── Activation.ts    # σ and σ' for each activation type
│   ├── Layer.ts         # DenseLayer: forward/backward, dropout, L2, SGD/Adam
│   └── Network.ts        # MLP: training step, evaluation, gradient check hooks
├── datasets/
│   └── index.ts         # Spiral · Circles · XOR · Blobs + train/val split
├── composables/
│   └── useTrainer.ts    # rAF training loop, mini-batching, LR schedule, metrics
├── components/
│   ├── DecisionBoundary.vue  # Batched canvas probability field
│   ├── NetworkGraph.vue      # SVG weight visualization
│   ├── TrainingChart.vue     # Train/val loss & accuracy chart
│   └── ControlPanel.vue      # Hyperparameter UI (draft-based, no prop mutation)
└── App.vue

tests/
└── engine.test.ts       # Tensor · RNG · activations · gradient check · training

.github/workflows/ci.yml # lint → format → typecheck → test → build
```

Data flow is one-directional: **engine → `useTrainer` → components**. The engine has zero knowledge of Vue and can be imported and tested in isolation.

---

## 🧩 Design decisions

- **`Float32Array` everywhere** — cache-friendly, GPU-parity numerics, and a realistic constraint that the gradient check accounts for.
- **`shallowRef` for the network** — the hot training object is mutated thousands of times per second; deep reactivity would be a performance trap.
- **Batched decision boundary** — the probability field is one matrix multiply over a 100×100 grid instead of ~78k individual forward passes per render.
- **Draft-based control panel** — the UI edits a local copy of the config and emits it on _Apply_, so props are never mutated (clean one-way data flow).
- **One seed to rule them all** — initialization, data, split and shuffling share a seed, making every experiment reproducible.

---

## 🗺 Roadmap

- Multi-class classification (softmax + categorical cross-entropy).
- Additional optimizers (Momentum, RMSProp) and batch normalization.
- Model export/import (JSON weights).
- Interactive weight inspection on the network graph.

---

## 📄 License

MIT — built by [Rujuta Ekhande](https://github.com/One-Of-The-Ekhandes).
