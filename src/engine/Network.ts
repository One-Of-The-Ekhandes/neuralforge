/**
 * Network — Multi-layer perceptron with training and inference.
 *
 * Supports binary classification with binary cross-entropy loss.
 * The final layer must use sigmoid activation (output ∈ [0,1]).
 *
 * Training step:
 *   1. Forward pass through all layers (dropout active when training)
 *   2. Compute BCE loss and dZ for the output layer
 *   3. Backpropagate
 *   4. Update weights with the chosen optimizer (+ optional L2 decay)
 */

import { Tensor } from './Tensor';
import { DenseLayer } from './Layer';
import { Rng } from './Rng';
import type { ActivationType } from './Activation';

export type OptimizerType = 'sgd' | 'adam';

export interface LayerConfig {
  units: number;
  activation: ActivationType;
}

export interface NetworkConfig {
  inputSize: number;
  layers: LayerConfig[];
  optimizer: OptimizerType;
  learningRate: number;
  /** L2 weight-decay coefficient (0 = off). */
  l2?: number;
  /** Dropout probability applied to hidden layers (0 = off). */
  dropout?: number;
  /** Seed for reproducible weight initialization. */
  seed?: number;
}

export interface Metrics {
  loss: number;
  accuracy: number;
}

export class Network {
  layers: DenseLayer[] = [];
  readonly config: NetworkConfig;
  /** Number of gradient-descent steps performed. */
  steps = 0;
  private rng: Rng;

  constructor(config: NetworkConfig) {
    this.config = config;
    this.rng = new Rng(config.seed ?? 1234);
    this._build();
  }

  private _build() {
    this.rng.seed(this.config.seed ?? 1234);
    this.layers = [];
    let prev = this.config.inputSize;
    const dropout = this.config.dropout ?? 0;
    const last = this.config.layers.length - 1;
    this.config.layers.forEach((lc, i) => {
      const layer = new DenseLayer(prev, lc.units, lc.activation, this.rng);
      // Dropout on hidden layers only — never on the output layer.
      layer.dropout = i < last ? dropout : 0;
      this.layers.push(layer);
      prev = lc.units;
    });
  }

  /** Full forward pass → output Tensor [m, outputSize]. */
  forward(input: Tensor, training = false): Tensor {
    let x = input;
    for (const layer of this.layers) x = layer.forward(x, training);
    return x;
  }

  /**
   * Binary cross-entropy: L = -1/m Σ [y·log(p) + (1-y)·log(1-p)]
   *
   * For a sigmoid output the gradient w.r.t. the pre-activation simplifies
   * exactly to dL/dZ ∝ (p − y). We return the raw (p − y) and feed it to the
   * output layer via `backwardFromDZ`; the single 1/m batch averaging is
   * applied once inside the layer's weight-gradient computation. This applies
   * the sigmoid derivative exactly once (not twice) and averages exactly once.
   */
  private bce(pred: Tensor, target: Tensor): [number, Tensor] {
    const m = pred.rows;
    const ε = 1e-7;
    let loss = 0;
    const dZ = new Tensor(m, 1);

    for (let i = 0; i < m; i++) {
      const p = Math.min(Math.max(pred.get(i, 0), ε), 1 - ε);
      const y = target.get(i, 0);
      loss += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
      dZ.set(i, 0, p - y);
    }
    return [loss / m, dZ];
  }

  /** Sum of squared weights across all layers (for the L2 penalty term). */
  private l2Norm(): number {
    let s = 0;
    for (const layer of this.layers) {
      const w = layer.weights.data;
      for (let i = 0; i < w.length; i++) s += w[i] * w[i];
    }
    return s;
  }

  /** One gradient-descent step on a (mini-)batch. Returns training metrics. */
  trainStep(xBatch: Tensor, yBatch: Tensor, lr: number): Metrics {
    const pred = this.forward(xBatch, true);
    const [dataLoss, dZ] = this.bce(pred, yBatch);

    // Backward: output layer consumes dZ directly; hidden layers use dA.
    const last = this.layers.length - 1;
    let grad = this.layers[last].backwardFromDZ(dZ);
    for (let i = last - 1; i >= 0; i--) grad = this.layers[i].backward(grad);

    const l2 = this.config.l2 ?? 0;
    for (const layer of this.layers) {
      if (this.config.optimizer === 'adam') layer.updateAdam(lr, l2);
      else layer.updateSGD(lr, l2);
    }

    this.steps++;
    const reportedLoss = dataLoss + (l2 > 0 ? (l2 / 2) * this.l2Norm() : 0);
    return { loss: reportedLoss, accuracy: this.accuracy(pred, yBatch) };
  }

  /** Inference-only evaluation (no dropout, no weight updates). */
  evaluate(X: Tensor, y: Tensor): Metrics {
    const pred = this.forward(X, false);
    const [loss] = this.bce(pred, y);
    return { loss, accuracy: this.accuracy(pred, y) };
  }

  /** Data-term BCE loss only (no dropout, no L2). Used by gradient checks. */
  dataLoss(X: Tensor, y: Tensor): number {
    return this.bce(this.forward(X, false), y)[0];
  }

  /**
   * Forward + backward WITHOUT updating weights or applying dropout/L2.
   * Populates each layer's `dWeights`/`dBiases` with the analytic gradient
   * of the data loss. Returns that loss. Used by the finite-difference
   * gradient check in the test suite.
   */
  computeGradients(X: Tensor, y: Tensor): number {
    const pred = this.forward(X, false);
    const [loss, dZ] = this.bce(pred, y);
    const last = this.layers.length - 1;
    let grad = this.layers[last].backwardFromDZ(dZ);
    for (let i = last - 1; i >= 0; i--) grad = this.layers[i].backward(grad);
    return loss;
  }

  accuracy(pred: Tensor, target: Tensor): number {
    let correct = 0;
    for (let i = 0; i < pred.rows; i++) {
      const p = pred.get(i, 0) >= 0.5 ? 1 : 0;
      if (p === target.get(i, 0)) correct++;
    }
    return correct / pred.rows;
  }

  /** Predict class probability for a single 2D point (decision boundary). */
  predict2D(x: number, y: number): number {
    return this.forward(new Tensor(1, 2, [x, y]), false).get(0, 0);
  }

  /** Reset weights and optimizer state (reproducible for a fixed seed). */
  reset() {
    this.steps = 0;
    this._build();
  }
}
