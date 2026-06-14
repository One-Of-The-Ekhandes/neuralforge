/**
 * Network — Multi-layer perceptron with training and inference.
 *
 * Supports binary classification with binary cross-entropy loss.
 * The final layer must use sigmoid activation (output ∈ [0,1]).
 *
 * Training loop per step:
 *   1. Forward pass through all layers
 *   2. Compute BCE loss and its gradient w.r.t. output
 *   3. Backward pass (backpropagation)
 *   4. Update weights with chosen optimizer
 */

import { Tensor } from './Tensor';
import { DenseLayer } from './Layer';
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
}

export class Network {
  layers: DenseLayer[];
  readonly config: NetworkConfig;
  epoch = 0;

  constructor(config: NetworkConfig) {
    this.config = config;
    this.layers = [];

    let prev = config.inputSize;
    for (const lc of config.layers) {
      this.layers.push(new DenseLayer(prev, lc.units, lc.activation));
      prev = lc.units;
    }
  }

  /** Full forward pass → output Tensor [m, outputSize] */
  forward(input: Tensor): Tensor {
    let x = input;
    for (const layer of this.layers) x = layer.forward(x);
    return x;
  }

  /**
   * Binary cross-entropy: L = -1/m Σ [y·log(p) + (1-y)·log(1-p)]
   * Gradient w.r.t. p:   dL/dp = (p - y) / (m · p(1-p))
   * Simplified gradient when last layer is sigmoid: dL/dZ_last = (p - y) / m
   */
  private bce(pred: Tensor, target: Tensor): [number, Tensor] {
    const m = pred.rows;
    const ε = 1e-7;
    let loss = 0;
    const dLoss = new Tensor(m, 1);

    for (let i = 0; i < m; i++) {
      const p = Math.min(Math.max(pred.get(i, 0), ε), 1 - ε);
      const y = target.get(i, 0);
      loss += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
      // Simplified gradient: works because last activation is sigmoid
      dLoss.set(i, 0, (p - y) / m);
    }
    return [loss / m, dLoss];
  }

  /** One complete train step. Returns { loss, accuracy }. */
  step(xBatch: Tensor, yBatch: Tensor): { loss: number; accuracy: number } {
    // 1. Forward
    const pred = this.forward(xBatch);

    // 2. Loss
    const [loss, dLoss] = this.bce(pred, yBatch);

    // 3. Backward
    let grad = dLoss;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      grad = this.layers[i].backward(grad);
    }

    // 4. Update
    const lr = this.config.learningRate;
    for (const layer of this.layers) {
      if (this.config.optimizer === 'adam') layer.updateAdam(lr);
      else layer.updateSGD(lr);
    }

    this.epoch++;
    return { loss, accuracy: this.accuracy(pred, yBatch) };
  }

  accuracy(pred: Tensor, target: Tensor): number {
    let correct = 0;
    for (let i = 0; i < pred.rows; i++) {
      const p = pred.get(i, 0) >= 0.5 ? 1 : 0;
      if (p === target.get(i, 0)) correct++;
    }
    return correct / pred.rows;
  }

  /** Predict class probability for a 2D point (for decision boundary) */
  predict2D(x: number, y: number): number {
    const input = new Tensor(1, 2, [x, y]);
    return this.forward(input).get(0, 0);
  }

  /** Reset weights and Adam state */
  reset() {
    this.epoch = 0;
    const config = this.config;
    let prev = config.inputSize;
    this.layers = [];
    for (const lc of config.layers) {
      this.layers.push(new DenseLayer(prev, lc.units, lc.activation));
      prev = lc.units;
    }
  }
}
