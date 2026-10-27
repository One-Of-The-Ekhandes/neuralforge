/**
 * DenseLayer — Fully connected layer with forward and backward pass.
 *
 * Math:
 *   Forward:   Z = X @ W + b,   A = σ(Z)   (optionally inverted dropout on A)
 *   Backward:  dZ = dA ⊙ σ'(Z)
 *              dW = Xᵀ @ dZ / m  (+ L2 weight decay)
 *              db = colSum(dZ) / m
 *              dX = dZ @ Wᵀ          ← passed to previous layer
 *
 * Weight initialization:
 *   He   (ReLU/LeakyReLU): scale = sqrt(2 / fan_in)
 *   Xavier (others):        scale = sqrt(1 / fan_in)
 */

import { Tensor } from './Tensor';
import { Activations, type ActivationType } from './Activation';
import { defaultRng, type Rng } from './Rng';

export class DenseLayer {
  readonly inputSize: number;
  readonly outputSize: number;
  readonly activation: ActivationType;

  weights: Tensor; // [inputSize, outputSize]
  biases: Tensor; // [1, outputSize]
  dWeights!: Tensor;
  dBiases!: Tensor;

  // Adam optimizer first/second moment estimates
  mW!: Tensor;
  vW!: Tensor;
  mB!: Tensor;
  vB!: Tensor;
  adamT = 0;

  /** Probability of dropping a unit during training (0 = disabled). */
  dropout = 0;

  private rng: Rng;
  private _input: Tensor | null = null;
  private _z: Tensor | null = null;
  private _dropMask: Tensor | null = null;

  constructor(
    inputSize: number,
    outputSize: number,
    activation: ActivationType = 'relu',
    rng: Rng = defaultRng,
  ) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.activation = activation;
    this.rng = rng;

    const scale =
      activation === 'relu' || activation === 'leaky_relu'
        ? Math.sqrt(2 / inputSize) // He
        : Math.sqrt(1 / inputSize); // Xavier

    this.weights = Tensor.randn(inputSize, outputSize, scale, rng);
    this.biases = Tensor.zeros(1, outputSize);
    this._resetGrads();
    this._resetAdam();
  }

  private _resetGrads() {
    this.dWeights = Tensor.zeros(this.inputSize, this.outputSize);
    this.dBiases = Tensor.zeros(1, this.outputSize);
  }

  private _resetAdam() {
    this.mW = Tensor.zeros(this.inputSize, this.outputSize);
    this.vW = Tensor.zeros(this.inputSize, this.outputSize);
    this.mB = Tensor.zeros(1, this.outputSize);
    this.vB = Tensor.zeros(1, this.outputSize);
  }

  /**
   * Forward pass. When `training` is true and `dropout > 0`, applies
   * inverted dropout to the activations and caches the mask for backward.
   */
  forward(input: Tensor, training = false): Tensor {
    this._input = input;
    this._z = Tensor.add(Tensor.matmul(input, this.weights), this.biases);
    let a = Tensor.map(this._z, Activations[this.activation].forward);

    if (training && this.dropout > 0) {
      const keep = 1 - this.dropout;
      const mask = new Tensor(a.rows, a.cols);
      for (let i = 0; i < mask.data.length; i++) {
        // Inverted dropout: scale kept units by 1/keep so the expected
        // activation (and hence inference) needs no rescaling.
        mask.data[i] = this.rng.next() < keep ? 1 / keep : 0;
      }
      this._dropMask = mask;
      a = Tensor.hadamard(a, mask);
    } else {
      this._dropMask = null;
    }
    return a;
  }

  /**
   * Backward pass from dA (gradient w.r.t. this layer's output).
   * Returns dX to pass upstream.
   */
  backward(dA: Tensor): Tensor {
    let dOut = dA;
    if (this._dropMask) dOut = Tensor.hadamard(dA, this._dropMask);
    const dZ = Tensor.hadamard(dOut, Tensor.map(this._z!, Activations[this.activation].backward));
    return this.backwardFromDZ(dZ);
  }

  /**
   * Backward pass when dZ (gradient w.r.t. the pre-activation) is already
   * known. Used for the output layer where the BCE + sigmoid gradient
   * simplifies exactly to dZ = (p − y) / m, so the activation derivative
   * must NOT be applied again.
   */
  backwardFromDZ(dZ: Tensor): Tensor {
    const m = this._input!.rows;
    this.dWeights = Tensor.scale(Tensor.matmul(Tensor.transpose(this._input!), dZ), 1 / m);
    this.dBiases = Tensor.scale(Tensor.colSum(dZ), 1 / m);
    return Tensor.matmul(dZ, Tensor.transpose(this.weights));
  }

  updateSGD(lr: number, l2 = 0) {
    for (let i = 0; i < this.weights.data.length; i++) {
      const g = this.dWeights.data[i] + l2 * this.weights.data[i];
      this.weights.data[i] -= lr * g;
    }
    for (let i = 0; i < this.biases.data.length; i++) {
      this.biases.data[i] -= lr * this.dBiases.data[i];
    }
  }

  updateAdam(lr: number, l2 = 0, β1 = 0.9, β2 = 0.999, ε = 1e-8) {
    this.adamT++;
    const t = this.adamT;
    const bc1 = 1 - β1 ** t;
    const bc2 = 1 - β2 ** t;

    for (let i = 0; i < this.weights.data.length; i++) {
      const g = this.dWeights.data[i] + l2 * this.weights.data[i];
      this.mW.data[i] = β1 * this.mW.data[i] + (1 - β1) * g;
      this.vW.data[i] = β2 * this.vW.data[i] + (1 - β2) * g * g;
      this.weights.data[i] -=
        (lr * (this.mW.data[i] / bc1)) / (Math.sqrt(this.vW.data[i] / bc2) + ε);
    }

    for (let i = 0; i < this.biases.data.length; i++) {
      const g = this.dBiases.data[i];
      this.mB.data[i] = β1 * this.mB.data[i] + (1 - β1) * g;
      this.vB.data[i] = β2 * this.vB.data[i] + (1 - β2) * g * g;
      this.biases.data[i] -=
        (lr * (this.mB.data[i] / bc1)) / (Math.sqrt(this.vB.data[i] / bc2) + ε);
    }
  }

  /** Snapshot of weight magnitudes for visualization */
  weightSnapshot(): { min: number; max: number; flat: number[] } {
    const flat = Array.from(this.weights.data);
    return {
      min: Math.min(...flat),
      max: Math.max(...flat),
      flat,
    };
  }
}
