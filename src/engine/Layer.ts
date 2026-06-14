/**
 * DenseLayer — Fully connected layer with forward and backward pass.
 *
 * Math:
 *   Forward:   Z = X @ W + b,   A = σ(Z)
 *   Backward:  dZ = dA ⊙ σ'(Z)
 *              dW = Xᵀ @ dZ / m
 *              db = colSum(dZ) / m
 *              dX = dZ @ Wᵀ          ← passed to previous layer
 *
 * Weight initialization:
 *   He   (ReLU/LeakyReLU): scale = sqrt(2 / fan_in)
 *   Xavier (others):        scale = sqrt(1 / fan_in)
 */

import { Tensor } from './Tensor';
import { Activations, type ActivationType } from './Activation';

export class DenseLayer {
  weights: Tensor;  // [inputSize, outputSize]
  biases: Tensor;   // [1, outputSize]
  dWeights!: Tensor;
  dBiases!: Tensor;

  // Adam optimizer first/second moment estimates
  mW!: Tensor; vW!: Tensor;
  mB!: Tensor; vB!: Tensor;
  adamT = 0;

  private _input: Tensor | null = null;
  private _z: Tensor | null = null;

  constructor(
    public readonly inputSize: number,
    public readonly outputSize: number,
    public readonly activation: ActivationType = 'relu',
  ) {
    const heScale = ['relu', 'leaky_relu'].includes(activation)
      ? Math.sqrt(2 / inputSize)
      : Math.sqrt(1 / inputSize);

    this.weights = Tensor.randn(inputSize, outputSize, heScale);
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

  forward(input: Tensor): Tensor {
    this._input = input;
    this._z = Tensor.add(Tensor.matmul(input, this.weights), this.biases);
    return Tensor.map(this._z, Activations[this.activation].forward);
  }

  /** Returns dX to pass upstream */
  backward(dA: Tensor): Tensor {
    const act = Activations[this.activation];
    const dZ = Tensor.hadamard(dA, Tensor.map(this._z!, act.backward));
    const m = this._input!.rows;

    this.dWeights = Tensor.scale(
      Tensor.matmul(Tensor.transpose(this._input!), dZ),
      1 / m,
    );
    this.dBiases = Tensor.scale(Tensor.colSum(dZ), 1 / m);

    return Tensor.matmul(dZ, Tensor.transpose(this.weights));
  }

  updateSGD(lr: number) {
    for (let i = 0; i < this.weights.data.length; i++) {
      this.weights.data[i] -= lr * this.dWeights.data[i];
    }
    for (let i = 0; i < this.biases.data.length; i++) {
      this.biases.data[i] -= lr * this.dBiases.data[i];
    }
  }

  updateAdam(lr: number, β1 = 0.9, β2 = 0.999, ε = 1e-8) {
    this.adamT++;
    const t = this.adamT;
    const bc1 = 1 - β1 ** t;
    const bc2 = 1 - β2 ** t;

    for (let i = 0; i < this.weights.data.length; i++) {
      const g = this.dWeights.data[i];
      this.mW.data[i] = β1 * this.mW.data[i] + (1 - β1) * g;
      this.vW.data[i] = β2 * this.vW.data[i] + (1 - β2) * g * g;
      this.weights.data[i] -= lr * (this.mW.data[i] / bc1) / (Math.sqrt(this.vW.data[i] / bc2) + ε);
    }

    for (let i = 0; i < this.biases.data.length; i++) {
      const g = this.dBiases.data[i];
      this.mB.data[i] = β1 * this.mB.data[i] + (1 - β1) * g;
      this.vB.data[i] = β2 * this.vB.data[i] + (1 - β2) * g * g;
      this.biases.data[i] -= lr * (this.mB.data[i] / bc1) / (Math.sqrt(this.vB.data[i] / bc2) + ε);
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
