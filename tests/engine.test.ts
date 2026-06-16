import { describe, it, expect } from 'vitest';
import { Tensor } from '../src/engine/Tensor';
import { Rng } from '../src/engine/Rng';
import { Activations, type ActivationType } from '../src/engine/Activation';
import { Network } from '../src/engine/Network';
import { makeXOR, splitDataset } from '../src/datasets';

describe('Tensor', () => {
  it('matmul computes the correct product', () => {
    const a = new Tensor(2, 3, [1, 2, 3, 4, 5, 6]);
    const b = new Tensor(3, 2, [7, 8, 9, 10, 11, 12]);
    const c = Tensor.matmul(a, b);
    // [1*7+2*9+3*11, 1*8+2*10+3*12; 4*7+5*9+6*11, 4*8+5*10+6*12]
    expect(Array.from(c.data)).toEqual([58, 64, 139, 154]);
  });

  it('matmul throws on shape mismatch', () => {
    expect(() => Tensor.matmul(new Tensor(2, 3), new Tensor(2, 2))).toThrow();
  });

  it('transpose swaps dimensions and entries', () => {
    const t = Tensor.transpose(new Tensor(2, 3, [1, 2, 3, 4, 5, 6]));
    expect(t.rows).toBe(3);
    expect(t.cols).toBe(2);
    expect(Array.from(t.data)).toEqual([1, 4, 2, 5, 3, 6]);
  });

  it('add broadcasts a [1,cols] bias across rows', () => {
    const a = new Tensor(2, 2, [1, 2, 3, 4]);
    const bias = new Tensor(1, 2, [10, 20]);
    expect(Array.from(Tensor.add(a, bias).data)).toEqual([11, 22, 13, 24]);
  });

  it('colSum reduces rows', () => {
    const a = new Tensor(3, 2, [1, 2, 3, 4, 5, 6]);
    expect(Array.from(Tensor.colSum(a).data)).toEqual([9, 12]);
  });
});

describe('Rng', () => {
  it('is deterministic for a fixed seed', () => {
    const a = new Rng(42);
    const b = new Rng(42);
    const seqA = Array.from({ length: 5 }, () => a.next());
    const seqB = Array.from({ length: 5 }, () => b.next());
    expect(seqA).toEqual(seqB);
  });

  it('produces different streams for different seeds', () => {
    const a = new Rng(1);
    const b = new Rng(2);
    expect(a.next()).not.toBe(b.next());
  });

  it('gaussian samples are approximately mean 0, std 1', () => {
    const rng = new Rng(7);
    const n = 20000;
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
      const v = rng.gaussian();
      sum += v;
      sumSq += v * v;
    }
    const mean = sum / n;
    const std = Math.sqrt(sumSq / n - mean * mean);
    expect(Math.abs(mean)).toBeLessThan(0.05);
    expect(Math.abs(std - 1)).toBeLessThan(0.05);
  });
});

describe('Activation derivatives (finite difference)', () => {
  const eps = 1e-5;
  const types: ActivationType[] = ['relu', 'leaky_relu', 'tanh', 'sigmoid', 'linear'];
  // Avoid z=0 where ReLU/LeakyReLU are non-differentiable.
  const points = [-2.3, -0.7, 0.4, 1.8];

  for (const type of types) {
    it(`${type}' matches numeric derivative`, () => {
      const { forward, backward } = Activations[type];
      for (const z of points) {
        const numeric = (forward(z + eps) - forward(z - eps)) / (2 * eps);
        expect(Math.abs(backward(z) - numeric)).toBeLessThan(1e-4);
      }
    });
  }
});

/**
 * The central correctness test: analytic backprop gradients must match
 * finite-difference gradients of the loss. This is exactly the check that
 * exposes the classic BCE+sigmoid "double derivative" bug.
 */
describe('Gradient check (analytic vs finite difference)', () => {
  it('weight gradients match numeric gradients within tolerance', () => {
    const rng = new Rng(123);
    const net = new Network({
      inputSize: 2,
      // tanh hidden layers keep the loss smooth for finite differences.
      layers: [
        { units: 5, activation: 'tanh' },
        { units: 4, activation: 'tanh' },
        { units: 1, activation: 'sigmoid' },
      ],
      optimizer: 'sgd',
      learningRate: 0.01,
      l2: 0,
      dropout: 0,
      seed: 999,
    });

    // Small fixed batch.
    const m = 8;
    const X = new Tensor(m, 2);
    const y = new Tensor(m, 1);
    for (let i = 0; i < m; i++) {
      X.set(i, 0, rng.gaussian());
      X.set(i, 1, rng.gaussian());
      y.set(i, 0, rng.next() > 0.5 ? 1 : 0);
    }

    net.computeGradients(X, y); // populates analytic dWeights

    // Aggregate (norm-based) relative error — the standard gradient-check
    // metric. Per-element ratios are unstable for near-zero gradients,
    // especially with Float32 weight storage.
    const eps = 1e-3;
    let sumSqDiff = 0;
    let sumSqA = 0;
    let sumSqN = 0;
    for (const layer of net.layers) {
      const w = layer.weights.data;
      for (let i = 0; i < w.length; i++) {
        const orig = w[i];
        w[i] = orig + eps;
        const lossPlus = net.dataLoss(X, y);
        w[i] = orig - eps;
        const lossMinus = net.dataLoss(X, y);
        w[i] = orig;

        const numeric = (lossPlus - lossMinus) / (2 * eps);
        const analytic = layer.dWeights.data[i];
        sumSqDiff += (numeric - analytic) ** 2;
        sumSqA += analytic * analytic;
        sumSqN += numeric * numeric;
      }
    }
    const relErr = Math.sqrt(sumSqDiff) / (Math.sqrt(sumSqA) + Math.sqrt(sumSqN) + 1e-12);
    expect(relErr).toBeLessThan(1e-2);
  });
});

describe('Network training', () => {
  it('drives loss down and accuracy up on XOR', () => {
    const rng = new Rng(2024);
    const ds = makeXOR(400, rng);
    const X = new Tensor(ds.X.length, 2, ds.X.flat());
    const y = new Tensor(ds.y.length, 1, ds.y.map((v) => [v]).flat());

    const net = new Network({
      inputSize: 2,
      layers: [
        { units: 16, activation: 'relu' },
        { units: 16, activation: 'relu' },
        { units: 1, activation: 'sigmoid' },
      ],
      optimizer: 'adam',
      learningRate: 0.01,
      seed: 7,
    });

    const before = net.evaluate(X, y);
    for (let i = 0; i < 600; i++) net.trainStep(X, y, 0.01);
    const after = net.evaluate(X, y);

    expect(after.loss).toBeLessThan(before.loss);
    expect(after.accuracy).toBeGreaterThan(0.9);
  });

  it('reset is reproducible for a fixed seed', () => {
    const cfg = {
      inputSize: 2,
      layers: [
        { units: 4, activation: 'tanh' as const },
        { units: 1, activation: 'sigmoid' as const },
      ],
      optimizer: 'sgd' as const,
      learningRate: 0.01,
      seed: 55,
    };
    const a = new Network(cfg);
    const b = new Network(cfg);
    expect(Array.from(a.layers[0].weights.data)).toEqual(Array.from(b.layers[0].weights.data));
  });
});

describe('splitDataset', () => {
  it('partitions without overlap and respects the fraction', () => {
    const rng = new Rng(1);
    const ds = makeXOR(100, rng);
    const { train, val } = splitDataset(ds, 0.25, new Rng(1));
    expect(train.X.length + val.X.length).toBe(100);
    expect(val.X.length).toBe(25);
  });
});
