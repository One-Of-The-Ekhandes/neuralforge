/**
 * Dataset generators for 2D binary classification problems.
 * All outputs are normalised to roughly [-1, 1] range.
 *
 * Every generator accepts an optional seeded `Rng` so datasets are
 * reproducible. When none is given, the shared `defaultRng` is used.
 */

import { defaultRng, Rng } from '../engine/Rng';

export interface Dataset {
  X: number[][]; // [n, 2]
  y: number[]; // [n] — 0 or 1
  name: string;
  description: string;
}

/** Classic XOR — non-linearly separable problem */
export function makeXOR(n = 400, rng: Rng = defaultRng): Dataset {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x1 = (rng.next() - 0.5) * 2;
    const x2 = (rng.next() - 0.5) * 2;
    X.push([x1, x2]);
    y.push(x1 * x2 > 0 ? 1 : 0);
  }
  return { X, y, name: 'XOR', description: 'Classic non-linearly separable XOR problem' };
}

/** Two interleaved spirals — notoriously hard for shallow networks */
export function makeSpiral(n = 300, rng: Rng = defaultRng): Dataset {
  const X: number[][] = [];
  const y: number[] = [];
  const pointsPerClass = n / 2;

  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < pointsPerClass; i++) {
      const r = (i / pointsPerClass) * 0.95 + 0.05;
      const t = (i / pointsPerClass) * 4 * Math.PI + c * Math.PI + (rng.next() - 0.5) * 0.5;
      X.push([r * Math.sin(t), r * Math.cos(t)]);
      y.push(c);
    }
  }
  return finalize(X, y, rng, 'Spiral', 'Two interleaved spirals — requires deep, wide networks');
}

/** Concentric circles */
export function makeCircles(n = 300, noise = 0.08, rng: Rng = defaultRng): Dataset {
  const X: number[][] = [];
  const y: number[] = [];
  const pointsPerClass = n / 2;

  for (let c = 0; c < 2; c++) {
    const r = c === 0 ? 0.35 : 0.85;
    for (let i = 0; i < pointsPerClass; i++) {
      const angle = (i / pointsPerClass) * 2 * Math.PI;
      const nx = (rng.next() - 0.5) * noise * 2;
      const ny = (rng.next() - 0.5) * noise * 2;
      X.push([r * Math.cos(angle) + nx, r * Math.sin(angle) + ny]);
      y.push(c);
    }
  }
  return finalize(X, y, rng, 'Circles', 'Concentric circles — tests radial decision boundaries');
}

/** Gaussian blobs — easy baseline */
export function makeBlobs(n = 300, rng: Rng = defaultRng): Dataset {
  const X: number[][] = [];
  const y: number[] = [];
  const centers = [
    [-0.5, -0.5],
    [0.5, 0.5],
  ];

  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < n / 2; i++) {
      X.push([centers[c][0] + rng.gaussian(0, 0.18), centers[c][1] + rng.gaussian(0, 0.18)]);
      y.push(c);
    }
  }
  return finalize(X, y, rng, 'Blobs', 'Two Gaussian clusters — easy baseline');
}

function finalize(
  X: number[][],
  y: number[],
  rng: Rng,
  name: string,
  description: string,
): Dataset {
  const combined = X.map((x, i) => ({ x, y: y[i] }));
  rng.shuffle(combined);
  return {
    X: combined.map((d) => d.x),
    y: combined.map((d) => d.y),
    name,
    description,
  };
}

/**
 * Split a dataset into train / validation subsets. The split is shuffled
 * with the supplied (seeded) generator so it is reproducible and stratum-free.
 */
export function splitDataset(
  ds: Dataset,
  valFraction = 0.2,
  rng: Rng = defaultRng,
): { train: Dataset; val: Dataset } {
  const idx = ds.X.map((_, i) => i);
  rng.shuffle(idx);
  const nVal = Math.max(1, Math.floor(ds.X.length * valFraction));
  const valIdx = new Set(idx.slice(0, nVal));

  const train: Dataset = { X: [], y: [], name: ds.name, description: ds.description };
  const val: Dataset = { X: [], y: [], name: ds.name, description: ds.description };
  for (let i = 0; i < ds.X.length; i++) {
    const target = valIdx.has(i) ? val : train;
    target.X.push(ds.X[i]);
    target.y.push(ds.y[i]);
  }
  return { train, val };
}

export const DATASETS = {
  spiral: (rng?: Rng) => makeSpiral(300, rng),
  circles: (rng?: Rng) => makeCircles(300, 0.08, rng),
  xor: (rng?: Rng) => makeXOR(400, rng),
  blobs: (rng?: Rng) => makeBlobs(300, rng),
};

export type DatasetKey = keyof typeof DATASETS;
