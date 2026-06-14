/**
 * Dataset generators for 2D binary classification problems.
 * All outputs are normalised to roughly [-1, 1] range.
 */

export interface Dataset {
  X: number[][];  // [n, 2]
  y: number[];    // [n] — 0 or 1
  name: string;
  description: string;
}

function shuffle<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/** Classic XOR — 4-class non-linearly separable problem */
export function makeXOR(n = 400): Dataset {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x1 = (Math.random() - 0.5) * 2;
    const x2 = (Math.random() - 0.5) * 2;
    X.push([x1, x2]);
    y.push(x1 * x2 > 0 ? 1 : 0);
  }
  return { X, y, name: 'XOR', description: 'Classic non-linearly separable XOR problem' };
}

/** Two interleaved spirals — notoriously hard for shallow networks */
export function makeSpiral(n = 300): Dataset {
  const X: number[][] = [];
  const y: number[] = [];
  const pointsPerClass = n / 2;

  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < pointsPerClass; i++) {
      const r = (i / pointsPerClass) * 0.95 + 0.05;
      const t = (i / pointsPerClass) * 4 * Math.PI + c * Math.PI + (Math.random() - 0.5) * 0.5;
      X.push([r * Math.sin(t), r * Math.cos(t)]);
      y.push(c);
    }
  }

  const combined = X.map((x, i) => ({ x, y: y[i] }));
  shuffle(combined);
  return {
    X: combined.map((d) => d.x),
    y: combined.map((d) => d.y),
    name: 'Spiral',
    description: 'Two interleaved spirals — requires deep, wide networks',
  };
}

/** Concentric circles */
export function makeCircles(n = 300, noise = 0.08): Dataset {
  const X: number[][] = [];
  const y: number[] = [];
  const pointsPerClass = n / 2;

  for (let c = 0; c < 2; c++) {
    const r = c === 0 ? 0.35 : 0.85;
    for (let i = 0; i < pointsPerClass; i++) {
      const angle = (i / pointsPerClass) * 2 * Math.PI;
      const nx = (Math.random() - 0.5) * noise * 2;
      const ny = (Math.random() - 0.5) * noise * 2;
      X.push([r * Math.cos(angle) + nx, r * Math.sin(angle) + ny]);
      y.push(c);
    }
  }

  const combined = X.map((x, i) => ({ x, y: y[i] }));
  shuffle(combined);
  return {
    X: combined.map((d) => d.x),
    y: combined.map((d) => d.y),
    name: 'Circles',
    description: 'Concentric circles — tests radial decision boundaries',
  };
}

/** Gaussian blobs — easy baseline */
export function makeBlobs(n = 300): Dataset {
  const X: number[][] = [];
  const y: number[] = [];
  const centers = [[-0.5, -0.5], [0.5, 0.5]];

  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < n / 2; i++) {
      const u1 = Math.random() + 1e-10;
      const u2 = Math.random();
      const nx = 0.18 * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const ny = 0.18 * Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
      X.push([centers[c][0] + nx, centers[c][1] + ny]);
      y.push(c);
    }
  }

  const combined = X.map((x, i) => ({ x, y: y[i] }));
  shuffle(combined);
  return {
    X: combined.map((d) => d.x),
    y: combined.map((d) => d.y),
    name: 'Blobs',
    description: 'Two Gaussian clusters — easy baseline',
  };
}

export const DATASETS = {
  spiral: makeSpiral,
  circles: makeCircles,
  xor: makeXOR,
  blobs: makeBlobs,
};

export type DatasetKey = keyof typeof DATASETS;
