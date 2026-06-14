/**
 * Activation functions and their analytic derivatives.
 *
 * All activation functions are expressed as point-wise operations so they
 * can be applied element-wise via Tensor.map(). The derivative receives the
 * PRE-activation value z (not the post-activation), which avoids storing
 * a separate activation cache and simplifies the backward pass.
 */

export type ActivationType = 'relu' | 'sigmoid' | 'tanh' | 'linear' | 'leaky_relu';

interface ActivationFn {
  forward(z: number): number;
  backward(z: number): number;
  label: string;
}

export const Activations: Record<ActivationType, ActivationFn> = {
  relu: {
    label: 'ReLU',
    forward: (z) => Math.max(0, z),
    backward: (z) => (z > 0 ? 1 : 0),
  },
  leaky_relu: {
    label: 'Leaky ReLU',
    forward: (z) => (z > 0 ? z : 0.01 * z),
    backward: (z) => (z > 0 ? 1 : 0.01),
  },
  sigmoid: {
    label: 'Sigmoid',
    forward: (z) => 1 / (1 + Math.exp(-z)),
    backward: (z) => {
      const s = 1 / (1 + Math.exp(-z));
      return s * (1 - s);
    },
  },
  tanh: {
    label: 'Tanh',
    forward: (z) => Math.tanh(z),
    backward: (z) => 1 - Math.tanh(z) ** 2,
  },
  linear: {
    label: 'Linear',
    forward: (z) => z,
    backward: (_z) => 1,
  },
};
