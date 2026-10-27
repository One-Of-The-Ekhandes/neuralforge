/**
 * Tensor — Row-major 2D matrix backed by Float32Array.
 *
 * Implements all linear algebra primitives needed for deep learning
 * from scratch: matmul, transpose, broadcasting add, element-wise ops.
 * Uses Float32Array for cache-friendly memory layout and GPU-parity semantics.
 */
import { defaultRng, type Rng } from './Rng';

export class Tensor {
  readonly data: Float32Array;
  readonly rows: number;
  readonly cols: number;

  constructor(rows: number, cols: number, data?: ArrayLike<number>) {
    this.rows = rows;
    this.cols = cols;
    this.data = data ? new Float32Array(data) : new Float32Array(rows * cols);
  }

  get(r: number, c: number): number {
    return this.data[r * this.cols + c];
  }

  set(r: number, c: number, v: number): void {
    this.data[r * this.cols + c] = v;
  }

  static zeros(rows: number, cols: number): Tensor {
    return new Tensor(rows, cols);
  }

  /** Normally-distributed entries ~ N(0, scale²), drawn from a seedable RNG. */
  static randn(rows: number, cols: number, scale = 1.0, rng: Rng = defaultRng): Tensor {
    const t = new Tensor(rows, cols);
    for (let i = 0; i < t.data.length; i++) {
      t.data[i] = rng.gaussian(0, scale);
    }
    return t;
  }

  /** Matrix multiply: [m,k] @ [k,n] → [m,n] */
  static matmul(a: Tensor, b: Tensor): Tensor {
    if (a.cols !== b.rows) {
      throw new Error(`Shape mismatch: [${a.rows},${a.cols}] @ [${b.rows},${b.cols}]`);
    }
    const c = new Tensor(a.rows, b.cols);
    for (let i = 0; i < a.rows; i++) {
      for (let k = 0; k < a.cols; k++) {
        const aik = a.data[i * a.cols + k];
        for (let j = 0; j < b.cols; j++) {
          c.data[i * c.cols + j] += aik * b.data[k * b.cols + j];
        }
      }
    }
    return c;
  }

  /** Transpose: [m,n] → [n,m] */
  static transpose(a: Tensor): Tensor {
    const t = new Tensor(a.cols, a.rows);
    for (let i = 0; i < a.rows; i++) {
      for (let j = 0; j < a.cols; j++) {
        t.data[j * t.cols + i] = a.data[i * a.cols + j];
      }
    }
    return t;
  }

  /** Add with broadcasting: b can be [1,cols] applied to each row of a */
  static add(a: Tensor, b: Tensor): Tensor {
    const c = new Tensor(a.rows, a.cols);
    if (b.rows === 1 && a.rows > 1) {
      for (let i = 0; i < a.rows; i++) {
        for (let j = 0; j < a.cols; j++) {
          c.data[i * a.cols + j] = a.data[i * a.cols + j] + b.data[j];
        }
      }
    } else {
      for (let i = 0; i < a.data.length; i++) {
        c.data[i] = a.data[i] + b.data[i];
      }
    }
    return c;
  }

  static subtract(a: Tensor, b: Tensor): Tensor {
    const c = new Tensor(a.rows, a.cols);
    for (let i = 0; i < a.data.length; i++) c.data[i] = a.data[i] - b.data[i];
    return c;
  }

  /** Element-wise multiply (Hadamard product) */
  static hadamard(a: Tensor, b: Tensor): Tensor {
    const c = new Tensor(a.rows, a.cols);
    for (let i = 0; i < a.data.length; i++) c.data[i] = a.data[i] * b.data[i];
    return c;
  }

  static scale(a: Tensor, s: number): Tensor {
    const c = new Tensor(a.rows, a.cols);
    for (let i = 0; i < a.data.length; i++) c.data[i] = a.data[i] * s;
    return c;
  }

  static map(a: Tensor, fn: (v: number) => number): Tensor {
    const c = new Tensor(a.rows, a.cols);
    for (let i = 0; i < a.data.length; i++) c.data[i] = fn(a.data[i]);
    return c;
  }

  /** Sum each column → [1, cols]. Used for bias gradients. */
  static colSum(a: Tensor): Tensor {
    const result = new Tensor(1, a.cols);
    for (let i = 0; i < a.rows; i++) {
      for (let j = 0; j < a.cols; j++) {
        result.data[j] += a.data[i * a.cols + j];
      }
    }
    return result;
  }

  clone(): Tensor {
    return new Tensor(this.rows, this.cols, this.data);
  }

  toArray2D(): number[][] {
    const out: number[][] = [];
    for (let i = 0; i < this.rows; i++) {
      out.push(Array.from(this.data.slice(i * this.cols, (i + 1) * this.cols)));
    }
    return out;
  }
}
