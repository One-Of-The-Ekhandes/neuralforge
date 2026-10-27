/**
 * Rng — Seedable pseudo-random number generator.
 *
 * Uses the `mulberry32` algorithm: a fast, well-distributed 32-bit PRNG.
 * A fixed seed produces a fully deterministic stream, which makes training
 * runs, weight initialization and dataset generation reproducible.
 *
 * `gaussian()` draws a standard-normal sample via the Box–Muller transform,
 * reusing the uniform stream so the whole pipeline stays reproducible.
 */
export class Rng {
  private state: number;

  constructor(seed = 0x2545f491) {
    this.state = seed >>> 0;
  }

  /** Reseed the generator in place. */
  seed(s: number): void {
    this.state = s >>> 0;
  }

  /** Uniform float in [0, 1). */
  next(): number {
    this.state = (this.state + 0x6d2b79f5) | 0;
    let t = Math.imul(this.state ^ (this.state >>> 15), 1 | this.state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Standard-normal sample (Box–Muller), scaled to N(mean, std²). */
  gaussian(mean = 0, std = 1): number {
    const u1 = this.next() + 1e-10;
    const u2 = this.next();
    return mean + std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  /** In-place Fisher–Yates shuffle using this generator's stream. */
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this.next() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
}

/**
 * Process-wide default generator. Engine code uses this when no explicit
 * generator is supplied; call `defaultRng.seed(n)` to make a run reproducible.
 */
export const defaultRng = new Rng(0x9e3779b9);
