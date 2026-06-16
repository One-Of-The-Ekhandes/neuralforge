<script setup lang="ts">
/**
 * DecisionBoundary — Canvas-based 2D decision boundary renderer.
 *
 * Computes the network's output probability on a GRID×GRID lattice with a
 * single batched forward pass (instead of one forward call per canvas pixel),
 * renders it to a small offscreen buffer, then upscales it to the display
 * canvas. Training data points are overlaid on top.
 */
import { ref, watch, onMounted, type ShallowRef } from 'vue';
import { Tensor } from '../engine/Tensor';
import type { Network } from '../engine/Network';
import type { Dataset } from '../datasets';

const props = defineProps<{
  network: ShallowRef<Network>;
  dataset: ShallowRef<Dataset>;
  epoch: number;
}>();

const canvasRef = ref<HTMLCanvasElement | null>(null);
const GRID = 100; // probability-field resolution (GRID² forward inputs)

// Reusable input lattice: row = gy*GRID + gx, columns = [x, -y].
const gridInput = (() => {
  const t = new Tensor(GRID * GRID, 2);
  for (let gy = 0; gy < GRID; gy++) {
    for (let gx = 0; gx < GRID; gx++) {
      const x = (gx / (GRID - 1)) * 2 - 1;
      const y = (gy / (GRID - 1)) * 2 - 1;
      const idx = gy * GRID + gx;
      t.set(idx, 0, x);
      t.set(idx, 1, -y); // flip y for screen coordinates
    }
  }
  return t;
})();

function render() {
  const canvas = canvasRef.value;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;

  // One batched forward pass over the whole lattice.
  const preds = props.network.value.forward(gridInput, false);

  const field = ctx.createImageData(GRID, GRID);
  for (let i = 0; i < GRID * GRID; i++) {
    const prob = preds.get(i, 0);
    const r = prob > 0.5 ? Math.round(255 * (prob - 0.5) * 2) : 0;
    const b = prob < 0.5 ? Math.round(255 * (0.5 - prob) * 2) : 0;
    const o = i * 4;
    field.data[o] = r + 30;
    field.data[o + 1] = 30;
    field.data[o + 2] = b + 80;
    field.data[o + 3] = 255;
  }

  // Render the small field to an offscreen buffer, then upscale smoothly.
  const off = document.createElement('canvas');
  off.width = GRID;
  off.height = GRID;
  off.getContext('2d')!.putImageData(field, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.clearRect(0, 0, W, H);
  ctx.drawImage(off, 0, 0, W, H);

  // Overlay training data points.
  const ds = props.dataset.value;
  for (let i = 0; i < ds.X.length; i++) {
    const x = ((ds.X[i][0] + 1) / 2) * W;
    const y = (1 - (ds.X[i][1] + 1) / 2) * H;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fillStyle = ds.y[i] === 1 ? '#fb923c' : '#60a5fa';
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.5)';
    ctx.lineWidth = 0.5;
    ctx.stroke();
  }
}

watch(() => props.epoch, render);
watch(() => props.dataset.value, render);
onMounted(render);
</script>

<template>
  <div class="boundary-container">
    <h3 class="panel-title">Decision Boundary</h3>
    <canvas ref="canvasRef" width="280" height="280" class="boundary-canvas" />
    <div class="legend">
      <span class="dot blue"></span> Class 0 <span class="dot orange"></span> Class 1
    </div>
  </div>
</template>

<style scoped>
.boundary-container {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.panel-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
}
.boundary-canvas {
  width: 100%;
  height: auto;
  border-radius: 6px;
}
.legend {
  display: flex;
  gap: 12px;
  font-size: 11px;
  color: var(--muted);
  align-items: center;
}
.dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
}
.dot.blue {
  background: #60a5fa;
}
.dot.orange {
  background: #fb923c;
}
</style>
