<script setup lang="ts">
/**
 * DecisionBoundary — Canvas-based 2D decision boundary renderer.
 *
 * Draws a color grid: each pixel represents the network's output
 * probability for that (x,y) coordinate. Overlays actual training data.
 * Re-renders whenever the network or dataset changes.
 */
import { ref, watch, onMounted, type ShallowRef } from 'vue';
import type { Network } from '../engine/Network';
import type { Dataset } from '../datasets';

const props = defineProps<{
  network: ShallowRef<Network>;
  dataset: ShallowRef<Dataset>;
  epoch: number;
}>();

const canvasRef = ref<HTMLCanvasElement | null>(null);
const GRID = 80; // grid resolution — higher = more accurate, slower

function render() {
  const canvas = canvasRef.value;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;

  const imageData = ctx.createImageData(W, H);
  const net = props.network.value;

  for (let px = 0; px < W; px++) {
    for (let py = 0; py < H; py++) {
      // Map pixel → data space [-1, 1]
      const x = (px / W) * 2 - 1;
      const y = (py / H) * 2 - 1;
      const prob = net.predict2D(x, -y); // flip y for screen coords

      // Class 0 → blue, Class 1 → orange
      const r = prob > 0.5 ? Math.round(255 * (prob - 0.5) * 2) : 0;
      const g = 0;
      const b = prob < 0.5 ? Math.round(255 * (0.5 - prob) * 2) : 0;

      const idx = (py * W + px) * 4;
      imageData.data[idx] = r + 30;
      imageData.data[idx + 1] = g + 30;
      imageData.data[idx + 2] = b + 80;
      imageData.data[idx + 3] = 255;
    }
  }

  ctx.putImageData(imageData, 0, 0);

  // Draw decision boundary (p=0.5 contour) — approximate with darker zone
  // already implied by color transition

  // Overlay training data points
  const ds = props.dataset.value;
  for (let i = 0; i < ds.X.length; i++) {
    const x = (ds.X[i][0] + 1) / 2 * W;
    const y = (1 - (ds.X[i][1] + 1) / 2) * H;
    const label = ds.y[i];

    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fillStyle = label === 1 ? '#fb923c' : '#60a5fa';
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.5)';
    ctx.lineWidth = 0.5;
    ctx.stroke();
  }
}

// Re-render every 10 epochs via watch
watch(() => props.epoch, (e) => {
  if (e % 10 === 0 || e === 0) render();
});

onMounted(render);
</script>

<template>
  <div class="boundary-container">
    <h3 class="panel-title">Decision Boundary</h3>
    <canvas ref="canvasRef" width="280" height="280" class="boundary-canvas" />
    <div class="legend">
      <span class="dot blue"></span> Class 0
      <span class="dot orange"></span> Class 1
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
  image-rendering: pixelated;
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
  width: 10px; height: 10px;
  border-radius: 50%;
}
.dot.blue { background: #60a5fa; }
.dot.orange { background: #fb923c; }
</style>
