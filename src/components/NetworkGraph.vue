<script setup lang="ts">
/**
 * NetworkGraph — Animated SVG visualization of the neural network.
 *
 * Draws neurons as circles and connections as lines.
 * Line colour encodes weight sign (blue=positive, red=negative).
 * Line thickness encodes weight magnitude.
 * Updates reactively when weight snapshots change.
 */
import { computed } from 'vue';
import type { WeightSnapshot } from '../composables/useTrainer';

const props = defineProps<{
  snapshots: WeightSnapshot[];
  layerConfigs: { units: number; activation: string }[];
  inputSize: number;
}>();

const W = 520;
const H = 280;
const NODE_R = 10;

interface Node { x: number; y: number; layerIdx: number; neuronIdx: number; }
interface Edge { x1: number; y1: number; x2: number; y2: number; weight: number; absW: number; }

const layers = computed(() => {
  // Build layer sizes: input + hidden/output layers
  const sizes = [props.inputSize, ...props.layerConfigs.map((l) => l.units)];
  const labels = ['Input', ...props.layerConfigs.map((l) => l.activation)];
  return sizes.map((size, i) => ({ size, label: labels[i], idx: i }));
});

const nodes = computed<Node[]>(() => {
  const result: Node[] = [];
  const totalLayers = layers.value.length;
  layers.value.forEach(({ size }, li) => {
    const x = (W / (totalLayers + 1)) * (li + 1);
    for (let ni = 0; ni < Math.min(size, 8); ni++) {
      const displaySize = Math.min(size, 8);
      const y = (H / (displaySize + 1)) * (ni + 1);
      result.push({ x, y, layerIdx: li, neuronIdx: ni });
    }
  });
  return result;
});

const edges = computed<Edge[]>(() => {
  const result: Edge[] = [];
  const snaps = props.snapshots;

  for (let li = 0; li < snaps.length; li++) {
    const snap = snaps[li];
    const fromNodes = nodes.value.filter((n) => n.layerIdx === li);
    const toNodes = nodes.value.filter((n) => n.layerIdx === li + 1);

    const fromSize = Math.min(snap.inputSize, 8);
    const toSize = Math.min(snap.outputSize, 8);
    const range = Math.max(Math.abs(snap.min), Math.abs(snap.max), 0.01);

    for (let fi = 0; fi < fromSize; fi++) {
      for (let ti = 0; ti < toSize; ti++) {
        const wIdx = fi * snap.outputSize + ti;
        const w = snap.weights[wIdx] ?? 0;
        const from = fromNodes[fi];
        const to = toNodes[ti];
        if (!from || !to) continue;
        result.push({
          x1: from.x, y1: from.y,
          x2: to.x, y2: to.y,
          weight: w,
          absW: Math.abs(w) / range,
        });
      }
    }
  }
  return result;
});

function edgeColor(w: number): string {
  const alpha = Math.min(0.9, 0.15 + Math.abs(w) * 0.75);
  return w >= 0
    ? `rgba(96,165,250,${alpha.toFixed(2)})`   // blue = positive
    : `rgba(248,113,113,${alpha.toFixed(2)})`;  // red  = negative
}

function edgeWidth(absW: number): number {
  return 0.3 + absW * 2.5;
}

function activationColor(act: string): string {
  const map: Record<string, string> = {
    relu: '#a78bfa', sigmoid: '#34d399', tanh: '#fb923c',
    leaky_relu: '#c084fc', linear: '#94a3b8', Input: '#64748b',
  };
  return map[act] ?? '#6b7280';
}
</script>

<template>
  <div class="graph-container">
    <h3 class="panel-title">Network Architecture</h3>
    <svg :viewBox="`0 0 ${W} ${H}`" class="graph-svg">
      <!-- Edges -->
      <line
        v-for="(e, i) in edges"
        :key="'e' + i"
        :x1="e.x1" :y1="e.y1"
        :x2="e.x2" :y2="e.y2"
        :stroke="edgeColor(e.weight)"
        :stroke-width="edgeWidth(e.absW)"
      />

      <!-- Nodes -->
      <g v-for="n in nodes" :key="`n${n.layerIdx}-${n.neuronIdx}`">
        <circle
          :cx="n.x" :cy="n.y" :r="NODE_R"
          :fill="'var(--card)'"
          :stroke="activationColor(layers[n.layerIdx]?.label ?? 'relu')"
          stroke-width="2"
        />
        <text
          :x="n.x" :y="n.y + 4"
          text-anchor="middle"
          font-size="7"
          fill="var(--muted)"
        >{{ n.neuronIdx + 1 }}</text>
      </g>

      <!-- Layer labels -->
      <text
        v-for="(layer, li) in layers"
        :key="'lbl' + li"
        :x="(W / (layers.length + 1)) * (li + 1)"
        :y="H - 4"
        text-anchor="middle"
        font-size="9"
        :fill="activationColor(layer.label)"
      >{{ layer.label }}</text>
    </svg>

    <!-- Legend -->
    <div class="edge-legend">
      <span><span class="edge-dot blue"></span>+weight</span>
      <span><span class="edge-dot red"></span>−weight</span>
      <span style="font-size:10px;color:var(--muted)">thickness = magnitude</span>
    </div>
  </div>
</template>

<style scoped>
.graph-container {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px;
}
.panel-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 6px;
}
.graph-svg { width: 100%; height: auto; }
.edge-legend {
  display: flex;
  gap: 14px;
  font-size: 11px;
  color: var(--muted);
  align-items: center;
  margin-top: 6px;
}
.edge-dot {
  display: inline-block;
  width: 18px; height: 3px;
  border-radius: 2px;
  vertical-align: middle;
  margin-right: 3px;
}
.edge-dot.blue { background: #60a5fa; }
.edge-dot.red { background: #f87171; }
</style>
