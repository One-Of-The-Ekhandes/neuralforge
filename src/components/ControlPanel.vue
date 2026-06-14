<script setup lang="ts">
import type { NetworkConfig } from '../engine/Network';
import type { DatasetKey } from '../datasets';
import type { ActivationType } from '../engine/Activation';

const props = defineProps<{
  config: NetworkConfig;
  datasetKey: DatasetKey;
  isTraining: boolean;
  epoch: number;
  loss: number;
  accuracy: number;
  stepsPerFrame: number;
}>();

const emit = defineEmits<{
  start: [];
  pause: [];
  reset: [];
  changeDataset: [key: DatasetKey];
  applyConfig: [];
  'update:stepsPerFrame': [v: number];
}>();

const DATASETS: { key: DatasetKey; label: string }[] = [
  { key: 'spiral', label: '🌀 Spiral' },
  { key: 'circles', label: '⭕ Circles' },
  { key: 'xor', label: '⊕ XOR' },
  { key: 'blobs', label: '🔵 Blobs' },
];

const OPTIMIZERS = ['adam', 'sgd'] as const;
const ACTIVATIONS: ActivationType[] = ['relu', 'leaky_relu', 'tanh', 'sigmoid', 'linear'];

function addLayer() {
  props.config.layers.splice(props.config.layers.length - 1, 0, { units: 8, activation: 'relu' });
}

function removeLayer() {
  if (props.config.layers.length > 2) {
    props.config.layers.splice(props.config.layers.length - 2, 1);
  }
}
</script>

<template>
  <div class="panel">
    <!-- Live stats -->
    <div class="stats-row">
      <div class="stat-box">
        <span class="stat-value">{{ epoch.toLocaleString() }}</span>
        <span class="stat-label">Epoch</span>
      </div>
      <div class="stat-box">
        <span class="stat-value red">{{ loss.toFixed(4) }}</span>
        <span class="stat-label">Loss</span>
      </div>
      <div class="stat-box">
        <span class="stat-value green">{{ (accuracy * 100).toFixed(1) }}%</span>
        <span class="stat-label">Accuracy</span>
      </div>
    </div>

    <!-- Controls -->
    <div class="btn-row">
      <button class="btn primary" @click="isTraining ? emit('pause') : emit('start')">
        {{ isTraining ? '⏸ Pause' : '▶ Train' }}
      </button>
      <button class="btn" @click="emit('reset')">↺ Reset</button>
    </div>

    <hr class="divider" />

    <!-- Dataset -->
    <label class="label">Dataset</label>
    <div class="chip-row">
      <button
        v-for="ds in DATASETS"
        :key="ds.key"
        class="chip"
        :class="{ active: datasetKey === ds.key }"
        @click="emit('changeDataset', ds.key)"
      >{{ ds.label }}</button>
    </div>

    <hr class="divider" />

    <!-- Architecture editor -->
    <label class="label">Architecture</label>

    <div class="layer-list">
      <div class="layer-row fixed">
        <span class="layer-icon">📥</span>
        <span class="layer-name">Input (2D)</span>
      </div>

      <div
        v-for="(layer, i) in config.layers.slice(0, -1)"
        :key="i"
        class="layer-row"
      >
        <span class="layer-icon">⬡</span>
        <input
          v-model.number="layer.units"
          type="number" min="1" max="64"
          class="unit-input"
          title="Units"
        />
        <select v-model="layer.activation" class="act-select">
          <option v-for="a in ACTIVATIONS" :key="a" :value="a">{{ a }}</option>
        </select>
      </div>

      <div class="layer-row fixed">
        <span class="layer-icon">📤</span>
        <span class="layer-name">Output (sigmoid)</span>
      </div>
    </div>

    <div class="btn-row small">
      <button class="btn small" @click="addLayer">+ Layer</button>
      <button class="btn small" @click="removeLayer" :disabled="config.layers.length <= 2">− Layer</button>
    </div>

    <hr class="divider" />

    <!-- Hyperparams -->
    <label class="label">Learning Rate</label>
    <input
      v-model.number="config.learningRate"
      type="range" min="0.0001" max="0.1" step="0.0001"
      class="slider"
    />
    <span class="slider-val">{{ config.learningRate.toFixed(4) }}</span>

    <label class="label">Optimizer</label>
    <div class="chip-row">
      <button
        v-for="opt in OPTIMIZERS"
        :key="opt"
        class="chip"
        :class="{ active: config.optimizer === opt }"
        @click="config.optimizer = opt"
      >{{ opt.toUpperCase() }}</button>
    </div>

    <label class="label">Speed (steps/frame)</label>
    <input
      :value="stepsPerFrame"
      @input="emit('update:stepsPerFrame', Number(($event.target as HTMLInputElement).value))"
      type="range" min="1" max="50" step="1"
      class="slider"
    />
    <span class="slider-val">{{ stepsPerFrame }}</span>

    <button class="btn primary full" style="margin-top:10px" @click="emit('applyConfig')">
      Apply & Restart
    </button>
  </div>
</template>

<style scoped>
.panel {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 220px;
}
.stats-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
}
.stat-box {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 4px;
  text-align: center;
  display: flex;
  flex-direction: column;
}
.stat-value {
  font-size: 15px;
  font-weight: 700;
  color: var(--text);
}
.stat-value.red { color: #f87171; }
.stat-value.green { color: #34d399; }
.stat-label { font-size: 9px; color: var(--muted); margin-top: 2px; }
.btn-row {
  display: flex;
  gap: 6px;
}
.btn {
  flex: 1;
  padding: 7px 8px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--bg);
  color: var(--text);
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s;
}
.btn:hover { border-color: #a78bfa; }
.btn.primary { background: #7c3aed; color: white; border-color: #7c3aed; }
.btn.primary:hover { background: #6d28d9; }
.btn.small { padding: 4px 8px; font-size: 11px; flex: unset; }
.btn.full { width: 100%; }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }
.divider { border: none; border-top: 1px solid var(--border); margin: 2px 0; }
.label {
  font-size: 11px;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.chip-row { display: flex; flex-wrap: wrap; gap: 4px; }
.chip {
  padding: 3px 10px;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: var(--bg);
  color: var(--muted);
  font-size: 11px;
  cursor: pointer;
  transition: all 0.15s;
}
.chip.active {
  background: #7c3aed;
  color: white;
  border-color: #7c3aed;
}
.layer-list { display: flex; flex-direction: column; gap: 4px; }
.layer-row {
  display: flex;
  align-items: center;
  gap: 6px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 5px 8px;
}
.layer-row.fixed { opacity: 0.6; }
.layer-icon { font-size: 14px; }
.layer-name { font-size: 11px; color: var(--muted); }
.unit-input {
  width: 48px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text);
  font-size: 12px;
  padding: 2px 4px;
  text-align: center;
}
.act-select {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text);
  font-size: 11px;
  padding: 2px 4px;
  flex: 1;
}
.slider { width: 100%; accent-color: #7c3aed; }
.slider-val { font-size: 11px; color: var(--muted); font-family: monospace; }
</style>
