<script setup lang="ts">
/**
 * TrainingChart — Dual-axis SVG chart for loss and accuracy.
 * Hand-built from scratch: no chart library used.
 */
const props = defineProps<{
  lossHistory: number[];
  accuracyHistory: number[];
  epochHistory: number[];
}>();

const W = 500;
const H = 180;
const PAD = { top: 16, right: 60, bottom: 28, left: 44 };
const CW = W - PAD.left - PAD.right;
const CH = H - PAD.top - PAD.bottom;

function toSvgPath(values: number[], minV: number, maxV: number): string {
  if (values.length < 2) return '';
  const range = maxV - minV || 1;
  return values.map((v, i) => {
    const x = PAD.left + (i / (values.length - 1)) * CW;
    const y = PAD.top + CH - ((v - minV) / range) * CH;
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
}

function yTicks(min: number, max: number, count = 4): number[] {
  const step = (max - min) / count;
  return Array.from({ length: count + 1 }, (_, i) => min + step * i);
}
</script>

<template>
  <div class="chart-container">
    <h3 class="panel-title">Training Progress</h3>
    <svg :viewBox="`0 0 ${W} ${H}`" class="chart-svg">
      <!-- Grid -->
      <line
        v-for="t in yTicks(0, 1)"
        :key="t"
        :x1="PAD.left" :x2="PAD.left + CW"
        :y1="PAD.top + CH - t * CH"
        :y2="PAD.top + CH - t * CH"
        stroke="var(--border)" stroke-width="0.5" stroke-dasharray="4 4"
      />

      <!-- Accuracy area fill -->
      <path
        v-if="accuracyHistory.length > 1"
        :d="toSvgPath(accuracyHistory, 0, 1) + ` L${PAD.left + CW},${PAD.top + CH} L${PAD.left},${PAD.top + CH} Z`"
        fill="rgba(52,211,153,0.1)"
      />

      <!-- Accuracy line -->
      <path
        v-if="accuracyHistory.length > 1"
        :d="toSvgPath(accuracyHistory, 0, 1)"
        stroke="#34d399" stroke-width="2" fill="none" stroke-linecap="round"
      />

      <!-- Loss line (scale: 0 → max loss) -->
      <path
        v-if="lossHistory.length > 1"
        :d="toSvgPath(lossHistory, 0, Math.max(...lossHistory.slice(0, 3), 1))"
        stroke="#f87171" stroke-width="2" fill="none" stroke-linecap="round" stroke-dasharray="6 2"
      />

      <!-- Y axis left: accuracy 0–1 -->
      <text
        v-for="t in yTicks(0, 1)"
        :key="'ya' + t"
        :x="PAD.left - 6"
        :y="PAD.top + CH - t * CH + 4"
        text-anchor="end" font-size="9" fill="var(--muted)"
      >{{ (t * 100).toFixed(0) }}%</text>

      <!-- X axis epoch labels -->
      <text v-if="epochHistory.length > 1"
        :x="PAD.left" :y="H - 4"
        font-size="9" fill="var(--muted)">0</text>
      <text v-if="epochHistory.length > 1"
        :x="PAD.left + CW" :y="H - 4"
        text-anchor="end" font-size="9" fill="var(--muted)">
        {{ epochHistory[epochHistory.length - 1] }}
      </text>
      <text :x="PAD.left + CW / 2" :y="H - 4"
        text-anchor="middle" font-size="9" fill="var(--muted)">Epoch</text>

      <!-- Axes -->
      <line :x1="PAD.left" :y1="PAD.top" :x2="PAD.left" :y2="PAD.top + CH" stroke="var(--border)" />
      <line :x1="PAD.left" :y1="PAD.top + CH" :x2="PAD.left + CW" :y2="PAD.top + CH" stroke="var(--border)" />
    </svg>

    <!-- Legend & live stats -->
    <div class="chart-footer">
      <span class="stat"><span class="dot red-dot"></span>Loss: {{ lossHistory[lossHistory.length - 1]?.toFixed(4) ?? '—' }}</span>
      <span class="stat"><span class="dot green-dot"></span>Accuracy: {{ accuracyHistory.length ? (accuracyHistory[accuracyHistory.length - 1] * 100).toFixed(1) + '%' : '—' }}</span>
    </div>
  </div>
</template>

<style scoped>
.chart-container {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px;
}
.panel-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 4px;
}
.chart-svg { width: 100%; height: auto; }
.chart-footer {
  display: flex; gap: 16px; font-size: 12px; margin-top: 6px; color: var(--text);
}
.stat { display: flex; align-items: center; gap: 6px; }
.dot { width: 10px; height: 3px; border-radius: 2px; display: inline-block; }
.red-dot { background: #f87171; }
.green-dot { background: #34d399; }
</style>
