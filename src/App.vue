<script setup lang="ts">
import { useTrainer } from './composables/useTrainer';
import DecisionBoundary from './components/DecisionBoundary.vue';
import NetworkGraph from './components/NetworkGraph.vue';
import TrainingChart from './components/TrainingChart.vue';
import ControlPanel from './components/ControlPanel.vue';

const trainer = useTrainer();
</script>

<template>
  <div class="app">
    <header class="header">
      <h1 class="logo">⬡ NeuralForge</h1>
      <span class="tagline">Neural network engine — built from scratch in TypeScript</span>
    </header>

    <main class="grid">
      <!-- Left: control panel -->
      <aside class="sidebar">
        <ControlPanel
          :config="trainer.networkConfig.value"
          :dataset-key="trainer.datasetKey.value"
          :is-training="trainer.isTraining.value"
          :epoch="trainer.epoch.value"
          :loss="trainer.currentLoss.value"
          :accuracy="trainer.currentAccuracy.value"
          :steps-per-frame="trainer.stepsPerFrame.value"
          @start="trainer.start()"
          @pause="trainer.pause()"
          @reset="trainer.reset()"
          @change-dataset="trainer.changeDataset"
          @apply-config="trainer.applyConfig()"
          @update:steps-per-frame="trainer.stepsPerFrame.value = $event"
        />
      </aside>

      <!-- Centre: decision boundary + chart -->
      <section class="centre">
        <DecisionBoundary
          :network="trainer.network"
          :dataset="trainer.dataset"
          :epoch="trainer.epoch.value"
        />
        <TrainingChart
          :loss-history="trainer.lossHistory.value"
          :accuracy-history="trainer.accuracyHistory.value"
          :epoch-history="trainer.epochHistory.value"
        />
      </section>

      <!-- Right: network graph -->
      <section class="right">
        <NetworkGraph
          :snapshots="trainer.weightSnapshots.value"
          :layer-configs="trainer.networkConfig.value.layers"
          :input-size="trainer.networkConfig.value.inputSize"
        />
        <div class="info-card">
          <h4>What you're watching</h4>
          <p>The <strong>decision boundary</strong> updates in real time as the network learns to separate the two classes.</p>
          <p>Each line in the <strong>network graph</strong> is a weight — blue = positive, red = negative, thickness = magnitude.</p>
          <p>The <strong>loss curve</strong> should decrease as accuracy climbs. If it diverges, lower the learning rate.</p>
        </div>
      </section>
    </main>

    <footer class="footer">
      NeuralForge — Pure TypeScript neural network engine. No TensorFlow. No PyTorch. No magic.
    </footer>
  </div>
</template>

<style>
:root {
  --bg: #0d0f18;
  --card: #141720;
  --border: #1f2333;
  --text: #e2e8f0;
  --muted: #64748b;
  --accent: #7c3aed;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', system-ui, sans-serif;
  -webkit-font-smoothing: antialiased;
}
#app { width: 100%; }
</style>

<style scoped>
.app { min-height: 100vh; display: flex; flex-direction: column; }
.header {
  display: flex;
  align-items: baseline;
  gap: 16px;
  padding: 14px 24px;
  border-bottom: 1px solid var(--border);
  background: var(--card);
  position: sticky;
  top: 0;
  z-index: 10;
}
.logo { font-size: 20px; font-weight: 700; color: #a78bfa; }
.tagline { font-size: 12px; color: var(--muted); }
.grid {
  display: grid;
  grid-template-columns: 240px 1fr 300px;
  gap: 16px;
  padding: 16px;
  flex: 1;
  align-items: start;
}
.sidebar { position: sticky; top: 66px; }
.centre { display: flex; flex-direction: column; gap: 14px; }
.right { display: flex; flex-direction: column; gap: 14px; }
.info-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.info-card h4 {
  font-size: 12px;
  font-weight: 600;
  color: #a78bfa;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.info-card p {
  font-size: 11px;
  color: var(--muted);
  line-height: 1.6;
}
.footer {
  border-top: 1px solid var(--border);
  padding: 10px 24px;
  font-size: 11px;
  color: var(--muted);
  text-align: center;
}
@media (max-width: 1100px) {
  .grid { grid-template-columns: 220px 1fr; }
  .right { display: none; }
}
@media (max-width: 700px) {
  .grid { grid-template-columns: 1fr; }
  .sidebar { position: static; }
}
</style>
