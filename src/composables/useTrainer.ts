/**
 * useTrainer — Reactive training loop composable.
 *
 * Drives the training loop via requestAnimationFrame so the UI never
 * blocks. Each animation frame runs `stepsPerFrame` gradient descent
 * steps, then signals Vue to update the reactive state.
 *
 * Architecture: composable owns the Network instance and exposes
 * reactive snapshots (loss history, accuracy, weight snapshots) that
 * drive all visualizations.
 */

import { ref, computed, shallowRef, type ShallowRef } from 'vue';
import { Network, type NetworkConfig } from '../engine/Network';
import { Tensor } from '../engine/Tensor';
import { DATASETS, type DatasetKey, type Dataset } from '../datasets';

export interface TrainSnapshot {
  epoch: number;
  loss: number;
  accuracy: number;
}

export interface WeightSnapshot {
  layerIdx: number;
  inputSize: number;
  outputSize: number;
  weights: number[];
  min: number;
  max: number;
}

export function useTrainer() {
  // Reactive state
  const isTraining = ref(false);
  const history = ref<TrainSnapshot[]>([]);
  const currentLoss = ref(0);
  const currentAccuracy = ref(0);
  const epoch = ref(0);
  const stepsPerFrame = ref(5);
  const datasetKey = ref<DatasetKey>('spiral');
  const weightSnapshots = ref<WeightSnapshot[]>([]);

  // Network config (reactive for UI binding)
  const networkConfig = ref<NetworkConfig>({
    inputSize: 2,
    learningRate: 0.01,
    optimizer: 'adam',
    layers: [
      { units: 8, activation: 'relu' },
      { units: 8, activation: 'relu' },
      { units: 1, activation: 'sigmoid' },
    ],
  });

  // Non-reactive references (mutated frequently during training)
  const network = shallowRef<Network>(new Network(networkConfig.value));
  const dataset = shallowRef<Dataset>(DATASETS[datasetKey.value]());
  let X: Tensor = buildTensor(dataset.value.X);
  let y: Tensor = buildTensor(dataset.value.y.map((v) => [v]));
  let rafHandle: number | null = null;

  function buildTensor(data: number[][]): Tensor {
    const rows = data.length;
    const cols = data[0].length;
    const flat = data.flat();
    return new Tensor(rows, cols, flat);
  }

  function rebuildDataset() {
    dataset.value = DATASETS[datasetKey.value]();
    X = buildTensor(dataset.value.X);
    y = buildTensor(dataset.value.y.map((v) => [v]));
  }

  function rebuildNetwork() {
    network.value = new Network(networkConfig.value);
    history.value = [];
    epoch.value = 0;
    currentLoss.value = 0;
    currentAccuracy.value = 0;
    weightSnapshots.value = [];
  }

  function snapshotWeights() {
    weightSnapshots.value = network.value.layers.map((layer, i) => {
      const snap = layer.weightSnapshot();
      return {
        layerIdx: i,
        inputSize: layer.inputSize,
        outputSize: layer.outputSize,
        weights: snap.flat,
        min: snap.min,
        max: snap.max,
      };
    });
  }

  function trainFrame() {
    if (!isTraining.value) return;

    for (let s = 0; s < stepsPerFrame.value; s++) {
      const result = network.value.step(X, y);
      currentLoss.value = result.loss;
      currentAccuracy.value = result.accuracy;
      epoch.value = network.value.epoch;
    }

    // Record snapshot every 10 epochs
    if (epoch.value % 10 === 0) {
      history.value = [
        ...history.value,
        { epoch: epoch.value, loss: currentLoss.value, accuracy: currentAccuracy.value },
      ].slice(-200);
      snapshotWeights();
    }

    rafHandle = requestAnimationFrame(trainFrame);
  }

  function start() {
    if (isTraining.value) return;
    isTraining.value = true;
    rafHandle = requestAnimationFrame(trainFrame);
  }

  function pause() {
    isTraining.value = false;
    if (rafHandle !== null) {
      cancelAnimationFrame(rafHandle);
      rafHandle = null;
    }
  }

  function reset() {
    pause();
    rebuildNetwork();
    snapshotWeights();
  }

  function changeDataset(key: DatasetKey) {
    pause();
    datasetKey.value = key;
    rebuildDataset();
    rebuildNetwork();
    snapshotWeights();
  }

  function applyConfig() {
    pause();
    // Rebuild network with updated config
    networkConfig.value = { ...networkConfig.value };
    rebuildNetwork();
    snapshotWeights();
  }

  // Initial weight snapshot
  snapshotWeights();

  const lossHistory = computed(() => history.value.map((h) => h.loss));
  const accuracyHistory = computed(() => history.value.map((h) => h.accuracy));
  const epochHistory = computed(() => history.value.map((h) => h.epoch));

  return {
    network,
    dataset,
    isTraining,
    history,
    lossHistory,
    accuracyHistory,
    epochHistory,
    currentLoss,
    currentAccuracy,
    epoch,
    stepsPerFrame,
    datasetKey,
    networkConfig,
    weightSnapshots,
    start,
    pause,
    reset,
    changeDataset,
    applyConfig,
  };
}
