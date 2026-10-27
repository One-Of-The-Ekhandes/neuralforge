/**
 * useTrainer — Reactive training-loop composable.
 *
 * Drives training via requestAnimationFrame so the UI never blocks. Each
 * frame runs `batchesPerFrame` mini-batch gradient steps; a full pass over
 * the training set counts as one epoch. The composable owns the Network and
 * exposes reactive snapshots (train/val loss & accuracy, weight snapshots)
 * that drive every visualization.
 *
 * Reproducibility: a single `seed` drives dataset generation, the train/val
 * split, mini-batch shuffling and weight initialization, so any run can be
 * reproduced exactly.
 */

import { ref, computed, shallowRef } from 'vue';
import { Network, type NetworkConfig } from '../engine/Network';
import { Tensor } from '../engine/Tensor';
import { Rng } from '../engine/Rng';
import { DATASETS, splitDataset, type DatasetKey, type Dataset } from '../datasets';

export interface TrainSnapshot {
  epoch: number;
  loss: number;
  accuracy: number;
  valLoss: number;
  valAccuracy: number;
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
  const currentValLoss = ref(0);
  const currentValAccuracy = ref(0);
  const epoch = ref(0);
  const currentLr = ref(0);

  // Training controls
  const batchesPerFrame = ref(5);
  const batchSize = ref(32);
  const valFraction = ref(0.2);
  const lrDecay = ref(1.0); // per-epoch multiplier (1.0 = constant LR)
  const datasetKey = ref<DatasetKey>('spiral');

  const weightSnapshots = ref<WeightSnapshot[]>([]);

  // Network config (reactive for UI binding)
  const networkConfig = ref<NetworkConfig>({
    inputSize: 2,
    learningRate: 0.03,
    optimizer: 'adam',
    l2: 0,
    dropout: 0,
    seed: 1234,
    layers: [
      { units: 8, activation: 'relu' },
      { units: 8, activation: 'relu' },
      { units: 1, activation: 'sigmoid' },
    ],
  });

  // Non-reactive references (mutated frequently during training)
  const network = shallowRef<Network>(new Network(networkConfig.value));
  const dataset = shallowRef<Dataset>(makeDataset());

  let trainX: number[][] = [];
  let trainY: number[] = [];
  let valXT: Tensor = new Tensor(1, 2);
  let valYT: Tensor = new Tensor(1, 1);
  let trainXT: Tensor = new Tensor(1, 2);
  let trainYT: Tensor = new Tensor(1, 1);

  let order: number[] = [];
  let cursor = 0;
  let batchRng = new Rng(networkConfig.value.seed ?? 1234);
  let rafHandle: number | null = null;

  function makeDataset(): Dataset {
    const seed = networkConfig.value.seed ?? 1234;
    return DATASETS[datasetKey.value](new Rng(seed));
  }

  function buildTensor(data: number[][]): Tensor {
    const rows = data.length;
    const cols = data[0].length;
    return new Tensor(rows, cols, data.flat());
  }

  function rebuildData() {
    const seed = networkConfig.value.seed ?? 1234;
    dataset.value = makeDataset();
    const { train, val } = splitDataset(dataset.value, valFraction.value, new Rng(seed ^ 0x55));
    trainX = train.X;
    trainY = train.y;
    trainXT = buildTensor(trainX);
    trainYT = buildTensor(trainY.map((v) => [v]));
    valXT = buildTensor(val.X);
    valYT = buildTensor(val.y.map((v) => [v]));
    order = trainX.map((_, i) => i);
    cursor = 0;
    batchRng = new Rng(seed ^ 0xa5);
  }

  function effectiveBatchSize(): number {
    const bs = batchSize.value;
    return bs <= 0 || bs > trainX.length ? trainX.length : bs;
  }

  function buildBatch(indices: number[]): { x: Tensor; y: Tensor } {
    const cols = trainX[0].length;
    const x = new Tensor(indices.length, cols);
    const y = new Tensor(indices.length, 1);
    indices.forEach((src, row) => {
      for (let c = 0; c < cols; c++) x.set(row, c, trainX[src][c]);
      y.set(row, 0, trainY[src]);
    });
    return { x, y };
  }

  function currentEffectiveLr(): number {
    const base = networkConfig.value.learningRate;
    return base * Math.pow(lrDecay.value, epoch.value);
  }

  function recordEpoch() {
    const tr = network.value.evaluate(trainXT, trainYT);
    const va = network.value.evaluate(valXT, valYT);
    currentLoss.value = tr.loss;
    currentAccuracy.value = tr.accuracy;
    currentValLoss.value = va.loss;
    currentValAccuracy.value = va.accuracy;
    history.value = [
      ...history.value,
      {
        epoch: epoch.value,
        loss: tr.loss,
        accuracy: tr.accuracy,
        valLoss: va.loss,
        valAccuracy: va.accuracy,
      },
    ].slice(-300);
    snapshotWeights();
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

  function rebuildNetwork() {
    network.value = new Network(networkConfig.value);
    history.value = [];
    epoch.value = 0;
    currentLoss.value = 0;
    currentAccuracy.value = 0;
    currentValLoss.value = 0;
    currentValAccuracy.value = 0;
    currentLr.value = networkConfig.value.learningRate;
    weightSnapshots.value = [];
  }

  function trainFrame() {
    if (!isTraining.value) return;
    const bs = effectiveBatchSize();
    const lr = currentEffectiveLr();
    currentLr.value = lr;

    for (let s = 0; s < batchesPerFrame.value; s++) {
      if (cursor === 0) batchRng.shuffle(order);
      const batchIdx = order.slice(cursor, cursor + bs);
      const { x, y } = buildBatch(batchIdx);
      const result = network.value.trainStep(x, y, lr);
      currentLoss.value = result.loss;
      currentAccuracy.value = result.accuracy;

      cursor += bs;
      if (cursor >= order.length) {
        cursor = 0;
        epoch.value += 1;
        recordEpoch();
      }
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
    rebuildData();
    rebuildNetwork();
    recordEpoch();
  }

  function changeDataset(key: DatasetKey) {
    pause();
    datasetKey.value = key;
    rebuildData();
    rebuildNetwork();
    recordEpoch();
  }

  /** Apply a new network configuration and restart training from scratch. */
  function applyConfig(config: NetworkConfig) {
    pause();
    networkConfig.value = config;
    rebuildData();
    rebuildNetwork();
    recordEpoch();
  }

  // Initial build
  rebuildData();
  rebuildNetwork();
  recordEpoch();

  const lossHistory = computed(() => history.value.map((h) => h.loss));
  const accuracyHistory = computed(() => history.value.map((h) => h.accuracy));
  const valLossHistory = computed(() => history.value.map((h) => h.valLoss));
  const valAccuracyHistory = computed(() => history.value.map((h) => h.valAccuracy));
  const epochHistory = computed(() => history.value.map((h) => h.epoch));

  return {
    network,
    dataset,
    isTraining,
    history,
    lossHistory,
    accuracyHistory,
    valLossHistory,
    valAccuracyHistory,
    epochHistory,
    currentLoss,
    currentAccuracy,
    currentValLoss,
    currentValAccuracy,
    currentLr,
    epoch,
    batchesPerFrame,
    batchSize,
    valFraction,
    lrDecay,
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
