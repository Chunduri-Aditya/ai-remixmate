import type { NeuralFeatureModelState, NeuralFeatureSpec, TrainingEvent } from "./NeuralTypes";

function clamp(value: number, low: number, high: number): number {
  if (!Number.isFinite(value)) return low;
  return Math.max(low, Math.min(high, value));
}

function sigmoid(value: number): number {
  if (value >= 0) {
    const z = Math.exp(-value);
    return 1 / (1 + z);
  }
  const z = Math.exp(value);
  return z / (1 + z);
}

function stableSeed(text: string): number {
  let total = 17;
  [...text].forEach((char, index) => {
    total = (total * 31 + (index + 1) * char.charCodeAt(0)) % 2147483647;
  });
  return total;
}

function seededRandom(seed: number): () => number {
  let state = seed || 1;
  return () => {
    state = (state * 48271) % 2147483647;
    return state / 2147483647;
  };
}

export class TinyOnlineMlp {
  readonly spec: NeuralFeatureSpec;
  w1: number[][];
  b1: number[];
  w2: number[][];
  b2: number[];
  examplesSeen: number;
  lastLoss: number;

  constructor(spec: NeuralFeatureSpec, state?: NeuralFeatureModelState) {
    this.spec = spec;
    if (state) {
      this.w1 = state.weights.w1.map((row) => [...row]);
      this.b1 = [...state.weights.b1];
      this.w2 = state.weights.w2.map((row) => [...row]);
      this.b2 = [...state.weights.b2];
      this.examplesSeen = state.examplesSeen;
      this.lastLoss = state.lastLoss;
      return;
    }
    const rand = seededRandom(stableSeed(spec.name));
    const limit1 = 1 / Math.sqrt(spec.inputSize);
    const limit2 = 1 / Math.sqrt(spec.hiddenSize);
    this.w1 = Array.from({ length: spec.hiddenSize }, () =>
      Array.from({ length: spec.inputSize }, () => (rand() * 2 - 1) * limit1)
    );
    this.b1 = Array.from({ length: spec.hiddenSize }, () => 0);
    this.w2 = Array.from({ length: spec.outputSize }, () =>
      Array.from({ length: spec.hiddenSize }, () => (rand() * 2 - 1) * limit2)
    );
    this.b2 = Array.from({ length: spec.outputSize }, () => 0);
    this.examplesSeen = 0;
    this.lastLoss = 0;
  }

  predict(inputVector: number[]): number[] {
    return this.forward(inputVector).outputs;
  }

  trainOne(inputVector: number[], targetVector: number[], weight = 1): number {
    const x = this.input(inputVector);
    const target = this.target(targetVector);
    const eventWeight = clamp(weight, 0, 10);
    const { hidden, outputs } = this.forward(x);
    const loss = outputs.reduce((sum, output, index) => sum + (output - target[index]) ** 2, 0) / target.length;
    const outputDelta = outputs.map((output, index) => eventWeight * (output - target[index]) * output * (1 - output));
    const oldW2 = this.w2.map((row) => [...row]);

    for (let outputIndex = 0; outputIndex < this.spec.outputSize; outputIndex += 1) {
      for (let hiddenIndex = 0; hiddenIndex < this.spec.hiddenSize; hiddenIndex += 1) {
        this.w2[outputIndex][hiddenIndex] -= this.spec.learningRate * outputDelta[outputIndex] * hidden[hiddenIndex];
      }
      this.b2[outputIndex] -= this.spec.learningRate * outputDelta[outputIndex];
    }

    const hiddenDelta = Array.from({ length: this.spec.hiddenSize }, (_, hiddenIndex) => {
      const downstream = outputDelta.reduce((sum, delta, outputIndex) => sum + delta * oldW2[outputIndex][hiddenIndex], 0);
      return (1 - hidden[hiddenIndex] ** 2) * downstream;
    });

    for (let hiddenIndex = 0; hiddenIndex < this.spec.hiddenSize; hiddenIndex += 1) {
      for (let inputIndex = 0; inputIndex < this.spec.inputSize; inputIndex += 1) {
        this.w1[hiddenIndex][inputIndex] -= this.spec.learningRate * hiddenDelta[hiddenIndex] * x[inputIndex];
      }
      this.b1[hiddenIndex] -= this.spec.learningRate * hiddenDelta[hiddenIndex];
    }

    this.examplesSeen += 1;
    this.lastLoss = loss;
    return loss;
  }

  trainEvent(event: TrainingEvent): number {
    if (event.featureName !== this.spec.name) {
      throw new Error(`event is for ${event.featureName}, not ${this.spec.name}`);
    }
    return this.trainOne(event.inputVector, event.targetVector, event.weight ?? 1);
  }

  toState(): NeuralFeatureModelState {
    return {
      name: this.spec.name,
      description: this.spec.description,
      inputSize: this.spec.inputSize,
      hiddenSize: this.spec.hiddenSize,
      outputSize: this.spec.outputSize,
      learningRate: this.spec.learningRate,
      version: this.spec.version,
      examplesSeen: this.examplesSeen,
      lastLoss: this.lastLoss,
      weights: {
        w1: this.w1.map((row) => [...row]),
        b1: [...this.b1],
        w2: this.w2.map((row) => [...row]),
        b2: [...this.b2]
      }
    };
  }

  private forward(inputVector: number[]): { hidden: number[]; outputs: number[] } {
    const x = this.input(inputVector);
    const hidden = this.w1.map((row, index) => Math.tanh(row.reduce((sum, weight, inputIndex) => sum + weight * x[inputIndex], this.b1[index])));
    const outputs = this.w2.map((row, index) => sigmoid(row.reduce((sum, weight, hiddenIndex) => sum + weight * hidden[hiddenIndex], this.b2[index])));
    return { hidden, outputs };
  }

  private input(inputVector: number[]): number[] {
    if (inputVector.length !== this.spec.inputSize) {
      throw new Error(`${this.spec.name} expects ${this.spec.inputSize} inputs`);
    }
    return inputVector.map((value) => clamp(value, -1, 1));
  }

  private target(targetVector: number[]): number[] {
    if (targetVector.length !== this.spec.outputSize) {
      throw new Error(`${this.spec.name} expects ${this.spec.outputSize} targets`);
    }
    return targetVector.map((value) => clamp(value, 0, 1));
  }
}
