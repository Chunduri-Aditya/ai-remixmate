export type TrainingEventSource =
  | "user_feedback"
  | "manual_correction"
  | "render_review"
  | "test_fixture"
  | "imported_label";

export interface NeuralFeatureSpec {
  name: string;
  inputSize: number;
  hiddenSize: number;
  outputSize: number;
  learningRate: number;
  description: string;
  version: number;
}

export interface TrainingEvent {
  id: string;
  featureName: string;
  inputVector: number[];
  targetVector: number[];
  source: TrainingEventSource;
  createdAt: string;
  weight?: number;
  metadata?: Record<string, unknown>;
}

export interface NeuralFeatureModelState {
  name: string;
  description: string;
  inputSize: number;
  hiddenSize: number;
  outputSize: number;
  learningRate: number;
  version: number;
  examplesSeen: number;
  lastLoss: number;
  weights: {
    w1: number[][];
    b1: number[];
    w2: number[][];
    b2: number[];
  };
}

export interface ModelRegistryState {
  schemaVersion: string;
  models: Record<string, NeuralFeatureModelState>;
  updatedAt?: string;
  eventCount?: number;
}
