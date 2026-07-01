import { NEURAL_FEATURE_SPECS } from "./featureRegistry";
import type { ModelRegistryState, TrainingEvent } from "./NeuralTypes";
import { TinyOnlineMlp } from "./tinyOnlineMlp";

export class OnlineLearningController {
  readonly models: Record<string, TinyOnlineMlp>;
  eventCount: number;

  constructor(models?: Record<string, TinyOnlineMlp>) {
    this.models = models ?? Object.fromEntries(
      Object.entries(NEURAL_FEATURE_SPECS).map(([name, spec]) => [name, new TinyOnlineMlp(spec)])
    );
    this.eventCount = 0;
  }

  predict(featureName: string, inputVector: number[]): number[] {
    return this.model(featureName).predict(inputVector);
  }

  learn(event: TrainingEvent): { featureName: string; loss: number; examplesSeen: number; prediction: number[] } {
    const model = this.model(event.featureName);
    const loss = model.trainEvent(event);
    this.eventCount += 1;
    return {
      featureName: event.featureName,
      loss,
      examplesSeen: model.examplesSeen,
      prediction: model.predict(event.inputVector)
    };
  }

  toState(): ModelRegistryState {
    return {
      schemaVersion: "1.0",
      updatedAt: new Date().toISOString(),
      eventCount: this.eventCount,
      models: Object.fromEntries(Object.entries(this.models).map(([name, model]) => [name, model.toState()]))
    };
  }

  static fromState(state: ModelRegistryState): OnlineLearningController {
    const models = Object.fromEntries(
      Object.entries(state.models).map(([name, modelState]) => [
        name,
        new TinyOnlineMlp({
          name: modelState.name,
          inputSize: modelState.inputSize,
          hiddenSize: modelState.hiddenSize,
          outputSize: modelState.outputSize,
          learningRate: modelState.learningRate,
          description: modelState.description,
          version: modelState.version
        }, modelState)
      ])
    );
    const controller = new OnlineLearningController(models);
    controller.eventCount = state.eventCount ?? 0;
    return controller;
  }

  private model(featureName: string): TinyOnlineMlp {
    const model = this.models[featureName];
    if (!model) throw new Error(`unknown neural feature model: ${featureName}`);
    return model;
  }
}
