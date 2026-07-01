import type { NeuralFeatureSpec } from "./NeuralTypes";

export const TRACK_VECTOR_SIZE = 16;
export const PAIR_VECTOR_SIZE = 40;

export const NEURAL_FEATURE_SPECS: Record<string, NeuralFeatureSpec> = {
  bpm_compatibility: {
    name: "bpm_compatibility",
    inputSize: PAIR_VECTOR_SIZE,
    hiddenSize: 12,
    outputSize: 1,
    learningRate: 0.05,
    version: 1,
    description: "Learns tempo blend quality from feedback."
  },
  key_compatibility: {
    name: "key_compatibility",
    inputSize: PAIR_VECTOR_SIZE,
    hiddenSize: 12,
    outputSize: 1,
    learningRate: 0.05,
    version: 1,
    description: "Learns harmonic acceptability beyond simplified Camelot rules."
  },
  energy_compatibility: {
    name: "energy_compatibility",
    inputSize: PAIR_VECTOR_SIZE,
    hiddenSize: 12,
    outputSize: 1,
    learningRate: 0.05,
    version: 1,
    description: "Learns preferred energy-flow behavior."
  },
  timbre_compatibility: {
    name: "timbre_compatibility",
    inputSize: PAIR_VECTOR_SIZE,
    hiddenSize: 12,
    outputSize: 1,
    learningRate: 0.05,
    version: 1,
    description: "Learns perceived sonic texture fit."
  },
  vocal_clash_risk: {
    name: "vocal_clash_risk",
    inputSize: PAIR_VECTOR_SIZE,
    hiddenSize: 12,
    outputSize: 1,
    learningRate: 0.05,
    version: 1,
    description: "Learns vocal overlap risk from render reviews."
  },
  compatibility_score: {
    name: "compatibility_score",
    inputSize: PAIR_VECTOR_SIZE,
    hiddenSize: 16,
    outputSize: 1,
    learningRate: 0.05,
    version: 1,
    description: "Learns overall transition approval."
  },
  transition_planning: {
    name: "transition_planning",
    inputSize: PAIR_VECTOR_SIZE,
    hiddenSize: 16,
    outputSize: 1,
    learningRate: 0.05,
    version: 1,
    description: "Learns whether generated cue timing was accepted."
  },
  remix_recipe_quality: {
    name: "remix_recipe_quality",
    inputSize: PAIR_VECTOR_SIZE,
    hiddenSize: 16,
    outputSize: 1,
    learningRate: 0.05,
    version: 1,
    description: "Learns generated recipe usefulness."
  },
  automix_next_track: {
    name: "automix_next_track",
    inputSize: PAIR_VECTOR_SIZE,
    hiddenSize: 16,
    outputSize: 1,
    learningRate: 0.05,
    version: 1,
    description: "Learns set-order approval."
  },
  track_match: {
    name: "track_match",
    inputSize: PAIR_VECTOR_SIZE,
    hiddenSize: 16,
    outputSize: 1,
    learningRate: 0.05,
    version: 1,
    description: "Learns recommendation relevance."
  },
  beatgrid_confidence: {
    name: "beatgrid_confidence",
    inputSize: TRACK_VECTOR_SIZE,
    hiddenSize: 12,
    outputSize: 1,
    learningRate: 0.05,
    version: 1,
    description: "Learns whether beatgrids need correction."
  },
  stem_quality: {
    name: "stem_quality",
    inputSize: TRACK_VECTOR_SIZE,
    hiddenSize: 12,
    outputSize: 1,
    learningRate: 0.05,
    version: 1,
    description: "Learns stem usefulness and bleed risk."
  },
  waveform_interest: {
    name: "waveform_interest",
    inputSize: TRACK_VECTOR_SIZE,
    hiddenSize: 12,
    outputSize: 1,
    learningRate: 0.05,
    version: 1,
    description: "Learns section salience for cues and loops."
  }
};

export function neuralFeatureNames(): string[] {
  return Object.keys(NEURAL_FEATURE_SPECS).sort();
}
