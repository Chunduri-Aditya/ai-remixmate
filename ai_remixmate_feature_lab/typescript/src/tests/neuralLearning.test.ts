import { describe, expect, it } from "vitest";
import type { TrackAnalysis } from "../models/TrackAnalysis";
import { neuralFeatureNames } from "../neural/featureRegistry";
import { OnlineLearningController } from "../neural/onlineLearning";
import { pairFeatureVector, trackFeatureVector } from "../neural/vectorizers";

const beatgrid = {
  bpm: 128,
  beatTimes: [0, 0.46875, 0.9375, 1.40625, 1.875],
  downbeats: [0, 1.875],
  beatsPerBar: 4,
  confidence: 0.95
};

function track(id: string, bpm: number, camelot: string): TrackAnalysis {
  return {
    id,
    title: id.toUpperCase(),
    durationSec: 240,
    bpm,
    camelot,
    beatgrid: { ...beatgrid, bpm },
    energyCurve: [0.2, 0.4, 0.7],
    timbreVector: [1, 0, 0],
    vocalActivity: [0, 0.1, 0.2],
    stemManifest: {
      trackId: id,
      stems: [
        { type: "vocals", path: "vocals.wav", available: true },
        { type: "drums", path: "drums.wav", available: true },
        { type: "bass", path: "bass.wav", available: true },
        { type: "other", path: "other.wav", available: false }
      ]
    }
  };
}

describe("neural learning", () => {
  it("defines adaptive feature models", () => {
    const names = neuralFeatureNames();
    expect(names).toContain("compatibility_score");
    expect(names).toContain("transition_planning");
    expect(names).toContain("beatgrid_confidence");
    expect(names.length).toBeGreaterThanOrEqual(13);
  });

  it("builds bounded vector inputs", () => {
    const a = track("a", 128, "8A");
    const b = track("b", 64, "8B");
    expect(trackFeatureVector(a)).toHaveLength(16);
    expect(pairFeatureVector(a, b)).toHaveLength(40);
    expect(pairFeatureVector(a, b).every((value) => value >= -1 && value <= 1)).toBe(true);
  });

  it("learns from explicit training events", () => {
    const controller = new OnlineLearningController();
    const inputVector = pairFeatureVector(track("a", 128, "8A"), track("b", 64, "8B"));
    const before = controller.predict("compatibility_score", inputVector)[0];
    for (let index = 0; index < 40; index += 1) {
      controller.learn({
        id: `evt-${index}`,
        featureName: "compatibility_score",
        inputVector,
        targetVector: [1],
        source: "test_fixture",
        createdAt: new Date().toISOString()
      });
    }
    const after = controller.predict("compatibility_score", inputVector)[0];
    expect(after).toBeGreaterThan(before);
    expect(after).toBeLessThanOrEqual(1);
  });

  it("round-trips registry state", () => {
    const controller = new OnlineLearningController();
    const inputVector = pairFeatureVector(track("a", 128, "8A"), track("b", 130, "9A"));
    controller.learn({
      id: "persist-1",
      featureName: "track_match",
      inputVector,
      targetVector: [0.8],
      source: "test_fixture",
      createdAt: new Date().toISOString()
    });
    const loaded = OnlineLearningController.fromState(controller.toState());
    expect(loaded.eventCount).toBe(controller.eventCount);
    expect(loaded.predict("track_match", inputVector)[0]).toBeCloseTo(controller.predict("track_match", inputVector)[0], 8);
  });
});
