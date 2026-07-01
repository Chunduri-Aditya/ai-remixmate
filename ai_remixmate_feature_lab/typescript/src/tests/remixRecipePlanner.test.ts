import { describe, expect, it } from "vitest";
import { remixRecipePlanner } from "../intelligence/remixRecipePlanner";

const beatgrid = { bpm: 128, beatTimes: [0, 0.46875, 0.9375, 1.40625, 1.875, 2.34375, 2.8125, 3.28125], downbeats: [0, 1.875], beatsPerBar: 4, confidence: 0.95 };
const trackA = { id: "a", title: "A", durationSec: 240, bpm: 128, camelot: "8A", beatgrid, energyCurve: [0.2, 0.4, 0.7], timbreVector: [1, 0, 0], vocalActivity: [0, 0.2, 0.4] };
const trackB = { id: "b", title: "B", durationSec: 220, bpm: 64, camelot: "8B", beatgrid: { ...beatgrid, bpm: 64 }, energyCurve: [0.25, 0.45, 0.65], timbreVector: [0.9, 0.1, 0], vocalActivity: [0, 0.1, 0.2] };
const trackC = { id: "c", title: "C", durationSec: 210, bpm: 130, camelot: "9A", beatgrid: { ...beatgrid, bpm: 130 }, energyCurve: [0.5, 0.6, 0.8], timbreVector: [0.4, 0.6, 0.1], vocalActivity: [0.8, 0.7, 0.1] };


describe("remixRecipePlanner", () => {
  it("returns ordered recipe steps", () => {
    const recipe = remixRecipePlanner(trackA, trackB);
    expect(recipe.steps.length).toBeGreaterThan(3);
    expect(recipe.steps.map((step) => step.order)).toEqual([1, 2, 3, 4, 5]);
  });
});
