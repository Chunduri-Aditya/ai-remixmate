import { describe, expect, it } from "vitest";
import { adjustBpmByPercent, bpmRatio, isTempoChangeSafe, tempoPercent } from "../audio/tempoMath";
import { bpmCompatibility } from "../intelligence/bpmCompatibility";

describe("tempo math", () => {
  it("normalizes half and double BPM relations", () => {
    expect(bpmRatio(128, 64)).toBeCloseTo(1, 5);
    expect(bpmCompatibility(128, 64).score).toBeGreaterThan(0.95);
  });

  it("computes percent and safety", () => {
    expect(tempoPercent(100, 105)).toBeCloseTo(5, 5);
    expect(adjustBpmByPercent(100, 5)).toBeCloseTo(105, 5);
    expect(isTempoChangeSafe(100, 120, 8)).toBe(false);
  });
});
