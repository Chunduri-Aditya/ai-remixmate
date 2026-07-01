import { describe, expect, it } from "vitest";
import { clampCrossfader, equalPowerCrossfade, linearCrossfade } from "../audio/crossfader";

describe("crossfader", () => {
  it("clamps extremes", () => {
    expect(clampCrossfader(-2)).toBe(-1);
    expect(clampCrossfader(2)).toBe(1);
  });

  it("handles center and extremes linearly", () => {
    expect(linearCrossfade(-1)).toEqual({ leftGain: 1, rightGain: 0 });
    expect(linearCrossfade(0)).toEqual({ leftGain: 0.5, rightGain: 0.5 });
    expect(linearCrossfade(1)).toEqual({ leftGain: 0, rightGain: 1 });
  });

  it("uses equal-power behavior at center", () => {
    const gains = equalPowerCrossfade(0);
    expect(gains.leftGain).toBeCloseTo(Math.SQRT1_2, 5);
    expect(gains.rightGain).toBeCloseTo(Math.SQRT1_2, 5);
  });
});
