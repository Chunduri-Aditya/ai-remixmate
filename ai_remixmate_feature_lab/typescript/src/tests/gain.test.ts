import { describe, expect, it } from "vitest";
import { clampDb, dbToLinear, linearToDb } from "../audio/gain";

describe("gain", () => {
  it("round trips dB and linear values", () => {
    expect(linearToDb(dbToLinear(-6))).toBeCloseTo(-6, 5);
  });

  it("handles zero and negative linear values safely", () => {
    expect(linearToDb(0)).toBe(-120);
    expect(linearToDb(-1)).toBe(-120);
  });

  it("clamps dB", () => {
    expect(clampDb(20)).toBe(6);
    expect(clampDb(-90)).toBe(-60);
  });
});
