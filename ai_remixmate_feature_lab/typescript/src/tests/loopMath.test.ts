import { describe, expect, it } from "vitest";
import { createBeatLoop, doubleLoop, halveLoop, isValidLoop } from "../audio/loopMath";

describe("loop math", () => {
  const beats = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4];

  it("creates a valid beat loop", () => {
    const loop = createBeatLoop(0, 4, beats);
    expect(loop?.startSec).toBe(0);
    expect(loop?.endSec).toBe(2);
  });

  it("halves and doubles within boundaries", () => {
    expect(halveLoop(1 / 16)).toBe(1 / 16);
    expect(doubleLoop(32)).toBe(32);
    expect(halveLoop(4)).toBe(2);
  });

  it("validates loop times", () => {
    expect(isValidLoop(1, 2)).toBe(true);
    expect(isValidLoop(2, 1)).toBe(false);
  });
});
