import { describe, expect, it } from "vitest";
import { keyCompatibility } from "../intelligence/keyCompatibility";

describe("key compatibility", () => {
  it("scores same and relative keys highly", () => {
    expect(keyCompatibility("8A", "8A").score).toBe(1);
    expect(keyCompatibility("8A", "8B").score).toBeGreaterThan(0.9);
  });

  it("scores adjacent and unknown keys", () => {
    expect(keyCompatibility("8A", "9A").score).toBeGreaterThan(0.8);
    expect(keyCompatibility(undefined, "9A").score).toBe(0.5);
  });
});
