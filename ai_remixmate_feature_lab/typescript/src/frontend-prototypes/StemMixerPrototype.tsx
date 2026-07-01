import type { CSSProperties } from "react";
import type { StemManifest } from "../models/StemManifest";

const panelStyle: CSSProperties = {
  border: "1px solid #2a2a32",
  borderRadius: 8,
  padding: 16,
  background: "#111113",
  color: "#f5f5f7",
  fontFamily: "Inter, system-ui, sans-serif"
};

export interface StemMixerPrototypeProps {
  manifest: StemManifest;
  gains: Record<string, number>;
}

export function StemMixerPrototype({ manifest, gains }: StemMixerPrototypeProps) {
  // Future integration: route controls to stem-aware renderer automation.
  return (
    <section style={panelStyle}>
      <h3>Stem Mixer</h3>
      {manifest.stems.map((stem) => (
        <div key={stem.type} style={{ display: "grid", gridTemplateColumns: "100px 1fr 80px", gap: 8, alignItems: "center" }}>
          <strong>{stem.type}</strong>
          <div style={{ height: 8, background: "#26262d" }}><span style={{ display: "block", height: 8, width: `${Math.max(0, Math.min(1, gains[stem.type] ?? 0.75)) * 100}%`, background: stem.available ? "#34d399" : "#555" }} /></div>
          <span>{stem.available ? "ready" : "missing"}</span>
        </div>
      ))}
    </section>
  );
}
