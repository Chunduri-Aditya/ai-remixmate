import type { CSSProperties } from "react";
import type { CompatibilityScore } from "../models/CompatibilityScore";

const panelStyle: CSSProperties = {
  border: "1px solid #2a2a32",
  borderRadius: 8,
  padding: 16,
  background: "#111113",
  color: "#f5f5f7",
  fontFamily: "Inter, system-ui, sans-serif"
};

export interface CompatibilityPanelPrototypeProps {
  score: CompatibilityScore;
}

export function CompatibilityPanelPrototype({ score }: CompatibilityPanelPrototypeProps) {
  // Future integration: connect rows to API score breakdown and recommendation actions.
  const rows = [["BPM", score.bpmScore], ["Key", score.keyScore], ["Energy", score.energyScore], ["Timbre", score.timbreScore]] as const;
  return (
    <section style={panelStyle}>
      <h3>Compatibility {(score.overall * 100).toFixed(0)}</h3>
      {rows.map(([label, value]) => <p key={label}>{label}: {(value * 100).toFixed(0)}%</p>)}
      {score.explanation.map((line) => <p key={line}>{line}</p>)}
      {score.warnings.map((line) => <p key={line} style={{ color: "#f87171" }}>{line}</p>)}
    </section>
  );
}
