import type { CSSProperties } from "react";
import type { CuePoint } from "../models/CuePoint";
import type { LoopState } from "../models/LoopState";

const panelStyle: CSSProperties = {
  border: "1px solid #2a2a32",
  borderRadius: 8,
  padding: 16,
  background: "#111113",
  color: "#f5f5f7",
  fontFamily: "Inter, system-ui, sans-serif"
};

export interface CueLoopPanelPrototypeProps {
  cues: CuePoint[];
  loop?: LoopState;
}

export function CueLoopPanelPrototype({ cues, loop }: CueLoopPanelPrototypeProps) {
  // Future integration: wire cue clicks to quantized deck transport actions.
  return (
    <section style={panelStyle}>
      <h3>Cues and Loop</h3>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 8 }}>
        {cues.map((cue) => <button key={cue.id} style={{ padding: 8, borderRadius: 6, border: "1px solid #444", background: cue.color ?? "#1c1c20", color: "white" }}>{cue.label}<br />{cue.timestampSec.toFixed(2)}s</button>)}
      </div>
      {loop && <p>Loop {loop.active ? "active" : "inactive"}: {loop.startSec.toFixed(2)}s to {loop.endSec.toFixed(2)}s ({loop.lengthBeats} beats)</p>}
    </section>
  );
}
