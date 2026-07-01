import type { CSSProperties } from "react";
import type { CuePoint } from "../models/CuePoint";

const panelStyle: CSSProperties = {
  border: "1px solid #2a2a32",
  borderRadius: 8,
  padding: 16,
  background: "#111113",
  color: "#f5f5f7",
  fontFamily: "Inter, system-ui, sans-serif"
};

export interface WaveformDeckPrototypeProps {
  peaks: number[];
  durationSec: number;
  playheadSec: number;
  cuePoints?: CuePoint[];
}

export function WaveformDeckPrototype({ peaks, durationSec, playheadSec, cuePoints = [] }: WaveformDeckPrototypeProps) {
  const playheadPct = durationSec > 0 ? Math.max(0, Math.min(100, (playheadSec / durationSec) * 100)) : 0;
  // Future integration: replace bars with precomputed multiband waveform tiles.
  return (
    <section style={panelStyle}>
      <h3>Waveform</h3>
      <div style={{ position: "relative", height: 96, display: "flex", alignItems: "end", gap: 2, background: "#08080b", padding: 8 }}>
        {peaks.map((peak, index) => <span key={index} style={{ width: `${100 / Math.max(1, peaks.length)}%`, height: `${Math.max(4, peak * 80)}px`, background: "#38bdf8" }} />)}
        <span style={{ position: "absolute", left: `${playheadPct}%`, top: 0, bottom: 0, width: 2, background: "#f59e0b" }} />
        {cuePoints.map((cue) => <span key={cue.id} title={cue.label} style={{ position: "absolute", left: `${durationSec > 0 ? (cue.timestampSec / durationSec) * 100 : 0}%`, top: 0, bottom: 0, width: 1, background: cue.color ?? "#a78bfa" }} />)}
      </div>
    </section>
  );
}
