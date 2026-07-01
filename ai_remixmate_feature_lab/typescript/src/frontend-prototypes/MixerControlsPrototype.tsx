import type { CSSProperties } from "react";
import type { MixerState } from "../models/MixerState";

const panelStyle: CSSProperties = {
  border: "1px solid #2a2a32",
  borderRadius: 8,
  padding: 16,
  background: "#111113",
  color: "#f5f5f7",
  fontFamily: "Inter, system-ui, sans-serif"
};

export interface MixerControlsPrototypeProps {
  mixer: MixerState;
}

export function MixerControlsPrototype({ mixer }: MixerControlsPrototypeProps) {
  // Future integration: dispatch changes to the app store and audio engine adapter.
  return (
    <section style={panelStyle}>
      <h3>Mixer</h3>
      <p>Crossfader {mixer.crossfader.toFixed(2)} | Master {mixer.masterGainDb.toFixed(1)} dB</p>
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${Math.max(1, mixer.channels.length)}, minmax(160px, 1fr))`, gap: 12 }}>
        {mixer.channels.map((channel) => (
          <article key={channel.id} style={{ border: "1px solid #303038", borderRadius: 6, padding: 12 }}>
            <strong>{channel.id}</strong>
            <p>Volume {(channel.volume * 100).toFixed(0)}% | Gain {channel.gainDb.toFixed(1)} dB</p>
            <p>EQ L {channel.eq.lowDb} / M {channel.eq.midDb} / H {channel.eq.highDb}</p>
            <p>{channel.mute ? "Muted" : "Live"} {channel.solo ? "| Solo" : ""}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
