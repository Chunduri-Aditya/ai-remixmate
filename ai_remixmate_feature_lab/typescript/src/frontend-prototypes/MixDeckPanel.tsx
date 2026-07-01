import type { CSSProperties } from "react";
import type { DeckState } from "../models/DeckState";
import type { TrackAnalysis } from "../models/TrackAnalysis";

const panelStyle: CSSProperties = {
  border: "1px solid #2a2a32",
  borderRadius: 8,
  padding: 16,
  background: "#111113",
  color: "#f5f5f7",
  fontFamily: "Inter, system-ui, sans-serif"
};

export interface MixDeckPanelProps {
  deckA: DeckState;
  deckB: DeckState;
  tracksById: Record<string, TrackAnalysis>;
}

export function MixDeckPanel({ deckA, deckB, tracksById }: MixDeckPanelProps) {
  const decks = [deckA, deckB];
  // Future integration: replace local props with existing app store selectors.
  return (
    <section style={panelStyle}>
      <h3>Mix Deck</h3>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {decks.map((deck) => {
          const track = deck.trackId ? tracksById[deck.trackId] : undefined;
          return (
            <article key={deck.id} style={{ border: "1px solid #303038", borderRadius: 6, padding: 12 }}>
              <strong>Deck {deck.id}</strong>
              <p>{track?.title ?? "No track loaded"}</p>
              <p>BPM {track?.bpm?.toFixed(1) ?? "--"} | Key {track?.camelot ?? track?.key ?? "--"}</p>
              <p>Status {deck.status} | Time {deck.currentTimeSec.toFixed(1)}s | Tempo {deck.tempoPercent.toFixed(1)}%</p>
            </article>
          );
        })}
      </div>
    </section>
  );
}
