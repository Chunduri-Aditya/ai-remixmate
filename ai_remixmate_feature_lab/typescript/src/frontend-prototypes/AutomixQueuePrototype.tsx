import type { CSSProperties } from "react";


const panelStyle: CSSProperties = {
  border: "1px solid #2a2a32",
  borderRadius: 8,
  padding: 16,
  background: "#111113",
  color: "#f5f5f7",
  fontFamily: "Inter, system-ui, sans-serif"
};

export interface AutomixQueueItem {
  trackId: string;
  title: string;
  score?: number;
  warning?: string;
}

export interface AutomixQueuePrototypeProps {
  items: AutomixQueueItem[];
}

export function AutomixQueuePrototype({ items }: AutomixQueuePrototypeProps) {
  // Future integration: use planner output to seed Set Builder and chain remix jobs.
  return (
    <section style={panelStyle}>
      <h3>Automix Queue</h3>
      <ol>
        {items.map((item) => <li key={item.trackId}><strong>{item.title}</strong> {item.score != null ? `(${(item.score * 100).toFixed(0)}%)` : ""}{item.warning ? <span style={{ color: "#f87171" }}> - {item.warning}</span> : null}</li>)}
      </ol>
    </section>
  );
}
