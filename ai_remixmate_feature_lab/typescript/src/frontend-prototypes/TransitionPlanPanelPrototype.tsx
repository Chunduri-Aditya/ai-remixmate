import type { CSSProperties } from "react";
import type { TransitionPlan } from "../models/TransitionPlan";

const panelStyle: CSSProperties = {
  border: "1px solid #2a2a32",
  borderRadius: 8,
  padding: 16,
  background: "#111113",
  color: "#f5f5f7",
  fontFamily: "Inter, system-ui, sans-serif"
};

export interface TransitionPlanPanelPrototypeProps {
  plan: TransitionPlan;
}

export function TransitionPlanPanelPrototype({ plan }: TransitionPlanPanelPrototypeProps) {
  // Future integration: map plan items onto the existing transition timeline component.
  return (
    <section style={panelStyle}>
      <h3>Transition Plan</h3>
      <p>{plan.fromTrackId} to {plan.toTrackId}</p>
      <p>Exit {plan.exitTimeSec.toFixed(2)}s | Entry {plan.entryTimeSec.toFixed(2)}s | {plan.transitionLengthBars} bars</p>
      <h4>Automation</h4>
      {[...plan.eqAutomationNotes, ...plan.filterAutomationNotes].map((note) => <p key={note}>{note}</p>)}
      {plan.warnings.map((warning) => <p key={warning} style={{ color: "#f87171" }}>{warning}</p>)}
    </section>
  );
}
