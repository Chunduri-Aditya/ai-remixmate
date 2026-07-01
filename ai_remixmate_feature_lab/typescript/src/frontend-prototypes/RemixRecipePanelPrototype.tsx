import type { CSSProperties } from "react";
import type { RemixRecipe } from "../models/RemixRecipe";

const panelStyle: CSSProperties = {
  border: "1px solid #2a2a32",
  borderRadius: 8,
  padding: 16,
  background: "#111113",
  color: "#f5f5f7",
  fontFamily: "Inter, system-ui, sans-serif"
};

export interface RemixRecipePanelPrototypeProps {
  recipe: RemixRecipe;
}

export function RemixRecipePanelPrototype({ recipe }: RemixRecipePanelPrototypeProps) {
  // Future integration: make steps editable and exportable through Mix Vault.
  return (
    <section style={panelStyle}>
      <h3>Remix Recipe</h3>
      <p>{recipe.method}</p>
      <ol>
        {recipe.steps.map((step) => <li key={step.order}><strong>{step.title}</strong>: {step.instruction}</li>)}
      </ol>
      {recipe.stemSuggestions.map((item) => <p key={item}>{item}</p>)}
    </section>
  );
}
