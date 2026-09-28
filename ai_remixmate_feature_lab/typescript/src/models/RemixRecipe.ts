export interface RemixRecipeStep {
  order: number;
  title: string;
  instruction: string;
  startSec?: number;
  endSec?: number;
}

export interface RemixRecipe {
  id: string;
  transitionPlanId: string;
  method: string;
  stemSuggestions: string[];
  timingSuggestion: string;
  riskWarnings: string[];
  steps: RemixRecipeStep[];
}
