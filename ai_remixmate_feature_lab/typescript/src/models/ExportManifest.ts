import type { RemixRecipe } from "./RemixRecipe";

export interface ExportManifest {
  id: string;
  createdAt: string;
  schemaVersion: string;
  sourceTrackIds: string[];
  recipe: RemixRecipe;
  renderedFiles?: string[];
  warnings?: string[];
  metadata?: Record<string, unknown>;
}
