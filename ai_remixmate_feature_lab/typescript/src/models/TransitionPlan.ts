import type { CompatibilityScore } from "./CompatibilityScore";
import type { CuePoint } from "./CuePoint";
import type { LoopState } from "./LoopState";

export interface TransitionPlan {
  id: string;
  fromTrackId: string;
  toTrackId: string;
  entryTimeSec: number;
  exitTimeSec: number;
  transitionLengthBars: number;
  compatibility: CompatibilityScore;
  eqAutomationNotes: string[];
  filterAutomationNotes: string[];
  cueSuggestions: CuePoint[];
  loopSuggestions: LoopState[];
  warnings: string[];
}
