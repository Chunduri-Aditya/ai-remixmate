export interface CompatibilityScore {
  overall: number;
  bpmScore: number;
  keyScore: number;
  energyScore: number;
  timbreScore: number;
  vocalClashPenalty: number;
  explanation: string[];
  warnings: string[];
}
