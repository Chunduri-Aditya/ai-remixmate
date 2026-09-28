export type CuePointType = "memory" | "hot" | "load" | "mix_in" | "mix_out" | "drop" | "vocal";

export interface CuePoint {
  id: string;
  trackId: string;
  timestampSec: number;
  label: string;
  type: CuePointType;
  color?: string;
  beatIndex?: number;
  confidence?: number;
}
