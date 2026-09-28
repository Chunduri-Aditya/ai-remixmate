import type { LoopState } from "./LoopState";

export type DeckStatus = "empty" | "loaded" | "playing" | "paused" | "cueing";

export interface DeckState {
  id: string;
  trackId?: string;
  status: DeckStatus;
  currentTimeSec: number;
  tempoPercent: number;
  pitchSemitones?: number;
  volume: number;
  keyLock?: boolean;
  syncEnabled?: boolean;
  selectedCueId?: string;
  loop?: LoopState;
}
