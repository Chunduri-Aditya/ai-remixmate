export interface BeatGrid {
  bpm: number;
  beatTimes: number[];
  downbeats: number[];
  beatsPerBar: number;
  confidence: number;
  source?: string;
}
