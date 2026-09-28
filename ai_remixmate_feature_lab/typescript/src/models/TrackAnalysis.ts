import type { BeatGrid } from "./BeatGrid";
import type { CuePoint } from "./CuePoint";
import type { StemManifest } from "./StemManifest";

export interface TrackSection {
  label: string;
  startSec: number;
  endSec: number;
  confidence?: number;
}

export interface TrackAnalysis {
  id: string;
  title: string;
  artist?: string;
  durationSec: number;
  bpm: number;
  key?: string;
  mode?: "major" | "minor" | "unknown";
  camelot?: string;
  keyConfidence?: number;
  beatgrid: BeatGrid;
  energyCurve: number[];
  timbreVector?: number[];
  vocalActivity?: number[];
  sections?: TrackSection[];
  cuePoints?: CuePoint[];
  stemManifest?: StemManifest;
}
