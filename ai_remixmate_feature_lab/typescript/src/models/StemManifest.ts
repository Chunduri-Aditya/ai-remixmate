export type StemType = "vocals" | "drums" | "bass" | "other" | "instrumental" | "full";

export interface StemFile {
  type: StemType;
  path: string;
  available: boolean;
  durationSec?: number;
  sampleRate?: number;
  channels?: number;
  quality?: number;
}

export interface StemManifest {
  trackId: string;
  sourcePath?: string;
  model?: string;
  stems: StemFile[];
}
