export function estimatePhraseIndex(beatIndex: number, phraseLengthBars = 8, beatsPerBar = 4): number {
  const phraseBeats = Math.max(1, phraseLengthBars * beatsPerBar);
  return Math.floor(Math.max(0, beatIndex) / phraseBeats);
}

export function isPhraseBoundary(beatIndex: number, phraseLengthBars = 8, beatsPerBar = 4): boolean {
  const phraseBeats = Math.max(1, phraseLengthBars * beatsPerBar);
  return Math.max(0, beatIndex) % phraseBeats === 0;
}

export function nextPhraseBoundaryBeat(beatIndex: number, phraseLengthBars = 8, beatsPerBar = 4): number {
  const phraseBeats = Math.max(1, phraseLengthBars * beatsPerBar);
  const safeIndex = Math.max(0, beatIndex);
  if (safeIndex % phraseBeats === 0) return safeIndex;
  return safeIndex + (phraseBeats - (safeIndex % phraseBeats));
}
