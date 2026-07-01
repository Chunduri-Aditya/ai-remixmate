import type { TrackAnalysis } from "../models/TrackAnalysis";
import type { TransitionPlan } from "../models/TransitionPlan";
import { compatibilityScore } from "./compatibilityScore";

function nearestAtOrBefore(times: number[], target: number): number {
  const valid = times.filter((t) => Number.isFinite(t) && t >= 0 && t <= target).sort((a, b) => a - b);
  return valid.length ? valid[valid.length - 1] : 0;
}

function firstDownbeat(track: TrackAnalysis): number {
  return track.beatgrid.downbeats.find((t) => t >= 0) ?? track.beatgrid.beatTimes[0] ?? 0;
}

export function transitionPlanner(from: TrackAnalysis, to: TrackAnalysis, transitionLengthBars = 16): TransitionPlan {
  const compatibility = compatibilityScore(from, to);
  const beatDur = 60 / from.bpm;
  const transitionSec = transitionLengthBars * from.beatgrid.beatsPerBar * beatDur;
  const targetExit = Math.max(0, from.durationSec - transitionSec);
  const exitTimeSec = nearestAtOrBefore(from.beatgrid.downbeats.length ? from.beatgrid.downbeats : from.beatgrid.beatTimes, targetExit);
  const entryTimeSec = firstDownbeat(to);
  const warnings = [...compatibility.warnings];
  if (from.beatgrid.confidence < 0.6 || to.beatgrid.confidence < 0.6) warnings.push("Low beatgrid confidence; verify cues manually.");
  return {
    id: `${from.id}-to-${to.id}`,
    fromTrackId: from.id,
    toTrackId: to.id,
    entryTimeSec,
    exitTimeSec,
    transitionLengthBars,
    compatibility,
    eqAutomationNotes: [
      "Start reducing outgoing low EQ before the bass swap.",
      "Bring incoming drums and bass up before vocals if stems are available.",
      "Return both EQs to neutral after the transition."
    ],
    filterAutomationNotes: [
      "Use a gentle high-pass on the incoming track until the downbeat lands.",
      "Avoid heavy filter resonance when both bass stems overlap."
    ],
    cueSuggestions: [
      { id: `${from.id}-mix-out`, trackId: from.id, timestampSec: exitTimeSec, label: "Mix out", type: "mix_out", confidence: from.beatgrid.confidence },
      { id: `${to.id}-mix-in`, trackId: to.id, timestampSec: entryTimeSec, label: "Mix in", type: "mix_in", confidence: to.beatgrid.confidence }
    ],
    loopSuggestions: [
      { active: false, startSec: exitTimeSec, endSec: exitTimeSec + Math.max(beatDur * 4, 1), lengthBeats: 4, quantized: true }
    ],
    warnings
  };
}
