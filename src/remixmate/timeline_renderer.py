"""
Timeline renderer: renders clips on timeline to final mix.
"""
import numpy as np
import logging
from .timeline_planner import Clip, ArrangementPlan
from .dj_mixing import time_stretch_audio, apply_eq_filter, create_crossfade_curve

logger = logging.getLogger(__name__)

class TimelineRenderer:
    """Renders timeline of clips into final audio mix."""
    
    def __init__(self, sr=44100):
        self.sr = sr
    
    def render_timeline(self, plan: ArrangementPlan, stems_a: dict, stems_b: dict,
                       bpm_a=128.0, bpm_b=130.0, target_bpm=128.0):
        """
        Render arrangement plan to final mix.
        
        Args:
            plan: ArrangementPlan with clips
            stems_a: Dict with stems from track A
            stems_b: Dict with stems from track B
            bpm_a: BPM of track A
            bpm_b: BPM of track B
            target_bpm: Target BPM for output
        
        Returns:
            Final mixed audio array
        """
        # Pre-allocate output buffer
        total_samples = int(plan.total_duration * self.sr)
        output = np.zeros((total_samples, 2))  # Stereo
        
        # Calculate stretch ratios
        stretch_a = target_bpm / bpm_a if bpm_a > 0 else 1.0
        stretch_b = target_bpm / bpm_b if bpm_b > 0 else 1.0
        
        clips_by_track = {"A": 0, "B": 0}
        
        # Render each clip
        for clip in plan.clips:
            try:
                # Get source stem
                stems = stems_a if clip.source_track == "A" else stems_b
                stretch = stretch_a if clip.source_track == "A" else stretch_b
                
                if clip.stem not in stems or stems[clip.stem] is None:
                    continue
                
                source_stem = stems[clip.stem]
                
                # Extract segment
                start_sample = int(clip.start_time_src * self.sr)
                end_sample = int(clip.end_time_src * self.sr)
                end_sample = min(end_sample, len(source_stem))
                
                if start_sample >= end_sample:
                    continue
                
                segment = source_stem[start_sample:end_sample]
                
                # Time-stretch if needed
                if abs(stretch - 1.0) > 0.01:
                    segment = time_stretch_audio(segment, stretch, self.sr)
                
                # Apply effects
                if clip.effects:
                    segment = self._apply_effects(segment, clip.effects)
                
                # Apply gain
                segment = segment * clip.gain
                
                # Apply fades
                if clip.fade_in > 0:
                    fade_samples = int(clip.fade_in * self.sr)
                    fade_curve = create_crossfade_curve(fade_samples, "s-curve")
                    if len(segment) >= fade_samples:
                        segment[:fade_samples] *= fade_curve
                
                if clip.fade_out > 0:
                    fade_samples = int(clip.fade_out * self.sr)
                    fade_curve = create_crossfade_curve(fade_samples, "s-curve")
                    if len(segment) >= fade_samples:
                        segment[-fade_samples:] *= (1.0 - fade_curve)
                
                # Place in output buffer
                out_start = int(clip.start_time_out * self.sr)
                out_end = out_start + len(segment)
                out_end = min(out_end, total_samples)
                
                if out_start < total_samples and out_end > out_start:
                    segment_len = out_end - out_start
                    if len(segment) >= segment_len:
                        # Ensure stereo
                        if len(segment.shape) == 1:
                            segment_stereo = np.column_stack([segment[:segment_len], segment[:segment_len]])
                        else:
                            segment_stereo = segment[:segment_len]
                        
                        output[out_start:out_end] += segment_stereo
                        clips_by_track[clip.source_track] += 1
            except Exception as e:
                logger.error(f"Error rendering clip {clip.source_track}/{clip.stem}: {e}")
                continue
        
        logger.info(f"Rendered {clips_by_track['A']} clips from track A, {clips_by_track['B']} clips from track B")
        
        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val
        
        return output
    
    def _apply_effects(self, audio, effects):
        """Apply effects to audio."""
        result = audio.copy()
        
        # High-pass filter
        if "hp_filter" in effects:
            result = apply_eq_filter(result, low_cut=effects["hp_filter"], sr=self.sr)
        
        # Low-pass filter
        if "lp_filter" in effects:
            result = apply_eq_filter(result, high_cut=effects["lp_filter"], sr=self.sr)
        
        return result

