"""
Timeline planner: creates DJ-style arrangement plans with clips.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional
from .structure_detection import Section, Phrase

logger = logging.getLogger(__name__)

@dataclass
class Clip:
    """Represents a clip on the timeline."""
    source_track: str  # "A" or "B"
    stem: str  # "vocals", "drums", "bass", "other"
    start_time_src: float  # Start in original song (seconds)
    end_time_src: float  # End in original song (seconds)
    start_time_out: float  # Where to place in output (seconds)
    gain: float = 1.0
    fade_in: float = 0.0
    fade_out: float = 0.0
    effects: dict = None
    
    def __post_init__(self):
        if self.effects is None:
            self.effects = {}

@dataclass
class ArrangementPlan:
    """Complete arrangement plan with timeline of clips."""
    clips: List[Clip]
    total_duration: float
    explanation: str = ""
    energy_curve: Optional[List[float]] = None

class TimelinePlanner:
    """Plans DJ-style arrangements."""
    
    def plan_arrangement(self, sections_a: List[Section], sections_b: List[Section],
                        beats_a, beats_b, genre="edm", mode="mashup",
                        aggressiveness=0.5, energy_shape="chill_to_peak"):
        """
        Plan arrangement based on sections and parameters.
        
        Args:
            sections_a: Sections from track A
            sections_b: Sections from track B
            beats_a: Beat grid for track A
            beats_b: Beat grid for track B
            genre: Genre hint
            mode: Remix mode
            aggressiveness: 0.0-1.0 (how much to rearrange)
            energy_shape: Target energy curve shape
        
        Returns:
            ArrangementPlan
        """
        if not sections_a or not sections_b:
            return self._fallback_plan(sections_a, sections_b)
        
        # Choose template based on genre
        if genre in ["edm", "electronic", "techno"]:
            return self._edm_standard_plan(sections_a, sections_b, mode, aggressiveness)
        elif genre in ["hip-hop", "rap"]:
            return self._hiphop_cut_plan(sections_a, sections_b, mode, aggressiveness)
        else:
            return self._edm_standard_plan(sections_a, sections_b, mode, aggressiveness)
    
    def _edm_standard_plan(self, sections_a: List[Section], sections_b: List[Section],
                          mode: str, aggressiveness: float):
        """EDM standard template: Intro(A) → Build(A) → Drop(A) → Transition → Drop(B) → Outro(B)"""
        clips = []
        current_time = 0.0
        explanation_parts = []
        
        # Find sections
        intro_a = next((s for s in sections_a if s.label == "intro"), None)
        build_a = next((s for s in sections_a if s.label == "build"), None)
        drop_a = next((s for s in sections_a if s.label == "drop"), None)
        drop_b = next((s for s in sections_b if s.label == "drop"), None)
        break_b = next((s for s in sections_b if s.label == "break"), None)
        outro_b = next((s for s in sections_b if s.label == "outro"), None)
        
        # Fallback if specific sections not found
        if not intro_a:
            intro_a = sections_a[0] if sections_a else None
        if not build_a:
            build_a = sections_a[1] if len(sections_a) > 1 else intro_a
        if not drop_a:
            drop_a = sections_a[-2] if len(sections_a) > 1 else sections_a[0] if sections_a else None
        if not drop_b:
            drop_b = sections_b[1] if len(sections_b) > 1 else sections_b[0] if sections_b else None
        if not outro_b:
            outro_b = sections_b[-1] if sections_b else None
        
        # Ensure we have both tracks
        if not sections_a or not sections_b:
            return self._fallback_plan(sections_a, sections_b)
        
        # Intro(A)
        if intro_a:
            duration = intro_a.end_time - intro_a.start_time
            for stem in ["drums", "bass", "other"]:
                clips.append(Clip(
                    "A", stem, intro_a.start_time, intro_a.end_time,
                    current_time, gain=1.0, fade_in=2.0
                ))
            current_time += duration
            explanation_parts.append(f"Intro(A) @ {intro_a.start_time:.1f}s")
        
        # Build(A)
        if build_a:
            duration = build_a.end_time - build_a.start_time
            for stem in ["drums", "bass", "other", "vocals"]:
                clips.append(Clip(
                    "A", stem, build_a.start_time, build_a.end_time,
                    current_time, gain=1.0
                ))
            
            # Start bringing in B's drums during build (last 8 bars)
            if drop_b:
                transition_start = current_time + duration * 0.5
                clips.append(Clip(
                    "B", "drums", drop_b.start_time, drop_b.start_time + 8.0,
                    transition_start, gain=0.5, fade_in=4.0,
                    effects={"hp_filter": 0.7}
                ))
                explanation_parts.append("Bringing in drums(B) during build")
            
            current_time += duration
        
        # Drop(A)
        if drop_a:
            duration = drop_a.end_time - drop_a.start_time
            for stem in ["drums", "bass", "other", "vocals"]:
                clips.append(Clip(
                    "A", stem, drop_a.start_time, drop_a.end_time,
                    current_time, gain=1.0
                ))
            
            # Bass swap transition (last 2 phrases of drop)
            if drop_b:
                bass_swap_start = current_time + duration * 0.75
                # Fade out A's bass
                clips.append(Clip(
                    "A", "bass", drop_a.start_time + (drop_a.end_time - drop_a.start_time) * 0.75,
                    drop_a.end_time, bass_swap_start, gain=1.0, fade_out=4.0
                ))
                # Fade in B's bass
                clips.append(Clip(
                    "B", "bass", drop_b.start_time, drop_b.start_time + 8.0,
                    bass_swap_start, gain=1.0, fade_in=4.0
                ))
                explanation_parts.append("Bass swap transition")
            
            current_time += duration
        
        # Drop(B) with optional vocals from A
        if drop_b:
            duration = min(drop_b.end_time - drop_b.start_time, 32.0)  # Limit to 32s
            for stem in ["drums", "bass", "other"]:
                clips.append(Clip(
                    "B", stem, drop_b.start_time, drop_b.start_time + duration,
                    current_time, gain=1.0
                ))
            
            # Overlay vocals from A if mashup mode
            if mode == "mashup" and drop_a:
                clips.append(Clip(
                    "A", "vocals", drop_a.start_time, drop_a.start_time + duration,
                    current_time, gain=1.2, effects={}
                ))
                explanation_parts.append("Vocals(A) over Drop(B) (mashup)")
            
            current_time += duration
        
        # Break(B) or Outro(B)
        if break_b:
            duration = break_b.end_time - break_b.start_time
            for stem in ["drums", "bass", "other", "vocals"]:
                clips.append(Clip(
                    "B", stem, break_b.start_time, break_b.end_time,
                    current_time, gain=0.8, fade_out=4.0
                ))
            current_time += duration
        
        if outro_b:
            duration = outro_b.end_time - outro_b.start_time
            for stem in ["drums", "bass", "other"]:
                clips.append(Clip(
                    "B", stem, outro_b.start_time, outro_b.end_time,
                    current_time, gain=0.6, fade_out=duration
                ))
            current_time += duration
        
        explanation = " | ".join(explanation_parts) if explanation_parts else "EDM standard arrangement"
        
        return ArrangementPlan(
            clips=clips,
            total_duration=current_time,
            explanation=explanation
        )
    
    def _hiphop_cut_plan(self, sections_a: List[Section], sections_b: List[Section],
                        mode: str, aggressiveness: float):
        """Hip-hop quick cut template."""
        clips = []
        current_time = 0.0
        
        # Verse(A) → Quick cut → Drop(B)
        verse_a = next((s for s in sections_a if s.label == "verse"), sections_a[0] if sections_a else None)
        drop_b = next((s for s in sections_b if s.label == "drop"), sections_b[0] if sections_b else None)
        
        if verse_a:
            duration = verse_a.end_time - verse_a.start_time
            for stem in ["drums", "bass", "other", "vocals"]:
                clips.append(Clip("A", stem, verse_a.start_time, verse_a.end_time, current_time, gain=1.0))
            current_time += duration
        
        if drop_b:
            duration = drop_b.end_time - drop_b.start_time
            for stem in ["drums", "bass", "other", "vocals"]:
                clips.append(Clip("B", stem, drop_b.start_time, drop_b.end_time, current_time, gain=1.0))
            current_time += duration
        
        return ArrangementPlan(
            clips=clips,
            total_duration=current_time,
            explanation="Hip-hop quick cut arrangement"
        )
    
    def _fallback_plan(self, sections_a: List[Section], sections_b: List[Section]):
        """Fallback plan when section detection is incomplete."""
        clips = []
        current_time = 0.0
        
        # Use first half of A, second half of B
        if sections_a:
            first_a = sections_a[0]
            mid_time_a = (first_a.start_time + first_a.end_time) / 2
            duration_a = (first_a.end_time - first_a.start_time) / 2
            
            for stem in ["drums", "bass", "other", "vocals"]:
                clips.append(Clip(
                    "A", stem, first_a.start_time, mid_time_a,
                    current_time, gain=1.0, fade_out=2.0
                ))
            current_time += duration_a
        
        if sections_b:
            last_b = sections_b[-1]
            mid_time_b = (last_b.start_time + last_b.end_time) / 2
            
            for stem in ["drums", "bass", "other", "vocals"]:
                clips.append(Clip(
                    "B", stem, mid_time_b, last_b.end_time,
                    current_time, gain=1.0, fade_in=2.0
                ))
            current_time += (last_b.end_time - mid_time_b)
        
        return ArrangementPlan(
            clips=clips,
            total_duration=current_time,
            explanation="Fallback arrangement (first half A, second half B)"
        )

