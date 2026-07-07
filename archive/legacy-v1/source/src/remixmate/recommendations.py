"""
Track analysis and song recommendations system.

This module provides:
- Track characteristic analysis (BPM, key, genre, energy, danceability)
- Song compatibility scoring
- Recommendation generation
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard library
import os
import json
import logging

# Third-party
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
from .dj_mixing import detect_bpm, estimate_key, are_keys_compatible
from .ml_audio_features import (
    extract_deep_features,
    classify_genre,
    predict_energy,
    predict_danceability
)

# ============================================================================
# CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models",
    "song_embeddings.json"
)

# Compatibility scoring weights
BPM_WEIGHT = 0.40
KEY_WEIGHT = 0.30
GENRE_WEIGHT = 0.20
MFCC_WEIGHT = 0.10

# ============================================================================
# TRACK ANALYSIS
# ============================================================================

def analyze_track_characteristics(file_path, use_ml=True):
    """
    Analyze track and return characteristics.
    
    Args:
        file_path: Path to audio file
        use_ml: Whether to use ML models (default: True)
    
    Returns:
        Dict with keys: bpm, key, energy, danceability, genre,
        genre_confidence, tempo_category
    """
    try:
        # Load audio (first 60 seconds for speed)
        y, sr = librosa.load(file_path, sr=44100, duration=60)
        
        # Basic features
        bpm = detect_bpm(y, sr)
        key = estimate_key(y, sr)
        
        # ML features
        if use_ml:
            genre, genre_confidence = classify_genre(file_path, use_ml=True)
            energy = predict_energy(file_path)
            danceability = predict_danceability(file_path)
        else:
            # Rule-based fallback
            genre, genre_confidence = classify_genre(file_path, use_ml=False)
            energy = 0.5
            danceability = 0.5
        
        # Tempo category
        if bpm < 100:
            tempo_category = "slow"
        elif bpm < 120:
            tempo_category = "medium"
        else:
            tempo_category = "fast"
        
        return {
            "bpm": bpm,
            "key": key,
            "energy": energy,
            "danceability": danceability,
            "genre": genre,
            "genre_confidence": genre_confidence,
            "tempo_category": tempo_category
        }
    except Exception as e:
        logger.error(f"Track analysis failed: {e}")
        return {
            "bpm": 128.0,
            "key": "8A",
            "energy": 0.5,
            "danceability": 0.5,
            "genre": "pop",
            "genre_confidence": 0.5,
            "tempo_category": "medium"
        }

def analyze_track_for_display(file_path):
    """
    Analyze track and format for UI display.
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Formatted markdown string with track analysis
    """
    analysis = analyze_track_characteristics(file_path)
    
    return f"""
**Track Analysis:**
- **BPM**: {analysis['bpm']:.1f} BPM
- **Key/Scale**: {analysis['key']}
- **Genre**: {analysis['genre'].title()} ({analysis['genre_confidence']*100:.0f}% confidence)
- **Energy**: {analysis['energy']*100:.0f}%
- **Tempo Category**: {analysis['tempo_category'].title()}
- **Danceability**: {analysis['danceability']*100:.0f}%
"""

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

def _calculate_bpm_score(bpm1, bpm2):
    """Calculate BPM compatibility score."""
    bpm_diff = abs(bpm1 - bpm2)
    # Perfect match = 1.0, 50 BPM diff = 0.0
    return max(0, 1.0 - (bpm_diff / 50.0))

def _calculate_key_score(key1, key2):
    """Calculate key compatibility score."""
    return 1.0 if are_keys_compatible(key1, key2) else 0.5

def _calculate_genre_score(genre1, genre2):
    """Calculate genre similarity score."""
    return 1.0 if genre1 == genre2 else 0.5

def _calculate_mfcc_score(mfcc1, mfcc2):
    """Calculate MFCC similarity score."""
    if len(mfcc2) != 13:
        return 0.5
    
    try:
        similarity = cosine_similarity([mfcc1], [mfcc2])[0][0]
        return float(similarity)
    except Exception:
        return 0.5

def _generate_reasons(track1_features, song_data, bpm_score, key_score, genre_score):
    """Generate human-readable reasons for compatibility."""
    reasons = []
    
    # BPM reasons
    bpm1 = track1_features['bpm']
    bpm2 = song_data.get('tempo', 128.0)
    bpm_diff = abs(bpm1 - bpm2)
    
    if bpm_diff < 5:
        reasons.append("✓ Similar BPM (perfect match)")
    elif bpm_diff < 15:
        reasons.append("✓ Close BPM (good match)")
    else:
        reasons.append("⚠ Moderate BPM difference (beatmatching recommended)")
    
    # Key reasons
    if key_score == 1.0:
        reasons.append("✓ Compatible keys (harmonic match)")
    else:
        reasons.append("⚠ Key compatibility check recommended")
    
    # Genre reasons
    if genre_score == 1.0:
        genre1 = track1_features['genre']
        reasons.append(f"✓ Same genre ({genre1})")
    
    return reasons

def find_compatible_songs(track1_path, top_k=5):
    """
    Find compatible songs from database.
    
    Uses multi-factor compatibility scoring:
    - BPM similarity (40%)
    - Key compatibility (30%)
    - Genre match (20%)
    - MFCC similarity (10%)
    
    Args:
        track1_path: Path to track 1
        top_k: Number of recommendations (default: 5)
    
    Returns:
        List of (song_name, score, reasons) tuples, sorted by score
    """
    if not os.path.exists(DB_PATH):
        logger.warning("Song database not found")
        return []
    
    try:
        # Load database
        with open(DB_PATH, 'r') as f:
            db = json.load(f)
        
        if not db:
            return []
        
        # Analyze track 1
        track1_features = analyze_track_characteristics(track1_path)
        track1_deep = extract_deep_features(track1_path)
        
        if track1_deep is None:
            return []
        
        # Calculate compatibility scores
        scores = []
        
        for song_name, song_data in db.items():
            # BPM similarity (40% weight)
            bpm1 = track1_features['bpm']
            bpm2 = song_data.get('tempo', 128.0)
            bpm_score = _calculate_bpm_score(bpm1, bpm2)
            
            # Key compatibility (30% weight)
            key1 = track1_features['key']
            key2 = song_data.get('key', '8A')
            key_score = _calculate_key_score(key1, key2)
            
            # Genre similarity (20% weight)
            genre1 = track1_features['genre']
            genre2 = song_data.get('genre', 'pop')
            genre_score = _calculate_genre_score(genre1, genre2)
            
            # MFCC similarity (10% weight)
            mfcc1 = np.array(track1_deep['mfcc_mean'])
            mfcc2 = np.array(song_data.get('mfcc', [0] * 13))
            mfcc_score = _calculate_mfcc_score(mfcc1, mfcc2)
            
            # Weighted total score
            total_score = (
                BPM_WEIGHT * bpm_score +
                KEY_WEIGHT * key_score +
                GENRE_WEIGHT * genre_score +
                MFCC_WEIGHT * mfcc_score
            )
            
            # Generate reasons
            reasons = _generate_reasons(
                track1_features, song_data,
                bpm_score, key_score, genre_score
            )
            
            scores.append((song_name, total_score, reasons))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
        
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        return []

def format_recommendations(recommendations):
    """
    Format recommendations for UI display.
    
    Args:
        recommendations: List of (song_name, score, reasons) tuples
    
    Returns:
        Formatted markdown string
    """
    if not recommendations:
        return "No recommendations available. Add songs to the database first."
    
    lines = ["**🎯 Recommended Songs to Mix With:**\n"]
    
    for i, (song_name, score, reasons) in enumerate(recommendations, 1):
        lines.append(f"{i}. **{song_name}** ({score*100:.0f}% match)")
        for reason in reasons:
            lines.append(f"   - {reason}")
        lines.append("")
    
    return "\n".join(lines)
