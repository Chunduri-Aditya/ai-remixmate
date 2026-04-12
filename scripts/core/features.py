"""
Audio feature extraction and processing utilities.

This module provides functions for extracting audio features like tempo, chroma,
MFCC, and calculating similarities between audio files.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import librosa
from dataclasses import dataclass, asdict


# Audio processing constants
TARGET_SR = 22050  # Target sample rate for consistency
HOP_LENGTH = 512
N_MFCC = 13  # Standard MFCC count
N_CHROMA = 12  # Chroma features


@dataclass
class FeatureVector:
    """Container for audio features."""
    tempo: float
    chroma: np.ndarray  # (12,) - mean chroma features
    mfcc: np.ndarray    # (n_mfcc,) - mean MFCC features
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "tempo": float(self.tempo),
            "chroma": self.chroma.tolist() if hasattr(self.chroma, 'tolist') else list(self.chroma),
            "mfcc": self.mfcc.tolist() if hasattr(self.mfcc, 'tolist') else list(self.mfcc),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> FeatureVector:
        """Create from dictionary (JSON deserialization)."""
        return cls(
            tempo=float(data["tempo"]),
            chroma=np.array(data["chroma"], dtype=np.float32),
            mfcc=np.array(data["mfcc"], dtype=np.float32),
        )


def load_audio(file_path: Path, sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return (audio_array, sample_rate).
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate (default: TARGET_SR)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    y, sr = librosa.load(str(file_path), sr=sr, mono=True)
    return y, sr


def tempo_feature(y: np.ndarray, sr: int) -> float:
    """Extract tempo (BPM) from audio."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    return float(tempo)


def chroma_feature(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract mean chroma features (12-dimensional vector)."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    return np.mean(chroma, axis=1).astype(np.float32)


def mfcc_feature(y: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Extract mean MFCC features (n_mfcc-dimensional vector)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=HOP_LENGTH)
    return np.mean(mfcc, axis=1).astype(np.float32)


def extract_features_from_wav(file_path: Path) -> FeatureVector:
    """
    Extract all features from a WAV file.
    
    Args:
        file_path: Path to WAV file
    
    Returns:
        FeatureVector containing tempo, chroma, and MFCC features
    """
    y, sr = load_audio(file_path)
    
    tempo = tempo_feature(y, sr)
    chroma = chroma_feature(y, sr)
    mfcc = mfcc_feature(y, sr)
    
    return FeatureVector(tempo=tempo, chroma=chroma, mfcc=mfcc)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a, b: Input vectors
    
    Returns:
        Cosine similarity in range [-1, 1]
    """
    # Normalize vectors
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(a, b) / (norm_a * norm_b))


def load_embeddings() -> Dict[str, Dict]:
    """
    Load the embeddings database from JSON file.
    
    Returns:
        Dictionary mapping song names to feature dictionaries
    """
    from .paths import embeddings_path
    
    embeddings_file = embeddings_path()
    if not embeddings_file.exists():
        return {}
    
    try:
        with embeddings_file.open('r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_embeddings(embeddings: Dict[str, Dict]) -> None:
    """
    Save embeddings database to JSON file.
    
    Args:
        embeddings: Dictionary mapping song names to feature dictionaries
    """
    from .paths import embeddings_path, ensure_directories
    
    ensure_directories()
    embeddings_file = embeddings_path()
    
    with embeddings_file.open('w', encoding='utf-8') as f:
        json.dump(embeddings, f, indent=2, ensure_ascii=False)


def find_similar_songs(
    query_features: FeatureVector,
    embeddings_db: Dict[str, Dict],
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Find most similar songs based on audio features.

    Uses GPU-accelerated batch cosine similarity when available (50-100x faster
    for large libraries) with automatic CPU fallback.

    Args:
        query_features: Features of the query song
        embeddings_db: Database of song embeddings
        top_k: Number of top matches to return

    Returns:
        List of (song_name, similarity_score) tuples, sorted by similarity
    """
    query_vec = np.concatenate([query_features.chroma, query_features.mfcc])
    q_norm = np.linalg.norm(query_vec)
    if q_norm < 1e-10:
        return []
    query_vec = query_vec / q_norm

    # Build matrix of all candidate vectors
    names = []
    vecs = []
    for song_name, features_dict in embeddings_db.items():
        try:
            features = FeatureVector.from_dict(features_dict)
            candidate_vec = np.concatenate([features.chroma, features.mfcc])
            c_norm = np.linalg.norm(candidate_vec)
            if c_norm < 1e-10:
                continue
            vecs.append(candidate_vec / c_norm)
            names.append(song_name)
        except (KeyError, ValueError, TypeError):
            continue

    if not vecs:
        return []

    matrix = np.stack(vecs)

    # GPU-accelerated batch cosine similarity
    try:
        from scripts.core.gpu import gpu_cosine_similarity
        sims = gpu_cosine_similarity(query_vec, matrix)
    except (ImportError, Exception):
        sims = matrix @ query_vec  # CPU fallback — already normalized

    # Sort by similarity (descending)
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(names[i], float(sims[i])) for i in top_indices]
