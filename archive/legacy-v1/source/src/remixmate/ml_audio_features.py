"""
ML-based audio feature extraction, genre classification, and adaptive remix strategy.
"""
import os
import json
import logging
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

logger = logging.getLogger(__name__)

# Model storage
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "ml_models")
os.makedirs(MODELS_DIR, exist_ok=True)

GENRE_MODEL_PATH = os.path.join(MODELS_DIR, "genre_classifier.pkl")
ENERGY_MODEL_PATH = os.path.join(MODELS_DIR, "energy_regressor.pkl")
DANCEABILITY_MODEL_PATH = os.path.join(MODELS_DIR, "danceability_regressor.pkl")
VALENCE_MODEL_PATH = os.path.join(MODELS_DIR, "valence_regressor.pkl")
PREFERENCES_PATH = os.path.join(MODELS_DIR, "user_preferences.json")

# Genre labels
GENRES = ["blues", "classical", "country", "disco", "hip-hop", "jazz", "metal", "pop", "reggae", "rock"]

def extract_deep_features(audio_path, sr=44100):
    """
    Extract comprehensive deep audio features.
    
    Returns:
        Dictionary of features
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=60)  # First 60 seconds for speed
        
        # MFCC (13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Chroma features (with fallback)
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        except AttributeError:
            try:
                chroma = librosa.feature.chroma(y=y, sr=sr)
            except AttributeError:
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Tempo and rhythm
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        onset_strength = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
        
        # RMS energy
        rms = np.mean(librosa.feature.rms(y=y))
        
        return {
            "mfcc_mean": mfcc_mean.tolist(),
            "mfcc_std": mfcc_std.tolist(),
            "chroma_mean": chroma_mean.tolist(),
            "spectral_centroid": float(spectral_centroid),
            "spectral_rolloff": float(spectral_rolloff),
            "spectral_bandwidth": float(spectral_bandwidth),
            "zero_crossing_rate": float(zero_crossing_rate),
            "tempo": float(tempo),
            "onset_strength": float(onset_strength),
            "rms": float(rms)
        }
    except Exception as e:
        logger.error(f"Error extracting deep features: {e}")
        return None

def _features_to_vector(features):
    """Convert features dict to numpy vector."""
    if features is None:
        return None
    
    vector = []
    vector.extend(features["mfcc_mean"])
    vector.extend(features["mfcc_std"])
    vector.extend(features["chroma_mean"])
    vector.append(features["spectral_centroid"])
    vector.append(features["spectral_rolloff"])
    vector.append(features["spectral_bandwidth"])
    vector.append(features["zero_crossing_rate"])
    vector.append(features["tempo"])
    vector.append(features["onset_strength"])
    vector.append(features["rms"])
    
    return np.array(vector)

def classify_genre(audio_path, use_ml=True):
    """
    Classify genre using ML model or rule-based fallback.
    
    Returns:
        (genre, confidence) tuple
    """
    features = extract_deep_features(audio_path)
    if features is None:
        return ("unknown", 0.0)
    
    # Try ML model first
    if use_ml and os.path.exists(GENRE_MODEL_PATH):
        try:
            with open(GENRE_MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            vector = _features_to_vector(features)
            if vector is not None:
                genre_idx = model.predict([vector])[0]
                proba = model.predict_proba([vector])[0]
                confidence = float(np.max(proba))
                return (GENRES[genre_idx], confidence)
        except Exception as e:
            logger.warning(f"ML genre classification failed: {e}, using rule-based")
    
    # Rule-based fallback
    tempo = features["tempo"]
    energy = features["rms"]
    spectral_centroid = features["spectral_centroid"]
    
    if tempo > 120 and energy > 0.1:
        return ("edm", 0.7)
    elif tempo > 90 and energy > 0.08:
        return ("hip-hop", 0.7)
    elif tempo < 100 and energy < 0.05:
        return ("ambient", 0.7)
    elif tempo > 110:
        return ("dance", 0.7)
    else:
        return ("pop", 0.7)

def predict_energy(audio_path):
    """Predict energy level (0.0-1.0) similar to Spotify."""
    features = extract_deep_features(audio_path)
    if features is None:
        return 0.5
    
    # Try ML model
    if os.path.exists(ENERGY_MODEL_PATH):
        try:
            with open(ENERGY_MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            vector = _features_to_vector(features)
            if vector is not None:
                return float(model.predict([vector])[0])
        except Exception as e:
            logger.warning(f"ML energy prediction failed: {e}, using rule-based")
    
    # Rule-based: weighted combination
    rms = features["rms"]
    spectral_centroid = features["spectral_centroid"] / 5000.0  # Normalize
    zero_crossing = features["zero_crossing_rate"]
    tempo = min(features["tempo"] / 200.0, 1.0)  # Normalize
    onset = features["onset_strength"]
    
    energy = (
        0.30 * min(rms * 10, 1.0) +
        0.25 * min(spectral_centroid, 1.0) +
        0.15 * min(zero_crossing * 10, 1.0) +
        0.20 * tempo +
        0.10 * min(onset, 1.0)
    )
    
    return float(np.clip(energy, 0.0, 1.0))

def predict_danceability(audio_path):
    """Predict danceability (0.0-1.0) similar to Spotify."""
    features = extract_deep_features(audio_path)
    if features is None:
        return 0.5
    
    # Try ML model
    if os.path.exists(DANCEABILITY_MODEL_PATH):
        try:
            with open(DANCEABILITY_MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            vector = _features_to_vector(features)
            if vector is not None:
                return float(model.predict([vector])[0])
        except Exception as e:
            logger.warning(f"ML danceability prediction failed: {e}, using rule-based")
    
    # Rule-based
    tempo = features["tempo"]
    beat_strength = features["onset_strength"]
    rms = features["rms"]
    
    # Normalize
    tempo_norm = min(tempo / 200.0, 1.0)
    beat_norm = min(beat_strength, 1.0)
    rms_norm = min(rms * 10, 1.0)
    
    danceability = (
        0.30 * tempo_norm +
        0.25 * beat_norm +
        0.20 * rms_norm +
        0.15 * features["onset_strength"] +
        0.10 * min(features["spectral_centroid"] / 5000.0, 1.0)
    )
    
    return float(np.clip(danceability, 0.0, 1.0))

def predict_valence(audio_path):
    """Predict valence/positivity (0.0-1.0) similar to Spotify."""
    features = extract_deep_features(audio_path)
    if features is None:
        return 0.5
    
    # Try ML model
    if os.path.exists(VALENCE_MODEL_PATH):
        try:
            with open(VALENCE_MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            vector = _features_to_vector(features)
            if vector is not None:
                return float(model.predict([vector])[0])
        except Exception as e:
            logger.warning(f"ML valence prediction failed: {e}, using rule-based")
    
    # Rule-based: based on tempo and brightness
    tempo = features["tempo"]
    brightness = features["spectral_centroid"] / 5000.0
    
    # Higher tempo and brightness = more positive
    valence = 0.5 + 0.3 * min(tempo / 200.0, 1.0) + 0.2 * min(brightness, 1.0)
    
    return float(np.clip(valence, 0.0, 1.0))

class AdaptiveRemixStrategyEngine:
    """Learns from user preferences to optimize mixing strategies."""
    
    def __init__(self):
        self.preferences = self._load_preferences()
        self.model = None
        self._load_or_create_model()
    
    def _load_preferences(self):
        """Load user preferences from JSON."""
        if os.path.exists(PREFERENCES_PATH):
            try:
                with open(PREFERENCES_PATH, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"examples": []}
    
    def _save_preferences(self):
        """Save user preferences to JSON."""
        with open(PREFERENCES_PATH, 'w') as f:
            json.dump(self.preferences, f, indent=2)
    
    def _load_or_create_model(self):
        """Load existing model or create new one."""
        model_path = os.path.join(MODELS_DIR, "strategy_model.pkl")
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                return
            except Exception:
                pass
        
        # Create new model
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    def _save_model(self):
        """Save model to disk."""
        model_path = os.path.join(MODELS_DIR, "strategy_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def predict_best_mixing_strategy(self, track1_features, track2_features, key1=None, key2=None):
        """
        Predict best mixing strategy based on track features.
        
        Returns:
            Dict with strategy and parameters
        """
        # Extract features
        bpm_diff = abs(track1_features.get("bpm", 128) - track2_features.get("bpm", 128))
        energy_diff = abs(track1_features.get("energy", 0.5) - track2_features.get("energy", 0.5))
        genre1 = track1_features.get("genre", "pop")
        genre2 = track2_features.get("genre", "pop")
        genre_match = 1.0 if genre1 == genre2 else 0.0
        
        # Key compatibility
        key_compat = 1.0
        if key1 and key2:
            # Simple compatibility check (same or adjacent Camelot keys)
            if key1 == key2:
                key_compat = 1.0
            elif abs(int(key1[:-1]) - int(key2[:-1])) <= 1:
                key_compat = 0.8
            else:
                key_compat = 0.5
        
        # Feature vector
        features = np.array([[
            bpm_diff / 50.0,  # Normalize
            energy_diff,
            genre_match,
            key_compat
        ]])
        
        # Predict if model is trained
        if len(self.preferences.get("examples", [])) >= 20:
            try:
                strategy_idx = self.model.predict(features)[0]
                strategies = ["crossfade", "bass_swap", "quick_cut", "hybrid"]
                strategy = strategies[strategy_idx]
            except Exception:
                strategy = self._rule_based_strategy(bpm_diff, energy_diff, genre1, genre2)
        else:
            strategy = self._rule_based_strategy(bpm_diff, energy_diff, genre1, genre2)
        
        # Determine parameters
        if strategy == "crossfade":
            if bpm_diff < 5 and key_compat > 0.8:
                crossfade_length = 16.0
            elif genre1 == "edm" or genre2 == "edm":
                crossfade_length = 12.0
            else:
                crossfade_length = 8.0
        elif strategy == "quick_cut":
            crossfade_length = 0.0
        else:
            crossfade_length = 8.0
        
        return {
            "technique": strategy,
            "crossfade_length": crossfade_length,
            "apply_beatmatching": bpm_diff > 5,
            "apply_harmonic_mixing": key_compat < 0.8
        }
    
    def _rule_based_strategy(self, bpm_diff, energy_diff, genre1, genre2):
        """Rule-based strategy selection."""
        if bpm_diff > 15 or (genre1 == "hip-hop" or genre2 == "hip-hop"):
            return "quick_cut"
        elif genre1 == "edm" or genre2 == "edm" or energy_diff > 0.3:
            return "bass_swap"
        else:
            return "crossfade"
    
    def learn_from_example(self, track1_features, track2_features, chosen_strategy, user_satisfaction):
        """
        Learn from a user's mixing choice.
        
        Args:
            track1_features: Features of track 1
            track2_features: Features of track 2
            chosen_strategy: Strategy the user chose
            user_satisfaction: Satisfaction score (0.0-1.0)
        """
        # Only learn from positive examples (satisfaction > 0.7)
        if user_satisfaction < 0.7:
            return
        
        # Extract features
        bpm_diff = abs(track1_features.get("bpm", 128) - track2_features.get("bpm", 128))
        energy_diff = abs(track1_features.get("energy", 0.5) - track2_features.get("energy", 0.5))
        genre1 = track1_features.get("genre", "pop")
        genre2 = track2_features.get("genre", "pop")
        genre_match = 1.0 if genre1 == genre2 else 0.0
        
        # Key compatibility
        key1 = track1_features.get("key")
        key2 = track2_features.get("key")
        key_compat = 1.0
        if key1 and key2:
            if key1 == key2:
                key_compat = 1.0
            elif abs(int(key1[:-1]) - int(key2[:-1])) <= 1:
                key_compat = 0.8
            else:
                key_compat = 0.5
        
        # Strategy mapping
        strategy_map = {"crossfade": 0, "bass_swap": 1, "quick_cut": 2, "hybrid": 3}
        strategy_label = strategy_map.get(chosen_strategy, 0)
        
        # Add example
        example = {
            "features": [bpm_diff / 50.0, energy_diff, genre_match, key_compat],
            "strategy": strategy_label,
            "satisfaction": user_satisfaction
        }
        
        self.preferences.setdefault("examples", []).append(example)
        self._save_preferences()
        
        # Retrain if we have enough examples
        if len(self.preferences["examples"]) >= 20 and len(self.preferences["examples"]) % 10 == 0:
            self._retrain_model()
    
    def _retrain_model(self):
        """Retrain the strategy model from collected examples."""
        examples = self.preferences.get("examples", [])
        if len(examples) < 20:
            return
        
        X = np.array([ex["features"] for ex in examples])
        y = np.array([ex["strategy"] for ex in examples])
        
        # Weight by satisfaction
        sample_weights = np.array([ex["satisfaction"] for ex in examples])
        
        try:
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            self.model.fit(X, y, sample_weight=sample_weights)
            self._save_model()
            logger.info("Strategy model retrained successfully")
        except Exception as e:
            logger.error(f"Failed to retrain model: {e}")

# Global engine instance
_strategy_engine = None

def get_strategy_engine():
    """Get or create the global strategy engine."""
    global _strategy_engine
    if _strategy_engine is None:
        _strategy_engine = AdaptiveRemixStrategyEngine()
    return _strategy_engine

