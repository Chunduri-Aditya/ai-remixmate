"""
Lyrics extraction with automatic fallback chain:
Whisper → SpeechRecognition (Google) → PocketSphinx
"""
import os
import logging

logger = logging.getLogger(__name__)

# Check availability
WHISPER_AVAILABLE = False
SPEECHRECOGNITION_AVAILABLE = False
SPHINX_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    pass

try:
    import speech_recognition as sr
    SPEECHRECOGNITION_AVAILABLE = True
except ImportError:
    pass

try:
    from pocketsphinx import pocketsphinx
    SPHINX_AVAILABLE = True
except ImportError:
    pass

def extract_lyrics_whisper(audio_path, model_size="base"):
    """Extract lyrics using Whisper AI."""
    if not WHISPER_AVAILABLE:
        return None
    
    try:
        logger.info(f"Using Whisper ({model_size} model) for lyrics extraction")
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path, verbose=False)
        
        return {
            "full_text": result.get("text", ""),
            "segments": [
                {
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "text": seg.get("text", "").strip()
                }
                for seg in result.get("segments", [])
            ],
            "language": result.get("language", "en"),
            "method": "whisper"
        }
    except Exception as e:
        logger.error(f"Whisper extraction failed: {e}")
        return None

def extract_lyrics_speechrecognition(audio_path):
    """Extract lyrics using SpeechRecognition (Google API or Sphinx)."""
    if not SPEECHRECOGNITION_AVAILABLE:
        return None
    
    try:
        r = sr.Recognizer()
        
        # Try Google API first
        try:
            with sr.AudioFile(audio_path) as source:
                audio = r.record(source)
            
            logger.info("Using SpeechRecognition (Google API) for lyrics extraction")
            text = r.recognize_google(audio)
            
            return {
                "full_text": text,
                "segments": [{"start": 0.0, "end": 0.0, "text": text}],
                "language": "en",
                "method": "speechrecognition_google"
            }
        except sr.UnknownValueError:
            logger.warning("Google API could not understand audio")
        except sr.RequestError as e:
            logger.warning(f"Google API error: {e}")
            
            # Fallback to Sphinx if available
            if SPHINX_AVAILABLE:
                try:
                    logger.info("Falling back to PocketSphinx")
                    with sr.AudioFile(audio_path) as source:
                        audio = r.record(source)
                    text = r.recognize_sphinx(audio)
                    
                    return {
                        "full_text": text,
                        "segments": [{"start": 0.0, "end": 0.0, "text": text}],
                        "language": "en",
                        "method": "speechrecognition_sphinx"
                    }
                except Exception as e:
                    logger.error(f"Sphinx extraction failed: {e}")
        
        return None
    except Exception as e:
        logger.error(f"SpeechRecognition extraction failed: {e}")
        return None

def extract_lyrics(audio_path, model_size="base", use_fallback=True):
    """
    Extract lyrics with automatic fallback chain.
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        use_fallback: Whether to try fallback methods if Whisper fails
    
    Returns:
        Dict with lyrics data or None if all methods fail
    """
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return None
    
    # Try Whisper first
    result = extract_lyrics_whisper(audio_path, model_size)
    if result:
        return result
    
    # Fallback to SpeechRecognition if enabled
    if use_fallback:
        result = extract_lyrics_speechrecognition(audio_path)
        if result:
            return result
    
    # All methods failed
    logger.warning("No lyrics extraction methods available. Install Whisper or SpeechRecognition.")
    if not WHISPER_AVAILABLE and not SPEECHRECOGNITION_AVAILABLE:
        logger.info("Installation instructions:")
        logger.info("  - Whisper: pip install openai-whisper")
        logger.info("  - SpeechRecognition: pip install SpeechRecognition")
        logger.info("  - PocketSphinx (offline): pip install SpeechRecognition pocketsphinx")
    
    return None

