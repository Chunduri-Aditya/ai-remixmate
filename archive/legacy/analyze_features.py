# scripts/analyze_features.py

import librosa
import librosa.display
import numpy as np

def analyze_audio(file_path):
    print(f"🎧 Analyzing: {file_path}")

    y, sr = librosa.load(file_path)
    
    # Tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Chroma = Pitch content (for key estimation)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    avg_chroma = np.mean(chroma, axis=1)

    # MFCC = timbre/texture fingerprint
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    avg_mfcc = np.mean(mfcc, axis=1)

    print(f"🕺 Tempo (BPM): {tempo}")
    print(f"🎼 Avg Chroma Vector: {avg_chroma}")
    print(f"🔉 Avg MFCC Vector: {avg_mfcc}")

    return {
        "tempo": tempo,
        "chroma": avg_chroma,
        "mfcc": avg_mfcc
    }

if __name__ == "__main__":
    analyze_audio("../separated/htdemucs/HER/vocals.wav")