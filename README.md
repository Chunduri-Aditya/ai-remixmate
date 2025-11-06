# 🎵 AI RemixMate

**AI RemixMate** is an intelligent music remixing engine that takes one song and finds the most compatible match based on musical features, then blends the two using dynamic mixing techniques. It uses audio feature extraction (MFCC, tempo, chroma), intelligent similarity analysis, and vocal/instrumental separation for high-quality remixes.

---

## 🚀 Features

- 🎧 **Remix two songs intelligently** (manual or auto match)
- 🧠 **Find most similar songs** using MFCC, tempo, chroma
- 🎤 **Separate vocals and instruments** using [Demucs](https://github.com/facebookresearch/demucs)
- 🗃️ **Build your own song database** for similarity search
- 🎼 **Lyrics extraction** (via Whisper or local ASR)
- 📦 Modular and expandable script-based design

---

## 🗂️ Project Structure

```bash
ai-remixmate/
├── scripts/                # All utility scripts
├── audio_input/           # Place input songs here
├── audio_output/          # Final remixes get saved here
├── separated/             # Separated stems (vocals/instruments)
├── stems/                 # Intermediate stem files
├── output/                # Optional audio exports
├── lyrics/                # Extracted lyrics
├── models/                # Model files or song embeddings
├── requirements.txt       # Python dependencies
├── .gitignore             # Ignore unnecessary files
└── README.md              # Project documentation
```

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Chunduri-Aditya/ai-remixmate.git
cd ai-remixmate

# Create and activate virtual environment (optional but recommended)
python3 -m venv remix-env
source remix-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🛠️ Usage

### ➤ 1. Add Songs to Database

```bash
python3 scripts/song_database.py
# You’ll be prompted to enter .wav file path and song name
```

---

### ➤ 2. Automatic Smart Remix (recommended)

```bash
python3 scripts/smart_remix.py --base "YourBaseSongName"
```

- Finds best match
- Lets you pick match
- Automatically remixes vocals + instrumentals

---

### ➤ 3. Manual Remix from Similarity

```bash
python3 scripts/remix_from_match.py "YourBaseSongName"
```

- Shows similar matches
- Lets you manually pick match for remix

---

## 📊 Audio Features Used for Matching

- `tempo`: beats per minute
- `mfcc`: Mel-frequency cepstral coefficients
- `chroma`: harmonic content

These features are extracted using `librosa` and stored in a JSON file for similarity search.

---

## 🔍 Matching Algorithm

- Songs are compared using cosine similarity over a feature vector made from MFCC, chroma, and tempo.
- The top-k matches (default 5) are sorted and shown to the user.

---

## 🧠 Models Used

| Task              | Model     |
|-------------------|-----------|
| Stem Separation   | Demucs    |
| Lyrics Extraction | OpenAI Whisper or local ASR |
| Feature Extraction| Librosa   |

---

## 🧪 Example Workflow

```bash
# Add songs
python3 scripts/song_database.py

# Remix automatically
python3 scripts/smart_remix.py --base "Timeless"

# View output
open audio_output/
```

⚠️ *Note: Remix results are experimental and may require fine-tuning for musical smoothness or genre transitions. This is a research-grade prototype and not yet production-ready.*

---

## 🧹 Folder Notes

All key folders are tracked using `.gitkeep` so the structure stays intact on GitHub. They will remain even if empty:
- `audio_input/`, `audio_output/`, `stems/`, `lyrics/`, `models/`, etc.

---

## 🧰 Requirements

Python 3.8 or higher  
Dependencies listed in `requirements.txt` (e.g., `librosa`, `numpy`, `pydub`, etc.)

---

## 🤝 Contributions

Pull requests and suggestions welcome!

---

## ⚖️ Disclaimer

This project is intended for educational and research purposes only.  
The author **is not responsible** for any misuse, copyright infringement, or distribution of generated remixes.  
It is the user’s responsibility to ensure lawful and ethical use of AI RemixMate.  
All music rights remain with original creators and copyright holders.

---

## 📄 License

MIT License

---

## 💡 Author

**Aditya Chunduri**  
📧 chunduri@usc.edu  
🌐 [GitHub](https://github.com/Chunduri-Aditya)
