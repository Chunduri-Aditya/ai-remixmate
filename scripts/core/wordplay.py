"""
scripts/core/wordplay.py — Wordplay / NLP lyric-transition matcher.

Finds tracks where the closing lyrics of one song share a phrase, theme,
or semantic fingerprint with the opening lyrics of the next — the "wordplay"
transition technique used by radio DJs and mixtape curators.

Pipeline
────────
1. Fetch lyrics from Genius API via ``lyricsgenius`` (with local disk cache).
2. Tokenise + clean with NLTK (stop-word removal, lemmatisation).
3. Extract n-gram phrase fingerprints from the first/last N lines.
4. Score pairwise overlap: exact n-gram Jaccard + optional TF-IDF cosine.
5. Return ranked ``WordplayPair`` list ready for use in SetlistPlanner.

Soft dependencies
─────────────────
  pip install lyricsgenius nltk

Both are optional — the module degrades gracefully if missing:
  • No lyricsgenius → no lyrics, all pairs score 0.0.
  • No nltk data → falls back to whitespace tokenisation.

Environment variable:
  GENIUS_TOKEN   — Genius client-access token (read-only, safe to expose).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

log = logging.getLogger(__name__)

# ── optional imports ──────────────────────────────────────────────────────────

try:
    import lyricsgenius  # type: ignore

    _HAS_GENIUS = True
except ImportError:
    _HAS_GENIUS = False
    log.debug("lyricsgenius not installed — lyric fetch disabled")

try:
    import nltk  # type: ignore
    from nltk.corpus import stopwords  # type: ignore
    from nltk.stem import WordNetLemmatizer  # type: ignore
    from nltk.tokenize import word_tokenize  # type: ignore

    _HAS_NLTK = True
    _ensure_nltk_data_done = False
except ImportError:
    _HAS_NLTK = False
    log.debug("nltk not installed — falling back to whitespace tokenisation")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

if TYPE_CHECKING:
    pass


# ── constants ────────────────────────────────────────────────────────────────

_CACHE_DIR = Path(os.getenv("REMIXMATE_CACHE_DIR", "~/.cache/remixmate/lyrics")).expanduser()
_GENIUS_TOKEN_ENV = "GENIUS_TOKEN"

# How many lines from start/end of song to use for matching
_HEAD_LINES = 8   # opening lines (where next track "begins")
_TAIL_LINES = 8   # closing lines (where current track "ends")

_NGRAM_SIZES = (1, 2, 3)          # unigrams, bigrams, trigrams
_STOP_WORDS: set[str] = set()     # populated lazily after NLTK data confirmed


# ── data classes ─────────────────────────────────────────────────────────────

@dataclass
class LyricProfile:
    """Cleaned token fingerprint of a track's head/tail lyrics."""

    track_id: str          # arbitrary stable identifier (e.g. Spotify track ID or title slug)
    title: str
    artist: str

    head_tokens: list[str] = field(default_factory=list)   # opening lyric tokens
    tail_tokens: list[str] = field(default_factory=list)   # closing lyric tokens
    head_raw: str = ""                                      # raw opening text (debug)
    tail_raw: str = ""                                      # raw closing text (debug)

    @property
    def has_lyrics(self) -> bool:
        return bool(self.head_tokens or self.tail_tokens)


@dataclass
class WordplayPair:
    """
    A candidate lyric-transition between two tracks.

    ``source`` → ``target`` means: source plays first, target plays next.
    The score reflects how well the closing words of source echo the opening
    words of target (or vice-versa — the matcher checks both directions).
    """

    source_id: str
    source_title: str
    source_artist: str

    target_id: str
    target_title: str
    target_artist: str

    score: float                     # 0.0 = no match, 1.0 = perfect
    matched_phrases: list[str] = field(default_factory=list)
    match_type: str = "ngram"        # "ngram" | "tfidf" | "combined"

    def __str__(self) -> str:
        phrases = ", ".join(f'"{p}"' for p in self.matched_phrases[:3])
        return (
            f'{self.source_title} → {self.target_title}  '
            f'[score={self.score:.2f}, phrases={phrases or "none"}]'
        )


# ── NLTK bootstrap ────────────────────────────────────────────────────────────

def _ensure_nltk_data() -> None:
    global _ensure_nltk_data_done, _STOP_WORDS
    if not _HAS_NLTK or _ensure_nltk_data_done:
        return
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg
                           else f"corpora/{pkg}" if pkg in ("stopwords", "wordnet", "omw-1.4")
                           else pkg)
        except LookupError:
            nltk.download(pkg, quiet=True)
    _STOP_WORDS = set(stopwords.words("english"))
    _ensure_nltk_data_done = True


# ── text helpers ──────────────────────────────────────────────────────────────

_BRACKET_RE = re.compile(r"\[.*?\]|\(.*?\)")    # [Chorus], (Verse 1), etc.
_PUNCT_RE = re.compile(r"[^\w\s']")


def _clean_lyrics(raw: str) -> list[str]:
    """Return a list of non-empty, lowercased lyric lines."""
    lines = []
    for line in raw.splitlines():
        line = _BRACKET_RE.sub("", line).strip()
        if line:
            lines.append(line.lower())
    return lines


def _tokenise(text: str) -> list[str]:
    """Tokenise a text block; returns lemmatised, stop-word-free tokens."""
    _ensure_nltk_data()
    text = _PUNCT_RE.sub(" ", text.lower())
    if _HAS_NLTK:
        tokens = word_tokenize(text)
        lemmatiser = WordNetLemmatizer()
        return [
            lemmatiser.lemmatize(t)
            for t in tokens
            if t.isalpha() and t not in _STOP_WORDS and len(t) > 1
        ]
    # Fallback: plain whitespace split
    return [w for w in text.split() if len(w) > 2]


def _ngrams(tokens: list[str], n: int) -> set[str]:
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _ngram_jaccard(a: list[str], b: list[str]) -> tuple[float, list[str]]:
    """Return Jaccard similarity + list of shared n-gram strings."""
    if not a or not b:
        return 0.0, []

    shared: list[str] = []
    union_count = 0
    intersect_count = 0

    for n in _NGRAM_SIZES:
        sa, sb = _ngrams(a, n), _ngrams(b, n)
        inter = sa & sb
        union_count += len(sa | sb)
        intersect_count += len(inter)
        # Weight longer n-grams more in the phrase list
        shared.extend(sorted(inter, key=lambda x: -len(x.split())))

    if union_count == 0:
        return 0.0, []

    jaccard = intersect_count / union_count
    # Deduplicate: remove sub-phrases already covered by longer ones
    deduped: list[str] = []
    for phrase in shared:
        if not any(phrase in longer for longer in deduped):
            deduped.append(phrase)

    return jaccard, deduped[:5]


def _tfidf_cosine(docs: list[str]) -> list[list[float]]:
    """Return pairwise TF-IDF cosine matrix for a list of text documents."""
    if not _HAS_SKLEARN or len(docs) < 2:
        return [[1.0 if i == j else 0.0 for j in range(len(docs))] for i in range(len(docs))]
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    try:
        mat = vec.fit_transform(docs)
        sim = cosine_similarity(mat).tolist()
        return sim
    except ValueError:
        n = len(docs)
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


# ── Genius lyric cache ────────────────────────────────────────────────────────

def _cache_path(title: str, artist: str) -> Path:
    slug = hashlib.md5(f"{title}|||{artist}".encode()).hexdigest()
    return _CACHE_DIR / f"{slug}.json"


def _load_from_cache(title: str, artist: str) -> str | None:
    p = _cache_path(title, artist)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            return data.get("lyrics")
        except Exception:
            pass
    return None


def _save_to_cache(title: str, artist: str, lyrics: str | None) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _cache_path(title, artist)
    p.write_text(json.dumps({"title": title, "artist": artist, "lyrics": lyrics}))


def _fetch_lyrics_genius(
    title: str,
    artist: str,
    genius: "lyricsgenius.Genius",
) -> str | None:
    """Fetch raw lyrics from Genius, using disk cache to avoid repeat calls."""
    cached = _load_from_cache(title, artist)
    if cached is not None:
        return cached or None   # empty string → not found

    log.debug("Genius fetch: %s — %s", artist, title)
    try:
        song = genius.search_song(title, artist, get_full_info=False)
        lyrics = song.lyrics if song else None
    except Exception as exc:
        log.warning("Genius error for %s — %s: %s", artist, title, exc)
        lyrics = None

    _save_to_cache(title, artist, lyrics or "")
    time.sleep(0.4)   # polite rate limiting (Genius allows ~25 req/s)
    return lyrics


# ── profile builder ───────────────────────────────────────────────────────────

def build_lyric_profile(
    track_id: str,
    title: str,
    artist: str,
    genius: "lyricsgenius.Genius | None" = None,
) -> LyricProfile:
    """
    Fetch + parse lyrics for one track.  Returns a ``LyricProfile`` regardless
    of whether lyrics were found (``has_lyrics`` will be False if not).
    """
    profile = LyricProfile(track_id=track_id, title=title, artist=artist)

    raw: str | None = None
    if genius is not None and _HAS_GENIUS:
        raw = _fetch_lyrics_genius(title, artist, genius)

    if not raw:
        return profile

    lines = _clean_lyrics(raw)
    if not lines:
        return profile

    head_lines = lines[:_HEAD_LINES]
    tail_lines = lines[-_TAIL_LINES:]

    profile.head_raw = " ".join(head_lines)
    profile.tail_raw = " ".join(tail_lines)
    profile.head_tokens = _tokenise(profile.head_raw)
    profile.tail_tokens = _tokenise(profile.tail_raw)

    return profile


# ── pairwise scorer ───────────────────────────────────────────────────────────

def score_pair(
    source: LyricProfile,
    target: LyricProfile,
    tfidf_matrix: list[list[float]] | None = None,
    source_idx: int = 0,
    target_idx: int = 0,
) -> WordplayPair:
    """
    Score one (source → target) lyric transition.

    Combines:
      • N-gram Jaccard between source *tail* and target *head* (primary signal)
      • Optional TF-IDF cosine (secondary signal, weighted lower)
    """
    jaccard, phrases = _ngram_jaccard(source.tail_tokens, target.head_tokens)

    tfidf_score = 0.0
    if tfidf_matrix is not None:
        try:
            tfidf_score = float(tfidf_matrix[source_idx][target_idx])
        except (IndexError, TypeError):
            pass

    # Combine: 70% jaccard (phrase match) + 30% tfidf (semantic overlap)
    combined = 0.70 * jaccard + 0.30 * tfidf_score

    match_type = "ngram" if tfidf_matrix is None else "combined"

    return WordplayPair(
        source_id=source.track_id,
        source_title=source.title,
        source_artist=source.artist,
        target_id=target.track_id,
        target_title=target.title,
        target_artist=target.artist,
        score=round(combined, 4),
        matched_phrases=phrases,
        match_type=match_type,
    )


# ── main public API ───────────────────────────────────────────────────────────

@dataclass
class TrackInput:
    """Minimal track descriptor for wordplay matching (no full TrackNode required)."""

    track_id: str
    title: str
    artist: str


def find_wordplay_pairs(
    tracks: Sequence[TrackInput],
    genius_token: str | None = None,
    min_similarity: float = 0.05,
    use_tfidf: bool = True,
    max_pairs: int = 200,
) -> list[WordplayPair]:
    """
    Find all lyric-adjacent track pairs that score above ``min_similarity``.

    Parameters
    ──────────
    tracks          Ordered list of tracks to analyse (typically a playlist).
    genius_token    Genius client-access token. Falls back to GENIUS_TOKEN env var.
    min_similarity  Minimum combined score to include a pair (0.0–1.0).
    use_tfidf       Whether to add TF-IDF cosine on top of n-gram Jaccard.
    max_pairs       Cap on returned results (sorted descending by score).

    Returns
    ───────
    Sorted list of ``WordplayPair`` objects (highest score first).
    """
    token = genius_token or os.getenv(_GENIUS_TOKEN_ENV)
    genius_client = None

    if token and _HAS_GENIUS:
        try:
            genius_client = lyricsgenius.Genius(
                token,
                skip_non_songs=True,
                excluded_terms=["(Remix)", "(Live)", "(Acoustic)"],
                verbose=False,
                timeout=8,
            )
        except Exception as exc:
            log.warning("Could not initialise Genius client: %s", exc)

    # ── Build lyric profiles ──────────────────────────────────────────────────
    profiles: list[LyricProfile] = []
    for t in tracks:
        p = build_lyric_profile(t.track_id, t.title, t.artist, genius=genius_client)
        profiles.append(p)
        if not p.has_lyrics:
            log.debug("No lyrics found: %s — %s", t.artist, t.title)

    # ── TF-IDF matrix (tail docs as rows) ────────────────────────────────────
    tfidf_matrix: list[list[float]] | None = None
    if use_tfidf and _HAS_SKLEARN:
        tail_docs = [p.tail_raw if p.tail_raw else " " for p in profiles]
        head_docs = [p.head_raw if p.head_raw else " " for p in profiles]
        # We want similarity between each tail_i and head_j — build cross-matrix
        try:
            all_docs = tail_docs + head_docs
            vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
            mat = vec.fit_transform(all_docs)
            n = len(profiles)
            tail_vecs = mat[:n]
            head_vecs = mat[n:]
            cross = cosine_similarity(tail_vecs, head_vecs).tolist()
            tfidf_matrix = cross   # shape [n_tracks × n_tracks]
        except Exception as exc:
            log.warning("TF-IDF matrix failed: %s", exc)

    # ── Score all (source, target) pairs ─────────────────────────────────────
    pairs: list[WordplayPair] = []
    n = len(profiles)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            source = profiles[i]
            target = profiles[j]

            if not source.has_lyrics or not target.has_lyrics:
                continue

            tfidf_val: list[list[float]] | None = None
            if tfidf_matrix is not None:
                tfidf_val = [[tfidf_matrix[i][j]]]
                pair = score_pair(source, target, tfidf_val, 0, 0)
            else:
                pair = score_pair(source, target)

            if pair.score >= min_similarity:
                pairs.append(pair)

    pairs.sort(key=lambda p: -p.score)
    return pairs[:max_pairs]


def best_next_wordplay(
    current_id: str,
    candidates: Sequence[TrackInput],
    all_pairs: list[WordplayPair],
) -> WordplayPair | None:
    """
    From a pre-computed pairs list, return the best wordplay transition
    from ``current_id`` into any of the candidate tracks.

    Used by SetlistPlanner to augment the harmonic/BPM cost function.
    """
    relevant = [
        p for p in all_pairs
        if p.source_id == current_id
        and p.target_id in {c.track_id for c in candidates}
    ]
    return relevant[0] if relevant else None


def wordplay_bonus(
    current_id: str,
    target_id: str,
    all_pairs: list[WordplayPair],
    weight: float = 0.10,
) -> float:
    """
    Return a cost *reduction* (0.0–``weight``) to apply when the target track
    has a strong lyric connection to the current track.

    Plug into SetlistPlanner's transition_cost as:
        total_cost -= wordplay_bonus(...)
    """
    for p in all_pairs:
        if p.source_id == current_id and p.target_id == target_id:
            return weight * p.score
    return 0.0
