"""
scripts/core/gpu.py — Centralized GPU device detection and audio utilities.

Every module in AI RemixMate imports from here to decide whether to run on
GPU (MPS / CUDA) or CPU. This avoids duplicated detection logic and gives
one place to force a device override via the REMIXMATE_DEVICE env var.

Usage
─────
    from scripts.core.gpu import get_device, to_tensor, to_numpy, gpu_stft

    device = get_device()                          # "mps" | "cuda" | "cpu"
    t      = to_tensor(np_array)                   # numpy → GPU tensor
    arr    = to_numpy(t)                           # GPU tensor → numpy
    S      = gpu_stft(audio_tensor, n_fft=2048)    # STFT on GPU

Environment
───────────
    REMIXMATE_DEVICE=cpu      # force CPU even if GPU exists
    REMIXMATE_DEVICE=mps      # force MPS
    REMIXMATE_DEVICE=cuda     # force CUDA
"""

from __future__ import annotations

import logging
import os
import platform
from functools import lru_cache
from typing import Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy torch import — not every call path needs torch
# ---------------------------------------------------------------------------
_torch = None


def _import_torch():
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            _torch = False          # sentinel: tried and failed
    return _torch if _torch is not False else None


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_device() -> str:
    """
    Detect the best available compute device.

    Priority:
      1. REMIXMATE_DEVICE env var (explicit override)
      2. Apple Silicon MPS  (macOS + arm64 + torch.backends.mps.is_available)
      3. NVIDIA CUDA        (torch.cuda.is_available)
      4. CPU fallback

    Returns one of: "mps", "cuda", "cpu"
    """
    # ── Explicit override ──────────────────────────────────────────────────
    env = os.environ.get("REMIXMATE_DEVICE", "").strip().lower()
    if env in ("mps", "cuda", "cpu"):
        log.info("[gpu] Device forced via REMIXMATE_DEVICE=%s", env)
        return env

    torch = _import_torch()

    # ── Apple Silicon MPS ──────────────────────────────────────────────────
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        if torch is not None:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                log.info("[gpu] Detected Apple Silicon MPS — GPU enabled")
                return "mps"
        else:
            # torch isn't importable here but the machine IS Apple Silicon.
            # External tools (Demucs CLI) can still accept -d mps.
            log.info("[gpu] Apple Silicon detected (torch not available in this process) — reporting mps")
            return "mps"

    # ── NVIDIA CUDA ────────────────────────────────────────────────────────
    if torch is not None and torch.cuda.is_available():
        log.info("[gpu] Detected NVIDIA CUDA — GPU enabled")
        return "cuda"

    log.info("[gpu] No GPU detected — using CPU")
    return "cpu"


def is_gpu_available() -> bool:
    """Return True if a GPU (MPS or CUDA) is available."""
    return get_device() != "cpu"


def get_torch_device():
    """Return a torch.device object for the detected device."""
    torch = _import_torch()
    if torch is None:
        raise ImportError("PyTorch is not installed")
    return torch.device(get_device())


# ---------------------------------------------------------------------------
# Tensor ↔ NumPy helpers
# ---------------------------------------------------------------------------

def to_tensor(
    arr: np.ndarray,
    dtype=None,
    device: Optional[str] = None,
):
    """
    Convert a numpy array to a torch tensor on the best available device.

    Parameters
    ----------
    arr     : numpy array (any shape)
    dtype   : optional torch dtype (defaults to float32)
    device  : override device string; None → auto-detect
    """
    torch = _import_torch()
    if torch is None:
        raise ImportError("PyTorch is not installed — cannot use GPU acceleration")
    dev = device or get_device()
    if dtype is None:
        dtype = torch.float32
    t = torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=dtype, device=dev)
    return t


def to_numpy(tensor) -> np.ndarray:
    """
    Convert a torch tensor back to a numpy array (always on CPU).
    """
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


# ---------------------------------------------------------------------------
# GPU-accelerated audio primitives
# ---------------------------------------------------------------------------

def gpu_stft(
    audio: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Compute STFT on GPU if available, falling back to numpy/librosa if not.

    Parameters
    ----------
    audio      : 1-D float32 numpy array
    n_fft      : FFT window size
    hop_length : hop size in samples
    device     : optional device override

    Returns
    -------
    Complex spectrogram as numpy array (n_freq, n_frames)
    """
    torch = _import_torch()
    dev = device or get_device()

    if torch is not None and dev != "cpu":
        t = torch.from_numpy(audio.astype(np.float32)).to(dev)
        window = torch.hann_window(n_fft, device=dev)
        S = torch.stft(
            t, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
            window=window, return_complex=True,
        )
        return S.detach().cpu().numpy()
    else:
        # CPU fallback — use librosa if available
        try:
            import librosa
            return librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        except ImportError:
            # Pure numpy fallback (basic)
            from scipy.signal import stft as _stft
            _, _, Zxx = _stft(audio, nperseg=n_fft, noverlap=n_fft - hop_length)
            return Zxx


def gpu_cosine_similarity(
    query: np.ndarray,
    matrix: np.ndarray,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and a matrix of vectors.

    On GPU this is a single batched operation — 50-100x faster for large libraries.

    Parameters
    ----------
    query  : 1-D array (D,)
    matrix : 2-D array (N, D)

    Returns
    -------
    1-D array (N,) of similarity scores in [-1, 1]
    """
    torch = _import_torch()
    dev = device or get_device()

    if torch is not None and dev != "cpu":
        q = torch.from_numpy(query.astype(np.float32)).unsqueeze(0).to(dev)
        m = torch.from_numpy(matrix.astype(np.float32)).to(dev)
        sims = torch.nn.functional.cosine_similarity(q, m, dim=1)
        return sims.detach().cpu().numpy()
    else:
        # CPU fallback
        q_norm = query / (np.linalg.norm(query) + 1e-10)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        m_norm = matrix / norms
        return m_norm @ q_norm


def gpu_time_stretch(
    audio: np.ndarray,
    rate: float,
    sr: int = 44100,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Time-stretch audio on GPU if available, falling back to librosa.

    Uses torchaudio's phase vocoder when available for significant speedup,
    especially in batch scenarios (instrument lab).

    Parameters
    ----------
    audio  : 1-D float32 numpy array
    rate   : stretch factor (>1 = speed up, <1 = slow down)
    sr     : sample rate
    device : optional device override

    Returns
    -------
    Time-stretched audio as numpy array
    """
    if abs(rate - 1.0) < 0.001:
        return audio  # no stretch needed

    torch = _import_torch()
    dev = device or get_device()

    if torch is not None and dev != "cpu":
        try:
            import torchaudio
            t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(dev)
            n_fft = 2048
            hop = 512
            window = torch.hann_window(n_fft, device=dev)
            stft = torch.stft(t, n_fft=n_fft, hop_length=hop,
                              win_length=n_fft, window=window, return_complex=True)
            stretched = torchaudio.functional.phase_vocoder(stft, rate, torch.tensor([hop]))
            result = torch.istft(stretched, n_fft=n_fft, hop_length=hop,
                                 win_length=n_fft, window=window)
            return result.squeeze(0).detach().cpu().numpy()
        except (ImportError, Exception) as e:
            log.debug("[gpu] torchaudio phase_vocoder unavailable (%s), falling back to librosa", e)

    # CPU fallback
    try:
        import librosa
        return librosa.effects.time_stretch(audio, rate=rate)
    except ImportError:
        log.warning("[gpu] Neither torchaudio nor librosa available for time_stretch")
        return audio


def gpu_resample(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Resample audio on GPU if available.

    Parameters
    ----------
    audio     : 1-D float32 numpy array
    orig_sr   : original sample rate
    target_sr : target sample rate

    Returns
    -------
    Resampled audio as numpy array
    """
    if orig_sr == target_sr:
        return audio

    torch = _import_torch()
    dev = device or get_device()

    if torch is not None and dev != "cpu":
        try:
            import torchaudio
            t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(dev)
            resampled = torchaudio.functional.resample(t, orig_sr, target_sr)
            return resampled.squeeze(0).detach().cpu().numpy()
        except (ImportError, Exception):
            pass

    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        log.warning("[gpu] No resampling backend available")
        return audio


def gpu_filter(
    audio: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Apply an IIR filter. Uses torch conv1d on GPU for FIR-approximated filtering,
    falls back to scipy.signal.lfilter on CPU.

    For most audio use cases, this is called from mastering / audio_enhance
    where it's applied to full-length audio arrays.
    """
    torch = _import_torch()
    dev = device or get_device()

    if torch is not None and dev != "cpu" and len(a) == 1:
        # FIR filter (a=[1]) — can use conv1d directly
        t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(dev)
        kernel = torch.from_numpy(b.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(dev)
        pad = len(b) // 2
        filtered = torch.nn.functional.conv1d(t, kernel, padding=pad)
        return filtered.squeeze().detach().cpu().numpy()[:len(audio)]

    # CPU fallback — IIR filters need recursive computation
    from scipy.signal import lfilter
    return lfilter(b, a, audio)


def log_device_info() -> str:
    """Log and return a human-readable device info string."""
    dev = get_device()
    torch = _import_torch()
    info_parts = [f"Device: {dev.upper()}"]

    if torch is not None:
        info_parts.append(f"PyTorch: {torch.__version__}")
        if dev == "cuda":
            info_parts.append(f"GPU: {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            info_parts.append(f"VRAM: {mem:.1f} GB")
        elif dev == "mps":
            info_parts.append("GPU: Apple Silicon (Metal Performance Shaders)")
    else:
        info_parts.append("PyTorch: not installed")

    info = " · ".join(info_parts)
    log.info("[gpu] %s", info)
    return info
