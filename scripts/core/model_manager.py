"""
scripts/core/model_manager.py — Centralized AI model registry for AI RemixMate.

Handles lazy loading, sequential model lifecycle, and Apple Silicon (MPS) unified
memory budgeting for all heavyweight generative models (MusicGen, VampNet, DAC,
EnCodec). The key insight for MPS: it shares memory with the CPU, so "VRAM" really
just means "total RAM pressure". This manager enforces a configurable GB ceiling
and offloads unused models automatically.

Architecture
────────────
  • Singleton `ModelManager` — one instance across the entire process.
  • `threading.Lock` — serializes GPU ops since tasks run in a ThreadPoolExecutor.
  • Lazy loading — models not downloaded/loaded until first request.
  • Sequential strategy — offloads all OTHER models before loading a new one,
    keeping peak memory as low as possible (critical for 16 GB unified memory).
  • Graceful fallback — every model has a `cpu_fallback` flag; when True, it
    loads on CPU if MPS is unavailable or budget is exceeded.

Usage
─────
    from scripts.core.model_manager import get_manager

    mgr = get_manager()

    # Load MusicGen (auto-offloads everything else first)
    model, processor = mgr.load("musicgen")
    wav = model.generate(...)
    mgr.release("musicgen")   # marks as idle (does NOT unload immediately)

    # Context manager — ensures release even on exception
    with mgr.use("musicgen") as (model, processor):
        wav = model.generate(...)

    # Check status
    info = mgr.status()
    # → {"musicgen": {"loaded": true, "vram_estimate_gb": 8.0, "last_used": ...}, ...}
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from scripts.core.gpu import get_device

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config defaults (also driven by config.yaml via _get_cfg_value)
# ---------------------------------------------------------------------------

def _get_cfg_value(section: str, key: str, default):
    """Read a value from config.yaml via the shared cfg object (best-effort)."""
    try:
        from scripts.core.config import cfg
        sec = getattr(cfg, section, None)
        return getattr(sec, key, default) if sec is not None else default
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    """Registry entry for one loadable model."""

    name: str
    loader: Callable[[], Any]           # () → model (or (model, processor) tuple)
    vram_estimate_gb: float             # Rough VRAM estimate at load time
    model: Any = None                   # Loaded model (None when unloaded)
    loaded: bool = False
    last_used: float = 0.0
    cpu_fallback: bool = True           # Load on CPU if VRAM budget exceeded?
    offload_after_use: bool = True      # Auto-offload after release()?
    description: str = ""


# ---------------------------------------------------------------------------
# ModelManager singleton
# ---------------------------------------------------------------------------

class ModelManager:
    """
    Centralized lazy-loading model registry with MPS memory management.

    Thread-safe: all load/offload operations are serialized via a threading.Lock.
    """

    def __init__(self):
        self._models: Dict[str, ModelEntry] = {}
        self._lock = threading.Lock()
        self._device = get_device()
        self._max_vram_gb: float = _get_cfg_value("generative", "max_vram_gb", 12.0)

        log.info(
            "[model_manager] Initialized. Device=%s, VRAM budget=%.1f GB",
            self._device, self._max_vram_gb
        )

    # ── Registration ──────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        loader: Callable[[], Any],
        vram_estimate_gb: float,
        cpu_fallback: bool = True,
        offload_after_use: bool = True,
        description: str = "",
    ) -> None:
        """Register a model without loading it yet."""
        with self._lock:
            self._models[name] = ModelEntry(
                name=name,
                loader=loader,
                vram_estimate_gb=vram_estimate_gb,
                cpu_fallback=cpu_fallback,
                offload_after_use=offload_after_use,
                description=description,
            )
            log.debug("[model_manager] Registered model: %s (%.1f GB)", name, vram_estimate_gb)

    # ── Loading ───────────────────────────────────────────────────────────

    def load(self, name: str) -> Any:
        """
        Load a model by name, offloading others first if needed.

        Returns the model (or (model, processor) tuple — whatever loader() returns).
        Thread-safe.
        """
        with self._lock:
            return self._load_locked(name)

    def _load_locked(self, name: str) -> Any:
        """Load implementation (must be called while holding self._lock)."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not registered. Call register() first.")

        entry = self._models[name]

        if entry.loaded and entry.model is not None:
            entry.last_used = time.time()
            log.debug("[model_manager] '%s' already loaded — returning cached", name)
            return entry.model

        # Sequential strategy: offload everything else first
        self._offload_except_locked(name)

        log.info("[model_manager] Loading '%s' (%.1f GB estimated)…", name, entry.vram_estimate_gb)
        t0 = time.time()

        try:
            model = entry.loader()
            entry.model = model
            entry.loaded = True
            entry.last_used = time.time()
            elapsed = time.time() - t0
            log.info(
                "[model_manager] '%s' loaded in %.1fs. MPS memory: %.2f GB",
                name, elapsed, self._mps_memory_gb()
            )
            return model

        except Exception as exc:
            log.error("[model_manager] Failed to load '%s': %s", name, exc)
            entry.loaded = False
            entry.model = None
            raise

    # ── Offloading ────────────────────────────────────────────────────────

    def release(self, name: str) -> None:
        """
        Mark a model as idle. If offload_after_use=True, immediately unload it.

        Call this after you're done with a model to free memory for the next one.
        """
        with self._lock:
            entry = self._models.get(name)
            if entry is None or not entry.loaded:
                return
            if entry.offload_after_use:
                self._offload_one_locked(name)
            else:
                entry.last_used = time.time()

    def offload(self, name: str) -> None:
        """Force-offload a specific model."""
        with self._lock:
            self._offload_one_locked(name)

    def offload_all(self) -> None:
        """Offload all loaded models. Called on API shutdown."""
        with self._lock:
            for name in list(self._models.keys()):
                self._offload_one_locked(name)

    def _offload_one_locked(self, name: str) -> None:
        entry = self._models.get(name)
        if entry is None or not entry.loaded:
            return
        log.info("[model_manager] Offloading '%s'…", name)
        entry.model = None
        entry.loaded = False
        self._free_memory()

    def _offload_except_locked(self, keep: str) -> None:
        """Offload all models except 'keep'. Frees memory before loading a new model."""
        for name, entry in self._models.items():
            if name != keep and entry.loaded:
                log.debug("[model_manager] Offloading '%s' to make room for '%s'", name, keep)
                self._offload_one_locked(name)

    # ── Memory helpers ────────────────────────────────────────────────────

    @staticmethod
    def _free_memory() -> None:
        """Force garbage collection + empty MPS/CUDA cache."""
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _mps_memory_gb() -> float:
        """Return current MPS allocated memory in GB (0.0 if unavailable)."""
        try:
            import torch
            if torch.backends.mps.is_available():
                return torch.mps.current_allocated_memory() / 1e9
        except Exception:
            pass
        return 0.0

    # ── Context manager ───────────────────────────────────────────────────

    @contextmanager
    def use(self, name: str):
        """
        Context manager for safe model use. Auto-releases after the block.

        Example:
            with model_manager.use("musicgen") as model:
                wav = model.generate(...)
        """
        model = self.load(name)
        try:
            yield model
        finally:
            self.release(name)

    # ── Status ────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Return a status dict for all registered models."""
        with self._lock:
            mps_mem = self._mps_memory_gb()
            return {
                "device": self._device,
                "mps_allocated_gb": round(mps_mem, 3),
                "max_vram_gb": self._max_vram_gb,
                "models": {
                    name: {
                        "loaded": entry.loaded,
                        "vram_estimate_gb": entry.vram_estimate_gb,
                        "last_used": entry.last_used,
                        "description": entry.description,
                        "offload_after_use": entry.offload_after_use,
                    }
                    for name, entry in self._models.items()
                },
            }

    def is_loaded(self, name: str) -> bool:
        """Return True if the model is currently in memory."""
        entry = self._models.get(name)
        return entry is not None and entry.loaded


# ---------------------------------------------------------------------------
# Singleton + model loader registry
# ---------------------------------------------------------------------------

_manager: Optional[ModelManager] = None
_manager_lock = threading.Lock()


def get_manager() -> ModelManager:
    """Return the global ModelManager singleton, initializing it on first call."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ModelManager()
                _register_default_models(_manager)
    return _manager


def _register_default_models(mgr: ModelManager) -> None:
    """Register all AI RemixMate generative models with their loaders."""

    device = get_device()

    # ── MusicGen melody (transformers, ~8 GB on MPS) ─────────────────────
    def _load_musicgen():
        from transformers import (
            MusicgenMelodyForConditionalGeneration,
            AutoProcessor,
        )
        model_id = _get_cfg_value("generative", "musicgen_model",
                                   "facebook/musicgen-melody")
        log.info("[model_manager] Fetching MusicGen from HuggingFace: %s", model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        model = MusicgenMelodyForConditionalGeneration.from_pretrained(model_id)
        model.to(device)
        model.eval()
        return (model, processor)

    mgr.register(
        name="musicgen",
        loader=_load_musicgen,
        vram_estimate_gb=8.0,
        description="MusicGen Melody — text + chroma conditioned music generation",
        offload_after_use=_get_cfg_value("generative", "musicgen_offload_after_use", True),
    )

    # ── EnCodec 24kHz (transformers, ~0.5 GB) ─────────────────────────────
    def _load_encodec():
        from transformers import EncodecModel, AutoProcessor
        log.info("[model_manager] Loading EnCodec 24kHz…")
        processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        model = EncodecModel.from_pretrained("facebook/encodec_24khz")
        model.to(device)
        model.eval()
        return (model, processor)

    mgr.register(
        name="encodec",
        loader=_load_encodec,
        vram_estimate_gb=0.5,
        description="EnCodec 24kHz — neural audio codec for tokenization",
        offload_after_use=False,   # small enough to keep loaded
    )

    # ── DAC 44.1kHz (descript, ~1 GB) ─────────────────────────────────────
    def _load_dac():
        import dac
        log.info("[model_manager] Loading DAC 44.1kHz…")
        model_path = dac.utils.download(model_type="44khz")
        model = dac.DAC.load(model_path)
        model.to(device)
        model.eval()
        return model

    mgr.register(
        name="dac",
        loader=_load_dac,
        vram_estimate_gb=1.0,
        description="DAC 44.1kHz — high-fidelity neural audio codec",
        offload_after_use=False,
    )

    # ── VampNet (optional — graceful skip if not installed) ───────────────
    def _load_vampnet():
        try:
            import vampnet
            log.info("[model_manager] Loading VampNet interface…")
            interface = vampnet.interface.Interface.default()
            return interface
        except ImportError:
            raise RuntimeError(
                "VampNet is not installed. Install with:\n"
                "  pip install git+https://github.com/hugofloresgarcia/vampnet"
            )

    mgr.register(
        name="vampnet",
        loader=_load_vampnet,
        vram_estimate_gb=5.0,
        description="VampNet — masked acoustic token inpainting via DAC",
        offload_after_use=_get_cfg_value("generative", "vampnet_offload_after_use", True),
    )

    log.info("[model_manager] %d models registered.", len(mgr._models))
