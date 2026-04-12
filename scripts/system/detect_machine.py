"""
scripts/system/detect_machine.py — Machine profile detection utility.

Emits a structured JSON object describing the current machine's hardware
capabilities, and recommends appropriate processing parameters.

Output schema (MachineProfile):
    hostname            str      — machine name
    platform            str      — "macOS 14.5" / "Linux 22.04" / "Windows 11"
    cpu_model           str      — processor brand string
    cpu_cores_physical  int      — physical core count
    cpu_cores_logical   int      — logical core count (with HT)
    ram_gb              float    — total RAM in gigabytes
    gpu_backend         str      — "cuda" | "mps" | "cpu"
    gpu_name            str|null — GPU model name if detectable
    gpu_vram_gb         float|null
    demucs_device       str      — recommended -d flag for Demucs
    recommended_batch_size int   — conservative batch size for stem splits
    tier                str      — "low" | "mid" | "high" | "pro"
    torch_version       str|null
    python_version      str

Run directly:
    python -m scripts.system.detect_machine
    python -m scripts.system.detect_machine --json-only

Imported by FastAPI events router for heartbeat data.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MachineProfile:
    hostname: str
    platform: str
    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_gb: float
    gpu_backend: str                   # "cuda" | "mps" | "cpu"
    gpu_name: Optional[str]
    gpu_vram_gb: Optional[float]
    demucs_device: str
    recommended_batch_size: int
    tier: str                          # "low" | "mid" | "high" | "pro"
    torch_version: Optional[str]
    python_version: str
    # Derived extras
    is_apple_silicon: bool = field(default=False)
    cuda_version: Optional[str] = field(default=None)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _platform_string() -> str:
    sys_name = platform.system()
    if sys_name == "Darwin":
        ver = platform.mac_ver()[0]
        return f"macOS {ver}" if ver else "macOS"
    elif sys_name == "Linux":
        try:
            import distro  # type: ignore[import]
            return f"{distro.name()} {distro.version()}"
        except ImportError:
            return f"Linux {platform.release()}"
    elif sys_name == "Windows":
        return f"Windows {platform.version()}"
    return sys_name


def _cpu_model() -> str:
    sys_name = platform.system()
    if sys_name == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).decode().strip()
            if out:
                return out
        except Exception:
            pass
        # Fallback for Apple Silicon (no brand_string sysctl)
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPHardwareDataType"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode()
            for line in out.splitlines():
                if "Chip" in line or "Processor Name" in line:
                    return line.split(":", 1)[-1].strip()
        except Exception:
            pass
    elif sys_name == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[-1].strip()
        except Exception:
            pass
    elif sys_name == "Windows":
        try:
            import winreg  # type: ignore[import]
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except Exception:
            pass
    return platform.processor() or "Unknown CPU"


def _cpu_cores() -> tuple[int, int]:
    """Return (physical_cores, logical_cores)."""
    try:
        import psutil  # type: ignore[import]
        physical = psutil.cpu_count(logical=False) or 1
        logical  = psutil.cpu_count(logical=True)  or 1
        return physical, logical
    except ImportError:
        import os
        logical = os.cpu_count() or 1
        return max(1, logical // 2), logical


def _ram_gb() -> float:
    try:
        import psutil  # type: ignore[import]
        return round(psutil.virtual_memory().total / 1_073_741_824, 1)
    except ImportError:
        # macOS fallback
        if platform.system() == "Darwin":
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"],
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                ).decode().strip()
                return round(int(out) / 1_073_741_824, 1)
            except Exception:
                pass
        return 0.0


def _gpu_info() -> tuple[str, Optional[str], Optional[float], Optional[str]]:
    """
    Returns (backend, gpu_name, vram_gb, cuda_version).
    Delegates to scripts.core.gpu when available; otherwise re-implements detection.
    """
    try:
        # Prefer existing gpu.py — single source of truth
        from scripts.core.gpu import get_device, _import_torch  # type: ignore[import]
        backend = get_device()
        torch = _import_torch()

        gpu_name: Optional[str] = None
        vram_gb: Optional[float] = None
        cuda_ver: Optional[str] = None

        if torch is not None:
            if backend == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                vram_gb = round(props.total_memory / 1_073_741_824, 1)
                cuda_ver = torch.version.cuda  # type: ignore[attr-defined]
            elif backend == "mps":
                # Apple Silicon — try system_profiler for GPU name
                try:
                    out = subprocess.check_output(
                        ["system_profiler", "SPHardwareDataType"],
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                    ).decode()
                    for line in out.splitlines():
                        if "Chip" in line:
                            gpu_name = line.split(":", 1)[-1].strip()
                            break
                except Exception:
                    gpu_name = "Apple Silicon GPU"

        return backend, gpu_name, vram_gb, cuda_ver

    except ImportError:
        # gpu.py not available — re-implement detection inline
        return _gpu_info_fallback()


def _gpu_info_fallback() -> tuple[str, Optional[str], Optional[float], Optional[str]]:
    """Fallback when scripts.core.gpu is not importable."""
    try:
        import torch  # type: ignore[import]
        if platform.machine() == "arm64" and platform.system() == "Darwin":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps", "Apple Silicon GPU", None, None
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram = round(props.total_memory / 1_073_741_824, 1)
            return "cuda", name, vram, torch.version.cuda  # type: ignore[attr-defined]
    except ImportError:
        pass
    return "cpu", None, None, None


def _torch_version() -> Optional[str]:
    try:
        import torch  # type: ignore[import]
        return torch.__version__
    except ImportError:
        return None


def _tier(ram_gb: float, gpu_backend: str, cpu_cores: int) -> str:
    """
    Classify the machine into a processing tier.
    Used by the frontend to show capability badges and set defaults.

    Tiers:
        pro   — CUDA GPU with ≥8 GB VRAM, or MPS on M-series Pro/Max/Ultra
        high  — MPS (any), or CUDA with <8 GB VRAM, or ≥32 GB RAM + 8 cores
        mid   — ≥16 GB RAM + 4 cores (CPU only)
        low   — everything else
    """
    if gpu_backend == "cuda":
        return "pro"
    if gpu_backend == "mps":
        # Distinguish M1 base vs Pro/Max/Ultra by RAM as proxy
        return "pro" if ram_gb >= 32 else "high"
    if ram_gb >= 32 and cpu_cores >= 8:
        return "high"
    if ram_gb >= 16 and cpu_cores >= 4:
        return "mid"
    return "low"


def _recommended_batch_size(tier: str) -> int:
    return {"pro": 8, "high": 4, "mid": 2, "low": 1}.get(tier, 1)


# ---------------------------------------------------------------------------
# Main detection
# ---------------------------------------------------------------------------

def detect() -> MachineProfile:
    """Run full machine detection and return a MachineProfile."""
    hostname    = socket.gethostname()
    plat        = _platform_string()
    cpu_model   = _cpu_model()
    phys, logic = _cpu_cores()
    ram         = _ram_gb()
    backend, gpu_name, vram, cuda_ver = _gpu_info()
    torch_ver   = _torch_version()
    py_ver      = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    tier         = _tier(ram, backend, phys)
    batch        = _recommended_batch_size(tier)
    is_apple_si  = platform.system() == "Darwin" and platform.machine() == "arm64"

    return MachineProfile(
        hostname            = hostname,
        platform            = plat,
        cpu_model           = cpu_model,
        cpu_cores_physical  = phys,
        cpu_cores_logical   = logic,
        ram_gb              = ram,
        gpu_backend         = backend,
        gpu_name            = gpu_name,
        gpu_vram_gb         = vram,
        demucs_device       = backend,
        recommended_batch_size = batch,
        tier                = tier,
        torch_version       = torch_ver,
        python_version      = py_ver,
        is_apple_silicon    = is_apple_si,
        cuda_version        = cuda_ver,
    )


def to_dict(profile: MachineProfile) -> dict:
    return asdict(profile)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect machine hardware profile for AI RemixMate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--json-only",
        action="store_true",
        help="Print only the raw JSON (no pretty-print header)",
    )
    p.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON (default: True)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    profile = detect()
    data = to_dict(profile)

    if args.json_only:
        print(json.dumps(data))
    else:
        indent = 2 if args.pretty else None
        print("\n── AI RemixMate · Machine Profile ──────────────────────")
        print(json.dumps(data, indent=indent))
        print()

        # Human-readable summary
        print(f"  Host       : {profile.hostname}")
        print(f"  Platform   : {profile.platform}")
        print(f"  CPU        : {profile.cpu_model} ({profile.cpu_cores_physical}p / {profile.cpu_cores_logical}L cores)")
        print(f"  RAM        : {profile.ram_gb} GB")
        print(f"  GPU        : {profile.gpu_backend.upper()}", end="")
        if profile.gpu_name:
            print(f" · {profile.gpu_name}", end="")
        if profile.gpu_vram_gb:
            print(f" · {profile.gpu_vram_gb} GB VRAM", end="")
        print()
        print(f"  Tier       : {profile.tier.upper()}")
        print(f"  Batch size : {profile.recommended_batch_size}")
        print(f"  Demucs -d  : {profile.demucs_device}")
        if profile.torch_version:
            print(f"  PyTorch    : {profile.torch_version}")
        print("────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
