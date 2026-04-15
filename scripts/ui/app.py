"""
scripts/ui/app.py — AI RemixMate Streamlit Frontend

Talks to the FastAPI backend. Server-side API host resolution order:
  1. REMIXMATE_API_URL env var (full URL, e.g. http://api:8000 in Docker Compose)
  2. localhost:8000 (default for local single-process or dual-process runs)

Browser-facing links (audio playback, direct file downloads) always use the
machine's LAN IP so they work from phones/tablets on the same network.

Run:
    streamlit run scripts/ui/app.py
    # or in Docker: REMIXMATE_API_URL=http://api:8000 streamlit run scripts/ui/app.py
"""

from __future__ import annotations

import os
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import streamlit as st
import streamlit.components.v1 as st_components

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _api_base_url() -> str:
    """
    Resolve the server-side API base URL.

    Resolution order:
      1. REMIXMATE_API_URL env var — use verbatim (Docker Compose sets this to
         http://api:8000 so the Streamlit container can reach the API container
         via Docker's internal DNS).
      2. Fall back to http(s)://localhost:8000 for local runs.

    Never uses the LAN IP for server-side calls — localhost always resolves
    correctly whether we're in single-process or dual-process local mode.
    """
    env_url = os.environ.get("REMIXMATE_API_URL", "").strip().rstrip("/")
    if env_url:
        return env_url
    proto = "https" if _HTTPS else "http"
    return f"{proto}://localhost:8000"

def _lan_ip() -> str:
    """Return this machine's LAN IP address (for browser-facing links)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

def _detect_https() -> bool:
    """Detect if we're running in HTTPS mode by checking for certs."""
    cert_dir = Path(__file__).parents[2] / "certs"
    return (cert_dir / "cert.pem").exists() and (cert_dir / "key.pem").exists()

_LAN_IP = _lan_ip()
_HTTPS = os.environ.get("REMIXMATE_HTTPS", "").lower() in ("1", "true", "yes")
_PROTO = "https" if _HTTPS else "http"
# Server-side API URL — respects REMIXMATE_API_URL for Docker/multi-host setups
API = _api_base_url()
# Browser-visible API URL (used in embedded HTML/links that the phone browser loads)
# Always uses LAN IP so links work from phones/tablets; never affected by REMIXMATE_API_URL
API_PUBLIC = f"{_PROTO}://{_LAN_IP}:8000"
PROJECT_ROOT = Path(__file__).parents[2]

st.set_page_config(
    page_title="AI RemixMate",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Visual theme
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    /* ══════════════════════════════════════════════════════════════════
       🧠 ADHD-CODED KEYFRAME ANIMATIONS
       — Everything moves. Nothing is dead. Dopamine on every pixel.
    ══════════════════════════════════════════════════════════════════ */

    /* Page content entrance — faster, snappier */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Equalizer bar pulse */
    @keyframes eqBar {
        0%, 100% { transform: scaleY(0.15); }
        50%       { transform: scaleY(1); }
    }

    /* Vinyl record rotation */
    @keyframes vinylSpin {
        from { transform: rotate(0deg); }
        to   { transform: rotate(360deg); }
    }

    /* ── NEW: Holographic rainbow shimmer ── */
    @keyframes holoShimmer {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ── NEW: Breathing glow for ambient vibe ── */
    @keyframes breathe {
        0%, 100% { opacity: 0.4; filter: blur(60px); }
        50%       { opacity: 0.7; filter: blur(80px); }
    }

    /* ── NEW: Neon flicker for accents ── */
    @keyframes neonFlicker {
        0%, 19%, 21%, 23%, 25%, 54%, 56%, 100% { text-shadow: 0 0 7px #c084fc, 0 0 10px #c084fc, 0 0 21px #c084fc, 0 0 42px #7c3aed, 0 0 82px #7c3aed; }
        20%, 24%, 55% { text-shadow: none; }
    }

    /* ── NEW: Liquid morph for backgrounds ── */
    @keyframes liquidMorph {
        0%   { border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%; }
        25%  { border-radius: 30% 60% 70% 40% / 50% 60% 30% 60%; }
        50%  { border-radius: 50% 60% 30% 60% / 30% 40% 70% 60%; }
        75%  { border-radius: 60% 40% 60% 40% / 60% 30% 50% 70%; }
        100% { border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%; }
    }

    /* ── NEW: Rainbow border for active elements ── */
    @keyframes rainbowBorder {
        0%   { border-color: #c084fc; }
        25%  { border-color: #818cf8; }
        50%  { border-color: #60a5fa; }
        75%  { border-color: #a78bfa; }
        100% { border-color: #c084fc; }
    }

    /* ── NEW: Count-up animation for numbers ── */
    @keyframes countPop {
        0%   { transform: scale(0.5) translateY(10px); opacity: 0; }
        60%  { transform: scale(1.15); opacity: 1; }
        100% { transform: scale(1) translateY(0); opacity: 1; }
    }

    /* ── NEW: Magnetic pull effect ── */
    @keyframes magnetPull {
        0%   { transform: translateY(0); }
        50%  { transform: translateY(-2px); }
        100% { transform: translateY(0); }
    }

    /* Animated gradient shift on titles */
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Button glow breathing */
    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 0 18px rgba(124,58,237,0.35), 0 2px 12px rgba(0,0,0,0.4); }
        50%       { box-shadow: 0 0 42px rgba(124,58,237,0.75), 0 0 70px rgba(192,132,252,0.25), 0 2px 12px rgba(0,0,0,0.4); }
    }

    /* Progress bar shimmer */
    @keyframes shimmer {
        0%   { background-position: -300% 0; }
        100% { background-position: 300% 0; }
    }

    /* Soft float bob for icons */
    @keyframes floatBob {
        0%, 100% { transform: translateY(0px); }
        50%       { transform: translateY(-5px); }
    }

    /* Metric card pop-in */
    @keyframes cardIn {
        from { opacity: 0; transform: scale(0.92) translateY(10px); }
        to   { opacity: 1; transform: scale(1) translateY(0); }
    }

    /* Alert slide-in */
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-12px); }
        to   { opacity: 1; transform: translateX(0); }
    }

    /* Waveform scan line */
    @keyframes scanLine {
        0%   { left: -5%; opacity: 0.6; }
        50%  { opacity: 1; }
        100% { left: 105%; opacity: 0.6; }
    }

    /* Orb ambient pulse */
    @keyframes orbPulse {
        0%, 100% { opacity: 0.06; transform: scale(1); }
        50%       { opacity: 0.12; transform: scale(1.08); }
    }

    /* ══════════════════════════════════════════════════════════════════
       BASE — ADHD CODED: alive background, reactive everything
    ══════════════════════════════════════════════════════════════════ */
    html, body, [data-testid="stAppViewContainer"] {
        background: #06060a;
        color: #e8e8f0;
        font-family: 'Inter', 'SF Pro Display', sans-serif;
    }

    /* ── Animated gradient mesh background ── */
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 20% 50%, rgba(124,58,237,0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(79,70,229,0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 80%, rgba(192,132,252,0.04) 0%, transparent 50%),
            #06060a !important;
    }

    /* ── Living ambient orbs — multiple, different speeds ── */
    [data-testid="stAppViewContainer"]::before,
    [data-testid="stAppViewContainer"]::after {
        content: '';
        position: fixed;
        pointer-events: none;
        z-index: 0;
    }
    [data-testid="stAppViewContainer"]::before {
        width: 500px; height: 500px;
        background: radial-gradient(circle, rgba(124,58,237,0.15) 0%, transparent 70%);
        top: -150px; right: -100px;
        animation: breathe 6s ease-in-out infinite, liquidMorph 15s ease-in-out infinite;
    }
    [data-testid="stAppViewContainer"]::after {
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(236,72,153,0.08) 0%, transparent 70%);
        bottom: -100px; left: -50px;
        animation: breathe 8s ease-in-out infinite 2s, liquidMorph 20s ease-in-out infinite reverse;
    }

    /* ── Subtle grid pattern overlay ── */
    [data-testid="stMain"]::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image:
            linear-gradient(rgba(124,58,237,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(124,58,237,0.03) 1px, transparent 1px);
        background-size: 60px 60px;
        pointer-events: none;
        z-index: 0;
        mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 40%, transparent 100%);
        -webkit-mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 40%, transparent 100%);
    }

    /* Page content fade-up on render */
    section[data-testid="stMain"] > div {
        animation: fadeUp 0.4s cubic-bezier(0.22,1,0.36,1) both;
        position: relative;
        z-index: 1;
    }

    /* ══════════════════════════════════════════════════════════════════
       SIDEBAR — ADHD: Glass morphism + neon accents + hover glow
    ══════════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(8,8,14,0.95) 0%, rgba(12,12,20,0.98) 100%) !important;
        border-right: 1px solid rgba(124,58,237,0.15) !important;
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
    }
    /* Sidebar ambient glow at top */
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 200px;
        background: radial-gradient(ellipse at 50% 0%, rgba(124,58,237,0.12) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.88rem;
        padding: 8px 0;
        color: #6868a0;
        transition: all 0.25s cubic-bezier(0.22,1,0.36,1);
        cursor: pointer;
        letter-spacing: 0.01em;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        color: #c084fc;
        text-shadow: 0 0 20px rgba(192,132,252,0.6);
        transform: translateX(4px);
    }
    [data-testid="stSidebar"] [aria-checked="true"] ~ div label {
        color: #c084fc !important;
        text-shadow: 0 0 20px rgba(192,132,252,0.7);
        font-weight: 600 !important;
    }

    /* ══════════════════════════════════════════════════════════════════
       HEADERS — ADHD: Neon gradient + glow + alive text
    ══════════════════════════════════════════════════════════════════ */
    h1 {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        background: linear-gradient(270deg, #c084fc, #818cf8, #60a5fa, #ec4899, #a78bfa, #c084fc);
        background-size: 600% 600%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
        animation: holoShimmer 4s ease infinite;
        filter: drop-shadow(0 0 12px rgba(192,132,252,0.15));
    }
    h2 {
        color: #d4d4f0 !important;
        font-weight: 700 !important;
        position: relative;
        display: inline-block;
    }
    h2::after {
        content: '';
        position: absolute;
        bottom: -4px;
        left: 0;
        width: 40px;
        height: 3px;
        background: linear-gradient(90deg, #c084fc, transparent);
        border-radius: 2px;
    }
    h3 {
        color: #b8b8d0 !important;
        font-weight: 600 !important;
    }

    /* ══════════════════════════════════════════════════════════════════
       CARDS — ADHD: Deep glass + rainbow hover border + 3D lift
    ══════════════════════════════════════════════════════════════════ */
    [data-testid="stExpander"],
    [data-testid="stForm"] {
        background: rgba(12,12,22,0.8) !important;
        backdrop-filter: blur(20px) saturate(1.5) !important;
        -webkit-backdrop-filter: blur(20px) saturate(1.5) !important;
        border: 1px solid rgba(42,42,80,0.5) !important;
        border-radius: 16px !important;
        transition: all 0.3s cubic-bezier(0.22,1,0.36,1);
        position: relative;
        overflow: hidden;
    }
    [data-testid="stExpander"]::before,
    [data-testid="stForm"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(192,132,252,0.3), transparent);
        opacity: 0;
        transition: opacity 0.3s;
    }
    [data-testid="stExpander"]:hover::before,
    [data-testid="stForm"]:hover::before { opacity: 1; }
    [data-testid="stExpander"]:hover,
    [data-testid="stForm"]:hover {
        border-color: rgba(192,132,252,0.25) !important;
        box-shadow: 0 8px 40px rgba(124,58,237,0.15), 0 0 0 1px rgba(192,132,252,0.08) !important;
        transform: translateY(-3px);
    }

    /* ══════════════════════════════════════════════════════════════════
       METRIC TILES — ADHD: Neon glow + pop-in + reactive hover
    ══════════════════════════════════════════════════════════════════ */
    [data-testid="stMetric"] {
        background: rgba(12,12,22,0.85) !important;
        backdrop-filter: blur(16px) saturate(1.4) !important;
        border: 1px solid rgba(42,42,80,0.4) !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        animation: cardIn 0.45s cubic-bezier(0.22,1,0.36,1) both;
        transition: all 0.28s cubic-bezier(0.22,1,0.36,1);
        position: relative;
        overflow: hidden;
    }
    /* Top accent line */
    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0; left: 20%; right: 20%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #c084fc, transparent);
        opacity: 0;
        transition: opacity 0.3s, left 0.3s, right 0.3s;
    }
    [data-testid="stMetric"]:hover::before {
        opacity: 1;
        left: 10%;
        right: 10%;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px) scale(1.02);
        border-color: rgba(192,132,252,0.3) !important;
        box-shadow:
            0 12px 40px rgba(124,58,237,0.2),
            0 0 20px rgba(192,132,252,0.08),
            inset 0 1px 0 rgba(192,132,252,0.1) !important;
    }
    [data-testid="stMetricLabel"]  {
        color: #5858a0 !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    [data-testid="stMetricValue"]  {
        color: #f0f0ff !important;
        font-weight: 800 !important;
        font-size: 1.5rem !important;
        animation: countPop 0.5s cubic-bezier(0.22,1,0.36,1) both;
    }
    [data-testid="stMetricDelta"]  { color: #c084fc !important; font-weight: 600; }

    /* Staggered card entrances — faster cascade */
    [data-testid="stMetric"]:nth-child(1) { animation-delay: 0.03s; }
    [data-testid="stMetric"]:nth-child(2) { animation-delay: 0.08s; }
    [data-testid="stMetric"]:nth-child(3) { animation-delay: 0.13s; }
    [data-testid="stMetric"]:nth-child(4) { animation-delay: 0.18s; }
    [data-testid="stMetric"]:nth-child(5) { animation-delay: 0.23s; }

    /* ══════════════════════════════════════════════════════════════════
       BUTTONS — ADHD: Glow pulse + shimmer sweep + satisfying press
    ══════════════════════════════════════════════════════════════════ */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 50%, #4f46e5 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.7rem 1.6rem !important;
        animation: glowPulse 2.5s ease-in-out infinite;
        transition: all 0.2s cubic-bezier(0.22,1,0.36,1);
        position: relative;
        overflow: hidden;
        letter-spacing: 0.02em;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
    .stButton > button[kind="primary"]::after {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 60%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent);
        transition: left 0.5s ease;
    }
    .stButton > button[kind="primary"]:hover::after { left: 150%; }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.01);
        filter: brightness(1.1);
    }
    .stButton > button[kind="primary"]:active {
        transform: translateY(0) scale(0.99);
    }

    .stButton > button[kind="secondary"] {
        background: rgba(22,22,32,0.7) !important;
        border: 1px solid rgba(58,58,92,0.9) !important;
        color: #9090c0 !important;
        border-radius: 10px !important;
        transition: border-color 0.2s, color 0.2s, box-shadow 0.2s;
        backdrop-filter: blur(8px);
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: rgba(192,132,252,0.6) !important;
        color: #c084fc !important;
        box-shadow: 0 0 18px rgba(192,132,252,0.15);
    }

    /* ══════════════════════════════════════════════════════════════════
       FORM INPUTS
    ══════════════════════════════════════════════════════════════════ */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stMultiSelect"] > div > div {
        background: rgba(18,18,28,0.85) !important;
        border-color: rgba(42,42,72,0.9) !important;
        border-radius: 10px !important;
        color: #e0e0f0 !important;
        transition: border-color 0.2s;
    }
    [data-testid="stSelectbox"] > div > div:focus-within,
    [data-testid="stMultiSelect"] > div > div:focus-within {
        border-color: rgba(192,132,252,0.5) !important;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: rgba(18,18,28,0.85) !important;
        border-color: rgba(42,42,72,0.9) !important;
        border-radius: 10px !important;
        color: #e0e0f0 !important;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: rgba(192,132,252,0.5) !important;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
    }

    /* ══════════════════════════════════════════════════════════════════
       SLIDERS
    ══════════════════════════════════════════════════════════════════ */
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background: #c084fc !important;
        box-shadow: 0 0 10px rgba(192,132,252,0.7), 0 0 20px rgba(192,132,252,0.3) !important;
        transition: box-shadow 0.2s;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] div[class*="Track"] {
        background: rgba(42,42,72,0.8) !important;
    }

    /* ══════════════════════════════════════════════════════════════════
       RADIO
    ══════════════════════════════════════════════════════════════════ */
    [data-testid="stRadio"] label span { color: #9090c0 !important; }
    [data-testid="stRadio"] [aria-checked="true"] + div span {
        color: #c084fc !important;
        text-shadow: 0 0 10px rgba(192,132,252,0.4);
    }

    /* ══════════════════════════════════════════════════════════════════
       PROGRESS BAR — ADHD: Rainbow shimmer + glow trail
    ══════════════════════════════════════════════════════════════════ */
    [data-testid="stProgressBar"] > div {
        background: rgba(20,20,35,0.6) !important;
        border-radius: 8px !important;
        overflow: hidden;
    }
    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #7c3aed, #a855f7, #c084fc, #ec4899, #a855f7, #7c3aed) !important;
        background-size: 400% 100% !important;
        border-radius: 8px !important;
        animation: shimmer 1.5s linear infinite !important;
        box-shadow: 0 0 16px rgba(192,132,252,0.4), 0 0 40px rgba(124,58,237,0.15) !important;
        position: relative;
    }

    /* ══════════════════════════════════════════════════════════════════
       AUDIO PLAYER
    ══════════════════════════════════════════════════════════════════ */
    audio {
        width: 100%;
        border-radius: 10px;
        filter: invert(1) hue-rotate(180deg) brightness(0.85);
        transition: box-shadow 0.3s;
    }
    audio:hover {
        box-shadow: 0 0 24px rgba(192,132,252,0.2);
    }

    /* ══════════════════════════════════════════════════════════════════
       ALERTS
    ══════════════════════════════════════════════════════════════════ */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
        border-left-width: 3px !important;
        animation: slideIn 0.3s cubic-bezier(0.22,1,0.36,1) both;
        backdrop-filter: blur(8px);
    }

    /* ══════════════════════════════════════════════════════════════════
       MISC
    ══════════════════════════════════════════════════════════════════ */
    hr { border-color: rgba(30,30,50,0.8) !important; }

    [data-testid="stCode"] {
        background: rgba(6,6,14,0.9) !important;
        border: 1px solid rgba(30,30,50,0.9) !important;
        border-radius: 10px !important;
    }

    [data-testid="stCaptionContainer"] { color: #44446a !important; }

    [data-testid="stSpinner"] { color: #c084fc !important; }

    /* Status widget */
    [data-testid="stStatus"] {
        background: rgba(14,14,22,0.9) !important;
        border-color: rgba(124,58,237,0.3) !important;
        border-radius: 12px !important;
    }

    /* Checkbox */
    [data-testid="stCheckbox"] label span { color: #9090c0 !important; transition: color 0.15s; }
    [data-testid="stCheckbox"] label:hover span { color: #c084fc !important; }

    /* Number input */
    [data-testid="stNumberInput"] input {
        background: rgba(18,18,28,0.85) !important;
        border-color: rgba(42,42,72,0.9) !important;
        border-radius: 10px !important;
        color: #e0e0f0 !important;
    }

    /* ══════════════════════════════════════════════════════════════════
       SELECTBOX / MULTISELECT DROPDOWN FIX
       Prevents the dropdown list from being clipped by parent overflow.
    ══════════════════════════════════════════════════════════════════ */

    /* Ensure dropdown popover can escape its container */
    div[data-baseweb="select"] {
        overflow: visible !important;
    }
    div[data-baseweb="popover"] {
        z-index: 9999 !important;
        max-height: 50vh !important;
    }

    /* Make the dropdown list scrollable with a generous max-height */
    ul[role="listbox"] {
        max-height: 45vh !important;
        overflow-y: auto !important;
    }

    /* Ensure parent containers don't clip the dropdown */
    .stSelectbox, .stMultiSelect {
        overflow: visible !important;
    }
    .stSelectbox > div, .stMultiSelect > div {
        overflow: visible !important;
    }

    /* Fix for columns and expanders that clip dropdowns */
    div[data-testid="column"],
    div[data-testid="stVerticalBlock"],
    div[data-testid="stHorizontalBlock"],
    section[data-testid="stSidebar"],
    .element-container {
        overflow: visible !important;
    }

    /* Main content area: allow dropdowns to overflow */
    .main .block-container {
        overflow: visible !important;
    }

    /* Song picker search box styling */
    .song-picker-search input {
        border-color: rgba(124,58,237,0.5) !important;
    }

    /* ══════════════════════════════════════════════════════════════════
       ENHANCED ANIMATIONS & MICRO-INTERACTIONS
       — Smooth, satisfying, frustration-proof UX
    ══════════════════════════════════════════════════════════════════ */

    /* Ripple effect on all buttons */
    @keyframes ripple {
        0%   { transform: translate(-50%,-50%) scale(0); opacity: 0.5; }
        100% { transform: translate(-50%,-50%) scale(4); opacity: 0; }
    }
    .stButton > button {
        position: relative;
        overflow: hidden;
        -webkit-tap-highlight-color: transparent;
        touch-action: manipulation;       /* instant tap — no 300ms delay */
        user-select: none;
        min-height: 44px;                 /* big touch target for angry taps */
        cursor: pointer;
    }
    .stButton > button::before {
        content: '';
        position: absolute;
        top: var(--ripple-y, 50%); left: var(--ripple-x, 50%);
        width: 80px; height: 80px;
        border-radius: 50%;
        background: rgba(192,132,252,0.35);
        transform: translate(-50%,-50%) scale(0);
        pointer-events: none;
    }
    .stButton > button:active::before {
        animation: ripple 0.6s ease-out forwards;
    }

    /* Satisfying button press — instant visual feedback */
    .stButton > button:active {
        transform: scale(0.96) !important;
        transition: transform 0.06s ease !important;
    }
    .stButton > button {
        transition: transform 0.18s cubic-bezier(0.22,1,0.36,1),
                    box-shadow 0.18s ease,
                    filter 0.18s ease,
                    background 0.25s ease !important;
    }

    /* Bounce-back on button release */
    @keyframes bounceBack {
        0%   { transform: scale(0.96); }
        50%  { transform: scale(1.02); }
        100% { transform: scale(1); }
    }

    /* ── Smooth page transitions ─────────────────────────────── */
    @keyframes pageEnter {
        from { opacity: 0; transform: translateY(20px) scale(0.99); filter: blur(4px); }
        to   { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
    }
    section[data-testid="stMain"] > div {
        animation: pageEnter 0.55s cubic-bezier(0.22,1,0.36,1) both !important;
    }

    /* Stagger child elements within the page */
    @keyframes staggerIn {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    section[data-testid="stMain"] .element-container {
        animation: staggerIn 0.4s cubic-bezier(0.22,1,0.36,1) both;
    }
    section[data-testid="stMain"] .element-container:nth-child(1) { animation-delay: 0.02s; }
    section[data-testid="stMain"] .element-container:nth-child(2) { animation-delay: 0.06s; }
    section[data-testid="stMain"] .element-container:nth-child(3) { animation-delay: 0.10s; }
    section[data-testid="stMain"] .element-container:nth-child(4) { animation-delay: 0.14s; }
    section[data-testid="stMain"] .element-container:nth-child(5) { animation-delay: 0.18s; }
    section[data-testid="stMain"] .element-container:nth-child(6) { animation-delay: 0.22s; }
    section[data-testid="stMain"] .element-container:nth-child(7) { animation-delay: 0.26s; }
    section[data-testid="stMain"] .element-container:nth-child(8) { animation-delay: 0.30s; }

    /* ── Card hover 3D tilt effect ───────────────────────────── */
    [data-testid="stExpander"]:hover,
    [data-testid="stForm"]:hover {
        transform: translateY(-2px) perspective(800px) rotateX(1deg) !important;
    }

    /* ── Input focus glow ring ───────────────────────────────── */
    @keyframes focusGlow {
        0%, 100% { box-shadow: 0 0 0 3px rgba(124,58,237,0.15); }
        50%       { box-shadow: 0 0 0 5px rgba(192,132,252,0.25); }
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        animation: focusGlow 2s ease-in-out infinite !important;
    }

    /* ── Sidebar nav items — ADHD: Glow bar + slide reveal ────── */
    [data-testid="stSidebar"] .stRadio > div > label {
        position: relative;
        transition: all 0.25s cubic-bezier(0.22,1,0.36,1);
        padding-left: 14px !important;
        border-left: 3px solid transparent;
        border-radius: 0 10px 10px 0;
    }
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        padding-left: 20px !important;
        border-left: 3px solid rgba(192,132,252,0.6);
        background: linear-gradient(90deg, rgba(124,58,237,0.1), transparent) !important;
        box-shadow: -4px 0 20px rgba(192,132,252,0.1);
    }
    /* Active nav item glow */
    [data-testid="stSidebar"] .stRadio [aria-checked="true"] + div {
        position: relative;
    }
    [data-testid="stSidebar"] .stRadio [aria-checked="true"] ~ div label {
        border-left: 3px solid #c084fc !important;
        background: linear-gradient(90deg, rgba(192,132,252,0.08), transparent) !important;
        padding-left: 20px !important;
        box-shadow: -4px 0 24px rgba(192,132,252,0.15);
    }

    /* ── Toast-style alerts ──────────────────────────────────── */
    @keyframes alertPop {
        0%   { opacity: 0; transform: translateY(-10px) scale(0.95); }
        60%  { transform: translateY(2px) scale(1.01); }
        100% { opacity: 1; transform: translateY(0) scale(1); }
    }
    [data-testid="stAlert"] {
        animation: alertPop 0.4s cubic-bezier(0.22,1,0.36,1) both !important;
    }

    /* ── Success alert celebration pulse ─────────────────────── */
    @keyframes successPulse {
        0%   { box-shadow: 0 0 0 0 rgba(16,185,129,0.4); }
        70%  { box-shadow: 0 0 0 12px rgba(16,185,129,0); }
        100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
    }
    div[data-testid="stAlert"][data-baseweb*="positive"],
    div.stSuccess {
        animation: alertPop 0.4s cubic-bezier(0.22,1,0.36,1) both, successPulse 1s ease 0.3s !important;
    }

    /* ── Spinner upgrade: orbital dots ────────────────────────── */
    @keyframes spinOrbit {
        0%   { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    [data-testid="stSpinner"] > div {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Audio player glow on hover ───────────────────────────── */
    @keyframes audioGlow {
        0%, 100% { box-shadow: 0 0 0 rgba(192,132,252,0); }
        50%       { box-shadow: 0 0 28px rgba(192,132,252,0.3); }
    }
    audio:hover {
        animation: audioGlow 2s ease-in-out infinite;
    }

    /* ── Smooth scrollbar ─────────────────────────────────────── */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(10,10,14,0.4);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(124,58,237,0.35);
        border-radius: 3px;
        transition: background 0.2s;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(192,132,252,0.55);
    }
    * {
        scroll-behavior: smooth;
        scrollbar-width: thin;
        scrollbar-color: rgba(124,58,237,0.35) rgba(10,10,14,0.4);
    }

    /* ── Download button — ADHD: Green glow + pulse ─────────── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        transition: all 0.22s cubic-bezier(0.22,1,0.36,1) !important;
        min-height: 44px;
        box-shadow: 0 0 20px rgba(16,185,129,0.2);
        animation: glowPulse 3s ease-in-out infinite;
        --glow-color: rgba(16,185,129,0.35);
    }
    @keyframes greenGlow {
        0%, 100% { box-shadow: 0 0 18px rgba(16,185,129,0.25); }
        50%       { box-shadow: 0 0 36px rgba(16,185,129,0.5), 0 0 60px rgba(52,211,153,0.15); }
    }
    .stDownloadButton > button { animation: greenGlow 3s ease-in-out infinite !important; }
    .stDownloadButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 32px rgba(16,185,129,0.4) !important;
        filter: brightness(1.1);
    }
    .stDownloadButton > button:active {
        transform: scale(0.96) !important;
    }

    /* ── Tabs — ADHD: Glow underline + hover lift ────────────── */
    [data-testid="stTabs"] button {
        transition: all 0.25s cubic-bezier(0.22,1,0.36,1) !important;
        border-bottom: 2px solid transparent !important;
        color: #5858a0 !important;
        font-weight: 600 !important;
        padding: 10px 16px !important;
    }
    [data-testid="stTabs"] button:hover {
        color: #c084fc !important;
        background: rgba(124,58,237,0.08) !important;
        transform: translateY(-1px);
        border-radius: 8px 8px 0 0;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #c084fc !important;
        border-bottom: 2px solid #c084fc !important;
        text-shadow: 0 0 12px rgba(192,132,252,0.4);
        box-shadow: 0 2px 12px rgba(192,132,252,0.15);
    }

    /* ── Expander smooth open ─────────────────────────────────── */
    [data-testid="stExpander"] summary {
        transition: color 0.2s, background 0.2s;
        border-radius: 14px;
    }
    [data-testid="stExpander"] summary:hover {
        background: rgba(124,58,237,0.06);
    }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        animation: staggerIn 0.3s ease both;
    }

    /* ── Table / dataframe styling ────────────────────────────── */
    [data-testid="stDataFrame"] {
        border-radius: 12px !important;
        overflow: hidden;
        animation: cardIn 0.4s ease both;
    }

    /* ── Column layout gap fix ────────────────────────────────── */
    [data-testid="stHorizontalBlock"] {
        gap: 12px;
    }

    /* ── Divider — animated gradient sweep ──────────────────── */
    hr {
        background: linear-gradient(90deg, transparent 0%, rgba(124,58,237,0.3) 30%, rgba(192,132,252,0.4) 50%, rgba(124,58,237,0.3) 70%, transparent 100%) !important;
        background-size: 200% 100% !important;
        border: none !important;
        height: 1px !important;
        animation: shimmer 4s linear infinite;
    }

    /* ── Image / chart entrance ────────────────────────────────── */
    [data-testid="stImage"],
    .stPlotlyChart {
        animation: cardIn 0.5s cubic-bezier(0.22,1,0.36,1) both;
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Toast notification helper class ──────────────────────── */
    @keyframes toastSlideIn {
        from { transform: translateX(100%); opacity: 0; }
        to   { transform: translateX(0); opacity: 1; }
    }
    @keyframes toastSlideOut {
        from { transform: translateX(0); opacity: 1; }
        to   { transform: translateX(100%); opacity: 0; }
    }

    /* ── Skeleton loading shimmer (for custom loading states) ── */
    @keyframes skeletonShimmer {
        0%   { background-position: -400px 0; }
        100% { background-position: 400px 0; }
    }
    .skeleton {
        background: linear-gradient(90deg, rgba(18,18,28,0.8) 25%, rgba(42,42,72,0.6) 50%, rgba(18,18,28,0.8) 75%);
        background-size: 800px 100%;
        animation: skeletonShimmer 1.5s ease infinite;
        border-radius: 8px;
    }

    /* ── Frustration-proof: prevent double-click text selection ─ */
    .stButton > button,
    [data-testid="stSidebar"] .stRadio label {
        -webkit-user-select: none;
        user-select: none;
    }

    /* ── Frustration-proof: larger click targets everywhere ───── */
    [data-testid="stSidebar"] .stRadio > div > label {
        min-height: 38px;
        display: flex;
        align-items: center;
    }

    /* ── All interactive elements: no outline jank ─────────────── */
    button:focus-visible,
    input:focus-visible,
    select:focus-visible,
    textarea:focus-visible {
        outline: 2px solid rgba(192,132,252,0.5) !important;
        outline-offset: 2px;
    }
    button:focus:not(:focus-visible),
    input:focus:not(:focus-visible) {
        outline: none !important;
    }

    /* ── Tooltip-style help text ──────────────────────────────── */
    [data-testid="stTooltipIcon"] {
        transition: color 0.15s;
    }
    [data-testid="stTooltipIcon"]:hover {
        color: #c084fc !important;
        filter: drop-shadow(0 0 6px rgba(192,132,252,0.4));
    }

    /* ── Number input spin buttons ─────────────────────────────── */
    [data-testid="stNumberInput"] button {
        transition: background 0.15s, color 0.15s !important;
        border-radius: 6px !important;
    }
    [data-testid="stNumberInput"] button:hover {
        background: rgba(124,58,237,0.15) !important;
        color: #c084fc !important;
    }

    /* ── Quick-action card hover upgrade ──────────────────────── */
    div[style*="cursor: pointer"] {
        transition: all 0.22s cubic-bezier(0.22,1,0.36,1) !important;
    }
    div[style*="cursor: pointer"]:hover {
        transform: translateY(-3px) scale(1.01) !important;
        filter: brightness(1.08);
    }
    div[style*="cursor: pointer"]:active {
        transform: translateY(0) scale(0.98) !important;
    }

    /* ── Control Room theme override ─────────────────────────── */
    :root {
        --rm-bg-0: #061017;
        --rm-bg-1: #0b1820;
        --rm-bg-2: #10242b;
        --rm-panel: rgba(10, 24, 31, 0.86);
        --rm-panel-strong: rgba(12, 31, 39, 0.94);
        --rm-line: rgba(33, 201, 176, 0.16);
        --rm-line-strong: rgba(33, 201, 176, 0.34);
        --rm-text: #ebf7f4;
        --rm-muted: #8da6ab;
        --rm-accent: #1fc9b0;
        --rm-accent-2: #ffb24c;
        --rm-success: #9ef07a;
        --rm-danger: #ff7352;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 18% 12%, rgba(255,178,76,0.08), transparent 24%),
            radial-gradient(circle at 82% 16%, rgba(31,201,176,0.12), transparent 28%),
            radial-gradient(circle at 50% 120%, rgba(158,240,122,0.06), transparent 36%),
            linear-gradient(180deg, var(--rm-bg-0) 0%, var(--rm-bg-1) 55%, #08131a 100%) !important;
        color: var(--rm-text);
        font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    }

    [data-testid="stAppViewContainer"]::before,
    [data-testid="stAppViewContainer"]::after {
        background: none !important;
    }

    [data-testid="stMain"]::before {
        background-image:
            linear-gradient(rgba(31,201,176,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(31,201,176,0.03) 1px, transparent 1px) !important;
        background-size: 48px 48px !important;
    }

    section[data-testid="stMain"] > div {
        max-width: 1450px;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(6, 18, 24, 0.98) 0%, rgba(9, 23, 30, 0.98) 100%) !important;
        border-right: 1px solid rgba(31,201,176,0.16) !important;
    }

    [data-testid="stSidebar"]::before {
        background:
            radial-gradient(ellipse at 50% 0%, rgba(255,178,76,0.14) 0%, transparent 58%) !important;
        height: 220px !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #7f9aa0 !important;
        font-size: 0.86rem !important;
        font-weight: 500 !important;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        color: var(--rm-accent) !important;
        text-shadow: none !important;
        transform: translateX(3px) !important;
    }

    [data-testid="stSidebar"] [aria-checked="true"] ~ div label {
        color: #f4fbf8 !important;
        text-shadow: none !important;
        font-weight: 700 !important;
    }

    h1, h2, h3, h4, h5 {
        font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    }

    h1 {
        background: linear-gradient(110deg, #f7fffd, var(--rm-accent), var(--rm-accent-2)) !important;
        background-size: 240% 240% !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        filter: none !important;
    }

    h2, h3 {
        color: #ecfbf7 !important;
    }

    h2::after {
        background: linear-gradient(90deg, var(--rm-accent), transparent) !important;
    }

    p, label, .stCaption, [data-testid="stMarkdownContainer"] {
        color: var(--rm-text);
    }

    [data-testid="stExpander"],
    [data-testid="stForm"],
    [data-testid="stMetric"] {
        background: var(--rm-panel) !important;
        border: 1px solid rgba(31, 201, 176, 0.12) !important;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.18) !important;
    }

    [data-testid="stExpander"]:hover,
    [data-testid="stForm"]:hover,
    [data-testid="stMetric"]:hover {
        border-color: rgba(31, 201, 176, 0.28) !important;
        box-shadow: 0 22px 60px rgba(0, 0, 0, 0.22), 0 0 0 1px rgba(31, 201, 176, 0.08) !important;
    }

    [data-testid="stMetric"] label,
    [data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #8ea6ac !important;
        font-family: 'IBM Plex Mono', 'JetBrains Mono', monospace !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase;
    }

    [data-testid="stMetricValue"] {
        color: #f2fffb !important;
    }

    button[kind],
    [data-testid="stButton"] button,
    [data-testid="stDownloadButton"] button,
    [data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, rgba(31,201,176,0.95), rgba(18,142,129,0.95)) !important;
        color: #041015 !important;
        border: 1px solid rgba(31,201,176,0.22) !important;
        border-radius: 12px !important;
        font-family: 'IBM Plex Mono', 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
        box-shadow: none !important;
    }

    button[kind]:hover,
    [data-testid="stButton"] button:hover,
    [data-testid="stDownloadButton"] button:hover,
    [data-testid="baseButton-secondary"]:hover {
        background: linear-gradient(135deg, rgba(255,178,76,0.98), rgba(242,138,76,0.98)) !important;
        color: #0a1317 !important;
        transform: translateY(-1px) !important;
    }

    [data-testid="stRadio"] label p,
    [data-testid="stSelectbox"] label,
    [data-testid="stTextInput"] label,
    [data-testid="stSlider"] label,
    [data-testid="stFileUploader"] label,
    [data-testid="stNumberInput"] label {
        color: #d9ece8 !important;
        font-weight: 600 !important;
    }

    [data-baseweb="select"] > div,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input {
        background: rgba(8, 19, 25, 0.92) !important;
        border: 1px solid rgba(31,201,176,0.14) !important;
        color: #ecfbf7 !important;
        border-radius: 12px !important;
        font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    }

    [data-baseweb="select"] svg {
        color: var(--rm-accent) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(10, 24, 31, 0.88) !important;
        border: 1px solid rgba(31,201,176,0.1) !important;
        border-radius: 999px !important;
        color: #a7bec2 !important;
    }

    .stTabs [aria-selected="true"] {
        color: #071116 !important;
        background: linear-gradient(135deg, var(--rm-accent), var(--rm-accent-2)) !important;
        border-color: transparent !important;
    }

    </style>
    """, unsafe_allow_html=True)


_inject_css()


# ---------------------------------------------------------------------------
# ADHD JavaScript injection — cursor glow, particles, confetti, haptic feel
# ---------------------------------------------------------------------------

def _inject_adhd_js() -> None:
    """Inject dynamic JavaScript for ADHD-coded interactivity.

    Uses st_components.html() instead of st.markdown() because Streamlit
    strips <script> tags from markdown, causing raw JS text to leak onto
    the page.
    """
    st_components.html("""
    <div id="cursor-glow" style="
        position: fixed;
        width: 300px; height: 300px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(192,132,252,0.06) 0%, transparent 70%);
        pointer-events: none;
        z-index: 9998;
        transform: translate(-50%, -50%);
        transition: opacity 0.3s;
        mix-blend-mode: screen;
    "></div>
    <canvas id="particle-canvas" style="
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        pointer-events: none;
        z-index: 9997;
        opacity: 0.6;
    "></canvas>
    <script>
    (function() {
        // Target the parent Streamlit document, not the iframe
        const parentDoc = window.parent.document;

        // ── Cursor Glow Follow ──
        // Create glow element in parent so it follows cursor across the whole page
        let glow = parentDoc.getElementById('rmx-cursor-glow');
        if (!glow) {
            glow = parentDoc.createElement('div');
            glow.id = 'rmx-cursor-glow';
            glow.style.cssText = `
                position: fixed;
                width: 300px; height: 300px;
                border-radius: 50%;
                background: radial-gradient(circle, rgba(192,132,252,0.06) 0%, transparent 70%);
                pointer-events: none;
                z-index: 9998;
                transform: translate(-50%, -50%);
                transition: opacity 0.3s;
                mix-blend-mode: screen;
            `;
            parentDoc.body.appendChild(glow);
            let mx = 0, my = 0, gx = 0, gy = 0;
            parentDoc.addEventListener('mousemove', e => { mx = e.clientX; my = e.clientY; });
            function animGlow() {
                gx += (mx - gx) * 0.08;
                gy += (my - gy) * 0.08;
                glow.style.left = gx + 'px';
                glow.style.top = gy + 'px';
                requestAnimationFrame(animGlow);
            }
            animGlow();
        }

        // ── Floating Particles ──
        let canvas = parentDoc.getElementById('rmx-particle-canvas');
        if (!canvas) {
            canvas = parentDoc.createElement('canvas');
            canvas.id = 'rmx-particle-canvas';
            canvas.style.cssText = `
                position: fixed;
                top: 0; left: 0;
                width: 100vw; height: 100vh;
                pointer-events: none;
                z-index: 9997;
                opacity: 0.6;
            `;
            parentDoc.body.appendChild(canvas);

            const ctx = canvas.getContext('2d');
            let W, H;
            const particles = [];
            const PARTICLE_COUNT = 35;

            function resize() {
                W = canvas.width = window.parent.innerWidth;
                H = canvas.height = window.parent.innerHeight;
            }
            resize();
            window.parent.addEventListener('resize', resize);

            for (let i = 0; i < PARTICLE_COUNT; i++) {
                particles.push({
                    x: Math.random() * W,
                    y: Math.random() * H,
                    r: Math.random() * 2 + 0.5,
                    vx: (Math.random() - 0.5) * 0.3,
                    vy: (Math.random() - 0.5) * 0.3,
                    alpha: Math.random() * 0.4 + 0.1,
                    pulse: Math.random() * Math.PI * 2,
                });
            }

            function drawParticles() {
                ctx.clearRect(0, 0, W, H);
                for (const p of particles) {
                    p.x += p.vx;
                    p.y += p.vy;
                    p.pulse += 0.02;
                    if (p.x < 0) p.x = W;
                    if (p.x > W) p.x = 0;
                    if (p.y < 0) p.y = H;
                    if (p.y > H) p.y = 0;
                    const a = p.alpha * (0.5 + 0.5 * Math.sin(p.pulse));
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(192,132,252,${a})`;
                    ctx.fill();
                }

                // ── Connect nearby particles with lines ──
                for (let i = 0; i < particles.length; i++) {
                    for (let j = i + 1; j < particles.length; j++) {
                        const dx = particles[i].x - particles[j].x;
                        const dy = particles[i].y - particles[j].y;
                        const dist = Math.sqrt(dx*dx + dy*dy);
                        if (dist < 120) {
                            ctx.beginPath();
                            ctx.moveTo(particles[i].x, particles[i].y);
                            ctx.lineTo(particles[j].x, particles[j].y);
                            ctx.strokeStyle = `rgba(124,58,237,${0.08 * (1 - dist/120)})`;
                            ctx.lineWidth = 0.5;
                            ctx.stroke();
                        }
                    }
                }
                requestAnimationFrame(drawParticles);
            }
            drawParticles();
        }

        // ── Button click ripple with position tracking ──
        if (!parentDoc._rmxClickBound) {
            parentDoc._rmxClickBound = true;
            parentDoc.addEventListener('click', function(e) {
                const btn = e.target.closest('button');
                if (btn) {
                    const rect = btn.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;
                    btn.style.setProperty('--ripple-x', x + 'px');
                    btn.style.setProperty('--ripple-y', y + 'px');

                    // Spawn micro-particles on click
                    for (let i = 0; i < 6; i++) {
                        const spark = parentDoc.createElement('div');
                        spark.style.cssText = `
                            position: fixed;
                            left: ${e.clientX}px;
                            top: ${e.clientY}px;
                            width: 4px; height: 4px;
                            background: #c084fc;
                            border-radius: 50%;
                            pointer-events: none;
                            z-index: 99999;
                            transition: all 0.6s cubic-bezier(0.22,1,0.36,1);
                        `;
                        parentDoc.body.appendChild(spark);
                        const angle = (Math.PI * 2 / 6) * i;
                        const dist = 20 + Math.random() * 30;
                        requestAnimationFrame(() => {
                            spark.style.transform = `translate(${Math.cos(angle)*dist}px, ${Math.sin(angle)*dist}px) scale(0)`;
                            spark.style.opacity = '0';
                        });
                        setTimeout(() => spark.remove(), 700);
                    }
                }
            });
        }

        // ── Magnetic hover effect on cards ──
        if (!parentDoc._rmxHoverBound) {
            parentDoc._rmxHoverBound = true;
            parentDoc.addEventListener('mousemove', function(e) {
                const cards = parentDoc.querySelectorAll('[data-testid="stMetric"]');
                cards.forEach(card => {
                    const rect = card.getBoundingClientRect();
                    const cx = rect.left + rect.width / 2;
                    const cy = rect.top + rect.height / 2;
                    const dx = e.clientX - cx;
                    const dy = e.clientY - cy;
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    if (dist < 200) {
                        const strength = (200 - dist) / 200;
                        card.style.transform = `translateY(-${strength * 4}px) scale(${1 + strength * 0.02})`;
                    }
                });
            });
        }
    })();
    </script>
    """, height=0)


_inject_adhd_js()


# ---------------------------------------------------------------------------
# Enhanced UI helpers — animated loading, feedback, toast patterns
# ---------------------------------------------------------------------------

def _loading_card(message: str = "Processing…", sub: str = "") -> None:
    """Display a beautiful animated loading card instead of a plain spinner."""
    st.markdown(f"""
    <div style="
        background: rgba(18,18,28,0.85);
        border: 1px solid rgba(124,58,237,0.3);
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        animation: cardIn 0.4s cubic-bezier(0.22,1,0.36,1) both;
    ">
        <div style="
            width: 48px; height: 48px; margin: 0 auto 16px;
            border: 3px solid rgba(42,42,72,0.6);
            border-top: 3px solid #c084fc;
            border-radius: 50%;
            animation: spinOrbit 0.8s linear infinite;
        "></div>
        <div style="color: #c4c4e0; font-weight: 600; font-size: 1rem;">{message}</div>
        {'<div style="color: #6060a0; font-size: 0.82rem; margin-top: 6px;">' + sub + '</div>' if sub else ''}
    </div>
    """, unsafe_allow_html=True)


def _success_banner(message: str, icon: str = "✅") -> None:
    """ADHD: Celebratory success banner with confetti burst + glow.

    Uses st_components.html() for the confetti script so Streamlit
    doesn't strip the <script> tag and leak raw JS text.
    """
    _confetti_id = f"confetti_{hash(message) % 99999}"
    # Banner markup (no script) — safe for st.markdown
    st.markdown(f"""
    <div id="{_confetti_id}" style="
        background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(52,211,153,0.06));
        border: 1px solid rgba(16,185,129,0.4);
        border-radius: 16px;
        padding: 18px 24px;
        display: flex; align-items: center; gap: 14px;
        animation: alertPop 0.4s cubic-bezier(0.22,1,0.36,1) both, successPulse 1s ease 0.3s;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 30px rgba(16,185,129,0.12);
    ">
        <div style="position:absolute;top:0;left:0;right:0;height:1px;
            background:linear-gradient(90deg, transparent, rgba(52,211,153,0.6), transparent);"></div>
        <span style="font-size: 1.6rem; animation: floatBob 2s ease-in-out infinite;
            filter: drop-shadow(0 0 6px rgba(52,211,153,0.5));">{icon}</span>
        <span style="color: #a7f3d0; font-weight: 700; font-size: 1rem; letter-spacing: 0.01em;">{message}</span>
    </div>
    """, unsafe_allow_html=True)
    # Confetti script — must use st_components.html() to avoid Streamlit stripping <script>
    st_components.html(f"""
    <script>
    (function() {{
        const parentDoc = window.parent.document;
        const el = parentDoc.getElementById('{_confetti_id}');
        if (!el) return;
        const colors = ['#c084fc', '#10b981', '#818cf8', '#f59e0b', '#ec4899', '#60a5fa'];
        for (let i = 0; i < 20; i++) {{
            const c = parentDoc.createElement('div');
            const color = colors[Math.floor(Math.random()*colors.length)];
            c.style.cssText = `
                position:absolute;
                width:${{4+Math.random()*4}}px;
                height:${{4+Math.random()*4}}px;
                background:${{color}};
                border-radius:${{Math.random()>0.5?'50%':'2px'}};
                left:${{Math.random()*100}}%;
                top:50%;
                pointer-events:none;
                z-index:10;
                opacity:1;
                transition: all ${{0.6+Math.random()*0.6}}s cubic-bezier(0.22,1,0.36,1);
            `;
            el.appendChild(c);
            requestAnimationFrame(() => {{
                c.style.transform = `translate(${{(Math.random()-0.5)*120}}px, ${{-40-Math.random()*80}}px) rotate(${{Math.random()*360}}deg)`;
                c.style.opacity = '0';
            }});
            setTimeout(() => c.remove(), 1500);
        }}
    }})();
    </script>
    """, height=0)


def _progress_card(progress: float, message: str = "", status: str = "running") -> None:
    """Animated progress card with percentage, bar, and status."""
    pct = max(0.0, min(1.0, progress))
    pct_display = int(pct * 100)
    bar_color = "#c084fc" if status == "running" else ("#10b981" if status == "done" else "#ef4444")
    st.markdown(f"""
    <div style="
        background: rgba(18,18,28,0.85);
        border: 1px solid rgba(124,58,237,0.25);
        border-radius: 14px;
        padding: 18px 22px;
        animation: cardIn 0.3s ease both;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="color: #c4c4e0; font-weight: 600; font-size: 0.9rem;">{message}</span>
            <span style="
                color: {bar_color}; font-weight: 700; font-size: 1.1rem;
                text-shadow: 0 0 12px {bar_color}44;
            ">{pct_display}%</span>
        </div>
        <div style="
            height: 6px;
            background: rgba(42,42,72,0.6);
            border-radius: 3px;
            overflow: hidden;
        ">
            <div style="
                width: {pct_display}%;
                height: 100%;
                background: linear-gradient(90deg, #7c3aed, {bar_color});
                border-radius: 3px;
                transition: width 0.4s cubic-bezier(0.22,1,0.36,1);
                box-shadow: 0 0 12px {bar_color}44;
            "></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _empty_state(icon: str, title: str, subtitle: str = "") -> None:
    """Beautiful empty state with animated icon."""
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 48px 24px;
        animation: fadeUp 0.5s ease both;
    ">
        <div style="
            font-size: 3.5rem;
            margin-bottom: 16px;
            animation: floatBob 3s ease-in-out infinite;
        ">{icon}</div>
        <div style="
            color: #9090c0;
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 6px;
        ">{title}</div>
        {'<div style="color: #505075; font-size: 0.85rem;">' + subtitle + '</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def _section_header(icon: str, title: str, subtitle: str = "") -> None:
    """ADHD-coded animated section header with gradient, glow, and pulse."""
    st.markdown(f"""
    <div style="
        display: flex; align-items: center; gap: 16px;
        margin: 12px 0 22px;
        padding: 14px 20px;
        background: linear-gradient(135deg, rgba(124,58,237,0.06), rgba(79,70,229,0.03));
        border: 1px solid rgba(124,58,237,0.12);
        border-radius: 16px;
        animation: fadeUp 0.4s cubic-bezier(0.22,1,0.36,1) both;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute; top: 0; left: 0; right: 0; height: 1px;
            background: linear-gradient(90deg, transparent, rgba(192,132,252,0.4), transparent);
        "></div>
        <span style="
            font-size: 2rem;
            animation: floatBob 3s ease-in-out infinite;
            filter: drop-shadow(0 0 8px rgba(192,132,252,0.3));
        ">{icon}</span>
        <div>
            <span style="
                font-size: 1.4rem; font-weight: 800;
                background: linear-gradient(270deg, #c084fc, #818cf8, #60a5fa, #ec4899);
                background-size: 400% 400%;
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                background-clip: text;
                animation: holoShimmer 4s ease infinite;
            ">{title}</span>
            {'<div style="color: #505080; font-size: 0.8rem; margin-top: 4px; letter-spacing: 0.02em;">' + subtitle + '</div>' if subtitle else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Reusable smart song picker — case-insensitive search + dynamic loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def _fetch_all_song_names(with_wav: bool = False, with_stems: bool = False) -> list:
    """Fetch ALL song names from the lightweight /library/names endpoint."""
    params = {}
    if with_wav:
        params["with_wav"] = "true"
    if with_stems:
        params["with_stems"] = "true"
    try:
        resp = api_get("/library/names", params=params)
        return resp.get("names", []) if resp else []
    except Exception:
        # Fallback to paginated endpoint
        try:
            lib = api_get("/library?per_page=5000")
            if lib and lib.get("songs"):
                names = [s["name"] for s in lib["songs"]]
                if with_wav:
                    names = [s["name"] for s in lib["songs"] if s.get("has_full_wav")]
                return names
        except Exception:
            pass
        return []


def _song_picker(
    label: str,
    songs: list,
    key: str,
    default: str = None,
    help_text: str = None,
) -> Optional[str]:
    """
    Smart song picker with case-insensitive search filtering.

    Shows a text_input for search + a selectbox with filtered results.
    Works with any size library — no scroll issues since the list is pre-filtered.
    """
    # Search box
    search_key = f"_search_{key}"
    search_term = st.text_input(
        f"🔍 Search {label.lower()}",
        key=search_key,
        placeholder="Type to filter (case-insensitive)…",
        help=help_text,
    )

    # Case-insensitive filter
    if search_term:
        pat = search_term.lower()
        filtered = [s for s in songs if pat in s.lower()]
    else:
        filtered = songs

    if not filtered:
        st.caption(f"No songs match '{search_term}'")
        return None

    # Determine default index
    idx = 0
    if default and default in filtered:
        idx = filtered.index(default)

    selected = st.selectbox(label, filtered, index=idx, key=key)
    return selected


# ---------------------------------------------------------------------------
# Session-state bootstrap
# ---------------------------------------------------------------------------

if "active_jobs" not in st.session_state:
    st.session_state.active_jobs: Dict[str, Dict] = {}
if "job_results" not in st.session_state:
    st.session_state.job_results: Dict[str, Dict] = {}
if "nav_idx" not in st.session_state:
    st.session_state.nav_idx = 0
# Pre-filled song selectors (set by Library quick-actions)
if "pending_analyze_song" not in st.session_state:
    st.session_state.pending_analyze_song = None
if "pending_remix_a" not in st.session_state:
    st.session_state.pending_remix_a = None
if "pending_compat_a" not in st.session_state:
    st.session_state.pending_compat_a = None
if "pending_similar" not in st.session_state:
    st.session_state.pending_similar = None


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

# SSL verification: enabled by default. Only disable via explicit env var
# (SKIP_SSL_VERIFY=1) for local dev with self-signed certs.
# Never derive verify=False from the protocol flag — CodeQL CWE-297.
import os as _os
_VERIFY_SSL = not (_os.getenv("SKIP_SSL_VERIFY", "").lower() in ("1", "true", "yes"))

def api_get(path: str, params: dict = None) -> Optional[Dict]:
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=10, verify=_VERIFY_SSL)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach API at localhost:8000. Is the server running?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def _api_get_silent(path: str) -> Optional[Dict]:
    """Like api_get but never shows error widgets — used by background polling."""
    try:
        r = requests.get(f"{API}{path}", timeout=5, verify=_VERIFY_SSL)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(path: str, body: dict) -> Optional[Dict]:
    try:
        r = requests.post(f"{API}{path}", json=body, timeout=15, verify=_VERIFY_SSL)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach API at localhost:8000. Is the server running?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def poll_job(job_id: str, placeholder, timeout: int = 120) -> Optional[Dict]:
    """Poll a job until done or failed, updating a progress bar in-place."""
    bar = placeholder.progress(0.0, text="Starting…")
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = _api_get_silent(f"/jobs/{job_id}")
        if not job:
            return None
        prog = job.get("progress", 0.0)
        msg  = job.get("message", "")
        bar.progress(min(prog, 1.0), text=msg)
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(1.5)
    st.error("Job timed out.")
    return None


def submit_job(
    name: str,
    endpoint: str,
    body: dict,
    icon: str = "⚙️",
    page_key: Optional[str] = None,
) -> Optional[str]:
    """
    Non-blocking job submission.

    Posts to the API endpoint, registers the job in session_state.active_jobs,
    and returns the job_id immediately.  The sidebar Jobs Tray fragment will
    poll it in the background every 2 s while the user does other things.

    Parameters
    ----------
    name      : Human-readable label shown in the Jobs Tray.
    endpoint  : API path, e.g. '/dj-remix'.
    body      : JSON body for the POST.
    icon      : Emoji shown in the tray next to the job name.
    page_key  : Optional string key so the page can retrieve the result from
                st.session_state.job_results[page_key] after completion.
    """
    resp = api_post(endpoint, body)
    if not resp:
        return None
    job_id = resp["job_id"]
    st.session_state.active_jobs[job_id] = {
        "id":           job_id,
        "name":         name,
        "icon":         icon,
        "status":       "running",
        "progress":     0.0,
        "message":      "Queued…",
        "result":       None,
        "submitted_at": time.time(),
        "page_key":     page_key,
    }
    return job_id


def _inline_job_progress(page_key: str) -> Optional[Dict]:
    """
    Render a LIVE inline progress card that auto-refreshes every 2 seconds.

    Uses a @st.fragment so the progress card polls independently of the rest
    of the page — same pattern as the sidebar Jobs Tray.

    Returns the job info dict when the job is done (so the calling page can
    render results below the fragment), or None if still running / no job.
    """
    # First: check if there's a running job for this page_key
    active = st.session_state.get("active_jobs", {})
    has_running = any(
        info.get("page_key") == page_key and info["status"] == "running"
        for info in active.values()
    )

    if has_running:
        # Render the auto-polling fragment
        _inline_progress_fragment(page_key)
        return None  # Still running — caller should not render results yet

    # No running job — check for completed results
    result_info = st.session_state.get("job_results", {}).get(page_key)
    return result_info


@st.fragment(run_every=2)
def _inline_progress_fragment(page_key: str) -> None:
    """
    Auto-refreshing fragment that polls the active job for *page_key* every 2s.

    This fragment re-executes independently of the main page, so the progress
    bar updates in real time without requiring manual page refresh.
    """
    active = st.session_state.get("active_jobs", {})
    job_info = None
    for info in active.values():
        if info.get("page_key") == page_key:
            job_info = info
            break

    if not job_info:
        return

    # Poll the API for fresh status
    if job_info["status"] == "running":
        data = _api_get_silent(f"/jobs/{job_info['id']}")
        if data:
            job_info["status"]   = data["status"]
            job_info["progress"] = data.get("progress", 0.0)
            job_info["message"]  = data.get("message", "")
            if data["status"] in ("done", "failed"):
                job_info["result"] = data.get("result")
                pk = job_info.get("page_key")
                if pk:
                    st.session_state.job_results[pk] = job_info

    status  = job_info["status"]
    prog    = min(float(job_info.get("progress", 0.0)), 1.0)
    pct     = int(prog * 100)
    msg     = job_info.get("message", "Working…")
    icon    = job_info.get("icon", "⚙️")
    name    = job_info.get("name", "Job")
    elapsed = int(time.time() - job_info.get("submitted_at", time.time()))
    elapsed_str = f"{elapsed//60}m {elapsed%60}s" if elapsed >= 60 else f"{elapsed}s"

    # ETA display
    eta_str = ""
    if status == "running" and prog > 0.05 and elapsed > 3:
        rate = prog / (elapsed + 1e-8)
        remaining = (1.0 - prog) / (rate + 1e-8)
        if remaining < 60:
            eta_str = f" · ~{int(remaining)}s left"
        elif remaining < 3600:
            eta_str = f" · ~{int(remaining//60)}m left"

    if status == "done":
        color = "#22c55e"
        border_color = "rgba(34,197,94,0.4)"
        bg_grad = "rgba(34,197,94,0.12), rgba(34,197,94,0.04)"
        status_text = "✅ Complete!"
    elif status == "failed":
        color = "#ef4444"
        border_color = "rgba(239,68,68,0.4)"
        bg_grad = "rgba(239,68,68,0.12), rgba(239,68,68,0.04)"
        status_text = f"❌ {msg}"
    else:
        color = "#7c3aed"
        border_color = "rgba(124,58,237,0.3)"
        bg_grad = "rgba(124,58,237,0.12), rgba(34,197,94,0.08)"
        status_text = msg

    st.markdown(
        f"""<div style="
            background: linear-gradient(135deg, {bg_grad});
            border: 1px solid {border_color};
            border-radius: 14px; padding: 18px 22px; margin: 16px 0;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:1rem; font-weight:700; color:#e0e0ff;">
                    {icon} {name}
                </span>
                <span style="font-size:0.8rem; color:{color}; font-weight:600;">
                    {pct}%{eta_str} · {elapsed_str}
                </span>
            </div>
            <div style="font-size:0.85rem; color:#a0a0c0; margin-bottom:10px;">
                {status_text}
            </div>
            <div style="background:rgba(20,20,40,0.6); border-radius:8px; height:10px; overflow:hidden;">
                <div style="
                    background: linear-gradient(90deg, {color}, {color}aa, {color}66);
                    width:{pct}%; height:100%; border-radius:8px;
                    transition: width 1.5s ease;
                "></div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # If done/failed, trigger a full page rerun so the caller can render results
    if status in ("done", "failed"):
        st.rerun()


def _strudel_embed(code: str, height: int = 400) -> None:
    """
    Render a live Strudel REPL inside the page using st.components.
    Loads @strudel/web from CDN and mounts a <strudel-editor> web component.
    The user can edit, play (Ctrl+Enter), stop, and record directly here.
    """
    # Escape HTML special chars so the code is safe inside element content
    safe = (code or "")
    safe = safe.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <style>
    html, body {{
      margin: 0; padding: 0; height: 100%;
      background: #0e1117; overflow: hidden;
    }}
    strudel-editor {{
      display: block;
      width: 100%;
      height: {height - 4}px;
    }}
  </style>
</head>
<body>
  <script type="module" src="https://unpkg.com/@strudel/web@latest"></script>
  <strudel-editor>{safe}</strudel-editor>
</body>
</html>"""
    st_components.html(html, height=height, scrolling=False)


def _show_spectrogram(
    wav_path: str,
    title: str = "",
    duration: float = 60.0,
    colormap: str = "magma",
    show_chroma: bool = False,
) -> None:
    """
    Render a waveform, mel spectrogram, and optional chromagram for any WAV file.
    Uses librosa + matplotlib (lazy-imported so they don't slow Streamlit startup).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import librosa
        import librosa.display
    except ImportError as e:
        st.error(f"Missing library for spectrogram: {e}  —  run `pip install librosa matplotlib`")
        return

    try:
        y, sr = librosa.load(wav_path, sr=None, mono=True, duration=duration)
    except Exception as e:
        st.error(f"Could not load audio for spectrogram: {e}")
        return

    dark_bg = "#0e1117"
    tick_col = "#888888"

    def _style(fig, ax):
        fig.patch.set_facecolor(dark_bg)
        ax.set_facecolor(dark_bg)
        ax.tick_params(colors=tick_col, labelsize=7)
        ax.xaxis.label.set_color(tick_col)
        ax.yaxis.label.set_color(tick_col)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a2a")

    # ── Waveform ─────────────────────────────────────────────────────────────
    times = np.linspace(0.0, len(y) / sr, len(y))
    fig_w, ax_w = plt.subplots(figsize=(12, 1.6))
    _style(fig_w, ax_w)
    ax_w.plot(times, y, color="#ff4b4b", linewidth=0.35, alpha=0.9)
    ax_w.set_xlabel("Time (s)")
    ax_w.set_ylabel("Amplitude")
    if title:
        ax_w.set_title(f"Waveform — {title}", color="#dddddd", fontsize=9, pad=4)
    fig_w.tight_layout(pad=0.6)
    st.pyplot(fig_w, use_container_width=True)
    plt.close(fig_w)

    # ── Mel Spectrogram ───────────────────────────────────────────────────────
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr // 2)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig_s, ax_s = plt.subplots(figsize=(12, 3.5))
    _style(fig_s, ax_s)
    img = librosa.display.specshow(
        S_dB, x_axis="time", y_axis="mel",
        sr=sr, fmax=sr // 2, ax=ax_s, cmap=colormap,
    )
    cbar = fig_s.colorbar(img, ax=ax_s, format="%+2.0f dB", pad=0.01)
    cbar.ax.yaxis.set_tick_params(color=tick_col, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=tick_col)
    cbar.outline.set_edgecolor("#2a2a2a")
    ax_s.set_xlabel("Time (s)")
    ax_s.set_ylabel("Frequency (Hz)")
    if title:
        ax_s.set_title(f"Mel Spectrogram — {title}", color="#dddddd", fontsize=9, pad=4)
    fig_s.tight_layout(pad=0.6)
    st.pyplot(fig_s, use_container_width=True)
    plt.close(fig_s)

    # ── Chromagram (optional) ─────────────────────────────────────────────────
    if show_chroma:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        fig_c, ax_c = plt.subplots(figsize=(12, 2.5))
        _style(fig_c, ax_c)
        img_c = librosa.display.specshow(
            chroma, x_axis="time", y_axis="chroma",
            sr=sr, ax=ax_c, cmap="viridis",
        )
        cbar_c = fig_c.colorbar(img_c, ax=ax_c, pad=0.01)
        cbar_c.ax.yaxis.set_tick_params(color=tick_col, labelsize=7)
        plt.setp(cbar_c.ax.yaxis.get_ticklabels(), color=tick_col)
        cbar_c.outline.set_edgecolor("#2a2a2a")
        ax_c.set_xlabel("Time (s)")
        ax_c.set_ylabel("Pitch class")
        if title:
            ax_c.set_title(f"Chromagram — {title}", color="#dddddd", fontsize=9, pad=4)
        fig_c.tight_layout(pad=0.6)
        st.pyplot(fig_c, use_container_width=True)
        plt.close(fig_c)


def score_color(score: float) -> str:
    if score >= 0.75: return "🟢"
    if score >= 0.5:  return "🟡"
    return "🔴"


# ---------------------------------------------------------------------------
# Jobs Tray — persistent background-job monitor in the sidebar
# ---------------------------------------------------------------------------

_JOB_ICON = {"running": "🔄", "done": "✅", "failed": "❌"}
_PHASE_COLORS = {
    "running": "#7c3aed",
    "done":    "#22c55e",
    "failed":  "#ef4444",
}

@st.fragment(run_every=2)
def _jobs_tray() -> None:
    """
    Runs every 2 s inside the sidebar regardless of which page is active.
    Polls all active jobs, updates session state, and renders a compact tray.
    """
    active: Dict[str, Dict] = st.session_state.get("active_jobs", {})
    if not active:
        return

    now = time.time()

    # ── Poll all running jobs ─────────────────────────────────────────────────
    for job_id, info in list(active.items()):
        if info["status"] == "running":
            data = _api_get_silent(f"/jobs/{job_id}")
            if data:
                info["status"]   = data["status"]
                info["progress"] = data.get("progress", 0.0)
                info["message"]  = data.get("message", "")
                info["eta_sec"]  = data.get("eta_sec")
                if data["status"] in ("done", "failed"):
                    info["result"] = data.get("result")
                    # Store result keyed by page_key so pages can retrieve it
                    pk = info.get("page_key")
                    if pk:
                        st.session_state.job_results[pk] = info

    # ── Expire completed jobs older than 90 s ─────────────────────────────────
    to_remove = [
        jid for jid, info in active.items()
        if info["status"] != "running" and (now - info.get("submitted_at", now)) > 90
    ]
    for jid in to_remove:
        active.pop(jid, None)

    if not active:
        return

    # ── Render tray ───────────────────────────────────────────────────────────
    running_count = sum(1 for v in active.values() if v["status"] == "running")
    label = f"⚡ {running_count} job{'s' if running_count != 1 else ''} running" \
            if running_count else "📋 Recent jobs"

    st.divider()
    st.markdown(f"**{label}**")

    for info in sorted(active.values(), key=lambda x: -x.get("submitted_at", 0)):
        status  = info["status"]
        icon    = _JOB_ICON.get(status, "⚙️")
        prog    = min(float(info.get("progress", 0.0)), 1.0)
        pct     = int(prog * 100)
        msg     = info.get("message", "")
        elapsed = int(now - info.get("submitted_at", now))

        # Compact card
        color = _PHASE_COLORS.get(status, "#7c3aed")
        elapsed_str = f"{elapsed//60}m {elapsed%60}s" if elapsed >= 60 else f"{elapsed}s"

        # ETA from backend
        eta_sec = info.get("eta_sec")
        eta_str = ""
        if status == "running" and eta_sec is not None and eta_sec > 0:
            if eta_sec < 60:
                eta_str = f" · ~{eta_sec}s left"
            else:
                eta_str = f" · ~{eta_sec // 60}m left"

        st.markdown(
            f"<div style='"
            f"background:rgba(18,18,30,0.85);border:1px solid {color}44;"
            f"border-radius:10px;padding:10px 12px;margin:4px 0;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<span style='font-size:0.8rem;font-weight:600;color:#d0d0f0;'>"
            f"{icon} {info['icon']} {info['name'][:34]}</span>"
            f"<span style='font-size:0.68rem;color:{color};font-weight:600;'>"
            f"{pct}%{eta_str} · {elapsed_str}</span>"
            f"</div>"
            f"<div style='font-size:0.7rem;color:#505075;margin:4px 0 6px;"
            f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{msg[:60]}</div>"
            f"<div style='background:#1a1a2e;border-radius:4px;height:6px;overflow:hidden;'>"
            f"<div style='background:{color};width:{pct}%;height:100%;"
            f"border-radius:4px;transition:width 1.8s ease;'></div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

PAGES = [
    "🏠 Home",
    "📚 Library",
    "⬇️ Download",
    "🟢 Spotify",
    "🔮 Smart Search",
    "🔗 Compatibility",
    "🎛️ DJ Remix",
    "🎚️ DJ Chain",
    "🧪 Instrument Lab",
    "🤖 AI Studio",
    "🎵 My Mixes",
    "🔍 Analyze",
    "🎨 Visualizer",
    "🗜️ Compress Library",
    "🚀 Initialize Library",
]

# ── Section boundaries for sidebar labels ──────────────────────────────────
_PAGE_SECTIONS = {
    "🏠 Home":             None,           # no label before first item
    "📚 Library":          "🎵  MUSIC",
    "⬇️ Download":         None,
    "🟢 Spotify":          None,
    "🔮 Smart Search":     None,
    "🔗 Compatibility":    "🎛️  MIX",
    "🎛️ DJ Remix":         None,
    "🎚️ DJ Chain":         None,
    "🧪 Instrument Lab":   None,
    "🤖 AI Studio":        "🤖  GENERATIVE AI",
    "🎵 My Mixes":         None,
    "🔍 Analyze":          "🔧  TOOLS",
    "🎨 Visualizer":       None,
    "🗜️ Compress Library": None,
    "🚀 Initialize Library": None,
}


def _go_to_page(page_name: str) -> None:
    """Navigate to a named page and trigger a rerun."""
    if page_name in PAGES:
        st.session_state.nav_idx = PAGES.index(page_name)
        st.rerun()


with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 8px; animation: fadeUp 0.5s ease both;">
        <div style="font-size: 2.8rem; margin-bottom: 4px; filter: drop-shadow(0 0 12px rgba(192,132,252,0.4));
            animation: floatBob 4s ease-in-out infinite;">🎛️</div>
        <div style="
            font-size: 1.3rem; font-weight: 900; letter-spacing: -0.5px;
            background: linear-gradient(270deg, #c084fc, #818cf8, #60a5fa, #ec4899, #c084fc);
            background-size: 600% 600%;
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: holoShimmer 4s ease infinite;
        ">AI RemixMate</div>
        <div style="
            font-size: 0.65rem; color: #4848a0; text-transform: uppercase;
            letter-spacing: 0.12em; font-weight: 600; margin-top: 4px;
            font-family: 'JetBrains Mono', monospace;
        ">GPU-Accelerated DJ Engine</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Background Jobs Tray — TOP of sidebar so it's always visible ──────────
    _jobs_tray()

    st.divider()

    # ── Section labels injected above the radio ────────────────────────────────
    _section_css = """
    <style>
    .nav-section {
        font-size: 0.62rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #40406a;
        margin: 10px 0 2px 4px;
    }
    </style>
    """
    st.markdown(_section_css, unsafe_allow_html=True)

    # Render section headers inline with the radio by pre-computing HTML
    # (Streamlit doesn't natively support grouped radios, so we overlay labels)
    _nav_labels_html = '<div style="margin-bottom:-8px;">'
    for _p in PAGES:
        _sec = _PAGE_SECTIONS.get(_p)
        if _sec:
            _nav_labels_html += f'<div class="nav-section">{_sec}</div>'
    _nav_labels_html += '</div>'
    # We just show the section labels once above the radio as a guide
    st.markdown(
        '<div class="nav-section" style="margin-top:4px;">OVERVIEW</div>'
        '<div class="nav-section" style="margin-top:10px;">🎵 MUSIC  ·  🎛️ MIX  ·  🔧 TOOLS</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        PAGES,
        label_visibility="collapsed",
        index=min(st.session_state.nav_idx, len(PAGES) - 1),
        key="nav_radio",
    )
    # Keep nav_idx in sync so _go_to_page works
    st.session_state.nav_idx = PAGES.index(page)

    # Live API status — ADHD: pulsing indicator
    st.divider()
    _health_sidebar = _api_get_silent("/health")
    if _health_sidebar:
        _lib_count = _health_sidebar.get("library_songs", "?")
        st.markdown(f"""
        <div style="
            display: flex; align-items: center; gap: 10px;
            padding: 10px 14px;
            background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(16,185,129,0.03));
            border: 1px solid rgba(16,185,129,0.25);
            border-radius: 12px;
            animation: fadeUp 0.3s ease both;
        ">
            <div style="
                width: 8px; height: 8px; border-radius: 50%;
                background: #22c55e;
                box-shadow: 0 0 8px #22c55e, 0 0 16px rgba(34,197,94,0.3);
                animation: breathe 2s ease-in-out infinite;
            "></div>
            <div>
                <div style="font-size: 0.78rem; font-weight: 700; color: #a7f3d0;">API Online</div>
                <div style="font-size: 0.65rem; color: #059669; font-family: 'JetBrains Mono', monospace;">{_lib_count} songs · GPU ready</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            display: flex; align-items: center; gap: 10px;
            padding: 10px 14px;
            background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.03));
            border: 1px solid rgba(239,68,68,0.25);
            border-radius: 12px;
        ">
            <div style="width: 8px; height: 8px; border-radius: 50%; background: #ef4444;
                box-shadow: 0 0 8px #ef4444;"></div>
            <div style="font-size: 0.78rem; font-weight: 700; color: #fca5a5;">API Offline</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Initialize Library — live stats fragment (defined here, called inside elif)
# ---------------------------------------------------------------------------

@st.fragment(run_every=3)
def _init_library_live_stats() -> None:
    """
    Auto-refreshing fragment — polls library + index stats every 3 s and renders
    animated stat cards + pipeline stage stepper. Completely independent of the
    main page rerun, so numbers update live while a job is running.
    """
    lib        = api_get("/library?per_page=5000") or {}
    songs      = lib.get("songs", [])
    n_total    = len(songs)
    n_stems    = sum(1 for s in songs if s.get("stems"))
    n_no_stems = n_total - n_stems
    stems_pct  = int((n_stems / n_total * 100)) if n_total else 0

    idx_stats   = api_get("/index/stats") or {}
    n_indexed   = idx_stats.get("total_indexed", 0)
    n_unindexed = n_total - n_indexed
    index_pct   = int((n_indexed / n_total * 100)) if n_total else 0

    active_jobs  = st.session_state.get("active_jobs", {})
    running_init = any(
        info.get("page_key") == "init_library" and info["status"] == "running"
        for info in active_jobs.values()
    )
    job_progress = 0.0
    job_message  = ""
    if running_init:
        for info in active_jobs.values():
            if info.get("page_key") == "init_library":
                job_progress = float(info.get("progress", 0.0))
                job_message  = info.get("message", "")
                break

    stage_idx = 0 if job_progress < 0.33 else (1 if job_progress < 0.66 else 2)

    stems_color = "#22c55e" if n_no_stems == 0 else ("#f59e0b" if n_no_stems < n_total * 0.3 else "#a855f7")
    index_color = "#22c55e" if n_unindexed == 0 else ("#f59e0b" if n_unindexed < n_total * 0.3 else "#3b82f6")
    stems_glow  = "34,197,94" if n_no_stems == 0 else ("245,158,11" if n_no_stems < n_total * 0.3 else "168,85,247")
    index_glow  = "34,197,94" if n_unindexed == 0 else "59,130,246"
    pulse_css   = "animation: ringPulse 1.2s ease-in-out infinite;" if running_init else ""
    stems_sub   = "all done \u2713" if n_no_stems == 0 else f"{n_stems} / {n_total} split"
    index_sub   = "all indexed \u2713" if n_unindexed == 0 else f"{n_indexed} / {n_total} indexed"
    stems_dash  = int(stems_pct * 1.068)
    index_dash  = int(index_pct * 1.068)

    s1_class = "active" if (running_init and stage_idx == 0) else ("done" if stems_pct == 100 else "")
    s1_num   = "\u2713" if stems_pct == 100 else "1"
    s1_badge = "active" if (running_init and stage_idx == 0) else ("done" if stems_pct == 100 else "idle")
    s1_dot   = "blinking" if (running_init and stage_idx == 0) else ""
    s1_label = "Running" if (running_init and stage_idx == 0) else ("Complete" if stems_pct == 100 else "Pending")

    s2_done  = running_init and stage_idx > 1
    s2_class = "active" if (running_init and stage_idx == 1) else ("done" if s2_done else "")
    s2_num   = "\u2713" if s2_done else "2"
    s2_badge = "active" if (running_init and stage_idx == 1) else ("done" if s2_done else "idle")
    s2_dot   = "blinking" if (running_init and stage_idx == 1) else ""
    s2_label = "Running" if (running_init and stage_idx == 1) else ("Complete" if s2_done else "Pending")

    s3_done  = index_pct == 100 and n_total > 0
    s3_class = "active" if (running_init and stage_idx == 2) else ("done" if s3_done else "")
    s3_num   = "\u2713" if s3_done else "3"
    s3_badge = "active" if (running_init and stage_idx == 2) else ("done" if s3_done else "idle")
    s3_dot   = "blinking" if (running_init and stage_idx == 2) else ""
    s3_label = "Running" if (running_init and stage_idx == 2) else ("Complete" if s3_done else "Pending")

    msg_html = (
        f"<div style='font-family:Inter,sans-serif;font-size:12px;color:rgba(168,85,247,0.85);"
        f"margin-top:10px;padding:10px 14px;background:rgba(168,85,247,0.08);"
        f"border:1px solid rgba(168,85,247,0.2);border-radius:8px;'>&#9889; {job_message}</div>"
        if running_init and job_message else ""
    )

    st_components.html(f"""
    <style>
      @keyframes countUp {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
      }}
      @keyframes ringPulse {{
        0%, 100% {{ filter: drop-shadow(0 0 4px currentColor); }}
        50%       {{ filter: drop-shadow(0 0 14px currentColor); }}
      }}
      @keyframes stagePulse {{
        0%, 100% {{ box-shadow: 0 0 0 0 rgba(168,85,247,0.5); }}
        50%       {{ box-shadow: 0 0 0 8px rgba(168,85,247,0); }}
      }}
      @keyframes dotBlink {{
        0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.2; }}
      }}
      .rm-stats-grid {{
        display: grid; grid-template-columns: repeat(4, 1fr);
        gap: 14px; margin-bottom: 20px; font-family: 'Inter', sans-serif;
      }}
      .rm-stat-card {{
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09);
        border-radius: 14px; padding: 18px 20px 16px;
        position: relative; overflow: hidden; transition: border-color 0.3s;
      }}
      .rm-stat-card::before {{
        content: ''; position: absolute; inset: 0;
        background: radial-gradient(ellipse at top left, var(--card-glow, transparent) 0%, transparent 70%);
        pointer-events: none;
      }}
      .rm-stat-label {{
        font-size: 10px; font-weight: 600; letter-spacing: 0.08em;
        text-transform: uppercase; color: rgba(255,255,255,0.45); margin-bottom: 10px;
      }}
      .rm-stat-number {{
        font-size: 36px; font-weight: 800; color: #fff; line-height: 1;
        animation: countUp 0.4s ease both;
      }}
      .rm-stat-sub {{ font-size: 12px; margin-top: 8px; color: rgba(255,255,255,0.45); font-weight: 500; }}
      .rm-ring-wrap {{ position: absolute; top: 14px; right: 16px; width: 42px; height: 42px; }}
      .rm-ring-wrap svg {{ {pulse_css} }}
      .rm-pipeline {{
        display: flex; align-items: stretch; gap: 0;
        margin-top: 4px; margin-bottom: 4px; font-family: 'Inter', sans-serif;
      }}
      .rm-stage {{
        flex: 1; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09);
        border-radius: 12px; padding: 14px 16px; transition: all 0.3s;
      }}
      .rm-stage.active {{
        border-color: rgba(168,85,247,0.5); background: rgba(168,85,247,0.08);
        animation: stagePulse 1.5s ease-in-out infinite;
      }}
      .rm-stage.done {{ border-color: rgba(34,197,94,0.35); background: rgba(34,197,94,0.06); }}
      .rm-stage-connector {{
        width: 28px; flex-shrink: 0; display: flex; align-items: center;
        justify-content: center; color: rgba(255,255,255,0.2); font-size: 18px;
      }}
      .rm-stage-num {{
        width: 22px; height: 22px; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 11px; font-weight: 700; margin-bottom: 8px;
        background: rgba(255,255,255,0.1); color: rgba(255,255,255,0.5);
      }}
      .rm-stage.active .rm-stage-num {{ background: #a855f7; color: #fff; }}
      .rm-stage.done   .rm-stage-num {{ background: #22c55e; color: #fff; }}
      .rm-stage-title {{ font-size: 13px; font-weight: 600; color: rgba(255,255,255,0.85); margin-bottom: 3px; }}
      .rm-stage-desc  {{ font-size: 11px; color: rgba(255,255,255,0.4); }}
      .rm-badge {{
        display: inline-flex; align-items: center; gap: 5px; font-size: 10px;
        font-weight: 600; padding: 3px 9px; border-radius: 20px; margin-top: 8px;
        letter-spacing: 0.05em; text-transform: uppercase;
      }}
      .rm-badge.idle   {{ background: rgba(255,255,255,0.07); color: rgba(255,255,255,0.35); }}
      .rm-badge.active {{ background: rgba(168,85,247,0.2);   color: #c084fc; }}
      .rm-badge.done   {{ background: rgba(34,197,94,0.15);   color: #4ade80; }}
      .rm-dot {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor; }}
      .rm-dot.blinking {{ animation: dotBlink 1s ease infinite; }}
    </style>

    <div class="rm-stats-grid">
      <div class="rm-stat-card" style="--card-glow: rgba(192,132,252,0.12);">
        <div class="rm-stat-label">Songs in library</div>
        <div class="rm-stat-number" style="color:#c084fc;">{n_total}</div>
        <div class="rm-stat-sub">tracks on disk</div>
      </div>
      <div class="rm-stat-card" style="--card-glow: rgba({stems_glow},0.12);">
        <div class="rm-ring-wrap">
          <svg viewBox="0 0 42 42" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="21" cy="21" r="17" stroke="rgba(255,255,255,0.08)" stroke-width="4"/>
            <circle cx="21" cy="21" r="17" stroke="{stems_color}" stroke-width="4"
              stroke-dasharray="{stems_dash} 106.8" stroke-dashoffset="26.7" stroke-linecap="round"
              style="transition:stroke-dasharray 0.6s ease;"/>
            <text x="21" y="25" text-anchor="middle" font-size="9" font-weight="700" fill="{stems_color}">{stems_pct}%</text>
          </svg>
        </div>
        <div class="rm-stat-label">Missing stems</div>
        <div class="rm-stat-number" style="color:{stems_color};">{n_no_stems}</div>
        <div class="rm-stat-sub">{stems_sub}</div>
      </div>
      <div class="rm-stat-card" style="--card-glow: rgba({index_glow},0.12);">
        <div class="rm-ring-wrap">
          <svg viewBox="0 0 42 42" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="21" cy="21" r="17" stroke="rgba(255,255,255,0.08)" stroke-width="4"/>
            <circle cx="21" cy="21" r="17" stroke="{index_color}" stroke-width="4"
              stroke-dasharray="{index_dash} 106.8" stroke-dashoffset="26.7" stroke-linecap="round"
              style="transition:stroke-dasharray 0.6s ease;"/>
            <text x="21" y="25" text-anchor="middle" font-size="9" font-weight="700" fill="{index_color}">{index_pct}%</text>
          </svg>
        </div>
        <div class="rm-stat-label">Not yet indexed</div>
        <div class="rm-stat-number" style="color:{index_color};">{n_unindexed}</div>
        <div class="rm-stat-sub">{index_sub}</div>
      </div>
      <div class="rm-stat-card" style="--card-glow: rgba(34,197,94,0.10);">
        <div class="rm-stat-label">Indexed</div>
        <div class="rm-stat-number" style="color:#22c55e;">{n_indexed}</div>
        <div class="rm-stat-sub">ready for Smart Search</div>
      </div>
    </div>

    <div class="rm-pipeline">
      <div class="rm-stage {s1_class}">
        <div class="rm-stage-num">{s1_num}</div>
        <div class="rm-stage-title">Stem Separation</div>
        <div class="rm-stage-desc">Demucs &middot; vocals, drums, bass, other</div>
        <div class="rm-badge {s1_badge}"><span class="rm-dot {s1_dot}"></span>{s1_label}</div>
      </div>
      <div class="rm-stage-connector">&#8594;</div>
      <div class="rm-stage {s2_class}">
        <div class="rm-stage-num">{s2_num}</div>
        <div class="rm-stage-title">FLAC Compression</div>
        <div class="rm-stage-desc">Lossless &middot; ~50% size savings</div>
        <div class="rm-badge {s2_badge}"><span class="rm-dot {s2_dot}"></span>{s2_label}</div>
      </div>
      <div class="rm-stage-connector">&#8594;</div>
      <div class="rm-stage {s3_class}">
        <div class="rm-stage-num">{s3_num}</div>
        <div class="rm-stage-title">RAG Index Rebuild</div>
        <div class="rm-stage-desc">35-dim embeddings &middot; instant search</div>
        <div class="rm-badge {s3_badge}"><span class="rm-dot {s3_dot}"></span>{s3_label}</div>
      </div>
    </div>

    {msg_html}
    """, height=315)


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

if page == "🏠 Home":
    # ── Animated hero ─────────────────────────────────────────────────────────
    health = api_get("/health")
    lib    = api_get("/library?per_page=1")
    _songs  = health["library_songs"] if health else "—"
    _stems  = lib["stats"]["songs_with_stems"] if lib else "—"
    _gb     = lib["stats"]["total_size_gb"] if lib else "—"
    _gb_str = f"{_gb} GB" if isinstance(_gb, float) else str(_gb)

    st_components.html(f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background: transparent; font-family: 'Inter', 'SF Pro Display', sans-serif; overflow: hidden; }}

  @keyframes gradientShift {{
    0%   {{ background-position: 0% 50%; }}
    50%  {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
  }}
  @keyframes eqBar {{
    0%, 100% {{ transform: scaleY(0.12); }}
    50%       {{ transform: scaleY(1); }}
  }}
  @keyframes vinylSpin {{
    from {{ transform: rotate(0deg); }}
    to   {{ transform: rotate(360deg); }}
  }}
  @keyframes floatBob {{
    0%, 100% {{ transform: translateY(0); }}
    50%       {{ transform: translateY(-6px); }}
  }}
  @keyframes fadeUp {{
    from {{ opacity:0; transform:translateY(16px); }}
    to   {{ opacity:1; transform:translateY(0); }}
  }}
  @keyframes scanLine {{
    0%   {{ left: -10%; opacity: 0; }}
    10%  {{ opacity: 0.6; }}
    90%  {{ opacity: 0.6; }}
    100% {{ left: 110%; opacity: 0; }}
  }}
  @keyframes orbFloat {{
    0%, 100% {{ transform: translate(0,0) scale(1); opacity: 0.15; }}
    33%       {{ transform: translate(20px,-15px) scale(1.05); opacity: 0.22; }}
    66%       {{ transform: translate(-10px,10px) scale(0.97); opacity: 0.12; }}
  }}
  @keyframes statPop {{
    from {{ opacity:0; transform: scale(0.88); }}
    to   {{ opacity:1; transform: scale(1); }}
  }}
  @keyframes glowPulse {{
    0%,100% {{ text-shadow: 0 0 18px rgba(192,132,252,0.5); }}
    50%      {{ text-shadow: 0 0 36px rgba(192,132,252,0.9), 0 0 60px rgba(124,58,237,0.4); }}
  }}

  .hero {{
    position: relative;
    width: 100%;
    height: 260px;
    background: linear-gradient(135deg, #080812 0%, #0d0820 40%, #0a0616 70%, #080812 100%);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 24px;
    overflow: hidden;
    display: flex;
    align-items: center;
    padding: 0 40px;
    gap: 44px;
  }}
  /* Animated top border */
  .hero::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #c084fc, #818cf8, #ec4899, #c084fc, transparent);
    background-size: 300% 100%;
    animation: shimmer 3s linear infinite;
  }}
  @keyframes shimmer {{
    0% {{ background-position: -300% 0; }}
    100% {{ background-position: 300% 0; }}
  }}
  /* Grid lines inside hero */
  .hero::after {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
      linear-gradient(rgba(124,58,237,0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(124,58,237,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    mask-image: radial-gradient(ellipse 60% 60% at 50% 50%, black 20%, transparent 100%);
    -webkit-mask-image: radial-gradient(ellipse 60% 60% at 50% 50%, black 20%, transparent 100%);
    pointer-events: none;
  }}

  /* Background orbs */
  .orb {{
    position: absolute;
    border-radius: 50%;
    pointer-events: none;
  }}
  .orb1 {{
    width: 340px; height: 340px;
    background: radial-gradient(circle, rgba(124,58,237,0.18) 0%, transparent 70%);
    top: -100px; right: 60px;
    animation: orbFloat 9s ease-in-out infinite;
  }}
  .orb2 {{
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(79,70,229,0.12) 0%, transparent 70%);
    bottom: -80px; left: 280px;
    animation: orbFloat 12s ease-in-out infinite reverse;
  }}

  /* Scan line */
  .scanline {{
    position: absolute;
    top: 0; height: 100%; width: 3px;
    background: linear-gradient(180deg, transparent, rgba(192,132,252,0.6), transparent);
    animation: scanLine 5s ease-in-out infinite;
    pointer-events: none;
  }}

  /* Vinyl */
  .vinyl-wrap {{
    position: relative;
    flex-shrink: 0;
    animation: floatBob 4s ease-in-out infinite;
  }}
  .vinyl {{
    width: 140px; height: 140px;
    border-radius: 50%;
    background: conic-gradient(
      from 0deg,
      #1a0a2e 0deg, #2d1b4e 30deg, #1a0a2e 60deg,
      #2d1b4e 90deg, #1a0a2e 120deg, #2d1b4e 150deg,
      #1a0a2e 180deg, #2d1b4e 210deg, #1a0a2e 240deg,
      #2d1b4e 270deg, #1a0a2e 300deg, #2d1b4e 330deg,
      #1a0a2e 360deg
    );
    box-shadow:
      0 0 0 3px rgba(124,58,237,0.4),
      0 0 30px rgba(124,58,237,0.3),
      0 0 60px rgba(124,58,237,0.1),
      inset 0 0 20px rgba(0,0,0,0.7);
    animation: vinylSpin 4s linear infinite;
    display: flex; align-items: center; justify-content: center;
  }}
  .vinyl-groove {{
    position: absolute;
    border-radius: 50%;
    border: 1px solid rgba(192,132,252,0.12);
  }}
  .vinyl-center {{
    width: 32px; height: 32px;
    border-radius: 50%;
    background: radial-gradient(circle, #c084fc 0%, #7c3aed 60%, #1a0a2e 100%);
    box-shadow: 0 0 14px rgba(192,132,252,0.8);
    z-index: 2;
    position: absolute;
  }}

  /* Title block */
  .title-block {{
    flex: 1;
    animation: fadeUp 0.6s ease both;
  }}
  .hero-title {{
    font-size: 3.2rem;
    font-weight: 900;
    line-height: 1.05;
    background: linear-gradient(270deg, #c084fc, #818cf8, #60a5fa, #ec4899, #a78bfa, #c084fc);
    background-size: 600% 600%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 4s ease infinite;
    letter-spacing: -1.5px;
    filter: drop-shadow(0 0 20px rgba(192,132,252,0.15));
  }}
  .hero-sub {{
    font-size: 0.82rem;
    color: #5858a0;
    margin-top: 10px;
    font-weight: 500;
    letter-spacing: 0.04em;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
  }}
  .hero-sub span {{
    color: #7c7cbb;
    border-bottom: 1px solid rgba(124,58,237,0.3);
    padding-bottom: 1px;
  }}

  /* Stats row */
  .stats-row {{
    display: flex;
    gap: 14px;
    margin-top: 18px;
  }}
  .stat-chip {{
    background: rgba(124,58,237,0.1);
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 10px;
    padding: 8px 16px;
    font-size: 0.75rem;
    color: #b090e0;
    font-weight: 600;
    animation: statPop 0.5s cubic-bezier(0.22,1,0.36,1) both;
    transition: all 0.25s ease;
    cursor: default;
    font-family: 'JetBrains Mono', monospace;
    position: relative;
    overflow: hidden;
  }}
  .stat-chip::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(135deg, rgba(192,132,252,0.08), transparent);
    opacity: 0;
    transition: opacity 0.3s;
  }}
  .stat-chip:hover::before {{ opacity: 1; }}
  .stat-chip:hover {{
    border-color: rgba(192,132,252,0.5);
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(124,58,237,0.2);
  }}
  .stat-chip:nth-child(1) {{ animation-delay: 0.2s; }}
  .stat-chip:nth-child(2) {{ animation-delay: 0.3s; }}
  .stat-chip:nth-child(3) {{ animation-delay: 0.4s; }}
  .stat-chip .val {{
    font-size: 1.15rem;
    font-weight: 800;
    color: #e0c8ff;
    text-shadow: 0 0 10px rgba(192,132,252,0.3);
  }}

  /* Equalizer */
  .eq-wrap {{
    display: flex;
    align-items: flex-end;
    gap: 4px;
    height: 80px;
    flex-shrink: 0;
  }}
  .eq-bar {{
    width: 8px;
    border-radius: 4px 4px 2px 2px;
    transform-origin: bottom;
    animation: eqBar var(--dur) ease-in-out infinite;
    animation-delay: var(--delay);
  }}
</style>
</head>
<body>
<div class="hero">
  <div class="orb orb1"></div>
  <div class="orb orb2"></div>
  <div class="scanline"></div>

  <!-- Vinyl disc -->
  <div class="vinyl-wrap">
    <div class="vinyl">
      <div class="vinyl-groove" style="width:120px;height:120px;"></div>
      <div class="vinyl-groove" style="width:100px;height:100px;"></div>
      <div class="vinyl-groove" style="width:80px;height:80px;"></div>
      <div class="vinyl-groove" style="width:60px;height:60px;"></div>
      <div class="vinyl-center"></div>
    </div>
  </div>

  <!-- Title + stats -->
  <div class="title-block">
    <div class="hero-title">AI RemixMate</div>
    <div class="hero-sub"><span>Demucs</span> · <span>Librosa</span> · <span>Camelot</span> · <span>RAG Index</span> · <span>GPU-Accelerated</span></div>
    <div class="stats-row">
      <div class="stat-chip"><span class="val">{_songs}</span> songs</div>
      <div class="stat-chip"><span class="val">{_stems}</span> stem-split</div>
      <div class="stat-chip"><span class="val">{_gb_str}</span> library</div>
    </div>
  </div>

  <!-- Equalizer bars -->
  <div class="eq-wrap">
    {''.join(
      f'<div class="eq-bar" style="height:80px; background: linear-gradient(180deg, #c084fc, #7c3aed); --dur:{0.5 + i*0.07:.2f}s; --delay:{i*0.06:.2f}s; opacity:{0.6 + (i%3)*0.13:.2f};"></div>'
      for i in range(16)
    )}
  </div>
</div>
</body>
</html>
    """, height=280)

    # ── Stat metrics row ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Songs in Library", _songs)
    with col2:
        st.metric("Library Size", _gb_str)
    with col3:
        st.metric("Songs with Stems", _stems)
    with col4:
        cap = lib["stats"]["cap_gb"] if lib else "—"
        st.metric("Storage Cap", f"{cap} GB" if isinstance(cap, float) else cap)

    st.divider()

    # ── Feature cards ─────────────────────────────────────────────────────────
    st_components.html("""
<!DOCTYPE html>
<html><head>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:transparent; font-family:'Inter','SF Pro Display',sans-serif; }
  @keyframes fadeUp {
    from { opacity:0; transform:translateY(14px); }
    to   { opacity:1; transform:translateY(0); }
  }
  @keyframes iconBob {
    0%, 100% { transform: translateY(0) scale(1); }
    50%       { transform: translateY(-3px) scale(1.08); }
  }
  @keyframes glowBorder {
    0%, 100% { border-color: rgba(42,42,72,0.7); }
    50%       { border-color: rgba(192,132,252,0.3); }
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    padding: 4px 0;
  }
  .card {
    background: rgba(18,18,30,0.85);
    border: 1px solid rgba(42,42,72,0.7);
    border-radius: 16px;
    padding: 20px 22px;
    transition: transform 0.28s cubic-bezier(0.22,1,0.36,1),
                border-color 0.3s,
                box-shadow 0.3s,
                background 0.3s;
    animation: fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) both;
    cursor: default;
    position: relative;
    overflow: hidden;
  }
  .card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(circle at var(--mx, 50%) var(--my, 50%),
                rgba(192,132,252,0.08), transparent 60%);
    opacity: 0;
    transition: opacity 0.3s;
    pointer-events: none;
  }
  .card:hover::before { opacity: 1; }
  .card:hover {
    transform: translateY(-5px) scale(1.01);
    border-color: rgba(192,132,252,0.4);
    box-shadow: 0 12px 40px rgba(124,58,237,0.18), 0 0 0 1px rgba(192,132,252,0.1);
    background: rgba(22,18,36,0.9);
  }
  .card:hover .icon { animation: iconBob 0.6s ease; }
  .card:nth-child(1) { animation-delay:0.05s; }
  .card:nth-child(2) { animation-delay:0.10s; }
  .card:nth-child(3) { animation-delay:0.15s; }
  .card:nth-child(4) { animation-delay:0.20s; }
  .card:nth-child(5) { animation-delay:0.25s; }
  .card:nth-child(6) { animation-delay:0.30s; }
  .card:nth-child(7) { animation-delay:0.35s; }
  .card:nth-child(8) { animation-delay:0.40s; }
  .card:nth-child(9) { animation-delay:0.45s; }
  .icon { font-size:1.8rem; margin-bottom:10px; display:inline-block; transition: transform 0.2s; }
  .title { font-size:0.9rem; font-weight:700; color:#d4d4f0; margin-bottom:5px; letter-spacing:-0.01em; }
  .desc  { font-size:0.76rem; color:#606090; line-height:1.55; }
</style>
</head><body>
<div class="grid">
  <div class="card" onmousemove="this.style.setProperty('--mx',event.offsetX+'px');this.style.setProperty('--my',event.offsetY+'px')">
    <div class="icon">🎹</div>
    <div class="title">Camelot Wheel</div>
    <div class="desc">Harmonic key compatibility scoring — instant, no download needed</div>
  </div>
  <div class="card" onmousemove="this.style.setProperty('--mx',event.offsetX+'px');this.style.setProperty('--my',event.offsetY+'px')">
    <div class="icon">🔬</div>
    <div class="title">Genre Detection</div>
    <div class="desc">Spectral centroid, sub-bass ratio, ZCR &amp; dynamic range analysis</div>
  </div>
  <div class="card" onmousemove="this.style.setProperty('--mx',event.offsetX+'px');this.style.setProperty('--my',event.offsetY+'px')">
    <div class="icon">🎛️</div>
    <div class="title">Stem-Aware Mixing</div>
    <div class="desc">Per-stem similarity scoring — drums, bass, vocals mixed independently</div>
  </div>
  <div class="card" onmousemove="this.style.setProperty('--mx',event.offsetX+'px');this.style.setProperty('--my',event.offsetY+'px')">
    <div class="icon">🔮</div>
    <div class="title">RAG Vector Index</div>
    <div class="desc">35-dim embeddings · cosine similarity · sub-millisecond retrieval</div>
  </div>
  <div class="card" onmousemove="this.style.setProperty('--mx',event.offsetX+'px');this.style.setProperty('--my',event.offsetY+'px')">
    <div class="icon">🗜️</div>
    <div class="title">FLAC Compression</div>
    <div class="desc">Lossless stems at 50% WAV size · transparent to DJ engine</div>
  </div>
  <div class="card" onmousemove="this.style.setProperty('--mx',event.offsetX+'px');this.style.setProperty('--my',event.offsetY+'px')">
    <div class="icon">⚡</div>
    <div class="title">Async Job Queue</div>
    <div class="desc">Download, analyze, remix run in parallel background threads</div>
  </div>
  <div class="card" onmousemove="this.style.setProperty('--mx',event.offsetX+'px');this.style.setProperty('--my',event.offsetY+'px')">
    <div class="icon">🚀</div>
    <div class="title">GPU Accelerated</div>
    <div class="desc">MPS / CUDA auto-detect · 50-100x faster similarity search</div>
  </div>
  <div class="card" onmousemove="this.style.setProperty('--mx',event.offsetX+'px');this.style.setProperty('--my',event.offsetY+'px')">
    <div class="icon">🔐</div>
    <div class="title">Hardened Security</div>
    <div class="desc">TLS, CORS lockdown, path traversal prevention, XSRF protection</div>
  </div>
  <div class="card" onmousemove="this.style.setProperty('--mx',event.offsetX+'px');this.style.setProperty('--my',event.offsetY+'px')">
    <div class="icon">🎧</div>
    <div class="title">ITU Mastering</div>
    <div class="desc">BS.1770-4 LUFS + true-peak limiter — broadcast-grade loudness</div>
  </div>
</div>
</body></html>
    """, height=400)

    # ── Quick Actions ─────────────────────────────────────────────────────────
    _section_header("⚡", "Quick Actions", "Jump straight to any workflow — one click")

    qa_cols = st.columns(3)
    _quick_actions = [
        ("⬇️", "Download a Song",       "⬇️ Download",         "#7c3aed"),
        ("🎛️", "DJ Remix",              "🎛️ DJ Remix",          "#4f46e5"),
        ("🎚️", "DJ Chain",              "🎚️ DJ Chain",          "#6d28d9"),
        ("🔮", "Smart Search",          "🔮 Smart Search",      "#0ea5e9"),
        ("🧪", "Instrument Lab",        "🧪 Instrument Lab",    "#f59e0b"),
        ("🔍", "Analyze a Song",        "🔍 Analyze",           "#059669"),
    ]
    for i, (icon, label, dest, color) in enumerate(_quick_actions):
        with qa_cols[i % 3]:
            st.markdown(
                f"""<div style="
                    background: linear-gradient(135deg, {color}18, {color}08);
                    border: 1px solid {color}33;
                    border-radius: 16px; padding: 18px 20px; margin-bottom: 10px;
                    cursor: pointer;
                    transition: all 0.28s cubic-bezier(0.22,1,0.36,1);
                    position: relative; overflow: hidden;
                " onmouseenter="this.style.transform='translateY(-4px) scale(1.02)';this.style.borderColor='{color}88';this.style.boxShadow='0 8px 32px {color}22'"
                  onmouseleave="this.style.transform='';this.style.borderColor='{color}33';this.style.boxShadow='none'"
                  onmousedown="this.style.transform='scale(0.97)'"
                  onmouseup="this.style.transform='translateY(-4px) scale(1.02)'"
                >
                    <div style="font-size:1.8rem; margin-bottom:6px; display:inline-block;
                         transition: transform 0.2s;">{icon}</div>
                    <div style="font-size:0.92rem; font-weight:600; color:#e0e0ff; letter-spacing:-0.01em;">{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button(f"Go →", key=f"qa_{dest}", use_container_width=True):
                _go_to_page(dest)

    st.divider()

    # ── Recent Mixes ──────────────────────────────────────────────────────────
    import pathlib as _pathlib

    _outputs_root = _pathlib.Path(__file__).parents[2] / "outputs"
    _recent_mixes = []
    if _outputs_root.exists():
        for _d in sorted(_outputs_root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not _d.is_dir():
                continue
            if _d.name.startswith("beat_"):  # skip beat previews
                continue
            _wavs = list(_d.glob("*.wav"))
            if _wavs:
                _wav = _wavs[0]
                _kind = "🎛️ DJ Mix" if _d.name.startswith("dj_") else "🎚️ Chain Mix"
                _size_mb = round(_wav.stat().st_size / 1_048_576, 1)
                import time as _time
                _age_min = int((_time.time() - _wav.stat().st_mtime) / 60)
                _age_str = (
                    f"{_age_min}m ago" if _age_min < 60
                    else f"{_age_min // 60}h ago" if _age_min < 1440
                    else f"{_age_min // 1440}d ago"
                )
                _recent_mixes.append({
                    "kind": _kind,
                    "name": _wav.name,
                    "session": _d.name,
                    "size_mb": _size_mb,
                    "age": _age_str,
                    "wav": _wav,
                })

    if _recent_mixes:
        st.markdown("### 🎵 Recent Mixes")
        st.caption(f"Last {min(len(_recent_mixes), 6)} rendered outputs.")
        _mix_cols = st.columns(2)
        for _mi, _mix in enumerate(_recent_mixes[:6]):
            with _mix_cols[_mi % 2]:
                with st.expander(f"{_mix['kind']}  ·  {_mix['age']}  ·  {_mix['size_mb']} MB"):
                    st.caption(_mix["name"])
                    _audio_url = f"{API_PUBLIC}/outputs/{_mix['session']}/{requests.utils.quote(_mix['name'])}"
                    st.markdown(f"[▶ Stream mix]({_audio_url})")
                    st.audio(str(_mix["wav"]))
    else:
        st.markdown("### 🎵 Recent Mixes")
        _empty_state("🎵", "No mixes rendered yet", "Head to DJ Remix or DJ Chain to create your first mix")


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

elif page == "📚 Library":
    st.title("📚 Song Library")

    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("Search songs", placeholder="e.g. REZZ, Anyma, house…")
    with col2:
        per_page = st.selectbox("Per page", [25, 50, 100], index=0)

    params = {"per_page": per_page}
    if search:
        params["search"] = search

    data = api_get("/library", params=params)
    if not data:
        st.stop()

    stats = data["stats"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total songs", stats["total_songs"])
    col2.metric("Library size", f"{stats['total_size_gb']} GB / {stats['cap_gb']} GB")
    col3.metric("With stems", stats["songs_with_stems"])

    st.divider()

    songs = data["songs"]
    if not songs:
        _empty_state("🔍", "No songs match your search", "Try a different keyword or clear the search")
    else:
        st.caption(f"Showing {len(songs)} songs")
        for song in songs:
            with st.expander(f"**{song['name']}**  ·  {song['size_mb']} MB"):
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f"**WAV:** {'✅' if song['has_full_wav'] else '❌'}")
                c2.markdown(f"**Stems:** {', '.join(song['stems']) or 'none'}")
                c3.markdown(f"**Source:** {song.get('source') or '—'}")
                c4.markdown(f"**License:** `{song.get('license_type') or '—'}`")

                if song["has_full_wav"]:
                    audio_url = f"{API}/library/{requests.utils.quote(song['name'])}/audio"
                    st.markdown(f"[▶ Stream audio]({audio_url})")

                # ── Quick action buttons ──────────────────────────────────
                st.divider()
                qa1, qa2, qa3, qa4 = st.columns(4)
                if qa1.button("🔍 Analyze", key=f"lib_analyze_{song['name'][:30]}", use_container_width=True):
                    st.session_state.pending_analyze_song = song["name"]
                    _go_to_page("🔍 Analyze")
                if qa2.button("🎛️ Remix A→", key=f"lib_remix_a_{song['name'][:30]}", use_container_width=True):
                    st.session_state.pending_remix_a = song["name"]
                    _go_to_page("🎛️ DJ Remix")
                if qa3.button("🔗 Compat.", key=f"lib_compat_{song['name'][:30]}", use_container_width=True):
                    st.session_state.pending_compat_a = song["name"]
                    _go_to_page("🔗 Compatibility")
                if qa4.button("🔮 Similar", key=f"lib_similar_{song['name'][:30]}", use_container_width=True):
                    st.session_state.pending_similar = song["name"]
                    _go_to_page("🔮 Smart Search")


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------

elif page == "🔗 Compatibility":
    st.title("🔗 Compatibility Check")
    st.markdown("Instant Camelot + BPM compatibility — uses local analysis, no download needed.")

    # Dynamic song list — all names, no pagination
    song_names = _fetch_all_song_names()

    # Pre-select from Library quick-action
    _default_compat_a = st.session_state.pop("pending_compat_a", None)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Song A")
        song_a = _song_picker("Select song A", song_names, key="compat_a", default=_default_compat_a)
    with col2:
        st.subheader("Song B")
        song_b = _song_picker("Select song B", song_names, key="compat_b")

    if st.button("Check Compatibility", type="primary", use_container_width=True):
        with st.spinner("Analysing…"):
            result = api_post("/compatibility", {"song_a": song_a, "song_b": song_b})

        if result:
            st.divider()
            compat_label = "✅ COMPATIBLE" if result["compatible"] else "⚠️ MARGINAL"
            st.markdown(f"## {compat_label}")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Overall", f"{result['overall']:.0%}")
            col2.metric("BPM Score", f"{result['bpm_score']:.0%}")
            col3.metric("Key Score",  f"{result['key_score']:.0%}")
            col4.metric("Energy",     f"{result['energy_score']:.0%}")

            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{song_a}**")
                st.markdown(f"BPM: `{result['bpm_a']}`")
                st.markdown(f"Camelot: `{result.get('camelot_a') or 'unknown'}`")
                st.markdown(f"Genre: `{result.get('genre_a') or 'unknown'}`")
            with col2:
                st.markdown(f"**{song_b}**")
                st.markdown(f"BPM: `{result['bpm_b']}`")
                st.markdown(f"Camelot: `{result.get('camelot_b') or 'unknown'}`")
                st.markdown(f"Genre: `{result.get('genre_b') or 'unknown'}`")

            bpm_diff = abs(result["bpm_a"] - result["bpm_b"])
            stretch = round(result["bpm_a"] / result["bpm_b"], 3) if result["bpm_b"] else 1.0
            st.caption(f"BPM delta: {bpm_diff:.1f} · Stretch ratio: {stretch}")


# ---------------------------------------------------------------------------
# DJ Remix
# ---------------------------------------------------------------------------

elif page == "🎛️ DJ Remix":
    st.title("🎛️ DJ Remix Studio")
    st.markdown("Render a phrase-aligned DJ transition between two library songs.")

    # Dynamic song list — all songs with wav
    song_names = _fetch_all_song_names(with_wav=True)

    # ── Session state for recommendation cache ─────────────────────────────
    if "dj_rec_cache" not in st.session_state:
        st.session_state.dj_rec_cache: dict = {}

    # Pre-select Song A from Library quick-action
    _default_remix_a = st.session_state.pop("pending_remix_a", None)

    # ── Song A selector ─────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Song A  →  exits")
        song_a = _song_picker("Song A", song_names, key="dj_a", default=_default_remix_a)

        # Auto-fetch recommendations whenever a new Song A is selected
        if song_a and song_a not in st.session_state.dj_rec_cache:
            with st.spinner("🔍 Finding compatible songs…"):
                rec_data = api_get(
                    f"/recommend/{requests.utils.quote(song_a)}",
                    params={"limit": 5},
                )
            st.session_state.dj_rec_cache[song_a] = (
                rec_data.get("recommendations", []) if rec_data else []
            )

    # ── Song B selector with ranked recommendations ─────────────────────────
    recommendations = st.session_state.dj_rec_cache.get(song_a, [])

    with col2:
        st.subheader("Song B  →  enters")

        if recommendations:
            rec_names   = [r["name"] for r in recommendations]
            other_songs = [s for s in song_names if s not in rec_names and s != song_a]
            song_b_opts = rec_names + other_songs
        else:
            song_b_opts = [s for s in song_names if s != song_a]

        song_b = _song_picker("Song B", song_b_opts, key="dj_b")

    # ── Recommendation badge row ─────────────────────────────────────────────
    if recommendations:
        st.markdown("#### 🎯 Top picks for Song B")
        badge_cols = st.columns(len(recommendations))
        for i, rec in enumerate(recommendations):
            icon = score_color(rec["overall"])
            short_name = rec["name"][:22] + ("…" if len(rec["name"]) > 22 else "")
            badge_cols[i].metric(
                label=short_name,
                value=f"{rec['overall']:.0%}",
                delta=f"{rec['bpm']} BPM",
            )
            badge_cols[i].caption(icon)
        st.caption(
            "Scores based on BPM proximity (including half/double-time matching). "
            "🟢 ≥ 75 %   🟡 ≥ 50 %   🔴 < 50 %"
        )
        st.divider()
    elif song_a:
        st.info(
            "No BPM cache found yet — run **Analyze** on a few songs to warm up "
            "the recommendation engine, or wait while it builds the cache in the background."
        )

    # ── Transition settings ─────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        bars = st.select_slider("Transition length (bars)", options=[8, 16, 32], value=16)
    with col2:
        preset = st.selectbox("Genre preset", ["auto", "house", "techno", "hiphop", "trap",
                                                "pop", "rnb", "dnb", "ambient", "rock", "jazz"])

    # ── Beat Bridge ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🥁 Beat Bridge")
    st.caption(
        "Layer a drum beat underneath the transition — it fades in with Song A, "
        "peaks at the midpoint, then fades out as Song B takes over."
    )

    bridge_mode = st.radio(
        "Bridge beat source",
        options=["none", "auto", "strudel"],
        format_func=lambda x: {
            "none":    "🚫 None — clean crossfade only",
            "auto":    "⚡ Auto-Generate — instant Python synthesis",
            "strudel": "🎨 Strudel — design your own, I'll mix it in",
        }[x],
        horizontal=True,
        key="bridge_mode",
    )

    bridge_beat_genre   = "auto"
    bridge_beat_intensity = 0.38
    bridge_beat_path_val  = None

    if bridge_mode == "auto":
        col1, col2 = st.columns(2)
        with col1:
            bridge_beat_genre = st.selectbox(
                "Beat genre",
                ["auto", "techno", "house", "hiphop", "trap", "dnb", "ambient"],
                key="bridge_genre",
            )
        with col2:
            bridge_beat_intensity = st.slider(
                "Intensity", 0.1, 0.8, 0.38, 0.05,
                help="How loud the beat is relative to the songs",
                key="bridge_intensity",
            )

        # Preview button — generates the beat and plays it inline
        if st.button("🔊 Preview beat", key="preview_beat"):
            with st.spinner("Synthesising…"):
                preview = api_get(
                    "/beat/synthesize",
                    params={
                        "bpm":       recommendations[0]["bpm"] if recommendations else 128,
                        "genre":     bridge_beat_genre,
                        "bars":      4,
                        "intensity": 0.7,
                    },
                )
            if preview and preview.get("audio_url"):
                st.audio(f"{API}{preview['audio_url']}", format="audio/wav")
                st.caption(f"4-bar preview at {preview['bpm']} BPM · genre: {preview['genre']}")
            else:
                st.warning("Could not generate preview — check the API is running.")

    elif bridge_mode == "strudel":
        # Derive a sensible BPM for the Strudel code
        src_bpm = recommendations[0]["bpm"] if recommendations else 128.0

        # Fetch an auto-generated starter pattern from the API
        with st.spinner("Generating starter pattern…"):
            synth_data = api_get(
                "/beat/synthesize",
                params={"bpm": src_bpm, "genre": preset, "bars": 4},
            )
        strudel_snippet = (
            synth_data.get("strudel_code", "") if synth_data else ""
        ) or (
            f'stack(\n'
            f'  s("bd sd bd sd").bank("RolandTR909"),\n'
            f'  s("hh*8").gain(0.6)\n'
            f').cpm({int(src_bpm)} / 4)'
        )

        st.markdown("**🎹 Live Strudel Editor** — edit and play your pattern right here:")
        st.caption(
            "▶ **Ctrl+Enter** to play · ■ **Ctrl+.** to stop · "
            "⏺ click the **record dot** inside the editor to export WAV"
        )

        # ── Embedded Strudel REPL ─────────────────────────────────────────
        _strudel_embed(strudel_snippet, height=420)

        st.divider()
        st.markdown("**📤 Upload your recording to blend into the mix:**")
        st.caption("Hit record inside the editor above, download the WAV, then drop it here.")
        uploaded = st.file_uploader(
            "Drop your Strudel recording here",
            type=["wav", "mp3", "ogg"],
            key="strudel_upload",
        )
        if uploaded:
            with st.spinner("Uploading beat…"):
                resp = requests.post(
                    f"{API}/beat/upload",
                    files={"file": (uploaded.name, uploaded.read(), uploaded.type)},
                )
            if resp.ok:
                upload_data = resp.json()
                bridge_beat_path_val = upload_data.get("audio_path")
                st.success(f"✅ Beat uploaded: `{upload_data['filename']}` ({upload_data['size_kb']} KB)")
            else:
                st.error("Upload failed — check the API is running.")

        col1, col2 = st.columns(2)
        with col1:
            bridge_beat_genre = st.selectbox(
                "Beat genre label", ["auto", "techno", "house", "hiphop", "trap", "dnb", "ambient"],
                key="strudel_genre_label",
            )
        with col2:
            bridge_beat_intensity = st.slider(
                "Intensity", 0.1, 0.8, 0.38, 0.05,
                key="strudel_intensity",
            )

    # ── Render button ────────────────────────────────────────────────────────
    st.divider()
    api_mode = "file" if (bridge_mode == "strudel" and bridge_beat_path_val) else bridge_mode
    if bridge_mode == "strudel" and not bridge_beat_path_val:
        api_mode = "none"

    if st.button("🎚️ Render DJ Mix", type="primary", use_container_width=True):
        jid = submit_job(
            name=f"{song_a[:20]} → {song_b[:20]}",
            endpoint="/dj-remix",
            body={
                "song_a":                song_a,
                "song_b":                song_b,
                "transition_bars":       bars,
                "preset":                preset,
                "bridge_beat_mode":      api_mode,
                "bridge_beat_genre":     bridge_beat_genre,
                "bridge_beat_intensity": bridge_beat_intensity,
                "bridge_beat_path":      bridge_beat_path_val,
            },
            icon="🎛️",
            page_key="remix",
        )
        if jid:
            st.toast("🎛️ Mix rendering in background — you can use other pages now!", icon="🚀")

    # ── Live inline progress + result ────────────────────────────────────────
    remix_info = _inline_job_progress("remix")
    if remix_info and remix_info.get("status") == "done" and remix_info.get("result"):
        r = remix_info["result"]

        _bpm_a = r.get("bpm_a", "?")
        _bpm_b = r.get("bpm_b", "?")
        _sa    = r.get("song_a", song_a)
        _sb    = r.get("song_b", song_b)
        st_components.html(f"""
<!DOCTYPE html><html><head>
<style>
  * {{ margin:0;padding:0;box-sizing:border-box; }}
  body {{ background:transparent; font-family:'Inter',sans-serif; overflow:hidden; }}
  @keyframes eqBar {{ 0%,100% {{ transform:scaleY(0.12); }} 50% {{ transform:scaleY(1); }} }}
  @keyframes fadeUp {{ from {{ opacity:0;transform:translateY(10px); }} to {{ opacity:1;transform:translateY(0); }} }}
  @keyframes scanLine {{ 0% {{ left:-8%;opacity:0; }} 8% {{ opacity:0.5; }} 92% {{ opacity:0.5; }} 100% {{ left:108%;opacity:0; }} }}
  @keyframes glowText {{ 0%,100% {{ text-shadow:0 0 12px rgba(192,132,252,0.4); }} 50% {{ text-shadow:0 0 28px rgba(192,132,252,0.9); }} }}
  .banner {{ position:relative;overflow:hidden;background:linear-gradient(135deg,#0c081a,#110d22,#0c081a);border:1px solid rgba(124,58,237,0.3);border-radius:16px;padding:18px 28px;display:flex;align-items:center;gap:24px;animation:fadeUp 0.5s ease both; }}
  .scanline {{ position:absolute;top:0;height:100%;width:2px;background:linear-gradient(180deg,transparent,rgba(192,132,252,0.5),transparent);animation:scanLine 4s ease-in-out infinite;pointer-events:none; }}
  .dot {{ width:8px;height:8px;border-radius:50%;background:#c084fc;box-shadow:0 0 10px rgba(192,132,252,0.8);animation:eqBar 0.8s ease-in-out infinite;flex-shrink:0; }}
  .label {{ font-size:0.68rem;color:#6060a0;font-weight:600;text-transform:uppercase;letter-spacing:0.1em; }}
  .track-name {{ font-size:1.05rem;font-weight:700;color:#e0d8ff;margin-top:2px;animation:glowText 3s ease-in-out infinite; }}
  .eq-wrap {{ display:flex;align-items:flex-end;gap:3px;height:40px;flex-shrink:0;margin-left:auto; }}
  .eq-bar {{ width:5px;border-radius:3px 3px 1px 1px;transform-origin:bottom;animation:eqBar var(--dur) ease-in-out infinite;animation-delay:var(--delay);background:linear-gradient(180deg,#c084fc,#7c3aed); }}
  .bpm-tag {{ background:rgba(124,58,237,0.18);border:1px solid rgba(124,58,237,0.3);border-radius:6px;padding:4px 10px;font-size:0.75rem;font-weight:700;color:#c084fc;flex-shrink:0; }}
</style></head><body>
<div class="banner">
  <div class="scanline"></div>
  <div class="now-playing"><div class="dot"></div></div>
  <div style="flex:1;min-width:0;">
    <div class="label">Now Playing</div>
    <div class="track-name">{_sa} <span style="color:#5050a0;font-weight:400;">→</span> {_sb}</div>
  </div>
  <div class="bpm-tag">{_bpm_a} → {_bpm_b} BPM</div>
  <div class="eq-wrap">
    {''.join(f'<div class="eq-bar" style="height:40px;--dur:{0.45+i*0.06:.2f}s;--delay:{i*0.055:.2f}s;opacity:{0.55+(i%3)*0.15:.2f};"></div>' for i in range(18))}
  </div>
</div>
</body></html>
        """, height=100)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("BPM A→B", f"{r['bpm_a']} → {r['bpm_b']}")
        col2.metric("Transition", f"{r['transition_bars']} bars · {r['transition_sec']}s")
        col3.metric("Mix duration", f"{r['duration_sec']}s")
        col4.metric("Bridge beat", r.get("bridge_beat", "none"))
        stem_badge = "🎛️ Stem blend" if r.get("stem_blend") else "🎵 Standard mix"
        col5.metric("Mix mode", stem_badge)
        if r.get("stem_blend"):
            st.info("✨ **Intelligent stem blend active** — drums, bass, vocals and melody mixed independently.", icon="🎛️")

        out_path = r.get("output", "")
        if out_path:
            parts = out_path.split("/")
            try:
                outputs_idx = parts.index("outputs")
                session     = parts[outputs_idx + 1]
                filename    = parts[-1]
                audio_url   = f"{API}/outputs/{session}/{filename}"
                st.markdown("**▶ Play mix**")
                st.audio(audio_url, format="audio/wav")
                st.caption(f"Saved to: `{out_path}`")
            except Exception:
                st.caption(f"Saved to: `{out_path}`")

            with st.expander("📊 View spectrogram of mix", expanded=False):
                sc1, sc2, sc3 = st.columns(3)
                spec_dur    = sc1.slider("Duration (s)", 30, 180, 60, key="remix_spec_dur")
                spec_cmap   = sc2.selectbox("Color map", ["magma", "inferno", "plasma", "viridis", "hot"], key="remix_spec_cmap")
                spec_chroma = sc3.checkbox("Show chromagram", value=False, key="remix_spec_chroma")
                _show_spectrogram(out_path, title=f"{r.get('song_a','?')} → {r.get('song_b','?')}", duration=spec_dur, colormap=spec_cmap, show_chroma=spec_chroma)

    elif remix_info and remix_info.get("status") == "failed":
        st.error(f"Mix failed: {remix_info.get('result', {})}")


# ---------------------------------------------------------------------------
# DJ Chain — N-song continuous mix
# ---------------------------------------------------------------------------

elif page == "🎚️ DJ Chain":
    st.title("🎚️ DJ Chain")
    st.markdown(
        "Build a **continuous mix from any number of songs**. "
        "Each transition is phrase-locked and beat-synced — one seamless set."
    )

    all_songs = _fetch_all_song_names(with_wav=True)

    if not all_songs:
        _empty_state("🎵", "No songs with audio in library", "Download some songs first from the Download page")
        st.stop()

    st.divider()

    # ── Number of songs ──────────────────────────────────────────────────
    n_songs = st.number_input(
        "How many songs do you want to mix?",
        min_value=2,
        max_value=8,
        value=3,
        step=1,
        help="Choose between 2 and 8 songs. They'll play in order with automatic DJ transitions.",
    )
    n_songs = int(n_songs)

    st.caption(f"You'll get **{n_songs - 1} transition{'s' if n_songs > 2 else ''}** in a single continuous mix.")
    st.divider()

    # ── Song selectors ───────────────────────────────────────────────────
    st.markdown("### 🎵 Set the order")
    st.caption("Songs play top-to-bottom. Each flows into the next.")

    # Recommendation cache: keyed by song name
    if "chain_rec_cache" not in st.session_state:
        st.session_state.chain_rec_cache: dict = {}

    selected_songs: list = []

    for i in range(n_songs):
        col_label, col_select = st.columns([1, 5])
        with col_label:
            st.markdown(
                f"<div style='padding-top:36px;color:#c084fc;font-weight:700;font-size:1.1rem'>"
                f"{'▶' if i == 0 else '⬇'} {i + 1}</div>",
                unsafe_allow_html=True,
            )
        with col_select:
            # Exclude already-chosen songs from the dropdown
            # (allow repeats only if user explicitly wants them — no hard block)
            prev = selected_songs[-1] if selected_songs else None

            # Auto-fetch recommendations for prev → this slot
            if prev and prev not in st.session_state.chain_rec_cache:
                with st.spinner(f"Finding matches for slot {i + 1}…"):
                    rec_data = api_get(
                        f"/recommend/{requests.utils.quote(prev)}",
                        params={"limit": 5},
                    )
                st.session_state.chain_rec_cache[prev] = (
                    rec_data.get("recommendations", []) if rec_data else []
                )

            recs = st.session_state.chain_rec_cache.get(prev, []) if prev else []
            rec_names   = [r["name"] for r in recs]
            other_songs = [s for s in all_songs if s not in rec_names]
            opts        = rec_names + other_songs

            # Default index: skip songs already picked
            default_idx = 0
            for candidate_idx, name in enumerate(opts):
                if name not in selected_songs:
                    default_idx = candidate_idx
                    break

            label = (
                f"Song {i + 1}  ({'first — sets the BPM reference' if i == 0 else 'next in chain'})"
            )
            # Case-insensitive search for chain songs
            _chain_search = st.text_input(f"🔍 Search song {i+1}", key=f"chain_search_{i}", placeholder="Type to filter…")
            if _chain_search:
                _chain_pat = _chain_search.lower()
                opts = [s for s in opts if _chain_pat in s.lower()]
                default_idx = 0
                if not opts:
                    st.caption(f"No match for '{_chain_search}'")
                    opts = all_songs
            chosen = st.selectbox(label, opts, index=min(default_idx, len(opts)-1), key=f"chain_song_{i}")
            selected_songs.append(chosen)

            # Show BPM badge if recommendation data available
            for r in recs:
                if r["name"] == chosen:
                    st.caption(
                        f"🎯 Recommended · {r['bpm']} BPM · {score_color(r['overall'])} {r['overall']:.0%} compatible with previous"
                    )
                    break

        # Arrow between songs
        if i < n_songs - 1:
            st.markdown(
                "<div style='text-align:center;color:#4a4a6a;font-size:1.2rem;margin:-4px 0'>↓ transition</div>",
                unsafe_allow_html=True,
            )

    # ── Chain preview card ────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📋 Mix order")
    flow_parts = []
    for i, name in enumerate(selected_songs):
        short = name[:22] + ("…" if len(name) > 22 else "")
        flow_parts.append(f"**{i+1}.** {short}")
    st.markdown("  →  ".join(flow_parts))

    # ── Transition settings ───────────────────────────────────────────────
    st.divider()
    st.markdown("### ⚙️ Transition settings")
    st.caption("Applied to every transition in the chain.")
    col1, col2 = st.columns(2)
    with col1:
        chain_bars   = st.select_slider("Bars per transition", options=[8, 16, 32], value=16, key="chain_bars")
    with col2:
        chain_preset = st.selectbox("Genre preset", ["auto", "house", "techno", "hiphop", "trap",
                                                      "pop", "rnb", "dnb", "ambient", "rock", "jazz"],
                                    key="chain_preset")

    # ── Beat Bridge ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🥁 Beat Bridge")
    st.caption("Drum layer fades in/out at every transition.")

    chain_bridge_mode = st.radio(
        "Bridge beat source",
        options=["none", "auto", "strudel"],
        format_func=lambda x: {
            "none":    "🚫 None",
            "auto":    "⚡ Auto-Generate",
            "strudel": "🎨 Strudel upload",
        }[x],
        horizontal=True,
        key="chain_bridge_mode",
    )

    chain_bridge_genre     = "auto"
    chain_bridge_intensity = 0.38
    chain_bridge_path      = None

    if chain_bridge_mode == "auto":
        col1, col2 = st.columns(2)
        with col1:
            chain_bridge_genre = st.selectbox(
                "Beat genre", ["auto", "techno", "house", "hiphop", "trap", "dnb", "ambient"],
                key="chain_bridge_genre",
            )
        with col2:
            chain_bridge_intensity = st.slider(
                "Intensity", 0.1, 0.8, 0.38, 0.05, key="chain_bridge_intensity",
            )

    elif chain_bridge_mode == "strudel":
        # Use first song's BPM for the starter pattern (if known)
        chain_src_bpm = 128.0
        if st.session_state.get("chain_rec_cache"):
            first_song = selected_songs[0] if selected_songs else None
            if first_song and first_song in st.session_state["chain_rec_cache"]:
                recs = st.session_state["chain_rec_cache"][first_song]
                if recs:
                    chain_src_bpm = recs[0].get("bpm", 128.0)

        with st.spinner("Generating starter pattern…"):
            chain_synth = api_get(
                "/beat/synthesize",
                params={"bpm": chain_src_bpm, "genre": chain_bridge_genre, "bars": 4},
            )
        chain_strudel_code = (
            chain_synth.get("strudel_code", "") if chain_synth else ""
        ) or (
            f'stack(\n'
            f'  s("bd sd bd sd").bank("RolandTR909"),\n'
            f'  s("hh*8").gain(0.6)\n'
            f').cpm({int(chain_src_bpm)} / 4)'
        )

        st.markdown("**🎹 Live Strudel Editor** — edit and play your bridge pattern:")
        st.caption(
            "▶ **Ctrl+Enter** to play · ■ **Ctrl+.** to stop · "
            "⏺ click the **record dot** inside the editor to export WAV"
        )

        # ── Embedded Strudel REPL ─────────────────────────────────────────
        _strudel_embed(chain_strudel_code, height=420)

        st.divider()
        st.markdown("**📤 Upload your recording to blend into every transition:**")
        st.caption("Record inside the editor above, download the WAV, then drop it here.")
        uploaded_chain = st.file_uploader(
            "Upload your Strudel recording (WAV/MP3)",
            type=["wav", "mp3", "ogg"],
            key="chain_strudel_upload",
        )
        if uploaded_chain:
            with st.spinner("Uploading beat…"):
                resp = requests.post(
                    f"{API}/beat/upload",
                    files={"file": (uploaded_chain.name, uploaded_chain.read(), uploaded_chain.type)},
                )
            if resp.ok:
                data = resp.json()
                chain_bridge_path = data.get("audio_path")
                st.success(f"✅ Beat uploaded: `{data['filename']}` ({data['size_kb']} KB)")
            else:
                st.error("Upload failed.")
        chain_bridge_intensity = st.slider("Intensity", 0.1, 0.8, 0.38, 0.05, key="chain_strudel_intensity")

    # ── Render button ─────────────────────────────────────────────────────
    st.divider()
    chain_api_mode = "file" if (chain_bridge_mode == "strudel" and chain_bridge_path) else chain_bridge_mode
    if chain_bridge_mode == "strudel" and not chain_bridge_path:
        chain_api_mode = "none"

    render_disabled = len(set(selected_songs)) < 2
    if render_disabled:
        st.warning("Select at least 2 different songs to render a chain.")

    if st.button(
        f"🎚️ Render {n_songs}-Song Chain Mix",
        type="primary",
        use_container_width=True,
        disabled=render_disabled,
    ):
        payload = {
            "songs":                selected_songs,
            "transition_bars":      chain_bars,
            "preset":               chain_preset,
            "bridge_beat_mode":     chain_api_mode,
            "bridge_beat_genre":    chain_bridge_genre,
            "bridge_beat_intensity": chain_bridge_intensity,
            "bridge_beat_path":     chain_bridge_path,
        }
        jid = submit_job(
            name=f"DJ Chain · {n_songs} songs",
            endpoint="/dj-chain",
            body=payload,
            icon="🎚️",
            page_key="chain",
        )
        if jid:
            st.toast(f"🎚️ Chain mix running in background — {n_songs} songs, {n_songs-1} transitions!", icon="🚀")

    # ── Live inline progress + result ────────────────────────────────────────
    chain_info = _inline_job_progress("chain")
    if chain_info and chain_info.get("status") == "done" and chain_info.get("result"):
        r = chain_info["result"]
        st.success(f"✅ Chain mix ready! {r['n_songs']} songs · {r['duration_sec']}s total")
        col1, col2, col3 = st.columns(3)
        col1.metric("Songs mixed",    r["n_songs"])
        col2.metric("Total duration", f"{r['duration_sec']}s")
        col3.metric("Reference BPM",  f"{r['bpm_reference']} BPM")

        st.markdown("#### Transition breakdown")
        for t in r.get("transitions", []):
            st.markdown(f"🔀 **{t['from'][:20]}** `{t['bpm_from']} BPM` → **{t['to'][:20]}** `{t['bpm_to']} BPM` · {t['transition_bars']} bars · {t['transition_sec']}s")

        out_path = r.get("output", "")
        if out_path:
            parts = out_path.split("/")
            try:
                outputs_idx = parts.index("outputs")
                audio_url   = f"{API}/outputs/{parts[outputs_idx+1]}/{parts[-1]}"
                st.markdown("**▶ Play full chain mix**")
                st.audio(audio_url, format="audio/wav")
                st.caption(f"Saved to: `{out_path}`")
            except Exception:
                st.caption(f"Saved to: `{out_path}`")
            with st.expander("📊 View spectrogram of chain mix", expanded=False):
                cc1, cc2, cc3 = st.columns(3)
                cspec_dur    = cc1.slider("Duration (s)", 30, 300, 120, key="chain_spec_dur")
                cspec_cmap   = cc2.selectbox("Color map", ["magma", "inferno", "plasma", "viridis", "hot"], key="chain_spec_cmap")
                cspec_chroma = cc3.checkbox("Show chromagram", value=False, key="chain_spec_chroma")
                chain_label  = " → ".join([t["from"] for t in r.get("transitions", [])] + ([r["transitions"][-1]["to"]] if r.get("transitions") else []))
                _show_spectrogram(out_path, title=chain_label[:80] or "Chain Mix", duration=cspec_dur, colormap=cspec_cmap, show_chroma=cspec_chroma)
    elif chain_info and chain_info.get("status") == "failed":
        st.error(f"Chain render failed: {chain_info.get('result', {})}")


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Instrument Lab — stem swap experiments
# ---------------------------------------------------------------------------

elif page == "🧪 Instrument Lab":
    import pathlib as _il_pathlib

    st.title("🧪 Instrument Lab")
    st.markdown(
        "**Swap instruments between songs** — take vocals from one track, drums from another, "
        "bass from a third, and hear every combination. Requires Demucs stem separation."
    )

    # ── Fetch songs with stems ────────────────────────────────────────────
    _lab_songs_resp = api_get("/instrument-lab/songs")
    _lab_songs = _lab_songs_resp.get("songs", []) if _lab_songs_resp else []

    if len(_lab_songs) < 2:
        st.warning(
            f"**{len(_lab_songs)} song(s) have stems** — you need at least 2. "
            "Go to **🚀 Initialize Library** to run Demucs stem separation first.",
            icon="🔬",
        )
        if st.button("🚀 Go to Initialize Library", key="lab_go_init"):
            _go_to_page("🚀 Initialize Library")
    else:
        st.success(f"**{len(_lab_songs)} songs** have stems and are ready for experiments.", icon="🧬")

        # ── Song selection ────────────────────────────────────────────────
        st.markdown("### Select Songs")
        _lab_selected = st.multiselect(
            "Pick 2–4 songs to experiment with",
            _lab_songs,
            max_selections=4,
            key="lab_song_select",
        )

        if len(_lab_selected) >= 2:
            # ── Experiment controls ───────────────────────────────────────
            st.markdown("### Experiment Settings")

            _lab_c1, _lab_c2, _lab_c3 = st.columns(3)

            with _lab_c1:
                _lab_mode = st.radio(
                    "Combination mode",
                    ["targeted", "all"],
                    index=0,
                    key="lab_mode",
                    help="**Targeted**: swap one stem at a time (manageable). "
                         "**All**: every possible permutation (can be many!).",
                )
            with _lab_c2:
                _lab_stems = st.multiselect(
                    "Stems to swap",
                    ["vocals", "drums", "bass", "other"],
                    default=["vocals", "drums", "bass", "other"],
                    key="lab_stems",
                )
            with _lab_c3:
                _lab_duration = st.slider(
                    "Preview duration (sec)",
                    min_value=10,
                    max_value=120,
                    value=30,
                    step=5,
                    key="lab_duration",
                    help="Trim each combo to this length (saves time).",
                )

            # ── Combo count estimate ──────────────────────────────────────
            n_songs = len(_lab_selected)
            n_stems = len(_lab_stems)
            if _lab_mode == "targeted":
                _est_combos = n_songs * (n_songs - 1) * n_stems
            else:
                _est_combos = n_songs ** n_stems
            st.info(
                f"This will generate approximately **{_est_combos} combinations** "
                f"({n_songs} songs × {n_stems} stems, {_lab_mode} mode).",
                icon="🔢",
            )

            # ── Visual combo preview ──────────────────────────────────────
            st.markdown("### Preview: Stem Sources")
            _short_names = {s: chr(65 + i) for i, s in enumerate(_lab_selected)}
            _legend = " · ".join(f"**{v}** = {k[:25]}" for k, v in _short_names.items())
            st.caption(f"Legend: {_legend}")

            _preview_cols = st.columns(len(_lab_stems))
            for ci, stem_name in enumerate(_lab_stems):
                with _preview_cols[ci]:
                    st.markdown(
                        f"<div style='text-align:center; padding:10px; "
                        f"background:#1a1a2e; border-radius:8px; border:1px solid #333;'>"
                        f"<div style='font-size:24px;'>{'🎤' if stem_name == 'vocals' else '🥁' if stem_name == 'drums' else '🎸' if stem_name == 'bass' else '🎹'}</div>"
                        f"<div style='font-size:13px; font-weight:600; color:#e0e0e0;'>{stem_name.title()}</div>"
                        f"<div style='font-size:11px; color:#888;'>{n_songs} sources</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            st.divider()

            # ── Launch experiment ──────────────────────────────────────────
            _lab_disabled = len(_lab_selected) < 2 or len(_lab_stems) == 0
            if st.button(
                f"🧪 Run Instrument Lab ({_est_combos} combos)",
                disabled=_lab_disabled,
                type="primary",
                use_container_width=True,
                key="lab_run",
            ):
                _lab_payload = {
                    "songs": _lab_selected,
                    "mode": _lab_mode,
                    "swap_stems": _lab_stems if _lab_stems != list(["vocals", "drums", "bass", "other"]) else None,
                    "target_duration": float(_lab_duration),
                    "include_pure": False,
                }
                _lab_resp = api_post("/instrument-lab", _lab_payload)

                if _lab_resp and "job_id" in _lab_resp:
                    st.session_state["lab_job_id"] = _lab_resp["job_id"]
                    st.toast(f"🧪 Instrument Lab started — {_est_combos} combos queuing…", icon="🚀")
                    st.rerun()
                else:
                    st.error("Failed to start Instrument Lab job.")

            # ── Show active / completed lab job ───────────────────────────
            if "lab_job_id" in st.session_state:
                _lab_jid = st.session_state["lab_job_id"]
                _lab_job = api_get(f"/jobs/{_lab_jid}")

                if _lab_job:
                    _lab_status = _lab_job.get("status", "unknown")
                    _lab_progress = _lab_job.get("progress", 0)
                    _lab_message = _lab_job.get("message", "")

                    if _lab_status == "running":
                        st.progress(_lab_progress, text=_lab_message)
                        import time as _lab_time
                        _lab_time.sleep(2)
                        st.rerun()

                    elif _lab_status == "done":
                        _lab_result = _lab_job.get("result", {})
                        _rendered = _lab_result.get("rendered", 0)
                        _failed = _lab_result.get("failed", 0)

                        st.success(f"✅ Instrument Lab complete: **{_rendered} combos** rendered, {_failed} failed.", icon="🧪")

                        # Show results grid
                        _combos_data = _lab_result.get("combos", [])
                        if _combos_data:
                            st.markdown("### Results")
                            _res_cols = st.columns(2)
                            for _ri, _combo in enumerate(_combos_data):
                                if not _combo.get("success"):
                                    continue
                                with _res_cols[_ri % 2]:
                                    _mapping = _combo.get("mapping", {})
                                    _label = _combo.get("label", "?")
                                    _out = _combo.get("output")

                                    # Card
                                    _parts_html = ""
                                    for _sn in ["vocals", "drums", "bass", "other"]:
                                        _src = _mapping.get(_sn, "?")
                                        _icon = {"vocals": "🎤", "drums": "🥁", "bass": "🎸", "other": "🎹"}.get(_sn, "🎵")
                                        _parts_html += (
                                            f"<div style='display:inline-block; margin:2px 4px; padding:3px 8px; "
                                            f"background:#2a2a3e; border-radius:6px; font-size:11px;'>"
                                            f"{_icon} {_src[:20]}</div>"
                                        )

                                    st.markdown(
                                        f"<div style='background:#1a1a2e; border:1px solid #333; "
                                        f"border-radius:10px; padding:12px; margin-bottom:10px;'>"
                                        f"<div style='font-size:12px; color:#888; margin-bottom:6px;'>"
                                        f"Combo #{_ri + 1} · {_combo.get('bpm', 0):.0f} BPM · "
                                        f"{_combo.get('duration_sec', 0):.0f}s · {_combo.get('lufs', -70):.1f} LUFS</div>"
                                        f"<div>{_parts_html}</div>"
                                        f"</div>",
                                        unsafe_allow_html=True,
                                    )

                                    if _out:
                                        _out_path = _il_pathlib.Path(_out)
                                        if _out_path.exists():
                                            st.audio(str(_out_path))

                    elif _lab_status == "failed":
                        st.error(f"Instrument Lab failed: {_lab_job.get('message', 'Unknown error')}")

        else:
            st.caption("Select at least 2 songs above to start experimenting.")


# ---------------------------------------------------------------------------
# AI Studio — Style Transfer + VampNet Inpainting + Model Status
# ---------------------------------------------------------------------------

elif page == "🤖 AI Studio":
    st.title("🤖 AI Studio")
    st.markdown(
        "**Generative AI tools powered by MusicGen + VampNet.** "
        "Style-transfer a song's harmonic fingerprint into a new style, "
        "or generate creative transition fills between two tracks."
    )

    # ── Helper: fetch library song names ─────────────────────────────────
    _ai_songs_all  = _fetch_all_song_names()
    _ai_songs_wav  = _fetch_all_song_names(with_wav=True)

    if not _ai_songs_all:
        st.warning(
            "No songs in library yet. Go to **⬇️ Download** first.",
            icon="📭",
        )
    else:
        _ai_tabs = st.tabs(["🎨 Style Transfer", "🧩 Inpainting", "⚡ Tokenize", "📊 Model Status"])

        # ────────────────────────────────────────────────────────────────
        # Tab 1 — Style Transfer
        # ────────────────────────────────────────────────────────────────
        with _ai_tabs[0]:
            st.subheader("🎨 Style Transfer via MusicGen Melody")
            st.markdown(
                "Pick a source song — MusicGen extracts its **CQT chromagram** as a melody "
                "reference, then generates a new segment in your described style. "
                "The harmonic DNA is preserved; everything else transforms."
            )
            st.info(
                "⏱️ **First run downloads MusicGen (~1.5 GB).** Subsequent runs are fast. "
                "Keep duration ≤ 20 s on 16 GB RAM. MusicGen offloads after each run.",
                icon="ℹ️",
            )

            _st_col1, _st_col2 = st.columns([1, 1])

            with _st_col1:
                _st_song = st.selectbox(
                    "Source song",
                    _ai_songs_wav,
                    key="st_song",
                    help="Song to extract melody from. Needs full.wav.",
                )
                _st_desc = st.text_area(
                    "Style description",
                    value="dark melodic techno, 128 BPM, heavy sub-bass, melancholic chords",
                    height=80,
                    key="st_desc",
                    help=(
                        "Describe the genre, mood, instrumentation, tempo. "
                        "Examples: 'chill lo-fi hip-hop with piano and vinyl crackle' · "
                        "'epic orchestral remix cinematic strings'"
                    ),
                )
                _st_stem = st.selectbox(
                    "Melody source stem",
                    ["full", "other", "vocals", "bass"],
                    key="st_stem",
                    help=(
                        "'full' = complete mix · 'other' = chords/melody without drums/bass/vocals"
                    ),
                )

            with _st_col2:
                _st_dur = st.slider(
                    "Duration (seconds)",
                    min_value=4.0, max_value=30.0, value=15.0, step=1.0,
                    key="st_dur",
                )
                _st_guide = st.slider(
                    "Guidance scale (higher = more literal)",
                    min_value=1.0, max_value=10.0, value=3.0, step=0.5,
                    key="st_guide",
                )
                _st_temp = st.slider(
                    "Temperature (higher = more creative)",
                    min_value=0.1, max_value=2.0, value=1.0, step=0.1,
                    key="st_temp",
                )
                _st_seed_on = st.checkbox("Fix random seed", value=False, key="st_seed_on")
                _st_seed = st.number_input("Seed", value=42, key="st_seed") if _st_seed_on else None
                _st_fmt = st.selectbox("Output format", ["wav", "flac"], key="st_fmt")

            st.divider()
            if st.button("🚀 Generate Style Transfer", key="st_run", type="primary",
                          disabled=not _st_song):
                if not _st_song:
                    st.error("Select a source song first.")
                else:
                    payload = {
                        "song_name": _st_song,
                        "description": _st_desc,
                        "duration_sec": _st_dur,
                        "source_stem": _st_stem,
                        "guidance_scale": _st_guide,
                        "temperature": _st_temp,
                        "seed": _st_seed,
                        "output_format": _st_fmt,
                    }
                    resp = api_post("/ai/style-transfer", payload)
                    if resp and "job_id" in resp:
                        st.session_state["ai_studio_job"] = resp["job_id"]
                        st.session_state["ai_studio_type"] = "style_transfer"
                        st.rerun()
                    else:
                        st.error(f"Failed to start job: {resp}")

            # ── Job progress / result ─────────────────────────────────
            if st.session_state.get("ai_studio_type") == "style_transfer":
                _jid = st.session_state.get("ai_studio_job")
                if _jid:
                    @st.fragment(run_every=2)
                    def _st_poll():
                        _j = api_get(f"/jobs/{_jid}")
                        if not _j:
                            return
                        _status = _j.get("status", "")
                        _prog   = _j.get("progress", 0.0)
                        _msg    = _j.get("message", "")
                        st.progress(_prog, text=f"[{_status.upper()}] {_msg}")

                        if _status == "done":
                            _res = _j.get("result", {})
                            st.success(f"✅ Generated in {_res.get('generation_time_sec', 0):.1f}s")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Duration", f"{_res.get('duration_sec', 0):.1f}s")
                                st.metric("LUFS", f"{_res.get('lufs', 0):.1f}")
                            with col_b:
                                st.metric("Source Key", _res.get("source_key", "—"))
                                st.metric("Camelot", _res.get("source_camelot", "—"))
                            _audio_url = _res.get("audio_url")
                            if _audio_url:
                                full_url = f"{API_PUBLIC}{_audio_url}"
                                st.audio(full_url)
                                st.markdown(f"[⬇️ Download]({full_url})")
                        elif _status == "failed":
                            st.error(f"❌ {_j.get('error', 'Generation failed')}")
                    _st_poll()

        # ────────────────────────────────────────────────────────────────
        # Tab 2 — Inpainting
        # ────────────────────────────────────────────────────────────────
        with _ai_tabs[1]:
            st.subheader("🧩 VampNet Transition Inpainting")
            st.markdown(
                "Pick two songs — VampNet generates a **creative transition fill** using "
                "masked DAC token prediction. The tail of Song A and head of Song B anchor "
                "the context; everything in between is generated. "
                "Falls back to cosine token interpolation if VampNet is not installed."
            )

            _ip_col1, _ip_col2 = st.columns([1, 1])
            with _ip_col1:
                _ip_song_a = st.selectbox("Song A (outgoing)", _ai_songs_wav, key="ip_song_a")
                _ip_song_b = st.selectbox("Song B (incoming)", _ai_songs_wav, key="ip_song_b",
                                           index=min(1, len(_ai_songs_wav) - 1))
                _ip_mask = st.selectbox(
                    "Mask type",
                    ["prefix_suffix", "periodic", "beat_driven", "compression"],
                    key="ip_mask",
                    help=(
                        "prefix_suffix = keep A tail + B head, generate middle\n"
                        "periodic = keep every N-th token (rhythmic anchor)\n"
                        "beat_driven = keep beat positions\n"
                        "compression = keep coarse codebooks only"
                    ),
                )
                _ip_stem = st.selectbox(
                    "Source stem", ["full", "other", "vocals"], key="ip_stem"
                )

            with _ip_col2:
                _ip_prefix = st.slider("Prefix bars (from A)", 1, 16, 4, key="ip_prefix")
                _ip_suffix = st.slider("Suffix bars (from B)", 1, 16, 4, key="ip_suffix")
                _ip_steps  = st.slider(
                    "Sampling steps (VampNet quality)", 8, 128, 36, step=4, key="ip_steps",
                    help="More steps = better quality, slower. 36 is a good default."
                )
                _ip_bpm_on = st.checkbox("Override BPM", value=False, key="ip_bpm_on")
                _ip_bpm = st.number_input("BPM", min_value=40.0, max_value=250.0,
                                           value=128.0, key="ip_bpm") if _ip_bpm_on else None
                _ip_fmt = st.selectbox("Output format", ["wav", "flac"], key="ip_fmt")

            st.divider()
            if st.button("🧩 Generate Transition Fill", key="ip_run", type="primary",
                          disabled=not (_ip_song_a and _ip_song_b)):
                if _ip_song_a == _ip_song_b:
                    st.warning("Select two different songs.")
                else:
                    payload = {
                        "song_a": _ip_song_a,
                        "song_b": _ip_song_b,
                        "mask_type": _ip_mask,
                        "prefix_bars": _ip_prefix,
                        "suffix_bars": _ip_suffix,
                        "sampling_steps": _ip_steps,
                        "source_stem": _ip_stem,
                        "bpm": _ip_bpm,
                        "output_format": _ip_fmt,
                    }
                    resp = api_post("/ai/inpaint", payload)
                    if resp and "job_id" in resp:
                        st.session_state["ai_studio_job"] = resp["job_id"]
                        st.session_state["ai_studio_type"] = "inpaint"
                        st.rerun()
                    else:
                        st.error(f"Failed to start job: {resp}")

            if st.session_state.get("ai_studio_type") == "inpaint":
                _jid = st.session_state.get("ai_studio_job")
                if _jid:
                    @st.fragment(run_every=2)
                    def _ip_poll():
                        _j = api_get(f"/jobs/{_jid}")
                        if not _j:
                            return
                        _status = _j.get("status", "")
                        _prog   = _j.get("progress", 0.0)
                        _msg    = _j.get("message", "")
                        st.progress(_prog, text=f"[{_status.upper()}] {_msg}")
                        if _status == "done":
                            _res = _j.get("result", {})
                            _method = _res.get("method", "unknown")
                            _icon = "✅" if _method == "vampnet" else "⚡"
                            st.success(
                                f"{_icon} Generated via **{_method}** "
                                f"in {_res.get('generation_time_sec', 0):.1f}s"
                            )
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Duration", f"{_res.get('duration_sec', 0):.1f}s")
                                st.metric("LUFS", f"{_res.get('lufs', 0):.1f}")
                            with col_b:
                                st.metric("Method", _method)
                                _meta = _res.get("metadata", {})
                                st.metric("BPM", f"{_meta.get('bpm', 0):.1f}")
                            _audio_url = _res.get("audio_url")
                            if _audio_url:
                                full_url = f"{API_PUBLIC}{_audio_url}"
                                st.audio(full_url)
                                st.markdown(f"[⬇️ Download]({full_url})")
                        elif _status == "failed":
                            st.error(f"❌ {_j.get('error', 'Inpainting failed')}")
                    _ip_poll()

        # ────────────────────────────────────────────────────────────────
        # Tab 3 — Tokenize
        # ────────────────────────────────────────────────────────────────
        with _ai_tabs[2]:
            st.subheader("⚡ Neural Codec Tokenization")
            st.markdown(
                "Encode Demucs stems into discrete codec tokens (EnCodec 24kHz or DAC 44.1kHz). "
                "Tokens are saved as `.npz` files under `library/{song}/tokens/`. "
                "Required for VampNet inpainting with the DAC codec."
            )
            _tok_song  = st.selectbox("Song to tokenize", _ai_songs_all, key="tok_song")
            _tok_codec = st.radio("Codec", ["encodec", "dac"], horizontal=True, key="tok_codec",
                                   help="EnCodec (24kHz, 75 Hz) · DAC (44.1kHz, 86 Hz, higher fidelity)")
            _tok_bw    = st.select_slider(
                "EnCodec bandwidth (kbps)",
                options=[1.5, 3.0, 6.0, 12.0, 24.0],
                value=6.0,
                key="tok_bw",
                disabled=(_tok_codec == "dac"),
            )
            if st.button("⚡ Tokenize Stems", key="tok_run", type="primary",
                          disabled=not _tok_song):
                payload = {"song_name": _tok_song, "codec": _tok_codec, "bandwidth": _tok_bw}
                resp = api_post("/ai/tokenize", payload)
                if resp and "job_id" in resp:
                    st.session_state["ai_studio_job"] = resp["job_id"]
                    st.session_state["ai_studio_type"] = "tokenize"
                    st.rerun()
                else:
                    st.error(f"Failed: {resp}")

            if st.session_state.get("ai_studio_type") == "tokenize":
                _jid = st.session_state.get("ai_studio_job")
                if _jid:
                    @st.fragment(run_every=2)
                    def _tok_poll():
                        _j = api_get(f"/jobs/{_jid}")
                        if not _j:
                            return
                        _s = _j.get("status", "")
                        st.progress(_j.get("progress", 0.0), text=f"[{_s.upper()}] {_j.get('message', '')}")
                        if _s == "done":
                            _r = _j.get("result", {})
                            st.success(f"✅ Tokenized {len(_r.get('stems', []))} stems")
                            col_a, col_b, col_c = st.columns(3)
                            col_a.metric("Total tokens", f"{_r.get('total_tokens', 0):,}")
                            col_b.metric("Token rate", f"{_r.get('token_rate_hz', 0):.0f} Hz")
                            col_c.metric("Codebooks", _r.get("num_codebooks", 0))
                        elif _s == "failed":
                            st.error(f"❌ {_j.get('error', 'Tokenization failed')}")
                    _tok_poll()

        # ────────────────────────────────────────────────────────────────
        # Tab 4 — Model Status
        # ────────────────────────────────────────────────────────────────
        with _ai_tabs[3]:
            st.subheader("📊 AI Model Registry")
            st.markdown("Live status of loaded generative models and MPS memory usage.")

            @st.fragment(run_every=5)
            def _model_status_panel():
                _ms = api_get("/ai/models")
                if not _ms:
                    st.warning("Cannot reach API.")
                    return
                _dev = _ms.get("device", "unknown").upper()
                _alloc = _ms.get("mps_allocated_gb", 0.0)
                _max   = _ms.get("max_vram_gb", 12.0)
                c1, c2, c3 = st.columns(3)
                c1.metric("Device", _dev)
                c2.metric("Allocated", f"{_alloc:.2f} GB")
                c3.metric("Budget", f"{_max:.0f} GB")
                if _max > 0:
                    st.progress(min(1.0, _alloc / _max), text=f"Memory: {_alloc:.2f} / {_max:.0f} GB")
                st.divider()
                for _m in _ms.get("models", []):
                    _loaded = _m.get("loaded", False)
                    _name   = _m.get("name", "?")
                    _vram   = _m.get("vram_estimate_gb", 0)
                    _desc   = _m.get("description", "")
                    _status_icon = "🟢" if _loaded else "⚪"
                    st.markdown(
                        f"{_status_icon} **{_name}** "
                        f"({'loaded' if _loaded else 'idle'}) — "
                        f"~{_vram:.1f} GB — {_desc}"
                    )
            _model_status_panel()


# ---------------------------------------------------------------------------
# My Mixes
# ---------------------------------------------------------------------------

elif page == "🎵 My Mixes":
    import pathlib as _mm_pathlib
    import time as _mm_time

    st.title("🎵 My Mixes")
    st.markdown("All rendered outputs — browse, preview, and download your mixes.")

    _outputs_root = _mm_pathlib.Path(__file__).parents[2] / "outputs"

    # ── Collect all mix sessions ──────────────────────────────────────────
    _all_mixes: list[dict] = []
    if _outputs_root.exists():
        for _session_dir in sorted(
            _outputs_root.iterdir(),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        ):
            if not _session_dir.is_dir():
                continue
            if _session_dir.name.startswith("beat_"):
                continue  # skip beat preview dirs

            # Find all audio files (wav + flac)
            _audio_files = sorted(
                list(_session_dir.glob("*.wav")) + list(_session_dir.glob("*.flac")),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if not _audio_files:
                continue

            for _af in _audio_files:
                _size_mb = round(_af.stat().st_size / 1_048_576, 1)
                _mtime = _af.stat().st_mtime
                _age_min = int((_mm_time.time() - _mtime) / 60)
                _age_str = (
                    f"{_age_min}m ago" if _age_min < 60
                    else f"{_age_min // 60}h {_age_min % 60}m ago" if _age_min < 1440
                    else f"{_age_min // 1440}d ago"
                )

                # Determine type from directory name convention
                _sname = _session_dir.name
                if _sname.startswith("dj_"):
                    _kind = "🎛️ DJ Mix"
                elif _sname.startswith("chain_"):
                    _kind = "🎚️ Chain Mix"
                else:
                    _kind = "🎵 Mix"

                # Extract song names from filename if possible (format: SongA_to_SongB.wav)
                _stem = _af.stem
                _display = _stem.replace("_to_", " → ").replace("_", " ")

                _all_mixes.append({
                    "kind": _kind,
                    "kind_raw": _sname[:2] if _sname.startswith(("dj", "ch")) else "ot",
                    "display": _display,
                    "filename": _af.name,
                    "session": _session_dir.name,
                    "size_mb": _size_mb,
                    "age": _age_str,
                    "mtime": _mtime,
                    "format": _af.suffix.upper().lstrip("."),
                    "path": _af,
                })

    # ── Filters + controls ────────────────────────────────────────────────
    if _all_mixes:
        _ctrl_col1, _ctrl_col2, _ctrl_col3 = st.columns([2, 2, 2])

        with _ctrl_col1:
            _kind_options = ["All", "🎛️ DJ Mix", "🎚️ Chain Mix", "🎵 Mix"]
            _kind_filter = st.selectbox("Filter by type", _kind_options, key="mymixes_kind")

        with _ctrl_col2:
            _sort_options = ["Newest first", "Oldest first", "Largest first", "Smallest first"]
            _sort_by = st.selectbox("Sort by", _sort_options, key="mymixes_sort")

        with _ctrl_col3:
            _fmt_options = ["All formats", "WAV", "FLAC"]
            _fmt_filter = st.selectbox("Format", _fmt_options, key="mymixes_fmt")

        # Apply filters
        _filtered = _all_mixes
        if _kind_filter != "All":
            _filtered = [m for m in _filtered if m["kind"] == _kind_filter]
        if _fmt_filter != "All formats":
            _filtered = [m for m in _filtered if m["format"] == _fmt_filter]

        # Apply sort
        if _sort_by == "Newest first":
            _filtered = sorted(_filtered, key=lambda m: m["mtime"], reverse=True)
        elif _sort_by == "Oldest first":
            _filtered = sorted(_filtered, key=lambda m: m["mtime"])
        elif _sort_by == "Largest first":
            _filtered = sorted(_filtered, key=lambda m: m["size_mb"], reverse=True)
        elif _sort_by == "Smallest first":
            _filtered = sorted(_filtered, key=lambda m: m["size_mb"])

        st.divider()
        st.caption(f"Showing **{len(_filtered)}** of {len(_all_mixes)} mixes")

        if not _filtered:
            _empty_state("🔍", "No mixes match the current filters", "Try adjusting the filter or sort options")
        else:
            # ── Mix cards: 2-column grid ──────────────────────────────────
            _card_cols = st.columns(2)
            for _mi, _mix in enumerate(_filtered):
                with _card_cols[_mi % 2]:
                    # Card header
                    st.markdown(
                        f"<div style='"
                        f"background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); "
                        f"border: 1px solid #333; border-radius: 10px; "
                        f"padding: 14px 16px; margin-bottom: 12px;'>"
                        f"<div style='font-size:12px; color:#888; margin-bottom:4px;'>"
                        f"{_mix['kind']}  ·  {_mix['format']}  ·  {_mix['size_mb']} MB  ·  {_mix['age']}"
                        f"</div>"
                        f"<div style='font-size:15px; font-weight:600; color:#e0e0e0; "
                        f"white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>"
                        f"{_mix['display']}"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # Inline audio player
                    st.audio(str(_mix["path"]))

                    # Download + stream links
                    _dl_col, _stream_col = st.columns(2)
                    with _dl_col:
                        with open(_mix["path"], "rb") as _fh:
                            st.download_button(
                                label="⬇️ Download",
                                data=_fh,
                                file_name=_mix["filename"],
                                mime=(
                                    "audio/flac" if _mix["format"] == "FLAC"
                                    else "audio/wav"
                                ),
                                key=f"dl_{_mix['session']}_{_mix['filename']}",
                                use_container_width=True,
                            )
                    with _stream_col:
                        _audio_url = (
                            f"{API_PUBLIC}/outputs/"
                            f"{_mix['session']}/"
                            f"{requests.utils.quote(_mix['filename'])}"
                        )
                        st.link_button(
                            "▶ Stream",
                            _audio_url,
                            use_container_width=True,
                        )

    else:
        # Empty state
        st.markdown(
            """
            <div style='text-align:center; padding:60px 20px;'>
                <div style='font-size:64px; margin-bottom:16px;'>🎛️</div>
                <div style='font-size:20px; font-weight:600; color:#e0e0e0; margin-bottom:8px;'>
                    No mixes yet
                </div>
                <div style='color:#888;'>
                    Head to <strong>DJ Remix</strong> or <strong>DJ Chain</strong> to render your first mix.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🎛️ Go to DJ Remix", key="mymixes_goto_remix"):
            _go_to_page("🎛️ DJ Remix")


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

elif page == "🔍 Analyze":
    st.title("🔍 Song Analysis")
    st.markdown("Detect genre, BPM, and song structure (intro / break / chorus / outro).")

    song_names = _fetch_all_song_names(with_wav=True)
    _default_analyze = st.session_state.pop("pending_analyze_song", None)
    song = _song_picker("Select a song to analyze", song_names, key="analyze_song", default=_default_analyze)

    if st.button("Analyze", type="primary", use_container_width=True):
        jid = submit_job(name=f"Analyze · {song[:30]}", endpoint="/analyze",
                         body={"song": song}, icon="🔍", page_key="analyze")
        if jid:
            st.toast(f"🔍 Analysis queued — check sidebar for progress!", icon="🚀")

    analyze_info = _inline_job_progress("analyze")
    if analyze_info and analyze_info.get("status") == "done" and analyze_info.get("result"):
        r = analyze_info["result"]
        st.success("✅ Analysis complete")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Genre", r["genre"].title())
        col2.metric("Confidence", f"{r['confidence']:.0%}")
        col3.metric("BPM", r["bpm"])
        col4.metric("Duration", f"{r['duration']}s")

        st.divider()
        st.subheader("Song Structure")
        sections = r.get("sections", [])
        if sections:
            import pandas as pd
            df = pd.DataFrame(sections)
            df["duration"] = (df["end_time"] - df["start_time"]).round(1)
            df.columns = ["Type", "Start Bar", "End Bar", "Start (s)", "End (s)", "Duration (s)"]
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.subheader("Timeline")
            total = r["duration"]
            for s in sections:
                pct = (s["end_time"] - s["start_time"]) / total
                colors = {
                    "intro": "🟦", "verse": "🟩", "chorus": "🟨",
                    "break": "🟪", "build": "🟧", "drop": "🟥", "outro": "⬜"
                }
                icon = colors.get(s["type"], "⬜")
                bar_str = icon * max(1, int(pct * 30))
                st.markdown(f"`{s['type']:<7}` {bar_str}  *{s['start_time']:.0f}s → {s['end_time']:.0f}s*")

    elif analyze_info and analyze_info.get("status") == "failed":
        st.error(f"Analysis failed: {analyze_info.get('result', {})}")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

elif page == "⬇️ Download":
    st.title("⬇️ Download Songs")
    st.markdown("Add songs to your library via YouTube Music URL or search query.")
    st.warning("⚠️ Downloaded audio is subject to YouTube ToS. Personal use only.", icon="⚠️")

    st.info(
        "🤖 **Auto-Demucs is ON** — every download automatically runs Demucs stem "
        "separation (vocals / drums / bass / other) and indexes the song into the "
        "RAG similarity engine.  This enables stem-aware DJ mixing and Smart Search.",
        icon="🎛️",
    )

    dl_tab1, dl_tab2 = st.tabs(["🎵 Single Track", "📋 Playlist"])

    with dl_tab1:
        query = st.text_input("YouTube Music URL or search query",
                               placeholder="e.g. https://music.youtube.com/watch?v=... or 'Eric Prydz Opus'",
                               key="dl_single_query")
        name_override = st.text_input("Custom name (optional)",
                                       placeholder="Leave blank to use auto-detected title",
                                       key="dl_single_name")

        if st.button("Download + Separate + Index", type="primary",
                     use_container_width=True, disabled=not query, key="dl_single_btn"):
            jid = submit_job(
                name=f"Download · {(name_override or query)[:28]}",
                endpoint="/download",
                body={"query": query, "name": name_override or None, "separate": True},
                icon="⬇️",
                page_key="download",
            )
            if jid:
                st.toast("⬇️ Download running in background — you can use other pages now!", icon="🚀")

        dl_info = _inline_job_progress("download")
        if dl_info and dl_info.get("status") == "done" and dl_info.get("result"):
            r = dl_info["result"]
            st.success(f"✅ Downloaded & indexed: **{r['name']}**")
            if r.get("license_warning"):
                st.warning(r["license_warning"], icon="⚠️")
            if r.get("stems"):
                st.caption(f"Stems: {', '.join(r['stems'].keys())}")
            if r.get("indexed"):
                st.caption("📊 Added to RAG similarity index")
        elif dl_info and dl_info.get("status") == "failed":
            st.error(f"Download failed: {dl_info.get('result', {})}")

    with dl_tab2:
        st.markdown("Download every track from a YouTube / YouTube Music **playlist URL** at once.")
        st.caption("Each track is automatically stem-separated and indexed — same as single downloads.")

        pl_url = st.text_input(
            "Playlist URL",
            placeholder="https://www.youtube.com/playlist?list=... or music.youtube.com/playlist?...",
            key="dl_playlist_url",
        )
        pl_col1, pl_col2 = st.columns(2)
        with pl_col1:
            pl_limit = st.number_input(
                "Max tracks (0 = all)",
                min_value=0, max_value=200, value=0, step=1,
                help="Set to 0 to download the entire playlist.",
                key="dl_playlist_limit",
            )
        with pl_col2:
            pl_separate = st.checkbox(
                "Run Demucs on each track",
                value=True,
                help="Recommended — enables stem-aware mixing.",
                key="dl_playlist_separate",
            )

        if st.button("📋 Download Playlist", type="primary",
                     use_container_width=True, disabled=not pl_url, key="dl_playlist_btn"):
            jid = submit_job(
                name=f"Playlist · {pl_url[:28]}",
                endpoint="/download-playlist",
                body={
                    "url": pl_url,
                    "separate": pl_separate,
                    "limit": int(pl_limit) if pl_limit > 0 else None,
                },
                icon="📋",
                page_key="download_playlist",
            )
            if jid:
                st.toast("📋 Playlist download started — tracks queue up in the background!", icon="🚀")

        pl_info = _inline_job_progress("download_playlist")
        if pl_info and pl_info.get("status") == "done" and pl_info.get("result"):
            r = pl_info["result"]
            st.success(f"✅ Playlist done — {r.get('downloaded', '?')}/{r.get('total', '?')} tracks saved")
            st.metric("Playlist", r.get("playlist_title", "—"))
            if r.get("errors"):
                st.warning(f"⚠️ {len(r['errors'])} tracks failed — see details:")
                for err in r["errors"][:5]:
                    st.caption(f"  ✗ {err.get('name', '?')} — {err.get('error', '?')[:80]}")
        elif pl_info and pl_info.get("status") == "failed":
            st.error(f"Playlist download failed: {pl_info.get('result', {})}")


# ---------------------------------------------------------------------------
# Smart Search  (RAG vector index)
# ---------------------------------------------------------------------------

elif page == "🔮 Smart Search":
    st.title("🔮 Smart Search")
    st.markdown(
        "Find songs that are **deeply compatible** with any library track — "
        "not just by BPM, but across all dimensions simultaneously: "
        "**key, energy, groove, vocal density, and timbre**."
    )
    st.caption(
        "Powered by the RAG vector index: each song is embedded into a "
        "35-dimensional feature space.  Similarity = weighted cosine distance.  "
        "Sub-millisecond retrieval from pre-computed vectors in RAM."
    )

    song_names = _fetch_all_song_names(with_wav=True)
    _default_similar = st.session_state.pop("pending_similar", None)

    col1, col2 = st.columns([3, 1])
    with col1:
        query_song = _song_picker("Source song", song_names, key="smart_search_song", default=_default_similar)
    with col2:
        k_results = st.slider("Results", min_value=3, max_value=20, value=8, key="smart_k")

    if st.button("🔍 Find Similar Songs", type="primary",
                 use_container_width=True, disabled=not query_song):
        with st.spinner("Searching RAG index…"):
            data = api_get(
                f"/library/similar/{requests.utils.quote(query_song)}",
                params={"k": k_results},
            )

        if data and data.get("similar"):
            results = data["similar"]
            st.markdown(f"### Top {len(results)} matches for **{query_song}**")

            for i, r in enumerate(results, 1):
                score = r.get("score", 0)
                icon  = score_color(score)
                bkd   = r.get("breakdown") or {}
                with st.expander(
                    f"{icon} {i}. {r['name']}  —  **{score:.0%}** overall similarity",
                    expanded=i <= 3,
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("BPM",     f"{r.get('bpm', '?')}")
                    c2.metric("Key",     f"{r.get('key', '?')} {r.get('mode', '')}")
                    c3.metric("Camelot", r.get("camelot") or "?")
                    c4.metric("Genre",   r.get("genre") or "?")

                    if bkd:
                        st.markdown("**Similarity breakdown:**")
                        bd_cols = st.columns(5)
                        labels = ["BPM", "Key", "Energy", "Rhythm", "Timbre"]
                        keys   = ["bpm_sim", "key_sim", "energy_sim",
                                  "rhythm_sim", "timbre_sim"]
                        for col, lbl, key in zip(bd_cols, labels, keys):
                            val = bkd.get(key, 0)
                            col.metric(lbl, f"{val:.0%}")

                    d_val = r.get("danceability")
                    v_val = r.get("vocal_density")
                    if d_val is not None:
                        st.progress(d_val, text=f"Danceability: {d_val:.0%}")
                    if v_val is not None:
                        st.progress(v_val, text=f"Vocal density: {v_val:.0%}")
        elif data and data.get("similar") == []:
            st.info(
                "No results yet — the song may not be indexed.  "
                "Run **Analyze** on it first, or trigger an index rebuild below."
            )
        else:
            st.warning("Search failed or index is empty.  Trigger a rebuild below.")

    st.divider()
    st.markdown("### 🔧 Index Management")

    idx_stats = api_get("/index/stats")
    if idx_stats:
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Indexed songs",   idx_stats.get("indexed_songs", 0))
        sc2.metric("Vector dims",     idx_stats.get("vector_dims", 35))
        sc3.metric("Last updated",
                   (idx_stats.get("last_updated") or "never")[:19].replace("T", " "))

    if st.button("🔄 Rebuild Full Index", use_container_width=True):
        jid = submit_job(name="Rebuild RAG Index", endpoint="/index/rebuild",
                         body={}, icon="🔮", page_key="index_rebuild")
        if jid:
            st.toast("🔮 Index rebuild queued — see sidebar for progress", icon="🚀")


# ---------------------------------------------------------------------------
# Compress Library  (FLAC stem compression)
# ---------------------------------------------------------------------------

elif page == "🗜️ Compress Library":
    st.title("🗜️ Compress Library to FLAC")
    st.markdown(
        "Convert all stem WAVs to **lossless FLAC** — same quality, "
        "~50% smaller files.  Retrieval speed stays effectively identical "
        "(FLAC decodes in microseconds on modern hardware)."
    )

    st.markdown("### Why FLAC?")
    col1, col2, col3 = st.columns(3)
    col1.metric("Format",   "FLAC",    delta="lossless")
    col2.metric("Size",     "~50%",    delta="-50% vs WAV")
    col3.metric("Quality",  "100%",    delta="identical")

    st.markdown(
        "FLAC is a lossless codec — it's **mathematically identical** to WAV.  "
        "Demucs stems compressed to FLAC can be decoded back to bit-perfect WAV "
        "at any time.  The DJ engine loads FLAC transparently (prefers FLAC over WAV "
        "when both exist)."
    )
    st.divider()

    lib = api_get("/library?per_page=5000")
    song_names = [s["name"] for s in lib["songs"] if lib and s.get("stems")] if lib else []
    n_with_stems = len(song_names)
    n_total      = len(lib["songs"]) if lib else 0

    st.info(
        f"**{n_with_stems}** songs have stems  ·  "
        f"**{n_total}** total in library",
        icon="📊",
    )

    delete_wav = st.checkbox(
        "Delete WAV stems after FLAC encode (recommended — saves disk space)",
        value=True,
    )
    skip_existing = st.checkbox(
        "Skip songs that already have FLAC stems",
        value=True,
    )

    if st.button("🗜️ Compress All Stems to FLAC", type="primary",
                 use_container_width=True, disabled=(n_with_stems == 0)):
        jid = submit_job(
            name=f"FLAC compress · {n_with_stems} songs",
            endpoint=f"/stems/compress-batch?delete_wav={str(delete_wav).lower()}&skip_existing={str(skip_existing).lower()}",
            body={},
            icon="🗜️",
            page_key="compress_batch",
        )
        if jid:
            st.toast(f"🗜️ Compression queued for {n_with_stems} songs — running in background!", icon="🚀")

    compress_info = _inline_job_progress("compress_batch")
    if compress_info and compress_info.get("status") == "done" and compress_info.get("result"):
        r = compress_info["result"]
        st.success(f"✅ Done — **{r.get('converted','?')}** compressed · **{r.get('skipped','?')}** skipped · **{r.get('failed','?')}** failed")

    st.divider()
    st.markdown("### 🔬 Single-song compress")
    all_names = _fetch_all_song_names(with_stems=True)
    single_song = _song_picker("Song", all_names, key="compress_single")
    if st.button("Compress stems for this song", disabled=not single_song):
        jid = submit_job(
            name=f"Compress · {single_song[:30]}",
            endpoint=f"/stems/compress?song={requests.utils.quote(single_song)}&delete_wav={str(delete_wav).lower()}",
            body={},
            icon="🗜️",
            page_key="compress_single",
        )
        if jid:
            st.toast(f"🗜️ Compressing {single_song} in background", icon="🚀")


elif page == "🚀 Initialize Library":
    # ── Hero header ──────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .init-hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(135deg, #c084fc 0%, #818cf8 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 4px;
    }
    .init-hero-sub {
        font-size: 14px;
        color: rgba(255,255,255,0.5);
        margin-bottom: 24px;
        font-family: 'Inter', sans-serif;
    }
    </style>
    <div class="init-hero-title">🚀 Initialize Library</div>
    <div class="init-hero-sub">
        One-click pipeline · Stem separation → FLAC compression → RAG index rebuild ·
        <span style="color: rgba(168,85,247,0.8);">Live stats · Auto-updates every 3s</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Auto-refreshing stats + pipeline stepper ──────────────────────────────
    _init_library_live_stats()

    st.divider()

    # ── Options ───────────────────────────────────────────────────────────────
    with st.expander("⚙️ Pipeline options", expanded=False):
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            demucs_model = st.selectbox(
                "Demucs model",
                ["htdemucs", "htdemucs_ft", "mdx_extra"],
                index=0,
                help="htdemucs is fastest; htdemucs_ft is higher quality but slower.",
            )
            enhance_audio = st.checkbox("Enhance audio before stem split", value=True)
        with col_o2:
            delete_wav = st.checkbox(
                "Delete WAV stems after FLAC encode (saves ~50% disk)",
                value=True,
            )
            run_compress = st.checkbox("Run FLAC compression after splitting", value=True)
            run_index    = st.checkbox("Rebuild RAG index after compression", value=True)
    # Keep values accessible even when expander is collapsed
    if "demucs_model" not in st.session_state:
        st.session_state["demucs_model"] = "htdemucs"
    if "enhance_audio" not in st.session_state:
        st.session_state["enhance_audio"] = True
    if "delete_wav" not in st.session_state:
        st.session_state["delete_wav"] = True
    if "run_compress" not in st.session_state:
        st.session_state["run_compress"] = True
    if "run_index" not in st.session_state:
        st.session_state["run_index"] = True

    # ── Launch button ─────────────────────────────────────────────────────────
    lib_snap     = api_get("/library?per_page=5000") or {}
    n_total_snap = len(lib_snap.get("songs", []))

    if n_total_snap == 0:
        st.warning("No songs in library yet — head to ⬇️ Download to add some.")
        st.stop()

    st.markdown("### Run pipeline")
    if st.button(
        f"▶ Initialize {n_total_snap} songs",
        type="primary",
        use_container_width=True,
    ):
        jid = submit_job(
            name=f"Initialize Library · {n_total_snap} songs",
            endpoint="/library/initialize",
            body={
                "enhance":      enhance_audio,
                "model":        demucs_model,
                "delete_wav":   delete_wav,
                "run_compress": run_compress,
                "run_index":    run_index,
            },
            icon="🚀",
            page_key="init_library",
        )
        if jid:
            st.toast(
                "🚀 Library initialization running in background!  "
                "Stats above update live every 3s — you can keep using the app.",
                icon="🚀",
            )

    # ── Live inline progress + result ────────────────────────────────────────
    init_info = _inline_job_progress("init_library")
    if init_info and init_info.get("status") == "done" and init_info.get("result"):
        r = init_info["result"]
        st.balloons()
        st.success(
            "✅ **All done!** Your library is fully stem-split, compressed, and indexed.  \n"
            "DJ mixing now uses intelligent stem blending, Smart Search works instantly."
        )
        st_components.html(f"""
        <style>
        .rm-fin-grid {{
            display: grid; grid-template-columns: repeat(4,1fr); gap: 12px;
            font-family: 'Inter', sans-serif;
        }}
        .rm-fin-card {{
            background: rgba(34,197,94,0.07);
            border: 1px solid rgba(34,197,94,0.25);
            border-radius: 12px;
            padding: 16px 18px;
            text-align: center;
        }}
        .rm-fin-num {{
            font-size: 30px; font-weight: 800; color: #4ade80;
            animation: countUp 0.5s ease both;
        }}
        .rm-fin-label {{
            font-size: 11px; font-weight: 600; letter-spacing: 0.07em;
            text-transform: uppercase; color: rgba(255,255,255,0.45); margin-top: 6px;
        }}
        @keyframes countUp {{
            from {{ opacity:0; transform:translateY(8px); }}
            to   {{ opacity:1; transform:translateY(0); }}
        }}
        </style>
        <div class="rm-fin-grid">
          <div class="rm-fin-card"><div class="rm-fin-num">{r.get("split_done","?")}</div><div class="rm-fin-label">Stems split</div></div>
          <div class="rm-fin-card"><div class="rm-fin-num">{r.get("compress_converted","?")}</div><div class="rm-fin-label">Compressed</div></div>
          <div class="rm-fin-card"><div class="rm-fin-num">{r.get("total_indexed","?")}</div><div class="rm-fin-label">Songs indexed</div></div>
          <div class="rm-fin-card" style="border-color:rgba(239,68,68,0.25);background:rgba(239,68,68,0.06);">
            <div class="rm-fin-num" style="color:#f87171;">{r.get("split_failed",0)+r.get("compress_failed",0)}</div>
            <div class="rm-fin-label">Failed</div>
          </div>
        </div>
        """, height=110)
    elif init_info and init_info.get("status") == "failed":
        st.error(f"Initialization failed: {init_info.get('result', {})}")


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

elif page == "🎨 Visualizer":
    import pathlib

    st.title("🎨 Audio Visualizer")
    st.markdown(
        "Visualise any track or rendered mix as a **waveform**, "
        "**mel spectrogram**, and **chromagram**."
    )

    # ── Source picker ─────────────────────────────────────────────────────────
    viz_source = st.radio(
        "What do you want to visualise?",
        ["Library song", "Compare two songs", "Upload a file"],
        horizontal=True,
        key="viz_source",
    )

    # Helper: path to a library song's full.wav
    PROJECT_ROOT = pathlib.Path(__file__).parents[2]

    def _library_wav(name: str) -> str:
        return str(PROJECT_ROOT / "library" / name / "full.wav")

    # Fetch library song list — dynamic, all names
    lib_songs = _fetch_all_song_names(with_wav=True)

    # ── Display controls (shared) ─────────────────────────────────────────────
    st.divider()
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 1])
    viz_dur    = ctrl1.slider("Duration to analyse (s)", 15, 300, 60, key="viz_dur")
    viz_cmap   = ctrl2.selectbox(
        "Color map",
        ["magma", "inferno", "plasma", "viridis", "cividis", "hot"],
        key="viz_cmap",
    )
    viz_chroma = ctrl3.checkbox("Show chromagram", value=True, key="viz_chroma")
    viz_go     = ctrl4.button("▶ Generate", type="primary", use_container_width=True, key="viz_go")

    st.divider()

    # ── Single library song ───────────────────────────────────────────────────
    if viz_source == "Library song":
        if not lib_songs:
            _empty_state("📥", "No songs in library yet", "Head to Download to add some tracks")
        else:
            chosen = _song_picker("Pick a song", lib_songs, key="viz_single_song")
            if viz_go and chosen:
                wav = _library_wav(chosen)
                if pathlib.Path(wav).exists():
                    _show_spectrogram(
                        wav,
                        title=chosen,
                        duration=viz_dur,
                        colormap=viz_cmap,
                        show_chroma=viz_chroma,
                    )
                else:
                    st.error(f"WAV not found at `{wav}` — try re-downloading the song.")

    # ── Compare two library songs side by side ────────────────────────────────
    elif viz_source == "Compare two songs":
        if len(lib_songs) < 2:
            st.info("Need at least 2 songs in library to compare.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                song_va = _song_picker("Song A", lib_songs, key="viz_song_a")
            with col_b:
                song_vb = _song_picker("Song B", [s for s in lib_songs if s != song_va] or lib_songs, key="viz_song_b")

            if viz_go:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"#### 🅐 {song_va}")
                    wav_a = _library_wav(song_va)
                    if pathlib.Path(wav_a).exists():
                        _show_spectrogram(
                            wav_a,
                            title=song_va,
                            duration=viz_dur,
                            colormap=viz_cmap,
                            show_chroma=viz_chroma,
                        )
                    else:
                        st.error("WAV not found — try re-downloading.")
                with col_b:
                    st.markdown(f"#### 🅑 {song_vb}")
                    wav_b = _library_wav(song_vb)
                    if pathlib.Path(wav_b).exists():
                        _show_spectrogram(
                            wav_b,
                            title=song_vb,
                            duration=viz_dur,
                            colormap=viz_cmap,
                            show_chroma=viz_chroma,
                        )
                    else:
                        st.error("WAV not found — try re-downloading.")

    # ── Upload any WAV/MP3 ────────────────────────────────────────────────────
    elif viz_source == "Upload a file":
        uploaded_viz = st.file_uploader(
            "Drop any WAV or MP3", type=["wav", "mp3", "ogg"], key="viz_upload"
        )
        if uploaded_viz and viz_go:
            # Write to a temp file so librosa can load it
            import tempfile, os
            suffix = pathlib.Path(uploaded_viz.name).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_viz.read())
                tmp_path = tmp.name
            try:
                _show_spectrogram(
                    tmp_path,
                    title=uploaded_viz.name,
                    duration=viz_dur,
                    colormap=viz_cmap,
                    show_chroma=viz_chroma,
                )
            finally:
                os.unlink(tmp_path)


# =============================================================================
# 🟢 Spotify — discovery, import, and playlist sync
# =============================================================================

elif page == "🟢 Spotify":
    st.title("🟢 Spotify")
    st.markdown(
        "Search Spotify's catalogue, browse your playlists, and import tracks "
        "directly into your library — metadata pre-loaded (BPM, key, energy)."
    )

    # ── Check config + connection status ──────────────────────────────────────
    _sp_status = _api_get_silent("/spotify/status") or {}
    _sp_configured = _sp_status.get("configured", False)
    _sp_connected  = _sp_status.get("connected", False)

    # Handle OAuth return params
    _qparams = st.query_params
    if _qparams.get("spotify_connected"):
        st.success("✅ Spotify connected! Your playlists and top tracks are now accessible.")
        st.query_params.clear()
    if _qparams.get("spotify_error"):
        st.error(f"❌ Spotify connection failed: {_qparams.get('spotify_error')}")
        st.query_params.clear()

    # ── Setup banner ──────────────────────────────────────────────────────────
    if not _sp_configured:
        st.warning("⚙️  Spotify is not configured yet.")
        with st.expander("📋  Quick setup (2 minutes)", expanded=True):
            st.markdown("""
**Step 1 — Create a free Spotify developer app:**
1. Go to [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard)
2. Click **Create app**, give it any name
3. Under **Redirect URIs**, add: `http://localhost:8000/spotify/callback`
4. Save — then copy your **Client ID** and **Client Secret**

**Step 2 — Add credentials to your project:**

Create `config.local.yaml` in the project root (it's gitignored):
```yaml
spotify:
  client_id: "YOUR_CLIENT_ID"
  client_secret: "YOUR_CLIENT_SECRET"
  redirect_uri: "http://localhost:8000/spotify/callback"
```

**Step 3 — Restart the API** (Ctrl+C → `./start.sh`) and come back here.

> **Free tier note:** Spotify restricted their audio features endpoint (BPM, key, energy) 
> for new apps created after November 2024. If your app was created after that date, 
> tracks will still import correctly — they'll just be analysed locally by librosa instead.
""")
        st.stop()

    # ── Connection status card ─────────────────────────────────────────────────
    _sc1, _sc2 = st.columns([3, 1])
    with _sc1:
        if _sp_connected:
            st.success("🟢  Connected to your Spotify account — playlists and top tracks available.")
        else:
            st.info("🔍  Search mode active (no user login). Connect to access your playlists.")
    with _sc2:
        if _sp_connected:
            if st.button("Disconnect", key="sp_disconnect"):
                r = requests.delete(f"{API}/spotify/disconnect")
                if r.ok:
                    st.success("Disconnected.")
                    st.rerun()
        else:
            st.link_button("🔑  Connect Spotify", f"{API}/spotify/auth", use_container_width=True)

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    _sp_tabs = st.tabs(["🔍 Search", "📋 My Playlists", "⭐ My Top Tracks"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Search
    # ════════════════════════════════════════════════════════════════════════
    with _sp_tabs[0]:
        st.markdown("### Search Spotify")
        _sp_q = st.text_input(
            "Track, artist, or album",
            placeholder="e.g.  Anyma Voices In My Head",
            key="sp_search_q",
        )
        _sp_limit = st.slider("Results", 5, 30, 10, key="sp_limit")

        if _sp_q:
            with st.spinner("Searching Spotify…"):
                _sp_resp = _api_get_silent(f"/spotify/search?q={requests.utils.quote(_sp_q)}&limit={_sp_limit}")

            if not _sp_resp or not _sp_resp.get("tracks"):
                st.warning("No results found.")
            else:
                _sp_tracks = _sp_resp["tracks"]
                st.caption(f"{len(_sp_tracks)} results for **{_sp_q}**")

                for _t in _sp_tracks:
                    with st.container():
                        _tc1, _tc2, _tc3 = st.columns([4, 3, 1])
                        with _tc1:
                            st.markdown(f"**{_t['name']}**  \n{_t['artist_str']}")
                            st.caption(f"💿 {_t['album']}  •  ⏱ {_t['duration_str']}  •  🔥 {_t['popularity']}/100")
                        with _tc2:
                            _chips = []
                            if _t.get("bpm"):
                                _chips.append(f"🎵 {_t['bpm']:.0f} BPM")
                            if _t.get("camelot"):
                                _chips.append(f"🎹 {_t['camelot']} ({_t.get('key_name','')})")
                            if _t.get("energy") is not None:
                                _chips.append(f"⚡ Energy {_t['energy']:.0%}")
                            if _t.get("danceability") is not None:
                                _chips.append(f"💃 Dance {_t['danceability']:.0%}")
                            if _chips:
                                st.markdown("  ·  ".join(_chips))
                            else:
                                st.caption("*Audio features not available for this app — will be analysed locally after import*")
                        with _tc3:
                            if st.button("⬇️ Import", key=f"sp_import_{_t['id']}", use_container_width=True):
                                _payload = {
                                    "query":    _t["download_query"],
                                    "separate": False,
                                    "camelot":  _t.get("camelot", ""),
                                    "bpm":      _t.get("bpm"),
                                    "energy":   _t.get("energy"),
                                }
                                _imp_r = requests.post(f"{API}/spotify/import", json=_payload)
                                if _imp_r.ok:
                                    st.success(f"Queued: {_t['download_query']}")
                                else:
                                    st.error(f"Import failed: {_imp_r.text}")
                        st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — My Playlists
    # ════════════════════════════════════════════════════════════════════════
    with _sp_tabs[1]:
        if not _sp_connected:
            st.info("🔑  Connect your Spotify account (button above) to browse your playlists.")
        else:
            st.markdown("### Your Playlists")
            _pl_resp = _api_get_silent("/spotify/playlists")
            if not _pl_resp or not _pl_resp.get("playlists"):
                st.info("No playlists found.")
            else:
                _playlists = _pl_resp["playlists"]
                st.caption(f"{len(_playlists)} playlists")

                for _pl in _playlists:
                    with st.expander(f"📋  {_pl['name']}  ({_pl['track_count']} tracks)"):
                        st.caption(f"Owner: {_pl['owner']}  •  {'Public' if _pl['public'] else 'Private'}")
                        if _pl.get("description"):
                            st.caption(_pl["description"])

                        _pl_col1, _pl_col2 = st.columns([1, 1])
                        with _pl_col1:
                            if st.button(f"👁️ Preview tracks", key=f"sp_preview_{_pl['id']}"):
                                st.session_state[f"sp_pl_open_{_pl['id']}"] = True

                        with _pl_col2:
                            if st.button(f"⬇️ Import all {_pl['track_count']} tracks", key=f"sp_import_pl_{_pl['id']}"):
                                with st.spinner(f"Queueing {_pl['name']}…"):
                                    _imp_pl = requests.post(
                                        f"{API}/spotify/import-playlist",
                                        json={"playlist_id": _pl["id"], "separate": False},
                                    )
                                if _imp_pl.ok:
                                    _queued = _imp_pl.json().get("queued", 0)
                                    st.success(f"✅  {_queued} tracks queued for download!")
                                else:
                                    st.error(f"Failed: {_imp_pl.text}")

                        # Show track preview if requested
                        if st.session_state.get(f"sp_pl_open_{_pl['id']}"):
                            with st.spinner("Loading tracks…"):
                                _pl_tracks_resp = _api_get_silent(f"/spotify/playlists/{_pl['id']}/tracks?limit=20")
                            if _pl_tracks_resp and _pl_tracks_resp.get("tracks"):
                                for _plt in _pl_tracks_resp["tracks"]:
                                    _parts = [f"**{_plt['name']}** — {_plt['artist_str']}"]
                                    if _plt.get("bpm"):
                                        _parts.append(f"🎵 {_plt['bpm']:.0f} BPM")
                                    if _plt.get("camelot"):
                                        _parts.append(f"🎹 {_plt['camelot']}")
                                    st.markdown("  ·  ".join(_parts))

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — My Top Tracks
    # ════════════════════════════════════════════════════════════════════════
    with _sp_tabs[2]:
        if not _sp_connected:
            st.info("🔑  Connect your Spotify account (button above) to see your top tracks.")
        else:
            st.markdown("### Your Top Tracks")
            _tr = st.radio("Time range", ["Last 4 weeks", "Last 6 months", "All time"], horizontal=True, key="sp_tr")
            _tr_map = {"Last 4 weeks": "short_term", "Last 6 months": "medium_term", "All time": "long_term"}
            _top_resp = _api_get_silent(f"/spotify/top?limit=20&time_range={_tr_map[_tr]}")

            if not _top_resp or not _top_resp.get("tracks"):
                st.info("No top tracks found.")
            else:
                _top_tracks = _top_resp["tracks"]
                for _i, _t in enumerate(_top_tracks, 1):
                    _tt1, _tt2, _tt3 = st.columns([1, 5, 1])
                    with _tt1:
                        st.markdown(f"**#{_i}**")
                    with _tt2:
                        _meta = [f"**{_t['name']}** — {_t['artist_str']}"]
                        if _t.get("bpm"):
                            _meta.append(f"🎵 {_t['bpm']:.0f} BPM")
                        if _t.get("camelot"):
                            _meta.append(f"🎹 {_t['camelot']}")
                        if _t.get("energy") is not None:
                            _meta.append(f"⚡ {_t['energy']:.0%}")
                        st.markdown("  ·  ".join(_meta))
                    with _tt3:
                        if st.button("⬇️", key=f"sp_top_{_t['id']}", help=f"Import {_t['name']}"):
                            _imp_r = requests.post(f"{API}/spotify/import", json={
                                "query":   _t["download_query"],
                                "separate": False,
                                "camelot": _t.get("camelot", ""),
                                "bpm":     _t.get("bpm"),
                                "energy":  _t.get("energy"),
                            })
                            if _imp_r.ok:
                                st.success(f"Queued!")
                            else:
                                st.error("Import failed.")
