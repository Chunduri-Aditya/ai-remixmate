# ─── AI RemixMate — Multi-stage Docker Build ───────────────────────────────
FROM python:3.11-slim AS base

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
# Canonical runtime dirs match scripts/core/paths.py:
#   library/ — downloaded songs
#   outputs/ — rendered mixes  (was: output/ + mixes/ — stale, removed)
#   data/    — SQLite DB, embeddings, audit log
#   models/  — ML model weights
RUN useradd -m -s /bin/bash remixmate \
    && mkdir -p /app/library /app/data /app/outputs /app/models \
    && chown -R remixmate:remixmate /app
USER remixmate

# Expose API + Streamlit ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: start API server
CMD ["python", "-m", "scripts.api.main"]
