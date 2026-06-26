"""
scripts/api/routers/events.py — Server-Sent Events (SSE) backbone.

Provides a real-time event stream consumed by the React frontend.
All significant state changes (job progress, library updates, system status)
are pushed to connected clients without polling.

Endpoints
─────────
  GET /events/stream   — SSE stream (text/event-stream)
  GET /events/health   — quick check: how many clients are connected

SSE event format (JSON in the `data` field):
  {
    "type": "heartbeat" | "job_created" | "job_updated" | "job_completed"
           | "job_failed" | "job_cancelled" | "library_changed" | "system_status",
    "data": { ... },
    "ts":   "2026-04-06T10:30:00.000Z"
  }

Architecture
────────────
  - SSEBroadcaster is a module-level singleton.
  - FastAPI job store calls broadcast() after every status change.
  - A background asyncio task sends heartbeat events every 15 seconds.
  - Clients reconnect automatically on disconnect (EventSource semantics).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

log = logging.getLogger(__name__)

router = APIRouter(prefix="/events", tags=["Events"])

# ---------------------------------------------------------------------------
# SSE Broadcaster — module-level singleton
# ---------------------------------------------------------------------------

class SSEBroadcaster:
    """
    Fanout broadcaster: maintains a set of asyncio Queues, one per connected
    client. Broadcasting puts a message on every queue; each streaming
    generator drains its own queue.
    """

    def __init__(self) -> None:
        self._queues: set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()
        self._start_time = time.time()

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        async with self._lock:
            self._queues.add(q)
        log.debug("[sse] Client subscribed — %d active connections", len(self._queues))
        return q

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        async with self._lock:
            self._queues.discard(q)
        log.debug("[sse] Client unsubscribed — %d active connections", len(self._queues))

    async def broadcast(self, event_type: str, data: Any) -> None:
        """Push a typed event to all connected clients (non-blocking)."""
        payload = json.dumps({
            "type": event_type,
            "data": data,
            "ts":   datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        })
        dead: list[asyncio.Queue] = []
        async with self._lock:
            for q in self._queues:
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    # Slow consumer — mark for removal to avoid memory leak
                    dead.append(q)
            for q in dead:
                self._queues.discard(q)
                log.warning("[sse] Dropped slow consumer queue")

    @property
    def client_count(self) -> int:
        return len(self._queues)

    @property
    def uptime_seconds(self) -> int:
        return int(time.time() - self._start_time)


# Module-level singleton — imported and called by job store and other modules
broadcaster = SSEBroadcaster()


# ---------------------------------------------------------------------------
# Heartbeat task — fires every 15 seconds
# ---------------------------------------------------------------------------

async def _heartbeat_loop() -> None:
    """Background coroutine: sends a heartbeat SSE event every 15 seconds."""
    while True:
        await asyncio.sleep(15)
        try:
            # Import here to avoid circular imports at module load time
            from scripts.api.jobs import list_jobs  # type: ignore[import]
            from scripts.system.detect_machine import detect, to_dict  # type: ignore[import]

            jobs = list_jobs()
            active = sum(
                1 for j in jobs
                if str(j["status"]).split(".")[-1].upper() in ("PENDING", "RUNNING")
            )

            # Include machine profile — cached after first call
            if not hasattr(_heartbeat_loop, "_profile_cache"):
                try:
                    profile = to_dict(detect())
                    _heartbeat_loop._profile_cache = profile  # type: ignore[attr-defined]
                except Exception:
                    _heartbeat_loop._profile_cache = None  # type: ignore[attr-defined]

            await broadcaster.broadcast("heartbeat", {
                "uptime_seconds":  broadcaster.uptime_seconds,
                "active_jobs":     active,
                "api_version":     "1.0.0",
                "connected_clients": broadcaster.client_count,
                "machine_profile": getattr(_heartbeat_loop, "_profile_cache", None),
            })
        except Exception as exc:
            log.debug("[sse] Heartbeat error: %s", exc)


_heartbeat_task: asyncio.Task | None = None


def start_heartbeat() -> None:
    """Schedule the heartbeat background task. Call from FastAPI lifespan startup."""
    global _heartbeat_task
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running() and (_heartbeat_task is None or _heartbeat_task.done()):
            _heartbeat_task = loop.create_task(_heartbeat_loop())
            log.info("[sse] Heartbeat task started")
    except RuntimeError:
        pass  # no event loop yet — will be started when first request arrives


def stop_heartbeat() -> None:
    """Cancel the heartbeat task. Call from FastAPI lifespan shutdown."""
    global _heartbeat_task
    if _heartbeat_task and not _heartbeat_task.done():
        _heartbeat_task.cancel()
        _heartbeat_task = None


# ---------------------------------------------------------------------------
# Convenience wrappers — called by job store and library modules
# ---------------------------------------------------------------------------

async def emit_job_event(event_type: str, job_dict: dict) -> None:
    """Push a job lifecycle event to all SSE clients."""
    await broadcaster.broadcast(event_type, job_dict)


async def emit_library_changed() -> None:
    """Notify clients that the library has been modified."""
    await broadcaster.broadcast("library_changed", {"ts": datetime.now(timezone.utc).isoformat()})


async def emit_system_status(status: str, message: str | None = None) -> None:
    """Push a system status change (ok / degraded / down)."""
    data: dict[str, Any] = {"status": status}
    if message:
        data["message"] = message
    await broadcaster.broadcast("system_status", data)


# ---------------------------------------------------------------------------
# SSE streaming generator
# ---------------------------------------------------------------------------

async def _sse_generator(queue: asyncio.Queue) -> AsyncIterator[str]:
    """
    Async generator that drains a subscriber queue and yields SSE frames.
    Sends a comment ping every 25 s to keep connections alive through proxies.
    """
    # Send an immediate connection-ack event
    ack = json.dumps({
        "type": "system_status",
        "data": {"status": "ok", "message": "SSE stream connected"},
        "ts":   datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    })
    yield f"data: {ack}\n\n"

    while True:
        try:
            # Wait up to 25 seconds; if nothing arrives, send a keep-alive comment
            payload = await asyncio.wait_for(queue.get(), timeout=25.0)
            yield f"data: {payload}\n\n"
        except asyncio.TimeoutError:
            yield ": ping\n\n"   # SSE comment — keeps proxies alive
        except asyncio.CancelledError:
            break
        except GeneratorExit:
            break


# ---------------------------------------------------------------------------
# Router endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/stream",
    summary="SSE event stream",
    description=(
        "Server-Sent Events stream. Connect with EventSource('/events/stream'). "
        "Events: heartbeat, job_created, job_updated, job_completed, job_failed, "
        "job_cancelled, library_changed, system_status."
    ),
)
async def sse_stream() -> StreamingResponse:
    queue = await broadcaster.subscribe()

    async def event_stream():
        try:
            async for chunk in _sse_generator(queue):
                yield chunk
        finally:
            await broadcaster.unsubscribe(queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",     # disable Nginx buffering
            "Connection":        "keep-alive",
        },
    )


@router.get(
    "/health",
    summary="SSE health check",
)
async def sse_health() -> dict:
    return {
        "connected_clients": broadcaster.client_count,
        "uptime_seconds":    broadcaster.uptime_seconds,
    }
