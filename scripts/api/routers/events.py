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
import weakref
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from scripts.api.auth import DEFAULT_USER, get_user_id

log = logging.getLogger(__name__)

router = APIRouter(prefix="/events", tags=["Events"])

# Set by the lifespan on shutdown; every stream generator wakes on it and ends.
# Generators await closing.wait(), which binds the event to the first loop that
# waits on it (the serving loop in production).
closing = asyncio.Event()

_PING_SEC = 25.0    # idle time before a keep-alive comment

# Machine profile from detect(), filled once at startup off the event loop.
_machine_profile: dict | None = None


def _frame(event_type: str, data: Any) -> str:
    return json.dumps({
        "type": event_type,
        "data": data,
        "ts":   datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    })


def _payload_owner(data: Any) -> str | None:
    """Owner of a job payload (``data["meta"]["user_id"]``); None for owner-less frames."""
    if isinstance(data, dict) and isinstance(data.get("meta"), dict):
        return data["meta"].get("user_id") or DEFAULT_USER
    return None


# ---------------------------------------------------------------------------
# SSE Broadcaster — module-level singleton
# ---------------------------------------------------------------------------

class SSEBroadcaster:
    """
    Fanout broadcaster: maintains one asyncio Queue per connected client,
    tagged with the client's user id. Job payloads go only to the owner's
    queues; owner-less frames (library, system) go to every queue. Each
    streaming generator drains its own queue.
    """

    def __init__(self) -> None:
        self._queues: dict[asyncio.Queue, str] = {}
        self._dropped: weakref.WeakSet[asyncio.Queue] = weakref.WeakSet()
        self._lock = asyncio.Lock()
        self._start_time = time.time()

    async def subscribe(self, user_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        async with self._lock:
            self._queues[q] = user_id
        log.debug("[sse] Client subscribed — %d active connections", len(self._queues))
        return q

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        async with self._lock:
            self._queues.pop(q, None)
            self._dropped.discard(q)
        log.debug("[sse] Client unsubscribed — %d active connections", len(self._queues))

    def is_dropped(self, q: asyncio.Queue) -> bool:
        """True once a queue was dropped for being full; its stream should end."""
        return q in self._dropped

    def _put_or_drop(self, q: asyncio.Queue, payload: str) -> None:
        """Enqueue without blocking; a full queue is dropped. Caller holds the lock."""
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            # Slow consumer — drop it so its stream closes instead of leaking
            self._queues.pop(q, None)
            self._dropped.add(q)
            log.warning("[sse] Dropped slow consumer queue")

    async def broadcast(self, event_type: str, data: Any) -> None:
        """Push a typed event (non-blocking): job payloads to their owner, the rest to all."""
        payload = _frame(event_type, data)
        owner = _payload_owner(data)
        async with self._lock:
            for q, uid in list(self._queues.items()):
                if owner is None or uid == owner:
                    self._put_or_drop(q, payload)

    async def send_heartbeat(self) -> None:
        """One heartbeat tick: each client gets its own user's active job count and the client count."""
        # Import here to avoid circular imports at module load time
        from scripts.api.jobs import _job_user, list_jobs  # type: ignore[import]

        active: dict[str, int] = {}
        for j in list_jobs(user_id="*"):  # box-wide read, split per user below
            if str(j["status"]).split(".")[-1].upper() in ("PENDING", "RUNNING"):
                uid = _job_user(j)
                active[uid] = active.get(uid, 0) + 1

        async with self._lock:
            clients = len(self._queues)
            for q, uid in list(self._queues.items()):
                self._put_or_drop(q, _frame("heartbeat", {
                    "active_jobs": active.get(uid, 0),
                    "clients":     clients,
                }))

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
            await broadcaster.send_heartbeat()
        except Exception as exc:
            log.debug("[sse] Heartbeat error: %s", exc)


def _detect_machine_once() -> None:
    """Run detect() and cache the profile. Runs in a worker thread, never on the loop."""
    global _machine_profile
    from scripts.system.detect_machine import detect, to_dict  # type: ignore[import]

    try:
        _machine_profile = to_dict(detect())
    except Exception as exc:
        log.debug("[sse] Machine detection failed: %s", exc)
        _machine_profile = None


_heartbeat_task: asyncio.Task | None = None
_detect_task: asyncio.Task | None = None


def start_heartbeat() -> None:
    """Schedule the heartbeat task and the one-time machine detection. Call from lifespan startup."""
    global _heartbeat_task, _detect_task
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running() and (_detect_task is None or _detect_task.done()):
            _detect_task = loop.create_task(asyncio.to_thread(_detect_machine_once))
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
    Ends as soon as ``closing`` is set, and once a dropped (full) queue has
    been drained.
    """
    # Send an immediate connection-ack event
    ack = json.dumps({
        "type": "system_status",
        "data": {"status": "ok", "message": "SSE stream connected"},
        "ts":   datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    })
    yield f"data: {ack}\n\n"

    closed = asyncio.ensure_future(closing.wait())
    getter: asyncio.Future | None = None
    try:
        while not closing.is_set():
            if queue.empty() and broadcaster.is_dropped(queue):
                break
            # Wake on the next payload or on `closing`, whichever comes first
            getter = asyncio.ensure_future(queue.get())
            done, _ = await asyncio.wait(
                {getter, closed}, timeout=_PING_SEC, return_when=asyncio.FIRST_COMPLETED,
            )
            if getter in done:
                yield f"data: {getter.result()}\n\n"
                continue
            # A cancelled queue.get() leaves any pending item in the queue
            getter.cancel()
            if not done:
                yield ": ping\n\n"   # SSE comment — keeps proxies alive
    except asyncio.CancelledError:
        pass
    except GeneratorExit:
        pass
    finally:
        if getter is not None and not getter.done():
            getter.cancel()
        closed.cancel()


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
    queue = await broadcaster.subscribe(get_user_id())

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
