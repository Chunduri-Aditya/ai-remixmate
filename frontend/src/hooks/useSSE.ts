/* ============================================================
   AI RemixMate — SSE (Server-Sent Events) hook
   Connects to /events, dispatches typed events into Zustand.
   Auto-reconnects with exponential back-off on disconnect.
   ============================================================ */

import { useEffect, useRef } from 'react'
import { useAppStore } from '@/stores/appStore'
import { normalizeJob, EVENTS_URL } from '@/lib/api'
import type { SSEEvent, HeartbeatData } from '@/types'

const SSE_URL = EVENTS_URL
const MIN_RECONNECT_MS = 1_000
const MAX_RECONNECT_MS = 30_000

export function useSSE() {
  const esRef = useRef<EventSource | null>(null)
  const reconnectDelay = useRef(MIN_RECONNECT_MS)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const unmounted = useRef(false)

  const {
    setSseConnected,
    setApiHealth,
    setUptimeSeconds,
    setMachineProfile,
    upsertJob,
    pushActivity,
  } = useAppStore.getState()

  function connect() {
    if (unmounted.current) return

    const es = new EventSource(SSE_URL)
    esRef.current = es

    es.onopen = () => {
      setSseConnected(true)
      setApiHealth('ok')
      reconnectDelay.current = MIN_RECONNECT_MS
    }

    es.onmessage = (raw) => {
      try {
        const event = JSON.parse(raw.data) as SSEEvent
        handleEvent(event)
      } catch {
        // malformed frame — ignore
      }
    }

    es.onerror = () => {
      es.close()
      setSseConnected(false)
      scheduleReconnect()
    }
  }

  function scheduleReconnect() {
    if (unmounted.current) return
    reconnectTimer.current = setTimeout(() => {
      reconnectDelay.current = Math.min(
        reconnectDelay.current * 2,
        MAX_RECONNECT_MS,
      )
      connect()
    }, reconnectDelay.current)
  }

  function handleEvent(event: SSEEvent) {
    switch (event.type) {
      case 'heartbeat': {
        const d = event.data as HeartbeatData
        setUptimeSeconds(d.uptime_seconds)
        setApiHealth('ok')
        if (d.machine_profile) setMachineProfile(d.machine_profile)
        break
      }

      case 'job_created':
      case 'job_updated': {
        const job = normalizeJob(event.data as Record<string, unknown>)
        upsertJob(job)
        break
      }

      case 'job_completed': {
        const job = normalizeJob(event.data as Record<string, unknown>)
        upsertJob(job)
        pushActivity({
          level: 'success',
          message: `Job ${job.type} completed`,
          job_id: job.job_id,
        })
        break
      }

      case 'job_failed': {
        const job = normalizeJob(event.data as Record<string, unknown>)
        upsertJob(job)
        pushActivity({
          level: 'error',
          message: `Job ${job.type} failed: ${job.error ?? 'unknown error'}`,
          job_id: job.job_id,
        })
        break
      }

      case 'job_cancelled': {
        const job = normalizeJob(event.data as Record<string, unknown>)
        upsertJob(job)
        pushActivity({
          level: 'warn',
          message: `Job ${job.type} cancelled`,
          job_id: job.job_id,
        })
        break
      }

      case 'library_changed': {
        pushActivity({ level: 'info', message: 'Library updated' })
        break
      }

      case 'system_status': {
        const d = event.data as { status: 'ok' | 'degraded' | 'down'; message?: string }
        setApiHealth(d.status)
        if (d.message) {
          pushActivity({ level: d.status === 'ok' ? 'info' : 'warn', message: d.message })
        }
        break
      }
    }
  }

  useEffect(() => {
    unmounted.current = false
    connect()

    return () => {
      unmounted.current = true
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      esRef.current?.close()
      setSseConnected(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
}
