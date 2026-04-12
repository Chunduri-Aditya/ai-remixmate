/* ============================================================
   AI RemixMate — Job polling fallback hook
   Used when SSE is unavailable; polls /api/jobs every N seconds
   and syncs results into the Zustand job store.
   ============================================================ */

import { useEffect, useRef } from 'react'
import { useAppStore } from '@/stores/appStore'
import { jobsApi } from '@/lib/api'

const POLL_INTERVAL_MS = 4_000

export function useJobPoller(enabled = true) {
  const { sseConnected, setJobs, setApiHealth } = useAppStore.getState()
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    // Only poll if SSE is not connected
    if (!enabled || sseConnected) return

    async function poll() {
      try {
        const jobs = await jobsApi.list()
        setJobs(jobs)
        setApiHealth('ok')
      } catch {
        setApiHealth('degraded')
      }
    }

    poll()
    timerRef.current = setInterval(poll, POLL_INTERVAL_MS)

    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [enabled, sseConnected, setJobs, setApiHealth])
}
