import { useEffect, useState } from 'react'
import type { Job } from '@/types'

/** Returns live elapsed seconds and an ETA estimate for a running job. */
export function useJobTimer(job: Job) {
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    const start = new Date(job.created_at).getTime()
    const tick = () => setElapsed(Math.floor((Date.now() - start) / 1000))
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [job.created_at])

  // Need at least 2 % progress to give a meaningful estimate.
  const eta =
    job.progress > 2
      ? Math.round((elapsed / job.progress) * (100 - job.progress))
      : null

  return { elapsed, eta }
}
