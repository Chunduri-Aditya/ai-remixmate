/* ============================================================
   Toast context + consumer hook.

   Split out of Toast.tsx so that file exports only the
   ToastProvider component — react-refresh/only-export-components
   needs component files to export components and nothing else.
   ============================================================ */
import { createContext, useContext } from 'react'

export interface ToastItem {
  id: string
  message: string
  level: 'success' | 'error' | 'info'
  jobId?: string
}

export const ToastContext = createContext<{ push: (t: Omit<ToastItem, 'id'>) => void }>(
  { push: () => {} },
)

export function useToast() { return useContext(ToastContext) }
