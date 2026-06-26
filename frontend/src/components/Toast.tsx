import { createContext, useContext, useState } from 'react'
import './Toast.css'

export interface ToastItem {
  id: string
  message: string
  level: 'success' | 'error' | 'info'
  jobId?: string
}

const ToastContext = createContext<{ push: (t: Omit<ToastItem, 'id'>) => void }>(
  { push: () => {} },
)

export function useToast() { return useContext(ToastContext) }

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([])

  function push(t: Omit<ToastItem, 'id'>) {
    const id = crypto.randomUUID()
    setToasts((prev) => [...prev.slice(-4), { ...t, id }])
    setTimeout(() => setToasts((prev) => prev.filter((x) => x.id !== id)), 4000)
  }

  return (
    <ToastContext.Provider value={{ push }}>
      {children}
      <div className="toast-area" aria-live="polite">
        {toasts.map((t) => (
          <div key={t.id} className={`toast toast--${t.level}`}>
            <span className="toast__message">{t.message}</span>
            <button
              className="toast__close"
              onClick={() => setToasts((prev) => prev.filter((x) => x.id !== t.id))}
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}
