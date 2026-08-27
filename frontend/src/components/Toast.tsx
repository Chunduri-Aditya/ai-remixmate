import { useState } from 'react'
import { ToastContext, type ToastItem } from './toastContext'
import './Toast.css'

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
