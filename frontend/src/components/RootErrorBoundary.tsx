import { Component, type ErrorInfo, type ReactNode } from 'react'
import './PageErrorBoundary.css'

interface Props { children: ReactNode }
interface State { hasError: boolean }

/** Last line catch for a render error anywhere in the app: swaps the whole tree for a reload prompt. */
export class RootErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false }

  static getDerivedStateFromError(): State {
    return { hasError: true }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[RootErrorBoundary]', error, info)
  }

  render() {
    if (!this.state.hasError) return this.props.children
    return (
      <div className="peb-container" role="alert">
        <div className="peb-card">
          <span className="peb-icon">⚠</span>
          <h2 className="peb-title font-display">Something went wrong</h2>
          <p className="peb-message text-muted">
            The app hit an unexpected error. Reload to start again.
          </p>
          <button className="peb-retry" onClick={() => window.location.reload()}>
            Reload
          </button>
        </div>
      </div>
    )
  }
}
