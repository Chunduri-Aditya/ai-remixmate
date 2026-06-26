import { Component, type ErrorInfo, type ReactNode } from 'react'
import './PageErrorBoundary.css'

interface Props { children: ReactNode; pageName: string }
interface State { hasError: boolean; error: Error | null }

export class PageErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`[PageErrorBoundary:${this.props.pageName}]`, error, info)
  }

  render() {
    if (!this.state.hasError) return this.props.children
    return (
      <div className="peb-container">
        <div className="peb-card">
          <span className="peb-icon">⚠</span>
          <h2 className="peb-title font-display">{this.props.pageName} failed to load</h2>
          <p className="peb-message text-muted">
            {this.state.error?.message ?? 'An unexpected error occurred.'}
          </p>
          <button
            className="peb-retry"
            onClick={() => this.setState({ hasError: false, error: null })}
          >
            Try again
          </button>
        </div>
      </div>
    )
  }
}
