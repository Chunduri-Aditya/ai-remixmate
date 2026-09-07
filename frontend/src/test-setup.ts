import '@testing-library/jest-dom'
import { afterEach } from 'vitest'
import { cleanup } from '@testing-library/react'

// Global unmount between tests. Without this every render() in every file
// leaks a mounted tree — which surfaced as duplicate effect teardown counts
// in AuthProvider.test.tsx. Files may still call cleanup() themselves (no-op).
afterEach(cleanup)
