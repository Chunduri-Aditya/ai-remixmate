import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// start.sh passes its ports in; a direct `npm run dev` uses the defaults.
const apiPort = Number(process.env.REMIXMATE_API_PORT) || 8000
const uiPort = Number(process.env.REMIXMATE_UI_PORT) || 5173

// https://vitejs.dev/config/
export default defineConfig({
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
  },
  // For GitHub Pages project sites set VITE_BASE=/ai-remixmate/ at build time
  base: process.env.VITE_BASE || '/',
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: true,          // bind to 0.0.0.0 for LAN access
    port: uiPort,
    proxy: {
      // All /api/* requests forwarded to FastAPI
      '/api': {
        target: `http://localhost:${apiPort}`,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
      // SSE stream
      '/events': {
        target: `http://localhost:${apiPort}`,
        changeOrigin: true,
      },
      // Static output files (audio, mixes)
      '/outputs': {
        target: `http://localhost:${apiPort}`,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
