/* ============================================================
   AI RemixMate — Floating DJ Widget
   Compact song-recommendation panel for live mixing.
   - Pick the "now playing" track (search over library)
   - Harmonic/BPM recommendations from the RAG vector index
   - Set-list mode: queue tracks, always suggests the next one
   - Pop-out: Document Picture-in-Picture (always-on-top, Chrome/Edge)
   ============================================================ */
import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  Radio,
  Search,
  PictureInPicture2,
  Plus,
  X,
  Music2,
  ListMusic,
  RefreshCw,
} from 'lucide-react'
import { libraryApi, analysisApi } from '@/lib/api'
import { CamelotWheel } from '@/components/CamelotWheel'
import type { SimilarTrack, SongInfo } from '@/types'
import './Widget.css'

// --- Camelot wheel colors (rough hue per position) ---

function camelotHue(camelot?: string): string {
  if (!camelot) return 'var(--color-text-muted)'
  const n = parseInt(camelot, 10)
  if (Number.isNaN(n)) return 'var(--color-text-muted)'
  return `hsl(${(n - 1) * 30}, 70%, 60%)`
}

// --- Recommendation row ---

function RecRow({
  rec,
  onQueue,
  onMakeCurrent,
}: {
  rec: SimilarTrack
  onQueue: (name: string) => void
  onMakeCurrent: (name: string) => void
}) {
  return (
    <div className="wgt-rec" onDoubleClick={() => onMakeCurrent(rec.name)} title="Double-click to set as now playing">
      <span className="wgt-rec__camelot font-mono" style={{ color: camelotHue(rec.camelot) }}>
        {rec.camelot ?? '—'}
      </span>
      <div className="wgt-rec__info">
        <span className="wgt-rec__name">{rec.name}</span>
        <span className="wgt-rec__meta text-muted font-mono">
          {rec.genre ?? ''}
        </span>
      </div>
      <span className="wgt-rec__bpm font-mono text-muted">
        {rec.bpm ? `${rec.bpm.toFixed(0)} bpm` : ''}
      </span>
      <span className="wgt-rec__score font-mono">{Math.round(rec.score * 100)}%</span>
      <button className="wgt-rec__add" onClick={() => onQueue(rec.name)} title="Add to set list">
        <Plus size={13} />
      </button>
    </div>
  )
}

// --- The panel itself (rendered inline or portaled into the PiP window) ---

function WidgetPanel({
  inPip,
  onPopOut,
  canPip,
}: {
  inPip: boolean
  onPopOut: () => void
  canPip: boolean
}) {
  const [library, setLibrary] = useState<SongInfo[]>([])
  const [query, setQuery] = useState('')
  const [showResults, setShowResults] = useState(false)
  const [current, setCurrent] = useState<string | null>(null)
  const [setList, setSetList] = useState<string[]>([])
  const [recs, setRecs] = useState<SimilarTrack[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [currentInfo, setCurrentInfo] = useState<SongInfo | null>(null)

  // Load the library once for the search box
  useEffect(() => {
    libraryApi
      .list()
      .then(setLibrary)
      .catch(() => setError('Cannot reach the local API — is it running on port 8000?'))
  }, [])

  // Fetch current track info for key display in CamelotWheel
  useEffect(() => {
    if (!current) { setCurrentInfo(null); return }
    libraryApi.get(current).then(setCurrentInfo).catch(() => setCurrentInfo(null))
  }, [current])

  // The track recommendations are based on: last set-list entry, else manual pick
  const seedTrack = setList.length > 0 ? setList[setList.length - 1] : current

  useEffect(() => {
    if (!seedTrack) {
      setRecs([])
      return
    }
    let stale = false
    setLoading(true)
    setError(null)
    analysisApi
      .similar(seedTrack, 8)
      .then((r) => { if (!stale) setRecs(r) })
      .catch((e) => { if (!stale) setError(e instanceof Error ? e.message : String(e)) })
      .finally(() => { if (!stale) setLoading(false) })
    return () => { stale = true }
  }, [seedTrack])

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return []
    return library.filter((s) => s.name.toLowerCase().includes(q)).slice(0, 12)
  }, [query, library])

  function pick(name: string) {
    setCurrent(name)
    setQuery('')
    setShowResults(false)
  }

  function queueTrack(name: string) {
    setSetList((sl) => (sl[sl.length - 1] === name ? sl : [...sl, name]))
  }

  function removeFromSetList(index: number) {
    setSetList((sl) => sl.filter((_, i) => i !== index))
  }

  return (
    <div className={`wgt ${inPip ? 'wgt--pip' : ''}`}>
      {/* Header */}
      <header className="wgt-head">
        <Radio size={15} className="wgt-head__logo" />
        <span className="wgt-head__title font-display">RemixMate&nbsp;· DJ Assist</span>
        {!inPip && canPip && (
          <button className="wgt-head__pip" onClick={onPopOut} title="Float on top (Picture-in-Picture)">
            <PictureInPicture2 size={14} />
          </button>
        )}
      </header>

      {/* Now playing / search */}
      <div className="wgt-search">
        <Search size={13} className="wgt-search__icon" />
        <input
          className="wgt-search__input"
          placeholder={current ? current : 'Search your library…'}
          value={query}
          onChange={(e) => { setQuery(e.target.value); setShowResults(true) }}
          onFocus={() => setShowResults(true)}
        />
        {showResults && matches.length > 0 && (
          <div className="wgt-search__results">
            {matches.map((s) => (
              <button key={s.name} className="wgt-search__item" onClick={() => pick(s.name)}>
                <Music2 size={11} /> {s.name}
              </button>
            ))}
          </div>
        )}
      </div>

      {seedTrack && (
        <div className="wgt-now">
          <span className="wgt-now__label text-muted">
            {setList.length > 0 ? 'NEXT FROM' : 'NOW PLAYING'}
          </span>
          <span className="wgt-now__name">{seedTrack}</span>
        </div>
      )}

      {current && currentInfo?.key && (
        <div className="wgt-wheel-wrap">
          <CamelotWheel keyA={currentInfo.key} size={160} />
        </div>
      )}

      {/* Recommendations */}
      <div className="wgt-section">
        <div className="wgt-section__head">
          <span className="wgt-section__label">Recommended next</span>
          {loading && <RefreshCw size={11} className="wgt-spin" />}
        </div>
        <div className="wgt-recs">
          {error && <div className="wgt-error">{error}</div>}
          {!error && !seedTrack && (
            <div className="wgt-empty text-muted">Pick a track to get harmonic matches</div>
          )}
          {!error && seedTrack && !loading && recs.length === 0 && (
            <div className="wgt-empty text-muted">No matches — try another track</div>
          )}
          {recs.map((r) => (
            <RecRow key={r.name} rec={r} onQueue={queueTrack} onMakeCurrent={pick} />
          ))}
        </div>
      </div>

      {/* Set list */}
      <div className="wgt-section wgt-section--setlist">
        <div className="wgt-section__head">
          <span className="wgt-section__label">
            <ListMusic size={11} style={{ marginRight: 4, verticalAlign: -1 }} />
            Set list
          </span>
          {setList.length > 0 && (
            <button className="wgt-clear text-muted" onClick={() => setSetList([])}>clear</button>
          )}
        </div>
        <div className="wgt-setlist">
          {setList.length === 0 ? (
            <div className="wgt-empty text-muted">Queue tracks with + — suggestions follow the last one</div>
          ) : (
            setList.map((name, i) => (
              <div key={`${name}-${i}`} className="wgt-setlist__item">
                <span className="wgt-setlist__num font-mono">{i + 1}</span>
                <span className="wgt-setlist__name">{name}</span>
                <button className="wgt-setlist__rm" onClick={() => removeFromSetList(i)} title="Remove">
                  <X size={11} />
                </button>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

// --- Document Picture-in-Picture wrapper ---

interface DocumentPiP {
  requestWindow(opts?: { width?: number; height?: number }): Promise<Window>
}

function getDocumentPiP(): DocumentPiP | null {
  return (window as unknown as { documentPictureInPicture?: DocumentPiP })
    .documentPictureInPicture ?? null
}

function copyStylesTo(win: Window) {
  for (const sheet of Array.from(document.styleSheets)) {
    try {
      const css = Array.from(sheet.cssRules).map((r) => r.cssText).join('\n')
      const style = win.document.createElement('style')
      style.textContent = css
      win.document.head.appendChild(style)
    } catch {
      if (sheet.href) {
        const link = win.document.createElement('link')
        link.rel = 'stylesheet'
        link.href = sheet.href
        win.document.head.appendChild(link)
      }
    }
  }
}

export default function Widget() {
  const [pipWindow, setPipWindow] = useState<Window | null>(null)
  const pipRef = useRef<Window | null>(null)

  async function popOut() {
    const dpp = getDocumentPiP()
    if (!dpp) {
      // Fallback: plain popup (not always-on-top, but separate small window)
      window.open(window.location.href, 'remixmate-widget', 'width=380,height=620,popup=yes')
      return
    }
    try {
      const win = await dpp.requestWindow({ width: 380, height: 620 })
      copyStylesTo(win)
      win.document.title = 'RemixMate — DJ Assist'
      win.document.body.classList.add('wgt-pip-body')
      win.addEventListener('pagehide', () => {
        pipRef.current = null
        setPipWindow(null)
      })
      pipRef.current = win
      setPipWindow(win)
    } catch {
      window.open(window.location.href, 'remixmate-widget', 'width=380,height=620,popup=yes')
    }
  }

  // Close the PiP window if the host page unmounts
  useEffect(() => () => pipRef.current?.close(), [])

  return (
    <div className="wgt-host">
      {pipWindow ? (
        <>
          <div className="wgt-floating-note">
            <PictureInPicture2 size={28} strokeWidth={1.25} />
            <p className="font-display">Widget is floating</p>
            <p className="text-muted">Close the floating window to bring it back here.</p>
          </div>
          {createPortal(
            <WidgetPanel inPip onPopOut={() => {}} canPip={false} />,
            pipWindow.document.body,
          )}
        </>
      ) : (
        // Pop-out always offered — falls back to a plain popup outside Chrome/Edge
        <WidgetPanel inPip={false} onPopOut={popOut} canPip />
      )}
    </div>
  )
}
