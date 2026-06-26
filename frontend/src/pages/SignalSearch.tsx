/* ============================================================
   AI RemixMate — Signal Search
   Semantic similarity search: pick a reference track, get
   nearest neighbours from the 35-dim embedding index.
   ============================================================ */
import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  Search,
  Zap,
  Music2,
  ChevronDown,
  RefreshCw,
  BarChart2,
} from 'lucide-react'
import { libraryApi, analysisApi } from '@/lib/api'
import type { SimilarTrack, SongInfo } from '@/types'
import './PageBase.css'
import './SignalSearch.css'

function pct(n: number) {
  return Math.round(n * 100)
}

function ScoreRing({ score }: { score: number }) {
  const p = pct(score)
  const color =
    p >= 80 ? 'var(--color-green-500)' :
    p >= 60 ? 'var(--color-amber-500)' :
              'var(--color-ice-400)'
  return (
    <div className="ss-ring" title={`Similarity: ${p}%`}>
      <svg viewBox="0 0 36 36" className="ss-ring__svg">
        <circle cx="18" cy="18" r="14" className="ss-ring__track" />
        <circle
          cx="18" cy="18" r="14"
          className="ss-ring__fill"
          style={{
            stroke: color,
            strokeDashoffset: `${88 - (88 * p) / 100}`,
          }}
        />
      </svg>
      <span className="ss-ring__label font-mono">{p}</span>
    </div>
  )
}

interface ResultCardProps {
  track: SimilarTrack
  rank: number
  refSong: string
}

function ResultCard({ track, rank, refSong }: ResultCardProps) {
  const navigate = useNavigate()
  const [expanded, setExpanded] = useState(false)
  const hasBreakdown = track.breakdown && Object.keys(track.breakdown).length > 0

  return (
    <div className="ss-card">
      <div className="ss-card__top">
        <span className="ss-card__rank font-mono text-muted">#{rank}</span>
        <ScoreRing score={track.score} />
        <div className="ss-card__info">
          <span className="ss-card__name">{track.name}</span>
          <div className="ss-card__meta">
            {track.bpm    && <span className="ss-chip ss-chip--amber">{track.bpm?.toFixed(1)} bpm</span>}
            {track.key    && <span className="ss-chip ss-chip--ice">{track.key}</span>}
            {track.camelot && <span className="ss-chip ss-chip--violet">{track.camelot}</span>}
            {track.genre  && <span className="ss-chip ss-chip--default">{track.genre}</span>}
          </div>
        </div>
        {hasBreakdown && (
          <button
            className="ss-expand-btn"
            onClick={() => setExpanded((v) => !v)}
            title="Show dimension breakdown"
          >
            <ChevronDown
              size={14}
              style={{ transform: expanded ? 'rotate(180deg)' : 'none', transition: 'transform 0.15s' }}
            />
          </button>
        )}
      </div>

      {expanded && hasBreakdown && (
        <div className="ss-breakdown">
          {Object.entries(track.breakdown!).map(([dim, val]) => (
            <div key={dim} className="ss-breakdown__row">
              <span className="ss-breakdown__dim font-mono">{dim}</span>
              <div className="ss-breakdown__bar">
                <div
                  className="ss-breakdown__fill"
                  style={{ width: `${Math.round(val * 100)}%` }}
                />
              </div>
              <span className="ss-breakdown__val font-mono">{Math.round(val * 100)}</span>
            </div>
          ))}
        </div>
      )}

      <div className="ss-card__actions">
        <button
          className="ss-card__action-btn"
          onClick={() => navigate(`/mix-deck?song_a=${encodeURIComponent(refSong)}`)}
        >
          → A
        </button>
        <button
          className="ss-card__action-btn"
          onClick={() => navigate(`/mix-deck?song_b=${encodeURIComponent(track.name)}`)}
        >
          → B
        </button>
        <button
          className="ss-card__action-btn ss-card__action-btn--mix"
          onClick={() => navigate(`/mix-deck?song_a=${encodeURIComponent(refSong)}&song_b=${encodeURIComponent(track.name)}`)}
        >
          ⚡ Mix
        </button>
      </div>
    </div>
  )
}

export default function SignalSearch() {
  const [refSong, setRefSong]         = useState('')
  const [k, setK]                     = useState(10)
  const [results, setResults]         = useState<SimilarTrack[] | null>(null)
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState<string | null>(null)
  const [bpmMin, setBpmMin]           = useState('')
  const [bpmMax, setBpmMax]           = useState('')
  const [keyFilter, setKeyFilter]     = useState('')
  const [energyMin, setEnergyMin]     = useState('')

  const { data: songs = [] } = useQuery<SongInfo[]>({
    queryKey: ['library-atlas'],
    queryFn: libraryApi.list,
    staleTime: 60_000,
  })

  const songNames = useMemo(() => songs.map((s) => s.name).sort(), [songs])

  async function runSearch() {
    if (!refSong) return
    setError(null)
    setLoading(true)
    try {
      const raw = await analysisApi.similar(refSong, k)
      setResults(raw)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Search failed')
    } finally {
      setLoading(false)
    }
  }

  const filteredResults = useMemo(() => {
    if (!results) return null
    let list = [...results]
    const bMin = parseFloat(bpmMin)
    const bMax = parseFloat(bpmMax)
    const eMin = parseFloat(energyMin) / 100
    if (!isNaN(bMin)) list = list.filter((t) => (t.bpm ?? 0) >= bMin)
    if (!isNaN(bMax)) list = list.filter((t) => (t.bpm ?? 999) <= bMax)
    if (keyFilter.trim()) {
      const q = keyFilter.toLowerCase()
      list = list.filter(
        (t) =>
          t.key?.toLowerCase().includes(q) ||
          t.camelot?.toLowerCase().includes(q),
      )
    }
    if (!isNaN(eMin)) list = list.filter((t) => (t.score ?? 0) >= eMin)
    return list
  }, [results, bpmMin, bpmMax, keyFilter, energyMin])

  const refSongInfo = useMemo(
    () => songs.find((s) => s.name === refSong),
    [songs, refSong],
  )

  return (
    <div className="page-base">
      <header className="page-base__header">
        <Search size={20} strokeWidth={1.5} className="page-base__header-icon" />
        <div>
          <h1 className="page-base__title font-display">Signal Search</h1>
          <p className="page-base__sub text-muted">
            35-dim embedding similarity — find tracks that feel like your reference
          </p>
        </div>
      </header>

      <div className="page-base__body ss-body">
        {/* Left: controls */}
        <aside className="ss-panel">
          <div className="ss-panel__section">
            <label className="ss-label">Reference track</label>
            <div className="ss-select-wrap">
              <Music2 size={13} className="ss-select-icon" />
              <select
                className="ss-select"
                value={refSong}
                onChange={(e) => { setRefSong(e.target.value); setResults(null) }}
              >
                <option value="">— pick a song —</option>
                {songNames.map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </div>

            {refSongInfo && (
              <div className="ss-ref-meta">
                {refSongInfo.bpm && <span className="ss-chip ss-chip--amber">{refSongInfo.bpm.toFixed(1)} bpm</span>}
                {refSongInfo.key && <span className="ss-chip ss-chip--ice">{refSongInfo.key}</span>}
                {refSongInfo.camelot && <span className="ss-chip ss-chip--violet">{refSongInfo.camelot}</span>}
              </div>
            )}
          </div>

          <div className="ss-panel__section">
            <label className="ss-label">Results to fetch</label>
            <div className="ss-k-row">
              {[5, 10, 20, 50].map((n) => (
                <button
                  key={n}
                  className={`ss-k-btn font-mono ${k === n ? 'ss-k-btn--active' : ''}`}
                  onClick={() => setK(n)}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          <button
            className="ss-run-btn"
            disabled={!refSong || loading}
            onClick={runSearch}
          >
            {loading
              ? <RefreshCw size={14} className="ss-spin" />
              : <Zap size={14} />}
            {loading ? 'Searching…' : 'Find Similar'}
          </button>

          {/* Post-fetch filters */}
          {results && (
            <div className="ss-panel__section ss-panel__section--filters">
              <label className="ss-label">
                <BarChart2 size={11} style={{ display: 'inline', marginRight: 4 }} />
                Filter results
              </label>
              <div className="ss-filter-row">
                <input className="ss-mini-input" placeholder="BPM min" value={bpmMin} onChange={(e) => setBpmMin(e.target.value)} />
                <span className="text-muted" style={{ fontSize: 'var(--text-xs)' }}>–</span>
                <input className="ss-mini-input" placeholder="max" value={bpmMax} onChange={(e) => setBpmMax(e.target.value)} />
              </div>
              <input
                className="ss-mini-input ss-mini-input--full"
                placeholder="Key / Camelot (e.g. 8A)"
                value={keyFilter}
                onChange={(e) => setKeyFilter(e.target.value)}
              />
              <input
                className="ss-mini-input ss-mini-input--full"
                placeholder="Min similarity % (e.g. 70)"
                value={energyMin}
                onChange={(e) => setEnergyMin(e.target.value)}
              />
            </div>
          )}

          {error && <div className="ss-error">{error}</div>}
        </aside>

        {/* Right: results */}
        <section className="ss-results">
          {!results && !loading && (
            <div className="ss-splash">
              <Search size={40} strokeWidth={0.75} />
              <p className="font-display" style={{ fontSize: 'var(--text-xl)' }}>Pick a track to find its nearest neighbours</p>
              <p className="text-muted" style={{ fontSize: 'var(--text-sm)' }}>
                The 35-dim vector index encodes BPM, key, energy, spectral texture, and more.
              </p>
            </div>
          )}

          {loading && (
            <div className="ss-splash">
              <RefreshCw size={28} className="ss-spin" />
              <p className="text-muted">Running similarity search…</p>
            </div>
          )}

          {filteredResults && !loading && (
            <>
              <div className="ss-results__header">
                <span className="ss-results__count">
                  {filteredResults.length} match{filteredResults.length !== 1 ? 'es' : ''}
                  {results && filteredResults.length < results.length
                    ? ` (filtered from ${results.length})`
                    : ''}
                </span>
                <span className="text-muted" style={{ fontSize: 'var(--text-xs)' }}>
                  Reference: <strong>{refSong}</strong>
                </span>
              </div>

              <div className="ss-list">
                {filteredResults.length === 0 ? (
                  <div className="ss-empty">No tracks match your filters</div>
                ) : (
                  filteredResults.map((t, i) => (
                    <ResultCard key={t.name} track={t} rank={i + 1} refSong={refSong} />
                  ))
                )}
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  )
}
