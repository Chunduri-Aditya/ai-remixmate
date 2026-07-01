/* ============================================================
   AI RemixMate — Library Atlas
   Song table with search, filters, favorites, per-song actions.
   ============================================================ */
import { useState, useMemo, useCallback, Fragment, useRef, useEffect, CSSProperties } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { useNavigate } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Library,
  Search,
  Star,
  StarOff,
  Cpu,
  Scissors,
  Trash2,
  ChevronUp,
  ChevronDown,
  RefreshCw,
  Filter,
  X,
  Music2,
  Loader2,
  BarChart2,
  Layers,
  ArrowRight,
} from 'lucide-react'
import { libraryApi, analysisApi, stemsApi, favoritesApi, cratesApi } from '@/lib/api'
import { useAppStore } from '@/stores/appStore'
import type { SongInfo, Recommendation, SimilarTrack } from '@/types'
import { StemsPlayerRow } from '@/components/StemsPlayerRow'
import './PageBase.css'
import './LibraryAtlas.css'

type SortKey = 'name' | 'bpm' | 'key' | 'energy' | 'duration'
type SortDir = 'asc' | 'desc'

function fmt(n: number | undefined, decimals = 0) {
  if (n === undefined || n === null) return '—'
  return n.toFixed(decimals)
}

function fmtDuration(s: number | undefined): string {
  if (!s) return '—'
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}

function EnergyBar({ value }: { value?: number }) {
  if (value === undefined || value === null) return <span className="la-muted">—</span>
  const pct = Math.round(value * 100)
  const color =
    pct > 70 ? 'var(--color-crimson-500)' :
    pct > 40 ? 'var(--color-amber-500)' :
               'var(--color-ice-400)'
  return (
    <div className="la-energy">
      <div className="la-energy__bar">
        <div className="la-energy__fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="la-energy__label font-mono">{pct}</span>
    </div>
  )
}

function KeyBadge({ k, camelot }: { k?: string; camelot?: string }) {
  if (!k && !camelot) return <span className="la-muted">—</span>
  return (
    <div className="la-key-wrap">
      {k && <span className="la-badge la-badge--ice">{k}</span>}
      {camelot && <span className="la-badge la-badge--violet">{camelot}</span>}
    </div>
  )
}

interface SongRowProps {
  song: SongInfo
  isFav: boolean
  expandedStems: string | null
  selected: boolean
  onFav: (name: string, add: boolean) => void
  onAnalyze: (name: string) => void
  onStems: (name: string) => void
  onDelete: (name: string) => void
  onSelect: (song: SongInfo) => void
  onExpandStems: (name: string) => void
  onToggleSelect: (name: string) => void
}

function SongRow({ song, isFav, expandedStems, selected, onFav, onAnalyze, onStems, onDelete, onSelect, onExpandStems, onToggleSelect }: SongRowProps) {
  const stemsExpanded = expandedStems === song.name
  return (
    <tr className="la-row la-row--clickable" onClick={() => onSelect(song)}>
      <td className="la-cell" onClick={(e) => e.stopPropagation()}>
        <input
          type="checkbox"
          className="la-checkbox"
          checked={selected}
          onChange={() => onToggleSelect(song.name)}
        />
      </td>
      <td className="la-cell la-cell--name" title={song.name}>
        <div className="la-name">
          <button
            className={`la-fav-btn ${isFav ? 'la-fav-btn--active' : ''}`}
            onClick={(e) => { e.stopPropagation(); onFav(song.name, !isFav) }}
            title={isFav ? 'Remove from favorites' : 'Add to favorites'}
          >
            {isFav ? <Star size={12} /> : <StarOff size={12} />}
          </button>
          <span className="la-name__text">{song.name}</span>
        </div>
      </td>
      <td className="la-cell la-cell--mono">{fmt(song.bpm, 1)}</td>
      <td className="la-cell">
        <KeyBadge k={song.key} camelot={song.camelot} />
      </td>
      <td className="la-cell"><EnergyBar value={song.energy} /></td>
      <td className="la-cell la-cell--mono">{fmtDuration(song.duration)}</td>
      <td className="la-cell">
        <div className="la-chips">
          {song.has_stems    && <span className="la-chip la-chip--violet">stems</span>}
          {song.has_analysis && <span className="la-chip la-chip--green">analyzed</span>}
          {song.stems && song.stems.length > 0 && (
            <button
              className={`la-chip la-chip--stems-btn ${stemsExpanded ? 'la-chip--stems-btn--active' : ''}`}
              onClick={(e) => { e.stopPropagation(); onExpandStems(song.name) }}
              title={stemsExpanded ? 'Collapse stems player' : 'Expand stems player'}
            >
              ♪ Stems
            </button>
          )}
        </div>
      </td>
      <td className="la-cell la-cell--actions" onClick={(e) => e.stopPropagation()}>
        <div className="la-actions">
          <button
            className="la-action-btn"
            onClick={() => onAnalyze(song.name)}
            title={song.has_analysis ? 'Already analyzed' : 'Analyze track'}
            disabled={song.has_analysis}
          >
            <Cpu size={12} />
          </button>
          <button
            className="la-action-btn"
            onClick={() => onStems(song.name)}
            title={song.has_stems ? 'Stems already split' : 'Split stems'}
            disabled={song.has_stems}
          >
            <Scissors size={12} />
          </button>
          <button
            className="la-action-btn la-action-btn--danger"
            onClick={() => onDelete(song.name)}
            title="Delete track"
          >
            <Trash2 size={12} />
          </button>
        </div>
      </td>
    </tr>
  )
}

function StemExpansionRow({ song }: { song: SongInfo }) {
  return (
    <tr className="la-stems-expansion-row">
      <td colSpan={8} className="la-stems-expansion-cell">
        <StemsPlayerRow name={song.name} stems={song.stems!} />
      </td>
    </tr>
  )
}

// ── Song Detail Drawer ────────────────────────────────────────────────────────

function CratesSection({ song }: { song: SongInfo }) {
  const queryClient = useQueryClient()

  const { data: allCrates = [] } = useQuery({
    queryKey: ['crates'],
    queryFn: cratesApi.list,
    staleTime: 30_000,
  })

  const { data: songCrates = [] } = useQuery({
    queryKey: ['crate-songs', song.name],
    queryFn: async () => {
      const results = await Promise.all(
        allCrates.map((c) =>
          cratesApi.songs(c.id).then((names) => ({ id: c.id, has: names.includes(song.name) }))
        ),
      )
      return results.filter((r) => r.has).map((r) => r.id)
    },
    enabled: allCrates.length > 0,
  })

  const [newName, setNewName] = useState('')
  const [creating, setCreating] = useState(false)

  async function toggleCrate(crateId: number, currently: boolean) {
    if (currently) {
      await cratesApi.removeSong(crateId, song.name)
    } else {
      await cratesApi.addSong(crateId, song.name)
    }
    queryClient.invalidateQueries({ queryKey: ['crate-songs', song.name] })
  }

  async function createCrate() {
    if (!newName.trim()) return
    setCreating(true)
    try {
      await cratesApi.create(newName.trim())
      queryClient.invalidateQueries({ queryKey: ['crates'] })
      setNewName('')
    } finally {
      setCreating(false)
    }
  }

  return (
    <div className="la-drawer__section">
      <span className="la-drawer__section-title">Crates</span>
      <div className="la-crates-list">
        {allCrates.map((c) => (
          <label key={c.id} className="la-crate-item">
            <input
              type="checkbox"
              className="la-checkbox"
              checked={songCrates.includes(c.id)}
              onChange={() => toggleCrate(c.id, songCrates.includes(c.id))}
            />
            <span>{c.name}</span>
            <span className="text-muted font-mono la-crate-count">{c.song_count ?? ''}</span>
          </label>
        ))}
      </div>
      <div className="la-crate-create">
        <input
          className="la-crate-input"
          placeholder="New crate…"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && createCrate()}
        />
        <button
          className="la-crate-add-btn"
          onClick={createCrate}
          disabled={creating || !newName.trim()}
        >
          +
        </button>
      </div>
    </div>
  )
}

function SongDrawer({
  song,
  onClose,
  onLoadDeck,
}: {
  song: SongInfo | null
  onClose: () => void
  onLoadDeck: (name: string, deck: 'a' | 'b') => void
}) {
  const { data: recs = [], isLoading: recsLoading } = useQuery<Recommendation[]>({
    queryKey: ['recommend-drawer', song?.name],
    queryFn: () => analysisApi.recommend(song!.name, 6),
    enabled: !!song,
    staleTime: 5 * 60_000,
    retry: false,
  })

  const { data: similar = [], isLoading: simLoading } = useQuery<SimilarTrack[]>({
    queryKey: ['similar-drawer', song?.name],
    queryFn: () => analysisApi.similar(song!.name, 6),
    enabled: !!song,
    staleTime: 5 * 60_000,
    retry: false,
  })

  return (
    <aside className={`la-drawer ${song ? 'la-drawer--open' : ''}`}>
      <div className="la-drawer__header">
        <span className="la-drawer__title font-display" title={song?.name ?? ''}>
          {song?.name ?? ''}
        </span>
        <button className="la-drawer__close" onClick={onClose} title="Close">
          <X size={14} />
        </button>
      </div>

      {song && (
        <div className="la-drawer__body">
          {/* Full metadata */}
          <div className="la-drawer__section">
            <div className="la-drawer__meta-grid">
              {song.bpm      != null && <><span className="la-drawer__meta-label">BPM</span><span className="font-mono">{song.bpm.toFixed(1)}</span></>}
              {song.key                    && <><span className="la-drawer__meta-label">Key</span><span>{song.key}</span></>}
              {song.camelot               && <><span className="la-drawer__meta-label">Camelot</span><span className="font-mono">{song.camelot}</span></>}
              {song.energy   != null && <><span className="la-drawer__meta-label">Energy</span><span className="font-mono">{Math.round(song.energy * 100)}%</span></>}
              {song.duration != null && <><span className="la-drawer__meta-label">Duration</span><span className="font-mono">{fmtDuration(song.duration)}</span></>}
              {song.genre                 && <><span className="la-drawer__meta-label">Genre</span><span>{song.genre}</span></>}
            </div>
            <div className="la-chips" style={{ marginTop: 'var(--space-2)' }}>
              {song.has_stems    && <span className="la-chip la-chip--violet">stems</span>}
              {song.has_analysis && <span className="la-chip la-chip--green">analyzed</span>}
            </div>
          </div>

          {/* Deck quick-load */}
          <div className="la-drawer__section">
            <div className="la-drawer__deck-row">
              <button
                className="la-drawer__deck-btn la-drawer__deck-btn--a"
                onClick={() => onLoadDeck(song.name, 'a')}
              >
                <Music2 size={12} /> Load Deck A
              </button>
              <button
                className="la-drawer__deck-btn la-drawer__deck-btn--b"
                onClick={() => onLoadDeck(song.name, 'b')}
              >
                <Music2 size={12} /> Load Deck B
              </button>
            </div>
          </div>

          {/* Recommendations */}
          <div className="la-drawer__section">
            <span className="la-drawer__section-title">Recommendations</span>
            {recsLoading ? (
              <div className="la-drawer__loading"><Loader2 size={14} className="la-spin" /></div>
            ) : recs.length === 0 ? (
              <p className="la-drawer__empty">No data — analyze this track first</p>
            ) : (
              <div className="la-drawer__rec-list">
                {recs.slice(0, 6).map((rec) => {
                  const score = rec.score ?? rec.overall
                  return (
                    <div key={rec.name} className="la-drawer__rec-item">
                      <span className="la-drawer__rec-name">{rec.name}</span>
                      {score !== undefined && (
                        <span className="la-drawer__rec-score font-mono">
                          {Math.round(score * 100)}%
                        </span>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </div>

          {/* Similar tracks with score rings */}
          <div className="la-drawer__section">
            <span className="la-drawer__section-title">Similar Tracks</span>
            {simLoading ? (
              <div className="la-drawer__loading"><Loader2 size={14} className="la-spin" /></div>
            ) : similar.length === 0 ? (
              <p className="la-drawer__empty">No data — rebuild embedding index first</p>
            ) : (
              <div className="la-drawer__sim-list">
                {similar.slice(0, 6).map((track) => {
                  const p = Math.round(track.score * 100)
                  const ringColor =
                    p >= 80 ? 'var(--color-green-500)'  :
                    p >= 60 ? 'var(--color-amber-500)'  :
                              'var(--color-ice-400)'
                  return (
                    <div key={track.name} className="la-drawer__sim-item">
                      <div className="la-drawer__sim-ring" title={`${p}% similar`}>
                        <svg viewBox="0 0 36 36" className="la-drawer__ring-svg">
                          <circle cx="18" cy="18" r="14" className="la-drawer__ring-track" />
                          <circle
                            cx="18" cy="18" r="14"
                            className="la-drawer__ring-fill"
                            style={{
                              stroke: ringColor,
                              strokeDashoffset: `${88 - (88 * p) / 100}`,
                            }}
                          />
                        </svg>
                        <span className="la-drawer__ring-label font-mono">{p}</span>
                      </div>
                      <div className="la-drawer__sim-info">
                        <span className="la-drawer__sim-name">{track.name}</span>
                        <div className="la-chips">
                          {track.bpm     && <span className="la-chip la-chip--amber">{track.bpm.toFixed(0)}</span>}
                          {track.camelot && <span className="la-chip la-chip--violet">{track.camelot}</span>}
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>

          {/* Crates membership */}
          <CratesSection song={song} />
        </div>
      )}
    </aside>
  )
}

export default function LibraryAtlas() {
  const [search, setSearch]               = useState('')
  const [sortKey, setSortKey]             = useState<SortKey>('name')
  const [sortDir, setSortDir]             = useState<SortDir>('asc')
  const [filterStems, setFilterStems]     = useState(false)
  const [filterAnalyzed, setFilterAnalyzed] = useState(false)
  const [filterFavs, setFilterFavs]       = useState(false)
  const [filterHigh, setFilterHigh]       = useState(false)
  const [selectedSong, setSelectedSong]   = useState<SongInfo | null>(null)
  const [expandedStems, setExpandedStems] = useState<string | null>(null)
  const [selected, setSelected]           = useState<Set<string>>(new Set())
  const headerCheckboxRef                 = useRef<HTMLInputElement>(null)
  const scrollRef                         = useRef<HTMLDivElement>(null)

  const upsertJob     = useAppStore((s) => s.upsertJob)
  const setActiveNav  = useAppStore((s) => s.setActiveNav)
  const queryClient   = useQueryClient()
  const navigate      = useNavigate()

  const { data: songs = [], isLoading: songsLoading, refetch } = useQuery({
    queryKey: ['library-atlas'],
    queryFn:  libraryApi.list,
    staleTime: 30_000,
  })

  const { data: favorites = [], refetch: refetchFavs } = useQuery({
    queryKey: ['favorites'],
    queryFn:  favoritesApi.list,
    staleTime: 30_000,
  })

  const favSet = useMemo(() => new Set(favorites), [favorites])

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    else { setSortKey(key); setSortDir('asc') }
  }

  const filtered = useMemo(() => {
    let list = [...songs]
    if (search.trim()) {
      const q = search.toLowerCase()
      list = list.filter((s) => s.name.toLowerCase().includes(q))
    }
    if (filterStems)    list = list.filter((s) => s.has_stems)
    if (filterAnalyzed) list = list.filter((s) => s.has_analysis)
    if (filterFavs)     list = list.filter((s) => favSet.has(s.name))
    if (filterHigh)     list = list.filter((s) => (s.energy ?? 0) >= 0.7)

    list.sort((a, b) => {
      let av: number | string = 0
      let bv: number | string = 0
      switch (sortKey) {
        case 'name':     av = a.name.toLowerCase(); bv = b.name.toLowerCase(); break
        case 'bpm':      av = a.bpm      ?? 0;       bv = b.bpm      ?? 0;      break
        case 'key':      av = a.key      ?? '';       bv = b.key      ?? '';     break
        case 'energy':   av = a.energy   ?? 0;        bv = b.energy   ?? 0;      break
        case 'duration': av = a.duration ?? 0;        bv = b.duration ?? 0;      break
      }
      if (av < bv) return sortDir === 'asc' ? -1 : 1
      if (av > bv) return sortDir === 'asc' ?  1 : -1
      return 0
    })
    return list
  }, [songs, search, filterStems, filterAnalyzed, filterFavs, filterHigh, sortKey, sortDir, favSet])

  // Virtual scrolling — only active when no stems row is expanded.
  // Padding-row technique: two sentinel <tr>s maintain correct scroll height
  // without requiring position:absolute on rows (which breaks column widths).
  const virtualizer = useVirtualizer({
    count: expandedStems === null ? filtered.length : 0,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 48,
    overscan: 8,
  })

  const handleFav = useCallback(
    async (name: string, add: boolean) => {
      try {
        if (add) await favoritesApi.add(name)
        else     await favoritesApi.remove(name)
        refetchFavs()
      } catch { /* silent */ }
    },
    [refetchFavs],
  )

  const handleAnalyze = useCallback(
    async (name: string) => {
      try {
        const res = await analysisApi.analyze(name)
        if ('job_id' in res) {
          upsertJob({
            job_id: res.job_id, status: 'PENDING', type: 'analyze',
            progress: 0, message: `Analyzing ${name}`,
            created_at: new Date().toISOString(), updated_at: new Date().toISOString(),
          })
        }
      } catch { /* silent */ }
    },
    [upsertJob],
  )

  const handleStems = useCallback(
    async (name: string) => {
      try {
        const res = await stemsApi.split(name)
        if (res?.job_id) {
          upsertJob({
            job_id: res.job_id, status: 'PENDING', type: 'stems',
            progress: 0, message: `Splitting ${name}`,
            created_at: new Date().toISOString(), updated_at: new Date().toISOString(),
          })
        }
      } catch { /* silent */ }
    },
    [upsertJob],
  )

  const handleDelete = useCallback(
    async (name: string) => {
      if (!confirm(`Delete "${name}" from library?`)) return
      try {
        await libraryApi.delete(name)
        queryClient.invalidateQueries({ queryKey: ['library-atlas'] })
        queryClient.invalidateQueries({ queryKey: ['library-stats'] })
      } catch { /* silent */ }
    },
    [queryClient],
  )

  const handleSelect = useCallback((song: SongInfo) => {
    setSelectedSong((prev) => (prev?.name === song.name ? null : song))
  }, [])

  const handleExpandStems = useCallback((name: string) => {
    setExpandedStems((prev) => (prev === name ? null : name))
  }, [])

  const handleToggleSelect = useCallback((name: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }, [])

  const handleSelectAll = useCallback(() => {
    const allSelected = filtered.length > 0 && filtered.every((s) => selected.has(s.name))
    setSelected(allSelected ? new Set() : new Set(filtered.map((s) => s.name)))
  }, [filtered, selected])

  const handleBulkAnalyze = useCallback(async () => {
    const names = Array.from(selected)
    await Promise.allSettled(
      names.map(async (name) => {
        try {
          const res = await analysisApi.analyze(name)
          if ('job_id' in res) {
            upsertJob({
              job_id: res.job_id, status: 'PENDING', type: 'analyze',
              progress: 0, message: `Analyzing ${name}`,
              created_at: new Date().toISOString(), updated_at: new Date().toISOString(),
            })
          }
        } catch { /* silent */ }
      }),
    )
    setSelected(new Set())
  }, [selected, upsertJob])

  const handleBulkStems = useCallback(async () => {
    const names = Array.from(selected)
    await Promise.allSettled(
      names.map(async (name) => {
        try {
          const res = await stemsApi.split(name)
          if (res?.job_id) {
            upsertJob({
              job_id: res.job_id, status: 'PENDING', type: 'stems',
              progress: 0, message: `Splitting ${name}`,
              created_at: new Date().toISOString(), updated_at: new Date().toISOString(),
            })
          }
        } catch { /* silent */ }
      }),
    )
    setSelected(new Set())
  }, [selected, upsertJob])

  // Songs missing BPM/key/energy — backend's has_analysis() check, mirrored
  // here from has_analysis flag on each row. Drives the "Analyze missing"
  // button below; no manual row-selection required.
  const missingCount = useMemo(
    () => songs.filter((s) => !s.has_analysis).length,
    [songs],
  )
  const [analyzingMissing, setAnalyzingMissing] = useState(false)

  const handleAnalyzeMissing = useCallback(async () => {
    setAnalyzingMissing(true)
    try {
      const res = await analysisApi.analyzeMissing()
      if ('job_id' in res) {
        upsertJob({
          job_id: res.job_id, status: 'PENDING', type: 'analyze',
          progress: 0, message: `Analyzing ${missingCount} song(s) missing data…`,
          created_at: new Date().toISOString(), updated_at: new Date().toISOString(),
        })
      }
    } catch { /* surfaced via job store / toast elsewhere */ } finally {
      setAnalyzingMissing(false)
    }
  }, [upsertJob, missingCount])

  // Sync header checkbox indeterminate state
  useEffect(() => {
    if (!headerCheckboxRef.current) return
    const allSelected = filtered.length > 0 && filtered.every((s) => selected.has(s.name))
    const someSelected = filtered.some((s) => selected.has(s.name))
    headerCheckboxRef.current.indeterminate = someSelected && !allSelected
  }, [selected, filtered])

  const handleLoadDeck = useCallback(
    (name: string, deck: 'a' | 'b') => {
      setSelectedSong(null)
      setActiveNav('mix-deck')
      navigate(`/mix-deck?song_${deck}=${encodeURIComponent(name)}`)
    },
    [navigate, setActiveNav],
  )

  function SortIcon({ col }: { col: SortKey }) {
    if (sortKey !== col)
      return <span className="la-sort-icon la-sort-icon--inactive"><ChevronUp size={10} /></span>
    return sortDir === 'asc'
      ? <ChevronUp size={10} className="la-sort-icon" />
      : <ChevronDown size={10} className="la-sort-icon" />
  }

  const activeFilters =
    [filterStems, filterAnalyzed, filterFavs, filterHigh].filter(Boolean).length

  return (
    <div className="page-base la-page">
      <header className="page-base__header">
        <Library size={20} strokeWidth={1.5} className="page-base__header-icon" />
        <div style={{ flex: 1 }}>
          <h1 className="page-base__title font-display">Library Atlas</h1>
          <p className="page-base__sub text-muted">
            {songsLoading
              ? 'Loading…'
              : `${filtered.length.toLocaleString()} / ${songs.length.toLocaleString()} tracks`}
          </p>
        </div>
        <button className="la-refresh-btn" onClick={() => refetch()} title="Refresh library">
          <RefreshCw size={14} className={songsLoading ? 'la-spin' : ''} />
        </button>
      </header>

      {/* Toolbar */}
      <div className="la-toolbar">
        <div className="la-search-wrap">
          <Search size={13} className="la-search-icon" />
          <input
            className="la-search"
            data-library-search-input
            aria-label="Search library"
            placeholder="Search by name…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          {search && (
            <button className="la-search-clear" onClick={() => setSearch('')}>
              <X size={12} />
            </button>
          )}
        </div>

        <div className="la-filters">
          <Filter size={11} className="la-filter-icon text-muted" />
          <button
            className={`la-filter-chip ${filterFavs ? 'la-filter-chip--active' : ''}`}
            onClick={() => setFilterFavs((v) => !v)}
          >
            <Star size={10} /> Favorites
          </button>
          <button
            className={`la-filter-chip ${filterStems ? 'la-filter-chip--active' : ''}`}
            onClick={() => setFilterStems((v) => !v)}
          >
            Stems only
          </button>
          <button
            className={`la-filter-chip ${filterAnalyzed ? 'la-filter-chip--active' : ''}`}
            onClick={() => setFilterAnalyzed((v) => !v)}
          >
            Analyzed
          </button>
          <button
            className={`la-filter-chip ${filterHigh ? 'la-filter-chip--active la-filter-chip--crimson' : ''}`}
            onClick={() => setFilterHigh((v) => !v)}
          >
            High energy
          </button>
          {activeFilters > 0 && (
            <button
              className="la-filter-chip la-filter-chip--clear"
              onClick={() => {
                setFilterStems(false)
                setFilterAnalyzed(false)
                setFilterFavs(false)
                setFilterHigh(false)
              }}
            >
              <X size={10} /> Clear {activeFilters}
            </button>
          )}
        </div>

        {missingCount > 0 && (
          <button
            className="la-filter-chip la-filter-chip--missing"
            onClick={handleAnalyzeMissing}
            disabled={analyzingMissing}
            title={`${missingCount} song(s) have no BPM/key/energy data`}
          >
            {analyzingMissing
              ? <Loader2 size={10} className="la-spin" />
              : <BarChart2 size={10} />}
            Analyze missing ({missingCount})
          </button>
        )}
      </div>

      {/* Table + drawer */}
      <div className="la-body">
        <div className="la-table-wrap" ref={scrollRef}>
          {songsLoading ? (
            <div className="la-state">
              <RefreshCw size={20} className="la-spin" />
              <span className="text-muted">Loading library…</span>
            </div>
          ) : songs.length === 0 ? (
            <div className="la-state">
              <Library size={32} strokeWidth={1} style={{ color: 'var(--color-text-faint)' }} />
              <span className="text-muted">Library is empty — add songs in Operations</span>
              <button className="la-empty-action" onClick={() => navigate('/operations')}>
                Add songs in Operations <ArrowRight size={13} />
              </button>
            </div>
          ) : filtered.length === 0 ? (
            <div className="la-state">
              <Search size={28} strokeWidth={1} style={{ color: 'var(--color-text-faint)' }} />
              <span className="text-muted">No tracks match your filters</span>
              <button
                className="la-empty-action"
                onClick={() => {
                  setSearch('')
                  setFilterStems(false)
                  setFilterAnalyzed(false)
                  setFilterFavs(false)
                  setFilterHigh(false)
                }}
              >
                Reset filters <ArrowRight size={13} />
              </button>
            </div>
          ) : (
            <table className="la-table">
              <thead>
                <tr>
                  <th className="la-th la-th--cb">
                    <input
                      ref={headerCheckboxRef}
                      type="checkbox"
                      className="la-checkbox"
                      checked={filtered.length > 0 && filtered.every((s) => selected.has(s.name))}
                      onChange={handleSelectAll}
                    />
                  </th>
                  <th className="la-th la-th--name la-th--sort" onClick={() => toggleSort('name')}>
                    Name <SortIcon col="name" />
                  </th>
                  <th className="la-th la-th--sort" onClick={() => toggleSort('bpm')}>
                    BPM <SortIcon col="bpm" />
                  </th>
                  <th className="la-th la-th--sort" onClick={() => toggleSort('key')}>
                    Key <SortIcon col="key" />
                  </th>
                  <th className="la-th la-th--sort" onClick={() => toggleSort('energy')}>
                    Energy <SortIcon col="energy" />
                  </th>
                  <th className="la-th la-th--sort" onClick={() => toggleSort('duration')}>
                    Dur. <SortIcon col="duration" />
                  </th>
                  <th className="la-th">Tags</th>
                  <th className="la-th la-th--actions">Actions</th>
                </tr>
              </thead>
              {expandedStems !== null ? (
                // Fallback: normal rendering when a stems row is expanded.
                // Virtual scrolling and absolute-positioned expansion rows
                // interact awkwardly, so we disable it for this case.
                <tbody>
                  {filtered.map((song: SongInfo) => (
                    <Fragment key={song.name}>
                      <SongRow
                        song={song}
                        isFav={favSet.has(song.name)}
                        expandedStems={expandedStems}
                        selected={selected.has(song.name)}
                        onFav={handleFav}
                        onAnalyze={handleAnalyze}
                        onStems={handleStems}
                        onDelete={handleDelete}
                        onSelect={handleSelect}
                        onExpandStems={handleExpandStems}
                        onToggleSelect={handleToggleSelect}
                      />
                      {expandedStems === song.name && song.stems && song.stems.length > 0 && (
                        <StemExpansionRow song={song} />
                      )}
                    </Fragment>
                  ))}
                </tbody>
              ) : (
                // Virtual path: padding-row technique keeps correct scroll height
                // and table column widths without position:absolute on <tr>.
                <tbody>
                  {virtualizer.getVirtualItems().length > 0 && (
                    <tr style={{ border: 0 } as CSSProperties}>
                      <td
                        colSpan={8}
                        style={{ height: virtualizer.getVirtualItems()[0].start, padding: 0, border: 0 } as CSSProperties}
                      />
                    </tr>
                  )}
                  {virtualizer.getVirtualItems().map((vRow) => {
                    const song = filtered[vRow.index]
                    return (
                      <SongRow
                        key={song.name}
                        song={song}
                        isFav={favSet.has(song.name)}
                        expandedStems={expandedStems}
                        selected={selected.has(song.name)}
                        onFav={handleFav}
                        onAnalyze={handleAnalyze}
                        onStems={handleStems}
                        onDelete={handleDelete}
                        onSelect={handleSelect}
                        onExpandStems={handleExpandStems}
                        onToggleSelect={handleToggleSelect}
                      />
                    )
                  })}
                  {(() => {
                    const items = virtualizer.getVirtualItems()
                    if (items.length === 0) return null
                    const bottomPad = virtualizer.getTotalSize() - items[items.length - 1].end
                    return bottomPad > 0 ? (
                      <tr style={{ border: 0 } as CSSProperties}>
                        <td
                          colSpan={8}
                          style={{ height: bottomPad, padding: 0, border: 0 } as CSSProperties}
                        />
                      </tr>
                    ) : null
                  })()}
                </tbody>
              )}
            </table>
          )}
        </div>

        <SongDrawer
          song={selectedSong}
          onClose={() => setSelectedSong(null)}
          onLoadDeck={handleLoadDeck}
        />
      </div>

      {/* Bulk action bar — fixed, slides up from bottom */}
      {selected.size > 0 && (
        <div className="la-bulk-bar">
          <span className="text-muted font-mono" style={{ fontSize: 'var(--text-xs)' }}>
            {selected.size} selected
          </span>
          <button className="la-bulk-btn" onClick={handleBulkAnalyze}>
            <BarChart2 size={12} /> Analyze
          </button>
          <button className="la-bulk-btn" onClick={handleBulkStems}>
            <Layers size={12} /> Split Stems
          </button>
          <button className="la-bulk-btn la-bulk-btn--ghost" onClick={() => setSelected(new Set())}>
            Clear
          </button>
        </div>
      )}
    </div>
  )
}
