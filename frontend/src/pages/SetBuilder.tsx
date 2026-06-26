/* ============================================================
   AI RemixMate — Set Builder
   Sequence songs into a DJ set; energy arc view; chain remix.
   ============================================================ */
import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
} from '@dnd-kit/core'
import {
  SortableContext,
  verticalListSortingStrategy,
  useSortable,
  arrayMove,
  sortableKeyboardCoordinates,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import {
  ListMusic,
  Plus,
  Minus,
  Trash2,
  GitBranch,
  RefreshCw,
  Download,
  Loader2,
  Search,
  Shuffle,
  Settings,
  GripVertical,
  ArrowRight,
} from 'lucide-react'
import { libraryApi, remixApi, setlistApi } from '@/lib/api'
import { useAppStore } from '@/stores/appStore'
import { RemixControls, RemixOptions, REMIX_DEFAULTS } from '@/components/RemixControls'
import type { SongInfo } from '@/types'
import './PageBase.css'
import './SetBuilder.css'

interface SetEntry {
  id: string
  name: string
  bpm?: number
  key?: string
  camelot?: string
  energy?: number
}

function EnergyArc({ entries }: { entries: SetEntry[] }) {
  if (entries.length === 0) return null
  const w = 600
  const h = 80
  const pad = 20
  const usable = w - pad * 2
  const energies = entries.map((e) => (e.energy ?? 0.5))
  const pts = energies.map((e, i) => {
    const x = pad + (i / Math.max(entries.length - 1, 1)) * usable
    const y = h - pad - e * (h - pad * 2)
    return `${x},${y}`
  })
  const polyline = pts.join(' ')

  return (
    <div className="sb-arc">
      <span className="sb-arc__label text-muted">Energy arc</span>
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="sb-arc__svg">
        <defs>
          <linearGradient id="arc-grad" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="var(--color-amber-500)" stopOpacity="0.5" />
            <stop offset="100%" stopColor="var(--color-amber-500)" stopOpacity="0.05" />
          </linearGradient>
        </defs>
        {/* Fill area */}
        <polygon
          points={`${pad},${h} ${polyline} ${pad + usable},${h}`}
          fill="url(#arc-grad)"
        />
        {/* Line */}
        <polyline
          points={polyline}
          fill="none"
          stroke="var(--color-amber-500)"
          strokeWidth="2"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {/* Dots */}
        {pts.map((pt, i) => {
          const [x, y] = pt.split(',').map(Number)
          return <circle key={i} cx={x} cy={y} r="3" fill="var(--color-amber-500)" />
        })}
      </svg>
      <div className="sb-arc__labels">
        {entries.slice(0, 6).map((e) => (
          <span key={e.id} className="sb-arc__name text-muted">{e.name.slice(0, 12)}</span>
        ))}
        {entries.length > 6 && (
          <span className="sb-arc__name text-muted">+{entries.length - 6}</span>
        )}
      </div>
    </div>
  )
}

interface SetRowProps {
  entry: SetEntry
  index: number
  onRemove: (id: string) => void
}

function SetRow({ entry, index, onRemove }: SetRowProps) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({ id: entry.id })

  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.4 : 1,
  }

  const energyPct = Math.round((entry.energy ?? 0.5) * 100)
  const color =
    energyPct > 70 ? 'var(--color-crimson-500)' :
    energyPct > 40 ? 'var(--color-amber-500)' :
                     'var(--color-ice-400)'
  return (
    <div className="sb-row" ref={setNodeRef} style={style}>
      <span className="sb-drag-handle" {...attributes} {...listeners} title="Drag to reorder">
        <GripVertical size={14} />
      </span>
      <span className="sb-row__num font-mono text-muted">{index + 1}</span>
      <div className="sb-row__bar" style={{ background: color, width: `${energyPct}%` }} />
      <div className="sb-row__info">
        <span className="sb-row__name">{entry.name}</span>
        <div className="sb-row__meta">
          {entry.bpm && <span className="sb-chip sb-chip--amber">{entry.bpm.toFixed(0)}</span>}
          {entry.key && <span className="sb-chip sb-chip--ice">{entry.key}</span>}
          {entry.camelot && <span className="sb-chip sb-chip--violet">{entry.camelot}</span>}
        </div>
      </div>
      <div className="sb-row__actions">
        <button className="sb-icon-btn sb-icon-btn--danger" onClick={() => onRemove(entry.id)} title="Remove">
          <Minus size={12} />
        </button>
      </div>
    </div>
  )
}

export default function SetBuilder() {
  const navigate = useNavigate()
  const [set, setSet]           = useState<SetEntry[]>([])
  const [search, setSearch]     = useState('')
  const [chainJobId, setChainJobId]       = useState<string | null>(null)
  const [submitting, setSubmitting]       = useState(false)
  const [optimizing, setOptimizing]       = useState(false)
  const [error, setError]                 = useState<string | null>(null)
  const [chainOpts, setChainOpts]         = useState<RemixOptions>(REMIX_DEFAULTS)
  const [showSettings, setShowSettings]   = useState(false)

  const upsertJob = useAppStore((s) => s.upsertJob)
  const chainJob  = useAppStore((s) => (chainJobId ? s.jobs[chainJobId] : null))

  const { data: songs = [] } = useQuery<SongInfo[]>({
    queryKey: ['library-atlas'],
    queryFn: libraryApi.list,
    staleTime: 60_000,
  })

  const poolSongs = useMemo(() => {
    const inSet = new Set(set.map((e) => e.name))
    let list = songs.filter((s) => !inSet.has(s.name))
    if (search.trim()) {
      const q = search.toLowerCase()
      list = list.filter((s) => s.name.toLowerCase().includes(q))
    }
    return list.sort((a, b) => a.name.localeCompare(b.name))
  }, [songs, set, search])

  function addToSet(song: SongInfo) {
    setSet((prev) => [
      ...prev,
      {
        id: `${song.name}-${Date.now()}`,
        name: song.name,
        bpm: song.bpm,
        key: song.key,
        camelot: song.camelot,
        energy: song.energy,
      },
    ])
  }

  function removeFromSet(id: string) {
    setSet((prev) => prev.filter((e) => e.id !== id))
  }

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  )

  function handleDragEnd(event: DragEndEvent) {
    const { active, over } = event
    if (over && active.id !== over.id) {
      setSet((prev) => {
        const oldIndex = prev.findIndex((e) => e.id === active.id)
        const newIndex = prev.findIndex((e) => e.id === over.id)
        return arrayMove(prev, oldIndex, newIndex)
      })
    }
  }

  function clearSet() {
    if (set.length && !confirm('Clear the whole set?')) return
    setSet([])
  }

  function exportSet() {
    const text = set.map((e, i) => `${i + 1}. ${e.name}`).join('\n')
    const blob = new Blob([text], { type: 'text/plain' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href = url; a.download = 'set.txt'; a.click()
    URL.revokeObjectURL(url)
  }

  async function autoSequence() {
    if (set.length < 2 || optimizing) return
    setError(null)
    setOptimizing(true)
    try {
      const orderedNames = await setlistApi.optimize(set)
      setSet((prev) => {
        const byName = new Map(prev.map((e) => [e.name, e]))
        return orderedNames.flatMap((name) => {
          const entry = byName.get(name)
          return entry ? [entry] : []
        })
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Auto-sequence failed')
    } finally {
      setOptimizing(false)
    }
  }

  async function launchChain() {
    if (set.length < 2 || submitting) return
    setError(null)
    setSubmitting(true)
    try {
      const res = await remixApi.chain(set.map((e) => e.name), {
        transition_bars:       chainOpts.transition_bars,
        preset:                chainOpts.preset,
        transition_effect:     chainOpts.transition_effect,
        bridge_beat_mode:      chainOpts.bridge_beat_mode,
        bridge_beat_genre:     chainOpts.bridge_beat_genre,
        bridge_beat_intensity: chainOpts.bridge_beat_intensity,
      })
      setChainJobId(res.job_id)
      upsertJob({
        job_id: res.job_id, status: 'PENDING', type: 'dj_chain',
        progress: 0, message: `Chaining ${set.length} tracks…`,
        created_at: new Date().toISOString(), updated_at: new Date().toISOString(),
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Chain remix failed')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="page-base">
      <header className="page-base__header">
        <ListMusic size={20} strokeWidth={1.5} className="page-base__header-icon" />
        <div style={{ flex: 1 }}>
          <h1 className="page-base__title font-display">Set Builder</h1>
          <p className="page-base__sub text-muted">
            Sequence tracks · visualize energy arc · render a full chained mix
          </p>
        </div>
        {set.length >= 2 && (
          <div className="sb-header-actions">
            <button className="sb-btn sb-btn--ghost" onClick={exportSet}>
              <Download size={13} /> Export
            </button>
            <button className="sb-btn sb-btn--ghost" disabled={optimizing} onClick={autoSequence}>
              {optimizing ? <Loader2 size={13} className="sb-spin" /> : <Shuffle size={13} />}
              Auto-sequence
            </button>
            <button
              className={`sb-btn sb-btn--ghost ${showSettings ? 'sb-btn--ghost--active' : ''}`}
              onClick={() => setShowSettings((v) => !v)}
            >
              <Settings size={13} /> Settings
            </button>
            <button className="sb-btn sb-btn--primary" disabled={submitting} onClick={launchChain}>
              {submitting ? <Loader2 size={13} className="sb-spin" /> : <GitBranch size={13} />}
              Chain remix ({set.length} tracks)
            </button>
          </div>
        )}
      </header>

      {showSettings && set.length >= 2 && (
        <div className="sb-settings">
          <RemixControls value={chainOpts} onChange={setChainOpts} />
        </div>
      )}

      <div className="page-base__body sb-body">
        {/* Left: pool */}
        <aside className="sb-pool">
          <div className="sb-pool__header">
            <span className="sb-section-label">Song Pool</span>
            <span className="text-muted" style={{ fontSize: 'var(--text-xs)' }}>{poolSongs.length}</span>
          </div>
          <div className="sb-pool__search">
            <Search size={12} className="sb-pool__search-icon" />
            <input
              className="sb-search"
              placeholder="Filter pool…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <div className="sb-pool__list">
            {poolSongs.length === 0 ? (
              <div className="sb-pool__empty">
                <ListMusic size={22} strokeWidth={0.9} />
                <span className="text-muted">
                  {songs.length === 0 ? 'Library is empty — add songs first' : 'All available songs are already in the set'}
                </span>
                <button
                  className="sb-empty-action sb-empty-action--small"
                  onClick={() => navigate(songs.length === 0 ? '/operations' : '/library-atlas')}
                >
                  {songs.length === 0 ? 'Add songs in Operations' : 'Browse Library Atlas'} <ArrowRight size={12} />
                </button>
              </div>
            ) : (
              poolSongs.map((song) => (
                <button
                  key={song.name}
                  className="sb-pool__item"
                  onClick={() => addToSet(song)}
                  title={`Add "${song.name}" to set`}
                >
                  <span className="sb-pool__item-name">{song.name}</span>
                  <div className="sb-pool__item-meta">
                    {song.bpm && <span className="sb-chip sb-chip--amber">{song.bpm.toFixed(0)}</span>}
                    {song.camelot && <span className="sb-chip sb-chip--violet">{song.camelot}</span>}
                  </div>
                  <Plus size={13} className="sb-pool__item-add" />
                </button>
              ))
            )}
          </div>
        </aside>

        {/* Right: set + arc */}
        <div className="sb-right">
          {set.length > 1 && <EnergyArc entries={set} />}

          <div className="sb-set-header">
            <span className="sb-section-label">Set List</span>
            <div style={{ display: 'flex', gap: 'var(--space-2)', alignItems: 'center' }}>
              <span className="text-muted" style={{ fontSize: 'var(--text-xs)' }}>{set.length} tracks</span>
              {set.length > 0 && (
                <button className="sb-icon-btn sb-icon-btn--danger" onClick={clearSet} title="Clear set">
                  <Trash2 size={12} />
                </button>
              )}
            </div>
          </div>

          {set.length === 0 ? (
            <div className="sb-empty">
              <ListMusic size={36} strokeWidth={0.75} />
              <p className="font-display" style={{ fontSize: 'var(--text-xl)' }}>Your set is empty</p>
              <p className="text-muted" style={{ fontSize: 'var(--text-sm)' }}>Add tracks from the pool or browse the library first.</p>
              <button className="sb-empty-action" onClick={() => navigate(songs.length === 0 ? '/operations' : '/library-atlas')}>
                {songs.length === 0 ? 'Add songs in Operations' : 'Browse Library Atlas'} <ArrowRight size={13} />
              </button>
            </div>
          ) : (
            <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
              <SortableContext items={set.map((e) => e.id)} strategy={verticalListSortingStrategy}>
                <div className="sb-set-list">
                  {set.map((entry, i) => (
                    <SetRow
                      key={entry.id}
                      entry={entry}
                      index={i}
                      onRemove={removeFromSet}
                    />
                  ))}
                </div>
              </SortableContext>
            </DndContext>
          )}

          {error && <div className="sb-error">{error}</div>}

          {chainJob && (
            <div className="sb-chain-status">
              {chainJob.status === 'RUNNING' && <RefreshCw size={13} className="sb-spin" style={{ color: 'var(--color-amber-500)' }} />}
              <span className="font-mono" style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-secondary)' }}>
                {chainJob.status === 'RUNNING'
                  ? `${chainJob.progress}% — ${chainJob.message}`
                  : chainJob.status.toLowerCase()}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
