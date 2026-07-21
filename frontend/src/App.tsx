import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Activity, Box, Brush, ChevronRight, CircleHelp, Download, Film, FolderOpen, Gauge, ImagePlus, Layers3, LoaderCircle, MemoryStick, Paintbrush, Play, RefreshCw, RotateCcw, ScanLine, Settings2, Sparkles, Timer, Upload, UserRound, X } from 'lucide-react'
import StudioViewport from './StudioViewport'

const API = 'http://127.0.0.1:8000/api'
type Tab = 'generate' | 'refine' | 'retopologize' | 'rig' | 'paint'
type Artifact = { id: string; name: string; kind: string; size_bytes: number; metadata?: Record<string, unknown> }
type Capabilities = { gpu: { available: boolean; name: string; vram_total_mb: number; vram_used_mb: number; cuda: boolean; torch_cuda_available?: boolean; native_inference_ready?: boolean; model_variant: string; recommended_resolution: number }; ram: { total_mb: number; used_mb: number }; blender: { available: boolean }; models: Record<string, boolean>; diagnostics?: string[]; memory: { loaded_model: string | null }; output_directory?: string }
type Job = { id: string; kind: string; status: string; progress: number; message: string; logs: string[]; warnings: string[]; output_artifact_id?: string | null; error?: string | null }
type DownloadState = { id: string; model: string; status: string; progress: number; message: string; logs: string[]; error?: string | null; file_name?: string | null; file_index?: number; file_count?: number; file_bytes?: number; file_total_bytes?: number }
type PaintStroke = { x: number; y: number; radius: number; color: string; opacity: number; hardness: number }
type ModelDetails = { vertices: number; triangles: number; materials: number; meshes: number; format: string; size_bytes: number }

const tabs: { id: Tab; label: string; icon: typeof Sparkles; kicker: string }[] = [
  { id: 'generate', label: 'Generate', icon: Sparkles, kicker: 'Image to 3D' },
  { id: 'refine', label: 'Refine', icon: Sparkles, kicker: 'Image-guided reconstruction' },
  { id: 'retopologize', label: 'Retopology', icon: ScanLine, kicker: 'Clean topology' },
  { id: 'rig', label: 'Rig + Animate', icon: UserRound, kicker: 'Mesh2Motion workspace' },
  { id: 'paint', label: 'Texture Paint', icon: Paintbrush, kicker: 'Material studio' },
]

async function json<T>(url: string, init?: RequestInit): Promise<T> { const res = await fetch(`${API}${url}`, init); if (!res.ok) { const body = await res.json().catch(() => ({})); const detail = body.detail; const message = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map((item) => item.msg || 'Invalid request').join('; ') : `Request failed (${res.status})`; throw new Error(message) }; return res.json() }
async function upload(file: File): Promise<Artifact> { const form = new FormData(); form.append('file', file); return json<Artifact>('/assets/upload', { method: 'POST', body: form }) }

function formatMb(mb: number) { return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB` }
function formatDuration(milliseconds: number) { const seconds = Math.max(0, Math.floor(milliseconds / 1000)); return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}` }

export default function App() {
  const [tab, setTab] = useState<Tab>('generate')
  const [caps, setCaps] = useState<Capabilities | null>(null)
  const [source, setSource] = useState<Artifact | null>(null)
  const [output, setOutput] = useState<Artifact | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  // Default to 1024 (Fine mode); 512 maps to Coarse mode.
  const [resolution, setResolution] = useState<512 | 768 | 1024>(1024)
  const [seed, setSeed] = useState(0)
  const [refineTargetVertices, setRefineTargetVertices] = useState(25000)
  const [refineRepairHoles, setRefineRepairHoles] = useState(false)
  const [retopoTargetVertices, setRetopoTargetVertices] = useState(25000)
  const [retopoRepairHoles, setRetopoRepairHoles] = useState(false)
  const [gpuFirst, setGpuFirst] = useState(false)
  const [refineImage, setRefineImage] = useState<Artifact | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [paintColor, setPaintColor] = useState('#7c5cff')
  const [roughness, setRoughness] = useState(42)
  const [metallic, setMetallic] = useState(18)
  const [paintStrokes, setPaintStrokes] = useState<PaintStroke[]>([])
  const [brushSize, setBrushSize] = useState(9)
  const [brushOpacity, setBrushOpacity] = useState(90)
  const [brushHardness, setBrushHardness] = useState(80)
  const [downloads, setDownloads] = useState<DownloadState[]>([])
  const [modelDetails, setModelDetails] = useState<ModelDetails | null>(null)
  const [generationStartedAt, setGenerationStartedAt] = useState<number | null>(null)
  const [generationElapsed, setGenerationElapsed] = useState(0)
  const [generationTime, setGenerationTime] = useState<number | null>(null)
  const [generationPendingPreview, setGenerationPendingPreview] = useState(false)
  const [refineStartedAt, setRefineStartedAt] = useState<number | null>(null)
  const [refineElapsed, setRefineElapsed] = useState(0)
  const [refineTime, setRefineTime] = useState<number | null>(null)
  const [refinePendingPreview, setRefinePendingPreview] = useState(false)
  const [retopoStartedAt, setRetopoStartedAt] = useState<number | null>(null)
  const [retopoElapsed, setRetopoElapsed] = useState(0)
  const [retopoTime, setRetopoTime] = useState<number | null>(null)
  const [retopoPendingPreview, setRetopoPendingPreview] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const modelFileRef = useRef<HTMLInputElement>(null)
  const refineImageRef = useRef<HTMLInputElement>(null)
  const refineDefaultedAsset = useRef<string | null>(null)
  const retopoDefaultedAsset = useRef<string | null>(null)
  const displayAsset = output || source
  const activeSource = tab === 'generate' ? source : output || source
 
  const refreshCaps = () => json<Capabilities>('/system/capabilities').then(setCaps).catch(() => setError('Start the Python worker on port 8000 to connect the local studio.'))
  useEffect(() => { refreshCaps(); const timer = window.setInterval(refreshCaps, 5000); return () => window.clearInterval(timer) }, [])
  useEffect(() => { const active = downloads.filter((item) => ['queued', 'running'].includes(item.status)); if (!active.length) return; const timer = window.setInterval(() => { Promise.all(active.map((item) => json<DownloadState>(`/downloads/${item.id}`))).then((fresh) => { setDownloads((current) => current.map((item) => fresh.find((next) => next.id === item.id) || item)); if (fresh.some((item) => item.status === 'completed')) refreshCaps() }).catch(() => undefined) }, 700); return () => window.clearInterval(timer) }, [downloads])
  useEffect(() => { if (!job || !['queued', 'running'].includes(job.status)) return; const timer = window.setInterval(() => json<Job>(`/jobs/${job.id}`).then(setJob).catch(() => undefined), 500); return () => window.clearInterval(timer) }, [job])
  useEffect(() => { if (job?.status === 'completed' && job.output_artifact_id) { json<Artifact>(`/assets/${job.output_artifact_id}/download`, { method: 'HEAD' }).catch(() => undefined); setOutput({ id: job.output_artifact_id, name: `${job.kind} result.glb`, kind: 'glb', size_bytes: 0 }) } }, [job?.status])
  useEffect(() => { if (job?.kind === 'generate' && ['failed', 'cancelled'].includes(job?.status || '')) { setGenerationPendingPreview(false); setGenerationStartedAt(null) } }, [job?.kind, job?.status])
  useEffect(() => { if (!generationStartedAt || !generationPendingPreview) return; const timer = window.setInterval(() => setGenerationElapsed(Date.now() - generationStartedAt), 250); return () => window.clearInterval(timer) }, [generationStartedAt, generationPendingPreview])
  useEffect(() => { if (job?.kind === 'retopologize' && ['failed', 'cancelled'].includes(job?.status || '')) { setRefinePendingPreview(false); setRefineStartedAt(null) } }, [job?.kind, job?.status])
  useEffect(() => { if (!refineStartedAt || !refinePendingPreview) return; const timer = window.setInterval(() => setRefineElapsed(Date.now() - refineStartedAt), 250); return () => window.clearInterval(timer) }, [refineStartedAt, refinePendingPreview])
  useEffect(() => { if (job?.kind === 'retopo' && ['failed', 'cancelled'].includes(job?.status || '')) { setRetopoPendingPreview(false); setRetopoStartedAt(null) } }, [job?.kind, job?.status])
  useEffect(() => { if (!retopoStartedAt || !retopoPendingPreview) return; const timer = window.setInterval(() => setRetopoElapsed(Date.now() - retopoStartedAt), 250); return () => window.clearInterval(timer) }, [retopoStartedAt, retopoPendingPreview])
  useEffect(() => { if (!displayAsset || displayAsset.kind !== 'glb') { setModelDetails(null); return }; json<ModelDetails>(`/assets/${displayAsset.id}/details`).then(setModelDetails).catch(() => setModelDetails(null)) }, [displayAsset?.id, displayAsset?.kind])
  useEffect(() => { if (tab !== 'refine' || !activeSource || !modelDetails || displayAsset?.id !== activeSource.id || refineDefaultedAsset.current === activeSource.id) return; setRefineTargetVertices(modelDetails.vertices > 120000 ? 120000 : Math.max(500, Math.round(modelDetails.vertices * 0.5))); refineDefaultedAsset.current = activeSource.id }, [tab, activeSource?.id, displayAsset?.id, modelDetails?.vertices])
  useEffect(() => { if (tab !== 'retopologize' || !activeSource || !modelDetails || displayAsset?.id !== activeSource.id || retopoDefaultedAsset.current === activeSource.id) return; setRetopoTargetVertices(modelDetails.vertices > 120000 ? 120000 : Math.max(500, Math.round(modelDetails.vertices * 0.5))); retopoDefaultedAsset.current = activeSource.id }, [tab, activeSource?.id, displayAsset?.id, modelDetails?.vertices])
 
  const gpuPercent = useMemo(() => caps?.gpu.vram_total_mb ? Math.min(100, (caps.gpu.vram_used_mb / caps.gpu.vram_total_mb) * 100) : 0, [caps])
  const ramPercent = useMemo(() => caps?.ram?.total_mb ? Math.min(100, (caps.ram.used_mb / caps.ram.total_mb) * 100) : 0, [caps])
  const activeTab = tabs.find((item) => item.id === tab)!
  const requiredDownload = useMemo(() => {
    if (!caps || tab === 'paint' || tab === 'rig') return null
    if (tab === 'generate') {
      return { name: 'trellis-cpp', label: 'Generate model', ready: Boolean(caps.models.trellis_cpp_ready), detail: 'Download the model to generate 3D assets.' }
    }
    if (tab === 'refine') return { name: 'trellis2', label: 'Refine model', ready: Boolean(caps.models.trellis_refiner_ready), detail: 'Download the model to refine an asset.' }
    if (tab === 'retopologize') return { name: 'instant-meshes', label: 'Retopology engine', ready: Boolean(caps.models.instant_meshes_native), detail: 'Download the engine to retopologize an asset.' }
    return null
  }, [caps, tab])
  const chooseFile = async (file: File | undefined, target: 'input' | 'model' | 'refine-image' = 'input') => { if (!file) return; setError(''); try { const artifact = await upload(file); if (target === 'refine-image') setRefineImage(artifact); else if (target === 'input' && tab === 'generate') { setSource(artifact); setOutput(null) } else setOutput(artifact) } catch (e) { setError(e instanceof Error ? e.message : 'Upload failed') } }
  const runJob = async () => {
    setError(''); setBusy(true)
    try {
      const kind = tab === 'paint' ? 'paint-bake' : tab === 'refine' ? 'retopologize' : tab === 'retopologize' ? 'retopo' : tab
      const solverTargetVertices = tab === 'retopologize' ? Math.max(500, Math.round(retopoTargetVertices * 0.66)) : refineTargetVertices
      const created = await json<Job>('/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ kind, source_artifact_id: activeSource?.id, resolution, seed, target_vertices: solverTargetVertices, settings: { color: paintColor, roughness, metallic, strokes: paintStrokes, repair_holes: tab === 'retopologize' ? retopoRepairHoles : refineRepairHoles, gpu_first: gpuFirst, refine_image_artifact_id: refineImage?.id } }) })
      if (kind === 'generate') { const started = Date.now(); setGenerationStartedAt(started); setGenerationElapsed(0); setGenerationTime(null); setGenerationPendingPreview(true) }
      if (kind === 'retopologize') { const started = Date.now(); setRefineStartedAt(started); setRefineElapsed(0); setRefineTime(null); setRefinePendingPreview(true) }
      if (kind === 'retopo') { const started = Date.now(); setRetopoStartedAt(started); setRetopoElapsed(0); setRetopoTime(null); setRetopoPendingPreview(true) }
      setJob(created)
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not start job') } finally { setBusy(false) }
  }
  const clearMemory = async () => { try { await json('/system/clear-memory', { method: 'POST' }); refreshCaps() } catch (e) { setError(e instanceof Error ? e.message : 'Memory cleanup failed') } }
  const clearRam = async () => { try { await json('/system/clear-ram', { method: 'POST' }); refreshCaps() } catch (e) { setError(e instanceof Error ? e.message : 'RAM cleanup failed') } }
  const cancelTask = async () => { if (!job || !['queued', 'running'].includes(job.status)) return; try { setJob(await json<Job>(`/jobs/${job.id}/cancel`, { method: 'POST' })) } catch (e) { setError(e instanceof Error ? e.message : 'Could not cancel task') } }
  const openLocation = async () => { if (displayAsset) await json(`/assets/${displayAsset.id}/open-location`, { method: 'POST' }).catch((e) => setError(e.message)) }
  const chooseOutputLocation = async () => { try { await json('/system/output-location/select', { method: 'POST' }); refreshCaps() } catch (e) { setError(e instanceof Error ? e.message : 'Could not choose output location') } }
  const startDownload = async (requirement: { name: string; ready: boolean }) => { if (requirement.ready) return; setError(''); try { const state = await json<DownloadState>(`/models/${requirement.name}/download`, { method: 'POST' }); setDownloads((current) => current.some((item) => item.id === state.id) ? current : [...current, state]) } catch (e) { setError(e instanceof Error ? e.message : 'Could not start model download') } }
  useEffect(() => {
    const receiveMesh2MotionExport = async (event: MessageEvent<{ source?: string; type?: string; filename?: string; buffer?: ArrayBuffer }>) => {
      if (event.origin !== window.location.origin || event.data?.source !== 'aismith-mesh2motion' || event.data.type !== 'export-glb' || !(event.data.buffer instanceof ArrayBuffer)) return
      try {
        setError('')
        const artifact = await upload(new File([event.data.buffer], event.data.filename || 'rigged-animated-model.glb', { type: 'model/gltf-binary' }))
        setOutput(artifact)
      } catch (e) { setError(e instanceof Error ? e.message : 'Could not save the Mesh2Motion export') }
    }
    window.addEventListener('message', receiveMesh2MotionExport)
    return () => window.removeEventListener('message', receiveMesh2MotionExport)
  }, [])

  return <div className="app-shell">
    <header className="topbar">
      <div className="brand"><div className="brand-mark"><Box size={19} /></div><div><strong>AISMITH 3D</strong><span>LOCAL 3D STUDIO</span></div></div>
      <div className="top-tabs">{tabs.map(({ id, label, icon: Icon }) => <button className={`top-tab ${tab === id ? 'active' : ''}`} onClick={() => setTab(id)} key={id}><Icon size={15} />{label}</button>)}</div>
      <div className="top-actions"><div className="connection"><span className={`status-dot ${caps ? 'on' : ''}`} /> {caps ? 'Local worker ready' : 'Connecting'}</div><button className="icon-button" title="Help"><CircleHelp size={17} /></button><button className="avatar">AK</button></div>
    </header>
    <div className="subbar"><div className="crumb"><span className="crumb-muted">Workspace</span><ChevronRight size={13} /><span>{activeTab.label}</span>{displayAsset && <><ChevronRight size={13} /><span className="crumb-muted">{displayAsset.name}</span></>}</div><div className="subbar-right"><div className="memory"><MemoryStick size={14} /><span>RAM</span><strong>{caps ? `${formatMb(caps.ram.used_mb)} / ${formatMb(caps.ram.total_mb)}` : '—'}</strong><div className="memory-track"><i className={ramPercent > 80 ? 'danger' : ramPercent > 55 ? 'warn' : ''} style={{ width: `${Math.max(5, ramPercent)}%` }} /></div></div><button className="quiet-button" onClick={clearRam}><RefreshCw size={13} /> Clear RAM</button><div className="memory"><MemoryStick size={14} /><span>VRAM</span><strong>{caps ? `${formatMb(caps.gpu.vram_used_mb)} / ${formatMb(caps.gpu.vram_total_mb)}` : '—'}</strong><div className="memory-track"><i className={gpuPercent > 80 ? 'danger' : gpuPercent > 55 ? 'warn' : ''} style={{ width: `${Math.max(5, gpuPercent)}%` }} /></div></div><button className="quiet-button" onClick={clearMemory}><RefreshCw size={13} /> Clear VRAM</button></div></div>
    <input ref={fileRef} type="file" hidden accept={tab === 'generate' ? 'image/png,image/jpeg,image/webp' : tab === 'rig' ? '.glb,model/gltf-binary' : '.glb,.gltf,.fbx,.obj'} onChange={(e) => chooseFile(e.target.files?.[0], 'input')} />
    <main className={tab === 'rig' ? 'rig-workspace' : 'workspace'}>
      {tab === 'rig' ? <RigAnimateWorkspace source={activeSource} onUpload={() => fileRef.current?.click()} /> : <>
      <aside className="control-panel">
        <div className="panel-heading"><div><div className="eyebrow">{activeTab.kicker}</div><h1>{activeTab.label}</h1></div><button className="icon-button"><Settings2 size={17} /></button></div>
        <div className="panel-scroll">
          {tab === 'generate' && <GenerateControls resolution={resolution} setResolution={setResolution} seed={seed} setSeed={setSeed} caps={caps} source={source} outputDirectory={caps?.output_directory} onChooseOutput={chooseOutputLocation} onUpload={() => fileRef.current?.click()} />}
        {tab === 'refine' && <AiRefineRetopoControls targetVertices={refineTargetVertices} setTargetVertices={setRefineTargetVertices} repairHoles={refineRepairHoles} setRepairHoles={setRefineRepairHoles} gpuFirst={gpuFirst} setGpuFirst={setGpuFirst} suggestedVertices={modelDetails ? (modelDetails.vertices > 120000 ? 120000 : Math.max(500, Math.round(modelDetails.vertices * 0.5))) : null} source={activeSource} refineImage={refineImage} onAttachImage={() => refineImageRef.current?.click()} onUpload={() => fileRef.current?.click()} />}
        {tab === 'retopologize' && <InstantRetopoControls targetVertices={retopoTargetVertices} setTargetVertices={setRetopoTargetVertices} repairHoles={retopoRepairHoles} setRepairHoles={setRetopoRepairHoles} suggestedVertices={modelDetails ? (modelDetails.vertices > 120000 ? 120000 : Math.max(500, Math.round(modelDetails.vertices * 0.5))) : null} source={activeSource} onUpload={() => fileRef.current?.click()} />}
        {tab === 'paint' && <PaintControls color={paintColor} setColor={setPaintColor} roughness={roughness} setRoughness={setRoughness} metallic={metallic} setMetallic={setMetallic} brushSize={brushSize} setBrushSize={setBrushSize} brushOpacity={brushOpacity} setBrushOpacity={setBrushOpacity} brushHardness={brushHardness} setBrushHardness={setBrushHardness} strokes={paintStrokes} setStrokes={setPaintStrokes} source={activeSource} onUpload={() => fileRef.current?.click()} />}
          <input ref={modelFileRef} type="file" hidden accept=".glb,model/gltf-binary" onChange={(e) => chooseFile(e.target.files?.[0], 'model')} />
          <input ref={refineImageRef} type="file" hidden accept="image/png,image/jpeg,image/webp" onChange={(e) => chooseFile(e.target.files?.[0], 'refine-image')} />
        </div>
        <div className="panel-footer">
        {error && <div className="error-box"><X size={14} />{error}</div>}
        {!error && tab !== 'generate' && caps?.diagnostics?.[0] && <div className="info-line"><Settings2 size={14} /> {caps.diagnostics[0]}</div>}
        <button className="primary-action" onClick={runJob} disabled={busy || !activeSource}>{busy ? <LoaderCircle className="spin" size={17} /> : <Play size={17} fill="currentColor" />}{busy ? 'Starting…' : tab === 'generate' ? 'Generate 3D asset' : `Run ${activeTab.label}`}</button>
        {job && <div className={`job-status ${job.status}`}><div className="job-line"><Activity size={14} /><span>{job.message}</span><strong>{job.progress}%</strong></div><div className="job-track"><i style={{ width: `${job.progress}%` }} /></div>{job.warnings?.[0] && <small>{job.warnings[0]}</small>}</div>}
        </div>
      </aside>
      <section className="canvas-area"><StudioViewport artifactId={displayAsset?.id} artifactKind={displayAsset?.kind} artifactName={displayAsset?.name} onImportModel={() => modelFileRef.current?.click()} onModelLoaded={() => { if (generationPendingPreview && generationStartedAt) { const elapsed = Date.now() - generationStartedAt; setGenerationElapsed(elapsed); setGenerationTime(elapsed); setGenerationPendingPreview(false); setGenerationStartedAt(null) }; if (refinePendingPreview && refineStartedAt) { const elapsed = Date.now() - refineStartedAt; setRefineElapsed(elapsed); setRefineTime(elapsed); setRefinePendingPreview(false); setRefineStartedAt(null) }; if (retopoPendingPreview && retopoStartedAt) { const elapsed = Date.now() - retopoStartedAt; setRetopoElapsed(elapsed); setRetopoTime(elapsed); setRetopoPendingPreview(false); setRetopoStartedAt(null) } }} />{generationPendingPreview && <div className="generation-blocker"><div><Timer size={38} /><span>Generating 3D model</span><strong>{formatDuration(generationElapsed)}</strong><button className="secondary-button cancel-task" onClick={cancelTask}>Cancel task</button></div></div>}{refinePendingPreview && <div className="generation-blocker"><div><Timer size={38} /><span>Refining 3D model</span><strong>{formatDuration(refineElapsed)}</strong><button className="secondary-button cancel-task" onClick={cancelTask}>Cancel task</button></div></div>}{retopoPendingPreview && <div className="generation-blocker"><div><Timer size={38} /><span>Retopologizing 3D model</span><strong>{formatDuration(retopoElapsed)}</strong><button className="secondary-button cancel-task" onClick={cancelTask}>Cancel task</button></div></div>}<div className="canvas-footer"><div><span className="eyebrow">CURRENT OUTPUT</span><strong>{displayAsset ? displayAsset.name : 'No asset loaded'}</strong></div><div className="footer-actions">{generationTime !== null && <span className="generation-time">Generation time: {formatDuration(generationTime)}</span>}{refineTime !== null && <span className="generation-time">Refine time: {formatDuration(refineTime)}</span>}{retopoTime !== null && <span className="generation-time">Retopology time: {formatDuration(retopoTime)}</span>}{displayAsset && <><button className="secondary-button" onClick={openLocation}><FolderOpen size={14} /> Open location</button>{displayAsset.kind === 'glb' && <a className="download-button" href={`${API}/assets/${displayAsset.id}/download`}><Download size={14} /> Export GLB</a>}</>}</div></div></section>
      <AssetInspector asset={displayAsset} details={modelDetails} requirement={requiredDownload} downloads={downloads} onDownload={() => requiredDownload && startDownload(requiredDownload)} />
      </>}
    </main>
  </div>
}

function ModelRequirement({ requirement, onDownload, downloading }: { requirement: { name: string; label: string; ready: boolean; detail: string }; onDownload: () => void; downloading: boolean; loaded: boolean }) { const state = downloading ? 'Downloading' : requirement.ready ? 'Ready' : 'Setup required'; const button = requirement.name === 'instant-meshes' ? 'Download Instant Meshes' : 'Download model'; return <div className={`model-requirement ${requirement.ready ? 'ready' : ''}`}><div className="requirement-top"><div><div className="eyebrow">MODEL STATUS</div><strong>{requirement.label}</strong></div><span className="requirement-state">{state}</span></div><p>{requirement.ready ? 'Ready to create.' : requirement.detail}</p>{!requirement.ready && <button className="download-model-button" onClick={onDownload} disabled={downloading}><Download size={14} /> {downloading ? 'Downloading…' : button}</button>}</div> }
function DownloadDock({ downloads }: { downloads: DownloadState[] }) { const visible = downloads.filter((item) => ['queued', 'running', 'completed', 'failed'].includes(item.status)); if (!visible.length) return null; return <div className="download-dock"><div className="download-dock-title"><Download size={14} /><strong>Local downloads</strong><span>{visible.filter((item) => ['queued', 'running'].includes(item.status)).length || ' '}</span></div>{visible.slice(-3).map((item) => { const hasFile = Boolean(item.file_name); const fileProgress = item.file_total_bytes ? `${formatMb((item.file_bytes || 0) / 1024 / 1024)} / ${formatMb(item.file_total_bytes / 1024 / 1024)}` : ''; const label = item.model === 'trellis-cpp' ? 'C++ geometry models' : item.model === 'trellis2' || item.model === 'trellis-refiner' ? 'TRELLIS.2 FP8 Refine models' : item.model === 'instant-meshes' ? 'Instant Meshes engine' : 'Local model'; return <div className="download-item" key={item.id}><div className="download-item-line"><span>{label}</span><strong>{item.status === 'completed' ? 'Ready' : item.status === 'failed' ? 'Failed' : `${item.progress}%`}</strong></div><div className="download-track"><i className={item.status === 'failed' ? 'failed' : item.status === 'completed' ? 'complete' : ''} style={{ width: `${Math.max(4, item.progress)}%` }} /></div><small>{item.error || (hasFile ? `(${item.file_index || 0}/${item.file_count || 10}) ${item.file_name} · ${fileProgress}` : item.message)}</small></div> })}</div> }

function UploadCard({ source, onUpload, label = 'Upload model', hint = 'GLB, GLTF, FBX or OBJ · max 20 MB' }: { source: Artifact | null; onUpload: () => void; label?: string; hint?: string }) { return <button className={`upload-card ${source ? 'has-file' : ''}`} onClick={onUpload}>{source ? <><div className="upload-preview"><Box size={22} /></div><div><strong>{source.name}</strong><span>Replace source</span></div></> : <><div className="upload-icon"><Upload size={19} /></div><div><strong>{label}</strong><span>{hint}</span></div></>}</button> }
function SectionLabel({ children }: { children: ReactNode }) { return <div className="section-label">{children}</div> }
function DownloadProgress({ download }: { download: DownloadState }) { const bytes = download.file_total_bytes ? `${formatMb((download.file_bytes || 0) / 1024 / 1024)} / ${formatMb(download.file_total_bytes / 1024 / 1024)}` : ''; return <div className="right-download-progress"><div className="download-item-line"><span>Downloading</span><strong>{download.progress}%</strong></div><div className="download-track"><i style={{ width: `${Math.max(4, download.progress)}%` }} /></div><small>{download.file_name ? `${download.file_name} · ${bytes}` : download.message}</small></div> }
function AssetInspector({ asset, details, requirement, downloads, onDownload }: { asset: Artifact | null; details: ModelDetails | null; requirement: { name: string; label: string; ready: boolean; detail: string } | null; downloads: DownloadState[]; onDownload: () => void }) { const value = (number?: number) => number === undefined ? '—' : number.toLocaleString(); const activeDownload = requirement ? downloads.find((item) => item.model === requirement.name && ['queued', 'running'].includes(item.status)) : undefined; return <aside className="inspector"><div className="inspector-body inspector-main"><div className="asset-icon"><Box size={23} /></div><h2>{asset ? asset.name.replace(/\.glb$/i, '') : 'Untitled asset'}</h2><p className="muted">{asset ? details ? 'Read from the loaded model' : 'Reading model details…' : 'Generate or import a GLB to inspect it here.'}</p><div className="spec-grid"><div><span>Vertices</span><strong>{value(details?.vertices)}</strong></div><div><span>Triangles</span><strong>{value(details?.triangles)}</strong></div><div><span>Materials</span><strong>{value(details?.materials)}</strong></div><div><span>Meshes</span><strong>{value(details?.meshes)}</strong></div><div><span>Format</span><strong>{details?.format || '—'}</strong></div><div><span>File size</span><strong>{details ? formatMb(details.size_bytes / 1024 / 1024) : '—'}</strong></div></div></div>{activeDownload && <div className="inspector-download-area"><DownloadProgress download={activeDownload} /></div>}{requirement && <div className="right-download-slot"><ModelRequirement requirement={requirement} onDownload={onDownload} downloading={false} loaded={false} /></div>}</aside> }
function GenerateControls({ resolution, setResolution, seed, setSeed, source, outputDirectory, onChooseOutput, onUpload }: { resolution: 512 | 768 | 1024; setResolution: (v: 512 | 768 | 1024) => void; seed: number; setSeed: (v: number) => void; caps: Capabilities | null; source: Artifact | null; outputDirectory?: string; onChooseOutput: () => void; onUpload: () => void }) { return <><div className="mode-switch"><button className="selected"><ImagePlus size={14} /> Image to 3D</button><button disabled><Layers3 size={14} /> Multi-view <span>soon</span></button></div><SectionLabel>Source image</SectionLabel><UploadCard source={source} onUpload={onUpload} label="Upload reference image" /><SectionLabel>Output folder</SectionLabel><div className="output-location"><span title={outputDirectory}>{outputDirectory || 'AISmith 3D artifact cache'}</span><button onClick={onChooseOutput}>Choose…</button></div><div className="option-row"><label>Detail level</label><div className="segmented"><button className={resolution === 512 ? 'selected' : ''} onClick={() => setResolution(512)}>Fast</button><button className={resolution === 1024 || resolution === 768 ? 'selected' : ''} onClick={() => setResolution(1024)}>High detail</button></div></div><div className="option-row"><label>Seed</label><input type="number" min="0" max="2147483647" value={seed} onChange={(event) => setSeed(Math.max(0, Math.min(2147483647, Number(event.target.value) || 0)))} /></div><div className="info-line"><Gauge size={14} /> Same seed, same result.</div></> }
function RetopoControls({ targetVertices, setTargetVertices, repairHoles, setRepairHoles, method, setMethod, source, onUpload }: { targetVertices: number; setTargetVertices: (v: number) => void; repairHoles: boolean; setRepairHoles: (value: boolean) => void; method: 'instant-meshes' | 'ai-refine'; setMethod: (value: 'instant-meshes' | 'ai-refine') => void; source: Artifact | null; onUpload: () => void }) { return <><SectionLabel>Source model</SectionLabel><UploadCard source={source} onUpload={onUpload} /><SectionLabel>Method</SectionLabel><div className="segmented"><button className={method === 'instant-meshes' ? 'selected' : ''} onClick={() => setMethod('instant-meshes')}>Instant Meshes</button><button className={method === 'ai-refine' ? 'selected' : ''} onClick={() => setMethod('ai-refine')}>AI Refine</button></div>{method === 'ai-refine' && <div className="info-line"><Sparkles size={14} /> Uses the original Generate image to restore geometry before extracting the requested quad topology.</div>}<SectionLabel>{method === 'ai-refine' ? 'AI Refine + Instant Meshes' : 'Instant Meshes configuration'}</SectionLabel><div className="retopo-options"><div><span>Remesh as</span><strong>Quad-dominant (4/4)</strong></div><div><span>Extrinsic</span><strong>Enabled</strong></div><div><span>Align to boundaries</span><strong>Enabled</strong></div><div><span>Smoothing</span><strong>0 iterations</strong></div></div><div className="range-row"><label>Target vertex count <strong>{targetVertices.toLocaleString()}</strong></label><input type="range" min="500" max={Math.max(200000, targetVertices)} step="500" value={targetVertices} onChange={(e) => setTargetVertices(Number(e.target.value))} /></div><div className="retopo-target-note">The resulting vertex count may vary slightly after the full export process.</div>{method === 'instant-meshes' && <label className={`retopo-repair-toggle ${repairHoles ? 'selected' : ''}`}><input type="checkbox" checked={repairHoles} onChange={(event) => setRepairHoles(event.target.checked)} /><span><strong>Repair holes</strong><small>Use MeshFix after retopology to repair holes and self-intersections. Repair patches may contain triangles.</small></span></label>}<div className="retopo-summary"><span>Solver stages</span><strong>{method === 'ai-refine' ? 'Image refine → quad reconstruction → extract' : 'Orientation → Position → Extract'}</strong></div><div className="info-line"><ScanLine size={14} /> Blender retains the untouched source GLB, unwraps the new quad mesh, and bakes the source appearance into the exported GLB.</div></> }

function InstantRetopoControls({ targetVertices, setTargetVertices, repairHoles, setRepairHoles, suggestedVertices, source, onUpload }: { targetVertices: number; setTargetVertices: (v: number) => void; repairHoles: boolean; setRepairHoles: (value: boolean) => void; suggestedVertices: number | null; source: Artifact | null; onUpload: () => void }) {
  return <>
    <SectionLabel>Source model</SectionLabel><UploadCard source={source} onUpload={onUpload} />
    <SectionLabel>Topology</SectionLabel><div className="retopo-options"><div><span>Output</span><strong>Quad-dominant</strong></div><div><span>Boundaries</span><strong>Preserve</strong></div></div>
    <div className="range-row"><label>Target vertex count <strong>{targetVertices.toLocaleString()}</strong></label><input type="range" min="500" max="1000000" step="500" value={targetVertices} onChange={(event) => setTargetVertices(Number(event.target.value))} /></div>
    {suggestedVertices !== null && <div className="retopo-target-note">Suggested for this model: {suggestedVertices.toLocaleString()}. Feel free to enter your desired value.</div>}
    <label className={`retopo-repair-toggle ${repairHoles ? 'selected' : ''}`}><input type="checkbox" checked={repairHoles} onChange={(event) => setRepairHoles(event.target.checked)} /><span><strong>Repair holes</strong><small>May add triangle patches.</small></span></label>
  </>
}

function AiRefineRetopoControls({ targetVertices, setTargetVertices, repairHoles, setRepairHoles, gpuFirst, setGpuFirst, suggestedVertices, source, refineImage, onAttachImage, onUpload }: { targetVertices: number; setTargetVertices: (v: number) => void; repairHoles: boolean; setRepairHoles: (value: boolean) => void; gpuFirst: boolean; setGpuFirst: (value: boolean) => void; suggestedVertices: number | null; source: Artifact | null; refineImage: Artifact | null; onAttachImage: () => void; onUpload: () => void }) {
  return <>
    <SectionLabel>Source model</SectionLabel><UploadCard source={source} onUpload={onUpload} />
    <SectionLabel>Reference image</SectionLabel><UploadCard source={refineImage} onUpload={onAttachImage} label="Attach reference image" hint="PNG, JPG or WebP · optional" /><div className="info-line"><Sparkles size={14} /> Required for imported models.</div>
    <SectionLabel>Refinement</SectionLabel>
    <div className="retopo-options"><div><span>Quality</span><strong>High</strong></div><div><span>Texture</span><strong>New PBR</strong></div></div>
    <div className="range-row"><label>Target vertex count <strong>{targetVertices.toLocaleString()}</strong></label><input type="range" min="500" max="1000000" step="500" value={Math.min(1_000_000, targetVertices)} onChange={(e) => setTargetVertices(Number(e.target.value))} /></div>
    {suggestedVertices !== null && <div className="retopo-target-note">Suggested for this model: {suggestedVertices.toLocaleString()}. Feel free to enter your desired value.</div>}
    <div className="retopo-target-note">The resulting vertex count may vary slightly after the full export process.</div>
    <label className={`retopo-repair-toggle ${repairHoles ? 'selected' : ''}`}><input type="checkbox" checked={repairHoles} onChange={(event) => setRepairHoles(event.target.checked)} /><span><strong>Fill holes</strong><small>May close intentional openings.</small></span></label>
    <label className={`retopo-repair-toggle ${gpuFirst ? 'selected' : ''}`}><input type="checkbox" checked={gpuFirst} onChange={(event) => setGpuFirst(event.target.checked)} /><span><strong>GPU-first (experimental)</strong><small>Faster if it fits. May run out of memory.</small></span></label>
    <div className="retopo-summary"><span>Output</span><strong>Refined mesh with PBR texture</strong></div>
  </>
}
function RigAnimateWorkspace({ source, onUpload }: { source: Artifact | null; onUpload: () => void }) {
  if (!source || source.kind !== 'glb') return <section className="rig-empty"><div><UserRound size={34} /><h1>Rig + Animate</h1><p>Choose a GLB from the studio or upload one to begin the Mesh2Motion workflow.</p><button className="primary-action" onClick={onUpload}><Upload size={17} /> Upload GLB</button></div></section>
  const sourceUrl = `${API}/assets/${source.id}/download`
  const workspaceUrl = `/mesh2motion.html?source=${encodeURIComponent(sourceUrl)}&extension=glb`
  return <iframe key={source.id} className="mesh2motion-frame" title="Mesh2Motion Rig and Animate workspace" src={workspaceUrl} />
}
function PaintControls({ color, setColor, roughness, setRoughness, metallic, setMetallic, brushSize, setBrushSize, brushOpacity, setBrushOpacity, brushHardness, setBrushHardness, strokes, setStrokes, source, onUpload }: { color: string; setColor: (v: string) => void; roughness: number; setRoughness: (v: number) => void; metallic: number; setMetallic: (v: number) => void; brushSize: number; setBrushSize: (v: number) => void; brushOpacity: number; setBrushOpacity: (v: number) => void; brushHardness: number; setBrushHardness: (v: number) => void; strokes: PaintStroke[]; setStrokes: (v: PaintStroke[]) => void; source: Artifact | null; onUpload: () => void }) { return <><SectionLabel>Source model</SectionLabel><UploadCard source={source} onUpload={onUpload} /><SectionLabel>Material</SectionLabel><div className="color-row"><input type="color" value={color} onChange={(e) => setColor(e.target.value)} /><div><strong>Base color</strong><span>{color.toUpperCase()}</span></div></div><div className="range-row"><label>Roughness <strong>{roughness}%</strong></label><input type="range" min="0" max="100" value={roughness} onChange={(e) => setRoughness(Number(e.target.value))} /></div><div className="range-row"><label>Metallic <strong>{metallic}%</strong></label><input type="range" min="0" max="100" value={metallic} onChange={(e) => setMetallic(Number(e.target.value))} /></div><SectionLabel>Brush mask</SectionLabel><PaintCanvas color={color} size={brushSize} opacity={brushOpacity} hardness={brushHardness} strokes={strokes} setStrokes={setStrokes} /><div className="range-row"><label>Size <strong>{brushSize}%</strong></label><input type="range" min="2" max="30" value={brushSize} onChange={(e) => setBrushSize(Number(e.target.value))} /></div><div className="range-row"><label>Opacity <strong>{brushOpacity}%</strong></label><input type="range" min="1" max="100" value={brushOpacity} onChange={(e) => setBrushOpacity(Number(e.target.value))} /></div><div className="range-row"><label>Hardness <strong>{brushHardness}%</strong></label><input type="range" min="1" max="100" value={brushHardness} onChange={(e) => setBrushHardness(Number(e.target.value))} /></div><div className="brush-row"><button className="brush selected"><Brush size={15} /></button><button className="brush" onClick={() => setStrokes([])}><RotateCcw size={15} /></button><span>{strokes.length ? `${strokes.length} stroke${strokes.length === 1 ? '' : 's'}` : 'Basic mask'}</span></div><div className="info-line"><Paintbrush size={14} /> Paint strokes are baked into a packed GLB texture</div></> }

function PaintCanvas({ color, size, opacity, hardness, strokes, setStrokes }: { color: string; size: number; opacity: number; hardness: number; strokes: PaintStroke[]; setStrokes: (v: PaintStroke[]) => void }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const painting = useRef(false)
  const draw = (event: React.PointerEvent<HTMLCanvasElement>) => { const canvas = ref.current; if (!canvas) return; const rect = canvas.getBoundingClientRect(); setStrokes([...strokes, { x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)), y: Math.max(0, Math.min(1, 1 - (event.clientY - rect.top) / rect.height)), radius: size / 100, color, opacity: opacity / 100, hardness: hardness / 100 }]) }
  useEffect(() => { const canvas = ref.current; if (!canvas) return; const ctx = canvas.getContext('2d')!; ctx.fillStyle = '#24232b'; ctx.fillRect(0, 0, canvas.width, canvas.height); ctx.strokeStyle = '#3b3945'; ctx.lineWidth = 1; for (let i = 1; i < 8; i++) { ctx.beginPath(); ctx.moveTo(i * canvas.width / 8, 0); ctx.lineTo(i * canvas.width / 8, canvas.height); ctx.stroke(); ctx.beginPath(); ctx.moveTo(0, i * canvas.height / 8); ctx.lineTo(canvas.width, i * canvas.height / 8); ctx.stroke() } strokes.forEach((stroke) => { ctx.globalAlpha = stroke.opacity; ctx.fillStyle = stroke.color; ctx.beginPath(); ctx.arc(stroke.x * canvas.width, (1 - stroke.y) * canvas.height, stroke.radius * canvas.width, 0, Math.PI * 2); ctx.fill() }); ctx.globalAlpha = 1 }, [strokes])
  return <canvas ref={ref} width={280} height={150} className="paint-canvas" onPointerDown={(event) => { painting.current = true; event.currentTarget.setPointerCapture(event.pointerId); draw(event) }} onPointerMove={(event) => { if (painting.current) draw(event) }} onPointerUp={() => { painting.current = false }} onPointerCancel={() => { painting.current = false }} />
}
