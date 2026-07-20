import { useEffect, useRef, useState } from 'react'
import { Box, Focus, Layers3, Maximize2, ScanLine, Upload } from 'lucide-react'
import '@google/model-viewer'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { ModelViewerElement } from '@google/model-viewer'

type ViewMode = 'scene' | 'solid' | 'wireframe'
type Props = { artifactId?: string; artifactKind?: string; artifactName?: string; onImportModel?: () => void }
const API = 'http://127.0.0.1:8000/api'

function WireframePreview({ src, onReady, onError }: { src: string; onReady: () => void; onError: () => void }) {
  const host = useRef<HTMLDivElement>(null)
  const onReadyRef = useRef(onReady)
  const onErrorRef = useRef(onError)
  useEffect(() => { onReadyRef.current = onReady; onErrorRef.current = onError }, [onReady, onError])

  useEffect(() => {
    const element = host.current
    if (!element) return
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 1000)
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setClearColor(0x0b0d11, 1)
    element.appendChild(renderer.domElement)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.07
    controls.target.set(0, 0, 0)
    const key = new THREE.DirectionalLight(0xffffff, 1.5)
    key.position.set(3, 4, 5)
    scene.add(key, new THREE.HemisphereLight(0xd7ddff, 0x151925, 1.6))
    const grid = new THREE.GridHelper(8, 16, 0x293047, 0x1d2231)
    grid.position.y = -1.2
    scene.add(grid)
    let frame = 0
    let disposed = false
    const resize = () => {
      const { width, height } = element.getBoundingClientRect()
      if (!width || !height) return
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setSize(width, height, false)
    }
    const observer = new ResizeObserver(resize)
    observer.observe(element)
    resize()
    new GLTFLoader().load(src, (gltf) => {
      if (disposed) return
      const lines = new THREE.Group()
      gltf.scene.updateMatrixWorld(true)
      gltf.scene.traverse((node) => {
        if (!(node instanceof THREE.Mesh) || !node.geometry) return
        const wire = new THREE.WireframeGeometry(node.geometry)
        const material = new THREE.LineBasicMaterial({ color: 0xa98cff, transparent: true, opacity: 0.94 })
        const segment = new THREE.LineSegments(wire, material)
        node.updateWorldMatrix(true, false)
        segment.applyMatrix4(node.matrixWorld)
        lines.add(segment)
      })
      if (!lines.children.length) { onErrorRef.current(); return }
      const bounds = new THREE.Box3().setFromObject(lines)
      const size = bounds.getSize(new THREE.Vector3())
      const center = bounds.getCenter(new THREE.Vector3())
      lines.position.sub(center)
      const radius = Math.max(size.length() * 0.58, 0.25)
      camera.position.set(radius * 1.25, radius * 0.9, radius * 1.65)
      camera.near = Math.max(radius / 100, 0.001)
      camera.far = radius * 100
      camera.updateProjectionMatrix()
      controls.target.set(0, 0, 0)
      controls.update()
      scene.add(lines)
      onReadyRef.current()
    }, undefined, () => onErrorRef.current())
    const render = () => { controls.update(); renderer.render(scene, camera); frame = requestAnimationFrame(render) }
    render()
    return () => {
      disposed = true
      cancelAnimationFrame(frame)
      observer.disconnect()
      controls.dispose()
      scene.traverse((child) => {
        if (child instanceof THREE.LineSegments) { child.geometry.dispose(); (child.material as THREE.Material).dispose() }
      })
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [src])

  return <div className="wireframe-preview" ref={host} />
}

export default function StudioViewport({ artifactId, artifactKind, artifactName, onImportModel }: Props) {
  const viewerRef = useRef<ModelViewerElement>(null)
  const [status, setStatus] = useState<'empty' | 'reference' | 'loading' | 'ready' | 'error'>(artifactKind === 'image' ? 'reference' : artifactId ? 'loading' : 'empty')
  const [mode, setMode] = useState<ViewMode>('scene')
  const isModel = artifactKind === 'glb'
  const src = artifactId ? `${API}/assets/${artifactId}/download` : ''

  useEffect(() => {
    setMode('scene')
    setStatus(artifactKind === 'image' ? 'reference' : isModel ? 'loading' : 'empty')
  }, [artifactId, artifactKind, isModel])

  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer || !isModel || mode === 'wireframe') return
    const ready = () => {
      if (mode === 'solid') {
        viewer.model?.materials.forEach((material) => {
          const pbr = material.pbrMetallicRoughness
          pbr.baseColorTexture?.setTexture(null)
          pbr.setBaseColorFactor('#8f78ff')
          pbr.setMetallicFactor(0.05)
          pbr.setRoughnessFactor(0.64)
        })
      }
      setStatus('ready')
    }
    const failed = () => setStatus('error')
    viewer.addEventListener('load', ready)
    viewer.addEventListener('error', failed)
    return () => { viewer.removeEventListener('load', ready); viewer.removeEventListener('error', failed) }
  }, [artifactId, isModel, mode])

  const changeMode = (next: ViewMode) => {
    setStatus('loading')
    setMode(next)
  }
  const resetCamera = () => {
    const viewer = viewerRef.current
    if (!viewer) return
    viewer.cameraOrbit = '35deg 75deg auto'
    viewer.fieldOfView = '30deg'
    viewer.jumpCameraToGoal()
  }
  const label = artifactName || (isModel ? 'Untitled model.glb' : 'Reference image')

  return <div className="viewport">
    {isModel && artifactId && mode !== 'wireframe' && <model-viewer
      key={`${artifactId}-${mode}`}
      ref={viewerRef}
      src={src}
      alt={label}
      camera-controls
      touch-action="pan-y"
      interaction-prompt="none"
      camera-orbit="35deg 75deg auto"
      field-of-view="30deg"
      min-camera-orbit="auto auto 60%"
      max-camera-orbit="auto auto 300%"
      environment-image="neutral"
      exposure="1.15"
      loading="eager"
      className="model-preview"
    />}
    {isModel && artifactId && mode === 'wireframe' && <WireframePreview src={src} onReady={() => setStatus('ready')} onError={() => setStatus('error')} />}

    {isModel && <div className="viewer-toolbar" aria-label="Viewer controls">
      <button className={mode === 'scene' ? 'active' : ''} onClick={() => changeMode('scene')} title="Textured scene"><Layers3 size={15} /> Scene</button>
      <button className={mode === 'solid' ? 'active' : ''} onClick={() => changeMode('solid')} title="Single-material solid view"><Box size={15} /> Solid</button>
      <button className={mode === 'wireframe' ? 'active' : ''} onClick={() => changeMode('wireframe')} title="Mesh wireframe"><ScanLine size={15} /> Wireframe</button>
      <span className="viewer-divider" />
      <button onClick={resetCamera} title="Reset camera"><Focus size={15} /> Reset view</button>
      {onImportModel && <button onClick={onImportModel} title="Import a GLB from your computer"><Upload size={15} /> Import GLB</button>}
    </div>}
    {!isModel && onImportModel && <button className="viewport-import" onClick={onImportModel}><Upload size={15} /> Import GLB</button>}
    {isModel && <div className="asset-chip"><Box size={14} /> <span>{label}</span>{status === 'loading' && <em>Loading…</em>}</div>}
    {status === 'reference' && artifactId && <img className="reference-preview" src={src} alt="Uploaded reference" />}
    {status === 'empty' && <div className="viewport-empty">Generate or import a GLB to preview it here.</div>}
    {status === 'error' && <div className="viewport-empty">Could not load this GLB. Try exporting it again or import another file.</div>}
    {status === 'ready' && <div className="viewport-hint">Drag to orbit · right-drag to pan · scroll to zoom</div>}
  </div>
}
