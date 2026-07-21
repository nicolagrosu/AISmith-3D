import { Mesh2MotionEngine } from './Mesh2MotionEngine'

export class CustomModelUploadBootstrap {
  private readonly mesh2motion_engine: Mesh2MotionEngine

  constructor () {
    this.mesh2motion_engine = new Mesh2MotionEngine()
    this.load_aismith_source()
  }

  /**
   * AISmith passes the current artifact as a same-origin URL. Keeping this in
   * the bootstrap means the upstream editor keeps owning its interactive UI,
   * while our studio continues to own artifacts and pipeline handoff.
   */
  private load_aismith_source (): void {
    const source = new URLSearchParams(window.location.search).get('source')
    if (source === null || source.length === 0) return

    const extension = new URLSearchParams(window.location.search).get('extension')?.toLowerCase()
    if (extension !== 'glb') {
      console.error('AISmith Rig + Animate requires a GLB artifact')
      return
    }

    this.mesh2motion_engine.load_model_step.load_model_file(source, extension)
  }
}

// instantiate the class to setup event listeners
const app = new CustomModelUploadBootstrap()
