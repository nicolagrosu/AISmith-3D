# AISmith 3D — Local AI 3D Studio

AISmith 3D is a Windows-local studio for taking a reference image through 3D generation, AI-assisted refinement, texture painting, and browser-based rigging/animation. It keeps assets and inference on the local machine; large model files are deliberately downloaded from the app only when a user selects a workflow.

> Hackathon note: this repository contains the application and its source dependencies, but **does not include any model weights**. Use the in-app download controls after startup.

## What it does

| Workflow | Result | Prerequisites |
| --- | --- | --- |
| **Generate** | Image → geometry GLB through the native `trellis.cpp` CUDA worker | NVIDIA CUDA-capable GPU |
| **Refine** | Image + mesh → reconstructed, simplified, PBR-textured GLB with TRELLIS.2 FP8 | NVIDIA CUDA-capable GPU |
| **Rig + Animate** | Fit a skeleton, bind weights, preview/select animation, export GLB | Modern browser; no GPU or Blender required |
| **Texture Paint** | Paint a mesh in the browser, bake a base-color texture, export GLB | Blender 4.2+ |

The app runs its API locally at `127.0.0.1:8000` and its Vite UI locally at `127.0.0.1:5173`. It does not expose a ComfyUI server: the TRELLIS.2 refiner is invoked in an isolated local worker.

## Requirements

- Windows 10 or 11
- Python 3.11 or 3.12
- Node.js 20+ with npm
- For **Generate** and **Refine**: NVIDIA GPU, current NVIDIA driver, and CUDA-compatible PyTorch. CPU-only machines can still use the UI and Rig + Animate workflow, but inference buttons return a clear CUDA requirement error.
- For **Texture Paint** and Blender-assisted conversion: Blender 4.2 or later.

Disk space is required only when a model workflow is chosen in the UI: Generate downloads approximately 9.6 GB of geometry GGUFs; Refine downloads approximately 8.8 GB for its FP8 runtime and models. Keep additional working space for generated assets.

## Install and run

Clone the repository, then run the one-command launcher from PowerShell:

```powershell
git clone https://github.com/intisarGIT/AISmith-3D.git
cd AISmith-3D
.\start.ps1 -Install
```

This creates an isolated `.venv`, installs the Python and frontend dependencies, detects NVIDIA hardware, and starts the API and UI. Open [http://127.0.0.1:5173](http://127.0.0.1:5173).

To let the setup script install Blender through `winget` when it is missing:

```powershell
.\start.ps1 -Install -InstallBlender
```

The initial setup must be run in a PowerShell session permitted to create the virtual environment and install packages. Later launches need only:

```powershell
.\start.ps1
```

If a previous AISmith API is still listening on port 8000, replace it with:

```powershell
.\start.ps1 -RestartApi
```

### Model downloads

After the UI opens, use its **Download models** action in the Generate or Refine workflow. The project intentionally does not download model weights during installation. Downloads are stored below `studio-data/`, which is ignored by Git.

### Optional configuration

Copy `.env.example` to `.env` only when you need non-default paths:

```powershell
Copy-Item .env.example .env
```

| Variable | Purpose |
| --- | --- |
| `BLENDER_EXE` | Explicit Blender executable path |
| `TRELLIS_CPP_DIR` | Local `trellis.cpp` runtime directory |
| `TRELLIS_CPP_MODEL_DIR` | Geometry-model storage location |
| `AUTOREMESHER_EXE` | Optional AutoRemesher executable path |
| `PYTORCH_INDEX_URL` | Override the CUDA PyTorch package index |

## Verification

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m compileall -q backend
.\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v
npm.cmd --prefix frontend run build
```

## Architecture

`reference image → native trellis.cpp geometry → TRELLIS.2 FP8 refine / reconstruct / PBR → Mesh2Motion rig + animation`

Heavyweight inference is serialized through a coordinator lease. TRELLIS.2 runs in a short-lived process, Blender stages are headless and short-lived, and tabs pass artifact IDs/files rather than large meshes through browser memory.

## License and acknowledgements

AISmith 3D’s original application code is available under the [MIT License](LICENSE). It incorporates or integrates third-party projects under their own licenses; their license texts remain with the relevant vendored source. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution, source links, and licensing details.

In particular, thank you to the maintainers and contributors of [TRELLIS.2](https://github.com/microsoft/TRELLIS.2), [trellis.cpp](https://github.com/ilintar/trellis.cpp), [trellis2cpp](https://github.com/rms80/trellis2cpp), [ComfyUI](https://github.com/comfyanonymous/ComfyUI), [ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2), [AutoRemesher](https://github.com/huxingyi/autoremesher), and [Mesh2Motion](https://github.com/Mesh2Motion/mesh2motion-app).

Model weights are downloaded separately and may have terms different from this repository. Review the applicable Hugging Face model cards and upstream project licenses before redistributing models or outputs.

## Built with Codex

Codex/GPT-5.6 assisted with application architecture, native worker boundaries, GPU/VRAM policy, artifact API, UI, texture painter, retargeting layer, tests, and documentation. Retain the primary Codex task/session evidence with the hackathon submission materials if the event requires it.
