# AISmith 3D — Local 3D Generator Studio

AISmith 3D is a Windows-local React/FastAPI studio for the complete pipeline:

The active hybrid workflow is: single image -> trellis.cpp geometry GLB -> TRELLIS.2 FP8 refine/reconstruct/PBR -> SkinToken -> Mesh2Motion animation.

`single image → TRELLIS.2 FP8 → PBR GLB → TRELLIS.2 refine/reconstruct → SkinToken → Mesh2Motion animation`

The application does not use ComfyUI as its frontend. Generate uses the bundled trellis.cpp CUDA executable for the alpha-aware C++ workflow. Refine invokes the TRELLIS.2 FP8 Python nodes directly in a short-lived, isolated worker for mesh-conditioned refinement, reconstruction, and PBR texturing without exposing a ComfyUI server.

## What is implemented

- Generate: standalone trellis.cpp CUDA geometry worker with alpha-aware transparent 1024 input padding, 512/1024 selection, and `--no-texture`. The C++ texture GGUFs are intentionally not downloaded.
- Refine: image + mesh-conditioned TRELLIS.2 refinement, CuMesh surface reconstruction, target-face simplification, optional Meshlib hole repair, then TRELLIS.2 PBR texturing on the final mesh. No Instant Meshes or Blender source-texture bake participates in this path.
- Auto Rig: direct SkinToken model loader/generator, headless Blender path, articulated bone convention, and GLB output. No ComfyUI workflow is required.
- Texture Paint: browser brush canvas plus material controls; Blender creates UVs when necessary, bakes UV-space strokes to a packed base-color texture, and exports a GLB.
- Animate: Mesh2Motion's bundled human animation GLB is used as the clip library. Idle/Walk/Run/Wave actions are mapped onto the SkinToken articulated skeleton; incompatible skeletons fail instead of exporting corrupted animation.
- Memory: only one heavyweight job owns the coordinator lease. Trellis2 and SkinToken run in separate processes, and every Blender stage is short-lived. The API also clears Python and CUDA caches.
- Artifact handoff: tabs pass artifact IDs and files on disk, not large meshes through browser memory.

## Requirements

For image-to-3D and refinement: Windows 10/11 and an NVIDIA GPU with current drivers. Generate downloads roughly 9.6 GB of trellis.cpp geometry GGUFs. Refine downloads roughly 8.8 GB for `visualbruno/TRELLIS.2-4B-FP8`, DINOv3, and its required sparse decoder. The formats are different and cannot share weights, so the full hybrid install is roughly 18.4 GB. The C++ Generate worker runs at 512 or 1024; Refine is fixed at its 1024 high-quality workflow.

Blender 4.2+ is required for Texture Paint, SkinToken, and animation. The setup script detects it and uses winget to install it when available.

The current development machine has no NVIDIA GPU and CPU-only PyTorch. Therefore the native Trellis2/SkinToken inference paths are implemented but cannot be executed on this machine; they return a clear CUDA setup error. Blender texture baking and Mesh2Motion retargeting have been exercised locally.

## Run on Windows

```powershell
.\start.ps1 -Install
```

`-Install` creates `.venv` inside the project. The launcher always starts the API with that interpreter, detects NVIDIA hardware with `nvidia-smi`, and installs the Python dependencies needed for the studio. Generate downloads its C++ geometry-only bundle on demand. Refine installs its matching CUDA extensions and downloads the FP8 model set separately on demand. Large model downloads are runtime-gated and shown in the bottom-right download dock.

If an older AISmith 3D API is already listening on port 8000, use `./start.ps1 -RestartApi` to replace it with the dedicated-environment API. The launcher checks for FastAPI, Uvicorn, and PyTorch inside `.venv`; an incomplete environment is repaired automatically.

Then open [http://127.0.0.1:5173](http://127.0.0.1:5173).

Copy `.env.example` to `.env` or set the variables in the process environment. Important paths are:

```text
TRELLIS_REFINER_DIR=studio-data/tools/trellis2-refiner
SKINTOKEN_REPO_DIR=vendor/ComfyUI-SkinTokens
SKINTOKEN_MODELS_DIR=studio-data/models/skintoken
MESH2MOTION_LIBRARY=vendor/mesh2motion-app/static/animations/human-base-animations.glb
BLENDER_EXE=C:/Program Files/Blender Foundation/Blender 5.1/blender.exe
```

The Generate tab downloads the alpha-aware C++ geometry bundle when requested. The Refine tab downloads its isolated FP8 CUDA runtime and image+mesh refinement/PBR weights when requested. Auto Rig downloads SkinToken checkpoints only when requested.

## API

- `GET /api/system/capabilities`
- `POST /api/assets/upload`
- `POST /api/jobs`
- `GET /api/jobs/{id}`
- `POST /api/models/{name}/unload`
- `POST /api/system/clear-memory`
- `GET /api/assets/{id}/download`
- `POST /api/assets/{id}/open-location`
## Verification

```powershell
python -m compileall -q backend
python -m unittest discover -s backend/tests -v
npm.cmd --prefix frontend run build
```

## Hackathon / Codex record

Codex/GPT-5.6 was used to implement the application architecture, native worker boundaries, GPU/VRAM policy, artifact API, UI, texture painter, retargeting layer, tests, and documentation. Keep the primary Codex task/session evidence with the submission materials as required by the hackathon rules.
