# AISmith 3D — Local 3D Generator Studio

AISmith 3D is a Windows-local React/FastAPI studio for the complete pipeline:

`single image → trellis.cpp GGUF → GLB → AutoRemesher → SkinToken → Mesh2Motion animation`

The application does not use ComfyUI as its frontend or runtime. Image-to-3D runs through the standalone CUDA build of `pwilkin/trellis.cpp`; it owns background removal, model loading, mesh extraction, texture baking, and GLB export. The remaining stages use AutoRemesher, SkinToken, and short-lived headless Blender processes.

## What is implemented

- Generate: native `trellis-cli.exe` with the complete `ilintar/trellis2-gguf` model set, BiRefNet background removal, 512/1024 selection, UV-textured GLB export, and process exit for VRAM release.
- Retopologize: GLB/GLTF/FBX/OBJ conversion to AutoRemesher's OBJ CLI boundary, target quad count and quality settings, then conversion back to GLB.
- Auto Rig: direct SkinToken model loader/generator, headless Blender path, articulated bone convention, and GLB output. No ComfyUI workflow is required.
- Texture Paint: browser brush canvas plus material controls; Blender creates UVs when necessary, bakes UV-space strokes to a packed base-color texture, and exports a GLB.
- Animate: Mesh2Motion's bundled human animation GLB is used as the clip library. Idle/Walk/Run/Wave actions are mapped onto the SkinToken articulated skeleton; incompatible skeletons fail instead of exporting corrupted animation.
- Memory: only one heavyweight job owns the coordinator lease. Trellis2 and SkinToken run in separate processes, and every Blender stage is short-lived. The API also clears Python and CUDA caches.
- Artifact handoff: tabs pass artifact IDs and files on disk, not large meshes through browser memory.

## Requirements

For image-to-3D: Windows 10/11, an NVIDIA GPU with current drivers, and about 17 GB of free disk space. `trellis.cpp` downloads its CUDA runtime and one complete 16.5-GB GGUF set; it is not the incompatible Q4/Q8 ComfyUI package. On 8-GB GPUs AISmith 3D permits the lightweight 512 path only. CUDA PyTorch remains required for SkinToken.

Blender 4.2+ is required for native conversion, texture export, SkinToken, and animation. The setup script detects it and uses winget to install it when available. AutoRemesher must be built as `autoremesher.exe` with its Qt/MSVC toolchain; the source is included under `vendor/autoremesher` but the binary is not checked in because it is not a pip-installable dependency.

The current development machine has no NVIDIA GPU and CPU-only PyTorch. Therefore the native Trellis2/SkinToken inference paths are implemented but cannot be executed on this machine; they return a clear CUDA setup error. Blender texture baking and Mesh2Motion retargeting have been exercised locally.

## Run on Windows

```powershell
.\start.ps1 -Install
```

`-Install` creates `.venv` inside the project. The launcher always starts the API with that interpreter, detects NVIDIA hardware with `nvidia-smi`, and installs the Python dependencies needed for SkinToken and the rest of the studio. trellis.cpp is installed as a separate native CUDA runtime when the Generate tab downloads its models; it does not depend on Python, ComfyUI, PyTorch SDPA, or FlashAttention. Large model downloads are runtime-gated and shown in the bottom-right download dock.

If an older AISmith 3D API is already listening on port 8000, use `./start.ps1 -RestartApi` to replace it with the dedicated-environment API. The launcher checks for FastAPI, Uvicorn, and PyTorch inside `.venv`; an incomplete environment is repaired automatically.

Then open [http://127.0.0.1:5173](http://127.0.0.1:5173).

Copy `.env.example` to `.env` or set the variables in the process environment. Important paths are:

```text
TRELLIS_CPP_DIR=vendor/trellis.cpp
TRELLIS_CPP_MODEL_DIR=studio-data/models/trellis-cpp
TRELLIS_CPP_EXE=C:/path/to/trellis-cli.exe
SKINTOKEN_REPO_DIR=vendor/ComfyUI-SkinTokens
SKINTOKEN_MODELS_DIR=studio-data/models/skintoken
AUTOREMESHER_EXE=C:/path/to/autoremesher.exe
MESH2MOTION_LIBRARY=vendor/mesh2motion-app/static/animations/human-base-animations.glb
BLENDER_EXE=C:/Program Files/Blender Foundation/Blender 5.1/blender.exe
```

The Generate tab downloads trellis.cpp's Windows CUDA release plus its full GGUF model set when the user requests it. Auto Rig downloads SkinToken checkpoints only when requested. AutoRemesher remains the one non-Python dependency: its Windows Qt/MSVC executable must be built or supplied because it is not safely distributable through pip.

## Native AutoRemesher build boundary

The backend calls the upstream CLI with `--input`, `--output`, `--report`, `--target-quads`, `--edge-scaling`, `--sharp-edge`, `--smooth-normal`, and `--adaptivity`. Follow `vendor/autoremesher/README.md` to build the Windows release, then set `AUTOREMESHER_EXE`. A missing executable is reported in the job error; the source mesh is never silently returned as a fake remesh.

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
