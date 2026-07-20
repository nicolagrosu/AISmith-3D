from __future__ import annotations

import asyncio
import gc
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import native


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "studio-data"
UPLOADS = DATA / "uploads"
ARTIFACTS = DATA / "artifacts"
JOBS_DIR = DATA / "jobs"
DOWNLOADS_DIR = DATA / "downloads"
SETTINGS_PATH = DATA / "settings.json"
for directory in (UPLOADS, ARTIFACTS, JOBS_DIR, DOWNLOADS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

ALLOWED_UPLOADS = {"png", "jpg", "jpeg", "webp", "glb", "gltf", "fbx", "obj"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class JobRequest(BaseModel):
    kind: Literal["generate", "retopologize", "rig", "paint-bake", "animate"]
    source_artifact_id: str | None = None
    # 768 is AISmith 3D's balanced export preset: it runs the native 1024
    # cascade, then creates a viewport-friendly PBR asset.
    resolution: Literal[512, 768, 1024] = 768
    target_faces: int = Field(default=10000, ge=500, le=200000)
    quality: Literal["game", "balanced", "high"] = "game"
    animation: str = "Idle"
    settings: dict[str, Any] = Field(default_factory=dict)


class Artifact(BaseModel):
    id: str
    name: str
    kind: str
    path: str
    size_bytes: int
    created_at: float
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class JobState:
    id: str
    kind: str
    status: str = "queued"
    progress: int = 0
    message: str = "Queued"
    logs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    output_artifact_id: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class DownloadState:
    id: str
    model: str
    status: str = "queued"
    progress: int = 0
    message: str = "Queued"
    logs: list[str] = field(default_factory=list)
    error: str | None = None
    file_name: str | None = None
    file_index: int = 0
    file_count: int = 0
    file_bytes: int = 0
    file_total_bytes: int = 0
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._")
    return cleaned[:100] or "asset"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def output_directory() -> Path:
    """The user-selected export directory, defaulting to AISmith 3D's artifact cache."""
    if SETTINGS_PATH.exists():
        try:
            selected = str(_read_json(SETTINGS_PATH).get("output_directory", "")).strip()
            if selected:
                path = Path(selected).expanduser().resolve()
                if path.exists() and path.is_dir():
                    return path
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return ARTIFACTS.resolve()


def set_output_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError("Selected output directory does not exist")
    _write_json(SETTINGS_PATH, {"output_directory": str(resolved)})
    return resolved


def _artifact_path(artifact_id: str) -> Path:
    metadata = ARTIFACTS / f"{artifact_id}.json"
    if not metadata.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    artifact = _read_json(metadata)
    path = Path(artifact["path"]).resolve()
    allowed_roots = {ARTIFACTS.resolve(), output_directory()}
    if not any(path.parent == root or root in path.parents for root in allowed_roots):
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    return path


def save_artifact(source: Path, *, name: str, kind: str, metadata: dict[str, Any] | None = None) -> Artifact:
    artifact_id = uuid.uuid4().hex
    suffix = source.suffix.lower() or ".bin"
    destination = output_directory() / f"{artifact_id}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    artifact = Artifact(
        id=artifact_id,
        name=name,
        kind=kind,
        path=str(destination),
        size_bytes=destination.stat().st_size,
        created_at=time.time(),
        metadata=metadata or {},
    )
    _write_json(ARTIFACTS / f"{artifact_id}.json", artifact.model_dump())
    return artifact


def read_artifact(artifact_id: str) -> Artifact:
    metadata = ARTIFACTS / f"{artifact_id}.json"
    if not metadata.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return Artifact.model_validate(_read_json(metadata))


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False, "name": "CPU / undetected", "vram_total_mb": 0, "vram_used_mb": 0, "cuda": False, "torch_cuda_available": False, "compute_capability": None}
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,compute_cap", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and result.stdout.strip():
            fields = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
            name, total, used = fields[:3]
            compute = fields[3] if len(fields) > 3 else None
            info.update({"available": True, "name": name, "vram_total_mb": int(float(total)), "vram_used_mb": int(float(used)), "cuda": True, "compute_capability": compute})
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    try:
        import torch
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            info.update({"available": True, "name": properties.name, "vram_total_mb": round(properties.total_memory / 1024 / 1024), "cuda": True, "torch_cuda_available": True, "compute_capability": f"{properties.major}.{properties.minor}"})
    except Exception:
        pass
    info["model_variant"] = "GGUF CUDA"
    info["recommended_resolution"] = 512 if info["vram_total_mb"] <= 8192 else 1024
    info["native_inference_ready"] = bool(info["available"] and info["torch_cuda_available"] and info["compute_capability"])
    return info


class ModelManager:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.loaded: str | None = None
        self.loaded_at: float | None = None

    async def acquire(self, model_name: str) -> None:
        await self.lock.acquire()
        self.loaded = model_name
        self.loaded_at = time.time()

    async def release(self) -> None:
        try:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
            self.loaded = None
            self.loaded_at = None
        finally:
            if self.lock.locked():
                self.lock.release()


app = FastAPI(title="Local 3D Studio API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])
model_manager = ModelManager()
jobs: dict[str, JobState] = {}
downloads: dict[str, DownloadState] = {}


def _model_readiness() -> dict[str, bool]:
    skintoken_root = native.skintoken_model_dir()
    return {
        "trellis_cpp_ready": native.trellis_gguf_ready(),
        "trellis_gguf_ready": native.trellis_gguf_ready(),
        "skintoken_ready": (skintoken_root / "experiments/skin_vae_2_10_32768/last.ckpt").exists() and (skintoken_root / "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt").exists(),
    }


def _download_log(state: DownloadState, message: str, progress: int | None = None) -> None:
    state.message = message
    if progress is not None:
        state.progress = progress
    if message and (not state.logs or state.logs[-1] != message):
        state.logs.append(message[-500:])
    _write_json(DOWNLOADS_DIR / f"{state.id}.json", state.as_dict())


def _run_model_download(state: DownloadState) -> None:
    command = [sys.executable, "-u", "-m", "backend.bootstrap", "--download-model", state.model]
    state.status = "running"
    _download_log(state, f"Preparing {state.model} download", 3)
    try:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        import re
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event = None
            if isinstance(event, dict) and event.get("event") == "file_progress":
                state.file_name = str(event.get("file") or "") or None
                state.file_index = int(event.get("index") or 0)
                state.file_count = int(event.get("count") or 0)
                state.file_bytes = int(event.get("bytes") or 0)
                state.file_total_bytes = int(event.get("total") or 0)
                fraction = state.file_bytes / state.file_total_bytes if state.file_total_bytes else 0
                aggregate = ((max(1, state.file_index) - 1) + fraction) / max(1, state.file_count)
                _download_log(state, f"Downloading {state.file_name or 'model file'}", min(94, max(5, round(aggregate * 94))))
                continue
            percent = re.search(r"(\d{1,3})%", line)
            parsed_progress = int(percent.group(1)) if percent else min(94, max(5, state.progress + 1))
            _download_log(state, line, min(94, max(5, parsed_progress)))
        return_code = process.wait()
        if return_code != 0:
            hint = next((line for line in reversed(state.logs) if "Could not download" in line or "incomplete" in line or "not found" in line or "403 Forbidden" in line or "Cannot access gated repo" in line), "")
            raise RuntimeError(f"Model downloader exited with code {return_code}" + (f": {hint}" if hint else ""))
        state.status = "completed"
        _download_log(state, f"{state.model} is ready", 100)
    except Exception as exc:
        state.status = "failed"
        state.error = str(exc)
        _download_log(state, f"Download failed: {exc}", state.progress)


async def _execute_download(state: DownloadState) -> None:
    await asyncio.to_thread(_run_model_download, state)


def _job_log(job: JobState, message: str, progress: int | None = None) -> None:
    job.message = message
    if progress is not None:
        job.progress = progress
    job.logs.append(message)
    _write_json(JOBS_DIR / f"{job.id}.json", job.as_dict())


def _trellis_job_progress(message: str, current: int) -> int:
    """Map direct GGUF workflow stages to useful, non-jumping progress."""
    xatlas = re.search(r"xatlas: Computing charts:\s+(\d{1,3})%", message, re.IGNORECASE)
    if xatlas:
        # UV charting is the long CPU-bound section after the model stages.
        # Surface its own progress instead of pinning the UI at 78%.
        return min(92, 78 + round(int(xatlas.group(1)) * 14 / 100))
    stage_values = {"Loading Trellis": 14, "Removing background": 20, "stage 1/5": 30, "stage 2/5": 42, "stage 3/5": 56, "stage 4/5": 68, "stage 5/5": 77, "Remeshing": 80, "Exported": 94}
    for marker, value in stage_values.items():
        if marker in message:
            return value
    return min(78, max(18, current + 1))


async def _execute_job(job: JobState, request: JobRequest) -> None:
    source = read_artifact(request.source_artifact_id) if request.source_artifact_id else None
    model_name: str | None = None
    try:
        job.status = "running"
        _job_log(job, "Preparing native local worker", 5)
        model_name = {"generate": "trellis2", "rig": "skintoken", "animate": "hymotion"}.get(request.kind)
        if model_name:
            await model_manager.acquire(model_name)

        output = ARTIFACTS / f"job-{job.id}.glb"
        metadata = {"stage": request.kind, "settings": request.settings, "native": True, "worker_isolated": True}
        if request.kind == "generate":
            if source is None or source.kind not in {"image", "png", "jpg", "jpeg", "webp"}:
                raise ValueError("Generate requires a single uploaded image")
            mode_name = "coarse (marching-cubes)" if request.resolution == 512 else "fine (dual-grid)"
            _job_log(job, f"Selected direct Trellis2 C++ workflow in {mode_name} mode", 12)
            result = await asyncio.to_thread(native.run_trellis_gguf, Path(source.path), output, request.resolution, lambda message: _job_log(job, message, _trellis_job_progress(message, job.progress)))
            metadata.update(result)
        elif source:
            _job_log(job, f"Loading {source.name} into the {request.kind} worker", 15)
            if request.kind == "retopologize":
                metadata.update(await asyncio.to_thread(native.run_autoremesher, Path(source.path), output, request.target_faces, request.quality, lambda message: _job_log(job, message, min(78, max(18, job.progress + 3)))))
            elif request.kind == "rig":
                metadata.update(await asyncio.to_thread(native.run_skintoken, Path(source.path), output, lambda message: _job_log(job, message, min(78, max(18, job.progress + 3)))))
            elif request.kind == "animate":
                metadata.update(await asyncio.to_thread(native.run_animate, Path(source.path), output, request.animation, lambda message: _job_log(job, message, min(78, max(18, job.progress + 3)))))
            elif request.kind == "paint-bake":
                metadata.update(await asyncio.to_thread(native.run_paint, Path(source.path), output, request.settings, lambda message: _job_log(job, message, min(78, max(18, job.progress + 3)))))
            else:
                raise ValueError(f"Unsupported job kind: {request.kind}")
        else:
            raise ValueError(f"{request.kind} requires a source artifact")

        _job_log(job, "Validating GLB artifact", 82)
        if not output.exists() or output.stat().st_size < 64:
            raise ValueError("Adapter did not produce a valid GLB")
        artifact = save_artifact(output, name=f"{request.kind.title()} result", kind="glb", metadata=metadata)
        output.unlink(missing_ok=True)
        job.output_artifact_id = artifact.id
        job.status = "completed"
        _job_log(job, "Ready to pass to the next studio tab", 100)
    except asyncio.CancelledError:
        job.status = "cancelled"
        job.error = "Job cancelled"
        _job_log(job, job.error)
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        _job_log(job, f"Failed: {exc}")
    finally:
        if model_name:
            await model_manager.release()


@app.get("/api/system/capabilities")
async def capabilities() -> dict[str, Any]:
    gpu = _gpu_info()
    blender_path = native.find_blender()
    remesher = native.autoremesher_exe()
    skintoken_repo = native.skintoken_repo()
    readiness = _model_readiness()
    diagnostics: list[str] = []
    if gpu["available"] and not gpu["torch_cuda_available"]:
        diagnostics.append("NVIDIA GPU detected, but the active Python environment has CPU-only PyTorch; SkinToken requires CUDA PyTorch.")
    if gpu["available"] and gpu["compute_capability"]:
        try:
            if float(str(gpu["compute_capability"])) < 8.0:
                diagnostics.append("SkinToken requires Ampere-or-newer compute capability (8.0+).")
        except ValueError:
            pass
    if remesher is None:
        diagnostics.append("AutoRemesher executable is not configured; build vendor/autoremesher and set AUTOREMESHER_EXE.")
    if blender_path is None:
        diagnostics.append("Blender 4.2+ was not found; GLB conversion, texture baking, rigging, and animation export are unavailable.")
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "environment": {"python_executable": sys.executable, "dedicated": str(ROOT / ".venv") in sys.executable},
        "gpu": gpu,
        "blender": {"command": str(blender_path or "blender"), "available": blender_path is not None, "minimum_version": "4.2"},
        "models": {
            "trellis_cpp": False,
            "trellis_cpp_ready": readiness["trellis_cpp_ready"],
            "trellis_gguf_ready": readiness["trellis_gguf_ready"],
            "trellis2_native": readiness["trellis_gguf_ready"],
            "autoremesher_native": remesher is not None,
            "skintoken_native": (skintoken_repo / "nodes.py").exists(),
            "skintoken_ready": readiness["skintoken_ready"],
            "mesh2motion_native": native.mesh2motion_library().exists(),
        },
        "diagnostics": diagnostics,
        "downloads": {download_id: state.as_dict() for download_id, state in downloads.items() if state.status in {"queued", "running"}},
        "memory": {"loaded_model": model_manager.loaded, "loaded_at": model_manager.loaded_at},
        "output_directory": str(output_directory()),
    }


@app.post("/api/assets/upload", response_model=Artifact)
async def upload_asset(file: UploadFile = File(...)) -> Artifact:
    name = _safe_name(file.filename or "upload")
    suffix = Path(name).suffix.lower().lstrip(".")
    if suffix not in ALLOWED_UPLOADS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type. Use: {', '.join(sorted(ALLOWED_UPLOADS))}")
    destination = UPLOADS / f"{uuid.uuid4().hex}-{name}"
    total = 0
    with destination.open("wb") as handle:
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Upload exceeds 20 MB")
            handle.write(chunk)
    return save_artifact(destination, name=name, kind="image" if suffix in {"png", "jpg", "jpeg", "webp"} else suffix, metadata={"source": "upload"})


@app.post("/api/jobs", response_model=JobState)
async def create_job(request: JobRequest, background_tasks: BackgroundTasks) -> JobState:
    if request.source_artifact_id:
        read_artifact(request.source_artifact_id)
    readiness = _model_readiness()
    if request.kind == "generate":
        if not readiness["trellis_cpp_ready"]:
            raise HTTPException(status_code=409, detail="The direct Trellis2 GGUF Q4 workflow bundle is not downloaded. Open the Generate tab and download it before starting inference.")
    if request.kind == "rig" and not readiness["skintoken_ready"]:
        raise HTTPException(status_code=409, detail="SkinToken checkpoints are not downloaded. Open the Auto Rig tab and download them before starting rigging.")
    if request.kind != "generate" and not request.source_artifact_id:
        raise HTTPException(status_code=400, detail=f"{request.kind} requires source_artifact_id")
    job = JobState(id=uuid.uuid4().hex, kind=request.kind)
    jobs[job.id] = job
    _write_json(JOBS_DIR / f"{job.id}.json", job.as_dict())
    background_tasks.add_task(_execute_job, job, request)
    return job


@app.get("/api/downloads")
async def list_downloads() -> list[dict[str, Any]]:
    return [state.as_dict() for state in downloads.values() if state.status in {"queued", "running"}]


@app.post("/api/models/{name}/download", response_model=DownloadState)
async def download_model(name: str, background_tasks: BackgroundTasks) -> DownloadState:
    if name not in {"trellis2", "skintoken"}:
        raise HTTPException(status_code=404, detail="This tab has no downloadable model package")
    gpu = _gpu_info()
    if name == "skintoken" and not gpu["native_inference_ready"]:
        raise HTTPException(status_code=400, detail="CUDA-enabled PyTorch is not ready. Run start.ps1 -Install before downloading GPU models.")
    readiness = _model_readiness()
    if name == "trellis2":
        ready = readiness["trellis_cpp_ready"]
    else:
        ready = readiness["skintoken_ready"]
    if ready:
        state = DownloadState(id=uuid.uuid4().hex, model=name, status="completed", progress=100, message="Already downloaded")
        return state
    existing = next((state for state in downloads.values() if state.model == name and state.status in {"queued", "running"}), None)
    if existing:
        return existing
    state = DownloadState(id=uuid.uuid4().hex, model=name)
    downloads[state.id] = state
    _write_json(DOWNLOADS_DIR / f"{state.id}.json", state.as_dict())
    background_tasks.add_task(_execute_download, state)
    return state


@app.get("/api/downloads/{download_id}", response_model=DownloadState)
async def get_download(download_id: str) -> DownloadState:
    state = downloads.get(download_id)
    if state is None:
        stored = DOWNLOADS_DIR / f"{download_id}.json"
        if not stored.exists():
            raise HTTPException(status_code=404, detail="Download not found")
        state = DownloadState(**_read_json(stored))
        downloads[download_id] = state
    return state


@app.get("/api/jobs/{job_id}", response_model=JobState)
async def get_job(job_id: str) -> JobState:
    if job_id not in jobs:
        stored = JOBS_DIR / f"{job_id}.json"
        if not stored.exists():
            raise HTTPException(status_code=404, detail="Job not found")
        jobs[job_id] = JobState(**_read_json(stored))
    return jobs[job_id]


@app.post("/api/models/{name}/unload")
async def unload_model(name: str) -> dict[str, Any]:
    if model_manager.loaded and (name == "all" or name == model_manager.loaded):
        await model_manager.release()
    return {"ok": True, "loaded_model": model_manager.loaded}


@app.post("/api/system/clear-memory")
async def clear_memory() -> dict[str, Any]:
    if model_manager.lock.locked():
        return {"ok": False, "message": "A job is running; memory will be released when it completes.", "gpu": _gpu_info()}
    if model_manager.loaded:
        await model_manager.release()
    else:
        gc.collect()
    return {"ok": True, "message": "GPU cache and Python references cleared.", "gpu": _gpu_info()}


@app.post("/api/system/output-location/select")
async def select_output_location() -> dict[str, str]:
    """Open the native folder picker; the selected folder becomes future output storage."""
    def choose() -> str:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(initialdir=str(output_directory()), title="Choose AISmith 3D output folder")
        root.destroy()
        return selected

    selected = await asyncio.to_thread(choose)
    if not selected:
        return {"path": str(output_directory()), "changed": "false"}
    return {"path": str(set_output_directory(Path(selected))), "changed": "true"}


@app.get("/api/assets/{artifact_id}/download")
async def download_artifact(artifact_id: str) -> FileResponse:
    artifact = read_artifact(artifact_id)
    path = _artifact_path(artifact_id)
    return FileResponse(path, filename=artifact.name if Path(artifact.name).suffix else f"{artifact.name}.glb", media_type="model/gltf-binary")


@app.post("/api/assets/{artifact_id}/open-location")
async def open_location(artifact_id: str) -> dict[str, Any]:
    artifact = read_artifact(artifact_id)
    folder = str(Path(artifact.path).parent.resolve())
    if os.name == "nt":
        os.startfile(folder)  # type: ignore[attr-defined]
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", folder])
    else:
        subprocess.Popen(["xdg-open", folder])
    return {"ok": True, "folder": folder}


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
