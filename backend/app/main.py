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
import urllib.request
import zipfile
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
# Generated GLBs commonly exceed 20 MB once they carry PBR textures. Keep a
# bounded local-import limit, but allow normal studio assets to round-trip.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


class JobRequest(BaseModel):
    kind: Literal["generate", "retopologize", "retopo", "paint-bake"]
    source_artifact_id: str | None = None
    # 768 is AISmith 3D's balanced export preset: it runs the native 1024
    # cascade, then creates a viewport-friendly PBR asset.
    resolution: Literal[512, 768, 1024] = 768
    seed: int = Field(default=0, ge=0, le=2_147_483_647)
    target_faces: int = Field(default=10000, ge=500, le=200000)
    target_vertices: int | None = Field(default=None, ge=500, le=1_000_000)
    quality: Literal["game", "balanced", "high"] = "game"
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
    info["model_variant"] = "C++ Generate + TRELLIS.2 FP8 Refine"
    info["recommended_resolution"] = 512 if info["vram_total_mb"] <= 8192 else 1024
    info["native_inference_ready"] = bool(info["available"] and info["torch_cuda_available"] and info["compute_capability"])
    return info


def _ram_info() -> dict[str, int]:
    """System RAM snapshot for the local-only status bar."""
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            total_mb = round(status.ullTotalPhys / 1024 / 1024)
            available_mb = round(status.ullAvailPhys / 1024 / 1024)
            return {"total_mb": total_mb, "used_mb": max(0, total_mb - available_mb)}
    except Exception:
        pass
    return {"total_mb": 0, "used_mb": 0}


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
    return {
        "trellis_cpp_ready": native.trellis_cli() is not None and native.trellis_workflow_ready(),
        # Kept for older frontend clients; it now means the C++ geometry set.
        "trellis_gguf_ready": native.trellis_cli() is not None and native.trellis_workflow_ready(),
        "instant_meshes_ready": native.instant_meshes_exe() is not None,
        "trellis_refiner_ready": native.trellis_refiner_ready(),
    }


def _download_log(state: DownloadState, message: str, progress: int | None = None) -> None:
    state.message = message
    if progress is not None:
        state.progress = progress
    if message and (not state.logs or state.logs[-1] != message):
        state.logs.append(message[-500:])
    _write_json(DOWNLOADS_DIR / f"{state.id}.json", state.as_dict())


def _run_model_download(state: DownloadState) -> None:
    if state.model == "instant-meshes":
        _run_instant_meshes_download(state)
        return
    if state.model in {"trellis2", "trellis-refiner"}:
        _run_trellis_refiner_download(state)
        return
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


def _run_instant_meshes_download(state: DownloadState) -> None:
    """Fetch the upstream Windows bundle and safely unpack it into app data."""
    url = "https://instant-meshes.s3.eu-central-1.amazonaws.com/Release/instant-meshes-windows.zip"
    archive = DOWNLOADS_DIR / f"{state.id}-instant-meshes.zip"
    install = native.instant_meshes_dir()
    state.status = "running"
    _download_log(state, "Downloading Instant Meshes for Windows", 3)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "AISmith-3D/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            state.file_name = "instant-meshes-windows.zip"
            state.file_index = state.file_count = 1
            state.file_total_bytes = total
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                state.file_bytes += len(chunk)
                fraction = state.file_bytes / total if total else 0
                _download_log(state, "Downloading Instant Meshes for Windows", min(88, max(4, round(fraction * 88))))
        _download_log(state, "Extracting Instant Meshes executable", 90)
        install.mkdir(parents=True, exist_ok=True)
        destination_root = install.resolve()
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                destination = (destination_root / member.filename).resolve()
                if destination != destination_root and destination_root not in destination.parents:
                    raise RuntimeError("Instant Meshes archive contains an unsafe file path")
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
        if native.instant_meshes_exe() is None:
            raise RuntimeError("The download completed but Instant Meshes.exe was not found in the archive")
        state.status = "completed"
        _download_log(state, "Instant Meshes is ready", 100)
    except Exception as exc:
        state.status = "failed"
        state.error = str(exc)
        _download_log(state, f"Download failed: {exc}", state.progress)
    finally:
        archive.unlink(missing_ok=True)


def _run_trellis_refiner_download(state: DownloadState) -> None:
    """Install the isolated ComfyUI runtime and FP8 refiner checkpoints.

    It deliberately uses the app's CUDA Python environment rather than the
    generation C++ runtime.  The two stacks must never share a process.
    """
    runtime = native.trellis_refiner_dir()
    archive = DOWNLOADS_DIR / f"{state.id}-comfyui.zip"
    comfy = runtime / "ComfyUI"
    wrapper_source = ROOT / "vendor" / "ComfyUI-Trellis2"
    state.status = "running"
    _download_log(state, "Preparing isolated TRELLIS.2 FP8 runtime", 2)
    try:
        if not wrapper_source.is_dir():
            raise RuntimeError("The bundled ComfyUI-Trellis2 integration is missing")
        if not (comfy / "folder_paths.py").exists():
            _download_log(state, "Downloading the local CUDA refiner runtime", 5)
            request = urllib.request.Request("https://github.com/comfyanonymous/ComfyUI/archive/refs/heads/master.zip", headers={"User-Agent": "AISmith-3D/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as handle:
                total = int(response.headers.get("Content-Length") or 0)
                state.file_name, state.file_index, state.file_count, state.file_total_bytes = "ComfyUI runtime", 1, 5, total
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                    state.file_bytes += len(chunk)
                    fraction = state.file_bytes / total if total else 0
                    _download_log(state, "Downloading the local CUDA refiner runtime", min(20, 5 + round(fraction * 15)))
            extracted = runtime / "ComfyUI-master"
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    destination = (runtime / member.filename).resolve()
                    if runtime.resolve() not in destination.parents and destination != runtime.resolve():
                        raise RuntimeError("ComfyUI archive contains an unsafe file path")
                    bundle.extract(member, runtime)
            if not extracted.exists():
                raise RuntimeError("ComfyUI runtime archive was incomplete")
            extracted.rename(comfy)
        _download_log(state, "Installing the TRELLIS.2 refiner CUDA extensions", 25)
        target = comfy / "custom_nodes" / "ComfyUI-Trellis2"
        if not target.exists():
            shutil.copytree(wrapper_source, target)
        wheel_root = wrapper_source / "wheels" / "Windows" / "Torch280"
        wheels = [
            wheel_root / "cumesh-1.0-cp312-cp312-win_amd64.whl",
            wheel_root / "nvdiffrast-0.4.0-cp312-cp312-win_amd64.whl",
            wheel_root / "nvdiffrec_render-0.0.0-cp312-cp312-win_amd64.whl",
            wheel_root / "flex_gemm-0.0.1-cp312-cp312-win_amd64.whl",
            wheel_root / "o_voxel-0.0.1-cp312-cp312-win_amd64.whl",
        ]
        if not all(wheel.exists() for wheel in wheels):
            raise RuntimeError("No CUDA extension wheels match the app's Python 3.12 / Torch 2.8 runtime")
        command = [sys.executable, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed", "-r", str(comfy / "requirements.txt"), "huggingface_hub", *map(str, wheels)]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=1800)
        if completed.returncode != 0:
            raise RuntimeError(f"Could not install TRELLIS.2 refiner dependencies: {(completed.stdout + completed.stderr)[-1200:]}")
        _download_log(state, "Downloading TRELLIS.2 FP8 refinement and PBR models", 35)
        from huggingface_hub import snapshot_download
        models = runtime / "models"
        for index, repo in enumerate((
            "visualbruno/TRELLIS.2-4B-FP8",
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
        ), start=1):
            state.file_name, state.file_index, state.file_count = repo, index, 2
            _download_log(state, f"Downloading {repo}", 35 + index * 25)
            snapshot_download(repo_id=repo, local_dir=models / repo, local_dir_use_symlinks=False)
        # The wrapper requires the sparse-structure decoder but not the full
        # 3-GB legacy TRELLIS-image-large repository.
        state.file_name, state.file_index, state.file_count = "TRELLIS-image-large decoder", 3, 3
        _download_log(state, "Downloading required TRELLIS image decoder", 88)
        from huggingface_hub import hf_hub_download
        for filename in ("ckpts/ss_dec_conv3d_16l8_fp16.json", "ckpts/ss_dec_conv3d_16l8_fp16.safetensors"):
            hf_hub_download(repo_id="microsoft/TRELLIS-image-large", filename=filename, local_dir=models / "microsoft" / "TRELLIS-image-large")
        if not native.trellis_refiner_ready():
            raise RuntimeError("The TRELLIS.2 refiner download did not contain all required model files")
        state.status = "completed"
        _download_log(state, "TRELLIS.2 FP8 Refine is ready", 100)
    except Exception as exc:
        state.status = "failed"
        state.error = str(exc)
        _download_log(state, f"Download failed: {exc}", state.progress)
    finally:
        archive.unlink(missing_ok=True)


async def _execute_download(state: DownloadState) -> None:
    await asyncio.to_thread(_run_model_download, state)


def _job_log(job: JobState, message: str, progress: int | None = None) -> None:
    job.message = message
    if progress is not None:
        job.progress = progress
    job.logs.append(message)
    _write_json(JOBS_DIR / f"{job.id}.json", job.as_dict())


def _trellis_job_progress(message: str, current: int) -> int:
    """Map one-shot FP8 worker logs to useful, non-jumping progress."""
    xatlas = re.search(r"xatlas: Computing charts:\s+(\d{1,3})%", message, re.IGNORECASE)
    if xatlas:
        # UV charting is the long CPU-bound section after the model stages.
        # Surface its own progress instead of pinning the UI at 78%.
        return min(92, 78 + round(int(xatlas.group(1)) * 14 / 100))
    stage_values = {"Loading TRELLIS": 14, "Generating": 30, "Refining geometry": 30, "Reconstructing": 55, "Simplifying": 65, "Filling holes": 70, "Generating PBR": 78, "Exporting": 94, "job complete": 96}
    for marker, value in stage_values.items():
        if marker in message:
            return value
    return min(78, max(18, current + 1))


async def _execute_job(job: JobState, request: JobRequest) -> None:
    source = read_artifact(request.source_artifact_id) if request.source_artifact_id else None
    model_name: str | None = None
    try:
        if job.status == "cancelled":
            return
        job.status = "running"
        _job_log(job, "Preparing native local worker", 5)
        model_name = {"generate": "trellis-cpp", "retopologize": "trellis2-fp8", "retopo": "instant-meshes"}.get(request.kind)
        if model_name:
            await model_manager.acquire(model_name)
        if job.status == "cancelled":
            return

        output = ARTIFACTS / f"job-{job.id}.glb"
        metadata = {"stage": request.kind, "settings": request.settings, "native": True, "worker_isolated": True}
        # Keep provenance with every generated asset.  The optional image-guided
        # TRELLIS refiner needs the exact original reference image; it must not
        # guess one from the texture baked into a GLB.
        if source is not None:
            metadata["source_artifact_id"] = source.id
        if request.kind == "generate" and source is not None:
            metadata["source_image_artifact_id"] = source.id
        if request.kind == "generate":
            if source is None or source.kind not in {"image", "png", "jpg", "jpeg", "webp"}:
                raise ValueError("Generate requires a single uploaded image")
            mode_name = "fast" if request.resolution == 512 else "high-detail"
            _job_log(job, f"Selected alpha-aware C++ geometry workflow in {mode_name} mode", 12)
            result = await asyncio.to_thread(native.run_trellis, Path(source.path), output, request.resolution, request.seed, lambda message: _job_log(job, message, _trellis_job_progress(message, job.progress)))
            metadata.update(result)
        elif source:
            _job_log(job, f"Loading {source.name} into the {request.kind} worker", 15)
            if request.kind == "retopologize":
                target_vertices = request.target_vertices or request.target_faces
                repair_holes = bool(request.settings.get("repair_holes", False))
                gpu_first = bool(request.settings.get("gpu_first", False))
                # An explicitly attached image is required for imported GLBs and
                # takes precedence over Generate provenance.
                image_id = str(request.settings.get("refine_image_artifact_id") or source.metadata.get("source_image_artifact_id") or "")
                if not image_id:
                    parent_id = str(source.metadata.get("source_artifact_id") or "")
                    if parent_id:
                        parent = read_artifact(parent_id)
                        image_id = str(parent.metadata.get("source_image_artifact_id") or (parent.id if parent.kind == "image" else ""))
                if not image_id:
                    raise ValueError("Refine & Reconstruct requires a reference image. Attach one for imported GLBs.")
                image = read_artifact(image_id)
                if image.kind != "image":
                    raise ValueError("The retained refinement reference is not an image artifact.")
                metadata["source_image_artifact_id"] = image.id
                metadata.update(await asyncio.to_thread(native.run_trellis_ai_refine, Path(source.path), Path(image.path), output, target_vertices, repair_holes, gpu_first, lambda message: _job_log(job, message, min(94, max(18, job.progress + 4)))))
            elif request.kind == "retopo":
                target_vertices = request.target_vertices or request.target_faces
                repair_holes = bool(request.settings.get("repair_holes", False))
                metadata.update(await asyncio.to_thread(native.run_instant_meshes, Path(source.path), output, target_vertices, repair_holes, lambda message: _job_log(job, message, min(94, max(18, job.progress + 4)))))
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
        if job.status == "cancelled":
            _job_log(job, "Cancelled")
            return
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
    readiness = _model_readiness()
    diagnostics: list[str] = []
    if blender_path is None:
        diagnostics.append("Blender 4.2+ was not found; GLB conversion and texture baking are unavailable.")
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "environment": {"python_executable": sys.executable, "dedicated": str(ROOT / ".venv") in sys.executable},
        "gpu": gpu,
        "ram": _ram_info(),
        "blender": {"command": str(blender_path or "blender"), "available": blender_path is not None, "minimum_version": "4.2"},
        "models": {
            "trellis_cpp": native.trellis_cli() is not None,
            "trellis_cpp_ready": readiness["trellis_cpp_ready"],
            "trellis_gguf_ready": readiness["trellis_gguf_ready"],
            "trellis2_native": native.trellis_cli() is not None,
            "instant_meshes_native": native.instant_meshes_exe() is not None,
            "trellis_refiner_ready": readiness["trellis_refiner_ready"],
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
    exceeded_limit = False
    with destination.open("wb") as handle:
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                exceeded_limit = True
                break
            handle.write(chunk)
    if exceeded_limit:
        # On Windows the file cannot be removed from inside its active ``with``
        # block; close it first so a size rejection remains a clean 413 error.
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail="Upload exceeds 100 MB")
    return save_artifact(destination, name=name, kind="image" if suffix in {"png", "jpg", "jpeg", "webp"} else suffix, metadata={"source": "upload"})


@app.post("/api/jobs", response_model=JobState)
async def create_job(request: JobRequest, background_tasks: BackgroundTasks) -> JobState:
    if request.source_artifact_id:
        read_artifact(request.source_artifact_id)
    readiness = _model_readiness()
    if request.kind == "generate":
        if not readiness["trellis_cpp_ready"]:
            raise HTTPException(status_code=409, detail="The trellis.cpp geometry workflow is not downloaded. Open Generate and download it before starting inference.")
    if request.kind == "retopologize" and not readiness["trellis_refiner_ready"]:
        raise HTTPException(status_code=409, detail="TRELLIS.2 FP8 Refine is not downloaded. Open Refine and download it before starting.")
    if request.kind == "retopo" and not readiness["instant_meshes_ready"]:
        raise HTTPException(status_code=409, detail="Instant Meshes is not downloaded. Open Retopology and download it before starting.")
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
    if name not in {"trellis-cpp", "trellis2", "instant-meshes", "trellis-refiner"}:
        raise HTTPException(status_code=404, detail="This tab has no downloadable model package")
    gpu = _gpu_info()
    readiness = _model_readiness()
    if name in {"trellis2", "trellis-refiner"}:
        ready = readiness["trellis_refiner_ready"]
    elif name == "trellis-cpp":
        ready = readiness["trellis_cpp_ready"]
    else:
        ready = readiness["instant_meshes_ready"]
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


@app.post("/api/jobs/{job_id}/cancel", response_model=JobState)
async def cancel_job(job_id: str) -> JobState:
    job = jobs.get(job_id)
    if job is None:
        stored = JOBS_DIR / f"{job_id}.json"
        if not stored.exists():
            raise HTTPException(status_code=404, detail="Job not found")
        job = JobState(**_read_json(stored))
        jobs[job_id] = job
    if job.status not in {"queued", "running"}:
        return job
    job.status = "cancelled"
    job.error = "Cancelled by user"
    native.cancel_active_process()
    _job_log(job, "Cancelling task")
    return job


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


@app.post("/api/system/clear-ram")
async def clear_ram() -> dict[str, Any]:
    if model_manager.lock.locked():
        return {"ok": False, "message": "A job is running; RAM is released when it completes.", "ram": _ram_info()}
    gc.collect()
    return {"ok": True, "message": "AISmith idle Python memory cleared.", "ram": _ram_info()}


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


@app.get("/api/assets/{artifact_id}/details")
async def asset_details(artifact_id: str) -> dict[str, Any]:
    """Read actual mesh counts from generated and uploaded GLB artifacts."""
    artifact = read_artifact(artifact_id)
    path = _artifact_path(artifact_id)
    if path.suffix.lower() not in {".glb", ".gltf"}:
        raise HTTPException(status_code=415, detail="Model details are available for GLB and glTF assets")
    try:
        import trimesh

        scene = trimesh.load(path, force="scene", process=False)
        geometries = list(scene.geometry.values())
        materials = {
            id(getattr(getattr(mesh, "visual", None), "material", None))
            for mesh in geometries
            if getattr(getattr(mesh, "visual", None), "material", None) is not None
        }
        return {
            "vertices": sum(len(mesh.vertices) for mesh in geometries),
            "triangles": sum(len(mesh.faces) for mesh in geometries),
            "materials": len(materials),
            "meshes": len(geometries),
            "format": path.suffix.lstrip(".").upper(),
            "size_bytes": path.stat().st_size,
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not inspect this model: {exc}") from exc


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
