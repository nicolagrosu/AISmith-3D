from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
Progress = Callable[[str], None]
_ACTIVE_PROCESS_LOCK = threading.Lock()
_ACTIVE_PROCESS: subprocess.Popen[str] | None = None


def _set_active_process(process: subprocess.Popen[str] | None) -> None:
    global _ACTIVE_PROCESS
    with _ACTIVE_PROCESS_LOCK:
        _ACTIVE_PROCESS = process


def cancel_active_process() -> bool:
    """Stop only the currently running AISmith child worker, if any."""
    with _ACTIVE_PROCESS_LOCK:
        process = _ACTIVE_PROCESS
    if process is None or process.poll() is not None:
        return False
    process.terminate()
    return True


def configured_path(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser().resolve() if value else default.resolve()


def find_blender() -> Path | None:
    configured = os.getenv("BLENDER_EXE", "").strip()
    candidates = [Path(configured)] if configured else []
    candidates += [
        Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"),
        Path(r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"),
    ]
    found = shutil.which("blender")
    if found:
        candidates.append(Path(found))
    return next((path for path in candidates if path and path.exists()), None)


def trellis_install_dir() -> Path:
    return configured_path("TRELLIS_CPP_DIR", ROOT / "vendor" / "trellis.cpp")


def trellis_model_dir() -> Path:
    return configured_path("TRELLIS_CPP_MODEL_DIR", ROOT / "studio-data" / "models" / "trellis-cpp")


def trellis_gguf_model_dir() -> Path:
    """Model root for the direct Aero-Ex Q4 workflow (not trellis.cpp)."""
    return configured_path("TRELLIS_GGUF_MODEL_DIR", ROOT / "studio-data" / "models" / "Trellis2")


def trellis_gguf_manifest() -> list[tuple[str, str]]:
    """All model files required for the full geometry+texture pipeline.
    Texture models are optional — geometry still works without them."""
    return [
        ("ggufs/dino_f16.gguf",        "ggufs/dino_f16.gguf"),
        ("ggufs/ss_flow_f16.gguf",     "ggufs/ss_flow_f16.gguf"),
        ("ggufs/ss_dec_f16.gguf",      "ggufs/ss_dec_f16.gguf"),
        ("ggufs/slat_flow_f16.gguf",   "ggufs/slat_flow_f16.gguf"),
        ("ggufs/shape_dec_f16.gguf",   "ggufs/shape_dec_f16.gguf"),
        # Texture pipeline (optional but included in the download target)
        ("ggufs/shape_enc_f16.gguf",           "ggufs/shape_enc_f16.gguf"),
        ("ggufs/tex_dec_f16.gguf",             "ggufs/tex_dec_f16.gguf"),
        ("ggufs/tex_slat_flow_512_f16.gguf",   "ggufs/tex_slat_flow_512_f16.gguf"),
        ("ggufs/tex_slat_flow_1024_f16.gguf",  "ggufs/tex_slat_flow_1024_f16.gguf"),
    ]


REQUIRED_GEOMETRY_GGUFS = {
    "ggufs/dino_f16.gguf",
    "ggufs/ss_flow_f16.gguf",
    "ggufs/ss_dec_f16.gguf",
    "ggufs/slat_flow_f16.gguf",
    "ggufs/shape_dec_f16.gguf",
}


def trellis_gguf_missing_files() -> list[Path]:
    """Returns files missing from the required geometry-only set."""
    root = trellis_gguf_model_dir()
    return [root / local for _, local in trellis_gguf_manifest()
            if local in REQUIRED_GEOMETRY_GGUFS
            and (not (root / local).is_file() or (root / local).stat().st_size == 0)]


def trellis_texture_ready() -> bool:
    """True when all four texture GGUFs are present on disk."""
    root = trellis_gguf_model_dir()
    tex_files = [
        root / "ggufs/shape_enc_f16.gguf",
        root / "ggufs/tex_dec_f16.gguf",
        root / "ggufs/tex_slat_flow_512_f16.gguf",
        root / "ggufs/tex_slat_flow_1024_f16.gguf",
    ]
    return all(p.is_file() and p.stat().st_size > 0 for p in tex_files)


def trellis_gguf_ready() -> bool:
    return not trellis_gguf_missing_files()


def trellis_model_files() -> list[Path]:
    """Geometry model set used by the standalone trellis.cpp generator."""
    return [trellis_model_dir() / filename for filename in (
        "birefnet.gguf", "dinov3.gguf", "shape_dec.gguf", "shape_flow_512.gguf",
        "shape_flow_1024.gguf", "ss_dec.gguf", "ss_flow.gguf",
    )]


def trellis_workflow_missing_files() -> list[Path]:
    return [path for path in trellis_model_files() if not path.is_file() or path.stat().st_size == 0]


def trellis_workflow_ready() -> bool:
    return not trellis_workflow_missing_files()


def trellis_cli() -> Path | None:
    """Return the standalone trellis.cpp executable from the release bundle.

    The Windows v0.4.3 release names this binary ``trellis-server.exe``.  It
    also supports one-shot ``-i``/``-o`` generation, which is what AISmith 3D
    uses so each job can exit and release its VRAM.
    """
    configured = os.getenv("TRELLIS_CPP_EXE", "").strip()
    candidates = [Path(configured)] if configured else []
    install = trellis_install_dir()
    candidates += [install / "trellis-cli.exe", install / "bin" / "trellis-cli.exe", install / "trellis-server.exe", install / "bin" / "trellis-server.exe"]
    if install.exists():
        candidates += install.rglob("trellis-cli.exe")
        candidates += install.rglob("trellis-server.exe")
    found = shutil.which("trellis-cli")
    if found:
        candidates.append(Path(found))
    return next((path.resolve() for path in candidates if path.exists()), None)


def prepare_trellis_input(input_image: Path) -> bytes:
    """Letterbox an image into a transparent square without changing its aspect ratio.

    trellis.cpp's bundled BiRefNet path stretches non-square images to 1024².
    A transparent, aspect-fit PNG makes the runtime use its alpha-aware path
    instead; alpha-zero padding is excluded from the object cutout.
    """
    import io
    from PIL import Image

    canvas_size = 1024
    with Image.open(input_image) as source:
        rgba = source.convert("RGBA")
        width, height = rgba.size
        if width <= 0 or height <= 0:
            raise ValueError("Trellis2 input image has invalid dimensions")
        scale = min(canvas_size / width, canvas_size / height)
        resized = rgba.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    offset = ((canvas_size - resized.width) // 2, (canvas_size - resized.height) // 2)
    canvas.alpha_composite(resized, dest=offset)
    output = io.BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()


def autoremesher_exe() -> Path | None:
    configured = os.getenv("AUTOREMESHER_EXE", "").strip()
    candidates = [Path(configured)] if configured else []
    candidates += [ROOT / "vendor" / "autoremesher" / "release" / "autoremesher.exe", ROOT / "vendor" / "autoremesher" / "build" / "release" / "autoremesher.exe"]
    found = shutil.which("autoremesher")
    if found:
        candidates.append(Path(found))
    return next((path.resolve() for path in candidates if path.exists()), None)


def instant_meshes_dir() -> Path:
    """Managed install location for the official Instant Meshes Windows ZIP."""
    return configured_path("INSTANT_MESHES_DIR", ROOT / "studio-data" / "tools" / "instant-meshes")


def instant_meshes_exe() -> Path | None:
    configured = os.getenv("INSTANT_MESHES_EXE", "").strip()
    candidates = [Path(configured)] if configured else []
    install = instant_meshes_dir()
    candidates += [install / "Instant Meshes.exe", install / "InstantMeshes.exe"]
    if install.exists():
        candidates += install.rglob("Instant Meshes.exe")
        candidates += install.rglob("InstantMeshes.exe")
    return next((path.resolve() for path in candidates if path.exists()), None)


def trellis_refiner_dir() -> Path:
    """Managed, isolated ComfyUI runtime for image-guided TRELLIS.2 FP8 refine."""
    return configured_path("TRELLIS_REFINER_DIR", ROOT / "studio-data" / "tools" / "trellis2-refiner")


def trellis_refiner_ready() -> bool:
    """Whether the isolated FP8 TRELLIS.2 refinement runtime is ready."""
    runtime = trellis_refiner_dir()
    return all(path.exists() for path in (
        runtime / "ComfyUI" / "folder_paths.py",
        runtime / "ComfyUI" / "custom_nodes" / "ComfyUI-Trellis2" / "nodes.py",
        runtime / "models" / "visualbruno" / "TRELLIS.2-4B-FP8" / "pipeline_fp8.json",
        runtime / "models" / "facebook" / "dinov3-vitl16-pretrain-lvd1689m" / "model.safetensors",
        runtime / "models" / "microsoft" / "TRELLIS-image-large" / "ckpts" / "ss_dec_conv3d_16l8_fp16.safetensors",
    ))


def run_process(command: list[str], *, cwd: Path = ROOT, timeout: int = 3600, progress: Progress | None = None, env: dict[str, str] | None = None) -> str:
    if progress:
        progress(f"Launching {Path(command[0]).name}")
    merged = os.environ.copy()
    if env:
        merged.update(env)
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=merged)
    _set_active_process(process)
    lines: list[str] = []

    def relay() -> None:
        if process.stdout is None:
            return
        for raw in process.stdout:
            line = raw.strip()
            if not line:
                continue
            lines.append(line)
            if progress:
                progress(line[:500])

    reader = threading.Thread(target=relay, name=f"aismith-{Path(command[0]).stem}-log", daemon=True)
    reader.start()
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.wait(timeout=15)
        raise RuntimeError(f"{Path(command[0]).name} exceeded its {timeout // 60}-minute time limit") from exc
    finally:
        reader.join(timeout=5)
        _set_active_process(None)
    output = "\n".join(lines).strip()
    if return_code != 0:
        raise RuntimeError(f"{Path(command[0]).name} failed ({return_code}): {output[-1800:]}")
    return output


def blender_convert(input_path: Path, output_path: Path, progress: Progress | None = None) -> None:
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required for GLB/OBJ conversion but was not found")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    spec = json.dumps({"op": "convert", "input": str(input_path), "output": str(output_path)})
    run_process([str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec], progress=progress, timeout=600)


def blender_export_obj(input_path: Path, output_path: Path, progress: Progress | None = None) -> None:
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required for GLB/OBJ conversion but was not found")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    spec = json.dumps({"op": "export-obj", "input": str(input_path), "output": str(output_path)})
    run_process([str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec], progress=progress, timeout=900)


def blender_bake_retopology(source_glb: Path, remeshed_obj: Path, output_glb: Path, progress: Progress | None = None) -> None:
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required to bake source textures onto the retopologized mesh")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    spec = json.dumps({"op": "retopo-bake", "source": str(source_glb), "target": str(remeshed_obj), "output": str(output_glb)})
    run_process([str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec], progress=progress, timeout=1800)


def run_trellis(input_image: Path, output_glb: Path, resolution: int, seed: int = 0, progress: Progress | None = None) -> dict[str, Any]:
    cli = trellis_cli()
    if cli is None:
        raise RuntimeError("trellis.cpp CUDA runtime is not installed. Click Download models to install its Windows CUDA release.")
    missing = trellis_workflow_missing_files()
    if missing:
        raise RuntimeError("trellis.cpp model download is incomplete: " + ", ".join(path.name for path in missing))
    # trellis.cpp publishes the Windows binary as trellis-server.exe. It is not
    # a user-facing service here: AISmith 3D starts it on a private loopback port
    # for one job, sends the image itself, then terminates it to release VRAM.
    import requests
    from PIL import Image

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    actual_resolution = 512 if resolution <= 512 else 1024
    command = [str(cli), "--models", str(trellis_model_dir()), "--res", str(actual_resolution), "--no-texture", "--require-gpu", "--seed", str(seed), "--host", "127.0.0.1", "--port", str(port)]
    if progress:
        progress("Starting private native trellis.cpp worker")
    server = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    _set_active_process(server)
    server_output = ""
    log_lines: list[str] = []

    def relay_native_log() -> None:
        if server.stdout is None:
            return
        for raw_line in server.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            log_lines.append(line)
            if progress and ("[" in line and "/6]" in line):
                progress(line)

    log_thread = threading.Thread(target=relay_native_log, name="aismith-trellis-log", daemon=True)
    log_thread.start()

    def stop_server() -> str:
        nonlocal server_output
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=15)
        log_thread.join(timeout=3)
        _set_active_process(None)
        server_output = "\n".join(log_lines)
        return server_output

    try:
        deadline = time.monotonic() + 90
        health_url = f"http://127.0.0.1:{port}/health"
        while time.monotonic() < deadline:
            if server.poll() is not None:
                output = stop_server()
                raise RuntimeError(f"trellis.cpp worker exited while starting: {output[-1200:]}")
            try:
                if requests.get(health_url, timeout=1).ok:
                    break
            except requests.RequestException:
                time.sleep(0.25)
        else:
            raise RuntimeError("trellis.cpp worker did not become ready within 90 seconds")
        if progress:
            progress("Preparing aspect-preserving transparent input for Trellis2")
        prepared_png = prepare_trellis_input(input_image)
        if progress:
            progress("Running native Trellis image-to-3D")
        response = requests.post(
            f"http://127.0.0.1:{port}/generate",
            files={"image": ("reference.png", prepared_png, "image/png")},
            data={"seed": str(seed), "resolution": str(actual_resolution)},
            timeout=7200,
        )
        if not response.ok:
            native_log = stop_server().strip()
            raise RuntimeError(f"trellis.cpp generation failed ({response.status_code}): {response.text[-500:]}\nNative runtime log:\n{native_log[-1800:]}")
        output_glb.write_bytes(response.content)
    finally:
        stop_server()
    return {"backend": "trellis.cpp", "model_set": "ilintar/trellis2-gguf geometry", "resolution": actual_resolution, "seed": seed, "textured": False}


def run_trellis_gguf(input_image: Path, output_glb: Path, resolution: int, progress: Progress | None = None) -> dict[str, Any]:
    """Run the full Trellis2 geometry+texture C++ pipeline.

    Uses the pr-1 trellis2cpp DLL which implements the complete pipeline:
    sparse-structure -> shape SLAT -> mesh + PBR texture flow -> UV-atlas GLB.
    Falls back gracefully to geometry-only clay mesh when texture GGUFs are absent.
    """
    from backend.app.trellis2_wrapper import (
        Trellis2Wrapper, T2_PIPE_AUTO, T2_PIPE_512, T2_PIPE_1024,
        T2_PIPE_COARSE, T2_BACKGROUND_AUTO, T2_CAP_TEXTURE,
    )
    import random
    import io
    from PIL import Image

    gguf_dir = trellis_gguf_model_dir() / "ggufs"

    # ── Required geometry GGUFs ───────────────────────────────────────────
    dino      = gguf_dir / "dino_f16.gguf"
    ss_flow   = gguf_dir / "ss_flow_f16.gguf"
    ss_dec    = gguf_dir / "ss_dec_f16.gguf"
    slat_flow = gguf_dir / "slat_flow_f16.gguf"
    shape_dec = gguf_dir / "shape_dec_f16.gguf"

    for p in (dino, ss_flow, ss_dec):
        if not p.is_file():
            raise RuntimeError(
                f"Required geometry model is missing: {p.name}. "
                "Download models from the Generate tab first."
            )

    # ── Optional texture GGUFs ────────────────────────────────────────────
    shape_enc     = gguf_dir / "shape_enc_f16.gguf"
    tex_dec       = gguf_dir / "tex_dec_f16.gguf"
    tex_flow_512  = gguf_dir / "tex_slat_flow_512_f16.gguf"
    tex_flow_1024 = gguf_dir / "tex_slat_flow_1024_f16.gguf"

    def _opt(p: Path) -> str | None:
        return str(p) if p.is_file() and p.stat().st_size > 0 else None

    has_slat  = slat_flow.is_file() and shape_dec.is_file()
    has_tex   = all(p.is_file() and p.stat().st_size > 0
                    for p in (shape_enc, tex_dec, tex_flow_512))

    # ── Pick pipeline type ────────────────────────────────────────────────
    if resolution <= 512:
        pipe_type = T2_PIPE_COARSE
        mode_label = "coarse (marching-cubes)"
    elif resolution >= 1024 and has_slat:
        pipe_type = T2_PIPE_1024
        mode_label = "fine 1024 cascade"
    elif has_slat:
        pipe_type = T2_PIPE_512
        mode_label = "fine 512 dual-grid"
    else:
        pipe_type = T2_PIPE_COARSE
        mode_label = "coarse (marching-cubes, slat models missing)"

    if has_tex:
        mode_label += " + PBR texture"

    # ── Convert input to PNG bytes ─────────────────────────────────────────
    with Image.open(input_image) as img:
        img_rgba = img.convert("RGBA")
        png_io = io.BytesIO()
        img_rgba.save(png_io, format="PNG")
        image_bytes = png_io.getvalue()

    if progress:
        progress(f"Loading Trellis2 pipeline ({mode_label})...")

    wrapper = Trellis2Wrapper(ROOT / "vendor" / "trellis2cpp" / "build")
    pipeline = wrapper.load_pipeline(
        dino=str(dino),
        ss_flow=str(ss_flow),
        ss_dec=str(ss_dec),
        slat_flow=_opt(slat_flow),
        slat_hr_flow=None,               # reserved for future HR cascade
        shape_dec=_opt(shape_dec),
        shape_enc=_opt(shape_enc),
        tex_dec=_opt(tex_dec),
        tex_flow=_opt(tex_flow_512),
        tex_flow_hr=_opt(tex_flow_1024),
    )

    try:
        _STAGE_LABELS = {
            0:  "Pre-processing image",
            1:  "Encoding with DINOv3",
            2:  "Sparse-structure flow",
            3:  "Decoding occupancy grid",
            4:  "Shape SLAT flow",
            5:  "Decoding shape geometry",
            6:  "Extracting mesh",
            7:  "Upsampling to 1024 voxels",
            8:  "1024 shape SLAT flow",
            9:  "1024 shape decode",
            10: "PBR texture flow",
        }

        def on_progress(stage: int, step: int, total: int) -> None:
            label = _STAGE_LABELS.get(stage, f"Stage {stage}")
            if total > 0:
                label = f"{label} (step {step}/{total})"
            if progress:
                progress(label)

        if progress:
            progress("Running Trellis2 image-to-3D generation...")

        seed = random.randint(0, 2 ** 32 - 1)
        mesh_data = wrapper.generate(
            pipeline,
            image_bytes,
            seed=seed,
            pipeline_type=pipe_type,
            background_mode=T2_BACKGROUND_AUTO,
            steps=0,           # use C++ defaults (12)
            guidance=-1.0,     # use C++ defaults (7.5)
            texture_steps=0,   # use C++ defaults (12)
            progress_cb=on_progress,
        )
    finally:
        wrapper.free_pipeline(pipeline)

    if progress:
        progress("Baking GLB artifact...")

    if mesh_data["has_pbr"]:
        # Full PBR path: let the C++ xatlas bake produce the final GLB
        # This avoids trimesh stripping UV data and gives the best quality.
        if progress:
            progress("Baking UV-atlas textured GLB (xatlas)...")
        glb_bytes = wrapper.bake_glb(
            verts=mesh_data["vertices"],
            tris=mesh_data["faces"],
            pbr=mesh_data["pbr"],
            texture_size=0,       # auto: 2048 @1024, 1024 @512
            component_filter=0,   # remove tiny floating islands
        )
        output_glb.write_bytes(glb_bytes)
        textured = True
    else:
        # Geometry-only path: use trimesh with a clay PBR material
        import trimesh
        from trimesh.visual.material import PBRMaterial

        vertices = [
            [mesh_data["vertices"][i],
             mesh_data["vertices"][i + 2],
             -mesh_data["vertices"][i + 1]]
            for i in range(0, len(mesh_data["vertices"]), 3)
        ]
        faces = [
            [mesh_data["faces"][i],
             mesh_data["faces"][i + 1],
             mesh_data["faces"][i + 2]]
            for i in range(0, len(mesh_data["faces"]), 3)
        ]

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.fix_normals()
        trimesh.repair.fill_holes(mesh)
        vnormals = mesh.vertex_normals
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                               vertex_normals=vnormals)
        mesh.visual.material = PBRMaterial(
            metallicFactor=0.1,
            roughnessFactor=0.7,
            baseColorFactor=[220, 220, 220, 255],
            doubleSided=True,
        )
        mesh.export(output_glb, file_type="glb")
        textured = False

    return {
        "backend": "trellis2cpp",
        "model_set": "rms80/trellis2cpp pr-1",
        "resolution": resolution,
        "mode": mode_label,
        "textured": textured,
    }


def repair_meshfix(input_obj: Path, output_obj: Path, progress: Progress | None = None) -> dict[str, int]:
    """Repair an Instant Meshes output without modifying the original OBJ.

    MeshFix operates on triangles, so this is intentionally an opt-in repair
    stage for broken/non-manifold regions rather than part of normal retopo.
    """
    try:
        import numpy as np
        import pymeshfix
        import trimesh
    except ImportError as exc:
        raise RuntimeError("MeshFix is not installed. Run start.ps1 -Install to add the optional repair dependency.") from exc
    if progress:
        progress("MeshFix: repairing holes and self-intersections")
    source = trimesh.load(input_obj, force="mesh", process=False)
    if not isinstance(source, trimesh.Trimesh) or source.faces.size == 0:
        raise RuntimeError("MeshFix could not read a triangle mesh from Instant Meshes output")
    before_vertices, before_faces = len(source.vertices), len(source.faces)
    before_bounds = source.bounds.copy()
    fixer = pymeshfix.MeshFix(np.asarray(source.vertices), np.asarray(source.faces), verbose=False)
    # Preserve disconnected pieces such as accessories and do not bridge them.
    fixer.repair(joincomp=False, remove_smallest_components=False)
    repaired = trimesh.Trimesh(vertices=np.asarray(fixer.points), faces=np.asarray(fixer.faces), process=False)
    if repaired.faces.size == 0 or not np.isfinite(repaired.vertices).all():
        raise RuntimeError("MeshFix produced an invalid mesh")
    if len(repaired.faces) > before_faces * 3 or len(repaired.vertices) > before_vertices * 3:
        raise RuntimeError("MeshFix repair was rejected because it expanded the mesh too much")
    source_diagonal = float(np.linalg.norm(before_bounds[1] - before_bounds[0]))
    repaired_diagonal = float(np.linalg.norm(repaired.bounds[1] - repaired.bounds[0]))
    if progress and source_diagonal > 0:
        progress(f"MeshFix bounds: {source_diagonal:.4g} → {repaired_diagonal:.4g} ({repaired_diagonal / source_diagonal:.0%})")
    # Repairing self-intersecting scraps can legitimately trim an extreme
    # broken fragment. Guard against unexpected expansion, but permit a
    # bounded shrink when the user explicitly opts into MeshFix.
    if source_diagonal > 0 and (repaired_diagonal < source_diagonal * 0.5 or repaired_diagonal > source_diagonal * 1.25):
        raise RuntimeError("MeshFix repair was rejected because it changed the model bounds by more than the safe limit")
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    repaired.export(output_obj, file_type="obj")
    return {
        "input_vertices": before_vertices,
        "input_faces": before_faces,
        "output_vertices": len(repaired.vertices),
        "output_faces": len(repaired.faces),
    }


def run_instant_meshes(input_glb: Path, output_glb: Path, target_vertices: int, repair_holes: bool = False, progress: Progress | None = None) -> dict[str, Any]:
    """Retopologize a GLB via Instant Meshes' native batch mode.

    The official executable exposes this exact non-interactive workflow:
    extrinsic field (the default), boundary alignment, target vertex count,
    orientation solve, position solve, and extraction. The GUI's default
    extraction uses a quad-dominant mesh and zero smoothing passes.
    """
    exe = instant_meshes_exe()
    if exe is None:
        raise RuntimeError("Instant Meshes is not installed. Open Retopologize and download the Instant Meshes engine first.")
    if find_blender() is None:
        raise RuntimeError("Blender 4.2+ is required for the GLB-to-OBJ conversion and texture bake")

    temp_dir = output_glb.parent / f"instant-meshes-{output_glb.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    source_obj = temp_dir / "source.obj"
    result_obj = temp_dir / "remeshed.obj"
    repaired_obj = temp_dir / "remeshed-repaired.obj"
    repair_stats: dict[str, int] | None = None
    try:
        if progress:
            progress("Converting the full source GLB to OBJ")
        blender_export_obj(input_glb, source_obj, progress)
        if progress:
            progress("Instant Meshes: solving orientation and position fields")
        run_process([
            str(exe), "--output", str(result_obj), "--vertices", str(target_vertices),
            "--boundaries", "--dominant", "--smooth", "0", str(source_obj),
        ], progress=progress, timeout=3600)
        if not result_obj.exists() or result_obj.stat().st_size < 64:
            raise RuntimeError("Instant Meshes completed without writing remeshed.obj")
        mesh_for_bake = result_obj
        if repair_holes:
            repair_stats = repair_meshfix(result_obj, repaired_obj, progress)
            mesh_for_bake = repaired_obj
        if progress:
            progress("UV unwrapping and baking source textures onto the quad mesh")
        blender_bake_retopology(input_glb, mesh_for_bake, output_glb, progress)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return {
        "backend": "instant-meshes",
        "target_vertices": target_vertices,
        "extrinsic": True,
        "align_to_boundaries": True,
        "texture_transfer": "blender-selected-to-active-bake",
        "meshfix_repair": repair_stats,
    }


def run_trellis_ai_refine(input_glb: Path, source_image: Path, output_glb: Path, target_vertices: int, repair_holes: bool = False, gpu_first: bool = False, progress: Progress | None = None) -> dict[str, Any]:
    """Image-guided FP8 refinement, CuMesh reconstruction, and PBR texturing."""
    if not trellis_refiner_ready():
        raise RuntimeError("TRELLIS.2 FP8 is not downloaded. Download the generation model first.")
    worker = ROOT / "backend" / "workers" / "trellis2_fp8_worker.py"
    def run_worker(use_gpu_first: bool, safe_reconstruction: bool = False) -> None:
        spec = json.dumps({
            "operation": "refine", "runtime": str(trellis_refiner_dir()),
            "input_glb": str(input_glb), "source_image": str(source_image),
            "output_glb": str(output_glb), "resolution": 1024,
            "target_faces": target_vertices, "repair_holes": repair_holes, "gpu_first": use_gpu_first,
            "safe_reconstruction": safe_reconstruction,
        })
        run_process([sys.executable, str(worker), spec], progress=progress, timeout=7200)

    used_gpu_first = gpu_first
    used_safe_reconstruction = False
    try:
        run_worker(gpu_first)
    except RuntimeError as exc:
        quad_crashed = "3221225477" in str(exc) or "QEF" in str(exc)
        if not gpu_first and not quad_crashed:
            raise
        if quad_crashed and not gpu_first:
            if progress:
                progress("Quad reconstruction could not complete; retrying compatible reconstruction")
            used_safe_reconstruction = True
            run_worker(False, safe_reconstruction=True)
            return {
                "backend": "trellis2-fp8-refine-reconstruct-texture",
                "source_image_guided": True,
                "model": "visualbruno/TRELLIS.2-4B-FP8",
                "target_vertices": target_vertices,
                "resolution": 1024,
                "gpu_first_requested": gpu_first,
                "gpu_first_used": False,
                "reconstruction": "compatible",
                "texture_transfer": "trellis2-image-conditioned-pbr",
            }
        # A full-resident FP8 pipeline can overflow or crash native CUDA
        # extensions on 8-GB cards. Retry the same job with staged offloading
        # so an experimental speed option never loses the user's work.
        if gpu_first and progress:
            progress("GPU-first could not complete; retrying in standard mode")
        used_gpu_first = False
        try:
            run_worker(False)
        except RuntimeError as retry_exc:
            if not (quad_crashed or "3221225477" in str(retry_exc) or "QEF" in str(retry_exc)):
                raise
            if progress:
                progress("Quad reconstruction could not complete; retrying compatible reconstruction")
            used_safe_reconstruction = True
            run_worker(False, safe_reconstruction=True)
    return {
        "backend": "trellis2-fp8-refine-reconstruct-texture",
        "source_image_guided": True,
        "model": "visualbruno/TRELLIS.2-4B-FP8",
        "target_vertices": target_vertices,
        "resolution": 1024,
        "gpu_first_requested": gpu_first,
        "gpu_first_used": used_gpu_first,
        "reconstruction": "compatible" if used_safe_reconstruction else "quad",
        "texture_transfer": "trellis2-image-conditioned-pbr",
    }


def run_trellis_fp8(input_image: Path, output_glb: Path, resolution: int, progress: Progress | None = None) -> dict[str, Any]:
    """Generate a PBR GLB through the same FP8 TRELLIS.2 runtime as refine."""
    if not trellis_refiner_ready():
        raise RuntimeError("TRELLIS.2 FP8 is not downloaded. Download the generation model first.")
    worker = ROOT / "backend" / "workers" / "trellis2_fp8_worker.py"
    actual_resolution = 512 if resolution <= 512 else 1024
    spec = json.dumps({
        "operation": "generate", "runtime": str(trellis_refiner_dir()),
        "source_image": str(input_image), "output_glb": str(output_glb),
        "resolution": actual_resolution,
    })
    run_process([sys.executable, str(worker), spec], progress=progress, timeout=7200)
    return {"backend": "trellis2-fp8", "model": "visualbruno/TRELLIS.2-4B-FP8", "resolution": actual_resolution, "textured": True}


def run_paint(input_glb: Path, output_glb: Path, settings: dict[str, Any], progress: Progress | None = None) -> dict[str, Any]:
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required for texture baking but was not found")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    spec = json.dumps({"op": "paint", "input": str(input_glb), "output": str(output_glb), "settings": settings})
    run_process([str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec], progress=progress, timeout=900)
    return {"backend": "blender-texture-bake", "texture_size": int(settings.get("texture_size", 1024))}
