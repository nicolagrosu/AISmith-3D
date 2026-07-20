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
    """Complete model set required by the standalone trellis.cpp runtime."""
    return [trellis_model_dir() / filename for filename in (
        "birefnet.gguf", "dinov3.gguf", "shape_dec.gguf", "shape_flow_512.gguf",
        "shape_flow_1024.gguf", "ss_dec.gguf", "ss_flow.gguf", "tex_dec.gguf",
        "tex_flow_512.gguf", "tex_flow_1024.gguf",
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


def skintoken_repo() -> Path:
    return configured_path("SKINTOKEN_REPO_DIR", ROOT / "vendor" / "ComfyUI-SkinTokens")


def skintoken_model_dir() -> Path:
    return configured_path("SKINTOKEN_MODELS_DIR", ROOT / "studio-data" / "models" / "skintoken")


def mesh2motion_library() -> Path:
    return configured_path("MESH2MOTION_LIBRARY", ROOT / "vendor" / "mesh2motion-app" / "static" / "animations" / "human-base-animations.glb")


def autoremesher_exe() -> Path | None:
    configured = os.getenv("AUTOREMESHER_EXE", "").strip()
    candidates = [Path(configured)] if configured else []
    candidates += [ROOT / "vendor" / "autoremesher" / "release" / "autoremesher.exe", ROOT / "vendor" / "autoremesher" / "build" / "release" / "autoremesher.exe"]
    found = shutil.which("autoremesher")
    if found:
        candidates.append(Path(found))
    return next((path.resolve() for path in candidates if path.exists()), None)


def run_process(command: list[str], *, cwd: Path = ROOT, timeout: int = 3600, progress: Progress | None = None, env: dict[str, str] | None = None) -> str:
    if progress:
        progress(f"Launching {Path(command[0]).name}")
    merged = os.environ.copy()
    if env:
        merged.update(env)
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=merged)
    output = (completed.stdout + "\n" + completed.stderr).strip()
    if progress:
        for line in output.splitlines()[-12:]:
            if line.strip():
                progress(line.strip()[:500])
    if completed.returncode != 0:
        raise RuntimeError(f"{Path(command[0]).name} failed ({completed.returncode}): {output[-1800:]}")
    return output


def blender_convert(input_path: Path, output_path: Path, progress: Progress | None = None) -> None:
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required for GLB/OBJ conversion but was not found")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    spec = json.dumps({"op": "convert", "input": str(input_path), "output": str(output_path)})
    run_process([str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec], progress=progress, timeout=600)


def run_trellis(input_image: Path, output_glb: Path, resolution: int, progress: Progress | None = None) -> dict[str, Any]:
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
    command = [str(cli), "--models", str(trellis_model_dir()), "--res", str(resolution), "--require-gpu", "--seed", "0", "--host", "127.0.0.1", "--port", str(port)]
    if progress:
        progress("Starting private native trellis.cpp worker")
    server = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
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
            data={"seed": "0", "resolution": str(resolution)},
            timeout=7200,
        )
        if not response.ok:
            native_log = stop_server().strip()
            raise RuntimeError(f"trellis.cpp generation failed ({response.status_code}): {response.text[-500:]}\nNative runtime log:\n{native_log[-1800:]}")
        output_glb.write_bytes(response.content)
    finally:
        stop_server()
    return {"backend": "trellis.cpp", "model_set": "ilintar/trellis2-gguf", "resolution": resolution}


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


def run_autoremesher(input_glb: Path, output_glb: Path, target_faces: int, quality: str, progress: Progress | None = None) -> dict[str, Any]:
    exe = autoremesher_exe()
    if exe is None:
        raise RuntimeError("AutoRemesher executable was not found. Build vendor/autoremesher with Qt 5.15/MSVC, then set AUTOREMESHER_EXE.")
    temp_dir = output_glb.parent / f"remesh-{output_glb.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    source_obj = temp_dir / "source.obj"
    result_obj = temp_dir / "remeshed.obj"
    # Blender's GLB export is canonical; use its OBJ exporter for AutoRemesher's CLI boundary.
    blender = find_blender()
    worker = ROOT / "backend" / "workers" / "blender_obj_worker.py"
    if not worker.exists():
        raise RuntimeError("OBJ conversion worker is missing")
    spec = json.dumps({"input": str(input_glb), "output": str(source_obj)})
    run_process([str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec], progress=progress, timeout=600)
    quality_values = {"game": (1.0, 70.0, 0.0, 1.0), "balanced": (1.0, 70.0, 0.0, 0.5), "high": (1.0, 90.0, 0.0, 0.2)}
    edge, sharp, smooth, adapt = quality_values.get(quality, quality_values["game"])
    run_process([str(exe), "--input", str(source_obj), "--output", str(result_obj), "--report", str(temp_dir / "report.txt"), "--target-quads", str(max(250, target_faces // 2)), "--edge-scaling", str(edge), "--sharp-edge", str(sharp), "--smooth-normal", str(smooth), "--adaptivity", str(adapt)], progress=progress, timeout=1800)
    spec = json.dumps({"input": str(result_obj), "output": str(output_glb)})
    run_process([str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec], progress=progress, timeout=600)
    return {"backend": "autoremesher", "target_faces": target_faces, "quality": quality}


def run_skintoken(input_glb: Path, output_glb: Path, progress: Progress | None = None) -> dict[str, Any]:
    blender = find_blender()
    repo = skintoken_repo()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required for SkinToken rigging but was not found")
    if not (repo / "nodes.py").exists():
        raise RuntimeError(f"SkinToken repository is missing: {repo}")
    worker = ROOT / "backend" / "workers" / "skintoken_worker.py"
    spec = json.dumps({"repo": str(repo), "model_dir": str(skintoken_model_dir()), "input": str(input_glb), "output_dir": str(output_glb.parent), "output": str(output_glb)})
    env_path = str(blender.parent) + os.pathsep + os.environ.get("PATH", "")
    run_process([sys.executable, str(worker), "--", spec], progress=progress, timeout=3600, env={"PATH": env_path, "BLENDER_EXE": str(blender)})
    return {"backend": "skintoken", "format": "glb", "bone_convention": "articulated"}


def run_animate(input_glb: Path, output_glb: Path, animation: str, progress: Progress | None = None) -> dict[str, Any]:
    blender = find_blender()
    library = mesh2motion_library()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required for animation export but was not found")
    if not library.exists():
        raise RuntimeError(f"Mesh2Motion human animation library was not found: {library}")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    spec = json.dumps({"op": "animate", "input": str(input_glb), "output": str(output_glb), "library": str(library), "animation": animation})
    run_process([str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec], progress=progress, timeout=900)
    return {"backend": "mesh2motion", "animation": animation, "retargeted": True, "library": str(library)}


def run_paint(input_glb: Path, output_glb: Path, settings: dict[str, Any], progress: Progress | None = None) -> dict[str, Any]:
    blender = find_blender()
    if blender is None:
        raise RuntimeError("Blender 4.2+ is required for texture baking but was not found")
    worker = ROOT / "backend" / "workers" / "blender_worker.py"
    spec = json.dumps({"op": "paint", "input": str(input_glb), "output": str(output_glb), "settings": settings})
    run_process([str(blender), "--background", "--factory-startup", "--python", str(worker), "--", spec], progress=progress, timeout=900)
    return {"backend": "blender-texture-bake", "texture_size": int(settings.get("texture_size", 1024))}
