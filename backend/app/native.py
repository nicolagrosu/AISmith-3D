from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
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
    return [
        ("ggufs/dino_f16.gguf", "ggufs/dino_f16.gguf"),
        ("ggufs/ss_flow_f16.gguf", "ggufs/ss_flow_f16.gguf"),
        ("ggufs/ss_dec_f16.gguf", "ggufs/ss_dec_f16.gguf"),
        ("ggufs/slat_flow_f16.gguf", "ggufs/slat_flow_f16.gguf"),
        ("ggufs/shape_dec_f16.gguf", "ggufs/shape_dec_f16.gguf"),
    ]


def trellis_gguf_missing_files() -> list[Path]:
    root = trellis_gguf_model_dir()
    return [root / local for _, local in trellis_gguf_manifest() if not (root / local).is_file() or (root / local).stat().st_size == 0]


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
    command = [str(cli), "--models", str(trellis_model_dir()), "--res", str(resolution), "--bg-removal", "birefnet", "--require-gpu", "--seed", "0", "--host", "127.0.0.1", "--port", str(port)]
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
            progress("Running native Trellis image-to-3D with BiRefNet background removal")
        # trellis-server's multipart handler writes requests to a .png temp
        # file.  Convert WebP/JPEG uploads first so the bytes always match the
        # extension consumed by BiRefNet.
        with Image.open(input_image) as source_image:
            prepared = source_image.convert("RGBA") if "A" in source_image.getbands() else source_image.convert("RGB")
            png_handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            png_path = Path(png_handle.name)
            png_handle.close()
            prepared.save(png_path, format="PNG")
        try:
            image_handle = png_path.open("rb")
            response = requests.post(
                f"http://127.0.0.1:{port}/generate",
                files={"image": ("reference.png", image_handle, "image/png")},
                data={"seed": "0", "resolution": str(resolution), "bg_removal": "birefnet"},
                timeout=7200,
            )
            image_handle.close()
        finally:
            png_path.unlink(missing_ok=True)
        if not response.ok:
            native_log = stop_server().strip()
            raise RuntimeError(f"trellis.cpp generation failed ({response.status_code}): {response.text[-500:]}\nNative runtime log:\n{native_log[-1800:]}")
        output_glb.write_bytes(response.content)
    finally:
        stop_server()
    return {"backend": "trellis.cpp", "model_set": "ilintar/trellis2-gguf", "resolution": resolution}


def run_trellis_gguf(input_image: Path, output_glb: Path, resolution: int, progress: Progress | None = None) -> dict[str, Any]:
    """Run the trellis2cpp C++ pipeline directly in-process via ctypes."""
    from backend.app.trellis2_wrapper import Trellis2Wrapper
    import trimesh
    import random
    
    gguf_dir = trellis_gguf_model_dir() / "ggufs"
    dino = gguf_dir / "dino_f16.gguf"
    ss_flow = gguf_dir / "ss_flow_f16.gguf"
    ss_dec = gguf_dir / "ss_dec_f16.gguf"
    slat_flow = gguf_dir / "slat_flow_f16.gguf"
    shape_dec = gguf_dir / "shape_dec_f16.gguf"

    # Make sure required GGUFs exist
    for path in (dino, ss_flow, ss_dec):
        if not path.is_file():
            raise RuntimeError(f"Required model file is missing: {path.name}. Please run conversion first.")

    has_fine = (resolution > 512) and slat_flow.is_file() and shape_dec.is_file()
    
    from PIL import Image
    import io

    # Convert any input format (including WebP) to PNG before passing to C++
    with Image.open(input_image) as img:
        img_rgba = img.convert("RGBA")
        png_io = io.BytesIO()
        img_rgba.save(png_io, format="PNG")
        image_bytes = png_io.getvalue()
    
    if progress:
        progress("Loading Trellis2 C++ pipeline models...")
        
    wrapper = Trellis2Wrapper(ROOT / "vendor" / "trellis2cpp" / "build")
    pipeline = wrapper.load_pipeline(
        str(dino),
        str(ss_flow),
        str(ss_dec),
        str(slat_flow) if has_fine else None,
        str(shape_dec) if has_fine else None
    )
    
    try:
        def on_cpp_progress(stage, step, total):
            stages = {
                0: "Pre-processing reference image",
                1: "Encoding image with DINOv3",
                2: f"Sampling sparse structure (step {step}/{total})" if total > 0 else "Sampling sparse structure",
                3: "Decoding occupancy grid",
                4: f"Sampling shape SLAT (step {step}/{total})" if total > 0 else "Sampling shape SLAT",
                5: "Decoding fine shape geometry",
                6: "Extracting triangle mesh"
            }
            msg = stages.get(stage, "Running Trellis2 generation")
            if progress:
                progress(msg)

        if progress:
            progress("Running direct Trellis2 C++ image-to-3D generation...")
            
        seed = random.randint(0, 2**32 - 1)
        
        mesh_data = wrapper.generate(
            pipeline,
            image_bytes,
            seed=seed,
            steps=12,
            guidance=7.5,
            progress_cb=on_cpp_progress
        )
    finally:
        wrapper.free_pipeline(pipeline)

    if progress:
        progress("Building GLB mesh artifact...")
        
    vertices = []
    for i in range(0, len(mesh_data["vertices"]), 3):
        # Rotate Z-up (Trellis) to Y-up (gLTF/GLB): y_new = z, z_new = -y
        x = mesh_data["vertices"][i]
        y = mesh_data["vertices"][i+1]
        z = mesh_data["vertices"][i+2]
        vertices.append([x, z, -y])
        
    faces = []
    for i in range(0, len(mesh_data["faces"]), 3):
        faces.append([mesh_data["faces"][i], mesh_data["faces"][i+1], mesh_data["faces"][i+2]])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()
    trimesh.repair.fill_holes(mesh)
    
    # Extract the computed vertex normals and watertight geometry, and re-create Trimesh so they are exported
    vertex_normals = mesh.vertex_normals
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_normals=vertex_normals)
    
    # Assign a standard clay PBR material so the model viewer renders a solid, shaded mesh
    from trimesh.visual.material import PBRMaterial
    mesh.visual.material = PBRMaterial(
        metallicFactor=0.1,
        roughnessFactor=0.7,
        baseColorFactor=[220, 220, 220, 255],
        doubleSided=True
    )
    
    mesh.export(output_glb, file_type="glb")
    
    return {
        "backend": "trellis2cpp",
        "model_set": "rms80/trellis2cpp",
        "resolution": resolution,
        "mode": "fine (dual-grid)" if has_fine else "coarse (marching-cubes)"
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
