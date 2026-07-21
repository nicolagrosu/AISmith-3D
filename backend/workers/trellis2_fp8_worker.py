"""Isolated FP8 TRELLIS.2 generation and mesh-refinement worker.

This intentionally invokes the ComfyUI wrapper's Python nodes without starting
ComfyUI's server.  The model stack is loaded in this one-shot process so CUDA
memory is returned to Windows as soon as the job exits.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path


def _load_wrapper(wrapper_root: Path):
    init = wrapper_root / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "aismith_trellis2", init, submodule_search_locations=[str(wrapper_root)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the managed TRELLIS.2 package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return sys.modules["aismith_trellis2.nodes"]


def _load_nodes(runtime: Path):
    comfy = runtime / "ComfyUI"
    wrapper = comfy / "custom_nodes" / "ComfyUI-Trellis2"
    if not (comfy / "folder_paths.py").is_file() or not (wrapper / "nodes.py").is_file():
        raise RuntimeError("TRELLIS.2 FP8 is not installed. Download the model from Generate first.")
    sys.path.insert(0, str(comfy))
    import folder_paths

    # ComfyUI removed ``set_base_directory``.  This worker loads only the
    # wrapper nodes, so pointing its model registry at our managed runtime is
    # sufficient and works with both current and older ComfyUI builds.
    models_dir = str(runtime / "models")
    folder_paths.models_dir = models_dir
    for name, (paths, extensions) in folder_paths.folder_names_and_paths.items():
        folder_paths.folder_names_and_paths[name] = (
            [path.replace(str(comfy / "models"), models_dir) for path in paths], extensions
        )
    folder_paths.set_output_directory(str(runtime / "output"))
    folder_paths.set_temp_directory(str(runtime / "temp"))
    folder_paths.set_input_directory(str(runtime / "input"))
    return _load_wrapper(wrapper)


def _load_pipeline(nodes, gpu_first: bool = False):
    print("Loading TRELLIS.2-4B FP8", flush=True)
    # Current ComfyUI-Trellis2 unconditionally copies a helper config into
    # this directory before it checks the FP8 option.  The config itself is
    # tiny, but its parent needs to exist on a clean install.
    import folder_paths
    Path(folder_paths.models_dir, "microsoft", "TRELLIS.2-4B").mkdir(parents=True, exist_ok=True)
    return nodes.Trellis2LoadModel().process(
        modelname="visualbruno/TRELLIS.2-4B-FP8", backend="sdpa", device="cuda",
        low_vram=not gpu_first, keep_models_loaded=gpu_first, conv_backend="flex_gemm",
        sparse_backend="flash_attn", use_reconviagen=False,
    )[0]


def _image_tensor(nodes, path: Path):
    from PIL import Image

    with Image.open(path) as image:
        return nodes.pil2tensor(image.convert("RGBA"))


def _export(trimesh, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    trimesh.export(output, file_type="glb")
    if not output.is_file() or output.stat().st_size < 64:
        raise RuntimeError("TRELLIS.2 did not produce a valid GLB")


def _texture(nodes, pipeline, image, trimesh, resolution: int):
    print("Generating PBR textures for final geometry", flush=True)
    return nodes.Trellis2MeshTexturing().process(
        pipeline, image, trimesh, random.randint(0, 0x7FFFFFFF),
        12, 3.0, 0.20, 3.0, resolution, 2048, "OPAQUE", False,
        0.0, 0.90, 1, False, False, 60.0, "euler", "telea", False,
        0.0, 4, 1.0,
    )[0]


def _generate(nodes, pipeline, image, output: Path, resolution: int) -> None:
    preset = "512" if resolution <= 512 else "1024_cascade"
    print(f"Generating {preset} geometry and PBR material", flush=True)
    mesh, _ = nodes.Trellis2MeshWithVoxelGenerator().process(
        pipeline, image, random.randint(0, 0x7FFFFFFF), preset,
        12, 12, 12, 49152, 1, 32, True, True, "euler", True, 1,
        "flood_fill", True,
    )
    print("Exporting textured GLB", flush=True)
    # This is the wrapper's native O-Voxel PBR export, preserving the generated
    # texture latent rather than baking a legacy source texture in Blender.
    glb = nodes.Trellis2OvoxelExportToGLB().process(
        mesh, 512 if resolution <= 512 else 1024, 2048, 250000
    )[0]
    _export(glb, output)


def _refine(nodes, pipeline, image, input_mesh: Path, output: Path, resolution: int, target_faces: int, repair_holes: bool, safe_reconstruction: bool = False) -> None:
    print("Loading source mesh", flush=True)
    source = nodes.Trellis2LoadMesh().load(str(input_mesh))[0]
    print(f"Refining geometry at {resolution}", flush=True)
    refined, _ = nodes.Trellis2MeshRefiner().process(
        pipeline, source, image, random.randint(0, 0x7FFFFFFF), resolution,
        12, 6.5, 0.05, 4.0, 1, 3.0, 0.20, 3.0, 49152, False, 16,
        0.10, 1.0, 0.0, 0.90, True, 1, "euler", False, 0.0, 4, 1.0,
    )
    if safe_reconstruction:
        print("Reconstructing compatible CuMesh surface", flush=True)
        rebuilt = nodes.Trellis2ReconstructMesh().process(refined, 1.0, 512)[0]
    else:
        print("Reconstructing clean CuMesh surface", flush=True)
        rebuilt = nodes.Trellis2ReconstructMeshWithQuad().process(
            refined, 1.0, resolution, True, True
        )[0]
    print(f"Simplifying reconstructed mesh to {target_faces:,} faces", flush=True)
    rebuilt = nodes.Trellis2SimplifyMesh().process(rebuilt, target_faces, "Cumesh")[0]
    if repair_holes:
        print("Filling holes with Meshlib", flush=True)
        rebuilt = nodes.Trellis2FillHolesNicelyWithMeshlib().process(rebuilt)[0]
    final_mesh = nodes.Trellis2MeshWithVoxelToTrimesh().process(rebuilt, "90 degrees")[0]
    textured = _texture(nodes, pipeline, image, final_mesh, resolution)
    _export(textured, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    job = json.loads(parser.parse_args().spec)
    runtime = Path(job["runtime"]).resolve()
    output = Path(job["output_glb"]).resolve()
    nodes = _load_nodes(runtime)
    pipeline = _load_pipeline(nodes, bool(job.get("gpu_first", False)))
    image = _image_tensor(nodes, Path(job["source_image"]))
    try:
        if job["operation"] == "generate":
            _generate(nodes, pipeline, image, output, int(job["resolution"]))
        elif job["operation"] == "refine":
            _refine(nodes, pipeline, image, Path(job["input_glb"]), output,
                    int(job["resolution"]), int(job["target_faces"]), bool(job.get("repair_holes")), bool(job.get("safe_reconstruction", False)))
        else:
            raise ValueError(f"Unknown TRELLIS.2 operation: {job['operation']}")
    finally:
        try:
            import torch
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    print("TRELLIS.2 job complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
