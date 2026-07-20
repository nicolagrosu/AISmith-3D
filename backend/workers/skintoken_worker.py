"""Direct SkinToken worker using the upstream model and headless Blender path."""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path


def install_folder_paths(model_dir: Path, input_dir: Path, output_dir: Path) -> None:
    module = types.ModuleType("folder_paths")
    module.models_dir = str(model_dir.parent)
    module.get_input_directory = lambda: str(input_dir)
    module.get_output_directory = lambda: str(output_dir)
    module.get_full_path = lambda category, name: str(model_dir / name)
    module.get_filename_list = lambda category: []
    sys.modules["folder_paths"] = module


def main() -> None:
    spec = json.loads(sys.argv[sys.argv.index("--") + 1])
    repo = Path(spec["repo"]).resolve()
    model_dir = Path(spec["model_dir"]).resolve()
    input_path = Path(spec["input"]).resolve()
    output_dir = Path(spec["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    install_folder_paths(model_dir, input_path.parent, output_dir)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("SkinToken requires a CUDA-enabled PyTorch installation and an NVIDIA GPU")
    import nodes
    checkpoint = spec.get("checkpoint", "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt")
    model = nodes.SkinTokensModelLoader().load_model(checkpoint)[0]
    output = nodes.SkinTokensGenerator().generate(
        model=model,
        input_mesh=str(input_path),
        top_k=5,
        top_p=0.95,
        temperature=1.0,
        repetition_penalty=2.0,
        num_beams=10,
        use_skeleton=False,
        use_transfer=True,
        use_postprocess=False,
        bone_names="articulated",
        output_format=".glb",
        bpy_server_mode="Headless (Blender)",
    )[0]
    result = Path(output)
    if not result.exists():
        raise RuntimeError(f"SkinToken reported success but did not create {result}")
    requested = Path(spec["output"])
    requested.parent.mkdir(parents=True, exist_ok=True)
    requested.write_bytes(result.read_bytes())
    print(json.dumps({"event": "log", "message": "SkinToken exported a rigged GLB; worker is exiting to release model VRAM"}), flush=True)


if __name__ == "__main__":
    main()
