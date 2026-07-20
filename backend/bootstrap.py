"""First-run native dependency/model preparation.

This is intentionally separate from the API process. It may download several
gigabytes, and a failed download must never leave a partially loaded GPU worker.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from tqdm.auto import tqdm as BaseTqdm


ROOT = Path(__file__).resolve().parents[1]
TRELLIS_GGUF_MODELS = "Aero-Ex/Trellis2-GGUF"
DINOV3_MODELS = "Aero-Ex/Dinov3"
BIREFNET_MODEL = "ZhengPeng7/BiRefNet"
# LocalAI-io prebuilt f16 GGUFs (geometry + texture pipeline)
LOCALAI_T2   = "LocalAI-io/TRELLIS.2-4B-GGUF"
BIREFNET_REQUIRED_FILES = ("config.json", "BiRefNet_config.py", "birefnet.py", "model.safetensors")


class AISmithDownloadProgress(BaseTqdm):
    """Forward Hugging Face byte progress to the API as newline-delimited JSON."""

    file_name = ""
    file_index = 0
    file_count = 10

    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)
        self._last_reported = -1

    def _report(self, force: bool = False) -> None:
        downloaded = int(self.n or 0)
        total = int(self.total or 0)
        if not force and total and downloaded - self._last_reported < max(1024 * 1024, total // 200):
            return
        self._last_reported = downloaded
        print(json.dumps({"event": "file_progress", "file": self.file_name, "index": self.file_index, "count": self.file_count, "bytes": downloaded, "total": total}), flush=True)

    def update(self, n=1):
        result = super().update(n)
        self._report()
        return result

    def close(self):
        self._report(force=True)
        return super().close()


def gpu_vram_mb() -> int:
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True, timeout=5)
        return int(float(output.splitlines()[0].strip()))
    except Exception:
        return 0


def download_release(url: str, destination: Path) -> None:
    """Use Windows curl for GitHub release redirects, retrying/resuming partial data."""
    partial = destination.with_suffix(destination.suffix + ".part")
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        raise RuntimeError("Windows curl.exe was not found; install the trellis.cpp CUDA release manually.")
    command = [curl, "--fail", "--location", "--retry", "5", "--retry-all-errors", "--retry-delay", "3", "--continue-at", "-", "--output", str(partial), url]
    result = subprocess.run(command, text=True, capture_output=True, timeout=1800)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-600:]
        raise RuntimeError(f"Could not download trellis.cpp CUDA runtime with curl: {detail}")
    if not partial.exists() or partial.stat().st_size < 1024 * 1024:
        raise RuntimeError("trellis.cpp CUDA runtime download was incomplete")
    partial.replace(destination)


def trellis_gguf_manifest() -> list[tuple[str, str, str]]:
    """The Q4 files resolved by the GGUF workflow's current model manager.

    Each row is (repository, remote filename, local relative filename).  This
    is intentionally explicit: end users see every file and resume a single
    interrupted file instead of one opaque snapshot download.
    """
    components = [
        ("decoders/Stage1/ss_dec_conv3d_16l8_fp16", "safetensors"),
        ("refiner/ss_flow_img_dit_1_3B_64_bf16", "gguf"),
        ("decoders/Stage2/shape_dec_next_dc_f16c32_fp16", "safetensors"),
        ("shape/slat_flow_img2shape_dit_1_3B_512_bf16", "gguf"),
        ("shape/slat_flow_img2shape_dit_1_3B_1024_bf16", "gguf"),
        ("decoders/Stage2/tex_dec_next_dc_f16c32_fp16", "safetensors"),
        ("texture/slat_flow_imgshape2tex_dit_1_3B_512_bf16", "gguf"),
        ("texture/slat_flow_imgshape2tex_dit_1_3B_1024_bf16", "gguf"),
        ("encoders/shape_enc_next_dc_f16c32_fp16", "safetensors"),
    ]
    files: list[tuple[str, str, str]] = [(TRELLIS_GGUF_MODELS, "pipeline.json", "pipeline.json")]
    dino_prefix = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    files += [(DINOV3_MODELS, f"{dino_prefix}/{name}", f"dinov3/{dino_prefix}/{name}") for name in ("config.json", "model.safetensors", "preprocessor_config.json")]
    for path, kind in components:
        suffixes = (".json", ".safetensors") if kind == "safetensors" else (".json", "_Q4_K_M.gguf")
        for suffix in suffixes:
            files.append((TRELLIS_GGUF_MODELS, f"{path}{suffix}", f"{path}{suffix}"))
    # Prebuilt texture GGUFs from LocalAI-io (direct f16 download, no conversion needed)
    for tex_file in (
        "shape_enc_f16.gguf",
        "tex_dec_f16.gguf",
        "tex_slat_flow_512_f16.gguf",
        "tex_slat_flow_1024_f16.gguf",
    ):
        files.append((LOCALAI_T2, tex_file, f"ggufs/{tex_file}"))
    return files


def download_model_files(model_root: Path) -> None:
    """Fetch the direct Q4 workflow bundle file-by-file with resumable status."""
    from huggingface_hub import hf_hub_download

    manifest = trellis_gguf_manifest()
    for index, (repo, remote, relative) in enumerate(manifest, start=1):
        path = model_root / relative
        if path.is_file() and path.stat().st_size > 0:
            print(f"[AISmith 3D] ({index}/{len(manifest)}) {relative} already present")
            continue
        for attempt in range(1, 4):
            try:
                AISmithDownloadProgress.file_name = relative
                AISmithDownloadProgress.file_index = index
                AISmithDownloadProgress.file_count = len(manifest)
                print(f"[AISmith 3D] ({index}/{len(manifest)}) Downloading {relative} (attempt {attempt}/3)...")
                # The upstream pipeline looks for DINO in
                # models/Trellis2/dinov3/facebook/..., whereas the HF filename
                # itself begins at facebook/.  Keep that one repository nested.
                destination_root = model_root / "dinov3" if repo == DINOV3_MODELS else model_root
                hf_hub_download(repo_id=repo, filename=remote, local_dir=destination_root, tqdm_class=AISmithDownloadProgress)
                break
            except Exception as exc:
                if attempt == 3:
                    raise RuntimeError(f"Could not download {relative} after 3 attempts: {exc}") from exc
                print(f"[AISmith 3D] ({index}/{len(manifest)}) {relative} connection failed; retrying...")
                time.sleep(attempt * 2)


def download_trellis() -> None:
    if gpu_vram_mb() <= 0:
        raise RuntimeError("An NVIDIA GPU is required for the Trellis2 GGUF CUDA workflow")
    model_root = ROOT / "studio-data" / "models" / "Trellis2"
    model_root.mkdir(parents=True, exist_ok=True)
    print("[AISmith 3D] Downloading the Trellis2 GGUF Q4 workflow bundle one file at a time...")
    download_model_files(model_root)
    birefnet_root = ROOT / "studio-data" / "models" / "BiRefNet"
    birefnet_weights = birefnet_root / "model.safetensors"
    birefnet_complete = birefnet_weights.is_file() and birefnet_weights.stat().st_size >= 100 * 1024 * 1024 and all((birefnet_root / file).is_file() for file in BIREFNET_REQUIRED_FILES)
    if not birefnet_complete:
        from huggingface_hub import snapshot_download
        for attempt in range(1, 4):
            try:
                print(f"[AISmith 3D] Downloading public BiRefNet background-removal model (attempt {attempt}/3)...")
                AISmithDownloadProgress.file_name = "BiRefNet/model.safetensors"
                AISmithDownloadProgress.file_index = len(trellis_gguf_manifest()) + 1
                AISmithDownloadProgress.file_count = len(trellis_gguf_manifest()) + 1
                # Serial transfer avoids the concurrent HEAD requests that
                # some Windows/home connections reset.  The Hub cache resumes
                # individual files automatically after a retry.
                snapshot_download(
                    repo_id=BIREFNET_MODEL,
                    local_dir=birefnet_root,
                    tqdm_class=AISmithDownloadProgress,
                    max_workers=1,
                )
                break
            except Exception as exc:
                if attempt == 3:
                    raise RuntimeError(f"Could not download BiRefNet after 3 attempts: {exc}") from exc
                print(f"[AISmith 3D] BiRefNet connection failed; retrying ({attempt}/3)...")
                time.sleep(attempt * 3)

    missing = [model_root / relative for _, _, relative in trellis_gguf_manifest() if not (model_root / relative).is_file()]
    missing.extend(birefnet_root / file for file in BIREFNET_REQUIRED_FILES if not (birefnet_root / file).is_file())
    if not birefnet_weights.is_file() or birefnet_weights.stat().st_size < 100 * 1024 * 1024:
        missing.append(birefnet_weights)
    if missing:
        formatted = "\n  - ".join(path.name for path in missing)
        raise RuntimeError(f"Trellis2 GGUF download is incomplete. Missing:\n  - {formatted}")
    print("[AISmith 3D] Trellis2 GGUF Q4 workflow bundle is ready.")


def download_trellis_texture() -> None:
    """Download only the 4 texture GGUF files needed for PBR output."""
    from huggingface_hub import hf_hub_download
    model_root = ROOT / "studio-data" / "models" / "Trellis2"
    gguf_dir = model_root / "ggufs"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    tex_files = [
        (LOCALAI_T2, "shape_enc_f16.gguf"),
        (LOCALAI_T2, "tex_dec_f16.gguf"),
        (LOCALAI_T2, "tex_slat_flow_512_f16.gguf"),
        (LOCALAI_T2, "tex_slat_flow_1024_f16.gguf"),
    ]
    for index, (repo, filename) in enumerate(tex_files, start=1):
        dest = gguf_dir / filename
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"[AISmith 3D] ({index}/{len(tex_files)}) {filename} already present")
            continue
        for attempt in range(1, 4):
            try:
                AISmithDownloadProgress.file_name = filename
                AISmithDownloadProgress.file_index = index
                AISmithDownloadProgress.file_count = len(tex_files)
                print(f"[AISmith 3D] ({index}/{len(tex_files)}) Downloading {filename} from {repo} (attempt {attempt}/3)...")
                hf_hub_download(repo_id=repo, filename=filename, local_dir=gguf_dir, tqdm_class=AISmithDownloadProgress)
                break
            except Exception as exc:
                if attempt == 3:
                    raise RuntimeError(f"Could not download {filename} after 3 attempts: {exc}") from exc
                print(f"[AISmith 3D] ({index}/{len(tex_files)}) {filename} connection failed; retrying...")
                time.sleep(attempt * 2)
    print("[AISmith 3D] Trellis2 texture GGUFs are ready.")


def download_skintoken() -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is not available for SkinToken model download")
    from huggingface_hub import hf_hub_download, snapshot_download
    target = Path(os.getenv("SKINTOKEN_MODELS_DIR", ROOT / "studio-data/models/skintoken")).resolve()
    target.mkdir(parents=True, exist_ok=True)
    print("[AISmith 3D] Downloading/verifying SkinToken checkpoints...")
    for filename in ("experiments/skin_vae_2_10_32768/last.ckpt", "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"):
        hf_hub_download(repo_id="VAST-AI/SkinTokens", filename=filename, local_dir=target)
    snapshot_download(repo_id="Qwen/Qwen3-0.6B", local_dir=target / "models/Qwen3-0.6B", ignore_patterns=["*.bin", "*.safetensors"])
    print("[AISmith 3D] SkinToken checkpoints are ready.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--download-model", choices=("trellis2", "trellis2-texture", "skintoken", "all"))
    args = parser.parse_args()
    if args.download_models:
        download_trellis()
        download_skintoken()
    elif args.download_model:
        if args.download_model in {"trellis2", "all"}:
            download_trellis()
        if args.download_model == "trellis2-texture":
            download_trellis_texture()
        if args.download_model in {"skintoken", "all"}:
            download_skintoken()


if __name__ == "__main__":
    main()
