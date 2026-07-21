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
TRELLIS_CPP_MODELS = "ilintar/trellis2-gguf"


class AISmithDownloadProgress(BaseTqdm):
    """Forward Hugging Face byte progress to the API as newline-delimited JSON."""

    file_name = ""
    file_index = 0
    file_count = 7

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
    """Geometry-only bundle for the standalone trellis.cpp generator.

    The app invokes trellis.cpp with ``--no-texture``.  Final PBR texturing is
    performed by the FP8 refinement worker, so downloading the three C++
    texture GGUFs would only consume disk space.
    """
    filenames = (
        "birefnet.gguf", "dinov3.gguf", "shape_dec.gguf", "shape_flow_512.gguf",
        "shape_flow_1024.gguf", "ss_dec.gguf", "ss_flow.gguf",
    )
    return [(TRELLIS_CPP_MODELS, filename, filename) for filename in filenames]


def download_model_files(model_root: Path) -> None:
    """Fetch the standalone geometry bundle file-by-file."""
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
                hf_hub_download(repo_id=repo, filename=remote, local_dir=model_root, tqdm_class=AISmithDownloadProgress)
                break
            except Exception as exc:
                if attempt == 3:
                    raise RuntimeError(f"Could not download {relative} after 3 attempts: {exc}") from exc
                print(f"[AISmith 3D] ({index}/{len(manifest)}) {relative} connection failed; retrying...")
                time.sleep(attempt * 2)


def download_trellis() -> None:
    if gpu_vram_mb() <= 0:
        raise RuntimeError("An NVIDIA GPU is required for the Trellis2 CUDA workflow")
    model_root = ROOT / "studio-data" / "models" / "trellis-cpp"
    model_root.mkdir(parents=True, exist_ok=True)
    print("[AISmith 3D] Downloading the trellis.cpp geometry workflow one file at a time...")
    download_model_files(model_root)
    missing = [model_root / relative for _, _, relative in trellis_gguf_manifest() if not (model_root / relative).is_file()]
    if missing:
        formatted = "\n  - ".join(path.name for path in missing)
        raise RuntimeError(f"Trellis2 model download is incomplete. Missing:\n  - {formatted}")
    print("[AISmith 3D] trellis.cpp geometry workflow is ready.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--download-model", choices=("trellis-cpp", "trellis2", "all"))
    args = parser.parse_args()
    if args.download_models:
        download_trellis()
    elif args.download_model:
        if args.download_model in {"trellis-cpp", "trellis2", "all"}:
            download_trellis()


if __name__ == "__main__":
    main()
