import unittest
import io
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app import native
from backend import bootstrap


client = TestClient(app)


class ApiTests(unittest.TestCase):
    def test_health_and_capabilities(self):
        self.assertEqual(client.get("/api/health").json()["status"], "ok")
        payload = client.get("/api/system/capabilities").json()
        self.assertIn("gpu", payload)
        self.assertEqual(payload["gpu"]["model_variant"], "GGUF CUDA")


    def test_trellis_cpp_requires_all_native_ggufs(self):
        files = {path.name for path in native.trellis_model_files()}
        self.assertEqual(files, {"birefnet.gguf", "dinov3.gguf", "shape_dec.gguf", "shape_flow_512.gguf", "shape_flow_1024.gguf", "ss_dec.gguf", "ss_flow.gguf", "tex_dec.gguf", "tex_flow_512.gguf", "tex_flow_1024.gguf"})

    def test_trellis_downloader_matches_standalone_runtime(self):
        downloaded = {local for _, _, local in bootstrap.trellis_gguf_manifest()}
        required = {path.name for path in native.trellis_model_files()}
        self.assertEqual(downloaded, required)

    def test_trellis_input_letterboxing_preserves_aspect_with_transparent_padding(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "portrait.png"
            Image.new("RGB", (508, 800), (100, 80, 60)).save(source)
            with Image.open(io.BytesIO(native.prepare_trellis_input(source))) as prepared:
                self.assertEqual(prepared.size, (1024, 1024))
                alpha = prepared.getchannel("A")
                self.assertEqual(alpha.getbbox(), (187, 0, 837, 1024))
                self.assertEqual(alpha.getpixel((0, 0)), 0)
                self.assertEqual(alpha.getpixel((512, 512)), 255)
