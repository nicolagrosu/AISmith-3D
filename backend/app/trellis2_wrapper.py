import os
import ctypes
from ctypes import c_int, c_char_p, c_void_p, c_uint64, c_float, POINTER
from pathlib import Path
from typing import Callable, Optional

# Progress callback type:
# typedef void (*t2_progress_fn)(void * user, int stage, int step, int total);
T2_PROGRESS_FN = ctypes.CFUNCTYPE(None, c_void_p, c_int, c_int, c_int)

class Trellis2Wrapper:
    def __init__(self, build_dir: Path):
        self.build_dir = build_dir
        bin_dir = build_dir / "bin" / "Release"
        rel_dir = build_dir / "Release"
        
        # Add DLL directories for Windows Python >= 3.8 dependency resolution
        if hasattr(os, "add_dll_directory"):
            for d in (bin_dir, rel_dir):
                if d.is_dir():
                    try:
                        os.add_dll_directory(str(d))
                    except Exception as e:
                        print(f"Warning: failed to add DLL directory {d}: {e}")
            for path_dir in os.environ.get("PATH", "").split(os.pathsep):
                if path_dir.strip():
                    p = Path(path_dir.strip())
                    if p.is_dir():
                        try:
                            os.add_dll_directory(str(p))
                        except Exception:
                            pass

        # Find paths for each library
        def find_lib(name):
            for d in (bin_dir, rel_dir):
                p = d / name
                if p.is_file():
                    return p
            raise FileNotFoundError(f"Shared library {name} not found in {bin_dir} or {rel_dir}")

        # Explicitly load dependencies in topological order to prevent resolution issues
        try:
            self.lib_base = ctypes.CDLL(str(find_lib("ggml-base.dll")))
            self.lib_cpu = ctypes.CDLL(str(find_lib("ggml-cpu.dll")))
            try:
                cuda_path = find_lib("ggml-cuda.dll")
                self.lib_cuda = ctypes.CDLL(str(cuda_path))
            except FileNotFoundError:
                self.lib_cuda = None
            self.lib = ctypes.CDLL(str(find_lib("trellis2.dll")))
        except Exception as e:
            raise RuntimeError(f"Failed to load C++ shared libraries: {e}")

        self._setup_signatures()

    def _setup_signatures(self):
        # int t2_abi_version(void);
        self.lib.t2_abi_version.argtypes = []
        self.lib.t2_abi_version.restype = c_int

        # Verify ABI version
        abi_ver = self.lib.t2_abi_version()
        if abi_ver != 1:
            raise RuntimeError(f"Trellis2 C++ API ABI version mismatch: expected 1, got {abi_ver}")

        # t2_pipeline * t2_pipeline_load(const char * dino_gguf, ...)
        self.lib.t2_pipeline_load.argtypes = [c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, c_int]
        self.lib.t2_pipeline_load.restype = c_void_p

        # void t2_pipeline_free(t2_pipeline * p);
        self.lib.t2_pipeline_free.argtypes = [c_void_p]
        self.lib.t2_pipeline_free.restype = None

        # const char * t2_pipeline_backend(t2_pipeline * p);
        self.lib.t2_pipeline_backend.argtypes = [c_void_p]
        self.lib.t2_pipeline_backend.restype = c_char_p

        # int t2_pipeline_is_fine(t2_pipeline * p);
        self.lib.t2_pipeline_is_fine.argtypes = [c_void_p]
        self.lib.t2_pipeline_is_fine.restype = c_int

        # t2_mesh_result * t2_generate(...)
        self.lib.t2_generate.argtypes = [
            c_void_p, c_void_p, c_int, c_uint64, c_int, c_float,
            T2_PROGRESS_FN, c_void_p, c_char_p, c_int
        ]
        self.lib.t2_generate.restype = c_void_p

        # int t2_mesh_n_verts(const t2_mesh_result * r);
        self.lib.t2_mesh_n_verts.argtypes = [c_void_p]
        self.lib.t2_mesh_n_verts.restype = c_int

        # int t2_mesh_n_tris (const t2_mesh_result * r);
        self.lib.t2_mesh_n_tris.argtypes = [c_void_p]
        self.lib.t2_mesh_n_tris.restype = c_int

        # const float * t2_mesh_verts  (const t2_mesh_result * r);
        self.lib.t2_mesh_verts.argtypes = [c_void_p]
        self.lib.t2_mesh_verts.restype = POINTER(c_float)

        # const float * t2_mesh_normals(const t2_mesh_result * r);
        self.lib.t2_mesh_normals.argtypes = [c_void_p]
        self.lib.t2_mesh_normals.restype = POINTER(c_float)

        # const int *   t2_mesh_tris   (const t2_mesh_result * r);
        self.lib.t2_mesh_tris.argtypes = [c_void_p]
        self.lib.t2_mesh_tris.restype = POINTER(c_int)

        # void t2_mesh_free(t2_mesh_result * r);
        self.lib.t2_mesh_free.argtypes = [c_void_p]
        self.lib.t2_mesh_free.restype = None

    def load_pipeline(self, dino: str, ss_flow: str, ss_dec: str, slat_flow: Optional[str] = None, shape_dec: Optional[str] = None) -> c_void_p:
        err_buf = ctypes.create_string_buffer(512)
        pipeline = self.lib.t2_pipeline_load(
            dino.encode('utf-8') if dino else None,
            ss_flow.encode('utf-8') if ss_flow else None,
            ss_dec.encode('utf-8') if ss_dec else None,
            slat_flow.encode('utf-8') if slat_flow else None,
            shape_dec.encode('utf-8') if shape_dec else None,
            err_buf,
            len(err_buf)
        )
        if not pipeline:
            raise RuntimeError(f"t2_pipeline_load failed: {err_buf.value.decode('utf-8', errors='ignore')}")
        return pipeline

    def free_pipeline(self, pipeline: c_void_p):
        self.lib.t2_pipeline_free(pipeline)

    def is_fine(self, pipeline: c_void_p) -> bool:
        return bool(self.lib.t2_pipeline_is_fine(pipeline))

    def backend_name(self, pipeline: c_void_p) -> str:
        return self.lib.t2_pipeline_backend(pipeline).decode('utf-8')

    def generate(self, pipeline: c_void_p, image_bytes: bytes, seed: int, steps: int = 12, guidance: float = 7.5,
                 progress_cb: Optional[Callable[[int, int, int], None]] = None) -> dict:
        
        # Keep reference to callback to prevent GC reclamation during C++ execution
        c_cb = None
        if progress_cb:
            def inner_cb(user, stage, step, total):
                try:
                    progress_cb(stage, step, total)
                except Exception as e:
                    print(f"Error in python progress callback: {e}")
            c_cb = T2_PROGRESS_FN(inner_cb)

        err_buf = ctypes.create_string_buffer(512)
        mesh_res = self.lib.t2_generate(
            pipeline,
            image_bytes,
            len(image_bytes),
            c_uint64(seed),
            c_int(steps),
            c_float(guidance),
            c_cb,
            None,
            err_buf,
            len(err_buf)
        )
        if not mesh_res:
            raise RuntimeError(f"t2_generate failed: {err_buf.value.decode('utf-8', errors='ignore')}")

        try:
            nv = self.lib.t2_mesh_n_verts(mesh_res)
            nt = self.lib.t2_mesh_n_tris(mesh_res)
            
            # Read buffers
            verts_ptr = self.lib.t2_mesh_verts(mesh_res)
            normals_ptr = self.lib.t2_mesh_normals(mesh_res)
            tris_ptr = self.lib.t2_mesh_tris(mesh_res)

            # Copy data into python arrays
            verts = [verts_ptr[i] for i in range(nv * 3)]
            normals = [normals_ptr[i] for i in range(nv * 3)]
            tris = [tris_ptr[i] for i in range(nt * 3)]
        finally:
            self.lib.t2_mesh_free(mesh_res)

        return {
            "vertices": verts,
            "normals": normals,
            "faces": tris
        }
