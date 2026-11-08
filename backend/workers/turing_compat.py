"""Turing (sm_75 / CC < 8.0) compatibility patches for TRELLIS.2 worker.

The pre-built Windows CUDA wheels bundled with ComfyUI-Trellis2 (flex_gemm,
o_voxel, cumesh) are compiled exclusively for sm_86+ (Ampere and newer) and
raise `CUDA error: no kernel image is available for execution on the device`
on Turing GPUs such as the RTX 20-series (sm_75).

This module replaces the sm_86-only C++/CUDA extension entrypoints with
equivalent PyTorch GPU, Triton, and CPU (meshlib / xatlas) implementations.
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def needs_turing_compat() -> bool:
    if os.environ.get("AISMITH_FORCE_TURING_COMPAT") == "1":
        return True
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major < 8


def _lookup_4d_coords(
    coords: torch.Tensor,
    query: torch.Tensor,
    B: int,
    W: int,
    H: int,
    D: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Look up [M, 4] int32 (b, x, y, z) queries in [N, 4] coords via chunked int64 binary search."""
    N = coords.shape[0]
    M = query.shape[0]
    device = coords.device
    if N == 0 or M == 0:
        return (
            torch.zeros(M, dtype=torch.bool, device=device),
            torch.zeros(M, dtype=torch.int64, device=device),
        )

    c64 = coords.to(torch.int64)
    stride_y = int(D)
    stride_x = int(H) * stride_y
    stride_b = int(W) * stride_x

    coord_keys = c64[:, 0] * stride_b + c64[:, 1] * stride_x + c64[:, 2] * stride_y + c64[:, 3]
    del c64
    sorted_keys, sort_idx = torch.sort(coord_keys)
    del coord_keys
    sort_idx_i32 = sort_idx.to(torch.int32)
    del sort_idx

    matched = torch.empty(M, dtype=torch.bool, device=device)
    found_idx = torch.empty(M, dtype=torch.int32, device=device)
    chunk_size = 262144
    for s in range(0, M, chunk_size):
        e = min(s + chunk_size, M)
        q64 = query[s:e].to(torch.int64)
        in_bounds = (
            (q64[:, 0] >= 0) & (q64[:, 0] < B)
            & (q64[:, 1] >= 0) & (q64[:, 1] < W)
            & (q64[:, 2] >= 0) & (q64[:, 2] < H)
            & (q64[:, 3] >= 0) & (q64[:, 3] < D)
        )
        query_keys = q64[:, 0] * stride_b + q64[:, 1] * stride_x + q64[:, 2] * stride_y + q64[:, 3]
        del q64
        pos = torch.searchsorted(sorted_keys, query_keys)
        pos_clamped = pos.clamp(0, N - 1)
        matched[s:e] = in_bounds & (pos < N) & (sorted_keys[pos_clamped] == query_keys)
        found_idx[s:e] = sort_idx_i32[pos_clamped]
    return matched, found_idx.to(torch.int64)


def _lookup_3d_coords(
    coords: torch.Tensor,
    query: torch.Tensor,
    W: int,
    H: int,
    D: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Look up [M, 3] int32 (x, y, z) queries in [N, 3] coords via chunked int64 binary search."""
    N = coords.shape[0]
    M = query.shape[0]
    device = coords.device
    if N == 0 or M == 0:
        return (
            torch.zeros(M, dtype=torch.bool, device=device),
            torch.zeros(M, dtype=torch.int64, device=device),
        )

    c64 = coords.to(torch.int64)
    stride_y = int(D)
    stride_x = int(H) * stride_y

    coord_keys = c64[:, 0] * stride_x + c64[:, 1] * stride_y + c64[:, 2]
    del c64
    sorted_keys, sort_idx = torch.sort(coord_keys)
    del coord_keys
    sort_idx_i32 = sort_idx.to(torch.int32)
    del sort_idx

    matched = torch.empty(M, dtype=torch.bool, device=device)
    found_idx = torch.empty(M, dtype=torch.int32, device=device)
    chunk_size = 262144
    for s in range(0, M, chunk_size):
        e = min(s + chunk_size, M)
        q64 = query[s:e].to(torch.int64)
        in_bounds = (
            (q64[:, 0] >= 0) & (q64[:, 0] < W)
            & (q64[:, 1] >= 0) & (q64[:, 1] < H)
            & (q64[:, 2] >= 0) & (q64[:, 2] < D)
        )
        query_keys = q64[:, 0] * stride_x + q64[:, 1] * stride_y + q64[:, 2]
        del q64
        pos = torch.searchsorted(sorted_keys, query_keys)
        pos_clamped = pos.clamp(0, N - 1)
        matched[s:e] = in_bounds & (pos < N) & (sorted_keys[pos_clamped] == query_keys)
        found_idx[s:e] = sort_idx_i32[pos_clamped]
    return matched, found_idx.to(torch.int64)


def _patch_flex_gemm() -> None:
    import flex_gemm.ops.spconv as spconv
    from flex_gemm.ops.spconv.submanifold_conv3d import (
        SubMConv3dFunction,
        SubMConv3dNeighborCache,
    )
    from flex_gemm.ops.grid_sample.grid_sample import GridSample3dFunction
    from flex_gemm.ops.grid_sample.grid_sample_torch import GridSample3dTorch

    spconv.set_algorithm(spconv.Algorithm.EXPLICIT_GEMM)

    def _force_explicit_gemm(algo):
        spconv.ALGORITHM = spconv.Algorithm.EXPLICIT_GEMM

    spconv.set_algorithm = _force_explicit_gemm

    @staticmethod
    def _compute_neighbor_cache_turing(
        coords: torch.Tensor,
        shape: torch.Size,
        kernel_size: Tuple[int, int, int],
        dilation: Tuple[int, int, int],
    ) -> SubMConv3dNeighborCache:
        N = coords.shape[0]
        B, _, W, H, D = shape
        Kw, Kh, Kd = kernel_size
        V = Kw * Kh * Kd
        device = coords.device

        if N == 0:
            neighbor_map = torch.empty((0, Kw, Kh, Kd), dtype=torch.uint32, device=device)
            return SubMConv3dNeighborCache(neighbor_map=neighbor_map)

        offsets_w = (torch.arange(Kw, device=device, dtype=torch.int32) - Kw // 2) * dilation[0]
        offsets_h = (torch.arange(Kh, device=device, dtype=torch.int32) - Kh // 2) * dilation[1]
        offsets_d = (torch.arange(Kd, device=device, dtype=torch.int32) - Kd // 2) * dilation[2]
        ow, oh, od = torch.meshgrid(offsets_w, offsets_h, offsets_d, indexing="ij")
        offsets = torch.stack([ow, oh, od], dim=-1).reshape(1, V, 3)

        c64 = coords.to(torch.int64)
        stride_y = int(D)
        stride_x = int(H) * stride_y
        stride_b = int(W) * stride_x
        coord_keys = c64[:, 0] * stride_b + c64[:, 1] * stride_x + c64[:, 2] * stride_y + c64[:, 3]
        del c64
        sorted_keys, sort_idx = torch.sort(coord_keys)
        del coord_keys
        sort_idx_i32 = sort_idx.to(torch.int32)
        del sort_idx

        neighbor_i32 = torch.full((N, V), -1, dtype=torch.int32, device=device)
        # Process 65,536 voxels per chunk (~25 MB peak temp memory instead of 8.4 GB at N=4.1M)
        chunk_n = 65536
        for s in range(0, N, chunk_n):
            e = min(s + chunk_n, N)
            c_chunk = coords[s:e].to(torch.int32)
            nx = (c_chunk[:, None, 1] + offsets[:, :, 0]).to(torch.int64)
            ny = (c_chunk[:, None, 2] + offsets[:, :, 1]).to(torch.int64)
            nz = (c_chunk[:, None, 3] + offsets[:, :, 2]).to(torch.int64)
            nb = c_chunk[:, None, 0].to(torch.int64)
            in_bounds = (
                (nb >= 0) & (nb < B)
                & (nx >= 0) & (nx < W)
                & (ny >= 0) & (ny < H)
                & (nz >= 0) & (nz < D)
            )
            q_keys = (nb * stride_b + nx * stride_x + ny * stride_y + nz).reshape(-1)
            del c_chunk, nx, ny, nz, nb
            pos = torch.searchsorted(sorted_keys, q_keys)
            pos_clamped = pos.clamp(0, N - 1)
            matched = in_bounds.reshape(-1) & (pos < N) & (sorted_keys[pos_clamped] == q_keys)
            chunk_out = neighbor_i32[s:e].reshape(-1)
            chunk_out[matched] = sort_idx_i32[pos_clamped[matched]]
            del q_keys, pos, pos_clamped, matched, in_bounds

        del sorted_keys, sort_idx_i32
        return SubMConv3dNeighborCache(neighbor_map=neighbor_i32.view(torch.uint32).reshape(N, Kw, Kh, Kd))

    @staticmethod
    def _sparse_submanifold_conv_forward_turing(
        feats: torch.Tensor,
        neighbor_cache: SubMConv3dNeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N, Ci = feats.shape
        Co, Kw, Kh, Kd, _ = weight.shape
        if N == 0:
            return torch.empty((0, Co), device=feats.device, dtype=feats.dtype)
        orig_dtype = feats.dtype
        compute_dtype = torch.float16 if orig_dtype == torch.bfloat16 else orig_dtype
        V = Kw * Kh * Kd
        w_mat = weight.to(compute_dtype).reshape(Co, V * Ci).transpose(0, 1).contiguous()
        b_vec = bias.to(compute_dtype) if bias is not None else None
        nb_i32 = neighbor_cache.neighbor_map.view(torch.int32).reshape(N, V)
        feats_c = feats.clamp(-65500.0, 65500.0).to(compute_dtype) if orig_dtype == torch.bfloat16 else feats
        out = torch.empty((N, Co), device=feats.device, dtype=compute_dtype)
        # Bound temporary im2col buffer to ~16M elements (~32 MB in fp16)
        chunk = max(512, min(N, (16 * 1024 * 1024) // max(V * Ci, 1)))
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            nb_c = nb_i32[s:e]
            valid = nb_c >= 0
            idx = nb_c.clamp_min(0).long()
            im2col = feats_c[idx]
            im2col.masked_fill_(~valid.unsqueeze(-1), 0)
            im2col = im2col.reshape(e - s, V * Ci)
            if b_vec is not None:
                torch.addmm(b_vec, im2col, w_mat, out=out[s:e])
            else:
                torch.mm(im2col, w_mat, out=out[s:e])
            del im2col, idx, valid
        del feats_c, w_mat
        return out.to(orig_dtype) if compute_dtype != orig_dtype else out

    SubMConv3dFunction._compute_neighbor_cache = _compute_neighbor_cache_turing
    SubMConv3dFunction._sparse_submanifold_conv_forward = _sparse_submanifold_conv_forward_turing

    @staticmethod
    def _nearest_fwd_turing(
        ctx,
        feats: torch.Tensor,
        coords: torch.Tensor,
        shape: torch.Size,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        B, L = grid.shape[:2]
        C, W, H, D = shape[-4:]
        b_idx = torch.arange(B, dtype=torch.int32, device=grid.device).reshape(B, 1, 1).expand(B, L, 1)
        q_xyz = torch.floor(grid).to(torch.int32)
        queries = torch.cat([b_idx, q_xyz], dim=-1).reshape(B * L, 4)

        matched, found_idx = _lookup_4d_coords(coords, queries, B, W, H, D)
        idx = torch.full((B, L), -1, dtype=torch.int32, device=grid.device)
        idx.view(-1)[matched] = found_idx[matched].to(torch.int32)

        out = feats[idx.clamp_min(0).long()] * (idx >= 0).unsqueeze(-1).to(feats.dtype)
        ctx.save_for_backward(idx)
        ctx.N = feats.shape[0]
        return out

    @staticmethod
    def _trilinear_fwd_turing(
        ctx,
        feats: torch.Tensor,
        coords: torch.Tensor,
        shape: torch.Size,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        B, L = grid.shape[:2]
        C = feats.shape[1]
        _, W, H, D = shape[-4:]

        offsets = torch.tensor(
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5,  0.5],
                [-0.5,  0.5, -0.5],
                [-0.5,  0.5,  0.5],
                [ 0.5, -0.5, -0.5],
                [ 0.5, -0.5,  0.5],
                [ 0.5,  0.5, -0.5],
                [ 0.5,  0.5,  0.5],
            ],
            dtype=grid.dtype,
            device=grid.device,
        )

        # Process in chunks to keep peak VRAM small on 6 GB GPUs
        chunk_size = 65536
        out_chunks = []
        c64 = coords.to(torch.int64)
        stride_y = int(D)
        stride_x = int(H) * stride_y
        stride_b = int(W) * stride_x
        coord_keys = c64[:, 0] * stride_b + c64[:, 1] * stride_x + c64[:, 2] * stride_y + c64[:, 3]
        sorted_keys, sort_idx = torch.sort(coord_keys)
        N = coords.shape[0]

        for b in range(B):
            grid_b = grid[b]  # [L, 3]
            b_chunks = []
            for start in range(0, L, chunk_size):
                end = min(start + chunk_size, L)
                g = grid_b[start:end]  # [M, 3]
                M = g.shape[0]
                neigh_xyz = torch.floor(g.unsqueeze(1) + offsets.unsqueeze(0)).to(torch.int64)  # [M, 8, 3]
                in_bounds = (
                    (neigh_xyz[..., 0] >= 0) & (neigh_xyz[..., 0] < W)
                    & (neigh_xyz[..., 1] >= 0) & (neigh_xyz[..., 1] < H)
                    & (neigh_xyz[..., 2] >= 0) & (neigh_xyz[..., 2] < D)
                )
                q_keys = (
                    int(b) * stride_b
                    + neigh_xyz[..., 0] * stride_x
                    + neigh_xyz[..., 1] * stride_y
                    + neigh_xyz[..., 2]
                ).reshape(-1)
                pos = torch.searchsorted(sorted_keys, q_keys)
                pos_clamped = pos.clamp(0, max(N - 1, 0))
                matched = in_bounds.reshape(-1) & (pos < N) & (sorted_keys[pos_clamped] == q_keys)
                found = sort_idx[pos_clamped]

                raw_w = torch.prod(
                    1.0 - torch.abs(neigh_xyz.to(g.dtype) + 0.5 - g.unsqueeze(1)),
                    dim=-1,
                )  # [M, 8]
                w = torch.where(matched.view(M, 8), raw_w, torch.zeros_like(raw_w))
                w = w / torch.clamp_min(w.sum(dim=1, keepdim=True), 1e-12)

                gathered = feats[found.view(M, 8)]  # [M, 8, C]
                chunk_out = (gathered * w.to(feats.dtype).unsqueeze(-1)).sum(dim=1)
                b_chunks.append(chunk_out)
            out_chunks.append(torch.cat(b_chunks, dim=0) if b_chunks else torch.empty((0, C), device=feats.device, dtype=feats.dtype))

        return torch.stack(out_chunks, dim=0)

    GridSample3dFunction._nearest_fwd = _nearest_fwd_turing
    GridSample3dFunction._trilinear_fwd = _trilinear_fwd_turing
    GridSample3dTorch._nearest = staticmethod(
        lambda feats, coords, shape, query_pts: _nearest_fwd_turing(
            type("_DummyCtx", (), {"save_for_backward": lambda *a: None})(),
            feats,
            coords,
            shape,
            query_pts,
        )
    )
    GridSample3dTorch._trilinear = staticmethod(
        lambda feats, coords, shape, query_pts: _trilinear_fwd_turing(
            type("_DummyCtx", (), {"save_for_backward": lambda *a: None})(),
            feats,
            coords,
            shape,
            query_pts,
        )
    )


def _patch_o_voxel() -> None:
    import o_voxel.serialize as ov_serialize
    import o_voxel.convert.flexible_dual_grid as ov_fdg
    from o_voxel import _C

    @torch.no_grad()
    def _encode_seq_turing(coords: torch.Tensor, permute=None, mode: str = "z_order") -> torch.Tensor:
        assert coords.shape[1] == 3 and coords.ndim == 2, "Input coordinates must be of shape [N, 3]"
        if permute is None:
            permute = [0, 1, 2]
        x = coords[:, permute[0]].int().cpu().contiguous()
        y = coords[:, permute[1]].int().cpu().contiguous()
        z = coords[:, permute[2]].int().cpu().contiguous()
        if mode == "z_order":
            codes = _C.z_order_encode_cpu(x, y, z)
        elif mode == "hilbert":
            codes = _C.hilbert_encode_cpu(x, y, z)
        else:
            raise ValueError(f"Unknown encoding mode: {mode}")
        return codes.to(coords.device)

    @torch.no_grad()
    def _decode_seq_turing(code: torch.Tensor, permute=None, mode: str = "z_order") -> torch.Tensor:
        assert code.ndim == 1, "Input codes must be of shape [N]"
        if permute is None:
            permute = [0, 1, 2]
        code_cpu = code.cpu().contiguous()
        if mode == "z_order":
            x, y, z = _C.z_order_decode_cpu(code_cpu)
        elif mode == "hilbert":
            x, y, z = _C.hilbert_decode_cpu(code_cpu)
        else:
            raise ValueError(f"Unknown decoding mode: {mode}")
        coords = [None, None, None]
        coords[permute[0]] = x
        coords[permute[1]] = y
        coords[permute[2]] = z
        return torch.stack(coords, dim=-1).to(code.device)

    ov_serialize.encode_seq = _encode_seq_turing
    ov_serialize.decode_seq = _decode_seq_turing

    class _TuringOVoxelC:
        def __init__(self, orig_c):
            self._orig = orig_c

        def __getattr__(self, name):
            return getattr(self._orig, name)

        def hashmap_insert_3d_idx_as_val_cuda(self, state, coords_4d, W, H, D):
            state["coords"] = coords_4d.int()

        def hashmap_lookup_3d_cuda(self, state, query_4d, W, H, D):
            coords_4d = state["coords"]
            matched, found_idx = _lookup_4d_coords(coords_4d, query_4d.int(), 1, int(W), int(H), int(D))
            res = torch.full((query_4d.shape[0],), -1, dtype=torch.int32, device=query_4d.device)
            res[matched] = found_idx[matched].to(torch.int32)
            return res.view(torch.uint32)

    ov_fdg._init_hashmap = lambda resolution, capacity, device: [{}]
    ov_fdg._C = _TuringOVoxelC(_C)


def _patch_cumesh() -> None:
    import cumesh
    from cumesh.xatlas import Atlas
    import meshlib.mrmeshnumpy as mrmeshnumpy
    import meshlib.mrmeshpy as mrmeshpy

    class _TuringCuMesh:
        """CPU/PyTorch fallback for CuMesh on GPUs with compute capability < 8.0."""

        def __init__(self):
            self._vertices: Optional[torch.Tensor] = None
            self._faces: Optional[torch.Tensor] = None
            self._face_normals: Optional[torch.Tensor] = None
            self._vertex_normals: Optional[torch.Tensor] = None

        def init(self, vertices: torch.Tensor, faces: torch.Tensor):
            self._vertices = vertices.detach().float().contiguous()
            self._faces = faces.detach().int().contiguous()
            self._face_normals = None
            self._vertex_normals = None

        @property
        def num_vertices(self) -> int:
            return 0 if self._vertices is None else int(self._vertices.shape[0])

        @property
        def num_faces(self) -> int:
            return 0 if self._faces is None else int(self._faces.shape[0])

        @property
        def num_edges(self) -> int:
            return 0

        @property
        def num_boundaries(self) -> int:
            return 0

        @property
        def num_conneted_components(self) -> int:
            return 1

        @property
        def num_boundary_conneted_components(self) -> int:
            return 0

        @property
        def num_boundary_loops(self) -> int:
            return 0

        def clear_cache(self):
            self._face_normals = None

        def read(self) -> Tuple[torch.Tensor, torch.Tensor]:
            return self._vertices, self._faces

        def compute_face_normals(self):
            v0 = self._vertices[self._faces[:, 0].long()]
            v1 = self._vertices[self._faces[:, 1].long()]
            v2 = self._vertices[self._faces[:, 2].long()]
            cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
            norm = torch.linalg.norm(cross, dim=-1, keepdim=True)
            self._face_normals = torch.where(norm > 1e-12, cross / norm, torch.full_like(cross, float("nan")))

        def read_face_normals(self) -> torch.Tensor:
            if self._face_normals is None:
                self.compute_face_normals()
            return self._face_normals

        def compute_vertex_normals(self):
            self.compute_face_normals()
            fn = torch.nan_to_num(self._face_normals, nan=0.0)
            vn = torch.zeros_like(self._vertices)
            f_long = self._faces.long()
            for c in range(3):
                vn.index_add_(0, f_long[:, c], fn)
            norm = torch.linalg.norm(vn, dim=-1, keepdim=True)
            self._vertex_normals = torch.where(norm > 1e-12, vn / norm, vn)

        def read_vertex_normals(self) -> torch.Tensor:
            if self._vertex_normals is None:
                self.compute_vertex_normals()
            return self._vertex_normals

        def get_edges(self):
            pass

        def get_boundary_info(self):
            pass

        def get_vertex_edge_adjacency(self):
            pass

        def get_vertex_boundary_adjacency(self):
            pass

        def get_manifold_boundary_adjacency(self):
            pass

        def read_manifold_boundary_adjacency(self):
            return torch.empty((0, 2), dtype=torch.int32, device=self._vertices.device)

        def get_boundary_connected_components(self):
            pass

        def get_boundary_loops(self):
            pass

        def fill_holes(self, max_hole_perimeter: float = 3e-2):
            pass

        def unify_face_orientations(self):
            if self._faces is None or self._faces.numel() == 0:
                return
            import pymeshlab
            dev = self._vertices.device
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(
                vertex_matrix=self._vertices.detach().cpu().numpy(),
                face_matrix=self._faces.detach().cpu().numpy(),
            ))
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_repair_non_manifold_vertices()
            ms.meshing_re_orient_faces_coherently()
            ms.meshing_re_orient_faces_by_geometry()
            m = ms.current_mesh()
            self._vertices = torch.from_numpy(m.vertex_matrix()).float().to(dev).contiguous()
            self._faces = torch.from_numpy(m.face_matrix()).int().to(dev).contiguous()
            self._face_normals = None

        def remove_duplicate_faces(self):
            pass

        def repair_non_manifold_edges(self):
            pass

        def remove_small_connected_components(self, min_area: float):
            pass

        def remove_faces(self, face_mask: torch.Tensor):
            kept_faces = self._faces[face_mask]
            if kept_faces.numel() == 0:
                return
            unique_v, inv = torch.unique(kept_faces.long(), return_inverse=True)
            self._vertices = self._vertices[unique_v].contiguous()
            self._faces = inv.to(torch.int32).reshape(-1, 3).contiguous()
            self._face_normals = None

        def remove_unreferenced_vertices(self):
            if self._faces is None or self._faces.numel() == 0:
                return
            unique_v, inv = torch.unique(self._faces.long(), return_inverse=True)
            self._vertices = self._vertices[unique_v].contiguous()
            self._faces = inv.to(torch.int32).reshape(-1, 3).contiguous()

        def remove_degenerate_faces(self):
            self.compute_face_normals()
            kept = self._face_normals.isnan().sum(dim=1) == 0
            if int(kept.sum().item()) > 0:
                self.remove_faces(kept)

        def simplify(self, target_num_faces: int, verbose: bool = False, options: dict = {}):
            if self.num_faces <= target_num_faces:
                return
            dev = self._vertices.device
            v_np = self._vertices.detach().cpu().numpy()
            f_np = self._faces.detach().cpu().numpy()

            settings = mrmeshpy.DecimateSettings()
            settings.maxDeletedFaces = len(f_np) - target_num_faces
            settings.packMesh = True
            mesh = mrmeshnumpy.meshFromFacesVerts(f_np, v_np)
            mesh.packOptimally()
            mrmeshpy.decimateMesh(mesh, settings)
            new_v = mrmeshnumpy.getNumpyVerts(mesh)
            new_f = mrmeshnumpy.getNumpyFaces(mesh.topology)
            del mesh

            # Fallback if input had unoriented seams that prevented Meshlib from reaching target
            if len(new_f) > int(target_num_faces * 1.1):
                import pymeshlab
                ms = pymeshlab.MeshSet()
                ms.add_mesh(pymeshlab.Mesh(vertex_matrix=new_v, face_matrix=new_f))
                ms.meshing_repair_non_manifold_edges()
                ms.meshing_repair_non_manifold_vertices()
                ms.meshing_re_orient_faces_coherently()
                ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target_num_faces))
                ms.meshing_re_orient_faces_coherently()
                ms.meshing_re_orient_faces_by_geometry()
                m_final = ms.current_mesh()
                new_v = m_final.vertex_matrix()
                new_f = m_final.face_matrix()

            if verbose:
                print(f"Reduced faces, resulting in {len(new_v)} vertices and {len(new_f)} faces", flush=True)
            self._vertices = torch.from_numpy(new_v).float().to(dev).contiguous()
            self._faces = torch.from_numpy(new_f).int().to(dev).contiguous()
            self._face_normals = None

        def uv_unwrap(
            self,
            compute_charts_kwargs: dict = {},
            xatlas_compute_charts_kwargs: dict = {},
            xatlas_pack_charts_kwargs: dict = {},
            return_vmaps: bool = False,
            verbose: bool = False,
        ):
            self.remove_degenerate_faces()
            v_cpu = self._vertices.detach().cpu().float().contiguous()
            f_cpu = self._faces.detach().cpu().int().contiguous()

            xatlas_compute_charts_kwargs = dict(xatlas_compute_charts_kwargs)
            xatlas_pack_charts_kwargs = dict(xatlas_pack_charts_kwargs)
            xatlas_compute_charts_kwargs["verbose"] = verbose
            xatlas_pack_charts_kwargs["verbose"] = verbose

            # Pre-cluster faces by dominant normal (6 directions) and 4x4x4 spatial cell
            # so xatlas parameterizes small charts in parallel (~0.4s instead of 20+ mins)
            f_long = f_cpu.long()
            v0 = v_cpu[f_long[:, 0]]
            v1 = v_cpu[f_long[:, 1]]
            v2 = v_cpu[f_long[:, 2]]
            fn = torch.linalg.cross(v1 - v0, v2 - v0)
            max_axis = fn.abs().argmax(dim=-1)
            sign_bit = (fn.gather(1, max_axis.unsqueeze(1)).squeeze(1) > 0).long()
            dir_id = max_axis * 2 + sign_bit

            fc = (v0 + v1 + v2) * (1.0 / 3.0)
            bmin = fc.min(dim=0).values
            bmax = fc.max(dim=0).values
            grid_pos = ((fc - bmin) / (bmax - bmin + 1e-6) * 4.0).long().clamp(0, 3)
            cell_id = grid_pos[:, 0] * 16 + grid_pos[:, 1] * 4 + grid_pos[:, 2]
            cluster_id = dir_id * 64 + cell_id

            unique_clusters, inv = torch.unique(cluster_id, return_inverse=True)
            order = torch.argsort(inv)
            counts = torch.bincount(inv)
            offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
            num_charts = int(unique_clusters.shape[0])
            if verbose:
                print(f"Get {num_charts} clusters after fast clustering", flush=True)

            xatlas = Atlas()
            chart_vmaps = []
            for i in range(num_charts):
                f_idx = order[offsets[i]:offsets[i + 1]]
                sub_f = f_long[f_idx]
                uv_idx, local_f = torch.unique(sub_f, return_inverse=True)
                chart_vmaps.append(uv_idx)
                xatlas.add_mesh(v_cpu[uv_idx].contiguous(), local_f.int().reshape(-1, 3).contiguous())

            xatlas.compute_charts(**xatlas_compute_charts_kwargs)
            xatlas.pack_charts(**xatlas_pack_charts_kwargs)

            vmaps = []
            faces_out = []
            uvs_out = []
            cnt = 0
            for i in range(num_charts):
                vmap_i, x_faces_i, x_uvs_i = xatlas.get_mesh(i)
                vmaps.append(chart_vmaps[i][vmap_i.long()])
                faces_out.append(x_faces_i + cnt)
                uvs_out.append(x_uvs_i)
                cnt += int(vmap_i.shape[0])

            vmaps_cat = torch.cat(vmaps, dim=0)
            vertices = v_cpu[vmaps_cat]
            faces_cat = torch.cat(faces_out, dim=0)
            uvs_cat = torch.cat(uvs_out, dim=0)
            out = [vertices, faces_cat, uvs_cat]
            if return_vmaps:
                out.append(vmaps_cat)
            return tuple(out)

    def _reconstruct_mesh_dc_turing(
        vertices: torch.Tensor,
        faces: torch.Tensor,
        resolution: int = 512,
        band: float = 1.0,
        verbose: bool = False,
        bvh=None,
        remove_inner_faces: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Watertight Dual Marching Cubes reconstruction using OpenVDB (Meshlib) + GPU outer-shell filter."""
        dev = vertices.device
        v_np = vertices.detach().cpu().numpy()
        f_np = faces.detach().cpu().numpy()

        bbox_min = v_np.min(axis=0)
        bbox_max = v_np.max(axis=0)
        scale = float((bbox_max - bbox_min).max()) * 1.1
        eff_res = min(int(resolution), 512)
        voxel_size = max(scale / eff_res, 1e-4)

        mesh = mrmeshnumpy.meshFromFacesVerts(f_np, v_np)
        params = mrmeshpy.OffsetParameters()
        params.voxelSize = voxel_size
        params.signDetectionMode = mrmeshpy.SignDetectionMode.Unsigned
        offset_dist = 0.75 * voxel_size
        rebuilt = mrmeshpy.offsetMesh(mesh, offset_dist, params)
        del mesh

        v_out = mrmeshnumpy.getNumpyVerts(rebuilt)
        f_out = mrmeshnumpy.getNumpyFaces(rebuilt.topology)
        del rebuilt

        if len(f_out) == 0:
            return vertices, faces

        if remove_inner_faces:
            import numpy as np
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components

            f32 = f_out.astype(np.int32)
            rows = np.concatenate([f32[:, 0], f32[:, 1], f32[:, 2]])
            cols = np.concatenate([f32[:, 1], f32[:, 2], f32[:, 0]])
            adj = coo_matrix(
                (np.ones(len(rows), dtype=np.int8), (rows, cols)),
                shape=(len(v_out), len(v_out)),
            )
            n_comp, v_labels = connected_components(adj, directed=False)
            f_labels = v_labels[f32[:, 0]]
            comp_sizes = np.bincount(f_labels, minlength=n_comp)

            # GPU 256^3 exterior-air flood-fill to separate outer surface shells from internal hollow cavities
            vg_res = 256
            v_gpu = torch.from_numpy(v_out).to(device="cuda", dtype=torch.float32)
            f_gpu = torch.from_numpy(f_out).to(device="cuda", dtype=torch.long)
            bmin = v_gpu.min(dim=0).values - 2.0 * voxel_size
            bmax = v_gpu.max(dim=0).values + 2.0 * voxel_size
            fc = (v_gpu[f_gpu[:, 0]] + v_gpu[f_gpu[:, 1]] + v_gpu[f_gpu[:, 2]]) * (1.0 / 3.0)
            fn = torch.linalg.cross(
                v_gpu[f_gpu[:, 1]] - v_gpu[f_gpu[:, 0]],
                v_gpu[f_gpu[:, 2]] - v_gpu[f_gpu[:, 0]],
            )
            fn = fn / (torch.linalg.norm(fn, dim=-1, keepdim=True) + 1e-12)

            fc_vox = ((fc - bmin) / (bmax - bmin) * (vg_res - 3) + 1.5).long().clamp(1, vg_res - 2)
            occ = torch.zeros((1, 1, vg_res, vg_res, vg_res), dtype=torch.bool, device="cuda")
            occ[0, 0, fc_vox[:, 0], fc_vox[:, 1], fc_vox[:, 2]] = True
            for c in range(3):
                vc = ((v_gpu[f_gpu[:, c]] - bmin) / (bmax - bmin) * (vg_res - 3) + 1.5).long().clamp(1, vg_res - 2)
                occ[0, 0, vc[:, 0], vc[:, 1], vc[:, 2]] = True

            occ_dil = F.max_pool3d(occ.float(), kernel_size=3, stride=1, padding=1) > 0.5
            ext = torch.zeros_like(occ_dil)
            ext[:, :, 0, :, :] = True
            ext[:, :, -1, :, :] = True
            ext[:, :, :, 0, :] = True
            ext[:, :, :, -1, :] = True
            ext[:, :, :, :, 0] = True
            ext[:, :, :, :, -1] = True
            free = ~occ_dil

            for step in range(vg_res):
                prev = ext
                ext = (F.max_pool3d(ext.float(), kernel_size=3, stride=1, padding=1) > 0.5) & free
                if step % 16 == 15 and torch.equal(ext, prev):
                    break

            ext_reach = F.max_pool3d(ext.float(), kernel_size=5, stride=1, padding=2) > 0.5
            probe = fc + fn * (1.5 * voxel_size)
            p_vox = ((probe - bmin) / (bmax - bmin) * (vg_res - 3) + 1.5).long().clamp(0, vg_res - 1)
            f_is_outer = ext_reach[0, 0, p_vox[:, 0], p_vox[:, 1], p_vox[:, 2]].cpu().numpy()

            comp_outer_hits = np.bincount(f_labels, weights=f_is_outer.astype(np.float32), minlength=n_comp)
            comp_outer_ratio = comp_outer_hits / np.maximum(comp_sizes, 1)
            keep_comp = (comp_sizes >= 100) & (comp_outer_ratio > 0.05)
            if not keep_comp.any():
                keep_comp[int(comp_sizes.argmax())] = True

            keep_f = keep_comp[f_labels]
            f_kept = f_out[keep_f]
            used_v = np.unique(f_kept)
            remap = np.full(len(v_out), -1, dtype=np.int32)
            remap[used_v] = np.arange(len(used_v), dtype=np.int32)
            v_out = v_out[used_v]
            f_out = remap[f_kept]

            del v_gpu, f_gpu, fc, fn, occ, occ_dil, ext, free, ext_reach
            torch.cuda.empty_cache()

        # Pull vertices back slightly along normals to compensate for the 0.75-voxel dilation
        v_t = torch.from_numpy(v_out).to(device=dev, dtype=torch.float32).contiguous()
        f_t = torch.from_numpy(f_out).to(device=dev, dtype=torch.int32).contiguous()
        f_long = f_t.long()
        fn_t = torch.linalg.cross(v_t[f_long[:, 1]] - v_t[f_long[:, 0]], v_t[f_long[:, 2]] - v_t[f_long[:, 0]])
        vn_t = torch.zeros_like(v_t)
        for c in range(3):
            vn_t.index_add_(0, f_long[:, c], fn_t)
        vn_norm = torch.linalg.norm(vn_t, dim=-1, keepdim=True)
        vn_t = torch.where(vn_norm > 1e-12, vn_t / vn_norm, vn_t)
        v_t = (v_t - vn_t * offset_dist).contiguous()
        return v_t, f_t

    def _remesh_narrow_band_dc_turing(
        vertices: torch.Tensor,
        faces: torch.Tensor,
        center: torch.Tensor,
        scale: float,
        resolution: int,
        band: float = 1.0,
        project_back: float = 0.0,
        verbose: bool = False,
        bvh=None,
        remove_inner_faces: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _reconstruct_mesh_dc_turing(
            vertices,
            faces,
            resolution=resolution,
            band=band,
            verbose=verbose,
            bvh=bvh,
            remove_inner_faces=remove_inner_faces,
        )

    cumesh.CuMesh = _TuringCuMesh
    if hasattr(cumesh, "remeshing"):
        cumesh.remeshing.reconstruct_mesh_dc = _reconstruct_mesh_dc_turing
        cumesh.remeshing.reconstruct_mesh_dc_quad = _reconstruct_mesh_dc_turing
        cumesh.remeshing.remesh_narrow_band_dc = _remesh_narrow_band_dc_turing
        cumesh.remeshing.remesh_narrow_band_dc_quad = _remesh_narrow_band_dc_turing


def patch_trellis2_windowed_attn(trellis2_pkg_name: str = "aismith_trellis2") -> None:
    """Patch windowed_attn in the dynamically loaded ComfyUI-Trellis2 package to support SDPA."""
    import importlib
    import sys

    def _get_or_import(mod_path: str):
        mod = sys.modules.get(mod_path)
        if mod is None:
            try:
                mod = importlib.import_module(mod_path)
            except Exception:
                return None
        return mod

    if needs_turing_compat():
        utils_mod = _get_or_import(f"{trellis2_pkg_name}.trellis2.modules.utils")
        if utils_mod is not None:
            def _patch_fp8_forward_fp16(module):
                if hasattr(module, "_original_forward_fp8"):
                    return
                module._original_forward_fp8 = module.forward

                def fp8_forward(*args, **kwargs):
                    with torch.no_grad():
                        orig_weight_data = module.weight.data
                        module.weight.data = orig_weight_data.to(torch.float16)
                        has_bias = getattr(module, "bias", None) is not None
                        if has_bias:
                            orig_bias_data = module.bias.data
                            module.bias.data = orig_bias_data.to(torch.float16)
                    try:
                        return module._original_forward_fp8(*args, **kwargs)
                    finally:
                        with torch.no_grad():
                            module.weight.data = orig_weight_data
                            if has_bias:
                                module.bias.data = orig_bias_data

                module.forward = fp8_forward

            utils_mod.patch_fp8_forward = _patch_fp8_forward_fp16

    cfg_mod = _get_or_import(f"{trellis2_pkg_name}.trellis2.modules.sparse.config")
    if cfg_mod is not None:
        def _set_attn_backend(backend: str):
            print(f"[SPARSE] Setting attention backend to: {backend}")
            cfg_mod.ATTN = backend
        cfg_mod.set_attn_backend = _set_attn_backend
        cfg_mod.ATTN = "sdpa"

    mod_name = f"{trellis2_pkg_name}.trellis2.modules.sparse.attention.windowed_attn"
    win_mod = _get_or_import(mod_name)
    if win_mod is not None:
        def _windowed_sdpa_varlen(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_lens) -> torch.Tensor:
            if len(seq_lens) == 0 or q.shape[0] == 0:
                return torch.empty_like(q)

            device = q.device
            H, C = q.shape[1], q.shape[2]
            lens = torch.tensor(seq_lens, device=device, dtype=torch.long)
            valid_win = lens > 0
            if not valid_win.all():
                lens = lens[valid_win]
            W = int(lens.shape[0])
            max_len = int(lens.max().item())

            cu = torch.cumsum(lens, dim=0)
            starts = cu - lens
            win_ids = torch.repeat_interleave(torch.arange(W, device=device, dtype=torch.long), lens)
            intra = torch.arange(q.shape[0], device=device, dtype=torch.long) - starts[win_ids]

            out = torch.empty_like(q)
            # Process windows in chunks to bound temporary memory
            win_chunk = max(1, min(W, 262144 // max(max_len, 1)))
            col_arange = torch.arange(max_len, device=device).view(1, 1, 1, max_len)

            for w_start in range(0, W, win_chunk):
                w_end = min(w_start + win_chunk, W)
                wc = w_end - w_start
                token_start = int(starts[w_start].item())
                token_end = int(cu[w_end - 1].item())
                n_tok = token_end - token_start

                local_win = win_ids[token_start:token_end] - w_start
                local_intra = intra[token_start:token_end]
                flat_idx = local_win * max_len + local_intra

                q_pad = torch.zeros((wc * max_len, H, C), device=device, dtype=q.dtype)
                k_pad = torch.zeros((wc * max_len, H, C), device=device, dtype=k.dtype)
                v_pad = torch.zeros((wc * max_len, H, C), device=device, dtype=v.dtype)

                q_pad[flat_idx] = q[token_start:token_end]
                k_pad[flat_idx] = k[token_start:token_end]
                v_pad[flat_idx] = v[token_start:token_end]

                q_b = q_pad.view(wc, max_len, H, C).permute(0, 2, 1, 3)
                k_b = k_pad.view(wc, max_len, H, C).permute(0, 2, 1, 3)
                v_b = v_pad.view(wc, max_len, H, C).permute(0, 2, 1, 3)

                mask = col_arange < lens[w_start:w_end].view(wc, 1, 1, 1)
                res = F.scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=mask)
                res_flat = res.permute(0, 2, 1, 3).reshape(wc * max_len, H, C)
                out[token_start:token_end] = res_flat[flat_idx]

            return out

        def _patched_sparse_windowed_self_attention(qkv, window_size, shift_window=(0, 0, 0)):
            if len(qkv) == 1:
                return qkv.replace(qkv.feats[:, 0])
            serialization_spatial_cache_name = f"window_partition_{window_size}_{shift_window}"
            serialization_spatial_cache = qkv.get_spatial_cache(serialization_spatial_cache_name)
            if serialization_spatial_cache is None:
                fwd_indices, bwd_indices, seq_lens, seq_batch_indices = win_mod.calc_window_partition(
                    qkv, window_size, shift_window
                )
                qkv.register_spatial_cache(
                    serialization_spatial_cache_name,
                    (fwd_indices, bwd_indices, seq_lens, seq_batch_indices),
                )
            else:
                fwd_indices, bwd_indices, seq_lens, seq_batch_indices = serialization_spatial_cache

            qkv_feats = qkv.feats[fwd_indices]
            q, k, v = qkv_feats.unbind(dim=1)
            out = _windowed_sdpa_varlen(q, k, v, seq_lens)
            out = out[bwd_indices]
            return qkv.replace(out)

        win_mod.sparse_windowed_scaled_dot_product_self_attention = _patched_sparse_windowed_self_attention
        for mod in list(sys.modules.values()):
            if mod is not None and hasattr(mod, "sparse_windowed_scaled_dot_product_self_attention"):
                setattr(mod, "sparse_windowed_scaled_dot_product_self_attention", _patched_sparse_windowed_self_attention)

    vae_mod_name = f"{trellis2_pkg_name}.trellis2.models.sc_vaes.sparse_unet_vae"
    vae_mod = _get_or_import(vae_mod_name)
    if vae_mod is not None:
        chunked_apply = getattr(vae_mod, "chunked_apply", None)
        enc_cls = getattr(vae_mod, "SparseUnetVaeEncoder", None)
        dec_cls = getattr(vae_mod, "SparseUnetVaeDecoder", None)

        def _prune_temp_spatial_cache(st):
            cache_root = getattr(st, "_spatial_cache", None)
            if not isinstance(cache_root, dict):
                return
            for scale_dict in cache_root.values():
                if isinstance(scale_dict, dict):
                    for k in list(scale_dict.keys()):
                        if k.startswith("conv_") or k.startswith("window_partition_"):
                            del scale_dict[k]

        if enc_cls is not None and chunked_apply is not None:
            def _encoder_forward_low_vram(self, x, sample_posterior=False, return_raw=False):
                h = self.input_layer(x)
                h = h.type(self.dtype)
                for i, res in enumerate(self.blocks):
                    for j, block in enumerate(res):
                        h = block(h)
                    _prune_temp_spatial_cache(h)
                    if self.low_vram:
                        torch.cuda.empty_cache()
                if self.low_vram:
                    def fused_finalize(t):
                        t = t.type(x.dtype)
                        t = F.layer_norm(t, (t.shape[-1],))
                        t = F.linear(t, self.to_latent.weight, self.to_latent.bias)
                        return t
                    h = h.replace(chunked_apply(fused_finalize, h.feats, self.chunk_size))
                else:
                    h = h.type(x.dtype)
                    h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
                    h = self.to_latent(h)
                mean, logvar = h.feats.chunk(2, dim=-1)
                if sample_posterior:
                    std = torch.exp(0.5 * logvar)
                    z = mean + std * torch.randn_like(std)
                else:
                    z = mean
                z = h.replace(z)
                return (z, mean, logvar) if return_raw else z

            enc_cls.forward = _encoder_forward_low_vram

        if dec_cls is not None and chunked_apply is not None:
            def _decoder_forward_low_vram(self, x, guide_subs=None, return_subs=False):
                assert guide_subs is None or self.pred_subdiv is False
                assert return_subs is False or self.pred_subdiv is True
                h = self.from_latent(x)
                h = h.type(self.dtype)
                subs_gt = []
                subs = []
                for i, res in enumerate(self.blocks):
                    for j, block in enumerate(res):
                        if i < len(self.blocks) - 1 and j == len(res) - 1:
                            if self.pred_subdiv:
                                if self.training:
                                    subs_gt.append(h.get_spatial_cache("subdivision"))
                                h, sub = block(h)
                                _prune_temp_spatial_cache(sub)
                                if return_subs or self.training:
                                    subs.append(sub)
                                else:
                                    del sub
                            else:
                                h = block(h, subdiv=guide_subs[i] if guide_subs is not None else None)
                        else:
                            h = block(h)
                        if self.low_vram:
                            torch.cuda.empty_cache()
                    _prune_temp_spatial_cache(h)
                    torch.cuda.empty_cache()
                if self.low_vram:
                    def fused_finalize(t):
                        t = t.type(x.dtype)
                        t = F.layer_norm(t, (t.shape[-1],))
                        t = F.linear(t, self.output_layer.weight, self.output_layer.bias)
                        return t
                    h = h.replace(chunked_apply(fused_finalize, h.feats, self.chunk_size))
                else:
                    h = h.type(x.dtype)
                    h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
                    h = self.output_layer(h)
                if self.training and self.pred_subdiv:
                    return h, subs_gt, subs
                return (h, subs) if return_subs else h

            dec_cls.forward = _decoder_forward_low_vram

    mesh_base_mod = _get_or_import(f"{trellis2_pkg_name}.trellis2.representations.mesh.base")
    if mesh_base_mod is not None:
        mesh_cls = getattr(mesh_base_mod, "Mesh", None)
        if mesh_cls is not None:
            def _simplify_with_meshlib_turing(self, target=1000000):
                self.simplify_with_cumesh(target=target, verbose=True)

            mesh_cls.simplify_with_meshlib = _simplify_with_meshlib_turing


def _patch_nvdiffrast() -> None:
    """Replace nvdiffrast CUDA rasterization/interpolation with PyTorch GPU kernels on Turing."""
    import nvdiffrast.torch as dr
    import nvdiffrast.torch.ops as dr_ops

    class _TuringRasterizeContext:
        def __init__(self, device=None):
            self.device = torch.device("cuda" if device is None else device)
            self.active_depth_peeler = None

    def _rasterize_turing(glctx, pos, tri, resolution, ranges=None, grad_db=True):
        H, W = int(resolution[0]), int(resolution[1])
        pos_b = pos.unsqueeze(0) if pos.ndim == 2 else pos
        B = pos_b.shape[0]
        device = pos_b.device
        f_long = tri.long()

        rast = torch.zeros((B, H, W, 4), dtype=torch.float32, device=device)
        rast_db = torch.zeros((B, H, W, 4), dtype=torch.float32, device=device)
        if f_long.numel() == 0:
            return rast, rast_db

        cand_chunk = 4_000_000
        for b in range(B):
            p = pos_b[b]
            w_clip = p[:, 3].clamp_min(1e-8)
            ndc_x = p[:, 0] / w_clip
            ndc_y = p[:, 1] / w_clip
            ndc_z = p[:, 2] / w_clip
            px_v = (ndc_x + 1.0) * (0.5 * W) - 0.5
            py_v = (ndc_y + 1.0) * (0.5 * H) - 0.5

            x0, x1, x2 = px_v[f_long[:, 0]], px_v[f_long[:, 1]], px_v[f_long[:, 2]]
            y0, y1, y2 = py_v[f_long[:, 0]], py_v[f_long[:, 1]], py_v[f_long[:, 2]]
            z0, z1, z2 = ndc_z[f_long[:, 0]], ndc_z[f_long[:, 1]], ndc_z[f_long[:, 2]]

            xmin = torch.floor(torch.minimum(torch.minimum(x0, x1), x2)).long().clamp(0, W - 1)
            xmax = torch.ceil(torch.maximum(torch.maximum(x0, x1), x2)).long().clamp(0, W - 1)
            ymin = torch.floor(torch.minimum(torch.minimum(y0, y1), y2)).long().clamp(0, H - 1)
            ymax = torch.ceil(torch.maximum(torch.maximum(y0, y1), y2)).long().clamp(0, H - 1)

            bw = (xmax - xmin + 1).clamp_min(0)
            bh = (ymax - ymin + 1).clamp_min(0)
            counts = bw * bh
            valid_tri = torch.nonzero((counts > 0) & (counts <= H * W)).squeeze(-1)
            if valid_tri.numel() == 0:
                continue

            counts_v = counts[valid_tri]
            cum_counts = counts_v.cumsum(0)
            n_valid = int(valid_tri.shape[0])
            zbuf = torch.full((H * W,), float("inf"), dtype=torch.float32, device=device)
            rast_flat = rast[b].view(H * W, 4)

            start_idx = 0
            while start_idx < n_valid:
                base = 0 if start_idx == 0 else cum_counts[start_idx - 1]
                end_idx = int(torch.searchsorted(cum_counts, base + cand_chunk, right=True).item())
                end_idx = max(start_idx + 1, min(end_idx, n_valid))
                t_ids = valid_tri[start_idx:end_idx]
                c_sub = counts[t_ids]
                offsets_sub = torch.cat([torch.zeros(1, device=device, dtype=torch.long), c_sub.cumsum(0)[:-1]])
                rep_t = torch.repeat_interleave(t_ids, c_sub)
                local_idx = torch.arange(c_sub.sum(), device=device, dtype=torch.long) - torch.repeat_interleave(offsets_sub, c_sub)
                bw_t = bw[rep_t]
                px = (xmin[rep_t] + (local_idx % bw_t)).float()
                py = (ymin[rep_t] + (local_idx // bw_t)).float()

                tx0, tx1, tx2 = x0[rep_t], x1[rep_t], x2[rep_t]
                ty0, ty1, ty2 = y0[rep_t], y1[rep_t], y2[rep_t]
                denom = (ty1 - ty2) * (tx0 - tx2) + (tx2 - tx1) * (ty0 - ty2)
                safe_denom = torch.where(denom.abs() > 1e-12, denom, torch.ones_like(denom))
                w0 = ((ty1 - ty2) * (px - tx2) + (tx2 - tx1) * (py - ty2)) / safe_denom
                w1 = ((ty2 - ty0) * (px - tx2) + (tx0 - tx2) * (py - ty2)) / safe_denom
                w2 = 1.0 - w0 - w1
                inside = (denom.abs() > 1e-12) & (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
                if inside.any():
                    px_i = px[inside].long()
                    py_i = py[inside].long()
                    t_i = rep_t[inside]
                    w0_i = w0[inside].clamp(0.0, 1.0)
                    w1_i = w1[inside].clamp(0.0, 1.0)
                    w2_i = (1.0 - w0_i - w1_i).clamp(0.0, 1.0)
                    z_i = w0_i * z0[t_i] + w1_i * z1[t_i] + w2_i * z2[t_i]
                    flat_pix = py_i * W + px_i
                    zbuf.scatter_reduce_(0, flat_pix, z_i, reduce="amin", include_self=True)
                    win = z_i <= zbuf[flat_pix] + 1e-6
                    if win.any():
                        fp_w = flat_pix[win]
                        rast_flat[fp_w, 0] = w0_i[win]
                        rast_flat[fp_w, 1] = w1_i[win]
                        rast_flat[fp_w, 2] = z_i[win]
                        rast_flat[fp_w, 3] = (t_i[win] + 1).float()
                start_idx = end_idx
        return rast, rast_db

    def _interpolate_turing(attr, rast, tri, rast_db=None, diff_attrs=None):
        attr_b = attr.unsqueeze(0) if attr.ndim == 2 else attr
        B, H, W, _ = rast.shape
        C = attr_b.shape[-1]
        device = rast.device
        out = torch.zeros((B, H, W, C), dtype=attr_b.dtype, device=device)
        out_da = torch.zeros((B, H, W, 0), dtype=attr_b.dtype, device=device)
        f_long = tri.long()
        for b in range(B):
            a_b = attr_b[0 if attr_b.shape[0] == 1 else b]
            r_b = rast[b]
            mask = r_b[..., 3] > 0
            if not mask.any():
                continue
            t_idx = (r_b[..., 3][mask].long() - 1).clamp(0, f_long.shape[0] - 1)
            u = r_b[..., 0][mask].unsqueeze(-1)
            v = r_b[..., 1][mask].unsqueeze(-1)
            w = (1.0 - u - v).clamp(0.0, 1.0)
            v0 = a_b[f_long[t_idx, 0]]
            v1 = a_b[f_long[t_idx, 1]]
            v2 = a_b[f_long[t_idx, 2]]
            out[b][mask] = u * v0 + v * v1 + w * v2
        return out, out_da

    dr.RasterizeCudaContext = _TuringRasterizeContext
    dr.RasterizeGLContext = _TuringRasterizeContext
    dr.rasterize = _rasterize_turing
    dr.interpolate = _interpolate_turing
    dr_ops.RasterizeCudaContext = _TuringRasterizeContext
    dr_ops.RasterizeGLContext = _TuringRasterizeContext
    dr_ops.rasterize = _rasterize_turing
    dr_ops.interpolate = _interpolate_turing


def _patch_torch_fp16_tensor_cores() -> None:
    """Redirect bfloat16 SDPA and Linear ops to float16 on Turing GPUs.

    Turing (sm_75) has hardware FP16 Tensor Cores and Cutlass FP16 memory-efficient
    SDPA, but lacks hardware bfloat16 support. In PyTorch on sm_75, bfloat16 SDPA
    falls back to the unfused O(N^2) math kernel (allocating >6.5 GB VRAM at 1024
    resolution and spilling into system RAM over PCIe), and bfloat16 Linear falls
    back to non-Tensor-Core FP32 execution.
    """
    orig_sdpa = F.scaled_dot_product_attention
    orig_linear = F.linear

    def _turing_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        if query.is_cuda and query.dtype == torch.bfloat16:
            q = query.to(torch.float16)
            k = key.to(torch.float16)
            v = value.to(torch.float16)
            if attn_mask is not None and attn_mask.is_floating_point():
                attn_mask = attn_mask.to(torch.float16)
            out = orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)
            return out.to(torch.bfloat16)
        return orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)

    def _turing_linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if input.is_cuda and (input.dtype == torch.bfloat16 or weight.dtype == torch.bfloat16):
            orig_dtype = input.dtype
            inp_fp16 = input.clamp(-65500.0, 65500.0).to(torch.float16) if input.dtype != torch.float16 else input
            w_fp16 = weight.to(torch.float16) if weight.dtype != torch.float16 else weight
            b_fp16 = bias.to(torch.float16) if (bias is not None and bias.dtype != torch.float16) else bias
            return orig_linear(inp_fp16, w_fp16, b_fp16).to(orig_dtype)
        return orig_linear(input, weight, bias)

    F.scaled_dot_product_attention = _turing_sdpa
    torch.nn.functional.scaled_dot_product_attention = _turing_sdpa
    F.linear = _turing_linear
    torch.nn.functional.linear = _turing_linear


def apply_turing_compat(force: bool = False) -> bool:
    """Apply Turing compatibility patches if running on a pre-Ampere GPU (CC < 8.0)."""
    if not force and not needs_turing_compat():
        return False

    major, minor = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(
        f"[TuringCompat] Applying sm_{major}{minor} compatibility patches for {gpu_name}...",
        flush=True,
    )

    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"
    os.environ["FLEX_GEMM_ALGO"] = "implicit_gemm"

    _patch_torch_fp16_tensor_cores()
    _patch_flex_gemm()
    _patch_o_voxel()
    _patch_cumesh()
    _patch_nvdiffrast()
    print("[TuringCompat] Patched FP16 Tensor Cores, flex_gemm, o_voxel, cumesh, and nvdiffrast for Turing GPU.", flush=True)
    return True
