# Third-party notices

AISmith 3D incorporates or integrates the following third-party projects. This file is an attribution guide, not a replacement for the full license texts retained in vendored directories or supplied by runtime downloads.

| Project | Use in AISmith 3D | License / source |
| --- | --- | --- |
| [Mesh2Motion](https://github.com/Mesh2Motion/mesh2motion-app) | Vendored same-origin Rig + Animate workspace | MIT; source text: `vendor/mesh2motion-app/LICENSE-MIT.MD`. Included Mesh2Motion models, rigs, and animation assets are CC0 1.0 Universal; source text: `vendor/mesh2motion-app/LICENSE-CC0.MD`. Copyright (c) 2025 Scott Petrovic. |
| [trellis2cpp](https://github.com/rms80/trellis2cpp) | Vendored optional native C++/ggml implementation | MIT; source text: `vendor/trellis2cpp/LICENSE`. Copyright (c) 2026 rms80. |
| [AutoRemesher](https://github.com/huxingyi/autoremesher) | Vendored optional remeshing source / executable integration | MIT; source text: `vendor/autoremesher/LICENSE`. Its own acknowledgements cover its bundled dependencies. |
| [ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2) | Vendored TRELLIS.2 node code used by the isolated refiner | MIT; source text: `vendor/ComfyUI-Trellis2/LICENSE`. Copyright (c) Microsoft Corporation. |
| [ComfyUI-Trellis2-GGUF](https://github.com/visualbruno/ComfyUI-Trellis2-GGUF) | Vendored optional GGUF-node integration | MIT; source text: `vendor/ComfyUI-Trellis2-GGUF/LICENSE`. Copyright (c) Microsoft Corporation. |
| [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | Underlying image-to-3D research/model implementation | See the upstream repository and model-card terms. |
| [trellis.cpp](https://github.com/ilintar/trellis.cpp) | Downloaded native CUDA geometry runtime | Downloaded on demand; see its upstream repository/release license. |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | Downloaded on demand as a local refiner runtime; no ComfyUI server is exposed | Downloaded on demand; see the upstream repository license. |

Python and JavaScript packages are installed from their respective package registries and remain subject to their own licenses. Model weights are not included in this repository and may impose separate use, redistribution, or attribution terms.
