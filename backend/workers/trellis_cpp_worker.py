"""Compatibility entrypoint for the native trellis.cpp command line.

The actual inference implementation deliberately lives in trellis.cpp, not in
Python or a ComfyUI node wrapper.  Keeping this tiny module gives local users a
single obvious integration boundary while native.run_trellis owns invocation.
"""

from __future__ import annotations

raise SystemExit("trellis.cpp is launched directly by backend.app.native")
