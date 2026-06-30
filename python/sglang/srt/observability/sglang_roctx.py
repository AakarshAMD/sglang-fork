# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""roctx marker helper for SGLang request-time-stats instrumentation.

ADDITIVE / OPT-IN. This module is part of the private roctx fork; it is a pure
addition (no upstream file is modified destructively). Every marker is gated
behind the ``SGLANG_ROCTX=1`` environment variable and is an exception-safe
no-op when the gate is unset → **zero behavioral change and zero overhead when
off**.

Why roctx (and not torch/kineto): roctx markers live in roctracer's external
"roctx" domain. torch/kineto does NOT ingest that domain, so these marks are
INVISIBLE to a torch profiler trace. They only surface under
``rocprofv3 --marker-trace`` (or RTL-lite's roctx shim). Capture accordingly.

Mechanism: we bind ``roctxMarkA`` from ``libroctx64.so`` via ctypes (no need to
link the SGLang/MORI build against libroctx). ``roctxMarkA`` emits an *instant*
marker (a point on the timeline), which is the right primitive here because a
request's lifecycle stamps fire across multiple threads/processes (scheduler
event loop, tokenizer/API frontend, TP workers) — a push/pop range would have
to begin and end on the same thread, which these stamps do not. Read a
start/end window as ``ts(end_mark) - ts(start_mark)`` for a matching request id.
"""

from __future__ import annotations

import os

__all__ = ["ROCTX_ENABLED", "roctx_mark", "roctx_available"]

# Gate read once at import. Accept "1" (and, defensively, common truthy spellings).
ROCTX_ENABLED = os.environ.get("SGLANG_ROCTX", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# _emit(name: bytes) -> None. Default is a no-op; replaced by the real ctypes
# binding below only when the gate is on AND libroctx64.so loads cleanly.
_emit = None  # type: ignore[assignment]


def _init_backend() -> None:
    """Bind roctxMarkA from libroctx64.so. Best-effort; leaves _emit=None on failure."""
    global _emit
    try:
        import ctypes

        # IMPORTANT: rocprofiler-sdk's `rocprofv3 --marker-trace` only intercepts
        # the rocprofiler-sdk ROCTx library (librocprofiler-sdk-roctx.so), NOT the
        # legacy roctracer libroctx64.so. Proven on ROCm 7.0: marks via
        # librocprofiler-sdk-roctx.so render as slices in the pftrace; libroctx64.so
        # marks do not. So prefer the sdk lib; fall back to libroctx64 (e.g. for an
        # RTL/roctracer capture context). All ship in /opt/rocm*/lib.
        lib = None
        for cand in (
            "librocprofiler-sdk-roctx.so",
            "librocprofiler-sdk-roctx.so.1",
            "libroctx64.so",
            "libroctx64.so.4",
        ):
            try:
                lib = ctypes.CDLL(cand)
                break
            except OSError:
                continue
        if lib is None:
            return
        fn = lib.roctxMarkA
        fn.argtypes = [ctypes.c_char_p]
        fn.restype = None

        def _emit_impl(name_bytes, _fn=fn):  # bound default avoids global lookups
            _fn(name_bytes)

        _emit = _emit_impl
    except Exception:
        # Any failure (missing lib, missing symbol, etc.) → stay a no-op.
        _emit = None


if ROCTX_ENABLED:
    _init_backend()


def roctx_available() -> bool:
    """True iff the gate is on and the roctx backend bound successfully."""
    return ROCTX_ENABLED and _emit is not None


def roctx_mark(message: str) -> None:
    """Emit an instant roctx marker. No-op (and never raises) when disabled.

    Safe to call on any hot path: when ``SGLANG_ROCTX`` is unset this returns
    immediately after a single boolean check and the backend is never bound.
    """
    if not ROCTX_ENABLED or _emit is None:
        return
    try:
        _emit(message.encode("ascii", "replace"))
    except Exception:
        # Observability must never break serving.
        pass
