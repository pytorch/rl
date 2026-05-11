# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Sandbox backends for the agentic toolkit.

The default :func:`default_sandbox` picks bubblewrap on Linux,
sandbox-exec on macOS, and falls back to :class:`UnsafeSubprocessSandbox`
elsewhere (with a :class:`UserWarning`).
"""
from __future__ import annotations

import sys
import warnings

from .base import ResourceLimits, Sandbox, SandboxError, SandboxResult
from .docker import DockerSandbox
from .e2b import E2BSandbox
from .modal import ModalSandbox
from .subprocess_bwrap import _has_bwrap, BubblewrapSandbox
from .subprocess_seatbelt import _has_sandbox_exec, SeatbeltSandbox
from .unsafe import UnsafeSubprocessSandbox


def default_sandbox(limits: ResourceLimits | None = None) -> Sandbox:
    """Return the best available sandbox for the current platform.

    - Linux with ``bwrap`` on PATH -> :class:`BubblewrapSandbox`.
    - macOS with ``sandbox-exec`` on PATH -> :class:`SeatbeltSandbox`.
    - Otherwise -> :class:`UnsafeSubprocessSandbox` with a warning.
    """
    if sys.platform.startswith("linux") and _has_bwrap:
        return BubblewrapSandbox(limits=limits)
    if sys.platform == "darwin" and _has_sandbox_exec:
        return SeatbeltSandbox(limits=limits)
    warnings.warn(
        "No hardened sandbox backend is available on this platform "
        f"({sys.platform!r}). Falling back to UnsafeSubprocessSandbox; "
        "this is fine for tests but NOT for running untrusted model "
        "output.",
        UserWarning,
        stacklevel=2,
    )
    return UnsafeSubprocessSandbox(limits=limits)


__all__ = [
    "BubblewrapSandbox",
    "DockerSandbox",
    "E2BSandbox",
    "ModalSandbox",
    "ResourceLimits",
    "Sandbox",
    "SandboxError",
    "SandboxResult",
    "SeatbeltSandbox",
    "UnsafeSubprocessSandbox",
    "default_sandbox",
]
