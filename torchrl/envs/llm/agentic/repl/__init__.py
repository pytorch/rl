# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""REPL backends for the agentic toolkit.

- :class:`JupyterRepl` -- IPython-kernel-backed; rich outputs, clean
  restarts. Optional dependency on ``jupyter_client``.
- :class:`SubprocessRepl` -- persistent ``python3`` subprocess; no extra
  dependency, no rich display.
"""
from __future__ import annotations

from .base import Repl, ReplDisplay, ReplError, ReplResult
from .jupyter import _has_jupyter_client, JupyterRepl
from .subprocess import SubprocessRepl

__all__ = [
    "JupyterRepl",
    "Repl",
    "ReplDisplay",
    "ReplError",
    "ReplResult",
    "SubprocessRepl",
    "_has_jupyter_client",
]
