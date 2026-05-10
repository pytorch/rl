# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Jupyter-kernel REPL.

Spawns an IPython kernel via :mod:`jupyter_client` and drives it through
the standard ZeroMQ channels. Supports rich display outputs, clean
restarts, and proper interrupts.

Optional dependency: install ``jupyter_client`` and ``ipykernel`` to use.
The import is gated by ``_has_jupyter_client`` and never imported at module
top level for the protocol path -- only inside :class:`JupyterRepl` methods.
"""
from __future__ import annotations

import asyncio
import importlib.util
import queue
from typing import Any, ClassVar

from torchrl._utils import logger as torchrl_logger

from ..sandbox.base import Sandbox, SandboxError
from .base import Repl, ReplDisplay, ReplError, ReplResult

_has_jupyter_client = importlib.util.find_spec("jupyter_client") is not None


_KERNEL_STARTUP_TIMEOUT = 30.0


class JupyterRepl:
    """IPython-kernel-backed REPL with rich outputs.

    Args:
        sandbox: The :class:`Sandbox` the kernel runs inside. Today the
            kernel binary is launched in the host process; binding it to a
            sandbox is on the TODO list (see ``__init__.py``). Treat the
            sandbox as advisory until then.
        kernel_name: Jupyter kernel spec name (default ``"python3"``).

    Raises:
        ImportError: at construction time if ``jupyter_client`` is not
            installed.

    Examples:
        >>> import asyncio  # doctest: +SKIP
        >>> from torchrl.envs.llm.agentic.sandbox import UnsafeSubprocessSandbox
        >>> from torchrl.envs.llm.agentic.repl import JupyterRepl
        >>> async def go():
        ...     async with UnsafeSubprocessSandbox() as s:
        ...         async with JupyterRepl(s) as r:
        ...             await r.execute("x = 41")
        ...             return (await r.execute("print(x + 1)")).stdout.strip()
    """

    name: ClassVar[str] = "jupyter"

    def __init__(
        self,
        sandbox: Sandbox,
        *,
        kernel_name: str = "python3",
    ) -> None:
        if not _has_jupyter_client:
            raise ImportError(
                "JupyterRepl requires jupyter_client. Install with "
                "`pip install jupyter_client ipykernel`."
            )
        self.sandbox = sandbox
        self._kernel_name = kernel_name
        self._km: Any = None
        self._kc: Any = None
        self._exec_count = 0

    async def open(self) -> None:
        if self._km is not None:
            return
        from jupyter_client.manager import KernelManager

        km = KernelManager(kernel_name=self._kernel_name)

        def _start() -> None:
            km.start_kernel()

        await asyncio.get_running_loop().run_in_executor(None, _start)
        kc = km.client()
        kc.start_channels()
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: kc.wait_for_ready(timeout=_KERNEL_STARTUP_TIMEOUT)
            )
        except RuntimeError as e:
            kc.stop_channels()
            km.shutdown_kernel(now=True)
            raise SandboxError(f"jupyter kernel did not become ready: {e}") from e
        self._km = km
        self._kc = kc

    async def close(self) -> None:
        if self._kc is not None:
            try:
                self._kc.stop_channels()
            except Exception:  # pragma: no cover
                pass
            self._kc = None
        if self._km is not None:
            try:
                self._km.shutdown_kernel(now=True)
            except Exception:  # pragma: no cover
                pass
            self._km = None

    async def __aenter__(self) -> JupyterRepl:
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def execute(
        self, code: str, *, timeout: float | None = None
    ) -> ReplResult:
        if self._kc is None:
            raise SandboxError("REPL is not open; call open() first")
        msg_id: str = self._kc.execute(code)
        loop = asyncio.get_running_loop()
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        displays: list[ReplDisplay] = []
        error: ReplError | None = None

        try:
            while True:
                msg = await loop.run_in_executor(
                    None,
                    lambda: _safe_get_iopub(self._kc, timeout or 1e9),
                )
                if msg is None:
                    return ReplResult(timed_out=True)
                parent = msg.get("parent_header") or {}
                if parent.get("msg_id") != msg_id:
                    continue
                mtype = msg.get("msg_type")
                content = msg.get("content") or {}
                if mtype == "stream":
                    if content.get("name") == "stdout":
                        stdout_chunks.append(content.get("text", ""))
                    else:
                        stderr_chunks.append(content.get("text", ""))
                elif mtype in ("execute_result", "display_data"):
                    data = content.get("data") or {}
                    for media_type, payload in data.items():
                        displays.append(
                            ReplDisplay(media_type=media_type, data=payload)
                        )
                elif mtype == "error":
                    error = ReplError(
                        ename=str(content.get("ename", "")),
                        evalue=str(content.get("evalue", "")),
                        traceback="\n".join(content.get("traceback") or ()),
                    )
                elif mtype == "status":
                    if content.get("execution_state") == "idle":
                        break
        except asyncio.TimeoutError:
            try:
                if self._km is not None:
                    self._km.interrupt_kernel()
            except Exception:  # pragma: no cover
                pass
            return ReplResult(timed_out=True)
        self._exec_count += 1
        return ReplResult(
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
            display=tuple(displays),
            error=error,
            timed_out=False,
            execution_count=self._exec_count,
        )

    async def interrupt(self) -> None:
        if self._km is not None:
            try:
                self._km.interrupt_kernel()
            except Exception:  # pragma: no cover
                torchrl_logger.warning("jupyter interrupt failed", exc_info=True)

    async def restart(self) -> None:
        if self._km is None:
            await self.open()
            return
        try:
            self._km.restart_kernel(now=True)
        except Exception:  # pragma: no cover
            await self.close()
            await self.open()
        self._exec_count = 0


def _safe_get_iopub(kc: Any, timeout: float) -> Any | None:
    try:
        return kc.get_iopub_msg(timeout=timeout)
    except queue.Empty:
        return None


__all__ = ["JupyterRepl"]
