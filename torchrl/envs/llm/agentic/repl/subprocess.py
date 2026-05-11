# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Subprocess-backed REPL.

Spawns a Python subprocess inside a :class:`Sandbox` and feeds it code via
stdin, reading stdout/stderr after each delimiter. State persists across
:meth:`execute` calls because the subprocess is long-lived. No rich
display protocol -- use :class:`JupyterRepl` for that.

Implementation note: this is *not* a perfect REPL. Each ``execute`` call
sends the full block + a sentinel print; we read until the sentinel
appears. Errors are returned as a :class:`ReplError` parsed from stderr.
"""
from __future__ import annotations

import asyncio
import os
import signal
import textwrap
import uuid
from typing import ClassVar

from ..sandbox.base import Sandbox, SandboxError
from .base import ReplError, ReplResult


_BOOT = textwrap.dedent(
    """\
    import sys, traceback
    _NS = {}
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            sentinel, n_lines = line.strip().split(' ', 1)
            n_lines = int(n_lines)
            code = ''.join(sys.stdin.readline() for _ in range(n_lines))
            try:
                exec(compile(code, '<repl>', 'exec'), _NS)
                err = None
            except BaseException:
                err = traceback.format_exc()
            sys.stdout.write(sentinel + '_END\\n')
            sys.stdout.flush()
            if err is not None:
                sys.stderr.write(err)
                sys.stderr.write(sentinel + '_ERR\\n')
                sys.stderr.flush()
            else:
                sys.stderr.write(sentinel + '_OK\\n')
                sys.stderr.flush()
        except Exception:
            traceback.print_exc()
    """
)


class SubprocessRepl:
    """Persistent Python subprocess used as a REPL.

    Args:
        sandbox: The :class:`Sandbox` the subprocess runs inside. Must be
            opened separately (or via ``async with``).
        python_argv: Argv used to launch the interpreter (default
            ``["python3", "-u", "-c", _BOOT]``).

    Examples:
        >>> import asyncio  # doctest: +SKIP
        >>> from torchrl.envs.llm.agentic.sandbox import UnsafeSubprocessSandbox
        >>> from torchrl.envs.llm.agentic.repl import SubprocessRepl
        >>> async def go():
        ...     async with UnsafeSubprocessSandbox() as s:
        ...         async with SubprocessRepl(s) as r:
        ...             await r.execute("x = 1")
        ...             out = await r.execute("print(x)")
        ...             return out.stdout.strip()
        >>> asyncio.run(go())
    """

    name: ClassVar[str] = "subprocess"

    def __init__(
        self,
        sandbox: Sandbox,
        *,
        python_argv: tuple[str, ...] = ("python3", "-u", "-c", _BOOT),
    ) -> None:
        self.sandbox = sandbox
        self._argv = python_argv
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            return
        # Bypass sandbox.run() for the long-lived process: we need the
        # subprocess handle, not just stdout/stderr at the end. We honor
        # sandbox lifecycle but spawn the process directly inside it.
        # For UnsafeSubprocessSandbox this is plain create_subprocess_exec;
        # for hardened backends, we ask the sandbox to wrap the argv.
        argv = await _wrap_argv_via_sandbox(self.sandbox, self._argv)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise SandboxError(f"could not start REPL: {e}") from e

    async def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.returncode is None:
                self._proc.kill()
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=2.0)
                except TimeoutError:  # pragma: no cover
                    pass
        finally:
            self._proc = None

    async def __aenter__(self) -> SubprocessRepl:
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def execute(self, code: str, *, timeout: float | None = None) -> ReplResult:
        if self._proc is None or self._proc.returncode is not None:
            raise SandboxError("REPL is not running; call open() first")
        async with self._lock:
            sentinel = "S" + uuid.uuid4().hex
            n_lines = code.count("\n") + 1
            payload = f"{sentinel} {n_lines}\n{code}"
            if not payload.endswith("\n"):
                payload += "\n"
            assert self._proc.stdin is not None
            self._proc.stdin.write(payload.encode("utf-8"))
            try:
                await self._proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as e:
                raise SandboxError(f"REPL stdin closed: {e}") from e
            try:
                stdout, stderr = await asyncio.wait_for(
                    self._read_until_sentinels(sentinel), timeout=timeout
                )
                err: ReplError | None = None
                if stderr.endswith(f"{sentinel}_ERR\n"):
                    body = stderr[: -len(f"{sentinel}_ERR\n")]
                    err = _parse_traceback(body)
                else:
                    # Strip the OK marker.
                    if stderr.endswith(f"{sentinel}_OK\n"):
                        stderr = stderr[: -len(f"{sentinel}_OK\n")]
                stdout_clean = stdout
                if stdout_clean.endswith(f"{sentinel}_END\n"):
                    stdout_clean = stdout_clean[: -len(f"{sentinel}_END\n")]
                return ReplResult(
                    stdout=stdout_clean,
                    stderr=stderr,
                    error=err,
                    timed_out=False,
                    execution_count=-1,
                )
            except TimeoutError:
                # Send SIGINT and let the boot loop recover. State is
                # preserved unless the user code is in an uninterruptible
                # syscall, in which case the user must call restart().
                try:
                    if self._proc.pid is not None:
                        os.kill(self._proc.pid, signal.SIGINT)
                except ProcessLookupError:  # pragma: no cover
                    pass
                return ReplResult(stdout="", stderr="", timed_out=True)

    async def interrupt(self) -> None:
        if self._proc is None or self._proc.returncode is not None:
            return
        try:
            if self._proc.pid is not None:
                os.kill(self._proc.pid, signal.SIGINT)
        except ProcessLookupError:  # pragma: no cover
            pass

    async def restart(self) -> None:
        await self.close()
        await self.open()

    async def _read_until_sentinels(self, sentinel: str) -> tuple[str, str]:
        # Read stdout until "<sentinel>_END\n" appears, then drain stderr
        # until "<sentinel>_OK\n" or "<sentinel>_ERR\n" appears.
        assert self._proc is not None
        out_buf: list[bytes] = []
        end = f"{sentinel}_END\n".encode()
        assert self._proc.stdout is not None
        while True:
            chunk = await self._proc.stdout.readline()
            if not chunk:
                break
            out_buf.append(chunk)
            if chunk == end:
                break
        err_buf: list[bytes] = []
        ok = f"{sentinel}_OK\n".encode()
        e_err = f"{sentinel}_ERR\n".encode()
        assert self._proc.stderr is not None
        while True:
            chunk = await self._proc.stderr.readline()
            if not chunk:
                break
            err_buf.append(chunk)
            if chunk == ok or chunk == e_err:
                break
        return (
            b"".join(out_buf).decode("utf-8", errors="replace"),
            b"".join(err_buf).decode("utf-8", errors="replace"),
        )


def _parse_traceback(tb: str) -> ReplError:
    """Parse the last line of a traceback into ``ename: evalue``."""
    lines = [line for line in tb.splitlines() if line]
    if not lines:
        return ReplError(ename="Error", evalue="", traceback=tb)
    last = lines[-1]
    if ":" in last:
        ename, _, evalue = last.partition(":")
        return ReplError(ename=ename.strip(), evalue=evalue.strip(), traceback=tb)
    return ReplError(ename=last.strip(), evalue="", traceback=tb)


async def _wrap_argv_via_sandbox(sandbox: Sandbox, argv: tuple[str, ...]) -> list[str]:
    """Ask the sandbox to compute the prefixed argv if it supports it.

    Falls back to the raw argv if the backend doesn't expose a
    ``_build_argv`` hook. Best-effort.
    """
    builder = getattr(sandbox, "_build_argv", None)
    if callable(builder):
        try:
            return list(builder(list(argv), sandbox.limits, None))
        except TypeError:
            try:
                return list(builder(list(argv), sandbox.limits))
            except Exception:
                return list(argv)
        except Exception:
            return list(argv)
    return list(argv)


__all__ = ["SubprocessRepl"]
