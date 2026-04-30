from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import psutil

__all__ = [
    "assert_no_new_python_processes",
    "is_python_process",
    "snapshot_python_processes",
]


def is_python_process(comm: str | None, args: str | None) -> bool:
    """Check if a process is a python process."""
    if comm is None:
        comm = ""
    comm = comm.lower()
    if comm.startswith(("python", "pypy")):
        return True
    if not args:
        return False
    return "python" in args.lower()


def snapshot_python_processes(
    root: psutil.Process | None = None,
) -> dict[tuple[int, float], dict[str, Any]]:
    """Snapshot python processes belonging to the given process tree.

    Returns a dict keyed by (pid, start_time) -> info.
    """
    if root is None:
        root = psutil.Process(os.getpid())

    uid = os.getuid()

    # Snapshot descendant PIDs first, then query process info via process_iter.
    # This avoids race conditions where a child exits between `children()` and
    # attribute access on a stale Process handle (common with Ray helpers).
    descendant_pids = {root.pid}
    descendant_pids.update(p.pid for p in root.children(recursive=True))

    out: dict[tuple[int, float], dict[str, Any]] = {}
    for proc in psutil.process_iter(
        attrs=["pid", "name", "cmdline", "create_time", "uids"], ad_value=None
    ):
        info = proc.info
        pid = info.get("pid")
        if pid is None or pid not in descendant_pids:
            continue
        uids = info.get("uids")
        if uids is None or uids.real != uid:
            continue

        name = info.get("name") or ""
        cmdline = info.get("cmdline") or []
        args = " ".join(cmdline) if isinstance(cmdline, (list, tuple)) else str(cmdline)
        if not is_python_process(name, args):
            continue

        start_time = float(info.get("create_time") or 0.0)
        key = (int(pid), start_time)
        out[key] = {
            "pid": int(pid),
            "start_time": start_time,
            "comm": name,
            "args": args,
        }
    return out


def assert_no_new_python_processes(
    *,
    baseline: dict[tuple[int, float], dict[str, Any]],
    baseline_time: float,
    timeout: float = 20.0,
    ignore_info_fn: Callable[[dict[str, Any]], bool] | None = None,
) -> None:
    """Assert that no python process started after baseline_time remains alive.

    The check is limited to the current process tree (pytest process + descendants).
    """
    if ignore_info_fn is None:

        def ignore_info_fn(_info: dict[str, Any]) -> bool:
            return False

    deadline = time.time() + timeout
    last_new: dict[tuple[int, float], dict[str, Any]] | None = None
    while time.time() < deadline:
        current = snapshot_python_processes()
        new: dict[tuple[int, float], dict[str, Any]] = {}
        for (pid, start_time), info in current.items():
            if pid == os.getpid():
                continue
            if ignore_info_fn(info):
                continue
            # Guard against pid reuse: only consider processes started after the baseline.
            if start_time and start_time < baseline_time - 1.0:
                continue
            if (pid, start_time) in baseline:
                continue
            new[(pid, start_time)] = info
        if not new:
            return
        last_new = new
        time.sleep(0.25)

    if last_new is None:
        return
    details = "\n".join(
        f"- pid={v['pid']} comm={v.get('comm')} args={v.get('args')}"
        for v in last_new.values()
    )
    raise AssertionError(
        "Leaked python processes detected after collector.shutdown().\n"
        f"Processes still alive:\n{details}"
    )
