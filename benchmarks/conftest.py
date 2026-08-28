# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import time
import warnings
from collections import defaultdict

import pytest
import torch
from torchrl._utils import logger as torchrl_logger

CALL_TIMES = defaultdict(float)
CUDA_MEMORY_STATS = {}
CUDA_MEMORY_STATS_ATTEMPTED = False


def _format_memory(num_bytes: int | float) -> str:
    return f"{num_bytes / 1024**2:8.2f} MiB"


def _node_name(request: pytest.FixtureRequest, *, include_params: bool) -> str:
    name = request.node.name
    class_name = request.cls.__name__ if request.cls else None
    if not include_params:
        name = name.split("[")[0]
    if class_name is not None:
        name = "::".join([class_name, name])
    file = os.path.basename(request.path)
    return f"{file}::{name}"


def _log_call_times(maxprint: int) -> None:
    out_str = """
Call times:
===========
"""
    keys = list(CALL_TIMES.keys())
    if len(keys) > 1:
        maxchar = max(*[len(key) for key in keys])
    elif len(keys) == 1:
        maxchar = len(keys[0])
    else:
        return
    for i, (key, item) in enumerate(
        sorted(CALL_TIMES.items(), key=lambda x: x[1], reverse=True)
    ):
        spaces = "  " + " " * (maxchar - len(key))
        out_str += f"\t{key}{spaces}{item: 4.4f}s\n"
        if i == maxprint - 1:
            break
    torchrl_logger.info(out_str)


def _cuda_memory_stats_report(maxprint: int) -> str | None:
    if not CUDA_MEMORY_STATS:
        if CUDA_MEMORY_STATS_ATTEMPTED and torch.cuda.device_count():
            return "CUDA memory stats were requested but none were recorded."
        return None
    out_str = """
CUDA memory peaks during benchmarks:
====================================
"""
    maxchar = max(len(key) for key in CUDA_MEMORY_STATS)
    rows = sorted(
        CUDA_MEMORY_STATS.items(),
        key=lambda x: x[1].get("cuda_peak_allocated_delta_bytes", 0),
        reverse=True,
    )
    for i, (key, stats) in enumerate(rows):
        spaces = "  " + " " * (maxchar - len(key))
        out_str += (
            f"\t{key}{spaces}"
            f"peak alloc Δ {_format_memory(stats['cuda_peak_allocated_delta_bytes'])}  "
            "peak reserved Δ "
            f"{_format_memory(stats['cuda_peak_reserved_delta_bytes'])}  "
            f"max alloc {_format_memory(stats['cuda_max_allocated_bytes'])}  "
            f"max reserved {_format_memory(stats['cuda_max_reserved_bytes'])}\n"
        )
        if i == maxprint - 1:
            break
    return out_str


def _log_cuda_memory_stats(maxprint: int) -> None:
    out_str = _cuda_memory_stats_report(maxprint)
    if out_str is not None:
        torchrl_logger.info(out_str)


def pytest_sessionfinish(maxprint=50):
    if not isinstance(maxprint, int):
        maxprint = 50
    _log_call_times(maxprint)


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    out_str = _cuda_memory_stats_report(maxprint=50)
    if out_str is not None:
        terminalreporter.write_line(out_str)


@pytest.fixture(autouse=True)
def measure_duration(request: pytest.FixtureRequest):
    start_time = time.time()

    def fin():
        duration = time.time() - start_time
        name = _node_name(request, include_params=False)
        CALL_TIMES[name] = CALL_TIMES[name] + duration

    request.addfinalizer(fin)


@pytest.fixture
def record_cuda_memory_stats(request: pytest.FixtureRequest):
    def record(benchmark, stats: dict[str, int | float] | None) -> None:
        global CUDA_MEMORY_STATS_ATTEMPTED
        CUDA_MEMORY_STATS_ATTEMPTED = True
        if stats is None:
            return
        name = _node_name(request, include_params=True)
        CUDA_MEMORY_STATS[name] = stats
        benchmark.extra_info.update(stats)

    return record


def pytest_addoption(parser):
    parser.addoption("--rank", action="store")


def pytest_configure(config: pytest.Config) -> None:
    try:
        torch._dynamo.config.reorderable_logging_functions.add(warnings.warn)
    except AttributeError:
        pass


@pytest.fixture(scope="session", autouse=True)
def set_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"Lazy modules are a new feature under heavy development",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"Couldn't cast the policy onto the desired device on remote process",
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r"Deprecated call to `pkg_resources.declare_namespace",
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r"Using or importing the ABCs",
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r"Please use `coo_matrix` from the `scipy.sparse` namespace",
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r"jax.tree_util.register_keypaths is deprecated|jax.ShapedArray is deprecated",
    )
