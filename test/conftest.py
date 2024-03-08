# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import functools
import os
import sys
import time
import warnings
from collections import defaultdict

import pytest
import torch

CALL_TIMES = defaultdict(lambda: 0.0)
IS_OSX = sys.platform == "darwin"


def pytest_sessionfinish(maxprint=50):
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


@pytest.fixture(autouse=True)
def measure_duration(request: pytest.FixtureRequest):
    start_time = time.time()

    def fin():
        duration = time.time() - start_time
        name = request.node.name
        class_name = request.cls.__name__ if request.cls else None
        name = name.split("[")[0]
        if class_name is not None:
            name = "::".join([class_name, name])
        file = os.path.basename(request.path)
        name = f"{file}::{name}"
        CALL_TIMES[name] = CALL_TIMES[name] + duration

    request.addfinalizer(fin)


@pytest.fixture(autouse=True)
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
        category=UserWarning,
        message=r"Skipping device Apple Paravirtual device",
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


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )

    parser.addoption(
        "--mp_fork",
        action="store_true",
        default=False,
        help="Use 'fork' start method for mp dedicated tests.",
    )

    parser.addoption(
        "--mp_fork_if_no_cuda",
        action="store_true",
        default=False,
        help="Use 'fork' start method for mp dedicated tests only if there is no cuda device available.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def maybe_fork_ParallelEnv(request):
    from torchrl.envs import ParallelEnv

    if not IS_OSX and (
        request.config.getoption("--mp_fork")
        or (
            request.config.getoption("--mp_fork_if_no_cuda")
            and not torch.cuda.is_available()
        )
    ):
        return functools.partial(ParallelEnv, mp_start_method="fork")
    return ParallelEnv
