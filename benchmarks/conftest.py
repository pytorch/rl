# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import time
import warnings
from collections import defaultdict

import pytest
from torchrl._utils import logger as torchrl_logger

CALL_TIMES = defaultdict(lambda: 0.0)


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
    torchrl_logger.info(out_str)


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


def pytest_addoption(parser):
    parser.addoption("--rank", action="store")


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
