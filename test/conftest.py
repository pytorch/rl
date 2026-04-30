# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import os
import random
import sys
import time
import warnings
from collections import defaultdict

import numpy as np

import pytest
import torch

CALL_TIMES = defaultdict(float)
IS_OSX = sys.platform == "darwin"


def pytest_sessionfinish(session, exitstatus, maxprint=50):
    """Print aggregated test times per function (across all parametrizations)."""
    keys = list(CALL_TIMES.keys())
    if not keys:
        return

    # Calculate total time
    total_time = sum(CALL_TIMES.values())

    out_str = f"""
================================================================================
AGGREGATED TEST TIMES (by function, across all parametrizations)
================================================================================
Total test time: {total_time:.1f}s ({total_time / 60:.1f} min)
Top {min(maxprint, len(keys))} slowest test functions:
--------------------------------------------------------------------------------
"""
    maxchar = max(len(key) for key in keys)
    for i, (key, item) in enumerate(
        sorted(CALL_TIMES.items(), key=lambda x: x[1], reverse=True)
    ):
        spaces = " " * (maxchar - len(key) + 2)
        pct = (item / total_time) * 100 if total_time > 0 else 0
        out_str += f"  {key}{spaces}{item:7.2f}s  ({pct:5.1f}%)\n"
        if i == maxprint - 1:
            break

    out_str += "================================================================================\n"
    sys.stdout.write(out_str)


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
        category=UserWarning,
        message=r"A lambda function was passed to ParallelEnv",
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

    parser.addoption(
        "--unity_editor",
        action="store_true",
        default=False,
        help="Run tests that require manually pressing play in the Unity editor.",
    )


def pytest_runtest_setup(item):
    if "unity_editor" in item.keywords and not item.config.getoption("--unity_editor"):
        pytest.skip("need --unity_editor option to run this test")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring a GPU (CUDA device)"
    )


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


# LLM testing fixtures
@pytest.fixture
def mock_transformer_model():
    """Fixture that provides a mock transformer model factory."""
    from torchrl.testing import MockTransformerModel

    def _make_model(
        vocab_size: int = 1024, device: torch.device | str | int = "cpu"
    ) -> MockTransformerModel:
        """Make a mock transformer model."""
        device = torch.device(device)
        return MockTransformerModel(vocab_size, device)

    return _make_model


@pytest.fixture
def mock_tokenizer():
    """Fixture that provides a mock tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")


@pytest.fixture(autouse=True)
def prevent_leaking_rng():
    # Prevent each test from leaking the rng to all other test when they call
    # torch.manual_seed() or random.seed() or np.random.seed().
    # Note: the numpy rngs should never leak anyway, as we never use
    # np.random.seed() and instead rely on np.random.RandomState instances

    torch_rng_state = torch.get_rng_state()
    builtin_rng_state = random.getstate()
    nunmpy_rng_state = np.random.get_state()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()

    yield

    torch.set_rng_state(torch_rng_state)
    random.setstate(builtin_rng_state)
    np.random.set_state(nunmpy_rng_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng_state)
