# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""General testing utilities for TorchRL tests."""

from __future__ import annotations

# Version for testing implement_for decorator
__version__ = "0.3"

import contextlib
import logging
import sys
import time
import unittest
from collections.abc import Callable
from functools import wraps

import pytest
import torch
import torch.cuda
from tensordict import tensorclass

from torchrl._utils import logger, seed_generator

__all__ = [
    "capture_log_records",
    "dtype_fixture",
    "generate_seeds",
    "get_available_devices",
    "get_default_devices",
    "IS_WIN",
    "make_tc",
    "mp_ctx",
    "PYTHON_3_9",
    "retry",
    "set_global_var",
]

IS_WIN = sys.platform == "win32"
if IS_WIN:
    mp_ctx = "spawn"
else:
    mp_ctx = "fork"

PYTHON_3_9 = sys.version_info.major == 3 and sys.version_info.minor <= 9


def get_available_devices():
    """Return a list of all available torch devices (CPU and all CUDA devices)."""
    devices = [torch.device("cpu")]
    n_cuda = torch.cuda.device_count()
    if n_cuda > 0:
        for i in range(n_cuda):
            devices += [torch.device(f"cuda:{i}")]
    return devices


def get_default_devices():
    """Return a sensible default list of devices for testing.

    Returns [cpu] if no CUDA, [cuda:0] if one GPU, all devices if multiple GPUs.
    """
    num_cuda = torch.cuda.device_count()
    if num_cuda == 0:
        return [torch.device("cpu")]
    elif num_cuda == 1:
        return [torch.device("cuda:0")]
    else:
        return get_available_devices()


def generate_seeds(seed, repeat):
    """Generate a list of seeds from a starting seed using the seed_generator."""
    seeds = [seed]
    for _ in range(repeat - 1):
        seed = seed_generator(seed)
        seeds.append(seed)
    return seeds


def retry(
    ExceptionToCheck: type[Exception],
    tries: int = 3,
    delay: int = 3,
    skip_after_retries: bool = False,
) -> Callable[[Callable], Callable]:
    """Decorator to retry a function upon certain Exceptions.

    Args:
        ExceptionToCheck: The exception type to catch and retry.
        tries: Number of attempts before giving up.
        delay: Seconds to wait between retries.
        skip_after_retries: If True, skip the test after all retries fail.

    Returns:
        A decorator that wraps the function with retry logic.
    """

    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToCheck as e:
                    msg = "%s, Retrying in %d seconds..." % (str(e), mdelay)
                    logger.info(msg)
                    time.sleep(mdelay)
                    mtries -= 1
            try:
                return f(*args, **kwargs)
            except ExceptionToCheck as e:
                if skip_after_retries:
                    raise pytest.skip(
                        f"Skipping after {tries} consecutive {str(e)}"
                    ) from e
                else:
                    raise e

        return f_retry

    return deco_retry


def capture_log_records(records, logger_qname, record_name):
    """Capture log records matching a name pattern from a specific logger.

    After calling this function, any log record whose name contains 'record_name'
    and is emitted from the logger that has qualified name 'logger_qname' is
    appended to the 'records' list.

    NOTE: This function is based on testing utilities for 'torch._logging'.
    """
    assert isinstance(records, list)
    log = logging.getLogger(logger_qname)

    class EmitWrapper:
        def __init__(self, old_emit):
            self.old_emit = old_emit

        def __call__(self, record):
            nonlocal records  # noqa: F824
            self.old_emit(record)
            if record_name in record.name:
                records.append(record)

    for handler in log.handlers:
        new_emit = EmitWrapper(handler.emit)
        contextlib.ExitStack().enter_context(
            unittest.mock.patch.object(handler, "emit", new_emit)
        )


@pytest.fixture
def dtype_fixture():
    """Pytest fixture that sets the default dtype to double for the test duration."""
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    yield dtype
    torch.set_default_dtype(dtype)


@contextlib.contextmanager
def set_global_var(module, var_name, value):
    """Context manager to temporarily set a module's global variable."""
    old_value = getattr(module, var_name)
    setattr(module, var_name, value)
    try:
        yield
    finally:
        setattr(module, var_name, old_value)


def make_tc(td):
    """Create a tensorclass type from a tensordict instance.

    Creates a new tensorclass with fields matching the keys of the input tensordict.
    """

    class MyClass:
        pass

    MyClass.__annotations__ = {}
    for key in td.keys():
        MyClass.__annotations__[key] = torch.Tensor
    return tensorclass(MyClass)
