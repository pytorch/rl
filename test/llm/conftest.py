# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shared fixtures for LLM tests.

This module provides common fixtures for test cleanup, especially for:
- Ray shutdown between test sessions
- vLLM engine cleanup
- GPU memory cleanup
"""

import gc

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def cleanup_ray_after_session():
    """Ensure Ray is properly shutdown after all tests complete."""
    yield
    try:
        import ray

        if ray.is_initialized():
            ray.shutdown()
    except ImportError:
        pass


@pytest.fixture(scope="function")
def ray_session():
    """Fixture that provides a Ray session with guaranteed cleanup.

    Usage:
        def test_something(ray_session):
            # Ray is initialized
            ...
        # Ray is shutdown after test
    """
    import ray

    was_initialized = ray.is_initialized()
    if not was_initialized:
        ray.init()
    yield
    if not was_initialized:
        ray.shutdown()


@pytest.fixture(scope="session", autouse=True)
def cleanup_gpu_memory_after_session():
    """Cleanup GPU memory after all tests complete."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@pytest.fixture(scope="function")
def cleanup_gpu_memory():
    """Cleanup GPU memory after each test function."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pytest_collection_modifyitems(config, items):
    """Add timeout marker to all tests if not already set."""
    for item in items:
        # Check if test already has a timeout marker
        existing_timeout = None
        for marker in item.iter_markers(name="timeout"):
            existing_timeout = marker
            break

        # If no timeout set, add default 5 minute timeout
        if existing_timeout is None:
            item.add_marker(pytest.mark.timeout(300))
