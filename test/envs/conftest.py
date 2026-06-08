# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import gc
import threading
import time
import warnings

import pytest

pytestmark = [
    pytest.mark.filterwarnings("error"),
    pytest.mark.filterwarnings(
        "ignore:Got multiple backends for torchrl.data.replay_buffers.storages"
    ),
    pytest.mark.filterwarnings("ignore:unclosed file"),
]


@pytest.fixture(autouse=False)  # Turn to True to enable
def check_no_lingering_multiprocessing_resources(request):
    """Fixture that checks for leftover multiprocessing resources after each test.

    This helps detect test pollution where one test leaves behind resource_sharer
    threads, zombie processes, or other multiprocessing state that can cause
    deadlocks in subsequent tests (especially with fork start method on Linux).

    See: https://bugs.python.org/issue30289
    """
    threads_before = {t.name for t in threading.enumerate()}
    resource_sharer_before = sum(
        1
        for t in threading.enumerate()
        if "_serve" in t.name or "resource_sharer" in t.name.lower()
    )

    yield

    gc.collect()
    time.sleep(0.05)

    resource_sharer_after = sum(
        1
        for t in threading.enumerate()
        if "_serve" in t.name or "resource_sharer" in t.name.lower()
    )

    if resource_sharer_after > resource_sharer_before:
        new_threads = {t.name for t in threading.enumerate()} - threads_before
        resource_sharer_threads = [
            t.name
            for t in threading.enumerate()
            if "_serve" in t.name or "resource_sharer" in t.name.lower()
        ]

        warnings.warn(
            f"Test {request.node.name} left behind {resource_sharer_after - resource_sharer_before} "
            f"resource_sharer thread(s): {resource_sharer_threads}. "
            f"New threads: {new_threads}. "
            "This can cause deadlocks in subsequent tests with fork start method.",
            UserWarning,
            stacklevel=1,
        )
