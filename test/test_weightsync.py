# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import time

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import multiprocessing as mp

from torchrl.weight_update import (
    MultiProcessWeightSyncScheme,
    NoWeightSyncScheme,
    SharedMemWeightSyncScheme,
)

_has_ray = importlib.util.find_spec("ray") is not None


def _sharedmem_worker(
    scheme, worker_idx, result_queue, initial_bias, updated_bias, event
):
    """Worker function for SharedMemWeightSyncScheme test."""
    # Create local model
    model = nn.Linear(4, 2, bias=True)

    # Phase 1: init_on_receiver (no communication)
    scheme.init_on_receiver(model_id="policy", model=model, worker_idx=worker_idx)

    # Phase 2: connect - receive initial weights via queue
    scheme.connect(worker_idx=worker_idx)

    # Check initial weights were applied (model should have shared memory params now)
    bias_val = model.bias.data[0].item()
    result_queue.put(("initial", abs(bias_val - initial_bias) < 0.01))

    # Signal sender that we're ready
    event.set()

    # Wait for weight update (shared memory - should see automatically via model params)
    time.sleep(0.5)

    # Check updated weights - access via model's parameters
    bias_val = model.bias.data[0].item()
    result_queue.put(("updated", abs(bias_val - updated_bias) < 0.01))


class TestSharedMemWeightSyncScheme:
    """Test SharedMemWeightSyncScheme end-to-end flow."""

    def test_sharedmem_flow(self):
        """Test init -> connect -> send flow for SharedMemWeightSyncScheme."""
        mp_ctx = mp.get_context("spawn")

        # Create source model with known weights
        model = nn.Linear(4, 2, bias=True)
        initial_bias = 1.5
        model.bias.data.fill_(initial_bias)

        # Create scheme
        scheme = SharedMemWeightSyncScheme(strategy="tensordict")

        # Phase 1: init_on_sender
        weights = TensorDict.from_module(model)
        scheme.init_on_sender(
            model_id="policy",
            weights=weights,
            devices=[torch.device("cpu")],
            num_workers=1,
        )

        # Create synchronization event
        event = mp_ctx.Event()

        # Start worker - pass the same scheme object so queues are shared
        result_queue = mp_ctx.Queue()
        updated_bias = 3.0
        worker = mp_ctx.Process(
            target=_sharedmem_worker,
            args=(scheme, 0, result_queue, initial_bias, updated_bias, event),
        )
        worker.start()

        # Phase 2: connect - send initial weights to queue
        scheme.connect()

        # Wait for worker to receive initial weights
        event.wait(timeout=10)

        # Update weights via shared memory - update the shared buffer directly
        shared_weights = scheme.shared_transport.unique_weights[0]
        shared_weights["bias"].data.fill_(updated_bias)

        # Check results
        worker.join(timeout=10)

        results = {}
        while not result_queue.empty():
            key, val = result_queue.get()
            results[key] = val

        assert results.get("initial", False), "Worker did not receive initial weights"
        assert results.get("updated", False), "Worker did not see updated weights"


def _mp_worker(scheme, worker_idx, result_queue, initial_bias, updated_bias, event):
    """Worker function for MultiProcessWeightSyncScheme test."""
    try:
        # Create local model
        model = nn.Linear(4, 2, bias=True)

        # Phase 1: init_on_receiver
        scheme.init_on_receiver(model_id="policy", model=model, worker_idx=worker_idx)

        # Phase 2: connect - receive initial weights
        scheme.connect(worker_idx=worker_idx)

        # Check initial weights
        bias_val = model.bias.data[0].item()
        result_queue.put(("initial", abs(bias_val - initial_bias) < 0.01))

        # Signal sender that we received initial weights
        event.set()

        # Receive weight update (must explicitly receive for MP scheme)
        scheme.receive()

        # Check updated weights
        bias_val = model.bias.data[0].item()
        result_queue.put(("updated", abs(bias_val - updated_bias) < 0.01))
    except Exception as e:
        result_queue.put(("error", str(e)))


class TestMultiProcessWeightSyncScheme:
    """Test MultiProcessWeightSyncScheme end-to-end flow."""

    def test_mp_flow(self):
        """Test init -> connect -> send flow for MultiProcessWeightSyncScheme."""
        mp_ctx = mp.get_context("spawn")

        # Create source model
        model = nn.Linear(4, 2, bias=True)
        initial_bias = 2.0
        model.bias.data.fill_(initial_bias)

        # Create scheme
        scheme = MultiProcessWeightSyncScheme(strategy="tensordict")

        # Phase 1: init_on_sender
        weights = TensorDict.from_module(model)
        scheme.init_on_sender(
            model_id="policy",
            weights=weights,
            devices=[torch.device("cpu")],
            num_workers=1,
        )

        # Create synchronization event
        event = mp_ctx.Event()

        # Start worker
        result_queue = mp_ctx.Queue()
        updated_bias = 4.0
        worker = mp_ctx.Process(
            target=_mp_worker,
            args=(scheme, 0, result_queue, initial_bias, updated_bias, event),
        )
        worker.start()

        # Phase 2: connect - send initial weights
        scheme.connect()

        # Wait for worker to receive initial weights
        event.wait(timeout=10)

        # Send updated weights
        model.bias.data.fill_(updated_bias)
        new_weights = TensorDict.from_module(model)
        scheme.send(new_weights)

        # Check results
        worker.join(timeout=10)

        results = {}
        while not result_queue.empty():
            key, val = result_queue.get()
            results[key] = val

        # Check for errors first
        if "error" in results:
            raise AssertionError(f"Worker raised exception: {results['error']}")

        assert results.get("initial", False), "Worker did not receive initial weights"
        assert results.get("updated", False), "Worker did not receive updated weights"


class TestNoWeightSyncScheme:
    """Test NoWeightSyncScheme (no-op)."""

    def test_noupdate_flow(self):
        """Test that NoWeightSyncScheme does nothing."""
        scheme = NoWeightSyncScheme()

        # Init should work
        scheme.init_on_sender(model_id="policy")

        # Connect should work (no-op)
        scheme.connect()

        # Send should work (no-op)
        scheme.send()

        # Receive should return False
        result = scheme.receive()
        assert result is False


# Skip distributed/RPC/Ray tests if dependencies not available
@pytest.mark.skipif(
    not torch.distributed.is_available(),
    reason="torch.distributed not available",
)
class TestDistributedWeightSyncScheme:
    """Test DistributedWeightSyncScheme (requires distributed setup)."""

    @pytest.mark.skip(
        reason="Requires full distributed setup - tested in test_distributed.py"
    )
    def test_distributed_flow(self):
        """Placeholder - distributed tests require special setup."""


@pytest.mark.skipif(
    not torch.distributed.is_available() or not hasattr(torch.distributed, "rpc"),
    reason="torch.distributed.rpc not available",
)
class TestRPCWeightSyncScheme:
    """Test RPCWeightSyncScheme (requires RPC setup)."""

    @pytest.mark.skip(reason="Requires full RPC setup - tested in test_distributed.py")
    def test_rpc_flow(self):
        """Placeholder - RPC tests require special setup."""


@pytest.mark.skipif(not _has_ray, reason="Ray not available")
class TestRayWeightSyncScheme:
    """Test RayWeightSyncScheme (requires Ray)."""

    @pytest.mark.skip(reason="Requires Ray actors - tested in test_distributed.py")
    def test_ray_flow(self):
        """Placeholder - Ray collector tests require remote actors."""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
