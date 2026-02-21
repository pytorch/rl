# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import concurrent.futures
import threading
import time

import pytest
import torch
import torch.nn as nn

from tensordict import lazy_stack, TensorDict
from tensordict.base import TensorDictBase
from tensordict.nn import TensorDictModule

from torchrl.modules.inference_server import (
    InferenceClient,
    InferenceServer,
    InferenceTransport,
    MPTransport,
    RayTransport,
    ThreadingTransport,
)
from torchrl.modules.inference_server._monarch import MonarchTransport

_has_ray = True
try:
    import ray
except ImportError:
    _has_ray = False

_has_monarch = True
try:
    import monarch  # noqa: F401
except ImportError:
    _has_monarch = False


# =============================================================================
# Helpers
# =============================================================================


class _MockTransport(InferenceTransport):
    """Minimal in-process transport for testing the core server logic."""

    def __init__(self):
        self._queue: list[TensorDictBase] = []
        self._futures: list[concurrent.futures.Future] = []
        self._lock = threading.Lock()
        self._event = threading.Event()

    def submit(self, td):
        fut = concurrent.futures.Future()
        with self._lock:
            self._queue.append(td)
            self._futures.append(fut)
        self._event.set()
        return fut

    def drain(self, max_items):
        with self._lock:
            n = min(len(self._queue), max_items)
            items = self._queue[:n]
            futs = self._futures[:n]
            del self._queue[:n]
            del self._futures[:n]
        return items, futs

    def wait_for_work(self, timeout):
        self._event.wait(timeout=timeout)
        self._event.clear()

    def resolve(self, callback, result):
        callback.set_result(result)

    def resolve_exception(self, callback, exc):
        callback.set_exception(exc)


def _make_policy():
    """A simple TensorDictModule for testing."""
    return TensorDictModule(
        nn.Linear(4, 2),
        in_keys=["observation"],
        out_keys=["action"],
    )


# =============================================================================
# Tests: core abstractions (Commit 1)
# =============================================================================


class TestInferenceTransportABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            InferenceTransport()

    def test_client_returns_inference_client(self):
        transport = _MockTransport()
        client = transport.client()
        assert isinstance(client, InferenceClient)


class TestInferenceServerCore:
    def test_start_and_shutdown(self):
        transport = _MockTransport()
        policy = _make_policy()
        server = InferenceServer(policy, transport, max_batch_size=4)
        server.start()
        assert server.is_alive
        server.shutdown()
        assert not server.is_alive

    def test_context_manager(self):
        transport = _MockTransport()
        policy = _make_policy()
        with InferenceServer(policy, transport, max_batch_size=4) as server:
            assert server.is_alive
        assert not server.is_alive

    def test_double_start_raises(self):
        transport = _MockTransport()
        policy = _make_policy()
        server = InferenceServer(policy, transport, max_batch_size=4)
        server.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                server.start()
        finally:
            server.shutdown()

    def test_single_request(self):
        transport = _MockTransport()
        policy = _make_policy()
        with InferenceServer(policy, transport, max_batch_size=4):
            td = TensorDict({"observation": torch.randn(4)})
            fut = transport.submit(td)
            result = fut.result(timeout=5.0)
            assert "action" in result.keys()
            assert result["action"].shape == (2,)

    def test_batch_of_requests(self):
        transport = _MockTransport()
        policy = _make_policy()
        n = 8
        with InferenceServer(policy, transport, max_batch_size=16):
            futures = [
                transport.submit(TensorDict({"observation": torch.randn(4)}))
                for _ in range(n)
            ]
            results = [f.result(timeout=5.0) for f in futures]
            assert len(results) == n
            for r in results:
                assert "action" in r.keys()
                assert r["action"].shape == (2,)

    def test_collate_fn_is_called(self):
        calls = []

        def tracking_collate(items):
            calls.append(len(items))
            return lazy_stack(items)

        transport = _MockTransport()
        policy = _make_policy()
        with InferenceServer(
            policy, transport, max_batch_size=16, collate_fn=tracking_collate
        ):
            futures = [
                transport.submit(TensorDict({"observation": torch.randn(4)}))
                for _ in range(4)
            ]
            for f in futures:
                f.result(timeout=5.0)

        assert len(calls) >= 1
        assert sum(calls) == 4  # all 4 items processed

    def test_max_batch_size_respected(self):
        """The collate_fn should never receive more than max_batch_size items."""
        max_bs = 4
        seen_sizes = []

        def tracking_collate(items):
            seen_sizes.append(len(items))
            return lazy_stack(items)

        transport = _MockTransport()
        policy = _make_policy()
        # Submit many items then start the server
        n = 20
        futures = [
            transport.submit(TensorDict({"observation": torch.randn(4)}))
            for _ in range(n)
        ]
        with InferenceServer(
            policy,
            transport,
            max_batch_size=max_bs,
            collate_fn=tracking_collate,
        ):
            for f in futures:
                f.result(timeout=5.0)

        for s in seen_sizes:
            assert s <= max_bs


class TestInferenceClient:
    def test_sync_call(self):
        transport = _MockTransport()
        policy = _make_policy()
        with InferenceServer(policy, transport, max_batch_size=4):
            client = InferenceClient(transport)
            td = TensorDict({"observation": torch.randn(4)})
            result = client(td)
            assert "action" in result.keys()

    def test_submit_returns_future(self):
        transport = _MockTransport()
        policy = _make_policy()
        with InferenceServer(policy, transport, max_batch_size=4):
            client = InferenceClient(transport)
            td = TensorDict({"observation": torch.randn(4)})
            fut = client.submit(td)
            assert isinstance(fut, concurrent.futures.Future)
            result = fut.result(timeout=5.0)
            assert "action" in result.keys()


# =============================================================================
# Tests: ThreadingTransport (Commit 2)
# =============================================================================


class TestThreadingTransport:
    def test_single_request(self):
        transport = ThreadingTransport()
        policy = _make_policy()
        with InferenceServer(policy, transport, max_batch_size=4):
            client = transport.client()
            td = TensorDict({"observation": torch.randn(4)})
            result = client(td)
            assert "action" in result.keys()
            assert result["action"].shape == (2,)

    def test_concurrent_actors(self):
        """Multiple threads submit concurrently; all get correct results."""
        transport = ThreadingTransport()
        policy = _make_policy()
        n_actors = 8
        n_requests = 50

        results_per_actor: list[list[TensorDictBase]] = [[] for _ in range(n_actors)]

        def actor_fn(actor_id, client):
            for _ in range(n_requests):
                td = TensorDict({"observation": torch.randn(4)})
                result = client(td)
                results_per_actor[actor_id].append(result)

        with InferenceServer(policy, transport, max_batch_size=16):
            client = transport.client()
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_actors) as pool:
                futs = [pool.submit(actor_fn, i, client) for i in range(n_actors)]
                concurrent.futures.wait(futs)
                # re-raise any exceptions
                for f in futs:
                    f.result()

        for actor_results in results_per_actor:
            assert len(actor_results) == n_requests
            for r in actor_results:
                assert "action" in r.keys()
                assert r["action"].shape == (2,)

    def test_timeout_fires_partial_batch(self):
        """A single request should be processed even below max_batch_size."""
        transport = ThreadingTransport()
        policy = _make_policy()
        # max_batch_size is large, but timeout should still fire
        with InferenceServer(policy, transport, max_batch_size=1024, timeout=0.05):
            client = transport.client()
            td = TensorDict({"observation": torch.randn(4)})
            result = client(td)
            assert "action" in result.keys()

    def test_max_batch_size_threading(self):
        """Verify max_batch_size is respected with real threading transport."""
        max_bs = 4
        seen_sizes = []

        def tracking_collate(items):
            seen_sizes.append(len(items))
            return lazy_stack(items)

        transport = ThreadingTransport()
        policy = _make_policy()
        n = 20

        # Submit many before starting so they queue up
        futures = [
            transport.submit(TensorDict({"observation": torch.randn(4)}))
            for _ in range(n)
        ]
        with InferenceServer(
            policy,
            transport,
            max_batch_size=max_bs,
            collate_fn=tracking_collate,
        ):
            for f in futures:
                f.result(timeout=5.0)

        for s in seen_sizes:
            assert s <= max_bs

    def test_model_exception_propagates(self):
        """If the model raises, the exception propagates to the caller."""

        def bad_model(td):
            raise ValueError("model error")

        transport = ThreadingTransport()
        with InferenceServer(bad_model, transport, max_batch_size=4):
            client = transport.client()
            td = TensorDict({"observation": torch.randn(4)})
            with pytest.raises(ValueError, match="model error"):
                client(td)


# =============================================================================
# Tests: MPTransport (Commit 3)
# =============================================================================


def _mp_actor_fn(client, obs_size, act_size, n_requests, result_queue):
    """Actor function that runs in a child process."""
    for _ in range(n_requests):
        td = TensorDict({"observation": torch.randn(obs_size)})
        result = client(td)
        assert "action" in result.keys()
        assert result["action"].shape == (act_size,)
    result_queue.put(True)


class TestMPTransport:
    @pytest.mark.slow
    def test_single_request_in_process(self):
        """MPTransport client works from the parent process."""
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        transport = MPTransport(ctx=ctx)
        client = transport.client()
        policy = _make_policy()
        with InferenceServer(policy, transport, max_batch_size=4):
            td = TensorDict({"observation": torch.randn(4)})
            result = client(td)
            assert "action" in result.keys()
            assert result["action"].shape == (2,)

    @pytest.mark.slow
    def test_cross_process_actors(self):
        """Actors in separate processes get correct results."""
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        transport = MPTransport(ctx=ctx)
        policy = _make_policy()
        n_actors = 2
        n_requests = 10

        result_queue = ctx.Queue()
        # Create clients before spawning (queues inherited)
        clients = [transport.client() for _ in range(n_actors)]

        with InferenceServer(policy, transport, max_batch_size=8):
            procs = []
            for i in range(n_actors):
                p = ctx.Process(
                    target=_mp_actor_fn,
                    args=(clients[i], 4, 2, n_requests, result_queue),
                )
                p.start()
                procs.append(p)

            for p in procs:
                p.join(timeout=30.0)
                assert p.exitcode == 0

        # All actors reported success
        for _ in range(n_actors):
            assert result_queue.get(timeout=1.0) is True

    @pytest.mark.slow
    def test_mp_exception_propagates(self):
        """Model exceptions propagate through MPTransport."""
        import multiprocessing as mp

        def bad_model(td):
            raise ValueError("mp model error")

        ctx = mp.get_context("spawn")
        transport = MPTransport(ctx=ctx)
        client = transport.client()
        with InferenceServer(bad_model, transport, max_batch_size=4):
            td = TensorDict({"observation": torch.randn(4)})
            with pytest.raises(ValueError, match="mp model error"):
                client(td)


# =============================================================================
# Tests: RayTransport (Commit 4)
# =============================================================================


@pytest.mark.skipif(not _has_ray, reason="ray not installed")
class TestRayTransport:
    @classmethod
    def setup_class(cls):
        if not ray.is_initialized():
            ray.init(num_cpus=4, ignore_reinit_error=True)

    def test_single_request(self):
        transport = RayTransport()
        client = transport.client()
        policy = _make_policy()
        with InferenceServer(policy, transport, max_batch_size=4):
            td = TensorDict({"observation": torch.randn(4)})
            result = client(td)
            assert "action" in result.keys()
            assert result["action"].shape == (2,)

    def test_concurrent_clients(self):
        """Multiple clients submit concurrently from threads (simulating Ray actors)."""
        transport = RayTransport()
        policy = _make_policy()
        n_clients = 4
        n_requests = 20

        clients = [transport.client() for _ in range(n_clients)]
        results_per_client: list[list[TensorDictBase]] = [[] for _ in range(n_clients)]

        def client_fn(client_idx):
            for _ in range(n_requests):
                td = TensorDict({"observation": torch.randn(4)})
                result = clients[client_idx](td)
                results_per_client[client_idx].append(result)

        with InferenceServer(policy, transport, max_batch_size=8):
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_clients) as pool:
                futs = [pool.submit(client_fn, i) for i in range(n_clients)]
                concurrent.futures.wait(futs)
                for f in futs:
                    f.result()

        for client_results in results_per_client:
            assert len(client_results) == n_requests
            for r in client_results:
                assert "action" in r.keys()
                assert r["action"].shape == (2,)

    def test_ray_remote_actor(self):
        """A Ray remote actor can use the client to get inference results."""
        transport = RayTransport()
        client = transport.client()
        policy = _make_policy()

        @ray.remote
        def remote_actor_fn(client, n_requests):
            results = []
            for _ in range(n_requests):
                td = TensorDict({"observation": torch.randn(4)})
                result = client(td)
                results.append(result["action"].shape)
            return results

        with InferenceServer(policy, transport, max_batch_size=8):
            ref = remote_actor_fn.remote(client, 5)
            shapes = ray.get(ref, timeout=30.0)
            assert len(shapes) == 5
            for s in shapes:
                assert s == (2,)

    def test_ray_exception_propagates(self):
        def bad_model(td):
            raise ValueError("ray model error")

        transport = RayTransport()
        client = transport.client()
        with InferenceServer(bad_model, transport, max_batch_size=4):
            td = TensorDict({"observation": torch.randn(4)})
            with pytest.raises(ValueError, match="ray model error"):
                client(td)


# =============================================================================
# Tests: MonarchTransport (Commit 5)
# =============================================================================


@pytest.mark.skipif(not _has_monarch, reason="monarch not installed")
class TestMonarchTransport:
    def test_single_request(self):
        transport = MonarchTransport()
        client = transport.client()
        policy = _make_policy()
        with InferenceServer(policy, transport, max_batch_size=4):
            td = TensorDict({"observation": torch.randn(4)})
            result = client(td)
            assert "action" in result.keys()
            assert result["action"].shape == (2,)

    def test_concurrent_clients(self):
        """Multiple Monarch clients submit concurrently."""
        transport = MonarchTransport()
        policy = _make_policy()
        n_clients = 4
        n_requests = 20

        clients = [transport.client() for _ in range(n_clients)]
        results_per_client: list[list[TensorDictBase]] = [[] for _ in range(n_clients)]

        def client_fn(client_idx):
            for _ in range(n_requests):
                td = TensorDict({"observation": torch.randn(4)})
                result = clients[client_idx](td)
                results_per_client[client_idx].append(result)

        with InferenceServer(policy, transport, max_batch_size=8):
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_clients) as pool:
                futs = [pool.submit(client_fn, i) for i in range(n_clients)]
                concurrent.futures.wait(futs)
                for f in futs:
                    f.result()

        for client_results in results_per_client:
            assert len(client_results) == n_requests
            for r in client_results:
                assert "action" in r.keys()
                assert r["action"].shape == (2,)


class TestMonarchTransportImport:
    def test_import_without_monarch(self):
        """MonarchTransport class can be imported even without monarch."""
        # This test verifies the lazy import pattern works.
        # The class itself is importable; only instantiation requires monarch.
        assert MonarchTransport is not None

    @pytest.mark.skipif(_has_monarch, reason="test requires monarch NOT installed")
    def test_instantiation_without_monarch_raises(self):
        with pytest.raises(ImportError, match="Monarch is required"):
            MonarchTransport()


# =============================================================================
# Tests: WeightSyncScheme integration (Commit 6)
# =============================================================================


class _SimpleWeightSync:
    """Minimal mock that mimics the WeightSyncScheme receiver interface.

    Stores a queue of weight TensorDicts. ``receive(timeout=...)`` pops
    the next one and applies it to the model via
    ``TensorDict.from_module / to_module``.
    """

    def __init__(self):
        self._queue: list[TensorDictBase] = []
        self._model = None
        self.initialized_on_receiver = False
        self.synchronized_on_receiver = False

    def init_on_receiver(self, *, model_id, model=None, worker_idx=0, **kwargs):
        self._model = model
        self.initialized_on_receiver = True

    def connect(self, *, worker_idx=0):
        self.synchronized_on_receiver = True

    def receive(self, timeout=None):
        if self._queue:
            weights = self._queue.pop(0)
            weights.to_module(self._model)
            return weights
        return None

    def push(self, weights: TensorDictBase):
        """Test helper: enqueue weights for the server to pick up."""
        self._queue.append(weights)


class TestWeightSyncIntegration:
    def test_weight_sync_init_called(self):
        """Server calls init_on_receiver and connect at startup."""
        transport = ThreadingTransport()
        policy = _make_policy()
        ws = _SimpleWeightSync()

        with InferenceServer(policy, transport, weight_sync=ws):
            # Give the worker thread a moment to start
            time.sleep(0.1)
            assert ws.initialized_on_receiver
            assert ws.synchronized_on_receiver

    def test_weight_update_applied(self):
        """Weights pushed via weight_sync are applied to the model."""
        transport = ThreadingTransport()
        policy = _make_policy()
        ws = _SimpleWeightSync()

        with InferenceServer(
            policy, transport, max_batch_size=4, weight_sync=ws
        ) as server:
            client = transport.client()

            # Get initial prediction
            td = TensorDict({"observation": torch.ones(4)})
            client(td)

            # Mutate the model weights externally and push via weight_sync
            new_weights = TensorDict.from_module(policy)
            for key in new_weights.keys(True, True):
                new_weights[key] = torch.zeros_like(new_weights[key])
            ws.push(new_weights)

            # Give the server loop a chance to apply the update
            time.sleep(0.2)

            # Now inference should reflect zero weights
            result_after = client(td)
            # With zero weights the linear output should be zero (bias=0 too)
            assert torch.allclose(result_after["action"], torch.zeros(2), atol=1e-6)

    def test_inference_continues_after_weight_update(self):
        """The server keeps serving after a weight update."""
        transport = ThreadingTransport()
        policy = _make_policy()
        ws = _SimpleWeightSync()

        with InferenceServer(policy, transport, max_batch_size=4, weight_sync=ws):
            client = transport.client()

            # Initial requests
            for _ in range(5):
                td = TensorDict({"observation": torch.randn(4)})
                result = client(td)
                assert "action" in result.keys()

            # Push weight update
            new_weights = TensorDict.from_module(policy)
            ws.push(new_weights)

            time.sleep(0.1)

            # Continue making requests
            for _ in range(5):
                td = TensorDict({"observation": torch.randn(4)})
                result = client(td)
                assert "action" in result.keys()
                assert result["action"].shape == (2,)

    def test_no_weight_sync(self):
        """Server works fine when weight_sync is None."""
        transport = ThreadingTransport()
        policy = _make_policy()
        with InferenceServer(policy, transport, max_batch_size=4):
            client = transport.client()
            td = TensorDict({"observation": torch.randn(4)})
            result = client(td)
            assert "action" in result.keys()
