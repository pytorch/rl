# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import concurrent.futures
import importlib.util
import multiprocessing as mp
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
    InferenceDeviceConfig,
    InferenceServer,
    InferenceServerConfig,
    InferenceTransport,
    MPTransport,
    PolicyClientModule,
    ProcessInferenceServer,
    RayTransport,
    RemotePolicy,
    SlotTransport,
    ThreadingTransport,
)
from torchrl.modules.inference_server._monarch import MonarchTransport

_has_ray = importlib.util.find_spec("ray") is not None
_has_monarch = importlib.util.find_spec("monarch") is not None
_ray = None


def _ray_lib():
    global _ray
    if _ray is None:
        import ray

        _ray = ray
    return _ray


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

    def test_collate_error_resolves_futures_and_server_survives(self):
        """A collate failure must reject the affected futures, not kill the loop."""

        def fragile_collate(items):
            if any("poison" in item.keys() for item in items):
                raise ValueError("cannot collate poisoned batch")
            return lazy_stack(items)

        transport = _MockTransport()
        policy = _make_policy()
        with InferenceServer(
            policy, transport, max_batch_size=1, collate_fn=fragile_collate
        ) as server:
            bad = transport.submit(
                TensorDict({"observation": torch.randn(4), "poison": torch.ones(1)})
            )
            with pytest.raises(ValueError, match="poisoned"):
                bad.result(timeout=5.0)
            # The serve loop must still be alive and processing requests
            good = transport.submit(TensorDict({"observation": torch.randn(4)}))
            result = good.result(timeout=5.0)
            assert "action" in result.keys()
            assert server.is_alive

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

    def test_policy_and_output_device_handoff(self):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        transport = ThreadingTransport()
        policy = _make_policy()
        with InferenceServer(
            policy,
            transport,
            max_batch_size=4,
            policy_device=device,
            output_device="cpu",
        ):
            client = transport.client()
            td = TensorDict({"observation": torch.randn(4)}, device="cpu")
            result = client(td)
        assert result["action"].device.type == "cpu"
        assert next(policy.parameters()).device.type == torch.device(device).type

    def test_stats_accounting(self):
        transport = ThreadingTransport()
        policy = _make_policy()
        n = 8
        with InferenceServer(
            policy,
            transport,
            max_batch_size=n,
            min_batch_size=4,
            timeout=0.5,
        ) as server:
            client = transport.client()
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
                futs = [
                    pool.submit(
                        lambda: client(TensorDict({"observation": torch.randn(4)}))
                    )
                    for _ in range(n)
                ]
                for fut in futs:
                    fut.result(timeout=5.0)
            stats = server.stats()
        assert stats["requests"] == n
        assert stats["batches"] >= 1
        assert stats["avg_batch_size"] > 0
        assert stats["p95_forward_ms"] >= 0
        assert stats["policy_version"] == 0

    def test_structured_config(self):
        transport = ThreadingTransport()
        policy = _make_policy()
        server_config = InferenceServerConfig(max_batch_size=2, timeout=0.001)
        device_config = InferenceDeviceConfig(policy_device="cpu", output_device="cpu")
        with InferenceServer(
            policy,
            transport,
            server_config=server_config,
            device_config=device_config,
        ) as server:
            # The config values must actually land on the server
            assert server.max_batch_size == 2
            assert server.timeout == 0.001
            assert server.policy_device == torch.device("cpu")
            assert server.output_device == torch.device("cpu")
            client = transport.client()
            result = client(TensorDict({"observation": torch.randn(4)}))
            stats = server.stats()
        assert result["action"].device.type == "cpu"
        assert stats["requests"] == 1

    def test_server_config_exclusive_even_at_default_values(self):
        """Passing an explicit kwarg equal to the default still raises."""
        transport = ThreadingTransport()
        policy = _make_policy()
        with pytest.raises(ValueError, match="mutually exclusive"):
            InferenceServer(
                policy,
                transport,
                max_batch_size=64,
                server_config=InferenceServerConfig(max_batch_size=8),
            )
        with pytest.raises(ValueError, match="mutually exclusive"):
            InferenceServer(
                policy,
                transport,
                device="cpu",
                device_config=InferenceDeviceConfig(policy_device="cpu"),
            )

    def test_device_config_env_device_fallback_and_storing_device_rejected(self):
        config = InferenceDeviceConfig(env_device="cpu")
        assert config.server_output_device() == torch.device("cpu")
        config = InferenceDeviceConfig(env_device="cpu", output_device="meta")
        assert config.server_output_device() == torch.device("meta")
        transport = ThreadingTransport()
        policy = _make_policy()
        with pytest.raises(ValueError, match="storing_device is a collector-level"):
            InferenceServer(
                policy,
                transport,
                device_config=InferenceDeviceConfig(storing_device="cpu"),
            )

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_device_config_cuda_roundtrip(self):
        """device_config must actually drive the policy and output moves."""
        transport = ThreadingTransport()
        policy = _make_policy()
        seen_devices = []
        inner_forward = policy.module.forward

        def recording_forward(x):
            seen_devices.append(x.device)
            return inner_forward(x)

        policy.module.forward = recording_forward
        with InferenceServer(
            policy,
            transport,
            device_config=InferenceDeviceConfig(
                policy_device="cuda:0", output_device="cpu"
            ),
        ):
            client = transport.client()
            result = client(TensorDict({"observation": torch.randn(4)}, device="cpu"))
        assert result["action"].device.type == "cpu"
        assert all(device.type == "cuda" for device in seen_devices)
        assert next(policy.parameters()).device.type == "cuda"

    def test_policy_version_is_returned(self):
        transport = ThreadingTransport()
        policy = _make_policy()
        with InferenceServer(
            policy,
            transport,
            policy_version=12,
            policy_version_key=("meta", "policy_version"),
        ):
            client = transport.client()
            result = client(TensorDict({"observation": torch.randn(4)}))
        assert result["meta", "policy_version"].item() == 12

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_policy_cpu_output(self):
        transport = ThreadingTransport()
        policy = _make_policy()
        with InferenceServer(
            policy,
            transport,
            max_batch_size=4,
            policy_device="cuda:0",
            output_device="cpu",
        ):
            client = transport.client()
            result = client(TensorDict({"observation": torch.randn(4)}, device="cpu"))
        assert result["action"].device.type == "cpu"
        assert next(policy.parameters()).device.type == "cuda"


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


class TestPolicyClientModule:
    def test_remote_policy_alias(self):
        assert RemotePolicy is PolicyClientModule

    def test_forward_as_tensordict_module(self):
        transport = ThreadingTransport()
        policy = _make_policy()
        with InferenceServer(policy, transport, max_batch_size=4):
            remote_policy = PolicyClientModule(
                transport,
                in_keys=["observation"],
                out_keys=["action"],
            )
            td = TensorDict({"observation": torch.randn(4)})
            result = remote_policy(td)
        assert result["action"].shape == (2,)
        assert remote_policy.in_keys == ["observation"]
        assert remote_policy.out_keys == ["action"]

    def test_submit(self):
        transport = ThreadingTransport()
        policy = _make_policy()
        with InferenceServer(policy, transport, max_batch_size=4):
            remote_policy = PolicyClientModule(transport)
            future = remote_policy.submit(TensorDict({"observation": torch.randn(4)}))
            result = future.result(timeout=5.0)
        assert "action" in result.keys()

    def test_nested_keys(self):
        """in_keys/out_keys accept nested keys end to end."""
        transport = ThreadingTransport()
        policy = TensorDictModule(
            nn.Linear(4, 2),
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "action")],
        )
        with InferenceServer(policy, transport, max_batch_size=4):
            remote_policy = PolicyClientModule(
                transport,
                in_keys=[("agents", "observation")],
                out_keys=[("agents", "action")],
            )
            td = TensorDict({("agents", "observation"): torch.randn(4)})
            result = remote_policy(td)
        assert result["agents", "action"].shape == (2,)
        assert remote_policy.in_keys == [("agents", "observation")]
        assert remote_policy.out_keys == [("agents", "action")]

    def test_plain_callable_client_defers_errors(self):
        """A plain-callable client defers exceptions to result()."""

        def failing_client(td):
            raise ValueError("local policy failure")

        remote_policy = PolicyClientModule(failing_client)
        future = remote_policy.submit(TensorDict({}))
        assert future.done()
        with pytest.raises(ValueError, match="local policy failure"):
            future.result()

    def test_bounded_staleness_raises(self):
        transport = ThreadingTransport()
        policy = _make_policy()
        with InferenceServer(policy, transport, policy_version=1):
            remote_policy = PolicyClientModule(
                transport,
                target_policy_version=3,
                max_policy_lag=1,
            )
            with pytest.raises(RuntimeError, match="too stale"):
                remote_policy(TensorDict({"observation": torch.randn(4)}))

    def test_bounded_staleness_with_callable_target(self):
        """target_policy_version accepts a live callable source."""
        transport = ThreadingTransport()
        policy = _make_policy()
        with InferenceServer(policy, transport, policy_version=5) as server:
            remote_policy = PolicyClientModule(
                transport,
                target_policy_version=lambda: server.policy_version,
                max_policy_lag=0,
            )
            # version == target -> lag 0, passes
            remote_policy(TensorDict({"observation": torch.randn(4)}))
            server._mark_weight_update()
            server._mark_weight_update()
            # server is now at version 7; a fresh result carries 7 -> passes
            remote_policy(TensorDict({"observation": torch.randn(4)}))
            assert server.policy_version == 7

    def test_staleness_guard_warns_on_missing_version_key(self, caplog):
        """A configured guard warns (once) if results carry no version."""
        transport = ThreadingTransport()
        policy = _make_policy()
        # Server does not annotate versions; client expects them
        with InferenceServer(policy, transport, policy_version_key=None):
            remote_policy = PolicyClientModule(
                transport,
                target_policy_version=3,
                max_policy_lag=1,
            )
            with caplog.at_level("WARNING", logger="torchrl"):
                remote_policy(TensorDict({"observation": torch.randn(4)}))
                remote_policy(TensorDict({"observation": torch.randn(4)}))
        warnings_seen = [
            rec for rec in caplog.records if "staleness guard is inactive" in rec.message
        ]
        assert len(warnings_seen) == 1

    def test_update_policy_weights_cascade_bumps_version(self):
        """The weight-sync cascade hook increments the policy version."""
        transport = ThreadingTransport()
        policy = _make_policy()
        server = InferenceServer(policy, transport, policy_version=0)
        assert server.policy_version == 0
        server.update_policy_weights_()
        assert server.policy_version == 1


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
        ray = _ray_lib()
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
        ray = _ray_lib()
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
            assert result_after["policy_version"].item() == 1
            assert server.stats()["weight_updates"] == 1

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


# ---------------------------------------------------------------------------
# AsyncBatchedCollector tests
# ---------------------------------------------------------------------------

from torchrl.collectors import AsyncBatchedCollector
from torchrl.testing.mocking_classes import CountingEnv


def _counting_env_factory(max_steps=5):
    """Factory that returns a CountingEnv."""
    return CountingEnv(max_steps=max_steps)


class _BatchCountingPolicy(TensorDictModule):
    """A batch-aware policy that always outputs action=1 for CountingEnv."""

    def __init__(self):
        super().__init__(
            module=nn.Module(),  # placeholder
            in_keys=["observation"],
            out_keys=["action"],
        )

    def forward(self, td: TensorDictBase) -> TensorDictBase:
        obs = td.get("observation")
        action = torch.ones_like(obs)
        return td.set("action", action)


def _make_counting_policy():
    return _BatchCountingPolicy()


class _BadProcessPolicy(nn.Module):
    def forward(self, td: TensorDictBase) -> TensorDictBase:
        raise RuntimeError("process model crash")


def _make_bad_process_policy():
    return _BadProcessPolicy()


def _make_slow_policy():
    time.sleep(30.0)
    return _make_counting_policy()


class TestProcessInferenceServer:
    def test_process_server_start_shutdown(self):
        ctx = mp.get_context("spawn")
        transport = MPTransport(ctx=ctx)
        client = transport.client()
        with ProcessInferenceServer(
            policy_factory=_make_counting_policy,
            transport=transport,
            max_batch_size=4,
            mp_context=ctx,
        ) as server:
            assert server.is_alive
            assert server.stats() == {}
            result = client(TensorDict({"observation": torch.ones(1)}))
        assert "action" in result.keys()
        assert result["action"].shape == (1,)
        assert not server.is_alive

    def test_process_server_exception_propagates(self):
        ctx = mp.get_context("spawn")
        transport = MPTransport(ctx=ctx)
        client = transport.client()
        server = ProcessInferenceServer(
            policy_factory=_make_bad_process_policy,
            transport=transport,
            max_batch_size=4,
            mp_context=ctx,
        )
        server.start()
        try:
            with pytest.raises(RuntimeError, match="process model crash"):
                client(TensorDict({"observation": torch.ones(1)}))
        finally:
            server.shutdown()

    def test_startup_timeout(self):
        ctx = mp.get_context("spawn")
        transport = MPTransport(ctx=ctx)
        transport.client()
        server = ProcessInferenceServer(
            policy_factory=_make_slow_policy,
            transport=transport,
            mp_context=ctx,
            startup_timeout=0.5,
        )
        with pytest.raises(TimeoutError, match="did not report readiness"):
            server.start()
        assert not server.is_alive


class TestAsyncBatchedCollector:
    """Tests for :class:`AsyncBatchedCollector`."""

    def test_basic_collection(self):
        """Collector yields at least frames_per_batch frames."""
        num_envs = 3
        frames_per_batch = 20
        total_frames = 60
        policy = _make_counting_policy()

        collector = AsyncBatchedCollector(
            create_env_fn=[_counting_env_factory] * num_envs,
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            max_batch_size=num_envs,
            env_backend="threading",
        )
        total_collected = 0
        for batch in collector:
            assert batch is not None
            total_collected += batch.numel()
        stats = collector.server_stats()
        collector.shutdown()
        assert total_collected >= total_frames
        assert stats["requests"] > 0

    def test_policy_factory(self):
        """policy_factory is called to create the policy."""
        num_envs = 2
        collector = AsyncBatchedCollector(
            create_env_fn=[_counting_env_factory] * num_envs,
            policy_factory=_make_counting_policy,
            frames_per_batch=10,
            total_frames=20,
            max_batch_size=num_envs,
            env_backend="threading",
        )
        total_collected = 0
        for batch in collector:
            total_collected += batch.numel()
        collector.shutdown()
        assert total_collected >= 20

    def test_policy_xor_factory(self):
        """Providing both policy and policy_factory raises."""
        policy = _make_counting_policy()
        with pytest.raises(TypeError, match="mutually exclusive"):
            AsyncBatchedCollector(
                create_env_fn=[_counting_env_factory],
                policy=policy,
                policy_factory=_make_counting_policy,
                frames_per_batch=10,
            )

    def test_neither_policy_nor_factory(self):
        """Providing neither raises."""
        with pytest.raises(TypeError, match="must be provided"):
            AsyncBatchedCollector(
                create_env_fn=[_counting_env_factory],
                frames_per_batch=10,
            )

    def test_yield_completed_trajectories(self):
        """With yield_completed_trajectories, collector yields done trajectories."""
        num_envs = 3
        max_steps = 5
        policy = _make_counting_policy()

        collector = AsyncBatchedCollector(
            create_env_fn=[lambda: CountingEnv(max_steps=max_steps)] * num_envs,
            policy=policy,
            frames_per_batch=1,
            total_frames=30,
            yield_completed_trajectories=True,
            max_batch_size=num_envs,
            env_backend="threading",
        )
        count = 0
        for batch in collector:
            assert batch is not None
            # Each trajectory should end with done=True
            count += batch.numel()
        collector.shutdown()
        assert count >= 30

    def test_shutdown_idempotent(self):
        """Calling shutdown twice should not raise."""
        policy = _make_counting_policy()
        collector = AsyncBatchedCollector(
            create_env_fn=[_counting_env_factory] * 2,
            policy=policy,
            frames_per_batch=10,
            total_frames=10,
            env_backend="threading",
        )
        # Consume one batch to start
        for _batch in collector:
            break
        collector.shutdown()
        collector.shutdown()  # should not raise

    def test_endless_collector(self):
        """total_frames=-1 creates an endless collector; verify manual break works."""
        policy = _make_counting_policy()
        collector = AsyncBatchedCollector(
            create_env_fn=[_counting_env_factory] * 2,
            policy=policy,
            frames_per_batch=10,
            total_frames=-1,
            env_backend="threading",
        )
        collected = 0
        for batch in collector:
            collected += batch.numel()
            if collected >= 50:
                break
        collector.shutdown()
        assert collected >= 50

    def test_num_envs(self):
        """The collector knows the number of environments."""
        policy = _make_counting_policy()
        collector = AsyncBatchedCollector(
            create_env_fn=[_counting_env_factory] * 2,
            policy=policy,
            frames_per_batch=10,
            total_frames=10,
        )
        assert collector._num_envs == 2
        collector.shutdown()

    def test_postproc(self):
        """Post-processing callable is applied to every batch."""
        policy = _make_counting_policy()
        called = {"count": 0}

        def postproc(td):
            called["count"] += 1
            return td

        collector = AsyncBatchedCollector(
            create_env_fn=[_counting_env_factory] * 2,
            policy=policy,
            frames_per_batch=10,
            total_frames=20,
            postproc=postproc,
            env_backend="threading",
        )
        for _ in collector:
            pass
        collector.shutdown()
        assert called["count"] >= 1

    @pytest.mark.parametrize("env_backend", ["threading", "multiprocessing"])
    def test_env_backend_smoke(self, env_backend):
        """Thread and multiprocessing env backends collect data."""
        collector = AsyncBatchedCollector(
            create_env_fn=[_counting_env_factory] * 2,
            policy=_make_counting_policy(),
            frames_per_batch=10,
            total_frames=20,
            max_batch_size=2,
            env_backend=env_backend,
        )
        total = 0
        for batch in collector:
            total += batch.numel()
        collector.shutdown()
        assert total >= 20

    def test_process_server_backend_smoke(self):
        """Dedicated process server works through AsyncBatchedCollector."""
        collector = AsyncBatchedCollector(
            create_env_fn=[_counting_env_factory] * 2,
            policy_factory=_make_counting_policy,
            frames_per_batch=10,
            total_frames=20,
            max_batch_size=2,
            env_backend="threading",
            server_backend="process",
        )
        total = 0
        for batch in collector:
            total += batch.numel()
        collector.shutdown()
        assert total >= 20

    def test_max_policy_lag_wiring(self):
        """max_policy_lag reaches the clients with a live version source."""
        collector = AsyncBatchedCollector(
            create_env_fn=[_counting_env_factory] * 2,
            policy=_make_counting_policy(),
            frames_per_batch=10,
            total_frames=20,
            max_batch_size=2,
            env_backend="threading",
            max_policy_lag=3,
        )
        total = 0
        for batch in collector:
            total += batch.numel()
        assert collector.policy_version == 0
        clients = collector._clients
        collector.shutdown()
        assert total >= 20
        assert all(c.max_policy_lag == 3 for c in clients)
        assert all(callable(c.target_policy_version) for c in clients)

    def test_max_policy_lag_requires_version_key(self):
        with pytest.raises(ValueError, match="max_policy_lag requires"):
            AsyncBatchedCollector(
                create_env_fn=[_counting_env_factory] * 2,
                policy=_make_counting_policy(),
                frames_per_batch=10,
                max_policy_lag=1,
                policy_version_key=None,
            )

    def test_policy_version_key_none_disables_annotations(self):
        collector = AsyncBatchedCollector(
            create_env_fn=[_counting_env_factory] * 2,
            policy=_make_counting_policy(),
            frames_per_batch=10,
            total_frames=10,
            max_batch_size=2,
            env_backend="threading",
            policy_version_key=None,
        )
        for batch in collector:
            assert "policy_version" not in batch.keys()
        collector.shutdown()

    def test_invalid_server_backend_raises(self):
        with pytest.raises(ValueError, match="server_backend"):
            AsyncBatchedCollector(
                create_env_fn=[_counting_env_factory] * 2,
                policy_factory=_make_counting_policy,
                frames_per_batch=10,
                server_backend="proces",
            )

    def test_server_death_raises_instead_of_hanging(self):
        """Killing the server process surfaces an error in the iterator."""
        collector = AsyncBatchedCollector(
            create_env_fn=[_counting_env_factory] * 2,
            policy_factory=_make_counting_policy,
            frames_per_batch=10,
            total_frames=-1,
            max_batch_size=2,
            env_backend="threading",
            server_backend="process",
        )
        try:
            iterator = iter(collector)
            next(iterator)
            collector._server._process.kill()
            with pytest.raises(RuntimeError, match="inference server died"):
                # A couple of batches may still drain from already-queued
                # transitions before the watchdog trips.
                for _ in range(10):
                    next(iterator)
        finally:
            collector.shutdown(timeout=0.5)

    def test_device_config_and_server_config(self):
        """Collector accepts structured device and server config objects."""
        collector = AsyncBatchedCollector(
            create_env_fn=[_counting_env_factory] * 2,
            policy=_make_counting_policy(),
            frames_per_batch=10,
            total_frames=20,
            server_config=InferenceServerConfig(max_batch_size=2),
            device_config=InferenceDeviceConfig(
                policy_device="cpu",
                output_device="cpu",
                env_device="cpu",
                storing_device="cpu",
            ),
        )
        total = 0
        for batch in collector:
            assert batch.device is None or batch.device.type == "cpu"
            total += batch.numel()
        stats = collector.server_stats()
        collector.shutdown()
        assert total >= 20
        assert stats["requests"] > 0


# =============================================================================
# Tests: SlotTransport
# =============================================================================


class TestSlotTransport:
    def test_single_request(self):
        transport = SlotTransport(num_slots=4)
        policy = _make_policy()
        with InferenceServer(policy, transport, max_batch_size=4):
            client = transport.client()
            td = TensorDict({"observation": torch.randn(4)})
            result = client(td)
            assert "action" in result.keys()
            assert result["action"].shape == (2,)

    def test_concurrent_actors(self):
        """Multiple threads submit concurrently via slot clients."""
        n_actors = 4
        n_requests = 30
        transport = SlotTransport(num_slots=n_actors)
        policy = _make_policy()

        results_per_actor: list[list[TensorDictBase]] = [[] for _ in range(n_actors)]
        clients = [transport.client() for _ in range(n_actors)]

        def actor_fn(actor_id):
            for _ in range(n_requests):
                td = TensorDict({"observation": torch.randn(4)})
                result = clients[actor_id](td)
                results_per_actor[actor_id].append(result)

        with InferenceServer(policy, transport, max_batch_size=n_actors):
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_actors) as pool:
                futs = [pool.submit(actor_fn, i) for i in range(n_actors)]
                concurrent.futures.wait(futs)
                for f in futs:
                    f.result()

        for actor_results in results_per_actor:
            assert len(actor_results) == n_requests
            for r in actor_results:
                assert "action" in r.keys()
                assert r["action"].shape == (2,)

    def test_too_many_clients_raises(self):
        """Creating more clients than slots raises RuntimeError."""
        transport = SlotTransport(num_slots=2)
        transport.client()
        transport.client()
        with pytest.raises(RuntimeError, match="slots"):
            transport.client()

    def test_submit_raises(self):
        """Direct submit() on SlotTransport is not supported."""
        transport = SlotTransport(num_slots=1)
        td = TensorDict({"observation": torch.randn(4)})
        with pytest.raises(NotImplementedError):
            transport.submit(td)

    def test_exception_propagates(self):
        """Model exceptions propagate through SlotTransport."""

        def bad_model(td):
            raise ValueError("slot model error")

        transport = SlotTransport(num_slots=1)
        with InferenceServer(bad_model, transport, max_batch_size=4):
            client = transport.client()
            td = TensorDict({"observation": torch.randn(4)})
            with pytest.raises(ValueError, match="slot model error"):
                client(td)


# =============================================================================
# Tests: min_batch_size
# =============================================================================


class TestMinBatchSize:
    def test_min_batch_size_accumulates(self):
        """With min_batch_size > 1, the server waits for enough items."""
        min_bs = 4
        seen_sizes = []

        def tracking_collate(items):
            seen_sizes.append(len(items))
            return lazy_stack(items)

        transport = ThreadingTransport()
        policy = _make_policy()
        n = 8

        with InferenceServer(
            policy,
            transport,
            max_batch_size=16,
            min_batch_size=min_bs,
            collate_fn=tracking_collate,
            timeout=1.0,
        ):
            client = transport.client()
            # Submit items from threads to give the server time to accumulate
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
                futs = [
                    pool.submit(
                        lambda: client(TensorDict({"observation": torch.randn(4)}))
                    )
                    for _ in range(n)
                ]
                for f in futs:
                    f.result(timeout=10.0)

        # At least one batch should have >= min_batch_size items
        assert any(s >= min_bs for s in seen_sizes)


# =============================================================================
# Tests: bugfix regressions
# =============================================================================


class TestShutdownPendingFutures:
    def test_shutdown_resolves_pending_futures(self):
        """Pending futures receive an exception on shutdown (no hang)."""
        transport = ThreadingTransport()
        policy = _make_policy()
        server = InferenceServer(policy, transport, max_batch_size=1024)
        server.start()
        futures = [
            transport.submit(TensorDict({"observation": torch.randn(4)}))
            for _ in range(5)
        ]
        time.sleep(0.05)
        server.shutdown(timeout=5.0)
        for f in futures:
            try:
                f.result(timeout=2.0)
            except Exception:
                pass  # exception is acceptable; hanging is not


class TestThreadingTransportNoLostSignals:
    def test_rapid_submit_no_lost_signals(self):
        """Rapid submits from many threads don't lose signals."""
        transport = ThreadingTransport()
        policy = _make_policy()
        n = 100
        with InferenceServer(policy, transport, max_batch_size=4, timeout=0.001):
            client = transport.client()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futs = [
                    pool.submit(
                        lambda: client(TensorDict({"observation": torch.randn(4)}))
                    )
                    for _ in range(n)
                ]
                results = [f.result(timeout=10.0) for f in futs]
        assert len(results) == n
        for r in results:
            assert "action" in r.keys()


class TestWorkerCrashPropagation:
    def test_worker_crash_propagates(self):
        """If the model always fails, the collector propagates the error."""

        def bad_model(td):
            raise RuntimeError("model crash")

        collector = AsyncBatchedCollector(
            create_env_fn=[_counting_env_factory] * 2,
            policy=bad_model,
            frames_per_batch=10,
            total_frames=100,
        )
        with pytest.raises(RuntimeError, match="worker thread"):
            for _ in collector:
                pass
        collector.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])
