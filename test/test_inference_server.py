# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import concurrent.futures
import importlib.util
import multiprocessing as mp
import pickle
import queue
import threading
import time

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from tensordict import lazy_stack, TensorDict
from tensordict.base import TensorDictBase
from tensordict.nn import TensorDictModule
from tensordict.nn.probabilistic import (
    interaction_type,
    InteractionType,
    set_interaction_type,
)
from torchrl._comm import (
    CommandChannel,
    Mailbox,
    MailboxPeerClosedError,
    MailboxTransportError,
    MappingRendezvous,
    SharedBlock,
    TCPStoreRendezvous,
)

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
    SharedMemoryTransport,
    SlotTransport,
    ThreadingTransport,
)
from torchrl.modules.inference_server._config import _resolve_device_config
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


def test_mailbox_request_metadata_reply_and_exception():
    mailbox = Mailbox(queue.Queue(), queue.Queue)
    client = mailbox.client()
    success = client.submit("success")
    failure = client.submit("failure")

    mailbox.wait_for_work(timeout=1.0)
    payloads, callbacks, submitted_at = mailbox.drain(2)
    assert payloads == ["success", "failure"]
    assert all(timestamp is not None for timestamp in submitted_at)
    mailbox.resolve(callbacks[0], 1)
    mailbox.reject(callbacks[1], ValueError("remote failure"))

    assert success.result(timeout=1.0) == 1
    with pytest.raises(ValueError, match="remote failure"):
        failure.result(timeout=1.0)


def test_mailbox_peer_exit_and_transport_errors_are_distinct_from_timeouts():
    peer_alive = threading.Event()
    peer_alive.set()
    mailbox = Mailbox(queue.Queue(), queue.Queue, peer_alive=peer_alive)
    pending = mailbox.client().submit("pending")
    peer_alive.clear()
    with pytest.raises(MailboxPeerClosedError, match="peer closed"):
        pending.result()

    class _BrokenResponseQueue:
        def get(self, timeout=None):
            del timeout
            raise EOFError("response pipe closed")

    mailbox = Mailbox(queue.Queue(), _BrokenResponseQueue)
    pending = mailbox.client().submit("pending")
    with pytest.raises(MailboxTransportError, match="transport failed") as exc_info:
        pending.result(timeout=1.0)
    assert isinstance(exc_info.value.__cause__, EOFError)


def test_mailbox_reads_a_final_reply_before_reporting_peer_exit():
    peer_alive = threading.Event()
    peer_alive.set()

    class _ReplyBeforeExitQueue:
        def __init__(self):
            self.item = None

        def get(self, block=True, timeout=None):
            del timeout
            if block:
                self.item = (0, "final reply")
                peer_alive.clear()
                raise queue.Empty
            if self.item is None:
                raise queue.Empty
            item = self.item
            self.item = None
            return item

    response_queue = _ReplyBeforeExitQueue()
    mailbox = Mailbox(queue.Queue(), lambda: response_queue, peer_alive=peer_alive)
    pending = mailbox.client().submit("pending")

    assert pending.result() == "final reply"


def test_mailbox_drops_stale_callbacks_without_disrupting_valid_clients():
    mailbox = Mailbox(queue.Queue(), queue.Queue)
    assert not mailbox.resolve((123, 0), "stale")

    client = mailbox.client()
    pending = client.submit("valid")
    mailbox.wait_for_work(timeout=1.0)
    payloads, callbacks, _ = mailbox.drain(1)
    assert payloads == ["valid"]
    assert mailbox.resolve(callbacks[0], "result")
    assert pending.result(timeout=1.0) == "result"


def test_command_channel_order_timeout_and_close():
    channel = CommandChannel(Mailbox(queue.Queue(), queue.Queue))
    client = channel.client()
    first = client._mailbox_client.submit({"verb": "first", "payload": 1})
    second = client._mailbox_client.submit({"verb": "second", "payload": 2})

    first_request = channel.receive(timeout=1.0)
    second_request = channel.receive(timeout=1.0)
    assert first_request.verb == "first"
    assert second_request.verb == "second"
    channel.resolve(first_request, "one")
    channel.resolve(second_request, "two")
    assert first.result(timeout=1.0) == "one"
    assert second.result(timeout=1.0) == "two"
    assert channel.receive(timeout=0.01) is None

    pending = client._mailbox_client.submit({"verb": "pending", "payload": {}})
    channel.close()
    with pytest.raises(RuntimeError, match="closed"):
        pending.result(timeout=1.0)
    channel.close()


def test_shared_block_and_mapping_rendezvous():
    value = torch.zeros(2).share_memory_()
    block = SharedBlock(value)
    assert block.version == 0
    assert block.wait(after_version=0, timeout=0.01) is None
    assert block.publish(torch.ones(2)) == 1
    shared, version = block.wait(after_version=0, timeout=1.0)
    assert version == 1
    torch.testing.assert_close(shared, torch.ones(2))

    rendezvous = MappingRendezvous({})
    rendezvous.publish("worker", {"rank": 1})
    assert rendezvous.read("worker") == {"rank": 1}
    assert rendezvous.wait("worker", timeout=1.0) == {"rank": 1}


def test_tcp_store_rendezvous_preserves_custom_wire_encoding():
    class _Store:
        def __init__(self):
            self.data = {}

        def set(self, key, value):
            self.data[key] = value

        def get(self, key):
            return self.data[key]

    store = _Store()
    rendezvous = TCPStoreRendezvous(
        store,
        encode=lambda value: b"1" if value else b"0",
        decode=lambda value: value == b"1",
    )
    rendezvous.publish("stateful", True)
    assert store.data["stateful"] == b"1"
    assert rendezvous.read("stateful") is True


class TestInferenceTransportABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            InferenceTransport()

    def test_client_returns_inference_client(self):
        transport = _MockTransport()
        client = transport.client()
        assert isinstance(client, InferenceClient)


class TestInferenceServerCore:
    def test_canonical_thread_constructor(self):
        with InferenceServer(_make_policy(), transport="auto") as server:
            assert server.service_backend == "thread"
            assert server.transport_kind == "thread"
            result = server.client()(
                TensorDict({"observation": torch.randn(4)}, batch_size=[])
            )
        assert result["action"].shape == (2,)

    @pytest.mark.parametrize(
        ("service_backend", "transport"),
        [("thread", "ray"), ("process", "ray"), ("ray", "shared_memory")],
    )
    def test_invalid_backend_transport_rejected_before_start(
        self, service_backend, transport
    ):
        kwargs = {
            "service_backend": service_backend,
            "transport": transport,
        }
        if service_backend == "thread":
            kwargs["model"] = _make_policy()
        else:
            kwargs["policy_factory"] = _make_policy
        with pytest.raises(ValueError, match="incompatible"):
            InferenceServer(**kwargs)

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

    def test_batch_of_requests_with_mixed_root_device_metadata(self):
        transport = _MockTransport()
        policy = _make_policy()
        cpu_item = TensorDict(
            {"observation": torch.randn(4), "next": TensorDict({}, [])},
            [],
            device="cpu",
        )
        metadata_free_item = TensorDict(
            {"observation": torch.randn(4), "next": TensorDict({}, [])},
            [],
        )
        with InferenceServer(policy, transport, max_batch_size=2, min_batch_size=2):
            futures = [
                transport.submit(cpu_item),
                transport.submit(metadata_free_item),
            ]
            results = [f.result(timeout=5.0) for f in futures]

        assert len(results) == 2
        for result in results:
            assert "action" in result.keys()
            assert result["action"].shape == (2,)

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

    def test_update_model_increments_policy_version(self):
        transport = ThreadingTransport()
        policy = _make_policy()
        with InferenceServer(policy, transport) as server:
            client = transport.client()
            observation = torch.randn(4)
            before = client(TensorDict({"observation": observation.clone()}))

            def update(model):
                with torch.no_grad():
                    model.module.weight.zero_()
                    model.module.bias.fill_(1.0)

            server.update_model(update)
            after = client(TensorDict({"observation": observation.clone()}))
            stats = server.stats()

        assert before["policy_version"].item() == 0
        assert after["policy_version"].item() == 1
        assert stats["policy_version"] == 1
        assert stats["weight_updates"] == 1
        torch.testing.assert_close(after["action"], torch.ones(2))

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


class TestDeviceResolution:
    """Unit tests for the shared device-precedence resolution.

    ``_resolve_device_config`` is the single source of truth for the device
    precedence rules shared by the inference servers, AsyncBatchedCollector
    and the regular collectors (issue #3943).
    """

    def test_device_aliases_policy_device_only(self):
        resolved = _resolve_device_config(device="cpu")
        assert resolved.policy_device == torch.device("cpu")
        assert resolved.output_device is None
        assert resolved.env_device is None
        assert resolved.storing_device is None

    def test_explicit_policy_device_wins_over_device(self):
        resolved = _resolve_device_config(device="meta", policy_device="cpu")
        assert resolved.policy_device == torch.device("cpu")

    def test_output_device_falls_back_to_env_device(self):
        resolved = _resolve_device_config(env_device="cpu")
        assert resolved.output_device == torch.device("cpu")
        resolved = _resolve_device_config(env_device="cpu", output_device="meta")
        assert resolved.output_device == torch.device("meta")

    def test_config_passthrough(self):
        config = InferenceDeviceConfig(
            policy_device="cpu",
            output_device="meta",
            env_device="cpu",
            storing_device="cpu",
        )
        resolved = _resolve_device_config(config)
        assert resolved.policy_device == torch.device("cpu")
        assert resolved.output_device == torch.device("meta")
        assert resolved.env_device == torch.device("cpu")
        assert resolved.storing_device == torch.device("cpu")

    def test_device_config_exclusive_with_every_loose_kwarg(self):
        for kwarg in (
            "device",
            "policy_device",
            "output_device",
            "env_device",
            "storing_device",
        ):
            with pytest.raises(ValueError, match=f"mutually exclusive.*{kwarg}"):
                _resolve_device_config(InferenceDeviceConfig(), **{kwarg: "cpu"})

    def test_storing_device_rejected_when_disallowed(self):
        with pytest.raises(ValueError, match="storing_device is a collector-level"):
            _resolve_device_config(
                InferenceDeviceConfig(storing_device="cpu"),
                allow_storing_device=False,
            )
        with pytest.raises(ValueError, match="storing_device is a collector-level"):
            _resolve_device_config(storing_device="cpu", allow_storing_device=False)

    def test_collector_defaults_device_fills_all(self):
        resolved = _resolve_device_config(device="cpu", collector_defaults=True)
        assert resolved.policy_device == torch.device("cpu")
        assert resolved.env_device == torch.device("cpu")
        assert resolved.storing_device == torch.device("cpu")

    def test_collector_defaults_storing_falls_back_to_shared_device(self):
        resolved = _resolve_device_config(
            policy_device="cpu", env_device="cpu", collector_defaults=True
        )
        assert resolved.storing_device == torch.device("cpu")
        resolved = _resolve_device_config(
            policy_device="meta", env_device="cpu", collector_defaults=True
        )
        assert resolved.storing_device is None

    def test_collector_defaults_make_ordinal(self):
        # Device objects can be built without the backend being available, so
        # this runs on CPU-only CI as well.
        resolved = _resolve_device_config(device="mps", collector_defaults=True)
        assert resolved.policy_device == torch.device("mps", 0)
        assert resolved.env_device == torch.device("mps", 0)
        assert resolved.storing_device == torch.device("mps", 0)

    @pytest.mark.gpu
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="torch.device(0) cannot be constructed without an accelerator",
    )
    def test_integer_device_zero_is_not_dropped(self):
        # ``torch.device(0)`` is falsy-adjacent (int 0); the resolution must
        # treat it as an explicit device, not as unset.
        resolved = _resolve_device_config(storing_device=0, collector_defaults=True)
        assert resolved.storing_device == torch.device(0)

    def test_collector_get_devices_delegates(self):
        storing, policy, env = Collector._get_devices(
            storing_device=None,
            policy_device=None,
            env_device=None,
            device="cpu",
        )
        assert storing == policy == env == torch.device("cpu")
        storing, policy, env = Collector._get_devices(
            storing_device=None,
            policy_device="cpu",
            env_device="cpu",
            device=None,
        )
        assert storing == torch.device("cpu")
        storing, policy, env = Collector._get_devices(
            storing_device=None,
            policy_device=None,
            env_device=None,
            device=None,
        )
        assert storing is None and policy is None and env is None


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


def _echo_client(td):
    """Picklable stand-in client for pickling tests."""
    return td


class TestPolicyClientModule:
    def test_service_owner_is_automatically_restricted_to_client(self):
        transport = ThreadingTransport()
        policy = _make_policy()
        with InferenceServer(policy, transport, max_batch_size=4) as server:
            client = server.client()
            assert not hasattr(client, "shutdown")
            remote_policy = PolicyClientModule(
                server,
                in_keys=["observation"],
                out_keys=["action"],
            )
            result = remote_policy(TensorDict({"observation": torch.randn(4)}))
        assert result["action"].shape == (2,)

    def test_forward_as_tensordict_module(self):
        transport = ThreadingTransport()
        policy = _make_policy()
        with InferenceServer(policy, transport, max_batch_size=4):
            remote_policy = PolicyClientModule(
                transport,
                in_keys=["observation"],
                out_keys=["action"],
                max_inflight=1,
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

    def test_client_contract_picklable_no_lifecycle(self):
        """Clients pickle cleanly and expose no lifecycle methods."""
        remote_policy = PolicyClientModule(
            _echo_client, in_keys=["observation"], out_keys=["observation"]
        )
        restored = pickle.loads(pickle.dumps(remote_policy))
        td = TensorDict({"observation": torch.randn(4)})
        result = restored(td)
        assert "observation" in result.keys()
        assert restored.in_keys == ["observation"]
        # Clients carry no lifecycle rights over the service
        for lifecycle in ("start", "shutdown", "close", "flush"):
            assert not hasattr(restored, lifecycle)

    def test_plain_callable_client_defers_errors(self):
        """A plain-callable client defers exceptions to result()."""

        def failing_client(td):
            raise ValueError("local policy failure")

        remote_policy = PolicyClientModule(failing_client)
        future = remote_policy.submit(TensorDict({}))
        assert future.done()
        with pytest.raises(ValueError, match="local policy failure"):
            future.result()

    def test_update_policy_weights_cascade_bumps_version(self):
        """The weight-sync cascade hook increments the policy version."""
        transport = ThreadingTransport()
        policy = _make_policy()
        server = InferenceServer(policy, transport, policy_version=0)
        assert server.policy_version == 0
        server.update_policy_weights_()
        assert server.policy_version == 1

    def test_max_inflight_validation(self):
        with pytest.raises(ValueError, match="max_inflight must be at least 1"):
            PolicyClientModule(lambda td: td, max_inflight=0)
        with pytest.raises(ValueError, match="max_inflight must be at least 1"):
            PolicyClientModule(lambda td: td, max_inflight=-3)
        with pytest.raises(ValueError, match="max_inflight_per_env"):
            InferenceServerConfig(max_inflight_per_env=0)

    def test_max_inflight_survives_pickling(self):
        """The guard is rebuilt on unpickle; clients stay picklable."""
        remote_policy = PolicyClientModule(_echo_client, max_inflight=2)
        restored = pickle.loads(pickle.dumps(remote_policy))
        assert restored.max_inflight == 2
        assert restored._inflight_sem is not None
        # The rebuilt guard still enforces the limit
        release_one = restored._acquire_inflight()
        release_two = restored._acquire_inflight()
        acquired = restored._inflight_sem.acquire(blocking=False)
        assert not acquired
        release_one()
        release_two()
        result = restored(TensorDict({"observation": torch.randn(4)}))
        assert "observation" in result.keys()

    def test_max_inflight_blocks_until_completion(self):
        """The guard blocks a second submit and frees on completion (not result())."""

        class _ManualClient:
            def __init__(self):
                self.futures = []

            def submit(self, td):
                fut = concurrent.futures.Future()
                self.futures.append(fut)
                return fut

        client = _ManualClient()
        remote = PolicyClientModule(client, max_inflight=1)
        remote.submit(TensorDict({}))
        second_done = threading.Event()

        def _second_submit():
            remote.submit(TensorDict({}))
            second_done.set()

        t = threading.Thread(target=_second_submit, daemon=True)
        t.start()
        time.sleep(0.2)
        assert not second_done.is_set()  # guard enforced
        # Completing the first request frees the slot even though result()
        # is never called on it (done-callback release).
        client.futures[0].set_result(TensorDict({}))
        assert second_done.wait(timeout=5.0)
        t.join(timeout=5.0)

    def test_max_inflight_releases_on_error(self):
        """A failed request frees its slot; the guard cannot deadlock."""

        class _ManualClient:
            def __init__(self):
                self.futures = []

            def submit(self, td):
                fut = concurrent.futures.Future()
                self.futures.append(fut)
                return fut

        client = _ManualClient()
        remote = PolicyClientModule(client, max_inflight=1)
        fut = remote.submit(TensorDict({}))
        client.futures[0].set_exception(ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            fut.result(timeout=5.0)
        # Slot was released; this would deadlock otherwise.
        fut2 = remote.submit(TensorDict({}))
        client.futures[1].set_result(TensorDict({}))
        assert fut2.result(timeout=5.0) is not None

    def test_max_inflight_timeout_keeps_slot(self):
        """A result() timeout must not free the slot of a running request."""

        class _PullFuture:
            """Pull-based future without add_done_callback support."""

            def __init__(self):
                self._event = threading.Event()
                self._result = None

            def done(self):
                return self._event.is_set()

            def result(self, timeout=None):
                if not self._event.wait(timeout=timeout):
                    raise TimeoutError("still running")
                return self._result

            def set_result(self, result):
                self._result = result
                self._event.set()

        class _PullClient:
            def __init__(self):
                self.futures = []

            def submit(self, td):
                fut = _PullFuture()
                self.futures.append(fut)
                return fut

        client = _PullClient()
        remote = PolicyClientModule(client, max_inflight=1)
        fut = remote.submit(TensorDict({}))
        with pytest.raises(TimeoutError):
            fut.result(timeout=0.05)
        second_done = threading.Event()

        def _second_submit():
            remote.submit(TensorDict({}))
            second_done.set()

        t = threading.Thread(target=_second_submit, daemon=True)
        t.start()
        time.sleep(0.2)
        # The request is still inflight: its slot must still be held.
        assert not second_done.is_set()
        client.futures[0].set_result(TensorDict({}))
        assert fut.result(timeout=5.0) is not None
        assert second_done.wait(timeout=5.0)
        t.join(timeout=5.0)


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
    def test_single_request_with_manager_queues(self):
        """Manager-backed MPTransport works from the parent process."""
        ctx = mp.get_context("spawn")
        transport = MPTransport(ctx=ctx, use_manager=True)
        client = transport.client()
        policy = _make_policy()
        try:
            with InferenceServer(policy, transport, max_batch_size=4):
                td = TensorDict({"observation": torch.randn(4)})
                result = client(td)
                assert "action" in result.keys()
                assert result["action"].shape == (2,)
        finally:
            transport.close()

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
# Tests: SharedMemoryTransport
# =============================================================================


class _Doubler(nn.Module):
    def forward(self, x):
        return x * 2.0


def _make_shm_specs(obs_size=4, act_size=2, with_version=True):
    request_spec = TensorDict({"observation": torch.zeros(obs_size)})
    response = {"action": torch.zeros(act_size)}
    if with_version:
        response["policy_version"] = torch.zeros((), dtype=torch.long)
    return request_spec, TensorDict(response)


def _make_doubling_policy():
    return TensorDictModule(_Doubler(), in_keys=["observation"], out_keys=["action"])


def _shm_actor_fn(client, n_requests, result_queue):
    """Actor that submits known values and checks the doubled response."""
    for i in range(n_requests):
        obs = torch.full((4,), float(i + 1))
        result = client(TensorDict({"observation": obs}))
        assert torch.allclose(result["action"], obs * 2.0)
    result_queue.put(True)


class TestSharedMemoryTransport:
    def test_single_request_in_process(self):
        """A client in the owning process gets a correct result."""
        request_spec, response_spec = _make_shm_specs()
        transport = SharedMemoryTransport(request_spec, response_spec, num_slots=4)
        client = transport.client()
        policy = _make_policy()
        with InferenceServer(policy, transport, max_batch_size=4):
            td = TensorDict({"observation": torch.randn(4)})
            result = client(td)
            assert "action" in result.keys()
            assert "policy_version" in result.keys()
            assert result["action"].shape == (2,)

    def test_undeclared_keys_are_dropped(self):
        """Only spec-declared keys travel through the transport."""
        request_spec, response_spec = _make_shm_specs(act_size=4, with_version=False)
        transport = SharedMemoryTransport(request_spec, response_spec, num_slots=2)
        client = transport.client()
        with InferenceServer(_make_doubling_policy(), transport, max_batch_size=2):
            obs = torch.arange(4, dtype=torch.float32)
            td = TensorDict({"observation": obs, "extra": torch.randn(3)})
            result = client(td)
            assert set(result.keys()) == {"action"}
            assert torch.allclose(result["action"], obs * 2.0)

    def test_missing_declared_key_raises(self):
        """A request missing a declared key fails fast, before slot use."""
        request_spec, response_spec = _make_shm_specs()
        transport = SharedMemoryTransport(request_spec, response_spec, num_slots=1)
        client = transport.client()
        with pytest.raises(KeyError):
            client.submit(TensorDict({"wrong_key": torch.zeros(4)}))

    def test_multi_client_concurrent(self):
        """Concurrent clients each get their own routed results."""
        request_spec, response_spec = _make_shm_specs(act_size=4, with_version=False)
        transport = SharedMemoryTransport(request_spec, response_spec, num_slots=8)
        n_clients = 4
        clients = [transport.client() for _ in range(n_clients)]
        errors = []

        def run(idx):
            try:
                for i in range(5):
                    obs = torch.full((4,), float(idx * 10 + i))
                    result = clients[idx](TensorDict({"observation": obs}))
                    assert torch.allclose(result["action"], obs * 2.0)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        with InferenceServer(
            _make_doubling_policy(), transport, max_batch_size=n_clients
        ):
            threads = [
                threading.Thread(target=run, args=(i,)) for i in range(n_clients)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30.0)
        assert not errors

    def test_out_of_order_result_consumption(self):
        """Futures can be consumed in any order (response buffering)."""
        request_spec, response_spec = _make_shm_specs(act_size=4, with_version=False)
        transport = SharedMemoryTransport(request_spec, response_spec, num_slots=4)
        client = transport.client()
        with InferenceServer(_make_doubling_policy(), transport, max_batch_size=4):
            fut_a = client.submit(TensorDict({"observation": torch.full((4,), 1.0)}))
            fut_b = client.submit(TensorDict({"observation": torch.full((4,), 2.0)}))
            result_b = fut_b.result(timeout=10.0)
            result_a = fut_a.result(timeout=10.0)
        assert torch.allclose(result_b["action"], torch.full((4,), 4.0))
        assert torch.allclose(result_a["action"], torch.full((4,), 2.0))

    def test_exception_propagates_and_slot_is_released(self):
        """Model errors reach the client and free the slot for reuse."""

        def flaky_model(td):
            if (td["observation"] < 0).any():
                raise ValueError("shm model error")
            return td.set("action", td["observation"] * 2.0)

        request_spec, response_spec = _make_shm_specs(act_size=4, with_version=False)
        transport = SharedMemoryTransport(request_spec, response_spec, num_slots=1)
        client = transport.client()
        with InferenceServer(flaky_model, transport, max_batch_size=1):
            with pytest.raises(ValueError, match="shm model error"):
                client(TensorDict({"observation": -torch.ones(4)}))
            # With a single slot, this only completes if the failed request
            # released its slot. Run in a thread so a leak fails the test
            # instead of hanging it.
            obs = torch.ones(4)
            done = threading.Event()
            out = {}

            def resubmit():
                out["result"] = client(TensorDict({"observation": obs}))
                done.set()

            t = threading.Thread(target=resubmit, daemon=True)
            t.start()
            assert done.wait(timeout=10.0), "slot was not released after exception"
            assert torch.allclose(out["result"]["action"], obs * 2.0)

    def test_nested_keys(self):
        """Nested request and response keys round-trip through the slots."""
        request_spec = TensorDict({"agent": {"observation": torch.zeros(4)}})
        response_spec = TensorDict({"agent": {"action": torch.zeros(4)}})
        transport = SharedMemoryTransport(request_spec, response_spec, num_slots=2)
        client = transport.client()
        model = TensorDictModule(
            _Doubler(),
            in_keys=[("agent", "observation")],
            out_keys=[("agent", "action")],
        )
        with InferenceServer(model, transport, max_batch_size=2):
            obs = torch.arange(4, dtype=torch.float32)
            result = client(TensorDict({"agent": {"observation": obs}}))
            assert torch.allclose(result["agent", "action"], obs * 2.0)

    def test_slot_backpressure(self):
        """With all slots in flight, submit blocks until a slot is freed."""
        request_spec, response_spec = _make_shm_specs(with_version=False)
        transport = SharedMemoryTransport(request_spec, response_spec, num_slots=1)
        client = transport.client()
        fut1 = client.submit(TensorDict({"observation": torch.ones(4)}))
        submitted = threading.Event()
        futures = {}

        def second_submit():
            futures["fut2"] = client.submit(
                TensorDict({"observation": 2 * torch.ones(4)})
            )
            submitted.set()

        t = threading.Thread(target=second_submit, daemon=True)
        t.start()
        # The only slot is held by the first in-flight request.
        assert not submitted.wait(timeout=0.5)
        # Resolve the first request manually (server side).
        transport.wait_for_work(timeout=5.0)
        items, callbacks = transport.drain(4)
        assert len(items) == 1
        assert torch.allclose(items[0]["observation"], torch.ones(4))
        transport.resolve(callbacks[0], TensorDict({"action": torch.ones(2)}))
        assert torch.allclose(fut1.result(timeout=5.0)["action"], torch.ones(2))
        # Reading the result released the slot: the second submit unblocks.
        assert submitted.wait(timeout=10.0)
        transport.wait_for_work(timeout=5.0)
        items, callbacks = transport.drain(4)
        assert len(items) == 1
        assert torch.allclose(items[0]["observation"], 2 * torch.ones(4))
        transport.resolve(callbacks[0], TensorDict({"action": 2 * torch.ones(2)}))
        assert torch.allclose(
            futures["fut2"].result(timeout=5.0)["action"], 2 * torch.ones(2)
        )
        t.join(timeout=5.0)

    def test_result_timeout_keeps_slot(self):
        """A timed-out result() keeps the request in flight and retryable."""
        request_spec, response_spec = _make_shm_specs(with_version=False)
        transport = SharedMemoryTransport(request_spec, response_spec, num_slots=1)
        client = transport.client()
        fut = client.submit(TensorDict({"observation": torch.ones(4)}))
        with pytest.raises(queue.Empty):
            fut.result(timeout=0.1)
        transport.wait_for_work(timeout=5.0)
        items, callbacks = transport.drain(1)
        assert len(items) == 1
        transport.resolve(callbacks[0], TensorDict({"action": torch.ones(2)}))
        assert torch.allclose(fut.result(timeout=5.0)["action"], torch.ones(2))

    def test_copy_result_false_returns_borrowed_view(self):
        """copy_result=False returns a view into the shared response slot."""
        request_spec, response_spec = _make_shm_specs(with_version=False)
        transport = SharedMemoryTransport(
            request_spec, response_spec, num_slots=1, copy_result=False
        )
        client = transport.client()
        fut = client.submit(TensorDict({"observation": torch.ones(4)}))
        transport.wait_for_work(timeout=5.0)
        items, callbacks = transport.drain(1)
        transport.resolve(callbacks[0], TensorDict({"action": torch.ones(2)}))
        result = fut.result(timeout=5.0)
        assert (
            result["action"].data_ptr()
            == transport._response_slots["action"][0].data_ptr()
        )

    def test_spec_validation(self):
        """Bad specs are rejected at construction time."""
        response_spec = TensorDict({"action": torch.zeros(2)})
        with pytest.raises(TypeError, match="tensor leaves"):
            SharedMemoryTransport(
                TensorDict({"instruction": "hello"}), response_spec, num_slots=1
            )
        with pytest.raises(ValueError, match="at least one tensor leaf"):
            SharedMemoryTransport(TensorDict({}), response_spec, num_slots=1)
        with pytest.raises(ValueError, match="num_slots"):
            SharedMemoryTransport(
                TensorDict({"observation": torch.zeros(4)}),
                response_spec,
                num_slots=0,
            )

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_input_raises(self):
        """CUDA tensors are rejected: slots are CPU shared memory only."""
        request_spec, response_spec = _make_shm_specs()
        transport = SharedMemoryTransport(request_spec, response_spec, num_slots=1)
        client = transport.client()
        with pytest.raises(ValueError, match="CPU tensors"):
            client.submit(TensorDict({"observation": torch.randn(4, device="cuda")}))
        with pytest.raises(ValueError, match="CPU shared memory"):
            SharedMemoryTransport(
                TensorDict({"observation": torch.zeros(4, device="cuda")}),
                response_spec,
                num_slots=1,
            )

    @pytest.mark.slow
    def test_cross_process_actors(self):
        """Spawned child clients submit; the parent server resolves."""
        ctx = mp.get_context("spawn")
        request_spec, response_spec = _make_shm_specs(act_size=4, with_version=False)
        transport = SharedMemoryTransport(
            request_spec, response_spec, num_slots=8, ctx=ctx
        )
        n_actors = 2
        n_requests = 10
        result_queue = ctx.Queue()
        # Create clients before spawning (queues and slots inherited)
        clients = [transport.client() for _ in range(n_actors)]
        with InferenceServer(_make_doubling_policy(), transport, max_batch_size=8):
            procs = []
            for i in range(n_actors):
                p = ctx.Process(
                    target=_shm_actor_fn,
                    args=(clients[i], n_requests, result_queue),
                )
                p.start()
                procs.append(p)
            for p in procs:
                p.join(timeout=60.0)
                assert p.exitcode == 0
        for _ in range(n_actors):
            assert result_queue.get(timeout=1.0) is True


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

    def test_canonical_ray_owned_inference(self):
        ray = _ray_lib()
        server = InferenceServer(
            policy_factory=lambda: TensorDictModule(
                nn.Linear(4, 2),
                in_keys=["observation"],
                out_keys=["action"],
            ),
            service_backend="ray",
            service_backend_options={"remote_config": {"num_cpus": 0}},
            transport="auto",
        )
        try:
            assert server.service_backend == "ray"
            assert server.transport_kind == "ray"
            clients = server.clients(2)
            assert clients[0] is not clients[1]
            result = clients[0](
                TensorDict({"observation": torch.randn(4)}, batch_size=[])
            )
            assert result["action"].shape == (2,)
            assert not hasattr(clients[0], "shutdown")
        finally:
            server.shutdown()
        # The fixture owns Ray; component shutdown must leave it running.
        assert ray.is_initialized()

    def test_ray_owned_inference_with_gloo_transport(self):
        server = InferenceServer(
            policy_factory=lambda: TensorDictModule(
                nn.Linear(4, 2),
                in_keys=["observation"],
                out_keys=["action"],
            ),
            service_backend="ray",
            service_backend_options={"remote_config": {"num_cpus": 0}},
            transport="distributed",
            transport_options={"backend": "gloo", "timeout": 30.0},
        )
        try:
            client = server.client()
            result = client(
                TensorDict(
                    {"observation": torch.randn(4)},
                    batch_size=[],
                    device="cpu",
                ),
                timeout=30.0,
            )
            assert result["action"].shape == (2,)
            assert server.transport_kind == "distributed"
            assert not dist.is_initialized()
            server.shutdown()
            started = time.monotonic()
            with pytest.raises((MailboxPeerClosedError, MailboxTransportError)):
                client(
                    TensorDict({"observation": torch.randn(4)}, batch_size=[]),
                    timeout=5.0,
                )
            assert time.monotonic() - started < 3.0
        finally:
            server.shutdown()

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_ray_owned_inference_with_nccl_transport(self):
        server = InferenceServer(
            policy_factory=lambda: TensorDictModule(
                nn.Linear(4, 2),
                in_keys=["observation"],
                out_keys=["action"],
            ),
            service_backend="ray",
            service_backend_options={"remote_config": {"num_gpus": 1}},
            transport="distributed",
            transport_options={"backend": "nccl", "timeout": 30.0},
            output_device="cuda:0",
        )
        try:
            result = server.client()(
                TensorDict(
                    {"observation": torch.randn(4, device="cuda")},
                    batch_size=[],
                    device="cuda",
                ),
                timeout=30.0,
            )
            assert result["action"].is_cuda
            assert not dist.is_initialized()
        finally:
            server.shutdown()

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

    def test_policy_client_propagates_interaction_type(self):
        class _InteractionPolicy(nn.Module):
            def forward(self, td: TensorDictBase) -> TensorDictBase:
                value = 1 if interaction_type() is InteractionType.RANDOM else 0
                return TensorDict(
                    {"action": torch.full(td.batch_size, value)},
                    batch_size=td.batch_size,
                )

        transport = ThreadingTransport()
        policy = _InteractionPolicy()
        with InferenceServer(policy, transport, max_batch_size=4):
            client = PolicyClientModule(transport, out_keys=["action"])
            # The caller's exploration context reaches the server-side
            # forward without any opt-in.
            with set_interaction_type(InteractionType.RANDOM):
                result = client(TensorDict({}, batch_size=[1]))
            assert result["action"].item() == 1
            # Without an active context the server runs a bare forward.
            result = client(TensorDict({}, batch_size=[1]))
            assert result["action"].item() == 0


# ---------------------------------------------------------------------------
# AsyncBatchedCollector tests
# ---------------------------------------------------------------------------

from torchrl.collectors import AsyncBatchedCollector, Collector
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
    def test_canonical_process_constructor(self):
        server = InferenceServer(
            policy_factory=_make_policy,
            service_backend="process",
            transport="auto",
            service_backend_options={"mp_context": "spawn"},
        )
        try:
            assert server.service_backend == "process"
            assert server.transport_kind == "process"
            with server:
                result = server.client()(
                    TensorDict({"observation": torch.randn(4)}, batch_size=[])
                )
            assert result["action"].shape == (2,)
        finally:
            server.shutdown()

    def test_process_server_with_distributed_gloo_transport(self):
        request_spec = TensorDict({"observation": torch.zeros(4)}, batch_size=[])
        response_spec = TensorDict(
            {
                "action": torch.zeros(2),
                "policy_version": torch.zeros((), dtype=torch.long),
            },
            batch_size=[],
        )
        server = InferenceServer(
            policy_factory=_make_policy,
            service_backend="process",
            service_backend_options={"mp_context": "spawn"},
            transport="distributed",
            transport_options={"backend": "gloo", "timeout": 30.0},
            request_spec=request_spec,
            response_spec=response_spec,
        )
        try:
            with server:
                result = server.client()(
                    TensorDict({"observation": torch.randn(4)}, batch_size=[]),
                    timeout=30.0,
                )
                assert result["action"].shape == (2,)
                assert server.transport_kind == "distributed"
                assert not dist.is_initialized()
        finally:
            server.shutdown()

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
            result = client(TensorDict({"observation": torch.ones(1)}))
            stats = server.stats()
            health = server.health()
        assert "action" in result.keys()
        assert result["action"].shape == (1,)
        assert stats["requests"] == 1
        assert stats["avg_batch_size"] == 1
        assert health["process_alive"]
        assert not server.is_alive

    def test_process_server_client(self):
        ctx = mp.get_context("spawn")
        transport = MPTransport(ctx=ctx)
        with ProcessInferenceServer(
            policy_factory=_make_counting_policy,
            transport=transport,
            max_batch_size=4,
            mp_context=ctx,
        ) as server:
            client = server.client()
            assert not hasattr(client, "shutdown")
            result = client(TensorDict({"observation": torch.ones(1)}))
        assert result["action"].shape == (1,)

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

    def test_killed_server_unblocks_waiting_clients(self):
        """A killed server process makes blocked clients raise promptly.

        Clients created before the server object exist must also observe the
        liveness flag (MPTransport bakes it into every client), otherwise an
        untimed wait would block forever on a reply that never comes.
        """
        ctx = mp.get_context("spawn")
        transport = MPTransport(ctx=ctx)
        client = transport.client()
        server = ProcessInferenceServer(
            policy_factory=_make_counting_policy,
            transport=transport,
            mp_context=ctx,
        )
        server.start()
        try:
            result = client(TensorDict({"observation": torch.ones(1)}))
            assert "action" in result.keys()
            server._process.kill()
            with pytest.raises(MailboxPeerClosedError, match="peer closed"):
                client(TensorDict({"observation": torch.ones(1)}))
        finally:
            server.shutdown(timeout=1.0)

    def test_control_plane_thread_safe(self):
        """Concurrent stats()/health() callers must not steal replies."""
        ctx = mp.get_context("spawn")
        transport = MPTransport(ctx=ctx)
        transport.client()
        errors = []

        def _hammer(server, fn_name):
            try:
                for _ in range(10):
                    result = getattr(server, fn_name)()
                    assert isinstance(result, dict)
            except Exception as exc:
                errors.append(exc)

        with ProcessInferenceServer(
            policy_factory=_make_counting_policy,
            transport=transport,
            mp_context=ctx,
        ) as server:
            threads = [
                threading.Thread(target=_hammer, args=(server, fn_name))
                for fn_name in ("stats", "health", "stats", "health")
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30.0)
        assert not errors

    def test_stats_reset_and_dead_server_errors(self):
        ctx = mp.get_context("spawn")
        transport = MPTransport(ctx=ctx)
        client = transport.client()
        server = ProcessInferenceServer(
            policy_factory=_make_counting_policy,
            transport=transport,
            mp_context=ctx,
        )
        server.start()
        try:
            client(TensorDict({"observation": torch.ones(1)}))
            stats = server.stats(reset=True)
            assert stats["requests"] == 1
            stats = server.stats()
            assert stats["requests"] == 0
            with pytest.raises(RuntimeError, match="Unknown process-server verb"):
                server._request_control("bogus")
        finally:
            server.shutdown()
        with pytest.raises(RuntimeError, match="not running|not alive"):
            server.stats()
        # health() degrades instead of raising
        health = server.health()
        assert not health["process_alive"]


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
            env_backend="threading",
            server_config=InferenceServerConfig(
                service_backend="process", max_batch_size=2
            ),
        )
        total = 0
        for batch in collector:
            total += batch.numel()
        collector.shutdown()
        assert total >= 20

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
        with pytest.raises(ValueError, match="backend"):
            InferenceServerConfig(service_backend="not-a-backend")

    def test_server_death_raises_instead_of_hanging(self):
        """Killing the server process surfaces an error in the iterator."""
        collector = AsyncBatchedCollector(
            create_env_fn=[_counting_env_factory] * 2,
            policy_factory=_make_counting_policy,
            frames_per_batch=10,
            total_frames=-1,
            env_backend="threading",
            server_config=InferenceServerConfig(
                service_backend="process", max_batch_size=2
            ),
        )
        try:
            iterator = iter(collector)
            next(iterator)
            workers = list(collector._workers)
            collector._server._process.kill()
            # Worker errors after a kill are mailbox transport failures,
            # which _check_worker_result attributes to the dead server
            # deterministically (no is_alive race).
            with pytest.raises(RuntimeError, match="inference server died"):
                # A couple of batches may still drain from already-queued
                # transitions before the watchdog trips.
                for _ in range(10):
                    next(iterator)
            # Worker threads blocked on inference must observe the cleared
            # liveness flag and exit instead of leaking into later tests.
            for w in workers:
                w.join(timeout=10.0)
            assert not any(w.is_alive() for w in workers)
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
