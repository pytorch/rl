# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import threading
from typing import Any

import torch
from tensordict import TensorDict
from tensordict.base import TensorDictBase

from torchrl._comm.distributed import _DistributedTransport
from torchrl._comm.request_reply import ChannelServer, Message


def _extend_reply(
    result: Any, device: torch.device | str | None = None
) -> TensorDictBase:
    if isinstance(result, torch.Tensor):
        if device is not None:
            result = result.to(device)
        return TensorDict({"result": result}, batch_size=[], device=device)
    return TensorDict(
        {"completed": torch.ones((), dtype=torch.bool, device=device)},
        batch_size=[],
        device=device,
    )


class _DistributedReplayClient:
    """Restricted replay client backed by isolated distributed channels."""

    def __init__(
        self,
        extend_client,
        sample_client,
        control_client,
        priority_client,
        *,
        batch_size: int | None,
        sample_batch_size: int,
    ) -> None:
        self._extend_client = extend_client
        self._sample_client = sample_client
        self._control_client = control_client
        self._priority_client = priority_client
        self._batch_size = batch_size
        self._sample_batch_size = sample_batch_size

    @property
    def batch_size(self) -> int | None:
        return self._batch_size

    @property
    def write_count(self) -> int:
        return int(self._stats()["write_count"].item())

    def extend(self, data: TensorDictBase, *, timeout: float | None = None):
        response = self._extend_client(data, timeout=timeout)
        return response.get("result", None)

    def sample(self, batch_size: int | None = None, *, timeout: float | None = None):
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size is None:
            raise RuntimeError("A sample batch size must be provided.")
        if batch_size != self._sample_batch_size:
            raise ValueError(
                "The distributed replay transport has a fixed sample layout: "
                f"expected {self._sample_batch_size}, got {batch_size}."
            )
        request = TensorDict(
            {"batch_size": torch.tensor(batch_size, dtype=torch.int64)},
            batch_size=[],
        )
        return self._sample_client(request, timeout=timeout)

    def update_tensordict_priority(
        self, data: TensorDictBase, *, timeout: float | None = None
    ) -> None:
        self._priority_client(data, timeout=timeout)

    def _stats(self, *, timeout: float | None = None) -> TensorDictBase:
        request = TensorDict(
            {"operation": torch.zeros((), dtype=torch.int64)}, batch_size=[]
        )
        return self._control_client(request, timeout=timeout)

    def __len__(self) -> int:
        return int(self._stats()["size"].item())

    def __getattr__(self, name: str):
        if name in {"start", "shutdown", "close", "client", "clients"}:
            raise AttributeError(
                f"{type(self).__name__} has no lifecycle capability {name!r}."
            )
        raise AttributeError(name)


class _DistributedReplayService:
    """Serve replay operations over fixed-layout distributed transports."""

    def __init__(
        self,
        replay_buffer: Any,
        *,
        extend_spec: TensorDictBase | None = None,
        sample_spec: TensorDictBase | None = None,
        priority_spec: TensorDictBase | None = None,
        backend: str = "gloo",
        timeout: float = 300.0,
    ) -> None:
        self.replay_buffer = replay_buffer
        self._lock = threading.Lock()
        self._batch_size = getattr(replay_buffer, "batch_size", None)
        self._sample_batch_size = None
        self._wire_device = torch.device("cuda" if backend == "nccl" else "cpu")
        control_request = TensorDict(
            {"operation": torch.zeros((), dtype=torch.int64)}, batch_size=[]
        )
        control_response = TensorDict(
            {
                "size": torch.zeros((), dtype=torch.int64),
                "write_count": torch.zeros((), dtype=torch.int64),
            },
            batch_size=[],
        )
        self._common = {"backend": backend, "timeout": timeout}
        self.control_transport = _DistributedTransport(
            control_request,
            control_response,
            channel_name="replay.control",
            backend="gloo",
            timeout=timeout,
        )
        self._shared = {
            "_store": self.control_transport._store,
            "_store_info": self.control_transport._store_info,
            **self._common,
        }
        self.extend_transport = None
        self.sample_transport = None
        self.priority_transport = None
        self._servers = []
        self._started = False
        self._servers.append(
            ChannelServer(self.control_transport, self._control, name="ReplayControl")
        )
        if extend_spec is not None:
            # An explicit contract cannot know custom writer return values, so
            # use the standard tensor-index layout.
            extend_response = TensorDict(
                {
                    "result": torch.zeros(
                        extend_spec.batch_size,
                        dtype=torch.int64,
                        device=self._wire_device,
                    )
                },
                batch_size=[],
                device=self._wire_device,
            )
            self.bind_extend(extend_spec, extend_response)
        if sample_spec is not None:
            self.bind_sample(sample_spec)
            self.bind_priority(sample_spec if priority_spec is None else priority_spec)

    def _add_server(self, server: ChannelServer) -> None:
        self._servers.append(server)
        if self._started:
            server.start()

    def bind_extend(
        self, request_spec: TensorDictBase, response_spec: TensorDictBase
    ) -> None:
        if self.extend_transport is not None:
            return
        self.extend_transport = _DistributedTransport(
            request_spec,
            response_spec,
            channel_name="replay.extend",
            **self._shared,
        )
        self._add_server(
            ChannelServer(self.extend_transport, self._extend, name="ReplayExtend")
        )

    def bind_sample(self, sample_spec: TensorDictBase) -> None:
        if self.sample_transport is not None:
            return
        if not sample_spec.batch_size:
            raise ValueError("sample_spec must have a non-empty batch size.")
        self._sample_batch_size = int(sample_spec.batch_size[0])
        sample_request = TensorDict(
            {
                "batch_size": torch.zeros(
                    (), dtype=torch.int64, device=self._wire_device
                )
            },
            batch_size=[],
            device=self._wire_device,
        )
        self.sample_transport = _DistributedTransport(
            sample_request,
            sample_spec,
            channel_name="replay.sample",
            **self._shared,
        )
        self._add_server(
            ChannelServer(self.sample_transport, self._sample, name="ReplaySample")
        )

    def bind_priority(self, priority_spec: TensorDictBase) -> None:
        if self.priority_transport is not None:
            return
        priority_response = TensorDict(
            {"updated": torch.zeros((), dtype=torch.bool, device=self._wire_device)},
            batch_size=[],
            device=self._wire_device,
        )
        self.priority_transport = _DistributedTransport(
            priority_spec,
            priority_response,
            channel_name="replay.priority",
            **self._shared,
        )
        self._add_server(
            ChannelServer(
                self.priority_transport, self._priority, name="ReplayPriority"
            )
        )

    def _extend(self, messages: list[Message]) -> list[TensorDictBase]:
        results = []
        with self._lock:
            for message in messages:
                results.append(
                    _extend_reply(
                        self.replay_buffer.extend(message.payload), self._wire_device
                    )
                )
        return results

    def _sample(self, messages: list[Message]) -> list[TensorDictBase]:
        with self._lock:
            return [
                self.replay_buffer.sample(int(message.payload["batch_size"].item())).to(
                    self._wire_device
                )
                for message in messages
            ]

    def _control(self, messages: list[Message]) -> list[TensorDictBase]:
        with self._lock:
            size = len(self.replay_buffer)
            write_count = int(getattr(self.replay_buffer, "write_count", size))
        return [
            TensorDict(
                {
                    "size": torch.tensor(size, dtype=torch.int64),
                    "write_count": torch.tensor(write_count, dtype=torch.int64),
                },
                batch_size=[],
            )
            for _ in messages
        ]

    def _priority(self, messages: list[Message]) -> list[TensorDictBase]:
        update = getattr(self.replay_buffer, "update_tensordict_priority", None)
        results = []
        with self._lock:
            for message in messages:
                if update is not None:
                    update(message.payload)
                results.append(
                    TensorDict(
                        {
                            "updated": torch.tensor(
                                update is not None, device=self._wire_device
                            )
                        },
                        batch_size=[],
                        device=self._wire_device,
                    )
                )
        return results

    def start(self) -> _DistributedReplayService:
        self._started = True
        for server in self._servers:
            server.start()
        return self

    def extend_client(self):
        if self.extend_transport is None:
            raise RuntimeError("The replay extend schema has not been established.")
        return self.extend_transport.client()

    def sample_client(self):
        if self.sample_transport is None:
            raise RuntimeError("The replay sample schema has not been established.")
        return self.sample_transport.client()

    def priority_client(self):
        if self.priority_transport is None:
            raise RuntimeError("The replay priority schema has not been established.")
        return self.priority_transport.client()

    def control_client(self):
        return self.control_transport.client()

    def client(self) -> _DistributedReplayClient:
        if (
            self.extend_transport is None
            or self.sample_transport is None
            or self.priority_transport is None
            or self._sample_batch_size is None
        ):
            raise RuntimeError("The replay transport schemas are not established.")
        return _DistributedReplayClient(
            self.extend_transport.client(),
            self.sample_transport.client(),
            self.control_transport.client(),
            self.priority_transport.client(),
            batch_size=self._batch_size,
            sample_batch_size=self._sample_batch_size,
        )

    def shutdown(self) -> None:
        for server in reversed(self._servers):
            server.shutdown(timeout=5.0)
        for transport in (
            self.priority_transport,
            self.sample_transport,
            self.extend_transport,
            self.control_transport,
        ):
            if transport is not None:
                transport.close()


__all__ = ["_DistributedReplayClient", "_DistributedReplayService"]
