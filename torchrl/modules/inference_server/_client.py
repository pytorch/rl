# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Future

from tensordict.base import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey

from torchrl.modules.inference_server._transport import InferenceTransport


class _ImmediateFuture:
    def __init__(self, result: TensorDictBase | BaseException):
        self._result = result

    def done(self) -> bool:
        return True

    def result(self, timeout: float | None = None) -> TensorDictBase:
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


class PolicyClientModule(TensorDictModuleBase):
    """TensorDict policy wrapper for remote inference-server clients.

    ``PolicyClientModule`` makes a transport client look like a TorchRL policy:
    it accepts a :class:`~tensordict.TensorDictBase`, submits it to an
    :class:`~torchrl.modules.inference_server.InferenceServer`, and returns the
    TensorDict produced by the remote policy. It can be passed anywhere a
    TensorDict policy module is expected.

    .. note::
        Unlike a local :class:`~tensordict.nn.TensorDictModule`, the result
        crosses a transport boundary, so :meth:`forward` returns a *new*
        TensorDict rather than writing the ``out_keys`` into the input
        TensorDict. Use the return value; do not rely on in-place updates of
        the input.

    Args:
        client (Callable or InferenceTransport): actor-side inference client.
            If a transport is provided, ``transport.client()`` is called.

    Keyword Args:
        in_keys (sequence of NestedKey, optional): input keys advertised by the
            module. The full input TensorDict is still sent to the server.
        out_keys (sequence of NestedKey, optional): output keys advertised by
            the module.

    Examples:
        >>> import torch
        >>> import torch.nn as nn
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.inference_server import (
        ...     InferenceServer,
        ...     PolicyClientModule,
        ...     ThreadingTransport,
        ... )
        >>> policy = TensorDictModule(
        ...     nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"]
        ... )
        >>> transport = ThreadingTransport()
        >>> server = InferenceServer(policy, transport).start()
        >>> remote_policy = PolicyClientModule(
        ...     transport, in_keys=["observation"], out_keys=["action"]
        ... )
        >>> td = remote_policy(TensorDict({"observation": torch.randn(4)}))
        >>> "action" in td.keys()
        True
        >>> server.shutdown()
    """

    def __init__(
        self,
        client: Callable[[TensorDictBase], TensorDictBase] | InferenceTransport,
        *,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(client, InferenceTransport):
            client = client.client()
        self.client = client
        self.in_keys = list(in_keys or [])
        self.out_keys = list(out_keys or [])

    def submit(self, tensordict: TensorDictBase) -> Future | _ImmediateFuture:
        """Submit a TensorDict request and return a future-like object.

        Args:
            tensordict (TensorDictBase): observation TensorDict to send to the
                remote policy.

        Returns:
            Future-like object whose ``result()`` method returns a TensorDict.
            When the wrapped client exposes ``submit`` this is the transport's
            :class:`~concurrent.futures.Future` and submission errors raise
            synchronously; for a plain callable client the call runs eagerly
            and errors are deferred to ``result()`` on a reduced future that
            only implements ``done()`` and ``result()``.
        """
        submit = getattr(self.client, "submit", None)
        if submit is None:
            try:
                result = self.client(tensordict)
                return _ImmediateFuture(result)
            except Exception as exc:
                return _ImmediateFuture(exc)
        return submit(tensordict)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self.submit(tensordict).result()


RemotePolicy = PolicyClientModule
