# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Future

import torch
from tensordict.base import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey

from torchrl._utils import logger as torchrl_logger
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
        target_policy_version (int or Callable[[], int], optional): expected
            latest policy version used for bounded-staleness checks. A
            callable (e.g. ``lambda: server.policy_version``) is re-evaluated
            on every check, providing a live version source.
        max_policy_lag (int, optional): maximum allowed
            ``target_policy_version - returned_policy_version``. The lag is
            computed against the *oldest* version element in the result.
        policy_version_key (NestedKey or None, optional): key that contains
            the behavior policy version returned by the server. Must match the
            server's ``policy_version_key``; a mismatch triggers a one-time
            warning when the staleness guard is enabled. ``None`` disables the
            guard. Defaults to ``"policy_version"``.

    .. note::
        The server-side version counter is independent from the
        :class:`~torchrl.envs.llm.transforms.PolicyVersion` transform and the
        collectors' ``track_policy_version`` mechanism, but it uses the same
        default ``"policy_version"`` key. When combining an inference-server
        client with an env that carries the ``PolicyVersion`` transform, set
        distinct keys to avoid one writer overwriting the other.

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
        target_policy_version: int | Callable[[], int] | None = None,
        max_policy_lag: int | None = None,
        policy_version_key: NestedKey | None = "policy_version",
    ) -> None:
        super().__init__()
        if isinstance(client, InferenceTransport):
            client = client.client()
        self.client = client
        self.in_keys = list(in_keys or [])
        self.out_keys = list(out_keys or [])
        self.target_policy_version = target_policy_version
        self.max_policy_lag = max_policy_lag
        self.policy_version_key = policy_version_key
        self._warned_missing_version = False

    def _check_policy_lag(self, tensordict: TensorDictBase) -> None:
        if self.target_policy_version is None or self.max_policy_lag is None:
            return
        version = (
            tensordict.get(self.policy_version_key, default=None)
            if self.policy_version_key is not None
            else None
        )
        if version is None:
            # A user who configured the guard expects protection; a missing
            # key (server annotations disabled, or client/server key
            # mismatch) must not silently disable it.
            if not self._warned_missing_version:
                torchrl_logger.warning(
                    f"PolicyClientModule: max_policy_lag is set but the "
                    f"result carries no {self.policy_version_key!r} entry. "
                    f"The staleness guard is inactive; check that the server "
                    f"annotates versions and that policy_version_key matches "
                    f"on both sides."
                )
                self._warned_missing_version = True
            return
        if isinstance(version, torch.Tensor):
            # Bound the staleness of the worst-case (oldest) element.
            version = int(version.min().item())
        else:
            version = int(version)
        target = self.target_policy_version
        if callable(target):
            target = int(target())
        lag = target - version
        if lag > self.max_policy_lag:
            raise RuntimeError(
                f"Remote policy result is too stale: version={version}, "
                f"target_policy_version={target}, "
                f"max_policy_lag={self.max_policy_lag}."
            )

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
        result = self.submit(tensordict).result()
        self._check_policy_lag(result)
        return result


RemotePolicy = PolicyClientModule
