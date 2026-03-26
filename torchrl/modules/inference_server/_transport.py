# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
from concurrent.futures import Future

from tensordict.base import TensorDictBase


class InferenceTransport(abc.ABC):
    """Abstract base class for inference server transport backends.

    A transport handles the communication between actor-side clients and the
    server-side inference loop. Concrete implementations provide the mechanism
    for submitting requests, draining batches, and routing results back.

    Subclasses must implement :meth:`submit`, :meth:`drain`, :meth:`wait_for_work`,
    and :meth:`resolve`.
    """

    @abc.abstractmethod
    def submit(self, td: TensorDictBase) -> Future[TensorDictBase]:
        """Submit a single inference request.

        Called on the actor side. Returns a :class:`~concurrent.futures.Future`
        (or future-like object) that will be resolved with the inference result.

        Args:
            td (TensorDictBase): a single (unbatched) input tensordict.

        Returns:
            A Future that resolves to the output TensorDictBase.
        """
        ...

    @abc.abstractmethod
    def drain(self, max_items: int) -> tuple[list[TensorDictBase], list]:
        """Drain up to *max_items* pending requests from the queue.

        Called on the server side. Returns a pair ``(inputs, callbacks)`` where
        ``inputs`` is a list of TensorDicts and ``callbacks`` is a list of
        opaque objects that :meth:`resolve` knows how to fulfil.

        Args:
            max_items (int): maximum number of items to dequeue.

        Returns:
            Tuple of (inputs, callbacks).
        """
        ...

    @abc.abstractmethod
    def wait_for_work(self, timeout: float) -> None:
        """Block until new work is available or *timeout* seconds elapse.

        Called on the server side before :meth:`drain`.

        Args:
            timeout (float): maximum seconds to wait.
        """
        ...

    @abc.abstractmethod
    def resolve(self, callback, result: TensorDictBase) -> None:
        """Send a result back to the actor that submitted the request.

        Args:
            callback: an opaque handle returned by :meth:`drain`.
            result (TensorDictBase): the inference output for this request.
        """
        ...

    @abc.abstractmethod
    def resolve_exception(self, callback, exc: BaseException) -> None:
        """Propagate an exception back to the actor that submitted the request.

        Args:
            callback: an opaque handle returned by :meth:`drain`.
            exc (BaseException): the exception to propagate.
        """
        ...

    def client(self) -> InferenceClient:  # noqa: F821
        """Return an actor-side :class:`InferenceClient` bound to this transport."""
        from torchrl.modules.inference_server._server import InferenceClient

        return InferenceClient(self)
