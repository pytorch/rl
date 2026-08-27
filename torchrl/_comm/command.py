# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torchrl._comm.mailbox import Mailbox, MailboxClient


@dataclass(frozen=True)
class CommandRequest:
    """A command received by the service side of a command channel."""

    verb: str
    payload: Any
    callback: tuple[int, int]


class CommandClient:
    """Picklable caller-side handle for a :class:`CommandChannel`."""

    def __init__(self, mailbox_client: MailboxClient) -> None:
        self._mailbox_client = mailbox_client

    def call(
        self,
        verb: str,
        payload: Any = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        """Execute a command and wait for its reply."""
        return self._mailbox_client(
            {"verb": verb, "payload": {} if payload is None else payload},
            timeout=timeout,
        )


class CommandChannel:
    """Ordered duplex command channel built on a :class:`Mailbox`."""

    def __init__(self, mailbox: Mailbox) -> None:
        self._mailbox = mailbox
        self._closed = False

    def client(self) -> CommandClient:
        """Create a restricted command client."""
        if self._closed:
            raise RuntimeError("The command channel is closed.")
        return CommandClient(self._mailbox.client())

    def receive(self, timeout: float | None = None) -> CommandRequest | None:
        """Receive the next command, returning ``None`` on timeout."""
        if self._closed:
            return None
        self._mailbox.wait_for_work(0.0 if timeout is None else timeout)
        payloads, callbacks, _ = self._mailbox.drain(1)
        if not payloads:
            return None
        payload = payloads[0]
        return CommandRequest(
            verb=payload["verb"],
            payload=payload.get("payload", {}),
            callback=callbacks[0],
        )

    def resolve(self, request: CommandRequest, result: Any) -> None:
        """Return a successful command result."""
        self._mailbox.resolve(request.callback, result)

    def reject(self, request: CommandRequest, error: BaseException) -> None:
        """Return a failed command result."""
        self._mailbox.reject(request.callback, error)

    def close(self, error: BaseException | None = None) -> None:
        """Reject queued commands and prevent creation of new clients."""
        if self._closed:
            return
        self._closed = True
        if error is None:
            error = RuntimeError("The command channel is closed.")
        while True:
            payloads, callbacks, _ = self._mailbox.drain(64)
            for callback in callbacks:
                self._mailbox.reject(callback, error)
            if len(payloads) < 64:
                return
