from __future__ import annotations

from typing import Any

from torchrl.weight_update.weight_sync_schemes import (
    TransportBackend,
    WeightReceiver,
    WeightSender,
    WeightSyncScheme,
)


class NoWeightSyncScheme(WeightSyncScheme):
    """No-op weight synchronization scheme.

    This scheme disables weight synchronization entirely.
    """

    def _init_on_sender_impl(
        self,
        model_id: str,
        context: Any = None,
        **kwargs,
    ) -> None:
        """Initialize on the main process (sender side).

        Args:
            model_id: Identifier for the model being synchronized
            context: Optional context object (not used)
            **kwargs: Optional parameters (not used)
        """
        # Create a no-op sender
        sender = WeightSender(self)
        sender._model_id = model_id

        self._sender = sender
        self._initialized_on_sender = True

    def _init_on_receiver_impl(
        self,
        *,
        model_id: str,
        context: Any = None,
        **kwargs,
    ) -> None:
        """Initialize on worker process (receiver side).

        Args:
            model_id: Identifier for the model being synchronized
            context: Optional context object (not used)
            **kwargs: Optional parameters (not used)
        """
        # Create a no-op receiver
        receiver = WeightReceiver(self)
        receiver._model_ref = model_id

        self._receiver = receiver
        self._initialized_on_receiver = True

    def create_transport(self, pipe_or_context: Any) -> TransportBackend:
        """Create a no-op transport.

        Note:
            This is used internally by init_on_sender/init_on_receiver.
        """
        # Return a dummy transport that does nothing
        class NoOpTransport:
            def send_weights(self, weights: Any) -> None:
                pass

            def receive_weights(self, timeout: float = 1.0) -> tuple[str, Any] | None:
                return None

            def check_connection(self) -> bool:
                return True

        return NoOpTransport()
