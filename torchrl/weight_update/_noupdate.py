from __future__ import annotations

from typing import Any

from torchrl.weight_update.weight_sync_schemes import TransportBackend, WeightSyncScheme


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
        # Store model_id directly on scheme (no-op)
        self.model_id = model_id

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
        # Store model_id directly on scheme (no-op)
        self.model_id = model_id

    def create_transport(self, **kwargs) -> TransportBackend:
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

            def setup_connection_and_weights_on_sender(self) -> None:
                pass

            def setup_connection_and_weights_on_receiver(self, worker_idx: int) -> Any:
                return None

        return NoOpTransport()

    def send(
        self,
        weights: Any = None,
        worker_ids: int | list[int] | None = None,
    ) -> None:
        """No-op send - does nothing."""

    def receive(self, timeout: float = 0.001) -> bool:
        """No-op receive - always returns False."""
        return False

    def connect(self, *, worker_idx: int | None = None) -> None:
        """No-op synchronize - does nothing."""
        if self._initialized_on_sender:
            self.synchronized_on_sender = True
        elif self._initialized_on_receiver:
            self.synchronized_on_receiver = True
