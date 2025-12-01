# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import time

from collections.abc import Iterator

import torch
from tensordict import TensorDictBase
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors.weight_update import WeightUpdaterBase
from torchrl.modules.llm.backends.vllm import RLvLLMEngine

try:
    pass

    _has_transformers = True
except ImportError:
    _has_transformers = False


class vLLMUpdaterV2(WeightUpdaterBase):
    """Simplified vLLM weight updater using the RLvLLMEngine interface.

    This updater works with any vLLM engine that implements the RLvLLMEngine
    interface, automatically extracting configuration and handling weight updates
    through the engine's own methods.

    Args:
        vllm_engine: A vLLM engine implementing the RLvLLMEngine interface.

    .. note:: This class can be created through :class:`torchrl.collectors.llm.vLLMUpdater` with `v2=True`.

    """

    def __init__(self, vllm_engine: RLvLLMEngine):
        # Check that vllm_engine implements the RLvLLMEngine interface
        if not isinstance(vllm_engine, RLvLLMEngine):
            raise TypeError(
                f"vllm_engine must implement RLvLLMEngine interface, got {type(vllm_engine)}"
            )

        torchrl_logger.debug(f"=> in {type(self).__name__}.__init__")
        self.vllm_engine = vllm_engine
        self.initialized_group = None

        # Extract configuration from engine
        self.vllm_tp_size = vllm_engine.get_tp_size()
        self.master_address = vllm_engine.get_master_address()
        self.master_port = vllm_engine.get_master_port()
        self.model_metadata = vllm_engine.get_model_metadata()

        torchrl_logger.debug(
            f"Initialized vLLMUpdaterV2 with tp_size={self.vllm_tp_size}"
        )

    def get_tp_size(self) -> int:
        """Get the tensor parallel size."""
        return self.vllm_tp_size

    def init(
        self, model_metadata: dict[str, tuple[torch.dtype, torch.Size]] | None = None
    ) -> None:
        """Initialize the weight updater.

        Args:
            model_metadata: Optional model metadata. If not provided, uses engine's metadata.
        """
        if model_metadata is not None:
            self.model_metadata = model_metadata

        # Initialize the engine's weight update group
        self.vllm_engine.init_weight_update_group()
        self.initialized_group = True
        torchrl_logger.debug("Weight update group initialized")

    def push_weights(
        self, weights: Iterator[tuple[str, torch.Tensor]] | TensorDictBase
    ):
        """Push weights to the vLLM engine.

        Args:
            weights: Either an iterator of (name, tensor) pairs or a TensorDictBase
        """
        if isinstance(weights, TensorDictBase):
            weights = iter(weights.flatten_keys(".").items())

        if self.initialized_group is None:
            raise RuntimeError("Weight updater not initialized. Call init() first.")

        # Delegate to the engine's update_weights method
        self.vllm_engine.update_weights(weights)
        torchrl_logger.debug("Weight update completed")

        # Call post-hooks to increment policy version
        torchrl_logger.debug("Calling post-hooks...")
        self._call_post_hooks()
        torchrl_logger.debug("Post-hooks completed")

    def push_weights_from_transformers(self, transformers_model):
        """Push weights from a transformers model.

        Args:
            transformers_model: A transformers PreTrainedModel or TorchRL wrapper
        """
        if not _has_transformers:
            raise ImportError("transformers not available")
        t0 = time.time()
        # Extract state dict from model, handling LoRA models properly
        if hasattr(transformers_model, "model") and hasattr(
            transformers_model.model, "state_dict"
        ):
            # TorchRL wrapper (e.g., TransformersWrapper)
            model = transformers_model.model
            # Check if it's a LoRA model
            if hasattr(model, "merge_and_unload"):
                state_dict = model.merge_and_unload().state_dict()
            else:
                state_dict = model.state_dict()
        elif hasattr(transformers_model, "state_dict"):
            # Direct transformers model
            # Check if it's a LoRA model
            if hasattr(transformers_model, "merge_and_unload"):
                state_dict = transformers_model.merge_and_unload().state_dict()
            else:
                state_dict = transformers_model.state_dict()
        else:
            raise TypeError(
                f"Cannot extract state_dict from {type(transformers_model)}"
            )

        t1 = time.time()
        torchrl_logger.debug(f"Time to extract state_dict: {t1 - t0}")
        # Convert to iterator for memory efficiency
        weights_iter = iter(state_dict.items())
        self.push_weights(weights_iter)
        torchrl_logger.debug(f"Time to push weights: {time.time() - t1}")

    def push_weights_from_transformers_optimized(
        self, transformers_model, batch_size=50
    ):
        """Optimized version of push_weights_from_transformers with GPU pre-loading.

        This method provides several optimizations:
        1. Pre-loads all weights to GPU before transfer
        2. Optionally batches weights for better memory management
        3. Uses non-blocking transfers when possible

        Args:
            transformers_model: A transformers PreTrainedModel or TorchRL wrapper
            batch_size: Number of weights to transfer in each batch (0 = no batching)
        """
        if not _has_transformers:
            raise ImportError("transformers not available")

        t0 = time.time()

        # Extract state dict from model, handling LoRA models properly
        if hasattr(transformers_model, "model") and hasattr(
            transformers_model.model, "state_dict"
        ):
            # TorchRL wrapper (e.g., TransformersWrapper)
            model = transformers_model.model
            if hasattr(model, "merge_and_unload"):
                state_dict = model.merge_and_unload().state_dict()
            else:
                state_dict = model.state_dict()
        elif hasattr(transformers_model, "state_dict"):
            # Direct transformers model
            if hasattr(transformers_model, "merge_and_unload"):
                state_dict = transformers_model.merge_and_unload().state_dict()
            else:
                state_dict = transformers_model.state_dict()
        else:
            raise TypeError(
                f"Cannot extract state_dict from {type(transformers_model)}"
            )

        t1 = time.time()
        torchrl_logger.debug(f"Time to extract state_dict: {t1 - t0:.3f}s")

        # Pre-load all weights to GPU for faster transfer
        gpu_weights = {}
        with torch.device("cuda:0"):  # Ensure we're using the right GPU
            for name, weight in state_dict.items():
                if not weight.is_cuda:
                    gpu_weights[name] = weight.cuda(non_blocking=True)
                else:
                    gpu_weights[name] = weight

        # Synchronize to ensure all transfers are complete
        torch.cuda.synchronize()
        t2 = time.time()
        torchrl_logger.debug(f"Time to move weights to GPU: {t2 - t1:.3f}s")

        # Transfer weights (optionally in batches)
        if batch_size > 0:
            weight_items = list(gpu_weights.items())
            for i in range(0, len(weight_items), batch_size):
                batch = weight_items[i : i + batch_size]
                self.push_weights(iter(batch))
                torchrl_logger.debug(
                    f"Transferred batch {i // batch_size + 1}/{(len(weight_items) + batch_size - 1) // batch_size}"
                )
        else:
            # Transfer all at once
            self.push_weights(iter(gpu_weights.items()))

        t3 = time.time()
        torchrl_logger.debug(
            f"Time to push weights: {t3 - t2:.3f}s, total time: {t3 - t0:.3f}s"
        )

    # Required WeightUpdaterBase methods
    def _sync_weights_with_worker(self, *, worker_id=None, server_weights=None):
        """Sync weights with worker (delegates to push_weights)."""
        if server_weights is None:
            raise ValueError("server_weights cannot be None")

        if hasattr(server_weights, "items"):
            # Dict-like object
            self.push_weights(iter(server_weights.items()))
        else:
            # Assume it's a model with state_dict
            self.push_weights_from_transformers(server_weights)

    def _get_server_weights(self):
        """Not used - weights must be passed directly."""
        return None

    def _maybe_map_weights(self, server_weights):
        """Map weights to expected format."""
        return server_weights  # No mapping needed, handled in push_weights methods

    def all_worker_ids(self):
        """Return list of worker IDs."""
        return [0]

    def register_collector(self, collector):  # noqa: F821
        """Register a collector and set up policy version increment post-hook.

        Args:
            collector: The collector to register (DataCollectorBase)
        """
        result = super().register_collector(collector)

        # Only register the increment_version post-hook once for the first collector
        # This avoids N^2 complexity where each weight update calls increment_version
        # on all collectors N times (once per registered collector)
        if len(self.post_hooks) == 0:
            torchrl_logger.debug("Registering policy version increment post-hook")
            self.register_post_hook(self._increment_all_collector_versions)

        return result

    def _increment_all_collector_versions(self):
        """Increment version for all registered collectors efficiently."""
        torchrl_logger.debug(
            f"Incrementing policy version for {len(self.collectors)} collectors..."
        )
        for i, collector in enumerate(self.collectors):
            try:
                collector.increment_version()
                torchrl_logger.debug(
                    f"Incremented version for collector {i + 1}/{len(self.collectors)}"
                )
            except Exception as e:
                torchrl_logger.warning(
                    f"Failed to increment version for collector {i + 1}: {e}"
                )
        torchrl_logger.debug("All collector versions incremented")

    @classmethod
    def get_model_metadata(cls, model) -> dict[str, tuple[torch.dtype, torch.Size]]:
        """Get model metadata from a model.

        Args:
            model: A model with state_dict() method (e.g., TransformersWrapper)

        Returns:
            dict: Mapping of parameter names to (dtype, shape) tuples
        """
        if hasattr(model, "model") and hasattr(model.model, "state_dict"):
            # TorchRL wrapper (e.g., TransformersWrapper)
            model_obj = model.model
            # Check if it's a LoRA model
            if hasattr(model_obj, "merge_and_unload"):
                sd = model_obj.merge_and_unload().state_dict()
            else:
                sd = model_obj.state_dict()
        elif hasattr(model, "state_dict"):
            # Direct model
            # Check if it's a LoRA model
            if hasattr(model, "merge_and_unload"):
                sd = model.merge_and_unload().state_dict()
            else:
                sd = model.state_dict()
        else:
            raise TypeError(f"Cannot extract state_dict from {type(model)}")

        return {k: (v.dtype, v.shape) for k, v in sd.items()}

    # Remove the weakrefs from the updater for serialization
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_collector_wrs"] = None
        return state
