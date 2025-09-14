# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator

import torch
from tensordict.base import TensorCollection
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import WeightUpdaterBase
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

        torchrl_logger.info(f"=> in {type(self).__name__}.__init__")
        self.vllm_engine = vllm_engine
        self.initialized_group = None

        # Extract configuration from engine
        self.vllm_tp_size = vllm_engine.get_tp_size()
        self.master_address = vllm_engine.get_master_address()
        self.master_port = vllm_engine.get_master_port()
        self.model_metadata = vllm_engine.get_model_metadata()

        torchrl_logger.info(
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
        torchrl_logger.info("Weight update group initialized")

    def push_weights(
        self, weights: Iterator[tuple[str, torch.Tensor]] | TensorCollection
    ):
        """Push weights to the vLLM engine.

        Args:
            weights: Either an iterator of (name, tensor) pairs or a TensorCollection
        """
        if isinstance(weights, TensorCollection):
            weights = iter(weights.flatten_keys(".").items())

        if self.initialized_group is None:
            raise RuntimeError("Weight updater not initialized. Call init() first.")

        # Delegate to the engine's update_weights method
        self.vllm_engine.update_weights(weights)
        torchrl_logger.info("Weight update completed")

        # Call post-hooks to increment policy version
        self._call_post_hooks()

    def push_weights_from_transformers(self, transformers_model):
        """Push weights from a transformers model.

        Args:
            transformers_model: A transformers PreTrainedModel or TorchRL wrapper
        """
        if not _has_transformers:
            raise ImportError("transformers not available")

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

        # Convert to iterator for memory efficiency
        weights_iter = iter(state_dict.items())
        self.push_weights(weights_iter)

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
        self.register_post_hook(collector.increment_version)
        return result

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
