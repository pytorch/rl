# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Helper classes for Ray-based weight synchronization tests.

This module contains Ray actor classes that need to be importable by Ray workers.
These classes are used in test_updaters.py but must be defined at module level
so Ray can serialize and import them on remote workers.
"""

import torch
from torchrl._utils import logger


class WorkerVLLMNCCL:
    """Ray actor for vLLM inference worker (receiver) using NCCL collective communication."""

    def __init__(
        self,
        scheme_config: dict,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        trainer_actor_name: str = "Trainer",
    ):
        pass

        # Store config for deferred initialization
        self.scheme_config = scheme_config
        self.model_name = model_name
        self.trainer_actor_name = trainer_actor_name
        self.wrapper = None
        self.engine = None
        self.receiver = None
        self.scheme = None
        self.trainer = None
        self.model_metadata = None

    def setup(self):
        """Set up vLLM engine (deferred from __init__ to avoid blocking)."""
        from torchrl.modules.llm.backends import AsyncVLLM
        from torchrl.modules.llm.policies import vLLMWrapper

        # Create vLLM wrapper
        async_engine = AsyncVLLM.from_pretrained(
            self.model_name,
            num_replicas=2,  # Number of engine replicas
        )
        self.wrapper = vLLMWrapper(async_engine, input_mode="history")
        self.engine = self.wrapper.model

        # Create scheme from config
        from torchrl.weight_update.llm.vllm_nccl import VLLMWeightSyncScheme

        self.scheme = VLLMWeightSyncScheme(**self.scheme_config)

        # Create receiver (engine handles rank assignment automatically)
        self.receiver = self.scheme.create_receiver(self.engine)
        return "setup_complete"

    def init_metadata(self):
        """Initialize the receiver by fetching metadata from trainer."""
        import ray

        if self.receiver is None:
            raise RuntimeError("Must call setup() before init()")

        # Get trainer actor by name
        logger.info(f"Getting trainer actor by name {self.trainer_actor_name}")
        self.trainer = ray.get_actor(self.trainer_actor_name)

        # Fetch model metadata from trainer
        logger.info("Fetching model metadata from trainer (requires max_concurrency>1)")
        self.model_metadata = ray.get(self.trainer.get_model_metadata.remote())

    def init(self):
        if self.model_metadata is None:
            raise RuntimeError("Must call init_metadata() before init()")

        # Initialize receiver with metadata
        logger.info("Initializing receiver...")
        self.receiver.init_all_workers_group(self.model_metadata)
        self.initialized = True
        logger.info("Receiver initialized")
        return "initialized"

    def get_engine(self):
        """Get the vLLM engine reference for RPC coordination."""
        if self.engine is None:
            raise RuntimeError("Must call setup() first")
        return self.engine

    def get_sample_output(self):
        """Get a sample output to verify model works."""
        # Simple inference test
        return "vllm_ready"

    @classmethod
    def as_remote(cls, *args, **kwargs):
        import ray

        # No GPUs needed for the actor itself - vLLM workers manage their own placement group (2 GPUs)
        # AsyncVLLM service doesn't act as NCCL rank 0 when used with external trainer
        return ray.remote(num_cpus=4, num_gpus=0, max_concurrency=4)(cls)


class WorkerTransformerNCCL:
    """Ray actor for transformer trainer (sender) using NCCL collective communication."""

    def __init__(self, scheme_config: dict, model_name: str = "Qwen/Qwen2.5-0.5B"):
        from torchrl.weight_update.llm.vllm_nccl import (
            get_model_metadata,
            VLLMWeightSyncScheme,
        )
        from transformers import AutoModelForCausalLM

        # Create transformer model
        transformer = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
        )
        self.transformer = transformer.cuda()

        # Create scheme from config
        self.scheme = VLLMWeightSyncScheme(**scheme_config)

        # Create sender
        self.sender = self.scheme.create_sender()
        self.sender.register_model(self.transformer)

        # Extract and store model metadata
        self.model_metadata = get_model_metadata(self.transformer)

    def init(self, vllm_engine=None):
        """Initialize sender with optional vLLM engine for RPC coordination.

        Args:
            vllm_engine: Optional vLLM engine reference for calling collective_rpc
        """
        if self.model_metadata is None:
            raise RuntimeError("Must call init_metadata() before init()")

        self.sender.init_all_workers_group(self.model_metadata, vllm_engine=vllm_engine)
        self.initialized = True
        logger.info("Trainer initialized")
        return "initialized"

    def get_model_metadata(self):
        """Get model metadata to share with receiver."""
        return self.model_metadata

    def update_weights(self, modify_weights: bool = False):
        """Trigger a weight update broadcast.

        Args:
            modify_weights: If True, modifies weights before broadcasting
                            for verification purposes.

        Returns:
            str: "updated" status message
        """

        # Optionally modify weights for testing
        if modify_weights:
            with torch.no_grad():
                first_param = next(self.transformer.parameters())
                first_param.add_(0.01)

        # Broadcast weights to all vLLM workers
        self.sender.update_weights()
        return "updated"

    def get_first_param_sum(self):
        """Get sum of first parameter for verification."""
        return next(self.transformer.parameters()).sum().item()

    @classmethod
    def as_remote(cls, *args, **kwargs):
        import ray

        return ray.remote(num_cpus=4, num_gpus=1, max_concurrency=4)(cls)


class WorkerVLLMDoubleBuffer:
    """Ray actor for vLLM inference worker (receiver) using double-buffered storage."""

    def __init__(self, scheme_config: dict, model_name: str = "Qwen/Qwen2.5-0.5B"):
        # Store config for deferred initialization
        self.scheme_config = scheme_config
        self.model_name = model_name
        self.wrapper = None
        self.engine = None
        self.receiver = None
        self.scheme = None

    def setup(self):
        """Set up vLLM engine and receiver."""
        from torchrl.modules.llm.backends import AsyncVLLM
        from torchrl.modules.llm.policies import vLLMWrapper

        # Create vLLM wrapper
        async_engine = AsyncVLLM.from_pretrained(
            self.model_name,
            num_replicas=1,  # Single replica for simplicity
        )
        self.wrapper = vLLMWrapper(async_engine, input_mode="history")
        self.engine = self.wrapper.model

        # Create scheme from config
        from torchrl.weight_update.llm.vllm_double_buffer import (
            VLLMDoubleBufferSyncScheme,
        )

        self.scheme = VLLMDoubleBufferSyncScheme(**self.scheme_config)

        # Create receiver
        self.receiver = self.scheme.create_receiver(self.engine)
        logger.info("Receiver setup complete")
        return "setup_complete"

    def poll_and_apply_weights(self):
        """Poll for new weights and apply them to the engine."""
        if self.receiver is None:
            raise RuntimeError("Must call setup() first")

        success = self.receiver.poll_and_apply()
        return success

    def get_sample_output(self):
        """Get a sample output to verify model works."""
        return "vllm_ready"

    @classmethod
    def as_remote(cls, *args, **kwargs):
        import ray

        # vLLM worker needs 1 GPU
        return ray.remote(num_cpus=2, num_gpus=1, max_concurrency=4)(cls)


class WorkerTransformerDoubleBuffer:
    """Ray actor for transformer trainer (sender) using double-buffered storage."""

    def __init__(self, scheme_config: dict, model_name: str = "Qwen/Qwen2.5-0.5B"):
        from torchrl.weight_update.llm.vllm_double_buffer import (
            VLLMDoubleBufferSyncScheme,
        )
        from transformers import AutoModelForCausalLM

        # Create transformer model
        transformer = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
        )
        self.transformer = transformer.cuda()

        # Create scheme from config
        self.scheme = VLLMDoubleBufferSyncScheme(**scheme_config)

        # Create sender
        self.sender = self.scheme.create_sender()
        self.sender.register_model(self.transformer)
        logger.info("Trainer setup complete")

    def update_weights(self, modify_weights: bool = False):
        """Trigger a weight update by writing to shared storage.

        Args:
            modify_weights: If True, modifies weights before writing
                            for verification purposes.

        Returns:
            str: "updated" status message
        """
        # Optionally modify weights for testing
        if modify_weights:
            with torch.no_grad():
                first_param = next(self.transformer.parameters())
                first_param.add_(0.01)

        # Write weights to shared storage
        self.sender.update_weights()
        return "updated"

    def get_first_param_sum(self):
        """Get sum of first parameter for verification."""
        return next(self.transformer.parameters()).sum().item()

    @classmethod
    def as_remote(cls, *args, **kwargs):
        import ray

        return ray.remote(num_cpus=2, num_gpus=1, max_concurrency=4)(cls)
