# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc

import importlib.util

import torch
import torch.cuda
import torch.distributed

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase

from torchrl._utils import logger as torchrl_logger

from torchrl.collectors.weight_update import WeightUpdaterBase
from torchrl.modules.llm.backends import stateless_init_process_group

_has_vllm = importlib.util.find_spec("vllm") is not None

_has_ray = importlib.util.find_spec("ray") is not None


class vLLMUpdaterMeta(abc.ABCMeta):
    """Metaclass for vLLMUpdater that allows switching between V1 and V2 implementations.

    When instantiating vLLMUpdater with v2=True, returns a vLLMUpdaterV2 instance instead.
    This provides a unified entry point for both updater versions while maintaining
    backward compatibility.
    """

    def __call__(cls, *args, v2=False, **kwargs):
        if v2:
            # Import V2 here to avoid circular imports
            from .vllm_v2 import vLLMUpdaterV2

            # V2 has a different signature - it expects a vllm_engine parameter
            # If the user is providing the old signature, we need to handle this gracefully
            if args or any(
                k in kwargs
                for k in [
                    "master_address",
                    "master_port",
                    "model_metadata",
                    "vllm_tp_size",
                ]
            ):
                # Old signature detected - we can't auto-convert, user needs to update their code
                raise TypeError(
                    "When using v2=True, you must provide a vllm_engine parameter instead of "
                    "the v1 parameters (master_address, master_port, model_metadata, vllm_tp_size). "
                    "See vLLMUpdaterV2 documentation for details."
                )

            # Forward to V2 constructor
            return vLLMUpdaterV2(*args, **kwargs)
        else:
            # Use original V1 constructor
            return super().__call__(*args, **kwargs)


class vLLMUpdater(WeightUpdaterBase, metaclass=vLLMUpdaterMeta):
    """A class that sends weights to vLLM workers.

    This class handles synchronizing weights between a training policy and vLLM inference workers.
    It supports both local vLLM instances and remote Ray actors.

    Args:
        master_address (str, optional): The master address for distributed training. Defaults to localhost.
        master_port (int, optional): The master port for distributed training. If None, will auto-assign.
        model_metadata (dict[str, tuple[torch.dtype, torch.Size]], optional): Model metadata mapping
            parameter names to their dtype and shape. If not provided, will be extracted from policy.
        vllm_tp_size (int, optional): vLLM tensor parallel size. Defaults to 1.
        v2 (bool, optional): If True, returns a vLLMUpdaterV2 instance instead. This is an experimental
            feature that provides better integration with AsyncVLLM engines. When using v2=True, you must
            provide a vllm_engine parameter instead of the above parameters. Defaults to False.

    Methods:
        init: Initialize the updater with model metadata and initialize the group.
        _sync_weights_with_worker: Synchronize weights with a vLLM worker.
        _get_server_weights: Not used - weights must be passed directly.
        _maybe_map_weights: No mapping needed.
        all_worker_ids: Returns [0] since we only have one worker.

    .. note::
        This class assumes the policy is a transformers model that can be loaded by vLLM.
        The policy must have a state_dict() method that returns the model weights.

    .. warning::
        The v2=True option is experimental and may have backward-compatibility breaking changes
        in future releases. However, it is generally considered a better option for working with
        AsyncVLLM engines and provides improved performance and reliability.
    """

    def __init__(
        self,
        master_address: str | None = None,
        master_port: int | None = None,
        model_metadata: dict[str, tuple[torch.dtype, torch.Size]] | None = None,
        vllm_tp_size: int | None = None,
    ):
        torchrl_logger.debug(f"=> in {type(self).__name__}.__init__")
        self.master_address = master_address
        self.master_port = master_port
        self.model_metadata = model_metadata
        self.initialized_group = None
        if vllm_tp_size is None:
            vllm_tp_size = 1
        self.vllm_tp_size = vllm_tp_size
        self._model_ref = None

    def init(self, model_metadata: dict[str, tuple[torch.dtype, torch.Size]]) -> None:
        """Initialize the updater with model metadata and initialize the group.

        Args:
            model_metadata (dict[str, tuple[torch.dtype, torch.Size]]): The model metadata mapping
                parameter names to their dtype and shape.
        """
        self.model_metadata = model_metadata
        self.maybe_init_group()

    @property
    def master_address(self):
        if self._master_address is None:
            self._master_address = "localhost"  # get_ip()
        return self._master_address

    @master_address.setter
    def master_address(self, value):
        self._master_address = value

    @property
    def master_port(self):
        if self._master_port is None:
            try:
                from vllm.utils import get_open_port
                self._master_port = get_open_port()
            except ImportError:
                # Fallback if vLLM get_open_port not available
                import socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", 0))
                    self._master_port = s.getsockname()[1]
        return self._master_port

    @master_port.setter
    def master_port(self, value):
        self._master_port = value

    def _get_model_ref(self):
        """Get a reference to the model actor, either directly or through Ray.

        Returns:
            The model reference that can be used for weight updates
        """
        if self._model_ref is not None:
            return self._model_ref

        if hasattr(self.collector, "_collector"):
            # We're dealing with a RayLLMCollector
            import ray

            # Get direct reference to the model actor
            self._model_ref = ray.get(
                self.collector._collector.get_policy_model.remote()
            )
        else:
            # We're dealing with a local collector
            self._model_ref = self.collector.policy.model

        return self._model_ref

    def _init_group(self):
        import ray

        torchrl_logger.debug(f"=> in {type(self).__name__}._init_group")
        weight_sync_world_size = self.vllm_tp_size + 1
        torchrl_logger.debug(f"initializing group with {weight_sync_world_size=}...")
        torchrl_logger.debug(f"vllm_tp_size={self.vllm_tp_size}")

        model_ref = self._get_model_ref()

        torchrl_logger.debug(f"model_ref: {model_ref}")
        # Initialize the weight update group
        torchrl_logger.debug("Calling init_weight_update_group...")
        init_weight_update_group_getter = model_ref.collective_rpc.remote(
            "init_weight_update_group",
            args=(self.master_address, self.master_port, 1, weight_sync_world_size),
        )
        torchrl_logger.debug("init_weight_update_group remote call succeeded")

        torchrl_logger.debug("Calling stateless_init_process_group within updater...")
        self.vllm_comm_group = stateless_init_process_group(
            self.master_address,
            self.master_port,
            0,
            weight_sync_world_size,
            torch.device("cuda:0"),
        )

        ray.get(init_weight_update_group_getter)
        torchrl_logger.debug("init_weight_update_group getter succeeded")

        torchrl_logger.debug("group initialized")
        self.initialized_group = True

    def maybe_init_group(self):
        if self.initialized_group is None and self.collector is not None:
            self._init_group()

    def _sync_weights_with_worker(
        self,
        *,
        worker_id: int | torch.device | None = None,
        server_weights: TensorDictBase | TensorDictModuleBase | dict | None = None,
    ) -> None:
        """Synchronize weights with a vLLM worker.

        Args:
            worker_id: Not used - we only have one worker.
            server_weights: The weights to sync. Can be:
                - TensorDictModuleBase: A policy module whose weights will be extracted
                - TensorDictBase: A TensorDict containing weights
                - dict: A regular dict containing weights
                - None: Will try to get weights from server using _get_server_weights()
        """
        if server_weights is None:
            raise ValueError("server_weights cannot be None for vLLM updater")

        if self.initialized_group is None:
            raise RuntimeError(
                "Failed to update weights because sender is not initialized."
            )
        if self.model_metadata is None:
            raise RuntimeError(
                "Failed to update weights because model metadata is not set. "
                "In async mode, you must set the model metadata in the training actor "
                "before any weight updates."
            )

        model_ref = self._get_model_ref()

        # First broadcast metadata
        torchrl_logger.debug("broadcasting with update_weight_broadcast")
        remotes = []
        for k, (dtype, shape) in self.model_metadata.items():
            remotes.append(
                model_ref.collective_rpc.remote(
                    "update_weight_broadcast", args=(k, dtype, shape)
                )
            )

        # # Then update weights
        # remotes = []
        # pbar = tqdm.tqdm(server_weights.items(), desc="Updating weights", total=len(server_weights))
        # for k, val in pbar:
        #     pbar.set_description(f"Updating {k}")
        #     remotes.append(model_ref.collective_rpc.remote("update_weight", args=(k, val)))
        # # ray.get(remotes)

        # if self.vllm_comm_group is not True:
        torchrl_logger.debug("broadcasting...")
        for k in self.model_metadata:
            val = server_weights[k].to(torch.device("cuda:0"))
            self.vllm_comm_group.broadcast(
                val,
                src=0,
                stream=torch.cuda.current_stream(),
            )
            del val
        import ray

        ray.get(remotes)
        torchrl_logger.debug("done broadcasting")
        torch.cuda.synchronize()

    def _get_server_weights(self) -> TensorDictBase | None:
        """Not used - weights must be passed directly via policy."""
        return None

    def _maybe_map_weights(
        self, server_weights: TensorDictBase | TensorDictModuleBase | dict
    ) -> TensorDictBase:
        """Map weights from any format to the format expected by vLLM.

        Args:
            server_weights: The weights to map. Can be:
                - TensorDictModuleBase: A policy module whose weights will be extracted
                - TensorDictBase: A TensorDict containing weights
                - dict: A regular dict containing weights

        Returns:
            TensorDictBase: The mapped weights in TensorDict format
        """
        if isinstance(server_weights, TensorDictModuleBase):
            # Extract weights from policy module using merge_and_unload for LLMs
            if not hasattr(server_weights, "model"):
                raise ValueError("TensorDictModuleBase must have a 'model' attribute")
            # Check if it's a LoRA model
            if hasattr(server_weights.model, "merge_and_unload"):
                state_dict = server_weights.model.merge_and_unload().state_dict()
            else:
                state_dict = server_weights.model.state_dict()
            return TensorDict(state_dict, [])
        elif isinstance(server_weights, TensorDictBase):
            return server_weights
        elif isinstance(server_weights, dict):
            return TensorDict(server_weights, [])
        else:
            raise TypeError(
                f"server_weights must be TensorDictModuleBase, TensorDictBase or dict, got {type(server_weights)}"
            )

    @classmethod
    def get_model_metadata(
        cls, model: TensorDictModuleBase
    ) -> dict[str, tuple[torch.dtype, torch.Size]]:
        """Get the model metadata from a model.

        Args:
            model (TensorDictModuleBase): The model to get the metadata from.
                Must be a TransformersWrapper or equivalent.

        Returns:
            dict[str, tuple[torch.dtype, torch.Size]]: The model metadata.
        """
        # Check if the model has a LoRA adapter
        if hasattr(model.model, "merge_and_unload"):
            sd = model.model.merge_and_unload().state_dict()
        else:
            sd = model.model.state_dict()
        model_metadata = {k: (v.dtype, v.shape) for k, v in sd.items()}
        return model_metadata

    def all_worker_ids(self) -> list[int]:
        """Returns [0] since we only have one worker."""
        return [0]

    def register_collector(self, collector: BaseCollector):  # noqa: F821
        result = super().register_collector(collector)
        self.register_post_hook(collector.increment_version)
        return result
