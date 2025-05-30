# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
import torch.cuda
import torch.distributed
from tensordict import TensorDictBase

from torchrl._utils import logger as torchrl_logger

from torchrl.collectors import WeightUpdaterBase
from torchrl.modules.llm.backends.vllm import stateless_init_process_group
from vllm.utils import get_open_port


class vLLMUpdater(WeightUpdaterBase):
    """A class that sends weights to vLLM workers.

    Assumes that the vllm instance (the inference server) is a Ray Actor.

    """

    def __init__(
        self,
        master_address,
        master_port,
        model_metadata: dict[str, tuple[torch.dtype, torch.Size]],
        vllm_tp_size: int | None = None,
    ):
        torchrl_logger.info(f"=> in {type(self).__name__}.__init__")
        self.master_address = master_address
        self.master_port = master_port
        self.model_metadata = model_metadata
        self.initialized_group = None
        if vllm_tp_size is None:
            vllm_tp_size = 1
        self.vllm_tp_size = vllm_tp_size

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
            self._master_port = get_open_port()
        return self._master_port

    @master_port.setter
    def master_port(self, value):
        self._master_port = value

    def _init_group(self):
        torchrl_logger.info(f"=> in {type(self).__name__}._init_group")
        inference_server = self.collector.policy.model
        weight_sync_world_size = 2  # (
        # inference_server.llm_engine.parallel_config.tensor_parallel_size + 1
        # )
        torchrl_logger.info(f"initializing group with {weight_sync_world_size=}...")
        import vllm

        self.vllm_comm_group = None

        if isinstance(inference_server, vllm.LLM):
            self.vllm_comm_group = True
        else:
            torchrl_logger.info("Calling init_weight_update_group on remote worker...")
            inference_server.collective_rpc.remote(
                "init_weight_update_group",
                args=(self.master_address, self.master_port, 1, weight_sync_world_size),
            )

            vllm_tp_size = self.vllm_tp_size
            weight_sync_world_size = vllm_tp_size + 1

            torchrl_logger.info(
                "Calling stateless_init_process_group within updater..."
            )
            self.vllm_comm_group = stateless_init_process_group(
                self.master_address,
                self.master_port,
                0,
                weight_sync_world_size,
                torch.device("cuda:0"),
            )

        torchrl_logger.info("group initialized")
        self.initialized_group = True

    def _get_server_weights(self) -> TensorDictBase:
        return

    def maybe_init_group(self):
        if self.initialized_group is None and self.collector is not None:
            self._init_group()

    def _sync_weights_with_worker(self, worker_id: int | torch.device, server_weights):
        inference_server = self.collector.policy.model
        if self.initialized_group is None:
            raise RuntimeError(
                "Failed to update weights because sender is not initialized."
            )
        import vllm

        if not isinstance(inference_server, vllm.LLM):
            torchrl_logger.info("broadcasting with update_weight_broadcast")
            for k, (dtype, shape) in self.model_metadata.items():
                inference_server.collective_rpc.remote(
                    "update_weight_broadcast", args=(k, dtype, shape)
                )
        else:
            for k, val in server_weights.items():
                inference_server.collective_rpc("update_weight", args=(k, val))

        if self.vllm_comm_group is not True:
            torchrl_logger.info("broadcasting...")
            for k in self.model_metadata:
                val = server_weights[k]
                self.vllm_comm_group.broadcast(
                    val, src=0, stream=torch.cuda.current_stream()
                )
            torchrl_logger.info("done broadcasting")
            # # TODO: this may fail
            # self.vllm_comm_group.broadcast(
            #     self.version_tensor, src=0, stream=torch.cuda.current_stream()
            # )
            torch.cuda.synchronize()
        return

    def _maybe_map_weights(self, server_weights):
        return

    def all_worker_ids(self) -> list[int] | list[torch.device]:
        return [0]
