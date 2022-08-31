# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union, List

import numpy as np
import torch

from torchrl.data import TensorDict
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.model_based import ModelBasedEnv
from torchrl.modules.tensordict_module import TensorDictModule

__all__ = ["ModelBasedEnv"]


class MBPOEnv(ModelBasedEnv):
    def __init__(
        self,
        world_model: TensorDictModule,
        params: Optional[List[torch.Tensor]],
        buffers: Optional[List[torch.Tensor]],
        num_networks: int,
        device: DEVICE_TYPING = "cpu",
        dtype: Optional[Union[torch.dtype, np.dtype]] = None,
        batch_size: Optional[torch.Size] = None,
    ):
        super(MBPOEnv, self).__init__(
            world_model=world_model,
            params=params,
            buffers=buffers,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )
        self.num_networks = num_networks

    def _step(self, tensordict: TensorDict) -> TensorDict:
        tensordict_out = tensordict.clone()
        # Compute world state
        sampled_model_id = torch.randint(0, self.num_networks, tensordict_out.shape)
        tensordict_out = self.world_model(
            tensordict_out,
            params=self.world_model_params,
            buffers=self.world_model_buffers,
            vmap=True,
        )
        tensordict_out = tensordict_out[
            sampled_model_id, torch.arange(tensordict_out.shape[1])
        ]
        # Step requires a done flag. No sense for MBRL so we set it to False
        if "done" not in self.world_model.out_keys:
            tensordict_out["done"] = torch.zeros(tensordict_out.shape, dtype=torch.bool)
        return tensordict_out

    def _reset(self, tensordict=None, **kwargs) -> TensorDict:
        td = self.observation_spec.rand(shape=self.batch_size)
        td["action"] = self.action_spec.rand(shape=self.batch_size)
        td = self.step(td)
        return td

    def _set_seed(self, seed: Optional[int]) -> int:
        return seed
