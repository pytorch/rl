# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union
from typing import Tuple

import numpy as np
import torch
from torchrl.data import TensorDict, CompositeSpec, NdUnboundedContinuousTensorSpec

from ...data.utils import DEVICE_TYPING
from ...modules.tensordict_module import TensorDictModule
from .common import ModelBasedEnv
from ..common import EnvBase

class DreamerEnv(ModelBasedEnv):
    def __init__(
        self,
        world_model: TensorDictModule,
        prior_shape: Tuple[int, ...],
        belief_shape: Tuple[int, ...],
        obs_decoder: TensorDictModule = None,
        device: DEVICE_TYPING = "cpu",
        dtype: Optional[Union[torch.dtype, np.dtype]] = None,
        batch_size: Optional[torch.Size] = torch.Size([1]),
    ):
        super(DreamerEnv, self).__init__(world_model, device, dtype, batch_size)
        self.obs_decoder = obs_decoder
        self.prior_shape = prior_shape
        self.belief_shape = belief_shape
    def set_specs_from_env(self, env: EnvBase):
        """
        Sets the specs of the environment from the specs of the given environment.
        """
        super().set_specs_from_env(env)
        self.observation_spec = CompositeSpec(
            prior_state=NdUnboundedContinuousTensorSpec(shape=self.prior_shape),
            belief=NdUnboundedContinuousTensorSpec(shape=self.belief_shape),
        )
        self.input_spec = CompositeSpec(
            **self.observation_spec, action=self.action_spec
        )

    def _reset(self, tensordict=None, **kwargs) -> TensorDict:
        td = self.input_spec.rand(shape=self.batch_size)
        td = self.step(td)
        return td

    def _set_seed(self, seed: Optional[int]) -> int:
        return seed + 1

    def decode_obs(self, tensordict: TensorDict, compute_latents=False) -> TensorDict:
        if self.obs_decoder is None:
            raise ValueError("No observation decoder provided")
        if compute_latents:
            tensordict = self(tensordict)
        return self.obs_decoder(tensordict)
