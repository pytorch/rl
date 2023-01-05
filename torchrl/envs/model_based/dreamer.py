# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Union

import numpy as np
import torch
from tensordict import TensorDict

from torchrl.data import CompositeSpec
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs import EnvBase
from torchrl.envs.model_based import ModelBasedEnvBase
from torchrl.modules.tensordict_module import SafeModule


class DreamerEnv(ModelBasedEnvBase):
    """Dreamer simulation environment."""

    def __init__(
        self,
        world_model: SafeModule,
        prior_shape: Tuple[int, ...],
        belief_shape: Tuple[int, ...],
        obs_decoder: SafeModule = None,
        device: DEVICE_TYPING = "cpu",
        dtype: Optional[Union[torch.dtype, np.dtype]] = None,
        batch_size: Optional[torch.Size] = None,
    ):
        super(DreamerEnv, self).__init__(
            world_model, device=device, dtype=dtype, batch_size=batch_size
        )
        self.obs_decoder = obs_decoder
        self.prior_shape = prior_shape
        self.belief_shape = belief_shape

    def set_specs_from_env(self, env: EnvBase):
        """Sets the specs of the environment from the specs of the given environment."""
        super().set_specs_from_env(env)
        # self.observation_spec = CompositeSpec(
        #     next_state=UnboundedContinuousTensorSpec(
        #         shape=self.prior_shape, device=self.device
        #     ),
        #     next_belief=UnboundedContinuousTensorSpec(
        #         shape=self.belief_shape, device=self.device
        #     ),
        # )
        self.input_spec = CompositeSpec(
            state=self.observation_spec["state"],
            belief=self.observation_spec["belief"],
            action=self.action_spec.to(self.device),
        )

    def _reset(self, tensordict=None, **kwargs) -> TensorDict:
        batch_size = tensordict.batch_size if tensordict is not None else []
        device = tensordict.device if tensordict is not None else self.device
        td = self.input_spec.rand(shape=batch_size).to(device)
        td["reward"] = self.reward_spec.rand(shape=batch_size).to(device)
        td.update(self.observation_spec.rand(shape=batch_size).to(device))
        return td

    def decode_obs(self, tensordict: TensorDict, compute_latents=False) -> TensorDict:
        if self.obs_decoder is None:
            raise ValueError("No observation decoder provided")
        if compute_latents:
            tensordict = self.world_model(tensordict)
        return self.obs_decoder(tensordict)
