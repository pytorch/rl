# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Union

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from torchrl.data.tensor_specs import CompositeSpec
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.common import EnvBase
from torchrl.envs.model_based import ModelBasedEnvBase
from torchrl.modules.tensordict_module.world_models import DreamerlWrapper


class DreamerEnv(ModelBasedEnvBase):
    """Dreamer simulation environment."""

    def __init__(
            self,
            world_model: DreamerlWrapper,
            prior_shape: Tuple[int, ...],
            belief_shape: Tuple[int, ...],
            obs_decoder: TensorDictModule = None,
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
        self.pred_continue = self.world_model.pred_continue

    def set_specs_from_env(self, env: EnvBase):
        """Sets the specs of the environment from the specs of the given environment."""
        super().set_specs_from_env(env)
        self.state_spec = CompositeSpec(
            state=self.observation_spec["state"],
            belief=self.observation_spec["belief"],
            shape=env.batch_size,
        )

    def _reset(self, tensordict=None, **kwargs) -> TensorDict:
        batch_size = tensordict.batch_size if tensordict is not None else []
        device = tensordict.device if tensordict is not None else self.device
        td = self.state_spec.rand(shape=batch_size).to(device)
        td.set("action", self.action_spec.rand(shape=batch_size).to(device))
        td[("next", "reward")] = self.reward_spec.rand(shape=batch_size).to(device)
        td.update(self.observation_spec.rand(shape=batch_size).to(device))
        return td

    def _step(
            self,
            tensordict: TensorDict,
    ) -> TensorDict:
        tensordict_out = tensordict.clone(recurse=False)
        tensordict_out = self.world_model(tensordict_out)
        # done can be missing, it will be filled by `step`
        return tensordict_out.select(
            *self.observation_spec.keys(),
            *self.full_done_spec.keys(),
            *(["pred_continue"] if self.pred_continue else []),
            *self.full_reward_spec.keys(),
            strict=False,
        )

    def decode_obs(self, tensordict: TensorDict, compute_latents=False) -> TensorDict:
        if self.obs_decoder is None:
            raise ValueError("No observation decoder provided")
        if compute_latents:
            tensordict = self.world_model(tensordict)
        return self.obs_decoder(tensordict)
