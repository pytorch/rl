# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from torchrl.data.tensor_specs import Composite
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.common import EnvBase
from torchrl.envs.model_based import ModelBasedEnvBase
from torchrl.envs.transforms.transforms import Transform


class DreamerEnv(ModelBasedEnvBase):
    """Dreamer simulation environment."""

    def __init__(
        self,
        world_model: TensorDictModule,
        prior_shape: Tuple[int, ...],
        belief_shape: Tuple[int, ...],
        obs_decoder: TensorDictModule = None,
        device: DEVICE_TYPING = "cpu",
        batch_size: Optional[torch.Size] = None,
    ):
        super(DreamerEnv, self).__init__(
            world_model, device=device, batch_size=batch_size
        )
        self.obs_decoder = obs_decoder
        self.prior_shape = prior_shape
        self.belief_shape = belief_shape

    def set_specs_from_env(self, env: EnvBase):
        """Sets the specs of the environment from the specs of the given environment."""
        super().set_specs_from_env(env)
        self.action_spec = self.action_spec.to(self.device)
        self.state_spec = Composite(
            state=self.observation_spec["state"],
            belief=self.observation_spec["belief"],
            shape=env.batch_size,
        )

    def _reset(self, tensordict=None, **kwargs) -> TensorDict:
        batch_size = tensordict.batch_size if tensordict is not None else []
        device = tensordict.device if tensordict is not None else self.device
        if tensordict is None:
            td = self.state_spec.rand(shape=batch_size)
            # why don't we reuse actions taken at those steps?
            td.set("action", self.action_spec.rand(shape=batch_size))
            td[("next", "reward")] = self.reward_spec.rand(shape=batch_size)
            td.update(self.observation_spec.rand(shape=batch_size))
            if device is not None:
                td = td.to(device, non_blocking=True)
                if torch.cuda.is_available() and device.type == "cpu":
                    torch.cuda.synchronize()
                elif torch.backends.mps.is_available():
                    torch.mps.synchronize()
        else:
            td = tensordict.clone()
        return td

    def decode_obs(self, tensordict: TensorDict, compute_latents=False) -> TensorDict:
        if self.obs_decoder is None:
            raise ValueError("No observation decoder provided")
        if compute_latents:
            tensordict = self.world_model(tensordict)
        return self.obs_decoder(tensordict)


class DreamerDecoder(Transform):
    """A transform to record the decoded observations in Dreamer.

    Examples:
        >>> model_based_env = DreamerEnv(...)
        >>> model_based_env_eval = model_based_env.append_transform(DreamerDecoder())
    """

    def _call(self, tensordict):
        return self.parent.base_env.obs_decoder(tensordict)

    def _reset(self, tensordict, tensordict_reset):
        return self._call(tensordict_reset)

    def transform_observation_spec(self, observation_spec):
        return observation_spec
