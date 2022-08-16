# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
from typing import Optional, Union, List, Tuple, Callable
from typing import Optional, Union, List

import numpy as np
import torch
import torch.nn as nn

from torchrl.data import TensorDict
from torchrl.data import (
    TensorDict,
    TensorSpec,
    CompositeSpec,
    NdUnboundedContinuousTensorSpec,
)
from torchrl.modules.tensordict_module.world_models import WorldModelWrapper
from ..data.utils import DEVICE_TYPING
from ..modules.tensordict_module import TensorDictModule, TensorDictSequence
from .common import EnvBase


class ModelBasedEnv(EnvBase, metaclass=abc.ABCMeta):
    """
    Basic environnement for Model Based RL algorithms.

    This class is a wrapper around the model of the MBRL algorithm.
    This class is meant to give an env framework to a world model and a reward model.
    It is meant to behave as a classical environment.

    Properties:
        - observation_spec (CompositeSpec): sampling spec of the observations;
        - action_spec (TensorSpec): sampling spec of the actions;
        - input_spec (CompositeSpec): sampling spec of the actions and/or other inputs;
        - reward_spec (TensorSpec): sampling spec of the rewards;
        - batch_size (torch.Size): number of environments contained in the instance;
        - device (torch.device): device where the env input and output are expected to live
        - is_done (torch.Tensor): boolean value(s) indicating if the environment has reached a done state since the
            last reset

    Args:
        world_model (nn.Module): model that generates world states and its corresponding rewards;
        device (torch.device): device where the env input and output are expected to live
        dtype (torch.dtype): dtype of the env input and output
        batch_size (torch.Size): number of environments contained in the instance

    Methods:
        step (TensorDict -> TensorDict): step in the environment
        reset (TensorDict, optional -> TensorDict): reset the environment
        set_seed (int -> int): sets the seed of the environment
        rand_step (TensorDict, optional -> TensorDict): random step given the action spec
        rollout (Callable, ... -> TensorDict): executes a rollout in the environment with the given policy (or random
            steps if no policy is provided)

    """

    def __init__(
        self,
        world_model: Union[nn.Module, List[TensorDictModule], TensorDictSequence],
        device: DEVICE_TYPING = "cpu",
        dtype: Optional[Union[torch.dtype, np.dtype]] = None,
        batch_size: Optional[torch.Size] = None,
    ):
        super(ModelBasedEnv, self).__init__(
            device=device, dtype=dtype, batch_size=batch_size
        )
        self.world_model = world_model

    def set_specs_from_env(self, env: EnvBase):
        """
        Sets the specs of the environment from the specs of the given environment.
        """
        self.observation_spec = env.observation_spec
        self.action_spec = env.action_spec
        self.reward_spec = env.reward_spec

    def _step(
        self,
        tensordict: TensorDict,
    ) -> TensorDict:
        # step method requires to be immutable
        tensordict_in = tensordict.clone()
        # Compute world state
        tensordict_out = self.world_model(tensordict_in)
        # Step requires a done flag. No sense for MBRL so we set it to False
        tensordict_out["done"] = torch.zeros(tensordict_out.shape, dtype=torch.bool)
        return tensordict_out

    @abc.abstractmethod
    def _reset(self, tensordict: TensorDict, **kwargs) -> TensorDict:
        raise NotImplementedError

    @abc.abstractmethod
    def _set_seed(self, seed: Optional[int]) -> int:
        raise NotImplementedError

class DreamerEnv(ModelBasedEnv):
    def __init__(
        self,
        world_model: WorldModelWrapper,
        obs_decoder: TensorDictModule = None,
        device: DEVICE_TYPING = "cpu",
        dtype: Optional[Union[torch.dtype, np.dtype]] = None,
        batch_size: Optional[torch.Size] = torch.Size([1]),
    ):
        super(DreamerEnv, self).__init__(world_model, device, dtype, batch_size)
        self.obs_decoder = obs_decoder
        self._latent_spec = None

    @property
    def latent_spec(self) -> TensorSpec:
        if self._latent_spec is None:
            raise ValueError("No latent spec set")
        return self._latent_spec

    @latent_spec.setter
    def latent_spec(self, shapes: Tuple[torch.Size]) -> None:
        self._latent_spec = CompositeSpec(
            prior_state=NdUnboundedContinuousTensorSpec(shape=shapes[0]),
            belief=NdUnboundedContinuousTensorSpec(shape=shapes[1]),
        )

    def _reset(self, tensordict=None, **kwargs) -> TensorDict:
        td = self.latent_spec.rand(shape=self.batch_size)
        td["action"] = self.action_spec.rand(shape=self.batch_size)
        # td = td.unsqueeze(0).to_tensordict()
        td = self.step(td)
        return td

    def _set_seed(self, seed: Optional[int]) -> int:
        return seed

    def decode_obs(self, tensordict: TensorDict, compute_latents=False) -> TensorDict:
        if self.obs_decoder is None:
            raise ValueError("No observation decoder provided")
        if compute_latents:
            tensordict = self(tensordict)
        return self.obs_decoder(tensordict)

    def to(self, device: DEVICE_TYPING) -> DreamerEnv:
        super().to(device)
        if self.obs_decoder is not None:
            self.obs_decoder.to(device)
        self.latent_spec.to(device)
        return self
