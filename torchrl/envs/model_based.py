# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Optional, Union, List

import numpy as np
import torch
import torch.nn as nn

from torchrl.data import TensorDict
from ..data.utils import DEVICE_TYPING
from ..modules.tensordict_module import TensorDictModule, TensorDictSequence
from .common import EnvBase

dtype_map = {
    torch.float: np.float32,
    torch.double: np.float64,
    torch.bool: bool,
}


class ModelBasedEnv(EnvBase):
    """
    Basic environnement for Model Based RL algorithms.

    This class is a wrapper around the model of the MBRL algorithm.
    We can both train it and use it for inference.
    This wrapper is designed as a TensorDictSequence, so that we can use it as a TensorDictSequence.
    In particular, one can specify different input and output keys for the training and inference phases.

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
        world_model (nn.Module): model which will be used to generate the world state;
        reward_model (nn.Module): model which will be used to predict the reward;
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
        train_step (TensorDict -> TensorDict): step in the environment for training
        forward (TensorDict, optional -> TensorDict): forward pass of the model

    """

    def __init__(
        self,
        world_model: Union[nn.Module, List[TensorDictModule], TensorDictSequence],
        reward_model: Union[nn.Module, List[TensorDictModule], TensorDictSequence],
        device: DEVICE_TYPING = "cpu",
        dtype: Optional[Union[torch.dtype, np.dtype]] = None,
        batch_size: Optional[torch.Size] = None,
    ):
        super(ModelBasedEnv, self).__init__(
            device=device, dtype=dtype, batch_size=batch_size
        )
        self.word_model = world_model
        self.reward_model = reward_model

        self.inference_world_model = world_model
        self.inference_reward_model = reward_model
        self.set_optimizer()

    def forward(
        self,
        tensordict: TensorDict,
    ) -> TensorDict:
        tensordict = self.word_model(tensordict)
        # Compute rewards
        tensordict = self.reward_model(tensordict)
        return tensordict

    def train_step(self, tensordict: TensorDict) -> TensorDict:
        self.train()
        # Extract latent states
        tensordict = self(tensordict)
        return tensordict

    def _step(
        self,
        tensordict: TensorDict,
    ) -> TensorDict:
        tensordict = self.inference_world_model(tensordict)
        # Compute rewards
        tensordict = self.inference_reward_model(tensordict)
        return tensordict

    def step(self, tensordict: TensorDict) -> TensorDict:
        return self._step(tensordict)

    def to(self, device: DEVICE_TYPING) -> ModelBasedEnv:
        super().to(device)
        self.module.to(device)
        return self
