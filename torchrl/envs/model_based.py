# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
from random import randint
from typing import Optional, Union, List

import numpy as np
import torch
import torch.nn as nn

from torchrl.data import TensorDict
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
        params: Optional[List[torch.Tensor]] = None,
        buffers: Optional[List[torch.Tensor]] = None,
        device: DEVICE_TYPING = "cpu",
        dtype: Optional[Union[torch.dtype, np.dtype]] = None,
        batch_size: Optional[torch.Size] = None,
    ):
        super(ModelBasedEnv, self).__init__(
            device=device, dtype=dtype, batch_size=batch_size
        )
        self.world_model = world_model
        self.world_model_params = params
        self.world_model_buffers = buffers

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
        tensordict_out = tensordict.clone()
        # Compute world state
        if self.world_model_params is not None:
            tensordict_out = self.world_model(
                tensordict_out,
                params=self.world_model_params,
                buffers=self.world_model_buffers,
            )
        else:
            tensordict_out = self.world_model(tensordict_out)
        # Step requires a done flag. No sense for MBRL so we set it to False
        if "done" not in self.world_model.out_keys:
            tensordict_out["done"] = torch.zeros(tensordict_out.shape, dtype=torch.bool)
        return tensordict_out

    @abc.abstractmethod
    def _reset(self, tensordict: TensorDict, **kwargs) -> TensorDict:
        raise NotImplementedError

    @abc.abstractmethod
    def _set_seed(self, seed: Optional[int]) -> int:
        raise NotImplementedError


class MBPOEnv(ModelBasedEnv):
    def __init__(
        self,
        world_model: Union[nn.Module, List[TensorDictModule], TensorDictSequence],
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
