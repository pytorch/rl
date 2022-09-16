# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from copy import deepcopy
from typing import Optional, Union, List

import numpy as np
import torch

from torchrl.data import TensorDict
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.common import EnvBase
from torchrl.modules.tensordict_module import TensorDictModule

__all__ = ["ModelBasedEnv"]


class ModelBasedEnv(EnvBase, metaclass=abc.ABCMeta):
    """
    Basic environnement for Model Based RL algorithms.

    This class is a wrapper around the model of the MBRL algorithm.
    This class is meant to give an env framework to a world model (including but not limited to observations, reward, done state and safety constraints models).
    It is meant to behave as a classical environment.

    This class is meant to be used as a base class for other environments. It is not meant to be used directly.

    Example:
    >>> import torch
    >>> class MyMBEnv(ModelBasedEnv):
    >>>     def __init__(self, world_model, device="cpu", dtype=None, batch_size=None):
    >>>         super(MyEnv, self).__init__(world_model, device=device, dtype=dtype, batch_size=batch_size)
    >>>     def _reset(self):
    >>>         td = TensorDict(
    >>>             {
    >>>                 "hidden_observation": torch.randn(*self.batch_size, 4),
    >>>                 "next_hidden_observation": torch.randn(*self.batch_size, 4),
    >>>                 "action": torch.randn(*self.batch_size, 1),
    >>>             },
    >>>             batch_size=self.batch_size,
    >>>         )
    >>>         return td
    >>>     def _set_seed(self, seed: int) -> int:
    >>>         return seed + 1

    Then, you can use this environment as follows:

    >>> from torchrl.modules import MLP, WorldModelWrapper
    >>> import torch.nn as nn
    >>> world_model = WorldModelWrapper(
    >>>     TensorDictModule(
    >>>         MLP(out_features=4, activation_class=nn.ReLU, activate_last_layer=True, depth=0),
    >>>         in_keys=["hidden_observation", "action"],
    >>>         out_keys=["next_hidden_observation"],
    >>>     ),
    >>>     TensorDictModule(
    >>>         nn.Linear(4, 1),
    >>>         in_keys=["hidden_observation"],
    >>>         out_keys=["reward"],
    >>>     ),
    >>> )
    >>> world_model = MyWorldModel()
    >>> env = MyMBEnv(world_model)
    >>> td = env.reset()
    >>> env.rollout(td, max_steps=10)
    ```


    Properties:
        - observation_spec (CompositeSpec): sampling spec of the observations;
        - action_spec (TensorSpec): sampling spec of the actions;
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
        world_model: TensorDictModule,
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

    @classmethod
    def __new__(cls, *args, **kwargs):
        cls._inplace_update = False
        return super().__new__(cls, *args, _batch_locked=False, **kwargs)

    def set_specs_from_env(self, env: EnvBase):
        """
        Sets the specs of the environment from the specs of the given environment.
        """
        self.observation_spec = deepcopy(env.observation_spec)
        self.action_spec = deepcopy(env.action_spec)
        self.reward_spec = deepcopy(env.reward_spec)
        self.input_spec = deepcopy(env.input_spec)

    def _step(
        self,
        tensordict: TensorDict,
    ) -> TensorDict:
        # step method requires to be immutable
        tensordict_out = tensordict.clone(recursive=False)
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
            tensordict_out["done"] = torch.zeros(
                tensordict_out.shape, dtype=torch.bool, device=tensordict_out.device
            )
        return tensordict_out

    @abc.abstractmethod
    def _reset(self, tensordict: TensorDict, **kwargs) -> TensorDict:
        raise NotImplementedError

    @abc.abstractmethod
    def _set_seed(self, seed: Optional[int]) -> int:
        raise NotImplementedError
