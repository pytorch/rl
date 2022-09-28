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

__all__ = ["ModelBasedEnvBase"]


class ModelBasedEnvBase(EnvBase, metaclass=abc.ABCMeta):
    """Basic environnement for Model Based RL algorithms.

    Wrapper around the model of the MBRL algorithm.
    It is meant to give an env framework to a world model (including but not limited to observations, reward, done state and safety constraints models).
    and to behave as a classical environment.

    This is a base class for other environments and it should not be used directly.

    Example:
    >>> import torch
    >>> from torchrl.data import TensorDict, CompositeSpec, NdUnboundedContinuousTensorSpec
    >>> class MyMBEnv(ModelBasedEnvBase):
    ...     def __init__(self, world_model, device="cpu", dtype=None, batch_size=None):
    ...         super().__init__(world_model, device=device, dtype=dtype, batch_size=batch_size)
    ...         self.observation_spec = CompositeSpec(
    ...             next_hidden_observation=NdUnboundedContinuousTensorSpec((4,))
    ...         )
    ...         self.input_spec = CompositeSpec(
    ...             hidden_observation=NdUnboundedContinuousTensorSpec((4,)),
    ...             action=NdUnboundedContinuousTensorSpec((1,)),
    ...         )
    ...         self.reward_spec = NdUnboundedContinuousTensorSpec((1,))
    ...
    ...     def _reset(self, tensordict: TensorDict) -> TensorDict:
    ...         tensordict = TensorDict({},
    ...             batch_size=self.batch_size,
    ...             device=self.device,
    ...         )
    ...         tensordict = tensordict.update(self.input_spec.rand(self.batch_size))
    ...         tensordict = tensordict.update(self.observation_spec.rand(self.batch_size))
    ...         return tensordict
    >>> # This environment is used as follows:
    >>> from torchrl.modules import MLP, WorldModelWrapper
    >>> import torch.nn as nn
    >>> world_model = WorldModelWrapper(
    ...     TensorDictModule(
    ...         MLP(out_features=4, activation_class=nn.ReLU, activate_last_layer=True, depth=0),
    ...         in_keys=["hidden_observation", "action"],
    ...         out_keys=["next_hidden_observation"],
    ...     ),
    ...     TensorDictModule(
    ...         nn.Linear(4, 1),
    ...         in_keys=["hidden_observation"],
    ...         out_keys=["reward"],
    ...     ),
    ... )
    >>> env = MyMBEnv(world_model)
    >>> tensordict = env.rollout(max_steps=10)
    >>> print(tensordict)
    TensorDict(
        fields={
            action: Tensor(torch.Size([10, 1]), dtype=torch.float32),
            done: Tensor(torch.Size([10, 1]), dtype=torch.bool),
            hidden_observation: Tensor(torch.Size([10, 4]), dtype=torch.float32),
            next_hidden_observation: Tensor(torch.Size([10, 4]), dtype=torch.float32),
            reward: Tensor(torch.Size([10, 1]), dtype=torch.float32)},
        batch_size=torch.Size([10]),
        device=cpu,
        is_shared=False)


    Properties:
        - observation_spec (CompositeSpec): sampling spec of the observations;
        - action_spec (TensorSpec): sampling spec of the actions;
        - reward_spec (TensorSpec): sampling spec of the rewards;
        - input_spec (CompositeSpec): sampling spec of the inputs;
        - batch_size (torch.Size): batch_size to be used by the env. If not set, the env accept tensordicts of all batch sizes.
        - device (torch.device): device where the env input and output are expected to live
        - is_done (torch.Tensor): boolean value(s) indicating if the environment has reached a done state since the
            last reset

    Args:
        world_model (nn.Module): model that generates world states and its corresponding rewards;
        params (List[torch.Tensor], optional): list of parameters of the world model;
        buffers (List[torch.Tensor], optional): list of buffers of the world model;
        device (torch.device, optional): device where the env input and output are expected to live
        dtype (torch.dtype, optional): dtype of the env input and output
        batch_size (torch.Size, optional): number of environments contained in the instance
        run_type_check (bool, optional): whether to run type checks on the step of the env

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
        run_type_checks: bool = False,
    ):
        super(ModelBasedEnvBase, self).__init__(
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            run_type_checks=run_type_checks,
        )
        self.world_model = world_model.to(self.device)
        self.world_model_params = params
        self.world_model_buffers = buffers

    @classmethod
    def __new__(cls, *args, **kwargs):
        return super().__new__(
            cls, *args, _inplace_update=False, _batch_locked=False, **kwargs
        )

    def set_specs_from_env(self, env: EnvBase):
        """
        Sets the specs of the environment from the specs of the given environment.
        """
        self.observation_spec = deepcopy(env.observation_spec).to(self.device)
        self.reward_spec = deepcopy(env.reward_spec).to(self.device)
        self.input_spec = deepcopy(env.input_spec).to(self.device)

    def _step(
        self,
        tensordict: TensorDict,
    ) -> TensorDict:
        # step method requires to be immutable
        tensordict_out = tensordict.clone(recurse=False)
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
                tensordict_out.shape,
                dtype=torch.bool,
                device=tensordict_out.device,
            )
        return tensordict_out

    @abc.abstractmethod
    def _reset(self, tensordict: TensorDict, **kwargs) -> TensorDict:
        raise NotImplementedError

    def _set_seed(self, seed: Optional[int]) -> int:
        raise Warning("Set seed isn't needed for model based environments")
