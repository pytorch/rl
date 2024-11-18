# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import warnings
from typing import List, Optional

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.common import EnvBase


class ModelBasedEnvBase(EnvBase):
    """Basic environnement for Model Based RL sota-implementations.

    Wrapper around the model of the MBRL algorithm.
    It is meant to give an env framework to a world model (including but not limited to observations, reward, done state and safety constraints models).
    and to behave as a classical environment.

    This is a base class for other environments and it should not be used directly.

    Example:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import Composite, Unbounded
        >>> class MyMBEnv(ModelBasedEnvBase):
        ...     def __init__(self, world_model, device="cpu", dtype=None, batch_size=None):
        ...         super().__init__(world_model, device=device, dtype=dtype, batch_size=batch_size)
        ...         self.observation_spec = Composite(
        ...             hidden_observation=Unbounded((4,))
        ...         )
        ...         self.state_spec = Composite(
        ...             hidden_observation=Unbounded((4,)),
        ...         )
        ...         self.action_spec = Unbounded((1,))
        ...         self.reward_spec = Unbounded((1,))
        ...
        ...     def _reset(self, tensordict: TensorDict) -> TensorDict:
        ...         tensordict = TensorDict(
        ...             batch_size=self.batch_size,
        ...             device=self.device,
        ...         )
        ...         tensordict = tensordict.update(self.state_spec.rand())
        ...         tensordict = tensordict.update(self.observation_spec.rand())
        ...         return tensordict
        >>> # This environment is used as follows:
        >>> import torch.nn as nn
        >>> from torchrl.modules import MLP, WorldModelWrapper
        >>> world_model = WorldModelWrapper(
        ...     TensorDictModule(
        ...         MLP(out_features=4, activation_class=nn.ReLU, activate_last_layer=True, depth=0),
        ...         in_keys=["hidden_observation", "action"],
        ...         out_keys=["hidden_observation"],
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
                next: LazyStackedTensorDict(
                    fields={
                        hidden_observation: Tensor(torch.Size([10, 4]), dtype=torch.float32)},
                    batch_size=torch.Size([10]),
                    device=cpu,
                    is_shared=False),
                reward: Tensor(torch.Size([10, 1]), dtype=torch.float32)},
            batch_size=torch.Size([10]),
            device=cpu,
            is_shared=False)


    Properties:
        - observation_spec (Composite): sampling spec of the observations;
        - action_spec (TensorSpec): sampling spec of the actions;
        - reward_spec (TensorSpec): sampling spec of the rewards;
        - input_spec (Composite): sampling spec of the inputs;
        - batch_size (torch.Size): batch_size to be used by the env. If not set, the env accept tensordicts of all batch sizes.
        - device (torch.device): device where the env input and output are expected to live

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
        batch_size: Optional[torch.Size] = None,
        run_type_checks: bool = False,
    ):
        super(ModelBasedEnvBase, self).__init__(
            device=device,
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
        """Sets the specs of the environment from the specs of the given environment."""
        device = self.device
        output_spec = env.output_spec.clone()
        input_spec = env.input_spec.clone()
        if device is not None:
            output_spec = output_spec.to(device)
            input_spec = input_spec.to(device)
        self.__dict__["_output_spec"] = output_spec
        self.__dict__["_input_spec"] = input_spec
        self.empty_cache()

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
        # done can be missing, it will be filled by `step`
        tensordict_out = tensordict_out.select(
            *self.observation_spec.keys(),
            *self.full_done_spec.keys(),
            *self.full_reward_spec.keys(),
            strict=False,
        )
        return tensordict_out

    @abc.abstractmethod
    def _reset(self, tensordict: TensorDict, **kwargs) -> TensorDict:
        raise NotImplementedError

    def _set_seed(self, seed: Optional[int]) -> int:
        warnings.warn("Set seed isn't needed for model based environments")
        return seed
