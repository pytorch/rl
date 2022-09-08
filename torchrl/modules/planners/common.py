# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from typing import Optional

import torch

from torchrl.data.tensordict.tensordict import TensorDictBase
from torchrl.envs import EnvBase
from torchrl.modules import TensorDictModule

__all__ = ["MPCPlannerBase"]


class MPCPlannerBase(TensorDictModule, metaclass=abc.ABCMeta):
    """
    MPCPlannerBase Module. This is an abstract class and must be implemented by the user.

    This class inherits from TensorDictModule. Provided a TensorDict, this module will perform a Model Predictive Control (MPC) planning step.
    At the end of the planning step, the MPCPlanner will return the action that should be taken.

    Args:
        env (Environment): The environment to perform the planning step on (Can be ModelBasedEnv or EnvBase).
        action_key (str): The key in the TensorDict to use to store the action.

    Returns:
        TensorDict: The TensorDict with the action added.
    """

    def __init__(
        self,
        env: EnvBase,
        action_key: str = "action",
    ):
        # Check if env is stateless
        if env.batch_locked:
            raise ValueError("Environment is not stateless")
        out_keys = [action_key]
        in_keys = list(env.observation_spec.keys())
        super().__init__(env, in_keys=in_keys, out_keys=out_keys)
        self.env = env
        self.action_spec = env.action_spec

    @abc.abstractmethod
    def planning(self, td: TensorDictBase) -> torch.Tensor:
        """
        Perform the MPC planning step.
        Args:
            td (TensorDict): The TensorDict to perform the planning step on.
        Returns:
            TensorDict: The TensorDict with the action added.
        """
        raise NotImplementedError()

    def forward(
        self,
        tensordict: TensorDictBase,
        tensordict_out: Optional[TensorDictBase] = None,
        **kwargs,
    ) -> TensorDictBase:
        if "params" in kwargs or "vmap" in kwargs:
            raise ValueError("params not supported")
        action = self.planning(tensordict)
        action = self.action_spec.project(action)
        tensordict_out = self._write_to_tensordict(
            tensordict,
            (action,),
            tensordict_out,
        )
        return tensordict_out
