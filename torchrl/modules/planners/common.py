# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from typing import Optional

import torch
from tensordict.tensordict import TensorDictBase

from torchrl.envs import EnvBase
from torchrl.modules import SafeModule


class MPCPlannerBase(SafeModule, metaclass=abc.ABCMeta):
    """MPCPlannerBase abstract Module.

    This class inherits from :obj:`SafeModule`. Provided a :obj:`TensorDict`, this module will perform a Model Predictive Control (MPC) planning step.
    At the end of the planning step, the :obj:`MPCPlanner` will return a proposed action.

    Args:
        env (EnvBase): The environment to perform the planning step on (Can be :obj:`ModelBasedEnvBase` or :obj:`EnvBase`).
        action_key (str, optional): The key that will point to the computed action.
    """

    def __init__(
        self,
        env: EnvBase,
        action_key: str = "action",
    ):
        # Check if env is stateless
        if env.batch_locked:
            raise ValueError(
                "Environment is batch_locked. MPCPlanners need an environnement that accepts batched inputs with any batch size"
            )
        out_keys = [action_key]
        in_keys = list(env.observation_spec.keys())
        super().__init__(env, in_keys=in_keys, out_keys=out_keys)
        self.env = env
        self.action_spec = env.action_spec
        self.to(env.device)

    @abc.abstractmethod
    def planning(self, td: TensorDictBase) -> torch.Tensor:
        """Performs the MPC planning step.

        Args:
            td (TensorDict): The TensorDict to perform the planning step on.
        """
        raise NotImplementedError()

    def forward(
        self,
        tensordict: TensorDictBase,
        tensordict_out: Optional[TensorDictBase] = None,
        **kwargs,
    ) -> TensorDictBase:
        if "params" in kwargs or "vmap" in kwargs:
            raise ValueError(
                "MPCPlannerBase does not currently support functional programming."
            )
        action = self.planning(tensordict)
        action = self.action_spec.project(action)
        tensordict_out = self._write_to_tensordict(
            tensordict,
            (action,),
            tensordict_out,
        )
        return tensordict_out
