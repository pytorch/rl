# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Callable, Iterator, Optional, Union, Dict, List

import numpy as np
import torch
import torch.nn as nn

from torchrl.data import TensorDict, TensorSpec
from ..data.utils import DEVICE_TYPING
from ..modules.tensordict_module import TensorDictModule, TensorDictSequence
from .common import EnvBase, Specs

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
        model (TensorDictModule, list of TensorDictModule or TensorDictSequence): model for out MBRL algorithm
        in_keys_train (str): keys of the input tensors to the model used for training
        out_keys_train (str): keys of the output tensors to the model used for training
        in_keys_test (str): keys of the input tensors to the model used for inference
        out_keys_test (str): keys of the output tensors to the model used for inference
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
        model: Union[nn.Module, List[TensorDictModule]],
        in_keys_train: Optional[str] = None,
        out_keys_train: Optional[str] = None,
        in_keys_test: Optional[str] = None,
        out_keys_test: Optional[str] = None,
        device: DEVICE_TYPING = "cpu",
        dtype: Optional[Union[torch.dtype, np.dtype]] = None,
        batch_size: Optional[torch.Size] = None,
    ):
        super(ModelBasedEnv, self).__init__(
            device=device, dtype=dtype, batch_size=batch_size
        )
        if type(model) is TensorDictModule:
            modules = [model]
        elif type(model) is list:
            if not all([type(model[i]) is TensorDictModule] for i in range(len(model))):
                raise TypeError(
                    "model must be a list of TensorDictModule, a TensorDictSequence or an nn.Module"
                )
            else:
                modules = model
        elif type(model) is TensorDictSequence:
            modules = [model.module[i] for i in range(len(model.module))]
        else:
            raise TypeError(
                "model must be a TensorDictModule, a list of TensorDictModule or a TensorDictSequence"
            )

        self.module = TensorDictSequence(*modules)

        if in_keys_train is None:
            self.in_keys_train = self.module.in_keys

        if out_keys_train is None:
            self.out_keys_train = self.module.out_keys

        if in_keys_test is None:
            self.in_keys_test = self.module.in_keys

        if out_keys_test is None:
            self.out_keys_test = self.module.out_keys

        self.train_submodel = self.module.select_subsequence(
            in_keys_train, out_keys_train
        )
        self.test_submodel = self.module.select_subsequence(in_keys_test, out_keys_test)

    def set_specs(
        self,
        action_spec: TensorSpec,
        observation_spec: TensorSpec,
        reward_spec: TensorSpec,
    ) -> None:
        self.action_spec = action_spec
        self.observation_spec = observation_spec
        self.reward_spec = reward_spec

    def set_specs_from_env(self, env: EnvBase) -> None:
        self.set_specs(
            action_spec=env.action_spec,
            observation_spec=env.observation_spec,
            reward_spec=env.reward_spec,
        )

    def forward(
        self,
        tensordict: TensorDict,
    ) -> TensorDict:
        tensordict = self.module(
            tensordict,
        )
        return tensordict

    def train_step(self, tensordict: TensorDict) -> TensorDict:
        self.train()
        tensordict = self.train_submodel(
            tensordict,
        )
        return tensordict

    def _step(
        self,
        tensordict: TensorDict,
    ) -> TensorDict:
        self.eval()
        tensordict = self.test_submodel(
            tensordict,
        )
        return tensordict

    def step(self, tensordict: TensorDict) -> TensorDict:
        """Makes a step in the environment.
        Step accepts a single argument, tensordict, which usually carries an 'action' key which indicates the action
        to be taken.

        Args:
            tensordict (TensorDict): Tensordict containing the action to be taken.

        Returns:
            the input tensordict, modified in place with the resulting observations, done state and reward
            (+ others if needed).

        """

        # sanity check
        if tensordict.get("action").dtype is not self.action_spec.dtype:
            raise TypeError(
                f"expected action.dtype to be {self.action_spec.dtype} "
                f"but got {tensordict.get('action').dtype}"
            )

        tensordict = self._step(tensordict)

        for key in self._select_observation_keys(tensordict):
            obs = tensordict.get(key)
            self.observation_spec.type_check(obs, key)

        if tensordict._get_meta("reward").dtype is not self.reward_spec.dtype:
            raise TypeError(
                f"expected reward.dtype to be {self.reward_spec.dtype} "
                f"but got {tensordict.get('reward').dtype}"
            )
        return tensordict

    def to(self, device: DEVICE_TYPING) -> ModelBasedEnv:
        super().to(device)
        self.module.to(device)
        return self
