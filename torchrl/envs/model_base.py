# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Callable, Iterator, Optional, Union, Dict, List

import numpy as np
import torch.nn as nn
import torch

from .common import EnvBase
from ..data.tensordict.tensordict import TensorDictBase
from ..modules.tensordict_module import TensorDictModule, TensorDictSequence
from ..data.utils import DEVICE_TYPING

dtype_map = {
    torch.float: np.float32,
    torch.double: np.float64,
    torch.bool: bool,
}

class ModelBasedEnv(TensorDictSequence, EnvBase):
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
        step (TensorDictBase -> TensorDictBase): step in the environment
        reset (TensorDictBase, optional -> TensorDictBase): reset the environment
        set_seed (int -> int): sets the seed of the environment
        rand_step (TensorDictBase, optional -> TensorDictBase): random step given the action spec
        rollout (Callable, ... -> TensorDictBase): executes a rollout in the environment with the given policy (or random
            steps if no policy is provided)
        train_step (TensorDictBase -> TensorDictBase): step in the environment for training
        forward (TensorDictBase, optional -> TensorDictBase): forward pass of the model

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
    
        if type(model) is TensorDictModule:
            modules = [model]
        elif type(model) is list:
            if not all([type(model[i]) is  TensorDictModule] for i in range(len(model))):
                raise TypeError("model must be a list of TensorDictModule, a TensorDictSequence or an nn.Module")
            else:
                modules = model
        elif type(model) is TensorDictSequence:
            modules = [model.module[i] for i in range(len(model.module)) ]
        else:
            raise TypeError("model must be a TensorDictModule, a list of TensorDictModule or a TensorDictSequence")
        TensorDictSequence.__init__(self, *modules)
        # EnvBase.__init__(self, device=device, dtype=dtype, batch_size=batch_size)

        if self.in_keys_train is None:
            self.in_keys_train = self.in_keys
        else:
            self.in_keys_train = in_keys_train

        if self.out_keys_train is None:
            self.out_keys_train = self.out_keys
        else:
            self.out_keys_train = out_keys_train

        if self.in_keys_test is None:
            self.in_keys_test = self.in_keys
        else:
            self.in_keys_test = in_keys_test

        if self.out_keys_test is None:
            self.out_keys_test = self.out_keys
        else:
            self.out_keys_test = out_keys_test
            
    def forward(self, tensordict: TensorDictBase, in_keys_filter: Optional[str] = None, out_keys_filter: Optional[str] = None) -> TensorDictBase:
        tensordict = TensorDictSequence.forward(self, tensordict, in_keys_filter=in_keys_filter, out_keys_filter=out_keys_filter)
        return tensordict

    def train_step(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.train()
        tensordict = self(tensordict, in_keys_filter=self.in_keys_train, out_keys_filter=self.out_keys_train)
        return tensordict
    
    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        self.eval()
        tensordict = self(tensordict, in_keys_filter=self.in_keys_test, out_keys_filter=self.out_keys_test)
        return tensordict
    def to(self, device: DEVICE_TYPING) -> ModelBasedEnv:
        EnvStateful.to(device)
        TensorDictSequence.to(device)
        return self