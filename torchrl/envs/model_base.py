# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
from collections import OrderedDict
from numbers import Number
from typing import Any, Callable, Iterator, Optional, Union, Dict

import numpy as np
import torch.nn as nn
import torch

from common import EnvStateful
from ..data.tensordict.tensordict import TensorDictBase
from ..modules.tensordict_module import TensorDictModule
from ..data.utils import DEVICE_TYPING

dtype_map = {
    torch.float: np.float32,
    torch.double: np.float64,
    torch.bool: bool,
}

class ModelBasedEnv(EnvStateful):
    """
    Abstract environment parent class for TorchRL.

    Properties:
        - observation_spec (CompositeSpec): sampling spec of the observations;
        - action_spec (TensorSpec): sampling spec of the actions;
        - input_spec (CompositeSpec): sampling spec of the actions and/or other inputs;
        - reward_spec (TensorSpec): sampling spec of the rewards;
        - batch_size (torch.Size): number of environments contained in the instance;
        - device (torch.device): device where the env input and output are expected to live
        - is_done (torch.Tensor): boolean value(s) indicating if the environment has reached a done state since the
            last reset

    Methods:
        step (TensorDictBase -> TensorDictBase): step in the environment
        reset (TensorDictBase, optional -> TensorDictBase): reset the environment
        set_seed (int -> int): sets the seed of the environment
        rand_step (TensorDictBase, optional -> TensorDictBase): random step given the action spec
        rollout (Callable, ... -> TensorDictBase): executes a rollout in the environment with the given policy (or random
            steps if no policy is provided)

    """

    def __init__(
        self,
        model: nn.Module,
        in_keys_train: Optional[str] = None,
        out_keys_train: Optional[str] = None,
        in_keys_test: Optional[str] = None,
        out_keys_test: Optional[str] = None,
        device: DEVICE_TYPING = "cpu",
        dtype: Optional[Union[torch.dtype, np.dtype]] = None,
        batch_size: Optional[torch.Size] = None,
    ):
        super().__init__(device=device, dtype=dtype, batch_size=batch_size)

        self.model = TensorDictModule(model, in_keys=in_keys_train, out_keys=out_keys_train).to(device)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self.model(tensordict)

    def train_step(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self._step(tensordict)
    
    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        raise NotImplementedError
    
    def to(self, device: DEVICE_TYPING) -> EnvStateful:
        super().to(device)
        self.model.to(device)
        return self