# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.modules import ProbabilisticTensorDictModule
from torchrl.objectives.returns.vtrace import vtrace


class QValEstimator:
    def __init__(self, value_model: ProbabilisticTensorDictModule):
        self.value_model = value_model

    @property
    def device(self) -> torch.device:
        return next(self.value_model.parameters()).device

    def forward(self, tensordict: _TensorDict) -> None:
        tensordict_device = tensordict.to(self.device)
        self.value_model_device(tensordict_device)  # udpates the value key
        gamma = tensordict_device.get("gamma")
        reward = tensordict_device.get("reward")
        next_value = torch.cat(
            [
                tensordict_device.get("value")[:, 1:],
                torch.ones_like(reward[:, :1]),
            ],
            1,
        )
        q_value = reward + gamma * next_value
        tensordict_device.set("q_value", q_value)


class VTraceEstimator:
    def forward(self, tensordict: _TensorDict) -> _TensorDict:
        tensordict_device = tensordict.to(device)
        rewards = tensordict_device.get("reward")
        vals = tensordict_device.get("value")
        log_mu = tensordict_device.get("log_mu")
        log_pi = tensordict_device.get("log_pi")
        gamma = tensordict_device.get("gamma")
        v_trace, rho = vtrace(
            rewards,
            vals,
            log_pi,
            log_mu,
            gamma,
            rho_bar=self.rho_bar,
            c_bar=self.c_bar,
        )
        tensordict_device.set("v_trace", v_trace)
        tensordict_device.set("rho", rho)
        return tensordict_device


class ImpalaLoss:
    def forward(self, tensordict):
        tensordict_device = tensordict.to(device)
        self.q_val_estimator(tensordict_device)
        self.v_trace_estimator(tensordict_device)
