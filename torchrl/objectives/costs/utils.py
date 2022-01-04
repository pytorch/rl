__all__ = ["SoftUpdate", "HardUpdate", "distance_loss"]

from numbers import Number
from typing import Union

import torch
from torch import nn
from torch.nn import functional as F


def distance_loss(v1: torch.Tensor, v2: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "l2":
        value_loss = F.mse_loss(
            v1,
            v2,
            reduction="none",
        )

    elif loss_type == "l1":
        value_loss = F.l1_loss(
            v1,
            v2,
            reduction="none",
        )

    elif loss_type == "smooth_l1":
        value_loss = F.smooth_l1_loss(
            v1,
            v2,
            reduction="none",
        )
    else:
        raise NotImplementedError(f"Unknown loss {loss_type}")
    return value_loss


class ValueLoss:
    value_network: nn.Module
    target_value_network: nn.Module


class _TargetNetUpdate:
    def __init__(self, loss_module: Union["DQNLoss", "DDPGLoss"]):
        self.has_target_actor = hasattr(loss_module, '_target_actor_network')
        net = loss_module.value_network
        target_net = loss_module.target_value_network
        if self.has_target_actor:
            net = nn.ModuleList((net, loss_module.actor_network))
            target_net = nn.ModuleList((target_net, loss_module.target_actor_network))
        self.net = net
        self.target_net = target_net

        self.initialized = False

    def init_(self) -> None:
        for (n1, p1), (n2, p2) in zip(self.net.named_parameters(), self.target_net.named_parameters()):
            p2.data.copy_(p1.data)
        self.initialized = True


class SoftUpdate(_TargetNetUpdate):
    def __init__(self, loss_module: Union["DQNLoss", "DDPGLoss"], eps: Number = 0.999):
        assert eps < 1.0 and eps > 0.0
        super(SoftUpdate, self).__init__(loss_module)
        self.eps = eps

    def step(self) -> None:
        if not self.initialized:
            raise Exception(
                f'{self.__class__.__name__} must be initialized (`{self.__class__.__name__}.init_()`) before calling step()')
        for (n1, p1), (n2, p2) in zip(self.net.named_parameters(), self.target_net.named_parameters()):
            p2.data.copy_(p2.data * self.eps + p1.data * (1 - self.eps))


class HardUpdate(_TargetNetUpdate):
    def __init__(self, loss_module: Union["DQNLoss", "DDPGLoss"], value_network_update_interval: Number = 1000):
        super(HardUpdate, self).__init__(loss_module)
        self.value_network_update_interval = value_network_update_interval
        self.counter = 0

    def step(self) -> None:
        if not self.initialized:
            raise Exception(
                f'{self.__class__.__name__} must be initialized (`{self.__class__.__name__}.init_()`) before calling step()')
        if self.counter == self.value_network_update_interval:
            self.counter = 0
            print("updating target value network")
            self.target_net.load_state_dict(self.net.state_dict())
        else:
            self.counter += 1
