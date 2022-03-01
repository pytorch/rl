__all__ = ["SoftUpdate", "HardUpdate", "distance_loss", "hold_out_params"]

import functools
from collections import OrderedDict
from numbers import Number
from typing import Union, Iterable

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.envs.utils import step_tensor_dict
from torchrl.modules import ProbabilisticTDModule, TDModule


class _context_manager:
    def __init__(self, value=True):
        self.value = value
        self.prev = []

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_context(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_context


def distance_loss(
    v1: torch.Tensor, v2: torch.Tensor, loss_function: str, strict_shape: bool = True
) -> torch.Tensor:
    """
    Computes a distance loss between two tensors.

    Args:
        v1 (Tensor): a tensor with a shape compatible with v2
        v2 (Tensor): a tensor with a shape compatible with v1
        loss_function (str): One of "l2", "l1" or "smooth_l1" representing which loss function is to be used.
        strict_shape (bool): if False, v1 and v2 are allowed to have a different shape.
            Default is True.

    Returns: A tensor of the shape v1.view_as(v2) or v2.view_as(v1) with values equal to the distance loss between the
        two.

    """
    if v1.shape != v2.shape and strict_shape:
        raise RuntimeError(
            f"The input tensors have shapes {v1.shape} and {v2.shape} which are incompatible."
        )

    if loss_function == "l2":
        value_loss = F.mse_loss(
            v1,
            v2,
            reduction="none",
        )

    elif loss_function == "l1":
        value_loss = F.l1_loss(
            v1,
            v2,
            reduction="none",
        )

    elif loss_function == "smooth_l1":
        value_loss = F.smooth_l1_loss(
            v1,
            v2,
            reduction="none",
        )
    else:
        raise NotImplementedError(f"Unknown loss {loss_function}")
    return value_loss


class ValueLoss:
    value_network: nn.Module
    target_value_network: nn.Module


class _TargetNetUpdate:
    """
    An abstract class for target network update in Double DQN/DDPG.

    Args:
        loss_module (DQNLoss or DDPGLoss): loss module where the target network should be updated.

    """

    def __init__(
        self,
        loss_module: Union["DQNLoss", "DDPGLoss", "SACLoss"],
    ):

        _target_names = []
        for name in loss_module.__dict__:
            if (
                name.startswith("_target_")
                and (name.endswith("params") or name.endswith("buffers"))
                and (getattr(loss_module, name) is not None)
            ):
                _target_names.append(name)

        _source_names = ["".join(name.split("_target_")) for name in _target_names]

        if not all(
            (name in loss_module.__dict__) or (name in loss_module._modules)
            for name in _source_names
        ):
            ex_list = [
                name
                for name in _source_names
                if not (
                    (name in loss_module.__dict__) or (name in loss_module._modules)
                )
            ]
            raise RuntimeError(
                f"Incongruent target and source parameter lists: "
                f"{ex_list} are not in "
                f"loss_module.__dict__"
            )

        self._targets = OrderedDict(
            {name: getattr(loss_module, name) for name in _target_names}
        )
        self._sources = OrderedDict(
            {name: getattr(loss_module, name) for name in _source_names}
        )
        self.initialized = False

    def init_(self) -> None:
        for source, target in zip(self._sources.values(), self._targets.values()):
            print("source: ", source)
            print("target: ", target)
            for p_source, p_target in zip(source, target):
                if p_target.requires_grad:
                    raise RuntimeError("the target parameter is part of a graph.")
                p_target.data.copy_(p_source.data)
        self.initialized = True

    def step(self) -> None:
        if not self.initialized:
            raise Exception(
                f"{self.__class__.__name__} must be "
                f"initialized (`{self.__class__.__name__}.init_()`) before calling step()"
            )

        for source, target in zip(self._sources.values(), self._targets.values()):
            for p_source, p_target in zip(source, target):
                if p_target.requires_grad:
                    raise RuntimeError("the target parameter is part of a graph.")
                self._step(p_source, p_target)

    def _step(self, p_source: Tensor, p_target: Tensor) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        string = (
            f"{self.__class__.__name__}(sources={[name for name in self._sources]}, targets="
            f"{[name for name in self._targets]})"
        )
        return string


class SoftUpdate(_TargetNetUpdate):
    """
    A soft-update class for target network update in Double DQN/DDPG.
    This was proposed in "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING", https://arxiv.org/pdf/1509.02971.pdf

    Args:
        loss_module (DQNLoss or DDPGLoss): loss module where the target network should be updated.
        eps (scalar): epsilon in the update equation:
            param = prev_param * eps + new_param * (1-eps)
            default: 0.999
    """

    def __init__(
        self, loss_module: Union["DQNLoss", "DDPGLoss", "SACLoss"], eps: Number = 0.999
    ):
        if not (eps < 1.0 and eps > 0.0):
            raise ValueError(
                f"Got eps = {eps} when it was supposed to be between 0 and 1."
            )
        super(SoftUpdate, self).__init__(loss_module)
        self.eps = eps

    def _step(self, p_source: Tensor, p_target: Tensor) -> None:
        p_target.data.copy_(p_target.data * self.eps + p_source.data * (1 - self.eps))


class HardUpdate(_TargetNetUpdate):
    """
    A hard-update class for target network update in Double DQN/DDPG (by contrast with soft updates).
    This was proposed in the original Double DQN paper: "Deep Reinforcement Learning with Double Q-learning",
    https://arxiv.org/abs/1509.06461.

    Args:
        loss_module (DQNLoss or DDPGLoss): loss module where the target network should be updated.
        value_network_update_interval (scalar): how often the target network should be updated.
            default: 1000
    """

    def __init__(
        self,
        loss_module: Union["DQNLoss", "DDPGLoss", "SACLoss"],
        value_network_update_interval: Number = 1000,
    ):
        super(HardUpdate, self).__init__(loss_module)
        self.value_network_update_interval = value_network_update_interval
        self.counter = 0

    def _step(self, p_source: Tensor, p_target: Tensor) -> None:
        if self.counter == self.value_network_update_interval:
            p_target.data.copy_(p_source.data)

    def step(self) -> None:
        super().step()
        if self.counter == self.value_network_update_interval:
            self.counter = 0
        else:
            self.counter += 1


class hold_out_net(_context_manager):
    def __init__(self, network: nn.Module) -> None:
        self.network = network
        try:
            self.p_example = next(network.parameters())
        except StopIteration as err:
            raise RuntimeError(
                "hold_out_net requires the network parameter set to be non-empty."
            )
        self._prev_state = []

    def __enter__(self) -> None:
        self._prev_state.append(self.p_example.requires_grad)
        self.network.requires_grad_(False)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.network.requires_grad_(self._prev_state.pop())

class hold_out_params(_context_manager):
    def __init__(self, params: Iterable[Tensor]) -> None:
        self.params = params

    def __enter__(self) -> None:
        return (p.detach() for p in self.params)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

@torch.no_grad()
def next_state_value(
    tensor_dict: _TensorDict,
    operator: TDModule,
    next_val_key: str = "state_action_value",
    gamma: Number = 0.99,
    **kwargs,
) -> torch.Tensor:
    """
    Computes the next state value (without gradient) to compute a target for the MSE loss
        L = Sum[ (q_value - target_value)^2 ]
    The target value is computed as
        r + gamma ** n_steps_to_next * value_next_state
    If the reward is the immediate reward, n_steps_to_next=1. If N-steps rewards are used, n_steps_to_next is gathered
    from the input tensordict.

    Args:
        tensor_dict (_TensorDict): Tensordict containing a reward and done key (and a n_steps_to_next key for n-steps
            rewards).
        operator (ProbabilisticTDModule): the value function operator. Should write a 'next_val_key' key-value in the
            input tensordict when called.
        next_val_key (str): key where the next value will be written.
            Default: 'state_action_value'
        gamma (Number): return discount rate.
            default: 0.99

    Returns:
        a Tensor of the size of the input tensordict containing the predicted value state.
    """
    try:
        steps_to_next_obs = tensor_dict.get("steps_to_next_obs").squeeze(-1)
    except KeyError:
        steps_to_next_obs = 1

    rewards = tensor_dict.get("reward").squeeze(-1)
    done = tensor_dict.get("done").squeeze(-1)
    next_td = step_tensor_dict(tensor_dict)  # next_observation -> observation
    next_td = next_td.select(*operator.in_keys)
    operator(next_td, **kwargs)
    pred_next_val_detach = next_td.get(next_val_key).squeeze(-1)
    done = done.to(torch.float)
    target_value = (1 - done) * pred_next_val_detach
    rewards = rewards.to(torch.float)
    target_value = rewards + (gamma ** steps_to_next_obs) * target_value
    return target_value
