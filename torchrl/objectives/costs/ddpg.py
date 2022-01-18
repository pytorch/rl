from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Optional, Union, Tuple

import torch

from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.envs.utils import step_tensor_dict
from torchrl.modules import ProbabilisticOperator
from torchrl.modules.probabilistic_operators.actors import ActorCriticWrapper
from torchrl.objectives.costs.utils import distance_loss, next_state_value


class DDPGLoss:
    """
    The DDPG Loss class.
    Args:
        actor_network (ProbabilisticOperator): a policy operator.
        value_network (ProbabilisticOperator): a Q value operator.
        gamma (scalar): a discount factor for return computation.
        device (str, int or torch.device, optional): a device where the losses will be computed, if it can't be found
            via the value operator.
        loss_function (str): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".
    """

    def __init__(self,
                 actor_network: ProbabilisticOperator,
                 value_network: ProbabilisticOperator,
                 gamma: Number,
                 device: Optional[Union[str, int, torch.device]] = None,
                 loss_function: str = "l2",
                 ):

        self.value_network = value_network
        self.actor_network = actor_network
        self.actor_in_keys = actor_network.in_keys

        self.gamma = gamma
        self.loss_funtion = loss_function

        if device is None:
            try:
                device = next(value_network.parameters()).device
            except:
                # value_network does not have params, use obs
                device = None
        self.device = device

    def _get_networks(self) -> \
            Tuple[ProbabilisticOperator, ProbabilisticOperator, ProbabilisticOperator, ProbabilisticOperator]:
        actor_network = self.actor_network
        value_network = self.value_network
        target_actor_network = self.actor_network
        target_value_network = self.value_network
        return actor_network, value_network, target_actor_network, target_value_network

    def __call__(self, input_tensor_dict: _TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the DDPG losses given a tensordict sampled from the replay buffer.
        This function will also write a "td_error" key that can be used by prioritized replay buffers to assign
            a priority to items in the tensordict.

        Args:
            input_tensor_dict (_TensorDict): a tensordict with keys ["done", "reward"] and the in_keys of the actor
                and value networks.

        Returns: a tuple of 2 tensors containing the DDPG loss.

        """
        actor_network, value_network, target_actor_network, target_value_network = self._get_networks()
        device = self.device if self.device is not None else input_tensor_dict.device
        tensor_dict_device = input_tensor_dict.to(device)

        done = tensor_dict_device.get("done").squeeze(-1)
        rewards = tensor_dict_device.get("reward").squeeze(-1)

        gamma = self.gamma

        value_loss, td_error = self._value_loss(
            tensor_dict_device,
            value_network,
            target_actor_network,
            target_value_network,
            gamma)
        input_tensor_dict.set(
            "td_error",
            abs(td_error.detach().unsqueeze(1).to(input_tensor_dict.device)),
            inplace=True)
        actor_loss = self._actor_loss(tensor_dict_device, actor_network, value_network)
        return actor_loss.mean(), value_loss.mean()

    @property
    def target_value_network(self) -> ProbabilisticOperator:
        return self._get_networks()[-1]

    @property
    def target_actor_network(self) -> ProbabilisticOperator:
        return self._get_networks()[-2]

    def _actor_loss(self, tensor_dict: _TensorDict, actor_network: ProbabilisticOperator,
                    value_network: ProbabilisticOperator) -> torch.Tensor:
        td_copy = tensor_dict.select(*self.actor_in_keys).detach()
        td_copy = actor_network(td_copy)
        rg_status = next(value_network.parameters()).requires_grad
        value_network.requires_grad_(False)
        td_copy = value_network(td_copy)
        value_network.requires_grad_(rg_status)
        return -td_copy.get("state_action_value")

    def _value_loss(self, tensor_dict: _TensorDict, value_network: ProbabilisticOperator,
                    target_actor_network: ProbabilisticOperator,
                    target_value_network: ProbabilisticOperator,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # value loss
        td_copy = tensor_dict.select(*value_network.in_keys).detach()
        value_network(td_copy)
        pred_val = td_copy.get("state_action_value").squeeze(-1)

        actor_critic = ActorCriticWrapper(target_actor_network, target_value_network)
        target_value = next_state_value(tensor_dict, actor_critic, gamma=self.gamma)

        # td_error = pred_val - target_value
        value_loss = distance_loss(pred_val, target_value, loss_function=self.loss_funtion)

        return value_loss, value_loss


class DoubleDDPGLoss(DDPGLoss):
    """
    A Double DDPG loss class.
    As for Double DQN loss, this class separates the target value/actor networks from the value/actor networks used for
    data collection. Those target networks should be updated from their original counterparts with some delay using
    dedicated classes (SoftUpdate and HardUpdate in objectives.cost.utils).
    Note that the original networks will be copied at initialization using the copy.deepcopy method: in some rare cases
    this may lead to unexpected behaviours (for instance if the networks change in a way that won't be reflected by their
    state_dict). Please report any such bug if encountered.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_value_network = deepcopy(self.value_network)
        self._target_value_network.requires_grad_(False)
        self._target_actor_network = deepcopy(self.actor_network)
        self._target_actor_network.requires_grad_(False)

    def _get_networks(self) -> \
            Tuple[ProbabilisticOperator, ProbabilisticOperator, ProbabilisticOperator, ProbabilisticOperator]:
        actor_network = self.actor_network
        value_network = self.value_network
        target_actor_network = self._target_actor_network
        target_value_network = self._target_value_network
        return actor_network, value_network, target_actor_network, target_value_network
