from copy import deepcopy
from numbers import Number
from typing import Optional, Union

import torch

from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.envs.utils import step_tensor_dict
from torchrl.modules import ProbabilisticOperator
from torchrl.objectives.costs.utils import distance_loss


class DDPGLoss:
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
        self.loss_type = loss_function

        if device is None:
            try:
                device = next(value_network.parameters()).device
            except:
                # value_network does not have params, use obs
                device = None
        self.device = device

    def _get_networks(self):
        actor_network = self.actor_network
        value_network = self.value_network
        target_actor_network = self.actor_network
        target_value_network = self.value_network
        return actor_network, value_network, target_actor_network, target_value_network

    def __call__(self, input_tensor_dict: _TensorDict):
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
            done,
            rewards,
            gamma)
        input_tensor_dict.set(
            "td_error",
            abs(td_error.detach().unsqueeze(1).to(input_tensor_dict.device)),
            inplace=True)
        actor_loss = self._actor_loss(tensor_dict_device, actor_network, value_network)
        return actor_loss.mean(), value_loss.mean()

    @property
    def target_value_network(self):
        return self._get_networks()[-1]

    @property
    def target_actor_network(self):
        return self._get_networks()[-2]

    def _actor_loss(self, tensor_dict: _TensorDict, actor_network: ProbabilisticOperator,
                    value_network: ProbabilisticOperator):
        td_copy = tensor_dict.select(*self.actor_in_keys).detach()
        td_copy = actor_network(td_copy)
        rg_status = next(value_network.parameters()).requires_grad
        value_network.requires_grad_(False)
        td_copy = value_network(td_copy)
        value_network.requires_grad_(rg_status)
        return -td_copy.get("state_action_value")

    def _value_loss(self, tensor_dict: _TensorDict, value_network: ProbabilisticOperator,
                    target_actor_network: ProbabilisticOperator,
                    target_value_network: ProbabilisticOperator, done: torch.Tensor,
                    rewards: torch.Tensor, gamma: Number):
        # value loss
        td_copy = tensor_dict.select(*value_network.in_keys).detach()
        value_network(td_copy)
        pred_val = td_copy.get("state_action_value").squeeze(-1)
        try:
            steps_to_next_obs = tensor_dict.get("steps_to_next_obs").squeeze(-1)
        except:
            steps_to_next_obs = 1

        with torch.no_grad():
            next_td = step_tensor_dict(tensor_dict)
            next_td = next_td.select(*target_actor_network.in_keys)
            target_actor_network(next_td)
            target_value_network(next_td)
            pred_next_val_detach = next_td.get("state_action_value").squeeze(-1)
            done = done.to(torch.float)
            target_value = (1 - done) * pred_next_val_detach
            rewards = rewards.to(torch.float)
            target_value = rewards + (gamma ** steps_to_next_obs) * target_value

        td_error = pred_val - target_value
        value_loss = distance_loss(pred_val, target_value, loss_type=self.loss_type)

        return value_loss, td_error


class DoubleDDPGLoss(DDPGLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_value_network = deepcopy(self.value_network)
        self._target_value_network.requires_grad_(False)
        self._target_actor_network = deepcopy(self.actor_network)
        self._target_actor_network.requires_grad_(False)

    def _get_networks(self):
        actor_network = self.actor_network
        value_network = self.value_network
        target_actor_network = self._target_actor_network
        target_value_network = self._target_value_network
        return actor_network, value_network, target_actor_network, target_value_network
