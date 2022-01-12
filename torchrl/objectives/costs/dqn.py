from copy import deepcopy
from numbers import Number
from typing import Tuple, Optional
from uuid import uuid1
from warnings import warn

import torch

from torchrl.data import TensorDict
from torchrl.envs.utils import step_tensor_dict
from torchrl.modules import QValueActor, DistributionalQValueActor, reset_noise, ProbabilisticOperator
from .utils import distance_loss

__all__ = [
    "DQNLoss",
    "DoubleDQNLoss",
    "DistributionalDQNLoss",
    "DistributionalDoubleDQNLoss",
]

from ...data.tensordict.tensordict import _TensorDict

from ...data.utils import DEVICE_TYPING


class DQNLoss:
    """
    The DQN Loss class.
    Args:
        value_network (ProbabilisticOperator): a Q value operator.
        gamma (scalar): a discount factor for return computation.
        device (str, int or torch.device, optional): a device where the losses will be computed, if it can't be found
            via the Q value operator.
        loss_function (str): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".
    """

    def __init__(self, value_network: QValueActor,
                 gamma: Number, device: Optional[DEVICE_TYPING] = None,
                 loss_function: str = "l2"):
        self.value_network = value_network
        self.value_network_in_keys = value_network.in_keys
        if not isinstance(value_network, QValueActor):
            raise TypeError(f"DQNLoss requires value_network to be of QValueActor dtype, got {type(value_network)}")
        self.gamma = gamma
        self.loss_function = loss_function
        if device is None:
            try:
                device = next(value_network.parameters()).device
            except:
                # value_network does not have params, use obs
                device = None
        self.device = device

    def __call__(self, input_tensor_dict: _TensorDict) -> torch.Tensor:
        """
        Computes the DQN loss given a tensordict sampled from the replay buffer.
        This function will also write a "td_error" key that can be used by prioritized replay buffers to assign
            a priority to items in the tensordict.

        Args:
            input_tensor_dict (_TensorDict): a tensordict with keys ["done", "reward", "action"] and the in_keys of
                the value network.

        Returns: a tensor containing the DQN loss.

        """

        value_network, target_value_network = self._get_networks()
        device = self.device if self.device is not None else input_tensor_dict.device
        tensor_dict = input_tensor_dict.to(device)
        if tensor_dict.device != device:
            raise RuntimeError(f"device {device} was expected for "
                               f"{tensor_dict.__class__.__name__} but {tensor_dict.device} was found")
        for k, t in tensor_dict.items():
            if t.device != device:
                raise RuntimeError(f"found key value pair {k}-{t.shape} "
                                   f"with device {t.device} when {device} was required")
        action = tensor_dict.get("action")
        done = tensor_dict.get("done").squeeze(-1)
        rewards = tensor_dict.get("reward").squeeze(-1)

        gamma = self.gamma

        action = action.to(torch.float)
        td_copy = tensor_dict.clone()
        if td_copy.device != tensor_dict.device:
            raise RuntimeError(f"{tensor_dict} and {td_copy} have different devices")
        value_network(td_copy)
        pred_val = td_copy.get("action_value")
        pred_val_index = (pred_val * action).sum(-1)
        try:
            steps_to_next_obs = tensor_dict.get("steps_to_next_obs").squeeze(-1)
        except:
            steps_to_next_obs = 1

        with torch.no_grad():
            next_td = step_tensor_dict(tensor_dict)
            target_value_network(next_td)
            next_action = next_td.get("action")
            pred_next_val_detach = next_td.get("action_value")

            done = done.to(torch.float)
            target_value = (1 - done) * (pred_next_val_detach * next_action).sum(-1)
            rewards = rewards.to(torch.float)
            target_value = rewards + (gamma ** steps_to_next_obs) * target_value

        input_tensor_dict.set("td_error",
                              abs(pred_val_index - target_value).detach().unsqueeze(-1).to(input_tensor_dict.device),
                              inplace=True)
        loss = distance_loss(pred_val_index, target_value, self.loss_function)
        return loss.mean()

    def _get_networks(self) -> Tuple[ProbabilisticOperator, ProbabilisticOperator]:
        value_network = self.value_network
        target_value_network = self.value_network
        return value_network, target_value_network

    @property
    def target_value_network(self) -> ProbabilisticOperator:
        return self._get_networks()[1]


class DoubleDQNLoss(DQNLoss):
    """
    A Double DQN loss class.
    This class duplicates the value network into a new target value network, which differs from the value networks used
    for data collection in that it has a similar weight configuration but delayed of a certain number of optimization
    steps. The target network should be updated from its original counterpart with some delay using dedicated classes
    (SoftUpdate and HardUpdate in objectives.cost.utils).
    More information on double DQN can be found in "Deep Reinforcement Learning with Double Q-learning",
    https://arxiv.org/abs/1509.06461.

    Note that the original network will be copied at initialization using the copy.deepcopy method: in some rare cases
    this may lead to unexpected behaviours (for instance if the network changes in a way that won't be reflected by its
    state_dict). Please report any such bug if encountered.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_value_network = deepcopy(self.value_network)
        self._target_value_network.requires_grad_(False)

    def _get_networks(self) -> Tuple[ProbabilisticOperator, ProbabilisticOperator]:
        value_network = self.value_network
        target_value_network = self._target_value_network
        return value_network, target_value_network


class DistributionalDQNLoss:
    """
    A distributional DQN loss class.
    Distributional DQN uses a value network that outputs a distribution of values over a discrete support of discounted
    returns (unlike regular DQN where the value network outputs a single point prediction of the disctounted return).

    For more details regarding Distributional DQN, refer to "A Distributional Perspective on Reinforcement Learning",
    https://arxiv.org/pdf/1707.06887.pdf

    Args:
        value_network (DistributionalQValueActor): the distributional Q value operator.
        gamma (scalar): a discount factor for return computation.
        device (str, int or torch.device, optional): a device where the losses will be computed, if it can't be found
            via the value operator.
    """
    def __init__(self, value_network: DistributionalQValueActor, gamma: Number, device: Optional[DEVICE_TYPING] = None):
        self.gamma = gamma
        if not isinstance(value_network, DistributionalQValueActor):
            raise TypeError("Expected value_network to be of type DistributionalQValueActor "
                            f"but got {type(value_network)}")
        self.value_network = value_network
        if device is None:
            try:
                device = next(value_network.parameters()).device
            except AttributeError:
                # value_network does not have params, use obs
                device = None
        self.device = device

    def __call__(self, input_tensor_dict: _TensorDict) -> torch.Tensor:
        # from https://github.com/Kaixhin/Rainbow/blob/9ff5567ad1234ae0ed30d8471e8f13ae07119395/agent.py
        device = self.device
        tensor_dict = TensorDict(source=input_tensor_dict, batch_size=input_tensor_dict.batch_size).to(device)
        value_network, target_value_network = self._get_networks()
        if tensor_dict.batch_dims != 1:
            raise RuntimeError(f"{self.__class__.__name___} expects a 1-dimensional tensor_dict as input")
        batch_size = tensor_dict.batch_size[0]
        support = value_network.support
        atoms = support.numel()
        Vmin = support.min().item()
        Vmax = support.max().item()
        delta_z = (Vmax - Vmin) / (atoms - 1)

        action = tensor_dict.get("action")
        reward = tensor_dict.get("reward")
        done = tensor_dict.get("done")

        try:
            steps_to_next_obs = tensor_dict.get("steps_to_next_obs")
        except:
            steps_to_next_obs = 1
        discount = self.gamma ** steps_to_next_obs

        # Calculate current state probabilities (online network noise already sampled)
        td_clone = tensor_dict.clone()
        value_network(td_clone)  # Log probabilities log p(s_t, ·; θonline)
        action_log_softmax = td_clone.get("action_value")
        action_expand = action.unsqueeze(-2).expand_as(action_log_softmax)
        log_ps_a = action_log_softmax.masked_select(action_expand.to(torch.bool))
        log_ps_a = log_ps_a.view(batch_size, atoms)  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate nth next state probabilities
            next_td = step_tensor_dict(tensor_dict)
            value_network(next_td)  # Probabilities p(s_t+n, ·; θonline)
            argmax_indices_ns = next_td.get("action").argmax(-1)  # one-hot encoding

            self._reset_noise_target_net()
            target_value_network(next_td)  # Probabilities p(s_t+n, ·; θtarget)
            pns = next_td.get("action_value").exp()
            # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            pns_a = pns[range(batch_size), :, argmax_indices_ns]

            # Compute Tz (Bellman operator T applied to z)
            ## Tz = R^n + (γ^n)z (accounting for terminal states)
            if isinstance(discount, torch.Tensor):
                discount = discount.to('cpu')
            done = done.to('cpu')
            reward = reward.to('cpu')
            support = support.to('cpu')
            pns_a = pns_a.to('cpu')
            Tz = reward + (1 - done.to(reward.dtype)) * discount * support
            if Tz.shape != torch.Size([batch_size, atoms]):
                raise RuntimeError("Tz shape must be torch.Size([batch_size, atoms]), "
                                   f"got Tz.shape={Tz.shape} and batch_size={batch_size}, atoms={atoms}")
            ## Clamp between supported values
            Tz = Tz.clamp_(min=Vmin, max=Vmax)
            if not torch.isfinite(Tz).all():
                raise RuntimeError("Tz has some non-finite elements")
            ## Compute L2 projection of Tz onto fixed support z
            b = (Tz - Vmin) / delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) & (l == u)] -= 1
            u[(l < (atoms - 1)) & (l == u)] += 1

            # Distribute probability of Tz
            m = torch.zeros(batch_size, atoms)
            offset = (
                torch.linspace(
                    0,
                    ((batch_size - 1) * atoms),
                    batch_size,
                    dtype=torch.int64,
                    # device=device,
                )
                    .unsqueeze(1)
                    .expand(batch_size, atoms)
            )
            try:
                index = (l + offset).view(-1)
                tensor = (pns_a * (u.float() - b)).view(-1)
                # m_l = m_l + p(s_t+n, a*)(u - b)
                m.view(-1).index_add_(0, index, tensor)
                index = (u + offset).view(-1)
                tensor = (pns_a * (b - l.float())).view(-1)
                # m_u = m_u + p(s_t+n, a*)(b - l)
                m.view(-1).index_add_(0, index, tensor)
            except:
                print('Distributional dqn raised an error when computing target distribution')
                file = '_'.join(['dddqn', 'error', str(uuid1())]) + '.t'
                print(f'Saving tensors in {file}')
                torch.save({'index': index, 'tensor': tensor, 'm': m, 'reward': reward, 'done': done, 'pns_a': pns_a,
                            'discount': discount, 'Tz': Tz, }, file)

        # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        loss = -torch.sum(m.to(device) * log_ps_a, 1)
        input_tensor_dict.set("td_error", loss.detach().unsqueeze(1).to(input_tensor_dict.device), inplace=True)
        loss = loss.mean()
        return loss

    def _reset_noise_target_net(self) -> None:
        pass

    def _get_networks(self) -> Tuple[ProbabilisticOperator, ProbabilisticOperator]:
        value_network = self.value_network
        return value_network, value_network


class DistributionalDoubleDQNLoss(DistributionalDQNLoss):
    """
    A distributional, double DQN loss class.
    This class mixes distributional and double DQN losses.

    For more details regarding Distributional DQN, refer to "A Distributional Perspective on Reinforcement Learning",
    https://arxiv.org/pdf/1707.06887.pdf
    More information on double DQN can be found in "Deep Reinforcement Learning with Double Q-learning",
    https://arxiv.org/abs/1509.06461.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_value_network = deepcopy(self.value_network)
        self._target_value_network.requires_grad_(False)
        self.counter = 0

    @property
    def target_value_network(self) -> ProbabilisticOperator:
        return self._get_networks()[1]

    def step(self) -> None:
        if self.counter == self.value_network_update_interval:
            self.counter = 0
            print("updating target value network")
            self.target_value_network.load_state_dict(self.value_network.state_dict())
        else:
            self.counter += 1

    def _get_networks(self) -> Tuple[ProbabilisticOperator, ProbabilisticOperator]:
        value_network = self.value_network
        try:
            self._target_value_network.apply(reset_noise)
        except:
            warn("reset noise failed")
        target_value_network = self._target_value_network
        return value_network, target_value_network

    def _reset_noise_target_net(self) -> None:
        self.target_value_network.apply(reset_noise)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return super().__call__(*args, **kwargs)
