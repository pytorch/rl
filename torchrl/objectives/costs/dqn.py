# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchrl.data import TensorDict
from torchrl.envs.utils import step_tensor_dict
from torchrl.modules import (
    DistributionalQValueActor,
    QValueActor,
)
from ...data.tensordict.tensordict import _TensorDict
from .common import _LossModule
from .utils import distance_loss, next_state_value

__all__ = [
    "DQNLoss",
    "DoubleDQNLoss",
    "DistributionalDQNLoss",
    "DistributionalDoubleDQNLoss",
]


class DQNLoss(_LossModule):
    """
    The DQN Loss class.
    Args:
        value_network (ProbabilisticTDModule): a Q value operator.
        gamma (scalar): a discount factor for return computation.
        loss_function (str): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".
    """

    delay_value: bool = False

    def __init__(
        self,
        value_network: QValueActor,
        gamma: float,
        loss_function: str = "l2",
        priority_key: str = "td_error",
    ) -> None:

        super().__init__()
        self.convert_to_functional(
            value_network,
            "value_network",
            create_target_params=self.delay_value,
        )

        self.value_network_in_keys = value_network.in_keys
        if not isinstance(value_network, QValueActor):
            raise TypeError(
                f"DQNLoss requires value_network to be of QValueActor dtype, got {type(value_network)}"
            )
        self.gamma = gamma
        self.loss_function = loss_function
        self.priority_key = priority_key

    def forward(self, input_tensor_dict: _TensorDict) -> TensorDict:
        """
        Computes the DQN loss given a tensordict sampled from the replay buffer.
        This function will also write a "td_error" key that can be used by prioritized replay buffers to assign
            a priority to items in the tensordict.

        Args:
            input_tensor_dict (_TensorDict): a tensordict with keys ["done", "reward", "action"] and the in_keys of
                the value network.

        Returns:
            a tensor containing the DQN loss.

        """

        device = self.device if self.device is not None else input_tensor_dict.device
        tensor_dict = input_tensor_dict.to(device)
        if tensor_dict.device != device:
            raise RuntimeError(
                f"device {device} was expected for "
                f"{tensor_dict.__class__.__name__} but {tensor_dict.device} was found"
            )

        for k, t in tensor_dict.items():
            if t.device != device:
                raise RuntimeError(
                    f"found key value pair {k}-{t.shape} "
                    f"with device {t.device} when {device} was required"
                )

        action = tensor_dict.get("action")

        action = action.to(torch.float)
        td_copy = tensor_dict.clone()
        if td_copy.device != tensor_dict.device:
            raise RuntimeError(f"{tensor_dict} and {td_copy} have different devices")
        self.value_network(
            td_copy,
            params=self.value_network_params,
            buffers=self.value_network_buffers,
        )

        pred_val = td_copy.get("action_value")
        pred_val_index = (pred_val * action).sum(-1)

        with torch.no_grad():
            target_value = next_state_value(
                tensor_dict,
                self.value_network,
                gamma=self.gamma,
                params=self.target_value_network_params,
                buffers=self.target_value_network_buffers,
                next_val_key="chosen_action_value",
            )
        priority_tensor = abs(pred_val_index - target_value)
        priority_tensor = priority_tensor.detach().unsqueeze(-1)
        priority_tensor = priority_tensor.to(input_tensor_dict.device)

        input_tensor_dict.set(
            self.priority_key,
            priority_tensor,
            inplace=True,
        )
        loss = distance_loss(pred_val_index, target_value, self.loss_function)
        return TensorDict({"loss": loss.mean()}, [])


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

    delay_value: bool = True


class DistributionalDQNLoss(_LossModule):
    """
    A distributional DQN loss class.
    Distributional DQN uses a value network that outputs a distribution of
    values over a discrete support of discounted returns (unlike regular DQN
    where the value network outputs a single point prediction of the
    disctounted return).

    For more details regarding Distributional DQN, refer to "A Distributional
    Perspective on Reinforcement Learning",
    https://arxiv.org/pdf/1707.06887.pdf

    Args:
        value_network (DistributionalQValueActor): the distributional Q
            value operator.
        gamma (scalar): a discount factor for return computation.
    """

    delay_value: bool = False

    def __init__(
        self,
        value_network: DistributionalQValueActor,
        gamma: float,
        priority_key: str = "td_error",
    ):
        super().__init__()
        self.gamma = gamma
        self.priority_key = priority_key
        if not isinstance(value_network, DistributionalQValueActor):
            raise TypeError(
                "Expected value_network to be of type "
                "DistributionalQValueActor "
                f"but got {type(value_network)}"
            )
        self.convert_to_functional(
            value_network,
            "value_network",
            create_target_params=self.delay_value,
        )

    def forward(self, input_tensor_dict: _TensorDict) -> TensorDict:
        # from https://github.com/Kaixhin/Rainbow/blob/9ff5567ad1234ae0ed30d8471e8f13ae07119395/agent.py
        device = self.device
        tensor_dict = TensorDict(
            source=input_tensor_dict, batch_size=input_tensor_dict.batch_size
        ).to(device)

        if tensor_dict.batch_dims != 1:
            raise RuntimeError(
                f"{self.__class__.__name___} expects a 1-dimensional "
                "tensor_dict as input"
            )
        batch_size = tensor_dict.batch_size[0]
        support = self.value_network.support
        atoms = support.numel()
        Vmin = support.min().item()
        Vmax = support.max().item()
        delta_z = (Vmax - Vmin) / (atoms - 1)

        action = tensor_dict.get("action")
        reward = tensor_dict.get("reward")
        done = tensor_dict.get("done")

        steps_to_next_obs = tensor_dict.get("steps_to_next_obs", 1)
        discount = self.gamma ** steps_to_next_obs

        # Calculate current state probabilities (online network noise already
        # sampled)
        td_clone = tensor_dict.clone()
        self.value_network(
            td_clone,
            params=self.value_network_params,
            buffers=self.value_network_buffers,
        )  # Log probabilities log p(s_t, ·; θonline)
        action_log_softmax = td_clone.get("action_value")
        action_expand = action.unsqueeze(-2).expand_as(action_log_softmax)
        log_ps_a = action_log_softmax.masked_select(action_expand.to(torch.bool))
        log_ps_a = log_ps_a.view(batch_size, atoms)  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate nth next state probabilities
            next_td = step_tensor_dict(tensor_dict)
            self.value_network(
                next_td,
                params=self.value_network_params,
                buffers=self.value_network_buffers,
            )  # Probabilities p(s_t+n, ·; θonline)
            argmax_indices_ns = next_td.get("action").argmax(-1)  # one-hot encoding

            self.value_network(
                next_td,
                params=self.target_value_network_params,
                buffers=self.target_value_network_buffers,
            )  # Probabilities p(s_t+n, ·; θtarget)
            pns = next_td.get("action_value").exp()
            # Double-Q probabilities
            # p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            pns_a = pns[range(batch_size), :, argmax_indices_ns]

            # Compute Tz (Bellman operator T applied to z)
            # Tz = R^n + (γ^n)z (accounting for terminal states)
            if isinstance(discount, torch.Tensor):
                discount = discount.to("cpu")
            done = done.to("cpu")
            reward = reward.to("cpu")
            support = support.to("cpu")
            pns_a = pns_a.to("cpu")
            Tz = reward + (1 - done.to(reward.dtype)) * discount * support
            if Tz.shape != torch.Size([batch_size, atoms]):
                raise RuntimeError(
                    "Tz shape must be torch.Size([batch_size, atoms]), "
                    f"got Tz.shape={Tz.shape} and batch_size={batch_size}, "
                    f"atoms={atoms}"
                )
            # Clamp between supported values
            Tz = Tz.clamp_(min=Vmin, max=Vmax)
            if not torch.isfinite(Tz).all():
                raise RuntimeError("Tz has some non-finite elements")
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - Vmin) / delta_z  # b = (Tz - Vmin) / Δz
            low, up = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            low[(up > 0) & (low == up)] -= 1
            up[(low < (atoms - 1)) & (low == up)] += 1

            # Distribute probability of Tz
            m = torch.zeros(batch_size, atoms)
            offset = torch.linspace(
                0,
                ((batch_size - 1) * atoms),
                batch_size,
                dtype=torch.int64,
                # device=device,
            )
            offset = offset.unsqueeze(1).expand(batch_size, atoms)
            index = (low + offset).view(-1)
            tensor = (pns_a * (up.float() - b)).view(-1)
            # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, index, tensor)
            index = (up + offset).view(-1)
            tensor = (pns_a * (b - low.float())).view(-1)
            # m_u = m_u + p(s_t+n, a*)(b - l)
            m.view(-1).index_add_(0, index, tensor)

        # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        loss = -torch.sum(m.to(device) * log_ps_a, 1)
        input_tensor_dict.set(
            self.priority_key,
            loss.detach().unsqueeze(1).to(input_tensor_dict.device),
            inplace=True,
        )
        loss_td = TensorDict({"loss": loss.mean()}, [])
        return loss_td


class DistributionalDoubleDQNLoss(DistributionalDQNLoss):
    """
    A distributional, double DQN loss class.
    This class mixes distributional and double DQN losses.

    For more details regarding Distributional DQN, refer to "A Distributional
    Perspective on Reinforcement Learning",
    https://arxiv.org/pdf/1707.06887.pdf
    More information on double DQN can be found in "Deep Reinforcement
    Learning with Double Q-learning", https://arxiv.org/abs/1509.06461.

    """

    delay_value: bool = True
