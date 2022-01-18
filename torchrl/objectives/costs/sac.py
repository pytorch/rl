from copy import deepcopy
from numbers import Number
from typing import Tuple, Optional

import torch
from torch import Tensor

from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.modules import ProbabilisticOperator, Actor
from torchrl.modules.probabilistic_operators.actors import ActorCriticWrapper
from torchrl.objectives.costs.utils import hold_out_net, next_state_value, distance_loss

__all__ = ["SACLoss", "DoubleSACLoss"]


class SACLoss:
    """
    TorchRL implementation of the SAC loss, as presented in "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
    Reinforcement Learning with a Stochastic Actor" https://arxiv.org/pdf/1801.01290.pdf

    Args:
        actor_network (Actor): stochastic actor
        qvalue_network (ProbabilisticOperator): Q(s, a) parametric model
        value_network (ProbabilisticOperator): V(s) parametric model\
        qvalue_network_bis (ProbabilisticOperator, optional): if required, the Q-value can be computed twice
            independently using two separate networks. The minimum predicted value will then be used for inference.
        gamma (number): discount for return computation
        priority_key (str): tensordict key where to write the priority (for prioritized replay buffer usage).
    """

    def __init__(
            self,
            actor_network: Actor,
            qvalue_network: ProbabilisticOperator,
            value_network: ProbabilisticOperator,
            qvalue_network_bis: Optional[ProbabilisticOperator] = None,
            gamma: Number = 0.99,
            priotity_key: str = "td_error",
            loss_function: str = "smooth_l1"
    ) -> None:
        self.actor_network = actor_network
        self.qvalue_network = qvalue_network
        self.value_network = value_network
        self.qvalue_network_bis = qvalue_network_bis
        self.gamma = gamma
        self.priority_key = priotity_key
        self.loss_function = loss_function

    def parameters(self):
        for p in self.actor_network.parameters():
            yield p
        for p in self.qvalue_network.parameters():
            yield p
        if self.qvalue_network_bis is not None:
            for p in self.qvalue_network_bis.parameters():
                yield p
        for p in self.value_network.parameters():
            yield p

    @property
    def target_value_network(self):
        return self.value_network

    @property
    def target_qvalue_network(self):
        return self.qvalue_network

    @property
    def target_qvalue_network_bis(self):
        return self.qvalue_network_bis

    @property
    def target_actor_network(self):
        return self.actor_network

    @property
    def device(self) -> torch.device:
        for p in self.actor_network.parameters():
            return p.device
        for p in self.qvalue_network.parameters():
            return p.device
        for p in self.value_network.parameters():
            return p.device
        raise RuntimeError("At least one of the networks of SACLoss must have trainable parameters.")

    def __call__(self, tensordict: _TensorDict) -> Tuple[Tensor, Tensor, Tensor]:
        device = self.device
        td_device = tensordict.to(device)

        actor_loss = self._actor_loss(td_device)
        qvalue_loss, priority = self._qvalue_loss(td_device)
        value_loss = self._value_loss(td_device)
        tensordict.set(
            self.priority_key,
            priority,
            inplace=True)
        if (actor_loss.shape != qvalue_loss.shape) or (actor_loss.shape != value_loss.shape):
            raise RuntimeError(f"Losses shape mismatch: {actor_loss.shape}, {qvalue_loss.shape} and {value_loss.shape}")
        return actor_loss.mean(), qvalue_loss.mean(), value_loss.mean()

    def _actor_loss(self, tensordict: _TensorDict) -> Tensor:
        # KL lossa
        dist = self.actor_network.get_dist(tensordict)[0]
        a_reparm = dist.rsample()
        log_prob = dist.log_prob(a_reparm)

        ## TODO: assess if copmuting the q value using no_grad and writing a custom autograd.Function operator
        # works faster
        if self.target_qvalue_network_bis is not None:
            qval_nets = (self.target_qvalue_network_bis, self.target_qvalue_network)
        else:
            qval_nets = (self.target_qvalue_network,)

        q_logprob = []
        for qval_net in qval_nets:
            with hold_out_net(qval_net):
                td_q = tensordict.select(*self.qvalue_network.in_keys)
                td_q.set("action", a_reparm, inplace=False)
                q_logprob.append(
                    qval_net(td_q).get("state_action_value").squeeze(-1)
                )

        q_logprob = torch.stack(q_logprob, 0).min(dim=0)[0]
        if log_prob.shape != q_logprob.shape:
            raise RuntimeError(f"Losses shape mismatch: {log_prob.shape} and {q_logprob.shape}")

        return log_prob - q_logprob

    def _qvalue_loss(self, tensordict: _TensorDict) -> Tuple[Tensor, Tensor]:
        actor_critic = ActorCriticWrapper(self.target_actor_network, self.target_value_network)
        target_value = next_state_value(tensordict, actor_critic, gamma=self.gamma, next_val_key="state_value")

        # value loss
        if self.qvalue_network_bis is not None:
            nets = (self.qvalue_network_bis, self.qvalue_network)
        else:
            nets = (self.qvalue_network,)

        # Q-nets must be trained independently: as such, we split the data in 2 if required and train each q-net on
        # one half of the data.
        tensordict_chunks = tensordict.chunk(len(nets), dim=0)
        target_chunks = target_value.chunk(len(nets), dim=0)
        value_loss = []
        for _td, _target, _net in zip(tensordict_chunks, target_chunks, nets):
            td_copy = _td.select(*_net.in_keys).detach()
            _net(td_copy)
            pred_val = td_copy.get("state_action_value").squeeze(-1)
            value_loss.append(distance_loss(pred_val, _target, loss_function=self.loss_function))
        value_loss = torch.cat(value_loss, 0)

        return value_loss, value_loss

    def _value_loss(self, tensordict: _TensorDict) -> Tensor:
        # value loss
        td_copy = tensordict.select(*self.value_network.in_keys).detach()
        self.value_network(td_copy)
        pred_val = td_copy.get("state_value").squeeze(-1)

        with hold_out_net(self.target_actor_network):
            action_dist = self.target_actor_network.get_dist(td_copy)[0]  # resample an action
            action = action_dist.rsample()
            td_copy.set("action", action, inplace=False)

            if self.target_qvalue_network_bis is not None:
                qval_nets = (self.target_qvalue_network_bis, self.target_qvalue_network)
            else:
                qval_nets = (self.target_qvalue_network,)

            qval = []
            for qval_net in qval_nets:
                with hold_out_net(qval_net):
                    qval.append(qval_net(td_copy).get("state_action_value").squeeze(-1))
            qval = torch.stack(qval, 0).min(dim=0)[0]

            log_p = action_dist.log_prob(action)
            if log_p.shape != qval.shape:
                raise RuntimeError(f"Losses shape mismatch: {qval.shape} and {log_p.shape}")
            target_val = qval - log_p

        value_loss = distance_loss(pred_val, target_val, loss_function=self.loss_function)
        return value_loss


class DoubleSACLoss(SACLoss):
    """
    A Double SAC loss class.
    As for Double DDPG/DQN losses, this class separates the target critic/value/actor networks from the
    critic/value/actor networks used for data collection. Those target networks should be updated from their original
    counterparts with some delay using dedicated classes (SoftUpdate and HardUpdate in objectives.cost.utils).
    Note that the original networks will be copied at initialization using the copy.deepcopy method: in some rare cases
    this may lead to unexpected behaviours (for instance if the networks change in a way that won't be reflected by their
    state_dict). Please report any such bug if encountered.

    """

    def __init__(self, *args, delay_actor=False, delay_qvalue=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_qvalue = delay_qvalue
        if delay_qvalue:
            self._target_qvalue_network = deepcopy(self.qvalue_network)
            self._target_qvalue_network.requires_grad_(False)
            if self.qvalue_network_bis is not None:
                raise RuntimeError("qvalue_network_bis must be None if a separate qvalue target network has to be used")
        self._target_value_network = deepcopy(self.value_network)
        self._target_value_network.requires_grad_(False)
        self.delay_actor = delay_actor
        if delay_actor:
            self._target_actor_network = deepcopy(self.actor_network)
            self._target_actor_network.requires_grad_(False)

    @property
    def target_value_network(self):
        return self._target_value_network

    @property
    def target_qvalue_network(self):
        if self.delay_qvalue:
            return self._target_qvalue_network
        return self.qvalue_network

    @property
    def target_actor_network(self):
        if self.delay_actor:
            return self._target_actor_network
        return self.actor_network
