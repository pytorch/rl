import math
from copy import deepcopy
from numbers import Number
from typing import Tuple, Optional, Iterator, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter

from torchrl.data.tensordict.tensordict import _TensorDict, TensorDict
from torchrl.modules import ProbabilisticTDModule, Actor, reset_noise
from torchrl.modules.td_module.actors import ActorCriticWrapper
from torchrl.objectives.costs.utils import hold_out_net, next_state_value, distance_loss
from .common import _LossModule

__all__ = ["SACLoss", "DoubleSACLoss"]


class SACLoss(_LossModule):
    """
    TorchRL implementation of the SAC loss, as presented in "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
    Reinforcement Learning with a Stochastic Actor" https://arxiv.org/pdf/1801.01290.pdf

    Args:
        actor_network (Actor): stochastic actor
        qvalue_network (ProbabilisticTDModule): Q(s, a) parametric model
        value_network (ProbabilisticTDModule): V(s) parametric model\
        qvalue_network_bis (ProbabilisticTDModule, optional): if required, the Q-value can be computed twice
            independently using two separate networks. The minimum predicted value will then be used for inference.
        gamma (number, optional): discount for return computation
            default: 0.99
        priority_key (str, optional): tensordict key where to write the priority (for prioritized replay buffer usage).
            default: td_error
        loss_function (str, optional): loss function to be used with the value function loss.
            default: "smooth_l1"
        alpha_init (Number, optional): initial entropy multiplier.
            default: 1.0
        fixed_alpha (bool, optional): if True, alpha will be fixed to its initial value. Otherwise, alpha will be optimized to
            match the 'target_entropy' value.
            default: False
        target_entropy (Number or str, optional):
            default: "auto", where target entropy is computed as
    """

    def __init__(
        self,
        actor_network: Actor,
        qvalue_network: ProbabilisticTDModule,
        value_network: ProbabilisticTDModule,
        qvalue_network_bis: Optional[ProbabilisticTDModule] = None,
        gamma: Number = 0.99,
        priotity_key: str = "td_error",
        loss_function: str = "smooth_l1",
        alpha_init: Number = 1.0,
        fixed_alpha: bool = False,
        target_entropy: Union[str, Number] = "auto",
    ) -> None:
        super().__init__()
        self.actor_network = actor_network
        self.qvalue_network = qvalue_network
        self.value_network = value_network
        self.qvalue_network_bis = qvalue_network_bis
        self.gamma = gamma
        self.priority_key = priotity_key
        self.loss_function = loss_function
        self.register_buffer("alpha_init", torch.tensor(alpha_init))
        self.fixed_alpha = fixed_alpha
        if fixed_alpha:
            self.register_buffer("log_alpha", torch.tensor(math.log(alpha_init)))
        else:
            self.register_parameter(
                "log_alpha", torch.nn.Parameter(torch.tensor(math.log(alpha_init)))
            )

        if target_entropy == "auto":
            target_entropy = -float(np.prod(actor_network.spec.shape))
        self.register_buffer("target_entropy", torch.tensor(target_entropy))

    def parameters(self, recurse: bool = False) -> Iterator[Parameter]:
        for p in self.actor_network.parameters():
            yield p
        for p in self.qvalue_network.parameters():
            yield p
        if self.qvalue_network_bis is not None:
            for p in self.qvalue_network_bis.parameters():
                yield p
        for p in self.value_network.parameters():
            yield p
        if not self.fixed_alpha:
            yield self.log_alpha

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        _valid = set(self.parameters())
        for name, param in super().named_parameters():
            if param in _valid:
                yield name, param

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
        raise RuntimeError(
            "At least one of the networks of SACLoss must have trainable parameters."
        )

    def forward(self, tensordict: _TensorDict) -> _TensorDict:
        device = self.device
        td_device = tensordict.to(device)

        loss_actor = self._loss_actor(td_device)
        loss_qvalue, priority = self._loss_qvalue(td_device)
        loss_value = self._loss_value(td_device)
        loss_alpha = self._loss_alpha(td_device)
        tensordict.set(self.priority_key, priority, inplace=True)
        if (loss_actor.shape != loss_qvalue.shape) or (
            loss_actor.shape != loss_value.shape
        ):
            raise RuntimeError(
                f"Losses shape mismatch: {loss_actor.shape}, {loss_qvalue.shape} and {loss_value.shape}"
            )
        return TensorDict(
            {
                "loss_actor": loss_actor.mean(),
                "loss_qvalue": loss_qvalue.mean(),
                "loss_value": loss_value.mean(),
                "loss_alpha": loss_alpha.mean(),
                "alpha": self._alpha,
                "entropy": td_device.get("_log_prob").mean().detach(),
            },
            [],
        )

    def _loss_actor(self, tensordict: _TensorDict) -> Tensor:
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

        min_q_logprob = []
        for qval_net in qval_nets:
            with hold_out_net(qval_net):
                td_q = tensordict.select(*self.qvalue_network.in_keys)
                td_q.set("action", a_reparm, inplace=False)
                min_q_logprob.append(
                    qval_net(td_q).get("state_action_value").squeeze(-1)
                )

        min_q_logprob = torch.stack(min_q_logprob, 0).min(dim=0)[0]
        if log_prob.shape != min_q_logprob.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {log_prob.shape} and {min_q_logprob.shape}"
            )

        # write log_prob in tensordict for alpha loss
        tensordict.set("_log_prob", log_prob.detach())
        return self._alpha * log_prob - min_q_logprob

    def _loss_qvalue(self, tensordict: _TensorDict) -> Tuple[Tensor, Tensor]:
        actor_critic = ActorCriticWrapper(
            self.target_actor_network, self.target_value_network
        )
        target_value = next_state_value(
            tensordict, actor_critic, gamma=self.gamma, next_val_key="state_value"
        )

        # value loss
        if self.qvalue_network_bis is not None:
            nets = (self.qvalue_network_bis, self.qvalue_network)
        else:
            nets = (self.qvalue_network,)

        # Q-nets must be trained independently: as such, we split the data in 2 if required and train each q-net on
        # one half of the data.
        tensordict_chunks = tensordict.chunk(len(nets), dim=0)
        target_chunks = target_value.chunk(len(nets), dim=0)
        loss_value = []
        priority_value = []
        for _td, _target, _net in zip(tensordict_chunks, target_chunks, nets):
            td_copy = _td.select(*_net.in_keys).detach()
            _net(td_copy)
            pred_val = td_copy.get("state_action_value").squeeze(-1)
            loss_value.append(
                distance_loss(pred_val, _target, loss_function=self.loss_function)
            )
            priority_value.append(abs(pred_val - _target))

        loss_value = torch.cat(loss_value, 0)
        priority_value = torch.cat(priority_value, 0)

        return loss_value, priority_value

    def _loss_value(self, tensordict: _TensorDict) -> Tensor:
        # value loss
        td_copy = tensordict.select(*self.value_network.in_keys).detach()
        self.value_network(td_copy)
        pred_val = td_copy.get("state_value").squeeze(-1)

        with hold_out_net(self.target_actor_network):
            action_dist = self.target_actor_network.get_dist(td_copy)[
                0
            ]  # resample an action
            action = action_dist.rsample()
            td_copy.set("action", action, inplace=False)

            if self.target_qvalue_network_bis is not None:
                qval_nets = (self.target_qvalue_network_bis, self.target_qvalue_network)
            else:
                qval_nets = (self.target_qvalue_network,)

            min_qval = []
            for qval_net in qval_nets:
                with hold_out_net(qval_net):
                    min_qval.append(
                        qval_net(td_copy).get("state_action_value").squeeze(-1)
                    )
            min_qval = torch.stack(min_qval, 0).min(dim=0)[0]

            log_p = action_dist.log_prob(action)
            if log_p.shape != min_qval.shape:
                raise RuntimeError(
                    f"Losses shape mismatch: {min_qval.shape} and {log_p.shape}"
                )
            target_val = min_qval - self._alpha * log_p

        loss_value = distance_loss(
            pred_val, target_val, loss_function=self.loss_function
        )
        return loss_value

    def _loss_alpha(self, tensordict: _TensorDict) -> Tensor:
        log_pi = tensordict.get("_log_prob")
        if self.target_entropy is not None:
            # we can compute this loss even if log_alpha is not a parameter
            alpha_loss = -self.log_alpha.exp() * (log_pi.detach() + self.target_entropy)
        else:
            # placeholder
            alpha_loss = torch.zeros_like(log_pi)
        return alpha_loss

    @property
    def _alpha(self):
        with torch.no_grad():
            alpha = self.log_alpha.detach().exp()
        return alpha


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
                raise RuntimeError(
                    "qvalue_network_bis must be None if a separate qvalue target network has to be used"
                )
        self._target_value_network = deepcopy(self.value_network)
        self._target_value_network.requires_grad_(False)
        self.delay_actor = delay_actor
        if delay_actor:
            self._target_actor_network = deepcopy(self.actor_network)
            self._target_actor_network.requires_grad_(False)

    @property
    def target_value_network(self):
        self._target_value_network.apply(reset_noise)
        return self._target_value_network

    @property
    def target_qvalue_network(self):
        if self.delay_qvalue:
            self._target_qvalue_network.apply(reset_noise)
            return self._target_qvalue_network
        return self.qvalue_network

    @property
    def target_actor_network(self):
        if self.delay_actor:
            self._target_actor_network.apply(reset_noise)
            return self._target_actor_network
        return self.actor_network
