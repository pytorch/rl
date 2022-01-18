from numbers import Number
from typing import Iterable, Optional, Union

import numpy as np
import torch
from torch import distributions as d

from torchrl.envs.utils import set_exploration_mode
from .common import _forward_hook_safe_action
from ..probabilistic_operators import (
    ProbabilisticOperatorWrapper,
    ProbabilisticOperator,
)

__all__ = ["EGreedyWrapper", "OrnsteinUhlenbeckProcessWrapper"]

from ...data.tensordict.tensordict import _TensorDict


class EGreedyWrapper(ProbabilisticOperatorWrapper):
    """
    Epsilon-Greedy PO wrapper.

    Args:
        policy (ProbabilisticOperator): a deterministic policy
        eps_init (scalar): initial epsilon value.
            default: 1.0
        eps_end (scalar): final epsilon value.
            default: 0.1
        annealing_num_steps (int): number of steps it will take for epsilon to reach the eps_end value
    """
    def __init__(
            self,
            policy: ProbabilisticOperator,
            eps_init: Number = 1.0,
            eps_end: Number = 0.1,
            annealing_num_steps: int = 1000,
    ):
        super().__init__(policy)
        self.register_buffer("eps_init", torch.tensor([eps_init]))
        self.register_buffer("eps_end", torch.tensor([eps_end]))
        if self.eps_end > self.eps_init:
            raise RuntimeError("eps should decrease over time or be constant")
        self.annealing_num_steps = annealing_num_steps
        self.register_buffer("eps", torch.tensor([eps_init]))

    def step(self) -> None:
        """
        A step of epsilon decay.
        After self.annealing_num_steps, this function is a no-op.

        Returns: None

        """
        self.eps.data[0] = max(
            self.eps_end.item(),
            (
                    self.eps - (self.eps_init - self.eps_end) / self.annealing_num_steps
            ).item(),
        )

    @set_exploration_mode("random")
    def _dist_sample(self, dist: d.Distribution, *input: Iterable[torch.Tensor], **kwargs: dict) -> torch.Tensor:
        out = self.probabilistic_operator._dist_sample(dist, *input, **kwargs)
        eps = self.eps.item()
        cond = (torch.rand(out[..., :1].shape, device=out.device) < eps).to(out.dtype)
        out = cond * self.probabilistic_operator.random() + (1 - cond) * out
        return out


class OrnsteinUhlenbeckProcessWrapper(ProbabilisticOperatorWrapper):
    """
    Ornstein-Uhlenbeck exploration policy wrapper as presented in "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING",
    https://arxiv.org/pdf/1509.02971.pdf.

    The OU exploration is to be used with continuous control policies and introduces a auto-correlated exploration
    noise. This enables a sort of 'structured' exploration.

        Noise equation:
            noise = prev_noise + theta * (mu - prev_noise) * dt + current_sigma * sqrt(dt) * W
        Sigma equation:
            current_sigma = (-(sigma - sigma_min) / (n_steps_annealing) * n_steps + sigma).clamp_min(sigma_min)

    Args:
        policy (ProbabilisticOperator): a policy
        eps_init (scalar): initial epsilon value, determining the amount of noise to be added.
            default: 1.0
        eps_end (scalar): final epsilon value, determining the amount of noise to be added.
            default: 0.1
        annealing_num_steps (int): number of steps it will take for epsilon to reach the eps_end value.
            default: 1000
        theta (scalar): theta factor in the noise equation
            default: 0.15
        mu (scalar): OU average (mu in the noise equation).
            default: 0.0
        sigma (scalar): sigma value in the sigma equation.
            default: 0.2
        dt (scalar): dt in the noise equation.
            default: 0.01
        x0 (Tensor, ndarray, optional): initial value of the process.
            default: 0.0
        sigma_min (number, optional): sigma_min in the sigma equation.
            default: None
        n_steps_annealing (int): number of steps for the sigma annealing.
            default: 1000
        key (str): key of the action to be modified.
            default: "action"
        safe (bool): if True, actions that are out of bounds given the action specs will be projected in the space
            given the `TensorSpec.project` heuristic.
            default: True
    """
    def __init__(
            self,
            policy: ProbabilisticOperator,
            eps_init: Number = 1.0,
            eps_end: Number = 0.1,
            annealing_num_steps: int = 1000,
            theta: Number = 0.15,
            mu: Number = 0.0,
            sigma: Number = 0.2,
            dt: Number = 1e-2,
            x0: Optional[Union[torch.Tensor, np.ndarray]] = None,
            sigma_min: Optional[Number] = None,
            n_steps_annealing: int = 1000,
            key: str = 'action',
            safe: bool = True,
    ):
        super().__init__(policy)
        self.ou = _OrnsteinUhlenbeckProcess(
            theta=theta,
            mu=mu,
            sigma=sigma,
            dt=dt,
            x0=x0,
            sigma_min=sigma_min,
            n_steps_annealing=n_steps_annealing,
            key=key,
        )
        self.register_buffer("eps_init", torch.tensor([eps_init]))
        self.register_buffer("eps_end", torch.tensor([eps_end]))
        if self.eps_end > self.eps_init:
            raise ValueError("eps should decrease over time or be constant, "
                             f"got eps_init={eps_init} and eps_end={eps_end}")
        self.annealing_num_steps = annealing_num_steps
        self.register_buffer("eps", torch.tensor([eps_init]))
        self.out_keys = list(self.probabilistic_operator.out_keys) + [self.ou.out_keys]
        self.safe = safe
        if self.safe:
            self.register_forward_hook(_forward_hook_safe_action)

    def step(self) -> None:
        if self.annealing_num_steps > 0:
            self.eps.data[0] = max(
                self.eps_end.item(),
                (
                        self.eps - (self.eps_init - self.eps_end) / self.annealing_num_steps
                ).item(),
            )

    @set_exploration_mode("random")
    def forward(self, tensor_dict: _TensorDict) -> _TensorDict:
        tensor_dict = super().forward(tensor_dict)
        tensor_dict = self.ou.add_sample(tensor_dict, self.eps.item())
        return tensor_dict


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class _OrnsteinUhlenbeckProcess:
    def __init__(
            self,
            theta: Number,
            mu: Number = 0.0,
            sigma: Number = 0.2,
            dt: Number = 1e-2,
            x0: Optional[Union[torch.Tensor, np.ndarray]] = None,
            sigma_min: Optional[Number] = None,
            n_steps_annealing: int = 1000,
            key: str = 'action',
    ):
        self.mu = mu
        self.sigma = sigma

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.0
            self.c = sigma
            self.sigma_min = sigma

        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0 if x0 is not None else 0.0
        self.key = key
        self.noise_key = f'prev_noise_{id(self)}'
        self.steps_key = f'steps_key_{id(self)}'
        self.out_keys = [self.key, self.noise_key, self.steps_key]

    def _make_noise_pair(self, tensor_dict: _TensorDict) -> None:
        tensor_dict.set(self.noise_key,
                        torch.zeros(tensor_dict.get(self.key).shape,
                                    device=tensor_dict.device))
        tensor_dict.set(self.steps_key,
                        torch.zeros(torch.Size([*tensor_dict.batch_size, 1]), dtype=torch.long,
                                    device=tensor_dict.device))

    def add_sample(self, tensor_dict: _TensorDict, eps: Number = 1.0) -> _TensorDict:

        if not self.noise_key in set(tensor_dict.keys()):
            self._make_noise_pair(tensor_dict)

        prev_noise = tensor_dict.get(self.noise_key)
        # if (prev_noise.norm(dim=-1)==0).any():
        #     print(f"{(prev_noise.norm(dim=-1)==0).sum()} noises are 0")
        prev_noise = prev_noise + self.x0

        n_steps = tensor_dict.get(self.steps_key)

        noise = (
                prev_noise
                + self.theta * (self.mu - prev_noise) * self.dt
                + self.current_sigma(n_steps) * np.sqrt(self.dt) * torch.randn_like(prev_noise)
        )
        tensor_dict.set_(self.noise_key, noise - self.x0)
        tensor_dict.set_(self.key, tensor_dict.get(self.key) + eps * noise)
        tensor_dict.set_(self.steps_key, n_steps + 1)
        return tensor_dict

    def current_sigma(self, n_steps: torch.Tensor) -> torch.Tensor:
        sigma = (self.m * n_steps + self.c).clamp_min(self.sigma_min)
        return sigma
