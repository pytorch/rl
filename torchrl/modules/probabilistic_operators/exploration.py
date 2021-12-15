from ..probabilistic_operators import (
    ProbabilisticOperatorWrapper,
    ProbabilisticOperator,
)
from torchrl.envs.utils import exploration_mode, set_exploration_mode
import torch
from torchrl.modules.distributions import UNIFORM
import numpy as np

__all__ = ["EGreedyWrapper", "OrnsteinUhlenbeckProcess", "OrnsteinUhlenbeckProcessWrapper"]


class EGreedyWrapper(ProbabilisticOperatorWrapper):
    def __init__(
            self,
            probabilistic_operator: ProbabilisticOperator,
            eps_init=1.0,
            eps_end=0.1,
            annealing_num_steps=1000,
    ):
        super().__init__(probabilistic_operator)
        self.register_buffer("eps_init", torch.tensor([eps_init]))
        self.register_buffer("eps_end", torch.tensor([eps_end]))
        assert (
                self.eps_end <= self.eps_init
        ), "eps should decrease over time or be constant"
        self.annealing_num_steps = annealing_num_steps
        self.register_buffer("eps", torch.tensor([eps_init]))

    def step(self):
        self.eps.data[0] = max(
            self.eps_end.item(),
            (
                    self.eps - (self.eps_init - self.eps_end) / self.annealing_num_steps
            ).item(),
        )

    @set_exploration_mode(True)
    def _dist_sample(self, dist, *input, **kwargs):
        out = self.probabilistic_operator._dist_sample(dist, *input, **kwargs)
        eps = self.eps.item()
        cond = (torch.rand(out[..., :1].shape, device=out.device) < eps).to(out.dtype)
        out = cond * self.probabilistic_operator.random_sample(out.shape) + (1 - cond) * out
        return out


class OrnsteinUhlenbeckProcessWrapper(ProbabilisticOperatorWrapper):
    def __init__(
            self,
            probabilistic_operator: ProbabilisticOperator,
            eps_init=1.0,
            eps_end=0.1,
            annealing_num_steps=1000,
            theta=0.15,
            mu: float = 0.0,
            sigma: float = 0.2,
            dt: float = 1e-2,
            x0=None,
            sigma_min=None,
            n_steps_annealing=1000,
            key='action',
    ):
        super().__init__(probabilistic_operator)
        self.ou = OrnsteinUhlenbeckProcess(
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
        assert (
                self.eps_end <= self.eps_init
        ), "eps should decrease over time or be constant"
        self.annealing_num_steps = annealing_num_steps
        self.register_buffer("eps", torch.tensor([eps_init]))
        self.out_keys = list(self.probabilistic_operator.out_keys) + [self.ou.out_keys]

    def step(self):
        if self.annealing_num_steps>0:
            self.eps.data[0] = max(
                self.eps_end.item(),
                (
                        self.eps - (self.eps_init - self.eps_end) / self.annealing_num_steps
                ).item(),
            )

    @set_exploration_mode(True)
    def forward(self, tensor_dict):
        tensor_dict = super().forward(tensor_dict)
        tensor_dict = self.ou.add_sample(tensor_dict, self.eps.item())
        return tensor_dict


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckProcess:
    def __init__(
            self,
            theta: float,
            mu: float = 0.0,
            sigma: float = 0.2,
            dt: float = 1e-2,
            x0=None,
            sigma_min=None,
            n_steps_annealing=1000,
            key='action',
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

    def _make_noise_pair(self, tensor_dict):
        tensor_dict.set(self.noise_key,
                      torch.zeros(tensor_dict.get(self.key).shape,
                                  device=tensor_dict.device))
        tensor_dict.set(self.steps_key,
                      torch.zeros(torch.Size([*tensor_dict.batch_size, 1]), dtype=torch.long,
                                  device=tensor_dict.device))

    def add_sample(self, tensor_dict, eps=1.0):

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

    def current_sigma(self, n_steps):
        sigma = (self.m * n_steps + self.c).clamp_min(self.sigma_min)
        return sigma
