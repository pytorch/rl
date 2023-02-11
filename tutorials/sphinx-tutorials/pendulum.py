# -*- coding: utf-8 -*-
"""
Writing your environment with TorchRL
=====================================
**Author**: `Vincent Moens <https://github.com/vmoens>`_

Creating an environment (a simulator or a interface to a physical control system)
is an integrative part of reinforcement learning and control engineering.

TorchRL provides a set of tools to do this in multiple contexts.
This tutorial demonstrates how to use PyTorch and :py:mod:`torchrl` code a pendulum
simulator from the ground up.
It is freely inspired by the Pendulum-v1 implementation from `OpenAI-Gym/Farama-Gymnasium
control library <https://github.com/Farama-Foundation/Gymnasium>`__.

.. figure:: /_static/img/pendulum.gif
   :alt: Pendulum

   Simple Pendulum

Key learnings:

- How to design an environment in TorchRL:

    - Specs (input, observation and reward);
    - Methods: seeding, reset and step;
- Transforming your environment inputs and outputs;
- How to use :class:`tensordict.TensorDict` to carry arbitrary data structures
  from sep to step.

We will touch three crucial components of TorchRL:

* `environments <https://pytorch.org/rl/reference/envs.html>`__
* `transforms <https://pytorch.org/rl/reference/envs.html#transforms>`__
* `models (policy and value function) <https://pytorch.org/rl/reference/modules.html>`__

"""
######################################################################
# To give a sense of what can be achieved with TorchRL's environments, we will
# be designing a stateless environment. While stateful environments keep track of
# the latest physical state encountered and rely on this to simulate the state-to-state
# transition, stateless environments expect the current state to be provided to
# them at each step, along with the action undertaken.
#
# Modelling stateless environments gives users full control over the input and
# outputs of the simulator: one can reset an experiment at any stage. It also
# assumes that we have some control over a task, which may not always be the case
# (solving a problem where we cannot control the current state is more challenging
# but has a much wider set of applications).
#
# Another advantage of stateless environments is that most of the time they allow
# for batched execution of transition simulations. If the backend and the
# implementation allow it, an algebraic operation can be executed seamlessly on
# scalars, vectors or tensors. This tutorial gives such examples.
#
# This tutorial will be structured as follows:
#
# * We will first get acquainted with the environment properties:
#   its shape (``batch_size``), its methods (mainly `step`, `reset` and `set_seed`)
#   and finally its specs.
# * After having coded our simulator, we will demonstrate how it can be used
#   during training with transforms.
# * We will explore surprising new avenues that follow from the TorchRL's API,
#   including: the possibility of transforming inputs, the vectorized execution
#   of the simulation and the possibility of backpropagating through the
#   simulation.
# * Finally, will train a simple policy to solve the system we implemented.
#

from typing import Optional

import numpy as np
import torch
from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedDiscreteTensorSpec

from torchrl.envs import (
    CatTensors,
    Compose,
    EnvBase,
    ExcludeTransform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.utils import check_env_specs, step_mdp

DEFAULT_X = np.pi
DEFAULT_Y = 1.0


def make_composite_from_td(td):
    composite = CompositeSpec(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else UnboundedDiscreteTensorSpec(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite


def _reset(self, tensordict):
    if tensordict is None or tensordict.is_empty():
        tensordict = TensorDict(
            {"params": self.gen_params(batch_size=self.batch_size)}, self.batch_size
        )

    high_th = torch.tensor(DEFAULT_X, device=self.device)
    high_thdot = torch.tensor(DEFAULT_Y, device=self.device)
    low_th = -high_th
    low_thdot = -high_thdot

    th = torch.empty(tensordict.shape, device=self.device).uniform_(low_th, high_th)
    thdot = torch.empty(tensordict.shape, device=self.device).uniform_(
        low_thdot, high_thdot
    )
    out = TensorDict(
        {
            "th": th,
            "sin": th.sin(),
            "cos": th.cos(),
            "thdot": thdot,
            "params": tensordict["params"],
        },
        batch_size=tensordict.shape,
    )
    return out


def _step(self, tensordict):
    th, thdot = tensordict["th"], tensordict["thdot"]  # th := theta

    g_force = tensordict["params", "g"]
    mass = tensordict["params", "m"]
    length = tensordict["params", "l"]
    dt = tensordict["params", "dt"]
    u = tensordict["action"]
    u = u.clamp(-tensordict["params", "max_torque"], tensordict["params", "max_torque"])
    costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

    newthdot = (
        thdot
        + (3 * g_force / (2 * length) * th.sin() + 3.0 / (mass * length**2) * u) * dt
    )
    newthdot = newthdot.clamp(
        -tensordict["params", "max_speed"], tensordict["params", "max_speed"]
    )
    newth = th + newthdot * dt
    reward = -costs.view(*tensordict.shape, 1)
    out = TensorDict(
        {
            "th": newth,
            "sin": newth.sin(),
            "cos": newth.cos(),
            "thdot": newthdot,
            "params": tensordict["params"],
            "reward": reward,
            "done": torch.zeros_like(reward, dtype=torch.bool),
        },
        tensordict.shape,
    )
    return out


class PendulumEnv(EnvBase):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False

    @classmethod
    def gen_params(cls, g=10.0, batch_size=[]) -> TensorDictBase:
        td = TensorDict(
            {
                "max_speed": 8,
                "max_torque": 2.0,
                "dt": 0.05,
                "g": g,
                "m": 1.0,
                "l": 1.0,
            },
            [],
        )
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td

    def __init__(self, td_params=None, device="cpu"):
        if td_params is None:
            td_params = self.gen_params()

        self.clock = None
        self.isopen = True

        super().__init__(device=device, batch_size=td_params.batch_size)
        self.observation_spec = CompositeSpec(
            sin=BoundedTensorSpec(
                minimum=-1.0, maximum=1.0, shape=td_params.shape, dtype=torch.float32
            ),
            cos=BoundedTensorSpec(
                minimum=-1.0, maximum=1.0, shape=td_params.shape, dtype=torch.float32
            ),
            th=BoundedTensorSpec(
                minimum=-torch.pi,
                maximum=torch.pi,
                shape=td_params.shape,
                dtype=torch.float32,
            ),
            thdot=BoundedTensorSpec(
                minimum=-td_params["max_speed"],
                maximum=td_params["max_speed"],
                shape=td_params.shape,
                dtype=torch.float32,
            ),
            params=make_composite_from_td(td_params),
            shape=td_params.shape,
        )
        # since the environment is stateless, we expect the previous output as input
        self.input_spec = self.observation_spec.clone()
        # action-spec will be automatically wrapped in input_spec, but the convenient
        # self.action_spec = spec is supported
        self.action_spec = BoundedTensorSpec(
            minimum=-td_params["max_torque"],
            maximum=td_params["max_torque"],
            shape=td_params.shape,
            dtype=torch.float32,
        )
        self.reward_spec = UnboundedDiscreteTensorSpec(shape=(*td_params.shape, 1))

    def _set_seed(self, seed: Optional[int]):
        rng = torch.random.set_rng_state(seed)
        self.rng = rng
        return rng

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def close(self):
        pass


def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


if __name__ == "__main__":
    env = PendulumEnv()
    check_env_specs(env)
    print(env.observation_spec)
    td = env.reset()
    td = env.rand_step(td)
    print(td)

    env = TransformedEnv(
        env,
        Compose(
            UnsqueezeTransform(
                unsqueeze_dim=-1,
                in_keys=["sin", "cos", "thdot"],
                in_keys_inv=["sin", "cos", "thdot"],
            ),
            CatTensors(
                in_keys=["sin", "cos", "thdot"], out_key="observation", del_keys=False
            ),
        ),
    )
    print(env.observation_spec)
    td = env.reset()
    print("reset", td)
    td = env.rand_step(td)
    print("rand step", td)
    td = env.rand_step(step_mdp(td))
    print("rand step", td)
    td = env.rollout(5)
    print("rollout", td)

    td = env.reset().expand(10)
    print("reset (10)", td)
    td["action"] = env.action_spec.rand(td.shape)
    td = env.step(td)
    print("rand step (10)", td)
    td = step_mdp(td)
    td["action"] = env.action_spec.rand(td.shape)
    td = env.step(td)
    print("rand step (10)", td)

    # multidimensional env
    td_params = PendulumEnv.gen_params(batch_size=[16])
    base_env = PendulumEnv(td_params)
    env = TransformedEnv(
        base_env,
        Compose(
            UnsqueezeTransform(
                unsqueeze_dim=-1,
                in_keys=["sin", "cos", "thdot"],
                in_keys_inv=["sin", "cos", "thdot"],
            ),
            CatTensors(
                in_keys=["sin", "cos", "thdot"], out_key="observation", del_keys=False
            ),
        ),
    )
    init_td = env.reset()
    rollout = env.rollout(10, auto_reset=False, tensordict=init_td)
    print(rollout)
