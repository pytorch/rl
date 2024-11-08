# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import Bounded, Composite, Unbounded
from torchrl.envs.common import EnvBase
from torchrl.envs.utils import make_composite_from_td


class PendulumEnv(EnvBase):
    """A stateless Pendulum environment.

    See the Pendulum tutorial for more details: :ref:`tutorial <pendulum_tuto>`.

    Specs:
        >>> env = PendulumEnv()
        >>> env.specs
        Composite(
            output_spec: Composite(
                full_observation_spec: Composite(
                    th: BoundedContinuous(
                        shape=torch.Size([]),
                        space=ContinuousBox(
                            low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                            high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                        device=cpu,
                        dtype=torch.float32,
                        domain=continuous),
                    thdot: BoundedContinuous(
                        shape=torch.Size([]),
                        space=ContinuousBox(
                            low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                            high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                        device=cpu,
                        dtype=torch.float32,
                        domain=continuous),
                    params: Composite(
                        max_speed: UnboundedDiscrete(
                            shape=torch.Size([]),
                            space=ContinuousBox(
                                low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, contiguous=True),
                                high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, contiguous=True)),
                            device=cpu,
                            dtype=torch.int64,
                            domain=discrete),
                        max_torque: UnboundedContinuous(
                            shape=torch.Size([]),
                            space=ContinuousBox(
                                low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                                high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                            device=cpu,
                            dtype=torch.float32,
                            domain=continuous),
                        dt: UnboundedContinuous(
                            shape=torch.Size([]),
                            space=ContinuousBox(
                                low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                                high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                            device=cpu,
                            dtype=torch.float32,
                            domain=continuous),
                        g: UnboundedContinuous(
                            shape=torch.Size([]),
                            space=ContinuousBox(
                                low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                                high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                            device=cpu,
                            dtype=torch.float32,
                            domain=continuous),
                        m: UnboundedContinuous(
                            shape=torch.Size([]),
                            space=ContinuousBox(
                                low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                                high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                            device=cpu,
                            dtype=torch.float32,
                            domain=continuous),
                        l: UnboundedContinuous(
                            shape=torch.Size([]),
                            space=ContinuousBox(
                                low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                                high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                            device=cpu,
                            dtype=torch.float32,
                            domain=continuous),
                        device=None,
                        shape=torch.Size([])),
                    device=None,
                    shape=torch.Size([])),
                full_reward_spec: Composite(
                    reward: UnboundedContinuous(
                        shape=torch.Size([1]),
                        space=ContinuousBox(
                            low=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True),
                            high=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True)),
                        device=cpu,
                        dtype=torch.float32,
                        domain=continuous),
                    device=None,
                    shape=torch.Size([])),
                full_done_spec: Composite(
                    done: Categorical(
                        shape=torch.Size([1]),
                        space=CategoricalBox(n=2),
                        device=cpu,
                        dtype=torch.bool,
                        domain=discrete),
                    terminated: Categorical(
                        shape=torch.Size([1]),
                        space=CategoricalBox(n=2),
                        device=cpu,
                        dtype=torch.bool,
                        domain=discrete),
                    device=None,
                    shape=torch.Size([])),
                device=None,
                shape=torch.Size([])),
            input_spec: Composite(
                full_state_spec: Composite(
                    th: BoundedContinuous(
                        shape=torch.Size([]),
                        space=ContinuousBox(
                            low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                            high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                        device=cpu,
                        dtype=torch.float32,
                        domain=continuous),
                    thdot: BoundedContinuous(
                        shape=torch.Size([]),
                        space=ContinuousBox(
                            low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                            high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                        device=cpu,
                        dtype=torch.float32,
                        domain=continuous),
                    params: Composite(
                        max_speed: UnboundedDiscrete(
                            shape=torch.Size([]),
                            space=ContinuousBox(
                                low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, contiguous=True),
                                high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, contiguous=True)),
                            device=cpu,
                            dtype=torch.int64,
                            domain=discrete),
                        max_torque: UnboundedContinuous(
                            shape=torch.Size([]),
                            space=ContinuousBox(
                                low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                                high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                            device=cpu,
                            dtype=torch.float32,
                            domain=continuous),
                        dt: UnboundedContinuous(
                            shape=torch.Size([]),
                            space=ContinuousBox(
                                low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                                high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                            device=cpu,
                            dtype=torch.float32,
                            domain=continuous),
                        g: UnboundedContinuous(
                            shape=torch.Size([]),
                            space=ContinuousBox(
                                low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                                high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                            device=cpu,
                            dtype=torch.float32,
                            domain=continuous),
                        m: UnboundedContinuous(
                            shape=torch.Size([]),
                            space=ContinuousBox(
                                low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                                high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                            device=cpu,
                            dtype=torch.float32,
                            domain=continuous),
                        l: UnboundedContinuous(
                            shape=torch.Size([]),
                            space=ContinuousBox(
                                low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                                high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
                            device=cpu,
                            dtype=torch.float32,
                            domain=continuous),
                        device=None,
                        shape=torch.Size([])),
                    device=None,
                    shape=torch.Size([])),
                full_action_spec: Composite(
                    action: BoundedContinuous(
                        shape=torch.Size([1]),
                        space=ContinuousBox(
                            low=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True),
                            high=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True)),
                        device=cpu,
                        dtype=torch.float32,
                        domain=continuous),
                    device=None,
                    shape=torch.Size([])),
                device=None,
                shape=torch.Size([])),
            device=None,
            shape=torch.Size([]))

    """

    DEFAULT_X = np.pi
    DEFAULT_Y = 1.0

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False
    rng = None

    def __init__(self, td_params=None, seed=None, device=None):
        if td_params is None:
            td_params = self.gen_params(device=self.device)

        super().__init__(device=device)
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_(generator=self.rng).item()
        self.set_seed(seed)

    @classmethod
    def _step(cls, tensordict):
        th, thdot = tensordict["th"], tensordict["thdot"]  # th := theta

        g_force = tensordict["params", "g"]
        mass = tensordict["params", "m"]
        length = tensordict["params", "l"]
        dt = tensordict["params", "dt"]
        u = tensordict["action"].squeeze(-1)
        u = u.clamp(
            -tensordict["params", "max_torque"], tensordict["params", "max_torque"]
        )
        costs = cls.angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        new_thdot = (
            thdot
            + (3 * g_force / (2 * length) * th.sin() + 3.0 / (mass * length**2) * u)
            * dt
        )
        new_thdot = new_thdot.clamp(
            -tensordict["params", "max_speed"], tensordict["params", "max_speed"]
        )
        new_th = th + new_thdot * dt
        reward = -costs.view(*tensordict.shape, 1)
        done = torch.zeros_like(reward, dtype=torch.bool)
        out = TensorDict(
            {
                "th": new_th,
                "thdot": new_thdot,
                "params": tensordict["params"],
                "reward": reward,
                "done": done,
            },
            tensordict.shape,
        )
        return out

    def _reset(self, tensordict):
        batch_size = (
            tensordict.batch_size if tensordict is not None else self.batch_size
        )
        if tensordict is None or tensordict.is_empty():
            # if no ``tensordict`` is passed, we generate a single set of hyperparameters
            # Otherwise, we assume that the input ``tensordict`` contains all the relevant
            # parameters to get started.
            tensordict = self.gen_params(batch_size=batch_size, device=self.device)

        high_th = torch.tensor(self.DEFAULT_X, device=self.device)
        high_thdot = torch.tensor(self.DEFAULT_Y, device=self.device)
        low_th = -high_th
        low_thdot = -high_thdot

        # for non batch-locked environments, the input ``tensordict`` shape dictates the number
        # of simulators run simultaneously. In other contexts, the initial
        # random state's shape will depend upon the environment batch-size instead.
        th = (
            torch.rand(tensordict.shape, generator=self.rng, device=self.device)
            * (high_th - low_th)
            + low_th
        )
        thdot = (
            torch.rand(tensordict.shape, generator=self.rng, device=self.device)
            * (high_thdot - low_thdot)
            + low_thdot
        )
        out = TensorDict(
            {
                "th": th,
                "thdot": thdot,
                "params": tensordict["params"],
            },
            batch_size=batch_size,
        )
        return out

    def _make_spec(self, td_params):
        # Under the hood, this will populate self.output_spec["observation"]
        self.observation_spec = Composite(
            th=Bounded(
                low=-torch.pi,
                high=torch.pi,
                shape=(),
                dtype=torch.float32,
            ),
            thdot=Bounded(
                low=-td_params["params", "max_speed"],
                high=td_params["params", "max_speed"],
                shape=(),
                dtype=torch.float32,
            ),
            # we need to add the ``params`` to the observation specs, as we want
            # to pass it at each step during a rollout
            params=make_composite_from_td(
                td_params["params"], unsqueeze_null_shapes=False
            ),
            shape=(),
        )
        # since the environment is stateless, we expect the previous output as input.
        # For this, ``EnvBase`` expects some state_spec to be available
        self.state_spec = self.observation_spec.clone()
        # action-spec will be automatically wrapped in input_spec when
        # `self.action_spec = spec` will be called supported
        self.action_spec = Bounded(
            low=-td_params["params", "max_torque"],
            high=td_params["params", "max_torque"],
            shape=(1,),
            dtype=torch.float32,
        )
        self.reward_spec = Unbounded(shape=(*td_params.shape, 1))

    def make_composite_from_td(td):
        # custom function to convert a ``tensordict`` in a similar spec structure
        # of unbounded values.
        composite = Composite(
            {
                key: make_composite_from_td(tensor)
                if isinstance(tensor, TensorDictBase)
                else Unbounded(
                    dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
                )
                for key, tensor in td.items()
            },
            shape=td.shape,
        )
        return composite

    def _set_seed(self, seed: int):
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        self.rng = rng

    @staticmethod
    def gen_params(g=10.0, batch_size=None, device=None) -> TensorDictBase:
        """Returns a ``tensordict`` containing the physical parameters such as gravitational force and torque or speed limits."""
        if batch_size is None:
            batch_size = []
        td = TensorDict(
            {
                "params": TensorDict(
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
            },
            [],
            device=device,
        )
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td

    @staticmethod
    def angle_normalize(x):
        return ((x + torch.pi) % (2 * torch.pi)) - torch.pi
