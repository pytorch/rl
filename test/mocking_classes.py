# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch
import torch.nn as nn
from torchrl._utils import seed_generator
from torchrl.data.tensor_specs import (
    BinaryDiscreteTensorSpec,
    BoundedTensorSpec,
    CompositeSpec,
    MultOneHotDiscreteTensorSpec,
    NdBoundedTensorSpec,
    NdUnboundedContinuousTensorSpec,
    OneHotDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.data.tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.envs.common import EnvBase
from torchrl.envs.model_based.common import ModelBasedEnvBase

spec_dict = {
    "bounded": BoundedTensorSpec,
    "one_hot": OneHotDiscreteTensorSpec,
    "unbounded": UnboundedContinuousTensorSpec,
    "ndbounded": NdBoundedTensorSpec,
    "ndunbounded": NdUnboundedContinuousTensorSpec,
    "binary": BinaryDiscreteTensorSpec,
    "mult_one_hot": MultOneHotDiscreteTensorSpec,
    "composite": CompositeSpec,
}

default_spec_kwargs = {
    BoundedTensorSpec: {"minimum": -1.0, "maximum": 1.0},
    OneHotDiscreteTensorSpec: {"n": 7},
    UnboundedContinuousTensorSpec: {},
    NdBoundedTensorSpec: {"minimum": -torch.ones(4), "maxmimum": torch.ones(4)},
    NdUnboundedContinuousTensorSpec: {
        "shape": [
            7,
        ]
    },
    BinaryDiscreteTensorSpec: {"n": 7},
    MultOneHotDiscreteTensorSpec: {"nvec": [7, 3, 5]},
    CompositeSpec: {},
}


def make_spec(spec_str):
    target_class = spec_dict[spec_str]
    return target_class(**default_spec_kwargs[target_class])


class _MockEnv(EnvBase):
    @classmethod
    def __new__(
        cls,
        *args,
        **kwargs,
    ):
        for key, item in list(cls._observation_spec.items()):
            cls._observation_spec[key] = item.to(torch.get_default_dtype())
        cls._reward_spec = cls._reward_spec.to(torch.get_default_dtype())
        return super().__new__(*args, **kwargs)

    def __init__(self, seed: int = 100):
        super().__init__(
            device="cpu",
            dtype=torch.get_default_dtype(),
        )
        self.set_seed(seed)
        self.is_closed = False

    @property
    def maxstep(self):
        return 100

    def set_seed(self, seed: int, static_seed=False) -> int:
        self.seed = seed
        self.counter = seed % 17  # make counter a small number
        if static_seed:
            return seed
        return seed_generator(seed)

    def custom_fun(self):
        return 0

    custom_attr = 1

    @property
    def custom_prop(self):
        return 2

    @property
    def custom_td(self):
        return TensorDict({"a": torch.zeros(3)}, [])


class MockSerialEnv(EnvBase):
    @classmethod
    def __new__(
        cls,
        *args,
        observation_spec=None,
        action_spec=None,
        input_spec=None,
        reward_spec=None,
        **kwargs,
    ):
        if action_spec is None:
            action_spec = NdUnboundedContinuousTensorSpec((1,))
        if observation_spec is None:
            observation_spec = CompositeSpec(
                next_observation=NdUnboundedContinuousTensorSpec((1,))
            )
        if reward_spec is None:
            reward_spec = NdUnboundedContinuousTensorSpec((1,))
        if input_spec is None:
            input_spec = CompositeSpec(action=action_spec)
        cls._reward_spec = reward_spec
        cls._observation_spec = observation_spec
        cls._input_spec = input_spec
        return super().__new__(*args, **kwargs)

    def __init__(self, device):
        super(MockSerialEnv, self).__init__(device=device)
        self.is_closed = False

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        assert seed >= 1
        self.seed = seed
        self.counter = seed % 17  # make counter a small number
        self.max_val = max(self.counter + 100, self.counter * 2)
        if static_seed:
            return seed
        return seed_generator(seed)

    def _step(self, tensordict):
        self.counter += 1
        n = torch.tensor(
            [self.counter], device=self.device, dtype=torch.get_default_dtype()
        )
        done = self.counter >= self.max_val
        done = torch.tensor([done], dtype=torch.bool, device=self.device)
        return TensorDict(
            {"reward": n, "done": done, "next_observation": n.clone()}, []
        )

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        self.max_val = max(self.counter + 100, self.counter * 2)

        n = torch.tensor(
            [self.counter], device=self.device, dtype=torch.get_default_dtype()
        )
        done = self.counter >= self.max_val
        done = torch.tensor([done], dtype=torch.bool, device=self.device)
        return TensorDict({"done": done, "next_observation": n}, [])

    def rand_step(self, tensordict: Optional[TensorDictBase] = None) -> TensorDictBase:
        return self.step(tensordict)


class MockBatchedLockedEnv(EnvBase):
    """Mocks an env whose batch_size defines the size of the output tensordict"""

    @classmethod
    def __new__(
        cls,
        *args,
        observation_spec=None,
        action_spec=None,
        input_spec=None,
        reward_spec=None,
        **kwargs,
    ):
        if action_spec is None:
            action_spec = NdUnboundedContinuousTensorSpec((1,))
        if input_spec is None:
            input_spec = CompositeSpec(
                action=action_spec,
                observation=NdUnboundedContinuousTensorSpec((1,)),
            )
        if observation_spec is None:
            observation_spec = CompositeSpec(
                next_observation=NdUnboundedContinuousTensorSpec((1,))
            )
        if reward_spec is None:
            reward_spec = NdUnboundedContinuousTensorSpec((1,))
        cls._reward_spec = reward_spec
        cls._observation_spec = observation_spec
        cls._input_spec = input_spec
        return super().__new__(
            cls,
            *args,
            **kwargs,
        )

    def __init__(self, device, batch_size=None):
        super(MockBatchedLockedEnv, self).__init__(device=device, batch_size=batch_size)
        self.counter = 0

    set_seed = MockSerialEnv.set_seed
    rand_step = MockSerialEnv.rand_step

    def _step(self, tensordict):
        self.counter += 1
        # We use tensordict.batch_size instead of self.batch_size since this method will also be used by MockBatchedUnLockedEnv
        n = (
            torch.full(tensordict.batch_size, self.counter)
            .to(self.device)
            .to(torch.get_default_dtype())
        )
        done = self.counter >= self.max_val
        done = torch.full(
            tensordict.batch_size, done, dtype=torch.bool, device=self.device
        )

        return TensorDict(
            {"reward": n, "done": done, "next_observation": n},
            tensordict.batch_size,
            device=self.device,
        )

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        self.max_val = max(self.counter + 100, self.counter * 2)
        if tensordict is None:
            batch_size = self.batch_size
        else:
            batch_size = tensordict.batch_size

        n = (
            torch.full(batch_size, self.counter)
            .to(self.device)
            .to(torch.get_default_dtype())
        )
        done = self.counter >= self.max_val
        done = torch.full(batch_size, done, dtype=torch.bool, device=self.device)

        return TensorDict(
            {"reward": n, "done": done, "next_observation": n},
            batch_size,
            device=self.device,
        )


class MockBatchedUnLockedEnv(MockBatchedLockedEnv):
    """Mocks an env whose batch_size does not define the size of the output tensordict.

    The size of the output tensordict is defined by the input tensordict itself.

    """

    def __init__(self, device, batch_size=None):
        super(MockBatchedUnLockedEnv, self).__init__(
            batch_size=batch_size, device=device
        )

    @classmethod
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, _batch_locked=False, **kwargs)


class DiscreteActionVecMockEnv(_MockEnv):
    @classmethod
    def __new__(
        cls,
        *args,
        observation_spec=None,
        action_spec=None,
        input_spec=None,
        reward_spec=None,
        from_pixels=False,
        **kwargs,
    ):
        size = cls.size = 7
        if observation_spec is None:
            cls.out_key = "observation"
            observation_spec = CompositeSpec(
                next_observation=NdUnboundedContinuousTensorSpec(
                    shape=torch.Size([size])
                ),
                next_observation_orig=NdUnboundedContinuousTensorSpec(
                    shape=torch.Size([size])
                ),
            )
        if action_spec is None:
            action_spec = OneHotDiscreteTensorSpec(7)
        if reward_spec is None:
            reward_spec = UnboundedContinuousTensorSpec()

        if input_spec is None:
            cls._out_key = "observation_orig"
            input_spec = CompositeSpec(
                **{
                    cls._out_key: observation_spec["next_observation"],
                    "action": action_spec,
                }
            )
        cls._reward_spec = reward_spec
        cls._observation_spec = observation_spec
        cls._input_spec = input_spec
        cls.from_pixels = from_pixels
        return super().__new__(*args, **kwargs)

    def _get_in_obs(self, obs):
        return obs

    def _get_out_obs(self, obs):
        return obs

    def _reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.counter += 1
        state = torch.zeros(self.size) + self.counter
        if tensordict is None:
            tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict = tensordict.select().set(
            "next_" + self.out_key, self._get_out_obs(state)
        )
        tensordict = tensordict.set("next_" + self._out_key, self._get_out_obs(state))
        tensordict.set("done", torch.zeros(*tensordict.shape, 1, dtype=torch.bool))
        return tensordict

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        tensordict = tensordict.to(self.device)
        a = tensordict.get("action")
        assert (a.sum(-1) == 1).all()
        assert not self.is_done, "trying to execute step in done env"

        obs = self._get_in_obs(tensordict.get(self._out_key)) + a / self.maxstep
        tensordict = tensordict.select()  # empty tensordict

        tensordict.set("next_" + self.out_key, self._get_out_obs(obs))
        tensordict.set("next_" + self._out_key, self._get_out_obs(obs))

        done = torch.isclose(obs, torch.ones_like(obs) * (self.counter + 1))
        reward = done.any(-1).unsqueeze(-1)
        # set done to False
        done = torch.zeros_like(done).all(-1).unsqueeze(-1)
        tensordict.set("reward", reward.to(torch.get_default_dtype()))
        tensordict.set("done", done)
        return tensordict


class ContinuousActionVecMockEnv(_MockEnv):
    @classmethod
    def __new__(
        cls,
        *args,
        observation_spec=None,
        action_spec=None,
        input_spec=None,
        reward_spec=None,
        from_pixels=False,
        **kwargs,
    ):
        size = cls.size = 7
        if observation_spec is None:
            cls.out_key = "observation"
            observation_spec = CompositeSpec(
                next_observation=NdUnboundedContinuousTensorSpec(
                    shape=torch.Size([size])
                ),
                next_observation_orig=NdUnboundedContinuousTensorSpec(
                    shape=torch.Size([size])
                ),
            )
        if action_spec is None:
            action_spec = NdBoundedTensorSpec(-1, 1, (7,))
        if reward_spec is None:
            reward_spec = UnboundedContinuousTensorSpec()

        if input_spec is None:
            cls._out_key = "observation_orig"
            input_spec = CompositeSpec(
                **{
                    cls._out_key: observation_spec["next_observation"],
                    "action": action_spec,
                }
            )
        cls._reward_spec = reward_spec
        cls._observation_spec = observation_spec
        cls._input_spec = input_spec
        cls.from_pixels = from_pixels
        return super().__new__(*args, **kwargs)

    def _get_in_obs(self, obs):
        return obs

    def _get_out_obs(self, obs):
        return obs

    def _reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.counter += 1
        self.step_count = 0
        state = torch.zeros(self.size) + self.counter
        if tensordict is None:
            tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict = tensordict.select()
        tensordict.set("next_" + self.out_key, self._get_out_obs(state))
        tensordict.set("next_" + self._out_key, self._get_out_obs(state))
        tensordict.set("done", torch.zeros(*tensordict.shape, 1, dtype=torch.bool))
        return tensordict

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        self.step_count += 1
        tensordict = tensordict.to(self.device)
        a = tensordict.get("action")
        assert not self.is_done, "trying to execute step in done env"

        obs = self._obs_step(self._get_in_obs(tensordict.get(self._out_key)), a)
        tensordict = tensordict.select()  # empty tensordict

        tensordict.set("next_" + self.out_key, self._get_out_obs(obs))
        tensordict.set("next_" + self._out_key, self._get_out_obs(obs))

        done = torch.isclose(obs, torch.ones_like(obs) * (self.counter + 1))
        reward = done.any(-1).unsqueeze(-1)
        done = done.all(-1).unsqueeze(-1)
        tensordict.set("reward", reward.to(torch.get_default_dtype()))
        tensordict.set("done", done)
        return tensordict

    def _obs_step(self, obs, a):
        return obs + a / self.maxstep


class DiscreteActionVecPolicy:
    in_keys = ["observation"]
    out_keys = ["action"]

    def _get_in_obs(self, tensordict):
        obs = tensordict.get(*self.in_keys)
        return obs

    def __call__(self, tensordict):
        obs = self._get_in_obs(tensordict)
        max_obs = (obs == obs.max(dim=-1, keepdim=True)[0]).cumsum(-1).argmax(-1)
        k = tensordict.get(*self.in_keys).shape[-1]
        max_obs = (max_obs + 1) % k
        action = torch.nn.functional.one_hot(max_obs, k)
        tensordict.set(*self.out_keys, action)
        return tensordict


class DiscreteActionConvMockEnv(DiscreteActionVecMockEnv):
    @classmethod
    def __new__(
        cls,
        *args,
        observation_spec=None,
        action_spec=None,
        input_spec=None,
        reward_spec=None,
        from_pixels=True,
        **kwargs,
    ):
        if observation_spec is None:
            cls.out_key = "pixels"
            observation_spec = CompositeSpec(
                next_pixels=NdUnboundedContinuousTensorSpec(
                    shape=torch.Size([1, 7, 7])
                ),
                next_pixels_orig=NdUnboundedContinuousTensorSpec(
                    shape=torch.Size([1, 7, 7])
                ),
            )
        if action_spec is None:
            action_spec = OneHotDiscreteTensorSpec(7)
        if reward_spec is None:
            reward_spec = UnboundedContinuousTensorSpec()

        if input_spec is None:
            cls._out_key = "pixels_orig"
            input_spec = CompositeSpec(
                **{
                    cls._out_key: observation_spec["next_pixels_orig"],
                    "action": action_spec,
                }
            )
        return super().__new__(
            *args,
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            input_spec=input_spec,
            from_pixels=from_pixels,
            **kwargs,
        )

    def _get_out_obs(self, obs):
        obs = torch.diag_embed(obs, 0, -2, -1).unsqueeze(0)
        return obs

    def _get_in_obs(self, obs):
        return obs.diagonal(0, -1, -2).squeeze()


class DiscreteActionConvMockEnvNumpy(DiscreteActionConvMockEnv):
    @classmethod
    def __new__(
        cls,
        *args,
        observation_spec=None,
        action_spec=None,
        input_spec=None,
        reward_spec=None,
        from_pixels=True,
        **kwargs,
    ):
        if observation_spec is None:
            cls.out_key = "pixels"
            observation_spec = CompositeSpec(
                next_pixels=NdUnboundedContinuousTensorSpec(
                    shape=torch.Size([7, 7, 3])
                ),
                next_pixels_orig=NdUnboundedContinuousTensorSpec(
                    shape=torch.Size([7, 7, 3])
                ),
            )
        if action_spec is None:
            action_spec = OneHotDiscreteTensorSpec(7)
        if input_spec is None:
            cls._out_key = "pixels_orig"
            input_spec = CompositeSpec(
                **{
                    cls._out_key: observation_spec["next_pixels_orig"],
                    "action": action_spec,
                }
            )

        return super().__new__(
            *args,
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            input_spec=input_spec,
            from_pixels=from_pixels,
            **kwargs,
        )

    def _get_out_obs(self, obs):
        obs = torch.diag_embed(obs, 0, -2, -1).unsqueeze(-1)
        obs = obs.expand(*obs.shape[:-1], 3)
        return obs

    def _get_in_obs(self, obs):
        return obs.diagonal(0, -2, -3)[..., 0, :]

    def _obs_step(self, obs, a):
        return obs + a.unsqueeze(-1) / self.maxstep


class ContinuousActionConvMockEnv(ContinuousActionVecMockEnv):
    @classmethod
    def __new__(
        cls,
        *args,
        observation_spec=None,
        action_spec=None,
        input_spec=None,
        reward_spec=None,
        from_pixels=True,
        **kwargs,
    ):
        if observation_spec is None:
            cls.out_key = "pixels"
            observation_spec = CompositeSpec(
                next_pixels=NdUnboundedContinuousTensorSpec(
                    shape=torch.Size([1, 7, 7])
                ),
                next_pixels_orig=NdUnboundedContinuousTensorSpec(
                    shape=torch.Size([1, 7, 7])
                ),
            )

        if action_spec is None:
            action_spec = NdBoundedTensorSpec(-1, 1, (7,))

        if reward_spec is None:
            reward_spec = UnboundedContinuousTensorSpec()
        if input_spec is None:
            cls._out_key = "pixels_orig"
            input_spec = CompositeSpec(
                **{cls._out_key: observation_spec["next_pixels"], "action": action_spec}
            )
        return super().__new__(
            *args,
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            input_spec=input_spec,
            from_pixels=from_pixels,
            **kwargs,
        )

    def _get_out_obs(self, obs):
        obs = torch.diag_embed(obs, 0, -2, -1).unsqueeze(0)
        return obs

    def _get_in_obs(self, obs):
        return obs.diagonal(0, -1, -2).squeeze()


class ContinuousActionConvMockEnvNumpy(ContinuousActionConvMockEnv):
    @classmethod
    def __new__(
        cls,
        *args,
        observation_spec=None,
        action_spec=None,
        input_spec=None,
        reward_spec=None,
        from_pixels=True,
        **kwargs,
    ):
        if observation_spec is None:
            cls.out_key = "pixels"
            observation_spec = CompositeSpec(
                next_pixels=NdUnboundedContinuousTensorSpec(
                    shape=torch.Size([7, 7, 3])
                ),
                next_pixels_orig=NdUnboundedContinuousTensorSpec(
                    shape=torch.Size([7, 7, 3])
                ),
            )
        return super().__new__(
            *args,
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            input_spec=input_spec,
            from_pixels=from_pixels,
            **kwargs,
        )

    def _get_out_obs(self, obs):
        obs = torch.diag_embed(obs, 0, -2, -1).unsqueeze(-1)
        obs = obs.expand(*obs.shape[:-1], 3)
        return obs

    def _get_in_obs(self, obs):
        return obs.diagonal(0, -2, -3)[..., 0, :]

    def _obs_step(self, obs, a):
        return obs + a / self.maxstep


class DiscreteActionConvPolicy(DiscreteActionVecPolicy):
    in_keys = ["pixels"]
    out_keys = ["action"]

    def _get_in_obs(self, tensordict):
        obs = tensordict.get(*self.in_keys).diagonal(0, -1, -2).squeeze()
        return obs


class DummyModelBasedEnvBase(ModelBasedEnvBase):
    """Dummy environnement for Model Based RL algorithms.

    This class is meant to be used to test the model based environnement.

    Args:
        world_model (WorldModel): the world model to use for the environnement.
        device (str or torch.device, optional): the device to use for the environnement.
        dtype (torch.dtype, optional): the dtype to use for the environnement.
        batch_size (sequence of int, optional): the batch size to use for the environnement.
    """

    def __init__(
        self,
        world_model,
        device="cpu",
        dtype=None,
        batch_size=None,
    ):
        super().__init__(
            world_model,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )
        self.observation_spec = CompositeSpec(
            next_hidden_observation=NdUnboundedContinuousTensorSpec((4,))
        )
        self.input_spec = CompositeSpec(
            hidden_observation=NdUnboundedContinuousTensorSpec((4,)),
            action=NdUnboundedContinuousTensorSpec((1,)),
        )
        self.reward_spec = NdUnboundedContinuousTensorSpec((1,))

    def _reset(self, tensordict: TensorDict, **kwargs) -> TensorDict:
        td = TensorDict(
            {
                "hidden_observation": self.input_spec["hidden_observation"].rand(
                    self.batch_size
                ),
                "next_hidden_observation": self.observation_spec[
                    "next_hidden_observation"
                ].rand(self.batch_size),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return td


class ActionObsMergeLinear(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, observation, action):
        return self.linear(torch.cat([observation, action], dim=-1))
