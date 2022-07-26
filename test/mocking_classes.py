# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch
from torchrl import seed_generator
from torchrl.data.tensor_specs import (
    NdUnboundedContinuousTensorSpec,
    NdBoundedTensorSpec,
    CompositeSpec,
    MultOneHotDiscreteTensorSpec,
    BinaryDiscreteTensorSpec,
    BoundedTensorSpec,
    UnboundedContinuousTensorSpec,
    OneHotDiscreteTensorSpec,
)
from torchrl.data.tensordict.tensordict import TensorDictBase, TensorDict
from torchrl.envs.common import _EnvClass

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


class _MockEnv(_EnvClass):
    def __init__(self, seed: int = 100):
        super().__init__(
            device="cpu",
            dtype=torch.get_default_dtype(),
        )
        self.set_seed(seed)
        self.is_closed = False

        for key, item in list(self.observation_spec.items()):
            self.observation_spec[key] = item.to(torch.get_default_dtype())
        # self.action_spec = self.action_spec.to(torch.get_default_dtype())
        self.reward_spec = self.reward_spec.to(torch.get_default_dtype())

    @property
    def maxstep(self):
        return 100

    def set_seed(self, seed: int) -> int:
        self.seed = seed
        self.counter = seed % 17  # make counter a small number
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


class MockSerialEnv(_EnvClass):
    def __init__(self, device):
        super(MockSerialEnv, self).__init__(device=device)
        self.action_spec = NdUnboundedContinuousTensorSpec((1,))
        self.observation_spec = NdUnboundedContinuousTensorSpec((1,))
        self.reward_spec = NdUnboundedContinuousTensorSpec((1,))
        self.is_closed = False

    def set_seed(self, seed: int) -> int:
        assert seed >= 1
        self.seed = seed
        self.counter = seed % 17  # make counter a small number
        self.max_val = max(self.counter + 100, self.counter * 2)
        return seed_generator(seed)

    def _step(self, tensordict):
        self.counter += 1
        n = torch.tensor([self.counter]).to(self.device).to(torch.get_default_dtype())
        done = self.counter >= self.max_val
        done = torch.tensor([done], dtype=torch.bool, device=self.device)
        return TensorDict({"reward": n, "done": done, "next_observation": n}, [])

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        self.max_val = max(self.counter + 100, self.counter * 2)

        n = torch.tensor([self.counter]).to(self.device).to(torch.get_default_dtype())
        done = self.counter >= self.max_val
        done = torch.tensor([done], dtype=torch.bool, device=self.device)
        return TensorDict({"done": done, "next_observation": n}, [])

    def rand_step(self, tensordict: Optional[TensorDictBase] = None) -> TensorDictBase:
        return self.step(tensordict)


class DiscreteActionVecMockEnv(_MockEnv):
    size = 7
    observation_spec = CompositeSpec(
        next_observation=NdUnboundedContinuousTensorSpec(shape=torch.Size([size]))
    )
    action_spec = OneHotDiscreteTensorSpec(7)
    reward_spec = UnboundedContinuousTensorSpec()

    from_pixels = False

    out_key = "observation"
    _out_key = "observation_orig"
    input_spec = CompositeSpec(
        **{_out_key: observation_spec["next_observation"], "action": action_spec}
    )

    def _get_in_obs(self, obs):
        return obs

    def _get_out_obs(self, obs):
        return obs

    def _reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.counter += 1
        state = torch.zeros(self.size) + self.counter
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
    size = 7
    observation_spec = CompositeSpec(
        next_observation=NdUnboundedContinuousTensorSpec(shape=torch.Size([size]))
    )
    action_spec = NdBoundedTensorSpec(-1, 1, (7,))
    reward_spec = UnboundedContinuousTensorSpec()
    from_pixels = False

    out_key = "observation"
    _out_key = "observation_orig"
    input_spec = CompositeSpec(
        **{_out_key: observation_spec["next_observation"], "action": action_spec}
    )

    def _get_in_obs(self, obs):
        return obs

    def _get_out_obs(self, obs):
        return obs

    def _reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.counter += 1
        self.step_count = 0
        state = torch.zeros(self.size) + self.counter
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
    observation_spec = CompositeSpec(
        next_pixels=NdUnboundedContinuousTensorSpec(shape=torch.Size([1, 7, 7]))
    )
    action_spec = OneHotDiscreteTensorSpec(7)
    reward_spec = UnboundedContinuousTensorSpec()
    from_pixels = True

    out_key = "pixels"
    _out_key = "pixels_orig"
    input_spec = CompositeSpec(
        **{_out_key: observation_spec["next_pixels"], "action": action_spec}
    )

    def _get_out_obs(self, obs):
        obs = torch.diag_embed(obs, 0, -2, -1).unsqueeze(0)
        return obs

    def _get_in_obs(self, obs):
        return obs.diagonal(0, -1, -2).squeeze()


class DiscreteActionConvMockEnvNumpy(DiscreteActionConvMockEnv):
    observation_spec = CompositeSpec(
        next_pixels=NdUnboundedContinuousTensorSpec(shape=torch.Size([7, 7, 3]))
    )
    from_pixels = True

    def _get_out_obs(self, obs):
        obs = torch.diag_embed(obs, 0, -2, -1).unsqueeze(-1)
        obs = obs.expand(*obs.shape[:-1], 3)
        return obs

    def _get_in_obs(self, obs):
        return obs.diagonal(0, -2, -3)[..., 0, :]

    def _obs_step(self, obs, a):
        return obs + a.unsqueeze(-1) / self.maxstep


class ContinuousActionConvMockEnv(ContinuousActionVecMockEnv):
    observation_spec = CompositeSpec(
        next_pixels=NdUnboundedContinuousTensorSpec(shape=torch.Size([1, 7, 7]))
    )
    action_spec = NdBoundedTensorSpec(-1, 1, (7,))
    reward_spec = UnboundedContinuousTensorSpec()
    from_pixels = True

    out_key = "pixels"
    _out_key = "pixels_orig"
    input_spec = CompositeSpec(
        **{_out_key: observation_spec["next_pixels"], "action": action_spec}
    )

    def _get_out_obs(self, obs):
        obs = torch.diag_embed(obs, 0, -2, -1).unsqueeze(0)
        return obs

    def _get_in_obs(self, obs):
        return obs.diagonal(0, -1, -2).squeeze()


class ContinuousActionConvMockEnvNumpy(ContinuousActionConvMockEnv):
    observation_spec = CompositeSpec(
        next_pixels=NdUnboundedContinuousTensorSpec(shape=torch.Size([7, 7, 3]))
    )
    from_pixels = True

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
