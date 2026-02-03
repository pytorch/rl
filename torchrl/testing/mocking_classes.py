# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import random
import string

import numpy as np
import torch
import torch.nn as nn
from tensordict import tensorclass, TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import expand_right, NestedKey
from torchrl._utils import logger as torchrl_logger
from torchrl.data import (
    Binary,
    Bounded,
    Categorical,
    Composite,
    MultiOneHot,
    NonTensor,
    OneHot,
    TensorSpec,
    Unbounded,
)
from torchrl.data.utils import consolidate_spec
from torchrl.envs import Transform
from torchrl.envs.common import EnvBase
from torchrl.envs.model_based.common import ModelBasedEnvBase
from torchrl.envs.utils import (
    _terminated_or_truncated,
    check_marl_grouping,
    MarlGroupMapType,
)

spec_dict = {
    "bounded": Bounded,
    "one_hot": OneHot,
    "categorical": Categorical,
    "unbounded": Unbounded,
    "binary": Binary,
    "mult_one_hot": MultiOneHot,
    "composite": Composite,
}

default_spec_kwargs = {
    OneHot: {"n": 7},
    Categorical: {"n": 7},
    Bounded: {"minimum": -torch.ones(4), "maximum": torch.ones(4)},
    Unbounded: {
        "shape": [
            7,
        ]
    },
    Binary: {"n": 7},
    MultiOneHot: {"nvec": [7, 3, 5]},
    Composite: {},
}


def make_spec(spec_str):
    """Create a spec instance from a short spec name."""
    target_class = spec_dict[spec_str]
    return target_class(**default_spec_kwargs[target_class])


class _MockEnv(EnvBase):
    @classmethod
    def __new__(
        cls,
        *args,
        **kwargs,
    ):
        for key, item in list(cls._output_spec["full_observation_spec"].items()):
            cls._output_spec["full_observation_spec"][key] = item.to(
                torch.get_default_dtype()
            )
        reward_spec = cls._output_spec["full_reward_spec"]
        if isinstance(reward_spec, Composite):
            reward_spec = Composite(
                {
                    key: item.to(torch.get_default_dtype())
                    for key, item in reward_spec.items(True, True)
                },
                shape=reward_spec.shape,
                device=reward_spec.device,
            )
        else:
            reward_spec = reward_spec.to(torch.get_default_dtype())
        cls._output_spec["full_reward_spec"] = reward_spec
        if not isinstance(cls._output_spec["full_reward_spec"], Composite):
            cls._output_spec["full_reward_spec"] = Composite(
                reward=cls._output_spec["full_reward_spec"],
                shape=cls._output_spec["full_reward_spec"].shape[:-1],
            )
        if not isinstance(cls._output_spec["full_done_spec"], Composite):
            cls._output_spec["full_done_spec"] = Composite(
                done=cls._output_spec["full_done_spec"].clone(),
                terminated=cls._output_spec["full_done_spec"].clone(),
                shape=cls._output_spec["full_done_spec"].shape[:-1],
            )
        if not isinstance(cls._input_spec["full_action_spec"], Composite):
            cls._input_spec["full_action_spec"] = Composite(
                action=cls._input_spec["full_action_spec"],
                shape=cls._input_spec["full_action_spec"].shape[:-1],
            )
        dtype = kwargs.pop("dtype", torch.get_default_dtype())
        for spec in (cls._output_spec, cls._input_spec):
            if dtype != torch.get_default_dtype():
                for key, val in list(spec.items(True, True)):
                    if val.dtype == torch.get_default_dtype():
                        val = val.to(dtype)
                        spec[key] = val
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        *args,
        seed: int = 100,
        **kwargs,
    ):
        super().__init__(
            device=kwargs.pop("device", "cpu"),
            allow_done_after_reset=kwargs.pop("allow_done_after_reset", False),
        )
        self.set_seed(seed)
        self.is_closed = False

    @property
    def maxstep(self) -> int:
        return 100

    def _set_seed(self, seed: int | None) -> None:
        self.seed = seed
        self.counter = seed % 17  # make counter a small number

    def custom_fun(self) -> int:
        return 0

    custom_attr = 1

    @property
    def custom_prop(self) -> int:
        return 2

    @property
    def custom_td(self) -> TensorDict:
        return TensorDict({"a": torch.zeros(3)}, [])


class MockSerialEnv(EnvBase):
    """A simple counting env that is reset after a predefined max number of steps."""

    @classmethod
    def __new__(
        cls,
        *args,
        observation_spec=None,
        action_spec=None,
        state_spec=None,
        reward_spec=None,
        done_spec=None,
        **kwargs,
    ):
        batch_size = kwargs.setdefault("batch_size", torch.Size([]))
        if action_spec is None:
            action_spec = Unbounded(
                (
                    *batch_size,
                    1,
                )
            )
        if observation_spec is None:
            observation_spec = Composite(
                observation=Unbounded(
                    (
                        *batch_size,
                        1,
                    )
                ),
                shape=batch_size,
            )
        if reward_spec is None:
            reward_spec = Unbounded(
                (
                    *batch_size,
                    1,
                )
            )
        if done_spec is None:
            done_spec = Categorical(2, dtype=torch.bool, shape=(*batch_size, 1))
        if state_spec is None:
            state_spec = Composite(shape=batch_size)
        input_spec = Composite(
            full_action_spec=action_spec, full_state_spec=state_spec, shape=batch_size
        )
        cls._output_spec = Composite(shape=batch_size)
        cls._output_spec["full_reward_spec"] = reward_spec
        cls._output_spec["full_done_spec"] = done_spec
        cls._output_spec["full_observation_spec"] = observation_spec
        cls._input_spec = input_spec

        if not isinstance(cls._output_spec["full_reward_spec"], Composite):
            cls._output_spec["full_reward_spec"] = Composite(
                reward=cls._output_spec["full_reward_spec"], shape=batch_size
            )
        if not isinstance(cls._output_spec["full_done_spec"], Composite):
            cls._output_spec["full_done_spec"] = Composite(
                done=cls._output_spec["full_done_spec"], shape=batch_size
            )
        if not isinstance(cls._input_spec["full_action_spec"], Composite):
            cls._input_spec["full_action_spec"] = Composite(
                action=cls._input_spec["full_action_spec"], shape=batch_size
            )
        return super().__new__(*args, **kwargs)

    def __init__(self, device="cpu"):
        super().__init__(device=device)
        self.is_closed = False

    def _set_seed(self, seed: int | None) -> None:
        assert seed >= 1
        self.seed = seed
        self.counter = seed % 17  # make counter a small number
        self.max_val = max(self.counter + 100, self.counter * 2)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.counter += 1
        n = torch.tensor(
            [self.counter], device=self.device, dtype=torch.get_default_dtype()
        )
        done = self.counter >= self.max_val
        done = torch.tensor([done], dtype=torch.bool, device=self.device)
        return TensorDict(
            {
                "reward": n,
                "done": done,
                "terminated": done.clone(),
                "observation": n.clone(),
            },
            batch_size=[],
            device=self.device,
        )

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        self.max_val = max(self.counter + 100, self.counter * 2)

        n = torch.tensor(
            [self.counter], device=self.device, dtype=torch.get_default_dtype()
        )
        done = self.counter >= self.max_val
        done = torch.tensor([done], dtype=torch.bool, device=self.device)
        return TensorDict(
            {"done": done, "terminated": done.clone(), "observation": n},
            [],
            device=self.device,
        )

    def rand_step(self, tensordict: TensorDictBase | None = None) -> TensorDictBase:
        return self.step(tensordict)


class MockBatchedLockedEnv(EnvBase):
    """Mocks an env whose batch_size defines the size of the output tensordict."""

    @classmethod
    def __new__(
        cls,
        *args,
        observation_spec=None,
        action_spec=None,
        state_spec=None,
        reward_spec=None,
        done_spec=None,
        **kwargs,
    ):
        batch_size = kwargs.setdefault("batch_size", torch.Size([]))
        if action_spec is None:
            action_spec = Unbounded(
                (
                    *batch_size,
                    1,
                )
            )
        if state_spec is None:
            state_spec = Composite(
                observation=Unbounded(
                    (
                        *batch_size,
                        1,
                    )
                ),
                shape=batch_size,
            )
        if observation_spec is None:
            observation_spec = Composite(
                observation=Unbounded(
                    (
                        *batch_size,
                        1,
                    )
                ),
                shape=batch_size,
            )
        if reward_spec is None:
            reward_spec = Unbounded(
                (
                    *batch_size,
                    1,
                )
            )
        if done_spec is None:
            done_spec = Categorical(2, dtype=torch.bool, shape=(*batch_size, 1))
        cls._output_spec = Composite(shape=batch_size)
        cls._output_spec["full_reward_spec"] = reward_spec
        cls._output_spec["full_done_spec"] = done_spec
        cls._output_spec["full_observation_spec"] = observation_spec
        cls._input_spec = Composite(
            full_action_spec=action_spec,
            full_state_spec=state_spec,
            shape=batch_size,
        )
        if not isinstance(cls._output_spec["full_reward_spec"], Composite):
            cls._output_spec["full_reward_spec"] = Composite(
                reward=cls._output_spec["full_reward_spec"], shape=batch_size
            )
        if not isinstance(cls._output_spec["full_done_spec"], Composite):
            cls._output_spec["full_done_spec"] = Composite(
                done=cls._output_spec["full_done_spec"], shape=batch_size
            )
        if not isinstance(cls._input_spec["full_action_spec"], Composite):
            cls._input_spec["full_action_spec"] = Composite(
                action=cls._input_spec["full_action_spec"], shape=batch_size
            )
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, device="cpu", batch_size=None):
        super().__init__(device=device, batch_size=batch_size)
        self.counter = 0

    rand_step = MockSerialEnv.rand_step

    def _set_seed(self, seed: int | None) -> None:
        assert seed >= 1
        self.seed = seed
        self.counter = seed % 17  # make counter a small number
        self.max_val = max(self.counter + 100, self.counter * 2)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        if len(self.batch_size):
            leading_batch_size = (
                tensordict.shape[: -len(self.batch_size)]
                if tensordict is not None
                else []
            )
        else:
            leading_batch_size = tensordict.shape if tensordict is not None else []
        self.counter += 1
        # We use tensordict.batch_size instead of self.batch_size since this method will also be used by MockBatchedUnLockedEnv
        n = torch.full(
            [*leading_batch_size, *self.observation_spec["observation"].shape],
            self.counter,
            device=self.device,
            dtype=torch.get_default_dtype(),
        )
        done = self.counter >= self.max_val
        done = torch.full(
            (*leading_batch_size, *self.batch_size, 1),
            done,
            dtype=torch.bool,
            device=self.device,
        )
        return TensorDict(
            {"reward": n, "done": done, "terminated": done.clone(), "observation": n},
            batch_size=tensordict.batch_size,
            device=self.device,
        )

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        self.max_val = max(self.counter + 100, self.counter * 2)
        batch_size = self.batch_size
        if len(batch_size):
            leading_batch_size = (
                tensordict.shape[: -len(self.batch_size)]
                if tensordict is not None
                else []
            )
        else:
            leading_batch_size = tensordict.shape if tensordict is not None else []

        n = torch.full(
            [*leading_batch_size, *self.observation_spec["observation"].shape],
            self.counter,
            device=self.device,
            dtype=torch.get_default_dtype(),
        )
        done = self.counter >= self.max_val
        done = torch.full(
            (*leading_batch_size, *batch_size, 1),
            done,
            dtype=torch.bool,
            device=self.device,
        )
        return TensorDict(
            {"done": done, "terminated": done.clone(), "observation": n},
            [
                *leading_batch_size,
                *batch_size,
            ],
            device=self.device,
        )


class MockBatchedUnLockedEnv(MockBatchedLockedEnv):
    """Mocks an env which batch_size does not define the size of the output tensordict.

    The size of the output tensordict is defined by the input tensordict itself.

    """

    def __init__(self, device="cpu", batch_size=None):
        super().__init__(batch_size=batch_size, device=device)

    @classmethod
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, _batch_locked=False, **kwargs)


class StateLessCountingEnv(EnvBase):
    """A simple counting environment with no internal state beyond the input tensordict."""

    def __init__(self):
        self.observation_spec = Composite(
            count=Unbounded((1,), dtype=torch.int32),
            max_count=Unbounded((1,), dtype=torch.int32),
        )
        self.full_action_spec = Composite(
            action=Unbounded((1,), dtype=torch.int32),
        )
        self.full_done_spec = Composite(
            done=Unbounded((1,), dtype=torch.bool),
            termindated=Unbounded((1,), dtype=torch.bool),
            truncated=Unbounded((1,), dtype=torch.bool),
        )
        self.reward_spec = Composite(reward=Unbounded((1,), dtype=torch.float))
        super().__init__()
        self._batch_locked = False

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:

        max_count = None
        count = None
        if tensordict is not None:
            max_count = tensordict.get("max_count")
            count = tensordict.get("count")
            tensordict = TensorDict(
                batch_size=tensordict.batch_size, device=tensordict.device
            )
            shape = tensordict.batch_size
        else:
            shape = ()
            tensordict = TensorDict(device=self.device)
        tensordict.update(
            TensorDict(
                count=torch.zeros(
                    (
                        *shape,
                        1,
                    ),
                    dtype=torch.int32,
                )
                if count is None
                else count,
                max_count=torch.randint(
                    10,
                    20,
                    (
                        *shape,
                        1,
                    ),
                    dtype=torch.int32,
                )
                if max_count is None
                else max_count,
                **self.done_spec.zero(shape),
                **self.full_reward_spec.zero(shape),
            )
        )
        return tensordict

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        action = tensordict["action"]
        count = tensordict["count"] + action
        terminated = done = count >= tensordict["max_count"]
        truncated = torch.zeros_like(done)
        return TensorDict(
            count=count,
            max_count=tensordict["max_count"],
            done=done,
            terminated=terminated,
            truncated=truncated,
            reward=self.reward_spec.zero(tensordict.shape),
            batch_size=tensordict.batch_size,
            device=tensordict.device,
        )

    def _set_seed(self, seed: int | None) -> None:
        ...


class DiscreteActionVecMockEnv(_MockEnv):
    """Mock env with vector observations and discrete (one-hot/categorical) actions."""

    @classmethod
    def __new__(
        cls,
        *args,
        observation_spec=None,
        action_spec=None,
        state_spec=None,
        reward_spec=None,
        done_spec=None,
        from_pixels=False,
        categorical_action_encoding=False,
        **kwargs,
    ):
        batch_size = kwargs.setdefault("batch_size", torch.Size([]))
        size = cls.size = 7
        if observation_spec is None:
            cls.out_key = "observation"
            observation_spec = Composite(
                observation=Unbounded(shape=torch.Size([*batch_size, size])),
                observation_orig=Unbounded(shape=torch.Size([*batch_size, size])),
                shape=batch_size,
            )
        if action_spec is None:
            if categorical_action_encoding:
                action_spec_cls = Categorical
                action_spec = action_spec_cls(n=7, shape=batch_size)
            else:
                action_spec_cls = OneHot
                action_spec = action_spec_cls(n=7, shape=(*batch_size, 7))
        if reward_spec is None:
            reward_spec = Composite(reward=Unbounded(shape=(1,)))
        if done_spec is None:
            done_spec = Composite(
                terminated=Categorical(2, dtype=torch.bool, shape=(*batch_size, 1))
            )

        if state_spec is None:
            cls._out_key = "observation_orig"
            state_spec = Composite(
                {
                    cls._out_key: observation_spec["observation"],
                },
                shape=batch_size,
            )
        cls._output_spec = Composite(shape=batch_size)
        cls._output_spec["full_reward_spec"] = reward_spec
        cls._output_spec["full_done_spec"] = done_spec
        cls._output_spec["full_observation_spec"] = observation_spec
        cls._input_spec = Composite(
            full_action_spec=action_spec,
            full_state_spec=state_spec,
            shape=batch_size,
        )
        cls.from_pixels = from_pixels
        cls.categorical_action_encoding = categorical_action_encoding
        return super().__new__(*args, **kwargs)

    def _get_in_obs(self, obs):
        return obs

    def _get_out_obs(self, obs):
        return obs

    def _reset(self, tensordict: TensorDictBase = None) -> TensorDictBase:
        self.counter += 1
        state = torch.zeros(self.size) + self.counter
        if tensordict is None:
            tensordict = TensorDict(batch_size=self.batch_size, device=self.device)
        tensordict = tensordict.empty().set(self.out_key, self._get_out_obs(state))
        tensordict = tensordict.set(self._out_key, self._get_out_obs(state))
        tensordict.set("done", torch.zeros(*tensordict.shape, 1, dtype=torch.bool))
        tensordict.set(
            "terminated", torch.zeros(*tensordict.shape, 1, dtype=torch.bool)
        )
        return tensordict

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        tensordict = tensordict.to(self.device)
        a = tensordict.get("action")

        if not self.categorical_action_encoding:
            assert (a.sum(-1) == 1).all()

        obs = self._get_in_obs(tensordict.get(self._out_key)) + a / self.maxstep
        tensordict = tensordict.empty()

        tensordict.set(self.out_key, self._get_out_obs(obs))
        tensordict.set(self._out_key, self._get_out_obs(obs))

        done = torch.isclose(obs, torch.ones_like(obs) * (self.counter + 1))
        reward = done.any(-1).unsqueeze(-1)

        # set done to False
        done = torch.zeros_like(done).all(-1).unsqueeze(-1)
        tensordict.set("reward", reward.to(torch.get_default_dtype()))
        tensordict.set("done", done)
        tensordict.set("terminated", done.clone())
        return tensordict


class ContinuousActionVecMockEnv(_MockEnv):
    """Mock env with vector observations and continuous (bounded) actions."""

    adapt_dtype: bool = True

    @classmethod
    def __new__(
        cls,
        *args,
        observation_spec=None,
        action_spec=None,
        state_spec=None,
        reward_spec=None,
        done_spec=None,
        from_pixels=False,
        **kwargs,
    ):
        batch_size = kwargs.setdefault("batch_size", torch.Size([]))
        size = cls.size = 7
        if observation_spec is None:
            cls.out_key = "observation"
            observation_spec = Composite(
                observation=Unbounded(shape=torch.Size([*batch_size, size])),
                observation_orig=Unbounded(shape=torch.Size([*batch_size, size])),
                shape=batch_size,
            )
        if action_spec is None:
            action_spec = Bounded(
                -1,
                1,
                (
                    *batch_size,
                    7,
                ),
            )
        if reward_spec is None:
            reward_spec = Unbounded(shape=(*batch_size, 1))
        if done_spec is None:
            done_spec = Categorical(2, dtype=torch.bool, shape=(*batch_size, 1))

        if state_spec is None:
            cls._out_key = "observation_orig"
            state_spec = Composite(
                {
                    cls._out_key: observation_spec["observation"],
                },
                shape=batch_size,
            )
        cls._output_spec = Composite(shape=batch_size)
        cls._output_spec["full_reward_spec"] = reward_spec
        cls._output_spec["full_done_spec"] = done_spec
        cls._output_spec["full_observation_spec"] = observation_spec
        cls._input_spec = Composite(
            full_action_spec=action_spec,
            full_state_spec=state_spec,
            shape=batch_size,
        )
        cls.from_pixels = from_pixels
        return super().__new__(cls, *args, **kwargs)

    def _get_in_obs(self, obs):
        return obs

    def _get_out_obs(self, obs):
        return obs

    def _reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.counter += 1
        self.step_count = 0
        # state = torch.zeros(self.size) + self.counter
        if tensordict is None:
            tensordict = TensorDict(batch_size=self.batch_size, device=self.device)

        tensordict = tensordict.empty()
        tensordict.update(self.observation_spec.rand())
        # tensordict.set("next_" + self.out_key, self._get_out_obs(state))
        # tensordict.set("next_" + self._out_key, self._get_out_obs(state))
        tensordict.set("done", torch.zeros(*tensordict.shape, 1, dtype=torch.bool))
        tensordict.set(
            "terminated", torch.zeros(*tensordict.shape, 1, dtype=torch.bool)
        )
        return tensordict

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        self.step_count += 1
        tensordict = tensordict.to(self.device)
        a = tensordict.get("action")

        obs = self._obs_step(self._get_in_obs(tensordict.get(self._out_key)), a)

        tensordict = tensordict.empty()  # empty tensordict

        tensordict.set(self.out_key, self._get_out_obs(obs))
        tensordict.set(self._out_key, self._get_out_obs(obs))

        done = torch.isclose(obs, torch.ones_like(obs) * (self.counter + 1))
        while done.shape != tensordict.shape:
            done = done.any(-1)
        done = reward = done.unsqueeze(-1)
        tensordict.set(
            "reward",
            reward.to(
                self.reward_spec.dtype
                if self.adapt_dtype
                else torch.get_default_dtype()
            ).expand(self.reward_spec.shape),
        )
        tensordict.set("done", done)
        tensordict.set("terminated", done)
        return tensordict

    def _obs_step(self, obs, a):
        return obs + a / self.maxstep


class DiscreteActionVecPolicy(TensorDictModuleBase):
    """Deterministic policy for `DiscreteActionVecMockEnv`-like observations."""

    in_keys = ["observation"]
    out_keys = ["action"]

    def _get_in_obs(self, tensordict):
        obs = tensordict.get(*self.in_keys)
        return obs

    def __call__(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs = self._get_in_obs(tensordict)
        max_obs = (obs == obs.max(dim=-1, keepdim=True)[0]).cumsum(-1).argmax(-1)
        k = tensordict.get(*self.in_keys).shape[-1]
        max_obs = (max_obs + 1) % k
        action = torch.nn.functional.one_hot(max_obs, k)
        tensordict.set(*self.out_keys, action)
        return tensordict


class DiscreteActionConvMockEnv(DiscreteActionVecMockEnv):
    """Mock env with image-like observations and discrete (one-hot) actions."""

    @classmethod
    def __new__(
        cls,
        *args,
        observation_spec=None,
        action_spec=None,
        state_spec=None,
        reward_spec=None,
        done_spec=None,
        from_pixels=True,
        **kwargs,
    ):
        batch_size = kwargs.setdefault("batch_size", torch.Size([]))
        if observation_spec is None:
            cls.out_key = "pixels"
            observation_spec = Composite(
                pixels=Unbounded(shape=torch.Size([*batch_size, 1, 7, 7])),
                pixels_orig=Unbounded(shape=torch.Size([*batch_size, 1, 7, 7])),
                shape=batch_size,
            )
        if action_spec is None:
            action_spec = OneHot(7, shape=(*batch_size, 7))
        if reward_spec is None:
            reward_spec = Unbounded(shape=(*batch_size, 1))
        if done_spec is None:
            done_spec = Categorical(2, dtype=torch.bool, shape=(*batch_size, 1))

        if state_spec is None:
            cls._out_key = "pixels_orig"
            state_spec = Composite(
                {
                    cls._out_key: observation_spec["pixels_orig"].clone(),
                },
                shape=batch_size,
            )
        return super().__new__(
            *args,
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            state_spec=state_spec,
            from_pixels=from_pixels,
            done_spec=done_spec,
            **kwargs,
        )

    def _get_out_obs(self, obs):
        obs = torch.diag_embed(obs, 0, -2, -1).unsqueeze(0)
        return obs

    def _get_in_obs(self, obs):
        return obs.diagonal(0, -1, -2).squeeze()


class DiscreteActionConvMockEnvNumpy(DiscreteActionConvMockEnv):
    """Numpy-style variant of `DiscreteActionConvMockEnv` (channels-last pixels)."""

    @classmethod
    def __new__(
        cls,
        *args,
        observation_spec=None,
        action_spec=None,
        state_spec=None,
        reward_spec=None,
        done_spec=None,
        from_pixels=True,
        categorical_action_encoding=False,
        **kwargs,
    ):
        batch_size = kwargs.setdefault("batch_size", torch.Size([]))
        if observation_spec is None:
            cls.out_key = "pixels"
            observation_spec = Composite(
                pixels=Unbounded(shape=torch.Size([*batch_size, 7, 7, 3])),
                pixels_orig=Unbounded(shape=torch.Size([*batch_size, 7, 7, 3])),
                shape=batch_size,
            )
        if action_spec is None:
            action_spec_cls = Categorical if categorical_action_encoding else OneHot
            action_spec = action_spec_cls(7, shape=(*batch_size, 7))
        if state_spec is None:
            cls._out_key = "pixels_orig"
            state_spec = Composite(
                {
                    cls._out_key: observation_spec["pixels_orig"],
                },
                shape=batch_size,
            )

        return super().__new__(
            *args,
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            state_spec=state_spec,
            from_pixels=from_pixels,
            categorical_action_encoding=categorical_action_encoding,
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
    """Mock env with image-like observations and continuous (bounded) actions."""

    @classmethod
    def __new__(
        cls,
        *args,
        observation_spec=None,
        action_spec=None,
        state_spec=None,
        reward_spec=None,
        done_spec=None,
        from_pixels=True,
        pixel_shape=None,
        **kwargs,
    ):
        batch_size = kwargs.setdefault("batch_size", torch.Size([]))
        if pixel_shape is None:
            pixel_shape = [1, 7, 7]
        if observation_spec is None:
            cls.out_key = "pixels"
            observation_spec = Composite(
                pixels=Unbounded(shape=torch.Size([*batch_size, *pixel_shape])),
                pixels_orig=Unbounded(shape=torch.Size([*batch_size, *pixel_shape])),
                shape=batch_size,
            )

        if action_spec is None:
            action_spec = Bounded(-1, 1, [*batch_size, pixel_shape[-1]])
        if reward_spec is None:
            reward_spec = Unbounded(shape=(*batch_size, 1))
        if done_spec is None:
            done_spec = Categorical(2, dtype=torch.bool, shape=(*batch_size, 1))
        if state_spec is None:
            cls._out_key = "pixels_orig"
            state_spec = Composite(
                {cls._out_key: observation_spec["pixels"]}, shape=batch_size
            )
        return super().__new__(
            *args,
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            from_pixels=from_pixels,
            state_spec=state_spec,
            done_spec=done_spec,
            **kwargs,
        )

    def _get_out_obs(self, obs):
        obs = torch.diag_embed(obs, 0, -2, -1)
        return obs

    def _get_in_obs(self, obs):
        obs = obs.diagonal(0, -1, -2)
        return obs


class ContinuousActionConvMockEnvNumpy(ContinuousActionConvMockEnv):
    """Numpy-style variant of `ContinuousActionConvMockEnv` (channels-last pixels)."""

    @classmethod
    def __new__(
        cls,
        *args,
        observation_spec=None,
        action_spec=None,
        state_spec=None,
        reward_spec=None,
        done_spec=None,
        from_pixels=True,
        **kwargs,
    ):
        batch_size = kwargs.setdefault("batch_size", torch.Size([]))
        if observation_spec is None:
            cls.out_key = "pixels"
            observation_spec = Composite(
                pixels=Unbounded(shape=torch.Size([*batch_size, 7, 7, 3])),
                pixels_orig=Unbounded(shape=torch.Size([*batch_size, 7, 7, 3])),
            )
        return super().__new__(
            *args,
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            state_spec=state_spec,
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
    """Policy for discrete-action convolutional mock environments."""

    in_keys = ["pixels"]
    out_keys = ["action"]

    def _get_in_obs(self, tensordict):
        obs = tensordict.get(*self.in_keys).diagonal(0, -1, -2).squeeze()
        return obs


class DummyModelBasedEnvBase(ModelBasedEnvBase):
    """Dummy environment for Model Based RL sota-implementations.

    This class is meant to be used to test the model based environment.

    Args:
        world_model (WorldModel): the world model to use for the environment.
        device (str or torch.device, optional): the device to use for the environment.
        dtype (torch.dtype, optional): the dtype to use for the environment.
        batch_size (sequence of int, optional): the batch size to use for the environment.
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
            batch_size=batch_size,
        )
        self.observation_spec = Composite(
            hidden_observation=Unbounded(
                (
                    *self.batch_size,
                    4,
                )
            ),
            shape=self.batch_size,
        )
        self.state_spec = Composite(
            hidden_observation=Unbounded(
                (
                    *self.batch_size,
                    4,
                )
            ),
            shape=self.batch_size,
        )
        self.action_spec = Unbounded(
            (
                *self.batch_size,
                1,
            )
        )
        self.reward_spec = Unbounded(
            (
                *self.batch_size,
                1,
            )
        )

    def _reset(self, tensordict: TensorDict, **kwargs) -> TensorDict:
        td = TensorDict(
            {
                "hidden_observation": self.state_spec["hidden_observation"].rand(),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return td


class ActionObsMergeLinear(nn.Module):
    """Linear layer that consumes concatenated observation and action tensors."""

    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, observation, action):
        return self.linear(torch.cat([observation, action], dim=-1))


class CountingEnvCountPolicy(TensorDictModuleBase):
    """Policy that always returns an increment action for counting environments."""

    def __init__(self, action_spec: TensorSpec, action_key: NestedKey = "action"):
        super().__init__()
        assert not isinstance(action_spec, Composite)
        self.action_spec = action_spec
        self.action_key = action_key
        self.in_keys = []
        self.out_keys = [action_key]

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        return td.set(self.action_key, self.action_spec.zero() + 1)


class CountingEnvCountModule(nn.Module):
    """Module that returns a constant increment action given an action spec."""

    def __init__(self, action_spec: TensorSpec):
        super().__init__()
        self.action_spec = action_spec

    def forward(self, t):
        return self.action_spec.zero() + 1


class CountingEnv(EnvBase):
    """An env that is done after a given number of steps.

    The action is the count increment.

    """

    def __init__(self, max_steps: int = 5, start_val: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.max_steps = max_steps
        self.start_val = start_val

        self.observation_spec = Composite(
            observation=Unbounded(
                (
                    *self.batch_size,
                    1,
                ),
                dtype=torch.int32,
                device=self.device,
            ),
            shape=self.batch_size,
            device=self.device,
        )
        self.reward_spec = Unbounded(
            (
                *self.batch_size,
                1,
            ),
            device=self.device,
        )
        self.done_spec = Categorical(
            2,
            dtype=torch.bool,
            shape=(*self.batch_size, 1),
            device=self.device,
        )
        self.action_spec = Binary(n=1, shape=[*self.batch_size, 1], device=self.device)
        self.register_buffer(
            "count",
            torch.zeros((*self.batch_size, 1), device=self.device, dtype=torch.int),
        )

    def _set_seed(self, seed: int | None) -> None:
        torch.manual_seed(seed)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is not None and "_reset" in tensordict.keys():
            _reset = tensordict.get("_reset")
            self.count[_reset] = self.start_val
        else:
            self.count[:] = self.start_val
        return TensorDict(
            source={
                "observation": self.count.clone(),
                "done": self.count > self.max_steps,
                "terminated": self.count > self.max_steps,
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        action = tensordict.get(self.action_key)
        try:
            device = self.full_action_spec[self.action_key].device
        except KeyError:
            device = self.device
        self.count += action.to(
            dtype=torch.int,
            device=device if self.device is None else self.device,
        )
        if self.reward_keys:
            reward_spec = self.full_reward_spec[self.reward_keys[0]]
            reward_spec_dtype = reward_spec.dtype
        else:
            reward_spec_dtype = torch.get_default_dtype()
        tensordict = TensorDict(
            source={
                "observation": self.count.clone(),
                "done": self.count > self.max_steps,
                "terminated": self.count > self.max_steps,
                "reward": torch.zeros_like(self.count, dtype=reward_spec_dtype),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict


def get_random_string(min_size, max_size):
    """Return a random ASCII lowercase string with length in [min_size, max_size]."""
    size = random.randint(min_size, max_size)
    return "".join(random.choice(string.ascii_lowercase) for _ in range(size))


class CountingEnvWithString(CountingEnv):
    """`CountingEnv` variant that adds a non-tensor string observation."""

    def __init__(self, *args, **kwargs):
        self.max_size = kwargs.pop("max_size", 30)
        self.min_size = kwargs.pop("min_size", 4)
        super().__init__(*args, **kwargs)
        self.observation_spec.set(
            "string",
            NonTensor(
                shape=self.batch_size,
                device=self.device,
                example_data=self.get_random_string(),
            ),
        )

    def get_random_string(self):
        return get_random_string(self.min_size, self.max_size)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        res = super()._reset(tensordict, **kwargs)
        random_string = self.get_random_string()
        res["string"] = random_string
        return res

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        res = super()._step(tensordict)
        random_string = self.get_random_string()
        res["string"] = random_string
        return res


class MultiAgentCountingEnv(EnvBase):
    """A multi-agent env that is done after a given number of steps.

    All agents have identical specs.

    The count is incremented by 1 on each step.

    """

    def __init__(
        self,
        n_agents: int,
        group_map: MarlGroupMapType
        | dict[str, list[str]] = MarlGroupMapType.ALL_IN_ONE_GROUP,
        max_steps: int = 5,
        start_val: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_steps = max_steps
        self.start_val = start_val
        self.n_agents = n_agents
        self.agent_names = [f"agent_{idx}" for idx in range(n_agents)]

        if isinstance(group_map, MarlGroupMapType):
            group_map = group_map.get_group_map(self.agent_names)
        check_marl_grouping(group_map, self.agent_names)

        self.group_map = group_map

        observation_specs = {}
        reward_specs = {}
        done_specs = {}
        action_specs = {}

        for group_name, agents in group_map.items():
            observation_specs[group_name] = {}
            reward_specs[group_name] = {}
            done_specs[group_name] = {}
            action_specs[group_name] = {}

            for agent_name in agents:
                observation_specs[group_name][agent_name] = Composite(
                    observation=Unbounded(
                        (
                            *self.batch_size,
                            3,
                            4,
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    shape=self.batch_size,
                    device=self.device,
                )
                reward_specs[group_name][agent_name] = Composite(
                    reward=Unbounded(
                        (
                            *self.batch_size,
                            1,
                        ),
                        device=self.device,
                    ),
                    shape=self.batch_size,
                    device=self.device,
                )
                done_specs[group_name][agent_name] = Composite(
                    done=Categorical(
                        2,
                        dtype=torch.bool,
                        shape=(
                            *self.batch_size,
                            1,
                        ),
                        device=self.device,
                    ),
                    shape=self.batch_size,
                    device=self.device,
                )
                action_specs[group_name][agent_name] = Composite(
                    action=Binary(n=1, shape=[*self.batch_size, 1], device=self.device),
                    shape=self.batch_size,
                    device=self.device,
                )

        self.observation_spec = Composite(observation_specs)
        self.reward_spec = Composite(reward_specs)
        self.done_spec = Composite(done_specs)
        self.action_spec = Composite(action_specs)
        self.register_buffer(
            "count",
            torch.zeros((*self.batch_size, 1), device=self.device, dtype=torch.int),
        )

    def _set_seed(self, seed: int | None) -> None:
        torch.manual_seed(seed)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is not None and "_reset" in tensordict.keys():
            _reset = tensordict.get("_reset")
            self.count[_reset] = self.start_val
        else:
            self.count[:] = self.start_val

        source = {}
        for group_name, agents in self.group_map.items():
            source[group_name] = {}
            for agent_name in agents:
                source[group_name][agent_name] = TensorDict(
                    source={
                        "observation": torch.rand(
                            (*self.batch_size, 3, 4),
                            device=self.device,
                            dtype=self.full_observation_spec[
                                group_name, agent_name, "observation"
                            ].dtype,
                        ),
                        "done": self.count > self.max_steps,
                        "terminated": self.count > self.max_steps,
                    },
                    batch_size=self.batch_size,
                    device=self.device,
                )

        tensordict = TensorDict(source, batch_size=self.batch_size, device=self.device)
        return tensordict

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        self.count += 1
        source = {}
        for group_name, agents in self.group_map.items():
            source[group_name] = {}
            for agent_name in agents:
                source[group_name][agent_name] = TensorDict(
                    source={
                        "observation": torch.rand(
                            (*self.batch_size, 3, 4),
                            device=self.device,
                            dtype=self.full_observation_spec[
                                group_name, agent_name, "observation"
                            ].dtype,
                        ),
                        "done": self.count > self.max_steps,
                        "terminated": self.count > self.max_steps,
                        "reward": torch.zeros_like(
                            self.count,
                            dtype=self.full_reward_spec[
                                group_name, agent_name, "reward"
                            ].dtype,
                        ),
                    },
                    batch_size=self.batch_size,
                    device=self.device,
                )
        tensordict = TensorDict(source, batch_size=self.batch_size, device=self.device)
        return tensordict


class IncrementingEnv(CountingEnv):
    """`CountingEnv` variant that always increments the count by 1 regardless of action."""

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        self.count += 1  # The only difference with CountingEnv.
        tensordict = TensorDict(
            source={
                "observation": self.count.clone(),
                "done": self.count > self.max_steps,
                "terminated": self.count > self.max_steps,
                "reward": torch.zeros_like(self.count, dtype=torch.float),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict


class NestedCountingEnv(CountingEnv):
    """Counting environment with nested observation/action/reward/done structures."""

    def __init__(
        self,
        max_steps: int = 5,
        start_val: int = 0,
        nest_obs_action: bool = True,
        nest_done: bool = True,
        nest_reward: bool = True,
        nested_dim: int = 3,
        has_root_done: bool = False,
        **kwargs,
    ):
        super().__init__(max_steps=max_steps, start_val=start_val, **kwargs)

        self.nested_dim = nested_dim
        self.has_root_done = has_root_done

        self.nested_obs_action = nest_obs_action
        self.nested_done = nest_done
        self.nested_reward = nest_reward

        if self.nested_obs_action:
            self.observation_spec = Composite(
                {
                    "data": Composite(
                        {
                            "states": self.observation_spec["observation"]
                            .unsqueeze(-1)
                            .expand(*self.batch_size, self.nested_dim, 1)
                        },
                        shape=(
                            *self.batch_size,
                            self.nested_dim,
                        ),
                    )
                },
                shape=self.batch_size,
            )
            action_spec = self.full_action_spec[self.action_key]
            assert not isinstance(action_spec, Composite)
            self.full_action_spec = Composite(
                {
                    "data": Composite(
                        {
                            "action": action_spec.unsqueeze(-1).expand(
                                *self.batch_size, self.nested_dim, 1
                            )
                        },
                        shape=(
                            *self.batch_size,
                            self.nested_dim,
                        ),
                    )
                },
                shape=self.batch_size,
            )

        if self.nested_reward:
            self.reward_spec = Composite(
                {
                    "data": Composite(
                        {
                            "reward": self.reward_spec.unsqueeze(-1).expand(
                                *self.batch_size, self.nested_dim, 1
                            )
                        },
                        shape=(
                            *self.batch_size,
                            self.nested_dim,
                        ),
                    )
                },
                shape=self.batch_size,
            )

        if self.nested_done:
            done_spec = self.full_done_spec.unsqueeze(-1).expand(
                *self.batch_size, self.nested_dim
            )
            done_spec = Composite(
                {"data": done_spec},
                shape=self.batch_size,
            )
            if self.has_root_done:
                done_spec["done"] = Categorical(
                    2,
                    shape=(
                        *self.batch_size,
                        1,
                    ),
                    dtype=torch.bool,
                )
            self.done_spec = done_spec

    def _reset(self, tensordict):

        # check that reset works as expected
        if tensordict is not None:
            if self.nested_done:
                if not self.has_root_done:
                    assert "_reset" not in tensordict.keys()
            else:
                assert ("data", "_reset") not in tensordict.keys(True)

        tensordict_reset = super()._reset(tensordict)

        if self.nested_done:
            for done_key in self.done_keys:
                if isinstance(done_key, str):
                    continue
                else:
                    done = tensordict_reset.pop(done_key[-1], None)
                if done is None:
                    continue
                tensordict_reset.set(
                    done_key,
                    (done.unsqueeze(-2).expand(*self.batch_size, self.nested_dim, 1)),
                )
        if self.nested_obs_action:
            obs = tensordict_reset.pop("observation")
            tensordict_reset.set(
                ("data", "states"),
                (obs.unsqueeze(-1).expand(*self.batch_size, self.nested_dim, 1)),
            )
        if "data" in tensordict_reset.keys():
            tensordict_reset.get("data").batch_size = (
                *self.batch_size,
                self.nested_dim,
            )
        return tensordict_reset

    def _step(self, tensordict):
        if self.nested_obs_action:
            tensordict = tensordict.clone()
            tensordict["data"].batch_size = self.batch_size
            tensordict[self.action_key] = tensordict[self.action_key].max(-2)[0]
        next_tensordict = super()._step(tensordict)
        if self.nested_obs_action:
            tensordict[self.action_key] = (
                tensordict[self.action_key]
                .unsqueeze(-1)
                .expand(*self.batch_size, self.nested_dim, 1)
            )
        if "data" in tensordict.keys():
            tensordict["data"].batch_size = (*self.batch_size, self.nested_dim)
        if self.nested_done:
            for done_key in self.done_keys:
                if isinstance(done_key, str):
                    continue
                else:
                    done = next_tensordict.pop(done_key[-1], None)
                if done is None:
                    continue
                next_tensordict.set(
                    done_key,
                    (done.unsqueeze(-1).expand(*self.batch_size, self.nested_dim, 1)),
                )
        if self.nested_obs_action:
            next_tensordict.set(
                ("data", "states"),
                (
                    next_tensordict.pop("observation")
                    .unsqueeze(-1)
                    .expand(*self.batch_size, self.nested_dim, 1)
                ),
            )
        if self.nested_reward:
            next_tensordict.set(
                self.reward_key,
                (
                    next_tensordict.pop("reward")
                    .unsqueeze(-1)
                    .expand(*self.batch_size, self.nested_dim, 1)
                ),
            )
        if "data" in next_tensordict.keys():
            next_tensordict.get("data").batch_size = (*self.batch_size, self.nested_dim)
        return next_tensordict


class CountingBatchedEnv(EnvBase):
    """An env that is done after a given number of steps.

    The action is the count increment.

    Unlike ``CountingEnv``, different envs of the batch can have different max_steps
    """

    def __init__(
        self,
        max_steps: torch.Tensor = None,
        start_val: torch.Tensor = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if max_steps is None:
            max_steps = torch.tensor(5)
        if start_val is None:
            start_val = torch.zeros((), dtype=torch.int32)
        if max_steps.shape != self.batch_size:
            raise RuntimeError(
                f"batch_size and max_steps shape must match. Got self.batch_size={self.batch_size} and max_steps.shape={max_steps.shape}."
            )

        self.max_steps = max_steps

        self.observation_spec = Composite(
            observation=Unbounded(
                (
                    *self.batch_size,
                    1,
                ),
                dtype=torch.int32,
            ),
            shape=self.batch_size,
        )
        self.reward_spec = Unbounded(
            (
                *self.batch_size,
                1,
            )
        )
        self.done_spec = Categorical(
            2,
            dtype=torch.bool,
            shape=(
                *self.batch_size,
                1,
            ),
        )
        self.action_spec = Binary(n=1, shape=[*self.batch_size, 1])

        self.count = torch.zeros(
            (*self.batch_size, 1), device=self.device, dtype=torch.int
        )
        if start_val.numel() == self.batch_size.numel():
            self.start_val = start_val.view(*self.batch_size, 1)
        elif start_val.numel() <= 1:
            self.start_val = start_val.expand_as(self.count)

    def _set_seed(self, seed: int | None) -> None:
        torch.manual_seed(seed)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is not None and "_reset" in tensordict.keys():
            _reset = tensordict.get("_reset")
            self.count[_reset] = self.start_val[_reset].view_as(self.count[_reset])
        else:
            self.count[:] = self.start_val.view_as(self.count)
        return TensorDict(
            source={
                "observation": self.count.clone(),
                "done": self.count > self.max_steps.view_as(self.count),
                "terminated": self.count > self.max_steps.view_as(self.count),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        action = tensordict.get("action")
        self.count += action.to(torch.int).view_as(self.count)
        tensordict = TensorDict(
            source={
                "observation": self.count.clone(),
                "done": self.count > self.max_steps.unsqueeze(-1),
                "terminated": self.count > self.max_steps.unsqueeze(-1),
                "reward": torch.zeros_like(self.count, dtype=torch.float),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict


class HeterogeneousCountingEnvPolicy(TensorDictModuleBase):
    """Policy for `HeterogeneousCountingEnv` that outputs increment (or zero) actions."""

    def __init__(self, full_action_spec: TensorSpec, count: bool = True):
        super().__init__()
        self.full_action_spec = full_action_spec
        self.count = count

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        action_td = self.full_action_spec.zero()
        if self.count:
            action_td.apply_(lambda x: x + 1)
        return td.update(action_td)


class HeterogeneousCountingEnv(EnvBase):
    """A heterogeneous, counting Env."""

    def __init__(self, max_steps: int = 5, start_val: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.n_nested_dim = 3
        self.max_steps = max_steps
        self.start_val = start_val

        count = torch.zeros((*self.batch_size, 1), device=self.device, dtype=torch.int)
        count[:] = self.start_val

        self.register_buffer("count", count)
        self._make_specs()

    def _make_specs(self):
        obs_specs = []
        action_specs = []
        for index in range(self.n_nested_dim):
            obs_specs.append(self.get_agent_obs_spec(index))
            action_specs.append(self.get_agent_action_spec(index))
        obs_specs = torch.stack(obs_specs, dim=0)
        obs_spec_unlazy = consolidate_spec(obs_specs)
        action_specs = torch.stack(action_specs, dim=0)

        self.observation_spec_unbatched = Composite(
            lazy=obs_spec_unlazy,
            state=Unbounded(shape=(64, 64, 3)),
            device=self.device,
        )

        self.action_spec_unbatched = Composite(
            lazy=action_specs,
            device=self.device,
        )
        self.reward_spec_unbatched = Composite(
            {
                "lazy": Composite(
                    {"reward": Unbounded(shape=(self.n_nested_dim, 1))},
                    shape=(self.n_nested_dim,),
                )
            },
            device=self.device,
        )
        self.done_spec_unbatched = Composite(
            {
                "lazy": Composite(
                    {
                        "done": Categorical(
                            n=2,
                            shape=(self.n_nested_dim, 1),
                            dtype=torch.bool,
                        ),
                    },
                    shape=(self.n_nested_dim,),
                )
            },
            device=self.device,
        )

    def get_agent_obs_spec(self, i):
        camera = Bounded(low=0, high=200, shape=(7, 7, 3))
        vector_3d = Unbounded(shape=(3,))
        vector_2d = Unbounded(shape=(2,))
        lidar = Bounded(low=0, high=5, shape=(8,))

        tensor_0 = Unbounded(shape=(1,))
        tensor_1 = Bounded(low=0, high=3, shape=(1, 2))
        tensor_2 = Unbounded(shape=(1, 2, 3))

        if i == 0:
            return Composite(
                {
                    "camera": camera,
                    "lidar": lidar,
                    "vector": vector_3d,
                    "tensor_0": tensor_0,
                },
                device=self.device,
            )
        elif i == 1:
            return Composite(
                {
                    "camera": camera,
                    "lidar": lidar,
                    "vector": vector_2d,
                    "tensor_1": tensor_1,
                },
                device=self.device,
            )
        elif i == 2:
            return Composite(
                {
                    "camera": camera,
                    "vector": vector_2d,
                    "tensor_2": tensor_2,
                },
                device=self.device,
            )
        else:
            raise ValueError(f"Index {i} undefined for index 3")

    def get_agent_action_spec(self, i):
        action_3d = Bounded(low=-1, high=1, shape=(3,))
        action_2d = Bounded(low=-1, high=1, shape=(2,))

        # Some have 2d action and some 3d
        # TODO Introduce composite heterogeneous actions
        if i == 0:
            ret = action_3d
        elif i == 1:
            ret = action_2d
        elif i == 2:
            ret = action_2d
        else:
            raise ValueError(f"Index {i} undefined for index 3")

        return Composite({"action": ret})

    def _reset(
        self,
        tensordict: TensorDictBase = None,
        **kwargs,
    ) -> TensorDictBase:
        if tensordict is not None and self.reset_keys[0] in tensordict.keys(True):
            _reset = tensordict.get(self.reset_keys[0]).squeeze(-1).any(-1)
            self.count[_reset] = self.start_val
        else:
            self.count[:] = self.start_val

        reset_td = self.observation_spec.zero()
        reset_td.apply_(lambda x: x + expand_right(self.count, x.shape))
        reset_td.update(self.output_spec["full_done_spec"].zero())

        assert reset_td.batch_size == self.batch_size
        for key in reset_td.keys(True):
            assert "_reset" not in key
        return reset_td

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        actions = torch.zeros_like(self.count.squeeze(-1), dtype=torch.bool)
        for i in range(self.n_nested_dim):
            action = tensordict["lazy"][..., i]["action"]
            action = action[..., 0].to(torch.bool)
            actions += action

        self.count += actions.unsqueeze(-1).to(torch.int)

        td = self.observation_spec.zero()
        td.apply_(lambda x: x + expand_right(self.count, x.shape))
        td.update(self.output_spec["full_done_spec"].zero())
        td.update(self.output_spec["full_reward_spec"].zero())

        assert td.batch_size == self.batch_size
        for done_key in self.done_keys:
            td[done_key] = expand_right(
                self.count > self.max_steps,
                self.full_done_spec[done_key].shape,
            )

        return td

    def _set_seed(self, seed: int | None) -> None:
        torch.manual_seed(seed)


class MultiKeyCountingEnvPolicy(TensorDictModuleBase):
    """Policy for `MultiKeyCountingEnv` that can count deterministically or stochastically."""

    def __init__(
        self,
        full_action_spec: TensorSpec,
        count: bool = True,
        deterministic: bool = False,
    ):
        super().__init__()
        if not deterministic and not count:
            raise ValueError("Not counting policy is always deterministic")

        self.full_action_spec = full_action_spec
        self.count = count
        self.deterministic = deterministic

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        action_td = self.full_action_spec.zero()
        if self.count:
            if self.deterministic:
                action_td["nested_1", "action"] += 1
                action_td["nested_2", "azione"] += 1
                action_td["action"][..., 1] = 1
            else:
                # We choose an action at random
                choice = torch.randint(0, 3, ()).item()
                if choice == 0:
                    action_td["nested_1", "action"] += 1
                elif choice == 1:
                    action_td["nested_2", "azione"] += 1
                else:
                    action_td["action"][..., 1] = 1
        return td.update(action_td)


class MultiKeyCountingEnv(EnvBase):
    """Counting env with multiple action/observation keys and nested structures."""

    def __init__(self, max_steps: int = 5, start_val: int = 0, **kwargs):
        super().__init__(**kwargs)

        self.max_steps = max_steps
        self.start_val = start_val
        self.nested_dim_1 = 3
        self.nested_dim_2 = 2

        count = torch.zeros((*self.batch_size, 1), device=self.device, dtype=torch.int)
        count_nested_1 = torch.zeros(
            (*self.batch_size, self.nested_dim_1, 1),
            device=self.device,
            dtype=torch.int,
        )
        count_nested_2 = torch.zeros(
            (*self.batch_size, self.nested_dim_2, 1),
            device=self.device,
            dtype=torch.int,
        )

        count[:] = self.start_val
        count_nested_1[:] = self.start_val
        count_nested_2[:] = self.start_val

        self.register_buffer("count", count)
        self.register_buffer("count_nested_1", count_nested_1)
        self.register_buffer("count_nested_2", count_nested_2)

        self.make_specs()

    def make_specs(self):
        self.observation_spec_unbatched = Composite(
            nested_1=Composite(
                observation=Bounded(low=0, high=200, shape=(self.nested_dim_1, 3)),
                shape=(self.nested_dim_1,),
            ),
            nested_2=Composite(
                observation=Unbounded(shape=(self.nested_dim_2, 2)),
                shape=(self.nested_dim_2,),
            ),
            observation=Unbounded(
                shape=(
                    10,
                    10,
                    3,
                )
            ),
        )

        self.action_spec_unbatched = Composite(
            nested_1=Composite(
                action=Categorical(n=2, shape=(self.nested_dim_1,)),
                shape=(self.nested_dim_1,),
            ),
            nested_2=Composite(
                azione=Bounded(low=0, high=100, shape=(self.nested_dim_2, 1)),
                shape=(self.nested_dim_2,),
            ),
            action=OneHot(n=2),
        )

        self.reward_spec_unbatched = Composite(
            nested_1=Composite(
                gift=Unbounded(shape=(self.nested_dim_1, 1)),
                shape=(self.nested_dim_1,),
            ),
            nested_2=Composite(
                reward=Unbounded(shape=(self.nested_dim_2, 1)),
                shape=(self.nested_dim_2,),
            ),
            reward=Unbounded(shape=(1,)),
        )

        self.done_spec_unbatched = Composite(
            nested_1=Composite(
                done=Categorical(
                    n=2,
                    shape=(self.nested_dim_1, 1),
                    dtype=torch.bool,
                ),
                terminated=Categorical(
                    n=2,
                    shape=(self.nested_dim_1, 1),
                    dtype=torch.bool,
                ),
                shape=(self.nested_dim_1,),
            ),
            nested_2=Composite(
                done=Categorical(
                    n=2,
                    shape=(self.nested_dim_2, 1),
                    dtype=torch.bool,
                ),
                terminated=Categorical(
                    n=2,
                    shape=(self.nested_dim_2, 1),
                    dtype=torch.bool,
                ),
                shape=(self.nested_dim_2,),
            ),
            # done at the root always prevail
            done=Categorical(
                n=2,
                shape=(1,),
                dtype=torch.bool,
            ),
            terminated=Categorical(
                n=2,
                shape=(1,),
                dtype=torch.bool,
            ),
        )

    def _reset(
        self,
        tensordict: TensorDictBase = None,
        **kwargs,
    ) -> TensorDictBase:
        reset_all = False
        if tensordict is not None:
            _reset = tensordict.get("_reset", None)
            if _reset is not None:
                self.count[_reset.squeeze(-1)] = self.start_val
                self.count_nested_1[_reset.squeeze(-1)] = self.start_val
                self.count_nested_2[_reset.squeeze(-1)] = self.start_val
            else:
                reset_all = True

        if tensordict is None or reset_all:
            self.count[:] = self.start_val
            self.count_nested_1[:] = self.start_val
            self.count_nested_2[:] = self.start_val

        reset_td = self.observation_spec.zero()
        reset_td["observation"] += expand_right(
            self.count, reset_td["observation"].shape
        )
        reset_td["nested_1", "observation"] += expand_right(
            self.count_nested_1, reset_td["nested_1", "observation"].shape
        )
        reset_td["nested_2", "observation"] += expand_right(
            self.count_nested_2, reset_td["nested_2", "observation"].shape
        )

        reset_td.update(self.output_spec["full_done_spec"].zero())

        assert reset_td.batch_size == self.batch_size

        return reset_td

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:

        # Each action has a corresponding reward, done, and observation
        reward = self.output_spec["full_reward_spec"].zero()
        done = self.output_spec["full_done_spec"].zero()
        td = self.observation_spec.zero()

        one_hot_action = tensordict["action"]
        one_hot_action = one_hot_action.long().argmax(-1).unsqueeze(-1)
        reward["reward"] += one_hot_action.to(torch.float)
        self.count += one_hot_action.to(torch.int)
        td["observation"] += expand_right(self.count, td["observation"].shape)
        done["done"] = self.count > self.max_steps
        done["terminated"] = self.count > self.max_steps

        discrete_action = tensordict["nested_1"]["action"].unsqueeze(-1)
        reward["nested_1"]["gift"] += discrete_action.to(torch.float)
        self.count_nested_1 += discrete_action.to(torch.int)
        td["nested_1", "observation"] += expand_right(
            self.count_nested_1, td["nested_1", "observation"].shape
        )
        done["nested_1", "done"] = self.count_nested_1 > self.max_steps
        done["nested_1", "terminated"] = self.count_nested_1 > self.max_steps

        continuous_action = tensordict["nested_2"]["azione"]
        reward["nested_2"]["reward"] += continuous_action.to(torch.float)
        self.count_nested_2 += continuous_action.to(torch.bool)
        td["nested_2", "observation"] += expand_right(
            self.count_nested_2, td["nested_2", "observation"].shape
        )
        done["nested_2", "done"] = self.count_nested_2 > self.max_steps
        done["nested_2", "terminated"] = self.count_nested_2 > self.max_steps

        td.update(done)
        td.update(reward)

        assert td.batch_size == self.batch_size
        return td

    def _set_seed(self, seed: int | None) -> None:
        torch.manual_seed(seed)


class EnvWithMetadata(EnvBase):
    """Environment that emits both tensor and non-tensor observations (for metadata tests)."""

    def __init__(self):
        super().__init__()
        self.observation_spec = Composite(
            tensor=Unbounded(3),
            non_tensor=NonTensor(shape=()),
        )
        self._saved_obs_spec = self.observation_spec.clone()
        self.state_spec = Composite(
            non_tensor=NonTensor(shape=()),
        )
        self._saved_state_spec = self.state_spec.clone()
        self.reward_spec = Unbounded(1)
        self._saved_full_reward_spec = self.full_reward_spec.clone()
        self.action_spec = Unbounded(1)
        self._saved_full_action_spec = self.full_action_spec.clone()

    def _reset(self, tensordict):
        data = self._saved_obs_spec.zero()
        data.set_non_tensor("non_tensor", 0)
        data.update(self.full_done_spec.zero())
        return data

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        data = self._saved_obs_spec.zero()
        data.set_non_tensor("non_tensor", tensordict["non_tensor"] + 1)
        data.update(self.full_done_spec.zero())
        data.update(self._saved_full_reward_spec.zero())
        return data

    def _set_seed(self, seed: int | None) -> None:
        ...


class AutoResettingCountingEnv(CountingEnv):
    """`CountingEnv` variant that auto-resets when done is reached."""

    def _step(self, tensordict):
        tensordict = super()._step(tensordict)
        if tensordict["done"].any():
            td_reset = super().reset()
            tensordict.update(td_reset.exclude(*self.done_keys))
        return tensordict

    def _reset(self, tensordict=None):
        if tensordict is not None and "_reset" in tensordict:
            raise RuntimeError
        return super()._reset(tensordict)


class AutoResetHeteroCountingEnv(HeterogeneousCountingEnv):
    """`HeterogeneousCountingEnv` variant that partially resets done sub-episodes."""

    def __init__(self, max_steps: int = 5, start_val: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.n_nested_dim = 3
        self.max_steps = max_steps
        self.start_val = start_val

        count = torch.zeros(
            (*self.batch_size, self.n_nested_dim, 1),
            device=self.device,
            dtype=torch.int,
        )
        count[:] = self.start_val

        self.register_buffer("count", count)
        self._make_specs()

    def _step(self, tensordict):
        for i in range(self.n_nested_dim):
            action = tensordict["lazy"][..., i]["action"]
            action = action[..., 0].to(torch.bool)
            self.count[..., i, 0] += action

        td = self.observation_spec.zero()
        for done_key in self.done_keys:
            td[done_key] = self.count > self.max_steps

        any_done = _terminated_or_truncated(
            td,
            full_done_spec=self.output_spec["full_done_spec"],
            key=None,
        )
        if any_done:
            self.count[td["lazy", "done"]] = 0

        for i in range(self.n_nested_dim):
            lazy = tensordict["lazy"][..., i]
            for obskey in self.observation_spec.keys(True, True):
                if isinstance(obskey, tuple) and obskey[0] == "lazy":
                    lazy[obskey[1:]] += expand_right(
                        self.count[..., i, 0], lazy[obskey[1:]].shape
                    ).clone()
        td.update(self.full_done_spec.zero())
        td.update(self.full_reward_spec.zero())

        assert td.batch_size == self.batch_size
        return td

    def _reset(self, tensordict=None):
        if tensordict is not None and self.reset_keys[0] in tensordict.keys(True):
            raise RuntimeError
        self.count[:] = self.start_val

        reset_td = self.observation_spec.zero()
        reset_td.update(self.full_done_spec.zero())
        assert reset_td.batch_size == self.batch_size
        return reset_td


class EnvWithDynamicSpec(EnvBase):
    """Environment with dynamic (ragged) observation specs that grow over time."""

    def __init__(self, max_count=5):
        super().__init__(batch_size=())
        self.observation_spec = Composite(
            observation=Unbounded(shape=(3, -1, 2)),
        )
        self.action_spec = Bounded(low=-1, high=1, shape=(2,))
        self.full_done_spec = Composite(
            done=Binary(1, shape=(1,), dtype=torch.bool),
            terminated=Binary(1, shape=(1,), dtype=torch.bool),
            truncated=Binary(1, shape=(1,), dtype=torch.bool),
        )
        self.reward_spec = Unbounded((1,), dtype=torch.float)
        self.count = 0
        self.max_count = max_count

    def _reset(self, tensordict=None):
        self.count = 0
        data = TensorDict(
            {
                "observation": torch.full(
                    (3, self.count + 1, 2),
                    self.count,
                    dtype=self.observation_spec["observation"].dtype,
                )
            }
        )
        data.update(self.done_spec.zero())
        return data

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        self.count += 1
        done = self.count >= self.max_count
        observation = TensorDict(
            {
                "observation": torch.full(
                    (3, self.count + 1, 2),
                    self.count,
                    dtype=self.observation_spec["observation"].dtype,
                )
            }
        )
        done = self.full_done_spec.zero() | done
        reward = self.full_reward_spec.zero()
        return observation.update(done).update(reward)

    def _set_seed(self, seed: int | None) -> None:
        self.manual_seed = seed


class EnvWithScalarAction(EnvBase):
    """Environment exposing a scalar (or singleton) action spec for edge-case testing."""

    def __init__(self, singleton: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.singleton = singleton
        self.action_spec = Bounded(
            -1,
            1,
            shape=(
                *self.batch_size,
                1,
            )
            if self.singleton
            else self.batch_size,
        )
        self.observation_spec = Composite(
            observation=Unbounded(
                shape=(
                    *self.batch_size,
                    3,
                )
            ),
            shape=self.batch_size,
        )
        self.done_spec = Composite(
            done=Unbounded(self.batch_size + (1,), dtype=torch.bool),
            terminated=Unbounded(self.batch_size + (1,), dtype=torch.bool),
            truncated=Unbounded(self.batch_size + (1,), dtype=torch.bool),
            shape=self.batch_size,
        )
        self.reward_spec = Unbounded(
            shape=(
                *self.batch_size,
                1,
            )
        )

    def _reset(self, td: TensorDict):
        return TensorDict(
            observation=torch.randn(*self.batch_size, 3, device=self.device),
            done=torch.zeros(*self.batch_size, 1, dtype=torch.bool, device=self.device),
            truncated=torch.zeros(
                *self.batch_size, 1, dtype=torch.bool, device=self.device
            ),
            terminated=torch.zeros(
                *self.batch_size, 1, dtype=torch.bool, device=self.device
            ),
            device=self.device,
        )

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        return TensorDict(
            observation=torch.randn(*self.batch_size, 3, device=self.device),
            reward=torch.zeros(1, device=self.device),
            done=torch.zeros(*self.batch_size, 1, dtype=torch.bool, device=self.device),
            truncated=torch.zeros(
                *self.batch_size, 1, dtype=torch.bool, device=self.device
            ),
            terminated=torch.zeros(
                *self.batch_size, 1, dtype=torch.bool, device=self.device
            ),
        )

    def _set_seed(self, seed: int | None) -> None:
        ...


class EnvThatDoesNothing(EnvBase):
    """Environment whose reset/step return empty tensordicts (for plumbing tests)."""

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        return TensorDict(batch_size=self.batch_size, device=self.device)

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        return TensorDict(batch_size=self.batch_size, device=self.device)

    def _set_seed(self, seed: int | None) -> None:
        ...


class Str2StrEnv(EnvBase):
    """String-to-string environment with non-tensor observation/action fields."""

    def __init__(self, min_size=4, max_size=10, **kwargs):
        self.observation_spec = Composite(
            observation=NonTensor(example_data="an observation!", shape=())
        )
        self.full_action_spec = Composite(
            action=NonTensor(example_data="an action!", shape=())
        )
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float)
        self.min_size = min_size
        self.max_size = max_size
        super().__init__(**kwargs)

    def _step(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        assert isinstance(tensordict["action"], str)
        out = tensordict.empty()
        out.set("observation", self.get_random_string())
        out.set("done", torch.zeros(1, dtype=torch.bool).bernoulli_(0.01))
        out.set("reward", torch.zeros(1, dtype=torch.float).bernoulli_(0.01))
        return out

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        out = tensordict.empty() if tensordict is not None else TensorDict()
        out.set("observation", self.get_random_string())
        out.set("done", torch.zeros(1, dtype=torch.bool).bernoulli_(0.01))
        return out

    def get_random_string(self):
        return get_random_string(self.min_size, self.max_size)

    def _set_seed(self, seed: int | None) -> None:
        random.seed(seed)
        torch.manual_seed(0)


class EnvThatErrorsAfter10Iters(EnvBase):
    """Environment that raises after 10 steps (used to validate error propagation)."""

    def __init__(self):
        self.action_spec = Composite(action=Unbounded((1,)))
        self.reward_spec = Composite(reward=Unbounded((1,)))
        self.done_spec = Composite(done=Unbounded((1,)))
        self.observation_spec = Composite(observation=Unbounded((1,)))
        self.counter = 0
        super().__init__()

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDict:
        return self.full_observation_spec.zero().update(self.full_done_spec.zero())

    def _step(self, tensordict: TensorDictBase, **kwargs) -> TensorDict:
        if self.counter >= 10:
            raise RuntimeError("max steps!")
        self.counter += 1
        return (
            self.full_observation_spec.zero()
            .update(self.full_done_spec.zero())
            .update(self.full_reward_spec.zero())
        )

    def _set_seed(self, seed: int | None) -> None:
        ...


@tensorclass()
class TC:
    """Simple tensorclass used by `EnvWithTensorClass`."""

    field0: str
    field1: torch.Tensor


class EnvWithTensorClass(CountingEnv):
    """`CountingEnv` variant that carries a tensorclass observation."""

    tc_cls = TC

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spec["tc"] = Composite(
            field0=NonTensor(example_data="an observation!", shape=self.batch_size),
            field1=Unbounded(shape=self.batch_size),
            shape=self.batch_size,
            data_cls=TC,
        )

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        td = super()._reset(tensordict, **kwargs)
        td["tc"] = TC("0", torch.zeros(self.batch_size))
        return td

    def _step(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        td = super()._step(tensordict, **kwargs)
        default = TC("0", 0)
        f0 = tensordict.get("tc", default).field0
        if f0 is None:
            f0 = "0"
        f1 = tensordict.get("tc", default).field1
        if f1 is None:
            f1 = torch.zeros(self.batch_size)
        td["tc"] = TC(
            str(int(f0) + 1),
            f1 + 1,
        )
        return td


@tensorclass
class History:
    """Simple history record (role/content) used by `HistoryTransform`."""

    role: str
    content: str


class HistoryTransform(Transform):
    """A mocking class to record history."""

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        defaults = {
            "role": NonTensor(
                example_data="a role!",
                shape=(-1,),
            ),
            "content": NonTensor(
                example_data="a content!",
                shape=(-1,),
            ),
        }
        observation_spec["history"] = Composite(
            defaults,
            shape=(-1,),
            data_cls=History,
        )
        assert observation_spec.device == self.parent.device
        assert observation_spec["history"].device == self.parent.device
        return observation_spec

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        assert tensordict_reset.device == self.parent.device
        tensordict_reset["history"] = torch.stack(
            [
                History(role="system", content="0"),
                History(role="user", content="1"),
            ]
        )
        assert tensordict_reset["history"].device == self.parent.device
        return tensordict_reset

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        assert next_tensordict.device == self.parent.device
        history = tensordict["history"]
        local_history = History(
            role=np.random.choice(["user", "system", "assistant"]),
            content=str(int(history.content[-1]) + 1),
            device=history.device,
        )
        # history = tensordict["history"].append(local_history)
        try:
            history = torch.stack(list(history.unbind(0)) + [local_history])
        except Exception:
            raise
        assert isinstance(history, History)
        next_tensordict["history"] = history
        assert next_tensordict["history"].device == self.parent.device, (
            next_tensordict["history"],
            self.parent.device,
        )
        return next_tensordict


class DummyStrDataLoader:
    """Minimal iterator that yields random strings (for LLM tests)."""

    def __init__(self, batch_size=0):
        if isinstance(batch_size, tuple):
            batch_size = torch.Size(batch_size).numel()
        self.batch_size = batch_size

    def generate_random_string(self, length=10):
        """Generate a random string of a given length."""
        return "".join(random.choice(string.ascii_lowercase) for _ in range(length))

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_size == 0:
            return {"text": self.generate_random_string()}
        else:
            return {
                "text": [self.generate_random_string() for _ in range(self.batch_size)]
            }


class DummyTensorDataLoader:
    """Minimal iterator that yields random token tensors (for LLM tests)."""

    def __init__(self, batch_size=0, max_length=10, padding=False):
        if isinstance(batch_size, tuple):
            batch_size = torch.Size(batch_size).numel()
        self.batch_size = batch_size
        self.max_length = max_length
        self.padding = padding

    def generate_random_tensor(self):
        """Generate a tensor of random int64 values."""
        length = random.randint(1, self.max_length)
        rt = torch.randint(1, 10000, (length,))
        return rt

    def pad_tensor(self, tensor):
        """Pad a tensor to the maximum length."""
        padding_length = self.max_length - len(tensor)
        return torch.cat((torch.zeros(padding_length, dtype=torch.int64), tensor))

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_size == 0:
            tensor = self.generate_random_tensor()
            tokens = self.pad_tensor(tensor) if self.padding else tensor
        else:
            tensors = [self.generate_random_tensor() for _ in range(self.batch_size)]
            if self.padding:
                tensors = [self.pad_tensor(tensor) for tensor in tensors]
                tokens = torch.stack(tensors)
            else:
                tokens = tensors
        return {"tokens": tokens, "attention_mask": tokens != 0}


class MockNestedResetEnv(EnvBase):
    """To test behaviour of envs with nested done states - where the root done prevails over others."""

    def __init__(self, num_steps: int, done_at_root: bool) -> None:
        super().__init__(device="cpu")
        self._num_steps = num_steps
        self._counter = 0
        self.done_at_root = done_at_root
        self.done_spec = Composite(
            {
                ("agent_1", "done"): Binary(1, dtype=torch.bool),
                ("agent_2", "done"): Binary(1, dtype=torch.bool),
            }
        )
        if done_at_root:
            self.full_done_spec["done"] = Binary(1, dtype=torch.bool)

    def _reset(self, tensordict: TensorDict) -> TensorDict:
        torchrl_logger.info(f"Reset after {self._counter} steps!")
        if tensordict is not None:
            torchrl_logger.info(f"tensordict at reset {tensordict.to_dict()}")
        self._counter = 0
        result = TensorDict(
            {
                ("agent_1", "done"): torch.tensor([False], dtype=torch.bool),
                ("agent_2", "done"): torch.tensor([False], dtype=torch.bool),
            },
        )
        if self.done_at_root:
            result["done"] = torch.tensor([False], dtype=torch.bool)
        return result

    def _step(self, tensordict: TensorDict) -> TensorDict:
        self._counter += 1
        done = torch.tensor([self._counter >= self._num_steps], dtype=torch.bool)
        if self.done_at_root:
            return TensorDict(
                {
                    "done": done,
                    ("agent_1", "done"): torch.tensor([True], dtype=torch.bool),
                    ("agent_2", "done"): torch.tensor([False], dtype=torch.bool),
                },
            )
        else:
            return TensorDict(
                {
                    ("agent_1", "done"): done,
                    ("agent_2", "done"): torch.tensor([False], dtype=torch.bool),
                },
            )

    def _set_seed(self):
        pass


class EnvThatErrorsBecauseOfStack(EnvBase):
    """Environment crafted to trigger stacking errors with certain batch shapes."""

    def __init__(self, target: int = 5, batch_size: int | None = None):
        super().__init__(device="cpu", batch_size=batch_size)
        self.target = target
        self.observation_spec = Bounded(
            low=0, high=self.target, shape=(1,), dtype=torch.int64
        )
        self.action_spec = Categorical(n=2, shape=(1,), dtype=torch.int64)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.done_spec = Categorical(n=2, shape=(1,), dtype=torch.bool)

    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        if tensordict is None:
            tensordict = TensorDict(batch_size=self.batch_size, device=self.device)

        observation = torch.zeros(
            self.batch_size, dtype=self.observation_spec.dtype, device=self.device
        )
        reward = torch.zeros(
            self.batch_size + torch.Size([1]),
            dtype=self.reward_spec.dtype,
            device=self.device,
        )
        done = torch.zeros(
            self.batch_size + torch.Size([1]), dtype=torch.bool, device=self.device
        )
        terminated = torch.zeros_like(done)
        action = torch.zeros(
            self.batch_size + torch.Size([1]), dtype=torch.int64, device=self.device
        )

        tensordict.set(self.observation_keys[0], observation)
        tensordict.set(self.reward_key, reward)
        tensordict.set(self.done_keys[0], done)
        tensordict.set("terminated", terminated)
        tensordict.set(self.action_keys[0], action)

        return tensordict

    def _step(self, tensordict: TensorDict) -> TensorDict:
        obs = tensordict.get(
            self.observation_keys[0]
        )  # the counter value or the counters value if it is several batchs
        action = tensordict.get(self.action_keys[0]).squeeze(-1)

        new_obs = obs + (action == 1).to(obs.dtype)
        new_obs = new_obs.clamp_max(self.target)
        reward = (new_obs == self.target).to(self.reward_spec.dtype).unsqueeze(-1)
        done = (new_obs == self.target).to(torch.bool).unsqueeze(-1)
        terminated = done.clone()
        return TensorDict(
            {
                self.observation_keys[0]: new_obs,
                self.reward_keys[0]: reward,
                self.done_keys[0]: done,
                "terminated": terminated,
                self.action_keys[0]: action.unsqueeze(-1),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _set_seed(self, seed: int | None) -> None:
        return 0


class FastImageEnv(EnvBase):
    """Fast environment with image observations for benchmarking.

    This environment uses pre-allocated tensors to minimize step overhead,
    making it ideal for benchmarking collector and storage performance.

    Args:
        img_shape: Shape of the image observation (C, H, W). Default: (4, 84, 84).
        device: Device to create tensors on. Default: None (CPU).
        batch_size: Batch size of the environment. Default: ().
    """

    def __init__(self, img_shape=(4, 84, 84), device=None, batch_size=()):
        super().__init__(device=device, batch_size=batch_size)
        self.img_shape = img_shape
        self._make_spec()
        # Pre-allocate to minimize step overhead
        self._obs = torch.rand(*batch_size, *img_shape, device=device)

    def _make_spec(self):
        self.observation_spec = Composite(
            pixels=Unbounded(
                shape=(*self.batch_size, *self.img_shape), dtype=torch.float32
            ),
            shape=self.batch_size,
        )
        self.action_spec = Bounded(
            -1, 1, shape=(*self.batch_size, 4), dtype=torch.float32
        )
        self.reward_spec = Unbounded(shape=(*self.batch_size, 1), dtype=torch.float32)
        self.done_spec = Unbounded(shape=(*self.batch_size, 1), dtype=torch.bool)

    def _reset(self, tensordict):
        return TensorDict(
            {
                "pixels": self._obs.clone(),
                "done": torch.zeros(
                    *self.batch_size, 1, dtype=torch.bool, device=self.device
                ),
                "terminated": torch.zeros(
                    *self.batch_size, 1, dtype=torch.bool, device=self.device
                ),
            },
            batch_size=self.batch_size,
        )

    def _step(self, tensordict):
        return TensorDict(
            {
                "pixels": self._obs.clone(),
                "reward": torch.ones(
                    *self.batch_size, 1, dtype=torch.float32, device=self.device
                ),
                "done": torch.zeros(
                    *self.batch_size, 1, dtype=torch.bool, device=self.device
                ),
                "terminated": torch.zeros(
                    *self.batch_size, 1, dtype=torch.bool, device=self.device
                ),
            },
            batch_size=self.batch_size,
        )

    def _set_seed(self, seed):
        torch.manual_seed(seed)
