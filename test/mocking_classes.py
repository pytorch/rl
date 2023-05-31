# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch
import torch.nn as nn
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import (
    BinaryDiscreteTensorSpec,
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    MultiOneHotDiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs.common import EnvBase
from torchrl.envs.model_based.common import ModelBasedEnvBase

spec_dict = {
    "bounded": BoundedTensorSpec,
    "one_hot": OneHotDiscreteTensorSpec,
    "categorical": DiscreteTensorSpec,
    "unbounded": UnboundedContinuousTensorSpec,
    "binary": BinaryDiscreteTensorSpec,
    "mult_one_hot": MultiOneHotDiscreteTensorSpec,
    "composite": CompositeSpec,
}

default_spec_kwargs = {
    OneHotDiscreteTensorSpec: {"n": 7},
    DiscreteTensorSpec: {"n": 7},
    BoundedTensorSpec: {"minimum": -torch.ones(4), "maximum": torch.ones(4)},
    UnboundedContinuousTensorSpec: {
        "shape": [
            7,
        ]
    },
    BinaryDiscreteTensorSpec: {"n": 7},
    MultiOneHotDiscreteTensorSpec: {"nvec": [7, 3, 5]},
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
        for key, item in list(cls._output_spec["_observation_spec"].items()):
            cls._output_spec["_observation_spec"][key] = item.to(
                torch.get_default_dtype()
            )
        cls._output_spec["_reward_spec"] = cls._output_spec["_reward_spec"].to(
            torch.get_default_dtype()
        )
        if not isinstance(cls._output_spec["_reward_spec"], CompositeSpec):
            cls._output_spec["_reward_spec"] = CompositeSpec(
                reward=cls._output_spec["_reward_spec"],
                shape=cls._output_spec["_reward_spec"].shape[:-1],
            )
        if not isinstance(cls._output_spec["_done_spec"], CompositeSpec):
            cls._output_spec["_done_spec"] = CompositeSpec(
                done=cls._output_spec["_done_spec"],
                shape=cls._output_spec["_done_spec"].shape[:-1],
            )
        if not isinstance(cls._input_spec["_action_spec"], CompositeSpec):
            cls._input_spec["_action_spec"] = CompositeSpec(
                action=cls._input_spec["_action_spec"],
                shape=cls._input_spec["_action_spec"].shape[:-1],
            )
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        *args,
        seed: int = 100,
        **kwargs,
    ):
        super().__init__(
            device="cpu",
            dtype=torch.get_default_dtype(),
        )
        self.set_seed(seed)
        self.is_closed = False

    @property
    def maxstep(self):
        return 100

    def _set_seed(self, seed: Optional[int]):
        self.seed = seed
        self.counter = seed % 17  # make counter a small number

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
    """A simple counting env that is reset after a predifined max number of steps."""

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
            action_spec = UnboundedContinuousTensorSpec(
                (
                    *batch_size,
                    1,
                )
            )
        if observation_spec is None:
            observation_spec = CompositeSpec(
                observation=UnboundedContinuousTensorSpec(
                    (
                        *batch_size,
                        1,
                    )
                ),
                shape=batch_size,
            )
        if reward_spec is None:
            reward_spec = UnboundedContinuousTensorSpec(
                (
                    *batch_size,
                    1,
                )
            )
        if done_spec is None:
            done_spec = DiscreteTensorSpec(2, dtype=torch.bool, shape=(*batch_size, 1))
        if state_spec is None:
            state_spec = CompositeSpec(shape=batch_size)
        input_spec = CompositeSpec(
            _action_spec=action_spec, _state_spec=state_spec, shape=batch_size
        )
        cls._output_spec = CompositeSpec(shape=batch_size)
        cls._output_spec["_reward_spec"] = reward_spec
        cls._output_spec["_done_spec"] = done_spec
        cls._output_spec["_observation_spec"] = observation_spec
        cls._input_spec = input_spec

        if not isinstance(cls._output_spec["_reward_spec"], CompositeSpec):
            cls._output_spec["_reward_spec"] = CompositeSpec(
                reward=cls._output_spec["_reward_spec"], shape=batch_size
            )
        if not isinstance(cls._output_spec["_done_spec"], CompositeSpec):
            cls._output_spec["_done_spec"] = CompositeSpec(
                done=cls._output_spec["_done_spec"], shape=batch_size
            )
        if not isinstance(cls._input_spec["_action_spec"], CompositeSpec):
            cls._input_spec["_action_spec"] = CompositeSpec(
                action=cls._input_spec["_action_spec"], shape=batch_size
            )
        return super().__new__(*args, **kwargs)

    def __init__(self, device="cpu"):
        super(MockSerialEnv, self).__init__(device=device)
        self.is_closed = False

    def _set_seed(self, seed: Optional[int]):
        assert seed >= 1
        self.seed = seed
        self.counter = seed % 17  # make counter a small number
        self.max_val = max(self.counter + 100, self.counter * 2)

    def _step(self, tensordict):
        self.counter += 1
        n = torch.tensor(
            [self.counter], device=self.device, dtype=torch.get_default_dtype()
        )
        done = self.counter >= self.max_val
        done = torch.tensor([done], dtype=torch.bool, device=self.device)
        return TensorDict(
            {
                "next": TensorDict(
                    {"reward": n, "done": done, "observation": n.clone()}, batch_size=[]
                )
            },
            batch_size=[],
        )

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        self.max_val = max(self.counter + 100, self.counter * 2)

        n = torch.tensor(
            [self.counter], device=self.device, dtype=torch.get_default_dtype()
        )
        done = self.counter >= self.max_val
        done = torch.tensor([done], dtype=torch.bool, device=self.device)
        return TensorDict({"done": done, "observation": n}, [])

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
        state_spec=None,
        reward_spec=None,
        done_spec=None,
        **kwargs,
    ):
        batch_size = kwargs.setdefault("batch_size", torch.Size([]))
        if action_spec is None:
            action_spec = UnboundedContinuousTensorSpec(
                (
                    *batch_size,
                    1,
                )
            )
        if state_spec is None:
            state_spec = CompositeSpec(
                observation=UnboundedContinuousTensorSpec(
                    (
                        *batch_size,
                        1,
                    )
                ),
                shape=batch_size,
            )
        if observation_spec is None:
            observation_spec = CompositeSpec(
                observation=UnboundedContinuousTensorSpec(
                    (
                        *batch_size,
                        1,
                    )
                ),
                shape=batch_size,
            )
        if reward_spec is None:
            reward_spec = UnboundedContinuousTensorSpec(
                (
                    *batch_size,
                    1,
                )
            )
        if done_spec is None:
            done_spec = DiscreteTensorSpec(2, dtype=torch.bool, shape=(*batch_size, 1))
        cls._output_spec = CompositeSpec(shape=batch_size)
        cls._output_spec["_reward_spec"] = reward_spec
        cls._output_spec["_done_spec"] = done_spec
        cls._output_spec["_observation_spec"] = observation_spec
        cls._input_spec = CompositeSpec(
            _action_spec=action_spec,
            _state_spec=state_spec,
            shape=batch_size,
        )
        if not isinstance(cls._output_spec["_reward_spec"], CompositeSpec):
            cls._output_spec["_reward_spec"] = CompositeSpec(
                reward=cls._output_spec["_reward_spec"], shape=batch_size
            )
        if not isinstance(cls._output_spec["_done_spec"], CompositeSpec):
            cls._output_spec["_done_spec"] = CompositeSpec(
                done=cls._output_spec["_done_spec"], shape=batch_size
            )
        if not isinstance(cls._input_spec["_action_spec"], CompositeSpec):
            cls._input_spec["_action_spec"] = CompositeSpec(
                action=cls._input_spec["_action_spec"], shape=batch_size
            )
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, device="cpu", batch_size=None):
        super(MockBatchedLockedEnv, self).__init__(device=device, batch_size=batch_size)
        self.counter = 0

    rand_step = MockSerialEnv.rand_step

    def _set_seed(self, seed: Optional[int]):
        assert seed >= 1
        self.seed = seed
        self.counter = seed % 17  # make counter a small number
        self.max_val = max(self.counter + 100, self.counter * 2)

    def _step(self, tensordict):
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
        n = (
            torch.full(
                [*leading_batch_size, *self.observation_spec["observation"].shape],
                self.counter,
            )
            .to(self.device)
            .to(torch.get_default_dtype())
        )
        done = self.counter >= self.max_val
        done = torch.full(
            (*leading_batch_size, *self.batch_size, 1),
            done,
            dtype=torch.bool,
            device=self.device,
        )
        return TensorDict(
            {
                "next": TensorDict(
                    {"reward": n, "done": done, "observation": n},
                    tensordict.batch_size,
                    device=self.device,
                )
            },
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

        n = (
            torch.full(
                [*leading_batch_size, *self.observation_spec["observation"].shape],
                self.counter,
            )
            .to(self.device)
            .to(torch.get_default_dtype())
        )
        done = self.counter >= self.max_val
        done = torch.full(
            (*leading_batch_size, *batch_size, 1),
            done,
            dtype=torch.bool,
            device=self.device,
        )
        return TensorDict(
            {"reward": n, "done": done, "observation": n},
            [
                *leading_batch_size,
                *batch_size,
            ],
            device=self.device,
        )


class MockBatchedUnLockedEnv(MockBatchedLockedEnv):
    """Mocks an env whose batch_size does not define the size of the output tensordict.

    The size of the output tensordict is defined by the input tensordict itself.

    """

    def __init__(self, device="cpu", batch_size=None):
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
            observation_spec = CompositeSpec(
                observation=UnboundedContinuousTensorSpec(
                    shape=torch.Size([*batch_size, size])
                ),
                observation_orig=UnboundedContinuousTensorSpec(
                    shape=torch.Size([*batch_size, size])
                ),
                shape=batch_size,
            )
        if action_spec is None:
            action_spec_cls = (
                DiscreteTensorSpec
                if categorical_action_encoding
                else OneHotDiscreteTensorSpec
            )
            action_spec = action_spec_cls(n=7, shape=(*batch_size, 7))
        if reward_spec is None:
            reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        if done_spec is None:
            done_spec = DiscreteTensorSpec(2, dtype=torch.bool, shape=(*batch_size, 1))

        if state_spec is None:
            cls._out_key = "observation_orig"
            state_spec = CompositeSpec(
                {
                    cls._out_key: observation_spec["observation"],
                },
                shape=batch_size,
            )
        cls._output_spec = CompositeSpec(shape=batch_size)
        cls._output_spec["_reward_spec"] = reward_spec
        cls._output_spec["_done_spec"] = done_spec
        cls._output_spec["_observation_spec"] = observation_spec
        cls._input_spec = CompositeSpec(
            _action_spec=action_spec,
            _state_spec=state_spec,
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
            tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict = tensordict.select().set(self.out_key, self._get_out_obs(state))
        tensordict = tensordict.set(self._out_key, self._get_out_obs(state))
        tensordict.set("done", torch.zeros(*tensordict.shape, 1, dtype=torch.bool))
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
        tensordict = tensordict.select()  # empty tensordict

        tensordict.set(self.out_key, self._get_out_obs(obs))
        tensordict.set(self._out_key, self._get_out_obs(obs))

        done = torch.isclose(obs, torch.ones_like(obs) * (self.counter + 1))
        reward = done.any(-1).unsqueeze(-1)
        # set done to False
        done = torch.zeros_like(done).all(-1).unsqueeze(-1)
        tensordict.set("reward", reward.to(torch.get_default_dtype()))
        tensordict.set("done", done)
        return tensordict.select().set("next", tensordict)


class ContinuousActionVecMockEnv(_MockEnv):
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
            observation_spec = CompositeSpec(
                observation=UnboundedContinuousTensorSpec(
                    shape=torch.Size([*batch_size, size])
                ),
                observation_orig=UnboundedContinuousTensorSpec(
                    shape=torch.Size([*batch_size, size])
                ),
                shape=batch_size,
            )
        if action_spec is None:
            action_spec = BoundedTensorSpec(
                -1,
                1,
                (
                    *batch_size,
                    7,
                ),
            )
        if reward_spec is None:
            reward_spec = UnboundedContinuousTensorSpec(shape=(*batch_size, 1))
        if done_spec is None:
            done_spec = DiscreteTensorSpec(2, dtype=torch.bool, shape=(*batch_size, 1))

        if state_spec is None:
            cls._out_key = "observation_orig"
            state_spec = CompositeSpec(
                {
                    cls._out_key: observation_spec["observation"],
                },
                shape=batch_size,
            )
        cls._output_spec = CompositeSpec(shape=batch_size)
        cls._output_spec["_reward_spec"] = reward_spec
        cls._output_spec["_done_spec"] = done_spec
        cls._output_spec["_observation_spec"] = observation_spec
        cls._input_spec = CompositeSpec(
            _action_spec=action_spec,
            _state_spec=state_spec,
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
            tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict = tensordict.select()
        tensordict.update(self.observation_spec.rand())
        # tensordict.set("next_" + self.out_key, self._get_out_obs(state))
        # tensordict.set("next_" + self._out_key, self._get_out_obs(state))
        tensordict.set("done", torch.zeros(*tensordict.shape, 1, dtype=torch.bool))
        return tensordict

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        self.step_count += 1
        tensordict = tensordict.to(self.device)
        a = tensordict.get("action")

        obs = self._obs_step(self._get_in_obs(tensordict.get(self._out_key)), a)
        tensordict = tensordict.select()  # empty tensordict

        tensordict.set(self.out_key, self._get_out_obs(obs))
        tensordict.set(self._out_key, self._get_out_obs(obs))

        done = torch.isclose(obs, torch.ones_like(obs) * (self.counter + 1))
        while done.shape != tensordict.shape:
            done = done.any(-1)
        done = reward = done.unsqueeze(-1)
        tensordict.set("reward", reward.to(torch.get_default_dtype()))
        tensordict.set("done", done)
        return tensordict.select().set("next", tensordict)

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
        state_spec=None,
        reward_spec=None,
        done_spec=None,
        from_pixels=True,
        **kwargs,
    ):
        batch_size = kwargs.setdefault("batch_size", torch.Size([]))
        if observation_spec is None:
            cls.out_key = "pixels"
            observation_spec = CompositeSpec(
                pixels=UnboundedContinuousTensorSpec(
                    shape=torch.Size([*batch_size, 1, 7, 7])
                ),
                pixels_orig=UnboundedContinuousTensorSpec(
                    shape=torch.Size([*batch_size, 1, 7, 7])
                ),
                shape=batch_size,
            )
        if action_spec is None:
            action_spec = OneHotDiscreteTensorSpec(7, shape=(*batch_size, 7))
        if reward_spec is None:
            reward_spec = UnboundedContinuousTensorSpec(shape=(*batch_size, 1))
        if done_spec is None:
            done_spec = DiscreteTensorSpec(2, dtype=torch.bool, shape=(*batch_size, 1))

        if state_spec is None:
            cls._out_key = "pixels_orig"
            state_spec = CompositeSpec(
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
            done_spec=done_spec,
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
            observation_spec = CompositeSpec(
                pixels=UnboundedContinuousTensorSpec(
                    shape=torch.Size([*batch_size, 7, 7, 3])
                ),
                pixels_orig=UnboundedContinuousTensorSpec(
                    shape=torch.Size([*batch_size, 7, 7, 3])
                ),
                shape=batch_size,
            )
        if action_spec is None:
            action_spec_cls = (
                DiscreteTensorSpec
                if categorical_action_encoding
                else OneHotDiscreteTensorSpec
            )
            action_spec = action_spec_cls(7, shape=(*batch_size, 7))
        if state_spec is None:
            cls._out_key = "pixels_orig"
            state_spec = CompositeSpec(
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
            observation_spec = CompositeSpec(
                pixels=UnboundedContinuousTensorSpec(
                    shape=torch.Size([*batch_size, *pixel_shape])
                ),
                pixels_orig=UnboundedContinuousTensorSpec(
                    shape=torch.Size([*batch_size, *pixel_shape])
                ),
                shape=batch_size,
            )

        if action_spec is None:
            action_spec = BoundedTensorSpec(-1, 1, [*batch_size, pixel_shape[-1]])
        if reward_spec is None:
            reward_spec = UnboundedContinuousTensorSpec(shape=(*batch_size, 1))
        if done_spec is None:
            done_spec = DiscreteTensorSpec(2, dtype=torch.bool, shape=(*batch_size, 1))
        if state_spec is None:
            cls._out_key = "pixels_orig"
            state_spec = CompositeSpec(
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
        # if any(dim == 1 for dim in obs.shape):
        #     print("squeezing obs", obs.shape)
        #     obs = obs.squeeze()
        return obs


class ContinuousActionConvMockEnvNumpy(ContinuousActionConvMockEnv):
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
            observation_spec = CompositeSpec(
                pixels=UnboundedContinuousTensorSpec(
                    shape=torch.Size([*batch_size, 7, 7, 3])
                ),
                pixels_orig=UnboundedContinuousTensorSpec(
                    shape=torch.Size([*batch_size, 7, 7, 3])
                ),
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
            hidden_observation=UnboundedContinuousTensorSpec(
                (
                    *self.batch_size,
                    4,
                )
            ),
            shape=self.batch_size,
        )
        self.state_spec = CompositeSpec(
            hidden_observation=UnboundedContinuousTensorSpec(
                (
                    *self.batch_size,
                    4,
                )
            ),
            shape=self.batch_size,
        )
        self.action_spec = UnboundedContinuousTensorSpec(
            (
                *self.batch_size,
                1,
            )
        )
        self.reward_spec = UnboundedContinuousTensorSpec(
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
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, observation, action):
        return self.linear(torch.cat([observation, action], dim=-1))


class CountingEnv(EnvBase):
    """An env that is done after a given number of steps.

    The action is the count increment.

    """

    def __init__(self, max_steps: int = 5, start_val: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.max_steps = max_steps
        self.start_val = start_val

        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
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
        self.reward_spec = UnboundedContinuousTensorSpec(
            (
                *self.batch_size,
                1,
            ),
            device=self.device,
        )
        self.done_spec = DiscreteTensorSpec(
            2,
            dtype=torch.bool,
            shape=(
                *self.batch_size,
                1,
            ),
            device=self.device,
        )
        self.action_spec = BinaryDiscreteTensorSpec(
            n=1, shape=[*self.batch_size, 1], device=self.device
        )
        self.register_buffer(
            "count",
            torch.zeros((*self.batch_size, 1), device=self.device, dtype=torch.int),
        )

    def _set_seed(self, seed: Optional[int]):
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
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        action = tensordict.get("action")
        self.count += action.to(torch.int).to(self.device)
        tensordict = TensorDict(
            source={
                "observation": self.count.clone(),
                "done": self.count > self.max_steps,
                "reward": torch.zeros_like(self.count, dtype=torch.float),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict.select().set("next", tensordict)


class NestedRewardEnv(CountingEnv):
    # an env with nested reward and done states
    def __init__(self, max_steps: int = 5, start_val: int = 0, **kwargs):
        super().__init__(max_steps=max_steps, start_val=start_val, **kwargs)
        self.observation_spec = CompositeSpec(
            {("data", "states"): self.observation_spec["observation"].clone()},
            shape=self.batch_size,
        )
        self.reward_spec = CompositeSpec(
            {("data", "reward"): self.reward_spec.clone()}, shape=self.batch_size
        )
        self.done_spec = CompositeSpec(
            {("data", "done"): self.done_spec.clone()}, shape=self.batch_size
        )

    def _reset(self, td):
        td = super()._reset(td)
        td[self.done_key] = td["done"]
        del td["done"]
        td["data", "states"] = td["observation"]
        del td["observation"]
        return td

    def _step(self, td):
        td_root = super()._step(td)
        td = td_root["next"]
        td[self.reward_key] = td["reward"]
        del td["reward"]
        td[self.done_key] = td["done"]
        del td["done"]
        td["data", "states"] = td["observation"]
        del td["observation"]
        return td_root


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
        if not max_steps.shape == self.batch_size:
            raise RuntimeError("batch_size and max_steps shape must match.")

        self.max_steps = max_steps

        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                (
                    *self.batch_size,
                    1,
                ),
                dtype=torch.int32,
            ),
            shape=self.batch_size,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(
            (
                *self.batch_size,
                1,
            )
        )
        self.done_spec = DiscreteTensorSpec(
            2,
            dtype=torch.bool,
            shape=(
                *self.batch_size,
                1,
            ),
        )
        self.action_spec = BinaryDiscreteTensorSpec(n=1, shape=[*self.batch_size, 1])

        self.count = torch.zeros(
            (*self.batch_size, 1), device=self.device, dtype=torch.int
        )
        if start_val.numel() == self.batch_size.numel():
            self.start_val = start_val.view(*self.batch_size, 1)
        elif start_val.numel() <= 1:
            self.start_val = start_val.expand_as(self.count)

    def _set_seed(self, seed: Optional[int]):
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
                "reward": torch.zeros_like(self.count, dtype=torch.float),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict.select().set("next", tensordict)
