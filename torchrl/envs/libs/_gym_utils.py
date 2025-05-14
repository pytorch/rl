# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util

import torch
from tensordict.utils import unravel_key

from torch.utils._pytree import tree_map

from torchrl._utils import implement_for
from torchrl.data import Composite
from torchrl.envs import step_mdp, TransformedEnv
from torchrl.envs.libs.gym import _torchrl_to_gym_spec_transform, GYMNASIUM_1_ERROR

_has_gym = importlib.util.find_spec("gym", None) is not None
_has_gymnasium = importlib.util.find_spec("gymnasium", None) is not None


class _BaseGymWrapper:
    def __init__(
        self, *, entry_point, to_numpy=False, transform=None, info_keys=None, **kwargs
    ):
        super().__init__()
        torchrl_env = entry_point(**kwargs)
        if transform is not None:
            torchrl_env = TransformedEnv(torchrl_env, transform)
        self.torchrl_env = torchrl_env
        self.info_keys = info_keys
        self.action_space = _torchrl_to_gym_spec_transform(
            self.torchrl_env.action_spec,
            categorical_action_encoding=self.torchrl_env.__dict__.get(
                "categorical_action_encoding", True
            ),
        )
        self.observation_space = _torchrl_to_gym_spec_transform(
            Composite(
                {
                    key: self.torchrl_env.full_observation_spec[key].clone()
                    for key in self._observation_keys
                }
            ),
            categorical_action_encoding=self.torchrl_env.__dict__.get(
                "categorical_action_encoding", True
            ),
        )
        self.to_numpy = to_numpy

    def seed(self, seed: int):
        return self.torchrl_env.set_seed(seed)

    @property
    def info_keys(self):
        return self._info_keys

    @info_keys.setter
    def info_keys(self, value):
        if value is None:
            value = []
        self._info_keys = [unravel_key(v) for v in value]

    @property
    def _observation_keys(self):
        obs_keys = self.__dict__.get("_observation_keys", None)
        if obs_keys is None:
            keys = []
            if self.info_keys:

                def check_tuple_keys(key, info_key):
                    if isinstance(info_key, tuple):
                        return key[: len(info_key)] == info_key
                    else:
                        return key[0] == info_key

                for key in self.torchrl_env.observation_spec.keys(True):
                    if isinstance(key, tuple):
                        # check if an info key has the same start
                        if any(
                            check_tuple_keys(key, info_key)
                            for info_key in self.info_keys
                        ):
                            continue
                        keys.append(key)
                    else:
                        if any(
                            key == info_key
                            for info_key in self.info_keys
                            if isinstance(info_key, str)
                        ):
                            continue
                        keys.append(key)
            else:
                keys = self.torchrl_env.observation_spec.keys(True)
            obs_keys = self.__dict__["_observation_keys"] = sorted(
                keys,
                key=lambda x: ".".join(x) if isinstance(x, tuple) else x,
            )
        return obs_keys

    @property
    def _input_keys(self):
        input_keys = self.__dict__.get("_inp_keys", None)
        if input_keys is None:
            input_keys = self.__dict__["_inp_keys"] = sorted(
                set(self.torchrl_env.state_spec.keys(True)),
                key=lambda x: ".".join(x) if isinstance(x, tuple) else x,
            )
        return input_keys

    @property
    def _action_keys(self):
        action_keys = self.__dict__.get("_act_keys", None)
        if action_keys is None:
            action_keys = self.__dict__["_act_keys"] = sorted(
                set(self.torchrl_env.full_action_spec.keys(True)),
                key=lambda x: ".".join(x) if isinstance(x, tuple) else x,
            )
        return action_keys


if _has_gymnasium:
    import gymnasium

    class _TorchRLGymnasiumWrapper(gymnasium.Env, _BaseGymWrapper):
        @implement_for("gymnasium", "1.0.0", "1.1.0")
        def step(self, action):  # noqa: F811
            raise ImportError(GYMNASIUM_1_ERROR)

        @implement_for("gymnasium", None, "1.0.0")
        def step(self, action):  # noqa: F811
            action_keys = self._action_keys
            if len(action_keys) == 1:
                self._tensordict.set(action_keys[0], action)
            else:
                raise RuntimeError(
                    "Wrapping environments with more than one action key is not supported yet."
                )
            self.torchrl_env.step(self._tensordict)
            _tensordict = step_mdp(self._tensordict)
            observation = self._tensordict.get("next")
            if self.info_keys:
                info = observation.select(*self.info_keys).to_dict()
            else:
                info = {}
            observation = observation.select(*self._observation_keys).to_dict()
            reward = self._tensordict.get(("next", "reward"))
            terminated = self._tensordict.get(("next", "terminated"))
            truncated = self._tensordict.get(
                ("next", "truncated"), torch.zeros_like(terminated)
            )
            self._tensordict = _tensordict.select(*self._input_keys)
            out = (observation, reward, terminated, truncated, info)
            if self.to_numpy:
                out = tree_map(lambda x: x.detach().cpu().numpy(), out)
            return out

        @implement_for("gymnasium", "1.1.0")
        def step(self, action):  # noqa: F811
            action_keys = self._action_keys
            if len(action_keys) == 1:
                self._tensordict.set(action_keys[0], action)
            else:
                raise RuntimeError(
                    "Wrapping environments with more than one action key is not supported yet."
                )
            self.torchrl_env.step(self._tensordict)
            _tensordict = step_mdp(self._tensordict)
            observation = self._tensordict.get("next")
            if self.info_keys:
                info = observation.select(*self.info_keys).to_dict()
            else:
                info = {}
            observation = observation.select(*self._observation_keys).to_dict()
            reward = self._tensordict.get(("next", "reward"))
            terminated = self._tensordict.get(("next", "terminated"))
            truncated = self._tensordict.get(
                ("next", "truncated"), torch.zeros_like(terminated)
            )
            self._tensordict = _tensordict.select(*self._input_keys)
            out = (observation, reward, terminated, truncated, info)
            if self.to_numpy:
                out = tree_map(lambda x: x.detach().cpu().numpy(), out)
            return out

        @implement_for("gymnasium", None, "1.0.0")
        def reset(
            self, seed: int | None = None, options: dict | None = None
        ):  # noqa: F811
            if seed is not None:
                self.torchrl_env.set_seed(seed)
            if options is None:
                options = {}
            self._tensordict = self.torchrl_env.reset(**options)
            observation = self._tensordict
            if self.info_keys:
                info = observation.select(*self.info_keys).to_dict()
            else:
                info = {}
            observation = observation.select(*self._observation_keys).to_dict()
            out = observation, info
            if self.to_numpy:
                out = tree_map(lambda x: x.detach().cpu().numpy(), out)
            return out

        @implement_for("gymnasium", "1.0.0", "1.1.0")
        def reset(self):  # noqa: F811
            raise ImportError(GYMNASIUM_1_ERROR)

        @implement_for("gymnasium", "1.1.0")
        def reset(  # noqa: F811
            self, seed: int | None = None, options: dict | None = None
        ):
            if seed is not None:
                self.torchrl_env.set_seed(seed)
            if options is None:
                options = {}
            self._tensordict = self.torchrl_env.reset(**options)
            observation = self._tensordict
            if self.info_keys:
                info = observation.select(*self.info_keys).to_dict()
            else:
                info = {}
            observation = observation.select(*self._observation_keys).to_dict()
            out = observation, info
            if self.to_numpy:
                out = tree_map(lambda x: x.detach().cpu().numpy(), out)
            return out

else:

    class _TorchRLGymnasiumWrapper:
        # placeholder
        def __init__(self, *args, **kwargs):
            raise ImportError("Gymnasium could not be found.")


if _has_gym:
    import gym

    class _TorchRLGymWrapper(gym.Env, _BaseGymWrapper):
        @implement_for("gym", "0.26", None)
        def step(self, action):  # noqa: F811
            action_keys = self._action_keys
            if len(action_keys) == 1:
                self._tensordict.set(action_keys[0], action)
            else:
                raise RuntimeError(
                    "Wrapping environments with more than one action key is not supported yet."
                )
            self.torchrl_env.step(self._tensordict)
            _tensordict = step_mdp(self._tensordict)
            observation = self._tensordict.get("next")
            if self.info_keys:
                info = observation.select(*self.info_keys).to_dict()
            else:
                info = {}
            observation = observation.select(*self._observation_keys).to_dict()
            reward = self._tensordict.get(("next", "reward"))
            terminated = self._tensordict.get(("next", "terminated"))
            truncated = self._tensordict.get(
                ("next", "truncated"), torch.zeros_like(terminated)
            )
            self._tensordict = _tensordict.select(*self._input_keys)
            out = (observation, reward, terminated, truncated, info)
            if self.to_numpy:
                out = tree_map(lambda x: x.detach().cpu().numpy(), out)
            return out

        @implement_for("gym", None, "0.26")
        def step(self, action):  # noqa: F811
            action_keys = self._action_keys
            if len(action_keys) == 1:
                self._tensordict.set(action_keys[0], action)
            else:
                raise RuntimeError(
                    "Wrapping environments with more than one action key is not supported yet."
                )
            self.torchrl_env.step(self._tensordict)
            _tensordict = step_mdp(self._tensordict)
            observation = self._tensordict.get("next")
            if self.info_keys:
                info = observation.select(*self.info_keys).to_dict()
            else:
                info = {}
            observation = observation.select(*self._observation_keys).to_dict()
            reward = self._tensordict.get(("next", "reward"))
            done = self._tensordict.get(("next", "done"))
            self._tensordict = _tensordict.select(*self._input_keys)
            out = (observation, reward, done, info)
            if self.to_numpy:
                out = tree_map(lambda x: x.detach().cpu().numpy(), out)
            return out

        @implement_for("gym", None, "0.26")
        def reset(self):  # noqa: F811
            self._tensordict = self.torchrl_env.reset()
            observation = self._tensordict
            observation = observation.select(*self._observation_keys).to_dict()
            out = observation
            if self.to_numpy:
                out = tree_map(lambda x: x.detach().cpu().numpy(), out)
            return out

        @implement_for("gym", "0.26", None)
        def reset(self):  # noqa: F811
            self._tensordict = self.torchrl_env.reset()
            observation = self._tensordict
            if self.info_keys:
                info = observation.select(*self.info_keys).to_dict()
            else:
                info = {}
            observation = observation.select(*self._observation_keys).to_dict()
            out = observation, info
            if self.to_numpy:
                out = tree_map(lambda x: x.detach().cpu().numpy(), out)
            return out

else:

    class _TorchRLGymWrapper:
        # placeholder
        def __init__(self, *args, **kwargs):
            raise ImportError("Gym could not be found.")
