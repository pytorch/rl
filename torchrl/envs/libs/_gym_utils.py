# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util

from torch.utils._pytree import tree_map

from torchrl._utils import implement_for
from torchrl.envs import step_mdp, TransformedEnv
from torchrl.envs.libs.gym import _torchrl_to_gym_spec_transform

_has_gym = importlib.util.find_spec("gym", None) is not None
_has_gymnasium = importlib.util.find_spec("gymnasium", None) is not None


class _BaseGymWrapper:
    def __init__(self, *, entry_point, to_numpy=False, transform=None, **kwargs):
        torchrl_env = entry_point(**kwargs)
        if transform is not None:
            torchrl_env = TransformedEnv(torchrl_env, transform)
        self.torchrl_env = torchrl_env
        super().__init__()
        self.action_space = _torchrl_to_gym_spec_transform(
            self.torchrl_env.action_spec,
            categorical_action_encoding=self.torchrl_env.__dict__.get(
                "categorical_action_encoding", True
            ),
        )
        self.observation_space = _torchrl_to_gym_spec_transform(
            self.torchrl_env.observation_spec,
            categorical_action_encoding=self.torchrl_env.__dict__.get(
                "categorical_action_encoding", True
            ),
        )
        self.to_numpy = to_numpy

    def seed(self, seed: int):
        return self.torchrl_env.set_seed(seed)

    @property
    def _obs_keys(self):
        obs_keys = self.__dict__.get("_observation_keys", None)
        if obs_keys is None:
            obs_keys = self.__dict__["_observation_keys"] = list(self.torchrl_env.observation_spec.keys(True, True))
        return obs_keys


if _has_gymnasium:
    import gymnasium

    class _TorchRLGymnasiumWrapper(gymnasium.Env, _BaseGymWrapper):
        @implement_for("gymnasium")
        def step(self, action):  # noqa: F811
            self._tensordict.set("action", action)
            self.torchrl_env.step(self._tensordict)
            _tensordict = step_mdp(self._tensordict)
            observation = (
                self._tensordict.get("next")
                .select(*self._obs_keys)
                .to_dict()
            )
            reward = self._tensordict.get(("next", "reward"))
            terminated = self._tensordict.get(("next", "terminated"))
            truncated = self._tensordict.get(("next", "truncated"))
            info = {}
            self._tensordict = _tensordict
            out = (observation, reward, terminated, truncated, info)
            if self.to_numpy:
                out = tree_map(lambda x: x.detach().cpu().numpy(), out)
            return out

        @implement_for("gymnasium")
        def reset(self):  # noqa: F811
            self._tensordict = self.torchrl_env.reset()
            observation = self._tensordict.select(
                *self._obs_keys,
            ).to_dict()
            out = observation, {}
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
            self._tensordict.set("action", action)
            self.torchrl_env.step(self._tensordict)
            _tensordict = step_mdp(self._tensordict)
            observation = (
                self._tensordict.get("next")
                .select(*self._obs_keys)
                .to_dict()
            )
            reward = self._tensordict.get(("next", "reward"))
            terminated = self._tensordict.get(("next", "terminated"))
            truncated = self._tensordict.get(("next", "truncated"))
            info = {}
            self._tensordict = _tensordict
            out = (observation, reward, terminated, truncated, info)
            if self.to_numpy:
                out = tree_map(lambda x: x.detach().cpu().numpy(), out)
            return out

        @implement_for("gym", None, "0.26")
        def step(self, action):  # noqa: F811
            self._tensordict.set("action", action)
            self.torchrl_env.step(self._tensordict)
            _tensordict = step_mdp(self._tensordict)
            observation = (
                self._tensordict.get("next")
                .select(*self._obs_keys)
                .to_dict()
            )
            reward = self._tensordict.get(("next", "reward"))
            done = self._tensordict.get(("next", "done"))
            info = {}
            self._tensordict = _tensordict
            out = (observation, reward, done, info)
            if self.to_numpy:
                out = tree_map(lambda x: x.detach().cpu().numpy(), out)
            return out

        @implement_for("gym", None, "0.26")
        def reset(self):  # noqa: F811
            self._tensordict = self.torchrl_env.reset()
            observation = self._tensordict.select(
                *self._obs_keys
            ).to_dict()
            out = observation
            if self.to_numpy:
                out = tree_map(lambda x: x.detach().cpu().numpy(), out)
            return out

        @implement_for("gym", "0.26", None)
        def reset(self):  # noqa: F811
            self._tensordict = self.torchrl_env.reset()
            observation = self._tensordict.select(
                *self._obs_keys
            ).to_dict()
            out = observation, {}
            if self.to_numpy:
                out = tree_map(lambda x: x.detach().cpu().numpy(), out)
            return out

else:

    class _TorchRLGymWrapper:
        # placeholder
        def __init__(self, *args, **kwargs):
            raise ImportError("Gym could not be found.")
