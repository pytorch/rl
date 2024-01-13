# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from torch.utils._pytree import tree_map

from torchrl._utils import implement_for
from torchrl.envs.libs.gym import _torchrl_to_gym_spec_transform


class TorchRLGymWrapper(gymnasium.Env):
    def __init__(self, env_cls, to_numpy=False, **kwargs):
        self.torchrl_env = env_cls(**kwargs)
        super().__init__()
        self.action_space = _torchrl_to_gym_spec_transform(
            self.torchrl_env.action_spec,
            categorical_action_encoding=self.torchrl_env.categorical_action_encoding,
        )
        self.observation_space = _torchrl_to_gym_spec_transform(
            self.torchrl_env.observation_spec,
            categorical_action_encoding=self.torchrl_env.categorical_action_encoding,
        )
        self.to_numpy = to_numpy

    @implement_for("gymnasium")
    def step(self, action):
        self._tensordict.set("action", action)
        self.torchrl_env.step(self._tensordict)
        _tensordict = step_mdp(self._tensordict)
        keys = list(self.torchrl_env.observation_spec.keys())
        observation = (
            self._tensordict.get("next")
            .select(*self.torchrl_env.observation_spec.keys())
            .to_dict()
        )
        reward = self._tensordict.get(("next", "reward"))
        terminated = self._tensordict.get(("next", "terminated"))
        truncated = self._tensordict.get(("next", "truncated"))
        info = {}
        self._tensordict = _tensordict
        out = (observation, reward, terminated, truncated, info)
        if self.to_numpy():
            out = tree_map(lambda x: x.detach().cpu().numpy(), out)
        return out

    @implement_for("gym", "0.26", None)
    def step(self, action):
        self._tensordict.set("action", action)
        self.torchrl_env.step(self._tensordict)
        _tensordict = step_mdp(self._tensordict)
        keys = list(self.torchrl_env.observation_spec.keys())
        observation = (
            self._tensordict.get("next")
            .select(*self.torchrl_env.observation_spec.keys())
            .to_dict()
        )
        reward = self._tensordict.get(("next", "reward"))
        terminated = self._tensordict.get(("next", "terminated"))
        truncated = self._tensordict.get(("next", "truncated"))
        info = {}
        self._tensordict = _tensordict
        out = (observation, reward, terminated, truncated, info)
        if self.to_numpy():
            out = tree_map(lambda x: x.detach().cpu().numpy(), out)
        return out

    @implement_for("gym", None, "0.26")
    def step(self, action):
        self._tensordict.set("action", action)
        self.torchrl_env.step(self._tensordict)
        _tensordict = step_mdp(self._tensordict)
        keys = list(self.torchrl_env.observation_spec.keys())
        observation = (
            self._tensordict.get("next")
            .select(*self.torchrl_env.observation_spec.keys())
            .to_dict()
        )
        reward = self._tensordict.get(("next", "reward"))
        done = self._tensordict.get(("next", "done"))
        info = {}
        self._tensordict = _tensordict
        out = (observation, reward, done, info)
        if self.to_numpy():
            out = tree_map(lambda x: x.detach().cpu().numpy(), out)
        return out

    @implement_for("gymnasium")
    def reset(self):
        self._tensordict = self.torchrl_env.reset()
        observation = self._tensordict.select(
            *self.torchrl_env.observation_spec.keys()
        ).to_dict()
        out = observation, {}
        if self.to_numpy():
            out = tree_map(lambda x: x.detach().cpu().numpy(), out)
        return out

    @implement_for("gym", None, "0.26")
    def reset(self):
        self._tensordict = self.torchrl_env.reset()
        observation = self._tensordict.select(
            *self.torchrl_env.observation_spec.keys()
        ).to_dict()
        out = observation
        if self.to_numpy():
            out = tree_map(lambda x: x.detach().cpu().numpy(), out)
        return out

    @implement_for("gym", "0.26", None)
    def reset(self):
        self._tensordict = self.torchrl_env.reset()
        observation = self._tensordict.select(
            *self.torchrl_env.observation_spec.keys()
        ).to_dict()
        out = observation, {}
        if self.to_numpy():
            out = tree_map(lambda x: x.detach().cpu().numpy(), out)
        return out
