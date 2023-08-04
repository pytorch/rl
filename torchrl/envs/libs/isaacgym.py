# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import importlib.util
import warnings

import itertools
from typing import Union, Dict, Any, List

import numpy as np
import torch

from tensordict import TensorDict, TensorDictBase
from torchrl.envs import make_composite_from_td
from torchrl.envs.libs.gym import GymWrapper

_has_isaac = importlib.util.find_spec("isaacgym") is not None

class IsaacGymWrapper(GymWrapper):
    def __init__(self, env: "isaacgymenvs.tasks.base.vec_task.Env", **kwargs):
        num_envs = env.num_envs
        super().__init__(env, torch.device(env.device), batch_size = torch.Size([num_envs]), **kwargs)
        if not hasattr(self, 'task'):
            # by convention in IsaacGymEnvs
            self.task = env.__name__

    def _make_specs(self, env: "gym.Env") -> None:
        super()._make_specs(env, batch_size=self.batch_size)
        self.observation_spec["obs"] = self.observation_spec["observation"]
        del self.observation_spec["observation"]

        data = self.rollout(3).get('next')[..., 0]
        del data[self.reward_key]
        # del data[self.done_key]
        specs = make_composite_from_td(data)
        done_spec = specs[self.done_key]
        self.done_spec = done_spec
        del specs[self.done_key]

        print('specs', specs)
        obs_spec = self.observation_spec
        obs_spec.unlock_()
        obs_spec.update(specs)
        obs_spec.lock_()
        self.__dict__['_observation_spec'] = obs_spec

    @classmethod
    def _make_envs(cls, *, task, num_envs, device, seed=None, headless=True, **kwargs):
        import isaacgym  # noqa
        import isaacgymenvs  # noqa
        envs = isaacgymenvs.make(
            seed=seed,
            task=task,
            num_envs=num_envs,
            sim_device=str(device),
            rl_device=str(device),
            headless=headless,
            **kwargs,
        )
        return envs

    def _set_seed(self, seed: int) -> int:
        # as of #665c32170d84b4be66722eea405a1e08b6e7f761 the seed points nowhere in gym.make for IsaacGymEnvs
        return seed

    def read_action(self, action):
        """Reads the action obtained from the input TensorDict and transforms it in the format expected by the contained environment.

        Args:
            action (Tensor or TensorDict): an action to be taken in the environment

        Returns: an action in a format compatible with the contained environment.

        """
        return action

    def read_done(self, done):
        """Done state reader.

        Reads a done state and returns a tuple containing:
        - a done state to be set in the environment
        - a boolean value indicating whether the frame_skip loop should be broken

        Args:
            done (np.ndarray, boolean or other format): done state obtained from the environment

        """
        return done.bool(), done.any()

    def read_reward(self, total_reward, step_reward):
        """Reads a reward and the total reward so far (in the frame skip loop) and returns a sum of the two.

        Args:
            total_reward (torch.Tensor or TensorDict): total reward so far in the step
            step_reward (reward in the format provided by the inner env): reward of this particular step

        """
        return total_reward + step_reward

    def read_obs(
        self, observations: Union[Dict[str, Any], torch.Tensor, np.ndarray]
    ) -> Dict[str, Any]:
        """Reads an observation from the environment and returns an observation compatible with the output TensorDict.

        Args:
            observations (observation under a format dictated by the inner env): observation to be read.

        """
        if isinstance(observations, dict):
            if "state" in observations and "observation" not in observations:
                # we rename "state" in "observation" as "observation" is the conventional name
                # for single observation in torchrl.
                # naming it 'state' will result in envs that have a different name for the state vector
                # when queried with and without pixels
                observations["observation"] = observations.pop("state")
        if not isinstance(observations, (TensorDictBase, dict)):
            (key,) = itertools.islice(self.observation_spec.keys(True, True), 1)
            observations = {key: observations}
        return observations

class IsaacGymEnv(IsaacGymWrapper):
    @property
    def available_envs(cls) -> List[str]:
        import isaacgymenvs  # noqa
        return list(isaacgymenvs.tasks.isaacgym_task_map.keys())
    def __init__(self, task=None, *, env=None, num_envs, device, **kwargs):
        if env is not None and task is not None:
            raise RuntimeError("Cannot provide both `task` and `env` arguments.")
        elif env is not None:
            task = env
        envs = self._make_envs(task=task, num_envs=num_envs, device=device, **kwargs)
        self.task = task
        super().__init__(envs, **kwargs)
