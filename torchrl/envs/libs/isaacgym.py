# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import itertools
from typing import Union, Dict, Any

import numpy as np
import torch

from tensordict import TensorDict, TensorDictBase
from torchrl.envs.libs.gym import GymWrapper


class IsaacGymWrapper(GymWrapper):
    def __init__(self, env, *, num_envs, **kwargs):
        super().__init__(env, **kwargs)
        self.__dict__['_input_spec'] = self.input_spec.expand(num_envs, *self.input_spec.shape)
        self.__dict__['_output_spec'] = self.output_spec.expand(num_envs, *self.output_spec.shape)
        self.batch_size = torch.Size([num_envs])
        self.__dict__['_device'] = torch.device(self._env.device)

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
        return done, done.any()

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
        print('observation', TensorDict(observations, []))
        print('obs spec', self.observation_spec)
        return observations
