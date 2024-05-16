# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from warnings import warn

import numpy as np
import torch
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import (
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    MultiDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from torchrl.data.utils import numpy_to_torch_dtype_dict
from torchrl.envs.common import _EnvWrapper
from torchrl.envs.utils import _classproperty

IMPORT_ERR = None
try:
    from mlagents_envs.base_env import ActionSpec, ActionTuple, BaseEnv, ObservationSpec
    from mlagents_envs.environment import SideChannel, UnityEnvironment

    _has_mlagents = True
except ImportError as err:
    _has_mlagents = False
    IMPORT_ERR = err


__all__ = ["UnityWrapper", "UnityEnv"]


def _unity_to_torchrl_spec_transform(spec, dtype=None, device="cpu"):
    """Maps the Unity specs to the TorchRL specs."""
    if isinstance(spec, ObservationSpec):
        shape = spec.shape
        if not len(shape):
            shape = torch.Size([1])
        dtype = numpy_to_torch_dtype_dict[dtype]
        return UnboundedContinuousTensorSpec(shape, device=device, dtype=dtype)
    elif isinstance(spec, ActionSpec):
        if spec.continuous_size == len(spec.discrete_branches) == 0:
            raise ValueError("No available actions")
        action_mask_spec = None
        d_spec = c_spec = None
        if spec.discrete_size == 1:
            d_spec = DiscreteTensorSpec(
                spec.discrete_branches[0], shape=[spec.discrete_size], device=device
            )
        else:
            d_spec = MultiDiscreteTensorSpec(
                spec.discrete_branches, shape=[spec.discrete_size], device=device
            )
            # FIXME: Need tuple support as action masks are 2D arrays
            # action_mask_spec = MultiDiscreteTensorSpec(spec.discrete_branches, dtype=torch.bool, device=device)

        if spec.continuous_size > 0:
            dtype = numpy_to_torch_dtype_dict[dtype]
            c_spec = BoundedTensorSpec(
                -1, 1, (spec.continuous_size,), dtype=dtype, device=device
            )

        if d_spec and c_spec:
            return CompositeSpec(discrete=d_spec, continuous=c_spec), action_mask_spec
        else:
            return d_spec if d_spec else c_spec, action_mask_spec
    else:
        raise TypeError(f"Unknown spec of type {type(spec)} passed")


class UnityWrapper(_EnvWrapper):
    """Unity environment wrapper.

    Examples:
        >>> env = UnityWrapper(
        ...     UnityEnvironment(
        ...         "<<PATH TO UNITY APP>>",
        ...         side_channels=[],
        ...         additional_args=[],
        ...         log_folder=<<PATH TO WHERE LOGS SHOULD BE STORED>>,
        ...         device=device,
        ...     )
        ... )
    """

    git_url = "https://github.com/Unity-Technologies/ml-agents"
    libname = "mlagents_envs"

    def __init__(self, env=None, **kwargs):
        if env is not None:
            kwargs["env"] = env
        super().__init__(**kwargs)

    def _init_env(self):
        self._behavior_names = []

    def _set_seed(self, seed: int | None):
        warn(
            "Seeding through _set_seed has not been implemented. Please set the "
            "seed when you create the environment."
        )

    @_classproperty
    def available_envs(cls) -> list[str]:
        raise NotImplementedError("You must provide your own environments")

    def _build_env(self, env: BaseEnv):
        if not env.behavior_specs:
            # Take a single step so that the brain information will be sent over
            env.step()
        return env

    def _make_specs(self, env: BaseEnv) -> None:
        # TODO: Behavior specs are immutable but new ones
        # can be added if they are created in the environment.
        # Need to account for behavior specs that are added
        # throughout the environment lifecycle.

        # IMPORTANT: This assumes that all agents have the same
        # observations and actions. To change this, we need
        # some method to allow for different specs depending on
        # agent.
        #
        # A different `Parallel` version of this environment could be
        # made where the number of agents is fixed, and then you stack
        # all of the observations together. This design would allow
        # different observations and actions, but would require
        # a fixed agent count. The difficulty with implementing a
        # `Parallel` version though is that not all agents will request
        # a decision, so the spec would have to change depending
        # on which agents request a decision.

        first_behavior_name = next(iter(env.behavior_specs.keys()))
        behavior_unity_spec = env.behavior_specs[first_behavior_name]
        observation_specs = [
            _unity_to_torchrl_spec_transform(
                spec, dtype=np.dtype("float32"), device=self.device
            )
            for spec in behavior_unity_spec.observation_specs
        ]
        behavior_id_spec = UnboundedDiscreteTensorSpec(1, device=self.device)
        agent_id_spec = UnboundedDiscreteTensorSpec(1, device=self.device)
        # FIXME: Need Tuple support here so we can support observations of varying dimensions.
        # Thus, for now we use only the first observation.
        self.observation_spec = CompositeSpec(
            observation=observation_specs[0],
            behavior_id=behavior_id_spec,
            agent_id=agent_id_spec,
        )
        self.action_spec, self.action_mask_spec = _unity_to_torchrl_spec_transform(
            behavior_unity_spec.action_spec, dtype=np.int32, device=self.device
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=[1], device=self.device)
        self.done_spec = DiscreteTensorSpec(
            n=2, shape=[1], dtype=torch.bool, device=self.device
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(env={self._env}, batch_size={self.batch_size})"
        )

    def _check_kwargs(self, kwargs: dict):
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not isinstance(env, BaseEnv):
            raise TypeError("env is not of type 'mlagents_envs.base_env.BaseEnv'.")
        if "frame_skip" in kwargs and kwargs["frame_skip"] != 1:
            # This functionality is difficult to support because not all agents will request
            # decisions at each timestep and different agents might request decisions at
            # different timesteps. This makes it difficult to do things like keep track
            # of rewards.
            raise ValueError(
                "Currently, frame_skip is not supported for Unity environments."
            )

    def behavior_id_to_name(self, behavior_id: int):
        raise self._behavior_names[behavior_id]

    def read_reward(self, reward):
        return self.reward_spec.encode(reward, ignore_device=True)

    def read_obs(self, obs: np.ndarray, behavior_name: str, agent_id: int):
        behavior_id = self._behavior_names.index(behavior_name)
        observations = self.observation_spec.encode(
            {"observation": obs, "behavior_id": behavior_id, "agent_id": agent_id},
            ignore_device=True,
        )
        return observations

    def read_action(self, action):
        action = self.action_spec.to_numpy(action, safe=False)
        # We expand the dimensions at the 0 axis because actions are
        # defined to be 2D arrays with an extra first dimension being
        # used for the number of agents in the game.
        if action.ndim == 0:
            action = np.expand_dims(action, axis=0)
        if isinstance(self.action_spec, CompositeSpec):
            action = self.action_spec.to_numpy(action, safe=False)
            continuous_action = np.expand_dims(action["continuous"], axis=0)
            discrete_action = np.expand_dims(action["discrete"], axis=0)
            action = ActionTuple(continuous_action, discrete_action)
        elif isinstance(self.action_spec, DiscreteTensorSpec | MultiDiscreteTensorSpec):
            action = np.expand_dims(action, axis=0)
            action = ActionTuple(None, action)
        else:
            action = self.action_spec.to_numpy(action)
            action = np.expand_dims(action, axis=0)
            action = ActionTuple(action, None)
        return action

    def read_action_mask(self, action_mask):
        # if not self.action_mask_spec:
        #     return None
        # return self.action_mask_spec.encode(action_mask)
        return None

    def read_done(self, done):
        return self.done_spec.encode(done)

    def _behavior_name_update(self):
        self._live_behavior_names = list(self.behavior_specs.keys())
        for k in self._live_behavior_names:
            if k not in self._behavior_names:
                # We only add to self._behavior_names if the
                # behavior name doesn't exist. This helps us
                # ensure that the index of old behaviors stays
                # the same and that we don't have duplicate entries.
                # This is important since we use the index of the behavior
                # name as an id for that behavior.
                self._behavior_names.append(k)

    def _batch_update(self, behavior_name):
        self._current_step_idx = 0
        self._current_behavior_name = behavior_name
        self._decision_steps, self._terminal_steps = self.get_steps(behavior_name)

    def _get_next_tensordict(self):
        num_steps = len(self._decision_steps) + len(self._terminal_steps)
        if self._current_step_idx >= num_steps:
            raise ValueError("All agents already have actions")
        done = False if self._current_step_idx < len(self._decision_steps) else True
        steps = self._decision_steps if not done else self._terminal_steps
        agent_id = steps.agent_id[self._current_step_idx]
        step = steps[agent_id]
        # FIXME: Need Tuple support here so we can support observations of varying dimensions.
        # Thus, for now we use only the first observation.
        obs, reward = step.obs[0], step.reward
        tensordict_out = TensorDict(
            source=self.read_obs(
                obs, behavior_name=self._current_behavior_name, agent_id=agent_id
            ),
            batch_size=self.batch_size,
            device=self.device,
        )
        # tensordict_out.set("action_mask", self.read_action_mask(action_mask))
        tensordict_out.set("reward", self.read_reward(reward))
        tensordict_out.set("done", self.read_done(done))
        return tensordict_out

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # We step through each agent one at a time, and only perform
        # an environment step (send the actions to the environment)
        # once all agents are processed. This is because Unity requires
        # all agents to have actions before stepping or else the zero
        # action will be sent.
        #
        # In order to step through each agent, we first iterate through
        # all behaviors, and determine actions for each agent in that behavior
        # and then repeat the loop until all behavior's and their
        # agents are accounted for. We then perform an environment step.
        action = tensordict.get("action")
        unity_action = self.read_action(action)
        self.set_action_for_agent(
            self._current_behavior_name, tensordict.get("agent_id").item(), unity_action
        )
        self._current_step_idx += 1
        try:
            tensordict_out = self._get_next_tensordict()
            return tensordict_out.select().set("next", tensordict_out)
        except ValueError:
            behavior_id = self._live_behavior_names.index(self._current_behavior_name)
            # If we have more behaviors to go through, keep continuing. Otherwise step the environment and
            # then continue again.
            if behavior_id < len(self._live_behavior_names) - 1:
                self._current_behavior_name = self._live_behavior_names[behavior_id + 1]
                self._batch_update(self._current_behavior_name)
                tensordict_out = self._get_next_tensordict()
                return tensordict_out.select().set("next", tensordict_out)
            else:
                self._env.step()
                self._behavior_name_update()
                self._batch_update(self._live_behavior_names[0])
                tensordict_out = self._get_next_tensordict()
                return tensordict_out.select().set("next", tensordict_out)

    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs):
        self._env.reset(**kwargs)
        self._behavior_name_update()
        self._batch_update(self._live_behavior_names[0])
        return self._get_next_tensordict()


class UnityEnv(UnityWrapper):
    """Unity environment wrapper.

    Examples:
        >>> env = UnityEnv(
        ...     "<<PATH TO UNITY APP>>",
        ...     side_channels=[],
        ...     additional_args=[],
        ...     log_folder=<<PATH TO WHERE LOGS SHOULD BE STORED>>,
        ...     device=device,
        ... )
    """

    def __init__(
        self,
        file_name: str | None = None,
        seed: int = 0,
        no_graphics: bool = False,
        timeout_wait: int = 60,
        side_channels: list[SideChannel] | None = None,
        log_folder: str | None = None,
        **kwargs,
    ):
        kwargs["file_name"] = file_name
        kwargs["seed"] = seed
        kwargs["no_graphics"] = no_graphics
        kwargs["timeout_wait"] = timeout_wait
        kwargs["side_channels"] = side_channels
        kwargs["log_folder"] = log_folder
        super().__init__(**kwargs)

    def _check_kwargs(self, kwargs: dict):
        if "file_name" not in kwargs:
            raise TypeError("Could not find environment key 'file_name' in kwargs.")

    def _build_env(
        self,
        file_name: str | None = None,
        seed: int = 0,
        no_graphics: bool = False,
        timeout_wait: int = 60,
        side_channels: list[SideChannel] | None = None,
        log_folder: str | None = None,
        **env_kwargs,
    ):
        if not _has_mlagents:
            raise RuntimeError(
                f"Unity MLAgents not found, unable to create environment. "
                f"Consider downloading and installing Unity MLAgents from"
                f" {self.git_url}"
            )
        self.file_name = file_name
        return super()._build_env(
            UnityEnvironment(
                file_name,
                seed=seed,
                no_graphics=no_graphics,
                timeout_wait=timeout_wait,
                side_channels=side_channels,
                log_folder=log_folder,
                **env_kwargs,
            )
        )

    def __repr__(self):
        return f"{super().__repr__()}(file_name={self.file_name})"
