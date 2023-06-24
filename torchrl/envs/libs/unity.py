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
        return UnboundedContinuousTensorSpec(shape=shape, device=device, dtype=dtype)
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
            # noqa: E501 action_mask_spec = MultiDiscreteTensorSpec(spec.discrete_branches, dtype=torch.bool, device=device)

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
        pass

    def _compute_num_agents(self, env):
        num_agents = 0
        for behavior_name in env.behavior_specs.keys():
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            num_agents += len(decision_steps) + len(terminal_steps)
        return num_agents

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
        self._behavior_names = list(env.behavior_specs.keys())
        self.num_agents = self._compute_num_agents(env)
        self._agent_id_to_behavior = {}
        return env

    def _make_specs(self, env: BaseEnv) -> None:
        observation_specs = [None] * self.num_agents
        behavior_id_specs = [None] * self.num_agents
        agent_id_specs = [None] * self.num_agents
        action_specs = [None] * self.num_agents
        action_mask_specs = [None] * self.num_agents
        reward_specs = [None] * self.num_agents
        done_specs = [None] * self.num_agents
        valid_mask_specs = [None] * self.num_agents

        for behavior_name, behavior_unity_spec in env.behavior_specs.items():
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            for steps in [decision_steps, terminal_steps]:
                for agent_id in steps.agent_id:
                    self._agent_id_to_behavior[agent_id] = behavior_name

                    agent_observation_specs = [
                        _unity_to_torchrl_spec_transform(
                            spec, dtype=np.dtype("float32"), device=self.device
                        )
                        for spec in behavior_unity_spec.observation_specs
                    ]
                    agent_observation_spec = torch.stack(agent_observation_specs, dim=0)
                    observation_specs[agent_id] = agent_observation_spec

                    behavior_id_specs[agent_id] = UnboundedDiscreteTensorSpec(
                        shape=1, device=self.device, dtype=torch.int8
                    )
                    agent_id_specs[agent_id] = UnboundedDiscreteTensorSpec(
                        shape=1, device=self.device, dtype=torch.int8
                    )

                    (
                        agent_action_spec,
                        agent_action_mask_spec,
                    ) = _unity_to_torchrl_spec_transform(
                        behavior_unity_spec.action_spec,
                        dtype=np.int32,
                        device=self.device,
                    )
                    action_specs[agent_id] = agent_action_spec
                    action_mask_specs[agent_id] = agent_action_mask_spec

                    reward_specs[agent_id] = UnboundedContinuousTensorSpec(
                        shape=[1], device=self.device
                    )
                    done_specs[agent_id] = DiscreteTensorSpec(
                        n=2, shape=[1], dtype=torch.bool, device=self.device
                    )
                    valid_mask_specs[agent_id] = DiscreteTensorSpec(
                        n=2, shape=[1], dtype=torch.bool, device=self.device
                    )

        self.observation_spec = CompositeSpec(
            observation=torch.stack(observation_specs, dim=0)
        )
        self.behavior_id_spec = torch.stack(behavior_id_specs, dim=0)
        self.agent_id_spec = torch.stack(agent_id_specs, dim=0)
        self.action_spec = torch.stack(action_specs, dim=0)
        # FIXME: Support action masks
        # self.action_mask_spec = torch.stack(action_mask_specs, dim=0)
        self.reward_spec = torch.stack(reward_specs, dim=0)
        self.done_spec = torch.stack(done_specs, dim=0)
        self.valid_mask_spec = torch.stack(valid_mask_specs, dim=0)

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
            # This functionality is difficult to support because not all agents will
            # request decisions at each timestep and different agents might request
            # decisions at different timesteps. This makes it difficult to do things
            # like keep track of rewards.
            raise ValueError(
                "Currently, frame_skip is not supported for Unity environments."
            )

    def behavior_id_to_name(self, behavior_id: int):
        return self._behavior_names[behavior_id]

    def read_obs(self, obs):
        return self.observation_spec.encode({"observation": obs})

    def read_behavior(self, behavior_name):
        behavior_id = np.array(
            [self._behavior_names.index(name) for name in behavior_name]
        )
        return self.behavior_id_spec.encode(behavior_id)

    def read_agent_id(self, agent_id):
        return self.agent_id_spec.encode(agent_id)

    def read_reward(self, reward):
        return self.reward_spec.encode(reward)

    def read_valid_mask(self, valid):
        return self.valid_mask_spec.encode(valid)

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

    def _get_next_tensordict(self):
        observations = [None] * self.num_agents
        behavior_names = [None] * self.num_agents
        agent_ids = [None] * self.num_agents
        rewards = [None] * self.num_agents
        dones = [None] * self.num_agents
        valid_masks = [None] * self.num_agents

        for behavior_name in self.behavior_specs.keys():
            decision_steps, terminal_steps = self.get_steps(behavior_name)
            for i, steps in enumerate([decision_steps, terminal_steps]):
                for agent_id in steps.agent_id:
                    step = steps[agent_id]

                    rewards[agent_id] = step.reward
                    behavior_names[agent_id] = behavior_name
                    agent_ids[agent_id] = step.agent_id
                    observations[agent_id] = np.stack(step.obs, axis=0)
                    dones[agent_id] = False if i == 0 else True
                    valid_masks[agent_id] = True

        missing_agents = set(range(self.num_agents)) - set(agent_ids)
        for missing_agent in missing_agents:
            observations[missing_agent] = self.observation_spec["observation"][
                missing_agent
            ].zero()
            behavior_names[missing_agent] = self._agent_id_to_behavior[missing_agent]
            agent_ids[missing_agent] = missing_agent
            rewards[missing_agent] = self.reward_spec[missing_agent].zero()
            dones[missing_agent] = self.done_spec[missing_agent].zero()
            valid_masks[missing_agent] = False

        tensordict_out = TensorDict(
            source=self.read_obs(np.stack(observations, axis=0)),
            batch_size=self.batch_size,
            device=self.device,
        )
        tensordict_out.set("behavior_id", self.read_behavior(behavior_names))
        tensordict_out.set("agent_id", self.read_agent_id(np.stack(agent_ids, axis=0)))
        tensordict_out.set("reward", self.read_reward(np.stack(rewards, axis=0)))
        tensordict_out.set("done", self.read_done(np.stack(dones, axis=0)))
        tensordict_out.set(
            "valid_mask", self.read_valid_mask(np.stack(valid_masks, axis=0))
        )
        return tensordict_out

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        eligible_agent_mask = torch.logical_and(
            tensordict["valid_mask"], torch.logical_not(tensordict["done"])
        )
        behavior_ids = tensordict["behavior_id"][eligible_agent_mask]
        agent_ids = tensordict["agent_id"][eligible_agent_mask]
        actions = tensordict["action"].unsqueeze(-1)[eligible_agent_mask]
        for action, behavior_id, agent_id in zip(actions, behavior_ids, agent_ids):
            unity_action = self.read_action(action)
            self.set_action_for_agent(
                self.behavior_id_to_name(behavior_id.item()),
                agent_id.item(),
                unity_action,
            )
        self._env.step()
        tensordict_out = self._get_next_tensordict()
        return tensordict_out.select().set("next", tensordict_out)

    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs):
        self._env.reset(**kwargs)
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
