# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    CompositeSpec,
    DiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs.common import _EnvWrapper
from torchrl.envs.utils import _classproperty

_has_envpool = importlib.util.find_spec("envpool") is not None


class MultiThreadedEnvWrapper(_EnvWrapper):
    """Wrapper for envpool-based multithreaded environments."""

    _verbose: bool = False

    @_classproperty
    def lib(cls):
        import envpool

        return envpool

    def __init__(
        self,
        env: Optional["envpool.python.envpool.EnvPoolMixin"] = None,  # noqa: F821
        **kwargs,
    ):
        if not _has_envpool:
            raise ImportError(
                "envpool python package or one of its dependencies (gym, treevalue) were not found. Please install these dependencies."
            )
        if env is not None:
            kwargs["env"] = env
            self.num_workers = env.config["num_envs"]
            # For synchronous mode batch size is equal to the number of workers
            self.batch_size = torch.Size([self.num_workers])
        super().__init__(**kwargs)

        # Buffer to keep the latest observation for each worker
        # It's a TensorDict when the observation consists of several variables, e.g. "position" and "velocity"
        self.obs: Union[torch.tensor, TensorDict] = self.observation_spec.zero()

    def _check_kwargs(self, kwargs: Dict):
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        import envpool

        if not isinstance(env, (envpool.python.envpool.EnvPoolMixin,)):
            raise TypeError("env is not of type 'envpool.python.envpool.EnvPoolMixin'.")

    def _build_env(self, env: "envpool.python.envpool.EnvPoolMixin"):  # noqa: F821
        return env

    def _make_specs(
        self, env: "envpool.python.envpool.EnvPoolMixin"  # noqa: F821
    ) -> None:  # noqa: F821
        from torchrl.envs.libs.gym import set_gym_backend

        with set_gym_backend("gym"):
            self.action_spec = self._get_action_spec()
            output_spec = self._get_output_spec()
            self.observation_spec = output_spec["full_observation_spec"]
            self.reward_spec = output_spec["full_reward_spec"]
            self.done_spec = output_spec["full_done_spec"]

    def _init_env(self) -> Optional[int]:
        pass

    def _reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        if tensordict is not None:
            reset_workers = tensordict.get("_reset", None)
        else:
            reset_workers = None
        if reset_workers is not None:
            reset_data = self._env.reset(np.where(reset_workers.cpu().numpy())[0])
        else:
            reset_data = self._env.reset()
        tensordict_out = self._transform_reset_output(reset_data, reset_workers)
        self.is_closed = False
        return tensordict_out

    @torch.no_grad()
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get(self.action_key)
        # Action needs to be moved to CPU and converted to numpy before being passed to envpool
        action = action.to(torch.device("cpu"))
        step_output = self._env.step(action.numpy())
        tensordict_out = self._transform_step_output(step_output)
        return tensordict_out

    def _get_action_spec(self) -> TensorSpec:
        # local import to avoid importing gym in the script
        from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform

        # Envpool provides Gym-compatible specs as env.spec.action_space and
        # DM_Control-compatible specs as env.spec.action_spec(). We use the Gym ones.

        # Gym specs produced by EnvPool don't contain batch_size, we add it to satisfy checks in EnvBase
        action_spec = _gym_to_torchrl_spec_transform(
            self._env.spec.action_space,
            device=self.device,
            categorical_action_encoding=True,
        )
        action_spec = self._add_shape_to_spec(action_spec)
        return action_spec

    def _get_output_spec(self) -> TensorSpec:
        return CompositeSpec(
            full_observation_spec=self._get_observation_spec(),
            full_reward_spec=self._get_reward_spec(),
            full_done_spec=self._get_done_spec(),
            shape=(self.num_workers,),
            device=self.device,
        )

    def _get_observation_spec(self) -> TensorSpec:
        # local import to avoid importing gym in the script
        from torchrl.envs.libs.gym import _gym_to_torchrl_spec_transform

        # Gym specs produced by EnvPool don't contain batch_size, we add it to satisfy checks in EnvBase
        observation_spec = _gym_to_torchrl_spec_transform(
            self._env.spec.observation_space,
            device=self.device,
            categorical_action_encoding=True,
        )
        observation_spec = self._add_shape_to_spec(observation_spec)
        if isinstance(observation_spec, CompositeSpec):
            return observation_spec
        return CompositeSpec(
            observation=observation_spec,
            shape=(self.num_workers,),
            device=self.device,
        )

    def _add_shape_to_spec(self, spec: TensorSpec) -> TensorSpec:
        return spec.expand((self.num_workers, *spec.shape))

    def _get_reward_spec(self) -> TensorSpec:
        return UnboundedContinuousTensorSpec(
            device=self.device,
            shape=self.batch_size,
        )

    def _get_done_spec(self) -> TensorSpec:
        spec = DiscreteTensorSpec(
            2,
            device=self.device,
            shape=self.batch_size,
            dtype=torch.bool,
        )
        return CompositeSpec(
            done=spec,
            truncated=spec.clone(),
            terminated=spec.clone(),
            shape=self.batch_size,
            device=self.device,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_workers={self.num_workers}, device={self.device})"

    def _transform_reset_output(
        self,
        envpool_output: Tuple[
            Union["treevalue.TreeValue", np.ndarray], Any  # noqa: F821
        ],
        reset_workers: Optional[torch.Tensor],
    ):
        """Process output of envpool env.reset."""
        import treevalue

        observation, _ = envpool_output
        if reset_workers is not None:
            # Only specified workers were reset - need to set observation buffer values only for them
            if isinstance(observation, treevalue.TreeValue):
                # If observation contain several fields, it will be returned as treevalue.TreeValue.
                # Convert to treevalue.FastTreeValue to allow indexing
                observation = treevalue.FastTreeValue(observation)
            self.obs[reset_workers] = self._treevalue_or_numpy_to_tensor_or_dict(
                observation
            )
        else:
            # All workers were reset - rewrite the whole observation buffer
            self.obs = TensorDict(
                self._treevalue_or_numpy_to_tensor_or_dict(observation),
                self.batch_size,
                device=self.device,
            )

        obs = self.obs.clone(False)
        obs.update(self.full_done_spec.zero())
        return obs

    def _transform_step_output(
        self, envpool_output: Tuple[Any, Any, Any, ...]
    ) -> TensorDict:
        """Process output of envpool env.step."""
        out = envpool_output
        if len(out) == 4:
            obs, reward, done, info = out
            terminated = done
            truncated = info.get("TimeLimit.truncated", done * 0)
        elif len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = terminated | truncated
        else:
            raise TypeError(
                f"The output of step was had {len(out)} elements, but only 4 or 5 are supported."
            )
        obs = self._treevalue_or_numpy_to_tensor_or_dict(obs)
        reward_and_done = {self.reward_key: torch.tensor(reward)}
        reward_and_done["done"] = done
        reward_and_done["terminated"] = terminated
        reward_and_done["truncated"] = truncated
        obs.update(reward_and_done)
        self.obs = tensordict_out = TensorDict(
            obs,
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict_out

    def _treevalue_or_numpy_to_tensor_or_dict(
        self, x: Union["treevalue.TreeValue", np.ndarray]  # noqa: F821
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Converts observation returned by EnvPool.

        EnvPool step and reset return observation as a numpy array or a TreeValue of numpy arrays, which we convert
        to a tensor or a dictionary of tensors. Currently only supports depth 1 trees, but can easily be extended to
        arbitrary depth if necessary.
        """
        import treevalue

        if isinstance(x, treevalue.TreeValue):
            ret = self._treevalue_to_dict(x)
        elif not isinstance(x, dict):
            ret = {"observation": torch.tensor(x)}
        else:
            ret = x
        return ret

    def _treevalue_to_dict(
        self, tv: "treevalue.TreeValue"  # noqa: F821
    ) -> Dict[str, Any]:
        """Converts TreeValue to a dictionary.

        Currently only supports depth 1 trees, but can easily be extended to arbitrary depth if necessary.
        """
        import treevalue

        return {k[0]: torch.tensor(v) for k, v in treevalue.flatten(tv)}

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            print(
                "MultiThreadedEnvWrapper._set_seed ignored, as setting seed in an existing envorinment is not\
                   supported by envpool. Please create a new environment, passing the seed to the constructor."
            )


class MultiThreadedEnv(MultiThreadedEnvWrapper):
    """Multithreaded execution of environments based on EnvPool.

    An alternative to ParallelEnv based on multithreading. It's faster, as it doesn't require new process spawning, but
    less flexible, as it only supports environments implemented in EnvPool library.
    Currently only supports synchronous execution mode, when the batch size is equal to the number of workers, see
    https://envpool.readthedocs.io/en/latest/content/python_interface.html#batch-size.

    >>> env = MultiThreadedEnv(num_workers=3, env_name="Pendulum-v1")
    >>> env.reset()
    >>> env.rand_step()
    >>> env.rollout(5)
    >>> env.close()

    Args:
        num_workers: number of worker threads to create.
        env_name: name of the environment, corresponding to task_id in EnvPool.
        create_env_kwargs: additional arguments which will be passed to envpool.make.
    """

    def __init__(
        self,
        num_workers: int,
        env_name: str,
        create_env_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.env_name = env_name.replace("ALE/", "")  # Naming convention of EnvPool
        self.num_workers = num_workers
        self.batch_size = torch.Size([num_workers])
        self.create_env_kwargs = create_env_kwargs or {}

        kwargs["num_workers"] = num_workers
        kwargs["env_name"] = self.env_name
        kwargs["create_env_kwargs"] = create_env_kwargs
        super().__init__(**kwargs)

    def _build_env(
        self,
        env_name: str,
        num_workers: int,
        create_env_kwargs: Optional[Dict[str, Any]],
    ) -> Any:
        import envpool

        create_env_kwargs = create_env_kwargs or {}
        env = envpool.make(
            task_id=env_name,
            env_type="gym",
            num_envs=num_workers,
            gym_reset_return_info=True,
            **create_env_kwargs,
        )
        return super()._build_env(env)

    def _set_seed(self, seed: Optional[int]):
        """Library EnvPool only supports setting a seed by recreating the environment."""
        if seed is not None:
            logging.debug("Recreating EnvPool environment to set seed.")
            self.create_env_kwargs["seed"] = seed
            self._env = self._build_env(
                env_name=self.env_name,
                num_workers=self.num_workers,
                create_env_kwargs=self.create_env_kwargs,
            )

    def _check_kwargs(self, kwargs: Dict):
        for arg in ["num_workers", "env_name", "create_env_kwargs"]:
            if arg not in kwargs:
                raise TypeError(f"Expected '{arg}' to be part of kwargs")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(env={self.env_name}, num_workers={self.num_workers}, device={self.device})"
