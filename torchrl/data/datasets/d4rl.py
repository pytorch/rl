from typing import Callable, Optional

import numpy as np

from torchrl.collectors.utils import split_trajectories

GYM_ERR = None
try:
    import gym  # noqa

    _has_gym = True
except ModuleNotFoundError as err:
    _has_gym = False
    GYM_ERR = err

D4RL_ERR = None
try:
    import d4rl  # noqa

    _has_d4rl = True
except ModuleNotFoundError as err:
    _has_d4rl = False
    D4RL_ERR = err

import torch
from tensordict.tensordict import make_tensordict
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import Sampler
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.data.replay_buffers.writers import Writer

from torchrl.data.tensor_specs import CompositeSpec


class D4RLExperienceReplay(TensorDictReplayBuffer):
    """An Experience replay class for D4RL.

    To install D4RL, follow the instructions on the
    `official repo <https://github.com/Farama-Foundation/D4RL>`__.

    The replay buffer contains the env specs under D4RLExperienceReplay.specs.

    If present, metadata will be written in ``D4RLExperienceReplay.metadata``
    and excluded from the dataset.

    Args:
        name (str): the name of the D4RL env to get the data from.
        sampler (Sampler, optional): the sampler to be used. If none is provided
            a default RandomSampler() will be used.
        writer (Writer, optional): the writer to be used. If none is provided
            a default RoundRobinWriter() will be used.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s)/outputs.  Used when using batched
            loading from a map-style dataset.
        pin_memory (bool): whether pin_memory() should be called on the rb
            samples.
        prefetch (int, optional): number of next batches to be prefetched
            using multithreading.
        transform (Transform, optional): Transform to be executed when sample() is called.
            To chain transforms use the :obj:`Compose` class.
        split_trajs (bool, optional): if True, the trajectories will be split
            along the first dimension and padded to have a matching shape.
            Defaults to ``False``.
        from_env (bool, optional): if ``True``, :meth:`env.get_dataset` will
            be used to retrieve the dataset. Otherwise :func:`d4rl.qlearning_dataset`
            will be used. Defaults to ``True``.
        env_kwargs (key-value pairs): additional kwargs for
            :func:`d4rl.qlearning_dataset`. Supports ``terminate_on_end``
            (``False`` by default) or other kwargs if defined by D4RL library.


    Examples:
        >>> from torchrl.data.datasets.d4rl import D4RLExperienceReplay
        >>> from torchrl.envs import ObservationNorm
        >>> data = D4RLExperienceReplay("maze2d-umaze-v1")
        >>> # we can append transforms to the dataset
        >>> data.append_transform(ObservationNorm(loc=-1, scale=1.0))
        >>> data.sample(128)

    """

    def __init__(
        self,
        name,
        sampler: Optional[Sampler] = None,
        writer: Optional[Writer] = None,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        prefetch: Optional[int] = None,
        transform: Optional["Transform"] = None,  # noqa-F821
        split_trajs: bool = False,
        from_env: bool = True,
        **env_kwargs,
    ):

        if not _has_gym:
            raise ImportError("Could not import gym") from GYM_ERR
        if not _has_d4rl:
            raise ImportError("Could not import d4rl") from D4RL_ERR
        self.from_env = from_env
        if from_env:
            dataset = self._get_dataset_from_env(name, env_kwargs)
        else:
            dataset = self._get_dataset_direct(name, env_kwargs)

        if split_trajs:
            dataset = split_trajectories(dataset)
        storage = LazyMemmapStorage(dataset.shape[0])
        super().__init__(
            storage=storage,
            sampler=sampler,
            writer=writer,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch=prefetch,
            transform=transform,
        )
        self.extend(dataset)

    def _get_dataset_direct(self, name, env_kwargs):
        from torchrl.envs.libs.gym import GymWrapper

        env = GymWrapper(gym.make(name))
        dataset = d4rl.qlearning_dataset(env._env, **env_kwargs)

        dataset = make_tensordict(
            {
                k: torch.from_numpy(item)
                for k, item in dataset.items()
                if isinstance(item, np.ndarray)
            }
        )
        dataset = dataset.unflatten_keys("/")
        if "metadata" in dataset.keys():
            metadata = dataset.get("metadata")
            dataset = dataset.exclude("metadata")
            self.metadata = metadata
            # find batch size
            dataset = make_tensordict(dataset.flatten_keys("/").to_dict())
            dataset = dataset.unflatten_keys("/")
        else:
            self.metadata = {}
        dataset.rename_key("observations", "observation")
        dataset.rename_key("next_observations", ("next", "observation"))
        dataset.rename_key("terminals", "done")
        dataset.rename_key("rewards", "reward")
        dataset.rename_key("actions", "action")

        # let's make sure that the dtypes match what's expected
        for key, spec in env.observation_spec.items(True, True):
            dataset[key] = dataset[key].to(spec.dtype)
            dataset["next", key] = dataset["next", key].to(spec.dtype)
        for key, spec in env.input_spec.items(True, True):
            dataset[key] = dataset[key].to(spec.dtype)
        dataset["reward"] = dataset["reward"].to(env.reward_spec.dtype)
        dataset["done"] = dataset["done"].bool()

        dataset["done"] = dataset["done"].unsqueeze(-1)
        # dataset.rename_key("next_observations", "next/observation")
        dataset["reward"] = dataset["reward"].unsqueeze(-1)
        dataset["next"].update(dataset.select("done", "reward"))
        dataset["reward"][1:] = dataset["reward"][:-1]
        dataset["done"][1:] = dataset["done"][:-1]
        dataset["reward"][0] = 0
        dataset["done"][0] = 0
        self.specs = env.specs.clone()
        return dataset

    def _get_dataset_from_env(self, name, env_kwargs):
        """Creates an environment and retrieves the dataset using env.get_dataset().

        This method does not accept extra arguments.

        """
        if env_kwargs:
            raise RuntimeError("env_kwargs cannot be passed with using from_env=True")
        # we do a local import to avoid circular import issues
        from torchrl.envs.libs.gym import GymWrapper

        env = GymWrapper(gym.make(name))
        dataset = make_tensordict(
            {
                k: torch.from_numpy(item)
                for k, item in env.get_dataset().items()
                if isinstance(item, np.ndarray)
            }
        )
        dataset = dataset.unflatten_keys("/")
        if "metadata" in dataset.keys():
            metadata = dataset.get("metadata")
            dataset = dataset.exclude("metadata")
            self.metadata = metadata
            # find batch size
            dataset = make_tensordict(dataset.flatten_keys("/").to_dict())
            dataset = dataset.unflatten_keys("/")
        else:
            self.metadata = {}

        dataset.rename_key("observations", "observation")
        dataset.rename_key("terminals", "done")
        dataset.rename_key("rewards", "reward")
        dataset.rename_key("actions", "action")

        # let's make sure that the dtypes match what's expected
        for key, spec in env.observation_spec.items(True, True):
            dataset[key] = dataset[key].to(spec.dtype)
        for key, spec in env.input_spec.items(True, True):
            dataset[key] = dataset[key].to(spec.dtype)
        dataset["reward"] = dataset["reward"].to(env.reward_spec.dtype)
        dataset["done"] = dataset["done"].bool()

        dataset["done"] = dataset["done"].unsqueeze(-1)
        # dataset.rename_key("next_observations", "next/observation")
        dataset["reward"] = dataset["reward"].unsqueeze(-1)
        dataset = (
            dataset[:-1]
            # .exclude("reward")
            .set("next", dataset.select("observation", "reward", "done")[1:])
        )
        self.specs = env.specs.clone()
        return dataset
