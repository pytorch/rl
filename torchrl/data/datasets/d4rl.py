from typing import Callable, Optional

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
        env_kwargs (key-value pairs): additional kwargs for the env.

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
        **env_kwargs,
    ):
        # we do a local import to avoid circular import issues
        from torchrl.envs.libs.gym import GymWrapper

        if not _has_gym:
            raise ImportError("Could not import gym") from GYM_ERR
        if not _has_d4rl:
            raise ImportError("Could not import d4rl") from D4RL_ERR
        env = GymWrapper(gym.make(name, **env_kwargs))
        dataset = make_tensordict(
            {k: torch.from_numpy(item).clone() for k, item in env.get_dataset().items()}
        )
        dataset.rename_key("observations", "observation")
        dataset.rename_key("terminals", "done")
        dataset.rename_key("rewards", "reward")
        dataset.rename_key("actions", "action")
        dataset = dataset.unflatten_keys("/")

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
        self.specs = CompositeSpec(
            input_spec=env.input_spec,
            observation_spec=env.observation_spec,
            reward_spec=env.reward_spec,
        )
