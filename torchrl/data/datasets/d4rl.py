# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib
import os
import urllib
import warnings
from typing import Callable

import numpy as np

import torch

from tensordict import PersistentTensorDict
from tensordict.tensordict import make_tensordict

from torchrl.collectors.utils import split_trajectories
from torchrl.data.datasets.d4rl_infos import D4RL_DATASETS
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import Sampler
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.data.replay_buffers.writers import Writer


class D4RLExperienceReplay(TensorDictReplayBuffer):
    """An Experience replay class for D4RL.

    To install D4RL, follow the instructions on the
    `official repo <https://github.com/Farama-Foundation/D4RL>`__.

    The replay buffer contains the env specs under D4RLExperienceReplay.specs.

    If present, metadata will be written in ``D4RLExperienceReplay.metadata``
    and excluded from the dataset.

    The transitions are reconstructed using ``done = terminated | truncated`` and
    the ``("next", "observation")`` of ``"done"`` states are zeroed.

    Args:
        name (str): the name of the D4RL env to get the data from.
        batch_size (int): the batch size to use during sampling.
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
        split_trajs (bool, optional): if ``True``, the trajectories will be split
            along the first dimension and padded to have a matching shape.
            To split the trajectories, the ``"done"`` signal will be used, which
            is recovered via ``done = truncated | terminated``. In other words,
            it is assumed that any ``truncated`` or ``terminated`` signal is
            equivalent to the end of a trajectory. For some datasets from
            ``D4RL``, this may not be true. It is up to the user to make
            accurate choices regarding this usage of ``split_trajs``.
            Defaults to ``False``.
        from_env (bool, optional): if ``True``, :meth:`env.get_dataset` will
            be used to retrieve the dataset. Otherwise :func:`d4rl.qlearning_dataset`
            will be used. Defaults to ``True``.

            .. note::

              Using ``from_env=False`` will provide fewer data than ``from_env=True``.
              For instance, the info keys will be left out.
              Usually, ``from_env=False`` with ``terminate_on_end=True`` will
              lead to the same result as ``from_env=True``, with the latter
              containing meta-data and info entries that the former does
              not possess.

            .. note::

              The keys in ``from_env=True`` and ``from_env=False`` *may* unexpectedly
              differ. In particular, the ``"truncated"`` key (used to determine the
              end of an episode) may be absent when ``from_env=False`` but present
              otherwise, leading to a different slicing when ``traj_splits`` is enabled.
        direct_download (bool): if ``True``, the data will be downloaded without
            requiring D4RL. If ``None``, if ``d4rl`` is present in the env it will
            be used to download the dataset, otherwise the download will fall back
            on ``direct_download=True``.
            This is not compatible with ``from_env=True``.
            Defaults to ``None``.
        use_truncated_as_done (bool, optional): if ``True``, ``done = terminated | truncated``.
            Otherwise, only the ``terminated`` key is used. Defaults to ``True``.
        terminate_on_end (bool, optional): Set ``done=True`` on the last timestep
            in a trajectory. Default is ``False``, and will discard the
            last timestep in each trajectory.
        **env_kwargs (key-value pairs): additional kwargs for
            :func:`d4rl.qlearning_dataset`.


    Examples:
        >>> from torchrl.data.datasets.d4rl import D4RLExperienceReplay
        >>> from torchrl.envs import ObservationNorm
        >>> data = D4RLExperienceReplay("maze2d-umaze-v1", 128)
        >>> # we can append transforms to the dataset
        >>> data.append_transform(ObservationNorm(loc=-1, scale=1.0))
        >>> data.sample(128)

    """

    D4RL_ERR = None

    @classmethod
    def _import_d4rl(cls):
        cls._has_d4rl = importlib.util.find_spec("d4rl") is not None
        try:
            import d4rl  # noqa

        except ModuleNotFoundError as err:
            cls.D4RL_ERR = err
        except Exception:
            pass

    def __init__(
        self,
        name,
        batch_size: int,
        sampler: Sampler | None = None,
        writer: Writer | None = None,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        prefetch: int | None = None,
        transform: "torchrl.envs.Transform" | None = None,  # noqa-F821
        split_trajs: bool = False,
        from_env: bool = None,
        use_truncated_as_done: bool = True,
        direct_download: bool = None,
        terminate_on_end: bool = None,
        **env_kwargs,
    ):
        self.use_truncated_as_done = use_truncated_as_done

        if not from_env and direct_download is None:
            self._import_d4rl()
            direct_download = not self._has_d4rl

        if not direct_download:
            if from_env is None:
                warnings.warn(
                    "from_env will soon default to ``False``, ie the data will be "
                    "downloaded without relying on d4rl by default. "
                    "For now, ``True`` will still be the default. "
                    "To disable this warning, explicitly pass the ``from_env`` argument "
                    "during construction of the dataset.",
                    category=DeprecationWarning,
                )
                from_env = True
            self.from_env = from_env
            if terminate_on_end is None:
                # we use the default of d4rl
                terminate_on_end = False
            self._import_d4rl()

            if not self._has_d4rl:
                raise ImportError("Could not import d4rl") from self.D4RL_ERR

            if from_env:
                dataset = self._get_dataset_from_env(name, env_kwargs)
            else:
                if self.use_truncated_as_done:
                    warnings.warn(
                        "Using use_truncated_as_done=True + terminate_on_end=True "
                        "with from_env=False may not have the intended effect "
                        "as the timeouts (truncation) "
                        "can be absent from the static dataset."
                    )
                env_kwargs.update({"terminate_on_end": terminate_on_end})
                dataset = self._get_dataset_direct(name, env_kwargs)
        else:
            if from_env is None:
                from_env = False
            self.from_env = from_env
            if terminate_on_end is False:
                raise ValueError(
                    "Using terminate_on_end=False is not compatible with direct_download=True."
                )
            dataset = self._get_dataset_direct_download(name, env_kwargs)
        # Fill unknown next states with 0
        dataset["next", "observation"][dataset["next", "done"].squeeze()] = 0

        if split_trajs:
            dataset = split_trajectories(dataset)
            dataset["next", "done"][:, -1] = True

        storage = LazyMemmapStorage(dataset.shape[0])
        super().__init__(
            batch_size=batch_size,
            storage=storage,
            sampler=sampler,
            writer=writer,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch=prefetch,
            transform=transform,
        )
        self.extend(dataset)

    def _get_dataset_direct_download(self, name, env_kwargs):
        """Directly download and use a D4RL dataset."""
        if env_kwargs:
            raise RuntimeError(
                f"Cannot pass env_kwargs when `direct_download=True`. Got env_kwargs keys: {env_kwargs.keys()}"
            )
        url = D4RL_DATASETS.get(name, None)
        if url is None:
            raise KeyError(f"Env {name} not found.")
        h5path = _download_dataset_from_url(url)
        # h5path_parent = Path(h5path).parent
        dataset = PersistentTensorDict.from_h5(h5path)
        dataset = dataset.to_tensordict()
        with dataset.unlock_():
            dataset = self._process_data_from_env(dataset)
        return dataset

    def _get_dataset_direct(self, name, env_kwargs):
        from torchrl.envs.libs.gym import GymWrapper

        type(self)._import_d4rl()

        if not self._has_d4rl:
            raise ImportError("Could not import d4rl") from self.D4RL_ERR
        import d4rl
        import gym

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
        dataset.rename_key_("observations", "observation")
        dataset.set("next", dataset.select())
        dataset.rename_key_("next_observations", ("next", "observation"))
        dataset.rename_key_("terminals", "terminated")
        if "timeouts" in dataset.keys():
            dataset.rename_key_("timeouts", "truncated")
        if self.use_truncated_as_done:
            done = dataset.get("terminated") | dataset.get("truncated", False)
            dataset.set("done", done)
        else:
            dataset.set("done", dataset.get("terminated"))
        dataset.rename_key_("rewards", "reward")
        dataset.rename_key_("actions", "action")

        # let's make sure that the dtypes match what's expected
        for key, spec in env.observation_spec.items(True, True):
            dataset[key] = dataset[key].to(spec.dtype)
            dataset["next", key] = dataset["next", key].to(spec.dtype)
        dataset["action"] = dataset["action"].to(env.action_spec.dtype)
        dataset["reward"] = dataset["reward"].to(env.reward_spec.dtype)

        # format done etc
        dataset["done"] = dataset["done"].bool().unsqueeze(-1)
        dataset["terminated"] = dataset["terminated"].bool().unsqueeze(-1)
        if "truncated" in dataset.keys():
            dataset["truncated"] = dataset["truncated"].bool().unsqueeze(-1)
        dataset["reward"] = dataset["reward"].unsqueeze(-1)
        dataset["next"].update(
            dataset.select("reward", "done", "terminated", "truncated", strict=False)
        )
        dataset = (
            dataset.clone()
        )  # make sure that all tensors have a different data_ptr
        self._shift_reward_done(dataset)
        self.specs = env.specs.clone()
        return dataset

    def _get_dataset_from_env(self, name, env_kwargs):
        """Creates an environment and retrieves the dataset using env.get_dataset().

        This method does not accept extra arguments.

        """
        if env_kwargs:
            raise RuntimeError("env_kwargs cannot be passed with using from_env=True")
        import gym

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
        dataset = self._process_data_from_env(dataset, env)
        return dataset

    def _process_data_from_env(self, dataset, env=None):
        if "metadata" in dataset.keys():
            metadata = dataset.get("metadata")
            dataset = dataset.exclude("metadata")
            self.metadata = metadata
            # find batch size
            dataset = make_tensordict(dataset.flatten_keys("/").to_dict())
            dataset = dataset.unflatten_keys("/")
        else:
            self.metadata = {}

        dataset.rename_key_("observations", "observation")
        dataset.rename_key_("terminals", "terminated")
        if "timeouts" in dataset.keys():
            dataset.rename_key_("timeouts", "truncated")
        if self.use_truncated_as_done:
            dataset.set(
                "done",
                dataset.get("terminated") | dataset.get("truncated", False),
            )
        else:
            dataset.set("done", dataset.get("terminated"))

        dataset.rename_key_("rewards", "reward")
        dataset.rename_key_("actions", "action")
        try:
            dataset.rename_key_("infos", "info")
        except KeyError:
            pass

        # let's make sure that the dtypes match what's expected
        if env is not None:
            for key, spec in env.observation_spec.items(True, True):
                dataset[key] = dataset[key].to(spec.dtype)
            dataset["action"] = dataset["action"].to(env.action_spec.dtype)
            dataset["reward"] = dataset["reward"].to(env.reward_spec.dtype)

        # format done
        dataset["done"] = dataset["done"].bool().unsqueeze(-1)
        dataset["terminated"] = dataset["terminated"].bool().unsqueeze(-1)
        if "truncated" in dataset.keys():
            dataset["truncated"] = dataset["truncated"].bool().unsqueeze(-1)

        dataset["reward"] = dataset["reward"].unsqueeze(-1)
        if "next_observations" in dataset.keys():
            dataset = dataset[:-1].set(
                "next",
                dataset.select("info", strict=False)[1:],
            )
            dataset.rename_key_("next_observations", ("next", "observation"))
        else:
            dataset = dataset[:-1].set(
                "next",
                dataset.select("observation", "info", strict=False)[1:],
            )
        dataset["next"].update(
            dataset.select("reward", "done", "terminated", "truncated", strict=False)
        )
        dataset = (
            dataset.clone()
        )  # make sure that all tensors have a different data_ptr
        self._shift_reward_done(dataset)
        if env is not None:
            self.specs = env.specs.clone()
        else:
            self.specs = None
        return dataset

    def _shift_reward_done(self, dataset):
        dataset["reward"] = dataset["reward"].clone()
        dataset["reward"][1:] = dataset["reward"][:-1].clone()
        dataset["reward"][0] = 0
        for key in ("done", "terminated", "truncated"):
            if key not in dataset.keys():
                continue
            dataset[key] = dataset[key].clone()
            dataset[key][1:] = dataset[key][:-1].clone()
            dataset[key][0] = 0


def _download_dataset_from_url(dataset_url):
    dataset_filepath = _filepath_from_url(dataset_url)
    if not os.path.exists(dataset_filepath):
        print("Downloading dataset:", dataset_url, "to", dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    return dataset_filepath


def _filepath_from_url(dataset_url):
    _, dataset_name = os.path.split(dataset_url)
    dataset_filepath = os.path.join(DATASET_PATH, dataset_name)
    return dataset_filepath


def _set_dataset_path(path):
    global DATASET_PATH
    DATASET_PATH = path
    os.makedirs(path, exist_ok=True)


_set_dataset_path(
    os.environ.get(
        "D4RL_DATASET_DIR", os.path.expanduser("~/.cache/torchrl/data/d4rl/datasets")
    )
)

if __name__ == "__main__":
    data = D4RLExperienceReplay("kitchen-partial-v0", batch_size=128)
    print(data)
    for sample in data:
        print(sample)
        break
