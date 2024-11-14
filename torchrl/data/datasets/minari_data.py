# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import json
import os.path
import shutil
import tempfile

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch

from tensordict import PersistentTensorDict, TensorDict
from torchrl._utils import KeyDependentDefaultDict, logger as torchrl_logger
from torchrl.data.datasets.common import BaseDatasetExperienceReplay
from torchrl.data.datasets.utils import _get_root_dir
from torchrl.data.replay_buffers.samplers import Sampler
from torchrl.data.replay_buffers.storages import TensorStorage
from torchrl.data.replay_buffers.writers import ImmutableDatasetWriter, Writer
from torchrl.data.tensor_specs import Bounded, Categorical, Composite, Unbounded
from torchrl.envs.utils import _classproperty

_has_tqdm = importlib.util.find_spec("tqdm", None) is not None
_has_minari = importlib.util.find_spec("minari", None) is not None

_NAME_MATCH = KeyDependentDefaultDict(lambda key: key)
_NAME_MATCH["observations"] = "observation"
_NAME_MATCH["rewards"] = "reward"
_NAME_MATCH["truncations"] = "truncated"
_NAME_MATCH["terminations"] = "terminated"
_NAME_MATCH["actions"] = "action"
_NAME_MATCH["infos"] = "info"


_DTYPE_DIR = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int64": torch.int64,
    "int32": torch.int32,
    "uint8": torch.uint8,
}


class MinariExperienceReplay(BaseDatasetExperienceReplay):
    """Minari Experience replay dataset.

    Learn more about Minari on their website: https://minari.farama.org/

    The data format follows the :ref:`TED convention <TED-format>`.

    Args:
        dataset_id (str): The dataset to be downloaded. Must be part of MinariExperienceReplay.available_datasets
        batch_size (int): Batch-size used during sampling. Can be overridden by `data.sample(batch_size)` if
            necessary.

    Keyword Args:
        root (Path or str, optional): The Minari dataset root directory.
            The actual dataset memory-mapped files will be saved under
            `<root>/<dataset_id>`. If none is provided, it defaults to
            ``~/.cache/torchrl/minari`.
        download (bool or str, optional): Whether the dataset should be downloaded if
            not found. Defaults to ``True``. Download can also be passed as ``"force"``,
            in which case the downloaded data will be overwritten.
        sampler (Sampler, optional): the sampler to be used. If none is provided
            a default RandomSampler() will be used.
        writer (Writer, optional): the writer to be used. If none is provided
            a default :class:`~torchrl.data.replay_buffers.writers.ImmutableDatasetWriter` will be used.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s)/outputs.  Used when using batched
            loading from a map-style dataset.
        pin_memory (bool): whether pin_memory() should be called on the rb
            samples.
        prefetch (int, optional): number of next batches to be prefetched
            using multithreading.
        transform (Transform, optional): Transform to be executed when sample() is called.
            To chain transforms use the :class:`~torchrl.envs.transforms.transforms.Compose` class.
        split_trajs (bool, optional): if ``True``, the trajectories will be split
            along the first dimension and padded to have a matching shape.
            To split the trajectories, the ``"done"`` signal will be used, which
            is recovered via ``done = truncated | terminated``. In other words,
            it is assumed that any ``truncated`` or ``terminated`` signal is
            equivalent to the end of a trajectory.
            Defaults to ``False``.

    Attributes:
        available_datasets: a list of accepted entries to be downloaded.

    .. note::
      Text data is currenrtly discarded from the wrapped dataset, as there is not
      PyTorch native way of representing text data.
      If this feature is required, please post an issue on TorchRL's GitHub
      repository.

    Examples:
        >>> from torchrl.data.datasets.minari_data import MinariExperienceReplay
        >>> data = MinariExperienceReplay("door-human-v1", batch_size=32, download="force")
        >>> for sample in data:
        ...     torchrl_logger.info(sample)
        ...     break
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([32, 28]), device=cpu, dtype=torch.float32, is_shared=False),
                index: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.int64, is_shared=False),
                info: TensorDict(
                    fields={
                        success: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([32]),
                    device=cpu,
                    is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        info: TensorDict(
                            fields={
                                success: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.bool, is_shared=False)},
                            batch_size=torch.Size([32]),
                            device=cpu,
                            is_shared=False),
                        observation: Tensor(shape=torch.Size([32, 39]), device=cpu, dtype=torch.float64, is_shared=False),
                        reward: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.float64, is_shared=False),
                        state: TensorDict(
                            fields={
                                door_body_pos: Tensor(shape=torch.Size([32, 3]), device=cpu, dtype=torch.float64, is_shared=False),
                                qpos: Tensor(shape=torch.Size([32, 30]), device=cpu, dtype=torch.float64, is_shared=False),
                                qvel: Tensor(shape=torch.Size([32, 30]), device=cpu, dtype=torch.float64, is_shared=False)},
                            batch_size=torch.Size([32]),
                            device=cpu,
                            is_shared=False),
                        terminated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([32]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([32, 39]), device=cpu, dtype=torch.float64, is_shared=False),
                state: TensorDict(
                    fields={
                        door_body_pos: Tensor(shape=torch.Size([32, 3]), device=cpu, dtype=torch.float64, is_shared=False),
                        qpos: Tensor(shape=torch.Size([32, 30]), device=cpu, dtype=torch.float64, is_shared=False),
                        qvel: Tensor(shape=torch.Size([32, 30]), device=cpu, dtype=torch.float64, is_shared=False)},
                    batch_size=torch.Size([32]),
                    device=cpu,
                    is_shared=False)},
            batch_size=torch.Size([32]),
            device=cpu,
            is_shared=False)

    """

    def __init__(
        self,
        dataset_id,
        batch_size: int,
        *,
        root: str | Path | None = None,
        download: bool = True,
        sampler: Sampler | None = None,
        writer: Writer | None = None,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        prefetch: int | None = None,
        transform: "torchrl.envs.Transform" | None = None,  # noqa-F821
        split_trajs: bool = False,
    ):
        self.dataset_id = dataset_id
        if root is None:
            root = _get_root_dir("minari")
            os.makedirs(root, exist_ok=True)
        self.root = root
        self.split_trajs = split_trajs
        self.download = download
        if self.download == "force" or (self.download and not self._is_downloaded()):
            if self.download == "force":
                try:
                    if os.path.exists(self.data_path_root):
                        shutil.rmtree(self.data_path_root)
                    if self.data_path != self.data_path_root:
                        shutil.rmtree(self.data_path)
                except FileNotFoundError:
                    pass
            storage = self._download_and_preproc()
        elif self.split_trajs and not os.path.exists(self.data_path):
            storage = self._make_split()
        else:
            storage = self._load()
        storage = TensorStorage(storage)

        if writer is None:
            writer = ImmutableDatasetWriter()

        super().__init__(
            storage=storage,
            sampler=sampler,
            writer=writer,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch=prefetch,
            batch_size=batch_size,
            transform=transform,
        )

    @_classproperty
    def available_datasets(self):
        if not _has_minari:
            raise ImportError("minari library not found.")
        import minari

        return minari.list_remote_datasets().keys()

    def _is_downloaded(self):
        return os.path.exists(self.data_path_root)

    @property
    def data_path(self) -> Path:
        if self.split_trajs:
            return Path(self.root) / (self.dataset_id + "_split")
        return self.data_path_root

    @property
    def data_path_root(self) -> Path:
        return Path(self.root) / self.dataset_id

    @property
    def metadata_path(self) -> Path:
        return Path(self.root) / self.dataset_id / "env_metadata.json"

    def _download_and_preproc(self):
        if not _has_minari:
            raise ImportError("minari library not found.")
        import minari

        if _has_tqdm:
            from tqdm import tqdm

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MINARI_DATASETS_PATH"] = tmpdir
            minari.download_dataset(dataset_id=self.dataset_id)
            parent_dir = Path(tmpdir) / self.dataset_id / "data"

            td_data = TensorDict()
            total_steps = 0
            torchrl_logger.info("first read through data to create data structure...")
            h5_data = PersistentTensorDict.from_h5(parent_dir / "main_data.hdf5")
            # populate the tensordict
            episode_dict = {}
            for i, (episode_key, episode) in enumerate(h5_data.items()):
                episode_num = int(episode_key[len("episode_") :])
                episode_len = episode["actions"].shape[0]
                episode_dict[episode_num] = (episode_key, episode_len)
                # Get the total number of steps for the dataset
                total_steps += episode_len
                if i == 0:
                    td_data.set("episode", 0)
                    for key, val in episode.items():
                        match = _NAME_MATCH[key]
                        if key in ("observations", "state", "infos"):
                            if (
                                not val.shape
                            ):  # no need for this, we don't need the proper length: or steps != val.shape[0] - 1:
                                if val.is_empty():
                                    continue
                                val = _patch_info(val)
                            td_data.set(("next", match), torch.zeros_like(val[0]))
                            td_data.set(match, torch.zeros_like(val[0]))
                        if key not in ("terminations", "truncations", "rewards"):
                            td_data.set(match, torch.zeros_like(val[0]))
                        else:
                            td_data.set(
                                ("next", match),
                                torch.zeros_like(val[0].unsqueeze(-1)),
                            )

            # give it the proper size
            td_data["next", "done"] = (
                td_data["next", "truncated"] | td_data["next", "terminated"]
            )
            if "terminated" in td_data.keys():
                td_data["done"] = td_data["truncated"] | td_data["terminated"]
            td_data = td_data.expand(total_steps)
            # save to designated location
            torchrl_logger.info(f"creating tensordict data in {self.data_path_root}: ")
            td_data = td_data.memmap_like(self.data_path_root)
            torchrl_logger.info(f"tensordict structure: {td_data}")

            torchrl_logger.info(f"Reading data from {max(*episode_dict) + 1} episodes")
            index = 0
            with tqdm(total=total_steps) if _has_tqdm else nullcontext() as pbar:
                # iterate over episodes and populate the tensordict
                for episode_num in sorted(episode_dict):
                    episode_key, steps = episode_dict[episode_num]
                    episode = h5_data.get(episode_key)
                    idx = slice(index, (index + steps))
                    data_view = td_data[idx]
                    data_view.fill_("episode", episode_num)
                    for key, val in episode.items():
                        match = _NAME_MATCH[key]
                        if key in (
                            "observations",
                            "state",
                            "infos",
                        ):
                            if not val.shape or steps != val.shape[0] - 1:
                                if val.is_empty():
                                    continue
                                val = _patch_info(val)
                            if steps != val.shape[0] - 1:
                                raise RuntimeError(
                                    f"Mismatching number of steps for key {key}: was {steps} but got {val.shape[0] - 1}."
                                )
                            data_view["next", match].copy_(val[1:])
                            data_view[match].copy_(val[:-1])
                        elif key not in ("terminations", "truncations", "rewards"):
                            if steps is None:
                                steps = val.shape[0]
                            else:
                                if steps != val.shape[0]:
                                    raise RuntimeError(
                                        f"Mismatching number of steps for key {key}: was {steps} but got {val.shape[0]}."
                                    )
                            data_view[match].copy_(val)
                        else:
                            if steps is None:
                                steps = val.shape[0]
                            else:
                                if steps != val.shape[0]:
                                    raise RuntimeError(
                                        f"Mismatching number of steps for key {key}: was {steps} but got {val.shape[0]}."
                                    )
                            data_view[("next", match)].copy_(val.unsqueeze(-1))
                    data_view["next", "done"].copy_(
                        data_view["next", "terminated"] | data_view["next", "truncated"]
                    )
                    if "done" in data_view.keys():
                        data_view["done"].copy_(
                            data_view["terminated"] | data_view["truncated"]
                        )
                    if pbar is not None:
                        pbar.update(steps)
                        pbar.set_description(
                            f"index={index} - episode num {episode_num}"
                        )
                    index += steps
            h5_data.close()
            # Add a "done" entry
            if self.split_trajs:
                with td_data.unlock_():
                    from torchrl.objectives.utils import split_trajectories

                    td_data = split_trajectories(td_data).memmap_(self.data_path)
            with open(self.metadata_path, "w") as metadata_file:
                dataset = minari.load_dataset(self.dataset_id)
                self.metadata = asdict(dataset.spec)
                self.metadata["observation_space"] = _spec_to_dict(
                    self.metadata["observation_space"]
                )
                self.metadata["action_space"] = _spec_to_dict(
                    self.metadata["action_space"]
                )
                json.dump(self.metadata, metadata_file)
            self._load_and_proc_metadata()
            return td_data

    def _make_split(self):
        from torchrl.collectors.utils import split_trajectories

        self._load_and_proc_metadata()
        td_data = TensorDict.load_memmap(self.data_path_root)
        td_data = split_trajectories(td_data).memmap_(self.data_path)
        return td_data

    def _load(self):
        self._load_and_proc_metadata()
        return TensorDict.load_memmap(self.data_path)

    def _load_and_proc_metadata(self):
        with open(self.metadata_path, "r") as file:
            self.metadata = json.load(file)
        self.metadata["observation_space"] = _proc_spec(
            self.metadata["observation_space"]
        )
        self.metadata["action_space"] = _proc_spec(self.metadata["action_space"])


def _proc_spec(spec):
    if spec is None:
        return
    if spec["type"] == "Dict":
        return Composite(
            {key: _proc_spec(subspec) for key, subspec in spec["subspaces"].items()}
        )
    elif spec["type"] == "Box":
        if all(item == -float("inf") for item in spec["low"]) and all(
            item == float("inf") for item in spec["high"]
        ):
            return Unbounded(spec["shape"], dtype=_DTYPE_DIR[spec["dtype"]])
        return Bounded(
            shape=spec["shape"],
            low=torch.as_tensor(spec["low"]),
            high=torch.as_tensor(spec["high"]),
            dtype=_DTYPE_DIR[spec["dtype"]],
        )
    elif spec["type"] == "Discrete":
        return Categorical(
            spec["n"], shape=spec["shape"], dtype=_DTYPE_DIR[spec["dtype"]]
        )
    else:
        raise NotImplementedError(f"{type(spec)}")


def _spec_to_dict(spec):
    from torchrl.envs.libs.gym import gym_backend

    if isinstance(spec, gym_backend("spaces").Dict):
        return {
            "type": "Dict",
            "subspaces": {key: _spec_to_dict(val) for key, val in spec.items()},
        }
    if isinstance(spec, gym_backend("spaces").Box):
        return {
            "type": "Box",
            "low": spec.low.tolist(),
            "high": spec.high.tolist(),
            "dtype": str(spec.dtype),
            "shape": tuple(spec.shape),
        }
    if isinstance(spec, gym_backend("spaces").Discrete):
        return {
            "type": "Discrete",
            "dtype": str(spec.dtype),
            "n": int(spec.n),
            "shape": tuple(spec.shape),
        }
    if isinstance(spec, gym_backend("spaces").Text):
        return
    raise NotImplementedError(f"{type(spec)}, {str(spec)}")


def _patch_info(info_td):
    # Some info dicts have tensors with one less element than others
    # We explicitely assume that the missing item is in the first position because
    # it wasn't given at reset time.
    # An alternative explanation could be that the last element is missing because
    # deemed useless for training...
    unique_shapes = defaultdict(list)
    for subkey, subval in info_td.items():
        unique_shapes[subval.shape[0]].append(subkey)
    if len(unique_shapes) == 1:
        unique_shapes[subval.shape[0] + 1] = []
    if len(unique_shapes) != 2:
        raise RuntimeError(
            f"Unique shapes in a sub-tensordict can only be of length 2, got shapes {unique_shapes}."
        )
    val_td = info_td.to_tensordict()
    min_shape = min(*unique_shapes)  # can only be found at root
    max_shape = min_shape + 1
    val_td_sel = val_td.select(*unique_shapes[min_shape])
    val_td_sel = val_td_sel.apply(
        lambda x: torch.cat([torch.zeros_like(x[:1]), x], 0), batch_size=[min_shape + 1]
    )
    val_td_sel.update(val_td.select(*unique_shapes[max_shape]))
    return val_td_sel
