# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib
import json
import os
import pathlib
import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List

import numpy as np

import torch
from tensordict import PersistentTensorDict, TensorDict

from torchrl._utils import KeyDependentDefaultDict
from torchrl.data.datasets.utils import _get_root_dir
from torchrl.data.replay_buffers.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import Sampler
from torchrl.data.replay_buffers.storages import TensorStorage
from torchrl.data.replay_buffers.writers import Writer

from torchrl.envs.transforms import Compose, Resize, ToTensorImage
from torchrl.envs.utils import _classproperty

_has_tqdm = importlib.util.find_spec("tqdm", None) is not None
_has_h5py = importlib.util.find_spec("h5py", None) is not None
_has_hf_hub = importlib.util.find_spec("huggingface_hub", None) is not None

THIS_DIR = pathlib.Path(__file__).parent


class VD4RLExperienceReplay(TensorDictReplayBuffer):
    """V-D4RL experience replay dataset.

    This class downloads the H5/npz data from V-D4RL and processes it in a mmap
    format, which makes indexing (and therefore sampling) faster.

    Learn more about V-D4RL here: https://arxiv.org/abs/2206.04779

    The `"pixels"` entry is located at the root of the data, and all the data
    that is not reward, done-state, action or pixels is moved under a `"state"`
    node.

    Args:
        dataset_id (str): the dataset to be downloaded. Must be part of
            VD4RLExperienceReplay.available_datasets.
        batch_size (int): Batch-size used during sampling. Can be overridden by
            `data.sample(batch_size)` if necessary.

    Keyword Args:
        root (Path or str, optional): The V-D4RL dataset root directory.
            The actual dataset memory-mapped files will be saved under
            `<root>/<dataset_id>`. If none is provided, it defaults to
            ``~/.cache/torchrl/vd4rl`.
        download (bool or str, optional): Whether the dataset should be downloaded if
            not found. Defaults to ``True``. Download can also be passed as "force",
            in which case the downloaded data will be overwritten.
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
        totensor (bool, optional): if ``True``, a :class:`~torchrl.envs.transforms.ToTensorImage`
            transform will be included in the transform list (if not automatically
            detected). Defaults to ``True``.
        image_size (int, list of ints or None): if not ``None``, this argument
            will be used to create a :class:`~torchrl.envs.transforms.Resize`
            transform that will be appended to the transform list. Supports
            `int` types (square resizing) or a list/tuple of `int` (rectangular
            resizing). Defaults to ``None`` (no resizing).

    Attributes:
        available_datasets: a list of accepted entries to be downloaded. These
            names correspond to the directory path in the huggingface dataset
            repository. If possible, the list will be dynamically retrieved from
            huggingface. If no internet connection is available, it a cached
            version will be used.

    .. note:: Since not all experience replay have start and stop signals, we
        do not mark the episodes in the retrieved dataset.

    Examples:
        >>> import torch
        >>> torch.manual_seed(0)
        >>> from torchrl.data.datasets import VD4RLExperienceReplay
        >>> d = VD4RLExperienceReplay("main/walker_walk/random/64px", batch_size=32,
        ...     image_size=50)
        >>> for batch in d:
        ...     break
        >>> print(batch)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([32, 6]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                index: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.int64, is_shared=False),
                is_init: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: TensorDict(
                            fields={
                                height: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.float32, is_shared=False),
                                orientations: Tensor(shape=torch.Size([32, 14]), device=cpu, dtype=torch.float32, is_shared=False),
                                velocity: Tensor(shape=torch.Size([32, 9]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([32]),
                            device=cpu,
                            is_shared=False),
                        pixels: Tensor(shape=torch.Size([32, 3, 50, 50]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([32]),
                    device=cpu,
                    is_shared=False),
                observation: TensorDict(
                    fields={
                        height: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.float32, is_shared=False),
                        orientations: Tensor(shape=torch.Size([32, 14]), device=cpu, dtype=torch.float32, is_shared=False),
                        velocity: Tensor(shape=torch.Size([32, 9]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32]),
                    device=cpu,
                    is_shared=False),
                pixels: Tensor(shape=torch.Size([32, 3, 50, 50]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([32, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
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
        totensor: bool = True,
        image_size: int | List[int] | None = None,
        **env_kwargs,
    ):
        if not _has_h5py or not _has_hf_hub:
            raise ImportError(
                "h5py and huggingface_hub are required for V-D4RL datasets."
            )
        if dataset_id not in self.available_datasets:
            raise ValueError(
                f"The dataset_id {dataset_id} isn't part of the accepted datasets. "
                f"To check which dataset can be downloaded, call `{type(self)}.available_datasets`."
            )
        self.dataset_id = dataset_id
        if root is None:
            root = _get_root_dir("vd4rl")
            os.makedirs(root, exist_ok=True)
        self.root = root
        self.split_trajs = split_trajs
        self.download = download
        if self.download == "force" or (self.download and not self._is_downloaded()):
            if self.download == "force":
                try:
                    shutil.rmtree(self.data_path_root)
                    if self.data_path != self.data_path_root:
                        shutil.rmtree(self.data_path)
                except FileNotFoundError:
                    pass
            storage = self._download_and_preproc(dataset_id, data_path=self.data_path)
        elif self.split_trajs and not os.path.exists(self.data_path):
            storage = self._make_split()
        else:
            storage = self._load()
        if totensor and transform is None:
            transform = ToTensorImage(
                in_keys=["pixels", ("next", "pixels")], shape_tolerant=True
            )
        elif totensor and (
            not isinstance(transform, Compose)
            or not any(isinstance(t, ToTensorImage) for t in transform)
        ):
            transform = Compose(
                transform,
                ToTensorImage(
                    in_keys=["pixels", ("next", "pixels")], shape_tolerant=True
                ),
            )
        if image_size is not None:
            transform = Compose(
                transform, Resize(image_size, in_keys=["pixels", ("next", "pixels")])
            )
        storage = TensorStorage(storage)
        super().__init__(
            storage=storage,
            sampler=sampler,
            writer=writer,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch=prefetch,
            transform=transform,
            batch_size=batch_size,
        )

    @classmethod
    def _parse_datasets(cls):
        from huggingface_hub import HfApi

        dataset = HfApi().dataset_info("conglu/vd4rl")
        sibs = defaultdict(list)
        for sib in dataset.siblings:
            if sib.rfilename.endswith("npz") or sib.rfilename.endswith("hdf5"):
                path = Path(sib.rfilename)
                sibs[path.parent].append(path)
        return sibs

    @classmethod
    def _download_and_preproc(cls, dataset_id, data_path):
        from huggingface_hub import hf_hub_download

        files = []
        tds = []
        with tempfile.TemporaryDirectory() as tmpdir:
            sibs = cls._parse_datasets()
            # files = []
            total_steps = 0

            paths_to_proc = []
            files_to_proc = []

            for path in sibs:
                if dataset_id not in str(path):
                    continue
                for file in sibs[path]:
                    paths_to_proc.append(str(path))
                    files_to_proc.append(str(file.parts[-1]))

            with ThreadPoolExecutor(32) as executor:
                files = executor.map(
                    lambda path_file: hf_hub_download(
                        "conglu/vd4rl",
                        subfolder=path_file[0],
                        filename=path_file[1],
                        repo_type="dataset",
                        cache_dir=str(tmpdir),
                    ),
                    zip(paths_to_proc, files_to_proc),
                )
                files = list(files)
            print("Downloaded, processing files")
            if _has_tqdm:
                import tqdm

                pbar = tqdm.tqdm(files)
            else:
                pbar = files
            for local_path in pbar:
                if _has_tqdm:
                    pbar.set_description(f"file={local_path}")
                # we memmap temporarily the files for faster access later
                if local_path.endswith("hdf5"):
                    td = (
                        PersistentTensorDict.from_h5(local_path)
                        .to_tensordict()
                        .memmap(num_threads=32)
                    )
                else:
                    td = _from_npz(local_path).memmap(num_threads=32)
                td.unlock_()
                if total_steps == 0:
                    tdc = cls._process_data(td.clone())
                    td_save = tdc[0]
                tds.append(td)
                total_steps += td.shape[0]

        # From this point, the local paths are non needed anymore
        td_save = td_save.expand(total_steps).memmap_like(data_path, num_threads=32)
        print("Saved tensordict:", td_save)
        idx0 = 0
        idx1 = 0
        while len(files):
            _ = files.pop(0)
            td = tds.pop(0)
            td = cls._process_data(td)
            idx1 += td.shape[0]
            td_save[idx0:idx1] = td
            idx0 = idx1
        return td_save

    @classmethod
    def _process_data(cls, td: TensorDict):
        for name in list(td.keys()):
            # move remaining data
            if name not in _NAME_MATCH:
                td.rename_key_(name, ("state", name))
            elif name != _NAME_MATCH[name]:
                td.rename_key_(name, _NAME_MATCH[name])
        if ("next", "reward") in td.keys(True):
            td.set(("next", "reward"), td.get(("next", "reward")).unsqueeze(-1))
        if ("next", "done") in td.keys(True) and ("next", "terminated") in td.keys(
            True
        ):
            # first unsqueeze
            td.set(("next", "done"), td.get(("next", "done")).unsqueeze(-1))
            td.set(("next", "terminated"), td.get(("next", "terminated")).unsqueeze(-1))
            # create root vals
            td.set("done", torch.zeros_like(td.get(("next", "done"))))
            td.set("terminated", torch.zeros_like(td.get(("next", "terminated"))))
            # Add truncated
            td.set(
                ("next", "truncated"),
                td.get(("next", "done")) & ~td.get(("next", "terminated")),
            )

            td.set("truncated", torch.zeros_like(td.get(("next", "truncated"))))

        pixels = td.get("pixels")
        subtd = td.get_sub_tensordict(slice(0, -1))
        subtd.set(("next", "pixels"), pixels[1:], inplace=True)
        state = td.get("state", None)
        if state is not None:
            subtd.set(("next", "state"), state[1:], inplace=True)

        return td

    @_classproperty
    def available_datasets(cls):
        return cls._available_datasets()

    @classmethod
    def _available_datasets(cls):
        # try to gather paths from hf
        try:
            sibs = cls._parse_datasets()
            return [str(path)[6:] for path in sibs]
        except Exception:
            # return the default datasets
            with open(THIS_DIR / "vd4rl.json", "r") as file:
                return json.load(file)

    def _make_split(self):
        from torchrl.collectors.utils import split_trajectories

        td_data = TensorDict.load_memmap(self.data_path_root)
        td_data = split_trajectories(td_data).memmap_(self.data_path)
        return td_data

    def _load(self):
        return TensorDict.load_memmap(self.data_path)

    @property
    def data_path(self):
        if self.split_trajs:
            return Path(self.root) / (self.dataset_id + "_split")
        return self.data_path_root

    @property
    def data_path_root(self):
        return Path(self.root) / self.dataset_id

    def _is_downloaded(self):
        return os.path.exists(self.data_path_root)


def _from_npz(npz_path):
    npz = np.load(npz_path)
    npz_dict = {file: npz[file] for file in npz.files}
    return TensorDict.from_dict(npz_dict)


_NAME_MATCH = KeyDependentDefaultDict(lambda x: x)
_NAME_MATCH.update(
    {
        "is_first": "is_init",
        "is_last": ("next", "done"),
        "is_terminal": ("next", "terminated"),
        "reward": ("next", "reward"),
        "image": "pixels",
        "observation": "pixels",
        "discount": "discount",
        "action": "action",
    }
)
