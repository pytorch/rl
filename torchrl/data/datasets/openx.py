# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import torch

from tensordict import make_tensordict, NonTensorData, pad, TensorDict
from tensordict.utils import _is_non_tensor

from torchrl.data.datasets.common import BaseDatasetExperienceReplay
from torchrl.data.datasets.utils import _get_root_dir
from torchrl.data.replay_buffers.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import (
    Sampler,
    SliceSampler,
    SliceSamplerWithoutReplacement,
)
from torchrl.data.replay_buffers.storages import _collate_id, Storage, TensorStorage
from torchrl.data.replay_buffers.writers import ImmutableDatasetWriter, Writer

_has_datasets = importlib.util.find_spec("datasets", None) is not None
_has_tv = importlib.util.find_spec("torchvision", None) is not None


class OpenXExperienceReplay(BaseDatasetExperienceReplay):
    """Open X-Embodiment datasets experience replay.

    The Open X-Embodiment Dataset contains 1M+ real robot trajectories
    spanning 22 robot embodiments, collected through a collaboration between
    21 institutions, demonstrating 527 skills (160266 tasks).

    Website: https://robotics-transformer-x.github.io/

    GitHub: https://github.com/google-deepmind/open_x_embodiment

    Paper: https://arxiv.org/abs/2310.08864

    The data format follows the :ref:`TED convention <TED-format>`.

    .. note::
        Non-tensor data will be written in the tensordict data using the
        :class:`~tensordict.tensorclass.NonTensorData` primitive.
        For instance, the `language_instruction` field in the data will
        be stored in `data.get_non_tensor("language_instruction")` (or equivalently
        `data.get("language_instruction").data`). See the documentation of this
        class for more information on how to interact with non-tensor data
        stored in a :class:`~tensordict.TensorDict`.

    Args:
        dataset_id (str): The dataset to be downloaded.
            Must be part of ``OpenXExperienceReplay.available_datasets``.
        batch_size (int): Batch-size used during sampling.
            Can be overridden by `data.sample(batch_size)` if necessary.
            See ``num_slices`` and ``slice_len`` keyword arguments for a refined
            sampling strategy.
            If the ``batch_size`` is ``None`` (default), iterating over the
            dataset will deliver trajectories one at a time *whereas* calling
            :meth:`~.sample` will *still* require a batch-size to be provided.

    Keyword Args:
        shuffle (bool, optional): if ``True``, trajectories are delivered in a
            random order when the dataset is iterated over.
            If ``False``, the dataset is iterated over in the pre-defined order.

            .. warning::
              shuffle=False will also impact the sampling. We advice users to
              create a copy of the dataset where the ``shuffle`` attribute of the
              sampler is set to ``False`` if they wish to enjoy the two different
              behaviors (shuffled and not) within the same code base.

        num_slices (int, optional): the number of slices in a batch. This
            corresponds to the number of trajectories present in a batch.
            Once collected, the batch is presented as a concatenation of
            sub-trajectories that can be recovered through `batch.reshape(num_slices, -1)`.
            The `batch_size` must be divisible by `num_slices` if provided.
            This argument is exclusive with ``slice_len``.
            If the ``num_slices`` argument equates the ``batch_size``, each sample
            will belong to a different trajectory.
            If neither ``slice_len`` nor ``num_slice`` are provided:
            whenever a trajectory has a length shorter than the
            batch-size, a contiguous slice of it of length `batch_size` will be
            sampled. If the trajectory length is insufficient, an exception will
            be raised unless `pad` is not `None`.
        slice_len (int, optional): the length of slices in a batch. This
            corresponds to the length of trajectories present in a batch.
            Once collected, the batch is presented as a concatenation of
            sub-trajectories that can be recovered through `batch.reshape(-1, slice_len)`.
            The `batch_size` must be divisible by `slice_len` if provided.
            This argument is exclusive with ``num_slice``.
            If the ``slice_len`` argument equates ``1``, each sample
            will belong to a different trajectory.
            If neither ``slice_len`` nor ``num_slice`` are provided:
            whenever a trajectory has a length shorter than the
            batch-size, a contiguous slice of it of length `batch_size` will be
            sampled. If the trajectory length is insufficient, an exception will
            be raised unless `pad` is not `None`.

            .. note::
              The ``slice_len`` (but not ``num_slices``) can be used when
              iterating over a dataset without passing a batch-size in the,
              constructor. In these cases, a random sub-sequence of the
              trajectory will be chosen.

        replacement (bool, optional): if ``False``, sampling will be done
            without replacement. Defaults to ``True`` for downloaded datasets,
            ``False`` for streamed datasets.
        pad (bool, float or None): if ``True``, trajectories of insufficient length
            given the `slice_len` or `num_slices` arguments will be padded with
            0s. If another value is provided, it will be used for padding. If
            ``False`` or ``None`` (default) any encounter with a trajectory of
            insufficient length will raise an exception.
        root (Path or str, optional): The OpenX dataset root directory.
            The actual dataset memory-mapped files will be saved under
            `<root>/<dataset_id>`. If none is provided, it defaults to
            ``~/.cache/torchrl/openx`.
        streaming (bool, optional): if ``True``, the data won't be downloaded but
            read from a stream instead.

            .. note:: The formatting of the data __will change__ when `download=True`
                compared to `streaming=True`. If the data is downloaded and
                the sampler is left untouched (ie, `num_slices=None`, `slice_len=None`
                and `sampler=None`, transitions will be sampled randomly from
                the dataset. This isn't possible at a reasonable cost with
                `streaming=True`: in this case, trajectories will be sampled
                one at a time and delivered as such (with cropping to comply with
                the batch-size etc). The behavior of the two modalities is
                much more similar when `num_slices` and `slice_len` are specified,
                as in these cases, views of sub-episodes will be returned in both
                cases.

        download (bool or str, optional): Whether the dataset should be downloaded if
            not found. Defaults to ``True``. Download can also be passed as "force",
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
        strict_length (bool, optional): if ``False``, trajectories of length
            shorter than `slice_len` (or `batch_size // num_slices`) will be
            allowed to appear in the batch.
            Be mindful that this can result in effective `batch_size`  shorter
            than the one asked for! Trajectories can be split using
            :func:`torchrl.collectors.split_trajectories`. Defaults to ``True``.

    Examples:
        >>> from torchrl.data.datasets import OpenXExperienceReplay
        >>> import tempfile
        >>> # Download the data, and sample 128 elements in each batch out of two trajectories
        >>> num_slices = 2
        >>> with tempfile.TemporaryDirectory() as root:
        ...     dataset = OpenXExperienceReplay("cmu_stretch", batch_size=128,
        ...         num_slices=num_slices, download=True, streaming=False,
        ...         root=root,
        ...         )
        ...     for batch in dataset:
        ...         print(batch.reshape(num_slices, -1))
        ...         break
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([2, 64, 8]), device=cpu, dtype=torch.float64, is_shared=False),
                discount: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([2, 64, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                episode: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.int32, is_shared=False),
                index: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.int64, is_shared=False),
                is_init: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.bool, is_shared=False),
                language_embedding: Tensor(shape=torch.Size([2, 64, 512]), device=cpu, dtype=torch.float64, is_shared=False),
                language_instruction: NonTensorData(
                    data='lift open green garbage can lid',
                    batch_size=torch.Size([2, 64]),
                    device=cpu,
                    is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([2, 64, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: TensorDict(
                            fields={
                                image: Tensor(shape=torch.Size([2, 64, 3, 128, 128]), device=cpu, dtype=torch.uint8, is_shared=False),
                                state: Tensor(shape=torch.Size([2, 64, 4]), device=cpu, dtype=torch.float64, is_shared=False)},
                            batch_size=torch.Size([2, 64]),
                            device=cpu,
                            is_shared=False),
                        reward: Tensor(shape=torch.Size([2, 64, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([2, 64, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([2, 64, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([2, 64]),
                    device=cpu,
                    is_shared=False),
                observation: TensorDict(
                    fields={
                        image: Tensor(shape=torch.Size([2, 64, 3, 128, 128]), device=cpu, dtype=torch.uint8, is_shared=False),
                        state: Tensor(shape=torch.Size([2, 64, 4]), device=cpu, dtype=torch.float64, is_shared=False)},
                    batch_size=torch.Size([2, 64]),
                    device=cpu,
                    is_shared=False),
                terminated: Tensor(shape=torch.Size([2, 64, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([2, 64, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([2, 64]),
            device=cpu,
            is_shared=False)
        >>> # Read data from a stream. Deliver entire trajectories when iterating
        >>> dataset = OpenXExperienceReplay("cmu_stretch",
        ...     num_slices=num_slices, download=False, streaming=True)
        >>> for data in dataset: # data does not have a consistent shape
        ...     break
        >>> # Define batch-size dynamically
        >>> data = dataset.sample(128)  # delivers 2 sub-trajectories of length 64

    """

    available_datasets = [
        "fractal20220817_data",
        "kuka",
        "bridge",
        "taco_play",
        "jaco_play",
        "berkeley_cable_routing",
        "roboturk",
        "nyu_door_opening_surprising_effectiveness",
        "viola",
        "berkeley_autolab_ur5",
        "toto",
        "language_table",
        "columbia_cairlab_pusht_real",
        "stanford_kuka_multimodal_dataset_converted_externally_to_rlds",
        "nyu_rot_dataset_converted_externally_to_rlds",
        "stanford_hydra_dataset_converted_externally_to_rlds",
        "austin_buds_dataset_converted_externally_to_rlds",
        "nyu_franka_play_dataset_converted_externally_to_rlds",
        "maniskill_dataset_converted_externally_to_rlds",
        "furniture_bench_dataset_converted_externally_to_rlds",
        "cmu_franka_exploration_dataset_converted_externally_to_rlds",
        "ucsd_kitchen_dataset_converted_externally_to_rlds",
        "ucsd_pick_and_place_dataset_converted_externally_to_rlds",
        "austin_sailor_dataset_converted_externally_to_rlds",
        "austin_sirius_dataset_converted_externally_to_rlds",
        "bc_z",
        "usc_cloth_sim_converted_externally_to_rlds",
        "utokyo_pr2_opening_fridge_converted_externally_to_rlds",
        "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds",
        "utokyo_saytap_converted_externally_to_rlds",
        "utokyo_xarm_pick_and_place_converted_externally_to_rlds",
        "utokyo_xarm_bimanual_converted_externally_to_rlds",
        "robo_net",
        "berkeley_mvp_converted_externally_to_rlds",
        "berkeley_rpt_converted_externally_to_rlds",
        "kaist_nonprehensile_converted_externally_to_rlds",
        "stanford_mask_vit_converted_externally_to_rlds",
        "tokyo_u_lsmo_converted_externally_to_rlds",
        "dlr_sara_pour_converted_externally_to_rlds",
        "dlr_sara_grid_clamp_converted_externally_to_rlds",
        "dlr_edan_shared_control_converted_externally_to_rlds",
        "asu_table_top_converted_externally_to_rlds",
        "stanford_robocook_converted_externally_to_rlds",
        "eth_agent_affordances",
        "imperialcollege_sawyer_wrist_cam",
        "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
        "uiuc_d3field",
        "utaustin_mutex",
        "berkeley_fanuc_manipulation",
        "cmu_playing_with_food",
        "cmu_play_fusion",
        "cmu_stretch",
        "berkeley_gnm_recon",
        "berkeley_gnm_cory_hall",
        "berkeley_gnm_sac_son",
    ]

    # some very high number that should be above all trajecory lengths in the dataset
    _MAX_TRAJ_LEN = 1_000_000

    def __init__(
        self,
        dataset_id,
        batch_size: int | None = None,
        *,
        shuffle: bool = True,
        num_slices: int | None = None,
        slice_len: int | None = None,
        pad: float | bool | None = None,
        replacement: bool = None,
        streaming: bool | None = None,
        root: str | Path | None = None,
        download: bool | None = None,
        sampler: Sampler | None = None,
        writer: Writer | None = None,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        prefetch: int | None = None,
        transform: "torchrl.envs.Transform" | None = None,  # noqa-F821
        split_trajs: bool = False,
        strict_length: bool = True,
    ):
        if download is None and streaming is None:
            download = False
            streaming = True
        elif download is None:
            download = not streaming
        elif streaming is None:
            streaming = not download
        self.download = download
        self.streaming = streaming
        self.dataset_id = dataset_id
        self.split_trajs = split_trajs
        self.shuffle = shuffle
        self.num_slices = num_slices
        self.slice_len = slice_len
        self.pad = pad
        self.strict_length = strict_length
        if (self.num_slices is not None) and (self.slice_len is not None):
            raise ValueError("num_slices or slice_len can be not None, but not both.")
        if split_trajs:
            raise NotImplementedError
        if not streaming:
            if replacement is None:
                replacement = True
            if pad is not None:
                raise RuntimeError(
                    "the `pad` argument is to be used only with streaming datasets."
                )
            if root is None:
                root = _get_root_dir("openx")
                os.makedirs(root, exist_ok=True)
            self.root = Path(root)
            if self.download == "force" or (
                self.download and not self._is_downloaded()
            ):
                if download == "force" and os.path.exists(self.data_path_root):
                    shutil.rmtree(self.data_path_root)

                storage = self._download_and_preproc()
            else:
                storage = TensorStorage(TensorDict.load_memmap(self.root / dataset_id))
            if num_slices is not None or slice_len is not None:
                if sampler is not None:
                    raise ValueError(
                        "`num_slices` and `slice_len` are exclusive with the `sampler` argument."
                    )

                if replacement:
                    if not self.shuffle:
                        raise RuntimeError(
                            "shuffle=False can only be used when replacement=False."
                        )
                    sampler = SliceSampler(
                        num_slices=num_slices,
                        slice_len=slice_len,
                        strict_length=strict_length,
                    )
                else:
                    sampler = SliceSamplerWithoutReplacement(
                        num_slices=num_slices,
                        slice_len=slice_len,
                        strict_length=strict_length,
                        shuffle=self.shuffle,
                    )

        else:
            if replacement is True:
                # replacement can be False or None
                raise RuntimeError(
                    "replacement=True is not available with streamed datasets."
                )
            self.root = None
            if download:
                raise ValueError(
                    "download and streaming cannot be set to ``True`` concomitantly."
                )
            storage = _StreamingStorage(
                dataset_id=dataset_id,
                shuffle=self.shuffle,
                num_slices=self.num_slices,
                slice_len=self.slice_len,
                pad=self.pad,
            )
            if sampler is None:
                sampler = _StreamingSampler()
        if writer is None:
            writer = ImmutableDatasetWriter()
        if collate_fn is None:
            collate_fn = _collate_id
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

    def __iter__(self):
        if self._batch_size is None:
            # we can still iterate over the dataset
            if isinstance(self._storage, _StreamingStorage):
                yield from self._storage
            elif self.slice_len is not None and self.num_slices is None:
                try:
                    # truncate the trajs with slice_len
                    self._batch_size = self.slice_len
                    self.num_slices = 1
                    self.slice_len = None
                    yield from self
                finally:
                    self.slice_len = self._batch_size
                    self._batch_size = None
                    self.num_slices = None
            else:
                # if we don't have a batch size but we know how many trajectories
                # we want in each batch, we can build that on the fly.
                # The only time we can do this is if num_slices is given but not
                # slice_len.
                num_slices = self.num_slices
                if not num_slices:
                    num_slices = 1
                sampler = SliceSamplerWithoutReplacement(
                    num_slices=num_slices,
                    strict_length=False,
                    shuffle=self.shuffle,
                )
                batch_size = self._MAX_TRAJ_LEN
                yield from TensorDictReplayBuffer(
                    storage=self._storage,
                    sampler=sampler,
                    batch_size=batch_size,
                    transform=self._transform,
                )
        else:
            yield from super().__iter__()

    @property
    def data_path(self):
        if self.streaming:
            return None
        if self.split_trajs:
            return Path(self.root) / (self.dataset_id + "_split")
        return self.data_path_root

    @property
    def data_path_root(self):
        if self.streaming:
            return None
        return self.root / self.dataset_id

    def _is_downloaded(self):
        return os.path.exists(self.data_path_root)

    def _download_and_preproc(self):
        if not _has_datasets:
            raise ImportError(
                f"the `datasets` library is required for the dataset {self.dataset_id}."
            )
        import datasets

        with tempfile.TemporaryDirectory() as cache_dir:
            dataset = datasets.load_dataset(
                "jxu124/OpenX-Embodiment",
                self.dataset_id,
                streaming=False,
                split="train",
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            # iterate over the dataset a first time to count elements
            total_frames = 0

            try:
                import tqdm

                _has_tqdm = True
                pbar = tqdm.tqdm(dataset, desc="counting")
            except ImportError:
                _has_tqdm = False
                pbar = dataset

            for data in pbar:
                if total_frames == 0:
                    for step in data["data.pickle"]["steps"]:
                        td = _make_tensordict_image_conv(step).zero_()
                        # format td: requires td to have a non-null batch_size
                        td = td.expand(2, *td.shape)
                        _format_data(td, 0)
                        td = td[0]
                total_frames += len(data["data.pickle"]["steps"])
            td_data = td.expand(total_frames)

            def expand_non_tensor(x):
                if isinstance(x, NonTensorData):
                    return x.maybe_to_stack()
                return x

            td_data = td_data._apply_nest(
                expand_non_tensor,
                is_leaf=lambda x: issubclass(x, torch.Tensor) or _is_non_tensor(x),
            )
            td_data = td_data.memmap_like(self.root / self.dataset_id)
            if _has_tqdm:
                pbar = tqdm.tqdm(dataset, desc="preproc", total=total_frames)
            else:
                pbar = dataset
            idx0 = 0
            idx1 = 0
            episode = 0
            for data in pbar:
                current_ep = torch.stack(
                    [
                        _make_tensordict_image_conv(step)
                        for step in data["data.pickle"]["steps"]
                    ]
                ).contiguous()
                _format_data(current_ep, episode)
                episode += 1
                idx1 += len(current_ep)
                td_data[idx0:idx1] = current_ep
                idx0 = idx1
                if _has_tqdm:
                    pbar.update(current_ep.shape[0])
            return TensorStorage(td_data.lock_())


class _StreamingStorage(Storage):
    SLICE_MISMATCH = "The batch_size {} must be divisible by num_slices {} or slice_len {} if provided."

    def __init__(
        self,
        dataset_id: str,
        repo: str = "jxu124/OpenX-Embodiment",
        split="train",
        base_path="data.pickle",
        shuffle: bool = True,
        truncate: bool = True,
        num_slices=None,
        slice_len=None,
        pad=None,
    ):
        self.shuffle = shuffle
        self.dataset_id = dataset_id
        self.repo = repo
        self.split = split
        self._init()
        self.base_path = base_path
        self.truncate = truncate
        self.num_slices = num_slices
        self.slice_len = slice_len
        self.pad = pad

    def _init(self):
        if not _has_datasets:
            raise ImportError(
                f"the `datasets` library is required for the dataset {self.dataset_id}."
            )
        import datasets

        dataset = datasets.load_dataset(
            self.repo, self.dataset_id, streaming=True, split=self.split
        )
        if self.shuffle:
            dataset = dataset.shuffle()
        self.dataset = dataset
        self.dataset_iter = iter(dataset)

    def __iter__(self):
        episode = 0
        for data in self.dataset:
            if self.base_path:
                data = data[self.base_path]
            data = torch.stack(
                [_make_tensordict_image_conv(step) for step in data["steps"]]
            ).contiguous()
            _format_data(data, episode)
            if self.slice_len is not None:
                yield _slice_data(data, slice_len=self.slice_len, pad_value=self.pad)
            else:
                yield data

    def get(self, index: range | torch.Tensor) -> Any:
        if not isinstance(index, range):
            if (index[1:] != index[:-1] + 1).any():
                # we use a range to indicate how much data we want
                raise RuntimeError("iterable datasets do not support indexing.")
            index = range(index.shape[0])
        total = 0
        data_list = []
        episode = 0
        batch_size = index.stop
        if self.num_slices is not None:
            if batch_size % self.num_slices != 0:
                raise ValueError(
                    self.SLICE_MISMATCH.format(
                        batch_size, self.num_slices, self.slice_len
                    )
                )
            num_slices = self.num_slices
            slice_len = batch_size // num_slices
        else:
            if batch_size % self.slice_len != 0:
                raise ValueError(
                    self.SLICE_MISMATCH.format(
                        batch_size, self.num_slices, self.slice_len
                    )
                )
            slice_len = self.slice_len
            # num_slices = batch_size // slice_len

        while total < batch_size:
            try:
                data = next(self.dataset_iter)
            except StopIteration:
                self.dataset_iter = iter(self.dataset)
                data = next(self.dataset_iter)

            if self.base_path:
                data = data[self.base_path]
            data = torch.stack(
                [_make_tensordict_image_conv(step) for step in data["steps"]]
            ).contiguous()
            _format_data(data, episode)
            data = _slice_data(data, slice_len=slice_len, pad_value=self.pad)
            data_list.append(data)
            total += data.numel()
            episode += 1
        data = torch.cat(data_list)
        if self.truncate:
            return data[: index.stop]
        return data

    def dumps(self, path):
        path = Path(path)
        state_dict = self.state_dict()
        json.dump(state_dict, path / "state_dict.json")

    def state_dict(self) -> Dict[str, Any]:
        return {
            "repo": self.repo,
            "split": self.split,
            "dataset_id": self.dataset_id,
            "shuffle": self.shuffle,
            "base_path": self.base_path,
            "truncated": self.truncate,
            "num_slices": self.num_slices,
            "slice_len": self.slice_len,
            "pad": self.pad,
        }

    def loads(self, path):
        path = Path(path)
        state_dict = json.load(path / "state_dict.json")
        self.load_state_dict(state_dict)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for key, val in state_dict.items():
            setattr(self, key, val)
        self._init()

    def __len__(self):
        raise RuntimeError(
            f"{type(self)} does not have a length. Use a downloaded dataset to "
            f"access this property."
        )


def _slice_data(data: TensorDict, slice_len, pad_value):
    if data.shape[-1] < slice_len:
        if pad_value is None:
            raise RuntimeError(
                f"The trajectory length ({data.shape[-1]}) is shorter than the slice length ({slice_len}). "
                f"Decrease the slice length or provide a padding value."
            )
        if pad_value is True:
            pad_value = 0
        return pad(data, [0, slice_len - data.shape[-1]], value=pad_value)

    if data.ndim == 1:
        random_range = (
            ((data.shape[-1] - slice_len) * torch.rand(())).floor().int().item()
        )
        random_range = slice(random_range, random_range + slice_len)
    else:
        raise NotImplementedError(data)
    data = data[..., random_range]
    truncated = data.get(("next", "truncated"))
    truncated = torch.index_fill(
        truncated,
        dim=data.ndim - 1,
        value=True,
        index=torch.as_tensor(-1, device=truncated.device),
    )
    done = data.get(("next", "done"))
    data.set(("next", "truncated"), truncated)
    data.set(("next", "done"), truncated | done)
    return data


class _StreamingSampler(Sampler):
    def __init__(self):
        ...

    def sample(self, storage: Storage, batch_size: int) -> Tuple[Any, dict]:
        return range(batch_size), {}

    def _empty(self):
        return

    def dumps(self, path):
        ...

    def loads(self, path):
        ...

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ...


OPENX_KEY_MAP = {
    "is_first": "is_init",
    "is_last": ("next", "done"),
    "is_terminal": ("next", "terminated"),
    "reward": ("next", "reward"),
}


def _format_data(data: TensorDict, episode: int):
    observation_ = data.get("observation")
    observation_pad = pad(observation_[1:], [0, 1])
    data.set(("next", "observation"), observation_pad)
    for key, newkey in OPENX_KEY_MAP.items():
        data.rename_key_(key, newkey)
    data.set(
        ("next", "truncated"),
        data.get(("next", "done")) & ~data.get(("next", "terminated")),
    )

    for key in ("done", "terminated", "truncated", "reward"):
        data.set(("next", key), data.get(("next", key)).unsqueeze(-1))
        if key != "reward":
            data.set(key, torch.zeros_like(data.get(("next", key))))

    data.set(
        "episode", torch.full(data.shape, episode, device=data.device, dtype=torch.int)
    )


def _make_tensordict_image_conv(data):
    # in some datasets, the images are not well converted.
    # before building the tensordict, we load the PIL image and convert it to a tensor
    try:
        img_bytes = data["observation"]["image"]["bytes"]
        if not _has_tv:
            raise ImportError(
                "the `torchvision` library is required to read the image observation."
            )
        import torchvision.transforms.v2.functional
        from PIL import Image

        img = Image.open(io.BytesIO(img_bytes))
        tensor = torchvision.transforms.v2.functional.pil_to_tensor(img)
        data["observation"]["image"] = tensor
    except KeyError:
        pass
    return make_tensordict(data)
