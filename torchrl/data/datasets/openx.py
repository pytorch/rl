# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util

import io
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Tuple

import torch
import tqdm

from tensordict import make_tensordict, pad, TensorDict

from torchrl.data import ImmutableDatasetWriter, ReplayBuffer, Storage, Writer
from torchrl.data.datasets.utils import _get_root_dir
from torchrl.data.replay_buffers import Sampler
from torchrl.data.replay_buffers.storages import _collate_id, TensorStorage

_has_datasets = importlib.util.find_spec("datasets", None) is not None
_has_tv = importlib.util.find_spec("torchvision", None) is not None


class OpenXExperienceReplay(ReplayBuffer):
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

    """Open X-Embodiment datasets experience replay.
    
    The Open X-Embodiment Dataset contains 1M+ real robot trajectories 
    spanning 22 robot embodiments, collected through a collaboration between 
    21 institutions, demonstrating 527 skills (160266 tasks).
    
    .. note::
        Images ...

    .. note::
        Text data ...
    
    Args:
        TODO

    Keyword Args:
        TODO

    Examples:
        TODO

    """

    def __init__(
        self,
        dataset_id,
        batch_size: int | None,
        *,
        streaming: bool = True,
        root: str | Path | None = None,
        download: bool = False,
        sampler: Sampler | None = None,
        writer: Writer | None = None,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        prefetch: int | None = None,
        transform: "torchrl.envs.Transform" | None = None,  # noqa-F821
        split_trajs: bool = False,
    ):
        self.download = download
        self.streaming = streaming
        self.dataset_id = dataset_id
        self.split_trajs = split_trajs
        if split_trajs:
            raise NotImplementedError
        if not streaming:
            if root is None:
                root = _get_root_dir("openx")
                os.makedirs(root, exist_ok=True)
            self.root = Path(root)
            if self.download == "force" or (
                self.download and not self._is_downloaded()
            ):
                storage = self._download_and_preproc()
            else:
                storage = TensorStorage(TensorDict.load_memmap(self.root / dataset_id))
        else:
            self.root = None
            if download:
                raise ValueError(
                    "download and streaming cannot be set to ``True`` concomitantly."
                )
            storage = _StreamingStorage(dataset_id=dataset_id)
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
            )
            # iterate over the dataset a first time to count elements
            total_frames = 0
            pbar = tqdm.tqdm(dataset, desc="counting")
            for data in pbar:
                if total_frames == 0:
                    for step in data["data.pickle"]["steps"]:
                        td = _make_tensordict_image_conv(step).zero_()
                total_frames += len(data["data.pickle"]["steps"])
            td_data = (
                td.expand(total_frames)
                .memmap_like(self.root / self.dataset_id)
                .unlock_()
            )
            pbar = tqdm.tqdm(dataset, desc="preproc", total=total_frames)
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
                pbar.update(current_ep.shape[0])
            print("total episodes", td_data["next", "done"].sum())
            return TensorStorage(td_data.lock_())


class _StreamingStorage(Storage):
    def __init__(
        self,
        dataset_id: str,
        repo: str = "jxu124/OpenX-Embodiment",
        split="train",
        base_path="data.pickle",
        shuffle: bool = True,
        truncate: bool = True,
    ):
        if not _has_datasets:
            raise ImportError(
                f"the `datasets` library is required for the dataset {dataset_id}."
            )
        import datasets

        dataset = datasets.load_dataset(repo, dataset_id, streaming=True, split=split)
        if shuffle:
            dataset = dataset.shuffle()
        self.dataset = iter(dataset)
        self.base_path = base_path
        self.truncate = truncate

    def get(self, index: int) -> Any:
        if not isinstance(index, range):
            # we use a range to indicate how much data we want
            raise RuntimeError("iterable datasets do not support indexing.")
        total = 0
        data_list = []
        episode = 0
        while total < index.stop:
            data = next(self.dataset)
            if self.base_path:
                data = data[self.base_path]
            data = torch.stack(
                [_make_tensordict_image_conv(step) for step in data["steps"]]
            ).contiguous()
            _format_data(data, episode)
            data_list.append(data)
            total += data.numel()
            episode += 1
        data = torch.cat(data_list)
        if self.truncate:
            return data[: index.stop]
        return data

    def __len__(self):
        raise RuntimeError(
            f"{type(self)} does not have a length. Use a downloaded dataset to "
            f"access this property."
        )


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
        data.get(("next", "done")) ^ data.get(("next", "terminated")),
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
                f"the `torchvision` library is required to read the image observation."
            )
        import torchvision.transforms.v2.functional
        from PIL import Image

        img = Image.open(io.BytesIO(img_bytes))
        tensor = torchvision.transforms.v2.functional.pil_to_tensor(img)
        data["observation"]["image"] = tensor
    except KeyError:
        pass
    return make_tensordict(data)
