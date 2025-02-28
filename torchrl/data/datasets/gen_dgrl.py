# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import os
import shutil
import tarfile
import tempfile
import typing as tp
from pathlib import Path

import numpy as np
import torch

from tensordict import TensorDict
from torchrl._utils import logger as torchrl_logger
from torchrl.data.datasets.common import BaseDatasetExperienceReplay
from torchrl.data.datasets.utils import _get_root_dir
from torchrl.data.replay_buffers.storages import TensorStorage
from torchrl.envs.utils import _classproperty

_has_tqdm = importlib.util.find_spec("tqdm", None) is not None
_has_requests = importlib.util.find_spec("requests", None) is not None


class GenDGRLExperienceReplay(BaseDatasetExperienceReplay):
    """Gen-DGRL Experience Replay dataset.

    This dataset accompanies the paper "The Generalization Gap in Offline Reinforcement Learning".

    Arxiv: https://arxiv.org/abs/2312.05742

    GitHub: https://github.com/facebookresearch/gen_dgrl

    The data format follows the :ref:`TED convention <TED-format>`.

    This class gives you access to the ProcGen dataset. Each `dataset_id` registered
    in `GenDGRLExperienceReplay.available_datasets` consists in a particular task
    (`"bigfish"`, `"bossfight"`, ...) separated from a category (`"1M_E"`, `"1M_S"`, ...)
    by a comma (`"bigfish-1M_E"`, ...).

    During download and preparation, the data is downloaded as .tar files,
    where each trajectory is stored independently in a .npy file. Each of these
    files is extracted, written in a contiguous mmap tensor, and then cleared.
    This process can take several minutes per dataset. On a cluster, it is advisable
    to first run the download and preprocessing separately on different workers
    or processes for different datasets, and launch the training script in a second time.

    Args:
        dataset_id (str): the dataset to be downloaded. Must be part of
            :attr:`GenDGRLExperienceReplay.available_datasets`.
        batch_size (int, optional): Batch-size used during sampling. Can be overridden by
            `data.sample(batch_size)` if necessary.

    Keyword Args:
        root (Path or str, optional): The :class:`~torchrl.data.datasets.GenDGRLExperienceReplay`
            dataset root directory.
            The actual dataset memory-mapped files will be saved under
            `<root>/<dataset_id>`. If none is provided, it defaults to
            `~/.cache/torchrl/atari`.gen_dgrl`.
        download (bool or str, optional): Whether the dataset should be downloaded if
            not found. Defaults to ``True``. Download can also be passed as ``"force"``,
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

    Attributes:
        available_datasets: a list of accepted entries to be downloaded. These
            names correspond to the directory path in the huggingface dataset
            repository. If possible, the list will be dynamically retrieved from
            huggingface. If no internet connection is available, it a cached
            version will be used.

    Examples:
        >>> import torch
        >>> torch.manual_seed(0)
        >>> from torchrl.data.datasets import GenDGRLExperienceReplay
        >>> d = GenDGRLExperienceReplay("bigfish-1M_E", batch_size=32)
        >>> for batch in d:
        ...     break
        >>> print(batch)

    """

    BASE_URL = "https://dl.fbaipublicfiles.com/DGRL/Procgen/Datasets/Compressed"
    # number of files extracted at a time
    _PROCESS_NPY_BATCH = 32
    split_trajs: bool = False

    @_classproperty
    def available_datasets(cls):
        datasets = [
            "bigfish",
            "bossfight",
            "caveflyer",
            "chaser",
            "climber",
            "coinrun",
            "dodgeball",
            "fruitbot",
            "heist",
            "jumper",
            "leaper",
            "maze",
            "miner",
            "ninja",
            "plunder",
            "starpilot",
        ]
        categories = [
            "1M_E",
            "1M_S",
            "10M",
            "25M",
            "level_1_E",
            "level_1_S",
            "level_40_E",
            "level_40_S",
        ]

        return ["-".join((ds, cat)) for cat in categories for ds in datasets]

    def __init__(
        self,
        dataset_id: str,
        batch_size: int = None,
        *,
        download: bool = True,
        root: str | None = None,
        **kwargs,
    ):
        self.dataset_id = dataset_id
        try:
            dataset, category = dataset_id.split("-")
        except Exception:
            category = dataset_id
            dataset = None
        self._dataset_name = dataset
        self._category_name = category
        if root is None:
            root = _get_root_dir("gen_dgrl")
            os.makedirs(root, exist_ok=True)
        self.root = root
        if download == "force" or (download and not self._is_downloaded()):
            if download == "force" and os.path.exists(self.data_path_root):
                shutil.rmtree(self.data_path_root)
            storage = TensorStorage(self._download_and_preproc())
        else:
            storage = TensorStorage(TensorDict.load_memmap(self.data_path_root))
        super().__init__(storage=storage, batch_size=batch_size, **kwargs)

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

    def _download_and_preproc(self):
        dataset, category = self._dataset_name, self._category_name
        link = self._build_urls_with_category_name(dataset, category)
        data_link = (category, self._fetch_file_name_from_link(link), link)

        with tempfile.TemporaryDirectory() as tmpdir:
            self._download_category_file(
                tmpdir, skip_downloaded_files=True, link=data_link
            )
            return self._unpack_category_file(
                tmpdir, clear_archive=True, link=data_link, category_name=category
            )

    @classmethod
    def _build_urls_with_category_name(
        cls, dataset, category_name: str
    ) -> tp.List[str]:
        path = [cls.BASE_URL, cls._convert_category_name(category_name)]
        path += [f"{dataset}.tar.xz"]
        return os.path.join(*path)

    @staticmethod
    def _convert_category_name(category_name: str) -> str:
        if category_name == "1M_E":
            return "1M/level_200/expert"
        elif category_name == "1M_S":
            return "1M/level_200/suboptimal"
        elif category_name == "10M":
            return "10M/level_200/expert"
        elif category_name == "25M":
            return "25M/level_200/expert"
        elif category_name == "level_1_S":
            return "100k/level_1/suboptimal"
        elif category_name == "level_40_S":
            return "100k/level_40/suboptimal"
        elif category_name == "level_1_E":
            return "100k/level_1/expert"
        elif category_name == "level_40_E":
            return "100k/level_40/expert"
        else:
            raise ValueError(f"Unrecognized category name {category_name}!")

    @staticmethod
    def _fetch_file_name_from_link(url: str) -> str:
        return os.path.split(url)[-1]

    @classmethod
    def _get_category_len(cls, category_name):
        if "1M" in category_name:
            return 1_000_000
        if "10M" in category_name:
            return 10_000_000
        if "25M" in category_name:
            return 25_000_000
        return 100_000

    def _unpack_category_file(
        self,
        download_folder: str,
        clear_archive: bool,
        category_name,
        link: str,
        batch=None,
    ):
        if batch is None:
            batch = self._PROCESS_NPY_BATCH
        _, file_name, _ = link
        file_path = os.path.join(download_folder, file_name)
        torchrl_logger.info(
            f"Unpacking dataset file {file_path} ({file_name}) to {download_folder}."
        )
        idx = 0
        td_memmap = None
        dataset_len = self._get_category_len(category_name)
        if _has_tqdm:
            from tqdm import tqdm

            pbar = tqdm(total=dataset_len)
        else:
            pbar = None
        mode = "r:xz" if str(file_path).endswith("xz") else "r"
        full = False
        with tarfile.open(file_path, mode) as tar:
            members = list(tar.getmembers())
            for i in range(0, len(members), batch):
                if full:
                    break
                submembers = [
                    member for member in members[i : i + batch] if member.isfile()
                ]
                for member in submembers:
                    if pbar is not None:
                        pbar.set_description(member.name)
                    npybuffer = tar.extractfile(member=member)
                    # npyfile = Path(download_folder) / member.name
                    npfile = np.load(npybuffer, allow_pickle=True)
                    td = TensorDict.from_dict(npfile.tolist(), auto_batch_size=True)
                    td.set("observations", td.get("observations").to(torch.uint8))
                    td.set(("next", "observation"), td.get("observations")[1:])
                    td.set("observations", td.get("observations")[:-1])
                    td.rename_key_("observations", "observation")
                    td.rename_key_("dones", ("next", "done"))
                    td.rename_key_("actions", "action")
                    td.rename_key_("rewards", ("next", "reward"))
                    td.set(
                        ("next", "done"), td.get(("next", "done")).bool().unsqueeze(-1)
                    )
                    td.set(
                        ("next", "truncated"),
                        torch.zeros_like(td.get(("next", "done"))),
                    )
                    td.set(("next", "terminated"), td.get(("next", "done")))

                    td.set(
                        "terminated", torch.zeros_like(td.get(("next", "terminated")))
                    )
                    td.set("done", torch.zeros_like(td.get(("next", "done"))))
                    td.set("truncated", torch.zeros_like(td.get(("next", "truncated"))))

                    td.batch_size = td.get("observation").shape[:1]
                    if td_memmap is None:
                        td_memmap = (
                            td[0]
                            .expand(dataset_len)
                            .memmap_like(self.data_path_root, num_threads=16)
                        )
                    idx_end = idx + td.shape[0]
                    idx_end = min(idx_end, td_memmap.shape[0])
                    if pbar is not None:
                        pbar.update(td.shape[0])
                    length = idx_end - idx
                    if length > 0:
                        if length != td.shape[0]:
                            td_memmap[idx:idx_end] = td[:length]
                        else:
                            td_memmap[idx:idx_end] = td
                    else:
                        full = True
                    idx = idx_end
        if clear_archive:
            os.remove(file_path)
        return td_memmap

    @classmethod
    def _download_category_file(
        cls,
        download_folder: str,
        skip_downloaded_files: bool,
        link: str,
    ):
        _, file_name, url = link
        file_path = os.path.join(download_folder, file_name)

        if skip_downloaded_files and os.path.isfile(file_path):
            torchrl_logger.info(f"Skipping {file_path}, already downloaded!")
            return file_name, True

        in_progress_folder = os.path.join(download_folder, "_in_progress")
        os.makedirs(in_progress_folder, exist_ok=True)
        in_progress_file_path = os.path.join(in_progress_folder, file_name)

        torchrl_logger.info(
            f"Downloading dataset file {file_name} ({url}) to {in_progress_file_path}."
        )
        cls._download_with_progress_bar(url, in_progress_file_path)

        os.rename(in_progress_file_path, file_path)
        return file_name, True

    @classmethod
    def _download_with_progress_bar(cls, url: str, file_path: str):
        # taken from https://stackoverflow.com/a/62113293/986477
        if not _has_requests:
            raise ImportError(
                "The requests package is required for Gen-DGRL dataset download."
            )
        import requests

        resp = requests.get(url, stream=True)
        total = int(resp.headers.get("content-length", 0))
        if _has_tqdm:
            from tqdm import tqdm

            pbar = tqdm(
                desc=file_path,
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            )
        else:
            pbar = None
        with open(file_path, "wb") as file:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                if pbar is not None:
                    pbar.update(size)
