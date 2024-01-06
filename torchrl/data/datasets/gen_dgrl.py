# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import os
import tarfile
import tempfile
import typing as tp
from pathlib import Path

import numpy as np
import requests
import torch

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, TensorStorage
from torchrl.data.datasets.utils import _get_root_dir
from torchrl.envs.utils import _classproperty
from tqdm import tqdm


class GenDGRLExperienceReplay(TensorDictReplayBuffer):
    BASE_URL = "https://dl.fbaipublicfiles.com/DGRL/"

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
        categories = ["1M_E", "1M_S", "10M", "25M"]

        return ["-".join((ds, cat)) for ds in datasets for cat in categories] + [
            "level_1_E",
            "level_1_S",
            "level_40_E",
            "level_40_S",
        ]

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
        if download == "force" or (download and not self._downloaded):
            storage = TensorStorage(self._download_and_preproc())
        else:
            storage = TensorStorage(TensorDict.load_memmap(self.data_path_root))
        return super().__init__(storage=storage, batch_size=batch_size, **kwargs)

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
                tmpdir, True, data_link, category_name=category
            )

    @classmethod
    def _build_urls_with_category_name(
        cls, dataset, category_name: str
    ) -> tp.List[str]:
        if category_name in ["level_1_E", "level_1_S", "level_40_E", "level_40_S"]:
            return os.path.join(cls.BASE_URL, cls._convert_category_name(category_name))
        else:
            return os.path.join(
                cls.BASE_URL,
                cls._convert_category_name(category_name),
                f"{dataset}.tar.xz",
            )

    @staticmethod
    def _convert_category_name(category_name: str) -> str:
        if category_name == "1M_E":
            return "1M/expert"
        elif category_name == "1M_S":
            return "1M/suboptimal"
        elif category_name == "10M":
            return "10M"
        elif category_name == "25M":
            return "25M"
        elif category_name == "level_1_S":
            return "100k_procgen_dataset_1_suboptimal.tar"
        elif category_name == "level_40_S":
            return "100k_procgen_dataset_40_suboptimal.tar"
        elif category_name == "level_1_E":
            return "100k_procgen_dataset_1.tar"
        elif category_name == "level_40_E":
            return "100k_procgen_dataset_40.tar"
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
        batch=100,
    ):
        _, file_name, _ = link
        file_path = os.path.join(download_folder, file_name)
        print(f"Unpacking dataset file {file_path} ({file_name}) to {download_folder}.")
        idx = 0
        td_memmap = None
        dataset_len = self._get_category_len(category_name)
        pbar = tqdm(total=dataset_len)
        with tarfile.open(file_path, "r:xz") as tar:
            members = list(tar.getmembers())
            for i in range(0, len(members), batch):
                submembers = [
                    member for member in members[i : i + batch] if member.isfile()
                ]
                # for member in members:
                # Extract only regular files, not directories or special files
                # print(f"Extracting: {[member.name for member in submembers]}", idx)
                tar.extractall(
                    members=submembers, path=download_folder
                )  # Change 'output_directory' to your desired destination
                for member in submembers:
                    pbar.set_description(member.name)
                    npyfile = Path(download_folder) / member.name
                    npfile = np.load(npyfile, allow_pickle=True)
                    td = TensorDict.from_dict(npfile.tolist())
                    td.set("observations", td.get("observations").to(torch.uint8))
                    td.set(("next", "observations"), td.get("observations")[1:])
                    td.set("observations", td.get("observations")[:-1])
                    td.batch_size = td.get("observations").shape[:1]
                    td.rename_key_("observations", "observation")
                    td.rename_key_("dones", ("next", "done"))
                    td.rename_key_("actions", "action")
                    td.rename_key_("rewards", ("next", "reward"))
                    td.set(("next", "reward"), td.get(("next", "reward").unsqueeze(-1)))
                    td.set(
                        ("next", "done"), td.get(("next", "done").bool().unsqueeze(-1))
                    )
                    td.set(
                        ("next", "truncated"),
                        torch.zeros_like(td.get(("next", "done"))),
                    )
                    td.set(("next", "terminated"), td.get(("next", "done")))
                    if td_memmap is None:
                        td_memmap = (
                            td[0]
                            .expand(dataset_len)
                            .memmap_like(self.data_path_root, num_threads=16)
                        )
                    idx_end = idx + td.shape[0]
                    idx_end = min(idx_end, td.shape[0])
                    pbar.update(td.shape[0])
                    td_memmap[idx:idx_end] = td
                    idx = idx_end
                    os.remove(npyfile)

        # shutil.unpack_archive(file_path, download_folder)
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
            print(f"Skipping {file_path}, already downloaded!")
            return file_name, True

        in_progress_folder = os.path.join(download_folder, "_in_progress")
        os.makedirs(in_progress_folder, exist_ok=True)
        in_progress_file_path = os.path.join(in_progress_folder, file_name)

        print(
            f"Downloading dataset file {file_name} ({url}) to {in_progress_file_path}."
        )
        cls._download_with_progress_bar(url, in_progress_file_path)

        os.rename(in_progress_file_path, file_path)
        return file_name, True

    @classmethod
    def _download_with_progress_bar(cls, url: str, file_path: str):
        # taken from https://stackoverflow.com/a/62113293/986477
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get("content-length", 0))
        with open(file_path, "wb") as file, tqdm(
            desc=file_path,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
