# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
from torchrl._utils import KeyDependentDefaultDict
from tensordict import TensorDict, PersistentTensorDict
import tempfile
from torchrl._utils import print_directory_tree
from collections import defaultdict
import numpy as np
import json
import datasets
from huggingface_hub import hf_hub_download, HfApi
from pathlib import Path
from torchrl.data import TensorDictReplayBuffer

THIS_DIR = pathlib.Path(__file__).parent

class VD4RLExperienceReplay(TensorDictReplayBuffer):
    def __init__(self):
        ...

    @classmethod
    def _parse_datasets(cls):
        dataset = HfApi().dataset_info("conglu/vd4rl")
        sibs = defaultdict(list)
        for sib in dataset.siblings:
                if sib.rfilename.endswith("npz") or sib.rfilename.endswith("hdf5"):
                    path = Path(sib.rfilename)
                    sibs[path.parent].append(path)
        return sibs

    @classmethod
    def _download_and_preproc(cls, dataset_id):
        path = None
        files = []
        with tempfile.TemporaryDirectory() as datapath:
            sibs = cls._parse_datasets()
            # files = []
            total_steps = 0
            for path in sibs:
                if dataset_id not in str(path):
                    continue
                for file in sibs[path]:
                    # print(path, file)
                    local_path = hf_hub_download(
                        "conglu/vd4rl",
                        subfolder=str(path),
                        filename=str(file.parts[-1]),
                        repo_type="dataset",
                        cache_dir=str(datapath),
                    )
                    files.append(local_path)
                    # print_directory_tree(datapath)
                    if local_path.endswith("hdf5"):
                        td = PersistentTensorDict.from_h5(local_path)
                    else:
                        td = _from_npz(local_path)
                    if total_steps == 0:
                        td = td.to_tensordict()
                        cls._process_data(td)
                        td_save = td[0]
                    total_steps += td.shape[0]
            td_save = td_save.expand(total_steps).memmap_like(path)
            print(td_save)
            idx0 = 0
            idx1 = 0
            while len(files):
                local_path = files.pop(0)
                if local_path.endswith("hdf5"):
                    td = PersistentTensorDict.from_h5(local_path)
                else:
                    td = _from_npz(local_path)
                td = td.to_tensordict()
                cls._process_data(td)
                idx1 += td.shape[0]
                td_save[idx0:idx1] = td
                idx0 = idx1
            return td_save

    @classmethod
    def _process_data(cls, td: TensorDict):
        print(td)
        for name, val in list(td.items()):
            if name != _NAME_MATCH[name]:
                td.rename_key_(name, _NAME_MATCH[name])
        observation = td.get("observation")
        td.get_sub_tensordict(slice(0, -1)).set(("next", "observation"), observation[1:])
        print(td)

    @property
    def available_datasets(self):
        return self.available_datasets
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

def _from_npz(npz_path):
    npz = np.load(npz_path)
    npz_dict = {
        file: npz[file] for file in npz.files
    }
    return TensorDict.from_dict(npz_dict)

_NAME_MATCH = KeyDependentDefaultDict(lambda x: x)
_NAME_MATCH.update({
    "is_first": "is_init",
    "is_last": ("next", "done"),
    "is_terminal": ("next", "terminated"),
    "reward": ("next", "reward"),
})
