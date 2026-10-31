# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import importlib
import sys

import torch
from packaging import version
from packaging.version import parse
from tensordict import TensorDict

from torchrl.data import ReplayBuffer, TensorDictReplayBuffer


class _ReplayRowsExporter:
    def to_external(self, dataset):
        return [dataset[index].clone() for index in range(len(dataset))]


def _episode_data(obs, episode, step, *, device="cpu"):
    return TensorDict(
        {
            "obs": torch.tensor(obs, device=device),
            "episode": torch.tensor(episode, device=device),
            "step": torch.tensor(step, device=device),
        },
        [len(obs)],
        device=device,
    )


OLD_TORCH = parse(torch.__version__) < parse("2.0.0")
_has_tv = importlib.util.find_spec("torchvision") is not None
_has_gym = importlib.util.find_spec("gym") is not None
_has_snapshot = importlib.util.find_spec("torchsnapshot") is not None
_os_is_windows = sys.platform == "win32"
_has_transformers = importlib.util.find_spec("transformers") is not None
_has_ray = importlib.util.find_spec("ray") is not None
_has_zstandard = importlib.util.find_spec("zstandard") is not None
_has_h5py = importlib.util.find_spec("h5py") is not None

TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)

torch_2_3 = version.parse(
    ".".join([str(s) for s in version.parse(str(torch.__version__)).release])
) >= version.parse("2.3.0")

ReplayBufferRNG = functools.partial(ReplayBuffer, generator=torch.Generator())
TensorDictReplayBufferRNG = functools.partial(
    TensorDictReplayBuffer, generator=torch.Generator()
)
