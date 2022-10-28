# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# Get relative file path
# this returns relative path from current file.
import torch.cuda
from torchrl._utils import seed_generator
from torchrl.envs import EnvBase


def get_relative_path(curr_file, *path_components):
    return os.path.join(os.path.dirname(curr_file), *path_components)


def get_available_devices():
    devices = [torch.device("cpu")]
    n_cuda = torch.cuda.device_count()
    if n_cuda > 0:
        for i in range(n_cuda):
            devices += [torch.device(f"cuda:{i}")]
    return devices


def generate_seeds(seed, repeat):
    seeds = [seed]
    for _ in range(repeat - 1):
        seed = seed_generator(seed)
        seeds.append(seed)
    return seeds


def _test_fake_tensordict(env: EnvBase):
    fake_tensordict = env.fake_tensordict().flatten_keys(".")
    real_tensordict = env.rollout(3).flatten_keys(".")

    keys1 = set(fake_tensordict.keys())
    keys2 = set(real_tensordict.keys())
    assert keys1 == keys2
    fake_tensordict = fake_tensordict.expand(3).to_tensordict()
    fake_tensordict.zero_()
    real_tensordict.zero_()
    assert (fake_tensordict == real_tensordict).all()
    for key in keys2:
        assert fake_tensordict[key].shape == real_tensordict[key].shape
