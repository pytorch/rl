# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from warnings import warn

import torch

from tensordict import set_lazy_legacy

from torch import multiprocessing as mp

set_lazy_legacy(False).set()

if torch.cuda.device_count() > 1:
    n = torch.cuda.device_count() - 1
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(1 + (os.getpid() % n))

from ._extension import _init_extension


try:
    from .version import __version__
except ImportError:
    __version__ = None

_init_extension()

try:
    mp.set_start_method("spawn")
except RuntimeError as err:
    if str(err).startswith("context has already been set"):
        mp_start_method = mp.get_start_method()
        if mp_start_method != "spawn":
            warn(
                f"failed to set start method to spawn, "
                f"and current start method for mp is {mp_start_method}."
            )


import torchrl.collectors
import torchrl.data
import torchrl.envs
import torchrl.modules
import torchrl.objectives
import torchrl.trainers

# Filter warnings in subprocesses: True by default given the multiple optional
# deps of the library. This can be turned on via `torchrl.filter_warnings_subprocess = False`.
filter_warnings_subprocess = True

_THREAD_POOL_INIT = torch.get_num_threads()
