# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from warnings import warn

from torch import multiprocessing as mp

from ._extension import _init_extension


try:
    from .version import __version__
except ImportError:
    __version__ = None

_init_extension()

# if not HAS_OPS:
#     print("could not load C++ libraries")

_MP_START_METHOD = os.environ.get("MP_START_METHOD", "spawn")

try:
    mp.set_start_method(_MP_START_METHOD)
except RuntimeError as err:
    if str(err).startswith("context has already been set"):
        mp_start_method = mp.get_start_method()
        if mp_start_method != _MP_START_METHOD:
            warn(
                f"failed to set start method to {_MP_START_METHOD}, "
                f"and current start method for mp is {mp_start_method}."
            )


import torchrl.collectors
import torchrl.data
import torchrl.envs
import torchrl.modules
import torchrl.objectives
import torchrl.trainers
