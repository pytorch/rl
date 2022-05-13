# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import time
from warnings import warn

from torch import multiprocessing as mp

from ._extension import _init_extension

__version__ = "0.1"

_init_extension()

# if not HAS_OPS:
#     print("could not load C++ libraries")

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


class timeit:
    """
    A dirty but easy to use decorator for profiling code
    """

    _REG = {}

    def __init__(self, name):
        self.name = name

    def __call__(self, fn):
        def decorated_fn(*args, **kwargs):
            with self:
                out = fn(*args, **kwargs)
                return out

        return decorated_fn

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        t = time.time() - self.t0
        self._REG.setdefault(self.name, [0.0, 0.0, 0])

        count = self._REG[self.name][1]
        self._REG[self.name][0] = (self._REG[self.name][0] * count + t) / (count + 1)
        self._REG[self.name][1] = self._REG[self.name][1] + t
        self._REG[self.name][2] = count + 1

    @staticmethod
    def print():
        keys = list(timeit._REG)
        keys.sort()
        for name in keys:
            print(f"{name} took {timeit._REG[name][0] * 1000:4.4} msec (total = {timeit._REG[name][1]} sec)")
