# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import collections
import math
import time
import typing
from typing import Optional, Type, Tuple
from warnings import warn

import numpy as np
from torch import multiprocessing as mp

from ._extension import _init_extension

try:
    from .version import __version__
except ImportError:
    __version__ = None

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
    def print(prefix=None):
        keys = list(timeit._REG)
        keys.sort()
        for name in keys:
            strings = []
            if prefix:
                strings.append(prefix)
            strings.append(
                f"{name} took {timeit._REG[name][0] * 1000:4.4} msec (total = {timeit._REG[name][1]} sec)"
            )
            print(" -- ".join(strings))


def _check_for_faulty_process(processes):
    terminate = False
    for p in processes:
        if not p.is_alive():
            terminate = True
            for _p in processes:
                if _p.is_alive():
                    _p.terminate()
        if terminate:
            break
    if terminate:
        raise RuntimeError(
            "At least one process failed. Check for more infos in the log."
        )


def seed_generator(seed):
    max_seed_val = (
        2 ** 32 - 1
    )  # https://discuss.pytorch.org/t/what-is-the-max-seed-you-can-set-up/145688
    rng = np.random.default_rng(seed)
    seed = int.from_bytes(rng.bytes(8), "big")
    return seed % max_seed_val


class KeyDependentDefaultDict(collections.defaultdict):
    def __init__(self, fun=lambda x: x):
        self.fun = fun
        super().__init__()

    def __missing__(self, key):
        value = self.fun(key)
        return value


def prod(sequence):
    if hasattr(math, "prod"):
        return math.prod(sequence)
    else:
        return int(np.prod(sequence))
