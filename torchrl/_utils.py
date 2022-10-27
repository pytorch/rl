import collections
import math
import os
import time

import numpy as np


class timeit:
    """A dirty but easy to use decorator for profiling code."""

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
        val = self._REG.setdefault(self.name, [0.0, 0.0, 0])

        count = val[2]
        N = count + 1
        val[0] = val[0] * (count / N) + t / N
        val[1] += t
        val[2] = N

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

    @staticmethod
    def erase():
        for k in timeit._REG:
            timeit._REG[k] = [0.0, 0.0, 0]


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
    """A seed generator function.

    Given a seeding integer, generates a deterministic next seed to be used in a
    seeding sequence.

    Args:
        seed (int): initial seed.

    Returns: Next seed of the chain.

    """
    max_seed_val = (
        2 ** 32 - 1
    )  # https://discuss.pytorch.org/t/what-is-the-max-seed-you-can-set-up/145688
    rng = np.random.default_rng(seed)
    seed = int.from_bytes(rng.bytes(8), "big")
    return seed % max_seed_val


class KeyDependentDefaultDict(collections.defaultdict):
    """A key-dependent default dict.

    Examples:
        >>> my_dict = KeyDependentDefaultDict(lambda key: "foo_" + key)
        >>> print(my_dict["bar"])
        foo_bar
    """

    def __init__(self, fun):
        self.fun = fun
        super().__init__()

    def __missing__(self, key):
        value = self.fun(key)
        self[key] = value
        return value


def prod(sequence):
    """General prod function, that generalised usage across math and np.

    Created for multiple python versions compatibility).

    """
    if hasattr(math, "prod"):
        return math.prod(sequence)
    else:
        return int(np.prod(sequence))


def get_binary_env_var(key):
    """Parses and returns the binary enironment variable value.

    If not present in environment, it is considered `False`.

    Args:
        key (str): name of the environment variable.
    """
    val = os.environ.get(key, "False")
    if val in ("0", "False", "false"):
        val = False
    elif val in ("1", "True", "true"):
        val = True
    else:
        raise ValueError(
            f"Environment variable {key} should be in 'True', 'False', '0' or '1'. "
            f"Got {val} instead."
        )
    return val


class _Dynamic_CKPT_BACKEND:
    """Allows CKPT_BACKEND to be changed on-the-fly."""

    backends = ["torch", "torchsnapshot"]

    def _get_backend(self):
        backend = os.environ.get("CKPT_BACKEND", "torchsnapshot")
        if backend == "torchsnapshot":
            try:
                import torchsnapshot  # noqa: F401

                _has_ts = True
            except ImportError:
                _has_ts = False
            if not _has_ts:
                raise ImportError(
                    f"torchsnapshot not found, but the backend points to this library. Consider installing torchsnapshot or choose another backend (available backends: {self.backends})"
                )
        return backend

    def __getattr__(self, item):
        return getattr(self._get_backend(), item)

    def __eq__(self, other):
        return self._get_backend() == other

    def __ne__(self, other):
        return self._get_backend() != other

    def __repr__(self):
        return self._get_backend()


_CKPT_BACKEND = _Dynamic_CKPT_BACKEND()
