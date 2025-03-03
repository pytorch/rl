# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import weakref
from warnings import warn

import torch

from tensordict import set_lazy_legacy

from torch import multiprocessing as mp
from torch.distributions.transforms import _InverseTransform, ComposeTransform

set_lazy_legacy(False).set()

if torch.cuda.device_count() > 1:
    n = torch.cuda.device_count() - 1
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(1 + (os.getpid() % n))

from ._extension import _init_extension


try:
    from .version import __version__
except ImportError:
    __version__ = None

try:
    from torch.compiler import is_dynamo_compiling
except ImportError:
    from torch._dynamo import is_compiling as is_dynamo_compiling

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


from torchrl._utils import (
    auto_unwrap_transformed_env,
    compile_with_warmup,
    implement_for,
    set_auto_unwrap_transformed_env,
    timeit,
)

# Filter warnings in subprocesses: True by default given the multiple optional
# deps of the library. This can be turned on via `torchrl.filter_warnings_subprocess = False`.
filter_warnings_subprocess = True

_THREAD_POOL_INIT = torch.get_num_threads()


# monkey-patch dist transforms until https://github.com/pytorch/pytorch/pull/135001/ finds a home
@property
def _inv(self):
    """Patched version of Transform.inv.

    Returns the inverse :class:`Transform` of this transform.

    This should satisfy ``t.inv.inv is t``.
    """
    inv = None
    if self._inv is not None:
        inv = self._inv()
    if inv is None:
        inv = _InverseTransform(self)
        if not is_dynamo_compiling():
            self._inv = weakref.ref(inv)
    return inv


torch.distributions.transforms.Transform.inv = _inv


@property
def _inv(self):
    inv = None
    if self._inv is not None:
        inv = self._inv()
    if inv is None:
        inv = ComposeTransform([p.inv for p in reversed(self.parts)])
        if not is_dynamo_compiling():
            self._inv = weakref.ref(inv)
            inv._inv = weakref.ref(self)
        else:
            # We need inv.inv to be equal to self, but weakref can cause a graph break
            inv._inv = lambda out=self: out

    return inv


ComposeTransform.inv = _inv

__all__ = [
    "auto_unwrap_transformed_env",
    "compile_with_warmup",
    "implement_for",
    "set_auto_unwrap_transformed_env",
    "timeit",
]
