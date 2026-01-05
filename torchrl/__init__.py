# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import warnings
import weakref
from warnings import warn

import torch

# Silence noisy dependency warning triggered at import time on older torch stacks.
# (Emitted by tensordict when registering pytree nodes.)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"torch\.utils\._pytree\._register_pytree_node is deprecated\.",
)

from tensordict import set_lazy_legacy  # noqa: E402

from torch import multiprocessing as mp  # noqa: E402
from torch.distributions.transforms import (  # noqa: E402
    _InverseTransform,
    ComposeTransform,
)

torch._C._log_api_usage_once("torchrl")

set_lazy_legacy(False).set()

from ._extension import _init_extension  # noqa: E402

__version__ = None  # type: ignore
try:
    try:
        from importlib.metadata import version as _dist_version
    except ImportError:  # pragma: no cover
        from importlib_metadata import version as _dist_version  # type: ignore

    __version__ = _dist_version("torchrl")
except Exception:
    try:
        from ._version import __version__
    except Exception:
        try:
            from .version import __version__
        except Exception:
            __version__ = None  # type: ignore

try:
    from torch.compiler import is_dynamo_compiling
except ImportError:
    from torch._dynamo import is_compiling as is_dynamo_compiling

_init_extension()

from torchrl._utils import (  # noqa: E402
    _get_default_mp_start_method,
    auto_unwrap_transformed_env,
    compile_with_warmup,
    get_ray_default_runtime_env,
    implement_for,
    logger,
    merge_ray_runtime_env,
    set_auto_unwrap_transformed_env,
    timeit,
)

logger = logger

# TorchRL's multiprocessing default.
_preferred_start_method = _get_default_mp_start_method()
if _preferred_start_method == "spawn":
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
    "get_ray_default_runtime_env",
    "implement_for",
    "merge_ray_runtime_env",
    "set_auto_unwrap_transformed_env",
    "timeit",
    "logger",
    "logger",
]
