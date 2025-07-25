# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import warnings

from packaging.version import parse

try:
    from .version import __version__, pytorch_version
except ImportError:
    __version__ = None
    pytorch_version = "unknown"


def is_module_available(*modules: str) -> bool:
    """Returns if a top-level module with :attr:`name` exists *without** importing it.

    This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).
    """
    return all(importlib.util.find_spec(m) is not None for m in modules)


def _init_extension():
    if not is_module_available("torchrl._torchrl"):
        warnings.warn("torchrl C++ extension is not available.")
        return


def _is_nightly(version):
    if version is None:
        return True
    parsed_version = parse(version)
    return parsed_version.local is not None


if _is_nightly(__version__):
    EXTENSION_WARNING = (
        "Failed to import torchrl C++ binaries. Some modules (eg, prioritized replay buffers) may not work with your installation. "
        "You seem to be using the nightly version of TorchRL. If this is a local install, there might be an issue with "
        "the local installation. Here are some tips to debug this:\n"
        " - make sure ninja and cmake were installed\n"
        " - make sure you ran `python setup.py clean && python setup.py develop` and that no error was raised\n"
        " - make sure the version of PyTorch you are using matches the one that was present in your virtual env during "
        f"setup. This package was built with PyTorch {pytorch_version}."
    )

else:
    EXTENSION_WARNING = (
        "Failed to import torchrl C++ binaries. Some modules (eg, prioritized replay buffers) may not work with your installation. "
        "This is likely due to a discrepancy between your package version and the PyTorch version. "
        "TorchRL does not tightly pin PyTorch versions to give users freedom, but the trade-off is that C++ extensions like "
        "prioritized replay buffers can only be used with the PyTorch version they were built against. "
        f"This package was built with PyTorch {pytorch_version}. "
        "Workarounds include: (1) upgrading/downgrading PyTorch or TorchRL to compatible versions, "
        "or (2) making a local install using `pip install git+https://github.com/pytorch/rl.git@<version>`."
    )
