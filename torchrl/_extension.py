# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import warnings


def is_module_available(*modules: str) -> bool:
    r"""Returns if a top-level module with :attr:`name` exists *without** importing it.

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


EXTENSION_WARNING = (
    "Failed to import torchrl C++ binaries. Some modules (eg, prioritized replay buffers) may not work with your installation. "
    "If you installed TorchRL from PyPI, please report the bug on TorchRL github. "
    "If you installed TorchRL locally and/or in development mode, check that you have all the required compiling packages."
)
