# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings


def get_primers_from_module(module):
    """Get all tensordict primers from all submodules of a module."""
    primers = []

    def make_primers(submodule):
        if hasattr(submodule, "make_tensordict_primer"):
            primers.append(submodule.make_tensordict_primer())

    module.apply(make_primers)
    if not primers:
        raise warnings.warn("No primers found in the module.")
    elif len(primers) == 1:
        return primers[0]
    else:
        from torchrl.envs.transforms import Compose

        return Compose(primers)
