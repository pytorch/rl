# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os


def _get_root_dir(dataset: str):
    return os.path.join(os.path.expanduser("~"), ".cache", "torchrl", dataset)
