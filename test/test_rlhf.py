# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import pytest


def test_create_or_load_dataset():
    pass


def test_preproc_data():
    pass


def dataset_to_tensordict():
    pass


def test_get_dataloader():
    pass


def test_promptdata():
    pass


def test_pairwise_dataset():
    pass


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
