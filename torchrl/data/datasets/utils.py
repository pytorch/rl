# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import os

from .common import BaseDatasetExperienceReplay

_SUPPORTED_PREFIXES = ("minari", "d4rl")


def _get_root_dir(dataset: str):
    return os.path.join(os.path.expanduser("~"), ".cache", "torchrl", dataset)


def load_dataset(dataset_id: str, **kwargs) -> BaseDatasetExperienceReplay:
    """Parse a dataset ID string and return the appropriate dataset object.

    Supported prefixes:

    - ``"minari:<id>"`` → :class:`~torchrl.data.datasets.MinariExperienceReplay`
    - ``"d4rl:<id>"``   → :class:`~torchrl.data.datasets.D4RLExperienceReplay`

    Args:
        dataset_id (str): a prefixed dataset identifier, e.g.
            ``"minari:mujoco/hopper/expert-v0"`` or
            ``"d4rl:halfcheetah-medium-v2"``.
        **kwargs: forwarded to the dataset constructor.

    Returns:
        BaseDatasetExperienceReplay: the constructed dataset object.

    Examples:
        >>> ds = load_dataset("d4rl:halfcheetah-medium-v2", batch_size=256)
        >>> ds = load_dataset("minari:mujoco/hopper/expert-v0", split_trajs=True)
    """
    if ":" not in dataset_id:
        raise ValueError(
            f"dataset_id must be prefixed with a source identifier "
            f"(e.g. 'minari:...' or 'd4rl:...'). Got: {dataset_id!r}. "
            f"Supported prefixes: {_SUPPORTED_PREFIXES}."
        )

    prefix, name = dataset_id.split(":", 1)
    prefix = prefix.lower()

    if prefix == "minari":
        try:
            from torchrl.data.datasets.minari_data import MinariExperienceReplay
        except ImportError as e:
            raise ImportError(
                "minari is required to load Minari datasets. "
                "Install it with: pip install minari"
            ) from e
        return MinariExperienceReplay(dataset_id=name, **kwargs)

    if prefix == "d4rl":
        try:
            from torchrl.data.datasets.d4rl import D4RLExperienceReplay
        except ImportError as e:
            raise ImportError(
                "d4rl is required to load D4RL datasets. "
                "Install it with: pip install d4rl"
            ) from e
        return D4RLExperienceReplay(dataset_id=name, **kwargs)

    raise ValueError(
        f"Unknown dataset source {prefix!r}. "
        f"Supported prefixes: {_SUPPORTED_PREFIXES}."
    )
