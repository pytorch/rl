# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib
import os
from collections.abc import Callable

from .common import BaseDatasetExperienceReplay

DatasetFactory = Callable[..., BaseDatasetExperienceReplay]
DatasetFactorySpec = str | DatasetFactory

_DATASET_REGISTRY: dict[str, DatasetFactorySpec] = {}


def _get_root_dir(dataset: str):
    return os.path.join(os.path.expanduser("~"), ".cache", "torchrl", dataset)


def _normalize_prefix(prefix: str) -> str:
    prefix = prefix.lower().strip()
    if not prefix or ":" in prefix:
        raise ValueError(
            "Dataset source prefixes must be non-empty strings without ':'. "
            f"Got {prefix!r}."
        )
    return prefix


def _supported_prefixes() -> tuple[str, ...]:
    return tuple(sorted(_DATASET_REGISTRY))


def register_dataset(
    prefix: str,
    dataset: DatasetFactorySpec,
    *,
    replace: bool = False,
) -> None:
    """Register a dataset factory for :func:`load_dataset`.

    The registered prefix can then be used in strings of the form
    ``"<prefix>:<dataset-id>"``. The dataset factory is called as
    ``dataset(dataset_id, **kwargs)``.

    Args:
        prefix (str): source prefix used before the ``":"`` separator.
        dataset (Callable or str): dataset factory, or an import string of the
            form ``"module:attribute"`` resolved lazily when the prefix is used.
        replace (bool, optional): if ``True``, replace an existing registration.
            Defaults to ``False``.

    Examples:
        >>> from torchrl.data.datasets import register_dataset, load_dataset
        >>> class ToyDataset:
        ...     def __init__(self, dataset_id, **kwargs):
        ...         self.dataset_id = dataset_id
        >>> register_dataset("toy", ToyDataset, replace=True)
        >>> load_dataset("toy:example").dataset_id
        'example'
    """
    prefix = _normalize_prefix(prefix)
    if not isinstance(dataset, str) and not callable(dataset):
        raise TypeError(
            "dataset must be a callable dataset factory or a 'module:attribute' "
            f"string, got {type(dataset).__name__}."
        )
    if isinstance(dataset, str):
        if ":" not in dataset:
            raise ValueError(
                "String dataset factories must use the 'module:attribute' format."
            )
        module_name, attr_name = dataset.split(":", 1)
        if not module_name or not attr_name:
            raise ValueError(
                "String dataset factories must use the 'module:attribute' format."
            )
    if prefix in _DATASET_REGISTRY and not replace:
        raise KeyError(
            f"Dataset source {prefix!r} is already registered. "
            "Pass replace=True to overwrite it."
        )
    _DATASET_REGISTRY[prefix] = dataset


def _get_dataset_factory(prefix: str) -> DatasetFactory:
    prefix = _normalize_prefix(prefix)
    try:
        dataset = _DATASET_REGISTRY[prefix]
    except KeyError as err:
        raise ValueError(
            f"Unknown dataset source {prefix!r}. "
            f"Supported prefixes: {_supported_prefixes()}."
        ) from err
    if isinstance(dataset, str):
        module_name, attr_name = dataset.split(":", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    return dataset


def load_dataset(dataset_id: str, **kwargs) -> BaseDatasetExperienceReplay:
    """Parse a dataset ID string and return the registered dataset object.

    Built-in prefixes include ``"atari"``, ``"atari_dqn"``, ``"d4rl"``,
    ``"gen_dgrl"``, ``"lerobot"``, ``"minari"``, ``"openml"``, ``"openx"``,
    ``"roboset"``, and ``"vd4rl"``. Additional prefixes can be installed with
    :func:`register_dataset`.

    Args:
        dataset_id (str): a prefixed dataset identifier, e.g.
            ``"minari:mujoco/hopper/expert-v0"`` or
            ``"d4rl:halfcheetah-medium-v2"``.
        **kwargs: forwarded to the dataset constructor.

    Returns:
        BaseDatasetExperienceReplay: the constructed dataset object.

    Examples:
        >>> from torchrl.data.datasets import register_dataset
        >>> class ToyDataset:
        ...     def __init__(self, dataset_id, **kwargs):
        ...         self.dataset_id = dataset_id
        >>> register_dataset("toy", ToyDataset, replace=True)
        >>> load_dataset("toy:example").dataset_id
        'example'
    """
    if ":" not in dataset_id:
        raise ValueError(
            f"dataset_id must be prefixed with a source identifier "
            f"(e.g. 'minari:...' or 'd4rl:...'). Got: {dataset_id!r}. "
            f"Supported prefixes: {_supported_prefixes()}."
        )

    prefix, name = dataset_id.split(":", 1)
    factory = _get_dataset_factory(prefix)
    return factory(name, **kwargs)


register_dataset(
    "atari",
    "torchrl.data.datasets.atari_dqn:AtariDQNExperienceReplay",
)
register_dataset(
    "atari_dqn",
    "torchrl.data.datasets.atari_dqn:AtariDQNExperienceReplay",
)
register_dataset("d4rl", "torchrl.data.datasets.d4rl:D4RLExperienceReplay")
register_dataset(
    "gen_dgrl",
    "torchrl.data.datasets.gen_dgrl:GenDGRLExperienceReplay",
)
register_dataset("lerobot", "torchrl.data.datasets.lerobot:LeRobotExperienceReplay")
register_dataset("minari", "torchrl.data.datasets.minari_data:MinariExperienceReplay")
register_dataset("openml", "torchrl.data.datasets.openml:OpenMLExperienceReplay")
register_dataset("openx", "torchrl.data.datasets.openx:OpenXExperienceReplay")
register_dataset("roboset", "torchrl.data.datasets.roboset:RobosetExperienceReplay")
register_dataset("vd4rl", "torchrl.data.datasets.vd4rl:VD4RLExperienceReplay")
