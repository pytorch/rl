# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import numpy as np
from tensordict.tensordict import TensorDict

from torchrl.data.replay_buffers import (
    LazyMemmapStorage,
    Sampler,
    SamplerWithoutReplacement,
    TensorDictReplayBuffer,
    Writer,
)


class OpenMLExperienceReplay(TensorDictReplayBuffer):
    """An experience replay for OpenML data.

    This class provides an easy entry point for public datasets.
    See "Dua, D. and Graff, C. (2017) UCI Machine Learning Repository. http://archive.ics.uci.edu/ml"

    The data is accessed via scikit-learn. Make sure sklearn and pandas are
    installed before retrieving the data:

    .. code-block::

      $ pip install scikit-learn pandas -U

    Args:
        name (str): the following datasets are supported:
            ``"adult_num"``, ``"adult_onehot"``, ``"mushroom_num"``, ``"mushroom_onehot"``,
            ``"covertype"``, ``"shuttle"`` and ``"magic"``.
        batch_size (int): the batch size to use during sampling.
        sampler (Sampler, optional): the sampler to be used. If none is provided
            a default RandomSampler() will be used.
        writer (Writer, optional): the writer to be used. If none is provided
            a default RoundRobinWriter() will be used.
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s)/outputs.  Used when using batched
            loading from a map-style dataset.
        pin_memory (bool): whether pin_memory() should be called on the rb
            samples.
        prefetch (int, optional): number of next batches to be prefetched
            using multithreading.
        transform (Transform, optional): Transform to be executed when sample() is called.
            To chain transforms use the :obj:`Compose` class.

    """

    def __init__(
        self,
        name: str,
        batch_size: int,
        sampler: Optional[Sampler] = None,
        writer: Optional[Writer] = None,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        prefetch: Optional[int] = None,
        transform: Optional["Transform"] = None,  # noqa-F821
    ):

        if sampler is None:
            sampler = SamplerWithoutReplacement()

        dataset = self._get_data(
            name,
        )
        self.max_outcome_val = dataset["y"].max().item()

        storage = LazyMemmapStorage(dataset.shape[0])
        super().__init__(
            batch_size=batch_size,
            storage=storage,
            sampler=sampler,
            writer=writer,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch=prefetch,
            transform=transform,
        )
        self.extend(dataset)

    @classmethod
    def _get_data(cls, dataset_name):
        try:
            import pandas  # noqa: F401
            from sklearn.datasets import fetch_openml
            from sklearn.preprocessing import (
                LabelEncoder,
                OneHotEncoder,
                StandardScaler,
            )
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Make sure scikit-learn and pandas are installed before "
                f"creating a {cls.__name__} instance."
            )
        if dataset_name in ["adult_num", "adult_onehot"]:
            X, y = fetch_openml("adult", version=1, return_X_y=True)
            is_NaN = X.isna()
            row_has_NaN = is_NaN.any(axis=1)
            X = X[~row_has_NaN]
            # y = y[~row_has_NaN]
            y = X["occupation"]
            X = X.drop(["occupation"], axis=1)
            cat_ix = X.select_dtypes(include=["category"]).columns
            num_ix = X.select_dtypes(include=["int64", "float64"]).columns
            encoder = LabelEncoder()
            # now apply the transformation to all the columns:
            for col in cat_ix:
                X[col] = encoder.fit_transform(X[col])
            y = encoder.fit_transform(y)
            if dataset_name == "adult_onehot":
                cat_features = OneHotEncoder(sparse=False).fit_transform(X[cat_ix])
                num_features = StandardScaler().fit_transform(X[num_ix])
                X = np.concatenate((num_features, cat_features), axis=1)
            else:
                X = StandardScaler().fit_transform(X)
        elif dataset_name in ["mushroom_num", "mushroom_onehot"]:
            X, y = fetch_openml("mushroom", version=1, return_X_y=True)
            encoder = LabelEncoder()
            # now apply the transformation to all the columns:
            for col in X.columns:
                X[col] = encoder.fit_transform(X[col])
            # X = X.drop(["veil-type"],axis=1)
            y = encoder.fit_transform(y)
            if dataset_name == "mushroom_onehot":
                X = OneHotEncoder(sparse=False).fit_transform(X)
            else:
                X = StandardScaler().fit_transform(X)
        elif dataset_name == "covertype":
            # https://www.openml.org/d/150
            # there are some 0/1 features -> consider just numeric
            X, y = fetch_openml("covertype", version=3, return_X_y=True)
            X = StandardScaler().fit_transform(X)
            y = LabelEncoder().fit_transform(y)
        elif dataset_name == "shuttle":
            # https://www.openml.org/d/40685
            # all numeric, no missing values
            X, y = fetch_openml("shuttle", version=1, return_X_y=True)
            X = StandardScaler().fit_transform(X)
            y = LabelEncoder().fit_transform(y)
        elif dataset_name == "magic":
            # https://www.openml.org/d/1120
            # all numeric, no missing values
            X, y = fetch_openml("MagicTelescope", version=1, return_X_y=True)
            X = StandardScaler().fit_transform(X)
            y = LabelEncoder().fit_transform(y)
        else:
            raise RuntimeError("Dataset does not exist")
        return TensorDict({"X": X, "y": y}, X.shape[:1])
