# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from tensordict import NestedKey, TensorDictBase
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    TensorDictModuleBase,
    TensorDictSequential,
)
from torch import distributions as D
from torch.distributions import Categorical


class CategoricalSequential(TensorDictModuleBase):
    """A ProbabilisticTensorDictSequential subclass meant to work with LLMs.

    .. seealso:: :class:`~tensordict.nn.ProbabilisticTensorDictSequential` class.

    """

    def get_dist(
        self,
        tensordict: TensorDictBase,
        tensordict_out: TensorDictBase | None = None,
        **kwargs,
    ) -> D.Distribution:
        td_out = self(tensordict.copy())
        return Categorical(td_out.get("logits"))

    # Sampling is taken care of by the sub-modules
    forward = TensorDictSequential.forward

    @property
    def log_prob_keys(self) -> list[NestedKey]:
        return getattr(self, "_log_prob_keys", ["log_probs"])

    @log_prob_keys.setter
    def log_prob_keys(self, value: list[NestedKey]):
        self._log_prob_keys = value

    @property
    def log_prob_key(self) -> NestedKey:
        return self.log_prob_keys[0]

    @log_prob_key.setter
    def log_prob_key(self, value: NestedKey) -> None:
        self.log_prob_keys[0] = value

    @property
    def dist_params_keys(self) -> list[NestedKey]:
        raise NotImplementedError

    @property
    def dist_sample_keys(self) -> list[NestedKey]:
        return ["tokens_response"]

    def log_prob(self, data: TensorDictBase, **get_kwargs) -> TensorDictBase:
        if not self.generate:
            data = self(data)
            return data.get(self.log_prob_key, **get_kwargs)
        raise RuntimeError("log_prob not callable when generate=True.")
