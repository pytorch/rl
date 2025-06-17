# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from tensordict import NestedKey, TensorDictBase
from tensordict.nn import TensorDictModuleBase, TensorDictSequential
from torch import distributions as D
from torch.distributions import Categorical
from torchrl.modules import MaskedCategorical


class CategoricalSequential(TensorDictModuleBase):
    """A ProbabilisticTensorDictSequential subclass meant to work with LLMs.

    .. seealso:: :class:`~tensordict.nn.ProbabilisticTensorDictSequential` class.

    """

    generate: bool

    def get_dist(
        self,
        tensordict: TensorDictBase,
        tensordict_out: TensorDictBase | None = None,
        as_padded_tensor: bool | None = None,
        as_nested_tensor: bool | None = None,
        padding_value: float | None = None,
        padding_side: str = "right",
        layout: torch.layout | None = None,
        **kwargs,
    ) -> D.Distribution:
        td_out = self(tensordict.copy())
        # By default, pad and use masked categorical
        if as_padded_tensor is None:
            as_padded_tensor = as_nested_tensor is not True
            if padding_value is None:
                padding_value = 0.0
        if as_nested_tensor is None:
            as_nested_tensor = False
        logits = td_out.get(
            "logits",
            as_padded_tensor=as_padded_tensor,
            as_nested_tensor=as_nested_tensor,
            padding_value=padding_value,
            padding_side=padding_side,
            layout=layout,
        )
        if as_padded_tensor:
            # We can use MaskedCategorical
            dist = MaskedCategorical(
                logits=logits,
                mask=logits != padding_value,
                use_cross_entropy=True,
            )
            return dist
        return Categorical(logits)

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
