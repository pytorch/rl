# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Optional, Sequence, Type, Union

from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from tensordict.nn.prototype import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
)

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules.tensordict_module.common import _forward_hook_safe_action
from torchrl.modules.tensordict_module.sequence import SafeSequential
from torchrl.modules.distributions import Delta


class SafeProbabilisticModule(
    ProbabilisticTensorDictModule,
):
    def __init__(
        self,
        in_keys: Union[str, Sequence[str], dict],
        out_keys: Union[str, Sequence[str]],
        spec: Optional[TensorSpec] = None,
        safe: bool = False,
        default_interaction_mode: str = "mode",
        distribution_class: Type = Delta,
        distribution_kwargs: Optional[dict] = None,
        return_log_prob: bool = False,
        cache_dist: bool = False,
        n_empirical_estimate: int = 1000,
    ):
        super().__init__(
            in_keys=in_keys,
            out_keys=out_keys,
            default_interaction_mode=default_interaction_mode,
            distribution_class=distribution_class,
            distribution_kwargs=distribution_kwargs,
            return_log_prob=return_log_prob,
            cache_dist=cache_dist,
            n_empirical_estimate=n_empirical_estimate,
        )

        if spec is not None and not isinstance(spec, TensorSpec):
            raise TypeError("spec must be a TensorSpec subclass")
        elif spec is not None and not isinstance(spec, CompositeSpec):
            if len(self.out_keys) > 1:
                raise RuntimeError(
                    f"got more than one out_key for the SafeModule: {self.out_keys},\nbut only one spec. "
                    "Consider using a CompositeSpec object or no spec at all."
                )
            spec = CompositeSpec(**{self.out_keys[0]: spec})
        elif spec is not None and isinstance(spec, CompositeSpec):
            if "_" in spec.keys():
                warnings.warn('got a spec with key "_": it will be ignored')
        elif spec is None:
            spec = CompositeSpec()

        if set(spec.keys()) != set(self.out_keys):
            # then assume that all the non indicated specs are None
            for key in self.out_keys:
                if key not in spec:
                    spec[key] = None

        if set(spec.keys()) != set(self.out_keys):
            raise RuntimeError(
                f"spec keys and out_keys do not match, got: {set(spec.keys())} and {set(self.out_keys)} respectively"
            )

        self._spec = spec
        self.safe = safe
        if safe:
            if spec is None or (
                isinstance(spec, CompositeSpec)
                and all(_spec is None for _spec in spec.values())
            ):
                raise RuntimeError(
                    "`SafeProbabilisticModule(spec=None, safe=True)` is not a valid configuration as the tensor "
                    "specs are not specified"
                )
            self.register_forward_hook(_forward_hook_safe_action)

    @property
    def spec(self) -> CompositeSpec:
        return self._spec

    @spec.setter
    def spec(self, spec: CompositeSpec) -> None:
        if not isinstance(spec, CompositeSpec):
            raise RuntimeError(
                f"Trying to set an object of type {type(spec)} as a tensorspec but expected a CompositeSpec instance."
            )
        self._spec = spec

    def random(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Samples a random element in the target space, irrespective of any input.

        If multiple output keys are present, only the first will be written in the input :obj:`tensordict`.

        Args:
            tensordict (TensorDictBase): tensordict where the output value should be written.

        Returns:
            the original tensordict with a new/updated value for the output key.

        """
        key0 = self.out_keys[0]
        tensordict.set(key0, self.spec.rand(tensordict.batch_size))
        return tensordict

    def random_sample(self, tensordict: TensorDictBase) -> TensorDictBase:
        """See :obj:`SafeModule.random(...)`."""
        return self.random(tensordict)


class SafeProbabilisticSequential(ProbabilisticTensorDictSequential, SafeSequential):
    def __init__(
        self,
        *modules: Union[TensorDictModule, ProbabilisticTensorDictModule],
        partial_tolerant: bool = False,
    ) -> None:
        super().__init__(*modules, partial_tolerant=partial_tolerant)
        super(ProbabilisticTensorDictSequential, self).__init__(
            *modules, partial_tolerant=partial_tolerant
        )
