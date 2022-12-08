# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Optional, Sequence, Type, Union

from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)
from tensordict.tensordict import TensorDictBase

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules.distributions import Delta
from torchrl.modules.tensordict_module.common import _forward_hook_safe_action
from torchrl.modules.tensordict_module.sequence import SafeSequential


class SafeProbabilisticModule(ProbabilisticTensorDictModule):
    """A :obj:``SafeProbabilisticModule`` is an :obj:``tensordict.nn.ProbabilisticTensorDictModule`` subclass that accepts a :obj:``TensorSpec`` as argument to control the output domain.

    `SafeProbabilisticModule` is a non-parametric module representing a
    probability distribution. It reads the distribution parameters from an input
    TensorDict using the specified `in_keys`. The output is sampled given some rule,
    specified by the input :obj:`default_interaction_mode` argument and the
    :obj:`interaction_mode()` global function.

    :obj:`SafeProbabilisticModule` can be used to construct the distribution
    (through the :obj:`get_dist()` method) and/or sampling from this distribution
    (through a regular :obj:`__call__()` to the module).

    A :obj:`SafeProbabilisticModule` instance has two main features:
    - It reads and writes TensorDict objects
    - It uses a real mapping R^n -> R^m to create a distribution in R^d from
    which values can be sampled or computed.

    When the :obj:`__call__` / :obj:`forward` method is called, a distribution is
    created, and a value computed (using the 'mean', 'mode', 'median' attribute or
    the 'rsample', 'sample' method). The sampling step is skipped if the supplied
    TensorDict has all of the desired key-value pairs already.

    By default, SafeProbabilisticModule distribution class is a Delta
    distribution, making SafeProbabilisticModule a simple wrapper around
    a deterministic mapping function.

    Args:
        in_keys (str or iterable of str or dict): key(s) that will be read from the
            input TensorDict and used to build the distribution. Importantly, if it's an
            iterable of string or a string, those keys must match the keywords used by
            the distribution class of interest, e.g. :obj:`"loc"` and :obj:`"scale"` for
            the Normal distribution and similar. If in_keys is a dictionary,, the keys
            are the keys of the distribution and the values are the keys in the
            tensordict that will get match to the corresponding distribution keys.
        out_keys (str or iterable of str): keys where the sampled values will be
            written. Importantly, if these keys are found in the input TensorDict, the
            sampling step will be skipped.
        spec (TensorSpec): specs of the first output tensor. Used when calling
            td_module.random() to generate random values in the target space.
        safe (bool, optional): if True, the value of the sample is checked against the
            input spec. Out-of-domain sampling can occur because of exploration policies
            or numerical under/overflow issues. As for the :obj:`spec` argument, this
            check will only occur for the distribution sample, but not the other tensors
            returned by the input module. If the sample is out of bounds, it is
            projected back onto the desired space using the `TensorSpec.project` method.
            Default is :obj:`False`.
        default_interaction_mode (str, optional): default method to be used to retrieve
            the output value. Should be one of: 'mode', 'median', 'mean' or 'random'
            (in which case the value is sampled randomly from the distribution). Default
            is 'mode'.
            Note: When a sample is drawn, the :obj:`ProbabilisticTDModule` instance will
            fist look for the interaction mode dictated by the `interaction_mode()`
            global function. If this returns `None` (its default value), then the
            `default_interaction_mode` of the `ProbabilisticTDModule` instance will be
            used. Note that DataCollector instances will use `set_interaction_mode` to
            `"random"` by default.
        distribution_class (Type, optional): a torch.distributions.Distribution class to
            be used for sampling. Default is Delta.
        distribution_kwargs (dict, optional): kwargs to be passed to the distribution.
        return_log_prob (bool, optional): if True, the log-probability of the
            distribution sample will be written in the tensordict with the key
            `'sample_log_prob'`. Default is `False`.
        cache_dist (bool, optional): EXPERIMENTAL: if True, the parameters of the
            distribution (i.e. the output of the module) will be written to the
            tensordict along with the sample. Those parameters can be used to re-compute
            the original distribution later on (e.g. to compute the divergence between
            the distribution used to sample the action and the updated distribution in
            PPO). Default is `False`.
        n_empirical_estimate (int, optional): number of samples to compute the empirical
            mean when it is not available. Default is 1000

    """

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
    """A :obj:``SafeProbabilisticSequential`` is an :obj:``tensordict.nn.ProbabilisticTensorDictSequential`` subclass that accepts a :obj:``TensorSpec`` as argument to control the output domain.

    Similarly to :obj:`TensorDictSequential`, but enforces that the final module in the
    sequence is an :obj:`ProbabilisticTensorDictModule` and also exposes ``get_dist``
    method to recover the distribution object from the ``ProbabilisticTensorDictModule``

    Args:
         modules (iterable of TensorDictModules): ordered sequence of TensorDictModule
            instances, terminating in ProbabilisticTensorDictModule, to be run
            sequentially.
         partial_tolerant (bool, optional): if True, the input tensordict can miss some
            of the input keys. If so, the only module that will be executed are those
            who can be executed given the keys that are present. Also, if the input
            tensordict is a lazy stack of tensordicts AND if partial_tolerant is
            :obj:`True` AND if the stack does not have the required keys, then
            TensorDictSequential will scan through the sub-tensordicts looking for those
            that have the required keys, if any.

    """

    def __init__(
        self,
        *modules: Union[TensorDictModule, ProbabilisticTensorDictModule],
        partial_tolerant: bool = False,
    ) -> None:
        super().__init__(*modules, partial_tolerant=partial_tolerant)
        super(ProbabilisticTensorDictSequential, self).__init__(
            *modules, partial_tolerant=partial_tolerant
        )
