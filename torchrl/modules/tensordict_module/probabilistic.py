# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings

import torch
from tensordict import TensorDictBase, unravel_key_list
from tensordict.nn import (
    InteractionType,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)
from tensordict.utils import NestedKey
from torchrl.data.tensor_specs import Composite, TensorSpec
from torchrl.modules.distributions import Delta
from torchrl.modules.tensordict_module.common import _forward_hook_safe_action
from torchrl.modules.tensordict_module.sequence import SafeSequential


class SafeProbabilisticModule(ProbabilisticTensorDictModule):
    """:class:`tensordict.nn.ProbabilisticTensorDictModule` subclass that accepts a :class:`~torchrl.envs.TensorSpec` as an argument to control the output domain.

    `SafeProbabilisticModule` is a non-parametric module embedding a
    probability distribution constructor. It reads the distribution parameters from an input
    TensorDict using the specified `in_keys` and outputs a sample (loosely speaking) of the
    distribution.

    The output "sample" is produced given some rule, specified by the input ``default_interaction_type``
    argument and the ``interaction_type()`` global function.

    `SafeProbabilisticModule` can be used to construct the distribution
    (through the :meth:`get_dist` method) and/or sampling from this distribution
    (through a regular :meth:`__call__` to the module).

    A `SafeProbabilisticModule` instance has two main features:

    - It reads and writes from and to TensorDict objects;
    - It uses a real mapping R^n -> R^m to create a distribution in R^d from
      which values can be sampled or computed.

    When the :meth:`__call__` and :meth:`~.forward` method are called, a distribution is
    created, and a value computed (depending on the ``interaction_type`` value, 'dist.mean',
    'dist.mode', 'dist.median' attributes could be used, as well as
    the 'dist.rsample', 'dist.sample' method). The sampling step is skipped if the supplied
    TensorDict has all the desired key-value pairs already.

    By default, `SafeProbabilisticModule` distribution class is a :class:`~torchrl.modules.distributions.Delta`
    distribution, making `SafeProbabilisticModule` a simple wrapper around
    a deterministic mapping function.

    This class differs from :class:`tensordict.nn.ProbabilisticTensorDictModule` in that it accepts a :attr:`spec`
    keyword argument which can be used to control whether samples belong to the distribution or not. The :attr:`safe`
    keyword argument controls whether the samples values should be checked against the spec.

    Args:
        in_keys (NestedKey | List[NestedKey] | Dict[str, NestedKey]): key(s) that will be read from the input TensorDict
            and used to build the distribution.
            Importantly, if it's a list of NestedKey or a NestedKey, the leaf (last element) of those keys must match the keywords used by
            the distribution class of interest, e.g. ``"loc"`` and ``"scale"`` for
            the :class:`~torch.distributions.Normal` distribution and similar.
            If in_keys is a dictionary, the keys are the keys of the distribution and the values are the keys in the
            tensordict that will get match to the corresponding distribution keys.
        out_keys (NestedKey | List[NestedKey] | None): key(s) where the sampled values will be written.
            Importantly, if these keys are found in the input TensorDict, the sampling step will be skipped.
        spec (TensorSpec): specs of the first output tensor. Used when calling
            td_module.random() to generate random values in the target space.

    Keyword Args:
        safe (bool, optional): if ``True``, the value of the sample is checked against the
            input spec. Out-of-domain sampling can occur because of exploration policies
            or numerical under/overflow issues. As for the :obj:`spec` argument, this
            check will only occur for the distribution sample, but not the other tensors
            returned by the input module. If the sample is out of bounds, it is
            projected back onto the desired space using the `TensorSpec.project` method.
            Default is ``False``.
        default_interaction_type (InteractionType, optional): keyword-only argument.
            Default method to be used to retrieve
            the output value. Should be one of InteractionType: MODE, MEDIAN, MEAN or RANDOM
            (in which case the value is sampled randomly from the distribution). Default
            is MODE.

            .. note:: When a sample is drawn, the
                :class:`ProbabilisticTensorDictModule` instance will
                first look for the interaction mode dictated by the
                :func:`~tensordict.nn.probabilistic.interaction_type`
                global function. If this returns `None` (its default value), then the
                `default_interaction_type` of the `ProbabilisticTDModule`
                instance will be used. Note that
                :class:`~torchrl.collectors.collectors.DataCollectorBase`
                instances will use `set_interaction_type` to
                :class:`tensordict.nn.InteractionType.RANDOM` by default.

            .. note::
                In some cases, the mode, median or mean value may not be
                readily available through the corresponding attribute.
                To paliate this, :class:`~ProbabilisticTensorDictModule` will first attempt
                to get the value through a call to ``get_mode()``, ``get_median()`` or ``get_mean()``
                if the method exists.

        distribution_class (Type or Callable[[Any], Distribution], optional): keyword-only argument.
            A :class:`torch.distributions.Distribution` class to
            be used for sampling.
            Default is :class:`~tensordict.nn.distributions.Delta`.

            .. note::
                If the distribution class is of type
                :class:`~tensordict.nn.distributions.CompositeDistribution`, the ``out_keys``
                can be inferred directly form the ``"distribution_map"`` or ``"name_map"``
                keyword arguments provided through this class' ``distribution_kwargs``
                keyword argument, making the ``out_keys`` optional in such cases.

        distribution_kwargs (dict, optional): keyword-only argument.
            Keyword-argument pairs to be passed to the distribution.

            .. note:: if your kwargs contain tensors that you would like to transfer to device with the module, or
                tensors that should see their dtype modified when calling `module.to(dtype)`, you can wrap the kwargs
                in a :class:`~tensordict.nn.TensorDictParams` to do this automatically.

        return_log_prob (bool, optional): keyword-only argument.
            If ``True``, the log-probability of the
            distribution sample will be written in the tensordict with the key
            `log_prob_key`. Default is ``False``.
        log_prob_keys (List[NestedKey], optional): keys where to write the log_prob if ``return_log_prob=True``.
            Defaults to `'<sample_key_name>_log_prob'`, where `<sample_key_name>` is each of the :attr:`out_keys`.

            .. note:: This is only available when :func:`~tensordict.nn.probabilistic.composite_lp_aggregate` is set to ``False``.

        log_prob_key (NestedKey, optional): key where to write the log_prob if ``return_log_prob=True``.
            Defaults to `'sample_log_prob'` when :func:`~tensordict.nn.probabilistic.composite_lp_aggregate` is set to `True`
            or `'<sample_key_name>_log_prob'` otherwise.

            .. note:: When there is more than one sample, this is only available when :func:`~tensordict.nn.probabilistic.composite_lp_aggregate` is set to ``True``.

        cache_dist (bool, optional): keyword-only argument.
            EXPERIMENTAL: if ``True``, the parameters of the
            distribution (i.e. the output of the module) will be written to the
            tensordict along with the sample. Those parameters can be used to re-compute
            the original distribution later on (e.g. to compute the divergence between
            the distribution used to sample the action and the updated distribution in
            PPO). Default is ``False``.
        n_empirical_estimate (int, optional): keyword-only argument.
            Number of samples to compute the empirical
            mean when it is not available. Defaults to 1000.

    .. warning:: Running checks takes time! Using `safe=True` will guarantee that the samples are within the spec bounds
        given some heuristic coded in :meth:`~torchrl.data.TensorSpec.project`, but that requires checking whether the
        values are within the spec space, which will induce some overhead.

    .. seealso:: :class`The composite distribution in tensordict <~tensordict.nn.CompositeDistribution>` can be used
      to create multi-head policies.

    Example:
        >>> from torchrl.modules import SafeProbabilisticModule
        >>> from torchrl.data import Bounded
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import InteractionType
        >>> mod = SafeProbabilisticModule(
        ...     in_keys=["loc", "scale"],
        ...     out_keys=["action"],
        ...     distribution_class=torch.distributions.Normal,
        ...     safe=True,
        ...     spec=Bounded(low=-1, high=1, shape=()),
        ...     default_interaction_type=InteractionType.RANDOM
        ... )
        >>> _ = torch.manual_seed(0)
        >>> data = TensorDict(
        ...     loc=torch.zeros(10, requires_grad=True),
        ...     scale=torch.full((10,), 10.0),
        ...     batch_size=(10,))
        >>> data = mod(data)
        >>> print(data["action"]) # All actions are within bound
        tensor([ 1., -1., -1.,  1., -1., -1.,  1.,  1., -1., -1.],
               grad_fn=<ClampBackward0>)
        >>> data["action"].mean().backward()
        >>> print(data["loc"].grad) # clamp anihilates gradients
        tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    """

    def __init__(
        self,
        in_keys: NestedKey | list[NestedKey] | dict[str, NestedKey],
        out_keys: NestedKey | list[NestedKey] | None = None,
        spec: TensorSpec | None = None,
        *,
        safe: bool = False,
        default_interaction_type: InteractionType = InteractionType.DETERMINISTIC,
        distribution_class: type = Delta,
        distribution_kwargs: dict | None = None,
        return_log_prob: bool = False,
        log_prob_keys: list[NestedKey] | None = None,
        log_prob_key: NestedKey | None = None,
        cache_dist: bool = False,
        n_empirical_estimate: int = 1000,
        num_samples: int | torch.Size | None = None,
    ):
        super().__init__(
            in_keys=in_keys,
            out_keys=out_keys,
            default_interaction_type=default_interaction_type,
            distribution_class=distribution_class,
            distribution_kwargs=distribution_kwargs,
            return_log_prob=return_log_prob,
            cache_dist=cache_dist,
            n_empirical_estimate=n_empirical_estimate,
            log_prob_keys=log_prob_keys,
            log_prob_key=log_prob_key,
            num_samples=num_samples,
        )
        if spec is not None:
            spec = spec.clone()
        if spec is not None and not isinstance(spec, TensorSpec):
            raise TypeError("spec must be a TensorSpec subclass")
        elif spec is not None and not isinstance(spec, Composite):
            if len(self.out_keys) - return_log_prob > 1:
                raise RuntimeError(
                    f"got more than one out_key for the SafeModule: {self.out_keys},\nbut only one spec. "
                    "Consider using a Composite object or no spec at all."
                )
            spec = Composite({self.out_keys[0]: spec})
        elif spec is not None and isinstance(spec, Composite):
            if "_" in spec.keys():
                warnings.warn('got a spec with key "_": it will be ignored')
        elif spec is None:
            spec = Composite()
        spec_keys = set(unravel_key_list(list(spec.keys(True, True))))
        out_keys = set(unravel_key_list(self._out_keys))
        if spec_keys != out_keys:
            # then assume that all the non indicated specs are None
            for key in out_keys:
                if key not in spec_keys:
                    spec[key] = None
            spec_keys = set(unravel_key_list(list(spec.keys(True, True))))

        if spec_keys != out_keys:
            raise RuntimeError(
                f"spec keys and out_keys do not match, got: {spec_keys} and {out_keys} respectively"
            )

        self._spec = spec
        self.safe = safe
        if safe:
            if spec is None or (
                isinstance(spec, Composite)
                and all(_spec is None for _spec in spec.values())
            ):
                raise RuntimeError(
                    "`SafeProbabilisticModule(spec=None, safe=True)` is not a valid configuration as the tensor "
                    "specs are not specified"
                )
            self.register_forward_hook(_forward_hook_safe_action)

    @property
    def spec(self) -> Composite:
        return self._spec

    @spec.setter
    def spec(self, spec: Composite) -> None:
        if not isinstance(spec, Composite):
            raise RuntimeError(
                f"Trying to set an object of type {type(spec)} as a tensorspec but expected a Composite instance."
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


class SafeProbabilisticTensorDictSequential(
    ProbabilisticTensorDictSequential, SafeSequential
):
    """:class:`tensordict.nn.ProbabilisticTensorDictSequential` subclass that accepts a :class:`~torchrl.envs.TensorSpec` as argument to control the output domain.

    Similarly to :obj:`TensorDictSequential`, but enforces that the final module in the
    sequence is an :obj:`ProbabilisticTensorDictModule` and also exposes ``get_dist``
    method to recover the distribution object from the ``ProbabilisticTensorDictModule``

    Args:
         modules (iterable of TensorDictModules): ordered sequence of TensorDictModule
            instances, terminating in ProbabilisticTensorDictModule, to be run
            sequentially.
         partial_tolerant (bool, optional): if ``True``, the input tensordict can miss some
            of the input keys. If so, the only modules that will be executed are those
            which can be executed given the keys that are present. Also, if the input
            tensordict is a lazy stack of tensordicts AND if partial_tolerant is
            ``True`` AND if the stack does not have the required keys, then
            TensorDictSequential will scan through the sub-tensordicts looking for those
            that have the required keys, if any.

    """

    def __init__(
        self,
        *modules: TensorDictModule | ProbabilisticTensorDictModule,
        partial_tolerant: bool = False,
    ) -> None:
        super().__init__(*modules, partial_tolerant=partial_tolerant)
        super(ProbabilisticTensorDictSequential, self).__init__(
            *modules, partial_tolerant=partial_tolerant
        )
