# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence, Type, Union

from tensordict.nn import ProbabilisticTensorDictModule

from torchrl.data import TensorSpec
from torchrl.modules.distributions import Delta
from torchrl.modules.tensordict_module.common import SafeModule


class SafeProbabilisticModule(ProbabilisticTensorDictModule, SafeModule):
    """A :obj:``SafeProbabilisticModule`` is an :obj:``tensordict.nn.ProbabilisticTensorDictModule`` subclass that accepts a :obj:``TensorSpec`` as argument to control the output domain.

    It consists in a wrapper around another TDModule that returns a tensordict
    updated with the distribution parameters. :obj:`SafeProbabilisticModule` is
    responsible for constructing the distribution (through the :obj:`get_dist()` method)
    and/or sampling from this distribution (through a regular :obj:`__call__()` to the
    module).

    A :obj:`SafeProbabilisticModule` instance has two main features:
    - It reads and writes TensorDict objects
    - It uses a real mapping R^n -> R^m to create a distribution in R^d from
    which values can be sampled or computed.
    When the :obj:`__call__` / :obj:`forward` method is called, a distribution is created,
    and a value computed (using the 'mean', 'mode', 'median' attribute or
    the 'rsample', 'sample' method). The sampling step is skipped if the
    inner TDModule has already created the desired key-value pair.

    By default, SafeProbabilisticModule distribution class is a Delta
    distribution, making SafeProbabilisticModule a simple wrapper around
    a deterministic mapping function.

    Args:
        module (nn.Module): a nn.Module used to map the input to the output parameter space. Can be a functional
            module (FunctionalModule or FunctionalModuleWithBuffers), in which case the :obj:`forward` method will expect
            the params (and possibly) buffers keyword arguments.
        dist_in_keys (str or iterable of str or dict): key(s) that will be produced
            by the inner TDModule and that will be used to build the distribution.
            Importantly, if it's an iterable of string or a string, those keys must match the keywords used by the distribution
            class of interest, e.g. :obj:`"loc"` and :obj:`"scale"` for the Normal distribution
            and similar. If dist_in_keys is a dictionary,, the keys are the keys of the distribution and the values are the
            keys in the tensordict that will get match to the corresponding distribution keys.
        sample_out_key (str or iterable of str): keys where the sampled values will be
            written. Importantly, if this key is part of the :obj:`out_keys` of the inner model,
            the sampling step will be skipped.
        spec (TensorSpec): specs of the first output tensor. Used when calling td_module.random() to generate random
            values in the target space.
        safe (bool, optional): if True, the value of the sample is checked against the input spec. Out-of-domain sampling can
            occur because of exploration policies or numerical under/overflow issues. As for the :obj:`spec` argument,
            this check will only occur for the distribution sample, but not the other tensors returned by the input
            module. If the sample is out of bounds, it is projected back onto the desired space using the
            `TensorSpec.project`
            method.
            Default is :obj:`False`.
        default_interaction_mode (str, optional): default method to be used to retrieve the output value. Should be one of:
            'mode', 'median', 'mean' or 'random' (in which case the value is sampled randomly from the distribution).
            Default is 'mode'.
            Note: When a sample is drawn, the :obj:`ProbabilisticTDModule` instance will fist look for the interaction mode
            dictated by the `exploration_mode()` global function. If this returns `None` (its default value),
            then the `default_interaction_mode` of the `ProbabilisticTDModule` instance will be used.
            Note that DataCollector instances will use `set_exploration_mode` to `"random"` by default.
        distribution_class (Type, optional): a torch.distributions.Distribution class to be used for sampling.
            Default is Delta.
        distribution_kwargs (dict, optional): kwargs to be passed to the distribution.
        return_log_prob (bool, optional): if True, the log-probability of the distribution sample will be written in the
            tensordict with the key `f'{in_keys[0]}_log_prob'`. Default is `False`.
        cache_dist (bool, optional): EXPERIMENTAL: if True, the parameters of the distribution (i.e. the output of the module)
            will be written to the tensordict along with the sample. Those parameters can be used to
            re-compute the original distribution later on (e.g. to compute the divergence between the distribution
            used to sample the action and the updated distribution in PPO).
            Default is `False`.
        n_empirical_estimate (int, optional): number of samples to compute the empirical mean when it is not available.
            Default is 1000

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn.functional_modules import make_functional
        >>> from torchrl.data import CompositeSpec, NdUnboundedContinuousTensorSpec
        >>> from torchrl.modules import (
                NormalParamWrapper,
                SafeModule,
                SafeProbabilisticModule,
                TanhNormal,
            )
        >>> td = TensorDict({"input": torch.randn(3, 4), "hidden": torch.randn(3, 8)}, [3,])
        >>> spec = CompositeSpec(action=NdUnboundedContinuousTensorSpec(4), loc=None, scale=None)
        >>> net = NormalParamWrapper(torch.nn.GRUCell(4, 8))
        >>> module = SafeModule(net, in_keys=["input", "hidden"], out_keys=["loc", "scale"])
        >>> td_module = SafeProbabilisticModule(
        ...     module=module,
        ...     spec=spec,
        ...     dist_in_keys=["loc", "scale"],
        ...     sample_out_key=["action"],
        ...     distribution_class=TanhNormal,
        ...     return_log_prob=True,
        ... )
        >>> params = make_functional(td_module)
        >>> td_module(td, params=params)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 8]), dtype=torch.float32),
                input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                loc: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32),
                scale: Tensor(torch.Size([3, 4]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> # In the vmap case, the tensordict is again expended to match the batch:
        >>> from functorch import vmap
        >>> params = params.expand(4, *params.shape)
        >>> td_vmap = vmap(td_module, (None, 0))(td, params)
        >>> print(td_vmap)
        TensorDict(
            fields={
                action: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([4, 3, 8]), dtype=torch.float32),
                input: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                loc: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([4, 3, 1]), dtype=torch.float32),
                scale: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32)},
            batch_size=torch.Size([4, 3]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        module: SafeModule,
        dist_in_keys: Union[str, Sequence[str], dict],
        sample_out_key: Union[str, Sequence[str]],
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
            module=module,
            dist_in_keys=dist_in_keys,
            sample_out_key=sample_out_key,
            default_interaction_mode=default_interaction_mode,
            distribution_class=distribution_class,
            distribution_kwargs=distribution_kwargs,
            return_log_prob=return_log_prob,
            cache_dist=cache_dist,
            n_empirical_estimate=n_empirical_estimate,
        )
        super(ProbabilisticTensorDictModule, self).__init__(
            module=module,
            spec=spec,
            in_keys=self.in_keys,
            out_keys=self.out_keys,
            safe=safe,
        )
