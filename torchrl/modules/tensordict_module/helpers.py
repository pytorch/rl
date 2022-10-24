# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from torchrl.modules.distributions import NormalParamWrapper, Delta
from torchrl.modules.tensordict_module.actors import ProbabilisticActor
from torchrl.modules.tensordict_module.common import TensorDictModule
from torchrl.modules.tensordict_module.probabilistic import (
    ProbabilisticTensorDictModule,
)

__all__ = [
    "partial_tensordictmodule",
    "partial_probabilisticactor",
    "partial_probabilistictensordictmodule",
]

_WRAPPER_DICT = {
    "normal_param": NormalParamWrapper,
}


def partial_tensordictmodule(
    partial_module, in_keys, out_keys, spec=None, safe=False, wrapper=None, **kwargs
):
    """Creates a partially instantiated :obj:`TensorDictModule`.

    Args:
        partial_module (partially instantiated :obj:`nn.Module`): a module to instantiate using
            the kwargs provided to this function.
        in_keys (iterable of strings): in_keys of the :obj:`TensorDictModule`.
        out_keys (iterable of strings): out_keys of the :obj:`TensorDictModule`.
        spec (TensorSpec, optional): the optional TensorSpec to pass to the
            TensorDictModule.
        safe (bool, optional): safe arg for the :obj:`TensorDictModule`. Default: :obj:`False`.
        wrapper (module wrapper): an optional module wrapper, such as
            :obj:`NormalParamWrapper` (indicated by :obj:`"normal_param"`).

    """
    module = partial_module(**kwargs)
    if wrapper is not None:
        module = _WRAPPER_DICT[wrapper](module)

    return TensorDictModule(
        module=module,
        in_keys=in_keys,
        out_keys=out_keys,
        spec=spec,
        safe=safe,
    )


def partial_probabilisticactor(
    partial_tensordictmodule,
    dist_param_keys,
    out_key_sample=None,
    spec=None,
    safe=False,
    default_interaction_mode="mode",
    distribution_class=Delta,
    distribution_kwargs=None,
    return_log_prob=False,
    n_empirical_estimate: int = 1000,
    **kwargs,
):
    """Creates a partially instantiated :obj:`ProbabilisticActor`.

    Args:
        partial_tensordictmodule (partially instantiated :obj:`TensorDictModule`): a module to instantiate
            using the kwargs provided to this function.
        dist_param_keys (str or iterable of str or dict): key(s) that will be produced
            by the inner TDModule and that will be used to build the distribution.
            Importantly, if it's an iterable of string or a string, those keys must match the keywords used by the distribution
            class of interest, e.g. :obj:`"loc"` and :obj:`"scale"` for the Normal distribution
            and similar. If dist_param_keys is a dictionary,, the keys are the keys of the distribution and the values are the
            keys in the tensordict that will get match to the corresponding distribution keys.
        out_key_sample (str or iterable of str): keys where the sampled values will be
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
        distribution_class (Type or str, optional): a torch.distributions.Distribution class
            (or an equivalent string among `"normal", `"delta"`, `"independent_normal"`, `"tanh_normal"`, `"truncated_normal"`, `"onehot_categorical"`)
            to be used for sampling.
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
        **kwargs: optional arguments of the :obj:`partial_tensordictmodule`.
    """
    return ProbabilisticActor(
        module=partial_tensordictmodule(**kwargs),
        dist_param_keys=dist_param_keys,
        out_key_sample=out_key_sample,
        spec=spec,
        safe=safe,
        default_interaction_mode=default_interaction_mode,
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=return_log_prob,
        n_empirical_estimate=n_empirical_estimate,
    )


def partial_probabilistictensordictmodule(
    partial_tensordictmodule,
    dist_param_keys,
    out_key_sample=None,
    spec=None,
    safe=False,
    default_interaction_mode="mode",
    distribution_class=Delta,
    distribution_kwargs=None,
    return_log_prob=False,
    n_empirical_estimate: int = 1000,
    **kwargs,
):
    """Creates a partially instantiated :obj:`ProbabilisticTensorDictModule`.

    Args:
        partial_tensordictmodule (partially instantiated :obj:`TensorDictModule`): a module to instantiate
            using the kwargs provided to this function.
        dist_param_keys (str or iterable of str or dict): key(s) that will be produced
            by the inner TDModule and that will be used to build the distribution.
            Importantly, if it's an iterable of string or a string, those keys must match the keywords used by the distribution
            class of interest, e.g. :obj:`"loc"` and :obj:`"scale"` for the Normal distribution
            and similar. If dist_param_keys is a dictionary,, the keys are the keys of the distribution and the values are the
            keys in the tensordict that will get match to the corresponding distribution keys.
        out_key_sample (str or iterable of str): keys where the sampled values will be
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
        distribution_class (Type or str, optional): a torch.distributions.Distribution class
            (or an equivalent string among `"normal", `"delta"`, `"independent_normal"`, `"tanh_normal"`, `"truncated_normal"`, `"onehot_categorical"`)
            to be used for sampling.
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
        **kwargs: optional arguments of the :obj:`partial_tensordictmodule`.
    """
    return ProbabilisticTensorDictModule(
        module=partial_tensordictmodule(**kwargs),
        dist_param_keys=dist_param_keys,
        out_key_sample=out_key_sample,
        spec=spec,
        safe=safe,
        default_interaction_mode=default_interaction_mode,
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=return_log_prob,
        n_empirical_estimate=n_empirical_estimate,
    )
