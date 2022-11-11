# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from copy import deepcopy
from textwrap import indent
from typing import List, Sequence, Union, Type, Optional, Tuple

from torch import Tensor
from torch import distributions as d

from torchrl.data import TensorSpec
from torchrl.data.tensordict.tensordict import TensorDictBase
from torchrl.envs.utils import exploration_mode, set_exploration_mode
from torchrl.modules.distributions import distributions_maps, Delta
from torchrl.modules.tensordict_module.common import TensorDictModule, _check_all_str


class ProbabilisticTensorDictModule(TensorDictModule):
    """A probabilistic TD Module.

    `ProbabilisticTDModule` is a special case of a TDModule where the output is
    sampled given some rule, specified by the input :obj:`default_interaction_mode`
    argument and the :obj:`exploration_mode()` global function.

    It consists in a wrapper around another TDModule that returns a tensordict
    updated with the distribution parameters. :obj:`ProbabilisticTensorDictModule` is
    responsible for constructing the distribution (through the :obj:`get_dist()` method)
    and/or sampling from this distribution (through a regular :obj:`__call__()` to the
    module).

    A :obj:`ProbabilisticTensorDictModule` instance has two main features:
    - It reads and writes TensorDict objects
    - It uses a real mapping R^n -> R^m to create a distribution in R^d from
    which values can be sampled or computed.
    When the :obj:`__call__` / :obj:`forward` method is called, a distribution is created,
    and a value computed (using the 'mean', 'mode', 'median' attribute or
    the 'rsample', 'sample' method). The sampling step is skipped if the
    inner TDModule has already created the desired key-value pair.

    By default, ProbabilisticTensorDictModule distribution class is a Delta
    distribution, making ProbabilisticTensorDictModule a simple wrapper around
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
        >>> from torchrl.modules import ProbabilisticTensorDictModule
        >>> from torchrl.data import TensorDict, NdUnboundedContinuousTensorSpec
        >>> from torchrl.modules import  TanhNormal, NormalParamWrapper
        >>> import functorch, torch
        >>> td = TensorDict({"input": torch.randn(3, 4), "hidden": torch.randn(3, 8)}, [3,])
        >>> spec = NdUnboundedContinuousTensorSpec(4)
        >>> net = NormalParamWrapper(torch.nn.GRUCell(4, 8))
        >>> fnet, params, buffers = functorch.make_functional_with_buffers(net)
        >>> module = TensorDictModule(fnet, in_keys=["input", "hidden"], out_keys=["loc", "scale"])
        >>> td_module = ProbabilisticTensorDictModule(
        ...    module=module,
        ...    spec=spec,
        ...    dist_in_keys=["loc", "scale"],
        ...    sample_out_key=["action"],
        ...    distribution_class=TanhNormal,
        ...    return_log_prob=True,
        ...    )
        >>> _ = td_module(td, params=params, buffers=buffers)
        >>> print(td)
        TensorDict(
            fields={
                input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 8]), dtype=torch.float32),
                loc: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                scale: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                action: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=cpu,
            is_shared=False)

        >>> # In the vmap case, the tensordict is again expended to match the batch:
        >>> params = tuple(p.expand(4, *p.shape).contiguous().normal_() for p in params)
        >>> buffers = tuple(b.expand(4, *b.shape).contiguous().normal_() for p in buffers)
        >>> td_vmap = td_module(td, params=params, buffers=buffers, vmap=True)
        >>> print(td_vmap)
        TensorDict(
            fields={
                input: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([4, 3, 8]), dtype=torch.float32),
                loc: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                scale: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                action: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([4, 3, 1]), dtype=torch.float32)},
            batch_size=torch.Size([4, 3]),
            device=cpu,
            is_shared=False)

    """

    def __init__(
        self,
        module: TensorDictModule,
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
        in_keys = module.in_keys

        # if the module returns the sampled key we wont be sampling it again
        # then ProbabilisticTensorDictModule is presumably used to return the distribution using `get_dist`
        if isinstance(dist_in_keys, str):
            dist_in_keys = [dist_in_keys]
        if isinstance(sample_out_key, str):
            sample_out_key = [sample_out_key]
        if not isinstance(dist_in_keys, dict):
            dist_in_keys = {param_key: param_key for param_key in dist_in_keys}
        for key in dist_in_keys.values():
            if key not in module.out_keys:
                raise RuntimeError(
                    f"The key {key} could not be found in the wrapped module `{type(module)}.out_keys`."
                )
        module_out_keys = module.out_keys
        self.sample_out_key = sample_out_key
        _check_all_str(self.sample_out_key)
        sample_out_key = [key for key in sample_out_key if key not in module_out_keys]
        self._requires_sample = bool(len(sample_out_key))
        out_keys = sample_out_key + module_out_keys
        super().__init__(
            module=module, spec=spec, in_keys=in_keys, out_keys=out_keys, safe=safe
        )
        self.dist_in_keys = dist_in_keys
        _check_all_str(self.dist_in_keys.keys())
        _check_all_str(self.dist_in_keys.values())

        self.default_interaction_mode = default_interaction_mode
        if isinstance(distribution_class, str):
            distribution_class = distributions_maps.get(distribution_class.lower())
        self.distribution_class = distribution_class
        self.distribution_kwargs = (
            distribution_kwargs if distribution_kwargs is not None else dict()
        )
        self.n_empirical_estimate = n_empirical_estimate
        self._dist = None
        self.cache_dist = cache_dist if hasattr(distribution_class, "update") else False
        self.return_log_prob = return_log_prob

    def _call_module(
        self,
        tensordict: TensorDictBase,
        params: Optional[Union[TensorDictBase, List[Tensor]]] = None,
        buffers: Optional[Union[TensorDictBase, List[Tensor]]] = None,
        **kwargs,
    ) -> TensorDictBase:
        return self.module(tensordict, params=params, buffers=buffers, **kwargs)

    def make_functional_with_buffers(self, clone: bool = True, native: bool = False):
        module_params = self.parameters(recurse=False)
        if len(list(module_params)):
            raise RuntimeError(
                "make_functional_with_buffers cannot be called on ProbabilisticTensorDictModule"
                "that contain parameters on the outer level."
            )
        if clone:
            self_copy = deepcopy(self)
        else:
            self_copy = self

        self_copy.module, other = self_copy.module.make_functional_with_buffers(
            clone=True,
            native=native,
        )
        return self_copy, other

    def get_dist(
        self,
        tensordict: TensorDictBase,
        tensordict_out: Optional[TensorDictBase] = None,
        params: Optional[Union[TensorDictBase, List[Tensor]]] = None,
        buffers: Optional[Union[TensorDictBase, List[Tensor]]] = None,
        **kwargs,
    ) -> Tuple[d.Distribution, TensorDictBase]:
        interaction_mode = exploration_mode()
        if interaction_mode is None:
            interaction_mode = self.default_interaction_mode
        with set_exploration_mode(interaction_mode):
            tensordict_out = self._call_module(
                tensordict,
                tensordict_out=tensordict_out,
                params=params,
                buffers=buffers,
                **kwargs,
            )
        dist = self.build_dist_from_params(tensordict_out)
        return dist, tensordict_out

    def build_dist_from_params(self, tensordict_out: TensorDictBase) -> d.Distribution:
        try:
            selected_td_out = tensordict_out.select(*self.dist_in_keys.values())
            dist_kwargs = {
                dist_key: selected_td_out[td_key]
                for dist_key, td_key in self.dist_in_keys.items()
            }
            dist = self.distribution_class(**dist_kwargs)
        except TypeError as err:
            if "an unexpected keyword argument" in str(err):
                raise TypeError(
                    "distribution keywords and tensordict keys indicated by ProbabilisticTensorDictModule.dist_in_keys must match."
                    f"Got this error message: \n{indent(str(err), 4 * ' ')}\nwith dist_in_keys={self.dist_in_keys}"
                )
            elif re.search(r"missing.*required positional arguments", str(err)):
                raise TypeError(
                    f"TensorDict with keys {tensordict_out.keys()} does not match the distribution {self.distribution_class} keywords."
                )
            else:
                raise err
        return dist

    def forward(
        self,
        tensordict: TensorDictBase,
        tensordict_out: Optional[TensorDictBase] = None,
        params: Optional[Union[TensorDictBase, List[Tensor]]] = None,
        buffers: Optional[Union[TensorDictBase, List[Tensor]]] = None,
        **kwargs,
    ) -> TensorDictBase:

        dist, tensordict_out = self.get_dist(
            tensordict,
            tensordict_out=tensordict_out,
            params=params,
            buffers=buffers,
            **kwargs,
        )
        if self._requires_sample:
            out_tensors = self._dist_sample(dist, interaction_mode=exploration_mode())
            if isinstance(out_tensors, Tensor):
                out_tensors = (out_tensors,)
            tensordict_out.update(
                {key: value for key, value in zip(self.sample_out_key, out_tensors)}
            )
            if self.return_log_prob:
                log_prob = dist.log_prob(*out_tensors)
                tensordict_out.set("sample_log_prob", log_prob)
        elif self.return_log_prob:
            out_tensors = [tensordict_out.get(key) for key in self.sample_out_key]
            log_prob = dist.log_prob(*out_tensors)
            tensordict_out.set("sample_log_prob", log_prob)
            # raise RuntimeError(
            #     "ProbabilisticTensorDictModule.return_log_prob = True is incompatible with settings in which "
            #     "the submodule is responsible for sampling. To manually gather the log-probability, call first "
            #     "\n>>> dist, tensordict = tensordict_module.get_dist(tensordict)"
            #     "\n>>> tensordict.set('sample_log_prob', dist.log_prob(tensordict.get(sample_key))"
            # )
        return tensordict_out

    def _dist_sample(
        self,
        dist: d.Distribution,
        *tensors: Tensor,
        interaction_mode: bool = None,
    ) -> Union[Tuple[Tensor], Tensor]:
        if interaction_mode is None or interaction_mode == "":
            interaction_mode = self.default_interaction_mode
        if not isinstance(dist, d.Distribution):
            raise TypeError(f"type {type(dist)} not recognised by _dist_sample")

        if interaction_mode == "mode":
            if hasattr(dist, "mode"):
                return dist.mode
            else:
                raise NotImplementedError(
                    f"method {type(dist)}.mode is not implemented"
                )

        elif interaction_mode == "median":
            if hasattr(dist, "median"):
                return dist.median
            else:
                raise NotImplementedError(
                    f"method {type(dist)}.median is not implemented"
                )

        elif interaction_mode == "mean":
            try:
                return dist.mean
            except (AttributeError, NotImplementedError):
                if dist.has_rsample:
                    return dist.rsample((self.n_empirical_estimate,)).mean(0)
                else:
                    return dist.sample((self.n_empirical_estimate,)).mean(0)

        elif interaction_mode == "random":
            if dist.has_rsample:
                return dist.rsample()
            else:
                return dist.sample()
        else:
            raise NotImplementedError(f"unknown interaction_mode {interaction_mode}")

    @property
    def num_params(self):
        return self.module.num_params

    @property
    def num_buffers(self):
        return self.module.num_buffers
