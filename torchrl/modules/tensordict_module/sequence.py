# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import copy, deepcopy
from typing import Iterable, List, Optional, Tuple, Union

_has_functorch = False
try:
    import functorch

    _has_functorch = True
except ImportError:
    print(
        "failed to import functorch. TorchRL's features that do not require "
        "functional programming should work, but functionality and performance "
        "may be affected. Consider installing functorch and/or upgrating pytorch."
    )
    FUNCTORCH_ERROR = "functorch not installed. Consider installing functorch to use this functionality."

import torch
from torch import Tensor, nn

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.data.tensordict.tensordict import (
    LazyStackedTensorDict,
    TensorDict,
    TensorDictBase,
)
from torchrl.modules.tensordict_module.common import TensorDictModule
from torchrl.modules.tensordict_module.probabilistic import (
    ProbabilisticTensorDictModule,
)

__all__ = ["TensorDictSequential"]


class TensorDictSequential(TensorDictModule):
    """
    A sequence of TDModules.
    Similarly to `nn.Sequence` which passes a tensor through a chain of mappings that read and write a single tensor
    each, this module will read and write over a tensordict by querying each of the input modules.
    When calling a `TDSequence` instance with a functional module, it is expected that the parameter lists (and
    buffers) will be concatenated in a single list.

    Args:
         modules (iterable of TDModules): ordered sequence of TDModule instances to be run sequentially.
         partial_tolerant (bool, optional): if True, the input tensordict can miss some of the input keys.
            If so, the only module that will be executed are those who can be executed given the keys that
            are present.
            Also, if the input tensordict is a lazy stack of tensordicts AND if partial_tolerant is `True` AND if the
            stack does not have the required keys, then TensorDictSequential will scan through the sub-tensordicts
            looking for those that have the required keys, if any.

    TDSequence supports functional, modular and vmap coding:
    Examples:
        >>> from torchrl.modules.tensordict_module import ProbabilisticTensorDictModule
        >>> from torchrl.data import TensorDict, NdUnboundedContinuousTensorSpec
        >>> from torchrl.modules import  TanhNormal, TensorDictSequential, NormalParamWrapper
        >>> import torch, functorch
        >>> td = TensorDict({"input": torch.randn(3, 4)}, [3,])
        >>> spec1 = NdUnboundedContinuousTensorSpec(4)
        >>> net1 = NormalParamWrapper(torch.nn.Linear(4, 8))
        >>> fnet1, params1, buffers1 = functorch.make_functional_with_buffers(net1)
        >>> fmodule1 = TensorDictModule(fnet1, in_keys=["input"], out_keys=["loc", "scale"])
        >>> td_module1 = ProbabilisticTensorDictModule(
        ...    module=fmodule1,
        ...    spec=spec1,
        ...    dist_param_keys=["loc", "scale"],
        ...    out_key_sample=["hidden"],
        ...    distribution_class=TanhNormal,
        ...    return_log_prob=True,
        ...    )
        >>> spec2 = NdUnboundedContinuousTensorSpec(8)
        >>> module2 = torch.nn.Linear(4, 8)
        >>> fmodule2, params2, buffers2 = functorch.make_functional_with_buffers(module2)
        >>> td_module2 = TensorDictModule(
        ...    module=fmodule2,
        ...    spec=spec2,
        ...    in_keys=["hidden"],
        ...    out_keys=["output"],
        ...    )
        >>> td_module = TensorDictSequential(td_module1, td_module2)
        >>> params = params1 + params2
        >>> buffers = buffers1 + buffers2
        >>> _ = td_module(td, params=params, buffers=buffers)
        >>> print(td)
        TensorDict(
            fields={
                input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                loc: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                scale: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32),
                output: Tensor(torch.Size([3, 8]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=cpu,
            is_shared=False)

        >>> # The module spec aggregates all the input specs:
        >>> print(td_module.spec)
        CompositeSpec(
            hidden: NdUnboundedContinuousTensorSpec(
                 shape=torch.Size([4]),space=None,device=cpu,dtype=torch.float32,domain=continuous),
            output: NdUnboundedContinuousTensorSpec(
                 shape=torch.Size([8]),space=None,device=cpu,dtype=torch.float32,domain=continuous))

    In the vmap case:
        >>> params = tuple(p.expand(4, *p.shape).contiguous().normal_() for p in params)
        >>> buffers = tuple(b.expand(4, *b.shape).contiguous().normal_() for p in buffers)
        >>> td_vmap = td_module(td, params=params, buffers=buffers, vmap=True)
        >>> print(td_vmap)
        TensorDict(
            fields={
                input: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                loc: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                scale: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([4, 3, 1]), dtype=torch.float32),
                output: Tensor(torch.Size([4, 3, 8]), dtype=torch.float32)},
            batch_size=torch.Size([4, 3]),
            device=cpu,
            is_shared=False)


    """

    module: nn.ModuleList

    def __init__(
        self,
        *modules: TensorDictModule,
        partial_tolerant: bool = False,
    ):
        in_keys, out_keys = self._compute_in_and_out_keys(modules)

        super().__init__(
            spec=None,
            module=nn.ModuleList(list(modules)),
            in_keys=in_keys,
            out_keys=out_keys,
        )
        self.partial_tolerant = partial_tolerant

    def _compute_in_and_out_keys(self, modules: List[TensorDictModule]) -> Tuple[List]:
        in_keys = []
        out_keys = []
        for module in modules:
            # we sometimes use in_keys to select keys of a tensordict that are
            # necessary to run a TensorDictModule. If a key is an intermediary in
            # the chain, there is no reason why it should belong to the input
            # TensorDict.
            for in_key in module.in_keys:
                if in_key not in (out_keys + in_keys):
                    in_keys.append(in_key)
            out_keys += module.out_keys

        out_keys = [
            out_key
            for i, out_key in enumerate(out_keys)
            if out_key not in out_keys[i + 1 :]
        ]
        return in_keys, out_keys

    @staticmethod
    def _find_functional_module(module: TensorDictModule) -> nn.Module:
        if not _has_functorch:
            raise ImportError(FUNCTORCH_ERROR)
        fmodule = module
        while not isinstance(
            fmodule, (functorch.FunctionalModule, functorch.FunctionalModuleWithBuffers)
        ):
            try:
                fmodule = fmodule.module
            except AttributeError:
                raise AttributeError(
                    f"couldn't find a functional module in module of type {type(module)}"
                )
        return fmodule

    @property
    def num_params(self):
        return self.param_len[-1]

    @property
    def num_buffers(self):
        return self.buffer_len[-1]

    @property
    def param_len(self) -> List[int]:
        param_list = []
        prev = 0
        for module in self.module:
            param_list.append(module.num_params + prev)
            prev = param_list[-1]
        return param_list

    @property
    def buffer_len(self) -> List[int]:
        buffer_list = []
        prev = 0
        for module in self.module:
            buffer_list.append(module.num_buffers + prev)
            prev = buffer_list[-1]
        return buffer_list

    def _split_param(
        self, param_list: Iterable[Tensor], params_or_buffers: str
    ) -> Iterable[Iterable[Tensor]]:
        if params_or_buffers == "params":
            list_out = self.param_len
        elif params_or_buffers == "buffers":
            list_out = self.buffer_len
        list_in = [0] + list_out[:-1]
        out = []
        for a, b in zip(list_in, list_out):
            out.append(param_list[a:b])
        return out

    def select_subsequence(
        self, in_keys: Iterable[str] = None, out_keys: Iterable[str] = None
    ) -> "TensorDictSequential":
        """
        Returns a new TensorDictSequential with only the modules that are necessary to compute
        the given output keys with the given input keys.

        Args:
            in_keys: input keys of the subsequence we want to select
            out_keys: output keys of the subsequence we want to select

        Returns:
            A new TensorDictSequential with only the modules that are necessary acording to the given input and output keys.
        """
        if in_keys is None:
            in_keys = deepcopy(self.in_keys)
        if out_keys is None:
            out_keys = deepcopy(self.out_keys)
        id_to_keep = {i for i in range(len(self.module))}
        for i, module in enumerate(self.module):
            if all(key in in_keys for key in module.in_keys):
                in_keys.extend(module.out_keys)
            else:
                id_to_keep.remove(i)
        for i, module in reversed(list(enumerate(self.module))):
            if i in id_to_keep:
                if any(key in out_keys for key in module.out_keys):
                    out_keys.extend(module.in_keys)
                else:
                    id_to_keep.remove(i)
        id_to_keep = sorted(list(id_to_keep))

        modules = [self.module[i] for i in id_to_keep]

        if modules == []:
            raise ValueError(
                "No modules left after selection. Make sure that in_keys and out_keys are coherent."
            )

        return TensorDictSequential(*modules)

    def _run_module(
        self,
        module,
        tensordict,
        params: Optional[Union[TensorDictBase, List[Tensor]]] = None,
        buffers: Optional[Union[TensorDictBase, List[Tensor]]] = None,
        **kwargs,
    ):
        tensordict_keys = set(tensordict.keys())
        if not self.partial_tolerant or all(
            key in tensordict_keys for key in module.in_keys
        ):
            if params is not None or buffers is not None:
                tensordict = module(
                    tensordict, params=params, buffers=buffers, **kwargs
                )
            else:
                tensordict = module(tensordict, **kwargs)
        elif self.partial_tolerant and isinstance(tensordict, LazyStackedTensorDict):
            for sub_td in tensordict.tensordicts:
                tensordict_keys = set(sub_td.keys())
                if all(key in tensordict_keys for key in module.in_keys):
                    if params is not None or buffers is not None:
                        module(sub_td, params=params, buffers=buffers, **kwargs)
                    else:
                        module(sub_td, **kwargs)
            tensordict._update_valid_keys()
        return tensordict

    def forward(
        self,
        tensordict: TensorDictBase,
        tensordict_out=None,
        params: Optional[Union[TensorDictBase, List[Tensor]]] = None,
        buffers: Optional[Union[TensorDictBase, List[Tensor]]] = None,
        **kwargs,
    ) -> TensorDictBase:
        if params is not None and buffers is not None:
            if isinstance(params, TensorDictBase):
                # TODO: implement sorted values and items
                param_splits = list(zip(*sorted(list(params.items()))))[1]
                buffer_splits = list(zip(*sorted(list(buffers.items()))))[1]
            else:
                param_splits = self._split_param(params, "params")
                buffer_splits = self._split_param(buffers, "buffers")
            for i, (module, param, buffer) in enumerate(
                zip(self.module, param_splits, buffer_splits)
            ):
                if "vmap" in kwargs and i > 0:
                    # the tensordict is already expended
                    if not isinstance(kwargs["vmap"], tuple):
                        kwargs["vmap"] = (0, 0, *(0,) * len(module.in_keys))
                    else:
                        kwargs["vmap"] = (
                            *kwargs["vmap"][:2],
                            *(0,) * len(module.in_keys),
                        )
                tensordict = self._run_module(
                    module, tensordict, params=param, buffers=buffer, **kwargs
                )

        elif params is not None:
            if isinstance(params, TensorDictBase):
                # TODO: implement sorted values and items
                param_splits = list(zip(*sorted(list(params.items()))))[1]
            else:
                param_splits = self._split_param(params, "params")
            for i, (module, param) in enumerate(zip(self.module, param_splits)):
                if "vmap" in kwargs and i > 0:
                    # the tensordict is already expended
                    if not isinstance(kwargs["vmap"], tuple):
                        kwargs["vmap"] = (0, *(0,) * len(module.in_keys))
                    else:
                        kwargs["vmap"] = (
                            *kwargs["vmap"][:1],
                            *(0,) * len(module.in_keys),
                        )
                tensordict = self._run_module(
                    module, tensordict, params=param, **kwargs
                )

        elif not len(kwargs):
            for module in self.module:
                tensordict = self._run_module(module, tensordict, **kwargs)
        else:
            raise RuntimeError(
                "TensorDictSequential does not support keyword arguments other than 'tensordict_out', 'in_keys', 'out_keys' 'params', 'buffers' and 'vmap'"
            )
        if tensordict_out is not None:
            tensordict_out.update(tensordict, inplace=True)
            return tensordict_out
        return tensordict

    def __len__(self):
        return len(self.module)

    def __getitem__(self, index: Union[int, slice]) -> TensorDictModule:
        if isinstance(index, int):
            return self.module.__getitem__(index)
        else:
            return TensorDictSequential(*self.module.__getitem__(index))

    def __setitem__(self, index: int, tensordict_module: TensorDictModule) -> None:
        return self.module.__setitem__(idx=index, module=tensordict_module)

    def __delitem__(self, index: Union[int, slice]) -> None:
        self.module.__delitem__(idx=index)

    @property
    def spec(self):
        kwargs = {}
        for layer in self.module:
            out_key = layer.out_keys[0]
            spec = layer.spec
            if spec is not None and not isinstance(spec, TensorSpec):
                raise RuntimeError(
                    f"TensorDictSequential.spec requires all specs to be valid TensorSpec objects. Got "
                    f"{type(layer.spec)}"
                )
            if isinstance(spec, CompositeSpec):
                kwargs.update(spec._specs)
            else:
                kwargs[out_key] = spec
        return CompositeSpec(**kwargs)

    def make_functional_with_buffers(self, clone: bool = True, native: bool = False):
        """
        Transforms a stateful module in a functional module and returns its parameters and buffers.
        Unlike functorch.make_functional_with_buffers, this method supports lazy modules.

        Args:
            clone (bool, optional): if True, a clone of the module is created before it is returned.
                This is useful as it prevents the original module to be scraped off of its
                parameters and buffers.
                Defaults to True
            native (bool, optional): if True, TorchRL's functional modules will be used.
                Defaults to True

        Returns:
            A tuple of parameter and buffer tuples

        Examples:
            >>> from torchrl.data import NdUnboundedContinuousTensorSpec, TensorDict
            >>> lazy_module1 = nn.LazyLinear(4)
            >>> lazy_module2 = nn.LazyLinear(3)
            >>> spec1 = NdUnboundedContinuousTensorSpec(18)
            >>> spec2 = NdUnboundedContinuousTensorSpec(4)
            >>> td_module1 = TensorDictModule(spec=spec1, module=lazy_module1, in_keys=["some_input"], out_keys=["hidden"])
            >>> td_module2 = TensorDictModule(spec=spec2, module=lazy_module2, in_keys=["hidden"], out_keys=["some_output"])
            >>> td_module = TensorDictSequential(td_module1, td_module2)
            >>> _, (params, buffers) = td_module.make_functional_with_buffers()
            >>> print(params[0].shape) # the lazy module has been initialized
            torch.Size([4, 18])
            >>> print(td_module(
            ...    TensorDict({'some_input': torch.randn(18)}, batch_size=[]),
            ...    params=params,
            ...    buffers=buffers))
            TensorDict(
                fields={
                    some_input: Tensor(torch.Size([18]), dtype=torch.float32),
                    hidden: Tensor(torch.Size([4]), dtype=torch.float32),
                    some_output: Tensor(torch.Size([3]), dtype=torch.float32)},
                batch_size=torch.Size([]),
                device=cpu,
                is_shared=False)

        """
        if clone:
            self_copy = deepcopy(self)
            self_copy.module = copy(self_copy.module)
        else:
            self_copy = self
        params = [] if not native else TensorDict({}, [])
        buffers = [] if not native else TensorDict({}, [])
        for i, module in enumerate(self.module):
            self_copy.module[i], (
                _params,
                _buffers,
            ) = module.make_functional_with_buffers(clone=True, native=native)
            if native or not _has_functorch:
                params[str(i)] = _params
                buffers[str(i)] = _buffers
            else:
                params.extend(_params)
                buffers.extend(_buffers)
        return self_copy, (params, buffers)

    def get_dist(
        self,
        tensordict: TensorDictBase,
        **kwargs,
    ) -> Tuple[torch.distributions.Distribution, ...]:
        L = len(self.module)

        if isinstance(self.module[-1], ProbabilisticTensorDictModule):
            if "params" in kwargs and "buffers" in kwargs:
                params = kwargs["params"]
                buffers = kwargs["buffers"]
                if isinstance(params, TensorDictBase):
                    param_splits = list(zip(*sorted(list(params.items()))))[1]
                    buffer_splits = list(zip(*sorted(list(buffers.items()))))[1]
                else:
                    param_splits = self._split_param(kwargs["params"], "params")
                    buffer_splits = self._split_param(kwargs["buffers"], "buffers")
                kwargs_pruned = {
                    key: item
                    for key, item in kwargs.items()
                    if key not in ("params", "buffers")
                }
                for i, (module, param, buffer) in enumerate(
                    zip(self.module, param_splits, buffer_splits)
                ):
                    if "vmap" in kwargs_pruned and i > 0:
                        # the tensordict is already expended
                        kwargs_pruned["vmap"] = (0, 0, *(0,) * len(module.in_keys))
                    if i < L - 1:
                        tensordict = module(
                            tensordict, params=param, buffers=buffer, **kwargs_pruned
                        )
                    else:
                        out = module.get_dist(
                            tensordict, params=param, buffers=buffer, **kwargs_pruned
                        )

            elif "params" in kwargs:
                params = kwargs["params"]
                if isinstance(params, TensorDictBase):
                    param_splits = list(zip(*sorted(list(params.items()))))[1]
                else:
                    param_splits = self._split_param(kwargs["params"], "params")
                kwargs_pruned = {
                    key: item for key, item in kwargs.items() if key not in ("params",)
                }
                for i, (module, param) in enumerate(zip(self.module, param_splits)):
                    if "vmap" in kwargs_pruned and i > 0:
                        # the tensordict is already expended
                        kwargs_pruned["vmap"] = (0, *(0,) * len(module.in_keys))
                    if i < L - 1:
                        tensordict = module(tensordict, params=param, **kwargs_pruned)
                    else:
                        out = module.get_dist(tensordict, params=param, **kwargs_pruned)

            elif not len(kwargs):
                for i, module in enumerate(self.module):
                    if i < L - 1:
                        tensordict = module(tensordict)
                    else:
                        out = module.get_dist(tensordict)
            else:
                raise RuntimeError(
                    "TensorDictSequential does not support keyword arguments other than 'params', 'buffers' and 'vmap'"
                )

            return out
        else:
            raise RuntimeError(
                "Cannot call get_dist on a sequence of tensordicts that does not end with a probabilistic TensorDict"
            )
