from __future__ import annotations

from copy import copy, deepcopy
from numbers import Number
from typing import Tuple, List, Iterable, Type, Optional, Union, Any, Callable, Sequence

import functorch
import torch
from functorch import FunctionalModule, FunctionalModuleWithBuffers, vmap
from functorch._src.make_functional import _swap_state
from torch import nn, distributions as d, Tensor

from torchrl.data import TensorSpec, DEVICE_TYPING, CompositeSpec
from torchrl.data.tensordict.tensordict import _TensorDict, TensorDict
from torchrl.envs.utils import exploration_mode
from torchrl.modules.distributions import Delta, distributions_maps

__all__ = [
    "TDModule",
    "ProbabilisticTDModule",
    "TDSequence",
    "TDModuleWrapper",
]


def _forward_hook_safe_action(module, tensor_dict_in, tensor_dict_out):
    if not module.spec.is_in(tensor_dict_out.get(module.out_keys[0])):
        try:
            tensor_dict_out.set_(
                module.out_keys[0],
                module.spec.project(tensor_dict_out.get(module.out_keys[0])),
            )
        except RuntimeError:
            tensor_dict_out.set(
                module.out_keys[0],
                module.spec.project(tensor_dict_out.get(module.out_keys[0])),
            )


class TDModule(nn.Module):
    """A TDModule, for TensorDict module, is a python wrapper around a `nn.Module` that reads and writes to a
    TensorDict, instead of reading and returning tensors.

    Args:
        spec (TensorSpec): specs of the output tensor. If the module outputs multiple output tensors,
            spec characterize the space of the first output tensor.
        module (nn.Module): a nn.Module used to map the input to the output parameter space. Can be a functional
            module (FunctionalModule or FunctionalModuleWithBuffers), in which case the `forward` method will expect
            the params (and possibly) buffers keyword arguments.
        in_keys (iterable of str): keys to be read from input tensordict and passed to the module. If it
            contains more than one element, the values will be passed in the order given by the in_keys iterable.
        out_keys (iterable of str): keys to be written to the input tensordict. The length of out_keys must match the
            number of tensors returned by the embedded module.
        safe (bool): if True, the value of the output is checked against the input spec. Out-of-domain sampling can
            occur because of exploration policies or numerical under/overflow issues.
            If this value is out of bounds, it is projected back onto the desired space using the `TensorSpec.project`
            method. Default is `False`.

    Embedding a neural network in a TDModule only requires to specify the input and output keys. The domain spec can
        be passed along if needed. TDModule support functional and regular `nn.Module` objects. In the functional
        case, the 'params' (and 'buffers') keyword argument must be specified:

    Examples:
        >>> from torchrl.data import TensorDict, NdUnboundedContinuousTensorSpec
        >>> from torchrl.modules import TDModule
        >>> import torch, functorch
        >>> td = TensorDict({"input": torch.randn(3, 4), "hidden": torch.randn(3, 8)}, [3,])
        >>> spec = NdUnboundedContinuousTensorSpec(8)
        >>> module = torch.nn.GRUCell(4, 8)
        >>> fmodule, params, buffers = functorch.make_functional_with_buffers(module)
        >>> td_fmodule = TDModule(
        ...    spec=spec,
        ...    module=fmodule,
        ...    in_keys=["input", "hidden"],
        ...    out_keys=["output"],
        ...    )
        >>> td_functional = td_fmodule(td.clone(), params=params, buffers=buffers)
        >>> print(td_functional)
        TensorDict(
            fields={input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 8]), dtype=torch.float32),
                output: Tensor(torch.Size([3, 8]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([3]),
            device=cpu)

    In the stateful case:
        >>> td_module = TDModule(
        ...    spec=spec,
        ...    module=module,
        ...    in_keys=["input", "hidden"],
        ...    out_keys=["output"],
        ...    )
        >>> td_stateful = td_module(td.clone())
        >>> print(td_stateful)
        TensorDict(
            fields={input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 8]), dtype=torch.float32),
                output: Tensor(torch.Size([3, 8]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([3]),
            device=cpu)

    One can use a vmap operator to call the functional module. In this case the tensordict is expanded to match the
    batch size (i.e. the tensordict isn't modified in-place anymore):
        >>> # Model ensemble using vmap
        >>> params_repeat = tuple(param.expand(4, *param.shape).contiguous().normal_() for param in params)
        >>> buffers_repeat = tuple(param.expand(4, *param.shape).contiguous().normal_() for param in buffers)
        >>> td_vmap = td_fmodule(td.clone(), params=params_repeat, buffers=buffers_repeat, vmap=True)
        >>> print(td_vmap)
        TensorDict(
            fields={input: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([4, 3, 8]), dtype=torch.float32),
                output: Tensor(torch.Size([4, 3, 8]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([4, 3]),
            device=cpu)

    """

    def __init__(
        self,
        spec: Optional[TensorSpec],
        module: Union[
            FunctionalModule, FunctionalModuleWithBuffers, TDModule, nn.Module
        ],
        in_keys: Iterable[str],
        out_keys: Iterable[str],
        safe: bool = False,
    ):

        super().__init__()

        if not out_keys:
            raise RuntimeError(f"out_keys were not passed to {self.__class__.__name__}")
        if not in_keys:
            raise RuntimeError(f"in_keys were not passed to {self.__class__.__name__}")
        self.out_keys = out_keys
        self.in_keys = in_keys

        self._spec = spec
        self.safe = safe
        if safe:
            if spec is None:
                raise RuntimeError(
                    "`TDModule(spec=None, safe=True)` is not a valid configuration as the tensor "
                    "specs are not specified"
                )
            self.register_forward_hook(_forward_hook_safe_action)

        self.module = module

    def __setattr__(self, key: str, attribute: Any) -> None:
        if key == "spec" and isinstance(attribute, TensorSpec):
            self._spec = attribute
            return
        super().__setattr__(key, attribute)

    @property
    def spec(self) -> TensorSpec:
        return self._spec

    @spec.setter
    def _spec_set(self, spec: TensorSpec) -> None:
        if not isinstance(spec, TensorSpec):
            raise RuntimeError(
                f"Trying to set an object of type {type(spec)} as a tensorspec."
            )
        self._spec = spec

    def _write_to_tensor_dict(
        self,
        tensor_dict: _TensorDict,
        tensors: List,
        tensor_dict_out: Optional[_TensorDict] = None,
        out_keys: Optional[Iterable[str]] = None,
        vmap: Optional[int] = None,
    ) -> _TensorDict:

        if out_keys is None:
            out_keys = self.out_keys
        if (
            (tensor_dict_out is None)
            and vmap
            and (isinstance(vmap, bool) or vmap[-1] is None)
        ):
            dim = tensors[0].shape[0]
            shape = [dim, *tensor_dict.shape]
            tensor_dict_out = TensorDict(
                {key: val.expand(dim, *val.shape) for key, val in tensor_dict.items()},
                shape,
            )
        elif tensor_dict_out is None:
            tensor_dict_out = tensor_dict
        for _out_key, _tensor in zip(out_keys, tensors):
            tensor_dict_out.set(_out_key, _tensor)
        return tensor_dict_out

    def _make_vmap(self, kwargs, n_input):
        if "vmap" in kwargs and kwargs["vmap"]:
            if not isinstance(kwargs["vmap"], (tuple, bool)):
                raise RuntimeError(
                    "vmap argument must be a boolean or a tuple of dim expensions."
                )
            _buffers = "buffers" in kwargs
            _vmap = (
                kwargs["vmap"]
                if isinstance(kwargs["vmap"], tuple)
                else (0, 0, *(None,) * n_input)
                if _buffers
                else (0, *(None,) * n_input)
            )
            return _vmap

    def _call_module(
        self, tensors: Sequence[Tensor], **kwargs
    ) -> Union[Tensor, Sequence[Tensor]]:
        err_msg = "Did not find the {0} keyword argument to be used with the functional module."
        if isinstance(self.module, (FunctionalModule, FunctionalModuleWithBuffers)):
            _vmap = self._make_vmap(kwargs, len(tensors))
            if _vmap:
                module = vmap(self.module, _vmap)
            else:
                module = self.module

        if isinstance(self.module, FunctionalModule):
            if "params" not in kwargs:
                raise KeyError(err_msg.format("params"))
            kwargs_pruned = {
                key: item
                for key, item in kwargs.items()
                if key not in ("params", "vmap")
            }
            return module(kwargs["params"], *tensors, **kwargs_pruned)

        elif isinstance(self.module, FunctionalModuleWithBuffers):
            if "params" not in kwargs:
                raise KeyError(err_msg.format("params"))
            if "buffers" not in kwargs:
                raise KeyError(err_msg.format("buffers"))

            kwargs_pruned = {
                key: item
                for key, item in kwargs.items()
                if key not in ("params", "buffers", "vmap")
            }
            return module(
                kwargs["params"], kwargs["buffers"], *tensors, **kwargs_pruned
            )
        else:
            out = self.module(*tensors, **kwargs)
        return out

    def forward(
        self,
        tensor_dict: _TensorDict,
        tensor_dict_out: Optional[_TensorDict] = None,
        **kwargs,
    ) -> _TensorDict:
        tensors = tuple(tensor_dict.get(in_key) for in_key in self.in_keys)
        tensors = self._call_module(tensors, **kwargs)
        tensors = (tensors,)
        tensor_dict_out = self._write_to_tensor_dict(
            tensor_dict, tensors, tensor_dict_out, vmap=kwargs.get("vmap", False)
        )
        return tensor_dict_out

    def random(self, tensor_dict: _TensorDict) -> _TensorDict:
        """
        Samples a random element in the target space, irrespective of any input. If multiple output keys are present,
        only the first will be written in the input `tensordict`.

        Args:
            tensor_dict (_TensorDict): tensordict where the output value should be written.

        Returns: the original tensordict with a new/updated value for the output key.

        """
        key0 = self.out_keys[0]
        tensor_dict.set(key0, self.spec.rand(tensor_dict.batch_size))
        return tensor_dict

    def random_sample(self, tensordict: _TensorDict) -> _TensorDict:
        """
        see TDModule.random(...)

        """
        return self.random(tensordict)

    @property
    def device(self):
        for p in self.parameters():
            return p.device
        return torch.device("cpu")

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> TDModule:
        if self.spec is not None:
            self.spec = self.spec.to(dest)
        out = super().to(dest)
        return out  # type: ignore

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(module={self.module}, device={self.device})"

    def make_functional_with_buffers(self, clone: bool = False):
        """
        Transforms a stateful module in a functional module and returns its parameters and buffers.
        Unlike functorch.make_functional_with_buffers, this method supports lazy modules.

        Returns: A tuple of parameter and buffer tuples

        Examples:
            >>> from torchrl.data import NdUnboundedContinuousTensorSpec, TensorDict
            >>> lazy_module = nn.LazyLinear(4)
            >>> spec = NdUnboundedContinuousTensorSpec(18)
            >>> td_module = TDModule(spec, lazy_module, ["some_input"], ["some_output"])
            >>> _, (params, buffers) = td_module.make_functional_with_buffers()
            >>> print(params[0].shape)  # the lazy module has been initialized
            torch.Size([4, 18])
            >>> print(td_module(
            ...    TensorDict({'some_input': torch.randn(18)}, batch_size=[]),
            ...    params=params,
            ...    buffers=buffers))
            TensorDict(
                fields={
                    some_input: Tensor(torch.Size([18]), dtype=torch.float32),
                    some_output: Tensor(torch.Size([4]), dtype=torch.float32)},
                batch_size=torch.Size([]),
                device=cpu,
                is_shared=False)

        """
        if clone:
            self_copy = deepcopy(self)
        else:
            self_copy = self

        if isinstance(
            self_copy.module, (TDModule, FunctionalModule, FunctionalModuleWithBuffers)
        ):
            raise RuntimeError(
                "TDModule.make_functional_with_buffers requires the module to be a regular nn.Module. "
                f"Found type {type(self_copy.module)}"
            )

        # check if there is a non-initialized lazy module
        for m in self_copy.module.modules():
            if hasattr(m, "has_uninitialized_params") and m.has_uninitialized_params():
                pseudo_input = self_copy.spec.rand()
                self_copy.module(pseudo_input)
                break

        fmodule, params, buffers = functorch.make_functional_with_buffers(
            self_copy.module
        )
        self_copy.module = fmodule

        # Erase meta params
        none_state = [None for _ in params + buffers]
        _swap_state(fmodule.stateless_model, fmodule.split_names, none_state)

        return self_copy, (params, buffers)


class ProbabilisticTDModule(TDModule):
    """
    A probabilistic TD Module.
    ProbabilisticTDModule is a special case of a TDModule where the output is sampled given some rule, specified by
    the input `default_interaction_mode` argument and the `exploration_mode()` global function.

    A ProbabilisticTDModule instance has two main features:
    - It reads and writes TensorDict objects
    - It uses a real mapping R^n -> R^m to create a distribution in R^d from which values can be sampled or computed.
    When the __call__ / forward method is called, a distribution is created, and a value computed (using the 'mean',
    'mode', 'median' attribute or the 'rsample', 'sample' method).

    By default, ProbabilisticTDModule distribution class is a Delta distribution, making ProbabilisticTDModule a
    simple wrapper around a deterministic mapping function (i.e. it can be used interchangeably with its parent
    TDModule).

    Args:
        spec (TensorSpec): specs of the first output tensor. Used when calling td_module.random() to generate random
            values in the target space.
        module (nn.Module): a nn.Module used to map the input to the output parameter space. Can be a functional
            module (FunctionalModule or FunctionalModuleWithBuffers), in which case the `forward` method will expect
            the params (and possibly) buffers keyword arguments.
        in_keys (iterable of str): keys to be read from input tensordict and passed to the module. If it
            contains more than one element, the values will be passed in the order given by the in_keys iterable.
        out_keys (iterable of str): keys to be written to the input tensordict. The length of out_keys must match the
            number of tensors returned by the distribution sampling method plus the extra tensors returned by the
            module.
        distribution_class (Type, optional): a torch.distributions.Distribution class to be used for sampling.
            Default is Delta.
        distribution_kwargs (dict, optional): kwargs to be passed to the distribution.
        default_interaction_mode (str, optional): default method to be used to retrieve the output value. Should be one of:
            'mode', 'median', 'mean' or 'random' (in which case the value is sampled randomly from the distribution).
            Default is 'mode'.
            Note: When a sample is drawn, the `ProbabilisticTDModule` instance will fist look for the interaction mode
            dictated by the `exploration_mode()` global function. If this returns `None` (its default value),
            then the `default_interaction_mode` of the `ProbabilisticTDModule` instance will be used.
            Note that DataCollector instances will use `set_exploration_mode` to `"random"` by default.
        return_log_prob (bool, optional): if True, the log-probability of the distribution sample will be written in the
            tensordict with the key `f'{in_keys[0]}_log_prob'`. Default is `False`.
        safe (bool, optional): if True, the value of the sample is checked against the input spec. Out-of-domain sampling can
            occur because of exploration policies or numerical under/overflow issues. As for the `spec` argument,
            this check will only occur for the distribution sample, but not the other tensors returned by the input
            module. If the sample is out of bounds, it is projected back onto the desired space using the
            `TensorSpec.project`
            method.
            Default is `False`.
        save_dist_params (bool, optional): if True, the parameters of the distribution (i.e. the output of the module)
            will be written to the tensordict along with the sample. Those parameters can be used to
            re-compute the original distribution later on (e.g. to compute the divergence between the distribution
            used to sample the action and the updated distribution in PPO).
            Default is `False`.
        cache_dist (bool, optional): if True, the parameters of the distribution (i.e. the output of the module)
            will be written to the tensordict along with the sample. Those parameters can be used to
            re-compute the original distribution later on (e.g. to compute the divergence between the distribution
            used to sample the action and the updated distribution in PPO).
            Default is `False`.

    Examples:
        >>> from torchrl.data import TensorDict, NdUnboundedContinuousTensorSpec
        >>> from torchrl.modules import ProbabilisticTDModule, TanhNormal
        >>> import functorch, torch
        >>> td = TensorDict({"input": torch.randn(3, 4), "hidden": torch.randn(3, 8)}, [3,])
        >>> spec = NdUnboundedContinuousTensorSpec(4)
        >>> module = torch.nn.GRUCell(4, 8)
        >>> module_func, params, buffers = functorch.make_functional_with_buffers(module)
        >>> td_module = ProbabilisticTDModule(
        ...    spec=spec,
        ...    module=module_func,
        ...    in_keys=["input"],
        ...    out_keys=["output"],
        ...    distribution_class=TanhNormal,
        ...    return_log_prob=True,
        ...    )
        >>> _ = td_module(td, params=params, buffers=buffers)
        >>> print(td)
        TensorDict(
            fields={
                input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 8]), dtype=torch.float32),
                output: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                output_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
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
                input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 8]), dtype=torch.float32),
                output: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                output_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=cpu,
            is_shared=False)

    """

    def __init__(
        self,
        spec: TensorSpec,
        module: Union[Callable[[Tensor], Tensor], nn.Module],
        in_keys: Iterable[str],
        out_keys: Iterable[str],
        distribution_class: Type = Delta,
        distribution_kwargs: Optional[dict] = None,
        default_interaction_mode: str = "mode",
        _n_empirical_est: int = 1000,
        return_log_prob: bool = False,
        safe: bool = False,
        save_dist_params: bool = False,
        cache_dist: bool = True,
    ):

        super().__init__(
            spec=spec, module=module, out_keys=out_keys, in_keys=in_keys, safe=safe
        )

        self.save_dist_params = save_dist_params
        self._n_empirical_est = _n_empirical_est
        self.cache_dist = cache_dist
        self._dist = None

        if isinstance(distribution_class, str):
            distribution_class = distributions_maps.get(distribution_class.lower())
        self.distribution_class = distribution_class
        self.distribution_kwargs = (
            distribution_kwargs if distribution_kwargs is not None else dict()
        )
        self.return_log_prob = return_log_prob

        self.default_interaction_mode = default_interaction_mode
        self.interact = False

    def get_dist(
        self,
        tensor_dict: _TensorDict,
        **kwargs,
    ) -> Tuple[torch.distributions.Distribution, ...]:
        """
        Calls the module using the tensors retrieved from the 'in_keys' attribute and returns a distribution
        using its output.

        Args:
            tensor_dict (_TensorDict): tensordict with the input values for the creation of the distribution.

        Returns: a distribution along with other tensors returned by the module.

        """
        tensors = [tensor_dict.get(key, None) for key in self.in_keys]
        out_tensors = self._call_module(tensors, **kwargs)
        if isinstance(out_tensors, Tensor):
            out_tensors = (out_tensors,)
        if self.save_dist_params:
            for i, _tensor in enumerate(out_tensors):
                tensor_dict.set(f"{self.out_keys[0]}_dist_param_{i}", _tensor)
        dist, num_params = self.build_dist_from_params(out_tensors)
        tensors = out_tensors[num_params:]

        return (dist, *tensors)

    def build_dist_from_params(
        self, params: Tuple[Tensor, ...]
    ) -> Tuple[d.Distribution, int]:
        """
        Given a tuple of temsors, returns a distribution object and the number of parameters used for it.

        Args:
            params (Tuple[Tensor, ...]): tensors to be used for the distribution construction.

        Returns: a distribution object and the number of parameters used for its construction.

        """
        num_params = (
            getattr(self.distribution_class, "num_params")
            if hasattr(self.distribution_class, "num_params")
            else 1
        )
        if self.cache_dist and self._dist is not None:
            self._dist.update(*params[:num_params])
            dist = self._dist
        else:
            dist = self.distribution_class(
                *params[:num_params], **self.distribution_kwargs
            )
            if self.cache_dist:
                self._dist = dist
        return dist, num_params

    def forward(
        self,
        tensor_dict: _TensorDict,
        tensor_dict_out: Optional[_TensorDict] = None,
        **kwargs,
    ) -> _TensorDict:

        dist, *tensors = self.get_dist(tensor_dict, **kwargs)
        out_tensor = self._dist_sample(dist, interaction_mode=exploration_mode())
        tensor_dict_out = self._write_to_tensor_dict(
            tensor_dict,
            [out_tensor] + list(tensors),
            tensor_dict_out,
            vmap=kwargs.get("vmap", 0),
        )
        if self.return_log_prob:
            log_prob = dist.log_prob(out_tensor)
            tensor_dict_out.set("_".join([self.out_keys[0], "log_prob"]), log_prob)
        return tensor_dict_out

    def log_prob(self, tensor_dict: _TensorDict, **kwargs) -> _TensorDict:
        """
        Samples/computes an action using the module and writes this value onto the input tensordict along
        with its log-probability.

        Args:
            tensor_dict (_TensorDict): tensordict containing the in_keys specified in the initializer.

        Returns:
            the same tensordict with the out_keys values added/updated as well as a
                f"{out_keys[0]}_log_prob" key containing the log-probability of the first output.

        """
        dist, *_ = self.get_dist(tensor_dict, **kwargs)
        lp = dist.log_prob(tensor_dict.get(self.out_keys[0]))
        tensor_dict.set(self.out_keys[0] + "_log_prob", lp)
        return tensor_dict

    def _dist_sample(
        self, dist: d.Distribution, interaction_mode: bool = None, eps: Number = None
    ) -> Tensor:
        if interaction_mode is None:
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
            except:
                if dist.has_rsample:
                    return dist.rsample((self._n_empirical_est,)).mean(0)
                else:
                    return dist.sample((self._n_empirical_est,)).mean(0)

        elif interaction_mode == "random":
            if dist.has_rsample:
                return dist.rsample()
            else:
                return dist.sample()
        else:
            raise NotImplementedError(f"unknown interaction_mode {interaction_mode}")

    @property
    def device(self):
        for p in self.parameters():
            return p.device
        return torch.device("cpu")

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> ProbabilisticTDModule:
        if self.spec is not None:
            self.spec = self.spec.to(dest)
        out = super().to(dest)
        return out

    def __deepcopy__(self, memodict={}):
        self._dist = None
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memodict))
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(module={self.module}, distribution_class={self.distribution_class}, device={self.device})"


class TDSequence(TDModule):
    """
    A sequence of TDModules.
    Similarly to `nn.Sequence` which passes a tensor through a chain of mappings that read and write a single tensor
    each, this module will read and write over a tensordict by querying each of the input modules.
    When calling a `TDSequence` instance with a functional module, it is expected that the parameter lists (and
    buffers) will be concatenated in a single list.

    Args:
         modules (iterable of TDModules): ordered sequence of TDModule instances to be run sequentially.

    TDSequence supportse functional, modular and vmap coding:
    Examples:
        >>> from torchrl.data import TensorDict, NdUnboundedContinuousTensorSpec
        >>> from torchrl.modules import ProbabilisticTDModule, TanhNormal, TDSequence
        >>> import torch, functorch
        >>> td = TensorDict({"input": torch.randn(3, 4)}, [3,])
        >>> spec1 = NdUnboundedContinuousTensorSpec(4)
        >>> module1 = torch.nn.Linear(4, 8)
        >>> fmodule1, params1, buffers1 = functorch.make_functional_with_buffers(module1)
        >>> td_module1 = ProbabilisticTDModule(
        ...    spec=spec1,
        ...    module=fmodule1,
        ...    in_keys=["input"],
        ...    out_keys=["hidden"],
        ...    distribution_class=TanhNormal,
        ...    return_log_prob=True,
        ...    )
        >>> spec2 = NdUnboundedContinuousTensorSpec(8)
        >>> module2 = torch.nn.Linear(4, 8)
        >>> fmodule2, params2, buffers2 = functorch.make_functional_with_buffers(module2)
        >>> td_module2 = TDModule(
        ...    spec=spec2,
        ...    module=fmodule2,
        ...    in_keys=["hidden"],
        ...    out_keys=["output"],
        ...    )
        >>> td_module = TDSequence(td_module1, td_module2)
        >>> params = params1 + params2
        >>> buffers = buffers1 + buffers2
        >>> _ = td_module(td, params=params, buffers=buffers)
        >>> print(td)
        TensorDict(
            fields={input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32),
                output: Tensor(torch.Size([3, 8]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([3]),
            device=cpu)

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


    """

    def __init__(
        self,
        *modules: TDModule,
    ):
        in_keys_tmp = []
        out_keys = []
        for module in modules:
            in_keys_tmp += module.in_keys
            out_keys += module.out_keys
        in_keys = []
        for in_key in in_keys_tmp:
            if (in_key not in in_keys) and (in_key not in out_keys):
                in_keys.append(in_key)
        if not len(in_keys):
            raise RuntimeError(
                f"in_keys empty. Please ensure that there is at least one input key that is not part "
                f"of the output key set."
            )
        out_keys = [
            out_key
            for i, out_key in enumerate(out_keys)
            if out_key not in out_keys[i + 1 :]
        ]

        super().__init__(
            spec=None,
            module=nn.ModuleList(list(modules)),
            in_keys=in_keys,
            out_keys=out_keys,
        )

    @property
    def param_len(self) -> List[int]:
        param_list = []
        prev = 0
        for module in self.module:
            param_list.append(len(module.module.param_names) + prev)
            prev = param_list[-1]
        return param_list

    @property
    def buffer_len(self) -> List[int]:
        buffer_list = []
        prev = 0
        for module in self.module:
            buffer_list.append(len(module.module.buffer_names) + prev)
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

    def forward(self, tensor_dict: _TensorDict, **kwargs) -> _TensorDict:
        if "params" in kwargs and "buffers" in kwargs:
            param_splits = self._split_param(kwargs["params"], "params")
            buffer_splits = self._split_param(kwargs["buffers"], "buffers")
            kwargs_pruned = {
                key: item
                for key, item in kwargs.items()
                if key not in ("params", "buffers")
            }
            for i, (module, param, buffer) in enumerate(zip(self.module, param_splits, buffer_splits)):  # type: ignore
                if "vmap" in kwargs_pruned and i > 0:
                    # the tensordict is already expended
                    kwargs_pruned["vmap"] = (0, 0, *(0,) * len(module.in_keys))
                tensor_dict = module(
                    tensor_dict, params=param, buffers=buffer, **kwargs_pruned
                )

        elif "params" in kwargs:
            param_splits = self._split_param(kwargs["params"], "params")
            kwargs_pruned = {
                key: item for key, item in kwargs.items() if key not in ("params",)
            }
            for i, (module, param) in enumerate(zip(self.module, param_splits)):  # type: ignore
                if "vmap" in kwargs_pruned and i > 0:
                    # the tensordict is already expended
                    kwargs_pruned["vmap"] = (0, *(0,) * len(module.in_keys))
                tensor_dict = module(tensor_dict, params=param, **kwargs_pruned)

        elif not len(kwargs):
            for module in self.module:  # type: ignore
                tensor_dict = module(tensor_dict)
        else:
            raise RuntimeError(
                "TDSequence does not support keyword arguments other than 'params', 'buffers' and 'vmap'"
            )

        return tensor_dict

    def __len__(self):
        return len(self.module)  # type: ignore

    @property
    def spec(self):
        kwargs = {}
        for layer in self.module:  # type: ignore
            out_key = layer.out_keys[0]
            if not isinstance(layer.spec, TensorSpec):
                raise RuntimeError(
                    f"TDSequence.spec requires all specs to be valid TensorSpec objects. Got "
                    f"{type(layer.spec)}"
                )
            kwargs[out_key] = layer.spec
        return CompositeSpec(**kwargs)

    def make_functional_with_buffers(self, clone: bool = False):
        """
        Transforms a stateful module in a functional module and returns its parameters and buffers.
        Unlike functorch.make_functional_with_buffers, this method supports lazy modules.

        Returns: A tuple of parameter and buffer tuples

        Examples:
            >>> from torchrl.data import NdUnboundedContinuousTensorSpec, TensorDict
            >>> lazy_module1 = nn.LazyLinear(4)
            >>> lazy_module2 = nn.LazyLinear(3)
            >>> spec1 = NdUnboundedContinuousTensorSpec(18)
            >>> spec2 = NdUnboundedContinuousTensorSpec(4)
            >>> td_module1 = TDModule(spec1, lazy_module1, ["some_input"], ["hidden"])
            >>> td_module2 = TDModule(spec2, lazy_module2, ["hidden"], ["some_output"])
            >>> td_module = TDSequence(td_module1, td_module2)
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
            self_copy = copy(self)
            self_copy.module = copy(self_copy.module)
        else:
            self_copy = self
        params = []
        buffers = []
        for i, module in enumerate(self.module):  # type: ignore
            self_copy.module[i], (
                _params,
                _buffers,
            ) = module.make_functional_with_buffers()
            params.extend(_params)
            buffers.extend(_buffers)
        return self_copy, (params, buffers)


class TDModuleWrapper(nn.Module):
    """
    Wrapper calss for TDModule objects.
    Once created, a TDModuleWrapper will behave exactly as the TDModule it contains except for the methods that are
    overwritten.

    Args:
        probabilistic_operator (TDModule): operator to be wrapped.

    Examples:
        >>> #     This class can be used for exploration wrappers
        >>> import functorch
        >>> from torchrl.modules import TDModuleWrapper, TDModule
        >>> from torchrl.data import TensorDict, NdUnboundedContinuousTensorSpec, expand_as_right
        >>> import torch
        >>>
        >>> class EpsilonGreedyExploration(TDModuleWrapper):
        ...     eps = 0.5
        ...     def forward(self, tensordict, params, buffers):
        ...         rand_output_clone = self.random(tensordict.clone())
        ...         det_output_clone = self.td_module(tensordict.clone(), params, buffers)
        ...         rand_output_idx = torch.rand(tensordict.shape, device=rand_output_clone.device) < self.eps
        ...         for key in self.out_keys:
        ...             _rand_output = rand_output_clone.get(key)
        ...             _det_output =  det_output_clone.get(key)
        ...             rand_output_idx_expand = expand_as_right(rand_output_idx, _rand_output).to(_rand_output.dtype)
        ...             tensordict.set(key,
        ...                 rand_output_idx_expand * _rand_output + (1-rand_output_idx_expand) * _det_output)
        ...         return tensordict
        >>>
        >>> td = TensorDict({"input": torch.zeros(10, 4)}, [10])
        >>> module = torch.nn.Linear(4, 4, bias=False)  # should return a zero tensor if input is a zero tensor
        >>> fmodule, params, buffers = functorch.make_functional_with_buffers(module)
        >>> spec = NdUnboundedContinuousTensorSpec(4)
        >>> tdmodule = TDModule(spec=spec, module=fmodule, in_keys=["input"], out_keys=["output"])
        >>> tdmodule_wrapped = EpsilonGreedyExploration(tdmodule)
        >>> tdmodule_wrapped(td, params=params, buffers=buffers)
        >>> print(td.get("output"))
    """

    def __init__(self, probabilistic_operator: TDModule):
        super().__init__()
        self.td_module = probabilistic_operator
        if len(self.td_module._forward_hooks):
            for pre_hook in self.td_module._forward_hooks:
                self.register_forward_hook(self.td_module._forward_hooks[pre_hook])

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except:
            if name not in self.__dict__:
                return getattr(self._modules["td_module"], name)
            else:
                raise AttributeError(
                    f"attribute {name} not recognised in {type(self).__name__}"
                )

    def forward(self, *args, **kwargs):
        return self.td_module.forward(*args, **kwargs)
