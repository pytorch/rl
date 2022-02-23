from __future__ import annotations

from numbers import Number
from typing import Tuple, List, Iterable, Type, Optional, Union, Any, Callable

import torch
from torch import nn, distributions as d, Tensor

from torchrl.data import TensorSpec, DEVICE_TYPING, CompositeSpec
from torchrl.data.tensordict.tensordict import _TensorDict
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
    """
    A TDModule, for TensorDict module, is a python wrapper around a `nn.Module` that reads and writes to a
    TensorDict, instead of reading and returning tensors.

    Args:
        spec (TensorSpec): specs of the output tensor. If the module outputs multiple output tensors,
            spec characterize the space of the first output tensor.
        module (Callable or nn.Module): callable used to map the input to the output parameter space.
        in_keys (iterable of str): keys to be read from input tensordict and passed to the module. If it
            contains more than one element, the values will be passed in the order given by the in_keys iterable.
        out_keys (iterable of str): keys to be written to the input tensordict. The length of out_keys must match the
            number of tensors returned by the embedded module.
        safe (bool): if True, the value of the output is checked against the input spec. Out-of-domain sampling can
            occur because of exploration policies or numerical under/overflow issues.
            If this value is out of bounds, it is projected back onto the desired space using the `TensorSpec.project`
            method. Default is `False`.
    Example:
        >>> from torchrl.data import TensorDict, NdUnboundedContinuousTensorSpec
        >>> from torchrl.modules import TDModule
        >>> import torch
        >>> td = TensorDict({"input": torch.randn(3, 4)}, [3,])
        >>> spec = NdUnboundedContinuousTensorSpec(4)
        >>> module = torch.nn.Linear(4, 4)
        >>> td_module = TDModule(
        ...    spec=spec,
        ...    module=module,
        ...    in_keys=["input"],
        ...    out_keys=["output"],
        ...    )
        >>> td_module(td)
        >>> print(td)
        >>> print(td.get("output"))

    """

    def __init__(
        self,
        spec: Optional[TensorSpec],
        module: Union[Callable[[Tensor], Tensor], nn.Module],
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

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "spec" and isinstance(value, TensorSpec):
            self._spec = value
            return
        return super().__setattr__(key, value)

    def _write_to_tensor_dict(
        self, tensor_dict: _TensorDict, tensors: List, out_keys: Iterable[str] = None
    ) -> _TensorDict:
        if out_keys is None:
            out_keys = self.out_keys
        for _out_key, _tensor in zip(out_keys, tensors):
            tensor_dict.set(_out_key, _tensor)
        return tensor_dict

    def forward(self, tensor_dict: _TensorDict) -> _TensorDict:
        tensor_dict_unsqueezed = tensor_dict
        unsqueeze = False
        if not len(tensor_dict.batch_size):
            unsqueeze = True
            tensor_dict_unsqueezed = tensor_dict.unsqueeze(-1)
        tensors = tuple(tensor_dict_unsqueezed.get(in_key) for in_key in self.in_keys)
        tensors = self.module(*tensors)
        if isinstance(tensors, Tensor):
            tensors = (tensors,)
        self._write_to_tensor_dict(tensor_dict_unsqueezed, tensors)
        if unsqueeze:
            tensor_dict = tensor_dict_unsqueezed.squeeze(-1)
        return tensor_dict

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
        module (Callable or nn.Module): callable used to map the input to the output parameter space.
        in_keys (iterable of str): keys to be read from input tensordict and passed to the module. If it
            contains more than one element, the values will be passed in the order given by the in_keys iterable.
        out_keys (iterable of str): keys to be written to the input tensordict. The length of out_keys must match the
            number of tensors returned by the distribution sampling method plus the extra tensors returned by the
            module.
        distribution_class (Type): a torch.distributions.Distribution class to be used for sampling.
        distribution_kwargs (dict, optional): kwargs to be passed to the distribution.
        default_interaction_mode (str): default method to be used to retrieve the output value. Should be one of:
            'mode', 'median', 'mean' or 'random' (in which case the value is sampled randomly from the distribution).
            Default is 'mode'.
            Note: When a sample is drawn, the `ProbabilisticTDModule` instance will fist look for the interaction mode
            dictated by the `exploration_mode()` global function. If this returns `None` (its default value),
            then the `default_interaction_mode` of the `ProbabilisticTDModule` instance will be used.
            Note that DataCollector instances will use `set_exploration_mode` to `"random"` by default.
        return_log_prob (bool): if True, the log-probability of the distribution sample will be written in the
            tensordict with the key `f'{in_keys[0]}_log_prob'`. Default is `False`.
        safe (bool): if True, the value of the sample is checked against the input spec. Out-of-domain sampling can
            occur because of exploration policies or numerical under/overflow issues. As for the `spec` argument,
            this check will only occur for the distribution sample, but not the other tensors returned by the input
            module. If the sample is out of bounds, it is projected back onto the desired space using the
            `TensorSpec.project`
            method.
            Default is False.
        save_dist_params (bool): if True, the parameters of the distribution (i.e. the output of the module)
            will be written to the tensordict along with the sample. Those parameters can be used to
            re-compute the original distribution later on (e.g. to compute the divergence between the distribution
            used to sample the action and the updated distribution in PPO).
            Default is False.
    Example:
        >>> from torchrl.data import TensorDict, NdUnboundedContinuousTensorSpec
        >>> from torchrl.modules import ProbabilisticTDModule, TanhNormal
        >>> import torch
        >>> td = TensorDict({"input": torch.randn(3, 4)}, [3,])
        >>> spec = NdUnboundedContinuousTensorSpec(4)
        >>> module = torch.nn.Linear(4, 8)
        >>> td_module = ProbabilisticTDModule(
        ...    spec=spec,
        ...    module=module,
        ...    in_keys=["input"],
        ...    out_keys=["output"],
        ...    distribution_class=TanhNormal,
        ...    return_log_prob=True,
        ...    )
        >>> td_module(td)
        >>> print('output: ', td.get("output"))
        >>> print('output log-prob: ', td.get("output_log_prob"))
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
    ):

        super().__init__(
            spec=spec, module=module, out_keys=out_keys, in_keys=in_keys, safe=safe
        )

        self.save_dist_params = save_dist_params
        self._n_empirical_est = _n_empirical_est

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
        self, tensor_dict: _TensorDict
    ) -> Tuple[torch.distributions.Distribution, ...]:
        """
        Calls the module using the tensors retrieved from the 'in_keys' attribute and returns a distribution
        using its output.

        Args:
            tensor_dict (_TensorDict): tensordict with the input values for the creation of the distribution.

        Returns: a distribution along with other tensors returned by the module.

        """
        tensors = [tensor_dict.get(key, None) for key in self.in_keys]
        out_tensors = self.module(*tensors)
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
        dist = self.distribution_class(*params[:num_params], **self.distribution_kwargs)
        return dist, num_params

    def forward(self, tensor_dict: _TensorDict) -> _TensorDict:
        tensor_dict_unsqueezed = tensor_dict
        if not len(tensor_dict.batch_size):
            tensor_dict_unsqueezed = tensor_dict.unsqueeze(0)
        dist, *tensors = self.get_dist(tensor_dict_unsqueezed)
        out_tensor = self._dist_sample(dist, interaction_mode=exploration_mode())
        self._write_to_tensor_dict(tensor_dict_unsqueezed, [out_tensor] + list(tensors))
        if self.return_log_prob:
            log_prob = dist.log_prob(out_tensor)
            tensor_dict_unsqueezed.set(
                "_".join([self.out_keys[0], "log_prob"]), log_prob
            )
        return tensor_dict

    def log_prob(self, tensor_dict: _TensorDict) -> _TensorDict:
        """
        Samples/computes an action using the module and writes this value onto the input tensordict along
        with its log-probability.

        Args:
            tensor_dict (_TensorDict): tensordict containing the in_keys specified in the initializer.

        Returns:
            the same tensordict with the out_keys values added/updated as well as a
                f"{out_keys[0]}_log_prob" key containing the log-probability of the first output.

        """
        dist, *_ = self.get_dist(tensor_dict)
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(module={self.module}, distribution_class={self.distribution_class}, device={self.device})"


class TDSequence(TDModule):
    """
    A sequence of TDModules.
    Similarly to `nn.Sequence` which passes a tensor through a chain of mappings that read and write a single tensor
    each, this module will read and write over a tensordict by querying each of the input modules.

    Args:
         modules (iterable of TDModules): ordered sequence of TDModule instances to be run sequentially.

    Examples:
        >>> from torchrl.data import TensorDict, NdUnboundedContinuousTensorSpec
        >>> from torchrl.modules import ProbabilisticTDModule, TanhNormal, TDSequence
        >>> import torch
        >>> td = TensorDict({"input": torch.randn(3, 4)}, [3,])
        >>> spec1 = NdUnboundedContinuousTensorSpec(4)
        >>> module1 = torch.nn.Linear(4, 8)
        >>> td_module1 = ProbabilisticTDModule(
        ...    spec=spec1,
        ...    module=module1,
        ...    in_keys=["input"],
        ...    out_keys=["hidden"],
        ...    distribution_class=TanhNormal,
        ...    return_log_prob=True,
        ...    )
        >>> spec2 = NdUnboundedContinuousTensorSpec(8)
        >>> module2 = torch.nn.Linear(4, 8)
        >>> td_module2 = TDModule(
        ...    spec=spec2,
        ...    module=module2,
        ...    in_keys=["hidden"],
        ...    out_keys=["output"],
        ...    )
        >>> td_module = TDSequence(td_module1, td_module2)
        >>> _ = td_module(td)
        >>> print(td)
        TensorDict(
            fields={input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32),
                output: Tensor(torch.Size([3, 8]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([3]),
            device=cpu)
        >>> print(td_module.spec)
        CompositeSpec(
            hidden: NdUnboundedContinuousTensorSpec(shape=torch.Size([4]), space=None, device=device(type='cpu'),
                dtype=torch.float32, domain='continuous'),
            output: NdUnboundedContinuousTensorSpec(shape=torch.Size([8]), space=None, device=device(type='cpu'),
                dtype=torch.float32, domain='continuous'))


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

    def forward(self, tensor_dict: _TensorDict) -> _TensorDict:
        for module in self.module:  # type: ignore
            tensor_dict = module(tensor_dict)
        return tensor_dict

    def __len__(self):
        return len(self.module)  # type: ignore

    @property
    def spec(self):
        kwargs = {}
        for layer in self.module:
            out_key = layer.out_keys[0]
            kwargs[out_key] = layer.spec
        return CompositeSpec(**kwargs)


class TDModuleWrapper(nn.Module):
    """
    Wrapper calss for TDModule objects.
    Once created, a TDModuleWrapper will behave exactly as the TDModule it contains except for the methods that are
    overwritten.

    Args:
        probabilistic_operator (TDModule): operator to be wrapped.

    Examples:
        This class can be used for exploration wrappers
        >>> from torchrl.modules import TDModuleWrapper, TDModule
        >>> from torchrl.data import TensorDict, NdUnboundedContinuousTensorSpec, expand_as_right
        >>> import torch
        >>>
        >>> class EpsilonGreedyExploration(TDModuleWrapper):
        ...     eps = 0.1
        ...     def forward(self, tensordict):
        ...         rand_output_clone = self.random(tensordict.clone())
        ...         det_output_clone = self.td_module(tensordict.clone())
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
        >>> spec = NdUnboundedContinuousTensorSpec(4)
        >>> tdmodule = TDModule(spec=spec, module=module, in_keys=["input"], out_keys=["output"])
        >>> tdmodule_wrapped = EpsilonGreedyExploration(tdmodule)
        >>> tdmodule_wrapped(td)
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

    forward = TDModule.forward
