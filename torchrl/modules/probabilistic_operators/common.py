from numbers import Number
from typing import Tuple, List, Iterable, Type, Optional, Union, Any, Callable

import torch
from torch import nn, distributions as d

from torchrl.data import TensorSpec
from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.envs.utils import exploration_mode
from torchrl.modules.distributions import Delta, distributions_maps

__all__ = [
    "ProbabilisticOperator",
    "ProbabilisticOperatorWrapper",
]


def _forward_hook_safe_action(module, tensor_dict_in, tensor_dict_out):
    if not module.spec.is_in(tensor_dict_out.get(module.out_keys[0])):
        try:
            tensor_dict_out.set_(
                module.out_keys[0],
                module.spec.project(tensor_dict_out.get(module.out_keys[0]))
            )
        except RuntimeError:
            tensor_dict_out.set(
                module.out_keys[0],
                module.spec.project(tensor_dict_out.get(module.out_keys[0]))
            )


class ProbabilisticOperator(nn.Module):
    """
    A probabilistic operator.
    ProbabilisticOperator is the central class that acts as an interface between a model (e.g. a neural network) and
    an agent.
    ProbabilisticOperator is mainly used in the code to represent policies and value operators.

    A ProbabilisticOperator instance has two main features:
    - It reads and writes TensorDict objects
    - It uses a real mapping R^n -> R^m to create a distribution in R^d from which values can be sampled or computed.
    When the __call__ / forward method is called, a distribution is created, and a value computed (using the 'mean',
    'mode', 'median' attribute or the 'rsample', 'sample' method).

    By default, ProbabilisticOperator distribution class is a Delta distribution, making ProbabilisticOperator a
    simple wrapper around a deterministic mapping function.

    Args:
        spec (TensorSpec): specs of the output tensor. Used when calling prob_operator.random() to generate random
            values in the target space.
        mapping_operator (Callable or nn.Module): callable used to map the input to the output parameter space.
        out_keys (iterable of str): keys to be written to the input tensordict. The length of out_keys must match the
            number of tensors returned by the distribution sampling method.
        in_keys (iterable of str): keys to be read from input tensordict and passed to the mapping_operator. If it
            contains more than one element, the values will be passed in the order given by the in_keys iterable.
        distribution_class (Type): a torch.distributions.Distribution class to be used for sampling.
        distribution_kwargs (dict, optional): kwargs to be passed to the distribution.
        default_interaction_mode (str): default method to be used to retrieve the output value. Should be one of:
            'mode', 'median', 'mean' or 'random' (in which case the value is sampled randomly from the distribution).
            default: 'mode'
        return_log_prob (bool): if True, the log-probability of the distribution sample will be written in the
            tensordict.
            default = False
        safe (bool): if True, the value of the sample is checked against the input spec. Out-of-domain sampling can
            occur because of exploration policies or numerical under/overflow issues.
            If this value is out of bounds, it is projected back onto the desired space using the `TensorSpec.project`
            method.
            default = False
        save_dist_params (bool): if True, the parameters of the distribution (i.e. the output of the mapping_operator)
            will be written to the tensordict along with the sample. Those parameters can be used later on to
            re-compute the original distribution later on.
            default: False
    """

    def __init__(
            self,
            spec: TensorSpec,
            mapping_operator: Union[Callable[[torch.Tensor], torch.Tensor], nn.Module],
            out_keys: Iterable[str],
            in_keys: Iterable[str],
            distribution_class: Type = Delta,
            distribution_kwargs: Optional[dict] = None,
            default_interaction_mode: str = "mode",
            _n_empirical_est: int = 1000,
            return_log_prob: bool = False,
            safe: bool = False,
            save_dist_params: bool = False,
    ):

        super().__init__()

        if not out_keys:
            raise RuntimeError(f"out_keys were not passed to {self.__class__.__name__}")
        if not in_keys:
            raise RuntimeError(f"in_keys were not passed to {self.__class__.__name__}")
        self.out_keys = out_keys
        self.in_keys = in_keys

        self.spec = spec
        self.safe = safe
        if safe:
            self.register_forward_hook(_forward_hook_safe_action)
        self.save_dist_params = save_dist_params
        self._n_empirical_est = _n_empirical_est

        self.mapping_operator = mapping_operator

        if isinstance(distribution_class, str):
            distribution_class = distributions_maps.get(distribution_class.lower())
        self.distribution_class = distribution_class
        self.distribution_kwargs = distribution_kwargs if distribution_kwargs is not None else dict()
        self.return_log_prob = return_log_prob

        self.default_interaction_mode = default_interaction_mode
        self.interact = False

    def get_dist(self, tensor_dict: _TensorDict) -> Tuple[torch.distributions.Distribution, ...]:
        """
        Calls the mapping_operator using the tensors retrieved from the 'in_keys' attribute and returns a distribution
        using its output.

        Args:
            tensor_dict (_TensorDict): tensordict with the input values for the creation of the distribution.

        Returns: a distribution along with other tensors returned by the mapping_operator.

        """
        tensors = [tensor_dict.get(key) for key in self.in_keys]
        out_tensors = self.mapping_operator(*tensors)
        if isinstance(out_tensors, torch.Tensor):
            out_tensors = (out_tensors,)
        if self.save_dist_params:
            for i, _tensor in enumerate(out_tensors):
                tensor_dict.set(f"{self.out_keys[0]}_dist_param_{i}", _tensor)
        dist, num_params = self.build_dist_from_params(out_tensors)
        tensors = out_tensors[num_params:]

        return (dist, *tensors)

    def build_dist_from_params(self, params: Tuple[torch.Tensor, ...]) -> Tuple[d.Distribution, int]:
        """
        Given a tuple of temsors, returns a distribution object and the number of parameters used for it.

        Args:
            params (Tuple[torch.Tensor, ...]): tensors to be used for the distribution construction.

        Returns: a distribution object and the number of parameters used for its construction.

        """
        num_params = (
            getattr(self.distribution_class, "num_params")
            if hasattr(self.distribution_class, "num_params")
            else 1
        )
        dist = self.distribution_class(
            *params[:num_params], **self.distribution_kwargs
        )
        return dist, num_params

    def _write_to_tensor_dict(self, tensor_dict: _TensorDict, tensors: List) -> None:
        for _out_key, _tensor in zip(self.out_keys, tensors):
            tensor_dict.set(_out_key, _tensor)

    def forward(self, tensor_dict: _TensorDict) -> _TensorDict:
        tensor_dict_unsqueezed = tensor_dict
        if not len(tensor_dict.batch_size):
            tensor_dict_unsqueezed = tensor_dict.unsqueeze(0)
        dist, *tensors = self.get_dist(tensor_dict_unsqueezed)
        out_tensor = self._dist_sample(dist, interaction_mode=exploration_mode())
        self._write_to_tensor_dict(tensor_dict_unsqueezed, [out_tensor] + list(tensors))
        if self.return_log_prob:
            log_prob = dist.log_prob(out_tensor)
            log_prob = log_prob.unsqueeze(-1)
            tensor_dict_unsqueezed.set("_".join([self.out_keys[0], "log_prob"]), log_prob)
        return tensor_dict

    def random(self, tensor_dict: _TensorDict) -> _TensorDict:
        """
        Samples a random element in the target space, irrespective of any computed distribution.

        Args:
            tensor_dict (_TensorDict): tensordict where the output value should be written.

        Returns: the original tensordict with a new/updated value for the output key.

        """
        key0 = self.out_keys[0]
        tensor_dict.set(key0, self.spec.rand(tensor_dict.batch_size))
        return tensor_dict

    def random_sample(self, tensordict: _TensorDict) -> _TensorDict:
        """
        see ProbabilisticOperator.random(...)

        """
        return self.random(tensordict)

    def log_prob(self, tensor_dict: _TensorDict) -> _TensorDict:
        """
        Samples/computes an action using the mapping_operator and writes this value onto the input tensordict along
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

    def _dist_sample(self, dist: d.Distribution, interaction_mode: bool = None, eps: Number = None) -> torch.Tensor:
        if interaction_mode is None:
            interaction_mode = self.default_interaction_mode

        if not isinstance(dist, d.Distribution):
            raise TypeError(f"type {type(dist)} not recognised by _dist_sample")

        if interaction_mode == "mode":
            if hasattr(dist, "mode"):
                return dist.mode
            else:
                raise NotImplementedError(f"method {type(dist)}.mode is not implemented")

        elif interaction_mode == "median":
            if hasattr(dist, "median"):
                return dist.median
            else:
                raise NotImplementedError(f"method {type(dist)}.median is not implemented")

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
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mapping_operator={self.mapping_operator}, distribution_class={self.distribution_class}, device={self.device})"


class ProbabilisticOperatorWrapper(nn.Module):
    """
    Wrapper calss for ProbabilisticOperator objects.
    Once created, a ProbabilisticOperatorWrapper will behave exactly as the ProbabilisticOperator it contains except
    for the methods that are overwritten.

    Args:
        probabilistic_operator (ProbabilisticOperator): operator to be wrapped.

    Examples:
        This class can be used for exploration wrappers
        >>> class EpsilonGreedyExploration(ProbabilisticOperatorWrapper):
        >>>     eps = 0.1
        >>>     def forward(self, tensordict):
        >>>         if torch.rand(1)<self.eps:
        >>>             return self.random(tensordict)
        >>>         else:
        >>>             return self.probabilistic_operator(tensordict)

    """
    def __init__(self, probabilistic_operator: ProbabilisticOperator):
        super().__init__()
        self.probabilistic_operator = probabilistic_operator
        if len(self.probabilistic_operator._forward_hooks):
            for pre_hook in self.probabilistic_operator._forward_hooks:
                self.register_forward_hook(self.probabilistic_operator._forward_hooks[pre_hook])

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except:
            if name not in self.__dict__:
                return getattr(self._modules["probabilistic_operator"], name)
            else:
                raise AttributeError(
                    f"attribute {name} not recognised in {type(self).__name__}"
                )

    forward = ProbabilisticOperator.forward
