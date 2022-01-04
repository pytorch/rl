from numbers import Number
from typing import Tuple, List, Iterable, Type, Optional, Union, Any

import torch
from torch import nn, distributions as d

from torchrl.data import TensorSpec
from torchrl.data.tensordict.tensordict import _TensorDict
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

    def __init__(
            self,
            spec: TensorSpec,
            mapping_operator: nn.Module,
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

        assert out_keys, f"out_keys were not passed to {self.__class__.__name__}"
        assert in_keys, f"in_keys were not passed to {self.__class__.__name__}"
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

    def interaction(self, mode: bool = True) -> None:
        if mode:
            self.eval()
        else:
            self.train()
        self.interact = mode

    def learning(self) -> None:
        return self.interaction(False)

    def get_dist(self, tensor_dict) -> Tuple[torch.distributions.Distribution, ...]:
        # try:
        tensors = [tensor_dict.get(key) for key in self.in_keys]
        # except KeyError:
        #     for key in self.in_keys:
        #         if key not in tensordict.keys():
        #             raise KeyError(f"Key {key} is missing from tensordict with keys {list(tensordict.keys())} "
        #                            f"in {self.__class__.__name__}. "
        #                            f"Check that argument in_keys was set properly.")
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
            # try:
            #     assert isinstance(_tensor, torch.Tensor)
            #     tensordict.set_(_out_key, _tensor)
            # except:
            tensor_dict.set(_out_key, _tensor)

    def forward(self, tensor_dict: _TensorDict) -> _TensorDict:
        tensor_dict_unsqueezed = tensor_dict
        if not len(tensor_dict.batch_size):
            tensor_dict_unsqueezed = tensor_dict.unsqueeze(0)
        dist, *tensors = self.get_dist(tensor_dict_unsqueezed)
        out_tensor = self._dist_sample(dist)
        self._write_to_tensor_dict(tensor_dict_unsqueezed, [out_tensor] + list(tensors))
        if self.return_log_prob:
            log_prob = dist.log_prob(out_tensor)
            log_prob = log_prob.unsqueeze(-1)
            tensor_dict_unsqueezed.set("_".join([self.out_keys[0], "log_prob"]), log_prob)
        return tensor_dict

    def random(self, tensor_dict: _TensorDict) -> _TensorDict:
        key0 = self.out_keys[0]
        tensor_dict.set(key0, self.spec.rand(tensor_dict.batch_size))
        return tensor_dict

    def log_prob(self, tensor_dict: _TensorDict) -> _TensorDict:
        """log p(action | *observations)"""
        dist, *_ = self.get_dist(tensor_dict)
        lp = dist.log_prob(tensor_dict.get(self.out_keys[0]))
        tensor_dict.set(self.out_keys[0] + "_log_prob", lp)
        return tensor_dict

    def _dist_sample(self, dist: d.Distribution, interaction_mode: bool = None, eps: Number = None) -> torch.Tensor:
        if interaction_mode is None:
            interaction_mode = self.default_interaction_mode

        assert isinstance(
            dist, d.Distribution
        ), f"type {type(dist)} not recognised by _dist_sample"

        if interaction_mode == "mode":
            if hasattr(dist, "mode"):
                return dist.mode
            else:
                raise NotImplementedError("method {type(dist)}.mode is not implemented")

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

    def random_sample(self, out_shape: Union[torch.Size, Iterable]) -> torch.Tensor:
        """Returns a random sample (possibly uniform or Gaussian) with similar properties as samples gathered from _get_dist.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mapping_operator={self.mapping_operator}, distribution_class={self.distribution_class})"


class ProbabilisticOperatorWrapper(nn.Module):
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
