from typing import Optional, Iterable, Union, Tuple, Type, Iterator, Callable

import torch
from torch import nn, distributions as d

from torchrl.modules.distributions import Delta, OneHotCategorical
from .common import ProbabilisticOperator, ProbabilisticOperatorWrapper
from ..models.models import DistributionalDQNnet

__all__ = [
    "Actor",
    "ActorValueOperator",
    "ValueOperator",
    "QValueActor",
    "DistributionalQValueActor",
]

from ...data import TensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from ...data.tensordict.tensordict import _TensorDict


class Actor(ProbabilisticOperator):
    """
    General class for Actors in RL.
    The Actor class comes with default values for the in_keys and out_keys arguments (["observation"] and ["action"],
    respectively).

    """

    def __init__(
        self,
        action_spec: TensorSpec,
        mapping_operator: nn.Module,
        distribution_class: Type = Delta,
        distribution_kwargs: Optional[dict] = None,
        default_interaction_mode: str = "mode",
        _n_empirical_est: int = 1000,
        safe: bool = False,
        in_keys: Optional[Iterable[str]] = None,
        out_keys: Optional[Iterable[str]] = None,
        **kwargs,
    ):
        if in_keys is None:
            in_keys = ["observation"]
        if out_keys is None:
            out_keys = ["action"]

        super().__init__(
            action_spec,
            mapping_operator=mapping_operator,
            distribution_class=distribution_class,
            distribution_kwargs=distribution_kwargs,
            default_interaction_mode=default_interaction_mode,
            _n_empirical_est=_n_empirical_est,
            safe=safe,
            in_keys=in_keys,
            out_keys=out_keys,
            **kwargs,
        )


class ValueOperator(ProbabilisticOperator):
    def __init__(
        self,
        mapping_operator: nn.Module,
        in_keys: Optional[Iterable[str]] = None,
        out_keys: Optional[Iterable[str]] = None,
    ) -> None:

        if in_keys is None:
            in_keys = ["observation"]
        if out_keys is None:
            out_keys = ["state_value"] if "action" not in in_keys else ["state_action_value"]
        value_spec = UnboundedContinuousTensorSpec()
        super().__init__(
            value_spec,
            mapping_operator=mapping_operator,
            in_keys=in_keys,
            out_keys=out_keys,
        )


class QValueHook:
    """
    Q-Value hook for Q-value policies.
    Given a the output of a mapping operator, representing the values of the different discrete actions available,
    a QValueHook will transform these values into their argmax component.
    Currently, this is returned as a one-hot encoding.

    Args:
        action_space (str): Action space. Must be one of "one-hot", "mult_one_hot" or "binary".
        var_nums (int, optional): if action_space == "mult_one_hot", this value represents the cardinality of each
            action component.

    """

    def __init__(
        self,
        action_space: str,
        var_nums: Optional[int] = None,
    ):
        self.action_space = action_space
        self.var_nums = var_nums
        self.fun_dict = {
            "one_hot": self._one_hot,
            "mult_one_hot": self._mult_one_hot,
            "binary": self._binary,
        }

    def __call__(
        self, net: nn.Module, observation: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fun_dict[self.action_space](values), values

    @staticmethod
    def _one_hot(value: torch.Tensor) -> torch.Tensor:
        out = (value == value.max(dim=-1, keepdim=True)[0]).to(torch.long)
        return out

    def _mult_one_hot(self, value: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        values = value.split(self.var_nums, dim=-1)
        return torch.cat(
            [
                QValueHook._one_hot(
                    _value,
                )
                for _value in values
            ],
            -1,
        )

    @staticmethod
    def _binary(value: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DistributionalQValueHook(QValueHook):
    """
    Distributional Q-Value hook for Q-value policies.
    Given a the output of a mapping operator, representing the values of the different discrete actions available,
    a DistributionalQValueHook will transform these values into their argmax component using the provided support.
    Currently, this is returned as a one-hot encoding.
    For more details regarding Distributional DQN, refer to "A Distributional Perspective on Reinforcement Learning",
    https://arxiv.org/pdf/1707.06887.pdf

    Args:
        action_space (str): Action space. Must be one of "one-hot", "mult_one_hot" or "binary".
        support (torch.Tensor): support of the action values.
        var_nums (int, optional): if action_space == "mult_one_hot", this value represents the cardinality of each
            action component.

    """

    def __init__(
        self,
        action_space: str,
        support: torch.Tensor,
        var_nums: Optional[int] = None,
    ):
        self.action_space = action_space
        self.support = support
        self.var_nums = var_nums
        self.fun_dict = {
            "one_hot": self._one_hot,
            "mult_one_hot": self._mult_one_hot,
            "binary": self._binary,
        }

    def __call__(
        self, net: nn.Module, observation: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fun_dict[self.action_space](values, self.support), values

    def _support_expected(
        self, log_softmax_values: torch.Tensor, support: torch.Tensor
    ) -> torch.Tensor:
        support = support.to(log_softmax_values.device)
        if log_softmax_values.shape[-2] != support.shape[-1]:
            raise RuntimeError(
                "Support length and number of atoms in mapping_operator output should match, "
                f"got self.support.shape={support.shape} and mapping_operator(...).shape={log_softmax_values.shape}"
            )
        if (log_softmax_values > 0).any():
            raise ValueError(
                f"input to QValueHook must be log-softmax values (which are expected to be non-positive numbers). "
                f"got a maximum value of {log_softmax_values.max():4.4f}"
            )
        return (log_softmax_values.exp() * support.unsqueeze(-1)).sum(-2)

    def _one_hot(self, value: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"got value of type {value.__class__.__name__}")
        if not isinstance(support, torch.Tensor):
            raise TypeError(f"got support of type {support.__class__.__name__}")
        value = self._support_expected(value, support)
        out = (value == value.max(dim=-1, keepdim=True)[0]).to(torch.long)
        return out

    def _mult_one_hot(self, value: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        values = value.split(self.var_nums, dim=-1)
        return torch.cat(
            [
                self._one_hot(_value, _support)
                for _value, _support in zip(values, support)
            ],
            -1,
        )

    @staticmethod
    def _binary(value: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class QValueActor(Actor):
    """
    DQN Actor subclass.
    This class hooks the mapping_operator such that it returns a one-hot encoding of the argmax value.

    """

    def __init__(self, *args, action_space: int = "one_hot", **kwargs):
        out_keys = [
            "action",
            "action_value",
        ]
        super().__init__(*args, out_keys=out_keys, **kwargs)
        self.action_space = action_space
        self.mapping_operator.register_forward_hook(QValueHook(self.action_space))
        if not self.distribution_class is Delta:
            raise TypeError(
                f"{self.__class__.__name__} expects a distribution_class Delta, "
                f"but got {self.distribution_class.__name__} instead."
            )

class DistributionalQValueActor(QValueActor):
    """
    Distributional DQN Actor subclass.
    This class hooks the mapping_operator such that it returns a one-hot encoding of the argmax value on its support.

    """

    def __init__(
        self, *args, support: torch.Tensor, action_space: str = "one_hot", **kwargs
    ):
        out_keys = [
            "action",
            "action_value",
        ]
        super(QValueActor, self).__init__(*args, out_keys=out_keys, **kwargs)
        self.action_space = action_space

        self.register_buffer("support", support)
        self.action_space = action_space
        if not isinstance(self.mapping_operator, DistributionalDQNnet):
            self.mapping_operator = DistributionalDQNnet(self.mapping_operator)
        self.mapping_operator.register_forward_hook(
            DistributionalQValueHook(self.action_space, self.support)
        )
        if self.distribution_class is not Delta:
            raise TypeError(
                f"{self.__class__.__name__} expects a distribution_class Delta, "
                f"but got {self.distribution_class.__name__} instead."
            )


class ActorValueOperator(ProbabilisticOperator):
    """
    Actor-value operator.

    This class wraps together an actor and a value model that share a common observation embedding network:
               Obs
                v
        observation_embedding
            v    |     v
          actor  |   critic
            v    |     v
          action |   value

    To facilitate the workflow, this  class comes with a get_policy_operator() and get_value_operator() methods, which
    will both return a stand-alone ProbabilisticOperator with the dedicated functionality.

    Args:
        spec (TensorSpec): spec of the action
        in_keys (Iterable of str): keys of the input tensordict to be read by the common operator
        common_mapping_operator (Callable or nn.Module): operator reading the tensordict keys and producing a common
            embedding that is to be used by the actor and value network sub-modules.
        policy_operator (Callable or nn.Module): actor sub-module.
        value_operator (Callable or nn.Module): value network sub-module.
        policy_distribution_class (Type): distribution class for the policy.
            default: OneHotCategorical
        policy_distribution_kwargs (dict, optional): kwargs for the policy dist.
        value_distribution_class (Type): distribution class for the policy.
            default: Delta
        value_distribution_kwargs (dict, optional): kwargs for the value dist.
        policy_interaction_mode (str): interaction mode for the policy.
            default: "mode"
        value_interaction_mode (str): interaction mode for the value network.
            default: "mode"
        return_log_prob (bool): if True, the action_log_prob will be written in the tensordict.
            default is False.
    """

    # TODO: specs for action and value should be different. Use a CompositeSpec?

    def __init__(
        self,
        spec: TensorSpec,
        in_keys: Iterable[str],
        common_mapping_operator: Union[
            Callable[[torch.Tensor], torch.Tensor], nn.Module
        ],
        policy_operator: Union[Callable[[torch.Tensor], torch.Tensor], nn.Module],
        value_operator: Union[Callable[[torch.Tensor], torch.Tensor], nn.Module],
        out_keys: Optional[Iterable[str]] = None,
        policy_distribution_class: Type = OneHotCategorical,
        policy_distribution_kwargs: Optional[dict] = None,
        policy_interaction_mode: str = "mode",  # mode, random, mean,
        return_log_prob: bool = False,
        **kwargs,
    ):
        if out_keys:
            raise RuntimeError(
                "PolicyValueOperator out_keys are pre-defined and cannot be changed, "
                f"got out_keys={out_keys}"
            )
        value_out_keys = ["state_value"]
        policy_out_keys = ["action", "action_log_prob"]
        out_keys = policy_out_keys + value_out_keys
        super().__init__(
            spec=spec,
            mapping_operator=common_mapping_operator,
            in_keys=in_keys,
            out_keys=out_keys,
            **kwargs,
        )

        self.value_po = ValueOperator(
            in_keys=["hidden_obs"],
            out_keys=value_out_keys,
            mapping_operator=value_operator,
        )
        self.policy_po = Actor(
            spec,
            in_keys=["hidden_obs"],
            out_keys=policy_out_keys,
            mapping_operator=policy_operator,
            distribution_class=policy_distribution_class,
            distribution_kwargs=policy_distribution_kwargs,
            default_interaction_mode=policy_interaction_mode,
            return_log_prob=return_log_prob,
            **kwargs,
        )
        self.out_keys = out_keys

    def _get_mapping(self, tensor_dict: _TensorDict) -> _TensorDict:
        values = [tensor_dict.get(key) for key in self.in_keys]
        hidden_obs = self.mapping_operator(*values)
        tensor_dict.set("hidden_obs", hidden_obs)
        return tensor_dict

    def get_dist(self, tensor_dict: _TensorDict) -> Tuple[d.Distribution, ...]:
        self._get_mapping(tensor_dict)
        value_dist, *value_tensors = self.value_po.get_dist(tensor_dict)
        policy_dist, *action_tensors = self.policy_po.get_dist(tensor_dict)
        return (policy_dist, value_dist, *action_tensors, *value_tensors)

    def forward(self, tensor_dict: _TensorDict) -> _TensorDict:
        self._get_mapping(tensor_dict)
        self.policy_po(tensor_dict)
        self.value_po(tensor_dict)
        return tensor_dict

    def get_policy_operator(self) -> ProbabilisticOperatorWrapper:
        """

        Returns a stand-alone policy operator that maps an observation to an action.

        """
        return OperatorMaskWrapper(self, "policy_po")

    def get_value_operator(self) -> ProbabilisticOperatorWrapper:
        """

        Returns a stand-alone value network operator that maps an observation to a value estimate.

        """
        return OperatorMaskWrapper(self, "value_po")


class ActorCriticOperator(ProbabilisticOperator):
    """
    Actor-critic operator.

    This class wraps together an actor and a value model that share a common observation embedding network:
               Obs
                v
        observation_embedding
            v
          actor
            v
          action   >   critic
                         v
                       value

    To facilitate the workflow, this  class comes with a get_policy_operator() and get_value_operator() methods, which
    will both return a stand-alone ProbabilisticOperator with the dedicated functionality.

    Args:
        spec (TensorSpec): spec of the action
        in_keys (Iterable of str): keys of the input tensordict to be read by the common operator
        common_mapping_operator (Callable or nn.Module): operator reading the tensordict keys and producing a common
            embedding that is to be used by the actor and value network sub-modules.
        policy_operator (Callable or nn.Module): actor sub-module.
        value_operator (Callable or nn.Module): value network sub-module.
        policy_distribution_class (Type): distribution class for the policy.
            default: OneHotCategorical
        policy_distribution_kwargs (dict, optional): kwargs for the policy dist.
        value_distribution_class (Type): distribution class for the policy.
            default: Delta
        value_distribution_kwargs (dict, optional): kwargs for the value dist.
        policy_interaction_mode (str): interaction mode for the policy.
            default: "mode"
        value_interaction_mode (str): interaction mode for the value network.
            default: "mode"
    """

    # TODO: specs for action and value should be different. Use a CompositeSpec?

    def __init__(
        self,
        spec: TensorSpec,
        in_keys: Iterable[str],
        common_mapping_operator: Union[
            Callable[[torch.Tensor], torch.Tensor], nn.Module
        ],
        policy_operator: Union[Callable[[torch.Tensor], torch.Tensor], nn.Module],
        out_keys: Optional[Iterable[str]] = None,
        policy_distribution_class: Type = OneHotCategorical,
        policy_distribution_kwargs: Optional[dict] = None,
        policy_interaction_mode: str = "mode",  # mode, random, mean
        **kwargs,
    ):
        if out_keys:
            raise RuntimeError(
                "PolicyValueOperator out_keys are pre-defined and cannot be changed, "
                f"got out_keys={out_keys}"
            )
        critic_out_keys = ["state_action_value"]
        policy_out_keys = ["action", "action_log_prob"]
        out_keys = policy_out_keys + critic_out_keys
        super().__init__(
            spec=spec,
            mapping_operator=common_mapping_operator,
            in_keys=in_keys,
            out_keys=out_keys,
            **kwargs,
        )

        self.critic_po = ValueOperator(
            in_keys=["hidden_obs", "action"],
            out_keys=critic_out_keys,
        )
        self.policy_po = Actor(
            spec,
            in_keys=["hidden_obs"],
            out_keys=policy_out_keys,
            mapping_operator=policy_operator,
            distribution_class=policy_distribution_class,
            distribution_kwargs=policy_distribution_kwargs,
            default_interaction_mode=policy_interaction_mode,
            return_log_prob=True,
            **kwargs,
        )
        self.out_keys = out_keys

    def _get_mapping(self, tensor_dict: _TensorDict) -> _TensorDict:
        values = [tensor_dict.get(key) for key in self.in_keys]
        hidden_obs = self.mapping_operator(*values)
        tensor_dict.set("hidden_obs", hidden_obs)
        return tensor_dict

    def get_dist(self, tensor_dict: _TensorDict) -> Tuple[d.Distribution, ...]:
        raise NotImplementedError(
            "TODO: get_dist for ActorCritic should return a joint distribution over action and value."
        )

    def forward(self, tensor_dict: _TensorDict) -> _TensorDict:
        self._get_mapping(tensor_dict)
        self.policy_po(tensor_dict)
        self.critic_po(tensor_dict)
        return tensor_dict

    def get_policy_operator(self) -> ProbabilisticOperatorWrapper:
        """

        Returns a stand-alone policy operator that maps an observation to an action.

        """
        return OperatorMaskWrapper(self, "policy_po")

    def get_critic_operator(self) -> ProbabilisticOperatorWrapper:
        """

        Returns a stand-alone critic network operator that maps an observation to a critic estimate.

        """
        return OperatorMaskWrapper(self, "critic_po")


class ActorCriticWrapper(ProbabilisticOperator):
    def __init__(self, policy_po: ProbabilisticOperator, critic_po: ValueOperator):
        in_keys = policy_po.in_keys
        out_keys = policy_po.out_keys + critic_po.out_keys
        super().__init__(
            spec=CompositeSpec(
                action=policy_po.spec, state_action_value=critic_po.spec
            ),
            mapping_operator=nn.Identity(),
            in_keys=in_keys,
            out_keys=out_keys,
        )
        self.policy_po = policy_po
        self.critic_po = critic_po

    def get_dist(self, tensor_dict: _TensorDict) -> Tuple[d.Distribution, ...]:
        raise NotImplementedError(
            "TODO: get_dist for ActorCritic should return a joint distribution over action and value."
        )

    def forward(self, tensor_dict: _TensorDict) -> _TensorDict:
        self.policy_po(tensor_dict)
        self.critic_po(tensor_dict)
        return tensor_dict

    def get_policy_operator(self) -> ProbabilisticOperatorWrapper:
        """

        Returns a stand-alone policy operator that maps an observation to an action.

        """
        return self.policy_po

    def get_critic_operator(self) -> ProbabilisticOperatorWrapper:
        """

        Returns a stand-alone critic network operator that maps a state-action pair to a critic estimate.

        """
        return self.critic_po

    def get_value_operator(self) -> ProbabilisticOperatorWrapper:
        """

        Returns a stand-alone value network operator that maps a state description to a value estimate.

        """
        return self.critic_po


class OperatorMaskWrapper(ProbabilisticOperatorWrapper):
    """
    Given an actor-critic object and a target network (policy or value network), acts as a stand-alone operator with the
    dedicated functionality.

    Args:
        parent_operator (ActorValueOperator): actor-critic containing the target network
        target (str): name of the target network. By default, the policy network is named `actor_critic.policy_po` and
            the value network is named `actor_critic.value_po`.
    """

    def __init__(self, parent_operator: ActorValueOperator, target: str):
        super().__init__(getattr(parent_operator, target))
        self.target = target
        self.parent_operator = parent_operator
        self.in_keys = parent_operator.in_keys
        if not hasattr(parent_operator, target):
            raise AttributeError(
                f"{target} of OperatorMaskWrapper not found in the operator {type(parent_operator)}"
            )

    @property
    def target_operator(self) -> ProbabilisticOperator:
        return getattr(self.parent_operator, self.target)

    def get_dist(self, tensor_dict: _TensorDict) -> Tuple[d.Distribution, ...]:
        self.parent_operator._get_mapping(tensor_dict)
        dist, *tensors = self.target_operator.get_dist(tensor_dict)
        return (dist, *tensors)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, exclude_common_operator=False
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        if not recurse:
            raise ValueError(
                "OperatorMaskWrapper.named_parameters requires the recurse arg to be set to True"
            )

        for n, p in self.target_operator.named_parameters(prefix=prefix):
            yield n, p
        if not exclude_common_operator:
            for n, p in self.parent_operator.mapping_operator.named_parameters(
                prefix=prefix
            ):
                yield n, p
