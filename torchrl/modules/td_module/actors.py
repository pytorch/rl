from typing import Optional, Iterable, Union, Tuple, Type, Iterator, Callable

import torch
from torch import nn, distributions as d

from torchrl.modules.distributions import Delta, OneHotCategorical
from torchrl.modules.td_module.common import (
    ProbabilisticTDModule,
    TDModuleWrapper,
    TDModule,
    TDSequence,
)
from torchrl.modules.models.models import DistributionalDQNnet

__all__ = [
    "Actor",
    "ProbabilisticActor",
    "ActorValueOperator",
    "ValueOperator",
    "QValueActor",
    "ActorCriticOperator",
    "ActorCriticWrapper",
    "DistributionalQValueActor",
]

from torchrl.data import TensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.data.tensordict.tensordict import _TensorDict


class Actor(TDModule):
    """
    General class for deterministic actors in RL.
    The Actor class comes with default values for the in_keys and out_keys arguments (["observation"] and ["action"],
    respectively).

    Examples:
        >>> from torchrl.data import TensorDict, NdUnboundedContinuousTensorSpec
        >>> from torchrl.modules import Actor
        >>> import torch
        >>> td = TensorDict({"observation": torch.randn(3, 4)}, [3,])
        >>> action_spec = NdUnboundedContinuousTensorSpec(4)
        >>> module = torch.nn.Linear(4, 4)
        >>> td_module = Actor(
        ...    spec=action_spec,
        ...    module=module,
        ...    )
        >>> td_module(td)
        >>> print(td.get("action"))

    """

    def __init__(
        self,
        *args,
        in_keys: Optional[Iterable[str]] = None,
        out_keys: Optional[Iterable[str]] = None,
        **kwargs,
    ):
        if in_keys is None:
            in_keys = ["observation"]
        if out_keys is None:
            out_keys = ["action"]

        super().__init__(
            *args,
            in_keys=in_keys,
            out_keys=out_keys,
            **kwargs,
        )


class ProbabilisticActor(ProbabilisticTDModule):
    """
    General class for probabilistic actors in RL.
    The Actor class comes with default values for the in_keys and out_keys arguments (["observation"] and ["action"],
    respectively).

    Examples:
        >>> from torchrl.data import TensorDict, NdBoundedTensorSpec
        >>> from torchrl.modules import Actor, TanhNormal
        >>> import torch
        >>> td = TensorDict({"observation": torch.randn(3, 4)}, [3,])
        >>> action_spec = NdBoundedTensorSpec(shape=torch.Size([4]), minimum=-1, maximum=1)
        >>> module = torch.nn.Linear(4, 8)
        >>> td_module = ProbabilisticActor(
        ...    spec=action_spec,
        ...    module=module,
        ...    distribution_class=TanhNormal,
        ...    )
        >>> td_module(td)
        >>> print(td.get("action"))

    """

    def __init__(
        self,
        *args,
        in_keys: Optional[Iterable[str]] = None,
        out_keys: Optional[Iterable[str]] = None,
        **kwargs,
    ):
        if in_keys is None:
            in_keys = ["observation"]
        if out_keys is None:
            out_keys = ["action"]

        super().__init__(
            *args,
            in_keys=in_keys,
            out_keys=out_keys,
            **kwargs,
        )


class ValueOperator(TDModule):
    """
    General class for value functions in RL.
    The ValueOperator class comes with default values for the in_keys and out_keys arguments (["observation"] and [
    "state_value"] or ["state_action_value"], respectively and depending on whether the "action" key is part of the
    in_keys list).

    Examples:
        >>> from torchrl.data import TensorDict, NdUnboundedContinuousTensorSpec
        >>> from torchrl.modules import ValueOperator
        >>> import torch
        >>> from torch import nn
        >>> td = TensorDict({"observation": torch.randn(3, 4), "action": torch.randn(3, 2)}, [3,])
        >>> class CustomModule(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = torch.nn.Linear(6, 1)
        ...     def forward(self, obs, action):
        ...         return self.linear(torch.cat([obs, action], -1))
        >>> module = CustomModule()
        >>> td_module = ValueOperator(
        ...    in_keys=["observation", "action"],
        ...    module=module,
        ...    )
        >>> td_module(td)
        >>> print(td)
        TensorDict(
            fields={observation: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                action: Tensor(torch.Size([3, 2]), dtype=torch.float32),
                state_action_value: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([3]),
            device=cpu)


    """

    def __init__(
        self,
        module: nn.Module,
        in_keys: Optional[Iterable[str]] = None,
        out_keys: Optional[Iterable[str]] = None,
    ) -> None:

        if in_keys is None:
            in_keys = ["observation"]
        if out_keys is None:
            out_keys = (
                ["state_value"] if "action" not in in_keys else ["state_action_value"]
            )
        value_spec = UnboundedContinuousTensorSpec()
        super().__init__(
            value_spec,
            module=module,
            in_keys=in_keys,
            out_keys=out_keys,
        )


class QValueHook:
    """
    Q-Value hook for Q-value policies.
    Given a the output of a regular nn.Module, representing the values of the different discrete actions available,
    a QValueHook will transform these values into their argmax component (i.e. the resulting greedy action).
    Currently, this is returned as a one-hot encoding.

    Args:
        action_space (str): Action space. Must be one of "one-hot", "mult_one_hot" or "binary".
        var_nums (int, optional): if action_space == "mult_one_hot", this value represents the cardinality of each
            action component.

    Examples:
        >>> from torchrl.data import TensorDict, OneHotDiscreteTensorSpec
        >>> from torchrl.modules.td_module.actors import QValueHook, Actor
        >>> from torch import nn
        >>> import torch
        >>> td = TensorDict({'observation': torch.randn(5, 4)}, [5])
        >>> module = nn.Linear(4, 4)
        >>> action_spec = OneHotDiscreteTensorSpec(4)
        >>> hook = QValueHook("one_hot")
        >>> _ = module.register_forward_hook(hook)
        >>> qvalue_actor = Actor(spec=action_spec, module=module, out_keys=["action", "action_value"])
        >>> _ = qvalue_actor(td)
        >>> print(td)
        TensorDict(
            fields={observation: Tensor(torch.Size([5, 4]), dtype=torch.float32),
                action: Tensor(torch.Size([5, 4]), dtype=torch.int64),
                action_value: Tensor(torch.Size([5, 4]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([5]),
            device=cpu)

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
        if action_space not in self.fun_dict:
            raise ValueError(
                f"action_space must be one of {list(self.fun_dict.keys())}"
            )

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
        action_space (str): Action space. Must be one of "one_hot", "mult_one_hot" or "binary".
        support (torch.Tensor): support of the action values.
        var_nums (int, optional): if action_space == "mult_one_hot", this value represents the cardinality of each
            action component.

    Examples:
        >>> from torchrl.data import TensorDict, OneHotDiscreteTensorSpec
        >>> from torchrl.modules.td_module.actors import DistributionalQValueHook, Actor
        >>> from torch import nn
        >>> import torch
        >>> td = TensorDict({'observation': torch.randn(5, 4)}, [5])
        >>> nbins = 3
        >>> class CustomDistributionalQval(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = nn.Linear(4, nbins*4)
        ...
        ...     def forward(self, x):
        ...         return self.linear(x).view(-1, nbins, 4).log_softmax(-2)
        ...
        >>> module = CustomDistributionalQval()
        >>> action_spec = OneHotDiscreteTensorSpec(4)
        >>> hook = DistributionalQValueHook("one_hot", support = torch.arange(nbins))
        >>> _ = module.register_forward_hook(hook)
        >>> qvalue_actor = Actor(spec=action_spec, module=module, out_keys=["action", "action_value"])
        >>> _ = qvalue_actor(td)
        >>> print(td)
        TensorDict(
            fields={observation: Tensor(torch.Size([5, 4]), dtype=torch.float32),
                action: Tensor(torch.Size([5, 4]), dtype=torch.int64),
                action_value: Tensor(torch.Size([5, 3, 4]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([5]),
            device=cpu)

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
                "Support length and number of atoms in module output should match, "
                f"got self.support.shape={support.shape} and module(...).shape={log_softmax_values.shape}"
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
    This class hooks the module such that it returns a one-hot encoding of the argmax value.

    Examples:
        >>> from torchrl.data import TensorDict, OneHotDiscreteTensorSpec
        >>> from torchrl.modules.td_module.actors import QValueActor
        >>> from torch import nn
        >>> import torch
        >>> td = TensorDict({'observation': torch.randn(5, 4)}, [5])
        >>> module = nn.Linear(4, 4)
        >>> action_spec = OneHotDiscreteTensorSpec(4)
        >>> qvalue_actor = QValueActor(spec=action_spec, module=module)
        >>> _ = qvalue_actor(td)
        >>> print(td)
        TensorDict(
            fields={observation: Tensor(torch.Size([5, 4]), dtype=torch.float32),
                action: Tensor(torch.Size([5, 4]), dtype=torch.int64),
                action_value: Tensor(torch.Size([5, 4]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([5]),
            device=cpu)

    """

    def __init__(self, *args, action_space: int = "one_hot", **kwargs):
        out_keys = [
            "action",
            "action_value",
        ]
        super().__init__(*args, out_keys=out_keys, **kwargs)
        self.action_space = action_space
        self.module.register_forward_hook(QValueHook(self.action_space))


class DistributionalQValueActor(QValueActor):
    """
    Distributional DQN Actor subclass.
    This class hooks the module such that it returns a one-hot encoding of the argmax value on its support.

    Examples:
        >>> from torchrl.data import TensorDict, OneHotDiscreteTensorSpec
        >>> from torchrl.modules import DistributionalQValueActor, MLP
        >>> from torch import nn
        >>> import torch
        >>> td = TensorDict({'observation': torch.randn(5, 4)}, [5])
        >>> nbins = 3
        >>> module = MLP(out_features=(nbins, 4), depth=2)
        >>> action_spec = OneHotDiscreteTensorSpec(4)
        >>> qvalue_actor = DistributionalQValueActor(spec=action_spec, module=module, support=torch.arange(nbins))
        >>> _ = qvalue_actor(td)
        >>> print(td)
        TensorDict(
            fields={observation: Tensor(torch.Size([5, 4]), dtype=torch.float32),
                action: Tensor(torch.Size([5, 4]), dtype=torch.int64),
                action_value: Tensor(torch.Size([5, 3, 4]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([5]),
            device=cpu)

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
        if not isinstance(self.module, DistributionalDQNnet):
            self.module = DistributionalDQNnet(self.module)
        self.module.register_forward_hook(
            DistributionalQValueHook(self.action_space, self.support)
        )


class ActorValueOperator(TDSequence):
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
    will both return a stand-alone TDModule with the dedicated functionality.

    Args:
        common_operator (TDModule): a common operator that reads observations and produces a hidden variable
        policy_operator (TDModule): a policy operator that reads the hidden variable and returns an action
        value_operator (TDModule): a value operator, that reads the hidden variable and returns a value

    Examples:
        >>> from torchrl.data import TensorDict, NdUnboundedContinuousTensorSpec, NdBoundedTensorSpec
        >>> from torchrl.modules import ProbabilisticActor, ValueOperator, TanhNormal, ActorValueOperator
        >>> import torch
        >>> spec_hidden = NdUnboundedContinuousTensorSpec(4)
        >>> module_hidden = torch.nn.Linear(4, 4)
        >>> td_module_hidden = TDModule(
        ...    spec=spec_hidden,
        ...    module=module_hidden,
        ...    in_keys=["observation"],
        ...    out_keys=["hidden"],
        ...    )
        >>> spec_action = NdBoundedTensorSpec(-1, 1, torch.Size([8]))
        >>> module_action = torch.nn.Linear(4, 8)
        >>> td_module_action = ProbabilisticActor(
        ...    spec=spec_action,
        ...    module=module_action,
        ...    in_keys=["hidden"],
        ...    distribution_class=TanhNormal,
        ...    return_log_prob=True,
        ...    )
        >>> module_value = torch.nn.Linear(4, 1)
        >>> td_module_value = ValueOperator(
        ...    module=module_value,
        ...    in_keys=["hidden"],
        ...    )
        >>> td_module = ActorValueOperator(td_module_hidden, td_module_action, td_module_value)
        >>> td = TensorDict({"observation": torch.randn(3, 4)}, [3,])
        >>> td_clone = td_module(td.clone())
        >>> print(td_clone)
        TensorDict(
            fields={observation: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                action: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                action_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32),
                state_value: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([3]),
            device=cpu)
        >>> td_clone = td_module.get_policy_operator()(td.clone())
        >>> print(td_clone)  # no value
        TensorDict(
            fields={observation: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                action: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                action_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([3]),
            device=cpu)

        >>> td_clone = td_module.get_value_operator()(td.clone())
        >>> print(td_clone)  # no action
        TensorDict(
            fields={observation: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                state_value: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([3]),
            device=cpu)

    """

    def __init__(
        self,
        common_operator: TDModule,
        policy_operator: TDModule,
        value_operator: TDModule,
    ):
        super().__init__(
            common_operator,
            policy_operator,
            value_operator,
        )

    def get_policy_operator(self) -> TDSequence:
        """

        Returns a stand-alone policy operator that maps an observation to an action.

        """
        return TDSequence(self.module[0], self.module[1])

    def get_value_operator(self) -> TDSequence:
        """

        Returns a stand-alone value network operator that maps an observation to a value estimate.

        """
        return TDSequence(self.module[0], self.module[2])


class ActorCriticOperator(ActorValueOperator):
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

    To facilitate the workflow, this  class comes with a get_policy_operator() method, which
    will both return a stand-alone TDModule with the dedicated functionality. The get_critic_operator will return the
    parent object, as the value is computed based on the policy output.

    Args:
        common_operator (TDModule): a common operator that reads observations and produces a hidden variable
        policy_operator (TDModule): a policy operator that reads the hidden variable and returns an action
        value_operator (TDModule): a value operator, that reads the hidden variable and returns a value

    Examples:
        >>> from torchrl.data import TensorDict, NdUnboundedContinuousTensorSpec, NdBoundedTensorSpec
        >>> from torchrl.modules import ProbabilisticActor, ValueOperator, TanhNormal, ActorCriticOperator
        >>> import torch
        >>> spec_hidden = NdUnboundedContinuousTensorSpec(4)
        >>> module_hidden = torch.nn.Linear(4, 4)
        >>> td_module_hidden = TDModule(
        ...    spec=spec_hidden,
        ...    module=module_hidden,
        ...    in_keys=["observation"],
        ...    out_keys=["hidden"],
        ...    )
        >>> spec_action = NdBoundedTensorSpec(-1, 1, torch.Size([8]))
        >>> module_action = torch.nn.Linear(4, 8)
        >>> td_module_action = ProbabilisticActor(
        ...    spec=spec_action,
        ...    module=module_action,
        ...    in_keys=["hidden"],
        ...    distribution_class=TanhNormal,
        ...    return_log_prob=True,
        ...    )
        >>> module_value = torch.nn.Linear(4, 1)
        >>> td_module_value = ValueOperator(
        ...    module=module_value,
        ...    in_keys=["hidden"],
        ...    )
        >>> td_module = ActorCriticOperator(td_module_hidden, td_module_action, td_module_value)
        >>> td = TensorDict({"observation": torch.randn(3, 4)}, [3,])
        >>> td_clone = td_module(td.clone())
        >>> print(td_clone)
        TensorDict(
            fields={observation: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                action: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                action_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32),
                state_value: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([3]),
            device=cpu)
        >>> td_clone = td_module.get_policy_operator()(td.clone())
        >>> print(td_clone)  # no value
        TensorDict(
            fields={observation: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                action: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                action_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([3]),
            device=cpu)

        >>> td_clone = td_module.get_critic_operator()(td.clone())
        >>> print(td_clone)  # no action
        TensorDict(
            fields={observation: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                action: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                action_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32),
                state_value: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([3]),
            device=cpu)
    """

    def get_critic_operator(self) -> TDModuleWrapper:
        """

        Returns a stand-alone critic network operator that maps a state-action pair to a critic estimate.

        """
        return self

    def get_value_operator(self) -> TDModuleWrapper:
        raise RuntimeError(
            "value_operator is the term used for operators that associate a value with a "
            "state/observation. This class computes the value of a state-action pair: to get the "
            "network computing this value, please call td_sequence.get_critic_operator()"
        )


class ActorCriticWrapper(TDSequence):
    """
    Actor-value operator without common module.

    This class wraps together an actor and a value model that do not share a common observation embedding network:
                Obs
            v    |     v
          actor  |   critic
            v    |     v
          action |   value

    To facilitate the workflow, this  class comes with a get_policy_operator() and get_value_operator() methods, which
    will both return a stand-alone TDModule with the dedicated functionality.

    Args:
        policy_operator (TDModule): a policy operator that reads the hidden variable and returns an action
        value_operator (TDModule): a value operator, that reads the hidden variable and returns a value

    Examples:
        >>> from torchrl.data import TensorDict, NdUnboundedContinuousTensorSpec, NdBoundedTensorSpec
        >>> from torchrl.modules import ProbabilisticActor, ValueOperator, TanhNormal, ActorCriticWrapper
        >>> import torch
        >>> spec_action = NdBoundedTensorSpec(-1, 1, torch.Size([8]))
        >>> module_action = torch.nn.Linear(4, 8)
        >>> td_module_action = ProbabilisticActor(
        ...    spec=spec_action,
        ...    module=module_action,
        ...    distribution_class=TanhNormal,
        ...    return_log_prob=True,
        ...    )
        >>> module_value = torch.nn.Linear(4, 1)
        >>> td_module_value = ValueOperator(
        ...    module=module_value,
        ...    in_keys=["observation"],
        ...    )
        >>> td_module = ActorCriticWrapper(td_module_action, td_module_value)
        >>> td = TensorDict({"observation": torch.randn(3, 4)}, [3,])
        >>> td_clone = td_module(td.clone())
        >>> print(td_clone)
        TensorDict(
            fields={observation: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                action: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                action_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32),
                state_value: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([3]),
            device=cpu)
        >>> td_clone = td_module.get_policy_operator()(td.clone())
        >>> print(td_clone)  # no value
        TensorDict(
            fields={observation: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                action: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                action_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([3]),
            device=cpu)

        >>> td_clone = td_module.get_value_operator()(td.clone())
        >>> print(td_clone)  # no action
        TensorDict(
            fields={observation: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                state_value: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([3]),
            device=cpu)

    """

    def __init__(
        self,
        policy_operator: TDModule,
        value_operator: TDModule,
    ):
        super().__init__(
            policy_operator,
            value_operator,
        )

    def get_policy_operator(self) -> TDSequence:
        """

        Returns a stand-alone policy operator that maps an observation to an action.

        """
        return self.module[0]

    def get_value_operator(self) -> TDSequence:
        """

        Returns a stand-alone value network operator that maps an observation to a value estimate.

        """
        return self.module[1]
