# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence, Tuple, Union

import torch
from tensordict.nn import TensorDictModuleWrapper
from torch import nn

from torchrl.data import CompositeSpec, TensorSpec, UnboundedContinuousTensorSpec
from torchrl.modules.models.models import DistributionalDQNnet
from torchrl.modules.tensordict_module.common import SafeModule
from torchrl.modules.tensordict_module.probabilistic import SafeProbabilisticModule
from torchrl.modules.tensordict_module.sequence import SafeSequential


class Actor(SafeModule):
    """General class for deterministic actors in RL.

    The Actor class comes with default values for the out_keys (["action"])
    and if the spec is provided but not as a CompositeSpec object, it will be
    automatically translated into :obj:`spec = CompositeSpec(action=spec)`

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import NdUnboundedContinuousTensorSpec
        >>> from torchrl.modules import Actor
        >>> td = TensorDict({"observation": torch.randn(3, 4)}, [3,])
        >>> action_spec = NdUnboundedContinuousTensorSpec(4)
        >>> module = torch.nn.Linear(4, 4)
        >>> td_module = Actor(
        ...    module=module,
        ...    spec=action_spec,
        ...    )
        >>> td_module(td)
        >>> print(td.get("action"))

    """

    def __init__(
        self,
        *args,
        in_keys: Optional[Sequence[str]] = None,
        out_keys: Optional[Sequence[str]] = None,
        spec: Optional[TensorSpec] = None,
        **kwargs,
    ):
        if in_keys is None:
            in_keys = ["observation"]
        if out_keys is None:
            out_keys = ["action"]
        if (
            "action" in out_keys
            and spec is not None
            and not isinstance(spec, CompositeSpec)
        ):
            spec = CompositeSpec(action=spec)

        super().__init__(
            *args,
            in_keys=in_keys,
            out_keys=out_keys,
            spec=spec,
            **kwargs,
        )


class ProbabilisticActor(SafeProbabilisticModule):
    """General class for probabilistic actors in RL.

    The Actor class comes with default values for the out_keys (["action"])
    and if the spec is provided but not as a CompositeSpec object, it will be
    automatically translated into :obj:`spec = CompositeSpec(action=spec)`

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn.functional_modules import make_functional
        >>> from torchrl.data import NdBoundedTensorSpec
        >>> from torchrl.modules import ProbabilisticActor, NormalParamWrapper, SafeModule, TanhNormal
        >>> td = TensorDict({"observation": torch.randn(3, 4)}, [3,])
        >>> action_spec = NdBoundedTensorSpec(shape=torch.Size([4]),
        ...    minimum=-1, maximum=1)
        >>> module = NormalParamWrapper(torch.nn.Linear(4, 8))
        >>> params = make_functional(module)
        >>> tensordict_module = SafeModule(module, in_keys=["observation"], out_keys=["loc", "scale"])
        >>> td_module = ProbabilisticActor(
        ...    module=tensordict_module,
        ...    spec=action_spec,
        ...    dist_in_keys=["loc", "scale"],
        ...    distribution_class=TanhNormal,
        ...    )
        >>> td = td_module(td, params=params)
        >>> td
        TensorDict(
            fields={
                observation: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                loc: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                scale: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                action: Tensor(torch.Size([3, 4]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=cpu,
            is_shared=False)

    """

    def __init__(
        self,
        module: SafeModule,
        dist_in_keys: Union[str, Sequence[str]],
        sample_out_key: Optional[Sequence[str]] = None,
        spec: Optional[TensorSpec] = None,
        **kwargs,
    ):
        if sample_out_key is None:
            sample_out_key = ["action"]
        if (
            "action" in sample_out_key
            and spec is not None
            and not isinstance(spec, CompositeSpec)
        ):
            spec = CompositeSpec(action=spec)

        super().__init__(
            module=module,
            dist_in_keys=dist_in_keys,
            sample_out_key=sample_out_key,
            spec=spec,
            **kwargs,
        )


class ValueOperator(SafeModule):
    """General class for value functions in RL.

    The ValueOperator class comes with default values for the in_keys and
    out_keys arguments (["observation"] and ["state_value"] or
    ["state_action_value"], respectively and depending on whether the "action"
    key is part of the in_keys list).

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn.functional_modules import make_functional
        >>> from torch import nn
        >>> from torchrl.data import NdUnboundedContinuousTensorSpec
        >>> from torchrl.modules import ValueOperator
        >>> td = TensorDict({"observation": torch.randn(3, 4), "action": torch.randn(3, 2)}, [3,])
        >>> class CustomModule(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = torch.nn.Linear(6, 1)
        ...     def forward(self, obs, action):
        ...         return self.linear(torch.cat([obs, action], -1))
        >>> module = CustomModule()
        >>> td_module = ValueOperator(
        ...    in_keys=["observation", "action"], module=module
        ... )
        >>> params = make_functional(td_module)
        >>> td_module(td, params=params)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(torch.Size([3, 2]), dtype=torch.float32),
                observation: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                state_action_value: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)


    """

    def __init__(
        self,
        module: nn.Module,
        in_keys: Optional[Sequence[str]] = None,
        out_keys: Optional[Sequence[str]] = None,
    ) -> None:

        if in_keys is None:
            in_keys = ["observation"]
        if out_keys is None:
            out_keys = (
                ["state_value"] if "action" not in in_keys else ["state_action_value"]
            )
        value_spec = UnboundedContinuousTensorSpec()
        super().__init__(
            spec=value_spec,
            module=module,
            in_keys=in_keys,
            out_keys=out_keys,
        )


class QValueHook:
    """Q-Value hook for Q-value policies.

    Given a the output of a regular nn.Module, representing the values of the different discrete actions available,
    a QValueHook will transform these values into their argmax component (i.e. the resulting greedy action).
    Currently, this is returned as a one-hot encoding.

    Args:
        action_space (str): Action space. Must be one of "one-hot", "mult_one_hot", "binary" or "categorical".
        var_nums (int, optional): if action_space == "mult_one_hot", this value represents the cardinality of each
            action component.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn.functional_modules import make_functional
        >>> from torch import nn
        >>> from torchrl.data import OneHotDiscreteTensorSpec
        >>> from torchrl.modules.tensordict_module.actors import QValueHook, Actor
        >>> td = TensorDict({'observation': torch.randn(5, 4)}, [5])
        >>> module = nn.Linear(4, 4)
        >>> params = make_functional(module)
        >>> hook = QValueHook("one_hot")
        >>> module.register_forward_hook(hook)
        >>> action_spec = OneHotDiscreteTensorSpec(4)
        >>> qvalue_actor = Actor(module=module, spec=action_spec, out_keys=["action", "action_value"])
        >>> qvalue_actor(td, params=params)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(torch.Size([5, 4]), dtype=torch.int64),
                action_value: Tensor(torch.Size([5, 4]), dtype=torch.float32),
                observation: Tensor(torch.Size([5, 4]), dtype=torch.float32)},
            batch_size=torch.Size([5]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        action_space: str,
        var_nums: Optional[int] = None,
    ):
        self.action_space = action_space
        self.var_nums = var_nums
        self.action_func_mapping = {
            "one_hot": self._one_hot,
            "mult_one_hot": self._mult_one_hot,
            "binary": self._binary,
            "categorical": self._categorical,
        }
        self.action_value_func_mapping = {
            "categorical": self._categorical_action_value,
        }
        if action_space not in self.action_func_mapping:
            raise ValueError(
                f"action_space must be one of {list(self.action_func_mapping.keys())}"
            )

    def __call__(
        self, net: nn.Module, observation: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(values, tuple):
            values = values[0]
        action = self.action_func_mapping[self.action_space](values)

        action_value_func = self.action_value_func_mapping.get(
            self.action_space, self._default_action_value
        )
        chosen_action_value = action_value_func(values, action)
        return action, values, chosen_action_value

    @staticmethod
    def _one_hot(value: torch.Tensor) -> torch.Tensor:
        out = (value == value.max(dim=-1, keepdim=True)[0]).to(torch.long)
        return out

    @staticmethod
    def _categorical(value: torch.Tensor) -> torch.Tensor:
        return torch.argmax(value, dim=-1).to(torch.long)

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

    @staticmethod
    def _default_action_value(
        values: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return (action * values).sum(-1, True)

    @staticmethod
    def _categorical_action_value(
        values: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        if len(values.shape) == 1:
            return values[action].unsqueeze(-1)
        batch_size = values.size(0)
        return values[range(batch_size), action].unsqueeze(-1)


class DistributionalQValueHook(QValueHook):
    """Distributional Q-Value hook for Q-value policies.

    Given a the output of a mapping operator, representing the values of the different discrete actions available,
    a DistributionalQValueHook will transform these values into their argmax component using the provided support.
    Currently, this is returned as a one-hot encoding.
    For more details regarding Distributional DQN, refer to "A Distributional Perspective on Reinforcement Learning",
    https://arxiv.org/pdf/1707.06887.pdf

    Args:
        action_space (str): Action space. Must be one of "one_hot", "mult_one_hot", "binary" or "categorical".
        support (torch.Tensor): support of the action values.
        var_nums (int, optional): if action_space == "mult_one_hot", this value represents the cardinality of each
            action component.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn.functional_modules import make_functional
        >>> from torch import nn
        >>> from torchrl.data import OneHotDiscreteTensorSpec
        >>> from torchrl.modules.tensordict_module.actors import DistributionalQValueHook, Actor
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
        >>> params = make_functional(module)
        >>> action_spec = OneHotDiscreteTensorSpec(4)
        >>> hook = DistributionalQValueHook("one_hot", support = torch.arange(nbins))
        >>> module.register_forward_hook(hook)
        >>> qvalue_actor = Actor(module=module, spec=action_spec, out_keys=["action", "action_value"])
        >>> qvalue_actor(td, params=params)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(torch.Size([5, 4]), dtype=torch.int64),
                action_value: Tensor(torch.Size([5, 3, 4]), dtype=torch.float32),
                observation: Tensor(torch.Size([5, 4]), dtype=torch.float32)},
            batch_size=torch.Size([5]),
            device=None,
            is_shared=False)

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
        self.action_func_mapping = {
            "one_hot": self._one_hot,
            "mult_one_hot": self._mult_one_hot,
            "binary": self._binary,
            "categorical": self._categorical,
        }
        if action_space not in self.action_func_mapping:
            raise ValueError(
                f"action_space must be one of {list(self.action_func_mapping.keys())}"
            )

    def __call__(
        self, net: nn.Module, observation: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(values, tuple):
            values = values[0]
        action = self.action_func_mapping[self.action_space](values, self.support)
        return action, values

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

    def _categorical(self, value: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        value = value = self._support_expected(value, support)
        return torch.argmax(value, dim=-1).to(torch.long)

    @staticmethod
    def _binary(value: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class QValueActor(Actor):
    """DQN Actor subclass.

    This class hooks the module such that it returns a one-hot encoding of the argmax value.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn.functional_modules import make_functional
        >>> from torch import nn
        >>> from torchrl.data import OneHotDiscreteTensorSpec
        >>> from torchrl.modules.tensordict_module.actors import QValueActor
        >>> td = TensorDict({'observation': torch.randn(5, 4)}, [5])
        >>> module = nn.Linear(4, 4)
        >>> params= make_functional(module)
        >>> action_spec = OneHotDiscreteTensorSpec(4)
        >>> qvalue_actor = QValueActor(module=module, spec=action_spec)
        >>> qvalue_actor(td, params=params)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(torch.Size([5, 4]), dtype=torch.int64),
                action_value: Tensor(torch.Size([5, 4]), dtype=torch.float32),
                chosen_action_value: Tensor(torch.Size([5, 1]), dtype=torch.float32),
                observation: Tensor(torch.Size([5, 4]), dtype=torch.float32)},
            batch_size=torch.Size([5]),
            device=None,
            is_shared=False)

    """

    def __init__(self, *args, action_space: int = "one_hot", **kwargs):
        out_keys = [
            "action",
            "action_value",
            "chosen_action_value",
        ]
        super().__init__(*args, out_keys=out_keys, **kwargs)
        self.action_space = action_space
        self.module.register_forward_hook(QValueHook(self.action_space))


class DistributionalQValueActor(QValueActor):
    """Distributional DQN Actor subclass.

    This class hooks the module such that it returns a one-hot encoding of the argmax value on its support.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torch import nn
        >>> from torchrl.data import OneHotDiscreteTensorSpec
        >>> from torchrl.modules import DistributionalQValueActor, MLP
        >>> td = TensorDict({'observation': torch.randn(5, 4)}, [5])
        >>> nbins = 3
        >>> module = MLP(out_features=(nbins, 4), depth=2)
        >>> action_spec = OneHotDiscreteTensorSpec(4)
        >>> qvalue_actor = DistributionalQValueActor(module=module, spec=action_spec, support=torch.arange(nbins))
        >>> qvalue_actor(td)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(torch.Size([5, 4]), dtype=torch.int64),
                action_value: Tensor(torch.Size([5, 3, 4]), dtype=torch.float32),
                observation: Tensor(torch.Size([5, 4]), dtype=torch.float32)},
            batch_size=torch.Size([5]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        *args,
        support: torch.Tensor,
        action_space: str = "one_hot",
        **kwargs,
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


class ActorValueOperator(SafeSequential):
    """Actor-value operator.

    This class wraps together an actor and a value model that share a common observation embedding network:

    .. aafig::
        :aspect: 60
        :scale: 120
        :proportional:
        :textual:

            +-------------+
            |"Observation"|
            +-------------+
                   |
                   v
            +--------------+
            |"hidden state"|
            +--------------+
            |      |       |
            v      |       v
            actor  |       critic
            |      |       |
            v      |       v
         +--------+|+-------+
         |"action"|||"value"|
         +--------+|+-------+

    To facilitate the workflow, this  class comes with a get_policy_operator() and get_value_operator() methods, which
    will both return a stand-alone TDModule with the dedicated functionality.

    Args:
        common_operator (SafeModule): a common operator that reads observations and produces a hidden variable
        policy_operator (SafeModule): a policy operator that reads the hidden variable and returns an action
        value_operator (SafeModule): a value operator, that reads the hidden variable and returns a value

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.modules import ProbabilisticActor, SafeModule
        >>> from torchrl.data import NdUnboundedContinuousTensorSpec, NdBoundedTensorSpec
        >>> from torchrl.modules import ValueOperator, TanhNormal, ActorValueOperator, NormalParamWrapper
        >>> spec_hidden = NdUnboundedContinuousTensorSpec(4)
        >>> module_hidden = torch.nn.Linear(4, 4)
        >>> td_module_hidden = SafeModule(
        ...    module=module_hidden,
        ...    spec=spec_hidden,
        ...    in_keys=["observation"],
        ...    out_keys=["hidden"],
        ...    )
        >>> spec_action = NdBoundedTensorSpec(-1, 1, torch.Size([8]))
        >>> module_action = SafeModule(
        ...     NormalParamWrapper(torch.nn.Linear(4, 8)),
        ...     in_keys=["hidden"],
        ...     out_keys=["loc", "scale"],
        ...     )
        >>> td_module_action = ProbabilisticActor(
        ...    module=module_action,
        ...    spec=spec_action,
        ...    dist_in_keys=["loc", "scale"],
        ...    sample_out_key=["action"],
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
            fields={
                action: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                loc: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                observation: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32),
                scale: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                state_value: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> td_clone = td_module.get_policy_operator()(td.clone())
        >>> print(td_clone)  # no value
        TensorDict(
            fields={
                action: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                loc: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                observation: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32),
                scale: Tensor(torch.Size([3, 4]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> td_clone = td_module.get_value_operator()(td.clone())
        >>> print(td_clone)  # no action
        TensorDict(
            fields={
                hidden: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                observation: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                state_value: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        common_operator: SafeModule,
        policy_operator: SafeModule,
        value_operator: SafeModule,
    ):
        super().__init__(
            common_operator,
            policy_operator,
            value_operator,
        )

    def get_policy_operator(self) -> SafeSequential:
        """Returns a stand-alone policy operator that maps an observation to an action."""
        return SafeSequential(self.module[0], self.module[1])

    def get_value_operator(self) -> SafeSequential:
        """Returns a stand-alone value network operator that maps an observation to a value estimate."""
        return SafeSequential(self.module[0], self.module[2])


class ActorCriticOperator(ActorValueOperator):
    """Actor-critic operator.

    This class wraps together an actor and a value model that share a common observation embedding network:

    .. aafig::
        :aspect: 60
        :scale: 120
        :proportional:
        :textual:

          +-----------+
          |Observation|
          +-----------+
            |
            v
            actor
            |
            v
        +------+
        |action| --> critic
        +------+      |
                      v
                   +-----+
                   |value|
                   +-----+

    To facilitate the workflow, this  class comes with a get_policy_operator() method, which
    will both return a stand-alone TDModule with the dedicated functionality. The get_critic_operator will return the
    parent object, as the value is computed based on the policy output.

    Args:
        common_operator (SafeModule): a common operator that reads observations and produces a hidden variable
        policy_operator (SafeModule): a policy operator that reads the hidden variable and returns an action
        value_operator (SafeModule): a value operator, that reads the hidden variable and returns a value

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.modules import ProbabilisticActor, SafeModule
        >>> from torchrl.data import NdUnboundedContinuousTensorSpec, NdBoundedTensorSpec
        >>> from torchrl.modules import  ValueOperator, TanhNormal, ActorCriticOperator, NormalParamWrapper, MLP
        >>> spec_hidden = NdUnboundedContinuousTensorSpec(4)
        >>> module_hidden = torch.nn.Linear(4, 4)
        >>> td_module_hidden = SafeModule(
        ...    module=module_hidden,
        ...    spec=spec_hidden,
        ...    in_keys=["observation"],
        ...    out_keys=["hidden"],
        ...    )
        >>> spec_action = NdBoundedTensorSpec(-1, 1, torch.Size([8]))
        >>> module_action = NormalParamWrapper(torch.nn.Linear(4, 8))
        >>> module_action = SafeModule(module_action, in_keys=["hidden"], out_keys=["loc", "scale"])
        >>> td_module_action = ProbabilisticActor(
        ...    module=module_action,
        ...    spec=spec_action,
        ...    dist_in_keys=["loc", "scale"],
        ...    sample_out_key=["action"],
        ...    distribution_class=TanhNormal,
        ...    return_log_prob=True,
        ...    )
        >>> module_value = MLP(in_features=8, out_features=1, num_cells=[])
        >>> td_module_value = ValueOperator(
        ...    module=module_value,
        ...    in_keys=["hidden", "action"],
        ...    out_keys=["state_action_value"],
        ...    )
        >>> td_module = ActorCriticOperator(td_module_hidden, td_module_action, td_module_value)
        >>> td = TensorDict({"observation": torch.randn(3, 4)}, [3,])
        >>> td_clone = td_module(td.clone())
        >>> print(td_clone)
        TensorDict(
            fields={
                action: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                loc: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                observation: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32),
                scale: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                state_action_value: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> td_clone = td_module.get_policy_operator()(td.clone())
        >>> print(td_clone)  # no value
        TensorDict(
            fields={
                action: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                loc: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                observation: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32),
                scale: Tensor(torch.Size([3, 4]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> td_clone = td_module.get_critic_operator()(td.clone())
        >>> print(td_clone)  # no action
        TensorDict(
            fields={
                action: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                loc: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                observation: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32),
                scale: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                state_action_value: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self[2].out_keys[0] == "state_value":
            raise RuntimeError(
                "Value out_key is state_value, which may lead to errors in downstream usages"
                "of that module. Consider setting `'state_action_value'` instead."
                "Make also sure that `'action'` is amongst the input keys of the value network."
                "If you are confident that action should not be used to compute the value, please"
                "user `ActorValueOperator` instead."
            )

    def get_critic_operator(self) -> TensorDictModuleWrapper:
        """Returns a stand-alone critic network operator that maps a state-action pair to a critic estimate."""
        return self

    def get_value_operator(self) -> TensorDictModuleWrapper:
        raise RuntimeError(
            "value_operator is the term used for operators that associate a value with a "
            "state/observation. This class computes the value of a state-action pair: to get the "
            "network computing this value, please call td_sequence.get_critic_operator()"
        )


class ActorCriticWrapper(SafeSequential):
    """Actor-value operator without common module.

    This class wraps together an actor and a value model that do not share a common observation embedding network:

    .. aafig::
        :aspect: 60
        :scale: 120
        :proportional:
        :textual:

          +-----------+
          |Observation|
          +-----------+
          |     |   |
          v     |   v
          actor |   critic
          |     |   |
          v     |   v
        +------+|+-------+
        |action||| value |
        +------+|+-------+

    To facilitate the workflow, this  class comes with a get_policy_operator() and get_value_operator() methods, which
    will both return a stand-alone TDModule with the dedicated functionality.

    Args:
        policy_operator (SafeModule): a policy operator that reads the hidden variable and returns an action
        value_operator (SafeModule): a value operator, that reads the hidden variable and returns a value

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import NdUnboundedContinuousTensorSpec, NdBoundedTensorSpec
        >>> from torchrl.modules import (
                ActorCriticWrapper,
                ProbabilisticActor,
                NormalParamWrapper,
                SafeModule,
                TanhNormal,
                ValueOperator,
            )
        >>> action_spec = NdBoundedTensorSpec(-1, 1, torch.Size([8]))
        >>> action_module = SafeModule(
                NormalParamWrapper(torch.nn.Linear(4, 8)),
                in_keys=["observation"],
                out_keys=["loc", "scale"],
            )
        >>> td_module_action = ProbabilisticActor(
        ...    module=action_module,
        ...    spec=action_spec,
        ...    dist_in_keys=["loc", "scale"],
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
            fields={
                action: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                loc: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                observation: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32),
                scale: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                state_value: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> td_clone = td_module.get_policy_operator()(td.clone())
        >>> print(td_clone)  # no value
        TensorDict(
            fields={
                action: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                loc: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                observation: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([3, 1]), dtype=torch.float32),
                scale: Tensor(torch.Size([3, 4]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> td_clone = td_module.get_value_operator()(td.clone())
        >>> print(td_clone)  # no action
        TensorDict(
            fields={
                observation: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                state_value: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        policy_operator: SafeModule,
        value_operator: SafeModule,
    ):
        super().__init__(
            policy_operator,
            value_operator,
        )

    def get_policy_operator(self) -> SafeSequential:
        """Returns a stand-alone policy operator that maps an observation to an action."""
        return self.module[0]

    def get_value_operator(self) -> SafeSequential:
        """Returns a stand-alone value network operator that maps an observation to a value estimate."""
        return self.module[1]
