# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch

from tensordict import TensorDictBase, unravel_key
from tensordict.nn import (
    dispatch,
    TensorDictModule,
    TensorDictModuleBase,
    TensorDictModuleWrapper,
    TensorDictSequential,
)
from tensordict.utils import NestedKey
from torch import nn
from torch.distributions import Categorical

from torchrl.data.tensor_specs import CompositeSpec, TensorSpec
from torchrl.data.utils import _process_action_space_spec
from torchrl.modules.models.models import DistributionalDQNnet
from torchrl.modules.tensordict_module.common import SafeModule
from torchrl.modules.tensordict_module.probabilistic import (
    SafeProbabilisticModule,
    SafeProbabilisticTensorDictSequential,
)
from torchrl.modules.tensordict_module.sequence import SafeSequential


class Actor(SafeModule):
    """General class for deterministic actors in RL.

    The Actor class comes with default values for the out_keys (["action"])
    and if the spec is provided but not as a CompositeSpec object, it will be
    automatically translated into :obj:`spec = CompositeSpec(action=spec)`

    Args:
        module (nn.Module): a :class:`torch.nn.Module` used to map the input to
            the output parameter space.
        in_keys (iterable of str, optional): keys to be read from input
            tensordict and passed to the module. If it
            contains more than one element, the values will be passed in the
            order given by the in_keys iterable.
            Defaults to ``["observation"]``.
        out_keys (iterable of str): keys to be written to the input tensordict.
            The length of out_keys must match the
            number of tensors returned by the embedded module. Using "_" as a
            key avoid writing tensor to output.
            Defaults to ``["action"]``.
        spec (TensorSpec, optional): Keyword-only argument.
            Specs of the output tensor. If the module
            outputs multiple output tensors,
            spec characterize the space of the first output tensor.
        safe (bool): Keyword-only argument.
            If ``True``, the value of the output is checked against the
            input spec. Out-of-domain sampling can
            occur because of exploration policies or numerical under/overflow
            issues. If this value is out of bounds, it is projected back onto the
            desired space using the :obj:`TensorSpec.project`
            method. Default is ``False``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import UnboundedContinuousTensorSpec
        >>> from torchrl.modules import Actor
        >>> torch.manual_seed(0)
        >>> td = TensorDict({"observation": torch.randn(3, 4)}, [3,])
        >>> action_spec = UnboundedContinuousTensorSpec(4)
        >>> module = torch.nn.Linear(4, 4)
        >>> td_module = Actor(
        ...    module=module,
        ...    spec=action_spec,
        ...    )
        >>> td_module(td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> print(td.get("action"))
        tensor([[-1.3635, -0.0340,  0.1476, -1.3911],
                [-0.1664,  0.5455,  0.2247, -0.4583],
                [-0.2916,  0.2160,  0.5337, -0.5193]], grad_fn=<AddmmBackward0>)

    """

    def __init__(
        self,
        module: nn.Module,
        in_keys: Optional[Sequence[NestedKey]] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
        *,
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
            module,
            in_keys=in_keys,
            out_keys=out_keys,
            spec=spec,
            **kwargs,
        )


class ProbabilisticActor(SafeProbabilisticTensorDictSequential):
    """General class for probabilistic actors in RL.

    The Actor class comes with default values for the out_keys (["action"])
    and if the spec is provided but not as a CompositeSpec object, it will be
    automatically translated into :obj:`spec = CompositeSpec(action=spec)`

    Args:
        module (nn.Module): a :class:`torch.nn.Module` used to map the input to
            the output parameter space.
        in_keys (str or iterable of str or dict): key(s) that will be read from the
            input TensorDict and used to build the distribution. Importantly, if it's an
            iterable of string or a string, those keys must match the keywords used by
            the distribution class of interest, e.g. :obj:`"loc"` and :obj:`"scale"` for
            the Normal distribution and similar. If in_keys is a dictionary,, the keys
            are the keys of the distribution and the values are the keys in the
            tensordict that will get match to the corresponding distribution keys.
        out_keys (str or iterable of str): keys where the sampled values will be
            written. Importantly, if these keys are found in the input TensorDict, the
            sampling step will be skipped.
        spec (TensorSpec, optional): keyword-only argument containing the specs
            of the output tensor. If the module outputs multiple output tensors,
            spec characterize the space of the first output tensor.
        safe (bool): keyword-only argument. if ``True``, the value of the output is checked against the
            input spec. Out-of-domain sampling can
            occur because of exploration policies or numerical under/overflow
            issues. If this value is out of bounds, it is projected back onto the
            desired space using the :obj:`TensorSpec.project`
            method. Default is ``False``.
        default_interaction_type=InteractionType.RANDOM (str, optional): keyword-only argument.
            Default method to be used to retrieve
            the output value. Should be one of: 'mode', 'median', 'mean' or 'random'
            (in which case the value is sampled randomly from the distribution). Default
            is 'mode'.
            Note: When a sample is drawn, the :obj:`ProbabilisticTDModule` instance will
            first look for the interaction mode dictated by the `interaction_typ()`
            global function. If this returns `None` (its default value), then the
            `default_interaction_type` of the `ProbabilisticTDModule` instance will be
            used. Note that DataCollector instances will use `set_interaction_type` to
            :class:`tensordict.nn.InteractionType.RANDOM` by default.
        distribution_class (Type, optional): keyword-only argument.
            A :class:`torch.distributions.Distribution` class to
            be used for sampling.
            Default is :class:`tensordict.nn.distributions.Delta`.
        distribution_kwargs (dict, optional): keyword-only argument.
            Keyword-argument pairs to be passed to the distribution.
        return_log_prob (bool, optional): keyword-only argument.
            If ``True``, the log-probability of the
            distribution sample will be written in the tensordict with the key
            `'sample_log_prob'`. Default is ``False``.
        cache_dist (bool, optional): keyword-only argument.
            EXPERIMENTAL: if ``True``, the parameters of the
            distribution (i.e. the output of the module) will be written to the
            tensordict along with the sample. Those parameters can be used to re-compute
            the original distribution later on (e.g. to compute the divergence between
            the distribution used to sample the action and the updated distribution in
            PPO). Default is ``False``.
        n_empirical_estimate (int, optional): keyword-only argument.
            Number of samples to compute the empirical
            mean when it is not available. Defaults to 1000.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule, make_functional
        >>> from torchrl.data import BoundedTensorSpec
        >>> from torchrl.modules import ProbabilisticActor, NormalParamWrapper, TanhNormal
        >>> td = TensorDict({"observation": torch.randn(3, 4)}, [3,])
        >>> action_spec = BoundedTensorSpec(shape=torch.Size([4]),
        ...    low=-1, high=1)
        >>> module = NormalParamWrapper(torch.nn.Linear(4, 8))
        >>> tensordict_module = TensorDictModule(module, in_keys=["observation"], out_keys=["loc", "scale"])
        >>> td_module = ProbabilisticActor(
        ...    module=tensordict_module,
        ...    spec=action_spec,
        ...    in_keys=["loc", "scale"],
        ...    distribution_class=TanhNormal,
        ...    )
        >>> params = make_functional(td_module)
        >>> td = td_module(td, params=params)
        >>> td
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                loc: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                scale: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)

    Probabilistic actors also support compound actions through the
    :class:`tensordict.nn.CompositeDistribution` class. This distribution takes
    a tensordict as input (typically `"params"`) and reads it as a whole: the
    content of this tensordict is the input to the distributions contained in the
    compound one.

    Examples:
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import CompositeDistribution, TensorDictModule
        >>> from torchrl.modules import ProbabilisticActor
        >>> from torch import nn, distributions as d
        >>> import torch
        >>>
        >>> class Module(nn.Module):
        ...     def forward(self, x):
        ...         return x[..., :3], x[..., 3:6], x[..., 6:]
        >>> module = TensorDictModule(Module(),
        ...                           in_keys=["x"],
        ...                           out_keys=[("params", "normal", "loc"),
        ...                              ("params", "normal", "scale"),
        ...                              ("params", "categ", "logits")])
        >>> actor = ProbabilisticActor(module,
        ...                            in_keys=["params"],
        ...                            distribution_class=CompositeDistribution,
        ...                            distribution_kwargs={"distribution_map": {
        ...                                 "normal": d.Normal, "categ": d.Categorical}}
        ...                           )
        >>> data = TensorDict({"x": torch.rand(10)}, [])
        >>> actor(data)
        TensorDict(
            fields={
                categ: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                normal: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                params: TensorDict(
                    fields={
                        categ: TensorDict(
                            fields={
                                logits: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=None,
                            is_shared=False),
                        normal: TensorDict(
                            fields={
                                loc: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                                scale: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([]),
                            device=None,
                            is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False),
                x: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        module: TensorDictModule,
        in_keys: Union[NestedKey, Sequence[NestedKey]],
        out_keys: Optional[Sequence[NestedKey]] = None,
        *,
        spec: Optional[TensorSpec] = None,
        **kwargs,
    ):
        if out_keys is None:
            out_keys = ["action"]
        if (
            len(out_keys) == 1
            and spec is not None
            and not isinstance(spec, CompositeSpec)
        ):
            spec = CompositeSpec({out_keys[0]: spec})

        super().__init__(
            module,
            SafeProbabilisticModule(
                in_keys=in_keys, out_keys=out_keys, spec=spec, **kwargs
            ),
        )


class ValueOperator(TensorDictModule):
    """General class for value functions in RL.

    The ValueOperator class comes with default values for the in_keys and
    out_keys arguments (["observation"] and ["state_value"] or
    ["state_action_value"], respectively and depending on whether the "action"
    key is part of the in_keys list).

    Args:
        module (nn.Module): a :class:`torch.nn.Module` used to map the input to
            the output parameter space.
        in_keys (iterable of str, optional): keys to be read from input
            tensordict and passed to the module. If it
            contains more than one element, the values will be passed in the
            order given by the in_keys iterable.
            Defaults to ``["observation"]``.
        out_keys (iterable of str): keys to be written to the input tensordict.
            The length of out_keys must match the
            number of tensors returned by the embedded module. Using "_" as a
            key avoid writing tensor to output.
            Defaults to ``["action"]``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import make_functional
        >>> from torch import nn
        >>> from torchrl.data import UnboundedContinuousTensorSpec
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
        >>> td = td_module(td, params=params)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 2]), device=cpu, dtype=torch.float32, is_shared=False),
                observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                state_action_value: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)


    """

    def __init__(
        self,
        module: nn.Module,
        in_keys: Optional[Sequence[NestedKey]] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
    ) -> None:
        if in_keys is None:
            in_keys = ["observation"]
        if out_keys is None:
            out_keys = (
                ["state_value"] if "action" not in in_keys else ["state_action_value"]
            )
        super().__init__(
            module=module,
            in_keys=in_keys,
            out_keys=out_keys,
        )


class QValueModule(TensorDictModuleBase):
    """Q-Value TensorDictModule for Q-value policies.

    This module processes a tensor containing action value into is argmax
    component (i.e. the resulting greedy action), following a given
    action space (one-hot, binary or categorical).
    It works with both tensordict and regular tensors.

    Args:
        action_space (str, optional): Action space. Must be one of
            ``"one-hot"``, ``"mult-one-hot"``, ``"binary"`` or ``"categorical"``.
            This argument is exclusive with ``spec``, since ``spec``
            conditions the action_space.
        action_value_key (str or tuple of str, optional): The input key
            representing the action value. Defaults to ``"action_value"``.
        action_mask_key (str or tuple of str, optional): The input key
            representing the action mask. Defaults to ``"None"`` (equivalent to no masking).
        out_keys (list of str or tuple of str, optional): The output keys
            representing the actions, action values and chosen action value.
            Defaults to ``["action", "action_value", "chosen_action_value"]``.
        var_nums (int, optional): if ``action_space = "mult-one-hot"``,
            this value represents the cardinality of each
            action component.
        spec (TensorSpec, optional): if provided, the specs of the action (and/or
            other outputs). This is exclusive with ``action_space``, as the spec
            conditions the action space.
        safe (bool): if ``True``, the value of the output is checked against the
            input spec. Out-of-domain sampling can
            occur because of exploration policies or numerical under/overflow issues.
            If this value is out of bounds, it is projected back onto the
            desired space using the :obj:`TensorSpec.project`
            method. Default is ``False``.

    Returns:
        if the input is a single tensor, a triplet containing the chosen action,
        the values and the value of the chose action is returned. If a tensordict
        is provided, it is updated with these entries at the keys indicated by the
        ``out_keys`` field.

    Examples:
        >>> from tensordict import TensorDict
        >>> action_space = "categorical"
        >>> action_value_key = "my_action_value"
        >>> actor = QValueModule(action_space, action_value_key=action_value_key)
        >>> # This module works with both tensordict and regular tensors:
        >>> value = torch.zeros(4)
        >>> value[-1] = 1
        >>> actor(my_action_value=value)
        (tensor(3), tensor([0., 0., 0., 1.]), tensor([1.]))
        >>> actor(value)
        (tensor(3), tensor([0., 0., 0., 1.]), tensor([1.]))
        >>> actor(TensorDict({action_value_key: value}, []))
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                action_value: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                chosen_action_value: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                my_action_value: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        action_space: Optional[str],
        action_value_key: Optional[NestedKey] = None,
        action_mask_key: Optional[NestedKey] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
        var_nums: Optional[int] = None,
        spec: Optional[TensorSpec] = None,
        safe: bool = False,
    ):
        if isinstance(action_space, TensorSpec):
            warnings.warn(
                "Using specs in action_space will be deprecated soon,"
                " please use the 'spec' argument if you want to provide an action spec",
                category=DeprecationWarning,
            )
        action_space, spec = _process_action_space_spec(action_space, spec)
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
                f"action_space must be one of {list(self.action_func_mapping.keys())}, got {action_space}"
            )
        if action_value_key is None:
            action_value_key = "action_value"
        self.action_mask_key = action_mask_key
        in_keys = [action_value_key]
        if self.action_mask_key is not None:
            in_keys.append(self.action_mask_key)
        self.in_keys = in_keys
        if out_keys is None:
            out_keys = ["action", action_value_key, "chosen_action_value"]
        elif action_value_key not in out_keys:
            raise RuntimeError(
                f"Expected the action-value key to be '{action_value_key}' but got {out_keys[1]} instead."
            )
        self.out_keys = out_keys
        action_key = out_keys[0]
        if not isinstance(spec, CompositeSpec):
            spec = CompositeSpec({action_key: spec})
        super().__init__()
        self.register_spec(safe=safe, spec=spec)

    register_spec = SafeModule.register_spec

    @property
    def spec(self) -> CompositeSpec:
        return self._spec

    @spec.setter
    def spec(self, spec: CompositeSpec) -> None:
        if not isinstance(spec, CompositeSpec):
            raise RuntimeError(
                f"Trying to set an object of type {type(spec)} as a tensorspec but expected a CompositeSpec instance."
            )
        self._spec = spec

    @property
    def action_value_key(self):
        return self.in_keys[0]

    @dispatch(auto_batch_size=False)
    def forward(self, tensordict: torch.Tensor) -> TensorDictBase:
        action_values = tensordict.get(self.action_value_key, None)
        if action_values is None:
            raise KeyError(
                f"Action value key {self.action_value_key} not found in {tensordict}."
            )
        if self.action_mask_key is not None:
            action_mask = tensordict.get(self.action_mask_key, None)
            if action_mask is None:
                raise KeyError(
                    f"Action mask key {self.action_mask_key} not found in {tensordict}."
                )
            action_values = torch.where(
                action_mask, action_values, torch.finfo(action_values.dtype).min
            )

        action = self.action_func_mapping[self.action_space](action_values)

        action_value_func = self.action_value_func_mapping.get(
            self.action_space, self._default_action_value
        )
        chosen_action_value = action_value_func(action_values, action)
        tensordict.update(
            dict(zip(self.out_keys, (action, action_values, chosen_action_value)))
        )
        return tensordict

    @staticmethod
    def _one_hot(value: torch.Tensor) -> torch.Tensor:
        out = (value == value.max(dim=-1, keepdim=True)[0]).to(torch.long)
        return out

    @staticmethod
    def _categorical(value: torch.Tensor) -> torch.Tensor:
        return torch.argmax(value, dim=-1).to(torch.long)

    def _mult_one_hot(
        self, value: torch.Tensor, support: torch.Tensor = None
    ) -> torch.Tensor:
        if self.var_nums is None:
            raise ValueError(
                "var_nums must be provided to the constructor for multi one-hot action spaces."
            )
        values = value.split(self.var_nums, dim=-1)
        return torch.cat(
            [
                self._one_hot(
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
        return values.gather(-1, action.unsqueeze(-1))
        # if values.ndim == 1:
        #     return values[action].unsqueeze(-1)
        # batch_size = values.size(0)
        # return values[range(batch_size), action].unsqueeze(-1)


class DistributionalQValueModule(QValueModule):
    """Distributional Q-Value hook for Q-value policies.

    This module processes a tensor containing action value logits into is argmax
    component (i.e. the resulting greedy action), following a given
    action space (one-hot, binary or categorical).
    It works with both tensordict and regular tensors.

    The input action value is expected to be the result of a log-softmax
    operation.

    For more details regarding Distributional DQN, refer to "A Distributional Perspective on Reinforcement Learning",
    https://arxiv.org/pdf/1707.06887.pdf

    Args:
        action_space (str, optional): Action space. Must be one of
            ``"one-hot"``, ``"mult-one-hot"``, ``"binary"`` or ``"categorical"``.
            This argument is exclusive with ``spec``, since ``spec``
            conditions the action_space.
        support (torch.Tensor): support of the action values.
        action_value_key (str or tuple of str, optional): The input key
            representing the action value. Defaults to ``"action_value"``.
        action_mask_key (str or tuple of str, optional): The input key
            representing the action mask. Defaults to ``"None"`` (equivalent to no masking).
        out_keys (list of str or tuple of str, optional): The output keys
            representing the actions and action values.
            Defaults to ``["action", "action_value"]``.
        var_nums (int, optional): if ``action_space = "mult-one-hot"``,
            this value represents the cardinality of each
            action component.
        spec (TensorSpec, optional): if provided, the specs of the action (and/or
            other outputs). This is exclusive with ``action_space``, as the spec
            conditions the action space.
        safe (bool): if ``True``, the value of the output is checked against the
            input spec. Out-of-domain sampling can
            occur because of exploration policies or numerical under/overflow issues.
            If this value is out of bounds, it is projected back onto the
            desired space using the :obj:`TensorSpec.project`
            method. Default is ``False``.

    Examples:
        >>> from tensordict import TensorDict
        >>> torch.manual_seed(0)
        >>> action_space = "categorical"
        >>> action_value_key = "my_action_value"
        >>> support = torch.tensor([-1, 0.0, 1.0]) # the action value is between -1 and 1
        >>> actor = DistributionalQValueModule(action_space, support=support, action_value_key=action_value_key)
        >>> # This module works with both tensordict and regular tensors:
        >>> value = torch.full((3, 4), -100)
        >>> # the first bin (-1) of the first action is high: there's a high chance that it has a low value
        >>> value[0, 0] = 0
        >>> # the second bin (0) of the second action is high: there's a high chance that it has an intermediate value
        >>> value[1, 1] = 0
        >>> # the third bin (0) of the thid action is high: there's a high chance that it has an high value
        >>> value[2, 2] = 0
        >>> actor(my_action_value=value)
        (tensor(2), tensor([[   0, -100, -100, -100],
                [-100,    0, -100, -100],
                [-100, -100,    0, -100]]))
        >>> actor(value)
        (tensor(2), tensor([[   0, -100, -100, -100],
                [-100,    0, -100, -100],
                [-100, -100,    0, -100]]))
        >>> actor(TensorDict({action_value_key: value}, []))
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                my_action_value: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.int64, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        action_space: Optional[str],
        support: torch.Tensor,
        action_value_key: Optional[NestedKey] = None,
        action_mask_key: Optional[NestedKey] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
        var_nums: Optional[int] = None,
        spec: TensorSpec = None,
        safe: bool = False,
    ):
        if action_value_key is None:
            action_value_key = "action_value"
        if out_keys is None:
            out_keys = ["action", action_value_key]
        super().__init__(
            action_space=action_space,
            action_value_key=action_value_key,
            action_mask_key=action_mask_key,
            out_keys=out_keys,
            var_nums=var_nums,
            spec=spec,
            safe=safe,
        )
        self.register_buffer("support", support)

    @dispatch(auto_batch_size=False)
    def forward(self, tensordict: torch.Tensor) -> TensorDictBase:
        action_values = tensordict.get(self.action_value_key, None)
        if action_values is None:
            raise KeyError(
                f"Action value key {self.action_value_key} not found in {tensordict}."
            )
        if self.action_mask_key is not None:
            action_mask = tensordict.get(self.action_mask_key, None)
            if action_mask is None:
                raise KeyError(
                    f"Action mask key {self.action_mask_key} not found in {tensordict}."
                )
            action_values = torch.where(
                action_mask, action_values, torch.finfo(action_values.dtype).min
            )

        action = self.action_func_mapping[self.action_space](action_values)

        tensordict.update(
            dict(
                zip(
                    self.out_keys,
                    (
                        action,
                        action_values,
                    ),
                )
            )
        )
        return tensordict

    def _support_expected(
        self, log_softmax_values: torch.Tensor, support=None
    ) -> torch.Tensor:
        if support is None:
            support = self.support
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

    def _one_hot(self, value: torch.Tensor, support=None) -> torch.Tensor:
        if support is None:
            support = self.support
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"got value of type {value.__class__.__name__}")
        if not isinstance(support, torch.Tensor):
            raise TypeError(f"got support of type {support.__class__.__name__}")
        value = self._support_expected(value)
        out = (value == value.max(dim=-1, keepdim=True)[0]).to(torch.long)
        return out

    def _mult_one_hot(self, value: torch.Tensor, support=None) -> torch.Tensor:
        if support is None:
            support = self.support
        values = value.split(self.var_nums, dim=-1)
        return torch.cat(
            [
                self._one_hot(_value, _support)
                for _value, _support in zip(values, support)
            ],
            -1,
        )

    def _categorical(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:
        value = self._support_expected(
            value,
        )
        return torch.argmax(value, dim=-1).to(torch.long)

    def _binary(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "'binary' is currently not supported for DistributionalQValueModule."
        )


class QValueHook:
    """Q-Value hook for Q-value policies.

    Given the output of a regular nn.Module, representing the values of the
    different discrete actions available,
    a QValueHook will transform these values into their argmax component (i.e.
    the resulting greedy action).

    Args:
        action_space (str): Action space. Must be one of
            ``"one-hot"``, ``"mult-one-hot"``, ``"binary"`` or ``"categorical"``.
        var_nums (int, optional): if ``action_space = "mult-one-hot"``,
            this value represents the cardinality of each
            action component.
        action_value_key (str or tuple of str, optional): to be used when hooked on
            a TensorDictModule. The input key representing the action value. Defaults
            to ``"action_value"``.
        action_mask_key (str or tuple of str, optional): The input key
            representing the action mask. Defaults to ``"None"`` (equivalent to no masking).
        out_keys (list of str or tuple of str, optional): to be used when hooked on
            a TensorDictModule. The output keys representing the actions, action values
            and chosen action value. Defaults to ``["action", "action_value", "chosen_action_value"]``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn.functional_modules import make_functional
        >>> from torch import nn
        >>> from torchrl.data import OneHotDiscreteTensorSpec
        >>> from torchrl.modules.tensordict_module.actors import QValueHook, Actor
        >>> td = TensorDict({'observation': torch.randn(5, 4)}, [5])
        >>> module = nn.Linear(4, 4)
        >>> hook = QValueHook("one_hot")
        >>> module.register_forward_hook(hook)
        >>> action_spec = OneHotDiscreteTensorSpec(4)
        >>> qvalue_actor = Actor(module=module, spec=action_spec, out_keys=["action", "action_value"])
        >>> td = qvalue_actor(td)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.int64, is_shared=False),
                action_value: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                observation: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([5]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        action_space: str,
        var_nums: Optional[int] = None,
        action_value_key: Optional[NestedKey] = None,
        action_mask_key: Optional[NestedKey] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
    ):
        if isinstance(action_space, TensorSpec):
            warnings.warn(
                "Using specs in action_space will be deprecated soon,"
                " please use the 'spec' argument if you want to provide an action spec",
                category=DeprecationWarning,
            )
        action_space, _ = _process_action_space_spec(action_space, None)

        self.qvalue_model = QValueModule(
            action_space=action_space,
            var_nums=var_nums,
            action_value_key=action_value_key,
            action_mask_key=action_mask_key,
            out_keys=out_keys,
        )
        action_value_key = self.qvalue_model.in_keys[0]
        if isinstance(action_value_key, tuple):
            action_value_key = "_".join(action_value_key)
        # uses "dispatch" to get and return tensors
        self.action_value_key = action_value_key

    def __call__(
        self, net: nn.Module, observation: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kwargs = {self.action_value_key: values}
        return self.qvalue_model(**kwargs)


class DistributionalQValueHook(QValueHook):
    """Distributional Q-Value hook for Q-value policies.

    Given the output of a mapping operator, representing the log-probability of the
    different action value bin available,
    a DistributionalQValueHook will transform these values into their argmax
    component using the provided support.

    For more details regarding Distributional DQN, refer to "A Distributional Perspective on Reinforcement Learning",
    https://arxiv.org/pdf/1707.06887.pdf

    Args:
        action_space (str): Action space. Must be one of
            ``"one-hot"``, ``"mult-one-hot"``, ``"binary"`` or ``"categorical"``.
        action_value_key (str or tuple of str, optional): to be used when hooked on
            a TensorDictModule. The input key representing the action value. Defaults
            to ``"action_value"``.
        action_mask_key (str or tuple of str, optional): The input key
            representing the action mask. Defaults to ``"None"`` (equivalent to no masking).
        support (torch.Tensor): support of the action values.
        var_nums (int, optional): if ``action_space = "mult-one-hot"``, this
            value represents the cardinality of each
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
        action_value_key: Optional[NestedKey] = None,
        action_mask_key: Optional[NestedKey] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
    ):
        if isinstance(action_space, TensorSpec):
            warnings.warn(
                "Using specs in action_space will be deprecated soon,"
                " please use the 'spec' argument if you want to provide an action spec",
                category=DeprecationWarning,
            )
        action_space, _ = _process_action_space_spec(action_space, None)
        self.qvalue_model = DistributionalQValueModule(
            action_space=action_space,
            var_nums=var_nums,
            support=support,
            action_value_key=action_value_key,
            action_mask_key=action_mask_key,
            out_keys=out_keys,
        )
        action_value_key = self.qvalue_model.in_keys[0]
        if isinstance(action_value_key, tuple):
            action_value_key = "_".join(action_value_key)
        # uses "dispatch" to get and return tensors
        self.action_value_key = action_value_key


class QValueActor(SafeSequential):
    """A Q-Value actor class.

    This class appends a :class:`~.QValueModule` after the input module
    such that the action values are used to select an action.

    Args:
        module (nn.Module): a :class:`torch.nn.Module` used to map the input to
            the output parameter space. If the class provided is not compatible
            with :class:`tensordict.nn.TensorDictModuleBase`, it will be
            wrapped in a :class:`tensordict.nn.TensorDictModule` with
            ``in_keys`` indicated by the following keyword argument.

    Keyword Args:
        in_keys (iterable of str, optional): If the class provided is not
            compatible with :class:`tensordict.nn.TensorDictModuleBase`, this
            list of keys indicates what observations need to be passed to the
            wrapped module to get the action values.
            Defaults to ``["observation"]``.
        spec (TensorSpec, optional): Keyword-only argument.
            Specs of the output tensor. If the module
            outputs multiple output tensors,
            spec characterize the space of the first output tensor.
        safe (bool): Keyword-only argument.
            If ``True``, the value of the output is checked against the
            input spec. Out-of-domain sampling can
            occur because of exploration policies or numerical under/overflow
            issues. If this value is out of bounds, it is projected back onto the
            desired space using the :obj:`TensorSpec.project`
            method. Default is ``False``.
        action_space (str, optional): Action space. Must be one of
            ``"one-hot"``, ``"mult-one-hot"``, ``"binary"`` or ``"categorical"``.
            This argument is exclusive with ``spec``, since ``spec``
            conditions the action_space.
        action_value_key (str or tuple of str, optional): if the input module
            is a :class:`tensordict.nn.TensorDictModuleBase` instance, it must
            match one of its output keys. Otherwise, this string represents
            the name of the action-value entry in the output tensordict.
        action_mask_key (str or tuple of str, optional): The input key
            representing the action mask. Defaults to ``"None"`` (equivalent to no masking).

    .. note::
        ``out_keys`` cannot be passed. If the module is a :class:`tensordict.nn.TensorDictModule`
        instance, the out_keys will be updated accordingly. For regular
        :class:`torch.nn.Module` instance, the triplet ``["action", action_value_key, "chosen_action_value"]``
        will be used.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn.functional_modules import make_functional
        >>> from torch import nn
        >>> from torchrl.data import OneHotDiscreteTensorSpec
        >>> from torchrl.modules.tensordict_module.actors import QValueActor
        >>> td = TensorDict({'observation': torch.randn(5, 4)}, [5])
        >>> # with a regular nn.Module
        >>> module = nn.Linear(4, 4)
        >>> action_spec = OneHotDiscreteTensorSpec(4)
        >>> qvalue_actor = QValueActor(module=module, spec=action_spec)
        >>> td = qvalue_actor(td)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.int64, is_shared=False),
                action_value: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                chosen_action_value: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                observation: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([5]),
            device=None,
            is_shared=False)
        >>> # with a TensorDictModule
        >>> td = TensorDict({'obs': torch.randn(5, 4)}, [5])
        >>> module = TensorDictModule(lambda x: x, in_keys=["obs"], out_keys=["action_value"])
        >>> action_spec = OneHotDiscreteTensorSpec(4)
        >>> qvalue_actor = QValueActor(module=module, spec=action_spec)
        >>> td = qvalue_actor(td)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.int64, is_shared=False),
                action_value: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                chosen_action_value: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                obs: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([5]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        module,
        *,
        in_keys=None,
        spec=None,
        safe=False,
        action_space: Optional[str] = None,
        action_value_key=None,
        action_mask_key: Optional[NestedKey] = None,
    ):
        if isinstance(action_space, TensorSpec):
            warnings.warn(
                "Using specs in action_space will be deprecated soon,"
                " please use the 'spec' argument if you want to provide an action spec",
                category=DeprecationWarning,
            )
        action_space, spec = _process_action_space_spec(action_space, spec)

        self.action_space = action_space
        self.action_value_key = action_value_key
        if action_value_key is None:
            action_value_key = "action_value"
        out_keys = [
            "action",
            action_value_key,
            "chosen_action_value",
        ]
        if isinstance(module, TensorDictModuleBase):
            if action_value_key not in module.out_keys:
                raise KeyError(
                    f"The key '{action_value_key}' is not part of the module out-keys."
                )
        else:
            if in_keys is None:
                in_keys = ["observation"]
            module = TensorDictModule(
                module, in_keys=in_keys, out_keys=[action_value_key]
            )
        if spec is None:
            spec = CompositeSpec()
        if isinstance(spec, CompositeSpec):
            spec = spec.clone()
            if "action" not in spec.keys():
                spec["action"] = None
        else:
            spec = CompositeSpec(action=spec, shape=spec.shape[:-1])
        spec[action_value_key] = None
        spec["chosen_action_value"] = None
        qvalue = QValueModule(
            action_value_key=action_value_key,
            out_keys=out_keys,
            spec=spec,
            safe=safe,
            action_space=action_space,
            action_mask_key=action_mask_key,
        )

        super().__init__(module, qvalue)


class DistributionalQValueActor(QValueActor):
    """A Distributional DQN actor class.

    This class appends a :class:`~.QValueModule` after the input module
    such that the action values are used to select an action.

    Args:
        module (nn.Module): a :class:`torch.nn.Module` used to map the input to
            the output parameter space.
            If the module isn't of type :class:`torchrl.modules.DistributionalDQNnet`,
            :class:`~.DistributionalQValueActor` will ensure that a log-softmax
            operation is applied to the action value tensor along dimension ``-2``.
            This can be deactivated by turning off the ``make_log_softmax``
            keyword argument.

    Keyword Args:
        in_keys (iterable of str, optional): keys to be read from input
            tensordict and passed to the module. If it
            contains more than one element, the values will be passed in the
            order given by the in_keys iterable.
            Defaults to ``["observation"]``.
        spec (TensorSpec, optional): Keyword-only argument.
            Specs of the output tensor. If the module
            outputs multiple output tensors,
            spec characterize the space of the first output tensor.
        safe (bool): Keyword-only argument.
            If ``True``, the value of the output is checked against the
            input spec. Out-of-domain sampling can
            occur because of exploration policies or numerical under/overflow
            issues. If this value is out of bounds, it is projected back onto the
            desired space using the :obj:`TensorSpec.project`
            method. Default is ``False``.
        var_nums (int, optional): if ``action_space = "mult-one-hot"``,
            this value represents the cardinality of each
            action component.
        support (torch.Tensor): support of the action values.
        action_space (str, optional): Action space. Must be one of
            ``"one-hot"``, ``"mult-one-hot"``, ``"binary"`` or ``"categorical"``.
            This argument is exclusive with ``spec``, since ``spec``
            conditions the action_space.
        make_log_softmax (bool, optional): if ``True`` and if the module is not
            of type :class:`torchrl.modules.DistributionalDQNnet`, a log-softmax
            operation will be applied along dimension -2 of the action value tensor.
        action_value_key (str or tuple of str, optional): if the input module
            is a :class:`tensordict.nn.TensorDictModuleBase` instance, it must
            match one of its output keys. Otherwise, this string represents
            the name of the action-value entry in the output tensordict.
        action_mask_key (str or tuple of str, optional): The input key
            representing the action mask. Defaults to ``"None"`` (equivalent to no masking).

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule, TensorDictSequential
        >>> from torch import nn
        >>> from torchrl.data import OneHotDiscreteTensorSpec
        >>> from torchrl.modules import DistributionalQValueActor, MLP
        >>> td = TensorDict({'observation': torch.randn(5, 4)}, [5])
        >>> nbins = 3
        >>> module = MLP(out_features=(nbins, 4), depth=2)
        >>> # let us make sure that the output is a log-softmax
        >>> module = TensorDictSequential(
        ...     TensorDictModule(module, ["observation"], ["action_value"]),
        ...     TensorDictModule(lambda x: x.log_softmax(-2), ["action_value"], ["action_value"]),
        ... )
        >>> action_spec = OneHotDiscreteTensorSpec(4)
        >>> qvalue_actor = DistributionalQValueActor(
        ...     module=module,
        ...     spec=action_spec,
        ...     support=torch.arange(nbins))
        >>> td = qvalue_actor(td)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.int64, is_shared=False),
                action_value: Tensor(shape=torch.Size([5, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                observation: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([5]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        module,
        support: torch.Tensor,
        in_keys=None,
        spec=None,
        safe=False,
        var_nums: Optional[int] = None,
        action_space: Optional[str] = None,
        action_value_key: str = "action_value",
        action_mask_key: Optional[NestedKey] = None,
        make_log_softmax: bool = True,
    ):
        if isinstance(action_space, TensorSpec):
            warnings.warn(
                "Using specs in action_space will be deprecated soon,"
                " please use the 'spec' argument if you want to provide an action spec",
                category=DeprecationWarning,
            )
        action_space, spec = _process_action_space_spec(action_space, spec)
        self.action_space = action_space
        self.action_value_key = action_value_key
        out_keys = [
            "action",
            action_value_key,
        ]
        if isinstance(module, TensorDictModuleBase):
            if action_value_key not in module.out_keys:
                raise KeyError(
                    f"The key '{action_value_key}' is not part of the module out-keys."
                )
        else:
            if in_keys is None:
                in_keys = ["observation"]
            module = TensorDictModule(
                module, in_keys=in_keys, out_keys=[action_value_key]
            )
        if spec is None:
            spec = CompositeSpec()
        if isinstance(spec, CompositeSpec):
            spec = spec.clone()
            if "action" not in spec.keys():
                spec["action"] = None
        else:
            spec = CompositeSpec(action=spec, shape=spec.shape[:-1])
        spec[action_value_key] = None

        qvalue = DistributionalQValueModule(
            action_value_key=action_value_key,
            out_keys=out_keys,
            spec=spec,
            safe=safe,
            action_space=action_space,
            action_mask_key=action_mask_key,
            support=support,
            var_nums=var_nums,
        )
        self.make_log_softmax = make_log_softmax
        if make_log_softmax and not isinstance(module, DistributionalDQNnet):
            log_softmax_module = DistributionalDQNnet(
                in_keys=qvalue.in_keys, out_keys=qvalue.in_keys
            )
            super(QValueActor, self).__init__(module, log_softmax_module, qvalue)
        else:
            super(QValueActor, self).__init__(module, qvalue)
        self.register_buffer("support", support)


class ActorValueOperator(SafeSequential):
    """Actor-value operator.

    This class wraps together an actor and a value model that share a common
    observation embedding network:

    .. aafig::
        :aspect: 60
        :scale: 120
        :proportional:
        :textual:

               +---------------+
               |Observation (s)|
               +---------------+
                        |
                       "common"
                        |
                        v
                 +------------+
                 |Hidden state|
                 +------------+
                   |         |
                  actor     critic
                   |         |
                   v         v
        +-------------+ +------------+
        |Action (a(s))| |Value (V(s))|
        +-------------+ +------------+

    .. note::
      For a similar class that returns an action and a Quality value :math:`Q(s, a)`
      see :class:`~.ActorCriticOperator`. For a version without common embeddig
      refet to :class:`~.ActorCriticWrapper`.

    To facilitate the workflow, this  class comes with a get_policy_operator() and get_value_operator() methods, which
    will both return a standalone TDModule with the dedicated functionality.

    Args:
        common_operator (TensorDictModule): a common operator that reads
            observations and produces a hidden variable
        policy_operator (TensorDictModule): a policy operator that reads the
            hidden variable and returns an action
        value_operator (TensorDictModule): a value operator, that reads the
            hidden variable and returns a value

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.modules import ProbabilisticActor, SafeModule
        >>> from torchrl.modules import ValueOperator, TanhNormal, ActorValueOperator, NormalParamWrapper
        >>> module_hidden = torch.nn.Linear(4, 4)
        >>> td_module_hidden = SafeModule(
        ...    module=module_hidden,
        ...    in_keys=["observation"],
        ...    out_keys=["hidden"],
        ...    )
        >>> module_action = TensorDictModule(
        ...     NormalParamWrapper(torch.nn.Linear(4, 8)),
        ...     in_keys=["hidden"],
        ...     out_keys=["loc", "scale"],
        ...     )
        >>> td_module_action = ProbabilisticActor(
        ...    module=module_action,
        ...    in_keys=["loc", "scale"],
        ...    out_keys=["action"],
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
                action: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                hidden: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                loc: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                sample_log_prob: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                scale: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                state_value: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> td_clone = td_module.get_policy_operator()(td.clone())
        >>> print(td_clone)  # no value
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                hidden: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                loc: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                sample_log_prob: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                scale: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> td_clone = td_module.get_value_operator()(td.clone())
        >>> print(td_clone)  # no action
        TensorDict(
            fields={
                hidden: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                state_value: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        common_operator: TensorDictModule,
        policy_operator: TensorDictModule,
        value_operator: TensorDictModule,
    ):
        super().__init__(
            common_operator,
            policy_operator,
            value_operator,
        )

    def get_policy_operator(self) -> SafeSequential:
        """Returns a standalone policy operator that maps an observation to an action."""
        if isinstance(self.module[1], SafeProbabilisticTensorDictSequential):
            return SafeProbabilisticTensorDictSequential(
                self.module[0], *self.module[1].module
            )
        return SafeSequential(self.module[0], self.module[1])

    def get_value_operator(self) -> SafeSequential:
        """Returns a standalone value network operator that maps an observation to a value estimate."""
        return SafeSequential(self.module[0], self.module[2])

    def get_policy_head(self) -> SafeSequential:
        """Returns the policy head."""
        return self.module[1]

    def get_value_head(self) -> SafeSequential:
        """Returns the value head."""
        return self.module[2]


class ActorCriticOperator(ActorValueOperator):
    """Actor-critic operator.

    This class wraps together an actor and a value model that share a common
    observation embedding network:

    .. aafig::
        :aspect: 60
        :scale: 120
        :proportional:
        :textual:

                 +---------------+
                 |Observation (s)|
                 +---------------+
                         |
                         v
                        "common"
                         |
                         v
                  +------------+
                  |Hidden state|
                  +------------+
                    |        |
                    v        v
                   actor --> critic
                    |        |
                    v        v
            +-------------+ +----------------+
            |Action (a(s))| |Quality (Q(s,a))|
            +-------------+ +----------------+

    .. note::
      For a similar class that returns an action and a state-value :math:`V(s)`
      see :class:`~.ActorValueOperator`.


    To facilitate the workflow, this  class comes with a get_policy_operator() method, which
    will both return a standalone TDModule with the dedicated functionality. The get_critic_operator will return the
    parent object, as the value is computed based on the policy output.

    Args:
        common_operator (TensorDictModule): a common operator that reads
            observations and produces a hidden variable
        policy_operator (TensorDictModule): a policy operator that reads the
            hidden variable and returns an action
        value_operator (TensorDictModule): a value operator, that reads the
            hidden variable and returns a value

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.modules import ProbabilisticActor
        >>> from torchrl.modules import  ValueOperator, TanhNormal, ActorCriticOperator, NormalParamWrapper, MLP
        >>> module_hidden = torch.nn.Linear(4, 4)
        >>> td_module_hidden = SafeModule(
        ...    module=module_hidden,
        ...    in_keys=["observation"],
        ...    out_keys=["hidden"],
        ...    )
        >>> module_action = NormalParamWrapper(torch.nn.Linear(4, 8))
        >>> module_action = TensorDictModule(module_action, in_keys=["hidden"], out_keys=["loc", "scale"])
        >>> td_module_action = ProbabilisticActor(
        ...    module=module_action,
        ...    in_keys=["loc", "scale"],
        ...    out_keys=["action"],
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
                action: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                hidden: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                loc: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                sample_log_prob: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                scale: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                state_action_value: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> td_clone = td_module.get_policy_operator()(td.clone())
        >>> print(td_clone)  # no value
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                hidden: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                loc: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                sample_log_prob: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                scale: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> td_clone = td_module.get_critic_operator()(td.clone())
        >>> print(td_clone)  # no action
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                hidden: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                loc: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                sample_log_prob: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                scale: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                state_action_value: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        common_operator: TensorDictModule,
        policy_operator: TensorDictModule,
        value_operator: TensorDictModule,
    ):
        super().__init__(
            common_operator,
            policy_operator,
            value_operator,
        )
        if self[2].out_keys[0] == "state_value":
            raise RuntimeError(
                "Value out_key is state_value, which may lead to errors in downstream usages"
                "of that module. Consider setting `'state_action_value'` instead."
                "Make also sure that `'action'` is amongst the input keys of the value network."
                "If you are confident that action should not be used to compute the value, please"
                "user `ActorValueOperator` instead."
            )

    def get_critic_operator(self) -> TensorDictModuleWrapper:
        """Returns a standalone critic network operator that maps a state-action pair to a critic estimate."""
        return self

    def get_value_operator(self) -> TensorDictModuleWrapper:
        raise RuntimeError(
            "value_operator is the term used for operators that associate a value with a "
            "state/observation. This class computes the value of a state-action pair: to get the "
            "network computing this value, please call td_sequence.get_critic_operator()"
        )

    def get_policy_head(self) -> SafeSequential:
        """Returns the policy head."""
        return self.module[1]

    def get_value_head(self) -> SafeSequential:
        """Returns the value head."""
        return self.module[2]


class ActorCriticWrapper(SafeSequential):
    """Actor-value operator without common module.

    This class wraps together an actor and a value model that do not share a common observation embedding network:

    .. aafig::
        :aspect: 60
        :scale: 120
        :proportional:
        :textual:

                 +---------------+
                 |Observation (s)|
                 +---------------+
                    |    |    |
                    v    |    v
                   actor |    critic
                    |    |    |
                    v    |    v
        +-------------+  |  +------------+
        |Action (a(s))|  |  |Value (V(s))|
        +-------------+  |  +------------+


    To facilitate the workflow, this  class comes with a get_policy_operator() and get_value_operator() methods, which
    will both return a standalone TDModule with the dedicated functionality.

    Args:
        policy_operator (TensorDictModule): a policy operator that reads the hidden variable and returns an action
        value_operator (TensorDictModule): a value operator, that reads the hidden variable and returns a value

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules import (
        ...      ActorCriticWrapper,
        ...      ProbabilisticActor,
        ...      NormalParamWrapper,
        ...      TanhNormal,
        ...      ValueOperator,
        ...  )
        >>> action_module = TensorDictModule(
        ...        NormalParamWrapper(torch.nn.Linear(4, 8)),
        ...        in_keys=["observation"],
        ...        out_keys=["loc", "scale"],
        ...    )
        >>> td_module_action = ProbabilisticActor(
        ...    module=action_module,
        ...    in_keys=["loc", "scale"],
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
                action: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                loc: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                sample_log_prob: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                scale: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                state_value: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> td_clone = td_module.get_policy_operator()(td.clone())
        >>> print(td_clone)  # no value
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                loc: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                sample_log_prob: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                scale: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> td_clone = td_module.get_value_operator()(td.clone())
        >>> print(td_clone)  # no action
        TensorDict(
            fields={
                observation: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                state_value: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        policy_operator: TensorDictModule,
        value_operator: TensorDictModule,
    ):
        super().__init__(
            policy_operator,
            value_operator,
        )

    def get_policy_operator(self) -> SafeSequential:
        """Returns a standalone policy operator that maps an observation to an action."""
        return self.module[0]

    def get_value_operator(self) -> SafeSequential:
        """Returns a standalone value network operator that maps an observation to a value estimate."""
        return self.module[1]

    get_policy_head = get_policy_operator
    get_value_head = get_value_operator


class DecisionTransformerInferenceWrapper(TensorDictModuleWrapper):
    """Inference Action Wrapper for the Decision Transformer.

    A wrapper specifically designed for the Decision Transformer, which will mask the
    input tensordict sequences to the inferece context.
    The output will be a TensorDict with the same keys as the input, but with only the last
    action of the predicted action sequence and the last return to go.

    This module creates returns a modified copy of the tensordict, ie. it does
    **not** modify the tensordict in-place.

    .. note:: If the action, observation or reward-to-go key is not standard,
        the method :meth:`~.set_tensor_keys` should be used, e.g.

            >>> dt_inference_wrapper.set_tensor_keys(action="foo", observation="bar", return_to_go="baz")

    The in_keys are the observation, action and return-to-go keys. The out-keys
    match the in-keys, with the addition of any other out-key from the policy
    (eg., parameters of the distribution or hidden values).

    Args:
        policy (TensorDictModule): The policy module that takes in
            observations and produces an action value

    Keyword Args:
        inference_context (int): The number of previous actions that will not be masked in the context.
            For example for an observation input of shape [batch_size, context, obs_dim] with context=20 and inference_context=5, the first 15 entries
            of the context will be masked. Defaults to 5.
        spec (Optional[TensorSpec]): The spec of the input TensorDict. If None, it will be inferred from the policy module.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules import (
        ...      ProbabilisticActor,
        ...      TanhDelta,
        ...      DTActor,
        ...      DecisionTransformerInferenceWrapper,
        ...  )
        >>> dtactor = DTActor(state_dim=4, action_dim=2,
        ...             transformer_config=DTActor.default_config()
        ... )
        >>> actor_module = TensorDictModule(
        ...         dtactor,
        ...         in_keys=["observation", "action", "return_to_go"],
        ...         out_keys=["param"])
        >>> dist_class = TanhDelta
        >>> dist_kwargs = {
        ...     "min": -1.0,
        ...     "max": 1.0,
        ... }
        >>> actor = ProbabilisticActor(
        ...     in_keys=["param"],
        ...     out_keys=["action"],
        ...     module=actor_module,
        ...     distribution_class=dist_class,
        ...     distribution_kwargs=dist_kwargs)
        >>> inference_actor = DecisionTransformerInferenceWrapper(actor)
        >>> sequence_length = 20
        >>> td = TensorDict({"observation": torch.randn(1, sequence_length, 4),
        ...                 "action": torch.randn(1, sequence_length, 2),
        ...                 "return_to_go": torch.randn(1, sequence_length, 1)}, [1,])
        >>> result = inference_actor(td)
        >>> print(result)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([1, 2]), device=cpu, dtype=torch.float32, is_shared=False),
                observation: Tensor(shape=torch.Size([1, 20, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                param: Tensor(shape=torch.Size([1, 20, 2]), device=cpu, dtype=torch.float32, is_shared=False),
                return_to_go: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([1]),
            device=None,
            is_shared=False)
    """

    def __init__(
        self,
        policy: TensorDictModule,
        *,
        inference_context: int = 5,
        spec: Optional[TensorSpec] = None,
    ):
        super().__init__(policy)
        self.observation_key = "observation"
        self.action_key = "action"
        self.out_action_key = "action"
        self.return_to_go_key = "return_to_go"
        self.inference_context = inference_context
        if spec is not None:
            if not isinstance(spec, CompositeSpec) and len(self.out_keys) >= 1:
                spec = CompositeSpec({self.action_key: spec}, shape=spec.shape[:-1])
            self._spec = spec
        elif hasattr(self.td_module, "_spec"):
            self._spec = self.td_module._spec.clone()
            if self.action_key not in self._spec.keys():
                self._spec[self.action_key] = None
        elif hasattr(self.td_module, "spec"):
            self._spec = self.td_module.spec.clone()
            if self.action_key not in self._spec.keys():
                self._spec[self.action_key] = None
        else:
            self._spec = CompositeSpec({key: None for key in policy.out_keys})
        self.checked = False

    @property
    def in_keys(self):
        return [self.observation_key, self.action_key, self.return_to_go_key]

    @property
    def out_keys(self):
        return sorted(
            set(self.td_module.out_keys).union(
                {self.observation_key, self.action_key, self.return_to_go_key}
            ),
            key=str,
        )

    def set_tensor_keys(self, **kwargs):
        """Sets the input keys of the module.

        Keyword Args:
            observation (NestedKey, optional): The observation key.
            action (NestedKey, optional): The action key (input to the network).
            return_to_go (NestedKey, optional): The return_to_go key.
            out_action (NestedKey, optional): The action key (output of the network).

        """
        observation_key = unravel_key(kwargs.pop("observation", self.observation_key))
        action_key = unravel_key(kwargs.pop("action", self.action_key))
        out_action_key = unravel_key(kwargs.pop("out_action", self.out_action_key))
        return_to_go_key = unravel_key(
            kwargs.pop("return_to_go", self.return_to_go_key)
        )
        if kwargs:
            raise TypeError(
                f"Got unknown input(s) {kwargs.keys()}. Accepted keys are 'action', 'return_to_go' and 'observation'."
            )
        self.observation_key = observation_key
        self.action_key = action_key
        self.return_to_go_key = return_to_go_key
        if out_action_key not in self.td_module.out_keys:
            raise ValueError(
                f"The value of out_action_key ({out_action_key}) must be "
                f"within the actor output keys ({self.td_module.out_keys})."
            )
        self.out_action_key = out_action_key

    def step(self, frames: int = 1) -> None:
        pass

    @staticmethod
    def _check_tensor_dims(reward, obs, action):
        if not (reward.shape[:-1] == obs.shape[:-1] == action.shape[:-1]):
            raise ValueError(
                "Mismatched tensor dimensions. This is not supported yet, file an issue on torchrl"
            )

    def mask_context(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Mask the context of the input sequences."""
        observation = tensordict.get(self.observation_key).clone()
        action = tensordict.get(self.action_key).clone()
        return_to_go = tensordict.get(self.return_to_go_key).clone()
        self._check_tensor_dims(return_to_go, observation, action)

        observation[..., : -self.inference_context, :] = 0
        action[
            ..., : -(self.inference_context - 1), :
        ] = 0  # as we add zeros to the end of the action
        action = torch.cat(
            [
                action[..., 1:, :],
                torch.zeros(
                    *action.shape[:-2], 1, action.shape[-1], device=action.device
                ),
            ],
            dim=-2,
        )
        return_to_go[..., : -self.inference_context, :] = 0

        tensordict.set(self.observation_key, observation)
        tensordict.set(self.action_key, action)
        tensordict.set(self.return_to_go_key, return_to_go)
        return tensordict

    def check_keys(self):
        # an exception will be raised if the action key mismatch
        self.set_tensor_keys()
        self.checked = True

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not self.checked:
            self.check_keys()
        """Forward pass of the inference wrapper."""
        tensordict = tensordict.clone(False)
        obs = tensordict.get(self.observation_key)
        # Mask the context of the input sequences
        tensordict = self.mask_context(tensordict)
        # forward pass
        tensordict = self.td_module.forward(tensordict)
        # get last action prediction
        out_action = tensordict.get(self.out_action_key)
        if tensordict.ndim == out_action.ndim - 1:
            # then time dimension is in the TD's dimensions, and we must get rid of it
            tensordict.batch_size = tensordict.batch_size[:-1]
        out_action = out_action[..., -1, :]
        tensordict.set(self.out_action_key, out_action)

        out_rtg = tensordict.get(self.return_to_go_key)
        out_rtg = out_rtg[..., -1, :]
        tensordict.set(self.return_to_go_key, out_rtg)

        # set unmasked observation
        tensordict.set(self.observation_key, obs)
        return tensordict


class TanhModule(TensorDictModuleBase):
    """A Tanh module for deterministic policies with bounded action space.

    This transform is to be used as a TensorDictModule layer to map a network
    output to a bounded space.

    Args:
        in_keys (list of str or tuples of str): the input keys of the module.
        out_keys (list of str or tuples of str, optional): the output keys of the module.
            If none is provided, the same keys as in_keys are assumed.

    Keyword Args:
        spec (TensorSpec, optional): if provided, the spec of the output.
            If a CompositeSpec is provided, its key(s) must match the key(s)
            in out_keys. Otherwise, the key(s) of out_keys are assumed and the
            same spec is used for all outputs.
        low (float, np.ndarray or torch.Tensor): the lower bound of the space.
            If none is provided and no spec is provided, -1 is assumed. If a
            spec is provided, the minimum value of the spec will be retrieved.
        high (float, np.ndarray or torch.Tensor): the higher bound of the space.
            If none is provided and no spec is provided, 1 is assumed. If a
            spec is provided, the maximum value of the spec will be retrieved.
        clamp (bool, optional): if ``True``, the outputs will be clamped to be
            within the boundaries but at a minimum resolution from them.
            Defaults to ``False``.

    Examples:
        >>> from tensordict import TensorDict
        >>> # simplest use case: -1 - 1 boundaries
        >>> torch.manual_seed(0)
        >>> in_keys = ["action"]
        >>> mod = TanhModule(
        ...     in_keys=in_keys,
        ... )
        >>> data = TensorDict({"action": torch.randn(5) * 10}, [])
        >>> data = mod(data)
        >>> data['action']
        tensor([ 1.0000, -0.9944, -1.0000,  1.0000, -1.0000])
        >>> # low and high can be customized
        >>> low = -2
        >>> high = 1
        >>> mod = TanhModule(
        ...     in_keys=in_keys,
        ...     low=low,
        ...     high=high,
        ... )
        >>> data = TensorDict({"action": torch.randn(5) * 10}, [])
        >>> data = mod(data)
        >>> data['action']
        tensor([-2.0000,  0.9991,  1.0000, -2.0000, -1.9991])
        >>> # A spec can be provided
        >>> from torchrl.data import BoundedTensorSpec
        >>> spec = BoundedTensorSpec(low, high, shape=())
        >>> mod = TanhModule(
        ...     in_keys=in_keys,
        ...     low=low,
        ...     high=high,
        ...     spec=spec,
        ...     clamp=False,
        ... )
        >>> # One can also work with multiple keys
        >>> in_keys = ['a', 'b']
        >>> spec = CompositeSpec(
        ...     a=BoundedTensorSpec(-3, 0, shape=()),
        ...     b=BoundedTensorSpec(0, 3, shape=()))
        >>> mod = TanhModule(
        ...     in_keys=in_keys,
        ...     spec=spec,
        ... )
        >>> data = TensorDict(
        ...     {'a': torch.randn(10), 'b': torch.randn(10)}, batch_size=[])
        >>> data = mod(data)
        >>> data['a']
        tensor([-2.3020, -1.2299, -2.5418, -0.2989, -2.6849, -1.3169, -2.2690, -0.9649,
                -2.5686, -2.8602])
        >>> data['b']
        tensor([2.0315, 2.8455, 2.6027, 2.4746, 1.7843, 2.7782, 0.2111, 0.5115, 1.4687,
                0.5760])
    """

    def __init__(
        self,
        in_keys,
        out_keys=None,
        *,
        spec=None,
        low=None,
        high=None,
        clamp: bool = False,
    ):
        super(TanhModule, self).__init__()
        self.in_keys = in_keys
        if out_keys is None:
            out_keys = in_keys
        if len(in_keys) != len(out_keys):
            raise ValueError(
                "in_keys and out_keys should have the same length, "
                f"got in_keys={in_keys} and out_keys={out_keys}"
            )
        self.out_keys = out_keys
        # action_spec can be a composite spec or not
        if isinstance(spec, CompositeSpec):
            for out_key in self.out_keys:
                if out_key not in spec.keys(True, True):
                    spec[out_key] = None
        else:
            # if one spec is present, we assume it is the same for all keys
            spec = CompositeSpec(
                {out_key: spec for out_key in out_keys},
            )

        leaf_specs = [spec[out_key] for out_key in self.out_keys]
        self.spec = spec
        self.non_trivial = {}
        for out_key, leaf_spec in zip(out_keys, leaf_specs):
            _low, _high = self._make_low_high(low, high, leaf_spec)
            key = out_key if isinstance(out_key, str) else "_".join(out_key)
            self.register_buffer(f"{key}_low", _low)
            self.register_buffer(f"{key}_high", _high)
            self.non_trivial[out_key] = (_high != 1).any() or (_low != -1).any()
            if (_high < _low).any():
                raise ValueError(f"Got high < low in {type(self)}.")
        self.clamp = clamp

    def _make_low_high(self, low, high, leaf_spec):
        if low is None and leaf_spec is None:
            low = -torch.ones(())
        elif low is None:
            low = leaf_spec.space.low
        elif leaf_spec is not None:
            if (low != leaf_spec.space.low).any():
                raise ValueError(
                    f"The minimum value ({low}) provided to {type(self)} does not match the action spec one ({leaf_spec.space.low})."
                )
        if not isinstance(low, torch.Tensor):
            low = torch.tensor(low)
        if high is None and leaf_spec is None:
            high = torch.ones(())
        elif high is None:
            high = leaf_spec.space.high
        elif leaf_spec is not None:
            if (high != leaf_spec.space.high).any():
                raise ValueError(
                    f"The maximum value ({high}) provided to {type(self)} does not match the action spec one ({leaf_spec.space.high})."
                )
        if not isinstance(high, torch.Tensor):
            high = torch.tensor(high)
        return low, high

    @dispatch
    def forward(self, tensordict):
        inputs = [tensordict.get(key) for key in self.in_keys]
        # map
        for out_key, feature in zip(self.out_keys, inputs):
            key = out_key if isinstance(out_key, str) else "_".join(out_key)
            low_key = f"{key}_low"
            high_key = f"{key}_high"
            low = getattr(self, low_key)
            high = getattr(self, high_key)
            feature = feature.tanh()
            if self.clamp:
                eps = torch.finfo(feature.dtype).resolution
                feature = feature.clamp(-1 + eps, 1 - eps)
            if self.non_trivial:
                feature = low + (high - low) * (feature + 1) / 2
            tensordict.set(out_key, feature)
        return tensordict


class LMHeadActorValueOperator(ActorValueOperator):
    """Builds an Actor-Value operator from an huggingface-like *LMHeadModel.

    This method:
        - takes as input an huggingface-like *LMHeadModel
        - extracts the final linear layer uses it as a base layer of the actor_head and
            adds the sampling layer
        - uses the common transformer as common model
        - adds a linear critic

    Args:
        base_model (nn.Module): a torch model composed by a `.transformer` model and `.lm_head` linear layer

      .. note:: For more details regarding the class construction, please refer to :class:`~.ActorValueOperator`.
    """

    def __init__(self, base_model):
        actor_head = base_model.lm_head
        value_head = nn.Linear(actor_head.in_features, 1, bias=False)
        common = TensorDictSequential(
            TensorDictModule(
                base_model.transformer,
                in_keys={"input_ids": "input_ids", "attention_mask": "attention_mask"},
                out_keys=["x"],
            ),
            TensorDictModule(lambda x: x[:, -1, :], in_keys=["x"], out_keys=["x"]),
        )
        actor_head = TensorDictModule(actor_head, in_keys=["x"], out_keys=["logits"])
        actor_head = SafeProbabilisticTensorDictSequential(
            actor_head,
            SafeProbabilisticModule(
                in_keys=["logits"],
                out_keys=["action"],
                distribution_class=Categorical,
                return_log_prob=True,
            ),
        )
        value_head = TensorDictModule(
            value_head, in_keys=["x"], out_keys=["state_value"]
        )

        super().__init__(common, actor_head, value_head)
