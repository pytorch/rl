# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass

import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import dispatch, TensorDictModule
from tensordict.utils import NestedKey
from torch import nn

from torchrl.data.tensor_specs import TensorSpec
from torchrl.data.utils import _find_action_space
from torchrl.modules import SafeSequential
from torchrl.modules.tensordict_module.actors import QValueActor
from torchrl.modules.tensordict_module.common import ensure_tensordict_compatible
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _cache_values,
    _GAMMA_LMBDA_DEPREC_ERROR,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)
from torchrl.objectives.value import TDLambdaEstimator
from torchrl.objectives.value.advantages import TD0Estimator, TD1Estimator


class QMixerLoss(LossModule):
    """The QMixer loss class.

    Mixes local agent q values into a global q value according to a mixing network and then
    uses DQN updates on the global value.
    This loss is for multi-agent applications.
    Therefore, it expects the 'local_value', 'action_value' and 'action' keys
    to have an agent dimension (this is visible in the default AcceptedKeys).
    This dimension will be mixed by the mixer which will compute a 'global_value' key, used for a DQN objective.
    The premade mixers of type :class:`torchrl.modules.models.multiagent.Mixer` will expect the multi-agent
    dimension to be the penultimate one.

    Args:
        local_value_network (QValueActor or nn.Module): a local Q value operator.
        mixer_network (TensorDictModule or nn.Module): a mixer network mapping the agents' local Q values
            and an optional state to the global Q value. It is suggested to provide a TensorDictModule
            wrapping a mixer from :class:`torchrl.modules.models.multiagent.Mixer`.

    Keyword Args:
        loss_function (str, optional): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".
            Defaults to "l2".
        delay_value (bool, optional): whether to duplicate the value network
            into a new target value network to
            create a double DQN. Default is ``False``.
        action_space (str or TensorSpec, optional): Action space. Must be one of
            ``"one-hot"``, ``"mult_one_hot"``, ``"binary"`` or ``"categorical"``,
            or an instance of the corresponding specs (:class:`torchrl.data.OneHot`,
            :class:`torchrl.data.MultiOneHot`,
            :class:`torchrl.data.Binary` or :class:`torchrl.data.Categorical`).
            If not provided, an attempt to retrieve it from the value network
            will be made.
        priority_key (NestedKey, optional): [Deprecated, use .set_keys(priority_key=priority_key) instead]
            The key at which priority is assumed to be stored within TensorDicts added
            to this ReplayBuffer.  This is to be used when the sampler is of type
            :class:`~torchrl.data.PrioritizedSampler`.  Defaults to ``"td_error"``.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules import QValueModule, SafeSequential
        >>> from torchrl.modules.models.multiagent import QMixer
        >>> from torchrl.objectives.multiagent import QMixerLoss
        >>> n_agents = 4
        >>> module = TensorDictModule(
        ...     nn.Linear(10,3), in_keys=[("agents", "observation")], out_keys=[("agents", "action_value")]
        ... )
        >>> value_module = QValueModule(
        ...     action_value_key=("agents", "action_value"),
        ...     out_keys=[
        ...         ("agents", "action"),
        ...         ("agents", "action_value"),
        ...         ("agents", "chosen_action_value"),
        ...     ],
        ...     action_space="categorical",
        ... )
        >>> qnet = SafeSequential(module, value_module)
        >>> qmixer = TensorDictModule(
        ...    module=QMixer(
        ...        state_shape=(64, 64, 3),
        ...        mixing_embed_dim=32,
        ...        n_agents=n_agents,
        ...        device="cpu",
        ...    ),
        ...    in_keys=[("agents", "chosen_action_value"), "state"],
        ...    out_keys=["chosen_action_value"],
        ... )
        >>> loss = QMixerLoss(qnet, qmixer, action_space="categorical")
        >>> td = TensorDict(
        ...    {
        ...        "agents": TensorDict(
        ...            {"observation": torch.zeros(32, n_agents, 10)}, [32, n_agents]
        ...        ),
        ...        "state": torch.zeros(32, 64, 64, 3),
        ...        "next": TensorDict(
        ...           {
        ...                "agents": TensorDict(
        ...                     {"observation": torch.zeros(32, n_agents, 10)}, [32, n_agents]
        ...                ),
        ...                "state": torch.zeros(32, 64, 64, 3),
        ...                "reward": torch.zeros(32, 1),
        ...                "done": torch.zeros(32, 1, dtype=torch.bool),
        ...                "terminated": torch.zeros(32, 1, dtype=torch.bool),
        ...            },
        ...            [32],
        ...        ),
        ...    },
        ...    [32],
        ... )
        >>> loss(qnet(td))
        TensorDict(
            fields={
                loss: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            advantage (NestedKey): The input tensordict key where the advantage is expected.
                Will be used for the underlying value estimator. Defaults to ``"advantage"``.
            value_target (NestedKey): The input tensordict key where the target state value is expected.
                Will be used for the underlying value estimator Defaults to ``"value_target"``.
            local_value (NestedKey): The input tensordict key where the local chosen action value is expected.
                Will be used for the underlying value estimator. Defaults to ``("agents", "chosen_action_value")``.
            global_value (NestedKey): The input tensordict key where the global chosen action value is expected.
                Will be used for the underlying value estimator. Defaults to ``"chosen_action_value"``.
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``("agents", "action")``.
            action_value (NestedKey): The input tensordict key where the action value is expected.
                Defaults to ``("agents", "action_value")``.
            priority (NestedKey): The input tensordict key where the target priority is written to.
                Defaults to ``"td_error"``.
            reward (NestedKey): The input tensordict key where the reward is expected.
                Will be used for the underlying value estimator. Defaults to ``"reward"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value estimator.
                Defaults to ``"done"``.
            terminated (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is terminated. Will be used for the underlying value estimator.
                Defaults to ``"terminated"``.
        """

        advantage: NestedKey = "advantage"
        value_target: NestedKey = "value_target"
        local_value: NestedKey = ("agents", "chosen_action_value")
        global_value: NestedKey = "chosen_action_value"
        action_value: NestedKey = ("agents", "action_value")
        action: NestedKey = ("agents", "action")
        priority: NestedKey = "td_error"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

    tensor_keys: _AcceptedKeys
    default_keys = _AcceptedKeys
    default_value_estimator = ValueEstimators.TD0
    out_keys = ["loss"]

    local_value_network: TensorDictModule
    local_value_network_params: TensorDictParams
    target_local_value_network_params: TensorDictParams
    mixer_network: TensorDictModule
    mixer_network_params: TensorDictParams
    target_mixer_network_params: TensorDictParams

    def __init__(
        self,
        local_value_network: QValueActor | nn.Module,
        mixer_network: TensorDictModule | nn.Module,
        *,
        loss_function: str | None = "l2",
        delay_value: bool = True,
        gamma: float | None = None,
        action_space: str | TensorSpec = None,
        priority_key: str = None,
    ) -> None:
        super().__init__()
        self._in_keys = None
        self._set_deprecated_ctor_keys(priority=priority_key)
        self.delay_value = delay_value
        local_value_network = ensure_tensordict_compatible(
            module=local_value_network,
            wrapper_type=QValueActor,
            action_space=action_space,
        )
        if not isinstance(mixer_network, TensorDictModule):
            # If it is not a TensorDictModule we make it one with default keys
            mixer_network = ensure_tensordict_compatible(
                module=mixer_network,
                in_keys=[self.tensor_keys.local_value],
                out_keys=[self.tensor_keys.global_value],
            )

        global_value_network = SafeSequential(local_value_network, mixer_network)
        params = TensorDict.from_module(global_value_network)
        with params.apply(
            self._make_meta_params, device=torch.device("meta"), filter_empty=False
        ).to_module(global_value_network):
            self.__dict__["global_value_network"] = deepcopy(global_value_network)

        self.convert_to_functional(
            local_value_network,
            "local_value_network",
            create_target_params=self.delay_value,
        )
        self.convert_to_functional(
            mixer_network,
            "mixer_network",
            create_target_params=self.delay_value,
        )
        self.global_value_network.module[0] = self.local_value_network
        self.global_value_network.module[1] = self.mixer_network

        self.global_value_network_in_keys = global_value_network.in_keys

        self.loss_function = loss_function
        if action_space is None:
            # infer from value net
            try:
                action_space = local_value_network.spec
            except AttributeError:
                # let's try with action_space then
                try:
                    action_space = local_value_network.action_space
                except AttributeError:
                    raise ValueError(self.ACTION_SPEC_ERROR)
        if action_space is None:
            warnings.warn(
                "action_space was not specified. QMixerLoss will default to 'one-hot'."
                "This behavior will be deprecated soon and a space will have to be passed."
                "Check the QMixerLoss documentation to see how to pass the action space. "
            )
            action_space = "one-hot"

        self.action_space = _find_action_space(action_space)

        if gamma is not None:
            raise TypeError(_GAMMA_LMBDA_DEPREC_ERROR)

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                advantage=self.tensor_keys.advantage,
                value_target=self.tensor_keys.value_target,
                value=self.tensor_keys.global_value,
                reward=self.tensor_keys.reward,
                done=self.tensor_keys.done,
                terminated=self.tensor_keys.terminated,
            )
        self._set_in_keys()

    def _set_in_keys(self):
        keys = [
            self.tensor_keys.action,
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.done),
            ("next", self.tensor_keys.terminated),
            *self.global_value_network.in_keys,
            *[("next", key) for key in self.global_value_network.in_keys],
        ]
        self._in_keys = list(set(keys))

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        hp = dict(default_value_kwargs(value_type))
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        hp.update(hyperparams)
        if value_type is ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(
                **hp, value_network=self.global_value_network
            )
        elif value_type is ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(
                **hp, value_network=self.global_value_network
            )
        elif value_type is ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type is ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                **hp, value_network=self.global_value_network
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        tensor_keys = {
            "advantage": self.tensor_keys.advantage,
            "value_target": self.tensor_keys.value_target,
            "value": self.tensor_keys.global_value,
            "reward": self.tensor_keys.reward,
            "done": self.tensor_keys.done,
            "terminated": self.tensor_keys.terminated,
        }
        self._value_estimator.set_keys(**tensor_keys)

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        td_copy = tensordict.clone(False)
        with self.local_value_network_params.to_module(self.local_value_network):
            self.local_value_network(
                td_copy,
            )

        action = tensordict.get(self.tensor_keys.action)
        pred_val = td_copy.get(
            self.tensor_keys.action_value
        )  # [*B, n_agents, n_actions]

        if self.action_space == "categorical":
            if action.shape != pred_val.shape:
                # unsqueeze the action if it lacks on trailing singleton dim
                action = action.unsqueeze(-1)
            pred_val_index = torch.gather(pred_val, -1, index=action)
        else:
            action = action.to(torch.float)
            pred_val_index = (pred_val * action).sum(-1, keepdim=True)

        td_copy.set(self.tensor_keys.local_value, pred_val_index)  # [*B, n_agents, 1]
        with self.mixer_network_params.to_module(self.mixer_network):
            self.mixer_network(td_copy)
        pred_val_index = td_copy[self.tensor_keys.global_value].squeeze(-1)
        # [*B] this is global and shared among the agents as will be the target

        target_value = self.value_estimator.value_estimate(
            td_copy,
            target_params=self._cached_target_params,
        ).squeeze(-1)

        with torch.no_grad():
            priority_tensor = (pred_val_index - target_value).pow(2)
            priority_tensor = priority_tensor.unsqueeze(-1)
        if tensordict.device is not None:
            priority_tensor = priority_tensor.to(tensordict.device)

        tensordict.set(
            self.tensor_keys.priority,
            priority_tensor,
            inplace=True,
        )
        loss = distance_loss(pred_val_index, target_value, self.loss_function)
        return TensorDict({"loss": loss.mean()}, [])

    @property
    @_cache_values
    def _cached_target_params(self):
        target_params = TensorDict(
            {
                "module": {
                    "0": self.target_local_value_network_params,
                    "1": self.target_mixer_network_params,
                }
            },
            batch_size=self.target_local_value_network_params.batch_size,
            device=self.target_local_value_network_params.device,
        )
        return target_params
