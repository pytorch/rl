# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Union

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import make_functional, repopulate_module, TensorDictModule
from tensordict.utils import NestedKey
from torch import nn

from torchrl.data.tensor_specs import TensorSpec
from torchrl.modules import SafeSequential
from torchrl.modules.tensordict_module.actors import QValueActor
from torchrl.modules.tensordict_module.common import ensure_tensordict_compatible
from torchrl.modules.utils.utils import _find_action_space
from torchrl.objectives import LossModule, ValueEstimators
from torchrl.objectives.utils import (
    _GAMMA_LMBDA_DEPREC_WARNING,
    default_value_kwargs,
    distance_loss,
)
from torchrl.objectives.value import TD0Estimator, TD1Estimator, TDLambdaEstimator


class QMixLoss(LossModule):
    """The QMIX Loss class.

    Args:
        value_network (QValueActor or nn.Module): a Q value operator.
        mixer_network (TensorDictModule or nn.Module): a mixer network mapping the agents' local Q values
         (contained in "chosen_action_value") and an optional state to the global Q value (rewritten in "chosen_action_value")

    Keyword Args:
        loss_function (str): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".
        delay_value (bool, optional): whether to duplicate the value network
            into a new target value network to
            create a double DQN. Default is ``False``.
        action_space (str or TensorSpec, optional): Action space. Must be one of
            ``"one-hot"``, ``"mult_one_hot"``, ``"binary"`` or ``"categorical"``,
            or an instance of the corresponding specs (:class:`torchrl.data.OneHotDiscreteTensorSpec`,
            :class:`torchrl.data.MultiOneHotDiscreteTensorSpec`,
            :class:`torchrl.data.BinaryDiscreteTensorSpec` or :class:`torchrl.data.DiscreteTensorSpec`).
            If not provided, an attempt to retrieve it from the value network
            will be made.
        priority_key (str, optional): [Deprecated, use .set_keys(priority_key=priority_key) instead]
            The key at which priority is assumed to be stored within TensorDicts added
            to this ReplayBuffer.  This is to be used when the sampler is of type
            :class:`~torchrl.data.PrioritizedSampler`.  Defaults to ``"td_error"``.

    Examples:
        >>> from torchrl.modules import MLP
        >>> from torchrl.data import OneHotDiscreteTensorSpec
        >>> n_obs, n_act = 4, 3
        >>> value_net = MLP(in_features=n_obs, out_features=n_act)
        >>> spec = OneHotDiscreteTensorSpec(n_act)
        >>> actor = QValueActor(value_net, in_keys=["observation"], action_space=spec)
        >>> loss = DQNLoss(actor, action_space=spec)
        >>> batch = [10,]
        >>> data = TensorDict({
        ...     "observation": torch.randn(*batch, n_obs),
        ...     "action": spec.rand(batch),
        ...     ("next", "observation"): torch.randn(*batch, n_obs),
        ...     ("next", "done"): torch.zeros(*batch, 1, dtype=torch.bool),
        ...     ("next", "reward"): torch.randn(*batch, 1)
        ... }, batch)
        >>> loss(data)
        TensorDict(
            fields={
                loss: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    This class is compatible with non-tensordict based modules too and can be
    used without recurring to any tensordict-related primitive. In this case,
    the expected keyword arguments are:
    ``["observation", "next_observation", "action", "next_reward", "next_done"]``,
    and a single loss value is returned.

    Examples:
        >>> from torchrl.objectives import DQNLoss
        >>> from torchrl.data import OneHotDiscreteTensorSpec
        >>> from torch import nn
        >>> import torch
        >>> n_obs = 3
        >>> n_action = 4
        >>> action_spec = OneHotDiscreteTensorSpec(n_action)
        >>> value_network = nn.Linear(n_obs, n_action) # a simple value model
        >>> dqn_loss = DQNLoss(value_network, action_space=action_spec)
        >>> # define data
        >>> observation = torch.randn(n_obs)
        >>> next_observation = torch.randn(n_obs)
        >>> action = action_spec.rand()
        >>> next_reward = torch.randn(1)
        >>> next_done = torch.zeros(1, dtype=torch.bool)
        >>> loss_val = dqn_loss(
        ...     observation=observation,
        ...     next_observation=next_observation,
        ...     next_reward=next_reward,
        ...     next_done=next_done,
        ...     action=action)

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
            value (NestedKey): The input tensordict key where the state value is expected.
                Will be used for the underlying value estimator. Defaults to ``"state_value"``.
            state_action_value (NestedKey): The input tensordict key where the state action value is expected.
                Defaults to ``"state_action_value"``.
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"action"``.
            priority (NestedKey): The input tensordict key where the target priority is written to.
                Defaults to ``"td_error"``.
            reward (NestedKey): The input tensordict key where the reward is expected.
                Will be used for the underlying value estimator. Defaults to ``"reward"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value estimator.
                Defaults to ``"done"``.
        """

        advantage: NestedKey = "advantage"
        value_target: NestedKey = "value_target"
        value: NestedKey = "chosen_action_value"
        action_value: NestedKey = "action_value"
        action: NestedKey = "action"
        priority: NestedKey = "td_error"
        reward: NestedKey = "reward"
        done: NestedKey = "done"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.TD0
    out_keys = ["loss"]

    def __init__(
        self,
        value_network: Union[QValueActor, nn.Module],
        mixer_network: Union[TensorDictModule, nn.Module],
        *,
        loss_function: str = "l2",
        delay_value: bool = False,
        gamma: float = None,
        action_space: Union[str, TensorSpec] = None,
        priority_key: str = None,
    ) -> None:

        super().__init__()
        self._in_keys = None
        self._set_deprecated_ctor_keys(priority=priority_key)
        self.delay_value = delay_value
        value_network = ensure_tensordict_compatible(
            module=value_network,
            wrapper_type=QValueActor,
            action_space=action_space,
        )
        mixer_network = ensure_tensordict_compatible(
            module=mixer_network,
        )

        global_value_network = SafeSequential(value_network, mixer_network)
        params = make_functional(global_value_network)
        self.global_value_network = deepcopy(global_value_network)
        repopulate_module(value_network, params["module", "0"])
        repopulate_module(mixer_network, params["module", "1"])

        self.convert_to_functional(
            value_network,
            "value_network",
            create_target_params=self.delay_value,
        )
        self.convert_to_functional(
            mixer_network,
            "mixer_network",
            create_target_params=self.delay_value,
        )
        self.global_value_network.module[0] = self.value_network
        self.global_value_network.module[1] = self.mixer_network

        self.global_value_network_in_keys = global_value_network.in_keys

        self.loss_function = loss_function
        if action_space is None:
            # infer from value net
            try:
                action_space = value_network.spec
            except AttributeError:
                # let's try with action_space then
                try:
                    action_space = value_network.action_space
                except AttributeError:
                    raise ValueError(self.ACTION_SPEC_ERROR)
        if action_space is None:
            warnings.warn(
                "action_space was not specified. QMixLoss will default to 'one-hot'."
                "This behaviour will be deprecated soon and a space will have to be passed."
                "Check the QMixLoss documentation to see how to pass the action space. "
            )
            action_space = "one-hot"
        self.action_space = _find_action_space(action_space)

        if gamma is not None:
            warnings.warn(_GAMMA_LMBDA_DEPREC_WARNING, category=DeprecationWarning)
            self.gamma = gamma

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                advantage=self.tensor_keys.advantage,
                value_target=self.tensor_keys.value_target,
                value=self.tensor_keys.value,
                reward=self.tensor_keys.reward,
                done=self.tensor_keys.done,
            )
        self._set_in_keys()

    def _set_in_keys(self):
        keys = [
            self.tensor_keys.action,
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.done),
            *self.value_network.in_keys,
            *[("next", key) for key in self.value_network.in_keys],
        ]
        self._in_keys = list(set(keys))

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys

    @in_keys.setter
    def in_keys(self, values):
        self._in_keys = values

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
            "value": self.tensor_keys.value,
            "reward": self.tensor_keys.reward,
            "done": self.tensor_keys.done,
        }
        self._value_estimator.set_keys(**tensor_keys)

    def forward(self, input_tensordict: TensorDictBase) -> TensorDict:
        device = self.device if self.device is not None else input_tensordict.device
        tensordict = input_tensordict.to(device)
        if tensordict.device != device:
            raise RuntimeError(
                f"device {device} was expected for "
                f"{tensordict.__class__.__name__} but {tensordict.device} was found"
            )

        for k, t in tensordict.items():
            if t.device != device:
                raise RuntimeError(
                    f"found key value pair {k}-{t.shape} "
                    f"with device {t.device} when {device} was required"
                )

        td_copy = tensordict.clone()
        if td_copy.device != tensordict.device:
            raise RuntimeError(f"{tensordict} and {td_copy} have different devices")
        assert hasattr(self.value_network, "_is_stateless")
        self.value_network(
            td_copy,
            params=self.value_network_params,
        )

        action = tensordict.get(self.tensor_keys.action)
        pred_val = td_copy.get(
            self.tensor_keys.action_value
        )  # [*B, n_agents, n_actions]

        if self.action_space == "categorical":
            if action.shape != pred_val.shape:
                action = action.unsqueeze(-1)
            pred_val_index = pred_val.gather(-1, action)
        else:
            action = action.to(torch.float)
            pred_val_index = (pred_val * action).sum(-1, keepdim=True)

        td_copy.set(self.tensor_keys.value, pred_val_index)  # [*B, n_agents, 1]
        self.mixer_network(td_copy, params=self.mixer_network_params)
        pred_val_index = td_copy.get(self.tensor_keys.value)
        # [*B, 1] this is global and shared among the agents as will be the target

        target_params = TensorDict(
            {
                "module": {
                    "0": self.target_value_network_params,
                    "1": self.target_mixer_network_params,
                }
            },
            batch_size=self.target_value_network_params.batch_size,
            device=self.target_value_network_params.device,
        )

        target_value = self.value_estimator.value_estimate(
            tensordict.clone(False),
            target_params=target_params,
        )  # [*B, 1]

        priority_tensor = (pred_val_index - target_value).pow(2)
        priority_tensor = priority_tensor.detach().unsqueeze(-1)
        if input_tensordict.device is not None:
            priority_tensor = priority_tensor.to(input_tensordict.device)

        input_tensordict.set(
            self.tensor_keys.priority,
            priority_tensor,
            inplace=True,
        )
        loss = distance_loss(pred_val_index, target_value, self.loss_function)
        return TensorDict({"loss": loss.mean()}, [])
