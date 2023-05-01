# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import Union

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.data.tensor_specs import TensorSpec

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
    """The DQN Loss class.

    Args:
        value_network (QValueActor or nn.Module): a Q value operator.

    Keyword Args:
        loss_function (str): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".
        priority_key (str, optional): the key at which priority is assumed to
            be stored within TensorDicts added to this ReplayBuffer.
            This is to be used when the sampler is of type
            :class:`~torchrl.data.PrioritizedSampler`.
            Defaults to ``"td_error"``.
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

    """

    default_value_estimator = ValueEstimators.TD0

    def __init__(
        self,
        value_network: Union[QValueActor, nn.Module],
        mixer_network: Union[TensorDictModule, nn.Module],
        *,
        loss_function: str = "l2",
        priority_key: str = "td_error",
        delay_value: bool = False,
        gamma: float = None,
        action_space: Union[str, TensorSpec] = None,
    ) -> None:

        super().__init__()
        self.delay_value = delay_value
        value_network = ensure_tensordict_compatible(
            module=value_network,
            wrapper_type=QValueActor,
            action_space=action_space,
        )
        mixer_network = ensure_tensordict_compatible(
            module=mixer_network,
            out_keys=["chosen_action_value"],
        )
        global_value_network = TensorDictSequential(value_network, mixer_network)

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
        self.convert_to_functional(
            global_value_network,
            "global_value_network",
            create_target_params=self.delay_value,
        )

        self.loss_function = loss_function
        self.priority_key = priority_key
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
                **hp,
                value_network=self.global_value_network,
                advantage_key="advantage",
                value_target_key="value_target",
                value_key="chosen_action_value",
            )
        elif value_type is ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(
                **hp,
                value_network=self.global_value_network,
                advantage_key="advantage",
                value_target_key="value_target",
                value_key="chosen_action_value",
            )
        elif value_type is ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type is ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                **hp,
                value_network=self.global_value_network,
                advantage_key="advantage",
                value_target_key="value_target",
                value_key="chosen_action_value",
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

    def forward(self, input_tensordict: TensorDictBase) -> TensorDict:
        """Computes the DQN loss given a tensordict sampled from the replay buffer.

        This function will also write a "td_error" key that can be used by prioritized replay buffers to assign
            a priority to items in the tensordict.

        Args:
            input_tensordict (TensorDictBase): a tensordict with keys ["action"] and the in_keys of
                the value network (observations, "done", "reward" in a "next" tensordict).

        Returns:
            a tensor containing the DQN loss.

        """
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

        action = tensordict.get("action")
        pred_val = td_copy.get("action_value")

        if self.action_space == "categorical":
            if action.shape != pred_val.shape:
                action = action.unsqueeze(-1)
            pred_val_index = pred_val.gather(-1, action)
        else:
            action = action.to(torch.float)
            pred_val_index = (pred_val * action).sum(-1, keepdim=True)

        td_copy.set("chosen_action_value", pred_val_index)
        self.mixer_network(td_copy, params=self.mixer_network_params)
        pred_val_index = td_copy.get("chosen_action_value")

        target_value = self.value_estimator.value_estimate(
            tensordict.clone(False),
            target_params=self.target_global_value_network_params,
        )

        priority_tensor = (pred_val_index - target_value).pow(2)
        priority_tensor = priority_tensor.detach().unsqueeze(-1)
        if input_tensordict.device is not None:
            priority_tensor = priority_tensor.to(input_tensordict.device)

        input_tensordict.set(
            self.priority_key,
            priority_tensor,
            inplace=True,
        )
        loss = distance_loss(pred_val_index, target_value, self.loss_function)
        return TensorDict({"loss": loss.mean()}, [])
