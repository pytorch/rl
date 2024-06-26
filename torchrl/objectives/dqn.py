# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import dispatch, TensorDictModule
from tensordict.utils import NestedKey
from torch import nn
from torchrl.data.tensor_specs import TensorSpec

from torchrl.data.utils import _find_action_space

from torchrl.envs.utils import step_mdp
from torchrl.modules.tensordict_module.actors import (
    DistributionalQValueActor,
    QValueActor,
)
from torchrl.modules.tensordict_module.common import ensure_tensordict_compatible

from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _GAMMA_LMBDA_DEPREC_ERROR,
    _reduce,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)
from torchrl.objectives.value import TDLambdaEstimator
from torchrl.objectives.value.advantages import TD0Estimator, TD1Estimator


class DQNLoss(LossModule):
    """The DQN Loss class.

    Args:
        value_network (QValueActor or nn.Module): a Q value operator.

    Keyword Args:
        loss_function (str, optional): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".
            Defaults to "l2".
        delay_value (bool, optional): whether to duplicate the value network
            into a new target value network to
            create a DQN with a target network. Default is ``False``.
        double_dqn (bool, optional): whether to use Double DQN, as described in
            https://arxiv.org/abs/1509.06461. Defaults to ``False``.
        action_space (str or TensorSpec, optional): Action space. Must be one of
            ``"one-hot"``, ``"mult_one_hot"``, ``"binary"`` or ``"categorical"``,
            or an instance of the corresponding specs (:class:`torchrl.data.OneHotDiscreteTensorSpec`,
            :class:`torchrl.data.MultiOneHotDiscreteTensorSpec`,
            :class:`torchrl.data.BinaryDiscreteTensorSpec` or :class:`torchrl.data.DiscreteTensorSpec`).
            If not provided, an attempt to retrieve it from the value network
            will be made.
        priority_key (NestedKey, optional): [Deprecated, use .set_keys(priority_key=priority_key) instead]
            The key at which priority is assumed to be stored within TensorDicts added
            to this ReplayBuffer.  This is to be used when the sampler is of type
            :class:`~torchrl.data.PrioritizedSampler`.  Defaults to ``"td_error"``.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output, ``"sum"``: the output will be summed. Default: ``"mean"``.

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
        ...     ("next", "terminated"): torch.zeros(*batch, 1, dtype=torch.bool),
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
    ``["observation", "next_observation", "action", "next_reward", "next_done", "next_terminated"]``,
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
        >>> next_terminated = torch.zeros(1, dtype=torch.bool)
        >>> loss_val = dqn_loss(
        ...     observation=observation,
        ...     next_observation=next_observation,
        ...     next_reward=next_reward,
        ...     next_done=next_done,
        ...     next_terminated=next_terminated,
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
            value (NestedKey): The input tensordict key where the chosen action value is expected.
                Will be used for the underlying value estimator. Defaults to ``"chosen_action_value"``.
            action_value (NestedKey): The input tensordict key where the action value is expected.
                Defaults to ``"action_value"``.
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"action"``.
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
        value: NestedKey = "chosen_action_value"
        action_value: NestedKey = "action_value"
        action: NestedKey = "action"
        priority: NestedKey = "td_error"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.TD0
    out_keys = ["loss"]

    value_network: TensorDictModule
    value_network_params: TensorDictParams
    target_value_network_params: TensorDictParams

    def __init__(
        self,
        value_network: Union[QValueActor, nn.Module],
        *,
        loss_function: Optional[str] = "l2",
        delay_value: bool = True,
        double_dqn: bool = False,
        gamma: float = None,
        action_space: Union[str, TensorSpec] = None,
        priority_key: str = None,
        reduction: str = None,
    ) -> None:
        if reduction is None:
            reduction = "mean"
        super().__init__()
        self._in_keys = None
        if double_dqn and not delay_value:
            raise ValueError("double_dqn=True requires delay_value=True.")
        self.double_dqn = double_dqn
        self._set_deprecated_ctor_keys(priority=priority_key)
        self.delay_value = delay_value
        value_network = ensure_tensordict_compatible(
            module=value_network,
            wrapper_type=QValueActor,
            action_space=action_space,
        )

        self.convert_to_functional(
            value_network,
            "value_network",
            create_target_params=self.delay_value,
        )

        self.value_network_in_keys = value_network.in_keys

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
                    raise ValueError(
                        "The action space could not be retrieved from the value_network. "
                        "Make sure it is available to the DQN loss module."
                    )
        if action_space is None:
            warnings.warn(
                "action_space was not specified. DQNLoss will default to 'one-hot'."
                "This behaviour will be deprecated soon and a space will have to be passed."
                "Check the DQNLoss documentation to see how to pass the action space. "
            )
            action_space = "one-hot"
        self.action_space = _find_action_space(action_space)
        self.reduction = reduction
        if gamma is not None:
            raise TypeError(_GAMMA_LMBDA_DEPREC_ERROR)

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                advantage=self.tensor_keys.advantage,
                value_target=self.tensor_keys.value_target,
                value=self.tensor_keys.value,
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
            *self.value_network.in_keys,
            *[("next", key) for key in self.value_network.in_keys],
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
            self._value_estimator = TD1Estimator(**hp, value_network=self.value_network)
        elif value_type is ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(**hp, value_network=self.value_network)
        elif value_type is ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type is ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                **hp, value_network=self.value_network
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        tensor_keys = {
            "advantage": self.tensor_keys.advantage,
            "value_target": self.tensor_keys.value_target,
            "value": self.tensor_keys.value,
            "reward": self.tensor_keys.reward,
            "done": self.tensor_keys.done,
            "terminated": self.tensor_keys.terminated,
        }
        self._value_estimator.set_keys(**tensor_keys)

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        """Computes the DQN loss given a tensordict sampled from the replay buffer.

        This function will also write a "td_error" key that can be used by prioritized replay buffers to assign
            a priority to items in the tensordict.

        Args:
            tensordict (TensorDictBase): a tensordict with keys ["action"] and the in_keys of
                the value network (observations, "done", "terminated", "reward" in a "next" tensordict).

        Returns:
            a tensor containing the DQN loss.

        """
        td_copy = tensordict.clone(False)
        with self.value_network_params.to_module(self.value_network):
            self.value_network(td_copy)

        action = tensordict.get(self.tensor_keys.action)
        pred_val = td_copy.get(self.tensor_keys.action_value)

        if self.action_space == "categorical":
            if action.ndim != pred_val.ndim:
                # unsqueeze the action if it lacks on trailing singleton dim
                action = action.unsqueeze(-1)
            pred_val_index = torch.gather(pred_val, -1, index=action).squeeze(-1)
        else:
            action = action.to(torch.float)
            pred_val_index = (pred_val * action).sum(-1)

        if self.double_dqn:
            step_td = step_mdp(td_copy, keep_other=False)
            step_td_copy = step_td.clone(False)
            # Use online network to compute the action
            with self.value_network_params.data.to_module(self.value_network):
                self.value_network(step_td)
                next_action = step_td.get(self.tensor_keys.action)

            # Use target network to compute the values
            with self.target_value_network_params.to_module(self.value_network):
                self.value_network(step_td_copy)
                next_pred_val = step_td_copy.get(self.tensor_keys.action_value)

            if self.action_space == "categorical":
                if next_action.ndim != next_pred_val.ndim:
                    # unsqueeze the action if it lacks on trailing singleton dim
                    next_action = next_action.unsqueeze(-1)
                next_value = torch.gather(next_pred_val, -1, index=next_action)
            else:
                next_value = (next_pred_val * next_action).sum(-1, keepdim=True)
        else:
            next_value = None
        target_value = self.value_estimator.value_estimate(
            td_copy,
            target_params=self.target_value_network_params,
            next_value=next_value,
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
        loss = _reduce(loss, reduction=self.reduction)
        td_out = TensorDict({"loss": loss}, [])
        return td_out


class DistributionalDQNLoss(LossModule):
    """A distributional DQN loss class.

    Distributional DQN uses a value network that outputs a distribution of
    values over a discrete support of discounted returns (unlike regular DQN
    where the value network outputs a single point prediction of the
    disctounted return).

    For more details regarding Distributional DQN, refer to "A Distributional
    Perspective on Reinforcement Learning",
    https://arxiv.org/pdf/1707.06887.pdf

    Args:
        value_network (DistributionalQValueActor or nn.Module): the distributional Q
            value operator.
        gamma (scalar): a discount factor for return computation.
            .. note::
              Unlike :class:`DQNLoss`, this class does not currently support
              custom value functions. The next value estimation is always
              bootstrapped.
        delay_value (bool): whether to duplicate the value network into a new
            target value network to create double DQN
        priority_key (str, optional): [Deprecated, use .set_keys(priority_key=priority_key) instead]
            The key at which priority is assumed to be stored within TensorDicts added
            to this ReplayBuffer.  This is to be used when the sampler is of type
            :class:`~torchrl.data.PrioritizedSampler`.  Defaults to ``"td_error"``.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output, ``"sum"``: the output will be summed. Default: ``"mean"``.
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values

        Attributes:
            state_action_value (NestedKey): The input tensordict key where the state action value is expected.
                Defaults to ``"state_action_value"``.
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"action"``.
            priority (NestedKey): The input tensordict key where the target priority is written to.
                Defaults to ``"td_error"``.
            reward (NestedKey): The input tensordict key where the reward is expected.
                Defaults to ``"reward"``.
            done (NestedKey): The input tensordict key where the flag if a trajectory is done is expected.
                Defaults to ``"done"``.
            terminated (NestedKey): The input tensordict key where the flag if a trajectory is done is expected.
                Defaults to ``"terminated"``.
            steps_to_next_obs (NestedKey): The input tensordict key where the steps_to_next_obs is exptected.
                Defaults to ``"steps_to_next_obs"``.
        """

        action_value: NestedKey = "action_value"
        action: NestedKey = "action"
        priority: NestedKey = "td_error"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"
        steps_to_next_obs: NestedKey = "steps_to_next_obs"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.TD0

    value_network: TensorDictModule
    value_network_params: TensorDictParams
    target_value_network_params: TensorDictParams

    def __init__(
        self,
        value_network: Union[DistributionalQValueActor, nn.Module],
        *,
        gamma: float,
        delay_value: bool = True,
        priority_key: str = None,
        reduction: str = None,
    ):
        if reduction is None:
            reduction = "mean"
        super().__init__()
        self._set_deprecated_ctor_keys(priority=priority_key)
        self.register_buffer("gamma", torch.tensor(gamma))
        self.delay_value = delay_value

        value_network = ensure_tensordict_compatible(
            module=value_network, wrapper_type=DistributionalQValueActor
        )

        self.convert_to_functional(
            value_network,
            "value_network",
            create_target_params=self.delay_value,
        )
        self.action_space = self.value_network.action_space
        self.reduction = reduction

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    @staticmethod
    def _log_ps_a_default(action, action_log_softmax, batch_size, atoms):
        action_expand = action.unsqueeze(-2).expand_as(action_log_softmax)
        log_ps_a = action_log_softmax.masked_select(action_expand.to(torch.bool))
        log_ps_a = log_ps_a.view(batch_size, atoms)  # log p(s_t, a_t; θonline)
        return log_ps_a

    @staticmethod
    def _log_ps_a_categorical(action, action_log_softmax):
        # Reshaping action of shape `[*batch_sizes, 1]` to `[*batch_sizes, atoms, 1]` for gather.
        if action.shape[-1] != 1:
            action = action.unsqueeze(-1)
        action = action.unsqueeze(-2)
        new_shape = [-1] * len(action.shape)
        new_shape[-2] = action_log_softmax.shape[-2]  # calculating atoms
        action = action.expand(new_shape)
        return torch.gather(action_log_softmax, -1, index=action).squeeze(-1)

    def forward(self, input_tensordict: TensorDictBase) -> TensorDict:
        # from https://github.com/Kaixhin/Rainbow/blob/9ff5567ad1234ae0ed30d8471e8f13ae07119395/agent.py
        tensordict = TensorDict(
            source=input_tensordict, batch_size=input_tensordict.batch_size
        )

        if tensordict.batch_dims != 1:
            raise RuntimeError(
                f"{self.__class__.__name___} expects a 1-dimensional "
                "tensordict as input"
            )
        batch_size = tensordict.batch_size[0]
        support = self.value_network_params["support"]
        atoms = support.numel()
        Vmin = support.min().item()
        Vmax = support.max().item()
        delta_z = (Vmax - Vmin) / (atoms - 1)

        action = tensordict.get(self.tensor_keys.action)
        reward = tensordict.get(("next", self.tensor_keys.reward))
        done = tensordict.get(("next", self.tensor_keys.done))
        terminated = tensordict.get(("next", self.tensor_keys.terminated), default=done)

        steps_to_next_obs = tensordict.get(self.tensor_keys.steps_to_next_obs, 1)
        discount = self.gamma**steps_to_next_obs

        # Calculate current state probabilities (online network noise already
        # sampled)
        td_clone = tensordict.clone()
        with self.value_network_params.to_module(self.value_network):
            self.value_network(
                td_clone,
            )  # Log probabilities log p(s_t, ·; θonline)
        action_log_softmax = td_clone.get(self.tensor_keys.action_value)

        if self.action_space == "categorical":
            log_ps_a = self._log_ps_a_categorical(action, action_log_softmax)
        else:
            log_ps_a = self._log_ps_a_default(
                action, action_log_softmax, batch_size, atoms
            )

        with torch.no_grad(), self.value_network_params.to_module(self.value_network):
            # Calculate nth next state probabilities
            next_td = step_mdp(tensordict)
            self.value_network(next_td)  # Probabilities p(s_t+n, ·; θonline)

            next_td_action = next_td.get(self.tensor_keys.action)
            if self.action_space == "categorical":
                argmax_indices_ns = next_td_action.squeeze(-1)
            else:
                argmax_indices_ns = next_td_action.argmax(-1)  # one-hot encoding
            with self.target_value_network_params.to_module(self.value_network):
                self.value_network(next_td)  # Probabilities p(s_t+n, ·; θtarget)
            pns = next_td.get(self.tensor_keys.action_value).exp()
            # Double-Q probabilities
            # p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            pns_a = pns[range(batch_size), :, argmax_indices_ns]

            # Compute Tz (Bellman operator T applied to z)
            # Tz = R^n + (γ^n)z (accounting for terminal states)
            if isinstance(discount, torch.Tensor):
                discount = discount.to("cpu")
            # done = done.to("cpu")
            terminated = terminated.to("cpu")
            reward = reward.to("cpu")
            support = support.to("cpu")
            pns_a = pns_a.to("cpu")

            Tz = reward + (1 - terminated.to(reward.dtype)) * discount * support
            if Tz.shape != torch.Size([batch_size, atoms]):
                raise RuntimeError(
                    "Tz shape must be torch.Size([batch_size, atoms]), "
                    f"got Tz.shape={Tz.shape} and batch_size={batch_size}, "
                    f"atoms={atoms}"
                )
            # Clamp between supported values
            Tz = Tz.clamp_(min=Vmin, max=Vmax)
            if not torch.isfinite(Tz).all():
                raise RuntimeError("Tz has some non-finite elements")
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - Vmin) / delta_z  # b = (Tz - Vmin) / Δz
            low, up = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            low[(up > 0) & (low == up)] -= 1
            up[(low < (atoms - 1)) & (low == up)] += 1

            # Distribute probability of Tz
            m = torch.zeros(batch_size, atoms)
            offset = torch.linspace(
                0,
                ((batch_size - 1) * atoms),
                batch_size,
                dtype=torch.int64,
                # device=device,
            )
            offset = offset.unsqueeze(1).expand(batch_size, atoms)
            index = (low + offset).view(-1)
            tensor = (pns_a * (up.float() - b)).view(-1)
            # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, index, tensor)
            index = (up + offset).view(-1)
            tensor = (pns_a * (b - low.float())).view(-1)
            # m_u = m_u + p(s_t+n, a*)(b - l)
            m.view(-1).index_add_(0, index, tensor)

        # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        loss = -torch.sum(m.to(input_tensordict.device) * log_ps_a, 1)
        input_tensordict.set(
            self.tensor_keys.priority,
            loss.detach().unsqueeze(1).to(input_tensordict.device),
            inplace=True,
        )
        loss = _reduce(loss, reduction=self.reduction)
        td_out = TensorDict({"loss": loss}, [])
        return td_out

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        if value_type is ValueEstimators.TD1:
            raise NotImplementedError(
                f"value type {value_type} is not implemented for {self.__class__.__name__}."
            )
        elif value_type is ValueEstimators.TD0:
            # see forward call
            pass
        elif value_type is ValueEstimators.GAE:
            raise NotImplementedError(
                f"value type {value_type} is not implemented for {self.__class__.__name__}."
            )
        elif value_type is ValueEstimators.TDLambda:
            raise NotImplementedError(
                f"value type {value_type} is not implemented for {self.__class__.__name__}."
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

    def _default_value_estimator(self):
        self.make_value_estimator(ValueEstimators.TD0)
