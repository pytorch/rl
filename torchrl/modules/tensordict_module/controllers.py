# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase, TensorDictSequential
from tensordict.utils import NestedKey, unravel_key

from torchrl.data.tensor_specs import Categorical, Composite
from torchrl.modules.utils.utils import get_primers_from_module

if TYPE_CHECKING:
    from torchrl.envs.transforms import Transform


class LowLevelController(TensorDictModuleBase):
    """Deploy a TensorDict policy on independent controller instances.

    The adapter and policy run on a private, flattened TensorDict. Only the
    low-level action and the namespaced next state are written back. Policy
    weights are shared; recurrent and adapter state remain independent for
    every environment and every member of the selected group.

    Args:
        policy (TensorDictModuleBase): pretrained policy with explicit TensorDict
            inputs, outputs, and primers for persistent state.
        decision_spec (Composite): unbatched spec for one controller's high-level
            decisions. All leaves are held fixed during a closed-loop macro step.

    Keyword Args:
        adapter (TensorDictModuleBase or Transform, optional): module applied
            before the policy, using keys relative to the group. Defaults to
            None (pass inputs directly to the policy).
        group_key (NestedKey, optional): group whose batch dimensions enumerate
            controller instances. Defaults to None (the root TensorDict).
        state_key (NestedKey, optional): state namespace within the group.
            Defaults to "_controller".
        policy_action_key (NestedKey, optional): action output of the wrapped
            policy. Defaults to "action".
        action_key (NestedKey, optional): destination action within the group.
            Defaults to "action".
        reset_key (NestedKey, optional): additional per-instance reset signal in
            the group's next observation, such as "fallen". Defaults to None
            (ordinary environment resets only).

    The module does not disable gradients or change the policy's training mode.
    :class:`~torchrl.envs.transforms.ClosedLoopMultiAction` controls inference
    when the controller is deployed in an environment. See also
    :class:`~torchrl.trainers.algorithms.configs.LowLevelControllerConfig`.

    Examples:
        >>> import torch
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.data import Bounded, Composite
        >>> from torchrl.envs.transforms import ClosedLoopMultiAction
        >>> from torchrl.testing.mocking_classes import CountingEnv
        >>> policy = TensorDictModule(
        ...     torch.nn.Identity(), in_keys=["command"], out_keys=["action"])
        >>> controller = LowLevelController(
        ...     policy, Composite(command=Bounded(0, 1, shape=(1,))))
        >>> env = ClosedLoopMultiAction.from_env(CountingEnv(), controller, steps=2)
        >>> td = env.reset().set("command", torch.ones(1))
        >>> env.step(td)["next", "observation"]
        tensor([2], dtype=torch.int32)
        >>> env.close()
    """

    _owns_tensordict_primers = True

    def __init__(
        self,
        policy: TensorDictModuleBase,
        decision_spec: Composite,
        *,
        adapter: TensorDictModuleBase | Transform | None = None,
        group_key: NestedKey | None = None,
        state_key: NestedKey = "_controller",
        policy_action_key: NestedKey = "action",
        action_key: NestedKey = "action",
        reset_key: NestedKey | None = None,
    ):
        super().__init__()
        if not isinstance(decision_spec, Composite) or decision_spec.shape:
            raise ValueError(
                "decision_spec must be an unbatched Composite for one controller."
            )
        if not list(decision_spec.keys(True, True)):
            raise ValueError("decision_spec must contain at least one decision.")
        self.decision_spec = decision_spec.clone()
        self.group_key = None if group_key is None else unravel_key(group_key)
        self.state_key = unravel_key(state_key)
        self.policy_action_key = unravel_key(policy_action_key)
        self.action_key = unravel_key(action_key)
        self.reset_signal = None if reset_key is None else unravel_key(reset_key)
        self.pipeline = TensorDictSequential(
            *([adapter] if adapter is not None else []), policy
        )
        group_path = (
            ()
            if group_key is None
            else (
                (self.group_key,) if isinstance(self.group_key, str) else self.group_key
            )
        )
        state_path = (
            (self.state_key,) if isinstance(self.state_key, str) else self.state_key
        )
        action_path = (
            (self.action_key,) if isinstance(self.action_key, str) else self.action_key
        )
        self._state_path = unravel_key((*group_path, *state_path))
        self._next_state_path = ("next", *group_path, *state_path)
        self._action_path = unravel_key((*group_path, *action_path))

        # Runtime imports are necessary: env transforms import torchrl.modules.
        from torchrl.envs.transforms import Compose, TensorDictPrimer

        discovered = get_primers_from_module(self.pipeline, warn=False)
        pending = [] if discovered is None else [discovered]
        templates = []
        state_keys = set()
        while pending:
            primer = pending.pop(0)
            if isinstance(primer, Compose):
                pending[0:0] = list(primer.transforms)
                continue
            if not isinstance(primer, TensorDictPrimer):
                raise TypeError(
                    "Controller state must be declared with TensorDictPrimer."
                )
            keys = set(primer.primers.keys(True, True))
            if state_keys.intersection(keys) or "is_init" in keys:
                raise ValueError(
                    "Controller primers must declare distinct state keys; is_init is reserved."
                )
            state_keys.update(keys)
            templates.append(primer)
        templates.append(
            TensorDictPrimer(
                Composite(is_init=Categorical(2, shape=(1,), dtype=torch.bool)),
                default_value=True,
            )
        )
        self._primer_templates = templates
        self._state_pairs = [
            (key, ("next", *((key,) if isinstance(key, str) else key)))
            for key in sorted(state_keys, key=str)
        ]
        self._input_keys = [
            key
            for key in self.pipeline.in_keys
            if key not in state_keys and key != "is_init"
        ]
        self.in_keys = [
            unravel_key((*group_path, *((key,) if isinstance(key, str) else key)))
            for key in self._input_keys
        ] + [self._state_path]
        self.out_keys = [self._action_path, self._next_state_path]

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        group = tensordict if self.group_key is None else tensordict.get(self.group_key)
        state = group.get(self.state_key)
        work = group.select(*self._input_keys).clone()
        work.update(state.clone())
        work = self.pipeline(work.reshape(-1)).reshape(group.batch_size)
        next_state = state.clone(recurse=False)
        for key, future_key in self._state_pairs:
            value = work.get(future_key, None)
            if value is None:
                value = work.get(key, state.get(key))
            next_state.set(key, value)
        next_state.set("is_init", torch.zeros_like(state.get("is_init")))
        tensordict.set(self._action_path, work.get(self.policy_action_key))
        tensordict.set(self._next_state_path, next_state)
        return tensordict

    def make_tensordict_primer(self) -> Transform:
        """Return the group's state initialization and partial-reset transforms.

        Returns:
            Transform: primers that infer group shapes from the parent env and
            retain the original policy and adapter initialization values.
        """
        # Runtime import breaks the env/modules initialization cycle.
        from torchrl.envs.transforms._base import Compose
        from torchrl.envs.transforms._env import _ControllerPrimer

        return Compose(
            *[
                _ControllerPrimer(
                    template,
                    group_key=self.group_key,
                    state_key=self.state_key,
                    reset_signal=self.reset_signal,
                )
                for template in self._primer_templates
            ]
        )
