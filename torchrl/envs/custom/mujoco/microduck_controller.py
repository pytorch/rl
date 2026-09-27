# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from tensordict import NestedKey
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import unravel_key

from torchrl.data.tensor_specs import Bounded, Categorical, Composite, Unbounded
from torchrl.envs.common import EnvBase
from torchrl.envs.custom.mujoco.microduck import MicroDuckEnv, MicroDuckTask
from torchrl.envs.transforms import (
    ClosedLoopMultiAction,
    TensorDictPrimer,
    Transform,
    TransformedEnv,
)
from torchrl.modules.tensordict_module.controllers import LowLevelController

if TYPE_CHECKING:
    from torchrl.modules.tensordict_module.zoo import MicroDuckSkills


class _MicroDuckAdapter(TensorDictModuleBase):
    def __init__(self, task_library, skill_ids, argument_key, control_period_s):
        super().__init__()
        self.argument_key = argument_key
        self.control_period_s = control_period_s
        self.in_keys = ["observation", "skill", "gait_phase", "gait_elapsed"]
        if argument_key is not None:
            self.in_keys.append(argument_key)
        self.out_keys = [
            "observation",
            "task_id",
            ("next", "gait_phase"),
            ("next", "gait_elapsed"),
        ]
        self.register_buffer("skill_task_ids", skill_ids)
        rows = task_library[skill_ids]
        for name in (
            "command_low",
            "command_high",
            "gait_frequency_hz",
            "gait_frequency_per_mps",
        ):
            self.register_buffer(name, getattr(rows, name).clone())

    def make_tensordict_primer(self):
        return TensorDictPrimer(
            gait_phase=Unbounded((), device=self.command_low.device),
            gait_elapsed=Unbounded((), device=self.command_low.device),
            default_value={
                "gait_phase": MicroDuckEnv.GAIT_PHASE_OFFSET,
                "gait_elapsed": 0.0,
            },
        )

    def forward(self, td):
        skill = td["skill"]
        low, high = self.command_low[skill], self.command_high[skill]
        command = (
            (low + high) / 2
            if self.argument_key is None
            else low + 0.5 * (td.get(self.argument_key) + 1) * (high - low)
        )
        observation = td["observation"][..., : MicroDuckEnv.OBSERVATION_DIM].clone()
        phase, elapsed = td["gait_phase"], td["gait_elapsed"]
        start = MicroDuckEnv.COMMAND_START
        observation[..., start : start + 2] = command
        start = MicroDuckEnv.GAIT_PHASE_START
        observation[..., start] = phase.sin()
        observation[..., start + 1] = phase.cos()
        observation[..., start + 2] = (
            elapsed / MicroDuckEnv.GAIT_RAMP_DURATION_S
        ).clamp(max=1)
        frequency = self.gait_frequency_hz[skill] + self.gait_frequency_per_mps[
            skill
        ] * command.norm(dim=-1)
        td["observation"] = observation
        td["task_id"] = self.skill_task_ids[skill].unsqueeze(-1)
        td["next", "gait_phase"] = (
            phase + 2 * math.pi * frequency * self.control_period_s
        )
        td["next", "gait_elapsed"] = elapsed + self.control_period_s
        return td


class MicroDuckSkillController(LowLevelController):
    """Translate high-level MicroDuck skill decisions into joint targets.

    A skill selects a row of the library used to train ``skill_policy``. The
    adapter replaces the command and gait fields in the leading MicroDuck
    observation and supplies the original task id. The generic controller
    manages independent recurrent state and gait clocks for all agents.

    Args:
        skill_policy: Policy reading ``observation``, ``task_id`` and any
            explicit recurrent state, and writing the joint ``action``.
        task_library: Training task library in its original task-id order.

    Keyword Args:
        skill_ids: Task indices offered as high-level skills.
            Defaults to all tasks. High-level skill values index this selection.
        group_key: Agent group. Defaults to ``"agents"``.
            Pass None for a single controller at the root.
        argument_key: Normalized two-dimensional command argument in
            ``[-1, 1]``, relative to the group. Defaults to None, using the
            selected task's command-box midpoint.
        control_period_s: Seconds per physical step.
            Defaults to 0.02; must match the deployed environment.
        reset_key: Per-agent respawn signal relative to the
            group. Defaults to "fallen". Pass None if only episodes reset.

    Examples:
        Bind a policy to the exact task ids represented by its embeddings:

        >>> import torch
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules import MLP
        >>> task_library = torch.stack([
        ...     MicroDuckEnv.standing_task(),
        ...     MicroDuckEnv.tracking_task(0.2),
        ... ])
        >>> skill_policy = TensorDictModule(
        ...     MLP(in_features=MicroDuckEnv.OBSERVATION_DIM,
        ...         out_features=MicroDuckEnv.NUM_JOINTS, num_cells=[32]),
        ...     in_keys=["observation"], out_keys=["action"])
        >>> controller = MicroDuckSkillController(
        ...     skill_policy, task_library, skill_ids=[1, 0]
        ... )
        >>> controller.decision_spec["skill"].n
        2

    .. seealso::
        :class:`MicroDuckSkillEnv` incorporates this controller into a
        high-level environment; :class:`~torchrl.modules.LowLevelController`
        supplies the generic state routing; and
        :class:`~torchrl.modules.tensordict_module.zoo.MicroDuckSkills`
        packages a trained policy with its task library.
    """

    def __init__(
        self,
        skill_policy: TensorDictModuleBase,
        task_library: MicroDuckTask | Sequence[MicroDuckTask],
        *,
        skill_ids: Sequence[int] | None = None,
        group_key: NestedKey | None = "agents",
        argument_key: NestedKey | None = None,
        control_period_s: float = 0.02,
        reset_key: NestedKey | None = "fallen",
    ):
        if not math.isfinite(control_period_s) or control_period_s <= 0:
            raise ValueError("control_period_s must be finite and positive.")
        task_library = MicroDuckEnv.stack_tasks(task_library)
        tensor = next(
            iter(skill_policy.parameters()),
            next(iter(skill_policy.buffers()), task_library.command_low),
        )
        task_library = task_library.to(tensor.device)
        if skill_ids is None:
            skill_ids = list(range(task_library.shape[0]))
        if not len(skill_ids) or any(
            not isinstance(skill, int)
            or isinstance(skill, bool)
            or skill < 0
            or skill >= task_library.shape[0]
            for skill in skill_ids
        ):
            raise ValueError(
                "skill_ids must be non-empty integer indices into task_library."
            )
        skill_ids = torch.tensor(skill_ids, dtype=torch.long, device=tensor.device)
        argument_key = None if argument_key is None else unravel_key(argument_key)
        decision_spec = Composite(
            skill=Categorical(len(skill_ids), device=tensor.device)
        )
        if argument_key is not None:
            decision_spec[argument_key] = Bounded(
                -1.0, 1.0, shape=(2,), device=tensor.device
            )
        super().__init__(
            skill_policy,
            decision_spec,
            adapter=_MicroDuckAdapter(
                task_library, skill_ids, argument_key, control_period_s
            ),
            group_key=group_key,
            reset_key=reset_key,
        )


class _MicroDuckSkillHistory(TensorDictPrimer):
    """Accumulate raw falls after controller reset transforms have consumed them."""

    def __init__(self, env, group_key, num_skills):
        group = (
            ()
            if group_key is None
            else ((group_key,) if isinstance(group_key, str) else group_key)
        )
        self.fallen_path = (*group, "_skill_summary", "fallen")
        self.skill_path = (*group, "_skill_summary", "skill")
        self.group_key = group_key
        spec = (
            env.observation_spec
            if group_key is None
            else env.observation_spec[group_key]
        )
        primers = Composite(shape=env.batch_size, device=env.device)
        if group_key is not None:
            primers[group_key] = Composite(shape=spec.shape, device=spec.device)
        primers[self.fallen_path] = Categorical(
            2, shape=(*spec.shape, 1), dtype=torch.bool, device=spec.device
        )
        primers[self.skill_path] = Categorical(
            num_skills, shape=spec.shape, device=spec.device
        )
        super().__init__(primers, default_value=0, expand_specs=False)

    def _step(self, td, next_td):
        group = next_td if self.group_key is None else next_td[self.group_key]
        source = td if self.group_key is None else td[self.group_key]
        next_td[self.fallen_path] = td[self.fallen_path] | group["fallen"]
        skill = source["skill"]
        active = td.get("_step", None)
        if active is not None:
            active = active.reshape(
                (*active.shape, *([1] * (skill.ndim - active.ndim)))
            )
            skill = torch.where(active, skill, td[self.skill_path])
        next_td[self.skill_path] = skill.clone()
        return next_td


class _MicroDuckSkillObservation(Transform):
    """Expose the last skill and one fall flag per high-level decision."""

    def __init__(self, group_key, num_skills):
        super().__init__()
        self.group_key = group_key
        self.num_skills = num_skills

    def _inv_call(self, td):
        group = td if self.group_key is None else td[self.group_key]
        group["observation"] = group["observation"][..., : -self.num_skills]
        group["_skill_summary", "fallen"] = torch.zeros_like(group["fallen"])
        return td

    def _call(self, td):
        group = td if self.group_key is None else td[self.group_key]
        skill = torch.nn.functional.one_hot(
            group["_skill_summary", "skill"], self.num_skills
        )
        group["observation"] = torch.cat(
            (group["observation"], skill.to(group["observation"].dtype)), -1
        )
        group["fallen"] = group["_skill_summary", "fallen"]
        return td

    def _reset(self, td, reset_td):
        return self._call(reset_td)

    def transform_observation_spec(self, spec):
        group = spec if self.group_key is None else spec[self.group_key]
        observation = group["observation"]
        group["observation"] = Unbounded(
            (*observation.shape[:-1], observation.shape[-1] + self.num_skills),
            dtype=observation.dtype,
            device=observation.device,
        )
        return spec


class MicroDuckSkillEnv(TransformedEnv):
    """High-level environment whose actions select frozen MicroDuck skills.

    A :class:`MicroDuckSkills` object is part of this environment's transition
    dynamics: each high-level decision is held fixed while its policy computes
    fresh joint targets for up to ``control_steps_per_decision`` physical
    steps. Rewards are summed, termination can end the decision early, and
    recurrent policy state is reset independently for ducks that fall.

    Construct this class with :meth:`from_env`; the wrapped task environment
    supplies physics, observations and rewards, while this class changes its
    action space from joint targets to skill decisions.

    Examples:
        Load the published skill policy and promote a joint-level game into a
        high-level skill environment:

        >>> from torchrl.envs import MicroDuckSkillEnv
        >>> from torchrl.modules.tensordict_module.zoo import MicroDuckSkills
        >>> skills = MicroDuckSkills.from_pretrained()  # doctest: +SKIP
        >>> base_env = make_microduck_game_env()  # doctest: +SKIP
        >>> env = MicroDuckSkillEnv.from_env(  # doctest: +SKIP
        ...     base_env, skills, control_steps_per_decision=5
        ... )
        >>> rollout = env.rollout(10, high_level_policy)  # doctest: +SKIP

    .. seealso::
        :class:`MicroDuckSkillController` maps one skill decision to policy
        inputs and joint targets;
        :class:`~torchrl.modules.tensordict_module.zoo.MicroDuckSkills`
        keeps the policy and task metadata together; and
        :class:`~torchrl.envs.transforms.ClosedLoopMultiAction` implements the
        generic repeated closed-loop execution.
    """

    @classmethod
    def from_env(
        cls,
        env: EnvBase,
        skills: MicroDuckSkills,
        *,
        skill_ids: Sequence[int] | None = None,
        control_steps_per_decision: int = 5,
        control_period_s: float = 0.02,
        group_key: NestedKey | None = "agents",
        argument_key: NestedKey | None = None,
        reset_key: NestedKey | None = "fallen",
    ) -> MicroDuckSkillEnv:
        """Build a skill-level environment from joint-level task dynamics.

        Args:
            env: Joint-level task exposing a leading MicroDuck observation and
                a fall signal in the selected group.
            skills: Policy, ordered task library and action scale to deploy.
            skill_ids: Task-library rows offered as high-level skills.
            control_steps_per_decision: Maximum physical controller steps per
                high-level decision.
            control_period_s: Duration of one physical step in seconds.
            group_key: Agent group, or None for a controller at the root.
            argument_key: Optional normalized two-dimensional command argument.
            reset_key: Per-agent reset signal, or None for episode resets only.

        Returns:
            A high-level :class:`MicroDuckSkillEnv`.
        """
        env_action_scale = getattr(env, "action_scale", None)
        if env_action_scale is not None and not math.isclose(
            float(env_action_scale), skills.action_scale
        ):
            raise ValueError(
                "The skill policy was trained with "
                f"action_scale={skills.action_scale}; the environment uses "
                f"action_scale={float(env_action_scale)}."
            )
        controller = MicroDuckSkillController(
            skills.policy,
            skills.task_library,
            skill_ids=skill_ids,
            group_key=group_key,
            argument_key=argument_key,
            control_period_s=control_period_s,
            reset_key=reset_key,
        )
        controller.to(env.device)
        spec = (
            env.observation_spec
            if group_key is None
            else env.observation_spec[group_key]
        )
        if spec["observation"].shape[-1] < MicroDuckEnv.OBSERVATION_DIM:
            raise ValueError(
                "The task observation must start with the MicroDuck observation."
            )
        num_skills = controller.decision_spec["skill"].n
        inner = ClosedLoopMultiAction.from_env(
            env, controller, steps=control_steps_per_decision
        )
        inner.insert_transform(-1, _MicroDuckSkillHistory(env, group_key, num_skills))
        return cls(
            inner,
            _MicroDuckSkillObservation(group_key, num_skills),
            auto_unwrap=False,
        )
