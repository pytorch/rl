# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import math
from collections.abc import Sequence

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


class _MicroDuckAdapter(TensorDictModuleBase):
    def __init__(self, tasks, skills, argument_key, control_period_s):
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
        self.register_buffer("skill_task_ids", skills)
        rows = tasks[skills]
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


class MicroDuckController(LowLevelController):
    """Deploy a MicroDuck locomotion policy using discrete or parameterized skills.

    A skill selects a row of the task library used to train the walker. The
    adapter replaces the command and gait fields in the leading MicroDuck
    observation and supplies the original task id. The generic controller
    manages independent recurrent state and gait clocks for all agents.

    Args:
        walker (TensorDictModuleBase): policy reading observation and optionally
            task_id and explicit recurrent state, and writing action.
        tasks (MicroDuckTask or sequence of MicroDuckTask): training task library
            in its original task-id order.

    Keyword Args:
        skills (sequence of int, optional): task indices offered as skills.
            Defaults to all tasks. High-level skill values index this selection.
        group_key (NestedKey, optional): agent group. Defaults to "agents".
            Pass None for a single controller at the root.
        argument_key (NestedKey, optional): normalized two-dimensional command
            argument in [-1, 1], relative to the group. Defaults to None, using
            the selected task's command-box midpoint.
        control_period_s (float, optional): seconds per physical step.
            Defaults to 0.02; must match the deployed environment.
        reset_key (NestedKey, optional): per-agent respawn signal relative to the
            group. Defaults to "fallen". Pass None if only episodes reset.

    Examples:
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules import MLP
        >>> tasks = [MicroDuckEnv.standing_task(), MicroDuckEnv.tracking_task(0.2)]
        >>> walker = TensorDictModule(
        ...     MLP(in_features=MicroDuckEnv.OBSERVATION_DIM,
        ...         out_features=MicroDuckEnv.NUM_JOINTS, num_cells=[32]),
        ...     in_keys=["observation"], out_keys=["action"])
        >>> controller = MicroDuckController(walker, tasks, skills=[1, 0])
        >>> controller.decision_spec["skill"].n
        2
    """

    def __init__(
        self,
        walker: TensorDictModuleBase,
        tasks: MicroDuckTask | Sequence[MicroDuckTask],
        *,
        skills: Sequence[int] | None = None,
        group_key: NestedKey | None = "agents",
        argument_key: NestedKey | None = None,
        control_period_s: float = 0.02,
        reset_key: NestedKey | None = "fallen",
    ):
        if not math.isfinite(control_period_s) or control_period_s <= 0:
            raise ValueError("control_period_s must be finite and positive.")
        library = MicroDuckEnv.stack_tasks(tasks)
        # Place supplied task metadata alongside the supplied policy.
        tensor = next(
            iter(walker.parameters()), next(iter(walker.buffers()), library.command_low)
        )
        library = library.to(tensor.device)
        if skills is None:
            skills = list(range(library.shape[0]))
        if not len(skills) or any(
            not isinstance(skill, int)
            or isinstance(skill, bool)
            or skill < 0
            or skill >= library.shape[0]
            for skill in skills
        ):
            raise ValueError("skills must be non-empty integer indices into tasks.")
        skill_ids = torch.tensor(skills, dtype=torch.long, device=tensor.device)
        argument_key = None if argument_key is None else unravel_key(argument_key)
        decision_spec = Composite(skill=Categorical(len(skills), device=tensor.device))
        if argument_key is not None:
            decision_spec[argument_key] = Bounded(
                -1.0, 1.0, shape=(2,), device=tensor.device
            )
        super().__init__(
            walker,
            decision_spec,
            adapter=_MicroDuckAdapter(
                library, skill_ids, argument_key, control_period_s
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


def microduck_skill_env(
    env: EnvBase,
    walker: TensorDictModuleBase,
    tasks: MicroDuckTask | Sequence[MicroDuckTask],
    *,
    skills: Sequence[int] | None = None,
    steps: int = 5,
    control_period_s: float = 0.02,
    group_key: NestedKey | None = "agents",
    argument_key: NestedKey | None = None,
) -> TransformedEnv:
    """Wrap a task with skill decisions, summed rewards, and MicroDuck observations.

    Uses MicroDuckController and ClosedLoopMultiAction for all execution and
    recurrent state. A one-hot encoding of the last skill is appended to each
    observation. The returned fallen flag reports any fall during the current
    decision; raw inner fall signals reset the corresponding controller first.

    Args:
        env (EnvBase): joint-level task exposing observation and fallen in the
            selected group. Its leading observation must use MicroDuck's layout.
        walker (TensorDictModuleBase): pretrained MicroDuck policy.
        tasks (MicroDuckTask or sequence of MicroDuckTask): walker's task library.

    Keyword Args:
        skills (sequence of int, optional): selected library indices, in decision
            order. Defaults to all tasks.
        steps (int, optional): physical steps per decision. Defaults to 5.
        control_period_s (float, optional): physical step duration. Defaults to 0.02.
        group_key (NestedKey, optional): controller group. Defaults to "agents".
        argument_key (NestedKey, optional): normalized command argument key.
            Defaults to None for discrete decisions.

    Returns:
        TransformedEnv: high-level task usable by ordinary collectors and losses.

    Examples:
        >>> # Given a task env and a walker trained on this library:
        >>> env = microduck_skill_env(task_env, walker, tasks, steps=5)  # doctest: +SKIP
        >>> data = env.rollout(10, high_level_actor)  # doctest: +SKIP
    """
    controller = MicroDuckController(
        walker,
        tasks,
        skills=skills,
        group_key=group_key,
        argument_key=argument_key,
        control_period_s=control_period_s,
    )
    # The caller supplied an env and policy; align their existing tensors.
    controller.to(env.device)
    spec = (
        env.observation_spec if group_key is None else env.observation_spec[group_key]
    )
    if spec["observation"].shape[-1] < MicroDuckEnv.OBSERVATION_DIM:
        raise ValueError(
            "The task observation must start with the MicroDuck observation."
        )
    num_skills = controller.decision_spec["skill"].n
    result = ClosedLoopMultiAction.from_env(env, controller, steps=steps)
    result.insert_transform(-1, _MicroDuckSkillHistory(env, group_key, num_skills))
    # Run outer reporting even when termination skips the final physical step.
    return TransformedEnv(
        result, _MicroDuckSkillObservation(group_key, num_skills), auto_unwrap=False
    )
