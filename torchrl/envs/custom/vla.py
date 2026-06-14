# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from tensordict import NonTensorStack, TensorDict, TensorDictBase

from torchrl.data.tensor_specs import (
    Bounded,
    Categorical,
    Composite,
    NonTensor,
    Unbounded,
)
from torchrl.envs.common import EnvBase

__all__ = ["ToyVLAEnv"]

_DEFAULT_INSTRUCTION = "push the T-shaped block onto the target"


class ToyVLAEnv(EnvBase):
    """A minimal synthetic environment speaking the canonical VLA TensorDict schema.

    Observations follow the canonical VLA layout (see :ref:`the VLA reference
    page <ref_vla>`): a random camera ``image`` and a proprioceptive ``state``
    under ``observation``, plus a constant ``language_instruction`` at the
    root. The state's first ``action_dim`` entries echo the action executed at
    the previous step, which makes execution machinery directly observable:
    the cadence of a chunk-executing policy (e.g.
    :class:`~torchrl.modules.tensordict_module.MultiStepActorWrapper`) can be
    read off ``("next", "observation", "state")``.

    Two modes are available:

    - **Echo mode** (default, ``success_steps=None``): the reward is the
      negative action norm (an effort penalty) and episodes never terminate
      on their own. This is a smoke-test mode for VLA plumbing -- tutorials,
      tests, pipeline checks without simulator dependencies -- not a
      learnable task.
    - **Tracking mode** (``success_steps=k``): a per-episode target action is
      sampled at reset and exposed in the state at
      ``state[..., action_dim:2 * action_dim]`` (requires
      ``state_dim >= 2 * action_dim``). A boolean ``success`` entry turns
      ``True`` -- and the episode terminates -- once the executed action stays
      within ``success_tol`` (infinity-norm) of the target for ``k``
      consecutive steps. The reward is the negative tracking error
      ``-||action - target||``. An oracle that reads the target back from the
      state succeeds with certainty, while a uniform random policy almost
      never does (per-step hit probability ``success_tol ** action_dim``),
      which makes "success rate climbs" a meaningful learning signal for
      sparse-reward RL recipes (pair with
      :class:`~torchrl.envs.transforms.SuccessReward` for a binary
      success-only reward).

    Args:
        action_dim (int, optional): size of the continuous action, bounded in
            ``[-1, 1]``. Defaults to ``4``.
        state_dim (int, optional): size of the proprioceptive state; must be
            at least ``action_dim`` (``2 * action_dim`` in tracking mode).
            Defaults to ``6``.
        image_shape (tuple of int, optional): ``(C, H, W)`` shape of the
            ``uint8`` camera image. Defaults to ``(3, 16, 16)``.
        instruction (str, optional): the constant language instruction.
            Defaults to ``"push the T-shaped block onto the target"``.

    Keyword Args:
        from_pixels (bool, optional): if ``True``, add a root ``pixels`` entry
            (a ``(render_size, render_size, 3)`` ``uint8`` HWC frame) rendering
            the scene: the executed action is drawn as a red marker and, in
            tracking mode, the target as a green one, both mapped from the
            ``[-1, 1]`` action plane. This is the canonical torchrl
            pixels-rendering hook (matching :class:`~torchrl.envs.GymEnv` and
            others) and feeds :class:`~torchrl.record.VideoRecorder` directly.
            Unlike the always-present ``("observation", "image")`` (random
            noise, a stand-in for a camera feed), ``pixels`` visualizes the
            task, so an eval video shows the policy learning to track. Defaults
            to ``False``.
        render_size (int, optional): side length of the square ``pixels``
            frame. Only used when ``from_pixels=True``. Defaults to ``64``.
        success_steps (int, optional): number of consecutive in-tolerance
            steps required for success. ``None`` (default) selects the echo
            mode (no ``success`` entry, never done).
        success_tol (float, optional): per-dimension tolerance around the
            target action. Defaults to ``0.25``. Targets are sampled
            uniformly in ``[-0.5, 0.5]`` so the tolerance ball always fits
            inside the action bounds.
        group_repeats (int, optional): grouped-rollout mode (tracking mode
            only, single environment only): the same target is replayed for
            ``group_repeats`` consecutive episodes before a new one is
            sampled, and an integer ``group_id`` observation entry identifies
            the group. This is the init-state control GRPO-style group
            advantages require (n rollouts per initial state, e.g. grouped by
            :class:`~torchrl.objectives.llm.MCAdvantage`). Defaults to
            ``None`` (a fresh target every episode, no ``group_id`` entry).
        batch_size (torch.Size, optional): number of vectorized copies.
            Defaults to ``torch.Size([])`` (a single environment).
        device (torch.device, optional): device of the specs.
        seed (int, optional): seed for the random images and targets.

    Examples:
        >>> import torch
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.envs import TransformedEnv
        >>> from torchrl.envs.custom import ToyVLAEnv
        >>> from torchrl.envs.transforms import InitTracker
        >>> from torchrl.modules import MultiStepActorWrapper
        >>> env = ToyVLAEnv(batch_size=[2])
        >>> td = env.reset()
        >>> td["observation", "image"].shape, td["observation", "image"].dtype
        (torch.Size([2, 3, 16, 16]), torch.uint8)
        >>> td["language_instruction"][0]
        'push the T-shaped block onto the target'
        >>> # the state echoes the executed action
        >>> td["action"] = 0.5 * torch.ones(2, 4)
        >>> env.step(td)["next", "observation", "state"][:, :4].unique()
        tensor([0.5000])
        >>> # pair it with a chunk-executing policy: the actor predicts 3
        >>> # actions per call and is only re-queried when the cache is empty
        >>> chunk_actor = TensorDictModule(
        ...     lambda state: state[..., :4].unsqueeze(-2).expand(2, 3, 4) + 0.1,
        ...     in_keys=[("observation", "state")],
        ...     out_keys=["action"],
        ... )
        >>> policy = MultiStepActorWrapper(chunk_actor, n_steps=3)
        >>> env = TransformedEnv(ToyVLAEnv(batch_size=[2]), InitTracker())
        >>> env.rollout(4, policy)["action"].shape
        torch.Size([2, 4, 4])
        >>> # tracking mode: an oracle reading the target off the state succeeds
        >>> env = ToyVLAEnv(action_dim=2, state_dim=4, success_steps=2, seed=0)
        >>> td = env.reset()
        >>> for _ in range(2):
        ...     td["action"] = td["observation", "state"][..., 2:4]
        ...     td = env.step(td)["next"]
        >>> bool(td["success"]), bool(td["terminated"])
        (True, True)
    """

    def __init__(
        self,
        action_dim: int = 4,
        state_dim: int = 6,
        image_shape: tuple[int, int, int] = (3, 16, 16),
        instruction: str = _DEFAULT_INSTRUCTION,
        *,
        from_pixels: bool = False,
        render_size: int = 64,
        success_steps: int | None = None,
        success_tol: float = 0.25,
        group_repeats: int | None = None,
        batch_size: torch.Size | None = None,
        device: torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        if success_steps is None and state_dim < action_dim:
            raise ValueError(
                f"state_dim ({state_dim}) must be at least action_dim "
                f"({action_dim}): the state echoes the executed action."
            )
        if success_steps is not None:
            if success_steps < 1:
                raise ValueError(f"success_steps must be >= 1, got {success_steps}.")
            if state_dim < 2 * action_dim:
                raise ValueError(
                    f"state_dim ({state_dim}) must be at least 2 * action_dim "
                    f"({2 * action_dim}) in tracking mode: the state holds the "
                    "executed action followed by the target action."
                )
            if not 0.0 < success_tol <= 0.5:
                raise ValueError(
                    "success_tol must be in (0, 0.5] so the tolerance ball "
                    f"around a target in [-0.5, 0.5] stays reachable, got {success_tol}."
                )
        if group_repeats is not None:
            if success_steps is None:
                raise ValueError(
                    "group_repeats requires the tracking mode: set success_steps."
                )
            if group_repeats < 1:
                raise ValueError(f"group_repeats must be >= 1, got {group_repeats}.")
        super().__init__(
            batch_size=torch.Size(batch_size) if batch_size is not None else None,
            device=device,
        )
        if render_size < 1:
            raise ValueError(f"render_size must be >= 1, got {render_size}.")
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.image_shape = tuple(int(dim) for dim in image_shape)
        self.instruction = str(instruction)
        self.from_pixels = bool(from_pixels)
        self.render_size = int(render_size)
        self.success_steps = int(success_steps) if success_steps is not None else None
        self.success_tol = float(success_tol)
        self.group_repeats = int(group_repeats) if group_repeats is not None else None
        if self.group_repeats is not None and self.batch_size.numel() > 1:
            raise ValueError(
                "group_repeats only supports a single environment "
                f"(batch_size () or (1,)), got batch_size={tuple(self.batch_size)}. "
                "General init-state control belongs to the environment adapters."
            )
        batch = self.batch_size
        observation = Composite(
            image=Unbounded(
                shape=(*batch, *self.image_shape),
                dtype=torch.uint8,
                device=self.device,
            ),
            state=Unbounded(shape=(*batch, self.state_dim), device=self.device),
            shape=batch,
        )
        self.observation_spec = Composite(
            observation=observation,
            language_instruction=NonTensor(
                shape=batch, example_data=self.instruction, device=self.device
            ),
            shape=batch,
        )
        if self.success_steps is not None:
            self.observation_spec["success"] = Categorical(
                2, dtype=torch.bool, shape=(*batch, 1), device=self.device
            )
        if self.group_repeats is not None:
            self.observation_spec["group_id"] = Unbounded(
                shape=(*batch, 1), dtype=torch.int64, device=self.device
            )
        if self.from_pixels:
            self.observation_spec["pixels"] = Unbounded(
                shape=(*batch, self.render_size, self.render_size, 3),
                dtype=torch.uint8,
                device=self.device,
            )
        self.action_spec = Bounded(
            -1.0, 1.0, shape=(*batch, self.action_dim), device=self.device
        )
        self.reward_spec = Unbounded(shape=(*batch, 1), device=self.device)
        self.done_spec = Categorical(
            2, dtype=torch.bool, shape=(*batch, 1), device=self.device
        )
        # Tracking-mode episode state: per-env target and in-tolerance streak.
        # Registered as (non-persistent) buffers so env.to(device) moves them.
        self.register_buffer(
            "_target",
            torch.zeros(*batch, self.action_dim, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "_streak",
            torch.zeros(*batch, 1, dtype=torch.int64, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "_group_id",
            torch.zeros(*batch, 1, dtype=torch.int64, device=self.device),
            persistent=False,
        )
        self._episode_count = 0
        self._rng = torch.Generator()
        self._set_seed(seed)

    def _instruction_stack(self):
        if not self.batch_size:
            return self.instruction
        stack = [self.instruction] * self.batch_size.numel()
        return NonTensorStack(*stack).reshape(self.batch_size)

    def _sample_target(self) -> torch.Tensor:
        target = (
            torch.rand(
                *self.batch_size,
                self.action_dim,
                generator=self._rng,
            )
            - 0.5
        )
        return target.to(self.device)

    def _obs(self, state: torch.Tensor) -> TensorDict:
        image = torch.randint(
            0,
            256,
            (*self.batch_size, *self.image_shape),
            dtype=torch.uint8,
            generator=self._rng,
        ).to(self.device)
        out = TensorDict(
            {
                "observation": {"image": image, "state": state},
                "language_instruction": self._instruction_stack(),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        if self.from_pixels:
            out["pixels"] = self._render(state)
        return out

    def _render(self, state: torch.Tensor) -> torch.Tensor:
        """Render the scene to a ``(*batch, S, S, 3)`` uint8 HWC frame.

        The executed action (``state[..., :action_dim]``) is a red marker and,
        in tracking mode, the target (``state[..., action_dim:2*action_dim]``)
        a green one, both mapped from the ``[-1, 1]`` plane onto the canvas.
        Off the hot path: only reached when ``from_pixels=True`` (recording).
        """
        size = self.render_size
        canvas = torch.full(
            (*self.batch_size, size, size, 3),
            30,
            dtype=torch.uint8,
            device=self.device,
        )

        def to_px(coord: torch.Tensor) -> torch.Tensor:
            return (((coord.clamp(-1.0, 1.0) + 1.0) * 0.5) * (size - 1)).round().long()

        action = state[..., : self.action_dim]
        ax = to_px(action[..., 0])
        ay = to_px(action[..., 1]) if self.action_dim > 1 else torch.zeros_like(ax)
        self._draw_marker(canvas, ay, ax, (220, 60, 60))
        if self.success_steps is not None:
            target = state[..., self.action_dim : 2 * self.action_dim]
            tx = to_px(target[..., 0])
            ty = to_px(target[..., 1]) if self.action_dim > 1 else torch.zeros_like(tx)
            self._draw_marker(canvas, ty, tx, (60, 200, 60))
        return canvas

    def _draw_marker(self, canvas, y, x, color, radius: int = 2) -> None:
        # square marker; loops over the (tiny, recording-only) batch so the
        # per-env top-left corner can index a contiguous block
        size = canvas.shape[-2]
        flat = canvas.reshape(-1, size, size, 3)
        ys = y.reshape(-1)
        xs = x.reshape(-1)
        col = torch.tensor(color, dtype=torch.uint8, device=canvas.device)
        for i in range(flat.shape[0]):
            yi = int(ys[i])
            xi = int(xs[i])
            flat[
                i,
                max(0, yi - radius) : yi + radius + 1,
                max(0, xi - radius) : xi + radius + 1,
                :,
            ] = col

    def _make_state(self, action: torch.Tensor | None) -> torch.Tensor:
        state = torch.zeros(*self.batch_size, self.state_dim, device=self.device)
        if action is not None:
            state[..., : self.action_dim] = action
        if self.success_steps is not None:
            state[..., self.action_dim : 2 * self.action_dim] = self._target
        return state

    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs) -> TensorDict:
        if self.success_steps is not None:
            reset = None
            if tensordict is not None:
                reset = tensordict.get("_reset")
            if reset is None:
                reset = torch.ones(
                    *self.batch_size, 1, dtype=torch.bool, device=self.device
                )
            if self.group_repeats is not None:
                # grouped rollouts (single env): replay the same target for
                # group_repeats consecutive episodes and stamp the group id
                if self._episode_count % self.group_repeats == 0:
                    self._target = self._sample_target()
                self._group_id = torch.full_like(
                    self._group_id, self._episode_count // self.group_repeats
                )
                self._episode_count += 1
            else:
                self._target = torch.where(reset, self._sample_target(), self._target)
            self._streak = torch.where(
                reset, torch.zeros_like(self._streak), self._streak
            )
        out = self._obs(self._make_state(None))
        if self.success_steps is not None:
            out["success"] = torch.zeros(
                *self.batch_size, 1, dtype=torch.bool, device=self.device
            )
        if self.group_repeats is not None:
            out["group_id"] = self._group_id.clone()
        out.update(self.full_done_spec.zero())
        return out

    def _step(self, tensordict: TensorDictBase) -> TensorDict:
        action = tensordict.get("action")
        out = self._obs(self._make_state(action))
        if self.success_steps is None:
            out["reward"] = -action.norm(dim=-1, keepdim=True)
            out.update(self.full_done_spec.zero())
            return out
        error = action - self._target
        reward = -error.norm(dim=-1, keepdim=True)
        in_tol = (error.abs() <= self.success_tol).all(-1, keepdim=True)
        streak = torch.where(in_tol, self._streak + 1, 0)
        step_mask = tensordict.get("_step", None)
        if step_mask is not None:
            # partial-step contract (see EnvBase.step): batch-locked envs are
            # trusted to handle the "_step" mask themselves. Masked-out envs
            # (e.g. done inside a MultiAction chunk) keep their streak frozen
            # - so success/done persist - and emit a zero reward.
            step_mask = step_mask.view(self._streak.shape)
            streak = torch.where(step_mask, streak, self._streak)
            reward = torch.where(step_mask, reward, torch.zeros_like(reward))
        self._streak = streak
        success = self._streak >= self.success_steps
        out["reward"] = reward
        out["success"] = success
        if self.group_repeats is not None:
            out["group_id"] = self._group_id.clone()
        out["terminated"] = success
        out["done"] = success
        return out

    def _set_seed(self, seed: int | None) -> None:
        if seed is not None:
            self._rng.manual_seed(seed)
