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
    the cadence of :class:`~torchrl.envs.transforms.ActionChunkExecutor`, for
    instance, can be read off ``("next", "observation", "state")``. The reward
    is the negative action norm (an effort penalty) and episodes never
    terminate on their own.

    This is a smoke-test environment for VLA plumbing -- tutorials, tests,
    pipeline checks without simulator dependencies -- not a learnable task.

    Args:
        action_dim (int, optional): size of the continuous action, bounded in
            ``[-1, 1]``. Defaults to ``4``.
        state_dim (int, optional): size of the proprioceptive state; must be
            at least ``action_dim``. Defaults to ``6``.
        image_shape (tuple of int, optional): ``(C, H, W)`` shape of the
            ``uint8`` camera image. Defaults to ``(3, 16, 16)``.
        instruction (str, optional): the constant language instruction.
            Defaults to ``"push the T-shaped block onto the target"``.

    Keyword Args:
        batch_size (torch.Size, optional): number of vectorized copies.
            Defaults to ``torch.Size([])`` (a single environment).
        device (torch.device, optional): device of the specs.
        seed (int, optional): seed for the random images.

    Examples:
        >>> import torch
        >>> from torchrl.envs import TransformedEnv
        >>> from torchrl.envs.custom import ToyVLAEnv
        >>> from torchrl.envs.transforms import ActionChunkExecutor
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
        >>> # attach a chunk executor for closed-loop chunked control
        >>> env = TransformedEnv(
        ...     ToyVLAEnv(batch_size=[2]),
        ...     ActionChunkExecutor(chunk_size=3, replan_interval=2),
        ... )
        >>> env.full_action_spec["action_chunk"].shape
        torch.Size([2, 3, 4])
    """

    def __init__(
        self,
        action_dim: int = 4,
        state_dim: int = 6,
        image_shape: tuple[int, int, int] = (3, 16, 16),
        instruction: str = _DEFAULT_INSTRUCTION,
        *,
        batch_size: torch.Size | None = None,
        device: torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        if state_dim < action_dim:
            raise ValueError(
                f"state_dim ({state_dim}) must be at least action_dim "
                f"({action_dim}): the state echoes the executed action."
            )
        super().__init__(
            batch_size=torch.Size(batch_size) if batch_size is not None else None,
            device=device,
        )
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.image_shape = tuple(int(dim) for dim in image_shape)
        self.instruction = str(instruction)
        batch = self.batch_size
        self.observation_spec = Composite(
            observation=Composite(
                image=Unbounded(
                    shape=(*batch, *self.image_shape),
                    dtype=torch.uint8,
                    device=self.device,
                ),
                state=Unbounded(
                    shape=(*batch, self.state_dim), device=self.device
                ),
                shape=batch,
            ),
            language_instruction=NonTensor(
                shape=batch, example_data=self.instruction, device=self.device
            ),
            shape=batch,
        )
        self.action_spec = Bounded(
            -1.0, 1.0, shape=(*batch, self.action_dim), device=self.device
        )
        self.reward_spec = Unbounded(shape=(*batch, 1), device=self.device)
        self.done_spec = Categorical(
            2, dtype=torch.bool, shape=(*batch, 1), device=self.device
        )
        self._rng = torch.Generator()
        self._set_seed(seed)

    def _instruction_stack(self):
        if not self.batch_size:
            return self.instruction
        stack = [self.instruction] * self.batch_size.numel()
        return NonTensorStack(*stack).reshape(self.batch_size)

    def _obs(self, state: torch.Tensor) -> TensorDict:
        image = torch.randint(
            0,
            256,
            (*self.batch_size, *self.image_shape),
            dtype=torch.uint8,
            generator=self._rng,
        ).to(self.device)
        return TensorDict(
            {
                "observation": {"image": image, "state": state},
                "language_instruction": self._instruction_stack(),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs) -> TensorDict:
        state = torch.zeros(*self.batch_size, self.state_dim, device=self.device)
        out = self._obs(state)
        out.update(self.full_done_spec.zero())
        return out

    def _step(self, tensordict: TensorDictBase) -> TensorDict:
        action = tensordict.get("action")
        state = torch.zeros(*self.batch_size, self.state_dim, device=self.device)
        state[..., : self.action_dim] = action
        out = self._obs(state)
        out["reward"] = -action.norm(dim=-1, keepdim=True)
        out.update(self.full_done_spec.zero())
        return out

    def _set_seed(self, seed: int | None) -> None:
        if seed is not None:
            self._rng.manual_seed(seed)
