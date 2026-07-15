# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Wrappers exposing external (e.g. LeRobot) policies as VLA policies."""
from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import Any

import torch
from tensordict import TensorDictBase
from tensordict.utils import NestedKey

from torchrl.data.vla.schema import IMAGE_KEY, INSTRUCTION_KEY, STATE_KEY
from torchrl.modules.vla.common import InputMode, OutputMode, VLAWrapperBase

_has_lerobot = importlib.util.find_spec("lerobot") is not None

__all__ = ["LeRobotPolicyWrapper"]


class LeRobotPolicyWrapper(VLAWrapperBase):
    """Expose an external (LeRobot-style) policy as a TorchRL VLA policy.

    This adapts a pretrained action-chunk policy -- such as a LeRobot
    ``PreTrainedPolicy`` (ACT, Diffusion Policy, SmolVLA, pi0, ...) -- to the
    canonical VLA key contract (see :class:`~torchrl.modules.vla.VLAWrapperBase`),
    so an off-the-shelf checkpoint can be evaluated or fine-tuned inside the
    TorchRL stack. On :meth:`forward` it builds a LeRobot-style batch dict from
    the canonical observation keys (``observation.state``,
    ``observation.images.<camera>``, ``task``), calls the wrapped policy, and
    writes the returned continuous action chunk under
    ``("vla_action", "chunk")``.

    The wrapped object can be any callable / module that maps a LeRobot batch
    dict to an action chunk of shape ``[B, chunk_size, action_dim]``; by default
    the wrapper tries the policy's ``predict_action_chunk``, ``select_action``
    then ``forward`` methods (override with ``predict_fn`` for a specific API).

    Args:
        policy: the wrapped policy (a callable / ``nn.Module`` that returns an
            action chunk given a LeRobot batch dict).

    Keyword Args:
        action_dim (int): the dimensionality of a single action.
        chunk_size (int): the action-chunk horizon.
        predict_fn (Callable, optional): a ``(policy, batch) -> chunk`` callable
            overriding the default policy-call dispatch.
        camera_name (str): the LeRobot camera name to use for the image key
            (``observation.images.<camera_name>``). Defaults to ``"image"``.
        use_state (bool): whether to forward the proprioceptive state.
            Defaults to ``True``.
        image_key, state_key, instruction_key (NestedKey): canonical input keys.

    .. warning::
        Loading a real LeRobot checkpoint (:meth:`from_pretrained`) requires the
        optional ``lerobot`` package and targets its documented API; that path
        is **best-effort / not exercised in CI**. The key mapping and base-class
        integration are tested with a stand-in policy.

    .. note::
        Only the continuous chunk head is supported -- external policies emit
        continuous chunks, not TorchRL action-token logits.

    Examples:
        >>> import torch
        >>> from tensordict import NonTensorStack, TensorDict
        >>> from torchrl.modules.vla import LeRobotPolicyWrapper
        >>> class DummyPolicy:
        ...     def predict_action_chunk(self, batch):
        ...         b = batch["observation.state"].shape[0]
        ...         return torch.zeros(b, 4, 7)
        >>> policy = LeRobotPolicyWrapper(DummyPolicy(), action_dim=7, chunk_size=4)
        >>> td = TensorDict(
        ...     {
        ...         "observation": {
        ...             "image": torch.zeros(2, 3, 16, 16),
        ...             "state": torch.zeros(2, 5),
        ...         },
        ...         "language_instruction": NonTensorStack("pick", "place"),
        ...     },
        ...     batch_size=[2],
        ... )
        >>> policy(td)["vla_action", "chunk"].shape
        torch.Size([2, 4, 7])
    """

    def __init__(
        self,
        policy: Any,
        *,
        action_dim: int,
        chunk_size: int,
        predict_fn: Callable | None = None,
        camera_name: str = "image",
        use_state: bool = True,
        image_key: NestedKey = IMAGE_KEY,
        state_key: NestedKey = STATE_KEY,
        instruction_key: NestedKey = INSTRUCTION_KEY,
        input_mode: InputMode = "canonical",
        output_mode: OutputMode | None = None,
        inplace: bool | str | None = True,
    ) -> None:
        super().__init__(
            action_dim=action_dim,
            chunk_size=chunk_size,
            action_head="continuous",
            use_state=use_state,
            input_mode=input_mode,
            output_mode=output_mode,
            inplace=inplace,
        )
        self.set_keys(image=image_key, state=state_key, instruction=instruction_key)
        self.policy = policy
        self.predict_fn = predict_fn
        self.camera_name = camera_name

    def _to_lerobot_batch(self, tensordict: TensorDictBase) -> dict:
        batch = {f"observation.images.{self.camera_name}": self._get_image(tensordict)}
        if self.use_state:
            batch["observation.state"] = self._get_state(tensordict)
        instruction = self._get_instruction(tensordict)
        batch["task"] = getattr(instruction, "tolist", lambda: instruction)()
        return batch

    def _call_policy(self, batch: dict) -> torch.Tensor:
        if self.predict_fn is not None:
            return self.predict_fn(self.policy, batch)
        # NB: ``select_action`` is intentionally not in this list -- LeRobot's
        # select_action returns a single-step action, not a chunk. Use a
        # ``predict_fn`` for policies with a different chunk-prediction API.
        for name in ("predict_action_chunk", "forward"):
            method = getattr(self.policy, name, None)
            if callable(method):
                return method(batch)
        if callable(self.policy):
            return self.policy(batch)
        raise TypeError(
            "policy must expose 'predict_action_chunk' or 'forward', or be "
            "callable, or a predict_fn must be provided."
        )

    def _predict(self, tensordict: TensorDictBase) -> torch.Tensor:
        chunk = self._call_policy(self._to_lerobot_batch(tensordict))
        if chunk.ndim < 3 or chunk.shape[-2:] != (self.chunk_size, self.action_dim):
            raise ValueError(
                f"the wrapped policy must return an action chunk of shape "
                f"[..., {self.chunk_size}, {self.action_dim}], got "
                f"{tuple(chunk.shape)}."
            )
        # the base re-expands the trailing dim into (chunk_size, action_dim)
        return chunk.flatten(-2, -1)

    @classmethod
    def from_pretrained(
        cls, repo_id: str, *, action_dim: int, chunk_size: int, **kwargs
    ) -> LeRobotPolicyWrapper:
        """Load a pretrained LeRobot policy and wrap it (requires ``lerobot``)."""
        if not _has_lerobot:
            raise ImportError(
                "The `lerobot` package is required to load a pretrained LeRobot "
                "policy. Install it with `pip install lerobot`, or wrap an "
                "already-instantiated policy with LeRobotPolicyWrapper(policy, ...)."
            )
        # Lazy import of the optional `lerobot` dependency. NB: this API is
        # written against the documented LeRobot interface and is not exercised
        # in CI -- see the class docstring warning.
        from lerobot.common.policies.factory import make_policy

        policy = make_policy(repo_id)
        return cls(policy, action_dim=action_dim, chunk_size=chunk_size, **kwargs)
