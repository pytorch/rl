# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Canonical TensorDict schema for Vision-Language-Action (VLA) data.

A VLA trajectory maps one or more camera images, optional proprioceptive
state, and a natural-language instruction to robot actions (typically emitted
as a short *action chunk* of future steps). This module defines the canonical
:class:`~tensordict.utils.NestedKey` layout used by the VLA transforms,
policies and losses so that datasets, models and objectives agree on a single
key convention. The layout mirrors the keys produced by
:class:`~torchrl.data.datasets.OpenXExperienceReplay` and the LeRobot dataset
format::

    TensorDict(
        observation: TensorDict(
            image: {<camera>: uint8/float [*B, T, C, H, W]},  # or a single tensor
            state: float [*B, T, state_dim],                  # proprioception
        ),
        language_instruction: NonTensorData | Text,           # raw or tokenized (per-traj)
        action: float [*B, T, action_dim],                    # raw, per-step
        vla_action: VLAAction(
            chunk: float [*B, T, chunk, action_dim],          # built for training
            tokens: long [*B, T, chunk, action_dim],          # tokenized actions
        ),
        action_is_pad: bool [*B, T, chunk],                   # chunk validity mask
        next: TensorDict(...),                                 # TED layout
    )

Nothing here is mandatory: every VLA component exposes its keys via
``set_keys`` / constructor arguments and these constants are merely the shared
defaults.
"""
from __future__ import annotations

import torch
from tensordict import TensorDictBase
from tensordict.utils import is_non_tensor, NestedKey

__all__ = [
    "OBSERVATION_KEY",
    "IMAGE_KEY",
    "STATE_KEY",
    "INSTRUCTION_KEY",
    "ACTION_KEY",
    "VLA_ACTION_KEY",
    "ACTION_CHUNK_KEY",
    "ACTION_IS_PAD_KEY",
    "ACTION_TOKENS_KEY",
    "validate_vla_tensordict",
]

# -- Observation keys --------------------------------------------------------
#: Root of the observation sub-tensordict.
OBSERVATION_KEY: NestedKey = "observation"
#: Camera image(s). A single tensor, or a sub-tensordict keyed by camera name.
IMAGE_KEY: NestedKey = ("observation", "image")
#: Proprioceptive robot state (joint angles, end-effector pose, ...).
STATE_KEY: NestedKey = ("observation", "state")
#: Natural-language task instruction (raw text or tokenized). Stored at the
#: tensordict root (a per-trajectory field) to match the layout of
#: :class:`~torchrl.data.datasets.OpenXExperienceReplay`.
INSTRUCTION_KEY: NestedKey = "language_instruction"

# -- Action keys -------------------------------------------------------------
#: Raw per-step continuous action ``[*B, T, action_dim]`` (dataset native).
ACTION_KEY: NestedKey = "action"
#: Structured action output container.
VLA_ACTION_KEY: NestedKey = "vla_action"
#: Continuous action chunk ``[*B, T, chunk, action_dim]`` (training target).
ACTION_CHUNK_KEY: NestedKey = (VLA_ACTION_KEY, "chunk")
#: Boolean mask ``[*B, T, chunk]`` marking valid (non-padded) chunk steps.
ACTION_IS_PAD_KEY: NestedKey = "action_is_pad"
#: Discrete action tokens ``[*B, T, chunk, action_dim]`` or ``[*B, T, L]``.
ACTION_TOKENS_KEY: NestedKey = (VLA_ACTION_KEY, "tokens")


def validate_vla_tensordict(
    tensordict: TensorDictBase,
    *,
    instruction_key: NestedKey = INSTRUCTION_KEY,
    action_key: NestedKey = ACTION_KEY,
    image_key: NestedKey = IMAGE_KEY,
    state_key: NestedKey = STATE_KEY,
    require_instruction: bool = True,
    require_action: bool = True,
    require_perception: bool = True,
    check_finite: bool = True,
    raise_on_error: bool = True,
) -> list[str]:
    """Validate that a tensordict follows the canonical VLA schema.

    The check is intentionally permissive: it verifies the presence of the
    keys a VLA pipeline relies on and that action tensors are finite, without
    constraining shapes beyond what is necessary.

    Args:
        tensordict (TensorDictBase): the tensordict to validate.

    Keyword Args:
        instruction_key (NestedKey): language-instruction key.
            Defaults to ``"language_instruction"``.
        action_key (NestedKey): action key. Defaults to ``"action"``.
        image_key (NestedKey): image key.
            Defaults to ``("observation", "image")``.
        state_key (NestedKey): proprioceptive-state key.
            Defaults to ``("observation", "state")``.
        require_instruction (bool): if ``True``, a missing instruction is an
            error. Defaults to ``True``.
        require_action (bool): if ``True``, a missing action is an error.
            Defaults to ``True``.
        require_perception (bool): if ``True``, at least one of the image or
            state keys must be present. Defaults to ``True``.
        check_finite (bool): if ``True``, float action tensors must be finite.
            Defaults to ``True``.
        raise_on_error (bool): if ``True`` (default), raise a ``ValueError``
            when any issue is found; otherwise return the list of issues.

    Returns:
        a list of human-readable issue strings (empty if the tensordict is
        valid). When ``raise_on_error`` is ``True`` a non-empty list raises a
        ``ValueError`` instead of being returned.

    Examples:
        >>> import torch
        >>> from tensordict import NonTensorData, TensorDict
        >>> from torchrl.data.vla import validate_vla_tensordict
        >>> td = TensorDict(
        ...     {
        ...         "observation": {
        ...             "image": torch.zeros(2, 3, 8, 8, dtype=torch.uint8),
        ...         },
        ...         "language_instruction": NonTensorData("pick the cube"),
        ...         "action": torch.zeros(2, 7),
        ...     },
        ...     batch_size=[2],
        ... )
        >>> validate_vla_tensordict(td)
        []
    """
    issues: list[str] = []
    _missing = object()

    def _get(key: NestedKey):
        # ``get`` with a sentinel default resolves nested keys and surfaces
        # NonTensorData uniformly, unlike ``keys(leaves_only=True)``. A nested
        # key that descends into a leaf tensor raises ValueError -- treat it as
        # missing rather than letting the traceback escape.
        try:
            return tensordict.get(key, _missing)
        except (KeyError, TypeError, ValueError):
            return _missing

    instruction = _get(instruction_key)
    if require_instruction and instruction is _missing:
        issues.append(f"missing language instruction at key {instruction_key!r}")
    elif instruction is not _missing and is_non_tensor(instruction):
        data = getattr(instruction, "data", None)
        if isinstance(data, str) and not data.strip():
            issues.append(f"empty language instruction at key {instruction_key!r}")

    if (
        require_perception
        and _get(image_key) is _missing
        and _get(state_key) is _missing
    ):
        issues.append(
            f"no perception found: expected an image at {image_key!r} or state "
            f"at {state_key!r}"
        )

    action = _get(action_key)
    if require_action and action is _missing:
        issues.append(f"missing action at key {action_key!r}")
    elif (
        check_finite
        and isinstance(action, torch.Tensor)
        and action.is_floating_point()
        and not torch.isfinite(action).all()
    ):
        issues.append(f"non-finite values in action at key {action_key!r}")

    if issues and raise_on_error:
        raise ValueError(
            "Tensordict does not follow the canonical VLA schema:\n  - "
            + "\n  - ".join(issues)
        )
    return issues
