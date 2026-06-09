# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Metadata describing a robot / VLA dataset."""
from __future__ import annotations

import json
from typing import get_args, Literal

import torch
from tensordict import TensorClass
from tensordict.utils import NestedKey

from torchrl.data.tensor_specs import Bounded, TensorSpec, Unbounded
from torchrl.data.vla.schema import ACTION_KEY, INSTRUCTION_KEY, STATE_KEY

__all__ = ["RobotDatasetMetadata", "ActionSpace", "GripperMode"]

#: Coarse description of what an action vector represents.
ActionSpace = Literal[
    "joint_delta",
    "joint_position",
    "eef_delta",
    "eef_pose",
    "velocity",
    "mixed",
    "unknown",
]
#: How the gripper dimension(s) of an action are encoded.
GripperMode = Literal["continuous", "binary", "none", "unknown"]

_TENSOR_FIELDS = ("action_mean", "action_std", "action_low", "action_high")
_SINGLE_KEY_FIELDS = ("state_key", "instruction_key", "action_key")
_FIELDS = (
    "dataset_id",
    "embodiment_id",
    "action_dim",
    "action_names",
    "action_space",
    "gripper_mode",
    "control_frequency_hz",
    "camera_keys",
    "state_key",
    "instruction_key",
    "action_key",
    "action_mean",
    "action_std",
    "action_low",
    "action_high",
    "license",
    "source_url",
)


class RobotDatasetMetadata(TensorClass["nocast"]):
    """Metadata describing a robot dataset / embodiment for VLA workflows.

    This lightweight container records the information needed to normalize
    actions, build specs, prompt for embodiment, and sample mixed-embodiment
    batches. It travels alongside a dataset or a batch and is consumed by the
    VLA transforms, losses and env adapters.

    It is a :class:`~tensordict.TensorClass` so that the per-dimension action
    statistics (``action_mean`` / ``action_std`` / ``action_low`` /
    ``action_high``) are first-class tensors: the whole record moves with
    :meth:`~tensordict.TensorClassBase.to`, compares with ``==`` (use ``.all()``
    for a scalar), and serializes natively. The non-tensor fields (ids, action
    convention, keys, ...) are stored as non-tensor data.

    Args:
        dataset_id (str): identifier of the dataset (e.g. ``"bridge"``).

    Keyword Args:
        embodiment_id (str, optional): identifier of the robot embodiment.
        action_dim (int, optional): dimensionality of a single action.
        action_names (tuple[str, ...], optional): per-dimension action names.
        action_space (str): one of ``"joint_delta"``, ``"joint_position"``,
            ``"eef_delta"``, ``"eef_pose"``, ``"velocity"``, ``"mixed"`` or
            ``"unknown"`` (default).
        gripper_mode (str): one of ``"continuous"``, ``"binary"``, ``"none"``
            or ``"unknown"`` (default).
        control_frequency_hz (float, optional): control rate of the data.
        camera_keys (tuple[NestedKey, ...]): keys of the available cameras.
        state_key (NestedKey, optional): proprioceptive-state key.
        instruction_key (NestedKey, optional): language-instruction key.
        action_key (NestedKey): action key. Defaults to ``"action"``.
        action_mean (torch.Tensor, optional): per-dimension action mean.
        action_std (torch.Tensor, optional): per-dimension action std.
        action_low (torch.Tensor, optional): per-dimension action lower bound.
        action_high (torch.Tensor, optional): per-dimension action upper bound.
        license (str, optional): dataset license.
        source_url (str, optional): dataset source URL.

    Examples:
        >>> import torch
        >>> from torchrl.data.vla import RobotDatasetMetadata
        >>> meta = RobotDatasetMetadata(
        ...     "bridge",
        ...     action_dim=7,
        ...     action_space="eef_delta",
        ...     gripper_mode="binary",
        ...     action_mean=torch.zeros(7),
        ...     action_std=torch.ones(7),
        ... )
        >>> meta.make_action_spec().shape
        torch.Size([7])
        >>> bool((RobotDatasetMetadata.from_json(meta.to_json()) == meta).all())
        True

    .. note:: The ``action_mean``/``action_std`` (or ``action_low``/
        ``action_high``) statistics recorded here are intended to be consumed
        by the VLA action-normalization transform.
    """

    dataset_id: str
    embodiment_id: str | None = None
    action_dim: int | None = None
    action_names: tuple[str, ...] | None = None
    action_space: ActionSpace = "unknown"
    gripper_mode: GripperMode = "unknown"
    control_frequency_hz: float | None = None
    camera_keys: tuple[NestedKey, ...] = ()
    state_key: NestedKey | None = STATE_KEY
    instruction_key: NestedKey | None = INSTRUCTION_KEY
    action_key: NestedKey = ACTION_KEY
    action_mean: torch.Tensor | None = None
    action_std: torch.Tensor | None = None
    action_low: torch.Tensor | None = None
    action_high: torch.Tensor | None = None
    license: str | None = None
    source_url: str | None = None

    def __post_init__(self) -> None:
        # __post_init__ runs on every TensorClass reconstruction (``==``, ``.to``,
        # ``clone``, indexing, ...), where the fields may not hold their
        # user-input types (e.g. ``==`` turns the non-tensor fields into bools).
        # Each validation/coercion therefore guards on the input type so it only
        # fires on a genuine user construction.
        if isinstance(self.action_space, str) and self.action_space not in get_args(
            ActionSpace
        ):
            raise ValueError(
                f"action_space must be one of {get_args(ActionSpace)}, "
                f"got {self.action_space!r}."
            )
        if isinstance(self.gripper_mode, str) and self.gripper_mode not in get_args(
            GripperMode
        ):
            raise ValueError(
                f"gripper_mode must be one of {get_args(GripperMode)}, "
                f"got {self.gripper_mode!r}."
            )
        if (
            self.action_dim is not None
            and not isinstance(self.action_dim, torch.Tensor)
            and (not isinstance(self.action_dim, int) or self.action_dim <= 0)
        ):
            raise ValueError(
                f"action_dim must be a positive int, got {self.action_dim!r}."
            )
        # Coerce keys (lists deserialize from JSON as lists) back to tuples.
        if isinstance(self.camera_keys, (list, tuple)):
            self.camera_keys = tuple(
                tuple(key) if isinstance(key, list) else key for key in self.camera_keys
            )
        if isinstance(self.action_names, list):
            self.action_names = tuple(self.action_names)
        for attr in _SINGLE_KEY_FIELDS:
            value = getattr(self, attr)
            if isinstance(value, list):
                setattr(self, attr, tuple(value))
        # Coerce statistics to float tensors.
        for attr in _TENSOR_FIELDS:
            value = getattr(self, attr)
            if value is not None and not isinstance(value, torch.Tensor):
                setattr(self, attr, torch.as_tensor(value, dtype=torch.float32))

    def make_action_spec(
        self, *, shape_prefix: tuple[int, ...] = (), device=None
    ) -> TensorSpec:
        """Build an action :class:`~torchrl.data.TensorSpec` from the metadata.

        Returns a :class:`~torchrl.data.Bounded` spec when both
        ``action_low`` and ``action_high`` are set, otherwise an
        :class:`~torchrl.data.Unbounded` spec.

        Keyword Args:
            shape_prefix (tuple[int, ...]): leading batch/time dimensions to
                prepend to ``(action_dim,)``. Defaults to ``()``.
            device: device of the spec. Defaults to ``None``.
        """
        if self.action_dim is None:
            raise ValueError("action_dim must be set to build an action spec.")
        shape = (*shape_prefix, self.action_dim)
        if self.action_low is not None and self.action_high is not None:
            # low/high are per-dimension (action_dim,); broadcast them to the
            # full ``(*shape_prefix, action_dim)`` shape Bounded expects.
            low = self.action_low.expand(shape).clone()
            high = self.action_high.expand(shape).clone()
            return Bounded(
                low=low, high=high, shape=shape, device=device, dtype=torch.float32
            )
        return Unbounded(shape=shape, device=device, dtype=torch.float32)

    def to_json(self, path: str | None = None) -> str:
        """Serialize the metadata to a JSON string (and optionally a file).

        Tensors and tuples are converted to lists so the output is plain JSON;
        :meth:`from_json` reconstructs the metadata.
        """
        out: dict = {}
        for key, value in self.to_dict().items():
            if isinstance(value, torch.Tensor):
                out[key] = value.detach().cpu().tolist()
            elif isinstance(value, tuple):
                out[key] = [list(v) if isinstance(v, tuple) else v for v in value]
            else:
                out[key] = value
        payload = json.dumps(out, indent=2)
        if path is not None:
            with open(path, "w") as f:
                f.write(payload)
        return payload

    @classmethod
    def from_json(cls, source: str) -> RobotDatasetMetadata:
        """Load metadata from a JSON string or a path to a JSON file.

        Unknown keys are ignored so that metadata serialized by a newer
        version of the class remains loadable.
        """
        if source.lstrip().startswith("{"):
            data = json.loads(source)
        else:
            with open(source) as f:
                data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in _FIELDS})
