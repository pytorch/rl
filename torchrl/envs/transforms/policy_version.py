# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import cast, Literal

import torch
from tensordict import NonTensorData, TensorDictBase
from torchrl.data.tensor_specs import Composite, NonTensor, Unbounded
from torchrl.envs.transforms.transforms import Transform


@dataclass
class VersionChange:
    """Records a single version change event."""

    timestamp: datetime
    old_version: str | int | None
    new_version: str | int


class PolicyVersion(Transform):
    """A transform that keeps track of the version of the policy.

    This transform is used to track policy versions during training, particularly in asynchronous
    settings where policy weights are updated periodically. It works with TorchRL collectors to
    ensure data collection and training remain in sync.

    The version can be either a UUID (string) or an integer counter. When passed to a collector as
    its ``track_policy_version`` argument, the version is automatically incremented each time the
    policy weights are updated.

    Args:
        version_type: The type of versioning to use. Can be either:
            - str or "uuid": Uses UUID4 for versions (good for distributed systems)
            - int or "int": Uses incrementing integers (good for debugging)

    Examples:
        >>> from torchrl.envs.transforms import PolicyVersion
        >>> policy_version = PolicyVersion(version_type="int")
        >>> policy_version.version
        0
        >>> policy_version.increment_version()
        >>> policy_version.version
        1
    """

    def __init__(self, version_type: type | Literal["uuid", "int"] = int):
        super().__init__()
        self.version_type = version_type
        self.version_history: list[VersionChange] = []  # Track version changes
        self._current_version: str | int | None = None
        self._increment_version(init=True)
        self.cal_on_reset = True

    @property
    def version(self) -> str | int:
        """The current version of the policy."""
        if self._current_version is None:
            raise RuntimeError("Version not initialized")
        return self._current_version

    @version.setter
    def version(self, value: str | int) -> None:
        self._current_version = value

    def increment_version(self) -> None:
        """Increment the version number.

        This is called automatically by collectors when policy weights are updated.
        Can also be called manually if needed.
        """
        self._increment_version()

    def _increment_version(self, init: bool = False) -> str | int:
        """Internal method to handle version incrementing with history tracking."""
        old_version = self._current_version
        if self.version_type in (str, "uuid"):
            self._increment_version_uuid(init)
        elif self.version_type in (int, "int"):
            self._increment_version_int(init)
        else:
            raise ValueError(f"Invalid version type: {self.version_type}")

        # Record the version change
        self.version_history.append(
            VersionChange(
                timestamp=datetime.now(),
                old_version=old_version,
                new_version=self.version,
            )
        )
        return self.version

    def _increment_version_uuid(self, init: bool = False) -> None:
        """Generate a new UUID version.

        Args:
            init: If True, this is the initial version creation.
        """
        self.version = str(uuid.uuid4())

    def _increment_version_int(self, init: bool = False) -> None:
        """Increment the integer version counter.

        Args:
            init: If True, initialize counter to 0, otherwise increment by 1.
        """
        if init:
            self.version = 0
        else:
            # Cast to int to ensure type safety
            current = cast(int, self.version)
            self.version = current + 1

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Reset the environment and update version in the new tensordict.

        Args:
            tensordict: The current tensordict
            tensordict_reset: The tensordict to reset to

        Returns:
            The reset tensordict with updated version
        """
        tensordict_reset = self._step(None, tensordict_reset)
        return tensordict_reset

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        """Add the current version to the tensordict.

        This method is called on each environment step to ensure the collected
        data is tagged with the correct policy version.

        Args:
            tensordict: The tensordict to update with version info

        Returns:
            The tensordict with added version information
        """
        if self.version_type in (str, "uuid"):
            version = NonTensorData(self.version).expand(next_tensordict.shape)
        elif self.version_type in (int, "int"):
            device = next_tensordict.device
            kwargs = {"dtype": torch.int64}
            if device is not None:
                kwargs["device"] = device
            version = torch.full(
                next_tensordict.shape,
                cast(int, self.version),
                **kwargs,
            )
        else:
            raise ValueError(f"Invalid version type: {self.version_type}")

        next_tensordict.set("policy_version", version)
        return next_tensordict

    def transform_observation_spec(self, spec: Composite) -> Composite:
        """Update the environment spec to include the version field.

        Args:
            spec: The environment spec to update

        Returns:
            Updated spec including the version field
        """
        if self.version_type in (str, "uuid"):
            spec["policy_version"] = NonTensor(
                example_data=uuid.uuid4(), shape=spec.shape, device=spec.device
            )
        elif self.version_type in (int, "int"):
            spec["policy_version"] = Unbounded(
                shape=spec.shape, dtype=torch.int64, device=spec.device
            )
        else:
            raise ValueError(f"Invalid version type: {self.version_type}")
        return spec
