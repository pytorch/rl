# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase

from torchrl.envs import GymWrapper, TransformedEnv

__all__ = ["MujocoStateReader"]


class MujocoStateReader:
    """Reads simulator state from TorchRL-native and Gym MuJoCo environments.

    The reader keeps simulator state separate from policy observations. It
    accepts environments exposing a ``get_state()`` method that returns a
    TensorDict or mapping with a ``"qpos"`` entry, as well as TorchRL
    :class:`~torchrl.envs.GymWrapper` instances around Gymnasium MuJoCo
    environments.

    Examples:
        >>> from types import SimpleNamespace
        >>> import numpy as np
        >>> from torchrl.render.backends import MujocoStateReader
        >>> env = SimpleNamespace(
        ...     data=SimpleNamespace(
        ...         qpos=np.array([0.0, 1.0]),
        ...         qvel=np.array([0.5, 0.0]),
        ...         time=0.25,
        ...     )
        ... )
        >>> state = MujocoStateReader().capture(env)
        >>> state["qpos"].tolist()
        [0.0, 1.0]
    """

    def supports(self, env: Any) -> bool:
        """Returns whether ``env`` exposes a readable MuJoCo state."""
        try:
            self.capture(env)
        except (KeyError, NotImplementedError, TypeError):
            return False
        return True

    def capture(self, env: Any) -> TensorDictBase:
        """Returns a detached snapshot of the environment's MuJoCo state.

        Args:
            env: TorchRL-native MuJoCo environment, Gym-backed MuJoCo
                environment, or object exposing MuJoCo ``data`` directly.

        Returns:
            A TensorDict containing ``qpos`` and any available ``qvel``,
            ``act``, ``ctrl``, ``mocap_pos``, ``mocap_quat``, and ``time``
            entries.

        Raises:
            TypeError: If the environment does not expose a supported state.
            KeyError: If the exposed state does not contain ``qpos``.
        """
        source = _unwrap_transformed_env(env)
        state = _capture_state_provider(source)
        if state is None:
            state = _capture_mujoco_data(source)
        if state is None:
            raise TypeError(
                f"Environment of type {type(source).__name__} does not expose "
                "MuJoCo state through get_state() or data."
            )
        if "qpos" not in state.keys():
            raise KeyError("MuJoCo state does not contain a 'qpos' entry.")
        return state


def _unwrap_transformed_env(env: Any) -> Any:
    while isinstance(env, TransformedEnv):
        env = env.base_env
    return env


def _capture_state_provider(env: Any) -> TensorDictBase | None:
    get_state = getattr(type(env), "get_state", None)
    if callable(get_state):
        try:
            state = get_state(env)
        except NotImplementedError:
            return None
    else:
        get_state = vars(env).get("get_state")
        if not callable(get_state):
            return None
        try:
            state = get_state()
        except NotImplementedError:
            return None
    if isinstance(state, TensorDictBase):
        return state.clone()
    if isinstance(state, Mapping):
        return TensorDict(
            dict(state), batch_size=getattr(env, "batch_size", [])
        ).clone()
    return None


def _capture_mujoco_data(env: Any) -> TensorDictBase | None:
    device = getattr(env, "device", None)
    if isinstance(env, GymWrapper):
        env = env.unwrapped
    data = vars(env).get("data")
    if data is None:
        return None
    state = {}
    for key in (
        "qpos",
        "qvel",
        "act",
        "ctrl",
        "mocap_pos",
        "mocap_quat",
        "time",
    ):
        value = getattr(data, key, None)
        if value is None:
            continue
        tensor = _copy_to_tensor(value, device=device)
        if key != "time" and tensor.numel() == 0:
            continue
        state[key] = tensor
    if not state:
        return None
    return TensorDict(state, batch_size=[], device=device)


def _copy_to_tensor(value: Any, *, device: torch.device | None) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().clone().to(device=device)
    if isinstance(value, np.ndarray):
        value = value.copy()
    else:
        copy = getattr(value, "copy", None)
        if callable(copy):
            value = copy()
    return torch.as_tensor(value, device=device)
