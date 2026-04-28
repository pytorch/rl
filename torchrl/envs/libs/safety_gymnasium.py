# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
from types import ModuleType

import numpy as np
import torch

from torchrl.data.tensor_specs import Composite, Unbounded
from torchrl.envs.gym_like import default_info_dict_reader
from torchrl.envs.libs.gym import GymEnv, GymWrapper, set_gym_backend
from torchrl.envs.utils import _classproperty

_has_safety_gymnasium = importlib.util.find_spec("safety_gymnasium") is not None


def _make_cost_reader() -> default_info_dict_reader:
    cost_spec = Composite(
        cost=Unbounded(shape=(), dtype=torch.float64), shape=[]
    )
    return default_info_dict_reader(["cost"], spec=cost_spec)


class SafetyGymnasiumWrapper(GymWrapper):
    """Safety-Gymnasium environment wrapper.

    Safety-Gymnasium (https://github.com/PKU-Alignment/safety-gymnasium) is the
    actively-maintained successor to OpenAI's Safety-Gym. It provides
    constrained-RL benchmarks where each step emits a parallel ``cost`` signal
    alongside the standard reward, allowing agents to optimize reward subject
    to a safety budget.

    The underlying ``step`` API returns a 6-tuple
    ``(obs, reward, cost, terminated, truncated, info)``. This wrapper folds
    ``cost`` into the info dict so that the standard
    :class:`~torchrl.envs.libs.gym.GymWrapper` machinery can be reused, and
    registers an info-dict reader that exposes ``cost`` as a top-level key in
    the returned tensordict.

    Args:
        env (safety_gymnasium.Env): the environment to wrap.

    Examples:
        >>> import safety_gymnasium  # doctest: +SKIP
        >>> from torchrl.envs.libs.safety_gymnasium import SafetyGymnasiumWrapper
        >>> base = safety_gymnasium.make("SafetyPointGoal1-v0")  # doctest: +SKIP
        >>> env = SafetyGymnasiumWrapper(base)  # doctest: +SKIP
        >>> td = env.rollout(3)  # doctest: +SKIP
        >>> assert ("next", "cost") in td.keys(True)  # doctest: +SKIP

    """

    git_url = "https://github.com/PKU-Alignment/safety-gymnasium"
    libname = "safety-gymnasium"

    _make_specs = set_gym_backend("gymnasium")(GymEnv._make_specs)

    def __init__(self, env=None, **kwargs):
        super().__init__(env=env, **kwargs)
        self.set_info_dict_reader(_make_cost_reader())

    def _output_transform(self, step_outputs_tuple):
        observations, reward, cost, terminated, truncated, info = step_outputs_tuple
        info = dict(info) if info is not None else {}
        # The default info_dict_reader expects values with a `.dtype`
        # attribute. safety-gymnasium emits cost as a Python float, so we
        # promote it to a numpy scalar of fixed dtype.
        info["cost"] = np.asarray(cost, dtype=np.float64)
        return (
            observations,
            reward,
            terminated,
            truncated,
            terminated | truncated,
            info,
        )

    @_classproperty
    def available_envs(cls):
        if not _has_safety_gymnasium:
            return []
        # Curated list of canonical safety-gymnasium task ids. The library
        # registers more (different difficulty levels, vision variants, etc.);
        # this list mirrors the ones documented as primary benchmarks.
        return [
            # Point robot
            "SafetyPointGoal0-v0",
            "SafetyPointGoal1-v0",
            "SafetyPointGoal2-v0",
            "SafetyPointButton0-v0",
            "SafetyPointButton1-v0",
            "SafetyPointButton2-v0",
            "SafetyPointPush0-v0",
            "SafetyPointPush1-v0",
            "SafetyPointPush2-v0",
            "SafetyPointCircle0-v0",
            "SafetyPointCircle1-v0",
            "SafetyPointRace0-v0",
            "SafetyPointRace1-v0",
            "SafetyPointRace2-v0",
            # Car robot
            "SafetyCarGoal0-v0",
            "SafetyCarGoal1-v0",
            "SafetyCarGoal2-v0",
            "SafetyCarButton0-v0",
            "SafetyCarButton1-v0",
            "SafetyCarButton2-v0",
            "SafetyCarPush0-v0",
            "SafetyCarPush1-v0",
            "SafetyCarPush2-v0",
            "SafetyCarCircle0-v0",
            "SafetyCarCircle1-v0",
            "SafetyCarRace0-v0",
            "SafetyCarRace1-v0",
            "SafetyCarRace2-v0",
            # Mujoco velocity tasks
            "SafetyAntVelocity-v1",
            "SafetyHalfCheetahVelocity-v1",
            "SafetyHopperVelocity-v1",
            "SafetyHumanoidVelocity-v1",
            "SafetySwimmerVelocity-v1",
            "SafetyWalker2dVelocity-v1",
        ]


class SafetyGymnasiumEnv(GymEnv):
    """Safety-Gymnasium environment built from an env id.

    See :class:`SafetyGymnasiumWrapper` for behavior details. The constructor
    builds the environment via ``safety_gymnasium.make(env_name)`` and applies
    the same cost-extraction pipeline.

    Args:
        env_name (str): the safety-gymnasium task id, e.g.
            ``"SafetyPointGoal1-v0"``.

    Examples:
        >>> from torchrl.envs.libs.safety_gymnasium import SafetyGymnasiumEnv
        >>> env = SafetyGymnasiumEnv(env_name="SafetyPointGoal1-v0")  # doctest: +SKIP
        >>> td = env.rollout(3)  # doctest: +SKIP
        >>> assert ("next", "cost") in td.keys(True)  # doctest: +SKIP

    """

    git_url = "https://github.com/PKU-Alignment/safety-gymnasium"
    libname = "safety-gymnasium"

    available_envs = SafetyGymnasiumWrapper.available_envs

    @property
    def lib(self) -> ModuleType:
        if _has_safety_gymnasium:
            import safety_gymnasium

            return safety_gymnasium
        try:
            import safety_gymnasium  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "safety-gymnasium not found, install with "
                "`pip install safety-gymnasium`"
            ) from err

    _make_specs = set_gym_backend("gymnasium")(GymEnv._make_specs)

    def __init__(self, env_name=None, **kwargs):
        super().__init__(env_name=env_name, **kwargs)
        self.set_info_dict_reader(_make_cost_reader())

    def _output_transform(self, step_outputs_tuple):
        observations, reward, cost, terminated, truncated, info = step_outputs_tuple
        info = dict(info) if info is not None else {}
        # The default info_dict_reader expects values with a `.dtype`
        # attribute. safety-gymnasium emits cost as a Python float, so we
        # promote it to a numpy scalar of fixed dtype.
        info["cost"] = np.asarray(cost, dtype=np.float64)
        return (
            observations,
            reward,
            terminated,
            truncated,
            terminated | truncated,
            info,
        )
