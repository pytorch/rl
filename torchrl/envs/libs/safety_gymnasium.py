# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
from types import ModuleType

import torch

from torchrl.data.tensor_specs import Unbounded
from torchrl.envs.libs.gym import GymEnv, GymWrapper, set_gym_backend
from torchrl.envs.utils import _classproperty

_has_safety_gymnasium = importlib.util.find_spec("safety_gymnasium") is not None


def _list_safety_gymnasium_envs() -> list[str]:
    """Discover task ids exposed by safety-gymnasium.

    safety-gymnasium registers many id variants (``*Gymnasium``,
    ``*Vision*``, ``*Debug``, ``*FadingEasy*``, ...). We surface the
    canonical 6-tuple-step ids and skip the ``Gymnasium`` variants because
    those return the standard 5-tuple and would not match this wrapper's
    ``_output_transform``.
    """
    if not _has_safety_gymnasium:
        return []
    import gymnasium
    import safety_gymnasium  # noqa: F401  -- import side-effect: register envs

    return sorted(
        env_id
        for env_id in gymnasium.envs.registry
        if env_id.startswith("Safety") and "Gymnasium" not in env_id
    )


class _SafetyGymCostMixin:
    """Expose safety-gymnasium's per-step ``cost`` signal as a top-level observation key.

    safety-gymnasium's ``step`` returns a 6-tuple
    ``(obs, reward, cost, terminated, truncated, info)``. We collapse the
    extra ``cost`` element into a stashed attribute, then write it onto
    the step/reset tensordict so it travels with the observation rather
    than through the info-dict-reader machinery.
    """

    def _post_init_cost(self) -> None:
        self.observation_spec["cost"] = Unbounded(shape=(), dtype=torch.float64)
        self._last_cost = torch.zeros((), dtype=torch.float64)

    def _output_transform(self, step_outputs_tuple):
        observations, reward, cost, terminated, truncated, info = step_outputs_tuple
        self._last_cost = torch.as_tensor(cost, dtype=torch.float64)
        return (
            observations,
            reward,
            terminated,
            truncated,
            terminated | truncated,
            info,
        )

    def _step(self, tensordict):
        out = super()._step(tensordict)
        out.set("cost", self._last_cost)
        return out

    def _reset(self, tensordict=None, **kwargs):
        out = super()._reset(tensordict, **kwargs)
        out.set("cost", torch.zeros_like(self._last_cost))
        return out


class SafetyGymnasiumWrapper(_SafetyGymCostMixin, GymWrapper):
    """Safety-Gymnasium environment wrapper.

    Safety-Gymnasium (https://github.com/PKU-Alignment/safety-gymnasium) is
    the actively-maintained successor to OpenAI's Safety-Gym. It provides
    constrained-RL benchmarks where each step emits a parallel ``cost``
    signal alongside the standard reward, allowing agents to optimize
    reward subject to a safety budget.

    The underlying ``step`` API returns a 6-tuple. This wrapper folds
    ``cost`` into the output tensordict as a top-level key alongside
    ``reward``.

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
        self._post_init_cost()

    @_classproperty
    def available_envs(cls):
        return _list_safety_gymnasium_envs()


class SafetyGymnasiumEnv(_SafetyGymCostMixin, GymEnv):
    """Safety-Gymnasium environment built from an env id.

    See :class:`SafetyGymnasiumWrapper` for behavior details. The
    constructor builds the environment via
    ``safety_gymnasium.make(env_name)`` and applies the same
    cost-extraction pipeline.

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

    @_classproperty
    def available_envs(cls):
        return _list_safety_gymnasium_envs()

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
        self._post_init_cost()
