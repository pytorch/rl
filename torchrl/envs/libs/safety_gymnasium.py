# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import warnings
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

    def _init_cost(self, cost_trailing_dim: bool | None) -> None:
        """Register the ``cost`` spec, warning if its shape was left to the default."""
        if cost_trailing_dim is None:
            warnings.warn(
                "From torchrl v0.17, the safety-gymnasium `cost` entry will have a "
                "trailing dimension of size 1, like `reward`. Pass "
                "`cost_trailing_dim=True` to use the new shape now, or "
                "`cost_trailing_dim=False` to keep the current shape and silence "
                "this warning.",
                FutureWarning,
                # _init_cost <- __init__ <- two metaclass __call__ <- caller.
                stacklevel=5,
            )
            cost_trailing_dim = False
        if cost_trailing_dim:
            cost_shape = torch.Size([*self.batch_size, 1])
        else:
            cost_shape = self.batch_size
        self.observation_spec["cost"] = Unbounded(
            shape=cost_shape, dtype=torch.float64, device=self.device
        )
        self._step_cost = torch.zeros(
            cost_shape, dtype=torch.float64, device=self.device
        )

    def _output_transform(self, step_outputs_tuple):
        observations, reward, cost, terminated, truncated, info = step_outputs_tuple
        # Called once per skipped frame, so the cost is summed like the reward.
        cost = torch.as_tensor(
            cost, dtype=torch.float64, device=self._step_cost.device
        ).reshape(self._step_cost.shape)
        self._step_cost = self._step_cost + cost
        return (
            observations,
            reward,
            terminated,
            truncated,
            terminated | truncated,
            info,
        )

    def _step(self, tensordict):
        self._step_cost = torch.zeros_like(self._step_cost)
        out = super()._step(tensordict)
        if ("final", "cost") in out.keys(True):
            # Vector env: VecGymEnvTransform copies ("final", ...) into "next"
            # for the sub-envs that just finished and keeps what is in "next"
            # as their auto-reset observation. So the step's cost goes in
            # "final", and the reset observation gets a zero cost.
            done = out.get("done").reshape(self._step_cost.shape)
            out.set(("final", "cost"), self._step_cost)
            out.set("cost", self._step_cost.masked_fill(done, 0.0))
        else:
            out.set("cost", self._step_cost)
        return out

    def _reset(self, tensordict=None, **kwargs):
        out = super()._reset(tensordict, **kwargs)
        out.set("cost", torch.zeros_like(self._step_cost))
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
    ``reward``. With ``frame_skip > 1``, ``cost`` is summed over the skipped
    frames, like ``reward``.

    Args:
        env (safety_gymnasium.Env): the environment to wrap. A vector env from
            ``safety_gymnasium.vector.make`` gives one cost per sub-env.

    Keyword Args:
        cost_trailing_dim (bool, optional): if ``True``, ``cost`` has the same
            shape as ``reward``, ``(*batch_size, 1)``. If ``False``, it has
            shape ``batch_size``. Defaults to ``None``, which behaves like
            ``False`` and emits a :class:`FutureWarning`: the default changes
            to ``True`` in torchrl v0.17.

    Examples:
        >>> import safety_gymnasium  # doctest: +SKIP
        >>> from torchrl.envs.libs.safety_gymnasium import SafetyGymnasiumWrapper
        >>> base = safety_gymnasium.vector.make(
        ...     "SafetyPointGoal1-v0", num_envs=2, asynchronous=False
        ... )  # doctest: +SKIP
        >>> env = SafetyGymnasiumWrapper(base, cost_trailing_dim=True)  # doctest: +SKIP
        >>> td = env.rollout(3)  # doctest: +SKIP
        >>> td["next", "cost"].shape  # doctest: +SKIP
        torch.Size([2, 3, 1])

    """

    git_url = "https://github.com/PKU-Alignment/safety-gymnasium"
    libname = "safety-gymnasium"

    _make_specs = set_gym_backend("gymnasium")(GymEnv._make_specs)

    def __init__(self, env=None, *, cost_trailing_dim: bool | None = None, **kwargs):
        super().__init__(env=env, **kwargs)
        self._init_cost(cost_trailing_dim)

    @_classproperty
    def available_envs(cls):
        return _list_safety_gymnasium_envs()


class SafetyGymnasiumEnv(_SafetyGymCostMixin, GymEnv):
    """Safety-Gymnasium environment built from an env id.

    See :class:`SafetyGymnasiumWrapper` for behavior details. The
    constructor builds the environment via
    ``safety_gymnasium.make(env_name)`` and applies the same
    cost-extraction pipeline. With ``num_envs``, the copies run in a
    ``safety_gymnasium.vector.SafetyAsyncVectorEnv``.

    Args:
        env_name (str): the safety-gymnasium task id, e.g.
            ``"SafetyPointGoal1-v0"``.

    Keyword Args:
        cost_trailing_dim (bool, optional): if ``True``, ``cost`` has the same
            shape as ``reward``, ``(*batch_size, 1)``. If ``False``, it has
            shape ``batch_size``. Defaults to ``None``, which behaves like
            ``False`` and emits a :class:`FutureWarning`: the default changes
            to ``True`` in torchrl v0.17.

    Examples:
        >>> from torchrl.envs.libs.safety_gymnasium import SafetyGymnasiumEnv
        >>> env = SafetyGymnasiumEnv(
        ...     env_name="SafetyPointGoal1-v0", num_envs=2, cost_trailing_dim=True
        ... )  # doctest: +SKIP
        >>> td = env.rollout(3)  # doctest: +SKIP
        >>> td["next", "cost"].shape  # doctest: +SKIP
        torch.Size([2, 3, 1])

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

    def __init__(
        self, env_name=None, *, cost_trailing_dim: bool | None = None, **kwargs
    ):
        super().__init__(env_name=env_name, **kwargs)
        self._init_cost(cost_trailing_dim)

    def _async_env(self, *args, **kwargs):
        """Build the ``num_envs`` vector env with safety-gymnasium's class, which keeps the 6-tuple step."""
        return self.lib.vector.SafetyAsyncVectorEnv(*args, **kwargs)
