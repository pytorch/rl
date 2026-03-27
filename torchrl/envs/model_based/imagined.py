# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Sequence

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.data.tensor_specs import Composite, Unbounded
from torchrl.envs.common import EnvBase
from torchrl.envs.model_based import ModelBasedEnvBase


class ImaginedEnv(ModelBasedEnvBase):
    """Imagination environment for model-based policy search.

    Wraps a learned world model (e.g. a Gaussian Process) as a standard
    TorchRL environment so that imagined rollouts can be collected with
    :meth:`~torchrl.envs.EnvBase.rollout`. Observations carry both mean
    and covariance (under keys ``("observation", "mean")`` and
    ``("observation", "var")``) to support uncertainty-aware moment-matching
    controllers.

    The environment never terminates on its own -- rollout length is
    controlled solely by the ``max_steps`` argument of
    :meth:`~torchrl.envs.EnvBase.rollout`. The ``done`` and ``terminated``
    flags are always ``False``.

    Args:
        world_model_module (TensorDictModule): A :class:`~tensordict.nn.TensorDictModule`
            that takes ``"action"`` and ``"observation"`` entries and produces
            ``("next_observation", "mean")`` and ``("next_observation", "var")``.
        base_env (EnvBase): The real environment whose specs (observation, action,
            reward, done) are copied into this imagined environment.
        batch_size (int, Sequence[int], torch.Size, optional): Override batch size.
            If ``None``, inferred from ``base_env`` (with a minimum of ``[1]``).
        next_observation_key (str or tuple of str, optional): The key where the world
            model writes the predicted next observation. Defaults to ``("next", "observation")``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.envs.model_based import ImaginedEnv, ModelBasedEnvBase
        >>> from torchrl.data import Composite, Unbounded
        >>> base_env = GymEnv("Pendulum-v1")
        >>> obs_dim = base_env.observation_spec["observation"].shape[-1]
        >>> # A toy world model that returns zero-mean, identity covariance
        >>> class DummyWorldModel(torch.nn.Module):
        ...     def __init__(self, obs_dim):
        ...         super().__init__()
        ...         self.obs_dim = obs_dim
        ...     def forward(self, action, observation):
        ...         # Assuming observation comes in as a dict with a "mean" key
        ...         mean = observation.get("mean", observation)
        ...         var = torch.eye(self.obs_dim).expand(*mean.shape[:-1], -1, -1)
        ...         return mean, var
        >>> wm = TensorDictModule(
        ...     DummyWorldModel(obs_dim),
        ...     in_keys=["action", "observation"],
        ...     out_keys=[("next", "observation", "mean"), ("next", "observation", "var")],
        ... )
        >>> imagined_env = ImaginedEnv(wm, base_env, next_observation_key=("next", "observation"))
        >>> # Collect an imagined rollout
        >>> rollout = imagined_env.rollout(max_steps=5, policy=RandomPolicy(imagined_env.action_spec))
    """

    def __init__(
        self,
        world_model_module: TensorDictModule,
        base_env: EnvBase,
        batch_size: int | torch.Size | Sequence[int] | None = None,
        next_observation_key: str | tuple[str, ...] = ("next", "observation"),
        **kwargs,
    ) -> None:
        self.next_observation_key = next_observation_key

        if batch_size is not None:
            batch_size = (
                torch.Size(batch_size)
                if not isinstance(batch_size, torch.Size)
                else batch_size
            )
        elif len(base_env.batch_size) == 0:
            batch_size = torch.Size([1])
        else:
            batch_size = base_env.batch_size

        super().__init__(
            world_model_module,
            device=base_env.device,
            batch_size=batch_size,
            allow_done_after_reset=True,
            **kwargs,
        )

        self._base_observation_spec = base_env.observation_spec.expand(
            self.batch_size
        ).clone()["observation"]
        obs_dim = self._base_observation_spec.shape[-1]
        self.observation_spec = (
            Composite(
                observation=Composite(
                    mean=Unbounded(
                        shape=(obs_dim,),
                        device=self.device,
                        dtype=self._base_observation_spec.dtype,
                    ),
                    var=Unbounded(
                        shape=(obs_dim, obs_dim),
                        device=self.device,
                        dtype=self._base_observation_spec.dtype,
                    ),
                ),
                device=self.device,
            )
            .expand(self.batch_size)
            .clone()
        )
        self.action_spec = base_env.action_spec.expand(self.batch_size).clone()
        self.reward_spec = base_env.reward_spec.expand(self.batch_size).clone()
        self.done_spec = base_env.done_spec.expand(self.batch_size).clone()

    def any_done(self, tensordict) -> bool:
        """Returns False -- imagination rollouts never terminate.

        Overridden to avoid CUDA sync from ``done.any()`` in the parent class.
        """
        return False

    def maybe_reset(self, tensordict):
        """No-op -- imagination rollouts do not need partial resets.

        Overridden to avoid CUDA sync from done checks in the parent class.
        """
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = self.world_model(tensordict.clone(recurse=False))

        reward = torch.zeros(*tensordict.shape, 1, device=self.device)
        done = torch.zeros(*tensordict.shape, 1, dtype=torch.bool, device=self.device)
        out = TensorDict(
            {
                "observation": tensordict.get(self.next_observation_key),
                "reward": reward,
                "done": done,
                "terminated": done.clone(),
            },
            tensordict.shape,
        )
        return out

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        if tensordict is None:
            tensordict = TensorDict({}, batch_size=self.batch_size, device=self.device)

        if (
            tensordict.get(("observation", "var"), None) is not None
            and tensordict.get(("observation", "mean"), None) is not None
        ):
            return tensordict.copy()

        obs = tensordict.get("observation", None)
        if obs is None:
            obs = self._base_observation_spec.rand()
        if obs.ndim == 1:
            obs = obs.expand(self.batch_size[0], -1)

        obs = obs.to(self.device)
        B, D = obs.shape

        out = TensorDict(
            {
                ("observation", "mean"): obs,
                ("observation", "var"): torch.zeros(
                    B, D, D, dtype=obs.dtype, device=self.device
                ),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

        out.set("done", torch.zeros(B, 1, dtype=torch.bool, device=self.device))
        out.set(
            "terminated",
            torch.zeros(B, 1, dtype=torch.bool, device=self.device),
        )

        return out
