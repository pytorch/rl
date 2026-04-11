# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from tensordict import TensorDictBase
from torchrl.data.tensor_specs import Composite, Unbounded
from torchrl.envs.transforms.transforms import Transform


class MeanActionSelector(Transform):
    """Bridges Gaussian belief-space policies with standard environments.

    Gaussian policies used in moment-matching model-based RL (e.g. PILCO) operate
    on state *beliefs* -- ``(mean, covariance)`` pairs -- and produce
    action distributions with ``("action", "mean")``, ``("action", "var")``, etc.
    This transform adapts a standard environment so that such a policy can be
    used directly with :meth:`~torchrl.envs.EnvBase.rollout`:

    * **Forward** (env output -> policy input): wraps the flat ``"observation"``
      tensor into ``("observation", "mean")`` with a zero-covariance
      ``("observation", "var")``, representing a deterministic state belief.
    * **Inverse** (policy output -> env input): extracts ``("action", "mean")``
      from the policy output and writes it as the flat ``"action"`` for the
      base environment step.

    Args:
        observation_key (str, optional): The observation key to read from the
            base environment. Defaults to ``"observation"``.
        action_key (str, optional): The action key expected by the base
            environment. Defaults to ``"action"``.

    Examples:
        >>> import torch
        >>> from torchrl.envs import GymEnv, TransformedEnv
        >>> from torchrl.envs.transforms import MeanActionSelector
        >>> base_env = GymEnv("Pendulum-v1")
        >>> env = TransformedEnv(base_env, MeanActionSelector())
        >>> td = env.reset()
        >>> # The policy now sees ("observation", "mean") and ("observation", "var")
        >>> print(td["observation", "mean"].shape)
        >>> print(td["observation", "var"].shape)
    """

    def __init__(
        self,
        observation_key: str = "observation",
        action_key: str = "action",
    ) -> None:
        super().__init__(
            in_keys=[observation_key],
            out_keys=[(observation_key, "mean"), (observation_key, "var")],
            in_keys_inv=[action_key],
            out_keys_inv=[(action_key, "mean")],
        )
        self._observation_key = observation_key
        self._action_key = action_key

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs = tensordict.get(self._observation_key)

        is_nested = isinstance(obs, TensorDictBase)
        if is_nested:
            return tensordict

        batch_shape = obs.shape[:-1]
        D = obs.shape[-1]
        device = obs.device
        dtype = obs.dtype

        tensordict.pop(self._observation_key)

        tensordict.set(
            (self._observation_key, "mean"),
            obs,
        )
        tensordict.set(
            (self._observation_key, "var"),
            torch.zeros(*batch_shape, D, D, device=device, dtype=dtype),
        )

        return tensordict

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        action_mean = tensordict.get((self._action_key, "mean"), None)
        if action_mean is not None:
            tensordict.set(self._action_key, action_mean)
        return tensordict

    def transform_observation_spec(self, observation_spec):
        obs_spec = observation_spec[self._observation_key]
        D = obs_spec.shape[-1]
        observation_spec[self._observation_key] = Composite(
            mean=obs_spec.clone(),
            var=Unbounded(shape=(*obs_spec.shape, D), dtype=obs_spec.dtype),
            shape=obs_spec.shape,
        )
        return observation_spec

    def _reset(self, tensordict, tensordict_reset):
        return self._call(tensordict_reset)
