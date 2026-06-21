# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDictBase
from tensordict.utils import NestedKey
from torchrl.data.tensor_specs import Unbounded
from torchrl.envs.transforms.transforms import Transform


class RunningMeanStd(nn.Module):
    """Tracks running mean and variance using Welford's parallel algorithm.

    Buffers are registered so the statistics are included in ``state_dict()``
    and move correctly with ``.to(device)``.

    Args:
        shape (tuple): feature shape to track (e.g. ``(obs_dim,)`` or ``()`` for scalars).
        epsilon (float, optional): small initial count for numerical stability.
            Default: ``1e-4``.

    Examples:
        >>> rms = RunningMeanStd(shape=(4,))
        >>> rms.update(torch.randn(32, 4))
        >>> normed = rms.normalize(torch.randn(8, 4))
        >>> normed.shape
        torch.Size([8, 4])
    """

    def __init__(self, shape: tuple = (), epsilon: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float32))

    def update(self, x: torch.Tensor) -> None:
        """Update running statistics with a new batch.

        Args:
            x (torch.Tensor): batch of samples. All leading dimensions are
                treated as the batch dimension; trailing dimensions must match
                ``self.mean.shape``.
        """
        x = x.float()
        if self.mean.ndim == 0:
            x = x.reshape(-1)
            batch_count = x.shape[0]
            batch_mean = x.mean()
            batch_var = x.var(unbiased=False) if batch_count > 1 else x.new_zeros(())
        else:
            x = x.reshape(-1, *self.mean.shape)
            batch_count = x.shape[0]
            batch_mean = x.mean(0)
            batch_var = (
                x.var(0, unbiased=False)
                if batch_count > 1
                else torch.zeros_like(batch_mean)
            )

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean.copy_(self.mean + delta * (batch_count / tot_count))
        m2 = (
            self.var * self.count
            + batch_var * batch_count
            + delta.pow(2) * (self.count * batch_count / tot_count)
        )
        self.var.copy_(m2 / tot_count)
        self.count.copy_(tot_count)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` to approximately zero mean, unit variance."""
        return (x.float() - self.mean) / self.var.clamp_min(1e-8).sqrt()


class RNDTransform(Transform):
    """Random Network Distillation transform that computes an intrinsic reward.

    Implements the exploration bonus from:

        Burda et al., "Exploration by Random Network Distillation" (2018).
        https://arxiv.org/abs/1810.12894

    At every environment step the transform:

    1. Optionally normalizes the next observation with online running statistics
       and clips the result to ``[-obs_clip, obs_clip]`` sigma.
    2. Passes the (normalized) observation through both the frozen *target* and
       the trainable *predictor* networks.
    3. Writes the MSE prediction error as an intrinsic reward under ``out_keys[0]``.
    4. Optionally normalizes that reward by its running standard deviation.

    The predictor is **only** given gradient updates through :class:`RNDLoss`
    during training. The transform itself always runs under ``torch.no_grad()``.

    Running normalization statistics are lazily initialized on the first step so
    that the feature dimensionality does not need to be specified up-front. Pass
    ``normalize_obs=False`` to skip observation normalization (useful when the
    observation is already normalized by another transform).

    Args:
        target_network (torch.nn.Module): frozen random network providing fixed
            embeddings. Its parameters are frozen on construction.
        predictor_network (torch.nn.Module): trainable network that learns to
            predict target embeddings.
        in_keys (list of NestedKey, optional): tensordict keys to read
            observations from. Defaults to ``["observation"]``.
        out_keys (list of NestedKey, optional): tensordict keys to write the
            intrinsic reward to. Defaults to ``["intrinsic_reward"]``.
        normalize_obs (bool, optional): normalize observations with running
            mean/std before passing to the networks. Default: ``True``.
        normalize_reward (bool, optional): divide intrinsic reward by its
            running standard deviation. Default: ``True``.
        obs_clip (float, optional): clip normalized observations to
            ``[-obs_clip, obs_clip]``. Default: ``5.0``.
        reward_clip (float, optional): clip normalized intrinsic reward to
            ``[-reward_clip, reward_clip]``. Default: ``5.0``.

    Examples:
        >>> import torch.nn as nn
        >>> from torchrl.envs import GymEnv, TransformedEnv
        >>> from torchrl.envs.transforms import RNDTransform
        >>> target = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64))
        >>> predictor = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64))
        >>> env = TransformedEnv(GymEnv("CartPole-v1"), RNDTransform(target, predictor))
        >>> td = env.rollout(3)
        >>> td["next", "intrinsic_reward"].shape
        torch.Size([3, 1])
    """

    def __init__(
        self,
        target_network: nn.Module,
        predictor_network: nn.Module,
        in_keys: list[NestedKey] | None = None,
        out_keys: list[NestedKey] | None = None,
        normalize_obs: bool = True,
        normalize_reward: bool = True,
        obs_clip: float = 5.0,
        reward_clip: float = 5.0,
    ):
        if in_keys is None:
            in_keys = ["observation"]
        if out_keys is None:
            out_keys = ["intrinsic_reward"]
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.target_network = target_network
        self.predictor_network = predictor_network
        self.target_network.requires_grad_(False)
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        self.obs_clip = obs_clip
        self.reward_clip = reward_clip
        # Lazily initialized on first step; exposed as properties for sharing
        # with RNDLoss (see obs_rms / reward_rms properties below).
        self._obs_rms: RunningMeanStd | None = None
        self._reward_rms: RunningMeanStd | None = None

    # ------------------------------------------------------------------
    # Public properties so RNDLoss can share the same statistics objects.
    # ------------------------------------------------------------------

    @property
    def obs_rms(self) -> RunningMeanStd | None:
        """Running obs statistics, or ``None`` before the first step."""
        return self._obs_rms

    @property
    def reward_rms(self) -> RunningMeanStd | None:
        """Running intrinsic-reward statistics, or ``None`` before the first step."""
        return self._reward_rms

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_init_rms(self, obs: torch.Tensor) -> None:
        """Lazily create RunningMeanStd modules on the first observation.

        Assignment to self._obs_rms / self._reward_rms is enough:
        nn.Module.__setattr__ automatically registers Module instances in
        self._modules, so state_dict() and .to() pick them up without an
        explicit add_module() call.
        """
        if self.normalize_obs and self._obs_rms is None:
            self._obs_rms = RunningMeanStd(shape=obs.shape[-1:]).to(obs.device)
        if self.normalize_reward and self._reward_rms is None:
            self._reward_rms = RunningMeanStd(shape=()).to(obs.device)

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        obs = next_tensordict.get(self.in_keys[0]).float()
        self._maybe_init_rms(obs)

        if self._obs_rms is not None:
            if self.training:
                self._obs_rms.update(obs)
            obs_in = self._obs_rms.normalize(obs).clamp(-self.obs_clip, self.obs_clip)
        else:
            obs_in = obs

        with torch.no_grad():
            target_feat = self.target_network(obs_in)
            pred_feat = self.predictor_network(obs_in)

        intrinsic = (pred_feat - target_feat).pow(2).mean(dim=-1, keepdim=True)

        if self._reward_rms is not None:
            if self.training:
                self._reward_rms.update(intrinsic)
            intrinsic = (intrinsic / self._reward_rms.var.clamp_min(1e-8).sqrt()).clamp(
                -self.reward_clip, self.reward_clip
            )

        next_tensordict.set(self.out_keys[0], intrinsic)
        return next_tensordict

    def transform_reward_spec(self, reward_spec):
        for out_key in self.out_keys:
            reward_spec[out_key] = Unbounded(
                shape=reward_spec.shape,
                device=reward_spec.device,
                dtype=torch.float32,
            )
        return reward_spec
