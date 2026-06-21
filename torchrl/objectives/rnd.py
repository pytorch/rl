# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase
from tensordict.utils import NestedKey

from torchrl.envs.transforms.rnd import RunningMeanStd
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import _reduce


class RNDLoss(LossModule):
    """Loss module for training the predictor network in Random Network Distillation.

    Presented in:

        Burda et al., "Exploration by Random Network Distillation" (2018).
        https://arxiv.org/abs/1810.12894

    Computes the MSE between the *predictor* and the frozen *target* network on
    next observations sampled from a replay buffer.  Call this loss alongside
    your main policy objective; its gradients update the predictor so that
    familiar observations gradually yield lower intrinsic rewards.

    The :attr:`predictor_network` and :attr:`target_network` should be the
    **same objects** passed to :class:`~torchrl.envs.transforms.RNDTransform`
    so that reducing the predictor error here also reduces the intrinsic reward
    produced during collection.

    Observation normalization is optionally applied using the running statistics
    maintained by :class:`~torchrl.envs.transforms.RNDTransform`.  Pass
    ``obs_rms=transform.obs_rms`` after collecting initial data to keep the
    normalization consistent between collection and training.

    Args:
        predictor_network (torch.nn.Module): trainable network.
        target_network (torch.nn.Module): frozen random network. Its parameters
            are frozen on construction.

    Keyword Args:
        obs_rms (RunningMeanStd, optional): running observation statistics
            shared with :class:`~torchrl.envs.transforms.RNDTransform`.
            When provided, observations are normalized before being passed to
            the networks, matching the normalization done during collection.
            Defaults to ``None`` (no normalization).
        obs_clip (float, optional): clip normalized observations to
            ``[-obs_clip, obs_clip]``.  Only used when ``obs_rms`` is not
            ``None``.  Default: ``5.0``.
        reduction (str, optional): reduction over the per-sample losses:
            ``"mean"`` | ``"sum"`` | ``"none"``.  Default: ``"mean"``.
        update_fraction (float, optional): fraction of each batch used to
            compute the predictor loss, following the original paper (default
            25 %).  A random mask selects which samples contribute so the
            operation is ``torch.compile``-friendly.  Default: ``0.25``.

    Examples:
        >>> import torch
        >>> import torch.nn as nn
        >>> from tensordict import TensorDict
        >>> from torchrl.objectives.rnd import RNDLoss
        >>> predictor = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64))
        >>> target = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64))
        >>> loss_fn = RNDLoss(predictor, target)
        >>> batch = TensorDict({"next": {"observation": torch.randn(32, 4)}}, [32])
        >>> loss_td = loss_fn(batch)
        >>> loss_td["loss_predictor"].backward()
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        Attributes:
            observation (NestedKey): key where the next observation is read
                from.  Defaults to ``("next", "observation")``, matching the
                nested format produced by replay buffers.
        """

        observation: NestedKey = ("next", "observation")

    default_keys = _AcceptedKeys
    tensor_keys: _AcceptedKeys

    out_keys = ["loss_predictor"]

    def __init__(
        self,
        predictor_network: nn.Module,
        target_network: nn.Module,
        *,
        obs_rms: RunningMeanStd | None = None,
        obs_clip: float = 5.0,
        reduction: str = "mean",
        update_fraction: float = 0.25,
    ):
        super().__init__()
        self.predictor_network = predictor_network
        self.target_network = target_network
        self.target_network.requires_grad_(False)
        self.obs_rms = obs_rms
        self.obs_clip = obs_clip
        self.reduction = reduction
        self.update_fraction = update_fraction

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs = tensordict.get(self.tensor_keys.observation).float()

        if self.obs_rms is not None:
            obs = self.obs_rms.normalize(obs).clamp(-self.obs_clip, self.obs_clip)

        with torch.no_grad():
            target_feat = self.target_network(obs)
        pred_feat = self.predictor_network(obs)

        per_sample_loss = (pred_feat - target_feat).pow(2).mean(dim=-1)

        if self.update_fraction < 1.0:
            mask = (
                torch.rand(per_sample_loss.shape, device=per_sample_loss.device)
                < self.update_fraction
            )
            per_sample_loss = torch.where(
                mask, per_sample_loss, torch.zeros_like(per_sample_loss)
            )
            if self.reduction == "mean":
                loss = per_sample_loss.sum() / mask.sum().clamp_min(1)
            else:
                loss = _reduce(per_sample_loss, self.reduction)
        else:
            loss = _reduce(per_sample_loss, self.reduction)

        return TensorDict({"loss_predictor": loss}, batch_size=[])
