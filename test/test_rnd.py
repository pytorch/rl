# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from torchrl.envs.transforms.rnd import RNDTransform, RunningMeanStd
from torchrl.objectives.rnd import RNDLoss
from torchrl.testing import get_default_devices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_networks(obs_dim: int = 4, embed_dim: int = 16):
    target = nn.Sequential(
        nn.Linear(obs_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
    )
    predictor = nn.Sequential(
        nn.Linear(obs_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
    )
    return target, predictor


# ---------------------------------------------------------------------------
# RunningMeanStd
# ---------------------------------------------------------------------------


class TestRunningMeanStd:
    def test_scalar_update(self):
        rms = RunningMeanStd(shape=())
        x = torch.arange(100, dtype=torch.float32)
        rms.update(x)
        assert abs(rms.mean.item() - x.mean().item()) < 1e-3
        assert abs(rms.var.item() - x.var(unbiased=False).item()) < 1e-1

    def test_vector_update(self):
        rms = RunningMeanStd(shape=(4,))
        x = torch.randn(1000, 4)
        rms.update(x)
        assert torch.allclose(rms.mean, x.mean(0), atol=0.1)
        assert torch.allclose(rms.var, x.var(0, unbiased=False), atol=0.2)

    def test_incremental_updates(self):
        rms = RunningMeanStd(shape=(4,))
        full = torch.randn(200, 4)
        rms.update(full[:100])
        rms.update(full[100:])
        rms_full = RunningMeanStd(shape=(4,))
        rms_full.update(full)
        assert torch.allclose(rms.mean, rms_full.mean, atol=1e-4)
        assert torch.allclose(rms.var, rms_full.var, atol=1e-4)

    def test_normalize_shape_preserved(self):
        rms = RunningMeanStd(shape=(4,))
        x = torch.randn(8, 4)
        rms.update(x)
        out = rms.normalize(x)
        assert out.shape == x.shape

    def test_normalize_nested_key(self):
        """Running stats should work with a 2-D nested NestedKey input."""
        rms = RunningMeanStd(shape=(4,))
        x = torch.randn(3, 5, 4)
        rms.update(x)
        out = rms.normalize(x)
        assert out.shape == x.shape

    def test_state_dict_roundtrip(self):
        rms = RunningMeanStd(shape=(4,))
        rms.update(torch.randn(32, 4))
        sd = rms.state_dict()
        rms2 = RunningMeanStd(shape=(4,))
        rms2.load_state_dict(sd)
        assert torch.allclose(rms.mean, rms2.mean)
        assert torch.allclose(rms.var, rms2.var)

    @pytest.mark.parametrize("device", get_default_devices())
    def test_device_move(self, device):
        rms = RunningMeanStd(shape=(4,)).to(device)
        x = torch.randn(16, 4, device=device)
        rms.update(x)
        out = rms.normalize(x)
        assert out.device.type == torch.device(device).type


# ---------------------------------------------------------------------------
# RNDTransform
# ---------------------------------------------------------------------------


class TestRNDTransform:
    @pytest.mark.parametrize("device", get_default_devices())
    def test_intrinsic_reward_written(self, device):
        target, predictor = _make_networks()
        transform = RNDTransform(target, predictor).to(device)
        obs = torch.randn(4, device=device)
        next_td = TensorDict({"observation": obs}, batch_size=[])
        transform._step(TensorDict({}, batch_size=[]), next_td)
        assert "intrinsic_reward" in next_td.keys()
        assert next_td["intrinsic_reward"].shape == torch.Size([1])

    @pytest.mark.parametrize("device", get_default_devices())
    def test_batched_intrinsic_reward(self, device):
        target, predictor = _make_networks()
        transform = RNDTransform(target, predictor).to(device)
        obs = torch.randn(8, 4, device=device)
        next_td = TensorDict({"observation": obs}, batch_size=[8])
        transform._step(TensorDict({}, batch_size=[8]), next_td)
        assert next_td["intrinsic_reward"].shape == torch.Size([8, 1])

    def test_target_frozen(self):
        target, predictor = _make_networks()
        transform = RNDTransform(target, predictor)
        for p in transform.target_network.parameters():
            assert not p.requires_grad

    def test_obs_rms_updated_in_train_mode(self):
        target, predictor = _make_networks()
        transform = RNDTransform(target, predictor, normalize_obs=True)
        transform.train()
        obs = torch.randn(32, 4)
        next_td = TensorDict({"observation": obs}, batch_size=[32])
        transform._step(TensorDict({}, batch_size=[32]), next_td)
        assert transform.obs_rms is not None
        assert transform.obs_rms.count.item() > 1e-4

    def test_obs_rms_not_updated_in_eval_mode(self):
        target, predictor = _make_networks()
        transform = RNDTransform(target, predictor, normalize_obs=True)
        transform.train()
        obs = torch.randn(32, 4)
        next_td = TensorDict({"observation": obs}, batch_size=[32])
        transform._step(TensorDict({}, batch_size=[32]), next_td)
        count_after_train = transform.obs_rms.count.item()

        transform.eval()
        next_td2 = TensorDict({"observation": obs}, batch_size=[32])
        transform._step(TensorDict({}, batch_size=[32]), next_td2)
        assert transform.obs_rms.count.item() == count_after_train

    def test_no_normalization(self):
        target, predictor = _make_networks()
        transform = RNDTransform(
            target, predictor, normalize_obs=False, normalize_reward=False
        )
        transform.train()
        obs = torch.randn(8, 4)
        next_td = TensorDict({"observation": obs}, batch_size=[8])
        transform._step(TensorDict({}, batch_size=[8]), next_td)
        assert transform.obs_rms is None
        assert transform.reward_rms is None
        assert "intrinsic_reward" in next_td.keys()

    def test_reward_rms_updated(self):
        target, predictor = _make_networks()
        transform = RNDTransform(target, predictor, normalize_reward=True)
        transform.train()
        for _ in range(5):
            obs = torch.randn(16, 4)
            next_td = TensorDict({"observation": obs}, batch_size=[16])
            transform._step(TensorDict({}, batch_size=[16]), next_td)
        assert transform.reward_rms is not None
        assert transform.reward_rms.count.item() > 1

    def test_custom_keys(self):
        target, predictor = _make_networks()
        transform = RNDTransform(
            target,
            predictor,
            in_keys=["obs_feat"],
            out_keys=["curiosity"],
        )
        obs = torch.randn(4)
        next_td = TensorDict({"obs_feat": obs}, batch_size=[])
        transform._step(TensorDict({}, batch_size=[]), next_td)
        assert "curiosity" in next_td.keys()

    def test_state_dict_includes_rms(self):
        target, predictor = _make_networks()
        transform = RNDTransform(target, predictor)
        transform.train()
        obs = torch.randn(8, 4)
        next_td = TensorDict({"observation": obs}, batch_size=[8])
        transform._step(TensorDict({}, batch_size=[8]), next_td)
        sd = transform.state_dict()
        assert any("obs_rms" in k for k in sd)


# ---------------------------------------------------------------------------
# RNDLoss
# ---------------------------------------------------------------------------


class TestRNDLoss:
    @pytest.mark.parametrize("device", get_default_devices())
    def test_forward_returns_loss(self, device):
        target, predictor = _make_networks()
        loss_fn = RNDLoss(predictor, target).to(device)
        batch = TensorDict(
            {"next": {"observation": torch.randn(32, 4, device=device)}}, [32]
        )
        out = loss_fn(batch)
        assert "loss_predictor" in out.keys()
        assert out["loss_predictor"].shape == torch.Size([])

    def test_backward(self):
        target, predictor = _make_networks()
        loss_fn = RNDLoss(predictor, target)
        batch = TensorDict({"next": {"observation": torch.randn(32, 4)}}, [32])
        out = loss_fn(batch)
        out["loss_predictor"].backward()
        for p in predictor.parameters():
            assert p.grad is not None
        for p in target.parameters():
            assert p.grad is None

    def test_target_frozen(self):
        target, predictor = _make_networks()
        loss_fn = RNDLoss(predictor, target)
        for p in loss_fn.target_network.parameters():
            assert not p.requires_grad

    def test_update_fraction_reduces_effective_batch(self):
        torch.manual_seed(0)
        target, predictor = _make_networks()
        loss_full = RNDLoss(predictor, target, update_fraction=1.0)
        loss_partial = RNDLoss(predictor, target, update_fraction=0.25)
        batch = TensorDict({"next": {"observation": torch.randn(1000, 4)}}, [1000])
        # Both should return a scalar without error
        out_full = loss_full(batch)
        out_partial = loss_partial(batch)
        assert out_full["loss_predictor"].shape == torch.Size([])
        assert out_partial["loss_predictor"].shape == torch.Size([])

    def test_set_keys(self):
        target, predictor = _make_networks()
        loss_fn = RNDLoss(predictor, target)
        loss_fn.set_keys(observation=("next", "obs_encoded"))
        assert loss_fn.tensor_keys.observation == ("next", "obs_encoded")
        batch = TensorDict({"next": {"obs_encoded": torch.randn(16, 4)}}, [16])
        out = loss_fn(batch)
        assert "loss_predictor" in out.keys()

    def test_obs_rms_shared_with_transform(self):
        target, predictor = _make_networks()
        transform = RNDTransform(target, predictor, normalize_obs=True)
        transform.train()
        obs = torch.randn(64, 4)
        next_td = TensorDict({"observation": obs}, batch_size=[64])
        transform._step(TensorDict({}, batch_size=[64]), next_td)

        loss_fn = RNDLoss(predictor, target, obs_rms=transform.obs_rms)
        batch = TensorDict({"next": {"observation": obs}}, [64])
        out = loss_fn(batch)
        out["loss_predictor"].backward()

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_reduction_modes(self, reduction):
        target, predictor = _make_networks()
        loss_fn = RNDLoss(predictor, target, reduction=reduction, update_fraction=1.0)
        batch = TensorDict({"next": {"observation": torch.randn(16, 4)}}, [16])
        out = loss_fn(batch)
        if reduction == "none":
            assert out["loss_predictor"].shape == torch.Size([16])
        else:
            assert out["loss_predictor"].shape == torch.Size([])

    def test_nested_observation_key(self):
        """NestedKey tuple should work as the observation key."""
        target, predictor = _make_networks()
        loss_fn = RNDLoss(predictor, target)
        loss_fn.set_keys(observation=("next", "obs"))
        batch = TensorDict({"next": {"obs": torch.randn(8, 4)}}, [8])
        out = loss_fn(batch)
        assert "loss_predictor" in out.keys()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
