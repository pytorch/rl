# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the ``torchrl.objectives.vla`` losses: chunked behavior cloning
(:class:`~torchrl.objectives.vla.VLABCLoss`) and token GRPO fine-tuning
(:class:`~torchrl.objectives.vla.VLATokenGRPOLoss`)."""
from __future__ import annotations

import argparse

import pytest
import torch
from tensordict import NonTensorStack, TensorDict

from torchrl.modules.vla import TinyVLA
from torchrl.objectives.vla import VLABCLoss, VLATokenGRPOLoss


def _make_bc_td(batch=4, chunk=4, action_dim=3, pad=False):
    obs = {
        "image": torch.rand(batch, 3, 16, 16),
        "state": torch.randn(batch, 5),
    }
    is_pad = torch.zeros(batch, chunk, dtype=torch.bool)
    if pad:
        is_pad[:, -1] = True
    return TensorDict(
        {
            "observation": obs,
            "language_instruction": NonTensorStack(*[f"t{i}" for i in range(batch)]),
            "action_chunk": torch.randn(batch, chunk, action_dim),
            "action_is_pad": is_pad,
        },
        batch_size=[batch],
    )


class TestVLABCLoss:
    def test_loss_key_and_shape(self):
        policy = TinyVLA(action_dim=3, chunk_size=4)
        loss = VLABCLoss(policy)
        out = loss(_make_bc_td())
        assert list(out.keys()) == ["loss_vla_bc"]
        assert out["loss_vla_bc"].shape == torch.Size([])

    @pytest.mark.parametrize("loss_function", ["l1", "l2", "smooth_l1"])
    def test_loss_functions(self, loss_function):
        policy = TinyVLA(action_dim=3, chunk_size=4)
        loss = VLABCLoss(policy, loss_function=loss_function)
        assert torch.isfinite(loss(_make_bc_td())["loss_vla_bc"])

    def test_reduction_none(self):
        policy = TinyVLA(action_dim=3, chunk_size=4)
        # without a pad mask, "none" preserves the per-element shape
        td = _make_bc_td()
        td = td.exclude("action_is_pad")
        out = VLABCLoss(policy, reduction="none")(td)["loss_vla_bc"]
        assert out.shape == torch.Size([4, 4, 3])

    def test_masking_excludes_padded_steps(self):
        policy = TinyVLA(action_dim=3, chunk_size=4)
        loss = VLABCLoss(policy)
        td = _make_bc_td(pad=True)
        td["action_chunk"][:, -1, :] = 1e6  # huge target on the padded step
        masked = loss(td)["loss_vla_bc"]
        unmasked = loss(td.exclude("action_is_pad"))["loss_vla_bc"]
        assert masked < unmasked  # the padded (huge) step is ignored when masked

    def test_nested_action_chunk_key(self):
        policy = TinyVLA(action_dim=2, chunk_size=3)
        policy.set_keys(action_chunk=("targets", "chunk"))
        loss = VLABCLoss(policy)
        loss.set_keys(action_chunk=("targets", "chunk"))
        td = TensorDict(
            {
                "observation": {
                    "image": torch.rand(4, 3, 16, 16),
                    "state": torch.randn(4, 5),
                },
                "language_instruction": NonTensorStack(*[f"t{i}" for i in range(4)]),
                "targets": {"chunk": torch.randn(4, 3, 2)},
            },
            batch_size=[4],
        )
        assert torch.isfinite(loss(td)["loss_vla_bc"])

    def test_rejects_token_head_policy(self):
        policy = TinyVLA(
            action_dim=3, chunk_size=4, action_head="tokens", vocab_size=16
        )
        with pytest.raises(ValueError, match="continuous-head"):
            VLABCLoss(policy)

    def test_set_keys_invalidates_in_keys(self):
        policy = TinyVLA(action_dim=2, chunk_size=3)
        loss = VLABCLoss(policy)
        _ = loss.in_keys  # populate the cache
        loss.set_keys(action_chunk=("targets", "chunk"))
        assert ("targets", "chunk") in loss.in_keys
        assert "action_chunk" not in loss.in_keys

    def test_overfit(self):
        torch.manual_seed(0)
        policy = TinyVLA(action_dim=3, chunk_size=4, hidden_dim=64)
        loss = VLABCLoss(policy)
        td = _make_bc_td()
        l0 = loss(td)["loss_vla_bc"].item()  # first forward materializes lazy params
        opt = torch.optim.Adam(loss.parameters(), lr=1e-2)
        for _ in range(80):
            opt.zero_grad()
            loss(td)["loss_vla_bc"].backward()
            opt.step()
        assert loss(td)["loss_vla_bc"].item() < 0.5 * l0


def _make_obs_td(batch=2, h=16, state_dim=5, with_state=True):
    obs = {"image": torch.zeros(batch, 3, h, h, dtype=torch.uint8)}
    if with_state:
        obs["state"] = torch.randn(batch, state_dim)
    data = {
        "observation": obs,
        "language_instruction": NonTensorStack(*[f"task {i}" for i in range(batch)]),
    }
    return TensorDict(data, batch_size=[batch])


class TestVLATokenGRPOLoss:
    @staticmethod
    def _setup(mode="sample", advantage=1.0, batch=8):
        torch.manual_seed(0)
        policy = TinyVLA(
            action_dim=2, chunk_size=2, action_head="tokens", vocab_size=8, mode=mode
        )
        obs = _make_obs_td(batch=batch)
        with torch.no_grad():
            coll = policy(obs.clone())
        td = obs.clone()
        td["action_tokens"] = coll["action_tokens"]
        td["log_probs"] = coll["log_probs"].detach()
        td["advantage"] = torch.full((batch,), float(advantage))
        return policy, obs, td

    def test_loss_keys(self):
        policy, _, td = self._setup()
        out = VLATokenGRPOLoss(policy)(td)
        assert "loss_objective" in out.keys()
        assert "clip_fraction" in out.keys()
        assert out["loss_objective"].shape == torch.Size([])

    def test_requires_token_head(self):
        policy = TinyVLA(action_dim=2, chunk_size=2)  # continuous head
        with pytest.raises(ValueError, match="token-head"):
            VLATokenGRPOLoss(policy)

    def _taken_logprob(self, policy, obs, td):
        with torch.no_grad():
            return (
                policy.get_dist(obs.clone()).log_prob(td["action_tokens"]).sum().item()
            )

    def test_positive_advantage_increases_logprob(self):
        policy, obs, td = self._setup(advantage=1.0)
        loss = VLATokenGRPOLoss(policy, clip_epsilon=0.2)
        before = self._taken_logprob(policy, obs, td)
        opt = torch.optim.Adam(loss.parameters(), lr=2e-3)
        for _ in range(20):
            opt.zero_grad()
            loss(td)["loss_objective"].backward()
            opt.step()
        assert self._taken_logprob(policy, obs, td) > before

    def test_negative_advantage_decreases_logprob(self):
        policy, obs, td = self._setup(advantage=-1.0)
        loss = VLATokenGRPOLoss(policy, clip_epsilon=0.2)
        before = self._taken_logprob(policy, obs, td)
        opt = torch.optim.Adam(loss.parameters(), lr=2e-3)
        for _ in range(20):
            opt.zero_grad()
            loss(td)["loss_objective"].backward()
            opt.step()
        assert self._taken_logprob(policy, obs, td) < before

    def test_kl_to_ref(self):
        policy, _, td = self._setup()
        td["ref_log_probs"] = td["log_probs"]
        loss = VLATokenGRPOLoss(policy, kl_to_ref_coeff=0.1)
        out = loss(td)
        assert "loss_kl_to_ref" in out.keys()
        assert "kl_to_ref" in out.keys()
        assert torch.isfinite(out["loss_kl_to_ref"])
        # the KL keys are reflected in out_keys and ref_log_probs in in_keys
        assert "loss_kl_to_ref" in loss.out_keys
        assert "ref_log_probs" in loss.in_keys

    def test_advantage_shapes_equivalent(self):
        policy, _, td = self._setup(batch=4)
        td["advantage"] = torch.randn(4)
        loss = VLATokenGRPOLoss(policy, reduction="none")
        base = loss(td.copy())["loss_objective"]
        assert base.shape == torch.Size([4])
        for shape in [(4, 1), (4, 1, 1)]:
            td2 = td.copy()
            td2["advantage"] = td["advantage"].reshape(shape)
            torch.testing.assert_close(loss(td2)["loss_objective"], base)

    def test_single_sample_batch(self):
        policy, _, td = self._setup(batch=1)
        out = VLATokenGRPOLoss(policy, reduction="none")(td)
        assert out["loss_objective"].shape == torch.Size([1])

    def test_set_keys_and_nested_advantage(self):
        policy, _, td = self._setup()
        loss = VLATokenGRPOLoss(policy)
        loss.set_keys(advantage=("group", "adv"))
        assert ("group", "adv") in loss.in_keys  # cache invalidated
        td.set(("group", "adv"), td.get("advantage"))
        assert torch.isfinite(loss(td)["loss_objective"])

    def test_missing_advantage_raises(self):
        policy, _, td = self._setup()
        del td["advantage"]
        with pytest.raises(KeyError, match="advantage"):
            VLATokenGRPOLoss(policy)(td)

    def test_non_detached_sample_log_prob_raises(self):
        policy, _, td = self._setup()
        td["log_probs"] = td["log_probs"].clone().requires_grad_(True)
        with pytest.raises(RuntimeError, match="requires grad"):
            VLATokenGRPOLoss(policy)(td)

    def test_invalid_clip_epsilon(self):
        policy = TinyVLA(action_dim=2, chunk_size=2, action_head="tokens", vocab_size=8)
        with pytest.raises(ValueError, match="clip_epsilon"):
            VLATokenGRPOLoss(policy, clip_epsilon=1.0)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
