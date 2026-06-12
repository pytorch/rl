# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integration tests for VLA fine-tuning with the standard objectives:
chunked behavior cloning via :class:`~torchrl.objectives.BCLoss` (with its
``pad_mask`` key) and token RL fine-tuning via
:class:`~torchrl.objectives.ClipPPOLoss`."""
from __future__ import annotations

import argparse

import pytest
import torch
from tensordict import NonTensorStack, TensorDict

from torchrl.modules.vla import TinyVLA
from torchrl.objectives import BCLoss, ClipPPOLoss


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


def _make_chunked_bc_loss(policy, **kwargs):
    # chunked (VLA-style) behavior cloning is plain BCLoss with the chunk as
    # the action and the padding mask excluded from the loss
    kwargs.setdefault("loss_function", "l1")
    loss = BCLoss(policy, **kwargs)
    loss.set_keys(action="action_chunk", pad_mask="action_is_pad")
    return loss


class TestChunkedBC:
    @staticmethod
    def _policy(action_dim=3, chunk_size=4, **kwargs):
        policy = TinyVLA(action_dim=action_dim, chunk_size=chunk_size, **kwargs)
        # materialize the lazy parameters before BCLoss functionalizes them
        policy(_make_bc_td(action_dim=action_dim, chunk=chunk_size).clone())
        return policy

    def test_loss_key_and_shape(self):
        loss = _make_chunked_bc_loss(self._policy())
        out = loss(_make_bc_td())
        assert list(out.keys()) == ["loss_bc"]
        assert out["loss_bc"].shape == torch.Size([])
        assert "action_is_pad" in loss.in_keys

    @pytest.mark.parametrize("loss_function", ["l1", "l2", "smooth_l1"])
    def test_loss_functions(self, loss_function):
        loss = _make_chunked_bc_loss(self._policy(), loss_function=loss_function)
        assert torch.isfinite(loss(_make_bc_td())["loss_bc"])

    def test_reduction_none(self):
        td = _make_bc_td().exclude("action_is_pad")
        loss = _make_chunked_bc_loss(self._policy(), reduction="none")
        out = loss(td)["loss_bc"]
        assert out.shape == torch.Size([4, 4, 3])

    def test_masking_excludes_padded_steps(self):
        loss = _make_chunked_bc_loss(self._policy())
        td = _make_bc_td(pad=True)
        td["action_chunk"][:, -1, :] = 1e6  # huge target on the padded step
        masked = loss(td)["loss_bc"]
        unmasked = loss(td.exclude("action_is_pad"))["loss_bc"]
        assert masked < unmasked  # the padded (huge) step is ignored when masked

    def test_nested_action_chunk_key(self):
        policy = TinyVLA(action_dim=2, chunk_size=3)
        policy.set_keys(action_chunk=("targets", "chunk"))
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
        policy(td.clone())
        loss = BCLoss(policy, loss_function="l1")
        loss.set_keys(action=("targets", "chunk"))
        assert torch.isfinite(loss(td)["loss_bc"])
        assert ("targets", "chunk") in loss.in_keys

    def test_overfit(self):
        torch.manual_seed(0)
        loss = _make_chunked_bc_loss(self._policy(hidden_dim=64))
        td = _make_bc_td()
        l0 = loss(td)["loss_bc"].item()
        opt = torch.optim.Adam(loss.parameters(), lr=1e-2)
        for _ in range(80):
            opt.zero_grad()
            loss(td)["loss_bc"].backward()
            opt.step()
        assert loss(td)["loss_bc"].item() < 0.5 * l0


def _make_obs_td(batch=2, h=16, state_dim=5):
    return TensorDict(
        {
            "observation": {
                "image": torch.zeros(batch, 3, h, h, dtype=torch.uint8),
                "state": torch.randn(batch, state_dim),
            },
            "language_instruction": NonTensorStack(
                *[f"task {i}" for i in range(batch)]
            ),
        },
        batch_size=[batch],
    )


def _make_token_ppo_loss(policy, **kwargs):
    # token RL fine-tuning (GRPO-style: precomputed group advantages, no
    # critic) is plain ClipPPOLoss over the action tokens
    kwargs.setdefault("clip_epsilon", 0.2)
    loss = ClipPPOLoss(policy, critic_network=None, entropy_bonus=False, **kwargs)
    loss.set_keys(
        action="action_tokens", sample_log_prob="log_probs", advantage="advantage"
    )
    return loss


class TestTokenPPO:
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
        # the advantage carries the trailing singleton value-dim the PPO
        # losses expect; a flat [batch] advantage would broadcast wrong
        td["advantage"] = torch.full((batch, 1), float(advantage))
        return policy, obs, td

    def test_loss_keys(self):
        policy, _, td = self._setup()
        out = _make_token_ppo_loss(policy)(td)
        assert "loss_objective" in out.keys()
        assert "clip_fraction" in out.keys()
        assert out["loss_objective"].shape == torch.Size([])

    def test_sequence_level_log_probs(self):
        # the token head emits one (summed) log-prob per sample, matching the
        # sample_log_prob contract of the PPO losses
        policy, obs, td = self._setup(batch=3)
        assert td["log_probs"].shape == torch.Size([3])
        dist = policy.get_dist(obs.clone())
        assert dist.log_prob(td["action_tokens"]).shape == torch.Size([3])

    def _taken_logprob(self, policy, obs, td):
        with torch.no_grad():
            return (
                policy.get_dist(obs.clone()).log_prob(td["action_tokens"]).sum().item()
            )

    def test_positive_advantage_increases_logprob(self):
        policy, obs, td = self._setup(advantage=1.0)
        loss = _make_token_ppo_loss(policy)
        before = self._taken_logprob(policy, obs, td)
        opt = torch.optim.Adam(loss.parameters(), lr=2e-3)
        for _ in range(20):
            opt.zero_grad()
            loss(td)["loss_objective"].backward()
            opt.step()
        assert self._taken_logprob(policy, obs, td) > before

    def test_negative_advantage_decreases_logprob(self):
        policy, obs, td = self._setup(advantage=-1.0)
        loss = _make_token_ppo_loss(policy)
        before = self._taken_logprob(policy, obs, td)
        opt = torch.optim.Adam(loss.parameters(), lr=2e-3)
        for _ in range(20):
            opt.zero_grad()
            loss(td)["loss_objective"].backward()
            opt.step()
        assert self._taken_logprob(policy, obs, td) < before

    def test_single_sample_batch(self):
        policy, _, td = self._setup(batch=1)
        out = _make_token_ppo_loss(policy, reduction="none")(td)
        assert out["loss_objective"].shape == torch.Size([1])

    def test_per_sample_objective_matches_reference(self):
        # non-constant advantage against a hand-computed clipped surrogate:
        # guards against a flat [batch] advantage broadcasting into a
        # [batch, batch] outer product inside the PPO loss
        policy, obs, td = self._setup(batch=4)
        td["advantage"] = torch.tensor([[1.0], [-1.0], [2.0], [0.5]])
        td["log_probs"] = td["log_probs"] + 0.3  # ratio != 1 so clipping bites
        out = _make_token_ppo_loss(policy, reduction="none")(td)["loss_objective"]
        assert out.shape == torch.Size([4])
        with torch.no_grad():
            log_weight = (
                policy.get_dist(obs.clone()).log_prob(td["action_tokens"])
                - td["log_probs"]
            )
        ratio = log_weight.exp().unsqueeze(-1)
        gain = torch.min(
            ratio * td["advantage"],
            ratio.clamp(1 - 0.2, 1 + 0.2) * td["advantage"],
        )
        torch.testing.assert_close(out, -gain.squeeze(-1))


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
