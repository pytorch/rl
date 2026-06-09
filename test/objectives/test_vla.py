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
from torchrl.objectives.vla import VLABCLoss


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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
