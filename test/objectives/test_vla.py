# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integration tests for chunked (VLA-style) behavior cloning via
:class:`~torchrl.objectives.BCLoss` and its ``pad_mask`` key."""
from __future__ import annotations

import argparse

import pytest
import torch
from tensordict import NonTensorStack, TensorDict

from torchrl.modules.vla import TinyVLA
from torchrl.objectives import BCLoss


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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
