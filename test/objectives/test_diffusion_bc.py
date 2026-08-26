# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.modules import DiffusionActor
from torchrl.objectives import DiffusionBCLoss


class _FixedDDPM(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_steps = 1
        self.score_network = nn.Linear(5, 2)
        with torch.no_grad():
            self.score_network.weight.zero_()
            self.score_network.bias.copy_(torch.tensor([1.0, -1.0]))

    def forward(self, observation):
        return observation[..., :2]

    def add_noise(self, clean_action, t):
        return torch.zeros_like(clean_action), clean_action


class TestDiffusionBCLoss:
    def _make_actor(self, action_dim=2, obs_dim=4, num_steps=5):
        return DiffusionActor(
            action_dim=action_dim, obs_dim=obs_dim, num_steps=num_steps
        )

    def _make_batch(self, batch_size=8, action_dim=2, obs_dim=4):
        return TensorDict(
            {
                "observation": torch.randn(batch_size, obs_dim),
                "action": torch.randn(batch_size, action_dim),
            },
            batch_size=[batch_size],
        )

    def test_output_keys(self):
        actor = self._make_actor()
        loss_fn = DiffusionBCLoss(actor)
        td = self._make_batch()
        loss_td = loss_fn(td)
        assert "loss_diffusion_bc" in loss_td.keys()

    def test_loss_is_scalar(self):
        actor = self._make_actor()
        loss_fn = DiffusionBCLoss(actor)
        td = self._make_batch()
        loss_td = loss_fn(td)
        assert loss_td["loss_diffusion_bc"].shape == torch.Size([])

    def test_loss_is_positive(self):
        actor = self._make_actor()
        loss_fn = DiffusionBCLoss(actor)
        td = self._make_batch()
        loss_td = loss_fn(td)
        assert loss_td["loss_diffusion_bc"].item() >= 0.0

    def test_backward(self):
        actor = self._make_actor()
        loss_fn = DiffusionBCLoss(actor)
        td = self._make_batch()
        loss_td = loss_fn(td)
        loss_td["loss_diffusion_bc"].backward()
        grads = [p.grad for p in actor.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients flowed to actor parameters"

    def test_gradients_nonzero(self):
        actor = self._make_actor()
        loss_fn = DiffusionBCLoss(actor)
        td = self._make_batch()
        loss_td = loss_fn(td)
        loss_td["loss_diffusion_bc"].backward()
        for p in actor.parameters():
            if p.grad is not None:
                assert p.grad.abs().sum() > 0

    @pytest.mark.parametrize(
        ("reduction", "expected"),
        [
            ("none", [[1.0, 4.0], [1.0, 25.0]]),
            ("mean", 7.75),
            ("sum", 31.0),
        ],
    )
    def test_reduction(self, reduction, expected):
        actor = TensorDictModule(
            _FixedDDPM(), in_keys=["observation"], out_keys=["action"]
        )
        loss_fn = DiffusionBCLoss(actor, reduction=reduction)
        td = TensorDict(
            {
                "observation": torch.zeros(2, 2),
                "action": torch.tensor([[0.0, 1.0], [2.0, 4.0]]),
            },
            batch_size=[2],
        )
        loss = loss_fn(td)["loss_diffusion_bc"]
        torch.testing.assert_close(loss, loss.new_tensor(expected))

    def test_custom_keys(self):
        actor = self._make_actor()
        loss_fn = DiffusionBCLoss(actor)
        loss_fn.set_keys(action="demo_action", observation="obs")
        td = TensorDict(
            {
                "obs": torch.randn(8, 4),
                "demo_action": torch.randn(8, 2),
            },
            batch_size=[8],
        )
        loss_td = loss_fn(td)
        assert "loss_diffusion_bc" in loss_td.keys()

    def test_nested_keys(self):
        actor = self._make_actor()
        loss_fn = DiffusionBCLoss(actor)
        loss_fn.set_keys(
            action=("data", "demo_action"), observation=("data", "observation")
        )
        td = TensorDict(
            {
                "data": {
                    "observation": torch.randn(8, 4),
                    "demo_action": torch.randn(8, 2),
                }
            },
            batch_size=[8],
        )
        assert loss_fn(td)["loss_diffusion_bc"].isfinite()

    def test_in_keys(self):
        actor = self._make_actor()
        loss_fn = DiffusionBCLoss(actor)
        assert "observation" in loss_fn.in_keys
        assert "action" in loss_fn.in_keys

    def test_out_keys(self):
        actor = self._make_actor()
        loss_fn = DiffusionBCLoss(actor)
        assert "loss_diffusion_bc" in loss_fn.out_keys

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batch_sizes(self, batch_size):
        actor = self._make_actor()
        loss_fn = DiffusionBCLoss(actor)
        td = self._make_batch(batch_size=batch_size)
        loss_td = loss_fn(td)
        assert loss_td["loss_diffusion_bc"].shape == torch.Size([])

    @pytest.mark.parametrize(
        ("batch_size", "data_shape"),
        [
            ([], []),
            ([8], [8]),
            ([2, 3], [2, 3]),
            (None, [8]),
            ([8], [8, 5]),
        ],
    )
    def test_batch_layouts(self, batch_size, data_shape):
        actor = self._make_actor()
        loss_fn = DiffusionBCLoss(actor)
        data = {
            "observation": torch.randn(*data_shape, 4),
            "action": torch.randn(*data_shape, 2),
        }
        td = TensorDict(data) if batch_size is None else TensorDict(data, batch_size)
        loss = loss_fn(td)["loss_diffusion_bc"]
        loss.backward()
        assert loss.shape == torch.Size([])
        assert all(parameter.grad is not None for parameter in actor.parameters())

    @pytest.mark.parametrize("action_dim,obs_dim", [(2, 4), (4, 8), (6, 12)])
    def test_various_dims(self, action_dim, obs_dim):
        actor = self._make_actor(action_dim=action_dim, obs_dim=obs_dim)
        loss_fn = DiffusionBCLoss(actor)
        td = self._make_batch(action_dim=action_dim, obs_dim=obs_dim)
        loss_td = loss_fn(td)
        assert loss_td["loss_diffusion_bc"].isfinite()

    def test_loss_changes_with_training(self):
        """Loss should decrease after a few gradient steps on a fixed batch."""
        actor = self._make_actor(num_steps=5)
        loss_fn = DiffusionBCLoss(actor)
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        td = self._make_batch(batch_size=32)

        torch.manual_seed(0)
        initial_loss = loss_fn(td)["loss_diffusion_bc"].item()

        for _ in range(20):
            optimizer.zero_grad()
            loss = loss_fn(td)["loss_diffusion_bc"]
            loss.backward()
            optimizer.step()

        final_loss = loss_fn(td)["loss_diffusion_bc"].item()
        assert (
            final_loss < initial_loss
        ), f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
