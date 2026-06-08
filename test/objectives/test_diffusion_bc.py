# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from torchrl.modules import DiffusionActor
from torchrl.objectives import DiffusionBCLoss


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

    def test_reduction_none(self):
        """reduction='none' should not aggregate — loss is still scalar (MSE averages internally)."""
        actor = self._make_actor()
        loss_fn = DiffusionBCLoss(actor, reduction="none")
        td = self._make_batch()
        loss_td = loss_fn(td)
        # With reduction='none' the raw element-wise tensor is returned from mse_loss
        assert "loss_diffusion_bc" in loss_td.keys()

    def test_reduction_sum(self):
        actor = self._make_actor()
        loss_fn_mean = DiffusionBCLoss(actor, reduction="mean")
        loss_fn_sum = DiffusionBCLoss(actor, reduction="sum")
        td = self._make_batch(batch_size=4)
        # Both should produce a finite scalar
        assert loss_fn_mean(td)["loss_diffusion_bc"].isfinite()
        assert loss_fn_sum(td)["loss_diffusion_bc"].isfinite()

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
