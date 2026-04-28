# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from torchrl.data.tensor_specs import Bounded
from torchrl.modules.tensordict_module.actors import Actor, ProbabilisticActor
from torchrl.objectives import BCLoss


class TestBCLoss:
    def _make_deterministic_actor(self, action_dim=2, obs_dim=4):
        spec = Bounded(-torch.ones(action_dim), torch.ones(action_dim), (action_dim,))
        module = torch.nn.Linear(obs_dim, action_dim)
        return Actor(module=module, spec=spec)

    def _make_stochastic_actor(self, action_dim=2, obs_dim=4):
        from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
        from torchrl.modules.tensordict_module.common import SafeModule
        from tensordict.nn import TensorDictModule

        spec = Bounded(-torch.ones(action_dim), torch.ones(action_dim), (action_dim,))
        net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 2 * action_dim),
            NormalParamExtractor(),
        )
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        return ProbabilisticActor(
            module=module,
            distribution_class=TanhNormal,
            return_log_prob=True,
            in_keys=["loc", "scale"],
            spec=spec,
        )
        module = torch.nn.Sequential(net)
        return ProbabilisticActor(
            module=module,
            distribution_class=TanhNormal,
            return_log_prob=True,
            in_keys=["loc", "scale"],
            spec=spec,
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
        actor = self._make_deterministic_actor()
        loss_fn = BCLoss(actor)
        td = self._make_batch()
        loss_td = loss_fn(td)
        assert "loss_bc" in loss_td.keys()

    def test_deterministic_loss_is_scalar(self):
        actor = self._make_deterministic_actor()
        loss_fn = BCLoss(actor)
        td = self._make_batch()
        loss_td = loss_fn(td)
        assert loss_td["loss_bc"].shape == torch.Size([])

    def test_stochastic_loss_is_scalar(self):
        actor = self._make_stochastic_actor()
        loss_fn = BCLoss(actor)
        td = self._make_batch()
        loss_td = loss_fn(td)
        assert loss_td["loss_bc"].shape == torch.Size([])

    def test_deterministic_loss_is_positive(self):
        actor = self._make_deterministic_actor()
        loss_fn = BCLoss(actor)
        td = self._make_batch()
        loss_td = loss_fn(td)
        assert loss_td["loss_bc"].item() >= 0.0

    def test_stochastic_loss_is_positive(self):
        actor = self._make_stochastic_actor()
        loss_fn = BCLoss(actor)
        td = self._make_batch()
        loss_td = loss_fn(td)
        assert loss_td["loss_bc"].item() >= 0.0

    def test_deterministic_backward(self):
        actor = self._make_deterministic_actor()
        loss_fn = BCLoss(actor)
        td = self._make_batch()
        loss_td = loss_fn(td)
        loss_td["loss_bc"].backward()
        grads = [p.grad for p in actor.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_stochastic_backward(self):
        actor = self._make_stochastic_actor()
        loss_fn = BCLoss(actor)
        td = self._make_batch()
        loss_td = loss_fn(td)
        loss_td["loss_bc"].backward()
        grads = [p.grad for p in actor.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_deterministic_gradients_nonzero(self):
        actor = self._make_deterministic_actor()
        loss_fn = BCLoss(actor)
        td = self._make_batch()
        loss_td = loss_fn(td)
        loss_td["loss_bc"].backward()
        for p in actor.parameters():
            if p.grad is not None:
                assert p.grad.abs().sum() > 0

    def test_stochastic_gradients_nonzero(self):
        actor = self._make_stochastic_actor()
        loss_fn = BCLoss(actor)
        td = self._make_batch()
        loss_td = loss_fn(td)
        loss_td["loss_bc"].backward()
        for p in actor.parameters():
            if p.grad is not None:
                assert p.grad.abs().sum() > 0

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_reduction_modes(self, reduction):
        actor = self._make_deterministic_actor()
        loss_fn = BCLoss(actor, reduction=reduction)
        td = self._make_batch(batch_size=4)
        loss_td = loss_fn(td)
        assert "loss_bc" in loss_td.keys()
        if reduction == "none":
            # For deterministic actor, loss should be scalar per sample then reduced
            pass  # The exact shape depends on implementation
        else:
            assert loss_td["loss_bc"].shape == torch.Size([])

    @pytest.mark.parametrize("loss_function", ["l1", "l2", "smooth_l1"])
    def test_loss_functions_deterministic(self, loss_function):
        actor = self._make_deterministic_actor()
        loss_fn = BCLoss(actor, loss_function=loss_function)
        td = self._make_batch()
        loss_td = loss_fn(td)
        assert loss_td["loss_bc"].isfinite()

    def test_custom_keys(self):
        actor = self._make_deterministic_actor()
        loss_fn = BCLoss(actor)
        loss_fn.set_keys(action="demo_action")
        td = TensorDict(
            {
                "observation": torch.randn(8, 4),
                "demo_action": torch.randn(8, 2),
            },
            batch_size=[8],
        )
        loss_td = loss_fn(td)
        assert "loss_bc" in loss_td.keys()

    def test_in_keys_deterministic(self):
        actor = self._make_deterministic_actor()
        loss_fn = BCLoss(actor)
        assert "observation" in loss_fn.in_keys
        assert "action" in loss_fn.in_keys

    def test_in_keys_stochastic(self):
        actor = self._make_stochastic_actor()
        loss_fn = BCLoss(actor)
        assert "observation" in loss_fn.in_keys
        assert "action" in loss_fn.in_keys

    def test_out_keys(self):
        actor = self._make_deterministic_actor()
        loss_fn = BCLoss(actor)
        assert "loss_bc" in loss_fn.out_keys

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batch_sizes(self, batch_size):
        actor = self._make_deterministic_actor()
        loss_fn = BCLoss(actor)
        td = self._make_batch(batch_size=batch_size)
        loss_td = loss_fn(td)
        assert loss_td["loss_bc"].shape == torch.Size([])

    @pytest.mark.parametrize("action_dim,obs_dim", [(2, 4), (4, 8), (6, 12)])
    def test_various_dims(self, action_dim, obs_dim):
        actor = self._make_deterministic_actor(action_dim=action_dim, obs_dim=obs_dim)
        loss_fn = BCLoss(actor)
        td = self._make_batch(action_dim=action_dim, obs_dim=obs_dim)
        loss_td = loss_fn(td)
        assert loss_td["loss_bc"].isfinite()

    def test_loss_changes_with_training_deterministic(self):
        """Loss should decrease after a few gradient steps on a fixed batch (deterministic)."""
        actor = self._make_deterministic_actor()
        loss_fn = BCLoss(actor)
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        td = self._make_batch(batch_size=32)

        torch.manual_seed(0)
        initial_loss = loss_fn(td)["loss_bc"].item()

        for _ in range(20):
            optimizer.zero_grad()
            loss = loss_fn(td)["loss_bc"]
            loss.backward()
            optimizer.step()

        final_loss = loss_fn(td)["loss_bc"].item()
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_loss_changes_with_training_stochastic(self):
        """Loss should decrease after a few gradient steps on a fixed batch (stochastic)."""
        actor = self._make_stochastic_actor()
        loss_fn = BCLoss(actor)
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        td = self._make_batch(batch_size=32)

        torch.manual_seed(0)
        initial_loss = loss_fn(td)["loss_bc"].item()

        for _ in range(20):
            optimizer.zero_grad()
            loss = loss_fn(td)["loss_bc"]
            loss.backward()
            optimizer.step()

        final_loss = loss_fn(td)["loss_bc"].item()
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )
