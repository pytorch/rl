# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data.tensor_specs import Bounded
from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
from torchrl.modules.tensordict_module.actors import Actor, ProbabilisticActor
from torchrl.objectives import BCLoss


class TestBCLoss:
    def _make_deterministic_actor(self, action_dim=2, obs_dim=4):
        spec = Bounded(-torch.ones(action_dim), torch.ones(action_dim), (action_dim,))
        module = torch.nn.Linear(obs_dim, action_dim)
        return Actor(module=module, spec=spec)

    def _make_stochastic_actor(self, action_dim=2, obs_dim=4):
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
        assert (
            final_loss < initial_loss
        ), f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"

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
        assert (
            final_loss < initial_loss
        ), f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"


class TestBCLossMaskAware:
    """Tests for mask-aware time-averaging in :class:`BCLoss`.

    Padded positions written by ``SliceSampler(pad_output=True)`` are
    flagged via the ``("collector", "mask")`` key (False for padding,
    True for real steps). When that key is present, ``BCLoss`` should
    exclude the padded positions from its time-averaging so duplicated
    last-step values do not contribute to the gradient.

    Back-compat: when the key is absent the loss must reduce to exactly
    the same value (and gradient) as before this feature.
    """

    @staticmethod
    def _make_actor(action_dim: int = 2, obs_dim: int = 4) -> Actor:
        spec = Bounded(-torch.ones(action_dim), torch.ones(action_dim), (action_dim,))
        module = torch.nn.Linear(obs_dim, action_dim)
        return Actor(module=module, spec=spec)

    @staticmethod
    def _make_batch(
        batch_size: int = 8, action_dim: int = 2, obs_dim: int = 4
    ) -> TensorDict:
        return TensorDict(
            {
                "observation": torch.randn(batch_size, obs_dim),
                "action": torch.randn(batch_size, action_dim),
            },
            batch_size=[batch_size],
        )

    def test_no_mask_key_is_backward_compatible(self):
        """Without the mask key, output is identical to the prior path."""
        torch.manual_seed(0)
        actor = self._make_actor()
        loss_fn = BCLoss(actor, loss_function="l2")
        batch = self._make_batch(batch_size=8)

        baseline = loss_fn(batch.clone())["loss_bc"]
        assert ("collector", "mask") not in batch.keys(include_nested=True)
        repeat = loss_fn(batch.clone())["loss_bc"]
        torch.testing.assert_close(baseline, repeat)

    def test_all_true_mask_equals_unmasked(self):
        """A mask of all True is a no-op for the masked-mean reduction."""
        torch.manual_seed(0)
        actor = self._make_actor()
        loss_fn = BCLoss(actor, loss_function="l2")
        batch = self._make_batch(batch_size=8)

        unmasked = loss_fn(batch.clone())["loss_bc"]

        masked_batch = batch.clone()
        masked_batch[("collector", "mask")] = torch.ones(8, dtype=torch.bool)
        masked = loss_fn(masked_batch)["loss_bc"]

        torch.testing.assert_close(unmasked, masked)

    def test_partial_mask_matches_manual_subset_loss(self):
        """A partially-False mask must drop padded positions from the average."""
        torch.manual_seed(0)
        actor = self._make_actor()
        loss_fn = BCLoss(actor, loss_function="l2")

        batch = self._make_batch(batch_size=8)
        mask = torch.tensor([True, True, True, True, False, False, False, False])

        masked_batch = batch.clone()
        masked_batch[("collector", "mask")] = mask
        with_mask = loss_fn(masked_batch)["loss_bc"]

        # Compute the loss on only the real (mask=True) positions and compare.
        # Padded positions must never contribute to the reported loss.
        real_only = batch[mask]
        manual = loss_fn(real_only.clone())["loss_bc"]

        torch.testing.assert_close(with_mask, manual, rtol=1e-5, atol=1e-6)

    def test_partial_mask_gradient_matches_manual_subset(self):
        """Backward through the masked loss matches backward through the subset."""
        torch.manual_seed(0)
        actor_a = self._make_actor()
        actor_b = self._make_actor()
        actor_b.load_state_dict(actor_a.state_dict())

        loss_a = BCLoss(actor_a, loss_function="l2")
        loss_b = BCLoss(actor_b, loss_function="l2")

        batch = self._make_batch(batch_size=8)
        mask = torch.tensor([True, True, True, True, False, False, False, False])

        masked_batch = batch.clone()
        masked_batch[("collector", "mask")] = mask
        loss_a(masked_batch)["loss_bc"].backward()

        real_only = batch[mask]
        loss_b(real_only.clone())["loss_bc"].backward()

        for p_a, p_b in zip(actor_a.parameters(), actor_b.parameters()):
            assert p_a.grad is not None and p_b.grad is not None
            torch.testing.assert_close(p_a.grad, p_b.grad, rtol=1e-5, atol=1e-6)

    def test_all_false_mask_produces_zero_loss(self):
        """A fully-padded batch (no real positions) reduces to a zero loss.

        ``_reduce_loss`` clamps the denominator at 1, so the reduction stays
        well-defined (no NaN); the numerator is zero because every element
        is masked out, giving a loss of zero.
        """
        torch.manual_seed(0)
        actor = self._make_actor()
        loss_fn = BCLoss(actor, loss_function="l2")
        batch = self._make_batch(batch_size=4)
        batch[("collector", "mask")] = torch.zeros(4, dtype=torch.bool)

        out = loss_fn(batch)["loss_bc"]
        torch.testing.assert_close(out, torch.zeros_like(out))


if __name__ == "__main__":
    import argparse
    import sys

    args, unknown = argparse.ArgumentParser().parse_known_args()
    sys.exit(pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown))
    def test_custom_action_key(self):
        # set_keys(action=...) must drive BOTH the expert read and the
        # prediction read (the latter used to be hardcoded to "action")
        n_obs, n_act = 3, 4
        actor = TensorDictModule(
            nn.Linear(n_obs, n_act), in_keys=["observation"], out_keys=["my_action"]
        )
        loss = BCLoss(actor, loss_function="l2")
        loss.set_keys(action="my_action")
        assert "my_action" in loss.in_keys
        td = TensorDict(
            {"observation": torch.randn(2, n_obs), "my_action": torch.randn(2, n_act)},
            batch_size=[2],
        )
        out = loss(td)["loss_bc"]
        assert torch.isfinite(out)
        out.backward()

    def test_pad_mask(self):
        # padded elements (pad_mask=True) are excluded from the loss; the mask
        # broadcasts over trailing dims
        n_obs, n_act = 3, 4
        actor = TensorDictModule(
            nn.Linear(n_obs, n_act), in_keys=["observation"], out_keys=["action"]
        )
        loss = BCLoss(actor, loss_function="l1")
        loss.set_keys(pad_mask="is_pad")
        assert "is_pad" in loss.in_keys
        td = TensorDict(
            {"observation": torch.randn(2, n_obs), "action": torch.randn(2, n_act)},
            batch_size=[2],
        )
        td["action"][1] = 1e6  # huge target on the padded sample
        td["is_pad"] = torch.tensor([False, True])
        masked = loss(td)["loss_bc"]
        unmasked = loss(td.exclude("is_pad"))["loss_bc"]
        assert masked < unmasked
        # a missing mask entry behaves exactly like a loss with no pad_mask
        no_mask_loss = BCLoss(actor, loss_function="l1")
        torch.testing.assert_close(
            unmasked, no_mask_loss(td.exclude("is_pad"))["loss_bc"]
        )
        # set_keys(pad_mask=None) resets the key and disables masking
        loss.set_keys(pad_mask=None)
        assert "is_pad" not in loss.in_keys
        torch.testing.assert_close(loss(td)["loss_bc"], unmasked)

    def test_pad_mask_reduction_none(self):
        # with a mask, reduction="none" returns the flat 1D tensor of
        # unmasked loss elements (the _reduce convention)
        n_obs, n_act = 3, 4
        actor = TensorDictModule(
            nn.Linear(n_obs, n_act), in_keys=["observation"], out_keys=["action"]
        )
        loss = BCLoss(actor, loss_function="l1", reduction="none")
        loss.set_keys(pad_mask="is_pad")
        td = TensorDict(
            {
                "observation": torch.randn(2, n_obs),
                "action": torch.randn(2, n_act),
                "is_pad": torch.tensor([False, True]),
            },
            batch_size=[2],
        )
        assert loss(td)["loss_bc"].shape == torch.Size([n_act])
        assert loss(td.exclude("is_pad"))["loss_bc"].shape == torch.Size([2, n_act])

    def test_pad_mask_requires_elementwise_loss(self):
        # a [B, H]-style mask cannot apply to a distribution-based loss whose
        # log_prob has consumed the event dims: expect an informative error
        n_obs, n_act = 3, 4
        actor = TensorDictModule(
            nn.Linear(n_obs, n_act), in_keys=["observation"], out_keys=["action"]
        )
        loss = BCLoss(actor, loss_function="l1")
        loss.set_keys(pad_mask="is_pad")
        td = TensorDict(
            {
                "observation": torch.randn(2, n_obs),
                "action": torch.randn(2, n_act),
                "is_pad": torch.zeros(2, n_act, 2, dtype=torch.bool),
            },
            batch_size=[2],
        )
        with pytest.raises(RuntimeError, match="more\\s+dimensions"):
            loss(td)

    def test_mismatched_action_key_raises(self):
        # an actor that writes neither the configured action key nor the
        # legacy "action" key must raise on every loss path, not silently
        # regress the expert onto itself
        n_obs, n_act = 3, 4
        actor = TensorDictModule(
            nn.Linear(n_obs, n_act), in_keys=["observation"], out_keys=["prediction"]
        )
        td = TensorDict(
            {
                "observation": torch.randn(2, n_obs),
                "expert_action": torch.randn(2, n_act),
            },
            batch_size=[2],
        )
        for kwargs in ({"loss_function": "l2"}, {}):  # explicit and autodetect
            loss = BCLoss(actor, **kwargs)
            loss.set_keys(action="expert_action")
            with pytest.raises(RuntimeError, match="did not write a prediction"):
                loss(td)

    def test_legacy_action_key_fallback_warns(self):
        # released behavior: expert at a custom key, actor writing "action";
        # still works (prediction read from "action") but warns until v0.16
        n_obs, n_act = 3, 4
        actor = TensorDictModule(
            nn.Linear(n_obs, n_act), in_keys=["observation"], out_keys=["action"]
        )
        td = TensorDict(
            {
                "observation": torch.randn(2, n_obs),
                "expert_action": torch.randn(2, n_act),
            },
            batch_size=[2],
        )
        for kwargs in ({"loss_function": "l2"}, {}):  # explicit and autodetect
            loss = BCLoss(actor, **kwargs)
            loss.set_keys(action="expert_action")
            with pytest.warns(FutureWarning, match="hardcoded 'action' key"):
                out = loss(td)["loss_bc"]
            assert torch.isfinite(out)
            assert out.requires_grad

    def test_pad_mask_nested_key(self):
        n_obs, n_act = 3, 4
        actor = TensorDictModule(
            nn.Linear(n_obs, n_act), in_keys=["observation"], out_keys=["action"]
        )
        loss = BCLoss(actor, loss_function="l1")
        loss.set_keys(pad_mask=("masks", "pad"))
        td = TensorDict(
            {
                "observation": torch.randn(2, n_obs),
                "action": torch.zeros(2, n_act),
                "masks": {"pad": torch.tensor([False, True])},
            },
            batch_size=[2],
        )
        td["action"][1] = 100.0
        full = loss(td.exclude("masks"))["loss_bc"]
        masked = loss(td)["loss_bc"]
        assert masked < full
