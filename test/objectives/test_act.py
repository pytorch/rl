# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from torchrl.modules.models import ACTModel
from torchrl.objectives import ACTLoss


# ── Shared helpers ─────────────────────────────────────────────────────────────

OBS_DIM = 14
ACTION_DIM = 7
CHUNK_SIZE = 10  # small for fast tests


def _make_actor(
    obs_dim=OBS_DIM, action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE, **kwargs
):
    model = ACTModel(
        obs_dim=obs_dim, action_dim=action_dim, chunk_size=chunk_size, **kwargs
    )
    return TensorDictModule(
        model,
        in_keys=["observation", "action_chunk"],
        out_keys=["action_pred", "mu", "log_var"],
    )


def _make_batch(
    batch_size=4, obs_dim=OBS_DIM, action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE
):
    return TensorDict(
        {
            "observation": torch.randn(batch_size, obs_dim),
            "action_chunk": torch.randn(batch_size, chunk_size, action_dim),
        },
        batch_size=[batch_size],
    )


# ── ACTModel unit tests ────────────────────────────────────────────────────────


class TestACTModel:
    def test_training_output_shapes(self):
        model = ACTModel(OBS_DIM, ACTION_DIM, CHUNK_SIZE)
        obs = torch.randn(4, OBS_DIM)
        chunk = torch.randn(4, CHUNK_SIZE, ACTION_DIM)
        action_pred, mu, log_var = model(obs, chunk)
        assert action_pred.shape == (4, CHUNK_SIZE, ACTION_DIM)
        assert mu.shape == (4, model.latent_dim)
        assert log_var.shape == (4, model.latent_dim)

    def test_inference_output_shapes(self):
        model = ACTModel(OBS_DIM, ACTION_DIM, CHUNK_SIZE)
        obs = torch.randn(4, OBS_DIM)
        action_pred, mu, log_var = model(obs)
        assert action_pred.shape == (4, CHUNK_SIZE, ACTION_DIM)
        assert mu.shape == (4, model.latent_dim)
        assert log_var.shape == (4, model.latent_dim)

    def test_inference_prior_is_zero(self):
        """At inference time (no action_chunk), mu and log_var must be zero."""
        model = ACTModel(OBS_DIM, ACTION_DIM, CHUNK_SIZE)
        obs = torch.randn(4, OBS_DIM)
        _, mu, log_var = model(obs)
        assert mu.eq(0).all(), "mu should be zero at inference"
        assert log_var.eq(0).all(), "log_var should be zero at inference"

    def test_training_vs_inference_differ(self):
        """Training and inference action preds should differ (different z)."""
        torch.manual_seed(42)
        model = ACTModel(OBS_DIM, ACTION_DIM, CHUNK_SIZE)
        obs = torch.randn(1, OBS_DIM)
        chunk = torch.randn(1, CHUNK_SIZE, ACTION_DIM)
        pred_train, _, _ = model(obs, chunk)
        pred_infer, _, _ = model(obs)
        assert not torch.allclose(pred_train, pred_infer)

    def test_backward_training(self):
        model = ACTModel(OBS_DIM, ACTION_DIM, CHUNK_SIZE)
        obs = torch.randn(4, OBS_DIM)
        chunk = torch.randn(4, CHUNK_SIZE, ACTION_DIM)
        action_pred, mu, log_var = model(obs, chunk)
        loss = action_pred.sum() + mu.sum() + log_var.sum()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batch_sizes(self, batch_size):
        model = ACTModel(OBS_DIM, ACTION_DIM, CHUNK_SIZE)
        obs = torch.randn(batch_size, OBS_DIM)
        chunk = torch.randn(batch_size, CHUNK_SIZE, ACTION_DIM)
        action_pred, mu, log_var = model(obs, chunk)
        assert action_pred.shape == (batch_size, CHUNK_SIZE, ACTION_DIM)

    @pytest.mark.parametrize(
        "obs_dim,action_dim,chunk_size",
        [(8, 4, 5), (14, 7, 10), (32, 6, 20)],
    )
    def test_various_dims(self, obs_dim, action_dim, chunk_size):
        model = ACTModel(obs_dim, action_dim, chunk_size)
        obs = torch.randn(2, obs_dim)
        chunk = torch.randn(2, chunk_size, action_dim)
        action_pred, mu, log_var = model(obs, chunk)
        assert action_pred.shape == (2, chunk_size, action_dim)

    def test_custom_hidden_dim(self):
        model = ACTModel(OBS_DIM, ACTION_DIM, CHUNK_SIZE, hidden_dim=128, nheads=4)
        obs = torch.randn(2, OBS_DIM)
        chunk = torch.randn(2, CHUNK_SIZE, ACTION_DIM)
        action_pred, _, _ = model(obs, chunk)
        assert action_pred.shape == (2, CHUNK_SIZE, ACTION_DIM)

    def test_custom_latent_dim(self):
        model = ACTModel(OBS_DIM, ACTION_DIM, CHUNK_SIZE, latent_dim=16)
        obs = torch.randn(2, OBS_DIM)
        chunk = torch.randn(2, CHUNK_SIZE, ACTION_DIM)
        _, mu, log_var = model(obs, chunk)
        assert mu.shape == (2, 16)
        assert log_var.shape == (2, 16)

    @pytest.mark.parametrize("batch_shape", [(2, 3), (2, 3, 4)])
    def test_multi_dim_batch(self, batch_shape):
        """Forward must handle arbitrary leading batch dims via the flat_b path."""
        model = ACTModel(OBS_DIM, ACTION_DIM, CHUNK_SIZE)
        obs = torch.randn(*batch_shape, OBS_DIM)
        chunk = torch.randn(*batch_shape, CHUNK_SIZE, ACTION_DIM)
        action_pred, mu, log_var = model(obs, chunk)
        assert action_pred.shape == (*batch_shape, CHUNK_SIZE, ACTION_DIM)
        assert mu.shape == (*batch_shape, model.latent_dim)
        assert log_var.shape == (*batch_shape, model.latent_dim)
        # Inference path through the same shapes
        action_pred_i, mu_i, log_var_i = model(obs)
        assert action_pred_i.shape == (*batch_shape, CHUNK_SIZE, ACTION_DIM)
        assert mu_i.shape == (*batch_shape, model.latent_dim)
        assert log_var_i.shape == (*batch_shape, model.latent_dim)

    def test_inference_zeros_are_independent(self):
        """At inference, mu/log_var must not share storage (no aliasing)."""
        model = ACTModel(OBS_DIM, ACTION_DIM, CHUNK_SIZE)
        obs = torch.randn(2, OBS_DIM)
        _, mu, log_var = model(obs)
        assert mu.data_ptr() != log_var.data_ptr()

    def test_state_dict_roundtrip(self):
        """save/load must not depend on the non-persistent enc_pos buffer."""
        torch.manual_seed(0)
        src = ACTModel(OBS_DIM, ACTION_DIM, CHUNK_SIZE)
        dst = ACTModel(OBS_DIM, ACTION_DIM, CHUNK_SIZE)
        # Sanity: parameters differ before load.
        for p_src, p_dst in zip(src.parameters(), dst.parameters()):
            if p_src.numel() and not torch.equal(p_src, p_dst):
                break
        else:
            pytest.skip("randomly initialized models happened to match")
        # state_dict should not contain enc_pos (registered as non-persistent).
        sd = src.state_dict()
        assert "enc_pos" not in sd
        dst.load_state_dict(sd)
        for p_src, p_dst in zip(src.parameters(), dst.parameters()):
            torch.testing.assert_close(p_src, p_dst)
        # And the (non-persistent) enc_pos buffer is still usable on dst.
        # Use inference mode (z=0, no reparameterisation noise) for a
        # deterministic comparison.
        src.eval()
        dst.eval()
        obs = torch.randn(2, OBS_DIM)
        a_src, _, _ = src(obs)
        a_dst, _, _ = dst(obs)
        torch.testing.assert_close(a_src, a_dst)

    def test_sinusoidal_pos_enc_rejects_odd_dim(self):
        from torchrl.modules.models.act import _sinusoidal_pos_enc

        with pytest.raises(ValueError, match="even"):
            _sinusoidal_pos_enc(4, 7)


# ── ACTLoss unit tests ─────────────────────────────────────────────────────────


class TestACTLoss:
    def test_output_keys(self):
        actor = _make_actor()
        loss_fn = ACTLoss(actor)
        td = _make_batch()
        loss_td = loss_fn(td)
        assert set(loss_td.keys()) == {"loss_act", "loss_reconstruction", "loss_kl"}

    def test_loss_is_scalar(self):
        actor = _make_actor()
        loss_fn = ACTLoss(actor)
        td = _make_batch()
        loss_td = loss_fn(td)
        assert loss_td["loss_act"].shape == torch.Size([])

    def test_backward(self):
        actor = _make_actor()
        loss_fn = ACTLoss(actor)
        td = _make_batch()
        loss_td = loss_fn(td)
        loss_td["loss_act"].backward()
        grads = [p.grad for p in actor.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients flowed to actor parameters"

    def test_gradients_nonzero(self):
        actor = _make_actor()
        loss_fn = ACTLoss(actor)
        td = _make_batch()
        loss_td = loss_fn(td)
        loss_td["loss_act"].backward()
        for p in actor.parameters():
            if p.grad is not None:
                assert p.grad.abs().sum() > 0

    @pytest.mark.parametrize("kl_weight", [0.0, 1.0, 10.0, 100.0])
    def test_kl_weight_decomposition(self, kl_weight):
        """loss_act == loss_reconstruction + kl_weight * loss_kl."""
        actor = _make_actor()
        loss_fn = ACTLoss(actor, kl_weight=kl_weight)
        td = _make_batch()
        loss_td = loss_fn(td)
        expected = loss_td["loss_reconstruction"] + kl_weight * loss_td["loss_kl"]
        torch.testing.assert_close(loss_td["loss_act"], expected)

    def test_zero_kl_weight(self):
        """With kl_weight=0, loss_act equals loss_reconstruction exactly."""
        actor = _make_actor()
        loss_fn = ACTLoss(actor, kl_weight=0.0)
        td = _make_batch()
        loss_td = loss_fn(td)
        torch.testing.assert_close(loss_td["loss_act"], loss_td["loss_reconstruction"])

    def test_reconstruction_and_kl_detached(self):
        """loss_reconstruction and loss_kl must not retain grad."""
        actor = _make_actor()
        loss_fn = ACTLoss(actor)
        td = _make_batch()
        loss_td = loss_fn(td)
        assert not loss_td["loss_reconstruction"].requires_grad
        assert not loss_td["loss_kl"].requires_grad

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_reduction(self, reduction):
        actor = _make_actor()
        loss_fn = ACTLoss(actor, reduction=reduction)
        td = _make_batch()
        loss_td = loss_fn(td)
        assert loss_td["loss_act"].shape == torch.Size([])
        assert loss_td["loss_act"].isfinite()

    def test_in_keys(self):
        actor = _make_actor()
        loss_fn = ACTLoss(actor)
        assert "observation" in loss_fn.in_keys
        assert "action_chunk" in loss_fn.in_keys

    def test_out_keys(self):
        actor = _make_actor()
        loss_fn = ACTLoss(actor)
        assert "loss_act" in loss_fn.out_keys
        assert "loss_reconstruction" in loss_fn.out_keys
        assert "loss_kl" in loss_fn.out_keys

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batch_sizes(self, batch_size):
        actor = _make_actor()
        loss_fn = ACTLoss(actor)
        td = _make_batch(batch_size=batch_size)
        loss_td = loss_fn(td)
        assert loss_td["loss_act"].shape == torch.Size([])

    def test_loss_decreases_with_training(self):
        """loss_act should decrease over a few gradient steps on a fixed batch."""
        torch.manual_seed(0)
        actor = _make_actor()
        # kl_weight=1.0 (vs the 10.0 default) keeps the KL term from
        # dominating early steps so the total loss decreases monotonically
        # within the short horizon used here.
        loss_fn = ACTLoss(actor, kl_weight=1.0)
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        td = _make_batch(batch_size=16)

        initial_loss = loss_fn(td)["loss_act"].item()
        for _ in range(30):
            optimizer.zero_grad()
            loss_fn(td)["loss_act"].backward()
            optimizer.step()
        final_loss = loss_fn(td)["loss_act"].item()

        assert (
            final_loss < initial_loss
        ), f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_set_keys(self):
        actor = _make_actor()
        loss_fn = ACTLoss(actor)
        loss_fn.set_keys(observation="obs", action_chunk="demo_actions")
        td = TensorDict(
            {
                "obs": torch.randn(4, OBS_DIM),
                "demo_actions": torch.randn(4, CHUNK_SIZE, ACTION_DIM),
            },
            batch_size=[4],
        )
        loss_td = loss_fn(td)
        assert loss_td["loss_act"].isfinite()

    def test_reduction_none(self):
        actor = _make_actor()
        loss_fn = ACTLoss(actor, reduction="none")
        td = _make_batch(batch_size=4)
        loss_td = loss_fn(td)
        assert loss_td["loss_act"].shape == torch.Size([4])
        assert loss_td["loss_reconstruction"].shape == torch.Size([4])
        assert loss_td["loss_kl"].shape == torch.Size([4])

    @pytest.mark.parametrize("batch_shape", [(2, 3), (2, 3, 4)])
    def test_multi_dim_batch(self, batch_shape):
        """Per-element losses must match ``tensordict.batch_size`` shape."""
        actor = _make_actor()
        loss_fn = ACTLoss(actor, reduction="none")
        td = TensorDict(
            {
                "observation": torch.randn(*batch_shape, OBS_DIM),
                "action_chunk": torch.randn(*batch_shape, CHUNK_SIZE, ACTION_DIM),
            },
            batch_size=list(batch_shape),
        )
        loss_td = loss_fn(td)
        assert loss_td["loss_act"].shape == torch.Size(batch_shape)
        assert loss_td["loss_reconstruction"].shape == torch.Size(batch_shape)
        assert loss_td["loss_kl"].shape == torch.Size(batch_shape)

    def test_reset_parameters_recursive(self):
        actor = _make_actor()
        loss_fn = ACTLoss(actor)
        params_before = [p.clone() for p in loss_fn.parameters()]
        loss_fn.reset_parameters_recursive()
        params_after = list(loss_fn.parameters())
        assert any(not torch.equal(a, b) for a, b in zip(params_before, params_after))

    @pytest.mark.parametrize(
        "obs_dim,action_dim,chunk_size",
        [(1, 1, 1), (3, 2, 1), (8, 4, 5), (32, 6, 20)],
    )
    def test_edge_case_dims(self, obs_dim, action_dim, chunk_size):
        actor = _make_actor(
            obs_dim=obs_dim, action_dim=action_dim, chunk_size=chunk_size
        )
        loss_fn = ACTLoss(actor)
        td = _make_batch(
            batch_size=2, obs_dim=obs_dim, action_dim=action_dim, chunk_size=chunk_size
        )
        loss_td = loss_fn(td)
        assert loss_td["loss_act"].isfinite()
