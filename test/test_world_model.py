# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import TensorDictReplayBuffer
from torchrl.modules import WorldModel
from torchrl.objectives import WorldModelLoss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OBS_DIM = 8
LATENT_DIM = 4
ACTION_DIM = 2
BATCH = 3


class _CatLinear(torch.nn.Module):
    """Concatenates all positional inputs along the last dim, then applies Linear."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat(tensors, dim=-1))


def _make_linear_world_model(
    with_done_head: bool = False, with_decoder: bool = False
) -> WorldModel:
    encoder = TensorDictModule(
        torch.nn.Linear(OBS_DIM, LATENT_DIM),
        in_keys=["observation"],
        out_keys=["latent"],
    )
    dynamics = TensorDictModule(
        _CatLinear(LATENT_DIM + ACTION_DIM, LATENT_DIM),
        in_keys=["latent", "action"],
        out_keys=[("next", "latent")],
    )
    reward_head = TensorDictModule(
        torch.nn.Linear(LATENT_DIM, 1),
        in_keys=[("next", "latent")],
        out_keys=[("next", "reward")],
    )
    done_head = None
    if with_done_head:
        done_head = TensorDictModule(
            torch.nn.Linear(LATENT_DIM, 1),
            in_keys=[("next", "latent")],
            out_keys=[("next", "done")],
        )
    decoder = None
    if with_decoder:
        decoder = TensorDictModule(
            torch.nn.Linear(LATENT_DIM, OBS_DIM),
            in_keys=["latent"],
            out_keys=["reconstructed_observation"],
        )
    return WorldModel(
        encoder, dynamics, reward_head, done_head=done_head, decoder=decoder
    )


def _make_start_td(batch: int = BATCH) -> TensorDict:
    wm = _make_linear_world_model()
    td = TensorDict({"observation": torch.randn(batch, OBS_DIM)}, batch_size=[batch])
    return wm.encode(td)


def _constant_policy(latent_dim: int = LATENT_DIM, action_dim: int = ACTION_DIM):
    return TensorDictModule(
        torch.nn.Linear(latent_dim, action_dim),
        in_keys=["latent"],
        out_keys=["action"],
    )


# ---------------------------------------------------------------------------
# WorldModel tests
# ---------------------------------------------------------------------------


class TestWorldModelForward:
    def test_output_keys_present(self):
        wm = _make_linear_world_model()
        td = TensorDict(
            {
                "observation": torch.randn(BATCH, OBS_DIM),
                "action": torch.randn(BATCH, ACTION_DIM),
            },
            batch_size=[BATCH],
        )
        out = wm(td)
        assert "latent" in out.keys()
        assert ("next", "latent") in out.keys(include_nested=True)
        assert ("next", "reward") in out.keys(include_nested=True)

    def test_encode_shortcut(self):
        wm = _make_linear_world_model()
        td = TensorDict(
            {"observation": torch.randn(BATCH, OBS_DIM)}, batch_size=[BATCH]
        )
        out = wm.encode(td)
        assert "latent" in out.keys()
        assert out["latent"].shape == (BATCH, LATENT_DIM)

    def test_step_shortcut(self):
        wm = _make_linear_world_model()
        td = TensorDict(
            {
                "latent": torch.randn(BATCH, LATENT_DIM),
                "action": torch.randn(BATCH, ACTION_DIM),
            },
            batch_size=[BATCH],
        )
        out = wm.step(td)
        assert ("next", "latent") in out.keys(include_nested=True)
        assert ("next", "reward") in out.keys(include_nested=True)

    def test_decode_shortcut(self):
        wm = _make_linear_world_model(with_decoder=True)
        td = TensorDict({"latent": torch.randn(BATCH, LATENT_DIM)}, batch_size=[BATCH])
        out = wm.decode(td)
        assert "reconstructed_observation" in out.keys()
        assert out["reconstructed_observation"].shape == (BATCH, OBS_DIM)

    def test_decode_without_decoder_raises(self):
        wm = _make_linear_world_model(with_decoder=False)
        td = TensorDict({"latent": torch.randn(BATCH, LATENT_DIM)}, batch_size=[BATCH])
        with pytest.raises(RuntimeError, match="decoder"):
            wm.decode(td)

    def test_nested_latent_key(self):
        """Exercises a nested tuple key for the latent (NestedKey requirement)."""
        encoder = TensorDictModule(
            torch.nn.Linear(OBS_DIM, LATENT_DIM),
            in_keys=["observation"],
            out_keys=[("agent", "latent")],
        )
        dynamics = TensorDictModule(
            _CatLinear(LATENT_DIM + ACTION_DIM, LATENT_DIM),
            in_keys=[("agent", "latent"), "action"],
            out_keys=[("next", "agent", "latent")],
        )
        reward_head = TensorDictModule(
            torch.nn.Linear(LATENT_DIM, 1),
            in_keys=[("next", "agent", "latent")],
            out_keys=[("next", "reward")],
        )
        wm = WorldModel(encoder, dynamics, reward_head)
        td = TensorDict(
            {
                "observation": torch.randn(BATCH, OBS_DIM),
                "action": torch.randn(BATCH, ACTION_DIM),
            },
            batch_size=[BATCH],
        )
        out = wm(td)
        assert ("next", "reward") in out.keys(include_nested=True)


class TestWorldModelRollout:
    def test_rollout_shape(self):
        wm = _make_linear_world_model()
        start_td = _make_start_td()
        policy = _constant_policy()
        horizon = 5
        out = wm.rollout(start_td, policy, horizon)
        assert out.shape == torch.Size([BATCH, horizon])

    def test_rollout_no_done_head(self):
        """Rollout with break_when_any_done=True but no done head runs full horizon."""
        wm = _make_linear_world_model(with_done_head=False)
        start_td = _make_start_td()
        policy = _constant_policy()
        out = wm.rollout(start_td, policy, horizon=4, break_when_any_done=True)
        assert out.shape == torch.Size([BATCH, 4])

    def test_rollout_break_when_done(self):
        """Rollout stops early when done is always True from the done head."""
        wm = _make_linear_world_model(with_done_head=True)
        # Override done head to always output True.
        wm.done_head = TensorDictModule(
            lambda x: torch.ones(x.shape[0], 1, dtype=torch.bool),
            in_keys=[("next", "latent")],
            out_keys=[("next", "done")],
        )
        # Rebuild _step_seq to include the new done_head.
        from tensordict.nn import TensorDictSequential

        wm._step_seq = TensorDictSequential(wm.dynamics, wm.reward_head, wm.done_head)

        start_td = _make_start_td()
        policy = _constant_policy()
        out = wm.rollout(start_td, policy, horizon=10, break_when_any_done=True)
        assert out.shape[1] == 1  # Stopped after first step.

    def test_rollout_contains_reward(self):
        wm = _make_linear_world_model()
        start_td = _make_start_td()
        policy = _constant_policy()
        out = wm.rollout(start_td, policy, horizon=3)
        assert ("next", "reward") in out.keys(include_nested=True)

    def test_rollout_replay_buffer_compatible(self):
        """Imagined rollout can be added directly to a TensorDictReplayBuffer."""
        from torchrl.data import LazyTensorStorage

        wm = _make_linear_world_model()
        start_td = _make_start_td()
        policy = _constant_policy()
        rollout_td = wm.rollout(start_td, policy, horizon=5)
        # Flatten batch+time into a single batch dimension.
        flat = rollout_td.reshape(-1)
        rb = TensorDictReplayBuffer(storage=LazyTensorStorage(max_size=100))
        rb.extend(flat)
        assert len(rb) == flat.batch_size[0]


# ---------------------------------------------------------------------------
# WorldModelLoss tests
# ---------------------------------------------------------------------------


def _make_real_batch(with_done: bool = False) -> TensorDict:
    data = {
        "observation": torch.randn(BATCH, OBS_DIM),
        "action": torch.randn(BATCH, ACTION_DIM),
        "next": {
            "reward": torch.randn(BATCH, 1),
            "latent": torch.randn(BATCH, LATENT_DIM),
        },
    }
    if with_done:
        data["next"]["done"] = torch.zeros(BATCH, 1, dtype=torch.bool)
        data["next"]["terminated"] = torch.zeros(BATCH, 1, dtype=torch.bool)
    return TensorDict(data, batch_size=[BATCH])


class TestWorldModelLoss:
    def test_reward_loss_only(self):
        wm = _make_linear_world_model()
        loss = WorldModelLoss(wm, losses=["reward"])
        batch = _make_real_batch()
        td_out = loss(batch)
        assert "loss_reward" in td_out.keys()
        assert td_out["loss_reward"].shape == torch.Size([])

    def test_reconstruction_loss(self):
        wm = _make_linear_world_model(with_decoder=True)
        loss = WorldModelLoss(wm, losses=["reward", "reconstruction"])
        batch = _make_real_batch()
        td_out = loss(batch)
        assert "loss_reward" in td_out.keys()
        assert "loss_reconstruction" in td_out.keys()

    def test_latent_loss(self):
        # Add a predicted_latent and target_latent key to the world model output.
        encoder = TensorDictModule(
            torch.nn.Linear(OBS_DIM, LATENT_DIM),
            in_keys=["observation"],
            out_keys=["latent"],
        )
        dynamics = TensorDictModule(
            _CatLinear(LATENT_DIM + ACTION_DIM, LATENT_DIM),
            in_keys=["latent", "action"],
            out_keys=["predicted_latent"],
        )
        reward_head = TensorDictModule(
            torch.nn.Linear(LATENT_DIM, 1),
            in_keys=["predicted_latent"],
            out_keys=[("next", "reward")],
        )
        wm = WorldModel(encoder, dynamics, reward_head)
        loss = WorldModelLoss(wm, losses=["reward", "latent"])
        batch = TensorDict(
            {
                "observation": torch.randn(BATCH, OBS_DIM),
                "action": torch.randn(BATCH, ACTION_DIM),
                "predicted_latent": torch.randn(BATCH, LATENT_DIM),
                "target_latent": torch.randn(BATCH, LATENT_DIM),
                "next": {"reward": torch.randn(BATCH, 1)},
            },
            batch_size=[BATCH],
        )
        td_out = loss(batch)
        assert "loss_latent" in td_out.keys()

    def test_unknown_loss_raises(self):
        wm = _make_linear_world_model()
        with pytest.raises(ValueError, match="Unknown loss type"):
            WorldModelLoss(wm, losses=["bad_loss"])

    def test_set_keys(self):
        wm = _make_linear_world_model()
        loss = WorldModelLoss(wm, losses=["reward"])
        loss.set_keys(reward="my_reward", true_reward="my_true_reward")
        assert loss.tensor_keys.reward == "my_reward"
        assert loss.tensor_keys.true_reward == "my_true_reward"

    def test_weights_applied(self):
        wm = _make_linear_world_model()
        loss_1x = WorldModelLoss(wm, losses=["reward"], reward_weight=1.0)
        loss_2x = WorldModelLoss(wm, losses=["reward"], reward_weight=2.0)
        batch = _make_real_batch()
        out_1x = loss_1x(batch)
        out_2x = loss_2x(batch)
        assert torch.allclose(out_2x["loss_reward"], 2.0 * out_1x["loss_reward"])

    def test_done_loss(self):
        wm = _make_linear_world_model(with_done_head=True)
        loss = WorldModelLoss(wm, losses=["reward", "done"])
        batch = _make_real_batch(with_done=True)
        td_out = loss(batch)
        assert "loss_done" in td_out.keys()

    def test_loss_is_differentiable(self):
        wm = _make_linear_world_model()
        loss = WorldModelLoss(wm, losses=["reward"])
        batch = _make_real_batch()
        td_out = loss(batch)
        td_out["loss_reward"].backward()
        for p in wm.parameters():
            assert p.grad is not None
