# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integration tests for the full recurrent-state lifecycle.

Covers the path policy -> SyncDataCollector -> ReplayBuffer -> loss with a
multi-trajectory batch and mid-batch ``done``, and asserts that hidden
state resets at trajectory boundaries (no leakage from one episode into
the next). Unit-level coverage lives in
``test/test_tensordictmodules.py::TestLSTMModule`` and
``test/objectives/test_values.py::TestValues::test_gae_recurrent``.
"""
from __future__ import annotations

import argparse

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import GymEnv, InitTracker, TransformedEnv
from torchrl.envs.libs.gym import _has_gym
from torchrl.modules import LSTMModule, set_recurrent_mode
from torchrl.modules.models.models import MLP
from torchrl.testing import CARTPOLE_VERSIONED


def _make_lstm_policy_and_value(obs_size: int, hidden_size: int = 16):
    lstm_module = LSTMModule(
        input_size=obs_size,
        hidden_size=hidden_size,
        in_keys=["observation", "rs_h", "rs_c"],
        out_keys=["intermediate", ("next", "rs_h"), ("next", "rs_c")],
        python_based=True,
        dropout=0,
    )
    policy_head = MLP(num_cells=[hidden_size], out_features=2)
    value_head = MLP(num_cells=[hidden_size], out_features=1)

    def _argmax(x: torch.Tensor) -> torch.Tensor:
        return x.argmax(-1)

    policy = Seq(
        lstm_module,
        Mod(policy_head, in_keys=["intermediate"], out_keys=["logits"]),
        Mod(_argmax, in_keys=["logits"], out_keys=["action"]),
    )
    value_net = Seq(
        lstm_module,
        Mod(value_head, in_keys=["intermediate"], out_keys=["state_value"]),
    )
    return lstm_module, policy, value_net


@pytest.mark.skipif(not _has_gym, reason="requires gym")
class TestLSTMLifecycleIntegration:
    """End-to-end checks for the recurrent-state lifecycle.

    The two scenarios under test:

    1. ``test_collector_buffer_loss_no_nan``: smoke the full pipeline
       (policy -> SyncDataCollector -> ReplayBuffer -> loss-style recurrent
       forward) and confirm that mid-batch ``done`` is exercised. CartPole
       is chosen for its naturally short episodes so a single rollout
       produces multi-trajectory batches.
    2. ``test_no_hidden_state_leakage_across_trajectory_boundary``: the
       rigorous boundary check — constructs two adjacent trajectories
       packed into one ``(1, T)`` batch, then asserts that the recurrent
       forward on the packed batch produces *bit-identical* outputs to a
       standalone forward on the second trajectory. If hidden state leaked
       across the ``is_init`` boundary, the two would diverge.
    """

    def test_collector_buffer_loss_no_nan(self):
        torch.manual_seed(0)

        def make_env():
            return TransformedEnv(GymEnv(CARTPOLE_VERSIONED()), InitTracker())

        probe_env = make_env()
        obs_size = probe_env.observation_spec["observation"].shape[-1]

        lstm_module, policy, value_net = _make_lstm_policy_and_value(obs_size)
        primer = lstm_module.make_tensordict_primer()

        def make_env_with_primer():
            return TransformedEnv(
                GymEnv(CARTPOLE_VERSIONED()),
                InitTracker(),
            ).append_transform(primer)

        env = make_env_with_primer()
        env.set_seed(0)

        frames_per_batch = 200  # CartPole episodes max out at ~200 steps, so we
        # expect at least one mid-batch done in most seeds.
        collector = SyncDataCollector(
            env,
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=frames_per_batch,
            reset_at_each_iter=False,
        )
        try:
            data = next(iter(collector))
        finally:
            collector.shutdown()

        # The collector should produce a batch shaped like (T,) for a single
        # env. is_init must exist and must be True at the very first step of
        # the rollout (the env was just reset).
        assert "is_init" in data.keys(), "InitTracker did not emit is_init"
        assert bool(
            data["is_init"][0].any().item()
        ), "is_init must be True at the first rollout step"

        # Recurrent state keys must have propagated from policy through
        # step_mdp; non-init steps should carry a non-zero hidden coming in.
        assert "rs_h" in data.keys(), "primer did not surface recurrent state"
        assert "rs_c" in data.keys()

        # Push the rollout into a replay buffer and pull it back. We use
        # in-order extraction (not random sampling) because the loss path
        # needs the temporal trajectory structure preserved — random
        # sampling would destroy the mid-batch ``is_init`` boundary signal.
        # Real training code with recurrent policies uses SliceSampler for
        # this; here we just round-trip to confirm the buffer preserves
        # is_init and the recurrent keys.
        buffer = ReplayBuffer(storage=LazyTensorStorage(frames_per_batch))
        buffer.extend(data)
        stored = buffer[:]
        # Restore the time dim that flat storage drops, giving (B=1, T).
        stored = stored.reshape(1, frames_per_batch)

        # is_init has shape (B, T, 1) from InitTracker; squeeze the trailing
        # dim so [:, 1:] indexes the time dim.
        is_init_flat = stored["is_init"].squeeze(-1)
        had_mid_batch_done = bool(is_init_flat[:, 1:].any().item())
        assert had_mid_batch_done, (
            "Expected at least one mid-batch trajectory boundary in a "
            "200-frame CartPole rollout; if this fails reproducibly, the "
            "test's environment or seed assumptions need updating"
        )

        # Run the value net under recurrent mode — this is what loss/GAE code
        # would do. With a mid-batch done present, this exercises the
        # split-and-pad path inside LSTMModule.forward.
        with set_recurrent_mode(True):
            out = value_net(stored.clone())

        assert "state_value" in out.keys()
        assert torch.isfinite(out["state_value"]).all(), (
            "recurrent value forward produced non-finite outputs — likely a "
            "shape mismatch in the split-and-pad path"
        )

    def test_no_hidden_state_leakage_across_trajectory_boundary(self):
        """Pack two trajectories into one batch and verify B does not see A's hidden.

        Construction:
            obs shape (1, T, obs_size). is_init = [True, False, ..., False,
            True, False, ..., False] with the second True at index t*.
            Steps [0..t*-1] are trajectory A; steps [t*..T-1] are trajectory B.

        Check:
            Running the LSTM on the packed batch under set_recurrent_mode(True)
            should produce, for indices [t*..T-1], the *same* hidden state as
            running just trajectory B (with is_init[0]=True) in isolation.

        Why bit-identical: the split-and-pad path zeros B's incoming hidden
        regardless of A's state, so an isolated B with zero initial hidden
        must match. Any divergence means A's hidden leaked into B.
        """
        torch.manual_seed(0)

        obs_size = 4
        hidden_size = 16
        T = 12
        t_star = 5  # trajectory B starts here

        lstm_module = LSTMModule(
            input_size=obs_size,
            hidden_size=hidden_size,
            in_keys=["observation", "rs_h", "rs_c"],
            out_keys=["intermediate", ("next", "rs_h"), ("next", "rs_c")],
            python_based=True,
            dropout=0,
            default_recurrent_mode=True,
        )
        lstm_module.eval()

        obs = torch.randn(1, T, obs_size)

        is_init = torch.zeros(1, T, dtype=torch.bool)
        is_init[0, 0] = True
        is_init[0, t_star] = True

        # Hidden state shape: (B, T, num_layers, hidden_size). We seed A's
        # incoming hidden with non-zero junk to make the leakage detectable —
        # if the split-and-pad path failed to zero B's incoming hidden, B's
        # output would carry traces of this.
        rs_h_in = torch.randn(1, T, lstm_module.lstm.num_layers, hidden_size)
        rs_c_in = torch.randn(1, T, lstm_module.lstm.num_layers, hidden_size)

        packed = TensorDict(
            {
                "observation": obs,
                "rs_h": rs_h_in,
                "rs_c": rs_c_in,
                "is_init": is_init,
            },
            batch_size=[1, T],
        )

        with set_recurrent_mode(True):
            packed_out = lstm_module(packed.clone())

        # Isolated trajectory B: just the [t_star..T] slice, is_init[0]=True.
        T_b = T - t_star
        is_init_b = torch.zeros(1, T_b, dtype=torch.bool)
        is_init_b[0, 0] = True
        b_alone = TensorDict(
            {
                "observation": obs[:, t_star:].clone(),
                # Same non-zero junk hidden — split-and-pad must override it.
                "rs_h": rs_h_in[:, t_star:].clone(),
                "rs_c": rs_c_in[:, t_star:].clone(),
                "is_init": is_init_b,
            },
            batch_size=[1, T_b],
        )

        with set_recurrent_mode(True):
            b_alone_out = lstm_module(b_alone.clone())

        # The intermediate (LSTM output) for trajectory B inside the packed
        # batch must equal trajectory B run alone. If hidden state leaked
        # from A, the values would differ.
        torch.testing.assert_close(
            packed_out["intermediate"][:, t_star:],
            b_alone_out["intermediate"],
            rtol=1e-5,
            atol=1e-6,
            msg="hidden state leaked across is_init trajectory boundary",
        )


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
