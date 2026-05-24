# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Integration coverage for the recurrent-state lifecycle.

Verifies that an LSTM policy run through
``SyncDataCollector -> ReplayBuffer`` with a mid-batch trajectory boundary
does **not** leak hidden state from one episode into the next. Unit-level
coverage for ``LSTMModule`` lives in ``test/modules/test_rnn.py``;
value-function-side coverage for ``set_recurrent_mode`` lives in
``test/objectives/test_values.py::TestValues::test_gae_recurrent``. This
file fills the policy -> collector -> buffer integration gap that neither
of those exercises.
"""
from __future__ import annotations

import argparse

import pytest
import torch
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.envs import InitTracker, TransformedEnv
from torchrl.modules import LSTMModule, set_recurrent_mode
from torchrl.testing.mocking_classes import CountingEnv, CountingEnvCountPolicy


MAX_STEPS = 2  # episode runs 3 transitions before done -> very short
HIDDEN_SIZE = 8
FRAMES_PER_BATCH = 16  # >> episode length, so mid-batch dones are guaranteed


def _build_env_and_policy():
    """Set up the deterministic CountingEnv + LSTM policy used by the test.

    The LSTM rides along the rollout so its hidden state propagates and
    resets at trajectory boundaries; the actual action is driven by
    :class:`CountingEnvCountPolicy` so the env terminates on a known
    schedule and we get reliable mid-batch ``done`` events.
    """
    base_env = CountingEnv(max_steps=MAX_STEPS)
    obs_size = base_env.observation_spec["observation"].shape[-1]

    lstm_module = LSTMModule(
        input_size=obs_size,
        hidden_size=HIDDEN_SIZE,
        in_keys=["obs_float", "rs_h", "rs_c"],
        out_keys=["intermediate", ("next", "rs_h"), ("next", "rs_c")],
        python_based=True,
        dropout=0,
    )
    lstm_module.eval()  # deterministic — no dropout, no train-only behavior

    env = TransformedEnv(base_env, InitTracker())
    env = env.append_transform(lstm_module.make_tensordict_primer())
    env.set_seed(0)

    # Cast int32 observation to float so the LSTM can consume it. Kept as a
    # tiny TensorDictModule rather than an env transform so we don't conflate
    # the recurrent-state lifecycle test with transform plumbing.
    def _to_float(obs: torch.Tensor) -> torch.Tensor:
        return obs.to(torch.float32)

    cast_obs = Mod(_to_float, in_keys=["observation"], out_keys=["obs_float"])
    action_module = CountingEnvCountPolicy(action_spec=base_env.action_spec)

    policy = Seq(cast_obs, lstm_module, action_module)
    return env, policy, lstm_module


class TestLSTMRecurrentStateLifecycle:
    """End-to-end integration test for the recurrent-state lifecycle.

    One test, structured in three phases:

    1. **Pipeline smoke**: build the policy, run it through
       :class:`SyncDataCollector`, push the rollout through a
       :class:`ReplayBuffer` with a :class:`SliceSampler` that respects
       trajectory boundaries. Assert ``is_init`` survives, mid-batch
       boundaries exist, and the recurrent-state keys propagate.
    2. **Sampled-slice structure**: pull slices back through the sampler
       and confirm each slice begins at a trajectory boundary (one
       ``is_init=True`` at slice index 0, none in the interior).
    3. **No-leakage check**: pack two adjacent trajectories from the
       collected rollout into one ``(1, T)`` batch, run the LSTM under
       ``set_recurrent_mode(True)``, then run trajectory B in isolation
       with reset hidden state and assert the two produce matching
       outputs over B's time range. If hidden state from A had leaked
       through the boundary, these would diverge.
    """

    def test_lstm_collector_replay_mid_batch_done_resets_hidden_state(self):
        torch.manual_seed(0)
        env, policy, lstm_module = _build_env_and_policy()

        # --- Phase 1: collect a batch through SyncDataCollector ---------
        collector = SyncDataCollector(
            env,
            policy=policy,
            frames_per_batch=FRAMES_PER_BATCH,
            total_frames=FRAMES_PER_BATCH,
            reset_at_each_iter=False,
        )
        try:
            data = next(iter(collector))
        finally:
            collector.shutdown()

        # Structural assertions: lifecycle keys are present, mid-batch
        # trajectory boundaries actually occurred.
        assert "is_init" in data.keys(), "InitTracker did not emit is_init"
        assert "rs_h" in data.keys(), "primer did not surface recurrent state h"
        assert "rs_c" in data.keys(), "primer did not surface recurrent state c"

        is_init = data["is_init"].squeeze(-1)
        assert bool(is_init[0].item()), "is_init must be True at the first step"
        n_resets = int(is_init.sum().item())
        assert n_resets >= 2, (
            f"expected at least 2 trajectory boundaries in {FRAMES_PER_BATCH} "
            f"frames with max_steps={MAX_STEPS}, got {n_resets}"
        )

        # Recurrent state at every is_init=True position must be the
        # primer zero — this is the per-step "reset" invariant. Without
        # it the LSTM's sequential-mode reset block is broken.
        rs_h_at_inits = data["rs_h"][is_init]
        rs_c_at_inits = data["rs_c"][is_init]
        assert torch.equal(rs_h_at_inits, torch.zeros_like(rs_h_at_inits)), (
            "incoming recurrent_state_h at is_init=True positions should be "
            "the primer zero — found non-zero values, suggesting hidden state "
            "leaked across a trajectory boundary in the collector"
        )
        assert torch.equal(rs_c_at_inits, torch.zeros_like(rs_c_at_inits)), (
            "incoming recurrent_state_c at is_init=True positions should be "
            "the primer zero — see above"
        )

        # --- Phase 2: round-trip through ReplayBuffer + SliceSampler ----
        # slice_len matches one full episode (MAX_STEPS+1 transitions), so
        # each sampled slice should be exactly one trajectory.
        slice_len = MAX_STEPS + 1
        num_slices = 2
        buffer = ReplayBuffer(
            storage=LazyTensorStorage(FRAMES_PER_BATCH),
            sampler=SliceSampler(
                slice_len=slice_len,
                end_key=("next", "done"),
                strict_length=True,
            ),
        )
        buffer.extend(data)
        sampled = buffer.sample(num_slices * slice_len)
        sampled = sampled.reshape(num_slices, slice_len)

        sampled_is_init = sampled["is_init"].squeeze(-1)
        assert sampled_is_init[:, 0].all(), (
            "every SliceSampler-returned slice should begin at a trajectory "
            "boundary (is_init=True at slice index 0)"
        )
        assert not sampled_is_init[:, 1:].any(), (
            "sliced trajectories should contain no interior is_init=True; "
            "the sampler must respect end_key=('next', 'done')"
        )

        # --- Phase 3: rigorous no-leakage check -------------------------
        # Build two adjacent trajectories from the flat rollout. Lengths
        # are computed from is_init: a new trajectory starts at every
        # is_init=True, so lengths are the gaps between consecutive trues.
        init_positions = is_init.nonzero(as_tuple=False).squeeze(-1).tolist()
        # Need at least 2 complete trajectories.
        assert len(init_positions) >= 3, (
            f"need at least 3 is_init boundaries to extract 2 complete "
            f"trajectories; got positions={init_positions}"
        )
        len_a = init_positions[1] - init_positions[0]
        len_b = init_positions[2] - init_positions[1]
        packed_T = len_a + len_b
        packed = data[:packed_T].reshape(1, packed_T).clone()
        b_alone = data[len_a : len_a + len_b].reshape(1, len_b).clone()

        # Seed packed's incoming hidden with non-zero noise to make any
        # leakage detectable: if the recurrent-mode forward fails to zero
        # B's hidden at its is_init=True boundary, B's outputs will pick
        # up this noise. b_alone gets the same noise; the split-and-pad
        # path inside LSTMModule.forward must override both.
        noise_h = torch.randn_like(packed["rs_h"])
        noise_c = torch.randn_like(packed["rs_c"])
        packed["rs_h"] = noise_h
        packed["rs_c"] = noise_c
        b_alone["rs_h"] = noise_h[:, len_a:].clone()
        b_alone["rs_c"] = noise_c[:, len_a:].clone()

        with set_recurrent_mode(True):
            packed_out = lstm_module(packed)
            b_alone_out = lstm_module(b_alone)

        # Trajectory B's LSTM outputs inside the packed batch must match
        # the standalone run. Hidden-state leakage from A through the
        # is_init boundary would make these diverge.
        torch.testing.assert_close(
            packed_out["intermediate"][:, len_a:],
            b_alone_out["intermediate"],
            rtol=1e-5,
            atol=1e-6,
            msg="hidden state leaked across is_init trajectory boundary",
        )


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
