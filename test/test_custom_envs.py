# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import random

import pytest
import torch
from tensordict import TensorDict

from torchrl.envs import (
    check_env_specs,
    LLMHashingEnv,
    MultiAction,
    PendulumEnv,
    TicTacToeEnv,
    ToyVLAEnv,
    TransformedEnv,
)
from torchrl.envs.custom.trading import FinancialRegimeEnv
from torchrl.testing import get_default_devices


class TestCustomEnvs:
    def test_financial_env_specs(self):
        """Check that the environment specs match TorchRL standards."""
        env = FinancialRegimeEnv()
        check_env_specs(env)

    def test_financial_env_trading_logic(self):
        """Verify that Buy/Hold/Sell actions affect holdings and rewards correctly."""
        torch.manual_seed(0)
        env = FinancialRegimeEnv()

        for _ in range(3):  # Test multiple episodes for robustness
            td = env.reset()

            # Test Buy action
            td["action"] = torch.tensor([1], dtype=torch.long)
            td = env.step(td)
            td = td["next"]
            assert (
                td["current_holdings"].item() is True
            ), "Failed to acquire holdings on BUY"

            # Test Hold action
            td["action"] = torch.tensor([0], dtype=torch.long)
            td = env.step(td)
            td = td["next"]
            assert td["current_holdings"].item() is True, "Holdings lost during HOLD"

            # Test Sell action
            td["action"] = torch.tensor([2], dtype=torch.long)
            td = env.step(td)
            td = td["next"]
            assert (
                td["current_holdings"].item() is False
            ), "Failed to clear holdings on SELL"

    def test_financial_env_rollout(self):
        """Test environment rollouts with different configurations."""
        torch.manual_seed(0)
        env = FinancialRegimeEnv()

        # Test single environment rollout
        for _ in range(3):
            r = env.rollout(10)
            assert r.shape == torch.Size((10,))
            assert "price_history" in r
            assert "current_holdings" in r
            # Reward is in the "next" nested structure during rollout
            assert "next" in r

        # Test longer rollout without early termination
        r = env.rollout(20, break_when_any_done=False)
        assert r.shape == torch.Size((20,))

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_financial_env_device(self):
        """Test environment works on different devices."""
        # Test CPU (always available)
        env_cpu = FinancialRegimeEnv(device="cpu")
        assert env_cpu.device.type == "cpu"
        check_env_specs(env_cpu)

        # Test rollout on CPU
        r = env_cpu.rollout(5)
        assert r.shape == torch.Size((5,))

        # Test CUDA if available
        if torch.cuda.is_available():
            env_cuda = FinancialRegimeEnv(device="cuda:0")
            assert env_cuda.device.type == "cuda"
            check_env_specs(env_cuda)


class TestTicTacToeEnv:
    def test_tictactoe_env(self):
        torch.manual_seed(0)
        env = TicTacToeEnv()
        check_env_specs(env)
        for _ in range(10):
            r = env.rollout(10)
            assert r.shape[-1] < 10
            r = env.rollout(10, tensordict=TensorDict(batch_size=[5]))
            assert r.shape[-1] < 10
        r = env.rollout(
            100, tensordict=TensorDict(batch_size=[5]), break_when_any_done=False
        )
        assert r.shape == (5, 100)

    def test_tictactoe_env_single(self):
        torch.manual_seed(0)
        env = TicTacToeEnv(single_player=True)
        check_env_specs(env)
        for _ in range(10):
            r = env.rollout(10)
            assert r.shape[-1] < 6
            r = env.rollout(10, tensordict=TensorDict(batch_size=[5]))
            assert r.shape[-1] < 6
        r = env.rollout(
            100, tensordict=TensorDict(batch_size=[5]), break_when_any_done=False
        )
        assert r.shape == (5, 100)


class TestPendulum:
    @pytest.mark.parametrize("device", [None, *get_default_devices()])
    def test_pendulum_env(self, device):
        env = PendulumEnv(device=device)
        assert env.device == device
        check_env_specs(env)

        for _ in range(10):
            r = env.rollout(10)
            assert r.shape == torch.Size((10,))
            r = env.rollout(10, tensordict=TensorDict(batch_size=[5], device=device))
            assert r.shape == torch.Size((5, 10))

    def test_pendulum_angle_wrapped(self):
        # ``th`` must stay within the ``Bounded(-pi, pi)`` observation/state
        # spec even when the dynamics push the angle past the boundary;
        # unbounded drift made ``check_env_specs`` flaky (#3798).
        env = PendulumEnv()
        td = env.reset()
        td["th"] = torch.tensor(torch.pi - 1e-3)
        td["thdot"] = torch.tensor(8.0)  # max speed, pushes past +pi
        td["action"] = torch.ones(1)
        for _ in range(10):
            td = env.step(td)["next"]
            assert env.observation_spec["th"].is_in(td["th"]), td["th"]
            td["action"] = torch.ones(1)

    def test_pendulum_set_state(self):
        env = PendulumEnv()
        torch.manual_seed(0)
        state = env.reset()
        th_target = state["th"].clone()
        thdot_target = state["thdot"].clone()

        # set_state=True honors the provided state (deterministic reset)
        out = env.reset(state.clone(), set_state=True)
        assert torch.allclose(out["th"], th_target)
        assert torch.allclose(out["thdot"], thdot_target)
        # the returned tensordict must be a distinct object from the input
        assert out is not state

        # set_state=False ignores the provided state (fresh random reset)
        torch.manual_seed(1)
        out_false = env.reset(state.clone(), set_state=False)
        assert not torch.allclose(out_false["th"], th_target)

        # set_state=True + select_reset_only is contradictory
        with pytest.raises(ValueError, match="select_reset_only"):
            env.reset(state.clone(), set_state=True, select_reset_only=True)

        # rollout threads set_state to the initial reset
        r = env.rollout(4, tensordict=state.clone(), set_state=True)
        assert torch.allclose(r["th"][0], th_target)

    def test_pendulum_set_state_transition_warning(self):
        env = PendulumEnv()
        state = env.reset()
        th_target = state["th"].clone()
        # unspecified set_state with state in the tensordict -> FutureWarning,
        # but the state is still honored (backwards-compatible behavior).
        with pytest.warns(FutureWarning, match="set_state"):
            out = env.reset(state.clone())
        assert torch.allclose(out["th"], th_target)

        # an empty tensordict (batch-size only) must not trigger the warning.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            env.reset(TensorDict(batch_size=[4]))
            env.rollout(3, tensordict=TensorDict(batch_size=[2]))

    def test_llm_hashing_env(self):
        vocab_size = 5

        class Tokenizer:
            def __call__(self, obj):
                return torch.randint(vocab_size, (len(obj.split(" ")),)).tolist()

            def decode(self, obj):
                words = ["apple", "banana", "cherry", "date", "elderberry"]
                return " ".join(random.choice(words) for _ in obj)

            def batch_decode(self, obj):
                return [self.decode(_obj) for _obj in obj]

            def encode(self, obj):
                return self(obj)

        tokenizer = Tokenizer()
        env = LLMHashingEnv(tokenizer=tokenizer, vocab_size=vocab_size)
        td = env.make_tensordict("some sentence")
        assert isinstance(td, TensorDict)
        env.check_env_specs(tensordict=td)


class TestToyVLAEnv:
    @pytest.mark.parametrize("batch_size", [(), (2,)])
    def test_env_specs(self, batch_size):
        env = ToyVLAEnv(batch_size=batch_size)
        check_env_specs(env)
        td = env.reset()
        assert td["observation", "image"].dtype == torch.uint8
        assert td["observation", "image"].shape == torch.Size([*batch_size, 3, 16, 16])
        instruction = td["language_instruction"]
        if batch_size:
            instruction = instruction[0]
        assert isinstance(instruction, str)

    def test_state_echoes_action(self):
        env = ToyVLAEnv(action_dim=3, state_dim=5, batch_size=[2])
        td = env.reset()
        td["action"] = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        nxt = env.step(td)["next", "observation", "state"]
        torch.testing.assert_close(nxt[..., :3], td["action"])
        assert (nxt[..., 3:] == 0).all()
        # reward is the negative action norm
        rew = env.step(td)["next", "reward"]
        torch.testing.assert_close(rew, -td["action"].norm(dim=-1, keepdim=True))

    def test_seeding(self):
        img1 = ToyVLAEnv(seed=7).reset()["observation", "image"]
        img2 = ToyVLAEnv(seed=7).reset()["observation", "image"]
        img3 = ToyVLAEnv(seed=8).reset()["observation", "image"]
        assert torch.equal(img1, img2)
        assert not torch.equal(img1, img3)

    def test_state_dim_validation(self):
        with pytest.raises(ValueError, match="state_dim"):
            ToyVLAEnv(action_dim=5, state_dim=3)

    @pytest.mark.parametrize("batch_size", [(), (2,)])
    def test_tracking_specs(self, batch_size):
        env = ToyVLAEnv(
            action_dim=2, state_dim=4, success_steps=2, batch_size=batch_size
        )
        check_env_specs(env)
        td = env.reset()
        assert td["success"].dtype == torch.bool
        assert td["success"].shape == torch.Size([*batch_size, 1])
        assert not td["success"].any()

    def test_tracking_validation(self):
        with pytest.raises(ValueError, match="2 \\* action_dim"):
            ToyVLAEnv(action_dim=4, state_dim=6, success_steps=2)
        with pytest.raises(ValueError, match="success_steps"):
            ToyVLAEnv(action_dim=2, state_dim=4, success_steps=0)
        with pytest.raises(ValueError, match="success_tol"):
            ToyVLAEnv(action_dim=2, state_dim=4, success_steps=2, success_tol=0.6)

    def test_tracking_oracle_succeeds(self):
        # the acceptance gate for the tracking task: an oracle that reads the
        # target back from the state succeeds in exactly success_steps steps
        env = ToyVLAEnv(action_dim=2, state_dim=4, success_steps=3, seed=0)

        def oracle(td):
            td["action"] = td["observation", "state"][..., 2:4]
            return td

        rollout = env.rollout(10, oracle)
        assert rollout.shape[0] == 3
        assert rollout["next", "success"][-1].all()
        assert rollout["next", "terminated"][-1].all()
        assert not rollout["next", "success"][:-1].any()
        # the reward is the negative tracking error: zero for the oracle
        torch.testing.assert_close(
            rollout["next", "reward"], torch.zeros_like(rollout["next", "reward"])
        )

    def test_tracking_random_fails(self):
        # ...while a random policy essentially never does (per-step hit
        # probability success_tol ** action_dim, squared for the streak)
        torch.manual_seed(0)
        env = ToyVLAEnv(action_dim=4, state_dim=8, success_steps=2, seed=0)
        rollout = env.rollout(200)
        assert rollout.shape[0] == 200
        assert not rollout["next", "success"].any()

    def test_tracking_streak_resets_on_miss(self):
        env = ToyVLAEnv(action_dim=2, state_dim=4, success_steps=2, seed=0)
        td = env.reset()
        target = td["observation", "state"][..., 2:4]
        # in-tolerance, then a miss, then in-tolerance again: the streak
        # restarts so success only fires after two fresh consecutive hits
        actions = [target, target + 1.0, target, target]
        successes = []
        for action in actions:
            td["action"] = action.clamp(-1.0, 1.0)
            td = env.step(td)["next"]
            successes.append(bool(td["success"]))
        assert successes == [False, False, False, True]

    def test_tracking_partial_reset(self):
        env = ToyVLAEnv(action_dim=2, state_dim=4, success_steps=3, batch_size=[2])
        td = env.reset()
        target = td["observation", "state"][..., 2:4].clone()
        # both envs take one in-tolerance step
        td["action"] = target
        td = env.step(td)["next"]
        # reset env 0 only: its target resamples, env 1 keeps target and streak
        td["_reset"] = torch.tensor([[True], [False]])
        td = env.reset(td)
        new_target = td["observation", "state"][..., 2:4]
        torch.testing.assert_close(new_target[1], target[1])
        # env 1 finishes its streak (2 more hits) while env 0 starts over
        for _ in range(2):
            td["action"] = new_target.clone()
            td = env.step(td)["next"]
        assert not td["success"][0]
        assert td["success"][1]

    def test_tracking_state_buffers(self):
        # the episode state must be registered (non-persistent) buffers so
        # env.to(device) moves it along with the specs
        env = ToyVLAEnv(action_dim=2, state_dim=4, success_steps=2)
        buffers = dict(env.named_buffers())
        assert "_target" in buffers
        assert "_streak" in buffers
        assert not env.state_dict()
        # the reassignments in reset/step keep the buffers registered
        td = env.reset()
        td["action"] = td["observation", "state"][..., 2:4]
        env.step(td)
        assert "_target" in dict(env.named_buffers())
        assert "_streak" in dict(env.named_buffers())

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_tracking_to_device(self):
        env = ToyVLAEnv(action_dim=2, state_dim=4, success_steps=2, seed=0).to("cuda")
        td = env.reset()
        assert td["observation", "state"].device.type == "cuda"
        for _ in range(2):
            td["action"] = td["observation", "state"][..., 2:4]
            td = env.step(td)["next"]
        assert td["success"].item() and td["terminated"].item()

    def test_tracking_partial_step_mask(self):
        # batch-locked partial stepping (the "_step" contract MultiAction
        # relies on): once a sub-env succeeds inside a chunk, its streak is
        # frozen - done cannot flip back - and its remaining slots pay zero
        # reward, while the other sub-env keeps stepping
        env = TransformedEnv(
            ToyVLAEnv(action_dim=2, state_dim=4, success_steps=2, batch_size=[2]),
            MultiAction(),
        )
        td = env.reset()
        target = td["observation", "state"][..., 2:4]
        # env 0: two hits (success), then far off target for the rest of the
        # chunk; env 1: always far off target
        chunk = target.unsqueeze(-2).repeat(1, 4, 1)
        chunk[0, 2:] = 1.0
        chunk[1] = -1.0
        td["action"] = chunk.clamp(-1.0, 1.0)
        out = env.step(td)["next"]
        assert out["done"][0].item(), "done must persist after an in-chunk success"
        assert out["success"][0].item()
        assert not out["done"][1].item()
        # env 0's post-success slots pay zero reward
        reward = out["reward"].squeeze(-1)
        torch.testing.assert_close(reward[0, :2], torch.zeros(2))
        assert (reward[0, 2:] == 0).all()
        assert (reward[1] != 0).all()

    def test_with_multi_step_actor(self):
        from tensordict.nn import TensorDictModuleBase
        from torchrl.envs.transforms import InitTracker
        from torchrl.modules import MultiStepActorWrapper

        calls = []

        class ChunkActor(TensorDictModuleBase):
            in_keys = [("observation", "state")]
            out_keys = ["action"]

            def forward(self, td):
                calls.append(1)
                value = float(len(calls))
                chunk = torch.full((*td.batch_size, 3, 4), value)
                chunk[..., 1, :] += 0.1
                chunk[..., 2, :] += 0.2
                return td.set("action", chunk)

        env = TransformedEnv(ToyVLAEnv(batch_size=[2]), InitTracker())
        policy = MultiStepActorWrapper(ChunkActor(), n_steps=3, replan_interval=2)
        rollout = env.rollout(4, policy)
        # the state echo exposes the executed cadence: two actions per chunk,
        # then a re-plan -- and the actor was only called on re-plan steps
        executed = rollout["next", "observation", "state"][0, :, 0]
        torch.testing.assert_close(executed, torch.tensor([1.0, 1.1, 2.0, 2.1]))
        assert len(calls) == 2


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
