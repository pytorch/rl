# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import random

import pytest
import torch
from tensordict import TensorDict

from torchrl.envs import check_env_specs, LLMHashingEnv, PendulumEnv, TicTacToeEnv
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
