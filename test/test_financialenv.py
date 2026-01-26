import torch

from torchrl.envs.custom.trading import FinancialRegimeEnv
from torchrl.envs.utils import check_env_specs


class TestFinancialRegimeEnv:
    def test_specs(self):
        """Check that the environment specs match TorchRL standards."""
        env = FinancialRegimeEnv()
        check_env_specs(env)

    def test_trading_logic(self):
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

    def test_rollout(self):
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

    def test_device_compatibility(self):
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
