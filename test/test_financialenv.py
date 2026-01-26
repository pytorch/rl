import pytest
import torch
from torchrl.envs.custom.trading import FinancialRegimeEnv
from torchrl.envs.utils import check_env_specs

def test_specs():
    """Check that the environment specs match TorchRL standards."""
    env = FinancialRegimeEnv()
    check_env_specs(env)

def test_trading_logic():
    """Verify that Buy/Hold/Sell actions affect holdings and rewards correctly."""
    env = FinancialRegimeEnv()
    td = env.reset()
    
    # Buy
    td["action"] = torch.tensor([1], dtype=torch.long)
    td = env.step(td)
    td = td["next"]
    assert td["current_holdings"].item() is True
    
    # Sell
    td["action"] = torch.tensor([2], dtype=torch.long)
    td = env.step(td)
    td = td["next"]
    assert td["current_holdings"].item() is False
