import torch

from torchrl.envs.custom.trading import FinancialRegimeEnv
from torchrl.envs.utils import check_env_specs


def test_specs():
    env = FinancialRegimeEnv()
    check_env_specs(env)


def test_trading_logic():
    env = FinancialRegimeEnv()
    td = env.reset()

    # Define your sequence: Buy (1), Hold (0), Sell (2)

    # Step 1: BUY
    td["action"] = torch.tensor([1], dtype=torch.long)
    td = env.step(td)
    td = td["next"]

    # Assert Holdings become True
    assert td["current_holdings"].item() is True, "Failed to acquire holdings on BUY"

    # Step 2: HOLD
    td["action"] = torch.tensor([0], dtype=torch.long)
    td = env.step(td)
    td = td["next"]

    # Assert Holdings stay True
    assert td["current_holdings"].item() is True, "Holdings lost during HOLD"

    # Step 3: SELL
    td["action"] = torch.tensor([2], dtype=torch.long)
    td = env.step(td)
    td = td["next"]

    # Assert Holdings become False
    assert td["current_holdings"].item() is False, "Failed to clear holdings on SELL"
