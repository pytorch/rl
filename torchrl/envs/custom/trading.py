# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Financial Trading Environment for TorchRL.

A PyTorch-native financial trading environment for reinforcement learning
applications in quantitative finance.
"""

import math

import torch
from tensordict import TensorDict, TensorDictBase

from torchrl.data.tensor_specs import (
    Binary,
    Categorical,
    Composite,
    UnboundedContinuous,
    UnboundedDiscrete,
)
from torchrl.envs.common import EnvBase
from torchrl.envs.utils import make_composite_from_td


class FinancialRegimeEnv(EnvBase):
    """A financial trading environment.

    This environment simulates financial market dynamics where an agent can take
    trading actions (Hold, Buy, Sell) based on price history and current holdings.

    Specs:
        >>> env = FinancialRegimeEnv()
        >>> env.specs
        Composite(
            output_spec: Composite(
                full_observation_spec: Composite(
                    price_history: UnboundedContinuous(
                        shape=torch.Size([50]),
                        device=cpu,
                        dtype=torch.float32,
                        domain=continuous),
                    current_holdings: BinaryDiscrete(
                        shape=torch.Size([1]),
                        device=cpu,
                        dtype=torch.bool,
                        domain=discrete),
                    entry_price: UnboundedContinuous(
                        shape=torch.Size([1]),
                        device=cpu,
                        dtype=torch.float32,
                        domain=continuous),
                    step_count: UnboundedDiscrete(
                        shape=torch.Size([]),
                        device=cpu,
                        dtype=torch.int64,
                        domain=discrete),
                    params: Composite(...),
                    device=None,
                    shape=torch.Size([])),
                full_reward_spec: Composite(
                    reward: UnboundedContinuous(
                        shape=torch.Size([1]),
                        device=cpu,
                        dtype=torch.float32,
                        domain=continuous),
                    device=None,
                    shape=torch.Size([])),
                full_done_spec: Composite(
                    done: Categorical(
                        shape=torch.Size([1]),
                        device=cpu,
                        dtype=torch.bool,
                        domain=discrete),
                    terminated: Categorical(
                        shape=torch.Size([1]),
                        device=cpu,
                        dtype=torch.bool,
                        domain=discrete),
                    truncated: Categorical(
                        shape=torch.Size([1]),
                        device=cpu,
                        dtype=torch.bool,
                        domain=discrete),
                    device=None,
                    shape=torch.Size([])),
                input_spec: Composite(
                    full_action_spec: Composite(
                        action: Categorical(
                            shape=torch.Size([1]),
                            space=CategoricalBox(n=3),
                            device=cpu,
                            dtype=torch.long,
                            domain=discrete),
                        device=None,
                        shape=torch.Size([])),
                    device=None,
                    shape=torch.Size([])),
                device=None,
                shape=torch.Size([]))
    """

    DEFAULT_WINDOW_SIZE = 50
    DEFAULT_INITIAL_PRICE = 100.0
    DEFAULT_MAX_STEPS = 252  # One trading year

    metadata = {
        "render_modes": ["human"],
        "render_fps": 1,
    }
    batch_locked = False

    def __init__(self, td_params=None, seed=None, device=None, window_size=None):
        """Initialize the FinancialRegimeEnv.

        Args:
            td_params: TensorDict containing environment parameters
            seed: Random seed for reproducibility
            device: PyTorch device for computations
            window_size: Size of price history window (default: 50)
        """
        if td_params is None:
            td_params = self.gen_params(device=device, window_size=window_size)

        super().__init__(device=device)
        self._make_spec(td_params)

        if seed is None:
            self.set_seed(0)
        else:
            self.set_seed(seed)

    def _step(self, tensordict):
        """Execute one step in the environment.

        Args:
            tensordict: Input TensorDict containing current state and action

        Returns:
            TensorDict containing next state, reward, and done flag
        """
        # Extract current state
        price_history = tensordict["price_history"]
        current_holdings = tensordict["current_holdings"]
        entry_price = tensordict["entry_price"]
        action = tensordict["action"].squeeze(-1)  # 0=Hold, 1=Buy, 2=Sell
        step_count = tensordict.get(
            "step_count", torch.zeros_like(action, dtype=torch.long)
        )

        # Environment parameters
        params = tensordict["params"]
        volatility = params["volatility"]
        drift = params["drift"]
        transaction_cost = params["transaction_cost"]
        max_steps = params["max_steps"]

        # Generate next price using geometric Brownian motion
        current_price = price_history[..., -1]
        dt = 1.0 / 252.0  # Daily time step

        # Use seeded generator for reproducibility
        # FIX: Pass shape directly (not unpacked) to handle scalar tensors safely
        random_shock = (
            torch.randn(current_price.shape, generator=self.rng, device=self.device)
            * volatility
            * math.sqrt(dt)
        )
        price_return = drift * dt + random_shock
        next_price = current_price * torch.exp(price_return)

        # Update price history (shift left and add new price)
        new_price_history = torch.cat(
            [
                price_history[..., 1:],  # Remove oldest price
                next_price.unsqueeze(-1),  # Add newest price
            ],
            dim=-1,
        )

        # Reward Calculation
        reward = torch.zeros_like(current_price)
        new_holdings = current_holdings.clone()
        new_entry_price = entry_price.clone()

        # Action 1: Buy
        buy_mask = (action == 1) & (~current_holdings.squeeze(-1))
        if buy_mask.any():
            new_holdings[buy_mask, 0] = True
            new_entry_price[buy_mask, 0] = current_price[buy_mask]
            # Pay transaction cost
            reward[buy_mask] -= transaction_cost[buy_mask] * current_price[buy_mask]

        # Action 2: Sell
        sell_mask = (action == 2) & (current_holdings.squeeze(-1))
        if sell_mask.any():
            new_holdings[sell_mask, 0] = False
            # Calculate P&L: (Exit Price - Entry Price) / Entry Price
            # Add epsilon to avoid division by zero if entry_price is somehow 0
            safe_entry = torch.clamp(entry_price[sell_mask, 0], min=1e-6)
            trade_return = (current_price[sell_mask] - safe_entry) / safe_entry
            reward[sell_mask] += trade_return
            # Pay transaction cost
            reward[sell_mask] -= transaction_cost[sell_mask] * current_price[sell_mask]
            # Reset entry price
            new_entry_price[sell_mask, 0] = 0.0

        # Holding reward (optional shaping)
        holding_mask = new_holdings.squeeze(-1)
        if holding_mask.any():
            # Small reward for holding winning positions (unrealized gain)
            # This is a shaping reward
            current_return = (
                next_price[holding_mask] - current_price[holding_mask]
            ) / current_price[holding_mask]
            reward[holding_mask] += current_return * 0.1

        # Update step count
        new_step_count = step_count + 1

        # Check if episode is done
        done = new_step_count >= max_steps
        terminated = torch.zeros_like(done)  # Not terminated naturally
        truncated = done.clone()  # Truncated due to time limit

        out = TensorDict(
            {
                "price_history": new_price_history,
                "current_holdings": new_holdings,
                "entry_price": new_entry_price,
                "step_count": new_step_count,
                "params": tensordict["params"],
                "reward": reward.unsqueeze(-1),
                "done": done.unsqueeze(-1),
                "terminated": terminated.unsqueeze(-1),
                "truncated": truncated.unsqueeze(-1),
            },
            tensordict.shape,
        )
        return out

    def _reset(self, tensordict):
        batch_size = (
            tensordict.batch_size if tensordict is not None else self.batch_size
        )

        if tensordict is None or "params" not in tensordict:
            tensordict = self.gen_params(batch_size=batch_size, device=self.device)

        out = self._reset_random_data(
            tensordict.shape, batch_size, tensordict["params"]
        )
        return out

    def _reset_random_data(self, shape, batch_size, params):
        window_size = params["window_size"]
        initial_price = params["initial_price"]
        volatility = params["volatility"]

        dt = 1.0 / 252.0

        # Generate random price history
        returns = (
            torch.randn((*shape, window_size), generator=self.rng, device=self.device)
            * volatility.unsqueeze(-1)
            * math.sqrt(dt)
        )

        log_prices = torch.cumsum(returns, dim=-1)
        price_history = initial_price.unsqueeze(-1) * torch.exp(log_prices)

        current_holdings = torch.zeros(
            (*shape, 1), dtype=torch.bool, device=self.device
        )
        entry_price = torch.zeros((*shape, 1), dtype=torch.float32, device=self.device)
        step_count = torch.zeros(shape, dtype=torch.long, device=self.device)

        out = TensorDict(
            {
                "price_history": price_history,
                "current_holdings": current_holdings,
                "entry_price": entry_price,
                "step_count": step_count,
                "params": params,
            },
            batch_size=batch_size,
        )
        return out

    def _make_spec(self, td_params):
        window_size = int(td_params["params", "window_size"].item())

        self.observation_spec = Composite(
            price_history=UnboundedContinuous(
                shape=(window_size,),
                dtype=torch.float32,
            ),
            current_holdings=Binary(
                shape=(1,),
                dtype=torch.bool,
            ),
            entry_price=UnboundedContinuous(
                shape=(1,),
                dtype=torch.float32,
            ),
            step_count=UnboundedDiscrete(
                shape=(),
                dtype=torch.long,
            ),
            params=make_composite_from_td(
                td_params["params"], unsqueeze_null_shapes=False
            ),
            shape=(),
        )

        self.state_spec = self.observation_spec.clone()

        self.done_spec = Composite(
            done=Categorical(n=2, shape=(1,), dtype=torch.bool),
            terminated=Categorical(n=2, shape=(1,), dtype=torch.bool),
            truncated=Categorical(n=2, shape=(1,), dtype=torch.bool),
        )

        self.action_spec = Categorical(
            n=3,
            shape=(1,),
            dtype=torch.long,
        )

        self.reward_spec = UnboundedContinuous(
            shape=(*td_params.shape, 1),
            dtype=torch.float32,
        )

    def _set_seed(self, seed: int) -> None:
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        self.rng = rng

    @staticmethod
    def gen_params(
        window_size=None,
        initial_price=100.0,
        volatility=0.2,
        drift=0.05,
        transaction_cost=0.001,
        max_steps=252,
        batch_size=None,
        device=None,
    ) -> TensorDictBase:
        if window_size is None:
            window_size = FinancialRegimeEnv.DEFAULT_WINDOW_SIZE
        if batch_size is None:
            batch_size = []

        td = TensorDict(
            {
                "params": TensorDict(
                    {
                        "window_size": window_size,
                        "initial_price": initial_price,
                        "volatility": volatility,
                        "drift": drift,
                        "transaction_cost": transaction_cost,
                        "max_steps": max_steps,
                    },
                    [],
                )
            },
            [],
            device=device,
        )

        if batch_size:
            td = td.expand(batch_size).contiguous()

        return td
