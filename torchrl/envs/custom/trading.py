# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Financial Trading Environment for TorchRL.

A PyTorch-native financial trading environment with market regime detection
for reinforcement learning applications in quantitative finance.
"""

import math
from enum import IntEnum

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import (
    Binary,
    Categorical,
    Composite,
    UnboundedContinuous,
)
from torchrl.envs.common import EnvBase
from torchrl.envs.utils import make_composite_from_td


class MarketRegime(IntEnum):
    """Market regime classifications as integer constants."""

    STRONG_BEAR = 0
    BEAR = 1
    CHOPPY_SIDEWAYS = 2
    CHOPPY = 3
    SIDEWAYS = 4
    BULL = 5
    STRONG_BULL = 6


class FinancialMarketDynamics:
    """PyTorch-native financial market dynamics for regime detection and risk management.

    This class provides tensor-based implementations of market regime detection,
    volatility calculations, and adaptive risk parameters for use in RL environments.
    """

    def __init__(self, device: torch.device | None = None):
        """Initialize the FinancialMarketDynamics class.

        Args:
            device: PyTorch device to use for computations (default: CPU)
        """
        self.device = device or torch.device("cpu")

        # Regime classification constants
        self.bullish_regimes = {MarketRegime.BULL, MarketRegime.STRONG_BULL}
        self.bearish_regimes = {MarketRegime.BEAR, MarketRegime.STRONG_BEAR}
        self.choppy_regimes = {MarketRegime.CHOPPY, MarketRegime.CHOPPY_SIDEWAYS}

        # Regime state tracking for transition smoothing
        self._regime_history = {}
        self._stable_regimes = {}

    def rolling_window(
        self, tensor: torch.Tensor, window_size: int, dim: int = -1
    ) -> torch.Tensor:
        """Create rolling windows from a tensor using torch.as_strided.

        Args:
            tensor: Input tensor of shape [..., sequence_length]
            window_size: Size of the rolling window
            dim: Dimension along which to create windows (default: -1)

        Returns:
            Tensor with rolling windows of shape [..., num_windows, window_size]
        """
        if dim != -1:
            tensor = tensor.transpose(dim, -1)

        shape = tensor.shape
        seq_len = shape[-1]

        if window_size > seq_len:
            raise ValueError(
                f"Window size {window_size} cannot be larger than sequence length {seq_len}"
            )

        # Calculate new shape and strides
        new_shape = shape[:-1] + (seq_len - window_size + 1, window_size)
        strides = tensor.stride()[:-1] + (tensor.stride(-1), tensor.stride(-1))

        windowed = torch.as_strided(tensor, size=new_shape, stride=strides)

        if dim != -1:
            windowed = windowed.transpose(dim, -2)

        return windowed

    def calculate_returns(self, prices: torch.Tensor, periods: int = 1) -> torch.Tensor:
        """Calculate percentage returns from price series.

        Args:
            prices: Price tensor of shape [..., sequence_length]
            periods: Number of periods for return calculation

        Returns:
            Returns tensor with NaN values for initial periods
        """
        if periods >= prices.shape[-1]:
            return torch.full_like(prices, float("nan"))

        returns = torch.full_like(prices, float("nan"))
        returns[..., periods:] = (
            prices[..., periods:] - prices[..., :-periods]
        ) / prices[..., :-periods]

        return returns

    def rolling_mean(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """Calculate rolling mean using convolution for efficiency.

        Args:
            tensor: Input tensor of shape [..., sequence_length]
            window_size: Size of the rolling window

        Returns:
            Rolling mean tensor with NaN for initial values
        """
        if window_size > tensor.shape[-1]:
            return torch.full_like(tensor, float("nan"))

        # Use 1D convolution for rolling mean
        kernel = torch.ones(window_size, device=self.device) / window_size

        # Reshape for conv1d: (batch_size, channels, sequence_length)
        original_shape = tensor.shape
        tensor_flat = tensor.view(-1, original_shape[-1]).unsqueeze(1)

        # Apply convolution with padding
        conv_result = torch.nn.functional.conv1d(
            tensor_flat, kernel.view(1, 1, -1), padding=window_size - 1
        )

        # Remove extra padding and reshape back
        conv_result = conv_result[:, 0, : original_shape[-1]]
        result = conv_result.view(original_shape)

        # Set initial values to NaN
        result[..., : window_size - 1] = float("nan")

        return result

    def rolling_std(self, tensor: torch.Tensor, window_size: int) -> torch.Tensor:
        """Calculate rolling standard deviation.

        Args:
            tensor: Input tensor of shape [..., sequence_length]
            window_size: Size of the rolling window

        Returns:
            Rolling standard deviation tensor
        """
        if window_size > tensor.shape[-1]:
            return torch.full_like(tensor, float("nan"))

        rolling_mean_vals = self.rolling_mean(tensor, window_size)

        # Calculate rolling variance using the formula: E[X²] - E[X]²
        tensor_squared = tensor**2
        rolling_mean_squared = self.rolling_mean(tensor_squared, window_size)
        rolling_variance = rolling_mean_squared - rolling_mean_vals**2

        # Handle numerical precision issues
        rolling_variance = torch.clamp(rolling_variance, min=0.0)

        return torch.sqrt(rolling_variance)

    def detect_market_regime(
        self, prices: torch.Tensor, lookback_days: int = 30
    ) -> torch.Tensor:
        """Detect market regime from price history.

        Args:
            prices: Price tensor of shape [..., sequence_length]
            lookback_days: Number of days to analyze for regime detection

        Returns:
            Regime tensor with integer values corresponding to MarketRegime enum
        """
        if prices.shape[-1] < lookback_days:
            batch_shape = prices.shape[:-1]
            return torch.full(
                batch_shape, MarketRegime.SIDEWAYS, dtype=torch.long, device=self.device
            )

        # Take recent data
        recent_prices = prices[..., -lookback_days:]

        if recent_prices.shape[-1] < 2:
            batch_shape = prices.shape[:-1]
            return torch.full(
                batch_shape, MarketRegime.SIDEWAYS, dtype=torch.long, device=self.device
            )

        # Calculate trend metrics
        start_price = recent_prices[..., 0]
        end_price = recent_prices[..., -1]

        # Avoid division by zero
        total_return = torch.where(
            start_price > 0,
            (end_price - start_price) / start_price,
            torch.zeros_like(start_price),
        )

        # Calculate moving averages
        sma_20 = self.rolling_mean(recent_prices, 20)[..., -1]
        sma_50 = (
            self.rolling_mean(recent_prices, 50)[..., -1]
            if recent_prices.shape[-1] >= 50
            else sma_20
        )

        # Calculate momentum indicators
        if recent_prices.shape[-1] >= 20:
            recent_10_mean = torch.mean(recent_prices[..., -10:], dim=-1)
            prev_10_mean = torch.mean(recent_prices[..., -20:-10], dim=-1)
            recent_momentum = torch.where(
                prev_10_mean > 0,
                (recent_10_mean - prev_10_mean) / prev_10_mean,
                torch.zeros_like(prev_10_mean),
            )
        else:
            recent_momentum = torch.zeros_like(total_return)

        trend_persistence = end_price > sma_20

        # Initialize regime tensor
        batch_shape = prices.shape[:-1]
        regime = torch.full(
            batch_shape, MarketRegime.SIDEWAYS, dtype=torch.long, device=self.device
        )

        # Regime classification based on returns and momentum
        # Strong Bull conditions
        strong_bull_cond1 = (
            (total_return > 0.20)
            & (end_price > sma_20)
            & (sma_20 > sma_50)
            & (recent_momentum > 0.05)
        )
        strong_bull_cond2 = (
            (total_return > 0.12) & trend_persistence & (recent_momentum > 0.04)
        )

        # Bull conditions
        bull_cond1 = (
            (total_return > 0.08) & (end_price > sma_20) & (recent_momentum > 0.025)
        )
        bull_cond2 = (
            (total_return > 0.04) & trend_persistence & (recent_momentum > 0.02)
        )
        bull_cond3 = (
            (total_return > 0.02) & trend_persistence & (recent_momentum > 0.015)
        )

        # Bear conditions
        strong_bear_cond = (
            (total_return < -0.25)
            & (end_price < sma_20)
            & (sma_20 < sma_50)
            & (recent_momentum < -0.05)
        )
        bear_cond = (
            (total_return < -0.15) & (end_price < sma_20) & (recent_momentum < -0.03)
        )

        # Apply regime classifications
        regime = torch.where(
            strong_bull_cond1 | strong_bull_cond2, MarketRegime.STRONG_BULL, regime
        )

        regime = torch.where(
            ~(strong_bull_cond1 | strong_bull_cond2)
            & (bull_cond1 | bull_cond2 | bull_cond3),
            MarketRegime.BULL,
            regime,
        )

        regime = torch.where(strong_bear_cond, MarketRegime.STRONG_BEAR, regime)

        regime = torch.where(~strong_bear_cond & bear_cond, MarketRegime.BEAR, regime)

        # Default to maintaining exposure when ambiguous
        ambiguous_bull_cond = (recent_momentum > 0.02) | (trend_persistence)
        regime = torch.where(
            (regime == MarketRegime.SIDEWAYS) & ambiguous_bull_cond,
            MarketRegime.BULL,
            regime,
        )

        return regime

    def calculate_price_volatility(
        self, prices: torch.Tensor, lookback_days: int = 30
    ) -> torch.Tensor:
        """Calculate annualized price volatility.

        Args:
            prices: Price tensor of shape [..., sequence_length]
            lookback_days: Number of days for volatility calculation

        Returns:
            Annualized volatility tensor
        """
        if prices.shape[-1] < lookback_days:
            batch_shape = prices.shape[:-1]
            return torch.full(batch_shape, 0.3, device=self.device)

        recent_prices = prices[..., -lookback_days:]
        returns = self.calculate_returns(recent_prices, periods=1)
        valid_returns = returns[..., 1:]  # Skip NaN

        volatility = torch.std(valid_returns, dim=-1) * math.sqrt(252)

        # Clamp volatility to reasonable range
        return torch.clamp(volatility, min=0.1, max=3.0)


class FinancialRegimeEnv(EnvBase):
    """A financial trading environment with market regime detection.

    This environment simulates financial market dynamics where an agent can take
    trading actions (Hold, Buy, Sell) based on price history and current holdings.
    The environment uses sophisticated market regime detection to provide realistic
    market behavior and reward signals.

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
                    device=None,
                    shape=torch.Size([])),
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
    rng = None

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

        # Initialize market dynamics
        self.market_dynamics = FinancialMarketDynamics(device=self.device)

        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_(generator=self.rng).item()
        self.set_seed(seed)

    @classmethod
    def _step(cls, tensordict):
        """Execute one step in the environment.

        Args:
            tensordict: Input TensorDict containing current state and action

        Returns:
            TensorDict containing next state, reward, and done flag
        """
        # Extract current state
        price_history = tensordict["price_history"]
        current_holdings = tensordict["current_holdings"]
        action = tensordict["action"].squeeze(-1)  # 0=Hold, 1=Buy, 2=Sell
        step_count = tensordict.get("step_count", torch.zeros_like(action))

        # Environment parameters
        params = tensordict["params"]
        volatility = params["volatility"]
        drift = params["drift"]
        transaction_cost = params["transaction_cost"]
        max_steps = params["max_steps"]

        # Generate next price using geometric Brownian motion with regime-aware parameters
        current_price = price_history[..., -1]
        dt = 1.0 / 252.0  # Daily time step

        # Add some randomness for price generation
        random_shock = torch.randn_like(current_price) * volatility * math.sqrt(dt)
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

        # Calculate reward based on action and price change
        price_change = (next_price - current_price) / current_price
        reward = torch.zeros_like(current_price)
        new_holdings = current_holdings.clone()

        # Apply trading logic
        # Action 0: Hold - no change in position
        # Action 1: Buy - if not holding, buy; if holding, hold
        # Action 2: Sell - if holding, sell; if not holding, hold

        # Buy action (1)
        buy_mask = (action == 1) & (~current_holdings.squeeze(-1))
        if buy_mask.any():
            new_holdings[buy_mask, 0] = True
            # Pay transaction cost for buying
            reward[buy_mask] = -transaction_cost[buy_mask]

        # Sell action (2)
        sell_mask = (action == 2) & (current_holdings.squeeze(-1))
        if sell_mask.any():
            new_holdings[sell_mask, 0] = False
            # Receive profit/loss minus transaction cost
            reward[sell_mask] = price_change[sell_mask] - transaction_cost[sell_mask]

        # If holding, accumulate unrealized gains (for reward shaping)
        holding_mask = new_holdings.squeeze(-1)
        if holding_mask.any():
            # Small reward shaping for unrealized gains
            reward[holding_mask] += price_change[holding_mask] * 0.1

        # Update step count
        new_step_count = step_count + 1

        # Check if episode is done
        done = new_step_count >= max_steps
        terminated = done.clone()

        # Create output tensordict
        out = TensorDict(
            {
                "price_history": new_price_history,
                "current_holdings": new_holdings,
                "step_count": new_step_count,
                "params": tensordict["params"],
                "reward": reward.unsqueeze(-1),
                "done": done.unsqueeze(-1),
                "terminated": terminated.unsqueeze(-1),
            },
            tensordict.shape,
        )
        return out

    def _reset(self, tensordict):
        """Reset the environment to initial state.

        Args:
            tensordict: Optional TensorDict containing reset parameters

        Returns:
            TensorDict containing initial state
        """
        batch_size = (
            tensordict.batch_size if tensordict is not None else self.batch_size
        )

        if tensordict is None or "params" not in tensordict:
            # Generate new parameters if not provided
            tensordict = self.gen_params(batch_size=batch_size, device=self.device)
        elif "price_history" in tensordict and "current_holdings" in tensordict:
            # Hard reset with provided state
            return tensordict

        out = self._reset_random_data(
            tensordict.shape, batch_size, tensordict["params"]
        )
        return out

    def _reset_random_data(self, shape, batch_size, params):
        """Generate random initial state data.

        Args:
            shape: Shape of the state tensors
            batch_size: Batch size for parallel environments
            params: Environment parameters

        Returns:
            TensorDict containing random initial state
        """
        window_size = params["window_size"]
        initial_price = params["initial_price"]
        volatility = params["volatility"]

        # Generate random price history using geometric Brownian motion
        dt = 1.0 / 252.0
        returns = (
            torch.randn((*shape, window_size), generator=self.rng, device=self.device)
            * volatility.unsqueeze(-1)
            * math.sqrt(dt)
        )

        # Create price history starting from initial_price
        log_prices = torch.cumsum(returns, dim=-1)
        price_history = initial_price.unsqueeze(-1) * torch.exp(log_prices)

        # Initialize with no holdings
        current_holdings = torch.zeros(
            (*shape, 1), dtype=torch.bool, device=self.device
        )

        # Initialize step count
        step_count = torch.zeros(shape, dtype=torch.long, device=self.device)

        out = TensorDict(
            {
                "price_history": price_history,
                "current_holdings": current_holdings,
                "step_count": step_count,
                "params": params,
            },
            batch_size=batch_size,
        )
        return out

    def _make_spec(self, td_params):
        """Create the environment specifications.

        Args:
            td_params: TensorDict containing environment parameters
        """
        window_size = int(td_params["params", "window_size"].item())

        # Observation spec
        self.observation_spec = Composite(
            price_history=UnboundedContinuous(
                shape=(window_size,),
                dtype=torch.float32,
            ),
            current_holdings=Binary(
                shape=(1,),
                dtype=torch.bool,
            ),
            step_count=UnboundedContinuous(
                shape=(),
                dtype=torch.long,
            ),
            # Include params in observation for stateless environment
            params=make_composite_from_td(
                td_params["params"], unsqueeze_null_shapes=False
            ),
            shape=(),
        )

        # State spec (same as observation for stateless environment)
        self.state_spec = self.observation_spec.clone()

        # Action spec: 0=Hold, 1=Buy, 2=Sell
        self.action_spec = Categorical(
            n=3,
            shape=(1,),
            dtype=torch.long,
        )

        # Reward spec
        self.reward_spec = UnboundedContinuous(
            shape=(*td_params.shape, 1),
            dtype=torch.float32,
        )

    def _set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
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
        """Generate environment parameters.

        Args:
            window_size: Size of price history window
            initial_price: Starting price for the asset
            volatility: Annual volatility of the asset
            drift: Annual drift (expected return) of the asset
            transaction_cost: Cost per transaction as fraction of price
            max_steps: Maximum steps per episode
            batch_size: Batch size for parallel environments
            device: PyTorch device

        Returns:
            TensorDict containing environment parameters
        """
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
