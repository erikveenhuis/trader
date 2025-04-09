import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

# Configure ONLY the TradingEnv logger for DEBUG messages
logger = logging.getLogger("TradingEnv")
# logger.setLevel(logging.DEBUG) # Level is now controlled by root config
# Remove handler setup - should be handled by root config
# if not logging.getLogger().hasHandlers():
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logging.getLogger().addHandler(handler)


class TradingEnv(gym.Env):
    """
    A trading environment for reinforcement learning.

    Features:
    - 60 normalized OHLCV observations
    - Action space: Discrete: 0=Hold, 1=Buy25%, 2=Buy50%, 3=Buy100%, 4=Sell25%, 5=Sell50%, 6=Sell100%
    - Account state information (position and balance)
    """

    def __init__(
        self,
        data_path: str,
        reward_pnl_scale: float,
        reward_cost_scale: float,
        initial_balance: float,
        transaction_fee: float,
        window_size: int,
    ):
        super(TradingEnv, self).__init__()

        logger.info(
            f"Initializing TradingEnv with window_size={window_size}, initial_balance={initial_balance}, transaction_fee={transaction_fee}"
        )

        self.data_path = data_path
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.reward_pnl_scale = reward_pnl_scale
        self.reward_cost_scale = reward_cost_scale

        assert (
            isinstance(window_size, int) and window_size >= 1
        ), f"window_size must be an integer >= 1, got {window_size}"
        assert (
            isinstance(initial_balance, (int, float)) and initial_balance >= 0
        ), f"initial_balance must be a non-negative number, got {initial_balance}"
        assert (
            isinstance(transaction_fee, (int, float)) and 0 <= transaction_fee < 1
        ), f"transaction_fee must be a number between 0 (inclusive) and 1 (exclusive), got {transaction_fee}"

        logger.info(f"Loading data from {data_path}")
        data_df = pd.read_csv(data_path).dropna()
        original_data_df = data_df.copy()

        required_columns = ["open", "high", "low", "close", "volume"]
        assert all(
            col in original_data_df.columns for col in required_columns
        ), f"Input data missing required columns: {required_columns}. Found: {list(original_data_df.columns)}"
        for col in required_columns:
            assert pd.api.types.is_numeric_dtype(
                original_data_df[col]
            ), f"Column '{col}' must be numeric."
        assert (
            len(original_data_df) >= self.window_size
        ), f"Data length ({len(original_data_df)}) must be >= window_size ({self.window_size})."

        self.original_close_prices = original_data_df["close"].values.astype(np.float32)
        self.original_data_len = len(self.original_close_prices)

        # Normalize data (operates on a copy internally now)
        norm_data_df = self._normalize_data(original_data_df)
        logger.info(f"Data loaded and normalized. Shape: {norm_data_df.shape}")

        norm_features = [
            "norm_open",
            "norm_high",
            "norm_low",
            "norm_close",
            "norm_volume",
        ]
        self.norm_data_array = norm_data_df[norm_features].values.astype(np.float32)
        self.data_len = len(self.norm_data_array)

        assert self.norm_data_array.shape == (
            self.data_len,
            len(norm_features),
        ), "Normalized data array shape mismatch."
        assert (
            self.norm_data_array.dtype == np.float32
        ), "Normalized data array dtype mismatch."
        assert self.original_close_prices.shape == (
            self.original_data_len,
        ), "Original close prices array shape mismatch."
        assert (
            self.original_close_prices.dtype == np.float32
        ), "Original close prices array dtype mismatch."
        assert self.data_len > 0, "Normalized data length must be > 0."
        assert self.original_data_len > 0, "Original data length must be > 0."

        self.action_space = spaces.Discrete(7)
        num_features = 5
        self.observation_space = spaces.Dict(
            {
                "market_data": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.window_size, num_features),
                    dtype=np.float32,
                ),
                "account_state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
                ),
            }
        )

        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0
        self.total_transaction_cost = 0.0
        self.portfolio_values = []
        self.previous_price = 0.0

    def _normalize_data(self, data_df):
        """Normalize the OHLCV data using min-max scaling. Returns normalized DataFrame."""
        logger.debug("Starting data normalization")
        df = data_df[["open", "high", "low", "close", "volume"]].copy()
        for col in ["open", "high", "low", "close", "volume"]:
            rolling_min = df[col].rolling(window=self.window_size).min()
            rolling_max = df[col].rolling(window=self.window_size).max()
            df[f"norm_{col}"] = (df[col] - rolling_min) / (
                rolling_max - rolling_min + 1e-8
            )
        initial_rows = len(df)
        df.dropna(inplace=True)
        assert (
            len(df) > 0
        ), f"No data remaining after normalization (dropna). Original: {initial_rows}, window: {self.window_size}."
        logger.info(
            f"Dropped {initial_rows - len(df)} rows with NaN values after normalization"
        )
        return df

    def _map_to_original_index(self, internal_index):
        """Helper method to map internal step index to original data index."""
        if np.isnan(internal_index) or np.isinf(internal_index):
            raise ValueError(f"Invalid internal_index ({internal_index}).")
        try:
            internal_index = int(internal_index)
        except (ValueError, TypeError):
            raise ValueError(
                f"Could not convert internal_index ({internal_index}) to integer."
            )
        return internal_index + (self.window_size - 1)

    def _get_observation(self):
        """Get the current observation."""
        observation_step = min(self.current_step, self.data_len - 1)
        start_index = max(0, observation_step - self.window_size + 1)
        market_data = self.norm_data_array[start_index : observation_step + 1]
        if len(market_data) < self.window_size:
            padding_shape = (self.window_size - len(market_data), market_data.shape[1])
            padding = np.zeros(padding_shape, dtype=market_data.dtype)
            market_data = np.vstack((padding, market_data))

        assert market_data.shape == (
            self.window_size,
            self.observation_space["market_data"].shape[1],
        ), "Market data shape mismatch."
        assert market_data.dtype == np.float32, "Market data dtype mismatch."

        original_index = min(
            self._map_to_original_index(observation_step), self.original_data_len - 1
        )
        assert (
            0 <= original_index < self.original_data_len
        ), f"original_index ({original_index}) out of bounds in _get_observation."
        current_price = self.original_close_prices[original_index]
        assert (
            current_price >= 0
        ), f"Negative price ({current_price}) in _get_observation."

        portfolio_value = max(0, self.balance + self.position * current_price)
        # --- DEBUG: Observation State Calc ---
        # print(f"[DEBUG ObsState] balance={self.balance:.8f}, position={self.position:.8f}, current_price={current_price:.8f} -> portfolio_value={portfolio_value:.8f}")
        # --- END DEBUG ---
        assert (
            portfolio_value >= -1e-9
        ), f"Negative portfolio_value ({portfolio_value}) in _get_observation."

        normalized_position = (
            (self.position * current_price / portfolio_value)
            if portfolio_value > 1e-9
            else 0
        )
        normalized_balance = (
            self.balance / self.initial_balance if self.initial_balance > 1e-9 else 0
        )
        # --- DEBUG: Observation State Calc ---
        # print(f"[DEBUG ObsState] norm_pos={normalized_position:.8f}, norm_bal={normalized_balance:.8f}")
        # --- END DEBUG ---
        account_state = np.array(
            [normalized_position, normalized_balance], dtype=np.float32
        )

        assert account_state.shape == (2,), "Account state shape mismatch."
        assert account_state.dtype == np.float32, "Account state dtype mismatch."
        assert np.isfinite(
            account_state
        ).all(), f"Non-finite values in account_state: {account_state}"
        assert (
            account_state[1] >= -1e-6
        ), f"Normalized balance ({account_state[1]}) negative."
        assert (
            account_state[0] >= -1e-6
        ), f"Normalized position ({account_state[0]}) negative."

        return {
            "market_data": market_data.astype(np.float32),
            "account_state": account_state,
        }

    def _get_info(self):
        """Get additional information about the current state."""
        info_step = min(self.current_step, self.data_len - 1)
        original_index = min(
            self._map_to_original_index(info_step), self.original_data_len - 1
        )
        assert (
            0 <= original_index < self.original_data_len
        ), f"original_index ({original_index}) out of bounds in _get_info."
        current_price = self.original_close_prices[original_index]
        assert current_price >= 0, f"Negative price ({current_price}) in _get_info."

        asset_value = self.position * current_price
        portfolio_value = self.balance + asset_value
        assert (
            asset_value >= -1e-9
        ), f"Negative asset_value ({asset_value}) in _get_info."
        assert (
            portfolio_value >= -1e-9
        ), f"Negative portfolio_value ({portfolio_value}) in _get_info."

        return {
            "step": self.current_step,
            "price": current_price,
            "balance": self.balance,
            "position": self.position,
            "portfolio_value": max(0, portfolio_value),
            "transaction_cost": self.total_transaction_cost,
        }

    def step(self, action):
        """Execute one step in the environment."""
        if self.current_step >= self.data_len:
            raise RuntimeError(
                f"Cannot call step() when the episode is already done (current_step={self.current_step} >= data_len={self.data_len}). Please call env.reset()."
            )

        assert self.action_space.contains(action), f"Invalid action: {action}"

        previous_portfolio_value = (
            self.portfolio_values[-1] if self.portfolio_values else self.initial_balance
        )
        invalid_action_penalty = -1.0  # Initialize penalty to a negative value

        try:
            original_index_current = min(
                self._map_to_original_index(self.current_step),
                self.original_data_len - 1,
            )
            current_price = self.original_close_prices[original_index_current]
        except (IndexError, ValueError) as e:
            logger.error(
                f"Error accessing price data at step {self.current_step}: {str(e)}"
            )
            current_price = self.position_price if self.position_price > 0 else self.previous_price
        assert (
            current_price > 1e-20
        ), f"current_price is non-positive ({current_price}) at step {self.current_step}."

        # Calculate current portfolio value BEFORE action logic
        current_portfolio_value = self.balance + self.position * current_price

        position_change = 0.0
        cash_change = 0.0
        step_transaction_cost = 0.0
        reward = 0.0

        if action == 0:
            if self.position > 1e-9:
                price_change = current_price - self.previous_price
                reward = (price_change * self.position / self.initial_balance) * self.reward_pnl_scale
            else:
                reward = 0.0
            step_transaction_cost = 0.0
        elif 1 <= action <= 3:
            buy_fraction = 0.0
            if action == 1:
                buy_fraction = 0.25
            elif action == 2:
                buy_fraction = 0.50
            elif action == 3:
                buy_fraction = 1.00
            if self.balance > 1e-9:
                buy_amount_cash = self.balance * buy_fraction
                step_transaction_cost = buy_amount_cash * self.transaction_fee # CALCULATION
                cash_for_crypto = buy_amount_cash - step_transaction_cost
                if cash_for_crypto > 0 and current_price > 1e-20:
                    position_change = cash_for_crypto / current_price
                    cash_change = -buy_amount_cash
                    if position_change > 1e-9:
                        old_value = self.position * self.position_price
                        new_value = position_change * current_price
                        total_position_after_buy = self.position + position_change
                        if total_position_after_buy > 1e-9:
                            self.position_price = (
                                old_value + new_value
                            ) / total_position_after_buy
                        else:
                            self.position_price = 0
                else:
                    # Penalize ineffectual buy attempt (low cash/price)
                    logger.debug(
                        f"Step {self.current_step}: Low cash/price for Buy {buy_fraction*100}%. Penalizing."
                    )
                    reward = invalid_action_penalty
                    step_transaction_cost = 0.0 # No cost if no trade
            else:
                # Penalize buy attempt with zero balance
                logger.debug(
                    f"Step {self.current_step}: Zero balance, cannot Buy {buy_fraction*100}%. Penalizing."
                )
                reward = invalid_action_penalty
                step_transaction_cost = 0.0 # No cost if no trade
        elif 4 <= action <= 6:
            assert (
                self.position >= -1e-9
            ), f"Negative position ({self.position}) at step {self.current_step} before sell."
            sell_fraction = 0.0
            if action == 4:
                sell_fraction = 0.25
            elif action == 5:
                sell_fraction = 0.50
            elif action == 6:
                sell_fraction = 1.00
            if self.position > 1e-9:
                sell_amount_crypto = self.position * sell_fraction
                cash_before_fee = sell_amount_crypto * current_price
                step_transaction_cost = cash_before_fee * self.transaction_fee
                cash_change = cash_before_fee - step_transaction_cost
                position_change = -sell_amount_crypto
                if abs(self.position + position_change) < 1e-9:
                    self.position_price = 0
                if cash_change < 0:
                    logger.warning(
                        f"Step {self.current_step}: Negative cash change ({cash_change}) on Sell {sell_fraction*100}%."
                    )
                    cash_change = 0
                    step_transaction_cost = cash_before_fee
            else:
                # Invalid sell attempt
                logger.debug(
                    f"Step {self.current_step}: Zero position, cannot Sell {sell_fraction*100}%. Penalizing."
                )  # Updated log message
                reward = invalid_action_penalty # Penalty is the only reward component
                step_transaction_cost = 0.0 # No cost for invalid action

        # Apply transaction cost penalty ONLY if a valid trade occurred (step_cost > 0)
        # and the reward wasn't already set by the invalid penalty
        if step_transaction_cost > 1e-9 and action != 0 and not (4 <= action <= 6 and self.position <= 1e-9):
            reward_cost = (
                -(step_transaction_cost / self.initial_balance) * self.reward_cost_scale
                if self.initial_balance > 1e-9
                else 0.0
            )
            # For buy/sell, calculate PnL based on portfolio change AFTER cost
            # This is tricky, let's stick to simpler cost penalty for now and potentially zero base reward for buy/sell
            # Alternative: reward = cash_change * scale ? Needs careful thought.
            # For now, let action 0 be the main reward source, cost is penalty.
            reward = reward_cost
            # TODO: Revisit reward for Buy/Sell actions. Maybe use portfolio change AFTER cost?
            reward_pnl = 0.0 # PnL is not added for buy/sell in current logic
            was_invalid = False
        elif (4 <= action <= 6 and self.position <= 1e-9) or \
             (1 <= action <= 3 and (self.balance <= 1e-9 or cash_for_crypto <= 0)):
             # Handle invalid actions explicitly to store their penalty for logging
             reward_pnl = 0.0
             reward_cost = 0.0
             was_invalid = True
             # Reward is already set to invalid_action_penalty in the action logic blocks
        else: # Action 0 or valid trade with zero cost (should not happen often)
             reward_pnl = reward # Reward for action 0 is PnL
             reward_cost = 0.0
             was_invalid = False

        # Accumulate total cost
        self.total_transaction_cost = self.total_transaction_cost + step_transaction_cost # Explicit addition

        new_balance = self.balance + cash_change
        new_position = self.position + position_change
        logger.debug(
            f"Step {self.current_step} Update | Action: {action} | Price: {current_price:.4f} | BEFORE: Bal={self.balance:.4f}, Pos={self.position:.4f} | CHANGES: Cash={cash_change:.4f}, Pos={position_change:.4f}, Cost={step_transaction_cost:.4f} | AFTER(calc): Bal={new_balance:.4f}, Pos={new_position:.4f}"
        )
        if np.isnan(new_position) or np.isinf(new_position):
            logger.warning(
                f"Step {self.current_step}: NaN/Inf new_position. Reverting."
            )
            new_position = self.position
        if np.isnan(new_balance) or np.isinf(new_balance):
            logger.warning(f"Step {self.current_step}: NaN/Inf new_balance. Reverting.")
            new_balance = self.balance
        self.position = max(0.0, float(new_position))
        self.balance = max(0.0, float(new_balance))
        logger.debug(
            f"Step {self.current_step} Update | POST: Balance={self.balance:.4f}, Position={self.position:.4f}"
        )
        assert self.balance >= -1e-9, f"Balance negative ({self.balance})"
        assert self.position >= -1e-9, f"Position negative ({self.position})"
        assert (
            self.position_price >= 0
        ), f"Position price negative ({self.position_price})"

        # --- Termination Conditions --- #
        # 1. Reached end of data
        end_of_data = self.current_step >= self.data_len - 1
        # 2. Portfolio value dropped below threshold (e.g., 10% of initial)
        portfolio_value_threshold = self.initial_balance * 0.10
        terminated_low_portfolio = current_portfolio_value < portfolio_value_threshold
        if terminated_low_portfolio:
            logger.warning(
                f"Episode terminated early at step {self.current_step} due to low portfolio value: "
                f"${current_portfolio_value:.2f} (< 10% of initial ${self.initial_balance:.2f})"
            )

        done = end_of_data or terminated_low_portfolio
        # --- End Termination --- #

        next_obs = self._get_observation()
        info = self._get_info()
        self.portfolio_values.append(info["portfolio_value"])

        if done:
            logger.info(
                f"Final portfolio value: ${info['portfolio_value']:.2f}, Total Transaction Cost: ${self.total_transaction_cost:.2f}"
            )

        # Update previous price for next step
        self.previous_price = current_price

        self.current_step += 1
        # --- Log Reward Breakdown --- # 
        logger.debug(f"[REWARD DEBUG] Step: {self.current_step-1}, Action: {action}, "
                     f"StepCost: {step_transaction_cost:.4f}, PnL_Rew: {reward_pnl:.4f}, "
                     f"Cost_Rew: {reward_cost:.4f}, InvalidPenalty: {invalid_action_penalty if was_invalid else 0.0:.4f}, "
                     f"Final_Reward: {reward:.4f}")
        # --- End Reward Breakdown --- #
        return next_obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        logger.info("Resetting environment")
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0
        self.total_transaction_cost = 0.0
        self.portfolio_values = []
        self.previous_price = self._get_info().get('price', 0.0)

        observation = self._get_observation()
        info = self._get_info()
        self.portfolio_values.append(info["portfolio_value"])
        logger.info(
            f"Environment reset. Initial portfolio value: ${info['portfolio_value']:.2f}"
        )
        return observation, info

    def close(self):
        logger.info("Closing trading environment")
        pass


# Example usage
if __name__ == "__main__":
    env = TradingEnv(
        data_path="3-ProcessedData/train/2018-12-06_ETH-USD.csv",
        window_size=60,
        initial_balance=10000.0,
        transaction_fee=0.001,
    )
    obs, info = env.reset()
    max_steps = 200
    print(f"Running episode for max {max_steps} steps...")
    for i in range(max_steps):
        action = env.action_space.sample()
        try:
            obs, reward, done, truncated, info = env.step(action)
            logger.debug(
                f"Step: {info['step']} | Portfolio: ${info['portfolio_value']:.2f} | Reward: {reward:.4f}"
            )
            if done or truncated:
                print(
                    f"\nEpisode finished at step {info['step']}. Done={done}, Truncated={truncated}"
                )
                break
        except RuntimeError as e:
            print(f"\nError during step {i+env.window_size}: {e}")
            break
    print("\nFinished simulation loop.")
    env.close()
    print("Environment closed.")
