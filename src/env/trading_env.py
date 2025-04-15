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
        # Check for NaN/Inf first
        if not isinstance(internal_index, (int, float)) or np.isnan(internal_index) or np.isinf(internal_index):
            raise ValueError(f"Invalid internal_index ({internal_index}). Must be a finite number.")
        try:
            # Attempt conversion to int, let it raise error for non-integers/strings
            internal_index_int = int(internal_index)
            # Additional check: ensure the original float wasn't something like 5.5
            if internal_index != internal_index_int:
                 raise ValueError # Trigger the except block for non-whole numbers
        except (ValueError, TypeError):
            # Catch errors from int() conversion or the manual check above
            raise ValueError(
                f"Could not convert internal_index ({internal_index}) to a whole integer."
            )
        # Use the successfully converted integer index
        return internal_index_int + (self.window_size - 1)

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

    # --- Action Handling Helpers ---
    def _handle_buy_action(self, buy_fraction: float, current_price: float) -> tuple:
        """Handles the logic for buy actions."""
        if self.balance > 1e-9:
            buy_amount_cash = self.balance * buy_fraction
            potential_step_transaction_cost = buy_amount_cash * self.transaction_fee
            total_required_cash = buy_amount_cash + potential_step_transaction_cost

            # Check affordability *including* fee first
            if self.balance + 1e-9 >= total_required_cash:
                # Affordable: Calculate actual changes
                step_transaction_cost = potential_step_transaction_cost
                cash_for_crypto = buy_amount_cash # Amount used to buy crypto (before fee)
                cash_change = -total_required_cash # Total cash reduction
                position_change = 0.0 # Initialize position change
                is_invalid_action = False # Assume valid initially

                if cash_for_crypto > 0 and current_price > 1e-20:
                    position_change = cash_for_crypto / current_price
                else:
                    # Edge case: Zero price or negligible crypto amount despite sufficient funds
                    logger.debug(f"Step {self.current_step}: Zero price or negligible cash_for_crypto in Buy {buy_fraction*100}%. Penalizing.")
                    is_invalid_action = True # Mark as invalid
                    # Reset changes for invalid edge case
                    step_transaction_cost = 0.0
                    position_change = 0.0
                    cash_change = 0.0
            else:
                # Not affordable
                logger.debug(f"Step {self.current_step}: Insufficient balance for Buy {buy_fraction*100}% including fee. Have {self.balance:.4f}, need {total_required_cash:.4f}. Penalizing.")
                is_invalid_action = True
                step_transaction_cost = 0.0 # Ensure cost is zeroed
                position_change = 0.0
                cash_change = 0.0 # Ensure cash change is zeroed
        else:
            # Penalize buy attempt with zero balance
            logger.debug(f"Step {self.current_step}: Zero balance, cannot Buy {buy_fraction*100}%. Penalizing.")
            is_invalid_action = True
            step_transaction_cost = 0.0 # Ensure cost is zeroed
            position_change = 0.0
            cash_change = 0.0 # Ensure cash change is zeroed

        return cash_change, position_change, step_transaction_cost, is_invalid_action

    def _handle_sell_action(self, sell_fraction: float, current_price: float) -> tuple:
        """Handles the logic for sell actions."""
        assert self.position >= -1e-9, f"Negative position ({self.position}) at step {self.current_step} before sell."
        if self.position > 1e-9:
            sell_amount_crypto = self.position * sell_fraction
            cash_before_fee = sell_amount_crypto * current_price
            step_transaction_cost = cash_before_fee * self.transaction_fee
            cash_change = cash_before_fee - step_transaction_cost
            position_change = -sell_amount_crypto
            # Ensure cash change isn't negative due to high fees (unlikely with fee < 1)
            if cash_change < 0:
                logger.warning(f"Step {self.current_step}: Negative cash change ({cash_change}) on Sell {sell_fraction*100}%.")
                cash_change = 0
                step_transaction_cost = cash_before_fee # Cost equals total cash received
            is_invalid_action = False
        else:
            # Invalid sell attempt (no position)
            logger.debug(f"Step {self.current_step}: Zero position, cannot Sell {sell_fraction*100}%. Penalizing.")
            is_invalid_action = True
            step_transaction_cost = 0.0 # No cost for invalid action
            cash_change = 0.0
            position_change = 0.0

        return cash_change, position_change, step_transaction_cost, is_invalid_action
    # --- End Action Handling Helpers ---

    # --- State Update and Calculation Helpers ---
    def _apply_state_changes(
        self,
        cash_change: float,
        position_change: float,
        step_transaction_cost: float,
        current_price: float,
        action: int # Pass action to know if it was a buy
    ):
        """Applies the calculated changes to the environment state and updates position price."""
        assert step_transaction_cost >= 0, "step_transaction_cost cannot be negative"
        self.total_transaction_cost += step_transaction_cost

        # Calculate potential new state
        new_balance = self.balance + cash_change
        new_position = self.position + position_change

        # Log state BEFORE applying changes
        logger.debug(
            f"Step {self.current_step} State Update | BEFORE: Bal={self.balance:.4f}, Pos={self.position:.4f}, PosPrice={self.position_price:.4f} | CHANGES: Cash={cash_change:.4f}, Pos={position_change:.4f}, Cost={step_transaction_cost:.4f}"
        )

        # --- Update Position Price --- #
        # Only update if there was a change in position
        if abs(position_change) > 1e-9:
            # If it was a BUY action (position increased)
            if 1 <= action <= 3 and position_change > 1e-9:
                # Calculate cost of the crypto bought (cash reduction excluding fees)
                cash_for_crypto = position_change * current_price
                old_total_cost = self.position * self.position_price
                new_cost = cash_for_crypto
                # Ensure new_position is calculated based on precise float values
                _precise_new_position = self.position + position_change
                if _precise_new_position > 1e-9:
                    self.position_price = (old_total_cost + new_cost) / _precise_new_position
                else:
                    # This case should be rare for buys unless price is near zero
                    self.position_price = 0
            # If it was a SELL action or position becomes zero
            elif abs(new_position) < 1e-9: # Check if position is effectively zero after change
                self.position_price = 0 # Reset average price if position is sold off
            # else: Sell action but position remains > 0 -> position_price doesn't change
        # --- End Update Position Price --- #

        # Apply changes and clamp to avoid small negative values or NaN/Inf
        if np.isnan(new_position) or np.isinf(new_position):
            logger.warning(f"Step {self.current_step}: NaN/Inf new_position detected. Keeping old value: {self.position}")
        else:
            self.position = max(0.0, float(new_position))

        if np.isnan(new_balance) or np.isinf(new_balance):
            logger.warning(f"Step {self.current_step}: NaN/Inf new_balance detected. Keeping old value: {self.balance}")
        else:
            self.balance = max(0.0, float(new_balance))

        # Log state AFTER applying changes
        logger.debug(
            f"Step {self.current_step} State Update | POST: Balance={self.balance:.4f}, Position={self.position:.4f}, PositionPrice={self.position_price:.4f}"
        )

        # Final checks on applied state
        if not np.isnan(self.position):
             assert self.position >= -1e-9, f"Position negative after apply ({self.position})"
        if not np.isnan(self.position_price):
             assert self.position_price >= 0, f"Position price negative after apply ({self.position_price})"

    def _calculate_reward(
        self,
        previous_portfolio_value: float,
        current_portfolio_value_after_action: float,
        is_invalid_action: bool,
        invalid_action_penalty: float,
    ) -> float:
        """Calculates the reward for the current step."""
        if is_invalid_action:
            reward = invalid_action_penalty
            logger.debug(f"[REWARD] Step {self.current_step}: Invalid action penalty applied: {reward:.8f}")
        else:
            # Assert components are finite before PnL calculation
            assert np.isfinite(current_portfolio_value_after_action), f"Current portfolio value not finite for reward: {current_portfolio_value_after_action}"
            assert np.isfinite(previous_portfolio_value), f"Previous portfolio value not finite for reward: {previous_portfolio_value}"
            pnl = current_portfolio_value_after_action - previous_portfolio_value
            # Scale reward by initial balance to keep it in a reasonable range
            reward = (pnl / self.initial_balance) * self.reward_pnl_scale if self.initial_balance > 1e-9 else 0.0
            logger.debug(f"[REWARD] Step {self.current_step}: PrevPort={previous_portfolio_value:.4f}, CurPort={current_portfolio_value_after_action:.4f}, PnL={pnl:.4f}, ScaledReward={reward:.8f}")

        assert not np.isnan(reward) and not np.isinf(reward), f"Reward calculation resulted in NaN or Inf: {reward}"
        return reward

    def _check_termination(self, current_portfolio_value_after_action: float) -> bool:
        """Checks if the episode should terminate."""
        # 1. Reached end of data
        end_of_data = self.current_step >= self.data_len - 1
        # 2. Portfolio value dropped below threshold (e.g., 10% of initial)
        portfolio_value_threshold = self.initial_balance * 0.10
        # Use the portfolio value AFTER the action for termination check
        terminated_low_portfolio = current_portfolio_value_after_action < portfolio_value_threshold
        if terminated_low_portfolio:
            logger.warning(
                f"Episode terminated early at step {self.current_step} due to low portfolio value: "
                f"${current_portfolio_value_after_action:.2f} (< {portfolio_value_threshold:.2f})"
            )

        return end_of_data or terminated_low_portfolio
    # --- End State Update and Calculation Helpers ---

    def step(self, action):
        """Execute one step in the environment."""
        if self.current_step >= self.data_len:
            raise RuntimeError(
                f"Cannot call step() when the episode is already done (current_step={self.current_step} >= data_len={self.data_len}). Please call env.reset()."
            )

        assert self.action_space.contains(action), f"Invalid action: {action}"

        # --- Store previous state for reward calculation ---
        # Use previous_price for valuing position BEFORE the current step's price is revealed
        previous_portfolio_value = max(0, self.balance + self.position * self.previous_price)
        # --- End Previous State ---

        invalid_action_penalty = -1.0  # Define penalty value

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

        # Calculate current portfolio value BEFORE action logic based on current price - Removed, not needed for new reward

        position_change = 0.0
        cash_change = 0.0
        step_transaction_cost = 0.0
        # reward = 0.0 # Reward calculated later
        is_invalid_action = False # Flag for invalid actions

        if action == 0:
            # Hold action: PnL comes from price change on existing position
            step_transaction_cost = 0.0
            # No changes to position or cash needed
        elif 1 <= action <= 3:
            # Buy actions
            buy_fraction = 0.0
            if action == 1: buy_fraction = 0.25
            elif action == 2: buy_fraction = 0.50
            elif action == 3: buy_fraction = 1.00
            cash_change, position_change, step_transaction_cost, is_invalid_action = self._handle_buy_action(buy_fraction, current_price)
        elif 4 <= action <= 6:
            # Sell actions
            sell_fraction = 0.0
            if action == 4: sell_fraction = 0.25
            elif action == 5: sell_fraction = 0.50
            elif action == 6: sell_fraction = 1.00
            cash_change, position_change, step_transaction_cost, is_invalid_action = self._handle_sell_action(sell_fraction, current_price)

        # --- Apply State Changes ---
        # Call the helper to update balance, position, total cost, and position price
        self._apply_state_changes(cash_change, position_change, step_transaction_cost, current_price, action)

        # --- Calculate Portfolio Value AFTER Action ---
        # Assert components are finite before calculation
        assert np.isfinite(self.balance), f"Balance is not finite after state change: {self.balance}"
        assert np.isfinite(self.position), f"Position is not finite after state change: {self.position}"
        assert np.isfinite(current_price), f"Current price is not finite: {current_price}"
        current_portfolio_value_after_action = self.balance + self.position * current_price
        # Add NaN check before assertion
        if not np.isnan(current_portfolio_value_after_action):
             assert current_portfolio_value_after_action >= -1e-9, f"Negative portfolio value after action ({current_portfolio_value_after_action})"
        current_portfolio_value_after_action = max(0, current_portfolio_value_after_action) # Ensure non-negative

        # --- Calculate Reward based on Portfolio Change ---
        reward = self._calculate_reward(
            previous_portfolio_value,
            current_portfolio_value_after_action,
            is_invalid_action,
            invalid_action_penalty,
        )

        # --- Termination Conditions --- #
        done = self._check_termination(current_portfolio_value_after_action)
        # --- End Termination --- #

        # --- Prepare and Return Results --- #
        next_obs = self._get_observation()
        info = self._get_info()
        # Add the cost incurred in THIS step to the info dict
        info['step_transaction_cost'] = step_transaction_cost
        self.portfolio_values.append(info["portfolio_value"])

        if done:
            logger.info(
                f"Final portfolio value: ${info['portfolio_value']:.2f}, Total Transaction Cost: ${self.total_transaction_cost:.2f}"
            )

        # Update previous price for next step's reward calculation
        self.previous_price = current_price

        self.current_step += 1
        return next_obs, reward, done, False, info # Return truncated=False

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
        data_path="data/train/2018-12-06_ETH-USD.csv",
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
