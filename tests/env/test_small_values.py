import pandas as pd
import sys
from pathlib import Path
import tempfile
import pytest

# Remove sys.path manipulation
# project_root = Path(__file__).resolve().parent.parent.parent
# sys.path.insert(0, str(project_root))

# Use updated import path
try:
    from src.env.trading_env import TradingEnv
except ImportError as e:
    print(f"Failed to import TradingEnv from src.env: {e}")
    print(f"sys.path: {sys.path}")
    raise


# Helper function (assuming it exists or is defined elsewhere)
# Need to make sure create_mock_csv is available
# For now, let's define a dummy version here if it's not imported
def create_mock_csv(data_dict, dir_name):
    path = Path(dir_name) / "mock_data.csv"
    pd.DataFrame(data_dict).to_csv(path, index=False)
    return str(path)


# Constants for testing
WINDOW_SIZE = 5
N_FEATURES = 5  # Should match TradingEnv internal feature count
ACCOUNT_STATE_DIM = 2
INITIAL_BALANCE = 1.0  # Very small initial balance
TRANSACTION_FEE = 0.05  # High transaction fee


@pytest.mark.unittest
class TestTradingEnvSmallValues:
    """Tests TradingEnv calculations with very small price values."""

    def setup_method(self, method):
        self.window_size = WINDOW_SIZE
        self.initial_balance = INITIAL_BALANCE
        self.transaction_fee = TRANSACTION_FEE
        self.small_price_base = 1e-10

        # Create mock data for setup
        # We need enough rows so that after normalization (drops win_size-1)
        # and reset (starts at step=win_size), we can take at least one step.
        # data_len = num_rows - (win_size - 1)
        # Need: win_size < data_len - 1 (to allow step at current_step=win_size)
        # Need: win_size < num_rows - (win_size - 1) - 1
        # Need: num_rows > 2 * win_size
        num_rows = 2 * self.window_size + 5  # Provides sufficient buffer

        self.temp_dir = tempfile.TemporaryDirectory()
        data_dict = {
            "open": [self.small_price_base * 2] * num_rows,
            "high": [self.small_price_base * 3] * num_rows,
            "low": [self.small_price_base * 1] * num_rows,
            # Make close prices slightly variable
            "close": [
                self.small_price_base * (1.0 + 0.1 * i / (num_rows - 1))
                for i in range(num_rows)
            ],
            "volume": [10] * num_rows,
        }
        self.mock_path = create_mock_csv(data_dict, self.temp_dir.name)

        # Initialize environment instance
        self.env = TradingEnv(
            data_path=self.mock_path,
            window_size=self.window_size,
            initial_balance=self.initial_balance,
            transaction_fee=self.transaction_fee,
            reward_pnl_scale=1.0,
            reward_cost_scale=0.5,
        )
        # print(f"[Setup Small Values] data_len={self.env.data_len}, original_len={len(data_dict['close'])}")

    def teardown_method(self, method):
        self.temp_dir.cleanup()

    def test_buy_with_small_price(self):
        """Test buying an asset when the price is extremely small."""
        # Reset -> current_step = window_size = 5.
        # First step uses price at index corresponding to original step 5
        # Original indices: 0 to num_rows-1
        # Norm drops indices 0 to win_size-2 (0 to 3)
        # Internal indices: 0 to num_rows - win_size (0 to 10)
        # Internal index 0 corresponds to original index win_size-1 (4)
        # Reset sets current_step = win_size = 5.
        # First step uses internal index 5. original_index = 5 + (5-1) = 9
        obs_reset, info_reset = self.env.reset()
        assert self.env.current_step == self.window_size

        action = 3  # Buy 100%

        # Price used in step calculation is at current_step (original index 9)
        original_index_for_step = self.env.current_step + (self.window_size - 1)
        current_price_in_step = self.small_price_base * (
            1.0
            + 0.1
            * original_index_for_step
            / (len(obs_reset["market_data"]) + self.window_size - 1)
        )  # Approximation based on data gen
        current_price_in_step = self.env.original_close_prices[
            self.env._map_to_original_index(self.env.current_step)
        ]

        # Calculate expected based on discrete action 3
        balance_before = self.env.balance
        buy_amount_cash = balance_before * 1.0
        transaction_cost = buy_amount_cash * self.transaction_fee
        cash_for_crypto = buy_amount_cash - transaction_cost
        expected_position = cash_for_crypto / current_price_in_step
        expected_balance = balance_before - buy_amount_cash

        # Execute the step (current_step becomes window_size + 1)
        next_obs, reward, done, truncated, info = self.env.step(action)

        # Check post-step state using pytest asserts
        assert (
            abs(self.env.balance - expected_balance) < 1e-9
        )  # Balance should be near zero
        assert (
            expected_position > 1e-9
        )  # Ensure expected position is not zero before relative check
        assert (
            abs(self.env.position - expected_position) / expected_position < 1e-7
        )  # Position should match calc
        assert (
            abs(info["transaction_cost"] - transaction_cost) < 1e-9
        )  # Cumulative cost
        assert info["step"] == self.window_size

        # Check observation shapes (basic check)
        assert next_obs["market_data"].shape == (self.window_size, N_FEATURES)
        assert next_obs["account_state"].shape == (ACCOUNT_STATE_DIM,)

    def test_sell_with_small_price(self):
        """Test selling an asset when the price is extremely small."""
        obs_reset, info_reset = self.env.reset()

        # First, buy something
        buy_action = 3  # Buy 100%
        _, _, _, _, info_buy = self.env.step(buy_action)
        position_after_buy = self.env.position
        balance_after_buy = self.env.balance
        cost_after_buy = self.env.total_transaction_cost
        assert position_after_buy > 1e-9
        assert balance_after_buy < 1e-9  # Balance should be near zero

        # Now sell it all
        sell_action = 6  # Sell 100%

        # Price used in step calculation (step = window_size + 1)
        original_index_for_step = self.env.current_step + (self.window_size - 1)
        current_price_in_step = self.env.original_close_prices[
            self.env._map_to_original_index(self.env.current_step)
        ]

        # Calculate expected
        sell_amount_crypto = position_after_buy * 1.0
        cash_before_fee = sell_amount_crypto * current_price_in_step
        step_transaction_cost = cash_before_fee * self.transaction_fee
        cash_change = cash_before_fee - step_transaction_cost
        expected_position = position_after_buy - sell_amount_crypto  # Near zero
        expected_balance = balance_after_buy + cash_change  # Should increase
        expected_total_cost = cost_after_buy + step_transaction_cost

        # Execute the step
        next_obs, reward, done, truncated, info = self.env.step(sell_action)

        # Check post-step state using pytest asserts
        assert abs(self.env.position) < 1e-9  # Position should be near zero
        assert (
            expected_balance > 1e-9
        )  # Ensure expected balance is not zero before relative check
        assert abs(self.env.balance - expected_balance) / expected_balance < 1e-7
        assert (
            expected_total_cost > 1e-9
        )  # Ensure expected cost is not zero before relative check
        assert (
            abs(info["transaction_cost"] - expected_total_cost) / expected_total_cost
            < 1e-7
        )
        assert info["step"] == self.window_size + 1
