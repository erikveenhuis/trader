import pytest
import pandas as pd
import sys
from pathlib import Path
import tempfile
import unittest

# Remove sys.path manipulation
# project_root = Path(__file__).resolve().parent.parent.parent # Adjust based on actual structure
# sys.path.insert(0, str(project_root))

# Use updated import path
try:
    from src.env.trading_env import TradingEnv
except ImportError as e:
    print(f"Failed to import TradingEnv from src.env: {e}")
    print(f"sys.path: {sys.path}")
    raise


# Helper function
def create_mock_csv(data_dict, dir_name):
    path = Path(dir_name) / "mock_data.csv"
    pd.DataFrame(data_dict).to_csv(path, index=False)
    return str(path)


# Constants for testing
WINDOW_SIZE = 10
N_FEATURES = 5
ACCOUNT_STATE_DIM = 2
INITIAL_BALANCE = 100.0  # Use smaller balance for edge cases
TRANSACTION_FEE = 0.01  # Higher fee


@pytest.mark.unittest
class TestTradingEnvEdgeCases(unittest.TestCase):
    """Tests specific edge cases like zero price."""

    def setup_method(self, method):
        self.window_size = WINDOW_SIZE
        self.initial_balance = INITIAL_BALANCE
        self.transaction_fee = TRANSACTION_FEE
        self.temp_dir = tempfile.TemporaryDirectory()
        # We don't create a default env here, tests will create specific ones

    def teardown_method(self, method):
        self.temp_dir.cleanup()

    def test_zero_price_step(self):
        """Test stepping when the current price is zero."""
        # Need enough data to take at least one step after reset
        num_rows = 2 * self.window_size + 5
        mock_data_dict = {
            "open": [1.0] * num_rows,
            "high": [1.1] * num_rows,
            "low": [0.9] * num_rows,
            "close": [1.0] * num_rows,
            "volume": [100] * num_rows,
        }
        # Set price to zero at the step index
        step_index_internal = self.window_size  # first step after reset
        original_step_index = step_index_internal + (self.window_size - 1)
        mock_data_dict["close"][original_step_index] = 0.0

        mock_path = create_mock_csv(mock_data_dict, self.temp_dir.name)
        env = TradingEnv(
            data_path=mock_path,
            window_size=self.window_size,
            initial_balance=self.initial_balance,
            transaction_fee=self.transaction_fee,
            reward_pnl_scale=1.0,
            reward_cost_scale=0.5,
        )
        env.reset()

        action = 3  # Buy 100%

        # Expect the environment's internal assertion to fail due to zero price
        with pytest.raises(AssertionError, match="current_price is non-positive"):
            env.step(action)

    def test_insufficient_balance_for_buy(self):
        """Test attempting to buy more than available balance."""
        # Setup with minimal balance and non-zero price
        initial_balance = 0.1
        num_rows = 2 * self.window_size + 5
        mock_data_dict = {
            "open": [1.0] * num_rows,
            "high": [1.1] * num_rows,
            "low": [0.9] * num_rows,
            "close": [1.0] * num_rows,
            "volume": [100] * num_rows,
        }
        mock_path = create_mock_csv(mock_data_dict, self.temp_dir.name)
        env = TradingEnv(
            data_path=mock_path,
            window_size=self.window_size,
            initial_balance=initial_balance,
            transaction_fee=self.transaction_fee,
            reward_pnl_scale=1.0,
            reward_cost_scale=0.5,
        )
        env.reset()

        action = 3  # Buy 100%
        # Buy amount = 0.1, fee = 0.1 * 0.01 = 0.001
        # Cash for crypto = 0.1 - 0.001 = 0.099. Price=1. Position = 0.099
        # This should work
        obs1, reward1, done1, _, info1 = env.step(action)
        assert env.balance < 1e-9
        assert abs(env.position - 0.099) < 1e-9

        # Try buying again with zero balance
        action = 1  # Buy 25%
        obs2, reward2, done2, _, info2 = env.step(action)
        assert env.balance < 1e-9  # Balance should remain near zero
        assert abs(env.position - 0.099) < 1e-9  # Position shouldn't change
        assert (
            abs(info2["transaction_cost"] - info1["transaction_cost"]) < 1e-9
        )  # No new cost

    def test_sell_with_zero_position(self):
        """Test attempting to sell when holding no position."""
        num_rows = 2 * self.window_size + 5
        mock_data_dict = {
            "open": [1.0] * num_rows,
            "high": [1.1] * num_rows,
            "low": [0.9] * num_rows,
            "close": [1.0] * num_rows,
            "volume": [100] * num_rows,
        }
        mock_path = create_mock_csv(mock_data_dict, self.temp_dir.name)
        env = TradingEnv(
            data_path=mock_path,
            window_size=self.window_size,
            initial_balance=self.initial_balance,
            transaction_fee=self.transaction_fee,
            reward_pnl_scale=1.0,
            reward_cost_scale=0.5,
        )
        env.reset()

        action = 6  # Sell 100%
        obs, reward, done, truncated, info = env.step(action)

        # Expect no change in state and potentially a penalty in reward
        assert env.position == 0
        assert env.balance == self.initial_balance
        assert info["transaction_cost"] == 0
        # Check if reward includes the penalty defined in env.step
        assert reward <= 0  # Should be negative or zero
        # Specific penalty check (assuming -0.1 from env code)
        # assert abs(reward - (-0.1)) < 1e-9
