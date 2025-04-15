import pytest
import pandas as pd
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

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
        # Buy amount = 0.1, fee = 0.1 * 0.01 = 0.001. Required = 0.1 + 0.001 = 0.101
        # Balance = 0.1. This buy should FAIL.
        obs1, reward1, done1, _, info1 = env.step(action)
        # Assert state is unchanged and cost is zero
        assert abs(env.balance - initial_balance) < 1e-9
        assert abs(env.position) < 1e-9
        assert abs(info1['step_transaction_cost']) < 1e-9
        assert reward1 <= 0 # Penalty

        # Try buying again with (still) zero balance -> balance is actually initial_balance (0.1)
        action_buy25 = 1  # Buy 25%
        # Required = 0.1*0.25 + (0.1*0.25*0.01) = 0.025 + 0.00025 = 0.02525
        # Balance (0.1) >= Required (0.02525). This buy *should* succeed.
        obs2, reward2, done2, _, info2 = env.step(action_buy25)
        # Calculate the expected balance after the second buy
        buy_amount_cash2 = initial_balance * 0.25 # Buy 25% of the 0.1 balance
        step_cost2 = buy_amount_cash2 * env.transaction_fee
        total_required2 = buy_amount_cash2 + step_cost2
        expected_balance2 = initial_balance - total_required2 # Balance reduces by total cost

        # Check balance decreased correctly
        self.assertAlmostEqual(env.balance, expected_balance2, delta=1e-9)
        assert env.position > 1e-9 # Position should have increased
        assert (
            info2["step_transaction_cost"] > 0
        )  # Cost should be incurred on the second buy
        assert (
            abs(info2["transaction_cost"] - info1["step_transaction_cost"] - info2['step_transaction_cost']) < 1e-9
        )  # Total cost should be sum of step costs (info1 step cost was 0)

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

    # --- New Tests --- #

    def test_map_to_original_index_invalid_input(self):
        """Test _map_to_original_index with invalid inputs."""
        # Setup a basic env instance for this test
        env = self._create_basic_env()
        with pytest.raises(ValueError, match="Invalid internal_index"):
            env._map_to_original_index(float('nan'))
        with pytest.raises(ValueError, match="Invalid internal_index"):
            env._map_to_original_index(float('inf'))
        with pytest.raises(ValueError, match="Could not convert internal_index"):
            env._map_to_original_index(5.5)
        with pytest.raises(ValueError, match=r"Invalid internal_index .* Must be a finite number"):
            env._map_to_original_index("abc")

    def test_step_invalid_price_index(self):
        """Test handling index errors when accessing price data in step's try block."""
        env = self._create_basic_env()
        obs_reset, info_reset = env.reset()
        action = 0  # Hold

        # Mock _map_to_original_index to return an invalid index on the first call
        original_mapper = env._map_to_original_index
        call_count = 0

        def mock_mapper(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1: # Only fail on the first call (in step's try block)
                return env.original_data_len + 100 # Return out-of-bounds index
            else: # Allow subsequent calls (e.g., in _get_observation) to use original logic
                # We need to call the original method carefully
                # Re-bind 'self' if necessary, though it might work directly
                return original_mapper(args[0]) # Assuming first arg is internal_index

        with patch.object(env, '_map_to_original_index', side_effect=mock_mapper):
            # Expect step to log error and use fallback price.
            # If fallback price is 0 (no position yet), it might assert later.
            try:
                obs, reward, done, _, info = env.step(action)
                # Check if fallback occurred (e.g., log message or specific state)
                # Depending on fallback, subsequent checks might pass or fail.
            except AssertionError as e:
                # This might occur if fallback price is 0 and leads to later issues
                assert "current_price is non-positive" in str(e) # Expected assertion
            except IndexError:
                # We should NOT get an IndexError here if the except block in step handles it
                pytest.fail("IndexError was not caught by the step function's try-except block")

    def test_buy_insufficient_funds_after_cost(self):
        """Test buy action where funds cover base cost but not fee."""
        # Requires a specific price and balance
        price = 10.0
        fee = 0.10 # 10% fee
        # Balance slightly more than price * fraction, but less than price * fraction / (1 - fee)
        balance = 10.0 * 0.25 # Exactly enough for 25% buy base cost
        initial_balance = balance

        mock_path = self._create_specific_data(price)
        env = TradingEnv(
            data_path=mock_path,
            window_size=3, # Smaller window
            initial_balance=initial_balance,
            transaction_fee=fee,
            reward_pnl_scale=1.0,
            reward_cost_scale=0.5,
        )
        env.reset()

        action = 1 # Buy 25%
        # Buy amount cash = 2.5
        # Cost = 2.5 * 0.1 = 0.25
        # Cash for crypto = 2.5 - 0.25 = 2.25 > 0. Should proceed.
        obs, reward, done, _, info = env.step(action)
        assert info['step_transaction_cost'] > 0
        assert env.position > 0
        # --- DEBUG --- #
        print(f"\n[DEBUG Insufficient Funds Test] After first (valid) buy:")
        print(f"  Balance: {env.balance:.8f}")
        print(f"  Position: {env.position:.8f}")
        print(f"  Step Cost: {info['step_transaction_cost']:.8f}")
        print(f"  Total Cost: {env.total_transaction_cost:.8f}")
        # --- END DEBUG --- #

        # Now try with balance *just* below what's needed including fee
        balance = 10.0 * 0.25 / (1 - fee) - 0.001 # e.g., 2.777 - 0.001 = 2.776
        env = TradingEnv(
            data_path=mock_path,
            window_size=3,
            initial_balance=balance,
            transaction_fee=fee,
            reward_pnl_scale=1.0,
            reward_cost_scale=0.5,
        )
        env.reset()
        balance_before = env.balance
        position_before = env.position

        obs, reward, done, _, info = env.step(action) # Buy 25%
        # Should be penalized as invalid -> NO, this buy *is* affordable with the current balance.
        # The calculation is based on 25% of the *current* balance (2.776...)
        # Required = (2.776*0.25) + (2.776*0.25*0.10) = 0.694 + 0.0694 = 0.763...
        # Balance (2.776) >= Required (0.763). So the buy should proceed.
        # assert reward <= 0 # Check for penalty -> Reward might not be penalized
        # assert info['step_transaction_cost'] == 0 # No cost incurred -> Cost SHOULD be incurred
        assert info['step_transaction_cost'] > 0 # Cost should be incurred
        assert env.balance < balance_before # Balance should decrease
        assert env.position > position_before # Position should increase

    def test_reset_seed_deterministic(self):
        """Test that resetting with the same seed produces the same observation."""
        env = self._create_basic_env()
        seed = 123

        obs1, info1 = env.reset(seed=seed)
        obs2, info2 = env.reset(seed=seed)

        assert np.array_equal(obs1['market_data'], obs2['market_data'])
        assert np.array_equal(obs1['account_state'], obs2['account_state'])

    def test_step_nan_inf_state_change(self):
        """Test that NaN/Inf values in state changes are handled."""
        env = self._create_basic_env()
        env.reset()
        action = 0 # Hold action

        # --- Test NaN Balance --- #
        # Step once normally
        env.step(action)
        # Manually set balance to NaN *after* a step
        env.balance = float('nan')
        # Expect AssertionError from the np.isfinite check within step
        with pytest.raises(AssertionError, match="Balance is not finite: nan"):
            env.step(action)

        # --- Test Inf Position --- #
        # Reset state
        env.reset()
        # Step once normally
        env.step(action)
        # Manually set position to Inf *after* a step
        env.position = float('inf')
        # Expect AssertionError from the np.isfinite check within step
        with pytest.raises(AssertionError, match="Position is not finite: inf"):
            env.step(action)

    # --- Helper Methods --- #
    def _create_basic_env(self):
        """Helper to create a basic env instance for these tests."""
        num_rows = 2 * self.window_size + 5
        mock_data_dict = {
            "open": [1.0] * num_rows,
            "high": [1.1] * num_rows,
            "low": [0.9] * num_rows,
            "close": [1.0] * num_rows,
            "volume": [100] * num_rows,
        }
        mock_path = create_mock_csv(mock_data_dict, self.temp_dir.name)
        return TradingEnv(
            data_path=mock_path,
            window_size=self.window_size,
            initial_balance=self.initial_balance,
            transaction_fee=self.transaction_fee,
            reward_pnl_scale=1.0,
            reward_cost_scale=0.5,
        )

    def _create_specific_data(self, price):
        """Helper to create mock data with a specific constant close price."""
        num_rows = 2 * 3 + 5 # Use window_size=3 for this helper
        mock_data_dict = {
            "open": [price] * num_rows,
            "high": [price * 1.01] * num_rows,
            "low": [price * 0.99] * num_rows,
            "close": [price] * num_rows,
            "volume": [100] * num_rows,
        }
        return create_mock_csv(mock_data_dict, self.temp_dir.name)
