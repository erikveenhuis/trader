import pandas as pd
import numpy as np
import sys
from pathlib import Path
import unittest  # Import unittest
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

# Constants for testing
WINDOW_SIZE = 10
N_FEATURES = 5
ACCOUNT_STATE_DIM = 2
INITIAL_BALANCE = 10000.0
TRANSACTION_FEE = 0.001


# Helper function to create mock data CSV
def create_mock_csv(data_dict, temp_dir_path):
    df = pd.DataFrame(data_dict)
    path = Path(temp_dir_path) / "mock_data.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.mark.unittest
class TestTradingEnvCoreFunctionality(
    unittest.TestCase
):  # Inherit from unittest.TestCase
    """Tests core features of the TradingEnv."""

    def setUp(self):  # Rename back to setUp
        """Sets up the environment for tests in this class."""
        # Create temporary directory for mock data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mock_data_path = self._create_mock_data(self.temp_dir.name)

        self.window_size = WINDOW_SIZE
        self.initial_balance = INITIAL_BALANCE
        self.transaction_fee = TRANSACTION_FEE
        self.env = self._create_env()  # Use helper to create env

    def tearDown(self):
        # Clean up temporary directory
        self.temp_dir.cleanup()
        if hasattr(self, "env") and self.env:  # Close env if it exists
            self.env.close()

    def _create_mock_data(self, dir_name):
        """Creates standard mock data."""
        self.num_data_points = WINDOW_SIZE + 15  # Store this for tests
        mock_data_dict = {
            "open": [100.0 + i - 1 for i in range(self.num_data_points)],
            "high": [100.0 + i + 1 for i in range(self.num_data_points)],
            "low": [100.0 + i - 2 for i in range(self.num_data_points)],
            "close": [100.0 + i for i in range(self.num_data_points)],
            "volume": [1000.0 + i * 10 for i in range(self.num_data_points)],
        }
        return create_mock_csv(mock_data_dict, dir_name)

    def _create_env(
        self,
        data_path=None,
        window_size=None,
        initial_balance=None,
        transaction_fee=None,
        reward_pnl_scale=1.0,
        reward_cost_scale=0.5,
    ):
        """Helper to initialize environment with overrides."""
        dp = data_path if data_path else self.mock_data_path
        ws = window_size if window_size is not None else self.window_size
        ib = initial_balance if initial_balance is not None else self.initial_balance
        tf = transaction_fee if transaction_fee is not None else self.transaction_fee
        return TradingEnv(
            data_path=dp,
            window_size=ws,
            initial_balance=ib,
            transaction_fee=tf,
            reward_pnl_scale=reward_pnl_scale,
            reward_cost_scale=reward_cost_scale,
        )

    def test_initialization(self):
        """Test environment initialization parameters."""
        self.assertEqual(self.env.window_size, self.window_size)
        self.assertEqual(self.env.initial_balance, self.initial_balance)
        self.assertEqual(self.env.transaction_fee, self.transaction_fee)
        self.assertTrue(len(self.env.norm_data_array) > 0)
        self.assertTrue(len(self.env.original_close_prices) > 0)

    def test_reset(self):
        """Test environment reset and account state observation."""
        self.env.reset()
        action = 3  # Buy 100%
        # This action is INVALID because balance cannot cover 100% + fee
        buy_amount_cash = self.initial_balance * 1.0
        expected_cost = buy_amount_cash * self.transaction_fee
        total_required = buy_amount_cash + expected_cost
        self.assertLess(self.initial_balance + 1e-9, total_required) # Verify it's unaffordable

        obs_step, reward, done, _, info_step = self.env.step(action)
        # Balance and position should remain unchanged
        self.assertAlmostEqual(info_step["balance"], self.initial_balance, delta=1e-9)
        self.assertAlmostEqual(info_step["position"], 0, delta=1e-9)
        self.assertEqual(info_step['step_transaction_cost'], 0.0)
        self.assertLessEqual(reward, 0) # Should be penalized

        # Reset should still work correctly
        obs_reset, info_reset = self.env.reset()

        self.assertEqual(self.env.current_step, self.window_size)
        self.assertAlmostEqual(self.env.balance, self.initial_balance, delta=1e-9)
        self.assertAlmostEqual(self.env.position, 0, delta=1e-9)
        self.assertAlmostEqual(self.env.position_price, 0, delta=1e-9)
        self.assertAlmostEqual(self.env.total_transaction_cost, 0, delta=1e-9)
        self.assertEqual(len(self.env.portfolio_values), 1)
        self.assertAlmostEqual(
            self.env.portfolio_values[0], self.initial_balance, delta=1e-9
        )

        self.assertIsInstance(obs_reset, dict)
        self.assertIn("market_data", obs_reset)
        self.assertIn("account_state", obs_reset)
        self.assertEqual(obs_reset["market_data"].shape, (self.window_size, N_FEATURES))
        self.assertEqual(obs_reset["account_state"].shape, (ACCOUNT_STATE_DIM,))
        self.assertEqual(obs_reset["market_data"].dtype, np.float32)
        self.assertEqual(obs_reset["account_state"].dtype, np.float32)
        self.assertAlmostEqual(obs_reset["account_state"][0], 0.0, delta=1e-6)
        self.assertAlmostEqual(obs_reset["account_state"][1], 1.0, delta=1e-6)

        self.assertIsInstance(info_reset, dict)
        # ... (check info keys) ...
        self.assertEqual(info_reset["step"], self.window_size)
        self.assertAlmostEqual(info_reset["balance"], self.initial_balance, delta=1e-9)
        self.assertAlmostEqual(info_reset["position"], 0, delta=1e-9)
        self.assertAlmostEqual(
            info_reset["portfolio_value"], self.initial_balance, delta=1e-9
        )
        self.assertAlmostEqual(info_reset["transaction_cost"], 0, delta=1e-9)

    def test_step_buy(self):
        """Test a single buy step."""
        obs, info = self.env.reset()
        initial_balance = info["balance"]

        action = 2  # Buy 50%
        buy_fraction = 0.5
        price_at_step = self.env.original_close_prices[
            self.env._map_to_original_index(self.env.current_step)
        ]

        buy_cash = initial_balance * buy_fraction
        cost = buy_cash * self.transaction_fee
        # Position based on pre-fee cash amount
        cash_for_crypto = buy_cash
        expected_position = cash_for_crypto / price_at_step
        # Balance reduction includes the fee
        expected_balance = initial_balance - (buy_cash + cost)

        next_obs, reward, done, truncated, info_step = self.env.step(action)

        self.assertAlmostEqual(info_step["balance"], expected_balance, delta=1e-9)
        self.assertAlmostEqual(info_step["position"], expected_position, delta=1e-9)
        self.assertAlmostEqual(info_step["transaction_cost"], cost, delta=1e-9)
        self.assertAlmostEqual(self.env.balance, expected_balance, delta=1e-9)
        self.assertAlmostEqual(self.env.position, expected_position, delta=1e-9)
        self.assertAlmostEqual(self.env.total_transaction_cost, cost, delta=1e-9)


# NOTE: Nested classes TestEnvRewardComponents etc. are still not handled


@pytest.mark.unittest
class TestEnvRewardComponents(
    TestTradingEnvCoreFunctionality
):  # Inherit from CoreFunctionality

    def test_reward_components_profit(self):
        """Test reward components when taking profit."""
        obs, info_reset = self.env.reset()
        # Need previous_price for reward calculation
        previous_price_before_buy = self.env.previous_price # Price at reset
        portfolio_before_buy = info_reset["portfolio_value"]  # 10000

        # Buy 50% (action 2)
        action_buy = 2
        # Get price for buy step
        price_buy = self.env.original_close_prices[
            self.env._map_to_original_index(self.env.current_step)
        ] # 119.0
        # Calculate portfolio value BEFORE buy action, using the previous price
        previous_portfolio_before_buy_action = max(0, self.env.balance + self.env.position * previous_price_before_buy)
        _, reward_buy, _, _, info_buy = self.env.step(action_buy)
        portfolio_after_buy_at_buy_price = info_buy[
            "portfolio_value"
        ]  # 9995.0 (value using price=119.0)
        balance_after_buy = self.env.balance  # 5000
        position_after_buy = self.env.position  # 41.974
        cost_after_buy = info_buy["transaction_cost"]  # 5.0
        # Environment updates previous_price AFTER step
        previous_price_before_sell = price_buy # Price used in the buy step

        # Price at next step (internal 6, original 15) = 120.0
        price_sell = self.env.original_close_prices[
            self.env._map_to_original_index(self.env.current_step)
        ] # 120.0

        # Calculate portfolio value BEFORE sell action, using the previous price (price_buy)
        previous_portfolio_before_sell_action = max(0, self.env.balance + self.env.position * previous_price_before_sell)

        # Sell 100% (action 6)
        action_sell = 6
        obs_sell, reward_sell, _, _, info_sell = self.env.step(action_sell)
        portfolio_final = info_sell["portfolio_value"]  # 10031.93 (value using price=120.0)

        # Calculate expected reward for the SELL step using env logic: (PnL / initial_balance) * pnl_scale
        pnl_sell_step = portfolio_final - previous_portfolio_before_sell_action
        expected_reward_sell = (pnl_sell_step / self.initial_balance) * self.env.reward_pnl_scale

        # --- DEBUG ---
        print(f"\nDEBUG (profit): Prev Port Before Sell (@{previous_price_before_sell:.2f}): {previous_portfolio_before_sell_action:.4f}")
        print(f"DEBUG (profit): Final Port After Sell (@{price_sell:.2f}): {portfolio_final:.4f}")
        print(f"DEBUG (profit): PnL Sell Step: {pnl_sell_step:.4f}")
        print(f"DEBUG (profit): Expected Reward Sell: {expected_reward_sell:.8f}")
        print(f"DEBUG (profit): Actual Reward Sell:   {reward_sell:.8f}")
        # --- END DEBUG ---

        self.assertAlmostEqual(reward_sell, expected_reward_sell, delta=1e-6) # Use tighter delta

    def test_reward_components_loss(self):
        """Test reward components when taking a loss."""
        # Set specific prices for this test via mocking the original prices
        # This avoids reliance on the default mock data which might change.
        original_prices = self.env.original_close_prices.copy()
        num_orig_points = len(original_prices)
        buy_step_internal = self.env.window_size # First step after reset
        sell_step_internal = buy_step_internal + 1
        buy_orig_index = self.env._map_to_original_index(buy_step_internal)
        sell_orig_index = self.env._map_to_original_index(sell_step_internal)

        # Ensure indices are valid
        assert 0 <= buy_orig_index < num_orig_points, f"Buy index {buy_orig_index} out of bounds"
        assert 0 <= sell_orig_index < num_orig_points, f"Sell index {sell_orig_index} out of bounds"

        price_buy = 110.0 # Price for the buy step
        price_sell = 100.0 # Price for the sell step (lower, causing loss)
        original_prices[buy_orig_index] = price_buy
        original_prices[sell_orig_index] = price_sell
        self.env.original_close_prices = original_prices

        # Reset to ensure the modified prices are used correctly from the start
        obs, info_reset = self.env.reset()
        previous_price_before_buy = self.env.previous_price
        portfolio_start = info_reset["portfolio_value"]  # 10000

        # Calculate portfolio value BEFORE buy action
        previous_portfolio_before_buy_action = max(0, self.env.balance + self.env.position * previous_price_before_buy)

        # Buy 50% (action 2) at price_buy=110.0
        action_buy = 2
        _, reward_buy, _, _, info_buy = self.env.step(action_buy)
        portfolio_after_buy_at_buy_price = info_buy["portfolio_value"]
        previous_price_before_sell = price_buy # Price used in buy step becomes previous for sell step

        # Calculate portfolio value BEFORE sell action, using previous_price (price_buy)
        previous_portfolio_before_sell_action = max(0, self.env.balance + self.env.position * previous_price_before_sell)

        # Sell 100% (action 6) at price_sell=100.0
        action_sell = 6
        obs_sell, reward_sell, _, _, info_sell = self.env.step(action_sell)
        portfolio_final = info_sell["portfolio_value"] # Portfolio value after sell, calculated using price_sell

        # Calculate expected reward for the SELL step using env logic
        pnl_sell_step = portfolio_final - previous_portfolio_before_sell_action
        expected_reward_sell = (pnl_sell_step / self.initial_balance) * self.env.reward_pnl_scale

        # --- DEBUG ---
        print(f"\nDEBUG (loss): Price Buy: {price_buy:.2f}, Price Sell: {price_sell:.2f}")
        print(f"DEBUG (loss): Prev Port Before Sell (@{previous_price_before_sell:.2f}): {previous_portfolio_before_sell_action:.4f}")
        print(f"DEBUG (loss): Final Port After Sell (@{price_sell:.2f}): {portfolio_final:.4f}")
        print(f"DEBUG (loss): PnL Sell Step: {pnl_sell_step:.4f}")
        print(f"DEBUG (loss): Expected Reward Sell: {expected_reward_sell:.8f}")
        print(f"DEBUG (loss): Actual Reward Sell:   {reward_sell:.8f}")
        # --- END DEBUG ---

        self.assertAlmostEqual(reward_sell, expected_reward_sell, delta=1e-6) # Use tighter delta

    def test_reward_components_cost_penalty(self):
        """Test that the reward reflects PnL, not just cost penalty."""
        obs_reset, info_reset = self.env.reset()
        previous_price_before_buy = self.env.previous_price
        portfolio_start = info_reset["portfolio_value"]  # 10000

        # Price at internal step 5 (index 14 in original data) = 119.0
        price_buy = self.env.original_close_prices[
            self.env._map_to_original_index(self.env.current_step)
        ]

        # Calculate portfolio value BEFORE buy action
        previous_portfolio_before_buy_action = max(0, self.env.balance + self.env.position * previous_price_before_buy)

        # Buy 50% (action 2)
        action_buy = 2
        obs_buy, reward_buy, _, _, info_buy = self.env.step(action_buy)
        portfolio_after_buy = info_buy["portfolio_value"] # Portfolio value AFTER buy, calculated using price_buy

        # Calculate expected reward for the BUY step using env logic
        pnl_buy_step = portfolio_after_buy - previous_portfolio_before_buy_action
        expected_reward_buy = (pnl_buy_step / self.initial_balance) * self.env.reward_pnl_scale

        # --- DEBUG ---
        print(f"\nDEBUG (cost): Prev Port Before Buy (@{previous_price_before_buy:.2f}): {previous_portfolio_before_buy_action:.4f}")
        print(f"DEBUG (cost): Port After Buy (@{price_buy:.2f}): {portfolio_after_buy:.4f}")
        print(f"DEBUG (cost): PnL Buy Step: {pnl_buy_step:.4f}")
        print(f"DEBUG (cost): Expected Reward Buy: {expected_reward_buy:.8f}")
        print(f"DEBUG (cost): Actual Reward Buy:   {reward_buy:.8f}")
        # --- END DEBUG ---

        self.assertAlmostEqual(reward_buy, expected_reward_buy, delta=1e-6) # Check PnL reward


@pytest.mark.unittest
class TestEnvActions(TestTradingEnvCoreFunctionality):  # Inherit from CoreFunctionality

    # Action definitions based on TradingEnv:
    # 0: Hold
    # 1: Buy 25%
    # 2: Buy 50%
    # 3: Buy 100%
    # 4: Sell 25%
    # 5: Sell 50%
    # 6: Sell 100%

    def test_action_buy_25(self):
        """Test action 1 (Buy 25%)."""
        obs, info = self.env.reset()
        action = 1

        # Get the price the env WILL use for this step
        price_for_step = self.env.original_close_prices[
            self.env._map_to_original_index(self.env.current_step)
        ]
        self.assertGreater(price_for_step, 1e-9)  # Ensure valid price

        balance_before = self.initial_balance
        buy_amount_cash = balance_before * 0.25
        step_transaction_cost = buy_amount_cash * self.transaction_fee
        # Position based on pre-fee cash amount
        cash_for_crypto = buy_amount_cash
        expected_position = cash_for_crypto / price_for_step
        # Balance reduction includes fee
        expected_balance = balance_before - (buy_amount_cash + step_transaction_cost)

        _, _, _, _, info_step = self.env.step(action)

        self.assertAlmostEqual(self.env.position, expected_position, places=5)
        self.assertAlmostEqual(self.env.balance, expected_balance, places=5)
        self.assertAlmostEqual(
            info_step["transaction_cost"], step_transaction_cost, places=5
        )
        # First buy, cost basis should reflect the crypto value bought (before fee)
        # Basis = (0*0 + cash_for_crypto) / expected_position = price_for_step
        self.assertAlmostEqual(self.env.position_price, price_for_step, places=5)

    def test_action_sell_25(self):
        """Test action 4 (Sell 25%)."""
        obs, info = self.env.reset()
        # Buy 50% first (action 2) - this should be affordable
        action_buy = 2
        price_buy = self.env.original_close_prices[
            self.env._map_to_original_index(self.env.current_step)
        ]
        # Verify buy is affordable
        buy_amount_cash = self.initial_balance * 0.5
        cost = buy_amount_cash * self.transaction_fee
        self.assertGreaterEqual(self.initial_balance + 1e-9, buy_amount_cash + cost)
        # Step with the buy action
        self.env.step(action_buy)
        position_after_buy = self.env.position
        balance_after_buy = self.env.balance
        # Basis after first buy should be price_buy
        cost_basis = self.env.position_price
        self.assertAlmostEqual(cost_basis, price_buy, places=5)
        self.assertGreater(position_after_buy, 1e-9) # Make sure buy actually happened

        # Now sell 25%
        action_sell = 4
        price_sell = self.env.original_close_prices[
            self.env._map_to_original_index(self.env.current_step)
        ]
        self.assertGreater(price_sell, 1e-9)

        sell_fraction = 0.25
        sell_amount_crypto = position_after_buy * sell_fraction
        cash_before_fee = sell_amount_crypto * price_sell
        step_transaction_cost = cash_before_fee * self.transaction_fee
        cash_change = cash_before_fee - step_transaction_cost
        position_change = -sell_amount_crypto

        expected_position = position_after_buy + position_change
        expected_balance = balance_after_buy + cash_change

        _, _, _, _, info_step = self.env.step(action_sell)

        self.assertAlmostEqual(self.env.position, expected_position, places=4)
        self.assertAlmostEqual(self.env.balance, expected_balance, places=4)
        # Basis shouldn't change on sell unless position becomes zero
        if expected_position > 1e-9:
            self.assertAlmostEqual(self.env.position_price, cost_basis, places=4)
        else:
            self.assertAlmostEqual(self.env.position_price, 0, places=4)

    def test_cost_basis_multiple_buys(self):
        """Test average cost basis calculation after multiple buys."""
        obs, info = self.env.reset()

        # Buy 1: 25% (action 1)
        action1 = 1
        price1 = self.env.original_close_prices[
            self.env._map_to_original_index(self.env.current_step)
        ]
        _, _, _, _, info1 = self.env.step(action1)
        pos1 = self.env.position
        bal1 = self.env.balance
        basis1 = self.env.position_price
        cost1 = info1["transaction_cost"]
        # Basis after first buy should be price1
        self.assertAlmostEqual(basis1, price1, places=4)

        # Buy 2: 50% of *remaining* balance (action 2)
        action2 = 2
        price2 = self.env.original_close_prices[
            self.env._map_to_original_index(self.env.current_step)
        ]
        balance_before_buy2 = bal1
        buy_amount_cash2 = balance_before_buy2 * 0.50
        step_cost2 = buy_amount_cash2 * self.transaction_fee
        # Basis uses pre-fee amount
        cash_for_crypto2 = buy_amount_cash2
        pos_change2 = cash_for_crypto2 / price2

        # Expected average cost basis calculation
        old_value = pos1 * basis1 # Value of existing position
        new_value = cash_for_crypto2 # Value of crypto bought (before fee)
        total_position = pos1 + pos_change2
        expected_basis2 = (
            (old_value + new_value) / total_position if total_position > 1e-9 else 0
        )

        _, _, _, _, info2 = self.env.step(action2)

        expected_pos2 = total_position
        # Balance reduction includes fee
        expected_bal2 = balance_before_buy2 - (buy_amount_cash2 + step_cost2)

        self.assertAlmostEqual(self.env.position, expected_pos2, places=4)
        self.assertAlmostEqual(self.env.balance, expected_bal2, places=4)
        # Check basis calculation
        self.assertAlmostEqual(self.env.position_price, expected_basis2, places=4)
        expected_cum_cost2 = cost1 + step_cost2
        self.assertAlmostEqual(info2["transaction_cost"], expected_cum_cost2, places=4)


@pytest.mark.unittest
class TestEnvInternals(
    TestTradingEnvCoreFunctionality
):  # Inherit from CoreFunctionality

    def test_normalization_values(self):
        """Test the output of the normalization process."""
        # Use simple data where normalization is easy to calculate
        num_points = 10
        window_size = 3
        mock_data_dict = {
            "open": [10.0] * num_points,
            "high": [float(10 + i) for i in range(num_points)],  # 10 to 19
            "low": [5.0] * num_points,
            "close": [float(i) for i in range(num_points)],  # 0 to 9
            "volume": [100.0] * num_points,
        }
        mock_path = create_mock_csv(mock_data_dict, self.temp_dir.name)
        env = self._create_env(data_path=mock_path, window_size=window_size)

        # Check values at a specific step (e.g., internal step index 5)
        # This corresponds to original data index 5+3-1 = 7
        # The window includes original indices 5, 6, 7
        # Prices: high=[15, 16, 17], close=[5, 6, 7]
        obs = env._get_observation()  # This uses current_step=window_size=3

        # Let's advance the environment step manually for testing internal arrays
        env.current_step = 5
        # Window used for normalization of step 5 uses original data indices 3,4,5
        # High: [13, 14, 15] -> min=13, max=15. Value at index 5 is 15.
        # Norm High = (15 - 13) / (15 - 13) = 1.0
        # Close: [3, 4, 5] -> min=3, max=5. Value at index 5 is 5.
        # Norm Close = (5 - 3) / (5 - 3) = 1.0

        # Check the normalized data array directly (internal index = step)
        # Step 5 corresponds to index 5 in the normalized array (after dropping NaNs)
        # The original data had 10 rows. After normalization with window=3, first 2 rows are dropped.
        # So, original index 2 -> norm index 0, orig index 5 -> norm index 3
        norm_index = env.current_step - window_size  # 5 - 3 = 2
        norm_data_step = env.norm_data_array[norm_index]
        self.assertAlmostEqual(norm_data_step[1], 1.0)  # norm_high
        self.assertAlmostEqual(norm_data_step[3], 1.0)  # norm_close

    def test_observation_account_state(self):
        """Test the normalized account state values in observations."""
        obs_reset, info_reset = self.env.reset()
        # Initial state: 0 position, balance=initial
        self.assertAlmostEqual(
            obs_reset["account_state"][0], 0.0
        )  # Normalized position
        self.assertAlmostEqual(
            obs_reset["account_state"][1], 1.0 if self.initial_balance > 1e-9 else 0.0
        )  # Initial balance normalized

        # Price at internal step 5 (index 14 in original data) = 119.0
        price_buy = self.env.original_close_prices[
            self.env._map_to_original_index(self.env.current_step)
        ]

        # Buy 50% (action 2)
        action_buy = 2
        obs_buy, _, _, _, info_buy = self.env.step(action_buy)
    
        # Observation normalization uses the price at the step where the observation is generated
        # which is price_buy (119.0) in this case, as the step function returns the NEXT observation
        # after updating the state based on the action and the price (price_buy) at that step.
        # The debug output confirms this uses the price from the step (119.0).
        balance_after_buy = self.env.balance  # 5000.0
        position_after_buy = self.env.position  # 41.9747
    
        # Use price_buy (119.0) for calculating the expected normalization
        portfolio_value_at_obs_time = (
            balance_after_buy + position_after_buy * price_buy
        )  # 5000 + 41.9747 * 119 = 9995.0
        expected_norm_pos = (
            (position_after_buy * price_buy) / portfolio_value_at_obs_time
            if portfolio_value_at_obs_time > 1e-9
            else 0
        )  # (41.9747 * 119) / 9995.0 = 0.49975
        expected_norm_bal = (
            balance_after_buy / self.initial_balance
        )  # 5000 / 10000 = 0.5
    
        self.assertAlmostEqual(obs_buy["account_state"][0], expected_norm_pos, places=5)
        self.assertAlmostEqual(obs_buy["account_state"][1], expected_norm_bal, places=5)

    def test_step_hold(self):  # Example of another converted test
        """Test holding a position without trading."""
        obs, info = self.env.reset()
        # Step 1: Buy 50%
        action_buy = 2
        _, _, _, _, info_buy = self.env.step(action_buy)
        position_after_buy = self.env.position
        balance_after_buy = self.env.balance
        cost_after_buy = info_buy["transaction_cost"]
        cost_basis_after_buy = self.env.position_price

        # Step 2: Hold
        action_hold = 0
        _, _, _, _, info_hold = self.env.step(action_hold)
        total_cumulative_cost = info_hold["transaction_cost"]

        # Assert that the TOTAL cumulative cost hasn't changed from after the buy step
        self.assertAlmostEqual(total_cumulative_cost, cost_after_buy, places=5)
        # Assert final position and balance match the state after buy step
        self.assertAlmostEqual(self.env.position, position_after_buy, places=5)
        self.assertAlmostEqual(self.env.balance, balance_after_buy, places=5)
        # Check cost basis remains unchanged during hold
        self.assertAlmostEqual(self.env.position_price, cost_basis_after_buy, places=5)


# Example usage (can be removed or kept for manual testing)
# if __name__ == "__main__":
#     unittest.main()
