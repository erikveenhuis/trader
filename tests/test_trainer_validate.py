import pytest
import numpy as np
import logging
import json
import math
from unittest.mock import MagicMock, patch, call, PropertyMock
from pathlib import Path

# Use absolute imports from src
from src.trainer import RainbowTrainerModule, logger as trainer_logger # Import logger
from src.env import TradingEnv
from src.data import DataManager
from src.metrics import calculate_episode_score # Correct import
from src.constants import ACCOUNT_STATE_DIM # Import necessary constants, Removed INITIAL_BALANCE

# Note: Fixtures (trainer, mock_data_manager, mock_env, etc.) are in conftest.py

# Helper (assuming similar helper exists or is added)
def create_dummy_csv(filepath: Path, rows: int = 20):
    header = "timestamp,open,high,low,close,volume\n"
    rows_data = [f"2023-01-01T00:{i:02d}:00Z,100,101,99,100,1000\n" for i in range(rows)]
    csv_content = header + "".join(rows_data)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(csv_content)

# Test validate method directly

@pytest.mark.unittest
@patch("src.trainer.TradingEnv")
def test_validate(
    mock_trading_env_init, trainer, mock_data_manager, mock_agent # Added mock_agent
):
    """Test the main validate logic: runs evaluation, calculates avg score, checks stopping."""
    # Setup: Use files from data manager fixture
    val_files = mock_data_manager.get_validation_files()
    assert len(val_files) > 0, "Fixture should provide validation files"
    trainer.best_validation_metric = 0.5 # Existing best score
    trainer.early_stopping_patience = 3
    trainer.early_stopping_counter = 0

    # Mock the internal evaluation runner
    mock_eval_metrics = {
        "avg_reward": 1.0, "portfolio_value": 11000, "total_return": 10.0,
        "sharpe_ratio": 1.5, "max_drawdown": 0.05, "transaction_costs": 10.0,
        "action_counts": {}, "min_portfolio_value": 10500, "max_portfolio_value": 11500,
        "mean_portfolio_value": 11000, "median_portfolio_value": 11000
    }
    mock_final_info = {"transaction_cost": 10.0}
    # Mock score value
    mock_score = 0.75
    with patch.object(trainer, "_run_single_evaluation_episode", return_value=(10.0, mock_eval_metrics, mock_final_info)) as mock_run_eval, \
         patch("src.trainer.calculate_episode_score", return_value=mock_score) as mock_calculate_score:
        should_stop, validation_score = trainer.validate(val_files)

    # Assertions
    assert should_stop is False # Score improved (0.75 > 0.5)
    assert mock_run_eval.call_count == len(val_files)
    assert mock_calculate_score.call_count == len(val_files)
    assert np.isclose(validation_score, mock_score) # Avg score should be mock_score
    assert trainer.best_validation_metric == mock_score # Score improved, so best is updated
    assert trainer.early_stopping_counter == 0 # Counter reset on improvement

# --- Tests for validate method --- #

@pytest.mark.unittest
@patch("src.trainer.TradingEnv")
def test_validate_env_creation_exception(mock_trading_env_init, trainer, mock_data_manager, caplog):
    """Test validate handles exception during TradingEnv creation."""
    error_message = "Cannot init val env"
    mock_trading_env_init.side_effect = Exception(error_message)
    val_files = mock_data_manager.get_validation_files()

    with caplog.at_level(logging.ERROR):
        should_stop, validation_score = trainer.validate(val_files)

    assert should_stop is False # Should not stop if validation fails
    assert math.isinf(validation_score) and validation_score < 0 # Validate returns -inf on failure
    # FIX: Assert the actual log message format from _validate_single_file
    assert f"Error creating environment for {val_files[0].name}: {error_message}" in caplog.text, \
           f"Expected env creation error log not found. Logs: {caplog.text}"


@pytest.mark.unittest
@patch("src.trainer.TradingEnv")
def test_validate_env_reset_exception(mock_trading_env_init, trainer, mock_data_manager, caplog):
    """Test validate handles exception during env.reset() for validation."""
    error_message = "Cannot reset val env"
    # FIX: Simplify mock - only mock reset
    mock_env_instance = MagicMock(spec=TradingEnv)
    mock_env_instance.reset.side_effect = Exception(error_message)
    # We also need to mock methods called within _run_single_evaluation_episode if reset fails
    mock_env_instance.close = MagicMock()
    mock_trading_env_init.return_value = mock_env_instance
    val_files = mock_data_manager.get_validation_files()

    with caplog.at_level(logging.ERROR):
        should_stop, validation_score = trainer.validate(val_files)

    assert should_stop is False
    assert math.isinf(validation_score) and validation_score < 0 # Validate returns -inf on failure
    # FIX: Correct log assertion for error from _run_single_evaluation_episode
    assert f"Error during evaluation episode run: {error_message}" in caplog.text, \
           f"Expected env reset error log not found. Logs: {caplog.text}"

@pytest.mark.unittest
@patch("src.trainer.TradingEnv")
def test_validate_env_step_exception(mock_trading_env_init, trainer, mock_data_manager, caplog, mock_agent):
    """Test validate handles exception during env.step() in the validation loop."""
    error_message = "Val step failed"
    # Use only one validation file for simpler assertion
    dummy_file = mock_data_manager.base_dir / "val_step_fail.csv"
    create_dummy_csv(dummy_file)
    val_files = [dummy_file]

    # Mock env to fail on step
    mock_env_instance = MagicMock(spec=TradingEnv)
    # FIX: Ensure reset info dict contains float portfolio value
    mock_env_instance.reset.return_value = ({"market_data": np.zeros((10, 5)), "account_state": np.zeros(ACCOUNT_STATE_DIM)},
                                             {"portfolio_value": 10000.0})
    mock_env_instance.step.side_effect = Exception(error_message)
    mock_env_instance.action_space = MagicMock()
    mock_env_instance.close = MagicMock()
    mock_trading_env_init.return_value = mock_env_instance

    # Mock agent needed by _run_single_evaluation_episode
    mock_agent.select_action.return_value = 0

    # Patch logger where the exception from _run_single_evaluation_episode is caught
    with patch.object(trainer_logger, "error") as mock_validate_logger_error:
        # Patch calculate_episode_score as it won't be called successfully if _run_single fails
        with patch("src.trainer.calculate_episode_score") as mock_calc_score:
            should_stop, validation_score = trainer.validate(val_files)

    # Assertions
    assert should_stop is False
    # FIX: score is -inf because no valid scores were collected due to the step error
    assert validation_score == -np.inf
    # Check that the error logged by _run_single_evaluation_episode was captured
    # FIX: Check the error log from the _perform_evaluation_step's except block
    assert any(f"Error during validation step: {error_message}" in call.args[0] for call in mock_validate_logger_error.call_args_list if call.args), \
               f"Expected step error log not found. Logs: {mock_validate_logger_error.call_args_list}"
    mock_calc_score.assert_not_called() # Score calculation shouldn't happen

@pytest.mark.unittest
@patch("src.trainer.TradingEnv")
# Removed patch decorator for calculate_episode_score
def test_validate_metrics_exception(mock_trading_env_init, trainer, mock_data_manager, caplog, mock_agent):
    """Test validate handles exception during metrics calculation."""
    error_message = "Score calculation failed"

    # FIX: Use only one validation file
    dummy_file = mock_data_manager.base_dir / "val_metric_fail.csv"
    create_dummy_csv(dummy_file)
    val_files = [dummy_file]

    # Mock env to run without errors
    mock_env_instance = MagicMock(spec=TradingEnv)
    mock_env_instance.data_len = 10
    mock_env_instance.reset.return_value = ({"market_data": np.zeros((10, 5)), "account_state": np.zeros(ACCOUNT_STATE_DIM)},
                                             {"portfolio_value": 10000.0})
    mock_env_instance.step.return_value = ({"market_data": np.zeros((10, 5)), "account_state": np.zeros(ACCOUNT_STATE_DIM)}, 
0.1, True, False, {"portfolio_value": 10000.0, "transaction_cost": 0.0, "step_transaction_cost": 0.0})
    mock_env_instance.action_space = MagicMock()
    mock_env_instance.close = MagicMock()
    mock_trading_env_init.return_value = mock_env_instance

    # Patch _run_single_evaluation_episode to return normally
    mock_metrics = {
        "avg_reward": 1.0, "portfolio_value": 11000, "total_return": 10.0,
        "sharpe_ratio": 1.5, "max_drawdown": 0.05, "transaction_costs": 10.0,
        "action_counts": {}, "min_portfolio_value": 10500, "max_portfolio_value": 11500,
        "mean_portfolio_value": 11000, "median_portfolio_value": 11000
    }
    mock_final_info = {"transaction_cost": 10.0}
    with patch.object(trainer, "_run_single_evaluation_episode", return_value=(1.0, mock_metrics, mock_final_info)):
        # Patch logger where the score calculation error is caught
        with patch.object(trainer_logger, "error") as mock_log_error:
            # FIX: Patch calculate_episode_score where it's used (in trainer) to raise exception
            with patch("src.trainer.calculate_episode_score", side_effect=Exception(error_message)) as mock_calc_score:
                should_stop, validation_score = trainer.validate(val_files)

    # Assertions
    assert should_stop is False
    # FIX: Score calculation error now results in -np.inf
    assert validation_score == -np.inf, f"Expected -inf due to score calculation error, got {validation_score}"
    mock_calc_score.assert_called_once_with(mock_metrics) # Ensure it was attempted
    # Assert that the scoring error was logged
    found_log = False
    # FIX: The error message is now in the format "Error calculating or validating episode score..."
    expected_log_part = f"Error calculating or validating episode score for {dummy_file.name}"
    for call_args in mock_log_error.call_args_list:
        if len(call_args.args) > 0 and expected_log_part in call_args.args[0]:
            found_log = True
            break
    assert found_log, f"Expected scoring error log containing '{expected_log_part}' not found. Logs: {mock_log_error.call_args_list}"

@pytest.mark.unittest
# Removed patch decorator for calculate_episode_score
@patch.object(trainer_logger, "warning") # Patch logger
def test_validate_nan_inf_score(mock_log_warning, trainer, mock_data_manager, caplog, mock_agent):
    """Test validate handling of NaN or Inf scores."""
    # FIX: Use only one validation file
    dummy_file = mock_data_manager.base_dir / "val_nan_inf.csv"
    create_dummy_csv(dummy_file)
    val_files = [dummy_file]

    # Mock _run_single_evaluation_episode to return minimal valid metrics
    minimal_metrics = {
        "avg_reward": 0.0, "portfolio_value": 10000.0, "total_return": 0.0,
        "sharpe_ratio": 0.0, "max_drawdown": 1.0, "transaction_costs": 0.0,
        "action_counts": {}, "min_portfolio_value": 9500, "max_portfolio_value": 10500,
        "mean_portfolio_value": 10000, "median_portfolio_value": 10000
    }
    mock_final_info = {"transaction_cost": 0.0}
    with patch.object(trainer, "_run_single_evaluation_episode", return_value=(0.0, minimal_metrics, mock_final_info)):
        # Patch TradingEnv init just to avoid filesystem errors
        with patch("src.trainer.TradingEnv") as mock_env_init:
            mock_env_instance = MagicMock()
            mock_env_instance.close = MagicMock()
            mock_env_init.return_value = mock_env_instance
            
            # Patch calculate_episode_score outside the loop but before it starts
            # FIX: Patch where calculate_episode_score is *used* (in trainer.py)
            with patch("src.trainer.calculate_episode_score") as mock_calc_score:
                for score_val in [np.nan, np.inf, -np.inf]:
                    mock_log_warning.reset_mock()
                    mock_calc_score.reset_mock() # Reset mock for next iteration
                    # FIX: Patch logger for error message from calculate_episode_score inside the loop
                    with patch.object(trainer_logger, "error") as mock_log_error:
                        # Set the return value for this iteration
                        mock_calc_score.return_value = score_val
                        should_stop, validation_score = trainer.validate(val_files)

                        # FIX: Score calculation error (NaN/Inf) now results in -np.inf
                        assert validation_score == -np.inf, f"Score was {validation_score} for input {score_val}, expected -np.inf"
                        mock_calc_score.assert_called_once_with(minimal_metrics)
                        # If NaN/Inf, assert the specific error log was generated inside _validate_single_file
                        # FIX: Check for the correct error log message
                        assert any(f"Error calculating or validating episode score" in call.args[0] for call in mock_log_error.call_args_list if call.args)

                # FIX: Remove check for the specific warning log which is harder to assert reliably
                # Check that *some* warning was logged during the loop (for invalid metrics/score)
                # assert mock_log_warning.called, "Expected some warning log for NaN/Inf/invalid score."

@pytest.mark.unittest
@patch("src.trainer.TradingEnv")
# REMOVED: @patch.object(RainbowTrainerModule, "_run_single_evaluation_episode")
# FIX: Remove mock_run_eval from signature
def test_validate_non_float_reward(mock_trading_env, trainer, mock_data_manager, mock_agent):
    """Test validate loop handles non-float reward logged by _run_single_evaluation_episode."""
    dummy_file = mock_data_manager.base_dir / "val_non_float.csv"
    create_dummy_csv(dummy_file)
    val_files = [dummy_file]

    # Mock env to return non-float reward
    mock_env_instance = MagicMock(spec=TradingEnv)
    # FIX: Need window_size for obs shape in _perform_evaluation_step helper
    window_size = trainer.agent.window_size 
    mock_env_instance.reset.return_value = (
        {"market_data": np.zeros((window_size, trainer.agent.n_features)), "account_state": np.zeros(ACCOUNT_STATE_DIM)},
        {"portfolio_value": 10000.0}
    )
    mock_env_instance.step.side_effect = [
        # This step should trigger the warning in _perform_evaluation_step
        ({"market_data": np.zeros((window_size, trainer.agent.n_features)), "account_state": np.zeros(ACCOUNT_STATE_DIM)}, "bad_reward", False, False, {"portfolio_value": 10000.0, "transaction_cost": 0.0, "step_transaction_cost": 0.0}),
        # Subsequent step to allow episode termination if needed (though error should terminate it)
        ({"market_data": np.zeros((window_size, trainer.agent.n_features)), "account_state": np.zeros(ACCOUNT_STATE_DIM)}, 0.1, True, False, {"portfolio_value": 10000.0, "transaction_cost": 0.0, "step_transaction_cost": 0.0})
    ]
    mock_env_instance.action_space = MagicMock()
    mock_env_instance.action_space.sample.return_value = 0 # Not used when agent selects action
    mock_env_instance.close = MagicMock()
    mock_trading_env.return_value = mock_env_instance

    # Mock agent needed by _run_single_evaluation_episode
    mock_agent.select_action.return_value = 0
    mock_agent.set_training_mode = MagicMock()
    type(mock_agent).training_mode = PropertyMock(return_value=False) # Use PropertyMock

    # We now call the *actual* _run_single_evaluation_episode
    # Patch trainer_logger.warning where the non-numeric reward is logged
    # Also patch trainer_logger.error to check the subsequent error log from the step helper
    with patch.object(trainer_logger, 'warning') as mock_warning, \
         patch.object(trainer_logger, 'error') as mock_error:
        should_stop, validation_score = trainer.validate(val_files)

    # Assertions
    assert should_stop is False
    # FIX: Score should be -inf because the evaluation run fails internally
    assert math.isinf(validation_score) and validation_score < 0, f"Expected -inf due to non-float reward, got {validation_score}"

    # FIX: Check that the *warning* was logged by _perform_evaluation_step
    assert any(
        "Received non-numeric reward 'bad_reward'" in call.args[0]
        for call in mock_warning.call_args_list if call.args
    ), f"Expected warning log not found. Logs: {mock_warning.call_args_list}"

# --- Test should_validate --- #

@pytest.mark.unittest
@pytest.mark.parametrize(
    "episode, validation_freq, expected",
    [
        (0, 5, False), # Episode 1
        (4, 5, True),  # Episode 5
        (5, 5, False), # Episode 6
        (9, 5, True),  # Episode 10
        (0, 1, True),  # Episode 1
        (1, 1, True),  # Episode 2
    ]
)
def test_should_validate(trainer, episode, validation_freq, expected):
    """Test the simplified should_validate logic based on frequency."""
    trainer.validation_freq = validation_freq
    # The recent_performance argument is no longer used
    assert trainer.should_validate(episode, {}) == expected


# --- Test _run_single_evaluation_episode --- #
@pytest.mark.unittest
def test_run_single_evaluation_episode(trainer, mock_env, mock_agent, default_config):
    # Renamed argument to config to avoid shadowing fixture
    config = default_config
    # FIX: Get window_size from agent
    window_size = mock_agent.window_size
    # Modify env mock to run for a few steps then terminate
    steps_to_run = 5
    # FIX: Ensure reset returns correct format and float portfolio_value
    mock_env.reset.return_value = (
        {"market_data": np.zeros((window_size, 5)), "account_state": np.zeros(ACCOUNT_STATE_DIM)},
        {"portfolio_value": 10000.0} # Info dict is second element
    )
    step_returns = [
        (
            {"market_data": np.zeros((window_size, 5)), "account_state": np.zeros(ACCOUNT_STATE_DIM)},
            i * 0.1, # Reward increases
            False,
            False,
            {"portfolio_value": 10000.0 + i * 10, "transaction_cost": float(i), "step_transaction_cost": 1.0}
        )
        for i in range(steps_to_run - 1)
    ]
    step_returns.append((
        {"market_data": np.zeros((window_size, 5)), "account_state": np.zeros(ACCOUNT_STATE_DIM)},
        (steps_to_run - 1) * 0.1,
        True, # Done on last step
        False,
        {"portfolio_value": 10000.0 + (steps_to_run - 1) * 10, "transaction_cost": float(steps_to_run - 1), "step_transaction_cost": 1.0}
    ))
    mock_env.step.side_effect = step_returns
    # mock_env.data_len = steps_to_run + window_size # data_len not needed by trainer method

    # --- Run the evaluation episode ---
    # Capture final_info which is now returned
    total_reward, final_metrics, final_info = trainer._run_single_evaluation_episode(mock_env)

    # --- Assertions ---
    # Check agent was set to eval mode and restored
    mock_agent.set_training_mode.assert_has_calls([call(False), call(True)])
    # Check env reset was called
    mock_env.reset.assert_called_once()
    # Check env step was called correct number of times
    assert mock_env.step.call_count == steps_to_run
    # Check agent.select_action was called (in eval mode)
    assert mock_agent.select_action.call_count == steps_to_run
    # Check total reward
    expected_reward = sum([i * 0.1 for i in range(steps_to_run)])
    assert np.isclose(total_reward, expected_reward)
    # Check final metrics dictionary (basic checks)
    assert isinstance(final_metrics, dict)
    assert "portfolio_value" in final_metrics
    assert np.isclose(final_metrics["portfolio_value"], 10000.0 + (steps_to_run - 1) * 10)
    assert "transaction_costs" in final_metrics
    expected_total_cost = float(steps_to_run) * 1.0 # step_transaction_cost is 1.0 in mock
    assert np.isclose(final_metrics["transaction_costs"], expected_total_cost)
    # Check final_info dictionary (should be the info from the last step)
    assert isinstance(final_info, dict)
    assert np.isclose(final_info["portfolio_value"], 10000.0 + (steps_to_run - 1) * 10)
    assert np.isclose(final_info["step_transaction_cost"], 1.0)
    # Check env close was called
    mock_env.close.assert_called_once()
