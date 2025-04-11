import pytest
import torch
import numpy as np
import os
import logging  # Added for handler setting
import shutil  # Added for fixture cleanup
from unittest.mock import MagicMock, patch, call, ANY
from pathlib import Path

# Use absolute imports from src
from src.trainer import RainbowTrainerModule
from src.agent import RainbowDQNAgent, ACCOUNT_STATE_DIM
from src.env import TradingEnv
from src.data import DataManager
from src.metrics import calculate_composite_score, calculate_episode_score
from src.constants import ACCOUNT_STATE_DIM


# Constants for tests
WINDOW_SIZE = 10
INITIAL_BALANCE = 10000
TRANSACTION_FEE = 0.001


# Fixtures
@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture
def mock_data_path(tmp_path):
    mock_data_file = tmp_path / "mock_data.csv"
    # Create a simple CSV with enough rows for window_size + a few steps
    rows = WINDOW_SIZE + 10
    header = "timestamp,open,high,low,close,volume\n"
    data_rows = [
        f"2023-01-01T00:{i:02d}:00Z,100,101,99,100,1000\n" for i in range(rows)
    ]
    mock_data_file.write_text(header + "".join(data_rows))
    return str(mock_data_file)


@pytest.fixture
def default_config(tmp_path):
    # Minimal config for testing trainer logic
    # Paths are relative to the test execution directory
    return {
        "agent": {
            "network": "rainbow_transformer",
            "window_size": WINDOW_SIZE, # Use constant
            "n_features": 5, # Needs to match mock data
            "num_actions": 7,
            "lr": 1e-4,
            "gamma": 0.99,
            "buffer_size": 1000,
            "batch_size": 4,
            "target_update_freq": 10,
            "n_steps": 3,
            "num_atoms": 51,
            "v_min": -10.0,
            "v_max": 10.0,
            "alpha": 0.6,
            "beta_start": 0.4,
            "beta_frames": 100,
            "noisy_std": 0.1,
            "seed": 42,
        },
        "environment": {
            "initial_balance": INITIAL_BALANCE, # Use constant
            "transaction_fee": TRANSACTION_FEE, # Use constant
            "window_size": WINDOW_SIZE, # Use constant
            "reward_pnl_scale": 1.0,
            "reward_cost_scale": 0.5,
            # Add other env-specific keys if needed by TradingEnv
        },
        "trainer": {
            "seed": 42,
            "warmup_steps": 10, # Reduced for faster tests
            "update_freq": 4,
            "log_freq": 5,
            "validation_freq": 2,
            "checkpoint_save_freq": 2,
            "reward_window": 2,
            "early_stopping_patience": 3,
            "min_validation_threshold": 0.0,
        },
        "run": {
            "mode": "train",
            "episodes": 5, # Default number of episodes for tests
            "model_dir": str(tmp_path / "models"), # Use tmp_path
            "resume": False,
        },
    }


@pytest.fixture
def mock_agent(default_config):
    agent = MagicMock(spec=RainbowDQNAgent)
    agent.network = MagicMock()
    agent.target_network = MagicMock()
    agent.optimizer = MagicMock()
    # Configure attributes based on default_config where needed by trainer
    agent.gamma = default_config["agent"]["gamma"]
    agent.lr = default_config["agent"]["lr"]
    agent.batch_size = default_config["agent"]["batch_size"]
    agent.target_update_freq = default_config["agent"]["target_update_freq"]
    agent.num_atoms = default_config["agent"]["num_atoms"]
    agent.v_min = default_config["agent"]["v_min"]
    agent.v_max = default_config["agent"]["v_max"]
    agent.n_steps = default_config["agent"]["n_steps"]
    # Mock buffer attributes needed by trainer
    agent.buffer = MagicMock()
    agent.buffer.alpha = default_config["agent"]["alpha"]
    agent.buffer.beta_start = default_config["agent"]["beta_start"]
    agent.buffer.__len__.return_value = agent.batch_size # Default to full enough
    agent.select_action.return_value = 0 # Default action
    agent.learn.return_value = 0.1 # Default loss
    agent.save_model = MagicMock()
    agent.load_model = MagicMock()
    agent.total_steps = 0 # Initialize step count
    agent.training_mode = True
    agent.window_size = default_config["agent"]["window_size"]
    agent.n_features = default_config["agent"]["n_features"]
    agent.num_actions = default_config["agent"]["num_actions"]
    return agent


@pytest.fixture
def mock_data_manager(tmp_path):
    # Create dummy files
    train_dir = tmp_path / "processed" / "train"
    val_dir = tmp_path / "processed" / "validation"
    test_dir = tmp_path / "processed" / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_files = [train_dir / "train_data_1.csv", train_dir / "train_data_2.csv"]
    val_files = [val_dir / "val_data_1.csv"]
    test_files = [test_dir / "test_data_1.csv"]

    for f in train_files + val_files + test_files:
        create_dummy_csv(f, rows=WINDOW_SIZE + 20) # Ensure enough rows

    manager = MagicMock(spec=DataManager)
    manager.base_dir = tmp_path
    manager.get_training_files.return_value = train_files
    manager.get_validation_files.return_value = val_files
    manager.get_test_files.return_value = test_files
    # Make get_random_training_file deterministic for tests
    manager.get_random_training_file.return_value = train_files[0]
    return manager


@pytest.fixture
def mock_env():
    # Basic mock env, configure further in tests if needed
    env = MagicMock(spec=TradingEnv)
    env.reset.return_value = (
        {"market_data": np.zeros((WINDOW_SIZE, 5)), "account_state": np.zeros(ACCOUNT_STATE_DIM)},
        {"portfolio_value": INITIAL_BALANCE}
    )
    env.step.return_value = (
        {"market_data": np.zeros((WINDOW_SIZE, 5)), "account_state": np.zeros(ACCOUNT_STATE_DIM)},
        0.1, # reward
        False, # done
        False, # truncated
        {"portfolio_value": INITIAL_BALANCE, "transaction_cost": 0.0, "step_transaction_cost": 0.0} # info
    )
    env.action_space = MagicMock()
    env.action_space.sample.return_value = 0
    env.action_space.contains = MagicMock(return_value=True)
    env.close = MagicMock()
    env.data_len = WINDOW_SIZE + 10 # Default length
    return env


@pytest.fixture
def trainer(mock_agent, mock_data_manager, default_config, tmp_path):
    device = torch.device("cpu")
    # Ensure model_dir in config uses the test's tmp_path
    default_config["run"]["model_dir"] = str(tmp_path / "models")
    trainer = RainbowTrainerModule(
        agent=mock_agent,
        device=device,
        data_manager=mock_data_manager,
        config=default_config,
    )
    return trainer


# --- Test Class --- #

# Test Class structure is good for related tests

@pytest.mark.unittest
def test_trainer_init(trainer, default_config, tmp_path):
    assert trainer.agent is not None
    assert trainer.device == torch.device("cpu")
    assert trainer.data_manager is not None
    assert trainer.config == default_config
    assert trainer.best_validation_metric == -np.inf
    assert trainer.early_stopping_counter == 0
    assert trainer.model_dir == str(tmp_path / "models")
    assert os.path.exists(trainer.model_dir)
    # Use the base prefix name
    assert trainer.best_model_base_prefix == str(
        tmp_path / "models" / "rainbow_transformer_best"
    )
    assert trainer.latest_trainer_checkpoint_path == str(
        tmp_path / "models" / "checkpoint_trainer_latest.pt"
    )
    # Store the BASE prefix for the best checkpoint.
    assert trainer.best_trainer_checkpoint_base_path == str(
        tmp_path / "models" / "checkpoint_trainer_best"
    )


@pytest.mark.unittest
@patch("torch.save")
def test_save_trainer_checkpoint(mock_torch_save, trainer):
    episode = 10
    total_steps = 1000
    trainer.best_validation_metric = 0.5
    trainer.early_stopping_counter = 1

    # Test saving latest
    trainer._save_trainer_checkpoint(episode, total_steps, is_best=False)
    expected_checkpoint_latest_only = {
        "episode": episode,
        "total_train_steps": total_steps,
        "best_validation_metric": 0.5,
        "early_stopping_counter": 1,
    }
    calls_latest_only = [call(expected_checkpoint_latest_only, trainer.latest_trainer_checkpoint_path)]
    mock_torch_save.assert_has_calls(calls_latest_only)
    mock_torch_save.reset_mock()

    # Test saving best
    test_score = 0.8
    trainer.best_validation_metric = test_score # Update best score

    expected_best_checkpoint = {
        "episode": episode,
        "total_train_steps": total_steps,
        "best_validation_metric": test_score, # Should use the new best score
        "early_stopping_counter": 1, # Counter doesn't reset here
        "validation_score": test_score # Score should be added when saving best
    }

    # Call save for best, providing a score
    trainer._save_trainer_checkpoint(episode, total_steps, is_best=True, validation_score=test_score)

    # Verify torch.save was called for both latest and best (with scored name)
    expected_best_path = f"{trainer.best_trainer_checkpoint_base_path}_score_{test_score:.4f}.pt"
    calls = [
        call(expected_best_checkpoint, trainer.latest_trainer_checkpoint_path), # Check latest save call (expects same dict as best)
        call(expected_best_checkpoint, expected_best_path) # Check best save call with score
    ]

    # Now assert the calls made to the mocked torch.save
    mock_torch_save.assert_has_calls(calls, any_order=False)

@pytest.mark.unittest
def test_run_single_evaluation_episode(trainer, mock_env, mock_agent, default_config):
    # Renamed argument to config to avoid shadowing fixture
    config = default_config
    # Reset side effect for consistent evaluation run
    dummy_market = np.random.rand(
        config["agent"]["window_size"], config["agent"]["n_features"]
    ).astype(np.float32)
    dummy_account = np.random.rand(ACCOUNT_STATE_DIM).astype(np.float32)
    next_obs = {"market_data": dummy_market, "account_state": dummy_account}
    # Ensure float for portfolio_value in this test's specific mock setup
    next_info = {"portfolio_value": 10100.0, "price": 102.0, "transaction_cost": 1.0, "step_transaction_cost": 0.5}
    mock_env.step.side_effect = [
        (next_obs, 10.0, False, False, next_info),
        (next_obs, 5.0, True, False, next_info),  # Done
    ]
    mock_agent.select_action.return_value = 0  # Hold

    # Test the renamed method
    total_reward, metrics, final_info = trainer._run_single_evaluation_episode(mock_env)

    assert isinstance(total_reward, float)
    assert total_reward == 15.0  # 10 + 5
    assert isinstance(metrics, dict)
    assert "portfolio_value" in metrics
    assert isinstance(final_info, dict)
    assert final_info == next_info # Should be the info from the last step
    mock_agent.select_action.assert_called()
    mock_env.reset.assert_called_once()
    assert mock_env.step.call_count == 2
    assert trainer.agent.training_mode # Should be reset to True after eval


@pytest.mark.unittest
@patch("json.dump")
def test_validate(
    mock_json_dump, trainer, mock_data_manager, mock_env
):
    val_files = mock_data_manager.get_validation_files()
    trainer.best_validation_metric = 0.5  # Initial best score
    trainer.early_stopping_counter = 0

    # Mock _run_single_evaluation_episode to return consistent results
    mock_metrics = {
        "avg_reward": 10.0,
        "portfolio_value": 11000,
        "total_return": 10.0,
        "sharpe_ratio": 1.2,
        "max_drawdown": 0.05,
        "win_rate": 0.6,
        "avg_action": 0.5,
        "transaction_costs": 50.0,
        # Add other keys if calculate_episode_score uses them
    }
    # Use the real episode score calculation for expected value
    expected_validation_score = calculate_episode_score(mock_metrics)
    with patch.object(
        trainer, "_run_single_evaluation_episode", return_value=(10.0, mock_metrics, {})
    ) as mock_run_episode:
        should_stop, validation_score = trainer.validate(val_files)

    assert should_stop is False
    assert np.isclose(validation_score, expected_validation_score)
    assert np.isclose(trainer.best_validation_metric, expected_validation_score) # Updated best score
    assert trainer.early_stopping_counter == 0  # Reset counter
    mock_run_episode.assert_called_once()  # Called once for the single validation file
    mock_json_dump.assert_called_once()  # Results should be saved


@pytest.mark.unittest
# Patch the evaluation method using decorator
@patch.object(RainbowTrainerModule, "_run_single_evaluation_episode")
def test_validate_early_stopping(mock_run_single_episode, trainer, mock_data_manager, tmp_path):
    trainer.best_validation_metric = 0.8  # Set a high best score
    trainer.early_stopping_patience = 3
    trainer.early_stopping_counter = trainer.early_stopping_patience - 1 # One step away
    trainer.best_model_base_prefix = str(tmp_path / "best_model") # Set base prefix for test

    # Define the zeroed metrics _run_single_evaluation_episode should return
    mock_returned_metrics = {
        "avg_reward": 0.0,
        "portfolio_value": 0.0,
        "total_return": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 1.0, # Worst drawdown
        "win_rate": 0.0,
        "transaction_costs": 0.0,
        # Add other keys if calculate_episode_score uses them
    }
    # Mock needs to return the defined metrics, not empty dict
    mock_run_single_episode.return_value = (-5.0, mock_returned_metrics, {}) # Corrected mock return

    # Create a dummy validation file path
    dummy_file = tmp_path / "dummy_val.csv"
    create_dummy_csv(dummy_file) # Use helper

    # Use the correct score function
    expected_score = calculate_episode_score(mock_returned_metrics)

    # Patch TradingEnv constructor within this test scope to avoid file errors
    with patch("src.trainer.TradingEnv") as mock_env_constructor:
        mock_env_instance = MagicMock(spec=TradingEnv)
        mock_env_constructor.return_value = mock_env_instance
        should_stop, validation_score = trainer.validate([dummy_file])

    assert should_stop is True  # Main check: early stopping triggered
    assert np.isclose(
        validation_score, expected_score
    )  # Check score matches calculation
    assert trainer.early_stopping_counter == trainer.early_stopping_patience
    assert trainer.best_validation_metric == 0.8 # Should not be updated
    mock_run_single_episode.assert_called_once_with(mock_env_instance)


@pytest.mark.unittest
@patch("src.trainer.TradingEnv")  # Patch constructor
@patch.object(DataManager, "get_random_training_file") # Patch file getter
def test_train_loop_handles_env_step_exception(
    mock_get_random_file, # Added mock
    mock_trading_env_init, # Patched constructor
    trainer, mock_agent, caplog, mock_data_manager # Added mock_data_manager
):
    """Test that the training loop properly handles exceptions during environment steps."""
    error_message = "Simulated environment error"
    num_episodes = 1

    # --- Configure Mocks --- #
    # Force train loop to use the same mock file path
    mock_file_path = Path("dummy/train.csv")
    mock_get_random_file.return_value = mock_file_path

    # Configure the instance that the mock constructor will return
    mock_instance = MagicMock(spec=TradingEnv)
    mock_instance.data_len = 100 # Sufficient length
    mock_instance.reset.return_value = (
        {"market_data": np.zeros((10, 5)), "account_state": np.zeros(ACCOUNT_STATE_DIM)},
        {"portfolio_value": 10000.0}
    )
    mock_instance.step.side_effect = Exception(error_message) # Raise exception
    mock_instance.action_space = MagicMock()
    mock_instance.action_space.sample.return_value = 0

    # Configure the patched TradingEnv constructor to return this instance
    mock_trading_env_init.return_value = mock_instance
    # --- End Mock Configuration --- #

    with caplog.at_level(logging.ERROR):
        trainer.train(
            env=MagicMock(spec=TradingEnv), # Initial env doesn't matter much here
            num_episodes=num_episodes,
            start_episode=0,
            start_total_steps=0,
            initial_best_score=-np.inf,
            initial_early_stopping_counter=0,
            specific_file=None
        )

    # Verify constructor called once with the forced path
    # Get the actual path returned by the (mocked) data manager
    expected_path = mock_data_manager.get_random_training_file()
    mock_trading_env_init.assert_called_once_with(
        data_path=str(expected_path), # Use path from mock data manager
        **trainer.env_config
    )
    # Verify the error was logged
    assert "Error during env.step" in caplog.text
    assert error_message in caplog.text
    # Check that the environment was closed even after the error
    mock_instance.close.assert_called_once()


@pytest.mark.unittest
@patch("src.trainer.TradingEnv") # Patch constructor
@patch.object(DataManager, "get_random_training_file") # Patch file getter
def test_train_loop_handles_agent_learn_exception(
    mock_get_random_file, # Added mock
    mock_trading_env_init, # Patched constructor
    trainer, mock_agent, caplog, mock_data_manager # Added mock_data_manager
):
    """Test that the training loop properly handles exceptions during agent learning."""
    error_message = "Simulated agent learning error"
    num_episodes = 1

    # --- Configure Mocks --- #
    # Force train loop to use the same mock file path
    mock_file_path = Path("dummy/train.csv")
    mock_get_random_file.return_value = mock_file_path

    # Configure the instance that the mock constructor will return
    mock_instance = MagicMock(spec=TradingEnv)
    # Ensure enough steps for learn exception
    mock_instance.data_len = trainer.warmup_steps + trainer.update_freq + 10
    mock_instance.reset.return_value = (
        {"market_data": np.zeros((10, 5)), "account_state": np.zeros(ACCOUNT_STATE_DIM)},
        {"portfolio_value": 10000.0}
    )
    step_return = (
        {"market_data": np.zeros((10, 5)), "account_state": np.zeros(ACCOUNT_STATE_DIM)}, # next_obs
        0.1, False, False, {"portfolio_value": 10000.0, "transaction_cost": 0.0}
    )
    # Configure step side effect to allow enough steps for learn
    mock_instance.step.side_effect = [step_return] * (mock_instance.data_len + 1)
    mock_instance.action_space = MagicMock()
    mock_instance.action_space.sample.return_value = 0
    mock_trading_env_init.return_value = mock_instance
    # --- End Mock Configuration --- #

    # Configure Agent mocks
    mock_agent.learn.side_effect = Exception(error_message)
    mock_agent.select_action.return_value = 1
    mock_agent.buffer = MagicMock()
    mock_agent.buffer.__len__.return_value = trainer.agent.batch_size

    with caplog.at_level(logging.ERROR):
        trainer.train(
            env=MagicMock(spec=TradingEnv), # Initial env doesn't matter much
            num_episodes=num_episodes,
            start_episode=0,
            start_total_steps=0,
            initial_best_score=-np.inf,
            initial_early_stopping_counter=0,
            specific_file=None
        )

    # Verify constructor called once with the forced path
    # Get the actual path returned by the (mocked) data manager
    expected_path = mock_data_manager.get_random_training_file()
    mock_trading_env_init.assert_called_once_with(
        data_path=str(expected_path), # Use path from mock data manager
        **trainer.env_config
    )
    # Verify the error was logged
    assert "EXCEPTION during learning update" in caplog.text
    assert error_message in caplog.text
    # Check that the environment was closed even after the error
    mock_instance.close.assert_called_once()
    # Ensure learn was attempted
    mock_agent.learn.assert_called()


# Helper function to create a dummy CSV file for testing
def create_dummy_csv(filepath: Path, rows: int = 20):
    # Simple header matching expected env columns
    header = "timestamp,open,high,low,close,volume\n"
    rows_data = [
        f"2023-01-01T00:{i:02d}:00Z,100,101,99,100,1000\n"
        for i in range(rows)
    ]
    csv_content = header + "".join(rows_data)
    filepath.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
    filepath.write_text(csv_content)

