import pytest
import torch
import numpy as np
import os
import shutil
from unittest.mock import MagicMock
from pathlib import Path

# Use absolute imports from src
from src.trainer import RainbowTrainerModule
from src.agent import RainbowDQNAgent
from src.env import TradingEnv
from src.data import DataManager
from src.constants import ACCOUNT_STATE_DIM


# Constants for tests
WINDOW_SIZE = 10
INITIAL_BALANCE = 10000
TRANSACTION_FEE = 0.001


# Helper function to create dummy CSV
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


# Fixtures
@pytest.fixture(scope="session") # Use session scope for factory-based fixture
def tmp_path_factory_session(tmp_path_factory):
    return tmp_path_factory

@pytest.fixture(scope="function") # Function scope ensures clean state for each test
def tmp_path(tmp_path_factory_session):
    # Use numbered=True to avoid potential conflicts if multiple tests run concurrently
    path = tmp_path_factory_session.mktemp("data", numbered=True)
    yield path
    # Optional: Clean up the directory after the test function finishes
    # shutil.rmtree(path)


@pytest.fixture
def default_config(tmp_path):
    # Minimal config for testing trainer logic
    # Paths are relative to the test execution directory
    return {
        "agent": {
            "network": "rainbow_transformer",
            "window_size": WINDOW_SIZE,
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
            "initial_balance": INITIAL_BALANCE,
            "reward_cost_scale": 0.5,
            "reward_pnl_scale": 1.0,
            "transaction_fee": TRANSACTION_FEE,
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
    agent.config = default_config["agent"]
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
    agent.select_action = MagicMock(return_value=0) # Default action, make mock
    agent.learn = MagicMock(return_value=0.1) # Default loss, make mock
    agent.save_model = MagicMock()
    agent.load_model = MagicMock()
    agent.total_steps = 0 # Initialize step count
    agent.training_mode = True
    agent.window_size = default_config["agent"]["window_size"]
    agent.n_features = default_config["agent"]["n_features"]
    agent.num_actions = default_config["agent"]["num_actions"]
    # Mock methods called by trainer/tests
    agent.act = MagicMock(return_value=agent.select_action.return_value)
    agent.set_training_mode = MagicMock()
    return agent


@pytest.fixture
def mock_data_manager(tmp_path):
    # Create dummy files
    processed_dir = tmp_path / "processed"
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "validation"
    test_dir = processed_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_files = [train_dir / "train_data_1.csv", train_dir / "train_data_2.csv"]
    val_files = [val_dir / "val_data_1.csv"]
    test_files = [test_dir / "test_data_1.csv"]

    for f in train_files + val_files + test_files:
        create_dummy_csv(f, rows=WINDOW_SIZE + 20) # Ensure enough rows

    # Mock DataManager methods
    manager = MagicMock(spec=DataManager)
    manager.base_dir = tmp_path # Base directory for the manager
    manager.processed_dir = processed_dir # Point to the created processed dir
    manager.train_dir = train_dir
    manager.validation_dir = val_dir
    manager.test_dir = test_dir
    # Mock file retrieval methods
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
    env.action_space.sample.return_value = 0 # Return int
    env.action_space.contains = MagicMock(return_value=True)
    env.close = MagicMock()
    env.data_len = WINDOW_SIZE + 10 # Default length
    return env


@pytest.fixture
def trainer(mock_agent, mock_data_manager, default_config, tmp_path):
    device = torch.device("cpu")
    # Ensure model_dir in config uses the test's tmp_path
    default_config["run"]["model_dir"] = str(tmp_path / "models")
    # Ensure DataManager uses the tmp_path structure correctly
    # Note: DataManager is mocked, but trainer might access config paths directly
    trainer_instance = RainbowTrainerModule(
        agent=mock_agent,
        device=device,
        data_manager=mock_data_manager, # Pass the mocked manager
        config=default_config,
    )
    return trainer_instance
