import pytest
import torch
import numpy as np
import os
import logging # Added for handler setting
from pathlib import Path
from unittest.mock import MagicMock, patch, call, ANY, PropertyMock
import src.trainer # Import the module containing the class/function

try:
    # Use direct imports now
    from trainer import RainbowTrainerModule
    from agent import RainbowDQNAgent, ACCOUNT_STATE_DIM
    from env import TradingEnv
    from data import DataManager
    from metrics import PerformanceTracker, calculate_composite_score
except ImportError as e:
    print(f"Failed to import required modules from src package: {e}")
    print(f"Current sys.path: {sys.path}")
    pytest.skip(f"Skipping trainer tests due to import error: {e}", allow_module_level=True)

# --- Test Configuration ---
@pytest.fixture(scope="module")
def default_config():
    """Provides a default configuration dictionary for trainer tests."""
    return {
        'agent': {
            'seed': 42,
            'gamma': 0.99,
            'lr': 1e-4,
            'replay_buffer_size': 500, # Small for tests
            'batch_size': 4,
            'target_update_freq': 5,
            'num_atoms': 51,
            'v_min': -10,
            'v_max': 10,
            'alpha': 0.6,
            'beta_start': 0.4,
            'beta_frames': 100,
            'n_steps': 3,
            'window_size': 10,
            'n_features': 5,
            'hidden_dim': 32,
            'num_actions': 3,
            'debug': False,
            'grad_clip_norm': 10.0
        },
        'environment': {
            'initial_balance': 10000,
            'transaction_fee': 0.001,
            # Add other env-specific keys if needed by TradingEnv
        },
        'trainer': {
            'early_stopping_patience': 3, # Short for tests
            'min_validation_threshold': 0.0,
            'validation_freq': 2, # Frequent for tests
            'checkpoint_save_freq': 2, # Frequent for tests
            'reward_window': 10,
            'update_freq': 1, # Update every step for easier testing
            'log_freq': 5, # Log frequently
            'warmup_steps': 10 # Short warmup
        },
        'run': {
            'model_dir': 'test_models' # Use a test-specific dir
        }
    }

# --- Mock Fixtures ---
@pytest.fixture
def mock_agent(default_config):
    agent = MagicMock(spec=RainbowDQNAgent)
    agent.config = default_config['agent']
    agent.device = torch.device('cpu')
    agent.gamma = default_config['agent']['gamma']
    agent.lr = default_config['agent']['lr']
    agent.batch_size = default_config['agent']['batch_size']
    agent.target_update_freq = default_config['agent']['target_update_freq']
    agent.num_atoms = default_config['agent']['num_atoms']
    agent.v_min = default_config['agent']['v_min']
    agent.v_max = default_config['agent']['v_max']
    agent.n_steps = default_config['agent']['n_steps']
    agent.num_actions = default_config['agent']['num_actions']
    agent.window_size = default_config['agent']['window_size']
    agent.n_features = default_config['agent']['n_features']
    agent.total_steps = 0 # Initial state
    agent.training_mode = True
    # Mock buffer with a length property
    agent.buffer = MagicMock()
    agent.buffer.__len__.return_value = default_config['agent']['replay_buffer_size'] # Assume full buffer
    agent.buffer.alpha = default_config['agent']['alpha']
    agent.buffer.beta_start = default_config['agent']['beta_start']
    # Mock learn method to return a dummy loss
    agent.learn.return_value = 0.1
    # Mock select_action to return a valid action
    agent.select_action.return_value = 1 # e.g., Buy
    # Mock optimizer and network presence
    agent.optimizer = MagicMock()
    agent.network = MagicMock()
    agent.target_network = MagicMock()
    return agent

@pytest.fixture
def mock_data_manager(tmp_path):
    dm = MagicMock(spec=DataManager)
    dm.base_dir = tmp_path / "data" # Point to temp dir
    train_dir = dm.base_dir / "train"
    val_dir = dm.base_dir / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Create minimal valid dummy CSV files
    header = "timestamp,open,high,low,close,volume,feature1,feature2,feature3,feature4,feature5\n"
    # Need enough rows for window_size + some steps
    rows = [f"2023-01-01T00:{i:02d}:00Z,100,101,99,100,1000,0.1,0.2,0.3,0.4,0.5\n" for i in range(15)]
    csv_content = header + "".join(rows)
    
    train_file_1 = train_dir / "train_data_1.csv"
    train_file_2 = train_dir / "train_data_2.csv"
    val_file_1 = val_dir / "val_data_1.csv"
    
    train_file_1.write_text(csv_content)
    train_file_2.write_text(csv_content)
    val_file_1.write_text(csv_content)

    dm.get_random_training_file.return_value = train_file_1
    dm.get_validation_files.return_value = [val_file_1]
    return dm

@pytest.fixture
def mock_env(default_config):
    env = MagicMock(spec=TradingEnv)
    env.action_space = MagicMock()
    env.action_space.sample.return_value = 0 # e.g., Hold
    env.action_space.contains.return_value = True
    # Define obs structure
    obs = {
        'market_data': np.random.rand(default_config['agent']['window_size'], default_config['agent']['n_features']).astype(np.float32),
        'account_state': np.random.rand(ACCOUNT_STATE_DIM).astype(np.float32)
    }
    # Ensure info values are floats
    info = {'portfolio_value': 10000.0, 'price': 100.0, 'transaction_cost': 0.0}
    env.reset.return_value = (obs, info)
    # Make step return slightly different obs/info to simulate progression
    next_obs = {
        'market_data': np.random.rand(default_config['agent']['window_size'], default_config['agent']['n_features']).astype(np.float32),
        'account_state': np.random.rand(ACCOUNT_STATE_DIM).astype(np.float32)
    }
    # Ensure next_info values are floats
    next_info = {'portfolio_value': 10050.0, 'price': 101.0, 'transaction_cost': 1.0}
    # Configure step to run a few times then be done
    env.step.side_effect = [
        (next_obs, 1.0, False, False, next_info), # Use float rewards
        (next_obs, 0.5, False, False, next_info),
        (next_obs, -0.2, True, False, next_info), # Done=True on 3rd step
    ] * 100 # Repeat pattern enough times for tests
    env.initial_balance = float(default_config['environment']['initial_balance'])
    return env

@pytest.fixture
def trainer(mock_agent, mock_data_manager, default_config, tmp_path):
    # Use tmp_path for model_dir
    config = default_config.copy()
    config['run']['model_dir'] = str(tmp_path / "models")
    
    # Mock log handlers
    train_handler = MagicMock()
    train_handler.level = 0 # Set a default level
    val_handler = MagicMock()
    val_handler.level = 0 # Set a default level
    
    # Standard trainer instantiation (no patching here)
    trainer_instance = RainbowTrainerModule(
        agent=mock_agent, 
        device=torch.device('cpu'),
        data_manager=mock_data_manager,
        config=config,
        # train_log_handler=train_handler, # Removed
        # validation_log_handler=val_handler # Removed
    )
    
    # Ensure model directory exists
    os.makedirs(trainer_instance.model_dir, exist_ok=True)
    return trainer_instance

# --- Test Cases ---

def test_trainer_init(trainer, default_config, tmp_path):
    assert trainer.agent is not None
    assert trainer.device == torch.device('cpu')
    assert trainer.data_manager is not None
    assert trainer.config == default_config
    assert trainer.best_validation_metric == -np.inf
    assert trainer.early_stopping_counter == 0
    assert trainer.model_dir == str(tmp_path / "models")
    assert os.path.exists(trainer.model_dir)
    assert trainer.best_model_path_prefix == str(tmp_path / "models" / "rainbow_transformer_best")
    assert trainer.latest_trainer_checkpoint_path == str(tmp_path / "models" / "checkpoint_trainer_latest.pt")
    assert trainer.best_trainer_checkpoint_path == str(tmp_path / "models" / "checkpoint_trainer_best.pt")

@patch('torch.save')
def test_save_trainer_checkpoint(mock_torch_save, trainer):
    episode = 10
    total_steps = 1000
    trainer.best_validation_metric = 0.5
    trainer.early_stopping_counter = 1

    # Test saving latest
    trainer._save_trainer_checkpoint(episode, total_steps, is_best=False)
    expected_checkpoint = {
        'episode': episode,
        'total_train_steps': total_steps,
        'best_validation_metric': 0.5,
        'early_stopping_counter': 1,
    }
    calls = [call(expected_checkpoint, trainer.latest_trainer_checkpoint_path)]
    mock_torch_save.assert_has_calls(calls)
    mock_torch_save.reset_mock()

    # Test saving best
    trainer.best_validation_metric = 0.8 # Update best score
    trainer._save_trainer_checkpoint(episode, total_steps, is_best=True)
    expected_best_checkpoint = {
        'episode': episode,
        'total_train_steps': total_steps,
        'best_validation_metric': 0.8,
        'early_stopping_counter': 1, # Counter doesn't reset here
    }
    calls = [
        call(expected_best_checkpoint, trainer.latest_trainer_checkpoint_path),
        call(expected_best_checkpoint, trainer.best_trainer_checkpoint_path)
    ]
    mock_torch_save.assert_has_calls(calls, any_order=False)

def test_evaluate_for_validation(trainer, mock_env, mock_agent, default_config):
    # Renamed argument to config to avoid shadowing fixture
    config = default_config
    # Reset side effect for consistent evaluation run
    # obs, info = mock_env.reset() # Removed initial call, reset happens inside trainer method
    # Create a dummy next_obs for the mock side_effect
    dummy_market = np.random.rand(config['agent']['window_size'], config['agent']['n_features']).astype(np.float32)
    dummy_account = np.random.rand(ACCOUNT_STATE_DIM).astype(np.float32)
    next_obs = {'market_data': dummy_market, 'account_state': dummy_account}
    # Ensure float for portfolio_value in this test's specific mock setup
    next_info = {'portfolio_value': 10100.0, 'price': 102.0, 'transaction_cost': 1.0}
    mock_env.step.side_effect = [
        (next_obs, 10.0, False, False, next_info),
        (next_obs, 5.0, True, False, next_info), # Done
    ]
    mock_agent.select_action.return_value = 0 # Hold

    total_reward, metrics = trainer.evaluate_for_validation(mock_env)

    assert isinstance(total_reward, float)
    assert total_reward == 15.0
    assert isinstance(metrics, dict)
    assert 'sharpe_ratio' in metrics
    assert 'portfolio_value' in metrics
    assert metrics['portfolio_value'] == 10100 # Final portfolio value from last step
    mock_agent.set_training_mode.assert_has_calls([call(False), call(True)]) # Eval and back to Train
    assert mock_env.reset.call_count == 1 # Reset is called once by evaluate_for_validation
    assert mock_env.step.call_count == 2

@patch('trainer.calculate_composite_score', return_value=0.75)
@patch('json.dump')
def test_validate(mock_json_dump, mock_calc_score, trainer, mock_data_manager, mock_env):
    val_files = mock_data_manager.get_validation_files()
    trainer.best_validation_metric = 0.5 # Initial best score
    trainer.early_stopping_counter = 0

    # Mock evaluate_for_validation to return consistent results
    mock_metrics = {
        'avg_reward': 10.0, 'portfolio_value': 11000, 'total_return': 10.0,
        'sharpe_ratio': 1.2, 'max_drawdown': 0.05, 'win_rate': 0.6,
        'avg_action': 0.5, 'transaction_costs': 50.0
    }
    with patch.object(trainer, 'evaluate_for_validation', return_value=(10.0, mock_metrics)) as mock_eval:
        should_stop, validation_score = trainer.validate(val_files)

    assert should_stop is False
    assert validation_score == 0.75 # From mock_calc_score
    assert trainer.best_validation_metric == 0.75 # Updated best score
    assert trainer.early_stopping_counter == 0 # Reset counter
    mock_eval.assert_called_once() # Called once for the single validation file
    mock_calc_score.assert_called_once() # Called once with avg metrics
    mock_json_dump.assert_called_once() # Results should be saved

# Patch evaluate_for_validation using decorator
@patch.object(RainbowTrainerModule, 'evaluate_for_validation', return_value=(-5.0, {}))
def test_validate_early_stopping(mock_evaluate_for_validation, trainer):
    trainer.best_validation_metric = 0.8 # Set a high best score
    trainer.early_stopping_counter = trainer.early_stopping_patience - 1 # One step away

    # Define the zeroed metrics evaluate_for_validation should return
    mock_returned_metrics = {
        'avg_reward': 0.0, 'portfolio_value': 0.0, 'total_return': 0.0,
        'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0,
        'avg_action': 0.0, 'transaction_costs': 0.0
    }
    mock_evaluate_for_validation.return_value = (-5.0, mock_returned_metrics)
    
    window_size = trainer.agent_config.get('window_size', 10)
    
    # DO NOT patch calculate_composite_score - let the real one run
    dummy_file = trainer.data_manager.base_dir / "dummy_val.csv"
    header = "timestamp,open,high,low,close,volume\n"
    rows = [f"2023-01-01T00:{i:02d}:00Z,100,101,99,100,1000\n" for i in range(window_size + 5)]
    dummy_file.write_text(header + "".join(rows))
    
    should_stop, validation_score = trainer.validate([dummy_file])

    # Calculate the expected score based on the real function and zeroed metrics
    expected_score = calculate_composite_score(mock_returned_metrics) # Should be 0.2

    assert should_stop is True # Main check: early stopping triggered
    assert np.isclose(validation_score, expected_score) # Check score matches calculation
    assert trainer.best_validation_metric == 0.8 # Best score didn't change
    assert trainer.early_stopping_counter == trainer.early_stopping_patience # Counter incremented
    mock_evaluate_for_validation.assert_called_once()
    # No assertion for calculate_composite_score mock as it wasn't mocked

# Test train method (simplified)
@patch('trainer.TradingEnv') # Mock env instantiation within train
@patch.object(RainbowTrainerModule, 'validate')
@patch.object(RainbowTrainerModule, '_save_trainer_checkpoint')
def test_train_loop_simple(mock_save_checkpoint, mock_validate, mock_trading_env_init, trainer, mock_agent, mock_env, mock_data_manager, default_config):
    num_episodes = 5
    steps_per_episode = 3 # Based on mock_env.step side_effect
    warmup_steps = default_config['trainer']['warmup_steps']
    update_freq = default_config['trainer']['update_freq']
    validation_freq = default_config['trainer']['validation_freq']
    checkpoint_freq = default_config['trainer']['checkpoint_save_freq']

    # Configure mock env instance returned by the mocked init
    mock_trading_env_init.return_value = mock_env
    # Configure validate to return (False, some_score)
    mock_validate.return_value = (False, 0.5)
    # Configure agent buffer length
    mock_agent.buffer.__len__.return_value = mock_agent.batch_size # Ensure learning happens immediately after warmup
    # Remove mock for should_validate as it's no longer complex
    # trainer.should_validate = MagicMock(side_effect=lambda ep, perf: (ep + 1) % validation_freq == 0)

    trainer.train(env=mock_env, # Use the mock_env fixture here
                  num_episodes=num_episodes,
                  start_episode=0, start_total_steps=0,
                  initial_best_score=-np.inf, initial_early_stopping_counter=0,
                  specific_file=None)

    total_expected_steps = num_episodes * steps_per_episode

    # Check agent calls
    assert mock_agent.set_training_mode.call_args_list[0] == call(True)
    # Warmup actions
    assert mock_env.action_space.sample.call_count == min(warmup_steps, total_expected_steps)
    # Agent actions (total steps - warmup steps)
    expected_agent_actions = max(0, total_expected_steps - warmup_steps)
    assert mock_agent.select_action.call_count == expected_agent_actions
    # Store transition called every step
    assert mock_agent.store_transition.call_count == total_expected_steps
    # Learn called after warmup, based on update_freq
    expected_learn_calls = 0
    current_steps = 0
    for ep in range(num_episodes):
        for st in range(steps_per_episode):
            current_steps += 1
            if current_steps > warmup_steps and current_steps % update_freq == 0:
                expected_learn_calls += 1
    assert mock_agent.learn.call_count == expected_learn_calls

    # Check validation calls (uses default frequency logic now)
    expected_validate_calls = sum(1 for i in range(num_episodes) if (i + 1) % validation_freq == 0)
    assert mock_validate.call_count == expected_validate_calls

    # Check checkpoint saving
    # Note: Checkpoint might be saved after validation OR periodically
    expected_checkpoint_saves = 0
    for i in range(num_episodes):
        ep_num = i + 1
        is_val_ep = ep_num % validation_freq == 0
        is_chkpt_ep = ep_num % checkpoint_freq == 0
        if is_val_ep: # Saved after validation
            expected_checkpoint_saves += 1
        elif is_chkpt_ep: # Saved periodically (if not a validation ep)
            expected_checkpoint_saves += 1
    # Final save after loop + initial saves
    assert mock_save_checkpoint.call_count >= expected_checkpoint_saves
    # Check final save call
    mock_save_checkpoint.assert_called_with(episode=num_episodes, total_steps=ANY, is_best=False)

    # Check final model save
    mock_agent.save_model.assert_called() # Should be called at least for final model

def test_evaluate(trainer, mock_env, mock_agent, default_config):
    # Renamed argument to config to avoid shadowing fixture
    config = default_config
    # Similar setup to evaluate_for_validation, but checks evaluate method
    # obs, info = mock_env.reset() # Removed initial call
    # Create a dummy next_obs for the mock side_effect
    dummy_market = np.random.rand(config['agent']['window_size'], config['agent']['n_features']).astype(np.float32)
    dummy_account = np.random.rand(ACCOUNT_STATE_DIM).astype(np.float32)
    next_obs = {'market_data': dummy_market, 'account_state': dummy_account}
    next_info = {'portfolio_value': 10100, 'price': 102.0, 'transaction_cost': 1.0}
    mock_env.step.side_effect = [
        (next_obs, 10.0, False, False, next_info),
        (next_obs, 5.0, True, False, next_info), # Done
    ]
    mock_agent.select_action.return_value = 1 # Buy

    total_reward, final_portfolio = trainer.evaluate(mock_env)

    assert isinstance(total_reward, float)
    assert total_reward == 15.0
    assert isinstance(final_portfolio, float)
    assert final_portfolio == 10100 # Final portfolio value from last step
    mock_agent.set_training_mode.assert_called_once_with(False) # Only called once for eval
    assert mock_env.reset.call_count == 1 # Reset is called once by evaluate()
    assert mock_env.step.call_count == 2

# Clean up test directory
@pytest.fixture(scope="session", autouse=True)
def cleanup_test_models():
    yield
    if os.path.exists('test_models'):
        shutil.rmtree('test_models')

# --- New Error Handling Tests ---

def test_train_loop_handles_env_step_exception(trainer, caplog):
    """Test trainer.train catches and logs exceptions from env.step."""
    # Create a simple mock env just for this test
    mock_env = MagicMock(spec=TradingEnv)
    
    # Configure reset to return valid initial state with correct shapes
    obs_shape = (trainer.agent.window_size, trainer.agent.n_features)
    mock_env.reset.return_value = (
        {'market_data': np.zeros(obs_shape), 'account_state': np.zeros(2)}, 
        {'portfolio_value': 10000.0} # Ensure valid initial portfolio
    )
    # Mock action space needed for warmup step
    mock_env.action_space = MagicMock()
    mock_env.action_space.sample.return_value = 0
    mock_env.action_space.contains.return_value = True

    # Configure step to raise error immediately
    mock_env.step.side_effect = RuntimeError("Simulated env.step crash!")
    
    caplog.set_level(logging.ERROR)
    
    # Run trainer for 1 episode. It should call reset, then step (which crashes).
    try:
        trainer.train(
            env=mock_env, # Pass the mock env
            num_episodes=1,
            start_episode=0,
            start_total_steps=0,
            initial_best_score=-np.inf,
            initial_early_stopping_counter=0
        )
    except RuntimeError as e:
        # Catch the exception if the trainer *doesn't* handle it
        if "Simulated env.step crash!" in str(e):
            pytest.fail(f"Trainer did not handle env.step exception: {e}")
        else:
            raise # Re-raise unexpected errors

    # Assertions
    mock_env.step.assert_called_once() # Ensure env.step was actually called
    assert "Error during env.step" in caplog.text
    assert "Simulated env.step crash!" in caplog.text # Check for specific error message

def test_train_loop_handles_agent_learn_exception(trainer, mock_agent, caplog):
    """Test trainer.train catches and logs exceptions from agent.learn."""
    # Create a simple mock env for this test
    mock_env = MagicMock(spec=TradingEnv)
    agent = mock_agent # Use the fixture agent

    # Configure env.step to return valid data just once (enough to get past warmup)
    obs_shape = (agent.window_size, agent.n_features) # Use agent config
    env_step_return = (
        {'market_data': np.zeros(obs_shape), 'account_state': np.zeros(2)}, # obs
        0.1, # reward 
        False, # done
        False, # truncated
        {'portfolio_value': 10100.0, 'transaction_cost': 1.0} # info
    )
    # Need enough steps returned to satisfy the loop until learn is called
    # Let's assume warmup=10, update_freq=4 => learn called at step 14
    # We need at least 14 successful steps mocked
    mock_env.step.side_effect = [env_step_return] * (trainer.warmup_steps + trainer.update_freq + 5)
    
    mock_env.reset.return_value = (
        {'market_data': np.zeros(obs_shape), 'account_state': np.zeros(2)}, 
        {'portfolio_value': 10000.0}
    )
    mock_env.action_space = MagicMock()
    mock_env.action_space.sample.return_value = 0
    mock_env.action_space.contains.return_value = True

    # Ensure buffer is full enough
    agent.buffer = MagicMock()
    agent.batch_size = 4 # Needs to match trainer config or be mocked
    agent.buffer.__len__.return_value = agent.batch_size
    # Mock select_action needed by trainer loop
    agent.select_action.return_value = 1 # Dummy action

    # Patch the agent.learn method to raise an error
    agent.learn.side_effect = ValueError("Simulated agent learn crash!")
    # with patch.object(agent, 'learn', side_effect=ValueError("Simulated agent learn crash!")) as mock_learn:
        
    caplog.set_level(logging.ERROR)

    # Run trainer long enough to trigger the learn call
    try:
        trainer.train(
            env=mock_env, # Pass the local mock env
            num_episodes=1, # One episode should be enough
            start_episode=0,
            start_total_steps=0,
            initial_best_score=-np.inf,
            initial_early_stopping_counter=0
        )
    except ValueError as e:
         if "Simulated agent learn crash!" in str(e):
             pytest.fail(f"Trainer did not handle agent.learn exception: {e}")
         else:
             raise

    # Assertions
    agent.learn.assert_called() # Ensure learn was actually called
    # mock_learn.assert_called() 
    assert "EXCEPTION during learning update" in caplog.text
    assert "Simulated agent learn crash!" in caplog.text

# --- End New Error Handling Tests --- 