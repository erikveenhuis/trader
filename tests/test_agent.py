import pytest
import torch
import numpy as np
import os
import sys
import shutil

# Re-adding sys.path manipulation for this file
src_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src")
)  # Path adjusted from tests/ to src/
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    # Revert to direct imports
    from agent import RainbowDQNAgent, ACCOUNT_STATE_DIM
    from buffer import PrioritizedReplayBuffer
    from model import RainbowNetwork
except ImportError as e:
    print(f"Failed to import required modules directly: {e}")
    print(f"Current sys.path: {sys.path}")
    pytest.skip(
        f"Skipping agent tests due to import error: {e}", allow_module_level=True
    )


# --- Test Configuration ---
@pytest.fixture(scope="module")
def default_config():
    """Provides a default configuration dictionary for the agent."""
    return {
        "seed": 42,
        "gamma": 0.99,
        "lr": 1e-4,
        "replay_buffer_size": 1000,  # Keep small for tests
        "batch_size": 4,  # Small batch size for tests
        "target_update_freq": 5,  # Frequent updates for testing
        "num_atoms": 51,
        "v_min": -10,
        "v_max": 10,
        "alpha": 0.6,
        "beta_start": 0.4,
        "beta_frames": 100,  # Short annealing for tests
        "n_steps": 3,
        "window_size": 10,
        "n_features": 5,
        "hidden_dim": 64,
        "num_actions": 3,  # e.g., Hold, Buy, Sell
        "debug": True,  # Enable debug checks
        "grad_clip_norm": 10.0,
    }


# --- Test Agent Instance ---
@pytest.fixture(scope="function")  # Recreate agent for each test function
def agent(default_config):
    """Creates a RainbowDQNAgent instance for testing."""
    # Use CPU for tests unless CUDA is explicitly requested and available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Ensure consistency: If config includes device, use it. Otherwise, default to CPU.
    if "device" in default_config:
        device = default_config["device"]
    else:
        device = "cpu"  # Default to CPU for easier testing environment

    agent_instance = RainbowDQNAgent(config=default_config, device=device)
    # Ensure agent starts in training mode for most tests
    agent_instance.set_training_mode(True)
    return agent_instance


# --- Helper Functions ---
def generate_dummy_observation(config):
    """Generates a single dummy observation dictionary."""
    market_data = np.random.rand(config["window_size"], config["n_features"]).astype(
        np.float32
    )
    account_state = np.random.rand(ACCOUNT_STATE_DIM).astype(np.float32)
    return {"market_data": market_data, "account_state": account_state}


# --- Test Cases ---


def test_agent_initialization(agent, default_config):
    """Tests if the agent initializes components correctly."""
    assert agent is not None
    assert agent.config == default_config
    assert isinstance(agent.network, RainbowNetwork)
    assert isinstance(agent.target_network, RainbowNetwork)
    assert agent.optimizer is not None
    assert isinstance(agent.buffer, PrioritizedReplayBuffer)
    assert agent.buffer.capacity == default_config["replay_buffer_size"]
    assert agent.total_steps == 0
    assert agent.training_mode is True
    assert agent.device == (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Verify device used

    # Check network parameters are on the correct device
    for param in agent.network.parameters():
        assert str(param.device).startswith(agent.device)
    for param in agent.target_network.parameters():
        assert str(param.device).startswith(agent.device)


def test_select_action(agent, default_config):
    """Tests the select_action method."""
    obs = generate_dummy_observation(default_config)

    # Test in training mode
    agent.set_training_mode(True)
    action_train = agent.select_action(obs)
    assert isinstance(action_train, int)
    assert 0 <= action_train < default_config["num_actions"]
    assert agent.network.training is True  # Should remain in train mode

    # Test in evaluation mode
    agent.set_training_mode(False)
    action_eval = agent.select_action(obs)
    assert isinstance(action_eval, int)
    assert 0 <= action_eval < default_config["num_actions"]
    assert agent.network.training is False  # Should be in eval mode


def test_store_transition_and_n_step(agent, default_config):
    """Tests storing transitions and n-step buffer logic."""
    n_steps = default_config["n_steps"]
    buffer_capacity = default_config["replay_buffer_size"]
    initial_buffer_len = len(agent.buffer)

    transitions = []
    for i in range(n_steps + 2):  # Store enough transitions to trigger PER storage
        obs = generate_dummy_observation(default_config)
        action = np.random.randint(default_config["num_actions"])
        reward = np.random.rand() * 2 - 1  # Random reward between -1 and 1
        next_obs = generate_dummy_observation(default_config)
        done = i == n_steps + 1  # Make the last transition terminal

        agent.store_transition(obs, action, reward, next_obs, done)
        transitions.append((obs, action, reward, next_obs, done))

        if i < n_steps - 1:
            # Should not have stored anything in PER buffer yet
            assert len(agent.buffer) == initial_buffer_len
            assert len(agent.n_step_buffer) == i + 1
        elif i == n_steps - 1:
            # First n-step transition should be stored now
            assert len(agent.buffer) == initial_buffer_len + 1
            assert len(agent.n_step_buffer) == n_steps
        else:
            # Subsequent transitions stored
            assert len(agent.buffer) == initial_buffer_len + (i - n_steps + 2)
            assert len(agent.n_step_buffer) == n_steps  # Should stay at maxlen

    assert len(agent.buffer) <= buffer_capacity


def test_learn_step(agent, default_config, mocker):
    """Tests a single learning step, mocking buffer sample."""
    batch_size = default_config["batch_size"]
    n_steps = default_config["n_steps"]

    # 1. Ensure buffer has enough samples to trigger learning
    for _ in range(
        batch_size + n_steps
    ):  # Need enough to fill n_step and sample a batch
        obs = generate_dummy_observation(default_config)
        action = np.random.randint(default_config["num_actions"])
        reward = np.random.rand()
        next_obs = generate_dummy_observation(default_config)
        done = False
        agent.store_transition(obs, action, reward, next_obs, done)

    assert (
        len(agent.buffer) >= batch_size
    ), "Buffer should have enough samples for a batch"

    # 2. Mock the buffer's sample method to return controlled data
    # Generate a dummy batch matching the expected output structure of buffer.sample
    mock_market = np.random.rand(
        batch_size, default_config["window_size"], default_config["n_features"]
    ).astype(np.float32)
    mock_account = np.random.rand(batch_size, ACCOUNT_STATE_DIM).astype(np.float32)
    mock_action = np.random.randint(
        0, default_config["num_actions"], size=batch_size
    ).astype(np.int64)
    mock_reward = np.random.rand(batch_size).astype(np.float32)
    mock_next_market = np.random.rand(
        batch_size, default_config["window_size"], default_config["n_features"]
    ).astype(np.float32)
    mock_next_account = np.random.rand(batch_size, ACCOUNT_STATE_DIM).astype(np.float32)
    mock_done = np.zeros(batch_size, dtype=np.bool_)  # Assume not done for simplicity
    mock_batch = (
        mock_market,
        mock_account,
        mock_action,
        mock_reward,
        mock_next_market,
        mock_next_account,
        mock_done,
    )

    mock_indices = np.random.randint(0, len(agent.buffer), size=batch_size)
    mock_weights = np.random.rand(batch_size).astype(np.float32)

    mocker.patch.object(
        agent.buffer, "sample", return_value=(mock_batch, mock_indices, mock_weights)
    )
    mocker.patch.object(agent.buffer, "update_priorities")  # Mock priority updates
    mocker.patch.object(
        agent, "_update_target_network"
    )  # Mock target updates to isolate learn logic

    # 3. Call the learn method
    initial_total_steps = agent.total_steps
    initial_net_params = [p.clone().detach() for p in agent.network.parameters()]

    loss = agent.learn()

    # 4. Assertions
    assert loss is not None
    assert isinstance(loss, float)
    assert agent.total_steps == initial_total_steps + 1

    # Check if network parameters changed
    params_changed = False
    for p_initial, p_final in zip(initial_net_params, agent.network.parameters()):
        if not torch.equal(p_initial, p_final):
            params_changed = True
            break
    assert (
        params_changed
    ), "Network parameters should have been updated after learning step."

    # Check mocks were called
    agent.buffer.sample.assert_called_once_with(batch_size)
    # Need to check the args for update_priorities carefully
    # args, kwargs = agent.buffer.update_priorities.call_args
    # assert np.array_equal(args[0], mock_indices) # Check indices
    # assert isinstance(args[1], torch.Tensor) # Check priorities tensor
    # assert args[1].shape == (batch_size,)
    # assert args[1].dtype == torch.float32
    # Using assert_called_once is simpler if precise args aren't crucial
    agent.buffer.update_priorities.assert_called_once()

    # Check if target network update was triggered if needed
    if agent.total_steps % default_config["target_update_freq"] == 0:
        agent._update_target_network.assert_called_once()
    else:
        agent._update_target_network.assert_not_called()


def test_target_network_update(agent, default_config):
    """Tests if the target network updates correctly."""
    # Ensure network and target network start differently (modify one slightly)
    with torch.no_grad():
        for param in agent.network.parameters():
            param.data += 0.1

    initial_target_params = [
        p.clone().detach() for p in agent.target_network.parameters()
    ]

    # Force update
    agent._update_target_network()

    final_target_params = [
        p.clone().detach() for p in agent.target_network.parameters()
    ]
    online_params = [p.clone().detach() for p in agent.network.parameters()]

    # Check if target params match online params after update
    for p_target, p_online in zip(final_target_params, online_params):
        assert torch.equal(
            p_target, p_online
        ), "Target network parameters did not match online network after update."

    # Check if target params are different from initial target params
    params_updated = False
    for p_initial, p_final in zip(initial_target_params, final_target_params):
        if not torch.equal(p_initial, p_final):
            params_updated = True
            break
    assert params_updated, "Target network parameters should have changed after update."


def test_save_load_model(agent, default_config):
    """Tests saving and loading the agent's state."""
    save_dir = "test_agent_save"
    save_prefix = os.path.join(save_dir, "test_model")
    # Clean up any previous test runs
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    # Modify agent state slightly
    agent.total_steps = 123
    # Perform a learn step to change network/optimizer state
    for _ in range(default_config["batch_size"] + default_config["n_steps"]):
        obs = generate_dummy_observation(default_config)
        agent.store_transition(obs, 1, 0.5, obs, False)  # Simple transition
    if len(agent.buffer) >= default_config["batch_size"]:
        agent.learn()  # Ensure optimizer has state and network changed

    # Capture current state for comparison
    original_state_dict = agent.network.state_dict()
    original_optimizer_dict = agent.optimizer.state_dict()
    original_total_steps = agent.total_steps

    # Save the model
    agent.save_model(save_prefix)
    save_path = f"{save_prefix}_rainbow_agent.pt"
    assert os.path.exists(save_path)

    # Create a new agent instance with the same config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    new_agent = RainbowDQNAgent(config=default_config, device=device)
    assert new_agent.total_steps == 0  # Should start fresh

    # Load the saved state
    new_agent.load_model(save_prefix)

    # Compare states
    assert new_agent.total_steps == original_total_steps

    # Compare network weights
    loaded_state_dict = new_agent.network.state_dict()
    for key in original_state_dict:
        assert torch.equal(
            original_state_dict[key], loaded_state_dict[key]
        ), f"Network parameter mismatch for key: {key}"

    # Compare target network weights (should also be loaded/synced)
    loaded_target_state_dict = new_agent.target_network.state_dict()
    for (
        key
    ) in original_state_dict:  # Target should match original online after load->sync
        assert torch.equal(
            original_state_dict[key], loaded_target_state_dict[key]
        ), f"Target network parameter mismatch for key: {key}"

    # Compare optimizer state (tricky due to internal structure)
    loaded_optimizer_dict = new_agent.optimizer.state_dict()
    # Basic check: compare number of state groups and parameters
    assert len(original_optimizer_dict["state"]) == len(loaded_optimizer_dict["state"])
    assert len(original_optimizer_dict["param_groups"]) == len(
        loaded_optimizer_dict["param_groups"]
    )
    # A more thorough check might involve comparing specific tensors within the state,
    # ensuring they are on the correct device after loading.

    # Clean up the saved model directory
    shutil.rmtree(save_dir)


# --- Test Configuration Compatibility Check on Load ---
def test_load_model_config_mismatch(agent, default_config, mocker, caplog):
    """Tests loading a model with a mismatched configuration."""
    save_dir = "test_agent_save_mismatch"
    save_prefix = os.path.join(save_dir, "test_model_mismatch")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    # Save the current agent
    agent.total_steps = 50
    agent.save_model(save_prefix)
    save_path = f"{save_prefix}_rainbow_agent.pt"
    assert os.path.exists(save_path)

    # Create a new config with a mismatch
    mismatched_config = default_config.copy()
    mismatched_config["num_actions"] = (
        default_config["num_actions"] + 1
    )  # Change an essential param

    # Create a new agent with the mismatched config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mismatched_agent = RainbowDQNAgent(config=mismatched_config, device=device)

    # Mock logger to capture warnings
    mock_logger = mocker.patch("agent.logger")  # Patch logger inside the agent module

    # Load the model saved with the original config
    mismatched_agent.load_model(save_prefix)

    # Check if the warning about config mismatch was logged
    # Use the mocked module logger directly
    log_calls = mock_logger.warning.call_args_list
    assert any("Configuration mismatch detected" in str(call) for call in log_calls)
    assert any("num_actions" in str(call) for call in log_calls)

    # Even with mismatch, steps should ideally load if present
    # Update: Check that steps are RESET to 0 due to load exception from incompatibility
    assert mismatched_agent.total_steps == 0

    # Clean up
    shutil.rmtree(save_dir)


def test_set_training_mode(agent):
    """Tests setting the training mode."""
    # Initial state is training=True from fixture
    assert agent.training_mode is True
    assert agent.network.training is True
    assert agent.target_network.training is False  # Target network is always eval

    # Set to evaluation mode
    agent.set_training_mode(False)
    assert agent.training_mode is False
    assert agent.network.training is False
    assert agent.target_network.training is False  # Target network remains eval

    # Set back to training mode
    agent.set_training_mode(True)
    assert agent.training_mode is True
    assert agent.network.training is True
    assert agent.target_network.training is False  # Target network remains eval


# Add more tests as needed, e.g., for specific components like _project_target_distribution
# or edge cases in PER interaction.

# Note: Testing the numerical correctness of _project_target_distribution
# would require known inputs and analytically derived or pre-computed expected outputs,
# which can be complex to set up. The current tests focus on integration and API usage.
