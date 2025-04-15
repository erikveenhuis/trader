import pytest
import torch
import logging
import numpy as np
from unittest.mock import MagicMock, patch, call, ANY
from pathlib import Path

# Use absolute imports from src
from src.trainer import RainbowTrainerModule # Keep for patching
from src.env import TradingEnv
from src.constants import ACCOUNT_STATE_DIM
from src.metrics import calculate_episode_score # Needed for test_train_save_best_checkpoint_on_validation


# Note: Fixtures (trainer, mock_agent, mock_data_manager, etc.) are in conftest.py


@pytest.mark.unittest
@patch("torch.save")
def test_save_trainer_checkpoint(mock_torch_save, trainer):
    episode = 10
    total_steps = 1000
    initial_best_score = 0.5
    trainer.best_validation_metric = initial_best_score
    trainer.early_stopping_counter = 1

    # Test saving latest
    trainer._save_checkpoint(episode, total_steps, is_best=False)
    expected_checkpoint_latest_only = {
        "episode": episode,
        "total_train_steps": total_steps,
        "best_validation_metric": initial_best_score,
        "early_stopping_counter": 1,
    }
    expected_checkpoint_latest_only.update({
        "agent_config": trainer.agent.config,
        "agent_total_steps": trainer.agent.total_steps,
        "network_state_dict": ANY,
        "target_network_state_dict": ANY,
        "optimizer_state_dict": ANY,
    })
    calls_latest_only = [call(expected_checkpoint_latest_only, trainer.latest_trainer_checkpoint_path)]
    mock_torch_save.assert_has_calls(calls_latest_only)
    mock_torch_save.reset_mock()

    # Test saving best
    test_score = 0.8

    expected_best_checkpoint_dict_content = {
        "episode": episode,
        "total_train_steps": total_steps,
        "best_validation_metric": initial_best_score, # The value *before* update
        "early_stopping_counter": 1,
        "validation_score": test_score # Score should be added when saving best
    }
    expected_best_checkpoint_dict_content.update({
        "agent_config": trainer.agent.config,
        "agent_total_steps": trainer.agent.total_steps,
        "network_state_dict": ANY,
        "target_network_state_dict": ANY,
        "optimizer_state_dict": ANY,
    })

    # Call save for best, providing a score
    trainer._save_checkpoint(episode, total_steps, is_best=True, validation_score=test_score)

    # Verify torch.save was called for both latest and best (with scored name)
    expected_best_path = f"{trainer.best_trainer_checkpoint_base_path}_score_{test_score:.4f}.pt"

    # Check calls made to the mocked torch.save
    # Latest checkpoint should contain the *original* best score, but the added current validation score
    mock_torch_save.assert_any_call(expected_best_checkpoint_dict_content, trainer.latest_trainer_checkpoint_path)
    # Best checkpoint should contain the *original* best score, and the added current validation score
    mock_torch_save.assert_any_call(expected_best_checkpoint_dict_content, expected_best_path)
    # Check total calls
    assert mock_torch_save.call_count == 2


@pytest.mark.unittest
@patch("torch.save")
def test_save_trainer_checkpoint_exception(mock_torch_save, trainer, caplog):
    """Test exception handling during checkpoint saving."""
    error_message = "Disk full error"
    mock_torch_save.side_effect = Exception(error_message)

    with caplog.at_level(logging.ERROR):
        trainer._save_checkpoint(episode=5, total_steps=500, is_best=False)

    assert f"Error saving latest checkpoint: {error_message}" in caplog.text

    # Test exception when saving best as well
    caplog.clear()
    # Make latest save succeed, best save fail
    mock_torch_save.side_effect = [
        None, # Simulate successful latest save
        Exception(error_message + " best"), # Fail best save
    ]
    with caplog.at_level(logging.ERROR):
        trainer._save_checkpoint(episode=5, total_steps=500, is_best=True, validation_score=0.9)

    # Latest save should succeed (no error logged)
    assert f"Error saving latest checkpoint" not in caplog.text
    assert f"Error saving best checkpoint: {error_message + ' best'}" in caplog.text


@pytest.mark.unittest
@patch("torch.save")
def test_save_trainer_checkpoint_no_optimizer_or_network(mock_torch_save, trainer, mock_agent, caplog):
    """Test warnings when saving checkpoint with missing agent components."""
    # Test no optimizer
    original_optimizer = trainer.agent.optimizer
    trainer.agent.optimizer = None
    with caplog.at_level(logging.WARNING):
        trainer._save_checkpoint(episode=1, total_steps=10, is_best=False)
    assert "Agent optimizer not initialized" in caplog.text
    mock_torch_save.assert_not_called()
    trainer.agent.optimizer = original_optimizer # Restore

    # Test no network
    caplog.clear()
    mock_torch_save.reset_mock()
    original_network = trainer.agent.network
    trainer.agent.network = None
    with caplog.at_level(logging.WARNING):
        trainer._save_checkpoint(episode=1, total_steps=10, is_best=False)
    assert "Agent networks not initialized" in caplog.text
    mock_torch_save.assert_not_called()
    trainer.agent.network = original_network # Restore

    # Test no target network
    caplog.clear()
    mock_torch_save.reset_mock()
    original_target_network = trainer.agent.target_network
    trainer.agent.target_network = None
    with caplog.at_level(logging.WARNING):
        trainer._save_checkpoint(episode=1, total_steps=10, is_best=False)
    assert "Agent networks not initialized" in caplog.text
    mock_torch_save.assert_not_called()
    trainer.agent.target_network = original_target_network # Restore

@pytest.mark.unittest
@patch("torch.save")
def test_save_trainer_checkpoint_best_no_score(mock_torch_save, trainer, caplog):
    """Test saving best checkpoint when validation_score is None (fallback)."""
    # The trainer state (best score, etc.) isn't updated here, only file save is tested
    with caplog.at_level(logging.INFO):
         # Note: validation_score is explicitly None
        trainer._save_checkpoint(episode=5, total_steps=500, is_best=True, validation_score=None)

    # Expected path without score
    expected_best_path_no_score = f"{trainer.best_trainer_checkpoint_base_path}.pt"

    # Check that torch.save was called for the best checkpoint with the non-scored name
    # Check both latest and best were saved
    latest_saved = False
    best_no_score_saved = False
    saved_checkpoint_dict = None
    for call_args in mock_torch_save.call_args_list:
        saved_dict, saved_path = call_args[0]
        saved_checkpoint_dict = saved_dict # Capture the dict
        if saved_path == trainer.latest_trainer_checkpoint_path:
            latest_saved = True
        if saved_path == expected_best_path_no_score:
            best_no_score_saved = True

    assert latest_saved, "Latest trainer checkpoint was not saved."
    assert best_no_score_saved, f"Best trainer checkpoint not saved to fallback path {expected_best_path_no_score}"
    # Ensure validation_score key is NOT in the saved dict when None is passed
    assert "validation_score" not in saved_checkpoint_dict


