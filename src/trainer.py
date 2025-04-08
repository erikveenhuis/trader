import torch
import numpy as np
import logging
import os
from pathlib import Path
from env import TradingEnv  # Updated import
from agent import RainbowDQNAgent  # Assume agent.py is in src
from data import DataManager  # Updated import
from metrics import (
    PerformanceTracker,
    calculate_composite_score,
)  # Assume metrics.py is in src
from collections import deque  # Keep deque for performance_tracker
from typing import List, Tuple  # Added Tuple back
import json  # Keep for saving results
from datetime import datetime  # Added datetime back

# Use root logger - configuration handled by main script
logger = logging.getLogger("Trainer")


class RainbowTrainerModule:
    def __init__(
        self,
        agent: RainbowDQNAgent,
        device: torch.device,
        data_manager: DataManager,
        config: dict,
    ):
        assert isinstance(
            agent, RainbowDQNAgent
        ), "Agent must be an instance of RainbowDQNAgent"
        assert isinstance(device, torch.device), "Device must be a torch.device"
        assert isinstance(
            data_manager, DataManager
        ), "Data manager must be an instance of DataManager"
        assert isinstance(config, dict), "Config must be a dictionary"
        self.agent = agent
        self.device = device  # Store the device
        self.data_manager = data_manager
        self.config = config
        self.agent_config = config["agent"]
        self.env_config = config["environment"]
        self.trainer_config = config["trainer"]
        self.run_config = config.get("run", {})
        self.best_validation_metric = -np.inf
        # Adjust path prefix for Rainbow models
        self.best_model_path_prefix = str(
            Path(self.run_config.get("model_dir", "models"))
            / "rainbow_transformer_best"
        )
        # Add checkpoint paths
        self.latest_trainer_checkpoint_path = str(
            Path(self.run_config.get("model_dir", "models"))
            / "checkpoint_trainer_latest.pt"
        )
        self.best_trainer_checkpoint_path = str(
            Path(self.run_config.get("model_dir", "models"))
            / "checkpoint_trainer_best.pt"
        )
        self.validation_metrics = []
        self.performance_tracker = PerformanceTracker()
        self.early_stopping_patience = self.trainer_config.get(
            "early_stopping_patience", 10
        )
        self.early_stopping_counter = 0
        self.min_validation_threshold = self.trainer_config.get(
            "min_validation_threshold", 0.0
        )
        self.validation_freq = self.trainer_config.get("validation_freq", 10)
        self.checkpoint_save_freq = self.trainer_config.get("checkpoint_save_freq", 10)
        self.reward_window = self.trainer_config.get("reward_window", 10)
        self.update_freq = self.trainer_config.get("update_freq", 4)
        self.log_freq = self.trainer_config.get("log_freq", 60)
        self.warmup_steps = self.trainer_config.get("warmup_steps", 50000)
        # Extract run parameters
        self.model_dir = self.run_config.get("model_dir", "models")
        os.makedirs(self.model_dir, exist_ok=True)

    def should_stop_early(self, validation_metrics: List[dict]) -> bool:
        """Check if training should stop early based on validation performance."""
        if not validation_metrics:
            return False

        current_score = calculate_composite_score(validation_metrics)

        if current_score > self.best_validation_metric:
            self.best_validation_metric = current_score
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return self.early_stopping_counter >= self.early_stopping_patience

    def should_validate(self, episode: int, recent_performance: dict) -> bool:
        """Determine if validation should be performed based on validation frequency."""
        # Removed complex logic based on improvement_rate and stability
        # Simplified to validate purely based on frequency
        return (episode + 1) % self.validation_freq == 0

    def _save_trainer_checkpoint(self, episode: int, total_steps: int, is_best: bool):
        """Save trainer-specific checkpoint (episode, validation score, etc.)."""
        assert (
            isinstance(episode, int) and episode >= 0
        ), "Invalid episode number for checkpoint"
        assert (
            isinstance(total_steps, int) and total_steps >= 0
        ), "Invalid total_steps for checkpoint"
        assert isinstance(is_best, bool), "is_best flag must be boolean"

        # Agent state (network, optimizer, agent_steps) is saved via agent.save_model()
        # This checkpoint primarily stores trainer state.
        checkpoint = {
            "episode": episode,
            "total_train_steps": total_steps,  # Store steps from trainer perspective
            "best_validation_metric": self.best_validation_metric,
            "early_stopping_counter": self.early_stopping_counter,
            # Replay buffer state is generally not saved due to size and complexity.
            # 'buffer_state': ...,
            # 'n_step_buffer_state': ...,
            # We can optionally save agent config hash or version here for compatibility checks
        }
        # Basic check on checkpoint contents
        # assert isinstance(checkpoint['network_state_dict'], dict), "Invalid network state dict in checkpoint"
        # assert isinstance(checkpoint['optimizer_state_dict'], dict), "Invalid optimizer state dict in checkpoint"
        assert isinstance(
            checkpoint["best_validation_metric"], float
        ), "Invalid best validation metric type in checkpoint"

        if not self.agent.optimizer:
            logger.warning("Agent optimizer not initialized, cannot save checkpoint.")
            return
        if self.agent.network is None or self.agent.target_network is None:
            logger.warning("Agent networks not initialized, cannot save checkpoint.")
            return

        # Save the latest checkpoint
        try:
            torch.save(checkpoint, self.latest_trainer_checkpoint_path)
            logger.info(
                f"Latest checkpoint saved to {self.latest_trainer_checkpoint_path}"
            )
            logger.info(f"  Episode: {episode}")
            logger.info(f"  Total Steps: {total_steps}")
            logger.info(f"  Best Validation Score: {self.best_validation_metric:.4f}")
            logger.info(f"  Early Stopping Counter: {self.early_stopping_counter}")
        except Exception as e:
            logger.error(f"Error saving latest checkpoint: {e}")

        # Save the best checkpoint if this is the best model so far
        if is_best:
            try:
                torch.save(checkpoint, self.best_trainer_checkpoint_path)
                logger.info(
                    f"Best checkpoint saved to {self.best_trainer_checkpoint_path}"
                )
                logger.info(
                    f"  New Best Validation Score: {self.best_validation_metric:.4f}"
                )
                logger.info(f"  Episode: {episode}")
                logger.info(f"  Total Steps: {total_steps}")
            except Exception as e:
                logger.error(f"Error saving best checkpoint: {e}")

    def train(
        self,
        env: TradingEnv,
        num_episodes: int,
        start_episode: int,
        start_total_steps: int,
        initial_best_score: float,
        initial_early_stopping_counter: int,
        specific_file: str | None = None,
    ):
        """Train the Rainbow DQN agent."""
        assert (
            isinstance(num_episodes, int) and num_episodes > 0
        ), "num_episodes must be a positive integer"
        assert (
            isinstance(start_episode, int) and start_episode >= 0
        ), "start_episode must be a non-negative integer"
        assert (
            isinstance(start_total_steps, int) and start_total_steps >= 0
        ), "start_total_steps must be non-negative"
        assert isinstance(
            specific_file, (str, type(None))
        ), "specific_file must be a string or None"

        # --- Initialize state from checkpoint or defaults ---
        self.best_validation_metric = initial_best_score
        self.early_stopping_counter = initial_early_stopping_counter
        total_train_steps = start_total_steps
        # ---------------------------------------------------

        self.agent.set_training_mode(True)

        # NOTE: Optimizer and buffer state loading is handled by agent.load_model in train.py

        total_rewards = []
        total_losses = []  # To track average loss per episode
        best_train_reward = -np.inf

        logger.info("====== STARTING/RESUMING RAINBOW DQN TRAINING ======")
        logger.info(f"Starting from Episode: {start_episode + 1}/{num_episodes}")
        logger.info(f"Starting from Total Steps: {total_train_steps}")
        logger.info(
            f"Episodes: {num_episodes}, Update Freq: {self.update_freq}, Log Freq: {self.log_freq}"
        )
        logger.info(
            f"Agent Gamma: {self.agent.gamma}, LR: {self.agent.lr}, Batch Size: {self.agent.batch_size}"
        )
        logger.info(f"Agent Target Update Freq: {self.agent.target_update_freq}")
        logger.info(
            f"Agent Atoms: {self.agent.num_atoms}, Vmin: {self.agent.v_min}, Vmax: {self.agent.v_max}"
        )
        logger.info(f"Agent N-Steps: {self.agent.n_steps}")
        logger.info(
            f"PER Alpha: {self.agent.buffer.alpha}, Beta Start: {self.agent.buffer.beta_start}"
        )
        logger.info(f"Warmup Steps: {self.warmup_steps}")
        logger.info(f"Early Stopping Patience: {self.early_stopping_patience}")

        # Get validation files
        val_files = self.data_manager.get_validation_files()
        if not val_files:
            logger.warning(
                "No validation files found. Training will proceed without validation."
            )

        # Modify episode loop to start from loaded episode
        for episode in range(start_episode, num_episodes):
            logger.info(f"--- Starting Episode {episode + 1}/{num_episodes} ---")
            # --- REMOVED file path logic ---
            # The provided 'env' argument should be used directly.
            # Ensure the passed env supports reset() for multiple episodes.

            # --- REMOVED internal env creation ---
            # try:
            #      env = TradingEnv(...)
            # except ...

            # Call reset *on the provided env*
            try:
                obs, _ = env.reset()  # Use the provided env instance
                assert isinstance(
                    _["portfolio_value"], (float, np.float32, np.float64)
                ), "Reset info missing valid portfolio_value"
                # --- Assert observation structure (Moved Here) ---
                assert isinstance(
                    obs, dict
                ), "Observation from env.reset() must be a dict"
                assert (
                    "market_data" in obs and "account_state" in obs
                ), "Observation missing keys"
                assert isinstance(
                    obs["market_data"], np.ndarray
                ), "Market data is not a numpy array"
                assert isinstance(
                    obs["account_state"], np.ndarray
                ), "Account state is not a numpy array"
                # --- End Assert ---
            except Exception:
                logger.error(
                    f"!!! Exception during env.reset() for {env.data_path} !!!",
                    exc_info=True,
                )
                continue  # Skip episode if reset fails

            done = False
            episode_reward = 0
            episode_loss = 0
            steps = 0  # Correctly reset per-episode steps

            # Reset performance tracker AFTER successful reset and store initial value
            self.performance_tracker = PerformanceTracker()
            initial_portfolio_value = _[
                "portfolio_value"
            ]  # Get initial value from reset info
            self.performance_tracker.add_initial_value(initial_portfolio_value)

            # Initialize deque to store recent step rewards for logging
            recent_step_rewards = deque(maxlen=self.log_freq)

            while not done:
                # logger.debug(f"  Ep {episode+1} Step {steps}: Top of main loop.") # REMOVED DEBUG
                # --- Assert observation shape before selecting action ---
                assert obs["market_data"].shape == (
                    self.agent.window_size,
                    self.agent.n_features,
                ), f"Market data shape mismatch before action selection: Expected {(self.agent.window_size, self.agent.n_features)}, got {obs['market_data'].shape}"
                assert obs["account_state"].shape == (
                    2,
                ), f"Account state shape mismatch before action selection: Expected (2,), got {obs['account_state'].shape}"
                # --- End Assert ---

                # Get action from agent - Add noise during training
                # During warmup, take random actions
                if total_train_steps < self.warmup_steps:
                    action = env.action_space.sample()
                    assert env.action_space.contains(
                        action
                    ), "Sampled warmup action is invalid"
                else:
                    action = self.agent.select_action(obs)
                    assert isinstance(
                        action, (int, np.integer)
                    ), "Agent action is not an integer"
                    assert (
                        0 <= action < self.agent.num_actions
                    ), "Agent action is out of bounds"

                # Take action in environment
                try:
                    next_obs, reward, done, _, info = env.step(action)
                    # --- Assert env.step outputs ---
                    assert isinstance(
                        next_obs, dict
                    ), "Next observation from env.step() must be a dict"
                    assert (
                        "market_data" in next_obs and "account_state" in next_obs
                    ), "Next observation missing keys"
                    assert isinstance(
                        next_obs["market_data"], np.ndarray
                    ), "Next market data is not a numpy array"
                    assert isinstance(
                        next_obs["account_state"], np.ndarray
                    ), "Next account state is not a numpy array"
                    assert isinstance(
                        reward, (float, np.float32, np.float64)
                    ), "Reward is not a float"
                    assert isinstance(
                        done, (bool, np.bool_)
                    ), "Done flag is not boolean"
                    assert isinstance(info, dict), "Info object is not a dict"
                    assert not np.isnan(reward) and not np.isinf(
                        reward
                    ), "Reward is NaN or Inf"
                    # --- End Assert ---
                except Exception as e:
                    logger.error(
                        f"Error during env.step at step {steps} in episode {episode}: {e}",
                        exc_info=True,
                    )
                    done = True  # End episode if environment crashes
                    reward = -10.0  # Penalize env crash (ensure float)
                    next_obs = obs  # Keep last valid observation
                    info = self._get_fallback_info(
                        obs, info if "info" in locals() else {}
                    )  # Get fallback info

                # Track portfolio value
                self.performance_tracker.update(
                    portfolio_value=info["portfolio_value"],
                    action=action,
                    reward=reward,
                    transaction_cost=info.get("transaction_cost", 0),
                )

                # Store reward for step logging
                recent_step_rewards.append(reward)

                episode_reward += reward

                # Store transition in replay buffer
                # logger.debug(f"  Ep {episode+1} Step {steps}: Storing transition...") # REMOVED DEBUG
                self.agent.store_transition(obs, action, reward, next_obs, done)
                # logger.debug(f"  Ep {episode+1} Step {steps}: Transition stored.") # REMOVED DEBUG

                # Update state
                obs = next_obs
                steps += 1
                total_train_steps += 1

                # Learn from experience - only update if buffer is full enough and after warmup
                if (
                    len(self.agent.buffer) >= self.agent.batch_size
                    and total_train_steps > self.warmup_steps
                    and total_train_steps % self.update_freq == 0
                ):
                    try:
                        loss_value = self.agent.learn()
                        if loss_value is not None:
                            assert isinstance(
                                loss_value, float
                            ), "Loss value is not a float"
                            assert not np.isnan(loss_value) and not np.isinf(
                                loss_value
                            ), "Loss value is NaN or Inf"
                            episode_loss += loss_value
                    except Exception as e:
                        logger.error(
                            f"!!! EXCEPTION during learning update at step {total_train_steps} !!!"
                        )
                        logger.exception(e)
                        done = True

                # Log step info (reduced frequency maybe)
                # Capture the reward from the step *before* potentially averaging
                current_step_reward = reward
                if steps % self.log_freq == 0:
                    metrics = self.performance_tracker.get_recent_metrics()
                    # Calculate mean reward over the last log_freq steps
                    mean_recent_reward = (
                        np.mean(list(recent_step_rewards))
                        if recent_step_rewards
                        else 0.0
                    )
                    logger.info(
                        f"  Ep {episode+1} Step {steps}: Port=${metrics['portfolio_value']:.2f}, "
                        f"Act={action:.0f}, StepRew={current_step_reward:.4f}, "
                        f"MeanRew-{self.log_freq}={mean_recent_reward:.4f}, "
                        f"Price=${info.get('price', 0):.8f}, TxCost=${info.get('transaction_cost', 0):.2f}"
                    )

            # Close the environment for this episode
            env.close()

            # End of episode logging
            total_rewards.append(episode_reward)
            current_window_size = min(self.reward_window, len(total_rewards))
            recent_avg_reward = np.mean(total_rewards[-current_window_size:])
            avg_reward = (
                np.mean(total_rewards[-100:])
                if len(total_rewards) >= 100
                else np.mean(total_rewards)
            )
            avg_loss = (
                episode_loss / (steps / self.update_freq)
                if steps > 0 and episode_loss != 0
                else 0
            )

            metrics = self.performance_tracker.get_metrics()
            logger.info(
                f"Ep {episode+1}: Reward={episode_reward:.2f}, Mean-{current_window_size}={recent_avg_reward:.2f}, "
                f"Loss={avg_loss:.4f}, "
                f"Portfolio=${metrics['portfolio_value']:.2f} ({metrics['total_return']:.2f}%), "
                f"Sharpe={metrics['sharpe_ratio']:.4f}, WinRate={metrics['win_rate']:.4f}"
            )

            # --- Checkpoint Saving ---
            save_now = (episode + 1) % self.checkpoint_save_freq == 0
            is_best = False  # Will be updated if validation improves
            # -------------------------

            # Run validation if needed
            if val_files and self.should_validate(
                episode, self.performance_tracker.get_recent_metrics()
            ):
                # logger.debug(f"Ep {episode+1}: Entering validation block.") # REMOVED DEBUG
                logger.info(f"--- Running validation after episode {episode + 1} ---")
                should_stop, validation_score = self.validate(val_files)
                # logger.debug(f"Ep {episode+1}: Exiting validation block. should_stop={should_stop}, score={validation_score:.4f}") # REMOVED DEBUG

                # Enhanced logging for validation score comparison
                logger.info("Validation Score Comparison:")
                logger.info(f"  Current Score: {validation_score:.4f}")
                logger.info(f"  Previous Best: {self.best_validation_metric:.4f}")
                logger.info(
                    f"  Difference: {(validation_score - self.best_validation_metric):.4f}"
                )

                if validation_score > self.best_validation_metric:
                    is_best = True
                    self.best_validation_metric = validation_score
                    self.agent.save_model(self.best_model_path_prefix)
                    logger.info("  >>> NEW BEST MODEL SAVED <<<")
                    logger.info(f"  Score: {validation_score:.4f}")
                    logger.info(
                        f"  Improvement: {(validation_score - self.best_validation_metric):.4f}"
                    )
                    logger.info(f"  Saved to: {self.best_model_path_prefix}*")
                else:
                    logger.info("  No improvement over previous best model")

                # Save checkpoint AFTER validation, potentially marking it as best
                self._save_trainer_checkpoint(
                    episode=episode + 1, total_steps=total_train_steps, is_best=is_best
                )
                save_now = False  # Don't save again immediately after validation

                if should_stop:
                    logger.info("Early stopping triggered. Training will stop.")
                    break  # Break after saving checkpoint

            # Periodic checkpoint saving (if not saved after validation)
            if save_now:
                self._save_trainer_checkpoint(
                    episode=episode + 1, total_steps=total_train_steps, is_best=False
                )

            # Fallback: Save based on training rewards if no validation or validation failed
            # Only save if avg_reward is improving and after some initial episodes
            save_interval_fallback = (
                20  # Save less frequently based on training rewards
            )
            if (
                not val_files
                and episode > min(50, num_episodes // 5)
                and (episode + 1) % save_interval_fallback == 0
            ):
                if avg_reward > best_train_reward:
                    best_train_reward = avg_reward
                    self.agent.save_model(
                        self.best_model_path_prefix
                    )  # Use same prefix
                    logger.info(
                        f"  SAVED BEST MODEL based on training reward: {self.best_model_path_prefix}* with avg reward: {best_train_reward:.2f}"
                    )

        # Save final model using prefix
        final_model_prefix = str(Path(self.model_dir) / "rainbow_transformer_final")
        self.agent.save_model(final_model_prefix)
        # Save final trainer checkpoint
        self._save_trainer_checkpoint(
            episode=num_episodes, total_steps=total_train_steps, is_best=False
        )
        # ------------------

        logger.info("====== RAINBOW DQN TRAINING COMPLETED ======")
        logger.info(f"Total steps: {total_train_steps}")
        logger.info(f"Final model saved to {final_model_prefix}*")

        # Log best validation score
        if val_files and self.validation_metrics:
            # Find best metric and episode
            best_run = max(self.validation_metrics, key=lambda x: x["metric"])
            logger.info(
                f"Best validation metric: {best_run['metric']:.4f} (at episode {best_run['episode']})"
            )
            logger.info(
                f"Best validation model saved to: {self.best_model_path_prefix}*"
            )
        elif not val_files:
            logger.info(f"Best average reward during training: {best_train_reward:.2f}")
            logger.info(f"Best training model saved to: {self.best_model_path_prefix}*")

    def evaluate_for_validation(self, env: TradingEnv) -> Tuple[float, dict]:
        """Evaluate the agent for validation purposes with comprehensive metrics."""
        assert isinstance(
            env, TradingEnv
        ), "env must be an instance of TradingEnv for validation"
        was_training = self.agent.training_mode
        self.agent.set_training_mode(False)

        try:
            obs, info = env.reset()
            # --- Assert observation structure ---
            assert isinstance(
                obs, dict
            ), "Validation: Observation from env.reset() must be a dict"
            assert (
                "market_data" in obs and "account_state" in obs
            ), "Validation: Observation missing keys"
            assert isinstance(
                obs["market_data"], np.ndarray
            ), "Validation: Market data is not a numpy array"
            assert isinstance(
                obs["account_state"], np.ndarray
            ), "Validation: Account state is not a numpy array"
            # --- End Assert ---
        except Exception as e:
            logger.error(f"Error resetting validation env: {e}")
            self.agent.set_training_mode(was_training)
            return -np.inf, {}

        done = False
        total_reward = 0
        tracker = PerformanceTracker()

        while not done:
            try:
                action = self.agent.select_action(obs)
                next_obs, reward, done, _, info = env.step(action)

                # Update performance tracker
                tracker.update(
                    portfolio_value=info["portfolio_value"],
                    action=action,
                    reward=reward,
                    transaction_cost=info.get("transaction_cost", 0),
                )

                total_reward += reward
                obs = next_obs
            except Exception as e:
                logger.error(f"Error during validation step: {e}")
                done = True
                total_reward -= 10.0  # Use float for subtraction

        metrics = tracker.get_metrics()
        self.agent.set_training_mode(was_training)

        return total_reward, metrics

    def validate(self, val_files: List[Path]) -> Tuple[bool, float]:
        """Run validation on validation files, log to validation file, and check for early stopping."""

        # Handle empty validation file list
        if not val_files:
            logger.warning(
                "validate() called with empty val_files list. Returning default score -inf."
            )
            return False, -np.inf  # No early stopping, worst possible score

        metrics = []
        results = []

        try:  # Main validation logic wrapped in try
            logger.info("============================================")
            logger.info(f"RUNNING VALIDATION ON {len(val_files)} FILES")
            logger.info(
                f"Current best validation score: {self.best_validation_metric:.4f}"
            )
            logger.info("============================================")

            for i, val_file in enumerate(val_files):
                logger.info(
                    f"--- VALIDATING ON FILE {i+1}/{len(val_files)}: {val_file.name} ---"
                )
                try:
                    env = TradingEnv(
                        data_path=str(val_file),
                        window_size=self.agent_config["window_size"],
                        **self.env_config,  # Pass initial_balance, transaction_fee etc.
                    )
                    reward, file_metrics = self.evaluate_for_validation(env)

                    # Store detailed results - convert numpy types to Python native types
                    result = {
                        "file": val_file.name,
                        "reward": float(reward),
                        "portfolio_value": float(file_metrics["portfolio_value"]),
                        "total_return": float(file_metrics["total_return"]),
                        "sharpe_ratio": float(file_metrics["sharpe_ratio"]),
                        "max_drawdown": float(file_metrics["max_drawdown"]),
                        "win_rate": float(file_metrics["win_rate"]),
                        "avg_action": float(file_metrics["avg_action"]),
                        "transaction_costs": float(file_metrics["transaction_costs"]),
                    }
                    results.append(result)
                    metrics.append(file_metrics)
                    env.close()

                    # Enhanced per-file logging
                    logger.info(f"  Results for {val_file.name}:")
                    logger.info(f"    Reward: {reward:.4f}")
                    logger.info(
                        f"    Portfolio Value: ${file_metrics['portfolio_value']:.2f}"
                    )
                    logger.info(
                        f"    Total Return: {file_metrics['total_return']:.2f}%"
                    )
                    logger.info(f"    Sharpe Ratio: {file_metrics['sharpe_ratio']:.4f}")
                    logger.info(f"    Max Drawdown: {file_metrics['max_drawdown']:.4f}")
                    logger.info(f"    Win Rate: {file_metrics['win_rate']:.4f}")
                    logger.info(f"    Average Action: {file_metrics['avg_action']:.4f}")
                    logger.info(
                        f"    Transaction Costs: ${file_metrics['transaction_costs']:.2f}"
                    )

                except Exception as e:
                    logger.error(f"Error validating on file {val_file.name}: {e}")
                    continue

            # Calculate average metrics
            avg_metrics = {
                "avg_reward": float(np.mean([m["avg_reward"] for m in metrics])),
                "portfolio_value": float(
                    np.mean([m["portfolio_value"] for m in metrics])
                ),
                "total_return": float(np.mean([m["total_return"] for m in metrics])),
                "sharpe_ratio": float(np.mean([m["sharpe_ratio"] for m in metrics])),
                "max_drawdown": float(np.mean([m["max_drawdown"] for m in metrics])),
                "win_rate": float(np.mean([m["win_rate"] for m in metrics])),
                "avg_action": float(np.mean([m["avg_action"] for m in metrics])),
                "transaction_costs": float(
                    np.mean([m["transaction_costs"] for m in metrics])
                ),
            }

            # Calculate composite score
            validation_score = calculate_composite_score(avg_metrics)

            # Log validation summary
            logger.info("\n=== VALIDATION SUMMARY ===")
            logger.info(f"Composite Score: {validation_score:.4f}")
            logger.info(f"Previous Best Score: {self.best_validation_metric:.4f}")
            logger.info(
                f"Score Difference: {validation_score - self.best_validation_metric:.4f}"
            )
            logger.info(f"Average Reward: {avg_metrics['avg_reward']:.2f}")
            logger.info(f"Average Portfolio: ${avg_metrics['portfolio_value']:.2f}")
            logger.info(f"Average Return: {avg_metrics['total_return']:.2f}%")
            logger.info(f"Average Sharpe: {avg_metrics['sharpe_ratio']:.4f}")
            logger.info(f"Average Max Drawdown: {avg_metrics['max_drawdown']:.4f}")
            logger.info(f"Average Win Rate: {avg_metrics['win_rate']:.4f}")
            logger.info(f"Average Action: {avg_metrics['avg_action']:.4f}")
            logger.info(
                f"Average Transaction Costs: ${avg_metrics['transaction_costs']:.2f}"
            )
            logger.info("============================================")

            # Save validation results to JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = Path(self.model_dir) / f"validation_results_{timestamp}.json"
            try:
                with open(results_file, "w") as f:
                    json_results = {
                        "timestamp": timestamp,
                        "validation_score": float(validation_score),
                        "best_validation_score": float(self.best_validation_metric),
                        "score_difference": float(
                            validation_score - self.best_validation_metric
                        ),
                        "average_metrics": avg_metrics,
                        "detailed_results": results,
                    }
                    json.dump(json_results, f, indent=4)
                logger.info(f"Validation results saved to {results_file}")
            except Exception as e:
                logger.error(f"Error saving validation results: {e}")

            # Check for early stopping
            should_stop = False
            if validation_score > self.best_validation_metric:
                self.best_validation_metric = validation_score
                self.early_stopping_counter = 0
                should_stop = False
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered after {self.early_stopping_counter} episodes without improvement"
                    )
                    should_stop = True

            return should_stop, validation_score

        finally:
            pass  # Keep finally block for potential future use

    def _get_fallback_info(self, last_obs: dict, last_info: dict) -> dict:
        """Provides a fallback info dictionary if env.step crashes."""
        # Try to get last known portfolio value, default to 0 if unavailable or invalid
        fallback_portfolio_value = last_info.get("portfolio_value", 0.0)
        if (
            not isinstance(fallback_portfolio_value, (float, np.float32, np.float64))
            or fallback_portfolio_value < 0
        ):
            fallback_portfolio_value = 0.0

        return {
            "step": last_info.get("step", -1),
            "price": last_info.get("price", 0.0),
            "balance": last_info.get("balance", 0.0),
            "position": last_info.get("position", 0.0),
            "portfolio_value": fallback_portfolio_value,  # Ensure valid value
            "transaction_cost": last_info.get("transaction_cost", 0.0),
            "error": "Environment step failed",
        }

    def evaluate(self, env: TradingEnv, render=False):
        """Evaluate the agent on one episode with detailed logging."""
        assert isinstance(
            env, TradingEnv
        ), "env must be an instance of TradingEnv for evaluation"
        self.agent.set_training_mode(False)  # Ensure evaluation mode

        try:
            obs, info = env.reset()
            # --- Assert observation structure ---
            assert isinstance(
                obs, dict
            ), "Evaluation: Observation from env.reset() must be a dict"
            assert (
                "market_data" in obs and "account_state" in obs
            ), "Evaluation: Observation missing keys"
            assert isinstance(
                obs["market_data"], np.ndarray
            ), "Evaluation: Market data is not numpy array"
            assert isinstance(
                obs["account_state"], np.ndarray
            ), "Evaluation: Account state is not numpy array"
            # --- End Assert ---
        except Exception as e:
            logger.error(f"Error resetting evaluation env: {e}")
            return -np.inf, 0  # Return poor score

        done = False
        total_reward = 0.0  # Initialize as float
        steps = 0

        actions_taken = []
        portfolio_values = []
        prices = []
        initial_portfolio = None

        logger.info("====== STARTING EVALUATION ======")
        initial_portfolio = info.get("portfolio_value", env.initial_balance)
        portfolio_values.append(initial_portfolio)
        prices.append(info.get("price", 0))  # Store initial price

        while not done:
            try:
                # --- Assert observation shape ---
                assert obs["market_data"].shape == (
                    self.agent.window_size,
                    self.agent.n_features,
                ), "Evaluation: Market data shape mismatch"
                assert obs["account_state"].shape == (
                    2,
                ), "Evaluation: Account state shape mismatch"
                # --- End Assert ---
                action = self.agent.select_action(obs)
                assert isinstance(
                    action, (int, np.integer)
                ), "Evaluation: Agent action is not an integer"

                # Store the integer action directly
                actions_taken.append(action)

                next_obs, reward, done, _, info = env.step(action)
                # --- Assert env.step outputs ---
                assert isinstance(
                    next_obs, dict
                ), "Evaluation: Next observation must be a dict"
                assert isinstance(
                    reward, (float, np.float32, np.float64)
                ), "Evaluation: Reward is not a float"
                assert isinstance(
                    done, (bool, np.bool_)
                ), "Evaluation: Done flag is not boolean"
                assert isinstance(info, dict), "Evaluation: Info object is not a dict"
                assert not np.isnan(reward) and not np.isinf(
                    reward
                ), "Evaluation: Reward is NaN or Inf"
                # --- End Assert ---

                current_price = info.get("price", prices[-1] if prices else 0)
                portfolio_value = info.get("portfolio_value", portfolio_values[-1])

                portfolio_values.append(portfolio_value)
                prices.append(current_price)

                total_reward += reward
                steps += 1

                if steps % 50 == 0:
                    # Use the integer action directly in the log message
                    logger.info(
                        f"  Step {steps}: Portfolio=${portfolio_value:.2f}, Action={action:d}, Price=${current_price:.2f}"
                    )

                obs = next_obs
            except Exception as e:
                logger.error(
                    f"Error during evaluation step {steps}: {e}", exc_info=True
                )
                done = True  # End evaluation on error

        # Log evaluation summary (using safer gets and fallbacks)
        final_portfolio = portfolio_values[-1] if portfolio_values else 0.0
        initial_portfolio_val = portfolio_values[0] if portfolio_values else 0.0
        return_pct = (
            (final_portfolio - initial_portfolio_val) / (initial_portfolio_val + 1e-9)
        ) * 100  # Add epsilon for safety

        # --- Ensure stats are calculated correctly even if actions_taken is empty ---
        if actions_taken:
            avg_action = np.mean(actions_taken)
            min_action = np.min(actions_taken)
            max_action = np.max(actions_taken)
        else:
            avg_action, min_action, max_action = 0.0, 0.0, 0.0

        logger.info("====== EVALUATION SUMMARY ======")
        logger.info(f"Steps: {steps}")
        logger.info(f"Total Reward: {total_reward:.2f}")
        logger.info(f"Initial Portfolio: ${initial_portfolio_val:.2f}")
        logger.info(f"Final Portfolio: ${final_portfolio:.2f}")
        logger.info(f"Return: {return_pct:.2f}%")
        logger.info(
            f"Action stats - Avg: {avg_action:.3f}, Min: {min_action:.3f}, Max: {max_action:.3f}"
        )

        # Portfolio vs Price Growth Comparison (added safety checks)
        portfolio_growth = (
            (portfolio_values[-1] / (portfolio_values[0] + 1e-9))
            if len(portfolio_values) > 0
            else 1.0
        )
        logger.info(f"Portfolio Growth Factor: {portfolio_growth:.3f}x")

        price_growth = (prices[-1] / (prices[0] + 1e-9)) if len(prices) > 1 else 1.0
        if len(prices) > 1:
            logger.info(f"Price Growth Factor: {price_growth:.3f}x")
            if price_growth > 1e-9:
                outperform = portfolio_growth / price_growth
                performance_str = (
                    f"OUTPERFORMS asset by {(outperform - 1) * 100:.2f}%"
                    if outperform > 1
                    else f"UNDERPERFORMS asset by {(1 - outperform) * 100:.2f}%"
                )
                logger.info(f"Strategy {performance_str}")
            else:
                logger.info(
                    "Price growth factor is zero or negative, cannot compare performance."
                )

        # --- Ensure returned values are floats ---
        final_reward = float(total_reward)
        final_portfolio_val = float(final_portfolio)
        assert isinstance(
            final_reward, float
        ), "Final reward type mismatch before return"
        assert isinstance(
            final_portfolio_val, float
        ), "Final portfolio type mismatch before return"
        # --- End Ensure ---

        return final_reward, final_portfolio_val
