#!/usr/bin/env python3
# Main training script for transformer trader (Rainbow DQN version)

import os
import logging
from pathlib import Path
import torch
import numpy as np
from datetime import datetime
import yaml # Added for config loading
import argparse # Added for command-line arguments

print("Starting Rainbow DQN training script...")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Assume environment, agent (DDPG version), trainer (DDPG version), utils are correct
from env.trading_env import TradingEnv
print("Imported TradingEnv")
from trainer import RainbowTrainerModule
from agent import RainbowDQNAgent
# Use the new unified logging setup function
from utils.utils import setup_global_logging, set_seeds, get_random_data_file
from data import DataManager
from metrics import PerformanceTracker, calculate_composite_score
from utils.checkpoint_utils import find_latest_checkpoint, load_checkpoint
from evaluation import evaluate_on_test_data
# from hyperparameters import parse_args # Import argument parser

# --- Standard Logging Setup --- 
log_file = Path("logs") / "training.log"
log_file.parent.mkdir(exist_ok=True)
setup_global_logging(log_file_path=log_file, root_level=logging.INFO)
# -----------------------------

# Define loggers using their names (only needed if you want to get it explicitly)
logger = logging.getLogger('Main')

def run_training(config: dict, data_manager: DataManager):
    """Runs the training loop for the Rainbow DQN agent."""
    # Extract relevant config sections
    agent_config = config['agent']
    env_config = config['environment']
    trainer_config = config['trainer']
    run_config = config.get('run', {}) # Optional run section for mode, etc.
    model_dir = run_config.get('model_dir', 'models')
    resume_training = run_config.get('resume', False)
    num_episodes = run_config.get('episodes', 1000) # Default if not in run config
    specific_file = run_config.get('specific_file', None)

    set_seeds(trainer_config['seed'])
    logger.info(f"Running training with config: {config}")

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using {device} device")

    # Agent class is fixed for this script
    AgentClass = RainbowDQNAgent
    # --- Add seed to agent_config --- # Added
    if 'seed' in trainer_config:
        agent_config['seed'] = trainer_config['seed']
    else:
        logger.warning("Seed not found in trainer config, agent may not be fully reproducible.")
        # Optionally set a default seed for the agent if missing entirely
        # agent_config['seed'] = agent_config.get('seed', 42) 
    # ------------------------------- #
    # Agent config validation happens within AgentClass.__init__ if needed
    logger.info(f"Configuring for {AgentClass.__name__} Agent.")

    # --- Initialize variables for potential checkpoint loading ---
    checkpoint = None
    start_episode = 0
    start_total_steps = 0
    initial_best_score = -np.inf
    initial_early_stopping_counter = 0
    optimizer_state = None
    # Buffer state loading is typically not done, but agent load_model now handles optimizer/steps
    # --- End Initialization ---

    # --- Load from Checkpoint if resuming ---
    if resume_training:
        # Find the latest valid checkpoint file (Agent save/load now uses suffix _rainbow_agent.pt)
        checkpoint_path = find_latest_checkpoint(model_dir, suffix='_rainbow_agent.pt')
        if checkpoint_path:
            # Load only metadata needed by trainer, agent loads its own state later
            checkpoint = load_checkpoint(checkpoint_path, map_location='cpu') # Load to CPU first
            if checkpoint:
                # Extract necessary state from the loaded checkpoint dictionary
                # Agent now saves/loads total_steps internally, trainer might need start_episode
                start_episode = checkpoint.get('episode', 0) # Trainer might save this separately if needed
                start_total_steps = checkpoint.get('total_train_steps', 0)
                initial_best_score = checkpoint.get('best_validation_metric', -np.inf)
                initial_early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
                # Logging is now handled within load_checkpoint
                logger.info(f"Resuming training from episode {start_episode}, step {start_total_steps}")
            else:
                logger.warning("Failed to load checkpoint data, starting from scratch.")
        else:
            logger.warning("Resume specified but no checkpoint found, starting from scratch.")

    # --- Instantiate Agent ---
    agent = AgentClass(
        config=agent_config, # Pass the agent's config section
        device=device,
    )
    assert isinstance(agent, RainbowDQNAgent), "Failed to instantiate RainbowDQNAgent"
    logger.info(f"Agent instantiated with {sum(p.numel() for p in agent.network.parameters()):,} parameters.")

    # --- Load Agent State (Network, Optimizer, Steps) if resuming --- 
    if resume_training and checkpoint_path: # Only load if resume is true AND a path was found
        try:
            # Use the agent's own loading method which handles network, optimizer, steps, and config checks
            # The path prefix should not include the suffix
            # Infer prefix from the found checkpoint path
            model_prefix = str(Path(checkpoint_path).parent / Path(checkpoint_path).stem.replace('_rainbow_agent', ''))
            agent.load_model(model_prefix)
            # Overwrite start_total_steps with the value loaded by the agent
            start_total_steps = agent.total_steps
            logger.info(f"Agent state loaded successfully, continuing from step {start_total_steps}.")
        except Exception as e:
            logger.error(f"Error loading agent state using agent.load_model: {e}. Training continues with fresh agent state.", exc_info=True)
            start_total_steps = 0 # Reset steps if loading failed
            start_episode = 0 # Reset episode if loading failed
            initial_best_score = -np.inf
            initial_early_stopping_counter = 0

    # --- Instantiate Trainer ---
    trainer = RainbowTrainerModule(
        agent=agent,
        device=device,
        data_manager=data_manager,
        config=config, # Pass the full config to the trainer
        # Remove handler passing, as root logger handles it now
        # train_log_handler=train_log_handler, 
        # validation_log_handler=validation_log_handler 
    )
    assert isinstance(trainer, RainbowTrainerModule), "Failed to instantiate RainbowTrainerModule"
    logger.info("RAINBOW Trainer initialized.")

    # --- Initial Env Setup ---
    try:
        initial_file = get_random_data_file(data_manager)
        assert isinstance(initial_file, Path), "Failed to get a valid initial data file path"
        logger.info(f"Using initial file for env setup check: {initial_file.name}")
        # Use env_config for environment parameters
        initial_env = TradingEnv(
            data_path=str(initial_file),
            window_size=agent_config['window_size'],
            **env_config # Pass initial_balance, transaction_fee, etc.
        )
        assert isinstance(initial_env, TradingEnv), "Failed to create initial TradingEnv instance"
    except Exception as e:
        logger.error(f"Failed to create initial environment: {e}")
        raise # Stop if initial env setup fails

    logger.info(f"=============================================")
    logger.info(f"STARTING RAINBOW TRAINING{' (Resuming)' if resume_training else ''}")
    logger.info(f"=============================================")

    # --- Run Training ---
    trainer.train(
        env=initial_env,
        num_episodes=num_episodes,
        start_episode=start_episode,
        start_total_steps=start_total_steps,
        initial_best_score=initial_best_score,
        initial_early_stopping_counter=initial_early_stopping_counter,
        specific_file=specific_file
        # Other params like validation_freq, gamma, batch_size etc. are now taken from config inside trainer
    )

    # Close the initial environment (might be redundant if trainer closes final env)
    try:
        initial_env.close()
    except Exception:
        pass # Ignore errors closing env that might already be closed

    return agent, trainer

def main(): # Remove default config_path
    """Main function to load config and run Rainbow DQN training/evaluation."""
    
    # --- Argument Parsing --- # Added
    parser = argparse.ArgumentParser(description='Run Rainbow DQN Training or Evaluation')
    parser.add_argument('--config_path', type=str, default='config/training_config.yaml', 
                        help='Path to the configuration YAML file.')
    args = parser.parse_args()
    config_path = args.config_path
    # ----------------------- #

    # --- Load Configuration --- # Use parsed config_path
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded successfully from {config_path}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}. Exiting.")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}. Exiting.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred loading config: {e}. Exiting.")
        return

    # --- Extract run parameters --- 
    run_config = config.get('run', {})
    mode = run_config.get('mode', 'train') # Default to train if not specified
    model_dir = run_config.get('model_dir', 'models')
    eval_model_prefix = run_config.get('eval_model_prefix', f'{model_dir}/rainbow_transformer_best')
    agent_config = config['agent']
    trainer_config = config['trainer']
    env_config = config['environment']

    # --- Initialize DataManager ---
    data_manager = DataManager()
    assert isinstance(data_manager, DataManager), "Failed to initialize DataManager"

    os.makedirs(model_dir, exist_ok=True)

    if mode == 'train':
        trained_agent, trained_trainer = run_training(config, data_manager)
        assert isinstance(trained_agent, RainbowDQNAgent), "run_training did not return a valid agent"
        assert isinstance(trained_trainer, RainbowTrainerModule), "run_training did not return a valid trainer"

        logger.info("--- Starting Evaluation on Test Data after Training (Rainbow) ---")
        # Pass necessary config parts to evaluation function
        evaluate_on_test_data(
            agent=trained_agent,
            trainer=trained_trainer, # Trainer might hold metrics or env creation logic
            config=config # Pass full config for evaluation needs
        )

    elif mode == 'eval':
        logger.info(f"--- Starting Evaluation Mode (Rainbow) --- ")
        assert isinstance(eval_model_prefix, str) and len(eval_model_prefix) > 0, "Invalid eval_model_prefix in config"
        logger.info(f"Loading model from prefix: {eval_model_prefix}")

        # Determine device
        if torch.backends.mps.is_available():
             device = torch.device("mps")
        elif torch.cuda.is_available():
             device = torch.device("cuda")
        else:
             device = torch.device("cpu")
        logger.info(f"Using device: {device}")

        # Instantiate Rainbow agent using loaded config for evaluation
        eval_agent = RainbowDQNAgent(config=agent_config, device=device)
        assert isinstance(eval_agent, RainbowDQNAgent), "Failed to instantiate agent for evaluation"

        # Load model weights
        # Note: load_model now doesn't need architecture args, they come from agent's config
        eval_agent.load_model(
             eval_model_prefix,
        )
        assert eval_agent.network is not None, f"Model loading failed for prefix {eval_model_prefix}, network is None"
        logger.info("Model loaded successfully for evaluation.")
        eval_agent.set_training_mode(False)

        # Pass full config to trainer for evaluation setup (if needed)
        eval_trainer = RainbowTrainerModule(agent=eval_agent, device=device, data_manager=data_manager, config=config)
        assert isinstance(eval_trainer, RainbowTrainerModule), "Failed to instantiate trainer for evaluation"
        # Run evaluation - internal asserts will check inputs
        evaluate_on_test_data(
            agent=eval_agent,
            trainer=eval_trainer, # Trainer might hold metrics or env creation logic
            config=config # Pass full config for evaluation needs
        )

    else:
        logger.error(f"Invalid mode specified in config run section: {mode}. Use 'train' or 'eval'.") # Or raise ValueError

    logger.info(f"Script finished ({mode} mode, agent: rainbow).")

if __name__ == "__main__":
    main() # Call main without arguments