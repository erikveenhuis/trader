import torch
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict, Any
import os

# Import necessary components from other modules
from agent import RainbowDQNAgent
from trainer import RainbowTrainerModule
from env import TradingEnv
from data import DataManager
from utils.utils import set_seeds
from metrics import PerformanceTracker, calculate_composite_score

logger = logging.getLogger('Evaluation')

def evaluate_on_test_data(agent: RainbowDQNAgent, trainer: RainbowTrainerModule, config: dict) -> List[Dict[str, Any]]:
    """Evaluate the trained Rainbow agent on test data.
    
    Args:
        agent: The trained agent instance.
        trainer: The trainer instance (used for data_manager and evaluate method).
        config: The configuration dictionary containing agent, env, trainer, and run settings.

    Returns:
        A list of dictionaries containing evaluation results for each test file.
    """
    # --- Start: Input Asserts ---
    assert isinstance(agent, RainbowDQNAgent), "evaluate_on_test_data requires a RainbowDQNAgent instance"
    assert isinstance(trainer, RainbowTrainerModule), "evaluate_on_test_data requires a RainbowTrainerModule instance"
    assert isinstance(config, dict), "Config must be a dictionary"
    # --- End: Input Asserts ---

    # Extract necessary parameters from config
    agent_config = config['agent']
    env_config = config['environment']
    trainer_config = config['trainer']
    run_config = config.get('run', {})
    seed = trainer_config['seed']
    window_size = agent_config['window_size']
    model_dir = run_config.get('model_dir', 'models')

    set_seeds(seed)
    data_manager = trainer.data_manager
    test_files = data_manager.get_test_files()

    if not test_files:
        logger.warning("No test files found. Skipping test evaluation.")
        return []

    logger.info(f"=============================================")
    logger.info(f"EVALUATING RAINBOW MODEL ON TEST DATA")
    logger.info(f"Found {len(test_files)} test files")
    logger.info(f"=============================================")

    agent.set_training_mode(False) 

    results = []
    for i, test_file in enumerate(test_files):
        logger.info(f"--- TEST FILE {i+1}/{len(test_files)}: {test_file.name} ---")
        try:
            test_env = TradingEnv(
                data_path=str(test_file),
                window_size=window_size,
                **env_config
            )
            assert isinstance(test_env, TradingEnv), f"Failed to create test env for {test_file.name}"
            
            reward, eval_metrics = trainer.evaluate_for_validation(test_env) # Re-use validation evaluation logic
            
            assert isinstance(reward, (float, np.float32, np.float64)), "Reward from trainer.evaluate is not float"
            assert isinstance(eval_metrics, dict), "Metrics from trainer.evaluate_for_validation is not dict"
            
            results.append({
                'file': test_file.name,
                'agent_type': 'rainbow', 
                'reward': reward,
                'portfolio_value': eval_metrics.get('portfolio_value', 0.0),
                'total_return': eval_metrics.get('total_return', 0.0),
                'sharpe_ratio': eval_metrics.get('sharpe_ratio', np.nan),
                'max_drawdown': eval_metrics.get('max_drawdown', np.nan),
                'win_rate': eval_metrics.get('win_rate', np.nan),
                'transaction_costs': eval_metrics.get('transaction_costs', 0.0)
            })
            test_env.close()
        except Exception as e:
            logger.error(f"Error during evaluation of {test_file.name}: {e}", exc_info=True)
            # Append placeholder for failed evaluation
            results.append({
                'file': test_file.name,
                'agent_type': 'rainbow',
                'reward': -np.inf,
                'portfolio_value': 0,
                'total_return': 0,
                'sharpe_ratio': np.nan,
                'max_drawdown': np.nan,
                'win_rate': np.nan,
                'transaction_costs': 0
            })
    
    # --- Log Evaluation Summary --- 
    successful_results = [r for r in results if r['reward'] > -np.inf]
    if successful_results:
        metrics_summary = {
            'avg_reward': np.mean([r['reward'] for r in successful_results]),
            'avg_portfolio': np.mean([r['portfolio_value'] for r in successful_results]),
            'avg_return': np.mean([r['total_return'] for r in successful_results]),
            'avg_sharpe': np.nanmean([r.get('sharpe_ratio', np.nan) for r in successful_results]), # Use nanmean
            'avg_drawdown': np.nanmean([r.get('max_drawdown', np.nan) for r in successful_results]), # Use nanmean
            'avg_win_rate': np.nanmean([r.get('win_rate', np.nan) for r in successful_results]),     # Use nanmean
            'avg_transaction_costs': np.mean([r.get('transaction_costs', 0.0) for r in successful_results])
        }
        logger.info(f"=============================================")
        logger.info(f"TEST EVALUATION SUMMARY (RAINBOW) ({len(successful_results)}/{len(test_files)} files successful)")
        logger.info(f"Average Reward: {metrics_summary['avg_reward']:.2f}")
        logger.info(f"Average Final Portfolio: ${metrics_summary['avg_portfolio']:.2f}")
        logger.info(f"Average Return: {metrics_summary['avg_return']:.2f}%")
        logger.info(f"Average Sharpe Ratio: {metrics_summary['avg_sharpe']:.4f}")
        logger.info(f"Average Max Drawdown: {metrics_summary['avg_drawdown']:.4f}")
        logger.info(f"Average Win Rate: {metrics_summary['avg_win_rate']:.4f}")
        logger.info(f"Average Transaction Costs: ${metrics_summary['avg_transaction_costs']:.2f}")
        logger.info(f"=============================================")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(model_dir) / f"test_results_rainbow_{timestamp}.json"
        
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            return obj

        json_results = {
            'summary': convert_numpy_types(metrics_summary),
            'detailed_results': convert_numpy_types(results)
        }
        
        try:
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=4, allow_nan=True) # Allow NaN for ratios
            logger.info(f"Detailed test results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Error saving test results to JSON: {e}")

        # Print detailed results
        logger.info("Results per test file:")
        for r in results:
             status = "Success" if r['reward'] > -np.inf else "Failed"
             logger.info(f"  {r['file']} ({status}):")
             logger.info(f"    Reward: {r['reward']:.2f}")
             logger.info(f"    Portfolio: ${r['portfolio_value']:.2f}")
             logger.info(f"    Return: {r['total_return']:.2f}%")
             logger.info(f"    Sharpe: {r.get('sharpe_ratio', 'N/A'):.4f}")
             logger.info(f"    Max Drawdown: {r.get('max_drawdown', 'N/A'):.4f}")
             logger.info(f"    Win Rate: {r.get('win_rate', 'N/A'):.4f}")
             logger.info(f"    Transaction Costs: ${r.get('transaction_costs', 'N/A'):.2f}")
    else:
        logger.warning("No successful evaluations on test data.")

    return results 