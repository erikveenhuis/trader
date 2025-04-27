import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict, Any
import os

# Import necessary components from other modules using relative imports
from .agent import RainbowDQNAgent
from .trainer import RainbowTrainerModule
from trading_env import TradingEnv, TradingEnvConfig
from .data import DataManager
from .utils.utils import set_seeds
from .metrics import PerformanceTracker, calculate_episode_score

logger = logging.getLogger("Evaluation")


def evaluate_on_test_data(
    agent: RainbowDQNAgent, trainer: RainbowTrainerModule, config: dict
) -> List[Dict[str, Any]]:
    """Evaluate the trained Rainbow agent on test data.

    Args:
        agent: The trained agent instance.
        trainer: The trainer instance (used for data_manager and evaluate method).
        config: The configuration dictionary containing agent, env, trainer, and run settings.

    Returns:
        A list of dictionaries containing evaluation results for each test file.
    """
    # --- Start: Input Asserts ---
    assert isinstance(
        agent, RainbowDQNAgent
    ), "evaluate_on_test_data requires a RainbowDQNAgent instance"
    assert isinstance(
        trainer, RainbowTrainerModule
    ), "evaluate_on_test_data requires a RainbowTrainerModule instance"
    assert isinstance(config, dict), "Config must be a dictionary"
    # --- End: Input Asserts ---

    # Extract necessary parameters from config
    agent_config = config["agent"]
    env_config = config["environment"]
    trainer_config = config["trainer"]
    run_config = config.get("run", {})
    seed = trainer_config["seed"]
    window_size = agent_config["window_size"]
    model_dir = run_config.get("model_dir", "models")

    set_seeds(seed)
    data_manager = trainer.data_manager
    test_files = data_manager.get_test_files()

    if not test_files:
        logger.warning("No test files found. Skipping test evaluation.")
        return []

    logger.info("=============================================")
    logger.info("EVALUATING RAINBOW MODEL ON TEST DATA")
    logger.info(f"Found {len(test_files)} test files")
    logger.info("=============================================")

    agent.set_training_mode(False)

    results = []
    all_episode_metrics = []
    for i, test_file in enumerate(test_files):
        logger.info(f"--- TEST FILE {i+1}/{len(test_files)}: {test_file.name} ---")
        try:
            # Add data_path to the env_config dictionary
            current_env_config = env_config.copy() # Avoid modifying original dict
            current_env_config['data_path'] = str(test_file)
            # Create config object first, now including data_path
            env_config_obj = TradingEnvConfig(**current_env_config)
            test_env = TradingEnv(
                # data_path=str(test_file), # Remove data_path, now in config
                config=env_config_obj # Pass the config object
            )
            assert isinstance(
                test_env, TradingEnv
            ), f"Failed to create test env for {test_file.name}"

            reward, eval_metrics, final_info = trainer._run_single_evaluation_episode(
                test_env
            )  # Use the renamed method

            assert isinstance(
                reward, (float, np.float32, np.float64)
            ), "Reward from trainer.evaluate is not float"
            assert isinstance(
                eval_metrics, dict
            ), "Metrics from trainer.evaluate_for_validation is not dict"

            results.append(
                {
                    "file": test_file.name,
                    "agent_type": "rainbow",
                    "reward": reward,
                    "portfolio_value": eval_metrics.get("portfolio_value", 0.0),
                    "total_return": eval_metrics.get("total_return", 0.0),
                    "sharpe_ratio": eval_metrics.get("sharpe_ratio", np.nan),
                    "max_drawdown": eval_metrics.get("max_drawdown", np.nan),
                    "transaction_costs": final_info.get("transaction_cost", 0.0),
                    "action_counts": eval_metrics.get("action_counts", {}),
                }
            )
            all_episode_metrics.append(eval_metrics)
            test_env.close()
        except Exception as e:
            logger.error(
                f"Error during evaluation of {test_file.name}: {e}", exc_info=True
            )
            # Append placeholder for failed evaluation
            results.append(
                {
                    "file": test_file.name,
                    "agent_type": "rainbow",
                    "reward": -np.inf,
                    "portfolio_value": 0,
                    "total_return": 0,
                    "sharpe_ratio": np.nan,
                    "max_drawdown": np.nan,
                    "transaction_costs": 0,
                    "action_counts": {},
                }
            )

    # --- Calculate Episode Scores and Average ---
    test_episode_scores = []
    if all_episode_metrics:
        for i, metrics_dict in enumerate(all_episode_metrics):
            try:
                score = calculate_episode_score(metrics_dict)
                assert 0.0 <= score <= 1.0, f"Test episode score out of range [0,1]: {score}"
                test_episode_scores.append(score)
                results[i]['episode_score'] = score
            except Exception as score_e:
                logger.error(f"Error calculating episode score for test file {results[i]['file']}: {score_e}")
                test_episode_scores.append(0.0)
                results[i]['episode_score'] = 0.0

    avg_test_episode_score = float(np.mean(test_episode_scores)) if test_episode_scores else 0.0
    # --- END SCORE CALCULATION ---

    # --- Log Evaluation Summary ---
    successful_results = [r for r in results if r["reward"] > -np.inf]
    if successful_results:
        portfolio_values = [r["portfolio_value"] for r in successful_results]
        rewards = [r["reward"] for r in successful_results]
        returns = [r["total_return"] for r in successful_results]
        tx_costs = [r["transaction_costs"] for r in successful_results]

        metrics_summary = {
            # Reward Stats
            "min_reward": np.min(rewards),
            "avg_reward": np.mean(rewards),
            "max_reward": np.max(rewards),
            # Portfolio Stats
            "min_portfolio": np.min(portfolio_values),
            "avg_portfolio": np.mean(portfolio_values),
            "max_portfolio": np.max(portfolio_values),
            # Return Stats
            "min_return": np.min(returns),
            "avg_return": np.mean(returns),
            "max_return": np.max(returns),
            # Other Averages
            "avg_sharpe": np.nanmean(
                [r.get("sharpe_ratio", np.nan) for r in successful_results]
            ),  # Use nanmean
            "avg_drawdown": np.nanmean(
                [r.get("max_drawdown", np.nan) for r in successful_results]
            ),  # Use nanmean
            # Transaction Cost Stats
            "min_transaction_costs": np.min(tx_costs),
            "avg_transaction_costs": np.mean(tx_costs),
            "max_transaction_costs": np.max(tx_costs),
            # --- ADDED: Average Test Episode Score ---
            "avg_test_episode_score": avg_test_episode_score,
            # --- END ADDED ---
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(model_dir) / f"test_results_rainbow_score_{avg_test_episode_score:.4f}_{timestamp}.json"

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
            "summary": convert_numpy_types(metrics_summary),
            "detailed_results": convert_numpy_types(results),
        }

        try:
            with open(results_file, "w") as f:
                json.dump(
                    json_results, f, indent=4, allow_nan=True
                )  # Allow NaN for ratios
            logger.info(f"Detailed test results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Error saving test results to JSON: {e}")

        # Print detailed results FIRST
        logger.info("=============================================")
        logger.info("RESULTS PER TEST FILE")
        logger.info("=============================================")
        for r in results:
            status = "Success" if r["reward"] > -np.inf else "Failed"
            logger.info(f"  {r['file']} ({status}):")
            logger.info(f"    Reward: {r['reward']:.2f}")
            logger.info(f"    Portfolio: ${r['portfolio_value']:.2f}")
            logger.info(f"    Return: {r['total_return']:.2f}%")
            logger.info(f"    Sharpe: {r.get('sharpe_ratio', 'N/A'):.4f}")
            logger.info(f"    Max Drawdown: {r.get('max_drawdown', 'N/A')*100:.2f}%")
            logger.info(f"    Action Counts: {r.get('action_counts', 'N/A')}")
            logger.info(
                f"    Transaction Costs: ${r.get('transaction_costs', 'N/A'):.2f}"
            )
            if 'episode_score' in r: # Log score if available
                logger.info(f"    Episode Score: {r['episode_score']:.4f}")

        # Now print the summary
        logger.info("=============================================")
        logger.info(
            f"TEST EVALUATION SUMMARY (RAINBOW) ({len(successful_results)}/{len(test_files)} files successful)"
        )
        # Log Min/Avg/Max for Reward, Portfolio, Return, TxCost
        logger.info(f"Total Reward (Min/Avg/Max): {metrics_summary['min_reward']:.2f} / {metrics_summary['avg_reward']:.2f} / {metrics_summary['max_reward']:.2f}")
        logger.info(f"Final Portfolio (Min/Avg/Max): ${metrics_summary['min_portfolio']:.2f} / ${metrics_summary['avg_portfolio']:.2f} / ${metrics_summary['max_portfolio']:.2f}")
        logger.info(f"Total Return % (Min/Avg/Max): {metrics_summary['min_return']:.2f}% / {metrics_summary['avg_return']:.2f}% / {metrics_summary['max_return']:.2f}%")
        logger.info(f"Transaction Costs (Min/Avg/Max): ${metrics_summary['min_transaction_costs']:.2f} / ${metrics_summary['avg_transaction_costs']:.2f} / ${metrics_summary['max_transaction_costs']:.2f}")
        # Log other averages
        logger.info(f"Average Sharpe Ratio: {metrics_summary['avg_sharpe']:.4f}")
        logger.info(f"Average Max Drawdown: {metrics_summary['avg_drawdown']*100:.2f}%")
        # --- ADDED: Log Average Test Score ---
        logger.info(f"Average Episode Score: {metrics_summary['avg_test_episode_score']:.4f}")
        # --- END ADDED ---
        logger.info("=============================================")

    else:
        logger.warning("No successful evaluations on test data.")

    return results
