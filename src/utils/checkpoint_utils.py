import os
import torch
import logging
import numpy as np
from typing import Optional, Dict, Any

logger = logging.getLogger("CheckpointUtils")


def find_latest_checkpoint(
    model_dir: str = "models", model_prefix: str = "checkpoint_rainbow"
) -> Optional[str]:
    """Finds the latest checkpoint file based on timestamp or name."""
    # Simple approach: look for specific latest/best files first
    latest_path = os.path.join(model_dir, f"{model_prefix}_latest.pt")
    if os.path.exists(latest_path):
        logger.info(f"Found latest checkpoint: {latest_path}")
        return latest_path

    best_path = os.path.join(model_dir, f"{model_prefix}_best.pt")
    if os.path.exists(best_path):
        logger.warning(
            f"Latest checkpoint not found, using best checkpoint: {best_path}"
        )
        return best_path

    logger.warning("No suitable checkpoint file found.")
    return None


def load_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Loads a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        The loaded checkpoint dictionary, or None if loading fails.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint path does not exist: {checkpoint_path}")
        return None

    logger.info(f"Attempting to load checkpoint: {checkpoint_path}")
    try:
        # Load to CPU first to avoid GPU memory issues if loading on different device
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Basic validation of checkpoint structure
        required_keys = [
            "episode",
            "total_train_steps",
            "network_state_dict",
            "optimizer_state_dict",
            "best_validation_metric",
            "target_network_state_dict",
            "agent_total_steps",
            "early_stopping_counter",
            "agent_config",
        ]
        if not all(key in checkpoint for key in required_keys):
            logger.error(
                f"Checkpoint {checkpoint_path} is missing required keys. Cannot resume."
            )
            return None

        logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
        logger.info(f"  Resuming from Episode: {checkpoint.get('episode', 'N/A')}")
        logger.info(
            f"  Resuming from Total Steps: {checkpoint.get('total_train_steps', 'N/A')}"
        )
        logger.info(
            f"  Previous Best Validation Score: {checkpoint.get('best_validation_metric', -np.inf):.4f}"
        )
        return checkpoint
    except Exception as e:
        logger.error(
            f"Failed to load checkpoint from {checkpoint_path}: {e}.", exc_info=True
        )
        return None
