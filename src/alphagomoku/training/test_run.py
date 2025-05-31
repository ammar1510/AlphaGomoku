import jax
import logging
import sys # For stdout

# --- Configure Logging ---
# Basic configuration, force=True ensures it applies even if other loggers are set up.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # Explicitly set to stdout to match print behavior
    force=True
)
logger = logging.getLogger(__name__)

# Set higher level for verbose libraries like absl (used by Orbax)
# This should be done after basicConfig if it affects root, or on specific loggers
logging.getLogger("absl").setLevel(logging.WARNING)

# --- Now import other libraries ---
import jax.numpy as jnp
import optax
import wandb
import orbax.checkpoint as ocp
import flax.linen as nn
from flax.training import train_state
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Tuple, Optional
import time
import os
from functools import partial
import hydra.utils


# --- Your main code would go here ---
logger.info("This is a test info message from the main module.")
print("This is a test print message.")

if __name__ == '__main__':
    logger.info(f"Script {__file__} started on process {jax.process_index()}.")
    # Add any other test logic here if needed
    logger.info("Script finished.")


# --- Initialize JAX Distributed System (before other heavy imports if it logs) ---
jax.distributed.initialize()
logger.info(f"JAX distributed system initialized on process {jax.process_index()} of {jax.process_count()}.")
