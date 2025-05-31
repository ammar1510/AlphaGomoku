import jax
import logging
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

# --- Configure Logging ---
# Get a logger for this module
logger = logging.getLogger(__name__)
# Basic configuration (can be enhanced)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Set higher level for verbose libraries like absl (used by Orbax)
logging.getLogger("absl").setLevel(logging.WARNING)


jax.distributed.initialize()
logger.info(f"JAX distributed system initialized on process {jax.process_index()}.")