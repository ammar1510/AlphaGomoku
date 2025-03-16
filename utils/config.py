import logging
import os
import random
import glob
import time
import sys
from flax import serialization
import yaml
import jax.random as jr


def load_config(config_path="cfg/train.yaml")->dict:
    """
    Load configuration from YAML file and validate required parameters.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
        yaml.YAMLError: If the YAML file is malformed.
        ValueError: If any required parameter is missing.
    """
    try:
        # Check if config path was provided via command line
        if len(sys.argv) > 1 and sys.argv[1].endswith(".yaml"):
            config_path = sys.argv[1]

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Required parameters that must be present
        required_params = [
            "board_size",
            "num_boards",
            "discount",
            "total_iterations",
            "learning_rate",
            "weight_decay",
            "render",
            "checkpoint_dir",
            "save_frequency",
            "grad_clip_norm",
            "seed",
            "initial_entropy_coef",
            "min_entropy_coef",
            "entropy_decay_steps",
        ]

        # Check for missing parameters
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            error_msg = (
                f"Missing required parameters in config: {', '.join(missing_params)}"
            )
            logging.error(error_msg)
            raise ValueError(error_msg)

        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}")
        raise


def log_config(config, config_path="cfg/train.yaml"):
    """
    Log the configuration parameters.
    
    Args:
        config: Configuration dictionary to log.
        config_path: Path to the configuration file that was loaded.
        
    Returns:
        None
    """
    # Log the configuration source
    if len(sys.argv) > 1 and sys.argv[1].endswith(".yaml"):
        logging.info(f"Using configuration from command line: {config_path}")
    else:
        logging.info(f"Using default configuration: {config_path}")
    
    # Log the loaded configuration
    logging.info(f"Loaded configuration:")
    for key, value in config.items():
        logging.info(f"  {key}: {value}")


def get_checkpoint_path(config)->str:
    """
    Get the checkpoint directory based on the configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        str: Path to the checkpoint directory.
    """
    board_size = config["board_size"]
    checkpoint_dir = os.path.join(
        config["checkpoint_dir"], f"{board_size}x{board_size}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def get_checkpoint_filename(checkpoint_dir)->str:
    """
    Get the checkpoint filename using a timestamp-based naming scheme.

    Args:
        checkpoint_dir: Directory for checkpoints.

    Returns:
        str: Checkpoint filename with timestamp.
    """
    timestamp = int(time.time())
    filename = f"model_{timestamp}.pkl"
    return os.path.join(checkpoint_dir, filename)


def list_checkpoints(checkpoint_dir)->list[str]:
    """
    List all available checkpoints.

    Args:
        checkpoint_dir: Directory for checkpoints.

    Returns:
        list: List of checkpoint filenames sorted by timestamp (newest first).
    """
    pattern = os.path.join(checkpoint_dir, "model_*.pkl")
    checkpoints = glob.glob(pattern)
    
    # Sort by timestamp (extract from filename)
    def get_timestamp(filepath):
        filename = os.path.basename(filepath)
        # Extract timestamp which is the numeric part after "model_"
        parts = filename.split('_')
        if len(parts) >= 2:
            try:
                return int(parts[1].split('.')[0])
            except ValueError:
                return 0
        return 0
    
    # Sort newest first
    checkpoints.sort(key=get_timestamp, reverse=True)
    
    return checkpoints


def select_random_checkpoint(checkpoint_dir)->str|None:
    """
    Select a random checkpoint from the pool.
    
    Args:
        checkpoint_dir: Directory for checkpoints.
        
    Returns:
        str or None: Path to a randomly selected checkpoint file, or None if no checkpoints exist
    """

    checkpoints = list_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        logging.info(f"No checkpoints available.")
        return None
    
    selected_checkpoint = random.choice(checkpoints)
    logging.info(f"Selected random checkpoint: {selected_checkpoint}")
    
    return selected_checkpoint


def manage_checkpoint_pool(checkpoint_dir, max_checkpoints=10):
    """
    Manage the pool of checkpoints, keeping only the specified maximum number.
    Removes the oldest checkpoints when the pool exceeds the maximum size.
    
    Args:
        checkpoint_dir: Directory for checkpoints.
        max_checkpoints: Maximum number of checkpoints to keep.
        
    Returns:
        None
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if len(checkpoints) > max_checkpoints:
        # checkpoints are sorted newest first, so remove the last entries
        to_remove = checkpoints[max_checkpoints:]
        for checkpoint in to_remove:
            try:
                os.remove(checkpoint)
                logging.info(f"Removed old checkpoint: {checkpoint}")
            except OSError as e:
                logging.error(f"Error removing checkpoint {checkpoint}: {e}")


def save_checkpoint(params, checkpoint_dir):
    """
    Save network checkpoint using a timestamp-based naming scheme.
    Only saves model parameters, not optimizer state.

    Args:
      params: network parameters to save.
      checkpoint_dir: Directory for checkpoints.

    Returns: None.
    """
    # Only save params, not opt_state
    checkpoint = {"params": params}

    checkpoint_path = get_checkpoint_filename(checkpoint_dir)
    
    with open(checkpoint_path, "wb") as f:
        f.write(serialization.to_bytes(checkpoint))
    
    logging.info(
        f"Model checkpoint saved to {checkpoint_path}"
    )
    time.sleep(0.3)
    
    # Manage the checkpoint pool
    manage_checkpoint_pool(checkpoint_dir, max_checkpoints=10)


def load_checkpoint(checkpoint_path):
    """
    Load network checkpoint.
    Only loads model parameters, not optimizer state.

    Args:
      checkpoint_path: File path for checkpoint.

    Returns:
      A tuple (params, None) if the checkpoint exists, 
      or (None, None) if not. The second element is always None
      since we no longer store optimizer state.
    """
    try:
        with open(checkpoint_path, "rb") as f:
            data = f.read()
        
        # Create template with only params field
        template = {"params": None}
        checkpoint = serialization.from_bytes(template, data)
        
        logging.info(
            f"Model checkpoint loaded from {checkpoint_path}"
        )
        assert checkpoint["params"] is not None
        
        return checkpoint["params"]
    except FileNotFoundError:
        logging.warning(
            f"Model checkpoint not found at {checkpoint_path}."
        )
        return None


def select_training_checkpoints(checkpoint_dir, rng_key=None):
    """
    Select two different checkpoint files from the checkpoint directory.
    
    Args:
        checkpoint_dir: Directory for checkpoints.
        rng_key: JAX random key for selection.
        
    Returns:
        tuple: (black_checkpoint_path, white_checkpoint_path)
              If no checkpoints exist, returns (None, None).
              If only one checkpoint exists, returns (path, None).
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        logging.info("No checkpoints found. Both models will start from scratch.")
        return None, None
    
    if len(checkpoints) == 1:
        return checkpoints[0], None
    
    if rng_key is not None:
        rng_key, subkey = jr.split(rng_key)
        indices = jr.permutation(subkey, len(checkpoints))
        black_checkpoint_path = checkpoints[indices[0]]
        white_checkpoint_path = checkpoints[indices[1]]
    else:
        selected_checkpoints = random.sample(checkpoints, 2)
        black_checkpoint_path, white_checkpoint_path = selected_checkpoints
    
    
    return black_checkpoint_path, white_checkpoint_path


