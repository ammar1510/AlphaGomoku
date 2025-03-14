import logging
import os
import random
import glob
import time
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
        import sys

        if len(sys.argv) > 1 and sys.argv[1].endswith(".yaml"):
            config_path = sys.argv[1]
            logging.info(f"Using configuration from command line: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Log the loaded configuration
        logging.info(f"Loaded configuration from {config_path}:")
        for key, value in config.items():
            logging.info(f"  {key}: {value}")

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


def save_checkpoint(params, opt_state, checkpoint_dir):
    """
    Save network checkpoint using a timestamp-based naming scheme.

    Args:
      params: network parameters.
      opt_state: optimizer state (optional).
      checkpoint_dir: Directory for checkpoints.

    Returns: None.
    """
    checkpoint = {"params": params, "opt_state": opt_state}

    checkpoint_path = get_checkpoint_filename(checkpoint_dir)
    
    with open(checkpoint_path, "wb") as f:
        f.write(serialization.to_bytes(checkpoint))
    
    logging.info(
        f"Model checkpoint saved to {checkpoint_path}"
    )
    
    # Manage the checkpoint pool
    manage_checkpoint_pool(checkpoint_dir, max_checkpoints=10)


def load_checkpoint(checkpoint_path):
    """
    Load network checkpoint.

    Args:
      checkpoint_path: File path for checkpoint.

    Returns:
      A tuple (params, opt_state) if the checkpoint exists, 
      or (None, None) if not.
    """
    try:
        with open(checkpoint_path, "rb") as f:
            data = f.read()
        
        # Create template with opt_state field
        template = {"params": None, "opt_state": None}
        checkpoint = serialization.from_bytes(template, data)
        
        logging.info(
            f"Model checkpoint loaded from {checkpoint_path}"
        )
        
        return checkpoint["params"], checkpoint["opt_state"]
    except FileNotFoundError:
        logging.info(
            f"Model checkpoint not found at {checkpoint_path}."
        )
        return None, None


def select_training_checkpoints(checkpoint_dir, rng_key=None):
    """
    Select two different checkpoint files from the checkpoint directory.
    
    Args:
        checkpoint_dir: Directory for checkpoints.
        rng_key: JAX random key for selection.
        
    Returns:
        tuple: (black_params, black_opt_state, white_params, white_opt_state)
              If no checkpoints or only one checkpoint exists, returns appropriate None values.
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        logging.info("No checkpoints found. Both models will start from scratch.")
        return None, None, None, None
    
    if len(checkpoints) == 1:
        # Only one checkpoint exists, one model will use it, the other will start fresh
        params, opt_state = load_checkpoint(checkpoints[0])
        logging.info(f"Only one checkpoint found. Black model will use it, white model will start from scratch.")
        return params, opt_state, None, None
    
    # Select two different checkpoints
    if rng_key is not None:
        rng_key, subkey = jr.split(rng_key)
        # Shuffle the checkpoints using JAX random
        indices = jr.permutation(subkey, len(checkpoints))
        black_checkpoint = checkpoints[indices[0]]
        white_checkpoint = checkpoints[indices[1]]
    else:
        # Use Python's random if no JAX key provided
        selected_checkpoints = random.sample(checkpoints, 2)
        black_checkpoint, white_checkpoint = selected_checkpoints
    
    # Load the checkpoints
    black_params, black_opt_state = load_checkpoint(black_checkpoint)
    white_params, white_opt_state = load_checkpoint(white_checkpoint)
    
    logging.info(f"Loaded different checkpoints for black and white models.")
    
    return black_params, black_opt_state, white_params, white_opt_state


