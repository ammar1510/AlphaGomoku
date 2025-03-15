import logging
import os
import sys
from datetime import datetime


def setup_logging(log_dir="logs", filename=None, level=logging.INFO):
    """
    Set up logging to both console and file.
    
    Args:
        log_dir (str): Directory to store log files
        filename (str, optional): Specific log filename. If None, a timestamp-based name will be used.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG)
        
    Returns:
        str: Path to the log file
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate a timestamp-based filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gomoku_{timestamp}.log"
    
    log_path = os.path.join(log_dir, filename)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove any existing handlers to avoid duplicate logging
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler with formatting
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    logging.info(f"Logging configured. Log file: {log_path}")
    
    return log_path


def get_experiment_logger(experiment_name, log_dir="logs", level=logging.INFO):
    """
    Create a specific logger for an experiment.
    
    Args:
        experiment_name (str): Name of the experiment
        log_dir (str): Directory to store log files
        level (int): Logging level
        
    Returns:
        logging.Logger: Configured logger for the experiment
        str: Path to the log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.log"
    
    log_path = setup_logging(log_dir, filename, level)
    
    # Get a named logger for the experiment
    logger = logging.getLogger(experiment_name)
    logger.setLevel(level)
    
    return logger, log_path 