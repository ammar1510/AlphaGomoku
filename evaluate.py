import logging
import time
import yaml
import os
import sys
import pygame

import jax
import jax.numpy as jnp

from env.functional_gomoku import (
    get_action_mask,
    init_env,
    reset_env,
    step_env,
)
from env.renderer import GomokuRenderer
from models.actor_critic import ActorCritic
from utils.config import (
    load_config,
    get_checkpoint_path,
    select_training_checkpoints,
    load_checkpoint,
)
from utils.logging_utils import setup_logging

# We'll replace this with our new logging setup
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )


def load_eval_config(config_path="cfg/eval.yaml"):
    """
    Load evaluation configuration from YAML file without enforcing required parameters.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        dict: Configuration dictionary.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        # Log the loaded configuration
        logging.info(f"Loaded evaluation configuration from {config_path}:")
        for key, value in config.items():
            logging.info(f"  {key}: {value}")
            
        return config
    except FileNotFoundError:
        logging.error(f"Evaluation configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}")
        raise


def evaluate_models():
    """
    Load two random model checkpoints and have them play against each other
    with visualization of each move. Continues playing new games after each one finishes.
    Tracks win statistics across all games.
    """
    # Set up logging to both console and file
    log_filename = f"gomoku_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_dir="logs", filename=log_filename)
    
    logging.info("Starting model evaluation...")
    
    # Load evaluation config using our custom function
    eval_config = load_eval_config("cfg/eval.yaml")
    
    # Load training config for model parameters
    train_config = load_config(eval_config["train_config_path"])
    
    # Override seed if provided in eval config
    if "seed" in eval_config:
        train_config["seed"] = eval_config["seed"]
    
    # Initialize random state
    rng = jax.random.PRNGKey(train_config["seed"])
    rng, init_key, model_select_key = jax.random.split(rng, 3)
    
    board_size = train_config["board_size"]
    
    # Get the checkpoint directory
    checkpoint_dir = get_checkpoint_path(train_config)
    
    # Initialize models
    black_actor_critic = ActorCritic(board_size=board_size)
    white_actor_critic = ActorCritic(board_size=board_size)
    
    # Select two random checkpoints
    black_checkpoint_path, white_checkpoint_path = select_training_checkpoints(
        checkpoint_dir, model_select_key
    )
    
    # Load parameters from the selected checkpoints
    black_params = load_checkpoint(black_checkpoint_path)
    white_params = load_checkpoint(white_checkpoint_path)
    
    if black_params is None or white_params is None:
        logging.error("Couldn't load at least one model checkpoint. Make sure you have trained models.")
        return
    
    logging.info("Loaded both black and white model checkpoints.")
    
    # Initialize stats counters
    game_count = 0
    black_wins = 0
    white_wins = 0
    draws = 0
    
    # Initialize renderer
    renderer = GomokuRenderer(board_size, cell_size=eval_config["cell_size"])
    
    # Create a flag to track if pygame is running
    pygame_running = True
    
    # Main game loop - continue until user interrupts
    try:
        while pygame_running:
            game_count += 1
            logging.info(f"\n===== Starting Game #{game_count} =====")
            
            # Initialize environment for new game
            rng, game_key = jax.random.split(rng)
            env = init_env(game_key, board_size, num_boards=1)
            env, obs = reset_env(env)
            
            # Render initial empty board
            renderer.render_board(env["board"][0])
            
            # Game loop for a single game
            done = False
            move_count = 0
            
            while not done and pygame_running:
                # Process pygame events to handle window closure
                try:
                    renderer.process_events()
                except pygame.error:
                    # Pygame window was closed
                    logging.info("Window closed by user. Exiting.")
                    pygame_running = False
                    break
                
                # Determine which player's turn it is
                is_black_turn = (env["current_player"][0] == 1)
                
                # Select actor-critic model based on current player
                if is_black_turn:
                    actor_critic = black_actor_critic
                    model_params = black_params
                    player_name = "Black"
                else:
                    actor_critic = white_actor_critic
                    model_params = white_params
                    player_name = "White"
                
                # Get action mask for valid moves
                action_mask = get_action_mask(env)
                
                # Forward pass through the model
                policy_logits, value = actor_critic.apply(model_params, obs)
                
                # Apply action mask to logits
                logits = jnp.where(action_mask, policy_logits, -jnp.inf)
                
                # Sample action
                rng, action_key = jax.random.split(rng)
                action = actor_critic.sample_action(logits, action_key)
                
                # Log the move
                row, col = action[0]
                logging.info(f"Move {move_count + 1}: {player_name} places at ({row}, {col})")
                
                # Take the step in the environment
                env, obs, rewards, dones = step_env(env, action)
                
                # Render the board
                renderer.render_board(env["board"][0])
                
                # Pause for visualization
                time.sleep(eval_config["move_delay"])
                
                # Check if game is over
                done = env["dones"][0]
                move_count += 1
                
                # Add a forced exit condition to prevent infinite games
                if move_count >= board_size * board_size:
                    logging.info("Maximum number of moves reached. Game ends in a draw.")
                    break
            
            # Skip stats if pygame was closed
            if not pygame_running:
                break
                
            # Record game result
            winner = env["winners"][0]
            if winner == 1:
                black_wins += 1
                logging.info("Black wins!")
            elif winner == -1:
                white_wins += 1
                logging.info("White wins!")
            else:
                draws += 1
                logging.info("Game ends in a draw.")
            
            # Display statistics
            total_games = black_wins + white_wins + draws
            logging.info(f"\n===== Game Statistics =====")
            logging.info(f"Total Games: {total_games}")
            logging.info(f"Black Wins: {black_wins} ({black_wins/total_games*100:.1f}%)")
            logging.info(f"White Wins: {white_wins} ({white_wins/total_games*100:.1f}%)")
            logging.info(f"Draws: {draws} ({draws/total_games*100:.1f}%)")
            
            # Short pause between games
            time.sleep(2)
            
    except KeyboardInterrupt:
        logging.info("\n\nEvaluation interrupted by user.")
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
    finally:
        # Display final statistics if any games were completed
        total_games = black_wins + white_wins + draws
        if total_games > 0:
            logging.info(f"\n===== Final Statistics =====")
            logging.info(f"Total Games Played: {total_games}")
            logging.info(f"Black Wins: {black_wins} ({black_wins/total_games*100:.1f}%)")
            logging.info(f"White Wins: {white_wins} ({white_wins/total_games*100:.1f}%)")
            logging.info(f"Draws: {draws} ({draws/total_games*100:.1f}%)")
        
        # Clean up
        try:
            renderer.close()
        except:
            pass  # Ignore errors during cleanup


if __name__ == "__main__":
    evaluate_models() 