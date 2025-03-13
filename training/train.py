import os
import time
import argparse
import yaml
from functools import partial
from typing import Dict, List, Tuple, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
import flax.linen as nn
from flax.training import checkpoints, train_state
import orbax.checkpoint

from env.functional_gomoku import (
    init_env, 
    reset_env, 
    step_env, 
    get_action_mask, 
    sample_action, 
)

from models.actor_critic import ActorCritic

# Constants
NUM_HISTORY = 1  # By default, use just the current state (not history)

# Default configuration
DEFAULT_CONFIG = {
    # Environment parameters
    "board_size": 15,
    
    # Model parameters
    "num_channels": 1,
    
    # Training parameters
    "batch_size": 128,
    "learning_rate": 0.001,
    "discount": 0.99,
    "self_play_games": 512,
    "epochs_per_iteration": 5,
    "total_iterations": 100,
    
    # Evaluation parameters
    "eval_frequency": 1,
    "eval_games": 128,
    
    # Checkpointing
    "save_frequency": 5,
    "checkpoint_dir": "checkpoints",
    
    # Miscellaneous
    "seed": 42
}

def load_config(config_path):
    """
    Load configuration from a YAML file with fallback to defaults.
    
    Args:
        config_path: Path to the YAML config file
        
    Returns:
        config: Dictionary with configuration parameters
    """
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # Load and merge values from YAML file
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                config.update(yaml_config)
                
    return config

def create_train_state(rng, model, learning_rate, board_size):
    """Create initial training state."""
    # Provide only the current board state
    dummy_input = jnp.zeros((1, board_size, board_size))
    params = model.init(rng, dummy_input)
    
    tx = optax.adam(learning_rate)
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


def get_observations(board: jnp.ndarray, current_player: jnp.ndarray) -> jnp.ndarray:
    """
    Create observation for model input.
    
    Args:
        board: Current board state (num_boards, board_size, board_size)
        current_player: Current player array (1 for black, -1 for white)
        
    Returns:
        observations: Board states from current player's perspective
    """
    # Normalize from perspective of current player
    current_player_reshaped = current_player[:, jnp.newaxis, jnp.newaxis]
    observations = board * current_player_reshaped
    
    return observations


def sample_action_from_policy(env_state, policy_logits, rng, temperature=1.0):
    """
    Sample actions from policy logits with temperature and mask for valid actions.
    
    Args:
        env_state: Environment state
        policy_logits: Policy logits from model (batch_size, board_size, board_size)
        rng: JAX random key
        temperature: Temperature for sampling (higher = more exploration)
        
    Returns:
        actions: Selected actions (batch_size, 2)
        new_rng: Updated random key
    """
    board_size = env_state['board_size']
    num_boards = env_state['num_boards']
    
    # Get action mask (True for valid actions)
    action_mask = get_action_mask(env_state)
    
    # Apply temperature and mask invalid actions (set to very negative)
    scaled_logits = policy_logits / temperature
    masked_logits = jnp.where(action_mask, scaled_logits, -jnp.inf)
    
    # Convert to probabilities (assuming masked_logits is a proper array)
    # This requires policy_logits to be properly passed as an array
    flat_logits = masked_logits.reshape(num_boards, -1)
    policy_probs = jax.nn.softmax(flat_logits, axis=1)
    
    # Sample from categorical distribution
    rng, subkey = jax.random.split(rng)
    flat_actions = jax.random.categorical(subkey, policy_probs)
    
    # Convert to 2D coordinates
    rows = flat_actions // board_size
    cols = flat_actions % board_size
    actions = jnp.stack([rows, cols], axis=1)
    
    return actions, rng


def compute_loss(params, apply_fn, batch, is_training=True):
    """
    Compute policy and value losses for training.
    
    Args:
        params: Model parameters
        apply_fn: Model apply function
        batch: Training batch
        is_training: Whether in training mode
        
    Returns:
        total_loss: Combined loss
        metrics: Dictionary of metrics
    """
    observations, target_policies, target_values = batch
    
    # Forward pass
    predicted_policies, predicted_values = apply_fn(params, observations)
    
    # Policy loss (cross-entropy loss)
    batch_size = observations.shape[0]
    board_size = observations.shape[1]
    
    # Flatten the policy predictions and targets
    flat_predicted_policies = predicted_policies.reshape(batch_size, -1)
    flat_target_policies = target_policies
    
    policy_loss = optax.softmax_cross_entropy(flat_predicted_policies, flat_target_policies).mean()
    
    # Value loss (MSE)
    value_loss = optax.l2_loss(predicted_values, target_values).mean()
    
    # Combine losses
    total_loss = policy_loss + value_loss
    
    # Calculate metrics
    metrics = {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'predicted_value_mean': predicted_values.mean(),
        'target_value_mean': target_values.mean(),
    }
    
    return total_loss, metrics


@partial(jax.jit, static_argnums=(3,))
def train_step(state, batch, rng, is_training=True):
    """Single training step."""
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params, state.apply_fn, batch, is_training)
    
    # Update parameters
    new_state = state.apply_gradients(grads=grads)
    
    return new_state, metrics, loss


@partial(jax.jit, static_argnums=(0, 1, 2))
def run_self_play_episode(board_size, num_boards, num_games, state, rng, temperature_schedule):
    """
    Run a batch of self-play episodes using the current model.
    
    Args:
        board_size: Board size
        num_boards: Number of parallel boards
        num_games: Number of games to play
        state: Training state with model parameters
        rng: JAX random key
        temperature_schedule: Function mapping step to temperature
        
    Returns:
        trajectories: List of game trajectories
        final_rewards: Final rewards for each game
        new_rng: Updated random key
    """
    # Initialize environment
    rng, env_rng = jax.random.split(rng)
    env_state = init_env(env_rng, board_size, num_boards)
    env_state, observations = reset_env(env_state)
    
    # Initialize rewards
    total_rewards = jnp.zeros((num_boards,))
    step_count = 0
    
    # Game loop
    def game_loop_cond(state_tuple):
        _, env_state, _, step_count, _ = state_tuple
        # Continue until all games are done or we reach max steps
        return (~jnp.all(env_state['dones'])) & (step_count < board_size * board_size)
    
    def game_loop_body(state_tuple):
        rng, env_state, total_rewards, step_count, game_history = state_tuple
        
        # Get current temperature based on step count
        temperature = temperature_schedule(step_count)
        
        # Get observations from current board state
        obs = get_observations(env_state['board'], env_state['current_player'])
        
        # Get policy and value from model
        rng, inference_rng = jax.random.split(rng)
        policy_logits, values = state.apply_fn(state.params, obs)
        
        # Ensure policy_logits is a valid array before passing to sample_action_from_policy
        if isinstance(policy_logits, tuple):
            # If it's a tuple (shouldn't happen with the right model), take first element
            policy_logits = policy_logits[0]
        
        # Sample action from policy
        actions, rng = sample_action_from_policy(env_state, policy_logits, inference_rng, temperature)
        
        # Store policy probabilities (flattened)
        # Again ensure it's a valid array before reshaping
        if isinstance(policy_logits, tuple):
            flat_logits = policy_logits[0].reshape(num_boards, -1)
        else:
            flat_logits = policy_logits.reshape(num_boards, -1)
            
        policy_probs = jax.nn.softmax(flat_logits, axis=1)
        
        # Take environment step
        next_env_state, next_obs, rewards, dones = step_env(env_state, actions)
        
        # Update histories
        game_history['boards'].append(env_state['board'])
        game_history['actions'].append(actions)
        game_history['policies'].append(policy_probs)
        game_history['rewards'].append(rewards)
        game_history['players'].append(env_state['current_player'])
        
        # Update total rewards
        next_total_rewards = total_rewards + rewards
        
        return rng, next_env_state, next_total_rewards, step_count + 1, game_history
    
    # Initialize game history
    game_history = {
        'boards': [env_state['board']],
        'actions': [],
        'policies': [],
        'rewards': [],
        'players': [env_state['current_player']],
    }
    
    # Run game loop
    rng, final_env_state, final_rewards, final_step_count, final_history = jax.lax.while_loop(
        game_loop_cond,
        game_loop_body,
        (rng, env_state, total_rewards, step_count, game_history)
    )
    
    return final_history, final_rewards, rng


def process_game_history(game_history, final_rewards, discount=0.99):
    """
    Process game history to create training examples.
    
    Args:
        game_history: Dictionary of game history
        final_rewards: Final rewards for each game
        discount: Reward discount factor
        
    Returns:
        observations: Input observations
        target_policies: Target policy distributions
        target_values: Target value predictions
    """
    # Extract history components
    boards = game_history['boards']
    actions = game_history['actions']
    policies = game_history['policies']
    rewards = game_history['rewards']
    players = game_history['players']
    
    # Number of steps and boards
    num_steps = len(actions)
    num_boards = boards[0].shape[0]
    board_size = boards[0].shape[1]
    
    # Initialize storage for training examples
    all_observations = []
    all_target_policies = []
    all_target_values = []
    
    # Calculate returns and create training examples for each step
    for t in range(num_steps):
        # Get observations at time t
        board_t = boards[t]
        player_t = players[t]
        
        # Normalize from perspective of current player
        player_reshaped = player_t[:, jnp.newaxis, jnp.newaxis]
        obs = board_t * player_reshaped
        
        # Get target policy (actual action taken)
        action_t = actions[t]
        policy_targets = jnp.zeros((num_boards, board_size * board_size))
        indices = jnp.arange(num_boards)
        flat_actions = action_t[:, 0] * board_size + action_t[:, 1]
        policy_targets = policy_targets.at[indices, flat_actions].set(1.0)
        
        # Calculate bootstrap returns
        # For terminal states, use the final reward
        # For non-terminal states, use TD(0) bootstrap
        if t == num_steps - 1:
            # Final step - use actual returns
            returns = final_rewards
        else:
            # Intermediate step - could bootstrap from values, but using actual returns for now
            returns = final_rewards
        
        # Flip returns for player perspective
        returns = returns * player_t
        
        # Store training examples
        all_observations.append(obs)
        all_target_policies.append(policy_targets)
        all_target_values.append(returns)
    
    # Stack all examples
    stacked_observations = jnp.concatenate(all_observations, axis=0)
    stacked_policies = jnp.concatenate(all_target_policies, axis=0)
    stacked_values = jnp.concatenate(all_target_values, axis=0)
    
    return stacked_observations, stacked_policies, stacked_values


def evaluate_against_random(state, model, board_size=15, num_eval_games=128, seed=42):
    """
    Evaluate current model against random player.
    
    Args:
        state: Training state with model parameters
        model: ActorCritic model instance
        board_size: Size of the board
        num_eval_games: Number of evaluation games
        seed: Random seed
        
    Returns:
        win_rate: Win rate against random player
    """
    # Initialize environment
    rng = jax.random.PRNGKey(seed)
    env_state = init_env(rng, board_size, num_eval_games)
    env_state, observations = reset_env(env_state)
    
    total_rewards = jnp.zeros((num_eval_games,))
    
    # Game loop
    while not jnp.all(env_state['dones']):
        # Current player's turn
        current_player = env_state['current_player']
        
        # If black's turn (1), use model; if white's turn (-1), use random
        is_model_turn = current_player == 1
        
        # Get observations from current board state when it's model's turn
        obs = get_observations(env_state['board'], current_player)
        
        # For model's turn, get policy from model
        policy_logits, _ = state.apply_fn(state.params, obs)
        
        # For random's turn, generate random policy
        rng, action_rng = jax.random.split(rng)
        
        # Combine: model policy for black, random for white
        if jnp.all(is_model_turn):
            actions, rng = sample_action_from_policy(env_state, policy_logits, action_rng, temperature=0.1)
            env_state_updated = env_state
        else:
            actions, env_state_updated = sample_action(env_state, env_state['rng'])
            
        # Update RNG in environment state
        env_state = env_state_updated
        
        # Take environment step
        env_state, obs, rewards, dones = step_env(env_state, actions)
        
        # Update rewards
        total_rewards = total_rewards + rewards
    
    # Calculate win rate (when model as black player (1) wins)
    black_wins = (env_state['winners'] == 1)
    win_rate = jnp.mean(black_wins)
    
    return win_rate


def main(config):
    print("Starting Gomoku training with the following parameters:")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    
    # Initialize random key
    rng = jax.random.PRNGKey(config["seed"])
    
    # Create ActorCritic model and optimizer
    rng, init_rng = jax.random.split(rng)
    model = ActorCritic(board_size=config["board_size"], channels=config["num_channels"])
    
    state = create_train_state(
        init_rng, model, config["learning_rate"], config["board_size"]
    )
    
    # Temperature schedule for self-play (start high, anneal down)
    def temperature_schedule(step):
        max_steps = config["board_size"] * config["board_size"]
        temp = 1.0 - 0.9 * min(1.0, step / (max_steps * 0.5))
        return jnp.maximum(0.1, temp)
    
    # Training loop
    best_eval_win_rate = 0.0
    
    for iteration in range(config["total_iterations"]):
        start_time = time.time()
        
        # Self-play phase
        print(f"Iteration {iteration+1}/{config['total_iterations']} - Starting self-play...")
        rng, self_play_rng = jax.random.split(rng)
        
        game_history, final_rewards, rng = run_self_play_episode(
            config["board_size"],
            config["batch_size"],
            config["self_play_games"],
            state,
            self_play_rng,
            temperature_schedule
        )
        
        # Process game history to create training examples
        observations, target_policies, target_values = process_game_history(
            game_history, final_rewards, discount=config["discount"]
        )
        
        # Learning phase
        print(f"Iteration {iteration+1}/{config['total_iterations']} - Starting learning...")
        
        # Shuffle and create batches
        num_examples = observations.shape[0]
        indices = jnp.arange(num_examples)
        rng, shuffle_rng = jax.random.split(rng)
        shuffled_indices = jax.random.permutation(shuffle_rng, indices)
        
        # Training loop
        for epoch in range(config["epochs_per_iteration"]):
            epoch_metrics = []
            
            for i in range(0, num_examples, config["batch_size"]):
                batch_indices = shuffled_indices[i:i+config["batch_size"]]
                batch = (
                    observations[batch_indices],
                    target_policies[batch_indices],
                    target_values[batch_indices]
                )
                
                # Training step
                rng, train_rng = jax.random.split(rng)
                state, metrics, loss = train_step(state, batch, train_rng)
                epoch_metrics.append(metrics)
            
            # Log epoch results
            avg_metrics = {k: jnp.mean(jnp.array([m[k] for m in epoch_metrics])) for k in epoch_metrics[0]}
            print(f"Epoch {epoch+1}/{config['epochs_per_iteration']}, Loss: {avg_metrics['total_loss']:.4f}, "
                  f"Policy Loss: {avg_metrics['policy_loss']:.4f}, Value Loss: {avg_metrics['value_loss']:.4f}")
        
        # Evaluation phase
        if (iteration + 1) % config["eval_frequency"] == 0:
            print(f"Iteration {iteration+1}/{config['total_iterations']} - Evaluating...")
            eval_win_rate = evaluate_against_random(
                state, model, board_size=config["board_size"], num_eval_games=config["eval_games"]
            )
            print(f"Evaluation win rate against random: {eval_win_rate:.2%}")
            
            # Save checkpoint if improved
            if eval_win_rate > best_eval_win_rate:
                best_eval_win_rate = eval_win_rate
                checkpoint_path = os.path.join(config["checkpoint_dir"], f"best_model")
                checkpoints.save_checkpoint(checkpoint_path, state, step=iteration, overwrite=True)
                print(f"Saved new best model with win rate {best_eval_win_rate:.2%}")
        
        # Always save latest model
        if (iteration + 1) % config["save_frequency"] == 0:
            checkpoint_path = os.path.join(config["checkpoint_dir"], f"model_{iteration+1}")
            checkpoints.save_checkpoint(checkpoint_path, state, step=iteration, overwrite=True)
            print(f"Saved model at iteration {iteration+1}")
        
        iteration_time = time.time() - start_time
        print(f"Iteration {iteration+1} completed in {iteration_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gomoku AlphaZero-style model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run training
    main(config)
