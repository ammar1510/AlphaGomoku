import logging
from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax import jit, lax

from env.functional_gomoku import get_action_mask, init_env, reset_env, step_env
from models.actor_critic import ActorCritic
from utils.config import (
    get_checkpoint_path,
    load_config,
    log_config,
    save_checkpoint,
    select_training_checkpoints,
    load_checkpoint,
)
from utils.logging_utils import setup_logging

# We'll replace this with our new logging setup in the main function
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )


def init_train(config: dict):
    """
    Initialize training components.

    Args:
        config: Configuration dictionary.

    Returns:
        tuple: (env, black_actor_critic, black_params, black_opt_state, black_optimizer,
               white_actor_critic, white_params, white_opt_state, white_optimizer,
               checkpoint_dir, rng, board_size)
    """
    board_size = config["board_size"]
    num_boards = 1 if config["render"] else config["num_boards"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    grad_clip_norm = config["grad_clip_norm"]

    checkpoint_dir = get_checkpoint_path(config)

    rng = jax.random.PRNGKey(config["seed"])
    rng, init_key, black_key, white_key = jax.random.split(rng, 4)

    # Create separate models for black and white players
    black_actor_critic = ActorCritic(board_size=board_size)
    white_actor_critic = ActorCritic(board_size=board_size)

    # Initialize models with dummy input
    dummy_input = jnp.ones((1, board_size, board_size), dtype=jnp.float32)
    black_params = black_actor_critic.init(black_key, dummy_input)
    white_params = white_actor_critic.init(white_key, dummy_input)

    # Create optimizers for both models
    black_optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )
    
    white_optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )

    # Load different checkpoints for black and white models if available
    black_checkpoint_path, white_checkpoint_path = select_training_checkpoints(checkpoint_dir, init_key)
    
    # Update black model if checkpoint was loaded
    if black_checkpoint_path is not None:
        loaded_black_params = load_checkpoint(black_checkpoint_path)
        if loaded_black_params is not None:
            black_params = loaded_black_params
    
    # Always initialize optimizer state from scratch
    black_opt_state = black_optimizer.init(black_params)
    
    # Update white model if checkpoint was loaded
    if white_checkpoint_path is not None:
        loaded_white_params = load_checkpoint(white_checkpoint_path)
        if loaded_white_params is not None:
            white_params = loaded_white_params
    
    # Always initialize optimizer state from scratch
    white_opt_state = white_optimizer.init(white_params)
    
    # Log training initialization
    if black_checkpoint_path is not None or white_checkpoint_path is not None:
        logging.info("Loaded existing model parameters from checkpoints.")
    else:
        logging.info("Starting training from scratch with new models.")
        

    env = init_env(rng, board_size, num_boards)

    return (
        env,
        black_actor_critic,
        black_params,
        black_opt_state,
        black_optimizer,
        white_actor_critic,
        white_params,
        white_opt_state,
        white_optimizer,
        checkpoint_dir,
        rng,
        board_size,
    )


@jit
def discount_rewards(rewards, gamma):
    """
    Calculate discounted rewards for batched trajectories.

    Args:
      rewards: a 2D jnp.array with shape (T, batch)
      gamma: discount factor.

    Returns:
      A jnp.array of discounted rewards with the same shape as `rewards`.
    """

    def discount_single(r):
        def scan_fn(carry, reward):
            new_carry = reward + gamma * carry
            return new_carry, new_carry

        _, discounted_reversed = lax.scan(scan_fn, 0.0, r[::-1])

        return discounted_reversed[::-1]

    discounted = jax.vmap(discount_single, in_axes=1, out_axes=1)(rewards)
    
    T = rewards.shape[0]
    alternating_signs = jnp.power(-1.0, jnp.arange(T) % 2)
    
    return discounted * alternating_signs[:, None]


def run_episode(env_state, black_actor_critic, black_params, white_actor_critic, white_params, gamma, rng):
    """
    Args:
      env_state: Dictionary containing the environment state.
      black_actor_critic: ActorCritic model for black player (first player).
      black_params: Parameters for black player model.
      white_actor_critic: ActorCritic model for white player (second player).
      white_params: Parameters for white player model.
      gamma: discount factor.
      rng: JAX RNG key.

    Returns:
      black_trajectory: dict with keys ("obs", "actions", "rewards", "masks") for black player
      white_trajectory: dict with keys ("obs", "actions", "rewards", "masks") for white player
      rng: updated RNG key.
    """
    env_state, obs = reset_env(env_state)

    board_size = env_state["board_size"]

    max_steps = board_size * board_size
    num_envs = env_state["num_boards"]

    observations = jnp.zeros(
        (max_steps, num_envs, board_size, board_size), dtype=jnp.float32
    )
    actions = jnp.zeros((max_steps, num_envs, 2), dtype=jnp.int32)
    rewards = jnp.zeros((max_steps, num_envs), dtype=jnp.float32)
    masks = jnp.zeros((max_steps, num_envs), dtype=jnp.bool_)

    loop_state = {
        "env_state": env_state,
        "obs": obs,
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "masks": masks,
        "step_idx": 0,
        "rng": rng,
    }

    @jit
    def cond_fn(state):
        return ~jnp.all(state["env_state"]["dones"])

    # Define separate functions for black and white players to avoid using lax.cond on Python objects
    @partial(jit, static_argnames=["actor_critic"])
    def black_turn_fn(state, actor_critic, params):
        policy_logits, value = actor_critic.apply(params, state["obs"])
        action_mask = get_action_mask(state["env_state"])
        logits = jnp.where(action_mask, policy_logits, -jnp.inf)

        rng, subkey = jax.random.split(state["rng"])
        action = actor_critic.sample_action(logits, subkey)

        step_idx = state["step_idx"]
        active_mask = ~state["env_state"]["dones"]
        observations = state["observations"].at[step_idx].set(state["obs"])
        actions = state["actions"].at[step_idx].set(action)
        masks = state["masks"].at[step_idx].set(active_mask)

        next_env_state, obs, step_rewards, dones = step_env(state["env_state"], action)
        rewards = state["rewards"].at[step_idx].set(step_rewards)

        return {
            "env_state": next_env_state,
            "obs": obs,
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "masks": masks,
            "step_idx": step_idx + 1,
            "rng": rng,
        }
    
    @partial(jit, static_argnames=["actor_critic"])
    def white_turn_fn(state, actor_critic, params):
        policy_logits, value = actor_critic.apply(params, state["obs"])
        action_mask = get_action_mask(state["env_state"])
        logits = jnp.where(action_mask, policy_logits, -jnp.inf)

        rng, subkey = jax.random.split(state["rng"])
        action = actor_critic.sample_action(logits, subkey)

        step_idx = state["step_idx"]
        active_mask = ~state["env_state"]["dones"]
        observations = state["observations"].at[step_idx].set(state["obs"])
        actions = state["actions"].at[step_idx].set(action)
        masks = state["masks"].at[step_idx].set(active_mask)

        next_env_state, obs, step_rewards, dones = step_env(state["env_state"], action)
        rewards = state["rewards"].at[step_idx].set(step_rewards)

        return {
            "env_state": next_env_state,
            "obs": obs,
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "masks": masks,
            "step_idx": step_idx + 1,
            "rng": rng,
        }

    @partial(jit, static_argnames=["black_actor_critic", "white_actor_critic"])
    def body_fn(state, black_actor_critic, black_params, white_actor_critic, white_params):
        # Determine which player's turn it is (black: even steps starting at 0, white: odd steps)
        current_step = state["step_idx"]
        is_black_turn = (current_step % 2 == 0)
        
        # Use JAX's where for pure values selection, not for Python objects
        return jax.lax.cond(
            is_black_turn,
            lambda s: black_turn_fn(s, black_actor_critic, black_params),
            lambda s: white_turn_fn(s, white_actor_critic, white_params),
            state
        )

    # Create a wrapper for body_fn that includes both actor-critic models and their params
    def body_fn_wrapped(state):
        return body_fn(state, black_actor_critic, black_params, white_actor_critic, white_params)

    final_state = lax.while_loop(cond_fn, body_fn_wrapped, loop_state)

    env_state = final_state["env_state"]
    step_idx = final_state["step_idx"]
    observations = final_state["observations"]
    actions = final_state["actions"]
    rewards = final_state["rewards"]
    masks = final_state["masks"]
    rng = final_state["rng"]

    actual_steps = step_idx

    obs_truncated = observations[:actual_steps]
    actions_truncated = actions[:actual_steps]
    rewards_truncated = rewards[:actual_steps]
    masks_truncated = masks[:actual_steps]

    discounted_rewards = discount_rewards(rewards_truncated, gamma)


    # Split trajectories for black (even indices) and white (odd indices) players
    black_trajectory = {
        "obs": obs_truncated[::2],
        "actions": actions_truncated[::2],
        "rewards": discounted_rewards[::2],
        "masks": masks_truncated[::2],
        "episode_length": (actual_steps + 1) // 2  # Ceiling division for odd lengths
    }
    
    white_trajectory = {
        "obs": obs_truncated[1::2],
        "actions": actions_truncated[1::2],
        "rewards": discounted_rewards[1::2],
        "masks": masks_truncated[1::2],
        "episode_length": actual_steps // 2  # Floor division
    }

    return black_trajectory, white_trajectory, rng


@partial(jax.jit, static_argnums=(3, 4))
def train_step(params, opt_state, trajectory, actor_critic, optimizer, entropy_coef):
    """
    Perform one training update with masked loss computation.

    Args:
      params: network parameters.
      opt_state: optimizer state.
      trajectory: collected trajectory with masks.
      actor_critic: network instance.
      optimizer: optax optimizer.
      entropy_coef: coefficient for entropy regularization.

    Returns:
      (updated_params, updated_opt_state, loss, aux, grad_norm)
    """
    masks = trajectory["masks"]

    returns = trajectory["rewards"]

    returns_mean = jnp.mean(returns, where=masks)
    returns_std = jnp.std(returns, where=masks) + 1e-8

    normalized_returns = (returns - returns_mean) / returns_std

    def loss_fn(params):
        obs = trajectory["obs"]
        actions = trajectory["actions"]

        board_size = obs.shape[2]  # obs shape is (T, B, board_size, board_size)

        T, B = obs.shape[0], obs.shape[1]

        obs_flat = obs.reshape(-1, board_size, board_size)
        actions_flat = actions.reshape(-1, 2)
        flat_actions = actions_flat[:, 0] * board_size + actions_flat[:, 1]
        masks_flat = masks.reshape(-1)
        returns_flat = normalized_returns.reshape(-1)

        logits, values = actor_critic.apply(params, obs_flat)
        values = values.reshape(-1)

        logits_flat = logits.reshape(-1, board_size * board_size)
        log_probs = jax.nn.log_softmax(logits_flat, axis=-1)

        batch_indices = jnp.arange(T * B)
        chosen_log_probs = log_probs[batch_indices, flat_actions]

        probs = jax.nn.softmax(logits_flat, axis=-1)
        entropy = -jnp.sum(probs * log_probs, axis=-1)

        advantages = returns_flat - values

        masked_actor_loss = (
            -chosen_log_probs * lax.stop_gradient(advantages) * masks_flat
        )
        masked_critic_loss = jnp.square(returns_flat - values) * masks_flat
        masked_entropy = entropy * masks_flat

        valid_steps_sum = jnp.sum(masks_flat)
        actor_loss = jnp.sum(masked_actor_loss) / valid_steps_sum
        critic_loss = jnp.sum(masked_critic_loss) / valid_steps_sum
        entropy_loss = -jnp.sum(masked_entropy) / valid_steps_sum

        # Use the provided entropy coefficient
        total_loss = actor_loss + 0.5 * critic_loss + entropy_coef * entropy_loss
        return total_loss, (actor_loss, critic_loss, entropy_loss, entropy_coef)

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    grads = jax.tree_util.tree_map(
        lambda g, p: jnp.zeros_like(p) if g is None else g, grads, params
    )
    grad_norm = optax.global_norm(grads)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux, grad_norm


def main():
    """
    Main training loop using separate models for black and white players
    with self-play against randomly selected previous model versions.
    
    Features:
    - Separate models for black and white players
    - Self-play against randomly selected previous model versions
    - Entropy coefficient annealing to balance exploration vs exploitation
      (high entropy early in training encourages exploration, while
      reducing entropy later encourages exploitation of learned strategies)
    """
    config = load_config()
    board_size = config["board_size"]
    
    log_filename = f"gomoku_training_{board_size}x{board_size}.log"
    setup_logging(log_dir="logs", filename=log_filename)
    
    log_config(config)
    
    logging.info("Starting AlphaGomoku training...")
    logging.info(f"JAX devices: {jax.devices()}")

    (
        env,
        black_actor_critic,
        black_params,
        black_opt_state,
        black_optimizer,
        white_actor_critic,
        white_params,
        white_opt_state,
        white_optimizer,
        checkpoint_dir,
        rng,
        board_size,
    ) = init_train(config)

    logging.info("Starting training loop.")

    num_episodes = config["total_iterations"]
    checkpoint_interval = config["save_frequency"]
    gamma = config["discount"]
    
    # Initialize entropy coefficient parameters
    initial_entropy_coef = config["initial_entropy_coef"]
    min_entropy_coef = config["min_entropy_coef"]
    entropy_decay_steps = config["entropy_decay_steps"]
    
    logging.info(f"Entropy annealing: initial={initial_entropy_coef}, "
                 f"min={min_entropy_coef}, decay_steps={entropy_decay_steps}")
    
    last_logged_entropy_coef = initial_entropy_coef
    entropy_log_threshold = 0.1  # Log when entropy coefficient changes by 10%
    
    for episode in range(1, num_episodes + 1):
        # Calculate current entropy coefficient using exponential decay
        progress = min(episode / entropy_decay_steps, 1.0)
        current_entropy_coef = max(
            initial_entropy_coef * (min_entropy_coef / initial_entropy_coef) ** progress,
            min_entropy_coef
        )
        
        # Log entropy coefficient when it changes significantly
        if (abs(current_entropy_coef - last_logged_entropy_coef) / last_logged_entropy_coef > entropy_log_threshold or
            episode == 1 or episode % 1000 == 0):
            logging.info(f"Episode {episode}: Entropy coefficient is now {current_entropy_coef:.6f}")
            last_logged_entropy_coef = current_entropy_coef
        
        black_traj, white_traj, rng = run_episode(
            env, 
            black_actor_critic, 
            black_params, 
            white_actor_critic, 
            white_params,  
            gamma, 
            rng
        )
        
        black_params, black_opt_state, black_loss, black_aux, black_grad_norm = train_step(
            black_params, black_opt_state, black_traj, black_actor_critic, black_optimizer,
            entropy_coef=current_entropy_coef
        )
        
        white_params, white_opt_state, white_loss, white_aux, white_grad_norm = train_step(
            white_params, white_opt_state, white_traj, white_actor_critic, white_optimizer,
            entropy_coef=current_entropy_coef
        )

        logging.info(
            f"Episode {episode}/{num_episodes}: "
            f"Black - Loss {black_loss:.4f} (Actor Loss {black_aux[0]:.4f}, "
            f"Critic Loss {black_aux[1]:.4f}, Entropy Loss {black_aux[2]:.4f}, Entropy Coef {black_aux[3]:.6f}) | "
            f"Grad Norm {black_grad_norm:.4f} | "
            f"White - Loss {white_loss:.4f} (Actor Loss {white_aux[0]:.4f}, "
            f"Critic Loss {white_aux[1]:.4f}, Entropy Loss {white_aux[2]:.4f}, Entropy Coef {white_aux[3]:.6f}) | "
            f"Grad Norm {white_grad_norm:.4f}"
        )

        if episode % checkpoint_interval == 0:
            save_checkpoint(black_params, checkpoint_dir)
            save_checkpoint(white_params, checkpoint_dir)
            logging.info(f"Saved both black and white models as checkpoints at episode {episode}")
            
            rng, select_key = jax.random.split(rng, 2)
            
            black_checkpoint_path, white_checkpoint_path = select_training_checkpoints(checkpoint_dir, select_key)
            
            rng, black_key, white_key = jax.random.split(rng, 3)
            
            if black_checkpoint_path is not None and jax.random.uniform(black_key) < 0.5:
                loaded_black_params = load_checkpoint(black_checkpoint_path)
                if loaded_black_params is not None:
                    black_params = loaded_black_params
                    black_opt_state = black_optimizer.init(black_params)
                    logging.info(f"Switched to checkpoint {black_checkpoint_path} for black player")
                else:
                    logging.info("Failed to load black model checkpoint, continuing with current model")
            else:
                logging.info("Continued with current black model")
            
            if white_checkpoint_path is not None and jax.random.uniform(white_key) < 0.5:
                loaded_white_params = load_checkpoint(white_checkpoint_path)
                if loaded_white_params is not None:
                    white_params = loaded_white_params
                    white_opt_state = white_optimizer.init(white_params)
                    logging.info(f"Switched to checkpoint {white_checkpoint_path} for white player")
                else:
                    logging.info("Failed to load white model checkpoint, continuing with current model")
            else:
                logging.info("Continued with current white model")
            
            logging.info(f"Selected models for next training round after episode {episode}")
            
            # TODO: Consider adding periodic evaluation here to measure model strength
            # This could involve playing against a fixed baseline model or self-play with
            # deterministic policy (argmax instead of sampling) to measure win rates

    logging.info("Training complete. Saving final model parameters.")
    save_checkpoint(black_params, checkpoint_dir)
    save_checkpoint(white_params, checkpoint_dir)


if __name__ == "__main__":
    main()
