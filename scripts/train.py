import jax
import jax.numpy as jnp
from jax import jit, lax
import optax
import logging
import os
from policy.actor_critic import ActorCritic
from gomoku.env import Gomoku
from flax import serialization
from functools import partial

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

BOARD_SIZE = 9
LEARNING_RATE = 4e-5
GAMMA = 0.99
NUM_EPISODES = 50000
RENDER = False
CHECKPOINT_DIR = f"checkpoints/{BOARD_SIZE}x{BOARD_SIZE}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
BLACK_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "black.pkl")
WHITE_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "white.pkl")
CHECKPOINT_INTERVAL = 100

@jit
def discount_rewards(rewards, gamma):
    """
    Calculate discounted rewards.
    
    Args:
      rewards: list/array of rewards.
      gamma: discount factor.
      
    Returns:
      jnp.array of discounted rewards.
    """
    rewards = jnp.asarray(rewards)
    
    def scan_fn(carry, reward):
        new_carry = reward + gamma * carry
        return new_carry, new_carry
    
    _, discounted_reversed = lax.scan(scan_fn, 0.0, rewards[::-1])
    return discounted_reversed[::-1]

def run_episode(env, white_actor_critic, black_actor_critic, white_params, black_params, rng):
    """
    Run one episode.
    
    Args:
      env: Gomoku environment.
      white_actor_critic: white network instance.
      black_actor_critic: black network instance.
      white_params: white network parameters.
      black_params: black network parameters.
      rng: JAX RNG key.
      
    Returns:
      (trajectory_white, trajectory_black, updated_rng)
    """
    obs = env.reset()
    done = False

    trajectory_white = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "log_probs": [],
        "values": []
    }
    trajectory_black = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "log_probs": [],
        "values": []
    }
    
    while not done:
        current_player = env.current_player
        obs_jax = jnp.expand_dims(jnp.array(obs, dtype=jnp.float32), axis=0)
        
        if current_player == 1:
            policy_logits, value = black_actor_critic.apply(black_params, obs_jax)
        else:
            policy_logits, value = white_actor_critic.apply(white_params, obs_jax)
        value = value[0]
        policy_logits = policy_logits[0]

        action_mask = env.get_action_mask()
        logits = jnp.where(action_mask, policy_logits, -jnp.inf)
        
        rng, black_subkey, white_subkey = jax.random.split(rng, 3)

        if current_player == 1:
            sampled = black_actor_critic.sample_action(logits[None, ...], black_subkey)
        else:
            sampled = white_actor_critic.sample_action(logits[None, ...], white_subkey)
        action = sampled[0]
        
        flat_logits = logits.reshape(-1)
        flat_log_probs = jax.nn.log_softmax(flat_logits)
        flat_index = jnp.int32(action[0] * BOARD_SIZE + action[1])
        log_prob = flat_log_probs[flat_index]

        if current_player == 1:
            trajectory_black["obs"].append(jnp.array(obs, dtype=jnp.float32))
            trajectory_black["actions"].append((jnp.int32(action[0]), jnp.int32(action[1])))
            trajectory_black["log_probs"].append(log_prob)
            trajectory_black["values"].append(value)
        else:
            trajectory_white["obs"].append(jnp.array(obs, dtype=jnp.float32))
            trajectory_white["actions"].append((jnp.int32(action[0]), jnp.int32(action[1])))
            trajectory_white["log_probs"].append(log_prob)
            trajectory_white["values"].append(value)
        
        obs, reward, done = env.step((jnp.int32(action[0]), jnp.int32(action[1])))

        if done:
            if current_player == 1:
                trajectory_black["rewards"].append(reward)
                trajectory_white["rewards"][-1] = -reward * GAMMA
            else:
                trajectory_white["rewards"].append(reward)  
                trajectory_black["rewards"][-1] = -reward * GAMMA
        else:
            if current_player == 1:
                trajectory_black["rewards"].append(reward)
            else:
                trajectory_white["rewards"].append(reward)
    
    return trajectory_white, trajectory_black, rng

@partial(jax.jit, static_argnums=(3, 4))
def train_step(params, opt_state, trajectory, actor_critic, optimizer, gamma):
    """
    Perform one training update.
    
    Args:
      params: network parameters.
      opt_state: optimizer state.
      trajectory: collected trajectory.
      actor_critic: network instance.
      optimizer: optax optimizer.
      gamma: discount factor.
      
    Returns:
      (updated_params, updated_opt_state, loss, aux, grad_norm)
    """
    returns = discount_rewards(jnp.array(trajectory["rewards"]), gamma)
    returns_mean = returns.mean()
    returns_std = returns.std() + 1e-8
    normalized_returns = (returns - returns_mean) / returns_std

    def loss_fn(params):
        obs = jnp.stack(trajectory["obs"], axis=0)
        actions = jnp.array([a[0] * BOARD_SIZE + a[1] for a in trajectory["actions"]])
        logits, values = actor_critic.apply(params, obs)
        T = obs.shape[0]
        logits = logits.reshape(T, -1)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        chosen_log_probs = log_probs[jnp.arange(T), actions]
        
        entropy = -jnp.sum(jax.nn.softmax(logits, axis=-1) * log_probs, axis=-1)
        
        advantages = normalized_returns - values.squeeze()
        actor_loss = -jnp.mean(chosen_log_probs * lax.stop_gradient(advantages))
        critic_loss = jnp.mean((normalized_returns - values.squeeze()) ** 2)
        entropy_loss = -jnp.mean(entropy)
        
        total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
        return total_loss, (actor_loss, critic_loss, entropy_loss)
    
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    grads = jax.tree_util.tree_map(lambda g, p: jnp.zeros_like(p) if g is None else g, grads, params)
    grad_norm = optax.global_norm(grads)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux, grad_norm

def save_checkpoint(params, episode, checkpoint_path, player):
    """
    Save network checkpoint.
    
    Args:
      params: network parameters.
      episode: current episode number.
      checkpoint_path: File path for checkpoint.
      player: Identifier string for the player ("white" or "black").
      
    Returns: None.
    """
    checkpoint = {"params": params, "episode": episode}
    with open(checkpoint_path, "wb") as f:
        f.write(serialization.to_bytes(checkpoint))
    logging.info(f"{player.capitalize()} checkpoint saved to {checkpoint_path} at episode {episode}")

def load_checkpoint(checkpoint_path, player):
    """
    Load network checkpoint.
    
    Args:
      checkpoint_path: File path for checkpoint.
      player: Identifier string for the player ("white" or "black").
      
    Returns:
      A tuple (params, episode) if the checkpoint exists, or (None, 0) if not.
    """
    try:
        with open(checkpoint_path, "rb") as f:
            data = f.read()
        checkpoint = serialization.from_bytes({"params": None, "episode": 0}, data)
        logging.info(f"{player.capitalize()} checkpoint loaded from {checkpoint_path} at episode {checkpoint['episode']}")
        return checkpoint["params"], checkpoint["episode"]
    except FileNotFoundError:
        logging.info(f"{player.capitalize()} checkpoint not found at {checkpoint_path}.")
        return None, 0

def main():
    """
    Main training loop.
    
    Args: None.
    
    Returns: None.
    """
    logging.info("Initializing environment and models.")
    env = Gomoku(board_size=BOARD_SIZE, mode="human" if RENDER else "train")
    white_actor_critic = ActorCritic(board_size=BOARD_SIZE)
    black_actor_critic = ActorCritic(board_size=BOARD_SIZE)
    
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, BOARD_SIZE, BOARD_SIZE), dtype=jnp.float32)
    white_params = white_actor_critic.init(rng, dummy_input)
    black_params = black_actor_critic.init(rng, dummy_input)
    
    optimizer = optax.adam(LEARNING_RATE)

    black_checkpoint_params, start_episode_black = load_checkpoint(BLACK_CHECKPOINT_PATH, "black")
    white_checkpoint_params, start_episode_white = load_checkpoint(WHITE_CHECKPOINT_PATH, "white")
    
    if white_checkpoint_params is not None and black_checkpoint_params is not None:
        white_params = white_checkpoint_params
        black_params = black_checkpoint_params
        start_episode = min(start_episode_white, start_episode_black)
        logging.info(f"Resuming training from episode {start_episode}.")
    else:
        start_episode = 1
        logging.info("No valid checkpoints found. Starting from scratch.")
    
    white_opt_state = optimizer.init(white_params)
    black_opt_state = optimizer.init(black_params)
    
    logging.info("Starting training loop.")
    for episode in range(start_episode, NUM_EPISODES + 1):
        traj_white, traj_black, rng = run_episode(
            env, white_actor_critic, black_actor_critic,
            white_params, black_params, rng
        )
        # Train on black first.
        if len(traj_black["rewards"]) > 0:
            black_params, black_opt_state, black_loss, black_aux, black_grad_norm = train_step(
                black_params, black_opt_state, traj_black, black_actor_critic, optimizer, GAMMA
            )
        if len(traj_white["rewards"]) > 0:
            white_params, white_opt_state, white_loss, white_aux, white_grad_norm = train_step(
                white_params, white_opt_state, traj_white, white_actor_critic, optimizer, GAMMA
            )
        
        if episode % CHECKPOINT_INTERVAL == 0:
            logging.info(
                f"Episode {episode}:"
                f" Black Loss {black_loss:.4f} (Actor {black_aux[0]:.4f}, Critic {black_aux[1]:.4f}, Entropy {black_aux[2]:.4f}) Grad Norm {black_grad_norm:.4f};"
                f" White Loss {white_loss:.4f} (Actor {white_aux[0]:.4f}, Critic {white_aux[1]:.4f}, Entropy {white_aux[2]:.4f}) Grad Norm {white_grad_norm:.4f}"
            )
            # Save checkpoints: Black first then White.
            save_checkpoint(black_params, episode, BLACK_CHECKPOINT_PATH, "black")
            save_checkpoint(white_params, episode, WHITE_CHECKPOINT_PATH, "white")
    
    logging.info("Training complete. Saving final model parameters.")
    save_checkpoint(black_params, NUM_EPISODES, BLACK_CHECKPOINT_PATH, "black")
    save_checkpoint(white_params, NUM_EPISODES, WHITE_CHECKPOINT_PATH, "white")
    
if __name__ == "__main__":
    main()
