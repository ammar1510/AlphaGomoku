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
NUM_BOARDS = 256
GAMMA = 0.99
NUM_EPISODES = 50000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
RENDER = True
CHECKPOINT_DIR = f"checkpoints/{BOARD_SIZE}x{BOARD_SIZE}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "black.pkl")
CHECKPOINT_INTERVAL = 100
GRAD_CLIP_NORM = 1.0

@jit
def discount_rewards(rewards, gamma):
    """
    Calculate discounted rewards for batched trajectories.

    Args:
      rewards: a 2D jnp.array with shape (batch, T)
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
    return jax.vmap(discount_single)(rewards)

def run_episode(env, actor_critic, params, gamma, rng):
    """
    Runs one episode using a single shared model for both players.
    
    Args:
      env: the Gomoku environment.
      actor_critic: the single shared ActorCritic model.
      params: the shared parameters.
      rng: JAX RNG key.
    
    Returns:
      trajectory: dict with keys ("obs", "actions", "rewards", "values", "dones")
      rng: updated RNG key.
    """
    obs, dones = env.reset()
    
    trajectory = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "values": [],
        "dones": []
    }
    
    while not jnp.all(dones):
        policy_logits, value = actor_critic.apply(params, obs)
        action_mask = env.get_action_mask()
        logits = jnp.where(action_mask, policy_logits, -jnp.inf)
        
        rng, subkey = jax.random.split(rng)
        action = actor_critic.sample_action(logits, subkey)
        
        trajectory["obs"].append(obs)
        trajectory["actions"].append(action)
        trajectory["values"].append(value)
        trajectory["dones"].append(dones)
        
        obs, rewards, dones = env.step(action)
        trajectory["rewards"].append(rewards)

    #obs -> (T, num_envs, board_size, board_size)
    #action -> (T, num_envs, 2)
    #reward -> (T, num_envs)
    #dones -> (T, num_envs)
    #values -> (T, num_envs)
    trajectory["rewards"] = discount_rewards(jnp.array(trajectory["rewards"]).transpose(1, 0), gamma).transpose(1, 0)
    trajectory["rewards"] = trajectory["rewards"]*env.winners[None,:]

    
    trajectory = preprocess_trajectory(trajectory)
    return trajectory, rng


def preprocess_trajectory(trajectory):
    """
    Args:
        trajectory: dict containing keys:
            "obs": jnp.ndarray with shape (T, num_envs, ...) e.g. (T, num_envs, board_size, board_size)
            "actions": jnp.ndarray with shape (T, num_envs, 2)
            "log_probs": jnp.ndarray with shape (T, num_envs)
            "rewards": jnp.ndarray with shape (T, num_envs)
            "dones": jnp.ndarray with shape (T, num_envs), where True indicates the episode is done.

    Returns:
        A dict with keys "obs", "actions", "log_probs", and "rewards"
        containing the valid, concatenated trajectory data from all environments.
    """

    trajectory["obs"] = jnp.array(trajectory["obs"])
    trajectory["actions"] = jnp.array(trajectory["actions"])
    trajectory["rewards"] = jnp.array(trajectory["rewards"])
    trajectory["dones"] = jnp.array(trajectory["dones"])

    T, num_envs = trajectory["obs"].shape[0], trajectory["obs"].shape[1]
    indices = jnp.argmax(trajectory["dones"], axis=0)
    mask = jnp.arange(T)[:, None] <= indices[None, :]
    valid_obs = trajectory["obs"][mask] #(T*num_envs, board_size, board_size)
    valid_actions = trajectory["actions"][mask]   #(T*num_envs,2)
    valid_rewards = trajectory["rewards"][mask] #(T*num_envs)

    return {
        "obs": valid_obs,
        "actions": valid_actions,
        "rewards": valid_rewards
    }

@partial(jax.jit, static_argnums=(3, 4))
def train_step(params, opt_state, trajectory, actor_critic, optimizer):
    """
    Perform one training update.
    
    Args:
      params: network parameters.
      opt_state: optimizer state.
      trajectory: collected trajectory.
      actor_critic: network instance.
      optimizer: optax optimizer.
    Returns:
      (updated_params, updated_opt_state, loss, aux, grad_norm)
    """
    returns = trajectory["rewards"]
    returns_mean = returns.mean()
    returns_std = returns.std() + 1e-8
    normalized_returns = (returns - returns_mean) / returns_std

    def loss_fn(params):
        obs = trajectory["obs"]
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
    updates, opt_state = optimizer.update(grads, opt_state, params)
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
    Main training loop using a single shared model for both players.
    """
    logging.info("Initializing environment and model.")

    # When rendering, we only need one board; otherwise, use multiple boards.
    num_boards = 1 if RENDER else NUM_BOARDS
    env = Gomoku(board_size=BOARD_SIZE, num_boards=num_boards, mode="human" if RENDER else "train")

    # Create a single shared ActorCritic model.
    actor_critic = ActorCritic(board_size=BOARD_SIZE)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, BOARD_SIZE, BOARD_SIZE), dtype=jnp.float32)
    params = actor_critic.init(rng, dummy_input)

    # Create an optimizer with gradient clipping included
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP_NORM),  # First clip gradients
        optax.adamw(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)  # Then apply Adam updates
    )
    opt_state = optimizer.init(params)

    # Load checkpoint (using a single shared checkpoint path, e.g., "model").
    checkpoint_params, start_episode = load_checkpoint(CHECKPOINT_PATH, "model")
    if checkpoint_params is not None:
        params = checkpoint_params
        logging.info(f"Resuming training from episode {start_episode}.")
    else:
        start_episode = 1
        logging.info("No valid checkpoint found. Starting from scratch.")

    logging.info("Starting training loop.")
    for episode in range(start_episode, NUM_EPISODES + 1):
        traj, rng = run_episode(env, actor_critic, params, GAMMA, rng)
        # Perform a training step and update parameters.
        params, opt_state, loss, aux, grad_norm = train_step(params, opt_state, traj, actor_critic, optimizer)

        # Log the loss and gradient norm (and auxiliary losses) for every episode.
        logging.info(
            f"Episode {episode}: Loss {loss:.4f} (Actor Loss {aux[0]:.4f}, Critic Loss {aux[1]:.4f}, Entropy Loss {aux[2]:.4f}) | Grad Norm {grad_norm:.4f}"
        )

        # Save checkpoint at the defined interval.
        if episode % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(params, episode, CHECKPOINT_PATH, "model")

    logging.info("Training complete. Saving final model parameters.")
    save_checkpoint(params, NUM_EPISODES, CHECKPOINT_PATH, "model")

if __name__ == "__main__":
    main()
