import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
import optax
import logging
import os
from policy.actor_critic import ActorCritic
from gomoku.env import Gomoku
from flax import serialization
from functools import partial


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Hyperparameters
BOARD_SIZE = 9         # Gomoku board size (9x9 board in this example)
LEARNING_RATE = 5e-5
GAMMA = 0.99
NUM_EPISODES = 1000
RENDER = False

@jit
def discount_rewards(rewards, gamma):
    """
    Compute the discounted reward (return) for a sequence using jax.lax.scan.
    
    Args:
        rewards (np.ndarray or jnp.ndarray): Array of rewards.
        gamma (float): Discount factor.
        
    Returns:
        jnp.ndarray: The discounted returns.
    """
    rewards = jnp.asarray(rewards)
    
    def scan_fn(carry, reward):
        new_carry = reward + gamma * carry
        return new_carry, new_carry
    
    _, discounted_reversed = lax.scan(scan_fn, 0.0, rewards[::-1])
    return discounted_reversed[::-1]

@partial(jit, static_argnums=(0,1))
def run_episode(env, actor_critic, params, rng):
    """
    Plays one full episode (game) in the environment using the given actor–critic network,
    recording the trajectory for policy update.
    
    Args:
        env: The Gomoku environment.
        actor_critic: An instance of ActorCritic.
        params: The current parameters of the model.
        rng: A JAX PRNGKey.
    
    Returns:
        trajectory (dict): A dictionary containing observations, actions, rewards,
                           log-probabilities, values, and player identifiers.
        rng: The updated PRNGKey.
    """
    # Gymnasium's reset() may return (obs, info)
    obs = env.reset()[0]
    done = False

    trajectory = {
        "obs": [],
        "actions": [],    # Actions as tuples (row, col)
        "rewards": [],
        "log_probs": [],
        "values": [],
        "players": []     # Acting player's identity (1 or -1)
    }
    
    while not done:
        current_player = env.current_player
        obs_jax = jnp.expand_dims(jnp.array(obs, dtype=jnp.float32), axis=0)
        
        # Before feeding the board into the network,
        # multiply the board state by the current player's indicator.
        # Assume `board_state` is the current board and `current_player` is either +1 or -1.
        board_state_conditioned = obs_jax * current_player

        # Then pass this conditioned board state to your ActorCritic network.
        policy_logits, value = actor_critic.apply(params, board_state_conditioned)
        policy_logits = policy_logits[0]  # Remove batch dimension.
        value = value[0]    # Scalar value for that state.

        action_mask = env.get_action_mask()
        logits = jnp.where(action_mask, policy_logits, -jnp.inf)

        rng, subkey = jax.random.split(rng)
        sampled = actor_critic.sample_action(logits[None, ...], subkey)
        action = sampled[0]  # tuple (row, col)
        
        flat_logits = logits.reshape(-1)
        flat_log_probs = jax.nn.log_softmax(flat_logits)
        flat_index = jnp.int32(action[0] * BOARD_SIZE + action[1])
        log_prob = flat_log_probs[flat_index]
        
        trajectory["obs"].append(jnp.array(obs, dtype=jnp.float32))
        trajectory["actions"].append((jnp.int32(action[0]), jnp.int32(action[1])))
        trajectory["log_probs"].append(log_prob)
        trajectory["values"].append(value)
        trajectory["players"].append(current_player)
        
        # Adjust for Gymnasium's step return. If your env returns four values, adjust accordingly.
        obs, reward, done, info = env.step((jnp.int32(action[0]), jnp.int32(action[1])))

        trajectory["rewards"].append(reward)
    
    return trajectory, rng

# hacky fix for now
@partial(jit, static_argnums=(3, 4))
def train_step(params, opt_state, trajectory, actor_critic, optimizer, gamma):
    """
    Performs one gradient update on an episode trajectory and returns the gradient norm.
    
    Args:
         params: The current model parameters.
         opt_state: The current optimizer state.
         trajectory: A dictionary containing the episode trajectory.
         actor_critic: The ActorCritic network instance (marked static).
         optimizer: The optax optimizer (marked static).
         gamma: Discount factor.
    
    Returns:
         new_params: The updated parameters.
         new_opt_state: The updated optimizer state.
         loss: The total loss.
         aux: A tuple with (actor_loss, critic_loss, entropy_loss) for logging.
         grad_norm: The global norm of the gradients.
    """
    returns = discount_rewards(jnp.array(trajectory["rewards"]), gamma)
    # Normalize returns for stability
    returns_mean = returns.mean()
    returns_std = returns.std() + 1e-8  # epsilon to avoid division by zero
    normalized_returns = (returns - returns_mean) / returns_std

    players = jnp.array(trajectory["players"])
    adjusted_returns = normalized_returns * players

    def loss_fn(params):
        obs = jnp.stack(trajectory["obs"], axis=0)
        actions = jnp.array([a[0] * BOARD_SIZE + a[1] for a in trajectory["actions"]])
        logits, values = actor_critic.apply(params, obs)
        T = obs.shape[0]
        logits = logits.reshape(T, -1)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        chosen_log_probs = log_probs[jnp.arange(T), actions]
        
        # Compute the entropy.
        entropy = -jnp.sum(jax.nn.softmax(logits, axis=-1) * log_probs, axis=-1)
        
        advantages = adjusted_returns - values
        # Detach the advantages so gradients from actor loss do not affect the critic
        detached_advantages = lax.stop_gradient(advantages)
        actor_loss = -jnp.mean(chosen_log_probs * detached_advantages)
        critic_loss = jnp.mean((adjusted_returns - values) ** 2)
        entropy_loss = -jnp.mean(entropy)
        
        total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
        return total_loss, (actor_loss, critic_loss, entropy_loss)
    
    # Compute gradients.
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    # Replace None gradients with zeros so that the tree structure matches.
    grads = jax.tree_map(lambda g: jnp.zeros_like(g) if g is None else g, grads)
    # Compute the global norm of the gradients to monitor for exploding values.
    grad_norm = optax.global_norm(grads)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux, grad_norm

def save_checkpoint(params, opt_state, filename=f"checkpoints/{BOARD_SIZE}x{BOARD_SIZE}.pkl"):
    """
    Saves a checkpoint containing both model parameters and optimizer state.
    """
    checkpoint = {"params": params, "opt_state": opt_state}
    with open(filename, "wb") as f:
        f.write(serialization.to_bytes(checkpoint))
    logging.info(f"Checkpoint saved to {filename}")

def load_checkpoint(filename=f"checkpoints/{BOARD_SIZE}x{BOARD_SIZE}.pkl"):
    """
    Loads a checkpoint containing model parameters and optimizer state.
    """
    with open(filename, "rb") as f:
        data = f.read()
    checkpoint = serialization.from_bytes({"params": None, "opt_state": None}, data)
    logging.info(f"Checkpoint loaded from {filename}")
    return checkpoint["params"], checkpoint["opt_state"]

def main():
    """
    Main training loop.
    
    Creates the environment and the ActorCritic model, then alternates between
    running an episode and updating the parameters from the collected trajectory.
    Logs progress, including gradient norms, and saves a checkpoint (model and optimizer state)
    every 10 episodes. Also attempts to load a checkpoint if one exists.
    """
    logging.info("Initializing environment and model.")
    env = Gomoku(board_size=BOARD_SIZE, mode="human" if RENDER else "train")
    actor_critic = ActorCritic(board_size=BOARD_SIZE)
    
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, BOARD_SIZE, BOARD_SIZE), dtype=jnp.float32)
    params = actor_critic.init(rng, dummy_input)
    
    # Incorporate gradient clipping into the optimizer.
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Adjust clipping threshold as needed.
        optax.adam(LEARNING_RATE)
    )
    opt_state = optimizer.init(params)
    
    # Load checkpoint if it exists
    checkpoint_path = f"checkpoints/{BOARD_SIZE}x{BOARD_SIZE}.pkl"
    if os.path.exists(checkpoint_path):
        params, opt_state = load_checkpoint(checkpoint_path)
        logging.info("Checkpoint loaded; resuming training.")
    
    logging.info("Starting training loop.")
    for episode in range(1, NUM_EPISODES + 1):
        trajectory, rng = run_episode(env, actor_critic, params, rng)
        params, opt_state, loss, (actor_loss, critic_loss, entropy_loss), grad_norm = train_step(
            params, opt_state, trajectory, actor_critic, optimizer, GAMMA)
        
        if episode % 10 == 0:
            logging.info(f"Episode {episode}: Loss {loss:.4f} Actor Loss {actor_loss:.4f} "
                         f"Critic Loss {critic_loss:.4f} Entropy Loss {entropy_loss:.4f} "
                         f"Grad Norm {grad_norm:.4f}")
            save_checkpoint(params, opt_state, checkpoint_path)
    
    logging.info("Training complete. Saving final model parameters.")
    save_checkpoint(params, opt_state, checkpoint_path)
    
if __name__ == "__main__":
    main()
