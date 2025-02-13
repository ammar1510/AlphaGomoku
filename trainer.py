import gymnasium as gym
import jax
import jax.numpy as jnp
from jax import jit,lax,vmap
import numpy as np
import optax
from policy.actor_critic import ActorCritic

# Hyperparameters
BOARD_SIZE = 15     # Gomoku board size (15x15)
CHANNELS = 1        # Number of board channels (adjust if you change the board representation)
LEARNING_RATE = 1e-3
GAMMA = 0.99
NUM_EPISODES = 1000

@jit
def discount_rewards(rewards, gamma):
    """
    Compute the discounted reward (return) for a sequence using jax.lax.scan.
    
    This function assumes that rewards is an array-like object (np.ndarray or jnp.ndarray) and computes
    the discounted returns by scanning over the rewards in reverse order, then reversing the result.
    
    Args:
        rewards (np.ndarray or jnp.ndarray): Array of rewards.
        gamma (float): Discount factor.
        
    Returns:
        jnp.ndarray: The discounted returns.
    """
    # Ensure rewards is a JAX array.
    rewards = jnp.asarray(rewards)
    
    def scan_fn(carry, reward):
        new_carry = reward + gamma * carry
        return new_carry, new_carry
    
    # Reverse rewards to start scanning from the last reward.
    _, discounted_reversed = jax.lax.scan(scan_fn, 0.0, rewards[::-1])
    # Reverse the computed discounted rewards to restore the original order.
    return discounted_reversed[::-1]

def run_episode(env, actor_critic, params, rng):
    """
    Plays one full episode (game) in the environment using the given actor–critic network,
    recording the trajectory for policy update. In addition to the observation, action, reward,
    log probability, and value, we also record the acting player's identity for reward flipping later.
    
    Args:
        env: The Gomoku environment (must have reset() and step() methods).
        actor_critic: An instance of ActorCritic.
        params: The current parameters of the model.
        rng: A JAX PRNGKey.
    
    Returns:
        trajectory (dict): A dictionary containing observations, actions, rewards,
                           log-probabilities, values, and player identifiers.
        rng: The updated PRNGKey.
    """
    obs = env.reset()[0]
    done = False

    # Trajectory data now also includes a list for the acting player's identity.
    trajectory = {
        "obs": [],
        "actions": [],    # Actions as tuples (row, col)
        "rewards": [],
        "log_probs": [],
        "values": [],
        "players": []     # Each element is 1 (black) or -1 (white)
    }
    
    while not done:
        # Record the acting player before taking an action.
        current_player = env.current_player

        # Ensure the observation is a JAX array with a batch dimension.
        obs_jax = jnp.expand_dims(jnp.array(obs, dtype=jnp.float32), axis=0)  
        
        # Get the policy logits and value prediction.
        logits, value = actor_critic.apply(params, obs_jax)
        logits = logits[0]  # Remove the batch dimension.
        value = value[0]    # Scalar value for that state.

        # Sample an action from the network.
        rng, subkey = jax.random.split(rng)
        sampled = actor_critic.sample_action(logits[None, ...], subkey)
        action = sampled[0]  # tuple (row, col)
        
        # Compute the log probability of the selected action.
        flat_logits = logits.reshape(-1)
        flat_log_probs = jax.nn.log_softmax(flat_logits)
        flat_index = int(action[0] * BOARD_SIZE + action[1])
        log_prob = flat_log_probs[flat_index]
        
        # Record information along with the acting player's identity.
        trajectory["obs"].append(jnp.array(obs, dtype=jnp.float32))
        trajectory["actions"].append((int(action[0]), int(action[1])))
        trajectory["log_probs"].append(log_prob)
        trajectory["values"].append(value)
        trajectory["players"].append(current_player)
        
        # Take the action in the environment.
        obs, reward, done, info = env.step((int(action[0]), int(action[1])))
        trajectory["rewards"].append(reward)
    
    return trajectory, rng

@jax.jit
def train_step(params, opt_state, trajectory, actor_critic, optimizer, gamma):
    """
    Performs one gradient update on an episode trajectory.
    Implements reward flipping such that, for every move,
    we transform the return based on the player's identity:
      - For player 1 (black), the target return is unchanged.
      - For player -1 (white), the target return is multiplied by -1.

    Args:
         params: The current model parameters.
         opt_state: The current optimizer state.
         trajectory: A dictionary containing the episode trajectory.
         actor_critic: The ActorCritic network instance.
         optimizer: The optax optimizer.
         gamma: Discount factor.
    
    Returns:
         new_params: The updated parameters.
         new_opt_state: The updated optimizer state.
         loss: The total loss.
         aux: A tuple with (actor_loss, critic_loss, entropy_loss) for logging.
    """
    # Compute discounted returns.
    returns = discount_rewards(np.array(trajectory["rewards"]), gamma)
    returns = jnp.array(returns)
    
    # Get the recorded player identities (shape: (T,)).
    players = jnp.array(trajectory["players"])
    # Adjust returns according to the player's perspective.
    adjusted_returns = returns * players

    def loss_fn(params):
        # Stack observations.
        obs = jnp.stack(trajectory["obs"], axis=0)  # shape: (T, BOARD_SIZE, BOARD_SIZE, CHANNELS)
        
        # Convert actions from (row, col) into flattened indices.
        actions = jnp.array([a[0] * BOARD_SIZE + a[1] for a in trajectory["actions"]])
        
        # Compute the forward pass over the whole trajectory.
        logits, values = actor_critic.apply(params, obs)
        T = obs.shape[0]
        logits = logits.reshape(T, -1)  # shape: (T, BOARD_SIZE*BOARD_SIZE)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        chosen_log_probs = log_probs[jnp.arange(T), actions]
        
        # Compute the entropy.
        entropy = -jnp.sum(jax.nn.softmax(logits, axis=-1) * log_probs, axis=-1)
        
        # Compute advantages using the adjusted returns.
        advantages = adjusted_returns - values
        
        # Actor loss uses the log-probabilities weighted by the advantages.
        actor_loss = -jnp.mean(chosen_log_probs * advantages)
        # Critic loss minimizes the squared difference.
        critic_loss = jnp.mean((adjusted_returns - values) ** 2)
        entropy_loss = -jnp.mean(entropy)
        
        total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
        return total_loss, (actor_loss, critic_loss, entropy_loss)
    
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

def main():
    """
    Main training loop.
    
    This function creates the environment and the model, then alternates between
    running an episode(run_episode()) and updating the parameters from the collected trajectory(train_step()).
    
    The agent plays Gomoku (self-play) with alternating turns.
    """
    # Create the environment.
    # (Make sure your Gym environment for Gomoku is available – here we assume its name is "Gomoku-v0".)
    env = gym.make("Gomoku")
    
    # Instantiate the ActorCritic model.
    actor_critic = ActorCritic(board_size=BOARD_SIZE, channels=CHANNELS)
    
    # Initialize PRNG key and the model parameters.
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, BOARD_SIZE, BOARD_SIZE, CHANNELS), dtype=jnp.float32)
    params = actor_critic.init(rng, dummy_input)
    
    # Set up the optimizer.
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)
    
    # Training loop.
    for episode in range(1, NUM_EPISODES + 1):
        # Run one episode and collect the trajectory.
        trajectory, rng = run_episode(env, actor_critic, params, rng)
        
        # Compute gradients and update model parameters based on the trajectory.
        params, opt_state, loss, (actor_loss, critic_loss, entropy_loss) = train_step(
            params, opt_state, trajectory, actor_critic, optimizer, GAMMA)
        
        # Logging progress every so often.
        if episode % 10 == 0:
            print(f"Episode {episode}: Total Loss {loss:.4f} Actor Loss {actor_loss:.4f} "
                  f"Critic Loss {critic_loss:.4f} Entropy Loss {entropy_loss:.4f}")
    
    # Optionally, you can save your parameters here.
    # For example, using Flax serialization or pickle.

if __name__ == "__main__":
    main()
