import jax
import jax.numpy as jnp
from jax import lax, jit
from functools import partial

from env.pong import get_action_mask, reset_env, step_env
from models.pong_actor_critic import PongActorCritic


def _initialize_trajectory_buffers(env_state, max_steps):
    """Initializes buffers to store trajectory data."""
    B, board_size, _ = env_state["boards"].shape
    observations = jnp.zeros((max_steps, B, board_size, board_size), dtype=jnp.float32)
    actions = jnp.zeros((max_steps, B, 2), dtype=jnp.int32)
    rewards = jnp.zeros((max_steps, B), dtype=jnp.float32)
    masks = jnp.zeros((max_steps, B), dtype=jnp.bool_)
    logprobs = jnp.zeros((max_steps, B), dtype=jnp.float32)
    return observations, actions, rewards, masks, logprobs


@partial(jit, static_argnames=["actor_critic"])
def player_move(loop_state, actor_critic, params):
    """Takes a single step in the environment using the provided actor-critic."""
    policy_logits, _ = actor_critic.apply(params, loop_state["obs"])
    action_mask = get_action_mask(loop_state["env_state"])
    masked_logits = jnp.where(action_mask, policy_logits, -jnp.inf)

    rng, subkey = jax.random.split(loop_state["rng"])
    action = actor_critic.sample_action(masked_logits, subkey)
    logprob = actor_critic.log_prob(masked_logits, action)
    step_idx = loop_state["step_idx"]
    active_mask = ~loop_state["env_state"]["dones"]

    observations = loop_state["observations"].at[step_idx].set(loop_state["obs"])
    actions = loop_state["actions"].at[step_idx].set(action)
    masks = loop_state["masks"].at[step_idx].set(active_mask)
    logprobs = loop_state["logprobs"].at[step_idx].set(logprob)
    next_env_state, obs, step_rewards, dones = step_env(loop_state["env_state"], action)
    rewards = loop_state["rewards"].at[step_idx].set(step_rewards)

    return {
        "env_state": next_env_state,
        "obs": obs,
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "masks": masks,
        "logprobs": logprobs,
        "step_idx": step_idx + 1,
        "rng": rng,
    }


@partial(jax.jit, static_argnames=["actor_critic"])
def run_selfplay(env_state, actor_critic, params, rng):
    """
    Collect complete trajectories until all games terminate.

    Args:
        env_state: Dictionary containing the environment state
        actor_critic: ActorCritic model
        params: Model parameters
        rng: JAX random key

    Returns:
        trajectory: dict with collected data (obs, actions, rewards, masks)
        rng: Updated random key
    """
    env_state, obs = reset_env(env_state)

    B, board_size, _ = env_state["boards"].shape
    max_steps = board_size * board_size

    observations, actions, rewards, masks, logprobs = _initialize_trajectory_buffers(
        env_state, max_steps
    )

    loop_state = {
        "env_state": env_state,
        "obs": obs,
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "masks": masks,
        "logprobs": logprobs,
        "step_idx": 0,
        "rng": rng,
    }

    def cond_fn(state):
        return ~jnp.all(state["env_state"]["dones"])

    def body_fn(state):
        return player_move(state, actor_critic, params)

    trajectory = lax.while_loop(cond_fn, body_fn, loop_state)
    trajectory["T"] = trajectory["step_idx"]
    trajectory.pop("step_idx")

    return trajectory, trajectory["rng"]


@partial(jax.jit, static_argnames=["black_actor_critic", "white_actor_critic"])
def run_episode(
    env_state,
    black_actor_critic,
    black_params,
    white_actor_critic,
    white_params,
    rng,
):
    """
    Collect trajectories for self-play with separate black and white models.

    Args:
        env_state: Dictionary containing the environment state
        black_actor_critic: ActorCritic model for black player (first player)
        black_params: Parameters for black player model
        white_actor_critic: ActorCritic model for white player (second player)
        white_params: Parameters for white player model
        rng: JAX RNG key

    Returns:
        black_trajectory: dict with trajectories for black player
        white_trajectory: dict with trajectories for white player
        rng: updated RNG key
    """
    env_state, obs = reset_env(env_state)

    B, board_size, _ = env_state["boards"].shape
    max_steps = board_size * board_size

    observations, actions, rewards, masks, logprobs = _initialize_trajectory_buffers(
        env_state, max_steps
    )
    initial_loop_state = {
        "env_state": env_state,
        "obs": obs,
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "masks": masks,
        "logprobs": logprobs,
        "step_idx": 0,
        "rng": rng,
    }

    def cond_fn(state):
        return ~jnp.all(state["env_state"]["dones"])

    @partial(jit, static_argnames=["black_actor_critic", "white_actor_critic"])
    def body_fn(
        state, black_actor_critic, black_params, white_actor_critic, white_params
    ):
        current_step = state["step_idx"]
        is_black_turn = current_step % 2 == 0

        return jax.lax.cond(
            is_black_turn,
            lambda s: player_move(s, black_actor_critic, black_params),
            lambda s: player_move(s, white_actor_critic, white_params),
            state,
        )

    def body_fn_wrapped(state):
        return body_fn(
            state, black_actor_critic, black_params, white_actor_critic, white_params
        )

    final_state = lax.while_loop(cond_fn, body_fn_wrapped, initial_loop_state)

    env_state = final_state["env_state"]
    step_idx = final_state["step_idx"]
    observations = final_state["observations"]
    actions = final_state["actions"]
    rewards = final_state["rewards"]
    masks = final_state["masks"]
    logprobs = final_state["logprobs"]
    rng = final_state["rng"]

    total_steps = step_idx

    black_trajectory = {
        "obs": observations[::2],  # Shape: ((T+1)//2, B, board_size, board_size)
        "actions": actions[::2],  # Shape: ((T+1)//2, B, 2)
        "rewards": rewards[::2],  # Shape: ((T+1)//2, B)
        "masks": masks[::2],  # Shape: ((T+1)//2, B)
        "logprobs": logprobs[::2],  # Shape: ((T+1)//2, B)
        "T": (total_steps + 1) // 2,  # Shape: ()
    }

    white_trajectory = {
        "obs": observations[1::2],  # Shape: (T//2, B, board_size, board_size)
        "actions": actions[1::2],  # Shape: (T//2, B, 2)
        "rewards": rewards[1::2],  # Shape: (T//2, B)
        "masks": masks[1::2],  # Shape: (T//2, B)
        "logprobs": logprobs[1::2],  # Shape: (T//2, B)
        "T": total_steps // 2,  # Shape: ()
    }

    return black_trajectory, white_trajectory, rng


@jax.jit
def calculate_returns(rewards, gamma):
    """
    Calculate returns at each state for batched trajectories.

    Args:
      rewards: a 2D jnp.array with shape (T, batch)
      gamma: discount factor.

    Returns:
      A jnp.array of discounted returns with the same shape as `rewards`.
    """

    def discount_single(r):
        def scan_fn(carry, reward):
            new_carry = reward + gamma * carry
            return new_carry, new_carry

        _, discounted_reversed = lax.scan(scan_fn, 0.0, r, reverse=True)
        return discounted_reversed

    discounted = jax.vmap(discount_single, in_axes=1, out_axes=1)(rewards)
    return discounted


@jax.jit
def calculate_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """
    Compute Generalized Advantage Estimation (GAE) using scan.
    Assumes trajectories always contain the terminal state if reached.

    Args:
        rewards: Rewards array, shape (steps, batch)
        values: Value estimates, shape (steps, batch)
        dones: Done flags, shape (steps, batch)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: array of shape (steps, batch)
    """
    T = rewards.shape[0] # Number of steps

    # Append a zero value and a non-done flag for the step *after* the last one.
    # This simplifies the scan logic as V(s_{T+1}) is handled implicitly.
    # If dones[T-1] is True, the mask in the scan will handle it.
    values_with_last = jnp.concatenate([values, jnp.zeros_like(values[0:1])], axis=0)

    def gae_scan_fn(carry, step_data):
        """
        Processes a single step in the reverse GAE calculation using lax.scan.
        Operates on data corresponding to a single time step t across the batch.

        Args:
            carry: GAE value from the next step (A_{t+1}) for the batch, shape (B,).
            step_data: Tuple (reward_t, value_t, done_t, next_value_t) for the batch at step t.
              - reward_t: Rewards at step t, shape (B,).
              - value_t: Value estimates V(s_t), shape (B,).
              - done_t: Done flags at step t, shape (B,).
              - next_value_t: Value estimates V(s_{t+1}), shape (B,).

        Returns:
            Tuple (new_carry, output_element), where both are the calculated
            GAE for the current step t (A_t) for the batch, shape (B,).
        """
        gae = carry
        reward, value, done, next_value = step_data # next_value = V(s_{t+1})

        mask = 1.0 - done # Mask based on current step's done flag
        delta = reward + gamma * next_value * mask - value
        gae = delta + gamma * gae_lambda * gae * mask # Use mask here
        return gae, gae

    # Prepare data for scan: (reward_t, value_t, done_t, value_{t+1})
    # We iterate T steps, needing T next_values (values_with_last[1:])
    scan_data = (rewards, values, dones, values_with_last[1:])

    # Initialize carry (gae) to 0.0 for the step after the last actual step.
    # Scan backwards from T-1 down to 0.
    # The carry needs to match the batch dimension shape.
    initial_carry = jnp.zeros(rewards.shape[1]) # Shape (B,)
    _, advantages = jax.lax.scan(gae_scan_fn, initial_carry, scan_data, reverse=True)

    return advantages


def run_pong_episode(env_state, actor_critic, params, rng):
    """
    Collect a trajectory from a single agent interacting with the Pong env.

    Args:
        env_state: Dictionary containing the environment state (Pong env wrapper).
        actor_critic: ActorCritic model (PongActorCritic).
        params: Model parameters.
        rng: JAX random key.

    Returns:
        trajectory: Dictionary with collected data (obs, actions, rewards, dones, values, log_probs).
        final_env_state: The environment state after the episode concludes.
        rng: Updated random key.
    """
    B = env_state["B"]
    rng, reset_rng = jax.random.split(rng)
    env_state, initial_obs = reset_env(env_state, new_rng=reset_rng)


    observations_list = []
    actions_list = []
    rewards_list = []
    dones_list = []
    values_list = []
    log_probs_list = []

    # Initial state for the loop - remove buffer references
    loop_state = {
        "env_state": env_state,
        "current_obs": initial_obs,
        "rng": rng,
        "done": jnp.zeros((B,), dtype=jnp.bool_), # Track if BATCH is done
    }

    # --- Python while loop (condition simplified) --- #
    while not jnp.all(loop_state["done"]):
        rng, action_rng = jax.random.split(loop_state["rng"])
        current_obs = loop_state["current_obs"]

        pi, value = actor_critic.apply({'params': params}, current_obs)
        action = pi.sample(seed=action_rng)
        log_prob = pi.log_prob(action)

        # Step environment
        new_env_state, next_obs, reward, done = step_env(loop_state["env_state"], action)

        # Append current step data to lists
        observations_list.append(current_obs)
        actions_list.append(action)
        rewards_list.append(reward)
        dones_list.append(done)
        values_list.append(value)
        log_probs_list.append(log_prob)

        loop_state["env_state"] = new_env_state
        loop_state["current_obs"] = next_obs
        loop_state["rng"] = rng
        loop_state["done"] = done 


    trajectory_length = len(rewards_list)

    final_env_state = loop_state["env_state"]

    trajectory = {
        "obs": jnp.stack(observations_list, axis=0), # (T, B, 128)
        "actions": jnp.stack(actions_list, axis=0), # (T, B, 2)
        "rewards": jnp.stack(rewards_list, axis=0), # (T, B)
        "dones": jnp.stack(dones_list, axis=0), # (T, B)
        "values": jnp.stack(values_list, axis=0), # (T, B)
        "log_probs": jnp.stack(log_probs_list, axis=0), # (T, B)
        "T": jnp.array(trajectory_length) # Store trajectory length
    }


    return trajectory, final_env_state
