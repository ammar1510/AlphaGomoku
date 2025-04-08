import jax
import jax.numpy as jnp
from jax import lax, jit
from functools import partial

from env.gomoku import get_action_mask, reset_env, step_env


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
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Rewards array, shape (steps, batch)
        values: Value estimates, shape (steps, batch)
        dones: Done flags, shape (steps, batch)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: array of shape (steps, batch)
    """

    def gae_single(rewards, values, dones, gamma, gae_lambda):
        def scan_fn(carry, step_data):
            next_advantage, next_value = carry
            reward, value, done = step_data
            delta = reward + gamma * next_value * (1 - done) - value
            advantage = delta + gamma * gae_lambda * next_advantage * (1 - done)

            return (advantage, value), advantage

        step_data = (rewards, values, dones)

        _, advantages = jax.lax.scan(scan_fn, (0.0, 0.0), step_data, reverse=True)
        return advantages

    advantages = jax.vmap(gae_single, in_axes=(1, 1, 1, None, None), out_axes=1)(
        rewards, values, dones, gamma, gae_lambda
    )
    return advantages
