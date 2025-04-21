import jax
import jax.numpy as jnp
from jax import lax, jit
from functools import partial
from typing import Dict, Any, Tuple, NamedTuple
import distrax  # Added import

from alphagomoku.environments.base import JaxEnvBase, EnvState


class LoopState(NamedTuple):
    state: EnvState
    obs: jnp.ndarray
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    logprobs: jnp.ndarray
    current_players: jnp.ndarray  
    step_idx: int
    rng: jax.random.PRNGKey
    termination_step_indices: (
        jnp.ndarray
    )  # Stores the step index 't' when done first becomes True for each batch element


@partial(jit, static_argnames=["env", "actor_critic"])
def player_move(
    loop_state: LoopState, env: JaxEnvBase, actor_critic: Any, params: Any
) -> LoopState:
    """Takes a single step in the environment using the provided actor-critic."""
    current_state: EnvState = loop_state.state
    current_obs: jnp.ndarray = loop_state.obs
    current_player: jnp.ndarray = (
        current_state.current_players
    ) 
    step_idx: int = loop_state.step_idx
    rng = loop_state.rng

    # Get policy distribution and value from the model
    pi_dist, _ = actor_critic.apply(
        {"params": params}, current_obs, current_player
    ) 

    # Get action mask from the environment
    action_mask = env.get_action_mask(current_state)  # (B, H, W)
    B, H, W = action_mask.shape
    flat_action_mask = action_mask.reshape(B, -1)  # (B, H*W)

    # Get original logits from the distribution
    original_logits = pi_dist.logits  # Shape (B, H*W)

    # Apply the mask to the logits
    masked_logits = jnp.where(flat_action_mask, original_logits, -jnp.inf)

    # Create a new distribution with masked logits
    masked_pi_dist = distrax.Categorical(logits=masked_logits)

    # Sample action from the masked distribution
    rng, subkey = jax.random.split(rng)
    flat_action = masked_pi_dist.sample(seed=subkey)  # Shape (B,)
    logprob = masked_pi_dist.log_prob(flat_action)  # Shape (B,)

    # Convert flat action back to (row, col) for the environment step
    action_row = flat_action // W
    action_col = flat_action % W
    action = jnp.stack([action_row, action_col], axis=-1)  # Shape (B, 2)

    observations = loop_state.observations.at[step_idx].set(current_obs)
    actions = loop_state.actions.at[step_idx].set(action)
    logprobs = loop_state.logprobs.at[step_idx].set(logprob)
    current_players = loop_state.current_players.at[step_idx].set(current_player)

    next_state, next_obs, step_rewards, dones, info = env.step(current_state, action)

    rewards = loop_state.rewards.at[step_idx].set(step_rewards)
    dones_buffer = loop_state.dones.at[step_idx].set(dones)

    # Update termination indices: if not already terminated and current step is done, record step_idx
    current_termination_indices = loop_state.termination_step_indices
    not_terminated_yet = current_termination_indices == jnp.iinfo(jnp.int32).max
    new_termination_indices = jnp.where(
        not_terminated_yet & dones,
        step_idx,  # Record current step index as termination step
        current_termination_indices,  # Keep existing index (either max or previously recorded step)
    )

    return loop_state._replace(
        state=next_state,
        obs=next_obs,
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones_buffer,
        logprobs=logprobs,
        current_players=current_players,
        step_idx=step_idx + 1,
        rng=rng,
        termination_step_indices=new_termination_indices,
    )


# @partial(jax.jit, static_argnames=["env", "actor_critic", "buffer_size"])
# def run_selfplay(env: JaxEnvBase, actor_critic: Any, params: Any, rng: jax.random.PRNGKey, buffer_size: int) -> Tuple[Dict[str, Any], jax.random.PRNGKey]:
#     """
#     Collect complete trajectories until all games terminate.
#     Buffers are allocated based on buffer_size.
#
#     Args:
#         env: An instance of JaxEnvBase (e.g., GomokuJaxEnv).
#         actor_critic: ActorCritic model instance.
#         params: Model parameters.
#         rng: JAX random key.
#         buffer_size: The size of the trajectory buffers to allocate.
#
#     Returns:
#         trajectory: dict with collected data (obs, actions, rewards, dones, logprobs, T)
#                     The arrays have length buffer_size, T indicates valid steps.
#         rng: Updated random key
#     """
#     initial_rng, reset_rng = jax.random.split(rng)
#     initial_state, initial_obs, _ = env.reset(reset_rng)
#
#     buffers = env.initialize_trajectory_buffers(buffer_size)
#     observations, actions, rewards, dones_buffer, logprobs = buffers
#
#     initial_loop_state = LoopState(
#         state=initial_state,
#         obs=initial_obs,
#         observations=observations,
#         actions=actions,
#         rewards=rewards,
#         dones=dones_buffer,
#         logprobs=logprobs,
#         step_idx=0,
#         rng=initial_rng,
#     )
#
#     def cond_fn(l_state: LoopState) -> bool:
#         return ~jnp.all(l_state.state.dones)
#
#
#     def body_fn(l_state: LoopState) -> LoopState:
#         return player_move(l_state, env, actor_critic, params)
#
#     final_loop_state = lax.while_loop(cond_fn, body_fn, initial_loop_state)
#
#     trajectory = {
#         "observations": final_loop_state.observations,
#         "actions": final_loop_state.actions,
#         "rewards": final_loop_state.rewards,
#         "dones": final_loop_state.dones,
#         "logprobs": final_loop_state.logprobs,
#         "T": final_loop_state.step_idx
#     }
#     final_rng = final_loop_state.rng
#
#     return trajectory, final_rng


@partial(
    jax.jit,
    static_argnames=["env", "black_actor_critic", "white_actor_critic", "buffer_size"],
)
def run_episode(
    env: JaxEnvBase,
    black_actor_critic: Any,
    black_params: Any,
    white_actor_critic: Any,
    white_params: Any,
    rng: jax.random.PRNGKey,
    buffer_size: int,
) -> Tuple[
    Dict[str, Any], EnvState, jax.random.PRNGKey
]:  # Return full buffers, final_state, rng
    """
    Collect trajectories for self-play with separate black and white models.
    Runs until all environments are done.
    Buffers are allocated based on buffer_size.

    Args:
        env: An instance of JaxEnvBase. Assumes player 1 is "black".
        black_actor_critic: ActorCritic model for black player (first player).
        black_params: Parameters for black player model.
        white_actor_critic: ActorCritic model for white player (second player).
        white_params: Parameters for white player model.
        rng: JAX RNG key.
        buffer_size: The size of the trajectory buffers to allocate.

    Returns:
        full_trajectory: Dict containing the full, un-sliced buffers (observations, actions, rewards, dones, logprobs, current_players, valid_mask, T, termination_indices).
                         Arrays have length buffer_size.
        final_state: The environment state after the final step taken in the loop.
        rng: updated RNG key.

    Note: This function can simulate the behavior of `run_selfplay` function
          by passing the same actor_critic model and params for both the black and white players.
    """
    initial_rng, reset_rng = jax.random.split(rng)
    initial_state, initial_obs, _ = env.reset(reset_rng)

    buffers = env.initialize_trajectory_buffers(buffer_size)
    observations, actions, rewards, dones_buffer, logprobs, current_players_buffer = (
        buffers  # Unpack players buffer
    )
    B = initial_obs.shape[0]  # Infer batch size
    initial_termination_indices = jnp.full(
        (B,), jnp.iinfo(jnp.int32).max, dtype=jnp.int32
    )

    initial_loop_state = LoopState(
        state=initial_state,
        obs=initial_obs,
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones_buffer,
        logprobs=logprobs,
        current_players=current_players_buffer, 
        step_idx=0,
        rng=initial_rng,
        termination_step_indices=initial_termination_indices,  # Initialize termination indices
    )

    def cond_fn(l_state: LoopState) -> bool:
        return ~jnp.all(l_state.state.dones)

    @partial(jit, static_argnames=["env", "black_actor_critic", "white_actor_critic"])
    def body_fn_alternating(
        l_state: LoopState,
        env: JaxEnvBase,
        black_actor_critic: Any,
        black_params: Any,
        white_actor_critic: Any,
        white_params: Any,
    ) -> LoopState:
        current_step = l_state.step_idx
        is_black_turn = current_step % 2 == 0

        return jax.lax.cond(
            is_black_turn,
            lambda s: player_move(s, env, black_actor_critic, black_params),
            lambda s: player_move(s, env, white_actor_critic, white_params),
            l_state,
        )

    def body_fn_wrapped(l_state: LoopState) -> LoopState:
        # Pass static args explicitly if needed by jit context, or rely on closure
        return body_fn_alternating(
            l_state,
            env,
            black_actor_critic,
            black_params,
            white_actor_critic,
            white_params,
        )

    final_state = lax.while_loop(cond_fn, body_fn_wrapped, initial_loop_state)

    term_indices = final_state.termination_step_indices  # Shape (B,)
    T = final_state.step_idx  # Use actual steps taken up to buffer_size
    B = initial_obs.shape[0]

    # Ensure T is used correctly for the mask dimensions even if less than buffer_size
    step_indices = jnp.arange(buffer_size)[:, None]  # Shape (buffer_size, 1)

    # Broadcast comparison: mask is True if step_index <= termination_index
    # Using '<=' ensures the terminal step itself is included as valid
    # We create a mask for the full buffer size
    valid_mask = step_indices <= term_indices[None, :]  # Shape (buffer_size, B)

    full_trajectory = {
        # Use the full buffers
        "observations": final_state.observations,  # Shape (buffer_size, B, ...)
        "actions": final_state.actions,  # Shape (buffer_size, B, ...)
        "rewards": final_state.rewards,  # Shape (buffer_size, B)
        "dones": final_state.dones,  # Shape (buffer_size, B)
        "logprobs": final_state.logprobs,  # Shape (buffer_size, B)
        "current_players": final_state.current_players,  # Add stored players
        "valid_mask": valid_mask,  # Add the calculated mask, shape (buffer_size, B)
        "T": T,  # Actual number of steps executed (can be less than buffer_size)
        "termination_step_indices": final_state.termination_step_indices,  # Keep this too if needed elsewhere
    }
    rng = final_state.rng

    # Return the final EnvState directly, contains final_obs and final_player
    return full_trajectory, final_state.state, rng


# --- Utility functions for GAE/Returns (remain the same, check types) ---
@jax.jit
def calculate_returns(
    rewards: jnp.ndarray, dones: jnp.ndarray, gamma: float
) -> jnp.ndarray:
    """
    Calculate discounted returns for batched trajectories.

    Args:
        rewards: Rewards array, shape (T, B).
        dones: Done flags, shape (T, B). Use dones *after* the step.
        gamma: Discount factor.

    Returns:
        Discounted returns, shape (T, B).
    """
    B = rewards.shape[1]
    dones = dones.astype(jnp.float32)  # Ensure float

    def scan_fn_batch(carry_batch, step_data_batch):
        reward_batch, done_batch = step_data_batch

        new_carry_batch = reward_batch + gamma * carry_batch * (1.0 - done_batch)

        return new_carry_batch, new_carry_batch

    scan_inputs = (rewards, dones)  # Structure: ((T, B), (T, B))

    initial_carry = jnp.zeros(B)  # Shape (B,)

    _, returns = lax.scan(scan_fn_batch, initial_carry, scan_inputs, reverse=True)

    return returns


@jax.jit
def calculate_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> jnp.ndarray:
    """
    Compute Generalized Advantage Estimation (GAE) using lax.scan directly on batched data.

    Args:
        rewards: Rewards array, shape (T, B).
        values: Value estimates, shape (T+1, B). Include value of *terminal* state.
        dones: Done flags, shape (T, B). Dones resulting from the action at step t.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.

    Returns:
        advantages: GAE advantages, shape (T, B).
        returns: GAE-based returns (advantages + values), shape (T, B).
    """
    T = rewards.shape[0]
    B = rewards.shape[1]
    assert (
        values.shape[0] == T + 1
    ), f"Values should have shape ({T+1}, B), but got {values.shape}"
    assert (
        values.shape[1] == B
    ), f"Values batch dimension mismatch: {values.shape[1]} vs {B}"
    assert (
        dones.shape[0] == T
    ), f"Dones time dimension mismatch: {dones.shape[0]} vs {T}"
    assert (
        dones.shape[1] == B
    ), f"Dones batch dimension mismatch: {dones.shape[1]} vs {B}"

    values_t = values[:-1]  # V(s_0)...V(s_{T-1}), shape (T, B)
    values_tp1 = values[1:]  # V(s_1)...V(s_T), shape (T, B)
    dones = dones.astype(jnp.float32)  # Ensure float, shape (T, B)

    # Calculate deltas: delta_t = r_t - gamma * V(s_{t+1}) * (1 - d_t) - V(s_t)
    # minus sign as the next value is wrt to opponent
    # not sure if this is correct
    deltas = rewards - gamma * values_tp1 * (1.0 - dones) - values_t  # Shape (T, B)

    def scan_fn(carry_gae_batch, step_data_batch):
        # carry_gae_batch: shape (B,)
        # step_data_batch: tuple (delta_batch, done_batch), each shape (B,)
        delta_batch, done_batch = step_data_batch

        # Calculate GAE for the batch: A_t = delta_t + gamma * lambda * A_{t+1} * (1 - d_t)
        # All operations are element-wise across the batch dimension.
        gae_batch = (
            delta_batch + gamma * gae_lambda * (1.0 - done_batch) * carry_gae_batch
        )  # Shape (B,)

        # Return the new carry (current GAE) and the value to store (also current GAE)
        return gae_batch, gae_batch

    # Prepare inputs for scan over time axis (0)
    # Scan operates on the leading dimension T.
    scan_inputs = (deltas, dones)  # Structure: ((T, B), (T, B))

    # Initial carry state for the scan needs to match the batch dimension
    initial_carry = jnp.zeros(B)  # Shape (B,)

    # Scan over axis 0 (time) in reverse.
    # Inputs structure ((T, B), (T, B)), step_data_batch will be ((B,), (B,))
    # Carry has shape (B,). Output ys will have shape (T, B).
    # lax.scan with reverse=True processes inputs from T-1 down to 0,
    # but returns the collected outputs in the original order (0..T-1).
    _, advantages = lax.scan(scan_fn, initial_carry, scan_inputs, reverse=True)

    # Calculate returns: R_t = A_t + V(s_t)
    returns = advantages + values_t  # Shape (T, B)

    return advantages, returns


# Keep the old GAE implementation commented out for reference if needed
# @jax.jit
# def calculate_gae_scan(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
#     """Compute GAE using lax.scan."""
#     # Requires values to be (T+1, B) including V(s_T)
#     T = rewards.shape[0]
#     assert values.shape[0] == T + 1
#     values_t = values[:-1] # V(s_0) to V(s_{T-1})
#     values_tp1 = values[1:] # V(s_1) to V(s_T)
#     dones = dones.astype(jnp.float32)

#     deltas = rewards + gamma * values_tp1 * (1.0 - dones) - values_t

#     def scan_fn(carry_gae, step_data):
#         delta, done = step_data
#         gae = delta + gamma * gae_lambda * (1.0 - done) * carry_gae
#         return gae, gae

#     # Scan backwards needs reversed data
#     reversed_deltas = jnp.flip(deltas, axis=0)
#     reversed_dones = jnp.flip(dones, axis=0)
#     reversed_step_data = (reversed_deltas, reversed_dones)

#     # vmap scan_fn over batch dimension
#     def scan_fn_batch(carry_gae_batch, step_data_batch):
#         delta_batch, done_batch = step_data_batch
#         gae_batch = delta_batch + gamma * gae_lambda * (1.0 - done_batch) * carry_gae_batch
#         return gae_batch, gae_batch

#     initial_carry = jnp.zeros(rewards.shape[1]) # Zeros for batch dimension
#     _, reversed_advantages = lax.scan(scan_fn_batch, initial_carry, reversed_step_data)

#     advantages = jnp.flip(reversed_advantages, axis=0)
#     returns = advantages + values_t
#     return advantages, returns
