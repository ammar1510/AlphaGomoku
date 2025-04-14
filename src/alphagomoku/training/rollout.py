import jax
import jax.numpy as jnp
from jax import lax, jit
from functools import partial
from typing import Dict, Any, Tuple, NamedTuple

# Import the base environment class and the type hint for state
from alphagomoku.environments.base import JaxEnvBase, EnvState
# Import the specific Gomoku environment state if needed for type hinting,
# but the functions should ideally work with any EnvState.
# from alphagomoku.environments.gomoku import GomokuState

# Assume actor_critic model has methods: apply, sample_action, log_prob
# These would typically be defined elsewhere (e.g., in a network module)

# Define the LoopState NamedTuple
class LoopState(NamedTuple):
    state: EnvState
    obs: jnp.ndarray
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    logprobs: jnp.ndarray
    step_idx: int
    rng: jax.random.PRNGKey


@partial(jit, static_argnames=["env", "actor_critic"])
def player_move(loop_state: LoopState, env: JaxEnvBase, actor_critic: Any, params: Any) -> LoopState:
    """Takes a single step in the environment using the provided actor-critic."""
    current_state: EnvState = loop_state.state
    current_obs: jnp.ndarray = loop_state.obs
    step_idx: int = loop_state.step_idx

    policy_logits, _ = actor_critic.apply(params, current_obs) # Assume actor_critic outputs logits and value
    action_mask = env.get_action_mask(current_state) # (B, ...) depends on env action space
    masked_logits = jnp.where(action_mask, policy_logits, -jnp.inf)

    rng, subkey = jax.random.split(loop_state.rng)
    action = actor_critic.sample_action(masked_logits, subkey)
    logprob = actor_critic.log_prob(masked_logits, action)

    observations = loop_state.observations.at[step_idx].set(current_obs)
    actions = loop_state.actions.at[step_idx].set(action)
    logprobs = loop_state.logprobs.at[step_idx].set(logprob)

    next_state, next_obs, step_rewards, dones, info = env.step(current_state, action)

    rewards = loop_state.rewards.at[step_idx].set(step_rewards)
    dones_buffer = loop_state.dones.at[step_idx].set(dones)

    return loop_state._replace(
        state=next_state,
        obs=next_obs,
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones_buffer,
        logprobs=logprobs,
        step_idx=step_idx + 1,
        rng=rng,
    )


@partial(jax.jit, static_argnames=["env", "actor_critic", "buffer_size"])
def run_selfplay(env: JaxEnvBase, actor_critic: Any, params: Any, rng: jax.random.PRNGKey, buffer_size: int) -> Tuple[Dict[str, Any], jax.random.PRNGKey]:
    """
    Collect complete trajectories until all games terminate.
    Buffers are allocated based on buffer_size.

    Args:
        env: An instance of JaxEnvBase (e.g., GomokuJaxEnv).
        actor_critic: ActorCritic model instance.
        params: Model parameters.
        rng: JAX random key.
        buffer_size: The size of the trajectory buffers to allocate.

    Returns:
        trajectory: dict with collected data (obs, actions, rewards, dones, logprobs, T)
                    The arrays have length buffer_size, T indicates valid steps.
        rng: Updated random key
    """
    initial_rng, reset_rng = jax.random.split(rng)
    initial_state, initial_obs, _ = env.reset(reset_rng)

    buffers = env.initialize_trajectory_buffers(buffer_size)
    observations, actions, rewards, dones_buffer, logprobs = buffers

    initial_loop_state = LoopState(
        state=initial_state,
        obs=initial_obs,
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones_buffer,
        logprobs=logprobs,
        step_idx=0,
        rng=initial_rng,
    )

    def cond_fn(l_state: LoopState) -> bool:
        return ~jnp.all(l_state.state.dones)
            

    def body_fn(l_state: LoopState) -> LoopState:
        return player_move(l_state, env, actor_critic, params)

    final_loop_state = lax.while_loop(cond_fn, body_fn, initial_loop_state)

    trajectory = {
        "observations": final_loop_state.observations,
        "actions": final_loop_state.actions,
        "rewards": final_loop_state.rewards,
        "dones": final_loop_state.dones,
        "logprobs": final_loop_state.logprobs,
        "T": final_loop_state.step_idx 
    }
    final_rng = final_loop_state.rng

    return trajectory, final_rng


@partial(jax.jit, static_argnames=["env", "black_actor_critic", "white_actor_critic", "buffer_size"])
def run_episode(
    env: JaxEnvBase,
    black_actor_critic: Any,
    black_params: Any,
    white_actor_critic: Any,
    white_params: Any,
    rng: jax.random.PRNGKey,
    buffer_size: int
) -> Tuple[Dict[str, Any], jax.random.PRNGKey]: # Return full buffers, rng
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
        full_trajectory: Dict containing the full, un-sliced buffers (observations, actions, rewards, dones, logprobs).
                         Arrays have length buffer_size.
        rng: updated RNG key.
    """
    initial_rng, reset_rng = jax.random.split(rng)
    initial_state, initial_obs, _ = env.reset(reset_rng)

    buffers = env.initialize_trajectory_buffers(buffer_size)
    observations, actions, rewards, dones_buffer, logprobs = buffers

    initial_loop_state = LoopState(
        state=initial_state,
        obs=initial_obs,
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones_buffer,
        logprobs=logprobs,
        step_idx=0,
        rng=initial_rng,
    )

    def cond_fn(l_state: LoopState) -> bool:
        return ~jnp.all(l_state.state.dones)

    @partial(jit, static_argnames=["env", "black_actor_critic", "white_actor_critic"])
    def body_fn_alternating(
        l_state: LoopState, env: JaxEnvBase,
        black_actor_critic: Any, black_params: Any,
        white_actor_critic: Any, white_params: Any
    ) -> LoopState:
        current_step = l_state.step_idx
        # Check current player from state - more robust than step index if env handles turns
        # Assuming player 1 is black, player -1 is white
        # is_black_turn = l_state.state.current_player == 1 # Check single element if needed jnp.all(l_state["state"].current_player == 1) ? Requires env state structure knowledge
        # Simpler approach using step index assuming strict alternation:
        is_black_turn = (current_step % 2 == 0)

        return jax.lax.cond(
            is_black_turn,
            lambda s: player_move(s, env, black_actor_critic, black_params),
            lambda s: player_move(s, env, white_actor_critic, white_params),
            l_state,
        )

    def body_fn_wrapped(l_state: LoopState) -> LoopState:
        # Pass static args explicitly if needed by jit context, or rely on closure
        return body_fn_alternating(
            l_state, env, black_actor_critic, black_params, white_actor_critic, white_params
        )

    final_state = lax.while_loop(cond_fn, body_fn_wrapped, initial_loop_state)

    full_trajectory = {
        "observations": final_state.observations,
        "actions": final_state.actions,
        "rewards": final_state.rewards,
        "dones": final_state.dones,
        "logprobs": final_state.logprobs,
        "T": final_state.step_idx
    }
    rng = final_state.rng

    return full_trajectory, rng


# --- Utility functions for GAE/Returns (remain the same, check types) ---
@jax.jit
def calculate_returns(rewards: jnp.ndarray, dones: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """
    Calculate discounted returns for batched trajectories.

    Args:
        rewards: Rewards array, shape (T, B).
        dones: Done flags, shape (T, B). Use dones *after* the step.
        gamma: Discount factor.

    Returns:
        Discounted returns, shape (T, B).
    """
    def scan_fn(carry, step_data):
        reward, done = step_data
        # Reset carry to 0 if the episode was done in the *previous* state.
        # However, standard return calculation usually discounts through termination.
        # Let's recalculate standard returns first. GAE handles termination explicitly.
        # If GAE is used, returns might not be needed separately or calculated differently.

        # Standard Discounted Return:
        # new_carry = reward + gamma * carry # Simple version
        # Let's consider dones for GAE/value bootstrapping later.
        # For simple returns, often we discount until the end.
        # Alternative: Reset carry if *previous* step was done.
        # Requires looking at dones shifted by one.

        # Let's implement the standard cumulative discounted return:
        # R_t = r_t + gamma * R_{t+1}
        new_carry = reward + gamma * carry * (1.0 - done) # Reset return if current state is terminal
        return new_carry, new_carry

    # Scan over the time dimension (axis=0) for each batch element (vmap over axis=1)
    def calculate_returns_single(r, d):
        # Flip rewards and dones for reverse scan
        step_data = (jnp.flip(r, axis=0), jnp.flip(d, axis=0))
        _, discounted_reversed = lax.scan(scan_fn, 0.0, step_data)
        # Flip back to original time order
        return jnp.flip(discounted_reversed, axis=0)

    # Apply to each batch element
    returns = jax.vmap(calculate_returns_single, in_axes=1, out_axes=1)(rewards, dones)
    return returns



@jax.jit
def calculate_gae(rewards: jnp.ndarray, values: jnp.ndarray, dones: jnp.ndarray, gamma: float = 0.99, gae_lambda: float = 0.95) -> jnp.ndarray:
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
    assert values.shape[0] == T + 1, f"Values should have shape ({T+1}, B), but got {values.shape}"
    assert values.shape[1] == B, f"Values batch dimension mismatch: {values.shape[1]} vs {B}"
    assert dones.shape[0] == T, f"Dones time dimension mismatch: {dones.shape[0]} vs {T}"
    assert dones.shape[1] == B, f"Dones batch dimension mismatch: {dones.shape[1]} vs {B}"

    values_t = values[:-1] # V(s_0)...V(s_{T-1}), shape (T, B)
    values_tp1 = values[1:] # V(s_1)...V(s_T), shape (T, B)
    dones = dones.astype(jnp.float32) # Ensure float, shape (T, B)

    # Calculate deltas: delta_t = r_t + gamma * V(s_{t+1}) * (1 - d_t) - V(s_t)
    deltas = rewards + gamma * values_tp1 * (1.0 - dones) - values_t # Shape (T, B)

    def scan_fn(carry_gae_batch, step_data_batch):
        # carry_gae_batch: shape (B,)
        # step_data_batch: tuple (delta_batch, done_batch), each shape (B,)
        delta_batch, done_batch = step_data_batch

        # Calculate GAE for the batch: A_t = delta_t + gamma * lambda * A_{t+1} * (1 - d_t)
        # All operations are element-wise across the batch dimension.
        gae_batch = delta_batch + gamma * gae_lambda * (1.0 - done_batch) * carry_gae_batch # Shape (B,)

        # Return the new carry (current GAE) and the value to store (also current GAE)
        return gae_batch, gae_batch

    # Prepare inputs for scan over time axis (0)
    # Scan operates on the leading dimension T.
    scan_inputs = (deltas, dones) # Structure: ((T, B), (T, B))

    # Initial carry state for the scan needs to match the batch dimension
    initial_carry = jnp.zeros(B) # Shape (B,)

    # Scan over axis 0 (time) in reverse.
    # Inputs structure ((T, B), (T, B)), step_data_batch will be ((B,), (B,))
    # Carry has shape (B,). Output ys will have shape (T, B).
    # lax.scan with reverse=True returns outputs in the original order (0..T-1).
    _, advantages = lax.scan(scan_fn, initial_carry, scan_inputs, reverse=True)

    # Calculate returns: R_t = A_t + V(s_t)
    returns = advantages + values_t # Shape (T, B)

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
