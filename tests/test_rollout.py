import pytest
import jax
import jax.numpy as jnp
import distrax
from functools import partial
from typing import NamedTuple, Tuple, Dict, Any

# Import the module to test
from alphagomoku.training import rollout
from alphagomoku.environments.base import JaxEnvBase, EnvState

# Basic configuration for tests
BOARD_SIZE = 5
BATCH_SIZE = 2
BUFFER_SIZE = 10 # Max episode length for tests
ACTION_DIM = BOARD_SIZE * BOARD_SIZE

# --- Mocks ---

class MockEnvState(NamedTuple):
    board: jnp.ndarray
    current_players: jnp.ndarray
    dones: jnp.ndarray
    step_count: int
    # Add other fields if your actual EnvState requires them for testing

class MockEnv(JaxEnvBase):
    """A simplified mock environment for testing rollout logic."""

    # Add implementations for abstract properties/methods from JaxEnvBase
    @property
    def observation_spec(self) -> Any:
        # Return a dummy spec matching the _get_obs output shape
        return (2, self.board_size, self.board_size)

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        # Return the shape matching observation_spec
        return self.observation_spec

    @property
    def action_spec(self) -> Any:
        # Return a dummy spec matching the action format (row, col)
        return (2,)

    @property
    def action_shape(self) -> Tuple[int, ...]:
        return (2,)

    @property
    def num_actions(self) -> int:
        return self.action_size

    def __init__(self, board_size: int, batch_size: int):
        self.board_size = board_size
        self.batch_size = batch_size
        self.action_size = board_size * board_size

    def reset(self, key: jax.random.PRNGKey) -> Tuple[MockEnvState, jnp.ndarray, Dict]:
        # Simple reset state
        initial_board = jnp.zeros((self.batch_size, self.board_size, self.board_size), dtype=jnp.int32)
        initial_players = jnp.ones((self.batch_size,), dtype=jnp.int32) # Player 1 starts
        initial_dones = jnp.zeros((self.batch_size,), dtype=jnp.bool_)
        initial_step_count = jnp.zeros((self.batch_size,), dtype=jnp.int32)

        initial_state = MockEnvState(
            board=initial_board,
            current_players=initial_players,
            dones=initial_dones,
            step_count=initial_step_count
        )
        # Observation can be simple, e.g., the board itself or features
        initial_obs = self._get_obs(initial_state)
        info = {} # No extra info needed for these tests
        return initial_state, initial_obs, info

    def step(self, state: MockEnvState, action: jnp.ndarray) -> Tuple[MockEnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict]:
        # action shape: (B, 2) [row, col]
        # Simulate a step: place piece, switch player, increment step, check basic done condition
        B = state.board.shape[0]
        # Dummy update board (just mark the chosen cell)
        # Note: In a real env, this would check validity and update board state
        # For simplicity, assume actions are valid for testing rollout
        row_indices = action[:, 0]
        col_indices = action[:, 1]
        batch_indices = jnp.arange(B)

        # Simplified update - does not handle placement logic, just marks
        new_board = state.board.at[batch_indices, row_indices, col_indices].set(state.current_players)

        next_players = 3 - state.current_players # Switch player (1 -> 2, 2 -> 1)
        next_step_count = state.step_count + 1

        # Simple done condition: game ends after 5 steps (for testing termination)
        # Also consider existing dones
        dones = (next_step_count >= 5) | state.dones
        rewards = jnp.where(dones & ~state.dones, 1.0, 0.0) # Reward 1 on first termination

        next_state = MockEnvState(
            board=new_board,
            current_players=next_players,
            dones=dones,
            step_count=next_step_count
        )
        next_obs = self._get_obs(next_state)
        info = {}
        return next_state, next_obs, rewards, dones, info

    def _get_obs(self, state: MockEnvState) -> jnp.ndarray:
        # Simple observation: return board state, add player plane
        player_plane = jnp.ones_like(state.board) * state.current_players[:, None, None]
        return jnp.stack([state.board, player_plane], axis=1) # Shape (B, C, H, W) -> (B, 2, 5, 5)

    def get_action_mask(self, state: MockEnvState) -> jnp.ndarray:
        # Return a mask where all non-occupied cells are valid
        # For simplicity, let's assume all actions are valid initially in tests
        # More realistically: return state.board == 0
        return jnp.ones((self.batch_size, self.board_size, self.board_size), dtype=jnp.bool_)

    def initialize_trajectory_buffers(self, buffer_size: int) -> Tuple[jnp.ndarray, ...]:
        # Based on the expected buffers in rollout.py LoopState
        obs_shape = (buffer_size, self.batch_size, 2, self.board_size, self.board_size) # (T, B, C, H, W)
        action_shape = (buffer_size, self.batch_size, 2) # (T, B, action_dim=2)
        value_shape = (buffer_size, self.batch_size) # (T, B) - Note: GAE needs T+1, handled separately
        reward_shape = (buffer_size, self.batch_size) # (T, B)
        done_shape = (buffer_size, self.batch_size) # (T, B)
        logprob_shape = (buffer_size, self.batch_size) # (T, B)
        player_shape = (buffer_size, self.batch_size) # (T, B)

        observations = jnp.zeros(obs_shape, dtype=jnp.float32)
        actions = jnp.zeros(action_shape, dtype=jnp.int32)
        values = jnp.zeros(value_shape, dtype=jnp.float32)
        rewards = jnp.zeros(reward_shape, dtype=jnp.float32)
        dones = jnp.zeros(done_shape, dtype=jnp.bool_)
        logprobs = jnp.zeros(logprob_shape, dtype=jnp.float32)
        current_players = jnp.zeros(player_shape, dtype=jnp.int32)

        return observations, actions, values, rewards, dones, logprobs, current_players


class MockActorCritic:
    """A mock actor-critic model."""
    def apply(self, params_dict: Dict, obs: jnp.ndarray, current_player: jnp.ndarray) -> Tuple[distrax.Distribution, jnp.ndarray]:
        # params_dict contains {'params': params}
        # obs shape: (B, C, H, W)
        # current_player: (B,) - Not used in this simple mock
        B, C, H, W = obs.shape
        action_dim = H * W

        # Return fixed logits and values for predictability
        # Make logits uniform for simplicity, let masking handle validity
        logits = jnp.zeros((B, action_dim))
        # Return a dummy value, e.g., proportional to step or fixed
        # Example: simple value based on sum of obs (not meaningful, just for testing flow)
        value = jnp.mean(obs, axis=(1, 2, 3)) * 0.1 # Shape (B,)

        pi_dist = distrax.Categorical(logits=logits)
        return pi_dist, value

# --- Pytest Fixtures ---

@pytest.fixture
def mock_env():
    return MockEnv(board_size=BOARD_SIZE, batch_size=BATCH_SIZE)

@pytest.fixture
def mock_actor_critic():
    return MockActorCritic()

@pytest.fixture
def dummy_params():
    # Params can be empty if MockActorCritic doesn't use them
    return {}

@pytest.fixture
def initial_rng():
    return jax.random.PRNGKey(42)

@pytest.fixture
def initial_loop_state(mock_env, mock_actor_critic, dummy_params, initial_rng):
    """Provides a starting LoopState for player_move tests."""
    rng, reset_rng = jax.random.split(initial_rng)
    state, obs, _ = mock_env.reset(reset_rng)
    buffers = mock_env.initialize_trajectory_buffers(BUFFER_SIZE)
    observations, actions, values, rewards, dones, logprobs, current_players = buffers

    initial_termination_indices = jnp.full(
        (BATCH_SIZE,), jnp.iinfo(jnp.int32).max, dtype=jnp.int32
    )

    loop_state = rollout.LoopState(
        state=state,
        obs=obs,
        observations=observations,
        actions=actions,
        values=values,
        rewards=rewards,
        dones=dones,
        logprobs=logprobs,
        current_players=current_players,
        step_idx=0,
        rng=rng,
        termination_step_indices=initial_termination_indices
    )
    return loop_state

# --- Test Functions ---

def test_player_move_updates_state(initial_loop_state, mock_env, mock_actor_critic, dummy_params):
    """Test if player_move correctly updates the environment state and step index."""
    initial_step_idx = initial_loop_state.step_idx
    initial_player = initial_loop_state.state.current_players

    new_loop_state = rollout.player_move(initial_loop_state, mock_env, mock_actor_critic, dummy_params)

    # Check step index increment
    assert new_loop_state.step_idx == initial_step_idx + 1

    # Check player switch (MockEnv switches 1 -> 2)
    assert jnp.all(new_loop_state.state.current_players == 3 - initial_player)

    # Check if observation is updated (mock env returns a simple board state)
    assert not jnp.array_equal(new_loop_state.obs, initial_loop_state.obs)

    # Check if RNG key is updated
    assert not jnp.array_equal(new_loop_state.rng, initial_loop_state.rng)

def test_player_move_stores_data(initial_loop_state, mock_env, mock_actor_critic, dummy_params):
    """Test if player_move correctly stores data in the trajectory buffers."""
    step_idx = initial_loop_state.step_idx
    initial_obs = initial_loop_state.obs
    initial_player = initial_loop_state.state.current_players

    # Get expected value from mock AC
    _, expected_value = mock_actor_critic.apply({"params": dummy_params}, initial_obs, initial_player)

    new_loop_state = rollout.player_move(initial_loop_state, mock_env, mock_actor_critic, dummy_params)

    # Check if data was stored at the correct index
    assert jnp.array_equal(new_loop_state.observations[step_idx], initial_obs)
    assert new_loop_state.actions[step_idx].shape == (BATCH_SIZE, 2) # Check shape
    assert jnp.all(new_loop_state.values[step_idx] == expected_value)
    assert new_loop_state.logprobs[step_idx].shape == (BATCH_SIZE,)
    assert jnp.all(new_loop_state.current_players[step_idx] == initial_player)
    assert new_loop_state.rewards[step_idx].shape == (BATCH_SIZE,)
    assert new_loop_state.dones[step_idx].shape == (BATCH_SIZE,)

def test_player_move_handles_masking(initial_loop_state, mock_env, mock_actor_critic, dummy_params):
    """Test that actions are sampled according to the mask."""
    # Modify the mock env to provide a restrictive mask
    original_mask_fn = mock_env.get_action_mask
    try:
        mask = jnp.zeros((BATCH_SIZE, BOARD_SIZE, BOARD_SIZE), dtype=jnp.bool_)
        # Allow only action (0, 0) for batch 0, and (1, 1) for batch 1
        mask = mask.at[0, 0, 0].set(True)
        mask = mask.at[1, 1, 1].set(True)
        mock_env.get_action_mask = lambda state: mask

        # Run player_move with the restrictive mask
        new_loop_state = rollout.player_move(initial_loop_state, mock_env, mock_actor_critic, dummy_params)

        # Get the action taken at step 0
        action_taken = new_loop_state.actions[0]

        # Check if the actions match the only allowed ones
        expected_action_0 = jnp.array([0, 0])
        expected_action_1 = jnp.array([1, 1])
        assert jnp.array_equal(action_taken[0], expected_action_0)
        assert jnp.array_equal(action_taken[1], expected_action_1)

        # Check logprob is not -inf (valid action was chosen)
        assert jnp.all(new_loop_state.logprobs[0] > -jnp.inf)

    finally:
        # Restore original mask function
        mock_env.get_action_mask = original_mask_fn

def test_player_move_updates_termination_index(initial_loop_state, mock_env, mock_actor_critic, dummy_params):
    """Test that termination index is recorded correctly when done becomes true."""
    # --- First Step (not done) ---
    state_step0 = rollout.player_move(initial_loop_state, mock_env, mock_actor_critic, dummy_params)
    assert jnp.all(state_step0.termination_step_indices == jnp.iinfo(jnp.int32).max)
    assert state_step0.step_idx == 1
    assert not jnp.any(state_step0.state.dones)

    # --- Force done on next step for one env ---
    # Manually set step count high for batch element 0 to trigger done in mock env
    intermediate_state = state_step0.state
    high_step_count = intermediate_state.step_count.at[0].set(4) # Next step will be 5
    intermediate_state = intermediate_state._replace(step_count=high_step_count)
    state_step0 = state_step0._replace(state=intermediate_state)

    # --- Second Step (one env becomes done) ---
    state_step1 = rollout.player_move(state_step0, mock_env, mock_actor_critic, dummy_params)

    # Check termination index: should be step_idx (1) for batch 0, max for batch 1
    expected_term_indices = jnp.array([1, jnp.iinfo(jnp.int32).max], dtype=jnp.int32)
    assert jnp.array_equal(state_step1.termination_step_indices, expected_term_indices)
    assert state_step1.state.dones[0] == True
    assert state_step1.state.dones[1] == False
    assert state_step1.step_idx == 2

    # --- Third Step (other env remains not done) ---
    state_step2 = rollout.player_move(state_step1, mock_env, mock_actor_critic, dummy_params)

    # Termination index for batch 0 should remain 1
    expected_term_indices_2 = jnp.array([1, jnp.iinfo(jnp.int32).max], dtype=jnp.int32)
    assert jnp.array_equal(state_step2.termination_step_indices, expected_term_indices_2)
    assert state_step2.state.dones[0] == True # Stays done
    assert state_step2.state.dones[1] == False
    assert state_step2.step_idx == 3


# Tests for run_episode will go here

def test_run_episode_terminates(mock_env, mock_actor_critic, dummy_params, initial_rng):
    """Test that run_episode runs until all envs are done."""
    # Mock env terminates after 5 steps
    expected_termination_step = 5

    trajectory, final_state, _ = rollout.run_episode(
        env=mock_env,
        black_actor_critic=mock_actor_critic,
        black_params=dummy_params,
        white_actor_critic=mock_actor_critic, # Use same for simplicity
        white_params=dummy_params,
        rng=initial_rng,
        buffer_size=BUFFER_SIZE # Buffer larger than termination step
    )

    # Check if the final state shows done for all environments
    assert jnp.all(final_state.dones)
    # Check the number of steps executed (T)
    # Note: step_idx in LoopState is the *next* step to take, so T is the final value
    assert trajectory["T"] == expected_termination_step

def test_run_episode_alternates_players(mock_env, mock_actor_critic, dummy_params, initial_rng):
    """Test that players alternate correctly during the episode."""
    trajectory, _, _ = rollout.run_episode(
        env=mock_env,
        black_actor_critic=mock_actor_critic,
        black_params=dummy_params,
        white_actor_critic=mock_actor_critic,
        white_params=dummy_params,
        rng=initial_rng,
        buffer_size=BUFFER_SIZE
    )

    T = trajectory["T"]
    current_players = trajectory["current_players"] # Shape (buffer_size, B)

    # Check players stored for executed steps
    assert jnp.all(current_players[0, :] == 1) # Step 0: Black (Player 1)
    if T > 1: assert jnp.all(current_players[1, :] == 2) # Step 1: White (Player 2)
    if T > 2: assert jnp.all(current_players[2, :] == 1) # Step 2: Black (Player 1)
    if T > 3: assert jnp.all(current_players[3, :] == 2) # Step 3: White (Player 2)

def test_run_episode_buffer_shapes(mock_env, mock_actor_critic, dummy_params, initial_rng):
    """Test the shapes of the returned trajectory buffers."""
    trajectory, final_state, _ = rollout.run_episode(
        env=mock_env,
        black_actor_critic=mock_actor_critic,
        black_params=dummy_params,
        white_actor_critic=mock_actor_critic,
        white_params=dummy_params,
        rng=initial_rng,
        buffer_size=BUFFER_SIZE
    )

    T = trajectory["T"] # Actual steps run
    B = BATCH_SIZE
    H, W = BOARD_SIZE, BOARD_SIZE
    OBS_C = 2 # Channels in mock obs

    # Check shapes based on buffer_size, not T
    assert trajectory["observations"].shape == (BUFFER_SIZE, B, OBS_C, H, W)
    assert trajectory["actions"].shape == (BUFFER_SIZE, B, 2)
    # Note: Values buffer in rollout.py seems sized T, not T+1 for GAE directly.
    # The GAE function expects T+1. Let's test the buffer shape as returned first.
    # Correction: The code actually appends the final value, making it T+1 in practice.
    # Let's refine this test based on actual GAE usage later if needed.
    # For now, check the raw buffer size from LoopState as stored.
    # assert trajectory["values"].shape == (BUFFER_SIZE, B) # Based on LoopState buffer
    # UPDATE based on implementation: run_episode calculates and appends final value, but returns the original buffer size
    # The modification happens *after* the loop. Let's verify the *returned* buffer size.
    assert trajectory["values"].shape == (BUFFER_SIZE, B)
    assert trajectory["rewards"].shape == (BUFFER_SIZE, B)
    assert trajectory["dones"].shape == (BUFFER_SIZE, B)
    assert trajectory["logprobs"].shape == (BUFFER_SIZE, B)
    assert trajectory["current_players"].shape == (BUFFER_SIZE, B)
    assert trajectory["valid_mask"].shape == (BUFFER_SIZE, B)
    assert isinstance(trajectory["T"], int) or trajectory["T"].ndim == 0 # Scalar
    assert trajectory["termination_step_indices"].shape == (B,)


def test_run_episode_valid_mask(mock_env, mock_actor_critic, dummy_params, initial_rng):
    """Test the calculation of the valid_mask."""
    # Mock env terminates at step 5 (index 4)
    termination_step_idx = 4 # step_idx when done becomes true

    trajectory, _, _ = rollout.run_episode(
        env=mock_env,
        black_actor_critic=mock_actor_critic,
        black_params=dummy_params,
        white_actor_critic=mock_actor_critic,
        white_params=dummy_params,
        rng=initial_rng,
        buffer_size=BUFFER_SIZE
    )

    valid_mask = trajectory["valid_mask"] # Shape (buffer_size, B)
    term_indices = trajectory["termination_step_indices"] # Shape (B,)
    T = trajectory["T"]

    # In mock env, all batches terminate at the same time (step 5, index 4)
    assert jnp.all(term_indices == termination_step_idx)
    assert T == termination_step_idx + 1

    # Check mask values: True up to and including termination_step_idx, False after
    for b in range(BATCH_SIZE):
        for t in range(BUFFER_SIZE):
            should_be_valid = (t <= termination_step_idx)
            assert valid_mask[t, b] == should_be_valid


def test_run_episode_final_state(mock_env, mock_actor_critic, dummy_params, initial_rng):
    """Test that the correct final environment state is returned."""
    trajectory, final_state, _ = rollout.run_episode(
        env=mock_env,
        black_actor_critic=mock_actor_critic,
        black_params=dummy_params,
        white_actor_critic=mock_actor_critic,
        white_params=dummy_params,
        rng=initial_rng,
        buffer_size=BUFFER_SIZE
    )

    assert isinstance(final_state, MockEnvState) # Or the actual EnvState type
    assert jnp.all(final_state.dones) # Should be done
    assert jnp.all(final_state.step_count == trajectory["T"]) # Step count matches T


# Tests for GAE/Returns will go here

@pytest.mark.parametrize("gamma", [0.99, 1.0])
def test_calculate_returns(gamma):
    """Test discounted return calculation with different gamma values."""
    # T=4, B=2
    rewards = jnp.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    # Env 0 done at step 3 (index 2), Env 1 done at step 4 (index 3)
    dones = jnp.array([
        [False, False],
        [False, False],
        [True, False], # Done after reward at index 2 for env 0
        [False, True]  # Done after reward at index 3 for env 1
    ])
    T = rewards.shape[0]

    returns = rollout.calculate_returns(rewards, dones, gamma)

    # Expected calculations (manual)
    # Env 0: R3=0, R2=1, R1=0+g*1, R0=0+g*(0+g*1)
    # Env 1: R3=1, R2=0+g*1, R1=0+g*(0+g*1), R0=0+g*(0+g*(0+g*1))
    expected_returns = jnp.zeros_like(rewards)
    # Env 0
    expected_returns = expected_returns.at[3, 0].set(0.0)
    expected_returns = expected_returns.at[2, 0].set(1.0)
    expected_returns = expected_returns.at[1, 0].set(gamma * 1.0)
    expected_returns = expected_returns.at[0, 0].set(gamma * gamma * 1.0)
    # Env 1
    expected_returns = expected_returns.at[3, 1].set(1.0)
    expected_returns = expected_returns.at[2, 1].set(gamma * 1.0)
    expected_returns = expected_returns.at[1, 1].set(gamma * gamma * 1.0)
    expected_returns = expected_returns.at[0, 1].set(gamma * gamma * gamma * 1.0)

    assert returns.shape == rewards.shape
    assert jnp.allclose(returns, expected_returns, atol=1e-6)

def test_calculate_returns_immediate_done():
    """Test calculate_returns when an episode ends immediately."""
    gamma = 0.99
    # T=1, B=1
    rewards = jnp.array([[1.0]])
    dones = jnp.array([[True]])
    returns = rollout.calculate_returns(rewards, dones, gamma)
    expected_returns = jnp.array([[1.0]])
    assert jnp.allclose(returns, expected_returns)

@pytest.mark.parametrize("gamma, gae_lambda", [(0.99, 0.95), (1.0, 1.0), (0.9, 0.0)])
def test_calculate_gae(gamma, gae_lambda):
    """Test GAE calculation with different gamma and lambda values."""
    # T=3, B=1
    rewards = jnp.array([[0.0], [0.0], [1.0]])
    # Values V(s0), V(s1), V(s2), V(s_terminal=s3)
    values = jnp.array([[0.1], [0.2], [0.3], [0.0]]) # V(sT) usually 0
    # Dones after step: done at step 3 (index 2)
    dones = jnp.array([[False], [False], [True]])
    T = rewards.shape[0]

    advantages, returns = rollout.calculate_gae(rewards, values, dones, gamma, gae_lambda)

    # Manual GAE calculation
    # delta_t = r_t - gamma * V(s_{t+1}) * (1 - d_t) - V(s_t)  # Corrected logic
    # A_t = delta_t - gamma * lambda * A_{t+1} * (1 - d_t)    # Corrected logic
    v_t = values[:-1] # (T, B)
    v_tp1 = values[1:] # (T, B)
    # Note: The minus signs reflect the zero-sum update
    deltas = rewards - gamma * v_tp1 * (1.0 - dones.astype(jnp.float32)) - v_t # Corrected logic
    # delta[2] = 1.0 - gamma * 0.0 * (1 - 1) - 0.3 = 0.7
    # delta[1] = 0.0 - gamma * 0.3 * (1 - 0) - 0.2 = -0.3*gamma - 0.2
    # delta[0] = 0.0 - gamma * 0.2 * (1 - 0) - 0.1 = -0.2*gamma - 0.1

    adv = jnp.zeros_like(rewards)
    gae_carry = 0.0 # Represents A_{t+1}
    # Iterate backwards from T-1 down to 0
    for t in reversed(range(T)):
        # Note the minus sign for the carry term reflecting the zero-sum game logic
        gae_t = deltas[t, 0] - gamma * gae_lambda * (1.0 - dones[t, 0]) * gae_carry # Corrected logic
        adv = adv.at[t, 0].set(gae_t)
        gae_carry = gae_t # Update carry for next (previous time step) iteration

    expected_advantages = adv
    expected_returns = expected_advantages + v_t # Returns calculation remains R_t = A_t + V(s_t)

    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    assert jnp.allclose(advantages, expected_advantages, atol=1e-6)
    assert jnp.allclose(returns, expected_returns, atol=1e-6)

def test_calculate_gae_shapes():
    """Test shapes returned by calculate_gae."""
    T = 5
    B = 4
    rewards = jnp.zeros((T, B))
    values = jnp.zeros((T + 1, B)) # Need T+1 values
    dones = jnp.zeros((T, B), dtype=bool)

    advantages, returns = rollout.calculate_gae(rewards, values, dones)

    assert advantages.shape == (T, B)
    assert returns.shape == (T, B) 