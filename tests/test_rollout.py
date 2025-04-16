import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey
from typing import Any, NamedTuple, Tuple

# Import the components to test
from alphagomoku.training.rollout import (
    run_episode,
    calculate_returns,
    calculate_gae,
    LoopState, # Import if needed for type checking/assertions
)
from alphagomoku.environments.gomoku import GomokuJaxEnv, GomokuState
from alphagomoku.environments.base import JaxEnvBase, EnvState

# --- Constants for Testing ---
TEST_BOARD_SIZE = 5 # Smaller board for faster tests
TEST_BATCH_SIZE = 2
TEST_WIN_LENGTH = 4 # Smaller win length
# Define a buffer size for tests, should be >= max possible steps for done termination tests
TEST_BUFFER_SIZE = TEST_BOARD_SIZE * TEST_BOARD_SIZE

# --- Mock Actor-Critic ---

class MockActorCritic:
    """A simple mock actor-critic for testing rollouts."""
    def __init__(self, env: JaxEnvBase):
        self.board_size = env.board_size
        self.action_shape = env.action_shape # (2,)
        self.num_actions = self.board_size * self.board_size

    def apply(self, params: Any, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns mock logits (uniform) and zero values."""
        batch_size = obs.shape[0]
        # Logits shape should match action mask (B, H, W) for Gomoku
        logits = jnp.zeros((batch_size, self.board_size, self.board_size))
        value = jnp.zeros(batch_size)
        return logits, value

    def sample_action(self, masked_logits: jnp.ndarray, rng: PRNGKey) -> jnp.ndarray:
        """Samples an action randomly from valid moves."""
        batch_size = masked_logits.shape[0]
        flat_logits = masked_logits.reshape(batch_size, -1)
        # Ensure at least one action is valid if possible, otherwise sample randomly (will be invalid move)
        # This mock doesn't need sophisticated handling of no-valid-moves case
        chosen_flat_action = jax.random.categorical(rng, flat_logits)
        chosen_action_unraveled = jnp.unravel_index(chosen_flat_action, (self.board_size, self.board_size))
        chosen_action = jnp.stack(chosen_action_unraveled, axis=-1) # Shape (B, 2)
        return chosen_action.astype(jnp.int32)

    def log_prob(self, masked_logits: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """Calculates log probability based on uniform random sampling over valid actions."""
        batch_size = masked_logits.shape[0]
        action_mask = (masked_logits > -jnp.inf)
        flat_mask = action_mask.reshape(batch_size, -1)
        num_valid_actions = flat_mask.sum(axis=-1, keepdims=True)
        # Avoid division by zero if no actions are valid (e.g., done state)
        num_valid_actions = jnp.maximum(num_valid_actions, 1)
        probs = flat_mask / num_valid_actions

        # Get log probability for the specific action taken
        flat_action_idx = action[:, 0] * self.board_size + action[:, 1]
        log_probs_flat = jnp.log(probs + 1e-9) # Add epsilon for numerical stability
        chosen_log_probs = jnp.take_along_axis(log_probs_flat, flat_action_idx[:, None], axis=-1).squeeze(-1)

        # If no actions were valid initially, log prob should be undefined or very small.
        # Let's mask it where num_valid_actions was originally 0.
        return jnp.where(num_valid_actions.squeeze(-1) > 0, chosen_log_probs, -jnp.inf)


# --- Pytest Fixtures ---

@pytest.fixture
def env() -> GomokuJaxEnv:
    """Provides a Gomoku environment instance for testing."""
    return GomokuJaxEnv(B=TEST_BATCH_SIZE, board_size=TEST_BOARD_SIZE, win_length=TEST_WIN_LENGTH)

@pytest.fixture
def mock_actor_critic(env: GomokuJaxEnv) -> MockActorCritic:
    """Provides a mock actor-critic model instance."""
    return MockActorCritic(env)

@pytest.fixture
def mock_params() -> Any:
    """Provides mock parameters (can be anything if model doesn't use them)."""
    return None # Or {} or any placeholder

@pytest.fixture
def rng() -> PRNGKey:
    """Provides a fixed PRNGKey for reproducibility."""
    return PRNGKey(42)

# --- Test Functions ---

# ---- Tests for run_selfplay (commented out in source) ----

# def test_run_selfplay_execution_and_shapes(env, mock_actor_critic, mock_params, rng):
#     """Tests if run_selfplay executes and returns data with expected shapes."""
#     trajectory, final_rng = run_selfplay(env, mock_actor_critic, mock_params, rng, TEST_BUFFER_SIZE)
#
#     assert isinstance(trajectory, dict)
#     assert final_rng is not None
#     assert not jnp.array_equal(rng, final_rng) # RNG should change
#
#     T = trajectory["T"]
#     assert T > 0 and T <= TEST_BUFFER_SIZE
#
#     # Check shapes of the arrays (should match buffer_size)
#     assert trajectory["observations"].shape == (TEST_BUFFER_SIZE, TEST_BATCH_SIZE) + env.observation_shape
#     assert trajectory["actions"].shape == (TEST_BUFFER_SIZE, TEST_BATCH_SIZE) + env.action_shape
#     assert trajectory["rewards"].shape == (TEST_BUFFER_SIZE, TEST_BATCH_SIZE)
#     assert trajectory["dones"].shape == (TEST_BUFFER_SIZE, TEST_BATCH_SIZE)
#     assert trajectory["logprobs"].shape == (TEST_BUFFER_SIZE, TEST_BATCH_SIZE)
#
#     # Check dtypes
#     assert trajectory["observations"].dtype == jnp.float32
#     assert trajectory["actions"].dtype == jnp.int32
#     assert trajectory["rewards"].dtype == jnp.float32
#     assert trajectory["dones"].dtype == jnp.bool_
#     assert trajectory["logprobs"].dtype == jnp.float32
#
#     # Check that data beyond T is zero (or expected fill value)
#     if T < TEST_BUFFER_SIZE:
#         np.testing.assert_array_equal(trajectory["observations"][T:], jnp.zeros_like(trajectory["observations"][T:]))
#         np.testing.assert_array_equal(trajectory["actions"][T:], jnp.zeros_like(trajectory["actions"][T:]))
#         np.testing.assert_array_equal(trajectory["rewards"][T:], jnp.zeros_like(trajectory["rewards"][T:]))
#         np.testing.assert_array_equal(trajectory["dones"][T:], jnp.zeros_like(trajectory["dones"][T:])) # Buffer is initialized with False
#         np.testing.assert_array_equal(trajectory["logprobs"][T:], jnp.zeros_like(trajectory["logprobs"][T:]))
#
#
# def test_run_selfplay_termination_done(env, mock_actor_critic, mock_params, rng):
#     """Tests if run_selfplay terminates when the environment is done."""
#     # Use a very small win length to force a quick game
#     quick_env = GomokuJaxEnv(B=TEST_BATCH_SIZE, board_size=TEST_BOARD_SIZE, win_length=2)
#     # Create a mock AC for this specific env
#     quick_mock_ac = MockActorCritic(quick_env)
#     # Use a buffer size guaranteed to be large enough
#     buffer_size = quick_env.board_size * quick_env.board_size
#
#     trajectory, _ = run_selfplay(quick_env, quick_mock_ac, mock_params, rng, buffer_size)
#
#     T = trajectory["T"]
#     assert T < buffer_size # Game should finish quickly
#
#     # Check that the last stored done flag corresponds to the step T-1
#     # The trajectory length T means steps 0 to T-1 were executed.
#     # The 'dones' buffer stores the done flag *after* the step.
#     # So, dones[T-1] should contain the terminal state for at least one env.
#     last_dones = trajectory["dones"][T - 1]
#     assert jnp.any(last_dones) # At least one game should be done at step T-1
#
#     # Check if all are done if T < buffer_size (while loop condition)
#     # If T < buffer_size, it implies the loop terminated because jnp.all(dones) was true.
#     if T < buffer_size:
#          assert jnp.all(last_dones)
#
#
# # No max_steps termination test needed as loop primarily terminates on done.
# # The buffer size check is just a safety measure.
#
# def test_run_selfplay_rng_update(env, mock_actor_critic, mock_params, rng):
#     """Tests if the PRNGKey is updated after run_selfplay."""
#     _, final_rng = run_selfplay(env, mock_actor_critic, mock_params, rng, TEST_BUFFER_SIZE)
#     assert not jnp.array_equal(rng, final_rng)
#
#     # Optional: run again and check RNG is different again
#     _, final_rng_2 = run_selfplay(env, mock_actor_critic, mock_params, final_rng, TEST_BUFFER_SIZE)
#     assert not jnp.array_equal(final_rng, final_rng_2)

# ---- Tests for run_episode ----

def test_run_episode_execution_and_shapes(env, mock_actor_critic, mock_params, rng):
    """Tests if run_episode executes and returns data with expected shapes for both players."""
    # Use the same mock AC and params for black and white for simplicity
    full_trajectory, final_rng = run_episode(
        env, mock_actor_critic, mock_params, mock_actor_critic, mock_params, rng, TEST_BUFFER_SIZE
    )

    assert isinstance(full_trajectory, dict)
    total_steps = full_trajectory["T"]
    assert isinstance(total_steps, int) or isinstance(total_steps, jnp.ndarray) # Check type here
    assert final_rng is not None
    assert not jnp.array_equal(rng, final_rng) # RNG should change
    assert total_steps > 0 and total_steps <= TEST_BUFFER_SIZE

    # Check shapes of the full buffers first
    assert full_trajectory["observations"].shape == (TEST_BUFFER_SIZE, TEST_BATCH_SIZE) + env.observation_shape
    assert full_trajectory["actions"].shape == (TEST_BUFFER_SIZE, TEST_BATCH_SIZE) + env.action_shape
    assert full_trajectory["rewards"].shape == (TEST_BUFFER_SIZE, TEST_BATCH_SIZE)
    assert full_trajectory["dones"].shape == (TEST_BUFFER_SIZE, TEST_BATCH_SIZE)
    assert full_trajectory["logprobs"].shape == (TEST_BUFFER_SIZE, TEST_BATCH_SIZE)
    assert "T" in full_trajectory # Check T key exists

    # --- Perform slicing outside the JITted function ---
    black_T = (total_steps + 1) // 2
    white_T = total_steps // 2

    black_indices = jnp.arange(0, total_steps, 2)
    white_indices = jnp.arange(1, total_steps, 2)

    # Extract black player data
    black_trajectory = {
        key: arr[black_indices] for key, arr in full_trajectory.items() if key != "T" # Exclude T during copy
    }
    black_trajectory["T"] = black_T

    # Extract white player data
    white_trajectory = {
        key: arr[white_indices] for key, arr in full_trajectory.items() if key != "T" # Exclude T during copy
    }
    white_trajectory["T"] = white_T
    # --- End Slicing ---

    # Check shapes for black player
    assert black_trajectory["observations"].shape == (black_T, TEST_BATCH_SIZE) + env.observation_shape
    assert black_trajectory["actions"].shape == (black_T, TEST_BATCH_SIZE) + env.action_shape
    assert black_trajectory["rewards"].shape == (black_T, TEST_BATCH_SIZE)
    assert black_trajectory["dones"].shape == (black_T, TEST_BATCH_SIZE)
    assert black_trajectory["logprobs"].shape == (black_T, TEST_BATCH_SIZE)

    # Check dtypes for black player
    assert black_trajectory["observations"].dtype == jnp.float32
    assert black_trajectory["actions"].dtype == jnp.int32
    assert black_trajectory["rewards"].dtype == jnp.float32
    assert black_trajectory["dones"].dtype == jnp.bool_
    assert black_trajectory["logprobs"].dtype == jnp.float32

    # Check shapes for white player
    assert white_trajectory["observations"].shape == (white_T, TEST_BATCH_SIZE) + env.observation_shape
    assert white_trajectory["actions"].shape == (white_T, TEST_BATCH_SIZE) + env.action_shape
    assert white_trajectory["rewards"].shape == (white_T, TEST_BATCH_SIZE)
    assert white_trajectory["dones"].shape == (white_T, TEST_BATCH_SIZE)
    assert white_trajectory["logprobs"].shape == (white_T, TEST_BATCH_SIZE)

    # Check dtypes for white player
    assert white_trajectory["observations"].dtype == jnp.float32
    assert white_trajectory["actions"].dtype == jnp.int32
    assert white_trajectory["rewards"].dtype == jnp.float32
    assert white_trajectory["dones"].dtype == jnp.bool_
    assert white_trajectory["logprobs"].dtype == jnp.float32


def test_run_episode_termination_done(env, mock_actor_critic, mock_params, rng):
    """Tests termination based on done flags."""
    # Use a quick game environment
    quick_env = GomokuJaxEnv(B=TEST_BATCH_SIZE, board_size=TEST_BOARD_SIZE, win_length=2)
    quick_mock_ac = MockActorCritic(quick_env)
    # Use a buffer size guaranteed to be large enough
    buffer_size = quick_env.board_size * quick_env.board_size

    full_traj_done, _ = run_episode(
        quick_env, quick_mock_ac, mock_params, quick_mock_ac, mock_params, rng, buffer_size
    )
    total_T_done = full_traj_done["T"]
    assert total_T_done < buffer_size, "Game should finish before buffer fills"

    # Check last done flag in the full trajectory buffer
    if total_T_done > 0:
        last_dones = full_traj_done["dones"][total_T_done - 1]
        assert jnp.any(last_dones) # Game ended for at least one env
        # Check if all environments were done (loop terminates on all done)
        assert jnp.all(last_dones)

def test_run_episode_rng_update(env, mock_actor_critic, mock_params, rng):
    """Tests if the PRNGKey is updated after run_episode."""
    _, final_rng = run_episode(
        env, mock_actor_critic, mock_params, mock_actor_critic, mock_params, rng, TEST_BUFFER_SIZE
    )
    assert not jnp.array_equal(rng, final_rng)

    # Optional: run again and check RNG is different again
    _, final_rng_2 = run_episode(
        env, mock_actor_critic, mock_params, mock_actor_critic, mock_params, final_rng, TEST_BUFFER_SIZE
    )
    assert not jnp.array_equal(final_rng, final_rng_2)


def test_calculate_returns_simple():
    """Tests calculate_returns with a simple sequence."""
    # Shape (T, B) = (3, 2)
    rewards = jnp.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    dones = jnp.array([[False, False], [False, False], [False, False]])
    gamma = 0.9

    # Expected returns (calculated manually, reverse):
    # R2 = 1.0 + 0.9 * 0 = 1.0
    # R1 = 1.0 + 0.9 * R2 = 1.0 + 0.9 * 1.0 = 1.9
    # R0 = 1.0 + 0.9 * R1 = 1.0 + 0.9 * 1.9 = 1.0 + 1.71 = 2.71
    expected_returns_b0 = jnp.array([2.71, 1.9, 1.0])
    # R2 = 0.0 + 0.9 * 0 = 0.0
    # R1 = 0.0 + 0.9 * R2 = 0.0
    # R0 = 0.0 + 0.9 * R1 = 0.0
    expected_returns_b1 = jnp.array([0.0, 0.0, 0.0])
    expected_returns = jnp.stack([expected_returns_b0, expected_returns_b1], axis=1)

    actual_returns = calculate_returns(rewards, dones, gamma)

    np.testing.assert_allclose(actual_returns, expected_returns, rtol=1e-6)

def test_calculate_returns_with_done():
    """Tests calculate_returns with an episode ending mid-sequence."""
    # Shape (T, B) = (4, 2)
    rewards = jnp.array([[0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]) # R for s0, s1, s2, s3
    dones = jnp.array([[False, False], [False, False], [True, False], [False, True]]) # d for a0, a1, a2, a3 -> s1, s2, s3(term), s4(term)
    gamma = 0.9

    # Expected returns (calculated manually):
    # Batch 0:
    # R3 = 0.0 + 0.9 * 0 * (1-T) = 0.0 (state s4 is terminal due to d[3]=T) - Note: The func impl doesn't use future dones
    # R2 = 1.0 + 0.9 * R3 * (1-F) = 1.0 + 0.9 * 0 = 1.0  (State s3 is terminal, d[2]=T, so returns stop here)
    # R1 = 0.0 + 0.9 * R2 * (1-F) = 0.0 + 0.9 * 1.0 = 0.9
    # R0 = 0.0 + 0.9 * R1 * (1-F) = 0.0 + 0.9 * 0.9 = 0.81
    # expected_returns_b0 = jnp.array([0.81, 0.9, 1.0, 0.0]) # ORIGINAL
    # Scan Logic: R_t = r_t + gamma * R_{t+1} * (1 - d_t)
    # Reverse Scan:
    # carry = 0.0
    # step (r3=0, d3=T): new_carry = 0.0 + 0.9 * 0 * (1-1) = 0.0. Returns = [0.0]
    # step (r2=1, d2=T): new_carry = 1.0 + 0.9 * 0 * (1-1) = 1.0. Returns = [1.0, 0.0]
    # step (r1=0, d1=F): new_carry = 0.0 + 0.9 * 1 * (1-0) = 0.9. Returns = [0.9, 1.0, 0.0]
    # step (r0=0, d0=F): new_carry = 0.0 + 0.9 * 0.9 * (1-0) = 0.81. Returns = [0.81, 0.9, 1.0, 0.0]
    expected_returns_b0 = jnp.array([0.81, 0.9, 1.0, 0.0])

    # Batch 1:
    # R3 = 0.0 + 0.9 * 0 * (1-T) = 0.0 (State s4 terminal)
    # R2 = 1.0 + 0.9 * R3 * (1-F) = 1.0 + 0.9 * 0 = 1.0
    # R1 = 1.0 + 0.9 * R2 * (1-F) = 1.0 + 0.9 * 1.0 = 1.9
    # R0 = 1.0 + 0.9 * R1 * (1-F) = 1.0 + 0.9 * 1.9 = 1.0 + 1.71 = 2.71
    # expected_returns_b1 = jnp.array([2.71, 1.9, 1.0, 0.0]) # ORIGINAL
    # Scan Logic: R_t = r_t + gamma * R_{t+1} * (1 - d_t)
    # Reverse Scan:
    # carry = 0.0
    # step (r3=0, d3=T): new_carry = 0.0 + 0.9 * 0 * (1-1) = 0.0. Returns = [0.0]
    # step (r2=1, d2=F): new_carry = 1.0 + 0.9 * 0 * (1-0) = 1.0. Returns = [1.0, 0.0]
    # step (r1=1, d1=F): new_carry = 1.0 + 0.9 * 1 * (1-0) = 1.9. Returns = [1.9, 1.0, 0.0]
    # step (r0=1, d0=F): new_carry = 1.0 + 0.9 * 1.9 * (1-0) = 1.0 + 1.71 = 2.71. Returns = [2.71, 1.9, 1.0, 0.0]
    expected_returns_b1 = jnp.array([2.71, 1.9, 1.0, 0.0])

    expected_returns = jnp.stack([expected_returns_b0, expected_returns_b1], axis=1)
    actual_returns = calculate_returns(rewards, dones, gamma)
    np.testing.assert_allclose(actual_returns, expected_returns, rtol=1e-6)


def test_calculate_gae_simple():
    """Tests calculate_gae with a simple sequence."""
    # Shape (T, B) = (3, 1)
    rewards = jnp.array([[1.0], [1.0], [1.0]])
    dones = jnp.array([[False], [False], [False]])
    # values need shape (T+1, B) = (4, 1)
    values = jnp.array([[0.5], [0.6], [0.7], [0.8]]) # V(s0), V(s1), V(s2), V(s3)
    gamma = 0.9
    gae_lambda = 0.95

    # Manual calculation (reverse):
    # t=2:
    # delta_2 = r_2 + gamma * V(s3) * (1-d2) - V(s2)
    #         = 1.0 + 0.9 * 0.8 * (1-0) - 0.7 = 1.0 + 0.72 - 0.7 = 1.02
    # gae_2   = delta_2 + gamma * lambda * (1-d2) * gae_3 (assume gae_3=0)
    #         = 1.02 + 0.9 * 0.95 * (1-0) * 0 = 1.02
    # t=1:
    # delta_1 = r_1 + gamma * V(s2) * (1-d1) - V(s1)
    #         = 1.0 + 0.9 * 0.7 * (1-0) - 0.6 = 1.0 + 0.63 - 0.6 = 1.03
    # gae_1   = delta_1 + gamma * lambda * (1-d1) * gae_2
    #         = 1.03 + 0.9 * 0.95 * (1-0) * 1.02 = 1.03 + 0.855 * 1.02 = 1.03 + 0.8721 = 1.9021
    # t=0:
    # delta_0 = r_0 + gamma * V(s1) * (1-d0) - V(s0)
    #         = 1.0 + 0.9 * 0.6 * (1-0) - 0.5 = 1.0 + 0.54 - 0.5 = 1.04
    # gae_0   = delta_0 + gamma * lambda * (1-d0) * gae_1
    #         = 1.04 + 0.9 * 0.95 * (1-0) * 1.9021 = 1.04 + 0.855 * 1.9021 = 1.04 + 1.6262955 = 2.6662955

    expected_advantages = jnp.array([[2.6662955], [1.9021], [1.02]])
    expected_returns = expected_advantages + values[:-1] # Add V(s0), V(s1), V(s2)

    actual_advantages, actual_returns = calculate_gae(rewards, values, dones, gamma, gae_lambda)

    np.testing.assert_allclose(actual_advantages, expected_advantages, rtol=1e-5)
    np.testing.assert_allclose(actual_returns, expected_returns, rtol=1e-5)


def test_calculate_gae_with_done():
    """Tests calculate_gae with episode termination."""
     # Shape (T, B) = (3, 1)
    rewards = jnp.array([[1.0], [1.0], [1.0]])
    dones = jnp.array([[False], [False], [True]]) # Episode ends after step 2 (action a2 leads to s3 which is terminal)
    # values need shape (T+1, B) = (4, 1)
    values = jnp.array([[0.5], [0.6], [0.7], [0.0]]) # V(s0), V(s1), V(s2), V(s3)=0 because s3 is terminal
    gamma = 0.9
    gae_lambda = 0.95

    # Manual calculation (reverse):
    # t=2:
    # delta_2 = r_2 + gamma * V(s3) * (1-d2) - V(s2)
    #         = 1.0 + 0.9 * 0.0 * (1-1) - 0.7 = 1.0 + 0 - 0.7 = 0.3
    # gae_2   = delta_2 + gamma * lambda * (1-d2) * gae_3 (assume gae_3=0)
    #         = 0.3 + 0.9 * 0.95 * (1-1) * 0 = 0.3
    # t=1:
    # delta_1 = r_1 + gamma * V(s2) * (1-d1) - V(s1)
    #         = 1.0 + 0.9 * 0.7 * (1-0) - 0.6 = 1.0 + 0.63 - 0.6 = 1.03
    # gae_1   = delta_1 + gamma * lambda * (1-d1) * gae_2
    #         = 1.03 + 0.9 * 0.95 * (1-0) * 0.3 = 1.03 + 0.855 * 0.3 = 1.03 + 0.2565 = 1.2865
    # t=0:
    # delta_0 = r_0 + gamma * V(s1) * (1-d0) - V(s0)
    #         = 1.0 + 0.9 * 0.6 * (1-0) - 0.5 = 1.0 + 0.54 - 0.5 = 1.04
    # gae_0   = delta_0 + gamma * lambda * (1-d0) * gae_1
    #         = 1.04 + 0.9 * 0.95 * (1-0) * 1.2865 = 1.04 + 0.855 * 1.2865 = 1.04 + 1.0999575 = 2.1399575

    expected_advantages = jnp.array([[2.1399575], [1.2865], [0.3]])
    expected_returns = expected_advantages + values[:-1] # Add V(s0), V(s1), V(s2)

    actual_advantages, actual_returns = calculate_gae(rewards, values, dones, gamma, gae_lambda)

    np.testing.assert_allclose(actual_advantages, expected_advantages, rtol=1e-5)
    np.testing.assert_allclose(actual_returns, expected_returns, rtol=1e-5)


def test_calculate_gae_shapes():
    """Tests the output shapes of calculate_gae."""
    T, B = 5, 4
    rewards = jnp.zeros((T, B))
    dones = jnp.zeros((T, B), dtype=bool)
    values = jnp.zeros((T + 1, B)) # Needs T+1 time steps
    gamma = 0.99
    gae_lambda = 0.95

    advantages, returns = calculate_gae(rewards, values, dones, gamma, gae_lambda)

    assert advantages.shape == (T, B)
    assert returns.shape == (T, B)
    assert advantages.dtype == jnp.float32 # Or float64 depending on precision
    assert returns.dtype == jnp.float32


def test_calculate_gae_batch():
    """Tests calculate_gae with a batch size > 1."""
    # Based on test_calculate_gae_simple, duplicated for B=2
    T = 3
    B = 2
    # Shape (T, B) = (3, 2)
    rewards = jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    dones = jnp.array([[False, False], [False, False], [False, False]])
    # values need shape (T+1, B) = (4, 2)
    values = jnp.array([[0.5, 0.5], [0.6, 0.6], [0.7, 0.7], [0.8, 0.8]]) # V(s0), V(s1), V(s2), V(s3)
    gamma = 0.9
    gae_lambda = 0.95

    # Expected results are the same as the simple case, just duplicated
    expected_advantages_single = jnp.array([[2.6662955], [1.9021], [1.02]])
    expected_advantages = jnp.concatenate([expected_advantages_single] * B, axis=1)

    expected_returns_single = expected_advantages_single + values[:-1, 0:1] # Add V(s0), V(s1), V(s2) for one batch item
    expected_returns = jnp.concatenate([expected_returns_single] * B, axis=1)

    actual_advantages, actual_returns = calculate_gae(rewards, values, dones, gamma, gae_lambda)

    assert actual_advantages.shape == (T, B)
    assert actual_returns.shape == (T, B)

    np.testing.assert_allclose(actual_advantages, expected_advantages, rtol=1e-5)
    np.testing.assert_allclose(actual_returns, expected_returns, rtol=1e-5)


# TODO: Add tests for run_selfplay and run_episode if they stabilize


