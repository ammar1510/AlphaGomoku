# tests/test_rollout.py
"""
Unit tests for the rollout functions in training.rollout.
Uses mocked environment and actor-critic models to isolate the rollout logic.
"""

import pytest
import jax
import jax.numpy as jnp
from flax.core import FrozenDict

# Functions under test
from training.rollout import (
    run_selfplay,
    run_episode,
    calculate_returns,
    calculate_gae,
)


# ==============================================================================
# == Mocks ==
# ==============================================================================
# Mock implementations of dependencies (ActorCritic model, Environment functions)
# to isolate the rollout functions during testing.


class MockActorCritic:
    """A simplified mock ActorCritic model for testing rollouts."""

    def apply(self, params, obs):
        """Returns fixed dummy logits and value estimates."""
        B = obs.shape[0]
        board_size = obs.shape[1]
        # Simple uniform logits, favoring action (0,0) slightly
        dummy_logits = jnp.zeros((B, board_size, board_size))
        dummy_logits = dummy_logits.at[:, 0, 0].set(1.0)
        dummy_value = jnp.ones(B) * 0.5
        return dummy_logits, dummy_value

    def sample_action(self, logits, rng_key):
        """Deterministically selects the action with the highest logit."""
        B = logits.shape[0]
        board_size = logits.shape[1]
        flat_logits = logits.reshape(B, -1)
        flat_action_idx = jnp.argmax(flat_logits, axis=-1)
        action_row = flat_action_idx // board_size
        action_col = flat_action_idx % board_size
        action = jnp.stack([action_row, action_col], axis=-1)
        return action

    def log_prob(self, logits, action):
        """Returns a fixed dummy log probability."""
        # Since sample_action is deterministic argmax, we could calculate
        # the actual log prob, but for testing rollout structure, a fixed
        # value is sufficient.
        B = logits.shape[0]
        return jnp.zeros(B) - 0.1  # Fixed dummy logprob

    def evaluate_actions(self, params, states, actions):
        """Returns dummy log probs, entropy, and values for evaluation."""
        # Needed for PPO trainer tests (though not directly by rollout funcs)
        T, B = states.shape[0], states.shape[1]
        action_log_probs = jnp.zeros((T, B)) - 1.0
        entropy = jnp.ones((T, B)) * 1.5
        values = jnp.ones((T, B)) * 0.5
        return action_log_probs, entropy, values


def mock_reset_env(env_state):
    """Mock environment reset function. Returns a zeroed-out observation."""
    initial_obs = jnp.zeros_like(env_state["boards"], dtype=jnp.float32)
    env_state["boards"] = jnp.zeros_like(env_state["boards"])
    env_state["dones"] = jnp.zeros_like(env_state["dones"])
    env_state["current_player"] = jnp.ones_like(env_state["current_player"])
    env_state["steps"] = 0
    return env_state, initial_obs


def mock_step_env(env_state, action):
    """Mock environment step function. Ends game after fixed steps."""
    max_mock_steps = 5  # Fixed episode length for mock environment
    env_state["steps"] += 1
    # Game ends if max steps reached or already done
    dones = (env_state["steps"] >= max_mock_steps) | env_state["dones"]
    # Reward 1.0 only on the step the game transitions to done
    rewards = jnp.where(dones & ~env_state["dones"], 1.0, 0.0)
    env_state["dones"] = dones
    env_state["current_player"] = -env_state["current_player"]  # Flip player
    # Dummy next observation changes with step number
    next_obs = (
        jnp.ones_like(env_state["boards"], dtype=jnp.float32) * env_state["steps"]
    )
    return env_state, next_obs, rewards, dones


def mock_get_action_mask(env_state):
    """Mock action mask function. Allows all actions."""
    return jnp.ones_like(env_state["boards"], dtype=jnp.bool_)


# ==============================================================================
# == Pytest Fixtures ==
# ==============================================================================
# Fixtures provide reusable setup code (mock instances, data) for tests.


@pytest.fixture
def mock_actor_critic_instance():
    """Provides an instance of the MockActorCritic."""
    return MockActorCritic()


@pytest.fixture
def mock_params():
    """Provides empty mock parameters (sufficient for MockActorCritic)."""
    return FrozenDict({})


@pytest.fixture
def mock_env_state():
    """Provides a sample initial environment state dictionary."""
    board_size = 5
    B = 2  # Batch size
    return {
        "board_size": board_size,
        "B": B,
        "boards": jnp.zeros((B, board_size, board_size), dtype=jnp.int32),
        "current_player": jnp.ones(B, dtype=jnp.int32),  # Player 1 starts
        "dones": jnp.zeros(B, dtype=jnp.bool_),
        "steps": 0,  # Custom field for mock step tracking
    }


@pytest.fixture
def rng():
    """Provides a fixed JAX PRNGKey for deterministic tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture(autouse=True)
def patch_env_functions(monkeypatch):
    """Automatically replaces real env functions with mocks for all tests."""
    monkeypatch.setattr("training.rollout.reset_env", mock_reset_env)
    monkeypatch.setattr("training.rollout.step_env", mock_step_env)
    monkeypatch.setattr("training.rollout.get_action_mask", mock_get_action_mask)


# ==============================================================================
# == Test Cases ==
# ==============================================================================

# --- Tests for calculate_returns ---


def test_calculate_returns():
    """Tests discounted return calculation with a standard trajectory."""
    rewards = jnp.array(
        [
            [0.0, 0.0],  # T=0
            [0.0, 0.0],  # T=1
            [1.0, 0.0],  # T=2
            [0.0, 1.0],  # T=3
            [0.0, 0.0],  # T=4
        ]
    )  # Shape (T=5, B=2)
    gamma = 0.9
    # Expected: R_t = r_t + gamma * R_{t+1}
    # R_4 = [0, 0]
    # R_3 = [0, 1] + 0.9 * [0, 0] = [0, 1]
    # R_2 = [1, 0] + 0.9 * [0, 1] = [1, 0.9]
    # R_1 = [0, 0] + 0.9 * [1, 0.9] = [0.9, 0.81]
    # R_0 = [0, 0] + 0.9 * [0.9, 0.81] = [0.81, 0.729]
    expected_returns = jnp.array(
        [
            [0.81, 0.729],
            [0.9, 0.81],
            [1.0, 0.9],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    returns = calculate_returns(rewards, gamma)
    assert returns.shape == rewards.shape
    assert jnp.allclose(returns, expected_returns, atol=1e-3)


def test_calculate_returns_single_step():
    """Tests discounted return calculation with a short trajectory."""
    rewards = jnp.array([[1.0], [2.0]])  # Shape (T=2, B=1)
    gamma = 0.5
    # R_1 = [2.0]
    # R_0 = [1.0] + 0.5 * [2.0] = [2.0]
    expected_returns = jnp.array([[2.0], [2.0]])
    returns = calculate_returns(rewards, gamma)
    assert returns.shape == rewards.shape
    assert jnp.allclose(returns, expected_returns, atol=1e-3)


# --- Tests for calculate_gae ---


def test_calculate_gae():
    """Tests Generalized Advantage Estimation (GAE) calculation."""
    # Note: Focuses on shapes and execution, specific values require careful manual calculation or a reference.
    rewards = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],  # Terminal state for batch 0 at T=2
            [0.0, 1.0],  # Terminal state for batch 1 at T=3
            [
                0.0,
                0.0,
            ],  # Should not occur if dones handled correctly, but included for test robustness
        ]
    )  # Shape (T=5, B=2)
    values = jnp.array(
        [
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],  # V(s_2)
            [0.4, 0.4],  # V(s_3)
            [0.5, 0.5],  # V(s_4) (value after terminal state)
        ]
    )  # Shape (T=5, B=2)
    dones = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],  # Done=True for B=0 at T=2
            [0.0, 1.0],  # Done=True for B=1 at T=3
            [1.0, 1.0],  # Assume dones propagate for subsequent steps for consistency
        ],
        dtype=jnp.float32,
    )  # Use float for masking calculations
    gamma = 0.9
    gae_lambda = 0.95

    # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
    # A_t = delta_t + gamma * gae_lambda * A_{t+1} * (1 - done_{t+1})
    # Example: B=0, T=2 (last step before done)
    # delta_2 = r_2 + gamma*V(s_3)*(1-done_3) - V(s_2) = 1.0 + 0.9*0.4*(1-0) - 0.3 = 1 + 0.36 - 0.3 = 1.06
    # A_3 is needed. Assume A_4 = 0 as it's terminal.
    # delta_3 = r_3 + gamma*V(s_4)*(1-done_4) - V(s_3) = 0.0 + 0.9*0.5*(1-1) - 0.4 = -0.4
    # A_3 = delta_3 + gamma*gae_lambda*A_4*(1-done_4) = -0.4 + 0.9*0.95*0*(1-1) = -0.4
    # A_2 = delta_2 + gamma*gae_lambda*A_3*(1-done_3) = 1.06 + 0.9*0.95*(-0.4)*(1-0) = 1.06 - 0.342 = 0.718

    advantages = calculate_gae(rewards, values, dones, gamma, gae_lambda)
    assert advantages.shape == rewards.shape
    # Add more specific value checks here if a trusted GAE implementation is available for comparison.
    # print("Calculated GAE:\n", advantages)


# --- Tests for run_selfplay ---


def test_run_selfplay(mock_env_state, mock_actor_critic_instance, mock_params, rng):
    """Tests the self-play rollout function using mocks."""
    # Relies on mocked environment functions via patch_env_functions fixture
    max_mock_steps = 5  # Must match mock_step_env
    board_size = mock_env_state["board_size"]
    B = mock_env_state["B"]

    # --- Execute ---
    trajectory, final_rng = run_selfplay(
        mock_env_state, mock_actor_critic_instance, mock_params, rng
    )

    # --- Assertions ---
    assert isinstance(trajectory, dict)
    # Check required keys exist
    assert "observations" in trajectory  # Renamed from 'obs' in function
    assert "actions" in trajectory
    assert "rewards" in trajectory
    assert "masks" in trajectory
    assert "T" in trajectory
    assert "env_state" in trajectory  # Contains final env state
    assert "obs" in trajectory  # Contains final observation
    assert "logprobs" in trajectory  # Check for logprobs

    # Check trajectory length
    T = trajectory["T"]
    assert T == max_mock_steps  # Mock env has fixed length

    # Check shapes of collected data (Time, Batch, ...)
    # Note: Arrays are pre-allocated to max_steps, check relevant dimensions
    assert len(trajectory["observations"].shape) == 4
    assert trajectory["observations"].shape[0] >= T  # Time dim >= actual steps
    assert trajectory["observations"].shape[1] == B  # Batch dim
    assert trajectory["observations"].shape[2:] == (
        board_size,
        board_size,
    )  # Observation dims

    assert len(trajectory["actions"].shape) == 3
    assert trajectory["actions"].shape[0] >= T
    assert trajectory["actions"].shape[1] == B
    assert trajectory["actions"].shape[2] == 2  # Action dim

    assert len(trajectory["rewards"].shape) == 2
    assert trajectory["rewards"].shape[0] >= T
    assert trajectory["rewards"].shape[1] == B

    assert len(trajectory["masks"].shape) == 2
    assert trajectory["masks"].shape[0] >= T
    assert trajectory["masks"].shape[1] == B

    # Check logprobs shape
    assert len(trajectory["logprobs"].shape) == 2
    assert trajectory["logprobs"].shape[0] >= T
    assert trajectory["logprobs"].shape[1] == B

    # Check final state termination
    assert "dones" in trajectory["env_state"]
    assert jnp.all(trajectory["env_state"]["dones"]), "All envs should be done"

    # Check RNG consumption
    assert not jnp.array_equal(rng, final_rng), "RNG key should be updated"


# --- Tests for run_episode ---


def test_run_episode(mock_env_state, mock_actor_critic_instance, mock_params, rng):
    """Tests the two-player episode rollout function using mocks."""
    # Relies on mocked environment functions via patch_env_functions fixture
    max_mock_steps = 5  # Must match mock_step_env
    board_size = mock_env_state["board_size"]
    B = mock_env_state["B"]

    # Use the same mock model for black and white players for simplicity
    black_ac = mock_actor_critic_instance
    white_ac = mock_actor_critic_instance
    black_params = mock_params
    white_params = mock_params

    # --- Execute ---
    black_traj, white_traj, final_rng = run_episode(
        mock_env_state, black_ac, black_params, white_ac, white_params, rng
    )

    # --- Assertions ---
    assert isinstance(black_traj, dict)
    assert isinstance(white_traj, dict)
    # Check required keys
    for traj in [black_traj, white_traj]:
        assert "obs" in traj
        assert "actions" in traj
        assert "rewards" in traj
        assert "masks" in traj
        assert "T" in traj
        assert "logprobs" in traj  # Check for logprobs

    # Check trajectory lengths (T represents number of turns for that player)
    total_steps = max_mock_steps
    expected_black_len = (total_steps + 1) // 2  # Black moves first (steps 0, 2, 4...)
    expected_white_len = total_steps // 2  # White moves second (steps 1, 3...)
    assert black_traj["T"] == expected_black_len
    assert white_traj["T"] == expected_white_len

    # Check black trajectory shapes (Turns, Batch, ...)
    # Note: Shape[0] will be based on max_steps due to JIT pre-allocation
    max_possible_steps = board_size * board_size
    expected_black_shape_len = (max_possible_steps + 1) // 2
    assert len(black_traj["obs"].shape) == 4
    assert (
        black_traj["obs"].shape[0] == expected_black_shape_len
    )  # Num turns based on max_steps
    assert black_traj["obs"].shape[1] == B
    assert black_traj["obs"].shape[2:] == (board_size, board_size)

    assert len(black_traj["actions"].shape) == 3
    assert black_traj["actions"].shape[0] == expected_black_shape_len
    assert black_traj["actions"].shape[1] == B
    assert black_traj["actions"].shape[2] == 2

    assert len(black_traj["rewards"].shape) == 2
    assert black_traj["rewards"].shape[0] == expected_black_shape_len
    assert black_traj["rewards"].shape[1] == B

    assert len(black_traj["masks"].shape) == 2
    assert black_traj["masks"].shape[0] == expected_black_shape_len
    assert black_traj["masks"].shape[1] == B

    # Check black logprobs shape
    assert len(black_traj["logprobs"].shape) == 2
    assert black_traj["logprobs"].shape[0] == expected_black_shape_len
    assert black_traj["logprobs"].shape[1] == B

    # Check white trajectory shapes (Turns, Batch, ...)
    expected_white_shape_len = max_possible_steps // 2
    assert len(white_traj["obs"].shape) == 4
    assert (
        white_traj["obs"].shape[0] == expected_white_shape_len
    )  # Num turns based on max_steps
    assert white_traj["obs"].shape[1] == B
    assert white_traj["obs"].shape[2:] == (board_size, board_size)

    assert len(white_traj["actions"].shape) == 3
    assert white_traj["actions"].shape[0] == expected_white_shape_len
    assert white_traj["actions"].shape[1] == B
    assert white_traj["actions"].shape[2] == 2

    assert len(white_traj["rewards"].shape) == 2
    assert white_traj["rewards"].shape[0] == expected_white_shape_len
    assert white_traj["rewards"].shape[1] == B

    assert len(white_traj["masks"].shape) == 2
    assert white_traj["masks"].shape[0] == expected_white_shape_len
    assert white_traj["masks"].shape[1] == B

    # Check white logprobs shape
    assert len(white_traj["logprobs"].shape) == 2
    assert white_traj["logprobs"].shape[0] == expected_white_shape_len
    assert white_traj["logprobs"].shape[1] == B

    # Check final state masks/termination implicitly via lengths and mock env logic
    # (More specific checks could be added if needed, e.g., final rewards)

    # Check RNG consumption
    assert not jnp.array_equal(rng, final_rng), "RNG key should be updated"
    print("black player rewards", black_traj["rewards"])
    print("white player rewards", white_traj["rewards"])
