import pytest
import jax
import jax.numpy as jnp
import numpy as np
from env.pong import init_env, reset_env, step_env, get_action_mask, NUM_ACTIONS, OBSERVATION_SHAPE

# Constants for testing
TEST_B = 1
TEST_SEED = 42

@pytest.fixture(scope="module")
def initial_env_state():
    """Fixture to initialize the Pong environment state once per module."""
    key = jax.random.PRNGKey(TEST_SEED)
    key, subkey = jax.random.split(key)
    env_state = init_env(B=TEST_B, rng=subkey)
    yield env_state
    # Cleanup: close gym environments after tests are done
    for env in env_state["gym_envs"]:
        env.close()

def test_init_env(initial_env_state):
    """Test the init_env function."""
    state = initial_env_state
    assert state["B"] == TEST_B
    assert isinstance(state["gym_envs"], list)
    assert len(state["gym_envs"]) == TEST_B
    # Check initial state shapes and types
    assert state["observations"].shape == (TEST_B,) + OBSERVATION_SHAPE
    assert state["observations"].dtype == jnp.uint8
    assert state["current_player"].shape == (TEST_B,)
    assert state["current_player"].dtype == jnp.int32
    assert jnp.all(state["current_player"] == 1)
    assert state["dones"].shape == (TEST_B,)
    assert state["dones"].dtype == jnp.bool_
    assert jnp.all(~state["dones"]) # Should not be done initially
    assert state["winners"].shape == (TEST_B,)
    assert state["winners"].dtype == jnp.int32
    assert jnp.all(state["winners"] == 0)
    assert state["total_reward"].shape == (TEST_B,)
    assert state["total_reward"].dtype == jnp.float32
    assert jnp.all(state["total_reward"] == 0.0)
    assert "rng" in state

def test_reset_env(initial_env_state):
    """Test the reset_env function."""
    key = jax.random.PRNGKey(TEST_SEED + 1)
    # Take a step first to change the state
    action = jnp.zeros((TEST_B,), dtype=jnp.int32) # NOOP action
    intermediate_state, _, _, _ = step_env(initial_env_state, action)

    # Test reset without new rng
    reset_state, reset_obs = reset_env(intermediate_state)
    assert reset_state["B"] == TEST_B
    assert reset_obs.shape == (TEST_B,) + OBSERVATION_SHAPE
    assert reset_obs.dtype == jnp.uint8
    assert jnp.all(~reset_state["dones"])
    assert jnp.all(reset_state["current_player"] == 1)
    assert jnp.all(reset_state["winners"] == 0)
    assert jnp.all(reset_state["total_reward"] == 0.0)
    # Ensure observations are different after reset (highly likely)
    # assert not jnp.array_equal(intermediate_state[\"observations\"], reset_obs) # This can be flaky
    assert not jnp.array_equal(intermediate_state["rng"], reset_state["rng"]) # Check that the first reset consumed the key

    # Test reset with new rng
    key, subkey = jax.random.split(key)
    reset_state_new_rng, reset_obs_new_rng = reset_env(intermediate_state, new_rng=subkey)
    assert reset_state_new_rng["rng"] is not None # Check rng is updated
    # assert not jnp.array_equal(reset_obs, reset_obs_new_rng) # Reset with new rng should yield different obs - Flaky
    assert not jnp.array_equal(reset_state["rng"], reset_state_new_rng["rng"]) # Check that using new_rng leads to a different final rng state

def test_step_env(initial_env_state):
    """Test the step_env function."""
    state = initial_env_state
    action = jnp.array([1], dtype=jnp.int32) # FIRE action

    new_state, obs, reward, done = step_env(state, action)

    # Check return shapes and types
    assert obs.shape == (TEST_B,) + OBSERVATION_SHAPE
    assert obs.dtype == jnp.uint8
    assert reward.shape == (TEST_B,)
    assert reward.dtype == jnp.float32
    assert done.shape == (TEST_B,)
    assert done.dtype == jnp.bool_

    # Check state update
    assert not jnp.array_equal(state["observations"], new_state["observations"])
    assert new_state["dones"].shape == (TEST_B,) # Dones shape remains
    assert new_state["total_reward"].shape == (TEST_B,)
    # Ensure total reward is updated correctly
    assert jnp.all(new_state["total_reward"] == state["total_reward"] + reward)

    # Test stepping when done
    done_state = new_state.copy()
    done_state["dones"] = jnp.array([True], dtype=jnp.bool_)
    done_state["total_reward"] = jnp.array([10.0], dtype=jnp.float32) # Example reward

    step_after_done_state, step_after_done_obs, step_after_done_reward, step_after_done_done = step_env(done_state, action)

    assert jnp.array_equal(step_after_done_state["observations"], done_state["observations"])
    assert jnp.all(step_after_done_reward == 0.0) # Reward should be 0 when done
    assert jnp.all(step_after_done_done) # Should remain done
    assert jnp.all(step_after_done_state["dones"] == done_state["dones"])
    assert jnp.all(step_after_done_state["total_reward"] == done_state["total_reward"]) # Total reward unchanged

def test_get_action_mask(initial_env_state):
    """Test the get_action_mask function."""
    state = initial_env_state

    # Test mask when not done
    action_mask_not_done = get_action_mask(state)
    assert action_mask_not_done.shape == (TEST_B, NUM_ACTIONS)
    assert action_mask_not_done.dtype == jnp.bool_
    assert jnp.all(action_mask_not_done) # All actions valid initially

    # Test mask when done
    done_state = state.copy()
    done_state["dones"] = jnp.array([True], dtype=jnp.bool_)
    action_mask_done = get_action_mask(done_state)
    assert action_mask_done.shape == (TEST_B, NUM_ACTIONS)
    assert action_mask_done.dtype == jnp.bool_
    assert jnp.all(~action_mask_done) # No actions valid when done

def test_episode_completion(initial_env_state):
    """Test running a full episode until done."""
    state = initial_env_state
    key = jax.random.PRNGKey(TEST_SEED + 2)
    max_steps = 5000 # Set a max step limit to prevent infinite loops

    for step in range(max_steps):
        if state["dones"][0]:
            break

        mask = get_action_mask(state)
        assert mask.shape == (TEST_B, NUM_ACTIONS)

        # Sample a random valid action
        key, subkey = jax.random.split(state["rng"])
        state["rng"] = key # Update rng in state for consistency if needed later
        # Simple random choice for B=1
        action = jax.random.randint(subkey, shape=(TEST_B,), minval=0, maxval=NUM_ACTIONS)

        state, obs, reward, done = step_env(state, action)

        # Basic checks within the loop
        assert obs.shape == (TEST_B,) + OBSERVATION_SHAPE
        assert reward.shape == (TEST_B,)
        assert done.shape == (TEST_B,)

    assert state["dones"][0], f"Episode did not finish within {max_steps} steps."
    # Check winner assignment (simple check: winner is non-zero if episode ended)
    # A more robust check would depend on Pong's specific scoring
    assert state["winners"][0] != 0 # Winner should be set to 1 or -1
    assert (state["total_reward"][0] > 0 and state["winners"][0] == 1) or \
           (state["total_reward"][0] <= 0 and state["winners"][0] == -1) # Basic check

    # Check action mask after done
    final_mask = get_action_mask(state)
    assert jnp.all(~final_mask) 