import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Import the functions to be tested.
from scripts.train import discount_rewards, preprocess_trajectory, run_episode

def test_discount_rewards():
    """
    Test that discount_rewards() returns correctly discounted rewards for a rewards array of shape (batch, T).
    
    For a rewards array:
         [[ 1,  2,  3],
          [10, 20, 30]]
          
    and gamma = 0.99, we expect for each batch:
    
      For the first batch:
          r[0] = 1   + 0.99*2   + (0.99)**2*3
          r[1] = 2   + 0.99*3
          r[2] = 3
    
      For the second batch:
          r[0] = 10  + 0.99*20  + (0.99)**2*30
          r[1] = 20  + 0.99*30
          r[2] = 30
    """
    gamma = 0.99
    rewards = jnp.array([
        [1.0, 2.0, 3.0],
        [10.0, 20.0, 30.0]
    ])  # shape: (batch, T)

    # Use vmap to apply discount_rewards over each reward sequence (i.e. over axis 0)
    computed = discount_rewards(rewards, gamma)
    expected = jnp.array([
        [1.0 + 2.0 * gamma + 3.0 * gamma**2, 2.0 + 3.0 * gamma, 3.0],
        [10.0 + 20.0 * gamma + 30.0 * gamma**2, 20.0 + 30.0 * gamma, 30.0]
    ])
    assert jnp.allclose(computed, expected, atol=1e-4), f"Expected {expected} but got {computed}"

def test_preprocess_trajectory():
    """
    Test preprocess_trajectory() over a simulated trajectory.

    We create a trajectory dictionary where:
      - The observations (obs) have shape (T, num_envs, board_size, board_size)
      - The actions are stored as 2-element tuples.
      - The rewards and dones are stored with shape (T, num_envs).

    For the dones flag, we simulate:
      - For env0: dones become True at time index 2.
      - For env1: dones become True at time index 1.
    With mask = jnp.arange(T)[:, None] <= indices[None, :], this means:
      - For env0: valid rows are indices 0, 1, and 2 (including where done=True)
      - For env1: valid rows are indices 0 and 1 (including where done=True)
    """
    board_size = 3
    T = 3
    num_envs = 2

    # Create dummy data.
    obs_data = jnp.arange(T * num_envs * board_size * board_size)\
        .reshape(T, num_envs, board_size, board_size)
    # Each action is a tuple [time_step, env_index]; shape (T, num_envs, 2)
    actions_data = jnp.array([[[t, e] for e in range(num_envs)] for t in range(T)])
    rewards_data = jnp.array([[t + float(e) for e in range(num_envs)] for t in range(T)], dtype=jnp.float32)
    dones_data = jnp.array([
        [False, False],
        [False, True],
        [True, True]
    ])

    # Build the trajectory dictionary (note: preprocess_trajectory expects lists).
    trajectory = {
        "obs": list(obs_data),
        "actions": list(actions_data),
        "rewards": list(rewards_data),
        "dones": list(dones_data)
    }

    processed = preprocess_trajectory(trajectory)

    # Expected valid entries based on mask:
    # For env0: valid rows are indices 0, 1, and 2; for env1: indices 0 and 1.
    # The mask (computed with <= instead of <) is:
    #   mask = [[True, True], [True, True], [True, False]]
    # so the valid (flattened) order is: (0,0), (0,1), (1,0), (1,1), (2,0)
    
    # First, check we have 5 valid entries
    assert processed["obs"].shape[0] == 5, f"Expected 5 valid observations, got {processed['obs'].shape[0]}"
    assert processed["actions"].shape[0] == 5, f"Expected 5 valid actions, got {processed['actions'].shape[0]}"
    assert processed["rewards"].shape[0] == 5, f"Expected 5 valid rewards, got {processed['rewards'].shape[0]}"
    
    # Check entry 0 (from 0,0)
    assert jnp.all(processed["obs"][0] == obs_data[0, 0]), "First observation mismatch"
    assert jnp.all(processed["actions"][0] == actions_data[0, 0]), "First action mismatch"
    assert jnp.allclose(processed["rewards"][0], rewards_data[0, 0]), "First reward mismatch"
    
    # Check entry 1 (from 0,1)
    assert jnp.all(processed["obs"][1] == obs_data[0, 1]), "Second observation mismatch"
    assert jnp.all(processed["actions"][1] == actions_data[0, 1]), "Second action mismatch"
    assert jnp.allclose(processed["rewards"][1], rewards_data[0, 1]), "Second reward mismatch"
    
    # Check entry 2 (from 1,0)
    assert jnp.all(processed["obs"][2] == obs_data[1, 0]), "Third observation mismatch"
    assert jnp.all(processed["actions"][2] == actions_data[1, 0]), "Third action mismatch" 
    assert jnp.allclose(processed["rewards"][2], rewards_data[1, 0]), "Third reward mismatch"
    
    # Check entry 3 (from 1,1)
    assert jnp.all(processed["obs"][3] == obs_data[1, 1]), "Fourth observation mismatch"
    assert jnp.all(processed["actions"][3] == actions_data[1, 1]), "Fourth action mismatch"
    assert jnp.allclose(processed["rewards"][3], rewards_data[1, 1]), "Fourth reward mismatch"
    
    # Check entry 4 (from 2,0)
    assert jnp.all(processed["obs"][4] == obs_data[2, 0]), "Fifth observation mismatch"
    assert jnp.all(processed["actions"][4] == actions_data[2, 0]), "Fifth action mismatch"
    assert jnp.allclose(processed["rewards"][4], rewards_data[2, 0]), "Fifth reward mismatch"

# Define dummy environment and dummy ActorCritic for testing run_episode.

class DummyEnv:
    """
    A dummy Gomoku environment that simulates a two-environment (num_envs=2)
    episode over two steps.
    - reset() returns an initial board with all zeros and dones = [False, False].
    - The first call to step() returns dones = [True, False] so that for
      env0 the episode ends, and the second call returns dones = [True, True],
      ending the episode for both.
    """
    def __init__(self):
        self.step_count = 0
        self.num_envs = 2
        self.board_size = 3
        self.winners = jnp.ones((self.num_envs,))

    def reset(self):
        self.step_count = 0
        obs = jnp.full((self.num_envs, self.board_size, self.board_size), 0)
        dones = jnp.array([False, False])
        return obs, dones

    def step(self, action):
        self.step_count += 1
        obs = jnp.full((self.num_envs, self.board_size, self.board_size), self.step_count)
        rewards = jnp.array([1.0, 2.0])
        if self.step_count == 1:
            # End episode for env0 only.
            dones = jnp.array([True, False])
        else:
            dones = jnp.array([True, True])
        return obs, rewards, dones

    def get_action_mask(self):
        # For testing, assume all actions are valid.
        return jnp.ones((self.num_envs, self.board_size * self.board_size), dtype=bool)

class DummyActorCritic:
    """
    A dummy ActorCritic that, when applied, returns constant logits and values.
    The sample_action() method returns zeros for both coordinates, batched for each environment.
    """
    board_size = 3
    def apply(self, params, obs):
        num_envs = obs.shape[0]
        # For simplicity, we return logits as ones (shape: (num_envs, board_size*board_size))
        logits = jnp.ones((num_envs, self.board_size*self.board_size))
        value = jnp.ones((num_envs, 1))
        return logits, value

    def sample_action(self, logits, rng):
        num_envs = logits.shape[0]
        return jnp.zeros((num_envs, 2), dtype=jnp.int32)

def create_dummy_actor_critic(board_size):
    """Helper to create a dummy actor_critic with board_size attached."""
    dummy = DummyActorCritic()
    dummy.board_size = board_size
    return dummy

def test_run_episode():
    """
    Test run_episode() using the dummy environment and dummy ActorCritic.
    In our simulation:
      - The environment runs for two steps.
      - The dones state after reset is [False, False]
      - After first step: dones = [True, False] (env0 ends, env1 continues)
      - After second step: dones = [True, True] (both end)
      
    With mask = jnp.arange(T)[:, None] <= indices[None, :], we'll include:
    - For env0: The initial observation and the observation after step 1 (2 observations)
    - For env1: The initial observation only (1 observation)
    Total: 3 valid observations
    """
    gamma = 0.99
    rng = jax.random.PRNGKey(42)
    dummy_env = DummyEnv()
    dummy_actor = create_dummy_actor_critic(dummy_env.board_size)
    params = None  # Dummy parameters not used within DummyActorCritic.
    trajectory, new_rng = run_episode(dummy_env, dummy_actor, params, gamma, rng)

    # Check that the trajectory dict has the expected keys.
    for key in ["obs", "actions", "rewards"]:
        assert key in trajectory, f"Key '{key}' not found in trajectory."

    # Based on our understanding of the mask (using <=), we expect 3 valid observations:
    # Two from env0 (the initial state and the state after step 1)
    # One from env1 (just the initial state)
    assert trajectory["obs"].ndim == 3, "Processed observations should be 3D (num_valid, board_size, board_size)."
    assert trajectory["obs"].shape[0] == 3, f"Expected 3 valid observations, got {trajectory['obs'].shape[0]}"
    
    # Check the first observation (from env0, initial state)
    assert jnp.all(trajectory["obs"][0] == 0), "First observation should be all zeros (from initial state)"
    
    # Check that all actions are (0, 0) as set by our dummy actor
    assert trajectory["actions"].shape == (3, 2), f"Expected actions shape to be (3,2), got {trajectory['actions'].shape}"
    assert jnp.all(trajectory["actions"] == jnp.zeros((3, 2))), "All actions should be zeros"

    # Check rewards shape
    assert trajectory["rewards"].shape == (3,), f"Expected rewards shape to be (3,), got {trajectory['rewards'].shape}"

if __name__ == "__main__":
    pytest.main() 