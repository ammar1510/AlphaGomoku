import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import optax

from training.train import (
    init_train,
    discount_rewards,
    run_episode,
    train_step
)
from models.actor_critic import ActorCritic
from env.functional_gomoku import init_env, reset_env


class TestTrainingFunctions(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment and parameters."""
        self.board_size = 5  # Small board for faster tests
        self.test_config = {
            "board_size": self.board_size,
            "num_boards": 2,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "grad_clip_norm": 1.0,
            "seed": 42,
            "render": False,
            "checkpoint_dir": tempfile.mkdtemp(),  # Temporary directory for checkpoints
            "discount": 0.99,
            "save_frequency": 10,
            "total_iterations": 10,
        }
        self.rng = jax.random.PRNGKey(42)
        
        # Create temporary checkpoint directory
        os.makedirs(self.test_config["checkpoint_dir"], exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary checkpoint directory
        import shutil
        if os.path.exists(self.test_config["checkpoint_dir"]):
            shutil.rmtree(self.test_config["checkpoint_dir"])
    
    def test_discount_rewards(self):
        """Test the discount_rewards function with known inputs and outputs."""
        # Case 1: Simple sequence with constant rewards
        rewards = jnp.ones((5, 2))  # 5 steps, 2 environments
        gamma = 0.9
        discounted = discount_rewards(rewards, gamma)
        
        # Expected values for first environment
        # With alternating signs: [1, -1, 1, -1, 1]
        # Discounted backwards:
        # [1 - 0.9 * (-1) + 0.9^2 * 1 - 0.9^3 * (-1) + 0.9^4 * 1, 
        #  -1 + 0.9 * 1 - 0.9^2 * (-1) + 0.9^3 * 1, 
        #  1 - 0.9 * (-1) + 0.9^2 * 1, 
        #  -1 + 0.9 * 1, 
        #  1]
        # This will give varying values based on the alternating signs and discount
        print("discounted: \n", discounted)
        
        # Basic tests - check shape and type
        self.assertEqual(discounted.shape, rewards.shape)
        
        # Test alternating signs
        # For each time step in the first environment, check if signs alternate
        first_env_disc = discounted[:, 0]
        expected_signs = jnp.array([1, -1, 1, -1, 1])
        self.assertTrue(jnp.all(jnp.sign(first_env_disc) == expected_signs))
        
        # Case 2: Test with zero rewards
        zero_rewards = jnp.zeros((3, 2))
        zero_discounted = discount_rewards(zero_rewards, gamma)
        self.assertTrue(jnp.all(zero_discounted == 0))
        
        # Case 3: Test with variable rewards
        variable_rewards = jnp.array([[1.0, 2.0], [0.5, 1.5], [2.0, 0.0]])
        var_discounted = discount_rewards(variable_rewards, gamma)
        self.assertEqual(var_discounted.shape, variable_rewards.shape)
    
    @patch('training.train.select_training_checkpoints')
    @patch('training.train.load_checkpoint')
    def test_init_train(self, mock_load_checkpoint, mock_select_checkpoints):
        """Test initialization of training components."""
        # Mock checkpoint selection to return None (no checkpoints)
        mock_select_checkpoints.return_value = (None, None)
        
        # Call init_train
        (env, black_ac, black_params, black_opt_state, black_optimizer,
         white_ac, white_params, white_opt_state, white_optimizer,
         checkpoint_dir, rng, board_size) = init_train(self.test_config)
        
        # Basic checks
        self.assertEqual(board_size, self.test_config["board_size"])
        self.assertEqual(env["board_size"], self.test_config["board_size"])
        self.assertEqual(env["num_boards"], self.test_config["num_boards"])
        
        # Check models
        self.assertIsInstance(black_ac, ActorCritic)
        self.assertIsInstance(white_ac, ActorCritic)
        
        # Check parameter shapes
        dummy_input = jnp.ones((1, self.board_size, self.board_size))
        _, black_value = black_ac.apply(black_params, dummy_input)
        _, white_value = white_ac.apply(white_params, dummy_input)
        
        self.assertEqual(black_value.shape, (1,))
        self.assertEqual(white_value.shape, (1,))
        
        # Now test with mock checkpoint for black player
        dummy_params = MagicMock()
        dummy_opt_state = MagicMock()
        black_checkpoint_path = "/fake/path/black_checkpoint.pkl"
        mock_select_checkpoints.return_value = (black_checkpoint_path, None)
        mock_load_checkpoint.return_value = (dummy_params, dummy_opt_state)
        
        # Call init_train again
        (env, black_ac, black_params, black_opt_state, black_optimizer,
         white_ac, white_params, white_opt_state, white_optimizer,
         checkpoint_dir, rng, board_size) = init_train(self.test_config)
        
        # Check that black model uses loaded params
        mock_load_checkpoint.assert_called_with(black_checkpoint_path)
        self.assertEqual(black_params, dummy_params)
        self.assertEqual(black_opt_state, dummy_opt_state)
        
        # Test with both black and white checkpoints
        black_checkpoint_path = "/fake/path/black_checkpoint.pkl"
        white_checkpoint_path = "/fake/path/white_checkpoint.pkl"
        mock_select_checkpoints.return_value = (black_checkpoint_path, white_checkpoint_path)
        mock_load_checkpoint.side_effect = [(dummy_params, dummy_opt_state), (dummy_params, dummy_opt_state)]
        
        # Call init_train again
        (env, black_ac, black_params, black_opt_state, black_optimizer,
         white_ac, white_params, white_opt_state, white_optimizer,
         checkpoint_dir, rng, board_size) = init_train(self.test_config)
        
        # Check both models use loaded params
        self.assertEqual(black_params, dummy_params)
        self.assertEqual(white_params, dummy_params)
    
    def test_run_episode(self):
        """Test running an episode between black and white players."""
        # Initialize environment and models
        rng, init_key = jax.random.split(self.rng)
        env = init_env(init_key, self.board_size, num_boards=1)
        env, _ = reset_env(env)
        
        black_actor_critic = ActorCritic(board_size=self.board_size)
        white_actor_critic = ActorCritic(board_size=self.board_size)
        
        # Initialize dummy parameters
        dummy_input = jnp.ones((1, self.board_size, self.board_size))
        rng, black_key, white_key = jax.random.split(rng, 3)
        black_params = black_actor_critic.init(black_key, dummy_input)
        white_params = white_actor_critic.init(white_key, dummy_input)
        
        # Run a short episode
        gamma = 0.99
        black_traj, white_traj, new_rng = run_episode(
            env, black_actor_critic, black_params, white_actor_critic, white_params, gamma, rng
        )
        
        # Check trajectory structures
        self.assertIn("obs", black_traj)
        self.assertIn("actions", black_traj)
        self.assertIn("rewards", black_traj)
        self.assertIn("masks", black_traj)
        
        self.assertIn("obs", white_traj)
        self.assertIn("actions", white_traj)
        self.assertIn("rewards", white_traj)
        self.assertIn("masks", white_traj)
        
        # Check trajectories have expected structures for a single environment
        # The black trajectory should have at least one observation
        self.assertGreater(len(black_traj["obs"]), 0)
        # The white trajectory should have at least one observation
        self.assertGreater(len(white_traj["obs"]), 0)
        
        # Check episode length field is present and reasonable
        self.assertIn("episode_length", black_traj)
        self.assertIn("episode_length", white_traj)
        
        # Black should have same or one more move than white
        black_steps = black_traj["episode_length"]
        white_steps = white_traj["episode_length"]
        self.assertTrue(black_steps >= white_steps)
        self.assertTrue(black_steps <= white_steps + 1)
    
    def test_train_step(self):
        """Test the training step function with mock data."""
        # Initialize actor-critic model
        actor_critic = ActorCritic(board_size=self.board_size)
        
        # Initialize model parameters
        rng, init_key = jax.random.split(self.rng)
        dummy_input = jnp.ones((1, self.board_size, self.board_size))
        params = actor_critic.init(init_key, dummy_input)
        
        # Initialize optimizer
        optimizer = optax.adam(learning_rate=0.001)
        opt_state = optimizer.init(params)
        
        # Create mock trajectory
        batch_size = 2
        seq_len = 3
        
        # Create simple mock trajectory with valid observations
        trajectory = {
            "obs": jnp.ones((seq_len, batch_size, self.board_size, self.board_size)),
            "actions": jnp.zeros((seq_len, batch_size, 2), dtype=jnp.int32),
            "rewards": jnp.ones((seq_len, batch_size)),
            "masks": jnp.ones((seq_len, batch_size), dtype=jnp.bool_),
        }
        
        # Call train_step
        new_params, new_opt_state, loss, aux, grad_norm = train_step(
            params, opt_state, trajectory, actor_critic, optimizer
        )
        
        # Check outputs
        self.assertIsNotNone(new_params)
        self.assertIsNotNone(new_opt_state)
        self.assertIsInstance(loss, jnp.ndarray)
        self.assertEqual(len(aux), 3)  # actor_loss, critic_loss, entropy_loss
        self.assertIsInstance(grad_norm, jnp.ndarray)
        
        # Params should be updated (not equal to original)
        params_flat = jax.tree_util.tree_leaves(params)
        new_params_flat = jax.tree_util.tree_leaves(new_params)
        
        any_changed = False
        for p1, p2 in zip(params_flat, new_params_flat):
            if not jnp.array_equal(p1, p2):
                any_changed = True
                break
        
        self.assertTrue(any_changed, "Parameters should be updated during training")


if __name__ == "__main__":
    unittest.main() 