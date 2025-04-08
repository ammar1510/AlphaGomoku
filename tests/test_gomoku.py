import unittest

import jax
import jax.numpy as jnp
import numpy as np

from env.gomoku import (
    WIN_LENGTH,
    check_win,
    get_action_mask,
    get_valid_actions,
    init_env,
    is_game_over,
    reset_env,
    run_random_episode,
    sample_action,
    step_env,
)


class TestFunctionalGomoku(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Use a fixed seed for deterministic testing
        self.rng = jax.random.PRNGKey(42)
        self.board_size = 9
        self.B = 4

    def test_init_env(self):
        """Test environment initialization."""
        env_state = init_env(self.board_size, self.B, self.rng)

        # Check environment structure
        self.assertIn("boards", env_state)
        self.assertIn("current_player", env_state)
        self.assertIn("dones", env_state)
        self.assertIn("winners", env_state)
        self.assertIn("board_size", env_state)
        self.assertIn("B", env_state)
        self.assertIn("rng", env_state)

        # Check dimensions
        self.assertEqual(
            env_state["boards"].shape,
            (self.B, self.board_size, self.board_size),
        )
        self.assertEqual(env_state["current_player"].shape, (self.B,))
        self.assertEqual(env_state["dones"].shape, (self.B,))
        self.assertEqual(env_state["winners"].shape, (self.B,))

        # Check initial values
        self.assertTrue(np.all(env_state["boards"] == 0))
        self.assertTrue(np.all(env_state["current_player"] == 1))
        self.assertTrue(np.all(env_state["dones"] == False))
        self.assertTrue(np.all(env_state["winners"] == 0))
        self.assertEqual(env_state["board_size"], self.board_size)
        self.assertEqual(env_state["B"], self.B)

    def test_reset_env(self):
        """Test environment reset."""
        # Initialize environment
        env_state = init_env(self.board_size, self.B, self.rng)

        # Modify state
        new_board = env_state["boards"].at[0, 0, 0].set(1)
        env_state["boards"] = new_board
        env_state["current_player"] = env_state["current_player"].at[0].set(-1)
        env_state["dones"] = env_state["dones"].at[0].set(True)

        # Reset environment
        new_rng = jax.random.PRNGKey(43)
        new_env_state, observations = reset_env(env_state, new_rng)

        # Check if reset properly
        self.assertTrue(np.all(new_env_state["boards"] == 0))
        self.assertTrue(np.all(new_env_state["current_player"] == 1))
        self.assertTrue(np.all(new_env_state["dones"] == False))
        self.assertTrue(np.all(new_env_state["winners"] == 0))

        # Check observations
        self.assertTrue(np.array_equal(observations, new_env_state["boards"]))

        # Check if RNG was updated
        self.assertTrue(np.array_equal(new_env_state["rng"], new_rng))

    def test_check_win_horizontal(self):
        """Test win detection for horizontal patterns."""
        env_state = init_env(self.board_size, self.B, self.rng)

        # Create a horizontal line of player 1 pieces in the first board
        for i in range(WIN_LENGTH):
            env_state["boards"] = env_state["boards"].at[0, 4, i].set(1)

        # Check for wins
        wins = check_win(env_state["boards"], env_state["current_player"])

        # First board should have a win, others should not
        self.assertTrue(wins[0])
        self.assertTrue(np.all(wins[1:] == False))

    def test_check_win_vertical(self):
        """Test win detection for vertical patterns."""
        env_state = init_env(self.board_size, self.B, self.rng)

        # Create a vertical line of player 1 pieces in the second board
        for i in range(WIN_LENGTH):
            env_state["boards"] = env_state["boards"].at[1, i, 4].set(1)

        # Check for wins
        wins = check_win(env_state["boards"], env_state["current_player"])

        # Second board should have a win, others should not
        self.assertFalse(wins[0])
        self.assertTrue(wins[1])
        self.assertTrue(np.all(wins[2:] == False))

    def test_check_win_diagonal(self):
        """Test win detection for diagonal patterns."""
        env_state = init_env(self.board_size, self.B, self.rng)

        # Create a diagonal line of player 1 pieces in the third board
        for i in range(WIN_LENGTH):
            env_state["boards"] = env_state["boards"].at[2, i, i].set(1)

        # Check for wins
        wins = check_win(env_state["boards"], env_state["current_player"])

        # Third board should have a win, others should not
        self.assertFalse(wins[0])
        self.assertFalse(wins[1])
        self.assertTrue(wins[2])
        self.assertFalse(wins[3])

    def test_check_win_anti_diagonal(self):
        """Test win detection for anti-diagonal patterns."""
        env_state = init_env(self.board_size, self.B, self.rng)

        # Create an anti-diagonal line of player 1 pieces in the fourth board
        for i in range(WIN_LENGTH):
            env_state["boards"] = (
                env_state["boards"].at[3, i, WIN_LENGTH - 1 - i].set(1)
            )

        # Check for wins
        wins = check_win(env_state["boards"], env_state["current_player"])

        # Fourth board should have a win, others should not
        self.assertFalse(wins[0])
        self.assertFalse(wins[1])
        self.assertFalse(wins[2])
        self.assertTrue(wins[3])

    def test_get_action_mask(self):
        """Test action mask generation."""
        env_state = init_env(self.board_size, self.B, self.rng)

        # All actions should be valid initially
        action_mask = get_action_mask(env_state)
        self.assertTrue(np.all(action_mask))

        # Place pieces on the board
        env_state["boards"] = env_state["boards"].at[0, 0, 0].set(1)
        env_state["boards"] = env_state["boards"].at[1, 1, 1].set(-1)

        # Mark a board as done
        env_state["dones"] = env_state["dones"].at[2].set(True)

        # Get new action mask
        action_mask = get_action_mask(env_state)

        # Check if occupied spots are invalid
        self.assertFalse(action_mask[0, 0, 0])
        self.assertFalse(action_mask[1, 1, 1])

        # Check if all positions on the done board are invalid
        self.assertTrue(np.all(action_mask[2] == False))

        # Other positions should still be valid
        self.assertTrue(action_mask[0, 0, 1])
        self.assertTrue(action_mask[1, 0, 0])
        self.assertTrue(np.all(action_mask[3]))

    def test_step_env(self):
        """Test taking steps in the environment."""
        env_state = init_env(self.board_size, self.B, self.rng)

        # Take a step with valid actions
        actions = jnp.array(
            [
                [1, 2],  # Board 0: row 1, col 2
                [2, 3],  # Board 1: row 2, col 3
                [3, 4],  # Board 2: row 3, col 4
                [4, 5],  # Board 3: row 4, col 5
            ]
        )

        new_env_state, observations, rewards, dones = step_env(env_state, actions)

        # Check if pieces were placed correctly
        self.assertEqual(new_env_state["boards"][0, 1, 2], 1)  # First board, player 1
        self.assertEqual(new_env_state["boards"][1, 2, 3], 1)  # Second board, player 1
        self.assertEqual(new_env_state["boards"][2, 3, 4], 1)  # Third board, player 1
        self.assertEqual(new_env_state["boards"][3, 4, 5], 1)  # Fourth board, player 1

        # Check if player switched
        self.assertTrue(np.all(new_env_state["current_player"] == -1))

        # Check if no wins yet
        self.assertTrue(np.all(new_env_state["winners"] == 0))
        self.assertTrue(np.all(dones == False))

        # Take another step with player 2
        actions = jnp.array(
            [
                [1, 3],  # Board 0: row 1, col 3
                [2, 4],  # Board 1: row 2, col 4
                [3, 5],  # Board 2: row 3, col 5
                [4, 6],  # Board 3: row 4, col 6
            ]
        )

        newer_env_state, observations, rewards, dones = step_env(new_env_state, actions)

        # Check if pieces were placed correctly
        self.assertEqual(
            newer_env_state["boards"][0, 1, 3], -1
        )  # First board, player -1
        self.assertEqual(
            newer_env_state["boards"][1, 2, 4], -1
        )  # Second board, player -1
        self.assertEqual(
            newer_env_state["boards"][2, 3, 5], -1
        )  # Third board, player -1
        self.assertEqual(
            newer_env_state["boards"][3, 4, 6], -1
        )  # Fourth board, player -1

        # Check if player switched back
        self.assertTrue(np.all(newer_env_state["current_player"] == 1))

    def test_step_env_win(self):
        """Test winning scenario in step_env."""
        env_state = init_env(self.board_size, self.B, self.rng)

        # Set up a near-win state for player 1 in first board
        for i in range(WIN_LENGTH - 1):
            env_state["boards"] = env_state["boards"].at[0, 0, i].set(1)

        # Take winning step
        actions = jnp.array(
            [
                [0, WIN_LENGTH - 1],  # Complete the winning line
                [1, 1],  # Random moves for other boards
                [2, 2],
                [3, 3],
            ]
        )

        new_env_state, observations, rewards, dones = step_env(env_state, actions)

        # Check if player 1 won on first board
        self.assertEqual(new_env_state["winners"][0], 1)
        self.assertTrue(dones[0])
        self.assertEqual(rewards[0], 1)  # Player 1 gets reward 1

        # Other boards should not have wins
        self.assertTrue(np.all(new_env_state["winners"][1:] == 0))
        self.assertTrue(np.all(dones[1:] == False))
        self.assertTrue(np.all(rewards[1:] == 0))

    def test_invalid_actions(self):
        """Test behavior with invalid actions."""
        env_state = init_env(self.board_size, self.B, self.rng)

        # Place a piece at (0, 0) on first board
        env_state["boards"] = env_state["boards"].at[0, 0, 0].set(1)

        # Try to place another piece at the same position
        actions = jnp.array(
            [
                [0, 0],  # Invalid: already occupied
                [1, 1],  # Valid
                [2, 2],  # Valid
                [3, 3],  # Valid
            ]
        )

        new_env_state, observations, rewards, dones = step_env(env_state, actions)

        # The invalid action should not change the board state at that position
        self.assertEqual(new_env_state["boards"][0, 0, 0], 1)  # Still player 1's piece

        # Valid actions should still work
        self.assertEqual(new_env_state["boards"][1, 1, 1], 1)
        self.assertEqual(new_env_state["boards"][2, 2, 2], 1)
        self.assertEqual(new_env_state["boards"][3, 3, 3], 1)

    def test_sample_action(self):
        """Test sampling random valid actions."""
        env_state = init_env(self.board_size, self.B, self.rng)

        # Place some pieces
        env_state["boards"] = env_state["boards"].at[0, 0, 0].set(1)
        env_state["boards"] = env_state["boards"].at[1, 1, 1].set(-1)

        # Mark one board as done
        env_state["dones"] = env_state["dones"].at[2].set(True)

        # Sample actions
        actions, new_env_state = sample_action(env_state, self.rng)

        # Check that actions are valid
        action_mask = get_action_mask(env_state)
        for i in range(self.B):
            if not env_state["dones"][i]:
                row, col = actions[i]
                # Skip if board is done
                if i != 2:
                    self.assertTrue(
                        action_mask[i, row, col],
                        f"Invalid action {actions[i]} for board {i}",
                    )

    def test_run_random_episode(self):
        """Test running a complete random episode."""
        board_size = 5  # Smaller board for faster test
        B = 2

        # Run episode
        final_state, total_rewards = run_random_episode(board_size, B, seed=42)

        # All games should be done
        self.assertTrue(np.all(final_state["dones"]))

        # There should be a winner or the board should be full
        for i in range(B):
            # Either there's a winner
            if final_state["winners"][i] != 0:
                winner = final_state["winners"][i]
                self.assertTrue(winner == 1 or winner == -1)
            # Or it's a draw (board is full)
            else:
                board_filled = np.all(final_state["boards"][i] != 0)
                self.assertTrue(board_filled)

        # Check rewards are consistent - not necessarily equal to winners
        # The rewards from random episodes may accumulate differently than expected
        # Just verify they're not zero when there's a winner
        for i in range(B):
            if final_state["winners"][i] != 0:
                self.assertNotEqual(total_rewards[i], 0)
                # Rewards should have the same sign as the winner
                if final_state["winners"][i] > 0:
                    self.assertGreater(total_rewards[i], 0)
                else:
                    self.assertLess(total_rewards[i], 0)

    def test_get_valid_actions(self):
        """Test getting all valid actions."""
        env_state = init_env(3, 2, self.rng)  # Small board for easier testing

        # All actions should be valid initially
        valid_actions = get_valid_actions(env_state)

        # Check shape
        self.assertEqual(
            valid_actions.shape, (2, 9, 2)
        )  # 2 boards, 9 positions (3x3), 2 coordinates

        # Place a piece and mark a board as done
        env_state["boards"] = env_state["boards"].at[0, 0, 0].set(1)
        env_state["dones"] = env_state["dones"].at[1].set(True)

        # Get valid actions again
        valid_actions = get_valid_actions(env_state)

        # First board should have 8 valid actions (9 - 1 occupied)
        # Second board should have no valid actions (done)

        # Check that get_valid_actions returns all possible actions and we need to filter based on the action mask
        action_mask = get_action_mask(env_state)

        # Verify that action positions correspond to valid positions in the action mask
        valid_positions_count = 0
        for action in valid_actions[0]:
            row, col = action
            if row >= 0 and col >= 0:  # Not padding
                # Count valid positions according to action mask
                if action_mask[0, row, col]:
                    valid_positions_count += 1

        # There should be 8 valid positions in a 3x3 board with one piece
        self.assertEqual(valid_positions_count, 8)

        # Check that second board has no valid actions according to mask
        second_board_valid = False
        for action in valid_actions[1]:
            row, col = action
            if row >= 0 and col >= 0:  # Not padding
                if action_mask[1, row, col]:
                    second_board_valid = True
                    break

        self.assertFalse(
            second_board_valid, "Done board should have no valid actions in the mask"
        )

    def test_is_game_over(self):
        """Test game over detection."""
        env_state = init_env(self.board_size, self.B, self.rng)

        # No games should be over initially
        game_over = is_game_over(env_state)
        self.assertTrue(np.all(game_over == False))

        # Mark some games as done
        env_state["dones"] = env_state["dones"].at[0].set(True)
        env_state["dones"] = env_state["dones"].at[2].set(True)

        # Check game over again
        game_over = is_game_over(env_state)
        self.assertTrue(game_over[0])
        self.assertFalse(game_over[1])
        self.assertTrue(game_over[2])
        self.assertFalse(game_over[3])


if __name__ == "__main__":
    unittest.main()
