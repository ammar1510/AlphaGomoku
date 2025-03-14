from re import T

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from env.gomoku import Gomoku


def test_init():
    """Test initialization with different parameters"""
    # Default initialization
    env = Gomoku()
    assert env.board_size == 15
    assert env.num_boards == 1
    assert env.mode == "train"

    # Custom parameters for training mode
    env = Gomoku(board_size=10, num_boards=5, mode="train")
    assert env.board_size == 10
    assert env.num_boards == 5
    assert env.mode == "train"
    assert env.board.shape == (5, 10, 10)

    # Human mode should always use num_boards=1
    env = Gomoku(board_size=10, num_boards=1, mode="human")
    assert env.board_size == 10
    assert env.num_boards == 1
    assert env.mode == "human"
    assert env.board.shape == (1, 10, 10)


def test_reset():
    """Test reset function"""
    env = Gomoku(board_size=10, num_boards=2)

    # Make some moves to change the state
    actions = jnp.array([[1, 1], [2, 2]])
    env.step(actions)

    # Reset and check if state is cleared
    env.reset()
    assert jnp.all(env.board == 0)
    assert jnp.all(env.current_player == 1)
    assert jnp.all(env.dones == False)
    assert jnp.all(env.winners == 0)
    assert env.board.shape == (2, 10, 10)


def test_step_valid_move():
    """Test making valid moves"""
    env = Gomoku(board_size=10, num_boards=1)

    # First player (black) plays at (1, 1)
    actions = jnp.array([[1, 1]])
    next_board, rewards, dones = env.step(actions)

    # Check board state
    assert env.board[0, 1, 1] == 1  # Black stone placed
    assert jnp.all(env.current_player == -1)  # Now white's turn
    assert jnp.all(rewards == 0)  # No win yet
    assert jnp.all(dones == False)  # Game not over

    # Second player (white) plays at (2, 2)
    actions = jnp.array([[2, 2]])
    next_board, rewards, dones = env.step(actions)

    # Check board state
    assert env.board[0, 2, 2] == -1  # White stone placed
    assert jnp.all(env.current_player == 1)  # Back to black's turn


def test_step_occupied_cell():
    """Test attempting to play on an occupied cell"""
    env = Gomoku(board_size=10, num_boards=1)

    # First player plays at (1, 1)
    actions = jnp.array([[1, 1]])
    env.step(actions)

    # Second player tries to play at the same location
    actions = jnp.array([[1, 1]])
    next_board, rewards, dones = env.step(actions)

    # Check that the move wasn't made (cell still contains first player's stone)
    assert env.board[0, 1, 1] == 1


def test_horizontal_win():
    """Test horizontal win detection"""
    env = Gomoku(board_size=10, num_boards=1)

    # Black places 5 stones in a row horizontally
    for i in range(5):
        actions = jnp.array([[1, i]])
        next_board, rewards, dones = env.step(actions)

        # White's non-interfering moves
        if i < 4:  # Skip last white move as game ends
            actions = jnp.array([[2, i]])
            next_board, rewards, dones = env.step(actions)

    # Check win condition
    assert jnp.all(dones == True)
    assert env.winners[0] == 1  # Black wins
    assert rewards[0] == 1  # Reward for winning


def test_vertical_win():
    """Test vertical win detection"""
    env = Gomoku(board_size=10, num_boards=1)

    # Black places 5 stones in a row vertically
    for i in range(5):
        actions = jnp.array([[i, 1]])
        next_board, rewards, dones = env.step(actions)

        # White's non-interfering moves
        if i < 4:  # Skip last white move as game ends
            actions = jnp.array([[i, 2]])
            next_board, rewards, dones = env.step(actions)

    # Check win condition
    assert jnp.all(dones == True)
    assert env.winners[0] == 1  # Black wins


def test_diagonal_win():
    """Test diagonal win detection"""
    env = Gomoku(board_size=10, num_boards=1)

    # Black places 5 stones in a row diagonally
    for i in range(5):
        actions = jnp.array([[i, i]])
        next_board, rewards, dones = env.step(actions)

        # White's non-interfering moves
        if i < 4:  # Skip last white move as game ends
            actions = jnp.array([[i, i + 1]])
            next_board, rewards, dones = env.step(actions)

    # Check win condition
    assert jnp.all(dones == True)
    assert env.winners[0] == 1  # Black wins


def test_anti_diagonal_win():
    """Test anti-diagonal win detection"""
    env = Gomoku(board_size=10, num_boards=1)

    # Black places 5 stones in a row anti-diagonally
    for i in range(5):
        actions = jnp.array([[i, 4 - i]])
        next_board, rewards, dones = env.step(actions)

        # White's non-interfering moves
        if i < 4:  # Skip last white move as game ends
            actions = jnp.array([[i, 5 - i]])
            next_board, rewards, dones = env.step(actions)

    # Check win condition
    assert jnp.all(dones == True)
    assert env.winners[0] == 1  # Black wins


def test_draw():
    """Test draw condition"""
    env = Gomoku(board_size=3, num_boards=1)  # Small board for quick draw

    # Fill board in a way that neither player wins
    # Row 0: B W B
    # Row 1: W B W
    # Row 2: W B W
    moves = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 0), (1, 2), (2, 1), (2, 0), (2, 2)]

    for row, col in moves:
        actions = jnp.array([[row, col]])
        next_board, rewards, dones = env.step(actions)

    # Check draw condition
    assert jnp.all(dones == True)
    assert env.winners[0] == 0  # No winner
    assert rewards[0] == 0  # No reward for draw


def test_action_mask():
    """Test get_action_mask function"""
    env = Gomoku(board_size=5, num_boards=1)

    # Initially all positions should be valid
    mask = env.get_action_mask()
    assert jnp.all(mask == True)

    # Make some moves
    actions = jnp.array([[1, 1]])
    env.step(actions)

    actions = jnp.array([[2, 2]])
    env.step(actions)

    # Check mask again
    mask = env.get_action_mask()
    assert mask[0, 1, 1] == False  # Occupied by first move
    assert mask[0, 2, 2] == False  # Occupied by second move
    assert mask[0, 0, 0] == True  # Still empty


def test_multiple_boards():
    """Test running multiple boards in parallel"""
    num_boards = 3
    env = Gomoku(board_size=10, num_boards=num_boards)

    # Make different moves on each board
    actions = jnp.array([[1, 1], [2, 2], [3, 3]])  # Board 0  # Board 1  # Board 2

    next_board, rewards, dones = env.step(actions)

    # Check each board state
    assert env.board[0, 1, 1] == 1
    assert env.board[1, 2, 2] == 1
    assert env.board[2, 3, 3] == 1

    # Create wins on boards 0 and 2 only, with no win on board 1
    for i in range(1, 6):
        # White moves for all boards
        if i < 5:
            white_actions = jnp.array(
                [[2, i], [4 + i, 4 - i], [4 + i, 3]]  # Board 0  # Board 1  # Board 2
            )
            next_board, rewards, dones = env.step(white_actions)

        # Black moves for all boards
        black_actions = jnp.array(
            [
                [1, 1 + i],  # Board 0 - continuing horizontal line for win
                [i % 3 + 1, i + 3],  # Board 1 - scattered pattern that won't win
                [3, 3 + i],  # Board 2 - horizontal line for win
            ]
        )
        next_board, rewards, dones = env.step(black_actions)

    # Print the final board state
    print("\nFinal board states:")
    for board_idx in range(num_boards):
        print(f"\nBoard {board_idx}:")
        board = np.array(env.board[board_idx])  # Convert to numpy for easier printing
        for row in range(env.board_size):
            row_str = ""
            for col in range(env.board_size):
                cell = board[row, col]
                if cell == 1:
                    row_str += "X "  # Black
                elif cell == -1:
                    row_str += "O "  # White
                else:
                    row_str += ". "  # Empty
            print(row_str)

    # Check win conditions
    assert dones[0] == True  # Board 0 should be done
    assert dones[1] == False  # Board 1 should NOT be done
    assert dones[2] == True  # Board 2 should be done
    assert env.winners[0] == 1  # Black won on board 0
    assert env.winners[1] == 0  # No winner on board 1
    assert env.winners[2] == 1  # Black won on board 2


def test_board_normalization():
    """Test board normalization from current player's perspective"""
    env = Gomoku(board_size=5, num_boards=1)

    # Black plays at (1, 1)
    actions = jnp.array([[1, 1]])
    next_board, _, _ = env.step(actions)

    # Board from white's perspective: black's stone should be -1
    assert next_board[0, 1, 1] == -1

    # White plays at (2, 2)
    actions = jnp.array([[2, 2]])
    next_board, _, _ = env.step(actions)

    # Board from black's perspective: white's stone should be -1, black's stone should be 1
    assert next_board[0, 1, 1] == 1
    assert next_board[0, 2, 2] == -1
