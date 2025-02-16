import pytest
import numpy as np
import jax
import jax.numpy as jnp
from gomoku.env import Gomoku, WIN_LENGTH, CELL_SIZE

# Dummy renderer to avoid opening an actual window in tests.
class DummyRenderer:
    def __init__(self):
        self.render_called = False
        self.events_called = False
        self.pause_called = False
        self.close_called = False

    def render_board(self, board):
        self.render_called = True

    def process_events(self):
        self.events_called = True

    def pause(self):
        self.pause_called = True

    def close(self):
        self.close_called = True

# ---------------------------
# Fixtures
# ---------------------------
@pytest.fixture
def env_train():
    # Create an environment in train mode (no renderer)
    env = Gomoku(board_size=9, mode="train")
    env.reset()
    return env

@pytest.fixture
def env_human(monkeypatch):
    # Instead of creating an actual renderer, override GomokuRenderer in the env module.
    dummy = DummyRenderer()
    monkeypatch.setattr("env.GomokuRenderer", lambda board_size, cell_size: dummy)
    env = Gomoku(board_size=9, mode="human")
    env.reset()  # This creates the renderer, which now is our dummy.
    return env, dummy

# ---------------------------
# Test Reset and Initialization
# ---------------------------
def test_reset_initialization():
    env = Gomoku(board_size=9, mode="train")
    board, info = env.reset()
    # Board should be 9x9 zeros.
    np_board = np.array(board)
    assert np_board.shape == (9, 9)
    assert np.all(np_board == 0)
    # Current player is reset to 1 and game is not over.
    assert env.current_player == 1
    assert env.done is False

# ---------------------------
# Test Action Mask
# ---------------------------
def test_get_action_mask(env_train):
    env = env_train
    mask = env.get_action_mask()
    np_mask = np.array(mask)
    # All positions available at reset.
    assert np.all(np_mask)
    # Place a stone in the middle.
    env.board = env.board.at[4, 4].set(1)
    mask = env.get_action_mask()
    np_mask = np.array(mask)
    # The (4,4) spot is now unavailable.
    assert np_mask[4, 4] is False
    assert np.count_nonzero(np_mask) == 9 * 9 - 1

# ---------------------------
# Test Legal Move (step)
# ---------------------------
def test_step_valid_move(env_train):
    env = env_train
    initial_player = env.current_player
    new_board, reward, done, info = env.step((0, 0))
    board_np = np.array(new_board)
    # The move should set (0,0) to the current player's value.
    assert board_np[0, 0] == initial_player
    # With only one move, there is no win or draw.
    assert reward == 0.0
    assert done is False
    assert env.current_player == -initial_player  # Player should flip.

# ---------------------------
# Test Illegal Move
# ---------------------------
def test_step_illegal_move(env_train):
    env = env_train
    env.step((0, 0))
    with pytest.raises(ValueError, match="Illegal move"):
        env.step((0, 0))

# ---------------------------
# Test Win Condition
# ---------------------------
def test_win_condition():
    env = Gomoku(board_size=9, mode="train")
    env.reset()
    # Simulate a win by manually setting five consecutive stones
    for col in range(WIN_LENGTH):
        env.board = env.board.at[0, col].set(env.current_player)
    # The _check_win should return True from the current player's perspective.
    assert env._check_win() is True

    # Also test that step() detects win.
    env.reset()
    # Force the board so that placing at (0, WIN_LENGTH-1) wins the game.
    env.current_player = 1
    for col in range(WIN_LENGTH - 1):
        env.board = env.board.at[0, col].set(1)
    new_board, reward, done, info = env.step((0, WIN_LENGTH - 1))
    assert done is True
    assert info.get("result") == "win"
    assert info.get("winner") == 1
    assert reward == 1.0  # Since current player 1 wins.

# ---------------------------
# Test Draw Condition
# ---------------------------
def test_draw_condition():
    # Use a small board (e.g., 3x3) where a win is impossible (WIN_LENGTH=5) so that full board implies a draw.
    env = Gomoku(board_size=3, mode="train")
    env.reset()
    # Fill the board manually with alternating moves.
    moves = [(i, j) for i in range(3) for j in range(3)]
    current = 1
    for move in moves[:-1]:
        env.board = env.board.at[move[0], move[1]].set(current)
        current *= -1
    # The final move should trigger a draw.
    new_board, reward, done, info = env.step(moves[-1])
    assert done is True
    assert info.get("result") == "draw"

# ---------------------------
# Test to() Method (Device Switching)
# ---------------------------
def test_to_device(env_train):
    env = env_train
    # Capture the current device (on CPU, this should be "cpu")
    initial_device = env.board.device_buffer.device()
    # Call to() with the same device.
    env.to(env.device)
    # Makes sure the board and kernels are on the expected device.
    assert env.board.device_buffer.device() == env.device
    assert env.kernels.device_buffer.device() == env.device

# ---------------------------
# Test Human Mode Rendering
# ---------------------------
def test_human_mode(env_human):
    env, dummy = env_human
    # When reset() is called in human mode, _update_human_display() is invoked,
    # which in turn calls renderer.render_board() and renderer.process_events()
    # Our dummy renderer should reflect that.
    assert dummy.render_called is True
    assert dummy.events_called is True 