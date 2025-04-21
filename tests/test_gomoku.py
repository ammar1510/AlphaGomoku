import pytest
import jax
import jax.numpy as jnp
from alphagomoku.environments.gomoku import GomokuJaxEnv, GomokuState

# --- Fixtures ---

@pytest.fixture
def env_config_default():
    """Default configuration for testing (5x5, win 3)."""
    return {"B": 2, "board_size": 5, "win_length": 3}

@pytest.fixture
def env_config_draw():
    """Configuration for draw testing (3x3, win 3)."""
    return {"B": 1, "board_size": 3, "win_length": 3}

@pytest.fixture
def gomoku_env_default(env_config_default):
    """Fixture to create a default GomokuJaxEnv instance."""
    return GomokuJaxEnv(**env_config_default)

@pytest.fixture
def gomoku_env_draw(env_config_draw):
    """Fixture to create a GomokuJaxEnv instance for draw tests."""
    return GomokuJaxEnv(**env_config_draw)

@pytest.fixture
def initial_rng():
    """Fixture for a repeatable RNG key."""
    return jax.random.PRNGKey(42)

# --- Tests ---

def test_env_initialization(gomoku_env_default, env_config_default):
    """Test if the environment initializes with correct parameters."""
    assert gomoku_env_default.B == env_config_default["B"]
    assert gomoku_env_default.board_size == env_config_default["board_size"]
    assert gomoku_env_default.win_length == env_config_default["win_length"]
    assert gomoku_env_default.win_kernels.shape == (env_config_default["win_length"], env_config_default["win_length"], 1, 4)
    assert gomoku_env_default.observation_shape == (env_config_default["board_size"], env_config_default["board_size"])
    assert gomoku_env_default.action_shape == (2,)

def test_init_state(env_config_default, initial_rng):
    """Test the static init_state method."""
    B = env_config_default["B"]
    board_size = env_config_default["board_size"]
    state = GomokuJaxEnv.init_state(initial_rng, B, board_size)

    assert isinstance(state, GomokuState)
    assert state.boards.shape == (B, board_size, board_size)
    assert jnp.all(state.boards == 0)
    assert state.current_players.shape == (B,)
    assert jnp.all(state.current_players == 1)
    assert state.dones.shape == (B,)
    assert jnp.all(state.dones == False)
    assert state.winners.shape == (B,)
    assert jnp.all(state.winners == 0)
    assert jnp.array_equal(state.rng, initial_rng)

@pytest.mark.parametrize(
    "board_config, player, expected_win",
    [
        # No win
        (jnp.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0,-1, 0,-1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=jnp.float32), 1, False),
        # Horizontal win (player 1)
        (jnp.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0,-1, 0,-1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=jnp.float32), 1, True),
        # Vertical win (player -1)
        (jnp.array([
            [0, 0,-1, 0, 0],
            [0, 1, -1, 1, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=jnp.float32), -1, True),
        # Diagonal win (player 1)
        (jnp.array([
            [1, 0, 0, 0, 0],
            [0, 1,-1, 0, 0],
            [0,-1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=jnp.float32), 1, True),
        # Anti-diagonal win (player -1)
        (jnp.array([
            [0, 0, 0,-1, 0],
            [0, 1,-1, 1, 0],
            [0,-1, 0, 0, 0],
            [-1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=jnp.float32), -1, True),
        # Border horizontal win (player 1)
        (jnp.array([
            [1, 1, 1, 0, 0],
            [0,-1,-1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=jnp.float32), 1, True),
        # Border vertical win (player -1)
        (jnp.array([
            [0, 0, 0, 0,-1],
            [0, 1, 1, 0,-1],
            [0, 0, 0, 0,-1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=jnp.float32), -1, True),
    ]
)
def test_check_win(gomoku_env_default, board_config, player, expected_win):
    """Test the _check_win method with various board configurations."""
    B = gomoku_env_default.B
    # Create batched inputs
    boards = jnp.stack([board_config] * B, axis=0) # Same board for both envs in batch
    players = jnp.array([player] * B, dtype=jnp.int32)

    win_results = gomoku_env_default._check_win(boards, players)

    assert win_results.shape == (B,)
    assert jnp.all(win_results == expected_win)

def test_step_valid_move(gomoku_env_default, env_config_default, initial_rng):
    """Test a single valid step in the environment."""
    B = env_config_default["B"]
    board_size = env_config_default["board_size"]
    state = GomokuJaxEnv.init_state(initial_rng, B, board_size)

    # Action: Place piece at (2, 2) for both environments
    actions = jnp.array([[2, 2]] * B, dtype=jnp.int32)

    new_state, observations, rewards, dones, info = gomoku_env_default.step(state, actions)

    # Check state update
    assert new_state.boards[0, 2, 2] == 1  # Player 1 moved
    assert new_state.boards[1, 2, 2] == 1
    assert jnp.all(new_state.current_players == -1) # Player should switch to -1
    assert jnp.all(new_state.dones == False)
    assert jnp.all(new_state.winners == 0)

    # Check outputs
    assert observations.shape == (B, board_size, board_size)
    assert jnp.all(observations == new_state.boards)
    assert rewards.shape == (B,)
    assert jnp.all(rewards == 0.0)
    assert dones.shape == (B,)
    assert jnp.all(dones == False)
    assert isinstance(info, dict)

def test_step_invalid_move_occupied(gomoku_env_default, env_config_default, initial_rng):
    """Test stepping on an already occupied square."""
    B = env_config_default["B"]
    board_size = env_config_default["board_size"]
    state = GomokuJaxEnv.init_state(initial_rng, B, board_size)

    # Make a first move
    actions_1 = jnp.array([[2, 2]] * B, dtype=jnp.int32)
    state, _, _, _, _ = gomoku_env_default.step(state, actions_1)
    # state.boards[b, 2, 2] is now 1, player is -1

    # Attempt to move on the same square again
    actions_2 = jnp.array([[2, 2]] * B, dtype=jnp.int32)
    new_state, observations, rewards, dones, info = gomoku_env_default.step(state, actions_2)

    # Check state didn't change (except RNG potentially)
    assert jnp.all(new_state.boards == state.boards)
    assert jnp.all(new_state.current_players == state.current_players) # Player -1 remains
    assert jnp.all(new_state.dones == state.dones)
    assert jnp.all(new_state.winners == state.winners)

    # Check outputs reflect no change
    assert jnp.all(observations == state.boards)
    assert jnp.all(rewards == 0.0)
    assert jnp.all(dones == False)

def test_step_invalid_move_after_done(gomoku_env_default, env_config_default, initial_rng):
    """Test stepping after the game has ended."""
    B = env_config_default["B"]
    board_size = env_config_default["board_size"]
    win_length = env_config_default["win_length"]

    # Create a state where player 1 has just won (needs win_length = 3)
    boards = jnp.zeros((B, board_size, board_size), dtype=jnp.float32)
    boards = boards.at[:, 1, 0:win_length].set(1)
    current_players = jnp.ones((B,), dtype=jnp.int32)
    dones = jnp.ones((B,), dtype=jnp.bool_)
    winners = jnp.ones((B,), dtype=jnp.int32)

    state = GomokuState(
        boards=boards,
        current_players=current_players, # Player is still 1, as game just ended
        dones=dones,
        winners=winners,
        rng=initial_rng
    )

    # Attempt to make another move
    actions = jnp.array([[3, 3]] * B, dtype=jnp.int32)
    new_state, observations, rewards, dones_out, info = gomoku_env_default.step(state, actions)

    # Check state didn't change
    assert jnp.all(new_state.boards == state.boards)
    assert jnp.all(new_state.current_players == state.current_players)
    assert jnp.all(new_state.dones == state.dones)
    assert jnp.all(new_state.winners == state.winners)

    # Check outputs reflect no change and game is done
    assert jnp.all(observations == state.boards)
    assert jnp.all(rewards == 0.0) # No reward for move after win
    assert jnp.all(dones_out == True)

def test_step_win_condition(gomoku_env_default, env_config_default, initial_rng):
    """Test the step function resulting in a win."""
    B = env_config_default["B"]
    board_size = env_config_default["board_size"]
    win_length = env_config_default["win_length"] # Needs 3 for this setup
    state = GomokuJaxEnv.init_state(initial_rng, B, board_size)

    # Setup board for player 1 to win on next move
    boards = state.boards
    boards = boards.at[:, 1, 0:win_length-1].set(1) # Player 1 places two
    boards = boards.at[:, 2, 0:win_length-1].set(-1) # Player -1 places two
    state = state._replace(boards=boards, current_players=jnp.ones((B,), dtype=jnp.int32))

    # Action: Player 1 places the winning piece
    actions = jnp.array([[1, win_length-1]] * B, dtype=jnp.int32)
    new_state, observations, rewards, dones, info = gomoku_env_default.step(state, actions)

    # Check state update for win
    assert new_state.boards[0, 1, win_length-1] == 1
    assert jnp.all(new_state.dones == True)
    assert jnp.all(new_state.winners == 1) # Player 1 wins
    assert jnp.all(new_state.current_players == 1) # Player does not switch after win

    # Check outputs for win
    assert jnp.all(observations == new_state.boards)
    assert jnp.all(rewards == 1.0) # Reward for winning move
    assert jnp.all(dones == True)

def test_step_draw_condition(gomoku_env_draw, env_config_draw, initial_rng):
    """Test the step function resulting in a draw (board full, no win)."""
    # Use 3x3 board, win_length 3, B=1
    B = env_config_draw["B"]
    board_size = env_config_draw["board_size"]
    state = GomokuJaxEnv.init_state(initial_rng, B, board_size)

    # Fill the board in a pattern that doesn't lead to a win
    # 1  -1   1
    # -1 -1   1
    # 1   1  -1 
    # Next move is player -1 at (2, 2)
    boards = jnp.array([
        [[ 1, -1,  1],
         [-1, -1,  1],
         [ 1,  1,  0]] # Last spot empty
    ], dtype=jnp.float32)
    current_players = jnp.array([-1], dtype=jnp.int32)
    state = state._replace(boards=boards, current_players=current_players)

    # Action: Player -1 places the last piece
    actions = jnp.array([[2, 2]], dtype=jnp.int32) # (B, 2)
    new_state, observations, rewards, dones, info = gomoku_env_draw.step(state, actions)

    # Check state update for draw
    assert new_state.boards[0, 2, 2] == -1
    assert jnp.all(new_state.boards != 0) # Board is full
    assert jnp.all(new_state.dones == True) # Game is done
    assert jnp.all(new_state.winners == 0) # No winner (draw)
    assert jnp.all(new_state.current_players == -1) # Player doesn't switch after draw

    # Check outputs for draw
    assert jnp.all(observations == new_state.boards)
    assert jnp.all(rewards == 0.0) # No reward for draw
    assert jnp.all(dones == True)

def test_reset(gomoku_env_default, env_config_default, initial_rng):
    """Test resetting the environment."""
    B = env_config_default["B"]
    board_size = env_config_default["board_size"]
    state = GomokuJaxEnv.init_state(initial_rng, B, board_size)

    # Take a step
    actions = jnp.array([[0, 0]] * B, dtype=jnp.int32)
    state, _, _, _, _ = gomoku_env_default.step(state, actions)

    # Reset the environment
    reset_rng = jax.random.PRNGKey(99)
    new_state, observations, info = gomoku_env_default.reset(reset_rng)

    # Check if the new state is like an initial state
    assert isinstance(new_state, GomokuState)
    assert new_state.boards.shape == (B, board_size, board_size)
    assert jnp.all(new_state.boards == 0)
    assert new_state.current_players.shape == (B,)
    assert jnp.all(new_state.current_players == 1)
    assert new_state.dones.shape == (B,)
    assert jnp.all(new_state.dones == False)
    assert new_state.winners.shape == (B,)
    assert jnp.all(new_state.winners == 0)
    assert jnp.array_equal(new_state.rng, reset_rng) # Check if new RNG key is used

    # Check outputs
    assert observations.shape == (B, board_size, board_size)
    assert jnp.all(observations == 0)
    assert isinstance(info, dict)

def test_get_action_mask_initial(gomoku_env_default, env_config_default, initial_rng):
    """Test action mask on initial state (all valid)."""
    B = env_config_default["B"]
    board_size = env_config_default["board_size"]
    state = GomokuJaxEnv.init_state(initial_rng, B, board_size)

    mask = gomoku_env_default.get_action_mask(state)

    assert mask.shape == (B, board_size, board_size)
    assert mask.dtype == jnp.bool_
    assert jnp.all(mask == True) # All actions initially valid

def test_get_action_mask_ongoing(gomoku_env_default, env_config_default, initial_rng):
    """Test action mask during an ongoing game."""
    B = env_config_default["B"]
    board_size = env_config_default["board_size"]
    state = GomokuJaxEnv.init_state(initial_rng, B, board_size)

    # Make a move
    actions = jnp.array([[1, 1]] * B, dtype=jnp.int32)
    state, _, _, _, _ = gomoku_env_default.step(state, actions)

    mask = gomoku_env_default.get_action_mask(state)

    assert mask.shape == (B, board_size, board_size)
    assert mask.dtype == jnp.bool_
    assert jnp.all(mask[:, 1, 1] == False) # Action at (1, 1) is now invalid
    # Check a random other spot is still valid
    assert jnp.all(mask[:, 0, 0] == True)
    # Count False values - should be B (one per board)
    assert jnp.sum(~mask) == B

def test_get_action_mask_done(gomoku_env_default, env_config_default, initial_rng):
    """Test action mask after game is done (all invalid)."""
    B = env_config_default["B"]
    board_size = env_config_default["board_size"]
    win_length = env_config_default["win_length"]

    # Create a state where player 1 has just won
    boards = jnp.zeros((B, board_size, board_size), dtype=jnp.float32)
    boards = boards.at[:, 1, 0:win_length].set(1)
    current_players = jnp.ones((B,), dtype=jnp.int32)
    dones = jnp.ones((B,), dtype=jnp.bool_)
    winners = jnp.ones((B,), dtype=jnp.int32)

    state = GomokuState(
        boards=boards,
        current_players=current_players,
        dones=dones, # Game is done
        winners=winners,
        rng=initial_rng
    )

    mask = gomoku_env_default.get_action_mask(state)

    assert mask.shape == (B, board_size, board_size)
    assert mask.dtype == jnp.bool_
    assert jnp.all(mask == False) # All actions invalid when done

def test_initialize_trajectory_buffers(gomoku_env_default, env_config_default):
    """Test the creation of trajectory buffers."""
    B = env_config_default["B"]
    max_steps = 10
    obs_shape = gomoku_env_default.observation_shape
    act_shape = gomoku_env_default.action_shape

    buffers = gomoku_env_default.initialize_trajectory_buffers(max_steps)
    observations, actions, rewards, dones, log_probs, current_players_buffer = buffers

    assert observations.shape == (max_steps, B) + obs_shape
    assert observations.dtype == jnp.float32
    assert jnp.all(observations == 0)

    assert actions.shape == (max_steps, B) + act_shape
    assert actions.dtype == jnp.int32
    assert jnp.all(actions == 0)

    assert rewards.shape == (max_steps, B)
    assert rewards.dtype == jnp.float32
    assert jnp.all(rewards == 0)

    assert dones.shape == (max_steps, B)
    assert dones.dtype == jnp.bool_
    assert jnp.all(dones == False)

    assert log_probs.shape == (max_steps, B)
    assert log_probs.dtype == jnp.float32
    assert jnp.all(log_probs == 0)

    assert current_players_buffer.shape == (max_steps, B)
    assert current_players_buffer.dtype == jnp.int32
    assert jnp.all(current_players_buffer == 0) 