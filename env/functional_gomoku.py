import time
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

# Constants
WIN_LENGTH = 5


def create_win_kernels():
    """
    Create convolution kernels for win detection.

    Returns:
        kernels: Array of kernels for win detection
    """
    ones = jnp.ones((1, WIN_LENGTH), dtype=jnp.float32)
    zeros = jnp.zeros((WIN_LENGTH - 1, WIN_LENGTH), dtype=jnp.float32)

    horizontal = jnp.expand_dims(jnp.vstack([ones, zeros]), axis=(0, 1))
    vertical = jnp.expand_dims(jnp.vstack([ones, zeros]).T, axis=(0, 1))
    diagonal = jnp.expand_dims(jnp.eye(WIN_LENGTH, dtype=jnp.float32), axis=(0, 1))
    anti_diagonal = jnp.expand_dims(
        jnp.fliplr(jnp.eye(WIN_LENGTH, dtype=jnp.float32)), axis=(0, 1)
    )

    kernels = jnp.concatenate(
        [horizontal, vertical, diagonal, anti_diagonal], axis=0
    ).transpose(2, 3, 1, 0)
    return kernels


# Pre-compute kernels (these are static and don't need to be part of the state)
WIN_KERNELS = create_win_kernels()


# @jax.jit
def init_env(rng, board_size=15, num_boards=1):
    """
    Initialize the Gomoku environment state.

    Args:
        rng: JAX random key
        board_size: Size of the Gomoku board
        num_boards: Number of parallel boards

    Returns:
        env_state: Dictionary containing the environment state
    """
    return {
        "board": jnp.zeros((num_boards, board_size, board_size), dtype=jnp.float32),
        "current_player": jnp.ones(
            (num_boards,), dtype=jnp.int32
        ),  # 1 for black, -1 for white
        "dones": jnp.zeros((num_boards,), dtype=jnp.bool_),
        "winners": jnp.zeros((num_boards,), dtype=jnp.int32),
        "board_size": board_size,
        "num_boards": num_boards,
        "rng": rng,
    }


# @jax.jit
def reset_env(env_state, new_rng=None):
    """
    Reset the environment to initial state.

    Args:
        env_state: Current environment state
        new_rng: Optional new random key

    Returns:
        new_env_state: Reset environment state
        observations: Initial observations
    """
    board_size = env_state["board_size"]
    num_boards = env_state["num_boards"]

    # Use new RNG if provided, otherwise use the one from env_state
    rng = new_rng if new_rng is not None else jax.random.split(env_state["rng"])[0]

    new_env_state = {
        "board": jnp.zeros((num_boards, board_size, board_size), dtype=jnp.float32),
        "current_player": jnp.ones((num_boards,), dtype=jnp.int32),
        "dones": jnp.zeros((num_boards,), dtype=jnp.bool_),
        "winners": jnp.zeros((num_boards,), dtype=jnp.int32),
        "board_size": board_size,
        "num_boards": num_boards,
        "rng": rng,
    }

    # Initial observation is the empty board from perspective of current player (always 1 at start)
    observations = new_env_state["board"]

    return new_env_state, observations


# @jax.jit
def check_win(board, current_player):
    """
    Check for wins using convolution.

    Args:
        board: Game boards with shape (num_boards, board_size, board_size)
        current_player: Current player values with shape (num_boards,)

    Returns:
        wins: Boolean array indicating which boards have wins
    """
    # Create player-specific boards (1 where player has pieces, 0 elsewhere)
    current_player_reshaped = current_player[:, jnp.newaxis, jnp.newaxis]
    player_boards = (board == current_player_reshaped).astype(jnp.float32)

    # Add channel dimension for convolution (NHWC format)
    player_boards = player_boards[:, :, :, jnp.newaxis]

    # Define padding to handle edge cases
    padding = ((WIN_LENGTH - 1, WIN_LENGTH - 1), (WIN_LENGTH - 1, WIN_LENGTH - 1))

    # Perform convolution
    conv_output = lax.conv_general_dilated(
        player_boards,
        WIN_KERNELS,
        window_strides=(1, 1),
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )

    # Check if any position has WIN_LENGTH in a row
    win_condition = conv_output == WIN_LENGTH

    # Check if any kernel detected a win for each board
    wins = jnp.any(win_condition, axis=(1, 2, 3))

    return wins


# @jax.jit
def get_action_mask(env_state):
    """
    Get mask of valid actions.

    Args:
        env_state: Current environment state

    Returns:
        action_mask: Boolean mask of valid actions with shape (num_boards, board_size, board_size)
    """
    # Valid actions are empty spaces on boards that aren't done
    board = env_state["board"]
    dones = env_state["dones"]

    # Create mask: True for empty spaces (0), False for occupied
    action_mask = board == 0

    # For done boards, no actions are valid
    action_mask = action_mask & (~dones[:, jnp.newaxis, jnp.newaxis])

    return action_mask


@jax.jit
def step_env(env_state, actions):
    """
    Take a step in the environment.

    Args:
        env_state: Current environment state
        actions: Actions to take, shape (num_boards, 2) where each action is [row, col]

    Returns:
        new_env_state: Updated environment state
        observations: New observations
        rewards: Rewards from this step
        dones: Done flags
    """
    # Create a new state dict to avoid modifying the input
    new_env_state = dict(env_state)

    # Extract state components
    board = env_state["board"]
    current_player = env_state["current_player"]
    dones = env_state["dones"]
    winners = env_state["winners"]

    # Get shapes directly from arrays
    num_envs = board.shape[0]

    # Extract action coordinates
    rows, cols = actions.T

    # Create action mask: only apply actions to non-done boards and empty spaces
    action_mask = (~dones) & (board[jnp.arange(num_envs), rows, cols] == 0)

    # Create a board mask for updates (num_envs, board_height, board_width)
    board_mask = jnp.zeros_like(board, dtype=bool)

    # Update only where action_mask is True (valid actions on non-done boards)
    # We need to create a 3D mask where the action positions are True
    board_mask = board_mask.at[jnp.arange(num_envs), rows, cols].set(action_mask)

    # Use the mask to update the board
    new_board = jnp.where(
        board_mask,  # Shape: (num_envs, height, width)
        current_player[
            :, jnp.newaxis, jnp.newaxis
        ],  # Shape: (num_envs, 1, 1) -> broadcasts to (num_envs, height, width)
        board,  # Shape: (num_envs, height, width)
    )

    new_env_state["board"] = new_board

    # Check for wins - only count new wins (not already done games)
    win_patterns = check_win(new_board, current_player)
    wins = win_patterns & ~dones
    new_winners = jnp.where(wins, current_player, winners)
    new_env_state["winners"] = new_winners

    # Check for draws (all spaces filled and no win)
    draws = jnp.all(new_board != 0, axis=(1, 2)) & ~(win_patterns | dones)

    # Calculate rewards (only for new wins)
    rewards = jnp.where(wins, current_player, 0.0)
    # reward -> 1 for black, -1 for white

    # Update done flags
    new_dones = dones | wins | draws
    new_env_state["dones"] = new_dones

    # Switch player
    new_env_state["current_player"] = -current_player

    # Create observation (board from perspective of current player)
    current_player_reshaped = new_env_state["current_player"][
        :, jnp.newaxis, jnp.newaxis
    ]
    observations = new_board * current_player_reshaped

    return new_env_state, observations, rewards, new_dones


@jax.jit
def get_valid_actions(env_state):
    """
    Get all valid actions for the current state.

    Args:
        env_state: Current environment state

    Returns:
        valid_actions: Array of valid action coordinates for each board
    try to not use this function, rely on the action mask instead
    """
    action_mask = get_action_mask(env_state)
    board_size = env_state["board_size"]

    # Get shapes directly from the array dimensions
    num_envs, height, width = action_mask.shape

    # Create meshgrid of all possible actions
    rows, cols = jnp.meshgrid(jnp.arange(width), jnp.arange(height))
    all_actions = jnp.stack([rows.flatten(), cols.flatten()], axis=1)

    # For each board, filter valid actions
    def get_board_valid_actions(board_idx):
        board_mask = action_mask[board_idx].flatten()
        valid_indices = jnp.where(board_mask, size=height * width)[0]
        # Pad with -1 to ensure fixed size
        padded_indices = jnp.pad(
            valid_indices,
            (0, height * width - valid_indices.shape[0]),
            constant_values=-1,
        )
        return all_actions[padded_indices]

    # Map over all boards
    valid_actions = jax.vmap(get_board_valid_actions)(jnp.arange(num_envs))

    return valid_actions


@jax.jit
def sample_action(env_state, rng):
    """
    Sample a random valid action.

    Args:
        env_state: Current environment state
        rng: JAX random key

    Returns:
        actions: Sampled actions with shape (num_boards, 2)
        new_env_state: Updated environment state with new RNG
    """
    action_mask = get_action_mask(env_state)
    board_size = env_state["board_size"]

    # Get shapes directly from the array dimensions
    num_envs, height, width = action_mask.shape

    # Flatten the action mask for each board
    flat_mask = action_mask.reshape(num_envs, -1)
    total_positions = height * width

    # Generate random numbers for each board
    rng, subkey = jax.random.split(rng)
    random_values = jax.random.uniform(subkey, shape=(num_envs, total_positions))

    # Set invalid action probabilities to -1 so they won't be selected
    masked_random = jnp.where(flat_mask, random_values, -1.0)

    # Get the indices of the maximum values (which will be valid actions)
    flat_actions = jnp.argmax(masked_random, axis=1)

    # Convert flat indices to 2D coordinates
    rows = flat_actions // width
    cols = flat_actions % width
    actions = jnp.stack([rows, cols], axis=1)

    # Update RNG in state
    new_env_state = dict(env_state)
    new_env_state["rng"] = rng

    return actions, new_env_state


@jax.jit
def is_game_over(env_state):
    """
    Check if games are over.

    Args:
        env_state: Current environment state

    Returns:
        game_over: Boolean array indicating which games are over
    """
    return env_state["dones"]


@partial(jax.jit, static_argnums=(0, 1))  # Mark board_size and num_boards as static
def run_random_episode(board_size=15, num_boards=256, seed=0):
    """
    Run a complete episode with random actions.

    Args:
        board_size: Size of the Gomoku board
        num_boards: Number of parallel boards
        seed: Random seed

    Returns:
        final_state: Final environment state
        total_rewards: Total rewards for each board
    """
    # Initialize environment
    rng = jax.random.PRNGKey(seed)
    init_state = init_env(rng, board_size, num_boards)
    env_state, obs = reset_env(init_state)

    # Initialize total rewards
    total_rewards = jnp.zeros((num_boards,))

    # Define loop body function
    def body_fun(loop_state):
        env_state, total_rewards = loop_state

        # Sample random actions
        actions, updated_env_state = sample_action(env_state, env_state["rng"])

        # Take step
        next_env_state, obs, rewards, dones = step_env(updated_env_state, actions)

        # Accumulate rewards
        next_total_rewards = total_rewards + rewards

        # Check if we should continue
        continue_loop = ~jnp.all(next_env_state["dones"])

        return (next_env_state, next_total_rewards), continue_loop

    # Use while_loop for dynamic iteration
    (final_state, final_rewards), _ = jax.lax.while_loop(
        lambda state_tup: state_tup[1],  # Continue while second element is True
        lambda state_tup: body_fun(state_tup[0]),
        ((env_state, total_rewards), True),
    )

    return final_state, final_rewards


def render(env_state, board_idx=0, mode="unicode"):
    """
    Create a string representation of the Gomoku board state.

    Args:
        env_state: Environment state dictionary
        board_idx: Index of the board to render (if multiple boards exist)
        mode: Rendering mode - 'unicode' for unicode characters, 'ascii' for simple ASCII

    Returns:
        board_str: String representation of the board
    """
    # Ensure board_idx is within range
    board_idx = min(board_idx, env_state["num_boards"] - 1)

    # Get board state and size
    board = env_state["board"][board_idx]
    board_size = env_state["board_size"]
    current_player = env_state["current_player"][board_idx]
    is_done = env_state["dones"][board_idx]
    winner = env_state["winners"][board_idx]

    # Choose rendering characters based on mode
    if mode == "unicode":
        h_line = "─"
        v_line = "│"
        cross = "┼"
        black = "●"
        white = "○"
        empty = " "
        corners = ["┌", "┐", "└", "┘"]
        t_joints = ["┬", "┤", "┴", "├"]
    else:  # ASCII mode
        h_line = "-"
        v_line = "|"
        cross = "+"
        black = "X"
        white = "O"
        empty = " "
        corners = ["+", "+", "+", "+"]
        t_joints = ["+", "+", "+", "+"]

    # Create board representation
    lines = []

    # Add top border
    top_border = (
        corners[0]
        + (h_line * 2 + t_joints[0]) * (board_size - 1)
        + h_line * 2
        + corners[1]
    )
    lines.append(top_border)

    # Add board rows
    for i in range(board_size):
        # Board row with pieces
        row = v_line
        for j in range(board_size):
            cell = board[i, j]
            if cell == 1:
                row += f" {black} "
            elif cell == -1:
                row += f" {white} "
            else:
                row += f" {empty} "
        lines.append(row + v_line)

        # Add horizontal separator if not the last row
        if i < board_size - 1:
            separator = (
                t_joints[3]
                + (h_line * 2 + cross + h_line) * (board_size - 1)
                + h_line * 2
                + t_joints[1]
            )
            lines.append(separator)

    # Add bottom border
    bottom_border = (
        corners[2]
        + (h_line * 2 + t_joints[2]) * (board_size - 1)
        + h_line * 2
        + corners[3]
    )
    lines.append(bottom_border)

    # Add status information
    if is_done:
        if winner == 1:
            status = "Game over - Black wins!"
        elif winner == -1:
            status = "Game over - White wins!"
        else:
            status = "Game over - Draw!"
    else:
        next_player = "Black" if current_player == 1 else "White"
        status = f"Current player: {next_player}"

    lines.append(status)

    return "\n".join(lines)


if __name__ == "__main__":
    # Configuration
    board_sizes = [9, 15]
    num_envs_options = [64, 256, 1024]

    print("Benchmarking run_random_episode function...")
    print("-" * 60)
    print(
        f"{'Board Size':<10} | {'Num Envs':<10} | {'Time (s)':<10} | {'Games/sec':<10} | {'Steps/Game':<10}"
    )
    print("-" * 60)

    for board_size in board_sizes:
        for num_envs in num_envs_options:
            # Warmup run to trigger compilation
            warmup_state, warmup_rewards = run_random_episode(
                board_size, num_envs, seed=0
            )
            # Force completion of all operations
            warmup_rewards.block_until_ready()

            # Timing run
            start_time = time.time()
            final_state, final_rewards = run_random_episode(
                board_size, num_envs, seed=42
            )
            # Force completion before stopping timer
            final_rewards.block_until_ready()
            end_time = time.time()

            # Calculate statistics
            elapsed = end_time - start_time
            win_count = jnp.sum(final_state["winners"] != 0).item()
            draw_count = num_envs - win_count
            total_steps = (
                jnp.mean(jnp.sum(final_state["board"] != 0, axis=(1, 2))).item() / 2
            )
            games_per_second = num_envs / elapsed

            print(
                f"{board_size:<10} | {num_envs:<10} | {elapsed:<10.3f} | {games_per_second:<10.1f} | {total_steps:<10.1f}"
            )

    print("-" * 60)
    print("Benchmark complete!")
