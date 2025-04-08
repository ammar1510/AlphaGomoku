import time
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

# Constants
WIN_LENGTH = 5


@jax.jit
def _create_win_kernels():
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


WIN_KERNELS = _create_win_kernels()


@partial(jax.jit, static_argnums=(0, 1))
def init_env(board_size, B, rng):
    """
    Initialize the Gomoku environment state.

    Args:
        rng: JAX random key
        board_size: Size of the Gomoku board
        B: Number of parallel boards / batch size

    Returns:
        env_state: Dictionary containing the environment state
    """

    return {
        "boards": jnp.zeros((B, board_size, board_size), dtype=jnp.float32),
        "current_player": jnp.ones((B,), dtype=jnp.int32),
        "dones": jnp.zeros((B,), dtype=jnp.bool_),
        "winners": jnp.zeros((B,), dtype=jnp.int32),
        "board_size": board_size,
        "B": B,
        "rng": rng,
    }


@jax.jit
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
    B, board_size, _ = env_state["boards"].shape

    rng = new_rng if new_rng is not None else jax.random.split(env_state["rng"])[0]

    new_env_state = {
        "boards": jnp.zeros((B, board_size, board_size), dtype=jnp.float32),
        "current_player": jnp.ones((B,), dtype=jnp.int32),
        "dones": jnp.zeros((B,), dtype=jnp.bool_),
        "winners": jnp.zeros((B,), dtype=jnp.int32),
        "board_size": board_size,
        "B": B,
        "rng": rng,
    }

    observations = new_env_state["boards"]

    return new_env_state, observations


@jax.jit
def check_win(board, current_player):
    """
    Check for wins using convolution.

    Args:
        board: Game boards with shape (B, board_size, board_size)
        current_player: Current player values with shape (B,)

    Returns:
        wins: Boolean array indicating which boards have wins
    """
    current_player_reshaped = current_player[:, jnp.newaxis, jnp.newaxis]
    player_boards = (board == current_player_reshaped).astype(jnp.float32)

    player_boards = player_boards[:, :, :, jnp.newaxis]

    padding = ((WIN_LENGTH - 1, WIN_LENGTH - 1), (WIN_LENGTH - 1, WIN_LENGTH - 1))

    conv_output = lax.conv_general_dilated(
        player_boards,
        WIN_KERNELS,
        window_strides=(1, 1),
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )

    win_condition = conv_output == WIN_LENGTH

    wins = jnp.any(win_condition, axis=(1, 2, 3))

    return wins


@jax.jit
def get_action_mask(env_state):
    """
    Get mask of valid actions.

    Args:
        env_state: Current environment state

    Returns:
        action_mask: Boolean mask of valid actions with shape (B, board_size, board_size)
    """

    boards = env_state["boards"]
    dones = env_state["dones"]

    action_mask = boards == 0

    action_mask = action_mask & (~dones[:, jnp.newaxis, jnp.newaxis])

    return action_mask


@jax.jit
def step_env(env_state, actions):
    """
    Take a step in the environment.

    Args:
        env_state: Current environment state
        actions: Actions to take, shape (B, 2) where each action is [row, col]

    Returns:
        new_env_state: Updated environment state
        observations: New observations
        rewards: Rewards from this step
        dones: Done flags
    """

    new_env_state = dict(env_state)

    boards = env_state["boards"]
    current_player = env_state["current_player"]
    dones = env_state["dones"]
    winners = env_state["winners"]

    B, board_size, _ = boards.shape

    rows, cols = actions.T

    new_boards = boards.at[jnp.arange(B), rows, cols].set(current_player)

    new_env_state["boards"] = new_boards

    win_patterns = check_win(new_boards, current_player)
    wins = win_patterns & ~dones
    new_winners = jnp.where(wins, current_player, winners)
    new_env_state["winners"] = new_winners

    draws = jnp.all(new_boards != 0, axis=(1, 2)) & ~(win_patterns | dones)

    rewards = jnp.where(wins, current_player, 0.0)

    new_dones = dones | wins | draws
    new_env_state["dones"] = new_dones

    new_env_state["current_player"] = -current_player

    current_player_reshaped = new_env_state["current_player"][
        :, jnp.newaxis, jnp.newaxis
    ]
    observations = new_boards * current_player_reshaped

    return new_env_state, observations, rewards, new_dones


@jax.jit
def sample_action(env_state, rng):
    """
    Sample a random valid action.

    Args:
        env_state: Current environment state
        rng: JAX random key

    Returns:
        actions: Sampled actions with shape (B, 2)
        new_env_state: Updated environment state with new RNG
    """
    action_mask = get_action_mask(env_state)
    B, board_size, _ = env_state["boards"].shape

    flat_mask = action_mask.reshape(B, -1)
    total_positions = flat_mask.shape[1]

    rng, subkey = jax.random.split(rng)
    random_values = jax.random.uniform(subkey, shape=(B, total_positions))

    masked_random = jnp.where(flat_mask, random_values, -1.0)

    flat_actions = jnp.argmax(masked_random, axis=1)

    rows = flat_actions // board_size
    cols = flat_actions % board_size
    actions = jnp.stack([rows, cols], axis=1)

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


@jax.jit
def get_valid_actions(env_state):
    """
    Get all valid actions for the current state.

    Args:
        env_state: Current environment state

    Returns:
        valid_actions: Array of valid action coordinates for each board
    """
    action_mask = get_action_mask(env_state)
    board_size = env_state["board_size"]

    num_envs, height, width = action_mask.shape

    rows, cols = jnp.meshgrid(jnp.arange(width), jnp.arange(height))
    all_actions = jnp.stack([rows.flatten(), cols.flatten()], axis=1)

    def get_board_valid_actions(board_idx):
        board_mask = action_mask[board_idx].flatten()
        valid_indices = jnp.where(board_mask, size=height * width)[0]

        padded_indices = jnp.pad(
            valid_indices,
            (0, height * width - valid_indices.shape[0]),
            constant_values=-1,
        )
        return all_actions[padded_indices]

    valid_actions = jax.vmap(get_board_valid_actions)(jnp.arange(num_envs))

    return valid_actions


@partial(jax.jit, static_argnums=(0, 1))  # Mark board_size and B as static
def run_random_episode(board_size=15, B=256, seed=0):
    """
    Run a complete episode with random actions.

    Args:
        board_size: Size of the Gomoku board
        B: Number of parallel boards
        seed: Random seed

    Returns:
        final_state: Final environment state
        total_rewards: Total rewards for each board
    """
    # Initialize environment
    rng = jax.random.PRNGKey(seed)
    init_state = init_env(board_size, B, rng)
    env_state, obs = reset_env(init_state)

    # Initialize total rewards
    total_rewards = jnp.zeros((B,))

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
    board_idx = min(board_idx, env_state["B"] - 1)

    # Get board state and size
    boards = env_state["boards"]
    board = boards[board_idx]
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
            cell = boards[board_idx, i, j]
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
                jnp.mean(jnp.sum(final_state["boards"] != 0, axis=(1, 2))).item() / 2
            )
            games_per_second = num_envs / elapsed

            print(
                f"{board_size:<10} | {num_envs:<10} | {elapsed:<10.3f} | {games_per_second:<10.1f} | {total_steps:<10.1f}"
            )

    print("-" * 60)
    print("Benchmark complete!")
