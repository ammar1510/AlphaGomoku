import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Dict, Any, NamedTuple, Optional
from functools import partial

from .base import JaxEnvBase, EnvState

# --- Constants and Kernel Generation ---
WIN_LENGTH = 5


# --- State Definition ---
class GomokuState(NamedTuple):
    """Holds the dynamic state of the batched Gomoku environment for a single step."""

    boards: jnp.ndarray  # (B, board_size, board_size) float32 tensor
    current_players: jnp.ndarray  # (B,) int32 tensor (1 or -1)
    dones: jnp.ndarray  # (B,) bool tensor
    winners: jnp.ndarray  # (B,) int32 tensor (1, -1, or 0 for draw/ongoing)
    rng: jax.random.PRNGKey  # JAX PRNGKey


# --- Environment Logic ---
class GomokuJaxEnv(JaxEnvBase):
    """
    Functional JAX-based Gomoku environment logic container.
    Observations are player agnostic, representing the board state 1 for black and -1 for white.

    Static Attributes:
        B: Batch size.
        board_size: Size of the board side.
        win_length: Length required to win (usually 5).
    """

    def __init__(self, B: int, board_size: int = 9, win_length: int = WIN_LENGTH):
        """
        Initializes the environment configuration holder. Does not create state.

        Args:
            B: Batch size.
            board_size: The size of the Gomoku board (e.g., 9 for 9x9).
            win_length: The number of consecutive pieces needed to win.
        """
        super().__init__(B=B)
        self.B = B
        self.board_size = board_size
        self.win_length = win_length

        self.win_kernels = GomokuJaxEnv._create_win_kernels(self.win_length)
        self.win_kernels_dn = ("NHWC", "HWIO", "NHWC")

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _create_win_kernels(win_len: int = WIN_LENGTH):
        """Create convolution kernels for win detection."""
        # Horizontal kernel
        kernel_h = jnp.zeros((win_len, win_len), dtype=jnp.float32)
        kernel_h = kernel_h.at[win_len // 2, :].set(1)
        # Vertical kernel
        kernel_v = jnp.zeros((win_len, win_len), dtype=jnp.float32)
        kernel_v = kernel_v.at[:, win_len // 2].set(1)
        # Diagonal kernel
        kernel_d1 = jnp.eye(win_len, dtype=jnp.float32)
        # Anti-diagonal kernel
        kernel_d2 = jnp.fliplr(kernel_d1)

        kernels = jnp.stack(
            [kernel_h, kernel_v, kernel_d1, kernel_d2], axis=-1
        )  # Shape (win_len, win_len, 4)
        # Reshape for lax.conv_general_dilated: (H, W, I, O) -> (H, W, 1, 4)
        kernels = jnp.expand_dims(kernels, axis=2)
        return kernels

    @staticmethod
    @partial(jax.jit, static_argnames=("B", "board_size"))
    def init_state(rng: jax.random.PRNGKey, B: int, board_size: int) -> GomokuState:
        """
        Creates the initial GomokuState.

        Args:
            rng: JAX PRNG key for initialization.
            B: Batch size.
            board_size: The size of the Gomoku board.

        Returns:
            The initial GomokuState.
        """
        return GomokuState(
            boards=jnp.zeros((B, board_size, board_size), dtype=jnp.float32),
            current_players=jnp.ones((B,), dtype=jnp.int32),
            dones=jnp.zeros((B,), dtype=jnp.bool_),
            winners=jnp.zeros((B,), dtype=jnp.int32),
            rng=rng,
        )

    @partial(jax.jit, static_argnames=("self",))
    def _check_win(self, board: jnp.ndarray, current_players: jnp.ndarray) -> jnp.ndarray:
        """
        Check for wins using convolution. Uses pre-computed kernels from self.

        Args:
            self: The GomokuJaxEnv instance.
            board: Game boards (B, H, W).
            current_players: Current player values (B,).

        Returns:
            wins: Boolean array (B,) indicating wins for the current player.
        """
        current_players_reshaped = current_players[:, None, None]  # (B, 1, 1)
        player_boards = (board == current_players_reshaped).astype(
            jnp.float32
        )  # (B, H, W)

        player_boards_nhwc = player_boards[:, :, :, None]  # (B, H, W, 1)

        conv_output = lax.conv_general_dilated(
            player_boards_nhwc,
            self.win_kernels, # <-- Use pre-computed kernels from self
            window_strides=(1, 1),
            padding="SAME_LOWER",
            dimension_numbers=self.win_kernels_dn, # <-- Use pre-computed DN from self
        )

        win_condition = conv_output == self.win_length # <-- Use self.win_length
        wins = jnp.any(win_condition, axis=(1, 2, 3))  # Shape (B,)
        return wins

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self, state: GomokuState, actions: jnp.ndarray
    ) -> Tuple[GomokuState, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """
        Takes a step in each environment based on the current state and actions. Pure function.

        Args:
            self: The GomokuJaxEnv instance (provides config like B, board_size, win_length).
            state: The current GomokuState.
            actions: JAX array of actions (row, col) for each env. Shape (B, 2).

        Returns:
            A tuple (new_state, observations, rewards, dones, info).
        """
        B = self.B
        rows, cols = actions[:, 0], actions[:, 1]

        current_boards = state.boards
        current_players = state.current_players
        current_dones = state.dones
        current_winners = state.winners
        current_rng = state.rng

        # make a move
        valid_move = (current_boards[jnp.arange(B), rows, cols] == 0) & (~current_dones)
        current_player_placing = current_players * valid_move.astype(jnp.int32)
        new_boards = current_boards.at[jnp.arange(B), rows, cols].set(
            current_boards[jnp.arange(B), rows, cols] + current_player_placing
        )

        # check for win
        win_patterns = self._check_win(new_boards, current_players)
        current_wins = win_patterns & (~current_dones)

        new_winners = jnp.where(current_wins, current_players, current_winners)

        # check for draw
        board_full = jnp.all(new_boards != 0, axis=(1, 2))
        current_draws = board_full & (~current_wins) & (~current_dones)

        new_dones = current_dones | current_wins | current_draws

        # rewards are always 1 for current player
        rewards = jnp.where(current_wins, 1.0, 0.0)

        # switch player
        switch_player = valid_move & (~new_dones)
        next_players = jnp.where(switch_player, -current_players, current_players)

        # observations are player agnostic, representing the board state 1 for black and -1 for white.
        observations = new_boards

        new_state = GomokuState(
            boards=new_boards,
            current_players=next_players,
            dones=new_dones,
            winners=new_winners,
            rng=current_rng,
        )

        info = {}

        return new_state, observations, rewards, new_dones, info

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, rng: jax.random.PRNGKey
    ) -> Tuple[GomokuState, jnp.ndarray, Dict[str, Any]]:
        """
        Resets environments to initial states using the provided RNG key. Pure function.

        Args:
            self: The GomokuJaxEnv instance (provides config like B, board_size).
            rng: JAX PRNGKey to use for initializing the new state.

        Returns:
            A tuple (new_state, initial_observations, info).
        """
        new_state = GomokuJaxEnv.init_state(rng, self.B, self.board_size)

        # observations are player agnostic, representing the board state 1 for black and -1 for white.
        initial_observations = new_state.boards

        info = {}
        return new_state, initial_observations, info

    def initialize_trajectory_buffers(self, max_steps: int) -> Tuple[jnp.ndarray, ...]:
        """
        Creates and returns pre-allocated JAX arrays for storing trajectory data.

        Args:
            max_steps: The maximum length of the trajectories to buffer.

        Returns:
            A tuple containing JAX arrays for observations, actions, rewards,
            dones, and log_probs.
        """
        obs_shape = self.observation_shape
        act_shape = self.action_shape  # Action is (row, col), shape (2,)

        observations = jnp.zeros((max_steps, self.B) + obs_shape, dtype=jnp.float32)
        actions = jnp.zeros((max_steps, self.B) + act_shape, dtype=jnp.int32)
        rewards = jnp.zeros((max_steps, self.B), dtype=jnp.float32)
        dones = jnp.zeros((max_steps, self.B), dtype=jnp.bool_)
        log_probs = jnp.zeros((max_steps, self.B), dtype=jnp.float32)
        current_players_buffer = jnp.zeros((max_steps, self.B), dtype=jnp.int32)
        # Could potentially add value estimates here too if needed by the algorithm
        # values = jnp.zeros((max_steps, self.B), dtype=jnp.float32)

        return observations, actions, rewards, dones, log_probs, current_players_buffer

    # Properties match base class (still using @property for convenience)
    @property
    def observation_shape(self) -> tuple:
        """Returns the shape of a single observation (board state)."""
        return (self.board_size, self.board_size)

    @property
    def action_shape(self) -> tuple:
        """Returns the shape of a single action (row, col)."""
        return (2,)

    @partial(jax.jit, static_argnames=("self",))
    def get_action_mask(self, state: GomokuState) -> jnp.ndarray:
        """
        Returns a boolean mask of valid actions based on the current state. Pure function.

        Args:
            self: Unused, but required by the abstract method signature.
            state: The current GomokuState.

        Returns:
            Boolean mask, shape (B, board_size, board_size). True indicates a valid move.
        """
        action_mask = state.boards == 0
        action_mask = action_mask & (~state.dones[:, None, None])
        return action_mask
