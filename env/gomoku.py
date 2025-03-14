import jax
import jax.numpy as jnp
from jax import lax

from .renderer import GomokuRenderer

WIN_LENGTH = 5
CELL_SIZE = 40  # Constant cell size


class Gomoku:
    def __init__(self, board_size=15, device=None, num_boards=1, mode="train"):
        self.board_size = board_size
        self.num_boards = num_boards
        self.current_player = jnp.ones(
            (num_boards), dtype=jnp.int32
        )  # 1 for black, -1 for white
        self.board = jnp.zeros((num_boards, board_size, board_size), dtype=jnp.float32)
        self.dones = jnp.zeros((num_boards), dtype=jnp.bool_)
        self.winners = jnp.zeros((num_boards), dtype=jnp.int32)
        self.seed = None
        self.kernels = self._create_kernels()
        self.device = device
        self.mode = mode

    def reset(self, seed=None):
        """
        Resets the game environment to its initial state.
        """
        self.seed = seed
        self.board = jnp.zeros(
            (self.num_boards, self.board_size, self.board_size), dtype=jnp.float32
        )
        self.current_player = jnp.ones((self.num_boards), dtype=jnp.int32)
        self.dones = jnp.zeros((self.num_boards), dtype=jnp.bool_)
        self.winners = jnp.zeros((self.num_boards), dtype=jnp.int32)

        if self.mode == "human":
            self.renderer = GomokuRenderer(self.board_size, CELL_SIZE)
            self._update_human_display()

        return self.board, self.dones

    def step(self, actions):
        """
        Executes one step in the Gomoku game in JIT-compatible manner.
        Args: actions: jnp.ndarray with shape (num_boards,2)
        Returns:
            board: jnp.ndarray with shape (num_boards, board_size, board_size)
            rewards: jnp.ndarray with shape (num_boards,)
            dones: jnp.ndarray with shape (num_boards,)
        """
        rows, cols = actions.T
        idx = jnp.arange(self.num_boards)
        action_mask = jnp.logical_not(self.dones) & (self.board[idx, rows, cols] == 0)
        active_idx = jnp.where(action_mask)[0]
        self.board = self.board.at[active_idx, rows[active_idx], cols[active_idx]].set(
            self.current_player[active_idx]
        )

        # Compute win and draw conditions as JAX booleans.
        wins = self._check_win()
        self.winners = jnp.where(wins, self.current_player, self.winners)

        # if not (won on this iteration or done previously)-> draw
        draws = jnp.all(self.board != 0, axis=(1, 2)) & ~(wins | self.dones)

        rewards = jnp.where(
            wins, self.current_player, 0.0
        )  # reward only for current wins
        self.current_player = -self.current_player
        self.dones = self.dones | wins | draws

        if self.mode == "human":
            self._update_human_display()
            self.renderer.pause()
            if self.dones:
                self.renderer.close()

        # Properly reshape current_player for broadcasting
        current_player_reshaped = self.current_player[:, jnp.newaxis, jnp.newaxis]
        return self.board * current_player_reshaped, rewards, self.dones

    def get_action_mask(self):
        return self.board == 0

    def _check_win(self):
        """
        Checks if the current board state has a winning condition.
        Uses convolution with predefined kernels.
        Returns a JAX boolean (without using .item()).
        """
        # Add necessary dimensions to current_player for broadcasting
        current_player_reshaped = self.current_player[:, jnp.newaxis, jnp.newaxis]
        player_boards = (self.board * current_player_reshaped)[:, :, :, jnp.newaxis]
        padding = ((WIN_LENGTH - 1, WIN_LENGTH - 1), (WIN_LENGTH - 1, WIN_LENGTH - 1))

        conv_output = lax.conv_general_dilated(
            player_boards,
            self.kernels,
            window_strides=(1, 1),
            padding=padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        win_condition = conv_output == WIN_LENGTH
        win = jnp.any(win_condition, axis=(-3, -2, -1)) & ~self.dones
        return win  # Do not call .item(), just return the traced value

    def _create_kernels(self):
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

    def _update_human_display(self):
        """
        Updates the display (rendering) and is only called in human mode.
        This method should not be used when running jitted training loops.
        """
        if self.mode == "human":
            board_list = self.board.tolist()[0]
            self.renderer.render_board(board_list)
            self.renderer.process_events()

    def to(self, device):
        self.device = device
        self.board = jax.device_put(self.board, device)
        self.kernels = jax.device_put(self.kernels, device)
        return self
