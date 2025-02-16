import jax
import jax.numpy as jnp
from jax import lax
from .renderer import GomokuRenderer

WIN_LENGTH = 5
CELL_SIZE = 40  # Constant cell size

class Gomoku:
    def __init__(self, board_size=15, device=None, mode="train"):
        self.board_size = board_size
        self.current_player = 1  # 1 for black, -1 for white
        self.board = jnp.zeros((board_size, board_size), dtype=jnp.float32)
        self.done = False
        self.seed = None
        self.kernels = self._create_kernels()
        self.device = device
        self.mode = mode
        self.cell_size = CELL_SIZE


    def reset(self, seed=None):
        """
        Resets the game environment to its initial state.
        """
        self.seed = seed
        self.board = jnp.zeros((self.board_size, self.board_size), dtype=jnp.float32)
        self.current_player = 1
        self.done = False

        if self.mode == "human":
            self.renderer = GomokuRenderer(self.board_size, self.cell_size)
            self._update_human_display()

        return self.board

    def step(self, action):
        """
        Executes one step in the Gomoku game in a JAX-friendly manner.
        Uses jax.lax.cond to avoid Python boolean conversion of traced booleans.
        Assumes that the provided action is legal.
        """
        row, col = action

        # Update the board immutably.
        new_board = self.board.at[row, col].set(self.current_player)
        self.board = new_board

        # Compute win and draw conditions as JAX booleans.
        win = self._check_win()
        draw = jnp.all(self.board != 0)

        new_done, new_reward, new_current_player = lax.cond(
            win,
            lambda _: (
                True,
                1.0,
                self.current_player,
            ),
            lambda _: lax.cond(
                draw,
                lambda _: (
                    True,
                    0.0,
                    self.current_player*-1,
                ),
                lambda _: (
                    False,
                    0.0,
                    self.current_player * -1,
                ),
                operand=None
            ),
            operand=None
        )

        self.done = new_done
        self.current_player = new_current_player

        if self.mode == "human":
            self._update_human_display()
            self.renderer.pause()
            if self.done:
                self.renderer.close()

        return self.board, new_reward, self.done

    def get_action_mask(self):
        return self.board == 0

    def _check_win(self):
        """
        Checks if the current board state has a winning condition.
        Uses convolution with predefined kernels.
        Returns a JAX boolean (without using .item()).
        """
        player_boards = (self.board * self.current_player)[jnp.newaxis, :, :, jnp.newaxis]
        padding = ((WIN_LENGTH - 1, WIN_LENGTH - 1), (WIN_LENGTH - 1, WIN_LENGTH - 1))

        conv_output = lax.conv_general_dilated(
            player_boards,
            self.kernels,
            window_strides=(1, 1),
            padding=padding,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        )
        win_condition = conv_output == WIN_LENGTH
        win = jnp.any(win_condition)
        return win  # Do not call .item(), just return the traced value

    def _create_kernels(self):
        ones = jnp.ones((1, WIN_LENGTH), dtype=jnp.float32)
        zeros = jnp.zeros((WIN_LENGTH - 1, WIN_LENGTH), dtype=jnp.float32)

        horizontal = jnp.expand_dims(jnp.vstack([ones, zeros]), axis=(0, 1))
        vertical = jnp.expand_dims(jnp.vstack([ones, zeros]).T, axis=(0, 1))
        diagonal = jnp.expand_dims(jnp.eye(WIN_LENGTH, dtype=jnp.float32), axis=(0, 1))
        anti_diagonal = jnp.expand_dims(jnp.fliplr(jnp.eye(WIN_LENGTH, dtype=jnp.float32)), axis=(0, 1))

        kernels = jnp.concatenate([horizontal, vertical, diagonal, anti_diagonal], axis=0).transpose(2, 3, 1, 0)
        return kernels

    def _update_human_display(self):
        """
        Updates the display (rendering) and is only called in human mode.
        This method should not be used when running jitted training loops.
        """
        if self.mode == "human":
            board_list = self.board.tolist()  # Safe since human mode is non-jitted.
            self.renderer.render_board(board_list)
            self.renderer.process_events()

    def to(self, device):
        self.device = device
        self.board = jax.device_put(self.board, device)
        self.kernels = jax.device_put(self.kernels, device)
        return self
