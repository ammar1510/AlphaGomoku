import jax
import jax.numpy as jnp
from jax import lax
from .renderer import GomokuRenderer

WIN_LENGTH = 5
CELL_SIZE = 40  # Constant cell size

class Gomoku:
    def __init__(self, board_size=15, device=jax.devices('cpu')[0], mode="train"):
        self.board_size = board_size
        self.current_player = 1  # 1 for black, -1 for white
        self.board = jnp.zeros((board_size, board_size), dtype=jnp.float32, device=device)
        self.done = False
        self.seed = None
        self.kernels = jax.device_put(self._create_kernels(), device)
        self.device = device
        self.mode = mode
        self.cell_size = CELL_SIZE

    def reset(self, seed=None):
        # The seed parameter is ignored in this minimal implementation.
        self.seed = seed
        self.board = jnp.zeros((self.board_size, self.board_size), dtype=jnp.float32)
        self.current_player = 1
        self.done = False

        # If human mode, (re)initialize rendering in reset and update display.
        if self.mode == "human":
            self.renderer = GomokuRenderer(self.board_size, self.cell_size)
            self._update_human_display()

        return self.board, {}

    def step(self, action):
        """
        Executes one step in the Gomoku game.
        
        Args:
            action (tuple): (row, col) indicating where to place the stone.
            
        Returns:
            observation (jnp.ndarray): The board state.
            reward (float): The reward for the move.
            done (bool): Whether the game has ended.
            info (dict): Additional information (e.g., win/draw indicator).
        """
        row, col = action

        # Assume legality is enforced externally.
        self.board = self.board.at[row, col].set(self.current_player)
        
        reward = 0.0
        info = {}
        # Instead of using Python if/elif statements on traced booleans,
        # one option is to simply compute the conditions as JAX booleans.
        win = self._check_win()  # See below for modifications.
        draw = jnp.all(self.board != 0)

        # (Here you choose a policy for branching. For example, if you're only using
        # this function in train mode with a static env, you may accept the lower-level
        # JAX booleans and process the outcome outside of JIT.)
        if win:
            self.done = True
            reward = 1.0 if self.current_player == 1 else -1.0
            info = {"result": "win", "winner": self.current_player}
        elif draw:
            self.done = True
            info = {"result": "draw"}
        else:
            self.done = False
            self.current_player *= -1

        if self.mode == "human":
            self._update_human_display()
            self.renderer.pause()
            if self.done:
                self.renderer.close()

        return self.board, reward, self.done, info

    def get_action_mask(self):
        return self.board == 0

    def _check_win(self):
        """
        Checks if placing a stone leads to a win for the current player.
        A win is defined as 5 or more consecutive stones in any direction.
        """
        # Prepare the board for convolution.
        player_boards = (self.board * self.current_player)[jnp.newaxis, :, :, jnp.newaxis]

        # Perform convolution.
        conv_output = lax.conv_general_dilated(
            player_boards,
            self.kernels,
            window_strides=(1, 1),
            padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=1
        )
        win_condition = conv_output == WIN_LENGTH
        winners = jnp.any(win_condition)
        return winners

    def _create_kernels(self):
        """
        Creates the kernels used to check for win conditions.
        """
        ones = jnp.ones((1, WIN_LENGTH), dtype=jnp.float32)
        zeros = jnp.zeros((WIN_LENGTH - 1, WIN_LENGTH), dtype=jnp.float32)

        horizontal = jnp.expand_dims(jnp.vstack([ones, zeros]), axis=(0, 1))
        vertical = jnp.expand_dims(jnp.vstack([ones, zeros]).T, axis=(0, 1))
        diagonal = jnp.expand_dims(jnp.eye(WIN_LENGTH, dtype=jnp.float32), axis=(0, 1))
        anti_diagonal = jnp.expand_dims(jnp.fliplr(jnp.eye(WIN_LENGTH, dtype=jnp.float32)), axis=(0, 1))

        kernels = jnp.concatenate([horizontal, vertical, diagonal, anti_diagonal], axis=0).transpose(2, 3, 1, 0)
        return kernels

    def to(self, device):
        self.device = device
        self.board = jax.device_put(self.board, device)
        self.kernels = jax.device_put(self.kernels, device)
        return self

    def _update_human_display(self):
        """
        Updates the pygame display with the current board state.
        """
        board_list = self.board.tolist()
        self.renderer.render_board(board_list)
        self.renderer.process_events()
