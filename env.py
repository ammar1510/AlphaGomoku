import jax.numpy as jnp
from jax import lax,jit
WIN_LENGTH = 5

class GomokuEnv:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.current_player = 1  # 1 for black, -1 for white
        self.board = jnp.zeros((board_size, board_size), dtype=jnp.float32)
        self.done = False
        self.seed = None
        self.kernels = self._create_kernels()

    def reset(self, seed=None):
        # The seed parameter is ignored in this minimal implementation.
        self.seed = seed
        self.board = jnp.zeros((self.board_size, self.board_size), dtype=jnp.float32)
        self.current_player = 1
        self.done = False
        return self._get_observation()

    def step(self, action):
        """
        Executes one step in the Gomoku game.
        
        Args:
            action (tuple): (row, col) indicating where to place the stone.
            
        Returns:
            observation (jnp.ndarray): The board state with shape (board_size, board_size, 1).
            reward (float): The reward for the move.
            done (bool): Whether the game has ended.
            info (dict): Additional information (e.g., move legality).
        """
        row, col = action
        
        # Check if the move is legal. Since self.board is a DeviceArray, we convert the entry to int.
        if int(self.board[row, col]) != 0:
            raise ValueError("Illegal move, position already occupied")

        # Place the stone using immutable update.
        self.board = self.board.at[row, col].set(self.current_player)
        
        # Check for win.
        if self._check_win():
            self.done = True
            reward = 1.0 if self.current_player == 1 else -1.0
            return self._get_observation(), reward, True, {"win": self.current_player}
        
        # Check for a draw: if all cells are nonzero.
        if bool(jnp.all(self.board != 0).item()):
            self.done = True
            return self._get_observation(), 0.0, True, {"draw": True}
        
        # Switch players.
        self.current_player *= -1
        return self._get_observation(), 0.0, False, {}

    def _get_observation(self):
        """
        Returns the current board state as a float32 array with an additional channel dimension.
        """
        return self.board

    def _check_win(self):
        """
        Checks if placing a stone at (row, col) leads to a win for the given player.
        A win is defined as having 5 or more consecutive stones in any direction.
        """
        # Prepare the board for convolution
        # Shape: (num_envs, board_size, board_size, 1)
        player_boards = (self.board * self.current_player)[jnp.newaxis,:,:,jnp.newaxis]

        # Perform convolution
        conv_output = lax.conv_general_dilated(
            player_boards,
            self.kernels,
            window_strides=(1, 1),
            padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=1
        )  # Shape: (active_boards_shape, board_size, board_size,4)

        win_condition = conv_output == WIN_LENGTH
        winners = jnp.any(win_condition)
        return winners.item()

    def _create_kernels(self):
        """
        Creates the kernels to check win conditions.
        """
        ones = jnp.ones((1, WIN_LENGTH), dtype=jnp.float32)  # shape: (1, win_length)
        zeros = jnp.zeros((WIN_LENGTH-1, WIN_LENGTH), dtype=jnp.float32)  # shape: (win_length-1, win_length)

        horizontal = jnp.expand_dims(jnp.vstack([ones, zeros]), axis=(0,1))  # shape: (1, 1, win_length, win_length)
        vertical = jnp.expand_dims(jnp.vstack([ones, zeros]).T, axis=(0,1))  # shape: (1, 1, win_length, win_length)
        diagonal = jnp.expand_dims(jnp.eye(WIN_LENGTH, dtype=jnp.float32), axis=(0, 1))  # shape: (1, 1, win_length, win_length)
        anti_diagonal = jnp.expand_dims(jnp.fliplr(jnp.eye(WIN_LENGTH, dtype=jnp.float32)), axis=(0, 1))  # shape: (1, 1, win_length, win_length)

        # Stack all kernels
        kernels = jnp.concatenate([horizontal, vertical, diagonal, anti_diagonal], axis=0).transpose(2,3,1,0)
        return kernels  # Shape: (win_length, win_length, 1, 4)


    def render(self):
        """
        Renders the board state to the terminal.
        """
        symbols = {1: 'X', -1: 'O', 0: '.'}
        print("-" * (self.board_size * 2))
        # Convert the board to a Python list for printing.
        board_list = self.board.tolist()
        for row in board_list:
            print(" ".join([symbols[val] for val in row]))
        print("-" * (self.board_size * 2))