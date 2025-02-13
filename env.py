import jax
import jax.numpy as jnp
from jax import lax

class GomokuEnv:
    def __init__(self, board_size=15, num_envs=1, device=jax.devices('cpu')[0]):
        """
        Initializes the Gomoku environment.

        Args:
            board_size (int): Size of the game board (default is 15).
            num_envs (int): Number of parallel environments (default is 1).
            device: JAX device to run the computations on.
        """
        self.board_size = board_size   # The size of each game board (i.e., board_size x board_size)
        self.num_envs = num_envs       # Number of parallel game instances (environments)
        self.device = device           # The JAX device on which computations will be performed
        self.win_length = 5            # Number of consecutive markers needed to win the game

        # The board for all environments is represented as a 3-dimensional array.
        # Each individual board is initialized as a 2D grid filled with zeros (indicating empty positions).
        # Shape: (num_envs, board_size, board_size)
        self.board = jnp.zeros(
            (self.num_envs, self.board_size, self.board_size),
            dtype=jnp.int8,
            device=device
        )

        # active_boards: A 1D array holding the indices of the active (ongoing) game boards.
        # Initially, all boards are active.
        # Shape: (num_envs,)
        self.active_boards = jnp.arange(num_envs, dtype=jnp.int32, device=device)

        # game_over: A boolean array that indicates whether each environment's game has ended
        # (due to a win or draw). Initially, all values are False because no game is over.
        # Shape: (num_envs,)
        self.game_over = jnp.zeros((num_envs,), dtype=jnp.bool_, device=device)

        # current_player: A 1D array representing the current player's turn for each environment.
        # A value of 1 indicates that it is Black's turn, and -1 indicates White's turn.
        # Black always starts first (thus initialized to 1 for all environments).
        # Shape: (num_envs,)
        self.current_player = jnp.ones((num_envs,), dtype=jnp.int8, device=device)

        # kernels: Precomputed convolution kernels that are used to detect winning moves.
        # These kernels cover horizontal, vertical, diagonal, and anti-diagonal directions,
        # each with a size that matches the number of markers required to win (win_length).
        self.kernels = self._create_kernels()

    def to(self,device=jax.devices('cpu')[0]):
        """
        Moves the environment to a different device.

        Args:
            device: JAX device to move the environment to.

        Returns:
            The environment object itself.
        """
        self.device = device
        self.board = jax.device_put(self.board, device)
        self.current_player = jax.device_put(self.current_player, device)
        self.active_boards = jax.device_put(self.active_boards, device)
        self.kernels = jax.device_put(self.kernels, device)
        self.game_over = jax.device_put(self.game_over, device)
        return self

    def _create_kernels(self):
        """
        Creates kernels for horizontal, vertical, diagonal, and anti-diagonal win detection.

        Returns:
            jnp.ndarray: A tensor of shape (4, 1, 5, 5) containing the kernels.
        """
        ones = jnp.ones((1, self.win_length), dtype=jnp.int8)  # shape: (1, win_length)
        zeros = jnp.zeros((self.win_length-1, self.win_length), dtype=jnp.int8)  # shape: (win_length-1, win_length)

        horizontal = jnp.expand_dims(jnp.vstack([ones, zeros]), axis=(0,1))  # shape: (1, 1, win_length, win_length)
        vertical = jnp.expand_dims(jnp.vstack([ones, zeros]).T, axis=(0,1))  # shape: (1, 1, win_length, win_length)
        diagonal = jnp.expand_dims(jnp.eye(self.win_length, dtype=jnp.int8), axis=(0, 1))  # shape: (1, 1, win_length, win_length)
        anti_diagonal = jnp.expand_dims(jnp.fliplr(jnp.eye(self.win_length, dtype=jnp.int8)), axis=(0, 1))  # shape: (1, 1, win_length, win_length)

        # Stack all kernels
        kernels = jnp.concatenate([horizontal, vertical, diagonal, anti_diagonal], axis=0).transpose(2,3,1,0)
        return kernels  # Shape: (win_length, win_length, 1, 4)

    def reset(self, env_indices=None):
        """
        Resets the environment(s).

        Args:
            env_indices (array-like, optional): Specific environments to reset. 
                                                If None, all environments are reset.

        Returns:
            Tuple of (board, current_player, game_over) after reset.
        """
        if env_indices is None:
            #reset all boards
            self.board = jnp.zeros(
                (self.num_envs, self.board_size, self.board_size),
                dtype=jnp.int8,
                device=self.device
            )
            self.current_player = jnp.ones((self.num_envs,), dtype=jnp.int8, device=self.device)
            self.game_over = jnp.zeros((self.num_envs,), dtype=jnp.bool_, device=self.device)
            self.active_boards = jnp.arange(self.num_envs, dtype=jnp.int32, device=self.device)
        else:
            self.board = self.board.at[env_indices,:,:].set(0)
            self.current_player = self.current_player.at[env_indices].set(1)
            self.active_boards = jnp.union1d(self.active_boards, jnp.array(env_indices))
            self.game_over = self.game_over.at[env_indices].set(False)
        return self.get_state()

    def step(self, actions: jnp.ndarray):
        """
        Performs a step in multiple environments by applying the given actions.

        Args:
            actions (jnp.ndarray): Array of shape (num_envs, 2) where each row is (row, col) indices 
                                   for the corresponding environment's action.

        Returns:
            Tuple of (board, current_player, game_over) after the actions.
        """
        actions = jnp.array(actions)
        assert actions.shape[0] == self.num_envs and actions.shape[1] == 2, "Number of actions must match the number of environments"

        rows, cols = actions[:, 0], actions[:, 1]

        # Ensure the actions are within the board
        valid_bounds = (0 <= rows) & (rows < self.board_size) & (0 <= cols) & (cols < self.board_size)

        # Check if the positions are already taken
        positions = self.board[self.active_boards, rows, cols]
        valid_positions = positions == 0

        # Combine validity checks
        valid_actions = valid_bounds & valid_positions

        if not jnp.all(valid_actions):
            invalid_envs = jnp.where(~valid_actions)
            raise ValueError(f"Invalid actions in environments: {invalid_envs}")

        self.board = self.board.at[self.active_boards, rows, cols].set(self.current_player)

        self.check_game_over()

        self.current_player = self.current_player.at[self.active_boards].set(-self.current_player)

        return self.get_state(),

    def check_game_over(self):
        """
        Checks if the latest actions resulted in a win or draw.

        Returns:
            Tuple of (winners, dones) where:
                - winners is a boolean array indicating if the current move caused a win.
                - dones is a boolean array indicating if the game is a draw.
        """
        # Prepare the board for convolution
        # Shape: (num_envs, board_size, board_size, 1)
        player_boards = (self.board * self.current_player[:,jnp.newaxis,jnp.newaxis]).astype(jnp.int8)[self.active_boards,:,:,jnp.newaxis]

        # Perform convolution
        conv_output = lax.conv_general_dilated(
            player_boards,
            self.kernels,
            window_strides=(1, 1),
            padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=1
        )  # Shape: (active_boards_shape, board_size, board_size,4)

        win_condition = conv_output == self.win_length
        winners = jnp.any(win_condition, axis=(1, 2, 3))  

        #check for draw
        empty_spaces = jnp.any(self.board == 0, axis=(1, 2))
        dones = ~empty_spaces
        self.game_over = self.game_over.at[self.active_boards].set(winners)
        self.game_over = self.game_over|dones

        self.active_boards = jnp.nonzero(~self.game_over, size=self.num_envs)[0]


    def get_action_mask(self):
        """
        Returns a mask of valid actions for each environment.

        Returns:
            jnp.ndarray: A boolean array of shape (num_envs, board_size, board_size)
                         where each element is True if the corresponding position is empty.
        """
        return self.board == 0

    def get_state(self):
        """
        Retrieves the current state of the environment(s).

        Returns:
            Tuple of (board, current_player, game_over).
        """
        return self.board, self.current_player, self.game_over

