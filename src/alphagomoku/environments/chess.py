import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Dict, Any, NamedTuple, Optional
from functools import partial

from .base import Env, EnvState
from alphagomoku.common.sharding import mesh_rules


# --- Constants ---
# Piece types
EMPTY = 0
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

# Players
WHITE = 1
BLACK = -1

# Action types (JAX JIT compatible - using integers instead of strings)
NORMAL = 0
CASTLING = 1
EN_PASSANT = 2
PROMOTION = 3
PAWN_DOUBLE = 4


# --- Helper Functions ---
def _create_move_tables() -> jnp.ndarray:
    """Create lookup table mapping move plane index to (delta_row, delta_col, promotion_piece)"""
    # Shape: (73, 3) - [delta_row, delta_col, promotion_piece]
    move_data = jnp.zeros((73, 3), dtype=jnp.int32)
    
    # Queen-like moves: 56 planes (8 directions * 7 distances)
    for i in range(56):
        # 8 directions: N, NE, E, SE, S, SW, W, NW
        direction_deltas = [
            (-1, 0),   # North
            (-1, 1),   # Northeast  
            (0, 1),    # East
            (1, 1),    # Southeast
            (1, 0),    # South
            (1, -1),   # Southwest
            (0, -1),   # West
            (-1, -1),  # Northwest
        ]
        
        direction_idx = i // 7  # Which direction (0-7)
        distance = (i % 7) + 1  # Distance 1-7
        
        base_delta_row, base_delta_col = direction_deltas[direction_idx]
        delta_row = base_delta_row * distance
        delta_col = base_delta_col * distance
        
        # Queen promotion for pawn moves to back rank
        promotion_piece = QUEEN
        
        move_data = move_data.at[i, :].set([delta_row, delta_col, promotion_piece])
    
    # Knight moves: 8 planes
    knight_moves = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1)
    ]
    for i, (delta_row, delta_col) in enumerate(knight_moves):
        plane_idx = 56 + i
        move_data = move_data.at[plane_idx, :].set([delta_row, delta_col, EMPTY])  # No promotion
    
    # Pawn underpromotions: 9 planes (3 move types * 3 pieces)
    pawn_move_types = [
        (-1, 0),   # Forward
        (-1, -1),  # Diagonal-left capture
        (-1, 1),   # Diagonal-right capture
    ]
    promotion_pieces = [KNIGHT, ROOK, BISHOP]
    
    for i in range(9):
        plane_idx = 64 + i
        pawn_move_type = i // 3  # Which move type (0-2)
        promotion_piece_idx = i % 3  # Which piece (0-2)
        
        delta_row, delta_col = pawn_move_types[pawn_move_type]
        promotion_piece = promotion_pieces[promotion_piece_idx]
        
        move_data = move_data.at[plane_idx, :].set([delta_row, delta_col, promotion_piece])
    
    return move_data


def move_piece(piece: int, init_pos: Tuple[int, int], move_plane_index: int, 
               move_tables: jnp.ndarray, en_passant_pawn: jnp.ndarray) -> Tuple[Tuple[int, int], int, int]:
    """
    Determine move details for a single piece.
    
    Args:
        piece: The piece being moved. Positive for White, negative for Black.
        init_pos: Initial position (row, col) of the piece
        move_plane_index: Move plane index (0-72)
        move_tables: Pre-computed lookup table (73, 3)
        en_passant_pawn: Position of pawn that can be captured en passant (2,) [row, col] or [-1, -1] if none
        
    Returns:
        Tuple of (delta_pos, action_type, promotion_piece):
            - delta_pos: (delta_row, delta_col) offset for the move
            - action_type: NORMAL, CASTLING, EN_PASSANT, PROMOTION, PAWN_DOUBLE (int constants)
            - promotion_piece: Piece type for promotion (EMPTY if no promotion)
    """
    # Lookup delta and promotion piece from pre-computed table
    delta_row = move_tables[move_plane_index, 0]
    delta_col = move_tables[move_plane_index, 1]
    base_promotion_piece = move_tables[move_plane_index, 2]
    
    # Handle Black pieces (flip deltas)
    delta_row = delta_row * jnp.sign(piece)
    delta_col = delta_col * jnp.sign(piece)
    
    # Calculate destination
    target_row = init_pos[0] + delta_row
    target_col = init_pos[1] + delta_col
    
    # Determine action type using JAX conditionals
    abs_piece = jnp.abs(piece)
    
    # King moving 2 horizontally = CASTLING
    is_castling = (abs_piece == KING) & (jnp.abs(delta_col) == 2) & (delta_row == 0)
    
    # Pawn moving 2 forward = PAWN_DOUBLE  
    is_pawn_double = (abs_piece == PAWN) & (jnp.abs(delta_row) == 2) & (delta_col == 0)
    
    # Pawn to 8th rank = PROMOTION
    is_promotion = (abs_piece == PAWN) & ((target_row == 0) | (target_row == 7))
    
    # Pawn diagonal move to capture en passant pawn = EN_PASSANT
    ep_pawn_valid = (en_passant_pawn[0] != -1) 
    is_pawn_diagonal = (abs_piece == PAWN) & (delta_col != 0)
    
    # En passant capture target is the square "behind" the en_passant_pawn
    # White captures by moving forward (+1), Black captures by moving backward (-1)
    ep_capture_row = en_passant_pawn[0] + jnp.sign(piece)
    ep_capture_col = en_passant_pawn[1]
    target_matches_ep = (target_row == ep_capture_row) & (target_col == ep_capture_col)
    is_en_passant = is_pawn_diagonal & ep_pawn_valid & target_matches_ep
    
    # Determine action type using priority-based switch
    # Priority order: CASTLING > PAWN_DOUBLE > PROMOTION > EN_PASSANT > NORMAL
    action_conditions = jnp.array([
        is_castling,
        is_pawn_double, 
        is_promotion,
        is_en_passant,
        True  # Default case (NORMAL)
    ])
    
    action_types = jnp.array([CASTLING, PAWN_DOUBLE, PROMOTION, EN_PASSANT, NORMAL])
    
    # Find first True condition (argmax returns first True index)
    condition_index = jnp.argmax(action_conditions)
    action_type = action_types[condition_index]
    
    # Final promotion piece (EMPTY if not promotion, otherwise from lookup)
    final_promotion_piece = lax.cond(is_promotion, 
                                   lambda: base_promotion_piece, 
                                   lambda: EMPTY)
    
    return (delta_row, delta_col), action_type, final_promotion_piece


@partial(jax.jit,static_argnums=(5,),donate_argnums=(0,1))
def move(board_state: jnp.ndarray, action: Tuple[int, int, int], 
         castling_rights: jnp.ndarray, en_passant_pawn: jnp.ndarray, 
         move_tables: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Apply a move to a single board state.
    
    Args:
        board_state: Single board state (8, 8)
        action: (from_row, from_col, move_plane_index)
        castling_rights: Castling rights (4,) for this board
        en_passant_pawn: Position of pawn that can be captured en passant (2,) for this board
        current_player: Current player (1 or -1)
        move_tables: Pre-computed lookup table (73, 3)
        
    Returns:
        Tuple of (new_board_state, new_castling_rights, new_en_passant_pawn):
            - new_board_state: Updated board after move
            - new_castling_rights: Updated castling rights
            - new_en_passant_pawn: Updated en passant pawn position
    """
    from_row, from_col, move_plane_index = action
    
    # Get piece at source position
    piece = board_state[from_row, from_col]
    
    # Get move details
    (delta_row, delta_col), action_type, promotion_piece = move_piece(
        piece, (from_row, from_col), move_plane_index, move_tables, en_passant_pawn
    )
    
    # Calculate destination
    to_row = from_row + delta_row
    to_col = from_col + delta_col
    
    # Apply move based on action type using JAX switch
    move_functions = [
        lambda: _apply_normal_move(board_state, from_row, from_col, to_row, to_col, piece),
        lambda: _apply_castling_move(board_state, from_row, from_col, to_row, to_col, piece),
        lambda: _apply_en_passant_move(board_state, from_row, from_col, to_row, to_col, piece),
        lambda: _apply_promotion_move(board_state, from_row, from_col, to_row, to_col, piece, promotion_piece),
        lambda: _apply_pawn_double_move(board_state, from_row, from_col, to_row, to_col, piece)
    ]
    
    new_board = lax.switch(action_type, move_functions)
    
    # Update castling rights and en passant pawn position
    new_castling_rights = _update_castling_rights(castling_rights, from_row, from_col, to_row, to_col, piece)
    new_en_passant_pawn = _update_en_passant_pawn(action_type, to_row, to_col)
    
    return new_board, new_castling_rights, new_en_passant_pawn


def _apply_normal_move(board_state: jnp.ndarray, from_row: int, from_col: int, 
                      to_row: int, to_col: int, piece: int) -> jnp.ndarray:
    """Apply a normal move: remove piece from source, place at destination."""
    new_board = board_state.at[from_row, from_col].set(EMPTY)
    new_board = new_board.at[to_row, to_col].set(piece)
    return new_board


def _apply_castling_move(board_state: jnp.ndarray, from_row: int, from_col: int, 
                        to_row: int, to_col: int, piece: int) -> jnp.ndarray:
    """Apply castling: move king + rook."""
    # Move king
    new_board = board_state.at[from_row, from_col].set(EMPTY)
    new_board = new_board.at[to_row, to_col].set(piece)
    
    # Move rook based on castling side
    is_kingside = to_col > from_col
    rook_from_col = lax.cond(is_kingside, lambda: 7, lambda: 0)
    rook_to_col = lax.cond(is_kingside, lambda: 5, lambda: 3)
    
    rook_type = board_state[from_row, rook_from_col]
    new_board = new_board.at[from_row, rook_from_col].set(EMPTY)
    new_board = new_board.at[from_row, rook_to_col].set(rook_type)
    
    return new_board


def _apply_en_passant_move(board_state: jnp.ndarray, from_row: int, from_col: int, 
                          to_row: int, to_col: int, piece: int) -> jnp.ndarray:
    """Apply en passant: move pawn diagonally, remove captured pawn."""
    # Move pawn
    new_board = board_state.at[from_row, from_col].set(EMPTY)
    new_board = new_board.at[to_row, to_col].set(piece)
    # Remove captured pawn (same row as source, same col as destination)
    new_board = new_board.at[from_row, to_col].set(EMPTY)
    return new_board


def _apply_promotion_move(board_state: jnp.ndarray, from_row: int, from_col: int, 
                         to_row: int, to_col: int, piece: int, promotion_piece: int) -> jnp.ndarray:
    """Apply promotion: replace pawn with promoted piece."""
    promoted_piece = promotion_piece * jnp.sign(piece)  # Keep color
    new_board = board_state.at[from_row, from_col].set(EMPTY)
    new_board = new_board.at[to_row, to_col].set(promoted_piece)
    return new_board


def _apply_pawn_double_move(board_state: jnp.ndarray, from_row: int, from_col: int, 
                           to_row: int, to_col: int, piece: int) -> jnp.ndarray:
    """Apply pawn double move: move pawn 2 squares."""
    new_board = board_state.at[from_row, from_col].set(EMPTY)
    new_board = new_board.at[to_row, to_col].set(piece)
    return new_board


def _update_castling_rights(castling_rights: jnp.ndarray, from_row: int, from_col: int, 
                           to_row: int, to_col: int, piece: int) -> jnp.ndarray:
    """Update castling rights based on the move."""
    new_rights = castling_rights
    abs_piece = jnp.abs(piece)
    
    # King move: lose all castling rights for that color
    is_king_move = abs_piece == KING
    is_white_king = piece == KING
    
    # White king move: lose both white castling rights
    new_rights = lax.cond(
        is_king_move & is_white_king,
        lambda: new_rights.at[0].set(False).at[1].set(False),  # White KS, QS
        lambda: new_rights
    )
    
    # Black king move: lose both black castling rights
    new_rights = lax.cond(
        is_king_move & (~is_white_king),
        lambda: new_rights.at[2].set(False).at[3].set(False),  # Black KS, QS
        lambda: new_rights
    )
    
    # Rook move: lose castling right for that side
    is_rook_move = abs_piece == ROOK
    is_white_rook = piece == ROOK
    
    # White rook moves
    is_white_ks_rook = is_white_rook & (from_row == 0) & (from_col == 7)
    is_white_qs_rook = is_white_rook & (from_row == 0) & (from_col == 0)
    
    new_rights = lax.cond(is_white_ks_rook, lambda: new_rights.at[0].set(False), lambda: new_rights)
    new_rights = lax.cond(is_white_qs_rook, lambda: new_rights.at[1].set(False), lambda: new_rights)
    
    # Black rook moves
    is_black_ks_rook = (~is_white_rook) & is_rook_move & (from_row == 7) & (from_col == 7)
    is_black_qs_rook = (~is_white_rook) & is_rook_move & (from_row == 7) & (from_col == 0)
    
    new_rights = lax.cond(is_black_ks_rook, lambda: new_rights.at[2].set(False), lambda: new_rights)
    new_rights = lax.cond(is_black_qs_rook, lambda: new_rights.at[3].set(False), lambda: new_rights)
    
    return new_rights


def _update_en_passant_pawn(action_type: int, to_row: int, to_col: int) -> jnp.ndarray:
    """Update en passant pawn position based on the move."""
    # Clear en passant pawn position by default
    new_pawn_pos = jnp.array([-1, -1], dtype=jnp.int32)
    
    # Set en passant pawn position if pawn double move (position of the pawn that moved)
    is_pawn_double = action_type == PAWN_DOUBLE
    
    new_pawn_pos = lax.cond(
        is_pawn_double,
        lambda: jnp.array([to_row, to_col], dtype=jnp.int32),  # Position of pawn that moved 2 squares
        lambda: new_pawn_pos
    )
    
    return new_pawn_pos


# --- State Definition ---
class ChessState(NamedTuple):
    """Holds the dynamic state of the batched Chess environment for a single step."""

    boards: jnp.ndarray  # (B, 8, 8) int32 tensor
    current_players: jnp.ndarray  # (B,) int32 tensor (WHITE for White, BLACK for Black)
    castling_rights: jnp.ndarray  # (B, 4) bool tensor [White KS, White QS, Black KS, Black QS]
    en_passant_target: jnp.ndarray  # (B, 2) int32 tensor (row, col) of pawn that can be captured en passant, (-1, -1) if none
    moves_count: jnp.ndarray  # (B,) int32 tensor for tracking number of moves made
    dones: jnp.ndarray  # (B,) bool tensor
    winners: jnp.ndarray  # (B,) int32 tensor (WHITE for White win, BLACK for Black win, EMPTY for draw/ongoing)
    rng: jax.random.PRNGKey  # For any stochasticity


# --- Environment Logic ---
class ChessEnv(Env):
    """
    Functional JAX-based Chess environment logic container.
    
    Static Attributes:
        B: Batch size.
    """

    def __init__(self, B: int):
        """
        Initializes the chess environment configuration holder. Does not create state.

        Args:
            B: Batch size.
        """
        super().__init__(B=B)
        self.B = B
        
        # Pre-compute move lookup tables
        self.move_tables = _create_move_tables()

    def init_state(self, rng: jax.random.PRNGKey) -> ChessState:
        """
        Creates the initial ChessState with standard chess starting position.

        Args:
            rng: JAX PRNG key for initialization.

        Returns:
            The initial ChessState.
        """
        # Standard chess starting position
        # Piece encoding: EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
        # Positive for White, negative for Black
        initial_board = jnp.array([
            [ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK],    # White back rank
            [PAWN, PAWN, PAWN, PAWN, PAWN, PAWN, PAWN, PAWN],    # White pawns
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],    # Empty
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],    # Empty
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],    # Empty
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],    # Empty
            [-PAWN, -PAWN, -PAWN, -PAWN, -PAWN, -PAWN, -PAWN, -PAWN],  # Black pawns
            [-ROOK, -KNIGHT, -BISHOP, -QUEEN, -KING, -BISHOP, -KNIGHT, -ROOK],  # Black back rank
        ], dtype=jnp.int32)
        
        # Broadcast to batch dimension
        boards = jnp.broadcast_to(initial_board[None, :, :], (self.B, 8, 8))
        
        return lax.with_sharding_constraint(
            ChessState(
                boards=boards,
                current_players=jnp.full((self.B,), WHITE, dtype=jnp.int32),  # White starts
                castling_rights=jnp.ones((self.B, 4), dtype=jnp.bool_),  # All castling available
                en_passant_target=jnp.full((self.B, 2), -1, dtype=jnp.int32),  # No EP pawn
                moves_count=jnp.zeros((self.B,), dtype=jnp.int32),
                dones=jnp.zeros((self.B,), dtype=jnp.bool_),
                winners=jnp.full((self.B,), EMPTY, dtype=jnp.int32),
                rng=rng,
            ),
            mesh_rules("batch"),
        )

    def step(
        self, state: ChessState, action: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> Tuple[ChessState, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """
        Applies actions to the current state to compute the next state and outputs. Pure function.

        Args:
            state: The current ChessState.
            action: Tuple of (from_row, from_col, move_plane_index) for each env in batch.

        Returns:
            A tuple (new_state, observations, rewards, dones, info):
                - new_state: The next ChessState.
                - observations: The next observations.
                - rewards: The rewards received. Shape (B,).
                - dones: Boolean flags indicating episode termination. Shape (B,).
                - info: Auxiliary dictionary.
        """
        from_row, from_col, move_plane_index = action
        
        # Create action tuples for each batch element
        batch_actions = (from_row, from_col, move_plane_index)
        
        # Apply moves using vmap to process all boards in parallel
        new_boards, new_castling_rights, new_en_passant_target = jax.vmap(
            move, in_axes=(0, 0, 0, 0, None)
        )(
            state.boards, 
            batch_actions, 
            state.castling_rights, 
            state.en_passant_target, 
            self.move_tables
        )
        
        # Increment moves count
        new_moves_count = state.moves_count + 1
        
        # Switch current players
        new_current_players = -state.current_players
        
        # TODO: Implement game termination logic (checkmate, stalemate, 100-move truncation)
        # For now, use placeholder values
        new_dones = state.dones  # Placeholder
        new_winners = state.winners  # Placeholder
        rewards = jnp.zeros((self.B,), dtype=jnp.float32)  # Placeholder
        
        # Create new state
        new_state = ChessState(
            boards=new_boards,
            current_players=new_current_players,
            castling_rights=new_castling_rights,
            en_passant_target=new_en_passant_target,
            moves_count=new_moves_count,
            dones=new_dones,
            winners=new_winners,
            rng=state.rng,
        )
        
        # Observations are the board state
        observations = new_state.boards
        
        info = {}
        return new_state, observations, rewards, new_dones, info

    def reset(
        self, rng: jax.random.PRNGKey
    ) -> Tuple[ChessState, jnp.ndarray, Dict[str, Any]]:
        """
        Resets environments to initial chess state using the provided RNG key. Pure function.

        Args:
            rng: JAX PRNG key for initialization.

        Returns:
            A tuple (initial_state, initial_observations, info):
                - initial_state: The initial ChessState.
                - initial_observations: Observations after reset.
                - info: Auxiliary dictionary.
        """
        new_state = self.init_state(rng)
        
        # Observations are the board state
        initial_observations = new_state.boards
        
        info = {}
        return new_state, initial_observations, info

    def get_action_mask(self, state: ChessState) -> jnp.ndarray:
        """
        Returns a boolean mask of valid actions for the given state. Pure function.

        Args:
            state: The current ChessState.

        Returns:
            A boolean JAX array mask (True=valid, False=invalid).
            Shape: (B, 8, 8, 73)
        """
        # TODO: Implement action mask generation
        # Phase 1: Generate pseudo-legal moves
        # Phase 2: Filter for full legality (king safety)
        
        raise NotImplementedError("Chess action mask logic not yet implemented")

    def initialize_trajectory_buffers(self, max_steps: int) -> Tuple[jnp.ndarray, ...]:
        """
        Creates and returns pre-allocated JAX arrays for storing trajectory data.

        Args:
            max_steps: The maximum length of the trajectories to buffer.

        Returns:
            A tuple containing JAX arrays for observations, actions, values,
            rewards, dones, log_probs, and current_players_buffer.
        """
        obs_shape = self.observation_shape
        act_shape = self.action_shape  # Action is (from_row, from_col, move_plane_index), shape (3,)

        observations = jnp.zeros((max_steps, self.B) + obs_shape, dtype=jnp.int32)
        actions = jnp.zeros((max_steps, self.B) + act_shape, dtype=jnp.int32)
        values = jnp.zeros((max_steps + 1, self.B), dtype=jnp.float32)
        rewards = jnp.zeros((max_steps, self.B), dtype=jnp.float32)
        dones = jnp.zeros((max_steps, self.B), dtype=jnp.bool_)
        log_probs = jnp.zeros((max_steps, self.B), dtype=jnp.float32)
        current_players_buffer = jnp.zeros((max_steps, self.B), dtype=jnp.int32)

        sharded_output = jax.tree.map(
            lambda x: lax.with_sharding_constraint(x, mesh_rules("buffer")),
            (
                observations,
                actions,
                values,
                rewards,
                dones,
                log_probs,
                current_players_buffer,
            ),
        )
        return sharded_output

    @property
    def observation_shape(self) -> tuple:
        """Returns the shape tuple of a single observation (excluding batch dim)."""
        # TODO: Define chess observation shape
        return (8, 8)  # Placeholder - board representation

    @property
    def action_shape(self) -> tuple:
        """Returns the shape tuple of a single action (excluding batch dim)."""
        # Action is (from_row, from_col, move_plane_index)
        return (3,) 