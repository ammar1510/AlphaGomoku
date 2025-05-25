# JAX-Based Chess Environment: Design Plan

This document outlines the design for a JAX-compatible, batched chess environment.

## 1. Core Environment Structure

-   **Framework:** JAX for JIT compilation and hardware acceleration.
-   **Batching:** All operations will be designed to support a batch of environments `B`.
-   **File Structure (Tentative):**
    -   `src/alphachess/environments/chess_env.py` (main environment logic)
    -   `src/alphachess/environments/chess_utils.py` (helper functions, move generation logic, constants)
-   **Game Truncation:** Games will be truncated at a maximum of 100 moves. No 50-move rule or 3-fold repetition detection.

## 2. State Representation (`ChessState` - NamedTuple)

-   `boards: jnp.ndarray`
    -   Shape: `(B, 8, 8)`
    -   Data Type: `jnp.int32`
    -   Encoding:
        -   0: Empty square
        -   1: White Pawn, 2: White Knight, ..., 6: White King
        -   -1: Black Pawn, -2: Black Knight, ..., -6: Black King
-   `current_players: jnp.ndarray`
    -   Shape: `(B,)`
    -   Data Type: `jnp.int32` (1 for White, -1 for Black)
-   `castling_rights: jnp.ndarray`
    -   Shape: `(B, 4)`
    -   Data Type: `jnp.bool_`
    -   Order: `[White Kingside, White Queenside, Black Kingside, Black Queenside]`
-   `en_passant_target: jnp.ndarray`
    -   Shape: `(B, 2)`
    -   Data Type: `jnp.int32`
    -   Content: `(row, col)` of the target square for en passant. A special value (e.g., `(-1, -1)`) if no EP target.
-   `dones: jnp.ndarray`
    -   Shape: `(B,)`
    -   Data Type: `jnp.bool_`
-   `winners: jnp.ndarray`
    -   Shape: `(B,)`
    -   Data Type: `jnp.int32` (1 for White win, -1 for Black win, 0 for draw/ongoing).
-   `rng: jax.random.PRNGKey` (For any stochasticity, though chess is deterministic post-initialization).

## 3. Action Representation

-   **Style:** AlphaZero-inspired.
-   **Policy Output Shape:** `(B, 8, 8, 73)`
    -   `(8, 8)`: Source square of the move.
    -   `73`: Number of "move planes" or "move types".
-   **Move Planes (K=73):**
    -   **56 Queen-like Moves:** 8 directions * 7 distances. Covers moves for Q, R, B, K, P (including pawn promotions to Queen).
    -   **8 Knight Moves:** Specific L-shaped moves.
    -   **9 Pawn Underpromotions:** 3 pawn move types (forward, diag-left, diag-right to back rank) * 3 underpromotion pieces (N, R, B).
-   **Action in `step` function:** Will be decoded from the chosen `(from_row, from_col, move_plane_index)`.

## 4. Key Functions & Logic

### `GomokuJaxEnv.step(state: ChessState, action: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[ChessState, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict]`

-   Decodes `action` (from_sq, plane_idx) to `(to_sq, promotion_piece)`.
-   Updates `boards` array:
    -   Move piece.
    -   Handle captures.
    -   Handle promotions (Queen promotions via queen-like planes, underpromotions via dedicated planes).
    -   Handle special moves:
        -   En Passant: If move is EP, remove captured pawn.
        -   Castling: If king move is a castling move, also move the corresponding rook.
-   Updates `castling_rights`, `en_passant_target`.
-   Determines `dones` (checkmate, stalemate, or 100-move truncation).
-   Determines `rewards`.
-   Switches `current_players`.
-   Returns `(new_state, observations, rewards, dones, info)`.

### `GomokuJaxEnv.get_action_mask(state: ChessState) -> jnp.ndarray`

-   **Output Shape:** `(B, 8, 8, 73)` boolean mask.
-   **Process:**
    1.  **Phase 1: Generate Pseudo-Legal Moves:**
        -   For each piece of the current player on each square:
            -   Iterate through all 73 move planes.
            -   Check basic piece movement rules, on-board conditions, target square occupancy (empty or opponent), and path clearance for sliding pieces.
            -   Handle pawn-specific logic (pushes, captures, EP, promotions).
            -   Handle castling pseudo-legality (rights, path clear).
        -   This phase results in a `pseudo_legal_mask`.
    2.  **Phase 2: Filter for Full Legality (King Safety):**
        -   For each `True` entry in `pseudo_legal_mask`:
            -   Temporarily apply the move.
            -   Check if the current player's king is in check on the temporary board.
            -   If king is not in check, the move is fully legal.
-   **Helper JAX Arrays for Plane Properties (shape `(73,)`):**
    -   `is_knight_plane: jnp.ndarray`
    -   `is_queenlike_plane: jnp.ndarray`
    -   `is_underpromotion_plane: jnp.ndarray`
    -   `plane_delta_row: jnp.ndarray`
    -   `plane_delta_col: jnp.ndarray`
    -   `plane_promotion_piece_type: jnp.ndarray` (0 if no promotion, or piece type for underpromotion planes, QUEEN implicitly for queen-like promotion moves).

### `GomokuJaxEnv.reset(rng: jax.random.PRNGKey) -> Tuple[ChessState, jnp.ndarray, Dict]`

-   Initializes the board to the standard chess starting position.
-   Sets initial player (White), castling rights, EP target (none), clocks.

## 5. Helper Functions (`chess_utils.py`)

-   `_is_square_attacked(board, r, c, attacker_color)`: Checks if a square is attacked.
-   `_is_path_clear(board, r1, c1, r2, c2)`: Checks for empty squares between two points.
-   `_find_king_pos(board_slice, king_color)`: Locates the king.
-   `_apply_move_on_temp_board(...)`: Simulates a move for king safety checks.
-   Functions for decoding plane indices to move details.
-   Functions for specific piece move generation logic (e.g., pawn moves, knight moves).
-   Functions for updating castling rights, EP target based on moves.

## 6. Constants

-   Piece encodings (PAWN=1, ..., KING=6).
-   Player encodings (WHITE=1, BLACK=-1).
-   Plane definitions and mappings.

This plan will serve as a living document. 