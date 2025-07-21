# Chess Environment Implementation Plan

## Step Method Implementation

### 1. Apply Moves (Batch Processing)
```python
new_boards, new_castling_rights, new_en_passant_target = jax.vmap(move)(
    state.boards, action, state.castling_rights, state.en_passant_target, 
    state.current_players, self.move_tables
)
```

### 2. Check Game Ending Conditions
- **Legal Move Detection**: Use `get_action_mask()` to check if opponent has legal moves
- **King Safety Check**: Implement `_is_in_check()` helper function
- **Game Ending Logic**:
  - Checkmate: No legal moves AND king in check → Winner = current player
  - Stalemate: No legal moves AND king NOT in check → Draw
  - 100-move limit: Truncate games → Draw

### 3. Update Game State
- **`dones`**: `state.dones | checkmate | stalemate | move_limit`
- **`winners`**: Current player if checkmate, EMPTY otherwise
- **`current_players`**: Switch players if game continues: `-state.current_players`
- **`boards`**: Use `new_boards` from move application
- **`castling_rights`**: Use `new_castling_rights` from move application
- **`en_passant_target`**: Use `new_en_passant_target` from move application

### 4. Calculate Rewards
- **Win**: 1.0 for current player on checkmate
- **Draw/Ongoing**: 0.0

### 5. Return Values
- **Observations**: `new_boards` (chess board states)
- **New State**: Updated ChessState with all fields
- **Apply sharding constraints**: `mesh_rules("batch")`

### Helper Functions Needed
- `_is_in_check(boards, current_players)` → Check if king is under attack
- Game ending detection using legal move analysis 