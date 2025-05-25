import jax
import jax.numpy as jnp
import numpy as np
import argparse
import os
import orbax.checkpoint as ocp
from flax.training import (
    train_state,
)  # Needed for type hint, even if not used directly for state creation
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Tuple

# Import necessary components from your project structure
# Adjust these paths if your project structure is different
from alphagomoku.environments.gomoku import GomokuJaxEnv, GomokuState
from alphagomoku.models.gomoku.actor_critic import ActorCritic

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Suppress verbose logging from libraries
logging.getLogger("absl").setLevel(logging.WARNING)

# --- Helper Functions ---


def render_board(board: np.ndarray):
    """Renders the Gomoku board to the console."""
    size = board.shape[0]
    print("   " + " ".join([f"{i:2}" for i in range(size)]))  # Column headers
    print("  " + "---" * size)
    for r in range(size):
        row_str = f"{r:2}|"  # Row header
        for c in range(size):
            if board[r, c] == 1:
                row_str += " X "  # Player 1 (e.g., Black)
            elif board[r, c] == -1:
                row_str += " O "  # Player -1 (e.g., White)
            else:
                row_str += " . "  # Empty
        print(row_str + "|")
    print("  " + "---" * size)


def get_human_move(board_size: int, valid_mask: np.ndarray) -> Tuple[int, int]:
    """Prompts the human player for a move and validates it."""
    while True:
        try:
            move_str = input(f"Enter your move (row col, e.g., '4 4'): ")
            row, col = map(int, move_str.split())
            if not (0 <= row < board_size and 0 <= col < board_size):
                print(
                    f"Invalid input: Coordinates must be between 0 and {board_size - 1}."
                )
            elif not valid_mask[row, col]:
                print("Invalid move: Cell is already occupied.")
            else:
                return row, col
        except ValueError:
            print(
                "Invalid input format. Please enter row and column separated by a space."
            )
        except Exception as e:
            print(f"An error occurred: {e}")


def get_agent_move(model_apply_fn, params, obs, current_player, valid_mask):
    """Determines the agent's move based on the policy."""
    # Add batch dimension if missing (should be shape (1, H, W))
    if obs.ndim == 2:
        obs = jnp.expand_dims(obs, axis=0)
    if current_player.ndim == 0:
        current_player = jnp.expand_dims(current_player, axis=0)

    # Get policy distribution
    pi_dist, _ = model_apply_fn({"params": params}, obs, current_player)

    # Mask invalid actions in logits
    flat_valid_mask = valid_mask.flatten()  # Flatten for distribution
    masked_logits = jnp.where(
        flat_valid_mask, pi_dist.logits[0], -jnp.inf
    )  # Index [0] for batch dim

    # Choose the action with the highest probability (argmax)
    best_flat_action = jnp.argmax(masked_logits)

    # Convert flat action back to (row, col)
    # valid_mask_batch shape is (1, H, W), so shape[1] gives H (board_size)
    board_size = valid_mask.shape[1]
    row = best_flat_action // board_size
    col = best_flat_action % board_size
    return jnp.array([row, col], dtype=jnp.int32)


# --- Main Play Function ---


def play(cfg: DictConfig):
    """Runs the interactive game playing session using Hydra config."""

    # Construct the full checkpoint path from config
    # Use Hydra's utils to get the original CWD if checkpoint_dir is relative
    try:
        original_cwd = hydra.utils.get_original_cwd()
        checkpoint_base_dir = os.path.join(original_cwd, cfg.checkpoint_dir)
    except (
        ValueError
    ):  # Handle case where original_cwd is not available (e.g., testing)
        checkpoint_base_dir = cfg.checkpoint_dir
        logging.warning(
            "Could not determine original CWD, assuming checkpoint_dir is absolute or relative to current dir."
        )

    full_checkpoint_path = os.path.join(checkpoint_base_dir, str(cfg.checkpoint_step))

    board_size = cfg.gomoku.board_size
    win_length = cfg.gomoku.win_length
    human_player = 1 if cfg.user_plays.lower() == "black" else -1

    logging.info(f"Loading agent from checkpoint: {full_checkpoint_path}")
    logging.info(f"Board size: {board_size}x{board_size}, Win length: {win_length}")
    logging.info(
        f"Human plays as {'X (Black)' if human_player == 1 else 'O (White)'}. Agent plays as {'O (White)' if human_player == 1 else 'X (Black)'}."
    )

    # --- Restore Checkpoint ---
    mngr_options = ocp.CheckpointManagerOptions(
        create=False
    )  # Don't create if it doesn't exist
    # Use the base checkpoint directory for the manager
    checkpointer = ocp.CheckpointManager(
        directory=checkpoint_base_dir, options=mngr_options
    )

    # Orbax uses the directory name (step number) directly for restoration
    target_step = cfg.checkpoint_step
    available_steps = checkpointer.all_steps()  # Get available steps first
    if target_step not in available_steps:
        logging.error(
            f"Checkpoint step {target_step} not found in directory: {checkpoint_base_dir}"
        )
        logging.error(f"Available steps: {available_steps}")
        return

    logging.info(f"Restoring checkpoint from step {target_step}...")
    # Restore the dictionary containing the training state(s)
    # The saved structure is {'black': {...}, 'white': {...}}
    # We must use Composite restore to match the Composite save structure.
    restore_args = ocp.args.Composite(
        black=ocp.args.StandardRestore(), white=ocp.args.StandardRestore()
    )
    restored_data = checkpointer.restore(target_step, args=restore_args)

    agent_role = cfg.agent_role.lower()  # Ensure lowercase for comparison
    if agent_role not in restored_data:
        logging.error(
            f"Agent role '{cfg.agent_role}' not found in the checkpoint keys: {list(restored_data.keys())}"
        )
        logging.error(
            "Please ensure 'agent_role' in your config (play.yaml) is either 'black' or 'white'."
        )
        return

    # Access the specific agent's data
    agent_data = restored_data[agent_role]

    if "params" not in agent_data:
        logging.error(
            f"Checkpoint for agent '{agent_role}' at step {target_step} does not contain 'params'. Contents: {agent_data.keys()}"
        )
        # Attempting to restore from legacy format (if the agent_data IS a TrainState - less likely now)
        if isinstance(agent_data, train_state.TrainState):
            logging.warning(
                f"Attempting to load params from legacy TrainState object for agent '{agent_role}'."
            )
            agent_params = agent_data.params
        else:
            logging.error(
                f"Could not find 'params' in the loaded data for agent '{agent_role}'."
            )
            return  # Cannot find params
    else:
        agent_params = agent_data["params"]
        logging.info(f"Parameters for '{agent_role}' agent loaded successfully.")

    # --- Initialize Model and Environment ---
    agent_model = ActorCritic(
        board_size=board_size, name=f"{agent_role}_player"
    )  # Optional: add name
    env = GomokuJaxEnv(
        B=1, board_size=board_size, win_length=win_length
    )  # Batch size 1

    # --- Game Setup ---
    rng = jax.random.PRNGKey(0)  # RNG for environment reset (can be fixed for play)
    rng, reset_rng = jax.random.split(rng)

    env_state, current_obs, _ = env.reset(reset_rng)
    # Remove batch dimension for single game interaction
    current_obs_np = np.array(current_obs[0])
    env_state = jax.tree.map(
        lambda x: x[0] if x.shape[0] == 1 else x, env_state
    )  # Squeeze batch dim from state

    logging.info("Starting new game.")
    render_board(current_obs_np)

    # --- Game Loop ---
    while not env_state.dones:
        current_player = int(env_state.current_players)  # Get scalar player value

        # Add batch dimension back for state before getting mask and stepping
        # Note: rng doesn't need batch dim typically, but let's batch everything for consistency
        env_state_batch = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), env_state)

        # Get valid mask using the batched state
        valid_mask_batch = env.get_action_mask(env_state_batch)
        valid_mask = np.array(
            valid_mask_batch[0]
        )  # Squeeze batch dim for human/agent logic

        if current_player == human_player:
            # --- Human's Turn ---
            logging.info("Your turn.")
            action_coords = get_human_move(board_size, valid_mask)
            action = jnp.array(action_coords, dtype=jnp.int32)

        else:
            # --- Agent's Turn ---
            logging.info("Agent's turn...")
            # Need to add batch dimension back for model input obs
            obs_batch = jnp.expand_dims(current_obs_np, axis=0)
            player_batch = jnp.expand_dims(
                env_state.current_players, axis=0
            )  # Already batched in env_state_batch
            # Pass the batched mask directly
            # valid_mask_batch = jnp.expand_dims(valid_mask, axis=0) # Already have batched mask

            action = get_agent_move(
                agent_model.apply,
                agent_params,
                obs_batch,
                player_batch,  # Use player from env_state_batch
                valid_mask_batch,  # Pass the batched valid mask
            )
            # Squeeze action batch dim for logging and stepping logic below
            # action = action[0] # REMOVED: get_agent_move already returns non-batched action
            # Convert action (jnp array) to tuple of standard Python ints for logging
            action_np = np.array(action)
            move_coords = (int(action_np[0]), int(action_np[1]))
            logging.info(f"Agent chose move: {move_coords}")

        # --- Step Environment ---
        # Add batch dimension back for action before stepping
        action_batch = jnp.expand_dims(action, axis=0)
        # env_state_batch is already created above

        env_state_batch, next_obs_batch, rewards_batch, dones_batch, _ = env.step(
            env_state_batch, action_batch
        )

        # Remove batch dimension for next iteration
        current_obs_np = np.array(next_obs_batch[0])
        env_state = jax.tree.map(lambda x: x[0], env_state_batch)  # Squeeze batch dim

        render_board(current_obs_np)

        # --- Check Game Over ---
        if env_state.dones:
            winner = int(env_state.winners)
            logging.info("=" * 20 + " Game Over " + "=" * 20)
            if winner == human_player:
                logging.info("Congratulations! You won!")
            elif winner == -human_player:
                logging.info("The agent won.")
            else:
                logging.info("It's a draw!")
            break  # Exit loop


# --- Hydra Main Block ---
@hydra.main(
    config_path="../../../config", config_name="play", version_base=None
)  # Adjust config_path
def main(cfg: DictConfig) -> None:
    print("Effective Hydra Configuration:")
    print(OmegaConf.to_yaml(cfg))
    play(cfg)


if __name__ == "__main__":
    main()
