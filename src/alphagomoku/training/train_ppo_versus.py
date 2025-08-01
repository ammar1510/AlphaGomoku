import jax
import nest_asyncio
nest_asyncio.apply()
import logging
import jax.numpy as jnp
import optax
import wandb
import orbax.checkpoint as ocp
import flax.linen as nn
from flax.training import train_state
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Tuple, Optional
import time
import os
from functools import partial
import hydra.utils
from orbax.checkpoint.utils import fully_replicated_host_local_array_to_global_array



# --- Configure Logging ---
# Get a logger for this module
logger = logging.getLogger(__name__)
# Basic configuration (can be enhanced)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Set higher level for verbose libraries like absl (used by Orbax)
logging.getLogger("absl").setLevel(logging.WARNING)


# --- Initialize JAX Distributed System ---
jax.distributed.initialize()
logger.info(f"JAX distributed system initialized on process {jax.process_index()}.")


# --- Import Modules ---
from alphagomoku.environments.gomoku import GomokuJaxEnv, GomokuState
from alphagomoku.models.gomoku.actor_critic import ActorCritic
from alphagomoku.policy.ppo import PPOConfig, PPOTrainer
from alphagomoku.common.rollout import run_episode
from alphagomoku.common.sharding import mesh_rules



# --- Evaluation Function ---
def run_evaluation_games(
    eval_env: GomokuJaxEnv, # Changed to accept a dedicated eval_env
    black_actor_critic: ActorCritic,
    black_params: Any,
    white_actor_critic: ActorCritic,
    white_params: Any,
    rng: jax.random.PRNGKey,
    # board_size: int, # Removed, will use eval_env.board_size
) -> Dict[str, Any]:

    logger.info("Compiling evaluation function...")
    """Runs a set of evaluation games between two agents using a dedicated evaluation environment."""
    num_eval_games = eval_env.B # Number of games is determined by the eval_env's batch size
    # logger.info(f"Running {num_eval_games} evaluation games...") # This log is now in the main loop

    eval_rng, _ = jax.random.split(rng)

    game_trajectory = run_episode(
        env=eval_env, # Use the dedicated evaluation environment
        black_actor_critic=black_actor_critic,
        black_params=black_params,
        white_actor_critic=white_actor_critic,
        white_params=white_params,
        rng=eval_rng,
        buffer_size=eval_env.board_size * eval_env.board_size, # Use board_size from eval_env
    )

    terminated_mask = game_trajectory["dones"]
    final_rewards = game_trajectory["rewards"]
    final_players = game_trajectory["current_players"]

    term_indices = jnp.argmax(terminated_mask, axis=0)
    rewards_at_termination = jnp.take_along_axis(final_rewards, term_indices[jnp.newaxis, :], axis=0).squeeze(0)
    player_at_termination = jnp.take_along_axis(final_players, term_indices[jnp.newaxis, :], axis=0).squeeze(0)

    black_wins = jnp.sum((player_at_termination == 1) & (rewards_at_termination == 1))
    white_wins = jnp.sum((player_at_termination == -1) & (rewards_at_termination == 1))
    
    # num_actual_games_played is now simply num_eval_games (eval_env.B)
    draws = num_eval_games - black_wins - white_wins

    metrics = {
        "eval/black_wins": black_wins,
        "eval/white_wins": white_wins,
        "eval/draws": draws,
        "eval/black_win_rate": (black_wins / num_eval_games) if num_eval_games > 0 else 0.0,
        "eval/white_win_rate": (white_wins / num_eval_games) if num_eval_games > 0 else 0.0,
        "eval/draw_rate": (draws / num_eval_games) if num_eval_games > 0 else 0.0,
    }
    # logger.info(f"Evaluation results: {metrics}") # More detailed logging will be done in the main loop
    # This function's primary role is to return the metrics dict.
    return metrics


# --- Training State ---
class TrainingState(train_state.TrainState):
    # Inherits apply_fn, params, tx, opt_state
    rng: jax.random.PRNGKey
    update_step: int = 0


# --- Main Training Function ---
@hydra.main(
    config_path="../../../config", config_name="train_versus", version_base=None
)
def train(cfg: DictConfig):
    """Runs the PPO training loop for two agents (black vs white), configured by Hydra."""

    logger.info("Starting adversarial training process...")
    logger.info("Effective Hydra Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Log JAX devices
    logger.info(f"training devices: {jax.devices()}")

    # === Initialization ===
    start_time = time.time()
    output_dir = os.getcwd()
    logger.info(f"Hydra output directory: {output_dir}")

    # Determine workspace root using Hydra's utility
    workspace_root = hydra.utils.get_original_cwd()
    logger.info(f"Detected workspace root: {workspace_root}")
    artifacts_dir = os.path.join(workspace_root, "artifacts")

    # Determine if this is the chief process for distributed training
    is_main_process = jax.process_index() == 0

    # WandB Setup
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # Determine the run name: append suffix if provided, otherwise let wandb decide

    run = None # Initialize run to None
    if is_main_process:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            config=wandb_config,
            mode=cfg.wandb.mode,
            dir=artifacts_dir,  # Use artifacts_dir directly, path = artifacts_dir/wandb
            sync_tensorboard=True,
            reinit=True,
        )
        print(f"WandB Run URL: {run.url}")
    else:
        logger.info(f"WandB initialization skipped on process {jax.process_index()}.")

    # RNG Setup
    rng = jax.random.PRNGKey(cfg.seed)
    rng, env_rng, model_rng_b, model_rng_w, train_rng_b, train_rng_w = jax.random.split(
        rng, 6
    )

    # Environment Setup
    env = GomokuJaxEnv(
        B=cfg.num_envs,
        board_size=cfg.gomoku.board_size,
        win_length=cfg.gomoku.win_length,
    )
    # Create a separate environment for evaluation
    eval_env = GomokuJaxEnv(
        B=cfg.eval_games, # Use eval_games for the batch size
        board_size=cfg.gomoku.board_size,
        win_length=cfg.gomoku.win_length,
    )

    # === Model Setup (Separate Black and White) ===
    black_model = ActorCritic(board_size=cfg.gomoku.board_size, name="black_agent")
    white_model = ActorCritic(board_size=cfg.gomoku.board_size, name="white_agent")

    dummy_obs = jnp.zeros((1, cfg.gomoku.board_size, cfg.gomoku.board_size))
    dummy_player = jnp.ones((1,), dtype=jnp.int32)

    black_params = black_model.init(model_rng_b, dummy_obs, dummy_player)["params"]
    white_params = white_model.init(model_rng_w, dummy_obs, dummy_player)["params"]
    black_params = jax.device_put(black_params, mesh_rules("replicated"))
    white_params = jax.device_put(white_params, mesh_rules("replicated"))

    # === Optimizer Setup (Separate Black and White) ===
    total_updates = cfg.num_epochs
    lr_schedule = optax.linear_schedule(
        init_value=cfg.ppo.learning_rate,
        end_value=cfg.ppo.learning_rate * cfg.ppo.lr_decay_factor,
        transition_steps=total_updates,
    )
    # Shared optimizer structure, but separate instances will be used
    optimizer_def = optax.chain(
        optax.clip_by_global_norm(cfg.ppo.max_grad_norm),
        optax.adam(learning_rate=lr_schedule, eps=cfg.ppo.adam_eps),
    )

    black_tx = optimizer_def
    white_tx = optimizer_def

    # === Training State Setup (Separate Black and White) ===
    black_train_state = TrainingState.create(
        apply_fn=black_model.apply,
        params=black_params,
        tx=black_tx,
        rng=train_rng_b,
        update_step=0,
    )
    white_train_state = TrainingState.create(
        apply_fn=white_model.apply,
        params=white_params,
        tx=white_tx,
        rng=train_rng_w,
        update_step=0,
    )

    # PPO Trainer Setup (Shared config, trainer is stateless)
    ppo_config = PPOConfig(
        learning_rate=cfg.ppo.learning_rate,  # Note: LR schedule is handled by optimizer state
        clip_eps=cfg.ppo.clip_eps,
        vf_coef=cfg.ppo.vf_coef,
        entropy_coef=cfg.ppo.entropy_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        update_epochs=cfg.ppo.update_epochs,
        num_minibatches=cfg.ppo.num_minibatches,
        seed=cfg.seed,  # Seed for internal shuffling if needed
    )
    ppo_trainer = PPOTrainer()

    # Checkpointing Setup
    checkpoint_path = os.path.join(
        artifacts_dir, cfg.checkpoint_dir
    )  # Use configured dir directly
    logger.info(f"Checkpoint path: {checkpoint_path}")
    os.makedirs(checkpoint_path, exist_ok=True)
    orbax_options = ocp.CheckpointManagerOptions(
        save_interval_steps=cfg.save_frequency, max_to_keep=cfg.max_checkpoints
    )
    checkpointer = ocp.CheckpointManager(
        directory=checkpoint_path, options=orbax_options
    )

    # --- Restore Checkpoint (Optional) ---
    latest_step = checkpointer.latest_step()
    start_epoch = 0
    total_env_steps = 0
    if latest_step is not None:
        logger.info(f"Restoring checkpoint from step {latest_step}...")

        # Restore the combined dictionary using PyTreeRestore
        # The target structure should match what was saved (a dict with 'black' and 'white' keys)
        restore_target = {
            "black": {
                "params": black_params,
                "opt_state": black_tx.init(black_params),
                "rng": train_rng_b,
                "update_step": 0,
                "total_env_steps": 0, # Expect total_env_steps for black
            },
            "white": {
                "params": white_params,
                "opt_state": white_tx.init(white_params),
                "rng": train_rng_w,
                "update_step": 0,
                "total_env_steps": 0, # Expect total_env_steps for white
            },
        }
        restored_data = checkpointer.restore(
            latest_step,
            args=ocp.args.Composite(
                black=ocp.args.StandardRestore(restore_target["black"]),
                white=ocp.args.StandardRestore(restore_target["white"]),
            ),
        )

        # Recreate the training states from the restored dictionaries
        black_train_state = TrainingState.create(
            apply_fn=black_model.apply,
            params=restored_data["black"]["params"],
            tx=black_tx,  # Optimizer structure is recreated
            opt_state=restored_data["black"]["opt_state"],
            rng=restored_data["black"]["rng"],
        )
        white_train_state = TrainingState.create(
            apply_fn=white_model.apply,
            params=restored_data["white"]["params"],
            tx=white_tx,  # Optimizer structure is recreated
            opt_state=restored_data["white"]["opt_state"],
            rng=restored_data["white"]["rng"],
        )

        # Set the epoch number (should be same for both, use black's)
        start_epoch = restored_data["black"]["update_step"]
        black_train_state = black_train_state.replace(update_step=start_epoch)
        white_train_state = white_train_state.replace(
            update_step=start_epoch
        )  # Ensure consistency

        logger.info(f"Restored state epoch: {start_epoch}")

        # Restore total_env_steps (assuming it was saved under 'black' for simplicity, but could be separate)
        total_env_steps = restored_data["black"].get("total_env_steps", 0)
        logger.info(f"Restored total_env_steps to: {total_env_steps}")
        if total_env_steps == 0:
            logger.warning(
                f"Restored state, but total_env_steps counter started from 0 (either initial training or older checkpoint format)."
            )
    else:
        logger.info("No checkpoint found, starting training from scratch.")
        # Initial state already set up correctly
        start_epoch = 0
        black_train_state = black_train_state.replace(update_step=0)
        white_train_state = white_train_state.replace(update_step=0)

    # --- Pre-compile functions---

    jit_rollout = jax.jit(
        run_episode,
        static_argnames=[
            "env",
            "black_actor_critic",
            "white_actor_critic",
            "buffer_size",
        ],
    )

    logger.info("Compiling update function for black agent...")
    jit_update_step_black = jax.jit(
        partial(
            PPOTrainer.update_step,
            model=black_model,
        ),
        static_argnames=["optimizer", "config"],
    )
    logger.info("Compiling update function for white agent...")
    jit_update_step_white = jax.jit(
        partial(
            PPOTrainer.update_step,
            model=white_model,
        ),
        static_argnames=["optimizer", "config"],
    )

    jit_ppo_gae = jax.jit(
        partial(
            PPOTrainer.compute_gae_targets,
            gamma=cfg.ppo.gamma,
            gae_lambda=cfg.ppo.gae_lambda,
        ),
        static_argnames=["gamma", "gae_lambda"],
    )

    jit_eval = jax.jit(
        run_evaluation_games,
        static_argnames=["eval_env", "black_actor_critic", "white_actor_critic"],
    )

    jit_prepare_batch = jax.jit(PPOTrainer.prepare_batch_for_update)

    # --- Training Loop --- (Iterates over epochs)
    logger.info(
        f"Starting training from epoch {start_epoch} for {cfg.num_epochs - start_epoch} more epochs ({cfg.num_epochs} total epochs configured)..."
    )

    # Initialize environment state outside the loop
    env_state, current_obs, _ = env.reset()

    for epoch in range(start_epoch, cfg.num_epochs):
        iter_start_time = time.time()
        # Get current parameters and RNGs for both agents
        black_params_current = black_train_state.params
        white_params_current = white_train_state.params
        black_rng_current = black_train_state.rng
        white_rng_current = white_train_state.rng  # Need separate RNG for white updates

        # === Rollout Phase (Using Black and White Agents) ===
        # Split black's RNG for rollout and its own update
        rollout_rng, black_rng_current = jax.random.split(black_rng_current)

        full_trajectory = jit_rollout(  # Rollout RNG is consumed here
            env=env,
            black_actor_critic=black_model,
            black_params=black_params_current,
            white_actor_critic=white_model,
            white_params=white_params_current,
            rng=rollout_rng,
            buffer_size=cfg.gomoku.board_size * cfg.gomoku.board_size,
        )

        # Update agent RNGs in their states *after* potential use in rollout/update
        black_train_state = black_train_state.replace(rng=black_rng_current)
        # white_train_state = white_train_state.replace(rng=white_rng_current) # White RNG updated after its update step

        steps_this_iter = full_trajectory["valid_mask"].sum().item()
        total_env_steps += steps_this_iter

        # === GAE Calculation Phase (Same for both perspectives initially) ===
        advantages, returns = jit_ppo_gae(
            rewards=full_trajectory["rewards"],
            values=full_trajectory[
                "values"
            ],  # Values are V(s) from the *current* player's perspective
            dones=full_trajectory["dones"],
            gamma=cfg.ppo.gamma,
            gae_lambda=cfg.ppo.gae_lambda,
        )

        # === Data Preparation Phase (Common Data) ===
        # Data generated by interaction of both agents
        batch_data = {
            "observations": full_trajectory["observations"],  # (T, B, H, W)
            "actions": full_trajectory["actions"],  # (T, B, 2)
            "logprobs_old": full_trajectory[
                "logprobs"
            ],  # (T, B) - Logprobs from the player who took the action
            "advantages": advantages,  # (T, B) - GAE calculated based on player-specific values
            "returns": returns,  # (T, B) - Returns calculated based on player-specific values
            "current_players": full_trajectory[
                "current_players"
            ],  # (T, B) - Player whose turn it was
            "valid_mask": full_trajectory["valid_mask"],  # (T, B)
        }

        # Prepare batch reshapes (T, B, ...) -> (T * B, ...)
        prepared_batch_flat = jit_prepare_batch(batch_data)

        # === Update Phase (Separate for Black and White using Masks) ===

        # --- Create Agent-Specific Masks ---
        # Original mask indicating valid steps in the rollout
        original_valid_mask_flat = prepared_batch_flat["valid_mask"]
        # Mask indicating steps taken by black
        is_black_player_flat = prepared_batch_flat["current_players"] == 1
        # Mask indicating steps taken by white
        is_white_player_flat = prepared_batch_flat["current_players"] == -1

        # Combine original validity with player-specific masks
        black_valid_mask_flat = original_valid_mask_flat & is_black_player_flat
        white_valid_mask_flat = original_valid_mask_flat & is_white_player_flat

        # --- Black Agent Update ---
        # Create a batch copy and insert the black-specific valid mask
        # The shapes of all tensors remain (T*B, ...), avoiding recompilation
        black_update_batch = prepared_batch_flat.copy()
        black_update_batch["valid_mask"] = black_valid_mask_flat

        # Check if black agent took any steps in this rollout (check mask sum)
        # update_rng_black, black_rng_current = jax.random.split(black_rng_current) # RNG split happens *before* call now

        # Split RNG *before* the call
        update_rng_black, black_rng_current = jax.random.split(black_rng_current)

        (
            update_rng_black,  # Consumed RNG - This is redundant, the JIT returns the *new* key state
            black_params_updated,
            black_opt_state_updated,
            black_update_metrics,
        ) = jit_update_step_black(  # Call the pre-jitted black version
            rng=update_rng_black,
            params=black_params_current,
            optimizer=black_tx,
            opt_state=black_train_state.opt_state,
            full_batch=black_update_batch,  # Pass the masked batch
            config=ppo_config,
        )
        # Update black agent state
        black_train_state = black_train_state.replace(
            params=black_params_updated,
            opt_state=black_opt_state_updated,
            # rng=update_rng_black, # Use the RNG returned by the jitted function
            rng=black_rng_current,  # Store the *new* state of the RNG key
            update_step=epoch + 1,
        )

        # --- White Agent Update ---
        # Create a batch copy and insert the white-specific valid mask
        white_update_batch = prepared_batch_flat.copy()
        white_update_batch["valid_mask"] = white_valid_mask_flat

        # Split RNG *before* the call
        update_rng_white, white_rng_current = jax.random.split(
            white_rng_current
        )  # Use white's RNG

        (
            update_rng_white,  # Consumed RNG - This is redundant
            white_params_updated,
            white_opt_state_updated,
            white_update_metrics,
        ) = jit_update_step_white(  # Call the pre-jitted white version
            rng=update_rng_white,
            params=white_params_current,
            optimizer=white_tx,
            opt_state=white_train_state.opt_state,
            full_batch=white_update_batch,  # Pass the masked batch
            config=ppo_config,
        )
        # Update white agent state
        white_train_state = white_train_state.replace(
            params=white_params_updated,
            opt_state=white_opt_state_updated,
            # rng=update_rng_white, # Use the RNG returned by the jitted function
            rng=white_rng_current,  # Store the *new* state of the RNG key
            update_step=epoch + 1,
        )

        # --- Construct and Log Training Data --- 
        # Calculate iteration-specific metrics
        iter_end_time = time.time()
        sps = steps_this_iter / (iter_end_time - iter_start_time) if (iter_end_time - iter_start_time) > 0 else 0
        current_epoch_for_log = epoch + 1 # Use current epoch for logging
        
        # Convert JAX arrays to Python scalars for logging
        log_data = {
            "train/epoch": current_epoch_for_log,
            "train/total_env_steps": total_env_steps,
            "train/sps": sps,
            "train/duration_s": iter_end_time - iter_start_time,
            "train/steps_this_epoch": steps_this_iter,
            "rollout/avg_episode_length": float(jnp.mean(
                full_trajectory["termination_step_indices"].astype(jnp.float32)
            ).item()),
        }
        # Add black agent metrics
        for k, v in black_update_metrics.items():
            log_data[f"ppo_black/{k}"] = float(v.item())
        # Add white agent metrics
        for k, v in white_update_metrics.items():
            log_data[f"ppo_white/{k}"] = float(v.item())

        if is_main_process:
            wandb.log(log_data, step=total_env_steps) # Log training data every epoch

        # === Checkpointing Phase ===
        save_epoch = epoch + 1  # Use epoch number for checkpoint step counter
        if (
            save_epoch % cfg.save_frequency == 0
            and save_epoch
            > start_epoch  # Avoid saving initial state immediately if resuming
        ):
            logger.info(
                f"Saving checkpoint at epoch {save_epoch} (total env steps ~{total_env_steps})..."
            )
            # Create a combined dictionary containing both states
            
            black_rng_to_save = black_train_state.rng
            white_rng_to_save = white_train_state.rng

            if jax.process_count() > 1:
                logger.info("Multi-host environment detected. Converting RNGs to global arrays for checkpointing.")
                black_rng_to_save = fully_replicated_host_local_array_to_global_array(black_rng_to_save)
                white_rng_to_save = fully_replicated_host_local_array_to_global_array(white_rng_to_save)

            save_data = {
                "black": {
                    "params": black_train_state.params,
                    "opt_state": black_train_state.opt_state,
                    "rng": black_rng_to_save, # Use potentially converted RNG
                    "update_step": black_train_state.update_step, 
                    "total_env_steps": total_env_steps, 
                },
                "white": {
                    "params": white_train_state.params,
                    "opt_state": white_train_state.opt_state,
                    "rng": white_rng_to_save, # Use potentially converted RNG
                    "update_step": white_train_state.update_step, 
                    "total_env_steps": total_env_steps, 
                },
            }
            # Save the combined dictionary using Composite save args
            save_args = ocp.args.Composite(
                black=ocp.args.StandardSave(save_data["black"]),
                white=ocp.args.StandardSave(save_data["white"]),
            )
            checkpointer.save(save_epoch, args=save_args, force=True)
            logger.info("Checkpoint saved.")

        # === Evaluation Phase ===
        eval_epoch = epoch + 1
        if eval_epoch % cfg.eval_frequency == 0:
            logger.info(f"Running evaluation at epoch {eval_epoch}...")
            rng, eval_rng_key = jax.random.split(rng)

            logger.info(f"Running {cfg.eval_games} evaluation games...")
            eval_metrics = jit_eval(
                eval_env=eval_env, # Pass the dedicated evaluation environment
                black_actor_critic=black_model,
                black_params=black_train_state.params,
                white_actor_critic=white_model,
                white_params=white_train_state.params,
                rng=eval_rng_key,
            )
            bw_s = eval_metrics["eval/black_wins"].item()
            ww_s = eval_metrics["eval/white_wins"].item()
            dr_s = eval_metrics["eval/draws"].item()
            # Ensure tg_s is calculated from Python numbers to be a Python number
            tg_s = bw_s + ww_s + dr_s 
            
            bwp_s = (bw_s / tg_s * 100) if tg_s > 0 else 0.0
            wwp_s = (ww_s / tg_s * 100) if tg_s > 0 else 0.0
            drp_s = (dr_s / tg_s * 100) if tg_s > 0 else 0.0
            
            logger.info(
                f"Evaluation epoch {eval_epoch} results (over {tg_s} games): "
                f"Black Wins: {bw_s} ({bwp_s:.2f}%), "
                f"White Wins: {ww_s} ({wwp_s:.2f}%), "
                f"Draws: {dr_s} ({drp_s:.2f}%)"
            )
            if is_main_process:
                wandb.log(eval_metrics, step=total_env_steps) # Evaluation metrics logged here



    # --- Final Cleanup ---
    logger.info("Waiting for checkpointer...")
    checkpointer.wait_until_finished()
    checkpointer.close()
    logger.info("Closing WandB run...")
    total_time = time.time() - start_time
    logger.info(f"Training finished in {total_time:.2f} seconds.")
    if is_main_process and run is not None:
        wandb.finish()


train()
